# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_h: the inter-chunk state recurrence. Per chunk, per head,
with S the state entering the chunk,

    snapshot = S
    V_new    = U - W @ S
    S        = exp(g_last) * S + K^T (V_new * exp(g_last - g))

Chunks carry state, so work is parallel over heads and sequential within one.

`state` holds the snapshot ENTERING each chunk, which is what `chunk_o` reads. The state
LEAVING the last chunk is what an inference cache carries forward, and
`_gdn_chunk_h_final` writes that as well -- transposed to `[V, K]` and widened to FP32 --
into a row of the pool `decode_delta_rule` reads. The decode state is FP32 because fp16
does not survive a long rollout, and its value axis leads because that puts the
recurrence's `sum_k` on the contiguous axis.
"""
import pypto.language as pl
import torch

from chunk_cumsum import cumsum_ref
from config import QWEN3_8_27B
from reference import REF_DTYPE, flat_state, frob_ok, make_inputs, to_hT
from scaled_dot_kkt import kkt_ref
from solve_tril import solve_tril_ref
from wy_fast import CHUNK_TILE, wy_fast_ref

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens
STATE_DYN = pl.dynamic("GDN_STATE_DYN")     # (T // CHUNK_TILE) * H * D, the snapshot rows
FSTATE_DYN = pl.dynamic("GDN_FSTATE_DYN")   # rows of the decode state pool

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
GRP = H // HG                               # value heads sharing one key head
KEY_WIDTH = HG * D
VAL_WIDTH = H * D

# tiling
HANDOFF_TILE = 64       # value rows per transpose step of the decode hand-off

# validation
# One chunk, the cross-chunk recurrence, then eight chunks. The token count must be a
# multiple of CHUNK_TILE.
DEFAULT_CASES = (128, 256, 1024)
FINAL_CASE = 256        # the hand-off body: two chunks, so the state it leaves is not the first
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len


def _gdn_chunk_h(
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    w: pl.Tensor[[T_DYN, H, D], pl.FP16],
    u: pl.Tensor[[T_DYN, H, D], pl.FP16],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    state: pl.Out[pl.Tensor[[STATE_DYN, D], pl.FP16]],
    v_new: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
):
    """The token count must be a multiple of CHUNK_TILE, and `state` hold (T // CHUNK_TILE) * H * D rows."""
    k.bind_dynamic(0, T_DYN)
    w.bind_dynamic(0, T_DYN)
    u.bind_dynamic(0, T_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    state.bind_dynamic(0, STATE_DYN)
    v_new.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(k, 0)
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    w_flat = pl.reshape(w, [t_dim, VAL_WIDTH])
    u_flat = pl.reshape(u, [t_dim, VAL_WIDTH])
    v_flat = pl.reshape(v_new, [t_dim, VAL_WIDTH])
    for hh in pl.spmd(H, name_hint="chunk_h",
                      optimizations=[pl.cross_core_slot(slot_num=1),
                                     pl.split(pl.SplitMode.UP_DOWN)]):
        s0 = pl.full([D, D], dtype=pl.FP32, value=0.0)
        for c, (s_cur,) in pl.range(t_dim // CHUNK_TILE, init_values=(s0,)):
            t0 = c * CHUNK_TILE
            row = (c * H + hh) * D
            d0 = hh * D
            # GQA: K comes from key head hh // GRP; W, U and the state are per
            # value head.
            dg0 = (hh // GRP) * D

            s16 = pl.cast(s_cur, target_type=pl.FP16, mode="rint")
            state[row : row + D, 0:D] = s16                 # state ENTERING this chunk

            # coeff[i] = exp(g_last - g_i), decay = exp(g_last); both arguments <= 0
            g_last = pl.read(g_sum, [hh, t0 + CHUNK_TILE - 1])
            g_row = g_sum[hh : hh + 1, t0 : t0 + CHUNK_TILE]
            zero_row = pl.full([1, CHUNK_TILE], dtype=pl.FP32, value=0.0)
            neg_g = pl.sub(zero_row, g_row)
            # exp before the [1,C]->[C,1] reshape, not after (pypto#2947)
            coeff = pl.reshape(pl.exp(pl.add(neg_g, g_last)), [CHUNK_TILE, 1])
            zero_d = pl.full([1, D], dtype=pl.FP32, value=0.0)
            decay = pl.reshape(pl.exp(pl.add(zero_d, g_last)), [D, 1])

            wc = w_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            ws = pl.matmul(wc, s16, out_dtype=pl.FP32)
            uc = u_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            vc = pl.sub(pl.cast(uc, target_type=pl.FP32), ws)   # V_new = U - W @ S
            vc16 = pl.cast(vc, target_type=pl.FP16, mode="rint")

            # S = exp(g_last) * S + K^T @ (coeff * V_new). The decay goes on V, not
            # on K: K^T (V c) == (K c)^T V. Under pl.split a matmul whose transposed
            # operand came from the vector unit cannot be lowered (pypto#2902).
            kc = k_flat[t0 : t0 + CHUNK_TILE, dg0 : dg0 + D]
            vs = pl.row_expand_mul(vc, coeff)
            vs16 = pl.cast(vs, target_type=pl.FP16, mode="rint")
            kv = pl.matmul(kc, vs16, a_trans=True, out_dtype=pl.FP32)
            s_next = pl.add(pl.row_expand_mul(s_cur, decay), kv)

            v_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D] = vc16
            s_end, = pl.yield_(s_next)
    return state, v_new


gdn_chunk_h = pl.jit.inline(_gdn_chunk_h)
gdn_chunk_h_test = pl.jit(_gdn_chunk_h)


def _gdn_chunk_h_final(
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    w: pl.Tensor[[T_DYN, H, D], pl.FP16],
    u: pl.Tensor[[T_DYN, H, D], pl.FP16],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    state: pl.Out[pl.Tensor[[STATE_DYN, D], pl.FP16]],
    v_new: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
    state_row: pl.Tensor[[1], pl.INT32],
    final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, D, D], pl.FP32]],
):
    """As `_gdn_chunk_h`, plus the state leaving the last chunk into `final_state`.

    `state_row` names the pool row this sequence owns, the same indirection
    `decode_delta_rule` uses, so the hand-off is a write into the pool rather than a
    copy of 3 MiB per sequence.
    """
    k.bind_dynamic(0, T_DYN)
    w.bind_dynamic(0, T_DYN)
    u.bind_dynamic(0, T_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    state.bind_dynamic(0, STATE_DYN)
    v_new.bind_dynamic(0, T_DYN)
    final_state.bind_dynamic(0, FSTATE_DYN)
    t_dim = pl.tensor.dim(k, 0)
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    w_flat = pl.reshape(w, [t_dim, VAL_WIDTH])
    u_flat = pl.reshape(u, [t_dim, VAL_WIDTH])
    v_flat = pl.reshape(v_new, [t_dim, VAL_WIDTH])
    # The pool is addressed through a flat view of the same buffer, as
    # `decode_delta_rule` addresses it: a `[1, H, D, D]` subview would need a
    # single-element row, which ptoas rejects on its 32-byte row alignment.
    fs_2d = pl.reshape(final_state, [pl.tensor.dim(final_state, 0) * H * D, D])
    fs_row = pl.cast(pl.read(state_row, [0]), pl.INDEX)
    carry = pl.create_tensor([H * D, D], dtype=pl.FP32)
    for hh in pl.spmd(H, name_hint="chunk_h",
                      optimizations=[pl.cross_core_slot(slot_num=1),
                                     pl.split(pl.SplitMode.UP_DOWN)]):
        s0 = pl.full([D, D], dtype=pl.FP32, value=0.0)
        for c, (s_cur,) in pl.range(t_dim // CHUNK_TILE, init_values=(s0,)):
            t0 = c * CHUNK_TILE
            row = (c * H + hh) * D
            d0 = hh * D
            # GQA: K comes from key head hh // GRP; W, U and the state are per
            # value head.
            dg0 = (hh // GRP) * D

            s16 = pl.cast(s_cur, target_type=pl.FP16, mode="rint")
            state[row : row + D, 0:D] = s16                 # state ENTERING this chunk

            # coeff[i] = exp(g_last - g_i), decay = exp(g_last); both arguments <= 0
            g_last = pl.read(g_sum, [hh, t0 + CHUNK_TILE - 1])
            g_row = g_sum[hh : hh + 1, t0 : t0 + CHUNK_TILE]
            zero_row = pl.full([1, CHUNK_TILE], dtype=pl.FP32, value=0.0)
            neg_g = pl.sub(zero_row, g_row)
            # exp before the [1,C]->[C,1] reshape, not after (pypto#2947)
            coeff = pl.reshape(pl.exp(pl.add(neg_g, g_last)), [CHUNK_TILE, 1])
            zero_d = pl.full([1, D], dtype=pl.FP32, value=0.0)
            decay = pl.reshape(pl.exp(pl.add(zero_d, g_last)), [D, 1])

            wc = w_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            ws = pl.matmul(wc, s16, out_dtype=pl.FP32)
            uc = u_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            vc = pl.sub(pl.cast(uc, target_type=pl.FP32), ws)   # V_new = U - W @ S
            vc16 = pl.cast(vc, target_type=pl.FP16, mode="rint")

            # S = exp(g_last) * S + K^T @ (coeff * V_new). The decay goes on V, not
            # on K: K^T (V c) == (K c)^T V. Under pl.split a matmul whose transposed
            # operand came from the vector unit cannot be lowered (pypto#2902).
            kc = k_flat[t0 : t0 + CHUNK_TILE, dg0 : dg0 + D]
            vs = pl.row_expand_mul(vc, coeff)
            vs16 = pl.cast(vs, target_type=pl.FP16, mode="rint")
            kv = pl.matmul(kc, vs16, a_trans=True, out_dtype=pl.FP32)
            s_next = pl.add(pl.row_expand_mul(s_cur, decay), kv)

            v_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D] = vc16
            s_end, = pl.yield_(s_next)

        # The state the loop ends with, still [K, V] and still FP32, parked for the
        # transpose pass below.
        carry[hh * D : hh * D + D, 0:D] = s_end

    # [K, V] -> [V, K] in a second pass, because `pl.split(UP_DOWN)` halves dim 0 and a
    # transpose swaps it -- the compiler rejects the pair outright. The split is not
    # optional above (the body needs 197632 B of a 188416 B buffer without it), so the
    # transpose moves out instead. It costs one more pass over 3 MiB of FP32 per
    # sequence, once, against a hand-off that happens once.
    for hh2 in pl.spmd(H, name_hint="chunk_h_handoff"):
        crow = hh2 * D
        frow = (fs_row * H + hh2) * D
        # In HANDOFF_TILE bands of the value axis: a whole head transposed at once
        # needs three [D, D] FP32 tiles live, 196608 B of a 188416 B buffer, and a
        # band of 64 value rows needs three of half that. `decode_delta_rule` tiles
        # its own state the same way for the same reason.
        for v0 in pl.range(0, D, HANDOFF_TILE):
            band = pl.slice(carry, [D, HANDOFF_TILE], [crow, v0])
            fs_2d[frow + v0 : frow + v0 + HANDOFF_TILE, 0:D] = pl.transpose(band, 0, 1)
    return state, v_new, final_state


gdn_chunk_h_final = pl.jit.inline(_gdn_chunk_h_final)
gdn_chunk_h_final_test = pl.jit(_gdn_chunk_h_final)


def chunk_h_ref(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_sum: torch.Tensor,
            chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """S6: the inter-chunk state recurrence.

    Returns the state snapshot ENTERING each chunk, [NCHUNK, H, D, D]; the
    residual-corrected values V_new, [T, H, D]; and the state LEAVING the last
    chunk, [H, D, D], which is what an inference cache carries forward.
    """
    t, hg, d = k.shape
    h = w.shape[1]
    grp = h // hg
    nc = t // chunk
    kf, wf, uf, gf = (x.to(REF_DTYPE) for x in (k, w, u, g_sum))
    state = torch.zeros(nc, h, d, d, dtype=REF_DTYPE)
    v_new = torch.zeros(t, h, d, dtype=REF_DTYPE)
    final_state = torch.zeros(h, d, d, dtype=REF_DTYPE)
    for hh in range(h):
        s = torch.zeros(d, d, dtype=REF_DTYPE)
        for ci in range(nc):
            t0 = ci * chunk
            gc = gf[t0 : t0 + chunk, hh]
            g_last = gc[-1]
            state[ci, hh] = s
            vc = uf[t0 : t0 + chunk, hh, :] - wf[t0 : t0 + chunk, hh, :] @ s
            v_new[t0 : t0 + chunk, hh, :] = vc
            kv = (kf[t0 : t0 + chunk, hh // grp, :].T
                  @ (vc * torch.exp(g_last - gc)[:, None]))
            s = torch.exp(g_last) * s + kv
        final_state[hh] = s
    return state, v_new, final_state


_INPUTS: dict = {}


def _inputs(t: int) -> dict:
    """The chain this stage consumes, memoised so the several specs pay for it once."""
    key = t
    if key not in _INPUTS:
        st = make_inputs(t, H, D, HG)
        st["g_sum"] = cumsum_ref(st["g"], CHUNK_TILE)
        a16 = kkt_ref(st["k"], st["beta"], st["g_sum"], CHUNK_TILE).to(torch.float16)
        a_inv16 = solve_tril_ref(a16, CHUNK_TILE).to(torch.float16)
        w, u = wy_fast_ref(st["k"], st["v"], st["beta"], a_inv16, st["g_sum"], CHUNK_TILE)
        st["w16"] = w.to(torch.float16)
        st["u16"] = u.to(torch.float16)
        _INPUTS[key] = st
    return _INPUTS[key]


def build_tensor_specs(t: int = BENCH_SEQ):
    from golden import TensorSpec

    def draw(key, transform=None):
        def make():
            value = _inputs(t)[key]
            return transform(value) if transform is not None else value

        return make

    nc = t // CHUNK_TILE

    return [
        TensorSpec("k", [t, HG, D], torch.float16,
                   init_value=draw("k")),
        TensorSpec("w", [t, H, D], torch.float16,
                   init_value=draw("w16")),
        TensorSpec("u", [t, H, D], torch.float16,
                   init_value=draw("u16")),
        TensorSpec("g_sum", [H, t], torch.float32,
                   init_value=draw("g_sum", to_hT)),
        TensorSpec("state", [nc * H * D, D], torch.float16),
        TensorSpec("v_new", [t, H, D], torch.float16),
    ]


def golden_gdn_chunk_h(tensors):
    t, _, d = tensors["k"].shape          # k has Hg heads under GQA
    h = tensors["w"].shape[1]             # W, U and the state are per value head
    chunk = t // (tensors["state"].shape[0] // (h * d))
    state, v_new, _ = chunk_h_ref(tensors["k"], tensors["w"], tensors["u"], tensors["g_sum"].t(), chunk)
    tensors["state"].copy_(flat_state(state))
    tensors["v_new"].copy_(v_new)


def build_final_tensor_specs(t: int = BENCH_SEQ, pool_rows: int = 4, state_row: int = 2):
    """`build_tensor_specs` plus the decode pool and the row this sequence owns.

    `state_row` defaults away from 0 so a kernel that ignored `state_row` and wrote row 0
    would leave the gated row untouched and pass.
    """
    from golden import TensorSpec

    def pool():
        # Non-zero, so a row the kernel fails to write shows up as the fill rather than
        # as a plausible zero state.
        return torch.full((pool_rows, H, D, D), -1.0)

    return build_tensor_specs(t=t) + [
        TensorSpec("state_row", [1], torch.int32,
                   init_value=lambda: torch.tensor([state_row], dtype=torch.int32)),
        TensorSpec("final_state", [pool_rows, H, D, D], torch.float32, init_value=pool),
    ]


def golden_gdn_chunk_h_final(tensors):
    """The snapshots and V_new as before, plus the hand-off transposed and widened."""
    t, _, d = tensors["k"].shape
    h = tensors["w"].shape[1]
    chunk = t // (tensors["state"].shape[0] // (h * d))
    state, v_new, final = chunk_h_ref(tensors["k"], tensors["w"], tensors["u"],
                                      tensors["g_sum"].t(), chunk)
    tensors["state"].copy_(flat_state(state))
    tensors["v_new"].copy_(v_new)
    # `chunk_h_ref` returns the hand-off as [H, K, V]; the decode pool holds [V, K].
    row = int(tensors["state_row"][0])
    tensors["final_state"][row] = final.transpose(-2, -1).to(tensors["final_state"].dtype)


def compare_final_state(state_row: int):
    """Score the row the kernel wrote, and prove it left the others alone.

    A whole-pool norm would be diluted: with four rows and one written, three quarters of
    the tensor is trivially equal. The untouched rows are also the control -- they carry
    -1.0, which no state takes, so a kernel that ignored `state_row` shows up as a row
    that should have been left alone and was not.
    """
    scored = frob_ok()

    def compare(actual, expected, **kwargs):
        ok, detail = scored(actual[state_row], expected[state_row], **kwargs)
        others = [r for r in range(actual.shape[0]) if r != state_row]
        if others and not bool((actual[others] == -1.0).all()):
            return False, f"{detail}\n    rows {others} were written and should not have been"
        return ok, detail

    compare.__name__ = "compare_final_state"
    return compare


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--final", action="store_true", default=False,
                        help="only the body that also writes the hand-off state")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % CHUNK_TILE:
        parser.error(f"--seq-len must be a multiple of CHUNK_TILE={CHUNK_TILE}")

    # Both bodies by default: the hand-off is a second program, not a branch inside the
    # first, so the plain entry cannot cover it.
    plan = [] if args.final else [
        (f"chunk_h T={t}", t, gdn_chunk_h_test, build_tensor_specs, golden_gdn_chunk_h,
         {"state": frob_ok(), "v_new": frob_ok()}) for t in cases]
    plan += [
        (f"chunk_h final T={t}", t, gdn_chunk_h_final_test, build_final_tensor_specs,
         golden_gdn_chunk_h_final,
         {"state": frob_ok(), "v_new": frob_ok(), "final_state": compare_final_state(2)})
        for t in (cases if args.seq_len or args.final else [FINAL_CASE])]
    if len(plan) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one program and one shape; "
                     "name them with --seq-len and --final")

    for label, t, fn, specs_fn, golden_fn, compare_fn in plan:
        print(f"[case] {label}", flush=True)
        result = run(
            fn=fn,
            specs=specs_fn(t=t),
            golden_fn=golden_fn,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            compare_fn=compare_fn,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case in this process reports a failure it did not cause.
            raise SystemExit(f"FAILED at {label}")
