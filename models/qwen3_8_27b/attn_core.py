# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B causal GQA attention core: `o = softmax(q k^T + causal mask) v`, per query head.

`q` arrives already scaled by `head_dim ** -0.5`. Query head `h` reads KV head
`h // (HQ // HKV)`. Online softmax over KV tiles, one query tile at a time: the
tiles below the diagonal unmasked, then the diagonal tile, the only one masked.
`p @ v` is two products over the two halves of the head dimension, and the
output accumulates as two halves.
"""
import pypto.language as pl

from config import PREFILL_SEQ, QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
HQ = QWEN3_8_27B.num_attention_heads    # query heads
HKV = QWEN3_8_27B.num_key_value_heads   # KV heads
D = QWEN3_8_27B.head_dim                # head dimension
T = PREFILL_SEQ                             # tokens (single sequence, B = 1)
# The simulators run the whole core as one task, and their scheduler declares a
# stall after 20 s without progress: at T = 8192 that kills the run long before
# the simulated kernel finishes. 1024 runs the same code -- both diagonal
# offsets and the tiles below them, every head -- in 15 s.
SIM_T = 1024

# tiling
# The PV product is ROW_TILE * D * 4 bytes of the 128 KiB accumulator buffer,
# so D = 256 caps the query tile at 64 rows.
ROW_TILE = 64           # query rows per tile
KV_TILE = 128           # keys per online-softmax step
# Cross-core ring depth. Each KV step makes four cube<->vector hand-offs; a
# second slot overlaps one step's with the next. A third needs 189056 B of the
# 188416 B vector buffer (pypto#2903).
SLOT_NUM = 2
MASK_NEG = -1.0e30      # additive mask; exp of it underflows to exactly zero


def causal_bias(row_tile: int = ROW_TILE, kv_tile: int = KV_TILE):
    """The mask table: `[row_tile, 2 * kv_tile - row_tile]` FP32, 0 or MASK_NEG.

    Column `c'` of row `i` is open when `c' <= i + kv_tile - row_tile`. A
    diagonal tile whose first key sits `diag` positions before its first query
    reads the `kv_tile` columns from `kv_tile - row_tile - diag`, which opens
    exactly the keys `c <= i + diag`.
    """
    import torch

    rows = torch.arange(row_tile)[:, None]
    cols = torch.arange(2 * kv_tile - row_tile)[None, :]
    open_ = cols <= rows + (kv_tile - row_tile)
    return torch.where(open_, 0.0, MASK_NEG).to(torch.float32)


Q_WIDTH = HQ * D
KV_WIDTH = HKV * D
GROUP = HQ // HKV                       # query heads sharing one KV head
RATIO = KV_TILE // ROW_TILE
BIAS_WIDTH = 2 * KV_TILE - ROW_TILE
HALF = D // 2


def _qwen3_attn_core(
    q: pl.Tensor[[T_DYN, Q_WIDTH], pl.BF16],
    k: pl.Tensor[[T_DYN, KV_WIDTH], pl.BF16],
    v: pl.Tensor[[T_DYN, KV_WIDTH], pl.BF16],
    mask_bias: pl.Tensor[[ROW_TILE, BIAS_WIDTH], pl.FP32],
    out: pl.Out[pl.Tensor[[T_DYN, Q_WIDTH], pl.BF16]],
):
    """The token count must be a multiple of KV_TILE; a step walks one whole tile."""
    q.bind_dynamic(0, T_DYN)
    k.bind_dynamic(0, T_DYN)
    v.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(q, 0)
    # One block per query head: every head costs the same, so the static
    # split is balanced.
    for hh in pl.spmd(HQ, name_hint="attn_core",
                      optimizations=[pl.cross_core_slot(slot_num=SLOT_NUM),
                                     pl.split(pl.SplitMode.UP_DOWN)]):
        qd0 = hh * D
        kd0 = (hh // GROUP) * D
        for qi in pl.range(t_dim // ROW_TILE):
            r0 = qi * ROW_TILE
            q_tile = q[r0 : r0 + ROW_TILE, qd0 : qd0 + D]

            # The running max and sum are [ROW_TILE, 1] columns, seeded from a
            # reduction so they take the layout the reductions produce.
            zero_scores = pl.full([ROW_TILE, KV_TILE], dtype=pl.FP32, value=0.0)
            zero_col = pl.row_max(zero_scores)
            m_init = pl.add(zero_col, MASK_NEG)
            lo_init = pl.full([ROW_TILE, HALF], dtype=pl.FP32, value=0.0)
            hi_init = pl.full([ROW_TILE, HALF], dtype=pl.FP32, value=0.0)

            # the tiles below the diagonal, unmasked
            n_below = qi // RATIO
            for j, (m_run, l_run, lo_run, hi_run) in pl.range(
                    n_below, init_values=(m_init, zero_col, lo_init, hi_init)):
                c0 = j * KV_TILE
                k_tile = k[c0 : c0 + KV_TILE, kd0 : kd0 + D]
                scores = pl.matmul(q_tile, k_tile, b_trans=True, out_dtype=pl.FP32)
                m_cur = pl.row_max(scores)
                m_new = pl.maximum(m_run, m_cur)
                alpha = pl.exp(pl.sub(m_run, m_new))
                probs = pl.exp(pl.row_expand_sub(scores, m_new))
                probs_bf = pl.cast(probs, target_type=pl.BF16, mode="rint")
                l_cur = pl.row_sum(pl.cast(probs_bf, target_type=pl.FP32))
                l_new = pl.add(pl.mul(l_run, alpha), l_cur)
                # p @ v in two column halves: see the module docstring
                v_lo = v[c0 : c0 + KV_TILE, kd0 : kd0 + HALF]
                v_hi = v[c0 : c0 + KV_TILE, kd0 + HALF : kd0 + D]
                pv_lo = pl.matmul(probs_bf, v_lo, out_dtype=pl.FP32)
                pv_hi = pl.matmul(probs_bf, v_hi, out_dtype=pl.FP32)
                lo_new = pl.add(pl.row_expand_mul(lo_run, alpha), pv_lo)
                hi_new = pl.add(pl.row_expand_mul(hi_run, alpha), pv_hi)
                m_below, l_below, lo_below, hi_below = pl.yield_(m_new, l_new, lo_new, hi_new)

            # the diagonal tile, masked
            c_diag = n_below * KV_TILE
            bias_c0 = KV_TILE - ROW_TILE - (r0 - c_diag)
            k_diag = k[c_diag : c_diag + KV_TILE, kd0 : kd0 + D]
            bias = mask_bias[:, bias_c0 : bias_c0 + KV_TILE]
            diag_raw = pl.matmul(q_tile, k_diag, b_trans=True, out_dtype=pl.FP32)
            diag_scores = pl.add(diag_raw, bias)
            m_diag = pl.row_max(diag_scores)
            m_fin = pl.maximum(m_below, m_diag)
            alpha_fin = pl.exp(pl.sub(m_below, m_fin))
            diag_probs = pl.exp(pl.row_expand_sub(diag_scores, m_fin))
            diag_probs_bf = pl.cast(diag_probs, target_type=pl.BF16, mode="rint")
            l_diag = pl.row_sum(pl.cast(diag_probs_bf, target_type=pl.FP32))
            l_fin = pl.add(pl.mul(l_below, alpha_fin), l_diag)
            v_diag_lo = v[c_diag : c_diag + KV_TILE, kd0 : kd0 + HALF]
            v_diag_hi = v[c_diag : c_diag + KV_TILE, kd0 + HALF : kd0 + D]
            pv_diag_lo = pl.matmul(diag_probs_bf, v_diag_lo, out_dtype=pl.FP32)
            pv_diag_hi = pl.matmul(diag_probs_bf, v_diag_hi, out_dtype=pl.FP32)
            lo_fin = pl.add(pl.row_expand_mul(lo_below, alpha_fin), pv_diag_lo)
            hi_fin = pl.add(pl.row_expand_mul(hi_below, alpha_fin), pv_diag_hi)

            out_lo = pl.row_expand_div(lo_fin, l_fin)
            out_hi = pl.row_expand_div(hi_fin, l_fin)
            out[r0 : r0 + ROW_TILE, qd0 : qd0 + HALF] = pl.cast(out_lo, target_type=pl.BF16, mode="rint")
            out[r0 : r0 + ROW_TILE, qd0 + HALF : qd0 + D] = pl.cast(out_hi, target_type=pl.BF16, mode="rint")
    return out


qwen3_attn_core = pl.jit.inline(_qwen3_attn_core)
qwen3_attn_core_test = pl.jit(_qwen3_attn_core)



# host only: the spread of the scores the input draw produces. q is drawn at
# SCORE_STD / sqrt(D) against unit k, which is what a pre-scaled q looks like.
SCORE_STD = 4.0


def build_tensor_specs(t: int = T, hq: int = HQ, hkv: int = HKV, d: int = D,
                       row_tile: int = ROW_TILE, kv_tile: int = KV_TILE):
    import torch
    from golden import TensorSpec

    def draw(seed, cols, scale):
        def make():
            gen = torch.Generator().manual_seed(seed)
            return (torch.randn(t, cols, generator=gen) * scale).to(torch.bfloat16)
        return make

    return [
        TensorSpec("q", [t, hq * d], torch.bfloat16, init_value=draw(42, hq * d, SCORE_STD * d ** -0.5)),
        TensorSpec("k", [t, hkv * d], torch.bfloat16, init_value=draw(43, hkv * d, 1.0)),
        TensorSpec("v", [t, hkv * d], torch.bfloat16, init_value=draw(44, hkv * d, 1.0)),
        TensorSpec("mask_bias", [row_tile, 2 * kv_tile - row_tile], torch.float32,
                   init_value=lambda: causal_bias(row_tile, kv_tile)),
        TensorSpec("out", [t, hq * d], torch.bfloat16),
    ]


# acceptance, against the float64 attention of the same bf16 operands. The
# kernel rounds p to BF16 for the second product and the output to BF16; the
# golden rounds only the output.
MAX_REL = 1e-2          # whole tensor, relative Frobenius
MAX_ROW_REL = 2e-2      # worst token row


def compare_out(actual, expected, **_kwargs):
    """Whole-tensor and worst-row relative distance to the float64 golden."""
    dev, ref = actual.double(), expected.double()
    rel = float((dev - ref).norm() / ref.norm())
    row = float(((dev - ref).norm(dim=-1) / ref.norm(dim=-1)).max())
    peak = float((dev - ref).abs().max())
    detail = (f"[stats] out vs golden: rel frob {rel:.3e} (gate {MAX_REL}), worst row {row:.3e} "
              f"(gate {MAX_ROW_REL}), max abs {peak:.3e}")
    print(detail, flush=True)
    ok = rel <= MAX_REL and row <= MAX_ROW_REL
    return ok, "" if ok else detail


def golden_qwen3_attn_core(tensors):
    import torch

    import reference

    q, k, v = tensors["q"], tensors["k"], tensors["v"]
    hkv = k.shape[1] // D
    hq = q.shape[1] // D
    ref = reference.causal_gqa_attention(q, k, v, hq, hkv, D)
    tensors["out"].copy_(ref.to(torch.bfloat16))


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help=f"tokens; default {T} on hardware, {SIM_T} on the simulators")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    t = args.seq_len or (SIM_T if args.platform.endswith("sim") else T)
    if t % KV_TILE:
        parser.error(f"--seq-len must be a multiple of KV_TILE={KV_TILE}")
    result = run(
        fn=qwen3_attn_core_test,
        specs=build_tensor_specs(t=t),
        golden_fn=golden_qwen3_attn_core,
        compare_fn={"out": compare_out},
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=1e-2,
        atol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
