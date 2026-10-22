# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B gated attention, the mixer of the 16 full-attention layers, in W8A8:

    out = o_proj( attention(rope(norm(q)), rope(norm(k)), v) * sigmoid(gate) )

Eight operators as one program: the per-token quantiser, `q_proj` (query and
gate per head), `k_proj` and `v_proj` against the same quantised activation,
the q/k norm and RoPE, the causal GQA attention core, the gate fused with the
quantiser `o_proj` wants, then `o_proj`. The projections are `a8w8_linear` at
four shapes; the other three operators are this stage's own.
"""
import pypto.language as pl

import a8w8_linear
import attn_core
from attn_core_pipe import qwen3_attn_core_pipe
from attn_gate_quant import qwen3_attn_gate_quant
from qk_rmsnorm_rope import qwen3_qk_rmsnorm_rope
from quant_x import gdn_quant_x
import attn_core_pipe
from config import QWEN3_8_27B

# model config
HIDDEN = QWEN3_8_27B.hidden_size
HQ = QWEN3_8_27B.num_attention_heads
HKV = QWEN3_8_27B.num_key_value_heads
D = QWEN3_8_27B.head_dim
ROT = QWEN3_8_27B.rotary_dim
EPS = QWEN3_8_27B.rms_norm_eps
QG_WIDTH = 2 * HQ * D           # q_proj: query and gate per head
KV_WIDTH = HKV * D
O_WIDTH = HQ * D
# validation
DEFAULT_CASES = (256, 512)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# ci: no-sim

# The per-ring heap holds task outputs and intermediates, and its compile
# default is 256 MiB. This program's intermediates are 0.54 GiB at T = 8192,
# `qg` at [T, 12288] BF16 the largest.
#
# Measured at T = 8192: 1 GiB runs and 512 MiB does not. Kept one power of two
# above that floor. The ratio to the 0.54 GiB above is not the rule -- the MLP
# runs on 512 MiB against 720 MiB of intermediates -- because the heap holds
# what is in flight, and this program keeps more of it live at once.
# Bytes per ring, one entry per scope depth, 0 leaving a ring at its default. The stage
# scopes allocate on ring 1; a scalar here would reserve the same bytes on all four.
ATTN_RING_HEAP = [0, 2 * 1024 * 1024 * 1024, 0, 0]

# tiling; each projection shape gets its own patch
M_TILE, N_TILE, K_TILE = 128, 128, 512
MAX_GROUP = 8           # largest output-tile patch a block walks, per axis
RING_SLOTS = 2          # GM transport depth; the on-chip reserve stays at 1


def _patch(blocks: int, cap: int = MAX_GROUP) -> int:
    """The largest even patch side dividing `blocks`, capped at `cap`."""
    for side in (8, 4, 2):
        if side <= cap and blocks % side == 0:
            return side
    return 1


def build_kernel(t: int = BENCH_SEQ, inline: bool = False):
    """The mixer at one shape, composing eight operators as inlined callees.

    Static shape only: a packed varlen caller needs a tail, which every
    operator in this directory still owes.
    """
    m_group = _patch(t // M_TILE)
    # k_proj and v_proj are 8 output tiles wide: an 8 x 8 patch would leave
    # T / 1024 blocks and most of the cube cores idle
    kv_m_group = _patch(t // M_TILE, cap=4)

    op_q = a8w8_linear.build_kernel(m=t, k=HIDDEN, n=QG_WIDTH, name="q_proj", inline=True,
                                    m_tile=M_TILE, n_tile=N_TILE, k_tile=K_TILE,
                                    m_group=m_group, n_group=_patch(QG_WIDTH // N_TILE),
                                    ring_slots=RING_SLOTS)
    op_k = a8w8_linear.build_kernel(m=t, k=HIDDEN, n=KV_WIDTH, name="k_proj", inline=True,
                                    m_tile=M_TILE, n_tile=N_TILE, k_tile=K_TILE,
                                    m_group=kv_m_group, n_group=2, ring_slots=RING_SLOTS)
    op_v = a8w8_linear.build_kernel(m=t, k=HIDDEN, n=KV_WIDTH, name="v_proj", inline=True,
                                    m_tile=M_TILE, n_tile=N_TILE, k_tile=K_TILE,
                                    m_group=kv_m_group, n_group=2, ring_slots=RING_SLOTS)
    core_kv_tile, core_row_tile = attn_core_pipe.KV_TILE, attn_core_pipe.ROW_TILE
    # o_proj is the same 6144 -> 5120 shape `out_proj.py` runs, and takes the
    # same 8 x 2 patch: `_patch` would give 8 x 8, which leaves 40 blocks for
    # 24 cores and measures 1186.8 us against 1056.5
    op_o = a8w8_linear.build_kernel(m=t, k=O_WIDTH, n=HIDDEN, name="o_proj", inline=True,
                                    m_tile=M_TILE, n_tile=N_TILE, k_tile=K_TILE,
                                    m_group=m_group, n_group=2, ring_slots=RING_SLOTS)
    bias_width = 2 * core_kv_tile - core_row_tile

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_gated_attention(
        x: pl.Tensor[[t, HIDDEN], pl.BF16],
        w_q: pl.Tensor[[QG_WIDTH, HIDDEN], pl.INT8],
        w_q_scale: pl.Tensor[[QG_WIDTH], pl.FP32],
        w_k: pl.Tensor[[KV_WIDTH, HIDDEN], pl.INT8],
        w_k_scale: pl.Tensor[[KV_WIDTH], pl.FP32],
        w_v: pl.Tensor[[KV_WIDTH, HIDDEN], pl.INT8],
        w_v_scale: pl.Tensor[[KV_WIDTH], pl.FP32],
        w_o: pl.Tensor[[HIDDEN, O_WIDTH], pl.INT8],
        w_o_scale: pl.Tensor[[HIDDEN], pl.FP32],
        q_norm_w: pl.Tensor[[D], pl.BF16],
        k_norm_w: pl.Tensor[[D], pl.BF16],
        rope_cos: pl.Tensor[[t, ROT], pl.FP32],
        rope_sin: pl.Tensor[[t, ROT], pl.FP32],
        mask_bias: pl.Tensor[[core_row_tile, bias_width], pl.FP32],
        out: pl.Out[pl.Tensor[[t, HIDDEN], pl.BF16]],
    ):
        x_q = pl.create_tensor([t, HIDDEN], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        gdn_quant_x(x, x_q, x_scale)

        # the three projections read the same quantised activation
        qg = pl.create_tensor([t, QG_WIDTH], dtype=pl.BF16)
        k = pl.create_tensor([t, KV_WIDTH], dtype=pl.BF16)
        v = pl.create_tensor([t, KV_WIDTH], dtype=pl.BF16)
        op_q(x_q, x_scale, w_q, w_q_scale, qg)
        op_k(x_q, x_scale, w_k, w_k_scale, k)
        op_v(x_q, x_scale, w_v, w_v_scale, v)

        q_rot = pl.create_tensor([t, O_WIDTH], dtype=pl.BF16)
        k_rot = pl.create_tensor([t, KV_WIDTH], dtype=pl.BF16)
        qwen3_qk_rmsnorm_rope(qg, k, q_norm_w, k_norm_w, rope_cos, rope_sin, q_rot, k_rot)

        attended = pl.create_tensor([t, O_WIDTH], dtype=pl.BF16)
        qwen3_attn_core_pipe(q_rot, k_rot, v, mask_bias, attended)

        act_q = pl.create_tensor([t, O_WIDTH], dtype=pl.INT8)
        act_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        qwen3_attn_gate_quant(attended, qg, act_q, act_scale)

        op_o(act_q, act_scale, w_o, w_o_scale, out)
        return out

    return qwen3_gated_attention


qwen3_gated_attention = build_kernel()


# acceptance: the gate is the distance to the W8A8 golden, as a fraction of
# what quantization itself costs, whole-tensor and on the worst token row.
# Set from the measurements in GDN_STATE.md; an output mirrored across the
# truth would sit at 2.0.
MAX_VS_GOLDEN = 0.50            # whole tensor
MAX_ROW_VS_GOLDEN = 0.75        # worst token row
ABS_VS_GOLDEN = 1e-2            # fallback when the harness supplies no inputs

INPUT_NAMES = ("x", "w_q", "w_q_scale", "w_k", "w_k_scale", "w_v", "w_v_scale",
               "w_o", "w_o_scale", "q_norm_w", "k_norm_w")


def _tables(t: int):
    """The RoPE tables for positions 0..t-1 and the attention core's mask table."""
    import torch

    import reference

    row_tile, kv_tile = attn_core_pipe.ROW_TILE, attn_core_pipe.KV_TILE
    cos, sin = reference.rotary_cos_sin(torch.arange(t), QWEN3_8_27B)
    return cos.float(), sin.float(), attn_core.causal_bias(row_tile, kv_tile)


def build_tensor_specs(t: int = BENCH_SEQ, weights: str | None = None):
    import torch
    from golden import TensorSpec

    import reference

    def draw(name):
        def make():
            return reference.attn_inputs(name, t, QWEN3_8_27B, weights)
        return make

    def table(i):
        return lambda: _tables(t)[i]

    row_tile, kv_tile = attn_core_pipe.ROW_TILE, attn_core_pipe.KV_TILE
    bias_width = 2 * kv_tile - row_tile
    return [
        TensorSpec("x", [t, HIDDEN], torch.bfloat16, init_value=draw("x")),
        TensorSpec("w_q", [QG_WIDTH, HIDDEN], torch.int8, init_value=draw("w_q")),
        TensorSpec("w_q_scale", [QG_WIDTH], torch.float32, init_value=draw("w_q_scale")),
        TensorSpec("w_k", [KV_WIDTH, HIDDEN], torch.int8, init_value=draw("w_k")),
        TensorSpec("w_k_scale", [KV_WIDTH], torch.float32, init_value=draw("w_k_scale")),
        TensorSpec("w_v", [KV_WIDTH, HIDDEN], torch.int8, init_value=draw("w_v")),
        TensorSpec("w_v_scale", [KV_WIDTH], torch.float32, init_value=draw("w_v_scale")),
        TensorSpec("w_o", [HIDDEN, O_WIDTH], torch.int8, init_value=draw("w_o")),
        TensorSpec("w_o_scale", [HIDDEN], torch.float32, init_value=draw("w_o_scale")),
        TensorSpec("q_norm_w", [D], torch.bfloat16, init_value=draw("q_norm_w")),
        TensorSpec("k_norm_w", [D], torch.bfloat16, init_value=draw("k_norm_w")),
        TensorSpec("rope_cos", [t, ROT], torch.float32, init_value=table(0)),
        TensorSpec("rope_sin", [t, ROT], torch.float32, init_value=table(1)),
        TensorSpec("mask_bias", [row_tile, bias_width], torch.float32, init_value=table(2)),
        TensorSpec("out", [t, HIDDEN], torch.bfloat16),
    ]


def compare_out():
    """Gate the mixer against its own W8A8 golden; score it against the truth beside.

    The **gate is the distance to `expected`**, the fake-quant chain the
    kernels implement, as whole-tensor and worst-row norms in units of the
    quantisation floor -- the MLP's shape of bound, for the MLP's reason:
    `o_proj` sums 6144 terms that cancel, so an element-wise bound calls a
    correct kernel wrong. The truth ratio is a diagnostic only; gating on it
    alone accepts an output mirrored across the truth.
    """
    import reference

    def compare(actual, expected, inputs=None, **_kwargs):
        dev, fq = actual.double(), expected.double()

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())

        truth = None
        if inputs:
            w = {name: inputs[name].cpu() for name in INPUT_NAMES}
            cos, sin = inputs["rope_cos"].cpu(), inputs["rope_sin"].cpu()
            truth = reference.attn_truth(w["x"], w, cos, sin, QWEN3_8_27B)
        dev_fq, dev_row_fq = rel(dev, fq), worst_row(dev, fq)
        lines = []
        if truth is not None:
            floor, row_floor = rel(fq, truth), worst_row(fq, truth)
            lines.append(f"[stats] out vs W8A8 golden: {dev_fq:.3e} = {dev_fq / floor:.3f} of the "
                         f"quantisation floor {floor:.3e} (gate {MAX_VS_GOLDEN})")
            lines.append(f"[stats] worst token row vs golden: {dev_row_fq:.3e} = "
                         f"{dev_row_fq / row_floor:.3f} of its floor {row_floor:.3e} "
                         f"(gate {MAX_ROW_VS_GOLDEN})")
            lines.append(f"[stats] diagnostic, vs float64 truth: kernel {rel(dev, truth):.3e}, "
                         f"W8A8 {floor:.3e} -> {rel(dev, truth) / floor:.3f}x")
            ok = dev_fq <= MAX_VS_GOLDEN * floor and dev_row_fq <= MAX_ROW_VS_GOLDEN * row_floor
        else:
            lines.append(f"[stats] out vs W8A8 golden: {dev_fq:.3e}, worst row {dev_row_fq:.3e} "
                         f"(no inputs; absolute bound {ABS_VS_GOLDEN})")
            ok = dev_fq <= ABS_VS_GOLDEN and dev_row_fq <= ABS_VS_GOLDEN
        detail = "\n".join(lines)
        return ok, "" if ok else detail

    return compare


def golden_qwen3_gated_attention(tensors):
    import torch

    import reference

    w = {name: tensors[name] for name in INPUT_NAMES}
    ref = reference.attn_w8a8(tensors["x"], w, tensors["rope_cos"], tensors["rope_sin"], QWEN3_8_27B)
    tensors["out"].copy_(ref.to(torch.bfloat16))


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--weights", type=str, default=None,
                        help="one layer's BF16 self_attn tensors from weights.py; default: random weights")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % attn_core_pipe.KV_TILE:
        parser.error(f"--seq-len must be a multiple of {attn_core_pipe.KV_TILE}")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    for t in cases:
        print(f"[case] gated_attention T={t}", flush=True)
        result = run(
            fn=build_kernel(t=t),
            specs=build_tensor_specs(t=t, weights=args.weights),
            golden_fn=golden_qwen3_gated_attention,
            compare_fn={"out": compare_out()},
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
                ring_heap=ATTN_RING_HEAP,
            ),
            rtol=1e-2,
            atol=1e-5,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at T={t}")
