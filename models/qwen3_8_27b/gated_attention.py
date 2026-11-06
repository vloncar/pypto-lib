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
import prefill_kv_read
import prefill_kv_write
from attn_core import qwen3_attn_core
from attn_gate_quant import qwen3_attn_gate_quant
from qk_rmsnorm_rope import qwen3_qk_rmsnorm_rope
from quant_x import gdn_quant_x
from config import KV_PAGE_SIZE, QWEN3_8_27B

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


def build_kernel(t: int = BENCH_SEQ, inline: bool = False,
                 max_pages: int | None = None):
    """The mixer at one shape, composing eight operators as inlined callees.

    Static shape only: a packed varlen caller needs a tail, which every
    operator in this directory still owes. `row_base` is the attention core's
    per-token row start; the other stages read the buffer as one row.
    """
    max_pages = max_pages if max_pages is not None else t // KV_PAGE_SIZE
    if max_pages < t // KV_PAGE_SIZE:
        raise ValueError("the cache must hold this chunk")
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
    core_kv_tile, core_row_tile = attn_core.KV_TILE, attn_core.ROW_TILE
    if core_kv_tile % KV_PAGE_SIZE:
        raise ValueError("the attention tile must contain whole cache pages")
    kv_write = prefill_kv_write.build_kernel(t=t, max_pages=max_pages, inline=True)
    kv_read = prefill_kv_read.build_kernel(t=t, max_pages=max_pages, inline=True)
    # o_proj takes an 8 x 2 patch of output tiles: `_patch` would give 8 x 8,
    # which leaves 40 blocks for 24 cores and measures 1186.8 us against 1056.5
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
        row_base: pl.Tensor[[t], pl.INT32],
        page_bases: pl.Tensor[[1], pl.INT32],
        block_table: pl.Tensor[[1, max_pages], pl.INT32],
        page_base: pl.Scalar[pl.INDEX],
        layer_base: pl.Scalar[pl.INDEX],
        k_pages: pl.InOut[pl.Tensor[[prefill_kv_write.KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[prefill_kv_write.KVROWS_DYN, KV_WIDTH], pl.BF16]],
        out: pl.Out[pl.Tensor[[t, HIDDEN], pl.BF16]],
    ):
        k_pages.bind_dynamic(0, prefill_kv_write.KVROWS_DYN)
        v_pages.bind_dynamic(0, prefill_kv_write.KVROWS_DYN)
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

        kv_write(k_rot, v, block_table, page_base, layer_base, k_pages, v_pages)
        k_all = pl.create_tensor([max_pages * KV_PAGE_SIZE, KV_WIDTH], dtype=pl.BF16)
        v_all = pl.create_tensor([max_pages * KV_PAGE_SIZE, KV_WIDTH], dtype=pl.BF16)
        kv_read(k_pages, v_pages, block_table, page_base, layer_base, k_all, v_all)

        attended = pl.create_tensor([t, O_WIDTH], dtype=pl.BF16)
        # This entry still owns one cache row; row_base is all zero and names owner 0.
        qwen3_attn_core(q_rot, k_all, v_all, mask_bias, row_base, row_base,
                        page_bases, attended)

        act_q = pl.create_tensor([t, O_WIDTH], dtype=pl.INT8)
        act_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        qwen3_attn_gate_quant(attended, qg, act_q, act_scale)

        op_o(act_q, act_scale, w_o, w_o_scale, out)
        return out, k_pages, v_pages

    return qwen3_gated_attention


qwen3_gated_attention = build_kernel()


def build_packed_kernel(t: int, rows: int, max_pages: int, inline: bool = False):
    """Full attention over packed rows with one independent paged cache per row."""
    if t % attn_core.KV_TILE or rows < 1 or rows * max_pages < t // KV_PAGE_SIZE:
        raise ValueError("packed attention needs whole KV tiles and sufficient cache pages")
    m_group = _patch(t // M_TILE)
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
    op_o = a8w8_linear.build_kernel(m=t, k=O_WIDTH, n=HIDDEN, name="o_proj", inline=True,
                                    m_tile=M_TILE, n_tile=N_TILE, k_tile=K_TILE,
                                    m_group=m_group, n_group=2, ring_slots=RING_SLOTS)
    kv_write = prefill_kv_write.build_packed_kernel(t=t, rows=rows,
                                                     max_pages=max_pages, inline=True)
    kv_read = prefill_kv_read.build_packed_kernel(rows=rows, max_pages=max_pages,
                                                   inline=True)
    bias_width = 2 * attn_core.KV_TILE - attn_core.ROW_TILE

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_gated_attention_packed(
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
        mask_bias: pl.Tensor[[attn_core.ROW_TILE, bias_width], pl.FP32],
        row_base: pl.Tensor[[t], pl.INT32],
        row_owner: pl.Tensor[[t], pl.INT32],
        page_bases: pl.Tensor[[rows], pl.INT32],
        visible_pages: pl.Tensor[[rows], pl.INT32],
        block_table: pl.Tensor[[rows, max_pages], pl.INT32],
        layer_base: pl.Scalar[pl.INDEX],
        k_pages: pl.InOut[pl.Tensor[[prefill_kv_write.KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[prefill_kv_write.KVROWS_DYN, KV_WIDTH], pl.BF16]],
        out: pl.Out[pl.Tensor[[t, HIDDEN], pl.BF16]],
    ):
        k_pages.bind_dynamic(0, prefill_kv_write.KVROWS_DYN)
        v_pages.bind_dynamic(0, prefill_kv_write.KVROWS_DYN)
        x_q = pl.create_tensor([t, HIDDEN], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        gdn_quant_x(x, x_q, x_scale)
        qg = pl.create_tensor([t, QG_WIDTH], dtype=pl.BF16)
        k = pl.create_tensor([t, KV_WIDTH], dtype=pl.BF16)
        v = pl.create_tensor([t, KV_WIDTH], dtype=pl.BF16)
        op_q(x_q, x_scale, w_q, w_q_scale, qg)
        op_k(x_q, x_scale, w_k, w_k_scale, k)
        op_v(x_q, x_scale, w_v, w_v_scale, v)
        q_rot = pl.create_tensor([t, O_WIDTH], dtype=pl.BF16)
        k_rot = pl.create_tensor([t, KV_WIDTH], dtype=pl.BF16)
        qwen3_qk_rmsnorm_rope(qg, k, q_norm_w, k_norm_w, rope_cos, rope_sin, q_rot, k_rot)
        kv_write(k_rot, v, row_base, row_owner, page_bases,
                 block_table, layer_base, k_pages, v_pages)
        scratch_rows = rows * max_pages * KV_PAGE_SIZE
        k_all = pl.create_tensor([scratch_rows, KV_WIDTH], dtype=pl.BF16)
        v_all = pl.create_tensor([scratch_rows, KV_WIDTH], dtype=pl.BF16)
        kv_read(k_pages, v_pages, block_table, visible_pages, layer_base, k_all, v_all)
        attended = pl.create_tensor([t, O_WIDTH], dtype=pl.BF16)
        qwen3_attn_core(q_rot, k_all, v_all, mask_bias, row_base, row_owner,
                        page_bases, attended)
        act_q = pl.create_tensor([t, O_WIDTH], dtype=pl.INT8)
        act_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        qwen3_attn_gate_quant(attended, qg, act_q, act_scale)
        op_o(act_q, act_scale, w_o, w_o_scale, out)
        return out, k_pages, v_pages

    return qwen3_gated_attention_packed


# acceptance: the gate is the distance to the W8A8 golden, as a fraction of
# what quantization itself costs, whole-tensor and on the worst token row.
# An output mirrored across the reference would sit at 2.0.
MAX_VS_GOLDEN = 0.50            # whole tensor
MAX_ROW_VS_GOLDEN = 0.75        # worst token row
ABS_VS_GOLDEN = 1e-2            # fallback when the harness supplies no inputs

INPUT_NAMES = ("x", "w_q", "w_q_scale", "w_k", "w_k_scale", "w_v", "w_v_scale",
               "w_o", "w_o_scale", "q_norm_w", "k_norm_w")


def prefix_cache(tensors):
    """The ordered cache rows decode reads before this chunk writes its own pages."""
    import reference

    length = int(tensors["page_base"]) * KV_PAGE_SIZE
    if not length:
        return None, None
    rows = reference.kv_page_rows(tensors["block_table"], 0, length, KV_PAGE_SIZE)
    base = int(tensors["layer_base"])
    return tensors["k_pages"][base + rows], tensors["v_pages"][base + rows]


def _tables(t: int, page_base: int = 0):
    """The RoPE tables for this chunk's absolute positions and the causal mask."""
    import torch

    import reference

    row_tile, kv_tile = attn_core.ROW_TILE, attn_core.KV_TILE
    start = page_base * KV_PAGE_SIZE
    cos, sin = reference.rotary_cos_sin(torch.arange(start, start + t), QWEN3_8_27B)
    return cos.float(), sin.float(), attn_core.causal_bias(row_tile, kv_tile)


def build_tensor_specs(t: int = BENCH_SEQ, weights: str | None = None,
                       page_base: int = 0, max_pages: int | None = None):
    import torch
    from golden import ScalarSpec, TensorSpec

    import reference

    def draw(name):
        def make():
            return reference.attn_inputs(name, t, QWEN3_8_27B, weights)
        return make

    def table(i):
        return lambda: _tables(t, page_base)[i]

    row_tile, kv_tile = attn_core.ROW_TILE, attn_core.KV_TILE
    bias_width = 2 * kv_tile - row_tile
    max_pages = max_pages if max_pages is not None else page_base + t // KV_PAGE_SIZE
    cache = prefill_kv_write._inputs(t, page_base, KV_PAGE_SIZE, max_pages, 3)
    pool_rows = cache["k_pages"].shape[0]
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
        TensorSpec("row_base", [t], torch.int32, init_value=lambda: attn_core.row_base_table(t)),
        TensorSpec("page_bases", [1], torch.int32,
                   init_value=lambda: torch.tensor([page_base], dtype=torch.int32)),
        TensorSpec("block_table", [1, max_pages], torch.int32,
                   init_value=lambda: cache["block_table"]),
        ScalarSpec("page_base", torch.int64, page_base, compile_runtime=True),
        ScalarSpec("layer_base", torch.int64, 0, compile_runtime=True),
        TensorSpec("k_pages", [pool_rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: cache["k_pages"]),
        TensorSpec("v_pages", [pool_rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: cache["v_pages"]),
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

    def compare(actual, expected, inputs=None, expected_outputs=None, **_kwargs):
        dev, fq = actual.double(), expected.double()

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())

        truth = None
        if inputs:
            w = {name: inputs[name].cpu() for name in INPUT_NAMES}
            cos, sin = inputs["rope_cos"].cpu(), inputs["rope_sin"].cpu()
            prefix_k, prefix_v = prefix_cache({**inputs,
                                               "k_pages": expected_outputs["k_pages"],
                                               "v_pages": expected_outputs["v_pages"]})
            truth = reference.attn_truth(w["x"], w, cos, sin, QWEN3_8_27B,
                                         prefix_k, prefix_v)
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


def compare_cache(name: str):
    """Check written KV against the quantisation floor and every other row exactly."""
    import torch

    import reference

    def compare(actual, expected, inputs=None, expected_outputs=None, **_kwargs):
        if inputs is None:
            return False, f"{name}: block table is required to check placement"
        start = int(inputs["page_base"]) * KV_PAGE_SIZE
        t = inputs["x"].shape[0]
        base = int(inputs["layer_base"])
        rows = base + reference.kv_page_rows(inputs["block_table"], 0, start + t,
                                             KV_PAGE_SIZE)[start:]
        untouched = torch.ones(actual.shape[0], dtype=torch.bool)
        untouched[rows] = False
        if not bool((actual[untouched] == expected[untouched]).all()):
            return False, f"{name}: rows outside this chunk changed"
        w = {key: inputs[key] for key in INPUT_NAMES}
        prefix_k, prefix_v = prefix_cache({**inputs,
                                           "k_pages": expected_outputs["k_pages"],
                                           "v_pages": expected_outputs["v_pages"]})
        _, k_truth, v_truth = reference.attn_truth(
            w["x"], w, inputs["rope_cos"], inputs["rope_sin"], QWEN3_8_27B,
            prefix_k, prefix_v, return_cache=True)
        truth = k_truth if name == "k_pages" else v_truth
        dev, fq, truth = actual[rows].double(), expected[rows].double(), truth.double()
        den = torch.linalg.vector_norm(truth)
        floor = float(torch.linalg.vector_norm(fq - truth) / den)
        over = float(torch.linalg.vector_norm(dev - truth) / den) / floor
        near = float(torch.linalg.vector_norm(dev - fq) / den) / floor
        peak = torch.linalg.vector_norm(truth, dim=1).max()
        row_floor = float(torch.linalg.vector_norm(fq - truth, dim=1).max() / peak)
        row_over = float(torch.linalg.vector_norm(dev - truth, dim=1).max() / peak) / row_floor
        row_near = float(torch.linalg.vector_norm(dev - fq, dim=1).max() / peak) / row_floor
        line = (f"[stats] {name}: vs truth {over:.3f}x floor, worst token {row_over:.3f}x; "
                f"vs fake quant {near:.3f}x, worst token {row_near:.3f}x")
        print(line, flush=True)
        ok = over <= 1.15 and row_over <= 1.25 and near <= 0.5 and row_near <= 0.75
        return ok, "" if ok else line

    return compare


def golden_qwen3_gated_attention(tensors):
    import torch

    import reference

    w = {name: tensors[name] for name in INPUT_NAMES}
    prefix_k, prefix_v = prefix_cache(tensors)
    ref, k_rot, v = reference.attn_w8a8(
        tensors["x"], w, tensors["rope_cos"], tensors["rope_sin"], QWEN3_8_27B,
        prefix_k, prefix_v, return_cache=True)
    tensors["out"].copy_(ref.to(torch.bfloat16))
    base = int(tensors["layer_base"])
    reference.write_kv_pages(tensors["k_pages"][base:], tensors["v_pages"][base:],
                             tensors["block_table"],
                             k_rot, v, int(tensors["page_base"]) * KV_PAGE_SIZE, KV_PAGE_SIZE)


def build_packed_tensor_specs(t: int = 768, starts: tuple[int, ...] = (0, 256),
                              bases: tuple[int, ...] = (0, 4), max_pages: int = 10,
                              weights: str | None = None):
    import torch
    from golden import ScalarSpec, TensorSpec

    import reference

    if any(base % (attn_core.KV_TILE // KV_PAGE_SIZE) for base in bases):
        raise ValueError("packed attention prefixes must align to the attention KV tile")
    cache = prefill_kv_write._packed_inputs(t, starts, bases, max_pages, KV_PAGE_SIZE)
    rows = len(starts)
    ends = starts[1:] + (t,)
    positions = torch.cat([
        torch.arange(base * KV_PAGE_SIZE, base * KV_PAGE_SIZE + end - start)
        for base, start, end in zip(bases, starts, ends)
    ])
    cos, sin = reference.rotary_cos_sin(positions, QWEN3_8_27B)
    base_specs = build_tensor_specs(t=t, weights=weights, max_pages=max_pages)
    pool_rows = cache["k_pages"].shape[0]
    return base_specs[:11] + [
        TensorSpec("rope_cos", [t, ROT], torch.float32, init_value=lambda: cos.float()),
        TensorSpec("rope_sin", [t, ROT], torch.float32, init_value=lambda: sin.float()),
        base_specs[13],
        TensorSpec("row_base", [t], torch.int32, init_value=lambda: cache["row_base"]),
        TensorSpec("row_owner", [t], torch.int32, init_value=lambda: cache["row_owner"]),
        TensorSpec("page_bases", [rows], torch.int32,
                   init_value=lambda: cache["page_bases"]),
        TensorSpec("visible_pages", [rows], torch.int32,
                   init_value=lambda: torch.tensor(
                       [base + (end - start) // KV_PAGE_SIZE
                        for base, start, end in zip(bases, starts, ends)], dtype=torch.int32)),
        TensorSpec("block_table", [rows, max_pages], torch.int32,
                   init_value=lambda: cache["block_table"]),
        ScalarSpec("layer_base", torch.int64, 0, compile_runtime=True),
        TensorSpec("k_pages", [pool_rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: cache["k_pages"]),
        TensorSpec("v_pages", [pool_rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: cache["v_pages"]),
        TensorSpec("out", [t, HIDDEN], torch.bfloat16),
    ]


def golden_qwen3_gated_attention_packed(tensors):
    import torch

    import reference

    table = tensors["block_table"]
    pages = tensors["page_bases"].tolist()
    starts = sorted(set(tensors["row_base"].tolist()))
    ends = starts[1:] + [tensors["x"].shape[0]]
    base = int(tensors["layer_base"])
    weights = {name: tensors[name] for name in INPUT_NAMES}
    for owner, (start, end) in enumerate(zip(starts, ends)):
        if int(tensors["visible_pages"][owner]) != pages[owner] + (end - start) // KV_PAGE_SIZE:
            raise ValueError("packed attention visible pages do not cover this row")
        prefix = pages[owner] * KV_PAGE_SIZE
        rows = reference.kv_page_rows(table, owner, prefix, KV_PAGE_SIZE) if prefix else None
        prefix_k = tensors["k_pages"][base + rows] if rows is not None else None
        prefix_v = tensors["v_pages"][base + rows] if rows is not None else None
        out, k_rot, v = reference.attn_w8a8(
            tensors["x"][start:end], weights, tensors["rope_cos"][start:end],
            tensors["rope_sin"][start:end], QWEN3_8_27B,
            prefix_k, prefix_v, return_cache=True)
        tensors["out"][start:end].copy_(out.to(torch.bfloat16))
        reference.write_kv_pages(tensors["k_pages"][base:], tensors["v_pages"][base:],
                                 table, k_rot, v, prefix, KV_PAGE_SIZE, seq=owner)


def compare_packed_out(actual, expected, inputs=None, expected_outputs=None, **_kwargs):
    import reference

    if inputs is None or expected_outputs is None:
        return False, "packed attention needs its input cache to score both owners"
    table = inputs["block_table"]
    starts = sorted(set(inputs["row_base"].tolist()))
    ends = starts[1:] + [inputs["x"].shape[0]]
    weights = {name: inputs[name] for name in INPUT_NAMES}
    for owner, (start, end) in enumerate(zip(starts, ends)):
        prefix = int(inputs["page_bases"][owner]) * KV_PAGE_SIZE
        rows = reference.kv_page_rows(table, owner, prefix, KV_PAGE_SIZE) if prefix else None
        base = int(inputs["layer_base"])
        prefix_k = expected_outputs["k_pages"][base + rows] if rows is not None else None
        prefix_v = expected_outputs["v_pages"][base + rows] if rows is not None else None
        truth = reference.attn_truth(
            inputs["x"][start:end], weights, inputs["rope_cos"][start:end],
            inputs["rope_sin"][start:end], QWEN3_8_27B, prefix_k, prefix_v)
        dev, fq = actual[start:end].double(), expected[start:end].double()
        floor = float((fq - truth).norm() / truth.norm())
        near = float((dev - fq).norm() / truth.norm())
        row_floor = float(((fq - truth).norm(dim=-1) / truth.norm(dim=-1)).max())
        row_near = float(((dev - fq).norm(dim=-1) / truth.norm(dim=-1)).max())
        if not (near <= MAX_VS_GOLDEN * floor and row_near <= MAX_ROW_VS_GOLDEN * row_floor):
            return False, (f"owner {owner}: output {near / floor:.3f}x floor, "
                           f"worst row {row_near / row_floor:.3f}x floor")
    return True, "both packed attention owners pass the W8A8 floor gate"


def compare_packed_cache(name: str):
    def compare(actual, expected, inputs=None, **_kwargs):
        import torch

        import reference

        if inputs is None:
            return False, f"{name}: missing packed block table"
        table = inputs["block_table"]
        starts = sorted(set(inputs["row_base"].tolist()))
        ends = starts[1:] + [inputs["x"].shape[0]]
        base = int(inputs["layer_base"])
        touched = torch.zeros(actual.shape[0], dtype=torch.bool)
        for owner, (start, end) in enumerate(zip(starts, ends)):
            first = int(inputs["page_bases"][owner]) * KV_PAGE_SIZE
            rows = reference.kv_page_rows(table, owner, first + end - start,
                                           KV_PAGE_SIZE)[first:]
            touched[base + rows] = True
            dev, want = actual[base + rows].double(), expected[base + rows].double()
            rel = float((dev - want).norm() / want.norm())
            if not rel <= ABS_VS_GOLDEN:
                return False, f"{name}: owner {owner} cache relative error {rel:.3e}"
        if not bool((actual[~touched] == expected[~touched]).all()):
            return False, f"{name}: unowned cache rows changed"
        return True, f"{name}: both owners pass; unowned rows untouched"

    return compare


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--page-base", type=int, default=0,
                        help="cached 128-token pages before this chunk, with --seq-len")
    parser.add_argument("--weights", type=str, default=None,
                        help="one layer's BF16 self_attn tensors from weights.py; default: random weights")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--packed", action="store_true", default=False)
    parser.add_argument("--row-starts", type=str, default="0,256")
    parser.add_argument("--row-page-bases", type=str, default="0,4")
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else ([768] if args.packed else list(DEFAULT_CASES))
    if args.seq_len and args.seq_len % attn_core.KV_TILE:
        parser.error(f"--seq-len must be a multiple of {attn_core.KV_TILE}")
    if args.page_base % (attn_core.KV_TILE // KV_PAGE_SIZE):
        parser.error("--page-base must align to the attention core's KV tile")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    for t in cases:
        if args.packed:
            starts = tuple(int(value) for value in args.row_starts.split(","))
            bases = tuple(int(value) for value in args.row_page_bases.split(","))
            if len(starts) != len(bases) or not starts:
                parser.error("packed attention needs one page base per row start")
            ends = starts[1:] + (t,)
            max_pages = max(base + (end - start) // KV_PAGE_SIZE
                            for base, start, end in zip(bases, starts, ends))
            print(f"[case] gated_attention packed T={t} starts={starts} bases={bases}",
                  flush=True)
            result = run(
                fn=build_packed_kernel(t=t, rows=len(starts), max_pages=max_pages),
                specs=build_packed_tensor_specs(t=t, starts=starts, bases=bases,
                                                max_pages=max_pages, weights=args.weights),
                golden_fn=None if args.compile_only else golden_qwen3_gated_attention_packed,
                compare_fn={"out": compare_packed_out,
                            "k_pages": compare_packed_cache("k_pages"),
                            "v_pages": compare_packed_cache("v_pages")},
                config=dict(backend_type=args.backend_type,
                            execution_mode=args.execution_mode, device_id=args.device,
                            ring_heap=ATTN_RING_HEAP),
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(f"FAILED at packed gated_attention T={t}")
            continue
        page_base = args.page_base if args.seq_len else 0
        max_pages = page_base + t // KV_PAGE_SIZE + 2
        print(f"[case] gated_attention T={t} page_base={page_base}", flush=True)
        result = run(
            fn=build_kernel(t=t, max_pages=max_pages),
            specs=build_tensor_specs(t=t, weights=args.weights,
                                     page_base=page_base, max_pages=max_pages),
            golden_fn=golden_qwen3_gated_attention,
            compare_fn={"out": compare_out(), "k_pages": compare_cache("k_pages"),
                        "v_pages": compare_cache("v_pages")},
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
