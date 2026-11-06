# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Qwen3.8-27B's whole Gated DeltaNet block: hidden states in, hidden states out.

    x_q, x_scale = quant_x(x)                     per-token int8 of the hidden states
    qkv          = in_proj_qkv(x_q, x_scale)      5120 -> 10240, A8W8
    z            = in_proj_z(x_q, x_scale)        5120 -> 6144,  A8W8
    a, b         = in_proj_ab(x)                  5120 -> 48 each, bf16
    q, k, v      = short_conv(qkv)                depthwise causal conv, then silu
    q, k, beta, g = qk_norm_gate(q, k, a, b)      L2-norm, sigmoid, softplus
    o            = gdn_layer(q, k, v, g, beta)    the delta rule, six stages
    y            = gated_rmsnorm(o, z)            the gated norm, bf16 out
    out          = out_proj(y)                    6144 -> 5120, BF16 both sides

Fourteen operators as one program. Everything between `x` and `out` is
allocated here and never leaves.

Three interfaces are shaped by the composition rather than by any one operator.
The convolution reads the K-1 tokens before each window, so `in_proj_qkv`
writes at `row_off` into the padded buffer the convolution wants, and the K-1
rows above are zeroed here; nothing copies `[T, C]` into `[T + K - 1, C]`. The
convolution writes q, k and v as three tensors rather than one, so each
consumer reads the rows of a contiguous buffer -- and v, which goes straight to
the delta rule, is written in fp16 there rather than cast by a later pass. And
`in_proj_ab` reads `x` in bf16 while everything else reads the quantised copy,
which is the one place the block reads the hidden states twice.

The dtype at each boundary is the one the standalone operators exchange,
with the same fp16 delta rule inside. `out_proj` is BF16 both sides because
the deployed checkpoint leaves that one projection unquantized -- `contract.py`
carries its published dtype -- so the block quantizes the hidden states and
nothing else.
"""
import pypto.language as pl

import a8w8_linear
import bf16_linear
from gated_rmsnorm import gdn_gated_rmsnorm_bf16
from gdn_layer import gdn_layer as gdn_layer_fn
import gdn_layer
from in_proj_ab import gdn_in_proj_ab
from qk_norm_gate import gdn_qk_norm_gate
from quant_x import gdn_quant_x
import in_proj_qkv
import in_proj_z
import short_conv
import solve_tril
from chunk_h import FSTATE_DYN
from chunk_cumsum import CHUNK_TILE
from config import QWEN3_8_27B

# model config
C = QWEN3_8_27B.hidden_size                 # the residual stream
CQKV = QWEN3_8_27B.qkv_width                # what in_proj_qkv writes
CV = QWEN3_8_27B.value_width                # what in_proj_z writes, out_proj reads
H = QWEN3_8_27B.linear_num_value_heads
HG = QWEN3_8_27B.linear_num_key_heads
D = QWEN3_8_27B.linear_value_head_dim
KCONV = QWEN3_8_27B.linear_conv_kernel_dim  # convolution taps
# validation
# `in_proj_z` binds the floor at M_GROUP * M_TILE. The token count must be a multiple of 2048.
DEFAULT_CASES = (2048, 4096)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# At T = 8192 the block holds about 1.3 GB of intermediates at once: 0.6 GB
# inside the delta rule, and the projections, the convolution's three outputs
# and the quantised hand-offs on top. `gdn_layer` needs 1 GiB for its 0.6 GB,
# so this follows the same ratio, rounded up: the runtime takes powers of two
# only. Under the default 256 MiB per ring the layer alone fails with
# orch_error_code=2 HEAP_RING_DEADLOCK.
BLOCK_RING_HEAP = 4 * 1024 * 1024 * 1024

# tiling
HALO_TILE = 1024        # channels per block of the zero fill above the projection



def run_config(backend_type: str = "a2a3", sim: bool = False, device: int = 0, swimlane: int = 0) -> dict:
    """The config `golden.run` needs for this block. The ring heap is not optional."""
    return dict(backend_type=backend_type, execution_mode=sim, device_id=device,
                ring_heap=BLOCK_RING_HEAP, enable_chip_swimlane=swimlane)


def build_kernel(t: int = BENCH_SEQ, h: int = H, hg: int = HG, d: int = D, chunk: int = CHUNK_TILE,
                 inline: bool = False):
    """The block at one shape, composing fourteen operators as inlined callees."""
    c, cqkv, cv, k = C, CQKV, CV, KCONV
    key_width = hg * d

    op_qkv = in_proj_qkv.build_kernel(t=t, inline=True, row_off=k - 1)
    op_z = in_proj_z.build_kernel(t=t, inline=True)
    conv = dict(t=t, c=cqkv, k=k, inline=True)
    op_conv_q = short_conv.build_kernel(**conv, c_off=0, c_out=key_width,
                                        out_heads=hg, name="conv_q")
    op_conv_k = short_conv.build_kernel(**conv, c_off=key_width, c_out=key_width,
                                        out_heads=hg, name="conv_k")
    op_conv_v = short_conv.build_kernel(**conv, c_off=2 * key_width, c_out=h * d,
                                        out_heads=h, out_dtype=pl.FP16, name="conv_v")
    op_out = bf16_linear.build_kernel(t, cv, c, name="out_proj", inline=True)
    op_state = short_conv.build_state_kernel(t=t, c=cqkv, k=k, inline=True)

    @(pl.jit.inline if inline else pl.jit)
    def gdn_block(
        x: pl.Tensor[[t, c], pl.BF16],
        w_qkv: pl.Tensor[[cqkv, c], pl.INT8],
        w_qkv_scale: pl.Tensor[[cqkv], pl.FP32],
        w_z: pl.Tensor[[cv, c], pl.INT8],
        w_z_scale: pl.Tensor[[cv], pl.FP32],
        w_ab: pl.Tensor[[2 * h, c], pl.BF16],
        conv_w: pl.Tensor[[k, cqkv], pl.FP32],
        a_log: pl.Tensor[[h], pl.BF16],
        dt_bias: pl.Tensor[[h], pl.BF16],
        norm_w: pl.Tensor[[d], pl.BF16],
        w_out: pl.Tensor[[c, cv], pl.BF16],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        mask_strict: pl.Tensor[[chunk, chunk], pl.FP32],
        eye: pl.Tensor[[chunk, chunk], pl.FP16],
        m_diag: pl.Tensor[[chunk, chunk], pl.FP16],
        m_low: pl.Tensor[[chunk, chunk], pl.FP16],
        row_base: pl.Tensor[[t], pl.INT32],
        state_row: pl.Scalar[pl.INDEX],
        conv_state: pl.InOut[pl.Tensor[[short_conv.CSTATE_DYN, k - 1, cqkv], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, h, d, d], pl.FP32]],
        out: pl.Out[pl.Tensor[[t, c], pl.BF16]],
    ):
        conv_state.bind_dynamic(0, short_conv.CSTATE_DYN)
        final_state.bind_dynamic(0, FSTATE_DYN)
        x_q = pl.create_tensor([t, c], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        gdn_quant_x(x, x_q, x_scale)

        # The convolution's first window reads the previous chunk's K-1 projected
        # tokens from the same pool decode consumes. New sequences own a zero row.
        qkv_pad = pl.create_tensor([t + k - 1, cqkv], dtype=pl.BF16)
        cs = pl.reshape(conv_state, [pl.tensor.dim(conv_state, 0) * (k - 1), cqkv])
        cs_row = state_row * (k - 1)
        for blk in pl.spmd(cqkv // HALO_TILE, name_hint="conv_halo"):
            c0 = blk * HALO_TILE
            qkv_pad[0 : k - 1, c0 : c0 + HALO_TILE] = cs[
                cs_row : cs_row + k - 1, c0 : c0 + HALO_TILE]
        op_qkv(x_q, x_scale, w_qkv, w_qkv_scale, qkv_pad)
        op_state(qkv_pad, state_row, conv_state)

        z = pl.create_tensor([t, cv], dtype=pl.BF16)
        op_z(x_q, x_scale, w_z, w_z_scale, z)

        a_proj = pl.create_tensor([t, h], dtype=pl.BF16)
        b_proj = pl.create_tensor([t, h], dtype=pl.BF16)
        gdn_in_proj_ab(x, w_ab, a_proj, b_proj)

        q_conv = pl.create_tensor([t, hg, d], dtype=pl.BF16)
        k_conv = pl.create_tensor([t, hg, d], dtype=pl.BF16)
        v_conv = pl.create_tensor([t, h, d], dtype=pl.FP16)
        op_conv_q(qkv_pad, conv_w, row_base, q_conv)
        op_conv_k(qkv_pad, conv_w, row_base, k_conv)
        op_conv_v(qkv_pad, conv_w, row_base, v_conv)

        q_n = pl.create_tensor([t, hg, d], dtype=pl.FP16)
        k_n = pl.create_tensor([t, hg, d], dtype=pl.FP16)
        beta = pl.create_tensor([h, t], dtype=pl.FP32)
        g = pl.create_tensor([t, h], dtype=pl.FP32)
        gdn_qk_norm_gate(q_conv, k_conv, a_proj, b_proj, a_log, dt_bias, q_n, k_n, beta, g)

        o = pl.create_tensor([t, h, d], dtype=pl.FP16)
        gdn_layer_fn(q_n, k_n, v_conv, g, beta, tril, mask_strict, eye,
                 m_diag, m_low, row_base, state_row, final_state, o)

        y = pl.create_tensor([t, cv], dtype=pl.BF16)
        gdn_gated_rmsnorm_bf16(o, z, norm_w, y)

        op_out(y, w_out, out)
        return out, conv_state, final_state

    return gdn_block


def build_packed_kernel(t: int, rows: int, chunk: int = CHUNK_TILE, inline: bool = False):
    """Full GDN block with separate convolution and recurrent cache owners per row."""
    if t % 2048 or rows < 1 or rows > t // 256:
        raise ValueError("packed GDN block needs 2048-token total and complete 256-token rows")
    c, cqkv, cv, k = C, CQKV, CV, KCONV
    op_qkv = in_proj_qkv.build_kernel(t=t, inline=True, row_off=k - 1)
    op_z = in_proj_z.build_kernel(t=t, inline=True)
    conv = dict(t=t, c=cqkv, k=k, inline=True)
    op_conv_q = short_conv.build_packed_kernel(**conv, rows=rows, c_off=0, c_out=HG * D,
                                               out_heads=HG, name="conv_q_packed")
    op_conv_k = short_conv.build_packed_kernel(**conv, rows=rows, c_off=HG * D,
                                               c_out=HG * D, out_heads=HG,
                                               name="conv_k_packed")
    op_conv_v = short_conv.build_packed_kernel(**conv, rows=rows, c_off=2 * HG * D,
                                               c_out=H * D, out_heads=H,
                                               out_dtype=pl.FP16, name="conv_v_packed")
    op_state = short_conv.build_packed_state_swap_kernel(t=t, rows=rows, c=cqkv, k=k,
                                                          inline=True)
    op_layer = gdn_layer.build_packed_kernel(rows=rows, inline=True)
    op_out = bf16_linear.build_kernel(t, cv, c, name="out_proj", inline=True)

    @(pl.jit.inline if inline else pl.jit)
    def gdn_block_packed(
        x: pl.Tensor[[t, c], pl.BF16],
        w_qkv: pl.Tensor[[cqkv, c], pl.INT8],
        w_qkv_scale: pl.Tensor[[cqkv], pl.FP32],
        w_z: pl.Tensor[[cv, c], pl.INT8],
        w_z_scale: pl.Tensor[[cv], pl.FP32],
        w_ab: pl.Tensor[[2 * H, c], pl.BF16],
        conv_w: pl.Tensor[[k, cqkv], pl.FP32],
        a_log: pl.Tensor[[H], pl.BF16],
        dt_bias: pl.Tensor[[H], pl.BF16],
        norm_w: pl.Tensor[[D], pl.BF16],
        w_out: pl.Tensor[[c, cv], pl.BF16],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        mask_strict: pl.Tensor[[chunk, chunk], pl.FP32],
        eye: pl.Tensor[[chunk, chunk], pl.FP16],
        m_diag: pl.Tensor[[chunk, chunk], pl.FP16],
        m_low: pl.Tensor[[chunk, chunk], pl.FP16],
        row_base: pl.Tensor[[t], pl.INT32],
        row_owner: pl.Tensor[[t], pl.INT32],
        row_ends: pl.Tensor[[rows], pl.INT32],
        state_rows: pl.Tensor[[rows], pl.INT32],
        conv_state: pl.InOut[pl.Tensor[[short_conv.CSTATE_DYN, k - 1, cqkv], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, D, D], pl.FP32]],
        out: pl.Out[pl.Tensor[[t, c], pl.BF16]],
    ):
        conv_state.bind_dynamic(0, short_conv.CSTATE_DYN)
        final_state.bind_dynamic(0, FSTATE_DYN)
        x_q = pl.create_tensor([t, c], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        gdn_quant_x(x, x_q, x_scale)
        qkv_pad = pl.create_tensor([t + k - 1, cqkv], dtype=pl.BF16)
        for blk in pl.spmd(cqkv // HALO_TILE, name_hint="conv_halo_packed"):
            c0 = blk * HALO_TILE
            qkv_pad[0 : k - 1, c0 : c0 + HALO_TILE] = pl.full(
                [k - 1, HALO_TILE], dtype=pl.BF16, value=0.0)
        op_qkv(x_q, x_scale, w_qkv, w_qkv_scale, qkv_pad)
        row_halos = pl.create_tensor([rows * (k - 1), cqkv], dtype=pl.BF16)
        op_state(qkv_pad, row_ends, state_rows, conv_state, row_halos)
        z = pl.create_tensor([t, cv], dtype=pl.BF16)
        op_z(x_q, x_scale, w_z, w_z_scale, z)
        a_proj = pl.create_tensor([t, H], dtype=pl.BF16)
        b_proj = pl.create_tensor([t, H], dtype=pl.BF16)
        gdn_in_proj_ab(x, w_ab, a_proj, b_proj)
        q_conv = pl.create_tensor([t, HG, D], dtype=pl.BF16)
        k_conv = pl.create_tensor([t, HG, D], dtype=pl.BF16)
        v_conv = pl.create_tensor([t, H, D], dtype=pl.FP16)
        op_conv_q(qkv_pad, conv_w, row_base, row_owner, row_halos, q_conv)
        op_conv_k(qkv_pad, conv_w, row_base, row_owner, row_halos, k_conv)
        op_conv_v(qkv_pad, conv_w, row_base, row_owner, row_halos, v_conv)
        q_n = pl.create_tensor([t, HG, D], dtype=pl.FP16)
        k_n = pl.create_tensor([t, HG, D], dtype=pl.FP16)
        beta = pl.create_tensor([H, t], dtype=pl.FP32)
        g = pl.create_tensor([t, H], dtype=pl.FP32)
        gdn_qk_norm_gate(q_conv, k_conv, a_proj, b_proj, a_log, dt_bias, q_n, k_n, beta, g)
        o = pl.create_tensor([t, H, D], dtype=pl.FP16)
        op_layer(q_n, k_n, v_conv, g, beta, tril, mask_strict, eye,
                 m_diag, m_low, state_rows, row_ends, final_state, o)
        y = pl.create_tensor([t, cv], dtype=pl.BF16)
        gdn_gated_rmsnorm_bf16(o, z, norm_w, y)
        op_out(y, w_out, out)
        return out, conv_state, final_state

    return gdn_block_packed


def _cache_pools(resume: bool):
    """The decode pools with one owned row; new sequences start at zero."""
    import torch

    conv = torch.full((4, KCONV - 1, CQKV), -99.0, dtype=torch.bfloat16)
    state = torch.full((4, H, D, D), -1.0, dtype=torch.float32)
    if resume:
        gen = torch.Generator().manual_seed(91)
        conv[2] = (torch.randn(KCONV - 1, CQKV, generator=gen) * 0.1).to(torch.bfloat16)
        state[2] = torch.randn(H, D, D, generator=gen) * 0.02
    else:
        conv[2] = 0.0
        state[2] = 0.0
    return conv, state


def build_tensor_specs(t: int = BENCH_SEQ, h: int = H, hg: int = HG, d: int = D,
                       chunk: int = CHUNK_TILE, weights: str | None = None,
                       resume: bool = False):
    """Every weight the block reads, plus the delta rule's three constants."""
    import torch
    from golden import ScalarSpec, TensorSpec

    import reference

    def block_weights():
        if weights is None:
            return reference.make_block_weights(QWEN3_8_27B)
        return torch.load(weights, weights_only=True)

    def quantized(module):
        q, scale = a8w8_linear.quantized_weight(QWEN3_8_27B, module, weights)
        return q, scale

    w_qkv, w_qkv_scale = quantized("in_proj_qkv")
    w_z, w_z_scale = quantized("in_proj_z")

    def stacked_ab():
        w = block_weights()
        return torch.cat([w["in_proj_a.weight"], w["in_proj_b.weight"]]).to(torch.bfloat16)

    def taps():
        return block_weights()["conv1d.weight"].reshape(CQKV, KCONV).t().contiguous().float()

    def param(name):
        return lambda: block_weights()[name].to(torch.bfloat16)

    def init_tril():
        return torch.tril(torch.ones(chunk, chunk, dtype=torch.float32))

    def init_mask_strict():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows > cols).float()

    return [
        TensorSpec("x", [t, C], torch.bfloat16,
                   init_value=lambda: reference.make_block_inputs(t, QWEN3_8_27B)),
        TensorSpec("w_qkv", [CQKV, C], torch.int8, init_value=w_qkv),
        TensorSpec("w_qkv_scale", [CQKV], torch.float32, init_value=w_qkv_scale),
        TensorSpec("w_z", [CV, C], torch.int8, init_value=w_z),
        TensorSpec("w_z_scale", [CV], torch.float32, init_value=w_z_scale),
        TensorSpec("w_ab", [2 * h, C], torch.bfloat16, init_value=stacked_ab),
        TensorSpec("conv_w", [KCONV, CQKV], torch.float32, init_value=taps),
        TensorSpec("a_log", [h], torch.bfloat16, init_value=param("A_log")),
        TensorSpec("dt_bias", [h], torch.bfloat16, init_value=param("dt_bias")),
        TensorSpec("norm_w", [d], torch.bfloat16, init_value=param("norm.weight")),
        TensorSpec("w_out", [C, CV], torch.bfloat16, init_value=param("out_proj.weight")),
        TensorSpec("tril", [chunk, chunk], torch.float32, init_value=init_tril),
        TensorSpec("mask_strict", [chunk, chunk], torch.float32, init_value=init_mask_strict),
        TensorSpec("eye", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.eye_block(chunk)),
        TensorSpec("m_diag", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.blk_masks(chunk)[0]),
        TensorSpec("m_low", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.blk_masks(chunk)[1]),
        TensorSpec("row_base", [t], torch.int32, init_value=lambda: torch.zeros(t, dtype=torch.int32)),
        ScalarSpec("state_row", torch.int64, 2, compile_runtime=True),
        TensorSpec("conv_state", [4, KCONV - 1, CQKV], torch.bfloat16,
                   init_value=lambda: _cache_pools(resume)[0]),
        TensorSpec("final_state", [4, H, D, D], torch.float32,
                   init_value=lambda: _cache_pools(resume)[1]),
        TensorSpec("out", [t, C], torch.bfloat16),
    ]


def block_weights(tensors):
    """The checkpoint-native weight dict `reference.block` reads, from the kernel's own.

    The int8 projections stay int8 with their scales beside them, which is the form
    `reference.weight` dequantises; `out_proj` is bf16 and carries no scale; `w_ab`'s
    halves split back into the two projections the spec concatenated; and `conv_w` turns
    back from the kernel's `[K, C]` to the checkpoint's `[C, 1, K]`.

    A composed caller needs this: its oracle has to read the same weights the device was
    given, not a second draw from the same source.
    """
    return {
        "in_proj_qkv.weight": tensors["w_qkv"],
        "in_proj_qkv.weight_scale": tensors["w_qkv_scale"],
        "in_proj_z.weight": tensors["w_z"],
        "in_proj_z.weight_scale": tensors["w_z_scale"],
        "out_proj.weight": tensors["w_out"],
        "in_proj_a.weight": tensors["w_ab"][:H],
        "in_proj_b.weight": tensors["w_ab"][H:],
        "conv1d.weight": tensors["conv_w"].t().contiguous().reshape(CQKV, 1, KCONV),
        "A_log": tensors["a_log"],
        "dt_bias": tensors["dt_bias"],
        "norm.weight": tensors["norm_w"],
    }


def golden_gdn_block(weights: str | None = None, resume: bool = False):
    """The fake-quant chain: the block in float64 with the two activation quantisations.

    A factory, like `compare_out`, because the weight set has to reach it: the
    specs draw the device inputs from *weights*, so a golden computed from the
    random set would be a different problem's answer. `compare_out` scores
    against its own chain and would not notice, but `--save-data` persists this
    tensor for a later `--golden-data` replay.
    """
    def golden(tensors):
        st = _chain(tensors["x"].shape[0], weights, quant=True, resume=resume)
        row = int(tensors["state_row"])
        tensors["out"].copy_(st["out"])
        tensors["conv_state"][row].copy_(st["conv_state_out"])
        tensors["final_state"][row].copy_(st["final_state"].transpose(-2, -1))

    return golden


def build_packed_tensor_specs(t: int = 2048, starts: tuple[int, ...] = (0, 256),
                              state_rows: tuple[int, ...] = (1, 3),
                              pool_rows: int = 5, weights: str | None = None,
                              resume: bool = False):
    import torch
    from golden import TensorSpec

    if (len(starts) != len(state_rows) or starts[0] != 0 or t % 2048
            or any(start % 256 for start in starts)
            or any(b - a < 256 for a, b in zip(starts, starts[1:] + (t,)))
            or len(set(state_rows)) != len(state_rows)
            or any(row < 0 or row >= pool_rows for row in state_rows)):
        raise ValueError("packed GDN block needs complete rows and distinct cache owners")
    ends = starts[1:] + (t,)

    def pools():
        conv = torch.full((pool_rows, KCONV - 1, CQKV), -99.0, dtype=torch.bfloat16)
        final = torch.full((pool_rows, H, D, D), -1.0, dtype=torch.float32)
        for owner, row in enumerate(state_rows):
            gen = torch.Generator().manual_seed(91 + owner)
            if resume:
                conv[row] = (torch.randn(KCONV - 1, CQKV, generator=gen) * 0.1).to(torch.bfloat16)
                final[row] = torch.randn(H, D, D, generator=gen) * 0.02
            else:
                conv[row] = 0.0
                final[row] = 0.0
        return conv, final

    conv, final = pools()
    base = build_tensor_specs(t=t, weights=weights, resume=resume)
    row_base = torch.empty(t, dtype=torch.int32)
    row_owner = torch.empty(t, dtype=torch.int32)
    for owner, (start, end) in enumerate(zip(starts, ends)):
        row_base[start:end] = start
        row_owner[start:end] = owner
    return base[:16] + [
        TensorSpec("row_base", [t], torch.int32, init_value=lambda: row_base),
        TensorSpec("row_owner", [t], torch.int32, init_value=lambda: row_owner),
        TensorSpec("row_ends", [len(starts)], torch.int32,
                   init_value=lambda: torch.tensor(ends, dtype=torch.int32)),
        TensorSpec("state_rows", [len(starts)], torch.int32,
                   init_value=lambda: torch.tensor(state_rows, dtype=torch.int32)),
        TensorSpec("conv_state", [pool_rows, KCONV - 1, CQKV], torch.bfloat16,
                   init_value=lambda: conv),
        TensorSpec("final_state", [pool_rows, H, D, D], torch.float32,
                   init_value=lambda: final),
    ] + base[20:]


_PACKED_TRUTH: dict[str, object] = {}


def golden_gdn_block_packed(tensors):
    import torch

    import reference

    weights = block_weights(tensors)
    ends = tensors["row_ends"].tolist()
    starts = [0] + ends[:-1]
    truth_out = torch.empty(tensors["out"].shape, dtype=torch.float64)
    truth_conv = {}
    truth_final = {}
    for owner, (start, end) in enumerate(zip(starts, ends)):
        row = int(tensors["state_rows"][owner])
        conv_in = tensors["conv_state"][row].clone()
        final_in = tensors["final_state"][row].clone()
        truth = reference.block(
            tensors["x"][start:end], weights, QWEN3_8_27B, CHUNK_TILE,
            quant_x=False, quant_y=False, conv_state=conv_in,
            state=final_in.transpose(-2, -1))
        truth_out[start:end].copy_(truth["out"])
        truth_conv[row] = truth["conv_state_out"]
        truth_final[row] = truth["final_state"].transpose(-2, -1)
        st = reference.block(
            tensors["x"][start:end], weights, QWEN3_8_27B, CHUNK_TILE,
            quant_x=True, quant_y=False,
            conv_state=tensors["conv_state"][row],
            state=tensors["final_state"][row].transpose(-2, -1))
        tensors["out"][start:end].copy_(st["out"].to(torch.bfloat16))
        tensors["conv_state"][row].copy_(st["conv_state_out"])
        tensors["final_state"][row].copy_(st["final_state"].transpose(-2, -1))
    _PACKED_TRUTH.clear()
    _PACKED_TRUTH.update(out=truth_out, conv_state=truth_conv, final_state=truth_final)


def compare_packed_out(actual, expected, inputs=None, **_kwargs):
    if inputs is None or "out" not in _PACKED_TRUTH:
        return False, "packed GDN block needs its golden truth for both owners"
    ends = inputs["row_ends"].tolist()
    starts = [0] + ends[:-1]
    for owner, (start, end) in enumerate(zip(starts, ends)):
        truth = _PACKED_TRUTH["out"][start:end]
        fq = expected[start:end].double()
        dev = actual[start:end].double()
        floor = float((fq - truth).norm() / truth.norm())
        over = float((dev - truth).norm() / truth.norm()) / floor
        near = float((dev - fq).norm() / truth.norm()) / floor
        row_floor = float(((fq - truth).norm(dim=-1) / truth.norm(dim=-1)).max())
        row_over = float(((dev - truth).norm(dim=-1) / truth.norm(dim=-1)).max()) / row_floor
        row_near = float(((dev - fq).norm(dim=-1) / truth.norm(dim=-1)).max()) / row_floor
        if not (over <= MAX_OVER_FLOOR and row_over <= MAX_ROW_OVER_FLOOR
                and near <= MAX_VS_GOLDEN and row_near <= MAX_ROW_VS_GOLDEN):
            return False, (f"owner {owner}: truth {over:.3f}x floor, worst row "
                           f"{row_over:.3f}x; fake-quant {near:.3f}x, "
                           f"worst row {row_near:.3f}x")
    return True, "both packed GDN owners pass the quantization-floor gate"


def compare_packed_cache(name: str, state_rows: tuple[int, ...]):
    fill = -99.0 if name == "conv_state" else -1.0

    def compare(actual, expected, **_kwargs):
        import torch

        if name not in _PACKED_TRUTH:
            return False, f"{name}: missing packed golden truth"
        for row in state_rows:
            dev = actual[row].double()
            fq = expected[row].double()
            truth = _PACKED_TRUTH[name][row].double()
            den = torch.linalg.vector_norm(truth)
            floor = float(torch.linalg.vector_norm(fq - truth) / den)
            over = float(torch.linalg.vector_norm(dev - truth) / den) / floor
            near = float(torch.linalg.vector_norm(dev - fq) / den) / floor
            peak = torch.linalg.vector_norm(truth.reshape(truth.shape[0], -1), dim=1).max()
            row_floor = float(torch.linalg.vector_norm(
                (fq - truth).reshape(truth.shape[0], -1), dim=1).max() / peak)
            row_over = float(torch.linalg.vector_norm(
                (dev - truth).reshape(truth.shape[0], -1), dim=1).max() / peak) / row_floor
            row_near = float(torch.linalg.vector_norm(
                (dev - fq).reshape(truth.shape[0], -1), dim=1).max() / peak) / row_floor
            if not (over <= 1.15 and row_over <= 1.25 and near <= 0.50 and row_near <= 0.75):
                return False, (f"{name} row {row}: truth {over:.3f}x floor, worst row "
                               f"{row_over:.3f}x; fake-quant {near:.3f}x, "
                               f"worst row {row_near:.3f}x")
        others = [row for row in range(actual.shape[0]) if row not in state_rows]
        if others and not bool((actual[others] == fill).all()):
            return False, f"{name}: unowned rows changed"
        return True, f"{name}: both owners pass; unowned rows untouched"

    return compare


_CHAIN_CACHE: dict[tuple, dict] = {}


def _chain(t: int, weights: str | None, quant: bool, chunk: int = CHUNK_TILE,
           resume: bool = False):
    """The float64 block, cached. *quant* picks the fake-quant chain or the truth."""
    import torch

    import reference

    a8w8_linear.host_torch_setup()
    key = (t, weights, quant, chunk, resume)
    if key not in _CHAIN_CACHE:
        if weights is None:
            w = reference.make_block_weights(QWEN3_8_27B)
        else:
            w = torch.load(weights, weights_only=True)
        if quant:
            w = reference.quantize_weights(w)
        # `quant_y` stays off: the norm hands `out_proj` bf16, so there is no
        # per-token quantisation of `y` to model, and `GDN_QUANT.weights` no
        # longer names that weight either.
        conv, state = _cache_pools(resume)
        _CHAIN_CACHE[key] = reference.block(
            reference.make_block_inputs(t, QWEN3_8_27B), w, QWEN3_8_27B, chunk,
            quant_x=quant, quant_y=False,
            conv_state=conv[2], state=state[2].transpose(-2, -1))
    return _CHAIN_CACHE[key]


# Acceptance, both against the quantization floor rather than a fixed number.
# The block may sit this far above what W8A8 alone costs, in total and on its
# worst token row.
MAX_OVER_FLOOR = 1.05
MAX_ROW_OVER_FLOOR = 1.25
# ...and the distance to the fake-quant chain itself, as a fraction of that
# floor. The ratios above compare two distances to a third quantity, so they
# do not on their own constrain how far the kernel is from the answer it
# should produce: an output mirrored across the truth, `2 * truth - fq`, sits
# at exactly the floor distance from the truth and passes them at 1.000x while
# being twice the floor from `fq`.
MAX_VS_GOLDEN = 1.00
MAX_ROW_VS_GOLDEN = 1.50


def compare_out(t: int, weights: str | None, resume: bool = False):
    """Score the block against the float64 truth, beside what quantization alone costs.

    **Not** an element-wise gate against the fake-quant chain, which is the
    obvious choice and the wrong one. The block quantizes the hidden states
    per token, so a difference far below the int8 step flips values sitting
    near a rounding boundary and every projection downstream then reads a
    different integer. An element-wise comparison after a quantization step
    measures which side of a boundary a value landed on, not whether the
    kernel is right.

    What is meaningful is how much the block adds to what W8A8 already costs,
    so both gates are ratios against the quantization floor and neither
    carries a tolerance of its own. The second one is per token row: a
    corrupted row barely moves the whole-tensor norm -- one dead row in 2048
    is 1.12x the floor -- while moving the worst row by more than 10x.

    Those two ratios are not sufficient on their own. Each compares a distance
    to the *truth* against another distance to the truth, so neither bounds how
    far the kernel is from the fake-quant chain it implements: an output
    mirrored across the truth, `2 * truth - fq`, is exactly the floor distance
    from the truth and passes both at 1.000x while being twice the floor from
    the answer it should have produced. Two further gates bound the distance to
    `fq` directly, whole-tensor and worst row, in the same floor-relative units.

    The worst-row gate is also what bounds a single element, which is why there is no
    per-point comparator on top: a row is 5120 wide, so an element wrong by about one
    row's RMS magnitude already moves that row's relative norm past the bound.
    """
    def compare(actual, expected, **_kwargs):
        dev = actual.double()
        fq = _chain(t, weights, quant=True, resume=resume)["out"]
        truth = _chain(t, weights, quant=False, resume=resume)["out"]

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())

        floor, dev_truth = rel(fq, truth), rel(dev, truth)
        row_floor, dev_row = worst_row(fq, truth), worst_row(dev, truth)
        over, row_over = dev_truth / floor, dev_row / row_floor
        dev_fq, dev_row_fq = rel(dev, fq), worst_row(dev, fq)
        near, row_near = dev_fq / floor, dev_row_fq / row_floor
        ok = (over <= MAX_OVER_FLOOR and row_over <= MAX_ROW_OVER_FLOOR
              and near <= MAX_VS_GOLDEN and row_near <= MAX_ROW_VS_GOLDEN)
        if ok:
            return True, ""
        head = (f"    out: rel frob {dev_truth:.3e} vs the float64 truth against a "
                f"quantization floor of {floor:.3e} ({over:.3f}x); worst token row "
                f"{dev_row:.3e} against {row_floor:.3e} ({row_over:.3f}x); "
                f"{dev_fq:.3e} vs the fake-quant chain ({near:.3f} of the floor, "
                f"worst row {row_near:.3f})")
        detail = (f"{over:.3f}x the quantization floor (<={MAX_OVER_FLOOR}), "
                  f"worst row {row_over:.3f}x (<={MAX_ROW_OVER_FLOOR}); "
                  f"vs the fake-quant chain {near:.3f} (<={MAX_VS_GOLDEN}), "
                  f"worst row {row_near:.3f} (<={MAX_ROW_VS_GOLDEN})")
        return False, f"{head}\n{detail}"

    return compare


def compare_cache(name: str, t: int, weights: str | None, resume: bool = False):
    """Score the owned state row against the quantisation floor and the other rows exactly."""
    key = "conv_state_out" if name == "conv_state" else "final_state"
    fill = -99.0 if name == "conv_state" else -1.0

    def value(chain):
        st = _chain(t, weights, quant=chain, resume=resume)[key]
        return st if name == "conv_state" else st.transpose(-2, -1)

    def compare(actual, expected, **_kwargs):
        import torch

        rows = [r for r in range(actual.shape[0]) if r != 2]
        if rows and not bool((actual[rows] == fill).all()):
            return False, f"    {name}: unowned rows changed"
        dev = actual[2].double()
        fq = value(True).double()
        truth = value(False).double()
        floor = float(torch.linalg.vector_norm(fq - truth) / torch.linalg.vector_norm(truth))
        over = float(torch.linalg.vector_norm(dev - truth) / torch.linalg.vector_norm(truth)) / floor
        near = float(torch.linalg.vector_norm(dev - fq) / torch.linalg.vector_norm(truth)) / floor
        peak = torch.linalg.vector_norm(truth.reshape(truth.shape[0], -1), dim=1).max()
        row_floor = float(torch.linalg.vector_norm((fq - truth).reshape(truth.shape[0], -1), dim=1).max() / peak)
        row_over = float(torch.linalg.vector_norm((dev - truth).reshape(truth.shape[0], -1), dim=1).max() / peak) / row_floor
        row_near = float(torch.linalg.vector_norm((dev - fq).reshape(truth.shape[0], -1), dim=1).max() / peak) / row_floor
        ok = over <= 1.15 and row_over <= 1.25 and near <= 0.50 and row_near <= 0.75
        detail = (f"    {name}: vs truth {over:.3f}x floor, worst row {row_over:.3f}x; "
                  f"vs fake quant {near:.3f}x, worst row {row_near:.3f}x")
        return ok, "" if ok else detail

    return compare


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--weights", type=str, default=None,
                        help="a real layer from weights.py; default: random weights")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--packed", action="store_true", default=False)
    parser.add_argument("--row-starts", type=str, default="0,256")
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the block over this many rounds (device only)")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % 2048:
        parser.error("--seq-len must be a multiple of 2048")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    for t in cases:
        if args.packed:
            starts = tuple(int(value) for value in args.row_starts.split(","))
            state_rows = tuple(2 * owner + 1 for owner in range(len(starts)))
            print(f"[case] gdn_block packed T={t} starts={starts} resume={args.resume}",
                  flush=True)
            result = run(
                fn=build_packed_kernel(t=t, rows=len(starts)),
                specs=build_packed_tensor_specs(t=t, starts=starts, state_rows=state_rows,
                                                pool_rows=2 * len(starts) + 1,
                                                weights=args.weights, resume=args.resume),
                golden_fn=None if args.compile_only else golden_gdn_block_packed,
                config=run_config(args.backend_type, args.execution_mode, args.device,
                                  args.enable_chip_swimlane),
                compare_fn={"out": compare_packed_out,
                            "conv_state": compare_packed_cache("conv_state", state_rows),
                            "final_state": compare_packed_cache("final_state", state_rows)},
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(f"FAILED at packed GDN block T={t}")
            continue
        print(f"[case] gdn_block T={t}", flush=True)
        result = run(
            fn=build_kernel(t=t),
            specs=build_tensor_specs(t=t, weights=args.weights, resume=args.resume),
            golden_fn=golden_gdn_block(args.weights, resume=args.resume),
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=run_config(args.backend_type, args.execution_mode, args.device, args.enable_chip_swimlane),
            rtol=1e-2,
            atol=1e-5,
            compare_fn={"out": compare_out(t, args.weights, args.resume),
                        "conv_state": compare_cache("conv_state", t, args.weights, args.resume),
                        "final_state": compare_cache("final_state", t, args.weights, args.resume)},
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run
            # lane, so every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
