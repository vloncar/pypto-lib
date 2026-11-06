# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Qwen3.8-27B's decoder layer over a prefill chunk, in both of the model's two forms.

    residual, norm_in  = add_norm(residual, delta, input_layernorm)
    mixer_out          = mixer(norm_in)              Gated DeltaNet, or gated attention
    residual, norm_mlp = add_norm(residual, mixer_out, post_attention_layernorm)
    mlp_out            = mlp(norm_mlp)               gate and up, SwiGLU, down

`gdn_block`'s fourteen operators or `gated_attention`'s eight, the MLP's four, and the two
fused add-and-norm stages around them, as one program.

**The layer boundary sits one addition behind the module boundary**, which is the shift
`decode_layer.py` uses and for the same reason: `rmsnorm.build_residual_kernel` computes an
addition and the norm that reads it as one stage, and a decoder layer's two additions each
pair with a norm. So `delta` is the previous layer's MLP output, `residual` is the stream it
belongs to, and the layer ends with its own MLP output still unadded -- which is what the
next layer takes as its `delta`. A chunk's first layer takes a zero residual and the
embedding as its delta; after the last layer the stream still carries one unadded output,
which the model's final norm consumes.

**The running residual is FP32 and everything else bf16.**
`rmsnorm.build_residual_kernel` carries the arithmetic: bf16 has 8 mantissa bits and a
64-layer accumulation would spend half of them, while each delta is a matmul output added
once.

Static shape per build, as every operator here is: the token count must be a multiple of
the mixer's own granularity, 2048 for the GDN variant and 256 for the attention one.
"""
import pypto.language as pl

import gated_attention
import gdn_block
import mlp
import rmsnorm
import short_conv
from chunk_h import FSTATE_DYN
from chunk_cumsum import CHUNK_TILE
from config import KV_PAGE_SIZE, QWEN3_8_27B
from prefill_kv_write import KVROWS_DYN

# model config
C = QWEN3_8_27B.hidden_size                 # the residual stream
CQKV = QWEN3_8_27B.qkv_width
CV = QWEN3_8_27B.value_width
KCONV = QWEN3_8_27B.linear_conv_kernel_dim
H = QWEN3_8_27B.linear_num_value_heads
HG = QWEN3_8_27B.linear_num_key_heads
DV = QWEN3_8_27B.linear_value_head_dim
INTER = QWEN3_8_27B.intermediate_size
HEAD_DIM = QWEN3_8_27B.head_dim
ROT = QWEN3_8_27B.rotary_dim
QG_WIDTH = gated_attention.QG_WIDTH
KV_WIDTH = gated_attention.KV_WIDTH
O_WIDTH = gated_attention.O_WIDTH

KINDS = ("gdn", "attn")
# validation
# Each variant runs the shapes its own mixer is gated on: the GDN layer's floor is
# `in_proj_z`'s 2048-token patch, the attention layer's `attn_core`'s 256-token KV tile.
# The MLP's 128 and the norm's 64 divide both.
DEFAULT_CASES = {"gdn": gdn_block.DEFAULT_CASES, "attn": gated_attention.DEFAULT_CASES}
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# Bytes per ring, one entry per scope depth, 0 leaving a ring at its default. The stage
# scopes allocate on ring 1, and the mixer sets the size there: `gdn_block` measured 4 GiB
# for its own intermediates, which are the layer's largest.
LAYER_RING_HEAP = [0, 4 * 1024 * 1024 * 1024, 0, 0]

# A random norm weight is drawn at the checkpoint's own scale, the module being `1 + w`:
# layer 0's two have standard deviations of 0.042 and 0.037.
NORM_W_SCALE = 0.05


def run_config(backend_type: str = "a2a3", sim: bool = False, device: int = 0,
               swimlane: int = 0) -> dict:
    """The config `golden.run` needs for a layer. The ring heap is not optional."""
    return dict(backend_type=backend_type, execution_mode=sim, device_id=device,
                ring_heap=LAYER_RING_HEAP, enable_chip_swimlane=swimlane)


def build_gdn_kernel(t: int = BENCH_SEQ, chunk: int = CHUNK_TILE, inline: bool = False):
    """A linear-attention layer at one shape, composing its stages as inlined callees."""
    norm_in = rmsnorm.build_residual_kernel(inline=True)
    norm_post = rmsnorm.build_residual_kernel(inline=True)
    mixer = gdn_block.build_kernel(t=t, chunk=chunk, inline=True)
    op_mlp = mlp.build_kernel(t=t, inline=True)

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_prefill_gdn_layer(
        residual: pl.Tensor[[t, C], pl.FP32],
        delta: pl.Tensor[[t, C], pl.BF16],
        norm_in_w: pl.Tensor[[C], pl.BF16],
        norm_post_w: pl.Tensor[[C], pl.BF16],
        w_qkv: pl.Tensor[[CQKV, C], pl.INT8],
        w_qkv_scale: pl.Tensor[[CQKV], pl.FP32],
        w_z: pl.Tensor[[CV, C], pl.INT8],
        w_z_scale: pl.Tensor[[CV], pl.FP32],
        w_ab: pl.Tensor[[2 * H, C], pl.BF16],
        conv_w: pl.Tensor[[KCONV, CQKV], pl.FP32],
        a_log: pl.Tensor[[H], pl.BF16],
        dt_bias: pl.Tensor[[H], pl.BF16],
        norm_w: pl.Tensor[[DV], pl.BF16],
        w_out: pl.Tensor[[C, CV], pl.BF16],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        mask_strict: pl.Tensor[[chunk, chunk], pl.FP32],
        eye: pl.Tensor[[chunk, chunk], pl.FP16],
        m_diag: pl.Tensor[[chunk, chunk], pl.FP16],
        m_low: pl.Tensor[[chunk, chunk], pl.FP16],
        row_base: pl.Tensor[[t], pl.INT32],
        state_row: pl.Scalar[pl.INDEX],
        conv_state: pl.InOut[pl.Tensor[[short_conv.CSTATE_DYN, KCONV - 1, CQKV], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, DV, DV], pl.FP32]],
        w_gate: pl.Tensor[[INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[INTER], pl.FP32],
        w_up: pl.Tensor[[INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[INTER], pl.FP32],
        w_down: pl.Tensor[[C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[C], pl.FP32],
        residual_out: pl.Out[pl.Tensor[[t, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[t, C], pl.BF16]],
    ):
        """`delta` is the previous layer's MLP output; `residual_out` is this layer's stream."""
        conv_state.bind_dynamic(0, short_conv.CSTATE_DYN)
        final_state.bind_dynamic(0, FSTATE_DYN)
        resid_mixer = pl.create_tensor([t, C], dtype=pl.FP32)
        mixer_in = pl.create_tensor([t, C], dtype=pl.BF16)
        norm_in(residual, delta, norm_in_w, resid_mixer, mixer_in)

        mixer_out = pl.create_tensor([t, C], dtype=pl.BF16)
        mixer(mixer_in, w_qkv, w_qkv_scale, w_z, w_z_scale, w_ab, conv_w, a_log, dt_bias,
              norm_w, w_out, tril, mask_strict, eye, m_diag, m_low,
              row_base, state_row, conv_state, final_state, mixer_out)

        mlp_in = pl.create_tensor([t, C], dtype=pl.BF16)
        norm_post(resid_mixer, mixer_out, norm_post_w, residual_out, mlp_in)

        op_mlp(mlp_in, w_gate, w_gate_scale, w_up, w_up_scale, w_down, w_down_scale, mlp_out)
        return residual_out, mlp_out, conv_state, final_state

    return qwen3_prefill_gdn_layer


def build_attn_kernel(t: int = BENCH_SEQ, inline: bool = False,
                      max_pages: int | None = None):
    """A full-attention layer at one shape, composing its stages as inlined callees."""
    norm_in = rmsnorm.build_residual_kernel(inline=True)
    norm_post = rmsnorm.build_residual_kernel(inline=True)
    max_pages = max_pages if max_pages is not None else t // KV_PAGE_SIZE
    mixer = gated_attention.build_kernel(t=t, inline=True, max_pages=max_pages)
    op_mlp = mlp.build_kernel(t=t, inline=True)
    bias_width = 2 * gated_attention.attn_core.KV_TILE - gated_attention.attn_core.ROW_TILE

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_prefill_attn_layer(
        residual: pl.Tensor[[t, C], pl.FP32],
        delta: pl.Tensor[[t, C], pl.BF16],
        norm_in_w: pl.Tensor[[C], pl.BF16],
        norm_post_w: pl.Tensor[[C], pl.BF16],
        w_q: pl.Tensor[[QG_WIDTH, C], pl.INT8],
        w_q_scale: pl.Tensor[[QG_WIDTH], pl.FP32],
        w_k: pl.Tensor[[KV_WIDTH, C], pl.INT8],
        w_k_scale: pl.Tensor[[KV_WIDTH], pl.FP32],
        w_v: pl.Tensor[[KV_WIDTH, C], pl.INT8],
        w_v_scale: pl.Tensor[[KV_WIDTH], pl.FP32],
        w_o: pl.Tensor[[C, O_WIDTH], pl.INT8],
        w_o_scale: pl.Tensor[[C], pl.FP32],
        q_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        k_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[t, ROT], pl.FP32],
        rope_sin: pl.Tensor[[t, ROT], pl.FP32],
        mask_bias: pl.Tensor[[gated_attention.attn_core.ROW_TILE, bias_width], pl.FP32],
        row_base: pl.Tensor[[t], pl.INT32],
        page_bases: pl.Tensor[[1], pl.INT32],
        block_table: pl.Tensor[[1, max_pages], pl.INT32],
        page_base: pl.Scalar[pl.INDEX],
        layer_base: pl.Scalar[pl.INDEX],
        k_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        w_gate: pl.Tensor[[INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[INTER], pl.FP32],
        w_up: pl.Tensor[[INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[INTER], pl.FP32],
        w_down: pl.Tensor[[C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[C], pl.FP32],
        residual_out: pl.Out[pl.Tensor[[t, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[t, C], pl.BF16]],
    ):
        """`delta` is the previous layer's MLP output; `residual_out` is this layer's stream."""
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        resid_mixer = pl.create_tensor([t, C], dtype=pl.FP32)
        mixer_in = pl.create_tensor([t, C], dtype=pl.BF16)
        norm_in(residual, delta, norm_in_w, resid_mixer, mixer_in)

        mixer_out = pl.create_tensor([t, C], dtype=pl.BF16)
        mixer(mixer_in, w_q, w_q_scale, w_k, w_k_scale, w_v, w_v_scale, w_o, w_o_scale,
              q_norm_w, k_norm_w, rope_cos, rope_sin, mask_bias, row_base, page_bases,
              block_table, page_base, layer_base, k_pages, v_pages, mixer_out)

        mlp_in = pl.create_tensor([t, C], dtype=pl.BF16)
        norm_post(resid_mixer, mixer_out, norm_post_w, residual_out, mlp_in)

        op_mlp(mlp_in, w_gate, w_gate_scale, w_up, w_up_scale, w_down, w_down_scale, mlp_out)
        return residual_out, mlp_out, k_pages, v_pages

    return qwen3_prefill_attn_layer


def build_kernel(kind: str, t: int = BENCH_SEQ, inline: bool = False, **kwargs):
    """Either variant at one shape."""
    if kind == "gdn":
        return build_gdn_kernel(t=t, inline=inline, **kwargs)
    if kind == "attn":
        return build_attn_kernel(t=t, inline=inline, **kwargs)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def build_packed_gdn_kernel(t: int, rows: int, chunk: int = CHUNK_TILE,
                            inline: bool = False):
    """Residual/MLP layer around the packed multi-owner GDN block."""
    norm_in = rmsnorm.build_residual_kernel(inline=True)
    norm_post = rmsnorm.build_residual_kernel(inline=True)
    mixer = gdn_block.build_packed_kernel(t=t, rows=rows, chunk=chunk, inline=True)
    op_mlp = mlp.build_kernel(t=t, inline=True)

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_prefill_gdn_layer_packed(
        residual: pl.Tensor[[t, C], pl.FP32],
        delta: pl.Tensor[[t, C], pl.BF16],
        norm_in_w: pl.Tensor[[C], pl.BF16],
        norm_post_w: pl.Tensor[[C], pl.BF16],
        w_qkv: pl.Tensor[[CQKV, C], pl.INT8],
        w_qkv_scale: pl.Tensor[[CQKV], pl.FP32],
        w_z: pl.Tensor[[CV, C], pl.INT8],
        w_z_scale: pl.Tensor[[CV], pl.FP32],
        w_ab: pl.Tensor[[2 * H, C], pl.BF16],
        conv_w: pl.Tensor[[KCONV, CQKV], pl.FP32],
        a_log: pl.Tensor[[H], pl.BF16],
        dt_bias: pl.Tensor[[H], pl.BF16],
        norm_w: pl.Tensor[[DV], pl.BF16],
        w_out: pl.Tensor[[C, CV], pl.BF16],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        mask_strict: pl.Tensor[[chunk, chunk], pl.FP32],
        eye: pl.Tensor[[chunk, chunk], pl.FP16],
        m_diag: pl.Tensor[[chunk, chunk], pl.FP16],
        m_low: pl.Tensor[[chunk, chunk], pl.FP16],
        row_base: pl.Tensor[[t], pl.INT32],
        row_owner: pl.Tensor[[t], pl.INT32],
        row_ends: pl.Tensor[[rows], pl.INT32],
        state_rows: pl.Tensor[[rows], pl.INT32],
        conv_state: pl.InOut[pl.Tensor[[short_conv.CSTATE_DYN, KCONV - 1, CQKV], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, DV, DV], pl.FP32]],
        w_gate: pl.Tensor[[INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[INTER], pl.FP32],
        w_up: pl.Tensor[[INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[INTER], pl.FP32],
        w_down: pl.Tensor[[C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[C], pl.FP32],
        residual_out: pl.Out[pl.Tensor[[t, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[t, C], pl.BF16]],
    ):
        conv_state.bind_dynamic(0, short_conv.CSTATE_DYN)
        final_state.bind_dynamic(0, FSTATE_DYN)
        resid_mixer = pl.create_tensor([t, C], dtype=pl.FP32)
        mixer_in = pl.create_tensor([t, C], dtype=pl.BF16)
        norm_in(residual, delta, norm_in_w, resid_mixer, mixer_in)
        mixer_out = pl.create_tensor([t, C], dtype=pl.BF16)
        mixer(mixer_in, w_qkv, w_qkv_scale, w_z, w_z_scale, w_ab, conv_w, a_log, dt_bias,
              norm_w, w_out, tril, mask_strict, eye, m_diag, m_low, row_base, row_owner,
              row_ends, state_rows, conv_state, final_state, mixer_out)
        mlp_in = pl.create_tensor([t, C], dtype=pl.BF16)
        norm_post(resid_mixer, mixer_out, norm_post_w, residual_out, mlp_in)
        op_mlp(mlp_in, w_gate, w_gate_scale, w_up, w_up_scale, w_down, w_down_scale, mlp_out)
        return residual_out, mlp_out, conv_state, final_state

    return qwen3_prefill_gdn_layer_packed


def build_packed_attn_kernel(t: int, rows: int, max_pages: int,
                             inline: bool = False):
    """Residual/MLP layer around packed multi-owner full attention."""
    norm_in = rmsnorm.build_residual_kernel(inline=True)
    norm_post = rmsnorm.build_residual_kernel(inline=True)
    mixer = gated_attention.build_packed_kernel(t=t, rows=rows,
                                                 max_pages=max_pages, inline=True)
    op_mlp = mlp.build_kernel(t=t, inline=True)
    bias_width = 2 * gated_attention.attn_core.KV_TILE - gated_attention.attn_core.ROW_TILE

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_prefill_attn_layer_packed(
        residual: pl.Tensor[[t, C], pl.FP32],
        delta: pl.Tensor[[t, C], pl.BF16],
        norm_in_w: pl.Tensor[[C], pl.BF16],
        norm_post_w: pl.Tensor[[C], pl.BF16],
        w_q: pl.Tensor[[QG_WIDTH, C], pl.INT8],
        w_q_scale: pl.Tensor[[QG_WIDTH], pl.FP32],
        w_k: pl.Tensor[[KV_WIDTH, C], pl.INT8],
        w_k_scale: pl.Tensor[[KV_WIDTH], pl.FP32],
        w_v: pl.Tensor[[KV_WIDTH, C], pl.INT8],
        w_v_scale: pl.Tensor[[KV_WIDTH], pl.FP32],
        w_o: pl.Tensor[[C, O_WIDTH], pl.INT8],
        w_o_scale: pl.Tensor[[C], pl.FP32],
        q_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        k_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[t, ROT], pl.FP32],
        rope_sin: pl.Tensor[[t, ROT], pl.FP32],
        mask_bias: pl.Tensor[[gated_attention.attn_core.ROW_TILE, bias_width], pl.FP32],
        row_base: pl.Tensor[[t], pl.INT32],
        row_owner: pl.Tensor[[t], pl.INT32],
        page_bases: pl.Tensor[[rows], pl.INT32],
        visible_pages: pl.Tensor[[rows], pl.INT32],
        block_table: pl.Tensor[[rows, max_pages], pl.INT32],
        layer_base: pl.Scalar[pl.INDEX],
        k_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        w_gate: pl.Tensor[[INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[INTER], pl.FP32],
        w_up: pl.Tensor[[INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[INTER], pl.FP32],
        w_down: pl.Tensor[[C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[C], pl.FP32],
        residual_out: pl.Out[pl.Tensor[[t, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[t, C], pl.BF16]],
    ):
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        resid_mixer = pl.create_tensor([t, C], dtype=pl.FP32)
        mixer_in = pl.create_tensor([t, C], dtype=pl.BF16)
        norm_in(residual, delta, norm_in_w, resid_mixer, mixer_in)
        mixer_out = pl.create_tensor([t, C], dtype=pl.BF16)
        mixer(mixer_in, w_q, w_q_scale, w_k, w_k_scale, w_v, w_v_scale, w_o, w_o_scale,
              q_norm_w, k_norm_w, rope_cos, rope_sin, mask_bias, row_base, row_owner,
              page_bases, visible_pages, block_table, layer_base, k_pages, v_pages, mixer_out)
        mlp_in = pl.create_tensor([t, C], dtype=pl.BF16)
        norm_post(resid_mixer, mixer_out, norm_post_w, residual_out, mlp_in)
        op_mlp(mlp_in, w_gate, w_gate_scale, w_up, w_up_scale, w_down, w_down_scale, mlp_out)
        return residual_out, mlp_out, k_pages, v_pages

    return qwen3_prefill_attn_layer_packed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def build_tensor_specs(kind: str, t: int = BENCH_SEQ, weights: str | None = None,
                       mlp_weights: str | None = None, norm_weights: str | None = None,
                       chunk: int = CHUNK_TILE, seed: int = 7,
                       resume: bool = False, page_base: int = 0,
                       max_pages: int | None = None, packed: bool = False,
                       starts: tuple[int, ...] = (0, 256),
                       bases: tuple[int, ...] = (0, 4),
                       state_rows: tuple[int, ...] = (1, 3)):
    """The mixer's own fixtures and the MLP's, plus the four tensors the layer adds.

    Both spec sets are taken unchanged apart from dropping their `x` and `out`, so a layer
    run reads exactly the tensors its stages were gated on and the oracle reads them back
    out of the harness rather than redrawing them.

    `residual` and `delta` are drawn at the same scale. A real residual stream grows with
    depth and is several times its per-layer delta, which would put less of the mixer's
    error into `residual_out`; equal scales are the harder case for the gate.
    """
    import torch
    from golden import TensorSpec

    if kind == "gdn" and packed:
        mixer_specs = gdn_block.build_packed_tensor_specs(
            t=t, starts=starts, state_rows=state_rows,
            pool_rows=max(state_rows) + 2, weights=weights, resume=resume)
    elif kind == "attn" and packed:
        max_pages = max_pages if max_pages is not None else max(bases) + t // KV_PAGE_SIZE + 2
        mixer_specs = gated_attention.build_packed_tensor_specs(
            t=t, starts=starts, bases=bases, max_pages=max_pages, weights=weights)
    elif kind == "gdn":
        mixer_specs = gdn_block.build_tensor_specs(t=t, chunk=chunk, weights=weights,
                                                   resume=resume)
    elif kind == "attn":
        mixer_specs = gated_attention.build_tensor_specs(t=t, weights=weights,
                                                         page_base=page_base,
                                                         max_pages=max_pages)
    else:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    mlp_specs = mlp.build_tensor_specs(t=t, weights=mlp_weights)

    def inner(specs):
        return [spec for spec in specs if spec.name not in ("x", "out")]

    def draw(offset, dtype):
        def make():
            gen = torch.Generator().manual_seed(seed + offset)
            return torch.randn(t, C, generator=gen).to(dtype)
        return make

    def norm_param(name, offset):
        def make():
            if norm_weights is not None:
                return torch.load(norm_weights, weights_only=True)[name].to(torch.bfloat16)
            gen = torch.Generator().manual_seed(seed + offset)
            return (torch.randn(C, generator=gen) * NORM_W_SCALE).to(torch.bfloat16)
        return make

    return ([
        TensorSpec("residual", [t, C], torch.float32, init_value=draw(0, torch.float32)),
        TensorSpec("delta", [t, C], torch.bfloat16, init_value=draw(1, torch.bfloat16)),
        TensorSpec("norm_in_w", [C], torch.bfloat16,
                   init_value=norm_param("input_layernorm.weight", 2)),
        TensorSpec("norm_post_w", [C], torch.bfloat16,
                   init_value=norm_param("post_attention_layernorm.weight", 3)),
    ] + inner(mixer_specs) + inner(mlp_specs) + [
        TensorSpec("residual_out", [t, C], torch.float32),
        TensorSpec("mlp_out", [t, C], torch.bfloat16),
    ])


# ---------------------------------------------------------------------------
# Oracles and acceptance
# ---------------------------------------------------------------------------


def mixer_oracle(kind: str, tensors, t: int, quant: bool, chunk: int = CHUNK_TILE):
    """One mixer oracle closed over the harness's own tensors.

    **One weight dict serves both chains, and it is the kernel's own.** The projections are
    already int8 with their scales beside them, which is the form `reference.weight`
    dequantises, so *quant* switches the activation quantiser and nothing else -- the weight
    quantisation is held common and the floor the gates divide by is the activation path's
    cost alone. Re-quantising that dict would be the identity with a scale of 1.0, which
    silently rescales every projection by 1/scale.

    The GDN chain quantises the hidden states per token and nothing else, `out_proj` being
    bf16 both sides; the attention chain has two oracles and `quant` picks between them.
    """
    import reference

    if kind == "gdn":
        w = gdn_block.block_weights(tensors)
        row = int(tensors["state_row"])
        conv_in = tensors["conv_state"][row].clone()
        state_in = tensors["final_state"][row].transpose(-2, -1).contiguous().clone()

        def mixer(x):
            st = reference.block(x, w, QWEN3_8_27B, chunk, quant_x=quant, quant_y=False,
                                 conv_state=conv_in, state=state_in)
            conv_out = st["conv_state_out"]
            state_out = st["final_state"].transpose(-2, -1)
            if quant:
                tensors["conv_state"][row].copy_(conv_out)
                tensors["final_state"][row].copy_(state_out)
            else:
                record_truth(conv_state=conv_out, final_state=state_out)
            return st["out"]

        return mixer

    # `INPUT_NAMES` carries the mixer's own `x`, which a layer does not have: its input is
    # the normed stream the oracle is handed per call.
    w = {name: tensors[name] for name in gated_attention.INPUT_NAMES if name != "x"}
    step = reference.attn_w8a8 if quant else reference.attn_truth
    prefix_k, prefix_v = gated_attention.prefix_cache(tensors)
    page_base = int(tensors["page_base"])
    layer_base = int(tensors["layer_base"])

    def mixer(x):
        out, k_current, v_current = step(
            x, w, tensors["rope_cos"], tensors["rope_sin"], QWEN3_8_27B,
            prefix_k, prefix_v, return_cache=True)
        if quant:
            reference.write_kv_pages(tensors["k_pages"][layer_base:],
                                     tensors["v_pages"][layer_base:],
                                     tensors["block_table"], k_current, v_current,
                                     page_base * KV_PAGE_SIZE, KV_PAGE_SIZE)
        else:
            record_truth(k_pages=k_current, v_pages=v_current)
        return out

    return mixer


_TRUTH: dict[str, object] = {}


def record_truth(**outputs) -> None:
    """Hand the float64 truth to `compare_fns`, for a caller that chains layers itself.

    `golden_prefill_layer` fills this for one layer; a stacked caller runs its own two
    chains and needs the same gate, so the stash has one writer and several users rather
    than a second copy of the comparison. The keys are output names, so a caller whose
    outputs are not a layer's -- `prefill_text_fwd`'s `logits` -- uses the same stash and
    the same comparator.
    """
    _TRUTH.update(outputs)


def golden_prefill_layer(kind: str, t: int, chunk: int = CHUNK_TILE):
    """Both chains: the fake-quant one written out, the float64 truth stashed for the gates.

    The same weights serve both. They are the tensors the kernel reads, dequantised in
    float64 for the truth, so the floor the gates divide by is the cost of the activation
    path alone -- the per-token quantisers and the bf16 hand-offs -- with the weight
    quantisation held common, which is `mlp_truth`'s convention and `attn_truth`'s.
    """
    import torch

    import a8w8_linear
    import reference

    def golden(tensors):
        a8w8_linear.host_torch_setup()
        mlp_w = {name: tensors[name] for name in
                 ("w_gate", "w_gate_scale", "w_up", "w_up_scale", "w_down", "w_down_scale")}
        head = (tensors["residual"], tensors["delta"], tensors["norm_in_w"],
                tensors["norm_post_w"])

        truth = reference.residual_layer(*head, mixer_oracle(kind, tensors, t, False, chunk),
                                         mlp_w, QWEN3_8_27B, quant=False)
        record_truth(residual_out=truth["residual_out"], mlp_out=truth["mlp_out"])

        chain = reference.residual_layer(*head, mixer_oracle(kind, tensors, t, True, chunk),
                                         mlp_w, QWEN3_8_27B, quant=True)
        tensors["residual_out"].copy_(chain["residual_out"].to(torch.float32))
        tensors["mlp_out"].copy_(chain["mlp_out"].to(torch.bfloat16))

    return golden


def golden_prefill_layer_packed(kind: str, t: int):
    """Run each packed row as an independent layer with its own initial cache."""
    import torch

    import a8w8_linear
    import reference

    def golden(tensors):
        a8w8_linear.host_torch_setup()
        ends = tensors["row_ends"].tolist() if kind == "gdn" else None
        starts = ([0] + ends[:-1] if ends is not None else
                  sorted(set(tensors["row_base"].tolist())))
        if ends is None:
            ends = starts[1:] + [t]
        mlp_w = {name: tensors[name] for name in
                 ("w_gate", "w_gate_scale", "w_up", "w_up_scale", "w_down", "w_down_scale")}
        truth_resid = torch.empty((t, C), dtype=torch.float64)
        truth_delta = torch.empty((t, C), dtype=torch.float64)
        truth_cache = {"conv_state": {}, "final_state": {}, "k_pages": {}, "v_pages": {}}

        for owner, (start, end) in enumerate(zip(starts, ends)):
            head = (tensors["residual"][start:end], tensors["delta"][start:end],
                    tensors["norm_in_w"], tensors["norm_post_w"])
            if kind == "gdn":
                row = int(tensors["state_rows"][owner])
                conv_in = tensors["conv_state"][row].clone()
                final_in = tensors["final_state"][row].clone()
                weights = gdn_block.block_weights(tensors)

                def make_mixer(quant):
                    def mixer(x):
                        st = reference.block(x, weights, QWEN3_8_27B, CHUNK_TILE,
                                             quant_x=quant, quant_y=False,
                                             conv_state=conv_in,
                                             state=final_in.transpose(-2, -1))
                        if quant:
                            tensors["conv_state"][row].copy_(st["conv_state_out"])
                            tensors["final_state"][row].copy_(
                                st["final_state"].transpose(-2, -1))
                        else:
                            truth_cache["conv_state"][row] = st["conv_state_out"]
                            truth_cache["final_state"][row] = st["final_state"].transpose(-2, -1)
                        return st["out"]
                    return mixer
            else:
                weights = {name: tensors[name] for name in gated_attention.INPUT_NAMES
                           if name != "x"}
                page_base = int(tensors["page_bases"][owner]) * KV_PAGE_SIZE
                layer_base = int(tensors["layer_base"])
                table = tensors["block_table"]
                prefix_rows = (reference.kv_page_rows(table, owner, page_base, KV_PAGE_SIZE)
                               if page_base else None)
                prefix_k = (tensors["k_pages"][layer_base + prefix_rows].clone()
                            if prefix_rows is not None else None)
                prefix_v = (tensors["v_pages"][layer_base + prefix_rows].clone()
                            if prefix_rows is not None else None)

                def make_mixer(quant):
                    def mixer(x):
                        step = reference.attn_w8a8 if quant else reference.attn_truth
                        out, k_current, v_current = step(
                            x, weights, tensors["rope_cos"][start:end],
                            tensors["rope_sin"][start:end], QWEN3_8_27B,
                            prefix_k, prefix_v, return_cache=True)
                        if quant:
                            reference.write_kv_pages(
                                tensors["k_pages"][layer_base:],
                                tensors["v_pages"][layer_base:], table,
                                k_current, v_current, page_base, KV_PAGE_SIZE, seq=owner)
                        else:
                            truth_cache["k_pages"][owner] = k_current
                            truth_cache["v_pages"][owner] = v_current
                        return out
                    return mixer

            truth = reference.residual_layer(*head, make_mixer(False), mlp_w,
                                             QWEN3_8_27B, quant=False)
            chain = reference.residual_layer(*head, make_mixer(True), mlp_w,
                                             QWEN3_8_27B, quant=True)
            truth_resid[start:end] = truth["residual_out"]
            truth_delta[start:end] = truth["mlp_out"]
            tensors["residual_out"][start:end].copy_(chain["residual_out"].to(torch.float32))
            tensors["mlp_out"][start:end].copy_(chain["mlp_out"].to(torch.bfloat16))
        record_truth(residual_out=truth_resid, mlp_out=truth_delta, **truth_cache)

    return golden


def compare_packed_vs_floor(name: str, **limits):
    scored = compare_vs_floor(name, **limits)

    def compare(actual, expected, inputs=None, **kwargs):
        if inputs is None:
            return False, f"{name}: packed row boundaries are missing"
        starts = sorted(set(inputs["row_base"].tolist()))
        ends = starts[1:] + [actual.shape[0]]
        truth = _TRUTH[name]
        for owner, (start, end) in enumerate(zip(starts, ends)):
            _TRUTH[name] = truth[start:end]
            ok, detail = scored(actual[start:end], expected[start:end], **kwargs)
            if not ok:
                _TRUTH[name] = truth
                return False, f"owner {owner}: {detail}"
        _TRUTH[name] = truth
        return True, f"{name}: all {len(starts)} packed owners pass"

    return compare


def compare_packed_cache(name: str, kind: str, state_rows: tuple[int, ...]):
    fill = -99.0 if name == "conv_state" else -1.0

    def compare(actual, expected, inputs=None, **_kwargs):
        import torch

        import reference

        if inputs is None:
            return False, f"{name}: missing cache ownership descriptors"
        touched = torch.zeros(actual.shape[0], dtype=torch.bool)
        starts = sorted(set(inputs["row_base"].tolist()))
        ends = starts[1:] + [inputs["delta"].shape[0]]
        for owner, (start, end) in enumerate(zip(starts, ends)):
            if kind == "gdn":
                row = state_rows[owner]
                dev, fq = actual[row].double(), expected[row].double()
                truth = _TRUTH[name][row].double()
                touched[row] = True
            else:
                page_base = int(inputs["page_bases"][owner]) * KV_PAGE_SIZE
                rows = reference.kv_page_rows(
                    inputs["block_table"], owner, page_base + end - start,
                    KV_PAGE_SIZE)[page_base:]
                physical = int(inputs["layer_base"]) + rows
                touched[physical] = True
                dev, fq = actual[physical].double(), expected[physical].double()
                truth = _TRUTH[name][owner].double()
            den = torch.linalg.vector_norm(truth)
            floor = float(torch.linalg.vector_norm(fq - truth) / den)
            over = float(torch.linalg.vector_norm(dev - truth) / den) / floor
            near = float(torch.linalg.vector_norm(dev - fq) / den) / floor
            if not (over <= 1.15 and near <= 0.50):
                return False, (f"{name} owner {owner}: truth {over:.3f}x floor, "
                               f"fake-quant {near:.3f}x")
        if not bool((actual[~touched] == expected[~touched]).all()):
            return False, f"{name}: unowned cache rows changed"
        if kind == "gdn" and not bool((actual[~touched] == fill).all()):
            return False, f"{name}: unowned state rows lost their sentinel"
        return True, f"{name}: both owners pass; unowned rows untouched"

    return compare


def compare_packed_fns(kind: str, state_rows: tuple[int, ...]):
    result = {name: compare_packed_vs_floor(name) for name in ("residual_out", "mlp_out")}
    names = ("conv_state", "final_state") if kind == "gdn" else ("k_pages", "v_pages")
    result.update({name: compare_packed_cache(name, kind, state_rows) for name in names})
    return result


# Acceptance, as ratios against the quantisation floor rather than fixed numbers, which is
# the shape both mixers and the MLP use: every output here is downstream of an int8
# hand-off, so an element-wise bound would measure which side of a rounding boundary a
# value landed on. The bounds are `decode_layer`'s, for the reason recorded there: the two
# cannot both bind, and a kernel allowed one floor of distance from the fake-quant chain
# can reach sqrt(2) floors from the truth when the two errors are orthogonal. The binding
# gate is the distance to the chain the kernel implements.
MAX_OVER_FLOOR = 1.15
MAX_ROW_OVER_FLOOR = 1.25
MAX_VS_GOLDEN = 1.00
MAX_ROW_VS_GOLDEN = 1.50

# ...and a bound on the floor itself, because every gate above is a ratio against it and
# they all degenerate when the reference is wrong rather than the kernel: a fake-quant chain
# that has gone off by a factor puts the same garbage in the numerator and the denominator of
# `MAX_VS_GOLDEN`, which then reads 1.000 from either side. A per-token int8 hand-off costs a
# few percent, so a chain more than this far from the truth is a broken oracle and says so
# instead of passing. Measured floors here are 1e-02 to 5e-02.
MAX_FLOOR = 0.25


def compare_vs_floor(name: str, *, max_over_floor: float = MAX_OVER_FLOOR,
                     max_row_over_floor: float = MAX_ROW_OVER_FLOOR,
                     max_vs_golden: float = MAX_VS_GOLDEN,
                     max_row_vs_golden: float = MAX_ROW_VS_GOLDEN,
                     max_floor: float = MAX_FLOOR,
                     valid_rows_input: str | None = None):
    """One layer output against the float64 truth, beside what W8A8 alone costs.

    The bounds default to this file's, which are a single layer's. A caller that chains
    layers passes its own: both the distance to the fake-quant chain and the floor itself
    grow with depth, while the distance to the truth does not. See `prefill_fwd`.

    `valid_rows_input` names a runtime scalar delimiting the written output prefix.
    """
    def compare(actual, expected, inputs=None, **_kwargs):
        truth = _TRUTH[name].double()
        fq = expected.double()
        dev = actual.double()
        if valid_rows_input is not None:
            if inputs is None or valid_rows_input not in inputs:
                return False, f"{name}: missing valid-row input {valid_rows_input}"
            live = int(inputs[valid_rows_input])
            if not 0 < live <= min(dev.shape[0], fq.shape[0], truth.shape[0]):
                return False, f"{name}: invalid valid-row count {live}"
            dev, fq, truth = dev[:live], fq[:live], truth[:live]

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).max())

        floor, floor_row = rel(fq, truth), worst_row(fq, truth)
        if floor > max_floor:
            return False, (f"[stats] {name}: the fake-quant chain is {floor:.3e} from the "
                           f"float64 truth, past {max_floor} -- no activation quantiser "
                           f"costs that, so the oracle is wrong and the ratios below would "
                           f"read ~1.000 whatever the kernel did")
        got, got_row = rel(dev, truth), worst_row(dev, truth)
        vs_fq, vs_fq_row = rel(dev, fq), worst_row(dev, fq)
        ratio = got / floor if floor else float("inf")
        ratio_row = got_row / floor_row if floor_row else float("inf")
        fq_ratio = vs_fq / floor if floor else float("inf")
        fq_ratio_row = vs_fq_row / floor_row if floor_row else float("inf")
        line = (f"[stats] {name}: rel frob {got:.3e} vs the float64 truth against a "
                f"quantisation floor of {floor:.3e} ({ratio:.3f}x); worst token row "
                f"{got_row:.3e} against {floor_row:.3e} ({ratio_row:.3f}x); {vs_fq:.3e} vs "
                f"the fake-quant chain ({fq_ratio:.3f} of the floor, gate {max_vs_golden}; "
                f"worst row {fq_ratio_row:.3f})")
        # Printed whether or not it passes. These bounds are argued from measurements,
        # and a margin that is only visible on failure cannot be re-argued when a stage
        # changes without deliberately breaking the gate to read it.
        print(line, flush=True)
        ok = (ratio <= max_over_floor and ratio_row <= max_row_over_floor
              and fq_ratio <= max_vs_golden and fq_ratio_row <= max_row_vs_golden)
        return ok, "" if ok else line

    return compare


def compare_cache_vs_floor(name: str):
    """The cache row the next chunk reads, with unowned rows checked exactly."""
    fill = -99.0 if name == "conv_state" else -1.0

    def compare(actual, expected, **_kwargs):
        import torch

        if not bool((actual[[0, 1, 3]] == fill).all()):
            return False, f"[stats] {name}: unowned rows changed"
        dev, fq, truth = actual[2].double(), expected[2].double(), _TRUTH[name].double()
        den = torch.linalg.vector_norm(truth)
        floor = float(torch.linalg.vector_norm(fq - truth) / den)
        over = float(torch.linalg.vector_norm(dev - truth) / den) / floor
        near = float(torch.linalg.vector_norm(dev - fq) / den) / floor
        peak = torch.linalg.vector_norm(truth.reshape(truth.shape[0], -1), dim=1).max()
        row_floor = float(torch.linalg.vector_norm((fq - truth).reshape(truth.shape[0], -1), dim=1).max() / peak)
        row_over = float(torch.linalg.vector_norm((dev - truth).reshape(truth.shape[0], -1), dim=1).max() / peak) / row_floor
        row_near = float(torch.linalg.vector_norm((dev - fq).reshape(truth.shape[0], -1), dim=1).max() / peak) / row_floor
        line = (f"[stats] {name}: vs truth {over:.3f}x floor, worst row {row_over:.3f}x; "
                f"vs fake quant {near:.3f}x, worst row {row_near:.3f}x")
        print(line, flush=True)
        ok = over <= 1.15 and row_over <= 1.25 and near <= 0.5 and row_near <= 0.75
        return ok, "" if ok else line

    return compare


def compare_attn_cache(name: str):
    """Gate the attention cache rows and preserve every other physical page."""
    import torch

    import reference

    def compare(actual, expected, inputs=None, **_kwargs):
        if inputs is None:
            return False, f"{name}: block table is needed to check physical pages"
        start = int(inputs["page_base"]) * KV_PAGE_SIZE
        t = inputs["delta"].shape[0]
        base = int(inputs["layer_base"])
        rows = base + reference.kv_page_rows(inputs["block_table"], 0, start + t,
                                             KV_PAGE_SIZE)[start:]
        untouched = torch.ones(actual.shape[0], dtype=torch.bool)
        untouched[rows] = False
        if not bool((actual[untouched] == expected[untouched]).all()):
            return False, f"{name}: rows outside this chunk changed"
        dev, fq, truth = actual[rows].double(), expected[rows].double(), _TRUTH[name].double()
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


def compare_fns(kind: str = "attn") -> dict:
    """The layer outputs, and the two cache rows for a GDN layer."""
    result = {name: compare_vs_floor(name) for name in ("residual_out", "mlp_out")}
    if kind == "gdn":
        result.update({name: compare_cache_vs_floor(name) for name in ("conv_state", "final_state")})
    elif kind == "attn":
        result.update({name: compare_attn_cache(name) for name in ("k_pages", "v_pages")})
    return result


__all__ = ["BENCH_SEQ", "DEFAULT_CASES", "KINDS", "LAYER_RING_HEAP", "NORM_W_SCALE",
           "build_attn_kernel", "build_gdn_kernel", "build_kernel", "build_tensor_specs",
           "compare_fns", "golden_prefill_layer", "mixer_oracle", "record_truth",
           "run_config"]


if __name__ == "__main__":
    import argparse
    import os

    import weights
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--kind", choices=KINDS, default=None,
                        help="one variant at its own DEFAULT_CASES; default: both at their floor")
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="seed the GDN cache from a nonzero prior chunk")
    parser.add_argument("--page-base", type=int, default=0,
                        help="cached attention pages before this chunk; requires --kind attn")
    parser.add_argument("--packed", action="store_true", default=False)
    parser.add_argument("--row-starts", type=str, default="0,256")
    parser.add_argument("--row-page-bases", type=str, default="0,4")
    parser.add_argument("--layer", type=int, default=None,
                        help="a real layer's three weight dumps; default: random weights")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="where those dumps live, when not beside this checkout")
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the layer over this many rounds (device only)")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    if args.kind:
        plan = [(args.kind, t) for t in ([args.seq_len] if args.seq_len
                                         else DEFAULT_CASES[args.kind])]
    elif args.seq_len:
        parser.error("--seq-len carries one shape for one variant; name it with --kind")
    else:
        plan = [(kind, DEFAULT_CASES[kind][0]) for kind in KINDS]
    for kind, t in plan:
        floor = DEFAULT_CASES[kind][0]
        if t % floor:
            parser.error(f"--seq-len must be a multiple of {floor} for --kind {kind}")
    if len(plan) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one case; name it with --kind and --seq-len")
    if args.page_base and args.kind != "attn":
        parser.error("--page-base requires --kind attn")
    if args.page_base % (gated_attention.attn_core.KV_TILE // KV_PAGE_SIZE):
        parser.error("--page-base must align to the attention core's KV tile")
    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    for kind, t in plan:
        paths = ({} if args.layer is None
                 else weights.layer_paths(args.layer, kind, args.weights_dir))
        if args.packed:
            starts = tuple(int(value) for value in args.row_starts.split(","))
            bases = tuple(int(value) for value in args.row_page_bases.split(","))
            state_rows = tuple(2 * owner + 1 for owner in range(len(starts)))
            max_pages = max(bases) + t // KV_PAGE_SIZE + 2
            print(f"[case] prefill_layer packed {kind} T={t} starts={starts}", flush=True)
            result = run(
                fn=(build_packed_gdn_kernel(t=t, rows=len(starts)) if kind == "gdn" else
                    build_packed_attn_kernel(t=t, rows=len(starts), max_pages=max_pages)),
                specs=build_tensor_specs(kind, t=t, resume=args.resume, packed=True,
                                         starts=starts, bases=bases, state_rows=state_rows,
                                         max_pages=max_pages, **paths),
                golden_fn=None if args.compile_only else golden_prefill_layer_packed(kind, t),
                compare_fn=None if args.compile_only else compare_packed_fns(kind, state_rows),
                config=run_config(args.backend_type, args.execution_mode, args.device,
                                  args.enable_chip_swimlane),
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(f"FAILED at packed {kind} T={t}")
            continue
        print(f"[case] prefill_layer {kind} T={t}", flush=True)
        page_base = args.page_base if kind == "attn" else 0
        max_pages = page_base + t // KV_PAGE_SIZE + 2 if kind == "attn" else None
        result = run(
            fn=build_kernel(kind, t=t, **({"max_pages": max_pages} if kind == "attn" else {})),
            specs=build_tensor_specs(kind, t=t, resume=args.resume, page_base=page_base,
                                     max_pages=max_pages, **paths),
            golden_fn=golden_prefill_layer(kind, t),
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=run_config(args.backend_type, args.execution_mode, args.device,
                              args.enable_chip_swimlane),
            rtol=1e-2,
            atol=1e-5,
            compare_fn=compare_fns(kind),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at {kind} T={t}")
