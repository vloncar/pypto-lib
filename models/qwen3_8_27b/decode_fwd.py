# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8 decoder layer groups over stacked weights and shared hybrid caches."""
import pypto.language as pl

import decode_delta_rule
import decode_kv_append
import decode_layer
import decode_short_conv
from config import QWEN3_8_27B
from decode_vector import check_batch, padded_batch

# model config
C = QWEN3_8_27B.hidden_size
INTER = QWEN3_8_27B.intermediate_size
LAYERS = QWEN3_8_27B.num_hidden_layers
INTERVAL = QWEN3_8_27B.full_attention_interval
GROUPS = LAYERS // INTERVAL
GDN_PER_GROUP = INTERVAL - 1
CQKV = decode_layer.CQKV
CV = decode_layer.CV
H = decode_layer.H
D = decode_layer.D
KCONV = decode_layer.KCONV
QKV_WIDTH = decode_layer.QKV_WIDTH
ATT_WIDTH = decode_layer.ATT_WIDTH
KV_WIDTH = decode_layer.KV_WIDTH
HEAD_DIM = decode_layer.HEAD_DIM
ROT = decode_layer.ROT
PAGE = decode_layer.PAGE

# tiling
SEED_ROW_TILE = 4
ACTIVE_HEAD_TILE = 2


def build_kernel(batch: int = 16, max_pages: int = 32, groups: int = GROUPS,
                 groups_per_call: int | None = None, inline: bool = False, active_rows: int = 0):
    """Run consecutive LLLA groups starting at the supplied group index."""
    if not 1 <= groups <= GROUPS:
        raise ValueError(f"groups must be in [1, {GROUPS}]")
    count = groups if groups_per_call is None else groups_per_call
    if not 1 <= count <= groups:
        raise ValueError("groups_per_call must be within the stacked group count")
    m = padded_batch(batch)
    check_batch(m)
    layers = groups * INTERVAL
    gdn_layers = groups * GDN_PER_GROUP
    gdn_options = {"inline": True, "active_rows": active_rows}
    if active_rows:
        gdn_options["head_tile"] = ACTIVE_HEAD_TILE
    linear_0 = decode_layer.build_gdn_kernel(m, **gdn_options)
    linear_1 = decode_layer.build_gdn_kernel(m, **gdn_options)
    linear_2 = decode_layer.build_gdn_kernel(m, **gdn_options)
    full = decode_layer.build_attn_kernel(m, max_pages, inline=True)
    jit = pl.jit.inline if inline else pl.jit

    @jit(auto_scope=False)
    def qwen3_decode_fwd(
        residual: pl.Tensor[[m, C], pl.FP32],
        delta: pl.Tensor[[m, C], pl.BF16],
        norm_in_w: pl.Tensor[[layers * C], pl.BF16],
        norm_post_w: pl.Tensor[[layers * C], pl.BF16],
        w_gate: pl.Tensor[[layers * INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[layers * INTER], pl.FP32],
        w_up: pl.Tensor[[layers * INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[layers * INTER], pl.FP32],
        w_down: pl.Tensor[[layers * C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[layers * C], pl.FP32],
        w_qkv: pl.Tensor[[gdn_layers * CQKV, C], pl.INT8],
        w_qkv_scale: pl.Tensor[[gdn_layers * CQKV], pl.FP32],
        w_z: pl.Tensor[[gdn_layers * CV, C], pl.INT8],
        w_z_scale: pl.Tensor[[gdn_layers * CV], pl.FP32],
        w_ab: pl.Tensor[[gdn_layers * 2 * H, C], pl.BF16],
        conv_w: pl.Tensor[[gdn_layers * KCONV, CQKV], pl.FP32],
        a_log: pl.Tensor[[gdn_layers * H], pl.BF16],
        dt_bias: pl.Tensor[[gdn_layers * H], pl.BF16],
        norm_w: pl.Tensor[[gdn_layers * D], pl.BF16],
        w_out: pl.Tensor[[gdn_layers * C, CV], pl.BF16],
        decode_w_qkv: pl.Tensor[[groups * QKV_WIDTH, C], pl.INT8],
        decode_w_qkv_scale: pl.Tensor[[groups * QKV_WIDTH], pl.FP32],
        w_o: pl.Tensor[[groups * C, ATT_WIDTH], pl.INT8],
        w_o_scale: pl.Tensor[[groups * C], pl.FP32],
        q_norm_w: pl.Tensor[[groups * HEAD_DIM], pl.BF16],
        k_norm_w: pl.Tensor[[groups * HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[m, ROT], pl.FP32],
        rope_sin: pl.Tensor[[m, ROT], pl.FP32],
        state_rows: pl.Tensor[[gdn_layers * m], pl.INT32],
        block_tables: pl.Tensor[[groups * m, max_pages], pl.INT32],
        seq_lens: pl.Tensor[[m], pl.INT32],
        tail_mask: pl.Tensor[[PAGE + 1, PAGE], pl.FP32],
        conv_state: pl.InOut[pl.Tensor[[decode_short_conv.CSTATE_DYN, KCONV - 1, CQKV], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[decode_delta_rule.DSTATE_DYN, H, D, D], pl.FP32]],
        k_pages: pl.InOut[pl.Tensor[[decode_kv_append.KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[decode_kv_append.KVROWS_DYN, KV_WIDTH], pl.BF16]],
        residual_out: pl.Out[pl.Tensor[[m, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[m, C], pl.BF16]],
        group_start: pl.Scalar[pl.INDEX],
    ):
        conv_state.bind_dynamic(0, decode_short_conv.CSTATE_DYN)
        final_state.bind_dynamic(0, decode_delta_rule.DSTATE_DYN)
        k_pages.bind_dynamic(0, decode_kv_append.KVROWS_DYN)
        v_pages.bind_dynamic(0, decode_kv_append.KVROWS_DYN)
        for row in pl.spmd(m // SEED_ROW_TILE, name_hint="decode_fwd_seed"):
            start = row * SEED_ROW_TILE
            residual_out[start : start + SEED_ROW_TILE, 0:C] = residual[start : start + SEED_ROW_TILE, 0:C]
            mlp_out[start : start + SEED_ROW_TILE, 0:C] = delta[start : start + SEED_ROW_TILE, 0:C]

        residual_b = pl.create_tensor([m, C], dtype=pl.FP32)
        delta_b = pl.create_tensor([m, C], dtype=pl.BF16)
        # Explicit sizes preserve child JIT metadata for dynamic layer offsets.
        for group in pl.range(group_start, group_start + count):
            gdn_base = group * GDN_PER_GROUP
            layer_base = group * INTERVAL

            norm_in_w_0 = pl.slice(norm_in_w, [C], [(layer_base + 0) * C])
            norm_post_w_0 = pl.slice(norm_post_w, [C], [(layer_base + 0) * C])
            w_gate_0 = pl.slice(w_gate, [INTER, C], [(layer_base + 0) * INTER, 0])
            w_gate_scale_0 = pl.slice(w_gate_scale, [INTER], [(layer_base + 0) * INTER])
            w_up_0 = pl.slice(w_up, [INTER, C], [(layer_base + 0) * INTER, 0])
            w_up_scale_0 = pl.slice(w_up_scale, [INTER], [(layer_base + 0) * INTER])
            w_down_0 = pl.slice(w_down, [C, INTER], [(layer_base + 0) * C, 0])
            w_down_scale_0 = pl.slice(w_down_scale, [C], [(layer_base + 0) * C])
            w_qkv_0 = pl.slice(w_qkv, [CQKV, C], [(gdn_base + 0) * CQKV, 0])
            w_qkv_scale_0 = pl.slice(w_qkv_scale, [CQKV], [(gdn_base + 0) * CQKV])
            w_z_0 = pl.slice(w_z, [CV, C], [(gdn_base + 0) * CV, 0])
            w_z_scale_0 = pl.slice(w_z_scale, [CV], [(gdn_base + 0) * CV])
            w_ab_0 = pl.slice(w_ab, [2 * H, C], [(gdn_base + 0) * (2 * H), 0])
            conv_w_0 = pl.slice(conv_w, [KCONV, CQKV], [(gdn_base + 0) * KCONV, 0])
            a_log_0 = pl.slice(a_log, [H], [(gdn_base + 0) * H])
            dt_bias_0 = pl.slice(dt_bias, [H], [(gdn_base + 0) * H])
            norm_w_0 = pl.slice(norm_w, [D], [(gdn_base + 0) * D])
            w_out_0 = pl.slice(w_out, [C, CV], [(gdn_base + 0) * C, 0])
            state_rows_0 = pl.slice(state_rows, [m], [(gdn_base + 0) * m])
            with pl.scope():
                linear_0(
                    residual_out, mlp_out, norm_in_w_0, norm_post_w_0,
                    w_qkv_0, w_qkv_scale_0, w_z_0, w_z_scale_0, w_ab_0, conv_w_0,
                    a_log_0, dt_bias_0, norm_w_0, w_out_0, state_rows_0, conv_state, final_state,
                    w_gate_0, w_gate_scale_0, w_up_0, w_up_scale_0, w_down_0, w_down_scale_0,
                    residual_b, delta_b,
                )

            norm_in_w_1 = pl.slice(norm_in_w, [C], [(layer_base + 1) * C])
            norm_post_w_1 = pl.slice(norm_post_w, [C], [(layer_base + 1) * C])
            w_gate_1 = pl.slice(w_gate, [INTER, C], [(layer_base + 1) * INTER, 0])
            w_gate_scale_1 = pl.slice(w_gate_scale, [INTER], [(layer_base + 1) * INTER])
            w_up_1 = pl.slice(w_up, [INTER, C], [(layer_base + 1) * INTER, 0])
            w_up_scale_1 = pl.slice(w_up_scale, [INTER], [(layer_base + 1) * INTER])
            w_down_1 = pl.slice(w_down, [C, INTER], [(layer_base + 1) * C, 0])
            w_down_scale_1 = pl.slice(w_down_scale, [C], [(layer_base + 1) * C])
            w_qkv_1 = pl.slice(w_qkv, [CQKV, C], [(gdn_base + 1) * CQKV, 0])
            w_qkv_scale_1 = pl.slice(w_qkv_scale, [CQKV], [(gdn_base + 1) * CQKV])
            w_z_1 = pl.slice(w_z, [CV, C], [(gdn_base + 1) * CV, 0])
            w_z_scale_1 = pl.slice(w_z_scale, [CV], [(gdn_base + 1) * CV])
            w_ab_1 = pl.slice(w_ab, [2 * H, C], [(gdn_base + 1) * (2 * H), 0])
            conv_w_1 = pl.slice(conv_w, [KCONV, CQKV], [(gdn_base + 1) * KCONV, 0])
            a_log_1 = pl.slice(a_log, [H], [(gdn_base + 1) * H])
            dt_bias_1 = pl.slice(dt_bias, [H], [(gdn_base + 1) * H])
            norm_w_1 = pl.slice(norm_w, [D], [(gdn_base + 1) * D])
            w_out_1 = pl.slice(w_out, [C, CV], [(gdn_base + 1) * C, 0])
            state_rows_1 = pl.slice(state_rows, [m], [(gdn_base + 1) * m])
            with pl.scope():
                linear_1(
                    residual_b, delta_b, norm_in_w_1, norm_post_w_1,
                    w_qkv_1, w_qkv_scale_1, w_z_1, w_z_scale_1, w_ab_1, conv_w_1,
                    a_log_1, dt_bias_1, norm_w_1, w_out_1, state_rows_1, conv_state, final_state,
                    w_gate_1, w_gate_scale_1, w_up_1, w_up_scale_1, w_down_1, w_down_scale_1,
                    residual_out, mlp_out,
                )

            norm_in_w_2 = pl.slice(norm_in_w, [C], [(layer_base + 2) * C])
            norm_post_w_2 = pl.slice(norm_post_w, [C], [(layer_base + 2) * C])
            w_gate_2 = pl.slice(w_gate, [INTER, C], [(layer_base + 2) * INTER, 0])
            w_gate_scale_2 = pl.slice(w_gate_scale, [INTER], [(layer_base + 2) * INTER])
            w_up_2 = pl.slice(w_up, [INTER, C], [(layer_base + 2) * INTER, 0])
            w_up_scale_2 = pl.slice(w_up_scale, [INTER], [(layer_base + 2) * INTER])
            w_down_2 = pl.slice(w_down, [C, INTER], [(layer_base + 2) * C, 0])
            w_down_scale_2 = pl.slice(w_down_scale, [C], [(layer_base + 2) * C])
            w_qkv_2 = pl.slice(w_qkv, [CQKV, C], [(gdn_base + 2) * CQKV, 0])
            w_qkv_scale_2 = pl.slice(w_qkv_scale, [CQKV], [(gdn_base + 2) * CQKV])
            w_z_2 = pl.slice(w_z, [CV, C], [(gdn_base + 2) * CV, 0])
            w_z_scale_2 = pl.slice(w_z_scale, [CV], [(gdn_base + 2) * CV])
            w_ab_2 = pl.slice(w_ab, [2 * H, C], [(gdn_base + 2) * (2 * H), 0])
            conv_w_2 = pl.slice(conv_w, [KCONV, CQKV], [(gdn_base + 2) * KCONV, 0])
            a_log_2 = pl.slice(a_log, [H], [(gdn_base + 2) * H])
            dt_bias_2 = pl.slice(dt_bias, [H], [(gdn_base + 2) * H])
            norm_w_2 = pl.slice(norm_w, [D], [(gdn_base + 2) * D])
            w_out_2 = pl.slice(w_out, [C, CV], [(gdn_base + 2) * C, 0])
            state_rows_2 = pl.slice(state_rows, [m], [(gdn_base + 2) * m])
            with pl.scope():
                linear_2(
                    residual_out, mlp_out, norm_in_w_2, norm_post_w_2,
                    w_qkv_2, w_qkv_scale_2, w_z_2, w_z_scale_2, w_ab_2, conv_w_2,
                    a_log_2, dt_bias_2, norm_w_2, w_out_2, state_rows_2, conv_state, final_state,
                    w_gate_2, w_gate_scale_2, w_up_2, w_up_scale_2, w_down_2, w_down_scale_2,
                    residual_b, delta_b,
                )

            norm_in_w_3 = pl.slice(norm_in_w, [C], [(layer_base + 3) * C])
            norm_post_w_3 = pl.slice(norm_post_w, [C], [(layer_base + 3) * C])
            w_gate_3 = pl.slice(w_gate, [INTER, C], [(layer_base + 3) * INTER, 0])
            w_gate_scale_3 = pl.slice(w_gate_scale, [INTER], [(layer_base + 3) * INTER])
            w_up_3 = pl.slice(w_up, [INTER, C], [(layer_base + 3) * INTER, 0])
            w_up_scale_3 = pl.slice(w_up_scale, [INTER], [(layer_base + 3) * INTER])
            w_down_3 = pl.slice(w_down, [C, INTER], [(layer_base + 3) * C, 0])
            w_down_scale_3 = pl.slice(w_down_scale, [C], [(layer_base + 3) * C])
            decode_w_qkv_3 = pl.slice(decode_w_qkv, [QKV_WIDTH, C], [group * QKV_WIDTH, 0])
            decode_w_qkv_scale_3 = pl.slice(decode_w_qkv_scale, [QKV_WIDTH], [group * QKV_WIDTH])
            w_o_3 = pl.slice(w_o, [C, ATT_WIDTH], [group * C, 0])
            w_o_scale_3 = pl.slice(w_o_scale, [C], [group * C])
            q_norm_w_3 = pl.slice(q_norm_w, [HEAD_DIM], [group * HEAD_DIM])
            k_norm_w_3 = pl.slice(k_norm_w, [HEAD_DIM], [group * HEAD_DIM])
            pages = pl.slice(block_tables, [m, max_pages], [group * m, 0])
            with pl.scope():
                full(
                    residual_b, delta_b, norm_in_w_3, norm_post_w_3,
                    decode_w_qkv_3, decode_w_qkv_scale_3, q_norm_w_3, k_norm_w_3, rope_cos, rope_sin,
                    pages, seq_lens, tail_mask, w_o_3, w_o_scale_3, k_pages, v_pages,
                    w_gate_3, w_gate_scale_3, w_up_3, w_up_scale_3, w_down_3, w_down_scale_3,
                    residual_out, mlp_out,
                )
        return residual_out, mlp_out, conv_state, final_state, k_pages, v_pages

    return qwen3_decode_fwd


def build_l3_kernel(batch: int, *, max_pages: int, kv_rows: int, state_count: int,
                    groups: int = GROUPS, active_rows: int = 0):
    """Build a HOST decoder entry over stacked weights and resident hybrid caches."""
    m = padded_batch(batch)
    check_batch(m)
    if max_pages < 1 or kv_rows < groups * max_pages * PAGE or state_count < groups * GDN_PER_GROUP:
        raise ValueError("decode cache extents must cover every layer")
    layers = groups * INTERVAL
    gdn_layers = groups * GDN_PER_GROUP
    states = state_count
    kernel = build_kernel(m, max_pages, groups=groups, active_rows=active_rows)

    @pl.jit.host
    def l3_decode_fwd(
        residual: pl.Tensor[[m, C], pl.FP32],
        delta: pl.Tensor[[m, C], pl.BF16],
        norm_in_w: pl.Tensor[[layers * C], pl.BF16],
        norm_post_w: pl.Tensor[[layers * C], pl.BF16],
        w_gate: pl.Tensor[[layers * INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[layers * INTER], pl.FP32],
        w_up: pl.Tensor[[layers * INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[layers * INTER], pl.FP32],
        w_down: pl.Tensor[[layers * C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[layers * C], pl.FP32],
        w_qkv: pl.Tensor[[gdn_layers * CQKV, C], pl.INT8],
        w_qkv_scale: pl.Tensor[[gdn_layers * CQKV], pl.FP32],
        w_z: pl.Tensor[[gdn_layers * CV, C], pl.INT8],
        w_z_scale: pl.Tensor[[gdn_layers * CV], pl.FP32],
        w_ab: pl.Tensor[[gdn_layers * 2 * H, C], pl.BF16],
        conv_w: pl.Tensor[[gdn_layers * KCONV, CQKV], pl.FP32],
        a_log: pl.Tensor[[gdn_layers * H], pl.BF16],
        dt_bias: pl.Tensor[[gdn_layers * H], pl.BF16],
        norm_w: pl.Tensor[[gdn_layers * D], pl.BF16],
        w_out: pl.Tensor[[gdn_layers * C, CV], pl.BF16],
        decode_w_qkv: pl.Tensor[[groups * QKV_WIDTH, C], pl.INT8],
        decode_w_qkv_scale: pl.Tensor[[groups * QKV_WIDTH], pl.FP32],
        w_o: pl.Tensor[[groups * C, ATT_WIDTH], pl.INT8],
        w_o_scale: pl.Tensor[[groups * C], pl.FP32],
        q_norm_w: pl.Tensor[[groups * HEAD_DIM], pl.BF16],
        k_norm_w: pl.Tensor[[groups * HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[m, ROT], pl.FP32],
        rope_sin: pl.Tensor[[m, ROT], pl.FP32],
        state_rows: pl.Tensor[[gdn_layers * m], pl.INT32],
        block_tables: pl.Tensor[[groups * m, max_pages], pl.INT32],
        seq_lens: pl.Tensor[[m], pl.INT32],
        tail_mask: pl.Tensor[[PAGE + 1, PAGE], pl.FP32],
        conv_state: pl.InOut[pl.Tensor[[states, KCONV - 1, CQKV], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[states, H, D, D], pl.FP32]],
        k_pages: pl.InOut[pl.Tensor[[kv_rows, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[kv_rows, KV_WIDTH], pl.BF16]],
        residual_out: pl.Out[pl.Tensor[[m, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[m, C], pl.BF16]],
        group_start: pl.Scalar[pl.INDEX],
    ):
        return kernel(
            residual, delta, norm_in_w, norm_post_w,
            w_gate, w_gate_scale, w_up, w_up_scale,
            w_down, w_down_scale, w_qkv, w_qkv_scale,
            w_z, w_z_scale, w_ab, conv_w,
            a_log, dt_bias, norm_w, w_out,
            decode_w_qkv, decode_w_qkv_scale, w_o, w_o_scale,
            q_norm_w, k_norm_w, rope_cos, rope_sin,
            state_rows, block_tables, seq_lens, tail_mask,
            conv_state, final_state, k_pages, v_pages,
            residual_out, mlp_out, group_start,
        )

    return l3_decode_fwd
