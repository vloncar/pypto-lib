# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Qwen3.8-27B decoder stack over packed prefill requests and their cache pools."""
import pypto.language as pl

import gated_attention
import gdn_block
import prefill_layer
import short_conv
from chunk_h import FSTATE_DYN
from chunk_cumsum import CHUNK_TILE
from config import FULL_ATTENTION, KV_PAGE_SIZE, LINEAR_ATTENTION, QWEN3_8_27B
from prefill_kv_write import KVROWS_DYN

# model config
C = QWEN3_8_27B.hidden_size                 # the residual stream
CQKV = QWEN3_8_27B.qkv_width
CV = QWEN3_8_27B.value_width
KCONV = QWEN3_8_27B.linear_conv_kernel_dim
H = QWEN3_8_27B.linear_num_value_heads
DV = QWEN3_8_27B.linear_value_head_dim
INTER = QWEN3_8_27B.intermediate_size
HEAD_DIM = QWEN3_8_27B.head_dim
ROT = QWEN3_8_27B.rotary_dim
QG_WIDTH = gated_attention.QG_WIDTH
KV_WIDTH = gated_attention.KV_WIDTH
O_WIDTH = gated_attention.O_WIDTH
LAYERS = QWEN3_8_27B.num_hidden_layers              # 64
INTERVAL = QWEN3_8_27B.full_attention_interval      # one full-attention layer per four
GROUPS = LAYERS // INTERVAL                         # 16
LINEAR_PER_GROUP = INTERVAL - 1                     # 3
ROW_TILE = gated_attention.attn_core.ROW_TILE
KV_TILE = gated_attention.attn_core.KV_TILE
BIAS_WIDTH = 2 * KV_TILE - ROW_TILE

# validation
# One group is the floor: four layers, both mixers, and the group boundary the ping-pong
# has to return to. Two groups is the first case where a layer reads a stack row no
# earlier layer touched. The token count follows the GDN mixer's 2048-token patch.
DEFAULT_GROUPS = (1, 2)
DEFAULT_SEQ = gdn_block.DEFAULT_CASES[0]
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len
SEED = 7

# Ring 0 holds the residual pair; ring 1 holds one layer's live intermediates.
RING_PAIR_BYTES = C * (4 + 2)       # ring 0: one token's worth of the scratch pair
RING_CHUNK_BYTES = 1024 * 1024      # ring 1: one token's worth of a layer's live set
RING_FLOOR = 1024                   # rings 2 and 3: nothing is charged to them
FWD_RING_HEAP = [256 * 1024 * 1024, 4 * 1024 * 1024 * 1024, RING_FLOOR, RING_FLOOR]


def fwd_ring_heap(t: int = DEFAULT_SEQ) -> list:
    """`FWD_RING_HEAP` with rings 0 and 1 scaled to the chunk, each rounded up to a power
    of two -- the runtime refuses any other size."""
    def at_least(want: int, floor: int) -> int:
        return max(floor, 1 << (max(1, want) - 1).bit_length())

    return [at_least(t * RING_PAIR_BYTES, FWD_RING_HEAP[0]),
            at_least(t * RING_CHUNK_BYTES, FWD_RING_HEAP[1]),
            RING_FLOOR, RING_FLOOR]

# tiling
SEED_ROWS = 4           # token rows per work item of the seed copy: 4 FP32 rows fit UB
SEED_BLOCKS = 48        # one per vector core


def run_config(backend_type: str = "a2a3", sim: bool = False, device: int = 0,
               swimlane: int = 0, ring_heap: list | None = None,
               task_window: list | None = None, dep_pool: list | None = None,
               log_level: str | None = None) -> dict:
    """The config `golden.run` needs for the stack. The ring heap is not optional.

    A ring carries three independent limits -- bytes, in-flight tasks and dependency edges
    -- and a program that exceeds any of them fails the same way, as a bare
    `507018 ACL_ERROR_RT_AICPU_EXCEPTION` with nothing device-classified in the host log.
    All three are settable here so a failing shape can be told apart from a defect.
    """
    cfg = dict(backend_type=backend_type, execution_mode=sim, device_id=device,
               ring_heap=list(FWD_RING_HEAP if ring_heap is None else ring_heap),
               enable_chip_swimlane=swimlane)
    if task_window is not None:
        cfg["ring_task_window"] = list(task_window)
    if dep_pool is not None:
        cfg["ring_dep_pool"] = list(dep_pool)
    if log_level is not None:
        cfg["log_level"] = log_level
    return cfg


def layer_plan(groups: int = GROUPS) -> list[tuple[str, int, int]]:
    """The schedule a build walks: `(mixer, global index, type-local index)` per layer.

    Taken from `config.layer_index` rather than written out again, so the kernel's weight
    offsets and the oracle's weight slices come from the one schedule the checkpoint was
    validated against.
    """
    plan = []
    for layer in range(groups * INTERVAL):
        mixer, local = QWEN3_8_27B.layer_index(layer)
        plan.append((mixer, layer, local))
    return plan


def build_kernel(t: int = BENCH_SEQ, groups: int = GROUPS, chunk: int = CHUNK_TILE,
                 inline: bool = False, max_pages: int | None = None,
                 kv_layer_stride: int | None = None, rows: int = 1):
    """The stack at one shape. `groups` is how many `LLLA` groups the loop runs."""
    if not 1 <= groups <= GROUPS:
        raise ValueError(f"groups={groups} must be in [1, {GROUPS}]")
    if rows < 1 or t < rows * KV_TILE:
        raise ValueError("each packed request needs at least one complete KV tile")
    if t % SEED_ROWS:
        raise ValueError(f"t={t} must be a multiple of SEED_ROWS={SEED_ROWS}")
    layers = groups * INTERVAL
    gdn_layers = groups * LINEAR_PER_GROUP
    max_pages = max_pages if max_pages is not None else t // KV_PAGE_SIZE
    kv_layer_stride = (kv_layer_stride if kv_layer_stride is not None
                       else max_pages * KV_PAGE_SIZE)
    if kv_layer_stride % KV_PAGE_SIZE or kv_layer_stride < max_pages * KV_PAGE_SIZE:
        raise ValueError("KV layer stride must hold all physical pages and be page aligned")
    # One build per call site: an inlined layer is spliced where it is called, and the
    # three linear slots differ only in which rows of the stacks they read.
    linear_0 = prefill_layer.build_packed_gdn_kernel(t=t, rows=rows, chunk=chunk, inline=True)
    linear_1 = prefill_layer.build_packed_gdn_kernel(t=t, rows=rows, chunk=chunk, inline=True)
    linear_2 = prefill_layer.build_packed_gdn_kernel(t=t, rows=rows, chunk=chunk, inline=True)
    full = prefill_layer.build_packed_attn_kernel(t=t, rows=rows, max_pages=max_pages, inline=True)

    # A scope is a reclaim boundary and `ring_idx = min(scope_depth, 3)`. Left to the
    # compiler, one AUTO scope spans the whole group, so four layers' live set is retained
    # until the group ends -- which is what the device log calls a provable head-of-line
    # deadlock when it does not fit, and what puts the chunk ceiling at 8192. `auto_scope`
    # is off here so each layer gets a scope of its own and releases four times a group;
    # the runtime's implicit top-level scope stands in for the function body.
    jit = pl.jit.inline if inline else pl.jit

    @jit(auto_scope=False)
    def qwen3_prefill_fwd(
        residual: pl.Tensor[[t, C], pl.FP32],
        delta: pl.Tensor[[t, C], pl.BF16],
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
        norm_w: pl.Tensor[[gdn_layers * DV], pl.BF16],
        w_out: pl.Tensor[[gdn_layers * C, CV], pl.BF16],
        w_q: pl.Tensor[[groups * QG_WIDTH, C], pl.INT8],
        w_q_scale: pl.Tensor[[groups * QG_WIDTH], pl.FP32],
        w_k: pl.Tensor[[groups * KV_WIDTH, C], pl.INT8],
        w_k_scale: pl.Tensor[[groups * KV_WIDTH], pl.FP32],
        w_v: pl.Tensor[[groups * KV_WIDTH, C], pl.INT8],
        w_v_scale: pl.Tensor[[groups * KV_WIDTH], pl.FP32],
        w_o: pl.Tensor[[groups * C, O_WIDTH], pl.INT8],
        w_o_scale: pl.Tensor[[groups * C], pl.FP32],
        q_norm_w: pl.Tensor[[groups * HEAD_DIM], pl.BF16],
        k_norm_w: pl.Tensor[[groups * HEAD_DIM], pl.BF16],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        mask_strict: pl.Tensor[[chunk, chunk], pl.FP32],
        eye: pl.Tensor[[chunk, chunk], pl.FP16],
        m_diag: pl.Tensor[[chunk, chunk], pl.FP16],
        m_low: pl.Tensor[[chunk, chunk], pl.FP16],
        rope_cos: pl.Tensor[[t, ROT], pl.FP32],
        rope_sin: pl.Tensor[[t, ROT], pl.FP32],
        mask_bias: pl.Tensor[[ROW_TILE, BIAS_WIDTH], pl.FP32],
        row_base: pl.Tensor[[t], pl.INT32],
        row_owner: pl.Tensor[[t], pl.INT32],
        row_ends: pl.Tensor[[rows], pl.INT32],
        state_rows: pl.Tensor[[gdn_layers * rows], pl.INT32],
        page_bases: pl.Tensor[[rows], pl.INT32],
        visible_pages: pl.Tensor[[rows], pl.INT32],
        block_table: pl.Tensor[[rows, max_pages], pl.INT32],
        conv_state: pl.InOut[pl.Tensor[[short_conv.CSTATE_DYN, KCONV - 1, CQKV], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, DV, DV], pl.FP32]],
        k_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        residual_out: pl.Out[pl.Tensor[[t, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[t, C], pl.BF16]],
    ):
        """`delta` is the embedding for a chunk's first layer; `mlp_out` stays unadded."""
        conv_state.bind_dynamic(0, short_conv.CSTATE_DYN)
        final_state.bind_dynamic(0, FSTATE_DYN)
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        # The output pair is the side of the ping-pong every group starts and ends on, so
        # it is where the caller's input has to be before the first layer reads it.
        for core in pl.spmd(SEED_BLOCKS, name_hint="prefill_fwd_seed"):
            for row in pl.range(core * SEED_ROWS, t, SEED_BLOCKS * SEED_ROWS):
                residual_out[row : row + SEED_ROWS, 0:C] = residual[row : row + SEED_ROWS, 0:C]
                mlp_out[row : row + SEED_ROWS, 0:C] = delta[row : row + SEED_ROWS, 0:C]

        resid_b = pl.create_tensor([t, C], dtype=pl.FP32)
        delta_b = pl.create_tensor([t, C], dtype=pl.BF16)

        for group in pl.range(groups):
            gdn_base = group * LINEAR_PER_GROUP     # this group's first linear layer
            glb_base = group * INTERVAL             # the same layer's global index

            # Linear attention, slot 0: the output pair into the scratch pair.
            norm_in_0 = pl.slice(norm_in_w, [C], [glb_base * C])
            norm_post_0 = pl.slice(norm_post_w, [C], [glb_base * C])
            w_qkv_0 = pl.slice(w_qkv, [CQKV, C], [gdn_base * CQKV, 0])
            w_qkv_scale_0 = pl.slice(w_qkv_scale, [CQKV], [gdn_base * CQKV])
            w_z_0 = pl.slice(w_z, [CV, C], [gdn_base * CV, 0])
            w_z_scale_0 = pl.slice(w_z_scale, [CV], [gdn_base * CV])
            w_ab_0 = pl.slice(w_ab, [2 * H, C], [gdn_base * 2 * H, 0])
            conv_w_0 = pl.slice(conv_w, [KCONV, CQKV], [gdn_base * KCONV, 0])
            a_log_0 = pl.slice(a_log, [H], [gdn_base * H])
            dt_bias_0 = pl.slice(dt_bias, [H], [gdn_base * H])
            norm_w_0 = pl.slice(norm_w, [DV], [gdn_base * DV])
            w_out_0 = pl.slice(w_out, [C, CV], [gdn_base * C, 0])
            w_gate_0 = pl.slice(w_gate, [INTER, C], [glb_base * INTER, 0])
            w_gate_scale_0 = pl.slice(w_gate_scale, [INTER], [glb_base * INTER])
            w_up_0 = pl.slice(w_up, [INTER, C], [glb_base * INTER, 0])
            w_up_scale_0 = pl.slice(w_up_scale, [INTER], [glb_base * INTER])
            w_down_0 = pl.slice(w_down, [C, INTER], [glb_base * C, 0])
            w_down_scale_0 = pl.slice(w_down_scale, [C], [glb_base * C])
            state_rows_0 = state_rows[gdn_base * rows : (gdn_base + 1) * rows]
            with pl.scope():
                linear_0(residual_out, mlp_out, norm_in_0, norm_post_0,
                         w_qkv_0, w_qkv_scale_0, w_z_0, w_z_scale_0, w_ab_0, conv_w_0,
                         a_log_0, dt_bias_0, norm_w_0, w_out_0,
                         tril, mask_strict, eye, m_diag, m_low, row_base, row_owner, row_ends,
                         state_rows_0, conv_state, final_state,
                         w_gate_0, w_gate_scale_0, w_up_0, w_up_scale_0, w_down_0, w_down_scale_0,
                         resid_b, delta_b)

            # Linear attention, slot 1: the scratch pair back into the output pair.
            norm_in_1 = pl.slice(norm_in_w, [C], [(glb_base + 1) * C])
            norm_post_1 = pl.slice(norm_post_w, [C], [(glb_base + 1) * C])
            w_qkv_1 = pl.slice(w_qkv, [CQKV, C], [(gdn_base + 1) * CQKV, 0])
            w_qkv_scale_1 = pl.slice(w_qkv_scale, [CQKV], [(gdn_base + 1) * CQKV])
            w_z_1 = pl.slice(w_z, [CV, C], [(gdn_base + 1) * CV, 0])
            w_z_scale_1 = pl.slice(w_z_scale, [CV], [(gdn_base + 1) * CV])
            w_ab_1 = pl.slice(w_ab, [2 * H, C], [(gdn_base + 1) * 2 * H, 0])
            conv_w_1 = pl.slice(conv_w, [KCONV, CQKV], [(gdn_base + 1) * KCONV, 0])
            a_log_1 = pl.slice(a_log, [H], [(gdn_base + 1) * H])
            dt_bias_1 = pl.slice(dt_bias, [H], [(gdn_base + 1) * H])
            norm_w_1 = pl.slice(norm_w, [DV], [(gdn_base + 1) * DV])
            w_out_1 = pl.slice(w_out, [C, CV], [(gdn_base + 1) * C, 0])
            w_gate_1 = pl.slice(w_gate, [INTER, C], [(glb_base + 1) * INTER, 0])
            w_gate_scale_1 = pl.slice(w_gate_scale, [INTER], [(glb_base + 1) * INTER])
            w_up_1 = pl.slice(w_up, [INTER, C], [(glb_base + 1) * INTER, 0])
            w_up_scale_1 = pl.slice(w_up_scale, [INTER], [(glb_base + 1) * INTER])
            w_down_1 = pl.slice(w_down, [C, INTER], [(glb_base + 1) * C, 0])
            w_down_scale_1 = pl.slice(w_down_scale, [C], [(glb_base + 1) * C])
            state_rows_1 = state_rows[(gdn_base + 1) * rows : (gdn_base + 2) * rows]
            with pl.scope():
                linear_1(resid_b, delta_b, norm_in_1, norm_post_1,
                         w_qkv_1, w_qkv_scale_1, w_z_1, w_z_scale_1, w_ab_1, conv_w_1,
                         a_log_1, dt_bias_1, norm_w_1, w_out_1,
                         tril, mask_strict, eye, m_diag, m_low, row_base, row_owner, row_ends,
                         state_rows_1, conv_state, final_state,
                         w_gate_1, w_gate_scale_1, w_up_1, w_up_scale_1, w_down_1, w_down_scale_1,
                         residual_out, mlp_out)

            # Linear attention, slot 2: the output pair into the scratch pair again.
            norm_in_2 = pl.slice(norm_in_w, [C], [(glb_base + 2) * C])
            norm_post_2 = pl.slice(norm_post_w, [C], [(glb_base + 2) * C])
            w_qkv_2 = pl.slice(w_qkv, [CQKV, C], [(gdn_base + 2) * CQKV, 0])
            w_qkv_scale_2 = pl.slice(w_qkv_scale, [CQKV], [(gdn_base + 2) * CQKV])
            w_z_2 = pl.slice(w_z, [CV, C], [(gdn_base + 2) * CV, 0])
            w_z_scale_2 = pl.slice(w_z_scale, [CV], [(gdn_base + 2) * CV])
            w_ab_2 = pl.slice(w_ab, [2 * H, C], [(gdn_base + 2) * 2 * H, 0])
            conv_w_2 = pl.slice(conv_w, [KCONV, CQKV], [(gdn_base + 2) * KCONV, 0])
            a_log_2 = pl.slice(a_log, [H], [(gdn_base + 2) * H])
            dt_bias_2 = pl.slice(dt_bias, [H], [(gdn_base + 2) * H])
            norm_w_2 = pl.slice(norm_w, [DV], [(gdn_base + 2) * DV])
            w_out_2 = pl.slice(w_out, [C, CV], [(gdn_base + 2) * C, 0])
            w_gate_2 = pl.slice(w_gate, [INTER, C], [(glb_base + 2) * INTER, 0])
            w_gate_scale_2 = pl.slice(w_gate_scale, [INTER], [(glb_base + 2) * INTER])
            w_up_2 = pl.slice(w_up, [INTER, C], [(glb_base + 2) * INTER, 0])
            w_up_scale_2 = pl.slice(w_up_scale, [INTER], [(glb_base + 2) * INTER])
            w_down_2 = pl.slice(w_down, [C, INTER], [(glb_base + 2) * C, 0])
            w_down_scale_2 = pl.slice(w_down_scale, [C], [(glb_base + 2) * C])
            state_rows_2 = state_rows[(gdn_base + 2) * rows : (gdn_base + 3) * rows]
            with pl.scope():
                linear_2(residual_out, mlp_out, norm_in_2, norm_post_2,
                         w_qkv_2, w_qkv_scale_2, w_z_2, w_z_scale_2, w_ab_2, conv_w_2,
                         a_log_2, dt_bias_2, norm_w_2, w_out_2,
                         tril, mask_strict, eye, m_diag, m_low, row_base, row_owner, row_ends,
                         state_rows_2, conv_state, final_state,
                         w_gate_2, w_gate_scale_2, w_up_2, w_up_scale_2, w_down_2, w_down_scale_2,
                         resid_b, delta_b)

            # Full attention, the group's fourth layer: back into the output pair, which
            # is where the next group expects to find the stream.
            norm_in_a = pl.slice(norm_in_w, [C], [(glb_base + 3) * C])
            norm_post_a = pl.slice(norm_post_w, [C], [(glb_base + 3) * C])
            w_q_a = pl.slice(w_q, [QG_WIDTH, C], [group * QG_WIDTH, 0])
            w_q_scale_a = pl.slice(w_q_scale, [QG_WIDTH], [group * QG_WIDTH])
            w_k_a = pl.slice(w_k, [KV_WIDTH, C], [group * KV_WIDTH, 0])
            w_k_scale_a = pl.slice(w_k_scale, [KV_WIDTH], [group * KV_WIDTH])
            w_v_a = pl.slice(w_v, [KV_WIDTH, C], [group * KV_WIDTH, 0])
            w_v_scale_a = pl.slice(w_v_scale, [KV_WIDTH], [group * KV_WIDTH])
            w_o_a = pl.slice(w_o, [C, O_WIDTH], [group * C, 0])
            w_o_scale_a = pl.slice(w_o_scale, [C], [group * C])
            q_norm_w_a = pl.slice(q_norm_w, [HEAD_DIM], [group * HEAD_DIM])
            k_norm_w_a = pl.slice(k_norm_w, [HEAD_DIM], [group * HEAD_DIM])
            w_gate_a = pl.slice(w_gate, [INTER, C], [(glb_base + 3) * INTER, 0])
            w_gate_scale_a = pl.slice(w_gate_scale, [INTER], [(glb_base + 3) * INTER])
            w_up_a = pl.slice(w_up, [INTER, C], [(glb_base + 3) * INTER, 0])
            w_up_scale_a = pl.slice(w_up_scale, [INTER], [(glb_base + 3) * INTER])
            w_down_a = pl.slice(w_down, [C, INTER], [(glb_base + 3) * C, 0])
            w_down_scale_a = pl.slice(w_down_scale, [C], [(glb_base + 3) * C])
            with pl.scope():
                full(resid_b, delta_b, norm_in_a, norm_post_a,
                     w_q_a, w_q_scale_a, w_k_a, w_k_scale_a, w_v_a, w_v_scale_a,
                     w_o_a, w_o_scale_a, q_norm_w_a, k_norm_w_a, rope_cos, rope_sin, mask_bias, row_base, row_owner,
                     page_bases, visible_pages, block_table, group * kv_layer_stride,
                     k_pages, v_pages,
                     w_gate_a, w_gate_scale_a, w_up_a, w_up_scale_a, w_down_a, w_down_scale_a,
                     residual_out, mlp_out)
        return residual_out, mlp_out, conv_state, final_state, k_pages, v_pages

    return qwen3_prefill_fwd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The stacking convention, once: per-layer rows on axis 0, the tail dims, the dtype, and
# which of the two indices addresses the stack. `build_tensor_specs` reads it for the
# shapes and `unstack` for the row offsets, so the kernel's offsets and the oracle's
# slices cannot drift apart.
GLOBAL = "global"


def stack_layout():
    """`(name, rows per layer, tail dims, dtype, index)` for every stacked weight."""
    import torch

    return (
        ("norm_in_w", C, [], torch.bfloat16, GLOBAL),
        ("norm_post_w", C, [], torch.bfloat16, GLOBAL),
        ("w_gate", INTER, [C], torch.int8, GLOBAL),
        ("w_gate_scale", INTER, [], torch.float32, GLOBAL),
        ("w_up", INTER, [C], torch.int8, GLOBAL),
        ("w_up_scale", INTER, [], torch.float32, GLOBAL),
        ("w_down", C, [INTER], torch.int8, GLOBAL),
        ("w_down_scale", C, [], torch.float32, GLOBAL),
        ("w_qkv", CQKV, [C], torch.int8, LINEAR_ATTENTION),
        ("w_qkv_scale", CQKV, [], torch.float32, LINEAR_ATTENTION),
        ("w_z", CV, [C], torch.int8, LINEAR_ATTENTION),
        ("w_z_scale", CV, [], torch.float32, LINEAR_ATTENTION),
        ("w_ab", 2 * H, [C], torch.bfloat16, LINEAR_ATTENTION),
        ("conv_w", KCONV, [CQKV], torch.float32, LINEAR_ATTENTION),
        ("a_log", H, [], torch.bfloat16, LINEAR_ATTENTION),
        ("dt_bias", H, [], torch.bfloat16, LINEAR_ATTENTION),
        ("norm_w", DV, [], torch.bfloat16, LINEAR_ATTENTION),
        ("w_out", C, [CV], torch.bfloat16, LINEAR_ATTENTION),
        ("w_q", QG_WIDTH, [C], torch.int8, FULL_ATTENTION),
        ("w_q_scale", QG_WIDTH, [], torch.float32, FULL_ATTENTION),
        ("w_k", KV_WIDTH, [C], torch.int8, FULL_ATTENTION),
        ("w_k_scale", KV_WIDTH, [], torch.float32, FULL_ATTENTION),
        ("w_v", KV_WIDTH, [C], torch.int8, FULL_ATTENTION),
        ("w_v_scale", KV_WIDTH, [], torch.float32, FULL_ATTENTION),
        ("w_o", C, [O_WIDTH], torch.int8, FULL_ATTENTION),
        ("w_o_scale", C, [], torch.float32, FULL_ATTENTION),
        ("q_norm_w", HEAD_DIM, [], torch.bfloat16, FULL_ATTENTION),
        ("k_norm_w", HEAD_DIM, [], torch.bfloat16, FULL_ATTENTION),
    )


def stack_index(index: str, mixer: str, glb: int, local: int) -> int | None:
    """Which row block of a stack one layer reads, or None when the stack is not its own."""
    if index == GLOBAL:
        return glb
    return local if index == mixer else None


def unstack(tensors: dict, mixer: str, glb: int, local: int) -> dict:
    """One layer's view of the stacked host tensors, keyed as its own operators name them.

    The oracle reads the weights **out of the tensors the device was given**, not from a
    second draw: a layer's slice here and the kernel's `pl.slice` have to name the same
    rows, and that is what the gate checks.
    """
    layer = {}
    for name, rows, _tail, _dtype, index in stack_layout():
        at = stack_index(index, mixer, glb, local)
        if at is not None:
            layer[name] = tensors[name][at * rows : (at + 1) * rows]
    return layer


def layer_fixture(mixer: str, glb: int, local: int, t: int, real: bool,
                  weights_dir: str | None, seed: int) -> dict:
    """Every per-layer tensor one layer reads, drawn per layer or loaded from a dump.

    The draws are the stages' own -- `reference.make_block_weights`, `attn_inputs` and
    `mlp_inputs` -- offset by the layer's index so no two layers share a weight. That is
    what makes a wrong stack offset visible: with one weight set repeated, reading the
    wrong rows would pass.

    *real* loads each layer's own extracted dump instead, by global index, so a real run
    is the model's own first layers rather than one layer repeated. That needs a dump per
    layer in the plan, which in practice makes `--real` a one-group check: `weights.py
    --fetch` writes them one layer and one group at a time.
    """
    import torch

    import reference

    layer = {}
    paths = {}
    if real:
        import weights as weights_mod
        kind = "gdn" if mixer == LINEAR_ATTENTION else "attn"
        paths = weights_mod.layer_paths(glb, kind, weights_dir)

    for name in ("w_gate", "w_gate_scale", "w_up", "w_up_scale", "w_down", "w_down_scale"):
        layer[name] = reference.mlp_inputs(name, t, C, INTER, seed=seed + glb,
                                           weights=paths.get("mlp_weights"))
    if real:
        norms = torch.load(paths["norm_weights"], weights_only=True)
        layer["norm_in_w"] = norms["input_layernorm.weight"].to(torch.bfloat16)
        layer["norm_post_w"] = norms["post_attention_layernorm.weight"].to(torch.bfloat16)
    else:
        for name, offset in (("norm_in_w", 2), ("norm_post_w", 3)):
            gen = torch.Generator().manual_seed(seed + glb * INTERVAL + offset)
            draw = torch.randn(C, generator=gen) * prefill_layer.NORM_W_SCALE
            layer[name] = draw.to(torch.bfloat16)

    if mixer == LINEAR_ATTENTION:
        if real:
            w = torch.load(paths["weights"], weights_only=True)
        else:
            w = reference.make_block_weights(QWEN3_8_27B, seed=seed + local)
        for name, module in (("w_qkv", "in_proj_qkv"), ("w_z", "in_proj_z")):
            q, scale = reference.quantize_rows(w[module + ".weight"])
            layer[name], layer[name + "_scale"] = q, scale.float()
        layer["w_ab"] = torch.cat([w["in_proj_a.weight"], w["in_proj_b.weight"]]).to(torch.bfloat16)
        layer["conv_w"] = w["conv1d.weight"].reshape(CQKV, KCONV).t().contiguous().float()
        layer["a_log"] = w["A_log"].to(torch.bfloat16)
        layer["dt_bias"] = w["dt_bias"].to(torch.bfloat16)
        layer["norm_w"] = w["norm.weight"].to(torch.bfloat16)
        layer["w_out"] = w["out_proj.weight"].to(torch.bfloat16)
    else:
        for name in ("w_q", "w_q_scale", "w_k", "w_k_scale", "w_v", "w_v_scale",
                     "w_o", "w_o_scale", "q_norm_w", "k_norm_w"):
            layer[name] = reference.attn_inputs(name, t, QWEN3_8_27B,
                                                weights=paths.get("weights"), seed=seed + local)
    return layer


_STACKS: dict[tuple, dict] = {}


def stacks(t: int, groups: int, real: bool, weights_dir: str | None, seed: int) -> dict:
    """Every stacked weight, built in one pass over the layers and cached.

    One pass because a `TensorSpec` is drawn on demand and there are 28 of them: built per
    spec, each of the 64 layers would be assembled 28 times. The peak is the stacks plus
    one layer, not two copies of the stacks, so a stack is filled band by band rather than
    concatenated.
    """
    import torch

    key = (t, groups, real, weights_dir, seed)
    held = _STACKS.get(key)
    if held is not None:
        return held
    plan = layer_plan(groups)
    counts = {GLOBAL: groups * INTERVAL, LINEAR_ATTENTION: groups * LINEAR_PER_GROUP,
              FULL_ATTENTION: groups}
    built = {name: torch.empty([counts[index] * rows] + tail, dtype=dtype)
             for name, rows, tail, dtype, index in stack_layout()}
    rows_of = {name: rows for name, rows, _t, _d, _i in stack_layout()}
    index_of = {name: index for name, _r, _t, _d, index in stack_layout()}
    for mixer, glb, local in plan:
        layer = layer_fixture(mixer, glb, local, t, real, weights_dir, seed)
        for name, piece in layer.items():
            at = stack_index(index_of[name], mixer, glb, local)
            if at is None:
                raise ValueError(f"layer {glb} ({mixer}) produced {name}, which is not its own")
            rows = rows_of[name]
            built[name][at * rows : (at + 1) * rows] = piece
    _STACKS.clear()
    _STACKS[key] = built
    return built


def build_tensor_specs(t: int = BENCH_SEQ, groups: int = GROUPS, chunk: int = CHUNK_TILE,
                       real: bool = False, weights_dir: str | None = None, seed: int = SEED,
                       page_base: int = 0, max_pages: int | None = None,
                       kv_layer_stride: int | None = None, resume: bool = False,
                       starts: tuple[int, ...] = (0,), bases: tuple[int, ...] | None = None):
    """The stacked weights, the shared tables, the chunk's two inputs and the two outputs.

    `residual` and `delta` are drawn at the same scale, as a layer's fixtures are: a real
    residual stream grows with depth and is several times its per-layer delta, and equal
    scales put more of the stack's error into `residual_out`.
    """
    import torch
    from golden import TensorSpec

    import reference
    import solve_tril

    bases = bases if bases is not None else (page_base,) * len(starts)
    ends = starts[1:] + (t,)
    if (not starts or starts[0] != 0 or len(starts) != len(bases)
            or t % DEFAULT_SEQ or any(start % KV_TILE for start in starts)
            or any(end - start < KV_TILE for start, end in zip(starts, ends))
            or any(base < 0 or base % (KV_TILE // KV_PAGE_SIZE) for base in bases)):
        raise ValueError("packed stack needs complete aligned requests and aligned cached prefixes")
    visible = [base + (end - start) // KV_PAGE_SIZE for base, start, end in zip(bases, starts, ends)]
    max_pages = max_pages if max_pages is not None else max(visible) + 2
    if max_pages < max(visible):
        raise ValueError("page table cannot hold a request's prefix and current tokens")
    rows = len(starts)
    first_phys = 3
    minimum_stride = (rows * max_pages + first_phys) * KV_PAGE_SIZE
    kv_layer_stride = kv_layer_stride if kv_layer_stride is not None else minimum_stride
    if kv_layer_stride < minimum_stride or kv_layer_stride % KV_PAGE_SIZE:
        raise ValueError("KV layer stride cannot hold the fixture's physical pages")
    gen = torch.Generator().manual_seed(seed + 101)
    block_table = (torch.randperm(rows * max_pages, generator=gen) + first_phys).reshape(rows, -1).int()
    k_pool = torch.randn(groups * kv_layer_stride, KV_WIDTH, generator=gen).to(torch.bfloat16)
    v_pool = torch.randn(groups * kv_layer_stride, KV_WIDTH, generator=gen).to(torch.bfloat16)
    owned = groups * LINEAR_PER_GROUP * rows
    state_rows = 2 * torch.randperm(owned, generator=gen).int() + 1
    conv_pool = torch.full((2 * owned + 1, KCONV - 1, CQKV), -99.0, dtype=torch.bfloat16)
    state_pool = torch.full((2 * owned + 1, H, DV, DV), -1.0, dtype=torch.float32)
    for offset, state_row in enumerate(state_rows.tolist()):
        owner = offset % rows
        if resume and (rows == 1 or bases[owner] > 0):
            conv_pool[state_row] = (torch.randn(KCONV - 1, CQKV, generator=gen) * 0.1).to(torch.bfloat16)
            state_pool[state_row] = torch.randn(H, DV, DV, generator=gen) * 0.02
        else:
            conv_pool[state_row] = 0.0
            state_pool[state_row] = 0.0
    row_base = torch.empty(t, dtype=torch.int32)
    row_owner = torch.empty(t, dtype=torch.int32)
    for owner, (start, end) in enumerate(zip(starts, ends)):
        row_base[start:end] = start
        row_owner[start:end] = owner

    def stacked(name, rows, tail, dtype, index):
        counts = {GLOBAL: groups * INTERVAL, LINEAR_ATTENTION: groups * LINEAR_PER_GROUP,
                  FULL_ATTENTION: groups}
        shape = [counts[index] * rows] + tail
        return TensorSpec(name, shape, dtype,
                          init_value=lambda: stacks(t, groups, real, weights_dir, seed)[name])

    def draw(offset, dtype):
        def make():
            gen = torch.Generator().manual_seed(seed + offset)
            return torch.randn(t, C, generator=gen).to(dtype)
        return make

    def rope(part):
        def make():
            positions = torch.cat([
                torch.arange(base * KV_PAGE_SIZE, base * KV_PAGE_SIZE + end - start)
                for base, start, end in zip(bases, starts, ends)
            ])
            cos, sin = reference.rotary_cos_sin(positions, QWEN3_8_27B)
            return (cos if part == "cos" else sin).float()
        return make

    stack_specs = [stacked(*entry) for entry in stack_layout()]
    return ([
        TensorSpec("residual", [t, C], torch.float32, init_value=draw(0, torch.float32)),
        TensorSpec("delta", [t, C], torch.bfloat16, init_value=draw(1, torch.bfloat16)),
    ] + stack_specs + [
        TensorSpec("tril", [chunk, chunk], torch.float32,
                   init_value=lambda: torch.tril(torch.ones(chunk, chunk, dtype=torch.float32))),
        TensorSpec("mask_strict", [chunk, chunk], torch.float32,
                   init_value=lambda: (torch.arange(chunk)[:, None] > torch.arange(chunk)[None, :]).float()),
        TensorSpec("eye", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.eye_block(chunk)),
        TensorSpec("m_diag", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.blk_masks(chunk)[0]),
        TensorSpec("m_low", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.blk_masks(chunk)[1]),
        TensorSpec("rope_cos", [t, ROT], torch.float32, init_value=rope("cos")),
        TensorSpec("rope_sin", [t, ROT], torch.float32, init_value=rope("sin")),
        TensorSpec("mask_bias", [ROW_TILE, BIAS_WIDTH], torch.float32,
                   init_value=lambda: gated_attention.attn_core.causal_bias(ROW_TILE, KV_TILE)),
        TensorSpec("row_base", [t], torch.int32, init_value=lambda: row_base),
        TensorSpec("row_owner", [t], torch.int32, init_value=lambda: row_owner),
        TensorSpec("row_ends", [rows], torch.int32, init_value=lambda: torch.tensor(ends, dtype=torch.int32)),
        TensorSpec("state_rows", [owned], torch.int32, init_value=lambda: state_rows),
        TensorSpec("page_bases", [rows], torch.int32, init_value=lambda: torch.tensor(bases, dtype=torch.int32)),
        TensorSpec("visible_pages", [rows], torch.int32, init_value=lambda: torch.tensor(visible, dtype=torch.int32)),
        TensorSpec("block_table", [rows, max_pages], torch.int32, init_value=lambda: block_table),
        TensorSpec("conv_state", list(conv_pool.shape), torch.bfloat16,
                   init_value=lambda: conv_pool),
        TensorSpec("final_state", list(state_pool.shape), torch.float32,
                   init_value=lambda: state_pool),
        TensorSpec("k_pages", list(k_pool.shape), torch.bfloat16,
                   init_value=lambda: k_pool),
        TensorSpec("v_pages", list(v_pool.shape), torch.bfloat16,
                   init_value=lambda: v_pool),
        TensorSpec("residual_out", [t, C], torch.float32),
        TensorSpec("mlp_out", [t, C], torch.bfloat16),
    ])


# ---------------------------------------------------------------------------
# Oracle and acceptance
# ---------------------------------------------------------------------------


def stack_chains(tensors: dict, residual, delta, t: int, groups: int = GROUPS,
                 chunk: int = CHUNK_TILE):
    """Independent float64 and fake-quant stack chains for each packed request."""
    import torch

    import a8w8_linear
    import reference

    a8w8_linear.host_torch_setup()
    cache_names = ("conv_state", "final_state", "k_pages", "v_pages")
    truth_cache = {name: tensors[name].double().clone() for name in cache_names}
    chain_cache = {name: tensors[name].clone() for name in cache_names}
    kv_layer_stride = tensors["k_pages"].shape[0] // groups
    ends = tensors["row_ends"].tolist()
    starts = [0] + ends[:-1]
    rows = len(ends)
    outputs = [torch.empty((t, C), dtype=torch.float64) for _ in range(4)]
    for owner, (start, end) in enumerate(zip(starts, ends)):
        page_base = int(tensors["page_bases"][owner])
        table = tensors["block_table"][owner : owner + 1]
        resid_t, delta_t = residual[start:end], delta[start:end]
        resid_q, delta_q = residual[start:end], delta[start:end]
        for mixer, glb, local in layer_plan(groups):
            weights = unstack(tensors, mixer, glb, local)
            weights["rope_cos"] = tensors["rope_cos"][start:end]
            weights["rope_sin"] = tensors["rope_sin"][start:end]
            kind = "gdn" if mixer == LINEAR_ATTENTION else "attn"
            mlp_w = {name: weights[name] for name in
                     ("w_gate", "w_gate_scale", "w_up", "w_up_scale", "w_down", "w_down_scale")}
            norms = (weights["norm_in_w"], weights["norm_post_w"])
            state_row = int(tensors["state_rows"][local * rows + owner]) if kind == "gdn" else 0
            lo = local * kv_layer_stride
            hi = lo + kv_layer_stride

            def layer_cache(pool):
                layer = dict(weights)
                if kind == "gdn":
                    layer.update(state_row=state_row, conv_state=pool["conv_state"],
                                 final_state=pool["final_state"])
                else:
                    layer.update(block_table=table, page_base=page_base, layer_base=0,
                                 k_pages=pool["k_pages"][lo:hi], v_pages=pool["v_pages"][lo:hi])
                return layer

            truth_mixer = prefill_layer.mixer_oracle(kind, layer_cache(truth_cache), end - start, False, chunk)
            truth = reference.residual_layer(resid_t, delta_t, *norms, truth_mixer, mlp_w, QWEN3_8_27B, quant=False)
            resid_t, delta_t = truth["residual_out"], truth["mlp_out"]
            if kind == "gdn":
                truth_cache["conv_state"][state_row].copy_(prefill_layer._TRUTH["conv_state"])
                truth_cache["final_state"][state_row].copy_(prefill_layer._TRUTH["final_state"])
            else:
                reference.write_kv_pages(
                    truth_cache["k_pages"][lo:hi], truth_cache["v_pages"][lo:hi], table,
                    prefill_layer._TRUTH["k_pages"], prefill_layer._TRUTH["v_pages"],
                    page_base * KV_PAGE_SIZE, KV_PAGE_SIZE,
                )
            chain_mixer = prefill_layer.mixer_oracle(kind, layer_cache(chain_cache), end - start, True, chunk)
            chain = reference.residual_layer(resid_q, delta_q, *norms, chain_mixer, mlp_w, QWEN3_8_27B, quant=True)
            resid_q, delta_q = chain["residual_out"], chain["mlp_out"]
        for output, value in zip(outputs, (resid_t, delta_t, resid_q, delta_q)):
            output[start:end] = value
    prefill_layer.record_truth(**truth_cache)
    for name in cache_names:
        tensors[name].copy_(chain_cache[name])
    return tuple(outputs)


def golden_prefill_fwd(t: int, groups: int = GROUPS, chunk: int = CHUNK_TILE):
    """The stack's two chains: the fake-quant one written out, the truth stashed."""
    import torch

    def golden(tensors):
        resid_t, delta_t, resid_q, delta_q = stack_chains(
            tensors, tensors["residual"], tensors["delta"], t, groups, chunk)
        prefill_layer.record_truth(residual_out=resid_t, mlp_out=delta_t)
        tensors["residual_out"].copy_(resid_q.to(torch.float32))
        tensors["mlp_out"].copy_(delta_q.to(torch.bfloat16))

    return golden


# Acceptance. The two gates against the float64 truth are `prefill_layer`'s unchanged --
# a stack of layers that each sit at the quantisation floor sits at the floor too, and
# measured it does: 1.001x at both four and eight layers. The two against the fake-quant
# chain are not, and the reason is geometric rather than a tolerance being too tight.
#
# The kernel and the host chain each round differently at every layer, and over depth
# those differences decorrelate. Two errors of equal size f that are independent are
# sqrt(2) * f apart, so the kernel-to-chain distance climbs from one floor towards
# **sqrt(2) floors** as layers accumulate while both stay one floor from the truth:
# measured 1.052 at four layers and 1.113 at eight, against the layer entry's bound of
# 1.00. The bound here is sqrt(2) with a little margin. It stays below 2.0 deliberately,
# because 2.0 is where an output mirrored across the truth sits -- that output passes the
# truth gate at 1.000x, so this is the only gate that refuses it, and widening it past
# 2.0 would cost the gate its purpose.
STACK_MAX_VS_GOLDEN = 1.45

# The floor itself grows with depth as well: 4.6e-02 over four layers, 6.0e-02 over eight,
# both well inside the single layer's 0.25. The guard exists to refuse an oracle that has
# broken rather than one that has compounded -- a chain that has lost its signal reads
# near 1.0, decorrelated from the truth -- so it is raised rather than removed, and the
# floor is printed either way.
STACK_MAX_FLOOR = 0.60


def cache_regions(name: str, inputs: dict, groups: int, pool_rows: int):
    """Owned cache rows, separated by layer and request."""
    import torch

    import reference

    ends = inputs["row_ends"].tolist()
    starts = [0] + ends[:-1]
    rows = len(ends)
    layers = groups * LINEAR_PER_GROUP if name in ("conv_state", "final_state") else groups
    for local in range(layers):
        for owner, (start, end) in enumerate(zip(starts, ends)):
            if name in ("conv_state", "final_state"):
                indices = inputs["state_rows"][local * rows + owner].reshape(1).long()
            else:
                stride = pool_rows // groups
                prefix = int(inputs["page_bases"][owner]) * KV_PAGE_SIZE
                indices = reference.kv_page_rows(inputs["block_table"], owner, prefix + end - start, KV_PAGE_SIZE)
                indices = local * stride + indices[prefix:]
            yield f"layer {local} owner {owner}", indices.to(dtype=torch.long, device="cpu")


def compare_stack_cache(name: str, groups: int, t: int):
    """Gate each layer/owner against its own floor and preserve every other row."""
    import torch

    def compare(actual, expected, inputs=None, actual_outputs=None,
                expected_outputs=None, **_kwargs):
        if inputs is None:
            return False, f"{name}: input cache descriptors are required"
        touched = torch.zeros(actual.shape[0], dtype=torch.bool)
        for label, rows in cache_regions(name, inputs, groups, actual.shape[0]):
            touched[rows] = True
            dev = actual[rows].double()
            fq = expected[rows].double()
            truth = prefill_layer._TRUTH[name][rows].double()
            den = torch.linalg.vector_norm(truth)
            floor = float(torch.linalg.vector_norm(fq - truth) / den)
            peak = torch.linalg.vector_norm(truth.reshape(truth.shape[0], -1), dim=1).max()
            floor_rows = (fq - truth).reshape(truth.shape[0], -1)
            row_floor = float(torch.linalg.vector_norm(floor_rows, dim=1).max() / peak)
            if not (0 < floor <= STACK_MAX_FLOOR and row_floor > 0):
                return False, f"{name} {label}: invalid quantisation floor {floor}, row floor {row_floor}"
            over = float(torch.linalg.vector_norm(dev - truth) / den) / floor
            near = float(torch.linalg.vector_norm(dev - fq) / den) / floor
            error_rows = (dev - truth).reshape(truth.shape[0], -1)
            near_rows = (dev - fq).reshape(truth.shape[0], -1)
            row_over = float(torch.linalg.vector_norm(error_rows, dim=1).max() / peak) / row_floor
            row_near = float(torch.linalg.vector_norm(near_rows, dim=1).max() / peak) / row_floor
            line = (f"[stats] {name} {label}: floor {floor:.3e}, truth {over:.3f}x, "
                    f"worst row {row_over:.3f}x; fake quant {near:.3f}x, worst row {row_near:.3f}x")
            print(line, flush=True)
            if not (over <= 1.15 and row_over <= 1.25 and near <= STACK_MAX_VS_GOLDEN and row_near <= 1.50):
                return False, line
        if not bool((actual[~touched] == expected[~touched]).all()):
            return False, f"{name}: rows outside the requests changed"
        if name == "v_pages":
            return compare_decode_handoff(inputs, actual_outputs, expected_outputs, groups, t)
        return True, ""

    return compare


def compare_decode_handoff(inputs, actual_outputs, expected_outputs, groups: int, t: int):
    """Consume each request's published caches and compare with a zero-cache control."""
    import torch

    import reference

    if actual_outputs is None or expected_outputs is None:
        return False, "decode consumer gate needs all published cache outputs"
    gen = torch.Generator().manual_seed(314)
    ends = inputs["row_ends"].tolist()
    starts = [0] + ends[:-1]
    rows = len(ends)
    next_x = torch.randn(rows, C, generator=gen).to(torch.bfloat16)
    actual = {name: actual_outputs[name].cpu() for name in
              ("conv_state", "final_state", "k_pages", "v_pages")}
    expected = {name: expected_outputs[name].cpu() for name in actual}
    scores = []
    for local in range(groups * LINEAR_PER_GROUP):
        glb = local + local // LINEAR_PER_GROUP
        w = gdn_block.block_weights(unstack(inputs, LINEAR_ATTENTION, glb, local))
        for owner in range(rows):
            state_row = int(inputs["state_rows"][local * rows + owner])

            def consume(conv, state):
                return reference.recurrent_block(
                    next_x[owner : owner + 1], w, QWEN3_8_27B, conv_state=conv,
                    state=state.transpose(-2, -1), quant_x=True,
                )["out"]

            conv = expected["conv_state"][state_row]
            state = expected["final_state"][state_row]
            expected_out = consume(conv, state)
            actual_out = consume(actual["conv_state"][state_row], actual["final_state"][state_row])
            corrupt_out = consume(torch.zeros_like(conv), torch.zeros_like(state))
            den = torch.linalg.vector_norm(corrupt_out - expected_out)
            score = float(torch.linalg.vector_norm(actual_out - expected_out) / den)
            if not score <= 0.5:
                return False, f"decode GDN layer {glb} owner {owner}: {score:.3f} of zero-cache control"
            scores.append(score)

    q = (torch.randn(rows, O_WIDTH, generator=gen) * 0.05).to(torch.bfloat16)
    stride = actual["k_pages"].shape[0] // groups
    for group in range(groups):
        lo, hi = group * stride, (group + 1) * stride
        for owner, (start, end) in enumerate(zip(starts, ends)):
            length = int(inputs["page_bases"][owner]) * KV_PAGE_SIZE + end - start
            table = inputs["block_table"][owner : owner + 1]
            seq_lens = torch.tensor([length], dtype=torch.int32)

            def consume(k_pool, v_pool):
                return reference.decode_gqa_attention(
                    q[owner : owner + 1], k_pool, v_pool, table, seq_lens,
                    QWEN3_8_27B.num_attention_heads, QWEN3_8_27B.num_key_value_heads,
                    QWEN3_8_27B.head_dim, KV_PAGE_SIZE,
                )

            expected_out = consume(expected["k_pages"][lo:hi], expected["v_pages"][lo:hi])
            actual_out = consume(actual["k_pages"][lo:hi], actual["v_pages"][lo:hi])
            zero_k, zero_v = expected["k_pages"][lo:hi].clone(), expected["v_pages"][lo:hi].clone()
            current_rows = reference.kv_page_rows(table, 0, length, KV_PAGE_SIZE)[-(end - start):]
            zero_k[current_rows], zero_v[current_rows] = 0, 0
            corrupt_out = consume(zero_k, zero_v)
            den = torch.linalg.vector_norm(corrupt_out - expected_out)
            score = float(torch.linalg.vector_norm(actual_out - expected_out) / den)
            if not score <= 0.5:
                return False, f"decode attention group {group} owner {owner}: {score:.3f} of zero-cache control"
            scores.append(score)
    line = f"[stats] decode consumers: max device error {max(scores):.3f} of zero-cache control"
    print(line, flush=True)
    return True, line


def compare_fns(groups: int = GROUPS, t: int = BENCH_SEQ) -> dict:
    """The stream and cache gates, at stack bounds."""
    result = {name: prefill_layer.compare_packed_vs_floor(name, max_vs_golden=STACK_MAX_VS_GOLDEN,
                                                    max_floor=STACK_MAX_FLOOR)
              for name in ("residual_out", "mlp_out")}
    result.update({name: compare_stack_cache(name, groups, t)
                   for name in ("conv_state", "final_state", "k_pages", "v_pages")})
    return result


__all__ = ["BENCH_SEQ", "DEFAULT_GROUPS", "DEFAULT_SEQ", "FWD_RING_HEAP", "GROUPS",
           "fwd_ring_heap",
           "build_kernel", "build_tensor_specs", "golden_prefill_fwd", "layer_plan",
           "compare_fns", "run_config", "stack_chains", "stack_index", "stack_layout",
           "stacks", "unstack"]


def heap_arg(text: str):
    """`--ring-heap`: one value for rings 1 to 3, or four comma-separated per-ring values.

    The per-ring form is the lever when one ring needs far more than the others and the
    weights leave no room to size them all the same.
    """
    parts = [int(p, 0) for p in text.split(",")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--ring-heap takes one value or four, one per ring")
    return parts


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--groups", type=int, default=None,
                        help=f"LLLA groups to run; default: {DEFAULT_GROUPS}")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ,
                        help="tokens in the chunk, a multiple of the GDN mixer's patch")
    parser.add_argument("--packed", action="store_true", help="pack independently owned requests")
    parser.add_argument("--row-starts", default="0,256", help="packed request starts, in tokens")
    parser.add_argument("--page-bases", default="0,0", help="cached 128-token pages per packed request")
    parser.add_argument("--real", action="store_true", default=False,
                        help="the extracted checkpoint layers instead of per-layer draws")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="seed nonzero GDN state and continue after prior KV pages")
    parser.add_argument("--page-base", type=int, default=0,
                        help="cached 128-token pages before this chunk")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="where those dumps live, when not beside this checkout")
    parser.add_argument("--ring-heap", type=heap_arg, default=0,
                        help="bytes on rings 1 to 3, or four comma-separated per-ring values")
    parser.add_argument("--task-window", type=int, default=0,
                        help="in-flight tasks on rings 1 to 3, a power of two at least 4")
    parser.add_argument("--dep-pool", type=int, default=0,
                        help="dependency edges on rings 1 to 3, at least 4")
    parser.add_argument("--ring", type=int, default=0, choices=range(4),
                        help="confine the three overrides above to one ring, to find which binds")
    parser.add_argument("--log-level", type=str, default=None,
                        help="runtime log threshold: debug, v0..v9, info, warn, error, null")
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the stack over this many rounds (device only)")
    parser.add_argument("--no-golden", action="store_true", default=False,
                        help="run without validating: the 64-layer reference is hours of host time")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    if args.seq_len % DEFAULT_SEQ:
        parser.error(f"--seq-len must be a multiple of {DEFAULT_SEQ}")
    if args.page_base % (KV_TILE // KV_PAGE_SIZE):
        parser.error("--page-base must align to the attention KV tile")
    starts = tuple(int(value) for value in args.row_starts.split(",")) if args.packed else (0,)
    bases = tuple(int(value) for value in args.page_bases.split(",")) if args.packed else (args.page_base,)
    if len(starts) != len(bases):
        parser.error("--row-starts and --page-bases must name the same requests")
    plan = [args.groups] if args.groups else ([1] if args.packed else list(DEFAULT_GROUPS))
    for groups in plan:
        if not 1 <= groups <= GROUPS:
            parser.error(f"--groups must be in [1, {GROUPS}], got {groups}")
    if len(plan) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one case; name it with --groups")
    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "2"

    def rings(value):
        """A per-ring list as given, one value on rings 1 to 3, or on the single `--ring`."""
        if isinstance(value, list):
            return value
        if not value:
            return None
        if args.ring:
            return [value if r == args.ring else 0 for r in range(4)]
        return [0, value, value, value]

    heap, window, pool = rings(args.ring_heap), rings(args.task_window), rings(args.dep_pool)
    for groups in plan:
        t = args.seq_len
        ends = starts[1:] + (t,)
        max_pages = max(base + (end - start) // KV_PAGE_SIZE for base, start, end in zip(bases, starts, ends)) + 2
        kv_layer_stride = (len(starts) * max_pages + 3) * KV_PAGE_SIZE
        weights = "real" if args.real else "random"
        print(f"[case] prefill_fwd groups={groups} ({groups * INTERVAL} layers) "
              f"T={t} starts={starts} page_bases={bases} resume={args.resume} {weights} weights", flush=True)
        result = run(
            fn=build_kernel(t=t, groups=groups, max_pages=max_pages,
                            kv_layer_stride=kv_layer_stride, rows=len(starts)),
            specs=build_tensor_specs(t=t, groups=groups, real=args.real,
                                     weights_dir=args.weights_dir, page_base=args.page_base,
                                     max_pages=max_pages, kv_layer_stride=kv_layer_stride,
                                     resume=args.resume, starts=starts, bases=bases),
            golden_fn=(None if args.no_golden
                       else golden_prefill_fwd(t, groups=groups)),
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=run_config(args.backend_type, args.execution_mode, args.device,
                              args.enable_chip_swimlane, heap or fwd_ring_heap(t), window, pool,
                              args.log_level),
            rtol=1e-2,
            atol=1e-5,
            compare_fn=({} if args.no_golden else compare_fns(groups, t)),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at groups={groups} T={t}")
