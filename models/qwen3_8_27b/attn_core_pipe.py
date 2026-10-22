# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B causal GQA attention core as an explicit cube/vector pipeline.

Same arithmetic as `attn_core.py`, with the cube and vector programs written
separately and joined by GM mailboxes and three events:

    AIC  QK  --scores-->  AIV softmax  --P-->  AIC PV  --PV-->  AIV update O

The cube runs `PRE_LAUNCH` steps ahead of the vector. The rescale a step
produces is consumed `PRE_LAUNCH` steps later, so the vector side keeps the last
two in a two-entry queue.

A step serves `PAIR` adjacent query blocks from one K and one V tile. Their rows
are contiguous, so the cube runs one `PAIR_ROWS`-row matmul rather than `PAIR`
of `ROW_TILE`, and the vector side keeps a separate accumulator per block.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
HQ = QWEN3_8_27B.num_attention_heads    # query heads
HKV = QWEN3_8_27B.num_key_value_heads   # KV heads
D = QWEN3_8_27B.head_dim                # head dimension
# validation
# one KV step, then two. Sized so a case also finishes inside the simulator's
# 20 s stall watchdog; the prefill shape comes from --seq-len.
DEFAULT_CASES = (256, 512)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# a5sim cannot compile the cross-core wait (pto-isa#612).

# tiling
ROW_TILE = 64           # query rows per tile; the vector side's unit
# 256 keys per step measures 7632.4 us against 8146.4 at 128 (T = 8192, a2a3).
# The auto-synchronised core could not reach 256: its ring slots are charged to
# the vector buffer, which 256 overflowed. This one's mailboxes are in GM.
KV_TILE = 256           # keys per online-softmax step
# The cube's QK and PV products are PAIR_ROWS * KV_TILE * 4 and PAIR_ROWS * D * 4
# bytes of the accumulator, which caps PAIR_ROWS at 128. The vector side is the
# tighter bound on ROW_TILE: at 128 its four [ROW_TILE / AIV_LANES, KV_TILE]
# tiles need 263936 bytes of a 188416-byte buffer. Pairing widens the matmul
# without widening those tiles.
PAIR = 2                # query blocks sharing one K/V load and one matmul
AIV_LANES = 2           # vector lanes per cube core, each owning half the rows
PRE_LAUNCH = 2          # steps the cube runs ahead of the vector
SLOTS = PRE_LAUNCH + 1  # mailbox slots, so a step in flight is never overwritten

# The running max seeds at a large negative number rather than -inf: the first
# step's rescale is `exp(seed - m0)`, which must evaluate to zero and not NaN.
M_SEED = -3.0e38

# event ids, one per hand-off direction
SCORE_READY = 0         # AIC -> AIV, FP32 scores
PROB_READY = 1          # AIV -> AIC, BF16 probabilities
PV_READY = 2            # AIC -> AIV, FP32 P @ V


Q_WIDTH = HQ * D
KV_WIDTH = HKV * D
GROUP = HQ // HKV                       # query heads sharing one KV head
RATIO = KV_TILE // ROW_TILE
BIAS_WIDTH = 2 * KV_TILE - ROW_TILE
LANE_ROWS = ROW_TILE // AIV_LANES
PAIR_ROWS = PAIR * ROW_TILE             # query rows a step retires
MAILBOX_ROWS = HQ * SLOTS * PAIR_ROWS
MAX_DIAG_OFF = PAIR_ROWS * (RATIO // PAIR - 1)
if ROW_TILE % AIV_LANES:
    raise ValueError(f"ROW_TILE={ROW_TILE} must divide by AIV_LANES={AIV_LANES}")
# A step's rescale is consumed PRE_LAUNCH steps after it is produced, and the
# queue that holds it is written out as exactly two entries. At any other depth
# the output update applies the wrong step's rescale -- a silent wrong answer.
if PRE_LAUNCH != 2:
    raise ValueError(f"PRE_LAUNCH={PRE_LAUNCH} is not supported: the rescale queue holds two entries")
# The vector side is written out as exactly two blocks, and they share one
# `n_kv` only while RATIO is a multiple of PAIR -- otherwise the blocks of a
# pair fall in different `qi // RATIO` buckets and need different step counts.
if PAIR != 2 or RATIO % PAIR:
    raise ValueError(f"PAIR={PAIR} is not supported: two blocks, and RATIO={RATIO} must be a multiple")
# The diagonal bias is read at KV_TILE - ROW_TILE - (row - c_diag), so the pair
# furthest below its diagonal tile reads the leftmost column, and its last block
# reads (PAIR - 1) * ROW_TILE further left again.
if KV_TILE - ROW_TILE - MAX_DIAG_OFF - (PAIR - 1) * ROW_TILE < 0:
    raise ValueError(f"the mask table starts before column 0 at PAIR={PAIR}, RATIO={RATIO}")


def _qwen3_attn_core_pipe(
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
    # GM mailboxes: one disjoint slot ring per query head, so no two cores
    # and no two steps in flight share rows. A slot holds the pair's rows.
    score_mb = pl.create_tensor([MAILBOX_ROWS, KV_TILE], dtype=pl.FP32)
    prob_mb = pl.create_tensor([MAILBOX_ROWS, KV_TILE], dtype=pl.BF16)
    pv_mb = pl.create_tensor([MAILBOX_ROWS, D], dtype=pl.FP32)
    ffts = pl.create_tensor([256], dtype=pl.INT64)

    for hh in pl.spmd(HQ, name_hint="attn_core_pipe"):
        pl.system.set_ffts(ffts)
        qd0 = hh * D
        kd0 = (hh // GROUP) * D
        head_row = hh * SLOTS * PAIR_ROWS

        for qp in pl.range(t_dim // PAIR_ROWS):
            r0 = qp * PAIR_ROWS
            r1 = r0 + ROW_TILE
            # The diagonal tile is the last step, and the only masked one. Both
            # blocks of the pair share it, RATIO being a multiple of PAIR.
            n_below = (qp * PAIR) // RATIO
            n_kv = n_below + 1
            c_diag = n_below * KV_TILE
            bias_c0 = KV_TILE - ROW_TILE - (r0 - c_diag)
            bias_c1 = bias_c0 - ROW_TILE

            # ---- cube: QK for step `tick`, PV for step `tick - PRE_LAUNCH`
            q_pair = pl.load(q, [r0, qd0], [PAIR_ROWS, D], target_memory=pl.MemorySpace.Mat)
            for tick in pl.range(n_kv + PRE_LAUNCH):
                if tick < n_kv:
                    qk_row = head_row + (tick % SLOTS) * PAIR_ROWS
                    k_tile = pl.load(k, [tick * KV_TILE, kd0], [KV_TILE, D],
                                     target_memory=pl.MemorySpace.Mat)
                    scores = pl.matmul(q_pair, pl.tile.transpose_view(k_tile), out_dtype=pl.FP32)
                    pl.store(scores, [qk_row, 0], score_mb)
                    pl.system.sync_set(SCORE_READY, pipe=pl.PipeType.FIX, ffts_mode=2,
                                       core_type=pl.KernelType.AIC)
                if tick >= PRE_LAUNCH:
                    step = tick - PRE_LAUNCH
                    pv_row = head_row + (step % SLOTS) * PAIR_ROWS
                    # V does not depend on the softmax, so its load overlaps it.
                    v_tile = pl.load(v, [step * KV_TILE, kd0], [KV_TILE, D],
                                     target_memory=pl.MemorySpace.Mat)
                    pl.system.sync_wait(PROB_READY, pipe=pl.PipeType.MTE2,
                                        core_type=pl.KernelType.AIC)
                    p_pair = pl.load(prob_mb, [pv_row, 0], [PAIR_ROWS, KV_TILE],
                                     target_memory=pl.MemorySpace.Mat)
                    pv = pl.matmul(p_pair, v_tile, out_dtype=pl.FP32)
                    pl.store(pv, [pv_row, 0], pv_mb)
                    pl.system.sync_set(PV_READY, pipe=pl.PipeType.FIX, ffts_mode=2,
                                       core_type=pl.KernelType.AIC)

            # ---- vector: softmax for step `tick`, output update for `tick - PRE_LAUNCH`
            for lane in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(ffts)
                lane_row = lane * LANE_ROWS
                red_tmp = pl.create_tile([LANE_ROWS, KV_TILE], dtype=pl.FP32,
                                         target_memory=pl.MemorySpace.Vec)
                zero_scores = pl.tile.full([LANE_ROWS, KV_TILE], dtype=pl.FP32, value=0.0)
                zrow = pl.row_max(zero_scores, red_tmp)
                # Each loop-carried tile needs its own initial tile: two seeded
                # from one share a buffer, and an iteration cannot preserve both.
                m_seed = pl.tile.adds(zrow, M_SEED)
                l_seed = pl.tile.move(zrow, target_memory=pl.MemorySpace.Vec)
                o_seed = pl.tile.full([LANE_ROWS, D], dtype=pl.FP32, value=0.0)
                # Two-entry rescale queue: step i's rescale is consumed
                # PRE_LAUNCH steps later, so it is held for that long.
                pend0_seed = pl.tile.adds(zrow, 1.0)
                pend1_seed = pl.tile.adds(zrow, 1.0)
                m_seed_b = pl.tile.adds(zrow, M_SEED)
                l_seed_b = pl.tile.move(zrow, target_memory=pl.MemorySpace.Vec)
                o_seed_b = pl.tile.full([LANE_ROWS, D], dtype=pl.FP32, value=0.0)
                pend0_seed_b = pl.tile.adds(zrow, 1.0)
                pend1_seed_b = pl.tile.adds(zrow, 1.0)

                for tick, (m_a, l_a, o_a, p0_a, p1_a,
                           m_b, l_b, o_b, p0_b, p1_b) in pl.range(
                        n_kv + PRE_LAUNCH,
                        init_values=(m_seed, l_seed, o_seed, pend0_seed, pend1_seed,
                                     m_seed_b, l_seed_b, o_seed_b, pend0_seed_b, pend1_seed_b)):
                    if tick < n_kv:
                        sm_row = head_row + (tick % SLOTS) * PAIR_ROWS + lane_row
                        sb_row = sm_row + ROW_TILE
                        pl.system.sync_wait(SCORE_READY, pipe=pl.PipeType.MTE2,
                                            core_type=pl.KernelType.AIV)
                        sc_a = pl.load(score_mb, [sm_row, 0], [LANE_ROWS, KV_TILE],
                                       target_memory=pl.MemorySpace.Vec)
                        if tick == n_kv - 1:
                            bias_a = pl.load(mask_bias, [lane_row, bias_c0], [LANE_ROWS, KV_TILE],
                                             target_memory=pl.MemorySpace.Vec)
                            sc_a = pl.add(sc_a, bias_a)
                        mc_a = pl.row_max(sc_a, red_tmp)
                        mn_a = pl.maximum(m_a, mc_a)
                        rs_a = pl.exp(pl.sub(m_a, mn_a))
                        pr_a = pl.exp(pl.row_expand_sub(sc_a, mn_a))
                        ln_a = pl.add(pl.mul(l_a, rs_a), pl.row_sum(pr_a, red_tmp))
                        pl.store(pl.cast(pr_a, target_type=pl.BF16, mode="rint"), [sm_row, 0], prob_mb)
                        sc_b = pl.load(score_mb, [sb_row, 0], [LANE_ROWS, KV_TILE],
                                       target_memory=pl.MemorySpace.Vec)
                        if tick == n_kv - 1:
                            bias_b = pl.load(mask_bias, [lane_row, bias_c1], [LANE_ROWS, KV_TILE],
                                             target_memory=pl.MemorySpace.Vec)
                            sc_b = pl.add(sc_b, bias_b)
                        mc_b = pl.row_max(sc_b, red_tmp)
                        mn_b = pl.maximum(m_b, mc_b)
                        rs_b = pl.exp(pl.sub(m_b, mn_b))
                        pr_b = pl.exp(pl.row_expand_sub(sc_b, mn_b))
                        ln_b = pl.add(pl.mul(l_b, rs_b), pl.row_sum(pr_b, red_tmp))
                        pl.store(pl.cast(pr_b, target_type=pl.BF16, mode="rint"), [sb_row, 0], prob_mb)
                        # one event for the pair: both blocks' probabilities are down
                        pl.system.sync_set(PROB_READY, pipe=pl.PipeType.MTE3, ffts_mode=2,
                                           core_type=pl.KernelType.AIV)
                        mt_a, lt_a, rt_a, mt_b, lt_b, rt_b = pl.yield_(mn_a, ln_a, rs_a, mn_b, ln_b, rs_b)
                    else:
                        mt_a, lt_a, rt_a, mt_b, lt_b, rt_b = pl.yield_(m_a, l_a, p1_a, m_b, l_b, p1_b)

                    if tick >= PRE_LAUNCH:
                        step = tick - PRE_LAUNCH
                        up_row = head_row + (step % SLOTS) * PAIR_ROWS + lane_row
                        pl.system.sync_wait(PV_READY, pipe=pl.PipeType.MTE2,
                                            core_type=pl.KernelType.AIV)
                        part_a = pl.load(pv_mb, [up_row, 0], [LANE_ROWS, D],
                                         target_memory=pl.MemorySpace.Vec)
                        up_a = pl.add(pl.row_expand_mul(o_a, p0_a), part_a)
                        part_b = pl.load(pv_mb, [up_row + ROW_TILE, 0], [LANE_ROWS, D],
                                         target_memory=pl.MemorySpace.Vec)
                        up_b = pl.add(pl.row_expand_mul(o_b, p0_b), part_b)
                        ot_a, ot_b = pl.yield_(up_a, up_b)
                    else:
                        ot_a, ot_b = pl.yield_(o_a, o_b)

                    p0n_a = pl.tile.move(p1_a, target_memory=pl.MemorySpace.Vec)
                    p1n_a = pl.tile.move(rt_a, target_memory=pl.MemorySpace.Vec)
                    p0n_b = pl.tile.move(p1_b, target_memory=pl.MemorySpace.Vec)
                    p1n_b = pl.tile.move(rt_b, target_memory=pl.MemorySpace.Vec)
                    (me_a, le_a, oe_a, q0_a, q1_a,
                     me_b, le_b, oe_b, q0_b, q1_b) = pl.yield_(mt_a, lt_a, ot_a, p0n_a, p1n_a,
                                                               mt_b, lt_b, ot_b, p0n_b, p1n_b)

                ctx_a = pl.row_expand_mul(oe_a, pl.recip(le_a, high_precision=True))
                pl.store(pl.cast(ctx_a, target_type=pl.BF16, mode="rint"), [r0 + lane_row, qd0], out)
                ctx_b = pl.row_expand_mul(oe_b, pl.recip(le_b, high_precision=True))
                pl.store(pl.cast(ctx_b, target_type=pl.BF16, mode="rint"), [r1 + lane_row, qd0], out)
    return out


qwen3_attn_core_pipe = pl.jit.inline(auto_scope=False)(_qwen3_attn_core_pipe)
qwen3_attn_core_pipe_test = pl.jit(auto_scope=False)(_qwen3_attn_core_pipe)


def build_tensor_specs(t: int = BENCH_SEQ, hq: int = HQ, hkv: int = HKV, d: int = D,
                       row_tile: int = ROW_TILE, kv_tile: int = KV_TILE):
    import attn_core

    return attn_core.build_tensor_specs(t=t, hq=hq, hkv=hkv, d=d,
                                        row_tile=row_tile, kv_tile=kv_tile)


def golden_qwen3_attn_core_pipe(tensors):
    import attn_core

    attn_core.golden_qwen3_attn_core(tensors)


def compare_out(actual, expected, **kwargs):
    import attn_core

    return attn_core.compare_out(actual, expected, **kwargs)


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % KV_TILE:
        parser.error(f"--seq-len must be a multiple of KV_TILE={KV_TILE}")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")
    compile_only = args.compile_only
    for seq_len in cases:
        print(f"[case] attn_core_pipe T={seq_len}", flush=True)
        result = run(
            fn=qwen3_attn_core_pipe_test,
            specs=build_tensor_specs(t=seq_len),
            golden_fn=None if compile_only else golden_qwen3_attn_core_pipe,
            compile_only=compile_only,
            compare_fn=None if compile_only else {"out": compare_out},
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
            ),
            rtol=1e-2,
            atol=1e-2,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at T={seq_len}")
