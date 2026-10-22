# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B paged attention, decode: one query row per sequence over a paged KV cache.

    o = softmax(q k^T) v        per query head, over every cached key

`q` arrives already q-normed, RoPE'd and scaled by `head_dim ** -0.5`, which is what
`qk_rmsnorm_rope` produces. Query head `h` reads KV head `h // (HQ // HKV)`.

**There is no causal mask.** The query is the newest position, so every cached key is in
its past -- the one simplification decode gets over the prefill core, which spends a mask
table on its diagonal tile.

One task owns one `(sequence, KV head)` pair and the `GROUP` query heads that share it.
The KV page it reads is the operand worth amortising: those `GROUP` heads would otherwise
load the same page once each.

`GROUP` is 6 and the cube's M fractal is 16, so the query tile is **16 rows with 6 real
ones**, declared through the slice's `valid_shape`. The padding is free for the same
reason it is free in the projections: this kernel is bound by the KV stream, not by cube
rows. A token reads `S * 4 KiB` per layer -- 64 KiB per token across the 16
full-attention layers -- so at 8192 context it is 0.537 GB per decode token against
about 2 MFLOP of arithmetic per page.

The sequence length rarely lands on a page boundary, so the last page holds rows that are
cached but not yet real. `tail_mask` is indexed by the page's valid count: row `v` opens
its first `v` columns and closes the rest, and row `PAGE` -- which every full page selects
-- is all open. That removes the branch between full and partial pages entirely, where
the prefill core needs a separate masked diagonal step.

`l` sums the **BF16-rounded** probabilities, not the FP32 ones, because the BF16 tile is
what the PV matmul consumes: the denominator has to be the sum of the numerator's own
terms.

**`SPLITS` partitions each sequence's pages across several tasks**, and it buys the length
of the serial chain, not occupancy: the online-softmax carry is sequential in pages, so one
task walking 64 pages exposes 64 dependent rescales. It is **1**, because at every batch the
decode contract allows, splitting costs more than that chain is worth. Composed attention
layer, a2a3, median of 50 rounds:

| batch, context (pages) | splits **1** | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 16, 320 (3) | **524.4** | 617.6 | 716.3 | 907.2 |
| 16, 1024 (8) | **572.6** | 625.6 | 747.8 | 972.5 |
| 16, 4096 (32) | **793.7** | 830.6 | 842.1 | 985.7 |
| 16, 8192 (64) | **1103.3** | 1136.2 | 1166.0 | 1223.1 |
| 48, 320 (3) | **724.0** | 1013.9 | 1301.7 | 1877.0 |
| 48, 4096 (32) | **1413.2** | 1501.6 | 1551.2 | 1961.7 |

One split wins in all six, monotonically, and the margin **shrinks with context and grows
with batch** -- 1.366x at (16, 320) to 1.057x at (16, 8192), and 1.798x at (48, 320). The
trend across context is the one that would produce a crossover and it flattens instead, so
there is none inside the model's context.

**The batch axis dominates the context axis**, which is why the earlier measurement of this
constant said four. That one timed the kernel alone at batch 1 and 8, where four splits beat
one by 2.56x at 8192 context -- correctly, because at batch 1 the unsplit kernel runs `batch
* HKV` = 4 tasks on 48 cores and splitting is the only parallelism available. At batch 16
there are already 64 tasks, so the cores are full before any splitting and the merge pass
and partial traffic are pure cost. `decode_vector.check_batch` requires a multiple of 16, so
the region that chose four is a region the decode path cannot present.

Splitting slightly **improves** accuracy, because each partial chain accumulates fewer BF16
rescales: 2.404e-03 at one split against 2.296e-03 at eight, same shape. One split is
therefore the worse end of a 4.7% spread in that statistic, which the composed gate measures
against the quantisation floor rather than in isolation.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")                     # decode sequences, one query row each
KVROWS_DYN = pl.dynamic("KVROWS_DYN")           # rows of the paged KV cache

# model config
HQ = QWEN3_8_27B.num_attention_heads            # 24 query heads
HKV = QWEN3_8_27B.num_key_value_heads           # 4 KV heads
D = QWEN3_8_27B.head_dim                        # 256
GROUP = HQ // HKV                               # query heads sharing one KV head
Q_WIDTH = HQ * D
KV_WIDTH = HKV * D

# validation
# one sequence, then the serving batch; a task is one sequence and KV head.
DEFAULT_CASES = (1, 4)

# tiling
PAGE = 128              # KV page size; the serving layer's choice, not the kernel's
SPLITS = 1              # page ranges per sequence; see the table above
Q_PAD = 16              # the cube's M fractal: GROUP real rows live in a 16-row tile

MASK_NEG = -1.0e30      # additive mask; its exp underflows to exactly zero
# A one-column FP32 store has a four-byte row, under ptoas's 32-byte minimum, so
# the split partials carry `m` and `l` broadcast across this many columns.
ML_COLS = 16


def build_tail_mask(page: int = PAGE):
    """`[page + 1, page]` FP32: row `v` opens columns `< v`. Row `page` is all open."""
    import torch

    cols = torch.arange(page).reshape(1, page)
    valid = torch.arange(page + 1).reshape(page + 1, 1)
    return torch.where(cols < valid, 0.0, MASK_NEG).float()


# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True


def build_kernel(max_pages: int, page: int = PAGE, splits: int = SPLITS,
                 inline: bool = False):
    """Decode attention for a cache of at most `max_pages` pages per sequence.

    `splits` partitions each sequence's pages across that many tasks, which is the only
    way to fill the machine at small batch: without it the task count is
    `batch * HKV`, so **batch 1 has four tasks** and reaches 13.5% of the copy roof
    against the 65% that batch 16 reaches with 64. Each split produces its own running
    `(m, l, acc)` and a second pass merges them, which is exact: the online-softmax
    state composes, so the merge is the same rescale the loop already does.
    """
    if page % 16:
        raise ValueError(f"page={page} must be a multiple of 16, the cube's fractal")
    if splits < 1:
        raise ValueError(f"splits={splits} must be at least 1")
    if splits > 1:
        return _build_split_kernel(max_pages, page, splits, inline)

    def _gdn_decode_attention(
        q: pl.Tensor[[B_DYN, Q_WIDTH], pl.BF16],
        k_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        v_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        block_table: pl.Tensor[[B_DYN, max_pages], pl.INT32],
        seq_lens: pl.Tensor[[B_DYN], pl.INT32],
        tail_mask: pl.Tensor[[page + 1, page], pl.FP32],
        o: pl.Out[pl.Tensor[[B_DYN, Q_WIDTH], pl.BF16]],
    ):
        """`seq_lens[i]` must be at least 1 and at most `max_pages * page`."""
        q.bind_dynamic(0, B_DYN)
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        block_table.bind_dynamic(0, B_DYN)
        seq_lens.bind_dynamic(0, B_DYN)
        o.bind_dynamic(0, B_DYN)
        b_dim = pl.tensor.dim(q, 0)
        # A head is one row of this view, so the GROUP heads sharing a KV head are a
        # contiguous row block and the query tile is a plain 2D slice.
        q_rows = pl.reshape(q, [b_dim * HQ, D])
        o_rows = pl.reshape(o, [b_dim * HQ, D])

        for task in pl.spmd(b_dim * HKV, name_hint="decode_attention",
                            allow_early_resolve=EARLY_RESOLVE):
            i = task // HKV
            j = task % HKV
            length = pl.cast(pl.read(seq_lens, [i]), pl.INDEX)
            pages = (length + page - 1) // page
            qr0 = i * HQ + j * GROUP
            kd0 = j * D
            q_tile = pl.slice(q_rows, [Q_PAD, D], [qr0, 0], valid_shape=[GROUP, D])

            # The running max and sum are [Q_PAD, 1] columns seeded from a reduction,
            # so they take the layout the reductions produce.
            zero_scores = pl.full([Q_PAD, page], dtype=pl.FP32, value=0.0)
            zero_col = pl.row_max(zero_scores)
            m_init = pl.add(zero_col, MASK_NEG)
            acc_init = pl.full([Q_PAD, D], dtype=pl.FP32, value=0.0)

            for p, (m_run, l_run, acc_run) in pl.range(
                    pages, init_values=(m_init, zero_col, acc_init)):
                phys = pl.cast(pl.read(block_table, [i, p]), pl.INDEX)
                r0 = phys * page
                k_tile = k_pages[r0 : r0 + page, kd0 : kd0 + D]
                scores = pl.matmul(q_tile, k_tile, b_trans=True, out_dtype=pl.FP32)
                # Row `page` of the table is all open, so a full page needs no branch.
                valid = pl.min(page, length - p * page)
                masked = pl.col_expand_add(scores, tail_mask[valid : valid + 1, 0:page])
                m_cur = pl.row_max(masked)
                m_new = pl.maximum(m_run, m_cur)
                alpha = pl.exp(pl.sub(m_run, m_new))
                probs = pl.exp(pl.row_expand_sub(masked, m_new))
                probs_bf = pl.cast(probs, target_type=pl.BF16, mode="rint")
                l_cur = pl.row_sum(pl.cast(probs_bf, target_type=pl.FP32))
                l_new = pl.add(pl.mul(l_run, alpha), l_cur)
                v_tile = v_pages[r0 : r0 + page, kd0 : kd0 + D]
                pv = pl.matmul(probs_bf, v_tile, out_dtype=pl.FP32)
                acc_new = pl.add(pl.row_expand_mul(acc_run, alpha), pv)
                m_end, l_end, acc_end = pl.yield_(m_new, l_new, acc_new)

            ctx = pl.row_expand_mul(acc_end, pl.recip(l_end, high_precision=True))
            o_rows[qr0 : qr0 + GROUP, 0:D] = pl.cast(
                pl.slice(ctx, [GROUP, D], [0, 0]), target_type=pl.BF16, mode="rint")
        return o

    return (pl.jit.inline(_gdn_decode_attention) if inline
            else pl.jit(_gdn_decode_attention))


def _build_split_kernel(max_pages: int, page: int, splits: int, inline: bool):
    """Two passes: partial softmax states per page range, then an exact merge.

    The partial `m` and `l` are `[Q_PAD, 1]` columns, and a one-column FP32 store has a
    four-byte row, which ptoas rejects on its 32-byte minimum. They are therefore stored
    broadcast across 16 columns and read back with `pl.row_max`: every column holds the
    same value, so the maximum is that value exactly, and nothing needs a width-one tile
    view.

    A split whose page range is empty -- which happens whenever `splits` exceeds a
    sequence's page count -- stores its seeds, `m = MASK_NEG` and `l = 0`. The merge then
    weights it by `exp(MASK_NEG - m*)`, which underflows to zero, so an empty split
    contributes nothing and needs no guard.
    """
    def _gdn_decode_attention_split(
        q: pl.Tensor[[B_DYN, Q_WIDTH], pl.BF16],
        k_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        v_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        block_table: pl.Tensor[[B_DYN, max_pages], pl.INT32],
        seq_lens: pl.Tensor[[B_DYN], pl.INT32],
        tail_mask: pl.Tensor[[page + 1, page], pl.FP32],
        o: pl.Out[pl.Tensor[[B_DYN, Q_WIDTH], pl.BF16]],
    ):
        """`seq_lens[i]` must be at least 1 and at most `max_pages * page`."""
        q.bind_dynamic(0, B_DYN)
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        block_table.bind_dynamic(0, B_DYN)
        seq_lens.bind_dynamic(0, B_DYN)
        o.bind_dynamic(0, B_DYN)
        b_dim = pl.tensor.dim(q, 0)
        q_rows = pl.reshape(q, [b_dim * HQ, D])
        o_rows = pl.reshape(o, [b_dim * HQ, D])

        part_rows = b_dim * HKV * splits * Q_PAD
        acc_part = pl.create_tensor([part_rows, D], dtype=pl.FP32)
        m_part = pl.create_tensor([part_rows, ML_COLS], dtype=pl.FP32)
        l_part = pl.create_tensor([part_rows, ML_COLS], dtype=pl.FP32)

        with pl.spmd(b_dim * HKV * splits, name_hint="decode_attention_split",
                     allow_early_resolve=EARLY_RESOLVE) as part_tid:
            task = pl.tile.get_block_idx()
            i = task // (HKV * splits)
            j = (task // splits) % HKV
            s = task % splits
            length = pl.cast(pl.read(seq_lens, [i]), pl.INDEX)
            pages = (length + page - 1) // page
            per_split = (pages + splits - 1) // splits
            p_lo = s * per_split
            n_pages = pl.max(pl.min(per_split, pages - p_lo), 0)

            qr0 = i * HQ + j * GROUP
            kd0 = j * D
            q_tile = pl.slice(q_rows, [Q_PAD, D], [qr0, 0], valid_shape=[GROUP, D])
            zero_scores = pl.full([Q_PAD, page], dtype=pl.FP32, value=0.0)
            zero_col = pl.row_max(zero_scores)
            m_init = pl.add(zero_col, MASK_NEG)
            acc_init = pl.full([Q_PAD, D], dtype=pl.FP32, value=0.0)

            for pp, (m_run, l_run, acc_run) in pl.range(
                    n_pages, init_values=(m_init, zero_col, acc_init)):
                p = p_lo + pp
                phys = pl.cast(pl.read(block_table, [i, p]), pl.INDEX)
                r0 = phys * page
                k_tile = k_pages[r0 : r0 + page, kd0 : kd0 + D]
                scores = pl.matmul(q_tile, k_tile, b_trans=True, out_dtype=pl.FP32)
                valid = pl.min(page, length - p * page)
                masked = pl.col_expand_add(scores, tail_mask[valid : valid + 1, 0:page])
                m_cur = pl.row_max(masked)
                m_new = pl.maximum(m_run, m_cur)
                alpha = pl.exp(pl.sub(m_run, m_new))
                probs = pl.exp(pl.row_expand_sub(masked, m_new))
                probs_bf = pl.cast(probs, target_type=pl.BF16, mode="rint")
                l_cur = pl.row_sum(pl.cast(probs_bf, target_type=pl.FP32))
                l_new = pl.add(pl.mul(l_run, alpha), l_cur)
                v_tile = v_pages[r0 : r0 + page, kd0 : kd0 + D]
                pv = pl.matmul(probs_bf, v_tile, out_dtype=pl.FP32)
                acc_new = pl.add(pl.row_expand_mul(acc_run, alpha), pv)
                m_end, l_end, acc_end = pl.yield_(m_new, l_new, acc_new)

            pr0 = ((i * HKV + j) * splits + s) * Q_PAD
            ones_ml = pl.full([Q_PAD, ML_COLS], dtype=pl.FP32, value=1.0)
            acc_part[pr0 : pr0 + Q_PAD, 0:D] = acc_end
            m_part[pr0 : pr0 + Q_PAD, 0:ML_COLS] = pl.row_expand_mul(ones_ml, m_end)
            l_part[pr0 : pr0 + Q_PAD, 0:ML_COLS] = pl.row_expand_mul(ones_ml, l_end)

        with pl.spmd(b_dim * HKV, name_hint="decode_attention_merge", deps=[part_tid],
                     allow_early_resolve=EARLY_RESOLVE):
            task = pl.tile.get_block_idx()
            i = task // HKV
            j = task % HKV
            base = (i * HKV + j) * splits * Q_PAD
            zero_scores = pl.full([Q_PAD, page], dtype=pl.FP32, value=0.0)
            zero_col = pl.row_max(zero_scores)
            m_init = pl.add(zero_col, MASK_NEG)
            acc_init = pl.full([Q_PAD, D], dtype=pl.FP32, value=0.0)
            for sp, (m_run, l_run, acc_run) in pl.range(
                    splits, init_values=(m_init, zero_col, acc_init)):
                sr0 = base + sp * Q_PAD
                # Every column holds the same value, so the maximum is that value.
                m_s = pl.row_max(pl.slice(m_part, [Q_PAD, ML_COLS], [sr0, 0]))
                l_s = pl.row_max(pl.slice(l_part, [Q_PAD, ML_COLS], [sr0, 0]))
                acc_s = pl.slice(acc_part, [Q_PAD, D], [sr0, 0])
                m_new = pl.maximum(m_run, m_s)
                alpha = pl.exp(pl.sub(m_run, m_new))
                beta = pl.exp(pl.sub(m_s, m_new))
                l_new = pl.add(pl.mul(l_run, alpha), pl.mul(l_s, beta))
                acc_new = pl.add(pl.row_expand_mul(acc_run, alpha),
                                 pl.row_expand_mul(acc_s, beta))
                m_end, l_end, acc_end = pl.yield_(m_new, l_new, acc_new)

            qr0 = i * HQ + j * GROUP
            ctx = pl.row_expand_mul(acc_end, pl.recip(l_end, high_precision=True))
            o_rows[qr0 : qr0 + GROUP, 0:D] = pl.cast(
                pl.slice(ctx, [GROUP, D], [0, 0]), target_type=pl.BF16, mode="rint")
        return o

    return (pl.jit.inline(_gdn_decode_attention_split) if inline
            else pl.jit(_gdn_decode_attention_split))


def golden_fn(tensors):
    """The float64 oracle, through `reference.decode_gqa_attention`.

    `q` already carries the `head_dim ** -0.5` scale, so the reference takes scale 1.
    """
    import reference

    out = reference.decode_gqa_attention(
        tensors["q"], tensors["k_pages"], tensors["v_pages"], tensors["block_table"],
        tensors["seq_lens"], HQ, HKV, D, PAGE, scale=1.0)
    tensors["o"].copy_(out.to(tensors["o"].dtype))


def build_tensor_specs(batch: int = 4, context: int = 320, max_pages: int = 8,
                       pool_pages: int | None = None):
    """A batch of sequences whose pages are scattered through the pool.

    The block table is deliberately not the identity: each sequence's pages are placed
    at a stride through the pool, so a kernel that ignored the table and read
    `sequence * max_pages` would fail. `context` is not a multiple of `PAGE` either,
    so the tail mask is exercised rather than skipped.
    """
    import torch
    from golden import TensorSpec

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    pages = (context + PAGE - 1) // PAGE
    if pages > max_pages:
        raise ValueError(f"context={context} needs {pages} pages, over max_pages={max_pages}")
    pool_pages = batch * max_pages if pool_pages is None else pool_pages
    gen = torch.Generator().manual_seed(23)

    table = torch.zeros(batch, max_pages, dtype=torch.int32)
    for s in range(batch):
        for p in range(pages):
            # A stride through the pool rather than a run: adjacent logical pages are
            # not adjacent physically, which is what a real allocator produces.
            table[s, p] = (p * batch + s) % pool_pages
    lens = torch.full((batch,), context, dtype=torch.int32)

    return [
        TensorSpec("q", [batch, Q_WIDTH], bf,
                   init_value=lambda: (torch.randn(batch, Q_WIDTH, generator=gen)
                                       * (D ** -0.5)).to(bf)),
        TensorSpec("k_pages", [pool_pages * PAGE, KV_WIDTH], bf,
                   init_value=lambda: torch.randn(pool_pages * PAGE, KV_WIDTH,
                                                  generator=gen).to(bf)),
        TensorSpec("v_pages", [pool_pages * PAGE, KV_WIDTH], bf,
                   init_value=lambda: torch.randn(pool_pages * PAGE, KV_WIDTH,
                                                  generator=gen).to(bf)),
        TensorSpec("block_table", [batch, max_pages], i32, init_value=lambda: table),
        TensorSpec("seq_lens", [batch], i32, init_value=lambda: lens),
        TensorSpec("tail_mask", [PAGE + 1, PAGE], f32, init_value=lambda: build_tail_mask()),
        TensorSpec("o", [batch, Q_WIDTH], bf),
    ]


def _rel_ok(threshold: float, label: str):
    def compare(actual, expected, **_kwargs):
        num = (actual.double() - expected.double()).norm()
        den = expected.double().norm()
        rel = float(num / den) if float(den) > 0 else float(num)
        peak = float((actual.double() - expected.double()).abs().max())
        detail = f"{label} rel {rel:.3e} (<= {threshold:.0e}), peak abs {peak:.3e}"
        ok = rel <= threshold
        return ok, "" if ok else detail

    return compare


if __name__ == "__main__":
    import argparse

    import torch
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--context", type=int, default=320)
    parser.add_argument("--max-pages", type=int, default=None,
                        help="pages per sequence; derived from --context when omitted, "
                             "because the two disagreeing is only ever a mistake")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--splits", type=int, default=SPLITS)
    parser.add_argument("--rounds", type=int, default=0)
    args = parser.parse_args()
    cases = [args.batch] if args.batch else list(DEFAULT_CASES)
    max_pages = args.max_pages or (args.context + PAGE - 1) // PAGE
    if max_pages * PAGE < args.context:
        parser.error(f"--max-pages {max_pages} holds {max_pages * PAGE} tokens, "
                     f"under --context {args.context}")
    torch.set_num_threads(min(torch.get_num_threads(), 32))
    torch.backends.mkldnn.enabled = False
    if args.rounds:
        import os
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    for batch in cases:
        print(f"[case] decode_attention batch={batch}", flush=True)
        result = run(
            fn=build_kernel(max_pages, splits=args.splits),
            specs=build_tensor_specs(batch, args.context, max_pages),
            golden_fn=None if args.rounds else golden_fn,
            compile_only=args.compile_only,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                        device_id=args.device),
            rtol=2e-2, atol=2e-2,
            compare_fn=None if args.rounds else {"o": _rel_ok(6e-3, "o")},
        )
        if args.rounds and result.bench is not None:
            import statistics
            s = [x for x in result.bench.per_round("effective") if x > 0]
            if s:
                kv = 2 * args.context * KV_WIDTH * 2 * batch
                med = statistics.median(s)
                print(f"[bench] batch={batch} context={args.context} "
                      f"median={med:.1f} us  KV {kv / 1e6:.2f} MB -> {kv / med / 1e3:.1f} GB/s",
                      flush=True)
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at batch={batch}")
