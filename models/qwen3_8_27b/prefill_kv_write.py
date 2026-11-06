# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Write a prefill chunk's K and V into the paged cache a decode step reads.

One sequence, `t` consecutive positions starting at `page_base * KV_PAGE_SIZE`, into
the rows the block table names:

    row = block_table[0, page_base + p] * KV_PAGE_SIZE   for logical page p

**A task is one page**, so it copies a page-contiguous `[KV_PAGE_SIZE, KV_WIDTH]` run
rather than scattering single rows: 64 tasks a layer at `t = 8192` against 8192, which
is the difference between a bandwidth cost and a per-task one. The page granularity is
free -- the runner pads every packed row to a multiple of 256 tokens, and the page is
128 -- so a chunk boundary is page-aligned and no page is ever split between two
chunks. `build_kernel` refuses a `t` that is not a multiple of the page, because that
case would drop the tail quietly.

K is the post-norm, post-RoPE `k_rot`, which is what `decode_attention` expects to find
cached; V is the projection output untouched. Both come out of `gated_attention` as
`[t, KV_WIDTH]` BF16, already the pool's row layout, so this kernel is a copy and its
gate is exact.
"""
import pypto.language as pl

from config import KV_PAGE_SIZE, QWEN3_8_27B

# Dynamic shape variables.
KVROWS_DYN = pl.dynamic("KVROWS_DYN")       # rows of the page pool

# model config
HKV = QWEN3_8_27B.num_key_value_heads
D = QWEN3_8_27B.head_dim
KV_WIDTH = HKV * D                          # one token's K or V, flat

# validation
# (tokens, first logical page): a chunk at the start of a sequence, then a resumed
# chunk whose pages sit past the ones an earlier chunk filled.
DEFAULT_CASES = ((256, 0), (512, 4))
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
ROW_TILE = 16           # tokens per copy; [16, 1024] BF16 is 32 KiB of UB


def build_kernel(t: int = BENCH_SEQ, max_pages: int = 64, page: int = KV_PAGE_SIZE,
                 row_tile: int = ROW_TILE, inline: bool = False):
    """The write for a chunk of `t` tokens into a cache of `max_pages` pages per sequence."""
    if t % page:
        raise ValueError(f"t={t} must be a multiple of the KV page {page}")
    if page % row_tile:
        raise ValueError(f"page={page} must be a multiple of row_tile {row_tile}")

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_prefill_kv_write(
        k_rot: pl.Tensor[[t, KV_WIDTH], pl.BF16],
        v: pl.Tensor[[t, KV_WIDTH], pl.BF16],
        block_table: pl.Tensor[[1, max_pages], pl.INT32],
        page_base: pl.Scalar[pl.INDEX],
        layer_base: pl.Scalar[pl.INDEX],
        k_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
    ):
        """`page_base` is the chunk's first logical page, `chunk_start // page`."""
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        for p in pl.spmd(t // page, name_hint="prefill_kv_write"):
            phys = pl.cast(pl.read(block_table, [0, page_base + p]), pl.INDEX)
            dst = layer_base + phys * page
            src = p * page
            for r in pl.range(0, page, row_tile):
                k_pages[dst + r : dst + r + row_tile, :] = k_rot[src + r : src + r + row_tile, :]
                v_pages[dst + r : dst + r + row_tile, :] = v[src + r : src + r + row_tile, :]
        return k_pages, v_pages

    return qwen3_prefill_kv_write


def build_packed_kernel(t: int, rows: int, max_pages: int, page: int = KV_PAGE_SIZE,
                        row_tile: int = ROW_TILE, inline: bool = False):
    """Scatter packed full pages to each row's own paged cache allocation.

    `row_base[token]` gives that row's start in the packed token buffer and
    `row_owner[token]` selects its block-table row. The caller supplies complete
    256-token rows, so no token in this entry is padding.
    """
    if t % (2 * page) or rows < 1 or rows > t // (2 * page):
        raise ValueError("packed KV rows need at least two whole pages each")
    if page % row_tile or rows * max_pages < t // page:
        raise ValueError("packed KV page or block-table extent is invalid")

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_prefill_kv_write_packed(
        k_rot: pl.Tensor[[t, KV_WIDTH], pl.BF16],
        v: pl.Tensor[[t, KV_WIDTH], pl.BF16],
        row_base: pl.Tensor[[t], pl.INT32],
        row_owner: pl.Tensor[[t], pl.INT32],
        page_bases: pl.Tensor[[rows], pl.INT32],
        block_table: pl.Tensor[[rows, max_pages], pl.INT32],
        layer_base: pl.Scalar[pl.INDEX],
        k_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
    ):
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        for packed_page in pl.spmd(t // page, name_hint="prefill_kv_write_packed"):
            t0 = packed_page * page
            owner = pl.cast(pl.read(row_owner, [t0]), pl.INDEX)
            start = pl.cast(pl.read(row_base, [t0]), pl.INDEX)
            first = pl.cast(pl.read(page_bases, [owner]), pl.INDEX)
            logical = first + (t0 - start) // page
            phys = pl.cast(pl.read(block_table, [owner, logical]), pl.INDEX)
            dst = layer_base + phys * page
            for r in pl.range(0, page, row_tile):
                k_pages[dst + r : dst + r + row_tile, :] = k_rot[t0 + r : t0 + r + row_tile, :]
                v_pages[dst + r : dst + r + row_tile, :] = v[t0 + r : t0 + r + row_tile, :]
        return k_pages, v_pages

    return qwen3_prefill_kv_write_packed


def pool_pages(page_base: int, t: int, page: int = KV_PAGE_SIZE) -> int:
    """Pages a sequence's table must carry to hold a chunk ending at this one."""
    return page_base + t // page


def _inputs(t: int, page_base: int, page: int, max_pages: int, first_page: int) -> dict:
    """The case's tensors, drawn once so the gate and its control share them.

    The table is a scattered permutation offset past `first_page`, so a kernel that
    ignored it and wrote `pos` would land on the wrong rows. The pool starts as noise
    rather than zeros, so a page the kernel fails to write keeps a value the oracle
    does not hold.
    """
    import torch

    gen = torch.Generator().manual_seed(42)
    rows = (first_page + max_pages) * page
    perm = torch.randperm(max_pages, generator=gen) + first_page
    return {
        "k_rot": torch.randn(t, KV_WIDTH, generator=gen).to(torch.bfloat16),
        "v": torch.randn(t, KV_WIDTH, generator=gen).to(torch.bfloat16),
        "block_table": perm.reshape(1, max_pages).to(torch.int32),
        "k_pages": torch.randn(rows, KV_WIDTH, generator=gen).to(torch.bfloat16),
        "v_pages": torch.randn(rows, KV_WIDTH, generator=gen).to(torch.bfloat16),
    }


def build_tensor_specs(t: int = BENCH_SEQ, page_base: int = 0, page: int = KV_PAGE_SIZE,
                       max_pages: int | None = None, first_page: int = 3):
    """The chunk, a scattered block table, and a pool pre-filled with other values."""
    import torch
    from golden import ScalarSpec, TensorSpec

    max_pages = max_pages if max_pages is not None else pool_pages(page_base, t, page)
    values = _inputs(t, page_base, page, max_pages, first_page)
    rows = values["k_pages"].shape[0]

    def fixed(name):
        return lambda: values[name]

    return [
        TensorSpec("k_rot", [t, KV_WIDTH], torch.bfloat16, init_value=fixed("k_rot")),
        TensorSpec("v", [t, KV_WIDTH], torch.bfloat16, init_value=fixed("v")),
        TensorSpec("block_table", [1, max_pages], torch.int32, init_value=fixed("block_table")),
        ScalarSpec("page_base", torch.int64, page_base, compile_runtime=True),
        ScalarSpec("layer_base", torch.int64, 0, compile_runtime=True),
        TensorSpec("k_pages", [rows, KV_WIDTH], torch.bfloat16, init_value=fixed("k_pages")),
        TensorSpec("v_pages", [rows, KV_WIDTH], torch.bfloat16, init_value=fixed("v_pages")),
    ]


def _packed_inputs(t: int, starts: tuple[int, ...], bases: tuple[int, ...],
                   max_pages: int, page: int, layer_base: int = 0) -> dict:
    import torch

    if not starts or starts[0] != 0 or len(starts) != len(bases):
        raise ValueError("packed KV rows need matching starts and page bases")
    ends = (*starts[1:], t)
    if any(start >= end or start % (2 * page) or (end - start) % (2 * page)
           for start, end in zip(starts, ends)):
        raise ValueError("packed KV rows must be positive and 256-token aligned")
    if any(base < 0 or base + (end - start) // page > max_pages
           for base, start, end in zip(bases, starts, ends)):
        raise ValueError("packed KV row exceeds its block table")
    if layer_base < 0 or layer_base % page:
        raise ValueError("packed KV layer offset must be nonnegative and page aligned")
    gen = torch.Generator().manual_seed(88)
    owners = torch.empty(t, dtype=torch.int32)
    row_base = torch.empty(t, dtype=torch.int32)
    for owner, (start, end) in enumerate(zip(starts, ends)):
        owners[start:end] = owner
        row_base[start:end] = start
    table = (torch.randperm(len(starts) * max_pages, generator=gen) + 3).reshape(
        len(starts), max_pages).int()
    pool_rows = layer_base + (len(starts) * max_pages + 3) * page
    return {
        "k_rot": torch.randn(t, KV_WIDTH, generator=gen).to(torch.bfloat16),
        "v": torch.randn(t, KV_WIDTH, generator=gen).to(torch.bfloat16),
        "row_base": row_base,
        "row_owner": owners,
        "page_bases": torch.tensor(bases, dtype=torch.int32),
        "block_table": table,
        "layer_base": torch.tensor(layer_base, dtype=torch.int64),
        "k_pages": torch.randn(pool_rows, KV_WIDTH, generator=gen).to(torch.bfloat16),
        "v_pages": torch.randn(pool_rows, KV_WIDTH, generator=gen).to(torch.bfloat16),
    }


def build_packed_tensor_specs(t: int = 768, starts: tuple[int, ...] = (0, 256),
                              bases: tuple[int, ...] = (0, 4), max_pages: int = 10,
                              page: int = KV_PAGE_SIZE, layer_base: int = 0):
    from golden import ScalarSpec, TensorSpec

    import torch

    values = _packed_inputs(t, starts, bases, max_pages, page, layer_base)
    rows = len(starts)
    pool_rows = values["k_pages"].shape[0]
    return [
        TensorSpec("k_rot", [t, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["k_rot"]),
        TensorSpec("v", [t, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["v"]),
        TensorSpec("row_base", [t], torch.int32,
                   init_value=lambda: values["row_base"]),
        TensorSpec("row_owner", [t], torch.int32,
                   init_value=lambda: values["row_owner"]),
        TensorSpec("page_bases", [rows], torch.int32,
                   init_value=lambda: values["page_bases"]),
        TensorSpec("block_table", [rows, max_pages], torch.int32,
                   init_value=lambda: values["block_table"]),
        ScalarSpec("layer_base", torch.int64, layer_base, compile_runtime=True),
        TensorSpec("k_pages", [pool_rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["k_pages"]),
        TensorSpec("v_pages", [pool_rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["v_pages"]),
    ]


def golden_qwen3_prefill_kv_write_packed(tensors):
    import reference

    owners = tensors["row_owner"]
    bases = tensors["row_base"]
    layer_base = int(tensors["layer_base"])
    for owner in range(tensors["block_table"].shape[0]):
        positions = (owners == owner).nonzero().flatten()
        if positions.numel() == 0:
            continue
        start = int(positions[0])
        end = int(positions[-1]) + 1
        if int(bases[start]) != start:
            raise ValueError("packed row base does not name its first token")
        reference.write_kv_pages(
            tensors["k_pages"][layer_base:], tensors["v_pages"][layer_base:],
            tensors["block_table"][owner : owner + 1],
            tensors["k_rot"][start:end], tensors["v"][start:end],
            int(tensors["page_bases"][owner]) * KV_PAGE_SIZE, KV_PAGE_SIZE,
        )


def packed_owner_control() -> None:
    """Swapping cache owners must move the pool writes without moving token data."""
    import torch

    values = _packed_inputs(768, (0, 256), (0, 4), 10, KV_PAGE_SIZE)
    truth = {name: value.clone() for name, value in values.items()}
    wrong = {name: value.clone() for name, value in values.items()}
    wrong["row_owner"][256:] = 0
    golden_qwen3_prefill_kv_write_packed(truth)
    golden_qwen3_prefill_kv_write_packed(wrong)
    if torch.equal(truth["k_pages"], wrong["k_pages"]):
        raise RuntimeError("packed KV owner control cannot detect wrong cache writes")


def golden_qwen3_prefill_kv_write(page: int = KV_PAGE_SIZE):
    """The oracle at one page size, which must be the one the kernel was built with."""

    def golden(tensors):
        import reference

        start_pos = int(tensors["page_base"]) * page
        base = int(tensors["layer_base"])
        reference.write_kv_pages(
            tensors["k_pages"][base:], tensors["v_pages"][base:], tensors["block_table"],
            tensors["k_rot"], tensors["v"], start_pos, page,
        )

    return golden


def control_has_teeth(t: int, page_base: int, page: int = KV_PAGE_SIZE,
                      max_pages: int | None = None, first_page: int = 3) -> bool:
    """Swapping two of the block table's pages must move the oracle's rows.

    The device comparison is exact, so only a kernel that used the table can pass it.
    This says the table it was given discriminates -- the scrambled control the paged
    reference makes, run on the gate's own inputs.
    """
    import torch

    import reference

    max_pages = max_pages if max_pages is not None else pool_pages(page_base, t, page)
    values = _inputs(t, page_base, page, max_pages, first_page)
    if t // page < 2:
        raise ValueError(f"the control needs at least two pages, got {t // page}")
    wrong = values["block_table"].clone()
    lo, hi = page_base, page_base + 1
    wrong[0, lo], wrong[0, hi] = wrong[0, hi].clone(), wrong[0, lo].clone()
    truth = values["k_pages"].clone()
    swapped = values["k_pages"].clone()
    reference.write_kv_pages(truth, values["v_pages"].clone(), values["block_table"],
                             values["k_rot"], values["v"], page_base * page, page)
    reference.write_kv_pages(swapped, values["v_pages"].clone(), wrong,
                             values["k_rot"], values["v"], page_base * page, page)
    return not torch.equal(truth, swapped)


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one chunk size instead of DEFAULT_CASES")
    parser.add_argument("--page-base", type=int, default=0, help="first logical page, with --seq-len")
    parser.add_argument("--row-tile", type=int, default=ROW_TILE)
    parser.add_argument("--packed", action="store_true", default=False,
                        help="run the two-row packed cache case")
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [] if args.packed else ([(args.seq_len, args.page_base)]
                                   if args.seq_len else list(DEFAULT_CASES))

    for t, page_base in cases:
        print(f"[case] prefill_kv_write t={t} page_base={page_base}", flush=True)
        specs = build_tensor_specs(t=t, page_base=page_base)
        max_pages = pool_pages(page_base, t)
        if not args.compile_only and not control_has_teeth(t, page_base):
            raise SystemExit(f"the block-table control is blind at t={t}")
        result = run(
            fn=build_kernel(t=t, max_pages=max_pages, row_tile=args.row_tile),
            specs=specs,
            golden_fn=golden_qwen3_prefill_kv_write(),
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                        device_id=args.device),
            rtol=0.0,
            atol=0.0,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at t={t} page_base={page_base}")

    packed_cases = ((768, (0, 256), (0, 4), 10), (768, (0, 256), (0, 0), 4))
    for t, starts, bases, max_pages in packed_cases if args.packed or args.seq_len is None else ():
        if not args.compile_only:
            packed_owner_control()
        print(f"[case] prefill_kv_write packed t={t} starts={starts} bases={bases}", flush=True)
        result = run(
            fn=build_packed_kernel(t=t, rows=len(starts), max_pages=max_pages,
                                   row_tile=args.row_tile),
            specs=build_packed_tensor_specs(t=t, starts=starts, bases=bases,
                                            max_pages=max_pages, layer_base=2 * KV_PAGE_SIZE),
            golden_fn=None if args.compile_only else golden_qwen3_prefill_kv_write_packed,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                        device_id=args.device),
            rtol=0.0,
            atol=0.0,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit("FAILED at packed KV write")
