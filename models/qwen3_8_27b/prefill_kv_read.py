# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stage the paged attention cache as contiguous KV for a prefill chunk."""

import pypto.language as pl

from config import KV_PAGE_SIZE, QWEN3_8_27B
from prefill_kv_write import KVROWS_DYN

# model config
HKV = QWEN3_8_27B.num_key_value_heads
D = QWEN3_8_27B.head_dim
KV_WIDTH = HKV * D

# validation
DEFAULT_CASES = ((256, 0), (256, 4), (512, 4))

# tiling
ROW_TILE = 16


def build_kernel(t: int, max_pages: int, page: int = KV_PAGE_SIZE,
                 row_tile: int = ROW_TILE, inline: bool = False):
    """Gather all cached pages through the current chunk into a contiguous buffer."""
    if t % page or page % row_tile:
        raise ValueError(f"t={t} must divide by page={page}, and page must divide by row_tile={row_tile}")
    if max_pages * page < t:
        raise ValueError(f"max_pages={max_pages} cannot hold t={t}")

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_prefill_kv_read(
        k_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        v_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        block_table: pl.Tensor[[1, max_pages], pl.INT32],
        page_base: pl.Scalar[pl.INDEX],
        layer_base: pl.Scalar[pl.INDEX],
        k_all: pl.InOut[pl.Tensor[[max_pages * page, KV_WIDTH], pl.BF16]],
        v_all: pl.InOut[pl.Tensor[[max_pages * page, KV_WIDTH], pl.BF16]],
    ):
        """`page_base` is the number of cache pages preceding the chunk."""
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        pages = page_base + t // page
        for p in pl.spmd(pages, name_hint="prefill_kv_read"):
            phys = pl.cast(pl.read(block_table, [0, p]), pl.INDEX)
            src = layer_base + phys * page
            dst = p * page
            for r in pl.range(0, page, row_tile):
                k_all[dst + r : dst + r + row_tile, :] = k_pages[src + r : src + r + row_tile, :]
                v_all[dst + r : dst + r + row_tile, :] = v_pages[src + r : src + r + row_tile, :]
        return k_all, v_all

    return qwen3_prefill_kv_read


def build_packed_kernel(rows: int, max_pages: int, page: int = KV_PAGE_SIZE,
                        row_tile: int = ROW_TILE, inline: bool = False):
    """Gather each packed row's visible pages into its own contiguous scratch band."""
    if rows < 1 or max_pages < 2 or page % row_tile:
        raise ValueError("packed KV gather dimensions are invalid")

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_prefill_kv_read_packed(
        k_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        v_pages: pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16],
        block_table: pl.Tensor[[rows, max_pages], pl.INT32],
        visible_pages: pl.Tensor[[rows], pl.INT32],
        layer_base: pl.Scalar[pl.INDEX],
        k_all: pl.InOut[pl.Tensor[[rows * max_pages * page, KV_WIDTH], pl.BF16]],
        v_all: pl.InOut[pl.Tensor[[rows * max_pages * page, KV_WIDTH], pl.BF16]],
    ):
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        for owner in pl.spmd(rows, name_hint="prefill_kv_read_packed"):
            count = pl.cast(pl.read(visible_pages, [owner]), pl.INDEX)
            for p in pl.range(count):
                phys = pl.cast(pl.read(block_table, [owner, p]), pl.INDEX)
                src = layer_base + phys * page
                dst = (owner * max_pages + p) * page
                for r in pl.range(0, page, row_tile):
                    k_all[dst + r : dst + r + row_tile, :] = k_pages[src + r : src + r + row_tile, :]
                    v_all[dst + r : dst + r + row_tile, :] = v_pages[src + r : src + r + row_tile, :]
        return k_all, v_all

    return qwen3_prefill_kv_read_packed


def _inputs(t: int, page_base: int, max_pages: int, page: int):
    import torch

    gen = torch.Generator().manual_seed(82)
    first_phys = 3
    rows = (first_phys + max_pages) * page
    table = (torch.randperm(max_pages, generator=gen) + first_phys).reshape(1, max_pages).int()
    return {
        "k_pages": torch.randn(rows, KV_WIDTH, generator=gen).to(torch.bfloat16),
        "v_pages": torch.randn(rows, KV_WIDTH, generator=gen).to(torch.bfloat16),
        "block_table": table,
        "k_all": torch.full((max_pages * page, KV_WIDTH), -99.0, dtype=torch.bfloat16),
        "v_all": torch.full((max_pages * page, KV_WIDTH), -99.0, dtype=torch.bfloat16),
    }


def build_tensor_specs(t: int, page_base: int, max_pages: int | None = None,
                       page: int = KV_PAGE_SIZE):
    import torch
    from golden import ScalarSpec, TensorSpec

    max_pages = max_pages if max_pages is not None else page_base + t // page + 2
    if max_pages < page_base + t // page:
        raise ValueError("the block table does not cover the chunk")
    values = _inputs(t, page_base, max_pages, page)
    rows = values["k_pages"].shape[0]
    return [
        TensorSpec("k_pages", [rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["k_pages"]),
        TensorSpec("v_pages", [rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["v_pages"]),
        TensorSpec("block_table", [1, max_pages], torch.int32,
                   init_value=lambda: values["block_table"]),
        ScalarSpec("page_base", torch.int64, page_base, compile_runtime=True),
        ScalarSpec("layer_base", torch.int64, 0, compile_runtime=True),
        TensorSpec("k_all", [max_pages * page, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["k_all"]),
        TensorSpec("v_all", [max_pages * page, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["v_all"]),
    ]


def _packed_inputs(t: int, starts: tuple[int, ...], bases: tuple[int, ...],
                   max_pages: int, page: int):
    import torch

    import prefill_kv_write

    values = prefill_kv_write._packed_inputs(t, starts, bases, max_pages, page)
    ends = (*starts[1:], t)
    values["visible_pages"] = torch.tensor(
        [base + (end - start) // page for base, start, end in zip(bases, starts, ends)],
        dtype=torch.int32,
    )
    shape = (len(starts) * max_pages * page, KV_WIDTH)
    values["k_all"] = torch.full(shape, -99.0, dtype=torch.bfloat16)
    values["v_all"] = torch.full(shape, -99.0, dtype=torch.bfloat16)
    values["layer_base"] = torch.tensor(0, dtype=torch.int64)
    return values


def build_packed_tensor_specs(t: int = 768, starts: tuple[int, ...] = (0, 256),
                              bases: tuple[int, ...] = (0, 4), max_pages: int = 10,
                              page: int = KV_PAGE_SIZE):
    import torch
    from golden import ScalarSpec, TensorSpec

    values = _packed_inputs(t, starts, bases, max_pages, page)
    rows = len(starts)
    pool_rows = values["k_pages"].shape[0]
    scratch_rows = rows * max_pages * page
    return [
        TensorSpec("k_pages", [pool_rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["k_pages"]),
        TensorSpec("v_pages", [pool_rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["v_pages"]),
        TensorSpec("block_table", [rows, max_pages], torch.int32,
                   init_value=lambda: values["block_table"]),
        TensorSpec("visible_pages", [rows], torch.int32,
                   init_value=lambda: values["visible_pages"]),
        ScalarSpec("layer_base", torch.int64, 0, compile_runtime=True),
        TensorSpec("k_all", [scratch_rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["k_all"]),
        TensorSpec("v_all", [scratch_rows, KV_WIDTH], torch.bfloat16,
                   init_value=lambda: values["v_all"]),
    ]


def golden_qwen3_prefill_kv_read_packed(tensors):
    import reference

    table = tensors["block_table"]
    max_pages = table.shape[1]
    base = int(tensors["layer_base"])
    for owner in range(table.shape[0]):
        count = int(tensors["visible_pages"][owner])
        if not 0 <= count <= max_pages:
            raise ValueError("packed KV visible page count exceeds block table")
        rows = reference.kv_page_rows(table, owner, count * KV_PAGE_SIZE, KV_PAGE_SIZE)
        start = owner * max_pages * KV_PAGE_SIZE
        end = start + count * KV_PAGE_SIZE
        tensors["k_all"][start:end].copy_(tensors["k_pages"][base + rows])
        tensors["v_all"][start:end].copy_(tensors["v_pages"][base + rows])


def packed_owner_control() -> None:
    import torch
    import prefill_kv_write

    values = _packed_inputs(768, (0, 256), (0, 4), 10, KV_PAGE_SIZE)
    truth = {name: value.clone() for name, value in values.items()}
    wrong = {name: value.clone() for name, value in values.items()}
    wrong["block_table"] = wrong["block_table"].flip(0)
    golden_qwen3_prefill_kv_read_packed(truth)
    golden_qwen3_prefill_kv_read_packed(wrong)
    if torch.equal(truth["k_all"], wrong["k_all"]):
        raise RuntimeError("packed KV owner control cannot detect wrong cache reads")
    roundtrip = _packed_inputs(768, (0, 256), (0, 4), 10, KV_PAGE_SIZE)
    prefill_kv_write.golden_qwen3_prefill_kv_write_packed(roundtrip)
    golden_qwen3_prefill_kv_read_packed(roundtrip)
    for owner, start, end, first in ((0, 0, 256, 0), (1, 256, 768, 4)):
        offset = (owner * 10 + first) * KV_PAGE_SIZE
        length = end - start
        if not torch.equal(roundtrip["k_all"][offset : offset + length],
                           roundtrip["k_rot"][start:end]):
            raise RuntimeError(f"packed K roundtrip failed for owner {owner}")
        if not torch.equal(roundtrip["v_all"][offset : offset + length],
                           roundtrip["v"][start:end]):
            raise RuntimeError(f"packed V roundtrip failed for owner {owner}")


def golden_qwen3_prefill_kv_read(t: int):
    import reference

    def golden(tensors):
        length = int(tensors["page_base"]) * KV_PAGE_SIZE + t
        rows = reference.kv_page_rows(tensors["block_table"], 0, length, KV_PAGE_SIZE)
        base = int(tensors["layer_base"])
        tensors["k_all"][:length].copy_(tensors["k_pages"][base + rows])
        tensors["v_all"][:length].copy_(tensors["v_pages"][base + rows])

    return golden


def compare_exact(actual, expected, **_kwargs):
    ok = bool((actual == expected).all())
    return ok, "" if ok else "    paged cache gather differs, including an untouched output row"


def control_has_teeth(t: int, page_base: int, max_pages: int, page: int = KV_PAGE_SIZE):
    import reference

    values = _inputs(t, page_base, max_pages, page)
    length = page_base * page + t
    table = values["block_table"]
    rows = reference.kv_page_rows(table, 0, length, page)
    changed = table.clone()
    changed[0, 0], changed[0, 1] = table[0, 1], table[0, 0]
    moved = reference.kv_page_rows(changed, 0, length, page)
    return not bool((values["k_pages"][rows] == values["k_pages"][moved]).all())


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--page-base", type=int, default=0)
    parser.add_argument("--packed", action="store_true", default=False,
                        help="run the two-row packed cache case")
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    cases = [] if args.packed else ([(args.seq_len, args.page_base)]
                                   if args.seq_len else DEFAULT_CASES)
    for t, page_base in cases:
        max_pages = page_base + t // KV_PAGE_SIZE + 2
        if not args.compile_only and not control_has_teeth(t, page_base, max_pages):
            raise SystemExit("the scrambled block-table control did not move the oracle")
        print(f"[case] prefill_kv_read t={t} page_base={page_base}", flush=True)
        result = run(
            fn=build_kernel(t, max_pages),
            specs=build_tensor_specs(t, page_base, max_pages),
            golden_fn=None if args.compile_only else golden_qwen3_prefill_kv_read(t),
            compare_fn=None if args.compile_only else {"k_all": compare_exact,
                                                        "v_all": compare_exact},
            config=dict(backend_type=args.backend_type,
                        execution_mode=args.execution_mode, device_id=args.device),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at t={t} page_base={page_base}")

    if args.packed or args.seq_len is None:
        t, starts, bases, max_pages = 768, (0, 256), (0, 4), 10
        if not args.compile_only:
            packed_owner_control()
        print(f"[case] prefill_kv_read packed t={t} starts={starts} bases={bases}", flush=True)
        result = run(
            fn=build_packed_kernel(rows=len(starts), max_pages=max_pages),
            specs=build_packed_tensor_specs(t=t, starts=starts, bases=bases,
                                            max_pages=max_pages),
            golden_fn=None if args.compile_only else golden_qwen3_prefill_kv_read_packed,
            compare_fn=None if args.compile_only else {"k_all": compare_exact,
                                                        "v_all": compare_exact},
            config=dict(backend_type=args.backend_type,
                        execution_mode=args.execution_mode, device_id=args.device),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit("FAILED at packed KV read")
