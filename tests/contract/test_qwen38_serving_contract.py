# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8 serving capabilities and read/write cache ABI."""

import ast
from pathlib import Path

import pytest

from models.qwen3_8_27b import config
from models.qwen3_8_27b.serving_contract import QWEN3_8_27B_SERVING_CONTRACT as CONTRACT


@pytest.mark.parametrize("num_tokens", [2048, 4096, 32768])
def test_fused_prefill_accepts_complete_extents(num_tokens):
    CONTRACT.validate_prefill_tokens(num_tokens)


@pytest.mark.parametrize("num_tokens", [0, -1, 1, 2047, 2049, 32769, 34816])
def test_fused_prefill_rejects_padding_or_oversized_extents(num_tokens):
    with pytest.raises(ValueError):
        CONTRACT.validate_prefill_tokens(num_tokens)


@pytest.mark.parametrize("num_tokens", [True, 2048.0, "2048"])
def test_fused_prefill_rejects_non_integer_extents(num_tokens):
    with pytest.raises(TypeError):
        CONTRACT.validate_prefill_tokens(num_tokens)


def test_cache_page_size_matches_model_cache_addressing():
    assert CONTRACT.kv_page_tokens == config.KV_PAGE_SIZE
    assert CONTRACT.prefill_request_alignment % config.KV_PAGE_SIZE == 0
    assert CONTRACT.prefill_tile_tokens % CONTRACT.prefill_request_alignment == 0


@pytest.mark.parametrize(("module", "kernel", "host"), [
    ("prefill_text_fwd", "qwen3_prefill_text_fwd", "l3_prefill_fwd"),
    ("decode_fwd", "qwen3_decode_fwd", "l3_decode_fwd"),
])
def test_host_entry_preserves_argument_order_and_mutable_cache_directions(module, kernel, host):
    path = Path(config.__file__).parent / (module + ".py")
    nodes = {n.name: n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}
    host_args = nodes[host].args.args
    assert [arg.arg for arg in host_args] == [arg.arg for arg in nodes[kernel].args.args]
    forwarded = next(n for n in ast.walk(nodes[host]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
    assert [arg.id for arg in forwarded.args] == [arg.arg for arg in host_args]
    for args in (host_args, nodes[kernel].args.args):
        for arg in args:
            if arg.arg in ("conv_state", "final_state", "k_pages", "v_pages"):
                assert isinstance(arg.annotation, ast.Subscript)
                assert ast.unparse(arg.annotation.value) == "pl.InOut"
