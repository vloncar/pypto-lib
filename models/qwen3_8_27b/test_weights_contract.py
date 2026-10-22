# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host checks for the Qwen3.8 whole-checkpoint loader and text contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.qwen3_8_27b import reference, weights
from models.qwen3_8_27b.config import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    QWEN3_8_27B,
    Qwen38Config,
)
from models.qwen3_8_27b.contract import QWEN3_8_27B_TEXT_CONTRACT, make_text_contract


def tiny_config() -> Qwen38Config:
    """Two architecture-faithful layers small enough for real safetensors I/O."""
    return Qwen38Config(
        name="Qwen3.8-weight-test",
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        vocab_size=32,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        full_attention_interval=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        attention_bias=False,
    )


def _tensor(spec, generator: torch.Generator) -> torch.Tensor:
    if spec.dtype == "int8":
        return torch.randint(-127, 128, spec.shape, dtype=torch.int8, generator=generator)
    if spec.dtype == "fp32":
        if spec.is_zero_offset:
            return torch.zeros(spec.shape, dtype=torch.float32)
        return torch.rand(spec.shape, dtype=torch.float32, generator=generator) + 1e-4
    return torch.randn(spec.shape, dtype=torch.bfloat16, generator=generator)


def _write_fixture(root: Path, contract) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(42)
    source = {spec.name: _tensor(spec, generator) for spec in contract.weight_specs}
    names = list(source)
    split = len(names) // 2
    shard_names = (
        "quant_model_weights-00001-of-00002.safetensors",
        "quant_model_weights-00002-of-00002.safetensors",
    )
    shard_tensors = (
        {name: source[name] for name in names[:split]},
        {name: source[name] for name in names[split:]},
    )
    # Non-text tensors are legal and must not leak into the text-only load.
    shard_tensors[0]["model.visual.blocks.0.norm1.weight"] = torch.ones(4, dtype=torch.bfloat16)
    for shard, tensors in zip(shard_names, shard_tensors, strict=True):
        save_file(tensors, root / shard)

    weight_map = {
        name: shard_names[0] if position < split else shard_names[1] for position, name in enumerate(names)
    }
    weight_map["model.visual.blocks.0.norm1.weight"] = shard_names[0]
    (root / weights.ECO_INDEX).write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    return source, weight_map


def check_contract() -> None:
    contract = QWEN3_8_27B_TEXT_CONTRACT
    assert contract.layer_types == (LINEAR_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION) * 16
    assert contract.dims.rotary_dim == 64
    assert len(contract.weight_specs) == 1555
    assert len(contract.prepared_weight_specs) == 1203
    assert len(contract.layer_specs(0)) == 24
    assert len(contract.layer_specs(3)) == 25
    assert contract.memory.checkpoint_weight_bytes == 31_004_212_224
    assert contract.memory.resident_weight_bytes == 30_989_597_696
    assert contract.memory.kv_cache_bytes_per_token == 64 * 2**10
    assert contract.memory.max_kv_cache_bytes == 16 * 2**30
    assert contract.memory.gdn_state_bytes_per_sequence == 144 * 2**20
    assert contract.memory.max_context_headroom_bytes > 18 * 2**30
    assert QWEN3_8_27B.num_linear_layers == 48
    assert QWEN3_8_27B.num_full_attention_layers == 16


def check_quantization_recipe() -> None:
    source = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    actual_q, actual_scale, actual_offset = weights.modelslim_quantize_rows(source)
    assert actual_q.tolist() == [[-127, -64, 0, 64], [0, 0, 0, 0]]
    assert actual_scale[:, 0].tolist() == [0.0157470703125, 7.897615432739258e-07]
    assert actual_scale.shape == (2, 1)
    assert torch.count_nonzero(actual_offset) == 0

    # Q4 predates the deployment checkpoint and deliberately computes its
    # synthetic scale in FP32. Keep the two recipes visibly distinct.
    _q4_q, q4_scale = reference.quantize_rows(source)
    assert not torch.equal(actual_scale[:, 0], q4_scale)


def check_loader() -> None:
    from safetensors.torch import save_file

    contract = make_text_contract(tiny_config())
    with tempfile.TemporaryDirectory(prefix="qwen38_weights_") as directory:
        root = Path(directory)
        source, weight_map = _write_fixture(root, contract)
        summary = weights.checkpoint_summary(root, contract)
        assert summary["text_tensors"] == len(contract.weight_specs)
        assert summary["prepared_tensors"] == len(contract.prepared_weight_specs)
        assert summary["shards"] == 2

        prepared = weights.prepare_qwen38_weights(root, lambda tensor: tensor.clone(), contract=contract)
        assert prepared.layer_types == (LINEAR_ATTENTION, FULL_ATTENTION)
        assert len(prepared.tensors) == len(contract.prepared_weight_specs)
        assert len(prepared.layer(0)) == len(contract.layer_specs(0)) - 5
        assert len(prepared.layer(1)) == len(contract.layer_specs(1)) - 7
        assert not any(name.endswith("weight_offset") for name in prepared.tensors)
        assert not any(name.startswith("model.visual") for name in prepared.tensors)

        for spec in contract.prepared_weight_specs:
            actual = prepared.tensors[spec.name]
            expected = source[spec.name]
            if spec.role == "weight_scale":
                expected = expected.reshape(-1)
            assert torch.equal(actual, expected), spec.name

        # The loader must reject an asymmetric checkpoint instead of silently
        # treating a non-zero offset as the symmetric kernel format.
        offset_name = next(spec.name for spec in contract.weight_specs if spec.is_zero_offset)
        source[offset_name] = torch.ones_like(source[offset_name])
        shard_name = weight_map[offset_name]
        shard_names = [name for name, shard in weight_map.items() if shard == shard_name]
        tensors = {name: source[name] for name in shard_names if name in source}
        if shard_name.endswith("00001-of-00002.safetensors"):
            tensors["model.visual.blocks.0.norm1.weight"] = torch.ones(4, dtype=torch.bfloat16)
        save_file(tensors, root / shard_name)
        try:
            weights.prepare_qwen38_weights(root, lambda tensor: tensor, contract=contract)
        except ValueError as error:
            assert "offset is not zero" in str(error)
        else:
            raise AssertionError("non-zero symmetric weight_offset was accepted")


def check_resumed_fetch_verifies_content() -> None:
    """A resumed fetch must judge an existing file by content, not by length.

    Only a real 29 GiB download exercises this path otherwise, and it fails in
    both directions: trusting a corrupted shard of the right size, or rejecting
    a good one and re-downloading it. Both are expensive to discover for real.
    """
    body = b"safetensors payload" * 256
    digest = hashlib.sha256(body).hexdigest()
    with tempfile.TemporaryDirectory(prefix="qwen38_fetch_") as directory:
        shard = Path(directory) / "quant_model_weights-00001-of-00010.safetensors"
        shard.write_bytes(body)
        marker = weights._digest_marker(shard)

        assert weights._verified_locally(shard, len(body), digest)
        assert marker.is_file(), "a verified file must record its digest so the next run need not rehash"
        assert weights._verified_locally(shard, len(body), digest)

        assert not weights._verified_locally(shard, len(body) - 1, digest)
        assert not weights._verified_locally(shard, len(body), "de" * 32)
        assert not weights._verified_locally(Path(directory) / "absent", len(body), digest)

        # A stale or hand-written marker must not be believed over the bytes.
        marker.write_text("0" * 64 + "\n")
        assert weights._verified_locally(shard, len(body), digest)

        # Rewriting the file invalidates the record even at the same length,
        # so a corrupted shard is re-hashed and rejected rather than trusted.
        # The record is keyed on mtime, so state the assumption rather than
        # let a filesystem with coarse timestamps pass this by accident.
        before = shard.stat().st_mtime_ns
        corrupted = bytearray(body)
        corrupted[0] ^= 0xFF
        shard.write_bytes(bytes(corrupted))
        assert shard.stat().st_mtime_ns != before, "rewriting a file must change its mtime for this to hold"
        assert not weights._verified_locally(shard, len(body), digest), "same length, different content"

        # The same rule must not cost a re-download when the content is fine:
        # a byte-identical rewrite is re-hashed once and accepted.
        shard.write_bytes(body)
        assert weights._verified_locally(shard, len(body), digest)
        assert weights._verified_locally(shard, len(body), digest)


def check_reference_consumes_prepared_weights() -> None:
    """The loader's output must reach the reference with no adapter in between.

    ``test_model_reference.py`` gates the maths, but it feeds the reference a
    Transformers state dict, where every projection is a plain float tensor.
    The deployed checkpoint is not: eight of each layer's projections arrive as
    INT8 plus a separate flattened ``weight_scale``. Nothing else exercises
    that pairing, so a renamed scale or a changed ``dequantize`` indexing would
    break the real load path while every other host check stayed green.
    """
    contract = make_text_contract(tiny_config())
    with tempfile.TemporaryDirectory(prefix="qwen38_w8a8_ref_") as directory:
        root = Path(directory)
        _write_fixture(root, contract)
        prepared = weights.load_text_weights(root, contract=contract)

        quantized = [name for name, value in prepared.items() if value.dtype == torch.int8]
        assert quantized, "the fixture produced no INT8 weights, so this proves nothing"
        for name in quantized:
            scale = prepared[name.removesuffix(".weight") + ".weight_scale"]
            assert scale.ndim == 1, f"{name}: dequantize indexes a flat [out] scale, got {tuple(scale.shape)}"

        ids = reference.make_text_inputs(6, tiny_config(), seed=1)
        out = reference.text_prefill(ids, prepared, tiny_config(), chunk=4)
        assert tuple(out["logits"].shape) == (6, tiny_config().vocab_size)
        assert torch.isfinite(out["logits"]).all()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
        help="accepted for CI's uniform invocation; nothing here runs on device",
    )
    parser.add_argument("-d", "--device", type=int, default=0, help="accepted and unused")
    parser.parse_args()

    if importlib.util.find_spec("safetensors") is None:
        print("SKIP: safetensors is not installed")
        return 0
    check_contract()
    check_quantization_recipe()
    check_loader()
    check_resumed_fetch_verifies_content()
    check_reference_consumes_prepared_weights()
    print("PASS: Qwen3.8 weight loader and text contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
