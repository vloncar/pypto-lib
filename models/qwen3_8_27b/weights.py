# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fetch, validate, and stream Qwen3.8-27B text weights.

There are two deliberately separate paths:

* The Q0-Q4 compatibility path range-fetches one BF16 GDN layer from the
  official checkpoint. Existing per-block validation scripts use its small
  ``.pt`` file.
* Q5 loads all 64 text layers, embeddings, final norm, and LM head from the
  pinned Eco-Tech W8A8 checkpoint. It excludes the vision tower and MTP layer,
  validates all 1,555 checkpoint tensors against :mod:`contract`, discards the
  symmetric all-zero offsets, flattens scales to the kernel convention, and
  exports one tensor at a time so host memory need not hold a second 29 GiB
  copy while weights move to device.

The whole checkpoint is a large, explicit download:

    python models/qwen3_8_27b/weights.py --fetch-checkpoint
    python models/qwen3_8_27b/weights.py --validate-checkpoint

``--verify-index`` is the cheap third option. It checks :mod:`contract`
against the *published* checkpoint -- names, every shape and dtype, and the
byte totals -- from the index and the ten safetensors headers alone, a few
hundred KiB rather than the checkpoint's 29.92 GiB:

    python models/qwen3_8_27b/weights.py --verify-index

With no checkpoint flag the command remains CI-safe and retains the original
single-layer behavior. Missing data is reported rather than downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .contract import (
        BF16,
        FP32,
        INT8,
        QWEN3_8_27B_TEXT_CONTRACT,
        Qwen38TextContract,
        WeightSpec,
    )
else:
    _REPO_ROOT = str(Path(__file__).resolve().parents[2])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from models.qwen3_8_27b.contract import (
        BF16,
        FP32,
        INT8,
        QWEN3_8_27B_TEXT_CONTRACT,
        Qwen38TextContract,
        WeightSpec,
    )


BF16_REPO = "Qwen/Qwen3.8-27B"
BF16_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
BF16_PREFIX = "model.language_model.layers.{layer}.linear_attn."
# Q0-Q4 compatibility names.
REPO = BF16_REPO
REVISION = BF16_REVISION
PREFIX = BF16_PREFIX

ECO_REPO = "Eco-Tech/Qwen3.8-27B-w8a8"
ECO_REVISION = "ca2937f18b7e399c0ee91c728a4cde9b837aebff"
ECO_INDEX = "quant_model_weights.safetensors.index.json"
ECO_METADATA_FILES = ("config.json", "quant_model_description.json", ECO_INDEX)

_SAFETENSOR_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_CONTRACT_DTYPES = {BF16: torch.bfloat16, FP32: torch.float32, INT8: torch.int8}

TensorExporter = Callable[[torch.Tensor], Any]


@dataclass(frozen=True)
class PreparedQwen38Weights:
    """Kernel/reference-ready text tensors exported from the Eco-Tech checkpoint."""

    tensors: dict[str, Any]
    layer_types: tuple[str, ...]
    checkpoint_weight_bytes: int
    resident_weight_bytes: int

    def layer(self, index: int) -> dict[str, Any]:
        """Return one layer's tensors with the layer prefix removed."""
        if not 0 <= index < len(self.layer_types):
            raise IndexError(f"layer index {index} outside [0, {len(self.layer_types)})")
        prefix = f"model.language_model.layers.{index}."
        return {name[len(prefix) :]: value for name, value in self.tensors.items() if name.startswith(prefix)}


# The tensor groups one layer can be fetched in. `mlp` and `norms` are on every
# layer; the two mixers are on the layers their type says.
MIXERS = ("linear_attn", "self_attn", "mlp", "norms")
_STEMS = {"linear_attn": "linear_attn", "self_attn": "full_attn", "mlp": "mlp",
          "norms": "norms"}

# `norms` is the only group that is not a submodule: the two layer norms sit
# directly under the layer, so they are named rather than taken by prefix.
LAYER_NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight")

# Tensors this far apart in a shard are fetched as separate range requests.
_FETCH_GAP = 1 << 20


def default_path(layer: int, quantized: bool = False, mixer: str = "linear_attn") -> Path:
    """Path for one extracted BF16 layer group: GDN, gated attention, or the MLP."""
    root = Path(__file__).resolve().parents[2]
    suffix = "_int8" if quantized else ""
    return root / "build_output" / "qwen3_8_27b" / f"{_STEMS[mixer]}_layer{layer}{suffix}.pt"


def default_checkpoint_path() -> Path:
    """Default local directory for the pinned Eco-Tech checkpoint."""
    root = Path(__file__).resolve().parents[2]
    return root / "build_output" / "qwen3_8_27b" / "Eco-Tech-Qwen3.8-27B-w8a8"


def _get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as response:
        if byte_range is not None and response.status != 206:
            raise RuntimeError(f"range request not honoured ({response.status}) for {url}")
        return response.read()


def fetch_layer(
    layer: int,
    repo: str = BF16_REPO,
    revision: str = BF16_REVISION,
    log: Callable[[str], None] = print,
    mixer: str = "linear_attn",
) -> dict[str, torch.Tensor]:
    """Fetch one official BF16 layer group -- a mixer, the ``mlp`` or the two ``norms`` -- by HTTP range."""
    if mixer not in MIXERS:
        raise ValueError(f"mixer must be one of {MIXERS}, got {mixer!r}")
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    prefix = f"model.language_model.layers.{layer}."
    if mixer != "norms":
        prefix += f"{mixer}."

    def wanted(name: str) -> bool:
        """Whether a prefix-stripped tensor name belongs to the requested group."""
        return name in LAYER_NORMS if mixer == "norms" else True

    index = json.loads(_get(base + "model.safetensors.index.json"))["weight_map"]
    names = [name for name in index if name.startswith(prefix) and wanted(name[len(prefix) :])]
    if not names:
        raise RuntimeError(f"layer {layer}: no {mixer} tensors in the index")
    shards = {index[name] for name in names}
    if len(shards) != 1:
        raise RuntimeError(f"layer {layer}: expected one shard, found {sorted(shards)}")
    shard = shards.pop()

    header_len = struct.unpack("<Q", _get(base + shard, (0, 7)))[0]
    header = json.loads(_get(base + shard, (8, 8 + header_len - 1)))
    header.pop("__metadata__", None)
    entries = {name[len(prefix) :]: info for name, info in header.items()
               if name.startswith(prefix) and wanted(name[len(prefix) :])}
    # One range request per contiguous run of tensors, not one spanning all of them. A
    # module's tensors are adjacent in the shard and coalesce into a single read, but the
    # two layer norms sit at opposite ends of the layer, and the span between them is the
    # whole layer -- 731 MiB to fetch 20 KiB.
    ordered = sorted(entries.items(), key=lambda kv: kv[1]["data_offsets"][0])
    runs: list[list[int]] = []
    for _, info in ordered:
        start, end = info["data_offsets"]
        if runs and start - runs[-1][1] <= _FETCH_GAP:
            runs[-1][1] = max(runs[-1][1], end)
        else:
            runs.append([start, end])
    fetched = sum(hi - lo for lo, hi in runs)
    log(
        f"[weights] layer {layer}: {len(entries)} tensors, {fetched / 2**20:.1f} MiB in "
        f"{len(runs)} range request(s) from {shard} at revision {revision[:8]}"
    )

    data_start = 8 + header_len
    blobs = [_get(base + shard, (data_start + lo, data_start + hi - 1)) for lo, hi in runs]
    out = {}
    for name, info in entries.items():
        start, end = info["data_offsets"]
        run = next(i for i, (lo, hi) in enumerate(runs) if lo <= start and end <= hi)
        raw = bytearray(blobs[run][start - runs[run][0] : end - runs[run][0]])
        out[name] = (
            torch.frombuffer(raw, dtype=_SAFETENSOR_DTYPES[info["dtype"]]).reshape(info["shape"]).clone()
        )
    return out


def load_layer(path: str | Path) -> dict[str, torch.Tensor]:
    """Load one legacy extracted GDN layer."""
    return torch.load(path, weights_only=True)


def modelslim_quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce ModelSlim's symmetric per-channel INT8 min/max layout.

    Returns ``weight``, ``weight_scale`` and ``weight_offset`` with the exact
    Eco-Tech shapes. ModelSlim performs the division in the BF16 source dtype,
    then saves the BF16-rounded scale in an FP32 tensor; preserving that order
    is required for bit identity. The offset is zero because the recipe is
    symmetric.

    One detail is reproduced but unconfirmed: the ``amax`` floor is Q4's own
    ``GDN_QUANT.amax_eps``, not a value read back from ModelSlim. The bit
    identity check that established this recipe used real ``in_proj_qkv`` rows,
    whose ``amax`` is orders of magnitude above the floor, so no comparison so
    far has exercised it. A checkpoint row that is entirely zero would be the
    first, and its scale would then rest on this assumption.
    """
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError(f"expected a BF16 [out, in] weight, got {tuple(weight.shape)} {weight.dtype}")
    amax = weight.abs().amax(dim=1).clamp_min(1e-4)
    native_scale = amax / 127.0
    quantized = torch.round(weight / native_scale[:, None]).clamp(-127, 127).to(torch.int8)
    scale = native_scale.float()[:, None].contiguous()
    offset = torch.zeros_like(scale)
    return quantized, scale, offset


def _load_checkpoint_index(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir) / ECO_INDEX
    if not path.is_file():
        raise FileNotFoundError(f"missing Eco-Tech checkpoint index: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data.get("weight_map"), dict):
        raise ValueError(f"invalid safetensors index (no weight_map): {path}")
    return data


def checkpoint_summary(
    checkpoint_dir: str | Path,
    contract: Qwen38TextContract = QWEN3_8_27B_TEXT_CONTRACT,
) -> dict[str, int]:
    """Validate the index and required shard files without allocating tensor data."""
    checkpoint_dir = Path(checkpoint_dir)
    index = _load_checkpoint_index(checkpoint_dir)
    weight_map = index["weight_map"]
    expected = {spec.name for spec in contract.weight_specs}
    actual_text = {
        name for name in weight_map if name == "lm_head.weight" or name.startswith("model.language_model.")
    }
    missing = sorted(expected - actual_text)
    unexpected = sorted(actual_text - expected)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} tensors (first: {missing[:3]})")
        if unexpected:
            detail.append(f"unexpected {len(unexpected)} text tensors (first: {unexpected[:3]})")
        raise ValueError("checkpoint does not match Qwen3.8-27B text contract: " + "; ".join(detail))

    shards = {weight_map[name] for name in expected}
    unsafe = sorted(name for name in shards if Path(name).name != name)
    if unsafe:
        raise ValueError(f"checkpoint index contains unsafe shard paths: {unsafe}")
    absent_shards = sorted(name for name in shards if not (checkpoint_dir / name).is_file())
    if absent_shards:
        raise FileNotFoundError(
            f"checkpoint is missing {len(absent_shards)} required shards: {absent_shards}"
        )
    return {
        "text_tensors": len(expected),
        "prepared_tensors": len(contract.prepared_weight_specs),
        "shards": len(shards),
        "checkpoint_weight_bytes": contract.memory.checkpoint_weight_bytes,
        "resident_weight_bytes": contract.memory.resident_weight_bytes,
    }


def _remote_header(repo: str, revision: str, shard: str) -> dict[str, Any]:
    """Read one remote shard's safetensors header by HTTP range, no payload."""
    url = _modelscope_url(repo, revision, shard)
    header_len = struct.unpack("<Q", _get(url, (0, 7)))[0]
    header = json.loads(_get(url, (8, 8 + header_len - 1)))
    header.pop("__metadata__", None)
    return header


def verify_published_contract(
    contract: Qwen38TextContract = QWEN3_8_27B_TEXT_CONTRACT,
    *,
    repo: str = ECO_REPO,
    revision: str = ECO_REVISION,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Check :mod:`contract` against the published checkpoint, downloading no weights.

    The assertions in :mod:`contract` pin the table against accidental edits,
    but they compare it only with itself: they would pass unchanged if the
    published checkpoint moved underneath us. This fetches the index and the
    ten safetensors headers -- a few hundred KiB in total, no tensor payload --
    and checks the names, every shape and dtype, and the byte totals against
    what the contract claims. It needs the network and never runs unasked.
    """
    index = json.loads(_get(_modelscope_url(repo, revision, ECO_INDEX)))["weight_map"]
    actual = {name for name in index if name == "lm_head.weight" or name.startswith("model.language_model.")}
    expected = {spec.name for spec in contract.weight_specs}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"published index disagrees with the text contract: {len(missing)} missing "
            f"(first: {missing[:3]}), {len(unexpected)} unexpected (first: {unexpected[:3]})"
        )
    log(f"[weights] index at {revision[:8]}: {len(actual)} text tensors, names match the contract")

    shards = sorted({index[name] for name in expected})
    header: dict[str, Any] = {}
    for position, shard in enumerate(shards, 1):
        entries = _remote_header(repo, revision, shard)
        header.update(entries)
        log(f"[weights] [{position}/{len(shards)}] header of {shard}: {len(entries)} tensors")

    names = {BF16: "BF16", FP32: "F32", INT8: "I8"}
    wrong = [
        (spec.name, spec.shape, spec.dtype, tuple(header[spec.name]["shape"]), header[spec.name]["dtype"])
        for spec in contract.weight_specs
        if tuple(header[spec.name]["shape"]) != spec.shape or header[spec.name]["dtype"] != names[spec.dtype]
    ]
    if wrong:
        raise ValueError(f"{len(wrong)} tensors disagree with the contract; first: {wrong[:3]}")

    def span(spec: WeightSpec) -> int:
        start, end = header[spec.name]["data_offsets"]
        return end - start

    checkpoint_bytes = sum(span(spec) for spec in contract.weight_specs)
    resident_bytes = sum(span(spec) for spec in contract.prepared_weight_specs)
    for label, actual_bytes, claimed in (
        ("checkpoint_weight_bytes", checkpoint_bytes, contract.memory.checkpoint_weight_bytes),
        ("resident_weight_bytes", resident_bytes, contract.memory.resident_weight_bytes),
    ):
        if actual_bytes != claimed:
            raise ValueError(f"{label}: published checkpoint has {actual_bytes}, contract claims {claimed}")

    log(
        f"[weights] {len(expected)} shapes and dtypes agree; payload {checkpoint_bytes} bytes "
        f"({checkpoint_bytes / 2**30:.2f} GiB), resident {resident_bytes} bytes "
        f"({resident_bytes / 2**30:.2f} GiB) -- both as pinned"
    )
    return {
        "text_tensors": len(expected),
        "shards": len(shards),
        "checkpoint_weight_bytes": checkpoint_bytes,
        "resident_weight_bytes": resident_bytes,
    }


def _validate_tensor(tensor: torch.Tensor, spec: WeightSpec) -> None:
    expected_dtype = _CONTRACT_DTYPES[spec.dtype]
    if tensor.dtype != expected_dtype or tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: got shape={tuple(tensor.shape)} dtype={tensor.dtype}; "
            f"expected shape={spec.shape} dtype={expected_dtype}"
        )


def prepare_qwen38_weights(
    checkpoint_dir: str | Path,
    tensor_exporter: TensorExporter,
    *,
    contract: Qwen38TextContract = QWEN3_8_27B_TEXT_CONTRACT,
    verify_zero_offsets: bool = True,
) -> PreparedQwen38Weights:
    """Stream the complete text checkpoint into the runtime's tensor exporter.

    A shard is opened once and tensors are exported individually. Symmetric
    ``weight_offset`` tensors are validated and omitted; ``weight_scale`` is
    flattened from checkpoint ``[out, 1]`` to kernel ``[out]``. All remaining
    names stay checkpoint-native, so the returned dictionary also feeds the Q5a
    reference directly.
    """
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("prepare_qwen38_weights requires the safetensors package") from error

    checkpoint_dir = Path(checkpoint_dir)
    summary = checkpoint_summary(checkpoint_dir, contract)
    index = _load_checkpoint_index(checkpoint_dir)
    weight_map = index["weight_map"]
    spec_by_name = {spec.name: spec for spec in contract.weight_specs}
    names_by_shard: dict[str, list[str]] = {}
    for spec in contract.weight_specs:
        names_by_shard.setdefault(weight_map[spec.name], []).append(spec.name)

    prepared: dict[str, Any] = {}
    for shard, names in names_by_shard.items():
        shard_path = checkpoint_dir / shard
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
            missing = [name for name in names if name not in shard_keys]
            if missing:
                raise ValueError(f"{shard}: index names absent from safetensors header: {missing[:3]}")
            for name in names:
                spec = spec_by_name[name]
                tensor = handle.get_tensor(name)
                _validate_tensor(tensor, spec)
                if spec.is_zero_offset:
                    if verify_zero_offsets and bool(torch.count_nonzero(tensor)):
                        raise ValueError(f"{name}: symmetric W8A8 checkpoint offset is not zero")
                    continue
                if spec.role == "weight_scale":
                    tensor = tensor.reshape(-1)
                prepared[name] = tensor_exporter(tensor.contiguous())

    expected_prepared = [spec.name for spec in contract.prepared_weight_specs]
    prepared = {name: prepared[name] for name in expected_prepared}
    return PreparedQwen38Weights(
        tensors=prepared,
        layer_types=contract.layer_types,
        checkpoint_weight_bytes=summary["checkpoint_weight_bytes"],
        resident_weight_bytes=summary["resident_weight_bytes"],
    )


def load_text_weights(
    checkpoint_dir: str | Path,
    *,
    contract: Qwen38TextContract = QWEN3_8_27B_TEXT_CONTRACT,
) -> dict[str, torch.Tensor]:
    """Load prepared text weights on CPU; intended for reference validation."""
    return prepare_qwen38_weights(checkpoint_dir, lambda tensor: tensor, contract=contract).tensors


def _modelscope_url(repo: str, revision: str, file_path: str) -> str:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo.split("/"))
    query = urllib.parse.urlencode({"Revision": revision, "FilePath": file_path})
    return f"https://modelscope.cn/api/v1/models/{quoted_repo}/repo?{query}"


def _modelscope_manifest(repo: str, revision: str) -> dict[str, dict[str, Any]]:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo.split("/"))
    query = urllib.parse.urlencode({"Revision": revision, "Root": ""})
    url = f"https://modelscope.cn/api/v1/models/{quoted_repo}/repo/files?{query}"
    response = json.loads(_get(url))
    if response.get("Code") != 200:
        raise RuntimeError(f"ModelScope manifest request failed: {response}")
    return {item["Name"]: item for item in response["Data"]["Files"] if item["Type"] == "blob"}


def _digest_marker(destination: Path) -> Path:
    """Sidecar recording the SHA-256 this code verified for ``destination``."""
    return destination.with_name(destination.name + ".sha256")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 2**20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verified_locally(destination: Path, expected_size: int, expected_sha256: str) -> bool:
    """Whether an existing file is known to match, hashing it once if need be.

    A size match alone cannot tell a complete shard from a corrupted one of the
    same length, and re-hashing 29.92 GiB on every invocation is not acceptable
    either. So a verified file records its digest beside it, together with the
    size and modification time it was computed from. A later run trusts that
    record only while all three still describe the file on disk, and otherwise
    hashes in place once and rewrites it.

    What this catches is a file replaced, truncated, resumed by another tool,
    or copied in by hand -- every case where something rewrote the bytes. It
    does not catch silent bit rot that leaves size and mtime untouched; only
    hashing every run would, and that is the cost being avoided. Delete the
    ``.sha256`` sidecar to force a re-check.
    """
    if not destination.is_file():
        return False
    status = destination.stat()
    if status.st_size != expected_size:
        return False
    marker = _digest_marker(destination)
    stamp = f"{expected_sha256} {status.st_size} {status.st_mtime_ns}"
    if marker.is_file() and marker.read_text().strip() == stamp:
        return True
    if _file_sha256(destination) != expected_sha256:
        return False
    _write_digest_marker(destination, expected_sha256)
    return True


def _write_digest_marker(destination: Path, sha256: str) -> None:
    status = destination.stat()
    _digest_marker(destination).write_text(f"{sha256} {status.st_size} {status.st_mtime_ns}\n")


def _download_file(url: str, destination: Path, expected_size: int, expected_sha256: str) -> None:
    part = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    written = 0
    try:
        with urllib.request.urlopen(url, timeout=600) as response, part.open("wb") as output:
            while True:
                chunk = response.read(8 * 2**20)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if written != expected_size:
            raise RuntimeError(f"{destination.name}: downloaded {written} bytes, expected {expected_size}")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"{destination.name}: sha256 {actual_sha256} != expected {expected_sha256}")
        os.replace(part, destination)
        _write_digest_marker(destination, expected_sha256)
    except BaseException:
        if part.exists():
            part.unlink()
        raise


def fetch_checkpoint(
    checkpoint_dir: str | Path,
    *,
    repo: str = ECO_REPO,
    revision: str = ECO_REVISION,
    log: Callable[[str], None] = print,
) -> Path:
    """Download the pinned Eco-Tech metadata and ten weight shards atomically."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = _modelscope_manifest(repo, revision)
    shard_names = sorted(
        name for name in manifest if name.startswith("quant_model_weights-") and name.endswith(".safetensors")
    )
    required = (*ECO_METADATA_FILES, *shard_names)
    missing_manifest = [name for name in required if name not in manifest]
    if missing_manifest:
        raise RuntimeError(f"ModelScope manifest is missing required files: {missing_manifest}")

    present = {
        name: _verified_locally(checkpoint_dir / name, int(manifest[name]["Size"]), str(manifest[name]["Sha256"]))
        for name in required
    }
    remaining = sum(int(manifest[name]["Size"]) for name in required if not present[name])
    free = shutil.disk_usage(checkpoint_dir).free
    if remaining > free:
        raise OSError(f"checkpoint needs {remaining / 2**30:.2f} GiB but only {free / 2**30:.2f} GiB is free")

    for position, name in enumerate(required, 1):
        info = manifest[name]
        destination = checkpoint_dir / name
        expected_size = int(info["Size"])
        if present[name]:
            log(f"[weights] [{position}/{len(required)}] present and verified: {name}")
            continue
        log(f"[weights] [{position}/{len(required)}] downloading {name} ({expected_size / 2**30:.2f} GiB)")
        _download_file(
            _modelscope_url(repo, revision, name),
            destination,
            expected_size,
            str(info["Sha256"]),
        )
    return checkpoint_dir


def _print_checkpoint_summary(checkpoint_dir: Path) -> None:
    summary = checkpoint_summary(checkpoint_dir)
    memory = QWEN3_8_27B_TEXT_CONTRACT.memory
    print(
        f"[weights] {summary['text_tensors']} raw text tensors in {summary['shards']} shards; "
        f"{summary['prepared_tensors']} tensors after zero offsets are dropped"
    )
    print(
        f"[weights] checkpoint text payload {summary['checkpoint_weight_bytes'] / 2**30:.2f} GiB; "
        f"resident weight payload {summary['resident_weight_bytes'] / 2**30:.2f} GiB"
    )
    print(
        f"[weights] native-context KV {memory.max_kv_cache_bytes / 2**30:.2f} GiB; "
        f"GDN state {memory.gdn_state_bytes_per_sequence / 2**20:.0f} MiB; "
        f"64-GiB headroom {memory.max_context_headroom_bytes / 2**30:.2f} GiB"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Accepted and ignored: this is host-side, but CI invokes runnable model
    # files with one uniform platform/device command line.
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3",
                        help="accepted for CI's uniform invocation; nothing here runs on device")
    parser.add_argument("--sim", dest="execution_mode", action="store_true",
                        help="accepted for CI's uniform invocation; unused")
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        help="accepted for CI's uniform invocation; unused",
    )
    checkpoint_mode = parser.add_mutually_exclusive_group()
    checkpoint_mode.add_argument(
        "--fetch-checkpoint",
        action="store_true",
        help="download the pinned ~30 GiB Eco-Tech W8A8 checkpoint",
    )
    checkpoint_mode.add_argument(
        "--validate-checkpoint",
        action="store_true",
        help="validate an already downloaded checkpoint index and shard set",
    )
    checkpoint_mode.add_argument(
        "--verify-index",
        action="store_true",
        help="check the contract against the published checkpoint's index and "
        "safetensors headers; needs the network, downloads no weights",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Eco-Tech checkpoint directory (default under build_output/qwen3_8_27b)",
    )
    parser.add_argument("--checkpoint-revision", type=str, default=ECO_REVISION)

    # Legacy one-layer options retained for Q0-Q4 validation.
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download one official BF16 GDN layer if absent",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--mixer",
        type=str,
        default="linear_attn",
        choices=MIXERS,
        help="which group of the layer's tensors to fetch: a mixer, the MLP, or the two layer norms",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--revision", type=str, default=BF16_REVISION)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="write the legacy layer's Q4 W8A8 form",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else default_checkpoint_path()
    if args.fetch_checkpoint:
        fetch_checkpoint(checkpoint_dir, revision=args.checkpoint_revision)
        _print_checkpoint_summary(checkpoint_dir)
        return 0
    if args.validate_checkpoint:
        _print_checkpoint_summary(checkpoint_dir)
        return 0
    if args.verify_index:
        verify_published_contract(revision=args.checkpoint_revision)
        return 0

    # Every layer has an MLP and two norms; the two mixers are on the layers their type says.
    if args.mixer not in ("mlp", "norms"):
        expected = "linear_attention" if args.mixer == "linear_attn" else "full_attention"
        actual = QWEN3_8_27B_TEXT_CONTRACT.layer_type(args.layer)
        if actual != expected:
            raise SystemExit(f"[weights] layer {args.layer} is {actual}, which has no {args.mixer}")
    source = default_path(args.layer, mixer=args.mixer)
    if not source.is_file() and not args.fetch:
        print(
            f"[weights] layer {args.layer} is not on disk ({source}); "
            f"pass --fetch to download it from {BF16_REPO}"
        )
        return 0

    if args.quantize:
        if args.mixer != "linear_attn":
            raise SystemExit("[weights] --quantize writes the legacy GDN layer only")
        if __package__:
            from . import reference
        else:
            import reference

        if not source.is_file():
            weights = fetch_layer(args.layer, revision=args.revision)
            source.parent.mkdir(parents=True, exist_ok=True)
            torch.save(weights, source)
        weights = reference.quantize_weights(load_layer(source))
        out = Path(args.out) if args.out else default_path(args.layer, quantized=True)
    elif source.is_file():
        print(f"[weights] layer {args.layer} already on disk ({source})")
        weights = load_layer(source)
        out = Path(args.out) if args.out else source
    else:
        weights = fetch_layer(args.layer, revision=args.revision, mixer=args.mixer)
        out = Path(args.out) if args.out else source
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out)
    for name, tensor in weights.items():
        print(f"  {name:<24} {str(tuple(tensor.shape)):<16} {tensor.dtype}")
    print(f"[weights] saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
