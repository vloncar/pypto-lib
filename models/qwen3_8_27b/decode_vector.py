# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The elementwise and norm stages at a decode batch, where their prefill tiles do not fit.

Every stage here sets its task count to a quotient of the token count and a tile. At
prefill `T = 8192` that is exact; at a decode batch it is zero, and a zero-task `pl.spmd`
is not a no-op -- the orchestrator asserts `block_num must be >= 1` and the process dies.
Two stages fail more quietly: `qk_rmsnorm_rope` fans out over a fixed 48 blocks with the
quotient inside, so it runs 48 empty tasks and **returns normally with its output
untouched**; and any stage at a token count above its tile but not a multiple of it drops
the tail the same silent way.

**The contract is the one the cube already imposes.** `decode_projection.padded_batch`
rounds the batch up to the 16-row M fractal because a matmul result is a full fractal
whatever M asks for, and padding there is free. Every tile below divides 16, so the same
padded batch satisfies the vector side at no extra cost. A caller that pads once owes
nothing else.

| stage | prefill tile | decode tile | tasks at batch 16 |
|---|---|---|---:|
| `rmsnorm` | `tok_tile=64` | `tok_tile=16` | 1 |
| `quant_x` | `tok_tile=16 group=4` (64 rows) | `tok_tile=16 group=1` | 1 |
| `in_proj_ab` | `m_tile=128 k_tile=256` | `m_tile=16 k_tile=1024` | 1 |
| `qk_norm_gate` | `row_tile=64 group=8`, `tok_tile=256` | `row_tile=16 group=1`, `tok_tile=16` | 16 norm, 1 gate |
| `qk_rmsnorm_rope` | `tok_tile=32` | `tok_tile=16` | 1 |
| `qk_rmsnorm_rope` fused | decode-only | `tok_tile=16` | 1 |
| `rmsnorm` + residual | `tok_tile=64` | `tok_tile=16` | 1 |
| `rmsnorm` + residual, fanned | decode-only | `col_groups=5` | 5 |
| `gated_rmsnorm`, fanned | decode-only | one task per token | 16 |
| `attn_gate_quant`, fanned | decode-only | one task per token | 16 |
| `decode_gate` | decode-only | `tok_tile=16` | 1 |

`in_proj_ab` is the one of these that has been timed: G5.14 measured 27.8 us at
`k_tile = 1024` against 29.1 at 256, so the whole sweep is worth 4% and the stage is its
own launch cost. It runs one task at a padded batch, 0.94 MiB of BF16 weight, and there is
no second axis to spread it over.

The rest are the tiles that make each stage **correct** at a decode batch, not tuned ones.
They are deliberately the largest tile that divides 16, which keeps one task per stage per
16 tokens; whether that is the right task count is a separate question, and C2/C3/C5 say
it will not have the same answer for all five. Re-measure before quoting any of them as a
performance choice (R6).

Four more stages already carry a tile of 16 and are correct at a padded batch, so they bind
the padding contract rather than needing a tile here: `gated_rmsnorm`, `act_quant`,
`swiglu_quant` and `attn_gate_quant`. Correct is not the same as fast, though -- a tile of 16
means **one task** at a decode batch -- and three of the four now have a decode form that
splits that task, listed above (G5.20, G5.26, G5.29). `swiglu_quant` is the one left, and the
composed MLP does not call it.
`embedding` walks one token per work item and is correct at any batch. `rms_lm_head`
walks rows with `pl.parallel` rather than an `spmd`, so below its tile it reads past the
end of the tensor instead of launching nothing; it needs a padded batch too.
"""
from decode_projection import M_FRACTAL, max_batch, padded_batch

# The decode tile of each stage, as keyword arguments to its builder.
DECODE_TILES = {
    "rmsnorm": dict(tok_tile=16),
    "quant_x": dict(tok_tile=16, group=1),
    "quant_x_fanned": dict(m=16, col_groups=4),
    "in_proj_ab": dict(m_tile=16, k_tile=1024),
    "qk_norm_gate": dict(row_tile=16, group=1, tok_tile=16),
    "qk_rmsnorm_rope": dict(tok_tile=16),
    "qk_rmsnorm_rope_fused": dict(tok_tile=16),
    "residual_rmsnorm": dict(tok_tile=16),
    "residual_rmsnorm_fanned": dict(m=16, col_groups=5),
    "gated_rmsnorm_fanned": dict(m=16),
    "attn_gate_quant_fanned": dict(m=16),
    "decode_gate": dict(tok_tile=16),
}

# Stages whose builder is not its module's `build_kernel`, as (module, function).
_BUILDERS = {
    "qk_rmsnorm_rope_fused": ("qk_rmsnorm_rope", "build_fused_kernel"),
    "quant_x_fanned": ("quant_x", "build_fanned_kernel"),
    "residual_rmsnorm": ("rmsnorm", "build_residual_kernel"),
    "residual_rmsnorm_fanned": ("rmsnorm", "build_fanned_residual_kernel"),
    "gated_rmsnorm_fanned": ("gated_rmsnorm", "build_fanned_kernel"),
    "attn_gate_quant_fanned": ("attn_gate_quant", "build_fanned_kernel"),
    "decode_gate": ("decode_gate", "build_kernel"),
}

# Stages whose prefill tile already divides the padded batch.
PADDED_ONLY = ("gated_rmsnorm", "act_quant", "swiglu_quant", "attn_gate_quant", "rms_lm_head")


def build(stage: str, inline: bool = False):
    """One stage at its decode tile, by the name `DECODE_TILES` uses."""
    import importlib

    if stage not in DECODE_TILES:
        if stage in PADDED_ONLY:
            raise ValueError(f"{stage} keeps its prefill tile at decode; it needs only a "
                             f"batch padded to {M_FRACTAL}")
        raise ValueError(f"unknown decode stage {stage!r}; expected one of {sorted(DECODE_TILES)}")
    module, func = _BUILDERS.get(stage, (stage, "build_kernel"))
    builder = getattr(importlib.import_module(module), func)
    return builder(inline=inline, **DECODE_TILES[stage])


def check_batch(batch: int) -> int:
    """The padding contract, enforced once. Returns the batch unchanged.

    Every tile in `DECODE_TILES` divides `M_FRACTAL` and every stage in `PADDED_ONLY`
    already has a tile of it, so a batch that is a positive multiple of 16 satisfies all of
    them **up to `decode_projection.max_batch()`**, which is 128 for the shipped tiles: a
    decode matmul takes `m_tile = m`, so the accumulator and the one BF16 projection's Mat
    tile both grow with the batch and both run out at 128 (G5.11). Without that upper bound
    this check promised any multiple of 16 and the refusal arrived as an on-chip capacity
    error from inside a matmul. This exists because **the tiles do not remove the dangerous failure mode**, they
    only move its threshold: `qk_rmsnorm_rope` fans out over a fixed block count with the
    token quotient inside, so below its tile it runs empty tasks and returns with its output
    untouched. Lowering that tile from 32 to 16 does not make it loud. A host-side check is
    the only thing between a decode launch and a wrong answer with no error in it.
    """
    if batch <= 0 or batch % M_FRACTAL:
        raise ValueError(f"decode batch must be a positive multiple of {M_FRACTAL}, got "
                         f"{batch}; pad it with padded_batch({batch}) = "
                         f"{padded_batch(max(batch, 1))}")
    ceiling = max_batch()
    if batch > ceiling:
        raise ValueError(f"decode batch {batch} exceeds {ceiling}, the largest the decode "
                         f"tiles leave on-chip room for; see decode_projection.max_batch")
    return batch


def check_tile(name: str, tile: int, batch: int) -> int:
    """A token tile that does not divide the padded batch drops the tail, silently.

    Measured, not reasoned: at batch 48 a 32-token tile builds and runs in `in_proj_ab`,
    `decode_gate` and the fused `qk_rmsnorm_rope`, passes an off-queue `compile_only`
    screen, and then fails the composed layer's gate on `state` and `residual_out` because
    one task covers 32 of the 48 rows and nothing writes the other 16 (G5.11). The shipped
    tiles all divide 16 and so divide every legal batch; this exists for the next person
    sweeping one of them, because the compiler has nothing to say about it.
    """
    if tile <= 0 or batch % tile:
        raise ValueError(f"{name}={tile} does not divide the padded batch {batch}, so the "
                         f"{batch % tile} rows past the last whole tile would be left "
                         f"untouched with no error; use a divisor of {batch}")
    return tile


__all__ = ["DECODE_TILES", "M_FRACTAL", "PADDED_ONLY", "build", "check_batch",
           "check_tile", "padded_batch"]
