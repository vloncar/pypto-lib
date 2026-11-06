# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The MLP at a decode batch, which is the largest single term in decode.

A decode token streams **255.15 MiB through every one of the 64 layers' MLPs** against
111.08 MiB through each of the 48 GDN mixers, so the MLP is 63.6% of the whole weight
stream and two and a half times the mixer. The four stages are the same ones `mlp.py`
composes for prefill; only the tiles differ, and they differ because at decode M is the
batch and each projection reads its whole weight to produce a handful of rows:

| stage | shape | reads |
|---|---|---:|
| `quant_x` | `[M, 5120]` | the activation |
| `gate_up_swiglu` | K=5120 N=17408, two INT8 weights | 170.00 MiB |
| `act_quant` | `[M, 17408]` | the intermediate |
| `down_proj` | K=17408 N=5120 | 85.00 MiB |

`m_tile` is M and `m_group` is 1, both settled by G5.6: below 16 rows the dequant scale
column is shorter than the matmul's 16-row fractal, and with one m-tile there is no patch
to walk. The batch must be a positive multiple of 16, which `decode_vector.check_batch`
states and `decode_projection.padded_batch` produces.

Measured at M = 16, a2a3, median of 30 rounds:

| stage | us | GB/s | of the copy roof |
|---|---:|---:|---:|
| `quant_x` | 23.8 | | |
| `gate_up_swiglu` | 164.0 | 1087 | **88%** |
| `act_quant` | 78.3 | | |
| `down_proj` | 93.3 | 955 | 77% |
| | **359.4** | | |

That total is these tiles. The sweep's best is 359.1, `down_proj` at
`n_tile=64 k_tile=1024 n_group=4` for 93.0 us against 93.3 -- 0.3%, not worth a second
triple to carry, and the one above is within 0.3% at both M = 16 and M = 32.

**The two matmuls are the efficient part.** 88% and 77% of the copy roof beat every GDN
projection G5.6 measured (65%, 54%, 51%), so the widest projection in the model is also the
best-behaved weight stream in it. What cost was the pair of per-token quantisers: 102.1 us
of the 359.4, **28% of the time for 0.3% of the traffic**, because each ran a single task
at a padded batch of 16 where prefill gives them 512. There is no token axis to fan them
over at decode, so the fan-out went to the columns: `act_quant` 78.3 -> 27.4 us (G5.20) and
`quant_x` 23.9 -> 18.2 (G5.23).

**Both of those reached this program only in G5.27**, which is why the composed figure is not
what the table above sums to:

| composed, a2a3, median of 30 rounds | batch 16 | batch 32 |
|---|---:|---:|
| before G5.27 | 374.2 us | 379.9 us |
| after | **290.0 us** | **300.8 us** |

84.2 us a layer at batch 16, **336.7 us/token over 64 layers**, and more than the two stages'
own savings add to -- a shorter stage also lets the next one start earlier. The two fanned
forms are bit-identical to the ones they replace, so the composed gate reports the same
statistics to four digits either way; `fanned=False` keeps the old chain buildable so that
both can be timed in one process.

**Two constraints the off-queue screen found, both in the kernels already.** The 512 KiB
Mat buffer caps `n_tile * k_tile`, and it binds sooner on `gate_up_swiglu` because that
kernel holds two weight tiles rather than one: `n_tile=128 k_tile=1024` needs 557056 bytes
and is refused, as is `n_tile=256 k_tile=512`. And `a8w8_linear` requires an even
`n_group`, its inner loop pairing n-tiles, so `down_proj` cannot run one n-tile per block.
"""
import pypto.language as pl

import a8w8_linear
import act_quant
import gate_up_swiglu
import quant_x
from config import QWEN3_8_27B
from decode_projection import M_FRACTAL, padded_batch
from decode_vector import check_batch

# model config
HIDDEN = QWEN3_8_27B.hidden_size
INTER = QWEN3_8_27B.intermediate_size
LAYERS = QWEN3_8_27B.num_hidden_layers          # every layer has an MLP

# The decode tiles of the two matmuls, as (n_tile, k_tile, n_group), measured at decode M
# over 109 configurations; `m_tile` is M and `m_group` is 1. Both land on the same triple
# and both hold from M = 16 to M = 32, but for different reasons, which is why the sweep
# had to cover both: `gate_up_swiglu` wants the largest operand tile the 512 KiB Mat buffer
# allows and takes 68 blocks with it, while `down_proj` wants **20 blocks against 24 cube
# cores** and falls off a cliff below that -- 93.3 us at 20 blocks, 136.4 at 10, 243.3 at 5.
DECODE_TILES = {
    "gate_up_swiglu": (128, 512, 2),        # 68 blocks, 164.0 us, 1087 GB/s at M = 16
    "down_proj": (128, 512, 2),             # 20 blocks,  93.3 us,  955 GB/s at M = 16
}

# Each quantiser's own measured decode column fan-out (G5.20, G5.23).
X_COL_GROUPS = quant_x.COL_GROUPS
ACT_COL_GROUPS = act_quant.COL_GROUPS


def weight_bytes_per_layer() -> int:
    """INT8 weight one layer's MLP reads, whatever the batch."""
    return 2 * HIDDEN * INTER + INTER * HIDDEN


def build_kernel(batch: int, inline: bool = False, tiles: dict | None = None,
                 fanned: bool = True):
    """The four decode MLP stages as one program. `batch` must be a multiple of 16.

    *fanned* picks each per-token quantiser's decode form, whose reduction is split across
    the column axis. It is a build argument rather than the only option so that both chains
    can be timed in one process on one card; the two forms are bit-identical, because a
    maximum is associative and the rescale is unchanged, so this is a speed knob and not a
    numerics one.
    """
    check_batch(batch)
    tiles = DECODE_TILES if tiles is None else tiles
    up_n_tile, up_k_tile, up_n_group = tiles["gate_up_swiglu"]
    down_n_tile, down_k_tile, down_n_group = tiles["down_proj"]

    if fanned:
        quant = quant_x.build_fanned_kernel(batch, col_groups=X_COL_GROUPS, inline=True)
        act_q_op = act_quant.build_fanned_kernel(batch, col_groups=ACT_COL_GROUPS, inline=True)
    else:
        quant = quant_x.build_kernel(tok_tile=M_FRACTAL, group=1, inline=True)
        act_q_op = act_quant.qwen3_act_quant
    op_gate_up = gate_up_swiglu.build_kernel(m=batch, k=HIDDEN, n=INTER, inline=True,
                                             m_tile=batch, n_tile=up_n_tile, k_tile=up_k_tile,
                                             m_group=1, n_group=up_n_group)
    # Two GM transport slots for the 16-row down projection.
    op_down = a8w8_linear.build_kernel(
        m=batch, k=INTER, n=HIDDEN, name="down_proj", inline=True,
        m_tile=batch, n_tile=down_n_tile, k_tile=down_k_tile,
        m_group=1, n_group=down_n_group, ring_slots=2 if batch == 16 else 1,
    )

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_decode_mlp(
        x: pl.Tensor[[batch, HIDDEN], pl.BF16],
        w_gate: pl.Tensor[[INTER, HIDDEN], pl.INT8],
        w_gate_scale: pl.Tensor[[INTER], pl.FP32],
        w_up: pl.Tensor[[INTER, HIDDEN], pl.INT8],
        w_up_scale: pl.Tensor[[INTER], pl.FP32],
        w_down: pl.Tensor[[HIDDEN, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[HIDDEN], pl.FP32],
        out: pl.Out[pl.Tensor[[batch, HIDDEN], pl.BF16]],
    ):
        x_q = pl.create_tensor([batch, HIDDEN], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, batch], dtype=pl.FP32)
        quant(x, x_q, x_scale)

        act = pl.create_tensor([batch, INTER], dtype=pl.BF16)
        op_gate_up(x_q, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, act)

        act_q = pl.create_tensor([batch, INTER], dtype=pl.INT8)
        act_scale = pl.create_tensor([1, batch], dtype=pl.FP32)
        act_q_op(act, act_q, act_scale)

        op_down(act_q, act_scale, w_down, w_down_scale, out)
        return out

    return qwen3_decode_mlp


__all__ = ["DECODE_TILES", "HIDDEN", "INTER", "LAYERS", "build_kernel", "padded_batch",
           "weight_bytes_per_layer"]
