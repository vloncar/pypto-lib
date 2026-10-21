# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The three A8W8 projections at decode's M, where they are a weight stream.

`a8w8_linear` is the same kernel prefill uses; only the tiles differ, and they differ
because the regime does. At decode M is the batch, so each projection reads its whole
INT8 weight to produce a few output rows:

| projection | K | N | INT8 weight |
|---|---:|---:|---:|
| `in_proj_qkv` | 5120 | 10240 | 52.43 MB |
| `in_proj_z` | 5120 | 6144 | 31.46 MB |
| `out_proj` | 6144 | 5120 | 31.46 MB |
| | | | **115.34 MB per GDN layer** |

That is 18x the 6.29 MB of recurrent state a token touches in the same layer, and unlike
the state it does not grow with the batch. Measured, the three projections cost
**162.7 us at batch 16 and 164.5 us at batch 32** -- doubling the batch is free, which is
what weight-bound means and is the cleanest statement of the regime available.

**M must be at least 16, and padding to it is free.** Below 16 the build fails inside
`row_expand_mul`: a matmul result is a full 16-row fractal while the dequant scale column
is `m_tile` rows. Padding wastes cube rows, and cube rows are not what this costs -- a
batch-1 token pays the same 162.7 us as a batch-16 launch. The caller owes the padded
activation; a padded decode batch, which is what a fused loop carries anyway, satisfies it
for free.

`m_group` is 1 by construction: with one m-tile there is no patch to walk, and a block
reading `n_group` n-tiles over the whole K touches each weight byte exactly once, so no
`n_group` re-reads anything. It changes only the block count, `N / n_tile / n_group`.

The tiles below are measured, not inherited -- every prefill constant here was swept at
T = 8192 and none of them survives (R6). Best of 72 configurations per projection, a2a3,
median of 30 rounds:

| projection | m_tile | n_tile | k_tile | n_group | us at M=16 | GB/s | of the copy roof |
|---|---:|---:|---:|---:|---:|---:|---:|
| `in_proj_qkv` | M | 128 | 512 | 4 | 65.6 | 798.8 | 65% |
| `in_proj_z` | M | 64 | 1024 | 4 | 47.1 | 668.3 | 54% |
| `out_proj` | M | 64 | 512 | 4 | 50.0 | 629.7 | 51% |

The roof quoted is this box's **vector-unit copy** roof, 1234 GB/s, which is the wrong
instrument for a GM -> L1 -> L0B weight stream and is cited only for scale. The sweep over
`n_tile` and `k_tile` bought 4.3% at the layer, so the easy headroom is gone; what is left
would need a different streaming structure, and no fused loop can reduce it, because the
weight has to be read whatever encloses it.

**G5.14 added the attention layers' three, and the block count turns out to be a wave
count.** `q_proj` has 96 n-tiles at `n_tile = 64`, fine enough to hit the cube's 24 cores
exactly, and the minimum is there:

| blocks | 3 | 4 | 6 | 8 | 12 | 16 | **24** | 32 | 48 | 96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| waves of 24 | .12 | .17 | .25 | .33 | .50 | .67 | **1.00** | 1.33 | 2.00 | 4.00 |
| best us | 321.5 | 245.8 | 157.8 | 126.5 | 94.7 | 81.3 | **71.7** | 88.5 | 76.7 | 92.8 |

Monotonic below one wave, because cores sit idle. Above it the curve is non-monotonic, and
`q_proj`'s minimum is at exactly one wave.

**That is a fact about `q_proj`, not a rule.** G5.21 measured the same kernel at
`N = 14336`, 17% wider, and the ordering inverts: 56 blocks (2.33 waves) is best at 90.9 us
while 28 blocks (1.17 waves) is 109.0 and 16 blocks (0.67) is 93.2. So one wave is not a
target and a whole wave does not beat a partial one in general; `q_proj` wants four
n-tile pairs per block and the wider shape wants one, and nothing here explains why.

What does hold across both, and across G5.15's two shapes, is the starvation side: below
about half a wave the cost climbs steeply and monotonically (`q_proj` 98.4 us at 12 blocks,
the fused shape 144.4 at 8 and 552.2 at 2). **Fill the cores, then sweep.** C5's rule
applies to the block count as much as to the task count: measure it per shape, because the
same kernel at a nearby width answers differently.

**`k_proj` and `v_proj` cannot reach one wave at all, and they are the worst-behaved
matmuls in the model.** `N = 1024` gives 16 n-tiles at `n_tile = 64`, and an even `n_group`
caps the block count at 8, a third of a wave. They cost 28.0 and 27.7 us for 5 MiB each,
**15% of the copy roof**, where `q_proj` moves twelve times the weight in two and a half
times the time. Their curve is monotonic in blocks (28.0 / 33.3 / 49.8 / 86.0 at 8 / 4 / 2
/ 1), so the tile matters a great deal even though the weight does not -- these are
parallelism-starved, not launch-bound.

**G5.21 fuses the three into `qkv_proj`, and what the fusion buys is a uniform loop.**
q, k and v read the same quantised activation, so one `a8w8_linear` over the three weights
concatenated by row is the same arithmetic: no row is requantised, `a8w8_linear` accumulates
INT32 over the whole K and dequantises once, and the fused output is therefore bit-identical
to the three separate ones at every tiling. Measured exact over 229376 elements at M = 16 and
458752 at M = 32, on random and on real layer-3 weights.

**This projection's cost differs by card**, so the figure is given per card rather than as
one number. Same script, 50 rounds at M = 16, the two runs back to back:

| | card 2 | card 3 |
|---|---:|---:|
| q + k + v | 130.3 | 126.2 |
| `qkv_proj` | **90.5** | **95.8** |
| the fusion saves, per layer | **39.8 us** | **30.4 us** |
| over 16 layers, per token at batch 16 | **39.8 us** | **30.4 us** |

`qkv_proj` runs 5.9% faster on card 2 while `q_proj` runs 4.8% faster on card 3, so the two
cards disagree in opposite directions on two shapes of the same kernel and the gap is not a
property of either alone. Both figures are real; **30.4 us/token is the one to plan with**,
being the worse of the two measured. Per token is the step's saving divided by the batch, the
convention `gate_up_swiglu`'s figures use: a layer runs once per step and a step yields
`batch` tokens, so 16 layers at batch 16 happen to make the per-layer and per-token numbers
equal, which they do not at any other batch. An earlier version of this file quoted 0.65 from a
card-2 session alone, which was accurate and unattributed, and that is the error to avoid:
quote the card.

The consumers read q and k by channel range out of the fused tensor and cost the same as the
narrow form to within 1.4%, so none of the producer's gain is given back;
`qk_rmsnorm_rope.build_fused_kernel` and `decode_kv_append.build_fused_kernel` are those two
forms. G5.22 then took the first of them from 53.1 to 16.6 us, and the whole front end of an
attention layer is now **198.1 -> 121.8 us on card 2 and 192.7 -> 127.3 on card 3**, which is
76.3 and 65.4 us/token for the two items together at batch 16.

**Keeping the output three tensors does not work, and why it does not is the point.** A
block's store target would depend on its n-tile index, which is a runtime scalar inside
`pl.spmd` and cannot be branched on at trace time. The only escape is three loop nests
striding the same block set, and that cannot balance: q has 96 n-tiles at `n_tile = 128`
against k's and v's 8, so 56 blocks give the first block six tiles and the last three where
the mean is two. Only a block count at or below 8 balances, and 8 blocks costs 144.4 us. So
the fusion's content is one uniform loop over 112 identical tiles, which needs one weight and
one wide output -- the consumer change is what it costs, not an incidental tidiness.

This does not reverse X4, which rejected fusing projections that share an activation because
activation traffic does not fall. It still does not: `weight_bytes_per_layer` returns the same
100 MiB either way. The decode gain is block count and launch count, which X4 never tested.

One measurement note, which the card table above is the resolution of. `q_proj` read 75.0 us
in one session against G5.14's 71.7 on the same box at the same tiling, and `qkv_proj` read
90.2 then 96.8. Neither was drift: an A/B of one script across cards 2 and 3 reproduced both
gaps in the same directions and nearly the same sizes, so **the card is the variable**. Two
consequences for anything measured here: re-time the baseline in the same process as the
variant, which the tables above do, and name the card, which they now also do.
"""
import a8w8_linear
import bf16_linear
from config import QWEN3_8_27B

# model config
HIDDEN = QWEN3_8_27B.hidden_size
QKV_WIDTH = QWEN3_8_27B.qkv_width
VAL_WIDTH = QWEN3_8_27B.linear_num_value_heads * QWEN3_8_27B.linear_value_head_dim
ATTN_QG_WIDTH = 2 * QWEN3_8_27B.num_attention_heads * QWEN3_8_27B.head_dim   # query and gate
ATTN_KV_WIDTH = QWEN3_8_27B.num_key_value_heads * QWEN3_8_27B.head_dim
ATTN_QKV_WIDTH = ATTN_QG_WIDTH + 2 * ATTN_KV_WIDTH

# The fused projection's channel layout, which is the ABI between it and its three
# consumers. It is the row order of the concatenated weight, nothing more; `fuse_qkv_weight`
# builds that weight and these offsets read it back.
QKV_QG_BASE = 0
QKV_K_BASE = ATTN_QG_WIDTH
QKV_V_BASE = ATTN_QG_WIDTH + ATTN_KV_WIDTH

# The cube's M fractal: a matmul result has 16 rows whatever M asks for.
M_FRACTAL = 16

# (K, N, n_tile, k_tile, n_group), measured at decode M. m_tile is M itself.
DECODE_TILES = {
    "in_proj_qkv": (HIDDEN, QKV_WIDTH, 128, 512, 4),
    "in_proj_z": (HIDDEN, VAL_WIDTH, 64, 1024, 4),
    # BF16 on both sides, because the deployed checkpoint leaves this one projection
    # unquantised and D1 settled on matching it. Swept as a BF16 matmul by G5.30; the INT8
    # triple it used to carry lives on in `o_proj`, which is the same shape and still int8.
    # The Mat buffer binds much earlier than it does for int8, and not where counting the
    # named tiles suggests: the compiler's reported usage is
    # `8 * n_tile * k_tile + 4 * m_tile * k_tile`, which is the weight tile in FOUR slots plus
    # a double-buffered activation, and it does not depend on `n_group` at all. At a padded
    # batch that leaves `n_tile * k_tile <= 32768`, so `(64, 1024)`, `(128, 512)` and
    # `(256, 256)` are all refused -- at 589824, 557056 and 540672 bytes respectively.
    #
    # Nine triples survive and the sweep says **the block count is the only thing that
    # matters**: every triple that lands 20 blocks reads 73.5 to 74.0 us, 40 blocks reads 78
    # to 79, 10 blocks 100 to 108 and 5 blocks 173. That is `down_proj`'s cliff again, 20
    # blocks against 24 cube cores. `(64, 512, 4)` is best at M = 16 and 1.6% off the best at
    # M = 32, where `(128, 256, 2)` wins by 73.8 to 75.0 -- the constant pins either way.
    "out_proj": (VAL_WIDTH, HIDDEN, 64, 512, 4),  # 20 blocks, 73.5 us, 856 GB/s at M = 16
    # The attention layers, added by G5.14. `o_proj` used to share `out_proj`'s entry, the
    # two being the same `K=6144 N=5120` matmul; they parted ways over dtype, not shape.
    "o_proj": (VAL_WIDTH, HIDDEN, 64, 512, 4),          # 20 blocks,  37.3 us on the path
    "q_proj": (HIDDEN, ATTN_QG_WIDTH, 64, 1024, 8),     # 24 blocks,  71.7 us, 877 GB/s
    "k_proj": (HIDDEN, ATTN_KV_WIDTH, 64, 1024, 2),     #  8 blocks,  28.0 us, 188 GB/s
    "v_proj": (HIDDEN, ATTN_KV_WIDTH, 64, 1024, 2),     #  8 blocks,  27.7 us, 189 GB/s
    # The three above as one matmul over one concatenated weight, added by G5.21.
    "qkv_proj": (HIDDEN, ATTN_QKV_WIDTH, 128, 512, 2),  # 56 blocks,  90.2 us, 814 GB/s
}


def padded_batch(batch: int) -> int:
    """The M a decode launch must present: the batch rounded up to the cube's fractal."""
    return max(M_FRACTAL, (batch + M_FRACTAL - 1) // M_FRACTAL * M_FRACTAL)


# The projections the deployed W8A8 checkpoint leaves in BF16. One entry, and it is not a
# tuning choice: `contract.py` carries the weight's dtype as published, D1 settled on matching
# it, and the GDN `out_proj` is the only projection in the model it applies to.
BF16_MODULES = frozenset({"out_proj"})

# What a fake-quant reference should quantise on the decode path, which is `GDN_QUANT.weights`
# minus the one above. The config tuple still describes the Q4 kernel scheme, which the
# prefill path implements until G4.1 makes the same change.
DECODE_QUANT_WEIGHTS = ("in_proj_qkv.weight", "in_proj_z.weight")


# a2a3's two on-chip buffers that a decode matmul grows into, because `m_tile` is the batch
# (G5.6). They cap it, and both are reached at the same value.
ACC_BYTES = 131072          # accumulator
MAT_BYTES = 524288          # Mat


def max_batch() -> int:
    """The largest decode batch the tiles leave room for: 128 for the shipped set.

    Two independent bounds, each measured to the byte off-queue (G5.11):

    * the accumulator holds an `[m, n_tile]` FP32 output tile in two slots, so
      `2 * m * n_tile * 4 <= ACC_BYTES`. At the widest decode `n_tile`, 128, that is
      `m <= 128`, and batch 144 is refused at exactly 163840 bytes.
    * `bf16_linear` holds `8 * n_tile * k_tile + 4 * m * k_tile` of Mat (G5.30), so for
      `out_proj` at `(64, 512)` that is also `m <= 128`, and with the accumulator term
      halved out of the way batch 144 is refused at exactly 557056 bytes.

    Halving the widest `n_tile` therefore does not raise the ceiling, it only changes which
    buffer refuses first. `decode_mlp`'s widest decode `n_tile` is the same 128, so it does
    not lower this; a retune there that widens it would be refused by the compiler at a
    batch this function still allows.
    """
    n_tile = max(t[2] for t in DECODE_TILES.values())
    acc = ACC_BYTES // (2 * 4 * n_tile)
    bf16 = [DECODE_TILES[name] for name in BF16_MODULES if name in DECODE_TILES]
    mat = min(((MAT_BYTES - 8 * t[2] * t[3]) // (4 * t[3]) for t in bf16), default=acc)
    return max(M_FRACTAL, min(acc, mat) // M_FRACTAL * M_FRACTAL)


def build_kernel(module: str, batch: int, inline: bool = False):
    """One projection at a decode batch. `batch` is rounded up to `M_FRACTAL`.

    `BF16_MODULES` decides which matmul this is: a BF16 module reads a BF16 weight and takes
    no dequant scale, so its kernel has one fewer parameter than the rest.
    """
    if module not in DECODE_TILES:
        raise ValueError(f"unknown projection {module!r}; expected one of "
                         f"{sorted(DECODE_TILES)}")
    k, n, n_tile, k_tile, n_group = DECODE_TILES[module]
    m = padded_batch(batch)
    if module in BF16_MODULES:
        return bf16_linear.build_kernel(m, k, n, name=module, inline=inline, m_tile=m,
                                        n_tile=n_tile, k_tile=k_tile, m_group=1,
                                        n_group=n_group)
    return a8w8_linear.build_kernel(m, k, n, name=module, inline=inline, m_tile=m,
                                    n_tile=n_tile, k_tile=k_tile, m_group=1,
                                    n_group=n_group)


# Which projections belong to which layer type, so a per-layer weight figure cannot
# accidentally sum across both.
GDN_MODULES = ("in_proj_qkv", "in_proj_z", "out_proj")
ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
FUSED_ATTENTION_MODULES = ("qkv_proj", "o_proj")                 # the same bytes in two launches


def weight_bytes_per_layer(layer: str = "linear_attention") -> int:
    """Weight one layer's projections read, whatever the batch, at each one's own dtype.

    **`BF16_MODULES` doubles its entries**, which is not a detail: D1 took the GDN
    `out_proj` to BF16 to match the checkpoint (G5.30), so that one projection reads
    `6144 * 5120 * 2` bytes and a linear-attention layer streams **140.00 MiB** of
    projection weight rather than 110.00. Over 48 layers that is 1.41 GiB more per decode
    token, 5.8% of the whole 24640 MiB the 64 layers stream before `lm_head` (G5.9).

    `o_proj` and `out_proj` are the same `K=6144 N=5120` matmul and one measured tile serves
    both, but they are **no longer the same bytes**: the checkpoint quantises the attention
    one and not the GDN one, so each layer type names its own.

    `full_attention` and `full_attention_fused` return the same number by construction --
    the fused weight is the three concatenated, so fusing moves no bytes.
    """
    modules = {"linear_attention": GDN_MODULES, "full_attention": ATTENTION_MODULES,
               "full_attention_fused": FUSED_ATTENTION_MODULES}
    if layer not in modules:
        raise ValueError(f"unknown layer type {layer!r}; expected one of {sorted(modules)}")
    return sum(DECODE_TILES[m][0] * DECODE_TILES[m][1] * (2 if m in BF16_MODULES else 1)
               for m in modules[layer])


def fuse_qkv_weight(q_w, k_w, v_w, q_scale, k_scale, v_scale):
    """The one weight `qkv_proj` reads, from the three `q_proj`/`k_proj`/`v_proj` carry.

    Each weight is `[N, K]` INT8 and each scale `[N]` FP32, so the fusion is a row
    concatenation in the order `QKV_QG_BASE`, `QKV_K_BASE`, `QKV_V_BASE` name. Nothing is
    requantised: a row keeps its own scale, which is why one matmul over the concatenation
    is numerically the three matmuls and not an approximation of them.
    """
    import torch

    rows = [(q_w, ATTN_QG_WIDTH), (k_w, ATTN_KV_WIDTH), (v_w, ATTN_KV_WIDTH)]
    for w, n in rows:
        if tuple(w.shape) != (n, HIDDEN):
            raise ValueError(f"expected a [{n}, {HIDDEN}] weight, got {tuple(w.shape)}")
    return (torch.cat([q_w, k_w, v_w], dim=0),
            torch.cat([q_scale.reshape(-1), k_scale.reshape(-1), v_scale.reshape(-1)]))


__all__ = ["ATTENTION_MODULES", "ATTN_QKV_WIDTH", "DECODE_TILES", "FUSED_ATTENTION_MODULES",
           "GDN_MODULES", "M_FRACTAL", "QKV_K_BASE", "QKV_QG_BASE", "QKV_V_BASE",
           "max_batch",
           "build_kernel", "fuse_qkv_weight", "padded_batch", "weight_bytes_per_layer"]
