# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B final RMSNorm and LM head: `logits = norm(hidden) @ lm_w^T`.

Takes the rows logits are wanted for, not the whole sequence. The norm is the
zero-centred `1 + w` of `rmsnorm.py`, fused here so the normed rows stay on chip
between the two halves. `lm_w` is BF16, so this projection is not W8A8.

The L1 Mat buffer bounds `vocab_tile` and `lm_k_tile` **together**, not separately:
the K pipeline double-buffers both operands, so
`(row_tile + vocab_tile) * lm_k_tile * 2 bytes * 2` must fit 524288, which is
`(row_tile + vocab_tile) * lm_k_tile <= 131072`. A smaller contraction step buys a
wider vocabulary tile.

**The tiles below are right and the sweep that checked them found nothing.** 2425 MiB of
BF16 weight is a long enough stream to amortise everything, and at 16 rows this reaches
**98.2% of the box's vector copy roof** -- 2098.9 us, 1211.5 GB/s, the highest fraction
measured anywhere in this model. Over 18 legal `(vocab_tile, lm_k_tile)` pairs the spread
is 1.32x and the best, `vocab_tile = 128`, is **0.6% ahead**; at 32 rows the best is
`vocab_tile = 256 lm_k_tile = 256`, 1.6% ahead. Neither is worth a batch-dependent
constant. `lm_cores = 24` is one wave of the cube, which G5.14 found is what a projection
wants.

So the only lever left on this operator is the weight itself: it is the one projection in
the model still BF16, and 2425 MiB is 9.4% of what a decode token streams.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
ROWS_DYN = pl.dynamic("ROWS_DYN")           # rows the head scores

# model config
HIDDEN = QWEN3_8_27B.hidden_size
VOCAB = QWEN3_8_27B.vocab_size
EPS = QWEN3_8_27B.rms_norm_eps

# case shape
ROWS = 16               # rows logits are produced for; one per sequence

# tiling
ROW_TILE = 16
RMS_K_TILE = 128        # hidden elements per norm pass step
LM_K_TILE = 512         # contraction step; 512 BF16 columns is one L2 line
# The L1 Mat buffer is 512 KiB and the K pipeline double-buffers both operands,
# so (ROW_TILE + VOCAB_TILE) * LM_K_TILE * 2 bytes * 2 must fit: VOCAB_TILE
# stays under 240. 160 is the largest multiple of 16 under that which divides
# 248320 = 2^9 * 5 * 97 -- 192, the qwen3_14b value, does not.
VOCAB_TILE = 160
LM_CORES = 24           # one per cube core

# host only: bands that keep the oracle and the input draw off the peak
GOLDEN_VOCAB_BAND = 16384
INIT_ROW_BAND = 16384


HIDDEN_INV = 1.0 / HIDDEN


def build_kernel(row_tile: int = ROW_TILE, rms_k_tile: int = RMS_K_TILE,
                 lm_k_tile: int = LM_K_TILE, vocab_tile: int = VOCAB_TILE,
                 lm_cores: int = LM_CORES, inline: bool = False):
    """The norm and head at given tiles, over a padded row extent and a valid row count.

    **The row extent is padded; `valid_rows` says how many of them are real.** Without that
    split the ordinary case could not call this kernel at all: one sequence is one row, and
    every loop here steps a whole `row_tile` of 16, so a one-row tensor gave `pl.range` zero
    iterations and returned with `logits` untouched -- no error, no output (G3.6).

    So `hidden_states` and `logits` carry a row extent that is a multiple of `row_tile`, the
    loops cover all of it, and only the store is trimmed, with `pl.tensor.set_validshape`.
    The padded rows are normed and projected and then not written, which costs at most one
    tile of arithmetic and keeps every read inside the tensor. `qwen3_14b/rms_lm_head.py`
    carries the same idiom.
    """

    @(pl.jit.inline if inline else pl.jit)
    def _qwen3_rms_lm_head(
        hidden_states: pl.Tensor[[ROWS_DYN, HIDDEN], pl.BF16],
        norm_w: pl.Tensor[[HIDDEN], pl.BF16],
        lm_w: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
        logits: pl.Out[pl.Tensor[[ROWS_DYN, VOCAB], pl.FP32]],
        valid_rows: pl.Scalar[pl.INDEX],
    ):
        """The row extent must be a multiple of row_tile; `valid_rows` may be anything."""
        hidden_states.bind_dynamic(0, ROWS_DYN)
        logits.bind_dynamic(0, ROWS_DYN)
        row_dim = pl.tensor.dim(hidden_states, 0)
        w_row = pl.reshape(norm_w[:], [1, HIDDEN])
        normed = pl.create_tensor([row_dim, HIDDEN], dtype=pl.BF16)
        for r0 in pl.parallel(0, row_dim, row_tile):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
                partial_sq = pl.full([1, row_tile], dtype=pl.FP32, value=0.0)
                for k0 in pl.range(0, HIDDEN, rms_k_tile):
                    sq_x = pl.cast(hidden_states[r0 : r0 + row_tile, k0 : k0 + rms_k_tile],
                                   target_type=pl.FP32)
                    sq_prod = pl.mul(sq_x, sq_x)
                    sq_row = pl.reshape(pl.row_sum(sq_prod), [1, row_tile])
                    partial_sq = pl.add(partial_sq, sq_row)
                mean_sq = pl.mul(partial_sq, HIDDEN_INV)
                variance = pl.reshape(mean_sq, [row_tile, 1])
                inv_rms = pl.rsqrt(pl.add(variance, EPS), high_precision=True)

                for k0 in pl.range(0, HIDDEN, rms_k_tile):
                    norm_x = pl.cast(hidden_states[r0 : r0 + row_tile, k0 : k0 + rms_k_tile],
                                     target_type=pl.FP32)
                    w_tile = pl.cast(w_row[:, k0 : k0 + rms_k_tile], target_type=pl.FP32)
                    gamma = pl.add(w_tile, 1.0)
                    scaled = pl.row_expand_mul(norm_x, inv_rms)
                    weighted = pl.col_expand_mul(scaled, gamma)
                    normed[r0 : r0 + row_tile, k0 : k0 + rms_k_tile] = pl.cast(
                        weighted, target_type=pl.BF16, mode="rint")

        # One grid-stride over the vocab tiles: each block owns a disjoint set of
        # columns, so there is no split-K and no atomic store.
        for core in pl.spmd(lm_cores, name_hint="lm_head"):
            for r0 in pl.range(0, row_dim, row_tile):
                for vb in pl.range(core, VOCAB // vocab_tile, lm_cores):
                    v0 = vb * vocab_tile
                    acc = pl.create_tensor([row_tile, vocab_tile], dtype=pl.FP32)
                    for kb in pl.pipeline(0, HIDDEN // lm_k_tile, stage=2):
                        k0 = kb * lm_k_tile
                        lm_h_tile = normed[r0 : r0 + row_tile, k0 : k0 + lm_k_tile]
                        lm_w_tile = lm_w[v0 : v0 + vocab_tile, k0 : k0 + lm_k_tile]
                        acc = pl.matmul_acc(acc, lm_h_tile, lm_w_tile, b_trans=True, init_cond=(kb == 0))
                    live = pl.min(row_tile, valid_rows - r0)
                    trimmed = pl.tensor.set_validshape(acc, live, vocab_tile)
                    logits[r0 : r0 + row_tile, v0 : v0 + vocab_tile] = trimmed
        return logits

    return _qwen3_rms_lm_head


qwen3_rms_lm_head = build_kernel(inline=True)
qwen3_rms_lm_head_test = build_kernel()




def _banded_normal(rows: int, cols: int, seed: int, scale: float = 1.0):
    """A BF16 `[rows, cols]` normal draw built one band at a time.

    `torch.randn(248320, 5120)` is 5.09 GB of FP32 before it is narrowed, and
    simulator CI builds these at the full vocabulary. Banding keeps the peak at
    the BF16 result plus one band.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    out = torch.empty(rows, cols, dtype=torch.bfloat16)
    for r0 in range(0, rows, INIT_ROW_BAND):
        rows_here = min(INIT_ROW_BAND, rows - r0)
        band = torch.randn(rows_here, cols, generator=gen)
        if scale != 1.0:
            band *= scale
        out[r0 : r0 + rows_here] = band.to(torch.bfloat16)
    return out

def padded_rows(rows: int, row_tile: int = ROW_TILE) -> int:
    """The row extent a launch must present for `rows` real rows."""
    return max(row_tile, (rows + row_tile - 1) // row_tile * row_tile)


def build_tensor_specs(rows: int = ROWS, hidden: int = HIDDEN, vocab: int = VOCAB,
                       row_tile: int = ROW_TILE):
    """Fixtures for `rows` real rows, in a tensor padded up to `row_tile`.

    The padding rows are drawn like the rest rather than zeroed: a zero row has zero
    variance, so it would normalise to whatever `EPS` allows and could not distinguish a
    kernel that trims its store from one that writes garbage into the tail.
    """
    import torch
    from golden import ScalarSpec, TensorSpec

    padded = padded_rows(rows, row_tile)

    def init_hidden():
        gen = torch.Generator().manual_seed(42)
        return torch.randn(padded, hidden, generator=gen).to(torch.bfloat16)

    def init_norm_w():
        gen = torch.Generator().manual_seed(43)
        return torch.randn(hidden, generator=gen).to(torch.bfloat16)

    def init_lm_w():
        return _banded_normal(vocab, hidden, seed=44, scale=hidden ** -0.5)

    return [
        TensorSpec("hidden_states", [padded, hidden], torch.bfloat16, init_value=init_hidden),
        TensorSpec("norm_w", [hidden], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("lm_w", [vocab, hidden], torch.bfloat16, init_value=init_lm_w),
        TensorSpec("logits", [padded, vocab], torch.float32),
        # A runtime parameter, not specialised into the artifact: one compiled program has
        # to serve every row count, which is the whole point of the trim.
        ScalarSpec("valid_rows", torch.int64, rows, compile_runtime=True),
    ]


def golden_qwen3_rms_lm_head(eps: float = EPS):
    """The oracle at one `eps`, which must be the `eps` the kernel was built with.

    A factory rather than a bare function: `build_kernel` takes `eps`, so a
    golden reading the module constant would score any non-default build
    against the wrong oracle.
    """
    import torch

    import reference

    def golden(tensors):
        # The kernel narrows the normed rows to BF16 before the head, as the
        # module does; leaving them in float64 would score the dtype rather
        # than the kernel.
        # Only the valid rows, matching the kernel's trimmed store: the padded tail is
        # normed and projected on device and then not written, so scoring it would compare
        # this oracle against whatever `logits` was initialised to.
        rows = tensors["valid_rows"]
        normed = reference.rmsnorm(tensors["hidden_states"][:rows], tensors["norm_w"], eps)
        normed_bf = normed.to(torch.bfloat16).to(reference.REF_DTYPE)
        lm_w, out = tensors["lm_w"], tensors["logits"]
        # Widen the weight one vocabulary band at a time. Whole, it is 9.47 GiB
        # in float64 against 2.37 in BF16, and simulator CI runs this file at
        # the full vocabulary.
        for v0 in range(0, lm_w.shape[0], GOLDEN_VOCAB_BAND):
            band = lm_w[v0 : v0 + GOLDEN_VOCAB_BAND].to(reference.REF_DTYPE)
            out[:rows, v0 : v0 + band.shape[0]] = (normed_bf @ band.t()).to(torch.float32)

    return golden


# The logits bound. A fixed `atol` judges the sample count rather than the kernel: at 48
# rows it rejects 24 of 11919360 elements, all near-zero logits about 1.4e-4 out, where the
# relative Frobenius is 5.80e-06 against 2.47e-06 at 16 rows. Same quality, three times the
# draws. `down_proj`-style cancellation over a 5120-term contraction is what puts logits
# near zero, so the bound is a norm ratio.
FROB_GATE = 1e-4


def compare_logits():
    """Relative Frobenius against the float64 oracle, with the distribution reported."""
    def compare(actual, expected, **_kwargs):
        a, e = actual.double(), expected.double()
        frob = float((a - e).norm() / e.norm())
        diff = (a - e).abs()
        over = int((diff > 1e-4 + 1e-2 * e.abs()).sum())
        print(f"[stats] logits rel frobenius {frob:.3e} (gate {FROB_GATE:.0e}); "
              f"max abs {float(diff.max()):.3e}; {over} of {a.numel()} outside a fixed "
              f"atol=1e-4 rtol=1e-2", flush=True)
        return frob <= FROB_GATE, f"rel frobenius {frob:.3e} (<= {FROB_GATE:.0e})"

    return compare


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--vocab", type=int, default=VOCAB)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run(
        fn=qwen3_rms_lm_head_test,
        specs=build_tensor_specs(rows=args.rows, vocab=args.vocab),
        golden_fn=golden_qwen3_rms_lm_head(args.eps),
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=1e-2,
        atol=1e-4,
        compare_fn={"logits": compare_logits()},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
