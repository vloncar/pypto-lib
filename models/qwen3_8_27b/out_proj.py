# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""`out_proj` in A8W8: the block's last step, the normed output back to hidden.

    out = y @ W^T          [T, 6144] -> [T, 5120], INT8 both sides, bf16 out

The arithmetic, the golden and the CLI are `a8w8_linear`; this file is the
shape, the tiles, and where the weight and the references come from.

Its activation is not quantized here: `gated_rmsnorm` writes `y_q` and
`y_scale` in its own epilogue, so the normed values never reach GM in bf16.
That is why there is no `quant_y` kernel beside `quant_x`.
"""
import a8w8_linear

from chunk_cumsum import CHUNK_TILE
from config import QWEN3_8_27B

# model config
K = QWEN3_8_27B.value_width                 # contraction: H value heads of D
N = QWEN3_8_27B.hidden_size                 # back to the residual stream

MODULE = "out_proj"
# validation
# one block of M_GROUP * M_TILE token rows, then two. The token count must be a multiple of 1024.
DEFAULT_CASES = (1024, 2048)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
M_TILE = 128
N_TILE = 128
K_TILE = 512
K_STAGGER = 1               # rotate this block's K start; see `a8w8_linear`
RING_SLOTS = 2              # GM transport depth; the on-chip reserve stays at 1
M_GROUP = 8
N_GROUP = 2



def build_kernel(t: int = BENCH_SEQ, k: int = K, n: int = N, inline: bool = False,
                 m_tile: int = M_TILE, n_tile: int = N_TILE, k_tile: int = K_TILE,
                 m_group: int = M_GROUP, n_group: int = N_GROUP):
    return a8w8_linear.build_kernel(t, k, n, name=MODULE, inline=inline, m_tile=m_tile,
                                    n_tile=n_tile, k_tile=k_tile, m_group=m_group,
                                    n_group=n_group, k_stagger=K_STAGGER, ring_slots=RING_SLOTS)


_ACT_CACHE: dict[tuple, dict] = {}


def _activation(t: int, chunk: int, weights: str | None):
    """`gated_rmsnorm`'s epilogue output, from the block chain that produces it.

    The whole float64 block runs to get here -- the normed output is the last
    thing before this projection -- so the three tensors share one chain and
    one cache.
    """
    import reference

    def load():
        key = (t, chunk, weights)
        cache = _ACT_CACHE.get(key)
        if cache is None:
            def draw(name):
                return reference.block_inputs(name, t, QWEN3_8_27B, chunk,
                                              weights=weights, quant_y=True)()
            cache = _ACT_CACHE[key] = dict(
                x=draw("y").reshape(t, -1), q=draw("y_q"),
                scale=draw("y_scale").float().reshape(1, -1))
        return cache

    return load


def build_tensor_specs(t: int = BENCH_SEQ, k: int = K, n: int = N, chunk: int = CHUNK_TILE,
                       weights: str | None = None, **_):
    act = _activation(t, chunk, weights)
    w_q, w_scale = a8w8_linear.quantized_weight(QWEN3_8_27B, MODULE, weights)
    return a8w8_linear.build_specs(t, k, n, lambda: act()["q"], lambda: act()["scale"],
                                   w_q, w_scale)


def compare_y(t: int, chunk: int, weights: str | None):
    act = _activation(t, chunk, weights)
    truth, fakequant = a8w8_linear.references(QWEN3_8_27B, MODULE, act, weights)
    return a8w8_linear.compare_y(truth, fakequant, "out")


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    parser = a8w8_linear.add_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
        M_TILE, N_TILE, K_TILE, M_GROUP, N_GROUP)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % 1024:
        parser.error("--seq-len must be a multiple of 1024")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")
    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    for t in cases:
        print(f"[case] out_proj T={t}", flush=True)
        result = run(
            fn=build_kernel(t=t, m_tile=args.m_tile, n_tile=args.n_tile,
                            k_tile=args.k_tile, m_group=args.m_group, n_group=args.n_group),
            specs=build_tensor_specs(t=t, weights=args.weights),
            golden_fn=a8w8_linear.golden_fn,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-3,
            atol=1e-5,
            compare_fn={"y": compare_y(t, CHUNK_TILE, args.weights)},
            compile_only=args.compile_only,
        )
        a8w8_linear.report("out_proj", t, args, result, K, N)
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run
            # lane, so every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
