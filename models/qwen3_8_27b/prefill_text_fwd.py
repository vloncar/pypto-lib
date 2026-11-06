# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Qwen3.8-27B packed text prefill from token IDs to logits and updated hybrid caches."""
import pypto.language as pl

import embedding
import prefill_fwd
import rms_lm_head
import short_conv
from chunk_h import FSTATE_DYN
from chunk_cumsum import CHUNK_TILE
from config import KV_PAGE_SIZE, QWEN3_8_27B
from embedding import qwen3_embedding
from prefill_kv_write import KVROWS_DYN

# model config, read off the stack so the two signatures cannot drift
C = prefill_fwd.C
CQKV = prefill_fwd.CQKV
CV = prefill_fwd.CV
KCONV = prefill_fwd.KCONV
H = prefill_fwd.H
DV = prefill_fwd.DV
INTER = prefill_fwd.INTER
HEAD_DIM = prefill_fwd.HEAD_DIM
ROT = prefill_fwd.ROT
QG_WIDTH = prefill_fwd.QG_WIDTH
KV_WIDTH = prefill_fwd.KV_WIDTH
O_WIDTH = prefill_fwd.O_WIDTH
ROW_TILE = prefill_fwd.ROW_TILE
BIAS_WIDTH = prefill_fwd.BIAS_WIDTH
INTERVAL = prefill_fwd.INTERVAL
GROUPS = prefill_fwd.GROUPS
EPS = QWEN3_8_27B.rms_norm_eps
VOCAB = QWEN3_8_27B.vocab_size

# validation
# One group is the floor and it is what this entry defaults to: the stack's own entry
# gates 1 and 2 groups, and what this one adds -- the embedding, the zero residual, the
# row gather and the head -- is the same at every group count.
DEFAULT_GROUPS = (1,)
DEFAULT_SEQ = prefill_fwd.DEFAULT_SEQ
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len
LOGIT_ROWS = 1          # rows the head scores: one per packed sequence
SEED = prefill_fwd.SEED


def text_ring_heap(t: int = DEFAULT_SEQ) -> list[int]:
    """Size both nested text/stack rings for a layer's live set."""
    heap = prefill_fwd.fwd_ring_heap(t)
    heap[2] = heap[1]
    heap[3] = heap[1]
    return heap


def build_kernel(t: int = BENCH_SEQ, groups: int = GROUPS, chunk: int = CHUNK_TILE,
                 logit_rows: int = LOGIT_ROWS, inline: bool = False,
                 max_pages: int | None = None, kv_layer_stride: int | None = None, rows: int = 1):
    """The whole text prefill at one shape. `logit_rows` is how many rows the head scores.

    **Every FP32 residual-stream tensor is a parameter, not one created here.** The GDN
    chain reads its token extent from the residual it is handed -- `rmsnorm`'s residual
    stage takes `pl.tensor.dim(residual, 0)` and every stage below inherits it -- and a
    tensor created in this body for either the stack's residual input or its residual
    output makes that extent unprovable at `chunk_h`'s carried `pl.range`, six inline
    levels down. The BF16 `delta` side is unaffected, which is why the embedding writes a
    created tensor and the residual does not. `residual_out` and `mlp_out` are outputs
    rather than internals for that reason, and gating them costs nothing.
    """
    layers = groups * INTERVAL
    gdn_layers = groups * prefill_fwd.LINEAR_PER_GROUP
    max_pages = max_pages if max_pages is not None else t // KV_PAGE_SIZE
    rows_pad = rms_lm_head.padded_rows(logit_rows)
    stack = prefill_fwd.build_kernel(t=t, groups=groups, chunk=chunk, inline=True,
                                     max_pages=max_pages, kv_layer_stride=kv_layer_stride, rows=rows)
    head = rms_lm_head.build_kernel(inline=True)

    @(pl.jit.inline if inline else pl.jit)
    def qwen3_prefill_text_fwd(
        input_ids: pl.Tensor[[t], pl.INT32],
        embed_weight: pl.Tensor[[VOCAB, C], pl.BF16],
        residual: pl.Tensor[[t, C], pl.FP32],
        norm_in_w: pl.Tensor[[layers * C], pl.BF16],
        norm_post_w: pl.Tensor[[layers * C], pl.BF16],
        w_gate: pl.Tensor[[layers * INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[layers * INTER], pl.FP32],
        w_up: pl.Tensor[[layers * INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[layers * INTER], pl.FP32],
        w_down: pl.Tensor[[layers * C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[layers * C], pl.FP32],
        w_qkv: pl.Tensor[[gdn_layers * CQKV, C], pl.INT8],
        w_qkv_scale: pl.Tensor[[gdn_layers * CQKV], pl.FP32],
        w_z: pl.Tensor[[gdn_layers * CV, C], pl.INT8],
        w_z_scale: pl.Tensor[[gdn_layers * CV], pl.FP32],
        w_ab: pl.Tensor[[gdn_layers * 2 * H, C], pl.BF16],
        conv_w: pl.Tensor[[gdn_layers * KCONV, CQKV], pl.FP32],
        a_log: pl.Tensor[[gdn_layers * H], pl.BF16],
        dt_bias: pl.Tensor[[gdn_layers * H], pl.BF16],
        norm_w: pl.Tensor[[gdn_layers * DV], pl.BF16],
        w_out: pl.Tensor[[gdn_layers * C, CV], pl.BF16],
        w_q: pl.Tensor[[groups * QG_WIDTH, C], pl.INT8],
        w_q_scale: pl.Tensor[[groups * QG_WIDTH], pl.FP32],
        w_k: pl.Tensor[[groups * KV_WIDTH, C], pl.INT8],
        w_k_scale: pl.Tensor[[groups * KV_WIDTH], pl.FP32],
        w_v: pl.Tensor[[groups * KV_WIDTH, C], pl.INT8],
        w_v_scale: pl.Tensor[[groups * KV_WIDTH], pl.FP32],
        w_o: pl.Tensor[[groups * C, O_WIDTH], pl.INT8],
        w_o_scale: pl.Tensor[[groups * C], pl.FP32],
        q_norm_w: pl.Tensor[[groups * HEAD_DIM], pl.BF16],
        k_norm_w: pl.Tensor[[groups * HEAD_DIM], pl.BF16],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        mask_strict: pl.Tensor[[chunk, chunk], pl.FP32],
        eye: pl.Tensor[[chunk, chunk], pl.FP16],
        m_diag: pl.Tensor[[chunk, chunk], pl.FP16],
        m_low: pl.Tensor[[chunk, chunk], pl.FP16],
        rope_cos: pl.Tensor[[t, ROT], pl.FP32],
        rope_sin: pl.Tensor[[t, ROT], pl.FP32],
        mask_bias: pl.Tensor[[ROW_TILE, BIAS_WIDTH], pl.FP32],
        row_base: pl.Tensor[[t], pl.INT32],
        row_owner: pl.Tensor[[t], pl.INT32],
        row_ends: pl.Tensor[[rows], pl.INT32],
        state_rows: pl.Tensor[[gdn_layers * rows], pl.INT32],
        page_bases: pl.Tensor[[rows], pl.INT32],
        visible_pages: pl.Tensor[[rows], pl.INT32],
        block_table: pl.Tensor[[rows, max_pages], pl.INT32],
        conv_state: pl.InOut[pl.Tensor[[short_conv.CSTATE_DYN, KCONV - 1, CQKV], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, DV, DV], pl.FP32]],
        k_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[KVROWS_DYN, KV_WIDTH], pl.BF16]],
        final_norm_w: pl.Tensor[[C], pl.BF16],
        lm_w: pl.Tensor[[VOCAB, C], pl.BF16],
        logit_ids: pl.Tensor[[rows_pad], pl.INT32],
        residual_out: pl.Out[pl.Tensor[[t, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[t, C], pl.BF16]],
        logits: pl.Out[pl.Tensor[[rows_pad, VOCAB], pl.FP32]],
        valid_rows: pl.Scalar[pl.INDEX],
    ):
        """`residual` is zero at the start of a sequence; `logit_ids[i]` names row `i`'s token."""
        conv_state.bind_dynamic(0, short_conv.CSTATE_DYN)
        final_state.bind_dynamic(0, FSTATE_DYN)
        k_pages.bind_dynamic(0, KVROWS_DYN)
        v_pages.bind_dynamic(0, KVROWS_DYN)
        hidden = pl.create_tensor([t, C], dtype=pl.BF16)
        qwen3_embedding(input_ids, embed_weight, hidden)

        stack(
            residual, hidden,
            norm_in_w, norm_post_w,
            w_gate, w_gate_scale, w_up, w_up_scale, w_down, w_down_scale,
            w_qkv, w_qkv_scale, w_z, w_z_scale, w_ab, conv_w, a_log, dt_bias, norm_w, w_out,
            w_q, w_q_scale, w_k, w_k_scale, w_v, w_v_scale, w_o, w_o_scale, q_norm_w, k_norm_w,
            tril, mask_strict, eye, m_diag, m_low,
            rope_cos, rope_sin, mask_bias, row_base, row_owner, row_ends, state_rows,
            page_bases, visible_pages, block_table, conv_state, final_state, k_pages, v_pages,
            residual_out, mlp_out,
        )

        # The head scores named rows, so they are gathered into one tile-aligned pair.
        head_resid = pl.create_tensor([rows_pad, C], dtype=pl.FP32)
        head_delta = pl.create_tensor([rows_pad, C], dtype=pl.BF16)
        for row in pl.spmd(rows_pad, name_hint="prefill_text_logit_rows"):
            src = pl.cast(pl.tensor.read(logit_ids, [row]), pl.INDEX)
            head_resid[row : row + 1, 0:C] = residual_out[src : src + 1, 0:C]
            head_delta[row : row + 1, 0:C] = mlp_out[src : src + 1, 0:C]

        head(head_resid, head_delta, final_norm_w, lm_w, logits, valid_rows)
        return residual_out, mlp_out, logits, conv_state, final_state, k_pages, v_pages

    return qwen3_prefill_text_fwd


def build_l3_kernel(t: int, *, rows: int, max_pages: int, kv_layer_stride: int,
                    state_count: int, groups: int = GROUPS):
    """Build a HOST prefill entry with concrete request and resident-cache extents."""
    from serving_contract import QWEN3_8_27B_SERVING_CONTRACT

    QWEN3_8_27B_SERVING_CONTRACT.validate_prefill_tokens(t)
    if not 1 <= rows <= t // prefill_fwd.KV_TILE:
        raise ValueError("each prefill request must own at least one complete attention tile")
    gdn_layers = groups * prefill_fwd.LINEAR_PER_GROUP
    if max_pages < 1 or kv_layer_stride < max_pages * KV_PAGE_SIZE or state_count < gdn_layers * rows:
        raise ValueError("prefill cache extents must cover all requests and layers")
    layers = groups * INTERVAL
    pages = max_pages
    kv_rows = groups * kv_layer_stride
    logit_rows = rms_lm_head.padded_rows(rows)
    kernel = build_kernel(
        t=t, groups=groups, rows=rows, logit_rows=rows,
        max_pages=max_pages, kv_layer_stride=kv_layer_stride,
    )

    @pl.jit.host
    def l3_prefill_fwd(
        input_ids: pl.Tensor[[t], pl.INT32],
        embed_weight: pl.Tensor[[VOCAB, C], pl.BF16],
        residual: pl.Tensor[[t, C], pl.FP32],
        norm_in_w: pl.Tensor[[layers * C], pl.BF16],
        norm_post_w: pl.Tensor[[layers * C], pl.BF16],
        w_gate: pl.Tensor[[layers * INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[layers * INTER], pl.FP32],
        w_up: pl.Tensor[[layers * INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[layers * INTER], pl.FP32],
        w_down: pl.Tensor[[layers * C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[layers * C], pl.FP32],
        w_qkv: pl.Tensor[[gdn_layers * CQKV, C], pl.INT8],
        w_qkv_scale: pl.Tensor[[gdn_layers * CQKV], pl.FP32],
        w_z: pl.Tensor[[gdn_layers * CV, C], pl.INT8],
        w_z_scale: pl.Tensor[[gdn_layers * CV], pl.FP32],
        w_ab: pl.Tensor[[gdn_layers * 2 * H, C], pl.BF16],
        conv_w: pl.Tensor[[gdn_layers * KCONV, CQKV], pl.FP32],
        a_log: pl.Tensor[[gdn_layers * H], pl.BF16],
        dt_bias: pl.Tensor[[gdn_layers * H], pl.BF16],
        norm_w: pl.Tensor[[gdn_layers * DV], pl.BF16],
        w_out: pl.Tensor[[gdn_layers * C, CV], pl.BF16],
        w_q: pl.Tensor[[groups * QG_WIDTH, C], pl.INT8],
        w_q_scale: pl.Tensor[[groups * QG_WIDTH], pl.FP32],
        w_k: pl.Tensor[[groups * KV_WIDTH, C], pl.INT8],
        w_k_scale: pl.Tensor[[groups * KV_WIDTH], pl.FP32],
        w_v: pl.Tensor[[groups * KV_WIDTH, C], pl.INT8],
        w_v_scale: pl.Tensor[[groups * KV_WIDTH], pl.FP32],
        w_o: pl.Tensor[[groups * C, O_WIDTH], pl.INT8],
        w_o_scale: pl.Tensor[[groups * C], pl.FP32],
        q_norm_w: pl.Tensor[[groups * HEAD_DIM], pl.BF16],
        k_norm_w: pl.Tensor[[groups * HEAD_DIM], pl.BF16],
        tril: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
        mask_strict: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
        eye: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
        m_diag: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
        m_low: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
        rope_cos: pl.Tensor[[t, ROT], pl.FP32],
        rope_sin: pl.Tensor[[t, ROT], pl.FP32],
        mask_bias: pl.Tensor[[ROW_TILE, BIAS_WIDTH], pl.FP32],
        row_base: pl.Tensor[[t], pl.INT32],
        row_owner: pl.Tensor[[t], pl.INT32],
        row_ends: pl.Tensor[[rows], pl.INT32],
        state_rows: pl.Tensor[[gdn_layers * rows], pl.INT32],
        page_bases: pl.Tensor[[rows], pl.INT32],
        visible_pages: pl.Tensor[[rows], pl.INT32],
        block_table: pl.Tensor[[rows, pages], pl.INT32],
        conv_state: pl.InOut[pl.Tensor[[state_count, KCONV - 1, CQKV], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[state_count, H, DV, DV], pl.FP32]],
        k_pages: pl.InOut[pl.Tensor[[kv_rows, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[kv_rows, KV_WIDTH], pl.BF16]],
        final_norm_w: pl.Tensor[[C], pl.BF16],
        lm_w: pl.Tensor[[VOCAB, C], pl.BF16],
        logit_ids: pl.Tensor[[logit_rows], pl.INT32],
        residual_out: pl.Out[pl.Tensor[[t, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[t, C], pl.BF16]],
        logits: pl.Out[pl.Tensor[[logit_rows, VOCAB], pl.FP32]],
        valid_rows: pl.Scalar[pl.INDEX],
    ):
        return kernel(
            input_ids, embed_weight, residual, norm_in_w, norm_post_w,
            w_gate, w_gate_scale, w_up, w_up_scale, w_down, w_down_scale,
            w_qkv, w_qkv_scale, w_z, w_z_scale, w_ab, conv_w, a_log, dt_bias,
            norm_w, w_out, w_q, w_q_scale, w_k, w_k_scale, w_v, w_v_scale,
            w_o, w_o_scale, q_norm_w, k_norm_w, tril, mask_strict, eye, m_diag,
            m_low, rope_cos, rope_sin, mask_bias, row_base, row_owner, row_ends, state_rows,
            page_bases, visible_pages, block_table,
            conv_state, final_state, k_pages, v_pages, final_norm_w, lm_w,
            logit_ids, residual_out, mlp_out, logits, valid_rows,
        )

    return l3_prefill_fwd

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def logit_id_table(t: int, rows: int = LOGIT_ROWS, row_tile: int = rms_lm_head.ROW_TILE,
                   starts: tuple[int, ...] | None = None):
    """Padded final-token indices for explicit requests or evenly spaced scored spans."""
    import torch

    import attn_core

    starts = attn_core.row_starts(t, rows) if starts is None else list(starts)
    if len(starts) != rows:
        raise ValueError("logit rows must name every packed request")
    ends = starts[1:] + [t]
    last = [end - 1 for end in ends]
    padded = rms_lm_head.padded_rows(rows, row_tile)
    return torch.tensor(last + [last[-1]] * (padded - rows), dtype=torch.int32)


def build_tensor_specs(t: int = DEFAULT_SEQ, groups: int = GROUPS, chunk: int = CHUNK_TILE,
                       logit_rows: int = LOGIT_ROWS, real: bool = False,
                       weights_dir: str | None = None, seed: int = SEED,
                       page_base: int = 0, max_pages: int | None = None,
                       kv_layer_stride: int | None = None, resume: bool = False,
                       starts: tuple[int, ...] = (0,), bases: tuple[int, ...] | None = None):
    """The stack's fixtures and cache pools, plus the text input and head tensors.

    The stacked weights, the shared tables and `row_base` are `prefill_fwd`'s unchanged and
    in its order, so one spec list serves both signatures. `input_ids` and `embed_weight`
    are `embedding.py`'s own draws and `lm_w` is `rms_lm_head.py`'s, so every stage here is
    gated on the tensors its own entry was gated on.

    **`real` covers the three non-layer tensors too**, from the one dump
    `weights.py --fetch --mixer text` writes: the embedding table, the model's final norm
    and the LM head, 4.74 GiB of BF16 between them. Without that dump a real run stops
    with the command that fetches it rather than quietly drawing them.
    """
    import torch
    from golden import TensorSpec

    import weights as weights_mod

    stack_specs = prefill_fwd.build_tensor_specs(t=t, groups=groups, chunk=chunk, real=real,
                                                 weights_dir=weights_dir, seed=seed,
                                                 page_base=page_base, max_pages=max_pages,
                                                 kv_layer_stride=kv_layer_stride, resume=resume,
                                                 starts=starts, bases=bases)
    carried = [spec for spec in stack_specs
               if spec.name not in ("residual", "delta", "residual_out", "mlp_out")]
    outputs = [spec for spec in stack_specs if spec.name in ("residual_out", "mlp_out")]
    # Zero, not the stack's drawn residual: the stream starts at the embedding, so the
    # first layer's addition has to produce the embedding itself.
    zero_resid = TensorSpec("residual", [t, C], torch.float32, init_value=torch.zeros)
    embed_specs = {spec.name: spec for spec in embedding.build_tensor_specs(t=t)}
    head_specs = {spec.name: spec for spec in rms_lm_head.build_tensor_specs(rows=logit_rows)}
    padded = rms_lm_head.padded_rows(logit_rows)
    embed_init = embed_specs["embed_weight"].init_value
    norm_init = head_specs["norm_w"].init_value
    lm_init = head_specs["lm_w"].init_value
    if real:
        path = weights_mod.text_path(weights_dir)
        if not path.is_file():
            raise SystemExit(f"[prefill_text_fwd] {path} is not on disk; fetch it with\n"
                             f"  python weights.py --fetch --mixer text")
        # Loaded once and shared: the three are 4.74 GiB, and a per-spec load would read
        # the dump three times.
        text = torch.load(path, weights_only=True)
        embed_init = lambda: text["embed_weight"].to(torch.bfloat16)        # noqa: E731
        norm_init = lambda: text["final_norm_w"].to(torch.bfloat16)         # noqa: E731
        lm_init = lambda: text["lm_w"].to(torch.bfloat16)                   # noqa: E731
    return ([embed_specs["input_ids"],
             TensorSpec("embed_weight", [VOCAB, C], torch.bfloat16, init_value=embed_init),
             zero_resid] + carried + [
        TensorSpec("final_norm_w", [C], torch.bfloat16, init_value=norm_init),
        TensorSpec("lm_w", [VOCAB, C], torch.bfloat16, init_value=lm_init),
        TensorSpec("logit_ids", [padded], torch.int32,
                   init_value=lambda: logit_id_table(t, logit_rows, starts=starts if len(starts) > 1 else None)),
    ] + outputs + [
        head_specs["logits"],
        head_specs["valid_rows"],
    ])


# ---------------------------------------------------------------------------
# Oracle and acceptance
# ---------------------------------------------------------------------------


def golden_prefill_text_fwd(t: int, groups: int = GROUPS, chunk: int = CHUNK_TILE,
                            eps: float = EPS):
    """Both chains from the token ids to the logits; the truth stashed for the gate.

    The embedding is a gather, which torch indexing does exactly. The stack is
    `prefill_fwd.stack_chains`, so the whole-model oracle and the stack's own entry score
    the same arithmetic over the same weights. The norm and head then run on the scored
    rows only and band the vocabulary, as `rms_lm_head`'s own oracle does: the whole
    sequence's logits in float64 are neither computable nor comparable here.
    """
    import torch

    import prefill_layer
    import reference

    def norm_and_head(resid, delta, norm_w, lm_w, rows, shape):
        out = torch.zeros(shape, dtype=reference.REF_DTYPE)
        total = (resid[rows].double() + delta[rows].double()).float()
        normed = reference.rmsnorm(total.double(), norm_w, eps)
        normed_bf = normed.to(torch.bfloat16).to(reference.REF_DTYPE)
        band = rms_lm_head.GOLDEN_VOCAB_BAND
        for v0 in range(0, lm_w.shape[0], band):
            piece = lm_w[v0 : v0 + band].to(reference.REF_DTYPE)
            out[: rows.numel(), v0 : v0 + piece.shape[0]] = normed_bf @ piece.t()
        return out

    def golden(tensors):
        ids = tensors["input_ids"].long()
        embed = tensors["embed_weight"][ids].to(reference.REF_DTYPE)
        resid_in = tensors["residual"].to(reference.REF_DTYPE)
        resid_t, delta_t, resid_q, delta_q = prefill_fwd.stack_chains(
            tensors, resid_in, embed, t, groups, chunk)

        # Score the written prefix. Host zeros beyond it are placeholders for undefined device rows.
        live = int(tensors["valid_rows"])
        rows = tensors["logit_ids"].long()[:live]
        norm_w, lm_w = tensors["final_norm_w"], tensors["lm_w"]
        shape = (tensors["logit_ids"].numel(), lm_w.shape[0])
        truth = norm_and_head(resid_t, delta_t, norm_w, lm_w, rows, shape)
        chain = norm_and_head(resid_q, delta_q, norm_w, lm_w, rows, shape)
        prefill_layer.record_truth(residual_out=resid_t, mlp_out=delta_t, logits=truth)
        tensors["residual_out"].copy_(resid_q.to(torch.float32))
        tensors["mlp_out"].copy_(delta_q.to(torch.bfloat16))
        tensors["logits"].copy_(chain.to(torch.float32))

    return golden


# Acceptance. The logits are one unquantised BF16 projection downstream of the stack, so
# the gate is the stack's: the distance to the fake-quant chain in units of what the
# activation path costs, with the stack's bounds rather than a single layer's. The LM head
# contracts 5120 terms that cancel, which is why this is a norm ratio and not an `atol` --
# `rms_lm_head`'s own entry makes the same argument for the same reason.
TEXT_MAX_VS_GOLDEN = prefill_fwd.STACK_MAX_VS_GOLDEN
TEXT_MAX_FLOOR = prefill_fwd.STACK_MAX_FLOOR


def compare_fns(groups: int = GROUPS, t: int = BENCH_SEQ) -> dict:
    """The gates: all three outputs against the quantisation floor, at the stack's bounds.

    The stack's own two outputs are gated here as well as in `prefill_fwd`'s entry. They
    cost nothing -- the oracle computes them on the way to the logits -- and they say
    whether a failure is the stack's or this file's three stages.
    """
    import prefill_layer

    result = prefill_fwd.compare_fns(groups, t)
    result["logits"] = prefill_layer.compare_vs_floor(
        "logits", max_vs_golden=TEXT_MAX_VS_GOLDEN, max_floor=TEXT_MAX_FLOOR,
        valid_rows_input="valid_rows",
    )
    return result


__all__ = ["BENCH_SEQ", "DEFAULT_GROUPS", "DEFAULT_SEQ", "GROUPS", "LOGIT_ROWS",
           "TEXT_MAX_FLOOR", "TEXT_MAX_VS_GOLDEN", "build_kernel", "build_tensor_specs",
           "compare_fns", "golden_prefill_text_fwd", "logit_id_table", "text_ring_heap"]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--groups", type=int, default=None,
                        help="LLLA groups instead of DEFAULT_GROUPS; 16 is the whole model")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ,
                        help="prefill tokens; BENCH_SEQ is the configured chunk")
    parser.add_argument("--rows", type=int, default=LOGIT_ROWS,
                        help="scored token positions in single-request mode")
    parser.add_argument("--packed", action="store_true", help="pack requests and score each final token")
    parser.add_argument("--row-starts", default="0,256", help="packed request starts, in tokens")
    parser.add_argument("--page-bases", default="0,0", help="cached 128-token pages per request")
    parser.add_argument("--real", action="store_true", default=False,
                        help="the model's own layers, drawn embedding and head")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="seed nonzero GDN state and continue after prior KV pages")
    parser.add_argument("--page-base", type=int, default=0,
                        help="cached 128-token pages before this chunk")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="a dump set that does not sit where --fetch writes")
    parser.add_argument("--ring-heap", type=prefill_fwd.heap_arg, default=0,
                        help="bytes on rings 1 to 3, or four comma-separated per-ring values")
    parser.add_argument("--rounds", type=int, default=0, help="timed benchmark rounds")
    parser.add_argument("--no-golden", action="store_true", default=False,
                        help="time the kernel without building the oracle")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    starts = tuple(int(value) for value in args.row_starts.split(",")) if args.packed else (0,)
    bases = tuple(int(value) for value in args.page_bases.split(",")) if args.packed else (args.page_base,)
    if len(starts) != len(bases):
        parser.error("--row-starts and --page-bases must name the same requests")
    logit_rows = len(starts) if args.packed else args.rows
    counts = [args.groups] if args.groups else ([1] if args.packed else list(DEFAULT_GROUPS))
    t = args.seq_len
    if args.page_base % (prefill_fwd.KV_TILE // KV_PAGE_SIZE):
        parser.error("--page-base must align to the attention KV tile")
    if len(counts) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --groups")
    if args.rounds:
        import os

        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)

    for groups in counts:
        ends = starts[1:] + (t,)
        max_pages = max(base + (end - start) // KV_PAGE_SIZE for base, start, end in zip(bases, starts, ends)) + 2
        kv_layer_stride = (len(starts) * max_pages + 3) * KV_PAGE_SIZE
        label = "real layers" if args.real else "random weights"
        print(f"[case] prefill_text_fwd groups={groups} ({groups * INTERVAL} layers) "
              f"T={t} starts={starts} page_bases={bases} logits={logit_rows} {label}", flush=True)
        heap = args.ring_heap or text_ring_heap(t)
        result = run(
            fn=build_kernel(t=t, groups=groups, logit_rows=logit_rows,
                            max_pages=max_pages, kv_layer_stride=kv_layer_stride, rows=len(starts)),
            specs=build_tensor_specs(t=t, groups=groups, logit_rows=logit_rows, real=args.real,
                                     weights_dir=args.weights_dir, page_base=args.page_base,
                                     max_pages=max_pages, kv_layer_stride=kv_layer_stride,
                                     resume=args.resume, starts=starts, bases=bases),
            golden_fn=None if args.no_golden else golden_prefill_text_fwd(t, groups),
            compare_fn=None if args.no_golden else compare_fns(groups, t),
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=prefill_fwd.run_config(backend_type=args.backend_type, sim=args.execution_mode,
                                          device=args.device, ring_heap=heap),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at groups={groups} T={t}")
