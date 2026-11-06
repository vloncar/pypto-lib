# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lightweight Qwen3.8 kernel shape and scheduling capabilities for serving."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen38ServingContract:
    """Limits for exact-token prefill, padded decode and hybrid cache ownership."""

    schema_version: str
    prefill_tile_tokens: int
    prefill_request_alignment: int
    max_prefill_tokens_per_request: int
    decode_batch_tile: int
    max_decode_batch_size: int
    kv_page_tokens: int
    requires_homogeneous_prefill_decode: bool
    preserve_prefill_chunks: bool

    def validate_prefill_tokens(self, num_tokens: int) -> None:
        """Require a full fused extent; recurrent state must not consume padding."""
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
            raise TypeError("prefill tokens must be an int")
        if not self.prefill_tile_tokens <= num_tokens <= self.max_prefill_tokens_per_request:
            raise ValueError(
                f"Qwen3.8 prefill requires {self.prefill_tile_tokens} to "
                f"{self.max_prefill_tokens_per_request} tokens"
            )
        if num_tokens % self.prefill_tile_tokens:
            raise ValueError(f"Qwen3.8 prefill requires a multiple of {self.prefill_tile_tokens} tokens")


QWEN3_8_27B_SERVING_CONTRACT = Qwen38ServingContract(
    schema_version="1",
    prefill_tile_tokens=2048,
    prefill_request_alignment=256,
    max_prefill_tokens_per_request=32768,
    decode_batch_tile=16,
    max_decode_batch_size=128,
    kv_page_tokens=128,
    requires_homogeneous_prefill_decode=True,
    preserve_prefill_chunks=True,
)
