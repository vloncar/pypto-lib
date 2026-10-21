# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compatibility imports for Qwen3.8-27B kernels.

Model and ABI constants live in :mod:`constants`, matching the complete-model
layout used by ``models/qwen3_14b``. Existing Q0-Q4 kernels imported this
module before Q5 added that split, so it remains as a stable forwarding layer.
"""

try:
    from .constants import (
        FULL_ATTENTION,
        GDN_QUANT,
        GDN_TILING,
        LINEAR_ATTENTION,
        PREFILL_SEQ,
        QWEN3_8_27B,
        QWEN3_8_27B_DIMS,
        QWEN3_8_27B_LAYER_TYPES,
        GdnQuant,
        GdnTiling,
        Qwen38Config,
        Qwen38Dimensions,
        dimensions,
    )
except ImportError:
    from constants import (
        FULL_ATTENTION,
        GDN_QUANT,
        GDN_TILING,
        LINEAR_ATTENTION,
        PREFILL_SEQ,
        QWEN3_8_27B,
        QWEN3_8_27B_DIMS,
        QWEN3_8_27B_LAYER_TYPES,
        GdnQuant,
        GdnTiling,
        Qwen38Config,
        Qwen38Dimensions,
        dimensions,
    )


__all__ = [
    "FULL_ATTENTION",
    "GDN_QUANT",
    "GDN_TILING",
    "LINEAR_ATTENTION",
    "PREFILL_SEQ",
    "QWEN3_8_27B",
    "QWEN3_8_27B_DIMS",
    "QWEN3_8_27B_LAYER_TYPES",
    "GdnQuant",
    "GdnTiling",
    "Qwen38Config",
    "Qwen38Dimensions",
    "dimensions",
]
