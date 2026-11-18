# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AMD Quark checkpoint support.

Quark exports integer-affine INT4 checkpoints whose layout differs from the
GPTQ/AWQ one in three ways that matter here: the packed codes run along the
*output* axis, the nibbles inside each int32 are AWQ-GEMM interleaved, and the
zero points carry no implicit ``+1`` offset. This package provides:

- :func:`is_quark_quant_config` / :func:`parse_quark_config` — reading Quark's
  nested ``global_quant_config.weight`` + ``exclude`` schema.
- :func:`preprocess_quark_weights` — converting packed projections into the
  ``MatMulNBits`` ``B``/``scales``/``zero_points`` layout.
- :func:`dequantize_quark_weight` / :func:`dequantize_quark_projections` —
  dense float reconstruction, for projections whose ONNX graph uses a plain
  float ``Linear`` even though the checkpoint quantized them.

Only ``uint4`` per-group static weights are supported; every other Quark layout
is rejected rather than approximated.
"""

from __future__ import annotations

from mobius.integrations.quark._config import (
    is_quark_quant_config,
    parse_quark_config,
)
from mobius.integrations.quark._weights import (
    dequantize_quark_projections,
    dequantize_quark_weight,
    preprocess_quark_weights,
)

__all__ = [
    "dequantize_quark_projections",
    "dequantize_quark_weight",
    "is_quark_quant_config",
    "parse_quark_config",
    "preprocess_quark_weights",
]
