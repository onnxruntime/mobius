# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVIDIA TensorRT Model Optimizer (ModelOpt) checkpoint support.

ModelOpt exports mixed-precision NVFP4 + FP8 checkpoints (e.g. Qwen3.6). This
package provides the weight-reconstruction math used to consume those
checkpoints:

- NVFP4 (``W4A16_NVFP4``): block-16 E2M1 4-bit weights with FP8-E4M3 block
  scales and a per-tensor FP32 global scale (``weight_scale_2``).
- FP8 (``E4M3``): per-tensor scaled float8 weights.

Nemotron-H's explicit ``keep_quantized=False, dtype="bf16"`` builder route
validates mixed-format inventories and reconstructs dense BF16 weights lazily.
It validates and accounts for activation and KV-cache scales, but does not
simulate their quantization. Native NVFP4/FP8 emission is unsupported; other
architectures remain blocked. NextN/MTP requires a separate validated contract.
"""

from __future__ import annotations

from mobius.integrations.modelopt._dequant import (
    FP4_E2M1_LUT,
    dequantize_fp8,
    dequantize_nvfp4,
    is_modelopt_quant_config,
    unpack_nvfp4_codes,
)

__all__ = [
    "FP4_E2M1_LUT",
    "dequantize_fp8",
    "dequantize_nvfp4",
    "is_modelopt_quant_config",
    "unpack_nvfp4_codes",
]
