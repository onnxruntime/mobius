# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded-row, explicitly dense reconstruction of ModelOpt checkpoint weights."""

from __future__ import annotations

import math

import numpy as np
import torch

from mobius.integrations.modelopt._dequant import dequantize_fp8, dequantize_nvfp4

_ROWS_PER_CHUNK = 256


def validate_modelopt_scale(scale: torch.Tensor, name: str, *, scalar: bool) -> None:
    """Validate scale values without broadcasting malformed rows or scalars."""
    if scalar and (scale.numel() != 1 or scale.ndim > 1):
        raise ValueError(f"ModelOpt scalar scale {name!r} must have shape [] or [1]")
    values = scale.to(torch.float32)
    invalid = (values <= 0).any() if scalar else (values < 0).any()
    if not torch.isfinite(values).all() or invalid:
        requirement = "positive" if scalar else "nonnegative"
        raise ValueError(f"ModelOpt scale {name!r} must be finite and {requirement}")


def reconstruct_modelopt_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None,
    *,
    name: str,
) -> torch.Tensor:
    """Decode E4M3 or packed E2M1 to BF16, never native FP4 or affine INT4.

    The destination is allocated once; numeric-helper temporaries are bounded
    to 256 rows. Scalar scaling happens in FP32 before one BF16 rounding.
    """
    validate_modelopt_scale(scale, name + ".weight_scale", scalar=global_scale is None)
    if weight.ndim != 2:
        raise ValueError(f"ModelOpt weight {name!r} must be rank 2")
    rows, width = weight.shape
    if global_scale is not None:
        if global_scale.dtype != torch.float32:
            raise TypeError("NVFP4 global scale must use FP32")
        validate_modelopt_scale(global_scale, name + ".weight_scale_2", scalar=True)
        if weight.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
            raise TypeError("NVFP4 requires uint8 weights and E4M3 block scales")
        width *= 2
        if width % 16 or tuple(scale.shape) != (rows, width // 16):
            raise ValueError(f"NVFP4 block scale shape mismatch for {name!r}")
        factor = float(global_scale.item())
    else:
        if scale.dtype != torch.float32:
            raise TypeError("ModelOpt FP8 scalar scale must use FP32")
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError("ModelOpt FP8 weights must use E4M3")
        factor = float(scale.item())
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"Invalid ModelOpt scalar scale for {name!r}")
    result = torch.empty((rows, width), dtype=torch.bfloat16)
    for start in range(0, rows, _ROWS_PER_CHUNK):
        stop = min(start + _ROWS_PER_CHUNK, rows)
        codes = weight[start:stop].contiguous().view(torch.uint8).numpy()
        if global_scale is not None:
            blocks = scale[start:stop].contiguous().view(torch.uint8).numpy()
            dense = dequantize_nvfp4(codes, blocks, factor)
        else:
            dense = dequantize_fp8(codes, factor)
        if not np.isfinite(dense.astype(np.float32)).all():
            raise ValueError(f"ModelOpt reconstruction overflow/nonfinite values in {name!r}")
        result[start:stop] = torch.from_numpy(dense.view(np.uint16)).view(torch.bfloat16)
    return result
