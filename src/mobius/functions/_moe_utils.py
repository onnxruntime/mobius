# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared annotation validation for native MoE function specializations."""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir


def float32_parameters(alpha: float, beta: float, limit: float) -> tuple[float, float, float]:
    """Match the ONNX FLOAT attributes used by calls and body constants."""
    with np.errstate(over="ignore"):
        return float(np.float32(alpha)), float(np.float32(beta)), float(np.float32(limit))


def validate_config_parameters(
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    k: int,
    alpha: float,
    beta: float,
    limit: float,
    dtype: ir.DataType,
    caller: str,
) -> tuple[float, float, float]:
    """Validate shared invariants and return binary32-rounded alpha/beta/limit."""
    if any(type(d) is not int or d <= 0 for d in (experts, hidden, intermediate)):
        raise ValueError(
            f"{caller}: expert, hidden and intermediate dimensions must be positive integers"
        )
    if type(k) is not int or not 1 <= k <= experts:
        raise ValueError(f"{caller}: k must be between 1 and num_experts ({experts})")
    if not math.isfinite(alpha) or not math.isfinite(beta) or math.isnan(limit) or limit < 0:
        raise ValueError(
            f"{caller}: activation alpha/beta must be finite and swiglu_limit nonnegative"
        )
    alpha, beta, limit = float32_parameters(alpha, beta, limit)
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError(f"{caller}: activation alpha/beta must be finite in ONNX FLOAT")
    if dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16):
        raise ValueError(f"{caller}: only FLOAT and FLOAT16 are supported")
    return alpha, beta, limit


def validate_block(block: int, hidden: int, intermediate: int) -> None:
    """Validate the qualified QMoE packed-block contract, not all native layouts."""
    if type(block) is not int or block < 16 or block & (block - 1):
        raise ValueError("QMoE fallback: block_size must be a power of two >= 16")
    if hidden % block or intermediate % block:
        raise ValueError(
            "QMoE fallback: hidden and intermediate dimensions must be divisible by block_size"
        )


def validate_output_metadata(node: ir.Node, x: ir.Value, *, caller: str) -> None:
    """Reject known contradictions, accepting absent metadata and refinements."""
    output = node.outputs[0]
    message = f"{caller}: output metadata must match input shape and dtype"
    if output.type is not None and (
        not isinstance(output.type, ir.TensorType) or output.dtype != x.dtype
    ):
        raise ValueError(message)
    if output.shape is not None:
        assert x.shape is not None
        xs, ys = tuple(x.shape), tuple(output.shape)
        if len(ys) != len(xs) or any(
            isinstance(b, int) and (b < 0 or (isinstance(a, int) and a != b))
            for a, b in zip(xs, ys, strict=False)
        ):
            raise ValueError(message)
