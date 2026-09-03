# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit graph I/O helpers for causal convolution streaming state."""

from __future__ import annotations

import onnx_ir as ir


def make_conv_cache_inputs(
    builder,
    specs: tuple[tuple[int, int], ...],
    batch: ir.SymbolicDim,
    dtype: ir.DataType,
    *,
    prefix: str = "past_conv",
) -> list[ir.Value]:
    """Register one channels-first causal cache input for every convolution."""
    return [
        builder.input(
            f"{prefix}.{index}",
            dtype=dtype,
            shape=[batch, channels, left_pad],
        )
        for index, (channels, left_pad) in enumerate(specs)
    ]


def register_conv_cache_outputs(
    builder,
    values: list[ir.Value],
    *,
    prefix: str = "present_conv",
) -> None:
    """Register causal convolution state outputs under a stable component prefix."""
    for index, value in enumerate(values):
        builder.add_output(value, f"{prefix}.{index}")
