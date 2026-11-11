# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared annotation validation for native MoE function specializations."""

from __future__ import annotations

import onnx_ir as ir


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
