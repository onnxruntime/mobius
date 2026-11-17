# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build shared local functions from explicit standard-ONNX component bodies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import TypeVar, overload

import onnx_ir as ir
from onnxscript import OpBuilder
from onnxscript._internal.builder import build_function

from mobius._build_context import ep_capabilities, get_build_opset_version
from mobius._flags import flags

FUNCTIONS_KEY = "mobius.builder.functions"
STANDARD_BODY_KEY = "mobius.standard_function"
STANDARD_BODY_VERSION = "1"
_TupleResult = TypeVar("_TupleResult", bound=tuple[ir.Value, ...])


@overload
def emit_standard_function(
    op: OpBuilder,
    name: str,
    body: Callable[..., ir.Value],
    inputs: Sequence[ir.Value],
    *,
    specialization: tuple[int | float | str | bool, ...] = (),
) -> ir.Value: ...


@overload
def emit_standard_function(
    op: OpBuilder,
    name: str,
    body: Callable[..., _TupleResult],
    inputs: Sequence[ir.Value],
    *,
    specialization: tuple[int | float | str | bool, ...] = (),
) -> _TupleResult: ...


def emit_standard_function(
    op: OpBuilder,
    name: str,
    body: Callable[..., ir.Value | tuple[ir.Value, ...]],
    inputs: Sequence[ir.Value],
    *,
    specialization: tuple[int | float | str | bool, ...] = (),
) -> ir.Value | tuple[ir.Value, ...]:
    """Emit a source body inline, or share it within an opted-in opset-29 model.

    All tensors, including parameters, must be explicit inputs. Specialization
    records every Python constant captured by the body. Formal shapes retain
    ranks only, so neither layer-specific symbols nor parameter data are baked
    into a shared function.
    """
    if not (
        flags.compact_standard_functions
        and get_build_opset_version() == 29
        and ep_capabilities().name == "onnx-standard"
    ):
        return body(op, *inputs)

    signature = [
        (
            int(value.dtype) if value.dtype is not None else None,
            len(value.shape) if value.shape is not None else None,
        )
        for value in inputs
    ]
    identity = json.dumps([specialization, signature], separators=(",", ":"), allow_nan=False)
    overload = "mobius_" + hashlib.sha256(identity.encode()).hexdigest()
    identifier = ("com.microsoft", name, overload)
    version = op.builder.graph.opset_imports.get(identifier[0])
    if version not in (None, 1):
        raise ValueError(
            f"Cannot emit compact function {identifier!r}: domain {identifier[0]!r} "
            f"already imports version {version}, not 1. Use a version-1 function domain."
        )
    functions = op.builder.functions
    function = functions.get(identifier)
    if function is None:
        formals = [
            ir.Value(
                name=f"input_{index}",
                type=value.type,
                shape=ir.Shape([None] * len(value.shape)) if value.shape is not None else None,
            )
            for index, value in enumerate(inputs)
        ]
        function = build_function(
            body,
            formals,
            domain=identifier[0],
            name=name,
            opset_imports={"": 29},
        )
        function.overload = overload
        function.metadata_props[STANDARD_BODY_KEY] = STANDARD_BODY_VERSION
        functions.update(function.graph.meta.get(FUNCTIONS_KEY, {}))
        functions[identifier] = function
    elif function.metadata_props.get(STANDARD_BODY_KEY) != STANDARD_BODY_VERSION:
        raise ValueError(
            f"Cannot register compact function {identifier!r}: the builder already "
            "contains a caller-owned definition. Use a distinct function identifier."
        )
    op.builder.graph.opset_imports[identifier[0]] = 1
    op.builder.root.graph.meta[FUNCTIONS_KEY] = functions
    outputs = [
        ir.Value(
            type=value.type,
            shape=value.shape.copy() if value.shape is not None else None,
        )
        for value in function.outputs
    ]
    result = op.builder.call(function, *inputs, _outputs=outputs)
    return result if isinstance(result, ir.Value) else tuple(result)
