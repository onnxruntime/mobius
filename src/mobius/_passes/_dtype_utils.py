# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared dtype helpers for graph passes that materialize new initializers.

Passes such as :class:`~mobius._passes.FoldConcatInitializersPass` and
:class:`~mobius._passes.FoldTransposedInitializerPass` pre-compute new
initializers from existing ones. They must stamp the *correct* dtype on the
result, otherwise an fp16 model can silently end up with fp32 weights that
onnxruntime rejects at load time (a MatMul binding fp16 and fp32 to the same
type parameter ``T``).
"""

from __future__ import annotations

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


def initializer_dtype(value: ir.Value) -> ir.DataType | None:
    """Return the effective dtype of an initializer ``value``.

    Uses the value's declared ``type`` dtype, but falls back to the dtype of its
    ``const_value`` when the type annotation is missing. When both are present
    but disagree, the ``const_value`` dtype wins (it is the data actually
    serialized) and a warning is logged, since a healthy initializer should
    never have a declared type that contradicts its data.

    Graph building can drop the declared ``type`` on an initializer while its
    actual tensor data (``const_value``) still carries the correct dtype. In
    that situation, defaulting to ``ir.DataType.FLOAT`` would emit fp32 weights
    into an otherwise fp16 model. Reading the dtype from ``const_value`` keeps
    folded initializers consistent with the weights they are derived from.

    Returns ``None`` only when neither the declared type nor ``const_value`` is
    available; callers decide on a final fallback.
    """
    declared = value.dtype
    const_dtype = value.const_value.dtype if value.const_value is not None else None

    if declared is not None and const_dtype is not None and declared != const_dtype:
        logger.warning(
            "Initializer %r declares dtype %s but its data is %s; using the data "
            "dtype. This indicates stale type metadata.",
            value.name,
            declared,
            const_dtype,
        )
        return const_dtype
    if declared is not None:
        return declared
    return const_dtype
