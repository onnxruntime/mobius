# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build-time EP context for components.

Provides a thread-safe, async-safe mechanism for components to query
EP capabilities during graph construction. Uses contextvars so
concurrent builds (threads, asyncio) are fully isolated.

Usage::

    from mobius._build_context import build_context, ep_capabilities, get_build_dtype

    # Components can read the active EP capabilities at any point:
    capabilities = ep_capabilities()
    if ir.DataType.FLOAT16 in capabilities.gqa_dtypes:
        ...  # emit GQA-specific ops

    # Build orchestration wraps graph construction in a context:
    with build_context(cuda_capabilities, ir.DataType.FLOAT16):
        pkg = task.build(module, config)
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

import onnx_ir as ir

from mobius._execution_providers import EpCapabilities

__all__ = [
    "build_context",
    "ep_capabilities",
    "get_build_dtype",
]

_DEFAULT_CAPABILITIES = EpCapabilities(name="default")

_current_ep: contextvars.ContextVar[EpCapabilities] = contextvars.ContextVar(
    "mobius_ep_capabilities", default=_DEFAULT_CAPABILITIES
)
_current_dtype: contextvars.ContextVar[ir.DataType] = contextvars.ContextVar(
    "mobius_build_dtype", default=ir.DataType.FLOAT
)


@contextmanager
def build_context(
    capabilities: EpCapabilities,
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> Iterator[None]:
    """Activate EP capabilities for the duration of graph construction.

    The context is thread-safe and async-safe: each thread or coroutine
    maintains its own independent context stack.

    Args:
        capabilities: EP capability descriptor to activate. Typically obtained
            via ``ep_registry.require(execution_provider)``.
        dtype: Active build dtype. Defaults to ``ir.DataType.FLOAT``.

    Example::

        from mobius._build_context import build_context
        from mobius._execution_providers import ep_registry

        capabilities = ep_registry.require("cuda")
        with build_context(capabilities, ir.DataType.FLOAT16):
            pkg = task.build(module, config)
    """
    capabilities_token = _current_ep.set(capabilities)
    dtype_token = _current_dtype.set(dtype)
    try:
        yield
    finally:
        _current_ep.reset(capabilities_token)
        _current_dtype.reset(dtype_token)


def ep_capabilities() -> EpCapabilities:
    """Return the active EP capabilities.

    Returns the default descriptor (no fusion, portable ONNX) when
    no :func:`build_context` is active.

    Example::

        from mobius._build_context import ep_capabilities
        import onnx_ir as ir

        capabilities = ep_capabilities()
        if ir.DataType.FLOAT16 in capabilities.gqa_dtypes:
            # emit GQA with fp16 inputs
            ...
    """
    return _current_ep.get()


def get_build_dtype() -> ir.DataType:
    """Return the active build dtype.

    Returns ``ir.DataType.FLOAT`` when no :func:`build_context` is active.

    Example::

        from mobius._build_context import get_build_dtype

        dtype = get_build_dtype()
        if dtype == ir.DataType.BFLOAT16:
            ...
    """
    return _current_dtype.get()
