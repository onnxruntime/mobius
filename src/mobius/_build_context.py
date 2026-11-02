# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
    "get_matmul_nbits_accuracy_level",
    "is_prefill_prefix_pruning_enabled",
    "prefill_prefix_pruning",
    "validate_matmul_nbits_accuracy_level",
]

_DEFAULT_CAPABILITIES = EpCapabilities(name="default")

_current_ep: contextvars.ContextVar[EpCapabilities] = contextvars.ContextVar(
    "mobius_ep_capabilities", default=_DEFAULT_CAPABILITIES
)
_current_dtype: contextvars.ContextVar[ir.DataType] = contextvars.ContextVar(
    "mobius_build_dtype", default=ir.DataType.FLOAT
)
_matmul_nbits_accuracy_level: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mobius_matmul_nbits_accuracy_level", default=None
)
_prune_prefill_prefix: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mobius_prune_prefill_prefix", default=False
)


@contextmanager
def build_context(
    capabilities: EpCapabilities,
    dtype: ir.DataType = ir.DataType.FLOAT,
    *,
    matmul_nbits_accuracy_level: int | None = None,
) -> Iterator[None]:
    """Activate EP capabilities for the duration of graph construction.

    The context is thread-safe and async-safe: each thread or coroutine
    maintains its own independent context stack.

    Args:
        capabilities: EP capability descriptor to activate. Typically obtained
            via ``ep_registry.require(execution_provider)``.
        dtype: Active build dtype. Defaults to ``ir.DataType.FLOAT``.
        matmul_nbits_accuracy_level: Override every emitted ``MatMulNBits``
            accuracy level (0 through 4), regardless of weight bit width.
            ``None`` preserves the EP-specific defaults.

    Example::

        from mobius._build_context import build_context
        from mobius._execution_providers import ep_registry

        capabilities = ep_registry.require("cuda")
        with build_context(capabilities, ir.DataType.FLOAT16):
            pkg = task.build(module, config)
    """
    validate_matmul_nbits_accuracy_level(matmul_nbits_accuracy_level)
    capabilities_token = _current_ep.set(capabilities)
    dtype_token = _current_dtype.set(dtype)
    accuracy_token = _matmul_nbits_accuracy_level.set(matmul_nbits_accuracy_level)
    try:
        yield
    finally:
        _current_ep.reset(capabilities_token)
        _current_dtype.reset(dtype_token)
        _matmul_nbits_accuracy_level.reset(accuracy_token)


def validate_matmul_nbits_accuracy_level(level: int | None) -> None:
    """Reject values outside the ORT ``MatMulNBits`` accuracy-level domain."""
    if level is not None and (type(level) is not int or not 0 <= level <= 4):
        raise ValueError("matmul_nbits_accuracy_level must be an integer from 0 to 4 or None")


def get_matmul_nbits_accuracy_level() -> int | None:
    """Return the explicit override, or ``None`` to preserve EP defaults."""
    return _matmul_nbits_accuracy_level.get()


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


@contextmanager
def prefill_prefix_pruning(enabled: bool) -> Iterator[None]:
    """Enable or disable prefill token-prefix pruning during graph construction."""
    token = _prune_prefill_prefix.set(enabled)
    try:
        yield
    finally:
        _prune_prefill_prefix.reset(token)


def is_prefill_prefix_pruning_enabled() -> bool:
    """Return whether the active task discards prefill tokens before the final token."""
    return _prune_prefill_prefix.get()
