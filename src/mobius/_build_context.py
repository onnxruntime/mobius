# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build-time graph and structural context for components.

Provides thread-safe, async-safe contexts for graph-construction capabilities
and target-specific structural requirements. Public Mobius export activates
neutral graph capabilities and derives a :class:`BuildContract` from the target
EP/device; graph rewrites run downstream.

Usage::

    from mobius._build_context import build_context, ep_capabilities, get_build_dtype

    contract = get_build_contract()
    if contract.layered_per_layer_inputs:
        ...  # preserve a target-required component interface

    # Build orchestration keeps graph and structural policies independent:
    with build_context(canonical_capabilities, ir.DataType.FLOAT16, contract=contract):
        pkg = task.build(module, config)
"""

from __future__ import annotations

import contextvars
import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager

import onnx_ir as ir

from mobius._execution_providers import EpCapabilities

__all__ = [
    "BuildContract",
    "build_context",
    "ep_capabilities",
    "get_build_contract",
    "get_build_dtype",
    "is_prefill_prefix_pruning_enabled",
    "prefill_prefix_pruning",
]


@dataclasses.dataclass(frozen=True)
class BuildContract:
    """Target-specific structural requirements that do not select graph rewrites."""

    target_execution_provider: str = "default"
    target_device: str | None = None
    max_buffer_size: int | None = None
    layered_per_layer_inputs: bool = False
    supports_range: bool = True

    @classmethod
    def from_capabilities(
        cls,
        capabilities: EpCapabilities,
        *,
        target_device: str | None = None,
    ) -> BuildContract:
        """Create a structural build contract from an EP capability descriptor."""
        return cls(
            target_execution_provider=capabilities.name,
            target_device=target_device,
            max_buffer_size=capabilities.max_buffer_size,
            layered_per_layer_inputs=capabilities.layered_per_layer_inputs,
            supports_range=capabilities.supports_range,
        )


_DEFAULT_CAPABILITIES = EpCapabilities(name="default")
_DEFAULT_BUILD_CONTRACT = BuildContract()

_current_ep: contextvars.ContextVar[EpCapabilities] = contextvars.ContextVar(
    "mobius_ep_capabilities", default=_DEFAULT_CAPABILITIES
)
_current_build_contract: contextvars.ContextVar[BuildContract] = contextvars.ContextVar(
    "mobius_build_contract", default=_DEFAULT_BUILD_CONTRACT
)
_current_dtype: contextvars.ContextVar[ir.DataType] = contextvars.ContextVar(
    "mobius_build_dtype", default=ir.DataType.FLOAT
)
_prune_prefill_prefix: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mobius_prune_prefill_prefix", default=False
)


@contextmanager
def build_context(
    capabilities: EpCapabilities,
    dtype: ir.DataType = ir.DataType.FLOAT,
    *,
    contract: BuildContract | None = None,
) -> Iterator[None]:
    """Activate EP capabilities for the duration of graph construction.

    The context is thread-safe and async-safe: each thread or coroutine
    maintains its own independent context stack.

    Args:
        capabilities: Graph-construction capability descriptor to activate.
        dtype: Active build dtype. Defaults to ``ir.DataType.FLOAT``.
        contract: Structural target requirements. When omitted, derives them
            from ``capabilities`` to preserve existing EP-aware builds.

    Example::

        from mobius._build_context import build_context
        from mobius._execution_providers import ep_registry

        capabilities = ep_registry.require("cuda")
        with build_context(capabilities, ir.DataType.FLOAT16):
            pkg = task.build(module, config)
    """
    contract = contract or BuildContract.from_capabilities(capabilities)
    capabilities_token = _current_ep.set(capabilities)
    contract_token = _current_build_contract.set(contract)
    dtype_token = _current_dtype.set(dtype)
    try:
        yield
    finally:
        _current_ep.reset(capabilities_token)
        _current_build_contract.reset(contract_token)
        _current_dtype.reset(dtype_token)


def ep_capabilities() -> EpCapabilities:
    """Return the active EP capabilities.

    Returns the default descriptor (no fusion, portable ONNX) when
    no :func:`build_context` is active.

    Example::

        from mobius._build_context import ep_capabilities
        import onnx_ir as ir

        capabilities = ep_capabilities()
    """
    return _current_ep.get()


def get_build_contract() -> BuildContract:
    """Return the active target-specific structural build contract."""
    return _current_build_contract.get()


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
