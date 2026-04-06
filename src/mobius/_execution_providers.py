# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execution provider capability registry.

Defines the :class:`EpCapabilities` descriptor and a global
:class:`EpRegistry` that maps EP names to their capabilities.

Third-party EPs can register via :func:`register_ep`::

    from mobius._execution_providers import EpCapabilities, register_ep

    register_ep(EpCapabilities(
        name="my_ep",
        gqa_dtypes=frozenset({ir.DataType.FLOAT16}),
    ))

Adding a new built-in EP = adding one :class:`EpCapabilities` entry to
:func:`_register_builtins`. No other code changes required.
"""

from __future__ import annotations

__all__ = [
    "EpCapabilities",
    "EpRegistry",
    "ep_registry",
    "get_ep",
    "register_ep",
]

import dataclasses
import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class EpCapabilities:
    """All EP-specific capability flags in one place.

    Adding a new EP = adding a single :class:`EpCapabilities` entry to
    :func:`_register_builtins`. No other code needs to change.

    The ``supports_X`` flags mean "don't decompose X". When ``True``, the
    op (or its function body) is left in the graph unchanged. When ``False``,
    :class:`~onnx_ir.passes.common.InlinePass` expands the op's registered
    standard-ONNX function body. Setting ``False`` is only appropriate when
    the runtime *cannot* handle the custom op even via function-body expansion.

    Attributes:
        name: Canonical EP name (e.g. ``"cuda"``).
        gqa_dtypes: dtypes for which GroupQueryAttention fusion is supported.
        packed_attn_dtypes: dtypes for which PackedAttention fusion is
            supported.
        supports_fused_rope: ``False`` triggers SeparateRoPE + UnpackQKV
            lowering (DML).
        supports_shape: ``False`` triggers EliminateShape lowering (WebGPU).
        supports_skip_layer_norm: ``False`` expands SkipLayerNormalization /
            SkipSimplifiedLayerNormalization via InlinePass (TRT-RTX).
        supports_fused_matmul: ``False`` expands ``FusedMatMul`` via InlinePass
            to ``Transpose + MatMul``.  Leave ``True`` for all ORT-based EPs
            that support the ``com.microsoft::FusedMatMul`` kernel.
        supports_fused_moe: ``False`` decomposes fused MoE ops.
        supports_packed_multi_head_attention: ``False`` expands
            ``PackedMultiHeadAttention`` via InlinePass to a
            block-diagonal attention bias + standard ``Attention``.
            Leave ``True`` for CUDA / DML EPs that ship the native kernel.
        default_int4_accuracy_level: Default accuracy level for INT4
            quantization (0 = highest accuracy, 4 = fastest).
        provider_options: Default ORT GenAI provider options dict for this EP.
        enable_graph_capture: Whether this EP defaults to GPU graph capture.
    """

    name: str
    gqa_dtypes: frozenset[ir.DataType] = dataclasses.field(default_factory=frozenset)
    packed_attn_dtypes: frozenset[ir.DataType] = dataclasses.field(default_factory=frozenset)
    supports_fused_rope: bool = True
    supports_shape: bool = True
    supports_skip_layer_norm: bool = True
    supports_fused_matmul: bool = True
    supports_fused_moe: bool = True
    supports_packed_multi_head_attention: bool = False
    default_int4_accuracy_level: int = 0
    provider_options: dict[str, str] = dataclasses.field(default_factory=dict)
    enable_graph_capture: bool = False


class EpRegistry:
    """Central registry mapping EP name → :class:`EpCapabilities`.

    Supports dict-compatible access (``get``, ``__contains__``,
    ``__iter__``, ``__len__``), membership testing, and registration.
    Out-of-tree EPs register via :meth:`register`.
    """

    def __init__(self) -> None:
        self._entries: dict[str, EpCapabilities] = {}

    def register(self, caps: EpCapabilities, *, overwrite: bool = False) -> None:
        """Register an EP's capabilities.

        Args:
            caps: The EP capability descriptor. ``caps.name`` is used as key.
            overwrite: If ``True``, silently replace an existing entry.
                Raises ``ValueError`` if ``False`` and the name is already
                registered.
        """
        if caps.name in self._entries and not overwrite:
            raise ValueError(
                f"EP {caps.name!r} is already registered. Use overwrite=True to replace."
            )
        self._entries[caps.name] = caps
        logger.debug("Registered EP: %s", caps.name)

    def get(self, name: str, default: EpCapabilities | None = None) -> EpCapabilities | None:
        """Look up an EP by name, returning *default* if not found.

        This matches the :meth:`dict.get` contract so that existing callers
        that do ``caps = registry.get(ep); if caps is None: raise …`` continue
        to work without modification.
        """
        return self._entries.get(name, default)

    def require(self, name: str) -> EpCapabilities:
        """Look up an EP by name. Raises :class:`ValueError` if not found."""
        if name not in self._entries:
            raise ValueError(
                f"Unknown execution provider {name!r}. Registered: {sorted(self._entries)}"
            )
        return self._entries[name]

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> frozenset[str]:
        """Return the set of all registered EP names."""
        return frozenset(self._entries)


#: Global EP registry singleton.  Import and call :func:`register_ep` to add
#: custom EPs; use :func:`get_ep` to look one up.
ep_registry = EpRegistry()


def register_ep(caps: EpCapabilities, *, overwrite: bool = False) -> None:
    """Register *caps* in the global :data:`ep_registry`."""
    ep_registry.register(caps, overwrite=overwrite)


def get_ep(name: str) -> EpCapabilities:
    """Return the :class:`EpCapabilities` for *name* from the global registry.

    Raises:
        ValueError: If *name* is not registered.
    """
    return ep_registry.require(name)


def _register_builtins() -> None:
    """Populate the global registry with the six built-in EPs.

    Called once at module import. Adding a new EP = adding one entry here.
    """
    _builtins = [
        # Generic ONNX-conformant runtime — no vendor-specific kernel fusions.
        # All custom ops with ONNX function bodies are portable (the body is
        # the executable fallback). Only cleanup + constant folding are applied.
        # supports_X = True means "don't decompose X" — function bodies make
        # them portable, so decomposition would be counterproductive.
        EpCapabilities(
            name="default",
            gqa_dtypes=frozenset(),  # no GQA fusion — keep standard Attention ops
            packed_attn_dtypes=frozenset(),  # no packed attention fusion
        ),
        EpCapabilities(
            name="cpu",
            gqa_dtypes=frozenset({ir.DataType.FLOAT}),
            packed_attn_dtypes=frozenset({ir.DataType.FLOAT}),
            default_int4_accuracy_level=4,
        ),
        EpCapabilities(
            name="cuda",
            gqa_dtypes=frozenset({ir.DataType.FLOAT16, ir.DataType.BFLOAT16}),
            packed_attn_dtypes=frozenset(
                {ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16}
            ),
            supports_packed_multi_head_attention=True,
            provider_options={
                "enable_cuda_graph": "0",
                "enable_skip_layer_norm_strict_mode": "1",
            },
        ),
        EpCapabilities(
            name="dml",
            gqa_dtypes=frozenset({ir.DataType.FLOAT16}),
            packed_attn_dtypes=frozenset({ir.DataType.FLOAT, ir.DataType.FLOAT16}),
            supports_packed_multi_head_attention=True,
            supports_fused_rope=False,
        ),
        EpCapabilities(
            name="webgpu",
            gqa_dtypes=frozenset({ir.DataType.FLOAT, ir.DataType.FLOAT16}),
            packed_attn_dtypes=frozenset({ir.DataType.FLOAT, ir.DataType.FLOAT16}),
            supports_shape=False,
            default_int4_accuracy_level=4,
            provider_options={"enableGraphCapture": "0", "validationMode": "basic"},
        ),
        EpCapabilities(
            name="trt-rtx",
            gqa_dtypes=frozenset({ir.DataType.FLOAT16, ir.DataType.BFLOAT16}),
            packed_attn_dtypes=frozenset(
                {ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16}
            ),
            supports_skip_layer_norm=False,
            enable_graph_capture=True,
            provider_options={"enable_cuda_graph": "1"},
        ),
    ]
    for caps in _builtins:
        ep_registry.register(caps)


_register_builtins()
