# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
        qkv_pack_dtypes: dtypes for which QKV weight packing for
            GroupQueryAttention is supported (PackQKV fusion).  Set to an
            empty frozenset for EPs that do not support packed QKV inputs
            (e.g. DML, which always unpacks via UnpackQKV in the lowering
            stage).
        supports_fused_rope: ``False`` triggers SeparateRoPE + UnpackQKV
            lowering (DML).
        supports_shape: ``False`` triggers EliminateShape lowering (WebGPU).
        supports_skip_layer_norm: ``False`` expands SkipLayerNormalization /
            SkipSimplifiedLayerNormalization via InlinePass (TRT-RTX).
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
    qkv_pack_dtypes: frozenset[ir.DataType] = dataclasses.field(default_factory=frozenset)
    supports_fused_rope: bool = True
    supports_shape: bool = True
    supports_skip_layer_norm: bool = True
    supports_fused_moe: bool = True
    supports_packed_multi_head_attention: bool = False
    default_int4_accuracy_level: int = 0
    provider_options: dict[str, str] = dataclasses.field(default_factory=dict)
    enable_graph_capture: bool = False

    def __post_init__(self) -> None:
        if not self.supports_fused_rope and self.qkv_pack_dtypes:
            raise ValueError(
                f"EP '{self.name}': qkv_pack_dtypes must be frozenset() when "
                f"supports_fused_rope=False — UnpackQKV lowering always fires for "
                f"this EP, so packing would be immediately undone."
            )


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
    """Populate the global registry with the seven built-in EPs.

    Called once at module import. Adding a new EP = adding one entry here.
    """
    _builtins = [
        # Generic ONNX-conformant runtime — no EP-specific fused ops (no GQA,
        # no PackQKV). Standard fusions (SkipNorm, Gelu) are applied but remain
        # portable: all custom ops have ONNX function bodies that any conformant
        # runtime can expand as a fallback.
        # supports_X = True means "don't decompose X" — function bodies make
        # them portable, so decomposition would be counterproductive.
        EpCapabilities(
            name="default",
            gqa_dtypes=frozenset(),  # no GQA fusion — keep standard Attention ops
            qkv_pack_dtypes=frozenset(),  # no QKV packing
        ),
        EpCapabilities(
            name="cpu",
            gqa_dtypes=frozenset({ir.DataType.FLOAT}),
            qkv_pack_dtypes=frozenset({ir.DataType.FLOAT}),
            default_int4_accuracy_level=4,
        ),
        EpCapabilities(
            name="cuda",
            gqa_dtypes=frozenset({ir.DataType.FLOAT16, ir.DataType.BFLOAT16}),
            qkv_pack_dtypes=frozenset(
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
            # DML does not support packed QKV in GQA — UnpackQKV always fires
            # (triggered by supports_fused_rope=False), so packing would be
            # immediately undone.  Leave empty to skip the pointless round-trip.
            qkv_pack_dtypes=frozenset(),
            supports_packed_multi_head_attention=True,
            supports_fused_rope=False,
        ),
        EpCapabilities(
            name="webgpu",
            gqa_dtypes=frozenset({ir.DataType.FLOAT, ir.DataType.FLOAT16}),
            qkv_pack_dtypes=frozenset({ir.DataType.FLOAT, ir.DataType.FLOAT16}),
            supports_shape=False,
            default_int4_accuracy_level=4,
            provider_options={"enableGraphCapture": "0", "validationMode": "basic"},
        ),
        EpCapabilities(
            name="trt-rtx",
            gqa_dtypes=frozenset({ir.DataType.FLOAT16, ir.DataType.BFLOAT16}),
            qkv_pack_dtypes=frozenset(
                {ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16}
            ),
            supports_skip_layer_norm=False,
            enable_graph_capture=True,
            provider_options={"enable_cuda_graph": "1"},
        ),
        # onnx-standard: ONNX-only runtime — emits zero custom-domain ops.
        # All com.microsoft ops (SkipLayerNorm, PackedMHA) are expanded via
        # InlinePass to their standard-ONNX function bodies. No GQA or QKV
        # packing fusion is applied. Use this EP to produce models that run
        # on any conformant ONNX runtime without ORT extensions.
        EpCapabilities(
            name="onnx-standard",
            gqa_dtypes=frozenset(),  # no GroupQueryAttention
            qkv_pack_dtypes=frozenset(),  # no PackQKV
            supports_fused_rope=False,  # no fused RoPE inside GQA (GQA not supported)
            supports_shape=True,  # Shape is a standard ONNX op — no elimination needed
            supports_skip_layer_norm=False,  # inline SkipLayerNorm
            supports_packed_multi_head_attention=False,  # inline PackedMHA
        ),
    ]
    for caps in _builtins:
        ep_registry.register(caps)


_register_builtins()
