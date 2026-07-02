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
        supports_skip_layer_norm: ``False`` expands SkipLayerNormalization /
            SkipSimplifiedLayerNormalization via InlinePass (TRT-RTX).
        supports_fused_moe: ``False`` decomposes fused MoE ops.
        supports_packed_multi_head_attention: ``False`` expands
            ``PackedMultiHeadAttention`` via InlinePass to a
            block-diagonal attention bias + standard ``Attention``.
            Leave ``True`` for CUDA / DML EPs that ship the native kernel.
        supports_rank4_rmsnorm: ``False`` reshapes rank-4 ``RMSNormalization``
            (query/key norm over the head dimension) to rank-3 and back via
            HtpRank4RMSNorm.  ``True`` leaves it unchanged.  Set ``False``
            only for the QNN HTP, which miscomputes rank-4 RMSNormalization.
        default_int4_accuracy_level: Default accuracy level for INT4
            quantization (0 = highest accuracy, 4 = fastest).
        provider_options: Default ORT GenAI provider options dict for this EP.
            Should not include graph-capture keys (``enable_cuda_graph`` /
            ``enableGraphCapture``); those are derived from
            ``enable_graph_capture`` so the flag is the single source of truth.
        enable_graph_capture: Whether this EP defaults to GPU graph capture.
            When ``True`` the generated genai_config provider options enable the
            EP-specific graph-capture option (``enable_cuda_graph`` for CUDA /
            TRT-RTX, ``enableGraphCapture`` + ``validationMode=disabled`` for
            WebGPU).
        supports_past_present_share_buffer: Whether past and present KV-cache
            tensors alias the same pre-allocated buffer.  When ``True``, the
            ORT GenAI runtime allocates a single KV-cache buffer at model load
            and maps both past and present as views into it, avoiding a
            per-step copy.  This is the recommended setting for every EP that
            supports ``GroupQueryAttention`` (CPU, CUDA, DML, WebGPU,
            TRT-RTX).  Set to ``False`` only for EPs that do not support GQA
            or cannot handle aliased KV-cache buffers.
        cap_kv_buffer_max_length: When ``True`` **and**
            ``supports_past_present_share_buffer`` is also ``True``, the
            generated ``max_length`` in genai_config is capped to avoid
            pre-allocating huge KV-cache buffers on memory-constrained
            devices.  ``True`` only for WebGPU (consumer GPU); ``False`` for
            CUDA / CPU / DML / TRT-RTX where the runtime can handle large
            pre-allocations.
    """

    name: str
    gqa_dtypes: frozenset[ir.DataType] = dataclasses.field(default_factory=frozenset)
    qkv_pack_dtypes: frozenset[ir.DataType] = dataclasses.field(default_factory=frozenset)
    supports_fused_rope: bool = True
    supports_skip_layer_norm: bool = True
    supports_fused_moe: bool = True
    supports_packed_multi_head_attention: bool = False
    supports_rank4_rmsnorm: bool = True
    default_int4_accuracy_level: int = 0
    provider_options: dict[str, str] = dataclasses.field(default_factory=dict)
    enable_graph_capture: bool = False
    supports_past_present_share_buffer: bool = False
    cap_kv_buffer_max_length: bool = False

    def __post_init__(self) -> None:
        if not self.supports_fused_rope and self.qkv_pack_dtypes:
            raise ValueError(
                f"EP '{self.name}': qkv_pack_dtypes must be frozenset() when "
                f"supports_fused_rope=False — UnpackQKV lowering always fires for "
                f"this EP, so packing would be immediately undone."
            )
        if self.cap_kv_buffer_max_length and not self.supports_past_present_share_buffer:
            raise ValueError(
                f"EP '{self.name}': cap_kv_buffer_max_length=True requires "
                f"supports_past_present_share_buffer=True — the cap only matters "
                f"when the runtime pre-allocates the full KV-cache buffer at load "
                f"time, which is what buffer sharing enables."
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
        # OpenVINO EP (via ORT GenAI). The OpenVINO EP consumes a portable ONNX
        # graph and compiles it internally for the selected device, so the graph
        # build mirrors "default" (standard Attention, no GQA/QKV packing). The
        # graph does not depend on the OpenVINO device, so we emit a sensible
        # default device_type ("NPU") in the genai_config provider options; a
        # different device can be selected downstream by editing genai_config
        # (e.g. by the Olive MobiusBuilder pass or the user) without rebuilding.
        EpCapabilities(
            name="openvino",
            gqa_dtypes=frozenset(),  # no GQA fusion — keep standard Attention ops
            qkv_pack_dtypes=frozenset(),  # no QKV packing
            provider_options={"device_type": "NPU"},
        ),
        EpCapabilities(
            name="cpu",
            gqa_dtypes=frozenset({ir.DataType.FLOAT}),
            qkv_pack_dtypes=frozenset({ir.DataType.FLOAT}),
            default_int4_accuracy_level=4,
            supports_past_present_share_buffer=True,
        ),
        EpCapabilities(
            name="cuda",
            gqa_dtypes=frozenset({ir.DataType.FLOAT16, ir.DataType.BFLOAT16}),
            qkv_pack_dtypes=frozenset(
                {ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16}
            ),
            supports_packed_multi_head_attention=True,
            provider_options={
                "enable_skip_layer_norm_strict_mode": "1",
            },
            supports_past_present_share_buffer=True,
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
            supports_past_present_share_buffer=True,
        ),
        EpCapabilities(
            name="webgpu",
            gqa_dtypes=frozenset({ir.DataType.FLOAT, ir.DataType.FLOAT16}),
            qkv_pack_dtypes=frozenset({ir.DataType.FLOAT, ir.DataType.FLOAT16}),
            default_int4_accuracy_level=4,
            enable_graph_capture=True,
            supports_past_present_share_buffer=True,
            cap_kv_buffer_max_length=True,
        ),
        EpCapabilities(
            name="trt-rtx",
            gqa_dtypes=frozenset({ir.DataType.FLOAT16, ir.DataType.BFLOAT16}),
            qkv_pack_dtypes=frozenset(
                {ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16}
            ),
            supports_skip_layer_norm=False,
            enable_graph_capture=True,
            supports_past_present_share_buffer=True,
        ),
        # Qualcomm Hexagon NPU via the QNN EP (onnxruntime-qnn QAIRT plugin),
        # HTP backend. The HTP runs a static-shaped, QDQ-quantized QNN context
        # binary with no kernels for ORT contrib fused ops, so everything is
        # decomposed to standard ONNX. The gemma4 multimodal decoder forgoes GQA
        # (bidirectional-vision overlay), so standard Attention is emitted and
        # static-shaped downstream. provider_options are the HTP launch defaults;
        # soc_model and the EP-context binary path are set per-device at build time.
        EpCapabilities(
            name="qnn",
            gqa_dtypes=frozenset(),  # no GroupQueryAttention (no QNN GQA builder)
            qkv_pack_dtypes=frozenset(),  # no PackQKV
            supports_fused_rope=False,  # SeparateRoPE + UnpackQKV
            supports_skip_layer_norm=False,  # inline Skip[Simplified]LayerNorm
            supports_packed_multi_head_attention=False,  # inline PackedMHA
            provider_options={
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
                "htp_graph_finalization_optimization_mode": "3",
                "enable_htp_shared_memory_allocator": "1",
            },
            supports_past_present_share_buffer=False,  # standard-Attention KV concat
            supports_rank4_rmsnorm=False,  # HTP miscomputes rank-4 RMSNorm (q/k norm)
        ),
        # onnx-standard: ONNX-only runtime — emits zero custom-domain ops.
        # All com.microsoft ops (SkipLayerNorm, PackedMHA) are expanded via
        # InlinePass to their standard-ONNX function bodies. No GQA or QKV
        # packing fusion is applied. Use this EP to produce models that run
        # on any conformant ONNX runtime without ORT extensions.
        # KV buffer sharing is unsupported here: GQA isn't emitted, so
        # standard Attention's concat-grow semantics handle the cache.
        EpCapabilities(
            name="onnx-standard",
            gqa_dtypes=frozenset(),  # no GroupQueryAttention
            qkv_pack_dtypes=frozenset(),  # no PackQKV
            supports_fused_rope=False,  # no fused RoPE inside GQA (GQA not supported)
            supports_skip_layer_norm=False,  # inline SkipLayerNorm
            supports_packed_multi_head_attention=False,  # inline PackedMHA
        ),
    ]
    for caps in _builtins:
        ep_registry.register(caps)


_register_builtins()
