# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native DeepSeek-V4 CSA/HCA export gate and op emission (C1: ratio-128 HCA).

The frozen ``pkg.nxrt::CompressedSparseAttention`` v1 kernel already exists,
is merged, and is op-parity-tested in ``onnx-genai``
(``crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs``).
This module is the Mobius *export* side of the first integration slice (C1):
it emits that op -- for the ratio-128 "Heavily Compressed Attention" (HCA)
subset only -- in place of the dense correctness fallback, behind the
default-off ``config.native_csa`` opt-in.

Design record: ``onnx-genai/.squad/decisions/inbox/`` ``deckard-deepseek-v4-
csa-hca-cuda-slice.md`` (§5 property ABI, §10 slice C1). The authoritative
op contract is the Rust factory/shape-inference:

* required inputs (order): ``query, current_kv, compressor_kv,
  compressor_gate, compressor_ape, compressor_norm, past_compressed_kv,
  past_compression_carry, seqlens_k, total_sequence_length, head_sink``;
* ratio-128 outputs (exactly 3): ``Y, present_compressed_kv,
  present_compression_carry``;
* attributes ``num_heads, head_dim, qk_rope_head_dim, compression_ratio=128,
  index_num_heads=index_head_dim=index_topk=0, causal=1,
  cache_layout_version=1, index_layout_version=1, sink_mode='logit_only',
  cache_format, scale`` (``scale=0`` selects ``1/sqrt(head_dim)``).

This is a **property gate, not a model-name allowlist**: emission requires the
config-derived properties to match the frozen v1 ratio-128 contract exactly.
Anything the frozen op cannot express when the feature is requested -- ratio-4
CSA, MTP recurrence, quantized compressor projections, an unknown per-layer
ratio, or a carry/state shape that does not match -- raises
:class:`NativeCsaExportError` rather than silently emitting the dense fallback.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
from onnxscript import OpBuilder

from mobius._configs import ArchitectureConfig

# Frozen op identity -------------------------------------------------------
CSA_OP_TYPE = "CompressedSparseAttention"
CSA_DOMAIN = "pkg.nxrt"
CSA_OPSET_VERSION = 1
# Matches ``LAYOUT_VERSION`` in the Rust kernel; the op rejects any other
# ``cache_layout_version``/``index_layout_version``.
CSA_LAYOUT_VERSION = 1

# The two compression ratios the frozen v1 op accepts.
HCA_COMPRESSION_RATIO = 128  # compressor-only "Heavily Compressed Attention"
CSA_COMPRESSION_RATIO = 4  # compressor + learned FP4 indexer + top-k

# ``past_compression_carry`` packs two planes per slot: the FP32 pooling
# accumulator and the FP32 running score-state (``-inf`` initialised).
CSA_CARRY_PLANES = 2

# v1 requires ``sink_mode='logit_only'`` for the learned per-head ``head_sink``
# denominator term, and ratio-128 attention-compressor records are stored as
# f32 (or the hybrid FP8/BF16 device format, out of scope for C1).
CSA_SINK_MODE = "logit_only"
HCA_CACHE_FORMAT_F32 = "f32"

# ``scale=0`` is the frozen-op sentinel for ``1/sqrt(head_dim)`` (the Rust
# kernel: ``if configured_scale == 0.0 { 1/sqrt(head_dim) }``), matching the
# op-parity test which omits the attribute entirely.
CSA_DEFAULT_SCALE_SENTINEL = 0.0

# Authoritative required-input order for the frozen v1 ratio-128 schema.
HCA_INPUT_NAMES: tuple[str, ...] = (
    "query",
    "current_kv",
    "compressor_kv",
    "compressor_gate",
    "compressor_ape",
    "compressor_norm",
    "past_compressed_kv",
    "past_compression_carry",
    "seqlens_k",
    "total_sequence_length",
    "head_sink",
)
HCA_OUTPUT_NAMES: tuple[str, ...] = (
    "Y",
    "present_compressed_kv",
    "present_compression_carry",
)


class NativeCsaExportError(ValueError):
    """Native CSA/HCA export was requested but cannot be satisfied.

    Raised (fail-closed) instead of silently emitting the dense correctness
    fallback whenever ``config.native_csa`` is set but a layer does not match
    the frozen ``pkg.nxrt::CompressedSparseAttention`` v1 ratio-128 contract.
    """


@dataclasses.dataclass(frozen=True)
class HcaLayerPlan:
    """Resolved, contract-checked ratio-128 emission plan for one layer.

    Every field is a native-op *property*, derived from the architecture
    config and validated against the frozen v1 ratio-128 schema. The
    deterministic state IO names let the graph task thread
    ``past_* -> present_*`` compressed state without guessing.
    """

    layer_id: int
    num_heads: int
    head_dim: int
    qk_rope_head_dim: int
    cache_format: str
    stored_width: int
    scale: float = CSA_DEFAULT_SCALE_SENTINEL
    compression_ratio: int = HCA_COMPRESSION_RATIO
    carry_slots: int = HCA_COMPRESSION_RATIO
    carry_planes: int = CSA_CARRY_PLANES

    @property
    def past_compressed_kv_name(self) -> str:
        return f"past_compressed_kv.{self.layer_id}"

    @property
    def past_compression_carry_name(self) -> str:
        return f"past_compression_carry.{self.layer_id}"

    @property
    def present_compressed_kv_name(self) -> str:
        return f"present_compressed_kv.{self.layer_id}"

    @property
    def present_compression_carry_name(self) -> str:
        return f"present_compression_carry.{self.layer_id}"


def _hca_stored_width(cache_format: str, head_dim: int, qk_rope_head_dim: int) -> int:
    """Per-record byte/element width of ``past_compressed_kv`` for a format.

    Mirrors ``CacheFormat::stored_width`` in the Rust kernel. C1 only emits
    the ``f32`` format (records stored as full ``head_dim`` f32 rows); the
    device FP8 hybrid format is validated by the runtime slice, not here.
    """
    if cache_format == HCA_CACHE_FORMAT_F32:
        return head_dim
    raise NativeCsaExportError(
        f"native CSA C1 only emits cache_format='{HCA_CACHE_FORMAT_F32}' for "
        f"ratio-{HCA_COMPRESSION_RATIO}; '{cache_format}' is a device-side "
        "format handled by the runtime slice, not the exporter"
    )


def _layer_compress_ratio(config: ArchitectureConfig, layer_id: int) -> int:
    ratios = config.compress_ratios or []
    return ratios[layer_id] if layer_id < len(ratios) else 0


def plan_native_csa(
    config: ArchitectureConfig,
    layer_id: int,
    *,
    is_mtp: bool = False,
) -> HcaLayerPlan | None:
    """Resolve a layer's native-CSA emission plan, or ``None`` for dense.

    Returns ``None`` when the feature is off, or when the layer is genuinely
    a dense (ratio-0) attention layer -- those are *not* a suppressed CSA
    fallback, they carry no compressor at all.

    Raises :class:`NativeCsaExportError` when the feature is requested but the
    layer cannot be expressed by the frozen ratio-128 op: ratio-4 CSA, an MTP
    layer that would need compressed-state recurrence (undefined in the pinned
    source), quantized compressor projections, an unknown ratio, or head/rope
    dims that violate the op contract.
    """
    if not config.native_csa:
        return None

    ratio = _layer_compress_ratio(config, layer_id)
    if ratio == 0:
        # A genuine dense/sliding layer (schedule value 0). No compressor
        # exists to route through the op; dense emission here is correct, not
        # a silent CSA fallback.
        return None

    if is_mtp:
        # The pinned checkpoint's MTP block is ratio-0 dense and its
        # recurrence/acceptance/KV-lifetime is undefined in source (design
        # blocker B5). A compressed MTP layer must never be invented.
        raise NativeCsaExportError(
            f"native CSA requested for MTP layer {layer_id} with "
            f"compression_ratio={ratio}; MTP compressed-state recurrence is "
            "undefined in the pinned DeepSeek-V4 source and must not be "
            "synthesised (design blocker B5). MTP stays ratio-0 dense"
        )

    if ratio == CSA_COMPRESSION_RATIO:
        raise NativeCsaExportError(
            f"native CSA requested for ratio-{CSA_COMPRESSION_RATIO} (learned "
            f"FP4 indexer + top-k) layer {layer_id}; the exporter's C1 slice "
            f"only emits the ratio-{HCA_COMPRESSION_RATIO} (HCA) subset. "
            "ratio-4 selection/top-k tensors are a follow-up slice"
        )

    if ratio != HCA_COMPRESSION_RATIO:
        raise NativeCsaExportError(
            f"layer {layer_id} has compression_ratio={ratio}; the frozen "
            f"pkg.nxrt::CompressedSparseAttention v1 op accepts only "
            f"{CSA_COMPRESSION_RATIO} or {HCA_COMPRESSION_RATIO}"
        )

    quantization = config.quantization
    if quantization is not None and quantization.quant_method != "none":
        raise NativeCsaExportError(
            f"native CSA requested for layer {layer_id} under quantization "
            f"'{quantization.quant_method}'; the frozen op consumes f32 "
            "compressor activations and C1 only wires unquantized compressor "
            "projections. Quantized compressor dequant is a follow-up slice"
        )

    head_dim = config.head_dim
    if head_dim is None or head_dim <= 0:
        raise NativeCsaExportError(
            f"native CSA layer {layer_id} requires a positive head_dim, got {head_dim}"
        )
    num_heads = config.num_attention_heads
    if num_heads is None or num_heads <= 0:
        raise NativeCsaExportError(
            f"native CSA layer {layer_id} requires positive num_attention_"
            f"heads, got {num_heads}"
        )
    qk_rope_head_dim = config.qk_rope_head_dim or 0
    if qk_rope_head_dim < 0 or qk_rope_head_dim > head_dim:
        raise NativeCsaExportError(
            f"native CSA layer {layer_id} requires 0 <= qk_rope_head_dim <= "
            f"head_dim ({head_dim}), got {qk_rope_head_dim}"
        )

    cache_format = HCA_CACHE_FORMAT_F32
    stored_width = _hca_stored_width(cache_format, head_dim, qk_rope_head_dim)
    return HcaLayerPlan(
        layer_id=layer_id,
        num_heads=num_heads,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        cache_format=cache_format,
        stored_width=stored_width,
    )


def _register_domain(op: OpBuilder) -> None:
    """Declare the ``pkg.nxrt`` opset import the ONNX checker requires.

    Custom-domain nodes need a matching ``opset_imports`` entry (the op call
    does not add it automatically), same as ``BlockQuantizedMatMul`` in
    ``components/_quantized_linear.py`` and ``IndexShare`` in
    ``models/glm_moe_dsa.py``.
    """
    op.builder.graph.opset_imports[CSA_DOMAIN] = CSA_OPSET_VERSION


def emit_hca_attention(
    op: OpBuilder,
    plan: HcaLayerPlan,
    *,
    query: ir.Value,
    current_kv: ir.Value,
    compressor_kv: ir.Value,
    compressor_gate: ir.Value,
    compressor_ape: ir.Value,
    compressor_norm: ir.Value,
    past_compressed_kv: ir.Value,
    past_compression_carry: ir.Value,
    seqlens_k: ir.Value,
    total_sequence_length: ir.Value,
    head_sink: ir.Value,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Emit one frozen ratio-128 ``pkg.nxrt::CompressedSparseAttention`` node.

    Callers pass already-shaped, f32 activations in the frozen input order;
    this only stamps the domain import and the full attribute set. Returns
    ``(Y, present_compressed_kv, present_compression_carry)``.
    """
    _register_domain(op)
    y, present_compressed_kv, present_compression_carry = op.CompressedSparseAttention(
        query,
        current_kv,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm,
        past_compressed_kv,
        past_compression_carry,
        seqlens_k,
        total_sequence_length,
        head_sink,
        num_heads=plan.num_heads,
        head_dim=plan.head_dim,
        qk_rope_head_dim=plan.qk_rope_head_dim,
        compression_ratio=plan.compression_ratio,
        index_num_heads=0,
        index_head_dim=0,
        index_topk=0,
        causal=1,
        cache_layout_version=CSA_LAYOUT_VERSION,
        index_layout_version=CSA_LAYOUT_VERSION,
        sink_mode=CSA_SINK_MODE,
        cache_format=plan.cache_format,
        scale=plan.scale,
        _domain=CSA_DOMAIN,
        _outputs=3,
    )
    return y, present_compressed_kv, present_compression_carry
