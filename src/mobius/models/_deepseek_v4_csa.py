# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native DeepSeek-V4 CSA/HCA export gate and op emission.

The frozen ``pkg.nxrt::CompressedSparseAttention`` v1 kernel already exists,
is merged, and is op-parity-tested in ``onnx-genai``
(``crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs``).
This module is the Mobius *export* side of the native integration: it emits
that op -- for both compression ratios the official ``deepseek-ai/DeepSeek-V4
-Flash`` schedule interleaves -- in place of the dense correctness fallback,
behind the default-off ``config.native_csa`` opt-in.

* ratio-128 "Heavily Compressed Attention" (HCA): compressor-only, f32 cache,
  11 inputs / 3 outputs;
* ratio-4 CSA: compressor + a learned FP4 indexer with top-k selection,
  fp8_e4m3_block64 attention cache + fp4_e2m1_block32 index cache, 19 inputs /
  6 outputs.

Design record: ``onnx-genai/.squad/decisions/inbox/`` ``deckard-deepseek-v4-
csa-hca-cuda-slice.md`` (§5 property ABI) and ``deckard-csa-hca-c2-ratio4-
export.md`` (ratio-4 contract). The authoritative op contract is the Rust
factory/shape-inference:

* required inputs (both ratios, order 0-10): ``query, current_kv,
  compressor_kv, compressor_gate, compressor_ape, compressor_norm,
  past_compressed_kv, past_compression_carry, seqlens_k,
  total_sequence_length, head_sink``;
* ratio-4 additionally appends (order 11-18): ``index_query, index_weight,
  index_compressor_kv, index_compressor_gate, index_compressor_ape,
  index_compressor_norm, past_index_key, past_index_carry``;
* ratio-128 outputs (exactly 3): ``Y, present_compressed_kv,
  present_compression_carry``;
* ratio-4 outputs (6): the three above plus ``present_index_key,
  present_index_carry, selected_indices``;
* attributes ``num_heads, head_dim, qk_rope_head_dim, compression_ratio,
  index_num_heads, index_head_dim, index_topk, causal=1,
  cache_layout_version=1, index_layout_version=1, sink_mode='logit_only',
  cache_format, scale`` (``scale=0`` selects ``1/sqrt(head_dim)``);
  ``index_*`` attrs are ``0`` for ratio-128.

This is a **property gate, not a model-name allowlist**: emission requires the
config-derived properties to match the frozen v1 contract exactly. Anything the
frozen op cannot express when the feature is requested -- MTP recurrence, an
unknown per-layer ratio, quantized compressor projections, or a carry/state
shape that does not match -- raises :class:`NativeCsaExportError` rather than
silently emitting the dense fallback.

Semantic conventions the exporter honours (from the frozen kernel):

* ``query``/``current_kv`` are fed *pre-rotated* (compressed RoPE theta), the
  same as every dense path; the op rotates only its *compressed records*
  internally.
* ALL compressor and indexer activations are fed *raw*, including
  ``index_query``: ``finalize_index_query`` in the kernel applies the
  compressed RoPE + Hadamard + FP4 round-trip to the index query internally.
* Learned top-k tie ordering and the unselected-slot sentinel (``-1``) are
  runtime-frozen; ties raise a typed ``unsupported`` error in the kernel. The
  exporter passes learned tensors through unmodified and never influences the
  selection, so CPU/CUDA oracle bit-parity is preserved by construction.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
from onnxscript import OpBuilder

from mobius._configs import ArchitectureConfig
from mobius.integrations._block_quant import (
    MXFP4_MICROSCALE_BLOCK,
    BlockQuantExportError,
    BlockQuantScheme,
    QuantizedTensorDescriptor,
    QuantKind,
    runtime_representation_gap,
)

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

# ``past_compression_carry``/``past_index_carry`` pack two planes per slot: the
# FP32 pooling accumulator and the FP32 running score-state (``-inf`` init).
CSA_CARRY_PLANES = 2
# Ratio-128 carries one pooling slot per compressed block (``ratio`` slots);
# ratio-4 carries a fixed 8-slot overlap window for both attention and index
# compressors (``CARRY_SLOTS`` in the Rust kernel's ``execute_ratio4``).
RATIO4_CARRY_SLOTS = 8

# v1 requires ``sink_mode='logit_only'`` for the learned per-head ``head_sink``
# denominator term.
CSA_SINK_MODE = "logit_only"

# Cache formats. Ratio-128 attention-compressor records are f32; ratio-4
# requires the packed hybrid FP8/BF16 attention record format and the packed
# FP4 index-key format.
HCA_CACHE_FORMAT_F32 = "f32"
CSA_CACHE_FORMAT_FP8 = "fp8_e4m3_block64"
CSA_INDEX_CACHE_FORMAT_FP4 = "fp4_e2m1_block32"

# Packed-record block geometry, mirroring the Rust kernel constants.
# fp8_e4m3_block64: each 64-wide non-RoPE block stores 1 E8M0 scale byte + 64
# E4M3 bytes (= 65), plus a little-endian BF16 (2-byte) tail for each RoPE dim.
_FP8_BLOCK = 64
_FP8_BYTES_PER_BLOCK = _FP8_BLOCK + 1
# fp4_e2m1_block32: each 32-wide block stores 1 E8M0 scale byte + 16
# adjacent-nibble bytes (= 17).
_FP4_BLOCK = 32
_FP4_BYTES_PER_BLOCK = 17

# ``scale=0`` is the frozen-op sentinel for ``1/sqrt(head_dim)`` (the Rust
# kernel: ``if configured_scale == 0.0 { 1/sqrt(head_dim) }``), matching the
# op-parity test which omits the attribute entirely.
CSA_DEFAULT_SCALE_SENTINEL = 0.0

# Authoritative required-input order shared by both ratios (positions 0-10).
CSA_BASE_INPUT_NAMES: tuple[str, ...] = (
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
# Ratio-4 appends the learned-indexer inputs (positions 11-18).
CSA_INDEX_INPUT_NAMES: tuple[str, ...] = (
    "index_query",
    "index_weight",
    "index_compressor_kv",
    "index_compressor_gate",
    "index_compressor_ape",
    "index_compressor_norm",
    "past_index_key",
    "past_index_carry",
)
HCA_INPUT_NAMES: tuple[str, ...] = CSA_BASE_INPUT_NAMES
CSA_RATIO4_INPUT_NAMES: tuple[str, ...] = CSA_BASE_INPUT_NAMES + CSA_INDEX_INPUT_NAMES

HCA_OUTPUT_NAMES: tuple[str, ...] = (
    "Y",
    "present_compressed_kv",
    "present_compression_carry",
)
CSA_RATIO4_OUTPUT_NAMES: tuple[str, ...] = (
    *HCA_OUTPUT_NAMES,
    "present_index_key",
    "present_index_carry",
    "selected_indices",
)


class NativeCsaExportError(ValueError):
    """Native CSA/HCA export was requested but cannot be satisfied.

    Raised (fail-closed) instead of silently emitting the dense correctness
    fallback whenever ``config.native_csa`` is set but a layer does not match
    the frozen ``pkg.nxrt::CompressedSparseAttention`` v1 contract.
    """


def _fp8_block64_width(head_dim: int, qk_rope_head_dim: int) -> int:
    """Packed ``fp8_e4m3_block64`` attention record width.

    Mirrors ``CacheFormat::Fp8E4m3Block64::stored_width``: the non-RoPE head
    dimension is packed in 64-wide E4M3 blocks (65 bytes each) and the RoPE
    tail is stored as little-endian BF16 (2 bytes per dim).
    """
    non_rope = head_dim - qk_rope_head_dim
    if non_rope < 0 or non_rope % _FP8_BLOCK != 0:
        raise NativeCsaExportError(
            f"ratio-{CSA_COMPRESSION_RATIO} cache_format='{CSA_CACHE_FORMAT_FP8}' "
            f"requires (head_dim - qk_rope_head_dim) >= 0 and divisible by "
            f"{_FP8_BLOCK}; got head_dim={head_dim}, qk_rope_head_dim="
            f"{qk_rope_head_dim} (non-RoPE width {non_rope})"
        )
    return (non_rope // _FP8_BLOCK) * _FP8_BYTES_PER_BLOCK + qk_rope_head_dim * 2


def _fp4_width(logical_width: int) -> int:
    """Packed ``fp4_e2m1_block32`` record width.

    Mirrors ``fp4_width``: each 32-wide block stores 1 E8M0 scale byte + 16
    adjacent-nibble bytes (17 bytes).
    """
    if logical_width <= 0 or logical_width % _FP4_BLOCK != 0:
        raise NativeCsaExportError(
            f"ratio-{CSA_COMPRESSION_RATIO} index cache_format="
            f"'{CSA_INDEX_CACHE_FORMAT_FP4}' requires a positive width divisible "
            f"by {_FP4_BLOCK}; got {logical_width}"
        )
    return (logical_width // _FP4_BLOCK) * _FP4_BYTES_PER_BLOCK


@dataclasses.dataclass(frozen=True)
class CsaLayerPlan:
    """Resolved, contract-checked native emission plan for one layer.

    Every field is a native-op *property*, derived from the architecture
    config and validated against the frozen v1 schema. The deterministic state
    IO names let the graph task thread ``past_* -> present_*`` compressed
    (and, for ratio-4, index) state without guessing.

    ``compressor_width`` doubles as the ``past_compression_carry`` trailing
    width: ratio-128 pools single-record blocks (``head_dim``) while ratio-4
    pools overlapping key+value records (``2 * head_dim``). ``index_*`` fields
    are ``0``/empty for ratio-128.
    """

    layer_id: int
    compression_ratio: int
    num_heads: int
    head_dim: int
    qk_rope_head_dim: int
    cache_format: str
    cache_dtype: ir.DataType
    stored_width: int
    compressor_width: int
    carry_slots: int
    carry_planes: int = CSA_CARRY_PLANES
    scale: float = CSA_DEFAULT_SCALE_SENTINEL
    # Ratio-4 learned indexer properties (0/empty for ratio-128).
    index_num_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    index_stored_width: int = 0
    index_compressor_width: int = 0
    index_cache_format: str = ""
    index_key_dtype: ir.DataType | None = None

    @property
    def is_ratio4(self) -> bool:
        return self.compression_ratio == CSA_COMPRESSION_RATIO

    # -- shared compressed-state IO names ---------------------------------
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

    # -- ratio-4 index-state IO names -------------------------------------
    @property
    def past_index_key_name(self) -> str:
        return f"past_index_key.{self.layer_id}"

    @property
    def past_index_carry_name(self) -> str:
        return f"past_index_carry.{self.layer_id}"

    @property
    def present_index_key_name(self) -> str:
        return f"present_index_key.{self.layer_id}"

    @property
    def present_index_carry_name(self) -> str:
        return f"present_index_carry.{self.layer_id}"

    @property
    def selected_indices_name(self) -> str:
        return f"selected_indices.{self.layer_id}"

    @property
    def past_records_axis_name(self) -> str:
        return f"past_compressed_records.{self.layer_id}"

    @property
    def present_records_axis_name(self) -> str:
        return f"present_compressed_records.{self.layer_id}"

    @property
    def selected_records_axis_name(self) -> str:
        return f"selected_records.{self.layer_id}"


def _layer_compress_ratio(config: ArchitectureConfig, layer_id: int) -> int:
    ratios = config.compress_ratios or []
    return ratios[layer_id] if layer_id < len(ratios) else 0


def _require_positive(value, name: str, layer_id: int) -> int:
    if value is None or value <= 0:
        raise NativeCsaExportError(
            f"native CSA layer {layer_id} requires a positive {name}, got {value}"
        )
    return value


def _plan_ratio128(
    config: ArchitectureConfig,
    layer_id: int,
    num_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
) -> CsaLayerPlan:
    return CsaLayerPlan(
        layer_id=layer_id,
        compression_ratio=HCA_COMPRESSION_RATIO,
        num_heads=num_heads,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        cache_format=HCA_CACHE_FORMAT_F32,
        cache_dtype=ir.DataType.FLOAT,
        stored_width=head_dim,
        compressor_width=head_dim,
        carry_slots=HCA_COMPRESSION_RATIO,
    )


def _plan_ratio4(
    config: ArchitectureConfig,
    layer_id: int,
    num_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
) -> CsaLayerPlan:
    index_num_heads = _require_positive(config.index_n_heads, "index_n_heads", layer_id)
    index_head_dim = _require_positive(config.index_head_dim, "index_head_dim", layer_id)
    index_topk = _require_positive(config.index_topk, "index_topk", layer_id)
    return CsaLayerPlan(
        layer_id=layer_id,
        compression_ratio=CSA_COMPRESSION_RATIO,
        num_heads=num_heads,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        cache_format=CSA_CACHE_FORMAT_FP8,
        cache_dtype=ir.DataType.UINT8,
        stored_width=_fp8_block64_width(head_dim, qk_rope_head_dim),
        compressor_width=2 * head_dim,
        carry_slots=RATIO4_CARRY_SLOTS,
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        index_stored_width=_fp4_width(index_head_dim),
        index_compressor_width=2 * index_head_dim,
        index_cache_format=CSA_INDEX_CACHE_FORMAT_FP4,
        index_key_dtype=ir.DataType.UINT8,
    )


def plan_native_csa(
    config: ArchitectureConfig,
    layer_id: int,
    *,
    is_mtp: bool = False,
) -> CsaLayerPlan | None:
    """Resolve a layer's native-CSA emission plan, or ``None`` for dense.

    Returns ``None`` when the feature is off, or when the layer is genuinely a
    dense (ratio-0) attention layer -- those are *not* a suppressed CSA
    fallback, they carry no compressor at all.

    Returns a :class:`CsaLayerPlan` for ratio-128 (HCA) or ratio-4 (CSA)
    layers whose config matches the frozen op contract. Raises
    :class:`NativeCsaExportError` when the feature is requested but the layer
    cannot be expressed: an MTP layer that would need compressed-state
    recurrence (undefined in the pinned source), quantized compressor
    projections, an unknown ratio, or head/rope dims that violate the op
    contract.
    """
    if not config.native_csa:
        return None

    ratio = _layer_compress_ratio(config, layer_id)
    if ratio == 0:
        # A genuine dense/sliding layer (schedule value 0). No compressor
        # exists to route through the op; dense emission here is correct, not
        # a silent CSA fallback. MTP (ratio-0) also lands here and stays dense.
        return None

    if is_mtp:
        # The pinned checkpoint's MTP block is ratio-0 dense (handled above).
        # A compressed MTP layer's recurrence/acceptance/KV-lifetime is
        # undefined in source (design blocker B5) and must never be synthesised.
        raise NativeCsaExportError(
            f"native CSA requested for MTP layer {layer_id} with "
            f"compression_ratio={ratio}; MTP compressed-state recurrence is "
            "undefined in the pinned DeepSeek-V4 source and must not be "
            "synthesised (design blocker B5). MTP stays ratio-0 dense"
        )

    if ratio not in (CSA_COMPRESSION_RATIO, HCA_COMPRESSION_RATIO):
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
            "compressor/index activations and this slice only wires unquantized "
            "compressor projections. Quantized compressor dequant is a "
            "follow-up slice"
        )

    head_dim = _require_positive(config.head_dim, "head_dim", layer_id)
    num_heads = _require_positive(config.num_attention_heads, "num_attention_heads", layer_id)
    qk_rope_head_dim = config.qk_rope_head_dim or 0
    if qk_rope_head_dim < 0 or qk_rope_head_dim > head_dim:
        raise NativeCsaExportError(
            f"native CSA layer {layer_id} requires 0 <= qk_rope_head_dim <= "
            f"head_dim ({head_dim}), got {qk_rope_head_dim}"
        )

    if ratio == HCA_COMPRESSION_RATIO:
        return _plan_ratio128(config, layer_id, num_heads, head_dim, qk_rope_head_dim)
    return _plan_ratio4(config, layer_id, num_heads, head_dim, qk_rope_head_dim)


# Runtime-capability gate --------------------------------------------------
#
# Graph construction for a native-CSA export of the block-scaled-FP8 /
# packed-FP4 DeepSeek-V4-Flash checkpoint is allowed to *progress* (the
# ``ArchitectureConfig`` records the deferred ``block_quant_scheme`` rather than
# rejecting at config resolution), so the CSA nodes and their compressed state
# IO can compose with the canonical planar block-quant producer. Runtime
# representability remains a property gate owned by
# ``mobius.integrations._block_quant``.


def _representative_block_fp8_descriptor(
    scheme: BlockQuantScheme,
) -> QuantizedTensorDescriptor:
    """A property-faithful stand-in for a block-FP8 projection tensor.

    Only the fields ``runtime_representation_gap`` reads for a ``BLOCK_FP8``
    tensor need be meaningful; the shapes are placeholders (the gate is a
    capability probe, not an emission).
    """
    block = tuple(int(x) for x in scheme.weight_block_size) or (128, 128)
    return QuantizedTensorDescriptor(
        name="model.layers.*.self_attn.kv_proj.weight",
        kind=QuantKind.BLOCK_FP8,
        weight_dtype=scheme.weight_fmt or "e4m3",
        logical_shape=(0, 0),
        packed_shape=(0, 0),
        weight_num_bytes=0,
        is_routed_expert=False,
        is_shared_expert=False,
        block_shape=block,
        scale_dtype=scheme.scale_fmt or "ue8m0",
        scale_shape=(0, 0),
        scale_layout="2d_block",
    )


def _representative_fp4_expert_descriptor(
    scheme: BlockQuantScheme,
) -> QuantizedTensorDescriptor:
    """A property-faithful stand-in for a packed-FP4 routed-expert tensor."""
    return QuantizedTensorDescriptor(
        name="model.layers.*.mlp.experts.*.gate_proj.weight",
        kind=QuantKind.FP4_PACKED,
        weight_dtype=scheme.expert_dtype or "fp4",
        logical_shape=(0, 0),
        packed_shape=(0, 0),
        weight_num_bytes=0,
        is_routed_expert=True,
        is_shared_expert=False,
        block_shape=(MXFP4_MICROSCALE_BLOCK,),
        scale_dtype="ue8m0",
        scale_shape=(0, 0),
        scale_layout="planar_microscale",
        microscale_kind="mxfp4",
    )


def native_runtime_block_quant_gap(
    scheme: BlockQuantScheme | None, *, runtime: str = "nxrt"
) -> str | None:
    """Return a runtime ABI-gap string if *runtime* cannot execute *scheme*.

    ``None`` means the gate is open (either there is no owned block-quant
    scheme, or the runtime can now represent every family it needs). Consumes
    ``mobius.integrations._block_quant.runtime_representation_gap`` so this gate
    tracks the real ``nxrt`` format strings and opens with no change here.
    """
    if scheme is None or not scheme.is_owned:
        return None
    gaps: list[str] = []
    if scheme.is_block_scaled_fp8:
        gaps.append(
            runtime_representation_gap(
                _representative_block_fp8_descriptor(scheme), runtime=runtime
            )
            or ""
        )
    if scheme.has_packed_fp4_experts:
        gaps.append(
            runtime_representation_gap(
                _representative_fp4_expert_descriptor(scheme), runtime=runtime
            )
            or ""
        )
    gaps = [g for g in gaps if g]
    return "\n".join(gaps) if gaps else None


def assert_native_runtime_supports_block_quant(
    config: ArchitectureConfig, *, runtime: str = "nxrt"
) -> None:
    """Full-export runtime-capability gate for a deferred block-quant scheme.

    No-op when the canonical runtime represents the recorded scheme. Raises the
    typed :class:`BlockQuantExportError` for an unknown/unrepresentable runtime;
    never selects a silent dense fallback.
    """
    scheme = getattr(config, "block_quant_scheme", None)
    gap = native_runtime_block_quant_gap(scheme, runtime=runtime)
    if gap is None:
        return
    raise BlockQuantExportError(
        "native CSA full export requires a runtime that can execute the "
        f"checkpoint's block-quant weights, but {runtime!r} cannot yet. Graph "
        "construction progressed (CSA nodes + compressed state IO are built), "
        "and no dense fallback is permitted (mobius.integrations._block_quant."
        "runtime_representation_gap / plan_routed_expert_bank). ABI gap:\n"
        f"{gap}"
    )


def _register_domain(op: OpBuilder) -> None:
    """Declare the ``pkg.nxrt`` opset import the ONNX checker requires.

    Custom-domain nodes need a matching ``opset_imports`` entry (the op call
    does not add it automatically), same as ``BlockQuantizedMatMul`` in
    ``components/_quantized_linear.py`` and ``IndexShare`` in
    ``models/glm_moe_dsa.py``.
    """
    op.builder.graph.opset_imports[CSA_DOMAIN] = CSA_OPSET_VERSION


def _csa_attributes(plan: CsaLayerPlan) -> dict:
    """The full frozen-v1 attribute set for a plan (index_* are 0 for HCA)."""
    return dict(
        num_heads=plan.num_heads,
        head_dim=plan.head_dim,
        qk_rope_head_dim=plan.qk_rope_head_dim,
        compression_ratio=plan.compression_ratio,
        index_num_heads=plan.index_num_heads,
        index_head_dim=plan.index_head_dim,
        index_topk=plan.index_topk,
        causal=1,
        cache_layout_version=CSA_LAYOUT_VERSION,
        index_layout_version=CSA_LAYOUT_VERSION,
        sink_mode=CSA_SINK_MODE,
        cache_format=plan.cache_format,
        scale=plan.scale,
    )


def emit_csa_attention(
    op: OpBuilder,
    plan: CsaLayerPlan,
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
    index_query: ir.Value | None = None,
    index_weight: ir.Value | None = None,
    index_compressor_kv: ir.Value | None = None,
    index_compressor_gate: ir.Value | None = None,
    index_compressor_ape: ir.Value | None = None,
    index_compressor_norm: ir.Value | None = None,
    past_index_key: ir.Value | None = None,
    past_index_carry: ir.Value | None = None,
) -> tuple[ir.Value, ...]:
    """Emit one frozen ``pkg.nxrt::CompressedSparseAttention`` node for a plan.

    Callers pass already-shaped activations in the frozen input order (f32 for
    the float inputs; uint8 for the packed ratio-4 ``past_*_key`` state); this
    only stamps the domain import and the full attribute set.

    * ratio-128 (HCA): 11 inputs, returns ``(Y, present_compressed_kv,
      present_compression_carry)``.
    * ratio-4 (CSA): 19 inputs, returns the three above plus
      ``(present_index_key, present_index_carry, selected_indices)``.

    Requesting a ratio-4 emission without every learned-indexer input raises
    :class:`NativeCsaExportError` -- fail-closed, never a partial/silent node.
    """
    _register_domain(op)
    attributes = _csa_attributes(plan)

    if not plan.is_ratio4:
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
            _domain=CSA_DOMAIN,
            _outputs=3,
            **attributes,
        )
        return y, present_compressed_kv, present_compression_carry

    index_inputs = {
        "index_query": index_query,
        "index_weight": index_weight,
        "index_compressor_kv": index_compressor_kv,
        "index_compressor_gate": index_compressor_gate,
        "index_compressor_ape": index_compressor_ape,
        "index_compressor_norm": index_compressor_norm,
        "past_index_key": past_index_key,
        "past_index_carry": past_index_carry,
    }
    missing = [name for name, value in index_inputs.items() if value is None]
    if missing:
        raise NativeCsaExportError(
            f"native CSA ratio-{CSA_COMPRESSION_RATIO} layer {plan.layer_id} "
            f"requires the learned-indexer inputs {missing}; none may be "
            "absent. The frozen op has no dense fallback for a partial indexer"
        )

    (
        y,
        present_compressed_kv,
        present_compression_carry,
        present_index_key,
        present_index_carry,
        selected_indices,
    ) = op.CompressedSparseAttention(
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
        index_query,
        index_weight,
        index_compressor_kv,
        index_compressor_gate,
        index_compressor_ape,
        index_compressor_norm,
        past_index_key,
        past_index_carry,
        _domain=CSA_DOMAIN,
        _outputs=6,
        **attributes,
    )
    return (
        y,
        present_compressed_kv,
        present_compression_carry,
        present_index_key,
        present_index_carry,
        selected_indices,
    )
