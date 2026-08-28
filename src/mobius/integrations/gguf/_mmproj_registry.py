# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned capability registry for ``general.architecture = "clip"`` sidecars.

The census and metadata keys come from llama.cpp commit
``8d9af256337d1a501250f9bbf4c0859a654bddd6``:

* ``tools/mtmd/clip-impl.h`` defines the 62-value enum (60 serialized strings,
  one unserialized ``MLP_NORM`` compatibility value, and ``UNKNOWN``).
* ``tools/mtmd/clip.cpp`` defines metadata requirements/defaults and dispatch.

Listing a projector is not a support claim. Only entries with all four
capabilities marked ``SUPPORTED`` may reach graph construction.
"""

from __future__ import annotations

__all__ = [
    "CLIP_METADATA_SCHEMA",
    "LLAMA_CPP_MMPROJ_SHA",
    "MMPROJ_ARTIFACT_AVAILABILITY_PINS",
    "MMPROJ_ARTIFACT_PINS",
    "MMPROJ_SOURCE_EVIDENCE",
    "ClipMetadataField",
    "BlockTensorVariantSpec",
    "CompanionTensorSpec",
    "DeferredCompanionSpec",
    "IndexedTensorSpec",
    "MMProjArtifactAvailabilityPin",
    "MMProjArtifactPin",
    "MMProjSourceEvidence",
    "MMProjModelRole",
    "MMProjModality",
    "MMProjTensorRole",
    "ProjectorSpec",
    "get_projector_spec",
    "iter_projector_specs",
    "iter_projector_source_evidence",
    "projector_source_evidence",
    "projector_type_for_modality",
    "supported_projector_types",
]

import dataclasses
import enum
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from mobius.integrations.gguf._remaining_projector_evidence import (
    REMAINING_MMPROJ_ARTIFACT_RECORDS,
    REMAINING_MMPROJ_SOURCE_RECORDS,
)
from mobius.integrations.gguf._spec import Support

LLAMA_CPP_MMPROJ_SHA = "86632248188c106d749fad34a1dcd237c95863d4"


class MMProjModality(enum.Enum):
    """Modalities represented by the pinned clip loader."""

    VISION = "vision"
    AUDIO = "audio"
    GENERATED_AUDIO = "gen.audio"


class MMProjTensorRole(enum.Enum):
    """Storage policy for a tensor owned by an mmproj sidecar."""

    ENCODER = "encoder"
    PROJECTOR = "projector"
    CALIBRATION = "calibration"
    GENERATED_AUDIO = "generated_audio"


class MMProjModelRole(enum.Enum):
    """Executable component roles produced from an mmproj sidecar."""

    VISION_ENCODER = "vision_encoder"
    AUDIO_ENCODER = "audio_encoder"
    SPEAKER_ENCODER = "speaker_encoder"


@dataclasses.dataclass(frozen=True, slots=True)
class ClipMetadataField:
    """One key in the pinned ``clip.*`` metadata schema."""

    key: str
    required_for: frozenset[MMProjModality] = frozenset()
    default: object | None = None
    note: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class BlockTensorVariantSpec:
    """Per-block suffix closure selected by the presence of one trigger tensor."""

    trigger_suffix: str
    default_suffixes: tuple[str, ...]
    triggered_suffixes: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class IndexedTensorSpec:
    """An auxiliary tensor family whose index count comes from metadata."""

    count_metadata: str
    prefix: str
    required_suffixes: tuple[str, ...]
    count_is_array_length: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class CompanionTensorSpec:
    """Exact tensor closure for a deferred modality sharing a supported sidecar."""

    modality: MMProjModality
    projector_type: str
    required_metadata: tuple[str, ...]
    required_top_tensors: tuple[str, ...]
    block_prefix: str
    block_suffixes: tuple[str, ...]
    optional_top_tensors: tuple[str, ...] = ()
    block_variant: BlockTensorVariantSpec | None = None
    indexed_tensors: tuple[IndexedTensorSpec, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class DeferredCompanionSpec:
    """Explicitly quarantined co-resident modality that is not built."""

    modality: MMProjModality
    projector_type: str
    tensor_prefixes: tuple[str, ...]
    reason: str


@dataclasses.dataclass(frozen=True, slots=True)
class ProjectorSpec:
    """Capabilities and exact loader closure for one serialized projector type."""

    projector_type: str
    enum_name: str
    modalities: frozenset[MMProjModality]
    target_architectures: frozenset[str] = frozenset()
    primary_modality: MMProjModality = MMProjModality.VISION
    metadata: Support = Support.DEFERRED
    tensor_map: Support = Support.DEFERRED
    graph: Support = Support.DEFERRED
    runtime: Support = Support.DEFERRED
    reason: str | None = None
    # ``builder`` assembles a paired text+sidecar package. ``sidecar_builder``
    # builds only the explicitly named encoder/projector components.
    builder: str | None = None
    sidecar_builder: str | None = None
    model_roles: tuple[MMProjModelRole, ...] = ()
    required_metadata: tuple[str, ...] = ()
    required_top_tensors: tuple[str, ...] = ()
    optional_top_tensors: tuple[str, ...] = ()
    block_prefix: str | None = None
    block_suffixes: tuple[str, ...] = ()
    block_suffix_variants: tuple[tuple[str, ...], ...] = ()
    auxiliary_tensor_patterns: tuple[str, ...] = ()
    block_variant: BlockTensorVariantSpec | None = None
    indexed_tensors: tuple[IndexedTensorSpec, ...] = ()
    optional_top_tensor_groups: tuple[tuple[str, ...], ...] = ()
    companion_tensors: tuple[CompanionTensorSpec, ...] = ()
    deferred_companions: tuple[DeferredCompanionSpec, ...] = ()
    tensor_roles: tuple[tuple[str, MMProjTensorRole], ...] = ()
    real_artifact_ids: tuple[str, ...] = ()
    source_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        verdicts = (self.metadata, self.tensor_map, self.graph, self.runtime)
        if any(verdict is not Support.SUPPORTED for verdict in verdicts) and not self.reason:
            raise ValueError(f"{self.projector_type}: unsupported capability needs a reason")
        has_loader_data = any(
            (
                self.builder,
                self.sidecar_builder,
                self.model_roles,
                self.required_metadata,
                self.required_top_tensors,
                self.optional_top_tensors,
                self.block_prefix,
                self.block_suffixes,
                self.block_suffix_variants,
                self.auxiliary_tensor_patterns,
                self.block_variant,
                self.indexed_tensors,
                self.optional_top_tensor_groups,
                self.companion_tensors,
                self.deferred_companions,
                self.tensor_roles,
                self.real_artifact_ids,
                self.source_evidence_ids,
            )
        )
        has_supported_schema = (
            self.metadata is Support.SUPPORTED and self.tensor_map is Support.SUPPORTED
        )
        if has_supported_schema and has_loader_data:
            if not (self.builder or self.sidecar_builder) or not self.target_architectures:
                raise ValueError(
                    f"{self.projector_type}: mapped projector needs a builder and target"
                )
            if self.primary_modality not in self.modalities:
                raise ValueError(
                    f"{self.projector_type}: primary modality must be one of its modalities"
                )
            if self.sidecar_builder and not self.model_roles:
                raise ValueError(
                    f"{self.projector_type}: standalone sidecar builder needs model roles"
                )
            if not self.required_metadata or not self.required_top_tensors:
                raise ValueError(
                    f"{self.projector_type}: mapped projector needs an exact loader closure"
                )
            if not (self.real_artifact_ids or self.source_evidence_ids):
                raise ValueError(
                    f"{self.projector_type}: mapped projector needs artifact or source evidence"
                )
        elif has_loader_data:
            raise ValueError(
                f"{self.projector_type}: rejected metadata/tensor mapping cannot expose loader data"
            )

    @property
    def verdicts(self) -> Mapping[str, Support]:
        return MappingProxyType(
            {
                "metadata": self.metadata,
                "tensor_map": self.tensor_map,
                "graph": self.graph,
                "runtime": self.runtime,
            }
        )

    @property
    def is_importable(self) -> bool:
        """Whether metadata, tensor mapping, and graph construction are supported."""
        return all(
            verdict is Support.SUPPORTED
            for verdict in (self.metadata, self.tensor_map, self.graph)
        )

    @property
    def is_supported(self) -> bool:
        """Whether the complete graph and runtime package are supported."""
        return self.is_importable and self.runtime is Support.SUPPORTED


@dataclasses.dataclass(frozen=True, slots=True)
class MMProjArtifactPin:
    """Audited real sidecar used as evidence for a supported registry entry."""

    artifact_id: str
    repository: str
    revision: str
    filename: str
    size: int
    lfs_sha256: str
    projector_types: tuple[str, ...]
    paired_text_architecture: str
    paired_text_target: str
    metadata: tuple[tuple[str, object], ...]
    tensor_qtypes: tuple[tuple[str, int], ...]
    tensor_count: int
    parity_test: str
    bounded_header_bytes: int | None = None
    bounded_header_sha256: str | None = None
    paired_text_repository: str | None = None
    paired_text_revision: str | None = None
    paired_text_size: int | None = None
    processor_repository: str | None = None
    processor_revision: str | None = None
    processor_files: tuple[str, ...] = ()
    processor_class: str | None = None
    processor_contract: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.size <= 0 or len(self.revision) != 40 or len(self.lfs_sha256) != 64:
            raise ValueError(f"{self.artifact_id}: immutable sidecar identity is required")
        paired_fields = (
            self.paired_text_repository,
            self.paired_text_revision,
            self.paired_text_size,
        )
        if any(value is not None for value in paired_fields):
            if (
                not all(value is not None for value in paired_fields)
                or len(self.paired_text_revision or "") != 40
                or int(self.paired_text_size or 0) <= 0
            ):
                raise ValueError(
                    f"{self.artifact_id}: paired text repository, immutable revision, "
                    "and positive size must be specified together"
                )
        if self.processor_repository is not None and len(self.processor_revision or "") != 40:
            raise ValueError(
                f"{self.artifact_id}: processor source requires an immutable revision"
            )
        if (self.bounded_header_bytes is None) != (self.bounded_header_sha256 is None):
            raise ValueError(
                f"{self.artifact_id}: bounded header size and SHA-256 must be specified together"
            )
        if self.bounded_header_bytes is not None and (
            self.bounded_header_bytes <= 0
            or self.bounded_header_bytes > self.size
            or len(self.bounded_header_sha256 or "") != 64
        ):
            raise ValueError(f"{self.artifact_id}: bounded header identity is invalid")


@dataclasses.dataclass(frozen=True, slots=True)
class MMProjArtifactAvailabilityPin:
    """Immutable sidecar identity that proves availability, not import parity."""

    artifact_id: str
    projector_type: str
    repository: str
    revision: str
    filename: str
    size: int
    lfs_sha256: str

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"{self.artifact_id}: artifact size must be positive")
        if len(self.revision) != 40 or len(self.lfs_sha256) != 64:
            raise ValueError(
                f"{self.artifact_id}: immutable revision and SHA-256 are required"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class MMProjSourceEvidence:
    """Immutable source-level proof used when no valid converter artifact exists."""

    evidence_id: str
    sources: tuple[tuple[str, str, str], ...]
    finding: str

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.finding:
            raise ValueError("mmproj source evidence needs an id and finding")
        if not self.sources:
            raise ValueError(f"{self.evidence_id}: source evidence cannot be empty")
        for repository, revision, path in self.sources:
            if not repository or len(revision) != 40 or not path:
                raise ValueError(
                    f"{self.evidence_id}: every source needs repository, commit, and path"
                )


_VISION_BASE = frozenset({MMProjModality.VISION})
_AUDIO_BASE = frozenset({MMProjModality.AUDIO})
_GEN_AUDIO_BASE = frozenset({MMProjModality.GENERATED_AUDIO})

# ``None`` means no upstream default: the key is contextual or model-specific.
# The loader requires the six formatted base fields for every active modality.
CLIP_METADATA_SCHEMA: tuple[ClipMetadataField, ...] = (
    ClipMetadataField("clip.projector_type", note="Fallback projector for every modality."),
    ClipMetadataField("clip.has_vision_encoder", default=False),
    ClipMetadataField("clip.has_audio_encoder", default=False),
    ClipMetadataField("clip.has_gen_audio_encoder", default=False),
    ClipMetadataField(
        "clip.has_text_encoder", note="Absent from the pinned ABI; always false."
    ),
    ClipMetadataField("clip.use_gelu", default=False),
    ClipMetadataField("clip.use_silu", default=False),
    ClipMetadataField(
        "clip.{modality}.embedding_length", _VISION_BASE | _AUDIO_BASE | _GEN_AUDIO_BASE
    ),
    ClipMetadataField(
        "clip.{modality}.feed_forward_length", _VISION_BASE | _AUDIO_BASE | _GEN_AUDIO_BASE
    ),
    ClipMetadataField(
        "clip.{modality}.block_count", _VISION_BASE | _AUDIO_BASE | _GEN_AUDIO_BASE
    ),
    ClipMetadataField(
        "clip.{modality}.projection_dim", _VISION_BASE | _AUDIO_BASE | _GEN_AUDIO_BASE
    ),
    ClipMetadataField(
        "clip.{modality}.attention.head_count", _VISION_BASE | _AUDIO_BASE | _GEN_AUDIO_BASE
    ),
    ClipMetadataField(
        "clip.{modality}.attention.head_count_kv", default="attention.head_count"
    ),
    ClipMetadataField("clip.{modality}.attention.head_dim"),
    ClipMetadataField(
        "clip.{modality}.attention.layer_norm_epsilon",
        _VISION_BASE | _AUDIO_BASE | _GEN_AUDIO_BASE,
    ),
    ClipMetadataField("clip.{modality}.feature_layer", default=()),
    ClipMetadataField("clip.vision.projector_type"),
    ClipMetadataField("clip.vision.image_size", _VISION_BASE),
    ClipMetadataField("clip.vision.image_min_pixels"),
    ClipMetadataField("clip.vision.image_max_pixels"),
    ClipMetadataField("clip.vision.preproc_min_tiles"),
    ClipMetadataField("clip.vision.preproc_max_tiles"),
    ClipMetadataField("clip.vision.preproc_image_size"),
    ClipMetadataField("clip.vision.patch_size", _VISION_BASE),
    ClipMetadataField("clip.vision.image_mean", _VISION_BASE),
    ClipMetadataField("clip.vision.image_std", _VISION_BASE),
    ClipMetadataField("clip.vision.projector.scale_factor"),
    ClipMetadataField("clip.vision.projector.query_side"),
    ClipMetadataField("clip.vision.projector.window_side"),
    ClipMetadataField("clip.vision.projector.spatial_offsets"),
    ClipMetadataField("clip.vision.spatial_merge_size"),
    ClipMetadataField("clip.vision.mm_patch_merge_type", default="flat"),
    ClipMetadataField("clip.vision.image_grid_pinpoints", default=()),
    ClipMetadataField("clip.vision.n_wa_pattern"),
    ClipMetadataField("clip.vision.wa_layer_indexes"),
    ClipMetadataField("clip.vision.is_deepstack_layers"),
    ClipMetadataField("clip.vision.wa_pattern_mode"),
    ClipMetadataField("clip.vision.window_size"),
    ClipMetadataField("clip.minicpmv_version"),
    ClipMetadataField("clip.minicpmv_query_num", default="version-dependent"),
    ClipMetadataField("clip.vision.sam.head_count"),
    ClipMetadataField("clip.vision.sam.block_count"),
    ClipMetadataField("clip.vision.sam.embedding_length"),
    ClipMetadataField(
        "clip.vision.expert_count_per_layer",
        note="Informational converter output; block tensor presence selects dense versus MoE.",
    ),
    ClipMetadataField("clip.vision.expert_used_count"),
    ClipMetadataField("clip.audio.projector_type"),
    ClipMetadataField("clip.audio.num_mel_bins", _AUDIO_BASE),
    ClipMetadataField("clip.audio.projector.stack_factor"),
    ClipMetadataField("clip.audio.chunk_size"),
    ClipMetadataField("clip.audio.conv_kernel_size"),
    ClipMetadataField("clip.audio.max_pos_emb"),
    ClipMetadataField("clip.audio.projector.window_size"),
    ClipMetadataField("clip.audio.projector.downsample_rate"),
    ClipMetadataField("clip.audio.projector.head_count"),
    ClipMetadataField("clip.audio.rvq.num_quantizers"),
    ClipMetadataField("clip.audio.rvq.codebook_size"),
    ClipMetadataField("clip.audio.wa_pattern_mode"),
    ClipMetadataField("clip.audio.window_size"),
    ClipMetadataField("clip.audio.local_block_count"),
    ClipMetadataField("clip.audio.local_group_size"),
    ClipMetadataField("clip.gen.audio.projector_type"),
    ClipMetadataField("clip.gen.audio.model_variant"),
    ClipMetadataField("clip.audio.subsampling_factor"),
)

_RANGE_STATS = (
    "attn_q.input_max",
    "attn_q.input_min",
    "attn_q.output_max",
    "attn_q.output_min",
    "attn_k.input_max",
    "attn_k.input_min",
    "attn_k.output_max",
    "attn_k.output_min",
    "attn_v.input_max",
    "attn_v.input_min",
    "attn_v.output_max",
    "attn_v.output_min",
    "attn_out.input_max",
    "attn_out.input_min",
    "attn_out.output_max",
    "attn_out.output_min",
    "ffn_gate.input_max",
    "ffn_gate.input_min",
    "ffn_gate.output_max",
    "ffn_gate.output_min",
    "ffn_up.input_max",
    "ffn_up.input_min",
    "ffn_up.output_max",
    "ffn_up.output_min",
    "ffn_down.input_max",
    "ffn_down.input_min",
    "ffn_down.output_max",
    "ffn_down.output_min",
)

_GEMMA4V_BLOCK_SUFFIXES = (
    "ln1.weight",
    "ln2.weight",
    "attn_post_norm.weight",
    "ffn_post_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_out.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    *_RANGE_STATS,
)

_GEMMA4A_CLIPPED_STEMS = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_out",
    "conv_pw1",
    "conv_pw2",
    "ffn_up",
    "ffn_down",
    "ffn_up_1",
    "ffn_down_1",
)

_GEMMA4A_BLOCK_SUFFIXES = (
    "ffn_norm.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "ffn_post_norm.weight",
    "ffn_norm_1.weight",
    "ffn_up_1.weight",
    "ffn_down_1.weight",
    "ffn_post_norm_1.weight",
    "attn_pre_norm.weight",
    "attn_post_norm.weight",
    "ln2.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_out.weight",
    "attn_k_rel.weight",
    "per_dim_scale.weight",
    "conv_norm.weight",
    "norm_conv.weight",
    "conv_pw1.weight",
    "conv_dw.weight",
    "conv_pw2.weight",
    *(
        f"{stem}.{bound}"
        for stem in _GEMMA4A_CLIPPED_STEMS
        for bound in ("input_min", "input_max", "output_min", "output_max")
    ),
)

_GEMMA4A_TOP_TENSORS = (
    "a.conv1d.0.norm.weight",
    "a.conv1d.0.weight",
    "a.conv1d.1.norm.weight",
    "a.conv1d.1.weight",
    "a.input_projection.weight",
    "a.pre_encode.out.bias",
    "a.pre_encode.out.weight",
    "mm.a.input_projection.weight",
)

_MUSE_GLIMMER_BLOCK_SUFFIXES = tuple(
    f"{stem}.{kind}"
    for stem in ("ln1", "ln2", "attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down")
    for kind in ("weight", "bias")
)

_GEMMA3_BLOCK_SUFFIXES = tuple(
    f"{stem}.{kind}"
    for stem in ("ln1", "ln2", "attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down")
    for kind in ("weight", "bias")
)

_GEMMA3N_AUDIO_BLOCK_SUFFIXES = (
    "attn_k.weight",
    "attn_out.weight",
    "attn_q.weight",
    "attn_v.weight",
    "conv_dw.weight",
    "conv_norm.weight",
    "conv_pw1.weight",
    "conv_pw2.weight",
    "ffn_down.weight",
    "ffn_down_1.weight",
    "ffn_norm.weight",
    "ffn_norm_1.weight",
    "ffn_post_norm.weight",
    "ffn_post_norm_1.weight",
    "ffn_up.weight",
    "ffn_up_1.weight",
    "layer_pre_norm.weight",
    "linear_pos.weight",
    "ln1.weight",
    "ln2.weight",
    "norm_conv.weight",
    "per_dim_scale",
)

_LAYER_NORM_GELU_BLOCK_SUFFIXES = (
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_k.bias",
    "attn_v.weight",
    "attn_v.bias",
    "attn_out.weight",
    "attn_out.bias",
    "ln1.weight",
    "ln1.bias",
    "ln2.weight",
    "ln2.bias",
    "ffn_down.weight",
    "ffn_down.bias",
    "ffn_up.weight",
    "ffn_up.bias",
)
_IDEFICS3_BLOCK_SUFFIXES = _LAYER_NORM_GELU_BLOCK_SUFFIXES
_INTERNVL_BLOCK_SUFFIXES = (
    *_LAYER_NORM_GELU_BLOCK_SUFFIXES,
    "ls1.weight",
    "ls2.weight",
)
_LLAMA4_BLOCK_SUFFIXES = _LAYER_NORM_GELU_BLOCK_SUFFIXES
_PIXTRAL_BLOCK_SUFFIXES = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_out.weight",
    "ln1.weight",
    "ln2.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)

_GEMMA3N_VISION_TOP = (
    "v.conv_stem.conv.weight",
    "v.conv_stem.conv.bias",
    "v.conv_stem.bn.weight",
    "v.msfa.ffn.pw_exp.conv.weight",
    "v.msfa.ffn.pw_exp.bn.weight",
    "v.msfa.ffn.pw_proj.conv.weight",
    "v.msfa.ffn.pw_proj.bn.weight",
    "v.msfa.norm.weight",
    "mm.embedding.weight",
    "mm.hard_emb_norm.weight",
    "mm.soft_emb_norm.weight",
    "mm.input_projection.weight",
)

_GEMMA3N_AUDIO_TOP = (
    "a.conv1d.0.weight",
    "a.conv1d.0.norm.weight",
    "a.conv1d.1.weight",
    "a.conv1d.1.norm.weight",
    "a.pre_encode.out.weight",
    "mm.a.embedding.weight",
    "mm.a.hard_emb_norm.weight",
    "mm.a.soft_emb_norm.weight",
    "mm.a.input_projection.weight",
)

_GEMMA4UV_TOP = (
    "v.patch_embd.weight",
    "v.patch_embd.bias",
    "v.patch_norm.1.weight",
    "v.patch_norm.1.bias",
    "v.patch_norm.2.weight",
    "v.patch_norm.2.bias",
    "v.position_embd.weight",
    "v.patch_norm.3.weight",
    "v.patch_norm.3.bias",
    "mm.input_projection.weight",
)

_QWEN2VL_BLOCK_SUFFIXES = tuple(
    f"{stem}.{kind}"
    for stem in (
        "ln1",
        "ln2",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_out",
        "ffn_up",
        "ffn_down",
    )
    for kind in ("weight", "bias")
)

_QWEN25VL_BLOCK_SUFFIXES = (
    "ln1.weight",
    "ln2.weight",
    *(
        f"{stem}.{kind}"
        for stem in (
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_out",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        )
        for kind in ("weight", "bias")
    ),
)

_QWEN3VL_BLOCK_SUFFIXES = tuple(
    f"{stem}.{kind}"
    for stem in (
        "ln1",
        "ln2",
        "attn_qkv",
        "attn_out",
        "ffn_up",
        "ffn_down",
    )
    for kind in ("weight", "bias")
)

_GLM4V_BLOCK_SUFFIXES = (
    "ln1.weight",
    "ln2.weight",
    "attn_qkv.weight",
    "attn_out.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)

_GLMOCR_BLOCK_SUFFIXES = (
    "ln1.weight",
    "ln2.weight",
    "attn_qkv.weight",
    "attn_qkv.bias",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_out.weight",
    "attn_out.bias",
    "ffn_gate.weight",
    "ffn_gate.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
)

_WHISPER_AUDIO_BLOCK_SUFFIXES = (
    "ln1.weight",
    "ln1.bias",
    "ln2.weight",
    "ln2.bias",
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_v.weight",
    "attn_v.bias",
    "attn_out.weight",
    "attn_out.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
)

_QWEN3A_BLOCK_SUFFIXES = (
    *_WHISPER_AUDIO_BLOCK_SUFFIXES,
    "attn_k.bias",
)

_COMMON_AUDIO_BODY_METADATA = (
    "clip.has_audio_encoder",
    "clip.audio.embedding_length",
    "clip.audio.feed_forward_length",
    "clip.audio.block_count",
    "clip.audio.projection_dim",
    "clip.audio.attention.head_count",
    "clip.audio.attention.layer_norm_epsilon",
    "clip.audio.num_mel_bins",
)

_QWEN2A_TOP_TENSORS = (
    "a.conv1d.1.weight",
    "a.conv1d.1.bias",
    "a.conv1d.2.weight",
    "a.conv1d.2.bias",
    "a.position_embd.weight",
    "a.post_ln.weight",
    "a.post_ln.bias",
    "mm.a.fc.weight",
    "mm.a.fc.bias",
)

_QWEN3A_TOP_TENSORS = (
    "a.conv2d.1.weight",
    "a.conv2d.1.bias",
    "a.conv2d.2.weight",
    "a.conv2d.2.bias",
    "a.conv2d.3.weight",
    "a.conv2d.3.bias",
    "a.conv_out.weight",
    "a.position_embd.weight",
    "a.post_ln.weight",
    "a.post_ln.bias",
    "mm.a.mlp.1.weight",
    "mm.a.mlp.1.bias",
    "mm.a.mlp.2.weight",
    "mm.a.mlp.2.bias",
)

_GLMA_TOP_TENSORS = (
    "a.conv1d.1.weight",
    "a.conv1d.1.bias",
    "a.conv1d.2.weight",
    "a.conv1d.2.bias",
    "a.position_embd.weight",
    "a.post_ln.weight",
    "a.post_ln.bias",
    "mm.a.mlp.1.weight",
    "mm.a.mlp.1.bias",
    "mm.a.mlp.2.weight",
    "mm.a.mlp.2.bias",
    "mm.a.norm_pre.weight",
    "mm.a.norm_pre.bias",
    "v.boi",
    "v.eoi",
)

_QWEN3TTS_SPEAKER_TOP_TENSOR_LIST = [
    "a.conv1d.0.weight",
    "a.conv1d.0.bias",
    "a.conv_out.weight",
    "a.conv_out.bias",
    "a.asp_attn.weight",
    "a.asp_attn.bias",
    "a.asp_tdnn.weight",
    "a.asp_tdnn.bias",
    "mm.a.fc.weight",
    "mm.a.fc.bias",
]
for _speaker_block in range(1, 4):
    for _speaker_stem in ("conv_pw1", "conv_pw2", "se_conv1", "se_conv2"):
        for _speaker_kind in ("weight", "bias"):
            _QWEN3TTS_SPEAKER_TOP_TENSOR_LIST.append(
                f"a.blk.{_speaker_block}.{_speaker_stem}.{_speaker_kind}"
            )
    for _speaker_branch in range(7):
        for _speaker_kind in ("weight", "bias"):
            _QWEN3TTS_SPEAKER_TOP_TENSOR_LIST.append(
                f"a.blk.{_speaker_block}.res2.{_speaker_branch}.{_speaker_kind}"
            )
_QWEN3TTS_SPEAKER_TOP_TENSORS = tuple(_QWEN3TTS_SPEAKER_TOP_TENSOR_LIST)

_COMMON_REQUIRED_VISION_METADATA = (
    "clip.has_vision_encoder",
    "clip.vision.embedding_length",
    "clip.vision.feed_forward_length",
    "clip.vision.block_count",
    "clip.vision.projection_dim",
    "clip.vision.attention.head_count",
    "clip.vision.attention.layer_norm_epsilon",
    "clip.vision.image_size",
    "clip.vision.patch_size",
    "clip.vision.image_mean",
    "clip.vision.image_std",
)

_COMMON_REQUIRED_AUDIO_METADATA = (
    "clip.has_audio_encoder",
    "clip.audio.projector_type",
    "clip.audio.embedding_length",
    "clip.audio.feed_forward_length",
    "clip.audio.block_count",
    "clip.audio.projection_dim",
    "clip.audio.attention.head_count",
    "clip.audio.attention.layer_norm_epsilon",
    "clip.audio.num_mel_bins",
)
_STANDALONE_AUDIO_METADATA = (
    "clip.has_audio_encoder",
    "clip.audio.embedding_length",
    "clip.audio.feed_forward_length",
    "clip.audio.block_count",
    "clip.audio.projection_dim",
    "clip.audio.attention.head_count",
    "clip.audio.attention.layer_norm_epsilon",
    "clip.audio.num_mel_bins",
)
_WHISPER_AUDIO_BLOCK_SUFFIXES = (
    "ln1.weight",
    "ln1.bias",
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_v.weight",
    "attn_v.bias",
    "attn_out.weight",
    "attn_out.bias",
    "ln2.weight",
    "ln2.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
)
_WHISPER_AUDIO_TOP = (
    "a.conv1d.1.weight",
    "a.conv1d.1.bias",
    "a.conv1d.2.weight",
    "a.conv1d.2.bias",
    "a.position_embd.weight",
    "a.post_ln.weight",
    "a.post_ln.bias",
)

_STANDALONE_AUDIO_METADATA = (
    "clip.has_audio_encoder",
    "clip.audio.embedding_length",
    "clip.audio.feed_forward_length",
    "clip.audio.block_count",
    "clip.audio.projection_dim",
    "clip.audio.attention.head_count",
    "clip.audio.attention.layer_norm_epsilon",
    "clip.audio.num_mel_bins",
)

_WHISPER_AUDIO_BLOCK_SUFFIXES = (
    "ln1.weight",
    "ln1.bias",
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_v.weight",
    "attn_v.bias",
    "attn_out.weight",
    "attn_out.bias",
    "ln2.weight",
    "ln2.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
)

_WHISPER_AUDIO_TOP = (
    "a.conv1d.1.weight",
    "a.conv1d.1.bias",
    "a.conv1d.2.weight",
    "a.conv1d.2.bias",
    "a.position_embd.weight",
    "a.post_ln.weight",
    "a.post_ln.bias",
)

_LFM2A_BLOCK_SUFFIXES = (
    "ffn_norm.weight",
    "ffn_norm.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
    "ln1.weight",
    "ln1.bias",
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_k.bias",
    "attn_v.weight",
    "attn_v.bias",
    "attn_out.weight",
    "attn_out.bias",
    "linear_pos.weight",
    "pos_bias_u",
    "pos_bias_v",
    "norm_conv.weight",
    "norm_conv.bias",
    "conv_pw1.weight",
    "conv_pw1.bias",
    "conv_dw.weight",
    "conv_dw.bias",
    "conv_norm.weight",
    "conv_norm.bias",
    "conv_pw2.weight",
    "conv_pw2.bias",
    "ffn_norm_1.weight",
    "ffn_norm_1.bias",
    "ffn_up_1.weight",
    "ffn_up_1.bias",
    "ffn_down_1.weight",
    "ffn_down_1.bias",
    "ln2.weight",
    "ln2.bias",
)

_PARAKEET_BLOCK_SUFFIXES = (
    "ffn_norm.weight",
    "ffn_norm.bias",
    "ffn_up.weight",
    "ffn_down.weight",
    "ln1.weight",
    "ln1.bias",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_out.weight",
    "linear_pos.weight",
    "pos_bias_u",
    "pos_bias_v",
    "norm_conv.weight",
    "norm_conv.bias",
    "conv_pw1.weight",
    "conv_dw.weight",
    "conv_norm.weight",
    "conv_norm.bias",
    "conv_norm_mean",
    "conv_norm_var",
    "conv_pw2.weight",
    "ffn_norm_1.weight",
    "ffn_norm_1.bias",
    "ffn_up_1.weight",
    "ffn_down_1.weight",
    "ln2.weight",
    "ln2.bias",
)

_GRANITE_SPEECH_BLOCK_SUFFIXES = (
    "ffn_norm.weight",
    "ffn_norm.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
    "ln1.weight",
    "ln1.bias",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_out.weight",
    "attn_out.bias",
    "attn_rel_pos_emb",
    "norm_conv.weight",
    "norm_conv.bias",
    "conv_pw1.weight",
    "conv_pw1.bias",
    "conv_dw.weight",
    "conv_norm.weight",
    "conv_norm.bias",
    "conv_pw2.weight",
    "conv_pw2.bias",
    "ffn_norm_1.weight",
    "ffn_norm_1.bias",
    "ffn_up_1.weight",
    "ffn_up_1.bias",
    "ffn_down_1.weight",
    "ffn_down_1.bias",
    "ln2.weight",
    "ln2.bias",
)

_MIMO_LOCAL_PATTERN = (
    r"mm\.a\.local_blk\.\d+\.(?:attn_q|attn_k|attn_v)\.(?:weight|bias)",
    r"mm\.a\.local_blk\.\d+\.attn_out\.weight",
    r"mm\.a\.local_blk\.\d+\.(?:ffn_gate|ffn_up|ffn_down|ln1|ln2)\.weight",
)

_POCKETTTS_BLOCK_SUFFIXES = (
    "ln1.weight",
    "ln1.bias",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_out.weight",
    "ls1.weight",
    "ln2.weight",
    "ln2.bias",
    "ffn_up.weight",
    "ffn_down.weight",
    "ls2.weight",
)


def _deferred(
    projector_type: str,
    enum_name: str,
    modalities: frozenset[MMProjModality],
    reason: str,
    *,
    target_architectures: frozenset[str] = frozenset(),
) -> ProjectorSpec:
    return ProjectorSpec(
        projector_type=projector_type,
        enum_name=enum_name,
        modalities=modalities,
        target_architectures=target_architectures,
        reason=reason,
    )


def _rejected(
    projector_type: str,
    enum_name: str,
    modalities: frozenset[MMProjModality],
    reason: str,
) -> ProjectorSpec:
    return ProjectorSpec(
        projector_type=projector_type,
        enum_name=enum_name,
        modalities=modalities,
        metadata=Support.REJECTED,
        tensor_map=Support.REJECTED,
        graph=Support.REJECTED,
        runtime=Support.REJECTED,
        reason=reason,
    )


_CLIP_TOP = (
    "v.class_embd",
    "v.patch_embd.weight",
    "v.position_embd.weight",
    "v.pre_ln.weight",
    "v.pre_ln.bias",
)
_SIGLIP_TOP = (
    "v.patch_embd.weight",
    "v.patch_embd.bias",
    "v.position_embd.weight",
    "v.post_ln.weight",
    "v.post_ln.bias",
)
_GENERIC_BLOCK_SUFFIXES = (
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_k.bias",
    "attn_v.weight",
    "attn_v.bias",
    "attn_out.weight",
    "attn_out.bias",
    "ln1.weight",
    "ln1.bias",
    "ln2.weight",
    "ln2.bias",
    "ffn_down.weight",
    "ffn_down.bias",
    "ffn_up.weight",
    "ffn_up.bias",
)
_FUSED_GELU_BLOCK_SUFFIXES = (
    "ln1.weight",
    "ln1.bias",
    "ln2.weight",
    "ln2.bias",
    "attn_qkv.weight",
    "attn_qkv.bias",
    "attn_out.weight",
    "attn_out.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
)
_FUSED_GATED_RMS_BLOCK_SUFFIXES = (
    "ln1.weight",
    "ln2.weight",
    "attn_qkv.weight",
    "attn_qkv.bias",
    "attn_out.weight",
    "attn_out.bias",
    "ffn_gate.weight",
    "ffn_gate.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
)
_STEP3VL_BLOCK_SUFFIXES = (
    *_FUSED_GELU_BLOCK_SUFFIXES,
    "ls1.weight",
    "ls2.weight",
)
_MIMOVL_BLOCK_SUFFIXES = _FUSED_GATED_RMS_BLOCK_SUFFIXES
_LDP_TOP = [
    "mm.model.mlp.1.weight",
    "mm.model.mlp.1.bias",
    "mm.model.mlp.3.weight",
    "mm.model.mlp.3.bias",
]
for _block in (1, 2):
    _prefix = f"mm.model.mb_block.{_block}.block"
    _LDP_TOP.extend(
        (
            f"{_prefix}.0.0.weight",
            f"{_prefix}.0.1.weight",
            f"{_prefix}.0.1.bias",
            f"{_prefix}.1.fc1.weight",
            f"{_prefix}.1.fc1.bias",
            f"{_prefix}.1.fc2.weight",
            f"{_prefix}.1.fc2.bias",
            f"{_prefix}.2.0.weight",
            f"{_prefix}.2.1.weight",
            f"{_prefix}.2.1.bias",
        )
    )

_DOTS_DENSE_SUFFIXES = (
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)
_DOTS_EXPERT_SUFFIXES = (
    "ffn_gate_inp.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "exp_probs_b.weight",
)
_DOTS_BLOCK_VARIANT = BlockTensorVariantSpec(
    trigger_suffix="ffn_gate_exps.weight",
    default_suffixes=_DOTS_DENSE_SUFFIXES,
    triggered_suffixes=_DOTS_EXPERT_SUFFIXES,
)
_DOTS_VISION_COMMON_SUFFIXES = (
    "attn_qkv.weight",
    "attn_out.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "ln1.weight",
    "ln2.weight",
)
_DOTS_VISION_TOP = (
    "v.patch_embd.weight",
    "v.patch_embd.bias",
    "v.pre_ln.weight",
    "mm.post_norm.weight",
    "mm.input_norm.weight",
    "mm.input_norm.bias",
    "mm.0.weight",
    "mm.0.bias",
    "mm.2.weight",
    "mm.2.bias",
)
_DOTS_AUDIO_SUFFIXES = (
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_v.weight",
    "attn_v.bias",
    "attn_out.weight",
    "attn_out.bias",
    "ffn_gate.weight",
    "ffn_gate.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
    "ln1.weight",
    "ln2.weight",
)
_DOTS_AUDIO_TOP = (
    "a.conv2d.1.weight",
    "a.conv2d.1.bias",
    "a.conv2d.2.weight",
    "a.conv2d.2.bias",
    "a.conv2d.3.weight",
    "a.conv2d.3.bias",
    "a.conv_out.weight",
    "a.post_ln.weight",
    "mm.a.norm_pre.weight",
    "mm.a.norm_pre.bias",
    "mm.a.mlp.1.weight",
    "mm.a.mlp.1.bias",
    "mm.a.mlp.3.weight",
    "mm.a.mlp.3.bias",
)
_DOTS_AUDIO_COMPANION = CompanionTensorSpec(
    modality=MMProjModality.AUDIO,
    projector_type="dots3note_a",
    required_metadata=(*_COMMON_REQUIRED_AUDIO_METADATA, "clip.use_silu"),
    required_top_tensors=_DOTS_AUDIO_TOP,
    block_prefix="a.blk.",
    block_suffixes=_DOTS_AUDIO_SUFFIXES,
)
_DOTS_VISION_COMPANION = CompanionTensorSpec(
    modality=MMProjModality.VISION,
    projector_type="dots3note_v",
    required_metadata=(
        *_COMMON_REQUIRED_VISION_METADATA,
        "clip.vision.projector_type",
        "clip.vision.spatial_merge_size",
        "clip.vision.image_min_pixels",
        "clip.vision.image_max_pixels",
        "clip.vision.expert_used_count",
        "clip.use_silu",
    ),
    required_top_tensors=_DOTS_VISION_TOP,
    block_prefix="v.blk.",
    block_suffixes=_DOTS_VISION_COMMON_SUFFIXES,
    block_variant=_DOTS_BLOCK_VARIANT,
)
_DEEPSEEK_SAM_SUFFIXES = (
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.out.weight",
    "attn.out.bias",
    "pre_ln.weight",
    "pre_ln.bias",
    "post_ln.weight",
    "post_ln.bias",
    "attn.pos_h.weight",
    "attn.pos_w.weight",
    "mlp.lin1.weight",
    "mlp.lin1.bias",
    "mlp.lin2.weight",
    "mlp.lin2.bias",
)
_DEEPSEEK_SAM_TOP = (
    "v.sam.pos_embd.weight",
    "v.sam.patch_embd.weight",
    "v.sam.patch_embd.bias",
    "v.sam.neck.0.weight",
    "v.sam.neck.1.weight",
    "v.sam.neck.1.bias",
    "v.sam.neck.2.weight",
    "v.sam.neck.3.weight",
    "v.sam.neck.3.bias",
    "v.sam.net_2.weight",
    "v.sam.net_3.weight",
)
_GRANITE4_PROJECTOR_SUFFIXES = (
    "img_pos",
    "query",
    "linear.weight",
    "linear.bias",
    "norm.weight",
    "norm.bias",
    "post_norm.weight",
    "post_norm.bias",
    "self_attn_q.weight",
    "self_attn_q.bias",
    "self_attn_k.weight",
    "self_attn_k.bias",
    "self_attn_v.weight",
    "self_attn_v.bias",
    "self_attn_out.weight",
    "self_attn_out.bias",
    "self_attn_norm.weight",
    "self_attn_norm.bias",
    "cross_attn_q.weight",
    "cross_attn_q.bias",
    "cross_attn_k.weight",
    "cross_attn_k.bias",
    "cross_attn_v.weight",
    "cross_attn_v.bias",
    "cross_attn_out.weight",
    "cross_attn_out.bias",
    "cross_attn_norm.weight",
    "cross_attn_norm.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
    "ffn_norm.weight",
    "ffn_norm.bias",
)

_SPECS: tuple[ProjectorSpec, ...] = (
    ProjectorSpec(
        "mlp",
        "PROJECTOR_TYPE_MLP",
        _VISION_BASE,
        target_architectures=frozenset({"llama"}),
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason="Graph-supported; paired LLaVA runtime parity remains deferred.",
        builder="generic_projector",
        required_metadata=_COMMON_REQUIRED_VISION_METADATA,
        required_top_tensors=(
            *_CLIP_TOP,
            "mm.0.weight",
            "mm.0.bias",
        ),
        optional_top_tensors=("mm.2.weight", "mm.2.bias"),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(("v.", MMProjTensorRole.ENCODER), ("mm.", MMProjTensorRole.PROJECTOR)),
        real_artifact_ids=("llava-llama3-8b-mlp-f16",),
    ),
    ProjectorSpec(
        "ldp",
        "PROJECTOR_TYPE_LDP",
        _VISION_BASE,
        target_architectures=frozenset({"llama"}),
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason="Graph-supported; paired MobileVLM runtime parity remains deferred.",
        builder="generic_projector",
        required_metadata=_COMMON_REQUIRED_VISION_METADATA,
        required_top_tensors=(*_CLIP_TOP, *_LDP_TOP),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(("v.", MMProjTensorRole.ENCODER), ("mm.", MMProjTensorRole.PROJECTOR)),
        real_artifact_ids=("mobilevlm-1.7b-ldp-f16",),
    ),
    ProjectorSpec(
        "ldpv2",
        "PROJECTOR_TYPE_LDPV2",
        _VISION_BASE,
        target_architectures=frozenset({"llama"}),
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason="Graph-supported; paired MobileVLM-V2 runtime parity remains deferred.",
        builder="generic_projector",
        required_metadata=_COMMON_REQUIRED_VISION_METADATA,
        required_top_tensors=(
            *_CLIP_TOP,
            "mm.model.mlp.0.weight",
            "mm.model.mlp.0.bias",
            "mm.model.mlp.2.weight",
            "mm.model.mlp.2.bias",
            "mm.model.peg.0.weight",
            "mm.model.peg.0.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(("v.", MMProjTensorRole.ENCODER), ("mm.", MMProjTensorRole.PROJECTOR)),
        real_artifact_ids=("mobilevlm-v2-1.7b-ldpv2-f16",),
    ),
    ProjectorSpec(
        "resampler",
        "PROJECTOR_TYPE_MINICPMV",
        _VISION_BASE,
        target_architectures=frozenset({"minicpm"}),
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=(
            "Metadata and exact tensor mapping are supported, including learned query "
            "positions. Graph construction is deferred because the real MiniCPM-V2 "
            "processor emits variable patch-aligned image heights and widths while the "
            "current vision graph fixes a 448x448 input and 32x32 position grid."
        ),
        builder="generic_projector",
        required_metadata=_COMMON_REQUIRED_VISION_METADATA,
        required_top_tensors=(
            *_SIGLIP_TOP,
            "resampler.query",
            "resampler.pos_embed",
            "resampler.proj.weight",
            "resampler.kv.weight",
            "resampler.attn.q.weight",
            "resampler.attn.k.weight",
            "resampler.attn.v.weight",
            "resampler.attn.q.bias",
            "resampler.attn.k.bias",
            "resampler.attn.v.bias",
            "resampler.attn.out.weight",
            "resampler.attn.out.bias",
            "resampler.ln_q.weight",
            "resampler.ln_q.bias",
            "resampler.ln_kv.weight",
            "resampler.ln_kv.bias",
            "resampler.ln_post.weight",
            "resampler.ln_post.bias",
        ),
        optional_top_tensors=("resampler.pos_embed_k",),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("resampler.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("minicpm-v2-resampler-f16",),
    ),
    ProjectorSpec(
        "adapter",
        "PROJECTOR_TYPE_GLM_EDGE",
        _VISION_BASE,
        target_architectures=frozenset({"chatglm"}),
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason="Graph-supported; paired GLM-Edge runtime parity remains deferred.",
        builder="generic_projector",
        required_metadata=_COMMON_REQUIRED_VISION_METADATA,
        required_top_tensors=(
            *_SIGLIP_TOP,
            "adapter.boi",
            "adapter.eoi",
            "adapter.conv.weight",
            "adapter.conv.bias",
            "adapter.linear.linear.weight",
            "adapter.linear.norm1.weight",
            "adapter.linear.norm1.bias",
            "adapter.linear.dense_h_to_4h.weight",
            "adapter.linear.gate.weight",
            "adapter.linear.dense_4h_to_h.weight",
        ),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("adapter.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("glm-edge-v-2b-adapter-f16",),
    ),
    ProjectorSpec(
        projector_type="qwen2vl_merger",
        enum_name="PROJECTOR_TYPE_QWEN2VL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"qwen2vl"}),
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Metadata, exact tensor closure, fused projector transforms, graph "
            "construction, and processor boundary contracts are covered, but no "
            "downstream multimodal runtime execution is claimed."
        ),
        builder="qwen_vl",
        required_metadata=(*_COMMON_REQUIRED_VISION_METADATA, "clip.projector_type"),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_QWEN2VL_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("qwen2-vl-2b-f16",),
    ),
    ProjectorSpec(
        projector_type="qwen2.5vl_merger",
        enum_name="PROJECTOR_TYPE_QWEN25VL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"qwen2vl"}),
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Metadata, exact tensor closure, fused projector transforms, window "
            "schedule, graph construction, and processor boundary contracts are "
            "covered, but no downstream multimodal runtime execution is claimed."
        ),
        builder="qwen_vl",
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.n_wa_pattern",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.post_ln.weight",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_QWEN25VL_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("qwen25-vl-3b-f16",),
    ),
    ProjectorSpec(
        projector_type="qwen3vl_merger",
        enum_name="PROJECTOR_TYPE_QWEN3VL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Standalone vision and DeepStack merger graphs are exact. Paired runtime "
            "execution remains unvalidated because the decoder must consume four-section "
            "MRoPE positions and hidden_size * (1 + deepstack_layers) embeddings."
        ),
        sidecar_builder="qwen_glm_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.spatial_merge_size",
            "clip.vision.is_deepstack_layers",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_QWEN3VL_BLOCK_SUFFIXES,
        auxiliary_tensor_patterns=(r"v\.deepstack\.\d+\.(?:norm|fc1|fc2)\.(?:weight|bias)",),
        tensor_roles=(
            ("v.deepstack.", MMProjTensorRole.PROJECTOR),
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("qwen3-vl-projector-f16",),
    ),
    ProjectorSpec(
        projector_type="step3vl",
        enum_name="PROJECTOR_TYPE_STEP3VL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"qwen3"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact absolute-plus-axial position tower, layer scales, two "
            "convolutional downsamplers, processor inputs, and independent numerical "
            "parity are covered; paired multimodal runtime insertion is unvalidated."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.preproc_image_size",
            "clip.vision.projector.scale_factor",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.position_embd.weight",
            "v.pre_ln.weight",
            "v.pre_ln.bias",
            "mm.0.weight",
            "mm.0.bias",
            "mm.1.weight",
            "mm.1.bias",
            "mm.model.fc.weight",
        ),
        block_prefix="v.blk.",
        block_suffixes=_STEP3VL_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("step3-vl-10b-f16-header",),
        source_evidence_ids=("step3vl-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="gemma3",
        enum_name="PROJECTOR_TYPE_GEMMA3",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"gemma3"}),
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Metadata, exact tensor closure, graph construction, and component parity "
            "are covered, but downstream multimodal runtime support is not claimed."
        ),
        builder="gemma3",
        required_metadata=(*_COMMON_REQUIRED_VISION_METADATA, "clip.projector_type"),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.soft_emb_norm.weight",
            "mm.input_projection.weight",
        ),
        block_prefix="v.blk.",
        block_suffixes=_GEMMA3_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("gemma3-4b-f16",),
    ),
    ProjectorSpec(
        projector_type="gemma3nv",
        enum_name="PROJECTOR_TYPE_GEMMA3NV",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"gemma3n"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact MobileNetV5 encoder and Gemma3n soft projector graph are "
            "supported; downstream paired text/runtime execution remains unvalidated."
        ),
        sidecar_builder="core_vlm_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.vision.projector_type",
        ),
        required_top_tensors=_GEMMA3N_VISION_TOP,
        optional_top_tensors=_GEMMA3N_AUDIO_TOP,
        auxiliary_tensor_patterns=(
            r"v\.blk\.\d+\.\d+\..+",
            r"a\.blk\.\d+\..+",
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("gemma3n-e4b-f16",),
    ),
    ProjectorSpec(
        projector_type="gemma3na",
        enum_name="PROJECTOR_TYPE_GEMMA3NA",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"gemma3n"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact Gemma3n USM Conformer and audio projector graph are supported; "
            "llama.cpp does not dispatch gemma3na and downstream runtime remains unvalidated."
        ),
        sidecar_builder="core_vlm_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_AUDIO_METADATA,
            "clip.audio.projector_type",
        ),
        required_top_tensors=_GEMMA3N_AUDIO_TOP,
        optional_top_tensors=_GEMMA3N_VISION_TOP,
        block_prefix="a.blk.",
        block_suffixes=_GEMMA3N_AUDIO_BLOCK_SUFFIXES,
        auxiliary_tensor_patterns=(r"v\.blk\.\d+\.\d+\..+",),
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("gemma3n-e4b-f16",),
    ),
    ProjectorSpec(
        projector_type="gemma4v",
        enum_name="PROJECTOR_TYPE_GEMMA4V",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"gemma4"}),
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Metadata, tensor closure, graph construction, and independent component parity "
            "are covered, but the paired text target, processor, and deterministic package "
            "generation have no end-to-end runtime evidence."
        ),
        builder="gemma4",
        required_metadata=(*_COMMON_REQUIRED_VISION_METADATA, "clip.vision.projector_type"),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.position_embd.weight",
            "mm.input_projection.weight",
        ),
        block_prefix="v.blk.",
        block_suffixes=_GEMMA4V_BLOCK_SUFFIXES,
        companion_tensors=(
            CompanionTensorSpec(
                modality=MMProjModality.AUDIO,
                projector_type="gemma4a",
                required_metadata=_COMMON_REQUIRED_AUDIO_METADATA,
                required_top_tensors=_GEMMA4A_TOP_TENSORS,
                block_prefix="a.blk.",
                block_suffixes=_GEMMA4A_BLOCK_SUFFIXES,
            ),
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.input_projection.weight", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("gemma4-e2b-f16",),
    ),
    ProjectorSpec(
        projector_type="gemma4a",
        enum_name="PROJECTOR_TYPE_GEMMA4A",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"gemma4"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact clipped Gemma4 Conformer graph and tensor transforms are "
            "supported; downstream any-to-any runtime execution remains unvalidated."
        ),
        builder="gemma4",
        sidecar_builder="core_vlm_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=_COMMON_REQUIRED_AUDIO_METADATA,
        required_top_tensors=_GEMMA4A_TOP_TENSORS,
        block_prefix="a.blk.",
        block_suffixes=_GEMMA4A_BLOCK_SUFFIXES,
        companion_tensors=(
            CompanionTensorSpec(
                modality=MMProjModality.VISION,
                projector_type="gemma4v",
                required_metadata=_COMMON_REQUIRED_VISION_METADATA,
                required_top_tensors=(
                    "v.patch_embd.weight",
                    "v.position_embd.weight",
                    "mm.input_projection.weight",
                ),
                block_prefix="v.blk.",
                block_suffixes=_GEMMA4V_BLOCK_SUFFIXES,
            ),
        ),
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.input_projection.weight", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("gemma4-e2b-f16",),
    ),
    ProjectorSpec(
        projector_type="gemma4uv",
        enum_name="PROJECTOR_TYPE_GEMMA4UV",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"gemma4"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The encoder-free Gemma4 unified patch embedder and projection graph are "
            "supported; downstream unified multimodal runtime execution is unvalidated."
        ),
        builder="gemma4",
        sidecar_builder="core_vlm_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.vision.projector_type",
        ),
        required_top_tensors=_GEMMA4UV_TOP,
        companion_tensors=(
            CompanionTensorSpec(
                modality=MMProjModality.AUDIO,
                projector_type="gemma4ua",
                required_metadata=_COMMON_REQUIRED_AUDIO_METADATA,
                required_top_tensors=("mm.a.input_projection.weight",),
                block_prefix="a.blk.",
                block_suffixes=(),
            ),
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.input_projection.weight", MMProjTensorRole.PROJECTOR),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("gemma4-unified-12b-f16",),
    ),
    ProjectorSpec(
        projector_type="gemma4ua",
        enum_name="PROJECTOR_TYPE_GEMMA4UA",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"gemma4"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The encoder-free Gemma4 unified raw-waveform projection graph is "
            "supported; downstream unified multimodal runtime execution is unvalidated."
        ),
        builder="gemma4",
        sidecar_builder="core_vlm_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_AUDIO_METADATA,
            "clip.audio.projector_type",
        ),
        required_top_tensors=("mm.a.input_projection.weight",),
        companion_tensors=(
            CompanionTensorSpec(
                modality=MMProjModality.VISION,
                projector_type="gemma4uv",
                required_metadata=_COMMON_REQUIRED_VISION_METADATA,
                required_top_tensors=_GEMMA4UV_TOP,
                block_prefix="v.blk.",
                block_suffixes=(),
            ),
        ),
        tensor_roles=(
            ("mm.a.", MMProjTensorRole.PROJECTOR),
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.input_projection.weight", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("gemma4-unified-12b-f16",),
    ),
    _deferred(
        "phi4",
        "PROJECTOR_TYPE_PHI4",
        _VISION_BASE,
        "The Phi-4 vision projector exists for HF weights but has no pinned GGUF tensor closure.",
    ),
    ProjectorSpec(
        projector_type="idefics3",
        enum_name="PROJECTOR_TYPE_IDEFICS3",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"llama"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The SigLIP tower and Idefics3 pixel-shuffle projector are supported; "
            "processor tiling and downstream package execution remain unvalidated."
        ),
        sidecar_builder="core_vlm_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.use_gelu",
            "clip.vision.projector.scale_factor",
        ),
        required_top_tensors=(
            *_SIGLIP_TOP,
            "mm.model.fc.weight",
        ),
        block_prefix="v.blk.",
        block_suffixes=_IDEFICS3_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("smolvlm-256m-idefics3-f16",),
    ),
    ProjectorSpec(
        projector_type="pixtral",
        enum_name="PROJECTOR_TYPE_PIXTRAL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"llama"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The dynamic Pixtral tower, MLP, and image-row break contract are "
            "supported; downstream dynamic media-shape package execution is unvalidated."
        ),
        sidecar_builder="core_vlm_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.use_silu",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.pre_ln.weight",
            "v.token_embd.img_break",
            "mm.1.weight",
            "mm.1.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_PIXTRAL_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.token_embd.img_break", MMProjTensorRole.PROJECTOR),
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("pixtral-12b-f16",),
    ),
    ProjectorSpec(
        projector_type="ultravox",
        enum_name="PROJECTOR_TYPE_ULTRAVOX",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"llama"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact Whisper encoder, frame stacking, swapped-SwiGLU projector, "
            "processor ABI, and independent numerical parity are covered; paired "
            "text-runtime insertion remains unvalidated."
        ),
        sidecar_builder="audio_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(
            *_STANDALONE_AUDIO_METADATA,
            "clip.projector_type",
            "clip.audio.projector.stack_factor",
        ),
        required_top_tensors=(
            *_WHISPER_AUDIO_TOP,
            "mm.a.mlp.1.weight",
            "mm.a.mlp.2.weight",
            "mm.a.norm_pre.weight",
            "mm.a.norm_mid.weight",
        ),
        block_prefix="a.blk.",
        block_suffixes=_WHISPER_AUDIO_BLOCK_SUFFIXES,
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("ultravox-v0.5-f16",),
    ),
    ProjectorSpec(
        projector_type="internvl",
        enum_name="PROJECTOR_TYPE_INTERNVL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"qwen2"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The CLS-last InternViT tower, layer scales, and pixel-shuffle MLP are "
            "supported; processor tiling and downstream package execution are unvalidated."
        ),
        sidecar_builder="core_vlm_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.use_gelu",
            "clip.vision.projector.scale_factor",
        ),
        required_top_tensors=(
            "v.class_embd",
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "mm.model.mlp.0.weight",
            "mm.model.mlp.0.bias",
            "mm.model.mlp.1.weight",
            "mm.model.mlp.1.bias",
            "mm.model.mlp.3.weight",
            "mm.model.mlp.3.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_INTERNVL_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("internvl25-1b-f16",),
    ),
    ProjectorSpec(
        projector_type="llama4",
        enum_name="PROJECTOR_TYPE_LLAMA4",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"llama4"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact standalone Llama4 vision/projector graph is supported; the "
            "text architecture and every complete public text artifact remain unavailable "
            "within the 16 GiB evidence budget, so paired runtime is unvalidated."
        ),
        sidecar_builder="core_vlm_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.use_gelu",
            "clip.vision.projector.scale_factor",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.class_embd",
            "v.position_embd.weight",
            "v.pre_ln.weight",
            "v.pre_ln.bias",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.model.mlp.1.weight",
            "mm.model.mlp.2.weight",
            "mm.model.fc.weight",
        ),
        block_prefix="v.blk.",
        block_suffixes=_LLAMA4_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("llama4-scout-f16",),
    ),
    ProjectorSpec(
        projector_type="qwen2a",
        enum_name="PROJECTOR_TYPE_QWEN2A",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"qwen2"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The standalone Whisper encoder, two-frame average pool, and affine audio "
            "projector are exact; paired audio-token runtime execution remains unvalidated."
        ),
        sidecar_builder="qwen_glm_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(*_COMMON_AUDIO_BODY_METADATA, "clip.projector_type"),
        required_top_tensors=_QWEN2A_TOP_TENSORS,
        block_prefix="a.blk.",
        block_suffixes=_WHISPER_AUDIO_BLOCK_SUFFIXES,
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("qwen2-audio-projector-f16",),
    ),
    ProjectorSpec(
        projector_type="qwen3a",
        enum_name="PROJECTOR_TYPE_QWEN3A",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"qwen3vl", "qwen3vlmoe"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The standalone 100-frame chunked Conv2D encoder and two-layer audio "
            "projector are exact; paired MRoPE audio-token runtime execution remains "
            "unvalidated."
        ),
        sidecar_builder="qwen_glm_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(
            *_COMMON_AUDIO_BODY_METADATA,
            "clip.audio.projector_type",
        ),
        required_top_tensors=_QWEN3A_TOP_TENSORS,
        block_prefix="a.blk.",
        block_suffixes=_QWEN3A_BLOCK_SUFFIXES,
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("qwen3-audio-projector-bf16",),
    ),
    ProjectorSpec(
        projector_type="glma",
        enum_name="PROJECTOR_TYPE_GLMA",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"llama"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The serialized legacy Whisper, frame-stack, MLP, and BOI/EOI graph is "
            "implemented. Current GLM-ASR cannot provide evidence: its converter misses "
            "the checkpoint architecture and merge_factor and its encoder uses partial "
            "RoPE instead of this additive-position topology."
        ),
        sidecar_builder="qwen_glm_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(
            *_COMMON_AUDIO_BODY_METADATA,
            "clip.projector_type",
            "clip.audio.projector.stack_factor",
        ),
        required_top_tensors=_GLMA_TOP_TENSORS,
        block_prefix="a.blk.",
        block_suffixes=_WHISPER_AUDIO_BLOCK_SUFFIXES,
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
            ("v.boi", MMProjTensorRole.PROJECTOR),
            ("v.eoi", MMProjTensorRole.PROJECTOR),
        ),
        source_evidence_ids=("glma-converter-checkpoint-drift",),
    ),
    ProjectorSpec(
        projector_type="qwen2.5o",
        enum_name="PROJECTOR_TYPE_QWEN25O",
        modalities=_VISION_BASE | _AUDIO_BASE,
        target_architectures=frozenset({"qwen2vl"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The legacy selector is resolved per modality into separate "
            "qwen2.5vl_merger and qwen2a components; no generic qwen2.5o graph exists. "
            "Paired multimodal runtime orchestration remains unvalidated."
        ),
        sidecar_builder="qwen_glm_projector",
        model_roles=(
            MMProjModelRole.VISION_ENCODER,
            MMProjModelRole.AUDIO_ENCODER,
        ),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            *_COMMON_AUDIO_BODY_METADATA,
            "clip.projector_type",
            "clip.vision.n_wa_pattern",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.post_ln.weight",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_QWEN25VL_BLOCK_SUFFIXES,
        companion_tensors=(
            CompanionTensorSpec(
                modality=MMProjModality.AUDIO,
                projector_type="qwen2.5o",
                required_metadata=_COMMON_AUDIO_BODY_METADATA,
                required_top_tensors=_QWEN2A_TOP_TENSORS,
                block_prefix="a.blk.",
                block_suffixes=_WHISPER_AUDIO_BLOCK_SUFFIXES,
            ),
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("qwen25-omni-projector-f16",),
    ),
    ProjectorSpec(
        projector_type="voxtral",
        enum_name="PROJECTOR_TYPE_VOXTRAL",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"llama"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact Whisper encoder, average pooling, frame stacking, GELU "
            "projector, processor ABI, and independent numerical parity are covered; "
            "the published packed sidecar fails closed and paired runtime is unvalidated."
        ),
        sidecar_builder="audio_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(
            *_STANDALONE_AUDIO_METADATA,
            "clip.projector_type",
            "clip.audio.projector.stack_factor",
        ),
        required_top_tensors=(
            *_WHISPER_AUDIO_TOP,
            "mm.a.mlp.1.weight",
            "mm.a.mlp.2.weight",
        ),
        block_prefix="a.blk.",
        block_suffixes=_WHISPER_AUDIO_BLOCK_SUFFIXES,
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        source_evidence_ids=("voxtral-mini-3b-source-and-q8-availability",),
    ),
    ProjectorSpec(
        projector_type="meralion",
        enum_name="PROJECTOR_TYPE_MERALION",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"gemma2"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact Whisper encoder, stack-before-normalization gated adapter, "
            "single-chunk processor ABI, tensor closure, and independent numerical "
            "parity are covered. Paired text extraction and multi-chunk runtime "
            "equivalence remain explicitly unvalidated."
        ),
        sidecar_builder="audio_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(
            *_STANDALONE_AUDIO_METADATA,
            "clip.projector_type",
            "clip.audio.projector.stack_factor",
        ),
        required_top_tensors=(
            *_WHISPER_AUDIO_TOP,
            "mm.a.norm_pre.weight",
            "mm.a.norm_pre.bias",
            *(f"mm.a.mlp.{index}.{kind}" for index in range(4) for kind in ("weight", "bias")),
        ),
        block_prefix="a.blk.",
        block_suffixes=_WHISPER_AUDIO_BLOCK_SUFFIXES,
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        source_evidence_ids=("meralion-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="musicflamingo",
        enum_name="PROJECTOR_TYPE_MUSIC_FLAMINGO",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"qwen2"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact Whisper encoder, average pooling, biased GELU projector, "
            "processor ABI, and independent numerical parity are covered; paired "
            "text-runtime insertion remains unvalidated."
        ),
        sidecar_builder="audio_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(*_STANDALONE_AUDIO_METADATA, "clip.projector_type"),
        required_top_tensors=(
            *_WHISPER_AUDIO_TOP,
            "mm.a.mlp.1.weight",
            "mm.a.mlp.1.bias",
            "mm.a.mlp.2.weight",
            "mm.a.mlp.2.bias",
        ),
        block_prefix="a.blk.",
        block_suffixes=_WHISPER_AUDIO_BLOCK_SUFFIXES,
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("music-flamingo-bf16",),
    ),
    ProjectorSpec(
        projector_type="lfm2",
        enum_name="PROJECTOR_TYPE_LFM2",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"lfm2"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact dynamic SigLIP position resize, pixel-unshuffle projector, "
            "processor-native NaFlex inputs, tensor closure, and independent parity "
            "are covered; paired multimodal runtime insertion is unvalidated."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.projector.scale_factor",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.input_norm.weight",
            "mm.input_norm.bias",
            "mm.1.weight",
            "mm.1.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("lfm2-vl-1-6b-f16-header",),
        source_evidence_ids=("lfm2-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="kimivl",
        enum_name="PROJECTOR_TYPE_KIMIVL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"deepseek2"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact learned-position, adjacent-pair 2D-RoPE tower, spatial "
            "merger, tensor closure, and independent parity are covered; upstream "
            "processor bounds and paired runtime remain explicitly unvalidated."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.projector.scale_factor",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.input_norm.weight",
            "mm.input_norm.bias",
            "mm.1.weight",
            "mm.1.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("kimi-vl-a3b-f16-header",),
        source_evidence_ids=("kimivl-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="paddleocr",
        enum_name="PROJECTOR_TYPE_PADDLEOCR",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"paddleocr"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Exact standalone vision graph export is supported; the paired PaddleOCR "
            "text architecture and downstream multimodal runtime remain unvalidated."
        ),
        sidecar_builder="ocr_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.image_min_pixels",
            "clip.vision.image_max_pixels",
            "clip.use_gelu",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.input_norm.weight",
            "mm.input_norm.bias",
            "mm.1.weight",
            "mm.1.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=(
            "attn_q.weight",
            "attn_q.bias",
            "attn_k.weight",
            "attn_k.bias",
            "attn_v.weight",
            "attn_v.bias",
            "attn_out.weight",
            "attn_out.bias",
            "ffn_up.weight",
            "ffn_up.bias",
            "ffn_down.weight",
            "ffn_down.bias",
            "ln1.weight",
            "ln1.bias",
            "ln2.weight",
            "ln2.bias",
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("paddleocr-vl-1.6-bf16",),
    ),
    ProjectorSpec(
        projector_type="lightonocr",
        enum_name="PROJECTOR_TYPE_LIGHTONOCR",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"qwen3"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Exact standalone Pixtral-style vision graph export is supported; paired "
            "Qwen3 package generation and downstream multimodal runtime remain unvalidated."
        ),
        sidecar_builder="ocr_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.spatial_merge_size",
            "clip.use_silu",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.pre_ln.weight",
            "mm.input_norm.weight",
            "mm.patch_merger.weight",
            "mm.1.weight",
            "mm.2.weight",
        ),
        optional_top_tensors=("mm.1.bias", "mm.2.bias"),
        block_prefix="v.blk.",
        block_suffixes=(
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_out.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "ln1.weight",
            "ln2.weight",
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("lightonocr-1b-1025-f16",),
    ),
    ProjectorSpec(
        projector_type="cogvlm",
        enum_name="PROJECTOR_TYPE_COGVLM",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"cogvlm"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact appended-CLS vision tower, split-to-fused QKV transform, "
            "post-normalized blocks, gated projector, and BOI/EOI token order are "
            "covered; the paired visual-expert text runtime remains unvalidated."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(*_COMMON_REQUIRED_VISION_METADATA, "clip.projector_type"),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.class_embd",
            "v.position_embd.weight",
            "mm.model.fc.weight",
            "mm.post_fc_norm.weight",
            "mm.post_fc_norm.bias",
            "mm.up.weight",
            "mm.gate.weight",
            "mm.down.weight",
            "v.boi",
            "v.eoi",
        ),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("cogvlm-chat-v1.1-f16-header",),
        source_evidence_ids=("cogvlm-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="janus_pro",
        enum_name="PROJECTOR_TYPE_JANUS_PRO",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"llama"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact fixed SigLIP tower, two-layer erf-GELU aligner, processor "
            "geometry, tensor closure, and independent parity are covered; paired "
            "multimodal runtime insertion is unvalidated."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(*_COMMON_REQUIRED_VISION_METADATA, "clip.projector_type"),
        required_top_tensors=(
            *_SIGLIP_TOP,
            "mm.0.weight",
            "mm.0.bias",
            "mm.1.weight",
            "mm.1.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("janus-pro-1b-f16-header",),
        source_evidence_ids=("janus-pro-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="dots_ocr",
        enum_name="PROJECTOR_TYPE_DOTS_OCR",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"qwen2"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Exact standalone packed vision graph export is supported; paired Qwen2 "
            "package generation and downstream multimodal runtime remain unvalidated."
        ),
        sidecar_builder="ocr_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.projector.scale_factor",
            "clip.vision.image_min_pixels",
            "clip.vision.image_max_pixels",
            "clip.use_silu",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.pre_ln.weight",
            "mm.post_norm.weight",
            "mm.input_norm.weight",
            "mm.input_norm.bias",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=(
            "attn_qkv.weight",
            "attn_out.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "ln1.weight",
            "ln2.weight",
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("dots-ocr-f16",),
    ),
    ProjectorSpec(
        projector_type="dots3note_v",
        enum_name="PROJECTOR_TYPE_DOTS3NOTE_V",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"dots3note"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Exact standalone progressive-MoE vision graph export is supported; the "
            "paired Dots3Note text architecture and downstream runtime remain unvalidated."
        ),
        sidecar_builder="ocr_projector",
        model_roles=(
            MMProjModelRole.VISION_ENCODER,
            MMProjModelRole.AUDIO_ENCODER,
        ),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.vision.projector_type",
            "clip.vision.spatial_merge_size",
            "clip.vision.image_min_pixels",
            "clip.vision.image_max_pixels",
            "clip.vision.expert_used_count",
            "clip.use_silu",
        ),
        required_top_tensors=_DOTS_VISION_TOP,
        block_prefix="v.blk.",
        block_suffixes=_DOTS_VISION_COMMON_SUFFIXES,
        block_variant=_DOTS_BLOCK_VARIANT,
        companion_tensors=(_DOTS_AUDIO_COMPANION,),
        tensor_roles=(
            ("mm.a.", MMProjTensorRole.PROJECTOR),
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
            ("v.", MMProjTensorRole.ENCODER),
        ),
        real_artifact_ids=("dots3note-prev-f16",),
    ),
    ProjectorSpec(
        projector_type="dots3note_a",
        enum_name="PROJECTOR_TYPE_DOTS3NOTE_A",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"dots3note"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Exact standalone mel-to-audio-feature graph export is supported; the paired "
            "Dots3Note text architecture and downstream runtime remain unvalidated."
        ),
        sidecar_builder="ocr_projector",
        model_roles=(
            MMProjModelRole.VISION_ENCODER,
            MMProjModelRole.AUDIO_ENCODER,
        ),
        required_metadata=(*_COMMON_REQUIRED_AUDIO_METADATA, "clip.use_silu"),
        required_top_tensors=_DOTS_AUDIO_TOP,
        block_prefix="a.blk.",
        block_suffixes=_DOTS_AUDIO_SUFFIXES,
        companion_tensors=(_DOTS_VISION_COMPANION,),
        tensor_roles=(
            ("mm.a.", MMProjTensorRole.PROJECTOR),
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
            ("v.", MMProjTensorRole.ENCODER),
        ),
        real_artifact_ids=("dots3note-prev-f16",),
    ),
    ProjectorSpec(
        projector_type="deepseekocr",
        enum_name="PROJECTOR_TYPE_DEEPSEEKOCR",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"deepseek2-ocr"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Exact standalone SAM plus CLIP graph export is supported; the paired "
            "DeepSeek2-OCR text architecture and downstream runtime remain unvalidated."
        ),
        sidecar_builder="ocr_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.sam.block_count",
            "clip.vision.sam.embedding_length",
            "clip.vision.sam.head_count",
            "clip.vision.window_size",
            "clip.use_gelu",
        ),
        required_top_tensors=(
            *_DEEPSEEK_SAM_TOP,
            "v.class_embd",
            "v.patch_embd.weight",
            "v.position_embd.weight",
            "v.pre_ln.weight",
            "v.pre_ln.bias",
            "v.image_newline",
            "v.view_seperator",
            "mm.model.fc.weight",
            "mm.model.fc.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=(
            "attn_qkv.weight",
            "attn_qkv.bias",
            "attn_out.weight",
            "attn_out.bias",
            "ffn_up.weight",
            "ffn_up.bias",
            "ffn_down.weight",
            "ffn_down.bias",
            "ln1.weight",
            "ln1.bias",
            "ln2.weight",
            "ln2.bias",
        ),
        indexed_tensors=(
            IndexedTensorSpec(
                "clip.vision.sam.block_count",
                "v.sam.blk.",
                _DEEPSEEK_SAM_SUFFIXES,
            ),
        ),
        tensor_roles=(
            ("v.sam.", MMProjTensorRole.ENCODER),
            ("v.blk.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
            ("v.image_", MMProjTensorRole.PROJECTOR),
            ("v.view_", MMProjTensorRole.PROJECTOR),
            ("v.", MMProjTensorRole.ENCODER),
        ),
        real_artifact_ids=("deepseek-ocr-bf16",),
    ),
    ProjectorSpec(
        projector_type="deepseekocr2",
        enum_name="PROJECTOR_TYPE_DEEPSEEKOCR2",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"deepseek2-ocr"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Exact standalone SAM plus dual-mask Qwen2 graph export is supported; the "
            "paired DeepSeek2-OCR text architecture and downstream runtime remain unvalidated."
        ),
        sidecar_builder="ocr_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.attention.head_count_kv",
            "clip.vision.sam.block_count",
            "clip.vision.sam.embedding_length",
            "clip.vision.sam.head_count",
            "clip.vision.window_size",
            "clip.use_gelu",
        ),
        required_top_tensors=(
            *_DEEPSEEK_SAM_TOP,
            "v.post_ln.weight",
            "v.resample_query_768.weight",
            "v.resample_query_1024.weight",
            "v.view_seperator",
            "mm.model.fc.weight",
            "mm.model.fc.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=(
            "attn_q.weight",
            "attn_q.bias",
            "attn_k.weight",
            "attn_k.bias",
            "attn_v.weight",
            "attn_v.bias",
            "attn_out.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "ln1.weight",
            "ln2.weight",
        ),
        indexed_tensors=(
            IndexedTensorSpec(
                "clip.vision.sam.block_count",
                "v.sam.blk.",
                _DEEPSEEK_SAM_SUFFIXES,
            ),
        ),
        tensor_roles=(
            ("v.sam.", MMProjTensorRole.ENCODER),
            ("v.blk.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
            ("v.resample_", MMProjTensorRole.PROJECTOR),
            ("v.view_", MMProjTensorRole.PROJECTOR),
            ("v.", MMProjTensorRole.ENCODER),
        ),
        real_artifact_ids=("deepseek-ocr2-bf16",),
    ),
    ProjectorSpec(
        projector_type="lfm2a",
        enum_name="PROJECTOR_TYPE_LFM2A",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"lfm2"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact Conformer subsampler, relative attention, GELU adapter, "
            "processor ABI, tensor closure, and independent parity are covered; "
            "paired text-runtime insertion remains unvalidated."
        ),
        sidecar_builder="audio_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(*_STANDALONE_AUDIO_METADATA, "clip.projector_type"),
        required_top_tensors=(
            "a.conv1d.0.weight",
            "a.conv1d.0.bias",
            "a.conv1d.2.weight",
            "a.conv1d.2.bias",
            "a.conv1d.3.weight",
            "a.conv1d.3.bias",
            "a.conv1d.5.weight",
            "a.conv1d.5.bias",
            "a.conv1d.6.weight",
            "a.conv1d.6.bias",
            "a.embd_to_logits.weight",
            "a.position_embd.weight",
            "a.position_embd_norm.weight",
            "a.pre_encode.out.weight",
            "a.pre_encode.out.bias",
            "mm.a.mlp.0.weight",
            "mm.a.mlp.0.bias",
            "mm.a.mlp.1.weight",
            "mm.a.mlp.1.bias",
            "mm.a.mlp.3.weight",
            "mm.a.mlp.3.bias",
        ),
        block_prefix="a.blk.",
        block_suffixes=_LFM2A_BLOCK_SUFFIXES,
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("lfm2.5-audio-1.5b-f16",),
    ),
    ProjectorSpec(
        projector_type="glm4v",
        enum_name="PROJECTOR_TYPE_GLM4V",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"glm4", "glm4moe"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The standalone RMS-normalized ViT, learned 2x2 downsampler, and gated "
            "projector are exact; paired four-section MRoPE runtime execution remains "
            "unvalidated."
        ),
        sidecar_builder="qwen_glm_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.patch_embd.bias",
            "v.post_ln.weight",
            "mm.model.fc.weight",
            "mm.up.weight",
            "mm.gate.weight",
            "mm.down.weight",
            "mm.post_norm.weight",
            "mm.post_norm.bias",
            "mm.patch_merger.weight",
            "mm.patch_merger.bias",
        ),
        optional_top_tensors=(
            "v.norm_embd.weight",
            "v.position_embd.weight",
        ),
        block_prefix="v.blk.",
        block_suffix_variants=(
            _GLM4V_BLOCK_SUFFIXES,
            _GLMOCR_BLOCK_SUFFIXES,
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("glm4v-projector-f16",),
    ),
    ProjectorSpec(
        projector_type="youtuvl",
        enum_name="PROJECTOR_TYPE_YOUTUVL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"deepseek2"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Exact standalone packed SigLIP2 vision graph export is supported; the paired "
            "DeepSeek2 text architecture and downstream runtime remain unvalidated."
        ),
        sidecar_builder="ocr_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.spatial_merge_size",
            "clip.vision.window_size",
            "clip.vision.wa_layer_indexes",
            "clip.use_gelu",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.input_norm.weight",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=(
            "attn_q.weight",
            "attn_q.bias",
            "attn_k.weight",
            "attn_k.bias",
            "attn_v.weight",
            "attn_v.bias",
            "attn_out.weight",
            "attn_out.bias",
            "ffn_up.weight",
            "ffn_up.bias",
            "ffn_down.weight",
            "ffn_down.bias",
            "ln1.weight",
            "ln1.bias",
            "ln2.weight",
            "ln2.bias",
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("youtu-vl-4b-bf16",),
    ),
    ProjectorSpec(
        projector_type="yasa2",
        enum_name="PROJECTOR_TYPE_YASA2",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"llama"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact ConvNeXtV2 stages, float32 GRN, pre-pool positions, fixed "
            "8x8 pooling, single-tile processor contract, and parity are covered; "
            "multi-tile composition and paired runtime remain unvalidated."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(*_COMMON_REQUIRED_VISION_METADATA, "clip.projector_type"),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.patch_ln.weight",
            "v.patch_ln.bias",
            "v.backbone_ln.weight",
            "v.backbone_ln.bias",
            "v.vision_pos_embed",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        auxiliary_tensor_patterns=(
            r"v\.stage\.\d+\.blk\.\d+\.(dw|ln|pw1|grn|pw2)\.(weight|bias)",
            r"v\.stage\.[1-9]\d*\.down\.(ln|conv)\.(weight|bias)",
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("yasa2-reka-edge-f16-header",),
        source_evidence_ids=("yasa2-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="kimik25",
        enum_name="PROJECTOR_TYPE_KIMIK25",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"deepseek2"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact bicubic learned-position tower, converter-permuted 2D RoPE, "
            "patch merger, tensor closure, and independent parity are covered; the "
            "oversized paired text runtime is not claimed."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.projector.scale_factor",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.input_norm.weight",
            "mm.input_norm.bias",
            "mm.1.weight",
            "mm.1.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_FUSED_GELU_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("kimi-k2-5-f16-header",),
        source_evidence_ids=("kimik25-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="nemotron_v2_vl",
        enum_name="PROJECTOR_TYPE_NEMOTRON_V2_VL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"nemotron_h"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact RADIO register-token tower, fixed position table, patch "
            "merge, RMSNorm, ReLU-squared projector, and parity are covered; the "
            "Parakeet companion and paired runtime remain separate unvalidated roles."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.projector.scale_factor",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.class_embd",
            "v.position_embd.weight",
            "mm.model.mlp.0.weight",
            "mm.model.mlp.1.weight",
            "mm.model.mlp.3.weight",
        ),
        block_prefix="v.blk.",
        block_suffixes=_FUSED_GELU_BLOCK_SUFFIXES,
        deferred_companions=(
            DeferredCompanionSpec(
                modality=MMProjModality.AUDIO,
                projector_type="parakeet",
                tensor_prefixes=("a.", "mm.a."),
                reason=(
                    "The co-resident Parakeet encoder is a separate audio role and "
                    "is quarantined from the vision graph."
                ),
            ),
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("nemotron-nano-v2-vl-bf16-header",),
        source_evidence_ids=("nemotron-v2-vl-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="exaone4_5",
        enum_name="PROJECTOR_TYPE_EXAONE4_5",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"exaone4"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact GQA dual-temporal vision tower, window schedule, spatial "
            "merger, processor inputs, tensor closure, and parity are covered; the "
            "oversized paired text runtime is not claimed."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.attention.head_count_kv",
            "clip.vision.image_min_pixels",
            "clip.vision.image_max_pixels",
            "clip.vision.n_wa_pattern",
            "clip.vision.window_size",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.post_ln.weight",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_FUSED_GATED_RMS_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("exaone4-5-33b-f16-header",),
        source_evidence_ids=("exaone4-5-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="hunyuanvl",
        enum_name="PROJECTOR_TYPE_HUNYUANVL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"hunyuan_vl"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact external-position ViT, convolutional perceiver, newline "
            "layout, boundary rows, tensor closure, and parity are covered; XD-RoPE "
            "text positioning and paired runtime remain unvalidated."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.spatial_merge_size",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "mm.pre_norm.weight",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
            "v.image_newline",
            "mm.model.fc.weight",
            "mm.model.fc.bias",
            "mm.image_begin",
            "mm.image_end",
            "mm.post_norm.weight",
        ),
        optional_top_tensors=("v.view_seperator",),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("hunyuanocr-bf16-header",),
        source_evidence_ids=("hunyuanvl-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="minicpmv4_6",
        enum_name="PROJECTOR_TYPE_MINICPMV4_6",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"qwen35"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact bucketed positions, inserted window-attention merger, final "
            "downsample MLP, packed processor inputs, tensor closure, and parity are "
            "covered; paired hybrid-runtime insertion remains unvalidated."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.projector.scale_factor",
            "clip.vision.wa_layer_indexes",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "v.vit_merger.ln1.weight",
            "v.vit_merger.ln1.bias",
            "v.vit_merger.attn_q.weight",
            "v.vit_merger.attn_q.bias",
            "v.vit_merger.attn_k.weight",
            "v.vit_merger.attn_k.bias",
            "v.vit_merger.attn_v.weight",
            "v.vit_merger.attn_v.bias",
            "v.vit_merger.attn_out.weight",
            "v.vit_merger.attn_out.bias",
            "v.vit_merger.ds_ln.weight",
            "v.vit_merger.ds_ln.bias",
            "v.vit_merger.ds_ffn_up.weight",
            "v.vit_merger.ds_ffn_up.bias",
            "v.vit_merger.ds_ffn_down.weight",
            "v.vit_merger.ds_ffn_down.bias",
            "mm.input_norm.weight",
            "mm.input_norm.bias",
            "mm.up.weight",
            "mm.up.bias",
            "mm.down.weight",
            "mm.down.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("minicpm-v4-6-bf16-header",),
        source_evidence_ids=("minicpmv4-6-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="granite_speech",
        enum_name="PROJECTOR_TYPE_GRANITE_SPEECH",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"granite"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact chunked Shaw-relative Conformer, CTC branch, two-layer "
            "Q-Former, processor ABI, tensor transforms, and independent parity "
            "are covered; paired text-runtime insertion remains unvalidated."
        ),
        sidecar_builder="audio_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(
            *_STANDALONE_AUDIO_METADATA,
            "clip.projector_type",
            "clip.audio.chunk_size",
            "clip.audio.conv_kernel_size",
            "clip.audio.max_pos_emb",
            "clip.audio.projector.window_size",
            "clip.audio.projector.downsample_rate",
            "clip.audio.projector.head_count",
        ),
        required_top_tensors=(
            "a.input_projection.weight",
            "a.input_projection.bias",
            "a.enc_ctc_out.weight",
            "a.enc_ctc_out.bias",
            "a.enc_ctc_out_mid.weight",
            "a.enc_ctc_out_mid.bias",
            "a.proj_query",
            "a.proj_norm.weight",
            "a.proj_norm.bias",
            "a.proj_linear.weight",
            "a.proj_linear.bias",
            *(
                f"a.proj_blk.{layer}.{stem}.{kind}"
                for layer in range(2)
                for stem in (
                    "self_attn_q",
                    "self_attn_k",
                    "self_attn_v",
                    "self_attn_out",
                    "self_attn_norm",
                    "cross_attn_q",
                    "cross_attn_k",
                    "cross_attn_v",
                    "cross_attn_out",
                    "cross_attn_norm",
                    "ffn_up",
                    "ffn_down",
                    "ffn_norm",
                )
                for kind in ("weight", "bias")
            ),
        ),
        block_prefix="a.blk.",
        block_suffixes=_GRANITE_SPEECH_BLOCK_SUFFIXES,
        tensor_roles=(
            ("a.proj_", MMProjTensorRole.PROJECTOR),
            ("a.", MMProjTensorRole.ENCODER),
        ),
        real_artifact_ids=("granite-speech-4.1-2b-f16",),
    ),
    ProjectorSpec(
        projector_type="mimovl",
        enum_name="PROJECTOR_TYPE_MIMOVL",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"mimo2"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact GQA row/column-window tower, attention sinks, float32 "
            "down-projection, merger, tensor closure, and parity are covered; the "
            "co-resident audio role and oversized paired text runtime are unvalidated."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.attention.head_count_kv",
            "clip.vision.wa_pattern_mode",
            "clip.vision.window_size",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.post_ln.weight",
            "mm.0.weight",
            "mm.2.weight",
        ),
        block_prefix="v.blk.",
        block_suffixes=_MIMOVL_BLOCK_SUFFIXES,
        auxiliary_tensor_patterns=(r"v\.blk\.\d+\.attn_sinks",),
        deferred_companions=(
            DeferredCompanionSpec(
                modality=MMProjModality.AUDIO,
                projector_type="mimo_audio",
                tensor_prefixes=("a.", "mm.a."),
                reason=(
                    "The co-resident MiMo audio encoder is a separate role and is "
                    "quarantined from the vision graph."
                ),
            ),
        ),
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("mimo-v2-5-f16-header",),
        source_evidence_ids=("mimovl-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="minimax_m3",
        enum_name="PROJECTOR_TYPE_MINIMAX_M3",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"minimax-m3"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Pinned source proves the exact dual-temporal patch embed, partial "
            "two-axis RoPE, and two-stage merger graph with independent synthetic "
            "parity. No bounded standalone sidecar or paired runtime is claimed."
        ),
        sidecar_builder="remaining_vision_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.spatial_merge_size",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "mm.1.weight",
            "mm.1.bias",
            "mm.2.weight",
            "mm.2.bias",
            "mm.merger.fc1.weight",
            "mm.merger.fc1.bias",
            "mm.merger.fc2.weight",
            "mm.merger.fc2.bias",
        ),
        block_prefix="v.blk.",
        block_suffixes=_GENERIC_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        source_evidence_ids=("minimax-m3-pinned-graph-source",),
    ),
    ProjectorSpec(
        projector_type="granite4_vision",
        enum_name="PROJECTOR_TYPE_GRANITE4_VISION",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"granite"}),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Exact standalone SigLIP plus Window-QFormer graph export is supported; "
            "deep-stack text injection and downstream runtime remain unvalidated."
        ),
        sidecar_builder="ocr_projector",
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.feature_layer",
            "clip.vision.image_grid_pinpoints",
            "clip.vision.projector.spatial_offsets",
            "clip.vision.projector.query_side",
            "clip.vision.projector.window_side",
            "clip.use_gelu",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "v.image_newline",
        ),
        block_prefix="v.blk.",
        block_suffixes=(
            "attn_q.weight",
            "attn_q.bias",
            "attn_k.weight",
            "attn_k.bias",
            "attn_v.weight",
            "attn_v.bias",
            "attn_out.weight",
            "attn_out.bias",
            "ffn_up.weight",
            "ffn_up.bias",
            "ffn_down.weight",
            "ffn_down.bias",
            "ln1.weight",
            "ln1.bias",
            "ln2.weight",
            "ln2.bias",
        ),
        indexed_tensors=(
            IndexedTensorSpec(
                "clip.vision.feature_layer",
                "v.proj_blk.",
                _GRANITE4_PROJECTOR_SUFFIXES,
                count_is_array_length=True,
            ),
        ),
        tensor_roles=(
            ("v.blk.", MMProjTensorRole.ENCODER),
            ("v.proj_blk.", MMProjTensorRole.PROJECTOR),
            ("v.image_", MMProjTensorRole.PROJECTOR),
            ("v.", MMProjTensorRole.ENCODER),
        ),
        real_artifact_ids=("granite4-vision-4.1-f16",),
    ),
    ProjectorSpec(
        projector_type="mimo_audio",
        enum_name="PROJECTOR_TYPE_MIMO_AUDIO",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"mimo2"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact causal audio transformer, RVQ, code-embedding bridge, local "
            "transformer, projection, processor ABI, and independent parity are "
            "covered from pinned source; no public compatible mmproj or paired runtime "
            "evidence is claimed."
        ),
        sidecar_builder="audio_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(
            *_STANDALONE_AUDIO_METADATA,
            "clip.audio.projector_type",
            "clip.audio.rvq.num_quantizers",
            "clip.audio.rvq.codebook_size",
            "clip.audio.wa_pattern_mode",
            "clip.audio.window_size",
            "clip.audio.local_block_count",
            "clip.audio.local_group_size",
        ),
        required_top_tensors=(
            "a.conv1d.1.weight",
            "a.conv1d.1.bias",
            "a.conv1d.2.weight",
            "a.conv1d.2.bias",
            "a.post_ln.weight",
            "a.post_ln.bias",
            "a.downsample.conv.weight",
            "a.downsample.norm.weight",
            "a.downsample.norm.bias",
            "a.rvq.codebook.weight",
            "mm.a.code_embd.weight",
            "mm.a.local_norm.weight",
            "mm.a.mlp.1.weight",
            "mm.a.mlp.2.weight",
        ),
        block_prefix="a.blk.",
        block_suffixes=_WHISPER_AUDIO_BLOCK_SUFFIXES,
        auxiliary_tensor_patterns=_MIMO_LOCAL_PATTERN,
        deferred_companions=(
            DeferredCompanionSpec(
                modality=MMProjModality.VISION,
                projector_type="mimovl",
                tensor_prefixes=("v.", "mm.0.", "mm.2."),
                reason=(
                    "The co-located MiMo vision encoder is a separate route and is "
                    "quarantined from the audio graph."
                ),
            ),
        ),
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        source_evidence_ids=("mimo-v2.5-audio-source",),
    ),
    ProjectorSpec(
        projector_type="parakeet",
        enum_name="PROJECTOR_TYPE_PARAKEET",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"nemotron_h"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact FastConformer encoder, frozen BatchNorm, relative attention, "
            "squared-ReLU projector, processor ABI, and independent parity are covered "
            "from pinned source; the published vision-only candidate is not treated as "
            "audio evidence and paired runtime remains unvalidated."
        ),
        sidecar_builder="audio_projector",
        model_roles=(MMProjModelRole.AUDIO_ENCODER,),
        required_metadata=(
            *_STANDALONE_AUDIO_METADATA,
            "clip.audio.projector_type",
            "clip.audio.subsampling_factor",
            "clip.audio.conv_kernel_size",
        ),
        required_top_tensors=(
            "a.mel_filters",
            "a.window",
            "a.conv1d.0.weight",
            "a.conv1d.0.bias",
            "a.conv1d.2.weight",
            "a.conv1d.2.bias",
            "a.conv1d.3.weight",
            "a.conv1d.3.bias",
            "a.conv1d.5.weight",
            "a.conv1d.5.bias",
            "a.conv1d.6.weight",
            "a.conv1d.6.bias",
            "a.pre_encode.out.weight",
            "a.pre_encode.out.bias",
            "mm.a.norm_pre.weight",
            "mm.a.mlp.1.weight",
            "mm.a.mlp.2.weight",
        ),
        optional_top_tensors=("mm.a.mlp.1.bias", "mm.a.mlp.2.bias"),
        block_prefix="a.blk.",
        block_suffixes=_PARAKEET_BLOCK_SUFFIXES,
        deferred_companions=(
            DeferredCompanionSpec(
                modality=MMProjModality.VISION,
                projector_type="nemotron_v2_vl",
                tensor_prefixes=("v.", "mm."),
                reason=(
                    "The co-located RADIO vision route is independently owned and "
                    "quarantined from the Parakeet graph."
                ),
            ),
        ),
        tensor_roles=(
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        source_evidence_ids=("nemotron-v2-parakeet-source",),
    ),
    ProjectorSpec(
        projector_type="qwen3tts_spkenc",
        enum_name="PROJECTOR_TYPE_QWEN3TTS_SPKENC",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"qwen3tts"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The standalone ECAPA-TDNN speaker encoder is exact. Runtime use remains "
            "unvalidated because its one-row output must be added to the tts_pad "
            "embedding inside the architecture-specific four-section MRoPE prompt, and "
            "current converters co-package an unowned stateful qwen3tts_gen runtime."
        ),
        sidecar_builder="qwen_glm_projector",
        model_roles=(MMProjModelRole.SPEAKER_ENCODER,),
        required_metadata=(
            *_COMMON_AUDIO_BODY_METADATA,
            "clip.audio.projector_type",
        ),
        required_top_tensors=_QWEN3TTS_SPEAKER_TOP_TENSORS,
        deferred_companions=(
            DeferredCompanionSpec(
                modality=MMProjModality.GENERATED_AUDIO,
                projector_type="qwen3tts_gen",
                tensor_prefixes=("a.gen.",),
                reason=(
                    "qwen3tts_gen owns autoregressive sampling and streaming codec "
                    "state, not speaker embedding projection."
                ),
            ),
        ),
        tensor_roles=(
            ("a.gen.", MMProjTensorRole.GENERATED_AUDIO),
            ("a.", MMProjTensorRole.ENCODER),
            ("mm.a.", MMProjTensorRole.PROJECTOR),
        ),
        source_evidence_ids=("qwen3tts-speaker-runtime-boundary",),
    ),
    _rejected(
        "qwen3tts_gen",
        "PROJECTOR_TYPE_QWEN3TTS_GEN",
        _GEN_AUDIO_BASE,
        "Generated-audio decoder sidecars are not multimodal projectors and cannot be paired with a text target package.",
    ),
    ProjectorSpec(
        projector_type="pockettts_spkenc",
        enum_name="PROJECTOR_TYPE_POCKETTTS_SPKENC",
        modalities=_AUDIO_BASE,
        target_architectures=frozenset({"pockettts"}),
        primary_modality=MMProjModality.AUDIO,
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "The exact raw-waveform SEANet, causal Mimi transformer, downsampler, "
            "speaker projection, frame ABI, RoPE transform, and independent parity "
            "are covered from pinned source; the generated-audio decoder remains "
            "explicitly quarantined and rejected."
        ),
        sidecar_builder="audio_projector",
        model_roles=(MMProjModelRole.SPEAKER_ENCODER,),
        required_metadata=(*_STANDALONE_AUDIO_METADATA, "clip.audio.projector_type"),
        required_top_tensors=(
            "a.seanet.conv_in.weight",
            "a.seanet.conv_in.bias",
            "a.seanet.conv_out.weight",
            "a.seanet.conv_out.bias",
            *(
                f"a.seanet.blk.{layer}.{stem}.{kind}"
                for layer in range(3)
                for stem in ("res_conv1", "res_conv2", "scale_conv")
                for kind in ("weight", "bias")
            ),
            "a.downsample.conv.weight",
            "a.speaker_proj.weight",
        ),
        block_prefix="a.blk.",
        block_suffixes=_POCKETTTS_BLOCK_SUFFIXES,
        deferred_companions=(
            DeferredCompanionSpec(
                modality=MMProjModality.GENERATED_AUDIO,
                projector_type="pockettts_gen",
                tensor_prefixes=("a.gen.",),
                reason=(
                    "PocketTTS generation is a separate rejected decoder role and is "
                    "never loaded into the speaker encoder."
                ),
            ),
        ),
        tensor_roles=(
            ("a.speaker_proj.", MMProjTensorRole.PROJECTOR),
            ("a.gen.", MMProjTensorRole.GENERATED_AUDIO),
            ("a.", MMProjTensorRole.ENCODER),
        ),
        source_evidence_ids=("pockettts-speaker-source",),
    ),
    _rejected(
        "pockettts_gen",
        "PROJECTOR_TYPE_POCKETTTS_GEN",
        _GEN_AUDIO_BASE,
        "Generated-audio decoder sidecars are not multimodal projectors and cannot be paired with a text target package.",
    ),
    ProjectorSpec(
        projector_type="muse-glimmer",
        enum_name="PROJECTOR_TYPE_MUSE_GLIMMER",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"muse-glimmer"}),
        model_roles=(MMProjModelRole.VISION_ENCODER,),
        metadata=Support.SUPPORTED,
        tensor_map=Support.SUPPORTED,
        graph=Support.SUPPORTED,
        runtime=Support.DEFERRED,
        reason=(
            "Metadata, tensor closure, graph construction, and independent component parity "
            "are covered, but the paired text target, processor, and deterministic package "
            "generation have no end-to-end runtime evidence."
        ),
        builder="muse_glimmer",
        required_metadata=(
            *_COMMON_REQUIRED_VISION_METADATA,
            "clip.projector_type",
            "clip.vision.spatial_merge_size",
        ),
        required_top_tensors=(
            "v.patch_embd.weight",
            "v.position_embd.weight",
            "v.pre_ln.weight",
            "v.pre_ln.bias",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.0.weight",
            "mm.1.weight",
            "mm.2.weight",
        ),
        block_prefix="v.blk.",
        block_suffixes=_MUSE_GLIMMER_BLOCK_SUFFIXES,
        tensor_roles=(
            ("v.", MMProjTensorRole.ENCODER),
            ("mm.", MMProjTensorRole.PROJECTOR),
        ),
        real_artifact_ids=("muse-glimmer-30b-bf16",),
    ),
)

_INDEX: Mapping[str, ProjectorSpec] = MappingProxyType(
    {spec.projector_type: spec for spec in _SPECS}
)

MMPROJ_SOURCE_EVIDENCE: tuple[MMProjSourceEvidence, ...] = (
    MMProjSourceEvidence(
        evidence_id="glma-converter-checkpoint-drift",
        sources=(
            (
                "ggml-org/llama.cpp",
                LLAMA_CPP_MMPROJ_SHA,
                "tools/mtmd/models/whisper-enc.cpp",
            ),
            (
                "ggml-org/llama.cpp",
                LLAMA_CPP_MMPROJ_SHA,
                "conversion/ultravox.py",
            ),
            (
                "zai-org/GLM-ASR-Nano-2512",
                "61ba4e0b3309b6656edea3e93e419f7bd5c61957",
                "config.json",
            ),
        ),
        finding=(
            "llama.cpp defines the legacy additive-position GLMA graph, but its converter "
            "registers GlmasrModel and requires a top-level merge_factor. The immutable "
            "current checkpoint declares GlmAsrForConditionalGeneration, omits "
            "merge_factor, and configures a partial-RoPE encoder. Mobius can import a "
            "structurally valid legacy sidecar without claiming that this checkpoint can "
            "produce one."
        ),
    ),
    MMProjSourceEvidence(
        evidence_id="qwen3tts-speaker-runtime-boundary",
        sources=(
            (
                "ggml-org/llama.cpp",
                LLAMA_CPP_MMPROJ_SHA,
                "tools/mtmd/models/qwen3tts-spkenc.cpp",
            ),
            (
                "ggml-org/llama.cpp",
                LLAMA_CPP_MMPROJ_SHA,
                "conversion/qwen3tts.py",
            ),
            (
                "ggml-org/llama.cpp",
                LLAMA_CPP_MMPROJ_SHA,
                "tools/mtmd/mtmd-helper-gen.cpp",
            ),
        ),
        finding=(
            "The speaker graph deterministically emits one text-width ECAPA embedding. "
            "The downstream helper adds it to the tts_pad embedding inside a hand-built "
            "four-section MRoPE prompt, while the converter co-emits a separate stateful "
            "qwen3tts_gen namespace. Mobius exports only the exact speaker_encoder role "
            "and makes no generated-audio runtime claim."
        ),
    ),
    *(MMProjSourceEvidence(**record) for record in REMAINING_MMPROJ_SOURCE_RECORDS),
)

MMPROJ_ARTIFACT_AVAILABILITY_PINS: tuple[MMProjArtifactAvailabilityPin, ...] = (
    MMProjArtifactAvailabilityPin(
        artifact_id="lfm2-vl-1.6b-f16",
        projector_type="lfm2",
        repository="LiquidAI/LFM2-VL-1.6B-GGUF",
        revision="6121de267003bb4d4f325fe10abdc735aee06747",
        filename="mmproj-LFM2-VL-1.6B-F16.gguf",
        size=830_339_008,
        lfs_sha256="b637bfa6060be2bc7503ec23ba48b407843d08c2ca83f52be206ea8563ccbae2",
    ),
    MMProjArtifactAvailabilityPin(
        artifact_id="lfm2-vl-1.6b-q8-0",
        projector_type="lfm2",
        repository="LiquidAI/LFM2-VL-1.6B-GGUF",
        revision="6121de267003bb4d4f325fe10abdc735aee06747",
        filename="mmproj-LFM2-VL-1.6B-Q8_0.gguf",
        size=564_115_648,
        lfs_sha256="65ec437db88d65fff93f472d00c145e09880769ac67fedff5cd1c0f8d8301d87",
    ),
    MMProjArtifactAvailabilityPin(
        artifact_id="pixtral-12b-q8-0",
        projector_type="pixtral",
        repository="ggml-org/pixtral-12b-GGUF",
        revision="cba1ea4420bc2b4f15f50fdec59e30769880a63c",
        filename="mmproj-pixtral-12b-Q8_0.gguf",
        size=463_091_616,
        lfs_sha256="5504fe00067629053e6f99abac05f628c653a50394f4929bcc185bc80a10daf4",
    ),
)

MMPROJ_ARTIFACT_PINS: tuple[MMProjArtifactPin, ...] = (
    MMProjArtifactPin(
        artifact_id="gemma3n-e4b-f16",
        repository="Qwe1325/gemma-3n-E4B-it-GGUF",
        revision="f26cfdb3f7e86ede704fc45410316e48ccb1a018",
        filename="mmproj-F16.gguf",
        size=1_967_809_568,
        lfs_sha256="a464216b97121e8065216569fc501880cc456ef859e41234235106b7348e0279",
        projector_types=("gemma3nv", "gemma3na"),
        paired_text_architecture="gemma3n",
        paired_text_target="gemma-3n-E4B-it-q3_k_m.gguf",
        metadata=(
            ("clip.vision.projector_type", "gemma3nv"),
            ("clip.audio.projector_type", "gemma3na"),
            ("clip.vision.embedding_length", 2048),
            ("clip.audio.embedding_length", 1536),
            ("clip.vision.image_size", 768),
            ("clip.vision.patch_size", 3),
        ),
        tensor_qtypes=(("F16", 418), ("F32", 407)),
        tensor_count=825,
        parity_test=(
            "gemma3nv: test_soft_path_matches_huggingface; gemma3na: test_matches_huggingface"
        ),
        paired_text_repository="Qwe1325/gemma-3n-E4B-it-GGUF",
        paired_text_revision="f26cfdb3f7e86ede704fc45410316e48ccb1a018",
        paired_text_size=3_442_332_096,
        processor_repository="unsloth/gemma-3n-E4B-it",
        processor_revision="45e9fb1dd0e34db5ff9db1f43a49ac5d8e8b8778",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
        ),
        processor_class="Gemma3nProcessor",
        processor_contract=(
            ("pixel_values", "float32[1,3,768,768]"),
            ("image_features", "256 rows per image"),
            ("input_features", "float32[1,time,128]"),
            ("audio_features", "188 rows per clip"),
            ("ordering", "batch-major raster image rows; batch-major audio rows"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="gemma4-unified-12b-f16",
        repository="unsloth/gemma-4-12b-it-GGUF",
        revision="fc034cfff751157913579611efad8462ac1be606",
        filename="mmproj-F16.gguf",
        size=175_115_840,
        lfs_sha256="91f086971e56d7a7d8d39e271873fccdb49541bd259d6e02c401a4f1cb7a219e",
        projector_types=("gemma4uv", "gemma4ua"),
        paired_text_architecture="gemma4",
        paired_text_target="gemma-4-12b-it-Q4_K_M.gguf",
        metadata=(
            ("clip.vision.projector_type", "gemma4uv"),
            ("clip.audio.projector_type", "gemma4ua"),
            ("clip.vision.embedding_length", 3840),
            ("clip.audio.embedding_length", 640),
            ("clip.vision.block_count", 0),
            ("clip.audio.block_count", 0),
        ),
        tensor_qtypes=(("F16", 2), ("F32", 9)),
        tensor_count=11,
        parity_test=(
            "gemma4uv: test_unified_vision_loader_restores_hf_patch_order; "
            "gemma4ua: test_gemma4_nonunified_and_unified_audio_configs_do_not_alias"
        ),
        paired_text_repository="unsloth/gemma-4-12b-it-GGUF",
        paired_text_revision="fc034cfff751157913579611efad8462ac1be606",
        paired_text_size=7_121_861_440,
        processor_repository="google/gemma-4-12B-it",
        processor_revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        processor_files=("config.json", "processor_config.json", "tokenizer_config.json"),
        processor_class="Gemma4UnifiedProcessor",
        processor_contract=(
            ("pixel_values", "float32[1,num_patches,6912]"),
            ("pixel_position_ids", "int64[1,num_patches,2]"),
            ("image_features", "one row per valid 48x48 merged patch; 40..280"),
            ("input_features", "float32[1,time,640] raw 40ms waveform frames"),
            ("audio_features", "one row per valid 640-sample frame"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="smolvlm-256m-idefics3-f16",
        repository="ggml-org/SmolVLM-256M-Instruct-GGUF",
        revision="b9e4379657e1450d04d02eec8e345667265b0a00",
        filename="mmproj-SmolVLM-256M-Instruct-f16.gguf",
        size=190_031_616,
        lfs_sha256="0802360aca1748f112ea510b8ff277c65b1361c8ef30ed89b83c9c7a60d08e96",
        projector_types=("idefics3",),
        paired_text_architecture="llama",
        paired_text_target="SmolVLM-256M-Instruct-f16.gguf",
        metadata=(
            ("clip.projector_type", "idefics3"),
            ("clip.vision.embedding_length", 768),
            ("clip.vision.projection_dim", 576),
            ("clip.vision.image_size", 512),
            ("clip.vision.patch_size", 16),
            ("clip.vision.projector.scale_factor", 4),
        ),
        tensor_qtypes=(("F16", 73), ("F32", 125)),
        tensor_count=198,
        parity_test="test_idefics3_projector_matches_independent_reference",
        paired_text_repository="ggml-org/SmolVLM-256M-Instruct-GGUF",
        paired_text_revision="b9e4379657e1450d04d02eec8e345667265b0a00",
        paired_text_size=327_809_728,
        processor_repository="HuggingFaceTB/SmolVLM-256M-Instruct",
        processor_revision="7e3e67edbbed1bf9888184d9df282b700a323964",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
        ),
        processor_class="Idefics3Processor",
        processor_contract=(
            ("pixel_values", "float32[num_tiles,3,512,512]"),
            ("image_features", "64 rows per tile; at most 1088 rows"),
            ("ordering", "refined raster tiles followed by overview"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="internvl25-1b-f16",
        repository="ggml-org/InternVL2_5-1B-GGUF",
        revision="d77253530c9a27486a28800afaf6ff5576c0bf17",
        filename="mmproj-InternVL2_5-1B-f16.gguf",
        size=619_876_960,
        lfs_sha256="0c672edd99ec0b99df01c75dbb6cc26ad2236d7d61f908c93b5fda9b4d9ddd20",
        projector_types=("internvl",),
        paired_text_architecture="qwen2",
        paired_text_target="InternVL2_5-1B-f16.gguf",
        metadata=(
            ("clip.projector_type", "internvl"),
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.projection_dim", 896),
            ("clip.vision.image_size", 448),
            ("clip.vision.patch_size", 14),
            ("clip.vision.projector.scale_factor", 2),
        ),
        tensor_qtypes=(("F16", 147), ("F32", 295)),
        tensor_count=442,
        parity_test="test_internvl_projector_matches_independent_reference",
        paired_text_repository="ggml-org/InternVL2_5-1B-GGUF",
        paired_text_revision="d77253530c9a27486a28800afaf6ff5576c0bf17",
        paired_text_size=1_265_481_408,
        processor_repository="OpenGVLab/InternVL2_5-1B",
        processor_revision="9d423ea1ae9f893897ee3f7493141073f5afcf22",
        processor_files=("config.json", "preprocessor_config.json", "tokenizer_config.json"),
        processor_class="InternVLChatModel processor",
        processor_contract=(
            ("pixel_values", "float32[num_tiles,3,448,448]"),
            ("image_features", "256 rows per tile; at most 3328 rows"),
            ("ordering", "refined raster tiles followed by overview"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="llama4-scout-f16",
        repository="ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF",
        revision="42675345da11ade9203a5187595da7b74d4ff2ac",
        filename="mmproj-Llama-4-Scout-17B-16E-Instruct-f16.gguf",
        size=1_746_780_608,
        lfs_sha256="a7eec12068ae70f993fbba6eb350c095727be20f7a6ecbe6e431940c1a8823fb",
        projector_types=("llama4",),
        paired_text_architecture="llama4",
        paired_text_target="complete Llama4 Scout text GGUF exceeds 16 GiB",
        metadata=(
            ("clip.projector_type", "llama4"),
            ("clip.vision.embedding_length", 1408),
            ("clip.vision.projection_dim", 5120),
            ("clip.vision.image_size", 336),
            ("clip.vision.patch_size", 14),
            ("clip.vision.projector.scale_factor", 2),
        ),
        tensor_qtypes=(("F16", 208), ("F32", 346)),
        tensor_count=554,
        parity_test=(
            "test_tiny_tower_matches_independent_torch_reference; "
            "test_llama4_projector_matches_independent_reference"
        ),
        processor_repository="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        processor_revision="92f3b1597a195b523d8d9e5700e57e4fbb8f20d3",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
        ),
        processor_class="Llama4Processor",
        processor_contract=(
            ("pixel_values", "float32[num_tiles,3,336,336]"),
            ("image_features", "144 rows per tile"),
            ("ordering", "refined raster tiles followed by overview"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="pixtral-12b-f16",
        repository="ggml-org/pixtral-12b-GGUF",
        revision="cba1ea4420bc2b4f15f50fdec59e30769880a63c",
        filename="mmproj-pixtral-12b-f16.gguf",
        size=870_070_176,
        lfs_sha256="b4819558d6524a2e5623a06104ee085253a6dfd2b51470c60771ec33976f81bb",
        projector_types=("pixtral",),
        paired_text_architecture="llama",
        paired_text_target="pixtral-12b-Q2_K.gguf",
        metadata=(
            ("clip.projector_type", "pixtral"),
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.projection_dim", 5120),
            ("clip.vision.image_size", 1024),
            ("clip.vision.patch_size", 16),
        ),
        tensor_qtypes=(("F16", 171), ("F32", 52)),
        tensor_count=223,
        parity_test="test_pixtral_projector_inserts_distinct_row_breaks",
        paired_text_repository="ggml-org/pixtral-12b-GGUF",
        paired_text_revision="cba1ea4420bc2b4f15f50fdec59e30769880a63c",
        paired_text_size=4_791_047_808,
        processor_repository="mistral-experimental/pixtral-12b",
        processor_revision="c2756cbbb9422eba9f6c5c439a214b0392dfc998",
        processor_files=("config.json", "preprocessor_config.json"),
        processor_class="PixtralProcessor",
        processor_contract=(
            ("pixel_values", "float32[1,3,H,W], H and W divisible by 16"),
            ("image_features", "(H/16)*(W/16)+(H/16)-1 rows"),
            ("ordering", "raster patches with [IMG_BREAK] between rows"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="llava-llama3-8b-mlp-f16",
        repository="xtuner/llava-llama-3-8b-v1_1-gguf",
        revision="344f1bfe987bcbdc7e650b134d23670d5ffb5892",
        filename="llava-llama-3-8b-v1_1-mmproj-f16.gguf",
        size=624_434_368,
        lfs_sha256="eb569aba7d65cf3da1d0369610eb6869f4a53ee369992a804d5810a80e9fa035",
        projector_types=("mlp",),
        paired_text_architecture="llama",
        paired_text_target="Meta-Llama-3-8B-Instruct-Q2_K.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.block_count", 23),
            ("clip.vision.image_size", 336),
            ("clip.vision.patch_size", 14),
        ),
        tensor_qtypes=(("F32", 235), ("F16", 142)),
        tensor_count=377,
        parity_test="TestGenericGGUFProjectors.test_mlp_matches_nonzero_reference",
        paired_text_repository="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        paired_text_revision="4ebc4aa83d60a5d6f9e1e1e9272a4d6306d770c1",
        paired_text_size=3_179_131_456,
        processor_repository="xtuner/llava-llama-3-8b-v1_1-transformers",
        processor_revision="b20fb3040caaf5d0b3751c0d86a94efdf5bb007d",
        processor_files=("config.json", "preprocessor_config.json"),
        processor_contract=(("pixel_values", "float32 [1,3,336,336]"),),
    ),
    MMProjArtifactPin(
        artifact_id="mobilevlm-1.7b-ldp-f16",
        repository="guinmoon/MobileVLM-1.7B-GGUF",
        revision="7e0cdbd2d642d938ce82fadde991360500c7d7cf",
        filename="MobileVLM-1.7B-mmproj-f16.gguf",
        size=620_384_896,
        lfs_sha256="7d9855d323cee2a1797a88f9d7057ce26b21dcd62a50b382c4ff44ea60c77e39",
        projector_types=("ldp",),
        paired_text_architecture="llama",
        paired_text_target="MobileVLM-1.7B-Q4_K.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.block_count", 23),
            ("clip.vision.image_size", 336),
            ("clip.vision.patch_size", 14),
        ),
        tensor_qtypes=(("F32", 247), ("F16", 150)),
        tensor_count=397,
        parity_test=(
            "TestGenericGGUFProjectors."
            "test_ldp_matches_nonzero_reference_and_144_token_contract"
        ),
        paired_text_repository="guinmoon/MobileVLM-1.7B-GGUF",
        paired_text_revision="7e0cdbd2d642d938ce82fadde991360500c7d7cf",
        paired_text_size=834_055_776,
    ),
    MMProjArtifactPin(
        artifact_id="mobilevlm-v2-1.7b-ldpv2-f16",
        repository="ZiangWu/MobileVLM_V2-1.7B-GGUF",
        revision="422c888cc387d71831bedf48d59f0a66b27fad68",
        filename="mmproj-model-f16.gguf",
        size=595_103_072,
        lfs_sha256="57966afa654e9d46a11b2a4b17989c2d487cd961f702c4fe310f86db5e30aab4",
        projector_types=("ldpv2",),
        paired_text_architecture="llama",
        paired_text_target="ggml-model-q4_k.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.block_count", 23),
            ("clip.vision.image_size", 336),
            ("clip.vision.patch_size", 14),
        ),
        tensor_qtypes=(("F32", 236), ("F16", 143)),
        tensor_count=379,
        parity_test=(
            "TestGenericGGUFProjectors."
            "test_ldpv2_matches_nonzero_reference_and_144_token_contract"
        ),
        paired_text_repository="ZiangWu/MobileVLM_V2-1.7B-GGUF",
        paired_text_revision="422c888cc387d71831bedf48d59f0a66b27fad68",
        paired_text_size=791_817_856,
    ),
    MMProjArtifactPin(
        artifact_id="glm-edge-v-2b-adapter-f16",
        repository="zai-org/glm-edge-v-2b-gguf",
        revision="d76cbe14f1d3a9405f664cbb5ae0c9537197429a",
        filename="mmproj-model-f16.gguf",
        size=933_229_600,
        lfs_sha256="69a11ec5f54219fef9fd6bf9bc3209f0e6ef1564462cc4705dd93b2cd2a8198c",
        projector_types=("adapter",),
        paired_text_architecture="chatglm",
        paired_text_target="ggml-model-Q4_0.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.block_count", 27),
            ("clip.vision.image_size", 672),
            ("clip.vision.patch_size", 14),
        ),
        tensor_qtypes=(("F32", 278), ("F16", 169)),
        tensor_count=447,
        parity_test=(
            "TestGenericGGUFProjectors."
            "test_adapter_matches_nonzero_reference_and_boundary_rows"
        ),
        paired_text_repository="zai-org/glm-edge-v-2b-gguf",
        paired_text_revision="d76cbe14f1d3a9405f664cbb5ae0c9537197429a",
        paired_text_size=931_269_056,
        processor_repository="THUDM/glm-edge-v-2b",
        processor_revision="2053707733f99ab52e943904f43c2359a94301ef",
        processor_files=("config.json", "preprocessor_config.json"),
        processor_contract=(("pixel_values", "float32 [1,3,672,672]"),),
    ),
    MMProjArtifactPin(
        artifact_id="minicpm-v2-resampler-f16",
        repository="openbmb/MiniCPM-V-2-gguf",
        revision="3a38804c39d96c935a6b542581f51171aefa06a5",
        filename="mmproj-model-f16.gguf",
        size=866_071_872,
        lfs_sha256="79611c59b5ad5b0547256602e3fb546a3041bcf6db5058091b6bcaa31f3a1c95",
        projector_types=("resampler",),
        paired_text_architecture="minicpm",
        paired_text_target="ggml-model-Q2_K.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.block_count", 26),
            ("clip.vision.image_size", 448),
            ("clip.vision.patch_size", 14),
        ),
        tensor_qtypes=(("F32", 276), ("F16", 164)),
        tensor_count=440,
        parity_test=(
            "component-only: TestGenericGGUFProjectors."
            "test_resampler_matches_nonzero_reference_including_query_positions"
        ),
        paired_text_repository="openbmb/MiniCPM-V-2-gguf",
        paired_text_revision="3a38804c39d96c935a6b542581f51171aefa06a5",
        paired_text_size=1_297_193_376,
    ),
    MMProjArtifactPin(
        artifact_id="qwen2-vl-2b-f16",
        repository="ggml-org/Qwen2-VL-2B-Instruct-GGUF",
        revision="bb307c036e8a1ed7b663bbd0c35b41c4c9294cfd",
        filename="mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
        size=1_331_656_160,
        lfs_sha256="ecb20cabcdd8dbc277de06bd6eb980aeb2adfaaba9f199a434e328d205675d03",
        projector_types=("qwen2vl_merger",),
        paired_text_architecture="qwen2vl",
        paired_text_target="Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1280),
            ("clip.vision.projection_dim", 1536),
            ("clip.vision.block_count", 32),
            ("clip.vision.image_size", 560),
            ("clip.vision.patch_size", 14),
        ),
        tensor_qtypes=(("F32", 324), ("F16", 196)),
        tensor_count=520,
        parity_test=("TestQwenVLMMProj.test_qwen_tensor_transform_values[qwen2vl_merger]"),
        processor_repository="Qwen/Qwen2-VL-2B-Instruct",
        processor_revision="895c3a49bc3fa70a340399125c650a463535e71c",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "chat_template.json",
        ),
        processor_class="Qwen2VLProcessor",
        processor_contract=(
            ("pixel_values", "float32[total_image_patches,1176]"),
            ("image_grid_thw", "int64[num_images,3]"),
            ("pixel_values_videos", "float32[total_video_patches,1176]"),
            ("video_grid_thw", "int64[num_videos,3]"),
            ("empty_media", "omit image/video pixel and grid keys"),
            ("ordering", "batch-major within independent image and video streams"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="qwen25-vl-3b-f16",
        repository="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        revision="5037fcf163dd95d1e41d1974465f0898ed108ca2",
        filename="mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
        size=1_338_428_128,
        lfs_sha256="b9160fe9d814d1fadf68395677468534778b39ac33c2e7561b7b218626e60d5e",
        projector_types=("qwen2.5vl_merger",),
        paired_text_architecture="qwen2vl",
        paired_text_target="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1280),
            ("clip.vision.projection_dim", 2048),
            ("clip.vision.block_count", 32),
            ("clip.vision.image_size", 560),
            ("clip.vision.patch_size", 14),
            ("clip.vision.n_wa_pattern", 8),
        ),
        tensor_qtypes=(("F32", 291), ("F16", 228)),
        tensor_count=519,
        parity_test=("TestQwenVLMMProj.test_qwen_tensor_transform_values[qwen2.5vl_merger]"),
        processor_repository="Qwen/Qwen2.5-VL-3B-Instruct",
        processor_revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "chat_template.json",
        ),
        processor_class="Qwen2_5_VLProcessor",
        processor_contract=(
            ("pixel_values", "float32[total_image_patches,1176]"),
            ("image_grid_thw", "int64[num_images,3]"),
            ("pixel_values_videos", "float32[total_video_patches,1176]"),
            ("video_grid_thw", "int64[num_videos,3]"),
            ("second_per_grid_ts", "float64[num_videos]"),
            ("empty_media", "omit image/video pixel, grid, and timing keys"),
            ("ordering", "batch-major within independent image and video streams"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="gemma3-4b-f16",
        repository="ggml-org/gemma-3-4b-it-GGUF",
        revision="ab31416aceb30cd095cb34cc27eea120940964e4",
        filename="mmproj-model-f16.gguf",
        size=851_251_104,
        lfs_sha256="8c0fb064b019a6972856aaae2c7e4792858af3ca4561be2dbf649123ba6c40cb",
        projector_types=("gemma3",),
        paired_text_architecture="gemma3",
        paired_text_target="gemma-3-4b-it-Q4_K_M.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.projection_dim", 2560),
            ("clip.vision.block_count", 27),
            ("clip.vision.image_size", 896),
            ("clip.vision.patch_size", 14),
        ),
        tensor_qtypes=(("F32", 276), ("F16", 163)),
        tensor_count=439,
        parity_test="TestGemma3VisionEncoder.test_projector_matches_numpy_reference",
        processor_repository="google/gemma-3-4b-it",
        processor_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        processor_files=(
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ),
        processor_class="Gemma3Processor",
        processor_contract=(
            ("pixel_values", "float32[num_images,3,896,896]"),
            ("vision_invocation", "split to one image row per vision graph call"),
            ("image_features", "concatenate 256 rows per image in processor row order"),
            ("empty_media", "omit pixel_values"),
            ("ordering", "batch-major image rows"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="gemma4-e2b-f16",
        repository="unsloth/gemma-4-E2B-it-GGUF",
        revision="0314792d7f1f7e229411f620751375812bb9faf2",
        filename="mmproj-F16.gguf",
        size=985_654_080,
        lfs_sha256="337ee849e80b6169ce9d1d573d424fc1653bcafa5f0cb0cbb901beba54f4b41c",
        projector_types=("gemma4v", "gemma4a"),
        paired_text_architecture="gemma4",
        paired_text_target="gemma-4-E2B-it-Q4_K_M.gguf",
        metadata=(
            ("general.name", "Gemma-4-E2B-It"),
            ("general.base_model.0.repo_url", "https://huggingface.co/google/gemma-4-E2B-it"),
            ("clip.vision.embedding_length", 768),
            ("clip.vision.projection_dim", 1536),
            ("clip.audio.embedding_length", 1024),
            ("clip.audio.projection_dim", 1536),
        ),
        tensor_qtypes=(("F32", 1163), ("F16", 248)),
        tensor_count=1411,
        parity_test=(
            "vision: TestVisionEncoderBuildAndRun.test_matches_independent_numpy_reference; "
            "audio: test_gemma4_audio_projector_matches_reference"
        ),
        paired_text_repository="unsloth/gemma-4-E2B-it-GGUF",
        paired_text_revision="0314792d7f1f7e229411f620751375812bb9faf2",
        paired_text_size=3_106_738_272,
        processor_repository="google/gemma-4-E2B-it",
        processor_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        processor_files=("config.json", "processor_config.json", "tokenizer_config.json"),
        processor_class="Gemma4Processor",
        processor_contract=(
            ("pixel_values", "float32[1,num_patches,768] with int64 positions"),
            ("image_features", "40..280 pooled rows"),
            ("input_features", "float32[1,time,128] log-mel frames"),
            ("audio_features", "ceil(time/4) valid rows"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="muse-glimmer-30b-bf16",
        repository="unsloth/Muse-Glimmer-30B-GGUF",
        revision="faa5b025c584459c13febfa5c59883516710ae39",
        filename="mmproj-Muse-Glimmer-30B-BF16.gguf",
        size=3_849_173_728,
        lfs_sha256="7aa788cfe25ae5e4bf4837511f64df22cabe595e58223708274a67b3136f53ab",
        projector_types=("muse-glimmer",),
        paired_text_architecture="muse-glimmer",
        paired_text_target="Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        metadata=(
            ("general.name", "Muse-Glimmer-30B"),
            ("clip.vision.embedding_length", 1536),
            ("clip.vision.projection_dim", 6656),
            ("clip.vision.spatial_merge_size", 2),
        ),
        tensor_qtypes=(("F32", 506), ("BF16", 303)),
        tensor_count=809,
        parity_test="TestMuseGlimmerVisionEncoder.test_matches_independent_numpy_reference",
    ),
    MMProjArtifactPin(
        artifact_id="qwen3-vl-projector-f16",
        repository="bartowski/Qwen_Qwen3-VL-2B-Instruct-GGUF",
        revision="e84f8ae7ffee8b04793a4ed771609e2b61d3f3cf",
        filename="mmproj-Qwen_Qwen3-VL-2B-Instruct-f16.gguf",
        size=819_394_848,
        lfs_sha256="8c3f6a56979a1ce7056b9a20be6cf6b6f6ad4837aa3da532b5afcfcfd1faa38b",
        projector_types=("qwen3vl_merger",),
        paired_text_architecture="qwen3vl",
        paired_text_target="Qwen_Qwen3-VL-2B-Instruct-Q4_K_M.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.projection_dim", 2048),
            ("clip.vision.block_count", 24),
            ("clip.vision.patch_size", 16),
            ("clip.vision.spatial_merge_size", 2),
            ("clip.vision.is_deepstack_layers", (5, 11, 17)),
        ),
        tensor_qtypes=(("F32", 210), ("F16", 106)),
        tensor_count=316,
        parity_test="test_qwen3vl_projector_matches_transformers",
        processor_repository="Qwen/Qwen3-VL-2B-Instruct",
        processor_revision="89644892e4d85e24eaac8bacfd4f463576704203",
        processor_files=(
            "chat_template.json",
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ),
        processor_class="Qwen3VLProcessor",
        processor_contract=(
            ("pixel_values", "float32[total_image_patches,1536]"),
            ("image_grid_thw", "int64[num_images,3]"),
            ("pixel_values_videos", "float32[total_video_patches,1536]"),
            ("video_grid_thw", "int64[num_videos,3]"),
            ("output_rows", "sum(T*H*W/4) in merge-block-major order"),
            ("output_width", "2048 final + 3x2048 DeepStack features"),
            ("decoder_positions", "int64[4,batch,sequence] MRoPE; y/x section order"),
            ("empty_media", "omit pixel and grid keys; do not invoke vision graph"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="qwen3-audio-projector-bf16",
        repository="ggml-org/Qwen3-ASR-0.6B-GGUF",
        revision="928ab958557df9aa2ef1c93e0e83c7ad0933fae2",
        filename="mmproj-Qwen3-ASR-0.6B-bf16.gguf",
        size=378_575_520,
        lfs_sha256="dae36c855f9a82a8916bea2238b24bda69a39d8da8b2f46dee7c103775656039",
        projector_types=("qwen3a",),
        paired_text_architecture="qwen3vl",
        paired_text_target="Qwen3-ASR-0.6B-bf16.gguf",
        metadata=(
            ("clip.audio.embedding_length", 896),
            ("clip.audio.projection_dim", 1024),
            ("clip.audio.block_count", 18),
            ("clip.audio.num_mel_bins", 128),
        ),
        tensor_qtypes=(("F32", 188), ("BF16", 110), ("F16", 4)),
        tensor_count=302,
        parity_test="test_qwen3a_projector_matches_transformers",
        processor_repository="Qwen/Qwen3-ASR-0.6B",
        processor_revision="5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        processor_files=(
            "chat_template.json",
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ),
        processor_class="Qwen3ASRProcessor",
        processor_contract=(
            ("input_features", "float32[1,128,frames_multiple_of_100]"),
            ("input_features_mask", "int32[1,frames_multiple_of_100]"),
            ("windowing", "at most 800 mel frames, right-pad to a multiple of 100"),
            ("output_rows", "13 * (padded_frames / 100)"),
            ("ordering", "window-major, then chunk-major"),
            ("empty_media", "do not invoke audio graph"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="qwen2-audio-projector-f16",
        repository="mradermacher/Qwen2-Audio-7B-Instruct-GGUF",
        revision="e1e68850ba33e38eafbc3817919c318d9c7e757b",
        filename="Qwen2-Audio-7B-Instruct.mmproj-f16.gguf",
        size=1_289_301_536,
        lfs_sha256="b52435dead2956f1fc113818c3b5ceb42a940cb487e59163cb1ffc69cae69347",
        projector_types=("qwen2a",),
        paired_text_architecture="qwen2",
        paired_text_target="Qwen2-Audio-7B-Instruct.Q4_K_M.gguf",
        metadata=(
            ("clip.audio.embedding_length", 1280),
            ("clip.audio.projection_dim", 4096),
            ("clip.audio.block_count", 32),
            ("clip.audio.num_mel_bins", 128),
        ),
        tensor_qtypes=(("F32", 294), ("F16", 195)),
        tensor_count=489,
        parity_test="test_qwen2a_projector_matches_transformers",
        processor_repository="Qwen/Qwen2-Audio-7B-Instruct",
        processor_revision="0a095220c30b7b31434169c3086508ef3ea5bf0a",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ),
        processor_class="Qwen2AudioProcessor",
        processor_contract=(
            ("input_features", "float32[1,128,3000]"),
            ("feature_attention_mask", "int32[1,3000]; graph consumes padded chunk"),
            ("output_rows", "750 per 30-second chunk"),
            ("ordering", "audio-major fixed chunks"),
            ("empty_media", "omit input_features; do not invoke audio graph"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="qwen25-omni-projector-f16",
        repository="ggml-org/Qwen2.5-Omni-3B-GGUF",
        revision="75f1b73b657a50f5092502799457ccb4a4a1f9df",
        filename="mmproj-Qwen2.5-Omni-3B-f16.gguf",
        size=2_623_983_328,
        lfs_sha256="f6d9276e9fa4f060c7abdbe886786cf31a8911b62770f8a54b7581b7b99fa27e",
        projector_types=("qwen2.5o", "qwen2.5vl_merger", "qwen2a"),
        paired_text_architecture="qwen2vl",
        paired_text_target="Qwen2.5-Omni-3B-Q4_K_M.gguf",
        metadata=(
            ("clip.projector_type", "qwen2.5o"),
            ("clip.vision.embedding_length", 1280),
            ("clip.vision.projection_dim", 2048),
            ("clip.vision.block_count", 32),
            ("clip.vision.n_wa_pattern", 8),
            ("clip.audio.embedding_length", 1280),
            ("clip.audio.projection_dim", 2048),
            ("clip.audio.block_count", 32),
        ),
        tensor_qtypes=(("F32", 586), ("F16", 422)),
        tensor_count=1008,
        parity_test="test_qwen25o_alias_builds_distinct_vision_and_audio_graphs",
        processor_repository="Qwen/Qwen2.5-Omni-3B",
        processor_revision="f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        processor_files=(
            "chat_template.json",
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ),
        processor_class="Qwen2_5OmniProcessor",
        processor_contract=(
            ("vision", "Qwen2.5-VL packed image/video patch ABI"),
            ("audio_processor", "float32[1,128,30000] plus int32 feature mask"),
            ("audio_graph", "split processor rows into 3000-frame qwen2a chunks"),
            ("roles", "separate vision_encoder and audio_encoder graphs"),
            ("empty_media", "invoke only the component for each present modality"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="glm4v-projector-f16",
        repository="mradermacher/GLM-OCR-GGUF",
        revision="3c1e642c0fa5df64831f0b04f3c674b57ce341af",
        filename="GLM-OCR.mmproj-f16.gguf",
        size=869_018_080,
        lfs_sha256="fe5805b3b70f3174d25a912b8d197569eaa8e1e3e6d9777a385b8cc4c622af6c",
        projector_types=("glm4v",),
        paired_text_architecture="glm4",
        paired_text_target="GLM-OCR.Q4_K_M.gguf",
        metadata=(
            ("clip.projector_type", "glm4v"),
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.projection_dim", 1536),
            ("clip.vision.block_count", 24),
            ("clip.vision.patch_size", 14),
        ),
        tensor_qtypes=(("F32", 221), ("F16", 127)),
        tensor_count=348,
        parity_test="test_glm4v_projector_matches_transformers",
        processor_repository="zai-org/GLM-OCR",
        processor_revision="ca5d8b3e287e52589e37c28385d9655ee4372f9d",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ),
        processor_class="Glm46VProcessor",
        processor_contract=(
            ("pixel_values", "float32[total_patches,1176]"),
            ("image_grid_thw", "int64[num_images,3]"),
            ("output_rows", "sum(T*H*W/4)"),
            ("ordering", "batch-major images, merge-block-major 2x2 patches"),
            ("empty_media", "omit pixel and grid keys; do not invoke vision graph"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="ultravox-v0.5-f16",
        repository="ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        revision="5390c7c41cbd6f261f7f205fc0c5ae61bbdca650",
        filename="mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf",
        size=1_371_123_616,
        lfs_sha256="b34dde1835752949d6b960528269af93c92fec91c61ea0534fcc73f96c1ed8b2",
        projector_types=("ultravox",),
        paired_text_architecture="llama",
        paired_text_target="fixie-ai/ultravox-v0_5-llama-3_2-1b",
        metadata=(
            ("clip.projector_type", "ultravox"),
            ("clip.audio.embedding_length", 1280),
            ("clip.audio.projection_dim", 4096),
            ("clip.audio.block_count", 32),
            ("clip.audio.projector.stack_factor", 8),
        ),
        tensor_qtypes=(("F16", 196), ("F32", 295)),
        tensor_count=491,
        parity_test="test_ultravox_projector_matches_independent_swapped_swiglu_reference",
        processor_repository="fixie-ai/ultravox-v0_5-llama-3_2-1b",
        processor_revision="b95bec8ab291eeb04b5cd600dd473377f6b79026",
        processor_class="UltravoxProcessor",
        processor_contract=(("audio", "float32[3000,128] Whisper log-mel at 16 kHz"),),
    ),
    MMProjArtifactPin(
        artifact_id="music-flamingo-bf16",
        repository="henry1477/music-flamingo-gguf",
        revision="a059053433697011c6928b1962110040f4bcb4d0",
        filename="mmproj-music-flamingo-bf16.gguf",
        size=1_324_506_912,
        lfs_sha256="d4be69ed65f25dae97062febd44f9f41c0f6b14178f1cfb530fd894d595a4f94",
        projector_types=("musicflamingo",),
        paired_text_architecture="qwen2",
        paired_text_target="nvidia/music-flamingo-hf",
        metadata=(
            ("clip.projector_type", "musicflamingo"),
            ("clip.audio.embedding_length", 1280),
            ("clip.audio.projection_dim", 3584),
            ("clip.audio.block_count", 32),
        ),
        tensor_qtypes=(("BF16", 194), ("F32", 297)),
        tensor_count=491,
        parity_test="test_whisper_gelu_projectors_match_independent_torch_reference",
        processor_repository="nvidia/music-flamingo-hf",
        processor_revision="35a2c9071753ee075b0f7fc2fd81151c21389530",
        processor_class="AudioFlamingo3Processor",
        processor_contract=(("audio", "float32[3000,128] Whisper log-mel at 16 kHz"),),
    ),
    MMProjArtifactPin(
        artifact_id="lfm2.5-audio-1.5b-f16",
        repository="LiquidAI/LFM2.5-Audio-1.5B-GGUF",
        revision="7d525f883a077e20afb782f2ff618edcae0e39e4",
        filename="mmproj-LFM2.5-Audio-1.5B-F16.gguf",
        size=458_806_624,
        lfs_sha256="71330d7820768417d950f2dce42227896c7f6146917453957a63ba765decf621",
        projector_types=("lfm2a",),
        paired_text_architecture="lfm2",
        paired_text_target="LiquidAI/LFM2.5-Audio-1.5B",
        metadata=(
            ("clip.projector_type", "lfm2a"),
            ("clip.audio.embedding_length", 512),
            ("clip.audio.projection_dim", 2048),
            ("clip.audio.block_count", 17),
        ),
        tensor_qtypes=(("F16", 157), ("F32", 493)),
        tensor_count=650,
        parity_test="test_lfm2a_adapter_matches_independent_layernorm_gelu_reference",
        processor_repository="LiquidAI/LFM2.5-Audio-1.5B",
        processor_revision="c362a0625dfe45aa588dce5f0ada28a7e5707628",
        processor_class="Lfm2AudioProcessor",
        processor_contract=(("audio", "float32[T,128] normalized natural-log mel at 16 kHz"),),
    ),
    MMProjArtifactPin(
        artifact_id="granite-speech-4.1-2b-f16",
        repository="ibm-granite/granite-speech-4.1-2b-GGUF",
        revision="8267dad2adc84209b0efd2702ec68a98356125eb",
        filename="mmproj-model-f16.gguf",
        size=1_159_354_752,
        lfs_sha256="0d3615076cbe1d35c3f60c43a60a4047b3e2eeee1b2c233580be60186faab5c5",
        projector_types=("granite_speech",),
        paired_text_architecture="granite",
        paired_text_target="ibm-granite/granite-speech-4.1-2b",
        metadata=(
            ("clip.projector_type", "granite_speech"),
            ("clip.audio.embedding_length", 1024),
            ("clip.audio.projection_dim", 2048),
            ("clip.audio.block_count", 16),
            ("clip.audio.num_mel_bins", 160),
        ),
        tensor_qtypes=(("F16", 152), ("F32", 407)),
        tensor_count=559,
        parity_test="test_granite_speech_shaw_attention_matches_independent_numpy_reference",
        processor_repository="ibm-granite/granite-speech-4.1-2b",
        processor_revision="de575db64086f84fdc79da4932d1076e965bc546",
        processor_class="GraniteSpeechProcessor",
        processor_contract=(("audio", "float32[T,160] paired-frame features at 16 kHz"),),
    ),
    *(MMProjArtifactPin(**record) for record in REMAINING_MMPROJ_ARTIFACT_RECORDS),
    MMProjArtifactPin(
        artifact_id="deepseek-ocr-bf16",
        repository="sabafallah/DeepSeek-OCR-GGUF",
        revision="d26779bcd1cb301fec3ff82adc672f18384776fc",
        filename="mmproj-deepseek-ocr-bf16.gguf",
        size=826_425_472,
        lfs_sha256="4caeed8b6c3c7d25dfebccfdb5cf34d6ae540ef4dc4fa2b9842b69cfa50ecbe2",
        projector_types=("deepseekocr",),
        paired_text_architecture="deepseek2-ocr",
        paired_text_target="deepseek-ocr-Q4_K_M.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.block_count", 24),
            ("clip.vision.sam.block_count", 12),
            ("clip.vision.projection_dim", 1280),
        ),
        tensor_qtypes=(("F32", 331), ("BF16", 145)),
        tensor_count=476,
        parity_test=(
            "test_deepseek_clip_stage_matches_quick_gelu_reference + "
            "test_deepseek_projector_matches_linear_concatenation"
        ),
        paired_text_repository="sabafallah/DeepSeek-OCR-GGUF",
        paired_text_revision="d26779bcd1cb301fec3ff82adc672f18384776fc",
        paired_text_size=1_950_326_592,
        processor_repository="deepseek-ai/DeepSeek-OCR",
        processor_revision="9f30c71f441d010e5429c532364a86705536c53a",
        processor_files=("config.json", "processor_config.json", "modeling_deepseekocr.py"),
        processor_class="DeepseekVLV2Processor",
        processor_contract=(
            ("global_view", "float32[1,3,1024,1024]"),
            ("local_rows", "float32[num_rows,3,640,640*num_tiles_per_row]"),
            ("ordering", "local tile rows followed by the global overview"),
            ("empty_media", "reject"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="deepseek-ocr2-bf16",
        repository="sabafallah/DeepSeek-OCR-2-GGUF",
        revision="d08e5af400c64fa8a9b89b04ba373b600b02e05d",
        filename="mmproj-deepseek-ocr-2-bf16.gguf",
        size=929_037_632,
        lfs_sha256="b65e8460acc82dd4e8546206c5abd1abed9c5c582223731ce1585689d95f5cdb",
        projector_types=("deepseekocr2",),
        paired_text_architecture="deepseek2-ocr",
        paired_text_target="deepseek-ocr-2-Q4_K_M.gguf",
        metadata=(
            ("clip.vision.embedding_length", 896),
            ("clip.vision.block_count", 24),
            ("clip.vision.sam.block_count", 12),
            ("clip.vision.projection_dim", 1280),
        ),
        tensor_qtypes=(("F32", 254), ("BF16", 219)),
        tensor_count=473,
        parity_test="test_deepseek_ocr2_query_mask_matches_independent_reference",
        paired_text_repository="sabafallah/DeepSeek-OCR-2-GGUF",
        paired_text_revision="d08e5af400c64fa8a9b89b04ba373b600b02e05d",
        paired_text_size=1_950_326_688,
        processor_repository="deepseek-ai/DeepSeek-OCR-2",
        processor_revision="aaa02f3811945a91062062994c5c4a3f4c0af2b0",
        processor_files=("config.json", "processor_config.json", "modeling_deepseekocr2.py"),
        processor_class="DeepseekVLV2Processor",
        processor_contract=(
            ("global_view", "float32[1,3,1024,1024] -> 257 feature rows"),
            ("local_tiles", "float32[num_tiles,3,768,768] -> 144 rows per tile"),
            ("ordering", "local tiles followed by the global overview"),
            ("empty_media", "reject"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="dots-ocr-f16",
        repository="ggml-org/dots.ocr-GGUF",
        revision="2c093a32ca360a396bc6d87d60408636130b9d9b",
        filename="mmproj-dots.ocr-f16.gguf",
        size=2_526_296_992,
        lfs_sha256="f462429dbf41729379df4252f84331c9646d80e53f092ccccfd2cb922a3b544e",
        projector_types=("dots_ocr",),
        paired_text_architecture="qwen2",
        paired_text_target="dots.ocr-Q8_0.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1536),
            ("clip.vision.block_count", 42),
            ("clip.vision.projector.scale_factor", 2),
            ("clip.vision.projection_dim", 1536),
        ),
        tensor_qtypes=(("F32", 92), ("BF16", 212)),
        tensor_count=304,
        parity_test="test_dots_vision_dense_route_matches_independent_reference",
        paired_text_repository="ggml-org/dots.ocr-GGUF",
        paired_text_revision="2c093a32ca360a396bc6d87d60408636130b9d9b",
        paired_text_size=1_894_530_272,
        processor_repository="dots-studio/dots.ocr",
        processor_revision="c0111ce6bc07803dbc267932ffef0ae3a51dc951",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "configuration_dots.py",
        ),
        processor_class="DotsVLProcessor",
        processor_contract=(
            ("pixel_values", "float32[total_patches,588]"),
            ("image_grid_thw", "int64[num_images,3]"),
            ("ordering", "batch-major 2x2 merge-block patch order"),
            ("empty_media", "omit pixel and grid keys"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="dots3note-prev-f16",
        repository="ggml-org/dots3-note-prev-GGUF",
        revision="e5e7f6692337c0782c2bb4e2395174fd99879448",
        filename="mmproj-dots3-note-prev-F16.gguf",
        size=15_546_328_640,
        lfs_sha256="5bfec6cc2e2fa8ffcc70fc89866345640a297e7046e0913fd4899a3ec06f5ace",
        projector_types=("dots3note_v", "dots3note_a"),
        paired_text_architecture="dots3note",
        paired_text_target="IQ2_S/dots3-note-prev-IQ2_S-00001-of-00003.gguf",
        metadata=(
            ("clip.vision.block_count", 42),
            ("clip.vision.expert_used_count", 2),
            ("clip.audio.block_count", 32),
            ("clip.audio.num_mel_bins", 128),
        ),
        tensor_qtypes=(("F32", 477), ("F16", 439)),
        tensor_count=916,
        parity_test=(
            "test_dots_sigmoid_moe_uses_bias_only_for_selection + "
            "test_dots3note_audio_route_matches_partial_rope_reference"
        ),
        paired_text_repository="ggml-org/dots3-note-prev-GGUF",
        paired_text_revision="e5e7f6692337c0782c2bb4e2395174fd99879448",
        paired_text_size=5_940_800,
        processor_repository="dots-studio/dots3-note-prev",
        processor_revision="1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b",
        processor_files=("config.json", "preprocessor_config.json", "chat_template.jinja"),
        processor_class="Qwen2VLImageProcessor + mtmd_audio_preprocessor_dots3note",
        processor_contract=(
            ("vision_pixel_values", "float32[total_patches,588]"),
            ("vision_grid_thw", "int64[num_images,3]"),
            ("audio_waveform", "float32 mono 16kHz split into <=60s chunks"),
            ("audio_features", "float32[1,128,num_valid_frames]"),
            ("empty_media", "omit absent modality; reject an empty supplied modality"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="paddleocr-vl-1.6-bf16",
        repository="PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
        revision="511b09642bb324401f15f97cc23bc67e8f0a291d",
        filename="PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        size=881_770_560,
        lfs_sha256="204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a",
        projector_types=("paddleocr",),
        paired_text_architecture="paddleocr",
        paired_text_target="PaddleOCR-VL-1.6-GGUF.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.block_count", 27),
            ("clip.vision.projection_dim", 1024),
            ("clip.vision.patch_size", 14),
        ),
        tensor_qtypes=(("F32", 279), ("BF16", 164)),
        tensor_count=443,
        parity_test="test_paddleocr_route_matches_raster_position_and_merger_reference",
        paired_text_repository="PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
        paired_text_revision="511b09642bb324401f15f97cc23bc67e8f0a291d",
        paired_text_size=935_769_056,
        processor_repository="PaddlePaddle/PaddleOCR-VL-1.6",
        processor_revision="c5630abae1d940eafe0697512a0325494b02ab42",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "image_processing_paddleocr_vl.py",
            "processing_paddleocr_vl.py",
        ),
        processor_class="PaddleOCRVLProcessor",
        processor_contract=(
            ("pixel_values", "float32[total_patches,3,14,14]"),
            ("image_grid_thw", "int64[num_images,3]"),
            ("ordering", "batch-major raster patch order"),
            ("empty_media", "omit pixel and grid keys"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="lightonocr-1b-1025-f16",
        repository="noctrex/LightOnOCR-1B-1025-GGUF",
        revision="fe9d27bebcd975de319b2129a700791e6c9e00ae",
        filename="mmproj-F16.gguf",
        size=819_312_608,
        lfs_sha256="af55bc472ee9b5c409b4545033b5d1810b2a9cd799d53922ad93e5069542cd72",
        projector_types=("lightonocr",),
        paired_text_architecture="qwen3",
        paired_text_target="LightOnOCR-1B-1025-Q4_K_S.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.block_count", 24),
            ("clip.vision.spatial_merge_size", 2),
            ("clip.vision.projection_dim", 1024),
        ),
        tensor_qtypes=(("F32", 50), ("F16", 172)),
        tensor_count=222,
        parity_test="test_lighton_projector_matches_unfold_reference",
        paired_text_repository="noctrex/LightOnOCR-1B-1025-GGUF",
        paired_text_revision="fe9d27bebcd975de319b2129a700791e6c9e00ae",
        paired_text_size=470_781_536,
        processor_repository="lightonai/LightOnOCR-1B-1025",
        processor_revision="7e3e7b0cb83e237e7d237af5a583a002ea632547",
        processor_files=("config.json", "preprocessor_config.json", "processor_config.json"),
        processor_class="LightOnOCRProcessor",
        processor_contract=(
            ("pixel_values", "float32[1,3,height,width], height/width multiples of 28"),
            ("image_sizes", "int64[1,2] original height and width"),
            ("ordering", "single image per encoder invocation"),
            ("empty_media", "omit pixel_values and image_sizes"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="youtu-vl-4b-bf16",
        repository="tencent/Youtu-VL-4B-Instruct-GGUF",
        revision="1b7e295135d85a169d93823aa4215faf2c427092",
        filename="mmproj-Youtu-VL-4b-Instruct-BF16.gguf",
        size=893_397_344,
        lfs_sha256="1bcb2b7a99687be9a47e9bf27ee96237b73fe94b6fda938b17b18e7d4f92f9f2",
        projector_types=("youtuvl",),
        paired_text_architecture="deepseek2",
        paired_text_target="Youtu-VL-4B-Instruct-Q8_0.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.block_count", 27),
            ("clip.vision.window_size", 256),
            ("clip.vision.projection_dim", 2560),
        ),
        tensor_qtypes=(("F32", 277), ("BF16", 164)),
        tensor_count=441,
        parity_test="test_youtuvl_route_matches_merge_ordered_reference",
        paired_text_repository="tencent/Youtu-VL-4B-Instruct-GGUF",
        paired_text_revision="1b7e295135d85a169d93823aa4215faf2c427092",
        paired_text_size=5_211_323_488,
        processor_repository="tencent/Youtu-VL-4B-Instruct",
        processor_revision="8d30a0e49662a1d628a472b12df264dbcd768753",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "image_processing_siglip2_fast.py",
            "processing_youtu_vl.py",
        ),
        processor_class="YoutuVLProcessor",
        processor_contract=(
            ("pixel_values", "float32[1,total_patches,768]"),
            ("pixel_attention_mask", "int32[1,total_patches], all valid for one image"),
            ("spatial_shapes", "int64[1,2] patch height and width"),
            ("ordering", "2x2 merge-block-major patch rows"),
            ("empty_media", "omit all image keys"),
        ),
    ),
    MMProjArtifactPin(
        artifact_id="granite4-vision-4.1-f16",
        repository="ibm-granite/granite-vision-4.1-4b-GGUF",
        revision="b1fa14294b0f5cac04c43076d1c4574091abf117",
        filename="mmproj-model-f16.gguf",
        size=1_162_347_936,
        lfs_sha256="573dd2579f6043649299f0b2225000a5691d92f320aabe909fb4c6e75450cad2",
        projector_types=("granite4_vision",),
        paired_text_architecture="granite",
        paired_text_target="granite-vision-4.1-4b-Q4_K_M.gguf",
        metadata=(
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.block_count", 27),
            ("clip.vision.projector.query_side", 4),
            ("clip.vision.projector.window_side", 8),
        ),
        tensor_qtypes=(("F32", 459), ("F16", 251)),
        tensor_count=710,
        parity_test="test_granite_window_qformer_matches_independent_reference",
        paired_text_repository="ibm-granite/granite-vision-4.1-4b-GGUF",
        paired_text_revision="b1fa14294b0f5cac04c43076d1c4574091abf117",
        paired_text_size=2_099_510_240,
        processor_repository="ibm-granite/granite-vision-4.1-4b",
        processor_revision="37d591f06319e8f1638b5adcf58bdf50e0f84f7a",
        processor_files=(
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "processing.py",
        ),
        processor_class="Granite4VisionProcessor",
        processor_contract=(
            ("pixel_values", "float32[1,num_tiles,3,384,384]"),
            ("image_sizes", "int64[1,2] original height and width"),
            ("tile_grid", "int64[2] derived from the selected image_grid_pinpoint"),
            ("ordering", "global overview followed by row-major any-resolution tiles"),
            ("empty_media", "omit pixel_values and image_sizes"),
        ),
    ),
)

_SOURCE_EVIDENCE_INDEX: Mapping[str, MMProjSourceEvidence] = MappingProxyType(
    {evidence.evidence_id: evidence for evidence in MMPROJ_SOURCE_EVIDENCE}
)


def iter_projector_specs() -> tuple[ProjectorSpec, ...]:
    """Return the exact 60-string pinned projector census."""
    return _SPECS


def iter_projector_source_evidence() -> tuple[MMProjSourceEvidence, ...]:
    """Return immutable source proofs for route-specific sidecar semantics."""
    return MMPROJ_SOURCE_EVIDENCE


def projector_source_evidence(evidence_id: str) -> MMProjSourceEvidence | None:
    """Return one immutable source proof by ID."""
    return _SOURCE_EVIDENCE_INDEX.get(evidence_id)


def get_projector_spec(projector_type: str) -> ProjectorSpec:
    """Return one projector spec or fail with an actionable pinned-census error."""
    try:
        return _INDEX[projector_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown clip projector type {projector_type!r}; it is not one of the "
            f"60 strings at llama.cpp {LLAMA_CPP_MMPROJ_SHA}."
        ) from exc


def supported_projector_types() -> tuple[str, ...]:
    """Return projector strings that may reach graph construction."""
    return tuple(spec.projector_type for spec in _SPECS if spec.is_importable)


def projector_type_for_modality(metadata: Mapping[str, Any], modality: MMProjModality) -> str:
    """Resolve a modality override before the global projector fallback."""
    projector_type = metadata.get(f"clip.{modality.value}.projector_type")
    if not projector_type:
        projector_type = metadata.get("clip.projector_type")
    if not isinstance(projector_type, str) or not projector_type:
        raise ValueError(
            f"clip mmproj has an active {modality.value} encoder but neither "
            f"'clip.projector_type' nor 'clip.{modality.value}.projector_type' is set."
        )
    return projector_type
