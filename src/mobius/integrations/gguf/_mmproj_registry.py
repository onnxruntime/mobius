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
    "ClipMetadataField",
    "CompanionTensorSpec",
    "DeferredCompanionSpec",
    "MMProjArtifactAvailabilityPin",
    "MMProjArtifactPin",
    "MMProjSourceEvidence",
    "MMProjModelRole",
    "MMProjModality",
    "MMProjTensorRole",
    "ProjectorSpec",
    "get_projector_spec",
    "iter_projector_specs",
    "projector_type_for_modality",
    "supported_projector_types",
]

import dataclasses
import enum
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from mobius.integrations.gguf._spec import Support

LLAMA_CPP_MMPROJ_SHA = "8d9af256337d1a501250f9bbf4c0859a654bddd6"


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
class CompanionTensorSpec:
    """Exact tensor closure for a deferred modality sharing a supported sidecar."""

    modality: MMProjModality
    projector_type: str
    required_metadata: tuple[str, ...]
    required_top_tensors: tuple[str, ...]
    block_prefix: str
    block_suffixes: tuple[str, ...]


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
    auxiliary_tensor_patterns: tuple[str, ...] = ()
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
                self.auxiliary_tensor_patterns,
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
    ClipMetadataField("clip.vision.wa_pattern_mode"),
    ClipMetadataField("clip.vision.window_size"),
    ClipMetadataField("clip.minicpmv_version"),
    ClipMetadataField("clip.minicpmv_query_num", default="version-dependent"),
    ClipMetadataField("clip.vision.sam.head_count"),
    ClipMetadataField("clip.vision.sam.block_count"),
    ClipMetadataField("clip.vision.sam.embedding_length"),
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

_SPECS: tuple[ProjectorSpec, ...] = (
    ProjectorSpec(
        "mlp",
        "PROJECTOR_TYPE_MLP",
        _VISION_BASE,
        target_architectures=frozenset({"llama"}),
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
    _deferred(
        "qwen3vl_merger",
        "PROJECTOR_TYPE_QWEN3VL",
        _VISION_BASE,
        "The Qwen3-VL merger/window ordering has no GGUF tensor-closure parity test.",
        target_architectures=frozenset({"qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe"}),
    ),
    _deferred(
        "step3vl",
        "PROJECTOR_TYPE_STEP3VL",
        _VISION_BASE,
        "Step3-VL vision and projector graph are not implemented.",
    ),
    ProjectorSpec(
        projector_type="gemma3",
        enum_name="PROJECTOR_TYPE_GEMMA3",
        modalities=_VISION_BASE,
        target_architectures=frozenset({"gemma3"}),
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
    _deferred(
        "ultravox",
        "PROJECTOR_TYPE_ULTRAVOX",
        _AUDIO_BASE,
        "Whisper encoder plus Ultravox stack projector is not implemented.",
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
    _deferred(
        "qwen2a",
        "PROJECTOR_TYPE_QWEN2A",
        _AUDIO_BASE,
        "Qwen2 audio encoder/projector is not implemented.",
    ),
    _deferred(
        "qwen3a",
        "PROJECTOR_TYPE_QWEN3A",
        _AUDIO_BASE,
        "Qwen3 audio encoder/projector is not implemented.",
        target_architectures=frozenset({"qwen3vl", "qwen3vlmoe"}),
    ),
    _deferred(
        "glma",
        "PROJECTOR_TYPE_GLMA",
        _AUDIO_BASE,
        "GLM audio encoder/projector is not implemented.",
    ),
    _deferred(
        "qwen2.5o",
        "PROJECTOR_TYPE_QWEN25O",
        _VISION_BASE | _AUDIO_BASE,
        "This legacy string changes meaning by modality; accepting it would create a false alias.",
        target_architectures=frozenset({"qwen2vl"}),
    ),
    _deferred(
        "voxtral",
        "PROJECTOR_TYPE_VOXTRAL",
        _AUDIO_BASE,
        "Voxtral Whisper encoder/projector is not implemented.",
    ),
    _deferred(
        "meralion",
        "PROJECTOR_TYPE_MERALION",
        _AUDIO_BASE,
        "Meralion audio projector is not implemented.",
    ),
    _deferred(
        "musicflamingo",
        "PROJECTOR_TYPE_MUSIC_FLAMINGO",
        _AUDIO_BASE,
        "Music Flamingo audio projector is not implemented.",
    ),
    _deferred(
        "lfm2",
        "PROJECTOR_TYPE_LFM2",
        _VISION_BASE,
        "The existing LFM2-VL HF graph has no pinned mmproj tensor closure or component parity.",
    ),
    _deferred(
        "kimivl",
        "PROJECTOR_TYPE_KIMIVL",
        _VISION_BASE,
        "Kimi-VL vision/projector graph is not implemented.",
    ),
    _deferred(
        "paddleocr",
        "PROJECTOR_TYPE_PADDLEOCR",
        _VISION_BASE,
        "PaddleOCR vision/projector graph is not implemented.",
        target_architectures=frozenset({"paddleocr"}),
    ),
    _deferred(
        "lightonocr",
        "PROJECTOR_TYPE_LIGHTONOCR",
        _VISION_BASE,
        "LightOnOCR Pixtral variant has no exact tensor mapping.",
    ),
    _deferred(
        "cogvlm",
        "PROJECTOR_TYPE_COGVLM",
        _VISION_BASE,
        "CogVLM feature output differs from LLaVA and is not implemented.",
        target_architectures=frozenset({"cogvlm"}),
    ),
    _deferred(
        "janus_pro",
        "PROJECTOR_TYPE_JANUS_PRO",
        _VISION_BASE,
        "Janus-Pro vision/projector graph is not implemented.",
    ),
    _deferred(
        "dots_ocr",
        "PROJECTOR_TYPE_DOTS_OCR",
        _VISION_BASE,
        "DotsOCR vision merger is not implemented.",
    ),
    _deferred(
        "dots3note_v",
        "PROJECTOR_TYPE_DOTS3NOTE_V",
        _VISION_BASE,
        "Dots3Note vision pyramid MoE is not implemented.",
    ),
    _deferred(
        "dots3note_a",
        "PROJECTOR_TYPE_DOTS3NOTE_A",
        _AUDIO_BASE,
        "Dots3Note audio graph is not implemented.",
    ),
    _deferred(
        "deepseekocr",
        "PROJECTOR_TYPE_DEEPSEEKOCR",
        _VISION_BASE,
        "DeepSeek-OCR SAM/projector graph is not implemented.",
        target_architectures=frozenset({"deepseek2-ocr"}),
    ),
    _deferred(
        "deepseekocr2",
        "PROJECTOR_TYPE_DEEPSEEKOCR2",
        _VISION_BASE,
        "DeepSeek-OCR2 SAM/projector graph is not implemented.",
        target_architectures=frozenset({"deepseek2-ocr"}),
    ),
    _deferred(
        "lfm2a",
        "PROJECTOR_TYPE_LFM2A",
        _AUDIO_BASE,
        "LFM2 conformer audio graph has no GGUF tensor mapping.",
    ),
    _deferred(
        "glm4v",
        "PROJECTOR_TYPE_GLM4V",
        _VISION_BASE,
        "GLM4V downsampler and projector are not implemented.",
    ),
    _deferred(
        "youtuvl",
        "PROJECTOR_TYPE_YOUTUVL",
        _VISION_BASE,
        "YouTu-VL vision/projector graph is not implemented.",
    ),
    _deferred(
        "yasa2",
        "PROJECTOR_TYPE_YASA2",
        _VISION_BASE,
        "YASA2 vision/projector graph is not implemented.",
    ),
    _deferred(
        "kimik25",
        "PROJECTOR_TYPE_KIMIK25",
        _VISION_BASE,
        "Kimi K2.5 vision/projector graph is not implemented.",
    ),
    _deferred(
        "nemotron_v2_vl",
        "PROJECTOR_TYPE_NEMOTRON_V2_VL",
        _VISION_BASE,
        "Nemotron V2 VL vision/projector graph is not implemented.",
    ),
    _deferred(
        "exaone4_5",
        "PROJECTOR_TYPE_EXAONE4_5",
        _VISION_BASE,
        "EXAONE 4.5 vision merger is not implemented.",
    ),
    _deferred(
        "hunyuanvl",
        "PROJECTOR_TYPE_HUNYUANVL",
        _VISION_BASE,
        "HunyuanVL vision/projector graph is not implemented.",
        target_architectures=frozenset({"hunyuan_vl"}),
    ),
    _deferred(
        "minicpmv4_6",
        "PROJECTOR_TYPE_MINICPMV4_6",
        _VISION_BASE,
        "MiniCPM-V 4.6 SAM/resampler graph is not implemented.",
    ),
    _deferred(
        "granite_speech",
        "PROJECTOR_TYPE_GRANITE_SPEECH",
        _AUDIO_BASE,
        "Granite Speech audio encoder/projector is not implemented.",
    ),
    _deferred(
        "mimovl",
        "PROJECTOR_TYPE_MIMOVL",
        _VISION_BASE,
        "MiMo-VL vision/projector graph is not implemented.",
    ),
    _deferred(
        "minimax_m3",
        "PROJECTOR_TYPE_MINIMAX_M3",
        _VISION_BASE,
        "MiniMax M3 vision/projector graph is not implemented.",
    ),
    _deferred(
        "granite4_vision",
        "PROJECTOR_TYPE_GRANITE4_VISION",
        _VISION_BASE,
        "Granite 4 vision sidecar graph is not implemented.",
    ),
    _deferred(
        "mimo_audio",
        "PROJECTOR_TYPE_MIMO_AUDIO",
        _AUDIO_BASE,
        "MiMo audio RVQ/local-transformer graph is not implemented.",
    ),
    _deferred(
        "parakeet",
        "PROJECTOR_TYPE_PARAKEET",
        _AUDIO_BASE,
        "Parakeet audio encoder graph is not implemented.",
    ),
    _deferred(
        "qwen3tts_spkenc",
        "PROJECTOR_TYPE_QWEN3TTS_SPKENC",
        _AUDIO_BASE,
        "Qwen3-TTS speaker encoder graph is not implemented.",
    ),
    _rejected(
        "qwen3tts_gen",
        "PROJECTOR_TYPE_QWEN3TTS_GEN",
        _GEN_AUDIO_BASE,
        "Generated-audio decoder sidecars are not multimodal projectors and cannot be paired with a text target package.",
    ),
    _deferred(
        "pockettts_spkenc",
        "PROJECTOR_TYPE_POCKETTTS_SPKENC",
        _AUDIO_BASE,
        "PocketTTS speaker encoder graph is not implemented.",
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
)


def iter_projector_specs() -> tuple[ProjectorSpec, ...]:
    """Return the exact 60-string pinned projector census."""
    return _SPECS


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
