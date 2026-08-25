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
    "MMPROJ_ARTIFACT_PINS",
    "ClipMetadataField",
    "CompanionTensorSpec",
    "MMProjArtifactPin",
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
class ProjectorSpec:
    """Capabilities and exact loader closure for one serialized projector type."""

    projector_type: str
    enum_name: str
    modalities: frozenset[MMProjModality]
    target_architectures: frozenset[str] = frozenset()
    metadata: Support = Support.DEFERRED
    tensor_map: Support = Support.DEFERRED
    graph: Support = Support.DEFERRED
    runtime: Support = Support.DEFERRED
    reason: str | None = None
    builder: str | None = None
    required_metadata: tuple[str, ...] = ()
    required_top_tensors: tuple[str, ...] = ()
    optional_top_tensors: tuple[str, ...] = ()
    block_prefix: str | None = None
    block_suffixes: tuple[str, ...] = ()
    companion_tensors: tuple[CompanionTensorSpec, ...] = ()
    tensor_roles: tuple[tuple[str, MMProjTensorRole], ...] = ()
    real_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        verdicts = (self.metadata, self.tensor_map, self.graph, self.runtime)
        if any(verdict is not Support.SUPPORTED for verdict in verdicts) and not self.reason:
            raise ValueError(f"{self.projector_type}: unsupported capability needs a reason")
        if self.is_importable:
            if not self.builder or not self.target_architectures:
                raise ValueError(
                    f"{self.projector_type}: importable projector needs builder and target"
                )
            if not self.required_metadata or not self.required_top_tensors:
                raise ValueError(
                    f"{self.projector_type}: importable projector needs an exact loader closure"
                )
            if not self.real_artifact_ids:
                raise ValueError(
                    f"{self.projector_type}: importable projector needs real artifact evidence"
                )
        elif any(
            (
                self.builder,
                self.required_metadata,
                self.required_top_tensors,
                self.optional_top_tensors,
                self.block_prefix,
                self.block_suffixes,
                self.companion_tensors,
                self.tensor_roles,
                self.real_artifact_ids,
            )
        ):
            raise ValueError(
                f"{self.projector_type}: deferred/rejected projector cannot expose loader data"
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


_SPECS: tuple[ProjectorSpec, ...] = (
    _deferred(
        "mlp",
        "PROJECTOR_TYPE_MLP",
        _VISION_BASE,
        "LLaVA MLP topology and class-token feature selection are not implemented by the GGUF builder.",
    ),
    _deferred(
        "ldp",
        "PROJECTOR_TYPE_LDP",
        _VISION_BASE,
        "MobileVLM LDP convolutional projector semantics are not implemented.",
    ),
    _deferred(
        "ldpv2",
        "PROJECTOR_TYPE_LDPV2",
        _VISION_BASE,
        "MobileVLM LDPv2 pooling/projector semantics are not implemented.",
    ),
    _deferred(
        "resampler",
        "PROJECTOR_TYPE_MINICPMV",
        _VISION_BASE,
        "MiniCPM-V query resampler and positional interpolation are not implemented.",
    ),
    _deferred(
        "adapter",
        "PROJECTOR_TYPE_GLM_EDGE",
        _VISION_BASE,
        "GLM-Edge adapter tensor closure and graph are not implemented.",
    ),
    _deferred(
        "qwen2vl_merger",
        "PROJECTOR_TYPE_QWEN2VL",
        _VISION_BASE,
        "The existing HF Qwen2-VL graph is not wired to the pinned GGUF merger ABI.",
        target_architectures=frozenset({"qwen2vl"}),
    ),
    _deferred(
        "qwen2.5vl_merger",
        "PROJECTOR_TYPE_QWEN25VL",
        _VISION_BASE,
        "The Qwen2.5-VL merger/window ordering has no GGUF tensor-closure parity test.",
        target_architectures=frozenset({"qwen2vl"}),
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
    _deferred(
        "gemma3",
        "PROJECTOR_TYPE_GEMMA3",
        _VISION_BASE,
        "Gemma3 mmproj feature selection and projector tensor map are not implemented.",
    ),
    _deferred(
        "gemma3nv",
        "PROJECTOR_TYPE_GEMMA3NV",
        _VISION_BASE,
        "Gemma3n vision sidecar routing is not implemented.",
        target_architectures=frozenset({"gemma3n"}),
    ),
    _deferred(
        "gemma3na",
        "PROJECTOR_TYPE_GEMMA3NA",
        _AUDIO_BASE,
        "Gemma3n audio sidecar routing is not implemented.",
        target_architectures=frozenset({"gemma3n"}),
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
    _deferred(
        "gemma4a",
        "PROJECTOR_TYPE_GEMMA4A",
        _AUDIO_BASE,
        "The sidecar carries a.pre_encode tensors that the current audio map drops, and independent Conformer parity is not established.",
    ),
    _deferred(
        "gemma4uv",
        "PROJECTOR_TYPE_GEMMA4UV",
        _VISION_BASE,
        "Encoder-free unified Gemma4 vision sidecars use a different patch embedder contract.",
    ),
    _deferred(
        "gemma4ua",
        "PROJECTOR_TYPE_GEMMA4UA",
        _AUDIO_BASE,
        "Encoder-free unified Gemma4 waveform embedding is not wired to GGUF.",
    ),
    _deferred(
        "phi4",
        "PROJECTOR_TYPE_PHI4",
        _VISION_BASE,
        "The Phi-4 vision projector exists for HF weights but has no pinned GGUF tensor closure.",
    ),
    _deferred(
        "idefics3",
        "PROJECTOR_TYPE_IDEFICS3",
        _VISION_BASE,
        "Idefics3 pixel-shuffle projector GGUF routing is not implemented.",
    ),
    _deferred(
        "pixtral",
        "PROJECTOR_TYPE_PIXTRAL",
        _VISION_BASE,
        "The Pixtral component has no pinned mmproj tensor mapping or positional-interpolation parity.",
        target_architectures=frozenset({"deepseek2", "llama", "mistral3", "mistral4"}),
    ),
    _deferred(
        "ultravox",
        "PROJECTOR_TYPE_ULTRAVOX",
        _AUDIO_BASE,
        "Whisper encoder plus Ultravox stack projector is not implemented.",
    ),
    _deferred(
        "internvl",
        "PROJECTOR_TYPE_INTERNVL",
        _VISION_BASE,
        "InternVL pixel-shuffle token ordering is not implemented for GGUF.",
    ),
    _deferred(
        "llama4",
        "PROJECTOR_TYPE_LLAMA4",
        _VISION_BASE,
        "Llama4 vision encoder and multimodal target package are out of scope.",
        target_architectures=frozenset({"llama4"}),
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

MMPROJ_ARTIFACT_PINS: tuple[MMProjArtifactPin, ...] = (
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
        parity_test="TestVisionEncoderBuildAndRun.test_matches_independent_numpy_reference",
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
    """Resolve the pinned global-key then modality-key projector fallback."""
    projector_type = metadata.get("clip.projector_type")
    if not projector_type:
        projector_type = metadata.get(f"clip.{modality.value}.projector_type")
    if not isinstance(projector_type, str) or not projector_type:
        raise ValueError(
            f"clip mmproj has an active {modality.value} encoder but neither "
            f"'clip.projector_type' nor 'clip.{modality.value}.projector_type' is set."
        )
    return projector_type
