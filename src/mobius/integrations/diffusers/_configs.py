# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for diffusers models."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import onnx_ir as ir

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mobius._configs import ArchitectureConfig
    from mobius.integrations.onnx_genai import HierarchicalAudioWorkflowConfig

MINIMAX_MUSIC3_AUDIO_END_TOKEN_ID = 151670
MINIMAX_MUSIC3_AUDIO_CODE_OFFSET = 151675
MINIMAX_MUSIC3_SEMANTIC_VOCAB_SIZE = 16384
MINIMAX_MUSIC3_NUM_CODEBOOKS = 8
MINIMAX_MUSIC3_FEEDBACK_SCALE = MINIMAX_MUSIC3_NUM_CODEBOOKS**-0.5

# --- MiniMax Music 3 hierarchical-audio workflow defaults --------------------
# Runtime-behaviour facts the checkpoint author publishes for Music 3's nested
# autoregressive / residual-vector-quantized / flow-matching / vocoder pipeline.
# They cannot be read back from the exported neural graphs, so mobius owns them
# here as the canonical model registry -- exactly like the token constants above
# and every other ``from_diffusers`` default in this module. They are applied
# through :class:`MiniMaxMusic3WorkflowConfig`, which fills a typed, generic
# :class:`~mobius.integrations.onnx_genai.HierarchicalAudioWorkflowConfig`; the
# generic metadata writer that consumes that structure never sees a MiniMax
# literal, and a divergent checkpoint overrides any of these by field.

#: ``<|audio_cfg|>`` special token spliced into classifier-free-guidance rows.
MINIMAX_MUSIC3_UNCONDITIONAL_TOKEN_ID = 151654
#: Classifier-free-guidance scales for the global-semantic, local-codebook and
#: flow-matching stages respectively.
MINIMAX_MUSIC3_SEMANTIC_GUIDANCE_SCALE = 1.5
MINIMAX_MUSIC3_LOCAL_GUIDANCE_SCALE = 1.5
MINIMAX_MUSIC3_FLOW_GUIDANCE_SCALE = 1.7
#: Top-k used when sampling semantic tokens from the global decoder.
MINIMAX_MUSIC3_SAMPLING_TOP_K = 50
#: Chunked flow-matching plan: frames per chunk, chunk stride, solver steps and
#: the overlap latents carried and cropped between consecutive chunks.
MINIMAX_MUSIC3_CHUNK_FRAMES = 200
MINIMAX_MUSIC3_CHUNK_HOP = 100
MINIMAX_MUSIC3_FLOW_STEPS = 30
MINIMAX_MUSIC3_CARRY_LENGTH = 172
MINIMAX_MUSIC3_CROP_LEFT_LATENTS = 86
MINIMAX_MUSIC3_CROP_RIGHT_LATENTS = 258
#: Prompt-processor input-token and output-frame ceilings.
MINIMAX_MUSIC3_MAX_PROMPT_TOKENS = 5000
MINIMAX_MUSIC3_MAX_AUDIO_FRAMES = 9000
#: Delivered waveform sample rate in Hz (independent of the vocoder's native
#: rate, which the metadata writer reads from the vocoder component config).
MINIMAX_MUSIC3_TARGET_SAMPLE_RATE = 32000
#: Classifier-free-guidance row assembly: first prompt row replaced by the
#: unconditional token and the number of trailing prompt rows preserved.
MINIMAX_MUSIC3_UNCONDITIONAL_REPLACE_FROM = 1
MINIMAX_MUSIC3_UNCONDITIONAL_PRESERVE_TRAILING = 2
#: Fallback global-decoder context window used only when the source language
#: config omits ``max_position_embeddings`` (normally it is read from there).
MINIMAX_MUSIC3_GLOBAL_CONTEXT = 10240

#: Ordered prompt-assembly template. Literal chat/lyrics markup is interleaved
#: with request-field transforms; the writer emits it verbatim so the runtime
#: reconstructs Music 3's ``caption``/``lyrics`` prompt exactly.
MINIMAX_MUSIC3_PROMPT_SEGMENTS: tuple[dict, ...] = (
    {"literal": "<|im_start|><|caption_start|>"},
    {
        "field": "instructions",
        "transforms": (
            {"kind": "rewrite_delimited_tags", "open": "<|", "close": "|>"},
            {"kind": "strip_markdown"},
            {"kind": "collapse_newlines"},
        ),
    },
    {"literal": "<|caption_end|><|lyrics_start|>[start]\n"},
    {
        "field": "input",
        "transforms": (
            {"kind": "keep_leading_bracket_tags"},
            {"kind": "replace", "from": "] ", "to": "]\n"},
            {"kind": "replace", "from": " [", "to": "\n["},
            {"kind": "replace", "from": " ^ ", "to": "\n"},
            {"kind": "lowercase_bracket_tags"},
        ),
    },
    {"literal": "<|lyrics_end|><|im_end|><|audio_start|>"},
)


class CLIPTextConfig:
    """Adapter that builds an :class:`ArchitectureConfig` for a CLIP text encoder.

    Classic Stable Diffusion 1.x/2.x pipelines use a ``transformers``
    ``CLIPTextModel`` as their prompt encoder. Its ``text_encoder/config.json``
    uses transformers field names, which this adapter maps onto the generic
    :class:`mobius._configs.ArchitectureConfig` consumed by
    :class:`mobius.models.clip.CLIPTextModel`.
    """

    @classmethod
    def from_diffusers(cls, config: dict) -> ArchitectureConfig:
        """Create an :class:`ArchitectureConfig` from a diffusers text-encoder config.

        Args:
            config: Parsed ``text_encoder/config.json`` dictionary.

        Returns:
            An :class:`ArchitectureConfig` describing the CLIP text encoder. The
            transformers ``layer_norm_eps`` is mapped onto ``rms_norm_eps`` (the
            field the from-scratch CLIP layers read for their ``LayerNorm`` eps),
            and ``head_dim`` is derived from ``hidden_size / num_attention_heads``.
        """
        from mobius._configs import ArchitectureConfig

        if hasattr(config, "to_dict"):
            config = dict(config.items())
        hidden_size = config.get("hidden_size", 768)
        num_attention_heads = config.get("num_attention_heads", 12)
        return ArchitectureConfig(
            vocab_size=config.get("vocab_size", 49408),
            hidden_size=hidden_size,
            intermediate_size=config.get("intermediate_size", 3072),
            num_hidden_layers=config.get("num_hidden_layers", 12),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=hidden_size // num_attention_heads,
            max_position_embeddings=config.get("max_position_embeddings", 77),
            rms_norm_eps=config.get("layer_norm_eps", 1e-5),
            hidden_act=config.get("hidden_act", "quick_gelu"),
        )


class T5TextEncoderConfig:
    """Adapter for a T5 encoder embedded in a diffusers pipeline."""

    @classmethod
    def from_diffusers(cls, config: dict) -> ArchitectureConfig:
        """Create the native Mobius T5 configuration."""
        import transformers

        from mobius._configs import ArchitectureConfig

        fields = dict(config)
        fields.pop("architectures", None)
        fields.pop("transformers_version", None)
        fields.pop("torch_dtype", None)
        model_type = fields.pop("model_type", "t5")
        hf_config = transformers.AutoConfig.for_model(model_type, **fields)
        return ArchitectureConfig.from_transformers(hf_config)


class QwenImageTextEncoderConfig:
    """Adapter for the Qwen2.5-VL prompt encoder bundled with Qwen Image Edit."""

    @classmethod
    def from_diffusers(cls, config: dict) -> ArchitectureConfig:
        """Build the native Mobius Qwen2.5-VL configuration tree."""
        import transformers

        from mobius._configs import ArchitectureConfig

        if hasattr(config, "to_dict"):
            config = config.to_dict()
        fields = dict(config)
        model_type = fields.pop("model_type", "qwen2_5_vl")
        hf_config = transformers.AutoConfig.for_model(model_type, **fields)
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        return ArchitectureConfig.from_transformers(text_config, parent_config=hf_config)


class MiniMaxMusic3LanguageConfig:
    """Adapter for the Qwen3 global language model bundled with Music 3."""

    @classmethod
    def from_diffusers(cls, config: dict) -> ArchitectureConfig:
        import transformers

        from mobius._configs import ArchitectureConfig

        fields = dict(config)
        fields.pop("architectures", None)
        fields.pop("transformers_version", None)
        fields.pop("dtype", None)
        model_type = fields.pop("model_type", "qwen3")
        hf_config = transformers.AutoConfig.for_model(model_type, **fields)
        return ArchitectureConfig.from_transformers(hf_config)


class MiniMaxMusic3WorkflowConfig:
    """Adapter that builds Music 3's hierarchical-audio workflow description.

    Mirrors the ``from_diffusers`` component adapters above, but at the pipeline
    level: it maps the source pipeline's component configs plus the Mobius-owned
    Music 3 defaults declared at the top of this module onto the typed, generic
    :class:`~mobius.integrations.onnx_genai.HierarchicalAudioWorkflowConfig` that
    the ONNX GenAI metadata writer consumes.

    Owning the defaults here -- not in the writer and not in a checked-in JSON
    contract -- keeps the writer fully model-agnostic while letting mobius act as
    the canonical model-name -> model-information registry. The structural role
    map (``components``) and the global-decoder context window are derived from
    the built graphs and source configs; everything else comes from the named
    ``MINIMAX_MUSIC3_*`` defaults, so a divergent checkpoint can override any
    single field without touching the generic writer.
    """

    @classmethod
    def from_diffusers(
        cls,
        *,
        components: Mapping[str, str],
        component_configs: Mapping[str, dict],
    ) -> HierarchicalAudioWorkflowConfig:
        """Build the workflow config from structural roles and source configs.

        Args:
            components: Structural role -> exported graph name, produced by the
                builder from the graphs it actually emitted.
            component_configs: Component name -> parsed source config. The global
                decoder's ``max_position_embeddings`` supplies the context window;
                the remaining semantic values come from the Music 3 defaults.

        Returns:
            A populated :class:`HierarchicalAudioWorkflowConfig`.
        """
        from mobius.integrations.onnx_genai import HierarchicalAudioWorkflowConfig

        language_config = component_configs.get(components["global_decoder"], {})
        global_context = int(
            language_config.get("max_position_embeddings", MINIMAX_MUSIC3_GLOBAL_CONTEXT)
        )
        return HierarchicalAudioWorkflowConfig(
            components=dict(components),
            semantic_vocabulary_start=MINIMAX_MUSIC3_AUDIO_CODE_OFFSET,
            semantic_vocabulary_size=MINIMAX_MUSIC3_SEMANTIC_VOCAB_SIZE,
            stop_token_id=MINIMAX_MUSIC3_AUDIO_END_TOKEN_ID,
            unconditional_token_id=MINIMAX_MUSIC3_UNCONDITIONAL_TOKEN_ID,
            semantic_guidance_scale=MINIMAX_MUSIC3_SEMANTIC_GUIDANCE_SCALE,
            local_guidance_scale=MINIMAX_MUSIC3_LOCAL_GUIDANCE_SCALE,
            flow_guidance_scale=MINIMAX_MUSIC3_FLOW_GUIDANCE_SCALE,
            sampling_top_k=MINIMAX_MUSIC3_SAMPLING_TOP_K,
            chunk_frames=MINIMAX_MUSIC3_CHUNK_FRAMES,
            chunk_hop=MINIMAX_MUSIC3_CHUNK_HOP,
            flow_steps=MINIMAX_MUSIC3_FLOW_STEPS,
            carry_length=MINIMAX_MUSIC3_CARRY_LENGTH,
            crop_left_latents=MINIMAX_MUSIC3_CROP_LEFT_LATENTS,
            crop_right_latents=MINIMAX_MUSIC3_CROP_RIGHT_LATENTS,
            max_prompt_tokens=MINIMAX_MUSIC3_MAX_PROMPT_TOKENS,
            max_audio_frames=MINIMAX_MUSIC3_MAX_AUDIO_FRAMES,
            global_context=global_context,
            target_sample_rate=MINIMAX_MUSIC3_TARGET_SAMPLE_RATE,
            unconditional_replace_from=MINIMAX_MUSIC3_UNCONDITIONAL_REPLACE_FROM,
            unconditional_preserve_trailing=MINIMAX_MUSIC3_UNCONDITIONAL_PRESERVE_TRAILING,
            prompt_segments=[
                _copy_prompt_segment(segment) for segment in MINIMAX_MUSIC3_PROMPT_SEGMENTS
            ],
        )


def _copy_prompt_segment(segment: Mapping[str, Any]) -> dict:
    """Deep-copy a prompt-segment template entry into plain mutable dicts."""
    copied = dict(segment)
    if "transforms" in copied:
        copied["transforms"] = [dict(transform) for transform in copied["transforms"]]
    return copied


@dataclasses.dataclass
class DiffusersPipelineConfig:
    """Non-neural diffusers pipeline metadata retained on a ModelPackage."""

    source_model_id: str
    pipeline_class: str
    component_configs: dict[str, dict]
    scheduler_config: dict
    processor_config: dict
    #: A typed, model-agnostic workflow config supplied by an explicit
    #: ``build_diffusers_pipeline`` argument or by the pipeline's mobius config
    #: adapter. ``None`` when no authoritative config was resolved.
    workflow_config: HierarchicalAudioWorkflowConfig | None = None
    #: Structural workflow kind detected from the built graph topology, set even
    #: when no ``workflow_config`` could be resolved so metadata emission can
    #: fail closed with a targeted instruction instead of misclassifying the
    #: package.
    workflow_kind: str | None = None
    model_type: str = "diffusers"


@dataclasses.dataclass
class MiniMaxMusic3RVQConfig:
    """Configuration for the MiniMax Music 3 residual-codebook depth decoder."""

    hidden_size: int = 4096
    num_layers: int = 4
    num_attention_heads: int = 16
    intermediate_size: int = 6144
    audio_vocab_size: int = 1024
    num_codebooks: int = 8
    max_position_embeddings: int = 16
    dtype: ir.DataType = ir.DataType.FLOAT

    @classmethod
    def from_diffusers(cls, config: dict) -> MiniMaxMusic3RVQConfig:
        return cls(
            **{
                field.name: config[field.name]
                for field in dataclasses.fields(cls)
                if field.name in config
            }
        )


@dataclasses.dataclass
class MiniMaxMusic3ConditionConfig:
    """Configuration for the MiniMax Music 3 frame-condition encoder."""

    condition_hidden_dim: int = 4096
    num_condition_layers: int = 8
    out_dim: int = 2048
    input_sampling_rate: int = 24000
    input_hop_length: int = 960
    output_sampling_rate: int = 44100
    output_hop_length: int = 512
    dtype: ir.DataType = ir.DataType.FLOAT

    @classmethod
    def from_diffusers(cls, config: dict) -> MiniMaxMusic3ConditionConfig:
        return cls(
            **{
                field.name: config[field.name]
                for field in dataclasses.fields(cls)
                if field.name in config
            }
        )


@dataclasses.dataclass
class MiniMaxMusic3TransformerConfig:
    """Configuration for the MiniMax Music 3 1D flow-matching transformer."""

    in_channels: int = 128
    condition_dim: int = 2048
    num_layers: int = 36
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    ff_inner_dim: int = 8192
    rotary_dim: int = 32
    fourier_embedding_dim: int = 256
    dtype: ir.DataType = ir.DataType.FLOAT

    @classmethod
    def from_diffusers(cls, config: dict) -> MiniMaxMusic3TransformerConfig:
        return cls(
            **{
                field.name: config[field.name]
                for field in dataclasses.fields(cls)
                if field.name in config
            }
        )


@dataclasses.dataclass
class MiniMaxMusic3VocoderConfig:
    """Configuration for the MiniMax Music 3 stereo DAC-style vocoder."""

    latent_channels: int = 128
    decoder_input_dim: int = 1024
    decoder_hidden_dim: int = 1536
    upsampling_ratios: tuple[int, ...] = (8, 8, 4, 2)
    sampling_rate: int = 44100
    dtype: ir.DataType = ir.DataType.FLOAT

    @classmethod
    def from_diffusers(cls, config: dict) -> MiniMaxMusic3VocoderConfig:
        values = {
            field.name: config[field.name]
            for field in dataclasses.fields(cls)
            if field.name in config
        }
        if "upsampling_ratios" in values:
            values["upsampling_ratios"] = tuple(values["upsampling_ratios"])
        return cls(**values)


@dataclasses.dataclass
class VAEConfig:
    """Configuration for AutoencoderKL (VAE) models."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    sample_size: int = 256
    scaling_factor: float = 0.18215
    mid_block_add_attention: bool = True
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True

    @classmethod
    def from_diffusers(cls, config: dict) -> VAEConfig:
        """Create a VAEConfig from a diffusers config dict."""
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            latent_channels=config.get("latent_channels", 4),
            block_out_channels=tuple(config.get("block_out_channels", [128, 256, 512, 512])),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
            act_fn=config.get("act_fn", "silu"),
            sample_size=config.get("sample_size", 256),
            scaling_factor=config.get("scaling_factor", 0.18215),
            mid_block_add_attention=config.get("mid_block_add_attention", True),
            use_quant_conv=config.get("use_quant_conv", True),
            use_post_quant_conv=config.get("use_post_quant_conv", True),
        )


@dataclasses.dataclass
class UNet2DConfig:
    """Configuration for UNet2DConditionModel."""

    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    cross_attention_dim: int = 768
    attention_head_dim: int = 8
    act_fn: str = "silu"
    sample_size: int = 64
    # Per-block type names from diffusers. A block gets cross-attention only when
    # its type name contains ``CrossAttn`` (Stable Diffusion 1.x uses a plain
    # ``DownBlock2D`` for the last down block and ``UpBlock2D`` for the first up
    # block). ``None`` = every block has cross-attention (legacy behavior).
    down_block_types: tuple[str, ...] | None = None
    up_block_types: tuple[str, ...] | None = None
    mid_block_type: str | None = "UNetMidBlock2DCrossAttn"
    addition_embed_type: str | None = None
    addition_time_embed_dim: int | None = None
    projection_class_embeddings_input_dim: int | None = None
    # Whether the Transformer2D blocks use a Linear (True) or 1x1 Conv (False,
    # Stable Diffusion 1.x default) for their proj_in/proj_out.
    use_linear_projection: bool = False
    # Runtime LoRA adapters to bake into the attention projections as
    # `(name, rank, scale)`. Each becomes a low-rank branch gated at run time by
    # a scalar `lora_gate.{name}` input (1.0 = on, 0.0 = off, or a blend
    # strength). Empty = no LoRA (plain projections).
    lora_adapters: tuple[tuple[str, int, float], ...] = ()

    @classmethod
    def from_diffusers(cls, config: dict) -> UNet2DConfig:
        """Create a UNet2DConfig from a diffusers config dict."""
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 4),
            out_channels=config.get("out_channels", 4),
            block_out_channels=tuple(config.get("block_out_channels", [320, 640, 1280, 1280])),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
            cross_attention_dim=config.get("cross_attention_dim", 768),
            attention_head_dim=config.get("attention_head_dim", 8),
            act_fn=config.get("act_fn", "silu"),
            sample_size=config.get("sample_size", 64),
            down_block_types=(
                tuple(config["down_block_types"])
                if config.get("down_block_types") is not None
                else None
            ),
            up_block_types=(
                tuple(config["up_block_types"])
                if config.get("up_block_types") is not None
                else None
            ),
            mid_block_type=config.get("mid_block_type", "UNetMidBlock2DCrossAttn"),
            addition_embed_type=config.get("addition_embed_type"),
            addition_time_embed_dim=config.get("addition_time_embed_dim"),
            projection_class_embeddings_input_dim=config.get(
                "projection_class_embeddings_input_dim"
            ),
            use_linear_projection=config.get("use_linear_projection", False),
        )


@dataclasses.dataclass
class CogVideoXConfig:
    """Configuration for CogVideoXTransformer3DModel.

    3D video diffusion transformer with dual-stream joint attention.
    """

    num_attention_heads: int = 30
    attention_head_dim: int = 64
    in_channels: int = 16
    out_channels: int = 16
    time_embed_dim: int = 512
    text_embed_dim: int = 4096
    num_layers: int = 30
    patch_size: int = 2
    patch_size_t: int | None = None
    sample_height: int = 60
    sample_width: int = 90
    sample_frames: int = 49
    temporal_compression_ratio: int = 4
    max_text_seq_length: int = 226
    spatial_interpolation_scale: float = 1.875
    temporal_interpolation_scale: float = 1.0
    use_learned_positional_embeddings: bool = False
    norm_eps: float = 1e-5
    # cross_attention_dim used by VideoDenoisingTask for text conditioning
    cross_attention_dim: int = 4096

    @classmethod
    def from_diffusers(cls, config: dict) -> CogVideoXConfig:
        """Create from a HF diffusers config dict."""
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        heads = config.get("num_attention_heads", 30)
        head_dim = config.get("attention_head_dim", 64)
        text_dim = config.get("text_embed_dim", 4096)
        return cls(
            num_attention_heads=heads,
            attention_head_dim=head_dim,
            in_channels=config.get("in_channels", 16),
            out_channels=config.get("out_channels", 16),
            time_embed_dim=config.get("time_embed_dim", 512),
            text_embed_dim=text_dim,
            num_layers=config.get("num_layers", 30),
            patch_size=config.get("patch_size", 2),
            patch_size_t=config.get("patch_size_t"),
            sample_height=config.get("sample_height", 60),
            sample_width=config.get("sample_width", 90),
            sample_frames=config.get("sample_frames", 49),
            temporal_compression_ratio=config.get("temporal_compression_ratio", 4),
            max_text_seq_length=config.get("max_text_seq_length", 226),
            spatial_interpolation_scale=config.get("spatial_interpolation_scale", 1.875),
            temporal_interpolation_scale=config.get("temporal_interpolation_scale", 1.0),
            use_learned_positional_embeddings=bool(
                config.get("use_learned_positional_embeddings", False)
            ),
            norm_eps=config.get("norm_eps", 1e-5),
            cross_attention_dim=text_dim,
        )


@dataclasses.dataclass
class QwenImageConfig:
    """Configuration for QwenImageTransformer2DModel."""

    in_channels: int = 64
    out_channels: int = 16
    patch_size: int = 2
    num_layers: int = 60
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 3584
    guidance_embeds: bool = False
    axes_dims_rope: tuple[int, ...] = (16, 56, 56)
    norm_eps: float = 1e-6
    dtype: ir.DataType = ir.DataType.FLOAT
    # cross_attention_dim is used by DenoisingTask for encoder_hidden_states shape
    cross_attention_dim: int = 3584

    @classmethod
    def from_diffusers(cls, config: dict) -> QwenImageConfig:
        """Create a QwenImageConfig from a diffusers config dict."""
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 64),
            out_channels=config.get("out_channels", 16),
            patch_size=config.get("patch_size", 2),
            num_layers=config.get("num_layers", 60),
            attention_head_dim=config.get("attention_head_dim", 128),
            num_attention_heads=config.get("num_attention_heads", 24),
            joint_attention_dim=config.get("joint_attention_dim", 3584),
            guidance_embeds=config.get("guidance_embeds", False),
            axes_dims_rope=tuple(config.get("axes_dims_rope", [16, 56, 56])),
            cross_attention_dim=config.get("joint_attention_dim", 3584),
        )


@dataclasses.dataclass
class QwenImageVAEConfig:
    """Configuration for AutoencoderKLQwenImage (3D causal VAE)."""

    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None
    dtype: ir.DataType = ir.DataType.FLOAT

    @classmethod
    def from_diffusers(cls, config: dict) -> QwenImageVAEConfig:
        """Create a QwenImageVAEConfig from a diffusers config dict."""
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            base_dim=config.get("base_dim", 96),
            z_dim=config.get("z_dim", 16),
            dim_mult=tuple(config.get("dim_mult", [1, 2, 4, 4])),
            num_res_blocks=config.get("num_res_blocks", 2),
            attn_scales=tuple(config.get("attn_scales", [])),
            temperal_downsample=tuple(config.get("temperal_downsample", [False, True, True])),
            dropout=config.get("dropout", 0.0),
            latents_mean=(
                tuple(config["latents_mean"])
                if config.get("latents_mean") is not None
                else None
            ),
            latents_std=(
                tuple(config["latents_std"]) if config.get("latents_std") is not None else None
            ),
        )
