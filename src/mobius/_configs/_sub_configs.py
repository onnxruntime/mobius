# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Sub-configuration dataclasses used by :class:`ArchitectureConfig`.

These are pure-data dataclasses with no extraction logic. The mapping
from HuggingFace fields lives in :mod:`mobius._configs._base` (or, for
model-specific quirks, in per-model extractor functions).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class RoPEConfig:
    """Configuration for rotary position embeddings (RoPE).

    Groups the 7 RoPE-related fields that were previously spread across
    :class:`ArchitectureConfig` as flat attributes.
    """

    rope_type: str = "default"
    rope_theta: float = 10_000.0
    rope_scaling: dict | None = None
    partial_rotary_factor: float = 1.0
    rope_local_base_freq: float | None = None
    original_max_position_embeddings: int | None = None
    rope_interleave: bool = False


@dataclasses.dataclass
class VisionConfig:
    """Configuration for the vision encoder in multimodal models.

    This groups all vision-related fields that were previously scattered
    as ``vision_*`` prefixed fields on :class:`ArchitectureConfig`.
    """

    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    image_size: int | None = None
    patch_size: int | None = None
    norm_eps: float = 1e-6
    mm_tokens_per_image: int | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
    # Pixtral / Mistral-3 vision fields
    model_type: str | None = None
    head_dim: int | None = None
    rope_theta: float | None = None
    # Qwen VL-specific
    out_hidden_size: int | None = None
    in_channels: int = 3
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    frame_windows_size: int = 4
    tokens_per_second: float = 1.0
    num_position_embeddings: int | None = None
    deepstack_visual_indexes: list[int] | None = None
    fullatt_block_indexes: list[int] | None = None
    window_size: int | None = None
    # MRoPE section (for multimodal position encoding)
    mrope_section: list[int] | None = None
    # Phi4MM image embedding
    image_crop_size: int | None = None
    # LoRA config
    lora: dict | None = None
    # Gemma4 SigLIP vision encoder uses clipped linear activations
    use_clipped_linears: bool = False
    # Gemma4 SigLIP patch position embedding table size (HF: position_embedding_size)
    position_embedding_size: int | None = None
    # Learned 2D position-embedding grid dimensions.
    position_embedding_height: int | None = None
    position_embedding_width: int | None = None
    # Gemma4 VisionPooler spatial average pooling kernel size (3 → 3x3 pooling, N→N/9 tokens)
    pooling_kernel_size: int | None = None
    # MLP activation for vision encoder layers (e.g. "gelu_pytorch_tanh" for Gemma4 SigLIP)
    hidden_act: str | None = None
    # Cosmos3-Edge pixel-shuffle projector intermediate size
    # (HF: projector_config.merger_intermediate_size). ``None`` means the model
    # does not use a Cosmos-style merger projector.
    projector_intermediate_size: int | None = None
    # CLIP-style feature extraction: which ``hidden_states`` index to output
    # (HuggingFace convention, e.g. -2 for Phi-3.5-Vision). ``None`` means use
    # the final hidden state (all layers + post_layernorm).
    feature_layer: int | None = None
    # Gemma3n multimodal embedder: the vision "vocabulary" occupies token ids
    # [vocab_offset, vocab_offset + vocab_size). Gemma3nMultimodalEmbedder uses
    # the offset to rebase hard token ids into its own 128-entry table.
    vocab_offset: int | None = None
    vocab_size: int | None = None
    # timm architecture name for towers that are not SigLIP/CLIP transformers
    # (HF ``Gemma3nVisionConfig.architecture``, e.g. "mobilenetv5_300m_enc").
    architecture: str | None = None
    # Whether the tower applies its final global pooling. Gemma3n sets False:
    # it needs the 16x16 spatial map, not a single pooled vector.
    do_pooling: bool = True
    # RMSNorm epsilon for the vision tower (may differ from the text decoder's).
    rms_norm_eps: float | None = None


@dataclasses.dataclass
class CodecDecoderConfig:
    """Configuration for the codec decoder (codes → waveform)."""

    codebook_dim: int = 512
    codebook_size: int = 2048
    latent_dim: int = 1024
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8000
    decoder_dim: int = 1536
    num_quantizers: int = 16
    upsample_rates: list[int] = dataclasses.field(default_factory=lambda: [8, 5, 4, 3])
    upsampling_ratios: list[int] = dataclasses.field(default_factory=lambda: [2, 2])


@dataclasses.dataclass
class CodecEncoderConfig:
    """Configuration for the codec encoder (waveform → codes)."""

    codebook_dim: int = 256
    codebook_size: int = 2048
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 64
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8000
    num_quantizers: int = 32
    num_semantic_quantizers: int = 1


@dataclasses.dataclass
class SpeakerEncoderConfig:
    """Configuration for the ECAPA-TDNN speaker encoder in TTS models."""

    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: list[int] = dataclasses.field(
        default_factory=lambda: [512, 512, 512, 512, 1536]
    )
    enc_kernel_sizes: list[int] = dataclasses.field(default_factory=lambda: [5, 3, 3, 3, 1])
    enc_dilations: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4, 1])
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128


@dataclasses.dataclass
class CodePredictorConfig:
    """Configuration for the TTS code predictor sub-model."""

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 2048
    num_code_groups: int = 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    hidden_act: str = "silu"
    layer_types: list[str] | None = None


@dataclasses.dataclass
class TTSConfig:
    """Configuration for Qwen3-TTS models.

    Groups TTS-specific fields: talker parameters, code predictor config,
    and speaker encoder config.
    """

    # Talker parameters
    text_hidden_size: int = 2048
    text_vocab_size: int = 151936
    num_code_groups: int = 16
    # Special token IDs
    codec_bos_id: int = 2149
    codec_eos_token_id: int = 2150
    codec_pad_id: int = 2148
    codec_think_id: int = 2154
    codec_nothink_id: int = 2155
    # Sub-configs
    code_predictor: CodePredictorConfig | None = None
    speaker_encoder: SpeakerEncoderConfig | None = None


@dataclasses.dataclass
class AudioConfig:
    """Configuration for the audio encoder in multimodal models."""

    attention_dim: int | None = None
    attention_heads: int | None = None
    num_blocks: int | None = None
    linear_units: int | None = None
    kernel_size: int | None = None
    input_size: int | None = None
    conv_channels: int | None = None
    t5_bias_max_distance: int | None = None
    projection_hidden_size: int | None = None
    token_id: int | None = None
    # Qwen3-ASR encoder config
    d_model: int | None = None
    encoder_layers: int | None = None
    encoder_attention_heads: int | None = None
    encoder_ffn_dim: int | None = None
    num_mel_bins: int | None = None
    max_source_positions: int | None = None
    downsample_hidden_size: int | None = None
    output_dim: int | None = None
    activation_function: str = "gelu"
    audio_token_id: int | None = None
    audio_start_token_id: int | None = None
    audio_end_token_id: int | None = None
    classify_num: int | None = None
    # RMSNorm epsilon for the audio encoder/embedder (may differ from the text
    # decoder's rms_norm_eps). Falls back to the text value when unset.
    rms_norm_eps: float | None = None
    # Qwen3-ASR chunked conv parameters. ``n_window`` is half the
    # number of mel frames per conv chunk (so chunk_size = 2 *
    # n_window). ``n_window_infer`` is the attention window in mel
    # frames; encoder self-attention is block-diagonal with windows
    # of n_window_infer / (2 * n_window) post-conv chunks (~=
    # n_window_infer * tokens_per_chunk / chunk_size_mel post-conv
    # tokens). HF reference: QwenLM/Qwen3-ASR
    # qwen_asr/core/transformers_backend/modeling_qwen3_asr.py
    n_window: int | None = None
    n_window_infer: int | None = None
    # Fun-ASR / SenseVoice encoder config
    tp_num_blocks: int | None = None
    adaptor_proj_dim: int | None = None
    adaptor_num_blocks: int | None = None
    adaptor_ffn_dim: int | None = None
    adaptor_num_heads: int | None = None
    # LoRA config
    lora: dict | None = None


@dataclasses.dataclass
class Gemma4AudioConfig(AudioConfig):
    """Configuration for the Gemma4 Conformer audio encoder.

    Extends :class:`AudioConfig` with Gemma4-specific fields:

    - ``num_layers``: number of Conformer encoder blocks (12 for Gemma4)
    - ``hidden_size``: encoder hidden dimension (1024 for Gemma4)
    - ``subsampling_conv_channels``: channel sizes for 2D convolutional
      subsampling layers (e.g. ``[128, 32]`` for Gemma4)
    - ``use_causal_chunked_attn``: whether attention is causal + chunked
      (streaming-compatible) rather than full bidirectional
    """

    num_layers: int = 12
    hidden_size: int = 1024
    subsampling_conv_channels: list[int] | None = None
    use_causal_chunked_attn: bool = False
    output_proj_dims: int | None = None


@dataclasses.dataclass
class Gemma3nAudioConfig(AudioConfig):
    """Configuration for the Gemma 3n USM Conformer audio encoder.

    Field names mirror HF ``Gemma3nAudioConfig`` so the extractor is a
    straight copy.  The encoder is a stack of ``conf_num_hidden_layers``
    Conformer blocks fed by a two-stage strided 2D convolutional subsampler
    ("SSCP"), which reduces the ``input_feat_size``-bin mel frames by
    ``conf_reduction_factor`` in time.

    Attention is causal and chunked for streaming: each query attends to its
    own ``conf_attention_chunk_size`` chunk plus ``conf_attention_context_left``
    frames of history and ``conf_attention_context_right`` of lookahead
    (0 for E4B, i.e. strictly causal).  ``conf_attention_logit_cap`` tanh-caps
    the QK logits.

    Differences from :class:`Gemma4AudioConfig`, which the components reuse
    heavily: the SSCP blocks normalise with a *cumulative* group norm over the
    time axis (``sscp_conv_group_norm_eps``) rather than a plain LayerNorm,
    and attention carries an explicit relative-position bias projection.

    ``vocab_offset``/``vocab_size`` describe the audio soft-token id range
    ([262272, 262400) for E4B) used by the multimodal embedder.
    """

    hidden_size: int = 1536
    conf_num_hidden_layers: int = 12
    conf_num_attention_heads: int = 8
    conf_attention_chunk_size: int = 12
    conf_attention_context_left: int = 13
    conf_attention_context_right: int = 0
    conf_attention_logit_cap: float = 50.0
    conf_conv_kernel_size: int = 5
    conf_reduction_factor: int = 4
    conf_residual_weight: float = 0.5
    input_feat_size: int = 128
    sscp_conv_channel_size: list[int] | None = None
    sscp_conv_kernel_size: list[list[int]] | None = None
    sscp_conv_stride_size: list[list[int]] | None = None
    sscp_conv_group_norm_eps: float = 1e-3
    gradient_clipping: float = 1e10
    vocab_offset: int | None = None
    vocab_size: int | None = None
