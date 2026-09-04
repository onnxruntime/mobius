# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Source-faithful staged VibeVoice Realtime text-to-speech model.

This module replicates Microsoft VibeVoice commit
``79e516a3e20b599f137c9da03410a2a0b473b63b``'s
``VibeVoiceStreamingForConditionalGenerationInference``. Inputs are split
into lower-Qwen2 text, upper-Qwen2 TTS, acoustic connector, diffusion, and
streaming acoustic-decoder graphs; the host owns scheduling and voice presets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import VibeVoiceStreamingConfig
from mobius.components import (
    DecoderLayer,
    Embedding,
    Linear,
    RMSNorm,
    initialize_rope,
)
from mobius.models.vibevoice import VibeVoiceDiffusionHead, VibeVoiceTokenizerDecoder

VIBEVOICE_STREAMING_MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
VIBEVOICE_STREAMING_REVISION = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"
VIBEVOICE_STREAMING_MICROSOFT_PROVENANCE_REVISION = "79e516a3e20b599f137c9da03410a2a0b473b63b"


class VibeVoiceStreamingEmbedding(nn.Module):
    """Embed Qwen2 text or pseudo-token IDs using the lower LM embedding table."""

    def __init__(self, config: VibeVoiceStreamingConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)

    def forward(self, op: OpBuilder, input_ids: ir.Value):
        return self.embed_tokens(op, input_ids)  # (batch, sequence, hidden)


class _VibeVoiceStreamingBackbone(nn.Module):
    """Qwen2 transformer partition with an optional final RMS normalization."""

    def __init__(self, config: VibeVoiceStreamingConfig, *, num_layers: int, norm: bool):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if norm else None
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: Sequence[tuple[ir.Value, ir.Value]],
    ):
        # Qwen2 RoPE positions are shared by all layers of one cache family.
        position_embeddings = self.rotary_emb(op, position_ids)
        # VibeVoice Realtime uses only a causal, prefix-valid mask. ORT Attention
        # requires its BOOL mask's penultimate axis to be the query length, so
        # expand the keep-mask without materializing a causal additive bias.
        valid = op.Cast(attention_mask, to=ir.DataType.BOOL)
        attention_bias = op.Expand(
            op.Unsqueeze(valid, [1]),
            op.Concat(
                op.Shape(inputs_embeds, start=0, end=2),
                op.Shape(valid, start=1, end=2),
                axis=0,
            ),
        )
        present_key_values = []
        hidden_states = inputs_embeds
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )
            present_key_values.append(present)
        if self.norm is not None:
            hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class VibeVoiceStreamingLMBackbone(_VibeVoiceStreamingBackbone):
    """Lower Qwen2 text encoder; source intentionally omits its final norm."""

    def __init__(self, config: VibeVoiceStreamingConfig):
        super().__init__(
            config,
            num_layers=config.lm_backbone_num_hidden_layers,
            norm=False,
        )


class VibeVoiceStreamingTTSBackbone(_VibeVoiceStreamingBackbone):
    """Upper Qwen2 TTS encoder with text/speech types and binary EOS prediction."""

    def __init__(self, config: VibeVoiceStreamingConfig):
        super().__init__(
            config,
            num_layers=config.tts_backbone_num_hidden_layers,
            norm=True,
        )
        self.tts_input_types = Embedding(2, config.hidden_size)
        self.tts_eos_classifier = _VibeVoiceStreamingBinaryClassifier(config.hidden_size)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: Sequence[tuple[ir.Value, ir.Value]],
        lm_last_hidden_state: ir.Value | None = None,
        tts_text_masks: ir.Value | None = None,
    ):
        if (lm_last_hidden_state is None) != (tts_text_masks is None):
            raise ValueError(
                "lm_last_hidden_state and tts_text_masks must be provided together"
            )
        if lm_last_hidden_state is not None:
            # Replace only the current tail: lower-LM text states or acoustic
            # connector outputs occupy these slots; preceding pseudo tokens are retained.
            prefix_length = op.Sub(
                op.Shape(inputs_embeds, start=1, end=2),
                op.Shape(lm_last_hidden_state, start=1, end=2),
            )
            prefix = op.Slice(
                inputs_embeds,
                op.Constant(value_ints=[0]),
                prefix_length,
                op.Constant(value_ints=[1]),
            )
            inputs_embeds = op.Concat(prefix, lm_last_hidden_state, axis=1)
            # Materialize the source's [B, 1, H] broadcast to [B, S, H]. This is
            # mathematically identical, but prevents ORT's skip-RMSNorm fusion from
            # receiving residual inputs with mismatched trailing dimensions.
            input_types = op.Expand(
                self.tts_input_types(op, op.Cast(tts_text_masks, to=ir.DataType.INT64)),
                op.Shape(inputs_embeds),
            )
            inputs_embeds = op.Add(inputs_embeds, input_types)
        hidden_states, present_key_values = super().forward(
            op,
            inputs_embeds,
            attention_mask,
            position_ids,
            past_key_values,
        )
        # Microsoft predicts end-of-speech only from the newest TTS hidden state.
        eos_logits = self.tts_eos_classifier(
            op,
            op.Squeeze(
                op.Slice(
                    hidden_states,
                    op.Constant(value_ints=[-1]),
                    op.Constant(value_ints=[2**63 - 1]),
                    op.Constant(value_ints=[1]),
                ),
                [1],
            ),
        )
        return eos_logits, hidden_states, present_key_values


class _VibeVoiceStreamingBinaryClassifier(nn.Module):
    """Source ``BinaryClassifier``: linear, ReLU, then a binary logit."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, 1)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return self.fc2(op, op.Relu(self.fc1(op, hidden_states)))


class VibeVoiceStreamingSpeechConnector(nn.Module):
    """Project one or more 64-D diffusion latents into upper-Qwen2 input space."""

    def __init__(self, config: VibeVoiceStreamingConfig):
        super().__init__()
        latent_size = config.acoustic_tokenizer.vae_dim
        self.fc1 = Linear(latent_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.fc2 = Linear(config.hidden_size, config.hidden_size)

    def forward(self, op: OpBuilder, speech_latents: ir.Value):
        hidden_states = self.fc1(op, speech_latents)
        hidden_states = self.norm(op, hidden_states)
        return self.fc2(op, hidden_states)  # (batch, acoustic_frames, hidden)


class VibeVoiceStreamingAudioDecoder(nn.Module):
    """Causal acoustic decoder with source ratio order and explicit conv history."""

    def __init__(self, config: VibeVoiceStreamingConfig):
        super().__init__()
        self.decoder = VibeVoiceTokenizerDecoder(
            config.acoustic_tokenizer.as_decoder_config(),
            reverse_ratios=False,
        )
        self.speech_scaling_factor = nn.Parameter([])
        self.speech_bias_factor = nn.Parameter([])
        self.cache_specs = self.decoder.cache_specs

    def forward(
        self,
        op: OpBuilder,
        speech_latents: ir.Value,
        past_conv_states: Sequence[ir.Value],
    ):
        # Diffusion emits normalized latents; Microsoft unnormalizes before
        # the causal tokenizer, retaining history between six-latent windows.
        latents = op.Sub(
            op.Div(speech_latents, self.speech_scaling_factor),
            self.speech_bias_factor,
        )
        return self.decoder(op, latents, past_conv_states)


class VibeVoiceStreamingForConditionalGeneration(nn.Module):
    """Six-stage VibeVoice Realtime TTS package with host-owned CFG orchestration.

    ```{mermaid}
    flowchart LR
        Text[5-token text window] --> Emb[Qwen2 embedding]
        Emb --> LM[Lower 4-layer Qwen2]
        LM --> TTS[Upper 20-layer Qwen2]
        TTS --> Pos[Positive condition]
        TTS --> Neg[Negative CFG condition]
        Pos --> Diffusion[DDPM/DPM-Solver host loop]
        Neg --> Diffusion
        Diffusion --> Latent[64-D acoustic latent]
        Latent --> Connector[Speech connector]
        Connector --> TTS
        Latent --> Decoder[Causal acoustic decoder]
        Decoder --> Audio[3200 samples per latent]
    ```

    The lower and upper Qwen2 backbones expose independent KV caches; the
    acoustic decoder exposes convolution histories. Voice presets are trusted
    prefilled cache artifacts, while CFG, text windowing, DPM-Solver scheduling,
    and stopping remain host responsibilities. This package is not an
    ONNX Runtime GenAI runnable model.
    """

    default_task = "vibevoice-streaming-tts"
    category = "Audio"
    config_class = VibeVoiceStreamingConfig

    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "embedding": ("model.language_model.embed_tokens",),
        "lm_backbone": ("model.language_model.layers",),
        "tts_backbone": (
            "model.tts_language_model.layers",
            "model.tts_language_model.norm",
            "model.tts_input_types",
            "tts_eos_classifier",
        ),
        "speech_connector": ("model.acoustic_connector",),
        "diffusion_head": ("model.prediction_head",),
        "audio_decoder": (
            "model.acoustic_tokenizer.decoder",
            "model.speech_scaling_factor",
            "model.speech_bias_factor",
        ),
    }
    INTENTIONALLY_UNUSED_SOURCE_WEIGHTS: ClassVar[frozenset[str]] = frozenset(
        {"model.tts_language_model.embed_tokens.weight"}
    )

    def __init__(self, config: VibeVoiceStreamingConfig):
        super().__init__()
        self.config = config
        self.embedding = VibeVoiceStreamingEmbedding(config)
        self.lm_backbone = VibeVoiceStreamingLMBackbone(config)
        self.tts_backbone = VibeVoiceStreamingTTSBackbone(config)
        self.speech_connector = VibeVoiceStreamingSpeechConnector(config)
        self.diffusion_head = VibeVoiceDiffusionHead(config)
        self.audio_decoder = VibeVoiceStreamingAudioDecoder(config)

    def forward(self, op: OpBuilder, *args, **kwargs):
        raise NotImplementedError(
            "VibeVoiceStreamingTask exports each required TTS stage independently"
        )

    @classmethod
    def _audio_decoder_target(cls, suffix: str) -> str:
        if suffix.startswith("upsample_layers."):
            _, index, block, remainder = suffix.split(".", 3)
            if block != "0":
                raise ValueError(
                    f"Unsupported VibeVoice Realtime decoder block in weight {suffix!r}"
                )
            if index == "0":
                return f"audio_decoder.decoder.stem.{remainder}"
            return f"audio_decoder.decoder.conv_layers.{int(index) - 1}.{remainder}"
        if suffix.startswith("stages."):
            _, index, remainder = suffix.split(".", 2)
            prefix = (
                "audio_decoder.decoder.stem.stage"
                if index == "0"
                else f"audio_decoder.decoder.conv_layers.{int(index) - 1}.stage"
            )
            # The source has SConv1d -> NormConv1d -> raw Conv1d nesting;
            # Mobius combines the first two wrappers while keeping the same math.
            remainder = remainder.replace(
                ".mixer.conv.conv.conv.",
                ".mixer.conv.",
            )
            return f"{prefix}.{remainder}"
        if suffix.startswith("head."):
            # The source's head includes a no-op NormConv wrapper, whereas the
            # shared primitive exposes its causal convolution directly.
            return "audio_decoder.decoder." + suffix.replace(".conv.conv.", ".conv.")
        raise ValueError(f"Unsupported VibeVoice Realtime decoder weight {suffix!r}")

    @classmethod
    def source_weight_targets(
        cls,
        state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, str | None]:
        """Classify every checkpoint tensor, failing closed on source drift."""
        targets: dict[str, str | None] = {}
        for source_name in state_dict:
            target: str | None
            if source_name in cls.INTENTIONALLY_UNUSED_SOURCE_WEIGHTS:
                target = None
            elif source_name.startswith("model.language_model.embed_tokens."):
                target = "embedding." + source_name.removeprefix("model.language_model.")
            elif source_name.startswith("model.language_model.layers."):
                target = "lm_backbone." + source_name.removeprefix("model.language_model.")
            elif source_name.startswith("model.tts_language_model.layers."):
                target = "tts_backbone." + source_name.removeprefix(
                    "model.tts_language_model."
                )
            elif source_name.startswith("model.tts_language_model.norm."):
                target = "tts_backbone." + source_name.removeprefix(
                    "model.tts_language_model."
                )
            elif source_name.startswith("model.tts_input_types."):
                target = "tts_backbone." + source_name.removeprefix("model.")
            elif source_name.startswith("tts_eos_classifier."):
                target = "tts_backbone." + source_name
            elif source_name.startswith("model.acoustic_connector."):
                target = "speech_connector." + source_name.removeprefix(
                    "model.acoustic_connector."
                )
            elif source_name.startswith("model.prediction_head."):
                suffix = source_name.removeprefix("model.prediction_head.")
                target = "diffusion_head." + (
                    suffix.replace("t_embedder.mlp.0.", "timestep_proj.fc1.")
                    .replace("t_embedder.mlp.2.", "timestep_proj.fc2.")
                    .replace("final_layer.adaLN_modulation.1.", "final_layer.linear_1.")
                    .replace("final_layer.linear.", "final_layer.linear_2.")
                    .replace("adaLN_modulation.1.", "linear.")
                )
            elif source_name.startswith("model.acoustic_tokenizer.decoder."):
                target = cls._audio_decoder_target(
                    source_name.removeprefix("model.acoustic_tokenizer.decoder.")
                )
            elif source_name == "model.speech_scaling_factor":
                target = "audio_decoder.speech_scaling_factor"
            elif source_name == "model.speech_bias_factor":
                target = "audio_decoder.speech_bias_factor"
            else:
                raise ValueError(
                    "Unsupported VibeVoice Realtime source weight "
                    f"{source_name!r}; the checkpoint layout changed"
                )
            targets[source_name] = target
        return targets

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Route every trained source tensor and document its one unused embedding."""
        return {
            target: state_dict[source_name]
            for source_name, target in self.source_weight_targets(state_dict).items()
            if target is not None
        }
