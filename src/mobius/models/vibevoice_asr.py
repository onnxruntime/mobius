# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Staged offline ASR for the original ``VibeVoiceForASRTraining`` checkpoint.

The model is not VibeVoice TTS. It runs 64-D acoustic and 128-D semantic
causal waveform encoders, sums their independent connectors, replaces audio
placeholder embeddings, then autoregressively decodes structured diarization
text through a Qwen2 decoder.
"""

from __future__ import annotations

import re
from typing import ClassVar

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import VibeVoiceASRConfig
from mobius.components import Embedding
from mobius.models.vibevoice import VibeVoiceDecoderModel, VibeVoiceMultiModalProjector
from mobius.models.vibevoice import VibeVoiceTokenizerEncoder

VIBEVOICE_ASR_MODEL_ID = "microsoft/VibeVoice-ASR"
VIBEVOICE_ASR_REVISION = "d0c9efdb8d614685062c04425d91e01b6f37d944"
VIBEVOICE_ASR_TRANSFORMERS_REVISION = "f62dc9bf2c90353b442a56e74391fbb8c689b55e"
VIBEVOICE_ASR_MICROSOFT_REVISION = "1541f590c7099820f10ea012f48d2399282df69f"


class VibeVoiceASRAudioEncoder(nn.Module):
    """One cached causal waveform encoder producing acoustic or semantic latents."""

    def __init__(self, config: VibeVoiceASRConfig, *, semantic: bool):
        super().__init__()
        tokenizer = config.semantic_tokenizer if semantic else config.acoustic_tokenizer
        self.encoder = VibeVoiceTokenizerEncoder(tokenizer)
        self.cache_specs = self.encoder.cache_specs
        self._dtype = config.dtype

    def forward(
        self,
        op: OpBuilder,
        input_values: ir.Value,
        past_conv_states: list[ir.Value],
    ) -> tuple[ir.Value, list[ir.Value]]:
        # Processor waveforms remain float32 at the boundary; model weights and
        # explicit convolution caches use the selected package precision.
        waveform = op.Cast(input_values, to=self._dtype)  # (B, 1, samples)
        return self.encoder(op, waveform, past_conv_states)  # (B, frames, latent)


class VibeVoiceASRConnectors(nn.Module):
    """Sample acoustic latents, add two projected paths, and remove padded frames."""

    def __init__(self, config: VibeVoiceASRConfig):
        super().__init__()
        self.acoustic_connector = VibeVoiceMultiModalProjector(
            config.acoustic_tokenizer.hidden_size,
            config.hidden_size,
        )
        self.semantic_connector = VibeVoiceMultiModalProjector(
            config.semantic_tokenizer.hidden_size,
            config.hidden_size,
        )
        self._acoustic_vae_std = config.acoustic_tokenizer.vae_std
        self._hop_length = config.acoustic_tokenizer.hop_length

    def forward(
        self,
        op: OpBuilder,
        acoustic_latents: ir.Value,
        semantic_latents: ir.Value,
        padding_mask: ir.Value,
        acoustic_noise_scale: ir.Value,
        acoustic_latent_noise: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        # Source sampling is ``mean + vae_std * randn(B) * randn_like(mean)``.
        # Named random draws keep exported ONNX deterministic and reproducible.
        noise_scale = op.Mul(acoustic_noise_scale, self._acoustic_vae_std)
        sampled_acoustic = op.Add(
            acoustic_latents,
            op.Mul(op.Unsqueeze(noise_scale, [1, 2]), acoustic_latent_noise),
        )  # (B, frames, 64)
        combined = op.Add(
            self.acoustic_connector(op, sampled_acoustic),
            self.semantic_connector(op, semantic_latents),
        )  # (B, frames, text_hidden)

        valid_samples = op.ReduceSum(
            op.Cast(padding_mask, to=ir.DataType.INT64),
            op.Constant(value_ints=[1]),
            keepdims=0,
        )
        valid_frames = op.Div(
            op.Add(valid_samples, self._hop_length - 1),
            self._hop_length,
        )  # (B,) ceil(valid_samples / 3200)
        frame_count = op.Squeeze(op.Shape(combined, start=1, end=2), [0])
        frame_positions = op.Range(
            op.Constant(value_int=0),
            frame_count,
            op.Constant(value_int=1),
        )
        valid_mask = op.Less(
            op.Unsqueeze(frame_positions, [0]),
            op.Unsqueeze(valid_frames, [1]),
        )
        valid_indices = op.Transpose(op.NonZero(valid_mask), perm=[1, 0])
        return op.GatherND(combined, valid_indices), valid_frames


class VibeVoiceASREmbeddingModel(nn.Module):
    """Replace audio-token placeholders with flattened, valid connector features."""

    def __init__(self, config: VibeVoiceASRConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self._audio_token_id = config.audio_token_id
        self._hidden_size = config.hidden_size

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        audio_features: ir.Value,
    ) -> ir.Value:
        inputs_embeds = self.embed_tokens(op, input_ids)
        is_audio = op.Equal(input_ids, self._audio_token_id)
        flat_audio_mask = op.Reshape(is_audio, [-1])
        flat_audio_indices = op.CumSum(
            op.Cast(flat_audio_mask, to=ir.DataType.INT64),
            op.Constant(value_int=0),
        )
        flat_audio_indices = op.Mul(
            flat_audio_indices,
            op.Cast(flat_audio_mask, to=ir.DataType.INT64),
        )
        indices = op.Reshape(flat_audio_indices, op.Shape(input_ids))
        zero_row = op.Unsqueeze(
            op.CastLike(
                op.Constant(value_floats=[0.0] * self._hidden_size),
                audio_features,
            ),
            [0],
        )
        features = op.Concat(zero_row, audio_features, axis=0)
        gathered = op.Gather(features, indices, axis=0)
        return op.Where(op.Unsqueeze(is_audio, [-1]), gathered, inputs_embeds)


class VibeVoiceASRDecoderModel(VibeVoiceDecoderModel):
    """Qwen2 decoder with the standard prefix-valid KV-cache contract."""

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        logits, _, present_key_values = super().forward(
            op,
            inputs_embeds,
            attention_mask,
            position_ids,
            past_key_values,
        )
        return logits, present_key_values


def _map_original_tokenizer_encoder_key(key: str, *, source: str, destination: str) -> str | None:
    """Apply the upstream tokenizer-converter map to one original encoder key."""
    if not key.startswith(source):
        return None
    key = key.replace(source, "encoder.", 1)
    replacements: tuple[tuple[str, str | tuple[str, int]], ...] = (
        (r"^encoder\.downsample_layers\.0\.0\.conv\.", "encoder.stem.conv.conv."),
        (r"^encoder\.stages\.0\.", "encoder.stem.stage."),
        (
            r"^encoder\.downsample_layers\.(\d+)\.0\.conv\.",
            (r"encoder.conv_layers.\1.conv.conv.", -1),
        ),
        (r"^encoder\.stages\.(\d+)\.", (r"encoder.conv_layers.\1.stage.", -1)),
        (r"^encoder\.head\.conv\.", "encoder.head."),
    )
    for pattern, replacement in replacements:
        if isinstance(replacement, tuple):
            target, shift = replacement

            def _shift(match: re.Match[str]) -> str:
                return target.replace(r"\1", str(int(match.group(1)) + shift))

            key = re.sub(pattern, _shift, key)
        else:
            key = re.sub(pattern, replacement, key)
    key = key.replace("mixer.conv.conv.conv.", "mixer.conv.")
    key = key.replace(".conv.conv.conv.", ".conv.conv.")
    return f"{destination}.{key}"


class VibeVoiceASRForConditionalGeneration(nn.Module):
    """Offline VibeVoice ASR/diarization stages for ``VibeVoiceForASRTraining``."""

    default_task: str = "vibevoice-asr"
    category: str = "Speech-to-Text"
    config_class = VibeVoiceASRConfig

    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "acoustic_encoder": ("model.acoustic_tokenizer.encoder",),
        "semantic_encoder": ("model.semantic_tokenizer.encoder",),
        "connectors": ("model.acoustic_connector", "model.semantic_connector"),
        "embedding": ("model.language_model.embed_tokens",),
        "decoder": ("model.language_model.layers", "model.language_model.norm", "lm_head"),
    }

    def __init__(self, config: VibeVoiceASRConfig):
        super().__init__()
        self.config = config
        self.acoustic_encoder = VibeVoiceASRAudioEncoder(config, semantic=False)
        self.semantic_encoder = VibeVoiceASRAudioEncoder(config, semantic=True)
        self.connectors = VibeVoiceASRConnectors(config)
        self.embedding = VibeVoiceASREmbeddingModel(config)
        self.decoder = VibeVoiceASRDecoderModel(config)

    def forward(self, op: OpBuilder, *args, **kwargs):
        raise NotImplementedError("VibeVoiceASRTask exports each ASR stage independently")

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Route every inference tensor and deliberately exclude unused VAE decoding."""
        routed: dict[str, torch.Tensor] = {}
        for source_key, value in state_dict.items():
            acoustic_key = _map_original_tokenizer_encoder_key(
                source_key,
                source="model.acoustic_tokenizer.encoder.",
                destination="acoustic_encoder",
            )
            semantic_key = _map_original_tokenizer_encoder_key(
                source_key,
                source="model.semantic_tokenizer.encoder.",
                destination="semantic_encoder",
            )
            if acoustic_key is not None:
                routed[acoustic_key] = value
            elif semantic_key is not None:
                routed[semantic_key] = value
            elif source_key.startswith("model.acoustic_tokenizer_encoder."):
                suffix = source_key.removeprefix("model.acoustic_tokenizer_encoder.")
                routed[f"acoustic_encoder.encoder.{suffix}"] = value
            elif source_key.startswith("model.semantic_tokenizer_encoder."):
                suffix = source_key.removeprefix("model.semantic_tokenizer_encoder.")
                routed[f"semantic_encoder.encoder.{suffix}"] = value
            elif source_key.startswith("model.acoustic_connector."):
                suffix = source_key.removeprefix("model.acoustic_connector.")
                suffix = suffix.replace("fc1.", "linear_1.").replace("norm.", "act.").replace(
                    "fc2.", "linear_2."
                )
                routed[f"connectors.acoustic_connector.{suffix}"] = value
            elif source_key.startswith("model.semantic_connector."):
                suffix = source_key.removeprefix("model.semantic_connector.")
                suffix = suffix.replace("fc1.", "linear_1.").replace("norm.", "act.").replace(
                    "fc2.", "linear_2."
                )
                routed[f"connectors.semantic_connector.{suffix}"] = value
            elif source_key.startswith("model.multi_modal_projector.acoustic_"):
                suffix = source_key.removeprefix("model.multi_modal_projector.acoustic_")
                suffix = suffix.replace("linear_1.", "linear_1.").replace("norm.", "act.").replace(
                    "linear_2.", "linear_2."
                )
                routed[f"connectors.acoustic_connector.{suffix}"] = value
            elif source_key.startswith("model.multi_modal_projector.semantic_"):
                suffix = source_key.removeprefix("model.multi_modal_projector.semantic_")
                suffix = suffix.replace("linear_1.", "linear_1.").replace("norm.", "act.").replace(
                    "linear_2.", "linear_2."
                )
                routed[f"connectors.semantic_connector.{suffix}"] = value
            elif source_key.startswith("model.language_model.embed_tokens."):
                suffix = source_key.removeprefix("model.language_model.embed_tokens.")
                routed[f"embedding.embed_tokens.{suffix}"] = value
            elif source_key.startswith(("model.language_model.layers.", "model.language_model.norm.")):
                suffix = source_key.removeprefix("model.language_model.")
                routed[f"decoder.{suffix}"] = value
            elif source_key == "lm_head.weight":
                routed["decoder.lm_head.weight"] = value
            elif source_key.startswith("model.acoustic_tokenizer.decoder."):
                # VibeVoiceForASRTraining loads a shared acoustic VAE, but only
                # calls its encoder. Waveform-decoder weights are not inference
                # stages and must not be exported as ASR components.
                continue
        return routed
