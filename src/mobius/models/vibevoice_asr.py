# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Staged offline ASR for the native ``VibeVoiceAsrForConditionalGeneration`` checkpoint.

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
from mobius.models.vibevoice import (
    VibeVoiceDecoderModel,
    VibeVoiceMultiModalProjector,
    VibeVoiceTokenizerEncoder,
)

VIBEVOICE_ASR_MODEL_ID = "microsoft/VibeVoice-ASR-HF"
VIBEVOICE_ASR_REVISION = "f22241c2062b3b25272bf117397e03d73381037a"
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
        is_final_chunk: ir.Value,
    ) -> tuple[ir.Value, list[ir.Value]]:
        # Processor waveforms remain float32 at the boundary; model weights and
        # explicit convolution caches use the selected package precision.
        waveform = op.Cast(input_values, to=self._dtype)  # (B, 1, samples)
        return self.encoder(  # (B, frames, latent)
            op,
            waveform,
            past_conv_states,
            is_final_chunk,
        )


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


def _map_original_tokenizer_encoder_key(
    key: str, *, source: str, destination: str
) -> str | None:
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

            def _shift(
                match: re.Match[str],
                target: str = target,
                shift: int = shift,
            ) -> str:
                return target.replace(r"\1", str(int(match.group(1)) + shift))

            key = re.sub(pattern, _shift, key)
        else:
            key = re.sub(pattern, replacement, key)
    key = key.replace("mixer.conv.conv.conv.", "mixer.conv.")
    key = key.replace(".conv.conv.conv.", ".conv.conv.")
    return f"{destination}.{key}"


class VibeVoiceASRForConditionalGeneration(nn.Module):
    """Offline VibeVoice ASR/diarization stages for ``VibeVoiceAsrForConditionalGeneration``.

    Mobius selects this model only when the shared ``vibevoice`` configuration
    declares ``VibeVoiceAsrForConditionalGeneration``. VibeVoice TTS remains
    ``VibeVoiceForConditionalGeneration``; unknown, streaming, and ambiguous
    VibeVoice architectures fail closed.

    ## Architecture and package contract

    The offline model uses 24 kHz waveform input with 3200-sample framing. Its
    64-D acoustic and 128-D semantic cached causal encoders feed independent
    connectors whose projected outputs are summed, flattened to valid frames,
    and substituted for audio-placeholder embeddings before Qwen2 decoding.
    The exported package has five stages: ``acoustic_encoder``,
    ``semantic_encoder``, ``connectors``, ``embedding``, and ``decoder``.

    ```mermaid
    flowchart LR
        WAV["24 kHz mono waveform"] --> AC["Acoustic encoder: cached 64-D latents"]
        WAV --> SE["Semantic encoder: cached 128-D latents"]
        AC --> C["Acoustic connector"]
        SE --> SC["Semantic connector"]
        C --> SUM["sum and select valid frames"]
        SC --> SUM
        P["Chat prompt and audio placeholders"] --> E["Embedding mixer"]
        SUM --> E
        E --> D["Qwen2 decoder with left-padded causal cache"]
        D --> J["JSON diarization records"]
    ```

    A host owns waveform normalization, 60-second chunking, encoder cache
    propagation, deterministic acoustic noise, fixed prompt construction, and
    JSON diarization parsing. It passes ``is_final_chunk`` only for each true
    terminal window; each causal convolution then performs the source's
    intermediate right padding. Right-padded variable-length batches are
    finalized per utterance. The decoder's ordinary prefix-valid left-padded
    causal mask remains eligible for normal Qwen2 GQA optimizations.

    ## Processor and evidence

    The prompt uses fixed Qwen turns and encloses one ``<|box_start|>`` audio
    placeholder per ``ceil(samples / 3200)`` frame with
    ``<|object_ref_start|>`` and ``<|object_ref_end|>``. ``context_info`` is
    source-provided background or hotword text; there is no separate hotword
    input. Output JSON records are normalized to ``start_time``, ``end_time``,
    ``speaker_id``, and ``text``.

    Microsoft publishes support for 51 language codes: ``en, zh, es, pt, de,
    ja, ko, fr, ru, id, sv, it, he, nl, pl, no, tr, th, ar, hu, ca, cs, da, fa,
    af, hi, fi, et, aa, el, ro, vi, bg, is, sl, sk, lt, sw, uk, kl, lv, hr, ne,
    sr, tl, yi, ms, ur, mn, hy, jv``. This records the upstream claim only; it
    does not make an independent quality claim.

    L1 builds every stage and cache ABI; L2 checks the pinned raw config and
    checkpoint index; L3 compares the staged package with the pinned
    Transformers source (``f62dc9bf2c90353b442a56e74391fbb8c689b55e``),
    including batch chunking, terminal frames, seeded sampling, replacement,
    and left padding. L4/L5 real-weight transcription and diarization are
    unverified because the approximately 8.67B BF16 checkpoint requires a
    suitable CUDA host.

    The checkpoint includes acoustic VAE waveform-decoder tensors, but the ASR
    source never executes that decoder. They are deliberately excluded from
    this inference package. ONNX Runtime GenAI cannot orchestrate the dual
    cached encoders and host protocol, so its export metadata is advisory rather
    than a runnable ``genai_config.json`` contract. The default checkpoint is
    pinned to ``microsoft/VibeVoice-ASR-HF@f22241c2062b3b25272bf117397e03d73381037a``.
    """

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

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route every inference tensor and deliberately exclude unused VAE decoding."""
        routed: dict[str, torch.Tensor] = {}
        for source_key, value in state_dict.items():
            if source_key.startswith(
                (
                    "acoustic_encoder.",
                    "semantic_encoder.",
                    "connectors.",
                    "embedding.",
                    "decoder.",
                )
            ):
                # Preserve already ONNX-aligned weights for direct package loading.
                routed[source_key] = value
                continue

            if source_key.startswith("acoustic_tokenizer_encoder."):
                suffix = source_key.removeprefix("acoustic_tokenizer_encoder.")
                routed[f"acoustic_encoder.encoder.{suffix}"] = value
                continue
            if source_key.startswith("semantic_tokenizer_encoder."):
                suffix = source_key.removeprefix("semantic_tokenizer_encoder.")
                routed[f"semantic_encoder.encoder.{suffix}"] = value
                continue
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
                suffix = (
                    suffix.replace("fc1.", "linear_1.")
                    .replace("norm.", "act.")
                    .replace("fc2.", "linear_2.")
                )
                routed[f"connectors.acoustic_connector.{suffix}"] = value
            elif source_key.startswith("model.semantic_connector."):
                suffix = source_key.removeprefix("model.semantic_connector.")
                suffix = (
                    suffix.replace("fc1.", "linear_1.")
                    .replace("norm.", "act.")
                    .replace("fc2.", "linear_2.")
                )
                routed[f"connectors.semantic_connector.{suffix}"] = value
            elif source_key.startswith(
                ("model.multi_modal_projector.acoustic_", "multi_modal_projector.acoustic_")
            ):
                suffix = source_key.removeprefix("model.multi_modal_projector.acoustic_")
                suffix = suffix.removeprefix("multi_modal_projector.acoustic_")
                suffix = (
                    suffix.replace("linear_1.", "linear_1.")
                    .replace("norm.", "act.")
                    .replace("linear_2.", "linear_2.")
                )
                routed[f"connectors.acoustic_connector.{suffix}"] = value
            elif source_key.startswith(
                ("model.multi_modal_projector.semantic_", "multi_modal_projector.semantic_")
            ):
                suffix = source_key.removeprefix("model.multi_modal_projector.semantic_")
                suffix = suffix.removeprefix("multi_modal_projector.semantic_")
                suffix = (
                    suffix.replace("linear_1.", "linear_1.")
                    .replace("norm.", "act.")
                    .replace("linear_2.", "linear_2.")
                )
                routed[f"connectors.semantic_connector.{suffix}"] = value
            elif source_key.startswith(
                ("model.language_model.embed_tokens.", "language_model.model.embed_tokens.")
            ):
                suffix = source_key.removeprefix("model.language_model.embed_tokens.")
                suffix = suffix.removeprefix("language_model.model.embed_tokens.")
                routed[f"embedding.embed_tokens.{suffix}"] = value
            elif source_key.startswith(
                (
                    "model.language_model.layers.",
                    "model.language_model.norm.",
                    "language_model.model.layers.",
                    "language_model.model.norm.",
                )
            ):
                suffix = source_key.removeprefix("model.language_model.")
                suffix = suffix.removeprefix("language_model.model.")
                routed[f"decoder.{suffix}"] = value
            elif source_key in ("language_model.lm_head.weight", "lm_head.weight"):
                routed["decoder.lm_head.weight"] = value
        return routed
