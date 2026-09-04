# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Host-side contracts for the staged VibeVoice offline ASR package.

The official checkpoint contains neural weights only. This module materializes
the source processor behavior that must remain outside ONNX: 24 kHz waveform
normalization, 60-second cached chunking, prompt construction, deterministic
latent draws, and JSON diarization parsing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output in JSON format."
)


@dataclass(frozen=True)
class VibeVoiceASRBatch:
    """Normalized, right-padded 24 kHz waveforms and their sample-validity mask."""

    input_values: np.ndarray
    padding_mask: np.ndarray


class VibeVoiceASRProcessor:
    """Processor protocol from Microsoft's VibeVoice-ASR inference source."""

    sampling_rate = 24_000
    hop_length = 3_200
    chunk_samples = 1_440_000
    normalize_audio = True
    target_dbfs = -25.0
    eps = 1e-6
    avoid_clipping = True

    def prepare_audio(
        self,
        waveforms: np.ndarray | list[np.ndarray],
        *,
        sampling_rate: int,
        normalize: bool = True,
    ) -> VibeVoiceASRBatch:
        """Normalize mono source waveforms and construct a right-padded batch."""
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"VibeVoice-ASR requires {self.sampling_rate} Hz audio, got {sampling_rate} Hz"
            )
        if isinstance(waveforms, np.ndarray):
            if waveforms.ndim == 1:
                source = [waveforms]
            elif waveforms.ndim == 2:
                source = list(waveforms)
            else:
                raise ValueError(
                    "waveforms array must have shape (samples,) or (batch, samples)"
                )
        else:
            source = waveforms
        if not isinstance(source, list) or not source:
            raise ValueError(
                "waveforms must be one non-empty 1-D array or a non-empty list of them"
            )
        normalized = [self._normalize(np.asarray(waveform), normalize) for waveform in source]
        max_samples = max(waveform.size for waveform in normalized)
        input_values = np.zeros((len(normalized), max_samples), dtype=np.float32)
        padding_mask = np.zeros((len(normalized), max_samples), dtype=np.bool_)
        for index, waveform in enumerate(normalized):
            input_values[index, : waveform.size] = waveform
            padding_mask[index, : waveform.size] = True
        return VibeVoiceASRBatch(input_values=input_values, padding_mask=padding_mask)

    def iter_chunks(self, batch: VibeVoiceASRBatch) -> Iterator[VibeVoiceASRBatch]:
        """Yield hop-aligned 60-second windows; encoder cache bridges adjacent windows."""
        for start in range(0, batch.input_values.shape[-1], self.chunk_samples):
            stop = min(start + self.chunk_samples, batch.input_values.shape[-1])
            input_values = batch.input_values[:, start:stop]
            # The source processor reserves ceil(valid_samples / hop) speech
            # placeholders. Pad only the terminal encoder window so the causal
            # convolution emits that final partial frame as well.
            final_padding = (-input_values.shape[-1]) % self.hop_length
            if stop == batch.input_values.shape[-1] and final_padding:
                input_values = np.pad(input_values, ((0, 0), (0, final_padding)))
            yield VibeVoiceASRBatch(
                input_values=input_values[:, None, :],
                padding_mask=batch.padding_mask[:, start:stop],
            )

    @staticmethod
    def make_prompt(
        *,
        audio_samples: int,
        context_info: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the original duration and speech-placeholder chat content."""
        if audio_samples <= 0:
            raise ValueError("audio_samples must be positive")
        audio_duration = audio_samples / VibeVoiceASRProcessor.sampling_rate
        audio_tokens = math.ceil(audio_samples / VibeVoiceASRProcessor.hop_length)
        request = (
            f"<|speech_start|>{'<|speech_pad|>' * audio_tokens}<|speech_end|>\n"
            f"This is a {audio_duration:.2f} seconds audio"
        )
        normalized_context = context_info.strip() if context_info else ""
        if normalized_context:
            request = (
                f"{request}, with extra info: {normalized_context}\n\n"
                "Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
            )
        else:
            request = (
                f"{request}, please transcribe it with these keys: "
                "Start time, End time, Speaker ID, Content"
            )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ]

    @staticmethod
    def build_input_ids(
        tokenizer: Any,
        *,
        audio_samples: int,
        context_info: str | None = None,
    ) -> tuple[list[int], list[bool]]:
        """Apply the source's separate system and user chat-template calls."""
        messages = VibeVoiceASRProcessor.make_prompt(
            audio_samples=audio_samples,
            context_info=context_info,
        )
        system_text = tokenizer.apply_chat_template([messages[0]], tokenize=False)
        system_ids = list(tokenizer.encode(system_text))
        user_ids = list(tokenizer.apply_chat_template([messages[1]], tokenize=True))
        input_ids = system_ids + user_ids
        speech_pad_id = tokenizer.convert_tokens_to_ids("<|speech_pad|>")
        return input_ids, [token == speech_pad_id for token in input_ids]

    @staticmethod
    def parse_diarization(text: str) -> list[dict[str, Any]]:
        """Parse the exact source JSON protocol into normalized diarization records."""
        json_text = _extract_json_payload(text)
        try:
            decoded = json.loads(json_text)
        except json.JSONDecodeError as error:
            raise ValueError("VibeVoice-ASR output is not valid diarization JSON") from error
        if isinstance(decoded, Mapping):
            decoded = decoded.get("segments", decoded.get("results", [decoded]))
        if not isinstance(decoded, list):
            raise TypeError(
                "VibeVoice-ASR diarization JSON must be an object or list of objects"
            )

        records: list[dict[str, Any]] = []
        for index, record in enumerate(decoded):
            if not isinstance(record, Mapping):
                raise TypeError(f"VibeVoice-ASR diarization record {index} is not an object")
            normalized = {
                "".join(character for character in key.lower() if character.isalnum()): value
                for key, value in record.items()
                if isinstance(key, str)
            }
            fields = {
                "start_time": normalized.get("starttime", normalized.get("start")),
                "end_time": normalized.get("endtime", normalized.get("end")),
                "speaker_id": normalized.get("speakerid", normalized.get("speaker")),
                "text": normalized.get("content", normalized.get("text")),
            }
            missing = [name for name, value in fields.items() if value is None]
            if missing:
                raise ValueError(
                    f"VibeVoice-ASR diarization record {index} is missing {', '.join(missing)}"
                )
            records.append(fields)
        return records

    def _normalize(self, waveform: np.ndarray, normalize: bool) -> np.ndarray:
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("each waveform must be a non-empty mono 1-D array")
        result = waveform.astype(np.float32, copy=False)
        if not normalize:
            return result
        rms = float(np.sqrt(np.mean(np.square(result), dtype=np.float64)))
        current_dbfs = 20.0 * np.log10(rms + self.eps)
        result = result * np.float32(10.0 ** ((self.target_dbfs - current_dbfs) / 20.0))
        if self.avoid_clipping:
            peak = float(np.max(np.abs(result)))
            if peak > 1.0:
                result = result / np.float32(peak + self.eps)
        return result


class VibeVoiceASRHost:
    """Execute the cache-sensitive encoder/connector portion of a saved package.

    ``run_stage`` takes a package component name plus its NumPy feeds and
    returns its named outputs. Tokenization and autoregressive decoder sampling
    deliberately stay with the caller because tokenizer files are absent from
    the official ASR checkpoint.
    """

    def __init__(
        self,
        run_stage: Callable[[str, Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
        make_initial_conv_cache: Callable[[str, int], Mapping[str, np.ndarray]],
        processor: VibeVoiceASRProcessor | None = None,
    ):
        self._run_stage = run_stage
        self._make_initial_conv_cache = make_initial_conv_cache
        self.processor = processor or VibeVoiceASRProcessor()

    def encode_audio(
        self,
        batch: VibeVoiceASRBatch,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return flattened audio features and per-item valid frame lengths."""
        acoustic_cache = dict(
            self._make_initial_conv_cache("acoustic_encoder", batch.input_values.shape[0])
        )
        semantic_cache = dict(
            self._make_initial_conv_cache("semantic_encoder", batch.input_values.shape[0])
        )
        acoustic_chunks: list[np.ndarray] = []
        semantic_chunks: list[np.ndarray] = []
        for chunk in self.processor.iter_chunks(batch):
            acoustic = self._run_stage(
                "acoustic_encoder",
                {"input_values": chunk.input_values, **acoustic_cache},
            )
            semantic = self._run_stage(
                "semantic_encoder",
                {"input_values": chunk.input_values, **semantic_cache},
            )
            acoustic_chunks.append(np.asarray(acoustic["audio_latents"]))
            semantic_chunks.append(np.asarray(semantic["audio_latents"]))
            acoustic_cache = _next_conv_cache(acoustic)
            semantic_cache = _next_conv_cache(semantic)

        acoustic_latents = np.concatenate(acoustic_chunks, axis=1)
        semantic_latents = np.concatenate(semantic_chunks, axis=1)
        generator = np.random.default_rng(seed)
        acoustic_noise_scale = generator.standard_normal(acoustic_latents.shape[0]).astype(
            acoustic_latents.dtype
        )
        acoustic_latent_noise = generator.standard_normal(acoustic_latents.shape).astype(
            acoustic_latents.dtype
        )
        connector = self._run_stage(
            "connectors",
            {
                "acoustic_latents": acoustic_latents,
                "semantic_latents": semantic_latents,
                "padding_mask": batch.padding_mask,
                "acoustic_noise_scale": acoustic_noise_scale,
                "acoustic_latent_noise": acoustic_latent_noise,
            },
        )
        return (
            np.asarray(connector["audio_features"]),
            np.asarray(connector["audio_feature_lengths"]),
        )


def _next_conv_cache(outputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name.replace("present_conv.", "past_conv."): np.asarray(value)
        for name, value in outputs.items()
        if name.startswith("present_conv.")
    }


def _extract_json_payload(text: str) -> str:
    """Extract one source-style JSON object/list from decoded assistant text."""
    fence = "```json"
    if fence in text:
        start = text.index(fence) + len(fence)
        end = text.find("```", start)
        if end == -1:
            raise ValueError("VibeVoice-ASR output has an unterminated JSON code fence")
        return text[start:end].strip()
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        return text
    start = min(starts)
    depth = 0
    for index, character in enumerate(text[start:], start):
        if character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("VibeVoice-ASR output has an unterminated JSON value")
