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
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

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
    target_dB_FS = -25.0
    eps = 1e-6

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
                raise ValueError("waveforms array must have shape (samples,) or (batch, samples)")
        else:
            source = waveforms
        if not isinstance(source, list) or not source:
            raise ValueError("waveforms must be one non-empty 1-D array or a non-empty list of them")
        normalized = [self._normalize(np.asarray(waveform), normalize) for waveform in source]
        max_samples = max(waveform.size for waveform in normalized)
        input_values = np.zeros((len(normalized), max_samples), dtype=np.float32)
        padding_mask = np.zeros((len(normalized), max_samples), dtype=np.bool_)
        for index, waveform in enumerate(normalized):
            input_values[index, : waveform.size] = waveform
            padding_mask[index, : waveform.size] = True
        return VibeVoiceASRBatch(input_values=input_values, padding_mask=padding_mask)

    def iter_chunks(self, batch: VibeVoiceASRBatch) -> Iterator[VibeVoiceASRBatch]:
        """Yield 60-second waveform windows; encoder cache bridges adjacent windows."""
        for start in range(0, batch.input_values.shape[-1], self.chunk_samples):
            stop = min(start + self.chunk_samples, batch.input_values.shape[-1])
            yield VibeVoiceASRBatch(
                input_values=batch.input_values[:, start:stop, None].transpose(0, 2, 1),
                padding_mask=batch.padding_mask[:, start:stop],
            )

    @staticmethod
    def make_prompt(context_info: str | None = None) -> list[dict[str, str]]:
        """Build the source chat request; ``context_info`` is its hotword mechanism."""
        request = (
            "Please transcribe the audio with the following JSON keys: Start time, End time, "
            "Speaker ID, Content."
        )
        if context_info:
            request = f"{request}\nContext information: {context_info}"
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ]

    @staticmethod
    def parse_diarization(text: str) -> list[dict[str, Any]]:
        """Parse the exact source JSON protocol into normalized diarization records."""
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("VibeVoice-ASR output is not valid diarization JSON") from error
        if isinstance(decoded, Mapping):
            decoded = decoded.get("segments", decoded.get("results", [decoded]))
        if not isinstance(decoded, list):
            raise ValueError("VibeVoice-ASR diarization JSON must be an object or list of objects")

        records: list[dict[str, Any]] = []
        for index, record in enumerate(decoded):
            if not isinstance(record, Mapping):
                raise ValueError(f"VibeVoice-ASR diarization record {index} is not an object")
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
        if rms <= self.eps:
            return result
        current_dB_FS = 20.0 * np.log10(rms)
        return result * np.float32(10.0 ** ((self.target_dB_FS - current_dB_FS) / 20.0))


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
        acoustic_cache = dict(self._make_initial_conv_cache("acoustic_encoder", batch.input_values.shape[0]))
        semantic_cache = dict(self._make_initial_conv_cache("semantic_encoder", batch.input_values.shape[0]))
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
