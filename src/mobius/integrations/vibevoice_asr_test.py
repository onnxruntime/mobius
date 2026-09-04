# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for VibeVoice-ASR host processing and staged audio orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from mobius.integrations.vibevoice_asr import (
    VibeVoiceASRBatch,
    VibeVoiceASRHost,
    VibeVoiceASRProcessor,
)


def test_processor_normalizes_pads_chunks_prompts_and_parses_diarization():
    processor = VibeVoiceASRProcessor()
    short = np.array([1.0, -1.0], dtype=np.float32)
    long = np.ones(processor.chunk_samples + 1, dtype=np.float32)
    batch = processor.prepare_audio([short, long], sampling_rate=24_000)

    assert batch.input_values.shape == (2, processor.chunk_samples + 1)
    assert batch.padding_mask[0].sum() == 2
    assert batch.padding_mask[1].sum() == processor.chunk_samples + 1
    # The normalized sine-like source has the source target RMS level.
    assert np.sqrt(np.mean(np.square(batch.input_values[0, : short.size]))) == pytest.approx(
        10 ** (-25 / 20)
    )
    chunks = list(processor.iter_chunks(batch))
    assert [chunk.input_values.shape for chunk in chunks] == [
        (2, 1, processor.chunk_samples),
        (2, 1, 1),
    ]
    assert processor.make_prompt("Mobius is a hotword.")[1]["content"].endswith(
        "Context information: Mobius is a hotword."
    )
    assert processor.parse_diarization(
        '[{"Start time": 0.0, "End_Time": 1.2, "Speaker ID": "S0", "Content": "hello"}]'
    ) == [{"start_time": 0.0, "end_time": 1.2, "speaker_id": "S0", "text": "hello"}]
    assert processor.parse_diarization(
        '[{"Start": 1.2, "End": 2.4, "Speaker": "S1", "Text": "world"}]'
    ) == [{"start_time": 1.2, "end_time": 2.4, "speaker_id": "S1", "text": "world"}]


def test_processor_rejects_wrong_rate_and_malformed_diarization():
    processor = VibeVoiceASRProcessor()
    with pytest.raises(ValueError, match="24000 Hz"):
        processor.prepare_audio(np.ones(10, dtype=np.float32), sampling_rate=16_000)
    with pytest.raises(ValueError, match="not valid diarization JSON"):
        processor.parse_diarization("not-json")
    with pytest.raises(ValueError, match="missing speaker_id"):
        processor.parse_diarization('[{"Start time": 0, "End time": 1, "Content": "hello"}]')


def test_host_preserves_chunk_cache_and_uses_seeded_connector_noise():
    processor = VibeVoiceASRProcessor()
    processor.chunk_samples = 4
    calls: list[tuple[str, dict[str, np.ndarray]]] = []

    def initial_cache(stage: str, batch_size: int) -> dict[str, np.ndarray]:
        assert stage in {"acoustic_encoder", "semantic_encoder"}
        return {"past_conv.0": np.zeros((batch_size, 1, 1), dtype=np.float32)}

    def run_stage(stage: str, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        calls.append((stage, feeds))
        if stage.endswith("encoder"):
            value = feeds["input_values"].mean(axis=-1, keepdims=False)[..., None]
            return {
                "audio_latents": value,
                "present_conv.0": feeds["past_conv.0"] + 1,
            }
        assert stage == "connectors"
        return {
            "audio_features": feeds["acoustic_latents"].reshape(-1, 1),
            "audio_feature_lengths": np.array([2], dtype=np.int64),
        }

    host = VibeVoiceASRHost(run_stage, initial_cache, processor)
    features, lengths = host.encode_audio(
        VibeVoiceASRBatch(
            input_values=np.ones((1, 6), dtype=np.float32),
            padding_mask=np.ones((1, 6), dtype=np.bool_),
        ),
        seed=9,
    )

    assert features.shape == (2, 1)
    assert lengths.tolist() == [2]
    # The second encoder window must consume the first window's state.
    assert calls[2][1]["past_conv.0"].item() == 1
    connector_feeds = calls[-1][1]
    assert connector_feeds["acoustic_noise_scale"].shape == (1,)
    assert connector_feeds["acoustic_latent_noise"].shape == (1, 2, 1)
