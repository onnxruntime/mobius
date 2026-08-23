# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end metadata-driven parity for a real Wav2Vec2 CTC checkpoint.

Exports ``facebook/wav2vec2-base-960h`` with Mobius, publishes the one-file
inference metadata, then drives the whole pipeline — preprocessing, encoder,
frame argmax, CTC collapse, transcript — using nothing but the emitted
document, and compares every stage against HuggingFace on real audio.

This is a frame-synchronous package: the encoder runs exactly once and there is
no generation loop.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
import soundfile as sf
import torch
import transformers

from mobius import build
from mobius._configs import MMSConfig
from mobius.integrations.onnx_genai import ctc_runtime
from mobius.integrations.onnx_genai.auto_export import write_onnx_genai_config

_MODEL_ID = "facebook/wav2vec2-base-960h"
_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
_AUDIO = Path("testdata") / "652-129742-0006.flac"
_EXPECTED = (
    "CAULIFLOWER MAYONAISE TAKE COLD BOILED CULIFLOWER BREAK INTO BRANCHES "
    "ADDING SALT PEPPER AND VINEGAR TO SEASON"
)


def _export(directory: Path) -> None:
    package = build(_MODEL_ID)
    ir.save(
        package["model"],
        str(directory / "model.onnx"),
        external_data="model.onnx.data",
    )
    write_onnx_genai_config(package, str(directory), config=package.config, source=_MODEL_ID)


def _hf_logits(waveform: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    model = transformers.Wav2Vec2ForCTC.from_pretrained(
        _MODEL_ID, revision=_REVISION, dtype=torch.float32
    ).eval()
    inputs = {"input_values": torch.tensor(waveform)}
    if mask is not None:
        inputs["attention_mask"] = torch.tensor(mask)
    with torch.no_grad():
        return model(**inputs).logits.numpy()


@pytest.mark.integration
def test_wav2vec2_ctc_metadata_pipeline_matches_huggingface():
    audio, sample_rate = sf.read(str(_AUDIO))
    audio = audio.astype(np.float32)
    assert np.any(audio != 0)

    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        _export(directory)
        result = ctc_runtime.transcribe(str(directory), [audio], sample_rate)

    processor = transformers.Wav2Vec2Processor.from_pretrained(_MODEL_ID, revision=_REVISION)
    reference = _hf_logits(
        processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_values.numpy()
    )

    # Logits agree to float32 accumulation noise.
    assert result["logits"].shape == reference.shape
    np.testing.assert_allclose(result["logits"], reference, atol=5e-3)

    # Every downstream decode stage agrees exactly.
    assert result["argmax_ids"][0] == reference.argmax(-1)[0].tolist()
    assert result["collapsed_ids"][0] == ctc_runtime.collapse_ctc(
        reference.argmax(-1)[0].tolist(), blank_id=0, collapse_repeats=True
    )
    assert (
        result["transcripts"][0]
        == processor.batch_decode(torch.tensor(reference).argmax(-1))[0]
    )
    assert result["transcripts"][0] == _EXPECTED


@pytest.mark.integration
def test_wav2vec2_ctc_padded_batch_is_segmented_by_declared_frame_lengths():
    audio, sample_rate = sf.read(str(_AUDIO))
    audio = audio.astype(np.float32)
    rows = [audio, audio[: 4 * sample_rate]]  # unequal lengths

    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        _export(directory)
        metadata = ctc_runtime.load_metadata(str(directory))
        preprocessed = ctc_runtime.run_audio_preprocessing(
            metadata["preprocessing"]["audio"], rows, sample_rate
        )
        result = ctc_runtime.transcribe(str(directory), rows, sample_rate)

    _, profile = ctc_runtime.select_profile(metadata, "transcription")
    # Group-normalizing over the padded time axis makes rows interdependent,
    # which the package must declare rather than leave for a caller to discover.
    assert profile["batch_invariance"] == "padding_sensitive"
    assert profile["decoding"]["lengths"] == "frame_lengths"

    config = MMSConfig.from_transformers(
        transformers.AutoConfig.from_pretrained(_MODEL_ID, revision=_REVISION)
    )
    expected_frames = [config.feature_extract_output_length(row.shape[0]) for row in rows]
    assert [len(ids) for ids in result["argmax_ids"]] == expected_frames

    # HuggingFace fed the identical padded batch is the reference for a padded
    # run; a solo run is a different computation for this checkpoint.
    reference = _hf_logits(preprocessed["input_values"], preprocessed["attention_mask"])
    processor = transformers.Wav2Vec2Processor.from_pretrained(_MODEL_ID, revision=_REVISION)
    for row, frames in enumerate(expected_frames):
        valid = reference[row, :frames]
        np.testing.assert_allclose(result["logits"][row, :frames], valid, atol=5e-3)
        assert result["argmax_ids"][row] == valid.argmax(-1).tolist()
        assert (
            result["transcripts"][row]
            == processor.batch_decode(torch.tensor(valid[None]).argmax(-1))[0]
        )
