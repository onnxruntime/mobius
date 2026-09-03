# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Real-checkpoint parity for IBM Granite Speech 5 Turbo CTC."""

from __future__ import annotations

import gc
from pathlib import Path

import librosa
import numpy as np
import pytest
import torch
import transformers

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

_MODEL_ID = "ibm-granite/granite-speech-5.0-470m-turboctc"
_REVISION = "7e74c6438b7cfb5090cb6a131538f5e8515a7de3"
_AUDIO = Path("testdata") / "652-129742-0006.flac"


def _require_native_transformers() -> None:
    if not hasattr(transformers, "GraniteSpeech5ForCTC"):
        pytest.skip("Granite Speech 5 requires Transformers 5.16.0 or newer")


def _audio_rows() -> tuple[list[np.ndarray], int]:
    audio, sample_rate = librosa.load(str(_AUDIO), sr=16_000)
    assert sample_rate == 16_000
    assert np.any(audio != 0)
    # The second row is both shorter and acoustically distinct, exercising
    # padding masks and row isolation rather than duplicating one utterance.
    return [audio, -audio[: 4 * sample_rate]], sample_rate


def _processor_inputs(rows: list[np.ndarray], sample_rate: int) -> dict[str, np.ndarray]:
    processor = transformers.AutoProcessor.from_pretrained(_MODEL_ID, revision=_REVISION)
    processed = processor(
        rows,
        sampling_rate=sample_rate,
        return_tensors="np",
        padding=True,
    )
    return {
        "input_features": np.asarray(processed["input_features"], dtype=np.float32),
        "attention_mask": np.asarray(processed["attention_mask"], dtype=np.int64),
    }


def _hf_logits(
    inputs: dict[str, np.ndarray],
    dtype: torch.dtype,
) -> np.ndarray:
    model = transformers.AutoModelForCTC.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        torch_dtype=dtype,
    ).eval()
    model.cuda()
    try:
        with torch.no_grad():
            outputs = model(
                input_features=torch.from_numpy(inputs["input_features"]).cuda(),
                attention_mask=torch.from_numpy(inputs["attention_mask"]).cuda(),
            )
        return outputs.logits.float().cpu().numpy()
    finally:
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _run_onnx(package, inputs: dict[str, np.ndarray], device: str):
    session = OnnxModelSession(package["model"], device=device)
    try:
        return session.run(inputs)
    finally:
        session.close()


def _decode_valid_frames(
    logits: np.ndarray,
    frame_lengths: np.ndarray,
) -> tuple[list[list[int]], list[str]]:
    processor = transformers.AutoProcessor.from_pretrained(_MODEL_ID, revision=_REVISION)
    frame_ids = [
        logits[row, : int(frame_lengths[row])].argmax(-1).astype(np.int64).tolist()
        for row in range(logits.shape[0])
    ]
    return frame_ids, processor.batch_decode(frame_ids, skip_special_tokens=True)


@pytest.mark.integration
def test_granite_speech5_real_weight_full_logits_cpu_cuda_and_padding():
    _require_native_transformers()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Granite Speech 5 real-weight parity")

    rows, sample_rate = _audio_rows()
    inputs = _processor_inputs(rows, sample_rate)
    assert inputs["input_features"].shape[0] == 2
    assert inputs["input_features"].shape[2] == 320
    assert inputs["input_features"].dtype == np.float32
    assert inputs["attention_mask"].dtype == np.int64
    assert np.any(inputs["input_features"][0] != 0)
    assert not np.array_equal(
        inputs["input_features"][0, :32],
        inputs["input_features"][1, :32],
    )

    expected = _hf_logits(inputs, torch.float32)
    package = build(
        _MODEL_ID,
        revision=_REVISION,
        dtype="f32",
        execution_provider="default",
    )
    cpu = _run_onnx(package, inputs, "cpu")
    cuda = _run_onnx(package, inputs, "cuda")

    expected_lengths = inputs["attention_mask"].sum(1) // 4
    np.testing.assert_array_equal(cpu["frame_lengths"], expected_lengths)
    np.testing.assert_array_equal(cuda["frame_lengths"], expected_lengths)
    assert cpu["logits"].shape == expected.shape == cuda["logits"].shape
    for row, length in enumerate(expected_lengths):
        valid = int(length)
        # Padded-tail logits are not part of the CTC sequence; compare every
        # vocabulary logit at every valid frame instead of accepting a prefix.
        np.testing.assert_allclose(
            cpu["logits"][row, :valid],
            expected[row, :valid],
            atol=3e-3,
            rtol=3e-3,
        )
        np.testing.assert_allclose(
            cuda["logits"][row, :valid],
            expected[row, :valid],
            atol=3e-3,
            rtol=3e-3,
        )
        np.testing.assert_allclose(
            cuda["logits"][row, :valid],
            cpu["logits"][row, :valid],
            atol=3e-3,
            rtol=3e-3,
        )

    # A padded row must reproduce the same valid logits when run alone.
    feature_length = int(inputs["attention_mask"][1].sum())
    solo_inputs = {
        # Slice the exact batched processor features so this isolates model
        # padding invariance from the delta extractor's right-boundary rule.
        "input_features": inputs["input_features"][1:2, :feature_length],
        "attention_mask": inputs["attention_mask"][1:2, :feature_length],
    }
    solo = _run_onnx(package, solo_inputs, "cuda")
    valid = int(cuda["frame_lengths"][1])
    assert valid == int(solo["frame_lengths"][0])
    np.testing.assert_allclose(
        cuda["logits"][1, :valid],
        solo["logits"][0, :valid],
        atol=3e-3,
        rtol=3e-3,
    )

    frame_ids, transcripts = _decode_valid_frames(cuda["logits"], cuda["frame_lengths"])
    assert [len(ids) for ids in frame_ids] == expected_lengths.tolist()
    assert transcripts[0]
    assert transcripts[1]
    assert transcripts[0] != transcripts[1]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mobius_dtype", "torch_dtype", "atol"),
    [
        pytest.param("f16", torch.float16, 5e-2, id="fp16"),
        pytest.param("bf16", torch.bfloat16, 5e-1, id="bf16"),
    ],
)
def test_granite_speech5_reduced_precision_full_logits_and_transcript(
    mobius_dtype: str,
    torch_dtype: torch.dtype,
    atol: float,
):
    _require_native_transformers()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Granite Speech 5 reduced-precision parity")

    rows, sample_rate = _audio_rows()
    inputs = _processor_inputs(rows[:1], sample_rate)
    expected = _hf_logits(inputs, torch_dtype)
    package = build(
        _MODEL_ID,
        revision=_REVISION,
        dtype=mobius_dtype,
        execution_provider="default",
    )
    actual = _run_onnx(package, inputs, "cuda")

    frame_lengths = inputs["attention_mask"].sum(1) // 4
    np.testing.assert_array_equal(actual["frame_lengths"], frame_lengths)
    assert actual["logits"].shape == expected.shape
    actual_f32 = actual["logits"].astype(np.float32)
    np.testing.assert_allclose(actual_f32, expected, atol=atol, rtol=atol)
    absolute_error = np.abs(actual_f32 - expected)
    cosine = np.dot(actual_f32.ravel(), expected.ravel()) / (
        np.linalg.norm(actual_f32) * np.linalg.norm(expected)
    )
    # BF16 CUDA kernels differ from PyTorch by roughly one BF16 ULP on many
    # logits after 16 conformer blocks; bound both the aggregate drift and the
    # final semantic sequence instead of relying on the 0.5 max bound alone.
    assert float(absolute_error.mean()) < 8e-2
    assert float(cosine) > 0.9999

    expected_ids, expected_text = _decode_valid_frames(expected, frame_lengths)
    actual_ids, actual_text = _decode_valid_frames(actual["logits"], frame_lengths)
    assert len(actual_ids[0]) == len(expected_ids[0]) == int(frame_lengths[0])
    assert actual_ids == expected_ids
    assert actual_text == expected_text
    assert actual_text[0]
