# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Real-checkpoint CUDA parity for NVIDIA Parakeet CTC."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import librosa
import numpy as np
import onnx_ir as ir
import pytest
import torch
import transformers

from mobius import build, build_from_module
from mobius._configs import ParakeetCTCConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius._weight_loading import apply_weights
from mobius.models import ParakeetForCTCModel
from mobius.tasks import FeatureCTCAsrTask

_MODEL_ID = "nvidia/parakeet-ctc-1.1b"
_REVISION = "20e63a0fed6aedba145b74b826dbd41df0941730"


@pytest.mark.integration
def test_parakeet_real_audio_real_weight_cuda_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Parakeet 1.1B real-weight parity")

    processor = transformers.AutoProcessor.from_pretrained(_MODEL_ID, revision=_REVISION)
    audio, sample_rate = librosa.load(
        str(Path("testdata") / "652-129742-0006.flac"),
        sr=16_000,
    )
    assert np.any(audio != 0)
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    hf_model = transformers.AutoModelForCTC.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        torch_dtype=torch.float32,
    ).eval()
    hf_model.cuda()
    with torch.no_grad():
        expected = (
            hf_model(**{name: value.cuda() for name, value in inputs.items()})
            .logits.cpu()
            .numpy()
        )
    hf_model.cpu()
    torch.cuda.empty_cache()

    config = ParakeetCTCConfig.from_transformers(hf_model.config)
    config.dtype = ir.DataType.FLOAT
    module = ParakeetForCTCModel(config)
    package = build_from_module(module, config, task=FeatureCTCAsrTask())
    apply_weights(
        package["model"],
        module.preprocess_weights(dict(hf_model.state_dict())),
    )
    del hf_model
    gc.collect()

    session = OnnxModelSession(package["model"], device="cuda")
    try:
        actual = session.run(
            {
                "input_features": inputs["input_features"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy().astype(bool),
            }
        )["logits"]
    finally:
        session.close()

    absolute_error = np.abs(actual - expected)
    print(
        "Parakeet CUDA parity: "
        f"max_abs_diff={absolute_error.max():.6f}, "
        f"mean_abs_diff={absolute_error.mean():.6f}"
    )
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.integration
def test_parakeet_real_audio_fp16_cuda_matches_ctc_frames():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Parakeet 1.1B fp16 validation")

    processor = transformers.AutoProcessor.from_pretrained(_MODEL_ID, revision=_REVISION)
    audio, sample_rate = librosa.load(
        str(Path("testdata") / "652-129742-0006.flac"),
        sr=16_000,
    )
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="np",
    )
    package = build(
        _MODEL_ID,
        revision=_REVISION,
        dtype="f16",
        execution_provider="cuda",
    )
    session = OnnxModelSession(package["model"], device="cuda")
    try:
        logits = session.run(
            {
                "input_features": inputs["input_features"].astype(np.float16),
                "attention_mask": inputs["attention_mask"].astype(bool),
            }
        )["logits"]
    finally:
        session.close()

    with open(
        Path("testdata") / "golden" / "audio" / "parakeet-ctc-1.1b_generation.json"
    ) as golden_file:
        expected_ids = np.array(json.load(golden_file)["generated_tokens"], dtype=np.int64)
    actual_ids = np.argmax(logits[0], axis=-1)
    np.testing.assert_array_equal(actual_ids, expected_ids)
    assert (
        processor.batch_decode(actual_ids[np.newaxis, :])[0]
        == "cauliflower mayonnaise take cold boiled cauliflower break into "
        "branches adding salt pepper and vinegar to season"
    )
