# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for ORT GenAI nemotron_speech config derivation."""

from __future__ import annotations

from mobius.integrations.nemo._genai_config import _LOG_EPS, _derive_params

_NEMO_CONFIG = {
    "preprocessor": {
        "sample_rate": 16000,
        "features": 128,
        "n_fft": 512,
        "window_size": 0.025,
        "window_stride": 0.01,
        "dither": 1e-05,
        "normalize": "NA",
    },
    "encoder": {
        "feat_in": 128,
        "d_model": 1024,
        "n_layers": 24,
        "subsampling_factor": 8,
        "conv_kernel_size": 9,
        "att_context_size": [[70, 13], [70, 6], [70, 1], [70, 0]],
    },
    "decoder": {"prednet": {"pred_hidden": 640, "pred_rnn_layers": 2}},
    "joint": {"num_classes": 1024},
    "decoding": {"greedy": {"max_symbols": 10}},
}


def test_derive_params_matches_architecture():
    p = _derive_params(_NEMO_CONFIG, chunk_seconds=1.12)
    # vocab/blank: num_classes + 1, blank is the final index.
    assert p["vocab_size"] == 1025
    assert p["blank_id"] == 1024
    # mel front-end (seconds -> samples).
    assert p["num_mels"] == 128
    assert p["fft_size"] == 512
    assert p["win_length"] == 400
    assert p["hop_length"] == 160
    assert p["sample_rate"] == 16000
    # encoder geometry.
    assert p["hidden_size"] == 1024
    assert p["num_hidden_layers"] == 24
    assert p["subsampling_factor"] == 8
    assert p["left_context"] == 70  # att_context[0][0]
    assert p["conv_context"] == 8  # conv_kernel_size - 1
    assert p["pre_encode_cache_size"] == 9  # NeMo default for 8x stem
    assert p["chunk_samples"] == round(1.12 * 16000)
    # decoder LSTM geometry.
    assert p["decoder_hidden"] == 640
    assert p["decoder_layers"] == 2
    assert p["max_symbols_per_step"] == 10


def test_derive_params_preemph_default():
    # The preprocessor omits preemph; NeMo's default (0.97) must be used.
    p = _derive_params(_NEMO_CONFIG, chunk_seconds=0.56)
    assert p["preemph"] == 0.97
    assert p["chunk_samples"] == round(0.56 * 16000)


def test_log_eps_is_nemo_zero_guard():
    # 2**-24, NeMo AudioToMelSpectrogramPreprocessor log zero-guard.
    assert abs(_LOG_EPS - 2.0**-24) < 1e-16
