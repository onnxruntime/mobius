# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned real-weight waveform parity for the standalone LiquidCodec decoder."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mobius._optimizations import optimize_model
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.liquid_codec import (
    LiquidCodecConfig,
    LiquidCodecDecoder,
    build_liquid_codec_decoder,
)

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
REVISION = "2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1"


def _load_reference():
    config_class = get_class_from_dynamic_module(
        "configuration_liquid_codec.LiquidCodecConfig", MODEL_ID, revision=REVISION
    )
    model_class = get_class_from_dynamic_module(
        "modeling_liquid_codec.LiquidCodecModel", MODEL_ID, revision=REVISION
    )
    config = config_class.from_pretrained(
        MODEL_ID, revision=REVISION, subfolder="audio_tokenizer"
    )
    model = model_class.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        subfolder="audio_tokenizer",
        config=config,
        dtype=torch.float32,
    ).eval()
    model.decoder.lfm.config.layer_types = [
        "conv" if kind == "linear_attention" else kind
        for kind in model.decoder.lfm.config.layer_types
    ]
    return model


@pytest.mark.integration
def test_nonzero_codes_match_pinned_upstream_waveform():
    reference = _load_reference()
    config = LiquidCodecConfig.from_transformers(reference.config)
    model = build_liquid_codec_decoder(config)
    weights = LiquidCodecDecoder.preprocess_weights(reference.state_dict())
    missing = {
        name
        for name, initializer in model.graph.initializers.items()
        if initializer.const_value is None and name not in weights
    }
    assert not missing, f"Missing codec decoder weights: {sorted(missing)}"
    for name, value in weights.items():
        if name in model.graph.initializers:
            model.graph.initializers[name].const_value = ir.tensor(
                value.detach().float().cpu().numpy()
            )
    optimize_model(model, ep="cpu", model_role="decoder")

    codes = torch.arange(config.n_q, dtype=torch.int64).reshape(1, config.n_q, 1)
    codes = (codes * 137 + 11) % config.codebook_size
    with torch.no_grad():
        expected = reference.decode(codes, return_dict=True).audio_values.float().numpy()

    session = OnnxModelSession(model)
    try:
        actual = session.run({"codes": codes.numpy()})["waveform"]
    finally:
        session.close()

    assert actual.dtype == np.float32
    assert actual.shape == expected.shape == (1, 1920)
    assert np.max(np.abs(expected)) > 1e-4
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)
