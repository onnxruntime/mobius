# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
import torch
from transformers import (
    ParakeetCTCConfig as HFParakeetCTCConfig,
)
from transformers import (
    ParakeetEncoderConfig as HFParakeetEncoderConfig,
)
from transformers import (
    ParakeetForCTC as HFParakeetForCTC,
)

from mobius import build_from_module
from mobius._configs import ParakeetCTCConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations._weight_loading import apply_weights
from mobius.integrations.ort_genai import write_ort_genai_config
from mobius.models import ParakeetForCTCModel
from mobius.tasks import FeatureCTCAsrTask


def _hf_config() -> HFParakeetCTCConfig:
    encoder = HFParakeetEncoderConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_mel_bins=8,
        subsampling_conv_channels=4,
        conv_kernel_size=5,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=64,
    )
    return HFParakeetCTCConfig(
        encoder_config=encoder,
        vocab_size=17,
        pad_token_id=16,
    )


def _build_tiny():
    hf_config = _hf_config()
    config = ParakeetCTCConfig.from_transformers(hf_config)
    config.dtype = ir.DataType.FLOAT
    module = ParakeetForCTCModel(config)
    package = build_from_module(module, config, task=FeatureCTCAsrTask())
    return hf_config, config, module, package


def test_parakeet_config_extracts_nested_encoder_fields():
    hf_config = _hf_config()
    hf_config.dtype = torch.bfloat16
    config = ParakeetCTCConfig.from_transformers(hf_config)

    assert config.model_type == "parakeet_ctc"
    assert config.vocab_size == 17
    assert config.pad_token_id == 16
    assert config.hidden_size == 16
    assert config.num_mel_bins == 8
    assert config.subsampling_factor == 8
    assert config.conv_kernel_size == 5
    assert config.dtype == ir.DataType.FLOAT


def test_parakeet_rejects_bfloat16():
    config = ParakeetCTCConfig.from_transformers(_hf_config())
    config.dtype = ir.DataType.BFLOAT16

    with pytest.raises(ValueError, match="bf16 is disabled"):
        ParakeetForCTCModel(config)


def test_parakeet_graph_io_and_hf_weight_names_align():
    hf_config, _, module, package = _build_tiny()
    hf_model = HFParakeetForCTC(hf_config)
    processed = module.preprocess_weights(dict(hf_model.state_dict()))
    model = package["model"]

    assert [value.name for value in model.graph.inputs] == [
        "input_features",
        "attention_mask",
    ]
    assert [value.name for value in model.graph.outputs] == ["logits"]
    parameter_names = {
        name
        for name, initializer in model.graph.initializers.items()
        if initializer.const_value is None
    }
    assert set(processed) == parameter_names
    assert not any(name.endswith(".num_batches_tracked") for name in processed)


def test_parakeet_graph_uses_fused_encoder_ops():
    _, config, _, package = _build_tiny()
    model = package["model"]
    op_types = [node.op_type for node in model.graph.all_nodes()]

    assert op_types.count("Attention") == config.num_hidden_layers
    assert op_types.count("BatchNormalization") == config.num_hidden_layers
    assert op_types.count("Swish") == 3 * config.num_hidden_layers
    assert op_types.count("SkipLayerNormalization") == 4 * config.num_hidden_layers
    assert "Sqrt" not in op_types


def test_parakeet_honors_non_silu_activation_in_ffn_and_convolution():
    hf_config = _hf_config()
    config = dataclasses.replace(
        ParakeetCTCConfig.from_transformers(hf_config),
        hidden_act="relu",
        dtype=ir.DataType.FLOAT,
    )
    module = ParakeetForCTCModel(config)
    model = build_from_module(module, config, task=FeatureCTCAsrTask())["model"]

    op_types = [node.op_type for node in model.graph]
    # The three configurable activations are the two Macaron FFNs and the
    # convolution module. Subsampling contributes three fixed ReLUs.
    assert op_types.count("Relu") == 6
    assert "Swish" not in op_types


def test_parakeet_synthetic_parity_with_padding():
    torch.manual_seed(42)
    hf_config, _, module, package = _build_tiny()
    hf_model = HFParakeetForCTC(hf_config).float().eval()
    apply_weights(
        package["model"],
        module.preprocess_weights(dict(hf_model.state_dict())),
    )

    rng = np.random.default_rng(123)
    input_features = rng.standard_normal((2, 24, 8)).astype(np.float32)
    attention_mask = np.ones((2, 24), dtype=bool)
    attention_mask[1, 17:] = False
    input_features[1, 17:] = 0.0

    with torch.no_grad():
        expected = hf_model(
            input_features=torch.from_numpy(input_features),
            attention_mask=torch.from_numpy(attention_mask),
        ).logits.numpy()

    session = OnnxModelSession(package["model"], device="cpu")
    try:
        actual = session.run(
            {
                "input_features": input_features,
                "attention_mask": attention_mask,
            }
        )["logits"]
    finally:
        session.close()

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_parakeet_exports_with_unsupported_runtime_status(tmp_path):
    _, _, _, package = _build_tiny()

    result = write_ort_genai_config(package, str(tmp_path))
    compatibility = json.loads(
        Path(result["runtime_compatibility"]).read_text(encoding="utf-8")
    )
    assert compatibility["runtime_validation_status"] == "unsupported-by-tested-runtime"
    assert "feature-input CTC ASR pipeline" in compatibility["warnings"][0]
