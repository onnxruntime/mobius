# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
import types

import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius._configs import GraniteSpeech5CTCConfig
from mobius.models.granite_speech5 import GraniteSpeech5ForCTCModel
from mobius.tasks import FeatureCTCAsrTask


def _tiny_config(**overrides) -> GraniteSpeech5CTCConfig:
    values = {
        "model_type": "granite_speech5_ctc",
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 16,
        "num_mel_bins": 4,
        "input_feature_size": 16,
        "context_size": 8,
        "conv_kernel_size": 3,
        "conv_expansion_factor": 2,
        "subsample_layers": (0, 1),
        "attention_bias": True,
        "pad_token_id": 0,
        "dtype": ir.DataType.FLOAT,
    }
    values.update(overrides)
    return GraniteSpeech5CTCConfig(**values)


def _build_tiny(config: GraniteSpeech5CTCConfig | None = None):
    config = config or _tiny_config()
    module = GraniteSpeech5ForCTCModel(config)
    return config, module, build_from_module(module, config, task=FeatureCTCAsrTask())


def test_granite_speech5_extracts_native_nested_config():
    encoder = types.SimpleNamespace(
        model_type="granite_speech5_encoder",
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=16,
        num_mel_bins=4,
        context_size=8,
        conv_kernel_size=3,
        conv_expansion_factor=2,
        subsample_layers=[0, 1],
        attention_bias=True,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    parent = types.SimpleNamespace(
        model_type="granite_speech5_ctc",
        encoder_config=encoder,
        vocab_size=32,
        pad_token_id=0,
        tie_word_embeddings=True,
        dtype=torch.bfloat16,
    )

    config = GraniteSpeech5CTCConfig.from_transformers(parent)

    assert config.model_type == "granite_speech5_ctc"
    assert config.input_feature_size == 16
    assert config.subsample_layers == (0, 1)
    assert config.dtype == ir.DataType.BFLOAT16
    assert config.tie_word_embeddings is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"input_feature_size": 15}, "num_mel_bins \\* 4"),
        ({"context_size": 17}, "context_size"),
        ({"conv_kernel_size": 4}, "must be odd"),
        ({"subsample_layers": (0, 4)}, "existing layers"),
    ],
)
def test_granite_speech5_rejects_invalid_architecture(overrides, message):
    with pytest.raises(ValueError, match=message):
        _tiny_config(**overrides)


def test_granite_speech5_graph_contract_and_weight_names():
    config, module, package = _build_tiny()
    model = package["model"]

    assert [value.name for value in model.graph.inputs] == [
        "input_features",
        "attention_mask",
    ]
    assert [value.dtype for value in model.graph.inputs] == [
        ir.DataType.FLOAT,
        ir.DataType.INT64,
    ]
    assert list(model.graph.inputs[0].shape) == ["batch", "frames", 16]
    assert [value.name for value in model.graph.outputs] == ["logits", "frame_lengths"]

    parameter_names = {
        name
        for name, initializer in model.graph.initializers.items()
        if initializer.const_value is None
    }
    assert "encoder.input_linear.weight" in parameter_names
    assert "encoder.layers.0.self_attn.rel_pos_emb.weight" in parameter_names
    assert "encoder.layers.0.conv.depthwise_conv.weight" in parameter_names
    assert "encoder.out.weight" in parameter_names
    assert "ctc_head.weight" not in parameter_names

    state_dict = {name: object() for name in parameter_names}
    state_dict["ctc_head.weight"] = object()
    state_dict["encoder.layers.0.conv.norm.num_batches_tracked"] = object()
    assert set(module.preprocess_weights(state_dict)) == parameter_names

    op_types = [node.op_type for node in model.graph.all_nodes()]
    assert op_types.count("Attention") == config.num_hidden_layers
    assert "BatchNormalization" not in op_types
    assert op_types.count("Sqrt") == config.num_hidden_layers
    assert op_types.count("Softmax") == 1
    assert op_types.count("Swish") == 3 * config.num_hidden_layers


def test_granite_speech5_audio_processor_forwards_revision(tmp_path, monkeypatch):
    from mobius.integrations.ort_genai.auto_export import _write_audio_processor_config
    from transformers import AutoFeatureExtractor

    calls = []

    class _FeatureExtractor:
        @staticmethod
        def to_json_file(path):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"feature_extractor_type": "GraniteSpeech5FeatureExtractor"}, handle)

    def _from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return _FeatureExtractor()

    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained", _from_pretrained)
    revision = "7e74c6438b7cfb5090cb6a131538f5e8515a7de3"
    path = _write_audio_processor_config(
        _tiny_config(),
        str(tmp_path),
        hf_model_id="ibm-granite/granite-speech-5.0-470m-turboctc",
        revision=revision,
    )

    assert calls == [
        (
            "ibm-granite/granite-speech-5.0-470m-turboctc",
            {"revision": revision, "trust_remote_code": False},
        )
    ]
    assert path == str(tmp_path / "audio_processor.json")


def test_granite_speech5_genai_metadata_does_not_invent_kv_cache(tmp_path):
    from mobius.integrations.ort_genai.auto_export import _write_genai_config

    config, _, package = _build_tiny()
    path = _write_genai_config(
        config,
        str(tmp_path),
        pkg=package,
        ort_model_type=config.model_type,
        ep="cpu",
        context_length=4096,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=0,
        is_vlm=False,
        has_speech=False,
    )
    with open(path, encoding="utf-8") as handle:
        generated = json.load(handle)

    decoder = generated["model"]["decoder"]
    assert decoder["inputs"] == {
        "input_features": "input_features",
        "attention_mask": "attention_mask",
    }
    assert decoder["outputs"] == {
        "logits": "logits",
        "frame_lengths": "frame_lengths",
    }
