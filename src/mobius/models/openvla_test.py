from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch
from transformers import PreTrainedConfig

from mobius import build_from_module
from mobius._configs import OpenVLAConfig, VisionConfig
from mobius.models import OpenVLAModel
from mobius.tasks import OpenVLATask


def _tiny_config():
    tower = VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
        norm_eps=1e-6,
    )
    return OpenVLAConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
        max_position_embeddings=64,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_type="default",
        dino=tower,
        siglip=dataclasses.replace(tower),
    )


def test_raw_generic_config_materializes_llama2_defaults():
    text = PreTrainedConfig(model_type="llama", vocab_size=32064)
    source = SimpleNamespace(
        text_config=text,
        use_fused_vision_backbone=True,
        timm_model_ids=[
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ],
        image_sizes=[224, 224],
    )
    config = OpenVLAConfig.from_transformers(text, source)
    config.validate()
    assert config.model_type == "openvla"
    assert (config.hidden_size, config.num_hidden_layers, config.vocab_size) == (
        4096,
        32,
        32064,
    )
    assert config.dino.intermediate_size == 4096
    assert config.siglip.intermediate_size == 4304


@pytest.mark.parametrize(
    "change",
    [
        {"num_hidden_layers": 1},
        {"hidden_size": 33},
        {"patch_size": 13},
        {"image_size": 42},
    ],
)
def test_rejects_invalid_vision_geometry(change):
    config = _tiny_config()
    config.dino = dataclasses.replace(config.dino, **change)
    with pytest.raises(ValueError, match="OpenVLA"):
        config.validate()


@pytest.mark.parametrize("ep", ["default", "cpu"])
def test_three_components_and_exact_source_weight_names(ep):
    config = _tiny_config()
    module = OpenVLAModel(config)
    package = build_from_module(module, config, task="openvla", execution_provider=ep)
    assert set(package) == {"decoder", "vision_encoder", "embedding"}
    vision = package["vision_encoder"].graph
    assert list(vision.inputs[0].shape) == ["batch", 6, 28, 28]
    parameters = dict(module.named_parameters())
    assert tuple(parameters["vision_encoder.vision_backbone.featurizer.pos_embed"].shape) == (
        1,
        4,
        32,
    )
    assert tuple(parameters["vision_encoder.vision_backbone.featurizer.reg_token"].shape) == (
        1,
        4,
        32,
    )
    assert len(module._omitted_shapes) == 43
    assert module.omitted_weight_reason("vision_backbone.featurizer.norm.weight")
    assert module.omitted_weight_reason("vision_backbone.featurizer.norm.made_up") is None
    assert module.omitted_weight_reason("vision_backbone.featurizer.blocks.1.attn.qkv.weight")
    assert (
        module.omitted_weight_reason("vision_backbone.featurizer.blocks.0.attn.qkv.weight")
        is None
    )


@pytest.mark.parametrize(
    "name,shape,message",
    [
        ("other.weight", (1,), "Unaccounted"),
        ("projector.unknown.weight", (1,), "Unknown"),
        ("projector.fc1.weight", (1,), "shape mismatch"),
        ("vision_backbone.featurizer.norm.weight", (1,), "omitted checkpoint shape"),
        ("vision_backbone.featurizer.norm.unknown", (32,), "Unknown"),
    ],
)
def test_weight_mapping_rejects_unknown_names_and_wrong_shapes(name, shape, message):
    module = OpenVLAModel(_tiny_config())
    with pytest.raises(ValueError, match=message):
        module.preprocess_weights({name: torch.zeros(shape)})


def test_weight_mapping_and_explicit_unused_parameter():
    module = OpenVLAModel(_tiny_config())
    embedding = torch.zeros((64, 32))
    result = module.preprocess_weights(
        {
            "language_model.model.embed_tokens.weight": embedding,
            "language_model.lm_head.weight": embedding,
            "vision_backbone.featurizer.norm.weight": torch.ones(32),
        }
    )
    assert set(result) == {"embedding.embed_tokens.weight", "decoder.lm_head.weight"}
    assert result["embedding.embed_tokens.weight"] is embedding


@pytest.mark.parametrize("image_sequence", [0, 4])
def test_bos_insertion_and_empty_image_decode_in_actual_ort(tmp_path, image_sequence):
    ort = pytest.importorskip("onnxruntime")
    config = _tiny_config()
    module = OpenVLAModel(config)
    package = OpenVLATask().build(module, config)
    weight = np.arange(64 * 32, dtype=np.float32).reshape(64, 32) / 1000
    package.apply_weights({"embedding.embed_tokens.weight": torch.from_numpy(weight)})
    model_path = tmp_path / "embedding.onnx"
    ir.save(package["embedding"], model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    tokens = np.array([[1, 7, 9], [1, 3, 4]], dtype=np.int64)
    features = np.full((2, image_sequence, 32), -0.5, dtype=np.float32)
    actual = session.run(None, {"input_ids": tokens, "image_features": features})[0]
    text = weight[tokens]
    expected = np.concatenate((text[:, :1], features, text[:, 1:]), axis=1)
    np.testing.assert_array_equal(actual, expected)
