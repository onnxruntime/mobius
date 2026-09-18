# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for Cosmos3-Edge weight routing and projector configuration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mobius._configs import ArchitectureConfig, VisionConfig
from mobius._configs.per_model._cosmos3_edge_vision import _cosmos3_edge_vision
from mobius.models.cosmos import (
    Cosmos3EdgeTextModel,
    Cosmos3EdgeVLModel,
    _Cosmos3EdgeVisionEncoderModel,
)
from mobius.tasks import Cosmos3EdgeVLTask


def _tiny_config(
    *,
    tie_word_embeddings: bool = False,
    spatial_merge_size: int | None = 2,
    out_hidden_size: int | None = 64,
    image_size: int = 28,
    patch_size: int = 14,
) -> ArchitectureConfig:
    return ArchitectureConfig(
        model_type="cosmos3_edge",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="relu2",
        rms_norm_eps=1e-6,
        attn_qk_norm=True,
        tie_word_embeddings=tie_word_embeddings,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=image_size,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            out_hidden_size=out_hidden_size,
            projector_intermediate_size=64,
        ),
    )


def test_text_attention_weights_map_to_mobius_names():
    module = Cosmos3EdgeTextModel(_tiny_config())
    weights = {
        "layers.0.self_attn.to_q.weight": torch.zeros(1),
        "layers.0.self_attn.to_k.weight": torch.zeros(1),
        "layers.0.self_attn.to_v.weight": torch.zeros(1),
        "layers.0.self_attn.to_out.0.weight": torch.zeros(1),
        "layers.0.self_attn.norm_q.weight": torch.zeros(1),
        "layers.0.self_attn.norm_k.weight": torch.zeros(1),
    }

    result = module.preprocess_weights(weights)

    assert "model.layers.0.self_attn.q_proj.weight" in result
    assert "model.layers.0.self_attn.k_proj.weight" in result
    assert "model.layers.0.self_attn.v_proj.weight" in result
    assert "model.layers.0.self_attn.o_proj.weight" in result
    assert "model.layers.0.self_attn.q_norm.weight" in result
    assert "model.layers.0.self_attn.k_norm.weight" in result


def test_vl_tied_embedding_populates_decoder_lm_head():
    module = Cosmos3EdgeVLModel(_tiny_config(tie_word_embeddings=True))
    weight = torch.randn(256, 64)

    result = module.preprocess_weights({"embed_tokens.weight": weight})

    assert result["embedding.embed_tokens.weight"] is weight
    assert result["decoder.lm_head.weight"] is weight


def test_vl_tied_lm_head_populates_embedding():
    module = Cosmos3EdgeVLModel(_tiny_config(tie_word_embeddings=True))
    weight = torch.randn(256, 64)

    result = module.preprocess_weights({"lm_head.weight": weight})

    assert result["decoder.lm_head.weight"] is weight
    assert result["embedding.embed_tokens.weight"] is weight


def test_vision_encoder_defaults_missing_spatial_merge_size():
    module = _Cosmos3EdgeVisionEncoderModel(_tiny_config(spatial_merge_size=None))

    assert module.multi_modal_projector._ms == 2


def test_vision_encoder_rejects_projector_width_mismatch():
    with pytest.raises(AssertionError, match="projector output must match"):
        _Cosmos3EdgeVisionEncoderModel(_tiny_config(out_hidden_size=32))


def test_vl_task_vision_output_matches_embedding_input():
    config = _tiny_config()
    package = Cosmos3EdgeVLTask().build(Cosmos3EdgeVLModel(config), config)

    pixel_values = package["vision_encoder"].graph.inputs[0]
    image_features = package["vision_encoder"].graph.outputs[0]
    embedding_features = next(
        value for value in package["embedding"].graph.inputs if value.name == "image_features"
    )

    assert pixel_values.shape[0] == 1
    assert len(image_features.shape) == 2
    assert len(embedding_features.shape) == 2
    assert image_features.shape[-1] == embedding_features.shape[-1] == config.hidden_size


def test_vision_encoder_rejects_non_integral_patch_grid():
    with pytest.raises(ValueError, match=r"image_size .* divisible by patch_size"):
        _Cosmos3EdgeVisionEncoderModel(_tiny_config(image_size=30, patch_size=14))


def test_vision_encoder_rejects_unmergeable_patch_grid():
    with pytest.raises(ValueError, match=r"grid_size .* divisible"):
        _Cosmos3EdgeVisionEncoderModel(
            _tiny_config(image_size=42, patch_size=14, spatial_merge_size=2)
        )


def test_vision_config_rejects_non_square_num_patches():
    vision_config = SimpleNamespace(num_patches=255, patch_size=16)
    config = SimpleNamespace(vision_config=vision_config)

    with pytest.raises(ValueError, match="num_patches must form a square grid"):
        _cosmos3_edge_vision(config, None, "cosmos3_edge", {"image_size": None})
