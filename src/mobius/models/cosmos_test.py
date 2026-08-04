# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for Cosmos3-Edge weight routing and projector configuration."""

from __future__ import annotations

import pytest
import torch

from mobius._configs import ArchitectureConfig, VisionConfig
from mobius.models.cosmos import (
    Cosmos3EdgeTextModel,
    Cosmos3EdgeVLModel,
    _Cosmos3EdgeVisionEncoderModel,
)


def _tiny_config(
    *,
    tie_word_embeddings: bool = False,
    spatial_merge_size: int | None = 2,
    out_hidden_size: int | None = 64,
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
            image_size=28,
            patch_size=14,
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
