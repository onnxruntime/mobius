# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import pytest
import torch

from mobius._component_quantization import (
    configure_component_quantization,
    preprocess_component_quantized_state_dict,
)
from mobius._configs import ArchitectureConfig, QuantizationConfig, VisionConfig
from mobius.models.llava import (
    LLaVAModel,
    _preprocess_llava_component_weights,
    _preprocess_pixtral_weights,
)


def _component_config() -> ArchitectureConfig:
    quantization = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        quantize_embeddings=True,
    )
    return ArchitectureConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        hidden_act="gelu",
        image_token_id=31,
        vision=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=4,
        ),
        quantization=quantization,
        component_quantization={
            "decoder": QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
            ),
            "vision_encoder": quantization,
            "embedding": quantization,
        },
    )


def test_component_quantization_routes_hf_namespaces_and_packed_embedding():
    config = _component_config()
    model = LLaVAModel(config)
    configure_component_quantization(model, config, model.default_task)
    qweight = torch.zeros(32, 8, dtype=torch.uint8)
    routed = model.preprocess_weights(
        {
            "model.language_model.model.layers.0.self_attn.q_proj.weight_qweight": (
                torch.zeros(16, 8, dtype=torch.uint8)
            ),
            "model.language_model.model.layers.0.self_attn.q_proj.weight_scales": (
                torch.ones(16, 1)
            ),
            "model.language_model.model.embed_tokens.weight_qweight": qweight,
            "model.language_model.model.embed_tokens.weight_scales": torch.ones(32, 1),
            "model.vision_tower.encoder.layers.0.mlp.fc1.weight_qweight": torch.zeros(
                32, 8, dtype=torch.uint8
            ),
            "model.vision_tower.encoder.layers.0.mlp.fc1.weight_scales": torch.ones(32, 1),
        }
    )

    assert "decoder.model.layers.0.self_attn.q_proj.weight_qweight" in routed
    assert routed["embedding.embed_tokens.weight_qweight"] is qweight
    assert not any(name.startswith("decoder.model.embed_tokens.") for name in routed)
    assert "vision_encoder.vision_tower.encoder.layers.0.mlp.up_proj.weight_qweight" in routed

    result = preprocess_component_quantized_state_dict(
        routed,
        model,
        config,
        model.default_task,
        ("decoder", "vision_encoder", "embedding"),
    )

    assert result["decoder.model.layers.0.self_attn.q_proj.weight"].shape == (16, 1, 8)
    assert result["vision_encoder.vision_tower.encoder.layers.0.mlp.up_proj.weight"].shape == (
        32,
        1,
        8,
    )
    assert result["embedding.embed_tokens.qweight"] is qweight


def test_pixtral_rejects_packed_tied_embedding_instead_of_corrupting_head():
    with pytest.raises(NotImplementedError, match="explicit LM-head copy"):
        _preprocess_pixtral_weights(
            {
                "language_model.model.embed_tokens.weight_qweight": torch.zeros(
                    32, 8, dtype=torch.uint8
                ),
                "language_model.model.embed_tokens.weight_scales": torch.ones(32, 1),
            },
            tie_word_embeddings=True,
            component_quantization=True,
        )


def test_component_quantization_routes_idefics_namespaces():
    qweight = torch.zeros(32, 8, dtype=torch.uint8)
    result = _preprocess_llava_component_weights(
        {
            "model.text_model.layers.0.self_attn.q_proj.weight_qweight": torch.zeros(
                16, 8, dtype=torch.uint8
            ),
            "model.text_model.embed_tokens.weight_qweight": qweight,
            "model.vision_model.encoder.layers.0.mlp.fc1.weight_qweight": torch.zeros(
                32, 8, dtype=torch.uint8
            ),
            "model.connector.linear_1.weight_qweight": torch.zeros(16, 8, dtype=torch.uint8),
        },
        tie_word_embeddings=False,
    )

    assert "decoder.model.layers.0.self_attn.q_proj.weight_qweight" in result
    assert result["embedding.embed_tokens.weight_qweight"] is qweight
    assert not any(name.startswith("decoder.model.embed_tokens.") for name in result)
    assert (
        "vision_encoder.vision_tower.vision_model.encoder.layers.0.mlp.up_proj.weight_qweight"
        in result
    )
    assert "vision_encoder.multi_modal_projector.linear_1.weight_qweight" in result


def test_quantization_metadata_tie_rejects_unmaterialized_packed_head():
    config = _component_config()
    config.tie_word_embeddings = False
    config.quantization.tie_word_embeddings = True
    model = LLaVAModel(config)

    with pytest.raises(NotImplementedError, match="explicit LM-head copy"):
        model.preprocess_weights(
            {
                "model.language_model.model.embed_tokens.weight_qweight": torch.zeros(
                    32, 8, dtype=torch.uint8
                ),
                "model.language_model.model.embed_tokens.weight_scales": torch.ones(32, 1),
            }
        )
