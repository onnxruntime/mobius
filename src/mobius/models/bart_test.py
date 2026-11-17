# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import pytest
import torch

from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius.models.bart import BartForConditionalGeneration


def _config(
    encoder: QuantizationConfig,
    decoder: QuantizationConfig,
    *,
    tie_word_embeddings: bool = False,
) -> ArchitectureConfig:
    return ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        num_hidden_layers=1,
        num_decoder_layers=1,
        vocab_size=256,
        max_position_embeddings=32,
        hidden_act="gelu",
        tie_word_embeddings=tie_word_embeddings,
        component_quantization={
            "encoder": encoder,
            "decoder": decoder,
        },
    )


def test_bart_preserves_and_fans_out_packed_shared_and_head_sidecars():
    quantization = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        quantize_embeddings=True,
        quantize_lm_head=True,
    )
    qweight = torch.zeros(256, 32, dtype=torch.uint8)
    scales = torch.ones(256, 4)

    result = BartForConditionalGeneration(
        _config(quantization, quantization)
    ).preprocess_weights(
        {
            "model.shared.weight_qweight": qweight,
            "model.shared.weight_scales": scales,
            "lm_head.weight_qweight": qweight,
            "lm_head.weight_scales": scales,
        }
    )

    assert result["encoder.embed_tokens.weight_qweight"] is qweight
    assert result["encoder.embed_tokens.weight_scales"] is scales
    assert result["decoder.embed_tokens.weight_qweight"] is qweight
    assert result["decoder.embed_tokens.weight_scales"] is scales
    assert result["decoder.lm_head.weight_qweight"] is qweight
    assert result["decoder.lm_head.weight_scales"] is scales
    assert not any(name.startswith("shared.") for name in result)


def test_bart_materializes_missing_packed_tied_head_for_split_decoder():
    quantization = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        quantize_embeddings=True,
        quantize_lm_head=True,
        tie_word_embeddings=True,
    )
    qweight = torch.zeros(256, 32, dtype=torch.uint8)
    scales = torch.ones(256, 4)

    result = BartForConditionalGeneration(
        _config(quantization, quantization, tie_word_embeddings=True)
    ).preprocess_weights(
        {
            "model.shared.weight_qweight": qweight,
            "model.shared.weight_scales": scales,
        }
    )

    assert result["decoder.lm_head.weight_qweight"] is qweight
    assert result["decoder.lm_head.weight_scales"] is scales


def test_bart_rejects_shared_packed_embedding_with_different_component_layouts():
    encoder = QuantizationConfig(
        bits=8,
        group_size=32,
        quant_method="olive",
        quantize_embeddings=True,
    )
    decoder = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        quantize_embeddings=True,
    )

    with pytest.raises(ValueError, match="same embedding quantization layout"):
        BartForConditionalGeneration(_config(encoder, decoder)).preprocess_weights(
            {
                "shared.weight_qweight": torch.zeros(256, 32, dtype=torch.uint8),
                "shared.weight_scales": torch.ones(256, 4),
            }
        )
