# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius.components import QuantizedEmbedding, QuantizedLinear
from mobius.models.gemma3_text import Gemma3CausalLMModel


def test_gemma3_uses_quantized_components_when_configured():
    config = ArchitectureConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        rope_type="default",
        rope_local_base_freq=10_000.0,
        attn_qk_norm=True,
        hidden_act="gelu_pytorch_tanh",
        quantization=QuantizationConfig(
            bits=4,
            group_size=32,
            quant_method="gguf",
            sym=True,
            quantize_embeddings=True,
            quantize_lm_head=True,
        ),
    )

    model = Gemma3CausalLMModel(config)

    assert isinstance(model.model.embed_tokens, QuantizedEmbedding)
    assert isinstance(model.model.layers[0].self_attn.q_proj, QuantizedLinear)
    assert isinstance(model.model.layers[0].mlp.gate_proj, QuantizedLinear)
    assert isinstance(model.lm_head, QuantizedLinear)


def test_gemma3_reties_lm_head_after_replacing_base_text_model():
    config = ArchitectureConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        rope_type="default",
        rope_local_base_freq=10_000.0,
        attn_qk_norm=True,
        hidden_act="gelu_pytorch_tanh",
        tie_word_embeddings=True,
    )

    model = Gemma3CausalLMModel(config)

    assert model.lm_head.weight is model.model.embed_tokens.weight
