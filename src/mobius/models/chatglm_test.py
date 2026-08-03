# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch

from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius._weight_utils import preprocess_gptq_weights
from mobius.components import MLP
from mobius.models.chatglm import ChatGLMCausalLMModel


def _gptq_tensor_shapes(k: int, n: int, group_size: int = 16):
    return {
        "qweight": torch.zeros(k // 8, n, dtype=torch.int32),
        "qzeros": torch.zeros(k // group_size, n // 8, dtype=torch.int32),
        "scales": torch.ones(k // group_size, n, dtype=torch.float16),
        "g_idx": torch.arange(k, dtype=torch.int32) // group_size,
    }


def _add_gptq(state_dict: dict[str, torch.Tensor], stem: str, k: int, n: int) -> None:
    for suffix, value in _gptq_tensor_shapes(k, n).items():
        state_dict[f"{stem}.{suffix}"] = value


def test_glm4_gptq_checkpoint_remap_and_fused_projection_split():
    config = ArchitectureConfig(
        model_type="chatglm",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=128,
        rope_type="default",
        partial_rotary_factor=0.5,
        rope_interleave=True,
        attn_qkv_bias=True,
        quantization=QuantizationConfig(bits=4, group_size=16, quant_method="gptq", sym=True),
    )
    model = ChatGLMCausalLMModel(config)
    assert isinstance(model.model.layers[0].mlp, MLP)
    state_dict = {
        "transformer.embedding.word_embeddings.weight": torch.zeros(64, 32),
        "transformer.encoder.final_layernorm.weight": torch.ones(32),
        "transformer.output_layer.weight": torch.zeros(64, 32),
        "transformer.encoder.layers.0.input_layernorm.weight": torch.ones(32),
        "transformer.encoder.layers.0.post_attention_layernorm.weight": torch.ones(32),
        "transformer.encoder.layers.0.self_attention.query_key_value.bias": torch.zeros(64),
        "transformer.encoder.layers.0.mlp.dense_h_to_4h.bias": torch.arange(
            32, dtype=torch.float16
        ),
    }
    _add_gptq(
        state_dict,
        "transformer.encoder.layers.0.self_attention.query_key_value",
        32,
        64,
    )
    _add_gptq(
        state_dict,
        "transformer.encoder.layers.0.self_attention.dense",
        32,
        32,
    )
    _add_gptq(
        state_dict,
        "transformer.encoder.layers.0.mlp.dense_h_to_4h",
        32,
        32,
    )
    _add_gptq(
        state_dict,
        "transformer.encoder.layers.0.mlp.dense_4h_to_h",
        16,
        32,
    )

    result = model.preprocess_weights(state_dict)

    assert result["model.layers.0.self_attn.q_proj.weight"].shape == (32, 2, 8)
    assert result["model.layers.0.self_attn.k_proj.weight"].shape == (16, 2, 8)
    assert result["model.layers.0.self_attn.v_proj.weight"].shape == (16, 2, 8)
    assert result["model.layers.0.self_attn.q_proj.scales"].shape == (32, 2)
    assert result["model.layers.0.self_attn.k_proj.bias"].shape == (16,)
    assert not any(key.endswith(".zero_points") for key in result)
    assert "model.layers.0.self_attn.qkv_proj.weight" not in result
    assert "model.layers.0.mlp.gate_up_proj.weight" not in result
    assert result["model.layers.0.mlp.gate_proj.weight"].shape == (16, 2, 8)
    assert result["model.layers.0.mlp.up_proj.weight"].shape == (16, 2, 8)
    assert torch.equal(
        result["model.layers.0.mlp.gate_proj.bias"],
        torch.arange(16, dtype=torch.float16),
    )
    assert torch.equal(
        result["model.layers.0.mlp.up_proj.bias"],
        torch.arange(16, 32, dtype=torch.float16),
    )
    assert "model.layers.0.mlp.down_proj.weight" in result
    assert "model.embed_tokens.weight" in result
    assert "model.norm.weight" in result
    assert "lm_head.weight" in result


def test_glm4_gptq_gate_up_split_preserves_packed_tensors():
    config = ArchitectureConfig(
        model_type="chatglm",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=128,
        rope_type="default",
        quantization=QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="gptq",
            sym=False,
        ),
    )
    model = ChatGLMCausalLMModel(config)
    source: dict[str, torch.Tensor] = {}
    stem = "transformer.encoder.layers.0.mlp.dense_h_to_4h"
    _add_gptq(source, stem, 32, 32)
    source[f"{stem}.qweight"][:, 16:] = 0x12345678
    source[f"{stem}.scales"][:, 16:] = 2
    qzeros = source[f"{stem}.qzeros"]
    qzeros[:, qzeros.shape[1] // 2 :] = 0x11111111

    expected = preprocess_gptq_weights(
        {
            key.replace(stem, "model.layers.0.mlp.gate_up_proj"): value
            for key, value in source.items()
        },
        bits=4,
        group_size=16,
    )
    result = model.preprocess_weights(source)

    for suffix in ("weight", "scales", "zero_points"):
        fused = expected[f"model.layers.0.mlp.gate_up_proj.{suffix}"]
        assert torch.equal(result[f"model.layers.0.mlp.gate_proj.{suffix}"], fused[:16])
        assert torch.equal(result[f"model.layers.0.mlp.up_proj.{suffix}"], fused[16:])
