# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import pytest

from mobius._testing import make_config
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.plm import PLMCausalLMModel
from mobius.rewrite_rules._testing_utils import fill_random_weights
from mobius.tasks import CausalLMTask


def _config(**overrides):
    values = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 24,
        "q_lora_rank": None,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 12,
        "hidden_act": "relu2",
        "tie_word_embeddings": True,
        "rope_interleave": True,
        "attn_qkv_bias": False,
        "mlp_bias": False,
    }
    values.update(overrides)
    return make_config(**values)


def test_plm_parameter_and_tied_output_contract() -> None:
    module = PLMCausalLMModel(_config())
    layer = module.model.layers[0]

    assert list(layer.self_attn.q_proj.weight.shape) == [4 * 24, 64]
    assert list(layer.self_attn.kv_a_proj_with_mqa.weight.shape) == [16 + 8, 64]
    assert list(layer.self_attn.kv_a_layernorm.weight.shape) == [16]
    assert list(layer.self_attn.kv_b_proj.weight.shape) == [4 * (16 + 12), 16]
    assert list(layer.self_attn.o_proj.weight.shape) == [64, 4 * 12]
    assert not hasattr(layer.mlp, "gate_proj")
    assert module.lm_head.weight is module.model.embed_tokens.weight


def test_plm_graph_uses_expanded_asymmetric_cache() -> None:
    config = _config()
    graph = CausalLMTask().build(PLMCausalLMModel(config), config)["model"].graph
    inputs = {value.name: value for value in graph.inputs}
    outputs = {value.name: value for value in graph.outputs}

    assert str(inputs["past_key_values.0.key"].shape) == "[batch,4,past_sequence_len,24]"
    assert str(inputs["past_key_values.0.value"].shape) == "[batch,4,past_sequence_len,12]"
    assert str(outputs["present.0.key"].shape) == (
        "[batch,4,past_sequence_len + sequence_len,24]"
    )
    assert str(outputs["present.0.value"].shape) == (
        "[batch,4,past_sequence_len + sequence_len,12]"
    )
    assert "lm_head.weight" not in graph.initializers

    op_types = [node.op_type for node in graph]
    assert "Attention" in op_types
    assert "Relu" in op_types
    assert "Mul" in op_types


def test_plm_prefill_decode_cache_matches_full_prefill() -> None:
    """Expanded-cache decode must equal the corresponding full-prefill token."""
    np.random.seed(0)
    config = _config(vocab_size=32)
    model = CausalLMTask().build(PLMCausalLMModel(config), config)["model"]
    fill_random_weights(model)
    session = OnnxModelSession(model, device="cpu")
    try:

        def _feeds(tokens, position, past_key, past_value):
            return {
                "input_ids": np.asarray([tokens], dtype=np.int64),
                "attention_mask": np.ones(
                    (1, past_key.shape[2] + len(tokens)), dtype=np.int64
                ),
                "position_ids": np.asarray([position], dtype=np.int64),
                "past_key_values.0.key": past_key,
                "past_key_values.0.value": past_value,
            }

        empty_key = np.zeros((1, 4, 0, 24), dtype=np.float32)
        empty_value = np.zeros((1, 4, 0, 12), dtype=np.float32)
        full = session.run(_feeds([1, 2, 3], [0, 1, 2], empty_key, empty_value))
        prefix = session.run(_feeds([1, 2], [0, 1], empty_key, empty_value))
        decode = session.run(
            _feeds(
                [3],
                [2],
                prefix["present.0.key"],
                prefix["present.0.value"],
            )
        )
    finally:
        session.close()

    np.testing.assert_allclose(decode["logits"][:, 0], full["logits"][:, -1], atol=1e-4)
    np.testing.assert_allclose(decode["present.0.key"], full["present.0.key"], atol=1e-5)
    np.testing.assert_allclose(decode["present.0.value"], full["present.0.value"], atol=1e-5)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"num_key_value_heads": 1}, "expanded K/V"),
        ({"qk_rope_head_dim": 7, "head_dim": 23}, "must be even"),
        ({"hidden_act": "silu"}, "relu2"),
        ({"q_lora_rank": 8}, "direct q_proj"),
        ({"tie_word_embeddings": False}, "tied"),
    ],
)
def test_plm_rejects_inexact_variants(override, match) -> None:
    with pytest.raises(ValueError, match=match):
        PLMCausalLMModel(_config(**override))
