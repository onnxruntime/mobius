# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import Qwen4ExpConfig
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.qwen4_exp import Qwen4ExpCausalLMModel


def _config(**overrides) -> Qwen4ExpConfig:
    values = dict(
        model_type="qwen4_exp_text",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        partial_rotary_factor=0.25,
        pad_token_id=0,
        eos_token_id=1,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=2,
        hc_lowrank=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        mtp_num_hidden_layers=0,
    )
    values.update(overrides)
    return Qwen4ExpConfig(**values)


def _build(config: Qwen4ExpConfig | None = None):
    config = config or _config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    return config, module, model


def test_pinned_config_fields_extract_and_normalize_schedule():
    text = SimpleNamespace(
        model_type="qwen4_exp_text",
        vocab_size=248320,
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=4,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        },
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
        eos_token_id=248044,
        mtp_num_hidden_layers=0,
    )
    config = Qwen4ExpConfig.from_transformers(
        SimpleNamespace(
            model_type="qwen4_exp",
            text_config=text,
            tie_word_embeddings=False,
        )
    )

    assert config.model_type == "qwen4_exp_text"
    assert config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "qwen_sparse_attention",
    ]
    assert config.linear_num_value_heads == 48
    assert config.num_local_experts == 512
    assert config.num_experts_per_tok == 10
    assert config.hc_count == 4
    assert config.ple_layer_ids == [2]
    assert config.indexer_budget == 2048


def test_mtp_metadata_is_preserved_but_dedicated_embeddings_fail_closed():
    assert _config(mtp_num_hidden_layers=1).mtp_num_hidden_layers == 1
    with pytest.raises(ValueError, match="dedicated MTP embeddings"):
        _config(mtp_use_dedicated_embeddings=True)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hc_count": 1}, "hc_count > 1"),
        ({"indexer_kv_heads": 2}, "indexer_kv_heads=1"),
        ({"indexer_budget": 3}, "divisible"),
        ({"ple_layer_ids": [2]}, "only supported on linear_attention"),
    ],
)
def test_exact_config_guards(override, message):
    with pytest.raises(ValueError, match=message):
        _config(**override)


def test_registry_routes_composite_and_text_model_types():
    from mobius._registry import _TEXT_ONLY_MODEL_TYPE

    for architecture in (
        "qwen4_exp",
        "qwen4_exp_text",
        "Qwen4ExpForConditionalGeneration",
    ):
        registration = registry.get_registration(architecture)
        assert registration.module_class is Qwen4ExpCausalLMModel
        assert registration.config_class is Qwen4ExpConfig
        assert registration.task == "qwen4-exp-text-generation"
    assert _TEXT_ONLY_MODEL_TYPE["qwen4_exp"] == "qwen4_exp_text"


def test_graph_exposes_exact_heterogeneous_state_abi():
    config, _module, model = _build()
    inputs = {value.name: value for value in model.graph.inputs}
    outputs = {value.name for value in model.graph.outputs}

    assert inputs["past_key_values.0.conv_state"].shape == ir.Shape(
        [ir.SymbolicDim("batch"), 32, config.linear_conv_kernel_dim]
    )
    assert inputs["past_key_values.0.recurrent_state"].shape[-3:] == (
        2,
        8,
        8,
    )
    assert inputs["past_key_values.0.ple_conv_state"].shape[-2:] == (
        config.hc_count * config.hidden_size,
        3,
    )
    assert inputs["past_key_values.0.ple_context"].shape[-1] == 2
    assert inputs["past_key_values.1.index_key"].shape[-1] == 8
    assert {
        "present_position_ids",
        "present.0.conv_state",
        "present.0.recurrent_state",
        "present.0.ple_conv_state",
        "present.0.ple_context",
        "present.1.key",
        "present.1.value",
        "present.1.index_key",
    } <= outputs
    assert model.metadata_props["mobius.cache_abi"].startswith("qwen4-exp:position_ids")


def test_parameter_names_match_upstream_modules():
    _config_value, module, _model = _build()
    names = {name for name, _ in module.named_parameters()}
    assert "model.layers.0.attn_hyper_connection.input_mix_weight_down.weight" in names
    assert "model.layers.0.mlp_hyper_connection.block_inject_weight.weight" in names
    assert "model.layers.0.ple.ple_embedding.ngram_embedding.weight" in names
    assert "model.layers.1.self_attn.indexer.index_qk_proj.weight" in names
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in names
    assert "model.layers.1.mlp.shared_expert_gate.weight" in names


def test_preprocess_unpacks_experts_and_joins_ple_shards():
    config = _config(split_ngram_parts=2)
    module = Qwen4ExpCausalLMModel(config)
    embedding_rows = module.model.layers[0].ple.ple_embedding.ngram_embedding.weight.shape[0]
    state = {
        "mtp.fc_embedding.weight": torch.zeros(16, 16),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.arange(
            2 * 16 * 16, dtype=torch.float32
        ).reshape(2, 16, 16),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.zeros(2, 16, 8),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight": (
            torch.zeros(embedding_rows // 2, 2)
        ),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight": (
            torch.ones(embedding_rows - embedding_rows // 2, 2)
        ),
    }
    result = module.preprocess_weights(state)

    assert result["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (
        8,
        16,
    )
    assert result["model.layers.0.mlp.experts.1.up_proj.weight"].shape == (
        8,
        16,
    )
    assert result["model.layers.0.mlp.experts.1.down_proj.weight"].shape == (
        16,
        8,
    )
    ple = result["model.layers.0.ple.ple_embedding.ngram_embedding.weight"]
    assert ple.shape == (embedding_rows, 2)
    assert torch.all(ple[embedding_rows // 2 :] == 1)
    assert not any(key.startswith("mtp.") for key in result)


def _initial_states() -> dict[str, np.ndarray]:
    return {
        "past_position_ids": np.zeros((1, 0), dtype=np.int64),
        "past_key_values.0.conv_state": np.zeros((1, 32, 2), dtype=np.float32),
        "past_key_values.0.recurrent_state": np.zeros((1, 2, 8, 8), dtype=np.float32),
        "past_key_values.0.ple_conv_state": np.zeros((1, 32, 3), dtype=np.float32),
        # The graph must replace zero-initialized context with eos_token_id
        # when past_position_ids has zero length.
        "past_key_values.0.ple_context": np.zeros((1, 2), dtype=np.int64),
        "past_key_values.1.key": np.zeros((1, 1, 0, 8), dtype=np.float32),
        "past_key_values.1.value": np.zeros((1, 1, 0, 8), dtype=np.float32),
        "past_key_values.1.index_key": np.zeros((1, 0, 8), dtype=np.float32),
    }


def test_random_weight_prefill_matches_token_by_token_decode():
    _config_value, _module, model = _build()
    rng = np.random.default_rng(0)
    for value in model.graph.initializers.values():
        if value.const_value is None:
            value.const_value = ir.tensor(
                rng.normal(0.0, 0.02, [int(dim) for dim in value.shape]).astype(np.float32)
            )

    input_ids = np.array([[2, 3, 4, 5]], dtype=np.int64)
    session = OnnxModelSession(model)
    try:
        full = session.run(
            _initial_states()
            | {
                "input_ids": input_ids,
                "attention_mask": np.ones((1, 4), dtype=np.int64),
                "position_ids": np.arange(4, dtype=np.int64)[None],
            }
        )["logits"]

        states = _initial_states()
        decode_logits = []
        for token_index in range(input_ids.shape[1]):
            outputs = session.run(
                states
                | {
                    "input_ids": input_ids[:, token_index : token_index + 1],
                    "attention_mask": np.ones((1, token_index + 1), dtype=np.int64),
                    "position_ids": np.array([[token_index]], dtype=np.int64),
                }
            )
            decode_logits.append(outputs["logits"])
            states = {
                "past_position_ids": outputs["present_position_ids"],
                "past_key_values.0.conv_state": outputs["present.0.conv_state"],
                "past_key_values.0.recurrent_state": outputs["present.0.recurrent_state"],
                "past_key_values.0.ple_conv_state": outputs["present.0.ple_conv_state"],
                "past_key_values.0.ple_context": outputs["present.0.ple_context"],
                "past_key_values.1.key": outputs["present.1.key"],
                "past_key_values.1.value": outputs["present.1.value"],
                "past_key_values.1.index_key": outputs["present.1.index_key"],
            }
    finally:
        session.close()

    np.testing.assert_allclose(
        np.concatenate(decode_logits, axis=1),
        full,
        rtol=1e-5,
        atol=1e-6,
    )


def test_left_padding_matches_unpadded_prefill_and_following_decode():
    _config_value, _module, model = _build()
    rng = np.random.default_rng(1)
    for value in model.graph.initializers.values():
        if value.const_value is None:
            value.const_value = ir.tensor(
                rng.normal(0.0, 0.02, [int(dim) for dim in value.shape]).astype(np.float32)
            )

    session = OnnxModelSession(model)
    try:
        unpadded = session.run(
            _initial_states()
            | {
                "input_ids": np.array([[2, 3, 4]], dtype=np.int64),
                "attention_mask": np.ones((1, 3), dtype=np.int64),
                "position_ids": np.array([[0, 1, 2]], dtype=np.int64),
            }
        )
        padded = session.run(
            _initial_states()
            | {
                "input_ids": np.array([[0, 2, 3, 4]], dtype=np.int64),
                "attention_mask": np.array([[0, 1, 1, 1]], dtype=np.int64),
                "position_ids": np.array([[0, 0, 1, 2]], dtype=np.int64),
            }
        )

        def decode(outputs):
            return session.run(
                {
                    "input_ids": np.array([[5]], dtype=np.int64),
                    "attention_mask": np.ones((1, 4), dtype=np.int64),
                    "position_ids": np.array([[3]], dtype=np.int64),
                    "past_position_ids": outputs["present_position_ids"][:, -3:],
                    "past_key_values.0.conv_state": outputs["present.0.conv_state"],
                    "past_key_values.0.recurrent_state": outputs["present.0.recurrent_state"],
                    "past_key_values.0.ple_conv_state": outputs["present.0.ple_conv_state"],
                    "past_key_values.0.ple_context": outputs["present.0.ple_context"],
                    "past_key_values.1.key": outputs["present.1.key"][:, :, -3:],
                    "past_key_values.1.value": outputs["present.1.value"][:, :, -3:],
                    "past_key_values.1.index_key": outputs["present.1.index_key"][:, -3:],
                }
            )

        unpadded_decode = decode(unpadded)
        padded_decode = decode(padded)
    finally:
        session.close()

    np.testing.assert_allclose(
        padded["logits"][:, -3:],
        unpadded["logits"],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        padded_decode["logits"],
        unpadded_decode["logits"],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.integration
def test_reduced_random_weight_huggingface_prefill_and_decode_parity():
    """Compare the complete tiny core when the pinned Qwen4-Exp HF class is installed."""
    try:
        from transformers import (
            DynamicCache,
            Qwen4ExpForCausalLM,
            Qwen4ExpTextConfig,
        )
    except ImportError:
        pytest.skip(
            "Installed transformers does not contain the pinned experimental "
            "Qwen4-Exp implementation"
        )

    from mobius.integrations._weight_loading import apply_weights

    torch.manual_seed(0)
    hf_config = Qwen4ExpTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.25,
        },
        layer_types=["linear_attention", "qwen_sparse_attention"],
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=2,
        hc_lowrank=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        eos_token_id=1,
        pad_token_id=0,
        mtp_num_hidden_layers=0,
    )
    hf_model = Qwen4ExpForCausalLM(hf_config).eval()
    config = Qwen4ExpConfig.from_transformers(hf_config)
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    apply_weights(model, module.preprocess_weights(dict(hf_model.state_dict())))

    input_ids = np.array([[2, 3, 4, 5]], dtype=np.int64)
    with torch.no_grad():
        hf_full = hf_model(torch.from_numpy(input_ids), use_cache=False).logits.numpy()
        hf_cache = DynamicCache(config=hf_config)
        hf_decode = np.concatenate(
            [
                hf_model(
                    torch.from_numpy(input_ids[:, index : index + 1]),
                    past_key_values=hf_cache,
                    use_cache=True,
                ).logits.numpy()
                for index in range(input_ids.shape[1])
            ],
            axis=1,
        )

    session = OnnxModelSession(model)
    try:
        onnx_full = session.run(
            _initial_states()
            | {
                "input_ids": input_ids,
                "attention_mask": np.ones((1, 4), dtype=np.int64),
                "position_ids": np.arange(4, dtype=np.int64)[None],
            }
        )["logits"]
    finally:
        session.close()

    np.testing.assert_allclose(onnx_full, hf_full, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(onnx_full, hf_decode, rtol=1e-3, atol=1e-3)
