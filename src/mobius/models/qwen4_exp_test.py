# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import Qwen4ExpConfig, VisionConfig
from mobius._registry import registry
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.qwen4_exp import (
    Qwen4ExpCausalLMModel,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpQSAIndexer,
)


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
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
        dtype="bfloat16",
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
        mamba_ssm_dtype="float32",
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
    assert config.dtype == ir.DataType.BFLOAT16
    assert config.mrope_interleaved
    assert config.mrope_section == [11, 11, 10]
    assert config.mamba_ssm_dtype == ir.DataType.FLOAT

    multimodal = Qwen4ExpConfig.from_transformers(
        SimpleNamespace(
            model_type="qwen4_exp",
            architectures=["Qwen4ExpForConditionalGeneration"],
            language_model_only=False,
            text_config=text,
            vision_config=SimpleNamespace(
                model_type="qwen4_exp",
                hidden_size=1152,
                intermediate_size=4304,
                num_hidden_layers=27,
                num_attention_heads=16,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                num_position_embeddings=2304,
                out_hidden_size=2560,
                hidden_act="gelu_pytorch_tanh",
                deepstack_visual_indexes=[],
            ),
            image_token_id=248056,
            video_token_id=248057,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
            tie_word_embeddings=False,
        )
    )
    assert multimodal.model_type == "qwen4_exp"
    assert multimodal.vision is not None
    assert multimodal.vision.out_hidden_size == 2560
    assert multimodal.image_token_id == 248056
    assert multimodal.video_token_id == 248057
    assert multimodal.vision_start_token_id == 248053
    assert multimodal.vision_end_token_id == 248054
    assert multimodal.deepstack_visual_indexes == []


def test_mtp_modes_fail_closed():
    with pytest.raises(ValueError, match="MTP is unsupported"):
        _config(mtp_num_hidden_layers=1)
    with pytest.raises(ValueError, match="dedicated MTP embeddings"):
        _config(mtp_use_dedicated_embeddings=True)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hc_count": 1}, "hc_count > 1"),
        ({"indexer_kv_heads": 2}, "indexer_kv_heads=1"),
        ({"indexer_budget": 3}, "divisible"),
        ({"ple_layer_ids": [2]}, "only supported on linear_attention"),
        (
            {"linear_num_key_heads": 2, "linear_num_value_heads": 3},
            "must be divisible",
        ),
        ({"linear_conv_kernel_dim": 0}, "must be positive"),
        ({"make_ngram_vocab_size_divisible_by": 0}, "must be > 0"),
        ({"mamba_ssm_dtype": ir.DataType.BFLOAT16}, "mamba_ssm_dtype=float32"),
    ],
)
def test_exact_config_guards(override, message):
    with pytest.raises(ValueError, match=message):
        _config(**override)


@pytest.mark.parametrize("interleaved", [False, True])
def test_qsa_rope_uses_full_rotary_width_from_half_width_frequency_cache(interleaved):
    config = _config(partial_rotary_factor=0.5)
    # The pinned model is half-split, but the reusable rotation path must
    # preserve ONNX RotaryEmbedding semantics for either layout.
    config.rope_interleave = interleaved
    indexer = Qwen4ExpQSAIndexer(config)
    builder, op, graph = create_test_builder()
    value = create_test_input(builder, "value", [1, 3, 2, 8])
    cos = create_test_input(builder, "cos", [1, 3, 2])
    sin = create_test_input(builder, "sin", [1, 3, 2])
    output = indexer._rotate(op, value, (cos, sin), num_heads=2)
    output.name = "output"
    graph.outputs.append(output)
    model = ir.Model(graph, ir_version=11)
    rotary_node = next(node for node in graph if node.op_type == "RotaryEmbedding")
    assert rotary_node.attributes["rotary_embedding_dim"].value == 4
    assert rotary_node.attributes["interleaved"].value == interleaved

    rng = np.random.default_rng(5)
    value_data = rng.normal(size=(1, 3, 2, 8)).astype(np.float32)
    cos_data = rng.normal(size=(1, 3, 2)).astype(np.float32)
    sin_data = rng.normal(size=(1, 3, 2)).astype(np.float32)
    session = OnnxModelSession(model)
    try:
        actual = session.run({"value": value_data, "cos": cos_data, "sin": sin_data})["output"]
    finally:
        session.close()

    rotary = value_data[..., :4]
    if interleaved:
        rotated_half = np.empty_like(rotary)
        rotated_half[..., 0::2] = -rotary[..., 1::2]
        rotated_half[..., 1::2] = rotary[..., 0::2]
        expanded_cos = np.repeat(cos_data[:, :, None, :], 2, axis=-1)
        expanded_sin = np.repeat(sin_data[:, :, None, :], 2, axis=-1)
    else:
        first, second = np.split(rotary, 2, axis=-1)
        rotated_half = np.concatenate((-second, first), axis=-1)
        expanded_cos = np.concatenate((cos_data, cos_data), axis=-1)[:, :, None, :]
        expanded_sin = np.concatenate((sin_data, sin_data), axis=-1)[:, :, None, :]
    expected_rotary = rotary * expanded_cos + rotated_half * expanded_sin
    expected = np.concatenate((expected_rotary, value_data[..., 4:]), axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_registry_routes_composite_and_text_model_types():
    from mobius._registry import _TEXT_ONLY_MODEL_TYPE

    for architecture in ("qwen4_exp", "Qwen4ExpForConditionalGeneration"):
        registration = registry.get_registration(architecture)
        assert registration.module_class is Qwen4ExpForConditionalGeneration
        assert registration.config_class is Qwen4ExpConfig
        assert registration.task == "qwen4-exp-vision-language"
    registration = registry.get_registration("qwen4_exp_text")
    assert registration.module_class is Qwen4ExpCausalLMModel
    assert registration.config_class is Qwen4ExpConfig
    assert registration.task == "qwen4-exp-text-generation"
    assert _TEXT_ONLY_MODEL_TYPE["qwen4_exp"] == "qwen4_exp_text"


def test_multimodal_package_exposes_exact_three_model_io():
    config = _config(
        model_type="qwen4_exp",
        vision=VisionConfig(
            model_type="qwen4_exp_vision",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=16,
            hidden_act="gelu_pytorch_tanh",
            deepstack_visual_indexes=[],
        ),
        image_token_id=30,
        video_token_id=31,
        vision_start_token_id=29,
        vision_end_token_id=28,
        mrope_section=[1, 1, 0],
        mrope_interleaved=True,
        deepstack_visual_indexes=[],
    )
    module = Qwen4ExpForConditionalGeneration(config)
    package = build_from_module(
        module,
        config,
        task="qwen4-exp-vision-language",
    )

    assert set(package) == {"decoder", "vision_encoder", "embedding"}
    decoder_inputs = {value.name: value for value in package["decoder"].graph.inputs}
    decoder_outputs = {value.name for value in package["decoder"].graph.outputs}
    assert {"inputs_embeds", "ple_input_ids", "past_position_ids"} <= decoder_inputs.keys()
    assert "input_ids" not in decoder_inputs
    assert decoder_inputs["position_ids"].shape[0] == 4
    assert decoder_inputs["past_position_ids"].shape[0] == 4
    assert {
        "present_position_ids",
        "present.0.conv_state",
        "present.0.recurrent_state",
        "present.0.ple_conv_state",
        "present.0.ple_context",
        "present.1.key",
        "present.1.value",
        "present.1.index_key",
    } <= decoder_outputs
    assert {value.name for value in package["vision_encoder"].graph.inputs} == {
        "pixel_values",
        "image_grid_thw",
    }
    assert {value.name for value in package["embedding"].graph.inputs} == {
        "input_ids",
        "image_features",
        "video_features",
    }


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
    assert inputs["past_key_values.0.recurrent_state"].dtype == ir.DataType.FLOAT
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


def test_bfloat16_graph_keeps_only_recurrent_math_and_state_in_float32():
    config, _module, model = _build(_config(dtype=ir.DataType.BFLOAT16))
    inputs = {value.name: value for value in model.graph.inputs}
    assert inputs["past_key_values.0.conv_state"].dtype == ir.DataType.BFLOAT16
    assert inputs["past_key_values.0.recurrent_state"].dtype == ir.DataType.FLOAT
    assert inputs["past_key_values.1.key"].dtype == ir.DataType.BFLOAT16

    linear_attention = next(node for node in model.graph if node.op_type == "LinearAttention")
    assert all(value.dtype == ir.DataType.FLOAT for value in linear_attention.inputs)
    assert linear_attention.outputs[1].dtype == config.mamba_ssm_dtype


def test_bfloat16_router_projects_before_float32_softmax():
    _config_value, _module, model = _build(_config(dtype=ir.DataType.BFLOAT16))
    router_matmul = next(
        node
        for node in model.graph
        if node.op_type == "MatMul"
        and node.outputs[0].name is not None
        and ".mlp.gate." in node.outputs[0].name
    )
    assert [value.dtype for value in router_matmul.inputs] == [
        ir.DataType.BFLOAT16,
        ir.DataType.BFLOAT16,
    ]
    router_cast = router_matmul.outputs[0].consumers()[0]
    assert router_cast.op_type == "Cast"
    assert router_cast.outputs[0].dtype == ir.DataType.FLOAT
    assert router_cast.outputs[0].consumers()[0].op_type == "Softmax"


def test_parameter_names_match_upstream_modules():
    _config_value, module, _model = _build()
    names = {name for name, _ in module.named_parameters()}
    assert "model.layers.0.attn_hyper_connection.input_mix_weight_down.weight" in names
    assert "model.layers.0.mlp_hyper_connection.block_inject_weight.weight" in names
    assert "model.layers.0.ple.ple_embedding.ngram_embedding.weight" in names
    assert "model.layers.1.self_attn.indexer.index_qk_proj.weight" in names
    assert "model.layers.1.mlp.experts.gate_up_proj" in names
    assert "model.layers.1.mlp.experts.down_proj" in names
    assert "model.layers.1.mlp.shared_expert_gate.weight" in names


def test_moe_executes_only_packed_topk_experts():
    _config_value, _module, model = _build()
    expert_nodes = [
        node
        for node in model.graph
        if node.outputs[0].name is not None and ".mlp.experts." in node.outputs[0].name
    ]
    assert sum(node.op_type == "Gather" for node in expert_nodes) == 4
    assert sum(node.op_type == "MatMul" for node in expert_nodes) == 4
    assert not any(".mlp.experts.0." in name for name in model.graph.initializers)


def test_preprocess_validates_packed_experts_and_joins_ple_shards():
    config = _config(split_ngram_parts=2)
    module = Qwen4ExpCausalLMModel(config)
    embedding_rows = module.model.layers[0].ple.ple_embedding.ngram_embedding.weight.shape[0]
    state = {
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

    assert result["model.layers.0.mlp.experts.gate_up_proj"].shape == (2, 16, 16)
    assert result["model.layers.0.mlp.experts.down_proj"].shape == (2, 16, 8)
    ple = result["model.layers.0.ple.ple_embedding.ngram_embedding.weight"]
    assert ple.shape == (embedding_rows, 2)
    assert torch.all(ple[embedding_rows // 2 :] == 1)
    assert not any(key.startswith("mtp.") for key in result)


def test_preprocess_fuses_exact_gguf_indexer_query_and_key_rows():
    config = _config()
    module = Qwen4ExpCausalLMModel(config)
    prefix = "model.layers.1.self_attn.indexer"
    query = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16)
    key = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16) + 1000

    result = module.preprocess_weights(
        {
            f"{prefix}.index_q_proj.weight": query,
            f"{prefix}.index_k_proj.weight": key,
        }
    )

    fused = result[f"{prefix}.index_qk_proj.weight"]
    assert fused.shape == (24, 16)
    torch.testing.assert_close(fused[:16], query)
    torch.testing.assert_close(fused[16:], key)


def test_preprocess_preserves_official_fused_hf_indexer_projection():
    module = Qwen4ExpCausalLMModel(_config())
    name = "model.layers.1.self_attn.indexer.index_qk_proj.weight"
    fused = torch.arange(24 * 16, dtype=torch.float32).reshape(24, 16)

    result = module.preprocess_weights({name: fused})

    assert result[name] is fused


def test_preprocess_fails_closed_on_incomplete_or_malformed_gguf_indexer_split():
    config = _config()
    module = Qwen4ExpCausalLMModel(config)
    prefix = "model.layers.1.self_attn.indexer"

    with pytest.raises(ValueError, match=r"missing parts \['k'\]"):
        module.preprocess_weights({f"{prefix}.index_q_proj.weight": torch.zeros(16, 16)})
    with pytest.raises(ValueError, match=r"query projection.*expected"):
        module.preprocess_weights(
            {
                f"{prefix}.index_q_proj.weight": torch.zeros(15, 16),
                f"{prefix}.index_k_proj.weight": torch.zeros(8, 16),
            }
        )


def test_preprocess_fails_closed_on_noncanonical_packed_or_deterministic_weights():
    config = _config()
    module = Qwen4ExpCausalLMModel(config)
    parameters = dict(module.named_parameters())
    buffer_key = "model.layers.0.ple.ple_embedding.layer_multipliers"
    canonical = torch.from_numpy(parameters[buffer_key]._const_value.numpy().copy())

    with pytest.raises(ValueError, match="does not match the pinned hash construction"):
        module.preprocess_weights({buffer_key: canonical + 2})
    with pytest.raises(ValueError, match="packed gate_up_proj has shape"):
        module.preprocess_weights(
            {"model.layers.0.mlp.experts.gate_up_proj": torch.zeros(2, 15, 16)}
        )


def test_preprocess_fails_closed_on_missing_or_unexpected_ple_shards():
    config = _config(split_ngram_parts=2)
    module = Qwen4ExpCausalLMModel(config)
    target = "model.layers.0.ple.ple_embedding.ngram_embedding"
    with pytest.raises(ValueError, match=r"missing shard indices \[1\]"):
        module.preprocess_weights({f"{target}.shard_0.weight": torch.zeros(1, 2)})
    with pytest.raises(ValueError, match=r"unexpected shard indices \[2\]"):
        module.preprocess_weights(
            {
                f"{target}.shard_0.weight": torch.zeros(1, 2),
                f"{target}.shard_1.weight": torch.zeros(1, 2),
                f"{target}.shard_2.weight": torch.zeros(1, 2),
            }
        )


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
        masked_input_ids = np.array([[2, 7, 3, 4]], dtype=np.int64)
        masked_attention = np.array([[1, 0, 1, 1]], dtype=np.int64)
        masked_position_ids = np.array([[0, 0, 1, 2]], dtype=np.int64)
        hf_masked = hf_model(
            torch.from_numpy(masked_input_ids),
            attention_mask=torch.from_numpy(masked_attention),
            position_ids=torch.from_numpy(masked_position_ids),
            use_cache=False,
        ).logits.numpy()

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
        onnx_masked = session.run(
            _initial_states()
            | {
                "input_ids": masked_input_ids,
                "attention_mask": masked_attention,
                "position_ids": masked_position_ids,
            }
        )["logits"]
    finally:
        session.close()

    np.testing.assert_allclose(onnx_full, hf_full, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(onnx_full, hf_decode, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(onnx_masked, hf_masked, rtol=1e-3, atol=1e-3)
