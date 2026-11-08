# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Qwen-VL weight preprocessing and decoder pruning parity."""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._component_quantization import (
    configure_component_quantization,
    preprocess_component_quantized_state_dict,
)
from mobius._configs import ArchitectureConfig, QuantizationConfig, VisionConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.qwen_vl import (
    Qwen3VL3ModelCausalLMModel,
    Qwen3VLDecoderModel,
    Qwen25VLCausalLMModel,
    Qwen25VLDecoderModel,
)
from mobius.rewrite_rules._testing_utils import fill_random_weights
from mobius.tasks import QwenVLTask

# Tiny config for weight preprocessing tests (no graph build needed)
_BASE_CONFIG = ArchitectureConfig(
    model_type="qwen2_5_vl",
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=100,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    hidden_act="silu",
    vision=VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
    ),
)


@pytest.mark.parametrize("runtime_indices", [False, True])
@pytest.mark.parametrize("deepstack", [False, True])
@pytest.mark.parametrize("task_object", [False, True])
def test_qwen3_vl_pruned_decoder_shared_weight_parity(runtime_indices, deepstack, task_object):
    config = dataclasses.replace(
        _BASE_CONFIG,
        model_type="qwen3_vl",
        dtype=ir.DataType.FLOAT,
        tie_word_embeddings=False,
        attn_qk_norm=True,
        mrope_section=[2, 3, 3],
        max_position_embeddings=16,
        deepstack_visual_indexes=[0, 1] if deepstack else None,
        vision=dataclasses.replace(
            _BASE_CONFIG.vision,
            out_hidden_size=_BASE_CONFIG.hidden_size,
            num_position_embeddings=4,
        ),
    )
    module = Qwen3VL3ModelCausalLMModel(config)
    full = build_from_module(module, config, task="qwen-vl", execution_provider="cpu")[
        "decoder"
    ]
    pruned = build_from_module(
        Qwen3VL3ModelCausalLMModel(config),
        config,
        task=(
            QwenVLTask(
                prune_prefill_prefix=True,
                runtime_last_token_indices=runtime_indices,
            )
            if task_object
            else "qwen-vl"
        ),
        execution_provider="cpu",
        prune_prefill_prefix=True,
        runtime_last_token_indices=runtime_indices and not task_object,
    )["decoder"]
    indices_input = next(
        (value for value in pruned.graph.inputs if value.name == "last_token_indices"),
        None,
    )
    assert (indices_input is not None) is runtime_indices
    if indices_input is not None:
        assert indices_input.dtype == ir.DataType.INT32
        assert indices_input.shape is not None
        assert len(indices_input.shape) == 3
        assert indices_input.shape[1] == indices_input.shape[2] == 1
    full_weights = {
        name: value
        for name, value in full.graph.initializers.items()
        if value.const_value is None
    }
    pruned_weights = {
        name: value
        for name, value in pruned.graph.initializers.items()
        if value.const_value is None
    }
    assert full_weights and full_weights.keys() == pruned_weights.keys()
    fill_random_weights(full)
    for name, initializer in pruned_weights.items():
        initializer.const_value = full_weights[name].const_value
    full_session = OnnxModelSession(full, device="cpu", intra_op_num_threads=1)
    pruned_session = OnnxModelSession(pruned, device="cpu", intra_op_num_threads=1)

    batch, sequence_length = 2, 5
    rng = np.random.default_rng(0)
    attention_mask = np.asarray([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=np.int64)
    positions = np.asarray([[0, 1, 0, 0, 0], [0, 1, 2, 3, 0]], dtype=np.int64)
    prefill = {
        "inputs_embeds": rng.standard_normal(
            (batch, sequence_length, config.hidden_size), dtype=np.float32
        ),
        "attention_mask": attention_mask,
        "position_ids": np.repeat(positions[None, ...], 3, axis=0),
    }
    cache_inputs = [
        name for name in full_session.input_names if name.startswith("past_key_values.")
    ]
    assert len(cache_inputs) == 2 * config.num_hidden_layers
    cache_shape = (batch, config.num_key_value_heads, 0, config.head_dim)
    prefill.update({name: np.zeros(cache_shape, dtype=np.float32) for name in cache_inputs})
    if deepstack:
        prefill["per_layer_inputs"] = rng.standard_normal(
            (batch, sequence_length, 2 * config.hidden_size), dtype=np.float32
        ) * attention_mask[..., None].astype(np.float32)
    selected_rows = np.asarray([1, 3] if runtime_indices else [-1, -1])
    pruned_prefill = dict(prefill)
    if runtime_indices:
        pruned_prefill["last_token_indices"] = selected_rows.astype(np.int32).reshape(
            batch, 1, 1
        )
    full_prefill = full_session.run(prefill)
    actual_prefill = pruned_session.run(pruned_prefill)
    assert full_prefill["logits"].shape == (batch, sequence_length, config.vocab_size)
    assert actual_prefill["logits"].shape == (batch, 1, config.vocab_size)
    np.testing.assert_allclose(
        actual_prefill["logits"],
        full_prefill["logits"][np.arange(batch), selected_rows][:, None, :],
        rtol=1e-4,
        atol=1e-4,
    )

    decode = {
        "inputs_embeds": rng.standard_normal((batch, 1, config.hidden_size), dtype=np.float32),
        "attention_mask": np.concatenate(
            [attention_mask, np.ones((batch, 1), dtype=np.int64)], axis=1
        ),
        "position_ids": np.repeat(attention_mask.sum(axis=1).reshape(1, batch, 1), 3, axis=0),
    }
    if deepstack:
        decode["per_layer_inputs"] = np.zeros(
            (batch, 1, 2 * config.hidden_size), dtype=np.float32
        )
    pruned_decode = dict(decode)
    cache_outputs = [output.name for output in full.graph.outputs if output.name != "logits"]
    assert len(cache_outputs) == len(cache_inputs)
    for past_name, present_name in zip(cache_inputs, cache_outputs, strict=True):
        decode[past_name] = full_prefill[present_name]
        pruned_decode[past_name] = actual_prefill[present_name]
    if runtime_indices:
        pruned_decode["last_token_indices"] = np.zeros((batch, 1, 1), dtype=np.int32)
    full_decode = full_session.run(decode)
    actual_decode = pruned_session.run(pruned_decode)
    assert actual_decode["logits"].shape == (batch, 1, config.vocab_size)
    np.testing.assert_allclose(
        actual_decode["logits"], full_decode["logits"], rtol=1e-4, atol=1e-4
    )
    for expected, actual, length in [
        (full_prefill, actual_prefill, sequence_length),
        (full_decode, actual_decode, sequence_length + 1),
    ]:
        for name in cache_outputs:
            assert (
                expected[name].shape
                == actual[name].shape
                == (
                    batch,
                    config.num_key_value_heads,
                    length,
                    config.head_dim,
                )
            )
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-4, atol=1e-4)


def _fake_state_dict_qwen25vl() -> dict[str, torch.Tensor]:
    """State dict mimicking HF Qwen2.5-VL with tie_word_embeddings=True.

    HF keys: model.embed_tokens.weight, model.layers.*, lm_head.weight (absent).
    """
    embed = torch.randn(100, 64)
    return {
        "model.embed_tokens.weight": embed,
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        "model.norm.weight": torch.randn(64),
    }


def _fake_state_dict_qwen3vl() -> dict[str, torch.Tensor]:
    """State dict mimicking HF Qwen3-VL with tie_word_embeddings=True.

    HF keys: model.visual.*, model.language_model.embed_tokens.weight,
    model.language_model.layers.*, model.language_model.lm_head.weight (absent).
    """
    embed = torch.randn(100, 64)
    return {
        "model.visual.patch_embed.proj.weight": torch.randn(64, 3, 14, 14),
        "model.language_model.embed_tokens.weight": embed,
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        "model.language_model.model.norm.weight": torch.randn(64),
    }


class TestQwen25VLCausalLMModelTiedWeights:
    """Composite 3-model CausalLM: decoder + vision + embedding."""

    def test_lm_head_present_when_tied(self):
        config = dataclasses.replace(_BASE_CONFIG, model_type="qwen2_5_vl")
        model = Qwen25VLCausalLMModel(config)
        sd = _fake_state_dict_qwen25vl()
        result = model.preprocess_weights(sd)

        assert "decoder.lm_head.weight" in result, (
            "lm_head.weight must be present for tied composite models"
        )

    def test_lm_head_shares_data_ptr_with_embed(self):
        config = dataclasses.replace(_BASE_CONFIG, model_type="qwen2_5_vl")
        model = Qwen25VLCausalLMModel(config)
        sd = _fake_state_dict_qwen25vl()
        result = model.preprocess_weights(sd)

        embed = result["decoder.model.embed_tokens.weight"]
        head = result["decoder.lm_head.weight"]
        assert embed.data_ptr() == head.data_ptr(), (
            "Tied weights must share the same data_ptr() for ONNX dedup"
        )

    def test_builds_different_decoder_vision_and_embedding_layouts(self):
        decoder = QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="olive",
            sym=True,
        )
        config = dataclasses.replace(
            _BASE_CONFIG,
            tie_word_embeddings=False,
            quantization=decoder,
            component_quantization={
                "decoder": decoder,
                "vision_encoder": QuantizationConfig(
                    bits=8,
                    group_size=32,
                    quant_method="olive",
                    sym=True,
                ),
                "embedding": QuantizationConfig(
                    bits=2,
                    group_size=16,
                    quant_method="olive",
                    sym=True,
                    quantize_embeddings=True,
                ),
            },
        )

        package = build_from_module(
            Qwen25VLCausalLMModel(config),
            config,
            task="qwen-vl",
        )

        decoder_layouts = {
            (
                node.attributes["bits"].as_int(),
                node.attributes["block_size"].as_int(),
            )
            for node in package["decoder"].graph
            if node.op_type == "MatMulNBits"
        }
        vision_layouts = {
            (
                node.attributes["bits"].as_int(),
                node.attributes["block_size"].as_int(),
            )
            for node in package["vision_encoder"].graph
            if node.op_type == "MatMulNBits"
        }

        assert decoder_layouts == {(4, 16)}
        assert vision_layouts == {(8, 32)}
        assert any(
            node.op_type == "GatherBlockQuantized"
            and node.attributes["bits"].as_int() == 2
            and node.attributes["block_size"].as_int() == 16
            for node in package["embedding"].graph
        )

    def test_packed_embedding_routes_only_to_embedding_component(self):
        decoder = QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="olive",
            sym=True,
        )
        config = dataclasses.replace(
            _BASE_CONFIG,
            tie_word_embeddings=False,
            quantization=decoder,
            component_quantization={
                "decoder": decoder,
                "embedding": QuantizationConfig(
                    bits=2,
                    group_size=16,
                    quant_method="olive",
                    sym=True,
                    quantize_embeddings=True,
                ),
            },
        )
        model = Qwen25VLCausalLMModel(config)
        configure_component_quantization(model, config, "qwen-vl")
        renamed = model.preprocess_weights(
            {
                "model.embed_tokens.weight_qweight": torch.zeros(100, 16, dtype=torch.uint8),
                "model.embed_tokens.weight_scales": torch.ones(100, 4),
            }
        )

        assert not any(key.startswith("decoder.") for key in renamed)
        result = preprocess_component_quantized_state_dict(
            renamed,
            model,
            config,
            "qwen-vl",
            ("decoder", "vision_encoder", "embedding"),
        )

        assert result["embedding.embed_tokens.qweight"].shape == (100, 16)
        assert result["embedding.embed_tokens.scales"].shape == (100, 4)


class TestQwen25VLDecoderModelTiedWeights:
    """Standalone decoder (no composite prefix)."""

    def test_lm_head_present_when_tied(self):
        config = dataclasses.replace(_BASE_CONFIG, model_type="qwen2_5_vl")
        model = Qwen25VLDecoderModel(config)
        sd = _fake_state_dict_qwen25vl()
        result = model.preprocess_weights(sd)

        assert "lm_head.weight" in result

    def test_lm_head_shares_data_ptr_with_embed(self):
        config = dataclasses.replace(_BASE_CONFIG, model_type="qwen2_5_vl")
        model = Qwen25VLDecoderModel(config)
        sd = _fake_state_dict_qwen25vl()
        result = model.preprocess_weights(sd)

        embed = result["model.embed_tokens.weight"]
        head = result["lm_head.weight"]
        assert embed.data_ptr() == head.data_ptr()


class TestQwen3VL3ModelCausalLMModelTiedWeights:
    """Composite 3-model CausalLM for Qwen3-VL."""

    def test_lm_head_present_when_tied(self):
        config = dataclasses.replace(
            _BASE_CONFIG,
            model_type="qwen3_vl",
        )
        model = Qwen3VL3ModelCausalLMModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        assert "decoder.lm_head.weight" in result

    def test_lm_head_shares_data_ptr_with_embed(self):
        config = dataclasses.replace(
            _BASE_CONFIG,
            model_type="qwen3_vl",
        )
        model = Qwen3VL3ModelCausalLMModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        embed = result["decoder.model.embed_tokens.weight"]
        head = result["decoder.lm_head.weight"]
        assert embed.data_ptr() == head.data_ptr()


class TestQwen3VLDecoderModelTiedWeights:
    """Standalone Qwen3-VL decoder."""

    def test_lm_head_present_when_tied(self):
        config = dataclasses.replace(
            _BASE_CONFIG,
            model_type="qwen3_vl",
        )
        model = Qwen3VLDecoderModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        assert "lm_head.weight" in result

    def test_lm_head_shares_data_ptr_with_embed(self):
        config = dataclasses.replace(
            _BASE_CONFIG,
            model_type="qwen3_vl",
        )
        model = Qwen3VLDecoderModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        embed = result["embed_tokens.weight"]
        head = result["lm_head.weight"]
        assert embed.data_ptr() == head.data_ptr()
