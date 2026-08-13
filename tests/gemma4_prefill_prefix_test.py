# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch

from mobius import build_from_module
from mobius._configs import Gemma4Config, VisionConfig
from mobius._registry import registry
from mobius.models.gemma4 import _split_per_layer_projection_weight


def _make_config(*, with_vision: bool = False) -> Gemma4Config:
    return Gemma4Config(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=256,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attn_qk_norm=True,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=8,
        global_head_dim=16,
        global_rope_theta=10_000.0,
        global_partial_rotary_factor=0.25,
        final_logit_softcapping=30.0,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=64,
        split_per_layer_embedding=True,
        image_token_id=255999 if with_vision else None,
        pad_token_id=0,
        tie_word_embeddings=False,
        num_kv_shared_layers=2,
        vision=(
            VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                patch_size=16,
                norm_eps=1e-6,
            )
            if with_vision
            else None
        ),
    )


def test_prunes_gemma4_shared_layer_prefix() -> None:
    config = _make_config()
    module = registry.get("gemma4_text")(config)
    model = build_from_module(
        module,
        config,
        task="gemma4-text-generation",
        execution_provider="webgpu",
        prune_prefill_prefix=True,
    )["model"]

    producer = next(
        node
        for node in model.graph
        if node.op_type == "MatMul" and "/per_layer_model_projection/" in node.name
    )
    consumer = next(
        node
        for node in model.graph
        if node.op_type == "MatMul" and "/per_layer_model_projection_consumer/" in node.name
    )
    assert producer.outputs[0].shape[1] != 1
    assert producer.outputs[0].shape[2] == 16
    assert consumer.inputs[0].shape[1:] == (1, 64)
    assert consumer.outputs[0].shape[1:] == (1, 16)

    first_shared_norm = next(
        node for node in model.graph if "layers.2/input_layernorm" in node.name
    )
    assert first_shared_norm.inputs[0].shape[1:] == (1, 64)
    logits = next(value for value in model.graph.outputs if value.name == "logits")
    assert logits.shape[1:] == (1, config.vocab_size)

    consumer_embedding_scale = next(
        node
        for node in model.graph
        if node.op_type == "Mul" and "embed_tokens_per_layer_split.2" in node.name
    )
    assert any(
        node.op_type == "Gather"
        and any(
            input_value is consumer_embedding_scale.outputs[0] for input_value in node.inputs
        )
        for node in model.graph
    )
    assert not any(node.op_type == "CastLike" for node in model.graph)


def test_multimodal_task_prunes_decoder_prefix() -> None:
    config = _make_config(with_vision=True)
    module = registry.get("gemma4")(config)
    package = build_from_module(
        module,
        config,
        task="gemma4",
        execution_provider="webgpu",
        prune_prefill_prefix=True,
    )

    logits = next(
        value for value in package["decoder"].graph.outputs if value.name == "logits"
    )
    assert logits.shape[1:] == (1, config.vocab_size)


def test_splits_per_layer_projection_weight() -> None:
    config = _make_config()
    original = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    state_dict = {"model.per_layer_model_projection.weight": original.clone()}

    _split_per_layer_projection_weight(state_dict, "model.", config)

    assert torch.equal(state_dict["model.per_layer_model_projection.weight"], original[:16])
    assert torch.equal(
        state_dict["model.per_layer_model_projection_consumer.weight"],
        original[16:],
    )
