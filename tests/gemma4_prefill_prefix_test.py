# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius._configs import Gemma4Config, VisionConfig
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
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


# ---------------------------------------------------------------------------
# Numerical parity: pruned package must reproduce the unpruned final row
# ---------------------------------------------------------------------------


def _parity_config() -> Gemma4Config:
    """Tiny hybrid config with a KV-shared tail and two distinct head sizes."""
    return Gemma4Config(
        num_hidden_layers=6,
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
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=8,
        # Distinct global head size: full-attention layers cache 32-wide K/V,
        # sliding layers cache 16-wide K/V.
        global_head_dim=32,
        global_rope_theta=10_000.0,
        global_partial_rotary_factor=0.25,
        final_logit_softcapping=30.0,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=64,
        split_per_layer_embedding=True,
        max_position_embeddings=256,
        pad_token_id=0,
        tie_word_embeddings=False,
        num_kv_shared_layers=2,
    )


def _build_text_model(config: Gemma4Config, *, execution_provider: str, prune: bool):
    module = registry.get("gemma4_text")(config)
    return build_from_module(
        module,
        config,
        task="gemma4-text-generation",
        execution_provider=execution_provider,
        prune_prefill_prefix=prune,
    )["model"]


def _fill_random_weights(model, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for initializer in model.graph.initializers.values():
        if initializer.const_value is None:
            array = (rng.standard_normal(tuple(initializer.shape)) * 0.05).astype(np.float32)
            initializer.const_value = ir.tensor(array, name=initializer.name)
        weights[initializer.name] = initializer.const_value.numpy()
    return weights


def _copy_weights(model, weights: dict[str, np.ndarray]) -> None:
    for initializer in model.graph.initializers.values():
        if initializer.name in weights:
            initializer.const_value = ir.tensor(
                weights[initializer.name], name=initializer.name
            )
        elif initializer.const_value is None:
            raise AssertionError(
                f"pruned graph declares weight {initializer.name!r} that the "
                "unpruned graph does not"
            )


def _feeds(config: Gemma4Config, seq_len: int) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {
        "input_ids": (np.arange(1, seq_len + 1, dtype=np.int64) % config.vocab_size)[None],
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64)[None],
    }
    # Cache-owning layers are the contiguous prefix before the KV-shared tail;
    # each carries its own head size (global layers are double-wide).
    kv_layers = config.num_hidden_layers - (config.num_kv_shared_layers or 0)
    for index in range(kv_layers):
        head_dim = 32 if config.layer_types[index] == "full_attention" else 16
        empty = np.zeros((1, config.num_key_value_heads, 0, head_dim), dtype=np.float32)
        feeds[f"past_key_values.{index}.key"] = empty
        feeds[f"past_key_values.{index}.value"] = empty.copy()
    return feeds


@pytest.mark.parametrize(
    "execution_provider",
    # "default" exercises the opset-24 Attention + RotaryEmbedding path;
    # "cpu" exercises the fused GroupQueryAttention path.
    ["default", "cpu"],
)
@pytest.mark.parametrize("seq_len", [5, 12])
def test_pruned_prefill_matches_unpruned_final_row(
    execution_provider: str, seq_len: int
) -> None:
    """Prefill-prefix pruning must be a pure graph-surface optimisation.

    Regression guard for the Gemma 4 mid-stack truncation: at the first
    KV-shared layer the hidden states narrow to a single query position, so the
    per-layer RoPE ``(cos, sin)`` caches and the additive attention bias must
    narrow with them.  Without that, the ``RotaryEmbedding``/``Attention`` path
    fails outright at load/run time; ``seq_len=12`` additionally reaches past
    the 8-token sliding window so global (full-attention) layers exercise a
    different key extent from the sliding ones.
    """
    config = _parity_config()
    base = _build_text_model(config, execution_provider=execution_provider, prune=False)
    pruned = _build_text_model(config, execution_provider=execution_provider, prune=True)

    weights = _fill_random_weights(base)
    _copy_weights(pruned, weights)

    feeds = _feeds(config, seq_len)
    base_out = OnnxModelSession(base).run(feeds)
    pruned_out = OnnxModelSession(pruned).run(feeds)

    assert set(base_out) == set(pruned_out), "pruning changed the model's output surface"

    # Logits: the pruned package emits only the final row.
    expected_logits = base_out["logits"][:, -1:, :]
    assert pruned_out["logits"].shape == expected_logits.shape
    np.testing.assert_allclose(pruned_out["logits"], expected_logits, atol=1e-4, rtol=0)
    assert np.argmax(pruned_out["logits"][0, 0]) == np.argmax(expected_logits[0, 0])

    # KV cache: pruning must not touch cache-owning layers at all.
    kv_layers = config.num_hidden_layers - (config.num_kv_shared_layers or 0)
    present_names = sorted(name for name in base_out if name.startswith("present."))
    assert len(present_names) == 2 * kv_layers, (
        f"expected {kv_layers} cache-owning layers, got {present_names}"
    )
    for name in present_names:
        np.testing.assert_allclose(pruned_out[name], base_out[name], atol=1e-5, rtol=0)

    # Double head size survives pruning: global layers stay twice as wide.
    assert base_out["present.0.key"].shape[-1] == config.head_dim
    assert base_out["present.1.key"].shape[-1] == config.global_head_dim
