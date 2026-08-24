# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the HRM-Text hierarchical recurrent export.

Covers the three things that are unique to this architecture and that a
graph-build test cannot reach:

* config extraction — ``num_hidden_layers`` is inflated to one slot per unique
  attention invocation, from *both* a trusted HuggingFace config (already
  inflated) and a raw pinned ``config.json`` (not inflated);
* ``preprocess_weights`` — the checkpoint packs gate/q/k/v into
  ``attn.gqkv_proj`` and gate/up into ``mlp.gate_up_proj``;
* the H/L recurrence itself — prefill *and* a cached decode step are compared
  against HuggingFace, which is what actually pins the KV-cache slot order.

All tiny random configs -- no checkpoint download.
"""

from __future__ import annotations

import types

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._configs import HrmTextConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations._weight_loading import apply_weights
from mobius.models.hrm_text import (
    HrmTextCausalLMModel,
    _resolve_layers_per_stack,
)
from mobius.tasks import get_task

_HIDDEN = 64
_HEADS = 4
_HEAD_DIM = 16
_INTERMEDIATE = 128
_VOCAB = 256
_PER_STACK = 2
_H_CYCLES = 2
_L_CYCLES = 2
# One cache slot per unique attention invocation.
_TOTAL_SLOTS = _PER_STACK * _H_CYCLES * (_L_CYCLES + 1)


def _raw_json_config(**overrides) -> types.SimpleNamespace:
    """A stand-in for a raw pinned ``config.json`` (no HF ``__post_init__``)."""
    fields = {
        "model_type": "hrm_text",
        "vocab_size": _VOCAB,
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTERMEDIATE,
        # Raw checkpoints carry the *per-stack* depth here.
        "num_hidden_layers": _PER_STACK,
        "num_attention_heads": _HEADS,
        "num_key_value_heads": _HEADS,
        "head_dim": _HEAD_DIM,
        "H_cycles": _H_CYCLES,
        "L_cycles": _L_CYCLES,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": False,
        "initializer_range": 0.025,
        "prefix_lm": True,
        "pad_token_id": 0,
        "hidden_act": "silu",
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _mobius_config(**overrides) -> HrmTextConfig:
    config = HrmTextConfig.from_transformers(_raw_json_config(**overrides))
    config.dtype = ir.DataType.FLOAT
    return config


def _hf_config(**overrides):
    """Build the upstream ``HrmTextConfig`` for the same tiny architecture."""
    transformers = pytest.importorskip("transformers")
    fields = {
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTERMEDIATE,
        "num_attention_heads": _HEADS,
        "num_hidden_layers": _PER_STACK,
        "vocab_size": _VOCAB,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "pad_token_id": 0,
        "head_dim": _HEAD_DIM,
        "H_cycles": _H_CYCLES,
        "L_cycles": _L_CYCLES,
        "initializer_range": 0.025,
        "prefix_lm": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
    }
    fields.update(overrides)
    return transformers.AutoConfig.for_model("hrm_text", **fields)


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------


def test_raw_json_config_inflates_layer_count():
    config = _mobius_config()
    assert config.num_layers_per_stack == _PER_STACK
    assert config.num_hidden_layers == _TOTAL_SLOTS


def test_trusted_hf_config_keeps_inflated_layer_count():
    hf_config = _hf_config()
    # Upstream already inflated in its own __post_init__.
    assert hf_config.num_hidden_layers == _TOTAL_SLOTS
    assert hf_config.num_layers_per_stack == _PER_STACK

    config = HrmTextConfig.from_transformers(hf_config)
    assert config.num_hidden_layers == _TOTAL_SLOTS
    assert config.num_layers_per_stack == _PER_STACK


def test_config_forces_multi_head_attention():
    # HrmTextAttention hardcodes num_key_value_groups = 1, so a checkpoint
    # claiming grouped-query heads must not shrink k_proj/v_proj.
    config = _mobius_config(num_key_value_heads=1)
    assert config.num_key_value_heads == config.num_attention_heads == _HEADS


def test_embedding_scale_defaults_to_inverse_initializer_range():
    raw = _raw_json_config(initializer_range=0.025)
    del raw.prefix_lm
    config = HrmTextConfig.from_transformers(raw)
    assert config.embedding_scale == pytest.approx(1.0 / 0.025)
    # ``prefix_lm`` defaults to True upstream.
    assert config.prefix_lm is True


def test_explicit_embedding_scale_is_preserved():
    config = _mobius_config(embedding_scale=39.191835884530846)
    assert config.embedding_scale == pytest.approx(39.191835884530846)


def test_config_rejects_non_positive_cycles():
    with pytest.raises(ValueError, match="positive H_cycles"):
        HrmTextConfig.from_transformers(_raw_json_config(L_cycles=0))


def test_resolve_layers_per_stack_rejects_inconsistent_total():
    config = _mobius_config()
    config.num_hidden_layers = _TOTAL_SLOTS + 1
    with pytest.raises(ValueError, match="num_layers_per_stack"):
        _resolve_layers_per_stack(config)


def test_resolve_layers_per_stack_derives_from_total_when_unset():
    config = _mobius_config()
    config.num_layers_per_stack = None
    assert _resolve_layers_per_stack(config) == _PER_STACK


# ---------------------------------------------------------------------------
# Graph shape
# ---------------------------------------------------------------------------


def _build_package(config: HrmTextConfig):
    module = HrmTextCausalLMModel(config)
    return module, get_task("text-generation").build(module, config)


def test_graph_exposes_one_cache_slot_per_attention_invocation():
    config = _mobius_config()
    _, pkg = _build_package(config)
    graph = pkg["model"].graph
    input_names = {value.name for value in graph.inputs}
    output_names = {value.name for value in graph.outputs}
    for slot in range(_TOTAL_SLOTS):
        assert f"past_key_values.{slot}.key" in input_names
        assert f"past_key_values.{slot}.value" in input_names
        assert f"present.{slot}.key" in output_names
        assert f"present.{slot}.value" in output_names
    assert f"past_key_values.{_TOTAL_SLOTS}.key" not in input_names


def test_stack_weights_are_shared_across_recurrence_steps():
    config = _mobius_config()
    _, pkg = _build_package(config)
    names = set(pkg["model"].graph.initializers)
    # Two stacks x _PER_STACK layers of parameters, regardless of how many
    # times the recurrence invokes them.
    for stack in ("L_module", "H_module"):
        for layer in range(_PER_STACK):
            prefix = f"model.{stack}.layers.{layer}"
            assert f"{prefix}.self_attn.q_proj.weight" in names
            assert f"{prefix}.self_attn.gate_proj.weight" in names
        assert f"model.{stack}.layers.{_PER_STACK}.self_attn.q_proj.weight" not in names
    assert "model.z_L_init" in names


# ---------------------------------------------------------------------------
# Weight preprocessing
# ---------------------------------------------------------------------------


def _fused_checkpoint_state_dict(config: HrmTextConfig) -> dict[str, torch.Tensor]:
    """Mimic the layout of ``sapientinc/HRM-Text-1B``'s ``model.safetensors``."""
    head_width = config.num_attention_heads * config.head_dim
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(config.vocab_size, config.hidden_size),
        "model.z_L_init": torch.zeros(config.hidden_size),
        "lm_head.weight": torch.randn(config.vocab_size, config.hidden_size),
    }
    for stack in ("L_module", "H_module"):
        for layer in range(config.num_layers_per_stack):
            prefix = f"model.{stack}.layers.{layer}"
            state[f"{prefix}.attn.gqkv_proj.weight"] = torch.randn(
                4 * head_width, config.hidden_size
            )
            state[f"{prefix}.attn.o_proj.weight"] = torch.randn(config.hidden_size, head_width)
            state[f"{prefix}.mlp.gate_up_proj.weight"] = torch.randn(
                2 * config.intermediate_size, config.hidden_size
            )
            state[f"{prefix}.mlp.down_proj.weight"] = torch.randn(
                config.hidden_size, config.intermediate_size
            )
    return state


def test_preprocess_weights_unfuses_checkpoint_layout():
    config = _mobius_config()
    module = HrmTextCausalLMModel(config)
    state = _fused_checkpoint_state_dict(config)
    result = module.preprocess_weights(state)

    head_width = config.num_attention_heads * config.head_dim
    prefix = "model.L_module.layers.0"
    fused = state[f"{prefix}.attn.gqkv_proj.weight"]
    # Upstream conversion_mapping order: gate, q, k, v.
    for index, part in enumerate(("gate_proj", "q_proj", "k_proj", "v_proj")):
        expected = fused[index * head_width : (index + 1) * head_width]
        torch.testing.assert_close(result[f"{prefix}.self_attn.{part}.weight"], expected)

    fused_mlp = state[f"{prefix}.mlp.gate_up_proj.weight"]
    torch.testing.assert_close(
        result[f"{prefix}.mlp.gate_proj.weight"], fused_mlp[: config.intermediate_size]
    )
    torch.testing.assert_close(
        result[f"{prefix}.mlp.up_proj.weight"], fused_mlp[config.intermediate_size :]
    )
    # ``attn`` -> ``self_attn`` rename, and no fused key survives.
    assert f"{prefix}.self_attn.o_proj.weight" in result
    assert not any("gqkv_proj" in key or "gate_up_proj" in key for key in result)


def test_preprocess_weights_covers_every_graph_parameter():
    config = _mobius_config()
    module, pkg = _build_package(config)
    result = module.preprocess_weights(_fused_checkpoint_state_dict(config))
    missing = {
        name
        for name, init in pkg["model"].graph.initializers.items()
        if init.const_value is None and name not in result
    }
    assert not missing, sorted(missing)


def test_preprocess_weights_is_identity_for_converted_names():
    config = _mobius_config()
    module, pkg = _build_package(config)
    aligned = {
        name: torch.ones(list(init.shape))
        for name, init in pkg["model"].graph.initializers.items()
        if init.const_value is None
    }
    result = module.preprocess_weights(aligned)
    assert set(aligned) <= set(result)


def test_preprocess_weights_rejects_wrong_fused_width():
    config = _mobius_config()
    module = HrmTextCausalLMModel(config)
    state = _fused_checkpoint_state_dict(config)
    state["model.L_module.layers.0.attn.gqkv_proj.weight"] = torch.randn(
        3 * config.num_attention_heads * config.head_dim, config.hidden_size
    )
    with pytest.raises(ValueError, match="fused gqkv_proj"):
        module.preprocess_weights(state)


# ---------------------------------------------------------------------------
# Numerical parity against HuggingFace (prefill + cached decode)
# ---------------------------------------------------------------------------


def test_recurrence_matches_huggingface_prefill_and_decode():
    """Prefill *and* a cached decode step must match upstream.

    The decode step is the part that pins the KV-cache slot layout: an
    off-by-one stack ordering still passes a cacheless prefill comparison but
    reads the wrong slots on the second step.
    """
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(11)
    hf_config = _hf_config()
    hf_model = transformers.AutoModelForCausalLM.from_config(hf_config).float().eval()

    config = HrmTextConfig.from_transformers(hf_config)
    config.dtype = ir.DataType.FLOAT
    module, pkg = _build_package(config)
    apply_weights(pkg["model"], module.preprocess_weights(dict(hf_model.state_dict())))

    rng = np.random.default_rng(11)
    input_ids = rng.integers(1, _VOCAB, size=(1, 5)).astype(np.int64)
    attention_mask = np.ones_like(input_ids)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[np.newaxis, :]

    session = OnnxModelSession(pkg["model"])
    try:
        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        empty = np.zeros((1, _HEADS, 0, _HEAD_DIM), dtype=np.float32)
        for slot in range(_TOTAL_SLOTS):
            feeds[f"past_key_values.{slot}.key"] = empty
            feeds[f"past_key_values.{slot}.value"] = empty
        prefill = session.run(feeds)

        with torch.no_grad():
            hf_prefill = hf_model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
                position_ids=torch.from_numpy(position_ids),
                use_cache=True,
            )
        np.testing.assert_allclose(
            prefill["logits"], hf_prefill.logits.numpy(), rtol=1e-3, atol=1e-3
        )

        next_id = np.array([[int(hf_prefill.logits[0, -1].argmax())]], dtype=np.int64)
        decode_mask = np.ones((1, input_ids.shape[1] + 1), dtype=np.int64)
        decode_pos = np.array([[input_ids.shape[1]]], dtype=np.int64)
        decode_feeds: dict[str, np.ndarray] = {
            "input_ids": next_id,
            "attention_mask": decode_mask,
            "position_ids": decode_pos,
        }
        for slot in range(_TOTAL_SLOTS):
            decode_feeds[f"past_key_values.{slot}.key"] = prefill[f"present.{slot}.key"]
            decode_feeds[f"past_key_values.{slot}.value"] = prefill[f"present.{slot}.value"]
        decode = session.run(decode_feeds)

        with torch.no_grad():
            hf_decode = hf_model(
                input_ids=torch.from_numpy(next_id),
                attention_mask=torch.from_numpy(decode_mask),
                position_ids=torch.from_numpy(decode_pos),
                past_key_values=hf_prefill.past_key_values,
                use_cache=True,
            )
        np.testing.assert_allclose(
            decode["logits"], hf_decode.logits.numpy(), rtol=1e-3, atol=1e-3
        )
    finally:
        session.close()
