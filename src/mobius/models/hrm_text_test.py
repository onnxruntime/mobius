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
from mobius.tasks import CausalLMTask, get_task

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


def test_raw_json_config_still_gets_rope():
    """A raw config.json must not silently export a position-free graph.

    ``HrmTextRotaryEmbedding`` is unconditional upstream, but the raw
    ``config.json`` only carries the *default* ``rope_theta`` of 10000.0, which
    the generic extractor ignores as a RoPE signal.
    """
    config = _mobius_config()
    assert config.rope_type == "default"
    assert config.rope_theta == pytest.approx(10_000.0)
    module, _ = _build_package(config)
    assert module.model.rotary_emb is not None


def test_trusted_hf_config_rope_matches_raw_json():
    hf_config = _hf_config()
    trusted = HrmTextConfig.from_transformers(hf_config)
    raw = _mobius_config()
    assert trusted.rope_type == raw.rope_type == "default"
    assert trusted.rope_theta == pytest.approx(raw.rope_theta)


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


def test_prefix_lm_graph_declares_token_type_ids():
    config = _mobius_config()
    assert config.prefix_lm is True
    module, pkg = _build_package(config)
    assert module.requires_token_type_ids is True
    input_names = [value.name for value in pkg["model"].graph.inputs]
    assert "token_type_ids" in input_names
    # Declared next to the other per-position inputs, before the cache.
    assert input_names[:4] == [
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
    ]


def test_causal_only_config_has_no_token_type_ids_input():
    """``prefix_lm=False`` keeps the standard causal input set (and GQA path)."""
    config = _mobius_config(prefix_lm=False)
    module, pkg = _build_package(config)
    assert module.requires_token_type_ids is False
    assert "token_type_ids" not in {value.name for value in pkg["model"].graph.inputs}


def test_prefix_lm_forward_requires_token_type_ids():
    """A direct forward() call must not silently fall back to causal masking."""
    config = _mobius_config()
    module = HrmTextCausalLMModel(config)
    with pytest.raises(ValueError, match="token_type_ids"):
        get_task("text-generation").build(_StripTokenTypeIds(module), config)


def test_prefix_lm_static_cache_reports_unsupported_mode():
    """Static-cache builds fail for the PrefixLM limitation, not a missing input."""
    config = _mobius_config()
    module = HrmTextCausalLMModel(config)
    with pytest.raises(NotImplementedError, match="not supported with the static KV cache"):
        CausalLMTask(static_cache=True, max_seq_len=128).build(module, config)


class _StripTokenTypeIds:
    """Wrap a module so the task's ``token_type_ids`` never reaches forward()."""

    def __init__(self, module):
        self._module = module
        self.requires_token_type_ids = False

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __call__(self, op, **kwargs):
        kwargs.pop("token_type_ids", None)
        return self._module(op, **kwargs)


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
    """PrefixLM prefill, causal fallback, and cached decode must match upstream.

    Three things are pinned here:

    * **PrefixLM prefill** — ``token_type_ids == 1`` over the whole prompt must
      reproduce upstream's ``block_sequence_ids = where(tt == 1, 0, -1)``
      bidirectional overlay.
    * **Causal fallback** — all-zero ``token_type_ids`` through the *same*
      graph must reproduce upstream's ``token_type_ids=None`` path, and the gap
      between the two modes must be the same size on both sides. Without that
      second half a graph that ignored the overlay entirely would still pass.
    * **Cached decode** — the KV-cache slot layout, plus the fact that a
      generated token leaves the prefix block and attends causally.
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
    # Two rows verify that z_L_init is broadcast to the actual batch rather
    # than accidentally initialized like a zero-length dynamic KV cache.
    input_ids = rng.integers(1, _VOCAB, size=(2, 6)).astype(np.int64)
    attention_mask = np.ones_like(input_ids)
    position_ids = np.broadcast_to(
        np.arange(input_ids.shape[1], dtype=np.int64), input_ids.shape
    ).copy()

    session = OnnxModelSession(pkg["model"])
    try:

        def _prefill(token_type_ids: np.ndarray) -> dict[str, np.ndarray]:
            feeds: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
            }
            empty = np.zeros((input_ids.shape[0], _HEADS, 0, _HEAD_DIM), dtype=np.float32)
            for slot in range(_TOTAL_SLOTS):
                feeds[f"past_key_values.{slot}.key"] = empty
                feeds[f"past_key_values.{slot}.value"] = empty
            return session.run(feeds)

        onnx_prefix = _prefill(np.ones_like(input_ids))
        onnx_causal = _prefill(np.zeros_like(input_ids))

        with torch.no_grad():
            hf_prefix = hf_model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
                position_ids=torch.from_numpy(position_ids),
                token_type_ids=torch.ones_like(torch.from_numpy(input_ids)),
                use_cache=True,
            )
            hf_causal_logits = hf_model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
                position_ids=torch.from_numpy(position_ids),
                use_cache=False,
            ).logits.numpy()
        hf_prefix_logits = hf_prefix.logits.numpy()

        np.testing.assert_allclose(
            onnx_prefix["logits"], hf_prefix_logits, rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            onnx_causal["logits"], hf_causal_logits, rtol=1e-3, atol=1e-3
        )

        hf_delta = float(np.abs(hf_prefix_logits - hf_causal_logits).max())
        onnx_delta = float(np.abs(onnx_prefix["logits"] - onnx_causal["logits"]).max())
        assert hf_delta > 1e-4, f"HF PrefixLM overlay had no effect ({hf_delta})"
        assert onnx_delta == pytest.approx(hf_delta, rel=1e-2, abs=1e-5)

        # Cached decode: the generated token is NOT part of the prefix block.
        next_id = np.argmax(hf_prefix_logits[:, -1], axis=-1, keepdims=True).astype(np.int64)
        decode_mask = np.ones((input_ids.shape[0], input_ids.shape[1] + 1), dtype=np.int64)
        decode_pos = np.full((input_ids.shape[0], 1), input_ids.shape[1], dtype=np.int64)
        decode_feeds: dict[str, np.ndarray] = {
            "input_ids": next_id,
            "attention_mask": decode_mask,
            "position_ids": decode_pos,
            "token_type_ids": np.zeros_like(next_id),
        }
        for slot in range(_TOTAL_SLOTS):
            decode_feeds[f"past_key_values.{slot}.key"] = onnx_prefix[f"present.{slot}.key"]
            decode_feeds[f"past_key_values.{slot}.value"] = onnx_prefix[
                f"present.{slot}.value"
            ]
        decode = session.run(decode_feeds)

        with torch.no_grad():
            hf_decode = hf_model(
                input_ids=torch.from_numpy(next_id),
                attention_mask=torch.from_numpy(decode_mask),
                position_ids=torch.from_numpy(decode_pos),
                past_key_values=hf_prefix.past_key_values,
                use_cache=True,
            )
        np.testing.assert_allclose(
            decode["logits"], hf_decode.logits.numpy(), rtol=1e-3, atol=1e-3
        )
    finally:
        session.close()
