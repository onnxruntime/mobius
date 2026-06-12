# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Scaffolding tests for the Gemma4-Assistant draft model.

Verifies the parts implemented in autonomous Phase 2 (gemma4):
- ``Gemma4AssistantConfig.from_transformers`` lifts the nested
  ``text_config`` plus assistant-specific fields onto the top level.
- ``Gemma4AssistantConfig.validate`` rejects unsupported feature
  combinations (per-layer inputs, MoE, double-wide MLP, KV-share count
  mismatch).
- Weight modules (``pre_projection``, ``post_projection``, ``lm_head``,
  ``norm``) have the correct shapes for downstream weight loading.
- Registry routes both ``model_type="gemma4_assistant"`` and
  ``architectures=["Gemma4AssistantForCausalLM"]`` to the same class +
  task.
- Task name resolves to a ``Gemma4AssistantTask``.

The end-to-end build path (``Gemma4AssistantTask.build`` →
``Gemma4AssistantCausalLMModel.forward`` → ONNX) is intentionally **not**
covered here — both raise ``NotImplementedError`` until the model
forward is implemented; see the checklist in
``mobius/models/gemma4_assistant.py``.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from mobius._configs import Gemma4AssistantConfig, Gemma4Config
from mobius._registry import registry
from mobius.models.gemma4_assistant import Gemma4AssistantCausalLMModel
from mobius.tasks import Gemma4AssistantTask, get_task


def _e2b_like_hf_config():
    """Build a SimpleNamespace mirroring google/gemma-4-E2B-it-assistant."""
    text = SimpleNamespace(
        model_type="gemma4_text",
        vocab_size=262144,
        hidden_size=256,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_kv_shared_layers=4,
        head_dim=256,
        global_head_dim=512,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        sliding_window=512,
        max_position_embeddings=131072,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        attention_bias=False,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        # Gemma4 RoPE knobs
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default",
            },
        },
    )
    return SimpleNamespace(
        model_type="gemma4_assistant",
        text_config=text,
        backbone_hidden_size=1536,
        use_ordered_embeddings=True,
        num_centroids=2048,
        centroid_intermediate_top_k=32,
        tie_word_embeddings=True,
        architectures=["Gemma4AssistantForCausalLM"],
    )


class TestGemma4AssistantConfigFromTransformers:
    def test_lifts_nested_text_config(self):
        cfg = Gemma4AssistantConfig.from_transformers(
            _e2b_like_hf_config(), parent_config=_e2b_like_hf_config()
        )
        assert isinstance(cfg, Gemma4AssistantConfig)
        assert isinstance(cfg, Gemma4Config)  # subclass check
        assert cfg.hidden_size == 256
        assert cfg.num_hidden_layers == 4
        assert cfg.num_key_value_heads == 1
        assert cfg.head_dim == 256
        assert cfg.global_head_dim == 512
        assert cfg.layer_types == [
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        ]
        assert cfg.sliding_window == 512
        assert cfg.vocab_size == 262144
        assert cfg.num_kv_shared_layers == 4

    def test_assistant_specific_fields(self):
        cfg = Gemma4AssistantConfig.from_transformers(
            _e2b_like_hf_config(), parent_config=_e2b_like_hf_config()
        )
        assert cfg.backbone_hidden_size == 1536
        assert cfg.use_ordered_embeddings is True
        assert cfg.num_centroids == 2048
        assert cfg.centroid_intermediate_top_k == 32


class TestGemma4AssistantConfigValidate:
    def _cfg(self, **overrides):
        cfg = Gemma4AssistantConfig.from_transformers(
            _e2b_like_hf_config(), parent_config=_e2b_like_hf_config()
        )
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        return cfg

    def test_validate_passes_on_e2b_shape(self):
        # No-throw is the test.
        self._cfg().validate()

    def test_rejects_partial_kv_sharing(self):
        with pytest.raises(ValueError, match="num_kv_shared_layers"):
            self._cfg(num_kv_shared_layers=2).validate()

    def test_rejects_per_layer_input_gating(self):
        with pytest.raises(ValueError, match="hidden_size_per_layer_input"):
            self._cfg(hidden_size_per_layer_input=64).validate()

    def test_rejects_moe(self):
        with pytest.raises(ValueError, match="enable_moe_block"):
            self._cfg(enable_moe_block=True).validate()

    def test_rejects_double_wide_mlp(self):
        with pytest.raises(ValueError, match="use_double_wide_mlp"):
            self._cfg(use_double_wide_mlp=True).validate()

    def test_rejects_per_layer_vocab(self):
        with pytest.raises(ValueError, match="vocab_size_per_layer_input"):
            self._cfg(vocab_size_per_layer_input=256).validate()


class TestGemma4AssistantModelWeights:
    def _cfg(self):
        return Gemma4AssistantConfig.from_transformers(
            _e2b_like_hf_config(), parent_config=_e2b_like_hf_config()
        )

    def test_pre_projection_shape(self):
        cfg = self._cfg()
        m = Gemma4AssistantCausalLMModel(cfg)
        # Linear weight is [out_features, in_features].
        out_features, in_features = m.pre_projection.weight.shape
        assert in_features == 2 * cfg.backbone_hidden_size
        assert out_features == cfg.hidden_size

    def test_post_projection_shape(self):
        cfg = self._cfg()
        m = Gemma4AssistantCausalLMModel(cfg)
        out_features, in_features = m.post_projection.weight.shape
        assert in_features == cfg.hidden_size
        assert out_features == cfg.backbone_hidden_size

    def test_lm_head_shape(self):
        cfg = self._cfg()
        m = Gemma4AssistantCausalLMModel(cfg)
        out_features, in_features = m.lm_head.weight.shape
        assert in_features == cfg.hidden_size
        assert out_features == cfg.vocab_size

    def test_no_pre_projection_bias(self):
        m = Gemma4AssistantCausalLMModel(self._cfg())
        assert m.pre_projection.bias is None
        assert m.post_projection.bias is None
        assert m.lm_head.bias is None

    def test_forward_raises_not_implemented(self):
        m = Gemma4AssistantCausalLMModel(self._cfg())
        with pytest.raises(NotImplementedError):
            m.forward(None)


class TestGemma4AssistantRegistry:
    def test_model_type_routes_to_assistant(self):
        assert "gemma4_assistant" in registry
        cls = registry.get("gemma4_assistant")
        assert cls is Gemma4AssistantCausalLMModel

    def test_architecture_routes_to_assistant(self):
        # build()'s architectures-based override looks up
        # parent_config.architectures[0] in the registry.
        assert "Gemma4AssistantForCausalLM" in registry
        cls = registry.get("Gemma4AssistantForCausalLM")
        assert cls is Gemma4AssistantCausalLMModel

    def test_registry_task_is_gemma4_assistant(self):
        assert registry.get_task("gemma4_assistant") == "gemma4-assistant"

    def test_registry_config_class_is_assistant_config(self):
        assert registry.get_config_class("gemma4_assistant") is Gemma4AssistantConfig

    def test_task_name_resolves_to_instance(self):
        task = get_task("gemma4-assistant")
        assert isinstance(task, Gemma4AssistantTask)
