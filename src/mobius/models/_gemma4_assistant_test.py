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

    def test_assistant_fields_resolved_via_parent_when_unwrapped(self):
        # Regression: ``build()`` unwraps ``hf_config.text_config`` before
        # calling _config_from_hf, so ``config`` arrives as the inner
        # Gemma4TextConfig (without backbone_hidden_size /
        # use_ordered_embeddings / num_centroids /
        # centroid_intermediate_top_k).  Those fields must be resolved
        # from ``parent_config`` (the original wrapper).
        full = _e2b_like_hf_config()
        text_only = full.text_config  # what build() would pass as ``config``
        cfg = Gemma4AssistantConfig.from_transformers(text_only, parent_config=full)
        assert cfg.backbone_hidden_size == 1536
        assert cfg.use_ordered_embeddings is True
        assert cfg.num_centroids == 2048
        assert cfg.centroid_intermediate_top_k == 32
        # And the standard Gemma4 fields are still correct.
        assert cfg.hidden_size == 256
        assert cfg.num_hidden_layers == 4


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

    def test_model_submodule_layout(self):
        m = Gemma4AssistantCausalLMModel(self._cfg())
        assert hasattr(m, "model")
        assert hasattr(m.model, "layers")
        assert hasattr(m.model, "norm")
        assert hasattr(m.model, "rotary_emb_local")
        assert hasattr(m.model, "rotary_emb_global")
        assert len(m.model.layers) == self._cfg().num_hidden_layers

    def test_layer_types_match_config(self):
        cfg = self._cfg()
        m = Gemma4AssistantCausalLMModel(cfg)
        assert [layer.layer_type for layer in m.model.layers] == cfg.layer_types


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


class TestGemma4AssistantBuildGraph:
    """End-to-end graph build: drive Gemma4AssistantTask + the real model."""

    def _cfg(self):
        return Gemma4AssistantConfig.from_transformers(
            _e2b_like_hf_config(), parent_config=_e2b_like_hf_config()
        )

    def _build(self):
        cfg = self._cfg()
        module = Gemma4AssistantCausalLMModel(cfg)
        from mobius._builder import build_from_module
        pkg = build_from_module(module, cfg, task=Gemma4AssistantTask())
        return cfg, pkg["model"]

    def test_build_returns_valid_model(self):
        import onnx_ir as ir
        _cfg, model = self._build()
        assert isinstance(model, ir.Model)
        assert model.graph.num_nodes() > 0

    def test_graph_inputs(self):
        cfg, model = self._build()
        names = [v.name for v in model.graph.inputs]
        assert "inputs_embeds" in names
        assert "position_ids" in names
        # E2B has both sliding and full layers — both shared_kv pairs present.
        assert "shared_kv.sliding_attention.key" in names
        assert "shared_kv.sliding_attention.value" in names
        assert "shared_kv.full_attention.key" in names
        assert "shared_kv.full_attention.value" in names
        # No own KV cache.
        assert not any(n.startswith("past_key_values.") for n in names)

    def test_graph_outputs(self):
        cfg, model = self._build()
        names = [v.name for v in model.graph.outputs]
        assert "logits" in names
        assert "projected_state" in names
        # No own KV cache.
        assert not any(n.startswith("present.") for n in names)

    def test_inputs_embeds_shape(self):
        cfg, model = self._build()
        ie = next(v for v in model.graph.inputs if v.name == "inputs_embeds")
        # Last dim = 2 * backbone_hidden_size.
        assert ie.shape[-1] == 2 * cfg.backbone_hidden_size

    def test_shared_kv_full_shape_uses_global_head_dim(self):
        cfg, model = self._build()
        full_k = next(v for v in model.graph.inputs if v.name == "shared_kv.full_attention.key")
        assert full_k.shape[-1] == (cfg.global_head_dim or cfg.head_dim)

    def test_shared_kv_sliding_shape_uses_local_head_dim(self):
        cfg, model = self._build()
        sliding_k = next(
            v for v in model.graph.inputs if v.name == "shared_kv.sliding_attention.key"
        )
        assert sliding_k.shape[-1] == cfg.head_dim

    def test_logits_last_dim_is_vocab(self):
        import onnx_ir as ir
        cfg, model = self._build()
        logits = next(v for v in model.graph.outputs if v.name == "logits")
        last = logits.shape[-1]
        # With the centroid-routed sparse head (use_ordered_embeddings=True
        # in the test fixture), the scatter-built output may carry a
        # symbolic last dim through shape inference even though at runtime
        # the values have shape vocab_size.  Accept either form.
        assert (
            last == cfg.vocab_size or isinstance(last, ir.SymbolicDim)
        ), f"expected vocab_size ({cfg.vocab_size}) or a symbolic dim, got {last!r}"

    def test_projected_state_last_dim_is_backbone(self):
        cfg, model = self._build()
        ps = next(v for v in model.graph.outputs if v.name == "projected_state")
        assert ps.shape[-1] == cfg.backbone_hidden_size


class TestGemma4AssistantPreprocessWeights:
    """Verify the HF state-dict bridge for tied weights and dropped extras."""

    def _model(self):
        cfg = Gemma4AssistantConfig.from_transformers(
            _e2b_like_hf_config(), parent_config=_e2b_like_hf_config()
        )
        return Gemma4AssistantCausalLMModel(cfg)

    def test_aliases_embed_tokens_to_lm_head(self):
        import torch

        m = self._model()
        embed = torch.zeros(262144, 256)
        sd = {"model.embed_tokens.weight": embed}
        sd = m.preprocess_weights(sd)
        assert "lm_head.weight" in sd
        assert sd["lm_head.weight"] is embed
        # The original embed_tokens key is dropped (no mobius consumer).
        assert "model.embed_tokens.weight" not in sd

    def test_does_not_overwrite_existing_lm_head(self):
        import torch

        m = self._model()
        existing_head = torch.ones(262144, 256)
        sd = {
            "lm_head.weight": existing_head,
            "model.embed_tokens.weight": torch.zeros(262144, 256),
        }
        sd = m.preprocess_weights(sd)
        assert sd["lm_head.weight"] is existing_head

    def test_drops_unsupported_masked_embedding_keys(self):
        import torch

        # Use a config with ordered_embeddings disabled so the model
        # has no masked_embedding module — the preprocess should drop
        # the keys.
        cfg = Gemma4AssistantConfig.from_transformers(
            _e2b_like_hf_config(), parent_config=_e2b_like_hf_config()
        )
        cfg = dataclasses.replace(cfg, use_ordered_embeddings=False)
        m = Gemma4AssistantCausalLMModel(cfg)
        sd = {
            "lm_head.weight": torch.zeros(262144, 256),
            "masked_embedding.centroids.weight": torch.zeros(2048, 256),
            "masked_embedding.token_ordering": torch.zeros(262144, dtype=torch.long),
        }
        sd = m.preprocess_weights(sd)
        assert "masked_embedding.centroids.weight" not in sd
        assert "masked_embedding.token_ordering" not in sd

    def test_idempotent_when_keys_missing(self):
        m = self._model()
        sd = {}
        # Should not raise even when both keys are absent.
        sd = m.preprocess_weights(sd)
        assert sd == {}

    def test_keeps_masked_embedding_keys_when_ordered_embeddings_on(self):
        import torch

        # E2B test fixture has use_ordered_embeddings=True.
        m = self._model()
        assert m.config.use_ordered_embeddings is True
        sd = {
            "lm_head.weight": torch.zeros(262144, 256),
            "masked_embedding.centroids.weight": torch.zeros(2048, 256),
            "masked_embedding.token_ordering": torch.zeros(262144, dtype=torch.long),
        }
        sd = m.preprocess_weights(sd)
        # Must be kept — the mobius masked_embedding module needs them.
        assert "masked_embedding.centroids.weight" in sd
        assert "masked_embedding.token_ordering" in sd


class TestGemma4AssistantOrderedEmbeddings:
    """When use_ordered_embeddings=True, the assistant must build the
    centroid-routed sparse LM head and route logits through it."""

    def _model(self, **cfg_over):
        cfg = Gemma4AssistantConfig.from_transformers(
            _e2b_like_hf_config(), parent_config=_e2b_like_hf_config()
        )
        if cfg_over:
            cfg = dataclasses.replace(cfg, **cfg_over)
        return cfg, Gemma4AssistantCausalLMModel(cfg)

    def test_masked_embedding_module_present(self):
        _cfg, m = self._model()
        assert m.masked_embedding is not None
        # Centroids: Linear[hidden -> num_centroids]
        out_f, in_f = m.masked_embedding.centroids.weight.shape
        assert in_f == 256
        assert out_f == 2048
        # token_ordering: [vocab_size] INT64
        ord_shape = m.masked_embedding.token_ordering.shape
        assert list(ord_shape) == [262144]

    def test_masked_embedding_absent_when_disabled(self):
        _cfg, m = self._model(use_ordered_embeddings=False)
        assert m.masked_embedding is None

    def test_build_with_ordered_embeddings_produces_logits(self):
        from mobius._builder import build_from_module
        cfg, module = self._model()
        pkg = build_from_module(module, cfg, task=Gemma4AssistantTask())
        names = [v.name for v in pkg["model"].graph.outputs]
        # Same outputs regardless of head choice (the sparse head produces
        # a dense [B, q_len, vocab] tensor with mask_value at unselected positions).
        assert "logits" in names
        assert "projected_state" in names

    def test_build_with_ordered_embeddings_has_centroid_weights(self):
        from mobius._builder import build_from_module
        cfg, module = self._model()
        pkg = build_from_module(module, cfg, task=Gemma4AssistantTask())
        init_names = list(pkg["model"].graph.initializers.keys())
        assert any("centroids" in n for n in init_names)
        assert any("token_ordering" in n for n in init_names)
