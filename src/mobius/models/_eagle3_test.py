# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build tests for the EAGLE-3 speculative-decoding draft model."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import Eagle3Config
from mobius._registry import registry
from mobius._testing import make_config
from mobius.models.eagle3 import Eagle3DraftModel
from mobius.tasks import Eagle3DraftTask, get_task


def _eagle3_config(**overrides) -> Eagle3Config:
    base = make_config(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    fields.update(
        num_hidden_layers=1,
        layer_types=["full_attention"],
        model_type="llama",
        rope_theta=1_000_000.0,
        partial_rotary_factor=1.0,
        vocab_size=151936,
        draft_vocab_size=50,
        tie_word_embeddings=False,
    )
    fields.update(overrides)
    return Eagle3Config(**fields)


class TestEagle3ModelParams:
    def test_head_has_expected_params(self):
        cfg = _eagle3_config()
        model = Eagle3DraftModel(cfg)
        names = [n for n, _ in model.named_parameters()]
        for name in (
            "fc.weight",
            "lm_head.weight",
            "input_layernorm.weight",
            "hidden_norm.weight",
            "post_attention_layernorm.weight",
            "norm.weight",
        ):
            assert name in names

    def test_head_borrows_embed_tokens(self):
        model = Eagle3DraftModel(_eagle3_config())
        names = [n for n, _ in model.named_parameters()]
        assert not any("embed_tokens" in n for n in names)

    def test_projection_shapes(self):
        cfg = _eagle3_config()
        model = Eagle3DraftModel(cfg)
        assert model.fc.weight.shape == (cfg.hidden_size, 3 * cfg.hidden_size)
        assert model.lm_head.weight.shape == (cfg.draft_vocab_size, cfg.hidden_size)
        assert model.self_attn.q_proj.weight.shape == (
            cfg.num_attention_heads * cfg.head_dim,
            2 * cfg.hidden_size,
        )
        assert model.self_attn.k_proj.weight.shape == (
            cfg.num_key_value_heads * cfg.head_dim,
            2 * cfg.hidden_size,
        )
        assert model.self_attn.v_proj.weight.shape == (
            cfg.num_key_value_heads * cfg.head_dim,
            2 * cfg.hidden_size,
        )
        assert model.self_attn.o_proj.weight.shape == (
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
        )
        assert model.self_attn.q_norm is None
        assert model.self_attn.k_norm is None


class TestEagle3PreprocessWeights:
    def test_strips_midlayer_and_drops_remaps(self):
        cfg = _eagle3_config()
        model = Eagle3DraftModel(cfg)
        h, dv = cfg.hidden_size, cfg.draft_vocab_size
        state = {
            "fc.weight": torch.zeros(h, 3 * h),
            "lm_head.weight": torch.zeros(dv, h),
            "d2t": torch.zeros(dv, dtype=torch.long),
            "t2d": torch.zeros(151936, dtype=torch.long),
            "midlayer.input_layernorm.weight": torch.zeros(h),
            "midlayer.self_attn.q_proj.weight": torch.zeros(4 * 16, 2 * h),
        }
        out = model.preprocess_weights(state)
        assert out.keys() == {
            "fc.weight",
            "lm_head.weight",
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
        }

    def test_remapped_keys_load_into_module(self):
        cfg = _eagle3_config()
        model = Eagle3DraftModel(cfg)
        param_names = {n for n, _ in model.named_parameters()}
        remapped = model.preprocess_weights(
            {
                "midlayer.hidden_norm.weight": torch.zeros(cfg.hidden_size),
                "norm.weight": torch.zeros(cfg.hidden_size),
            }
        )
        assert set(remapped) == {"hidden_norm.weight", "norm.weight"}
        assert set(remapped).issubset(param_names)

    def test_speculators_layout_strips_layers0_and_drops_embed(self):
        """Strip ``layers.0.`` and drop the borrowed ``embed_tokens`` copy."""
        cfg = _eagle3_config()
        model = Eagle3DraftModel(cfg)
        h, dv = cfg.hidden_size, cfg.draft_vocab_size
        state = {
            "fc.weight": torch.zeros(h, 3 * h),
            "lm_head.weight": torch.zeros(dv, h),
            "norm.weight": torch.zeros(h),
            "embed_tokens.weight": torch.zeros(151936, h),
            "d2t": torch.zeros(dv, dtype=torch.long),
            "t2d": torch.zeros(151936, dtype=torch.bool),
            "layers.0.hidden_norm.weight": torch.zeros(h),
            "layers.0.self_attn.q_proj.weight": torch.zeros(4 * 16, 2 * h),
        }
        out = model.preprocess_weights(state)
        assert out.keys() == {
            "fc.weight",
            "lm_head.weight",
            "norm.weight",
            "hidden_norm.weight",
            "self_attn.q_proj.weight",
        }


class TestEagle3SpeculatorsConfig:
    def test_nested_transformer_layer_config(self):
        """Parse the nested speculators arch config + top-level eagle fields."""
        hf = SimpleNamespace(
            draft_vocab_size=32000,
            norm_before_residual=True,
            target_hidden_size=None,
            transformer_layer_config={
                "model_type": "llama",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "num_hidden_layers": 1,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000,
                "vocab_size": 151936,
            },
        )
        cfg = Eagle3Config.from_transformers(hf)
        assert cfg.draft_vocab_size == 32000
        assert cfg.norm_before_residual is True
        assert cfg.hidden_size == 64
        assert cfg.num_hidden_layers == 1
        assert cfg.rope_theta == 1_000_000

    def test_norm_before_residual_builds(self):
        cfg = _eagle3_config(norm_before_residual=True)
        model = Eagle3DraftModel(cfg)
        assert model._norm_before_residual is True
        pkg = build_from_module(model, cfg, task=Eagle3DraftTask())
        names = [v.name for v in pkg["model"].graph.outputs]
        assert "draft_logits" in names and "next_hidden" in names

    def test_unsupported_options_raise(self):
        with pytest.raises(NotImplementedError):
            Eagle3DraftModel(_eagle3_config(norm_before_fc=True))
        with pytest.raises(NotImplementedError):
            Eagle3DraftModel(_eagle3_config(fc_norm=True))
        with pytest.raises(NotImplementedError):
            Eagle3DraftModel(_eagle3_config(target_hidden_size=128))


class TestEagle3TaskGraph:
    def _build(self, **overrides):
        cfg = _eagle3_config(**overrides)
        module = Eagle3DraftModel(cfg)
        return cfg, build_from_module(module, cfg, task=Eagle3DraftTask())["model"]

    def test_build_returns_valid_model(self):
        _cfg, model = self._build()
        assert isinstance(model, ir.Model)
        assert model.graph.num_nodes() > 0

    def test_inputs(self):
        _cfg, model = self._build()
        names = [v.name for v in model.graph.inputs]
        for name in (
            "inputs_embeds",
            "fused_hidden",
            "recycled_hidden",
            "attention_mask",
            "position_ids",
            "past_key_values.0.key",
            "past_key_values.0.value",
        ):
            assert name in names
        assert "input_ids" not in names

    def test_outputs(self):
        _cfg, model = self._build()
        names = [v.name for v in model.graph.outputs]
        for name in ("draft_logits", "next_hidden", "present.0.key", "present.0.value"):
            assert name in names

    def test_input_output_shapes(self):
        cfg, model = self._build()
        inputs = {v.name: v for v in model.graph.inputs}
        outputs = {v.name: v for v in model.graph.outputs}
        assert inputs["inputs_embeds"].shape[-1] == cfg.hidden_size
        assert inputs["fused_hidden"].shape[-1] == 3 * cfg.hidden_size
        assert inputs["recycled_hidden"].shape[-1] == cfg.hidden_size
        assert outputs["draft_logits"].shape[-1] == cfg.draft_vocab_size
        assert outputs["next_hidden"].shape[-1] == cfg.hidden_size

    def test_task_registered_by_name(self):
        task = get_task("eagle3-draft")
        assert isinstance(task, Eagle3DraftTask)


class TestEagle3ConfigFromTransformers:
    def _hf(self, **overrides):
        config = SimpleNamespace(
            model_type="llama",
            vocab_size=151936,
            draft_vocab_size=32000,
            hidden_size=2560,
            intermediate_size=9728,
            num_hidden_layers=1,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            max_position_embeddings=4096,
            rope_theta=1_000_000.0,
            tie_word_embeddings=False,
            architectures=["Eagle3LlamaForCausalLM"],
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_forces_single_full_attention_layer(self):
        cfg = Eagle3Config.from_transformers(self._hf(num_hidden_layers=64))
        assert cfg.num_hidden_layers == 1
        assert cfg.layer_types == ["full_attention"]

    def test_reads_eagle_fields(self):
        cfg = Eagle3Config.from_transformers(self._hf())
        assert cfg.hidden_size == 2560
        assert cfg.num_attention_heads == 32
        assert cfg.num_key_value_heads == 8
        assert cfg.head_dim == 128
        assert cfg.vocab_size == 151936
        assert cfg.draft_vocab_size == 32000
        assert cfg.rope_theta == pytest.approx(1_000_000.0)


class TestEagle3Registry:
    def test_architecture_routes_to_eagle3(self):
        assert "Eagle3LlamaForCausalLM" in registry
        reg = registry.get_registration("Eagle3LlamaForCausalLM")
        assert reg.module_class is Eagle3DraftModel
        assert reg.task == "eagle3-draft"
        assert reg.config_class is Eagle3Config

    def test_default_task(self):
        assert Eagle3DraftModel.default_task == "eagle3-draft"
        assert Eagle3DraftModel.config_class is Eagle3Config
