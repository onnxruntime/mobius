# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build tests for the DFlash speculative-decoding draft model.

Asserts graph structure: inputs/outputs declared by
:class:`mobius.tasks.DFlashDraftTask`, parameter wiring on
:class:`mobius.models.DFlashDraftModel`, and end-to-end ONNX construction
with tiny random configs.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest

from mobius._builder import build_from_module
from mobius._configs import DFlashConfig
from mobius._testing import make_config
from mobius.models.dflash import DFlashDraftModel
from mobius.tasks import DFlashDraftTask, get_task


def _dflash_config(**overrides) -> DFlashConfig:
    """Minimal DFlash test config built on top of ``make_config``."""
    base = make_config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    fields.update(
        target_layer_ids=[1, 3, 5],
        block_size=4,
        mask_token_id=99,
        num_target_layers=8,
    )
    fields.update(overrides)
    return DFlashConfig(**fields)


class TestDFlashDraftModelParams:
    def test_drafter_has_fc_and_norms(self):
        cfg = _dflash_config()
        model = DFlashDraftModel(cfg)
        names = [n for n, _ in model.named_parameters()]
        assert any("fc.weight" in n for n in names)
        assert any("hidden_norm.weight" in n for n in names)
        assert any("norm.weight" in n for n in names)
        # The drafter must NOT have its own embed_tokens or lm_head — those
        # are borrowed from the target at inference time.
        assert not any("embed_tokens" in n for n in names)
        assert not any("lm_head" in n for n in names)

    def test_drafter_layer_count(self):
        cfg = _dflash_config(num_hidden_layers=5)
        model = DFlashDraftModel(cfg)
        assert len(model.layers) == 5

    def test_fc_input_size_matches_target_layer_count(self):
        cfg = _dflash_config(target_layer_ids=[0, 4, 8, 12])
        model = DFlashDraftModel(cfg)
        # weight shape is [out_features, in_features]
        out_features, in_features = model.fc.weight.shape
        assert out_features == cfg.hidden_size
        assert in_features == len(cfg.target_layer_ids) * cfg.hidden_size

    def test_missing_target_layer_ids_raises(self):
        cfg = _dflash_config(target_layer_ids=None)
        with pytest.raises(ValueError, match="target_layer_ids"):
            DFlashDraftModel(cfg)

    def test_each_layer_has_cross_attention_projections(self):
        cfg = _dflash_config()
        model = DFlashDraftModel(cfg)
        for layer in model.layers:
            attn = layer.self_attn
            # Standard Q/K/V/O + Qwen3 per-head Q/K norm.
            assert attn.q_proj.weight.shape == (
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
            )
            assert attn.k_proj.weight.shape == (
                cfg.num_key_value_heads * cfg.head_dim,
                cfg.hidden_size,
            )
            assert attn.v_proj.weight.shape == (
                cfg.num_key_value_heads * cfg.head_dim,
                cfg.hidden_size,
            )
            assert attn.o_proj.weight.shape == (
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
            )
            assert attn.q_norm.weight.shape == (cfg.head_dim,)
            assert attn.k_norm.weight.shape == (cfg.head_dim,)


class TestDFlashDraftTaskGraph:
    def _build(self, **overrides):
        cfg = _dflash_config(**overrides)
        module = DFlashDraftModel(cfg)
        return cfg, build_from_module(module, cfg, task=DFlashDraftTask())["model"]

    def test_build_returns_valid_model(self):
        _cfg, model = self._build()
        assert isinstance(model, ir.Model)
        assert model.graph.num_nodes() > 0

    def test_inputs(self):
        cfg, model = self._build()
        names = [v.name for v in model.graph.inputs]
        assert "noise_embedding" in names
        assert "target_hidden" in names
        assert "position_ids" in names
        assert "q_position_ids" in names
        # Padding / attention_mask is intentionally absent for the non-causal
        # drafter — adding it back would be a regression for the speculative
        # decoding loop, which assumes batch=1 with no padding.
        assert "attention_mask" not in names
        for i in range(cfg.num_hidden_layers):
            assert f"past_key_values.{i}.key" in names
            assert f"past_key_values.{i}.value" in names

    def test_outputs(self):
        cfg, model = self._build()
        names = [v.name for v in model.graph.outputs]
        assert "draft_hidden" in names
        assert "logits" not in names  # drafter does NOT produce logits
        for i in range(cfg.num_hidden_layers):
            assert f"present.{i}.key" in names
            assert f"present.{i}.value" in names

    def test_target_hidden_shape_scales_with_target_layer_ids(self):
        cfg, model = self._build(target_layer_ids=[2, 7, 11, 15])
        target_hidden_input = next(
            v for v in model.graph.inputs if v.name == "target_hidden"
        )
        # Last dim must be len(target_layer_ids) * hidden_size.
        last_dim = target_hidden_input.shape[-1]
        assert last_dim == len(cfg.target_layer_ids) * cfg.hidden_size

    def test_task_registered_by_name(self):
        # Confirm "dflash-draft" resolves to a DFlashDraftTask instance, so
        # that build(... task="dflash-draft") works and so the task is
        # discoverable via get_task().
        task = get_task("dflash-draft")
        assert isinstance(task, DFlashDraftTask)


class TestDFlashConfigFromTransformers:
    def test_from_transformers_extracts_dflash_fields(self):
        from transformers import Qwen3Config

        # Build a Qwen3-shaped HF config with the DFlash side-channel fields
        # set the way ``z-lab/Qwen3-4B-DFlash-b16`` ships them.
        hf_cfg = Qwen3Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=100,
            max_position_embeddings=32,
        )
        hf_cfg.dflash_config = {
            "target_layer_ids": [1, 3, 5],
            "mask_token_id": 99,
        }
        hf_cfg.block_size = 8
        hf_cfg.num_target_layers = 12

        mobius_cfg = DFlashConfig.from_transformers(hf_cfg)
        assert isinstance(mobius_cfg, DFlashConfig)
        assert mobius_cfg.target_layer_ids == [1, 3, 5]
        assert mobius_cfg.mask_token_id == 99
        assert mobius_cfg.block_size == 8
        assert mobius_cfg.num_target_layers == 12
        # And the standard architecture fields are still populated:
        assert mobius_cfg.hidden_size == 64
        assert mobius_cfg.num_hidden_layers == 2
