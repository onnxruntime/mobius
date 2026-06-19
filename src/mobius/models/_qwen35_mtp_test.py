# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build tests for the Qwen3.6 MTP self-speculative head.

Asserts graph structure (inputs/outputs declared by
:class:`mobius.tasks.Qwen35MtpTask`), parameter wiring on
:class:`mobius.models.Qwen35MtpModel`, the ``mtp.*`` weight remapping, the
single-full-attention-layer ``from_transformers`` contract, and registry
routing — all with tiny random configs (no checkpoint download).

Like the DFlash drafter, the head borrows the target's shared
``embed_tokens`` / ``lm_head``: it consumes ``inputs_embeds`` and emits
``mtp_hidden`` (no embedding table, no LM head, no ``logits``).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import Qwen35MtpConfig
from mobius._registry import registry
from mobius._testing import make_config
from mobius.models.qwen35_mtp import Qwen35MtpModel
from mobius.tasks import Qwen35MtpTask, get_task


def _mtp_config(**overrides) -> Qwen35MtpConfig:
    """Minimal Qwen3.6 MTP test config built on top of ``make_config``."""
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
        partial_rotary_factor=0.5,
        vocab_size=100,
    )
    fields.update(overrides)
    return Qwen35MtpConfig(**fields)


class TestQwen35MtpModelParams:
    def test_head_has_fc_and_norms(self):
        cfg = _mtp_config()
        model = Qwen35MtpModel(cfg)
        names = [n for n, _ in model.named_parameters()]
        assert any("fc.weight" in n for n in names)
        assert any("pre_fc_norm_embedding.weight" in n for n in names)
        assert any("pre_fc_norm_hidden.weight" in n for n in names)
        assert any("norm.weight" in n for n in names)

    def test_head_borrows_embed_and_lm_head(self):
        """The MTP head must not own an embedding table or LM head.

        Those are borrowed from the target (mtp_use_dedicated_embeddings=False).
        """
        cfg = _mtp_config()
        model = Qwen35MtpModel(cfg)
        names = [n for n, _ in model.named_parameters()]
        assert not any("embed_tokens" in n for n in names)
        assert not any("lm_head" in n for n in names)

    def test_single_full_attention_layer(self):
        cfg = _mtp_config()
        model = Qwen35MtpModel(cfg)
        assert len(model.layers) == 1
        layer = model.layers[0]
        assert hasattr(layer, "self_attn")
        assert not hasattr(layer, "linear_attn")

    def test_fc_input_is_double_hidden(self):
        cfg = _mtp_config()
        model = Qwen35MtpModel(cfg)
        out_features, in_features = model.fc.weight.shape
        assert out_features == cfg.hidden_size
        assert in_features == 2 * cfg.hidden_size

    def test_attention_projection_shapes(self):
        cfg = _mtp_config()
        model = Qwen35MtpModel(cfg)
        attn = model.layers[0].self_attn
        # Qwen3.5 doubles the Q projection (query + output gate).
        assert attn.q_proj.weight.shape == (
            cfg.num_attention_heads * cfg.head_dim * 2,
            cfg.hidden_size,
        )
        assert attn.k_proj.weight.shape == (
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.hidden_size,
        )
        assert attn.q_norm.weight.shape == (cfg.head_dim,)
        assert attn.k_norm.weight.shape == (cfg.head_dim,)


class TestQwen35MtpPreprocessWeights:
    def test_strips_mtp_prefix_and_drops_rest(self):
        cfg = _mtp_config()
        model = Qwen35MtpModel(cfg)
        h, two_h, vocab = cfg.hidden_size, 2 * cfg.hidden_size, cfg.vocab_size
        state = {
            "mtp.fc.weight": torch.zeros(h, two_h),
            "mtp.pre_fc_norm_embedding.weight": torch.zeros(h),
            "mtp.pre_fc_norm_hidden.weight": torch.zeros(h),
            "mtp.norm.weight": torch.zeros(h),
            "mtp.layers.0.input_layernorm.weight": torch.zeros(h),
            # Shared / main-model weights that must be dropped (the head
            # borrows embed + lm_head; it does not own them).
            "model.language_model.embed_tokens.weight": torch.zeros(vocab, h),
            "lm_head.weight": torch.zeros(vocab, h),
            "model.layers.0.input_layernorm.weight": torch.zeros(h),
            "model.norm.weight": torch.zeros(h),
            "model.visual.patch_embed.weight": torch.zeros(4, 4),
        }
        out = model.preprocess_weights(state)
        assert out.keys() == {
            "fc.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
            "norm.weight",
            "layers.0.input_layernorm.weight",
        }

    def test_remapped_keys_load_into_module(self):
        cfg = _mtp_config()
        model = Qwen35MtpModel(cfg)
        # The remapped keys must be a subset of the module's own parameter
        # names so weight application can route them.
        param_names = {n for n, _ in model.named_parameters()}
        remapped = model.preprocess_weights(
            {
                "mtp.fc.weight": torch.zeros(cfg.hidden_size, 2 * cfg.hidden_size),
                "mtp.norm.weight": torch.zeros(cfg.hidden_size),
            }
        )
        assert set(remapped) == {"fc.weight", "norm.weight"}
        assert set(remapped).issubset(param_names)


class TestQwen35MtpTaskGraph:
    def _build(self, **overrides):
        cfg = _mtp_config(**overrides)
        module = Qwen35MtpModel(cfg)
        return cfg, build_from_module(module, cfg, task=Qwen35MtpTask())["model"]

    def test_build_returns_valid_model(self):
        _cfg, model = self._build()
        assert isinstance(model, ir.Model)
        assert model.graph.num_nodes() > 0

    def test_inputs(self):
        _cfg, model = self._build()
        names = [v.name for v in model.graph.inputs]
        assert "inputs_embeds" in names
        assert "hidden_states" in names
        assert "attention_mask" in names
        assert "position_ids" in names
        assert "past_key_values.0.key" in names
        assert "past_key_values.0.value" in names
        # The head borrows the target embedding — no token-id input.
        assert "input_ids" not in names

    def test_outputs(self):
        _cfg, model = self._build()
        names = [v.name for v in model.graph.outputs]
        assert "mtp_hidden" in names
        # No LM head of its own.
        assert "logits" not in names
        assert "present.0.key" in names
        assert "present.0.value" in names

    def test_inputs_embeds_and_hidden_shapes(self):
        cfg, model = self._build()
        for name in ("inputs_embeds", "hidden_states"):
            v = next(v for v in model.graph.inputs if v.name == name)
            assert v.shape[-1] == cfg.hidden_size

    def test_task_registered_by_name(self):
        task = get_task("qwen35-mtp")
        assert isinstance(task, Qwen35MtpTask)


class TestQwen35MtpConfigFromTransformers:
    def _hf(self, **text_overrides):
        text = SimpleNamespace(
            model_type="qwen3_5_text",
            vocab_size=248320,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=64,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=16,
            partial_rotary_factor=0.25,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            max_position_embeddings=4096,
            tie_word_embeddings=False,
            attn_output_gate=True,
            full_attention_interval=4,
            rope_parameters={
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_theta": 1e7,
                "rope_type": "default",
                "partial_rotary_factor": 0.25,
            },
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
        )
        for k, v in text_overrides.items():
            setattr(text, k, v)
        return SimpleNamespace(
            model_type="qwen3_5", text_config=text, tie_word_embeddings=False
        )

    def test_forces_single_full_attention_layer(self):
        cfg = Qwen35MtpConfig.from_transformers(self._hf())
        assert cfg.num_hidden_layers == 1
        assert cfg.layer_types == ["full_attention"]

    def test_reads_text_config_fields(self):
        cfg = Qwen35MtpConfig.from_transformers(self._hf())
        assert cfg.hidden_size == 128
        assert cfg.num_attention_heads == 8
        assert cfg.num_key_value_heads == 2
        assert cfg.head_dim == 16
        assert cfg.vocab_size == 248320
        assert cfg.partial_rotary_factor == pytest.approx(0.25)
        assert cfg.mrope_section == [11, 11, 10]
        assert cfg.mrope_interleaved is True


class TestQwen35MtpRegistry:
    def test_architecture_routes_to_mtp(self):
        assert "Qwen35MtpModel" in registry
        reg = registry.get_registration("Qwen35MtpModel")
        assert reg.module_class is Qwen35MtpModel
        assert reg.task == "qwen35-mtp"

    def test_default_task(self):
        assert Qwen35MtpModel.default_task == "qwen35-mtp"
        assert Qwen35MtpModel.config_class is Qwen35MtpConfig
