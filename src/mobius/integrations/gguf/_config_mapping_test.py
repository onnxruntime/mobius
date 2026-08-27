# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for architecture-specific GGUF config postprocessing."""

from __future__ import annotations

import dataclasses

import pytest

from mobius._registry import registry
from mobius.components import MLP


class _FakeDenseGGUF:
    def __init__(self, architecture: str, metadata: dict, tensor_names: list[str]):
        self.architecture = architecture
        self.metadata = metadata
        self.tensor_names = tensor_names

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)


def _dense_metadata(architecture: str) -> dict:
    return {
        f"{architecture}.embedding_length": 64,
        f"{architecture}.feed_forward_length": 128,
        f"{architecture}.block_count": 2,
        f"{architecture}.attention.head_count": 4,
        f"{architecture}.attention.head_count_kv": 2,
        f"{architecture}.context_length": 512,
        f"{architecture}.rope.freq_base": 10_000.0,
        f"{architecture}.rope.dimension_count": 16,
        f"{architecture}.vocab_size": 256,
    }


def _t5_metadata(architecture: str) -> dict:
    return {
        f"{architecture}.context_length": 512,
        f"{architecture}.embedding_length": 64,
        f"{architecture}.feed_forward_length": 128,
        f"{architecture}.block_count": 2,
        f"{architecture}.attention.head_count": 4,
        f"{architecture}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{architecture}.attention.relative_buckets_count": 32,
        f"{architecture}.vocab_size": 256,
        "tokenizer.ggml.padding_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
    }


class TestSecondHybridCohortConfig:
    @staticmethod
    def _metadata(architecture: str) -> dict:
        metadata = {
            f"{architecture}.embedding_length": 64,
            f"{architecture}.feed_forward_length": 128,
            f"{architecture}.block_count": 3,
            f"{architecture}.attention.head_count": 4,
            f"{architecture}.attention.head_count_kv": [0, 2, 0],
            f"{architecture}.attention.layer_norm_rms_epsilon": 1e-5,
            f"{architecture}.context_length": 512,
            f"{architecture}.vocab_size": 256,
            f"{architecture}.ssm.conv_kernel": 4,
            f"{architecture}.ssm.inner_size": 128,
            f"{architecture}.ssm.state_size": 8,
            f"{architecture}.ssm.time_step_rank": 8,
        }
        if architecture in {"nemotron_h", "nemotron_h_moe", "granitehybrid"}:
            metadata[f"{architecture}.ssm.group_count"] = 2
        if architecture in {"nemotron_h", "nemotron_h_moe"}:
            metadata[f"{architecture}.feed_forward_length"] = [0, 128, 0]
        return metadata

    @pytest.mark.parametrize(
        ("architecture", "config_type", "expected"),
        [
            ("jamba", "JambaConfig", ["mamba", "full_attention", "mamba"]),
            ("nemotron_h", "NemotronHConfig", ["mamba2", "mlp", "mamba2"]),
            (
                "granitehybrid",
                "GraniteMoeHybridConfig",
                ["mamba2", "full_attention", "mamba2"],
            ),
        ],
    )
    def test_exact_serialized_schedule(self, architecture, config_type, expected) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        config = gguf_to_config(
            _FakeDenseGGUF(architecture, self._metadata(architecture), ["token_embd.weight"])
        )
        assert type(config).__name__ == config_type
        assert config.layer_types == expected
        if architecture in {"nemotron_h", "granitehybrid"}:
            assert config.mamba_conv_bias is False

    def test_jamba_nope_config_builds_hybrid_graph(self) -> None:
        import torch

        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
        from mobius.models import JambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = gguf_to_config(
            _FakeDenseGGUF(
                "jamba",
                self._metadata("jamba"),
                [
                    "token_embd.weight",
                    "output.weight",
                    "blk.0.ssm_conv1d.bias",
                ],
            )
        )
        assert config.rope_type is None
        module = JambaCausalLMModel(config)
        package = HybridCausalLMTask().build(module, config)
        assert "model" in package

        mapped = {
            map_gguf_to_hf_names(name, "jamba"): torch.ones(1)
            for name in (
                "blk.0.ssm_dt_norm.weight",
                "blk.0.ssm_b_norm.weight",
                "blk.0.ssm_c_norm.weight",
            )
        }
        processed = module.preprocess_weights(mapped)
        expected = {
            "model.layers.0.mamba.ssm.dt_layernorm.weight",
            "model.layers.0.mamba.ssm.b_layernorm.weight",
            "model.layers.0.mamba.ssm.c_layernorm.weight",
        }
        assert set(processed) == expected
        assert expected <= set(package["model"].graph.initializers)

    @pytest.mark.parametrize("architecture", ["jamba", "nemotron_h", "granitehybrid"])
    def test_wrong_schedule_length_rejects(self, architecture: str) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata(architecture)
        metadata[f"{architecture}.attention.head_count_kv"] = [0, 2]
        with pytest.raises(ValueError, match=r"exactly 3|each contain exactly 3"):
            gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["token_embd.weight"]))

    def test_granitehybrid_moe_extracts_exact_geometry_and_scaling(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("granitehybrid")
        metadata.update(
            {
                "granitehybrid.expert_count": 4,
                "granitehybrid.expert_used_count": 2,
                "granitehybrid.expert_shared_feed_forward_length": 96,
                "granitehybrid.embedding_scale": 12.0,
                "granitehybrid.residual_scale": 0.5,
                "granitehybrid.attention.scale": 0.125,
                "granitehybrid.logit_scale": 16.0,
                "granitehybrid.rope.scaling.finetuned": False,
            }
        )
        config = gguf_to_config(
            _FakeDenseGGUF(
                "granitehybrid",
                metadata,
                [
                    "token_embd.weight",
                    "blk.1.attn_q.bias",
                    "blk.1.attn_output.bias",
                ],
            )
        )
        assert config.layer_types == ["mamba2", "full_attention", "mamba2"]
        assert config.num_local_experts == 4
        assert config.num_experts_per_tok == 2
        assert config.intermediate_size == 128
        assert config.shared_intermediate_size == 96
        assert config.norm_topk_prob is True
        assert config.routed_scaling_factor == pytest.approx(1.0)
        assert config.embedding_multiplier == pytest.approx(12.0)
        assert config.residual_multiplier == pytest.approx(0.5)
        assert config.attention_multiplier == pytest.approx(0.125)
        assert config.logits_scaling == pytest.approx(16.0)
        assert config.rope_type is None
        assert config.attn_qkv_bias is True
        assert config.attn_o_bias is True

    @pytest.mark.parametrize(
        ("updates", "match"),
        [
            ({"expert_count": 4}, "both be zero or both positive"),
            ({"expert_count": 1, "expert_used_count": 1}, "not a routed-MoE"),
            ({"expert_count": 4, "expert_used_count": 5}, "must be in"),
            (
                {
                    "expert_count": 4,
                    "expert_used_count": 2,
                    "expert_weights_norm": False,
                },
                "normalized top-k",
            ),
            (
                {
                    "expert_count": 4,
                    "expert_used_count": 2,
                    "expert_weights_scale": 2.0,
                },
                "does not define",
            ),
            ({"logit_scale": 0.0}, "must be nonzero"),
            ({"feed_forward_length": 0}, "must be greater than zero"),
        ],
    )
    def test_granitehybrid_invalid_moe_config_rejects(self, updates, match) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("granitehybrid")
        metadata.update({f"granitehybrid.{key}": value for key, value in updates.items()})
        with pytest.raises(ValueError, match=match):
            gguf_to_config(_FakeDenseGGUF("granitehybrid", metadata, ["token_embd.weight"]))

    def test_nemotron_h_moe_derives_exact_routed_schedule_and_defaults(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("nemotron_h_moe")
        metadata.update(
            {
                "nemotron_h_moe.expert_count": 4,
                "nemotron_h_moe.expert_used_count": 2,
                "nemotron_h_moe.expert_feed_forward_length": 96,
                "nemotron_h_moe.expert_shared_count": 1,
                "nemotron_h_moe.expert_shared_feed_forward_length": 192,
                "nemotron_h_moe.expert_weights_norm": True,
                "nemotron_h_moe.expert_weights_scale": 2.5,
                "nemotron_h_moe.moe_latent_size": 32,
            }
        )
        config = gguf_to_config(
            _FakeDenseGGUF(
                "nemotron_h_moe",
                metadata,
                [
                    "token_embd.weight",
                    "blk.1.ffn_gate_inp.weight",
                    "blk.1.ffn_latent_down.weight",
                    "blk.1.ffn_latent_up.weight",
                ],
            )
        )
        assert config.layer_types == ["mamba2", "moe", "mamba2"]
        assert config.num_local_experts == 4
        assert config.num_experts_per_tok == 2
        assert config.moe_intermediate_size == 96
        assert config.shared_expert_intermediate_size == 192
        assert config.moe_latent_size == 32
        assert config.norm_topk_prob is True
        assert config.routed_scaling_factor == pytest.approx(2.5)

    def test_nemotron_h_moe_rejects_grouped_expert_routing(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("nemotron_h_moe")
        metadata.update(
            {
                "nemotron_h_moe.expert_count": 4,
                "nemotron_h_moe.expert_used_count": 2,
                "nemotron_h_moe.expert_feed_forward_length": 96,
                "nemotron_h_moe.expert_shared_count": 1,
                "nemotron_h_moe.expert_shared_feed_forward_length": 192,
                "nemotron_h_moe.expert_group_count": 2,
                "nemotron_h_moe.expert_group_used_count": 1,
            }
        )
        with pytest.raises(ValueError, match="grouped expert routing is unsupported"):
            gguf_to_config(
                _FakeDenseGGUF(
                    "nemotron_h_moe",
                    metadata,
                    ["token_embd.weight", "blk.1.ffn_gate_inp.weight"],
                )
            )

    def test_dense_nemotron_h_rejects_moe_metadata(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("nemotron_h")
        metadata["nemotron_h.expert_count"] = 4
        metadata["nemotron_h.expert_used_count"] = 2
        with pytest.raises(ValueError, match="nemotron_h_moe"):
            gguf_to_config(_FakeDenseGGUF("nemotron_h", metadata, ["token_embd.weight"]))

    def test_jamba_derives_exact_routed_layer_schedule(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("jamba")
        metadata["jamba.expert_count"] = 4
        metadata["jamba.expert_used_count"] = 2
        config = gguf_to_config(
            _FakeDenseGGUF(
                "jamba",
                metadata,
                [
                    "token_embd.weight",
                    "blk.2.ffn_gate_inp.weight",
                    "blk.2.ffn_gate_exps.weight",
                    "blk.2.ffn_up_exps.weight",
                    "blk.2.ffn_down_exps.weight",
                ],
            )
        )
        assert config.expert_layer_indices == [2]
        assert config.num_local_experts == 4
        assert config.num_experts_per_tok == 2
        assert config.norm_topk_prob is False


def _diffusion_names(architecture: str, *, output: bool = True) -> list[str]:
    names = ["token_embd.weight", "output_norm.weight"]
    if output:
        names.append("output.weight")
    for layer in range(2):
        names.extend(
            [
                f"blk.{layer}.attn_norm.weight",
                f"blk.{layer}.attn_q.weight",
                f"blk.{layer}.attn_k.weight",
                f"blk.{layer}.attn_v.weight",
                f"blk.{layer}.attn_output.weight",
                f"blk.{layer}.ffn_norm.weight",
            ]
        )
        if architecture in {"llada-moe", "rnd1"}:
            names.extend(
                [
                    f"blk.{layer}.attn_q_norm.weight",
                    f"blk.{layer}.attn_k_norm.weight",
                    f"blk.{layer}.ffn_gate_inp.weight",
                    f"blk.{layer}.ffn_gate_exps.weight",
                    f"blk.{layer}.ffn_down_exps.weight",
                    f"blk.{layer}.ffn_up_exps.weight",
                ]
            )
        else:
            names.extend(
                [
                    f"blk.{layer}.ffn_gate.weight",
                    f"blk.{layer}.ffn_down.weight",
                    f"blk.{layer}.ffn_up.weight",
                ]
            )
    return names


def _diffusion_metadata(architecture: str) -> dict:
    metadata = _dense_metadata(architecture)
    metadata.update(
        {
            f"{architecture}.attention.layer_norm_rms_epsilon": 1e-5,
            f"{architecture}.attention.causal": True,
            "tokenizer.ggml.mask_token_id": 255,
        }
    )
    if architecture in {"llada-moe", "rnd1"}:
        metadata.update(
            {
                f"{architecture}.expert_count": 8,
                f"{architecture}.expert_used_count": 2,
            }
        )
    if architecture in {"llada", "llada-moe"}:
        metadata["diffusion.shift_logits"] = False
    return metadata


class TestLanguageDiffusionConfig:
    @pytest.mark.parametrize(
        ("architecture", "output", "normalized", "shifted"),
        [
            ("dream", False, None, True),
            ("llada", True, None, False),
            ("llada-moe", True, False, False),
            ("rnd1", False, True, True),
        ],
    )
    def test_pinned_defaults_and_architecture_differences(
        self,
        architecture: str,
        output: bool,
        normalized: bool | None,
        shifted: bool,
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        config = gguf_to_config(
            _FakeDenseGGUF(
                architecture,
                _diffusion_metadata(architecture),
                _diffusion_names(architecture, output=output),
            )
        )

        assert config.model_type == ("dream" if architecture == "dream" else "llada")
        assert config.tie_word_embeddings is not output
        assert config.hidden_act == "silu"
        assert config.rope_type == "default"
        assert config.mask_token_id == 255
        assert config.diffusion_shift_logits is shifted
        if normalized is not None:
            assert config.attn_qk_norm is True
            assert config.moe_intermediate_size == 64
            assert config.norm_topk_prob is normalized

    def test_dream_qkv_bias_and_fused_alternative(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        names = _diffusion_names("dream")
        for layer in range(2):
            for projection in ("q", "k", "v"):
                names.append(f"blk.{layer}.attn_{projection}.bias")
        separate = gguf_to_config(_FakeDenseGGUF("dream", _diffusion_metadata("dream"), names))
        assert separate.attn_qkv_bias is True

        fused_names = [
            name
            for name in _diffusion_names("dream")
            if not any(f".attn_{projection}.weight" in name for projection in ("q", "k", "v"))
        ]
        for layer in range(2):
            fused_names.extend([f"blk.{layer}.attn_qkv.weight", f"blk.{layer}.attn_qkv.bias"])
        fused = gguf_to_config(
            _FakeDenseGGUF("dream", _diffusion_metadata("dream"), fused_names)
        )
        assert fused.attn_qkv_bias is True

    def test_missing_mask_token_fails_closed(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _diffusion_metadata("llada")
        metadata.pop("tokenizer.ggml.mask_token_id")
        with pytest.raises(ValueError, match="mask_token_id is required"):
            gguf_to_config(_FakeDenseGGUF("llada", metadata, _diffusion_names("llada")))

    def test_partial_qkv_bias_is_rejected(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        names = [*_diffusion_names("dream"), "blk.0.attn_q.bias"]
        with pytest.raises(ValueError, match="bias must be present consistently"):
            gguf_to_config(_FakeDenseGGUF("dream", _diffusion_metadata("dream"), names))

    def test_llada_moe_requires_untied_output(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        with pytest.raises(ValueError, match=r"requires output\.weight"):
            gguf_to_config(
                _FakeDenseGGUF(
                    "llada-moe",
                    _diffusion_metadata("llada-moe"),
                    _diffusion_names("llada-moe", output=False),
                )
            )


class TestT5Config:
    @pytest.mark.parametrize(
        ("architecture", "model_type"),
        [("t5", "t5"), ("t5encoder", "t5encoder")],
    )
    def test_pinned_defaults_and_token_metadata(
        self, architecture: str, model_type: str
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        names = ["enc.blk.0.attn_rel_b.weight"]
        if architecture == "t5":
            names.append("dec.blk.0.attn_rel_b.weight")
        config = gguf_to_config(
            _FakeDenseGGUF(architecture, _t5_metadata(architecture), names)
        )

        assert config.model_type == model_type
        assert config.head_dim == 16
        assert config.num_key_value_heads == 4
        assert config.num_decoder_layers == (2 if architecture == "t5" else None)
        assert config.decoder_start_token_id is None
        assert config.relative_attention_num_buckets == 32
        assert config.relative_attention_max_distance == 128
        assert config.rms_norm_eps == pytest.approx(1e-6)
        assert config.hidden_act == "relu"
        assert config.is_gated_act is False
        assert config.pad_token_id == 0
        assert config.eos_token_id == 1

    def test_unequal_decoder_count_and_start_token_are_preserved(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _t5_metadata("t5")
        metadata.update(
            {
                "t5.decoder_block_count": 3,
                "t5.decoder_start_token_id": 0,
                "t5.attention.key_length": 8,
                "t5.attention.value_length": 8,
            }
        )
        names = ["enc.blk.0.attn_rel_b.weight", "dec.blk.0.attn_rel_b.weight"]
        config = gguf_to_config(_FakeDenseGGUF("t5", metadata, names))

        assert config.num_decoder_layers == 3
        assert config.decoder_start_token_id == 0
        assert config.head_dim == 8
        assert config.hidden_act == "relu"
        assert config.is_gated_act is False

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    def test_gated_activation_is_rejected_as_ambiguous(self, architecture: str) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        names = [
            "enc.blk.0.attn_rel_b.weight",
            "enc.blk.0.ffn_gate.weight",
            "enc.blk.1.ffn_gate.weight",
        ]
        if architecture == "t5":
            names.extend(
                [
                    "dec.blk.0.attn_rel_b.weight",
                    "dec.blk.0.ffn_gate.weight",
                    "dec.blk.1.ffn_gate.weight",
                ]
            )
        with pytest.raises(ValueError, match=r"gated FFNs are ambiguous"):
            gguf_to_config(_FakeDenseGGUF(architecture, _t5_metadata(architecture), names))

    @pytest.mark.parametrize(
        ("updates", "names", "message"),
        [
            ({"t5.attention.head_count_kv": 2}, None, "head_count_kv"),
            (
                {
                    "t5.attention.key_length": 8,
                    "t5.attention.value_length": 16,
                },
                None,
                "equal positive",
            ),
            ({}, ["dec.blk.0.attn_rel_b.weight"], "enc.blk.0.attn_rel_b"),
            (
                {},
                [
                    "enc.blk.0.attn_rel_b.weight",
                    "dec.blk.0.attn_rel_b.weight",
                    "enc.blk.0.ffn_gate.weight",
                ],
                "mixes gated and non-gated",
            ),
        ],
    )
    def test_unsupported_variants_are_rejected(
        self, updates: dict, names: list[str] | None, message: str
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _t5_metadata("t5")
        metadata.update(updates)
        if names is None:
            names = [
                "enc.blk.0.attn_rel_b.weight",
                "dec.blk.0.attn_rel_b.weight",
            ]
        with pytest.raises(ValueError, match=message):
            gguf_to_config(_FakeDenseGGUF("t5", metadata, names))


class TestDenseCohortConfig:
    @pytest.mark.parametrize("architecture", ["arcee", "smollm3", "exaone"])
    def test_rmsnorm_dense_configs(self, architecture: str) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata(architecture)
        metadata[f"{architecture}.attention.layer_norm_rms_epsilon"] = 1e-5
        config = gguf_to_config(
            _FakeDenseGGUF(
                architecture,
                metadata,
                ["token_embd.weight", "output.weight", "blk.0.attn_q.weight"],
            )
        )

        assert config.model_type == architecture
        assert config.rms_norm_eps == pytest.approx(1e-5)
        assert config.hidden_act == ("relu2" if architecture == "arcee" else "silu")

    def test_olmo_weight_free_layernorm_config(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata("olmo")
        metadata["olmo.attention.layer_norm_epsilon"] = 1e-5
        config = gguf_to_config(
            _FakeDenseGGUF("olmo", metadata, ["token_embd.weight", "output.weight"])
        )

        assert config.model_type == "olmo"
        assert config.rms_norm_eps == pytest.approx(1e-5)

    def test_olmo_rejects_nonzero_qkv_clamp(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata("olmo")
        metadata["olmo.attention.layer_norm_epsilon"] = 1e-5
        metadata["olmo.attention.clamp_kqv"] = 8.0
        with pytest.raises(ValueError, match="clamp_kqv"):
            gguf_to_config(_FakeDenseGGUF("olmo", metadata, ["token_embd.weight"]))

    def test_olmo2_qk_norm_and_olmo3_pattern_rejection(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata("olmo2")
        metadata["olmo2.attention.layer_norm_rms_epsilon"] = 1e-6
        config = gguf_to_config(
            _FakeDenseGGUF("olmo2", metadata, ["token_embd.weight", "output.weight"])
        )

        assert config.model_type == "olmo2"
        assert config.attn_qk_norm is True
        assert config.attn_qk_norm_full is True

        metadata["olmo2.attention.sliding_window"] = 128
        metadata["olmo2.attention.sliding_window_pattern"] = [True, False]
        with pytest.raises(ValueError, match="OLMo3 semantics"):
            gguf_to_config(
                _FakeDenseGGUF("olmo2", metadata, ["token_embd.weight", "output.weight"])
            )

    def test_cohere2_logit_scale_and_partial_rope(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata("cohere2")
        metadata["cohere2.block_count"] = 4
        metadata["cohere2.attention.layer_norm_epsilon"] = 1e-5
        metadata["cohere2.attention.sliding_window"] = 128
        metadata["cohere2.logit_scale"] = 0.0625
        metadata["cohere2.rope.dimension_count"] = 8
        config = gguf_to_config(
            _FakeDenseGGUF("cohere2", metadata, ["token_embd.weight", "output_norm.weight"])
        )

        assert config.model_type == "cohere2"
        assert config.logit_scale == pytest.approx(0.0625)
        assert config.head_dim == 16
        assert config.partial_rotary_factor == pytest.approx(0.5)
        assert config.rope_interleave is True
        assert config.layer_types == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        assert config.no_rope_layers == [1, 1, 1, 0]


class TestPureRecurrentConfig:
    @staticmethod
    def _metadata(architecture: str) -> dict:
        return {
            f"{architecture}.embedding_length": 64,
            f"{architecture}.feed_forward_length": 0,
            f"{architecture}.block_count": 2,
            f"{architecture}.attention.head_count": 0,
            f"{architecture}.attention.layer_norm_rms_epsilon": 1e-5,
            f"{architecture}.context_length": 1024,
            f"{architecture}.vocab_size": 256,
            f"{architecture}.ssm.conv_kernel": 4,
            f"{architecture}.ssm.inner_size": 128,
            f"{architecture}.ssm.state_size": 8,
            f"{architecture}.ssm.time_step_rank": 8,
        }

    def test_mamba_config_uses_ssm_metadata_not_attention_placeholders(self) -> None:
        from mobius._configs import MambaConfig
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        config = gguf_to_config(
            _FakeDenseGGUF(
                "mamba",
                self._metadata("mamba"),
                ["token_embd.weight"],
            )
        )

        assert isinstance(config, MambaConfig)
        assert config.intermediate_size == 128
        assert config.state_size == 8
        assert config.time_step_rank == 8
        assert config.conv_kernel == 4
        assert config.expand == 2
        assert config.tie_word_embeddings is True

    def test_mamba_optional_dt_b_c_norm_false_is_accepted(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("mamba")
        metadata["mamba.ssm.dt_b_c_rms"] = False
        config = gguf_to_config(_FakeDenseGGUF("mamba", metadata, ["token_embd.weight"]))
        assert config.model_type == "mamba"

    def test_mamba2_config_derives_head_geometry(self) -> None:
        from mobius._configs import Mamba2Config
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("mamba2")
        metadata["mamba2.ssm.group_count"] = 2
        config = gguf_to_config(
            _FakeDenseGGUF(
                "mamba2",
                metadata,
                ["token_embd.weight", "output.weight"],
            )
        )

        assert isinstance(config, Mamba2Config)
        assert config.num_heads == 8
        assert config.head_dim == 16
        assert config.n_groups == 2
        assert config.chunk_size == 256
        assert config.tie_word_embeddings is False

    def test_mamba2_rejects_nonintegral_head_dimension(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("mamba2")
        metadata["mamba2.ssm.group_count"] = 1
        metadata["mamba2.ssm.time_step_rank"] = 7
        with pytest.raises(ValueError, match="must be divisible"):
            gguf_to_config(_FakeDenseGGUF("mamba2", metadata, ["token_embd.weight"]))

    def test_mamba_rejects_unimplemented_dt_b_c_norm_variant(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("mamba")
        metadata["mamba.ssm.dt_b_c_rms"] = True
        with pytest.raises(ValueError, match="extra B/C/dt norms"):
            gguf_to_config(_FakeDenseGGUF("mamba", metadata, ["token_embd.weight"]))

    def test_mamba_optional_dt_b_c_norm_cannot_become_required(self) -> None:
        from mobius.integrations.gguf._arch_registry import get_arch_spec

        assert "ssm.dt_b_c_rms" not in get_arch_spec("mamba").required_metadata

    def test_recurrent_required_metadata_matches_pinned_loader(self) -> None:
        from mobius.integrations.gguf._arch_registry import get_arch_spec

        assert set(get_arch_spec("mamba").required_metadata) == {
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        }
        assert set(get_arch_spec("mamba2").required_metadata) == {
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        }

    def test_absent_optional_dt_b_c_norm_falsifies_required_metadata_mutation(
        self, monkeypatch
    ) -> None:
        from mobius.integrations.gguf import _config_mapping
        from mobius.integrations.gguf._arch_registry import get_arch_spec

        spec = get_arch_spec("mamba")
        mutated = dataclasses.replace(
            spec,
            required_metadata=(*spec.required_metadata, "ssm.dt_b_c_rms"),
        )
        monkeypatch.setattr(_config_mapping, "try_get_arch_spec", lambda architecture: mutated)

        with pytest.raises(ValueError, match=r"ssm\.dt_b_c_rms"):
            _config_mapping.gguf_to_config(
                _FakeDenseGGUF(
                    "mamba",
                    self._metadata("mamba"),
                    ["token_embd.weight"],
                )
            )

    def test_mamba2_rejects_group_count_that_does_not_divide_heads(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("mamba2")
        metadata["mamba2.ssm.group_count"] = 3
        with pytest.raises(ValueError, match="must divide both"):
            gguf_to_config(_FakeDenseGGUF("mamba2", metadata, ["token_embd.weight"]))


class TestConventionalMoEConfig:
    @staticmethod
    def _metadata(architecture: str) -> dict:
        metadata = _dense_metadata(architecture)
        metadata[f"{architecture}.attention.layer_norm_rms_epsilon"] = 1e-5
        metadata[f"{architecture}.expert_count"] = 4
        metadata[f"{architecture}.expert_used_count"] = 2
        return metadata

    @pytest.mark.parametrize("architecture", ["olmoe", "phimoe"])
    def test_fixed_width_moe_config(self, architecture: str) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        config = gguf_to_config(
            _FakeDenseGGUF(
                architecture,
                self._metadata(architecture),
                ["token_embd.weight", "output.weight", "blk.0.ffn_gate_inp.weight"],
            )
        )

        assert config.model_type == architecture
        assert config.num_local_experts == 4
        assert config.num_experts_per_tok == 2
        assert config.routed_scaling_factor == pytest.approx(1.0)
        if architecture == "olmoe":
            assert config.attn_qk_norm is True
            assert config.attn_qk_norm_full is True
            assert config.norm_topk_prob is False

    @pytest.mark.parametrize(
        ("architecture", "norm_topk_prob", "qk_norm"),
        [("qwen2moe", False, False), ("qwen3moe", True, True)],
    )
    def test_qwen_moe_width_fallbacks(
        self, architecture: str, norm_topk_prob: bool, qk_norm: bool
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        config = gguf_to_config(
            _FakeDenseGGUF(
                architecture,
                self._metadata(architecture),
                ["token_embd.weight", "output.weight", "blk.0.ffn_gate_inp.weight"],
            )
        )

        assert config.model_type == (
            "qwen2_moe" if architecture == "qwen2moe" else "qwen3_moe"
        )
        assert config.moe_intermediate_size == 64
        assert config.norm_topk_prob is norm_topk_prob
        assert config.attn_qk_norm is qk_norm
        if architecture == "qwen2moe":
            assert config.shared_expert_intermediate_size == 128

    def test_granitemoe_scaling_and_dense_dispatch(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("granitemoe")
        metadata.update(
            {
                "granitemoe.logit_scale": 16.0,
                "granitemoe.embedding_scale": 12.0,
                "granitemoe.residual_scale": 0.5,
                "granitemoe.attention.scale": 0.125,
                "granitemoe.expert_shared_feed_forward_length": 32,
            }
        )
        moe = gguf_to_config(
            _FakeDenseGGUF(
                "granitemoe",
                metadata,
                ["token_embd.weight", "blk.0.ffn_gate_inp.weight"],
            )
        )
        assert moe.model_type == "granitemoe"
        assert moe.logits_scaling == pytest.approx(16.0)
        assert moe.embedding_multiplier == pytest.approx(12.0)
        assert moe.residual_multiplier == pytest.approx(0.5)
        assert moe.attention_multiplier == pytest.approx(0.125)
        assert moe.shared_expert_intermediate_size == 32

        metadata.pop("granitemoe.expert_count")
        metadata.pop("granitemoe.expert_used_count")
        dense = gguf_to_config(
            _FakeDenseGGUF(
                "granitemoe",
                metadata,
                ["token_embd.weight", "blk.0.ffn_gate.weight"],
            )
        )
        assert dense.model_type == "granite"
        assert dense.num_local_experts is None
        assert dense.num_experts_per_tok is None

    @pytest.mark.parametrize("architecture", ["olmoe", "phimoe", "qwen2moe", "qwen3moe"])
    def test_fixed_moe_rejects_zero_experts(self, architecture: str) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata(architecture)
        metadata[f"{architecture}.expert_count"] = 0
        with pytest.raises(ValueError, match="expert_count"):
            gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["token_embd.weight"]))

    def test_moe_rejects_invalid_top_k(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("qwen3moe")
        metadata["qwen3moe.expert_used_count"] = 5
        with pytest.raises(ValueError, match="expert_used_count"):
            gguf_to_config(_FakeDenseGGUF("qwen3moe", metadata, ["token_embd.weight"]))

    def test_phimoe_reads_tensor_backed_longrope_factors(self) -> None:
        import numpy as np

        from mobius.integrations.gguf._config_mapping import gguf_to_config

        class _PhiGGUF(_FakeDenseGGUF):
            def get_tensor(self, name: str):
                return {
                    "rope_factors_long.weight": np.array([2.0] * 8, dtype=np.float32),
                    "rope_factors_short.weight": np.array([1.0] * 8, dtype=np.float32),
                }[name]

        metadata = self._metadata("phimoe")
        metadata["phimoe.rope.scaling.original_context_length"] = 4096
        config = gguf_to_config(
            _PhiGGUF(
                "phimoe",
                metadata,
                ["rope_factors_long.weight", "rope_factors_short.weight"],
            )
        )

        assert config.rope_type == "longrope"
        assert config.rope_scaling == {
            "long_factor": [2.0] * 8,
            "short_factor": [1.0] * 8,
        }
        assert config.original_max_position_embeddings == 4096


class TestDenseCohortConfigContinued:
    def test_smollm3_reconstructs_fixed_no_rope_schedule(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata("smollm3")
        metadata["smollm3.block_count"] = 8
        metadata["smollm3.attention.layer_norm_rms_epsilon"] = 1e-6
        config = gguf_to_config(
            _FakeDenseGGUF("smollm3", metadata, ["token_embd.weight", "output.weight"])
        )

        assert config.no_rope_layers == [1, 1, 1, 0, 1, 1, 1, 0]

    @pytest.mark.parametrize(
        ("architecture", "required_suffix"),
        [
            ("olmo", "attention.layer_norm_epsilon"),
            ("olmo2", "attention.layer_norm_rms_epsilon"),
            ("cohere2", "logit_scale"),
            ("arcee", "attention.layer_norm_rms_epsilon"),
        ],
    )
    def test_missing_required_metadata_is_rejected(
        self, architecture: str, required_suffix: str
    ) -> None:
        from mobius.integrations.gguf._arch_registry import get_arch_spec
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata(architecture)
        for suffix in get_arch_spec(architecture).required_metadata:
            metadata[f"{architecture}.{suffix}"] = 1
        del metadata[f"{architecture}.{required_suffix}"]

        with pytest.raises(ValueError, match=required_suffix):
            gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["token_embd.weight"]))


class TestConventionalSharedMoeConfig:
    @staticmethod
    def _metadata(architecture: str) -> dict:
        metadata = _dense_metadata(architecture)
        metadata.update(
            {
                f"{architecture}.attention.layer_norm_rms_epsilon": 1e-5,
                f"{architecture}.expert_count": 4,
                f"{architecture}.expert_used_count": 2,
                f"{architecture}.expert_feed_forward_length": 32,
                f"{architecture}.expert_shared_count": 2,
                f"{architecture}.expert_weights_scale": 1.25,
                f"{architecture}.expert_weights_norm": architecture != "deepseek",
                f"{architecture}.leading_dense_block_count": 1,
            }
        )
        if architecture == "dots1":
            metadata["dots1.expert_gating_func"] = 1
        return metadata

    @pytest.mark.parametrize(
        ("architecture", "model_type", "dense_prefix", "qk_norm"),
        [
            ("bailingmoe", "bailing_moe", 0, False),
            ("deepseek", "deepseek", 1, False),
            ("dots1", "dots1", 1, True),
        ],
    )
    def test_exact_shared_moe_contract(
        self, architecture: str, model_type: str, dense_prefix: int, qk_norm: bool
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata(architecture)
        if architecture == "bailingmoe":
            metadata["bailingmoe.leading_dense_block_count"] = 0
        names = ["token_embd.weight", "output.weight", "blk.1.ffn_gate_inp.weight"]
        config = gguf_to_config(_FakeDenseGGUF(architecture, metadata, names))

        assert config.model_type == model_type
        assert config.first_k_dense_replace == dense_prefix
        assert config.moe_intermediate_size == 32
        assert config.n_shared_experts == 2
        assert config.shared_expert_intermediate_size == 64
        assert config.scoring_func == "softmax"
        assert config.topk_method == "greedy"
        assert config.n_group == config.topk_group == 1
        assert config.norm_topk_prob is (architecture != "deepseek")
        assert config.routed_scaling_factor == pytest.approx(1.25)
        assert config.routing_weight_normalization_floor == (
            6.103515625e-5 if architecture == "dots1" else None
        )
        assert config.attn_qk_norm is qk_norm
        assert config.tie_word_embeddings is False

    def test_dots1_sigmoid_correction_bias_is_selection_only(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("dots1")
        metadata["dots1.expert_gating_func"] = 2
        config = gguf_to_config(
            _FakeDenseGGUF(
                "dots1",
                metadata,
                ["output.weight", "blk.1.exp_probs_b.bias"],
            )
        )
        assert config.scoring_func == "sigmoid"
        assert config.use_expert_bias is True

        config_without_bias = gguf_to_config(
            _FakeDenseGGUF("dots1", metadata, ["output.weight"])
        )
        assert config_without_bias.use_expert_bias is False

    @pytest.mark.parametrize("architecture", ["deepseek", "dots1"])
    @pytest.mark.parametrize("expert_metadata", ["absent", "zero"])
    def test_all_dense_schedule_does_not_require_active_experts(
        self, architecture: str, expert_metadata: str
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata(architecture)
        metadata[f"{architecture}.leading_dense_block_count"] = metadata[
            f"{architecture}.block_count"
        ]
        for suffix in (
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "expert_shared_count",
        ):
            key = f"{architecture}.{suffix}"
            if expert_metadata == "absent":
                metadata.pop(key)
            else:
                metadata[key] = 0
        config = gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["output.weight"]))

        assert config.first_k_dense_replace == config.num_hidden_layers
        assert config.num_local_experts is None
        assert config.num_experts_per_tok is None
        assert config.moe_intermediate_size is None
        assert config.n_shared_experts is None
        module = registry.get(config.model_type)(config)
        assert all(isinstance(layer.mlp, MLP) for layer in module.model.layers)

    @pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek", "dots1"])
    @pytest.mark.parametrize("scaling_type", ["linear", "dynamic", "longrope", "unknown"])
    def test_unsupported_rope_scaling_fails_closed(
        self, architecture: str, scaling_type: str
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata(architecture)
        metadata[f"{architecture}.rope.scaling.type"] = scaling_type

        with pytest.raises(ValueError, match="only unscaled and YaRN RoPE are exact"):
            gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["output.weight"]))

    @pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek", "dots1"])
    def test_yarn_rope_scaling_requires_complete_positive_metadata(
        self, architecture: str
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata(architecture)
        if architecture == "bailingmoe":
            metadata["bailingmoe.leading_dense_block_count"] = 0
        metadata[f"{architecture}.rope.scaling.type"] = "yarn"
        metadata[f"{architecture}.rope.scaling.factor"] = 4.0

        with pytest.raises(ValueError, match="missing required metadata"):
            gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["output.weight"]))

        metadata[f"{architecture}.rope.scaling.original_context_length"] = 4096
        config = gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["output.weight"]))
        assert config.rope_type == "yarn"
        assert config.rope_scaling == {
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        }

    @pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek", "dots1"])
    @pytest.mark.parametrize(
        ("suffix", "value"),
        [
            ("factor", 0.0),
            ("factor", float("nan")),
            ("original_context_length", 0),
        ],
    )
    def test_yarn_rope_scaling_rejects_nonpositive_or_nonfinite_values(
        self, architecture: str, suffix: str, value: float
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata(architecture)
        metadata.update(
            {
                f"{architecture}.rope.scaling.type": "yarn",
                f"{architecture}.rope.scaling.factor": 4.0,
                f"{architecture}.rope.scaling.original_context_length": 4096,
                f"{architecture}.rope.scaling.{suffix}": value,
            }
        )

        with pytest.raises(ValueError, match="must be finite and positive"):
            gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["output.weight"]))

    @pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek", "dots1"])
    @pytest.mark.parametrize(
        ("suffix", "value"),
        [
            ("yarn_beta_fast", 16.0),
            ("yarn_beta_slow", 2.0),
            ("attn_factor", 1.1),
        ],
    )
    def test_yarn_rope_scaling_rejects_unsupported_nondefault_parameters(
        self, architecture: str, suffix: str, value: float
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata(architecture)
        metadata.update(
            {
                f"{architecture}.rope.scaling.type": "yarn",
                f"{architecture}.rope.scaling.factor": 4.0,
                f"{architecture}.rope.scaling.original_context_length": 4096,
                f"{architecture}.rope.scaling.{suffix}": value,
            }
        )

        with pytest.raises(
            ValueError, match="must retain the supported pinned loader defaults"
        ):
            gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["output.weight"]))

    @pytest.mark.parametrize("architecture", ["bailingmoe", "dots1"])
    def test_optional_routing_defaults_match_pinned_loader(self, architecture: str) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata(architecture)
        metadata.pop(f"{architecture}.expert_weights_norm")
        metadata[f"{architecture}.expert_weights_scale"] = 0.0
        if architecture == "bailingmoe":
            metadata["bailingmoe.leading_dense_block_count"] = 0
        config = gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["output.weight"]))
        assert config.norm_topk_prob is False
        assert config.routed_scaling_factor == pytest.approx(1.0)

    @pytest.mark.parametrize("route_scale", [-1.0, float("nan"), float("inf")])
    def test_invalid_routing_scale_is_rejected(self, route_scale: float) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("deepseek")
        metadata["deepseek.expert_weights_scale"] = route_scale

        with pytest.raises(ValueError, match="finite positive"):
            gguf_to_config(_FakeDenseGGUF("deepseek", metadata, ["output.weight"]))

    @pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek"])
    def test_fixed_softmax_architectures_reject_sigmoid(self, architecture: str) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata(architecture)
        metadata[f"{architecture}.expert_gating_func"] = 2
        if architecture == "bailingmoe":
            metadata["bailingmoe.leading_dense_block_count"] = 0
        with pytest.raises(ValueError, match="SOFTMAX"):
            gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["output.weight"]))

    def test_inconsistent_shared_width_is_rejected(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("deepseek")
        metadata["deepseek.expert_shared_feed_forward_length"] = 63
        with pytest.raises(ValueError, match="expert_shared_feed_forward_length"):
            gguf_to_config(_FakeDenseGGUF("deepseek", metadata, ["output.weight"]))

    def test_bailingmoe_rejects_dense_prefix_metadata(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        with pytest.raises(ValueError, match="all-layer MoE"):
            gguf_to_config(
                _FakeDenseGGUF("bailingmoe", self._metadata("bailingmoe"), ["output.weight"])
            )

    def test_dots1_requires_explicit_gating_enum(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("dots1")
        del metadata["dots1.expert_gating_func"]
        with pytest.raises(ValueError, match="expert_gating_func"):
            gguf_to_config(_FakeDenseGGUF("dots1", metadata, ["output.weight"]))

    def test_mixed_correction_bias_presence_is_rejected(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = self._metadata("dots1")
        metadata["dots1.block_count"] = 3
        with pytest.raises(ValueError, match="every routed layer or none"):
            gguf_to_config(
                _FakeDenseGGUF(
                    "dots1",
                    metadata,
                    ["output.weight", "blk.1.exp_probs_b.bias"],
                )
            )


class TestGemma3Postprocess:
    """Gemma3 config postprocessing fills fields GGUF omits."""

    def test_defaults_local_rope_and_layer_types(self) -> None:
        from mobius._configs import ArchitectureConfig
        from mobius.integrations.gguf._config_mapping import _gemma3_postprocess

        config = ArchitectureConfig(
            hidden_size=1152,
            num_hidden_layers=26,
            num_attention_heads=4,
            num_key_value_heads=1,
            vocab_size=262144,
            intermediate_size=6912,
        )
        # GGUF carries only the global rope base and the sliding-window size.
        result = _gemma3_postprocess(config, {"gemma3.rope.freq_base": 1_000_000.0})

        assert result.rope_local_base_freq == pytest.approx(10_000.0)
        assert result.layer_types is not None
        assert len(result.layer_types) == 26
        # Every 6th layer (1-indexed) is full attention; the rest sliding.
        assert result.layer_types[5] == "full_attention"
        assert result.layer_types[0] == "sliding_attention"
        assert result.layer_types[11] == "full_attention"

    def test_respects_explicit_gguf_values(self) -> None:
        from mobius._configs import ArchitectureConfig
        from mobius.integrations.gguf._config_mapping import _gemma3_postprocess

        config = ArchitectureConfig(
            hidden_size=1152,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            vocab_size=256,
            intermediate_size=512,
        )
        result = _gemma3_postprocess(
            config,
            {
                "gemma3.rope.local_freq_base": 20_000.0,
                "gemma3.attention.sliding_window_pattern": 2,
            },
        )

        assert result.rope_local_base_freq == pytest.approx(20_000.0)
        # Pattern of 2 → every other layer (1-indexed even) is full attention.
        assert result.layer_types == [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]


class TestGemma4DoubleWideMlp:
    """Gemma4 per-layer feed_forward_length arrays collapse to a scalar base."""

    def _base_config(self, intermediate_size, num_layers=4):
        from mobius._configs import ArchitectureConfig

        return ArchitectureConfig(
            hidden_size=1536,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            num_key_value_heads=1,
            vocab_size=262144,
            intermediate_size=intermediate_size,
        )

    def test_uniform_array_collapses_to_scalar(self) -> None:
        from mobius.integrations.gguf._config_mapping import _gemma4_postprocess

        result = _gemma4_postprocess(self._base_config([8192, 8192, 8192, 8192]), {})

        assert result.intermediate_size == 8192
        assert result.use_double_wide_mlp is False

    def test_double_wide_tail_layers_set_flag(self) -> None:
        from mobius.integrations.gguf._config_mapping import _gemma4_postprocess

        # Last 2 KV-shared layers are double-wide (2x the base).
        result = _gemma4_postprocess(
            self._base_config([6144, 6144, 12288, 12288]),
            {"gemma4.attention.shared_kv_layers": 2},
        )

        assert result.intermediate_size == 6144
        assert result.use_double_wide_mlp is True
        assert result.num_kv_shared_layers == 2

    def test_scalar_intermediate_size_unchanged(self) -> None:
        from mobius.integrations.gguf._config_mapping import _gemma4_postprocess

        result = _gemma4_postprocess(self._base_config(8192), {})

        assert result.intermediate_size == 8192
        assert result.use_double_wide_mlp is False

    def test_mismatched_double_wide_pattern_raises(self) -> None:
        import pytest

        from mobius.integrations.gguf._config_mapping import _gemma4_postprocess

        # Double-wide layers must be the trailing KV-shared layers; a leading
        # wide layer does not match the expected pattern.
        with pytest.raises(ValueError, match="double-wide-MLP pattern"):
            _gemma4_postprocess(
                self._base_config([12288, 6144, 6144, 6144]),
                {"gemma4.attention.shared_kv_layers": 2},
            )


class TestDefaultActivation:
    """Tests for _default_activation()."""

    @pytest.mark.parametrize(
        "model_type",
        ["gemma", "gemma2", "gemma3_text", "gemma4_text"],
    )
    def test_gemma_uses_gelu_pytorch_tanh(self, model_type: str) -> None:
        from mobius.integrations.gguf._config_mapping import _default_activation

        assert _default_activation(model_type) == "gelu_pytorch_tanh"

    @pytest.mark.parametrize("model_type", ["gpt2", "bloom", "starcoder2", "t5"])
    def test_gelu_models(self, model_type: str) -> None:
        from mobius.integrations.gguf._config_mapping import _default_activation

        assert _default_activation(model_type) == "gelu"

    @pytest.mark.parametrize("model_type", ["llama", "qwen2", "mistral"])
    def test_silu_default(self, model_type: str) -> None:
        from mobius.integrations.gguf._config_mapping import _default_activation

        assert _default_activation(model_type) == "silu"


class TestQwen35MtpBlockExclusion:
    """Qwen3.5/3.8 GGUF ``block_count`` includes trailing MTP (nextn) blocks.

    ``gguf_to_config`` must subtract ``nextn_predict_layers`` so the decoder
    builds only the real transformer layers; otherwise it fabricates an extra
    layer whose weights are missing from the GGUF (the ``blk.<n>.nextn.*``
    prediction head is skipped during tensor mapping).
    """

    def _fake_model(self, metadata: dict) -> object:
        class _FakeGGUF:
            architecture = "qwen35"

            def __init__(self, md: dict) -> None:
                self.metadata = md

            def get_metadata(self, key, default=None):
                return self.metadata.get(key, default)

            @property
            def tensor_names(self) -> list[str]:
                return ["output.weight", "blk.0.attn_q.weight"]

        return _FakeGGUF(metadata)

    def _base_metadata(self, block_count: int) -> dict:
        return {
            "qwen35.embedding_length": 5120,
            "qwen35.block_count": block_count,
            "qwen35.attention.head_count": 24,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256,
            "qwen35.feed_forward_length": 17408,
            "qwen35.vocab_size": 248320,
            "qwen35.full_attention_interval": 4,
            "qwen35.rope.dimension_count": 64,
            "qwen35.rope.dimension_sections": [11, 11, 10, 0],
            "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
            "qwen35.ssm.conv_kernel": 4,
            "qwen35.ssm.group_count": 4,
            "qwen35.ssm.inner_size": 4096,
            "qwen35.ssm.state_size": 128,
            "qwen35.ssm.time_step_rank": 32,
        }

    def test_nextn_layers_excluded_from_decoder_count(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        md = self._base_metadata(block_count=65)
        md["qwen35.nextn_predict_layers"] = 1
        config = gguf_to_config(self._fake_model(md))

        assert config.num_hidden_layers == 64
        assert config.layer_types is not None
        assert len(config.layer_types) == 64
        # 3 linear + 1 full pattern (full at every 4th, 1-indexed).
        assert config.layer_types[3] == "full_attention"
        assert config.layer_types[0] == "linear_attention"

    def test_no_nextn_metadata_leaves_count_unchanged(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        md = self._base_metadata(block_count=64)
        config = gguf_to_config(self._fake_model(md))

        assert config.num_hidden_layers == 64

    def test_nextn_not_greater_than_block_count(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        md = self._base_metadata(block_count=1)
        md["qwen35.nextn_predict_layers"] = 1
        with pytest.raises(ValueError, match=r"nextn[_ ]predict[_ ]layers"):
            gguf_to_config(self._fake_model(md))


class TestQwen35RopeInterleave:
    """Qwen3.5 uses interleaved M-RoPE sections with split-half rotation.

    The flat ``rope_interleave`` flag selects GPT-J adjacent-pair rotation and
    must remain false. ``mrope_interleaved`` independently selects Qwen3.5's
    temporal/height/width channel assignment and must be true.
    """

    def _fake_model(self, metadata: dict, architecture: str = "qwen35") -> object:
        class _FakeGGUF:
            def __init__(self, md: dict, arch: str) -> None:
                self.metadata = md
                self.architecture = arch

            def get_metadata(self, key, default=None):
                return self.metadata.get(key, default)

            @property
            def tensor_names(self) -> list[str]:
                return ["output.weight", "blk.0.attn_q.weight"]

        return _FakeGGUF(metadata, architecture)

    def _base_metadata(self) -> dict:
        return {
            "qwen35.embedding_length": 5120,
            "qwen35.block_count": 64,
            "qwen35.attention.head_count": 24,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256,
            "qwen35.feed_forward_length": 17408,
            "qwen35.vocab_size": 248320,
            "qwen35.full_attention_interval": 4,
            "qwen35.rope.dimension_count": 64,
            "qwen35.rope.freq_base": 1e7,
            "qwen35.rope.dimension_sections": [11, 11, 10, 0],
            "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
            "qwen35.ssm.conv_kernel": 4,
            "qwen35.ssm.group_count": 4,
            "qwen35.ssm.inner_size": 4096,
            "qwen35.ssm.state_size": 128,
            "qwen35.ssm.time_step_rank": 32,
        }

    def test_dimension_sections_do_not_force_interleave(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        md = self._base_metadata()
        config = gguf_to_config(self._fake_model(md))

        assert config.rope_interleave is False
        assert config.mrope_interleaved is True

    def test_missing_sections_fail_closed(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        md = self._base_metadata()
        del md["qwen35.rope.dimension_sections"]

        with pytest.raises(ValueError, match=r"rope\.dimension_sections"):
            gguf_to_config(self._fake_model(md))


class TestHybridScheduleExtraction:
    @staticmethod
    def _qwen_metadata(architecture: str = "qwen35") -> dict:
        return {f"{architecture}.block_count": 5}

    def test_explicit_recurrent_schedule_wins_over_interval(self) -> None:
        from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout

        md = self._qwen_metadata()
        md["qwen35.full_attention_interval"] = 2
        md["qwen35.attention.recurrent_layers"] = [True, False, False, True, False]

        layers, schedule, mtp = _derive_hybrid_layout("qwen35", md)

        assert layers == 5
        assert mtp == 0
        assert schedule == [
            "linear_attention",
            "full_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]

    def test_default_interval_is_four_and_mtp_is_full_attention(self) -> None:
        from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout

        md = self._qwen_metadata("qwen3next")
        md["qwen3next.nextn_predict_layers"] = 1

        layers, schedule, mtp = _derive_hybrid_layout("qwen3next", md)

        assert (layers, mtp) == (4, 1)
        assert schedule == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]

    def test_minimax_default_interval_is_eight(self) -> None:
        from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout

        layers, schedule, mtp = _derive_hybrid_layout(
            "minimax-01", {"minimax-01.block_count": 9}
        )

        assert (layers, mtp) == (9, 0)
        assert schedule == ["lightning_attention"] * 7 + [
            "full_attention",
            "lightning_attention",
        ]

    def test_minimax_explicit_schedule_rejects_invalid_interval(self) -> None:
        from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout

        metadata = {
            "minimax-01.block_count": 2,
            "minimax-01.attention.recurrent_layers": [True, False],
            "minimax-01.full_attention_interval": 0,
        }
        with pytest.raises(ValueError, match="must be positive"):
            _derive_hybrid_layout("minimax-01", metadata)

    def test_dotted_nextn_metadata_is_rejected(self) -> None:
        from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout

        md = self._qwen_metadata("qwen3next")
        md["qwen3next.nextn.predict_layers"] = 1

        with pytest.raises(ValueError, match="non-pinned MTP metadata key"):
            _derive_hybrid_layout("qwen3next", md)

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            ({"qwen35.attention.recurrent_layers": [True]}, "exactly 5"),
            ({"qwen35.full_attention_interval": 0}, "must be positive"),
            (
                {
                    "qwen35.nextn_predict_layers": 1,
                    "qwen35.attention.recurrent_layers": [True] * 5,
                },
                "MTP block as recurrent",
            ),
        ],
    )
    def test_malformed_qwen_schedule_is_rejected(self, updates, message) -> None:
        from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout

        md = {**self._qwen_metadata(), **updates}
        with pytest.raises(ValueError, match=message):
            _derive_hybrid_layout("qwen35", md)

    def test_lfm2_schedule_comes_only_from_kv_head_array(self) -> None:
        from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout

        md = {
            "lfm2.block_count": 4,
            "lfm2.attention.head_count_kv": [0, 2, 0, 2],
        }
        _, schedule, _ = _derive_hybrid_layout("lfm2", md)
        assert schedule == ["conv", "full_attention", "conv", "full_attention"]

        md["lfm2.attention.head_count_kv"] = 2
        with pytest.raises(ValueError, match="per-layer array"):
            _derive_hybrid_layout("lfm2", md)


class TestLfm2MoePostprocess:
    """Pinned LFM2MoE routing defaults remain overrideable and fail closed."""

    @staticmethod
    def _base_config():
        from mobius._configs import ArchitectureConfig

        return ArchitectureConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
            intermediate_size=128,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
        )

    @staticmethod
    def _metadata() -> dict[str, int]:
        return {
            "lfm2moe.expert_gating_func": 2,
            "lfm2moe.leading_dense_block_count": 1,
        }

    def test_pinned_invariant_routing_defaults(self) -> None:
        from mobius.integrations.gguf._config_mapping import _lfm2moe_postprocess

        result = _lfm2moe_postprocess(self._base_config(), self._metadata())

        assert result.num_dense_layers == 1
        assert result.norm_topk_prob is True
        assert result.routed_scaling_factor == pytest.approx(1.0)
        assert result.use_expert_bias is True

    def test_explicit_pinned_routing_values_are_accepted(self) -> None:
        from mobius.integrations.gguf._config_mapping import _lfm2moe_postprocess

        metadata: dict[str, object] = {
            **self._metadata(),
            "lfm2moe.expert_weights_norm": True,
            "lfm2moe.expert_weights_scale": 1.0,
        }
        result = _lfm2moe_postprocess(self._base_config(), metadata)

        assert result.norm_topk_prob is True
        assert result.routed_scaling_factor == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("lfm2moe.expert_weights_norm", False),
            ("lfm2moe.expert_weights_scale", 0.75),
        ],
    )
    def test_non_pinned_routing_overrides_are_rejected(self, key: str, value: object) -> None:
        from mobius.integrations.gguf._config_mapping import _lfm2moe_postprocess

        metadata: dict[str, object] = {**self._metadata(), key: value}
        with pytest.raises(ValueError, match=key.split(".")[-1]):
            _lfm2moe_postprocess(self._base_config(), metadata)

    def test_missing_dense_prefix_uses_pinned_loader_default(self) -> None:
        from mobius.integrations.gguf._config_mapping import _lfm2moe_postprocess

        metadata = self._metadata()
        del metadata["lfm2moe.leading_dense_block_count"]

        assert _lfm2moe_postprocess(self._base_config(), metadata).num_dense_layers == 0

    def test_missing_gating_function_remains_fail_closed(self) -> None:
        from mobius.integrations.gguf._config_mapping import _lfm2moe_postprocess

        metadata = self._metadata()
        del metadata["lfm2moe.expert_gating_func"]

        with pytest.raises(KeyError, match="expert_gating_func"):
            _lfm2moe_postprocess(self._base_config(), metadata)

    def test_non_integer_gating_function_names_lfm2moe(self) -> None:
        from mobius.integrations.gguf._config_mapping import _lfm2moe_postprocess

        metadata = self._metadata()
        metadata["lfm2moe.expert_gating_func"] = 2.5

        with pytest.raises(ValueError, match=r"lfm2moe\.expert_gating_func"):
            _lfm2moe_postprocess(self._base_config(), metadata)

    def test_all_dense_prefix_is_valid(self) -> None:
        from mobius.integrations.gguf._config_mapping import _lfm2moe_postprocess

        metadata = self._metadata()
        metadata["lfm2moe.leading_dense_block_count"] = 2

        assert _lfm2moe_postprocess(self._base_config(), metadata).num_dense_layers == 2


class TestMuseGlimmerPostprocess:
    """Muse Glimmer config postprocessing.

    Ground truth is the published Muse-Glimmer-30B text config: a stride-4
    sliding-window pattern, NoPE on the full-attention layers,
    ``qk_scale_factor`` 3.87 and ``output_multiplier`` 0.19611613513818404.
    """

    @staticmethod
    def _base_config():
        from mobius._configs import ArchitectureConfig

        return ArchitectureConfig(
            hidden_size=6656,
            num_hidden_layers=8,
            num_attention_heads=32,
            num_key_value_heads=2,
            vocab_size=202048,
            intermediate_size=19968,
            rope_theta=500000.0,
        )

    @staticmethod
    def _metadata():
        return {
            "muse-glimmer.attention.sliding_window": 2048,
            "muse-glimmer.attention.sliding_window_pattern": 4,
            "muse-glimmer.final_logit_softcapping": 20.0,
            "muse-glimmer.logit_scale": 0.1961161345243454,
        }

    class _FakeModel:
        """Stands in for GGUFModel, exposing only what the mapping reads."""

        def __init__(self, q_norm):
            self._q_norm = q_norm

        def get_tensor(self, name: str):
            import numpy as np

            if name == "blk.0.attn_q_norm.weight":
                if self._q_norm is None:
                    raise KeyError(name)
                return np.asarray(self._q_norm, dtype=np.float32)
            raise KeyError(name)

    def test_layer_types_and_nope_layers_follow_the_stride(self) -> None:
        from mobius.integrations.gguf._config_mapping import _muse_glimmer_postprocess

        result = _muse_glimmer_postprocess(
            self._base_config(),
            self._metadata(),
            self._FakeModel([3.87] * 128),
        )

        assert result.layer_types == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        # Full-attention layers are the NoPE layers.
        assert result.no_rope_layers == [3, 7]
        assert result.layer_rope_theta == [
            500000.0,
            500000.0,
            500000.0,
            0,
            500000.0,
            500000.0,
            500000.0,
            0,
        ]
        assert result.sliding_window == 2048

    def test_scalars_come_from_metadata_and_the_q_norm_tensor(self) -> None:
        from mobius.integrations.gguf._config_mapping import _muse_glimmer_postprocess

        result = _muse_glimmer_postprocess(
            self._base_config(),
            self._metadata(),
            self._FakeModel([3.87] * 128),
        )

        assert result.qk_scale_factor == pytest.approx(3.87)
        assert result.output_multiplier == pytest.approx(0.19611613513818404)
        assert result.final_logit_softcapping == pytest.approx(20.0)
        # GGUF has no key for post_norm_eps; the checkpoint default stands.
        assert result.post_norm_eps == pytest.approx(1e-8)
        assert result.attn_qk_norm is True

    def test_missing_q_norm_falls_back_to_the_default_scale(self) -> None:
        from mobius.integrations.gguf._config_mapping import _muse_glimmer_postprocess

        result = _muse_glimmer_postprocess(
            self._base_config(),
            self._metadata(),
            self._FakeModel(None),
        )

        assert result.qk_scale_factor == pytest.approx(3.87)

    def test_non_constant_q_norm_is_rejected(self) -> None:
        from mobius.integrations.gguf._config_mapping import _muse_glimmer_postprocess

        with pytest.raises(ValueError, match="not a constant vector"):
            _muse_glimmer_postprocess(
                self._base_config(),
                self._metadata(),
                self._FakeModel([3.87] * 127 + [1.0]),
            )

    def test_underscore_spelled_architecture_is_read_the_same_way(self) -> None:
        """Both ``muse-glimmer`` and ``muse_glimmer`` name the same architecture.

        The metadata prefix follows whatever the file calls itself, so the
        postprocessor has to take the prefix from the model rather than assume
        the hyphenated spelling.
        """
        from mobius.integrations.gguf._config_mapping import (
            _ARCH_KEY_MAPS,
            _muse_glimmer_postprocess,
        )

        model = self._FakeModel([3.87] * 128)
        model.architecture = "muse_glimmer"
        metadata = {
            key.replace("muse-glimmer.", "muse_glimmer."): value
            for key, value in self._metadata().items()
        }

        result = _muse_glimmer_postprocess(self._base_config(), metadata, model)

        assert result.sliding_window == 2048
        assert result.no_rope_layers == [3, 7]
        assert result.output_multiplier == pytest.approx(0.19611613513818404)
        # The key map is what turns attention.key_length into head_dim, so it
        # has to answer to both spellings too.
        assert _ARCH_KEY_MAPS["muse_glimmer"] == _ARCH_KEY_MAPS["muse-glimmer"]

    def test_a_missing_sliding_window_pattern_is_rejected(self) -> None:
        """Guessing the stride would yield a different architecture.

        Without it every layer is left sliding and rotated, which loads and
        runs and is not Muse Glimmer. Refusing is the only honest answer.
        """
        from mobius.integrations.gguf._config_mapping import _muse_glimmer_postprocess

        metadata = self._metadata()
        del metadata["muse-glimmer.attention.sliding_window_pattern"]

        with pytest.raises(ValueError, match="sliding_window_pattern"):
            _muse_glimmer_postprocess(
                self._base_config(), metadata, self._FakeModel([3.87] * 128)
            )
