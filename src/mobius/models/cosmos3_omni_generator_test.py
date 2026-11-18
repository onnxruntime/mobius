# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the unified Cosmos3-Omni MoT transformer (Reasoner + Generator).

Coverage:

* config parsing of the public ``nvidia/Cosmos3-Nano`` ``transformer/config.json``
  and every shape-relationship validation;
* exact graph I/O contract for the generator-only and sound+action builds;
* initializer-name coverage for all core branches (both experts, both MLP
  variants, both QK-norm variants, the vision/sound/action heads);
* interleaved 3-axis mRoPE channel layout against upstream's index arithmetic;
* ``DomainAwareLinear`` graph semantics and numerics vs a PyTorch reference;
* ``preprocess_weights`` accept/drop/reject behaviour;
* full numerical parity of the packed attention setup (dual-pathway causal +
  non-causal attention, mRoPE, timestep conditioning, MoT experts) against a
  PyTorch transcription of upstream ``Cosmos3OmniTransformer.forward``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as functional
from onnxscript import nn

from mobius import build_from_module
from mobius._configs._cosmos3_omni_generator import Cosmos3OmniGeneratorConfig
from mobius._model_package import ModelPackage
from mobius.models.cosmos3_omni_generator import (
    Cosmos3OmniDomainAwareLinear,
    Cosmos3OmniGeneratorModel,
)
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cosmos3_omni_generator import (
    Cosmos3OmniGeneratorTask,
    expected_input_names,
    expected_output_names,
)

# Verbatim ``nvidia/Cosmos3-Nano`` ``transformer/config.json`` (public).
PUBLISHED_CONFIG: dict = {
    "_class_name": "Cosmos3OmniTransformer",
    "_diffusers_version": "0.37.1",
    "action_dim": 64,
    "action_gen": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "base_fps": 24,
    "dtype": "bfloat16",
    "enable_fps_modulation": True,
    "freeze_und": False,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "joint_attn_implementation": "two_way",
    "latent_channel": 48,
    "latent_patch_size": 2,
    "max_action_dim": 64,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_vl_text",
    "num_attention_heads": 32,
    "num_embodiment_domains": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "patch_latent_dim": 192,
    "position_embedding_type": "unified_3d_mrope",
    "qk_norm": False,
    "qk_norm_for_diffusion": True,
    "qk_norm_for_text": True,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    },
    "rope_theta": 5000000,
    "sound_dim": 64,
    "sound_gen": True,
    "sound_latent_fps": 25,
    "temporal_compression_factor_sound": 1,
    "timestep_scale": 0.001,
    "unified_3d_mrope_reset_spatial_ids": True,
    "unified_3d_mrope_temporal_modality_margin": 15000,
    "use_cache": True,
    "use_moe": True,
    "video_temporal_causal": False,
    "vocab_size": 151936,
}


def tiny_config(**overrides) -> Cosmos3OmniGeneratorConfig:
    """A 2-layer graph-construction config (hidden=32, head_dim=8)."""
    fields = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_act": "silu",
        "dtype": ir.DataType.FLOAT,
        "rope_theta": 5000.0,
        # sum == head_dim // 2 == 4
        "rope_axes_dim": (2, 1, 1),
        "latent_channel": 3,
        "latent_patch_size": 2,
        "patch_latent_dim": 12,
        "time_proj_channels": 8,
    }
    fields.update(overrides)
    return Cosmos3OmniGeneratorConfig(**fields)


def full_config(**overrides) -> Cosmos3OmniGeneratorConfig:
    """The tiny config with both optional heads enabled."""
    fields = {
        "sound_gen": True,
        "sound_dim": 6,
        "action_gen": True,
        "action_dim": 5,
        "num_embodiment_domains": 3,
    }
    fields.update(overrides)
    return tiny_config(**fields)


def build_package(config: Cosmos3OmniGeneratorConfig, **kwargs) -> ModelPackage:
    """Build the generator package for *config*."""
    return build_from_module(
        Cosmos3OmniGeneratorModel(config),
        config,
        task=Cosmos3OmniGeneratorTask(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_from_diffusers_parses_published_config(self):
        config = Cosmos3OmniGeneratorConfig.from_diffusers(PUBLISHED_CONFIG)

        assert config.hidden_size == 4096
        assert config.intermediate_size == 12288
        assert config.num_hidden_layers == 36
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128
        assert config.vocab_size == 151936
        assert config.hidden_act == "silu"
        assert config.dtype == ir.DataType.BFLOAT16
        assert config.rope_theta == pytest.approx(5_000_000.0)
        assert config.rms_norm_eps == pytest.approx(1e-6)
        assert config.attention_bias is False
        assert config.attn_qkv_bias is False
        assert config.attn_o_bias is False
        # Vision head shape relationship: 48 * 2**2 == 192.
        assert config.latent_channel == 48
        assert config.latent_patch_size == 2
        assert config.patch_latent_dim == 192
        assert config.timestep_scale == pytest.approx(0.001)
        # Optional heads.
        assert config.sound_gen is True
        assert config.sound_dim == 64
        assert config.action_gen is True
        assert config.action_dim == 64
        assert config.max_action_dim == 64
        assert config.num_embodiment_domains == 32
        # Host-side fields are parsed, not dropped.
        assert config.base_fps == 24
        assert config.enable_fps_modulation is True
        assert config.unified_3d_mrope_reset_spatial_ids is True
        assert config.unified_3d_mrope_temporal_modality_margin == 15000
        assert config.max_position_embeddings == 262144
        assert config.sound_latent_fps == pytest.approx(25.0)
        assert config.temporal_compression_factor_sound == 1

    def test_from_diffusers_derives_rope_axes_from_mrope_section(self):
        config = Cosmos3OmniGeneratorConfig.from_diffusers(PUBLISHED_CONFIG)
        assert config.rope_axes_dim == (24, 20, 20)
        # mRoPE channel budget must exactly fill the rotary dimension.
        assert sum(config.rope_axes_dim) == config.rotary_dim == 64

    def test_from_diffusers_defaults_rope_axes_without_rope_scaling(self):
        raw = dict(PUBLISHED_CONFIG)
        raw.pop("rope_scaling")
        assert Cosmos3OmniGeneratorConfig.from_diffusers(raw).rope_axes_dim == (24, 20, 20)

    def test_from_diffusers_rejects_unknown_dtype(self):
        raw = dict(PUBLISHED_CONFIG, dtype="int8")
        with pytest.raises(ValueError, match="Unsupported Cosmos3-Omni dtype"):
            Cosmos3OmniGeneratorConfig.from_diffusers(raw)

    def test_derived_properties(self):
        config = Cosmos3OmniGeneratorConfig.from_diffusers(PUBLISHED_CONFIG)
        assert config.rotary_dim == 64
        assert config.num_key_value_groups == 4
        assert config.attention_out_size == 32 * 128
        assert config.key_value_size == 8 * 128
        assert config.is_gated_mlp is True
        # ``use_und_k_norm_for_gen`` is inert while the und pathway has QK norm.
        assert config.has_und_k_norm_for_gen is False

    def test_has_und_k_norm_for_gen_requires_qk_norm_off(self):
        assert tiny_config(use_und_k_norm_for_gen=True).has_und_k_norm_for_gen is False
        assert (
            tiny_config(use_und_k_norm_for_gen=True, qk_norm_for_text=False)
        ).has_und_k_norm_for_gen is True

    def test_relu2_backbone_is_not_gated(self):
        assert tiny_config(hidden_act="relu2").is_gated_mlp is False

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"hidden_size": 0}, "hidden_size must be a positive integer"),
            ({"num_hidden_layers": -1}, "num_hidden_layers must be a positive integer"),
            ({"num_attention_heads": 3}, "divisible by num_key_value_heads"),
            ({"head_dim": 7, "rope_axes_dim": (1, 1, 1)}, "head_dim must be even"),
            ({"hidden_act": "gelu"}, "hidden_act must be one of"),
            ({"rms_norm_eps": 0.0}, "rms_norm_eps must be positive"),
            ({"dtype": ir.DataType.INT8}, "dtype must be one of"),
            ({"attention_dropout": 0.1}, "attention_dropout must be 0.0"),
            ({"rope_axes_dim": (2, 2)}, "exactly 3"),
            ({"rope_axes_dim": (2, 1, 0)}, "positive integers"),
            ({"rope_axes_dim": (2, 2, 2)}, "must equal head_dim // 2"),
            ({"rope_theta": 0.0}, "rope_theta must be positive"),
            ({"max_position_embeddings": 0}, "max_position_embeddings must be positive"),
            ({"latent_channel": 0}, "latent_channel must be positive"),
            ({"latent_patch_size": 0}, "latent_patch_size must be positive"),
            ({"patch_latent_dim": 13}, "latent_channel \\* latent_patch_size"),
            ({"timestep_scale": 0.0}, "timestep_scale must be positive"),
            ({"time_proj_channels": 7}, "positive even number"),
            ({"sound_gen": True}, "sound_dim must be a positive integer"),
            ({"sound_gen": True, "sound_dim": 4, "sound_latent_fps": 0}, "sound_latent_fps"),
            (
                {"sound_gen": True, "sound_dim": 4, "temporal_compression_factor_sound": 0},
                "temporal_compression_factor_sound",
            ),
            ({"action_gen": True}, "action_dim must be a positive integer"),
            (
                {"action_gen": True, "action_dim": 4, "num_embodiment_domains": 0},
                "num_embodiment_domains must be positive",
            ),
            (
                {"action_gen": True, "action_dim": 8, "max_action_dim": 4},
                "max_action_dim",
            ),
            ({"position_embedding_type": "rope"}, "unified_3d_mrope"),
            ({"joint_attn_implementation": "one_way"}, "two_way"),
            ({"use_moe": False}, "use_moe=False"),
            ({"video_temporal_causal": True}, "video_temporal_causal=True"),
            ({"qk_norm_for_diffusion": False}, "qk_norm_for_diffusion=False"),
        ],
    )
    def test_validate_rejects(self, overrides, match):
        config = tiny_config(**overrides)
        with pytest.raises(ValueError, match=match):
            config.validate()

    def test_validate_accepts_published_and_tiny_configs(self):
        Cosmos3OmniGeneratorConfig.from_diffusers(PUBLISHED_CONFIG).validate()
        tiny_config().validate()
        full_config().validate()
        tiny_config(hidden_act="relu2").validate()


# ---------------------------------------------------------------------------
# Graph I/O contract
# ---------------------------------------------------------------------------


class TestGraphContract:
    def test_generator_only_io(self):
        config = tiny_config()
        model = build_package(config)["model"]

        assert model.graph.name == "cosmos3_omni_generator"
        assert [value.name for value in model.graph.inputs] == list(
            expected_input_names(config)
        )
        assert [value.name for value in model.graph.outputs] == ["vision_pred"]

        shapes = {value.name: list(value.shape) for value in model.graph.inputs}
        dtypes = {value.name: value.dtype for value in model.graph.inputs}
        assert shapes["input_ids"] == [ir.SymbolicDim("num_text_tokens")]
        assert shapes["text_indexes"] == [ir.SymbolicDim("num_text_tokens")]
        assert shapes["position_ids"] == [3, ir.SymbolicDim("sequence_length")]
        assert shapes["und_len"] == [1]
        assert shapes["vision_tokens"] == [
            ir.SymbolicDim("num_vision_tokens"),
            config.patch_latent_dim,
        ]
        assert shapes["vision_mse_loss_indexes"] == [ir.SymbolicDim("num_vision_noisy_tokens")]
        assert dtypes["vision_tokens"] == ir.DataType.FLOAT
        # Timesteps are float32 regardless of model dtype.
        assert dtypes["vision_timesteps"] == ir.DataType.FLOAT
        for name in (
            "input_ids",
            "text_indexes",
            "position_ids",
            "und_len",
            "vision_sequence_indexes",
            "vision_timestep_token_indexes",
            "vision_mse_loss_indexes",
        ):
            assert dtypes[name] == ir.DataType.INT64, name

        output = model.graph.outputs[0]
        assert list(output.shape) == [
            ir.SymbolicDim("num_vision_noisy_tokens"),
            config.patch_latent_dim,
        ]
        assert output.dtype == ir.DataType.FLOAT

    def test_sound_and_action_io(self):
        config = full_config()
        model = build_package(config)["model"]

        assert [value.name for value in model.graph.inputs] == list(
            expected_input_names(config)
        )
        assert [value.name for value in model.graph.outputs] == list(
            expected_output_names(config)
        )
        shapes = {value.name: list(value.shape) for value in model.graph.inputs}
        assert shapes["sound_tokens"] == [
            ir.SymbolicDim("num_sound_tokens"),
            config.sound_dim,
        ]
        assert shapes["action_tokens"] == [
            ir.SymbolicDim("num_action_tokens"),
            config.action_dim,
        ]
        assert shapes["action_domain_ids"] == [ir.SymbolicDim("num_action_tokens")]
        assert shapes["action_pred_domain_ids"] == [ir.SymbolicDim("num_action_noisy_tokens")]
        outputs = {value.name: list(value.shape) for value in model.graph.outputs}
        assert outputs["sound_pred"] == [
            ir.SymbolicDim("num_sound_noisy_tokens"),
            config.sound_dim,
        ]
        assert outputs["action_pred"] == [
            ir.SymbolicDim("num_action_noisy_tokens"),
            config.action_dim,
        ]

    @pytest.mark.parametrize(
        ("overrides", "absent"),
        [
            ({}, ("sound", "action")),
            ({"sound_gen": True, "sound_dim": 6}, ("action",)),
            ({"action_gen": True, "action_dim": 5, "num_embodiment_domains": 3}, ("sound",)),
        ],
    )
    def test_disabled_heads_declare_no_inputs_or_outputs(self, overrides, absent):
        config = tiny_config(**overrides)
        model = build_package(config)["model"]
        names = [value.name for value in model.graph.inputs]
        names += [value.name for value in model.graph.outputs]
        for prefix in absent:
            assert not [name for name in names if name.startswith(prefix)]

    def test_bfloat16_keeps_timestep_path_in_float32(self):
        config = full_config(dtype=ir.DataType.BFLOAT16)
        model = build_package(config)["model"]
        initializers = model.graph.initializers

        # Model weights are bf16 ...
        assert initializers["proj_in.weight"].dtype == ir.DataType.BFLOAT16
        assert initializers["layers.0.self_attn.to_q.weight"].dtype == ir.DataType.BFLOAT16
        # ... but the timestep MLP and the rotary frequencies stay fp32.
        for name in (
            "time_embedder.linear_1.weight",
            "time_embedder.linear_1.bias",
            "time_embedder.linear_2.weight",
            "time_embedder.linear_2.bias",
            "time_proj.inv_freq",
            "rotary_emb.inv_freq",
        ):
            assert initializers[name].dtype == ir.DataType.FLOAT, name

        dtypes = {value.name: value.dtype for value in model.graph.inputs}
        assert dtypes["vision_tokens"] == ir.DataType.BFLOAT16
        assert dtypes["sound_tokens"] == ir.DataType.BFLOAT16
        assert dtypes["action_tokens"] == ir.DataType.BFLOAT16
        assert dtypes["vision_timesteps"] == ir.DataType.FLOAT
        assert dtypes["sound_timesteps"] == ir.DataType.FLOAT
        assert dtypes["action_timesteps"] == ir.DataType.FLOAT
        assert all(value.dtype == ir.DataType.BFLOAT16 for value in model.graph.outputs)

    def test_task_rejects_wrong_module_output_contract(self):
        class BadModule(nn.Module):
            def forward(self, op, **kwargs):
                return op.Identity(kwargs["vision_tokens"])

        with pytest.raises(TypeError, match="must return"):
            Cosmos3OmniGeneratorTask().build(BadModule(), tiny_config())

    def test_task_rejects_missing_configured_head_output(self):
        class NoSoundModule(nn.Module):
            def forward(self, op, **kwargs):
                return op.Identity(kwargs["vision_tokens"]), None, None

        config = tiny_config(sound_gen=True, sound_dim=6)
        with pytest.raises(TypeError, match="sound_pred=None"):
            Cosmos3OmniGeneratorTask().build(NoSoundModule(), config)

    def test_module_requires_all_inputs_of_a_configured_head(self):
        config = tiny_config(sound_gen=True, sound_dim=6)
        module = Cosmos3OmniGeneratorModel(config)
        _, builder = _make_graph()
        placeholder = builder.input("x", dtype=ir.DataType.INT64, shape=[1])
        with pytest.raises(ValueError, match="sound_gen=True requires all sound inputs"):
            module(
                builder.op,
                input_ids=placeholder,
                text_indexes=placeholder,
                position_ids=placeholder,
                und_len=placeholder,
                vision_tokens=placeholder,
                vision_sequence_indexes=placeholder,
                vision_timesteps=placeholder,
                vision_timestep_token_indexes=placeholder,
                vision_mse_loss_indexes=placeholder,
            )

    def test_module_rejects_inputs_for_a_disabled_head(self):
        module = Cosmos3OmniGeneratorModel(tiny_config())
        _, builder = _make_graph()
        placeholder = builder.input("x", dtype=ir.DataType.INT64, shape=[1])
        with pytest.raises(ValueError, match="action_gen=False but action inputs"):
            module(
                builder.op,
                input_ids=placeholder,
                text_indexes=placeholder,
                position_ids=placeholder,
                und_len=placeholder,
                vision_tokens=placeholder,
                vision_sequence_indexes=placeholder,
                vision_timesteps=placeholder,
                vision_timestep_token_indexes=placeholder,
                vision_mse_loss_indexes=placeholder,
                action_tokens=placeholder,
            )


# ---------------------------------------------------------------------------
# Initializer-name coverage
# ---------------------------------------------------------------------------


def published_layer_keys(layer: int, *, gated: bool, qk_norm: bool, und_k_norm: bool) -> set:
    """Flat published parameter names for one MoT layer."""
    prefix = f"layers.{layer}."
    keys = {
        prefix + "self_attn.to_q.weight",
        prefix + "self_attn.to_k.weight",
        prefix + "self_attn.to_v.weight",
        prefix + "self_attn.to_out.weight",
        prefix + "self_attn.add_q_proj.weight",
        prefix + "self_attn.add_k_proj.weight",
        prefix + "self_attn.add_v_proj.weight",
        prefix + "self_attn.to_add_out.weight",
        prefix + "self_attn.norm_added_q.weight",
        prefix + "self_attn.norm_added_k.weight",
        prefix + "input_layernorm.weight",
        prefix + "input_layernorm_moe_gen.weight",
        prefix + "post_attention_layernorm.weight",
        prefix + "post_attention_layernorm_moe_gen.weight",
        prefix + "mlp.up_proj.weight",
        prefix + "mlp.down_proj.weight",
        prefix + "mlp_moe_gen.up_proj.weight",
        prefix + "mlp_moe_gen.down_proj.weight",
    }
    if gated:
        keys |= {prefix + "mlp.gate_proj.weight", prefix + "mlp_moe_gen.gate_proj.weight"}
    if qk_norm:
        keys |= {prefix + "self_attn.norm_q.weight", prefix + "self_attn.norm_k.weight"}
    if und_k_norm:
        keys.add(prefix + "self_attn.k_norm_und_for_gen.weight")
    return keys


class TestInitializerNames:
    def test_core_and_head_initializers_match_published_names(self):
        config = full_config()
        model = build_package(config)["model"]
        initializers = set(model.graph.initializers)

        expected = {
            "embed_tokens.weight",
            "norm.weight",
            "norm_moe_gen.weight",
            "proj_in.weight",
            "proj_in.bias",
            "proj_out.weight",
            "proj_out.bias",
            "time_embedder.linear_1.weight",
            "time_embedder.linear_1.bias",
            "time_embedder.linear_2.weight",
            "time_embedder.linear_2.bias",
            "audio_proj_in.weight",
            "audio_proj_in.bias",
            "audio_proj_out.weight",
            "audio_proj_out.bias",
            "audio_modality_embed",
            "action_proj_in.fc.weight",
            "action_proj_in.bias.weight",
            "action_proj_out.fc.weight",
            "action_proj_out.bias.weight",
            "action_modality_embed",
        }
        for layer in range(config.num_hidden_layers):
            expected |= published_layer_keys(layer, gated=True, qk_norm=True, und_k_norm=False)
        assert expected <= initializers
        # The module's declared checkpoint surface is exactly that set.
        assert Cosmos3OmniGeneratorModel(config).expected_checkpoint_keys() == expected

    def test_relu2_backbone_drops_gate_proj(self):
        config = tiny_config(hidden_act="relu2")
        keys = Cosmos3OmniGeneratorModel(config).expected_checkpoint_keys()
        assert "layers.0.mlp.up_proj.weight" in keys
        assert "layers.0.mlp_moe_gen.up_proj.weight" in keys
        assert not [key for key in keys if "gate_proj" in key]

    def test_qk_norm_off_swaps_in_und_k_norm_for_gen(self):
        config = tiny_config(qk_norm_for_text=False, use_und_k_norm_for_gen=True)
        keys = Cosmos3OmniGeneratorModel(config).expected_checkpoint_keys()
        assert "layers.0.self_attn.k_norm_und_for_gen.weight" in keys
        assert "layers.0.self_attn.norm_q.weight" not in keys
        assert "layers.0.self_attn.norm_k.weight" not in keys
        # The generation pathway always keeps its QK norms.
        assert "layers.0.self_attn.norm_added_q.weight" in keys

    def test_disabled_heads_have_no_initializers(self):
        keys = Cosmos3OmniGeneratorModel(tiny_config()).expected_checkpoint_keys()
        assert not [key for key in keys if key.startswith(("audio_", "action_"))]

    def test_derived_constants_are_not_checkpoint_keys(self):
        config = full_config()
        model = build_package(config)["model"]
        keys = Cosmos3OmniGeneratorModel(config).expected_checkpoint_keys()
        for name in ("rotary_emb.inv_freq", "rotary_emb.h_mask", "rotary_emb.w_mask"):
            assert name in model.graph.initializers
            assert name not in keys
        assert "time_proj.inv_freq" in model.graph.initializers
        assert "time_proj.inv_freq" not in keys


# ---------------------------------------------------------------------------
# Interleaved 3-axis mRoPE layout
# ---------------------------------------------------------------------------


def upstream_interleaved_axis_map(rope_axes_dim, rotary_dim: int) -> np.ndarray:
    """Axis (0=T, 1=H, 2=W) per rotary channel, using upstream's slice logic.

    Mirrors ``Cosmos3VLTextRotaryEmbedding.apply_interleaved_mrope``: start
    from the T frequencies and overwrite ``slice(offset, axis_dim * 3, 3)``
    with the H (offset 1) then W (offset 2) frequencies.
    """
    axis_map = np.zeros(rotary_dim, dtype=np.int64)
    for dim, offset in enumerate((1, 2), start=1):
        length = rope_axes_dim[dim] * 3
        axis_map[slice(offset, min(length, rotary_dim), 3)] = dim
    return axis_map


class TestRotaryLayout:
    @pytest.mark.parametrize(
        ("head_dim", "rope_axes_dim"),
        [(128, (24, 20, 20)), (8, (2, 1, 1)), (64, (12, 10, 10))],
    )
    def test_interleaved_masks_match_upstream(self, head_dim, rope_axes_dim):
        config = tiny_config(head_dim=head_dim, rope_axes_dim=rope_axes_dim)
        model = build_package(config)["model"]
        h_mask = model.graph.initializers["rotary_emb.h_mask"].const_value.numpy()
        w_mask = model.graph.initializers["rotary_emb.w_mask"].const_value.numpy()

        axis_map = upstream_interleaved_axis_map(rope_axes_dim, config.rotary_dim)
        np.testing.assert_array_equal(h_mask, axis_map == 1)
        np.testing.assert_array_equal(w_mask, axis_map == 2)
        # Channel budget matches the config's (T, H, W) split exactly.
        assert int((axis_map == 0).sum()) == rope_axes_dim[0]
        assert int(h_mask.sum()) == rope_axes_dim[1]
        assert int(w_mask.sum()) == rope_axes_dim[2]
        # Interleaved, not chunked: T/H/W alternate in the low channels.
        assert axis_map[:3].tolist() == [0, 1, 2]

    def test_inv_freq_matches_upstream(self):
        config = tiny_config(head_dim=128, rope_axes_dim=(24, 20, 20), rope_theta=5e6)
        model = build_package(config)["model"]
        inv_freq = model.graph.initializers["rotary_emb.inv_freq"].const_value.numpy()
        expected = 1.0 / (5e6 ** (np.arange(0, 128, 2, dtype=np.float32) / 128))
        assert inv_freq.shape == (64,)
        np.testing.assert_allclose(inv_freq, expected, rtol=1e-6)

    def test_masks_are_disjoint(self):
        config = tiny_config(head_dim=128, rope_axes_dim=(24, 20, 20))
        model = build_package(config)["model"]
        h_mask = model.graph.initializers["rotary_emb.h_mask"].const_value.numpy()
        w_mask = model.graph.initializers["rotary_emb.w_mask"].const_value.numpy()
        assert not np.any(h_mask & w_mask)


# ---------------------------------------------------------------------------
# DomainAwareLinear
# ---------------------------------------------------------------------------


class TorchDomainAwareLinear(torch.nn.Module):
    """PyTorch transcription of upstream ``DomainAwareLinear``."""

    def __init__(self, in_features: int, out_features: int, num_domains: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Embedding(num_domains, out_features * in_features)
        self.bias = torch.nn.Embedding(num_domains, out_features)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        weight = self.fc(domain_id).view(
            domain_id.shape[0], self.in_features, self.out_features
        )
        bias = self.bias(domain_id).view(domain_id.shape[0], self.out_features)
        return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias


class _DomainAwareLinearTask(ModelTask):
    """Wraps a bare :class:`Cosmos3OmniDomainAwareLinear` in a 2-input graph."""

    def build(self, module, config):
        graph, builder = _make_graph(name="domain_aware_linear")
        tokens = builder.input(
            "tokens", dtype=ir.DataType.FLOAT, shape=["n", module.in_features]
        )
        domain_ids = builder.input("domain_ids", dtype=ir.DataType.INT64, shape=["n"])
        builder.add_output(module(builder.op, tokens, domain_ids), "projected")
        return ModelPackage({"model": _make_model(graph)}, config=config)


class TestDomainAwareLinear:
    def test_parameter_names_and_shapes(self):
        config = full_config()
        model = build_package(config)["model"]
        initializers = model.graph.initializers
        # fc rows are flattened [in, out]; bias rows are [out].
        assert list(initializers["action_proj_in.fc.weight"].shape) == [
            config.num_embodiment_domains,
            config.hidden_size * config.action_dim,
        ]
        assert list(initializers["action_proj_in.bias.weight"].shape) == [
            config.num_embodiment_domains,
            config.hidden_size,
        ]
        assert list(initializers["action_proj_out.fc.weight"].shape) == [
            config.num_embodiment_domains,
            config.action_dim * config.hidden_size,
        ]
        assert list(initializers["action_proj_out.bias.weight"].shape) == [
            config.num_embodiment_domains,
            config.action_dim,
        ]

    def test_uses_gather_not_a_single_shared_matmul(self):
        module = Cosmos3OmniDomainAwareLinear(4, 6, 3)
        model = _DomainAwareLinearTask().build(module, tiny_config())["model"]
        op_types = [node.op_type for node in model.graph]
        # Two Gathers (weight + bias) proves the per-domain table lookup; a
        # plain Linear would have neither.
        assert op_types.count("Gather") == 2
        assert "MatMul" in op_types

    def test_matches_torch_reference(self, tmp_path):
        torch.manual_seed(5)
        in_features, out_features, num_domains = 4, 6, 3
        reference = TorchDomainAwareLinear(in_features, out_features, num_domains).eval()
        module = Cosmos3OmniDomainAwareLinear(in_features, out_features, num_domains)
        package = _DomainAwareLinearTask().build(module, tiny_config())
        package.apply_weights(
            {
                "fc.weight": reference.fc.weight.data,
                "bias.weight": reference.bias.weight.data,
            }
        )
        package.save(str(tmp_path), progress_bar=False)

        session = ort.InferenceSession(
            str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        rng = np.random.default_rng(9)
        tokens = rng.standard_normal((5, in_features)).astype(np.float32)
        # Repeated and out-of-order ids exercise the per-token gather.
        domain_ids = np.array([2, 0, 2, 1, 0], dtype=np.int64)

        actual = session.run(None, {"tokens": tokens, "domain_ids": domain_ids})[0]
        with torch.no_grad():
            expected = reference(torch.from_numpy(tokens), torch.from_numpy(domain_ids))
        assert actual.shape == (5, out_features)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)

    def test_distinct_domains_produce_distinct_outputs(self, tmp_path):
        torch.manual_seed(6)
        module = Cosmos3OmniDomainAwareLinear(3, 3, 2)
        package = _DomainAwareLinearTask().build(module, tiny_config())
        package.apply_weights(
            {
                "fc.weight": torch.tensor(
                    [[1.0, 0, 0, 0, 1, 0, 0, 0, 1], [0.0, 0, 2, 0, 2, 0, 2, 0, 0]]
                ),
                "bias.weight": torch.tensor([[0.0, 0, 0], [1.0, 1, 1]]),
            }
        )
        package.save(str(tmp_path), progress_bar=False)
        session = ort.InferenceSession(
            str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        tokens = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        actual = session.run(
            None, {"tokens": tokens, "domain_ids": np.array([0, 1], dtype=np.int64)}
        )[0]
        # Domain 0 is the identity; domain 1 permutes/scales and adds a bias.
        np.testing.assert_allclose(actual[0], [1.0, 2.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(actual[1], [7.0, 5.0, 3.0], rtol=1e-6)


# ---------------------------------------------------------------------------
# preprocess_weights
# ---------------------------------------------------------------------------


def fake_checkpoint(module: Cosmos3OmniGeneratorModel) -> dict:
    """Build a flat state dict covering exactly the module's expected keys."""
    package = build_package(module.config)
    initializers = package["model"].graph.initializers
    return {
        name: torch.zeros(*[int(dim) for dim in initializers[name].shape])
        for name in sorted(module.expected_checkpoint_keys())
    }


class TestPreprocessWeights:
    def test_accepts_flat_published_keys_unchanged(self):
        module = Cosmos3OmniGeneratorModel(full_config())
        state_dict = fake_checkpoint(module)
        assert module.preprocess_weights(state_dict).keys() == state_dict.keys()

    def test_drops_unused_lm_head(self):
        module = Cosmos3OmniGeneratorModel(tiny_config())
        state_dict = fake_checkpoint(module)
        state_dict["lm_head.weight"] = torch.zeros(4, 4)
        assert "lm_head.weight" not in module.preprocess_weights(state_dict)

    def test_drops_reasoner_vision_tower_keys(self):
        module = Cosmos3OmniGeneratorModel(tiny_config())
        state_dict = fake_checkpoint(module)
        state_dict["blocks.0.attn.qkv.weight"] = torch.zeros(2, 2)
        state_dict["patch_embed.proj.weight"] = torch.zeros(2, 2)
        state_dict["merger.linear_fc1.weight"] = torch.zeros(2, 2)
        state_dict["deepstack_merger_list.0.norm.weight"] = torch.zeros(2)
        state_dict["model.projector.linear_fc1.weight"] = torch.zeros(2, 2)
        processed = module.preprocess_weights(state_dict)
        assert not [
            key
            for key in processed
            if key.startswith(("blocks.", "patch_embed.", "projector."))
        ]

    def test_drops_edge_framework_key_norm_duplicates(self):
        module = Cosmos3OmniGeneratorModel(
            tiny_config(
                qk_norm_for_text=False,
                use_und_k_norm_for_gen=True,
            )
        )
        state_dict = fake_checkpoint(module)
        state_dict[
            "model.net.language_model.model.layers.0.self_attn.k_norm_und_for_gen.weight"
        ] = torch.zeros(module.config.head_dim)

        processed = module.preprocess_weights(state_dict)

        assert not [key for key in processed if key.startswith("net.language_model.")]

    def test_strips_container_prefix(self):
        module = Cosmos3OmniGeneratorModel(tiny_config())
        state_dict = {f"transformer.{k}": v for k, v in fake_checkpoint(module).items()}
        processed = module.preprocess_weights(state_dict)
        assert "embed_tokens.weight" in processed
        assert not [key for key in processed if key.startswith("transformer.")]

    def test_normalizes_sequential_to_out(self):
        module = Cosmos3OmniGeneratorModel(tiny_config())
        state_dict = fake_checkpoint(module)
        value = state_dict.pop("layers.0.self_attn.to_out.weight")
        state_dict["layers.0.self_attn.to_out.0.weight"] = value
        processed = module.preprocess_weights(state_dict)
        assert "layers.0.self_attn.to_out.weight" in processed
        assert "layers.0.self_attn.to_out.0.weight" not in processed

    def test_drops_recomputed_buffers(self):
        module = Cosmos3OmniGeneratorModel(tiny_config())
        state_dict = fake_checkpoint(module)
        state_dict["rotary_emb.inv_freq"] = torch.zeros(4)
        assert "rotary_emb.inv_freq" not in module.preprocess_weights(state_dict)

    def test_rejects_unexpected_key(self):
        module = Cosmos3OmniGeneratorModel(tiny_config())
        state_dict = fake_checkpoint(module)
        state_dict["layers.0.self_attn.mystery_proj.weight"] = torch.zeros(2, 2)
        with pytest.raises(ValueError, match="Unexpected Cosmos3-Omni transformer weights"):
            module.preprocess_weights(state_dict)

    def test_rejects_missing_key(self):
        module = Cosmos3OmniGeneratorModel(tiny_config())
        state_dict = fake_checkpoint(module)
        del state_dict["layers.1.mlp_moe_gen.down_proj.weight"]
        with pytest.raises(ValueError, match="missing weights"):
            module.preprocess_weights(state_dict)

    def test_weight_shard_allows_partial_checkpoint(self):
        module = Cosmos3OmniGeneratorModel(tiny_config())
        state_dict = fake_checkpoint(module)
        first_name = next(iter(state_dict))

        processed = module.preprocess_weight_shard({first_name: state_dict[first_name]})

        assert processed == {first_name: state_dict[first_name]}

    def test_rejects_gated_checkpoint_against_relu2_config(self):
        """A gate_proj in the checkpoint contradicts a relu2 backbone."""
        module = Cosmos3OmniGeneratorModel(tiny_config(hidden_act="relu2"))
        state_dict = fake_checkpoint(module)
        state_dict["layers.0.mlp.gate_proj.weight"] = torch.zeros(48, 32)
        with pytest.raises(ValueError, match="Unexpected Cosmos3-Omni transformer weights"):
            module.preprocess_weights(state_dict)

    def test_rejects_action_weights_when_head_disabled(self):
        module = Cosmos3OmniGeneratorModel(tiny_config())
        state_dict = fake_checkpoint(module)
        state_dict["action_proj_in.fc.weight"] = torch.zeros(3, 4)
        with pytest.raises(ValueError, match="Unexpected Cosmos3-Omni transformer weights"):
            module.preprocess_weights(state_dict)

    def test_reports_missing_action_weights_when_head_enabled(self):
        module = Cosmos3OmniGeneratorModel(full_config())
        state_dict = fake_checkpoint(module)
        del state_dict["action_proj_out.fc.weight"]
        with pytest.raises(ValueError, match=r"action_proj_out\.fc\.weight"):
            module.preprocess_weights(state_dict)


# ---------------------------------------------------------------------------
# Numerical parity against a PyTorch transcription of upstream forward
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Upstream ``_rotate_half``."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Upstream ``Cosmos3NemotronRMSNorm`` / diffusers ``RMSNorm``."""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    return (weight.float() * (x * torch.rsqrt(variance + eps))).to(dtype)


class TorchCosmos3Generator(torch.nn.Module):
    """PyTorch transcription of ``Cosmos3OmniTransformer.forward`` (packed).

    Implements exactly the upstream math — interleaved 3-axis mRoPE, the two
    QK-normed pathways, causal understanding self-attention, non-causal
    generation attention over ``[und ; gen]`` K/V, MoT per-expert MLPs, fp32
    timestep conditioning and the per-domain action heads — but reads the same
    packed tensors as the ONNX graph instead of ragged Python lists.
    """

    def __init__(self, config: Cosmos3OmniGeneratorConfig, seed: int = 0):
        super().__init__()
        self.config = config
        generator = torch.Generator().manual_seed(seed)

        def parameter(*shape, scale=0.05):
            return torch.nn.Parameter(torch.randn(*shape, generator=generator) * scale)

        hidden, inter = config.hidden_size, config.intermediate_size
        head_dim = config.head_dim
        q_size = config.attention_out_size
        kv_size = config.key_value_size

        self.embed_tokens = parameter(config.vocab_size, hidden)
        self.layer_params: list[dict[str, torch.nn.Parameter]] = []
        for index in range(config.num_hidden_layers):
            layer = {
                "to_q": parameter(q_size, hidden),
                "to_k": parameter(kv_size, hidden),
                "to_v": parameter(kv_size, hidden),
                "to_out": parameter(hidden, q_size),
                "add_q_proj": parameter(q_size, hidden),
                "add_k_proj": parameter(kv_size, hidden),
                "add_v_proj": parameter(kv_size, hidden),
                "to_add_out": parameter(hidden, q_size),
                "norm_added_q": parameter(head_dim, scale=0.1),
                "norm_added_k": parameter(head_dim, scale=0.1),
                "input_layernorm": parameter(hidden, scale=0.1),
                "input_layernorm_moe_gen": parameter(hidden, scale=0.1),
                "post_attention_layernorm": parameter(hidden, scale=0.1),
                "post_attention_layernorm_moe_gen": parameter(hidden, scale=0.1),
                "mlp.up_proj": parameter(inter, hidden),
                "mlp.down_proj": parameter(hidden, inter),
                "mlp_moe_gen.up_proj": parameter(inter, hidden),
                "mlp_moe_gen.down_proj": parameter(hidden, inter),
            }
            if config.is_gated_mlp:
                layer["mlp.gate_proj"] = parameter(inter, hidden)
                layer["mlp_moe_gen.gate_proj"] = parameter(inter, hidden)
            if config.qk_norm_for_text:
                layer["norm_q"] = parameter(head_dim, scale=0.1)
                layer["norm_k"] = parameter(head_dim, scale=0.1)
            if config.has_und_k_norm_for_gen:
                layer["k_norm_und_for_gen"] = parameter(head_dim, scale=0.1)
            for name, value in layer.items():
                self.register_parameter(f"layer{index}_{name.replace('.', '_')}", value)
            self.layer_params.append(layer)

        self.norm = parameter(hidden, scale=0.1)
        self.norm_moe_gen = parameter(hidden, scale=0.1)
        self.proj_in_weight = parameter(hidden, config.patch_latent_dim)
        self.proj_in_bias = parameter(hidden)
        self.proj_out_weight = parameter(config.patch_latent_dim, hidden)
        self.proj_out_bias = parameter(config.patch_latent_dim)
        self.time1_weight = parameter(hidden, config.time_proj_channels)
        self.time1_bias = parameter(hidden)
        self.time2_weight = parameter(hidden, hidden)
        self.time2_bias = parameter(hidden)
        if config.sound_gen:
            self.audio_in_weight = parameter(hidden, config.sound_dim)
            self.audio_in_bias = parameter(hidden)
            self.audio_out_weight = parameter(config.sound_dim, hidden)
            self.audio_out_bias = parameter(config.sound_dim)
            self.audio_modality_embed = parameter(hidden)
        if config.action_gen:
            domains = config.num_embodiment_domains
            self.action_in_fc = parameter(domains, hidden * config.action_dim)
            self.action_in_bias = parameter(domains, hidden)
            self.action_out_fc = parameter(domains, config.action_dim * hidden)
            self.action_out_bias = parameter(domains, config.action_dim)
            self.action_modality_embed = parameter(hidden)

    def published_state_dict(self) -> dict:
        """Return the weights under the published flat checkpoint names."""
        config = self.config
        state: dict = {"embed_tokens.weight": self.embed_tokens.data}
        attention_names = {
            "to_q": "self_attn.to_q",
            "to_k": "self_attn.to_k",
            "to_v": "self_attn.to_v",
            "to_out": "self_attn.to_out",
            "norm_q": "self_attn.norm_q",
            "norm_k": "self_attn.norm_k",
            "add_q_proj": "self_attn.add_q_proj",
            "add_k_proj": "self_attn.add_k_proj",
            "add_v_proj": "self_attn.add_v_proj",
            "to_add_out": "self_attn.to_add_out",
            "norm_added_q": "self_attn.norm_added_q",
            "norm_added_k": "self_attn.norm_added_k",
            "k_norm_und_for_gen": "self_attn.k_norm_und_for_gen",
        }
        for index, layer in enumerate(self.layer_params):
            for key, value in layer.items():
                name = attention_names.get(key, key)
                state[f"layers.{index}.{name}.weight"] = value.data
        state["norm.weight"] = self.norm.data
        state["norm_moe_gen.weight"] = self.norm_moe_gen.data
        state["proj_in.weight"] = self.proj_in_weight.data
        state["proj_in.bias"] = self.proj_in_bias.data
        state["proj_out.weight"] = self.proj_out_weight.data
        state["proj_out.bias"] = self.proj_out_bias.data
        state["time_embedder.linear_1.weight"] = self.time1_weight.data
        state["time_embedder.linear_1.bias"] = self.time1_bias.data
        state["time_embedder.linear_2.weight"] = self.time2_weight.data
        state["time_embedder.linear_2.bias"] = self.time2_bias.data
        if config.sound_gen:
            state["audio_proj_in.weight"] = self.audio_in_weight.data
            state["audio_proj_in.bias"] = self.audio_in_bias.data
            state["audio_proj_out.weight"] = self.audio_out_weight.data
            state["audio_proj_out.bias"] = self.audio_out_bias.data
            state["audio_modality_embed"] = self.audio_modality_embed.data
        if config.action_gen:
            state["action_proj_in.fc.weight"] = self.action_in_fc.data
            state["action_proj_in.bias.weight"] = self.action_in_bias.data
            state["action_proj_out.fc.weight"] = self.action_out_fc.data
            state["action_proj_out.bias.weight"] = self.action_out_bias.data
            state["action_modality_embed"] = self.action_modality_embed.data
        return state

    def rotary(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        head_dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        freqs = position_ids.float().unsqueeze(-1) * inv_freq  # (3, seq, head_dim // 2)
        merged = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = config.rope_axes_dim[dim] * 3
            merged[..., slice(offset, length, 3)] = freqs[dim][..., slice(offset, length, 3)]
        emb = torch.cat((merged, merged), dim=-1)
        return emb.cos(), emb.sin()

    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        config = self.config
        scaled = timesteps * config.timestep_scale
        half = config.time_proj_channels // 2
        exponent = -np.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
        freqs = scaled[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)  # flip_sin_to_cos
        hidden = functional.silu(functional.linear(emb, self.time1_weight, self.time1_bias))
        return functional.linear(hidden, self.time2_weight, self.time2_bias)

    def _feed_forward(self, x: torch.Tensor, layer: dict, prefix: str) -> torch.Tensor:
        """Upstream ``Cosmos3VLTextMLP`` for both ``silu`` and ``relu2``."""
        up = functional.linear(x, layer[f"{prefix}.up_proj"])
        if not self.config.is_gated_mlp:
            # relu2: down_proj(relu(up_proj(x)) ** 2)
            return functional.linear(
                functional.relu(up).square(), layer[f"{prefix}.down_proj"]
            )
        gate = functional.silu(functional.linear(x, layer[f"{prefix}.gate_proj"]))
        return functional.linear(gate * up, layer[f"{prefix}.down_proj"])

    def _attention(self, query, key, value, is_causal: bool) -> torch.Tensor:
        config = self.config
        groups = config.num_key_value_groups
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, dim=1)
        value = value.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, dim=1)
        out = functional.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        return out.squeeze(0).transpose(0, 1).reshape(-1, config.attention_out_size)

    def forward(self, batch: dict) -> dict:
        config = self.config
        eps = config.rms_norm_eps
        head_dim = config.head_dim
        heads, kv_heads = config.num_attention_heads, config.num_key_value_heads

        sequence_length = batch["position_ids"].shape[1]
        hidden_states = torch.zeros(sequence_length, config.hidden_size)
        hidden_states[batch["text_indexes"]] = self.embed_tokens[batch["input_ids"]]

        vision = functional.linear(
            batch["vision_tokens"], self.proj_in_weight, self.proj_in_bias
        )
        vision = vision.index_add(
            0,
            batch["vision_timestep_token_indexes"],
            self.timestep_embedding(batch["vision_timesteps"]),
        )
        hidden_states[batch["vision_sequence_indexes"]] = vision

        if config.sound_gen:
            sound = functional.linear(
                batch["sound_tokens"], self.audio_in_weight, self.audio_in_bias
            )
            sound = sound + self.audio_modality_embed
            sound = sound.index_add(
                0,
                batch["sound_timestep_token_indexes"],
                self.timestep_embedding(batch["sound_timesteps"]),
            )
            hidden_states[batch["sound_sequence_indexes"]] = sound

        if config.action_gen:
            domain_ids = batch["action_domain_ids"]
            weight = self.action_in_fc[domain_ids].view(
                -1, config.action_dim, config.hidden_size
            )
            action = torch.bmm(batch["action_tokens"].unsqueeze(1), weight).squeeze(1)
            action = action + self.action_in_bias[domain_ids] + self.action_modality_embed
            action = action.index_add(
                0,
                batch["action_timestep_token_indexes"],
                self.timestep_embedding(batch["action_timesteps"]),
            )
            hidden_states[batch["action_sequence_indexes"]] = action

        cos, sin = self.rotary(batch["position_ids"])
        und_len = int(batch["und_len"][0])
        und_seq, gen_seq = hidden_states[:und_len], hidden_states[und_len:]
        cos_und, sin_und = cos[:und_len].unsqueeze(1), sin[:und_len].unsqueeze(1)
        cos_gen, sin_gen = cos[und_len:].unsqueeze(1), sin[und_len:].unsqueeze(1)

        for layer in self.layer_params:
            und_norm = rms_norm(und_seq, layer["input_layernorm"], eps)
            gen_norm = rms_norm(gen_seq, layer["input_layernorm_moe_gen"], eps)

            q_und = functional.linear(und_norm, layer["to_q"]).view(-1, heads, head_dim)
            k_und = functional.linear(und_norm, layer["to_k"]).view(-1, kv_heads, head_dim)
            v_und = functional.linear(und_norm, layer["to_v"]).view(-1, kv_heads, head_dim)
            q_gen = functional.linear(gen_norm, layer["add_q_proj"]).view(-1, heads, head_dim)
            k_gen = functional.linear(gen_norm, layer["add_k_proj"]).view(
                -1, kv_heads, head_dim
            )
            v_gen = functional.linear(gen_norm, layer["add_v_proj"]).view(
                -1, kv_heads, head_dim
            )

            if config.qk_norm_for_text:
                q_und = rms_norm(q_und, layer["norm_q"], eps)
                k_und = rms_norm(k_und, layer["norm_k"], eps)
            # Upstream only builds this norm when the und pathway has none.
            k_und_for_gen = (
                rms_norm(k_und, layer["k_norm_und_for_gen"], eps)
                if config.has_und_k_norm_for_gen
                else k_und
            )
            q_gen = rms_norm(q_gen, layer["norm_added_q"], eps)
            k_gen = rms_norm(k_gen, layer["norm_added_k"], eps)

            q_und = q_und * cos_und + rotate_half(q_und) * sin_und
            k_und = k_und * cos_und + rotate_half(k_und) * sin_und
            k_und_for_gen = k_und_for_gen * cos_und + rotate_half(k_und_for_gen) * sin_und
            q_gen = q_gen * cos_gen + rotate_half(q_gen) * sin_gen
            k_gen = k_gen * cos_gen + rotate_half(k_gen) * sin_gen

            und_attn = self._attention(q_und, k_und, v_und, is_causal=True)
            gen_attn = self._attention(
                q_gen,
                torch.cat([k_und_for_gen, k_gen], dim=0),
                torch.cat([v_und, v_gen], dim=0),
                is_causal=False,
            )
            und_seq = und_seq + functional.linear(und_attn, layer["to_out"])
            gen_seq = gen_seq + functional.linear(gen_attn, layer["to_add_out"])

            und_post = rms_norm(und_seq, layer["post_attention_layernorm"], eps)
            gen_post = rms_norm(gen_seq, layer["post_attention_layernorm_moe_gen"], eps)
            und_seq = und_seq + self._feed_forward(und_post, layer, "mlp")
            gen_seq = gen_seq + self._feed_forward(gen_post, layer, "mlp_moe_gen")

        last_hidden_state = torch.cat(
            [rms_norm(und_seq, self.norm, eps), rms_norm(gen_seq, self.norm_moe_gen, eps)],
            dim=0,
        )

        outputs = {
            "vision_pred": functional.linear(
                last_hidden_state[batch["vision_mse_loss_indexes"]],
                self.proj_out_weight,
                self.proj_out_bias,
            )
        }
        if config.sound_gen:
            outputs["sound_pred"] = functional.linear(
                last_hidden_state[batch["sound_mse_loss_indexes"]],
                self.audio_out_weight,
                self.audio_out_bias,
            )
        if config.action_gen:
            domain_ids = batch["action_pred_domain_ids"]
            weight = self.action_out_fc[domain_ids].view(
                -1, config.hidden_size, config.action_dim
            )
            selected = last_hidden_state[batch["action_mse_loss_indexes"]]
            outputs["action_pred"] = (
                torch.bmm(selected.unsqueeze(1), weight).squeeze(1)
                + self.action_out_bias[domain_ids]
            )
        return outputs


def packed_batch(config: Cosmos3OmniGeneratorConfig) -> dict:
    """Build a packed feed covering text, vision and the enabled heads."""
    rng = np.random.default_rng(3)
    und_len, num_vision, num_vision_noisy = 5, 4, 3
    cursor = und_len + num_vision

    batch: dict = {
        "input_ids": rng.integers(0, config.vocab_size, und_len).astype(np.int64),
        "text_indexes": np.arange(und_len, dtype=np.int64),
        "und_len": np.array([und_len], dtype=np.int64),
        "vision_tokens": rng.standard_normal((num_vision, config.patch_latent_dim)).astype(
            np.float32
        ),
        "vision_sequence_indexes": np.arange(und_len, cursor, dtype=np.int64),
        "vision_timesteps": rng.uniform(1, 1000, num_vision_noisy).astype(np.float32),
        # Non-contiguous noisy rows: only frames 0, 1 and 3 carry noise.
        "vision_timestep_token_indexes": np.array([0, 1, 3], dtype=np.int64),
        "vision_mse_loss_indexes": np.array(
            [und_len, und_len + 1, und_len + 3], dtype=np.int64
        ),
    }
    if config.sound_gen:
        num_sound = 2
        batch.update(
            {
                "sound_tokens": rng.standard_normal((num_sound, config.sound_dim)).astype(
                    np.float32
                ),
                "sound_sequence_indexes": np.arange(
                    cursor, cursor + num_sound, dtype=np.int64
                ),
                "sound_timesteps": rng.uniform(1, 1000, num_sound).astype(np.float32),
                "sound_timestep_token_indexes": np.arange(num_sound, dtype=np.int64),
                "sound_mse_loss_indexes": np.arange(
                    cursor, cursor + num_sound, dtype=np.int64
                ),
            }
        )
        cursor += num_sound
    if config.action_gen:
        num_action = 3
        batch.update(
            {
                "action_tokens": rng.standard_normal((num_action, config.action_dim)).astype(
                    np.float32
                ),
                # Mixed embodiments in one packed batch.
                "action_domain_ids": np.array([0, 2, 2], dtype=np.int64),
                "action_sequence_indexes": np.arange(
                    cursor, cursor + num_action, dtype=np.int64
                ),
                "action_timesteps": rng.uniform(1, 1000, 2).astype(np.float32),
                "action_timestep_token_indexes": np.array([0, 2], dtype=np.int64),
                "action_mse_loss_indexes": np.array([cursor, cursor + 2], dtype=np.int64),
                "action_pred_domain_ids": np.array([0, 2], dtype=np.int64),
            }
        )
        cursor += num_action

    sequence_length = cursor
    steps = np.arange(sequence_length, dtype=np.int64)
    # Distinct T/H/W tracks so a collapsed mRoPE axis would show up.
    batch["position_ids"] = np.stack([steps, (steps * 2) % 7, (steps * 3) % 5])
    return batch


class TestNumericalParity:
    @pytest.mark.parametrize(
        "variant",
        [
            "generator_only",
            "sound_and_action",
            "und_k_norm_for_gen",
            "relu2_backbone",
        ],
    )
    def test_matches_torch_reference(self, tmp_path, variant):
        config = {
            "generator_only": tiny_config(),
            "sound_and_action": full_config(),
            # qk_norm_for_text=False makes norm_q/norm_k Identity and adds the
            # separate k_norm_und_for_gen applied to the raw understanding keys.
            "und_k_norm_for_gen": tiny_config(
                qk_norm_for_text=False, use_und_k_norm_for_gen=True
            ),
            # Nemotron backbone: non-gated squared-ReLU feed-forward.
            "relu2_backbone": tiny_config(hidden_act="relu2"),
        }[variant]
        reference = TorchCosmos3Generator(config).eval()
        module = Cosmos3OmniGeneratorModel(config)

        package = build_package(config, execution_provider="cpu")
        package.apply_weights(module.preprocess_weights(reference.published_state_dict()))
        package.save(str(tmp_path), progress_bar=False)

        session = ort.InferenceSession(
            str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        batch = packed_batch(config)
        assert {value.name for value in package["model"].graph.inputs} == set(batch)

        output_names = [output.name for output in session.get_outputs()]
        actual = session.run(output_names, batch)
        with torch.no_grad():
            expected = reference(
                {name: torch.from_numpy(value) for name, value in batch.items()}
            )

        assert output_names == list(expected_output_names(config))
        for name, value in zip(output_names, actual, strict=True):
            np.testing.assert_allclose(
                value, expected[name].numpy(), rtol=1e-4, atol=1e-4, err_msg=name
            )

    def test_causal_understanding_pathway_ignores_future_text(self, tmp_path):
        """Understanding self-attention must stay causal.

        Perturbing the *last* understanding token cannot change the first
        understanding token's contribution, but it must change the generation
        predictions (which attend over all understanding keys non-causally).
        """
        config = tiny_config()
        reference = TorchCosmos3Generator(config).eval()
        module = Cosmos3OmniGeneratorModel(config)
        package = build_package(config, execution_provider="cpu")
        package.apply_weights(module.preprocess_weights(reference.published_state_dict()))
        package.save(str(tmp_path), progress_bar=False)
        session = ort.InferenceSession(
            str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
        )

        batch = packed_batch(config)
        baseline = session.run(["vision_pred"], batch)[0]
        perturbed_batch = dict(batch)
        perturbed_batch["input_ids"] = batch["input_ids"].copy()
        perturbed_batch["input_ids"][-1] = (batch["input_ids"][-1] + 7) % config.vocab_size
        perturbed = session.run(["vision_pred"], perturbed_batch)[0]
        # Generation tokens see every understanding key, so the change lands.
        assert not np.allclose(baseline, perturbed, atol=1e-6)

    def test_zero_length_optional_heads_run(self, tmp_path):
        """A configured head with no content this step accepts empty tensors."""
        config = full_config()
        reference = TorchCosmos3Generator(config).eval()
        module = Cosmos3OmniGeneratorModel(config)
        package = build_package(config, execution_provider="cpu")
        package.apply_weights(module.preprocess_weights(reference.published_state_dict()))
        package.save(str(tmp_path), progress_bar=False)
        session = ort.InferenceSession(
            str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
        )

        batch = packed_batch(config)
        empty = dict(batch)
        for name in (
            "sound_tokens",
            "sound_sequence_indexes",
            "sound_timesteps",
            "sound_timestep_token_indexes",
            "sound_mse_loss_indexes",
            "action_tokens",
            "action_domain_ids",
            "action_sequence_indexes",
            "action_timesteps",
            "action_timestep_token_indexes",
            "action_mse_loss_indexes",
            "action_pred_domain_ids",
        ):
            value = batch[name]
            empty[name] = np.zeros((0, *value.shape[1:]), dtype=value.dtype)

        outputs = session.run(None, empty)
        shapes = {
            output.name: value.shape
            for output, value in zip(session.get_outputs(), outputs, strict=True)
        }
        assert shapes["sound_pred"] == (0, config.sound_dim)
        assert shapes["action_pred"] == (0, config.action_dim)
        assert shapes["vision_pred"][0] == batch["vision_mse_loss_indexes"].shape[0]


def test_config_is_a_dataclass_with_replaceable_fields():
    """``dataclasses.replace`` is how the build pipeline derives variants."""
    config = Cosmos3OmniGeneratorConfig.from_diffusers(PUBLISHED_CONFIG)
    replaced = dataclasses.replace(config, dtype=ir.DataType.FLOAT16)
    replaced.validate()
    assert replaced.dtype == ir.DataType.FLOAT16
    assert replaced.sound_gen is config.sound_gen
