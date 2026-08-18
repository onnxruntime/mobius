# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build/weight tests for Qwen3.5-MoE export.

Focuses on the ``com.microsoft::QMoE`` emission path: when the quantization
config matches the native QMoE ABI (blk32 int4 Olive/GPTQ/AWQ),
:meth:`Qwen35MoECausalLMModel.preprocess_weights` keeps the fused expert-major
tensors and repacks them into ``fc1``/``fc2`` QMoE parameters (mirroring
DeepSeek-V3), instead of un-fusing into a per-expert dense fallback. All tiny
random configs -- no checkpoint download.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest
import torch

from mobius._configs import QuantizationConfig, VisionConfig
from mobius._testing import make_config
from mobius.models.qwen35 import (
    Qwen35CausalLMModel,
    Qwen35MoECausalLMModel,
    Qwen35MoEVL3ModelCausalLMModel,
    Qwen35VL3ModelCausalLMModel,
)

_E, _H, _INT, _BLK, _BITS = 8, 32, 16, 16, 4
_FC1_OUT = 2 * _INT


def _moe_config(quantization: QuantizationConfig | None) -> object:
    return make_config(
        num_hidden_layers=1,
        hidden_size=_H,
        intermediate_size=64,
        moe_intermediate_size=_INT,
        shared_expert_intermediate_size=_INT,
        num_local_experts=_E,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        quantization=quantization,
    )


def _olive_expert_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic HF-style fused Olive-quantized MoE expert tensors.

    Olive's on-disk suffix convention is an *underscore* suffix directly on
    the parameter name (``<pname>_qweight``/``_scales``/``_qzeros``), not a
    dotted one -- see ``olive/common/quant/state_dict.py``. For a fused MoE
    parameter like ``gate_up_proj`` (no nested ``nn.Linear``), this means
    ``experts.gate_up_proj_qweight``, not ``experts.gate_up_proj.qweight``.
    """
    p = "model.language_model.layers.0.mlp."
    return {
        p + "experts.gate_up_proj_qweight": torch.randint(
            0, 256, (_E, _FC1_OUT, _H * _BITS // 8), dtype=torch.uint8
        ),
        p + "experts.gate_up_proj_scales": torch.rand(_E, _FC1_OUT, _H // _BLK),
        p + "experts.gate_up_proj_qzeros": torch.randint(
            0, 256, (_E, _FC1_OUT, 1), dtype=torch.uint8
        ),
        p + "experts.down_proj_qweight": torch.randint(
            0, 256, (_E, _H, _INT * _BITS // 8), dtype=torch.uint8
        ),
        p + "experts.down_proj_scales": torch.rand(_E, _H, _INT // _BLK),
        p + "experts.down_proj_qzeros": torch.randint(0, 256, (_E, _H, 1), dtype=torch.uint8),
        p + "gate.weight": torch.rand(_E, _H),
    }


def _moe_vl_config(
    quantization: QuantizationConfig | None, *, tie_word_embeddings: bool = False
) -> object:
    return make_config(
        num_hidden_layers=1,
        hidden_size=_H,
        intermediate_size=64,
        moe_intermediate_size=_INT,
        shared_expert_intermediate_size=_INT,
        num_local_experts=_E,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        quantization=quantization,
        tie_word_embeddings=tie_word_embeddings,
        vision=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=2,
            in_channels=3,
            out_hidden_size=_H,
            num_position_embeddings=4,
        ),
        image_token_id=99,
        temporal_patch_size=1,
    )


class TestQwen35MoEQMoEExport:
    def test_moe_block_uses_qmoe_when_quantized(self):
        model = _moe_config(
            QuantizationConfig(bits=4, group_size=_BLK, quant_method="olive", sym=False)
        )
        model = Qwen35MoECausalLMModel(model)
        block = model.model.layers[0].mlp
        # Fused QMoE mode: no per-expert MLP ModuleList.
        assert block.experts is None
        assert hasattr(block, "fc1_experts_weights")

    def test_olive_preprocess_packs_qmoe_and_binds(self):
        config = _moe_config(
            QuantizationConfig(bits=4, group_size=_BLK, quant_method="olive", sym=False)
        )
        model = Qwen35MoECausalLMModel(config)
        out = model.preprocess_weights(_olive_expert_state_dict())

        prefix = "model.layers.0.mlp."
        assert out[prefix + "fc1_experts_weights"].shape == (_E, _FC1_OUT, _H * _BITS // 8)
        assert out[prefix + "fc1_scales"].shape == (_E, _FC1_OUT, _H // _BLK)
        assert out[prefix + "fc1_experts_zero_points"].shape == (_E, _FC1_OUT, 1)
        assert out[prefix + "fc2_experts_weights"].shape == (_E, _H, _INT * _BITS // 8)
        assert out[prefix + "fc2_scales"].shape == (_E, _H, _INT // _BLK)
        assert out[prefix + "fc2_experts_zero_points"].shape == (_E, _H, 1)

        # No per-expert dense-fallback storm leaked through.
        assert not any(".mlp.experts." in k for k in out)

        # Packed keys bind to real model parameters (weights actually load).
        param_names = {n for n, _ in model.named_parameters()}
        for suffix in (
            "fc1_experts_weights",
            "fc1_scales",
            "fc1_experts_zero_points",
            "fc2_experts_weights",
            "fc2_scales",
            "fc2_experts_zero_points",
        ):
            assert prefix + suffix in param_names

    def test_unquantized_preprocess_unfuses_dense_fallback(self):
        config = _moe_config(None)
        model = Qwen35MoECausalLMModel(config)
        p = "model.language_model.layers.0.mlp."
        fused = {
            p + "experts.gate_up_proj": torch.rand(_E, _FC1_OUT, _H),
            p + "experts.down_proj": torch.rand(_E, _H, _INT),
        }
        out = model.preprocess_weights(fused)

        # Dense fallback un-fuses into per-expert gate/up/down tensors.
        assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (_INT, _H)
        assert out["model.layers.0.mlp.experts.0.up_proj.weight"].shape == (_INT, _H)
        assert out[f"model.layers.0.mlp.experts.{_E - 1}.down_proj.weight"].shape == (
            _H,
            _INT,
        )
        assert not any("fc1_experts_weights" in k for k in out)

    def test_unsupported_qmoe_quantization_with_packed_experts_raises(self):
        """Packed quantized expert weights + an unsupported QMoE ABI must raise.

        Regression test: ``group_size=24`` (not a power of two) makes
        ``_supported_qmoe_quantization`` reject the config, so
        ``preprocess_weights`` takes the dense-fallback branch. But that
        branch's fused-tensor unfuser only knows how to split *unquantized*
        float ``experts.gate_up_proj``/``experts.down_proj`` tensors -- there
        is no code path that splits packed ``_qweight``/``_scales``/
        ``_qzeros`` fused-expert tensors into per-expert quantized Linear
        initializers. Previously this silently fell through to
        ``cleaned[key] = value`` and produced a graph whose per-expert
        Linear modules expect keys that were never generated.
        """
        config = _moe_config(
            QuantizationConfig(bits=8, group_size=_BLK, quant_method="olive", sym=False)
        )
        model = Qwen35MoECausalLMModel(config)
        try:
            model.preprocess_weights(_olive_expert_state_dict())
        except ValueError as e:
            assert "QMoE ABI" in str(e)
        else:
            raise AssertionError("expected ValueError for unsupported QMoE + packed experts")


class TestQwen35MoEVL3ModelQMoEExport:
    def test_olive_preprocess_packs_qmoe_and_binds(self):
        config = _moe_vl_config(
            QuantizationConfig(bits=4, group_size=_BLK, quant_method="olive", sym=False)
        )
        model = Qwen35MoEVL3ModelCausalLMModel(config)
        out = model.preprocess_weights(_olive_expert_state_dict())

        prefix = "decoder.model.layers.0.mlp."
        assert out[prefix + "fc1_experts_weights"].shape == (
            _E,
            _FC1_OUT,
            _H * _BITS // 8,
        )
        assert out[prefix + "fc1_scales"].shape == (_E, _FC1_OUT, _H // _BLK)
        assert out[prefix + "fc1_experts_zero_points"].shape == (_E, _FC1_OUT, 1)
        assert out[prefix + "fc2_experts_weights"].shape == (
            _E,
            _H,
            _INT * _BITS // 8,
        )
        assert out[prefix + "fc2_scales"].shape == (_E, _H, _INT // _BLK)
        assert out[prefix + "fc2_experts_zero_points"].shape == (_E, _H, 1)
        assert not any(".mlp.experts." in key for key in out)

        param_names = {name for name, _ in model.named_parameters()}
        for suffix in (
            "fc1_experts_weights",
            "fc1_scales",
            "fc1_experts_zero_points",
            "fc2_experts_weights",
            "fc2_scales",
            "fc2_experts_zero_points",
        ):
            assert prefix + suffix in param_names

    def test_unquantized_preprocess_unfuses_dense_fallback(self):
        model = Qwen35MoEVL3ModelCausalLMModel(_moe_vl_config(None))
        prefix = "model.language_model.layers.0.mlp."
        out = model.preprocess_weights(
            {
                prefix + "experts.gate_up_proj": torch.rand(_E, _FC1_OUT, _H),
                prefix + "experts.down_proj": torch.rand(_E, _H, _INT),
            }
        )

        target = "decoder.model.layers.0.mlp."
        assert out[target + "experts.0.gate_proj.weight"].shape == (_INT, _H)
        assert out[target + "experts.0.up_proj.weight"].shape == (_INT, _H)
        assert out[target + f"experts.{_E - 1}.down_proj.weight"].shape == (_H, _INT)
        assert not any("fc1_experts_weights" in key for key in out)

    def test_unsupported_qmoe_quantization_with_packed_experts_raises(self):
        config = _moe_vl_config(
            QuantizationConfig(bits=8, group_size=_BLK, quant_method="olive", sym=False)
        )
        model = Qwen35MoEVL3ModelCausalLMModel(config)

        try:
            model.preprocess_weights(_olive_expert_state_dict())
        except ValueError as error:
            assert "QMoE ABI" in str(error)
        else:
            raise AssertionError("expected ValueError for unsupported QMoE + packed experts")

    def test_gptq_qmoe_export_raises_clear_error(self):
        config = _moe_vl_config(
            QuantizationConfig(bits=4, group_size=_BLK, quant_method="gptq", sym=False)
        )
        model = Qwen35MoEVL3ModelCausalLMModel(config)
        packed_experts = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj.qweight": torch.zeros(
                _E, _H * _BITS // 32, _FC1_OUT, dtype=torch.int32
            )
        }

        try:
            model.preprocess_weights(packed_experts)
        except NotImplementedError as error:
            assert "only supports QMoE export for Olive-quantized checkpoints" in str(error)
            assert "gptq" in str(error)
        else:
            raise AssertionError("expected NotImplementedError for GPTQ VL-MoE export")

    def test_quantized_embeddings_or_lm_head_raise_clear_error(self):
        for flag, key in (
            ("quantize_embeddings", "model.language_model.embed_tokens.weight_qweight"),
            ("quantize_lm_head", "lm_head.weight_qweight"),
        ):
            config = _moe_vl_config(
                QuantizationConfig(
                    bits=4,
                    group_size=_BLK,
                    quant_method="olive",
                    sym=False,
                    **{flag: True},
                )
            )
            model = Qwen35MoEVL3ModelCausalLMModel(config)
            packed_weight = {
                key: torch.zeros(config.vocab_size, _H * _BITS // 8, dtype=torch.uint8)
            }

            try:
                model.preprocess_weights(packed_weight)
            except NotImplementedError as error:
                assert "Quantized embeddings and LM heads are not yet supported" in str(error)
                assert "decoder.model.embed_tokens" in str(error)
            else:
                raise AssertionError(
                    f"expected NotImplementedError when {flag}=True for VL-MoE export"
                )

    def test_quantization_preserves_vision_embedding_and_mtp_routing(self):
        config = _moe_vl_config(
            QuantizationConfig(
                bits=4,
                group_size=_BLK,
                quant_method="olive",
                sym=False,
                tie_word_embeddings=True,
            )
        )
        model = Qwen35MoEVL3ModelCausalLMModel(config)
        vision = torch.rand(32, 16)
        embedding = torch.rand(config.vocab_size, _H)
        state_dict = {
            **_olive_expert_state_dict(),
            "model.visual.blocks.0.mlp.linear_fc1.weight": vision,
            "model.language_model.embed_tokens.weight": embedding,
            "mtp_head.weight": torch.rand(1),
        }

        out = model.preprocess_weights(state_dict)

        assert out["vision_encoder.visual.blocks.0.mlp.up_proj.weight"] is vision
        assert out["decoder.model.embed_tokens.weight"] is embedding
        assert out["embedding.embed_tokens.weight"] is embedding
        assert out["decoder.lm_head.weight"] is embedding
        assert "mtp_head.weight" not in out


class TestQwen35VL3ModelQuantization:
    def test_olive_quantization_renames_and_binds(self):
        config = _moe_vl_config(
            QuantizationConfig(bits=4, group_size=_BLK, quant_method="olive", sym=False)
        )
        model = Qwen35VL3ModelCausalLMModel(config)
        target = "decoder.model.layers.0.self_attn.q_proj.weight"
        expected_shape = tuple(dict(model.named_parameters())[target].shape)
        key = "model.language_model.layers.0.self_attn.q_proj.weight_qweight"
        result = model.preprocess_weights(
            {
                key: torch.zeros(
                    expected_shape[0],
                    expected_shape[1] * expected_shape[2],
                    dtype=torch.uint8,
                )
            }
        )

        assert tuple(result[target].shape) == expected_shape

    def test_quantized_embeddings_or_lm_head_raise(self):
        for flag in ("quantize_embeddings", "quantize_lm_head"):
            config = _moe_vl_config(
                QuantizationConfig(
                    bits=4,
                    group_size=_BLK,
                    quant_method="olive",
                    sym=False,
                    **{flag: True},
                )
            )
            model = Qwen35VL3ModelCausalLMModel(config)

            try:
                model.preprocess_weights({})
            except NotImplementedError as error:
                assert "Quantized embeddings and LM heads" in str(error)
            else:
                raise AssertionError(
                    f"expected NotImplementedError when {flag}=True for dense VL export"
                )

    def test_gptq_and_awq_quantization(self):
        for quant_method in ("gptq", "awq"):
            config = _moe_vl_config(
                QuantizationConfig(
                    bits=4,
                    group_size=_BLK,
                    quant_method=quant_method,
                    sym=False,
                )
            )
            model = Qwen35VL3ModelCausalLMModel(config)
            target = "decoder.model.layers.0.self_attn.q_proj.weight"
            expected_shape = tuple(dict(model.named_parameters())[target].shape)
            k_packed = expected_shape[1] * _BLK * _BITS // 32
            key = "model.language_model.layers.0.self_attn.q_proj.qweight"
            result = model.preprocess_weights(
                {
                    key: torch.zeros(
                        k_packed,
                        expected_shape[0],
                        dtype=torch.int32,
                    )
                }
            )

            assert tuple(result[target].shape) == expected_shape

    def test_packed_embedding_or_lm_head_key_raises_without_config_flag(self):
        for quant_method, key in (
            ("olive", "model.language_model.embed_tokens.weight_qweight"),
            ("gptq", "model.language_model.embed_tokens.qweight"),
            ("awq", "lm_head.qweight"),
        ):
            config = _moe_vl_config(
                QuantizationConfig(
                    bits=4,
                    group_size=_BLK,
                    quant_method=quant_method,
                    sym=False,
                )
            )
            model = Qwen35VL3ModelCausalLMModel(config)

            try:
                model.preprocess_weights({key: torch.zeros(1)})
            except NotImplementedError as error:
                assert "Packed checkpoint key" in str(error)
            else:
                raise AssertionError(
                    f"expected packed {quant_method} embedding/head rejection"
                )

    def test_tied_lm_head_backfills_all_embedding_initializers(self):
        head = torch.randn(128, _H)
        for model_class in (
            Qwen35VL3ModelCausalLMModel,
            Qwen35MoEVL3ModelCausalLMModel,
        ):
            model = model_class(_moe_vl_config(None, tie_word_embeddings=True))
            result = model.preprocess_weights({"lm_head.weight": head})

            assert result["decoder.lm_head.weight"] is head
            assert result["decoder.model.embed_tokens.weight"] is head
            assert result["embedding.embed_tokens.weight"] is head

    def test_tying_partial_state_dict_without_tables_is_a_noop(self):
        for model_class in (
            Qwen35VL3ModelCausalLMModel,
            Qwen35MoEVL3ModelCausalLMModel,
        ):
            model = model_class(_moe_vl_config(None, tie_word_embeddings=True))
            assert model.preprocess_weights({}) == {}

    def test_unquantized_float_routing_is_unchanged(self):
        model = Qwen35VL3ModelCausalLMModel(_moe_vl_config(None))
        decoder_weight = torch.randn(_H, _H)
        vision_weight = torch.randn(32, 16)
        embedding = torch.randn(128, _H)
        result = model.preprocess_weights(
            {
                "model.language_model.layers.0.self_attn.q_proj.weight": decoder_weight,
                "model.visual.blocks.0.mlp.linear_fc1.weight": vision_weight,
                "model.language_model.embed_tokens.weight": embedding,
            }
        )

        assert result["decoder.model.layers.0.self_attn.q_proj.weight"] is decoder_weight
        assert result["vision_encoder.visual.blocks.0.mlp.up_proj.weight"] is vision_weight
        assert result["decoder.model.embed_tokens.weight"] is embedding
        assert result["embedding.embed_tokens.weight"] is embedding


# ---------------------------------------------------------------------------
# Mixed float/quantized decoder: Olive-quantized Qwen3.5/3.6-MoE keeps the
# GatedDeltaNet projections and shared_expert_gate in floating point. Dense
# Qwen3.5 and GPTQ/AWQ MoE checkpoints stay fully quantized.
# ---------------------------------------------------------------------------

# GatedDeltaNet dims chosen so in_proj_qkv is [key_dim * 2 + value_dim, _H]
# = [8 * 2 + 32, 32] = [48, 32] -- the exact shape from the Olive repro.
_LINEAR_ATTN_DIMS = dict(
    linear_num_value_heads=2,
    linear_num_key_heads=1,
    linear_key_head_dim=8,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
)
_QKV_OUT = 48
# Qwen3.5 full attention fuses the output gate into q_proj: 2 * heads * head_dim.
_Q_OUT = 2 * 4 * 8


def _hybrid_moe_config(
    quantization: QuantizationConfig | None, *, vision: bool = False
) -> object:
    """Two-layer Qwen3.5-MoE config: layer 0 linear attention, layer 1 full."""
    factory = _moe_vl_config if vision else _moe_config
    config = factory(quantization)
    config = dataclasses.replace(
        config,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        **_LINEAR_ATTN_DIMS,
    )
    return config


def _hybrid_dense_config(quantization: QuantizationConfig | None) -> object:
    """Two-layer dense Qwen3.5 config: layer 0 linear attention, layer 1 full."""
    return make_config(
        num_hidden_layers=2,
        hidden_size=_H,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        quantization=quantization,
        **_LINEAR_ATTN_DIMS,
    )


def _quant(method: str) -> QuantizationConfig:
    return QuantizationConfig(bits=4, group_size=_BLK, quant_method=method, sym=False)


def _olive_quant() -> QuantizationConfig:
    return _quant("olive")


def _packed_shape(out_features: int, in_features: int) -> tuple[int, int, int]:
    """MatMulNBits initializer shape for an int4 blk-``_BLK`` linear."""
    return (out_features, in_features // _BLK, _BLK * _BITS // 8)


class TestQwen35MixedPrecisionDecoder:
    """Olive-quantized Qwen3.5/3.6-**MoE** leaves some decoder modules float.

    Olive's quantization walk excludes GatedDeltaNet (``linear_attn``, mapped
    for ``qwen3_5_moe``/``qwen3_5_moe_text`` only) and ``shared_expert_gate``,
    while ordinary attention, the shared expert MLP and the routed experts
    stay quantized. The exported MoE graph must match that mixed layout, and
    only for Olive: dense Qwen3.5 and GPTQ/AWQ MoE checkpoints keep a fully
    quantized ``linear_attn``/gate.
    """

    def test_linear_attention_projections_are_float(self):
        model = Qwen35MoECausalLMModel(_hybrid_moe_config(_olive_quant()))
        linear_attn = model.model.layers[0].linear_attn

        assert tuple(linear_attn.in_proj_qkv.weight.shape) == (_QKV_OUT, _H)
        assert linear_attn.in_proj_qkv.weight.dtype != ir.DataType.UINT8
        # Float Linear has no packed sidecar parameters.
        assert not hasattr(linear_attn.in_proj_qkv, "scales")
        for name in ("in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            projection = getattr(linear_attn, name)
            assert len(projection.weight.shape) == 2, name
            assert projection.weight.dtype != ir.DataType.UINT8, name
            assert not hasattr(projection, "scales"), name

    def test_dense_olive_linear_attention_stays_quantized(self):
        """Dense Qwen3.5 is absent from Olive's MAMBA table -> still packed."""
        model = Qwen35CausalLMModel(_hybrid_dense_config(_olive_quant()))
        linear_attn = model.model.layers[0].linear_attn

        assert tuple(linear_attn.in_proj_qkv.weight.shape) == _packed_shape(_QKV_OUT, _H)
        assert linear_attn.in_proj_qkv.weight.dtype == ir.DataType.UINT8
        assert linear_attn.in_proj_qkv.scales is not None
        for name in ("in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            projection = getattr(linear_attn, name)
            assert projection.weight.dtype == ir.DataType.UINT8, name
        # Ordinary attention in the dense stack is quantized as before.
        assert model.model.layers[1].self_attn.q_proj.weight.dtype == ir.DataType.UINT8

    @pytest.mark.parametrize("method", ["gptq", "awq"])
    def test_non_olive_quantization_preserves_existing_graph(self, method: str):
        """The Olive-specific fix does not change other graph formats."""
        model = Qwen35MoECausalLMModel(_hybrid_moe_config(_quant(method)))

        linear_attn = model.model.layers[0].linear_attn
        assert tuple(linear_attn.in_proj_qkv.weight.shape) == _packed_shape(_QKV_OUT, _H)
        assert linear_attn.in_proj_qkv.weight.dtype == ir.DataType.UINT8

        block = model.model.layers[1].mlp
        assert tuple(block.shared_expert_gate.weight.shape) == _packed_shape(1, _H)
        assert block.shared_expert_gate.weight.dtype == ir.DataType.UINT8
        # Routed experts still take the fused QMoE path.
        assert block.experts is None
        assert tuple(block.fc1_experts_weights.shape) == (_E, _FC1_OUT, _H * _BITS // 8)

    def test_full_attention_and_shared_expert_stay_quantized(self):
        model = Qwen35MoECausalLMModel(_hybrid_moe_config(_olive_quant()))
        layer = model.model.layers[1]

        q_proj = layer.self_attn.q_proj
        assert tuple(q_proj.weight.shape) == _packed_shape(_Q_OUT, _H)
        assert q_proj.weight.dtype == ir.DataType.UINT8
        assert q_proj.scales is not None

        gate_proj = layer.mlp.shared_expert.gate_proj
        assert tuple(gate_proj.weight.shape) == _packed_shape(_INT, _H)
        assert gate_proj.weight.dtype == ir.DataType.UINT8
        assert gate_proj.scales is not None

    def test_shared_expert_gate_is_float_and_router_gate_unchanged(self):
        model = Qwen35MoECausalLMModel(_hybrid_moe_config(_olive_quant()))
        block = model.model.layers[1].mlp

        assert tuple(block.shared_expert_gate.weight.shape) == (1, _H)
        assert block.shared_expert_gate.weight.dtype != ir.DataType.UINT8
        assert not hasattr(block.shared_expert_gate, "scales")

        # Router gate behavior is untouched: still a plain float [E, H] linear.
        assert tuple(block.gate.weight.shape) == (_E, _H)
        assert block.gate.weight.dtype != ir.DataType.UINT8

    def test_routed_experts_remain_fused_qmoe(self):
        model = Qwen35MoECausalLMModel(_hybrid_moe_config(_olive_quant()))
        block = model.model.layers[1].mlp

        assert block.experts is None
        assert tuple(block.fc1_experts_weights.shape) == (_E, _FC1_OUT, _H * _BITS // 8)
        assert tuple(block.fc2_experts_weights.shape) == (_E, _H, _INT * _BITS // 8)

    def test_unquantized_hybrid_layer_is_unaffected(self):
        model = Qwen35MoECausalLMModel(_hybrid_moe_config(None))
        assert tuple(model.model.layers[0].linear_attn.in_proj_qkv.weight.shape) == (
            _QKV_OUT,
            _H,
        )
        assert tuple(model.model.layers[1].mlp.shared_expert_gate.weight.shape) == (1, _H)
        assert tuple(model.model.layers[1].self_attn.q_proj.weight.shape) == (_Q_OUT, _H)

    def test_vl_decoder_shares_the_mixed_layout(self):
        model = Qwen35MoEVL3ModelCausalLMModel(_hybrid_moe_config(_olive_quant(), vision=True))
        layers = model.decoder.model.layers

        assert tuple(layers[0].linear_attn.in_proj_qkv.weight.shape) == (_QKV_OUT, _H)
        assert layers[0].linear_attn.in_proj_qkv.weight.dtype != ir.DataType.UINT8
        assert tuple(layers[1].mlp.shared_expert_gate.weight.shape) == (1, _H)
        assert layers[1].mlp.shared_expert_gate.weight.dtype != ir.DataType.UINT8
        assert layers[1].self_attn.q_proj.weight.dtype == ir.DataType.UINT8

    def test_mixed_olive_state_dict_preprocesses_and_binds(self):
        """Post-#2630 Olive checkpoint (float deltanet + gate) binds cleanly."""
        config = _hybrid_moe_config(_olive_quant())
        model = Qwen35MoECausalLMModel(config)

        linear_attn = "model.language_model.layers.0.linear_attn."
        moe = "model.language_model.layers.1.mlp."
        attn = "model.language_model.layers.1.self_attn."
        state_dict: dict[str, torch.Tensor] = {
            # Float (unquantized by Olive) GatedDeltaNet projections.
            linear_attn + "in_proj_qkv.weight": torch.rand(_QKV_OUT, _H),
            linear_attn + "in_proj_z.weight": torch.rand(32, _H),
            linear_attn + "in_proj_b.weight": torch.rand(2, _H),
            linear_attn + "in_proj_a.weight": torch.rand(2, _H),
            linear_attn + "out_proj.weight": torch.rand(_H, 32),
            # Float shared-expert gate.
            moe + "shared_expert_gate.weight": torch.rand(1, _H),
            # Quantized ordinary attention + shared expert.
            attn + "q_proj.weight_qweight": torch.randint(
                0, 256, (_Q_OUT, _H * _BITS // 8), dtype=torch.uint8
            ),
            attn + "q_proj.weight_scales": torch.rand(_Q_OUT, _H // _BLK),
            moe + "shared_expert.gate_proj.weight_qweight": torch.randint(
                0, 256, (_INT, _H * _BITS // 8), dtype=torch.uint8
            ),
            moe + "shared_expert.gate_proj.weight_scales": torch.rand(_INT, _H // _BLK),
        }
        # Fused QMoE experts for layer 1.
        for key, value in _olive_expert_state_dict().items():
            state_dict[key.replace("layers.0.", "layers.1.")] = value

        out = model.preprocess_weights(state_dict)
        params = dict(model.named_parameters())

        # Every produced initializer binds to a parameter of matching shape.
        for key, value in out.items():
            assert key in params, key
            assert tuple(params[key].shape) == tuple(value.shape), key

        assert tuple(out["model.layers.0.linear_attn.in_proj_qkv.weight"].shape) == (
            _QKV_OUT,
            _H,
        )
        assert tuple(out["model.layers.1.mlp.shared_expert_gate.weight"].shape) == (1, _H)
        assert tuple(out["model.layers.1.self_attn.q_proj.weight"].shape) == _packed_shape(
            _Q_OUT, _H
        )
        assert "model.layers.1.mlp.fc1_experts_weights" in out
