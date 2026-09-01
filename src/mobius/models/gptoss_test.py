# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for native GPT-OSS MXFP4 QMoE export."""

from __future__ import annotations

from unittest.mock import patch

import onnx_ir as ir
import pytest
import torch

from mobius._build_context import build_context
from mobius._configs import (
    ArchitectureConfig,
    QuantizationConfig,
    QuantizedWeightFormat,
)
from mobius._execution_providers import EpCapabilities
from mobius._testing import make_config
from mobius._weight_utils import supported_qmoe_quantization
from mobius.models.base import linear_class_for_config
from mobius.models.gptoss import (
    GPTOSSCausalLMModel,
    _GptOssMoELayer,
    repack_gptoss_mxfp4_blocks,
)
from mobius.tasks import CausalLMTask

_E = 2
_H = 64
_I = 32
_LAYER = "model.layers.0.mlp"


def _mxfp4_quantization() -> QuantizationConfig:
    return QuantizationConfig(
        bits=4,
        group_size=32,
        quant_method="mxfp4",
        sym=True,
        weight_format=QuantizedWeightFormat.MXFP4,
    )


def _gptoss_config(*, native_mxfp4: bool = False, **overrides) -> ArchitectureConfig:
    options = dict(
        num_hidden_layers=1,
        hidden_size=_H,
        intermediate_size=_I,
        num_local_experts=_E,
        num_experts_per_tok=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        layer_types=["sliding_attention"],
        sliding_window=32,
        partial_rotary_factor=1.0,
        rope_interleave=False,
        attn_qkv_bias=True,
        attn_o_bias=True,
        quantization=_mxfp4_quantization() if native_mxfp4 else None,
    )
    options.update(overrides)
    return make_config(**options)


def _native_expert_state(layer: str = _LAYER) -> dict[str, torch.Tensor]:
    return {
        f"{layer}.experts.gate_up_proj_blocks": torch.randint(
            0, 256, (_E, 2 * _I, _H // 32, 16), dtype=torch.uint8
        ),
        f"{layer}.experts.gate_up_proj_scales": torch.randint(
            0, 255, (_E, 2 * _I, _H // 32), dtype=torch.uint8
        ),
        f"{layer}.experts.down_proj_blocks": torch.randint(
            0, 256, (_E, _H, _I // 32, 16), dtype=torch.uint8
        ),
        f"{layer}.experts.down_proj_scales": torch.randint(
            0, 255, (_E, _H, _I // 32), dtype=torch.uint8
        ),
        f"{layer}.experts.gate_up_proj_bias": torch.randn(_E, 2 * _I),
        f"{layer}.experts.down_proj_bias": torch.randn(_E, _H),
        f"{layer}.router.weight": torch.randn(_E, _H),
        f"{layer}.router.bias": torch.randn(_E),
    }


class TestRepackGPTOSSMXFP4:
    def test_preserves_every_nibble_in_qmoe_layout(self):
        blocks = torch.arange(1 * 4 * 2 * 16, dtype=torch.uint8).reshape(1, 4, 2, 16)

        packed = repack_gptoss_mxfp4_blocks(blocks)

        assert packed.shape == (1, 64, 2)
        checkpoint_codes = torch.empty(1, 4, 64, dtype=torch.uint8)
        checkpoint_codes[:, :, 0::2] = (blocks & 0x0F).reshape(1, 4, 32)
        checkpoint_codes[:, :, 1::2] = (blocks >> 4).reshape(1, 4, 32)
        qmoe_codes = torch.empty(1, 64, 4, dtype=torch.uint8)
        qmoe_codes[:, :, 0::2] = packed & 0x0F
        qmoe_codes[:, :, 1::2] = packed >> 4
        torch.testing.assert_close(qmoe_codes, checkpoint_codes.transpose(1, 2))

    def test_preserves_mapping_across_repack_chunk_boundaries(self):
        # 67 output pairs crosses the 64-pair production chunk boundary while
        # remaining tiny enough for a unit test.
        output_size = 134
        blocks = (
            torch.arange(_E * output_size * 2 * 16, dtype=torch.int64)
            .remainder(256)
            .to(torch.uint8)
            .reshape(_E, output_size, 2, 16)
        )

        packed = repack_gptoss_mxfp4_blocks(blocks)

        checkpoint_codes = torch.empty(_E, output_size, 64, dtype=torch.uint8)
        checkpoint_codes[:, :, 0::2] = (blocks & 0x0F).reshape(_E, output_size, 32)
        checkpoint_codes[:, :, 1::2] = (blocks >> 4).reshape(_E, output_size, 32)
        qmoe_codes = torch.empty(_E, 64, output_size, dtype=torch.uint8)
        qmoe_codes[:, :, 0::2] = packed & 0x0F
        qmoe_codes[:, :, 1::2] = packed >> 4
        torch.testing.assert_close(qmoe_codes, checkpoint_codes.transpose(1, 2))

    @pytest.mark.parametrize(
        ("blocks", "error", "message"),
        [
            (
                torch.zeros(1, 2, 1, 16, dtype=torch.int16),
                TypeError,
                "must be uint8",
            ),
            (
                torch.zeros(2, 1, 16, dtype=torch.uint8),
                ValueError,
                "rank 4",
            ),
            (
                torch.zeros(1, 2, 1, 15, dtype=torch.uint8),
                ValueError,
                "16 packed bytes",
            ),
            (
                torch.zeros(1, 3, 1, 16, dtype=torch.uint8),
                ValueError,
                "N must be even",
            ),
        ],
    )
    def test_rejects_malformed_blocks(self, blocks, error, message):
        with pytest.raises(error, match=message):
            repack_gptoss_mxfp4_blocks(blocks)


class TestGPTOSSNativeMXFP4Preprocess:
    def test_repacking_is_lossless_and_never_calls_hf_dequantizer(self):
        model = GPTOSSCausalLMModel(_gptoss_config(native_mxfp4=True))
        state = _native_expert_state()
        original_gate = state[f"{_LAYER}.experts.gate_up_proj_blocks"].clone()
        source_scales = state[f"{_LAYER}.experts.gate_up_proj_scales"]
        original_scales = source_scales.clone()
        source_scale_data_ptr = source_scales.data_ptr()
        source_expert_keys = {key for key in state if ".experts." in key}

        dequantizer = "transformers.integrations.mxfp4._convert_moe_packed_tensors"
        with patch(dequantizer) as convert:
            result = model.preprocess_weights(state)

        convert.assert_not_called()
        torch.testing.assert_close(
            result[f"{_LAYER}.fc1_experts_weights"],
            repack_gptoss_mxfp4_blocks(original_gate),
        )
        scales = result[f"{_LAYER}.fc1_scales"]
        assert scales.dtype == torch.float8_e8m0fnu
        torch.testing.assert_close(scales.view(torch.uint8), original_scales)
        assert scales.data_ptr() == source_scale_data_ptr
        assert result[f"{_LAYER}.fc1_global_scales"].dtype == torch.float32
        torch.testing.assert_close(result[f"{_LAYER}.fc1_global_scales"], torch.ones(_E))
        assert result[f"{_LAYER}.fc1_experts_bias"].shape == (_E, 2 * _I)
        assert result[f"{_LAYER}.fc2_experts_bias"].shape == (_E, _H)
        assert f"{_LAYER}.gate.weight" in result
        assert f"{_LAYER}.gate.bias" in result
        assert not any(
            ".experts." in key and key.endswith(("_blocks", "_scales")) for key in result
        )
        assert source_expert_keys.isdisjoint(state)
        assert source_expert_keys.isdisjoint(result)

    def test_pops_each_source_pair_before_allocating_its_destination(self):
        model = GPTOSSCausalLMModel(_gptoss_config(native_mxfp4=True))
        state = _native_expert_state()
        source_pairs = {
            state[block_key].data_ptr(): (
                block_key,
                block_key.removesuffix("_blocks") + "_scales",
            )
            for block_key in state
            if block_key.endswith("_blocks")
        }
        consumed_pairs: list[tuple[str, str]] = []

        def assert_consumed_then_repack(blocks):
            block_key, scale_key = source_pairs[blocks.data_ptr()]
            assert block_key not in state
            assert scale_key not in state
            consumed_pairs.append((block_key, scale_key))
            return repack_gptoss_mxfp4_blocks(blocks)

        with patch(
            "mobius.models.gptoss.repack_gptoss_mxfp4_blocks",
            side_effect=assert_consumed_then_repack,
        ):
            model.preprocess_weights(state)

        assert len(consumed_pairs) == 2

    def test_accepts_finite_e8m0_byte_boundaries(self):
        model = GPTOSSCausalLMModel(_gptoss_config(native_mxfp4=True))
        state = _native_expert_state()
        source_scales = state[f"{_LAYER}.experts.down_proj_scales"]
        source_scales.zero_()
        source_scales.reshape(-1)[-1] = 0xFE

        result = model.preprocess_weights(state)

        converted = result[f"{_LAYER}.fc2_scales"].view(torch.uint8)
        assert converted.reshape(-1)[0].item() == 0x00
        assert converted.reshape(-1)[-1].item() == 0xFE

    def test_rejects_e8m0_nan_byte(self):
        model = GPTOSSCausalLMModel(_gptoss_config(native_mxfp4=True))
        state = _native_expert_state()
        state[f"{_LAYER}.experts.down_proj_scales"].reshape(-1)[0] = 0xFF

        with pytest.raises(ValueError, match=r"0xff \(NaN\)"):
            model.preprocess_weights(state)

    def test_missing_block_scale_pair_fails_closed(self):
        model = GPTOSSCausalLMModel(_gptoss_config(native_mxfp4=True))
        state = _native_expert_state()
        del state[f"{_LAYER}.experts.down_proj_scales"]

        with pytest.raises(ValueError, match="matching scale tensor"):
            model.preprocess_weights(state)

    @pytest.mark.parametrize("failure", ["missing_bias", "malformed_bias", "invalid_scale"])
    def test_late_validation_failure_leaves_caller_state_unchanged(self, failure):
        model = GPTOSSCausalLMModel(_gptoss_config(native_mxfp4=True, num_hidden_layers=2))
        late_layer = "model.layers.1.mlp"
        state = {**_native_expert_state(), **_native_expert_state(late_layer)}
        if failure == "missing_bias":
            del state[f"{late_layer}.experts.down_proj_bias"]
        elif failure == "malformed_bias":
            state[f"{late_layer}.experts.down_proj_bias"] = torch.zeros(_E, _H - 1)
        else:
            state[f"{late_layer}.experts.down_proj_scales"].reshape(-1)[-1] = 0xFF
        original_objects = dict(state)
        original_values = {key: tensor.clone() for key, tensor in state.items()}

        with pytest.raises((TypeError, ValueError)):
            model.preprocess_weights(state)

        assert state.keys() == original_objects.keys()
        for key, tensor in state.items():
            assert tensor is original_objects[key]
            torch.testing.assert_close(tensor, original_values[key])

    def test_wholly_missing_expected_layer_fails_locally_without_mutation(self):
        model = GPTOSSCausalLMModel(_gptoss_config(native_mxfp4=True, num_hidden_layers=2))
        state = _native_expert_state()
        original = dict(state)

        with pytest.raises(ValueError, match=r"model\.layers\.1\.mlp"):
            model.preprocess_weights(state)

        assert state.keys() == original.keys()
        assert all(state[key] is tensor for key, tensor in original.items())

    @pytest.mark.parametrize(
        ("key", "replacement", "message"),
        [
            (
                f"{_LAYER}.experts.gate_up_proj_blocks",
                torch.zeros(_E, 2 * _I, _H // 32, 16, dtype=torch.int16),
                "must be uint8",
            ),
            (
                f"{_LAYER}.experts.gate_up_proj_blocks",
                torch.zeros(_E, 2 * _I - 1, _H // 32, 16, dtype=torch.uint8),
                "must have shape",
            ),
            (
                f"{_LAYER}.experts.gate_up_proj_scales",
                torch.zeros(_E, 2 * _I, _H // 32, dtype=torch.int16),
                "raw E8M0 bytes as uint8",
            ),
            (
                f"{_LAYER}.experts.gate_up_proj_scales",
                torch.zeros(_E, 2 * _I, 1, dtype=torch.uint8),
                "must have shape",
            ),
        ],
    )
    def test_malformed_block_or_scale_fails_closed(self, key, replacement, message):
        model = GPTOSSCausalLMModel(_gptoss_config(native_mxfp4=True))
        state = _native_expert_state()
        state[key] = replacement

        with pytest.raises((TypeError, ValueError), match=message):
            model.preprocess_weights(state)

    def test_packed_experts_without_native_descriptor_are_rejected(self):
        model = GPTOSSCausalLMModel(_gptoss_config())

        with pytest.raises(ValueError, match="does not declare native MXFP4"):
            model.preprocess_weights(_native_expert_state())


class TestGPTOSSNativeMXFP4Graph:
    def test_transformers_mxfp4_config_selects_typed_native_format(self):
        from transformers import GptOssConfig, Mxfp4Config

        config = ArchitectureConfig.from_transformers(
            GptOssConfig(quantization_config=Mxfp4Config())
        )

        assert config.quantization is not None
        assert config.quantization.quant_method == "mxfp4"
        assert config.quantization.bits == 4
        assert config.quantization.group_size == 32
        assert config.quantization.weight_format is QuantizedWeightFormat.MXFP4

    @pytest.mark.parametrize("build_dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_emits_one_qmoe_with_native_inputs_biases_and_semantics(self, build_dtype):
        config = _gptoss_config(native_mxfp4=True)
        module = GPTOSSCausalLMModel(config)

        with build_context(EpCapabilities(name="cuda"), build_dtype):
            graph = CausalLMTask().build(module, config)["model"].graph

        qmoe_nodes = [node for node in graph if node.op_type == "QMoE"]
        assert len(qmoe_nodes) == 1
        qmoe = qmoe_nodes[0]
        attrs = {name: attr.value for name, attr in qmoe.attributes.items()}
        assert qmoe.domain == "com.microsoft"
        assert attrs["quant_type"] == "fp4"
        assert "weights_prepacked" not in attrs
        assert attrs["expert_weight_bits"] == 4
        assert attrs["block_size"] == 32
        assert attrs["k"] == 1
        assert attrs["normalize_routing_weights"] == 1
        assert attrs["swiglu_fusion"] == 1
        assert attrs["activation_type"] == "swiglu"
        assert attrs["activation_alpha"] == pytest.approx(1.702)
        assert attrs["activation_beta"] == pytest.approx(1.0)
        assert attrs["swiglu_limit"] == pytest.approx(7.0)

        assert qmoe.inputs[2].name == f"{_LAYER}.fc1_experts_weights"
        assert qmoe.inputs[3].name == f"{_LAYER}.fc1_scales"
        assert qmoe.inputs[4].name == f"{_LAYER}.fc1_experts_bias"
        assert qmoe.inputs[5].name == f"{_LAYER}.fc2_experts_weights"
        assert qmoe.inputs[6].name == f"{_LAYER}.fc2_scales"
        assert qmoe.inputs[7].name == f"{_LAYER}.fc2_experts_bias"
        assert qmoe.inputs[15].name == f"{_LAYER}.fc1_global_scales"
        assert qmoe.inputs[16].name == f"{_LAYER}.fc2_global_scales"
        assert qmoe.inputs[3].dtype == ir.DataType.FLOAT8E8M0
        assert qmoe.inputs[6].dtype == ir.DataType.FLOAT8E8M0
        assert qmoe.inputs[15].dtype == ir.DataType.FLOAT
        assert qmoe.inputs[16].dtype == ir.DataType.FLOAT

        initializers = graph.initializers
        assert tuple(initializers[f"{_LAYER}.fc1_experts_weights"].shape) == (
            _E,
            _H,
            _I,
        )
        assert tuple(initializers[f"{_LAYER}.fc2_experts_weights"].shape) == (
            _E,
            _I,
            _H // 2,
        )
        # Router MatMul + additive bias feed QMoE directly; QMoE owns TopK and
        # selected-logit softmax normalization.
        assert qmoe.inputs[1].producer().op_type == "Reshape"
        assert any(node.op_type == "Add" for node in graph)

    @pytest.mark.parametrize(
        ("num_experts", "profile"),
        [(32, "gpt-oss-20b"), (128, "gpt-oss-120b")],
    )
    def test_official_profile_dimensions_only_allocate_parameter_metadata(
        self, num_experts, profile
    ):
        config = _gptoss_config(
            native_mxfp4=True,
            hidden_size=2880,
            intermediate_size=2880,
            num_local_experts=num_experts,
        )

        layer = _GptOssMoELayer(config)

        assert layer.experts is None, profile
        assert layer.fc1_experts_weights.const_value is None
        assert layer.fc1_experts_weights.shape == ir.Shape([num_experts, 2880, 2880])
        assert layer.fc2_experts_weights.shape == ir.Shape([num_experts, 2880, 1440])

    def test_explicit_qmoe_disable_has_no_silent_dense_fallback(self):
        with pytest.raises(ValueError, match="disable_qmoe=True"):
            _GptOssMoELayer(_gptoss_config(native_mxfp4=True, disable_qmoe=True))

    @pytest.mark.parametrize("target_ep", ["default", "cpu"])
    def test_non_cuda_ep_has_no_silent_dense_fallback(self, target_ep):
        config = _gptoss_config(native_mxfp4=True)
        module = GPTOSSCausalLMModel(config)

        with (
            build_context(EpCapabilities(name=target_ep), ir.DataType.FLOAT16),
            pytest.raises(
                NotImplementedError,
                match=r"--execution-provider cuda and --dtype f16",
            ),
        ):
            CausalLMTask().build(module, config)

    def test_cuda_float32_build_has_actionable_failure(self):
        config = _gptoss_config(native_mxfp4=True)
        module = GPTOSSCausalLMModel(config)

        with (
            build_context(EpCapabilities(name="cuda"), ir.DataType.FLOAT),
            pytest.raises(ValueError, match=r"--dtype f16 \(or bf16\)"),
        ):
            CausalLMTask().build(module, config)

    def test_generic_integer_factories_do_not_claim_mxfp4(self):
        quantization = _mxfp4_quantization()
        config = _gptoss_config(native_mxfp4=True)

        assert supported_qmoe_quantization(quantization) is None
        assert linear_class_for_config(config) is None


class TestGPTOSSUnquantized:
    def test_full_precision_experts_still_use_dense_loop_and_split_weights(self):
        config = _gptoss_config()
        model = GPTOSSCausalLMModel(config)
        layer = model.model.layers[0].mlp
        state = {
            f"{_LAYER}.experts.gate_up_proj": torch.randn(_E, _H, 2 * _I),
            f"{_LAYER}.experts.gate_up_proj_bias": torch.randn(_E, 2 * _I),
            f"{_LAYER}.experts.down_proj": torch.randn(_E, _I, _H),
            f"{_LAYER}.experts.down_proj_bias": torch.randn(_E, _H),
        }

        result = model.preprocess_weights(state)

        assert layer.experts is not None
        assert len(layer.experts) == _E
        for expert in range(_E):
            assert result[f"{_LAYER}.experts.{expert}.gate_proj.weight"].shape == (
                _I,
                _H,
            )
            assert result[f"{_LAYER}.experts.{expert}.up_proj.weight"].shape == (_I, _H)
            assert result[f"{_LAYER}.experts.{expert}.down_proj.weight"].shape == (
                _H,
                _I,
            )

    def test_non_expert_weights_pass_through(self):
        config = _gptoss_config()
        model = GPTOSSCausalLMModel(config)
        key = "model.layers.0.self_attn.q_proj.weight"
        state = {key: torch.randn(config.num_attention_heads * config.head_dim, _H)}

        result = model.preprocess_weights(state)

        assert result[key] is state[key]
