# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for native GPT-OSS MXFP4 QMoE export."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import onnx
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


def _make_gqa_parity_model(*, decode: bool) -> ir.Model:
    """Create a focused FP16 GQA graph for sink/window kernel parity."""
    batch = 2
    query_length = 1 if decode else 4
    past_length = 4 if decode else 0
    total_length = past_length + query_length
    num_heads = 2
    num_kv_heads = 1
    head_dim = 8

    def value(name: str, dtype: ir.DataType, shape: list[int]) -> ir.Value:
        return ir.Value(name=name, shape=ir.Shape(shape), type=ir.TensorType(dtype))

    inputs = [
        value("query", ir.DataType.FLOAT16, [batch, query_length, 16]),
        value("key", ir.DataType.FLOAT16, [batch, query_length, 8]),
        value("value", ir.DataType.FLOAT16, [batch, query_length, 8]),
        value(
            "past_key",
            ir.DataType.FLOAT16,
            [batch, num_kv_heads, past_length, head_dim],
        ),
        value(
            "past_value",
            ir.DataType.FLOAT16,
            [batch, num_kv_heads, past_length, head_dim],
        ),
        value("seqlens_k", ir.DataType.INT32, [batch]),
        value("total_sequence_length", ir.DataType.INT32, []),
        value("head_sink", ir.DataType.FLOAT16, [num_heads]),
    ]
    outputs = [
        value("output", ir.DataType.FLOAT16, [batch, query_length, 16]),
        value(
            "present_key",
            ir.DataType.FLOAT16,
            [batch, num_kv_heads, total_length, head_dim],
        ),
        value(
            "present_value",
            ir.DataType.FLOAT16,
            [batch, num_kv_heads, total_length, head_dim],
        ),
    ]
    node = ir.Node(
        domain="com.microsoft",
        op_type="GroupQueryAttention",
        inputs=[
            *inputs[:7],
            None,  # cos_cache
            None,  # sin_cache
            None,  # position_ids
            None,  # attention_bias
            inputs[7],
        ],
        outputs=outputs,
        attributes=ir.convenience.convert_attributes(
            {
                "num_heads": num_heads,
                "kv_num_heads": num_kv_heads,
                "scale": head_dim**-0.5,
                "local_window_size": 2,
                "do_rotary": 0,
            }
        ),
    )
    graph = ir.Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=[node],
        opset_imports={"": 24, "com.microsoft": 1},
        name="gptoss_gqa_parity",
    )
    return ir.Model(graph, ir_version=12)


def _require_usable_cuda_provider(ort) -> None:
    """Skip only when a separate valid CUDA probe cannot initialize the provider."""
    probe_input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1])
    probe_output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1])
    probe_weight = onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT, [1, 1], [1.0])
    probe = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["input", "weight"], ["output"])],
            "cuda_provider_probe",
            [probe_input],
            [probe_output],
            [probe_weight],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
        ir_version=10,
    )
    provider_errors = (
        ort.capi.onnxruntime_pybind11_state.Fail,
        ort.capi.onnxruntime_pybind11_state.RuntimeException,
        ort.capi.onnxruntime_pybind11_state.EPFail,
    )
    try:
        session = ort.InferenceSession(
            probe.SerializeToString(),
            providers=["CUDAExecutionProvider"],
        )
    except provider_errors as error:
        pytest.skip(f"CUDAExecutionProvider could not initialize valid probe: {error}")
    if session.get_providers()[0] != "CUDAExecutionProvider":
        pytest.skip("CUDAExecutionProvider could not initialize valid probe on this host")


def _manual_sink_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    head_sink: np.ndarray,
    *,
    past_key: np.ndarray,
    past_value: np.ndarray,
    sequence_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference sink attention for right-padded cache rows and a local window."""
    batch, query_length, _ = query.shape
    num_heads = 2
    num_kv_heads = 1
    head_dim = 8
    query_4d = query.astype(np.float32).reshape(batch, query_length, num_heads, head_dim)
    query_4d = query_4d.transpose(0, 2, 1, 3)
    key_4d = key.astype(np.float32).reshape(batch, query_length, num_kv_heads, head_dim)
    value_4d = value.astype(np.float32).reshape(batch, query_length, num_kv_heads, head_dim)
    key_4d = key_4d.transpose(0, 2, 1, 3)
    value_4d = value_4d.transpose(0, 2, 1, 3)
    past_key = past_key.astype(np.float32)
    past_value = past_value.astype(np.float32)
    present_width = past_key.shape[2] + query_length
    present_key = np.zeros((batch, num_kv_heads, present_width, head_dim), np.float32)
    present_value = np.zeros_like(present_key)
    output = np.empty((batch, num_heads, query_length, head_dim), np.float32)
    is_prefill = past_key.shape[2] == 0

    for batch_index, sequence_length in enumerate(sequence_lengths):
        past_length = 0 if is_prefill else int(sequence_length) - query_length
        present_key[batch_index, :, :past_length] = past_key[batch_index, :, :past_length]
        present_value[batch_index, :, :past_length] = past_value[batch_index, :, :past_length]
        present_key[batch_index, :, past_length : past_length + query_length] = key_4d[
            batch_index
        ]
        present_value[batch_index, :, past_length : past_length + query_length] = value_4d[
            batch_index
        ]

        expanded_key = np.repeat(present_key[batch_index], num_heads // num_kv_heads, axis=0)
        expanded_value = np.repeat(
            present_value[batch_index], num_heads // num_kv_heads, axis=0
        )
        causal_past_length = 0 if is_prefill else past_length
        for query_index in range(query_length):
            visible_length = min(
                causal_past_length + query_index + 1,
                int(sequence_length),
            )
            window_start = max(0, visible_length - 2)
            scores = np.einsum(
                "hd,hkd->hk",
                query_4d[batch_index, :, query_index],
                expanded_key[:, window_start:visible_length],
            )
            scores *= head_dim**-0.5
            combined = np.concatenate(
                [scores, head_sink.astype(np.float32)[:, None]],
                axis=-1,
            )
            probabilities = np.exp(combined - np.max(combined, axis=-1, keepdims=True))
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            output[batch_index, :, query_index] = np.einsum(
                "hk,hkd->hd",
                probabilities[:, :-1],
                expanded_value[:, window_start:visible_length],
            )

    output = output.transpose(0, 2, 1, 3).reshape(batch, query_length, -1)
    return output, present_key, present_value


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
        model = GPTOSSCausalLMModel(
            _gptoss_config(
                native_mxfp4=True,
                num_hidden_layers=2,
                layer_types=["sliding_attention", "full_attention"],
            )
        )
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
        model = GPTOSSCausalLMModel(
            _gptoss_config(
                native_mxfp4=True,
                num_hidden_layers=2,
                layer_types=["sliding_attention", "full_attention"],
            )
        )
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

    def test_explicit_dense_mxfp4_dequantizes_then_splits_experts(self):
        model = GPTOSSCausalLMModel(_gptoss_config())
        model._dequantize_mxfp4_checkpoint = True
        state = _native_expert_state()

        def dequantize(blocks, scales, *, dtype):
            assert blocks.dtype == torch.uint8
            assert scales.dtype == torch.uint8
            assert dtype == torch.bfloat16
            if blocks.shape[2] == _H // 32:
                return (
                    torch.arange(
                        _E * _H * 2 * _I,
                        dtype=torch.float32,
                    )
                    .reshape(_E, _H, 2 * _I)
                    .to(dtype)
                )
            return (
                torch.arange(
                    _E * _I * _H,
                    dtype=torch.float32,
                )
                .reshape(_E, _I, _H)
                .to(dtype)
            )

        with patch(
            "transformers.integrations.mxfp4._convert_moe_packed_tensors",
            side_effect=dequantize,
        ) as convert:
            result = model.preprocess_weights(state)

        assert convert.call_count == 2
        assert f"{_LAYER}.fc1_experts_weights" not in result
        assert f"{_LAYER}.fc2_experts_weights" not in result
        assert result[f"{_LAYER}.experts.0.gate_proj.weight"].shape == (_I, _H)
        assert result[f"{_LAYER}.experts.0.up_proj.weight"].shape == (_I, _H)
        assert result[f"{_LAYER}.experts.0.down_proj.weight"].shape == (_H, _I)
        assert result[f"{_LAYER}.experts.0.gate_proj.weight"].dtype == torch.bfloat16
        assert not any(key.endswith(("_blocks", "_scales")) for key in result)

    @pytest.mark.parametrize(
        "malformation",
        ["missing_projection", "wrong_dtype", "wrong_shape", "invalid_scale"],
    )
    def test_dense_preflight_rejects_before_mutation_or_converter(self, malformation):
        model = GPTOSSCausalLMModel(_gptoss_config())
        model._dequantize_mxfp4_checkpoint = True
        state = _native_expert_state()
        if malformation == "missing_projection":
            del state[f"{_LAYER}.experts.down_proj_blocks"]
        elif malformation == "wrong_dtype":
            key = f"{_LAYER}.experts.gate_up_proj_blocks"
            state[key] = state[key].to(torch.int16)
        elif malformation == "wrong_shape":
            key = f"{_LAYER}.experts.down_proj_scales"
            state[key] = state[key][..., :-1]
        else:
            state[f"{_LAYER}.experts.gate_up_proj_scales"].reshape(-1)[-1] = 0xFF
        original_objects = dict(state)
        original_values = {key: tensor.clone() for key, tensor in state.items()}

        with (
            patch("transformers.integrations.mxfp4._convert_moe_packed_tensors") as convert,
            pytest.raises((TypeError, ValueError)),
        ):
            model.preprocess_weights(state)

        convert.assert_not_called()
        assert state.keys() == original_objects.keys()
        for key, tensor in state.items():
            assert tensor is original_objects[key]
            torch.testing.assert_close(tensor, original_values[key])


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
        config = _gptoss_config(native_mxfp4=True, dtype=build_dtype)
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

    @pytest.mark.parametrize("build_dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_missing_layer_types_derives_model_builder_gqa_alternation(self, build_dtype):
        config = _gptoss_config(
            native_mxfp4=True,
            num_hidden_layers=4,
            layer_types=None,
            sliding_window=17,
            dtype=build_dtype,
            rope_type="yarn",
            rope_scaling={
                "rope_type": "yarn",
                "factor": 2.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 16,
            },
        )

        with build_context(EpCapabilities(name="cuda"), build_dtype):
            graph = CausalLMTask().build(GPTOSSCausalLMModel(config), config)["model"].graph

        gqa_nodes = [node for node in graph if node.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == config.num_hidden_layers
        assert all(node.domain == "com.microsoft" for node in gqa_nodes)
        for node in gqa_nodes:
            attrs = {name: attr.value for name, attr in node.attributes.items()}
            assert attrs["num_heads"] == config.num_attention_heads
            assert attrs["kv_num_heads"] == config.num_key_value_heads
            assert attrs["scale"] == pytest.approx(config.head_dim**-0.5)
            assert attrs["do_rotary"] == 0
            assert len(node.inputs) == 12
            assert node.inputs[0].producer().op_type == "RotaryEmbedding"
            assert node.inputs[0].producer().inputs[0].producer().op_type == "Add"
            assert node.inputs[1].producer().op_type == "RotaryEmbedding"
            assert node.inputs[1].producer().inputs[0].producer().op_type == "Add"
            assert node.inputs[2].producer().op_type == "Add"
            assert node.inputs[5].dtype == ir.DataType.INT32
            assert node.inputs[5].producer().inputs[0].producer().op_type == "Sub"
            assert node.inputs[6].dtype == ir.DataType.INT32
            assert node.inputs[6].producer().inputs[0].producer().op_type == "Gather"
            assert all(value is None for value in node.inputs[7:11])
            assert node.inputs[11] is not None
            assert node.inputs[11].dtype == build_dtype
            assert node.inputs[11].producer().inputs[0].name.endswith(".self_attn.sinks")

        for layer_id, node in enumerate(gqa_nodes):
            if layer_id % 2 == 0:
                assert node.attributes["local_window_size"].as_int() == 17
            else:
                assert "local_window_size" not in node.attributes
        assert sum(name.endswith(".self_attn.sinks") for name in graph.initializers) == 4
        cache_outputs = [value for value in graph.outputs if value.name != "logits"]
        assert all(
            value.producer().op_type == "GroupQueryAttention" for value in cache_outputs
        )
        # QMoE owns router normalization and GQA owns sink-aware attention:
        # no manual attention Softmax core can remain in this supported slice.
        assert not any(node.op_type == "Softmax" for node in graph)

    def test_native_and_unquantized_serialization_use_universal_ir12(self, tmp_path):
        packages = {}
        with build_context(EpCapabilities(name="cuda"), ir.DataType.FLOAT16):
            for name, native_mxfp4 in (("native", True), ("unquantized", False)):
                config = _gptoss_config(
                    native_mxfp4=native_mxfp4,
                    dtype=ir.DataType.FLOAT16,
                )
                packages[name] = CausalLMTask().build(GPTOSSCausalLMModel(config), config)

        for name, package in packages.items():
            assert package["model"].ir_version == 12
            output = tmp_path / name
            package.save(str(output), check_weights=False, progress_bar=False)
            proto = onnx.load_model(output / "model.onnx", load_external_data=False)

            assert proto.ir_version == 12
            assert {item.domain: item.version for item in proto.opset_import} == {
                "": 24,
                "com.microsoft": 1,
            }

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

    def test_dense_graph_uses_portable_experts_and_not_native_qmoe(self):
        config = _gptoss_config(dtype=ir.DataType.FLOAT16)
        module = GPTOSSCausalLMModel(config)

        with build_context(EpCapabilities(name="default"), ir.DataType.FLOAT16):
            graph = CausalLMTask().build(module, config)["model"].graph

        assert module.model.layers[0].mlp.experts is not None
        assert not any(node.op_type == "QMoE" for node in graph)
        assert not any(node.op_type == "GroupQueryAttention" for node in graph)

    @pytest.mark.parametrize("decode", [False, True], ids=["prefill", "decode"])
    def test_cuda_gqa_kernel_matches_manual_sink_attention_with_local_window(
        self, decode, tmp_path
    ):
        """Check the focused GQA kernel; production graph structure is tested above."""
        ort = pytest.importorskip("onnxruntime")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CUDAExecutionProvider is not available")
        _require_usable_cuda_provider(ort)
        model_path = tmp_path / f"gqa_{'decode' if decode else 'prefill'}.onnx"
        ir.save(_make_gqa_parity_model(decode=decode), model_path)
        session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider"],
        )
        assert session.get_providers()[0] == "CUDAExecutionProvider"

        rng = np.random.default_rng(5 if decode else 4)
        query_length = 1 if decode else 4
        query = rng.normal(size=(2, query_length, 16)).astype(np.float16)
        key = rng.normal(size=(2, query_length, 8)).astype(np.float16)
        value = rng.normal(size=(2, query_length, 8)).astype(np.float16)
        sequence_lengths = np.array([5, 3] if decode else [4, 2], dtype=np.int32)
        if not decode:
            query[1, 2:] = 0
            key[1, 2:] = 0
            value[1, 2:] = 0
        # Both heads have nonzero sink logits, including opposite signs.
        head_sink = np.array([-0.4, 0.7], dtype=np.float16)
        if decode:
            past_key = rng.normal(size=(2, 1, 4, 8)).astype(np.float16)
            past_value = rng.normal(size=(2, 1, 4, 8)).astype(np.float16)
            past_key[1, :, 2:] = 0
            past_value[1, :, 2:] = 0
        else:
            # ORT GenAI's emitted prefill ABI supplies cache inputs with a
            # zero-width sequence axis rather than omitting optional inputs.
            past_key = np.empty((2, 1, 0, 8), dtype=np.float16)
            past_value = np.empty((2, 1, 0, 8), dtype=np.float16)
        feeds = {
            "query": query,
            "key": key,
            "value": value,
            "past_key": past_key,
            "past_value": past_value,
            "seqlens_k": sequence_lengths - 1,
            "total_sequence_length": np.array(5 if decode else 4, dtype=np.int32),
            "head_sink": head_sink,
        }

        actual_output, actual_key, actual_value = session.run(None, feeds)
        expected_output, expected_key, expected_value = _manual_sink_attention(
            query,
            key,
            value,
            head_sink,
            past_key=past_key,
            past_value=past_value,
            sequence_lengths=sequence_lengths,
        )

        for batch_index, sequence_length in enumerate(sequence_lengths):
            valid_query_length = query_length if decode else int(sequence_length)
            np.testing.assert_allclose(
                actual_output[batch_index, :valid_query_length],
                expected_output[batch_index, :valid_query_length],
                atol=1e-3,
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                actual_key[batch_index, :, :sequence_length],
                expected_key[batch_index, :, :sequence_length],
                atol=1e-3,
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                actual_value[batch_index, :, :sequence_length],
                expected_value[batch_index, :, :sequence_length],
                atol=1e-3,
                rtol=1e-3,
            )


class TestGPTOSSUnquantized:
    def test_missing_layer_types_uses_model_builder_alternation(self):
        model = GPTOSSCausalLMModel(
            _gptoss_config(
                num_hidden_layers=4,
                layer_types=None,
                sliding_window=17,
            )
        )

        assert model.model._layer_types == [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]
        assert [layer.self_attn.local_window_size for layer in model.model.layers] == [
            17,
            -1,
            17,
            -1,
        ]

    def test_rejects_explicit_empty_layer_types(self):
        config = _gptoss_config(
            num_hidden_layers=2,
            layer_types=[],
        )

        with pytest.raises(
            ValueError,
            match=r"exactly one layer_types entry.*got 0 for num_hidden_layers=2",
        ):
            GPTOSSCausalLMModel(config)

    def test_rejects_layer_types_length_mismatch(self):
        config = _gptoss_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention"],
        )

        with pytest.raises(
            ValueError,
            match=r"exactly one layer_types entry.*got 1 for num_hidden_layers=2",
        ):
            GPTOSSCausalLMModel(config)

    def test_rejects_unknown_layer_type(self):
        config = _gptoss_config(layer_types=["slidding_attention"])

        with pytest.raises(
            ValueError,
            match=r"only supports sliding_attention and full_attention.*slidding_attention",
        ):
            GPTOSSCausalLMModel(config)

    @pytest.mark.parametrize("sliding_window", [None, 0, -1])
    def test_rejects_non_positive_sliding_window(self, sliding_window):
        config = _gptoss_config(sliding_window=sliding_window)

        with pytest.raises(ValueError, match="require a positive sliding_window"):
            GPTOSSCausalLMModel(config)

    def test_missing_layer_types_requires_positive_sliding_window(self):
        config = _gptoss_config(layer_types=None, sliding_window=None)

        with pytest.raises(ValueError, match="require a positive sliding_window"):
            GPTOSSCausalLMModel(config)

    def test_cuda_full_precision_fallback_keeps_decomposed_attention(self):
        config = _gptoss_config()

        with build_context(EpCapabilities(name="cuda"), ir.DataType.FLOAT16):
            graph = CausalLMTask().build(GPTOSSCausalLMModel(config), config)["model"].graph

        assert not any(node.op_type == "GroupQueryAttention" for node in graph)
        assert any(node.op_type == "Softmax" for node in graph)

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
