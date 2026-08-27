# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for compressed-tensors FP8/NVFP4 streaming reconstruction."""

from __future__ import annotations

import json
import types

import numpy as np
import onnx_ir as ir
import pytest
import safetensors.torch
import torch
from onnx_ir.passes.common import InlinePass

from mobius._builder import build_from_module
from mobius._configs import VisionConfig
from mobius._model_package import ModelPackage
from mobius._testing import create_test_builder, create_test_input, make_config
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import Linear
from mobius.integrations.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsError,
    stream_compressed_tensors_to_package,
)
from mobius.integrations.compressed_tensors import _loader as compressed_tensors_loader
from mobius.models.qwen35 import Qwen35VL3ModelCausalLMModel

_REVISION = "9e3d73c76eddb75f795cc24ccfbc5affe41c66bd"
_CONFIG_ETAG = "b6f6347774036d406eabed6cfffb0fec424ba075"
_INDEX_ETAG = "7608ff001dbfc8936318df32aaaaef7c8c9f340d"
_MODEL_LFS_SHA256 = "a0d562e22f1cdcf307ddb9d5967a46dc1deeabd485e7ec27312518b5f0c12974"
_MODEL_SIZE = 22_568_192_096
_MODEL_HEADER_LENGTH = 251_128
_INDEX_TOTAL_SIZE = 23_417_592_488
_OFFICIAL_BASE_REPOSITORY = "Qwen/Qwen3.8-27B"
_OFFICIAL_BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
_OFFICIAL_BASE_CONFIG_ETAG = "706cebd746c4b6f2b1d1f892630867acfdfd3df8"
_OFFICIAL_BASE_CONFIG_SIZE = 4_312
_NATIVE_ARTIFACT_REPOSITORY = "tlwu/Qwen3.8-27B-NVFP4-ONNX"
_NATIVE_ARTIFACT_REVISION = "16759da769f194f7bd760db3e2d2dc50652f7573"
_NATIVE_ARTIFACT_GRAPH_SHA256 = (
    "569740d8a0c83abee7e75948c420478406423cadc8dea55f37467d5d06f2d98b"
)
_NATIVE_ARTIFACT_GRAPH_SIZE = 1_057_196
_NATIVE_ARTIFACT_EXTERNAL_DATA_SIZE = 21_700_000_000

_HEADER_EVIDENCE = {
    "model.language_model.layers.3.self_attn.k_scale": ("BF16", [1], [2563940052, 2563940054]),
    "model.language_model.layers.3.self_attn.q_proj.weight": (
        "F8_E4M3",
        [12288, 5120],
        [7830205280, 7893119840],
    ),
    "model.language_model.layers.3.self_attn.q_proj.weight_scale": (
        "BF16",
        [12288, 1],
        [2563950806, 2563975382],
    ),
    "model.language_model.layers.55.mlp.gate_proj.weight_packed": (
        "U8",
        [17408, 2560],
        [21944038240, 21988602720],
    ),
    "model.language_model.layers.55.mlp.gate_proj.weight_scale": (
        "F8_E4M3",
        [17408, 320],
        [11406504800, 11412075360],
    ),
    "model.language_model.layers.55.mlp.gate_proj.weight_global_scale": (
        "F32",
        [1],
        [1236, 1240],
    ),
    "model.language_model.layers.56.mlp.gate_proj.weight": (
        "F8_E4M3",
        [17408, 5120],
        [11726975840, 11816104800],
    ),
    "model.language_model.layers.56.mlp.gate_proj.weight_scale": (
        "BF16",
        [17408, 1],
        [2589236084, 2589270900],
    ),
}


def _args(
    bits: int,
    strategy: str,
    *,
    dynamic: bool | str = False,
    group_size: int | None = None,
    scale_dtype: str | None = None,
) -> dict:
    return {
        "num_bits": bits,
        "type": "float",
        "strategy": strategy,
        "symmetric": True,
        "dynamic": dynamic,
        "group_size": group_size,
        "scale_dtype": scale_dtype,
    }


def _config(
    *,
    fp8_targets: list[str] | None = None,
    nvfp4_targets: list[str] | None = None,
    ignore: list[str] | None = None,
) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "version": "0.17.2.a20260716",
        "format": "mixed-precision",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "targets": fp8_targets or ["fp8"],
                "weights": _args(8, "channel"),
                "input_activations": _args(8, "token", dynamic=True),
            },
            "group_1": {
                "format": "nvfp4-pack-quantized",
                "targets": nvfp4_targets or ["nvfp4"],
                "weights": _args(
                    4,
                    "tensor_group",
                    group_size=16,
                    scale_dtype="torch.float8_e4m3fn",
                ),
                "input_activations": _args(
                    4,
                    "tensor_group",
                    dynamic="local",
                    group_size=16,
                    scale_dtype="torch.float8_e4m3fn",
                ),
            },
        },
        "ignore": ignore or [],
        "kv_cache_scheme": _args(8, "tensor"),
    }


def _linear_package(
    in_features: int = 16,
    out_features: int = 2,
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> ModelPackage:
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, in_features], dtype=dtype)
    output = Linear(in_features, out_features, bias=False)(op, x)
    output.name = "output"
    output.dtype = dtype
    graph.outputs.append(output)
    graph.initializers["weight"].type = ir.TensorType(dtype)
    return ModelPackage({"model": ir.Model(graph, ir_version=11)})


def _write_checkpoint(tmp_path, tensors: dict[str, torch.Tensor]) -> None:
    safetensors.torch.save_file(tensors, str(tmp_path / "model.safetensors"))


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Independent fixture encoder: adjacent low/high nibbles form one byte."""
    assert codes.dtype == torch.uint8 and codes.shape[1] % 2 == 0
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


class TestConfig:
    def test_last_layer_fp8_override_wins_before_general_nvfp4_regex(self):
        parsed = CompressedTensorsConfig.parse(
            _config(
                fp8_targets=[r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.gate_proj$"],
                nvfp4_targets=[r"re:.*mlp\.gate_proj$"],
            )
        )
        assert parsed.resolve("model.layers.55.mlp.gate_proj").format == (
            "nvfp4-pack-quantized"
        )
        assert parsed.resolve("model.layers.56.mlp.gate_proj").format == "float-quantized"

    def test_duplicate_target_uses_last_group_scheme(self):
        config = _config(fp8_targets=["same"], nvfp4_targets=["same"])
        parsed = CompressedTensorsConfig.parse(config)
        assert parsed.resolve("same").format == "nvfp4-pack-quantized"

    def test_exact_target_wins_over_matching_regex(self):
        parsed = CompressedTensorsConfig.parse(
            _config(
                fp8_targets=[r"re:.*q_proj$"],
                nvfp4_targets=["model.layers.0.q_proj"],
            )
        )

        assert parsed.resolve("model.layers.0.q_proj").format == ("nvfp4-pack-quantized")

    def test_matching_regexes_preserve_declaration_order(self):
        parsed = CompressedTensorsConfig.parse(
            _config(
                fp8_targets=[r"re:^model\..*q_proj$"],
                nvfp4_targets=[r"re:.*q_proj$"],
            )
        )

        assert parsed.resolve("model.layers.0.q_proj").format == "float-quantized"

    def test_attribute_config_is_normalized_recursively(self):
        config = _config()
        attribute_config = types.SimpleNamespace(
            **{
                key: (
                    types.SimpleNamespace(
                        **{
                            name: types.SimpleNamespace(**group)
                            for name, group in value.items()
                        }
                    )
                    if key == "config_groups"
                    else value
                )
                for key, value in config.items()
            }
        )

        parsed = CompressedTensorsConfig.parse(attribute_config)

        assert parsed.resolve("fp8").format == "float-quantized"

    def test_ignore_overrides_all_target_groups(self):
        parsed = CompressedTensorsConfig.parse(
            _config(
                fp8_targets=[r"re:.*q_proj$"],
                nvfp4_targets=[r"re:.*mlp.*$"],
                ignore=["model.layers.0.mlp.gate_proj"],
            )
        )
        assert parsed.resolve("model.layers.0.mlp.gate_proj") is None

    def test_output_activation_quantization_fails_closed(self):
        config = _config()
        config["config_groups"]["group_0"]["output_activations"] = _args(
            8, "token", dynamic=True
        )
        with pytest.raises(CompressedTensorsError, match="output_activations"):
            CompressedTensorsConfig.parse(config)

    def test_exact_parent_ignore_does_not_hide_targeted_child(self):
        parsed = CompressedTensorsConfig.parse(
            _config(
                fp8_targets=[r"re:.*linear_attn\.out_proj$"],
                nvfp4_targets=[r"re:.*mlp.*$"],
                ignore=["model.layers.0.linear_attn"],
            )
        )
        assert parsed.resolve("model.layers.0.linear_attn") is None
        assert parsed.resolve("model.layers.0.linear_attn.out_proj").format == (
            "float-quantized"
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("version", "0.18.0"),
            ("version", "0.17.20"),
            ("format", "nvfp4-pack-quantized"),
            ("quantization_status", "calibration"),
        ],
    )
    def test_unsupported_metadata_fails_closed(self, field, value):
        config = _config()
        config[field] = value
        with pytest.raises(CompressedTensorsError):
            CompressedTensorsConfig.parse(config)


class TestReconstruction:
    def test_nvfp4_reads_one_shard_handle_per_logical_tensor(self, tmp_path, monkeypatch):
        tensors = {
            "nvfp4.weight_packed": torch.zeros(2, 8, dtype=torch.uint8),
            "nvfp4.weight_scale": torch.ones(2, 1, dtype=torch.float8_e4m3fn),
            "nvfp4.weight_global_scale": torch.ones(1),
        }
        _write_checkpoint(tmp_path, tensors)
        path = str(tmp_path / "model.safetensors")
        key_index = {key: (path, list(tensor.shape), "") for key, tensor in tensors.items()}
        real_safe_open = compressed_tensors_loader.safe_open
        open_count = 0

        def counting_safe_open(*args, **kwargs):
            nonlocal open_count
            open_count += 1
            return real_safe_open(*args, **kwargs)

        monkeypatch.setattr(compressed_tensors_loader, "safe_open", counting_safe_open)

        compressed_tensors_loader._dequantize_nvfp4(key_index, "nvfp4", torch.float32)

        assert open_count == 1

    def test_nvfp4_ort_matmul_parity(self, tmp_path):
        # Codes and expected values are authored directly, not generated by the
        # implementation under test. Sign bit is 0x8; magnitude index is low 3 bits.
        codes = torch.tensor(
            [
                [2, 9, 6, 12, 0, 1, 3, 4, 5, 7, 10, 11, 13, 14, 15, 8],
                [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8],
            ],
            dtype=torch.uint8,
        )
        packed = _pack_codes(codes)
        scale = torch.tensor([[2.0], [0.5]], dtype=torch.float8_e4m3fn)
        _write_checkpoint(
            tmp_path,
            {
                "nvfp4.weight_packed": packed,
                "nvfp4.weight_scale": scale,
                "nvfp4.weight_global_scale": torch.tensor([2.0]),
                "nvfp4.input_global_scale": torch.tensor([4.0]),
            },
        )
        package = _linear_package()
        report = stream_compressed_tensors_to_package(
            package,
            str(tmp_path),
            CompressedTensorsConfig.parse(_config()),
            preprocess_weights=lambda state: {"weight": state["nvfp4.weight"]},
            fp8_kv_cache=True,
            keep_quantized=False,
        )

        magnitudes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        code_np = codes.numpy()
        expected_weight = magnitudes[code_np & 7]
        expected_weight = np.where((code_np & 8) != 0, -expected_weight, expected_weight)
        expected_weight *= np.array([[1.0], [0.25]], dtype=np.float32)
        x = np.arange(1, 17, dtype=np.float32)[None, :]
        actual = OnnxModelSession(package["model"], device="cpu").run({"x": x})["output"]

        np.testing.assert_allclose(actual, x @ expected_weight.T, rtol=0, atol=0)
        assert not report.output_is_nvfp4
        assert report.native_weight_formats == ()
        assert "not enabled" in report.kv_cache
        assert any(
            isinstance(initializer.const_value, ir.LazyTensor)
            for initializer in package["model"].graph.initializers.values()
        )

    def test_fp8_channel_scale_ort_matmul_parity(self, tmp_path):
        weight = torch.tensor(
            [[1.0, -2.0, 0.5, 3.0] * 4, [-1.0, 0.25, 2.0, -0.5] * 4],
            dtype=torch.float8_e4m3fn,
        )
        scale = torch.tensor([[0.25], [2.0]], dtype=torch.bfloat16)
        _write_checkpoint(
            tmp_path,
            {"fp8.weight": weight, "fp8.weight_scale": scale},
        )
        package = _linear_package()
        stream_compressed_tensors_to_package(
            package,
            str(tmp_path),
            CompressedTensorsConfig.parse(_config()),
            preprocess_weights=lambda state: {"weight": state["fp8.weight"]},
            keep_quantized=False,
        )

        x = np.arange(1, 17, dtype=np.float32)[None, :]
        expected = x @ (weight.float() * scale.float()).numpy().T
        actual = OnnxModelSession(package["model"], device="cpu").run({"x": x})["output"]
        np.testing.assert_array_equal(actual, expected)

    def test_native_nvfp4_storage_node_and_report(self, tmp_path):
        codes = torch.arange(16, dtype=torch.uint8).repeat(2, 1)
        packed = _pack_codes(codes)
        scale = torch.tensor([[0.5], [2.0]], dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor([4.0], dtype=torch.float32)
        input_global_scale = torch.tensor([8.0], dtype=torch.float32)
        _write_checkpoint(
            tmp_path,
            {
                "nvfp4.weight_packed": packed,
                "nvfp4.weight_scale": scale,
                "nvfp4.weight_global_scale": global_scale,
                "nvfp4.input_global_scale": input_global_scale,
            },
        )
        package = _linear_package(dtype=ir.DataType.FLOAT16)

        report = stream_compressed_tensors_to_package(
            package,
            str(tmp_path),
            CompressedTensorsConfig.parse(
                _config(
                    fp8_targets=["unused"],
                    nvfp4_targets=["nvfp4"],
                    ignore=["ignored.module"],
                )
            ),
            preprocess_weights=lambda state: {"weight": state["nvfp4.weight"]},
        )

        model = package["model"]
        native = next(
            node for node in model.graph if node.op_type == "MatMulBlockQuantizedFp4Weight"
        )
        assert native.domain == "com.microsoft"
        assert len(native.inputs) == 4
        assert native.attributes["block_size"].value == 16
        assert native.inputs[1].dtype == ir.DataType.UINT8
        assert native.inputs[1].shape == [2, 8]
        assert native.inputs[2].dtype == ir.DataType.UINT8
        assert native.inputs[2].shape == [2, 1]
        assert native.inputs[3].producer().op_type == "Div"
        assert "weight" not in model.graph.initializers
        assert all(
            not (
                initializer.dtype in {ir.DataType.FLOAT16, ir.DataType.FLOAT}
                and initializer.shape == [2, 16]
            )
            for initializer in model.graph.initializers.values()
        )
        assert report.storage_policy == "preserved-native-block"
        assert report.preserved_weight_formats == ("nvfp4-pack-quantized",)
        assert report.output_is_nvfp4
        assert "W4A16/W8A16" in report.activation_quantization
        assert "custom ONNX Runtime" in report.runtime_support
        metadata = json.loads(model.metadata_props["mobius.compressed_tensors.config"])
        assert metadata["groups"][0]["targets"] == ["unused"]
        assert metadata["groups"][1]["targets"] == ["nvfp4"]
        assert metadata["ignore"] == ["ignored.module"]

    def test_native_fp8_storage_remains_faithful(self, tmp_path):
        weight = torch.tensor(
            [[1.0, -2.0, 0.5, 3.0] * 4, [-1.0, 0.25, 2.0, -0.5] * 4],
            dtype=torch.float8_e4m3fn,
        )
        scale = torch.tensor([[0.25], [2.0]], dtype=torch.bfloat16)
        _write_checkpoint(
            tmp_path,
            {"fp8.weight": weight, "fp8.weight_scale": scale},
        )
        package = _linear_package(dtype=ir.DataType.FLOAT16)

        stream_compressed_tensors_to_package(
            package,
            str(tmp_path),
            CompressedTensorsConfig.parse(_config()),
            preprocess_weights=lambda state: {"weight": state["fp8.weight"]},
        )

        model = package["model"]
        native = next(
            node for node in model.graph if node.op_type == "MatMulBlockQuantizedFp8Weight"
        )
        assert native.domain == "com.microsoft"
        assert len(native.inputs) == 3
        assert native.attributes["block_size"].value == 16
        assert native.inputs[1].dtype == ir.DataType.FLOAT8E4M3FN
        assert native.inputs[1].shape == [2, 16]
        assert native.inputs[2].producer().op_type == "Cast"
        raw_scale = native.inputs[2].producer().inputs[0]
        assert raw_scale is not None
        assert raw_scale.dtype == ir.DataType.BFLOAT16
        assert raw_scale.shape == [2, 1]
        assert raw_scale.const_value.tobytes() == scale.view(torch.uint16).numpy().tobytes()
        assert (
            native.inputs[1].const_value.tobytes()
            == weight.view(torch.uint8).numpy().tobytes()
        )

    def test_native_nvfp4_function_inline_value_parity(self, tmp_path):
        codes = torch.tensor(
            [
                [2, 9, 6, 12, 0, 1, 3, 4, 5, 7, 10, 11, 13, 14, 15, 8],
                [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8],
            ],
            dtype=torch.uint8,
        )
        packed = _pack_codes(codes)
        scale = torch.tensor([[2.0], [0.5]], dtype=torch.float8_e4m3fn)
        _write_checkpoint(
            tmp_path,
            {
                "nvfp4.weight_packed": packed,
                "nvfp4.weight_scale": scale,
                "nvfp4.weight_global_scale": torch.tensor([2.0]),
                "nvfp4.input_global_scale": torch.tensor([4.0]),
            },
        )
        package = _linear_package(dtype=ir.DataType.FLOAT16)
        stream_compressed_tensors_to_package(
            package,
            str(tmp_path),
            CompressedTensorsConfig.parse(_config()),
            preprocess_weights=lambda state: {"weight": state["nvfp4.weight"]},
        )
        model = package["model"]

        InlinePass(
            criteria=lambda function: (
                function.domain == "com.microsoft"
                and function.name == "MatMulBlockQuantizedFp4Weight"
            )
        )(model)

        assert all(node.op_type != "MatMulBlockQuantizedFp4Weight" for node in model.graph)
        magnitudes = np.array(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=np.float32,
        )
        code_np = codes.numpy()
        expected_weight = magnitudes[code_np & 7]
        expected_weight = np.where(
            (code_np & 8) != 0,
            -expected_weight,
            expected_weight,
        )
        expected_weight *= np.array([[1.0], [0.25]], dtype=np.float32)
        x = np.arange(1, 17, dtype=np.float16)[None, :]
        actual = OnnxModelSession(model, device="cpu").run({"x": x})["output"]
        np.testing.assert_allclose(
            actual,
            x @ expected_weight.astype(np.float16).T,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_native_nvfp4_external_data_round_trip(self, tmp_path):
        n, k = 32, 64
        codes = torch.arange(n * k, dtype=torch.int64).reshape(n, k).to(torch.uint8)
        codes &= 0x0F
        packed = _pack_codes(codes)
        scale = torch.arange(n * (k // 16), dtype=torch.int64).reshape(n, k // 16)
        scale = (scale & 0x7E).to(torch.uint8).view(torch.float8_e4m3fn)
        global_scale = torch.tensor([3.0], dtype=torch.float32)
        input_global_scale = torch.tensor([5.0], dtype=torch.float32)
        _write_checkpoint(
            tmp_path,
            {
                "nvfp4.weight_packed": packed,
                "nvfp4.weight_scale": scale,
                "nvfp4.weight_global_scale": global_scale,
                "nvfp4.input_global_scale": input_global_scale,
            },
        )
        package = _linear_package(k, n, dtype=ir.DataType.FLOAT16)
        stream_compressed_tensors_to_package(
            package,
            str(tmp_path),
            CompressedTensorsConfig.parse(_config()),
            preprocess_weights=lambda state: {"weight": state["nvfp4.weight"]},
        )
        output = tmp_path / "output"

        package.save(
            str(output),
            external_data="onnx",
            progress_bar=False,
        )
        loaded = ModelPackage.load(str(output))
        initializers = loaded["model"].graph.initializers
        prefix = "weight.compressed_tensors"

        assert (
            initializers[f"{prefix}.weight"].const_value.tobytes() == packed.numpy().tobytes()
        )
        assert (
            initializers[f"{prefix}.weight_scale"].const_value.tobytes()
            == scale.view(torch.uint8).numpy().tobytes()
        )
        assert (
            initializers[f"{prefix}.weight_global_scale"].const_value.tobytes()
            == global_scale.numpy().tobytes()
        )
        assert loaded.quantization_report is None
        assert (
            loaded["model"].metadata_props["mobius.compressed_tensors.storage"]
            == "preserved-native-block"
        )
        assert (output / "model.onnx.data").stat().st_size < n * k * 2

    def test_shape_and_orphan_qparam_guards(self, tmp_path):
        _write_checkpoint(
            tmp_path,
            {
                "other.weight_packed": torch.zeros(2, 8, dtype=torch.uint8),
                "other.weight_scale": torch.ones(2, 1, dtype=torch.float8_e4m3fn),
                "other.weight_global_scale": torch.ones(1),
                "other.input_global_scale": torch.ones(1),
            },
        )
        with pytest.raises(CompressedTensorsError, match="Orphan quantization"):
            stream_compressed_tensors_to_package(
                _linear_package(),
                str(tmp_path),
                CompressedTensorsConfig.parse(_config()),
            )

    def test_targeted_fp8_rejects_dense_weight(self, tmp_path):
        _write_checkpoint(
            tmp_path,
            {"fp8.weight": torch.ones(2, 16, dtype=torch.bfloat16)},
        )
        with pytest.raises(CompressedTensorsError, match="dtype"):
            stream_compressed_tensors_to_package(
                _linear_package(),
                str(tmp_path),
                CompressedTensorsConfig.parse(_config()),
                preprocess_weights=lambda state: {"weight": state["fp8.weight"]},
            )

    def test_nvfp4_rejects_wrong_scale_shape(self, tmp_path):
        _write_checkpoint(
            tmp_path,
            {
                "nvfp4.weight_packed": torch.zeros(2, 8, dtype=torch.uint8),
                "nvfp4.weight_scale": torch.ones(2, 2, dtype=torch.float8_e4m3fn),
                "nvfp4.weight_global_scale": torch.ones(1),
                "nvfp4.input_global_scale": torch.ones(1),
            },
        )
        with pytest.raises(CompressedTensorsError, match="block-16 shape"):
            stream_compressed_tensors_to_package(
                _linear_package(),
                str(tmp_path),
                CompressedTensorsConfig.parse(_config()),
            )


def _tiny_qwen_vl():
    config = make_config(
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention", "full_attention"],
        tie_word_embeddings=False,
        dtype=ir.DataType.FLOAT16,
        vision=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=2,
            in_channels=3,
            out_hidden_size=32,
            num_position_embeddings=4,
        ),
        image_token_id=99,
        temporal_patch_size=1,
    )
    module = Qwen35VL3ModelCausalLMModel(config)
    return module, build_from_module(module, config, task=module.default_task)


def _source_name(target: str) -> str:
    if target.startswith("decoder.model."):
        return "model.language_model." + target[len("decoder.model.") :]
    if target.startswith("decoder.lm_head."):
        return target[len("decoder.") :]
    if target.startswith("embedding.embed_tokens."):
        return "model.language_model." + target[len("embedding.") :]
    if target.startswith("vision_encoder.visual."):
        source = "model.visual." + target[len("vision_encoder.visual.") :]
        source = source.replace(".mlp.up_proj.", ".mlp.linear_fc1.")
        return source.replace(".mlp.down_proj.", ".mlp.linear_fc2.")
    raise AssertionError(f"unexpected target weight {target}")


def test_qwen35_vlm_build_and_streaming_load(tmp_path):
    module, package = _tiny_qwen_vl()
    config = CompressedTensorsConfig.parse(
        _config(
            fp8_targets=[
                r"re:.*self_attn\.(q|k|v|o)_proj$",
                r"re:.*layers\.1\.mlp\.(gate|up|down)_proj$",
                r"re:.*lm_head$",
            ],
            nvfp4_targets=[r"re:.*mlp\.(gate|up|down)_proj$"],
            ignore=[r"re:model\.visual\..*"],
        )
    )
    tensors: dict[str, torch.Tensor] = {}
    for model in package.values():
        for target, initializer in model.graph.initializers.items():
            if initializer.const_value is not None:
                continue
            source = _source_name(target)
            if source in tensors or f"{source[:-7]}.weight_packed" in tensors:
                continue
            shape = tuple(int(dim) for dim in initializer.shape)
            module_name, _, parameter = source.rpartition(".")
            group = config.resolve(module_name) if parameter == "weight" else None
            if group is None:
                tensors[source] = torch.zeros(shape, dtype=torch.bfloat16)
            elif group.format == "float-quantized":
                tensors[source] = torch.ones(shape, dtype=torch.float8_e4m3fn)
                tensors[f"{module_name}.weight_scale"] = torch.ones(
                    shape[0], 1, dtype=torch.bfloat16
                )
            else:
                codes = torch.full(shape, 2, dtype=torch.uint8)
                tensors[f"{module_name}.weight_packed"] = _pack_codes(codes)
                tensors[f"{module_name}.weight_scale"] = torch.ones(
                    shape[0], shape[1] // 16, dtype=torch.float8_e4m3fn
                )
                tensors[f"{module_name}.weight_global_scale"] = torch.ones(1)
                tensors[f"{module_name}.input_global_scale"] = torch.ones(1)
    _write_checkpoint(tmp_path, tensors)

    report = stream_compressed_tensors_to_package(
        package,
        str(tmp_path),
        config,
        preprocess_weights=module.preprocess_weights,
    )

    assert report.assigned_initializers > 0
    assert all(
        initializer.const_value is not None
        for model in package.values()
        for initializer in model.graph.initializers.values()
    )
    assert not any(
        node.op_type == "Transpose"
        and node.inputs[0] is not None
        and node.inputs[0].is_initializer()
        for model in package.values()
        for node in model.graph
    )
    assert any(
        isinstance(initializer.const_value, ir.LazyTensor)
        for model in package.values()
        for initializer in model.graph.initializers.values()
    )


def test_pinned_qwen38_checkpoint_schema_evidence():
    evidence = {
        "repository": "unsloth/Qwen3.8-27B-NVFP4",
        "revision": _REVISION,
        "config_etag": _CONFIG_ETAG,
        "index_etag": _INDEX_ETAG,
        "model_lfs_sha256": _MODEL_LFS_SHA256,
        "model_size": _MODEL_SIZE,
        "model_header_length": _MODEL_HEADER_LENGTH,
        "index_total_size": _INDEX_TOTAL_SIZE,
        "header": _HEADER_EVIDENCE,
        "official_base": {
            "repository": _OFFICIAL_BASE_REPOSITORY,
            "revision": _OFFICIAL_BASE_REVISION,
            "config_etag": _OFFICIAL_BASE_CONFIG_ETAG,
            "config_size": _OFFICIAL_BASE_CONFIG_SIZE,
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
        },
    }
    # Stable JSON serialization makes accidental evidence edits obvious in review.
    encoded = json.dumps(evidence, sort_keys=True)
    assert _REVISION in encoded and _MODEL_LFS_SHA256 in encoded
    assert _OFFICIAL_BASE_REVISION in encoded
    assert _INDEX_TOTAL_SIZE == 23_417_592_488
    assert _HEADER_EVIDENCE["model.language_model.layers.55.mlp.gate_proj.weight_scale"][
        :2
    ] == ("F8_E4M3", [17408, 320])
    assert _HEADER_EVIDENCE["model.language_model.layers.56.mlp.gate_proj.weight"][:2] == (
        "F8_E4M3",
        [17408, 5120],
    )


def test_pinned_native_nvfp4_onnx_abi_evidence():
    evidence = {
        "repository": _NATIVE_ARTIFACT_REPOSITORY,
        "revision": _NATIVE_ARTIFACT_REVISION,
        "text_onnx_sha256": _NATIVE_ARTIFACT_GRAPH_SHA256,
        "text_onnx_size": _NATIVE_ARTIFACT_GRAPH_SIZE,
        "external_data_size": _NATIVE_ARTIFACT_EXTERNAL_DATA_SIZE,
        "opset_imports": {"": 21, "com.microsoft": 1},
        "nvfp4": {
            "op": "com.microsoft::MatMulBlockQuantizedFp4Weight",
            "inputs": ["A", "B", "weight_scale", "weight_scale_2"],
            "block_size": 16,
            "packed_shape": ["N", "K/2"],
            "scale_shape": ["N", "K/16"],
            "global_shape": [1],
            "semantics": "low-nibble-first E2M1 * raw-E4M3 * weight_scale_2",
        },
        "fp8": {
            "op": "com.microsoft::MatMulBlockQuantizedFp8Weight",
            "weight_dtype": "FLOAT8E4M3FN",
            "scale_shape": ["N", 1],
            "block_size": "K",
        },
        "default_compute": "W4A16/W8A16",
    }

    encoded = json.dumps(evidence, sort_keys=True)
    assert _NATIVE_ARTIFACT_REVISION in encoded
    assert _NATIVE_ARTIFACT_GRAPH_SHA256 in encoded
    assert evidence["nvfp4"]["inputs"] == [
        "A",
        "B",
        "weight_scale",
        "weight_scale_2",
    ]
    assert evidence["default_compute"] == "W4A16/W8A16"
