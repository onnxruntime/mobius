# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for compressed-tensors FP8/NVFP4 streaming reconstruction."""

from __future__ import annotations

import json

import numpy as np
import onnx_ir as ir
import pytest
import safetensors.torch
import torch

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
from mobius.models.qwen35 import Qwen35VL3ModelCausalLMModel

_REVISION = "9e3d73c76eddb75f795cc24ccfbc5affe41c66bd"
_CONFIG_ETAG = "b6f6347774036d406eabed6cfffb0fec424ba075"
_INDEX_ETAG = "7608ff001dbfc8936318df32aaaaef7c8c9f340d"
_MODEL_LFS_SHA256 = "a0d562e22f1cdcf307ddb9d5967a46dc1deeabd485e7ec27312518b5f0c12974"
_MODEL_SIZE = 22_568_192_096
_MODEL_HEADER_LENGTH = 251_128
_INDEX_TOTAL_SIZE = 23_417_592_488

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


def _linear_package() -> ModelPackage:
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 16])
    output = Linear(16, 2, bias=False)(op, x)
    output.name = "output"
    graph.outputs.append(output)
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

    def test_ignore_overrides_all_target_groups(self):
        parsed = CompressedTensorsConfig.parse(
            _config(
                fp8_targets=[r"re:.*q_proj$"],
                nvfp4_targets=[r"re:.*mlp.*$"],
                ignore=["model.layers.0.mlp.gate_proj"],
            )
        )
        assert parsed.resolve("model.layers.0.mlp.gate_proj") is None

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
        assert isinstance(package["model"].graph.initializers["weight"].const_value, ir.LazyTensor)

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
        )

        x = np.arange(1, 17, dtype=np.float32)[None, :]
        expected = x @ (weight.float() * scale.float()).numpy().T
        actual = OnnxModelSession(package["model"], device="cpu").run({"x": x})["output"]
        np.testing.assert_array_equal(actual, expected)

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
    }
    # Stable JSON serialization makes accidental evidence edits obvious in review.
    encoded = json.dumps(evidence, sort_keys=True)
    assert _REVISION in encoded and _MODEL_LFS_SHA256 in encoded
    assert _INDEX_TOTAL_SIZE == 23_417_592_488
    assert _HEADER_EVIDENCE[
        "model.language_model.layers.55.mlp.gate_proj.weight_scale"
    ][:2] == ("F8_E4M3", [17408, 320])
    assert _HEADER_EVIDENCE[
        "model.language_model.layers.56.mlp.gate_proj.weight"
    ][:2] == ("F8_E4M3", [17408, 5120])
