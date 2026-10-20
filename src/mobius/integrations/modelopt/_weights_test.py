# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Format-faithful mixed ModelOpt reconstruction and fail-closed loader regressions."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import safetensors.torch
import torch

from mobius.integrations._weight_loading import (
    StreamingWeightPlan,
    StreamingWeightSource,
    stream_preprocessed_safetensors_to_model,
)
from mobius.integrations.modelopt._config import ModelOptConfig
from mobius.integrations.modelopt._weights import reconstruct_modelopt_weight


def _mixed_config():
    return {
        "quant_method": "modelopt",
        "config_groups": {
            "fp8": {
                "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                "targets": ["backbone.layers.0.mixer.in_proj"],
            },
            "fp4": {
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": ["backbone.layers.1.mixer.experts.0.up_proj", "lm_head"],
            },
        },
    }


def test_exact_mixed_group_layout():
    config = ModelOptConfig.parse(_mixed_config())
    assert config.modules["backbone.layers.0.mixer.in_proj"] == "modelopt_fp8"
    assert config.modules["lm_head"] == "modelopt_nvfp4"
    assert config.activation_modules == frozenset({"backbone.layers.0.mixer.in_proj"})
    assert config.source_format == "modelopt-mixed-fp8-nvfp4"


@pytest.mark.parametrize("unknown", ["format", "strategy"])
def test_unknown_modelopt_fields_are_not_silently_ignored(unknown):
    raw = _mixed_config()
    raw[unknown] = "unsupported"
    with pytest.raises(NotImplementedError, match="configuration fields"):
        ModelOptConfig.parse(raw)
    raw = _mixed_config()
    raw["config_groups"]["fp4"][unknown] = "unsupported"
    with pytest.raises(NotImplementedError, match="group fields"):
        ModelOptConfig.parse(raw)


def test_redundant_format_and_cache_declarations_must_agree():
    raw = _mixed_config()
    raw["quantized_layers"] = {"lm_head": {"quant_algo": "FP8"}}
    with pytest.raises(ValueError, match="disagree"):
        ModelOptConfig.parse(raw)
    raw = _mixed_config()
    raw["kv_cache_scheme"] = {"dynamic": True, "num_bits": 8, "type": "float"}
    with pytest.raises(NotImplementedError, match="KV-cache"):
        ModelOptConfig.parse(raw)


@pytest.mark.parametrize(
    "change", ["group-size", "dynamic", "integer", "duplicate", "wildcard"]
)
def test_reject_unknown_or_ambiguous_group_layout(change):
    raw = _mixed_config()
    group = raw["config_groups"]["fp4"]
    if change == "group-size":
        group["weights"]["group_size"] = 32
    elif change == "dynamic":
        group["weights"]["dynamic"] = True
    elif change == "integer":
        group["weights"]["type"] = "int"
    elif change == "duplicate":
        group["targets"].append("lm_head")
    else:
        group["targets"] = ["backbone.layers.*"]
    with pytest.raises((ValueError, NotImplementedError)):
        ModelOptConfig.parse(raw)


def test_fp4_all_codes_distinct_rows_blocks_and_chunk_boundary():
    # E2M1 codes 0..15 exercise signs, nibble order and zero. 257 rows cross
    # the bounded-row materialization boundary; scales differ in every block.
    codes = torch.arange(16, dtype=torch.uint8).repeat(257, 2)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = torch.tensor([0.5, 2.0]).repeat(257, 1).to(torch.float8_e4m3fn)
    scales[256] = torch.tensor([1.0, 4.0]).to(torch.float8_e4m3fn)
    output = reconstruct_modelopt_weight(
        packed, scales, torch.tensor(0.25), name="experts.0.up_proj"
    )
    magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    expected = magnitude[(codes & 7).long()]
    expected = torch.where((codes & 8).bool(), -expected, expected)
    expected *= scales.float().repeat_interleave(16, dim=1) * 0.25
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, expected.bfloat16(), rtol=0, atol=0)
    assert torch.equal(torch.signbit(output), torch.signbit(expected))


def test_fp8_single_rounding_with_nonunit_scalar():
    weight = torch.tensor([[1.0, -2.0, 1.5, 0.25]]).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.123456, dtype=torch.float32)
    output = reconstruct_modelopt_weight(weight, scale, None, name="mamba.in_proj")
    torch.testing.assert_close(output, (weight.float() * scale).bfloat16(), rtol=0, atol=0)


def test_zero_nvfp4_block_scale_reconstructs_zero_block():
    packed = torch.full((1, 16), 0x92, dtype=torch.uint8)
    blocks = torch.tensor([[0.0, 2.0]]).to(torch.float8_e4m3fn)
    output = reconstruct_modelopt_weight(packed, blocks, torch.tensor(0.5), name="expert")
    assert torch.equal(output[:, :16], torch.zeros(1, 16, dtype=torch.bfloat16))
    torch.testing.assert_close(
        output[:, 16:],
        torch.tensor([[1.0, -0.5] * 8], dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_reject_nonpositive_nonfinite_scale(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        reconstruct_modelopt_weight(
            torch.ones(1, 16).to(torch.float8_e4m3fn),
            torch.tensor(bad),
            None,
            name="mamba",
        )


def test_reject_scale_row_broadcast():
    with pytest.raises(ValueError, match="shape mismatch"):
        reconstruct_modelopt_weight(
            torch.full((2, 8), 0x22, dtype=torch.uint8),
            torch.ones(1, 1).to(torch.float8_e4m3fn),
            torch.tensor(1.0),
            name="expert",
        )


def _stream_fixture(tmp_path, *, mutate=None):
    tensors = {
        "fp8.weight": torch.tensor([[1.0, -2.0] * 8]).to(torch.float8_e4m3fn),
        "fp8.weight_scale": torch.tensor(0.25),
        "fp4.weight": torch.full((2, 16), 0x92, dtype=torch.uint8),
        "fp4.weight_scale": torch.tensor([[0.5, 2.0], [1.0, 4.0]]).to(torch.float8_e4m3fn),
        "fp4.weight_scale_2": torch.tensor(0.5),
    }
    if mutate is not None:
        mutate(tensors)
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(tensors, path)
    initializers = {
        name: ir.Value(
            name=name, shape=ir.Shape(shape), type=ir.TensorType(ir.DataType.BFLOAT16)
        )
        for name, shape in {"fp8.weight": [1, 16], "fp4.weight": [2, 32]}.items()
    }
    model = ir.Model(
        ir.Graph([], [], nodes=[], initializers=initializers.values()), ir_version=11
    )
    plan = StreamingWeightPlan(
        {
            "fp8.weight": StreamingWeightSource(
                "fp8.weight", "modelopt_fp8", "fp8.weight_scale"
            ),
            "fp4.weight": StreamingWeightSource(
                "fp4.weight",
                "modelopt_nvfp4",
                "fp4.weight_scale",
                global_scale_name="fp4.weight_scale_2",
            ),
        }
    )
    return tensors, path, model, plan


def test_streaming_mixed_weights_materialize_and_report_dense(tmp_path):
    tensors, _path, model, plan = _stream_fixture(tmp_path)
    report = stream_preprocessed_safetensors_to_model(model, str(tmp_path), lambda *_: plan)
    assert report["output_weight_format"] == "dense"
    assert report["native_fp8"] is False
    for name in ("fp8.weight", "fp4.weight"):
        expected = reconstruct_modelopt_weight(
            tensors[name],
            tensors[name.replace(".weight", ".weight_scale")],
            tensors.get(name.replace(".weight", ".weight_scale_2")),
            name=name,
        )
        actual = model.graph.initializers[name].const_value.numpy().astype(np.float32)
        np.testing.assert_array_equal(actual, expected.float().numpy())
    # Exercise serialization and reload, not just a success-shaped lazy binding.
    out = tmp_path / "dense.onnx"
    ir.save(model, out)
    loaded = ir.load(out)
    assert all(x.dtype == ir.DataType.BFLOAT16 for x in loaded.graph.initializers.values())


@pytest.mark.parametrize(
    "change", ["missing-scale", "row-broadcast", "integer-scale", "extra"]
)
def test_streaming_preflight_rejects_before_any_binding(tmp_path, change):
    def mutate(tensors):
        if change == "missing-scale":
            del tensors["fp4.weight_scale_2"]
        elif change == "row-broadcast":
            tensors["fp4.weight_scale"] = torch.ones(1, 2).to(torch.float8_e4m3fn)
        elif change == "integer-scale":
            tensors["fp8.weight_scale"] = torch.tensor(1, dtype=torch.int64)
        else:
            tensors["unknown.weight"] = torch.ones(1)

    _tensors, _path, model, plan = _stream_fixture(tmp_path, mutate=mutate)
    with pytest.raises(ValueError):
        stream_preprocessed_safetensors_to_model(model, str(tmp_path), lambda *_: plan)
    assert all(x.const_value is None for x in model.graph.initializers.values())


def test_lazy_reconstruction_rechecks_source_header(tmp_path):
    tensors, path, model, plan = _stream_fixture(tmp_path)
    stream_preprocessed_safetensors_to_model(model, str(tmp_path), lambda *_: plan)
    tensors["fp4.weight_scale_2"] = torch.ones(2)
    safetensors.torch.save_file(tensors, path)
    with pytest.raises(ValueError, match="changed after indexing"):
        model.graph.initializers["fp4.weight"].const_value.numpy()


def test_fp8_descriptor_cannot_borrow_an_nvfp4_global_scale(tmp_path):
    _tensors, _path, model, original = _stream_fixture(tmp_path)
    plan = StreamingWeightPlan(
        {
            **original.targets,
            "fp8.weight": StreamingWeightSource(
                "fp8.weight",
                "modelopt_fp8",
                "fp8.weight_scale",
                global_scale_name="fp4.weight_scale_2",
            ),
        }
    )
    with pytest.raises(ValueError, match="must not declare"):
        stream_preprocessed_safetensors_to_model(model, str(tmp_path), lambda *_: plan)
    assert all(x.const_value is None for x in model.graph.initializers.values())
