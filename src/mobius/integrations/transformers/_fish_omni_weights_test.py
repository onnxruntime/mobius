# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import dataclasses
import json

import onnx_ir as ir
import pytest
import torch
from _test_configs import FISH_OMNI_CONFIG
from safetensors.torch import save_file

from mobius import build_from_module
from mobius.__main__ import main
from mobius._configs.fish_omni import DRAMA3_REVISION
from mobius._model_package import ModelPackage
from mobius.integrations.transformers import _builder
from mobius.integrations.transformers._config_resolver import _dict_to_pretrained_config
from mobius.integrations.transformers._fish_omni_weights import (
    _source_name,
    build_fish_omni_streaming_plan,
    stream_fish_omni_safetensors_to_package,
)
from mobius.models import FishOmniModel
from mobius.models.fish_omni_test import tiny_config


def _checkpoint(package):
    torch.manual_seed(19)
    return {
        _source_name(name): torch.randn(tuple(value.shape), dtype=torch.float32) * 0.02
        for model in package.values()
        for name, value in model.graph.initializers.items()
        if value.const_value is None
    }


def test_fish_omni_streams_all_components(tmp_path):
    config = tiny_config()
    package = build_from_module(FishOmniModel(config), config, task="fish-omni")
    weights = _checkpoint(package)
    save_file(weights, tmp_path / "model.safetensors")
    stream_fish_omni_safetensors_to_package(package, str(tmp_path), config, revision=None)
    for model in package.values():
        for name, value in model.graph.initializers.items():
            assert value.const_value is not None, name
            if name.startswith(
                (
                    "embedding.",
                    "decoder.model.layers.",
                    "decoder.lm_head.",
                    "audio_decoder.shared_layers.",
                    "audio_decoder.residual_layers.",
                    "audio_decoder.codebook_heads.",
                )
            ):
                torch.testing.assert_close(
                    torch.from_numpy(value.const_value.numpy()), weights[_source_name(name)]
                )
    assert package.weight_loading_report["native_fp8"] is False
    assert package.weight_loading_report["codec_included"] is False


@pytest.mark.parametrize("failure", ["unexpected", "missing", "shape"])
def test_fish_omni_rejects_bad_checkpoint_transactionally(tmp_path, failure):
    config = tiny_config()
    package = build_from_module(FishOmniModel(config), config, task="fish-omni")
    weights = _checkpoint(package)
    if failure == "unexpected":
        weights["unknown.weight"] = torch.ones(1)
    elif failure == "missing":
        weights.pop("lm_head.weight")
    else:
        weights["lm_head.weight"] = torch.ones(1)
    save_file(weights, tmp_path / "model.safetensors")
    unbound = [
        value
        for model in package.values()
        for value in model.graph.initializers.values()
        if value.const_value is None
    ]
    with pytest.raises(ValueError):
        stream_fish_omni_safetensors_to_package(package, str(tmp_path), config, revision=None)
    assert all(value.const_value is None for value in unbound)


def test_fish_omni_fp8_classification():
    name = "decoder.model.layers.0.mlp.experts.0.gate_proj.weight"
    value = ir.Value(
        name=name, shape=ir.Shape([128, 128]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    source = _source_name(name)
    index = {source: ("unused", [128, 128], "F8_E4M3")}
    plan = build_fish_omni_streaming_plan(
        index, {name: value}, dataclasses.replace(tiny_config(), fp8_experts=True)
    )
    assert plan.targets[name].mode == "fp8_block_128"
    assert plan.targets[name].reconstruction_dtype == torch.float32
    index[source] = ("unused", [128, 128], "BF16")
    with pytest.raises(ValueError, match="F8_E4M3"):
        build_fish_omni_streaming_plan(
            index, {name: value}, dataclasses.replace(tiny_config(), fp8_experts=True)
        )


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.BFLOAT16])
def test_fish_omni_fp8_scales_are_reconstructed_in_fp32(tmp_path, dtype):
    config = dataclasses.replace(tiny_config(), fp8_experts=True, dtype=dtype)
    name = "decoder.model.layers.0.mlp.experts.0.gate_proj.weight"
    value = ir.Value(name=name, shape=ir.Shape([128, 128]), type=ir.TensorType(dtype))
    model = ir.Model(ir.Graph([], [], nodes=[], initializers=[value]), ir_version=12)
    package = ModelPackage({"decoder": model}, config=config)
    source = _source_name(name)
    weight = torch.linspace(-1, 1, 128 * 128).reshape(128, 128).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.123456]], dtype=torch.float32)
    save_file(
        {
            source: weight,
            source.removesuffix(".weight") + ".weight_scale_inv": scale,
        },
        tmp_path / "model.safetensors",
    )
    stream_fish_omni_safetensors_to_package(package, str(tmp_path), config, revision=None)
    expected = weight.float() * scale
    if dtype == ir.DataType.BFLOAT16:
        expected = expected.to(torch.bfloat16).float()
    actual = torch.tensor(value.const_value.numpy().astype("float32"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fish_omni_cuda_banks_preserve_fp8_reconstruction_and_expert_order(tmp_path):
    config = dataclasses.replace(tiny_config(), fp8_experts=True)
    name = "decoder.model.layers.0.mlp.packed_experts.gate_proj.weight"
    count = config.num_local_experts
    value = ir.Value(
        name=name, shape=ir.Shape([count, 128, 128]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    package = ModelPackage(
        {"decoder": ir.Model(ir.Graph([], [], nodes=[], initializers=[value]), ir_version=12)},
        config=config,
    )
    tensors = {}
    expected = []
    for expert in range(count):
        source = f"model.language_model.layers.0.mlp.experts.{expert}.gate_proj.weight"
        weight = torch.linspace(-1, 1, 128 * 128).reshape(128, 128).to(torch.float8_e4m3fn)
        scale = torch.tensor([[0.123456 * (expert + 1)]], dtype=torch.float32)
        tensors[source] = weight
        tensors[source.removesuffix(".weight") + ".weight_scale_inv"] = scale
        expected.append(weight.float() * scale)
    save_file(tensors, tmp_path / "model.safetensors")
    stream_fish_omni_safetensors_to_package(package, str(tmp_path), config, revision=None)
    torch.testing.assert_close(
        torch.from_numpy(value.const_value.numpy()), torch.stack(expected), rtol=0, atol=0
    )


@pytest.mark.parametrize("revision", [None, "explicit-revision"])
def test_fish_omni_revision_reaches_first_config_read(monkeypatch, revision):
    seen = []

    def load(model_id, **kwargs):
        seen.append(kwargs["revision"])
        return _dict_to_pretrained_config(copy.deepcopy(FISH_OMNI_CONFIG)), True

    monkeypatch.setattr(_builder, "_load_transformers_config", load)
    package = _builder.build_transformers_model(
        "fishaudio/drama-3", revision=revision, load_weights=False
    )
    expected = revision or DRAMA3_REVISION
    assert seen == [expected]
    assert all(
        m.metadata_props["mobius.source_revision"] == expected for m in package.values()
    )


def test_fish_omni_requires_explicit_dequantization_before_weight_io(monkeypatch):
    raw = copy.deepcopy(FISH_OMNI_CONFIG)
    raw["quantization_config"] = {
        "quant_method": "bsgemm-moe",
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    }
    monkeypatch.setattr(
        _builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (_dict_to_pretrained_config(raw), True),
    )
    with pytest.raises(ValueError, match="--dequantize"):
        _builder.build_transformers_model("fishaudio/drama-3")


@pytest.mark.parametrize(
    "task,ep",
    [
        ("fish-omni", "default"),
        ("fish-omni-generation", "default"),
        ("fish-omni-generation", "cuda"),
    ],
)
def test_fish_omni_cli_persists_three_weighted_models(tmp_path, task, ep):
    config = tiny_config()
    package = build_from_module(FishOmniModel(config), config, task="fish-omni")
    (tmp_path / "config.json").write_text(json.dumps(FISH_OMNI_CONFIG), encoding="utf-8")
    save_file(_checkpoint(package), tmp_path / "model.safetensors")
    output = tmp_path / "output"
    main(
        [
            "build",
            "--model",
            str(tmp_path),
            "--dtype",
            "f32",
            "--task",
            task,
            "--execution-provider",
            ep,
            "--dequantize",
            "--output",
            str(output),
        ]
    )
    audio = "audio_generator" if task == "fish-omni-generation" else "audio_decoder"
    for component in ("embedding", "decoder", audio):
        model = ir.load(output / component / "model.onnx")
        assert all(
            value.const_value is not None for value in model.graph.initializers.values()
        )
    report = json.loads((output / "weight-loading-report.json").read_text())
    assert report["native_fp8"] is False
