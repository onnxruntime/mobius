# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Whole-package native storage accounting, byte fidelity and atomic failure."""

from __future__ import annotations

import dataclasses
import inspect

import onnx_ir as ir
import pytest
import safetensors.torch
import torch
from onnx_ir import tensor_adapters

from mobius import build_from_module
from mobius._configs import NemotronHOmniConfig
from mobius._configs.per_model._nemotron_h_omni_test import _omni_config
from mobius._flags import override_flags
from mobius.integrations.transformers._nemotron_h_omni_weights import (
    stream_nemotron_h_omni_safetensors_to_package,
)
from mobius.models import NemotronHOmniModel

_IQ2_SOURCE = "language_model.backbone.layers.1.mixer.experts.0.up_proj.weight"
_IQ2_TARGET = "decoder.model.layers.1.moe.experts.0.up_proj.weight"
_NVFP4_SOURCE = "language_model.backbone.layers.0.mixer.in_proj"
_NVFP4_TARGET = "decoder.model.layers.0.mamba.in_proj"


def _source_name(target):
    if target.startswith("decoder.model.layers."):
        name = target.removeprefix("decoder.model.")
        for mixer in ("mamba", "moe", "self_attn"):
            name = name.replace(f".{mixer}.", ".mixer.")
        return "language_model.backbone." + name
    if target.startswith("decoder.model.norm."):
        return target.replace("decoder.model.norm.", "language_model.backbone.norm_f.")
    if target.startswith("decoder.lm_head."):
        return target.replace("decoder.", "language_model.", 1)
    if target.startswith("vision_encoder.vision_model."):
        return target.replace(
            "vision_encoder.vision_model.", "vision_model.radio_model.model.", 1
        )
    if target.startswith("vision_encoder.projector."):
        return target.removeprefix("vision_encoder.projector.")
    if target == "embedding.embed_tokens.weight":
        return "language_model.backbone.embeddings.weight"
    raise AssertionError(f"Unclassified fixture parameter {target}")


def _fixture(*, sparse=False):
    source = _omni_config()
    source["llm_config"]["hidden_size"] = 256
    source["llm_config"]["moe_latent_size"] = 256
    source["quantization_config"]["quantized_layers"][
        "language_model.backbone.layers.0.mixer.out_proj"
    ] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    if sparse:
        source["llm_config"]["hidden_act"] = "relu2"
        for index, block in enumerate(source["llm_config"]["block_configs"]):
            if block["block_type"] != "moe":
                continue
            block["moe_intermediate_size"] = 256 if index == 1 else 512
            for expert in range(block["n_routed_experts"]):
                for projection in ("up_proj", "down_proj"):
                    root = (
                        f"language_model.backbone.layers.{index}.mixer."
                        f"experts.{expert}.{projection}"
                    )
                    source["quantization_config"]["quantized_layers"][root] = {
                        "quant_algo": "IQ2_XXS",
                        "group_size": 256,
                        "block_payload_bytes": 66,
                        "packing": "ggml",
                    }
    config = NemotronHOmniConfig.from_transformers(source)
    config = dataclasses.replace(config, dtype=ir.DataType.BFLOAT16)
    with override_flags(sparse_iq2_moe=sparse):
        package = build_from_module(
            NemotronHOmniModel(config),
            config,
            task="nemotron-h-omni",
            execution_provider="cuda" if sparse else "default",
        )
    state = {}
    generator = torch.Generator().manual_seed(7)
    for model in package.values():
        for name, initializer in model.graph.initializers.items():
            if initializer.const_value is not None:
                continue
            shape = tuple(initializer.shape)
            if ".packed_experts." in name:
                prefix, projection = name.split(".packed_experts.")
                for expert in range(shape[0]):
                    raw = (
                        (torch.arange(torch.tensor(shape[1:]).prod()) + 17 * expert)
                        .to(torch.uint8)
                        .reshape(shape[1:])
                    )
                    raw[..., :2] = torch.tensor([1.0009765625], dtype=torch.float16).view(
                        torch.uint8
                    )
                    target = f"{prefix}.experts.{expert}.{projection}.weight"
                    state[_source_name(target)] = raw
                continue
            source_name = _source_name(name)
            if name == _IQ2_TARGET:
                raw = torch.arange(torch.tensor(shape).prod()).to(torch.uint8).reshape(shape)
                raw[..., :2] = torch.tensor([0.125], dtype=torch.float16).view(torch.uint8)
                state[source_name] = raw
            elif name == "decoder.lm_head.weight":
                rows, blocks, _ = shape
                raw = torch.zeros((rows, blocks, 34), dtype=torch.uint8)
                raw[..., :2] = torch.tensor([1.0009765625], dtype=torch.float16).view(
                    torch.uint8
                )
                raw[..., 2:] = torch.arange(32, dtype=torch.uint8)
                state[source_name] = raw
            elif ".mamba." in name and name.endswith(".weight_scale"):
                state[source_name] = torch.full(shape, 1.5).to(torch.float8_e4m3fn)
            elif ".mamba." in name and name.endswith(".weight_scale_2"):
                state[source_name] = torch.tensor(0.10001, dtype=torch.float32)
            elif initializer.dtype == ir.DataType.UINT8:
                state[source_name] = (
                    torch.arange(torch.tensor(shape).prod()).to(torch.uint8).reshape(shape)
                )
            else:
                state[source_name] = (torch.randn(shape, generator=generator) * 0.02).to(
                    tensor_adapters.to_torch_dtype(initializer.dtype)
                )
    for suffix, values in (
        ("norm_mean", config.radio_norm_mean),
        ("norm_std", config.radio_norm_std),
    ):
        state["vision_model.radio_model.input_conditioner." + suffix] = torch.tensor(
            values, dtype=torch.float32
        ).reshape(3, 1, 1)
    state["vision_model.summary_idxs"] = torch.tensor(
        config.radio_summary_idxs, dtype=torch.int64
    )
    return config, package, state


def _checkpoint(directory, state):
    directory.mkdir()
    safetensors.torch.save_file(state, str(directory / "model.safetensors"))


def test_native_omni_all_sources_and_external_bytes_round_trip(tmp_path):
    config, package, state = _fixture()
    source = tmp_path / "source"
    _checkpoint(source, state)
    report = stream_nemotron_h_omni_safetensors_to_package(
        package, str(source), config, revision="pinned-fixture"
    )
    assert report["revision"] == "pinned-fixture"
    assert report["ignored_tensors"] == 0
    assert report["validated_constants"] == 3
    assert report["validated_summary_indices"] == [0, 1]
    assert report["synthesized_tensors"] == 0
    assert report["ggml_reconstructed_weight_dtype"] == "BFLOAT16"
    assert report["projection_formats"] == {"IQ2_XXS": 1, "Q8_0": 1, "W4A16_NVFP4": 2}
    assert report["native_iq2_storage"] is True
    assert report["streaming_external_data"] is True
    assert report["full_checkpoint_dequantization"] is False
    output = tmp_path / "output"
    package.save(str(output), progress_bar=False)
    decoder = ir.load(output / "decoder/model.onnx")

    def value(name):
        return decoder.graph.initializers[name].const_value.numpy()

    assert (value(_IQ2_TARGET) == state[_IQ2_SOURCE].numpy()).all()
    assert (
        value(_NVFP4_TARGET + ".weight_scale")
        == state[_NVFP4_SOURCE + ".weight_scale"].view(torch.uint8).numpy()
    ).all()
    assert value(_NVFP4_TARGET + ".weight_scale_2").dtype.name == "float32"
    assert (
        value(_NVFP4_TARGET + ".weight_scale_2").item()
        == state[_NVFP4_SOURCE + ".weight_scale_2"].item()
    )
    assert (
        value("decoder.lm_head.weight") == state["language_model.lm_head.weight"].numpy()
    ).all()
    assert "decoder.lm_head.scales" not in decoder.graph.initializers
    assert "decoder.lm_head.zero_points" not in decoder.graph.initializers
    assert (output / "vision_encoder/model.onnx").is_file()
    assert (output / "embedding/model.onnx").is_file()


def test_native_omni_default_save_bounds_shards_and_materialization(tmp_path, monkeypatch):
    config, package, state = _fixture()
    source = tmp_path / "source"
    _checkpoint(source, state)
    stream_nemotron_h_omni_safetensors_to_package(package, str(source), config)
    original_save = ir.save
    calls = []

    def record_save(model, path, *, max_workers=8, **kwargs):
        calls.append((max_workers, kwargs["max_shard_size_bytes"]))
        if "max_workers" in inspect.signature(original_save).parameters:
            kwargs["max_workers"] = max_workers
        return original_save(model, path, **kwargs)

    monkeypatch.setattr(ir, "save", record_save)
    package.save(str(tmp_path / "output"), progress_bar=False)
    assert len(calls) == 3
    assert all(workers == 1 and shard_bytes == 1 << 30 for workers, shard_bytes in calls)
    assert package.weight_loading_report["serializer_max_workers"] == 1
    assert package.weight_loading_report["external_data_shard_limit_bytes"] == 1 << 30


@pytest.mark.parametrize(
    "failure", ["extra", "missing", "shape", "scale", "conditioner", "summary_indices"]
)
def test_native_omni_binding_fails_atomically(tmp_path, failure):
    config, package, state = _fixture()
    original = dict(package)
    if failure == "extra":
        state["undeclared.weight"] = torch.ones(1)
    elif failure == "missing":
        del state["language_model.backbone.layers.0.norm.weight"]
    elif failure == "shape":
        state[_IQ2_SOURCE] = state[_IQ2_SOURCE][:1]
    elif failure == "scale":
        state[_IQ2_SOURCE][..., :2] = torch.tensor([float("inf")], dtype=torch.float16).view(
            torch.uint8
        )
    elif failure == "conditioner":
        state["vision_model.radio_model.input_conditioner.norm_mean"].add_(0.1)
    else:
        state["vision_model.summary_idxs"] = state["vision_model.summary_idxs"].flip(0)
    source = tmp_path / "source"
    _checkpoint(source, state)
    with pytest.raises(ValueError):
        stream_nemotron_h_omni_safetensors_to_package(package, str(source), config)
    assert all(package[name] is model for name, model in original.items())
    assert package["decoder"].graph.initializers[_IQ2_TARGET].const_value is None


def test_native_omni_rejects_eager_weight_loading():
    config, _, _ = _fixture()
    with pytest.raises(ValueError, match="source-accounted streaming"):
        NemotronHOmniModel(config).preprocess_weights({})


def test_sparse_iq2_banks_preserve_every_source_byte_and_split_package(tmp_path):
    config, package, state = _fixture(sparse=True)
    source = tmp_path / "source"
    _checkpoint(source, state)
    report = stream_nemotron_h_omni_safetensors_to_package(package, str(source), config)
    assert report["routed_moe_execution"] == "sparse-iq2"
    assert report["sparse_iq2_moe_layers"] == 2
    assert report["ignored_tensors"] == 0
    assert report["projection_formats"]["IQ2_XXS"] == 10
    assert report["full_checkpoint_dequantization"] is False
    package.save(str(tmp_path / "output"), progress_bar=False)
    decoder = ir.load(tmp_path / "output" / "decoder" / "model.onnx")
    for name, value in decoder.graph.initializers.items():
        if ".packed_experts." not in name:
            continue
        prefix, projection = name.split(".packed_experts.")
        actual = value.const_value.numpy()
        assert actual.dtype.name == "uint8"
        for expert in range(actual.shape[0]):
            target = f"{prefix}.experts.{expert}.{projection}.weight"
            expected = state[_source_name(target)].numpy()
            assert (actual[expert] == expected).all()
    assert (tmp_path / "output" / "vision_encoder" / "model.onnx").is_file()
    assert (tmp_path / "output" / "embedding" / "model.onnx").is_file()
    assert package.weight_loading_report["serializer_max_workers"] == 1
    assert package.weight_loading_report["external_data_shard_limit_bytes"] == 1 << 30


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "dtype", "scale"])
def test_sparse_iq2_bank_binding_fails_atomically(tmp_path, failure):
    config, package, state = _fixture(sparse=True)
    original = dict(package)
    if failure == "missing":
        del state[_IQ2_SOURCE]
    elif failure == "extra":
        state["unclassified.weight"] = torch.zeros(1)
    elif failure == "shape":
        state[_IQ2_SOURCE] = state[_IQ2_SOURCE][:1]
    elif failure == "dtype":
        state[_IQ2_SOURCE] = state[_IQ2_SOURCE].to(torch.int8)
    else:
        state[_IQ2_SOURCE][..., :2] = torch.tensor([float("nan")], dtype=torch.float16).view(
            torch.uint8
        )
    source = tmp_path / "source"
    _checkpoint(source, state)
    with pytest.raises(ValueError):
        stream_nemotron_h_omni_safetensors_to_package(package, str(source), config)
    assert all(package[name] is model for name, model in original.items())
    for name, value in package["decoder"].graph.initializers.items():
        if ".packed_experts." in name:
            assert value.const_value is None


@pytest.mark.parametrize("mutation", ["scale", "shape", "dtype"])
def test_sparse_iq2_serialization_revalidates_source_bytes(tmp_path, mutation):
    config, package, state = _fixture(sparse=True)
    source = tmp_path / "source"
    _checkpoint(source, state)
    stream_nemotron_h_omni_safetensors_to_package(package, str(source), config)
    if mutation == "scale":
        state[_IQ2_SOURCE][..., :2] = torch.tensor([float("inf")], dtype=torch.float16).view(
            torch.uint8
        )
    elif mutation == "shape":
        state[_IQ2_SOURCE] = state[_IQ2_SOURCE][:1]
    else:
        state[_IQ2_SOURCE] = state[_IQ2_SOURCE].to(torch.int8)
    safetensors.torch.save_file(state, str(source / "model.safetensors"))
    with pytest.raises(ValueError, match=r"Nonfinite GGML|changed after indexing"):
        package.save(str(tmp_path / "output"), progress_bar=False)


def test_sparse_iq2_plan_rejects_incompatible_projection_metadata(tmp_path):
    config, package, state = _fixture(sparse=True)
    projections = dict(config.quantized_projections)
    root = _IQ2_SOURCE.removesuffix(".weight")
    projections[root] = dataclasses.replace(projections[root], block_payload_bytes=65)
    config = dataclasses.replace(config, quantized_projections=projections)
    source = tmp_path / "source"
    _checkpoint(source, state)
    with pytest.raises(ValueError, match="requires GGML IQ2_XXS/256/66"):
        stream_nemotron_h_omni_safetensors_to_package(package, str(source), config)
