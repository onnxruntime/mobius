# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline public-builder and exact per-expert weight-format regressions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import safetensors.torch
import torch

from mobius import build
from mobius.__main__ import main
from mobius._configs import NemotronHConfig
from mobius.integrations._weight_loading import stream_preprocessed_safetensors_to_model
from mobius.integrations.transformers import _builder
from mobius.integrations.transformers._builder import _prepare_nemotron_modelopt_source
from mobius.integrations.transformers._nemotron_h_weights import (
    build_nemotron_h_streaming_plan,
)


def _config():
    # Reduced pinned Lightning topology/format, NOT reduced real checkpoint weights.
    return {
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "dtype": "bfloat16",
        "hidden_size": 32,
        "vocab_size": 48,
        "intermediate_size": 32,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "layers_block_type": ["mamba", "moe", "attention"],
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "moe_shared_expert_intermediate_size": 64,
        "mamba_num_heads": 4,
        "mamba_head_dim": 8,
        "n_groups": 2,
        "ssm_state_size": 8,
        "mamba_ssm_cache_dtype": "float32",
        "layer_norm_epsilon": 1e-5,
        "conv_kernel": 4,
        "routed_scaling_factor": 2.5,
        "num_nextn_predict_layers": 0,
    }


def _source_name(target):
    if target == "model.embed_tokens.weight":
        return "backbone.embeddings.weight"
    if target == "model.norm.weight":
        return "backbone.norm_f.weight"
    if target.startswith("model.layers."):
        name = target.replace("model.", "backbone.", 1)
        for kind in ("mamba", "moe", "self_attn"):
            name = name.replace(f".{kind}.", ".mixer.")
        return name
    return target


def _write_mixed_checkpoint(tmp_path):
    raw = _config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    graph = build(str(tmp_path), load_weights=False, dtype="bf16")["model"].graph
    tensors = {}
    fp8_modules = []
    fp4_modules = []
    expected = {}
    for name, initializer in graph.initializers.items():
        if initializer.const_value is not None:
            continue
        source = _source_name(name)
        shape = tuple(initializer.shape)
        module = source.removesuffix(".weight")
        if ".mixer.in_proj." in source or ".mixer.out_proj." in source:
            tensor = torch.full(shape, 1.5).to(torch.float8_e4m3fn)
            tensors[source] = tensor
            tensors[module + ".weight_scale"] = torch.tensor(0.25)
            tensors[module + ".input_scale"] = torch.tensor([0.5])
            fp8_modules.append(module)
            expected[name] = torch.full(shape, 0.375, dtype=torch.bfloat16)
        elif (
            ".experts." in source or ".shared_experts." in source or source == "lm_head.weight"
        ):
            tensors[source] = torch.full((shape[0], shape[1] // 2), 0x92, dtype=torch.uint8)
            tensors[module + ".weight_scale"] = torch.ones(shape[0], shape[1] // 16).to(
                torch.float8_e4m3fn
            )
            tensors[module + ".weight_scale_2"] = torch.tensor(0.5)
            fp4_modules.append(module)
            dense = torch.empty(shape, dtype=torch.bfloat16)
            dense[:, 0::2] = 0.5
            dense[:, 1::2] = -0.25
            expected[name] = dense
        else:
            tensors[source] = torch.full(
                shape,
                0.123456 if source.endswith(".gate.weight") else 0.125,
                # Pinned NVFP4 keeps both router tensors FP32; this source
                # contract must not be inferred from the target graph's dtype.
                dtype=torch.float32 if ".gate." in source else torch.bfloat16,
            )
            expected[name] = tensors[source]
    raw["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "producer": {"name": "modelopt", "version": "0.44.0rc5"},
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        "quantized_layers": {
            **{name: {"quant_algo": "FP8"} for name in fp8_modules},
            **{name: {"quant_algo": "W4A16_NVFP4", "group_size": 16} for name in fp4_modules},
        },
        "config_groups": {
            "group_0": {
                "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                "targets": fp8_modules,
            },
            "group_1": {
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": fp4_modules,
            },
        },
    }
    for projection, suffix in (("k_proj", "k_scale"), ("v_proj", "v_scale")):
        tensors[f"backbone.layers.2.mixer.{projection}.{suffix}"] = torch.tensor([0.25])
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    return raw, expected


def test_public_local_build_reconstructs_all_projection_families(tmp_path, monkeypatch):
    _, expected = _write_mixed_checkpoint(tmp_path)
    monkeypatch.setattr(
        _builder,
        "_download_weights",
        lambda *args, **kwargs: pytest.fail("ModelOpt must never reach eager/affine loader"),
    )
    package = build(str(tmp_path), keep_quantized=False, dtype="bf16")
    report = package.weight_loading_report
    assert report["storage_policy"] == "explicit-dense-bf16-reconstruction"
    assert report["native_nvfp4"] is False
    assert report["ignored_tensors"] == 0
    assert report["validated_constants"] == 4  # Input + KV scales, not simulated.
    assert report["validated_kv_cache_scale_count"] == 2
    assert report["kv_cache_quantization_preserved"] is False
    assert report["reconstructed_projection_count"] == 13
    graph = package["model"].graph
    assert not any(node.op_type in {"MatMulNBits", "QMoE"} for node in graph)
    for name, value in expected.items():
        actual = graph.initializers[name].const_value.numpy().astype("float32")
        torch.testing.assert_close(torch.from_numpy(actual), value.float(), rtol=0, atol=0)
    output = tmp_path / "export"
    package.save(output, external_data="onnx")
    assert (output / "model.onnx").is_file()
    loaded = ir.load(output / "model.onnx")
    assert loaded.metadata_props["mobius.source_weight_format"] == "modelopt-mixed-fp8-nvfp4"


@pytest.mark.parametrize("router_dtype", [torch.float32, torch.bfloat16])
def test_floating_per_expert_checkpoint_uses_exact_streaming_mapping(tmp_path, router_dtype):
    raw, expected = _write_mixed_checkpoint(tmp_path)
    expected["model.layers.1.moe.gate.weight"] = expected["model.layers.1.moe.gate.weight"].to(
        router_dtype
    )
    del raw["quantization_config"]
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    safetensors.torch.save_file(
        {_source_name(name): tensor for name, tensor in expected.items()},
        tmp_path / "model.safetensors",
    )
    package = build(str(tmp_path), load_weights=False, dtype="bf16")
    config = NemotronHConfig.from_transformers(SimpleNamespace(**raw))
    report = stream_preprocessed_safetensors_to_model(
        package["model"],
        str(tmp_path),
        lambda index, initializers: build_nemotron_h_streaming_plan(
            config,
            index,
            initializers,
        ),
    )
    assert report["ignored_tensors"] == 0
    assert report["storage_policy"] == "dense"
    assert report["reconstructed_projection_count"] == 0
    for name, tensor in expected.items():
        actual = (
            package["model"].graph.initializers[name].const_value.numpy().astype("float32")
        )
        torch.testing.assert_close(torch.from_numpy(actual), tensor.float(), rtol=0, atol=0)


def test_cli_explicit_dequantize_writes_dense_not_native_format(tmp_path):
    _write_mixed_checkpoint(tmp_path)
    output = tmp_path / "cli-export"
    main(
        [
            "build",
            "--config",
            str(tmp_path),
            "--output",
            str(output),
            "--ep",
            "cpu",
            "--dtype",
            "bf16",
            "--dequantize",
            "--external-data",
            "onnx",
        ]
    )
    model = ir.load(output / "model.onnx")
    assert (
        model.metadata_props["mobius.storage_policy"] == "explicit-dense-bf16-reconstruction"
    )
    report = json.loads((output / "weight-loading-report.json").read_text())
    assert report["source_weight_format"] == "modelopt-mixed-fp8-nvfp4"
    assert report["native_nvfp4"] is False


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({}, NotImplementedError),
        ({"keep_quantized": False}, ValueError),
        ({"keep_quantized": False, "dtype": "f16"}, ValueError),
    ],
)
def test_public_native_or_implicit_dtype_requests_fail_before_weight_io(
    tmp_path, monkeypatch, kwargs, error
):
    raw = _config()
    raw["quantization_config"] = {"quant_method": "modelopt"}
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(
        _builder,
        "_download_weights",
        lambda *args, **kwargs: pytest.fail("unsupported path attempted weight download"),
    )
    with pytest.raises(error):
        build(str(tmp_path), **kwargs)


def test_reconstruction_policy_does_not_mutate_source_config():
    raw = SimpleNamespace(
        model_type="nemotron_h",
        quantization_config={
            "quant_method": "modelopt",
            "config_groups": {
                "group": {
                    "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                    "targets": ["projection"],
                },
            },
        },
    )
    dense, parent, source = _prepare_nemotron_modelopt_source(
        raw, raw, keep_quantized=False, dtype="bf16"
    )
    assert dense is not raw and parent is dense
    assert dense.quantization_config is None
    assert raw.quantization_config["quant_method"] == "modelopt"
    assert source.modules == {"projection": "modelopt_fp8"}


def test_mtp_sources_fail_closed_even_when_config_omits_nextn(tmp_path, monkeypatch):
    _write_mixed_checkpoint(tmp_path)
    path = tmp_path / "model.safetensors"
    weights = safetensors.torch.load_file(path)
    weights["mtp.layers.0.hnorm.weight"] = torch.ones(32, dtype=torch.bfloat16)
    safetensors.torch.save_file(weights, path)
    monkeypatch.setattr(
        "mobius.integrations.transformers._nemotron_h_weights._load_indexed_tensor",
        lambda *args: pytest.fail("MTP source refusal must precede scale/value reads"),
    )
    with pytest.raises(NotImplementedError, match="MTP tensors"):
        build(str(tmp_path), keep_quantized=False, dtype="bf16")


@pytest.mark.parametrize("change", ["missing", "wrong-shape", "nonfinite", "undeclared"])
def test_cache_scale_contract_fails_before_initializer_binding(tmp_path, change):
    raw, _ = _write_mixed_checkpoint(tmp_path)
    path = tmp_path / "model.safetensors"
    weights = safetensors.torch.load_file(path)
    key = "backbone.layers.2.mixer.k_proj.k_scale"
    if change == "missing":
        del weights[key]
    elif change == "wrong-shape":
        weights[key] = torch.ones(2)
    elif change == "nonfinite":
        weights[key] = torch.tensor([float("nan")])
    else:
        del raw["quantization_config"]["kv_cache_scheme"]
        (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    safetensors.torch.save_file(weights, path)
    with pytest.raises(ValueError):
        build(str(tmp_path), keep_quantized=False, dtype="bf16")


def _write_mtp_checkpoint(tmp_path, *, mixed):
    raw, expected = _write_mixed_checkpoint(tmp_path)
    path = tmp_path / "model.safetensors"
    weights = safetensors.torch.load_file(path)
    if not mixed:
        del raw["quantization_config"]
        weights = {_source_name(name): tensor for name, tensor in expected.items()}
    # Omission accounting includes all storage/scale kinds, not only weights.
    mtp = {
        "mtp.layers.0.hnorm.weight": torch.ones(32, dtype=torch.bfloat16),
        "mtp.layers.0.mixer.weight": torch.zeros(4, 16, dtype=torch.uint8),
        "mtp.layers.0.mixer.weight_scale": torch.ones(4, 2).to(torch.float8_e4m3fn),
        "mtp.layers.0.mixer.weight_scale_2": torch.tensor(0.25),
    }
    weights.update(mtp)
    raw["num_nextn_predict_layers"] = 1
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    safetensors.torch.save_file(weights, path)
    return raw, expected, mtp


@pytest.mark.parametrize("mixed", [False, True])
def test_explicit_target_variant_accounts_for_every_mtp_tensor(tmp_path, mixed, monkeypatch):
    raw, expected, mtp = _write_mtp_checkpoint(tmp_path, mixed=mixed)
    source = SimpleNamespace(**raw)
    monkeypatch.setattr(_builder, "_load_transformers_config", lambda *a, **k: (source, True))
    monkeypatch.setattr(
        _builder, "_download_weights", lambda *a, **k: pytest.fail("must use strict streaming")
    )
    options = {"dtype": "bf16", "keep_quantized": not mixed}
    with pytest.raises(NotImplementedError, match="NextN/MTP"):
        build(str(tmp_path), **options)
    package = build(str(tmp_path), target_decoder_only=True, **options)
    assert source.num_nextn_predict_layers == 1
    assert source.__dict__ == raw
    report = package.weight_loading_report
    assert report["export_variant"] == "target-decoder-only"
    assert report["source_num_nextn_predict_layers"] == 1
    assert report["mtp_preserved"] is False
    assert report["omitted_mtp_tensor_count"] == len(mtp)
    assert report["ignored_tensors"] == len(mtp)
    assert set(report["omitted_mtp_tensors"]) == set(mtp)
    for name, tensor in mtp.items():
        entry = report["omitted_mtp_tensors"][name]
        assert entry["shape"] == list(tensor.shape)
        assert "explicit target-decoder-only" in entry["reason"]
    model = package["model"]
    assert "/target-decoder-only/" in model.graph.name
    assert model.metadata_props["mobius.source_num_nextn_predict_layers"] == "1"
    assert model.metadata_props["mobius.mtp_inventory_status"] == "inspected"
    assert not any(name.startswith("mtp.") for name in model.graph.initializers)
    for name, tensor in expected.items():
        actual = model.graph.initializers[name].const_value.numpy().astype("float32")
        torch.testing.assert_close(torch.from_numpy(actual), tensor.float(), rtol=0, atol=0)
    output = tmp_path / "target-decoder-only"
    package.save(output, external_data="onnx")
    loaded = ir.load(output / "model.onnx")
    assert loaded.metadata_props["mobius.export_variant"] == "target-decoder-only"
    assert set(json.loads(loaded.metadata_props["mobius.omitted_mtp_tensors"])) == set(mtp)
    saved_report = json.loads((output / "weight-loading-report.json").read_text())
    assert saved_report["omitted_mtp_tensors"] == report["omitted_mtp_tensors"]


@pytest.mark.parametrize("mixed", [False, True])
def test_target_variant_cli_and_graph_only_provenance(tmp_path, mixed):
    raw, _, mtp = _write_mtp_checkpoint(tmp_path, mixed=mixed)
    graph_only = build(
        str(tmp_path),
        dtype="bf16",
        keep_quantized=not mixed,
        target_decoder_only=True,
        load_weights=False,
    )["model"]
    assert graph_only.metadata_props["mobius.mtp_inventory_status"] == "not_inspected"
    assert "mobius.omitted_mtp_tensors" not in graph_only.metadata_props
    output = tmp_path / "cli-target"
    main(
        [
            "build",
            "--config",
            str(tmp_path),
            "--output",
            str(output),
            "--ep",
            "cpu",
            "--dtype",
            "bf16",
            "--target-decoder-only",
            "--external-data",
            "onnx",
            *(["--dequantize"] if mixed else []),
        ]
    )
    loaded = ir.load(output / "model.onnx")
    assert loaded.metadata_props["mobius.export_variant"] == "target-decoder-only"
    report = json.loads((output / "weight-loading-report.json").read_text())
    assert set(report["omitted_mtp_tensors"]) == set(mtp)
    assert json.loads((tmp_path / "config.json").read_text()) == raw


@pytest.mark.parametrize(
    "extra", ["unexpected.weight", "unexpected.weight_scale", "mtpx.weight"]
)
def test_target_variant_does_not_ignore_non_mtp_extras(tmp_path, extra):
    _write_mtp_checkpoint(tmp_path, mixed=True)
    path = tmp_path / "model.safetensors"
    tensors = safetensors.torch.load_file(path)
    tensors[extra] = torch.ones(1)
    safetensors.torch.save_file(tensors, path)
    with pytest.raises(ValueError, match=r"Unmapped|unclassified"):
        build(str(tmp_path), dtype="bf16", keep_quantized=False, target_decoder_only=True)


def test_target_variant_still_blocks_native_modelopt(tmp_path):
    _write_mtp_checkpoint(tmp_path, mixed=True)
    with pytest.raises(NotImplementedError, match="Native"):
        build(str(tmp_path), dtype="bf16", target_decoder_only=True)


def test_target_variant_only_accepts_nemotron(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    with pytest.raises(ValueError, match="only supported for Nemotron"):
        build(str(tmp_path), load_weights=False, target_decoder_only=True)


def test_declared_nextn_requires_source_inventory_when_loading(tmp_path):
    raw, _ = _write_mixed_checkpoint(tmp_path)
    raw["num_nextn_predict_layers"] = 1
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no mtp"):
        build(str(tmp_path), dtype="bf16", keep_quantized=False, target_decoder_only=True)


def test_target_variant_forwards_cli_model_and_revision(tmp_path, monkeypatch):
    from mobius import __main__ as cli

    received = {}

    def capture(source, **options):
        received.update(source=source, **options)
        return {}

    monkeypatch.setattr(cli, "build", capture)
    monkeypatch.setattr(cli, "_save_package", lambda *args: None)
    monkeypatch.setattr(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
        lambda *args, **kwargs: pytest.fail("target variant cannot bypass build policy"),
    )
    main(
        [
            "build",
            "--model",
            "nvidia/lightning",
            "--revision",
            "a" * 40,
            "--output",
            str(tmp_path / "target"),
            "--dtype",
            "bf16",
            "--target-decoder-only",
            "--dequantize",
        ]
    )
    assert received["source"] == "nvidia/lightning"
    assert received["target_decoder_only"] is True
    assert received["keep_quantized"] is False
    assert received["revision"] == "a" * 40
