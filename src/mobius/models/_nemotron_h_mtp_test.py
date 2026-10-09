# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline Lightning NextN inventory, independent math, and cache regressions."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import safetensors.torch
import torch
import torch.nn.functional as functional

from mobius import build
from mobius._builder import build_from_module
from mobius._configs import NemotronHConfig
from mobius._model_package import ModelPackage
from mobius._testing import create_test_builder, create_test_input
from mobius.integrations.transformers._nemotron_h_weights_test import (
    _config,
    _source_name,
    _write_mixed_checkpoint,
)
from mobius.models.nemotron_h import NemotronHMoELayer
from mobius.models.nemotron_h_mtp import (
    NemotronHMtpModel,
    NemotronHSpeculativeModel,
    _FinalMoELayer,
    mtp_source_name,
)


def _raw():
    return {
        **_config(),
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "mlp_hidden_act": "relu2",
    }


def _graph_package(dtype="f32"):
    config = NemotronHConfig.from_transformers(SimpleNamespace(**_raw()))
    config.dtype = {
        "f32": ir.DataType.FLOAT,
        "f16": ir.DataType.FLOAT16,
        "bf16": ir.DataType.BFLOAT16,
    }[dtype]
    return build_from_module(NemotronHSpeculativeModel(config), config, task="nemotron-h-mtp")


def test_final_moe_layer_accepts_and_forwards_base_positional_contract(monkeypatch):
    config = NemotronHConfig.from_transformers(SimpleNamespace(**_raw()))
    layer = _FinalMoELayer(config)
    builder, op, _graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, config.hidden_size])
    received = []

    def base_forward(
        self, op, hidden_states, attention_bias, position_embeddings, past_key_value
    ):
        received.append((attention_bias, position_embeddings, past_key_value))
        return hidden_states, (None, None)

    monkeypatch.setattr(NemotronHMoELayer, "forward", base_forward)
    layer(op, hidden)
    bias = create_test_input(builder, "bias", [1, 1, 2, 2])
    positions = (hidden, hidden)
    cache = (hidden, hidden)
    layer(op, hidden, bias, positions, cache)
    assert received == [(None, None, None), (bias, positions, cache)]


def _checkpoint(tmp_path, *, dtype=torch.float32):
    raw = _raw()
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    package = build(str(tmp_path), task="nemotron-h-mtp", dtype="f32", load_weights=False)
    generator = torch.Generator().manual_seed(187)
    tensors = {}
    for component, model in package.items():
        for name, value in model.graph.initializers.items():
            if value.const_value is not None:
                continue
            source = _source_name(name) if component == "decoder" else mtp_source_name(name)
            if source in tensors:
                continue
            shape = tuple(value.shape)
            tensor = torch.randn(shape, generator=generator) * 0.04
            if len(shape) == 1 and source.endswith("weight"):
                tensor += 1
            if source.endswith("A_log"):
                tensor.zero_()
            tensors[source] = tensor.to(
                torch.float32
                if source.endswith(("e_score_correction_bias", "A_log"))
                else dtype
            )
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    return tensors


def _reference(ids, seed, mask, past_k, past_v, weights):
    """Independent dense PyTorch equations, not the Mobius module/component tree."""

    def norm(x, key):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5) * weights[key]

    def linear(x, key):
        return functional.linear(x, weights[key])

    prefix = "mtp.layers.0."
    embed = functional.embedding(ids, weights["backbone.embeddings.weight"])
    fused = torch.cat(
        [norm(embed, prefix + "enorm.weight"), norm(seed, prefix + "hnorm.weight")], -1
    )
    h = linear(fused, prefix + "eh_proj.weight")
    n = norm(h, prefix + "norm.weight")
    b, s, _ = n.shape
    q = linear(n, prefix + "mixer.q_proj.weight").view(b, s, 4, 8).transpose(1, 2)
    k = linear(n, prefix + "mixer.k_proj.weight").view(b, s, 2, 8).transpose(1, 2)
    v = linear(n, prefix + "mixer.v_proj.weight").view(b, s, 2, 8).transpose(1, 2)
    k, v = torch.cat([past_k, k], 2), torch.cat([past_v, v], 2)
    scores = q @ k.repeat_interleave(2, 1).transpose(-2, -1) / 8**0.5
    causal = torch.arange(k.shape[2])[None, :] <= (torch.arange(s)[:, None] + past_k.shape[2])
    allowed = causal[None, None] & mask[:, None, None, :].bool()
    probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    attended = (probabilities @ v.repeat_interleave(2, 1)).transpose(1, 2).reshape(b, s, 32)
    h = h + linear(attended, prefix + "mixer.o_proj.weight")
    prefix = "mtp.layers.1."
    n = norm(h, prefix + "norm.weight")
    probabilities = linear(n.float(), prefix + "mixer.gate.weight").sigmoid()
    choice = probabilities + weights[prefix + "mixer.gate.e_score_correction_bias"]
    indices = choice.topk(2, -1).indices
    routing = probabilities.gather(-1, indices)
    routing = routing / (routing.sum(-1, keepdim=True) + 1e-20) * 2.5
    routed = torch.zeros_like(n)
    for i in range(4):
        expert = prefix + f"mixer.experts.{i}."
        output = linear(
            functional.relu(linear(n, expert + "up_proj.weight")).square(),
            expert + "down_proj.weight",
        )
        weight = (routing * (indices == i)).sum(-1, keepdim=True)
        routed += output * weight
    shared = prefix + "mixer.shared_experts."
    shared_output = linear(
        functional.relu(linear(n, shared + "up_proj.weight")).square(),
        shared + "down_proj.weight",
    )
    result = norm(h + routed + shared_output, prefix + "final_layernorm.weight")
    return linear(result, "lm_head.weight"), result, k, v


def _session(model, path):
    ir.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def test_explicit_graph_contract_and_shared_sources():
    package = _graph_package()
    assert set(package) == {"decoder", "mtp"}
    assert "mtp_seed" in {v.name for v in package["decoder"].graph.outputs}
    mtp = package["mtp"]
    assert [v.name for v in mtp.graph.inputs] == [
        "input_ids",
        "hidden_states",
        "attention_mask",
        "past_key_values.0.key",
        "past_key_values.0.value",
    ]
    assert [v.name for v in mtp.graph.outputs] == [
        "logits",
        "mtp_hidden",
        "present.0.key",
        "present.0.value",
    ]
    assert mtp.graph.inputs[1].shape[-1] == 32
    assert list(mtp.graph.inputs[3].shape)[1::2] == [2, 8]
    for value in mtp.graph.inputs:
        assert value.dtype == (
            ir.DataType.INT64
            if value.name in {"input_ids", "attention_mask"}
            else ir.DataType.FLOAT
        )
    expected_shapes = {
        "logits": ["batch", "sequence_len", "48"],
        "mtp_hidden": ["batch", "sequence_len", "32"],
        "present.0.key": ["batch", "2", "past_sequence_len + sequence_len", "8"],
        "present.0.value": ["batch", "2", "past_sequence_len + sequence_len", "8"],
    }
    for value in mtp.graph.outputs:
        assert value.dtype == ir.DataType.FLOAT
        assert [str(dimension) for dimension in value.shape] == expected_shapes[value.name]
    assert "position_ids" not in {v.name for v in mtp.graph.inputs}
    assert not any(n.op_type in {"RotaryEmbedding", "Sin", "Cos", "Scan"} for n in mtp.graph)


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_mtp_graph_dtypes(dtype):
    package = _graph_package(dtype)
    expected = {
        "f32": ir.DataType.FLOAT,
        "f16": ir.DataType.FLOAT16,
        "bf16": ir.DataType.BFLOAT16,
    }[dtype]
    for component in package.values():
        assert next(v for v in component.graph.outputs if v.name == "logits").dtype == expected
    router = package["mtp"].graph.initializers["layers.1.moe.gate.weight"]
    assert router.dtype == ir.DataType.FLOAT


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_public_package_independent_cpu_prefill_cached_decode_and_seed(tmp_path, source_dtype):
    weights = {n: t.float() for n, t in _checkpoint(tmp_path, dtype=source_dtype).items()}
    package = build(str(tmp_path), task="nemotron-h-mtp", dtype="f32")
    session = _session(package["mtp"], tmp_path / "mtp.onnx")
    generator = torch.Generator().manual_seed(32)
    past_k = past_v = torch.zeros(2, 2, 0, 8)
    for length in (3, 1, 2):
        ids = torch.randint(0, 48, (2, length), generator=generator)
        seed = torch.randn(2, length, 32, generator=generator)
        mask = torch.ones(2, past_k.shape[2] + length, dtype=torch.int64)
        mask[1, 1] = 0
        expected = _reference(ids, seed, mask, past_k, past_v, weights)
        actual = session.run(
            None,
            {
                "input_ids": ids.numpy(),
                "hidden_states": seed.numpy(),
                "attention_mask": mask.numpy(),
                "past_key_values.0.key": past_k.numpy(),
                "past_key_values.0.value": past_v.numpy(),
            },
        )
        for result, reference in zip(actual, expected):
            np.testing.assert_allclose(result, reference.numpy(), rtol=1e-4, atol=1e-5)
        past_k, past_v = expected[2:]
    decoder = _session(package["decoder"], tmp_path / "decoder.onnx")
    inputs = {
        "input_ids": np.array([[4, 5, 6], [7, 8, 9]], dtype=np.int64),
        "attention_mask": np.ones((2, 3), dtype=np.int64),
        "past_key_values.0.conv_state": np.zeros((2, 64, 3), dtype=np.float32),
        "past_key_values.0.ssm_state": np.zeros((2, 4, 8, 8), dtype=np.float32),
        "past_key_values.2.key": np.zeros((2, 2, 0, 8), dtype=np.float32),
        "past_key_values.2.value": np.zeros((2, 2, 0, 8), dtype=np.float32),
    }
    logits, seed = decoder.run(["logits", "mtp_seed"], inputs)
    np.testing.assert_allclose(
        logits, seed @ weights["lm_head.weight"].numpy().T, rtol=1e-5, atol=1e-6
    )
    # Exercise the real decoder→MTP bridge with shifted target tokens.
    aligned_ids = inputs["input_ids"][:, 1:]
    aligned_seed = seed[:, :-1]
    actual = session.run(
        None,
        {
            "input_ids": aligned_ids,
            "hidden_states": aligned_seed,
            "attention_mask": np.ones((2, 2), dtype=np.int64),
            "past_key_values.0.key": np.zeros((2, 2, 0, 8), dtype=np.float32),
            "past_key_values.0.value": np.zeros((2, 2, 0, 8), dtype=np.float32),
        },
    )
    expected = _reference(
        torch.from_numpy(aligned_ids),
        torch.from_numpy(aligned_seed),
        torch.ones(2, 2, dtype=torch.int64),
        torch.zeros(2, 2, 0, 8),
        torch.zeros(2, 2, 0, 8),
        weights,
    )
    np.testing.assert_allclose(actual[0], expected[0].numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "duplicate"])
def test_strict_mtp_inventory(tmp_path, mutation):
    tensors = _checkpoint(tmp_path)
    if mutation == "missing":
        del tensors["mtp.layers.1.mixer.shared_experts.up_proj.weight"]
    elif mutation == "extra":
        tensors["mtp.unexpected.weight"] = torch.zeros(1)
    elif mutation == "shape":
        tensors["mtp.layers.0.eh_proj.weight"] = torch.zeros(32, 32)
    else:
        tensors["model.embeddings.weight"] = tensors["backbone.embeddings.weight"].clone()
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError):
        build(str(tmp_path), task="nemotron-h-mtp", dtype="f32")


def test_package_accounting_save_and_default_guard(tmp_path):
    tensors = _checkpoint(tmp_path, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="NextN/MTP"):
        build(str(tmp_path), dtype="bf16", load_weights=False)
    package = build(str(tmp_path), task="nemotron-h-mtp", dtype="bf16")
    report = package.weight_loading_report
    assert report["source"] == "local-safetensors-checkpoint"
    assert report["decoder_component"]["source"] == "local-safetensors-checkpoint"
    assert report["mtp_component"]["source"] == "local-safetensors-checkpoint"
    assert report["mtp_preserved"] is True
    assert report["omitted_mtp_tensor_count"] == 0
    assert set(report["mtp_tensors"]) == {n for n in tensors if n.startswith("mtp.")}
    assert report["mtp_tensor_count"] == 22
    assert report["ignored_tensors"] == 0
    assert report["accounted_source_tensor_count"] == len(tensors)
    assert report["assigned_tensors"] == len(tensors) + 2  # Shared table copies.
    assert report["decoder_component"]["mtp_preservation_component"] == "mtp"
    assert package.config.num_nextn_predict_layers == 1
    before = (tmp_path / "model.safetensors").read_bytes()
    with pytest.raises(ValueError):
        package.save(tmp_path, external_data="safetensors")
    assert (tmp_path / "model.safetensors").read_bytes() == before
    output = tmp_path / "export"
    package.save(output, external_data="onnx")
    assert (output / "decoder" / "model.onnx").is_file()
    assert (output / "mtp" / "model.onnx").is_file()
    assert json.loads((output / "weight-loading-report.json").read_text())["mtp_preserved"]
    restored = ModelPackage.load(output)
    assert set(restored) == {"decoder", "mtp"}
    for model in restored.values():
        loading = json.loads(model.metadata_props["mobius.weight_loading"])
        assert loading["source"] == "local-safetensors-checkpoint"
        assert str(tmp_path) not in json.dumps(loading)
        assert model.metadata_props["mobius.source_num_nextn_predict_layers"] == "1"
        assert model.metadata_props["mobius.mtp_contract"] == "nemotron-h-lightning-nextn@1"
    assert restored.weight_loading_report["mtp_preserved"] is True
    assert str(tmp_path) not in json.dumps(restored.weight_loading_report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_nextn_predict_layers", 2),
        ("mtp_layers_block_type", ["attention"]),
        ("n_shared_experts", 0),
        ("moe_latent_size", 16),
        ("residual_in_fp32", True),
    ],
)
def test_architecture_variants_rejected(field, value):
    config = NemotronHConfig.from_transformers(SimpleNamespace(**_raw()))
    with pytest.raises(NotImplementedError):
        NemotronHSpeculativeModel(dataclasses.replace(config, **{field: value}))


def test_mixed_reconstruction_with_floating_mtp_and_shared_quantized_head(tmp_path):
    floating = tmp_path / "floating"
    floating.mkdir()
    mtp_tensors = {
        n: t
        for n, t in _checkpoint(floating, dtype=torch.bfloat16).items()
        if n.startswith("mtp.")
    }
    raw, expected = _write_mixed_checkpoint(tmp_path)
    raw.update(
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention", "moe"],
        n_shared_experts=1,
    )
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    tensors = safetensors.torch.load_file(tmp_path / "model.safetensors")
    tensors.update(mtp_tensors)
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(NotImplementedError, match="ModelOpt"):
        build(str(tmp_path), task="nemotron-h-mtp", dtype="bf16")
    package = build(str(tmp_path), task="nemotron-h-mtp", keep_quantized=False, dtype="bf16")
    for name, value in package["mtp"].graph.initializers.items():
        if name == "lm_head.weight":
            tensor = expected[name]
        elif name == "embed_tokens.weight":
            tensor = expected["model.embed_tokens.weight"]
        elif name.startswith("layers."):
            tensor = mtp_tensors[mtp_source_name(name)]
        else:
            continue
        np.testing.assert_array_equal(
            value.const_value.numpy().astype("float32"), tensor.float().numpy()
        )
    assert package.weight_loading_report["mtp_tensor_count"] == 22
    assert package.weight_loading_report["source_weight_format"] == "modelopt-mixed-fp8-nvfp4"
    assert package.weight_loading_report["generation_runtime_integration"] is False
    package.save(tmp_path / "export", external_data="onnx")


def test_cli_explicit_mtp_route(tmp_path):
    from mobius.__main__ import main

    _checkpoint(tmp_path, dtype=torch.bfloat16)
    output = tmp_path / "cli-export"
    main(
        [
            "build",
            "--config",
            str(tmp_path),
            "--output",
            str(output),
            "--task",
            "nemotron-h-mtp",
            "--dtype",
            "bf16",
            "--ep",
            "cpu",
            "--external-data",
            "onnx",
        ]
    )
    assert (output / "decoder" / "model.onnx").is_file()
    assert (output / "mtp" / "model.onnx").is_file()
    assert json.loads((output / "weight-loading-report.json").read_text())["mtp_preserved"]


def test_intrinsic_runtime_metadata_has_precise_external_contract(tmp_path):
    from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

    package = _graph_package()
    write_ort_genai_config(package, str(tmp_path))
    payload = json.loads((tmp_path / "mtp_config.json").read_text())
    assert payload["status"] == "runtime_unvalidated"
    assert payload["conditioning"]["target_hidden_output"] == "mtp_seed"
    assert payload["conditioning"]["target_hidden_normalization"] == "post_final_norm"
    assert payload["conditioning"]["token_alignment"] == "target_h_i_with_token_t_i_plus_1"
    assert payload["conditioning"]["draft_prediction"] == "token_t_i_plus_2"
    assert payload["conditioning"]["tables"] == "target_shared_sources_copied_into_draft_graph"
    assert payload["cache_namespaces"]["target"]["ports"]
    assert payload["cache_namespaces"]["mtp"]["ports"]
    assert payload["num_nextn_predict_layers"] == 1


def test_pinned_production_mtp_parameter_shapes():
    # Derived from independent range-read headers at the recipe's immutable
    # BF16/NVFP4 revisions, not inferred from this implementation's graph.
    raw = {
        **_raw(),
        "hidden_size": 2688,
        "vocab_size": 131072,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "n_routed_experts": 128,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 1856,
        "moe_shared_expert_intermediate_size": 3712,
    }
    config = NemotronHConfig.from_transformers(SimpleNamespace(**raw))
    model = NemotronHMtpModel(config)
    expected = {
        "mtp.layers.0.enorm.weight": [2688],
        "mtp.layers.0.hnorm.weight": [2688],
        "mtp.layers.0.eh_proj.weight": [2688, 5376],
        "mtp.layers.0.norm.weight": [2688],
        "mtp.layers.0.mixer.q_proj.weight": [4096, 2688],
        "mtp.layers.0.mixer.k_proj.weight": [256, 2688],
        "mtp.layers.0.mixer.v_proj.weight": [256, 2688],
        "mtp.layers.0.mixer.o_proj.weight": [2688, 4096],
        "mtp.layers.1.norm.weight": [2688],
        "mtp.layers.1.final_layernorm.weight": [2688],
        "mtp.layers.1.mixer.gate.weight": [128, 2688],
        "mtp.layers.1.mixer.gate.e_score_correction_bias": [128],
        "mtp.layers.1.mixer.shared_experts.up_proj.weight": [3712, 2688],
        "mtp.layers.1.mixer.shared_experts.down_proj.weight": [2688, 3712],
    }
    for index in range(128):
        expected[f"mtp.layers.1.mixer.experts.{index}.up_proj.weight"] = [1856, 2688]
        expected[f"mtp.layers.1.mixer.experts.{index}.down_proj.weight"] = [2688, 1856]
    actual = {
        mtp_source_name(name): list(value.shape)
        for name, value in model.named_parameters()
        if name.startswith("layers.")
    }
    assert actual == expected
    assert len(actual) == 270


def test_transformers_mtp_vocabulary_preserves_source_facts():
    from transformers import AutoConfig

    raw = _raw()
    source = AutoConfig.for_model(**raw)
    before = source.to_dict()
    config = NemotronHConfig.from_transformers(source)
    assert config.mtp_layers_block_type == ["attention", "moe"]
    assert config.num_nextn_predict_layers == 1
    assert source.to_dict() == before
    assert raw["mtp_layers_block_type"] == ["attention", "moe"]


def test_pinned_revision_reaches_config_shards_and_component_metadata(tmp_path, monkeypatch):
    from mobius.integrations.transformers import _builder

    _checkpoint(tmp_path)
    source = SimpleNamespace(**_raw())
    before = dict(source.__dict__)
    calls = []
    revision = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"

    def load_config(model_id, **kwargs):
        calls.append(("config", model_id, kwargs["revision"]))
        return source, True

    def resolve_shards(model_id, selected_revision):
        calls.append(("shards", model_id, selected_revision))
        return [str(tmp_path / "model.safetensors")]

    monkeypatch.setattr(_builder, "_load_transformers_config", load_config)
    monkeypatch.setattr(_builder, "_resolve_shard_paths", resolve_shards)
    package = build("nvidia/lightning", task="nemotron-h-mtp", revision=revision, dtype="f32")
    assert calls == [
        ("config", "nvidia/lightning", revision),
        ("shards", "nvidia/lightning", revision),
    ]
    assert source.__dict__ == before
    assert package.weight_loading_report["revision"] == revision
    assert package.weight_loading_report["mtp_component"]["revision"] == revision
    for model in package.values():
        assert model.metadata_props["mobius.source_revision"] == revision


def test_fp16_cpu_draft_prefill_and_cached_decode(tmp_path):
    weights = _checkpoint(tmp_path)
    package = build(str(tmp_path), task="nemotron-h-mtp", dtype="f16")
    session = _session(package["mtp"], tmp_path / "mtp-f16.onnx")
    reference_weights = {
        name: tensor.float() if ".gate." in name else tensor.half().float()
        for name, tensor in weights.items()
    }
    generator = torch.Generator().manual_seed(819)
    past_k = past_v = np.zeros((2, 2, 0, 8), dtype=np.float16)
    for length in (3, 1):
        ids = torch.randint(0, 48, (2, length), generator=generator)
        seed = torch.randn(2, length, 32, generator=generator).half().numpy()
        mask = torch.ones(2, past_k.shape[2] + length, dtype=torch.int64)
        expected = _reference(
            ids,
            torch.from_numpy(seed).float(),
            mask,
            torch.from_numpy(past_k).float(),
            torch.from_numpy(past_v).float(),
            reference_weights,
        )
        actual = session.run(
            None,
            {
                "input_ids": ids.numpy(),
                "hidden_states": seed,
                "attention_mask": mask.numpy(),
                "past_key_values.0.key": past_k,
                "past_key_values.0.value": past_v,
            },
        )
        for result, reference in zip(actual, expected):
            assert result.dtype == np.float16
            np.testing.assert_allclose(
                result.astype("float32"), reference.numpy(), rtol=1e-2, atol=1e-2
            )
        past_k, past_v = actual[2:]


@pytest.mark.parametrize(
    "options",
    [
        {"target_decoder_only": True},
        {"fp8_kv_cache": True},
        {"prune_prefill_prefix": True},
        {"export_paged_attention": True},
        {"output_layer_indices": [0]},
    ],
)
def test_explicit_mtp_rejects_unsupported_feature_combinations(tmp_path, options):
    (tmp_path / "config.json").write_text(json.dumps(_raw()), encoding="utf-8")
    with pytest.raises(ValueError, match="nemotron-h-mtp"):
        build(str(tmp_path), task="nemotron-h-mtp", load_weights=False, **options)
