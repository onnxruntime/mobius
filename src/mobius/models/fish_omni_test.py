# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import dataclasses
import json

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from _test_configs import FISH_OMNI_CONFIG
from onnxscript import GraphBuilder

from mobius import build_from_module
from mobius._configs.fish_omni import FishOmniConfig
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.transformers._config_resolver import (
    _config_from_hf,
    _dict_to_pretrained_config,
)
from mobius.models import FishOmniModel
from mobius.models.fish_omni import _apply_fish_attention


def tiny_config():
    return FishOmniConfig.from_transformers(
        _dict_to_pretrained_config(copy.deepcopy(FISH_OMNI_CONFIG))
    )


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.BFLOAT16])
def test_fish_omni_build(dtype):
    config = dataclasses.replace(tiny_config(), dtype=dtype)
    package = build_from_module(
        FishOmniModel(config),
        config,
        task="fish-omni",
        execution_provider="onnx-standard" if dtype == ir.DataType.BFLOAT16 else "default",
    )
    assert set(package) == {"decoder", "embedding", "audio_decoder"}
    assert set(package.policy_components) == {"main_sampler", "codebook_sampler"}
    assert {v.name for v in package["embedding"].graph.outputs} == {
        "inputs_embeds",
        "shared_input",
    }
    decoder = package["decoder"]
    assert "past_key_values.0.recurrent_state" in {v.name for v in decoder.graph.inputs}
    recurrent = next(v for v in decoder.graph.inputs if v.name.endswith("recurrent_state"))
    assert recurrent.dtype == ir.DataType.FLOAT
    assert {v.name for v in decoder.graph.outputs} >= {"logits", "hidden_states"}
    tower = package["audio_decoder"]
    if dtype == ir.DataType.BFLOAT16:
        assert sum(node.op_type == "If" for node in tower.graph) == 3
    assert tuple(tower.graph.outputs[0].shape)[-2:] == (2, 16)
    assert sum(v.name.startswith("past_key_values.") for v in tower.graph.inputs) == 6
    assert any("q_norm.weight" in key for key in tower.graph.initializers)
    assert all(
        v.const_value is not None or v.dtype == dtype
        for m in package.values()
        for v in m.graph.initializers.values()
    )


def test_fish_omni_registry_config_resolution():
    parent = _dict_to_pretrained_config(copy.deepcopy(FISH_OMNI_CONFIG))
    resolved = _config_from_hf(
        parent.text_config, parent_config=parent, module_class=registry.get("fish_omni")
    )
    assert isinstance(resolved, FishOmniConfig)
    assert resolved.partial_rotary_factor == resolved.tower.partial_rotary_factor
    assert resolved.tower.partial_rotary_factor == pytest.approx(0.25)
    assert resolved.tower.rope_interleave
    assert resolved.tower.max_position_embeddings == resolved.max_position_embeddings
    assert resolved.quantization is None


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16])
def test_fish_omni_rejects_unqualified_precision(dtype):
    with pytest.raises(ValueError, match="FP16 requires"):
        FishOmniModel(dataclasses.replace(tiny_config(), dtype=dtype))


def test_fish_omni_bfloat16_rejects_unqualified_ep():
    config = dataclasses.replace(tiny_config(), dtype=ir.DataType.BFLOAT16)
    with pytest.raises(ValueError, match="requires execution_provider='onnx-standard'"):
        build_from_module(FishOmniModel(config), config, task="fish-omni")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("text_config", "attn_output_gate", False),
        ("text_config", "layer_types", ["full_attention"]),
        ("text_config", "tie_word_embeddings", True),
        ("audio_decoder_config", "vq_feature_placement", "other"),
        ("audio_decoder_config", "dim", 64),
        ("audio_decoder_config", "rotary_interleaved", False),
        ("audio_decoder_config", "vocab_size", 17),
    ],
)
def test_fish_omni_rejects_incompatible_configs(section, key, value):
    raw = copy.deepcopy(FISH_OMNI_CONFIG)
    raw[section][key] = value
    with pytest.raises(ValueError):
        FishOmniConfig.from_transformers(_dict_to_pretrained_config(raw))


def test_fish_omni_ort_genai_metadata_rejects_incomplete_runtime(tmp_path):
    from mobius.integrations.ort_genai import write_ort_genai_config

    config = tiny_config()
    package = build_from_module(FishOmniModel(config), config, task="fish-omni")
    output = tmp_path / "ort-genai"
    with pytest.raises(ValueError, match="next-frame codebook sampling"):
        write_ort_genai_config(package, str(output), hf_model_id="must-not-download")
    assert not output.exists()


def test_fish_omni_onnx_genai_metadata_is_graph_derived_and_advisory(tmp_path):
    from mobius.integrations.onnx_genai import write_onnx_genai_config

    config = tiny_config()
    package = build_from_module(FishOmniModel(config), config, task="fish-omni")
    artifacts = write_onnx_genai_config(package, str(tmp_path))
    metadata = json.loads((tmp_path / "inference_metadata.yaml").read_text())
    assert set(artifacts) == {"inference_metadata", "runtime_compatibility"}
    assert metadata["runtime_validation_status"] == "unsupported-by-tested-runtime"
    assert set(metadata["components"]) == set(package)
    for name, model in package.items():
        component = metadata["components"][name]
        assert component["filename"] == f"{name}/model.onnx"
        assert component["inputs"] == [value.name for value in model.graph.inputs]
        assert component["outputs"] == [value.name for value in model.graph.outputs]


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="BF16 Attention requires CUDA",
)
@pytest.mark.parametrize("padded", [False, True])
def test_fish_attention_preserves_padding_and_cache(padded):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    dtype = ir.DataType.BFLOAT16
    inputs = {
        name: builder.input(name, dtype, shape)
        for name, shape in (
            ("query", [2, 2, 32]),
            ("key", [2, 2, 16]),
            ("value", [2, 2, 16]),
            ("past_key", [2, 2, 2, 8]),
            ("past_value", [2, 2, 2, 8]),
        )
    }
    bias = builder.input("bias", dtype, [2, 1, 2, 4])
    outputs = _apply_fish_attention(
        builder.op,
        inputs["query"],
        inputs["key"],
        inputs["value"],
        bias,
        inputs["past_key"],
        inputs["past_value"],
        dtype=dtype,
        num_attention_heads=4,
        num_key_value_heads=2,
        scale=8**-0.5,
    )
    for name, value, shape in zip(
        ("output", "present_key", "present_value"),
        outputs,
        ([2, 2, 32], [2, 2, 4, 8], [2, 2, 4, 8]),
    ):
        value.type, value.shape = ir.TensorType(dtype), ir.Shape(shape)
        builder.add_output(value, name)
    rng = np.random.default_rng(42)
    feeds = {
        name: rng.normal(0, 0.1, tuple(value.shape)).astype(ml_dtypes.bfloat16)
        for name, value in inputs.items()
    }
    mask = np.array([[0, 0, 0, -np.inf], [0, 0, 0, 0]], dtype=np.float32)
    feeds["bias"] = np.broadcast_to(mask, (2, 1, 2, 4)).copy()
    if padded:
        feeds["bias"][0, :, :, 0] = -np.inf
    feeds["bias"] = feeds["bias"].astype(ml_dtypes.bfloat16)
    tensors = {
        name: torch.tensor(value.astype(np.float32), device="cuda").to(torch.bfloat16)
        for name, value in feeds.items()
        if name != "bias"
    }
    q = tensors["query"].reshape(2, 2, 4, 8).transpose(1, 2)
    k = torch.cat(
        (tensors["past_key"], tensors["key"].reshape(2, 2, 2, 8).transpose(1, 2)), dim=2
    )
    v = torch.cat(
        (tensors["past_value"], tensors["value"].reshape(2, 2, 2, 8).transpose(1, 2)), dim=2
    )
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q,
            k.repeat_interleave(2, dim=1),
            v.repeat_interleave(2, dim=1),
            attn_mask=torch.tensor(np.isfinite(feeds["bias"]), device="cuda"),
        )
        .transpose(1, 2)
        .reshape(2, 2, 32)
    )
    ort.preload_dlls()
    session = OnnxModelSession(ir.Model(graph, ir_version=12), device="cuda")
    try:
        result = session.run(feeds)
        np.testing.assert_allclose(
            result["output"].astype(np.float32),
            expected.float().cpu().numpy(),
            atol=1e-2,
            rtol=1e-2,
        )
        np.testing.assert_array_equal(
            result["present_key"].astype(np.float32), k.float().cpu().numpy()
        )
        np.testing.assert_array_equal(
            result["present_value"].astype(np.float32), v.float().cpu().numpy()
        )
        if padded:
            feeds["past_value"][0, :, 0] = 1000
            changed = session.run(feeds)
            np.testing.assert_array_equal(result["output"], changed["output"])
    finally:
        session.close()
