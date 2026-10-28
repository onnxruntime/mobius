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
from onnxscript import GraphBuilder, nn

from mobius import build_from_module
from mobius._build_context import ep_capabilities
from mobius._builder import _cast_module_dtype
from mobius._configs.fish_omni import FishOmniConfig
from mobius._model_package import ModelPackage
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import Linear, OffsetRMSNorm, OutputMajorLinear
from mobius.components._ordered_reduction_test import _reference_row_reduce
from mobius.generation._fish_omni_context import (
    SOURCE_CONTEXT_CONTRACT,
    SOURCE_CONTEXT_METADATA,
)
from mobius.integrations.transformers._config_resolver import (
    _config_from_hf,
    _dict_to_pretrained_config,
)
from mobius.models import FishOmniModel
from mobius.models.fish_omni import (
    _apply_fish_attention,
    _FishGatedNorm,
    _FishMoEBlock,
    _FishSourceGatedDeltaNet,
    _FishSourceGatedNorm,
    _FishSourceOffsetNorm,
    _FP32OutputLinear,
    _sigmoid_float32,
)
from mobius.tasks import FishOmniGenerationTask, FishOmniTask, ModelTask
from mobius.tasks._base import _make_graph, _make_model


def tiny_config():
    return FishOmniConfig.from_transformers(
        _dict_to_pretrained_config(copy.deepcopy(FISH_OMNI_CONFIG))
    )


def learned_parameter_shapes(module):
    # Runtime RoPE replaces nonlearned low-dtype caches, never checkpoint weights.
    return {
        name: list(parameter.shape)
        for name, parameter in module.named_parameters()
        if not name.endswith(("rotary_emb.cos_cache", "rotary_emb.sin_cache"))
    }


class _ReducedPrecisionQualificationOnly:
    def _validate_precision(self, config):
        if not config.native_fp8_experts and config.dtype not in (
            ir.DataType.FLOAT16,
            ir.DataType.BFLOAT16,
        ):
            return super()._validate_precision(config)
        expected = (
            "cuda"
            if config.dtype == ir.DataType.FLOAT16 or config.native_fp8_experts
            else "onnx-standard"
        )
        if ep_capabilities().name != expected:
            raise ValueError("Qualification probe has the wrong execution provider")

    def build(self, module, config):
        package = super().build(module, config)
        for model in package.values():
            model.metadata_props["mobius.precision_validation"] = (
                "unqualified-native-fp8-prototype"
                if config.native_fp8_experts
                else "unqualified-reduced-precision-prototype"
            )
        return package


class _TeacherProbe(_ReducedPrecisionQualificationOnly, FishOmniTask):
    pass


class _GenerationProbe(_ReducedPrecisionQualificationOnly, FishOmniGenerationTask):
    pass


def reduced_precision_prototype_task(task):
    """Test only: probe prepared math without opening production admission."""
    return _GenerationProbe() if task == "fish-omni-generation" else _TeacherProbe()


def native_fp8_prototype_config(dtype=ir.DataType.FLOAT16):
    config = tiny_config()
    return dataclasses.replace(
        config,
        dtype=dtype,
        hidden_size=128,
        moe_intermediate_size=128,
        fp8_experts=True,
        native_fp8_experts=True,
        tower=dataclasses.replace(config.tower, hidden_size=128),
    )


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_fish_projection_layout_is_scoped_and_preserves_weight_names(dtype):
    baseline = FishOmniModel(tiny_config())
    module = FishOmniModel(dataclasses.replace(tiny_config(), dtype=dtype))
    assert learned_parameter_shapes(module) == learned_parameter_shapes(baseline)
    for component in (module.decoder, module.audio_decoder):
        projections = [child for child in component.modules() if isinstance(child, Linear)]
        assert projections
        if dtype == ir.DataType.FLOAT:
            assert all(
                type(child) is Linear or child is module.decoder.lm_head
                for child in projections
            )
        else:
            assert all(isinstance(child, OutputMajorLinear) for child in projections)
    # Constructing Fish modules must never change another architecture's Linear.
    graph, builder = _make_graph("unrelated_projection")
    x = builder.input("x", ir.DataType.FLOAT, [1, 2, 4])
    Linear(4, 3, bias=False)(builder.op, x)
    assert any(node.op_type == "MatMul" for node in graph)
    assert not any(node.op_type == "Gemm" for node in graph)


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_fish_norm_selection_is_scoped_and_preserves_parameter_paths(dtype):
    baseline = FishOmniModel(tiny_config())
    module = FishOmniModel(dataclasses.replace(tiny_config(), dtype=dtype))
    assert learned_parameter_shapes(module) == learned_parameter_shapes(baseline)
    baseline_offsets = {
        name for name, child in baseline.named_modules() if type(child) is OffsetRMSNorm
    }
    assert baseline_offsets
    actual_offsets = {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, OffsetRMSNorm)
    }
    assert actual_offsets.keys() == baseline_offsets
    expected_class = OffsetRMSNorm if dtype == ir.DataType.FLOAT else _FishSourceOffsetNorm
    assert all(type(child) is expected_class for child in actual_offsets.values())
    norms = [child for child in module.modules() if isinstance(child, _FishGatedNorm)]
    assert norms
    expected_class = _FishGatedNorm if dtype == ir.DataType.FLOAT else _FishSourceGatedNorm
    assert all(type(child) is expected_class for child in norms)
    graph, builder = _make_graph("unrelated_norm")
    x = builder.input("x", ir.DataType.FLOAT, [1, 32])
    OffsetRMSNorm(32)(builder.op, x)
    assert any(node.op_type == "RMSNormalization" for node in graph)
    assert not any(node.op_type == "Pad" for node in graph)


@pytest.mark.parametrize("kind", ["offset", "gated"])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
@pytest.mark.parametrize("shape", [(256,), (17, 256), (2, 2, 4, 256), (2, 0, 256)])
def test_fish_source_norm_numeric_rounding_and_shape(kind, dtype, shape):
    width = shape[-1]
    norm = _FishSourceOffsetNorm(width) if kind == "offset" else _FishSourceGatedNorm(width)
    rng = np.random.default_rng(74)
    values = (rng.standard_normal(shape) * 300).astype(dtype.numpy())
    weight = rng.uniform(-0.2, 0.2, width).astype(dtype.numpy())
    gate = rng.uniform(-3, 3, shape).astype(dtype.numpy())
    _cast_module_dtype(norm, dtype)
    norm.weight = nn.Parameter([width], data=ir.tensor(weight))
    graph, builder = _make_graph("source_norm")
    x = builder.input("x", dtype, [*[f"d{i}" for i in range(len(shape) - 1)], width])
    if kind == "gated":
        g = builder.input("gate", dtype, list(x.shape))
        y = norm(builder.op, x, g)
    else:
        y = norm(builder.op, x)
    assert y.shape == x.shape
    y.type = ir.TensorType(dtype)
    builder.add_output(y, "y")
    assert not any(node.op_type == "RMSNormalization" for node in graph)
    assert all(node.domain == "" for node in graph)
    wide = values.astype(np.float32)
    variance = _reference_row_reduce(wide * wide, normalize=True)
    normalized = wide * np.power(variance + np.float32(1e-6), np.float32(-0.5))
    if kind == "offset":
        expected = (normalized * (weight.astype(np.float32) + 1)).astype(dtype.numpy())
    else:
        # Two distinct low-dtype boundaries precede the FP32 gate multiplication.
        rounded = normalized.astype(dtype.numpy())
        scaled = (rounded.astype(np.float32) * weight.astype(np.float32)).astype(dtype.numpy())
        wide_gate = gate.astype(np.float32)
        silu = wide_gate / (1 + np.exp(-wide_gate))
        expected = (scaled.astype(np.float32) * silu).astype(dtype.numpy())
    feeds = {"x": values, **({"gate": gate} if kind == "gated" else {})}
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        actual = session.run(feeds)["y"]
    finally:
        session.close()
    assert actual.shape == shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    if kind == "gated" and values.size:
        unrounded = (normalized * weight.astype(np.float32) * silu).astype(dtype.numpy())
        assert np.any(unrounded.view(np.uint16) != expected.view(np.uint16))


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_fish_source_offset_norm_keeps_fp32_residual_and_low_dtype_weight(dtype):
    norm = _FishSourceOffsetNorm(32)
    _cast_module_dtype(norm, dtype)
    weight = np.full(32, 0.0004, dtype=dtype.numpy())
    norm.weight = nn.Parameter([32], data=ir.tensor(weight))
    graph, builder = _make_graph("residual_norm")
    x = builder.input("x", ir.DataType.FLOAT, ["rows", 32])
    y = norm(builder.op, x)
    y.type = ir.TensorType(ir.DataType.FLOAT)
    builder.add_output(y, "y")
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        actual = session.run({"x": np.full((3, 32), 120_000, dtype=np.float32)})["y"]
    finally:
        session.close()
    assert actual.dtype == np.float32
    expected = np.broadcast_to(1 + weight.astype(np.float32), (3, 32))
    np.testing.assert_allclose(actual, expected, rtol=2e-7)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_native_fp8_prototype_preserves_storage_and_dynamic_runtime_contract(dtype):
    config = native_fp8_prototype_config(dtype)
    package = build_from_module(
        FishOmniModel(config),
        config,
        task=_TeacherProbe(),
        execution_provider="cuda",
    )
    decoder = package["decoder"]
    nodes = [node for node in decoder.graph if node.op_type == "QMoE"]
    assert len(nodes) == config.num_hidden_layers
    for node in nodes:
        assert node.domain == "com.microsoft"
        assert node.attributes["activation_quantization"].as_string() == "dynamic_fp8"
        assert node.attributes["block_size"].as_int() == 128
        assert node.attributes["expert_weight_bits"].as_int() == 8
        assert node.attributes["swiglu_fusion"].as_int() == 0
        assert node.inputs[4] is None and node.inputs[7] is None
        assert all(node.inputs[i].dtype == ir.DataType.FLOAT8E4M3FN for i in (2, 5, 8))
        assert all(node.inputs[i].dtype == ir.DataType.FLOAT for i in (3, 6, 9))
        assert node.inputs[1].dtype == dtype
    banks = {
        name: value
        for name, value in decoder.graph.initializers.items()
        if ".packed_experts." in name
    }
    assert len(banks) == config.num_hidden_layers * 6
    assert all(
        value.dtype
        == (
            ir.DataType.FLOAT
            if name.endswith(".weight_scale_inv")
            else ir.DataType.FLOAT8E4M3FN
        )
        for name, value in banks.items()
    )
    assert "dynamic_fp8" in decoder.metadata_props["mobius.required_runtime_features"]
    assert decoder.metadata_props["mobius.precision_validation"].startswith("unqualified")
    with pytest.raises(ValueError, match="disabled pending real-weight"):
        build_from_module(
            FishOmniModel(config), config, task="fish-omni", execution_provider="cuda"
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "dtype,torch_dtype,numpy_dtype",
    [
        (ir.DataType.FLOAT16, torch.float16, np.float16),
        (ir.DataType.BFLOAT16, torch.bfloat16, ml_dtypes.bfloat16),
    ],
)
def test_fish_sigmoid_preserves_native_negative_tail_rounding(
    device, dtype, torch_dtype, numpy_dtype
):
    if device == "cuda" and (
        not torch.cuda.is_available()
        or "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        pytest.skip("CUDA sigmoid parity requires Torch and ORT CUDA")
    values = np.array(
        [
            -1000.0,
            -20.0,
            -15.4453125,
            -12.828125,
            -12.46875,
            -9.5234375,
            -9.1328125,
            -8.859375,
            -8.3984375,
            -8.3671875,
            -7.6015625,
            -7.08984375,
            -5.54296875,
            -3.875,
            -2.443359375,
            0.0,
            20.0,
            1000.0,
        ],
        dtype=numpy_dtype,
    )
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    x = builder.input("x", dtype, [len(values)])
    y = _sigmoid_float32(builder.op, x)
    y.type, y.shape = ir.TensorType(dtype), ir.Shape([len(values)])
    builder.add_output(y, "y")
    assert not any(node.op_type == "Sigmoid" for node in graph)
    reference = torch.tensor(values.astype(np.float32), device=device, dtype=torch_dtype)
    expected = reference.sigmoid().float().cpu().numpy()
    session = OnnxModelSession(ir.Model(graph, ir_version=12), device=device)
    try:
        actual = session.run({"x": values})["y"].astype(np.float32)
    finally:
        session.close()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_fish_omni_build(dtype):
    config = dataclasses.replace(tiny_config(), dtype=dtype)
    package = build_from_module(
        FishOmniModel(config),
        config,
        task=reduced_precision_prototype_task("fish-omni")
        if dtype != ir.DataType.FLOAT
        else "fish-omni",
        execution_provider={
            ir.DataType.FLOAT: "default",
            ir.DataType.FLOAT16: "cuda",
            ir.DataType.BFLOAT16: "onnx-standard",
        }[dtype],
    )
    assert set(package) == {"decoder", "embedding", "audio_decoder"}
    assert set(package.policy_components) == {"main_sampler", "codebook_sampler"}
    assert {v.name for v in package["embedding"].graph.outputs} == {
        "inputs_embeds",
        "shared_input",
    }
    decoder = package["decoder"]
    head = decoder.graph.initializers["decoder.lm_head.weight"]
    consumers = tuple(head.consumers())
    assert consumers
    assert all(node.op_type == "Gemm" for node in consumers)
    assert "past_key_values.0.recurrent_state" in {v.name for v in decoder.graph.inputs}
    recurrent = next(v for v in decoder.graph.inputs if v.name.endswith("recurrent_state"))
    assert recurrent.dtype == ir.DataType.FLOAT
    assert {v.name for v in decoder.graph.outputs} >= {"logits", "hidden_states"}
    tower = package["audio_decoder"]
    if dtype == ir.DataType.BFLOAT16:
        assert sum(node.op_type == "If" for node in tower.graph) == 3
        assert sum(node.op_type == "Gemm" for node in tower.graph) == 20
    assert tuple(tower.graph.outputs[0].shape)[-2:] == (2, 16)
    assert sum(v.name.startswith("past_key_values.") for v in tower.graph.inputs) == 6
    assert any("q_norm.weight" in key for key in tower.graph.initializers)
    assert all(
        v.const_value is not None or v.dtype == dtype
        for m in package.values()
        for v in m.graph.initializers.values()
    )
    rotary_caches = [
        value
        for model in package.values()
        for name, value in model.graph.initializers.items()
        if name.endswith((".cos_cache", ".sin_cache"))
    ]
    assert bool(rotary_caches) == (dtype == ir.DataType.FLOAT)
    assert all(value.dtype == ir.DataType.FLOAT for value in rotary_caches)
    for model in (decoder, tower):
        norms = [node for node in model.graph if node.op_type == "RMSNormalization"]
        if dtype == ir.DataType.FLOAT:
            assert norms
        else:
            assert not norms
            assert any(node.op_type == "Pad" for node in model.graph)
            assert any(node.op_type == "Pow" for node in model.graph)
            assert not any("Ordered" in node.op_type for node in model.graph)
            assert model.metadata_props["mobius.precision_validation"].startswith(
                "unqualified"
            )
    if dtype == ir.DataType.FLOAT:
        assert SOURCE_CONTEXT_METADATA not in decoder.metadata_props
    else:
        assert decoder.metadata_props[SOURCE_CONTEXT_METADATA] == SOURCE_CONTEXT_CONTRACT
        assert (
            "no mixed/static/capture qualification"
            in decoder.metadata_props["mobius.fish_omni.context_limits"]
        )
        assert (
            "not-cuBLAS-bit-parity"
            in decoder.metadata_props["mobius.fish_omni.gdn_arithmetic"]
        )
        assert not any(
            node.op_type
            in ("LinearAttention", "CausalConvWithState", "UnitLowerTriangularSolve")
            for node in ir.traversal.RecursiveGraphIterator(decoder.graph)
        )
        assert not decoder.functions


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_source_gdn_dispatch_uses_continuation_and_length_not_cache_values(dtype):
    module = _FishSourceGatedDeltaNet(dataclasses.replace(tiny_config(), dtype=dtype))
    graph, builder = _make_graph("source_dispatch")
    x = builder.input("x", dtype, ["batch", "time", module.hidden_size])
    conv = builder.input("conv", dtype, ["batch", module.conv_dim, 3])
    state = builder.input(
        "state",
        ir.DataType.FLOAT,
        ["batch", module.num_v_heads, module.head_k_dim, module.head_v_dim],
    )
    continuing = builder.input("continuing", ir.DataType.BOOL, [])
    module(builder.op, x, conv, state, continuing)
    dispatch = next(node for node in graph if node.op_type == "If")
    condition = dispatch.inputs[0].producer()
    assert condition.op_type == "And"
    assert continuing in condition.inputs
    length_test = next(
        value.producer() for value in condition.inputs if value is not continuing
    )
    assert length_test.op_type == "Equal"
    shape = length_test.inputs[0].producer().inputs[0].producer()
    assert shape.op_type == "Shape"
    assert shape.inputs[0] is x
    assert shape.attributes["start"].as_int() == 1
    assert shape.attributes["end"].as_int() == 2
    assert "recurrent" in dispatch.attributes["then_branch"].as_graph().name
    assert "chunks" in dispatch.attributes["else_branch"].as_graph().name
    runtime_exp = next(
        node
        for node in graph
        if node.op_type == "Exp" and node.inputs[0].producer().op_type == "Expand"
    )
    assert runtime_exp.inputs[0].producer().inputs[1].producer().op_type == "Concat"


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


def test_fish_omni_cuda_build_uses_sparse_fused_expert_banks():
    config = tiny_config()
    package = build_from_module(
        FishOmniModel(config), config, task="fish-omni", execution_provider="cuda"
    )
    graph = package["decoder"].graph
    assert sum(node.op_type == "MoE" for node in graph) == config.num_hidden_layers
    assert not any(node.op_type == "TopK" for node in graph)
    banks = [value for name, value in graph.initializers.items() if ".packed_experts." in name]
    assert len(banks) == config.num_hidden_layers * 2
    assert all(len(value.shape) == 3 for value in banks)
    for node in graph:
        if node.op_type == "MoE":
            assert len(node.inputs) <= 6 or node.inputs[6] is None
            assert node.attributes["activation_type"].as_string() == "swiglu"
            assert node.attributes["swiglu_fusion"].as_int() == 2
    assert not any(".experts.0." in name for name in graph.initializers)


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_fish_single_pass_task_reuses_tower_weights_and_layer_count(dtype):
    config = dataclasses.replace(tiny_config(), dtype=dtype)
    ep = {
        ir.DataType.FLOAT: "default",
        ir.DataType.FLOAT16: "cuda",
        ir.DataType.BFLOAT16: "onnx-standard",
    }[dtype]
    teacher = build_from_module(
        FishOmniModel(config),
        config,
        task=reduced_precision_prototype_task("fish-omni")
        if dtype != ir.DataType.FLOAT
        else "fish-omni",
        execution_provider=ep,
    )["audio_decoder"]
    generated = build_from_module(
        FishOmniModel(config),
        config,
        task=reduced_precision_prototype_task("fish-omni-generation")
        if dtype != ir.DataType.FLOAT
        else "fish-omni-generation",
        execution_provider=ep,
    )["audio_generator"]
    assert {
        name for name, value in teacher.graph.initializers.items() if value.const_value is None
    } == {
        name
        for name, value in generated.graph.initializers.items()
        if value.const_value is None
    }
    assert {value.name for value in generated.graph.outputs} >= {
        "frame_codes",
        "valid",
        "next_counter",
        "codebook_logits",
    }
    assert (
        sum(value.name.startswith("past_key_values.") for value in generated.graph.inputs) == 6
    )
    assert sum(
        node.op_type == "Attention"
        for node in ir.traversal.RecursiveGraphIterator(teacher.graph)
    ) == sum(
        node.op_type == "Attention"
        for node in ir.traversal.RecursiveGraphIterator(generated.graph)
    )


@pytest.mark.parametrize("dtype", [ir.DataType.DOUBLE])
def test_fish_omni_rejects_unqualified_precision(dtype):
    with pytest.raises(ValueError, match="requires floating"):
        FishOmniModel(dataclasses.replace(tiny_config(), dtype=dtype))


@pytest.mark.parametrize(
    "dtype,ep",
    [(ir.DataType.BFLOAT16, "onnx-standard"), (ir.DataType.FLOAT16, "cuda")],
)
def test_fish_omni_reduced_precision_rejects_unqualified_ep(dtype, ep):
    config = dataclasses.replace(tiny_config(), dtype=dtype)
    with pytest.raises(ValueError, match=f"requires execution_provider='{ep}'"):
        build_from_module(FishOmniModel(config), config, task="fish-omni")


@pytest.mark.parametrize(
    "dtype,ep",
    [(ir.DataType.FLOAT16, "cuda"), (ir.DataType.BFLOAT16, "onnx-standard")],
)
def test_reduced_precision_production_export_stays_closed_after_real_weight_failure(dtype, ep):
    config = dataclasses.replace(tiny_config(), dtype=dtype)
    with pytest.raises(ValueError, match="disabled pending real-weight numerical"):
        build_from_module(
            FishOmniModel(config), config, task="fish-omni", execution_provider=ep
        )


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


@pytest.mark.parametrize("task", ["fish-omni", "fish-omni-generation"])
def test_fish_omni_ort_genai_metadata_rejects_incomplete_runtime(tmp_path, task):
    from mobius.integrations.ort_genai import write_ort_genai_config

    config = tiny_config()
    package = build_from_module(FishOmniModel(config), config, task=task)
    output = tmp_path / "ort-genai"
    with pytest.raises(ValueError, match="next-frame codebook sampling"):
        write_ort_genai_config(package, str(output), hf_model_id="must-not-download")
    assert not output.exists()


@pytest.mark.parametrize("task", ["fish-omni", "fish-omni-generation"])
def test_fish_omni_onnx_genai_metadata_is_graph_derived_and_advisory(tmp_path, task):
    from mobius.integrations.onnx_genai import write_onnx_genai_config

    config = tiny_config()
    package = build_from_module(FishOmniModel(config), config, task=task)
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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="Native BF16 projection layout parity requires CUDA",
)
def test_bf16_down_projection_preserves_native_output_major_accumulation():
    class ProjectionTask(ModelTask):
        def build(self, module, config):
            graph, builder = _make_graph("projection")
            x = builder.input("x", config.dtype, [1, 3, 6144])
            output = module(builder.op, x)
            output.type = ir.TensorType(config.dtype)
            output.shape = ir.Shape([1, 3, 2048])
            builder.add_output(output, "output")
            return ModelPackage({"projection": _make_model(graph)})

    config = dataclasses.replace(tiny_config(), dtype=ir.DataType.BFLOAT16)
    graph = build_from_module(
        OutputMajorLinear(6144, 2048, bias=False),
        config,
        task=ProjectionTask(),
        execution_provider="onnx-standard",
    )["projection"]
    gemm = next(node for node in graph.graph if node.op_type == "Gemm")
    assert gemm.attributes["transB"].as_int() == 1
    assert not any(node.op_type == "Transpose" for node in graph.graph)
    rng = torch.Generator(device="cuda").manual_seed(74)
    weight = torch.randn(2048, 6144, dtype=torch.bfloat16, device="cuda", generator=rng)
    x = torch.randn(3, 6144, dtype=torch.bfloat16, device="cuda", generator=rng)
    expected = torch.nn.functional.linear(x, weight).float().cpu().numpy()[None]
    parameter = next(
        value for value in graph.graph.initializers.values() if value.const_value is None
    )
    parameter.const_value = ir.tensor(weight.float().cpu().numpy().astype(ml_dtypes.bfloat16))
    session = OnnxModelSession(graph, device="cuda")
    try:
        actual = session.run({"x": x.float().cpu().numpy().astype(ml_dtypes.bfloat16)[None]})[
            "output"
        ].astype(np.float32)
    finally:
        session.close()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="Mixed-output FP16 GEMM requires CUDA",
)
def test_fp16_projection_preserves_values_above_half_range():
    class ProjectionTask(ModelTask):
        def build(self, module, config):
            graph, builder = _make_graph("projection")
            x = builder.input("x", ir.DataType.FLOAT16, [1, 2, 4])
            output = module(builder.op, x)
            output.type = ir.TensorType(ir.DataType.FLOAT)
            output.shape = ir.Shape([1, 2, 2])
            builder.add_output(output, "output")
            return ModelPackage({"projection": _make_model(graph)})

    config = dataclasses.replace(tiny_config(), dtype=ir.DataType.FLOAT16)
    graph = build_from_module(
        _FP32OutputLinear(4, 2), config, task=ProjectionTask(), execution_provider="cuda"
    )["projection"]
    weights = [
        value for value in graph.graph.initializers.values() if value.const_value is None
    ]
    assert len(weights) == 1
    assert weights[0].dtype == ir.DataType.FLOAT16
    weights[0].const_value = ir.tensor(np.full((2, 4), 1000, dtype=np.float16))
    session = OnnxModelSession(graph, device="cuda")
    try:
        actual = session.run({"x": np.full((1, 2, 4), 1000, dtype=np.float16)})["output"]
    finally:
        session.close()
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.full((1, 2, 2), 4_000_000, dtype=np.float32))


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="FP16 residual stress test requires CUDA",
)
def test_fp16_backbone_residual_stream_does_not_overflow():
    class ConstantMLP(nn.Module):
        def forward(self, op, hidden):
            return op.Expand(op.CastLike(60000.0, hidden), op.Shape(hidden))

    class ConstantLinearAttention(ConstantMLP):
        def forward(self, op, hidden, conv_state, recurrent_state):
            return super().forward(op, hidden), op.Abs(conv_state), op.Abs(recurrent_state)

    class ConstantAttention(ConstantMLP):
        def forward(self, op, hidden, bias, positions, past):
            return super().forward(op, hidden), (op.Abs(past[0]), op.Abs(past[1]))

    config = dataclasses.replace(tiny_config(), dtype=ir.DataType.FLOAT16)
    module = FishOmniModel(config)
    for layer in module.decoder.model.layers:
        layer.mlp = ConstantMLP()
        if layer.layer_type == "linear_attention":
            layer.linear_attn = ConstantLinearAttention()
        else:
            layer.self_attn = ConstantAttention()
    graph = build_from_module(
        module,
        config,
        task=reduced_precision_prototype_task("fish-omni"),
        execution_provider="cuda",
    )["decoder"]
    for value in graph.graph.initializers.values():
        if value.const_value is None:
            value.const_value = ir.tensor(np.zeros(tuple(value.shape), dtype=np.float16))
    cache = {}
    for value in graph.graph.inputs:
        if value.name.startswith("past_key_values."):
            shape = list(value.shape)
            shape[0] = 1
            if value.name.endswith((".key", ".value")):
                shape[2] = 0
            dtype = np.float32 if value.dtype == ir.DataType.FLOAT else np.float16
            cache[value.name] = np.zeros(shape, dtype=dtype)
    session = OnnxModelSession(graph, device="cuda")
    try:
        result = session.run(
            {
                **cache,
                "inputs_embeds": np.full((1, 1, config.hidden_size), 60000, dtype=np.float16),
                "attention_mask": np.ones((1, 1), np.int64),
                "position_ids": np.zeros((1, 1), np.int64),
            }
        )
    finally:
        session.close()
    # Two 60000-valued sublayer outputs per layer exceed FP16 residual range.
    # Normalization of the finite FP32 residual is exactly one after FP16 rounding.
    np.testing.assert_array_equal(
        result["hidden_states"], np.ones((1, 1, config.hidden_size), dtype=np.float16)
    )


class _MoEOnlyTask(ModelTask):
    def build(self, module, config):
        graph, builder = _make_graph("experts")
        x = builder.input("x", config.dtype, ["tokens", config.hidden_size])
        output = module(builder.op, x)
        output.type = ir.TensorType(config.dtype)
        output.shape = ir.Shape(["tokens", config.hidden_size])
        builder.add_output(output, "output")
        return ModelPackage({"experts": _make_model(graph)})


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="FP16 sparse expert overflow test requires CUDA",
)
def test_unselected_fp16_expert_overflow_cannot_poison_output():
    config = dataclasses.replace(
        tiny_config(), dtype=ir.DataType.FLOAT16, num_local_experts=2, num_experts_per_tok=1
    )
    model = build_from_module(
        _FishMoEBlock(config), config, task=_MoEOnlyTask(), execution_provider="onnx-standard"
    )["experts"]
    for name, value in model.graph.initializers.items():
        if value.const_value is not None:
            continue
        data = np.zeros(tuple(value.shape), np.float16)
        if name == "gate.weight":
            data[0, 0], data[1, 0] = 1, -1
        elif "packed_experts" in name:
            # Expert 1's finite projections overflow if executed; it is never selected.
            data[1] = 100
        value.const_value = ir.tensor(data)
    session = OnnxModelSession(model, device="cuda")
    try:
        output = session.run({"x": np.ones((1, config.hidden_size), np.float16)})["output"]
    finally:
        session.close()
    np.testing.assert_array_equal(output, np.zeros_like(output))


@pytest.mark.parametrize("tokens", [1, 16, 17, 31, 32, 33])
def test_sparse_expert_chunk_boundaries_match_selected_only_math(tokens):
    config = tiny_config()
    model = build_from_module(
        _FishMoEBlock(config), config, task=_MoEOnlyTask(), execution_provider="onnx-standard"
    )["experts"]
    rng = np.random.default_rng(42)
    weights = {}
    for name, value in model.graph.initializers.items():
        if value.const_value is None:
            data = rng.normal(0, 0.1, tuple(value.shape)).astype(np.float32)
            value.const_value = ir.tensor(data)
            weights[name] = torch.tensor(data)
    x = torch.tensor(rng.normal(0, 0.1, (tokens, config.hidden_size)).astype(np.float32))
    probabilities = torch.nn.functional.linear(x, weights["gate.weight"]).softmax(-1)
    routing, ids = probabilities.topk(config.num_experts_per_tok, dim=-1)
    routing /= routing.sum(-1, keepdim=True)
    expected = torch.zeros_like(x)
    for row in range(tokens):
        for rank in range(config.num_experts_per_tok):
            expert = ids[row, rank]
            gate, up = torch.nn.functional.linear(
                x[row], weights["packed_experts.gate_up_proj.weight"][expert]
            ).chunk(2, dim=-1)
            projected = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up,
                weights["packed_experts.down_proj.weight"][expert],
            )
            expected[row] += routing[row, rank] * projected
    shared = torch.nn.functional.silu(
        torch.nn.functional.linear(x, weights["shared_expert.gate_proj.weight"])
    ) * torch.nn.functional.linear(x, weights["shared_expert.up_proj.weight"])
    shared = torch.nn.functional.linear(shared, weights["shared_expert.down_proj.weight"])
    expected += (
        shared * torch.nn.functional.linear(x, weights["shared_expert_gate.weight"]).sigmoid()
    )
    session = OnnxModelSession(model)
    try:
        actual = session.run({"x": x.numpy()})["output"]
    finally:
        session.close()
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-4, rtol=1e-4)
