# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Packed Ornith ABI and CUDA state/sequence isolation regression tests."""

import dataclasses
import hashlib
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import nn
from ornith_engine_export import (
    PackedDecoder,
    PackedMtp,
    PackedOrnithTask,
    bind_component_weights,
    engine_config,
    fuse_gated_rmsnorm,
    fuse_int2_swiglu,
    fuse_shared_expert_gates,
    pack_int2_projections,
)
from safetensors.numpy import save_file

from mobius._builder import build_from_module
from mobius._configs import Qwen35MtpConfig
from mobius._testing import make_config
from mobius.models.qwen35 import Qwen35VLDecoderModel
from mobius.models.qwen35_mtp import Qwen35MtpModel
from mobius.tasks._base import _make_graph, _make_model


@pytest.mark.parametrize("activation", ["Swish", "QuickGelu"])
@pytest.mark.parametrize(
    "reject", [None, "axis", "alpha", "shared", "width", "dtype", "epsilon"]
)
def test_gated_rmsnorm_rewrite(activation, reject):
    from onnx import TensorProto
    from onnx import helper as h

    width = 64 if reject == "width" else 128
    shape = [2, width]
    half = TensorProto.FLOAT if reject == "dtype" else TensorProto.FLOAT16
    nodes = [
        h.make_node("Cast", ["x"], ["xf"], to=TensorProto.FLOAT),
        h.make_node("Cast", ["gate"], ["gf"], to=TensorProto.FLOAT),
        h.make_node(
            "RMSNormalization",
            ["xf", "scale"],
            ["norm"],
            axis=0 if reject == "axis" else -1,
            stash_type=1,
            epsilon=-1.0 if reject == "epsilon" else 1e-6,
        ),
        h.make_node(
            activation,
            ["gf"],
            ["activated"],
            domain="com.microsoft" if activation == "QuickGelu" else "",
            alpha=0.5 if reject == "alpha" else 1.0,
        ),
        h.make_node("Mul", ["norm", "activated"], ["product"]),
        h.make_node("Cast", ["product"], ["y"], to=TensorProto.FLOAT16),
    ]
    outputs = [h.make_tensor_value_info("y", TensorProto.FLOAT16, shape)]
    if reject == "shared":
        outputs.append(h.make_tensor_value_info("xf", TensorProto.FLOAT, shape))
    proto = h.make_model(
        h.make_graph(
            nodes,
            "norm",
            [
                h.make_tensor_value_info("x", half, shape),
                h.make_tensor_value_info("gate", TensorProto.FLOAT16, shape),
                h.make_tensor_value_info("scale", TensorProto.FLOAT, [width]),
            ],
            outputs,
            value_info=[
                h.make_tensor_value_info(name, TensorProto.FLOAT, shape)
                for name in ("xf", "gf", "norm", "activated", "product")
            ],
        ),
        opset_imports=[h.make_opsetid("", 24), h.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    model = ir.serde.deserialize_model(proto)
    assert fuse_gated_rmsnorm(model) == (0 if reject else 1)
    if not reject:
        node = next(iter(model.graph))
        assert node.op_type == "GatedRMSNormExact"
        assert [v.name for v in node.inputs] == ["x", "gate", "scale"]
        assert [v.name for v in node.outputs] == ["y"]
        assert fuse_gated_rmsnorm(model) == 0


def _config(**kwargs):
    values = dict(
        hidden_size=128,
        vocab_size=256,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        dtype=ir.DataType.FLOAT16,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        mrope_section=[6, 5, 5],
        mrope_interleaved=True,
        max_position_embeddings=128,
    )
    values.update(kwargs)
    return make_config(**values)


def _module(config):
    module = nn.Module()
    module.decoder = PackedDecoder(Qwen35VLDecoderModel(config))
    return module


def _initialize(module):
    rng = np.random.default_rng(42)
    for name, parameter in module.named_parameters():
        if parameter.const_value is not None:
            continue
        shape = tuple(parameter.shape)
        if "norm" in name:
            data = (
                np.ones(shape)
                if ".norm.weight" in name and "linear_attn" in name
                else np.zeros(shape)
            )
        elif name.endswith("A_log"):
            data = np.full(shape, -0.5)
        elif name.endswith("dt_bias"):
            data = np.full(shape, -0.2)
        else:
            data = rng.normal(0, 0.03, shape)
        parameter.const_value = ir.tensor(data.astype(np.float32))


@pytest.mark.parametrize("recurrent_state_fp16", [False, True])
def test_packed_target_and_mtp_abi(recurrent_state_fp16):
    config = _config()
    package = build_from_module(
        _module(config),
        config,
        PackedOrnithTask(recurrent_state_fp16=recurrent_state_fp16),
        execution_provider="cuda",
    )
    graph = package["decoder"].graph
    inputs = {value.name: value for value in graph.inputs}
    outputs = {value.name: value for value in graph.outputs}
    expected_state_type = ir.DataType.FLOAT16 if recurrent_state_fp16 else ir.DataType.FLOAT
    assert inputs["past.0.recurrent"].dtype == expected_state_type
    assert outputs["present.0.recurrent"].dtype == expected_state_type
    assert outputs["state_update.0.recurrent_capsule"].dtype == ir.DataType.FLOAT
    assert inputs["input_ids"].shape.rank() == 1
    assert inputs["position_ids"].shape[0] == 3
    assert outputs["hidden_states"].shape.rank() == 2
    assert inputs["past_key_values.1.key"].dtype == ir.DataType.FLOAT8E4M3FN
    assert list(inputs["past.0.recurrent"].shape)[1:] == [2, 128, 128]
    assert outputs["state_update.0.conv_value"].shape[1:] == ir.Shape([8, 512])
    gdn = next(node for node in graph if node.op_type == "GatedDeltaNet")
    assert gdn.inputs[4].dtype == gdn.inputs[5].dtype == ir.DataType.FLOAT
    assert gdn.attributes["state_update_capacity"].as_int() == 8
    assert not any(
        node.op_type in {"LinearAttention", "GroupQueryAttention"} for node in graph
    )
    mtp_config = Qwen35MtpConfig(
        **dataclasses.asdict(_config(num_hidden_layers=1, layer_types=["full_attention"])),
        use_dedicated_embeddings=True,
        use_dedicated_lm_head=True,
    )
    head = build_from_module(
        PackedMtp(Qwen35MtpModel(mtp_config)),
        mtp_config,
        PackedOrnithTask(mtp=True),
        execution_provider="cuda",
    )["model"].graph
    assert head.outputs[1].name == "hidden_states_out"
    assert head.outputs[1].shape.rank() == 2
    assert not any("state_update" in value.name for value in head.inputs)
    assert sum(node.op_type == "PagedAttention" for node in head) == 1


def test_component_reuse_and_unfused_checkpoint_fallback(tmp_path):
    graph, builder = _make_graph()
    packed = builder.initializer(ir.tensor(np.arange(8, dtype=np.uint8)), name="fc.weight")
    builder.add_output(packed, "fc.weight")
    previous = tmp_path / "previous.onnx"
    ir.save(_make_model(graph), previous, external_data="weights.data")
    graph, builder = _make_graph()
    packed = ir.Value(
        name="fc.weight", shape=ir.Shape([8]), type=ir.TensorType(ir.DataType.UINT8)
    )
    norm = ir.Value(
        name="norm.weight", shape=ir.Shape([4]), type=ir.TensorType(ir.DataType.FLOAT16)
    )
    graph.initializers[packed.name] = packed
    graph.initializers[norm.name] = norm
    builder.add_output(packed, "fc.weight")
    builder.add_output(norm, "norm.weight")
    save_file(
        {"mtp.norm.weight": np.arange(4, dtype=np.float32)}, tmp_path / "source.safetensors"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.norm.weight": "source.safetensors"}})
    )
    model = _make_model(graph)
    bind_component_weights(model, previous, tmp_path, mtp=True)
    np.testing.assert_array_equal(packed.const_value.numpy(), np.arange(8, dtype=np.uint8))
    np.testing.assert_array_equal(norm.const_value.numpy(), np.arange(4, dtype=np.float16))


def test_native_engine_configuration():
    config = _config()
    result = engine_config(
        {
            "model": {
                "type": "qwen3_5_moe",
                "decoder": {"session_options": {}},
                "vision": {},
                "embedding": {},
            },
            "search": {},
        },
        config,
    )
    assert "vision" not in result["model"] and "embedding" not in result["model"]
    assert result["model"]["type"] == "qwen3_5_moe_text"
    assert result["model"]["mtp"]["main_hidden_states"] == "hidden_states"
    assert result["model"]["decoder"]["state_groups"] == [
        {"kind": "paged_kv", "layer_ids": [1]},
        {"kind": "fixed_conv", "layer_ids": [0], "state_update": {"capacity": 8}},
        {
            "kind": "fixed_recurrent",
            "layer_ids": [0],
            "state_update": {"capacity": 8, "key_head_count": 1},
        },
    ]
    assert not result["engine"]["dynamic_batching"]["prefix_caching"]
    assert (
        result["speculative"]["max_draft_tokens"] + 1
        <= result["model"]["decoder"]["state_update_capacity"]
    )


@pytest.mark.parametrize("context", [16, 2048, 8192, 16384, 32768])
def test_benchmark_prompts_have_exact_token_counts(context):
    from ornith_engine_benchmark import make_prompt

    unit = np.arange(15, dtype=np.int32)
    prompt = make_prompt(unit, context)
    assert prompt.shape == (context,)
    np.testing.assert_array_equal(prompt, np.resize(unit, context))


@pytest.mark.parametrize("mismatch", [None, 0, 1, 63])
def test_benchmark_parity_reports_first_mismatch(monkeypatch, mismatch):
    import ornith_engine_benchmark as benchmark

    tokens = list(range(64))
    actual = tokens.copy()
    if mismatch is not None:
        actual[mismatch] += 1
    calls = []
    engine, reference = object(), object()

    def turn(current, prompt, output_tokens, *, require_mtp=True):
        calls.append((current, len(prompt), output_tokens, require_mtp))
        return {
            "output_ids": actual if require_mtp else tokens,
            "speculative": {"draft_tokens_proposed": int(require_mtp)},
        }

    monkeypatch.setattr(benchmark, "run_turn", turn)
    result = benchmark.greedy_parity(engine, reference, np.arange(128))
    assert result["first_mismatch_index"] == mismatch
    assert result["status"] == ("passed" if mismatch is None else "failed")
    assert calls == [(engine, 128, 64, True), (reference, 128, 64, False)]


@pytest.mark.parametrize("mismatch", [False, True])
def test_benchmark_sequential_models_preserve_parity_gate(tmp_path, monkeypatch, mismatch):
    import ornith_engine_benchmark as benchmark

    models = weakref.WeakSet()
    calls = []

    class Config:
        def __init__(self, path):
            self.reference = False

        def overlay(self, payload):
            self.reference = (
                json.loads(payload).get("model", {}).get("mtp", {}).get("enabled") is False
            )

    class Model:
        def __init__(self, config):
            assert not models, "Reference and speculative models must not coexist"
            models.add(self)
            self.reference = config.reference

    class Engine:
        def __init__(self, model):
            self.model = model

    class Tokenizer:
        def __init__(self, model):
            self.model = model

        def encode(self, text):
            return [1, 2, 3]

    def turn(engine, prompt, output_tokens, *, require_mtp=True):
        assert engine.model.reference == (not require_mtp)
        calls.append(require_mtp)
        tokens = list(range(output_tokens))
        if mismatch and require_mtp:
            tokens[0] = -1
        return {
            "output_ids": tokens,
            "speculative": {"draft_tokens_proposed": int(require_mtp)},
            "ttft_seconds": 0.1,
            "total_seconds": 1.0,
            "decode_tokens_per_second": 10.0,
        }

    (tmp_path / "genai_config.json").write_text(
        json.dumps(
            {"model": {"context_length": 1024, "decoder": {"filename": "decoder.onnx"}}}
        )
    )
    (tmp_path / "decoder.onnx").touch()
    monkeypatch.setattr(benchmark.og, "Config", Config)
    monkeypatch.setattr(benchmark.og, "Model", Model)
    monkeypatch.setattr(benchmark.og, "Engine", Engine)
    monkeypatch.setattr(benchmark.og, "Tokenizer", Tokenizer)
    monkeypatch.setattr(benchmark, "run_turn", turn)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_name", lambda _: "test GPU")
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1024),
    )
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            str(tmp_path),
            "--report",
            str(report_path),
            "--contexts",
            "16",
            "--output-tokens",
            "8",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--report-only",
        ],
    )
    if mismatch:
        with pytest.raises(RuntimeError, match="differ at output index 0"):
            benchmark.main()
    else:
        benchmark.main()
    report = json.loads(report_path.read_text())
    assert calls == ([False, False, True] if mismatch else [False, False, True, True, True])
    assert report["status"] == ("failed_parity" if mismatch else "passed")
    assert report["parity_reference_lifecycle"] == "sequential"
    if not mismatch:
        assert report["timed_outputs_match_target_only"]
    assert not (tmp_path / "ornith_engine_export.json").exists()


@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("draft_tokens", [None, 1, 2, 4])
def test_memory_report_checks_outputs_and_records_allocator_options(
    tmp_path, monkeypatch, mismatch, draft_tokens
):
    import ornith_engine_memory as memory

    decoder = tmp_path / "decoder.onnx"
    decoder.touch()
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "parity": {"status": "passed"},
                "decoder_sha256": hashlib.sha256(b"").hexdigest(),
                "prompt_unit_token_ids": [1, 2],
                "config_overlay": (
                    {"speculative": {"max_draft_tokens": draft_tokens}}
                    if draft_tokens is not None
                    else {}
                ),
                "contexts": [{"context_tokens": 16, "runs": [{"output_ids": [3, 4, 5, 6]}]}],
            }
        )
    )
    overlays = []

    class Config:
        def __init__(self, path):
            pass

        def overlay(self, payload):
            overlays.append(json.loads(payload))

    monkeypatch.setattr(memory.og, "Config", Config)
    monkeypatch.setattr(memory.og, "Model", lambda config: object())
    monkeypatch.setattr(memory.og, "Engine", lambda model: object())
    monkeypatch.setattr(memory.torch.cuda, "init", lambda: None)
    monkeypatch.setattr(memory.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(memory.torch.cuda, "mem_get_info", lambda: (100, 1024))
    monkeypatch.setattr(memory.torch.cuda, "get_device_name", lambda _: "test GPU")
    monkeypatch.setattr(
        memory,
        "run_turn",
        lambda engine, prompt, output_tokens: {"output_ids": [3, 4, 5, 7 if mismatch else 6]},
    )
    report_path = tmp_path / "memory.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "memory",
            "--model",
            str(tmp_path),
            "--reference-report",
            str(reference_path),
            "--report",
            str(report_path),
            "--decoder-file",
            "decoder.onnx",
            "--context",
            "16",
            "--device-allocator-for-initializers",
            "--arena-extend-strategy",
            "kSameAsRequested",
        ],
    )
    if mismatch:
        with pytest.raises(RuntimeError, match="Output differs from validated reference"):
            memory.main()
    else:
        memory.main()
    report = json.loads(report_path.read_text())
    assert report["status"] == ("failed" if mismatch else "passed")
    assert report["sampled_peak_device_memory_bytes"] == 924
    assert len(report["runs"]) == (0 if mismatch else 4)
    options = overlays[0]["model"]["decoder"]["session_options"]
    assert options["session.use_device_allocator_for_initializers"] == "1"
    assert options["provider_options"][0]["cuda"]["arena_extend_strategy"] == "1"
    if draft_tokens is not None:
        assert overlays[0]["speculative"] == {"max_draft_tokens": draft_tokens}
        assert report["config_overlay"]["speculative"] == {"max_draft_tokens": draft_tokens}
    else:
        assert "speculative" not in overlays[0]


@pytest.mark.parametrize(
    "status", ["in_progress", "failed_parity", "failed_repeatability", "passed"]
)
def test_benchmark_manifest_requires_parity_and_completed_timings(tmp_path, status):
    from ornith_engine_benchmark import save_report

    manifest_path = tmp_path / "ornith_engine_export.json"
    manifest_path.write_text(
        json.dumps({"engine_validated": True, "correctness_blocker": "previous failure"})
    )
    report = {
        "status": status,
        "parity": {
            "status": "failed" if status == "failed_parity" else "passed",
            "first_mismatch_index": 1 if status == "failed_parity" else None,
        },
    }
    report_path = tmp_path / "benchmark.json"
    save_report(tmp_path, report_path, report)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["engine_validated"] == (status == "passed")
    assert manifest["accuracy_test"] is False
    assert json.loads(report_path.read_text()) == report
    if status == "passed":
        assert "correctness_blocker" not in manifest
    elif status == "failed_parity":
        assert "output index 1" in manifest["correctness_blocker"]


def test_experimental_benchmark_preserves_export_validation(tmp_path):
    from ornith_engine_benchmark import save_report

    manifest_path = tmp_path / "ornith_engine_export.json"
    original = '{"engine_validated": true, "engine_validation_report": "baseline.json"}\n'
    manifest_path.write_text(original)
    report = {"status": "in_progress"}
    report_path = tmp_path / "experimental.json"
    save_report(tmp_path, report_path, report, update_manifest=False)
    assert manifest_path.read_text() == original
    assert json.loads(report_path.read_text()) == report


@pytest.mark.parametrize(
    "arguments",
    [
        ["--decoder-file", "/outside.onnx"],
        ["--decoder-file", "../outside.onnx"],
        ["--decoder-file", "missing.onnx"],
        ["--draft-tokens", "0"],
        ["--draft-tokens", "8"],
    ],
)
def test_benchmark_rejects_invalid_tuning_overrides(tmp_path, monkeypatch, arguments):
    from ornith_engine_benchmark import main

    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {"decoder": {"state_update_capacity": 8}}})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            str(tmp_path),
            "--report",
            str(tmp_path / "report.json"),
            *arguments,
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        main()
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("rows", [1, 2, 8])
@pytest.mark.parametrize("extra_attribute", [False, True])
@pytest.mark.skipif(
    "CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required"
)
def test_packed_int2_projections_preserve_output_columns(tmp_path, rows, extra_attribute):
    rng = np.random.default_rng(57)
    graph, builder = _make_graph()
    inputs = ir.Value(
        name="input", shape=ir.Shape([rows, 512]), type=ir.TensorType(ir.DataType.FLOAT16)
    )
    graph.inputs.append(inputs)
    graph.opset_imports["com.microsoft"] = 1
    for index, width in enumerate((64, 128, 512)):
        weight = builder.initializer(
            ir.tensor(rng.integers(0, 256, (width, 4, 32), dtype=np.uint8)),
            name=f"weight{index}",
        )
        scales = builder.initializer(
            ir.tensor(rng.uniform(0.001, 0.01, (width, 4)).astype(np.float16)),
            name=f"scale{index}",
        )
        value = ir.Value(
            name=f"output{index}",
            shape=ir.Shape([rows, width]),
            type=ir.TensorType(ir.DataType.FLOAT16),
        )
        graph.append(
            ir.Node(
                "com.microsoft",
                "MatMulNBits",
                [inputs, weight, scales],
                [
                    ir.AttrInt64(name, number)
                    for name, number in (
                        ("K", 512),
                        ("N", width),
                        ("bits", 2),
                        ("block_size", 128),
                    )
                ],
                outputs=[value],
            )
        )
        graph.outputs.append(value)
    model = _make_model(graph)
    if extra_attribute:
        for node in graph:
            node.attributes["accuracy_level"] = ir.AttrInt64("accuracy_level", 1)
    before = tmp_path / "before.onnx"
    ir.save(model, before)
    feed = {"input": rng.standard_normal((rows, 512)).astype(np.float16)}
    expected = ort.InferenceSession(str(before), providers=["CUDAExecutionProvider"]).run(
        None, feed
    )
    assert pack_int2_projections(model) == (0 if extra_attribute else 2)
    assert pack_int2_projections(model) == 0
    assert sum(node.op_type == "MatMulNBits" for node in graph) == (
        3 if extra_attribute else 1
    )
    assert not any(node.op_type == "Concat" for node in graph)
    after = tmp_path / "after.onnx"
    ir.save(model, after)
    actual = ort.InferenceSession(str(after), providers=["CUDAExecutionProvider"]).run(
        None, feed
    )
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)


def _packed_swiglu_model():
    rng = np.random.default_rng(47)
    graph, builder = _make_graph()
    graph.opset_imports[""] = 24
    graph.opset_imports["com.microsoft"] = 1

    def value(name, width):
        return ir.Value(
            name=name,
            shape=ir.Shape(["rows", width]),
            type=ir.TensorType(ir.DataType.FLOAT16),
        )

    x, projected = value("x", 128), value("projected", 32)
    gate, up, swish, result = (value(name, 16) for name in ("gate", "up", "swish", "y"))
    graph.inputs.append(x)
    weight = builder.initializer(
        ir.tensor(rng.integers(0, 256, (32, 1, 32), dtype=np.uint8)), name="weight"
    )
    scales = builder.initializer(
        ir.tensor(rng.uniform(0.001, 0.01, (32, 1)).astype(np.float16)), name="scales"
    )
    sizes = builder.initializer(ir.tensor([16, 16], dtype=ir.DataType.INT64), name="sizes")
    graph.extend(
        [
            ir.Node(
                "com.microsoft",
                "MatMulNBits",
                [x, weight, scales],
                [
                    ir.AttrInt64(key, number)
                    for key, number in (
                        ("K", 128),
                        ("N", 32),
                        ("bits", 2),
                        ("block_size", 128),
                    )
                ],
                outputs=[projected],
            ),
            ir.Node(
                "", "Split", [projected, sizes], [ir.AttrInt64("axis", -1)], outputs=[gate, up]
            ),
            ir.Node("", "Swish", [gate], outputs=[swish]),
            ir.Node("", "Mul", [swish, up], outputs=[result]),
        ]
    )
    graph.outputs.append(result)
    return _make_model(graph)


@pytest.mark.parametrize(
    "reject",
    [
        None,
        "quick_gelu",
        "quick_gelu_missing_alpha",
        "alpha",
        "bits",
        "block_size",
        "attribute",
        "axis",
        "sizes",
        "dynamic_sizes",
        "extra_output",
        "reused_gate",
        "reused_up",
        "reused_swish",
        "dtype",
        "shape",
        "swapped_gate",
        "dynamic_weight",
    ],
)
def test_int2_swiglu_fusion_guards(reject):
    model = _packed_swiglu_model()
    graph = model.graph
    projection, split, swish, mul = list(graph)
    if reject in ("quick_gelu", "quick_gelu_missing_alpha"):
        swish.domain = "com.microsoft"
        swish.op_type = "QuickGelu"
        if reject == "quick_gelu":
            swish.attributes["alpha"] = ir.AttrFloat32("alpha", 1.0)
    elif reject == "alpha":
        swish.attributes["alpha"] = ir.AttrFloat32("alpha", 0.5)
    elif reject in ("bits", "block_size", "attribute"):
        name, number = {
            "bits": ("bits", 4),
            "block_size": ("block_size", 64),
            "attribute": ("accuracy_level", 1),
        }[reject]
        projection.attributes[name] = ir.AttrInt64(name, number)
    elif reject == "axis":
        split.attributes["axis"] = ir.AttrInt64("axis", 0)
    elif reject == "sizes":
        split.inputs[1].const_value = ir.tensor([8, 24], dtype=ir.DataType.INT64)
    elif reject == "dynamic_sizes":
        sizes = split.inputs[1]
        del graph.initializers[sizes.name]
        sizes.const_value = None
        graph.inputs.append(sizes)
    elif reject == "extra_output":
        graph.outputs.append(projection.outputs[0])
    elif reject in ("reused_gate", "reused_up", "reused_swish"):
        source = {
            "reused_gate": split.outputs[0],
            "reused_up": split.outputs[1],
            "reused_swish": swish.outputs[0],
        }[reject]
        graph.outputs.append(source)
    elif reject == "dtype":
        projection.inputs[2].type = ir.TensorType(ir.DataType.FLOAT)
    elif reject == "shape":
        mul.outputs[0].shape = ir.Shape(["rows", 32])
    elif reject == "swapped_gate":
        swish.replace_input_with(0, split.outputs[1])
        mul.replace_input_with(1, split.outputs[0])
    elif reject == "dynamic_weight":
        weight = projection.inputs[1]
        del graph.initializers[weight.name]
        graph.inputs.append(weight)
    accepted = reject in (None, "quick_gelu")
    assert fuse_int2_swiglu(model) == int(accepted)
    assert fuse_int2_swiglu(model) == 0
    if accepted:
        assert [node.op_type for node in graph] == ["MatMulNBitsSwiGLU"]
        assert graph.outputs[0].name == "y"
        assert graph.outputs[0].shape == ir.Shape(["rows", 16])
        assert "sizes" not in graph.initializers


def test_int2_swiglu_fusion_after_projection_packing():
    model = _packed_swiglu_model()
    graph = model.graph
    projection, split, _, _ = list(graph)
    nodes, outputs = [], []
    for index in range(2):
        parameters = []
        for value in projection.inputs[1:]:
            parameter = ir.Value(
                name=f"{value.name}_{index}",
                shape=ir.Shape([16, *value.shape[1:]]),
                type=value.type,
                const_value=ir.tensor(
                    value.const_value.numpy()[index * 16 : (index + 1) * 16]
                ),
            )
            graph.register_initializer(parameter)
            parameters.append(parameter)
        output = ir.Value(
            name=split.outputs[index].name,
            shape=split.outputs[index].shape,
            type=split.outputs[index].type,
        )
        attributes = dict(projection.attributes)
        attributes["N"] = ir.AttrInt64("N", 16)
        nodes.append(
            ir.Node(
                "com.microsoft",
                "MatMulNBits",
                [projection.inputs[0], *parameters],
                attributes,
                outputs=[output],
            )
        )
        outputs.append(output)
    ir.convenience.replace_nodes_and_values(
        graph, split, [projection, split], nodes, list(split.outputs), outputs
    )
    assert pack_int2_projections(model) == 1
    assert fuse_int2_swiglu(model) == 1
    assert [node.op_type for node in graph] == ["MatMulNBitsSwiGLU"]


def _cuda(array):
    return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(array), "cuda", 0)


@pytest.mark.parametrize(
    "reject", [None, "exposed_gate", "broadcast", "float32", "non_scalar", "extra_user"]
)
def test_shared_expert_gate_fusion_is_guarded_and_idempotent(reject):
    from onnx import TensorProto, helper, numpy_helper

    dtype = TensorProto.FLOAT if reject == "float32" else TensorProto.FLOAT16
    np_dtype = np.float32 if reject == "float32" else np.float16
    n, k, rows = 64, 128, 2
    gate_width = 2 if reject == "non_scalar" else 1
    shared_shape = [1, n] if reject == "broadcast" else [rows, n]
    nodes = [
        helper.make_node("Transpose", ["weight"], ["wt"], perm=[1, 0]),
        helper.make_node("MatMul", ["x", "wt"], ["dot"]),
        helper.make_node("Sigmoid", ["dot"], ["gate"]),
        helper.make_node("Mul", ["gate", "shared"], ["gated"]),
        helper.make_node("Add", ["gated", "routed"], ["combined"]),
        helper.make_node("Add", ["combined", "residual"], ["output"]),
    ]
    outputs = [helper.make_tensor_value_info("output", dtype, [rows, n])]
    if reject == "exposed_gate":
        outputs.append(helper.make_tensor_value_info("gate", dtype, [rows, gate_width]))
    if reject == "extra_user":
        nodes.append(helper.make_node("Identity", ["dot"], ["debug"]))
        outputs.append(helper.make_tensor_value_info("debug", dtype, [rows, gate_width]))
    proto = helper.make_model(
        helper.make_graph(
            nodes,
            "fusion",
            [
                helper.make_tensor_value_info("x", dtype, [rows, k]),
                helper.make_tensor_value_info("shared", dtype, shared_shape),
                helper.make_tensor_value_info("routed", dtype, [rows, n]),
                helper.make_tensor_value_info("residual", dtype, [rows, n]),
            ],
            outputs,
            [numpy_helper.from_array(np.zeros((gate_width, k), np_dtype), "weight")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model = ir.serde.deserialize_model(onnx.shape_inference.infer_shapes(proto))
    assert fuse_shared_expert_gates(model) == int(reject is None)
    assert fuse_shared_expert_gates(model) == 0
    assert sum(node.op_type == "MatMul" for node in model.graph) == int(reject is not None)
    if reject is None:
        assert len(model.graph) == 1
        node = next(iter(model.graph))
        assert node.op_type == "MatMulSigmoidGatedResidual"
        assert [value.name for value in node.inputs] == [
            "x",
            "weight",
            "shared",
            "routed",
            "residual",
        ]
        assert model.graph.outputs[0].name == "output"


@pytest.mark.parametrize(
    "reject",
    [
        None,
        "direct_router",
        "exposed_logits",
        "extra_user",
        "different_input",
        "float32",
        "int4",
        "bias",
        "nonidentity_reshape",
        "dynamic_reshape",
        "scalar_reshape",
        "wrong_experts",
    ],
)
@pytest.mark.parametrize("use_matmul", [False, True])
def test_moe_router_fusion_is_guarded_and_idempotent(reject, use_matmul):
    from onnx import TensorProto, helper, numpy_helper
    from ornith_engine_export import fuse_moe_routers

    dtype = TensorProto.FLOAT if reject == "float32" else TensorProto.FLOAT16
    inputs = [helper.make_tensor_value_info("x", dtype, [2, 128])]
    outputs = [helper.make_tensor_value_info("out", dtype, [2, 128])]
    weights = [
        numpy_helper.from_array(np.zeros((16, 128), np.float16), "router"),
        numpy_helper.from_array(np.array([-1, 16], np.int64), "shape"),
        numpy_helper.from_array(
            np.zeros((8 if reject == "wrong_experts" else 16, 256, 32), np.uint8), "fc1"
        ),
        numpy_helper.from_array(np.ones((16, 256, 1), np.float16), "s1"),
        numpy_helper.from_array(np.zeros((16, 128, 32), np.uint8), "fc2"),
        numpy_helper.from_array(np.ones((16, 128, 1), np.float16), "s2"),
    ]
    if reject == "nonidentity_reshape":
        weights[1].CopyFrom(numpy_helper.from_array(np.array([16, 2], np.int64), "shape"))
    elif reject == "scalar_reshape":
        weights[1].CopyFrom(numpy_helper.from_array(np.array(-1, np.int64), "shape"))
    elif reject == "dynamic_reshape":
        weights.pop(1)
        inputs.append(helper.make_tensor_value_info("shape", TensorProto.INT64, [2]))
    if reject == "different_input":
        inputs.append(helper.make_tensor_value_info("other", dtype, [2, 128]))
    if reject == "bias":
        weights.append(numpy_helper.from_array(np.zeros((16, 256), np.float16), "bias"))
    nodes = [
        helper.make_node("Transpose", ["router"], ["wt"], perm=[1, 0]),
        helper.make_node(
            "MatMul", ["other" if reject == "different_input" else "x", "wt"], ["dot"]
        ),
        helper.make_node("Reshape", ["dot", "shape"], ["scores"]),
        helper.make_node(
            "QMoE",
            ["x", "scores", "fc1", "s1", "bias" if reject == "bias" else "", "fc2", "s2"],
            ["out"],
            domain="com.microsoft",
            activation_type="swiglu",
            normalize_routing_weights=1,
            k=8,
            expert_weight_bits=4 if reject == "int4" else 2,
            block_size=128,
            swiglu_fusion=1,
            quant_type="int",
            weights_prepacked=0,
        ),
    ]
    if reject == "exposed_logits":
        outputs.append(helper.make_tensor_value_info("scores", dtype, [2, 16]))
    if reject == "extra_user":
        nodes.append(helper.make_node("Identity", ["dot"], ["debug"]))
        outputs.append(helper.make_tensor_value_info("debug", dtype, [2, 16]))
    if reject == "direct_router":
        nodes.pop(2)
        nodes[2].input[1] = "dot"
    proto = helper.make_model(
        helper.make_graph(
            nodes,
            "routers",
            inputs,
            outputs,
            weights,
            value_info=[
                helper.make_tensor_value_info("dot", dtype, [2, 16]),
                helper.make_tensor_value_info("scores", dtype, [2, 16]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model = ir.serde.deserialize_model(proto)
    accepted = reject in (None, "direct_router")
    assert fuse_moe_routers(model, use_matmul=use_matmul) == int(accepted)
    assert fuse_moe_routers(model, use_matmul=use_matmul) == 0
    assert sum(node.op_type == "MatMul" for node in model.graph) == int(not accepted)
    if accepted:
        assert len(model.graph) == 1
        node = next(iter(model.graph))
        assert node.op_type == "QMoEWithRouter"
        assert ("router_use_matmul" in node.attributes) == use_matmul
        if use_matmul:
            assert node.attributes["router_use_matmul"].as_int() == 1
        assert [value.name if value is not None else "" for value in node.inputs] == [
            "x",
            "router",
            "fc1",
            "s1",
            "",
            "fc2",
            "s2",
        ]
        assert model.graph.outputs[0].name == "out"


@pytest.mark.parametrize(
    "flag,required",
    [
        ("--moe-router-use-matmul", "--fuse-moe-routers"),
        ("--fuse-shared-expert-swiglu", "--pack-projections"),
    ],
)
def test_export_cli_requires_prerequisite_fusion(
    monkeypatch, tmp_path, capsys, flag, required
):
    from ornith_engine_export import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ornith_engine_export",
            "--checkpoint",
            str(tmp_path),
            "--components",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            flag,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert f"{flag} requires {required}" in capsys.readouterr().err


def _states(batch, block_count=4, recurrent_dtype=np.float32):
    states = {
        "past.0.conv": _cuda(np.zeros((batch, 512, 3), np.float16)),
        "past.0.recurrent": _cuda(np.zeros((batch, 2, 128, 128), recurrent_dtype)),
    }
    for side in ("key", "value"):
        # FP8 zero has the same bit pattern as UINT8 zero.
        cache = ort.OrtValue.ortvalue_from_shape_and_type(
            [block_count, 256, 1, 128], onnx.TensorProto.FLOAT8E4M3FN, "cuda", 0
        )
        cache.update_inplace(np.zeros((block_count, 256, 1, 128), np.uint8))
        states[f"past_key_values.1.{side}"] = cache
    return states


def _run(session, sequences, states, past=None, capture=0):
    lengths = [len(sequence) for sequence in sequences]
    batch = len(sequences)
    past = [0] * batch if past is None else past
    positions = np.concatenate(
        [np.arange(p, p + n) for p, n in zip(past, lengths, strict=True)]
    )
    feed = {
        "input_ids": np.concatenate(sequences).astype(np.int64),
        "position_ids": np.tile(positions, (3, 1)).astype(np.int64),
        "cumulative_sequence_lengths": np.array([0, *np.cumsum(lengths)], np.int32),
        "past_sequence_lengths": np.array(past, np.int32),
        "block_table": np.arange(batch, dtype=np.int32).reshape(batch, 1),
        "attention_metadata": np.array(
            [max(lengths), max(p + n for p, n in zip(past, lengths, strict=True)), 0], np.int32
        ),
        "logits_indices": np.arange(sum(lengths), dtype=np.int32),
        "state_update_capture_count": np.full(batch, capture, np.int32),
        "state_update_active": np.array([int(capture > 0)], np.int32),
    }
    binding = session.io_binding()
    for name, value in feed.items():
        binding.bind_cpu_input(name, value)
    for name, value in states.items():
        binding.bind_ortvalue_input(name, value)
    for output in session.get_outputs():
        if output.name.startswith("present.") and output.name.endswith((".key", ".value")):
            binding.bind_ortvalue_output(
                output.name, states[output.name.replace("present.", "past_key_values.", 1)]
            )
        else:
            binding.bind_output(output.name, "cuda", 0)
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    return dict(
        zip(
            [value.name for value in session.get_outputs()], binding.get_outputs(), strict=True
        )
    )


@pytest.mark.skipif(
    "CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required"
)
@pytest.mark.parametrize("recurrent_state_fp16", [False, True])
def test_cuda_packed_hybrid_sequence_isolation_and_append(tmp_path, recurrent_state_fp16):
    config = _config()
    module = _module(config)
    _initialize(module)
    model = build_from_module(
        module,
        config,
        PackedOrnithTask(recurrent_state_fp16=recurrent_state_fp16),
        execution_provider="cuda",
    )["decoder"]

    def states(batch):
        return _states(
            batch, recurrent_dtype=np.float16 if recurrent_state_fp16 else np.float32
        )

    path = tmp_path / "packed.onnx"
    ir.save(model, path)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "profile")
    session = ort.InferenceSession(str(path), options, providers=["CUDAExecutionProvider"])
    combined = _run(session, [[3, 5, 7], [11, 13]], states(2))
    first = _run(session, [[3, 5, 7]], states(1))
    second = _run(session, [[11, 13]], states(1))
    for output in ("logits", "hidden_states"):
        actual = combined[output].numpy()
        expected = np.concatenate([first[output].numpy(), second[output].numpy()])
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=1e-2)
    for output in ("present.0.conv", "present.0.recurrent"):
        np.testing.assert_allclose(
            combined[output].numpy(),
            np.concatenate([first[output].numpy(), second[output].numpy()]),
            atol=2e-5,
            rtol=1e-2,
        )
    appended_states = {
        "past.0.conv": combined["present.0.conv"],
        "past.0.recurrent": combined["present.0.recurrent"],
        "past_key_values.1.key": combined["present.1.key"],
        "past_key_values.1.value": combined["present.1.value"],
    }
    appended = _run(session, [[17], [19]], appended_states, past=[3, 2], capture=1)
    full_first = _run(session, [[3, 5, 7, 17]], states(1))
    full_second = _run(session, [[11, 13, 19]], states(1))
    for output in ("logits", "hidden_states"):
        expected = np.concatenate(
            [full_first[output].numpy()[-1:], full_second[output].numpy()[-1:]]
        )
        np.testing.assert_allclose(appended[output].numpy(), expected, atol=2e-3, rtol=2e-2)
    assert appended["state_update.0.conv_value"].shape() == [2, 8, 512]
    assert appended["state_update.0.recurrent_capsule"].shape() == [2, 8 * (2 + 128 + 256)]
    assert np.isfinite(appended["state_update.0.recurrent_capsule"].numpy()).all()
    events = json.loads(Path(session.end_profiling()).read_text())
    required = {"VarlenCausalConvWithState", "GatedDeltaNet", "PagedAttention"}
    executed = {
        event["args"]["op_name"]
        for event in events
        if event.get("args", {}).get("provider") == "CUDAExecutionProvider"
    }
    assert required <= executed


@pytest.mark.skipif(
    "CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required"
)
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_native_engine_mtp_rollback_matches_target_only_with_minimum_length(
    tmp_path, cuda_graph
):
    og = pytest.importorskip("onnxruntime_genai")
    from ornith_engine_benchmark import greedy_parity, run_turn

    config = _config()
    target = _module(config)
    mtp_config = Qwen35MtpConfig(
        **dataclasses.asdict(_config(num_hidden_layers=1, layer_types=["full_attention"])),
        use_dedicated_embeddings=True,
        use_dedicated_lm_head=True,
    )
    head = PackedMtp(Qwen35MtpModel(mtp_config))
    for module in (target, head):
        _initialize(module)
    target_graph = build_from_module(
        target, config, PackedOrnithTask(), execution_provider="cuda"
    )["decoder"]
    head_graph = build_from_module(
        head, mtp_config, PackedOrnithTask(mtp=True), execution_provider="cuda"
    )["model"]
    (tmp_path / "decoder").mkdir()
    ir.save(target_graph, tmp_path / "decoder/model.onnx")
    ir.save(head_graph, tmp_path / "mtp.onnx")
    runtime_config = engine_config(
        {
            "model": {
                "type": "qwen3_5_moe",
                "vocab_size": 256,
                "context_length": 128,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "decoder": {
                    "session_options": {},
                    "hidden_size": 128,
                    "head_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                },
            },
            "search": {"max_length": 128, "past_present_share_buffer": True},
        },
        config,
    )
    runtime_config["engine"]["dynamic_batching"]["num_blocks"] = 8
    (tmp_path / "genai_config.json").write_text(json.dumps(runtime_config))
    config_options = og.Config(str(tmp_path))
    if cuda_graph:
        config_options.overlay(
            '{"model":{"decoder":{"session_options":'
            '{"provider_options":[{"cuda":{"enable_cuda_graph":"1"}}]}}}}'
        )
    model = og.Model(config_options)
    engine = og.Engine(model)
    result = run_turn(engine, [3, 5, 7, 11], 32)
    assert result["speculative"]["zero_accept_rounds"] > 0

    reference_config = og.Config(str(tmp_path))
    if cuda_graph:
        reference_config.overlay(
            '{"model":{"decoder":{"session_options":'
            '{"provider_options":[{"cuda":{"enable_cuda_graph":"1"}}]}}}}'
        )
    reference_config.overlay('{"model":{"mtp":{"enabled":false}}}')
    reference_model = og.Model(reference_config)
    reference = og.Engine(reference_model)
    parity = greedy_parity(engine, reference, [3, 5, 7, 11], 32)
    assert parity["status"] == "passed"
    assert result["output_ids"] == parity["target_only"]["output_ids"]
