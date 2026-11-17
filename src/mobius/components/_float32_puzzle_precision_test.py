# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FP32-gamma and complete-SiLU stores, with opt-in pinned Bolt/CUDA controls."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnx_ir import tensor_adapters
from onnxscript import nn

from mobius._builder import _cast_module_dtype
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _ort_value_to_numpy
from mobius.components import (
    Float32AccumulationMamba2Block,
    Float32RMSNorm,
    Float32SiLUMamba2Block,
    RMSNorm,
)

_DTYPES = [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@pytest.fixture(scope="module")
def precision_runtime(tmp_path_factory):
    """Optional qualified subprocess keeps protected Python installations intact."""
    script = os.environ.get("MOBIUS_PRECISION_ORT_WORKER")
    if not script:
        yield None
        return
    spec = importlib.util.spec_from_file_location("precision_ort_worker", script)
    worker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_module)
    directory = tmp_path_factory.mktemp("precision-runtime") / "worker"
    worker = worker_module.RemoteOrt(
        directory,
        gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        build=Path(os.environ["MOBIUS_PRECISION_ORT_BUILD"]),
    )
    try:
        assert (
            worker.info["worker_script_sha256"] == os.environ["MOBIUS_PRECISION_WORKER_SHA256"]
        )
        assert worker.info["pybind_sha256"] == os.environ["MOBIUS_PRECISION_PYBIND_SHA256"]
        yield worker
        mapped = {}
        for line in Path(f"/proc/{worker.info['pid']}/maps").read_text().splitlines():
            if "onnxruntime" in line and "/" in line:
                path = Path(line.split(maxsplit=5)[5])
                if path.is_file():
                    mapped[str(path)] = _sha(path)
        assert os.environ["MOBIUS_PRECISION_CUDA_SHA256"] in mapped.values()
        assert worker.info["pybind_sha256"] in mapped.values()
        core = Path(os.environ["MOBIUS_PRECISION_ORT_BUILD"]) / "libonnxruntime.so.1.32.0"
        assert _sha(core) == os.environ["MOBIUS_PRECISION_CORE_SHA256"]
        (directory.parent / "runtime-identity.json").write_text(
            json.dumps(
                {
                    "runtime": worker.info,
                    "mapped_libraries": mapped,
                    "standalone_core_artifact": {
                        "path": str(core),
                        "sha256": _sha(core),
                        "mapped": str(core) in mapped,
                    },
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        worker.close()


def _run(model, feeds, directory, runtime):
    directory.mkdir(exist_ok=True)
    path = directory / "model.onnx"
    ir.save(model, path)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
    options.optimized_model_filepath = str(directory / "optimized.onnx")
    options.enable_profiling = True
    options.profile_file_prefix = str(directory / "profile")
    factory = ort.InferenceSession if runtime is None else runtime.session
    providers = (
        ["CPUExecutionProvider"]
        if runtime is None
        else [("CUDAExecutionProvider", {"use_tf32": "0"})]
    )
    session = factory(str(path), sess_options=options, providers=providers)
    try:
        owned = {}
        for name, array in feeds.items():
            dtype = ir.Tensor(array).dtype
            value = ort.OrtValue.ortvalue_from_shape_and_type(list(array.shape), int(dtype))
            if array.size:
                raw = array.view(np.uint16) if dtype == ir.DataType.BFLOAT16 else array
                value.update_inplace(np.ascontiguousarray(raw))
            owned[name] = value
        values = session.run_with_ort_values(None, owned)
        actual = [_ort_value_to_numpy(value).copy() for value in values]
        profile = Path(session.end_profiling())
        counts = {}
        for event in json.loads(profile.read_text()):
            args = event.get("args", {})
            provider, op = args.get("provider"), args.get("op_name")
            if not provider:
                continue
            key = f"{provider}::{op}"
            counts[key] = counts.get(key, 0) + 1
            if runtime is not None and provider == "CPUExecutionProvider":
                types = [
                    kind
                    for field in ("input_type_shape", "output_type_shape")
                    for entry in args.get(field, [])
                    for kind in entry
                ]
                assert op in ("Shape", "Size") or not any(
                    "float" in kind or kind in ("double", "complex64", "complex128")
                    for kind in types
                ), event
        if runtime is not None:
            assert any(key.startswith("CUDAExecutionProvider::") for key in counts)
        (directory / "provider-audit.json").write_text(
            json.dumps(
                {
                    "counts": counts,
                    "unexpected_cpu_floating": [],
                    "cuda_executed": runtime is not None,
                    "full_decoder_parity": "NOT_RUN",
                },
                indent=2,
            )
            + "\n"
        )
        return actual
    finally:
        if runtime is not None:
            session.close()


def _norm_model(dtype, shape, weight, deferred=False):
    norm = Float32RMSNorm(shape[-1], eps=1e-5)
    _cast_module_dtype(norm, dtype)
    norm.weight = nn.Parameter([shape[-1]], data=ir.Tensor(weight))
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", shape, dtype)
    hidden = op.Identity(x) if deferred else x
    if deferred:
        hidden.type = None
    output = norm(op, hidden)
    rms = next(node for node in graph if node.op_type == "RMSNormalization")
    assert all(value.dtype == ir.DataType.FLOAT for value in rms.inputs)
    assert rms.attributes["stash_type"].as_int() == 1
    assert output.producer().op_type == "CastLike"
    output.name, output.shape, output.dtype = "output", ir.Shape(shape), dtype
    graph.outputs.append(output)
    return ir.Model(graph, ir_version=11)


def _block(dtype, kernel=4, channels=2):
    block = Float32SiLUMamba2Block(
        d_model=4,
        d_inner=channels,
        num_heads=1,
        d_head=channels,
        d_state=1,
        n_groups=1,
        conv_kernel=kernel,
    )
    _cast_module_dtype(block, dtype)
    return block


def _production_block():
    block = Float32SiLUMamba2Block(
        d_model=4096,
        d_inner=8192,
        num_heads=128,
        d_head=64,
        d_state=128,
        n_groups=8,
        conv_kernel=4,
    )
    _cast_module_dtype(block, ir.DataType.BFLOAT16)
    return block


def _conv_model(block, dtype, length, deferred=False, trace=False):
    builder, op, graph = create_test_builder()
    shape = [1, block.conv_dim, length]
    x = create_test_input(builder, "x", shape, dtype)
    state = create_test_input(
        builder, "state", [1, block.conv_dim, block.conv_kernel - 1], dtype
    )
    hidden = op.Identity(x) if deferred else x
    if deferred:
        hidden.type = None
    output, present = block.conv1d(op, hidden, state)
    assert output.producer().op_type == "CastLike"
    swish = output.producer().inputs[0]
    assert swish.producer().op_type == "Swish"
    upcast = swish.producer().inputs[0]
    assert upcast.producer().op_type == "Cast"
    biased = upcast.producer().inputs[0]
    assert biased.producer().op_type == "CastLike"
    conv = biased.producer().inputs[0]
    assert conv.producer().op_type == "Conv" and conv.dtype == ir.DataType.FLOAT
    output.name, output.shape, output.dtype = "output", ir.Shape(shape), dtype
    present.name, present.dtype = "present", dtype
    present.shape = ir.Shape([1, block.conv_dim, block.conv_kernel - 1])
    graph.outputs.extend([output, present])
    if trace:
        biased.name, biased.shape, biased.dtype = "biased_conv", ir.Shape(shape), dtype
        graph.outputs.append(biased)
    return ir.Model(graph, ir_version=11)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(2, 4), (1, 1, 4), (1, 3, 4)])
@pytest.mark.parametrize("deferred", [False, True])
def test_float32_rms_norm_native_and_deferred(
    tmp_path, precision_runtime, dtype, shape, deferred
):
    weight = np.asarray([0.55859375, 0.54296875, 1.375, -0.6875], dtype=dtype.numpy())
    x = (
        np.linspace(-1.5, 2.0, np.prod(shape), dtype=np.float32)
        .reshape(shape)
        .astype(dtype.numpy())
    )
    xf = torch.from_numpy(x.astype(np.float32))
    expected = (
        (
            xf
            * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
            * torch.from_numpy(weight.astype(np.float32))
        )
        .to(tensor_adapters.to_torch_dtype(dtype))
        .float()
        .numpy()
    )
    (actual,) = _run(
        _norm_model(dtype, shape, weight, deferred), {"x": x}, tmp_path, precision_runtime
    )
    np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("length,kernel", [(1, 4), (3, 4), (3, 1)])
@pytest.mark.parametrize("deferred", [False, True])
def test_float32_silu_preserves_both_conv_stores(
    tmp_path, precision_runtime, dtype, length, kernel, deferred
):
    block = _block(dtype, kernel)
    native = tensor_adapters.to_torch_dtype(dtype)
    weight = (
        torch.linspace(-0.7, 0.8, block.conv_dim * kernel)
        .reshape(block.conv_dim, 1, kernel)
        .to(native)
    )
    bias = torch.linspace(-0.03, 0.02, block.conv_dim).to(native)
    for parameter, tensor in ((block.conv1d.weight, weight), (block.conv1d.bias, bias)):
        parameter.const_value = tensor_adapters.TorchTensor(tensor)
    x = (
        torch.linspace(-0.8, 0.9, block.conv_dim * length)
        .reshape(1, block.conv_dim, length)
        .to(native)
    )
    state = torch.ones((1, block.conv_dim, kernel - 1), dtype=native) * 0.3
    window = torch.cat((state, x), -1)
    biased = torch.nn.functional.conv1d(
        window.float(), weight.float(), bias.float(), groups=block.conv_dim
    ).to(native)
    expected = torch.nn.functional.silu(biased.float()).to(native).float().numpy()
    feeds = {
        "x": x.float().numpy().astype(dtype.numpy()),
        "state": state.float().numpy().astype(dtype.numpy()),
    }
    traced, traced_present, actual_biased = _run(
        _conv_model(block, dtype, length, deferred, trace=True),
        feeds,
        tmp_path / "instrumented",
        precision_runtime,
    )
    uninstrumented = ir.load(tmp_path / "instrumented/model.onnx")
    uninstrumented.graph.outputs.pop()
    actual, present = _run(uninstrumented, feeds, tmp_path, precision_runtime)
    np.testing.assert_array_equal(actual, traced)
    np.testing.assert_array_equal(present, traced_present)
    np.testing.assert_allclose(
        actual_biased.astype(np.float32), biased.float().numpy(), rtol=2e-6, atol=1e-7
    )
    np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=2e-6, atol=1e-7)
    expected_state = window[..., -kernel + 1 :] if kernel > 1 else window[..., :0]
    np.testing.assert_array_equal(present.astype(np.float32), expected_state.float().numpy())
    if dtype != ir.DataType.FLOAT:
        optimized = ir.load(tmp_path / "optimized.onnx")
        conv_stores = [
            node
            for node in optimized.graph
            if node.op_type == "Cast"
            and node.attributes["to"].as_int() == int(dtype)
            and node.inputs[0].producer() is not None
            and node.inputs[0].producer().op_type in ("Conv", "FusedConv")
        ]
        assert len(conv_stores) == 1
        assert any(
            node.op_type == "Cast"
            and node.attributes["to"].as_int() == int(ir.DataType.FLOAT)
            and node.inputs[0] is conv_stores[0].outputs[0]
            for node in optimized.graph
        )
        assert optimized.graph.outputs[0].producer().op_type == "Cast"
        assert optimized.graph.outputs[0].producer().attributes["to"].as_int() == int(dtype)


def test_variants_preserve_names_and_default_silu():
    assert [(n, list(p.shape)) for n, p in Float32RMSNorm(4).named_parameters()] == [
        (n, list(p.shape)) for n, p in RMSNorm(4).named_parameters()
    ]
    precise = _block(ir.DataType.BFLOAT16)
    original = Float32AccumulationMamba2Block(4, 2, 1, 2, 1, conv_kernel=4)
    _cast_module_dtype(original, ir.DataType.BFLOAT16)
    assert [(n, list(p.shape)) for n, p in precise.named_parameters()] == [
        (n, list(p.shape)) for n, p in original.named_parameters()
    ]
    builder, op, _ = create_test_builder()
    x = create_test_input(builder, "x", [1, 4, 1], ir.DataType.BFLOAT16)
    state = create_test_input(builder, "state", [1, 4, 3], ir.DataType.BFLOAT16)
    activated, _ = original.conv1d(op, x, state)
    assert activated.producer().op_type == "Swish"
    assert activated.producer().inputs[0].producer().op_type == "CastLike"


@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    ["norm_0", "norm_2", "silu_0_source", "silu_0_native", "silu_2_source", "silu_2_native"],
)
def test_pinned_bolt_primitive_values(tmp_path, precision_runtime, case):
    root = os.environ.get("MOBIUS_PRECISION_FIXTURE")
    if not root:
        pytest.skip("requires explicitly configured immutable Bolt primitive fixtures")
    assert precision_runtime is not None, (
        "configured real fixtures require qualified CUDA execution"
    )
    root = Path(root)
    metadata = json.loads((root / "fixture.json").read_text())
    with np.load(root / "fixture.npz") as archive:
        arrays = {
            key: archive[key].copy()
            for key in ("gamma", case + "_x", case + "_expected", case + "_actual")
        }
    for key, array in arrays.items():
        assert hashlib.sha256(array.tobytes()).hexdigest() == metadata["arrays"][key]["sha256"]
    dtype = ir.DataType.BFLOAT16
    x = arrays[case + "_x"].view(dtype.numpy())
    if case.startswith("norm"):
        model = _norm_model(dtype, list(x.shape), arrays["gamma"].view(dtype.numpy()))
    else:
        block = _production_block()
        builder, op, graph = create_test_builder()
        value = create_test_input(builder, "x", list(x.shape), dtype)
        output = block.conv1d._activate(op, value)
        output.name, output.shape, output.dtype = "output", ir.Shape(list(x.shape)), dtype
        graph.outputs.append(output)
        model = ir.Model(graph, ir_version=11)
    (actual,) = _run(model, {"x": x}, tmp_path, precision_runtime)
    expected = arrays[case + "_expected"].view(dtype.numpy()).astype(np.float64)
    old = arrays[case + "_actual"].view(dtype.numpy()).astype(np.float64)
    a = actual.astype(np.float64)
    error2, expected2, actual2 = (
        np.square(a - expected).sum(),
        np.square(expected).sum(),
        np.square(a).sum(),
    )
    nrmse = np.sqrt(error2 / expected2) if expected2 else (0.0 if error2 == 0 else np.inf)
    cosine = (
        np.vdot(a, expected) / np.sqrt(actual2 * expected2)
        if actual2 and expected2
        else (1.0 if not actual2 and not expected2 else 0.0)
    )
    assert np.isfinite(a).all() and np.isfinite(expected).all()
    assert nrmse <= 0.03 and cosine >= 0.999
    assert error2 < np.square(old - expected).sum() / 100
    np.savez(
        tmp_path / "values.npz",
        actual=actual.view(np.uint16),
        expected=arrays[case + "_expected"],
    )
    (tmp_path / "fixture-metrics.json").write_text(
        json.dumps(
            {
                "case": case,
                "shape": list(x.shape),
                "dtype": "BFLOAT16",
                "nrmse": float(nrmse),
                "cosine": float(cosine),
                "finite": True,
                "bitwise_equal_source": bool(
                    np.array_equal(actual.view(np.uint16), arrays[case + "_expected"])
                ),
                "different_elements": int(
                    np.count_nonzero(actual.view(np.uint16) != arrays[case + "_expected"])
                ),
                "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                "expected_sha256": metadata["arrays"][case + "_expected"]["sha256"],
                "full_decoder_parity": "NOT_RUN",
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.integration
@pytest.mark.parametrize("layer", [0, 2])
@pytest.mark.parametrize("prior", ["source", "native"])
def test_pinned_bolt_complete_conv_silu(tmp_path, precision_runtime, layer, prior):
    root = os.environ.get("MOBIUS_BOLT_CHECKPOINT_ROOT")
    evidence = os.environ.get("MOBIUS_BOLT_VALIDATION_ROOT")
    if not root or not evidence:
        pytest.skip("requires explicitly configured local pinned checkpoint/evidence")
    assert precision_runtime is not None
    from mobius.components._mamba2_precision_test import _checkpoint_parameter

    evidence = Path(evidence)
    block = _production_block()
    checkpoint_hashes = {}
    for parameter, suffix in (
        (block.conv1d.weight, "conv1d.weight"),
        (block.conv1d.bias, "conv1d.bias"),
    ):
        tensor = _checkpoint_parameter(Path(root), evidence, layer, suffix)
        checkpoint_hashes[suffix] = hashlib.sha256(
            tensor.contiguous().view(torch.uint16).numpy().tobytes()
        ).hexdigest()
        parameter.const_value = tensor_adapters.TorchTensor(tensor)
    case = f"source_inputs_{prior}_prior"
    path = (
        evidence
        / "diagnostic-native-v7-hidden-v1/mamba-causal"
        / f"source-layer{layer}-{case}.npz"
    )
    with np.load(path) as source:
        # The pinned cached branch squeezes sequence before projection/Conv.
        x = source["projection"][:, 8192:18432, None].copy()
        expected = source["conv_activated"][..., None].copy()
        biased = source["conv_before_silu"][..., None].copy()
    dtype = ir.DataType.BFLOAT16
    if prior == "source":
        with np.load(evidence / "reference-source-v1/decode-01.npz") as previous:
            state = previous[f"conv_states_{layer:02d}"][..., -3:].astype(dtype.numpy())
    else:
        with np.load(evidence / "full-native-v7-ordinary-v1/decoder-step-01.npz") as previous:
            state = previous[f"present.{layer}.conv_state"].view(dtype.numpy()).copy()
    feeds = {"x": x.astype(dtype.numpy()), "state": state}
    traced, traced_state, actual_biased = _run(
        _conv_model(block, dtype, 1, trace=True),
        feeds,
        tmp_path / "instrumented",
        precision_runtime,
    )
    uninstrumented = ir.load(tmp_path / "instrumented/model.onnx")
    uninstrumented.graph.outputs.pop()
    actual, actual_state = _run(uninstrumented, feeds, tmp_path, precision_runtime)
    np.testing.assert_array_equal(actual, traced)
    np.testing.assert_array_equal(actual_state, traced_state)
    np.testing.assert_array_equal(actual_biased.astype(np.float32), biased)
    np.testing.assert_array_equal(actual.astype(np.float32), expected)
    optimized = ir.load(tmp_path / "optimized.onnx")
    native_stores = [
        node
        for node in optimized.graph
        if node.op_type == "Cast" and node.attributes["to"].as_int() == int(dtype)
    ]
    assert len(native_stores) >= 2
    assert any(
        node.inputs[0].producer() is not None
        and node.inputs[0].producer().op_type in ("Conv", "FusedConv")
        for node in native_stores
    )
    conv_store = next(
        node
        for node in native_stores
        if node.inputs[0].producer() is not None
        and node.inputs[0].producer().op_type in ("Conv", "FusedConv")
    )
    assert any(
        node.op_type == "Cast"
        and node.attributes["to"].as_int() == int(ir.DataType.FLOAT)
        and node.inputs[0] is conv_store.outputs[0]
        for node in optimized.graph
    )
    assert optimized.graph.outputs[0].producer() in native_stores
    np.savez(
        tmp_path / "values.npz",
        actual_activation=actual.view(np.uint16),
        actual_biased_conv=actual_biased.view(np.uint16),
        actual_state=actual_state.view(np.uint16),
        expected_activation=expected.astype(dtype.numpy()).view(np.uint16),
        expected_biased_conv=biased.astype(dtype.numpy()).view(np.uint16),
    )
    (tmp_path / "fixture-metrics.json").write_text(
        json.dumps(
            {
                "case": f"complete_conv_layer{layer}_{prior}_prior",
                "shape": list(actual.shape),
                "dtype": "BFLOAT16",
                "native_stores_preserved_optimized": True,
                "common_outputs_bitwise_unchanged_by_instrumentation": True,
                "biased_conv_bitwise_source_exact": True,
                "activation_bitwise_source_exact": True,
                "nrmse": 0.0,
                "cosine": 1.0,
                "different_elements": 0,
                "source_array_file": str(path),
                "source_array_file_sha256": _sha(path),
                "checkpoint_tensor_sha256": checkpoint_hashes,
                "current_input_sha256": hashlib.sha256(feeds["x"].tobytes()).hexdigest(),
                "prior_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "optimized_native_store_values": [
                    node.outputs[0].name for node in native_stores
                ],
                "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                "full_decoder_parity": "NOT_RUN",
            },
            indent=2,
        )
        + "\n"
    )
