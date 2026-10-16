# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mamba2 accumulation-boundary regressions, including opt-in Bolt fixtures."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnx_ir import tensor_adapters

from mobius._builder import _cast_module_dtype
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _ort_value_to_numpy
from mobius.components import Float32AccumulationMamba2Block, Mamba2Block

_DTYPES = [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]


def _block(dtype, *, heads=1, head_dim=2, state_size=1, groups=1, kernel=2, model_size=4):
    block = Float32AccumulationMamba2Block(
        d_model=model_size,
        d_inner=heads * head_dim,
        num_heads=heads,
        d_head=head_dim,
        d_state=state_size,
        n_groups=groups,
        conv_kernel=kernel,
    )
    _cast_module_dtype(block, dtype)
    return block


def _parameter(parameter, tensor):
    parameter.const_value = tensor_adapters.TorchTensor(tensor)
    parameter.type = ir.TensorType(parameter.const_value.dtype)


def _conv_graph(block, dtype, length):
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, block.conv_dim, length], dtype)
    state = create_test_input(
        builder, "state", [1, block.conv_dim, block.conv_kernel - 1], dtype
    )
    activated, present = block.conv1d(op, x, state)
    assert activated.producer().op_type == "Swish"
    preactivation = activated.producer().inputs[0]
    assert preactivation.producer().op_type == "CastLike"
    conv = preactivation.producer().inputs[0].producer()
    assert conv.op_type == "Conv"
    if block.conv1d.bias is not None:
        assert len(conv.inputs) == 3 and conv.inputs[2] is not None
    assert all(value.dtype == ir.DataType.FLOAT for value in conv.inputs if value is not None)
    assert activated.dtype == present.dtype == dtype
    # Cut BEFORE activation: this regression must not hide the independently
    # tracked frozen-runtime Sigmoid problem or change activation semantics.
    graph = ir.convenience.extract(graph, [x, state], [preactivation, present])
    for value, name, shape in (
        (preactivation, "biased_conv", [1, block.conv_dim, length]),
        (present, "present", [1, block.conv_dim, block.conv_kernel - 1]),
    ):
        value.name, value.shape, value.dtype = name, ir.Shape(shape), dtype
    return ir.Model(graph, ir_version=11)


def _readout_graph(block, dtype, length=1):
    builder, op, graph = create_test_builder()
    readout = create_test_input(builder, "readout", [1, length, block.d_inner])
    x = create_test_input(builder, "x", [1, length, block.num_heads, block.d_head])
    reference = create_test_input(builder, "reference", [1, length, block.d_model], dtype)
    output = block._add_ssm_skip(op, readout, x, reference)
    # This focused helper call bypasses Module.__call__'s parameter realization.
    graph.initializers[block.D.name] = block.D
    assert output.producer().op_type == "CastLike"
    addition = output.producer().inputs[0].producer()
    assert addition.op_type == "Add"
    assert all(value.dtype == ir.DataType.FLOAT for value in addition.inputs)
    output.name, output.shape, output.dtype = (
        "output",
        ir.Shape([1, length, block.d_inner]),
        dtype,
    )
    graph.outputs.append(output)
    return ir.Model(graph, ir_version=11)


def _run(model, feeds, directory, *, cuda=False):
    directory.mkdir(exist_ok=True)
    path = directory / "model.onnx"
    ir.save(model, path)
    options = ort.SessionOptions()
    if cuda:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
        options.enable_profiling = True
        options.profile_file_prefix = str(directory / "profile")
    providers = (
        [("CUDAExecutionProvider", {"use_tf32": "0"})] if cuda else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    values = {}
    types = {
        "tensor(float)": ir.DataType.FLOAT,
        "tensor(float16)": ir.DataType.FLOAT16,
        "tensor(bfloat16)": ir.DataType.BFLOAT16,
    }
    for info in session.get_inputs():
        array, dtype = feeds[info.name], types[info.type]
        # Own storage in ORT rather than relying on temporary NumPy/BF16 views.
        value = ort.OrtValue.ortvalue_from_shape_and_type(list(array.shape), int(dtype))
        if array.size:
            raw = array.view(np.uint16) if dtype == ir.DataType.BFLOAT16 else array
            value.update_inplace(np.ascontiguousarray(raw))
        values[info.name] = value
    result = [_ort_value_to_numpy(v) for v in session.run_with_ort_values(None, values)]
    if cuda:
        path = Path(session.end_profiling())
        counts = {}
        for event in json.loads(path.read_text()):
            args = event.get("args", {})
            provider, op = args.get("provider"), args.get("op_name")
            if not provider:
                continue
            counts[f"{provider}::{op}"] = counts.get(f"{provider}::{op}", 0) + 1
            if provider == "CPUExecutionProvider":
                types_used = [
                    t
                    for field in ("input_type_shape", "output_type_shape")
                    for entry in args.get(field, [])
                    for t in entry
                ]
                assert op in ("Shape", "Size") or not any(
                    "float" in t or t in ("double", "complex64", "complex128")
                    for t in types_used
                ), event
        assert any(key.startswith("CUDAExecutionProvider::") for key in counts)
        (directory / "provider-audit.json").write_text(
            json.dumps(
                {
                    "runtime": ort.__version__,
                    "runtime_path": ort.__file__,
                    "counts": counts,
                    "unexpected_cpu_floating": [],
                    "activation": "NOT_EXECUTED",
                    "full_decoder_parity": "NOT_RUN",
                },
                indent=2,
            )
            + "\n"
        )
    return result


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("length,kernel", [(1, 2), (3, 2), (3, 1)])
def test_float32_biased_conv_store_and_cache(tmp_path, dtype, length, kernel):
    block = _block(dtype, kernel=kernel)
    native = tensor_adapters.to_torch_dtype(dtype)
    half_ulp = torch.finfo(native).eps / 2
    weight = torch.zeros((block.conv_dim, 1, kernel), dtype=native)
    weight[:, 0, 0] = 1
    if kernel > 1:
        weight[:, 0, 1] = half_ulp * 2
    bias = torch.full((block.conv_dim,), half_ulp, dtype=native)
    _parameter(block.conv1d.weight, weight)
    _parameter(block.conv1d.bias, bias)
    x = torch.full((1, block.conv_dim, length), 0.5, dtype=native)
    state = torch.ones((1, block.conv_dim, kernel - 1), dtype=native)
    window = torch.cat((state, x), -1)
    expected = torch.nn.functional.conv1d(
        window.float(), weight.float(), bias.float(), groups=block.conv_dim
    ).to(native)
    feeds = {
        "x": x.float().numpy().astype(dtype.numpy()),
        "state": state.float().numpy().astype(dtype.numpy()),
    }
    actual, present = _run(_conv_graph(block, dtype, length), feeds, tmp_path)
    np.testing.assert_array_equal(actual.astype(np.float32), expected.float().numpy())
    expected_state = window[..., -kernel + 1 :] if kernel > 1 else window[..., :0]
    np.testing.assert_array_equal(present.astype(np.float32), expected_state.float().numpy())
    if dtype != ir.DataType.FLOAT and kernel > 1:
        dot = torch.nn.functional.conv1d(
            window.float(), weight.float(), groups=block.conv_dim
        ).to(native)
        old = dot + bias[None, :, None]
        assert not torch.equal(old, expected), "fixture must detect premature bias rounding"


@pytest.mark.parametrize("dtype", _DTYPES)
def test_readout_skip_has_one_final_store(tmp_path, dtype):
    block = _block(dtype)
    native = tensor_adapters.to_torch_dtype(dtype)
    _parameter(block.D, torch.ones(block.num_heads, dtype=native))
    half_ulp = torch.finfo(native).eps / 2
    readout = np.full((1, 1, block.d_inner), 1 + half_ulp, dtype=np.float32)
    x = np.full((1, 1, block.num_heads, block.d_head), half_ulp, dtype=np.float32)
    expected = torch.from_numpy(readout + x.reshape(readout.shape)).to(native)
    (actual,) = _run(
        _readout_graph(block, dtype),
        {"readout": readout, "x": x, "reference": np.zeros((1, 1, 4), dtype=dtype.numpy())},
        tmp_path,
    )
    np.testing.assert_array_equal(actual.astype(np.float32), expected.float().numpy())
    if dtype != ir.DataType.FLOAT:
        old = torch.from_numpy(readout).to(native) + torch.from_numpy(
            x.reshape(readout.shape)
        ).to(native)
        assert not torch.equal(old, expected), "fixture must detect the two early casts"


def test_readout_skip_fp16_cancellation_stays_finite(tmp_path):
    block = _block(ir.DataType.FLOAT16)
    _parameter(block.D, torch.full((1,), 2.0, dtype=torch.float16))
    (actual,) = _run(
        _readout_graph(block, ir.DataType.FLOAT16),
        {
            "readout": np.full((1, 1, 2), 131009.0, dtype=np.float32),
            "x": np.full((1, 1, 1, 2), -65504.0, dtype=np.float32),
            "reference": np.zeros((1, 1, 4), dtype=np.float16),
        },
        tmp_path,
    )
    np.testing.assert_array_equal(actual, np.ones((1, 1, 2), dtype=np.float16))


def test_default_mamba2_keeps_native_add_and_parameter_names():
    precise = _block(ir.DataType.BFLOAT16)
    default = Mamba2Block(4, 2, 1, 2, 1, conv_kernel=2)
    _cast_module_dtype(default, ir.DataType.BFLOAT16)
    assert [(n, list(p.shape)) for n, p in precise.named_parameters()] == [
        (n, list(p.shape)) for n, p in default.named_parameters()
    ]
    builder, op, graph = create_test_builder()
    readout = create_test_input(builder, "readout", [1, 1, 2])
    x = create_test_input(builder, "x", [1, 1, 1, 2])
    reference = create_test_input(builder, "reference", [1, 1, 4], ir.DataType.BFLOAT16)
    result = default._add_ssm_skip(op, readout, x, reference)
    assert result.producer().op_type == "Add"
    assert all(v.producer().op_type == "CastLike" for v in result.producer().inputs)
    state = create_test_input(builder, "state", [1, 4, 1], ir.DataType.BFLOAT16)
    value = create_test_input(builder, "value", [1, 4, 1], ir.DataType.BFLOAT16)
    default.conv1d(op, value, state)
    conv = next(n for n in graph if n.op_type == "CausalConvWithState")
    assert conv.attributes["activation"].as_string() == "silu"


def test_float32_conv_without_bias(tmp_path):
    block = Float32AccumulationMamba2Block(4, 2, 1, 2, 1, conv_kernel=2, conv_bias=False)
    _cast_module_dtype(block, ir.DataType.BFLOAT16)
    weight = torch.ones((block.conv_dim, 1, 2), dtype=torch.bfloat16)
    _parameter(block.conv1d.weight, weight)
    x = np.full((1, block.conv_dim, 1), 0.5, dtype=ir.DataType.BFLOAT16.numpy())
    state = np.zeros_like(x)
    result, _ = _run(
        _conv_graph(block, ir.DataType.BFLOAT16, 1),
        {"x": x, "state": state},
        tmp_path,
    )
    np.testing.assert_array_equal(result.astype(np.float32), x.astype(np.float32))
    assert block.conv1d.bias is None


def _bolt_fixture():
    checkpoint = os.environ.get("MOBIUS_BOLT_CHECKPOINT_ROOT")
    validation = os.environ.get("MOBIUS_BOLT_VALIDATION_ROOT")
    if not checkpoint or not validation:
        pytest.skip("requires explicitly configured local pinned Bolt fixture roots")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider is required for actual checkpoint regression")
    root, evidence = Path(checkpoint), Path(validation)
    manifest = json.loads(
        (evidence / "reference-source-v1/artifact-manifest.json").read_text()
    )
    assert manifest["revision"] == "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a"
    return root, evidence


def _checkpoint_parameter(root, evidence, layer, suffix):
    from safetensors import safe_open

    key = f"language_model.backbone.layers.{layer}.mixer.{suffix}"
    index = json.loads((root / "source/model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(str(root / "checkpoint" / index[key]), framework="pt") as f:
        value = f.get_tensor(key)
    strict = json.loads((evidence / "reference-source-v1/strict-weights.json").read_text())
    digest = hashlib.sha256(
        value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    assert digest == strict["consumed"][key]["sha256"]
    return value


def _assert_source_gates(tmp_path, actual, expected, old):
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    error2 = np.square(a - e).sum()
    expected2, actual2 = np.square(e).sum(), np.square(a).sum()
    nrmse = np.sqrt(error2 / expected2) if expected2 else (0.0 if not error2 else np.inf)
    cosine = (
        np.dot(a.ravel(), e.ravel()) / np.sqrt(actual2 * expected2)
        if actual2 and expected2
        else (1.0 if not actual2 and not expected2 else 0.0)
    )
    assert np.isfinite(a).all() and np.isfinite(e).all()
    assert nrmse <= 0.03 and cosine >= 0.999
    old_error = np.square(old.astype(np.float64) - e).sum()
    assert error2 < old_error / 100, "actual fixture must distinguish the corrected boundary"
    np.savez(tmp_path / "arrays.npz", actual=actual, expected=expected, old=old)
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "nrmse": float(nrmse),
                "cosine": float(cosine),
                "error_squared_f64": float(error2),
                "old_error_squared_f64": float(old_error),
                "native_dtype": "BFLOAT16",
                "full_decoder_parity": "NOT_RUN",
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.integration
@pytest.mark.parametrize("layer", [0, 2])
@pytest.mark.parametrize("prior", ["source", "native"])
def test_bolt_actual_checkpoint_conv_bias_same_inputs(tmp_path, layer, prior):
    root, evidence = _bolt_fixture()
    block = _block(
        ir.DataType.BFLOAT16,
        heads=128,
        head_dim=64,
        state_size=128,
        groups=8,
        kernel=4,
        model_size=4096,
    )
    weight = _checkpoint_parameter(root, evidence, layer, "conv1d.weight")
    bias = _checkpoint_parameter(root, evidence, layer, "conv1d.bias")
    _parameter(block.conv1d.weight, weight)
    _parameter(block.conv1d.bias, bias)
    case = f"source_inputs_{prior}_prior"
    diagnostic = evidence / "diagnostic-native-v5-mamba-v1"
    with np.load(diagnostic / f"native-layer{layer}-{case}.npz") as archive:
        x = archive["projection"][..., 8192:18432].transpose(0, 2, 1).copy()
        old = archive["conv_before_silu"].view(ir.DataType.BFLOAT16.numpy()).astype(np.float32)
    if prior == "source":
        with np.load(evidence / "reference-source-v1/decode-01.npz") as previous:
            state = previous[f"conv_states_{layer:02d}"][..., -3:].astype(
                ir.DataType.BFLOAT16.numpy()
            )
    else:
        with np.load(evidence / "full-native-v5-ordinary-v2/decoder-step-01.npz") as previous:
            state = (
                previous[f"present.{layer}.conv_state"]
                .view(ir.DataType.BFLOAT16.numpy())
                .copy()
            )
    x = x.view(ir.DataType.BFLOAT16.numpy())
    window = (
        torch.from_numpy(np.concatenate((state, x), -1).astype(np.float32)).bfloat16().cuda()
    )
    with (
        torch.inference_mode(),
        torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=True, allow_tf32=False
        ),
    ):
        expected = (
            torch.nn.functional.conv1d(
                window, weight.cuda(), bias.cuda(), groups=block.conv_dim
            )
            .float()
            .cpu()
            .numpy()
        )
    actual, present = _run(
        _conv_graph(block, ir.DataType.BFLOAT16, 1),
        {"x": x, "state": state},
        tmp_path,
        cuda=True,
    )
    _assert_source_gates(tmp_path, actual.astype(np.float32), expected, old)
    np.testing.assert_array_equal(
        present.astype(np.float32), window[..., -3:].float().cpu().numpy()
    )


@pytest.mark.integration
@pytest.mark.parametrize("layer", [0, 2])
@pytest.mark.parametrize("prior", ["source", "native"])
def test_bolt_actual_checkpoint_readout_skip_same_inputs(tmp_path, layer, prior):
    root, evidence = _bolt_fixture()
    block = _block(
        ir.DataType.BFLOAT16, heads=128, head_dim=64, state_size=128, groups=8, model_size=4096
    )
    _parameter(block.D, _checkpoint_parameter(root, evidence, layer, "D"))
    case = f"source_inputs_{prior}_prior"
    with np.load(
        evidence / f"diagnostic-native-v5-mamba-v1/source-layer{layer}-{case}.npz"
    ) as source:
        readout = source["pinned_dot_fp32"].copy()
        x = source["conv_activated"][..., :8192].reshape(1, 1, 128, 64).copy()
        expected = source["source_readout_bf16"].copy()
        old = source["two_early_casts_readout"].copy()
    (actual,) = _run(
        _readout_graph(block, ir.DataType.BFLOAT16),
        {
            "readout": readout,
            "x": x,
            "reference": np.zeros((1, 1, block.d_model), dtype=ir.DataType.BFLOAT16.numpy()),
        },
        tmp_path,
        cuda=True,
    )
    _assert_source_gates(tmp_path, actual.astype(np.float32), expected, old)
