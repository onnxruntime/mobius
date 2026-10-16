# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph/API tests and opt-in sealed CUDA fixture replay for chunk prefill."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir.passes.common import CheckerPass

from mobius._optimizations import SymbolicShapeInferencePass
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy
from mobius.components import ChunkedPrefillScan
from mobius.components._chunked_prefill import _chunk_cumsum

_CONTRACT_SHA = "fb3f5767e22567bf8038d4a306f577bdd045415deaa7c9dc8df00d54231f77e0"
_FIXTURE_ROOT = os.environ.get("MOBIUS_CHUNK_PREFILL_FIXTURES")
_CASES = [
    f"scan-B{b}-S{s}-init{i}"
    for b in (1, 2)
    for s in (1, 127, 128, 129, 255, 256, 257)
    for i in (0, 1)
] + ["scan-B2-S129-init1-productionH128", "scan-B2-S257-init1-finite-clamp"]


def _graph(batch=2, length="time", heads=4, groups=2, head=64, state=128, *, initial=True):
    module = ChunkedPrefillScan(heads, groups, head, state)
    builder, op, graph = create_test_builder()
    values = [
        create_test_input(builder, name, shape, dtype)
        for name, shape, dtype in (
            ("x", [batch, length, heads, head], ir.DataType.BFLOAT16),
            ("dt", [batch, length, heads], ir.DataType.FLOAT),
            ("a", [heads], ir.DataType.FLOAT),
            ("b", [batch, length, groups, state], ir.DataType.BFLOAT16),
            ("c", [batch, length, groups, state], ir.DataType.BFLOAT16),
        )
    ]
    if initial:
        values.append(
            create_test_input(
                builder, "initial_state", [batch, heads, head, state], ir.DataType.FLOAT
            )
        )
    output, final = module(op, *values)
    for tensor, name, shape in (
        (output, "readout", [batch, length, heads, head]),
        (final, "final_state", [batch, heads, head, state]),
    ):
        tensor.name, tensor.shape = name, ir.Shape(shape)
        graph.outputs.append(tensor)
    return ir.Model(graph, ir_version=11)


@pytest.mark.parametrize("initial", [False, True])
def test_public_parameter_free_positional_api(initial):
    model = _graph(initial=initial)
    SymbolicShapeInferencePass()(model)
    CheckerPass()(model)
    assert list(ChunkedPrefillScan(4, 2, 64, 128).named_parameters()) == []
    assert all(v.dtype == ir.DataType.INT64 for v in model.graph.initializers.values())
    assert all(v.dtype == ir.DataType.FLOAT for v in model.graph.outputs)
    scan = next(node for node in model.graph if node.op_type == "Scan")
    assert scan.attributes["num_scan_inputs"].as_int() == 4
    body = scan.attributes["body"].as_graph()
    assert all(v.dtype == ir.DataType.FLOAT for v in body.inputs[:2])
    assert not any(node.domain for node in body)
    assert not any(node.op_type in ("CumSum", "LinearAttention") for node in body)
    dots = [node for node in body if node.op_type == "Scan"]
    assert len(dots) == 4
    for dot in dots:
        nested = dot.attributes["body"].as_graph()
        gemm = next(node for node in nested if node.op_type == "GemmFloat8")
        assert all(v.dtype == ir.DataType.BFLOAT16 for v in gemm.inputs)
        assert gemm.outputs[0].dtype == ir.DataType.FLOAT
        assert gemm.attributes["dtype"].as_int() == int(ir.DataType.FLOAT)
    assert any(
        node.op_type == "Cast" and node.outputs[0].dtype == ir.DataType.BFLOAT16
        for node in body
    )


@pytest.mark.parametrize("dimensions", [(0, 1, 1, 1), (3, 2, 4, 8), (1, 1, -1, 8)])
def test_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        ChunkedPrefillScan(*dimensions)


def test_rejects_empty_prefill():
    with pytest.raises(ValueError):
        _graph(length=0)


def test_rejects_wrong_static_head_dimension():
    builder, op, _ = create_test_builder()
    x = create_test_input(builder, "x", [1, 3, 2, 8], ir.DataType.BFLOAT16)
    dt = create_test_input(builder, "dt", [1, 3, 2])
    a = create_test_input(builder, "a", [2])
    b = create_test_input(builder, "b", [1, 3, 1, 8], ir.DataType.BFLOAT16)
    c = create_test_input(builder, "c", [1, 3, 1, 8], ir.DataType.BFLOAT16)
    with pytest.raises(ValueError, match="Dimension 3"):
        ChunkedPrefillScan(2, 1, 4, 8)(op, x, dt, a, b, c)


def test_padded_lane_tree_and_fma_on_cpu(tmp_path):
    builder, op, graph = create_test_builder()
    dt_input = create_test_input(builder, "dt", [2, 3, 128])
    a_input = create_test_input(builder, "a", [1, 3, 1])
    result = _chunk_cumsum(op, dt_input, a_input)
    result.name, result.shape = "decay", ir.Shape([2, 3, 128])
    graph.outputs.append(result)
    path = tmp_path / "tree.onnx"
    ir.save(ir.Model(graph, ir_version=11), path)
    rng = np.random.default_rng(2)
    dt = rng.uniform(0.02, 0.2, (2, 3, 128)).astype(np.float32)
    dt[..., -1] = 0
    a = -np.linspace(0.025, 0.125, 3, dtype=np.float32)[None, :, None]
    product = dt * a
    lanes = np.arange(128)
    shifted = np.take(product, np.maximum(lanes - 1, 0), axis=-1)
    expected = np.where(
        lanes % 32 > 0,
        (dt.astype(np.float64) * a + shifted.astype(np.float64)).astype(np.float32),
        product,
    )
    for distance in (2, 4, 8, 16):
        expected = np.where(
            lanes % 32 >= distance,
            expected + np.take(expected, np.maximum(lanes - distance, 0), axis=-1),
            expected,
        )
    totals = expected[..., 31::32].copy()
    prefix = totals[..., 0]
    for warp in range(1, 4):
        expected[..., warp * 32 : (warp + 1) * 32] += prefix[..., None]
        prefix = prefix + totals[..., warp]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (actual,) = session.run(None, {"dt": dt, "a": a})
    np.testing.assert_array_equal(actual, expected)
    # The zero-padded last lane is not interchangeable with the last valid
    # prefix: this would fail if the implementation used a token CumSum.
    assert np.any(expected[..., -1] != expected[..., -2])


def test_chunk_total_reads_full_padded_lane_not_last_valid_prefix():
    body = ChunkedPrefillScan(4, 2, 64, 128)._body()
    total_slices = [
        node
        for node in body
        if node.op_type == "Slice" and node.inputs[0].name == "prefill_chunk_log_decay"
    ]
    assert len(total_slices) == 1
    total = total_slices[0]
    np.testing.assert_array_equal(total.inputs[1].const_value.numpy(), [127])
    np.testing.assert_array_equal(total.inputs[2].const_value.numpy(), [128])
    assert body.outputs[0].dtype == ir.DataType.FLOAT


@pytest.mark.parametrize("index", range(6))
def test_rejects_wrong_input_dtype(index):
    module = ChunkedPrefillScan(2, 1, 4, 8)
    builder, op, _ = create_test_builder()
    specs = [
        ([1, 3, 2, 4], ir.DataType.BFLOAT16),
        ([1, 3, 2], ir.DataType.FLOAT),
        ([2], ir.DataType.FLOAT),
        ([1, 3, 1, 8], ir.DataType.BFLOAT16),
        ([1, 3, 1, 8], ir.DataType.BFLOAT16),
        ([1, 2, 4, 8], ir.DataType.FLOAT),
    ]
    values = [
        create_test_input(
            builder, f"input_{i}", shape, dtype if i != index else ir.DataType.FLOAT16
        )
        for i, (shape, dtype) in enumerate(specs)
    ]
    with pytest.raises(TypeError):
        module(op, *values)


@pytest.mark.parametrize("index", range(6))
def test_rejects_wrong_input_rank(index):
    module = ChunkedPrefillScan(2, 1, 4, 8)
    builder, op, _ = create_test_builder()
    values = [
        create_test_input(builder, f"input_{i}", shape[1:] if i == index else shape, dtype)
        for i, (shape, dtype) in enumerate(
            [
                ([1, 3, 2, 4], ir.DataType.BFLOAT16),
                ([1, 3, 2], ir.DataType.FLOAT),
                ([2], ir.DataType.FLOAT),
                ([1, 3, 1, 8], ir.DataType.BFLOAT16),
                ([1, 3, 1, 8], ir.DataType.BFLOAT16),
                ([1, 2, 4, 8], ir.DataType.FLOAT),
            ]
        )
    ]
    with pytest.raises(ValueError):
        module(op, *values)


def _load_fixture(name):
    root = Path(_FIXTURE_ROOT)
    contract = root / "algorithm-dtype-contract.json"
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == _CONTRACT_SHA
    path = root / (name + ".npz")
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["npz_sha256"]
    values = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive:
            raw = archive[key]
            spec = metadata["arrays"][key]
            assert hashlib.sha256(raw.tobytes()).hexdigest() == spec["raw_sha256"]
            assert list(raw.shape) == spec["shape"]
            if spec["dtype"] == "torch.bfloat16":
                assert raw.dtype == np.uint16
                raw = raw.view(ir.DataType.BFLOAT16.numpy())
            else:
                assert spec["dtype"] == "torch.float32" and raw.dtype == np.float32
            assert np.isfinite(raw.astype(np.float32)).all()
            values[key] = raw
    return values, metadata


def _add_observers(model):
    scan = next(node for node in model.graph if node.op_type == "Scan")
    body = scan.attributes["body"].as_graph()
    wanted = [
        "chunk_log_decay",
        "chunk_local_state",
        "chunk_cb",
        "chunk_incoming_state",
        "chunk_scaled_b",
        "chunk_coefficient",
    ]
    names = {v.name: v for v in body.inputs}
    names.update({v.name: v for node in body for v in node.outputs})
    for name in wanted:
        tensor = names.get(name, names.get("prefill_" + name))
        assert tensor is not None
        # A Scan graph input cannot simultaneously be a graph output.
        if name == "chunk_incoming_state":
            tensor = ir.Value(name="observed_incoming", type=tensor.type, shape=tensor.shape)
            body.append(ir.Node("", "Identity", [body.inputs[0]], outputs=[tensor]))
        body.outputs.append(tensor)
        scan.resize_outputs(len(scan.outputs) + 1)
        output = scan.outputs[-1]
        output.name, output.type = name, tensor.type
        model.graph.outputs.append(output)
    return model


def _audit_profile(path):
    counts = {}
    for event in json.loads(Path(path).read_text()):
        args = event.get("args", {})
        provider, operation = args.get("provider"), args.get("op_name")
        if not provider:
            continue
        counts[f"{provider}::{operation}"] = counts.get(f"{provider}::{operation}", 0) + 1
        if provider == "CPUExecutionProvider":
            types = [
                dtype
                for field in ("input_type_shape", "output_type_shape")
                for entry in args.get(field, [])
                for dtype in entry
            ]
            assert operation in ("Shape", "Size") or not any(
                "float" in dtype or dtype in ("double", "complex64", "complex128")
                for dtype in types
            ), event
    assert any(key.startswith("CUDAExecutionProvider::") for key in counts)
    return counts


@pytest.fixture(scope="module")
def cuda_worker():
    if not _FIXTURE_ROOT:
        pytest.skip("set MOBIUS_CHUNK_PREFILL_FIXTURES for sealed CUDA source replay")
    # The optional transport helper belongs to the qualification environment;
    # do not import or mutate a protected runtime inside the package.
    from bolt_ort_worker import RemoteOrt

    output = Path(os.environ["MOBIUS_CHUNK_PREFILL_EVIDENCE"])
    output.mkdir(parents=True, exist_ok=True)
    worker = RemoteOrt(
        output / "worker",
        gpu="0",
        build=os.environ["MOBIUS_CHUNK_PREFILL_ORT_BUILD"],
    )
    (output / "runtime.json").write_text(json.dumps(worker.info, indent=2) + "\n")
    try:
        yield worker, output
    finally:
        worker.close()


def _execute(worker, directory, model, name, feeds):
    path = directory / (name + ".onnx")
    ir.save(model, path)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
    options.enable_profiling = True
    options.profile_file_prefix = str(directory / name)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = worker.session(
        path,
        options,
        [("CUDAExecutionProvider", {"use_tf32": "0", "enable_cuda_graph": "0"})],
    )
    try:
        actual = [
            _ort_value_to_numpy(v)
            for v in session.run_with_ort_values(
                None,
                {
                    key: _numpy_to_ort_value(np.ascontiguousarray(v))
                    for key, v in feeds.items()
                },
            )
        ]
        profile = session.end_profiling()
    finally:
        session.close()
    return actual, _audit_profile(profile)


@pytest.mark.parametrize("name", _CASES)
def test_cuda_sealed_source_boundaries(cuda_worker, name):
    worker, directory = cuda_worker
    values, record = _load_fixture(name)
    model = _add_observers(
        _graph(
            record["batch"],
            "time",
            record["heads"],
            record["groups"],
            record["head_dim"],
            record["state_dim"],
        )
    )
    batch, length, heads = record["batch"], record["length"], record["heads"]
    dt = values["dt_processed"].transpose(0, 2, 3, 1).reshape(batch, -1, heads)[:, :length]
    feeds = dict(
        x=values["x"],
        dt=dt,
        a=values["A"],
        b=values["B"],
        c=values["C"],
        initial_state=values["initial_state"],
    )
    actual, counts = _execute(worker, directory, model, name, feeds)
    expected = [
        values["observed_no_D_accumulator_F32"],
        values["final_state_F32"],
        values["dA_cumsum"].transpose(2, 0, 1, 3),
        values["local_chunk_state_F32"].transpose(1, 0, 2, 3, 4),
        values["CB_F32"].transpose(1, 0, 2, 3, 4),
        values["incoming_chunk_state_F32"].transpose(1, 0, 2, 3, 4),
    ]
    metrics = {}
    for key, result, reference in zip(
        ["readout", "final", "decay", "local", "CB", "incoming"], actual[:6], expected
    ):
        assert result.dtype == reference.dtype == np.float32
        delta = result.astype(np.float64) - reference
        norm = np.linalg.norm(reference.astype(np.float64))
        metrics[key] = dict(
            max_abs=float(np.max(np.abs(delta))),
            nrmse=float(np.linalg.norm(delta) / norm)
            if norm
            else float(np.linalg.norm(delta)),
            different=int(np.count_nonzero(result != reference)),
        )
    (directory / (name + "-result.json")).write_text(
        json.dumps(
            {"metrics": metrics, "providers": counts, "fixture_sha256": record["npz_sha256"]},
            indent=2,
        )
        + "\n"
    )
    # These exact gates detect association and early-output-store changes.
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[4], expected[4])
    # CUDA Pow/Exp2 and source ex2.approx are not bitwise equivalent; a
    # one-ULP BF16 operand difference can be magnified by cancellation in a
    # readout element. Gate the full FP32 tensors, not selected elements or
    # token argmaxes. This is primitive numeric evidence, not a model gate.
    assert all(item["nrmse"] < 1e-5 for item in metrics.values())
    scaled_b = actual[6][:, :, 0].transpose(1, 0, 2, 3).astype(np.float32)
    source_b = values["scaled_B_BF16_observed_head0"]
    bf16 = ir.DataType.BFLOAT16.numpy()
    np.testing.assert_array_max_ulp(
        scaled_b.astype(bf16).astype(np.float32),
        source_b.astype(bf16).astype(np.float32),
        maxulp=65536,
    )
    assert actual[6].dtype == actual[7].dtype == bf16
    if name == "scan-B2-S255-init0":
        last_valid, padded_last = actual[2][-1, 0, 3, -2:]
        assert last_valid == np.float32(-1.3044987916946411)
        assert padded_last == np.float32(-1.3044989109039307)


def test_cuda_implicit_zero_initial_state(cuda_worker):
    worker, directory = cuda_worker
    values, _ = _load_fixture("scan-B2-S129-init0")
    dt = values["dt_processed"].transpose(0, 2, 3, 1).reshape(2, -1, 4)[:, :129]
    actual, _ = _execute(
        worker,
        directory,
        _graph(batch="batch", initial=False),
        "implicit-zero",
        dict(x=values["x"], dt=dt, a=values["A"], b=values["B"], c=values["C"]),
    )
    expected = values["final_state_F32"]
    assert np.linalg.norm(actual[1] - expected) / np.linalg.norm(expected) < 1e-5


def test_cuda_checkpoint_prefix_operand_stores(cuda_worker):
    worker, directory = cuda_worker
    path = (
        Path(_FIXTURE_ROOT).parent
        / "diagnostic-native-v8-upstream-prefix-v1"
        / ("production-prefill-regression-fixture.npz")
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "16aeedab8acad611e78c4e6a051b91c87769f593dcc1c1538bece16bcf167fea"
    )
    with np.load(path, allow_pickle=False) as archive:
        values = {k: archive[k].copy() for k in archive}
    bf16 = ir.DataType.BFLOAT16.numpy()
    model = _add_observers(_graph(1, "time", 1, 1, 64, 128, initial=False))
    feeds = dict(
        x=values["x"].view(bf16)[None, :, None, :],
        dt=values["dt_processed"][None, :, None],
        a=values["A"],
        b=values["B"].view(bf16)[None, :, None, :],
        c=values["C"].view(bf16)[None, :, None, :],
    )
    actual, counts = _execute(worker, directory, model, "checkpoint-prefix", feeds)
    readout, final = actual[:2]
    source_state = values["source_final_state"]
    assert np.linalg.norm(final[0, 0] - source_state) / np.linalg.norm(source_state) < 1e-5
    np.testing.assert_array_max_ulp(
        actual[6][0, 0, 0, :27].astype(np.float32),
        values["source_scaled_B_coefficients_observed"],
        maxulp=65536,
    )
    np.testing.assert_array_max_ulp(
        actual[7][0, 0, 0, :27, :27].astype(np.float32),
        values["source_scaled_CB_coefficients_observed"].view(bf16).astype(np.float32),
        maxulp=65536,
    )
    # Exercise the existing contract's external D-before-final-store suffix.
    stored = (readout[0, :, 0] + feeds["x"][0, :, 0].astype(np.float32) * values["D"]).astype(
        bf16
    )
    expected = values["source_readout"].view(bf16).astype(np.float32)
    error = np.linalg.norm(stored.astype(np.float32) - expected) / np.linalg.norm(expected)
    (directory / "checkpoint-prefix-result.json").write_text(
        json.dumps(
            {
                "bf16_readout_nrmse": float(error),
                "providers": counts,
                "different_stored_output_elements": int(
                    np.count_nonzero(stored.astype(np.float32) != expected)
                ),
                "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    assert error < 1e-5
