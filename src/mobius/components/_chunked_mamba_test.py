# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Sealed source-prefill/cache pairs at the post-convolution block boundary."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest

from mobius._builder import _cast_module_dtype
from mobius._testing import create_test_builder, create_test_input
from mobius.components import ChunkedPrefillMamba2Block, Float32SiLUMamba2Block
from mobius.components import _chunked_prefill_test as _core

_PAIRS = [
    f"wrapper-B{batch}-S{length}-{mask}"
    for batch in (1, 2)
    for length in (1, 127, 128, 129, 255, 256, 257)
    for mask in ("ones", "left", "right", "holes", "zero_row")
]
_ADAPTER_FIXTURE_MANIFEST_SHA256 = (
    "8520a2d3cd2085f1307946f8e2b9f13e8165b76285a54d36b5bf58d6b61587ad"
)


def _adapter_fixture_path(name: str, output: Path) -> Path:
    configured = os.environ.get("MOBIUS_CHUNK_ADAPTER_FIXTURES")
    if not configured:
        raise ValueError("Set MOBIUS_CHUNK_ADAPTER_FIXTURES to the sealed adapter inputs")
    root = Path(configured).resolve()
    output = output.resolve()
    if root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Adapter fixture inputs and evidence output must be separate")
    manifest_bytes = (root / "artifact-manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == _ADAPTER_FIXTURE_MANIFEST_SHA256
    entry = json.loads(manifest_bytes)["files"][name]
    path = root / name
    raw = path.read_bytes()
    assert len(raw) == entry["bytes"]
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    return path


@pytest.mark.parametrize("placement", ["same", "nested", "parent"])
def test_adapter_fixture_inputs_cannot_be_evidence_output(tmp_path, monkeypatch, placement):
    inputs = tmp_path / "inputs"
    output = {"same": inputs, "nested": inputs / "outputs", "parent": tmp_path}[placement]
    monkeypatch.setenv("MOBIUS_CHUNK_ADAPTER_FIXTURES", str(inputs))
    with pytest.raises(ValueError, match="must be separate"):
        _adapter_fixture_path("unused.npz", output)


def test_adapter_fixture_inputs_do_not_fall_back_to_evidence(tmp_path, monkeypatch):
    monkeypatch.delenv("MOBIUS_CHUNK_ADAPTER_FIXTURES", raising=False)
    with pytest.raises(ValueError, match="MOBIUS_CHUNK_ADAPTER_FIXTURES"):
        _adapter_fixture_path("unused.npz", tmp_path)


def test_adapter_fixture_inputs_require_pinned_manifest(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "artifact-manifest.json").write_text("{}")
    monkeypatch.setenv("MOBIUS_CHUNK_ADAPTER_FIXTURES", str(inputs))
    with pytest.raises(AssertionError):
        _adapter_fixture_path("unused.npz", tmp_path / "output")


def test_adapter_fixture_inputs_are_protected_before_worker_setup(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    monkeypatch.setenv("MOBIUS_CHUNK_PREFILL_FIXTURES", "configured")
    monkeypatch.setenv("MOBIUS_CHUNK_ADAPTER_FIXTURES", str(inputs))
    monkeypatch.setenv("MOBIUS_CHUNK_ADAPTER_EVIDENCE", str(inputs))
    with pytest.raises(ValueError, match="must be separate"):
        next(worker.__wrapped__())
    assert not inputs.exists()


def test_adapter_fixture_inputs_missing_before_worker_setup(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "runtime.json"
    sentinel.write_text("must remain untouched")
    calls = []

    def forbidden_worker(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Worker must not start without immutable fixture inputs")

    monkeypatch.setenv("MOBIUS_CHUNK_PREFILL_FIXTURES", "configured")
    monkeypatch.delenv("MOBIUS_CHUNK_ADAPTER_FIXTURES", raising=False)
    monkeypatch.setenv("MOBIUS_CHUNK_ADAPTER_EVIDENCE", str(output))
    monkeypatch.setitem(
        sys.modules, "bolt_ort_worker", SimpleNamespace(RemoteOrt=forbidden_worker)
    )
    with pytest.raises(ValueError, match="MOBIUS_CHUNK_ADAPTER_FIXTURES"):
        next(worker.__wrapped__())
    assert calls == []
    assert sentinel.read_text() == "must remain untouched"
    assert list(output.iterdir()) == [sentinel]


def _parameter(parameter, value):
    parameter.const_value = ir.tensor(np.asarray(value))
    parameter.type = ir.TensorType(parameter.const_value.dtype)


def _readout_graph(
    values,
    *,
    heads=1,
    groups=1,
    head=64,
    state=128,
    clamp=(0.0, None),
    cached_baseline=False,
    observe_preparation=False,
):
    cls = Float32SiLUMamba2Block if cached_baseline else ChunkedPrefillMamba2Block
    block = cls(
        d_model=4,
        d_inner=heads * head,
        num_heads=heads,
        d_head=head,
        d_state=state,
        n_groups=groups,
        time_step_min=clamp[0],
        time_step_max=clamp[1],
    )
    _cast_module_dtype(block, ir.DataType.BFLOAT16)
    _parameter(block.A_log, np.log(-values["A"]).astype(np.float32))
    _parameter(block.dt_bias, values["dt_bias"])
    _parameter(block.D, values["D"])
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", ["batch", "time", heads * head], ir.DataType.BFLOAT16)
    b = create_test_input(
        builder, "b", ["batch", "time", groups * state], ir.DataType.BFLOAT16
    )
    c = create_test_input(
        builder, "c", ["batch", "time", groups * state], ir.DataType.BFLOAT16
    )
    dt = create_test_input(builder, "dt", ["batch", "time", heads], ir.DataType.BFLOAT16)
    past = create_test_input(builder, "state", ["batch", heads, state, head])
    phase = create_test_input(builder, "has_previous_state", [], ir.DataType.BOOL)
    readout, final, x_4d = block._ssm_readout(
        op, x, b, c, dt, past, None if cached_baseline else phase
    )
    stored = block._add_ssm_skip(op, readout, x_4d, x)
    for parameter in (block.A_log, block.dt_bias, block.D):
        graph.initializers[parameter.name] = parameter
    for value, name, shape, dtype in (
        (readout, "readout", ["batch", "time", heads * head], ir.DataType.FLOAT),
        (final, "final_state", ["batch", heads, state, head], ir.DataType.FLOAT),
        (stored, "stored", ["batch", "time", heads * head], ir.DataType.BFLOAT16),
    ):
        value.name, value.shape, value.dtype = name, ir.Shape(shape), dtype
        graph.outputs.append(value)
    if observe_preparation:
        prepared = block._prefill_dt(op, dt, op.Cast(block.dt_bias, to=ir.DataType.FLOAT))
        prepared.name, prepared.shape = "prepared_dt", ir.Shape(["batch", "time", heads])
        negative_a = op.Neg(op.Exp(op.Cast(block.A_log, to=ir.DataType.FLOAT)))
        negative_a.name, negative_a.shape, negative_a.dtype = (
            "negative_a",
            ir.Shape([heads]),
            ir.DataType.FLOAT,
        )
        graph.outputs.extend([prepared, negative_a])
    graph.opset_imports["com.microsoft"] = 1
    return ir.Model(graph, ir_version=11)


@pytest.fixture(scope="module")
def worker():
    if not os.environ.get("MOBIUS_CHUNK_PREFILL_FIXTURES"):
        pytest.skip("requires explicitly configured sealed source fixtures")
    directory = Path(os.environ["MOBIUS_CHUNK_ADAPTER_EVIDENCE"])
    # Require inputs and reject overlaps before creating runtime/evidence files.
    _adapter_fixture_path("receipt.json", directory)
    from bolt_ort_worker import RemoteOrt

    directory.mkdir(parents=True, exist_ok=True)
    remote = RemoteOrt(
        directory / ("worker-" + uuid.uuid4().hex),
        gpu=os.environ.get("MOBIUS_CHUNK_ADAPTER_GPU", "0"),
        build=os.environ["MOBIUS_CHUNK_PREFILL_ORT_BUILD"],
    )
    (directory / "runtime.json").write_text(json.dumps(remote.info, indent=2) + "\n")
    try:
        yield remote, directory
    finally:
        remote.close()


def _feeds(values, cached, state=None):
    x = values["x"][:, None] if cached else values["x"]
    b = values["B"][:, None] if cached else values["B"]
    c = values["C"][:, None] if cached else values["C"]
    dt = values["dt_raw"][:, None] if cached else values["dt_raw"]
    batch = x.shape[0]
    if state is None:
        state = (
            values["prior_state"][:, None].transpose(0, 1, 3, 2)
            if cached
            else np.full((batch, 1, 128, 64), 17.25, dtype=np.float32)
        )
    return dict(
        x=x,
        b=b,
        c=c,
        dt=dt[..., None],
        state=state,
        has_previous_state=np.asarray(cached, dtype=np.bool_),
    )


def _metrics(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    norm = np.linalg.norm(expected.astype(np.float64))
    return {
        "nrmse": float(np.linalg.norm(delta) / norm) if norm else float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))),
        "different": int(np.count_nonzero(actual != expected)),
    }


@pytest.mark.parametrize("name", _PAIRS)
def test_cuda_checkpoint_mask_prefill_cached_transition(worker, name):
    remote, directory = worker
    pre, metadata = _core._load_fixture(name + "-prefill")
    cached, _ = _core._load_fixture(name + "-cached")
    model = _readout_graph(pre)
    initial_feeds = _feeds(pre, False)
    initial, providers = _core._execute(
        remote, directory, model, name + "-prefill", initial_feeds
    )
    final = initial[1][:, 0].transpose(0, 2, 1)
    expected = pre["final_state"]
    pre_metrics = _metrics(final, expected)
    assert pre_metrics["nrmse"] < 1e-5
    # Same source inputs, zero versus dirty prior. False-phase prefill must
    # overwrite, not consume, the native FP32 cached state even for S1.
    if pre["x"].shape[1] in (1, 129) and name.endswith("ones"):
        clean, _ = _core._execute(
            remote,
            directory,
            model,
            name + "-clean",
            {**initial_feeds, "state": np.zeros_like(initial_feeds["state"])},
        )
        for a, b in zip(initial, clean):
            np.testing.assert_array_equal(a, b)
    next_feeds = _feeds(cached, True, initial[1])
    following, cached_providers = _core._execute(
        remote, directory, model, name + "-cached", next_feeds
    )
    final = following[1][:, 0].transpose(0, 2, 1)
    cached_metrics = _metrics(final, cached["final_state"])
    assert cached_metrics["nrmse"] < 1e-5
    baseline, _ = _core._execute(
        remote,
        directory,
        _readout_graph(pre, cached_baseline=True),
        name + "-existing-cached",
        next_feeds,
    )
    for actual, original in zip(following, baseline):
        np.testing.assert_array_equal(actual, original)
    bf16 = ir.DataType.BFLOAT16.numpy()
    for result, source in (
        (initial[2], pre["scan_readout"]),
        (following[2], cached["scan_readout"][:, None]),
    ):
        assert result.dtype == bf16
        np.testing.assert_array_max_ulp(
            result.astype(np.float32), source.astype(np.float32), maxulp=65536
        )
    mask = np.asarray(metadata["mask_values"]) == 0
    assert not np.any(pre["x"].astype(np.float32)[mask])
    if np.any(mask):
        assert np.any(pre["B"].astype(np.float32)[mask])
        assert np.any(pre["C"].astype(np.float32)[mask])
    (directory / (name + "-result.json")).write_text(
        json.dumps(
            {
                "prefill": pre_metrics,
                "cached": cached_metrics,
                "prefill_fixture_sha256": metadata["npz_sha256"],
                "providers": providers,
                "cached_providers": cached_providers,
                "scope": "post-convolution SSM/readout/D boundary, head0 checkpoint slice; not whole block/decoder output parity",
            },
            indent=2,
        )
        + "\n"
    )


def test_cuda_cached_multitoken_is_rejected(worker):
    remote, directory = worker
    values, _ = _core._load_fixture("wrapper-B2-S129-ones-prefill")
    feeds = _feeds(values, False)
    feeds["has_previous_state"] = np.asarray(True)
    with pytest.raises(Exception, match=r"Reshape|shape"):
        _core._execute(
            remote, directory, _readout_graph(values), "cached-S129-rejected", feeds
        )


def _full_block_graph(baseline=False):
    cls = Float32SiLUMamba2Block if baseline else ChunkedPrefillMamba2Block
    block = cls(4, 4, 1, 4, 2)
    _cast_module_dtype(block, ir.DataType.BFLOAT16)
    random = np.random.default_rng(2107)
    for name, parameter in block.named_parameters():
        value = random.normal(0, 0.2, tuple(parameter.shape)).astype(parameter.dtype.numpy())
        if name == "A_log":
            value[:] = -1
        _parameter(parameter, value)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [2, "time", 4], ir.DataType.BFLOAT16)
    conv = create_test_input(builder, "conv", [2, 8, 3], ir.DataType.BFLOAT16)
    state = create_test_input(builder, "state", [2, 1, 2, 4])
    mask = create_test_input(builder, "mask", [2, "time", 1], ir.DataType.INT64)
    phase = create_test_input(builder, "has_previous_state", [], ir.DataType.BOOL)
    values = (
        block(op, hidden, conv, state, mask)
        if baseline
        else block(op, hidden, conv, state, mask, phase)
    )
    for value, name, shape, dtype in zip(
        values,
        ("output", "next_conv", "next_state"),
        ([2, "time", 4], [2, 8, 3], [2, 1, 2, 4]),
        (ir.DataType.BFLOAT16, ir.DataType.BFLOAT16, ir.DataType.FLOAT),
    ):
        value.name, value.shape, value.dtype = name, ir.Shape(shape), dtype
        graph.outputs.append(value)
    graph.opset_imports["com.microsoft"] = 1
    return ir.Model(graph, ir_version=11)


@pytest.mark.parametrize("length", [1, 129])
def test_cuda_entire_block_dirty_prefill_caches_and_cached_preservation(worker, length):
    remote, directory = worker
    model = _full_block_graph()
    bf16 = ir.DataType.BFLOAT16.numpy()
    feeds = dict(
        hidden=np.random.default_rng(48).normal(size=(2, length, 4)).astype(bf16),
        conv=np.full((2, 8, 3), 0.5, dtype=bf16),
        state=np.full((2, 1, 2, 4), 0.1234567, dtype=np.float32),
        mask=np.ones((2, length, 1), dtype=np.int64),
        has_previous_state=np.asarray(False),
    )
    dirty, _ = _core._execute(remote, directory, model, f"full-S{length}-dirty", feeds)
    clean, _ = _core._execute(
        remote,
        directory,
        model,
        f"full-S{length}-clean",
        {
            **feeds,
            "conv": np.zeros_like(feeds["conv"]),
            "state": np.zeros_like(feeds["state"]),
        },
    )
    for a, b in zip(dirty, clean):
        np.testing.assert_array_equal(a, b)
    next_feeds = {
        **feeds,
        "hidden": feeds["hidden"][:, :1],
        "mask": feeds["mask"][:, :1],
        "conv": dirty[1],
        "state": dirty[2],
        "has_previous_state": np.asarray(True),
    }
    cached, _ = _core._execute(remote, directory, model, f"full-S{length}-cached", next_feeds)
    existing, _ = _core._execute(
        remote, directory, _full_block_graph(True), f"full-S{length}-existing", next_feeds
    )
    for a, b in zip(cached, existing):
        np.testing.assert_array_equal(a, b)
    zero, _ = _core._execute(
        remote,
        directory,
        model,
        f"full-S{length}-cached-zero",
        {**next_feeds, "conv": np.zeros_like(dirty[1]), "state": np.zeros_like(dirty[2])},
    )
    assert np.any(cached[1] != zero[1])
    assert np.any(cached[2] != zero[2])


def _text_mask_graph():
    import dataclasses

    from mobius._configs import NemotronHPuzzleConfig
    from mobius._configs.per_model._nemotron_h_puzzle_test import _source_config
    from mobius.models import NemotronHPuzzleSourceChunkCausalLMModel

    config = dataclasses.replace(
        NemotronHPuzzleConfig.from_transformers(_source_config()),
        dtype=ir.DataType.BFLOAT16,
        num_hidden_layers=1,
        layer_types=["mamba2"],
        moe_block_configs=(None,),
    )
    module = NemotronHPuzzleSourceChunkCausalLMModel(config).model
    _cast_module_dtype(module, ir.DataType.BFLOAT16)
    random = np.random.default_rng(763)
    for name, parameter in module.named_parameters():
        value = random.normal(0, 0.2, tuple(parameter.shape)).astype(parameter.dtype.numpy())
        if name.endswith("A_log"):
            value[:] = -1
        _parameter(parameter, value)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [2, 1, 16], ir.DataType.BFLOAT16)
    mask = create_test_input(builder, "mask", [2, 1], ir.DataType.INT64)
    phase = create_test_input(builder, "has_previous_state", [], ir.DataType.BOOL)
    offset = create_test_input(builder, "cache_position", [1], ir.DataType.INT64)
    conv = create_test_input(builder, "conv", [2, 48, 3], ir.DataType.BFLOAT16)
    state = create_test_input(builder, "state", [2, 4, 4, 8])
    output, present = module(op, None, mask, offset, [(conv, state)], hidden, phase, offset)
    for value, name, shape, dtype in (
        (output, "output", [2, 1, 16], ir.DataType.BFLOAT16),
        (present[0][0], "next_conv", [2, 48, 3], ir.DataType.BFLOAT16),
        (present[0][1], "next_state", [2, 4, 4, 8], ir.DataType.FLOAT),
    ):
        value.name, value.shape, value.dtype = name, ir.Shape(shape), dtype
        graph.outputs.append(value)
    graph.opset_imports["com.microsoft"] = 1
    return ir.Model(graph, ir_version=11)


def test_cuda_source_mask_offset_is_independent_of_successful_cache_phase(worker):
    remote, directory = worker
    model = _text_mask_graph()
    bf16 = ir.DataType.BFLOAT16.numpy()
    base = dict(
        hidden=np.random.default_rng(32).normal(size=(2, 1, 16)).astype(bf16),
        conv=np.full((2, 48, 3), 0.25, dtype=bf16),
        state=np.full((2, 4, 4, 8), 0.1234567, dtype=np.float32),
    )
    results = {}
    for previous in (False, True):
        for offset in (0, 100):
            for mask in (0, 1):
                feeds = {
                    **base,
                    "mask": np.full((2, 1), mask, dtype=np.int64),
                    "has_previous_state": np.asarray(previous),
                    "cache_position": np.asarray([offset], dtype=np.int64),
                }
                results[previous, offset, mask], _ = _core._execute(
                    remote,
                    directory,
                    model,
                    f"mask-phase{previous}-offset{offset}-mask{mask}",
                    feeds,
                )
        # A positive source cache_position disables mask updates regardless of
        # the caller-owned phase; an offset-zero mask still applies in cached mode.
        for i in range(3):
            np.testing.assert_array_equal(
                results[previous, 100, 0][i], results[previous, 100, 1][i]
            )
            np.testing.assert_array_equal(
                results[previous, 100, 1][i], results[previous, 0, 1][i]
            )
        assert np.any(results[previous, 0, 0][1] != results[previous, 0, 1][1])
    # Fresh S1 with offset100 ignores dirty history. Cached S1 with offset0
    # consumes FP32 history even when x is masked to zero: no state reset.
    assert not np.any(results[False, 0, 0][2])
    assert np.any(results[True, 0, 0][2])
    clean, _ = _core._execute(
        remote,
        directory,
        model,
        "false-offset100-clean",
        {
            **base,
            "conv": np.zeros_like(base["conv"]),
            "state": np.zeros_like(base["state"]),
            "mask": np.ones((2, 1), dtype=np.int64),
            "has_previous_state": np.asarray(False),
            "cache_position": np.asarray([100], dtype=np.int64),
        },
    )
    for a, b in zip(clean, results[False, 100, 1]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "name",
    [
        "scan-B2-S129-init1-productionH128",
        "scan-B2-S257-init1-finite-clamp",
    ],
)
def test_cuda_production_prefill_primitive_integration_and_source_diagnostics(worker, name):
    remote, directory = worker
    values, metadata = _core._load_fixture(name)
    source_path = _adapter_fixture_path(
        name + "-source-native-prepared-initial-none.npz", directory
    )
    source_metadata = json.loads(
        _adapter_fixture_path(
            name + "-source-native-prepared-initial-none.json", directory
        ).read_text()
    )
    assert (
        hashlib.sha256(source_path.read_bytes()).hexdigest()
        == source_metadata["output_sha256"]
    )
    assert source_metadata["sealed_input_fixture_sha256"] == metadata["npz_sha256"]
    assert source_metadata["initial_states"] is None
    assert source_metadata["native_preparation_control"]["enabled"]
    preparation_path = _adapter_fixture_path(name + "-native-preparation.npz", directory)
    assert (
        hashlib.sha256(preparation_path.read_bytes()).hexdigest()
        == source_metadata["native_preparation_control"]["preparation_npz_sha256"]
    )
    with np.load(source_path) as archive:
        expected = {key: value.copy() for key, value in archive.items()}
    batch, length, heads = metadata["batch"], metadata["length"], metadata["heads"]
    p, n, groups = metadata["head_dim"], metadata["state_dim"], metadata["groups"]
    clamp = tuple(None if v == "Infinity" else v for v in metadata["dt_limit"])
    model = _readout_graph(
        values,
        heads=heads,
        groups=groups,
        head=p,
        state=n,
        clamp=clamp,
        observe_preparation=True,
    )
    feeds = dict(
        x=values["x"].reshape(batch, length, heads * p),
        b=values["B"].reshape(batch, length, groups * n),
        c=values["C"].reshape(batch, length, groups * n),
        dt=values["dt_raw"],
        state=values["initial_state"].transpose(0, 1, 3, 2),
        has_previous_state=np.asarray(False),
    )
    actual, providers = _core._execute(remote, directory, model, name + "-adapter", feeds)
    with np.load(preparation_path) as prepared:
        np.testing.assert_array_equal(actual[3], prepared["dt"])
        np.testing.assert_array_equal(actual[4], prepared["A"])
    source_dt = (
        values["dt_processed"].transpose(0, 2, 3, 1).reshape(batch, -1, heads)[:, :length]
    )
    assert actual[3].dtype == np.float32
    assert np.max(np.abs(actual[3] - source_dt)) < 2e-7
    readout = _metrics(actual[0], expected["no_D"].reshape(batch, length, heads * p))
    state = _metrics(actual[1], expected["final_state"].transpose(0, 1, 3, 2))
    stored = expected["output_bits"].view(ir.DataType.BFLOAT16.numpy())
    # Exact integration oracle: the accepted public primitive with precisely
    # the candidate's prepared operands. Source comparisons below deliberately
    # remain diagnostics, not weakened/relabelled source-parity gates.
    primitive, _ = _core._execute(
        remote,
        directory,
        _core._add_observers(_core._graph(batch, "time", heads, groups, p, n)),
        name + "-accepted-primitive-native-prepared",
        {
            "x": values["x"],
            "dt": actual[3],
            "a": actual[4],
            "b": values["B"],
            "c": values["C"],
            "initial_state": np.zeros((batch, heads, p, n), dtype=np.float32),
        },
    )
    np.testing.assert_array_equal(actual[0], primitive[0].reshape(batch, length, heads * p))
    np.testing.assert_array_equal(actual[1], primitive[1].transpose(0, 1, 3, 2))
    with np.load(
        _adapter_fixture_path(name + "-source-native-prepared-intermediates.npz", directory)
    ) as intermediates:
        source_decay = intermediates["dA"].transpose(2, 0, 1, 3)
        source_cb = intermediates["CB"].transpose(1, 0, 2, 3, 4)
        source_boundary_metrics = {
            "padded_cumsum": _metrics(primitive[2], source_decay),
            "CB_F32": _metrics(primitive[4], source_cb),
        }
        np.testing.assert_array_equal(primitive[2], source_decay)
        np.testing.assert_array_equal(primitive[4], source_cb)
    native_skip = (
        actual[0].reshape(batch, length, heads, p)
        + values["x"].astype(np.float32) * values["D"].astype(np.float32)[None, None, :, None]
    ).astype(ir.DataType.BFLOAT16.numpy())
    np.testing.assert_array_equal(actual[2], native_skip.reshape(batch, length, heads * p))
    clean, _ = _core._execute(
        remote,
        directory,
        model,
        name + "-adapter-clean",
        {**feeds, "state": np.zeros_like(feeds["state"])},
    )
    for a, b in zip(actual, clean):
        np.testing.assert_array_equal(a, b)
    with np.load(
        _adapter_fixture_path(name + "-source-model-initial-none.npz", directory)
    ) as raw:
        raw_readout = _metrics(actual[0], raw["no_D"].reshape(batch, length, heads * p))
        raw_state = _metrics(actual[1], raw["final_state"].transpose(0, 1, 3, 2))
    (directory / (name + "-adapter-result.json")).write_text(
        json.dumps(
            {
                "readout": readout,
                "state": state,
                "providers": providers,
                "conditioned_source_intermediates": source_boundary_metrics,
                "conditioned_source_readout_1e-5_gate": "PASS"
                if readout["nrmse"] < 1e-5
                else "FAIL",
                "source_BF16_stored_readout": _metrics(
                    actual[2], stored.reshape(batch, length, heads * p)
                ),
                "integration_gate": "BITWISE accepted public primitive on identical native-prepared operands; FP32 D*x + readout then sole native store",
                "source_zero_initial_sha256": source_metadata["output_sha256"],
                "source_fixture_root": str(source_path.parent),
                "source_fixture_manifest_sha256": _ADAPTER_FIXTURE_MANIFEST_SHA256,
                "raw_source_preparation_comparison": {
                    "readout": raw_readout,
                    "state": raw_state,
                    "existing_scan_1e-5_gate": "PASS"
                    if raw_readout["nrmse"] < 1e-5
                    else "FAIL",
                    "diagnosis": "Source versus native FP32 DT and arbitrary-A log/Exp reconstruction differ before scan. This control does not turn that raw-preparation failure into a model pass.",
                },
                "A_parameter_reconstruction": "FP32 log(-sealed_A) then native negative Exp; not bitwise equality to arbitrary sealed negative A",
                "limits": "numerical boundary candidate, not source-preparation, bitwise output/state or model qualification. Raw and conditioned source failures remain failures; no source gate/tolerance changed.",
            },
            indent=2,
        )
        + "\n"
    )
