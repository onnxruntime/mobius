# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Private reduced trained-weight GDN component proof, not full-model admission.

Collect with scripts/collect_drama3_gdn_fixture.py from already-authorized
pinned assets, then set MOBIUS_DRAMA3_GDN_FIXTURE to that private directory.
The checkpoint license is separate from Mobius; no weights are redistributed.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from _test_configs import FISH_OMNI_CONFIG
from onnxscript import nn

from mobius._configs.fish_omni import DRAMA3_REVISION, DRAMA3_SOURCE_REVISION, FishOmniConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.transformers._config_resolver import _dict_to_pretrained_config
from mobius.models.fish_omni import _FishSourceGatedDeltaNet, _FishTextModel
from mobius.tasks._base import _make_graph

pytestmark = pytest.mark.integration
LENGTHS = (1, 2, 30, 64, 65, 128, 129, 257)


def _assert_reference_close(actual, expected, dtype):
    tolerance = 1e-4 if dtype == ir.DataType.FLOAT else 1e-2
    assert actual.shape == expected.shape
    actual = actual.astype(np.float32)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    # Trained hidden-row slices have small amplitudes. Positive alignment
    # additionally rules out all-zero/sign-flipped outputs inside the absolute
    # gate. CPU native-half GEMM association is not a relative/bit-parity claim.
    assert np.vdot(actual, expected) > 0


@pytest.fixture(scope="module")
def trained_fixture():
    root = os.environ.get("MOBIUS_DRAMA3_GDN_FIXTURE")
    if not root:
        pytest.skip(
            "Set MOBIUS_DRAMA3_GDN_FIXTURE to the private pinned trained-weight fixture"
        )
    path = Path(root)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["checkpoint"] == DRAMA3_REVISION
    assert manifest["source"] == DRAMA3_SOURCE_REVISION
    assert (
        manifest["layers_sha256"]
        == "e4704b4fde7906bf021d6696c4132458b4bc52c836d22da702275252683540bc"
    )
    assert manifest["torch"].startswith("2.8.")
    assert manifest["device"] == "cpu"
    assert (
        hashlib.sha256((path / "weights.npz").read_bytes()).hexdigest()
        == manifest["weights_file_sha256"]
    )
    cases = {case["file"]: case for case in manifest["cases"]}
    assert len(cases) == 96
    for name, case in cases.items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == case["sha256"]
    return path, manifest


def _trained_module(root, dtype):
    config = FishOmniConfig.from_transformers(
        _dict_to_pretrained_config(copy.deepcopy(FISH_OMNI_CONFIG))
    )
    config = dataclasses.replace(
        config,
        dtype=dtype,
        hidden_size=32,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    module = _FishSourceGatedDeltaNet(config)
    with np.load(root / "weights.npz") as weights:
        for name, parameter in list(module.named_parameters()):
            array = weights[name].astype(dtype.numpy())
            owner = module
            segments = name.split(".")
            for segment in segments[:-1]:
                owner = getattr(owner, segment)
            setattr(
                owner, segments[-1], nn.Parameter(list(parameter.shape), data=ir.tensor(array))
            )
    return module


@pytest.fixture(scope="module", params=[ir.DataType.FLOAT, ir.DataType.FLOAT16])
def trained_session(request, trained_fixture):
    root, _ = trained_fixture
    dtype = request.param
    module = _trained_module(root, dtype)
    graph, builder = _make_graph("trained_source_gdn")
    x = builder.input("x", dtype, ["batch", "time", 32])
    conv = builder.input("conv", dtype, ["batch", 512, 3])
    state = builder.input("state", ir.DataType.FLOAT, ["batch", 2, 128, 128])
    continuing = builder.input("continuing", ir.DataType.BOOL, [])
    output, present_conv, present_state = module(builder.op, x, conv, state, continuing)
    for name, value, value_dtype, shape in (
        ("output", output, dtype, ["batch", "time", 32]),
        ("present_conv", present_conv, dtype, ["batch", 512, 3]),
        ("present_state", present_state, ir.DataType.FLOAT, ["batch", 2, 128, 128]),
    ):
        value.type, value.shape = ir.TensorType(value_dtype), ir.Shape(shape)
        builder.add_output(value, name)
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    yield session, dtype
    session.close()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("continuing", [False, True])
def test_trained_source_gdn_initial_reset_cached_multichunk(
    trained_fixture, trained_session, batch, length, continuing
):
    root, _ = trained_fixture
    session, dtype = trained_session
    label = "f32" if dtype == ir.DataType.FLOAT else "f16"
    with np.load(root / f"{label}-b{batch}-t{length}-c{int(continuing)}.npz") as reference:
        feeds = {
            "x": reference["x"].astype(dtype.numpy()),
            "conv": reference["conv"].astype(dtype.numpy()),
            "state": reference["state"],
            "continuing": np.array(continuing),
        }
        if not continuing:
            np.testing.assert_array_equal(feeds["conv"], np.full_like(feeds["conv"], 0.375))
            np.testing.assert_array_equal(feeds["state"], np.full_like(feeds["state"], -0.125))
        else:
            assert np.count_nonzero(feeds["conv"]) > 0
            assert np.count_nonzero(feeds["state"]) > 0
        actual = session.run(feeds)
        for name in ("output", "present_conv", "present_state"):
            _assert_reference_close(actual[name], reference[name], dtype)
            if batch == 2:
                assert not np.array_equal(actual[name][0], actual[name][1])
        if not continuing:
            zeroed = session.run(
                {
                    **feeds,
                    "conv": np.zeros_like(feeds["conv"]),
                    "state": np.zeros_like(feeds["state"]),
                }
            )
            for name in actual:
                np.testing.assert_array_equal(actual[name], zeroed[name])
            if length == 1:
                stale_nan = session.run(
                    {
                        **feeds,
                        "conv": np.full_like(feeds["conv"], np.nan),
                        "state": np.full_like(feeds["state"], np.nan),
                    }
                )
                for name in actual:
                    np.testing.assert_array_equal(actual[name], stale_nan[name])


def test_trained_returned_carries_feed_next_single_and_multitoken_calls(
    trained_fixture, trained_session
):
    root, _ = trained_fixture
    session, dtype = trained_session
    label = "f32" if dtype == ir.DataType.FLOAT else "f16"
    # The source collector independently generated continuation from precisely
    # this trained-input T30 prefill. Reuse ONNX carries rather than source carries.
    for batch in (1, 2):
        with np.load(root / f"{label}-b{batch}-t30-c0.npz") as initial:
            prefill = session.run(
                {
                    "x": initial["x"].astype(dtype.numpy()),
                    "conv": initial["conv"].astype(dtype.numpy()),
                    "state": initial["state"],
                    "continuing": np.array(False),
                }
            )
        for length in (1, 65, 129, 257):
            with np.load(root / f"{label}-b{batch}-t{length}-c1.npz") as expected:
                actual = session.run(
                    {
                        "x": expected["x"].astype(dtype.numpy()),
                        "conv": prefill["present_conv"],
                        "state": prefill["present_state"],
                        "continuing": np.array(True),
                    }
                )
                for name in actual:
                    _assert_reference_close(actual[name], expected[name], dtype)


def test_text_position_wiring_drives_real_trained_gdn_numerically(trained_fixture):
    root, _ = trained_fixture

    class Identity(nn.Module):
        def forward(self, op, x):
            return op.Identity(x)

    class GDNLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = _trained_module(root, ir.DataType.FLOAT16)

        def forward(self, op, hidden, bias, positions, cache, continue_state):
            output, conv, state = self.linear_attn(op, hidden, *cache, continue_state)
            return output, (conv, state)

    class ContextHarness(_FishTextModel):
        """Only the production text context plus a real GDN; not full decoder parity."""

        def __init__(self):
            nn.Module.__init__(self)
            self._dtype = ir.DataType.FLOAT16
            self.layers = nn.ModuleList([GDNLayer()])
            self.norm = Identity()
            self.rotary_emb = Identity()

    graph, builder = _make_graph("trained_position_context")
    x = builder.input("x", ir.DataType.FLOAT16, ["batch", "time", 32])
    conv = builder.input("conv", ir.DataType.FLOAT16, ["batch", 512, 3])
    state = builder.input("state", ir.DataType.FLOAT, ["batch", 2, 128, 128])
    positions = builder.input("positions", ir.DataType.INT64, ["batch", "time"])
    mask = builder.input("mask", ir.DataType.INT64, ["batch", "total"])
    output, present = ContextHarness()(
        builder.op, None, mask, positions, [(conv, state)], inputs_embeds=x
    )
    for name, value, dtype, shape in (
        ("output", output, ir.DataType.FLOAT16, ["batch", "time", 32]),
        ("present_conv", present[0][0], ir.DataType.FLOAT16, ["batch", 512, 3]),
        ("present_state", present[0][1], ir.DataType.FLOAT, ["batch", 2, 128, 128]),
    ):
        value.type, value.shape = ir.TensorType(dtype), ir.Shape(shape)
        builder.add_output(value, name)
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        for batch in (1, 2):
            for length in (1, 2, 65, 129, 257):
                for continuing in (False, True):
                    start = 30 if continuing else 0
                    with np.load(
                        root / f"f16-b{batch}-t{length}-c{int(continuing)}.npz"
                    ) as expected:
                        actual = session.run(
                            {
                                "x": expected["x"].astype(np.float16),
                                "conv": expected["conv"].astype(np.float16),
                                "state": expected["state"],
                                "positions": np.broadcast_to(
                                    np.arange(start, start + length, dtype=np.int64),
                                    (batch, length),
                                ).copy(),
                                "mask": np.ones((batch, start + length), np.int64),
                            }
                        )
                        for name in actual:
                            _assert_reference_close(
                                actual[name], expected[name], ir.DataType.FLOAT16
                            )
    finally:
        session.close()
