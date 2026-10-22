# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
import json
import sys

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper
from ornith_export import REVISION
from ornith_int2_overlay import main, replace_dense_int2
from safetensors.numpy import save_file


def model_and_weights(packed):
    names = ["decoder.model.layers.0.q.weight"]
    if packed:
        names.append("decoder.model.layers.0.k.weight")
    name = "__".join(names) + ("__axis_0__concat" if packed else "")
    scales_name = "__".join(n.replace(".weight", ".scales") for n in names) + (
        "__axis_0__concat" if packed else ""
    )
    node = helper.make_node(
        "MatMulNBits",
        ["x", name, scales_name],
        ["y"],
        domain="com.microsoft",
        bits=2,
        block_size=128,
        K=128,
        N=2 * len(names),
    )
    tensors = [
        numpy_helper.from_array(np.zeros((2 * len(names), 1, 32), np.uint8), name),
        numpy_helper.from_array(np.ones((2 * len(names), 1), np.float16), scales_name),
    ]
    model = helper.make_model(helper.make_graph([node], "test", [], [], tensors))
    weights = {
        key: (
            np.full((2, 1, 32), i + 7, np.uint8),
            np.full((2, 1), i + 2, np.float16),
        )
        for i, key in enumerate(names)
    }
    return model, weights


@pytest.mark.parametrize("packed", [False, True])
def test_dense_overlay_preserves_graph_and_projection_order(packed):
    model, weights = model_and_weights(packed)
    before = model.SerializeToString()
    result, changed = replace_dense_int2(model, weights)
    assert model.SerializeToString() == before
    assert result.graph.node == model.graph.node
    assert len(changed) == 2
    np.testing.assert_array_equal(
        numpy_helper.to_array(result.graph.initializer[0]),
        np.concatenate([pair[0] for pair in weights.values()]),
    )


@pytest.mark.parametrize("problem", ["missing", "extra", "dtype", "shape", "nan", "order"])
def test_dense_overlay_rejects_incompatible_calibration(problem):
    model, weights = model_and_weights(True)
    first = next(iter(weights))
    if problem == "missing":
        del weights[first]
    elif problem == "extra":
        weights["unknown"] = weights[first]
    elif problem == "dtype":
        weights[first] = (weights[first][0].astype(np.int32), weights[first][1])
    elif problem == "shape":
        weights[first] = (weights[first][0][:1], weights[first][1])
    elif problem == "nan":
        weights[first][1][:] = np.nan
    else:
        model.graph.node[0].input[2] = model.graph.node[0].input[1]
    with pytest.raises(ValueError):
        replace_dense_int2(model, weights)


def test_dense_overlay_accepts_fixed_layout_swiglu():
    model, weights = model_and_weights(True)
    node = model.graph.node[0]
    node.op_type = "MatMulNBitsSwiGLU"
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("K", 128), helper.make_attribute("N", 4)])
    _, changed = replace_dense_int2(model, weights)
    assert len(changed) == 2


def test_dense_overlay_cli_reuses_external_data_and_checks_manifest(tmp_path, monkeypatch):
    model, weights = model_and_weights(False)
    untouched = numpy_helper.from_array(np.array([3, 4], np.float16), "unchanged")
    onnx.external_data_helper.set_external_data(untouched, location="original.data")
    model.graph.initializer.append(untouched)
    source = tmp_path / "source.onnx"
    onnx.save_model(model, source)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    name, (codes, scales) = next(iter(weights.items()))
    shard = calibration / "weights.safetensors"
    save_file({"codes": codes.reshape(2, 32), "scales": scales}, shard)
    manifest = {
        "revision": REVISION,
        "bits": 2,
        "group_size": 128,
        "zero_point": 2,
        "method": "dense-gptq",
        "source_graph_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shards": {
            shard.name: {
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "weights": {name: {"codes": "codes", "scales": "scales"}},
            }
        },
    }
    manifest_path = calibration / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "experimental.onnx"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ornith_int2_overlay",
            "--model",
            str(source),
            "--calibration",
            str(calibration),
            "--output",
            str(output),
        ],
    )
    original_graph = onnx.load(source, load_external_data=False)
    main()
    result = onnx.load(output, load_external_data=False)
    assert result.graph.initializer[-1] == original_graph.graph.initializer[-1]
    loaded = onnx.load(output)
    np.testing.assert_array_equal(numpy_helper.to_array(loaded.graph.initializer[0]), codes)
    np.testing.assert_array_equal(
        numpy_helper.to_array(loaded.graph.initializer[-1]), np.array([3, 4], np.float16)
    )
    assert json.loads(output.with_suffix(".json").read_text())["status"] == (
        "exported_not_validated"
    )
    with pytest.raises(FileExistsError):
        main()
    other = tmp_path / "other.onnx"
    sys.argv[-1] = str(other)
    manifest["source_graph_sha256"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source graph"):
        main()
    assert not other.exists()
