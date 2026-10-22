#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Apply experimental calibrated dense INT2 weights without changing the graph ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import external_data_helper, numpy_helper
from ornith_export import REVISION
from safetensors import safe_open


def replace_dense_int2(model, weights):
    """Validate and replace only the dense projection weights/scales in a copy."""
    initializers = {value.name: value for value in model.graph.initializer}
    replacements, consumed = {}, set()
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type not in (
            "MatMulNBits",
            "MatMulNBitsSwiGLU",
        ):
            continue
        attributes = {
            attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute
        }
        compatible = (
            attributes.get("bits") == 2 and attributes.get("block_size") == 128
            if node.op_type == "MatMulNBits"
            else set(attributes) == {"K", "N"}
        )
        if not compatible:
            raise ValueError(f"Unsupported dense INT2 layout at {node.name}")
        if len(node.input) != 3 or any(name not in initializers for name in node.input[1:]):
            raise ValueError(f"Expected constant dense INT2 weights at {node.name}")
        name = node.input[1]
        parts = (
            name.removesuffix("__axis_0__concat").split("__")
            if name.endswith("__axis_0__concat")
            else [name]
        )
        if any(part not in weights for part in parts):
            raise ValueError(f"Missing calibrated dense projection for {name}")
        expected_scale_name = (
            "__".join(part.removesuffix(".weight") + ".scales" for part in parts)
            + "__axis_0__concat"
            if len(parts) > 1
            else parts[0].removesuffix(".weight") + ".scales"
        )
        if node.input[2] != expected_scale_name:
            raise ValueError(f"Packed projection order does not match its scales: {name}")
        codes = np.concatenate([weights[part][0] for part in parts], axis=0)
        scales = np.concatenate([weights[part][1] for part in parts], axis=0)
        for key, array, dtype in (
            (name, codes, np.uint8),
            (node.input[2], scales, np.float16),
        ):
            original = initializers[key]
            if (
                array.dtype != dtype
                or array.shape != tuple(original.dims)
                or numpy_helper.from_array(array).data_type != original.data_type
            ):
                raise ValueError(f"Calibrated tensor shape/dtype mismatch for {key}")
            replacements[key] = numpy_helper.from_array(array, key)
        if not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError(f"Invalid calibrated FP16 scales for {name}")
        consumed.update(parts)
    if not consumed or consumed != set(weights):
        raise ValueError("Calibrated dense projection coverage does not match the model")
    result = onnx.ModelProto()
    result.CopyFrom(model)
    for value in result.graph.initializer:
        if value.name in replacements:
            value.CopyFrom(replacements[value.name])
    return result, sorted(replacements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.model.resolve(), args.output.resolve()
    if source.parent != output.parent:
        raise ValueError("Output must share the source directory for external-weight reuse")
    data = output.with_name(output.name + ".data")
    provenance = output.with_suffix(".json")
    if any(path.exists() for path in (output, data, provenance)):
        raise FileExistsError("Refusing to overwrite experimental graph/data/provenance")
    manifest = json.loads((args.calibration / "manifest.json").read_text())
    if (
        manifest["revision"] != REVISION
        or manifest["bits"] != 2
        or manifest["group_size"] != 128
        or manifest["zero_point"] != 2
        or manifest["method"] != "dense-gptq"
    ):
        raise ValueError("Calibration manifest is incompatible with the Ornith INT2 ABI")
    with source.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if source_sha256 != manifest["source_graph_sha256"]:
        raise ValueError("Calibration manifest does not match the source graph")
    weights = {}
    for filename, entries in manifest["shards"].items():
        shard = (args.calibration / filename).resolve()
        if shard.parent != args.calibration.resolve():
            raise ValueError("Calibration shards must be direct children of the directory")
        with shard.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entries["sha256"]:
            raise ValueError(f"Calibration shard digest mismatch: {filename}")
        with safe_open(shard, framework="numpy") as handle:
            for name, keys in entries["weights"].items():
                if name in weights:
                    raise ValueError(f"Duplicate calibrated weight: {name}")
                codes = handle.get_tensor(keys["codes"])
                scales = handle.get_tensor(keys["scales"])
                if codes.ndim != 2 or scales.ndim != 2:
                    raise ValueError(f"Expected two-dimensional calibrated tensors: {name}")
                weights[name] = (
                    codes.reshape(codes.shape[0], -1, 32),
                    scales,
                )
    model = onnx.load(source, load_external_data=False)
    result, changed = replace_dense_int2(model, weights)
    for value in result.graph.initializer:
        if value.name in changed:
            external_data_helper.set_external_data(value, location=data.name)
    onnx.save_model(result, output)
    provenance.write_text(
        json.dumps(
            {
                "status": "exported_not_validated",
                "accuracy_test": False,
                "source": str(source),
                "calibration": manifest,
                "changed_initializers": changed,
                "unchanged_graph_nodes": True,
                "experts_lmhead_embedding_and_runtime_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {output}; runtime validation and benchmark still required")


if __name__ == "__main__":
    main()
