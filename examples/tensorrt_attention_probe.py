"""Build a separate TensorRT engine exposing Qwen3 layer-zero attention tensors."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx_ir as ir


def add_explicit_mask(model):
    attentions = [node for node in model.graph if node.op_type == "Attention"]
    inputs = {value.name: value for value in model.graph.inputs}
    capacity = inputs["key_cache.0"].shape[2]
    if not isinstance(capacity, int):
        raise TypeError("Explicit-mask experiment requires a fixed cache capacity")
    if any(
        node.inputs[1].shape[2] != capacity
        or node.inputs[3] is not None
        or node.inputs[0].dtype != ir.DataType.BFLOAT16
        for node in attentions
    ):
        raise ValueError("Expected uniform BF16 static-cache attention without existing masks")
    nodes = []

    def constant(name, array):
        value = ir.Value(name=f"experiment.{name}", const_value=ir.tensor(array))
        model.graph.initializers[value.name] = value
        return value

    def operation(name, op_type, arguments, dtype, shape):
        output = ir.Value(
            name=f"experiment.{name}", type=ir.TensorType(dtype), shape=ir.Shape(shape)
        )
        nodes.append(ir.Node("", op_type, inputs=arguments, outputs=[output]))
        return output

    key_positions = constant(
        "key_positions", np.arange(capacity, dtype=np.int64)[None, None, None, :]
    )
    query_axes = constant("query_axes", np.asarray([1, 3], dtype=np.int64))
    length_axes = constant("length_axes", np.asarray([1, 2, 3], dtype=np.int64))
    batch, sequence = inputs["position_ids"].shape
    query_positions = operation(
        "query_positions",
        "Unsqueeze",
        [inputs["position_ids"], query_axes],
        ir.DataType.INT64,
        [batch, 1, sequence, 1],
    )
    valid_length = operation(
        "valid_length",
        "Unsqueeze",
        [inputs["nonpad_kv_seqlen"], length_axes],
        ir.DataType.INT64,
        [batch, 1, 1, 1],
    )
    causal = operation(
        "causal",
        "LessOrEqual",
        [key_positions, query_positions],
        ir.DataType.BOOL,
        [batch, 1, sequence, capacity],
    )
    valid = operation(
        "valid",
        "Less",
        [key_positions, valid_length],
        ir.DataType.BOOL,
        [batch, 1, 1, capacity],
    )
    allowed = operation(
        "allowed", "And", [causal, valid], ir.DataType.BOOL, [batch, 1, sequence, capacity]
    )
    zero = constant("zero", np.asarray(0, dtype=ml_dtypes.bfloat16))
    negative_inf = constant("negative_inf", np.asarray(-np.inf, dtype=ml_dtypes.bfloat16))
    mask = operation(
        "mask",
        "Where",
        [allowed, zero, negative_inf],
        ir.DataType.BFLOAT16,
        [batch, 1, sequence, capacity],
    )
    model.graph.insert_before(attentions[0], nodes)
    for node in attentions:
        node.replace_input_with(3, mask)
        node.replace_input_with(6, None)
        node.attributes["is_causal"] = ir.AttrInt64("is_causal", 0)
    print(f"Applied explicit position/valid-length mask to {len(attentions)} layers.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--explicit-mask",
        action="store_true",
        help="Experiment with explicit causal/valid-length bias in every layer",
    )
    parser.add_argument(
        "--trtexec", type=Path, default=Path("C:/TensorRT-11.3.0.99/bin/trtexec.exe")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = ir.load(args.onnx)
    if args.explicit_mask:
        add_explicit_mask(model)
    attention = next(node for node in model.graph if node.op_type == "Attention")
    projection = next(
        node
        for node in model.graph
        if node.op_type == "MatMul" and "layers.0.self_attn.o_proj" in node.outputs[0].name
    )
    for label, value in (("query", attention.inputs[0]), ("attention", projection.inputs[0])):
        output = ir.Value(name=f"probe.{label}", shape=value.shape, type=value.type)
        model.graph.append(ir.Node("", "Identity", inputs=[value], outputs=[output]))
        model.graph.outputs.append(output)
    print("Attention attributes:", attention.attributes, flush=True)
    destination = args.output / "model.onnx"
    ir.save(model, destination, external_data="model.onnx.data")
    command = [
        str(args.trtexec),
        f"--onnx={destination}",
        f"--saveEngine={args.output / 'model.engine'}",
        "--skipInference",
        "--decomposableAttentions=*",
        "--profilingVerbosity=detailed",
    ]
    for option, length in (("minShapes", 1), ("optShapes", 32), ("maxShapes", 128)):
        shapes = []
        for value in model.graph.inputs:
            shape = [dim if isinstance(dim, int) else 1 for dim in value.shape]
            if value.name in ("input_ids", "position_ids"):
                shape = [1, length]
            shapes.append(f"{value.name}:{'x'.join(map(str, shape))}")
        command.append(f"--{option}={','.join(shapes)}")
    print("Building diagnostic engine; original engine is unchanged.", flush=True)
    with (args.output / "build.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(
        "\n".join((args.output / "build.log").read_text(encoding="utf-8").splitlines()[-20:])
    )
    if result.returncode:
        raise RuntimeError(f"TensorRT build failed; see {args.output / 'build.log'}")


if __name__ == "__main__":
    main()
