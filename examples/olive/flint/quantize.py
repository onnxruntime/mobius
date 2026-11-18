# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Quantize the routed expert banks of a complete CUDA BF16 Flint export with Olive."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import onnx_ir as ir
import onnxscript.optimizer
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.moe_quantization import OnnxMoEQuantization

REVISION = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"


def _materialize_owned(model: ir.Model, source: Path) -> None:
    """Read linked payloads through bounded, private copies without disabling IR checks."""
    groups = collections.defaultdict(list)
    for value in model.graph.initializers.values():
        tensor = value.const_value
        if isinstance(tensor, ir.ExternalTensor):
            location = Path(tensor.location)
            if location.is_absolute() or ".." in location.parts:
                raise ValueError(f"Expected package-local external data: {location}")
            path = source / location
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Expected a regular external data file: {path}")
            groups[path].append(value)
    with tempfile.TemporaryDirectory(
        prefix="flint-weight-stage-", dir=source.parent
    ) as temporary:
        staged = Path(temporary) / "payload.data"
        for index, (path, values) in enumerate(groups.items(), 1):
            linked = path.stat().st_nlink > 1
            if linked:
                shutil.copyfile(path, staged)
            for value in values:
                tensor = value.const_value
                owned = (
                    ir.ExternalTensor(
                        staged.name,
                        tensor.offset,
                        tensor.length,
                        tensor.dtype,
                        shape=tensor.shape,
                        name=value.name,
                        base_dir=staged.parent,
                    )
                    if linked
                    else tensor
                )
                value.const_value = ir.tensor(owned.numpy().copy())
                owned.release()
            if linked:
                staged.unlink()
            print(f"Loaded payload {index}/{len(groups)}", flush=True)


def _batch_routes(model: ir.Model) -> int:
    """Batch independent singleton routes, retaining their external FP32 weighting."""
    groups = collections.defaultdict(list)
    for node in model.graph:
        if node.domain == "com.microsoft" and node.op_type == "MoE":
            groups[(node.inputs[0], node.inputs[2], node.inputs[4])].append(node)
    for nodes in groups.values():
        if len(nodes) != 6:
            raise ValueError("Expected six singleton routes sharing each Flint expert bank.")
        first = nodes[0]
        attributes = {key: (attr.type, attr.value) for key, attr in first.attributes.items()}
        for node in nodes:
            if (
                node.attributes.get_int("k") != 1
                or node.attributes.get_int("normalize_routing_weights") != 1
                or node.attributes.get_int("swiglu_fusion") != 1
                or node.attributes.get_string("activation_type") != "swiglu"
            ):
                raise ValueError(
                    "The recipe requires externally weighted singleton SwiGLU routing."
                )
            if (
                node.inputs[2:] != first.inputs[2:]
                or {key: (attr.type, attr.value) for key, attr in node.attributes.items()}
                != attributes
            ):
                raise ValueError("Batched routes must share all expert inputs and attributes.")
        hidden = ir.node(
            "Concat",
            inputs=[n.inputs[0] for n in nodes],
            attributes={"axis": 0},
            name=f"{first.name}_batch_hidden",
        )
        routing = ir.node(
            "Concat",
            inputs=[n.inputs[1] for n in nodes],
            attributes={"axis": 0},
            name=f"{first.name}_batch_routing",
        )
        batched = ir.node(
            "MoE",
            domain="com.microsoft",
            inputs=[hidden.outputs[0], routing.outputs[0], *first.inputs[2:]],
            attributes=dict(first.attributes),
            name=f"{first.name}_batched",
        )
        split = ir.node(
            "Split",
            inputs=[batched.outputs[0]],
            num_outputs=len(nodes),
            attributes={"axis": 0, "num_outputs": len(nodes)},
            name=f"{first.name}_split_routes",
        )
        for node, output in zip(nodes, split.outputs):
            old = node.outputs[0]
            output.name, output.type, output.shape = old.name, old.type, old.shape
        ir.convenience.replace_nodes_and_values(
            model.graph,
            first,
            nodes,
            [hidden, routing, batched, split],
            [n.outputs[0] for n in nodes],
            split.outputs,
        )
    model.graph.sort()
    return len(groups)


class _PreparedFlint(ONNXModelHandler):
    def load_ir_model(self):
        model = ir.load(self.model_path)
        _materialize_owned(model, Path(self.model_path).parent)
        # The 1-GiB FC1 bank is [gate|up]; fold its interleaving before quantization.
        onnxscript.optimizer.fold_constants(
            model,
            input_size_limit=2**31,
            output_size_limit=2**31,
            should_fold=lambda n: (
                n.domain == "" and n.op_type in ("Constant", "Reshape", "Transpose")
            ),
        )
        count = _batch_routes(model)
        if count != 47:
            raise ValueError(f"Expected all 47 MoE layers, found {count}.")
        for node in model.graph:
            if node.op_type == "MoE" and any(
                node.inputs[i].const_value is None for i in (2, 4)
            ):
                raise ValueError("Expert interleaving did not fold to constant banks.")
        return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-model", type=Path, required=True, help="Complete BF16 CUDA package."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="A new, empty directory."
    )
    parser.add_argument("--recipe", type=Path, default=Path(__file__).with_name("int4.json"))
    args = parser.parse_args()
    source, output = args.input_model.resolve(), args.output_dir.resolve()
    for name in (
        "model.onnx",
        "weight-loading-report.json",
        "genai_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "lib/libonnxruntime_source_compat_cuda.so",
    ):
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing complete CUDA package prerequisite: {name}")
    report = json.loads((source / "weight-loading-report.json").read_text())
    if report["revision"] != REVISION or report["accounted_source_tensor_count"] != 36734:
        raise ValueError("The recipe requires the fully accounted pinned Flint checkpoint.")
    recipe = json.loads(args.recipe.read_text())
    if recipe["type"] != "OnnxMoEQuantization":
        raise ValueError("Expected the expert-bank Olive quantization pass.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    quantizer = create_pass_from_dict(
        OnnxMoEQuantization, recipe["config"], disable_search=True
    )
    result = quantizer.run(
        _PreparedFlint(model_path=source / "model.onnx"), str(output / "model.onnx")
    )
    model = result.load_ir_model()
    histogram = collections.Counter(n.op_type for n in model.graph)
    if histogram["QMoE"] != 47 or histogram["MoE"]:
        raise ValueError(f"Incomplete expert quantization: {dict(histogram)}")
    banks = {n.inputs[i] for n in model.graph if n.op_type == "QMoE" for i in (2, 5)}
    if len(banks) != 94 or any(v.dtype != ir.DataType.UINT8 for v in banks):
        raise ValueError("Expected exactly 94 packed, nonduplicated expert banks.")
    for name in (
        "genai_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    shutil.copytree(source / "lib", output / "lib")
    shutil.copy2(
        source / "weight-loading-report.json", output / "source-weight-loading-report.json"
    )
    receipt = {
        "target": f"RecursiveSI/flint-routed-int{recipe['config']['bits']}",
        "revision": REVISION,
        "source_header_sha256": hashlib.sha256(
            (source / "model.onnx").read_bytes()
        ).hexdigest(),
        "recipe": recipe,
        "expert_banks": len(banks),
        "qmoe_nodes": histogram["QMoE"],
        "initializer_bytes": sum(
            v.const_value.nbytes for v in model.graph.initializers.values()
        ),
        "package_file_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "quantization_seconds": time.perf_counter() - started,
        "status": "converted; runtime execution and quality must be validated separately",
        "limitations": "Experimental source model; publisher BF16 numerical qualification remains unresolved.",
    }
    (output / "quantization-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
