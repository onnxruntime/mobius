# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Lossless GGUF external weights and standard-ONNX weight-layout expressions.

Only metadata-sized torch tensors are used to run the existing name/layout
processors. Their operations become ONNX nodes; source payloads are never
decoded, repacked or copied. Source files are read for integrity hashing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import TypedDict

import numpy as np
import onnx_ir as ir
import torch
from torch.utils._python_dispatch import TorchDispatchMode

_TYPES = {
    torch.float32: ir.DataType.FLOAT,
    torch.float16: ir.DataType.FLOAT16,
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.float64: ir.DataType.DOUBLE,
    torch.int64: ir.DataType.INT64,
    torch.int32: ir.DataType.INT32,
    torch.int16: ir.DataType.INT16,
    torch.int8: ir.DataType.INT8,
    torch.uint8: ir.DataType.UINT8,
    torch.bool: ir.DataType.BOOL,
}
_FLOAT_TYPES = {
    "F32": ir.DataType.FLOAT,
    "F16": ir.DataType.FLOAT16,
    "BF16": ir.DataType.BFLOAT16,
    "F64": ir.DataType.DOUBLE,
    "I8": ir.DataType.INT8,
    "I16": ir.DataType.INT16,
    "I32": ir.DataType.INT32,
    "I64": ir.DataType.INT64,
}
_MANIFEST = "gguf-packed.json"
# DequantizePacked-29's fixed GGUF layouts: (elements per block, bytes per block).
_PACKED_FORMATS = {
    "gguf:q4_0": (32, 18),
    "gguf:q4_1": (32, 20),
    "gguf:q5_0": (32, 22),
    "gguf:q5_1": (32, 24),
    "gguf:q8_0": (32, 34),
    "gguf:q8_1": (32, 36),
    "gguf:q2_k": (256, 84),
    "gguf:q3_k": (256, 110),
    "gguf:q4_k": (256, 144),
    "gguf:q5_k": (256, 176),
    "gguf:q6_k": (256, 210),
    "gguf:q8_k": (256, 292),
    "gguf:iq2_xxs": (256, 66),
    "gguf:iq2_xs": (256, 74),
    "gguf:iq3_xxs": (256, 98),
    "gguf:iq1_s": (256, 50),
    "gguf:iq4_nl": (32, 18),
    "gguf:iq3_s": (256, 110),
    "gguf:iq2_s": (256, 82),
    "gguf:iq4_xs": (256, 136),
    "gguf:iq1_m": (256, 56),
    "gguf:tq1_0": (256, 54),
    "gguf:tq2_0": (256, 66),
    "gguf:mxfp4": (32, 17),
    "gguf:nvfp4": (64, 36),
    "gguf:q1_0": (128, 18),
    "gguf:q2_0": (64, 18),
}


def use_original_gguf_norms(module, architecture: str) -> bool:
    """Consume already-offset GGUF scales without a rounded subtract/add pair."""
    from mobius.components._rms_norm import OffsetRMSNorm
    from mobius.integrations.gguf._builder import (
        _OFFSET_NORM_GGUF_ARCHS,
        _replace_child_module,
    )

    if architecture not in _OFFSET_NORM_GGUF_ARCHS:
        return False

    class EffectiveRMSNorm(OffsetRMSNorm):
        def forward(self, op, hidden_states):
            hidden = op.Cast(hidden_states, to=ir.DataType.FLOAT)
            weight = op.Cast(self.weight, to=ir.DataType.FLOAT)
            result = op.RMSNormalization(
                hidden, weight, epsilon=self.variance_epsilon, axis=-1
            )
            return op.CastLike(result, hidden_states)

    for name, child in list(module.named_modules()):
        if type(child) is OffsetRMSNorm:
            _replace_child_module(
                module,
                name,
                EffectiveRMSNorm(int(child.weight.shape[0]), child.variance_epsilon),
            )
    return True


def use_original_gguf_decay(module, config) -> frozenset[str]:
    """Replace proven DeltaNet classes with direct negative-decay consumers."""
    from mobius.components import GatedDeltaNet, OriginalDecayGatedDeltaNet
    from mobius.integrations.gguf._builder import _replace_child_module
    from mobius.models.qwen4_exp import Qwen4ExpGatedDeltaNet
    from mobius.models.qwen35 import Qwen35GatedDeltaNet

    class OriginalDecayQwen35(OriginalDecayGatedDeltaNet, Qwen35GatedDeltaNet):
        """Retain the source model's per-head Q/K normalization epsilon."""

    class OriginalDecayQwen4Exp(OriginalDecayGatedDeltaNet, Qwen4ExpGatedDeltaNet):
        """Retain the source model's convolution and output-gate normalization."""

    replacements = {
        GatedDeltaNet: OriginalDecayGatedDeltaNet,
        Qwen35GatedDeltaNet: OriginalDecayQwen35,
        Qwen4ExpGatedDeltaNet: OriginalDecayQwen4Exp,
    }
    paths = set()
    for name, child in list(module.named_modules()):
        replacement = replacements.get(type(child))
        if replacement is not None:
            _replace_child_module(module, name, replacement(config))
            paths.add(name)
    return frozenset(paths)


class _PackedSource(TypedDict):
    path: str
    size: int
    sha256: str


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(tensor: ir.ExternalTensor) -> list[object]:
    dimensions = []
    for dimension in tensor.shape:
        if not isinstance(dimension, int):
            raise TypeError("GGUF external tensors must retain static source dimensions.")
        dimensions.append(dimension)
    return [
        str(tensor.location),
        tensor.offset,
        tensor.length,
        int(tensor.dtype),
        dimensions,
    ]


def _validate_source_initializers(model: ir.Model) -> None:
    encoded = model.metadata_props.get("mobius.gguf.external_initializers")
    if encoded is None:
        raise ValueError("Original GGUF binding metadata is missing; rebuild from the source.")
    try:
        expected = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Original GGUF binding metadata is malformed; rebuild the model."
        ) from error
    if not isinstance(expected, dict):
        raise TypeError("Original GGUF binding metadata must be a map; rebuild the model.")
    for name, info in expected.items():
        value = model.graph.initializers.get(name)
        tensor = None if value is None else value.const_value
        if not isinstance(tensor, ir.ExternalTensor) or _binding(tensor) != info:
            raise ValueError(
                f"Original GGUF binding for {name!r} was changed by a transform. "
                "Rebuild without weight folding/transcoding; preserve source shape and bytes."
            )


def _restore_expert_banks(model: ir.Model) -> None:
    """Cancel source-bank splitting followed by ordered graph bank assembly."""
    for node in list(model.graph):
        if node.domain or node.op_type != "Concat" or node.outputs[0] in model.graph.outputs:
            continue
        if node.attributes.get_int("axis", 0) != 0 or not node.inputs:
            continue
        source = None
        for index, value in enumerate(node.inputs):
            unsqueeze = None if value is None else value.producer()
            if (
                unsqueeze is None
                or unsqueeze.domain
                or unsqueeze.op_type != "Unsqueeze"
                or len(unsqueeze.inputs) != 2
            ):
                break
            axes = unsqueeze.inputs[1]
            if (
                axes is None
                or axes.const_value is None
                or axes.const_value.numpy().tolist() != [0]
            ):
                break
            selected = unsqueeze.inputs[0]
            gather = None if selected is None else selected.producer()
            if (
                gather is None
                or gather.domain
                or gather.op_type != "Gather"
                or gather.attributes.get_int("axis", 0) != 0
            ):
                break
            bank, selection = gather.inputs
            if (
                bank is None
                or selection is None
                or selection.const_value is None
                or selection.const_value.numpy().shape != ()
                or selection.const_value.numpy().item() != index
                or (source is not None and bank is not source)
            ):
                break
            source = bank
        else:
            if (
                source is not None
                and source.shape is not None
                and len(source.shape)
                and source.shape[0] == len(node.inputs)
            ):
                node.outputs[0].replace_all_uses_with(source, replace_graph_outputs=True)
                model.graph.remove(node, safe=True)


def _select_packed_rows(model: ir.Model) -> None:
    """Decode only requested embedding rows, not an entire packed vocabulary."""
    for node in list(model.graph):
        if node.op_type != "Gather" or node.domain or node.attributes.get_int("axis", 0) != 0:
            continue
        decoded, indices = node.inputs
        if decoded is None or indices is None:
            continue
        if indices.dtype != ir.DataType.INT64:
            continue
        producer = decoded.producer()
        dtype = decoded.dtype
        if dtype is None:
            continue
        if producer is not None and producer.op_type == "Cast":
            cast_input = producer.inputs[0]
            producer = None if cast_input is None else cast_input.producer()
        if producer is None or producer.op_type != "DequantizePacked":
            continue
        shape = producer.attributes["shape"].as_ints()
        if len(shape) != 2 or shape[0] == 0:
            continue
        payload = producer.inputs[0]
        tensor = None if payload is None else payload.const_value
        if payload is None or tensor is None:
            continue
        row_bytes = tensor.nbytes // shape[0]
        if row_bytes * shape[0] != tensor.nbytes:
            continue
        prefix = f"{node.outputs[0].name}.packed_rows"

        def constant(name, data, prefix=prefix):
            array = np.asarray(data, dtype=np.int64)
            tensor = ir.tensor(array, name=f"{prefix}.{name}")
            return ir.Value(
                name=tensor.name,
                const_value=tensor,
                shape=ir.Shape(array.shape),
                type=ir.TensorType(ir.DataType.INT64),
            )

        row = ir.Value(
            name=f"{prefix}.row", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.INT64)
        )
        stride = constant("stride", row_bytes)
        zero = constant("zero", 0)
        count = constant("count", shape[0])
        negative = ir.Value(name=f"{prefix}.negative")
        wrapped = ir.Value(name=f"{prefix}.wrapped")
        normalized = ir.Value(name=f"{prefix}.normalized")
        below = ir.Value(name=f"{prefix}.below")
        above = ir.Value(name=f"{prefix}.above")
        invalid = ir.Value(name=f"{prefix}.invalid")
        checked = ir.Value(name=f"{prefix}.checked")
        start = ir.Value(name=f"{prefix}.start")
        end = ir.Value(name=f"{prefix}.end")
        starts = ir.Value(name=f"{prefix}.starts")
        ends = ir.Value(name=f"{prefix}.ends")
        codes = ir.Value(
            name=f"{prefix}.codes",
            shape=ir.Shape([row_bytes]),
            type=ir.TensorType(ir.DataType.UINT8),
        )
        output = ir.Value(
            name=f"{prefix}.decoded", shape=ir.Shape([shape[1]]), type=ir.TensorType(dtype)
        )
        vector = constant("vector", [1])
        body = ir.Graph(
            [row],
            [output],
            nodes=[
                ir.Node("", "Less", [row, zero], outputs=[negative]),
                ir.Node("", "Add", [row, count], outputs=[wrapped]),
                ir.Node("", "Where", [negative, wrapped, row], outputs=[normalized]),
                ir.Node("", "Less", [normalized, zero], outputs=[below]),
                ir.Node("", "GreaterOrEqual", [normalized, count], outputs=[above]),
                ir.Node("", "Or", [below, above], outputs=[invalid]),
                ir.Node("", "Where", [invalid, count, normalized], outputs=[checked]),
                ir.Node("", "Mul", [checked, stride], outputs=[start]),
                ir.Node("", "Add", [start, stride], outputs=[end]),
                ir.Node("", "Reshape", [start, vector], outputs=[starts]),
                ir.Node("", "Reshape", [end, vector], outputs=[ends]),
                ir.Node("", "Slice", [payload, starts, ends], outputs=[codes]),
                ir.Node(
                    "",
                    "DequantizePacked",
                    [codes],
                    outputs=[output],
                    attributes=ir.convenience.convert_attributes(
                        {
                            "format": producer.attributes["format"].as_string(),
                            "shape": [shape[1]],
                            "output_dtype": int(dtype),
                        }
                    ),
                ),
            ],
            initializers=[stride, vector, zero, count],
            name=prefix,
        )
        flat_shape = constant("flat_shape", [-1])
        width = constant("width", [shape[1]])
        model.graph.initializers[flat_shape.name] = flat_shape
        model.graph.initializers[width.name] = width
        flat = ir.Value(
            name=f"{prefix}.indices",
            shape=ir.Shape([None]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        scan_output = ir.Value(
            name=f"{prefix}.values",
            shape=ir.Shape([None, shape[1]]),
            type=ir.TensorType(dtype),
        )
        index_shape = ir.Value(name=f"{prefix}.index_shape")
        final_shape = ir.Value(name=f"{prefix}.output_shape")
        replacement = ir.Value(
            name=node.outputs[0].name, shape=node.outputs[0].shape, type=node.outputs[0].type
        )
        scanned = ir.Value(
            name=f"{prefix}.scanned", shape=scan_output.shape, type=scan_output.type
        )
        nonempty_body = ir.Graph(
            [],
            [scanned],
            nodes=[
                ir.Node(
                    "",
                    "Scan",
                    [flat],
                    outputs=[scanned],
                    attributes=ir.convenience.convert_attributes(
                        {"num_scan_inputs": 1, "body": body}
                    ),
                )
            ],
            name=f"{prefix}.nonempty",
        )
        empty_tensor = ir.tensor(np.empty((0, shape[1]), dtype=dtype.numpy()))
        empty = ir.Value(
            name=f"{prefix}.empty",
            shape=ir.Shape([0, shape[1]]),
            type=ir.TensorType(dtype),
            const_value=empty_tensor,
        )
        empty_body = ir.Graph(
            [], [empty], nodes=[], initializers=[empty], name=f"{prefix}.empty_body"
        )
        size = ir.Value(name=f"{prefix}.size")
        nonempty = ir.Value(name=f"{prefix}.has_rows")
        empty_size = constant("empty_size", 0)
        model.graph.initializers[empty_size.name] = empty_size
        nodes = [
            ir.Node("", "Reshape", [indices, flat_shape], outputs=[flat]),
            ir.Node("", "Size", [flat], outputs=[size]),
            ir.Node("", "Greater", [size, empty_size], outputs=[nonempty]),
            ir.Node(
                "",
                "If",
                [nonempty],
                outputs=[scan_output],
                attributes=ir.convenience.convert_attributes(
                    {
                        "then_branch": nonempty_body,
                        "else_branch": empty_body,
                    }
                ),
            ),
            ir.Node("", "Shape", [indices], outputs=[index_shape]),
            ir.Node(
                "",
                "Concat",
                [index_shape, width],
                outputs=[final_shape],
                attributes=ir.convenience.convert_attributes({"axis": 0}),
            ),
            ir.Node(
                "",
                "Reshape",
                [scan_output, final_shape],
                outputs=[replacement],
                attributes=ir.convenience.convert_attributes({"allowzero": 1}),
            ),
        ]
        model.graph.insert_before(node, nodes)
        node.outputs[0].replace_all_uses_with(replacement, replace_graph_outputs=True)
        model.graph.remove(node)


class _LayoutGraph(TorchDispatchMode):
    """Translate supported weight-only transforms, failing on unknown semantics."""

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[int, ir.Value] = {}
        self.keepalive: list[torch.Tensor] = []
        self.nodes: list[ir.Node] = []
        self.initializers: dict[str, ir.Value] = {}
        self.counter = 0

    def name(self) -> str:
        self.counter += 1
        return f"gguf.layout.{self.counter}"

    def remember(self, tensor: torch.Tensor, value: ir.Value) -> None:
        self.keepalive.append(tensor)
        self.values[id(tensor)] = value

    def constant(self, data, dtype: ir.DataType = ir.DataType.INT64) -> ir.Value:
        name = self.name()
        array = np.asarray(data, dtype=dtype.numpy())
        tensor = ir.tensor(array, name=name)
        value = ir.Value(
            name=name,
            const_value=tensor,
            shape=ir.Shape(array.shape),
            type=ir.TensorType(dtype),
        )
        self.initializers[name] = value
        return value

    def value(self, tensor: torch.Tensor) -> ir.Value:
        value = self.values.get(id(tensor))
        if value is not None:
            return value
        if tensor.device.type == "meta":
            raise ValueError(
                "GGUF layout processor created an untracked meta tensor. "
                "Add an exact ONNX lowering for this processor before exporting."
            )
        return self.constant(tensor.numpy(), _TYPES[tensor.dtype])

    def emit(self, op: str, inputs, result: torch.Tensor, **attrs) -> ir.Value:
        output = ir.Value(
            name=self.name(),
            shape=ir.Shape(result.shape),
            type=ir.TensorType(_TYPES[result.dtype]),
        )
        self.nodes.append(
            ir.Node(
                "",
                op,
                inputs,
                attributes=ir.convenience.convert_attributes(attrs),
                outputs=[output],
            )
        )
        self.remember(result, output)
        return output

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        tensors = torch.utils._pytree.tree_leaves((args, kwargs))
        tracked = [t for t in tensors if isinstance(t, torch.Tensor) and id(t) in self.values]
        if not tracked:
            return func(*args, **kwargs)
        op = func._schema.name.removeprefix("aten::")
        supported = {
            "view",
            "reshape",
            "_unsafe_view",
            "unsqueeze",
            "squeeze",
            "permute",
            "transpose",
            "t",
            "slice",
            "select",
            "split",
            "split_with_sizes",
            "unbind",
            "cat",
            "stack",
            "index_select",
            "clone",
            "detach",
            "alias",
            "contiguous",
            "_to_copy",
            "neg",
        }
        if op not in supported:
            raise ValueError(
                f"GGUF lossless layout conversion cannot represent {func}. "
                "Consume original GGUF parameters directly in the model graph. "
                "Floating inverse/offset reparameterization can round source values; "
                "no dense/requantized fallback is permitted."
            )
        result = func(*args, **kwargs)
        x = self.value(args[0]) if isinstance(args[0], torch.Tensor) else None
        if op in {"clone", "detach", "alias", "contiguous"}:
            if x is None:
                raise ValueError("GGUF layout alias has no source tensor; fix its processor.")
            self.remember(result, x)
        elif op in {"view", "reshape", "_unsafe_view", "unsqueeze", "squeeze"}:
            self.emit("Reshape", [x, self.constant(list(result.shape))], result)
        elif op in {"permute", "transpose", "t"}:
            perm = list(range(args[0].ndim))
            if op == "permute":
                perm = list(args[1])
            elif op == "t":
                perm.reverse()
            else:
                i, j = args[1:3]
                perm[i], perm[j] = perm[j], perm[i]
            self.emit("Transpose", [x], result, perm=perm)
        elif op in {"slice", "select"}:
            axis = args[1]
            if op == "select":
                self.emit("Gather", [x, self.constant(args[2])], result, axis=axis)
            else:
                step = args[4] if len(args) > 4 else 1
                self.emit(
                    "Slice",
                    [
                        x,
                        self.constant([args[2]]),
                        self.constant([args[3]]),
                        self.constant([axis]),
                        self.constant([step]),
                    ],
                    result,
                )
        elif op in {"split", "split_with_sizes", "unbind"}:
            axis = (
                args[2]
                if op != "unbind" and len(args) > 2
                else (args[1] if op == "unbind" and len(args) > 1 else 0)
            )
            offset = 0
            for part in result:
                if op == "unbind":
                    self.emit("Gather", [x, self.constant(offset)], part, axis=axis)
                    offset += 1
                else:
                    end = offset + part.shape[axis]
                    self.emit(
                        "Slice",
                        [
                            x,
                            self.constant([offset]),
                            self.constant([end]),
                            self.constant([axis]),
                        ],
                        part,
                    )
                    offset = end
        elif op in {"cat", "stack"}:
            axis = args[1] if len(args) > 1 else 0
            inputs = [self.value(t) for t in args[0]]
            if op == "stack":
                expanded = []
                for tensor, value in zip(args[0], inputs, strict=True):
                    shape = list(tensor.shape)
                    shape.insert(axis if axis >= 0 else axis + tensor.ndim + 1, 1)
                    proxy = torch.empty(shape, device="meta", dtype=tensor.dtype)
                    expanded.append(self.emit("Reshape", [value, self.constant(shape)], proxy))
                inputs = expanded
            self.emit("Concat", inputs, result, axis=axis)
        elif op == "index_select":
            self.emit("Gather", [x, self.value(args[2])], result, axis=args[1])
        elif op == "_to_copy":
            self.emit("Cast", [x], result, to=int(_TYPES[result.dtype]))
        elif op == "neg":
            self.emit("Neg", [x], result)
        return result


def bind_packed_weights(
    package,
    source,
    module,
    config,
    *,
    name_mapper=None,
    preprocess=None,
    effective_norms=False,
    original_decay_paths=frozenset(),
) -> None:
    """Bind each component to the same immutable GGUF source payloads."""
    from mobius import ModelPackage

    sources = dict(getattr(package, "gguf_packed_sources", {}))
    for model in package.values():
        component = ModelPackage({"model": model})
        _bind_packed_model(
            component,
            source,
            module,
            config,
            name_mapper=name_mapper,
            preprocess=preprocess,
            effective_norms=effective_norms,
            original_decay_paths=original_decay_paths,
        )
        for location, record in component.gguf_packed_sources.items():
            if location in sources and sources[location] != record:
                raise ValueError(
                    f"GGUF source filename {location!r} identifies different files. "
                    "Rename the original sources uniquely before composing components."
                )
            sources[location] = record
        package.gguf_quantization_report = component.gguf_quantization_report
    package.gguf_packed_sources = sources
    package.register_lazy_source_artifacts(
        files=[record["path"] for record in sources.values()],
        directories=[Path(record["path"]).parent for record in sources.values()],
    )


def _bind_packed_model(
    package,
    source,
    module,
    config,
    *,
    name_mapper=None,
    preprocess=None,
    effective_norms=False,
    original_decay_paths=frozenset(),
) -> None:
    """Bind one graph to original typed/packed GGUF tensors and exact layouts."""
    from mobius.integrations.gguf._builder import _normalize_gguf_weights
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
    from mobius.integrations.gguf._tensor_processors import process_tensors

    mapper = name_mapper or map_gguf_to_hf_names
    if not source.is_little_endian:
        raise ValueError(
            "Lossless GGUF external data requires little-endian input. "
            "ONNX cannot reinterpret big-endian blocks without transcoding."
        )
    from mobius.integrations.gguf._tencent_q1_0 import is_tencent_q1_0_layout

    if is_tencent_q1_0_layout(source):
        raise ValueError(
            "Tencent SEQ Q1_0 is a different 2-bit layout from ONNX gguf:q1_0. "
            "An exact source-byte decoder is not implemented; refusing reinterpretation."
        )
    model = next(iter(package.values()))
    if model.opset_imports.get("", 0) != 29:
        raise ValueError(
            "Lossless GGUF requires a graph built at opset 29. Enable "
            "MOBIUS_EXPERIMENTAL_OPSET29=1 and use --ep onnx-standard."
        )
    layout = _LayoutGraph()
    state = {}
    records = []
    sources: dict[str, _PackedSource] = {}
    for name in source.tensor_names:
        target = mapper(name, source.architecture)
        if target is None:
            continue
        owner = source.tensor_source_model(name)
        path = owner._path.resolve()
        if (
            path.name
            in {
                "model.onnx",
                _MANIFEST,
                "quantization_report.json",
                "export_report.json",
                ".gguf-packed.lock",
            }
            or path.name.startswith(".gguf-packed-")
            or ":" in path.name
        ):
            raise ValueError(
                f"GGUF source filename {path.name!r} conflicts with package artifacts. "
                "Rename the original file before building; it must never be overwritten."
            )
        previous = sources.get(path.name)
        if previous is not None and previous["path"] != str(path):
            raise ValueError(
                f"GGUF sources share filename {path.name!r} in different directories; "
                "place uniquely named original shards beside one another before building."
            )
        offset, length, qtype = owner.tensor_storage_range(name)
        shape = owner.get_tensor_shape(name)
        dtype = _FLOAT_TYPES.get(qtype)
        if dtype is not None and length != math.prod(shape) * dtype.itemsize:
            raise ValueError(
                f"GGUF tensor {name!r} has {length} bytes for {shape} {qtype}; "
                "repair the malformed tensor descriptor before exporting."
            )
        name_ir = f"gguf.source.{name}"
        packed_shape = shape if dtype is not None else (length,)
        external = ir.ExternalTensor(
            path.name,
            offset,
            length,
            dtype or ir.DataType.UINT8,
            shape=ir.Shape(packed_shape),
            name=name_ir,
            base_dir=path.parent,
        )
        value = ir.Value(
            name=name_ir,
            const_value=external,
            shape=external.shape,
            type=ir.TensorType(external.dtype),
        )
        layout.initializers[name_ir] = value
        proxy = torch.empty(shape, device="meta", dtype=torch.float32)
        if dtype is None:
            format_name = f"gguf:{qtype.lower()}"
            if format_name not in _PACKED_FORMATS:
                raise ValueError(
                    f"GGUF tensor {name!r} uses {qtype}, not supported by this "
                    "DequantizePacked-29 importer. Add its exact standardized layout "
                    "before exporting; requantization is forbidden."
                )
            block, block_bytes = _PACKED_FORMATS[format_name]
            elements = math.prod(shape)
            if not shape or elements % block or length != elements // block * block_bytes:
                raise ValueError(
                    f"GGUF {name!r} ({qtype}, shape={shape}, bytes={length}) does not "
                    f"match DequantizePacked's {block}-element/{block_bytes}-byte layout. "
                    "Check source codec/version and reader; no truncated decoding is allowed."
                )
            decoded_shape = shape if shape[-1] % block == 0 else (elements,)
            decoded_proxy = torch.empty(decoded_shape, device="meta")
            value = layout.emit(
                "DequantizePacked",
                [value],
                decoded_proxy,
                format=format_name,
                shape=list(decoded_shape),
                output_dtype=int(ir.DataType.FLOAT),
            )
            if decoded_shape != shape:
                value = layout.emit("Reshape", [value, layout.constant(list(shape))], proxy)
        elif dtype != ir.DataType.FLOAT:
            value = layout.emit("Cast", [value], proxy, to=int(ir.DataType.FLOAT))
        layout.remember(proxy, value)
        state[target] = proxy
        records.append((name, qtype, length))
        if path.name not in sources:
            sources[path.name] = {
                "size": owner.source_size,
                "sha256": owner.source_sha256(),
                "path": str(path),
            }
    with layout:
        if preprocess is None:
            state = process_tensors(state, config)
            state = _normalize_gguf_weights(
                state,
                source.architecture,
                config,
                effective_norms=effective_norms,
                original_decay_paths=original_decay_paths,
            )
            state = module.preprocess_weights(state)
        else:
            state = preprocess(state)
    bound = set()
    for name, tensor in state.items():
        initializer = model.graph.initializers.get(name)
        if initializer is None:
            continue
        value = layout.value(tensor)
        if initializer.shape != value.shape:
            raise ValueError(
                f"Lossless GGUF shape mismatch for {name}: source expression "
                f"{value.shape}, model expects {initializer.shape}. Fix the layout mapping."
            )
        if initializer.dtype != value.dtype:
            proxy = torch.empty(
                tensor.shape,
                device="meta",
                dtype=next(t for t, d in _TYPES.items() if d == initializer.dtype),
            )
            value = layout.emit("Cast", [value], proxy, to=int(initializer.dtype))
        initializer.replace_all_uses_with(value, replace_graph_outputs=True)
        del model.graph.initializers[name]
        bound.add(name)
    missing = [
        name for name, value in model.graph.initializers.items() if value.const_value is None
    ]
    if missing:
        raise ValueError(
            f"Lossless GGUF is missing model parameters: {missing[:10]}. "
            "Fix the exact tensor mapping; no fabricated/default weights are allowed."
        )
    model.graph.initializers.update(layout.initializers)
    model.graph.extend(layout.nodes)
    model.graph.sort()
    _restore_expert_banks(model)
    _select_packed_rows(model)
    from onnxscript import optimizer

    optimizer.remove_unused_nodes(model)
    model.metadata_props["mobius.gguf.original_external_data"] = "1"
    model.metadata_props["mobius.gguf.sources"] = json.dumps(
        {
            name: {key: value for key, value in info.items() if key != "path"}
            for name, info in sources.items()
        },
        sort_keys=True,
    )
    model.metadata_props["mobius.gguf.external_initializers"] = json.dumps(
        {
            name: _binding(value.const_value)
            for name, value in model.graph.initializers.items()
            if isinstance(value.const_value, ir.ExternalTensor)
        },
        sort_keys=True,
    )
    package.gguf_packed_sources = sources
    package.register_lazy_source_artifacts(
        files=[s["path"] for s in sources.values()],
        directories=[Path(s["path"]).parent for s in sources.values()],
    )
    from mobius.integrations.gguf._quantization_report import (
        GGUFQuantizationReport,
        QuantizationDisposition,
        QuantizationTensorRecord,
    )

    package.gguf_quantization_report = GGUFQuantizationReport.create(
        source_qtypes=[(qtype, length) for _, qtype, length in records],
        tensor_records=[
            QuantizationTensorRecord(
                name,
                qtype,
                length,
                QuantizationDisposition.SOURCE_FLOAT
                if qtype in {"F32", "F16", "BF16", "F64"}
                else QuantizationDisposition.NATIVE_BYTES,
                "GGUF external data",
                "Original source bytes; standard ONNX decoding/layout nodes.",
            )
            for name, qtype, length in records
        ],
        target_storage_format="original GGUF external data",
        compute_mode="DequantizePacked-29 + standard ONNX",
        compute_capability="schema representation; runtime kernels must be qualified",
    )


def save_packed_package(package, directory, *, external_data, max_shard_size_bytes) -> None:
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".gguf-packed.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError(
            f"GGUF save is locked by {lock.name}. Wait for the other save, or "
            "remove this lock only after confirming no save is running."
        ) from error
    try:
        _save_packed_package(
            package,
            root,
            external_data=external_data,
            max_shard_size_bytes=max_shard_size_bytes,
        )
    finally:
        os.close(descriptor)
        lock.unlink()


def _save_packed_package(package, directory, *, external_data, max_shard_size_bytes) -> None:
    """Save without the ordinary saver rewriting external source tensors."""
    if external_data != "onnx" or max_shard_size_bytes is not None:
        raise ValueError(
            "Lossless GGUF requires external_data='onnx' without shard-size conversion. "
            "Original GGUF shards are already the external data."
        )
    root = Path(directory).resolve()
    models = []
    sources = {}
    current = package
    depth = 0
    while current is not None:
        for location, record in getattr(current, "gguf_packed_sources", {}).items():
            if location in sources and sources[location] != record:
                raise ValueError(
                    f"GGUF filename {location!r} refers to different source files. "
                    "Rename the original files uniquely before saving."
                )
            sources[location] = record
        models.extend((depth, name, model) for name, model in current.items())
        current = current.mtp_head
        depth += 1
    for _, _, model in models:
        encoded = model.metadata_props.get("mobius.gguf.sources")
        if encoded is None:
            continue
        try:
            records = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError(
                "GGUF model source records are malformed; restore the model or rebuild."
            ) from error
        if not isinstance(records, dict):
            raise TypeError("GGUF model source records must be a map; rebuild the model.")
        for location, record in records.items():
            if (
                not isinstance(location, str)
                or not location
                or location in {".", ".."}
                or any(character in location for character in "/\\:")
                or not isinstance(record, dict)
                or set(record) != {"size", "sha256"}
                or type(record["size"]) is not int
                or record["size"] < 0
                or not isinstance(record["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            ):
                raise ValueError(
                    "GGUF model source record is invalid; restore its original metadata."
                )
            previous = sources.get(location)
            if previous is not None and any(
                previous[key] != record[key] for key in ("size", "sha256")
            ):
                raise ValueError(
                    f"Components disagree about GGUF source {location!r}. "
                    "Use uniquely named original files before composing components."
                )
            if previous is None:
                sources[location] = {**record, "path": str(root / location)}
    if not sources:
        raise ValueError(
            "Original GGUF source records are missing; compose packages with intact "
            "source metadata or rebuild from the original files."
        )
    if not models:
        raise ValueError("Cannot save an empty GGUF package; build at least one component.")
    legacy = len(models) == 1 and package.mtp_head is None and models[0][1] == "model"
    generation = uuid.uuid4().hex
    filenames = [
        "model.onnx" if legacy else f"component-{generation}-{index}.onnx"
        for index in range(len(models))
    ]
    reserved = {
        *filenames,
        _MANIFEST,
        "quantization_report.json",
        "export_report.json",
        ".gguf-packed.lock",
    }
    if reserved & sources.keys() or any(name.startswith(".gguf-packed-") for name in sources):
        raise ValueError(
            "An original GGUF filename conflicts with an ONNX component filename. "
            "Rename the original source before saving."
        )
    allowed = {
        *sources,
        *filenames,
        _MANIFEST,
        "quantization_report.json",
        "export_report.json",
        ".gguf-packed.lock",
    }
    if root.exists():
        unexpected = {
            p.name
            for p in root.iterdir()
            if p.name not in allowed
            and re.fullmatch(r"component-(?:[0-9a-f]{32}-)?[0-9]+\.onnx", p.name) is None
        }
        if unexpected:
            raise ValueError(
                f"Lossless GGUF output contains unrelated/stale artifacts: "
                f"{sorted(unexpected)}. Use a clean directory containing only source GGUFs."
            )
        if any(p.is_symlink() or not p.is_file() for p in root.iterdir()):
            raise ValueError("GGUF package artifacts must be regular, non-symlinked files.")
    for location, info in sources.items():
        source = Path(info["path"])
        if source.parent.resolve() != root:
            raise ValueError(
                f"Save beside the original GGUF files ({source.parent}), not {root}. "
                "Mobius will not copy or transcode GGUF payloads."
            )
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != info["size"]
        ):
            raise ValueError(
                f"GGUF external source {location!r} changed; rebuild the package."
            )
        if _source_sha256(source) != info["sha256"]:
            raise ValueError(
                f"GGUF external source {location!r} bytes changed; rebuild the package."
            )
    for _, _, model in models:
        _validate_source_initializers(model)
        _validate_external_ranges(model, sources)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".gguf-packed-") as staging:
        stage = Path(staging)
        for (_, _, model), filename in zip(models, filenames, strict=True):
            ir.save(model, stage / filename)
        manifest = {
            "format": "mobius.gguf-packed.v1" if legacy else "mobius.gguf-packed.v2",
            "sources": {
                name: {k: v for k, v in info.items() if k != "path"}
                for name, info in sources.items()
            },
        }
        if not legacy:
            manifest["models"] = [
                {"depth": level, "component": name, "file": filename}
                for (level, name, _), filename in zip(models, filenames, strict=True)
            ]
        (stage / _MANIFEST).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        if package.gguf_quantization_report is not None:
            package.gguf_quantization_report.write_json(stage / "quantization_report.json")
        if package.export_report is not None:
            package.export_report.write_json(stage / "export_report.json")
        for artifact in stage.iterdir():
            if artifact.name != _MANIFEST:
                os.replace(artifact, root / artifact.name)
        os.replace(stage / _MANIFEST, root / _MANIFEST)


def _validate_external_ranges(model, sources) -> None:
    for graph in model.graphs():
        for value in graph.initializers.values():
            tensor = value.const_value
            if isinstance(tensor, ir.ExternalTensor):
                info = sources.get(str(tensor.location))
                if (
                    info is None
                    or tensor.offset is None
                    or tensor.length is None
                    or tensor.offset < 0
                    or tensor.length < 0
                    or tensor.offset + tensor.length > info["size"]
                ):
                    raise ValueError(
                        f"GGUF initializer {value.name!r} has an invalid source range. "
                        "Restore the original binding or rebuild the component."
                    )


def packed_model_filenames(directory) -> list[str] | None:
    """Select only graphs committed by the packed manifest, ignoring orphan generations."""
    root = Path(directory)
    if (root / ".gguf-packed.lock").exists():
        raise ValueError(
            "GGUF save is in progress or interrupted; finish recovery before loading."
        )
    manifest = root / _MANIFEST
    if not manifest.exists():
        return None
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError(
            "GGUF manifest must be a regular file; restore the original manifest."
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "GGUF manifest is invalid JSON; restore the original manifest."
        ) from error
    if isinstance(payload, dict) and payload.get("format") == "mobius.gguf-packed.v1":
        filenames = ["model.onnx"]
    elif isinstance(payload, dict) and payload.get("format") == "mobius.gguf-packed.v2":
        records = payload.get("models")
        if not isinstance(records, list) or not records:
            raise ValueError(
                "GGUF component manifest is empty or invalid; rebuild the package."
            )
        filenames = []
        for record in records:
            filename = record.get("file") if isinstance(record, dict) else None
            if (
                not isinstance(filename, str)
                or re.fullmatch(r"component-(?:[0-9a-f]{32}-)?[0-9]+\.onnx", filename) is None
            ):
                raise ValueError("Invalid GGUF component filename; restore the manifest.")
            if filename in filenames:
                raise ValueError("GGUF component manifest contains missing/duplicate models.")
            filenames.append(filename)
    else:
        raise ValueError("Invalid GGUF packed manifest; rebuild from the original GGUF.")
    for filename in filenames:
        path = root / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"GGUF manifest component {filename!r} is missing/unsafe; restore the package."
            )
    return filenames


def restore_packed_sources(package, directory) -> None:
    """Preserve source-file binding when a saved package is loaded and re-saved."""
    path = Path(directory) / _MANIFEST
    if (Path(directory) / ".gguf-packed.lock").exists():
        raise ValueError(
            "GGUF save is in progress or interrupted; finish recovery before loading."
        )
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            "GGUF packed manifest is missing or not a regular file; rebuild the package."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "GGUF packed manifest contains invalid JSON; restore it or rebuild."
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("format") not in {"mobius.gguf-packed.v1", "mobius.gguf-packed.v2"}
        or not isinstance(payload.get("sources"), dict)
        or not payload["sources"]
    ):
        raise ValueError("Invalid GGUF packed manifest; rebuild from the original GGUF.")
    sources = {}
    for location, info in payload["sources"].items():
        if (
            not isinstance(location, str)
            or location in {".", ".."}
            or Path(location).name != location
            or "/" in location
            or "\\" in location
            or ":" in location
        ):
            raise ValueError("GGUF external locations must be flat filenames.")
        if (
            not isinstance(info, dict)
            or set(info) != {"size", "sha256"}
            or type(info["size"]) is not int
            or info["size"] < 0
            or not isinstance(info["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) is None
        ):
            raise ValueError(f"Invalid GGUF source record {location!r}; rebuild the manifest.")
        source = Path(directory).resolve() / location
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != info["size"]
        ):
            raise ValueError(
                f"GGUF source {location!r} is missing/changed; restore the original file."
            )
        if _source_sha256(source) != info["sha256"]:
            raise ValueError(
                f"GGUF source {location!r} bytes changed; restore the original file."
            )
        sources[location] = {**info, "path": str(source)}
    if payload["format"] == "mobius.gguf-packed.v2":
        _restore_components(package, payload.get("models"))
    elif len(package) != 1:
        raise ValueError(
            "GGUF v1 manifest requires one model; restore the original package "
            "or rebuild a multi-component v2 package."
        )
    models = []
    current = package
    while current is not None:
        models.extend(current.values())
        current = current.mtp_head
    if not models or any(
        model.metadata_props.get("mobius.gguf.original_external_data") != "1"
        for model in models
    ):
        raise ValueError(
            "GGUF manifest requires original-external-data components; rebuild the package."
        )
    for model in models:
        _validate_external_ranges(model, sources)
        _validate_source_initializers(model)
    current = package
    while current is not None:
        current.gguf_packed_sources = sources
        current.register_lazy_source_artifacts(
            files=[info["path"] for info in sources.values()],
            directories=[Path(info["path"]).parent for info in sources.values()],
        )
        current = current.mtp_head


def _restore_components(package, records) -> None:
    from mobius import ModelPackage

    if not isinstance(records, list) or not records:
        raise ValueError("GGUF component manifest is empty or invalid; rebuild the package.")
    components = {}
    used = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"depth", "component", "file"}
            or type(record["depth"]) is not int
            or record["depth"] < 0
            or not isinstance(record["component"], str)
            or not record["component"]
            or not isinstance(record["file"], str)
            or re.fullmatch(r"component-(?:[0-9a-f]{32}-)?[0-9]+\.onnx", record["file"])
            is None
        ):
            raise ValueError("Invalid GGUF component record; restore the manifest or rebuild.")
        name = record["file"].removesuffix(".onnx")
        key = (record["depth"], record["component"])
        if name not in package or name in used or key in components:
            raise ValueError(
                "GGUF component manifest contains missing/duplicate models; "
                "restore all original ONNX components and the manifest."
            )
        used.add(name)
        components[key] = package[name]
    if used != set(package) or {level for level, _ in components} != set(
        range(max(level for level, _ in components) + 1)
    ):
        raise ValueError(
            "GGUF component manifest does not match the saved package; "
            "restore its original components and contiguous MTP chain."
        )
    package.clear()
    current = package
    for level in range(max(depth for depth, _ in components) + 1):
        if level:
            current.mtp_head = ModelPackage({})
            current = current.mtp_head
        current.update(
            (name, model) for (depth, name), model in components.items() if depth == level
        )
