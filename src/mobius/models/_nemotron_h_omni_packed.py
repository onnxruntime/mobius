# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit compact CUDA lowering for a source-accounted Bolt decoder.

The packed backend uses selected-route rather than expert-ID reduction order.
It is a separately qualified graph variant, not a numerically transparent
fusion. Mamba, attention, masks, media and cache interfaces remain unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import onnx_ir as ir
from onnx_ir.passes import common as common_passes
from onnxscript import GraphBuilder, OpBuilder

from mobius._configs import NemotronHOmniConfig
from mobius._model_package import ModelPackage
from mobius.components import packed_row_tiles


def _ggml_parts(tensor: ir.TensorProtocol, *, iq2: bool) -> tuple[np.ndarray, np.ndarray]:
    raw = tensor.numpy()
    payload = np.ascontiguousarray(raw[..., 2:])
    if not iq2:
        # GGML stores signed Q8 bytes; the packed CUDA ABI decodes byte - 128.
        payload = np.bitwise_xor(payload, np.uint8(128))
    payload = payload.reshape(raw.shape[0], -1, 8 if iq2 else 32)
    scales = np.ascontiguousarray(raw[..., :2]).view(np.float16).reshape(raw.shape[:2])
    return payload, scales


def _packed_initializer(
    graph: ir.Graph,
    name: str,
    sources: Sequence[ir.Value],
    *,
    iq2: bool,
    scales: bool,
    bank: bool,
) -> ir.Value:
    first = sources[0]
    if first.shape is None:
        raise ValueError(f"Missing packed weight shape: {first.name}")
    n, blocks, block_bytes = first.shape
    expected_bytes = 66 if iq2 else 34
    if (
        block_bytes != expected_bytes
        or not isinstance(n, int)
        or not isinstance(blocks, int)
        or n <= 0
        or blocks <= 0
    ):
        raise ValueError(f"Unexpected GGML block layout: {first.name}")
    shape = [n, blocks] if scales else [n, blocks * (8 if iq2 else 1), 8 if iq2 else 32]
    if bank:
        shape.insert(0, len(sources))
    tensors = []
    for source in sources:
        if (
            source.dtype != ir.DataType.UINT8
            or source.shape != first.shape
            or source.const_value is None
        ):
            raise ValueError(
                f"Packed CUDA requires loaded, uniform GGML sources: {source.name}"
            )
        tensors.append(source.const_value)
    if name in graph.initializers:
        raise ValueError(f"Packed initializer name already exists: {name}")

    def materialize() -> ir.TensorProtocol:
        # One packed expert bank at a time; never reconstruct dense weights.
        if not bank:
            return ir.tensor(_ggml_parts(tensors[0], iq2=iq2)[int(scales)])
        result = np.empty(shape, dtype=np.float16 if scales else np.uint8)
        for index, tensor in enumerate(tensors):
            result[index] = _ggml_parts(tensor, iq2=iq2)[int(scales)]
        return ir.tensor(result)

    dtype = ir.DataType.FLOAT16 if scales else ir.DataType.UINT8
    packed_shape = ir.Shape(shape)
    value = ir.Value(name=name, type=ir.TensorType(dtype), shape=packed_shape)
    value.const_value = ir.LazyTensor(
        materialize, dtype=dtype, shape=packed_shape, cache=False
    )
    graph.initializers[name] = value
    return value


def _native_ggml_initializer(
    graph: ir.Graph, name: str, sources: Sequence[ir.Value], *, iq2: bool, bank: bool
) -> ir.Value:
    first = sources[0]
    if first.shape is None or len(first.shape) != 3:
        raise ValueError(f"Missing native GGML weight shape: {first.name}")
    n, blocks, block_bytes = first.shape
    if (
        block_bytes != (66 if iq2 else 34)
        or not isinstance(n, int)
        or not isinstance(blocks, int)
        or n <= 0
        or blocks <= 0
    ):
        raise ValueError(f"Unexpected native GGML block layout: {first.name}")
    tensors = []
    for source in sources:
        if (
            source.dtype != ir.DataType.UINT8
            or source.shape != first.shape
            or source.const_value is None
        ):
            raise ValueError(
                f"Native CUDA requires loaded, uniform GGML sources: {source.name}"
            )
        tensors.append(source.const_value)
    if name in graph.initializers:
        raise ValueError(f"Native initializer name already exists: {name}")
    shape: list[int] = [n, blocks * (66 if iq2 else 34)]
    if bank:
        shape.insert(0, len(sources))

    def materialize() -> ir.TensorProtocol:
        # Retain every FP16 scale and record byte; only flatten the block axis.
        if not bank:
            return ir.tensor(tensors[0].numpy().reshape(shape))
        result = np.empty(shape, dtype=np.uint8)
        for index, tensor in enumerate(tensors):
            result[index] = tensor.numpy().reshape(shape[1:])
        return ir.tensor(result)

    packed_shape = ir.Shape(shape)
    value = ir.Value(
        name=name,
        type=ir.TensorType(ir.DataType.UINT8),
        shape=packed_shape,
        const_value=ir.LazyTensor(
            materialize, dtype=ir.DataType.UINT8, shape=packed_shape, cache=False
        ),
    )
    graph.initializers[name] = value
    return value


def _native_nvfp4_initializer(
    graph: ir.Graph, weights: ir.Value, scales: ir.Value, *, n: int, k: int
) -> ir.Value:
    if (
        weights.dtype != ir.DataType.UINT8
        or scales.dtype != ir.DataType.UINT8
        or weights.shape != ir.Shape([n, k // 2])
        or scales.shape != ir.Shape([n, k // 16])
        or weights.const_value is None
        or scales.const_value is None
    ):
        raise ValueError("Native NVFP4 requires loaded byte weights and per-16 byte scales")
    if weights.name is None or scales.name is None:
        raise ValueError("Native NVFP4 weights and scales must be named")
    name = weights.name + ".native_nvfp4." + scales.name
    if name in graph.initializers:
        raise ValueError(f"Native initializer name already exists: {name}")
    weight_tensor, scale_tensor = weights.const_value, scales.const_value
    shape = ir.Shape([n, (k // 64) * 36])

    def materialize() -> ir.TensorProtocol:
        scale_bytes = scale_tensor.numpy()
        if np.any(scale_bytes & np.uint8(128)) or np.any(scale_bytes == 127):
            raise ValueError("Native NVFP4 requires nonnegative finite E4M3 scales")
        packed = weight_tensor.numpy().reshape(n, k // 16, 8)
        elements = np.empty((n, k // 16, 16), dtype=np.uint8)
        elements[..., 0::2] = packed & np.uint8(15)
        elements[..., 1::2] = packed >> 4
        # GGUF pairs element i with i+8, unlike the source's adjacent pairs.
        codes = elements[..., :8] | (elements[..., 8:] << 4)
        result = np.empty((n, k // 64, 36), dtype=np.uint8)
        result[..., :4] = scale_bytes.reshape(n, k // 64, 4)
        result[..., 4:] = codes.reshape(n, k // 64, 32)
        return ir.tensor(result.reshape(n, -1))

    value = ir.Value(
        name=name,
        type=ir.TensorType(ir.DataType.UINT8),
        shape=shape,
        const_value=ir.LazyTensor(
            materialize, dtype=ir.DataType.UINT8, shape=shape, cache=False
        ),
    )
    graph.initializers[name] = value
    return value


def _input(node: ir.Node, index: int) -> ir.Value:
    value = node.inputs[index]
    if value is None:
        raise ValueError(f"Missing required input {index} of {node.op_type}")
    return value


def _producer(value: ir.Value | None, op_type: str) -> ir.Node:
    if value is None:
        raise ValueError(f"Missing input to {op_type}")
    node = value.producer()
    if node is None or node.domain or node.op_type != op_type:
        raise ValueError(f"Unsupported Bolt MoE expression; expected {op_type}")
    return node


def _flatten_sum(value: ir.Value | None) -> list[ir.Value]:
    if value is None:
        raise ValueError("Missing Bolt routed sum")
    node = value.producer()
    if node is not None and not node.domain and node.op_type == "Add":
        return _flatten_sum(node.inputs[0]) + _flatten_sum(node.inputs[1])
    return [value]


def _moe_boundary(graph: ir.Graph, root: str, experts: int):
    candidates = [
        node
        for node in graph
        if node.op_type == "Cast"
        and not node.domain
        and node.attributes.get_int("to") == ir.DataType.BFLOAT16
        and (node.outputs[0].name or "").startswith("v_" + root + ".CastLike_")
    ]
    matches = [node for node in candidates if len(_flatten_sum(node.inputs[0])) == experts]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify the original Bolt routed sum: {root}")
    anchor = matches[0]
    contributions = _flatten_sum(anchor.inputs[0])
    latent = routing = selected = None
    for expert, contribution in enumerate(contributions):
        product = _producer(contribution, "Mul")
        promoted = _producer(product.inputs[0], "Cast")
        if promoted.attributes.get_int("to") != ir.DataType.FLOAT:
            raise ValueError("Bolt expert mixing must accumulate in FP32")
        down = _input(promoted, 0).producer()
        if down is None or (down.domain, down.op_type) != (
            "com.microsoft",
            "MatMulGGMLIQ2XXS",
        ):
            raise ValueError("Packed Bolt experts require IQ2_XXS down projections")
        square = _producer(down.inputs[0], "Mul")
        if square.inputs[0] is not square.inputs[1]:
            raise ValueError("Packed Bolt experts require squared ReLU")
        relu = _producer(square.inputs[0], "Relu")
        up = _input(relu, 0).producer()
        if up is None or (up.domain, up.op_type) != ("com.microsoft", "MatMulGGMLIQ2XXS"):
            raise ValueError("Packed Bolt experts require IQ2_XXS up projections")
        for node, projection in ((up, "up_proj"), (down, "down_proj")):
            expected = f"{root}.experts.{expert}.{projection}.weight"
            if len(node.inputs) != 2 or node.inputs[1] is not graph.initializers.get(expected):
                raise ValueError(f"Unsupported expert projection or bias: {expected}")
        reduced = _producer(product.inputs[1], "ReduceSum")
        axes = _input(reduced, 1).const_value
        if (
            axes is None
            or axes.numpy().tolist() != [-1]
            or reduced.attributes.get_int("keepdims", 1) != 1
        ):
            raise ValueError("Unsupported expert routing reduction")
        weighted = _producer(reduced.inputs[0], "Mul")
        mask = _producer(weighted.inputs[1], "Cast")
        if mask.attributes.get_int("to") != ir.DataType.FLOAT:
            raise ValueError("Unsupported expert routing mask")
        equal = _producer(mask.inputs[0], "Equal")
        identifier = _input(equal, 1).const_value
        if identifier is None or identifier.numpy().item() != expert:
            raise ValueError("Unsupported expert selection order")
        current = (_input(up, 0), _input(weighted, 0), _input(equal, 0))
        if expert == 0:
            latent, routing, selected = current
        elif any(a is not b for a, b in zip(current, (latent, routing, selected))):
            raise ValueError("Expert inputs or routing differ within a Bolt block")
    # Refuse instrumented/modified graphs exposing replaced intermediates.
    boundary = {latent, routing, selected}
    internal: set[ir.Node] = set()

    def collect(value: ir.Value):
        if value in boundary or value.is_initializer():
            return
        node = value.producer()
        if node is None or node.op_type == "Constant" or node in internal:
            return
        internal.add(node)
        for inp in node.inputs:
            if inp is not None:
                collect(inp)

    collect(_input(anchor, 0))
    for node in internal:
        for value in node.outputs:
            if value in graph.outputs or any(
                use.node not in internal and use.node is not anchor for use in value.uses()
            ):
                raise ValueError("Packed Bolt cannot remove observed expert intermediates")
    return anchor, latent, routing, selected


def _replace(graph: ir.Graph, anchor: ir.Node, emit: Callable[[OpBuilder], ir.Value]) -> None:
    temporary = ir.Graph(
        [], [], nodes=[], opset_imports=dict(graph.opset_imports), name="packed"
    )
    builder = GraphBuilder(temporary)
    result = emit(builder.op)
    original = anchor.outputs[0]
    original_name = original.name or f"packed_{len(graph)}"
    prefix = original_name + ".packed"
    initializers = list(temporary.initializers.values())
    temporary.initializers.clear()
    for value in initializers:
        value.name = prefix + "." + str(value.name)
        graph.initializers[value.name] = value
    for index, node in enumerate(temporary.all_nodes()):
        node.name = f"{prefix}.node_{index}"
        for output_index, value in enumerate(node.outputs):
            value.name = f"{prefix}.value_{index}_{output_index}"
    result.name, original.name = original.name, original_name + ".unpacked"
    result.type, result.shape = original.type, original.shape
    # The temporary graph only supplies a tape; captured values retain identity.
    nodes = list(temporary)
    temporary.remove(nodes)
    graph.insert_before(anchor, nodes)
    anchor.outputs[0].replace_all_uses_with(result, replace_graph_outputs=True)
    graph.opset_imports["com.microsoft"] = 1


def _pack_moe(
    graph: ir.Graph, config: NemotronHOmniConfig, index: int, *, native_blocks: bool = False
):
    layer = config.for_layer(index)
    experts, topk, intermediate = (
        layer.num_local_experts,
        layer.num_experts_per_tok,
        layer.moe_intermediate_size,
    )
    if (
        experts is None
        or topk is None
        or intermediate is None
        or not 1 <= topk <= min(experts, 32)
        or experts > 512
        or layer.moe_latent_size != 1024
        or intermediate <= 0
        or intermediate % 256
        or layer.hidden_act != "relu2"
        or layer.mlp_bias
    ):
        raise ValueError(f"Unsupported packed Bolt MoE topology at layer {index}")
    root = f"decoder.model.layers.{index}.moe"
    anchor, latent, routing, selected = _moe_boundary(graph, root, experts)
    topk_node = _producer(selected, "TopK")
    k = _input(topk_node, 1).const_value
    if k is None or k.numpy().item() != topk or selected is not topk_node.outputs[1]:
        raise ValueError(f"Packed Bolt routing disagrees with configured top-k: {root}")
    banks = []
    for projection in ("up_proj", "down_proj"):
        sources = [
            graph.initializers[f"{root}.experts.{e}.{projection}.weight"]
            for e in range(experts)
        ]
        if native_blocks:
            banks.append(
                _native_ggml_initializer(
                    graph, f"{root}.native_{projection}_weight", sources, iq2=True, bank=True
                )
            )
            continue
        for scales in (False, True):
            banks.append(
                _packed_initializer(
                    graph,
                    f"{root}.packed_{projection}_{'scales' if scales else 'weight'}",
                    sources,
                    iq2=True,
                    scales=scales,
                    bank=True,
                )
            )
    up_shape = (
        [experts, intermediate, 264] if native_blocks else [experts, intermediate, 32, 8]
    )
    down_shape = (
        [experts, 1024, (intermediate // 256) * 66]
        if native_blocks
        else [experts, 1024, intermediate // 32, 8]
    )
    if banks[0].shape != ir.Shape(up_shape) or banks[
        1 if native_blocks else 2
    ].shape != ir.Shape(down_shape):
        raise ValueError(f"Packed expert dimensions disagree with configuration: {root}")

    def emit(op):
        # Route with the original gate; only expert dispatch/mixing is replaced.
        flat = op.Reshape(latent, [-1, 1024])
        indices = op.Reshape(op.Cast(selected, to=ir.DataType.INT32), [-1, topk])
        weights = op.Reshape(routing, [-1, topk])
        tile_rows = min(8, 128 // topk)
        rows = op.Shape(flat, start=0, end=1)
        # A single-token decode must not dispatch eight copies of its experts.
        tile_size = op.Min(rows, [tile_rows])
        padding = op.Mod(op.Neg(rows), tile_size, fmod=0)
        # Repeat whole valid rows, including unique IDs, then discard their outputs.
        row_indices = op.Min(
            op.Range(0, op.Squeeze(op.Add(rows, padding), [0]), 1), op.Sub(rows, [1])
        )
        inputs = [
            op.Reshape(
                op.Gather(value, row_indices, axis=0),
                op.Concat([-1], tile_size, [width], axis=0),
            )
            for value, width in ((flat, 1024), (indices, topk), (weights, topk))
        ]
        scan_inputs = [
            ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape([None, width]))
            for name, dtype, width in (
                ("expert_x", ir.DataType.BFLOAT16, 1024),
                ("expert_indices", ir.DataType.INT32, topk),
                ("expert_weights", ir.DataType.FLOAT, topk),
            )
        ]

        def body(body_op, x, ids, mix):
            if native_blocks:
                result = body_op.MoEPacked(
                    x,
                    None,
                    banks[0],
                    None,
                    banks[1],
                    ids,
                    mix,
                    k=topk,
                    activation_type="relu_squared",
                    gate_type=16,
                    down_type=16,
                    normalize_routing_weights=0,
                    _domain="com.microsoft",
                )
                result.dtype = ir.DataType.BFLOAT16
                result.shape = ir.Shape([None, 1024])
                return result
            result, hidden, expert_output, metadata = body_op.MoEPackedIQ2XXS(
                x, ids, mix, *banks, topk=topk, _domain="com.microsoft", _outputs=4
            )
            result.dtype = hidden.dtype = ir.DataType.BFLOAT16
            expert_output.dtype, metadata.dtype = ir.DataType.FLOAT, ir.DataType.INT32
            result.shape = ir.Shape([None, 1024])
            return result

        body_graph = op.builder.subgraph(
            body,
            inputs=scan_inputs,
            outputs=[ir.Value(name="expert_result")],
            name=f"packed_experts_{index}",
        )
        result = op.Scan(*inputs, body=body_graph, num_scan_inputs=3)
        result = op.Slice(op.Reshape(result, [-1, 1024]), [0], rows, [0])
        result = op.Reshape(result, op.Shape(latent))
        result.dtype, result.shape = ir.DataType.BFLOAT16, latent.shape
        return result

    _replace(graph, anchor, emit)


def _pack_projection(
    graph: ir.Graph,
    node: ir.Node,
    packed_weights: dict[tuple[ir.Value, ...], list[ir.Value]] | None = None,
    *,
    native_blocks: bool = False,
) -> bool:
    x, source = _input(node, 0), _input(node, 1)
    if source.shape is None or not source.is_initializer() or x.dtype != ir.DataType.BFLOAT16:
        return False
    iq2 = node.op_type == "MatMulGGMLIQ2XXS"
    if node.op_type in {"MatMulGGMLIQ2XXS", "MatMulGGMLQ8_0"}:
        n, blocks, _ = source.shape
        if not isinstance(n, int) or not isinstance(blocks, int):
            return False
        k = blocks * (256 if iq2 else 32)
        if not isinstance(k, int) or k % 64:
            return False
        if source.name is None:
            raise ValueError("Packed projection weights must be named")
        key: tuple[ir.Value, ...] = (source,)
        if packed_weights is not None and key in packed_weights:
            inputs = packed_weights[key]
        else:
            if native_blocks:
                inputs = [
                    _native_ggml_initializer(
                        graph, source.name + ".native_weight", [source], iq2=iq2, bank=False
                    )
                ]
            else:
                inputs = [
                    _packed_initializer(
                        graph,
                        source.name + (".packed_scales" if scale else ".packed_weight"),
                        [source],
                        iq2=iq2,
                        scales=scale,
                        bank=False,
                    )
                    for scale in (False, True)
                ]
            if packed_weights is not None:
                packed_weights[key] = inputs
        opname = (
            "MatMulPacked"
            if native_blocks
            else ("MatMulPackedIQ2XXS" if iq2 else "MatMulPackedQ8")
        )
        quant_type = "iq2_xxs" if iq2 else "q8_0"
        bias = node.inputs[2] if len(node.inputs) > 2 else None
    elif node.op_type == "MatMulNVFP4A16":
        n, half_k = source.shape
        if not isinstance(n, int) or not isinstance(half_k, int):
            return False
        k = half_k * 2
        # The kernel's single-row NVFP4 specialization supports only these K.
        if k not in {4096, 8192}:
            return False
        inputs = [_input(node, index) for index in range(1, 4)]
        if native_blocks:
            if inputs[2].dtype != ir.DataType.FLOAT or inputs[2].shape not in (
                ir.Shape([]),
                ir.Shape([1]),
            ):
                raise ValueError("Native NVFP4 requires a scalar FP32 global scale")
            if (
                inputs[2].const_value is None
                or not np.isfinite(inputs[2].const_value.numpy()).all()
            ):
                raise ValueError("Native NVFP4 requires a loaded finite global scale")
            key = (source, inputs[1])
            if packed_weights is not None and key in packed_weights:
                native = packed_weights[key][0]
            else:
                native = _native_nvfp4_initializer(graph, inputs[0], inputs[1], n=n, k=k)
                if packed_weights is not None:
                    packed_weights[key] = [native]
            inputs = [native, inputs[2]]
        opname, bias = ("MatMulPacked" if native_blocks else "MatMulPackedNVFP4"), None
        quant_type = "nvfp4"
    else:
        return False

    def emit(op):
        def project(body_op, rows):
            attrs = (
                {
                    "K": k,
                    "N": n,
                    "packed_type": {"q8_0": 8, "iq2_xxs": 16, "nvfp4": 40}[quant_type],
                    "dequantize_to_activation_dtype": 1,
                }
                if native_blocks
                else {}
            )
            result = getattr(body_op, opname)(rows, *inputs, _domain="com.microsoft", **attrs)
            result.dtype = ir.DataType.BFLOAT16
            result.shape = ir.Shape([rows.shape[0], n]) if rows.shape is not None else None
            return result

        result = packed_row_tiles(
            op, x, input_width=k, output_width=n, tile_rows=8, project=project
        )
        return op.Add(result, bias) if bias is not None else result

    _replace(graph, node, emit)
    return True


def optimize_nemotron_h_omni_cuda(
    package: ModelPackage, *, native_blocks: bool = False
) -> ModelPackage:
    """Return a compact decoder, leaving the source package and media untouched.

    Must run after the native source-accounted loader. Payload/scales are split
    lazily at serialization; no checkpoint reads or dense reconstruction occur.
    ``native_blocks=True`` selects MatMulPacked/MoEPacked with native block
    bytes and embedded FP16 scales, using ORT's integer ``packed_type`` ABI.
    The default preserves the original split-payload ABI.
    Requires a corresponding custom ORT build; neither variant implies parity.
    """
    config = package.config
    if not isinstance(config, NemotronHOmniConfig) or config.dtype != ir.DataType.BFLOAT16:
        raise ValueError("Packed Bolt CUDA export requires NemotronHOmniConfig and BF16")
    if "decoder" not in package or not config.quantized_projections:
        raise ValueError("Packed Bolt CUDA export requires a native split decoder")
    source = package["decoder"]
    if source.metadata_props.get("mobius.bolt.packed_cuda") == "1":
        if (
            native_blocks
            and source.metadata_props.get("mobius.bolt.native_packed_cuda_abi")
            != "packed_type_v1"
        ):
            raise ValueError("Packed ABI changes require the original source package")
        if native_blocks != (
            source.metadata_props.get("mobius.bolt.native_packed_cuda") == "1"
        ):
            raise ValueError("Packed ABI changes require the original source package")
        return package
    for value in source.graph.initializers.values():
        if value.const_value is None:
            raise ValueError(f"Packed Bolt CUDA export requires loaded weights: {value.name}")
    compact = source.clone()
    graph = compact.graph
    layers = [i for i, kind in enumerate(config.layer_types or []) if kind == "moe"]
    if not layers:
        raise ValueError("Packed Bolt CUDA export requires routed expert layers")
    for index in layers:
        _pack_moe(graph, config, index, native_blocks=native_blocks)
    common_passes.RemoveUnusedNodesPass()(compact)
    packed_weights: dict[tuple[ir.Value, ...], list[ir.Value]] = {}
    projections = sum(
        _pack_projection(graph, node, packed_weights, native_blocks=native_blocks)
        for node in list(graph)
        if node.domain == "com.microsoft"
        and node.op_type in {"MatMulGGMLIQ2XXS", "MatMulGGMLQ8_0", "MatMulNVFP4A16"}
    )
    common_passes.RemoveUnusedNodesPass()(compact)
    compact.metadata_props["mobius.bolt.packed_cuda"] = "1"
    if native_blocks:
        compact.metadata_props["mobius.bolt.native_packed_cuda"] = "1"
        compact.metadata_props["mobius.bolt.native_packed_cuda_abi"] = "packed_type_v1"
    compact.metadata_props["mobius.bolt.mamba_backend"] = "unchanged"
    compact.metadata_props["mobius.bolt.routing_backend"] = "unchanged"
    result = ModelPackage({**package, "decoder": compact}, config=config)
    result.weight_loading_report = {
        **(package.weight_loading_report or {}),
        "packed_cuda": {
            "storage": "native_blocks" if native_blocks else "split_payload_scales",
            "mamba": "unchanged",
            "routing": "unchanged",
            "full_checkpoint_dequantization": False,
            "moe_layers": len(layers),
            "projections": projections,
            "decoder_nodes_before": len(source.graph),
            "decoder_nodes_after": len(graph),
            "operators": dict(Counter(node.op_type for node in graph.all_nodes())),
        },
    }
    return result
