# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit compact CUDA lowering for a source-accounted Bolt decoder.

The packed backend uses full FP32 routing and selected-route expert reduction.
It is a separately qualified graph variant, not a numerically transparent
fusion. Mamba, attention, masks, media and cache interfaces remain unchanged.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import onnx_ir as ir
from onnx_ir.passes import common as common_passes
from onnxscript import GraphBuilder, OpBuilder

from mobius._configs import NemotronHOmniConfig
from mobius._model_package import ModelPackage
from mobius.components import packed_row_tiles


def _packed_initializer(
    graph: ir.Graph,
    name: str,
    sources: Sequence[ir.Value],
    *,
    iq2: bool,
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
    shape = [n, blocks * expected_bytes]
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
        # Preserve complete GGML blocks, including FP16 scales and signed Q8 bytes.
        if not bank:
            return ir.tensor(np.ascontiguousarray(tensors[0].numpy()).reshape(shape))
        result = np.empty(shape, dtype=np.uint8)
        for index, tensor in enumerate(tensors):
            result[index] = tensor.numpy().reshape(n, blocks * expected_bytes)
        return ir.tensor(result)

    dtype = ir.DataType.UINT8
    packed_shape = ir.Shape(shape)
    value = ir.Value(name=name, type=ir.TensorType(dtype), shape=packed_shape)
    value.const_value = ir.LazyTensor(
        materialize, dtype=dtype, shape=packed_shape, cache=False
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
    for value in (routing, selected):
        if value in graph.outputs:
            raise ValueError("Packed Bolt cannot remove observed routing intermediates")
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
            if value.name != "":
                value.name = f"{prefix}.value_{index}_{output_index}"
    result.name, original.name = original.name, original_name + ".unpacked"
    result.type, result.shape = original.type, original.shape
    # The temporary graph only supplies a tape; captured values retain identity.
    nodes = list(temporary)
    temporary.remove(nodes)
    graph.insert_before(anchor, nodes)
    anchor.outputs[0].replace_all_uses_with(result, replace_graph_outputs=True)
    graph.opset_imports["com.microsoft"] = 1


def _pack_moe(graph: ir.Graph, config: NemotronHOmniConfig, index: int):
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
    if (
        k is None
        or k.numpy().item() != topk
        or selected is not topk_node.outputs[1]
        or topk_node.attributes.get_int("axis", -1) != -1
        or topk_node.attributes.get_int("largest", 1) != 1
        or topk_node.attributes.get_int("sorted", 1) != 1
    ):
        raise ValueError(f"Packed Bolt routing disagrees with configured top-k: {root}")
    choice = _producer(topk_node.inputs[0], "Add")
    sigmoid = _producer(choice.inputs[0], "Sigmoid")
    probs, bias = sigmoid.outputs[0], _input(choice, 1)
    if (
        probs.dtype != ir.DataType.FLOAT
        or bias.dtype != ir.DataType.FLOAT
        or bias is not graph.initializers.get(root + ".gate.e_score_correction_bias")
        or not math.isfinite(layer.routed_scaling_factor)
    ):
        raise ValueError(f"Unsupported packed Bolt full-score routing: {root}")
    # Validate the selected-weight expression before moving selection into MoEPacked.
    weights = routing
    if layer.routed_scaling_factor != 1.0:  # noqa: RUF069
        scaled = _producer(weights, "Mul")
        factor = _input(scaled, 1).const_value
        if factor is None or factor.numpy().item() != np.float32(layer.routed_scaling_factor):
            raise ValueError(f"Packed Bolt routing scale disagrees with configuration: {root}")
        weights = _input(scaled, 0)
    if layer.norm_topk_prob:
        normalized = _producer(weights, "Div")
        denominator = _producer(normalized.inputs[1], "Add")
        total = _producer(denominator.inputs[0], "ReduceSum")
        epsilon, axes = _input(denominator, 1).const_value, _input(total, 1).const_value
        weights = _input(normalized, 0)
        if (
            total.inputs[0] is not weights
            or total.attributes.get_int("keepdims", 1) != 1
            or axes is None
            or axes.numpy().tolist() != [-1]
            or epsilon is None
            or epsilon.numpy().item() != np.float32(1e-20)
        ):
            raise ValueError(f"Unsupported packed Bolt routing normalization: {root}")
    gather = _producer(weights, "GatherElements")
    if (
        gather.inputs[0] is not probs
        or gather.inputs[1] is not selected
        or gather.attributes.get_int("axis", 0) != -1
    ):
        raise ValueError(f"Packed Bolt mixing must gather unbiased sigmoid values: {root}")
    banks = []
    for projection in ("up_proj", "down_proj"):
        sources = [
            graph.initializers[f"{root}.experts.{e}.{projection}.weight"]
            for e in range(experts)
        ]
        banks.append(
            _packed_initializer(
                graph,
                f"{root}.packed_{projection}_weight",
                sources,
                iq2=True,
                bank=True,
            )
        )
    if banks[0].shape != ir.Shape([experts, intermediate, 4 * 66]) or banks[
        1
    ].shape != ir.Shape([experts, 1024, intermediate // 256 * 66]):
        raise ValueError(f"Packed expert dimensions disagree with configuration: {root}")

    def emit(op):
        # Keep the original FP32 router projection; expose full scores without narrowing.
        logits = op.Reshape(_input(sigmoid, 0), [-1, experts])
        legacy, scores, mixing = op.MoEGroupRouter(
            logits,
            bias,
            top_k=topk,
            _domain="com.microsoft",
            _outputs=["", "selection_scores", "router_weights"],
        )
        legacy.name = ""
        scores.dtype = mixing.dtype = ir.DataType.FLOAT
        scores.shape = mixing.shape = ir.Shape([None, experts])
        normalize = int(layer.norm_topk_prob)
        if layer.routed_scaling_factor != 1.0:  # noqa: RUF069
            # Scale FP32 coefficients before the consumer's BF16 output store.
            # The shared API has no pre-store scale attribute. For normalized
            # routing, prepare a row denominator from the same full selection
            # scores; MoEPacked still selects and gathers its own expert routes.
            if normalize:
                # Match native lower-index tie breaking and rank-ordered FP32
                # addition; CUDA TopK and tree reductions can change the denominator.
                remaining = scores
                total = None
                mask = None
                for rank in range(topk):
                    chosen = op.ArgMax(remaining, axis=-1, keepdims=1, select_last_index=0)
                    chosen_weight = op.GatherElements(mixing, chosen, axis=-1)
                    total = chosen_weight if total is None else op.Add(total, chosen_weight)
                    if rank + 1 < topk:
                        if mask is None:
                            mask = op.ConstantOfShape(
                                op.Shape(chosen),
                                value=ir.tensor([-float("inf")], dtype=ir.DataType.FLOAT),
                            )
                        remaining = op.ScatterElements(remaining, chosen, mask, axis=-1)
                mixing = op.Div(mixing, op.Add(total, 1e-20))
            mixing = op.Mul(mixing, layer.routed_scaling_factor)
            normalize = 0
        result = op.MoEPacked(
            latent,
            scores,
            banks[0],
            None,
            banks[1],
            mixing,
            k=topk,
            normalize_routing_weights=normalize,
            activation_type="relu2",
            gate_type=16,
            down_type=16,
            _domain="com.microsoft",
        )
        result.dtype, result.shape = ir.DataType.BFLOAT16, latent.shape
        return result

    _replace(graph, anchor, emit)


def _pack_projection(
    graph: ir.Graph,
    node: ir.Node,
    packed_weights: dict[ir.Value, list[ir.Value]] | None = None,
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
        if packed_weights is not None and source in packed_weights:
            inputs = packed_weights[source]
        else:
            inputs = [
                _packed_initializer(
                    graph,
                    source.name + ".packed_weight",
                    [source],
                    iq2=iq2,
                    bank=False,
                )
            ]
            if packed_weights is not None:
                packed_weights[source] = inputs
        bias = node.inputs[2] if len(node.inputs) > 2 else None
    else:
        # NVFP4 uses a different nibble/scale layout; retain its existing native op.
        return False

    def emit(op):
        def project(body_op, rows):
            result = body_op.MatMulPacked(
                rows, *inputs, K=k, N=n, packed_type=16 if iq2 else 8, _domain="com.microsoft"
            )
            result.dtype = ir.DataType.BFLOAT16
            result.shape = ir.Shape([rows.shape[0], n]) if rows.shape is not None else None
            return result

        result = packed_row_tiles(
            op, x, input_width=k, output_width=n, tile_rows=8, project=project
        )
        return op.Add(result, bias) if bias is not None else result

    _replace(graph, node, emit)
    return True


def optimize_nemotron_h_omni_cuda(package: ModelPackage) -> ModelPackage:
    """Return a compact decoder, leaving the source package and media untouched.

    Must run after the native source-accounted loader. Complete packed blocks
    are concatenated lazily; no checkpoint reads or dense reconstruction occur.
    Requires a custom ORT build containing the packed projection and MoE ops.
    """
    config = package.config
    if not isinstance(config, NemotronHOmniConfig) or config.dtype != ir.DataType.BFLOAT16:
        raise ValueError("Packed Bolt CUDA export requires NemotronHOmniConfig and BF16")
    if "decoder" not in package or not config.quantized_projections:
        raise ValueError("Packed Bolt CUDA export requires a native split decoder")
    source = package["decoder"]
    if source.metadata_props.get("mobius.bolt.packed_cuda") == "3":
        return package
    if source.metadata_props.get("mobius.bolt.packed_cuda"):
        raise ValueError("Older packed CUDA API: re-export from the source-accounted package")
    for value in source.graph.initializers.values():
        if value.const_value is None:
            raise ValueError(f"Packed Bolt CUDA export requires loaded weights: {value.name}")
    compact = source.clone()
    graph = compact.graph
    layers = [i for i, kind in enumerate(config.layer_types or []) if kind == "moe"]
    if not layers:
        raise ValueError("Packed Bolt CUDA export requires routed expert layers")
    for index in layers:
        _pack_moe(graph, config, index)
    common_passes.RemoveUnusedNodesPass()(compact)
    packed_weights: dict[ir.Value, list[ir.Value]] = {}
    projections = sum(
        _pack_projection(graph, node, packed_weights)
        for node in list(graph)
        if node.domain == "com.microsoft"
        and node.op_type in {"MatMulGGMLIQ2XXS", "MatMulGGMLQ8_0", "MatMulNVFP4A16"}
    )
    common_passes.RemoveUnusedNodesPass()(compact)
    compact.metadata_props["mobius.bolt.packed_cuda"] = "3"
    compact.metadata_props["mobius.bolt.mamba_backend"] = "unchanged"
    compact.metadata_props["mobius.bolt.routing_backend"] = "full_score_fp32"
    result = ModelPackage({**package, "decoder": compact}, config=config)
    result.weight_loading_report = {
        **(package.weight_loading_report or {}),
        "packed_cuda": {
            "mamba": "unchanged",
            "routing": "full_score_fp32",
            "full_checkpoint_dequantization": False,
            "moe_layers": len(layers),
            "projections": projections,
            "decoder_nodes_before": len(source.graph),
            "decoder_nodes_after": len(graph),
            "operators": dict(Counter(node.op_type for node in graph.all_nodes())),
        },
    }
    return result
