# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fuse a dense-fallback native-block MoE subgraph into ``pkg.nxrt::BlockQuantizedMoE``.

GGUF-imported MoE checkpoints (GLM-5.2 UD-IQ1_S/M, DeepSeek, ...) are exported
as a *dense fallback*: every routed expert is materialised as its own
``gate_proj`` / ``up_proj`` / ``down_proj`` ``pkg.nxrt::BlockQuantizedMatMul``
MLP and the router picks the active experts with an
``Equal``/``CastLike``/``Mul``/``ReduceSum`` mask applied *after* every expert
has already been computed. Native decode therefore reads all ``E`` experts'
weights per token even though only ``k`` are selected.

The ``pkg.nxrt::BlockQuantizedMoE`` kernel does sparse top-k decode (reads only
the selected experts' native blocks). This module rewrites the dense-fallback
subgraph, one MoE layer at a time, into a single ``BlockQuantizedMoE`` node that
**reuses the existing native blocks byte-for-byte** -- it is a pure expert-major
stack/concat layout transform, never a dequantize-then-requantize. No scales or
zero-points exist: IQ/native blocks are self-contained (codebook + embedded
scale), so only the packed ``uint8`` weight bytes move.

Why this fusion is separate from :func:`fuse_dense_moe_to_qmoe`:

* Experts are ``BlockQuantizedMatMul`` (native blocks) not ``MatMulNBits``
  (int4 affine); there are no scale/zero-point inputs to carry.
* GLM-5.2 UD-IQ1 quantises the ``gate``/``up`` projections and the ``down``
  projection to *different* native formats (e.g. ``iq1_s`` gate/up, ``iq4_xs``
  down). ``BlockQuantizedMoE`` expresses this with ``block_layout_version=2``
  and per-projection ``fc1_format`` / ``fc2_format`` / ``fc3_format`` attributes
  (a uniform-format layer stays on ``block_layout_version=1``).
* Routing is reconstructed **gate-agnostically** from the already-computed
  ``selected_experts`` / ``routing_weights`` tensors, so a sigmoid+bias+scaling
  GLM gate fuses identically to a plain softmax top-k gate. The routing decision
  is re-materialised as a ``[tokens, experts]`` scatter and fed to the kernel as
  both ``router_logits`` (a one-hot selection mask) and ``router_weights`` (the
  final per-expert weights) with ``normalize_routing_weights=0`` -- the kernel
  re-selects exactly the same experts and gathers exactly the same weights.

Any surrounding **shared expert** is left untouched: only the routed per-expert
storm and its router/mask are fused, and the final ``Add(routed_sum, shared)``
is rewired to consume the ``BlockQuantizedMoE`` output.

The rewrite is **property-gated and fails closed**: a layer that is not a native
dense-expert storm, or whose experts use a mix of native formats that a single
expert-major bank cannot express, is left as the runnable dense fallback with a
typed reason logged. It never silently emits a node that would dense-evaluate
every expert.

Example::

    import onnx_ir as ir
    from mobius.rewrite_rules import fuse_block_quantized_moe

    model = ir.load("decoder/model.onnx")
    fused = fuse_block_quantized_moe(model)
    print(f"fused {fused} native-block MoE layers")
    ir.save(model, "decoder-bqmoe/model.onnx")
"""

from __future__ import annotations

import logging

import numpy as np
import onnx_ir as ir

logger = logging.getLogger(__name__)

_NXRT_DOMAIN = "pkg.nxrt"
_MATMUL = "BlockQuantizedMatMul"
_MOE = "BlockQuantizedMoE"

# ``BlockQuantizedMatMul`` native block geometry, mirroring
# ``mobius.components._quantized_linear._NATIVE_BLOCK_FORMATS``: format name ->
# (block_elements, block_bytes). Kept local so the fusion validates the packed
# weight shape it stacks without importing the component module.
_NATIVE_BLOCK_FORMATS = {
    "mxfp4": (32, 17),
    "iq4_nl": (32, 18),
    "iq4_xs": (256, 136),
    "iq3_s": (256, 110),
    "iq3_xxs": (256, 98),
    "iq2_xxs": (256, 66),
    "iq2_xs": (256, 74),
    "iq2_s": (256, 82),
    "iq1_s": (256, 50),
    "iq1_m": (256, 56),
}


class _UnfusableError(Exception):
    """A layer cannot be expressed as one ``BlockQuantizedMoE`` node.

    Carries a typed, human-readable reason so the caller logs precisely why the
    dense fallback was preserved instead of silently dense-evaluating experts.
    """


def _scalar_int(value: ir.Value | None) -> int | None:
    """Return the integer a value carries, whether an initializer or a ``Constant``."""
    if value is None:
        return None
    const = value.const_value
    if const is None:
        producer = value.producer()
        if producer is None or producer.op_type != "Constant":
            return None
        for attr in ("value", "value_int", "value_ints"):
            if attr in producer.attributes:
                const = producer.attributes[attr].value
                break
        else:
            return None
    arr = np.asarray(const.numpy() if hasattr(const, "numpy") else const).reshape(-1)
    if arr.size == 0:
        return None
    return int(arr[0])


def _consumers_of_type(value: ir.Value, op_type: str) -> list[ir.Node]:
    return [node for node, _ in value.uses() if node.op_type == op_type]


def _single_consumer(value: ir.Value, *op_types: str) -> ir.Node | None:
    for op_type in op_types:
        matches = _consumers_of_type(value, op_type)
        if matches:
            return matches[0]
    return None


def _array(value: ir.Value) -> np.ndarray:
    const = value.const_value
    if const is None:
        raise _UnfusableError(f"expected an initializer for {value.name!r}")
    return np.asarray(const.numpy())


def _make_initializer(
    graph: ir.Graph, name: str, arr: np.ndarray, dtype: ir.DataType
) -> ir.Value:
    tensor = ir.tensor(arr, name=name, dtype=dtype)
    value = ir.Value(
        name=name,
        shape=ir.Shape(arr.shape),
        type=ir.TensorType(dtype),
        const_value=tensor,
    )
    graph.register_initializer(value)
    return value


def _skip_cast(value: ir.Value) -> ir.Value:
    """Return ``value``'s source through a single ``Cast``, if one produces it.

    ``BlockQuantizedLinear`` casts its activation to FLOAT before the matmul and
    (for non-FLOAT models) casts the result back afterwards, so every projection
    boundary in an f16 export is wrapped in a ``Cast``. An f32 export has none.
    Walking through one ``Cast`` makes the trace dtype-agnostic.
    """
    producer = value.producer()
    if producer is not None and producer.op_type == "Cast" and producer.inputs[0] is not None:
        return producer.inputs[0]
    return value


def _producer_through_cast(value: ir.Value, op_type: str) -> ir.Node | None:
    source = _skip_cast(value)
    producer = source.producer()
    if producer is None or producer.op_type != op_type:
        return None
    return producer


class _ExpertProjections:
    """The three ``BlockQuantizedMatMul`` nodes making up one expert MLP."""

    __slots__ = ("down", "gate", "up")

    def __init__(self, gate: ir.Node, up: ir.Node, down: ir.Node) -> None:
        self.gate = gate
        self.up = up
        self.down = down


def _trace_expert(down: ir.Node) -> _ExpertProjections | None:
    """Walk back from a ``down_proj`` ``BlockQuantizedMatMul`` to gate/up.

    Expert MLP shape (SwiGLU): ``down(Swish(gate) * up)``. The legacy
    ``gate * Sigmoid(gate)`` decomposition remains accepted for imported graphs.
    Every projection boundary is walked through an optional ``Cast`` so both f32
    and f16 dense exports trace identically.
    """
    act_mul = _producer_through_cast(down.inputs[0], "Mul")
    if act_mul is None:
        return None
    a, b = act_mul.inputs[0], act_mul.inputs[1]
    # One operand is the up projection, the other the SiLU-activated gate.
    if _producer_through_cast(a, _MATMUL) is not None:
        up_out, act_out = a, b
    elif _producer_through_cast(b, _MATMUL) is not None:
        up_out, act_out = b, a
    else:
        return None
    up = _producer_through_cast(up_out, _MATMUL)
    act = _skip_cast(act_out).producer()
    if act is None:
        return None
    if act.op_type == "Swish":
        gate = _producer_through_cast(act.inputs[0], _MATMUL)
        if gate is None:
            return None
        return _ExpertProjections(gate, up, down)
    if act.op_type != "Mul":
        return None
    # Accept the legacy gate * Sigmoid(gate) decomposition.
    x, y = act.inputs[0], act.inputs[1]
    for gate_out, sigmoid_out in ((x, y), (y, x)):
        gate = _producer_through_cast(gate_out, _MATMUL)
        sigmoid = _skip_cast(sigmoid_out).producer()
        if (
            gate is not None
            and sigmoid is not None
            and sigmoid.op_type == "Sigmoid"
            and _producer_through_cast(sigmoid.inputs[0], _MATMUL) is gate
        ):
            return _ExpertProjections(gate, up, down)
    return None


def _find_moe_selectors(graph: ir.Graph) -> list[ir.Value]:
    """Return each ``selected_experts`` value driving a dense-fallback expert storm.

    Gate-agnostic anchor: a value consumed by two or more ``Equal`` nodes whose
    other operand is an integer constant is the ``selected_experts`` tensor of a
    per-expert routing mask, regardless of which gate (softmax / sigmoid+bias /
    sparse-mixer) produced it.
    """
    selectors: list[ir.Value] = []
    seen: set[int] = set()
    for node in graph:
        if node.op_type != "Equal":
            continue
        selected = node.inputs[0]
        if selected is None or _scalar_int(node.inputs[1]) is None:
            continue
        if len(_consumers_of_type(selected, "Equal")) < 2:
            continue
        key = id(selected)
        if key not in seen:
            seen.add(key)
            selectors.append(selected)
    return selectors


class _DenseMoELayer:
    """Everything the rewrite needs from one dense-fallback native-block MoE."""

    def __init__(self, selected_experts: ir.Value) -> None:
        self.selected_experts = selected_experts
        self.routing_weights: ir.Value | None = None
        self.experts: dict[int, _ExpertProjections] = {}
        self.contributions: list[ir.Value] = []
        self.routed_out: ir.Value | None = None
        self.k: int | None = None
        self._collect_experts()
        self._resolve_k()
        self._find_routed_output()

    def _collect_experts(self) -> None:
        for equal in _consumers_of_type(self.selected_experts, "Equal"):
            expert_id = _scalar_int(equal.inputs[1])
            if expert_id is None:
                continue
            cast = _single_consumer(equal.outputs[0], "CastLike", "Cast")
            if cast is None:
                continue
            weight_mul = _single_consumer(cast.outputs[0], "Mul")
            if weight_mul is None:
                continue
            reduce_sum = _single_consumer(weight_mul.outputs[0], "ReduceSum")
            if reduce_sum is None:
                continue
            contribution = _single_consumer(reduce_sum.outputs[0], "Mul")
            if contribution is None:
                continue
            routing_weights = (
                weight_mul.inputs[1]
                if weight_mul.inputs[0] is cast.outputs[0]
                else weight_mul.inputs[0]
            )
            weight_out = reduce_sum.outputs[0]
            expert_out = (
                contribution.inputs[1]
                if contribution.inputs[0] is weight_out
                else contribution.inputs[0]
            )
            down = _producer_through_cast(expert_out, _MATMUL)
            if down is None:
                continue
            projections = _trace_expert(down)
            if projections is None:
                continue
            # All experts must share one ``routing_weights`` tensor (they do in
            # the dense loop); a mismatch means this is not a single storm.
            if self.routing_weights is None:
                self.routing_weights = routing_weights
            elif self.routing_weights is not routing_weights:
                continue
            self.experts[expert_id] = projections
            self.contributions.append(contribution.outputs[0])

    def _resolve_k(self) -> None:
        shape = self.selected_experts.shape
        if shape is not None and len(shape) >= 1:
            last = shape[-1]
            if isinstance(last, int):
                self.k = last
                return
        producer = self.selected_experts.producer()
        if producer is not None and producer.op_type == "TopK" and len(producer.inputs) > 1:
            self.k = _scalar_int(producer.inputs[1])

    def _find_routed_output(self) -> None:
        if not self.contributions:
            return
        contrib_set = set(self.contributions)
        acc_outputs: set[ir.Value] = set()
        cur = self.contributions[0]
        advanced = True
        while advanced:
            advanced = False
            for node, index in cur.uses():
                if node.op_type != "Add":
                    continue
                other = node.inputs[1 - index]
                if other in contrib_set or other in acc_outputs:
                    cur = node.outputs[0]
                    acc_outputs.add(cur)
                    advanced = True
                    break
        self.routed_out = cur

    @property
    def is_candidate(self) -> bool:
        if self.routing_weights is None or self.routed_out is None:
            return False
        ids = sorted(self.experts)
        return ids == list(range(len(ids))) and len(ids) >= 1


def _projection_format(node: ir.Node, role: str) -> str:
    fmt = node.attributes.get("format")
    if fmt is None:
        raise _UnfusableError(f"{role} projection {node.name!r} has no 'format' attribute")
    name = fmt.value
    if name not in _NATIVE_BLOCK_FORMATS:
        raise _UnfusableError(f"{role} projection uses unknown native format {name!r}")
    return name


def _uniform_format(nodes: list[ir.Node], role: str) -> str:
    formats = {_projection_format(node, role) for node in nodes}
    if len(formats) != 1:
        raise _UnfusableError(
            f"{role} projections use mixed native formats {sorted(formats)}; "
            "a single expert-major bank requires one format per projection"
        )
    return next(iter(formats))


def _require_no_bias(nodes: list[ir.Node], role: str) -> None:
    for node in nodes:
        if len(node.inputs) > 2 and node.inputs[2] is not None:
            raise _UnfusableError(
                f"{role} projection {node.name!r} carries a bias; biased native "
                "MoE experts are not supported by this fusion yet"
            )


def _stack_weights(nodes: list[ir.Node], out_features: int, role: str) -> np.ndarray:
    """Stack per-expert packed ``[N, n_blocks, block_bytes]`` uint8 into ``[E, ...]``.

    The native block bytes are copied verbatim: this is an expert-major stack,
    never a requantization.
    """
    stacked = []
    for node in nodes:
        weight = _array(node.inputs[1])
        if weight.ndim != 3 or weight.shape[0] != out_features:
            raise _UnfusableError(
                f"{role} projection weight has shape {weight.shape}, expected "
                f"[{out_features}, n_blocks, block_bytes]"
            )
        stacked.append(weight)
    shapes = {w.shape for w in stacked}
    if len(shapes) != 1:
        raise _UnfusableError(f"{role} projections have ragged packed shapes {sorted(shapes)}")
    return np.stack(stacked, axis=0)


def _weight_out_features(node: ir.Node) -> int:
    return int(_array(node.inputs[1]).shape[0])


def _flat_last_dim_shape(
    graph: ir.Graph, source: ir.Value, prefix: str, neg_one: ir.Value
) -> tuple[list[ir.Node], ir.Value]:
    """Build the ``[-1, last_dim]`` shape tensor for flattening ``source`` to 2D."""
    last = ir.node(
        "Shape",
        inputs=[source],
        attributes={"start": -1},
        num_outputs=1,
        name=f"{prefix}.last_dim",
    )
    concat = ir.node(
        "Concat",
        inputs=[neg_one, last.outputs[0]],
        attributes={"axis": 0},
        num_outputs=1,
        name=f"{prefix}.flat_shape",
    )
    return [last, concat], concat.outputs[0]


def _build_routing(
    graph: ir.Graph, layer: _DenseMoELayer, experts: int, prefix: str
) -> tuple[list[ir.Node], ir.Value, ir.Value]:
    """Reconstruct gate-agnostic routing tensors for ``BlockQuantizedMoE``.

    Returns the new nodes plus ``(router_logits, router_weights)``, each a
    ``[tokens, experts]`` FLOAT tensor. ``router_logits`` is a one-hot selection
    mask (1.0 at each selected expert) so the kernel's top-k re-selects exactly
    the dense loop's experts; ``router_weights`` scatters the dense per-expert
    weights so that (with ``normalize_routing_weights=0``) the kernel gathers
    exactly those weights. The result is independent of the originating gate.
    """
    neg_one = _make_initializer(
        graph, f"{prefix}.neg_one", np.array([-1], dtype=np.int64), ir.DataType.INT64
    )
    expert_dim = _make_initializer(
        graph, f"{prefix}.experts", np.array([experts], dtype=np.int64), ir.DataType.INT64
    )
    zero_f = _make_initializer(
        graph, f"{prefix}.zero", np.array(0.0, dtype=np.float32), ir.DataType.FLOAT
    )
    one_f = _make_initializer(
        graph, f"{prefix}.one", np.array(1.0, dtype=np.float32), ir.DataType.FLOAT
    )

    nodes: list[ir.Node] = []

    sel_shape_nodes, sel_flat_shape = _flat_last_dim_shape(
        graph, layer.selected_experts, f"{prefix}.sel", neg_one
    )
    nodes.extend(sel_shape_nodes)
    sel2d = ir.node(
        "Reshape",
        inputs=[layer.selected_experts, sel_flat_shape],
        num_outputs=1,
        name=f"{prefix}.sel2d",
    )
    nodes.append(sel2d)

    rw_shape_nodes, rw_flat_shape = _flat_last_dim_shape(
        graph, layer.routing_weights, f"{prefix}.rw", neg_one
    )
    nodes.extend(rw_shape_nodes)
    rw2d = ir.node(
        "Reshape",
        inputs=[layer.routing_weights, rw_flat_shape],
        num_outputs=1,
        name=f"{prefix}.rw2d",
    )
    nodes.append(rw2d)
    if layer.routing_weights.dtype == ir.DataType.FLOAT:
        rw_float = rw2d.outputs[0]
    else:
        rw_cast = ir.node(
            "Cast",
            inputs=[rw2d.outputs[0]],
            attributes={"to": ir.DataType.FLOAT.value},
            num_outputs=1,
            name=f"{prefix}.rw_float",
        )
        nodes.append(rw_cast)
        rw_float = rw_cast.outputs[0]

    # rows = selected_experts flattened leading dim.
    rows = ir.node(
        "Shape",
        inputs=[sel2d.outputs[0]],
        attributes={"start": 0, "end": 1},
        num_outputs=1,
        name=f"{prefix}.rows",
    )
    nodes.append(rows)
    dense_shape = ir.node(
        "Concat",
        inputs=[rows.outputs[0], expert_dim],
        attributes={"axis": 0},
        num_outputs=1,
        name=f"{prefix}.dense_shape",
    )
    nodes.append(dense_shape)
    zeros = ir.node(
        "Expand",
        inputs=[zero_f, dense_shape.outputs[0]],
        num_outputs=1,
        name=f"{prefix}.zeros",
    )
    nodes.append(zeros)
    sel_kshape = ir.node(
        "Shape", inputs=[sel2d.outputs[0]], num_outputs=1, name=f"{prefix}.sel_kshape"
    )
    nodes.append(sel_kshape)
    ones = ir.node(
        "Expand",
        inputs=[one_f, sel_kshape.outputs[0]],
        num_outputs=1,
        name=f"{prefix}.ones",
    )
    nodes.append(ones)

    logits = ir.node(
        "ScatterElements",
        inputs=[zeros.outputs[0], sel2d.outputs[0], ones.outputs[0]],
        attributes={"axis": 1},
        num_outputs=1,
        name=f"{prefix}.router_logits",
    )
    logits.outputs[0].name = f"{prefix}.router_logits"
    nodes.append(logits)
    weights = ir.node(
        "ScatterElements",
        inputs=[zeros.outputs[0], sel2d.outputs[0], rw_float],
        attributes={"axis": 1},
        num_outputs=1,
        name=f"{prefix}.router_weights",
    )
    weights.outputs[0].name = f"{prefix}.router_weights"
    nodes.append(weights)
    return nodes, logits.outputs[0], weights.outputs[0]


def _fuse_layer(graph: ir.Graph, layer: _DenseMoELayer, index: int) -> None:
    ids = sorted(layer.experts)
    gate_nodes = [layer.experts[i].gate for i in ids]
    up_nodes = [layer.experts[i].up for i in ids]
    down_nodes = [layer.experts[i].down for i in ids]
    experts = len(ids)

    _require_no_bias(gate_nodes, "gate")
    _require_no_bias(up_nodes, "up")
    _require_no_bias(down_nodes, "down")

    gate_fmt = _uniform_format(gate_nodes, "gate")
    up_fmt = _uniform_format(up_nodes, "up")
    down_fmt = _uniform_format(down_nodes, "down")

    inter = _weight_out_features(gate_nodes[0])
    hidden = _weight_out_features(down_nodes[0])
    if _weight_out_features(up_nodes[0]) != inter:
        raise _UnfusableError(
            "gate and up projections have different output widths "
            f"({inter} vs {_weight_out_features(up_nodes[0])})"
        )

    gate_w = _stack_weights(gate_nodes, inter, "gate")
    up_w = _stack_weights(up_nodes, inter, "up")
    down_w = _stack_weights(down_nodes, hidden, "down")

    prefix = f"bqmoe.layer{index}"
    fc3_w_v: ir.Value | None = None
    projection_formats: dict[str, str] = {"fc2": down_fmt}
    if gate_fmt == up_fmt:
        # Fused SwiGLU: FC1 = concat(gate_rows, up_rows) in one native format.
        # ``swiglu_fusion=2`` tells the kernel the first ``inter`` rows are the
        # gate half and the next ``inter`` the up half.
        swiglu_fusion = 2
        fc1_w = np.concatenate([gate_w, up_w], axis=1)
        projection_formats["fc1"] = gate_fmt
    else:
        # Unfused SwiGLU: gate/up in different native formats stay as FC1 (gate)
        # and FC3 (up); the kernel computes ``swiglu(fc1(x), fc3(x))``.
        swiglu_fusion = 0
        fc1_w = gate_w
        fc3_w_v = _make_initializer(
            graph, f"{prefix}.fc3_experts_weights", up_w, ir.DataType.UINT8
        )
        projection_formats["fc1"] = gate_fmt
        projection_formats["fc3"] = up_fmt

    fc1_w_v = _make_initializer(
        graph, f"{prefix}.fc1_experts_weights", fc1_w, ir.DataType.UINT8
    )
    fc2_w_v = _make_initializer(
        graph, f"{prefix}.fc2_experts_weights", down_w, ir.DataType.UINT8
    )

    routing_nodes, router_logits, router_weights = _build_routing(
        graph, layer, experts, prefix
    )

    moe_input = gate_nodes[0].inputs[0]

    attributes: dict[str, object] = {
        "k": layer.k,
        "activation_type": "swiglu",
        "normalize_routing_weights": 0,
        "swiglu_fusion": swiglu_fusion,
    }
    distinct = set(projection_formats.values())
    if len(distinct) == 1:
        attributes["format"] = next(iter(distinct))
    else:
        attributes["block_layout_version"] = 2
        # ``format`` is the required base/fallback; per-projection attributes
        # override it where a projection uses a different native format.
        attributes["format"] = projection_formats["fc1"]
        attributes["fc1_format"] = projection_formats["fc1"]
        attributes["fc2_format"] = projection_formats["fc2"]
        if "fc3" in projection_formats:
            attributes["fc3_format"] = projection_formats["fc3"]

    moe = ir.node(
        _MOE,
        inputs=[
            moe_input,
            router_logits,
            fc1_w_v,
            None,
            fc2_w_v,
            None,
            fc3_w_v,
            None,
            router_weights,
        ],
        attributes=attributes,
        domain=_NXRT_DOMAIN,
        num_outputs=1,
        name=f"{prefix}.bqmoe",
    )
    moe_out = moe.outputs[0]
    moe_out.name = f"{prefix}.bqmoe_output"
    moe_out.type = ir.TensorType(ir.DataType.FLOAT)
    moe_out.shape = moe_input.shape

    new_nodes = [*routing_nodes, moe]
    routed_dtype = layer.routed_out.dtype
    if routed_dtype not in (None, ir.DataType.FLOAT):
        cast_back = ir.node(
            "Cast",
            inputs=[moe_out],
            attributes={"to": routed_dtype.value},
            num_outputs=1,
            name=f"{prefix}.bqmoe_cast",
        )
        cast_back.outputs[0].name = f"{prefix}.bqmoe_output_cast"
        cast_back.outputs[0].type = ir.TensorType(routed_dtype)
        cast_back.outputs[0].shape = layer.routed_out.shape
        new_nodes.append(cast_back)
        final_out = cast_back.outputs[0]
    else:
        final_out = moe_out

    graph.insert_after(layer.routed_out.producer(), new_nodes)
    layer.routed_out.replace_all_uses_with(final_out)
    graph.opset_imports[_NXRT_DOMAIN] = 1


def _remove_dead_nodes(graph: ir.Graph) -> None:
    """Iteratively drop nodes whose outputs are unused and not graph outputs."""
    graph_outputs = set(graph.outputs)
    changed = True
    while changed:
        changed = False
        for node in reversed(list(graph)):
            if any(out in graph_outputs for out in node.outputs):
                continue
            if all(len(list(out.uses())) == 0 for out in node.outputs):
                graph.remove(node, safe=True)
                changed = True
    used: set[str] = set()
    for node in graph:
        for inp in node.inputs:
            if inp is not None and inp.name is not None:
                used.add(inp.name)
    for name in list(graph.initializers):
        if name not in used:
            del graph.initializers[name]


def fuse_block_quantized_moe(model: ir.Model) -> int:
    """Fuse every dense-fallback native-block MoE subgraph into ``BlockQuantizedMoE``.

    Reuses the existing ``BlockQuantizedMatMul`` native blocks byte-for-byte
    (pure expert-major stack/concat, no requantization). Routing is reconstructed
    gate-agnostically from the graph's ``selected_experts`` / ``routing_weights``
    tensors, so softmax, sigmoid+bias+scaling, and other top-k gates all fuse.
    Mixed per-projection native formats are expressed with
    ``block_layout_version=2``. A layer whose experts cannot be expressed as a
    single expert-major bank is left as the runnable dense fallback with a typed
    reason logged -- the fusion never silently dense-evaluates every expert.

    Args:
        model: The IR model to rewrite in place.

    Returns:
        The number of MoE layers fused.
    """
    graph = model.graph
    selectors = _find_moe_selectors(graph)
    layers = [_DenseMoELayer(selected) for selected in selectors]
    fused = 0
    for index, layer in enumerate(layers):
        if not layer.is_candidate:
            logger.warning(
                "skipping native MoE at %s: unrecognised dense-fallback structure",
                layer.selected_experts.name,
            )
            continue
        if layer.k is None or layer.k < 1:
            logger.warning(
                "skipping native MoE at %s: could not determine top-k",
                layer.selected_experts.name,
            )
            continue
        try:
            _fuse_layer(graph, layer, index)
        except _UnfusableError as reason:
            logger.warning(
                "skipping native MoE at %s: %s; keeping dense fallback",
                layer.selected_experts.name,
                reason,
            )
            continue
        fused += 1
    if fused:
        _remove_dead_nodes(graph)
        graph.sort()
    return fused
