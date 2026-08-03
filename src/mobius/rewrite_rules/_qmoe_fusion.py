# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fuse a dense-fallback Qwen35/DeepSeek-style MoE subgraph into ``com.microsoft::QMoE``.

Some int4-quantized MoE checkpoints are exported (or Olive-graph quantized) as a
*dense fallback*: every expert is materialised as its own ``gate_proj`` /
``up_proj`` / ``down_proj`` ``MatMulNBits`` MLP and the router picks the active
experts with an ``Equal``/``CastLike``/``Mul``/``ReduceSum`` mask that is applied
*after* every expert has already been computed. Native decode therefore runs all
``E`` experts per token (full MoE weight traffic) even though only ``k`` are
selected.

The CUDA/CPU ``com.microsoft::QMoE`` kernel does sparse top-k decode (reads only
``k`` experts' weights). This module rewrites the dense-fallback subgraph, one MoE
layer at a time, into a single ``QMoE`` node that **reuses the existing int4
weights byte-for-byte** -- it is a pure expert-major concat/layout transform, not
a requantization:

* ``fc1_experts_weights`` = per-expert ``concat(flatten(gate_w), flatten(up_w))``
  (the packed ``uint8`` bytes are copied unchanged; ``swiglu_fusion=2`` tells the
  kernel the gate half precedes the up half).
* ``fc2_experts_weights`` = per-expert ``flatten(down_w)`` (bytes unchanged).
* scales are upcast ``float16 -> float32`` (value-exact, lossless) because the
  QMoE kernel requires ``float32`` scales and ``router_probs``.
* zero-points are copied unchanged (same low-nibble-first packing as
  ``MatMulNBits``).

The router logits (a quantized ``MatMulNBits`` gate) are ``Cast`` to ``float32``
and passed as ``router_probs`` with ``normalize_routing_weights=1`` -- matching
:class:`mobius.components._moe.TopKGate` (``Softmax(TopK(logits))``).

Any surrounding **shared expert** (Qwen2-MoE style ``shared_expert`` +
``shared_expert_gate``) is left untouched: only the routed per-expert storm and
its router/mask are fused, and the final ``Add(routed_sum, shared)`` is rewired to
consume the ``QMoE`` output.

Example::

    import onnx_ir as ir
    from mobius.rewrite_rules import fuse_dense_moe_to_qmoe

    model = ir.load("decoder/model.onnx")
    fused = fuse_dense_moe_to_qmoe(model)
    print(f"fused {fused} MoE layers")
    ir.save(model, "decoder-qmoe/model.onnx")
"""

from __future__ import annotations

import logging

import numpy as np
import onnx_ir as ir

logger = logging.getLogger(__name__)

_MS_DOMAIN = "com.microsoft"


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


def _require_producer(value: ir.Value, op_type: str) -> ir.Node | None:
    producer = value.producer()
    if producer is None or producer.op_type != op_type:
        return None
    return producer


def _array(value: ir.Value) -> np.ndarray:
    const = value.const_value
    if const is None:
        raise ValueError(f"expected an initializer for {value.name!r}")
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


class _ExpertProjections:
    """The three ``MatMulNBits`` nodes making up one expert MLP."""

    __slots__ = ("down", "gate", "up")

    def __init__(self, gate: ir.Node, up: ir.Node, down: ir.Node) -> None:
        self.gate = gate
        self.up = up
        self.down = down


def _trace_expert(down: ir.Node) -> _ExpertProjections | None:
    """Walk back from a ``down_proj`` ``MatMulNBits`` to its gate/up projections.

    Expert MLP shape (SwiGLU): ``down(  (gate*sigmoid(gate)) * up  )``.
    """
    act_mul = _require_producer(down.inputs[0], "Mul")
    if act_mul is None:
        return None
    a, b = act_mul.inputs[0], act_mul.inputs[1]
    # one operand is the up projection (MatMulNBits), the other the SiLU Mul.
    if a.producer() and a.producer().op_type == "MatMulNBits":
        up_out, silu_out = a, b
    elif b.producer() and b.producer().op_type == "MatMulNBits":
        up_out, silu_out = b, a
    else:
        return None
    up = up_out.producer()
    silu = silu_out.producer()
    if silu is None or silu.op_type != "Mul":
        return None
    x, y = silu.inputs[0], silu.inputs[1]
    # gate_out feeds both the SiLU Mul directly and a Sigmoid.
    if x.producer() and x.producer().op_type == "MatMulNBits":
        gate = x.producer()
    elif y.producer() and y.producer().op_type == "MatMulNBits":
        gate = y.producer()
    else:
        return None
    return _ExpertProjections(gate, up, down)


def _router_gate(logits: ir.Value) -> ir.Node | None:
    """Resolve the quantized gate ``MatMulNBits`` behind the router logits.

    Some exported graphs (e.g. the ``merged`` variant) insert a ``Cast`` between
    the gate ``MatMulNBits`` and ``TopK``; walk through it transparently.
    """
    producer = logits.producer()
    if producer is None:
        return None
    if producer.op_type == "Cast":
        producer = producer.inputs[0].producer() if producer.inputs[0] is not None else None
    if producer is None or producer.op_type != "MatMulNBits":
        return None
    return producer


def _find_moe_layers(graph: ir.Graph) -> list[ir.Node]:
    """Return the ``TopK`` anchor node of every dense-fallback MoE layer."""
    anchors: list[ir.Node] = []
    for node in graph:
        if node.op_type != "TopK":
            continue
        logits = node.inputs[0]
        if logits is None or _router_gate(logits) is None:
            continue
        indices = node.outputs[1]
        if not _consumers_of_type(indices, "Equal"):
            continue
        anchors.append(node)
    return anchors


class _DenseMoELayer:
    """Everything the rewrite needs from one dense-fallback MoE subgraph."""

    def __init__(self, topk: ir.Node) -> None:
        self.topk = topk
        self.gate_router = _router_gate(topk.inputs[0])
        self.hidden = self.gate_router.inputs[0]
        self.logits = topk.inputs[0]
        self.k = _scalar_int(topk.inputs[1])
        softmax = _single_consumer(topk.outputs[0], "Softmax", "Cast")
        if softmax is not None and softmax.op_type == "Cast":
            softmax = _single_consumer(softmax.outputs[0], "Softmax")
        self.softmax = softmax
        self.experts: dict[int, _ExpertProjections] = {}
        self.contributions: list[ir.Value] = []
        self.routed_out: ir.Value | None = None
        self._collect_experts()
        self._find_routed_output()

    def _collect_experts(self) -> None:
        indices = self.topk.outputs[1]
        for equal in _consumers_of_type(indices, "Equal"):
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
            weight_out = reduce_sum.outputs[0]
            expert_out = (
                contribution.inputs[1]
                if contribution.inputs[0] is weight_out
                else contribution.inputs[0]
            )
            down = _require_producer(expert_out, "MatMulNBits")
            if down is None:
                continue
            projections = _trace_expert(down)
            if projections is None:
                continue
            self.experts[expert_id] = projections
            self.contributions.append(contribution.outputs[0])

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
    def is_valid(self) -> bool:
        if self.k is None or self.softmax is None or self.routed_out is None:
            return False
        ids = sorted(self.experts)
        return ids == list(range(len(ids))) and len(ids) >= 1


def _expert_geometry(down: ir.Node) -> tuple[int, int, int]:
    """Return ``(bits, block_size, has_zero_point)`` from a ``MatMulNBits`` node."""
    bits = int(down.attributes["bits"].value)
    block_size = int(down.attributes["block_size"].value)
    has_zp = len(down.inputs) > 3 and down.inputs[3] is not None
    return bits, block_size, has_zp


def _pack_projection(nodes: list[ir.Node], slot: int) -> np.ndarray:
    """Stack ``flatten(-2)`` of one ``MatMulNBits`` weight input across experts."""
    stacked = [
        _array(node.inputs[slot]).reshape(_array(node.inputs[slot]).shape[0], -1)
        for node in nodes
    ]
    return np.stack(stacked, axis=0)


def _pack_scales(nodes: list[ir.Node], slot: int, out_features: int) -> np.ndarray:
    stacked = []
    for node in nodes:
        arr = _array(node.inputs[slot]).reshape(out_features, -1)
        stacked.append(arr.astype(np.float32))
    return np.stack(stacked, axis=0)


def _pack_zero_points(nodes: list[ir.Node], slot: int, out_features: int) -> np.ndarray:
    stacked = [
        _array(node.inputs[slot]).reshape(out_features, -1).astype(np.uint8) for node in nodes
    ]
    return np.stack(stacked, axis=0)


def _fuse_layer(graph: ir.Graph, layer: _DenseMoELayer, index: int) -> None:
    ids = sorted(layer.experts)
    gate_nodes = [layer.experts[i].gate for i in ids]
    up_nodes = [layer.experts[i].up for i in ids]
    down_nodes = [layer.experts[i].down for i in ids]

    bits, block_size, has_zp = _expert_geometry(down_nodes[0])

    inter = _array(gate_nodes[0].inputs[1]).shape[0]
    hidden_dim = _array(down_nodes[0].inputs[1]).shape[0]

    gate_w = _pack_projection(gate_nodes, 1)
    up_w = _pack_projection(up_nodes, 1)
    fc1_w = np.concatenate([gate_w, up_w], axis=1)
    fc2_w = _pack_projection(down_nodes, 1)

    gate_s = _pack_scales(gate_nodes, 2, inter)
    up_s = _pack_scales(up_nodes, 2, inter)
    fc1_s = np.concatenate([gate_s, up_s], axis=1)
    fc2_s = _pack_scales(down_nodes, 2, hidden_dim)

    fc1_zp = fc2_zp = None
    if has_zp:
        gate_z = _pack_zero_points(gate_nodes, 3, inter)
        up_z = _pack_zero_points(up_nodes, 3, inter)
        fc1_zp = np.concatenate([gate_z, up_z], axis=1)
        fc2_zp = _pack_zero_points(down_nodes, 3, hidden_dim)

    prefix = f"moe.layer{index}"
    fc1_w_v = _make_initializer(
        graph, f"{prefix}.fc1_experts_weights", fc1_w, ir.DataType.UINT8
    )
    fc1_s_v = _make_initializer(graph, f"{prefix}.fc1_scales", fc1_s, ir.DataType.FLOAT)
    fc2_w_v = _make_initializer(
        graph, f"{prefix}.fc2_experts_weights", fc2_w, ir.DataType.UINT8
    )
    fc2_s_v = _make_initializer(graph, f"{prefix}.fc2_scales", fc2_s, ir.DataType.FLOAT)
    fc1_zp_v = (
        _make_initializer(graph, f"{prefix}.fc1_zero_points", fc1_zp, ir.DataType.UINT8)
        if fc1_zp is not None
        else None
    )
    fc2_zp_v = (
        _make_initializer(graph, f"{prefix}.fc2_zero_points", fc2_zp, ir.DataType.UINT8)
        if fc2_zp is not None
        else None
    )

    cast = ir.node(
        "Cast",
        inputs=[layer.logits],
        attributes={"to": ir.DataType.FLOAT.value},
        num_outputs=1,
        name=f"{prefix}.router_probs_cast",
    )
    router_probs = cast.outputs[0]
    router_probs.name = f"{prefix}.router_probs"

    qmoe = ir.node(
        "QMoE",
        inputs=[
            layer.hidden,
            router_probs,
            fc1_w_v,
            fc1_s_v,
            None,
            fc2_w_v,
            fc2_s_v,
            None,
            None,
            None,
            None,
            fc1_zp_v,
            fc2_zp_v,
            None,
            None,
        ],
        attributes={
            "activation_type": "swiglu",
            "normalize_routing_weights": 1,
            "k": layer.k,
            "expert_weight_bits": bits,
            "block_size": block_size,
            "swiglu_fusion": 2,
        },
        domain=_MS_DOMAIN,
        num_outputs=1,
        name=f"{prefix}.qmoe",
    )
    qmoe_out = qmoe.outputs[0]
    qmoe_out.name = f"{prefix}.qmoe_output"
    qmoe_out.type = layer.hidden.type
    qmoe_out.shape = layer.routed_out.shape

    graph.insert_after(layer.topk, [cast, qmoe])
    layer.routed_out.replace_all_uses_with(qmoe_out)


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


def fuse_dense_moe_to_qmoe(model: ir.Model) -> int:
    """Fuse every dense-fallback MoE subgraph in ``model`` into ``QMoE`` nodes.

    Reuses the existing int4 ``MatMulNBits`` expert weights byte-for-byte (pure
    expert-major concat/layout transform, no requantization). ``float16`` scales
    are upcast to the ``float32`` the QMoE kernel requires (lossless). Any shared
    expert is preserved.

    Args:
        model: The IR model to rewrite in place.

    Returns:
        The number of MoE layers fused.
    """
    graph = model.graph
    anchors = _find_moe_layers(graph)
    layers = [_DenseMoELayer(topk) for topk in anchors]
    fused = 0
    for index, layer in enumerate(layers):
        if not layer.is_valid:
            logger.warning(
                "skipping MoE layer at %s: unrecognised dense-fallback structure",
                layer.topk.name,
            )
            continue
        _fuse_layer(graph, layer, index)
        fused += 1
    if fused:
        _remove_dead_nodes(graph)
        graph.sort()
    return fused
