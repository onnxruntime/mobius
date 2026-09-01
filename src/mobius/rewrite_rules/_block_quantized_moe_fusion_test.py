# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the dense-fallback native-block MoE -> ``pkg.nxrt::BlockQuantizedMoE`` rewrite.

Native IQ/GGUF blocks are codebook-based and cannot be dequantized in NumPy, so
these tests are **structural** (the expert storm collapses; only selected experts
execute) and **byte-preservation** (the native blocks are stacked verbatim, never
requantized). Numerical parity of the fused node lives in the onnx-genai Rust
CPU-oracle tests. The routing reconstruction is proven separately against a small
NumPy model of the kernel's ``routing_weights`` selection.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from mobius._constants import OPSET_VERSION
from mobius.integrations.gguf import SparseMoEExportError
from mobius.rewrite_rules import fuse_block_quantized_moe
from mobius.rewrite_rules._block_quantized_moe_fusion import _NATIVE_BLOCK_FORMATS

# Tiny GLM-5.2-MoE-shaped geometry.
H = 32  # hidden size
INTER = 16  # moe intermediate size
E = 4  # experts
K = 2  # top-k
_NXRT = "pkg.nxrt"


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _native_weight(
    rng: np.random.Generator, out_features: int, in_features: int, fmt: str
) -> np.ndarray:
    """Random packed native block weight ``[N, n_blocks, block_bytes]`` uint8."""
    block_elements, block_bytes = _NATIVE_BLOCK_FORMATS[fmt]
    n_blocks = -(-in_features // block_elements)
    return rng.integers(0, 256, size=(out_features, n_blocks, block_bytes), dtype=np.uint8)


def _init(graph: ir.Graph, name: str, arr: np.ndarray, dtype: ir.DataType) -> ir.Value:
    tensor = ir.tensor(arr, name=name, dtype=dtype)
    value = ir.Value(
        name=name, shape=ir.Shape(arr.shape), type=ir.TensorType(dtype), const_value=tensor
    )
    graph.register_initializer(value)
    return value


def _bqmatmul(
    name: str,
    x: ir.Value,
    weight: np.ndarray,
    k: int,
    n: int,
    fmt: str,
    graph: ir.Graph,
    nodes: list[ir.Node],
    *,
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> ir.Value:
    w = _init(graph, f"{name}.weight", weight, ir.DataType.UINT8)
    node = ir.node(
        _MATMUL_OP,
        inputs=[x, w],
        attributes={"K": k, "N": n, "format": fmt, "block_layout_version": 1},
        domain=_NXRT,
        num_outputs=1,
        name=name,
    )
    node.outputs[0].name = f"{name}.out"
    node.outputs[0].type = ir.TensorType(dtype)
    nodes.append(node)
    return node.outputs[0]


_MATMUL_OP = "BlockQuantizedMatMul"


def _constant_int(nodes: list[ir.Node], name: str, value: int) -> ir.Value:
    node = ir.node(
        "Constant", inputs=[], attributes={"value_int": value}, num_outputs=1, name=name
    )
    nodes.append(node)
    node.outputs[0].name = f"{name}.out"
    return node.outputs[0]


def _build_dense_graph(
    *,
    gate_fmt: str = "iq1_s",
    up_fmt: str = "iq1_s",
    down_fmt: str = "iq4_xs",
    activation: str = "swish",
    dtype: ir.DataType = ir.DataType.FLOAT,
    bias_expert: int | None = None,
    corrupt_expert_gate_fmt: tuple[int, str] | None = None,
    break_expert: int | None = None,
    alien_routing_expert: int | None = None,
) -> tuple[ir.Model, dict[str, np.ndarray]]:
    """Build a tiny GLM-style dense-fallback native-block MoE graph.

    The gate is Sigmoid + TopK + normalize (GLM-style, non-softmax) to prove the
    fusion reconstructs routing gate-agnostically. Knobs drive the fail-closed
    tests: ``bias_expert`` adds a bias to one expert, ``corrupt_expert_gate_fmt``
    gives one expert a different gate format, ``break_expert`` replaces one
    expert's down projection with a non-native op, ``alien_routing_expert``
    feeds one expert a distinct (non-shared) routing-weights tensor.
    """
    rng = _rng()
    weights: dict[str, np.ndarray] = {}
    nodes: list[ir.Node] = []

    hidden = ir.Value(name="hidden", shape=ir.Shape(["T", H]), type=ir.TensorType(dtype))
    graph = ir.Graph([hidden], [], nodes=[], name="tiny_native_moe")

    def cast_to_f32(value: ir.Value, name: str) -> ir.Value:
        if dtype == ir.DataType.FLOAT:
            return value
        node = ir.node(
            "Cast",
            inputs=[value],
            attributes={"to": ir.DataType.FLOAT.value},
            num_outputs=1,
            name=name,
        )
        node.outputs[0].name = f"{name}.out"
        node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        nodes.append(node)
        return node.outputs[0]

    def cast_from_f32(value: ir.Value, name: str) -> ir.Value:
        if dtype == ir.DataType.FLOAT:
            return value
        node = ir.node(
            "Cast",
            inputs=[value],
            attributes={"to": dtype.value},
            num_outputs=1,
            name=name,
        )
        node.outputs[0].name = f"{name}.out"
        node.outputs[0].type = ir.TensorType(dtype)
        nodes.append(node)
        return node.outputs[0]

    # --- GLM-style gate: MatMul -> Sigmoid -> TopK -> normalize ---
    router_wt = _init(
        graph,
        "router.weight_t",
        rng.standard_normal((H, E)).astype(np.float32),
        ir.DataType.FLOAT,
    )
    logits = ir.node("MatMul", inputs=[hidden, router_wt], num_outputs=1, name="router.matmul")
    logits.outputs[0].name = "router.logits"
    logits.outputs[0].type = ir.TensorType(dtype)
    logits.outputs[0].shape = ir.Shape(["T", E])
    nodes.append(logits)
    sig = ir.node("Sigmoid", inputs=[logits.outputs[0]], num_outputs=1, name="router.sigmoid")
    sig.outputs[0].name = "router.probs"
    sig.outputs[0].type = ir.TensorType(dtype)
    nodes.append(sig)
    k_init = _init(graph, "topk.k", np.array([K], dtype=np.int64), ir.DataType.INT64)
    topk = ir.node(
        "TopK",
        inputs=[sig.outputs[0], k_init],
        attributes={"axis": -1},
        num_outputs=2,
        name="topk",
    )
    topk.outputs[0].name = "topk.values"
    topk.outputs[0].type = ir.TensorType(dtype)
    topk.outputs[0].shape = ir.Shape(["T", K])
    topk.outputs[1].name = "topk.indices"
    topk.outputs[1].type = ir.TensorType(ir.DataType.INT64)
    topk.outputs[1].shape = ir.Shape(["T", K])
    nodes.append(topk)
    axes = _init(graph, "reduce.axes", np.array([-1], dtype=np.int64), ir.DataType.INT64)
    wsum = ir.node(
        "ReduceSum",
        inputs=[topk.outputs[0], axes],
        attributes={"keepdims": 1},
        num_outputs=1,
        name="router.wsum",
    )
    wsum.outputs[0].name = "router.wsum.out"
    wsum.outputs[0].type = ir.TensorType(dtype)
    nodes.append(wsum)
    routing_weights = ir.node(
        "Div",
        inputs=[topk.outputs[0], wsum.outputs[0]],
        num_outputs=1,
        name="router.norm",
    )
    routing_weights.outputs[0].name = "routing_weights"
    routing_weights.outputs[0].type = ir.TensorType(dtype)
    routing_weights.outputs[0].shape = ir.Shape(["T", K])
    nodes.append(routing_weights)

    selected = topk.outputs[1]
    rweights = routing_weights.outputs[0]

    def act_of(gate_out: ir.Value, e: int) -> ir.Value:
        if activation == "swish":
            node = ir.node("Swish", inputs=[gate_out], num_outputs=1, name=f"expert{e}.act")
            node.outputs[0].name = f"expert{e}.act.out"
            node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
            nodes.append(node)
            return node.outputs[0]
        # legacy gate * Sigmoid(gate)
        sg = ir.node("Sigmoid", inputs=[gate_out], num_outputs=1, name=f"expert{e}.sig")
        sg.outputs[0].name = f"expert{e}.sig.out"
        sg.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        nodes.append(sg)
        mul = ir.node(
            "Mul", inputs=[gate_out, sg.outputs[0]], num_outputs=1, name=f"expert{e}.silu"
        )
        mul.outputs[0].name = f"expert{e}.silu.out"
        mul.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        nodes.append(mul)
        return mul.outputs[0]

    routed = None
    for e in range(E):
        eid = _constant_int(nodes, f"eid_{e}", e)
        equal = ir.node("Equal", inputs=[selected, eid], num_outputs=1, name=f"equal_{e}")
        equal.outputs[0].name = f"equal_{e}.out"
        nodes.append(equal)
        cast = ir.node(
            "CastLike", inputs=[equal.outputs[0], rweights], num_outputs=1, name=f"cast_{e}"
        )
        cast.outputs[0].name = f"cast_{e}.out"
        nodes.append(cast)
        this_rweights = rweights
        if alien_routing_expert == e:
            # A distinct routing-weights tensor for one expert means this is not
            # a single shared-gate storm; that expert is dropped and the layer
            # must fail closed rather than fuse a short bank.
            alien = ir.node("Identity", inputs=[rweights], num_outputs=1, name=f"alien_rw_{e}")
            alien.outputs[0].name = f"alien_rw_{e}.out"
            alien.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
            nodes.append(alien)
            this_rweights = alien.outputs[0]
        wmul = ir.node(
            "Mul", inputs=[this_rweights, cast.outputs[0]], num_outputs=1, name=f"wmul_{e}"
        )
        wmul.outputs[0].name = f"wmul_{e}.out"
        nodes.append(wmul)
        reduce = ir.node(
            "ReduceSum",
            inputs=[wmul.outputs[0], axes],
            attributes={"keepdims": 1},
            num_outputs=1,
            name=f"reduce_{e}",
        )
        reduce.outputs[0].name = f"reduce_{e}.out"
        nodes.append(reduce)

        this_gate_fmt = gate_fmt
        if corrupt_expert_gate_fmt is not None and corrupt_expert_gate_fmt[0] == e:
            this_gate_fmt = corrupt_expert_gate_fmt[1]
        x_f32 = cast_to_f32(hidden, f"expert{e}.gate.cast_in")
        gate_w = _native_weight(rng, INTER, H, this_gate_fmt)
        weights[f"g{e}"] = gate_w
        gate = _bqmatmul(
            f"expert{e}.gate_proj", x_f32, gate_w, H, INTER, this_gate_fmt, graph, nodes
        )
        gate = cast_from_f32(gate, f"expert{e}.gate.cast_out")
        act = act_of(gate, e)

        up_x = cast_to_f32(hidden, f"expert{e}.up.cast_in")
        up_w = _native_weight(rng, INTER, H, up_fmt)
        weights[f"u{e}"] = up_w
        up = _bqmatmul(f"expert{e}.up_proj", up_x, up_w, H, INTER, up_fmt, graph, nodes)
        up = cast_from_f32(up, f"expert{e}.up.cast_out")

        act_mul = ir.node("Mul", inputs=[act, up], num_outputs=1, name=f"expert{e}.mul")
        act_mul.outputs[0].name = f"expert{e}.mul.out"
        act_mul.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        nodes.append(act_mul)

        down_x = cast_to_f32(act_mul.outputs[0], f"expert{e}.down.cast_in")
        if break_expert == e:
            # Replace the down projection with a non-native op: this expert can
            # no longer be traced and the whole storm must fail closed.
            broken = ir.node(
                "Identity", inputs=[down_x], num_outputs=1, name=f"expert{e}.down_broken"
            )
            broken.outputs[0].name = f"expert{e}.down.out"
            broken.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
            nodes.append(broken)
            down_out = cast_from_f32(broken.outputs[0], f"expert{e}.down.cast_out")
        else:
            down_w = _native_weight(rng, H, INTER, down_fmt)
            weights[f"d{e}"] = down_w
            bias_inputs = None
            if bias_expert == e:
                bias_arr = rng.standard_normal(INTER).astype(np.float32)
                bias_inputs = _init(graph, f"expert{e}.down.bias", bias_arr, ir.DataType.FLOAT)
            dw = _init(graph, f"expert{e}.down_proj.weight", down_w, ir.DataType.UINT8)
            down_node = ir.node(
                _MATMUL_OP,
                inputs=[down_x, dw] + ([bias_inputs] if bias_inputs is not None else []),
                attributes={"K": INTER, "N": H, "format": down_fmt, "block_layout_version": 1},
                domain=_NXRT,
                num_outputs=1,
                name=f"expert{e}.down_proj",
            )
            down_node.outputs[0].name = f"expert{e}.down_proj.out"
            down_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
            nodes.append(down_node)
            down_out = cast_from_f32(down_node.outputs[0], f"expert{e}.down.cast_out")

        contrib = ir.node(
            "Mul",
            inputs=[down_out, reduce.outputs[0]],
            num_outputs=1,
            name=f"expert{e}.contrib",
        )
        contrib.outputs[0].name = f"expert{e}.contrib.out"
        contrib.outputs[0].type = ir.TensorType(dtype)
        nodes.append(contrib)
        if routed is None:
            routed = contrib.outputs[0]
        else:
            add = ir.node(
                "Add", inputs=[routed, contrib.outputs[0]], num_outputs=1, name=f"acc_{e}"
            )
            add.outputs[0].name = f"acc_{e}.out"
            add.outputs[0].type = ir.TensorType(dtype)
            nodes.append(add)
            routed = add.outputs[0]

    # --- Shared expert (must survive the rewrite untouched) ---
    s_x = cast_to_f32(hidden, "shared.gate.cast_in")
    sg_w = _native_weight(rng, INTER, H, gate_fmt)
    weights["sg"] = sg_w
    s_gate = _bqmatmul("shared.gate_proj", s_x, sg_w, H, INTER, gate_fmt, graph, nodes)
    s_gate = cast_from_f32(s_gate, "shared.gate.cast_out")
    s_act = ir.node("Swish", inputs=[s_gate], num_outputs=1, name="shared.act")
    s_act.outputs[0].name = "shared.act.out"
    s_act.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    nodes.append(s_act)
    su_x = cast_to_f32(hidden, "shared.up.cast_in")
    su_w = _native_weight(rng, INTER, H, up_fmt)
    weights["su"] = su_w
    s_up = _bqmatmul("shared.up_proj", su_x, su_w, H, INTER, up_fmt, graph, nodes)
    s_up = cast_from_f32(s_up, "shared.up.cast_out")
    s_mul = ir.node("Mul", inputs=[s_act.outputs[0], s_up], num_outputs=1, name="shared.mul")
    s_mul.outputs[0].name = "shared.mul.out"
    s_mul.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    nodes.append(s_mul)
    sd_x = cast_to_f32(s_mul.outputs[0], "shared.down.cast_in")
    sd_w = _native_weight(rng, H, INTER, down_fmt)
    weights["sd"] = sd_w
    s_down = _bqmatmul("shared.down_proj", sd_x, sd_w, INTER, H, down_fmt, graph, nodes)
    s_down = cast_from_f32(s_down, "shared.down.cast_out")

    final = ir.node("Add", inputs=[routed, s_down], num_outputs=1, name="final_add")
    final.outputs[0].name = "moe_out"
    final.outputs[0].shape = ir.Shape(["T", H])
    final.outputs[0].type = ir.TensorType(dtype)
    nodes.append(final)

    for node in nodes:
        graph.append(node)
    graph.outputs.append(final.outputs[0])

    model = ir.Model(graph, ir_version=10, producer_name="test")
    model.opset_imports[""] = OPSET_VERSION
    model.opset_imports[_NXRT] = 1
    return model, weights


def _count(graph: ir.Graph, op_type: str) -> int:
    return sum(1 for n in graph if n.op_type == op_type)


def _moe(graph: ir.Graph) -> ir.Node:
    return next(n for n in graph if n.op_type == "BlockQuantizedMoE")


# --------------------------------------------------------------------------- #
# Structural collapse                                                         #
# --------------------------------------------------------------------------- #
def test_fuses_expert_storm_into_single_bqmoe() -> None:
    model, _ = _build_dense_graph()
    graph = model.graph

    assert _count(graph, "BlockQuantizedMoE") == 0
    # E experts x 3 projections + 3 shared projections.
    assert _count(graph, _MATMUL_OP) == E * 3 + 3

    fused = fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)

    assert fused == 1
    assert _count(graph, "BlockQuantizedMoE") == 1
    # Only the three shared-expert projections remain; every routed expert
    # matmul collapsed into the single sparse node.
    assert _count(graph, _MATMUL_OP) == 3
    # The per-expert masking machinery is gone -> only selected experts execute.
    assert _count(graph, "Equal") == 0
    assert _count(graph, "CastLike") == 0
    # Two scatters reconstruct routing; the gate (TopK) is preserved.
    assert _count(graph, "ScatterElements") == 2
    assert _count(graph, "TopK") == 1


def test_domain_and_opset_registered() -> None:
    model, _ = _build_dense_graph()
    fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)
    moe = _moe(model.graph)
    assert moe.domain == _NXRT
    assert model.graph.opset_imports[_NXRT] == 1


def test_input_wiring_matches_runtime_abi() -> None:
    model, _ = _build_dense_graph(gate_fmt="iq1_s", up_fmt="iq1_s", down_fmt="iq4_xs")
    fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)
    moe = _moe(model.graph)
    # Canonical 12-input BlockQuantizedMoE ABI (fused SwiGLU: no fc3/scales).
    assert len(moe.inputs) == 12
    assert moe.inputs[0].name == "hidden"  # input
    assert moe.inputs[1].name.endswith("router_logits")  # router_logits
    assert moe.inputs[2].name.endswith("fc1_experts_weights")
    assert moe.inputs[3] is None  # fc1 bias
    assert moe.inputs[4].name.endswith("fc2_experts_weights")
    assert moe.inputs[5] is None  # fc2 bias
    assert moe.inputs[6] is None  # fc3 (fused -> absent)
    assert moe.inputs[7] is None  # fc3 bias
    assert moe.inputs[8].name.endswith("router_weights")
    assert moe.inputs[9] is None  # interleaved fc1 has no auxiliary scale
    assert moe.inputs[10] is None  # interleaved fc2 has no auxiliary scale
    assert moe.inputs[11] is None  # absent fc3 has no auxiliary scale


# --------------------------------------------------------------------------- #
# Byte preservation                                                           #
# --------------------------------------------------------------------------- #
def test_fused_swiglu_weights_are_byte_identical() -> None:
    """gate_fmt == up_fmt -> fc1 = concat(gate, up) rows; bytes untouched."""
    model, w = _build_dense_graph(gate_fmt="iq1_s", up_fmt="iq1_s", down_fmt="iq4_xs")
    fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)
    moe = _moe(model.graph)
    fc1 = moe.inputs[2].const_value.numpy()
    fc2 = moe.inputs[4].const_value.numpy()
    assert moe.inputs[6] is None  # fused: no fc3
    for e in range(E):
        expected_fc1 = np.concatenate([w[f"g{e}"], w[f"u{e}"]], axis=0)
        np.testing.assert_array_equal(fc1[e], expected_fc1)
        np.testing.assert_array_equal(fc2[e], w[f"d{e}"])


def test_unfused_swiglu_weights_are_byte_identical() -> None:
    """gate_fmt != up_fmt -> fc1 = gate, fc3 = up, fc2 = down; bytes untouched."""
    model, w = _build_dense_graph(gate_fmt="iq1_s", up_fmt="iq3_xxs", down_fmt="iq4_xs")
    fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)
    moe = _moe(model.graph)
    fc1 = moe.inputs[2].const_value.numpy()
    fc2 = moe.inputs[4].const_value.numpy()
    assert moe.inputs[6] is not None  # unfused: fc3 present
    fc3 = moe.inputs[6].const_value.numpy()
    for e in range(E):
        np.testing.assert_array_equal(fc1[e], w[f"g{e}"])
        np.testing.assert_array_equal(fc3[e], w[f"u{e}"])
        np.testing.assert_array_equal(fc2[e], w[f"d{e}"])


def test_expert_major_bank_dtype_is_uint8() -> None:
    model, _ = _build_dense_graph()
    fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)
    moe = _moe(model.graph)
    assert moe.inputs[2].dtype == ir.DataType.UINT8
    assert moe.inputs[4].dtype == ir.DataType.UINT8


# --------------------------------------------------------------------------- #
# Per-projection format attributes (canonical v1)                             #
# --------------------------------------------------------------------------- #
def test_uniform_format_stays_layout_version_1() -> None:
    model, _ = _build_dense_graph(gate_fmt="iq4_xs", up_fmt="iq4_xs", down_fmt="iq4_xs")
    fuse_block_quantized_moe(model)
    moe = _moe(model.graph)
    attrs = moe.attributes
    assert attrs["block_layout_version"].value == 1
    assert "format" not in attrs
    assert attrs["fc1_format"].value == "iq4_xs"
    assert attrs["fc2_format"].value == "iq4_xs"
    assert "fc3_format" not in attrs


def test_fused_mixed_format_emits_canonical_layout_version_1() -> None:
    model, _ = _build_dense_graph(gate_fmt="iq1_s", up_fmt="iq1_s", down_fmt="iq4_xs")
    fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)
    moe = _moe(model.graph)
    attrs = moe.attributes
    assert attrs["block_layout_version"].value == 1
    assert "format" not in attrs
    assert attrs["fc1_format"].value == "iq1_s"
    assert attrs["fc2_format"].value == "iq4_xs"
    assert "fc3_format" not in attrs  # fused -> no fc3
    assert attrs["swiglu_fusion"].value == 2


def test_unfused_mixed_format_emits_all_projection_formats() -> None:
    model, _ = _build_dense_graph(gate_fmt="iq1_s", up_fmt="iq3_xxs", down_fmt="iq4_xs")
    fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)
    moe = _moe(model.graph)
    attrs = moe.attributes
    assert attrs["block_layout_version"].value == 1
    assert attrs["fc1_format"].value == "iq1_s"
    assert attrs["fc3_format"].value == "iq3_xxs"
    assert attrs["fc2_format"].value == "iq4_xs"
    assert attrs["swiglu_fusion"].value == 0


def test_core_attributes_match_bqmoe_abi() -> None:
    model, _ = _build_dense_graph()
    fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)
    attrs = _moe(model.graph).attributes
    assert attrs["k"].value == K
    assert attrs["activation_type"].value == "swiglu"
    assert attrs["normalize_routing_weights"].value == 0


# --------------------------------------------------------------------------- #
# Per-projection canonical-v1 production path                                 #
# --------------------------------------------------------------------------- #
def test_mixed_format_fuses_to_v1_by_default() -> None:
    model, _ = _build_dense_graph(gate_fmt="iq1_s", up_fmt="iq1_s", down_fmt="iq4_xs")
    assert fuse_block_quantized_moe(model) == 1
    attrs = _moe(model.graph).attributes
    assert attrs["block_layout_version"].value == 1
    assert attrs["fc1_format"].value == "iq1_s"
    assert attrs["fc2_format"].value == "iq4_xs"


def test_uniform_format_still_fuses_to_v1() -> None:
    """A uniform-format layer is a v1 node, so it fuses under the fail-closed default."""
    model, _ = _build_dense_graph(gate_fmt="iq4_xs", up_fmt="iq4_xs", down_fmt="iq4_xs")
    assert fuse_block_quantized_moe(model) == 1
    attrs = _moe(model.graph).attributes
    assert attrs["block_layout_version"].value == 1
    assert attrs["fc1_format"].value == "iq4_xs"
    assert attrs["fc2_format"].value == "iq4_xs"


def test_legacy_schema_hook_still_emits_canonical_v1() -> None:
    model, _ = _build_dense_graph(gate_fmt="iq1_s", up_fmt="iq1_s", down_fmt="iq4_xs")
    assert fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True) == 1
    attrs = _moe(model.graph).attributes
    assert attrs["block_layout_version"].value == 1
    assert attrs["fc1_format"].value == "iq1_s"
    assert attrs["fc2_format"].value == "iq4_xs"


# --------------------------------------------------------------------------- #
# Gate-agnostic routing + activation variants                                 #
# --------------------------------------------------------------------------- #
def test_fuses_legacy_gate_sigmoid_activation() -> None:
    """The legacy ``gate * Sigmoid(gate)`` SwiGLU decomposition still fuses."""
    model, _ = _build_dense_graph(activation="legacy")
    fused = fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)
    assert fused == 1
    assert _count(model.graph, "BlockQuantizedMoE") == 1


def test_fuses_f16_graph_with_cast_wrapped_projections() -> None:
    """f16 exports wrap every projection in Cast; the trace and cast-back work."""
    model, _ = _build_dense_graph(dtype=ir.DataType.FLOAT16)
    graph = model.graph
    fused = fuse_block_quantized_moe(model, _allow_perproj_v2_schema=True)
    assert fused == 1
    moe = _moe(graph)
    # BlockQuantizedMoE output is FLOAT; a Cast restores the f16 routed dtype.
    consumer = next(n for n, _ in moe.outputs[0].uses())
    assert consumer.op_type == "Cast"
    assert consumer.attributes["to"].value == ir.DataType.FLOAT16.value


# --------------------------------------------------------------------------- #
# Fail-closed (typed SparseMoEExportError, graph left untouched)              #
# --------------------------------------------------------------------------- #
def test_mixed_format_across_experts_fails_closed() -> None:
    """One expert with a different gate format cannot form one expert-major bank."""
    model, _ = _build_dense_graph(corrupt_expert_gate_fmt=(1, "iq1_m"))
    graph = model.graph
    with pytest.raises(SparseMoEExportError):
        fuse_block_quantized_moe(model)
    assert _count(graph, "BlockQuantizedMoE") == 0
    assert _count(graph, "Equal") == E  # dense graph left untouched


def test_biased_expert_fails_closed() -> None:
    model, _ = _build_dense_graph(bias_expert=2)
    graph = model.graph
    with pytest.raises(SparseMoEExportError):
        fuse_block_quantized_moe(model)
    assert _count(graph, "BlockQuantizedMoE") == 0
    assert _count(graph, "Equal") == E


def test_incomplete_expert_group_fails_closed() -> None:
    """A non-native (untraceable) expert leaves an incomplete 0..E-1 set."""
    model, _ = _build_dense_graph(break_expert=2)
    graph = model.graph
    with pytest.raises(SparseMoEExportError):
        fuse_block_quantized_moe(model)
    assert _count(graph, "BlockQuantizedMoE") == 0


def test_dropped_trailing_expert_fails_closed() -> None:
    """Dropping the *highest* expert id must fail closed, not fuse a short bank.

    The survivors ``0..E-2`` still look contiguous, so a gap-only check would
    silently fuse an expert bank narrower than the router actually selects over
    (leaving ``ScatterElements`` to index out of bounds). The declared-id set
    from the ``Equal`` masks is the authority, so this must fail closed.
    """
    model, _ = _build_dense_graph(break_expert=E - 1)
    graph = model.graph
    with pytest.raises(SparseMoEExportError):
        fuse_block_quantized_moe(model)
    assert _count(graph, "BlockQuantizedMoE") == 0
    assert _count(graph, "Equal") == E  # dense graph left untouched


def test_alien_routing_weights_expert_fails_closed() -> None:
    """An expert wired to a non-shared routing-weights tensor must fail closed."""
    model, _ = _build_dense_graph(alien_routing_expert=E - 1)
    graph = model.graph
    with pytest.raises(SparseMoEExportError):
        fuse_block_quantized_moe(model)
    assert _count(graph, "BlockQuantizedMoE") == 0
    assert _count(graph, "Equal") == E  # dense graph left untouched


def test_allow_dense_moe_opt_in_keeps_dense_fallback() -> None:
    """``allow_dense_moe=True`` downgrades a fail-closed layer to a dense keep.

    The honesty opt-in (mirroring ``MOBIUS_ALLOW_DENSE_MOE_EXPERTS``) must not
    raise and must leave the runnable per-expert dense fallback intact.
    """
    model, _ = _build_dense_graph(corrupt_expert_gate_fmt=(1, "iq1_m"))
    graph = model.graph
    fused = fuse_block_quantized_moe(model, allow_dense_moe=True)
    assert fused == 0
    assert _count(graph, "BlockQuantizedMoE") == 0
    assert _count(graph, "Equal") == E  # dense fallback preserved, not fused


def test_no_moe_is_a_noop() -> None:
    hidden = ir.Value(
        name="x", shape=ir.Shape(["T", H]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    graph = ir.Graph([hidden], [hidden], nodes=[], name="empty")
    model = ir.Model(graph, ir_version=10, producer_name="test")
    model.opset_imports[""] = OPSET_VERSION
    assert fuse_block_quantized_moe(model) == 0


# --------------------------------------------------------------------------- #
# Routing reconstruction semantics (NumPy model of the kernel selection)      #
# --------------------------------------------------------------------------- #
def _kernel_route(logits_row: np.ndarray, weights_row: np.ndarray, k: int) -> dict[int, float]:
    """Mirror onnx-genai's ``routing_weights`` with ``normalize=0``.

    Top-k by ``logits`` (ties broken by lower index), gather ``weights`` at the
    selected experts, no renormalization.
    """
    order = sorted(range(len(logits_row)), key=lambda i: (-logits_row[i], i))
    return {i: float(weights_row[i]) for i in order[:k]}


def test_scatter_routing_reproduces_dense_selection_and_weights() -> None:
    """One-hot logits + scattered weights re-select exactly the dense experts."""
    rng = np.random.default_rng(1)
    tokens, experts, k = 7, E, K
    for _ in range(50):
        selected = np.stack([rng.permutation(experts)[:k] for _ in range(tokens)]).astype(
            np.int64
        )
        dense_weights = rng.random((tokens, k)).astype(np.float32)

        logits_full = np.zeros((tokens, experts), dtype=np.float32)
        weights_full = np.zeros((tokens, experts), dtype=np.float32)
        np.put_along_axis(logits_full, selected, 1.0, axis=1)
        np.put_along_axis(weights_full, selected, dense_weights, axis=1)

        for t in range(tokens):
            got = _kernel_route(logits_full[t], weights_full[t], k)
            expected = {int(selected[t, j]): float(dense_weights[t, j]) for j in range(k)}
            assert got == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
