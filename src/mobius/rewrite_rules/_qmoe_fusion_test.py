# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the dense-fallback MoE -> ``com.microsoft::QMoE`` graph rewrite."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from mobius._constants import OPSET_VERSION
from mobius.rewrite_rules import fuse_dense_moe_to_qmoe
from mobius.rewrite_rules._qmoe_fusion import _qmoe_abi_supported

# Tiny Qwen35-MoE-shaped geometry (mirrors the real 35B topology at 1/64 scale).
H = 32  # hidden size
INTER = 16  # moe intermediate size
E = 4  # experts
K = 2  # top-k
BLOCK = 16  # quantization block size (power-of-two >= 16, as the kernel requires)
BITS = 4


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _pack_weight(codes: np.ndarray, block_size: int) -> np.ndarray:
    """Pack int4 codes ``[N, Kdim]`` into ``MatMulNBits`` ``[N, n_blocks, blob]`` uint8."""
    n, kdim = codes.shape
    n_blocks = kdim // block_size
    blob = block_size // 2
    packed = np.zeros((n, n_blocks, blob), dtype=np.uint8)
    for b in range(n_blocks):
        block = codes[:, b * block_size : (b + 1) * block_size]
        lo = block[:, 0::2].astype(np.uint8)
        hi = block[:, 1::2].astype(np.uint8)
        packed[:, b, :] = lo | (hi << 4)
    return packed


def _pack_zero_points(codes: np.ndarray) -> np.ndarray:
    """Pack per-block int4 zero-points ``[N, n_blocks]`` into ``[N, ceil(n_blocks/2)]``."""
    n, n_blocks = codes.shape
    cols = (n_blocks + 1) // 2
    packed = np.zeros((n, cols), dtype=np.uint8)
    for b in range(n_blocks):
        nibble = codes[:, b].astype(np.uint8)
        if b % 2 == 0:
            packed[:, b // 2] |= nibble
        else:
            packed[:, b // 2] |= nibble << 4
    return packed


def _dequant(
    packed: np.ndarray, scales: np.ndarray, zp: np.ndarray | None, block_size: int
) -> np.ndarray:
    """Reconstruct a float weight ``[N, Kdim]`` from a ``MatMulNBits`` triple."""
    n, n_blocks, blob = (
        packed.shape
        if packed.ndim == 3
        else (
            packed.shape[0],
            (packed.shape[1] * 2) // block_size,
            block_size // 2,
        )
    )
    packed = packed.reshape(n, n_blocks, blob)
    kdim = n_blocks * block_size
    scales = scales.reshape(n, n_blocks).astype(np.float32)
    out = np.zeros((n, kdim), dtype=np.float32)
    for b in range(n_blocks):
        byte = packed[:, b, :].astype(np.int32)
        lo = byte & 0xF
        hi = (byte >> 4) & 0xF
        codes = np.empty((n, block_size), dtype=np.int32)
        codes[:, 0::2] = lo
        codes[:, 1::2] = hi
        if zp is not None:
            zp_reshaped = zp.reshape(n, -1)
            zc = (zp_reshaped[:, b // 2].astype(np.int32) >> ((b % 2) * 4)) & 0xF
        else:
            zc = np.full(n, 1 << (BITS - 1), dtype=np.int32)
        out[:, b * block_size : (b + 1) * block_size] = (codes - zc[:, None]) * scales[
            :, b : b + 1
        ]
    return out


class _Quant:
    """A randomly-generated int4 ``MatMulNBits`` weight triple."""

    def __init__(self, rng: np.random.Generator, n: int, kdim: int) -> None:
        n_blocks = kdim // BLOCK
        codes = rng.integers(0, 16, size=(n, kdim), dtype=np.int32)
        zp_codes = rng.integers(0, 16, size=(n, n_blocks), dtype=np.int32)
        self.weight = _pack_weight(codes, BLOCK)
        self.scales = rng.random((n, n_blocks), dtype=np.float32).astype(np.float16) * 0.1
        self.zero_points = _pack_zero_points(zp_codes)
        self.n = n
        self.kdim = kdim

    @property
    def dense(self) -> np.ndarray:
        return _dequant(self.weight, self.scales, self.zero_points, BLOCK)


def _init(graph: ir.Graph, name: str, arr: np.ndarray, dtype: ir.DataType) -> ir.Value:
    tensor = ir.tensor(arr, name=name, dtype=dtype)
    value = ir.Value(
        name=name, shape=ir.Shape(arr.shape), type=ir.TensorType(dtype), const_value=tensor
    )
    graph.register_initializer(value)
    return value


def _matmulnbits(name: str, x: ir.Value, q: _Quant, graph: ir.Graph) -> ir.Value:
    w = _init(graph, f"{name}.weight", q.weight, ir.DataType.UINT8)
    s = _init(graph, f"{name}.scales", q.scales, ir.DataType.FLOAT16)
    z = _init(graph, f"{name}.zero_points", q.zero_points, ir.DataType.UINT8)
    node = ir.node(
        "MatMulNBits",
        inputs=[x, w, s, z],
        attributes={"K": q.kdim, "N": q.n, "bits": BITS, "block_size": BLOCK},
        domain="com.microsoft",
        num_outputs=1,
        name=name,
    )
    node.outputs[0].name = f"{name}.out"
    return node.outputs[0]


def _constant_int(graph_nodes: list[ir.Node], name: str, value: int) -> ir.Value:
    node = ir.node(
        "Constant",
        inputs=[],
        attributes={"value_int": value},
        num_outputs=1,
        name=name,
    )
    graph_nodes.append(node)
    node.outputs[0].name = f"{name}.out"
    return node.outputs[0]


def _build_dense_graph(
    activation_dtype: ir.DataType = ir.DataType.FLOAT16,
) -> tuple[ir.Model, dict[str, _Quant], np.ndarray]:
    """Build a tiny dense-fallback Qwen35-MoE graph and return it with its weights."""
    rng = _rng()
    quants: dict[str, _Quant] = {}
    nodes: list[ir.Node] = []

    hidden = ir.Value(
        name="hidden",
        shape=ir.Shape(["T", H]),
        type=ir.TensorType(activation_dtype),
    )
    graph = ir.Graph([hidden], [], nodes=[], name="tiny_moe")

    def q(key: str, n: int, kdim: int) -> _Quant:
        quants[key] = _Quant(rng, n, kdim)
        return quants[key]

    # Router: quantized gate MatMulNBits -> TopK -> Softmax.
    router_out = _matmulnbits("gate", hidden, q("router", E, H), graph)
    nodes.append(router_out.producer())
    k_init = _init(graph, "topk_k", np.array([K], dtype=np.int64), ir.DataType.INT64)
    topk = ir.node(
        "TopK",
        inputs=[router_out, k_init],
        attributes={"axis": -1},
        num_outputs=2,
        name="topk",
    )
    topk.outputs[0].name = "topk.values"
    topk.outputs[1].name = "topk.indices"
    nodes.append(topk)
    softmax = ir.node(
        "Softmax",
        inputs=[topk.outputs[0]],
        attributes={"axis": -1},
        num_outputs=1,
        name="softmax",
    )
    softmax.outputs[0].name = "softmax.out"
    nodes.append(softmax)

    axes = _init(graph, "reduce_axes", np.array([-1], dtype=np.int64), ir.DataType.INT64)

    routed = None
    for e in range(E):
        eid = _constant_int(nodes, f"eid_{e}", e)
        equal = ir.node(
            "Equal", inputs=[topk.outputs[1], eid], num_outputs=1, name=f"equal_{e}"
        )
        equal.outputs[0].name = f"equal_{e}.out"
        nodes.append(equal)
        cast = ir.node(
            "CastLike",
            inputs=[equal.outputs[0], softmax.outputs[0]],
            num_outputs=1,
            name=f"cast_{e}",
        )
        cast.outputs[0].name = f"cast_{e}.out"
        nodes.append(cast)
        wmul = ir.node(
            "Mul",
            inputs=[softmax.outputs[0], cast.outputs[0]],
            num_outputs=1,
            name=f"wmul_{e}",
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

        gate = _matmulnbits(f"expert{e}.gate_proj", hidden, q(f"g{e}", INTER, H), graph)
        nodes.append(gate.producer())
        silu = ir.node("Swish", inputs=[gate], num_outputs=1, name=f"expert{e}.silu")
        silu.outputs[0].name = f"expert{e}.silu.out"
        nodes.append(silu)
        up = _matmulnbits(f"expert{e}.up_proj", hidden, q(f"u{e}", INTER, H), graph)
        nodes.append(up.producer())
        act = ir.node(
            "Mul", inputs=[silu.outputs[0], up], num_outputs=1, name=f"expert{e}.act"
        )
        act.outputs[0].name = f"expert{e}.act.out"
        nodes.append(act)
        down = _matmulnbits(
            f"expert{e}.down_proj", act.outputs[0], q(f"d{e}", H, INTER), graph
        )
        nodes.append(down.producer())
        contrib = ir.node(
            "Mul", inputs=[down, reduce.outputs[0]], num_outputs=1, name=f"expert{e}.contrib"
        )
        contrib.outputs[0].name = f"expert{e}.contrib.out"
        nodes.append(contrib)
        if routed is None:
            routed = contrib.outputs[0]
        else:
            add = ir.node(
                "Add", inputs=[routed, contrib.outputs[0]], num_outputs=1, name=f"acc_{e}"
            )
            add.outputs[0].name = f"acc_{e}.out"
            nodes.append(add)
            routed = add.outputs[0]

    # Shared expert (must survive the rewrite untouched).
    s_gate = _matmulnbits("shared.gate_proj", hidden, q("sg", INTER, H), graph)
    nodes.append(s_gate.producer())
    s_sig = ir.node("Sigmoid", inputs=[s_gate], num_outputs=1, name="shared.sigmoid")
    s_sig.outputs[0].name = "shared.sigmoid.out"
    nodes.append(s_sig)
    s_silu = ir.node(
        "Mul", inputs=[s_gate, s_sig.outputs[0]], num_outputs=1, name="shared.silu"
    )
    s_silu.outputs[0].name = "shared.silu.out"
    nodes.append(s_silu)
    s_up = _matmulnbits("shared.up_proj", hidden, q("su", INTER, H), graph)
    nodes.append(s_up.producer())
    s_act = ir.node("Mul", inputs=[s_silu.outputs[0], s_up], num_outputs=1, name="shared.act")
    s_act.outputs[0].name = "shared.act.out"
    nodes.append(s_act)
    s_down = _matmulnbits("shared.down_proj", s_act.outputs[0], q("sd", H, INTER), graph)
    nodes.append(s_down.producer())
    s_gscore = _matmulnbits("shared_gate", hidden, q("sgate", 1, H), graph)
    nodes.append(s_gscore.producer())
    s_gsig = ir.node("Sigmoid", inputs=[s_gscore], num_outputs=1, name="shared_gate.sigmoid")
    s_gsig.outputs[0].name = "shared_gate.sigmoid.out"
    nodes.append(s_gsig)
    s_scaled = ir.node(
        "Mul", inputs=[s_down, s_gsig.outputs[0]], num_outputs=1, name="shared.scaled"
    )
    s_scaled.outputs[0].name = "shared.scaled.out"
    nodes.append(s_scaled)

    final = ir.node(
        "Add", inputs=[routed, s_scaled.outputs[0]], num_outputs=1, name="final_add"
    )
    final.outputs[0].name = "moe_out"
    final.outputs[0].shape = ir.Shape(["T", H])
    final.outputs[0].type = ir.TensorType(activation_dtype)
    nodes.append(final)

    for node in nodes:
        graph.append(node)
    graph.outputs.append(final.outputs[0])

    model = ir.Model(graph, ir_version=10, producer_name="test")
    model.opset_imports[""] = OPSET_VERSION
    model.opset_imports["com.microsoft"] = 1
    return model, quants, rng.standard_normal((5, H)).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _dense_reference(quants: dict[str, _Quant], hidden: np.ndarray) -> np.ndarray:
    """Full dense-fallback MoE forward (routed experts + shared expert)."""
    logits = hidden @ quants["router"].dense.T
    topk_idx = np.argsort(-logits, axis=-1, kind="stable")[:, :K]
    topk_vals = np.take_along_axis(logits, topk_idx, axis=-1)
    sm = np.exp(topk_vals - topk_vals.max(-1, keepdims=True))
    sm = sm / sm.sum(-1, keepdims=True)

    routed = np.zeros_like(hidden)
    for e in range(E):
        gate = hidden @ quants[f"g{e}"].dense.T
        up = hidden @ quants[f"u{e}"].dense.T
        act = (gate * _sigmoid(gate)) * up
        down = act @ quants[f"d{e}"].dense.T
        weight = np.where(topk_idx == e, sm, 0.0).sum(-1, keepdims=True)
        routed += down * weight

    s_gate = hidden @ quants["sg"].dense.T
    s_up = hidden @ quants["su"].dense.T
    s_act = (s_gate * _sigmoid(s_gate)) * s_up
    s_down = s_act @ quants["sd"].dense.T
    s_score = _sigmoid(hidden @ quants["sgate"].dense.T)
    return routed + s_down * s_score


def _qmoe_reference(graph: ir.Graph, hidden: np.ndarray) -> np.ndarray:
    """Forward the rewritten graph: QMoE routed sum + surviving shared expert."""
    inits = dict(graph.initializers.items())

    def dq(prefix: str) -> np.ndarray:
        return _dequant(
            inits[f"{prefix}.weight"].const_value.numpy(),
            inits[f"{prefix}.scales"].const_value.numpy(),
            inits[f"{prefix}.zero_points"].const_value.numpy(),
            BLOCK,
        )

    logits = hidden @ dq("gate").T
    topk_idx = np.argsort(-logits, axis=-1, kind="stable")[:, :K]
    topk_vals = np.take_along_axis(logits, topk_idx, axis=-1)
    sm = np.exp(topk_vals - topk_vals.max(-1, keepdims=True))
    sm = sm / sm.sum(-1, keepdims=True)

    qmoe = next(n for n in graph if n.op_type == "QMoE")
    fc1_w = qmoe.inputs[2].const_value.numpy()
    fc1_s = qmoe.inputs[3].const_value.numpy()
    fc1_z = qmoe.inputs[11].const_value.numpy()
    fc2_w = qmoe.inputs[5].const_value.numpy()
    fc2_s = qmoe.inputs[6].const_value.numpy()
    fc2_z = qmoe.inputs[12].const_value.numpy()

    routed = np.zeros_like(hidden)
    for e in range(E):
        fc1 = _dequant(fc1_w[e], fc1_s[e], fc1_z[e], BLOCK)
        gate = hidden @ fc1[:INTER].T
        up = hidden @ fc1[INTER:].T
        act = (gate * _sigmoid(gate)) * up
        fc2 = _dequant(fc2_w[e], fc2_s[e], fc2_z[e], BLOCK)
        down = act @ fc2.T
        weight = np.where(topk_idx == e, sm, 0.0).sum(-1, keepdims=True)
        routed += down * weight

    s_gate = hidden @ dq("shared.gate_proj").T
    s_up = hidden @ dq("shared.up_proj").T
    s_act = (s_gate * _sigmoid(s_gate)) * s_up
    s_down = s_act @ dq("shared.down_proj").T
    s_score = _sigmoid(hidden @ dq("shared_gate").T)
    return routed + s_down * s_score


def _count(graph: ir.Graph, op_type: str) -> int:
    return sum(1 for n in graph if n.op_type == op_type)


def test_fuses_expert_storm_into_single_qmoe() -> None:
    model, _, _ = _build_dense_graph()
    graph = model.graph

    assert _count(graph, "QMoE") == 0
    assert _count(graph, "MatMulNBits") == 1 + E * 3 + 4  # router + experts + shared

    fused = fuse_dense_moe_to_qmoe(model)

    assert fused == 1
    assert _count(graph, "QMoE") == 1
    # Only the router gate and the four shared-expert projections remain.
    assert _count(graph, "MatMulNBits") == 1 + 4
    # The dense routing/mask machinery is gone.
    assert _count(graph, "TopK") == 0
    assert _count(graph, "Softmax") == 0
    assert _count(graph, "Equal") == 0
    assert _count(graph, "ReduceSum") == 0


def test_qmoe_weights_are_bit_identical_to_concatenated_experts() -> None:
    model, quants, _ = _build_dense_graph()
    fuse_dense_moe_to_qmoe(model)
    qmoe = next(n for n in model.graph if n.op_type == "QMoE")

    fc1_w = qmoe.inputs[2].const_value.numpy()
    fc2_w = qmoe.inputs[5].const_value.numpy()
    fc1_z = qmoe.inputs[11].const_value.numpy()
    fc2_z = qmoe.inputs[12].const_value.numpy()

    for e in range(E):
        expected_fc1 = np.concatenate(
            [
                quants[f"g{e}"].weight.reshape(INTER, -1),
                quants[f"u{e}"].weight.reshape(INTER, -1),
            ],
            axis=0,
        )
        np.testing.assert_array_equal(fc1_w[e], expected_fc1)
        np.testing.assert_array_equal(fc2_w[e], quants[f"d{e}"].weight.reshape(H, -1))
        expected_z = np.concatenate(
            [quants[f"g{e}"].zero_points, quants[f"u{e}"].zero_points], axis=0
        )
        np.testing.assert_array_equal(fc1_z[e], expected_z)
        np.testing.assert_array_equal(fc2_z[e], quants[f"d{e}"].zero_points)


@pytest.mark.parametrize(
    "activation_dtype",
    [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16],
)
def test_scales_match_activation_dtype(activation_dtype: ir.DataType) -> None:
    model, quants, _ = _build_dense_graph(activation_dtype)
    fuse_dense_moe_to_qmoe(model)
    qmoe = next(n for n in model.graph if n.op_type == "QMoE")

    fc1_s = qmoe.inputs[3].const_value.numpy()
    fc2_s = qmoe.inputs[6].const_value.numpy()
    assert qmoe.inputs[3].dtype == activation_dtype
    assert qmoe.inputs[6].dtype == activation_dtype
    assert fc1_s.dtype == activation_dtype.numpy()
    assert fc2_s.dtype == activation_dtype.numpy()
    for e in range(E):
        expected = np.concatenate(
            [
                quants[f"g{e}"].scales.reshape(INTER, -1),
                quants[f"u{e}"].scales.reshape(INTER, -1),
            ],
            axis=0,
        ).astype(activation_dtype.numpy())
        np.testing.assert_array_equal(fc1_s[e], expected)


def test_rewritten_forward_matches_dense_forward() -> None:
    model, quants, hidden = _build_dense_graph()
    expected = _dense_reference(quants, hidden)
    fuse_dense_moe_to_qmoe(model)
    got = _qmoe_reference(model.graph, hidden)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "activation_dtype",
    [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16],
)
def test_router_probs_cast_to_activation_dtype(activation_dtype: ir.DataType) -> None:
    model, _, _ = _build_dense_graph(activation_dtype)
    fuse_dense_moe_to_qmoe(model)
    graph = model.graph
    qmoe = next(n for n in graph if n.op_type == "QMoE")
    router_probs = qmoe.inputs[1]
    cast = router_probs.producer()
    assert cast.op_type == "Cast"
    assert cast.attributes["to"].value == activation_dtype.value


def test_attributes_match_qmoe_abi() -> None:
    model, _, _ = _build_dense_graph()
    fuse_dense_moe_to_qmoe(model)
    qmoe = next(n for n in model.graph if n.op_type == "QMoE")
    attrs = qmoe.attributes
    assert qmoe.domain == "com.microsoft"
    assert attrs["k"].value == K
    assert attrs["expert_weight_bits"].value == BITS
    assert attrs["block_size"].value == BLOCK
    assert attrs["swiglu_fusion"].value == 2
    assert attrs["normalize_routing_weights"].value == 1
    assert attrs["activation_type"].value == "swiglu"
    assert attrs["quant_type"].value == "int"
    assert attrs["weights_prepacked"].value == 0


@pytest.mark.parametrize(
    ("bits", "block_size", "expected"),
    [
        (4, 16, True),
        (4, 32, True),
        (4, 128, True),
        (4, 8, False),  # block_size < 16
        (4, 24, False),  # not a power of two
        (8, 32, False),  # 8-bit packing not byte-compatible with the reuse
        (2, 16, False),
    ],
)
def test_qmoe_abi_supported(bits: int, block_size: int, expected: bool) -> None:
    assert _qmoe_abi_supported(bits, block_size) is expected


def test_unsupported_geometry_keeps_dense_fallback() -> None:
    """A non-power-of-two block_size must not emit an unrunnable QMoE node."""
    model, _, _ = _build_dense_graph()
    graph = model.graph
    # Force every expert down_proj into an ABI-invalid geometry.
    for node in graph:
        if node.op_type == "MatMulNBits" and ".down_proj" in (node.name or ""):
            node.attributes["block_size"] = ir.AttrInt64("block_size", 24)

    fused = fuse_dense_moe_to_qmoe(model)

    assert fused == 0
    assert _count(graph, "QMoE") == 0
    # Dense routing machinery is preserved (still runnable).
    assert _count(graph, "TopK") == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
