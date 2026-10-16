# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parameter-free, BF16-dot / FP32-state selective scan for explicit prefill.

Replicates the 128-lane chunk algorithm of ``mamba_ssm``'s SSD prefill,
not the token recurrence. Inputs are post-convolution x/B/C and prepared
FP32 time steps. Readout is FP32 *before* the external D*x skip, gate and
normalization. This component neither selects a cache phase nor applies masks.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._scan_utils import create_body_graph, rename_subgraph_values


def _single_round_multiply_add(
    op: OpBuilder, x: ir.Value, scale: ir.Value, addend: ir.Value
) -> ir.Value:
    # ONNX has no elementwise FMA. Widen only this expression to reproduce
    # the source's single FP32 rounding; no widened state is stored or fed back.
    result: ir.Value = op.Cast(
        op.Add(
            op.Mul(op.Cast(x, to=ir.DataType.DOUBLE), op.Cast(scale, to=ir.DataType.DOUBLE)),
            op.Cast(addend, to=ir.DataType.DOUBLE),
        ),
        to=ir.DataType.FLOAT,
    )
    result.dtype = ir.DataType.FLOAT
    return result


def _chunk_cumsum(op: OpBuilder, dt: ir.Value, a: ir.Value) -> ir.Value:
    """Pinned 4x32-lane inclusive scan, including the padded lane 127."""
    product = op.Mul(dt, a)  # (B, H, 128), FP32
    lanes = list(range(128))
    shifted = op.Gather(product, [max(i - 1, 0) for i in lanes], axis=-1)
    result = op.Where(
        op.Greater(op.Constant(value_ints=[i % 32 for i in lanes]), op.Constant(value_int=0)),
        _single_round_multiply_add(op, dt, a, shifted),
        product,
    )
    for distance in (2, 4, 8, 16):
        shifted = op.Gather(result, [max(i - distance, 0) for i in lanes], axis=-1)
        result = op.Where(
            op.GreaterOrEqual(
                op.Constant(value_ints=[i % 32 for i in lanes]),
                op.Constant(value_int=distance),
            ),
            op.Add(result, shifted),
            result,
        )
    # Warp totals are combined left-to-right, then added to each warp's
    # local prefix. A generic CumSum need not preserve this association.
    warps = [op.Slice(result, [i], [i + 32], [-1]) for i in (0, 32, 64, 96)]
    prefix = op.Slice(warps[0], [31], [32], [-1])
    combined = [warps[0]]
    for warp in warps[1:]:
        combined.append(op.Add(warp, prefix))
        prefix = op.Add(prefix, op.Slice(warp, [31], [32], [-1]))
    output: ir.Value = op.Concat(*combined, axis=-1)
    output.dtype = ir.DataType.FLOAT
    return output


def _decay_exp(op: OpBuilder, x: ir.Value) -> ir.Value:
    # The pinned CUDA source lowers exp(x) to exp2(FP32(x * log2(e))).
    result: ir.Value = op.Pow(
        op.Constant(value_float=2.0),
        op.Mul(x, op.Constant(value_float=1.4426950408889634)),
    )
    result.dtype = ir.DataType.FLOAT
    return result


def _bf16_dot(op: OpBuilder, left: ir.Value, right: ir.Value) -> ir.Value:
    """Batched BF16 dot with an actual FP32 accumulator/output store."""
    op.builder.root.graph.opset_imports.setdefault("com.microsoft", 1)
    left_shape = op.Shape(left, start=-2)
    right_last = op.Shape(right, start=-1)
    left_flat = op.Reshape(left, op.Concat(op.Constant(value_ints=[-1]), left_shape, axis=0))
    right_flat = op.Reshape(
        right,
        op.Concat(op.Constant(value_ints=[-1]), op.Shape(right, start=-2), axis=0),
    )
    inputs = [
        ir.Value(
            name="dot_left",
            type=ir.TensorType(ir.DataType.BFLOAT16),
            shape=ir.Shape(["rows", "reduction"]),
        ),
        ir.Value(
            name="dot_right",
            type=ir.TensorType(ir.DataType.BFLOAT16),
            shape=ir.Shape(["reduction", "columns"]),
        ),
    ]
    body, builder = create_body_graph([], inputs, "bf16_float32_dot")
    # This schema explicitly supports BF16 inputs and FLOAT output. Unlike
    # BF16 MatMul it does not round the accumulator to the operand dtype.
    result = builder.op.GemmFloat8(
        *inputs, dtype=int(ir.DataType.FLOAT), _domain="com.microsoft"
    )
    result.dtype = ir.DataType.FLOAT
    result.shape = ir.Shape(["rows", "columns"])
    body.opset_imports["com.microsoft"] = 1
    body.outputs.append(result)
    rename_subgraph_values(body, "dot_")
    dots = op.Scan(left_flat, right_flat, body=body, num_scan_inputs=2)
    shape = op.Concat(
        op.Shape(left, end=-1),
        right_last,
        axis=0,
    )
    output: ir.Value = op.Reshape(dots, shape)
    output.dtype = ir.DataType.FLOAT
    return output


class ChunkedPrefillScan(nn.Module):
    """Explicit 128-lane prefill with BF16 operand stores and FP32 carry.

    ``forward(op, x, dt, a, b, c, initial_state=None)`` accepts:

    * x: BF16 (batch, time, num_heads, head_dim)
    * dt: prepared FP32 (batch, time, num_heads), nonnegative
    * a: negative FP32 (num_heads,), **not** A_log
    * b/c: BF16 (batch, time, num_groups, state_dim)
    * initial_state: optional FP32 (batch, num_heads, head_dim, state_dim)

    Returns FP32 readout (batch, time, num_heads, head_dim) and final state
    (batch, num_heads, head_dim, state_dim). State orientation follows SSD;
    a block storing (state_dim, head_dim) must transpose at its boundary.

    dt must already include FP32 dt+bias, threshold-20 softplus, and clamp.
    Only *artificial* chunk padding is zeroed here, after preparation.
    Real masked tokens must retain their B/C/dt; no reset/packed-sequence
    semantics are implemented. This is not a cached-update implementation.
    """

    def __init__(self, num_heads: int, num_groups: int, head_dim: int, state_dim: int):
        super().__init__()
        if min(num_heads, num_groups, head_dim, state_dim) <= 0:
            raise ValueError("All dimensions must be positive")
        if num_heads % num_groups:
            raise ValueError("num_heads must be divisible by num_groups")
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.state_dim = state_dim

    def _body(self) -> ir.Graph:
        h, g, p, n = self.num_heads, self.num_groups, self.head_dim, self.state_dim

        def value(name: str, shape: list[int | str], dtype: ir.DataType) -> ir.Value:
            return ir.Value(name=name, shape=ir.Shape(shape), type=ir.TensorType(dtype))

        state = value("chunk_incoming_state", ["batch", h, p, n], ir.DataType.FLOAT)
        x = value("chunk_x", ["batch", 128, h, p], ir.DataType.BFLOAT16)
        b = value("chunk_b", ["batch", 128, g, n], ir.DataType.BFLOAT16)
        c = value("chunk_c", ["batch", 128, g, n], ir.DataType.BFLOAT16)
        dt = value("chunk_dt", ["batch", 128, h], ir.DataType.FLOAT)
        a = value("chunk_a", [h], ir.DataType.FLOAT)
        # A is a loop invariant carried unchanged, rather than a captured
        # outer-scope value (Scan shape/type inference stays self-contained).
        graph, builder = create_body_graph([state, a], [x, dt, b, c], "chunked_prefill")
        op = builder.op
        dt = op.Transpose(dt, perm=[0, 2, 1])  # (B, H, 128)
        a_row = op.Unsqueeze(a, [0, 2])
        decay = _chunk_cumsum(op, dt, a_row)
        decay.name = "chunk_log_decay"
        total = op.Slice(decay, [127], [128], [-1])
        zero = op.Constant(value_float=0.0)
        scale = op.Mul(_decay_exp(op, op.Min(op.Sub(total, decay), zero)), dt)
        # Group selection is head // heads_per_group, never interleaving.
        indices = [head // (h // g) for head in range(h)]
        b_groups = op.Transpose(b, perm=[0, 2, 1, 3])
        c_groups = op.Transpose(c, perm=[0, 2, 1, 3])
        b_heads = op.Gather(b_groups, indices, axis=1)
        c_heads = op.Gather(c_groups, indices, axis=1)
        x_heads = op.Transpose(x, perm=[0, 2, 1, 3])
        # Weighted B is rounded as a complete operand, not B or dt*x alone.
        weighted_b = op.Cast(
            op.Mul(op.Cast(b_heads, to=ir.DataType.FLOAT), op.Unsqueeze(scale, [-1])),
            to=ir.DataType.BFLOAT16,
        )
        weighted_b.name = "chunk_scaled_b"
        local = _bf16_dot(
            op,
            op.Transpose(x_heads, perm=[0, 1, 3, 2]),
            weighted_b,
        )
        local.name = "chunk_local_state"
        new_state = _single_round_multiply_add(
            op, state, op.Unsqueeze(_decay_exp(op, total), [-1]), local
        )
        new_state.name = "chunk_final_state"
        new_state.shape = state.shape
        cb = _bf16_dot(
            op,
            c_groups,
            op.Transpose(b_groups, perm=[0, 1, 3, 2]),
        )
        cb.name = "chunk_cb"
        carry_operand = op.Cast(state, to=ir.DataType.BFLOAT16)
        carry = op.Mul(
            _bf16_dot(
                op,
                c_heads,
                op.Transpose(carry_operand, perm=[0, 1, 3, 2]),
            ),
            op.Unsqueeze(_decay_exp(op, decay), [-1]),
        )
        difference = op.Sub(op.Unsqueeze(decay, [-1]), op.Unsqueeze(decay, [-2]))
        coefficient = op.Mul(
            op.Mul(op.Gather(cb, indices, axis=1), _decay_exp(op, op.Min(difference, zero))),
            op.Unsqueeze(dt, [-2]),
        )
        causal = ir.tensor([[q >= k for k in range(128)] for q in range(128)])
        coefficient = op.Cast(
            op.Where(op.Constant(value=causal), coefficient, zero), to=ir.DataType.BFLOAT16
        )
        coefficient.name = "chunk_coefficient"
        output = op.Transpose(
            op.Add(
                carry,
                _bf16_dot(op, coefficient, x_heads),
            ),
            perm=[0, 2, 1, 3],
        )
        output.name = "chunk_readout"
        graph.outputs.extend([new_state, op.Identity(a), output])
        rename_subgraph_values(graph, "prefill_")
        return graph

    def forward(
        self,
        op: OpBuilder,
        x: ir.Value,
        dt: ir.Value,
        a: ir.Value,
        b: ir.Value,
        c: ir.Value,
        initial_state: ir.Value | None = None,
    ) -> tuple[ir.Value, ir.Value]:
        # Subgraph imports must also be declared by the enclosing model.
        op.builder.root.graph.opset_imports.setdefault("com.microsoft", 1)
        for name, tensor, dtype, rank in (
            ("x", x, ir.DataType.BFLOAT16, 4),
            ("dt", dt, ir.DataType.FLOAT, 3),
            ("a", a, ir.DataType.FLOAT, 1),
            ("b", b, ir.DataType.BFLOAT16, 4),
            ("c", c, ir.DataType.BFLOAT16, 4),
            ("initial_state", initial_state, ir.DataType.FLOAT, 4),
        ):
            if tensor is not None:
                if tensor.dtype != dtype:
                    raise TypeError(f"{name} must have dtype {dtype}")
                if tensor.shape is None or len(tensor.shape) != rank:
                    raise ValueError(f"{name} must have rank {rank}")
        expected_dimensions = (
            (x, {2: self.num_heads, 3: self.head_dim}),
            (dt, {2: self.num_heads}),
            (a, {0: self.num_heads}),
            (b, {2: self.num_groups, 3: self.state_dim}),
            (c, {2: self.num_groups, 3: self.state_dim}),
            (initial_state, {1: self.num_heads, 2: self.head_dim, 3: self.state_dim}),
        )
        for tensor, dimensions in expected_dimensions:
            if tensor is not None and tensor.shape is not None:
                for axis, size in dimensions.items():
                    actual = tensor.shape[axis]
                    if isinstance(actual, int) and actual != size:
                        raise ValueError(f"Dimension {axis} must be {size}, got {actual}")
        if x.shape is not None and x.shape[1] == 0:
            raise ValueError("Prefill requires at least one token")
        batch, time = op.Shape(x, start=0, end=1), op.Shape(x, start=1, end=2)
        padding = op.Mod(op.Sub(op.Constant(value_ints=[128]), op.Mod(time, [128])), [128])

        def chunks(tensor: ir.Value, tail: list[int]) -> ir.Value:
            rank = len(tail) + 2
            pads = op.Concat(
                op.Constant(value_ints=[0] * (rank + 1)),
                padding,
                op.Constant(value_ints=[0] * (rank - 2)),
                axis=0,
            )
            # Some CUDA providers lack BF16 Pad. The exact widening and
            # restore do not change any operand bits (including signed zero).
            padded = op.CastLike(op.Pad(op.Cast(tensor, to=ir.DataType.FLOAT), pads), tensor)
            shape = op.Concat(batch, op.Constant(value_ints=[-1, 128, *tail]), axis=0)
            # Scan iterates over chunks, not tokens.
            result: ir.Value = op.Transpose(
                op.Reshape(padded, shape), perm=[1, 0, *range(2, rank + 1)]
            )
            return result

        if initial_state is None:
            shape = op.Concat(
                batch,
                op.Constant(value_ints=[self.num_heads, self.head_dim, self.state_dim]),
                axis=0,
            )
            initial_state = op.ConstantOfShape(shape, value=ir.tensor([0.0]))
        final, _, readouts = op.Scan(
            initial_state,
            a,
            chunks(x, [self.num_heads, self.head_dim]),
            chunks(dt, [self.num_heads]),
            chunks(b, [self.num_groups, self.state_dim]),
            chunks(c, [self.num_groups, self.state_dim]),
            body=self._body(),
            num_scan_inputs=4,
            _outputs=3,
        )
        readouts = op.Transpose(readouts, perm=[1, 0, 2, 3, 4])
        readouts = op.Reshape(readouts, [0, -1, self.num_heads, self.head_dim])
        readout = op.Slice(readouts, [0], time, [1])
        readout.dtype = final.dtype = ir.DataType.FLOAT
        readout.shape, final.shape = x.shape, initial_state.shape
        return readout, final
