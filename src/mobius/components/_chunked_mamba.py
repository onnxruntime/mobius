# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit-phase Mamba2 candidate: chunked prefill, unchanged FP32 recurrence."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder

from mobius.components._chunked_prefill import ChunkedPrefillScan
from mobius.components._mamba_block import Float32SiLUMamba2Block


def validate_cache_phase(phase: ir.Value | None) -> ir.Value:
    """Require the caller-owned BOOL scalar, never infer phase from token count."""
    if phase is None:
        raise ValueError("has_previous_state must be explicitly supplied")
    if phase.dtype != ir.DataType.BOOL or phase.shape is None or len(phase.shape) != 0:
        raise TypeError("has_previous_state must be a BOOL scalar []")
    return phase


class ChunkedPrefillMamba2Block(Float32SiLUMamba2Block):
    """Opt-in BF16 block with explicit prefill versus cached-update semantics.

    Preserves all learned parameter paths and the FP32 D-before-native-store
    suffix. Prefill ignores dirty convolution/SSM caches. Cached execution
    consumes native (B,H,N,P) FP32 state and rejects sequences longer than one.
    """

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
        padding_mask: ir.Value | None = None,
        has_previous_state: ir.Value | None = None,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        phase = validate_cache_phase(has_previous_state)
        if hidden_states.dtype != ir.DataType.BFLOAT16:
            raise TypeError("Chunked prefill candidate requires BF16 activations")
        if ssm_state.dtype != ir.DataType.FLOAT:
            raise TypeError("Chunked prefill candidate requires FP32 cached SSM state")
        if conv_state.dtype != ir.DataType.BFLOAT16:
            raise TypeError("Chunked prefill candidate requires BF16 convolution cache")
        if ssm_state.shape is None or len(ssm_state.shape) != 4:
            raise TypeError("SSM cache requires rank4 (batch, heads, state_dim, head_dim)")
        for dimension, expected in zip(
            ssm_state.shape[1:], (self.num_heads, self.d_state, self.d_head)
        ):
            if isinstance(dimension, int) and dimension != expected:
                raise ValueError("SSM cache dimensions violate the native state orientation")
        if conv_state.shape is None or len(conv_state.shape) != 3:
            raise TypeError("Convolution cache requires rank3 (batch, channels, window)")
        for dimension, expected in zip(
            conv_state.shape[1:], (self.conv_dim, self.conv_kernel - 1)
        ):
            if isinstance(dimension, int) and dimension != expected:
                raise ValueError("Convolution cache dimensions violate the native window ABI")
        conv_shape = conv_state.shape
        zeros = op.CastLike(
            op.ConstantOfShape(op.Shape(conv_state), value=ir.tensor([0.0])), conv_state
        )
        conv_state = op.Where(phase, conv_state, zeros)
        conv_state.shape = conv_shape
        result: tuple[ir.Value, ir.Value, ir.Value] = super().forward(
            op, hidden_states, conv_state, ssm_state, padding_mask, phase
        )
        result[1].shape = conv_shape
        return result

    def _apply_conv_padding_mask(
        self, op: OpBuilder, activated: ir.Value, padding_mask: ir.Value
    ) -> ir.Value:
        # Real masked tokens still have B/C/dt and do not reset the SSM state.
        x, bc = op.Split(
            activated,
            [self.d_inner, 2 * self.n_groups * self.d_state],
            axis=-1,
            _outputs=2,
        )
        result: ir.Value = op.Concat(op.Mul(x, op.CastLike(padding_mask, x)), bc, axis=-1)
        return result

    def _prefill_dt(self, op: OpBuilder, dt_raw: ir.Value, bias: ir.Value) -> ir.Value:
        value = op.Add(op.Cast(dt_raw, to=ir.DataType.FLOAT), bias)
        threshold = op.Constant(value_float=20.0)
        dt: ir.Value = op.Where(
            op.LessOrEqual(value, threshold),
            op.Log(op.Add(op.Exp(value), op.Constant(value_float=1.0))),
            value,
        )
        if self.time_step_min > 0.0 or self.time_step_max is not None:
            dt = op.Clip(
                dt,
                op.Constant(value_float=self.time_step_min),
                op.Constant(value_float=self.time_step_max)
                if self.time_step_max is not None
                else None,
            )
        dt.dtype = ir.DataType.FLOAT
        return dt

    def _ssm_readout(
        self,
        op: OpBuilder,
        x_hidden: ir.Value,
        b_mat: ir.Value,
        c_mat: ir.Value,
        dt_raw: ir.Value,
        ssm_state: ir.Value,
        has_previous_state: ir.Value | None = None,
        prepared_dt: ir.Value | None = None,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        phase = validate_cache_phase(has_previous_state)
        if prepared_dt is not None:
            raise ValueError("Phase-aware block owns time-step preparation")
        # Realize these captured parameters in the enclosing module scope,
        # never under whichever If branch happened to be constructed first.
        bias = op.Cast(self.dt_bias, to=ir.DataType.FLOAT)
        a = op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)))
        x_4d = op.Reshape(
            op.Cast(x_hidden, to=ir.DataType.FLOAT),
            [0, 0, self.num_heads, self.d_head],
        )  # (B, T, H, P), FP32 for the unchanged D-skip suffix

        def cached(branch_op: OpBuilder) -> tuple[ir.Value, ir.Value]:
            # A selected cached S>1 branch must fail, not invent a source
            # recurrence. This reshape is an executable shape guard even
            # when the graph's sequence dimension is symbolic.
            single = branch_op.Reshape(x_hidden, [0, 1, self.d_inner])
            single.dtype = x_hidden.dtype
            dt = branch_op.Softplus(
                branch_op.Add(branch_op.Cast(dt_raw, to=ir.DataType.FLOAT), bias)
            )
            # Cached source has no prefill dt_limit. Keep the existing FP32
            # LinearAttention recurrence/readout, without prefill operand stores.
            readout, state, _ = super(ChunkedPrefillMamba2Block, self)._ssm_readout(
                branch_op, single, b_mat, c_mat, dt_raw, ssm_state, prepared_dt=dt
            )
            return readout, state

        def prefill(branch_op: OpBuilder) -> tuple[ir.Value, ir.Value]:
            x = branch_op.Reshape(x_hidden, [0, 0, self.num_heads, self.d_head])
            b = branch_op.Reshape(b_mat, [0, 0, self.n_groups, self.d_state])
            c = branch_op.Reshape(c_mat, [0, 0, self.n_groups, self.d_state])
            x.dtype = b.dtype = c.dtype = ir.DataType.BFLOAT16
            dt = self._prefill_dt(branch_op, dt_raw, bias)
            scan = ChunkedPrefillScan(self.num_heads, self.n_groups, self.d_head, self.d_state)
            readout, state = scan(branch_op, x, dt, a, b, c)
            # Scan state (B,H,P,N) -> the existing native cache (B,H,N,P).
            return (
                branch_op.Reshape(readout, [0, 0, self.d_inner]),
                branch_op.Transpose(state, perm=[0, 1, 3, 2]),
            )

        branches = []
        for name, callback in (("cached", cached), ("prefill", prefill)):
            op.builder.push_module(name)
            try:
                branches.append(
                    op.builder.subgraph(
                        callback,
                        inputs=[],
                        outputs=[
                            ir.Value(name=f"{name}_readout"),
                            ir.Value(name=f"{name}_state"),
                        ],
                        name=f"explicit_phase_{name}",
                    )
                )
            finally:
                op.builder.pop_module()
        readout, state = op.If(
            phase, then_branch=branches[0], else_branch=branches[1], _outputs=2
        )
        readout.dtype = state.dtype = ir.DataType.FLOAT
        readout.shape = x_hidden.shape
        state.shape = ssm_state.shape
        return readout, state, x_4d
