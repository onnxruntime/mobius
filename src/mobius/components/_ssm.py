# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Selective State Space Model (S6) component for Mamba architectures.

Implements the core selective scan recurrence from the Mamba paper.
Data-dependent B, C matrices enable content-aware state transitions,
distinguishing this from classical (fixed-parameter) SSMs.

Single-token decode recurrence:
    dt = softplus(dt_proj(x_proj(x)[:dt_rank]))
    B  = x_proj(x)[dt_rank : dt_rank + d_state]
    C  = x_proj(x)[dt_rank + d_state :]
    dA = exp(dt · A)                      # discretised decay
    state = dA * state + (dt · B) ⊗ x     # state update
    y = C · state + D * x                 # readout + skip

State carried across steps:
    ssm_state: (batch, d_inner, d_state)

Precision: The SSM recurrence (softplus, exp, state update) is computed
in float32 regardless of the model's compute dtype, matching HuggingFace
which upcasts A_log, dt, hidden_states, B, C to float32 for these ops.
The output and updated state are cast back to the input dtype.

HuggingFace reference: ``MambaMixer`` (SSM portion).
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import Linear
from mobius.components._scan_utils import create_body_graph, rename_subgraph_values


class SelectiveScan(nn.Module):
    """Core selective scan (S6) operation.

    Args:
        d_inner: Expanded hidden dimension (``expand * d_model``).
        d_state: SSM state dimension (typically 16).
        dt_rank: Rank of the time-step projection.
    """

    def __init__(self, d_inner: int, d_state: int, dt_rank: int):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Project x → (dt_raw, B, C): dt_rank + 2*d_state outputs
        self.x_proj = Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        # Project dt from reduced rank back to d_inner
        self.dt_proj = Linear(dt_rank, d_inner, bias=True)

        # A_log: (d_inner, d_state) — log of state decay matrix
        self.A_log = nn.Parameter([d_inner, d_state])
        # D: (d_inner,) — skip connection / feedthrough
        self.D = nn.Parameter([d_inner])

    def _project_ssm_params(self, op: OpBuilder, x_db):
        """Split x_proj output into dt_raw, B, C.

        Subclasses can override to apply extra transformations (e.g.
        layernorms on B/C/dt as in Jamba).

        Args:
            op: ONNX op builder.
            x_db: (batch, 1, dt_rank + 2*d_state) — x_proj output.

        Returns:
            Tuple of (dt_raw, b_mat, c_mat).
        """
        dt_raw, b_mat, c_mat = op.Split(
            x_db,
            [self.dt_rank, self.d_state, self.d_state],
            axis=-1,
            _outputs=3,
        )
        return dt_raw, b_mat, c_mat

    def forward(
        self,
        op: OpBuilder,
        x: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token selective scan step.

        Args:
            op: ONNX op builder.
            x: (batch, 1, d_inner) — input after conv1d + activation.
            ssm_state: (batch, d_inner, d_state) — carry state.

        Returns:
            y: (batch, 1, d_inner) — output.
            new_ssm_state: (batch, d_inner, d_state) — updated state.
        """
        # --- Project x to get dt, B, C ---
        # x_db: (batch, 1, dt_rank + 2*d_state)
        x_db = self.x_proj(op, x)

        # Split into dt, B, C (hookable for subclass overrides)
        dt_raw, b_mat, c_mat = self._project_ssm_params(op, x_db)
        # dt_raw: (batch, 1, dt_rank)
        # b_mat: (batch, 1, d_state)
        # c_mat: (batch, 1, d_state)

        # --- Compute time step dt ---
        # dt: (batch, 1, d_inner) via rank projection + softplus.
        # Upcast to fp32 for softplus (uses exp internally) to match
        # HuggingFace which computes the SSM recurrence in float32.
        dt = op.Cast(self.dt_proj(op, dt_raw), to=ir.DataType.FLOAT)
        dt = op.Softplus(dt)

        # --- Discretize state decay matrix A ---
        # a_neg = -exp(A_log) in fp32: (d_inner, d_state)
        a_neg = op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)))

        # Broadcast: dt (batch,1,d_inner,1) * A (1,1,d_inner,d_state)
        dt_4d = op.Unsqueeze(dt, [-1])  # (batch, 1, d_inner, 1)
        a_4d = op.Unsqueeze(a_neg, [0, 1])  # (1, 1, d_inner, d_state)
        # da = exp(dt * A) in fp32: (batch, 1, d_inner, d_state)
        da = op.Exp(op.Mul(dt_4d, a_4d))

        # --- Input contribution: dt * B * x (in fp32) ---
        b_4d = op.Unsqueeze(op.Cast(b_mat, to=ir.DataType.FLOAT), [2])
        dt_b = op.Mul(dt_4d, b_4d)  # (batch, 1, d_inner, d_state)
        x_4d = op.Unsqueeze(op.Cast(x, to=ir.DataType.FLOAT), [-1])
        db_x = op.Mul(dt_b, x_4d)  # (batch, 1, d_inner, d_state)

        # --- Squeeze seq dim (single token) ---
        da_t = op.Squeeze(da, [1])  # (batch, d_inner, d_state)
        db_x_t = op.Squeeze(db_x, [1])  # (batch, d_inner, d_state)

        # --- State update: h = dA * h_prev + dBx (in fp32) ---
        new_ssm_state = op.Add(op.Mul(da_t, op.Cast(ssm_state, to=ir.DataType.FLOAT)), db_x_t)

        # --- Readout: y = C · h ---
        c_t = op.Squeeze(op.Cast(c_mat, to=ir.DataType.FLOAT), [1])  # (batch, d_state)
        c_3d = op.Unsqueeze(c_t, [1])  # (batch, 1, d_state)
        # h: (batch, d_inner, d_state), C: (batch, 1, d_state)
        # → element-wise mul then sum over d_state → (batch, d_inner)
        y = op.ReduceSum(op.Mul(new_ssm_state, c_3d), [-1], keepdims=False)

        # --- Skip connection: y += D * x ---
        x_t = op.Squeeze(op.Cast(x, to=ir.DataType.FLOAT), [1])  # (batch, d_inner)
        y = op.Add(y, op.Mul(op.Cast(self.D, to=ir.DataType.FLOAT), x_t))

        # Restore seq dim and cast back to input dtype: (batch, 1, d_inner)
        y = op.CastLike(op.Unsqueeze(y, [1]), x)

        return y, op.CastLike(new_ssm_state, ssm_state)


class JambaSelectiveScan(SelectiveScan):
    """Jamba variant of SelectiveScan with layernorms on dt, B, C.

    Jamba applies additional RMSNorm to each of the SSM parameters
    (dt, B, C) after splitting from x_proj, before the dt projection.

    HuggingFace reference: ``JambaMambaMixer`` (SSM path).
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int,
        dt_rank: int,
        layer_norm_epsilon: float = 1e-5,
    ):
        super().__init__(d_inner, d_state, dt_rank)
        self.dt_layernorm = _RMSNorm(dt_rank, eps=layer_norm_epsilon)
        self.b_layernorm = _RMSNorm(d_state, eps=layer_norm_epsilon)
        self.c_layernorm = _RMSNorm(d_state, eps=layer_norm_epsilon)

    def _project_ssm_params(self, op, x_db):
        """Split + layernorm on dt, B, C."""
        dt_raw, b_mat, c_mat = op.Split(
            x_db,
            [self.dt_rank, self.d_state, self.d_state],
            axis=-1,
            _outputs=3,
        )
        dt_raw = self.dt_layernorm(op, dt_raw)
        b_mat = self.b_layernorm(op, b_mat)
        c_mat = self.c_layernorm(op, c_mat)
        return dt_raw, b_mat, c_mat

    def _repeat_for_channels(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        # Jamba Mamba-1 shares token-dependent B/C across all expanded channels.
        value = op.Unsqueeze(value, [2])  # (B, T, 1, d_state)
        value = op.Tile(value, [1, 1, self.d_inner, 1])
        return op.Reshape(value, [0, 0, self.d_inner * self.d_state])

    def forward(
        self,
        op: OpBuilder,
        x: ir.Value,
        ssm_state: ir.Value,
        padding_mask: ir.Value | None = None,
    ):
        """Run the complete multi-token Jamba selective scan."""
        dt_raw, b_mat, c_mat = self._project_ssm_params(op, self.x_proj(op, x))
        dt = op.Softplus(op.Cast(self.dt_proj(op, dt_raw), to=ir.DataType.FLOAT))
        del padding_mask

        x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
        decay = op.Mul(
            op.Unsqueeze(dt, [-1]),
            op.Unsqueeze(
                op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT))),
                [0, 1],
            ),
        )
        decay = op.Reshape(decay, [0, 0, self.d_inner * self.d_state])
        value = op.Mul(dt, x_f32)
        internal_state = op.Unsqueeze(
            op.Cast(ssm_state, to=ir.DataType.FLOAT),
            [-1],
        )
        output, present_state = op.LinearAttention(
            self._repeat_for_channels(op, op.Cast(c_mat, to=ir.DataType.FLOAT)),
            self._repeat_for_channels(op, op.Cast(b_mat, to=ir.DataType.FLOAT)),
            value,
            internal_state,
            decay,
            scale=1.0,
            q_num_heads=self.d_inner,
            kv_num_heads=self.d_inner,
            update_rule="gated",
            _domain="com.microsoft",
            _outputs=2,
        )
        output = op.Add(
            output,
            op.Mul(x_f32, op.Cast(self.D, to=ir.DataType.FLOAT)),
        )
        return (
            op.CastLike(output, x),
            op.CastLike(op.Squeeze(present_state, [-1]), ssm_state),
        )


class SequenceSelectiveScan(SelectiveScan):
    """Selective scan (S6) over a whole sequence, with no carried state.

    Unlike :class:`SelectiveScan`, which advances one token per call and
    threads ``ssm_state`` through the graph for autoregressive decode, this
    variant consumes the full ``(batch, seq_len, d_inner)`` activation in a
    single ONNX ``Scan`` and starts from a zero state.  It is the right
    building block for non-autoregressive encoders (speech enhancement,
    audio/vision Mamba backbones) that see the entire sequence at once.

    The recurrence body computes ``dA`` and ``dBx`` per step rather than
    materialising them for all timesteps up front.  Precomputing them would
    require two ``(batch, seq_len, d_inner, d_state)`` tensors, which for a
    typical audio backbone is orders of magnitude larger than the
    ``(batch, seq_len, d_inner)`` activations themselves.

    Parameters are identical to :class:`SelectiveScan` (``x_proj``,
    ``dt_proj``, ``A_log``, ``D``), so the two share checkpoints.

    Precision: the recurrence runs in float32 regardless of model dtype,
    matching the reference CUDA ``selective_scan_fn`` which accumulates the
    state in float32.
    """

    def forward(  # type: ignore[override]
        self,
        op: OpBuilder,
        x: ir.Value,
    ):
        """Run the selective scan over an entire sequence.

        Args:
            op: ONNX op builder.
            x: (batch, seq_len, d_inner) — input after conv1d + activation.

        Returns:
            y: (batch, seq_len, d_inner) — scan output including the ``D``
            skip connection.
        """
        # --- Project x to get dt, B, C for every timestep ---
        # x_db: (batch, seq_len, dt_rank + 2*d_state)
        x_db = self.x_proj(op, x)
        dt_raw, b_mat, c_mat = self._project_ssm_params(op, x_db)

        # dt: (batch, seq_len, d_inner). Upcast before softplus so the whole
        # recurrence (softplus/exp/state accumulation) stays in float32.
        dt = op.Softplus(op.Cast(self.dt_proj(op, dt_raw), to=ir.DataType.FLOAT))
        b_f32 = op.Cast(b_mat, to=ir.DataType.FLOAT)  # (batch, seq_len, d_state)
        c_f32 = op.Cast(c_mat, to=ir.DataType.FLOAT)  # (batch, seq_len, d_state)
        x_f32 = op.Cast(x, to=ir.DataType.FLOAT)  # (batch, seq_len, d_inner)

        # a_neg = -exp(A_log): (d_inner, d_state), the continuous-time decay.
        a_neg = op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)))

        # Zero initial state: (batch, d_inner, d_state). The batch dim is
        # dynamic, so build the shape from the activation at run time.
        batch_dim = op.Shape(x, start=0, end=1)
        state_shape = op.Concat(
            batch_dim,
            op.Constant(value_ints=[self.d_inner, self.d_state]),
            axis=0,
        )
        # ONNX's ConstantOfShape defaults to a float32 zero.
        initial_state = op.ConstantOfShape(state_shape)

        body = _build_sequence_scan_body()

        # Transpose to time-major for the Scan. `scan_input_axes` could name
        # axis 1 directly and skip these, but several execution providers
        # (e.g. the MLX plugin EP) only claim Scan nodes that iterate axis 0,
        # and falling back to CPU for the recurrence costs far more than the
        # transposes do.
        dt_t = op.Transpose(dt, perm=[1, 0, 2])  # (seq_len, batch, d_inner)
        b_t = op.Transpose(b_f32, perm=[1, 0, 2])  # (seq_len, batch, d_state)
        c_t = op.Transpose(c_f32, perm=[1, 0, 2])  # (seq_len, batch, d_state)
        x_t = op.Transpose(x_f32, perm=[1, 0, 2])  # (seq_len, batch, d_inner)

        # `a_neg` rides along as a pass-through carry so the body never has to
        # reach into the enclosing graph's scope for an initializer.
        _final_state, _a_out, y = op.Scan(
            initial_state,
            a_neg,
            dt_t,
            b_t,
            c_t,
            x_t,
            body=body,
            num_scan_inputs=4,
            _outputs=3,
        )
        # y: (seq_len, batch, d_inner) — restore batch-major.
        y = op.Transpose(y, perm=[1, 0, 2])  # (batch, seq_len, d_inner)

        # --- Skip connection: y += D * x ---
        y = op.Add(y, op.Mul(op.Cast(self.D, to=ir.DataType.FLOAT), x_f32))

        return op.CastLike(y, x)


def _build_sequence_scan_body() -> ir.Graph:
    """Build the ``Scan`` body for one timestep of the sequence selective scan.

    Body inputs (in order):
        1. ``state``: (batch, d_inner, d_state) — carry
        2. ``a_neg``: (d_inner, d_state) — carry, passed through unchanged
        3. ``dt_t``: (batch, d_inner) — scan input
        4. ``b_t``: (batch, d_state) — scan input
        5. ``c_t``: (batch, d_state) — scan input
        6. ``x_t``: (batch, d_inner) — scan input

    Body outputs:
        1. ``new_state``: (batch, d_inner, d_state) — carry
        2. ``a_neg_out``: (d_inner, d_state) — carry
        3. ``y_t``: (batch, d_inner) — scan output
    """
    f32 = ir.TensorType(ir.DataType.FLOAT)
    state_in = ir.Value(name="ssm_state", type=f32)
    a_in = ir.Value(name="a_neg", type=f32)
    dt_t = ir.Value(name="dt_t", type=f32)
    b_t = ir.Value(name="b_t", type=f32)
    c_t = ir.Value(name="c_t", type=f32)
    x_t = ir.Value(name="x_t", type=f32)

    body_graph, body_builder = create_body_graph(
        state_inputs=[state_in, a_in],
        scan_inputs=[dt_t, b_t, c_t, x_t],
        name="selective_scan_step",
    )
    bop = body_builder.op

    # dt_col: (batch, d_inner, 1) — broadcasts over the d_state axis.
    dt_col = bop.Unsqueeze(dt_t, [-1])

    # Discretised decay dA = exp(dt * A): (batch, d_inner, d_state).
    da = bop.Exp(bop.Mul(dt_col, bop.Unsqueeze(a_in, [0])))

    # Input contribution dBx = (dt * x) ⊗ B: (batch, d_inner, d_state).
    dbx = bop.Mul(
        bop.Mul(dt_col, bop.Unsqueeze(x_t, [-1])),
        bop.Unsqueeze(b_t, [1]),
    )

    # State update: h = dA * h_prev + dBx.
    new_state = bop.Add(bop.Mul(da, state_in), dbx)

    # Readout: y = C · h, summing over d_state → (batch, d_inner).
    y_t = bop.ReduceSum(bop.Mul(new_state, bop.Unsqueeze(c_t, [1])), [-1], keepdims=False)

    # The pass-through carry must be a distinct value from the graph input.
    a_out = bop.Identity(a_in)

    new_state.name = "ssm_state_out"
    a_out.name = "a_neg_out"
    y_t.name = "y_t"
    body_graph.outputs.extend([new_state, a_out, y_t])
    rename_subgraph_values(body_graph, "seq_scan_")
    return body_graph


class Mamba2Scan(nn.Module):
    """Multi-head selective scan for Mamba2/SSD architecture.

    Mamba2 uses a multi-head structure where each head independently
    scans its own ``d_head`` dimensions. B and C are shared across
    groups of heads (``n_groups``).

    Single-token decode recurrence per head ``h``:
        dt = softplus(dt_input + dt_bias)
        dA = exp(dt * A)
        state[h] = dA * state[h] + (dt * B) * x[h]
        y[h] = C . state[h] + D * x[h]

    State: (batch, num_heads, d_head, d_state)

    HuggingFace reference: ``BambaMixer`` (SSM portion).
    """

    def __init__(
        self,
        num_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int = 1,
        time_step_min: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.heads_per_group = num_heads // n_groups
        self.time_step_min = time_step_min

        self.A_log = nn.Parameter([num_heads])
        self.D = nn.Parameter([num_heads])
        self.dt_bias = nn.Parameter([num_heads])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        dt_input: ir.Value,
        b_mat: ir.Value,
        c_mat: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token Mamba2 multi-head scan step.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, num_heads * d_head) -- from conv output.
            dt_input: (batch, num_heads) -- time step from in_proj.
            b_mat: (batch, n_groups * d_state) -- B from conv output.
            c_mat: (batch, n_groups * d_state) -- C from conv output.
            ssm_state: (batch, num_heads, d_head, d_state) -- carry.

        Returns:
            y: (batch, num_heads * d_head) -- output.
            new_ssm_state: (batch, num_heads, d_head, d_state).
        """
        # dt = softplus(dt_input + dt_bias) in fp32: (batch, num_heads)
        # Upcast to fp32 for softplus/exp to match HuggingFace which
        # computes the SSM recurrence in float32.
        dt = op.Softplus(
            op.Add(
                op.Cast(dt_input, to=ir.DataType.FLOAT),
                op.Cast(self.dt_bias, to=ir.DataType.FLOAT),
            )
        )
        # Clamp dt to time_step_min (matches HF torch.clamp(dt, min=...))
        if self.time_step_min > 0.0:
            dt = op.Clip(dt, self.time_step_min)

        # A = -exp(A_log) in fp32: (num_heads,)
        a_neg = op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)))

        # Broadcast for state update (all in fp32)
        dt_4d = op.Unsqueeze(dt, [2, 3])  # (batch, num_heads, 1, 1)
        a_2d = op.Unsqueeze(a_neg, [0])  # (1, num_heads)
        a_4d = op.Unsqueeze(a_2d, [2, 3])  # (1, num_heads, 1, 1)
        da = op.Exp(op.Mul(dt_4d, a_4d))

        # Reshape hidden: (batch, num_heads, d_head)
        hidden_3d = op.Cast(
            op.Reshape(hidden_states, [0, self.num_heads, self.d_head]),
            to=ir.DataType.FLOAT,
        )

        # Expand B from groups to heads (in fp32)
        b_shape = [0, self.n_groups, 1, self.d_state]
        b_4d = op.Reshape(op.Cast(b_mat, to=ir.DataType.FLOAT), b_shape)
        expand_shape = [1, 1, self.heads_per_group, 1]
        b_expanded = op.Expand(b_4d, expand_shape)
        heads_shape = [0, self.num_heads, self.d_state]
        b_heads = op.Reshape(b_expanded, heads_shape)
        b_ssm = op.Unsqueeze(b_heads, [2])

        # dBx: dt * B * x (in fp32)
        dt_b = op.Mul(dt_4d, b_ssm)  # (batch, num_heads, 1, d_state)
        x_4d = op.Unsqueeze(hidden_3d, [3])
        db_x = op.Mul(dt_b, x_4d)  # (batch, num_heads, d_head, d_state)

        # State update: h = dA * h_prev + dBx (in fp32)
        new_ssm_state = op.Add(op.Mul(da, op.Cast(ssm_state, to=ir.DataType.FLOAT)), db_x)

        # Readout: y = C . h + D * x (in fp32)
        c_4d = op.Reshape(op.Cast(c_mat, to=ir.DataType.FLOAT), b_shape)
        c_expanded = op.Expand(c_4d, expand_shape)
        c_heads = op.Reshape(c_expanded, heads_shape)
        c_ssm = op.Unsqueeze(c_heads, [2])

        y = op.ReduceSum(
            op.Mul(new_ssm_state, c_ssm), [-1], keepdims=False
        )  # (batch, num_heads, d_head)

        # Skip: y += D * x
        d_3d = op.Unsqueeze(op.Cast(self.D, to=ir.DataType.FLOAT), [0, 2])
        y = op.Add(y, op.Mul(d_3d, hidden_3d))

        # Flatten and cast back to input dtype: (batch, num_heads * d_head)
        y = op.CastLike(
            op.Reshape(y, [0, self.num_heads * self.d_head]),
            hidden_states,
        )

        return y, op.CastLike(new_ssm_state, ssm_state)


class _RMSNorm(nn.Module):
    """Lightweight RMSNorm for SSM params (avoids circular import)."""

    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter([size])
        self._eps = eps

    def forward(self, op: OpBuilder, x: ir.Value):
        # Upcast to fp32 for variance computation (matching HF RMSNorm).
        x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
        variance = op.ReduceMean(op.Mul(x_f32, x_f32), [-1], keepdims=True)
        x_normed = op.Div(
            x_f32,
            op.Sqrt(op.Add(variance, self._eps)),
        )
        result = op.Mul(x_normed, op.Cast(self.weight, to=ir.DataType.FLOAT))
        return op.CastLike(result, x)
