# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mamba block component: Conv1D → SSM → gated output projection.

Implements the standard Mamba layer used in Mamba, Jamba, Bamba,
FalconMamba, and related architectures. Composes a causal depthwise
Conv1D with the SelectiveScan SSM and a SiLU-gated output path.

Architecture per layer:
    1. in_proj: x → (x_branch, z_gate)  [expansion to d_inner]
    2. Causal depthwise Conv1D on x_branch
    3. SiLU activation
    4. Selective scan (SSM) with recurrent state
    5. Output gating: y * SiLU(z)
    6. out_proj: project back to d_model

State carried across steps:
    conv_state:  (batch, d_inner, conv_kernel - 1)
    ssm_state:   (batch, d_inner, d_state)

Mamba2Block supports three multi-token modes controlled by
``flags.mamba_scan``:
    - ``"chunked_ssd"`` (default): parallel within chunks, O(T·CS) per
      chunk.  Matches HF ``torch_forward``.
    - ``"scan"``: ONNX Scan op iterating token-by-token.  Supports
      arbitrary seq_len but is sequential.
    - ``"single"``: single-token-only path (seq_len must be 1).
      Useful for debugging.

HuggingFace reference: ``MambaMixer``, ``BambaMixer``.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder
from onnxscript.onnx_types import FLOAT

from mobius._flags import flags
from mobius.components._common import INT64_MAX, Linear
from mobius.components._rms_norm import GatedRMSNorm
from mobius.components._ssm import Mamba2Scan, SelectiveScan


class _DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution with optional bias.

    Each input channel is convolved with its own kernel (groups=channels).
    Used for causal convolution in the Mamba block.
    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self.bias = nn.Parameter([channels]) if bias else None
        self._kernel_size = kernel_size
        self._channels = channels

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (batch, channels, seq_len)
        result = op.Conv(
            x,
            self.weight,
            kernel_shape=[self._kernel_size],
            strides=[1],
            pads=[0, 0],
            group=self._channels,
        )
        if self.bias is not None:
            # bias: (channels,) → (1, channels, 1) for broadcasting
            bias_3d = op.Unsqueeze(self.bias, [0, 2])
            result = op.Add(result, bias_3d)
        return result


# -----------------------------------------------------------------------
# Chunked SSD helpers — ONNX equivalents of the PyTorch helper functions
# -----------------------------------------------------------------------


def _segment_sum(
    op: builder.OpBuilder, x: ir.Value, chunk_size: int,
):
    """Stable segment sum via cumulative sums and masking.

    Input:  x with last dim = chunk_size  (..., chunk_size)
    Output: (..., chunk_size, chunk_size) lower-triangular cumsum.

    Matches HF's ``segment_sum()`` function.  ``chunk_size`` must be a
    compile-time constant so we can build the fixed triangular masks.
    """
    # Expand: (..., chunk_size) → (..., chunk_size, chunk_size)
    x_expanded = op.Unsqueeze(x, [-1])
    x_tiled = op.Expand(
        x_expanded,
        op.Concat(
            op.Shape(x), op.Constant(value_ints=[chunk_size]), axis=0,
        ),
    )

    # Strict lower-triangular mask (diagonal=-1)
    mask_strict = op.Constant(
        value=ir.tensor(
            np.tril(
                np.ones((chunk_size, chunk_size), dtype=np.float32),
                k=-1,
            )
        )
    )
    x_masked = op.Mul(x_tiled, mask_strict)

    # CumSum along the second-to-last axis
    cumsum = op.CumSum(x_masked, op.Constant(value_int=-2))

    # Full lower-triangular mask (including diagonal)
    mask_full_bool = op.Constant(
        value=ir.tensor(
            np.tril(
                np.ones((chunk_size, chunk_size), dtype=np.bool_), k=0,
            )
        )
    )
    neg_inf = op.Constant(
        value=ir.tensor(
            np.full(
                (chunk_size, chunk_size), float("-inf"), dtype=np.float32,
            )
        )
    )
    result = op.Where(mask_full_bool, cumsum, neg_inf)
    return result


def _segment_sum_dynamic(op: builder.OpBuilder, x: ir.Value):
    """Segment sum for dynamic-sized last dimension.

    Same logic as ``_segment_sum`` but uses ONNX ops (``Trilu``) to
    build masks dynamically, since nc+1 (number of chunks + 1) is not
    known at graph-build time.

    Input:  x with shape (..., K) where K is dynamic.
    Output: (..., K, K) lower-triangular cumsum.
    """
    K = op.Shape(x, start=-1, end=None)  # [K] as 1-d tensor

    # Expand: (..., K) → (..., K, K)
    x_expanded = op.Unsqueeze(x, [-1])
    new_shape = op.Concat(op.Shape(x), K, axis=0)
    x_tiled = op.Expand(x_expanded, new_shape)

    # Build KxK ones, then lower-triangular masks via Trilu
    KK_shape = op.Concat(K, K, axis=0)
    ones_KK = op.ConstantOfShape(
        KK_shape,
        value=ir.tensor(np.array([1.0], dtype=np.float32)),
    )
    # Strict lower triangle (diagonal=-1)
    lower_strict = op.Trilu(
        ones_KK, op.Constant(value_int=-1), upper=0,
    )
    x_masked = op.Mul(x_tiled, lower_strict)

    # CumSum along second-to-last axis
    cumsum = op.CumSum(x_masked, op.Constant(value_int=-2))

    # Full lower triangle (including diagonal)
    lower_full = op.Trilu(ones_KK, upper=0)
    lower_bool = op.Cast(lower_full, to=ir.DataType.BOOL)

    # Fill non-lower-triangle with -inf
    neg_inf_KK = op.ConstantOfShape(
        KK_shape,
        value=ir.tensor(np.array([float("-inf")], dtype=np.float32)),
    )
    result = op.Where(lower_bool, cumsum, neg_inf_KK)
    return result


class MambaBlock(nn.Module):
    """Standard Mamba layer: input projection → Conv1D → SSM → gated output.

    Args:
        d_model: Model hidden dimension.
        d_inner: Expanded inner dimension (typically ``expand * d_model``).
        d_state: SSM state dimension (typically 16).
        dt_rank: Rank of the SSM time-step projection.
            Defaults to ``ceil(d_model / 16)`` (Mamba convention).
        conv_kernel: Causal Conv1D kernel size (typically 4).
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        conv_kernel: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.conv_kernel = conv_kernel
        # Default dt_rank: ceil(d_model / 16) per Mamba convention
        self.dt_rank = dt_rank if dt_rank is not None else -(-d_model // 16)

        # Input projection: d_model → 2*d_inner (x_branch + z_gate)
        self.in_proj = Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise Conv1D (with bias, matching HuggingFace)
        self.conv1d = _DepthwiseConv1d(d_inner, conv_kernel, bias=True)

        # Core SSM component
        self.ssm = SelectiveScan(d_inner, d_state, self.dt_rank)

        # Output projection: d_inner → d_model
        self.out_proj = Linear(d_inner, d_model, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token forward pass for the Mamba layer.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, 1, d_model) — single token input.
            conv_state: (batch, d_inner, conv_kernel-1) — carry state.
            ssm_state: (batch, d_inner, d_state) — carry state.

        Returns:
            output: (batch, 1, d_model)
            new_conv_state: (batch, d_inner, conv_kernel-1)
            new_ssm_state: (batch, d_inner, d_state)
        """
        # --- Step 1: Input projection ---
        # projected: (batch, 1, 2*d_inner)
        projected = self.in_proj(op, hidden_states)

        # Split into x_branch and z_gate along last dim
        x_branch, z_gate = op.Split(
            projected,
            [self.d_inner, self.d_inner],
            axis=-1,
            _outputs=2,
        )
        # x_branch: (batch, 1, d_inner)
        # z_gate:   (batch, 1, d_inner)

        # --- Step 2: Causal Conv1D with state update ---
        # Transpose for conv: (batch, d_inner, 1)
        x_t = op.Transpose(x_branch, perm=[0, 2, 1])

        # Concatenate conv state + new token: (batch, d_inner, conv_kernel)
        conv_input = op.Concat(conv_state, x_t, axis=2)

        # Update conv state: drop oldest, keep last (conv_kernel-1)
        new_conv_state = op.Slice(
            conv_input, starts=[1], ends=[INT64_MAX], axes=[2],
        )

        # Apply depthwise conv: (batch, d_inner, 1)
        conv_out = self.conv1d(op, conv_input)

        # --- Step 3: SiLU activation ---
        conv_out = op.Mul(conv_out, op.Sigmoid(conv_out))

        # Transpose back: (batch, 1, d_inner)
        x_ssm = op.Transpose(conv_out, perm=[0, 2, 1])

        # --- Step 4: Selective scan ---
        y, new_ssm_state = self.ssm(op, x_ssm, ssm_state)
        # y: (batch, 1, d_inner)

        # --- Step 5: Output gating: y * SiLU(z) ---
        z_activated = op.Mul(z_gate, op.Sigmoid(z_gate))
        gated = op.Mul(y, z_activated)

        # --- Step 6: Output projection ---
        # output: (batch, 1, d_model)
        output = self.out_proj(op, gated)

        return output, new_conv_state, new_ssm_state


class Mamba2Block(nn.Module):
    """Mamba2/SSD block: in_proj → Conv1D → multi-head SSM → gated norm.

    Supports three multi-token modes controlled by ``flags.mamba_scan``:

    - **Chunked SSD** (default, ``"chunked_ssd"``): processes all
      tokens in parallel within chunks of ``chunk_size``, propagating
      SSM state across chunk boundaries.  Matches HF's
      ``torch_forward`` multi-token path.
    - **ONNX Scan** (``"scan"``): token-by-token iteration via the
      ONNX Scan op.  Supports arbitrary seq_len but is sequential.
    - **Single-token** (``"single"``): per-token recurrence
      (seq_len must be 1).  Useful for debugging numerical issues.

    Key differences from MambaBlock (Mamba1):
    - in_proj outputs [gate, xBC, dt] instead of [x, z]
    - Conv1D on wider xBC (conv_dim channels)
    - Multi-head SSM with grouped B/C
    - GatedRMSNorm instead of SiLU gating
    - dt direct from in_proj (no rank reduction), just bias

    HuggingFace reference: ``BambaMixer``, ``NemotronHMamba2Mixer``.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        num_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int = 1,
        chunk_size: int = 256,
        conv_kernel: int = 4,
        conv_bias: bool = True,
        proj_bias: bool = False,
        eps: float = 1e-5,
        norm_group_size: int | None = None,
        time_step_min: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.conv_kernel = conv_kernel
        self.heads_per_group = num_heads // n_groups
        self.time_step_min = time_step_min

        self.conv_dim = d_inner + 2 * n_groups * d_state

        proj_size = d_inner + self.conv_dim + num_heads
        self.in_proj = Linear(d_model, proj_size, bias=proj_bias)
        self.conv1d = _DepthwiseConv1d(
            self.conv_dim, conv_kernel, bias=conv_bias,
        )
        # SSM parameters live under self.ssm so that ONNX weight paths
        # stay as ``mamba.ssm.{A_log,D,dt_bias}`` — compatible with all
        # existing preprocess_weights rename rules.
        self.ssm = Mamba2Scan(
            num_heads, d_head, d_state, n_groups,
            time_step_min=time_step_min,
        )
        self.norm = GatedRMSNorm(
            d_inner, eps=eps, group_size=norm_group_size,
        )
        self.out_proj = Linear(d_inner, d_model, bias=proj_bias)

    # -----------------------------------------------------------------
    # Chunked SSD (multi-token parallel path)
    # -----------------------------------------------------------------

    def _chunked_ssd(
        self,
        op: builder.OpBuilder,
        x: ir.Value,
        dt: ir.Value,
        B_mat: ir.Value,
        C_mat: ir.Value,
        ssm_state_in: ir.Value,
    ):
        """Chunked SSD computation in ONNX ops.

        Implements the "ssd naive implementation without einsums" from
        HuggingFace's ``torch_forward``, translated to ONNX operations.

        All 5 stages of the SSD algorithm:
          1. Intra-chunk diagonal blocks (attention-like within each
             chunk).
          2. Inter-chunk state computation (B terms: how each chunk
             contributes to the running state).
          3. Inter-chunk SSM recurrence (A terms: decay across chunk
             boundaries).
          4. State-to-output per chunk (C terms: readout from
             propagated state).
          5. Combine intra-chunk and inter-chunk contributions.

        Args:
            op: ONNX op builder.
            x: (B, T, H, D) float32 — activated hidden (discretised).
            dt: (B, T, H) float32 — softplus(dt_raw + dt_bias).
            B_mat: (B, T, H, N) float32.
            C_mat: (B, T, H, N) float32.
            ssm_state_in: (B, H, D, N) — carry state.

        Returns:
            y: (B, T, H, D) float32 — output.
            new_ssm_state: (B, H, D, N).
        """
        CS = self.chunk_size
        H = self.num_heads
        D = self.d_head
        N = self.d_state

        # --- D residual: D[..., None] * x (before discretisation) ---
        D_param = op.Cast(self.ssm.D, to=ir.DataType.FLOAT)
        D_4d = op.Unsqueeze(D_param, [0, 1, 3])  # (1, 1, H, 1)
        D_residual = op.Mul(D_4d, x)  # (B, T, H, D)

        # --- Discretise ---
        # x = x * dt[..., None]: (B, T, H, D) * (B, T, H, 1)
        dt_4d = op.Unsqueeze(dt, [-1])
        x_disc = op.Mul(x, dt_4d)

        # A = -exp(A_log) in float32
        A_neg = op.Neg(
            op.Exp(op.Cast(self.ssm.A_log, to=ir.DataType.FLOAT))
        )
        # A_dt = A * dt: (H,) * (B, T, H) → (B, T, H)
        A_2d = op.Unsqueeze(A_neg, [0, 1])
        A_dt = op.Mul(A_2d, dt)

        # --- Reshape into chunks ---
        # Requires T divisible by CS (caller pads).
        x_chunked = op.Reshape(
            x_disc, op.Constant(value_ints=[0, -1, CS, H, D]),
        )  # (B, nc, CS, H, D)
        A_chunked = op.Reshape(
            A_dt, op.Constant(value_ints=[0, -1, CS, H]),
        )  # (B, nc, CS, H)
        B_chunked = op.Reshape(
            B_mat, op.Constant(value_ints=[0, -1, CS, H, N]),
        )
        C_chunked = op.Reshape(
            C_mat, op.Constant(value_ints=[0, -1, CS, H, N]),
        )

        # A_cumsum: cumsum of A within each chunk
        # (B, nc, CS, H) → permute to (B, H, nc, CS) for cumsum
        A_perm = op.Transpose(A_chunked, perm=[0, 3, 1, 2])
        A_cumsum = op.CumSum(A_perm, op.Constant(value_int=-1))

        # =============================================
        # 1. Intra-chunk (diagonal blocks)
        # =============================================
        # L = exp(segment_sum(A)): causal decay within chunk
        L = op.Exp(_segment_sum(op, A_perm, CS))  # (B,H,nc,CS,CS)

        # G = sum_n(C[l,n] * B[s,n]) — contraction over state dim
        C_exp = op.Unsqueeze(C_chunked, [3])  # (B,nc,CS,1,H,N)
        B_exp = op.Unsqueeze(B_chunked, [2])  # (B,nc,1,CS,H,N)
        G = op.ReduceSum(
            op.Mul(C_exp, B_exp), [-1], keepdims=False,
        )  # (B,nc,CS,CS,H)

        # M = G * L (permuted to match G layout)
        L_perm = op.Transpose(L, perm=[0, 2, 3, 4, 1])
        M = op.Mul(G, L_perm)  # (B, nc, CS, CS, H)

        # Y_diag = (M[...,None] * x_chunked[:,None]).sum(dim=3)
        M_exp = op.Unsqueeze(M, [-1])         # (B,nc,l,s,H,1)
        x_exp = op.Unsqueeze(x_chunked, [2])  # (B,nc,1,s,H,D)
        Y_diag = op.ReduceSum(
            op.Mul(M_exp, x_exp), [3], keepdims=False,
        )  # (B, nc, CS, H, D)

        # =============================================
        # 2. Inter-chunk state computation (B terms)
        # =============================================
        # decay_states = exp(A_last - A_cumsum)
        A_last = op.Slice(
            A_cumsum, starts=[-1], ends=[INT64_MAX], axes=[-1],
        )  # (B, H, nc, 1)
        decay_states = op.Exp(op.Sub(A_last, A_cumsum))

        # B_decay = B * decay_states (permuted)
        decay_perm = op.Transpose(
            decay_states, perm=[0, 2, 3, 1],
        )  # (B, nc, CS, H)
        decay_exp = op.Unsqueeze(decay_perm, [-1])
        B_decay = op.Mul(B_chunked, decay_exp)  # (B, nc, CS, H, N)

        # states = sum_over_chunk(B_decay * x_disc)
        B_decay_exp = op.Unsqueeze(B_decay, [-2])   # (B,nc,CS,H,1,N)
        x_disc_exp = op.Unsqueeze(x_chunked, [-1])  # (B,nc,CS,H,D,1)
        states = op.ReduceSum(
            op.Mul(B_decay_exp, x_disc_exp), [2], keepdims=False,
        )  # (B, nc, H, D, N)

        # =============================================
        # 3. Inter-chunk SSM recurrence (A terms)
        # =============================================
        # Prepend previous state along chunk dim
        prev = op.Unsqueeze(
            op.Cast(ssm_state_in, to=ir.DataType.FLOAT), [1],
        )  # (B, 1, H, D, N)
        states_cat = op.Concat(
            prev, states, axis=1,
        )  # (B, nc+1, H, D, N)

        # decay_chunk = exp(segment_sum(pad(A_ends, (1,0))))
        A_ends = op.Squeeze(
            op.Slice(
                A_cumsum, starts=[-1], ends=[INT64_MAX], axes=[-1],
            ),
            [-1],
        )  # (B, H, nc)
        A_ends_padded = op.Pad(
            A_ends,
            op.Constant(value_ints=[0, 0, 1, 0, 0, 0]),
            op.Constant(value_float=0.0),
        )  # (B, H, nc+1)
        decay_chunk = op.Exp(
            _segment_sum_dynamic(op, A_ends_padded),
        )  # (B, H, nc+1, nc+1)

        # Propagate state across chunks
        decay_chunk_t = op.Transpose(
            decay_chunk, perm=[0, 2, 3, 1],
        )  # (B, nc+1, nc+1, H)
        decay_exp2 = op.Unsqueeze(decay_chunk_t, [-1, -2])
        states_exp = op.Unsqueeze(states_cat, [1])
        new_states = op.ReduceSum(
            op.Mul(decay_exp2, states_exp), [2], keepdims=False,
        )  # (B, nc+1, H, D, N)

        # Split: first nc for output, last for carry
        states_out = op.Slice(
            new_states, starts=[0], ends=[-1], axes=[1],
        )  # (B, nc, H, D, N)
        new_ssm_state = op.Squeeze(
            op.Slice(
                new_states, starts=[-1], ends=[INT64_MAX], axes=[1],
            ),
            [1],
        )  # (B, H, D, N)

        # =============================================
        # 4. State → output per chunk (C terms)
        # =============================================
        state_decay_out = op.Exp(A_cumsum)  # (B, H, nc, CS)

        C_exp2 = op.Unsqueeze(C_chunked, [-2])    # (B,nc,CS,H,1,N)
        states_exp2 = op.Unsqueeze(states_out, [2])  # (B,nc,1,H,D,N)
        C_states_sum = op.ReduceSum(
            op.Mul(C_exp2, states_exp2), [-1], keepdims=False,
        )  # (B, nc, CS, H, D)

        sdo_perm = op.Transpose(
            state_decay_out, perm=[0, 2, 3, 1],
        )  # (B, nc, CS, H)
        sdo_exp = op.Unsqueeze(sdo_perm, [-1])
        Y_off = op.Mul(C_states_sum, sdo_exp)  # (B, nc, CS, H, D)

        # =============================================
        # 5. Combine
        # =============================================
        y = op.Add(Y_diag, Y_off)  # (B, nc, CS, H, D)

        # Reshape back to (B, T, H, D)
        bt_shape = op.Shape(x, start=0, end=2)
        y_shape = op.Concat(
            bt_shape, op.Constant(value_ints=[H, D]), axis=0,
        )
        y_reshaped = op.Reshape(y, y_shape)

        y_out = op.Add(y_reshaped, D_residual)  # (B, T, H, D)

        return y_out, new_ssm_state

    @staticmethod
    def _realize_submodule(
        parent_builder: builder.GraphBuilder,
        submodule: nn.Module,
    ) -> None:
        """Register a submodule's parameters as parent-graph initializers.

        When the chunked SSD path references ``self.ssm.A_log`` etc.
        directly (without calling ``self.ssm(...)``), the framework
        hasn't realized those parameters.  This helper pushes the naming
        context and calls ``_realize()`` on every parameter so they
        appear as graph initializers.
        """
        name = submodule._name or ""
        parent_builder.push_module(name, type(submodule).__qualname__)
        for param in submodule._parameters.values():
            param._realize(parent_builder)
        for child in submodule._modules.values():
            Mamba2Block._realize_submodule(parent_builder, child)
        parent_builder.pop_module()

    def _forward_chunked_ssd(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Multi-token forward using the chunked SSD algorithm.

        Processes the full sequence in parallel within chunks of
        ``chunk_size`` tokens, then propagates SSM state across chunk
        boundaries.  Matching HF's ``torch_forward`` multi-token path.

        Note: seq_len must be divisible by chunk_size (the task layer
        is responsible for padding if necessary).
        """
        # Realize SSM parameters so they are visible as graph
        # initializers when referenced directly (not via self.ssm()).
        self._realize_submodule(op.builder, self.ssm)
        H = self.num_heads
        D = self.d_head
        N = self.d_state

        # Step 1: Batch-project all tokens at once
        projected = self.in_proj(op, hidden_states)  # (B, T, proj)
        gate, x_bc, dt_raw = op.Split(
            projected,
            [self.d_inner, self.conv_dim, self.num_heads],
            axis=-1,
            _outputs=3,
        )

        # Step 2: Causal Conv1D over full sequence
        x_bc_t = op.Transpose(x_bc, perm=[0, 2, 1])
        # Pad left with K-1 zeros for causal convolution
        padded = op.Pad(
            x_bc_t,
            op.Constant(
                value_ints=[0, 0, self.conv_kernel - 1, 0, 0, 0],
            ),
            op.Constant(value_float=0.0),
        )  # (B, conv_dim, T+K-1)
        conv_out_raw = self.conv1d(op, padded)
        # SiLU activation
        conv_activated = op.Mul(
            conv_out_raw, op.Sigmoid(conv_out_raw),
        )
        hidden_B_C = op.Transpose(conv_activated, perm=[0, 2, 1])

        # Extract new conv_state: last K-1 positions from the combined
        # old state + new tokens.  For T >= K-1 we could just slice
        # x_bc_t, but for T < K-1 (e.g. T=1 during decode) we need
        # to include history from the old conv_state.
        conv_combined = op.Concat(
            conv_state, x_bc_t, axis=2,
        )  # (B, conv_dim, K-1+T)
        new_conv_state = op.Slice(
            conv_combined,
            starts=[-(self.conv_kernel - 1)],
            ends=[INT64_MAX],
            axes=[2],
        )  # (B, conv_dim, K-1)

        # Step 3: Split xBC → x, B, C
        gs = self.n_groups * self.d_state
        x_split, B_split, C_split = op.Split(
            hidden_B_C,
            [self.d_inner, gs, gs],
            axis=-1,
            _outputs=3,
        )

        # Step 4: Prepare SSM inputs
        dt_bias = op.Cast(self.ssm.dt_bias, to=ir.DataType.FLOAT)
        dt_bias_3d = op.Unsqueeze(dt_bias, [0, 1])
        dt = op.Softplus(
            op.Add(
                op.Cast(dt_raw, to=ir.DataType.FLOAT), dt_bias_3d,
            )
        )
        # Clamp dt to time_step_min (matches HF torch.clamp(dt, min=...))
        if self.time_step_min > 0.0:
            dt = op.Clip(dt, op.Constant(value_float=self.time_step_min))

        # Reshape x to (B, T, H, D)
        x_4d = op.Reshape(
            op.Cast(x_split, to=ir.DataType.FLOAT),
            op.Constant(value_ints=[0, 0, H, D]),
        )
        # Reshape B, C to (B, T, n_groups, N) then expand to heads
        B_grouped = op.Reshape(
            op.Cast(B_split, to=ir.DataType.FLOAT),
            op.Constant(value_ints=[0, 0, self.n_groups, N]),
        )
        C_grouped = op.Reshape(
            op.Cast(C_split, to=ir.DataType.FLOAT),
            op.Constant(value_ints=[0, 0, self.n_groups, N]),
        )
        # Expand groups → heads
        B_exp = op.Unsqueeze(B_grouped, [3])
        B_expand = op.Expand(
            B_exp,
            op.Concat(
                op.Shape(B_exp, start=0, end=3),
                op.Constant(value_ints=[self.heads_per_group]),
                op.Shape(B_exp, start=4, end=5),
                axis=0,
            ),
        )
        B_heads = op.Reshape(
            B_expand, op.Constant(value_ints=[0, 0, H, N]),
        )

        C_exp = op.Unsqueeze(C_grouped, [3])
        C_expand = op.Expand(
            C_exp,
            op.Concat(
                op.Shape(C_exp, start=0, end=3),
                op.Constant(value_ints=[self.heads_per_group]),
                op.Shape(C_exp, start=4, end=5),
                axis=0,
            ),
        )
        C_heads = op.Reshape(
            C_expand, op.Constant(value_ints=[0, 0, H, N]),
        )

        # Step 5: Pad seq_len to a multiple of chunk_size
        # pad_size = (CS - T % CS) % CS
        CS_c = op.Constant(value_int=self.chunk_size)
        T_val = op.Shape(x_4d, start=1, end=2)  # [T]
        T_scalar = op.Squeeze(T_val, [0])
        pad_size = op.Mod(
            op.Sub(CS_c, op.Mod(T_scalar, CS_c)), CS_c,
        )
        # Pad along dim=1: pads = [0, 0, 0, pad_size, 0...0]
        # For 4D: (B, T, H, D) → pad format [d0b,d1b,d2b,d3b,
        #                                     d0e,d1e,d2e,d3e]
        pad_size_1d = op.Reshape(pad_size, op.Constant(value_ints=[1]))
        zero_1d = op.Constant(value_ints=[0])
        pads_4d = op.Concat(
            zero_1d, zero_1d, zero_1d, zero_1d,  # begins
            zero_1d, pad_size_1d, zero_1d, zero_1d,  # ends
            axis=0,
        )
        pads_3d = op.Concat(
            zero_1d, zero_1d, zero_1d,
            zero_1d, pad_size_1d, zero_1d,
            axis=0,
        )
        x_4d = op.Pad(x_4d, pads_4d, op.Constant(value_float=0.0))
        dt = op.Pad(dt, pads_3d, op.Constant(value_float=0.0))
        B_heads = op.Pad(B_heads, pads_4d, op.Constant(value_float=0.0))
        C_heads = op.Pad(C_heads, pads_4d, op.Constant(value_float=0.0))

        # Step 6: Chunked SSD
        y, new_ssm_state = self._chunked_ssd(
            op, x_4d, dt, B_heads, C_heads, ssm_state,
        )

        # Trim padding: (B, T_padded, H, D) → (B, T, H, D)
        y = op.Slice(
            y, starts=[0], ends=T_val, axes=[1],
        )

        # Step 7: GatedRMSNorm (expects 2D — flatten B*T)
        y_flat = op.Reshape(
            y, op.Constant(value_ints=[0, 0, self.d_inner]),
        )
        y_2d = op.Reshape(
            y_flat, op.Constant(value_ints=[-1, self.d_inner]),
        )
        gate_2d = op.Reshape(
            gate, op.Constant(value_ints=[-1, self.d_inner]),
        )
        y_normed = self.norm(op, y_2d, gate_2d)

        # Reshape back to (B, T, d_inner)
        bt_shape = op.Shape(hidden_states, start=0, end=2)
        y_shape = op.Concat(
            bt_shape,
            op.Constant(value_ints=[self.d_inner]),
            axis=0,
        )
        y_3d = op.Reshape(y_normed, y_shape)

        # Step 7: Output projection — cast back to input dtype
        y_proj = op.CastLike(y_3d, hidden_states)
        output = self.out_proj(op, y_proj)

        return output, new_conv_state, new_ssm_state

    # -----------------------------------------------------------------
    # Single-token fallback (no chunking, seq_len must be 1)
    # -----------------------------------------------------------------

    def _forward_single_token(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token forward pass (no Scan, seq_len must be 1).

        Uses the Mamba2Scan recurrence directly.
        Useful for debugging numerical issues.
        """
        # Step 1: Input projection -> gate, xBC, dt
        projected = self.in_proj(op, hidden_states)
        gate, x_bc, dt = op.Split(
            projected,
            [self.d_inner, self.conv_dim, self.num_heads],
            axis=-1,
            _outputs=3,
        )

        # Step 2: Causal Conv1D with state update
        x_bc_t = op.Transpose(x_bc, perm=[0, 2, 1])
        conv_input = op.Concat(conv_state, x_bc_t, axis=2)
        new_conv_state = op.Slice(
            conv_input, starts=[1], ends=[INT64_MAX], axes=[2],
        )
        conv_out = self.conv1d(op, conv_input)

        # Step 3: SiLU activation
        conv_out = op.Mul(conv_out, op.Sigmoid(conv_out))
        x_bc_activated = op.Transpose(conv_out, perm=[0, 2, 1])

        # Step 4: Split xBC -> hidden, B, C
        gs = self.n_groups * self.d_state
        hidden_x, b_mat, c_mat = op.Split(
            x_bc_activated,
            [self.d_inner, gs, gs],
            axis=-1,
            _outputs=3,
        )

        # Squeeze seq dim for SSM: (B, 1, D) → (B, D)
        hidden_flat = op.Squeeze(hidden_x, [1])
        dt_flat = op.Squeeze(dt, [1])
        b_flat = op.Squeeze(b_mat, [1])
        c_flat = op.Squeeze(c_mat, [1])

        # Step 5: Multi-head selective scan
        y, new_ssm_state = self.ssm(
            op, hidden_flat, dt_flat, b_flat, c_flat, ssm_state,
        )

        # Step 6: Gated RMSNorm
        gate_flat = op.Squeeze(gate, [1])
        y_normed = self.norm(op, y, gate_flat)

        # Restore seq dim: (B, d_inner) → (B, 1, d_inner)
        y_3d = op.Unsqueeze(y_normed, [1])

        # Step 7: Output projection
        output = self.out_proj(op, y_3d)

        return output, new_conv_state, new_ssm_state

    # -----------------------------------------------------------------
    # ONNX Scan (token-by-token multi-token path)
    # -----------------------------------------------------------------

    @staticmethod
    def _realize_submodule(
        parent_builder: builder.GraphBuilder,
        submodule: nn.Module,
    ) -> None:
        """Register a submodule's parameters as parent-graph initializers.

        When Scan body graphs reference ``nn.Parameter`` objects from
        submodules, those parameters must be realized (name-qualified and
        registered) in the parent graph so they are visible as implicit
        inputs to the body subgraph.
        """
        name = submodule._name or ""
        parent_builder.push_module(name, type(submodule).__qualname__)
        for param in submodule._parameters.values():
            param._realize(parent_builder)
        for child in submodule._modules.values():
            Mamba2Block._realize_submodule(parent_builder, child)
        parent_builder.pop_module()

    def _scan_body(
        self,
        op: builder.OpBuilder,
        conv_state_in: ir.Value,
        ssm_state_in: ir.Value,
        xbc_in: ir.Value,
        dt_in: ir.Value,
    ):
        """Scan body: per-token conv state update + SSM step.

        Carry states (updated each iteration):
            conv_state: (B, conv_dim, conv_kernel-1)
            ssm_state:  (B, num_heads, d_head, d_state)

        Scan inputs (per-token, sliced along axis 1):
            xbc_in: (B, conv_dim) -- projected xBC for this token
            dt_in:  (B, num_heads) -- time step for this token

        Scan outputs (per-token, stacked along axis 1):
            y_t: (B, num_heads * d_head) -- SSM output before norm
        """
        # --- Conv state update (shift register) ---
        xbc_3d = op.Unsqueeze(xbc_in, [2])
        conv_cat = op.Concat(conv_state_in, xbc_3d, axis=2)
        new_conv_state = op.Slice(
            conv_cat, starts=[1], ends=[INT64_MAX], axes=[2],
        )

        # --- Apply conv1d (params already realized as implicit inputs) ---
        conv_out = self.conv1d(op, conv_cat)

        # SiLU activation, then squeeze: (B, conv_dim, 1) -> (B, conv_dim)
        conv_out = op.Mul(conv_out, op.Sigmoid(conv_out))
        x_bc_act = op.Squeeze(conv_out, [2])

        # --- Split xBC -> hidden_x, B, C ---
        gs = self.n_groups * self.d_state
        hidden_x, b_mat, c_mat = op.Split(
            x_bc_act,
            [self.d_inner, gs, gs],
            axis=-1,
            _outputs=3,
        )

        # --- SSM step (params already realized as implicit inputs) ---
        y_flat, new_ssm = self.ssm(
            op, hidden_x, dt_in, b_mat, c_mat, ssm_state_in,
        )

        return new_conv_state, new_ssm, y_flat

    def _forward_scan(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Multi-token forward via ONNX Scan (token-by-token iteration).

        Uses an ONNX Scan op to iterate over the sequence, carrying conv
        and SSM states across tokens.  Supports arbitrary seq_len but is
        sequential (no intra-sequence parallelism).
        """
        # Realize conv1d and ssm parameters in the parent graph so they
        # are visible as implicit inputs to the Scan body.
        parent_builder = op.builder
        self._realize_submodule(parent_builder, self.conv1d)
        self._realize_submodule(parent_builder, self.ssm)

        # Step 1: Batch-project all tokens at once
        projected = self.in_proj(op, hidden_states)  # (B, T, proj_size)
        gate, x_bc, dt = op.Split(
            projected,
            [self.d_inner, self.conv_dim, self.num_heads],
            axis=-1,
            _outputs=3,
        )

        # Step 2: Scan over axis 1 (time)
        body = parent_builder.subgraph(
            self._scan_body,
            inputs={
                "conv_state": FLOAT[...],
                "ssm_state": FLOAT[...],
                "xbc_t": FLOAT[...],
                "dt_t": FLOAT[...],
            },
            outputs={
                "new_conv_state": FLOAT[...],
                "new_ssm_state": FLOAT[...],
                "y_t": FLOAT[...],
            },
            name="mamba2_recurrence",
        )
        new_conv_state, new_ssm_state, y_all = op.Scan(
            conv_state,
            ssm_state,
            x_bc,
            dt,
            body=body,
            num_scan_inputs=2,
            scan_input_axes=[1, 1],
            scan_output_axes=[1],
            _outputs=3,
        )
        # y_all: (B, T, d_inner)

        # Step 3: GatedRMSNorm (expects 2D -- flatten B*T)
        y_flat = op.Reshape(y_all, [-1, self.d_inner])
        gate_flat = op.Reshape(gate, [-1, self.d_inner])
        y_normed = self.norm(op, y_flat, gate_flat)

        # Reshape back to (B, T, d_inner)
        bt_shape = op.Shape(hidden_states, start=0, end=2)
        y_shape = op.Concat(bt_shape, [self.d_inner], axis=0)
        y_3d = op.Reshape(y_normed, y_shape)

        # Step 4: Batch output projection
        output = self.out_proj(op, y_3d)

        return output, new_conv_state, new_ssm_state

    # -----------------------------------------------------------------
    # Public forward: dispatches based on flag
    # -----------------------------------------------------------------

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Forward pass supporting arbitrary sequence length.

        Dispatches to one of three implementations based on
        ``flags.mamba_scan``:

        - ``"chunked_ssd"`` (default): chunked SSD algorithm.
        - ``"scan"``: ONNX Scan op (token-by-token).
        - ``"single"``: single-token path (seq_len must be 1).

        Args:
            op: ONNX op builder.
            hidden_states: (batch, seq_len, d_model)
            conv_state: (batch, conv_dim, conv_kernel-1)
            ssm_state: (batch, num_heads, d_head, d_state)

        Returns:
            output: (batch, seq_len, d_model)
            new_conv_state: (batch, conv_dim, conv_kernel-1)
            new_ssm_state: (batch, num_heads, d_head, d_state)
        """
        mode = flags.mamba_scan
        if mode == "single":
            return self._forward_single_token(
                op, hidden_states, conv_state, ssm_state,
            )
        if mode == "scan":
            return self._forward_scan(
                op, hidden_states, conv_state, ssm_state,
            )
        return self._forward_chunked_ssd(
            op, hidden_states, conv_state, ssm_state,
        )

