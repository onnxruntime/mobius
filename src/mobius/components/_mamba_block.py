# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mamba block components: Conv1D → SSM → gated output projection.

This module provides:

- **MambaBlock** (Mamba1): standard Mamba layer for Mamba, Jamba,
  FalconMamba, etc.
- **Mamba2BlockBase**: shared ``__init__`` and helpers for all Mamba2
  multi-token modes.
- **Mamba2BlockSingle**: single-token path (seq_len must be 1).
- **Mamba2Block**: factory function that instantiates the correct
  subclass based on ``flags.mamba_scan``.

The full set of Mamba2 subclasses is:

- ``Mamba2BlockSingle``  — this file
- ``Mamba2BlockScan``    — ``_mamba_block_scan.py``
- ``Mamba2BlockChunkedSSD`` — ``_mamba_block_chunked.py``

HuggingFace reference: ``MambaMixer``, ``BambaMixer``,
``NemotronHMamba2Mixer``.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

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
            conv_input,
            starts=[1],
            ends=[INT64_MAX],
            axes=[2],
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


# =====================================================================
# Mamba2 base class and subclasses
# =====================================================================


class Mamba2BlockBase(nn.Module):
    """Base class for all Mamba2 block variants.

    Shared ``__init__`` defines the parameters (in_proj, conv1d, ssm,
    norm, out_proj) so that all subclasses produce identical ONNX
    weight paths.  Subclasses override ``forward()`` with the specific
    multi-token algorithm.

    Key differences from MambaBlock (Mamba1):
    - in_proj outputs [gate, xBC, dt] instead of [x, z]
    - Conv1D on wider xBC (conv_dim channels)
    - Multi-head SSM with grouped B/C
    - GatedRMSNorm instead of SiLU gating
    - dt direct from in_proj (no rank reduction), just bias
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
            self.conv_dim,
            conv_kernel,
            bias=conv_bias,
        )
        # SSM parameters live under self.ssm so that ONNX weight paths
        # stay as ``mamba.ssm.{A_log,D,dt_bias}`` — compatible with all
        # existing preprocess_weights rename rules.
        self.ssm = Mamba2Scan(
            num_heads,
            d_head,
            d_state,
            n_groups,
            time_step_min=time_step_min,
        )
        self.norm = GatedRMSNorm(
            d_inner,
            eps=eps,
            group_size=norm_group_size,
        )
        self.out_proj = Linear(d_inner, d_model, bias=proj_bias)

    @staticmethod
    def _realize_submodule(
        parent_builder: builder.GraphBuilder,
        submodule: nn.Module,
    ) -> None:
        """Register a submodule's parameters as parent-graph initializers.

        When a forward path references ``self.ssm.A_log`` etc. directly
        (without calling ``self.ssm(...)``), or when Scan body graphs
        need parameters as implicit inputs, this helper pushes the
        naming context and calls ``_realize()`` on every parameter so
        they appear as graph initializers.
        """
        name = submodule._name or ""
        parent_builder.push_module(name, type(submodule).__qualname__)
        for param in submodule._parameters.values():
            param._realize(parent_builder)
        for child in submodule._modules.values():
            Mamba2BlockBase._realize_submodule(parent_builder, child)
        parent_builder.pop_module()


class Mamba2BlockSingle(Mamba2BlockBase):
    """Mamba2 block: single-token path (seq_len must be 1).

    Uses the Mamba2Scan recurrence directly with no chunking or Scan.
    Useful for debugging numerical issues.
    """

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token forward pass (seq_len must be 1).

        Args:
            op: ONNX op builder.
            hidden_states: (batch, 1, d_model)
            conv_state: (batch, conv_dim, conv_kernel-1)
            ssm_state: (batch, num_heads, d_head, d_state)

        Returns:
            output: (batch, 1, d_model)
            new_conv_state: (batch, conv_dim, conv_kernel-1)
            new_ssm_state: (batch, num_heads, d_head, d_state)
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
            conv_input,
            starts=[1],
            ends=[INT64_MAX],
            axes=[2],
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
            op,
            hidden_flat,
            dt_flat,
            b_flat,
            c_flat,
            ssm_state,
        )

        # Step 6: Gated RMSNorm
        gate_flat = op.Squeeze(gate, [1])
        y_normed = self.norm(op, y, gate_flat)

        # Restore seq dim: (B, d_inner) → (B, 1, d_inner)
        y_3d = op.Unsqueeze(y_normed, [1])

        # Step 7: Output projection
        output = self.out_proj(op, y_3d)

        return output, new_conv_state, new_ssm_state


# =====================================================================
# Factory function
# =====================================================================


def Mamba2Block(*args, **kwargs) -> Mamba2BlockBase:
    """Instantiate the Mamba2 block variant for the current flag.

    Reads ``flags.mamba_scan`` at construction time and returns the
    matching subclass.  Callers use this exactly like the old
    ``Mamba2Block`` class — no API change.

    Available modes:

    - ``"single"`` → ``Mamba2BlockSingle`` (this file)
    - ``"scan"`` → ``Mamba2BlockScan`` (``_mamba_block_scan.py``)
    - ``"chunked_ssd"`` → ``Mamba2BlockChunkedSSD``
      (``_mamba_block_chunked.py``)
    """
    mode = flags.mamba_scan
    if mode == "single":
        return Mamba2BlockSingle(*args, **kwargs)
    if mode == "scan":
        from mobius.components._mamba_block_scan import Mamba2BlockScan

        return Mamba2BlockScan(*args, **kwargs)
    if mode == "chunked_ssd":
        from mobius.components._mamba_block_chunked import (
            Mamba2BlockChunkedSSD,
        )

        return Mamba2BlockChunkedSSD(*args, **kwargs)
    msg = f"Unknown mamba_scan mode {mode!r}. Expected 'single', 'scan', or 'chunked_ssd'."
    raise ValueError(msg)
