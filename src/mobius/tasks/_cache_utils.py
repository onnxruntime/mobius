# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""KV cache and linear attention utility functions for task graph construction.

Provides helpers for wiring KV cache state tensors (inputs/outputs) into
ONNX task graphs, and for registering hybrid (DeltaNet) linear-attention
function bodies.  Only cache-related utilities live here; higher-level
multi-component graph builders live in :mod:`mobius.tasks._base`.
"""

from __future__ import annotations

from typing import NamedTuple

import onnx_ir as ir
from onnxscript import GraphBuilder

from mobius._configs import BaseModelConfig

# Cache state pair: (key, value) or (conv_state, ssm_state) for stateful
# layers; (rec_state,) for lightning/conv attention (single state);
# or (None, None) for stateless layers (e.g. MLP-only).
StatePair = tuple[ir.Value, ...] | tuple[None, None]


class LinearAttentionDims(NamedTuple):
    """Dimension sizes for linear attention (DeltaNet) layers."""

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    key_dim: int  # = head_k_dim * num_k_heads
    value_dim: int  # = head_v_dim * num_v_heads
    conv_dim: int  # = key_dim * 2 + value_dim
    conv_kernel: int


def linear_attention_dims(config: BaseModelConfig) -> LinearAttentionDims:
    """Compute dimension sizes for linear attention from config.

    Raises ``TypeError`` if any required config field is ``None``.
    """
    num_k_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    head_k_dim = config.linear_key_head_dim
    head_v_dim = config.linear_value_head_dim
    conv_kernel = config.linear_conv_kernel_dim
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim
    return LinearAttentionDims(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        conv_dim=conv_dim,
        conv_kernel=conv_kernel,
    )


def _make_kv_cache_inputs(
    builder: GraphBuilder,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: ir.DataType,
    batch: ir.SymbolicDim,
    past_seq_len: ir.SymbolicDim,
    *,
    prefix: str = "past_key_values",
    key_head_dim: int | None = None,
    value_head_dim: int | None = None,
) -> list[tuple[ir.Value, ir.Value]]:
    """Create KV cache input values for ``num_layers`` layers.

    Uses ``builder.input()`` to create and register graph inputs directly.

    Args:
        builder: The graph builder to register inputs on.
        key_head_dim: Head dim for keys. Defaults to ``head_dim``.
            For MLA attention, this is ``qk_nope_head_dim + qk_rope_head_dim``.
        value_head_dim: Head dim for values. Defaults to ``head_dim``.
            For MLA attention, this is ``v_head_dim``.

    Returns:
        A list of ``(key, value)`` tuples for passing to the module.
    """
    k_dim = key_head_dim if key_head_dim is not None else head_dim
    v_dim = value_head_dim if value_head_dim is not None else head_dim
    pairs: list[tuple[ir.Value, ir.Value]] = []
    for i in range(num_layers):
        past_key = builder.input(
            f"{prefix}.{i}.key",
            dtype=dtype,
            shape=[batch, num_kv_heads, past_seq_len, k_dim],
        )
        past_value = builder.input(
            f"{prefix}.{i}.value",
            dtype=dtype,
            shape=[batch, num_kv_heads, past_seq_len, v_dim],
        )
        pairs.append((past_key, past_value))
    return pairs


def _register_kv_cache_outputs(
    builder: GraphBuilder,
    present_key_values: list[tuple[ir.Value, ir.Value]],
    *,
    prefix: str = "present",
) -> None:
    """Name and register KV cache outputs on the graph.

    Output shapes and dtypes are inferred by the shape inference pass
    that runs during model optimization.
    """
    for i, (present_key, present_value) in enumerate(present_key_values):
        builder.add_output(present_key, f"{prefix}.{i}.key")
        builder.add_output(present_value, f"{prefix}.{i}.value")


def _make_hybrid_cache_inputs(
    builder: GraphBuilder,
    config: BaseModelConfig,
    dtype: ir.DataType,
    batch: ir.SymbolicDim,
    past_seq_len: ir.SymbolicDim,
    *,
    prefix: str = "past_key_values",
) -> list[StatePair]:
    """Create cache inputs for hybrid models with mixed layer types.

    Uses ``builder.input()`` to create and register graph inputs directly.

    Supported layer types:
        ``"full_attention"`` — standard KV cache (key + value).
        ``"lightning_attention"`` — single recurrent state only; no conv_state.
        ``"conv"`` — ShortConv conv_state only; no SSM state.
        ``"linear_attention"`` (DeltaNet) — conv_state + recurrent_state.
        ``"mamba"`` / ``"mamba2"`` — conv_state + ssm_state.
        ``"mlp"`` — stateless, produces ``(None, None)`` pair.

    Returns:
        A list of state pairs, one per layer, with ``(None, None)``
        for stateless MLP layers.
    """
    layer_types = config.layer_types or []
    pairs: list[StatePair] = []

    # DeltaNet dimensions from config (computed once via shared helper)
    has_linear = "linear_attention" in layer_types
    if has_linear:
        dims = linear_attention_dims(config)

    # Mamba SSM dimensions from config (Jamba-style)
    mamba_expand = getattr(config, "mamba_expand", 2)
    mamba_d_inner = config.hidden_size * mamba_expand
    mamba_d_conv = getattr(config, "mamba_d_conv", 4)
    mamba_d_state = getattr(config, "mamba_d_state", 16)

    # Mamba2/SSD dimensions from config (Bamba-style).
    # Defaults are 0 so a missing field produces a clear shape error
    # rather than silently using model-specific values.
    mamba2_n_heads = getattr(config, "mamba_n_heads", 0)
    mamba2_d_head = getattr(config, "mamba_d_head", 0)
    mamba2_d_state = getattr(config, "mamba_d_state", 0)
    mamba2_n_groups = getattr(config, "mamba_n_groups", 1)
    # Prefer n_heads * d_head (NemotronH); fall back to hidden * expand (Bamba)
    mamba2_d_inner = (
        mamba2_n_heads * mamba2_d_head
        if mamba2_n_heads and mamba2_d_head
        else config.hidden_size * mamba_expand
    )
    mamba2_conv_dim = mamba2_d_inner + 2 * mamba2_n_groups * mamba2_d_state

    for i in range(config.num_hidden_layers):
        ltype = layer_types[i] if i < len(layer_types) else "full_attention"

        if ltype == "lightning_attention":
            # Lightning Attention: single recurrent state only (no conv_state)
            # State: (B, num_heads, head_dim, head_dim) — square matrix accumulator
            rec_state = builder.input(
                f"{prefix}.{i}.recurrent_state",
                dtype=dtype,
                shape=[batch, config.num_attention_heads, config.head_dim, config.head_dim],
            )
            pairs.append((rec_state,))  # 1-tuple: lightning has no conv_state
        elif ltype == "linear_attention":
            conv_state = builder.input(
                f"{prefix}.{i}.conv_state",
                dtype=dtype,
                shape=[batch, dims.conv_dim, dims.conv_kernel - 1],
            )
            rec_state = builder.input(
                f"{prefix}.{i}.recurrent_state",
                dtype=dtype,
                shape=[batch, dims.num_v_heads, dims.head_k_dim, dims.head_v_dim],
            )
            pairs.append((conv_state, rec_state))
        elif ltype == "conv":
            # ShortConv layers: conv_state only (no SSM state)
            # State: (batch, hidden_size, short_conv_kernel - 1)
            short_conv_kernel = getattr(config, "short_conv_kernel", 3)
            conv_state = builder.input(
                f"{prefix}.{i}.conv_state",
                dtype=dtype,
                shape=[batch, config.hidden_size, short_conv_kernel - 1],
            )
            pairs.append((conv_state,))  # 1-tuple: conv has no second state
        elif ltype in ("mlp", "moe"):
            # MLP and MoE layers are stateless — no cache inputs needed
            pairs.append((None, None))
        elif ltype == "mamba":
            conv_state = builder.input(
                f"{prefix}.{i}.conv_state",
                dtype=dtype,
                shape=[batch, mamba_d_inner, mamba_d_conv - 1],
            )
            ssm_state = builder.input(
                f"{prefix}.{i}.ssm_state",
                dtype=dtype,
                shape=[batch, mamba_d_inner, mamba_d_state],
            )
            pairs.append((conv_state, ssm_state))
        elif ltype == "mamba2":
            conv_state = builder.input(
                f"{prefix}.{i}.conv_state",
                dtype=dtype,
                shape=[batch, mamba2_conv_dim, mamba_d_conv - 1],
            )
            ssm_state = builder.input(
                f"{prefix}.{i}.ssm_state",
                dtype=dtype,
                shape=[batch, mamba2_n_heads, mamba2_d_state, mamba2_d_head],
            )
            pairs.append((conv_state, ssm_state))
        else:
            past_key = builder.input(
                f"{prefix}.{i}.key",
                dtype=dtype,
                shape=[batch, config.num_key_value_heads, past_seq_len, config.head_dim],
            )
            past_value = builder.input(
                f"{prefix}.{i}.value",
                dtype=dtype,
                shape=[batch, config.num_key_value_heads, past_seq_len, config.head_dim],
            )
            pairs.append((past_key, past_value))

    return pairs


def _register_hybrid_cache_outputs(
    builder: GraphBuilder,
    present_key_values: list[tuple[ir.Value, ...]],
    layer_types: list[str],
    *,
    prefix: str = "present",
) -> None:
    """Name and register hybrid cache outputs on the graph.

    Uses ``.key``/``.value`` for full attention layers,
    ``.recurrent_state`` for lightning attention layers (1-tuple),
    ``.conv_state`` for ShortConv layers (1-tuple),
    ``.conv_state``/``.recurrent_state`` for linear attention layers,
    and ``.conv_state``/``.ssm_state`` for mamba/mamba2 layers.

    Output shapes and dtypes are inferred by the shape inference pass
    that runs during model optimization.
    """
    for i, states in enumerate(present_key_values):
        ltype = layer_types[i] if i < len(layer_types) else "full_attention"
        if ltype == "mlp" or ltype == "moe":
            continue  # MLP and MoE layers produce no cache state
        if ltype == "lightning_attention":
            # Single recurrent state only (no conv_state for lightning)
            (state_a,) = states
            builder.add_output(state_a, f"{prefix}.{i}.recurrent_state")
        elif ltype == "conv":
            # ShortConv: single conv_state only
            (state_a,) = states
            builder.add_output(state_a, f"{prefix}.{i}.conv_state")
        else:
            state_a, state_b = states
            if ltype == "linear_attention":
                builder.add_output(state_a, f"{prefix}.{i}.conv_state")
                builder.add_output(state_b, f"{prefix}.{i}.recurrent_state")
            elif ltype in ("mamba", "mamba2"):
                builder.add_output(state_a, f"{prefix}.{i}.conv_state")
                builder.add_output(state_b, f"{prefix}.{i}.ssm_state")
            else:
                builder.add_output(state_a, f"{prefix}.{i}.key")
                builder.add_output(state_b, f"{prefix}.{i}.value")
