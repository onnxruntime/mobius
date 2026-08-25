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

_FUNCTIONS_DOMAIN = "com.microsoft"

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
    missing = [
        name
        for name, value in (
            ("linear_num_key_heads", num_k_heads),
            ("linear_num_value_heads", num_v_heads),
            ("linear_key_head_dim", head_k_dim),
            ("linear_value_head_dim", head_v_dim),
            ("linear_conv_kernel_dim", conv_kernel),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Linear-attention cache inputs requested but the following config "
            f"fields are unset: {', '.join(missing)}. This usually means the "
            "architecture's layer_types were mis-detected as 'linear_attention' "
            "(e.g. a hybrid model built without trust_remote_code)."
        )
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
    batch: ir.SymbolicDim | str | None = None,
    num_kv_heads: int | None = None,
    key_head_dim: int | None = None,
    value_head_dim: int | None = None,
    total_seq_len: ir.SymbolicDim | str | int | None = None,
    dtype: ir.DataType | None = None,
) -> None:
    """Name and register KV cache outputs on the graph.

    When every present-shape parameter (``batch``, ``num_kv_heads``,
    ``key_head_dim``, ``value_head_dim``, ``total_seq_len``, ``dtype``) is
    supplied, each ``present.{i}.{key,value}`` output is stamped with an
    explicit ``[batch, num_kv_heads, total_seq_len, head_dim]`` type, symmetric
    to the ``past_key_values`` inputs created by :func:`_make_kv_cache_inputs`.

    This explicit stamp is required for ``com.microsoft::GroupQueryAttention``
    exports: that contrib op's shape inference mis-derives the present
    ``head_dim`` (it divides the *packed* QKV query hidden by
    ``num_heads + 2 * kv_num_heads`` and lands on the wrong value), so the
    ``present.*`` outputs would otherwise declare the wrong ``head_dim`` even
    though the kernel produces correct data at runtime. The mismatch makes ORT
    log a shape-merge warning and breaks present->past chaining in consumers
    such as onnxruntime-genai that trust the declared shapes. The plain ONNX
    ``Attention`` op infers the present shape correctly, so the stamp is a
    no-op there.

    When the parameters are omitted, output shapes/dtypes are left to the
    shape inference pass that runs during model optimization.

    The present-shape parameters are all-or-nothing by design: pass every one
    to stamp the explicit type, or none to opt out and infer. A *partial* set
    is always a wiring slip (a caller wired some dims but dropped others) and is
    never legitimate, so it is rejected fail-closed: a partial set raises
    :class:`ValueError` naming the provided and missing parameters, rather than
    silently falling back to the known-wrong inference path.
    """
    params = {
        "batch": batch,
        "num_kv_heads": num_kv_heads,
        "key_head_dim": key_head_dim,
        "value_head_dim": value_head_dim,
        "total_seq_len": total_seq_len,
        "dtype": dtype,
    }
    provided = [name for name, value in params.items() if value is not None]
    stamp = len(provided) == len(params)
    if provided and not stamp:
        missing = [name for name in params if params[name] is None]
        raise ValueError(
            f"_register_kv_cache_outputs received a partial set of present-shape "
            f"parameters (provided {provided}, missing {missing}); these are "
            f"all-or-nothing. Pass all six to stamp explicit present.* types "
            f"(required for correct GroupQueryAttention head_dim), or none to opt "
            f"out and infer."
        )
    for i, (present_key, present_value) in enumerate(present_key_values):
        if stamp:
            present_key.shape = ir.Shape([batch, num_kv_heads, total_seq_len, key_head_dim])
            present_key.type = ir.TensorType(dtype)
            present_value.shape = ir.Shape(
                [batch, num_kv_heads, total_seq_len, value_head_dim]
            )
            present_value.type = ir.TensorType(dtype)
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
    if not layer_types:
        layer_types = ["full_attention"] * config.num_hidden_layers
    elif len(layer_types) != config.num_hidden_layers:
        raise ValueError("Hybrid layer_types must contain exactly num_hidden_layers entries")
    supported_layer_types = {
        "full_attention",
        "lightning_attention",
        "conv",
        "linear_attention",
        "mamba",
        "mamba2",
        "mlp",
        "moe",
    }
    unknown = sorted(set(layer_types) - supported_layer_types)
    if unknown:
        raise ValueError(f"Unsupported hybrid layer type(s): {unknown}")
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
        ltype = layer_types[i]

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
            # Fused causal-conv state: the K-1 preceding input values.
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
        elif ltype == "full_attention":
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
        else:
            raise AssertionError(f"Unhandled hybrid layer type: {ltype}")

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
    if not layer_types:
        layer_types = ["full_attention"] * len(present_key_values)
    if len(layer_types) != len(present_key_values):
        raise ValueError(
            "Hybrid output layer_types must match the number of layer state tuples"
        )
    for i, states in enumerate(present_key_values):
        ltype = layer_types[i]
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
            elif ltype == "full_attention":
                builder.add_output(state_a, f"{prefix}.{i}.key")
                builder.add_output(state_b, f"{prefix}.{i}.value")
            else:
                raise ValueError(f"Unsupported hybrid output layer type: {ltype!r}")


def _register_linear_attention_functions(
    model: ir.Model,
    config: BaseModelConfig,
) -> None:
    """Register CausalConvWithState and LinearAttention functions.

    Registers functions for DeltaNet (``linear_attention`` layers),
    Lightning Attention (``lightning_attention`` layers), and/or
    Mamba2 (``mamba2`` layers) as needed.
    Adds the ``com.microsoft`` opset import to the graph.
    """
    layer_types = getattr(config, "layer_types", None) or []
    has_deltanet = "linear_attention" in layer_types
    has_lightning = "lightning_attention" in layer_types
    has_mamba2 = "mamba2" in layer_types
    has_short_conv = "conv" in layer_types

    if not has_deltanet and not has_lightning and not has_mamba2 and not has_short_conv:
        return

    from mobius.functions import (
        causal_conv_nd_with_state,
        linear_attention,
    )

    if has_deltanet:
        dims = linear_attention_dims(config)
        conv_func = causal_conv_nd_with_state(
            kernel_size=dims.conv_kernel,
            channels=dims.conv_dim,
            ndim=1,
            activation="silu",
        )
        attn_func = linear_attention(
            q_num_heads=dims.num_k_heads,
            kv_num_heads=dims.num_v_heads,
            update_rule="gated_delta",
            scale=1.0 / (dims.head_k_dim**0.5),
            stash_type=config.dtype,
        )
        model.functions[conv_func.identifier()] = conv_func
        model.functions[attn_func.identifier()] = attn_func

    if has_lightning:
        head_dim = config.head_dim
        attn_func_gated = linear_attention(
            q_num_heads=config.num_attention_heads,
            kv_num_heads=config.num_attention_heads,
            update_rule="gated",
            scale=1.0 / (head_dim**0.5),
            stash_type=config.dtype,
        )
        model.functions[attn_func_gated.identifier()] = attn_func_gated

    if has_mamba2:
        mamba2_n_heads = getattr(config, "mamba_n_heads", 0)
        mamba2_d_head = getattr(config, "mamba_d_head", 0)
        mamba2_d_state = getattr(config, "mamba_d_state", 0)
        mamba2_n_groups = getattr(config, "mamba_n_groups", 1)
        mamba2_d_conv = getattr(config, "mamba_d_conv", 4)
        mamba_expand = getattr(config, "mamba_expand", 2)
        mamba2_d_inner = (
            mamba2_n_heads * mamba2_d_head
            if mamba2_n_heads and mamba2_d_head
            else config.hidden_size * mamba_expand
        )
        mamba2_conv_dim = mamba2_d_inner + 2 * mamba2_n_groups * mamba2_d_state
        conv_func = causal_conv_nd_with_state(
            kernel_size=mamba2_d_conv,
            channels=mamba2_conv_dim,
            ndim=1,
            activation="silu",
        )
        attn_func = linear_attention(
            q_num_heads=mamba2_n_heads,
            kv_num_heads=mamba2_n_heads,
            update_rule="gated",
            scale=1.0,
            stash_type=ir.DataType.FLOAT,
        )
        model.functions[conv_func.identifier()] = conv_func
        model.functions[attn_func.identifier()] = attn_func

    if has_short_conv:
        conv_func = causal_conv_nd_with_state(
            kernel_size=getattr(config, "short_conv_kernel", 3),
            channels=config.hidden_size,
            ndim=1,
            activation="none",
        )
        model.functions[conv_func.identifier()] = conv_func

    model.graph.opset_imports[_FUNCTIONS_DOMAIN] = 1


def _register_linear_attention_functions_for_ssm2(
    model: ir.Model,
    config: BaseModelConfig,
) -> None:
    """Register CausalConvWithState and LinearAttention for pure Mamba2 models.

    Unlike :func:`_register_linear_attention_functions` (which inspects
    ``layer_types``), this always registers the Mamba2 function ops.
    Called by :class:`SSM2CausalLMTask` for pure Mamba2 models that don't
    have a ``layer_types`` attribute.
    """
    from mobius._configs import Mamba2Config
    from mobius.functions import (
        causal_conv_nd_with_state,
        linear_attention,
    )

    assert isinstance(config, Mamba2Config)
    n_heads = config.num_heads
    d_state = config.state_size
    n_groups = config.n_groups
    d_inner = config.intermediate_size
    d_conv = config.conv_kernel
    conv_dim = d_inner + 2 * n_groups * d_state

    conv_func = causal_conv_nd_with_state(
        kernel_size=d_conv,
        channels=conv_dim,
        ndim=1,
        activation="silu",
    )
    attn_func = linear_attention(
        q_num_heads=n_heads,
        kv_num_heads=n_heads,
        update_rule="gated",
        scale=1.0,
        stash_type=ir.DataType.FLOAT,
    )
    model.functions[conv_func.identifier()] = conv_func
    model.functions[attn_func.identifier()] = attn_func
    model.graph.opset_imports[_FUNCTIONS_DOMAIN] = 1
