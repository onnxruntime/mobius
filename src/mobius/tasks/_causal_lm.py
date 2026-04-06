# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Causal language model tasks with internal, static, and GQA KV cache."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.components._attention import StaticCacheState
from mobius.components._gqa_attention import GQAContext
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_hybrid_cache_inputs,
    _make_kv_cache_inputs,
    _make_model,
    _register_hybrid_cache_outputs,
    _register_kv_cache_outputs,
    _register_linear_attention_functions,
)


def _find_rotary_emb(module: nn.Module) -> nn.Module | None:
    """Find the rotary embedding module by walking the module tree."""
    for name, child in module.named_modules():
        if name.endswith("rotary_emb") and hasattr(child, "cos_cache"):
            return child
    return None


class CausalLMTask(ModelTask):
    """Causal language model with KV cache for text generation.

    Supports three cache modes:

    **Dynamic cache** (default):
        Standard KV cache with dynamic sequence lengths. Past keys/values
        are concatenated internally by the Attention op.

        Inputs:
            - input_ids: [batch, sequence_len] INT64
            - attention_mask: [batch, total_seq_len] INT64
            - position_ids: [batch, sequence_len] INT64
            - past_key_values.{i}.key: [batch, num_kv_heads, past_seq_len, head_dim]
            - past_key_values.{i}.value: [batch, num_kv_heads, past_seq_len, head_dim]
        Outputs:
            - logits: FLOAT
            - present.{i}.key / present.{i}.value: FLOAT

    **Static cache** (``static_cache=True``):
        Pre-allocated KV cache buffers updated via TensorScatter.  Avoids
        repeated concatenation and produces a simpler graph that is easier
        to optimize.  Requires models using :class:`DecoderLayer` or
        :class:`MoEDecoderLayer`.

        Inputs:
            - input_ids: [batch, seq_len] INT64
            - position_ids: [batch, seq_len] INT64
            - key_cache.{i}: [batch, max_seq_len, kv_hidden] FLOAT per layer
            - value_cache.{i}: [batch, max_seq_len, kv_hidden] FLOAT per layer
            - write_indices: [batch] INT64
            - nonpad_kv_seqlen: [batch] INT64
        Outputs:
            - logits: FLOAT
            - updated_key_cache.{i} / updated_value_cache.{i}: FLOAT

        No ``attention_mask`` input — causal masking uses ``is_causal=1``.

    **GQA mode** (``gqa=True``):
        Uses ``com.microsoft::GroupQueryAttention`` with fused RoPE and
        in-place KV cache support.  Compatible with the onnxruntime-genai
        runtime (``past_present_share_buffer=true``).

        Inputs:
            - input_ids: [batch, sequence_len] INT64
            - attention_mask: [batch, total_seq_len] INT64
            - past_key_values.{i}.key: [batch, num_kv_heads, past_seq_len, head_dim]
            - past_key_values.{i}.value: [batch, num_kv_heads, past_seq_len, head_dim]
        Outputs:
            - logits: FLOAT
            - present.{i}.key / present.{i}.value: FLOAT

        No ``position_ids`` input — RoPE is fused inside GroupQueryAttention
        using cos_cache/sin_cache initializers.

    The module's ``forward()`` must accept
    ``(op, input_ids, attention_mask, position_ids, past_key_values)``
    and return ``(logits, list_of_(key, value)_tuples)``.  In static cache
    mode, ``attention_mask`` will be ``None`` and ``past_key_values``
    entries will be :class:`StaticCacheState` tuples.  In GQA mode,
    ``position_ids`` will be ``None`` and a ``gqa_context`` kwarg is
    passed.

    Args:
        static_cache: If ``True``, use pre-allocated static KV cache
            buffers instead of dynamic concatenation.
        max_seq_len: Maximum sequence length for static cache buffers.
            Only used when ``static_cache=True``.  Defaults to
            ``config.max_position_embeddings``.
        gqa: If ``True``, use ``com.microsoft::GroupQueryAttention``
            with fused RoPE and in-place KV cache support.
    """

    def __init__(
        self,
        *,
        static_cache: bool = False,
        max_seq_len: int | None = None,
        gqa: bool = False,
    ):
        self._static_cache = static_cache
        self._max_seq_len = max_seq_len
        self._gqa = gqa

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        if self._gqa:
            return self._build_gqa(module, config)

        static = self._static_cache

        # --- Static-cache pre-validation ---
        if static:
            max_seq_len = self._max_seq_len
            if max_seq_len is None:
                max_seq_len = getattr(config, "max_position_embeddings", None)
            if max_seq_len is None or max_seq_len <= 0:
                raise ValueError(
                    "max_seq_len must be a positive integer. Either pass it "
                    "to CausalLMTask(max_seq_len=...) or ensure "
                    "config.max_position_embeddings is set."
                )
            _validate_static_cache_support(module)

        # --- Symbolic dims ---
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        # --- Inputs common to both modes ---
        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        # --- Cache setup (static vs dynamic) ---
        if static:
            attention_mask = None
            graph_inputs = [input_ids, position_ids]
            cache_inputs, past_key_values = _make_static_cache_inputs(
                config.num_hidden_layers,
                config.num_key_value_heads,
                config.head_dim,
                config.dtype,
                batch,
                max_seq_len,
            )
        else:
            past_seq_len = ir.SymbolicDim("past_sequence_len")
            attention_mask = ir.Value(
                name="attention_mask",
                shape=ir.Shape([batch, "past_seq_len + seq_len"]),
                type=ir.TensorType(ir.DataType.INT64),
            )
            graph_inputs = [input_ids, attention_mask, position_ids]

            # MLA attention: K/V heads equal q heads (no GQA reduction in
            # latent space).  The ONNX Attention op is called with
            # kv_num_heads=num_attention_heads, so the KV cache must use
            # num_attention_heads (not num_key_value_heads).
            use_mla = (
                config.qk_nope_head_dim is not None and config.qk_nope_head_dim > 0
            ) or (config.qk_rope_head_dim is not None and config.qk_rope_head_dim > 0)
            num_kv_cache_heads = (
                config.num_attention_heads if use_mla else config.num_key_value_heads
            )

            cache_inputs, past_key_values = _make_kv_cache_inputs(
                config.num_hidden_layers,
                num_kv_cache_heads,
                config.head_dim,
                config.dtype,
                batch,
                past_seq_len,
                key_head_dim=((config.qk_nope_head_dim or 0) + (config.qk_rope_head_dim or 0))
                or None,
                value_head_dim=config.v_head_dim or None,
            )

        graph_inputs.extend(cache_inputs)

        # --- Build graph, invoke module, collect outputs ---
        graph, builder = _make_graph(graph_inputs)
        op = builder.op

        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)

        # --- Output registration (static vs dynamic) ---
        if static:
            _register_static_cache_outputs(
                graph,
                present_key_values,
            )
        else:
            _register_kv_cache_outputs(
                graph,
                present_key_values,
            )

        return ModelPackage({"model": _make_model(graph)}, config=config)

    def _build_gqa(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        """Build a model using com.microsoft::GroupQueryAttention.

        Creates the graph with:
        - No position_ids input (RoPE is fused inside GQA)
        - seqlens_k and total_seq_len computed from attention_mask
        - cos_cache and sin_cache as graph initializers
        - Standard 4D KV cache I/O (same naming as dynamic cache)
        """
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        # --- Graph inputs ---
        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        # No position_ids — GQA handles RoPE internally

        graph_inputs = [input_ids, attention_mask]

        # --- KV cache inputs (same 4D shape as dynamic cache) ---
        cache_inputs, past_key_values = _make_kv_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(cache_inputs)

        # --- Build graph ---
        graph, builder = _make_graph(graph_inputs)
        op = builder.op

        # --- Compute seqlens_k and total_seq_len from attention_mask ---
        # seqlens_k = Cast(ReduceSum(attention_mask, axis=1) - 1, INT32)
        axis = op.Constant(value_ints=[1])
        reduce_sum = op.ReduceSum(attention_mask, axis)
        one = op.Constant(value_ints=[1])
        seqlens_k = op.Cast(op.Sub(reduce_sum, one), to=ir.DataType.INT32)

        # total_seq_len = Cast(Gather(Shape(attention_mask), 1), INT32)
        mask_shape = op.Shape(attention_mask)
        idx_1 = op.Constant(value_int=1)
        total_seq_len = op.Cast(op.Gather(mask_shape, idx_1), to=ir.DataType.INT32)

        # --- Get cos/sin cache from the model's rotary embedding ---
        # The model must have a rotary_emb with cos_cache/sin_cache parameters.
        rotary_emb = _find_rotary_emb(module)
        if rotary_emb is None:
            raise ValueError(
                "GQA mode requires the model to have a rotary embedding "
                "module with cos_cache and sin_cache parameters. "
                "Ensure the model uses initialize_rope()."
            )
        # Register cos/sin caches as graph initializers so they get
        # serialized.  The rotary_emb module isn't called in GQA mode
        # (GQA handles RoPE internally), so its parameters won't be
        # auto-registered by the builder's module call machinery.
        cos_cache = builder.initializer(rotary_emb.cos_cache.const_value, name="cos_cache")
        sin_cache = builder.initializer(rotary_emb.sin_cache.const_value, name="sin_cache")

        gqa_context = GQAContext(
            seqlens_k=seqlens_k,
            total_seq_len=total_seq_len,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
        )

        # --- Invoke module ---
        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=past_key_values,
            gqa_context=gqa_context,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)

        model = _make_model(graph)
        # Register the com.microsoft domain opset import for GQA
        model.graph.opset_imports["com.microsoft"] = 1
        return ModelPackage({"model": model}, config=config)


class HybridCausalLMTask(ModelTask):
    """Causal LM with hybrid KV cache + DeltaNet recurrent states.

    For models with mixed ``"full_attention"`` and ``"linear_attention"``
    layers (e.g. Qwen3.5).  Full-attention layers use standard KV cache;
    linear-attention (DeltaNet) layers carry ``conv_state`` and
    ``recurrent_state`` tensors instead.

    Inputs (per layer):
        Full attention:
          - past_key_values.{i}.key: [batch, num_kv_heads, past_seq_len, head_dim]
          - past_key_values.{i}.value: [batch, num_kv_heads, past_seq_len, head_dim]
        Linear attention:
          - past_key_values.{i}.conv_state: [batch, conv_dim, kernel_size-1]
          - past_key_values.{i}.recurrent_state: [batch, num_v_heads, k_dim, v_dim]

    Outputs:
        - logits: FLOAT
        - present.{i}.{key|value|conv_state|recurrent_state}: FLOAT
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [input_ids, attention_mask, position_ids]

        cache_inputs, past_key_values = _make_hybrid_cache_inputs(
            config,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(cache_inputs)

        graph, builder = _make_graph(graph_inputs)
        op = builder.op

        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_hybrid_cache_outputs(
            graph,
            present_key_values,
            config.layer_types or [],
        )

        model = _make_model(graph)
        _register_linear_attention_functions(model, config)
        return ModelPackage({"model": model}, config=config)


def _make_static_cache_inputs(
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: ir.DataType,
    batch: ir.SymbolicDim,
    max_seq_len: int,
) -> tuple[list[ir.Value], list[StaticCacheState]]:
    """Create static KV cache inputs for ``num_layers`` layers.

    Returns:
        ``(flat_inputs, static_caches)`` where *flat_inputs* is a flat
        list suitable for extending ``graph_inputs``, and
        *static_caches* is a list of :class:`StaticCacheState` tuples
        for passing to the module via ``past_key_values``.
    """
    kv_hidden = num_key_value_heads * head_dim
    flat: list[ir.Value] = []
    cache_pairs: list[tuple[ir.Value, ir.Value]] = []

    for i in range(num_layers):
        key_cache = ir.Value(
            name=f"key_cache.{i}",
            shape=ir.Shape([batch, max_seq_len, kv_hidden]),
            type=ir.TensorType(dtype),
        )
        value_cache = ir.Value(
            name=f"value_cache.{i}",
            shape=ir.Shape([batch, max_seq_len, kv_hidden]),
            type=ir.TensorType(dtype),
        )
        flat.extend([key_cache, value_cache])
        cache_pairs.append((key_cache, value_cache))

    # Shared inputs across all layers
    write_indices = ir.Value(
        name="write_indices",
        shape=ir.Shape([batch]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    nonpad_kv_seqlen = ir.Value(
        name="nonpad_kv_seqlen",
        shape=ir.Shape([batch]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    flat.extend([write_indices, nonpad_kv_seqlen])

    # Build StaticCacheState for each layer (shared indices)
    static_caches: list[StaticCacheState] = []
    for key_cache, value_cache in cache_pairs:
        static_caches.append(
            StaticCacheState(
                key_cache=key_cache,
                value_cache=value_cache,
                write_indices=write_indices,
                nonpad_kv_seqlen=nonpad_kv_seqlen,
            )
        )

    return flat, static_caches


def _register_static_cache_outputs(
    graph: ir.Graph,
    present_key_values: list[tuple[ir.Value, ir.Value]],
) -> None:
    """Name and register static cache outputs on the graph.

    Output shapes and dtypes are inferred by the shape inference pass
    that runs during model optimization.
    """
    for i, (updated_key, updated_value) in enumerate(present_key_values):
        updated_key.name = f"updated_key_cache.{i}"
        updated_value.name = f"updated_value_cache.{i}"
        graph.outputs.append(updated_key)
        graph.outputs.append(updated_value)


def _validate_static_cache_support(module: nn.Module) -> None:
    """Check that the module's decoder layers support StaticCacheState.

    Only :class:`DecoderLayer` and :class:`MoEDecoderLayer` have the
    ``isinstance(StaticCacheState)`` dispatch in ``forward()``.  Custom
    decoder layers will silently unpack the NamedTuple as a regular
    ``(key, value)`` tuple, producing wrong results.

    NOTE: The following models are NOT yet supported in static cache
    mode and will raise TypeError from this check:

    - **Gemma2**: ``Gemma2Attention`` overrides ``forward()`` and calls
      ``op.Attention`` directly with ``attn_logit_softcapping``,
      bypassing ``_apply_attention()``.  Needs Attention refactoring to
      support softcap in the shared path.

    - **GPT-2**: Uses learned positional embeddings (not RoPE).
      ``_GPT2TextModel.forward()`` unconditionally calls
      ``create_attention_bias()``, so ``attention_mask=None`` would
      fail.  Needs position embedding adaptation.

    - **Falcon (ALiBi)**: The ALiBi variant uses ``is_causal=0`` with a
      position-dependent bias that encodes both causal masking and
      distance-based attention decay.  This is fundamentally
      incompatible with the ``is_causal=1`` static cache pattern.

    Raises:
        TypeError: If any decoder layer is not a supported type.
    """
    from mobius.components._decoder import DecoderLayer
    from mobius.models.moe import MoEDecoderLayer

    for name, child in module.named_modules():
        if not isinstance(child, nn.ModuleList):
            continue
        for i, layer in enumerate(child):
            if not isinstance(layer, nn.Module):
                continue
            # Check modules that look like decoder layers: they have an
            # attention sub-module named either "self_attn" (standard) or
            # "attn" (GPT-2 style).
            if not hasattr(layer, "self_attn") and not hasattr(layer, "attn"):
                continue
            if not isinstance(layer, (DecoderLayer, MoEDecoderLayer)):
                raise TypeError(
                    f"Static cache mode requires decoder layers that "
                    f"inherit from DecoderLayer or MoEDecoderLayer, but "
                    f"{name}[{i}] is {type(layer).__name__}. Either use a "
                    f"compatible model or add StaticCacheState dispatch to "
                    f"{type(layer).__name__}.forward()."
                )
