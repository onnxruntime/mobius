# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Causal language model tasks with internal and static KV cache."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import GraphBuilder, nn

from mobius._build_context import prefill_prefix_pruning
from mobius._configs import ArchitectureConfig
from mobius._constants import (
    STATIC_CACHE_KV_SEQUENCE_LENGTH,
    STATIC_CACHE_WRITE_INDICES,
)
from mobius._model_package import ModelPackage
from mobius.components._attention import StaticCacheState
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._cache_utils import (
    _make_hybrid_cache_inputs,
    _make_kv_cache_inputs,
    _register_hybrid_cache_outputs,
    _register_kv_cache_outputs,
    _register_linear_attention_functions,
)


class CausalLMTask(ModelTask):
    """Causal language model with KV cache for text generation.

    Supports two cache modes:

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

    The module's ``forward()`` must accept
    ``(op, input_ids, attention_mask, position_ids, past_key_values)``
    and return ``(logits, list_of_(key, value)_tuples)``.  In static cache
    mode, ``attention_mask`` will be ``None`` and ``past_key_values``
    entries will be :class:`StaticCacheState` tuples.

    Args:
        static_cache: If ``True``, use pre-allocated static KV cache
            buffers instead of dynamic concatenation.
        max_seq_len: Maximum sequence length for static cache buffers.
            Only used when ``static_cache=True``.  Defaults to
            ``config.max_position_embeddings``.
        prune_prefill_prefix: If ``True``, insert ``Gather(axis=1, index=-1)``
            before the LM head so only the last token's hidden state is
            projected to logits.  Output logits shape becomes ``[B, 1,
            vocab]`` instead of ``[B, S, vocab]``, reducing prefill cost
            for large-vocabulary models.  Set this when the downstream
            runtime only needs the final token's logits (single-token
            autoregressive generation).  Breaks workflows that require
            per-token logits (logprob scoring, speculative decoding,
            multi-token generation).
    """

    def __init__(
        self,
        *,
        static_cache: bool = False,
        paged_cache: bool = False,
        max_seq_len: int | None = None,
        prune_prefill_prefix: bool = False,
    ):
        if static_cache and paged_cache:
            raise ValueError("static_cache and paged_cache are mutually exclusive.")
        self._static_cache = static_cache
        self._paged_cache = paged_cache
        self._max_seq_len = max_seq_len
        self._prune_prefill_prefix = prune_prefill_prefix

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
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

        # --- Graph input dims ---
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        # --- Build graph first, then create inputs via builder ---
        graph, builder = _make_graph()
        op = builder.op

        # --- Inputs common to both modes ---
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])

        # --- Paged-cache mode: caller-owned LATENT PagedAttention cache. ---
        if self._paged_cache:
            return self._build_paged(module, config, graph, builder, op, input_ids, batch)

        # --- Cache setup (static vs dynamic) ---
        if static:
            attention_mask = None
            position_ids = builder.input(
                "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
            )
            # Models may expose per-cache-layer specs (e.g. Gemma4: only
            # non-KV-shared layers own a cache, and sliding vs full layers use
            # different head_dim). Uniform models leave this unset.
            specs_fn = getattr(module, "static_kv_cache_specs", None)
            cache_specs = specs_fn() if callable(specs_fn) else None
            past_key_values = _make_static_cache_inputs(
                builder,
                config.num_hidden_layers,
                config.num_key_value_heads,
                config.head_dim,
                config.dtype,
                batch,
                max_seq_len,
                cache_specs=cache_specs,
            )
        else:
            past_seq_len = ir.SymbolicDim("past_sequence_len")
            attention_mask = builder.input(
                "attention_mask",
                dtype=ir.DataType.INT64,
                shape=[batch, "past_seq_len + seq_len"],
            )
            position_ids = builder.input(
                "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
            )

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
            kv_key_head_dim = (
                (config.qk_nope_head_dim or 0) + (config.qk_rope_head_dim or 0)
            ) or config.head_dim
            kv_value_head_dim = config.v_head_dim or config.head_dim

            # Models whose trailing layers borrow K,V from an earlier layer
            # (Gemma 3n's ``num_kv_shared_layers``) own fewer cache entries
            # than they have layers; they report the count via this hook.
            count_fn = getattr(module, "kv_cache_layer_count", None)
            num_cache_layers = count_fn() if callable(count_fn) else config.num_hidden_layers

            past_key_values = _make_kv_cache_inputs(
                builder,
                num_cache_layers,
                num_kv_cache_heads,
                config.head_dim,
                config.dtype,
                batch,
                past_seq_len,
                key_head_dim=kv_key_head_dim,
                value_head_dim=kv_value_head_dim,
            )

        with prefill_prefix_pruning(self._prune_prefill_prefix):
            result = module(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
        intermediate_hidden_states: list | None = None
        final_hidden_state: ir.Value | None = None
        if len(result) == 4:
            logits, present_key_values, intermediate_hidden_states, final_hidden_state = result
        elif len(result) == 3:
            logits, present_key_values, intermediate_hidden_states = result
        else:
            logits, present_key_values = result

        _validate_pruned_logits(logits, self._prune_prefill_prefix, module)

        builder.add_output(logits, "logits")

        # --- Output registration (static vs dynamic) ---
        if static:
            _register_static_cache_outputs(
                builder,
                present_key_values,
            )
        else:
            # Stamp explicit present shapes symmetric to the past inputs so the
            # com.microsoft::GroupQueryAttention export declares the correct
            # head_dim (its contrib-op shape inference otherwise mis-derives it
            # from the packed QKV hidden). total_seq = past + current sequence.
            _register_kv_cache_outputs(
                builder,
                present_key_values,
                batch=batch,
                num_kv_heads=num_kv_cache_heads,
                key_head_dim=kv_key_head_dim,
                value_head_dim=kv_value_head_dim,
                total_seq_len="past_sequence_len + sequence_len",
                dtype=config.dtype,
            )

        if intermediate_hidden_states is not None:
            _register_intermediate_hidden_states(
                builder, config.output_layer_indices, intermediate_hidden_states
            )
        if final_hidden_state is not None:
            builder.add_output(final_hidden_state, "mtp_seed")

        return ModelPackage({"model": _make_model(graph)}, config=config)

    def _build_paged(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
        graph,
        builder: GraphBuilder,
        op,
        input_ids: ir.Value,
        batch: ir.SymbolicDim,
    ) -> ModelPackage:
        """Emit a decoder that binds caller-owned LATENT ``PagedAttention``.

        The page/cache tensors (``block_table``, ``slot_mapping``, cumulative
        lengths, past lengths, per-layer ``key_cache.N``) are graph inputs owned
        by the native page manager; this task never allocates or manages pages.
        """
        from mobius.components._paged_mla import (
            mla_paged_geometry,
        )

        # Eligibility must hold; an incompatible geometry is a typed error here
        # (never a silent dense fallback).
        geom = mla_paged_geometry(config)

        # LATENT PagedAttention applies RoPE in-op and derives each token's
        # absolute position from past_seqlens + cumulative_sequence_length, so
        # the graph takes no position_ids input.
        paged_states = _make_paged_cache_inputs(
            builder,
            config.num_hidden_layers,
            geom.head_size,
            config.dtype,
            batch,
        )

        result = module(
            op,
            input_ids=input_ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=paged_states,
        )
        if len(result) == 3:
            logits, present_key_values, _intermediate = result
        else:
            logits, present_key_values = result

        builder.add_output(logits, "logits")
        _register_paged_cache_outputs(builder, present_key_values)
        return ModelPackage({"model": _make_model(graph)}, config=config)


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

    Args:
        prune_prefill_prefix: If ``True``, insert ``Gather(axis=1, index=-1)``
            before the LM head so only the last token's logits are emitted.
            See :class:`CausalLMTask` for full documentation.
    """

    def __init__(self, *, prune_prefill_prefix: bool = False):
        self._prune_prefill_prefix = prune_prefill_prefix

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph()
        op = builder.op

        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_seq_len + seq_len"],
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )

        past_key_values = _make_hybrid_cache_inputs(
            builder,
            config,
            config.dtype,
            batch,
            past_seq_len,
        )

        with prefill_prefix_pruning(self._prune_prefill_prefix):
            result = module(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
        intermediate_hidden_states: list | None = None
        final_hidden_state: ir.Value | None = None
        if len(result) == 4:
            logits, present_key_values, intermediate_hidden_states, final_hidden_state = result
        elif len(result) == 3:
            logits, present_key_values, intermediate_hidden_states = result
        else:
            logits, present_key_values = result

        _validate_pruned_logits(logits, self._prune_prefill_prefix, module)

        builder.add_output(logits, "logits")
        _register_hybrid_cache_outputs(
            builder,
            present_key_values,
            config.layer_types or [],
        )

        if intermediate_hidden_states is not None:
            _register_intermediate_hidden_states(
                builder, config.output_layer_indices, intermediate_hidden_states
            )
        if final_hidden_state is not None:
            builder.add_output(final_hidden_state, "mtp_seed")

        model = _make_model(graph)
        _register_linear_attention_functions(model, config)
        if config.model_type == "jamba":
            model.metadata_props["mobius.runtime_support"] = (
                "Deferred: heterogeneous attention KV and Mamba recurrent state "
                "discovery is tracked by https://github.com/onnxruntime/mobius#605"
            )
        return ModelPackage({"model": model}, config=config)


def _validate_pruned_logits(
    logits: ir.Value,
    prune_prefill_prefix: bool,
    module: nn.Module,
) -> None:
    """Fail when a custom model forward ignores the task's pruning request."""
    if not prune_prefill_prefix:
        return
    shape = logits.shape
    if shape is None or len(shape) != 3 or shape[1] != 1:
        raise ValueError(
            f"{type(module).__name__} does not support prune_prefill_prefix. "
            "The model must select the final hidden state before its LM-head projection."
        )


def _register_intermediate_hidden_states(
    builder: GraphBuilder,
    indices: list[int] | None,
    intermediate_hidden_states: list[ir.Value],
) -> None:
    """Register selected per-layer hidden-state tensors as graph outputs.

    Each entry ``intermediate_hidden_states[i]`` is the post-residual
    output of decoder layer ``indices[i]`` (before the final ``self.norm``).
    The outputs are named ``hidden_states.{idx}`` so a downstream
    speculative-decoding draft can address them by layer index.

    See :class:`ArchitectureConfig.output_layer_indices` for the index
    convention.
    """
    if not indices:
        return
    if len(indices) != len(intermediate_hidden_states):
        raise ValueError(
            f"output_layer_indices has {len(indices)} entries but the model "
            f"returned {len(intermediate_hidden_states)} intermediate "
            "hidden-state tensors."
        )
    for idx, hs in zip(indices, intermediate_hidden_states):
        builder.add_output(hs, f"hidden_states.{idx}")


def _make_static_cache_inputs(
    builder: GraphBuilder,
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: ir.DataType,
    batch: ir.SymbolicDim,
    max_seq_len: int,
    cache_specs: list[tuple[int, int]] | None = None,
) -> list[StaticCacheState]:
    """Create static KV cache inputs for the cache-owning layers.

    Uses ``builder.input()`` to create and register graph inputs directly.

    Args:
        cache_specs: Optional per-cache-layer ``(num_key_value_heads, head_dim)``
            list. When provided (e.g. from a model's ``static_kv_cache_specs()``),
            one buffer is allocated per entry with its own ``kv_hidden`` — this
            supports models where only a subset of layers own a cache and/or the
            head_dim varies per layer (Gemma4: KV-shared layers borrow K,V, and
            sliding vs full layers use different head_dim). When ``None``, falls
            back to ``num_layers`` uniform buffers of
            ``num_key_value_heads * head_dim``.

    Returns:
        A list of :class:`StaticCacheState` tuples for passing to the
        module via ``past_key_values``.
    """
    if cache_specs is None:
        cache_specs = [(num_key_value_heads, head_dim)] * num_layers

    cache_pairs: list[tuple[ir.Value, ir.Value]] = []
    for i, (kv_heads, layer_head_dim) in enumerate(cache_specs):
        kv_hidden = kv_heads * layer_head_dim
        key_cache = builder.input(
            f"key_cache.{i}",
            dtype=dtype,
            shape=[batch, max_seq_len, kv_hidden],
        )
        value_cache = builder.input(
            f"value_cache.{i}",
            dtype=dtype,
            shape=[batch, max_seq_len, kv_hidden],
        )
        cache_pairs.append((key_cache, value_cache))

    # Shared inputs across all layers
    write_indices = builder.input(
        STATIC_CACHE_WRITE_INDICES,
        dtype=ir.DataType.INT64,
        shape=[batch],
    )
    nonpad_kv_seqlen = builder.input(
        STATIC_CACHE_KV_SEQUENCE_LENGTH,
        dtype=ir.DataType.INT64,
        shape=[batch],
    )

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

    return static_caches


def _register_static_cache_outputs(
    builder: GraphBuilder,
    present_key_values: list[tuple[ir.Value, ir.Value]],
) -> None:
    """Name and register static cache outputs on the graph.

    Output shapes and dtypes are inferred by the shape inference pass
    that runs during model optimization.
    """
    for i, (updated_key, updated_value) in enumerate(present_key_values):
        builder.add_output(updated_key, f"updated_key_cache.{i}")
        builder.add_output(updated_value, f"updated_value_cache.{i}")


def _make_paged_cache_inputs(
    builder: GraphBuilder,
    num_layers: int,
    head_size: int,
    dtype: ir.DataType,
    batch: ir.SymbolicDim,
):
    """Create caller-owned LATENT ``PagedAttention`` cache inputs.

    Per-layer ``key_cache.{i}`` LATENT buffers plus the shared index inputs
    (``block_table``, ``slot_mapping``, ``cumulative_sequence_length``,
    ``past_seqlens``). ``cos_cache``/``sin_cache`` are supplied by the model
    from its RoPE parameters, so the returned states leave them ``None``.

    All buffers are graph inputs owned by the native page manager; this task
    allocates none of them and never creates a second cache authority.
    """
    from mobius.components._paged_mla import PagedCacheState

    num_blocks = ir.SymbolicDim("num_blocks")
    block_size = ir.SymbolicDim("block_size")
    max_blocks = ir.SymbolicDim("max_blocks_per_seq")
    num_tokens = ir.SymbolicDim("num_tokens")

    # Shared, sequence-level index inputs (int32 positional constraint S).
    block_table = builder.input(
        "block_table", dtype=ir.DataType.INT32, shape=[batch, max_blocks]
    )
    slot_mapping = builder.input("slot_mapping", dtype=ir.DataType.INT32, shape=[num_tokens])
    cumulative_sequence_length = builder.input(
        "cumulative_sequence_length", dtype=ir.DataType.INT32, shape=["batch + 1"]
    )
    past_seqlens = builder.input("past_seqlens", dtype=ir.DataType.INT32, shape=[batch])

    states = []
    for i in range(num_layers):
        key_cache = builder.input(
            f"key_cache.{i}",
            dtype=dtype,
            shape=[num_blocks, block_size, 1, head_size],
        )
        states.append(
            PagedCacheState(
                key_cache=key_cache,
                block_table=block_table,
                slot_mapping=slot_mapping,
                cumulative_sequence_length=cumulative_sequence_length,
                past_seqlens=past_seqlens,
                cos_cache=None,
                sin_cache=None,
            )
        )
    return states


def _register_paged_cache_outputs(
    builder: GraphBuilder,
    present_key_values,
) -> None:
    """Name the in-place LATENT cache outputs (each aliases its ``key_cache``)."""
    for i, present in enumerate(present_key_values):
        updated_key = present[0] if isinstance(present, (tuple, list)) else present
        builder.add_output(updated_key, f"updated_key_cache.{i}")


def _validate_static_cache_support(module: nn.Module) -> None:
    """Check that the module's decoder layers support StaticCacheState.

    Shared decoder layers have the ``isinstance(StaticCacheState)`` dispatch
    in ``forward()``. Custom decoder layers must opt in with the
    ``_supports_static_cache`` marker after implementing equivalent handling;
    otherwise they may silently unpack the NamedTuple as a regular
    ``(key, value)`` tuple, producing wrong results.

    Also warns when the model uses sliding-window attention, since the
    static cache path does not enforce window constraints (the Attention
    op uses ``is_causal=1`` without ``local_window_size``).

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

    # Whitelist-based validation: only check layers that have self_attn/attn
    # (decoder-like), and accept shared implementations or explicit opt-ins.
    # This naturally skips vision/audio encoder layers since they use
    # different classes (e.g. Gemma4VisionEncoderLayer).
    for name, child in module.named_modules():
        if not isinstance(child, nn.ModuleList):
            continue
        for i, layer in enumerate(child):
            if not isinstance(layer, nn.Module):
                continue
            if not hasattr(layer, "self_attn") and not hasattr(layer, "attn"):
                continue
            # A layer qualifies if it inherits the shared DecoderLayer/MoEDecoderLayer
            # static dispatch, or opts in via the ``_supports_static_cache`` class
            # marker (custom layers that implement StaticCacheState handling
            # themselves, e.g. Gemma4DecoderLayer with KV-shared + dual head_dim).
            if isinstance(layer, (DecoderLayer, MoEDecoderLayer)):
                continue
            if getattr(type(layer), "_supports_static_cache", False):
                continue
            raise TypeError(
                f"Static cache mode requires decoder layers that "
                f"inherit from DecoderLayer or MoEDecoderLayer (or set "
                f"_supports_static_cache=True), but "
                f"{name}[{i}] is {type(layer).__name__}. Either use a "
                f"compatible model or add StaticCacheState dispatch to "
                f"{type(layer).__name__}.forward()."
            )
