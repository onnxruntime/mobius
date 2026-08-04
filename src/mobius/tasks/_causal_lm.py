# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Causal language model tasks with internal and static KV cache."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import GraphBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.components._attention import PagedCacheState, StaticCacheState
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

    **Paged cache** (``paged_cache=True``):
        Block-table / paged KV cache (onnx-genai ``docs/DESIGN.md`` §39.4
        Option C).  KV lives in a shared *page pool* of fixed-size pages that
        are non-contiguous per sequence.  New tokens are written to physical
        slots via ``ScatterND`` and the sequence's pages are assembled
        contiguously via ``Gather(pool, block_table)`` before attention.  This
        is the vLLM PagedAttention layout; because sequences can share physical
        pages through their ``block_table``, it also supports SGLang
        RadixAttention (shared prefix pages) with no graph change.

        Inputs:
            - input_ids: [batch, seq_len] INT64
            - position_ids: [batch, seq_len] INT64
            - key_pool.{i}: [num_pages, page_size, kv_hidden] FLOAT per layer
            - value_pool.{i}: [num_pages, page_size, kv_hidden] FLOAT per layer
            - block_table: [num_blocks] INT64 (physical page ids, logical order)
            - slot_mapping: [seq_len] INT64 (flat slot per new token)
            - nonpad_kv_seqlen: [batch] INT64
        Outputs:
            - logits: FLOAT
            - updated_key_pool.{i} / updated_value_pool.{i}: FLOAT

        No ``attention_mask`` input — causal masking uses ``is_causal=1``.
        Targets a single active sequence (``batch == 1``); multi-sequence
        batching is a documented TODO.

    The module's ``forward()`` must accept
    ``(op, input_ids, attention_mask, position_ids, past_key_values)``
    and return ``(logits, list_of_(key, value)_tuples)``.  In static/paged
    cache mode, ``attention_mask`` will be ``None`` and ``past_key_values``
    entries will be :class:`StaticCacheState` / :class:`PagedCacheState`
    tuples.

    Args:
        static_cache: If ``True``, use pre-allocated static KV cache
            buffers instead of dynamic concatenation.
        paged_cache: If ``True``, use a paged / block-table KV cache (page
            pool + block_table + slot_mapping).  Mutually exclusive with
            ``static_cache``.
        max_seq_len: Maximum sequence length for static cache buffers.
            Only used when ``static_cache=True``.  Defaults to
            ``config.max_position_embeddings``.
        page_size: Number of tokens per page for the paged cache.  Only used
            when ``paged_cache=True``.  Defaults to 16.
        num_pages: Number of physical pages in the pool.  Only used when
            ``paged_cache=True``.  Left symbolic (dynamic) when ``None`` so the
            runtime can size the pool; pass an int to stamp a fixed pool size.
    """

    def __init__(
        self,
        *,
        static_cache: bool = False,
        paged_cache: bool = False,
        max_seq_len: int | None = None,
        page_size: int = 16,
        num_pages: int | None = None,
    ):
        if static_cache and paged_cache:
            raise ValueError(
                "static_cache and paged_cache are mutually exclusive; enable at most one."
            )
        self._static_cache = static_cache
        self._paged_cache = paged_cache
        self._max_seq_len = max_seq_len
        self._page_size = page_size
        self._num_pages = num_pages

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        static = self._static_cache
        paged = self._paged_cache

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

        # --- Paged-cache pre-validation ---
        if paged:
            if self._page_size is None or self._page_size <= 0:
                raise ValueError("page_size must be a positive integer for paged cache.")
            if self._num_pages is not None and self._num_pages <= 0:
                raise ValueError("num_pages must be a positive integer when provided.")
            # Paged cache reuses the same DecoderLayer/MoEDecoderLayer dispatch
            # as the static cache (both flow their state through past_key_value).
            _validate_static_cache_support(module)

        # --- Graph input dims ---
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        # --- Build graph first, then create inputs via builder ---
        graph, builder = _make_graph()
        op = builder.op

        # --- Inputs common to both modes ---
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])

        # --- Cache setup (paged vs static vs dynamic) ---
        if paged:
            attention_mask = None
            position_ids = builder.input(
                "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
            )
            past_key_values = _make_paged_cache_inputs(
                builder,
                config.num_hidden_layers,
                config.num_key_value_heads,
                config.head_dim,
                config.dtype,
                page_size=self._page_size,
                num_pages=self._num_pages,
            )
        elif static:
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

            past_key_values = _make_kv_cache_inputs(
                builder,
                config.num_hidden_layers,
                num_kv_cache_heads,
                config.head_dim,
                config.dtype,
                batch,
                past_seq_len,
                key_head_dim=kv_key_head_dim,
                value_head_dim=kv_value_head_dim,
            )

        result = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        intermediate_hidden_states: list | None = None
        if len(result) == 3:
            logits, present_key_values, intermediate_hidden_states = result
        else:
            logits, present_key_values = result

        builder.add_output(logits, "logits")

        # --- Output registration (paged vs static vs dynamic) ---
        if paged:
            _register_paged_cache_outputs(
                builder,
                present_key_values,
            )
        elif static:
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
    """

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

        result = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        intermediate_hidden_states: list | None = None
        if len(result) == 3:
            logits, present_key_values, intermediate_hidden_states = result
        else:
            logits, present_key_values = result

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

        model = _make_model(graph)
        _register_linear_attention_functions(model, config)
        return ModelPackage({"model": model}, config=config)


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
        "write_indices",
        dtype=ir.DataType.INT64,
        shape=[batch],
    )
    nonpad_kv_seqlen = builder.input(
        "nonpad_kv_seqlen",
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
    num_key_value_heads: int,
    head_dim: int,
    dtype: ir.DataType,
    *,
    page_size: int,
    num_pages: int | None,
) -> list[PagedCacheState]:
    """Create paged (block-table) KV cache inputs for ``num_layers`` layers.

    Emits a per-layer page pool ``key_pool.{i}`` / ``value_pool.{i}`` of shape
    ``[num_pages, page_size, kv_hidden]`` plus the shared ``block_table``,
    ``slot_mapping`` and ``nonpad_kv_seqlen`` inputs, and packs them into one
    :class:`PagedCacheState` per layer (shared block/slot/nonpad tensors).

    ``num_pages`` is left symbolic (dynamic dimension ``num_pages``) when
    ``None`` so the runtime can size the pool; passing an int stamps a fixed
    pool size.  ``block_table`` (``[num_blocks]``) and ``slot_mapping``
    (``[seq_len]``) are 1-D — the current implementation targets a single
    active sequence (``batch == 1``).

    Returns:
        A list of :class:`PagedCacheState` tuples for passing to the module
        via ``past_key_values``.
    """
    kv_hidden = num_key_value_heads * head_dim
    pages_dim: int | str = num_pages if num_pages is not None else "num_pages"

    pool_pairs: list[tuple[ir.Value, ir.Value]] = []
    for i in range(num_layers):
        key_pool = builder.input(
            f"key_pool.{i}",
            dtype=dtype,
            shape=[pages_dim, page_size, kv_hidden],
        )
        value_pool = builder.input(
            f"value_pool.{i}",
            dtype=dtype,
            shape=[pages_dim, page_size, kv_hidden],
        )
        pool_pairs.append((key_pool, value_pool))

    # Shared inputs across all layers.
    block_table = builder.input(
        "block_table",
        dtype=ir.DataType.INT64,
        shape=["num_blocks"],
    )
    slot_mapping = builder.input(
        "slot_mapping",
        dtype=ir.DataType.INT64,
        shape=["sequence_len"],
    )
    nonpad_kv_seqlen = builder.input(
        "nonpad_kv_seqlen",
        dtype=ir.DataType.INT64,
        shape=["batch"],
    )

    paged_caches: list[PagedCacheState] = []
    for key_pool, value_pool in pool_pairs:
        paged_caches.append(
            PagedCacheState(
                key_pool=key_pool,
                value_pool=value_pool,
                block_table=block_table,
                slot_mapping=slot_mapping,
                nonpad_kv_seqlen=nonpad_kv_seqlen,
            )
        )
    return paged_caches


def _register_paged_cache_outputs(
    builder: GraphBuilder,
    present_key_values: list[tuple[ir.Value, ir.Value]],
) -> None:
    """Name and register paged (block-table) cache outputs on the graph.

    Each layer produces the UPDATED page pools (the ``ScatterND`` result),
    registered as ``updated_key_pool.{i}`` / ``updated_value_pool.{i}``.
    Shapes/dtypes are inferred by the shape inference pass.
    """
    for i, (updated_key_pool, updated_value_pool) in enumerate(present_key_values):
        builder.add_output(updated_key_pool, f"updated_key_pool.{i}")
        builder.add_output(updated_value_pool, f"updated_value_pool.{i}")


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
