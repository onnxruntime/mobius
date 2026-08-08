# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4 task classes.

The unified :class:`Gemma4Task` builds a 3- or 4-model package:

1. **decoder** (text decoder): ``inputs_embeds`` → logits + KV cache
2. **vision** (vision encoder): ``pixel_values, pixel_position_ids`` → ``image_features``
3. **embedding**: ``input_ids, image_features[, audio_features]`` → ``inputs_embeds``
4. **audio** (audio encoder, only when ``config.audio is not None``):
   ``input_features, input_features_mask`` → ``audio_features``

The decoder uses per-layer KV cache where local (sliding_attention) layers
use ``config.head_dim`` and global (full_attention) layers use
``config.global_head_dim``.  When ``config.num_kv_shared_layers > 0``, the
last ``num_kv_shared_layers`` decoder layers share K,V from earlier layers
and therefore have NO separate KV cache entries — the cache only contains
entries for the first ``num_hidden_layers - num_kv_shared_layers`` layers.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import GraphBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._configs import Gemma4Config
from mobius._model_package import ModelPackage
from mobius._pipeline_contract import (
    declare_component_presence,
    declare_optional_input,
)
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._cache_utils import (
    _register_kv_cache_outputs,
)


def _register_hybrid_cache_outputs(
    builder: GraphBuilder,
    present_key_values: list[tuple[ir.Value, ir.Value]],
    config: Gemma4Config,
) -> None:
    """Register cache outputs for hybrid static/dynamic Gemma4 models.

    Full-attention layers use ``updated_key_cache.{i}`` / ``updated_value_cache.{i}``.
    Sliding-attention layers use ``present.{i}.key`` / ``present.{i}.value``.
    """
    layer_types = config.layer_types or (["sliding_attention"] * config.num_hidden_layers)

    for i, (k, v) in enumerate(present_key_values):
        lt = layer_types[i] if i < len(layer_types) else "sliding_attention"
        if lt == "full_attention":
            builder.add_output(k, f"updated_key_cache.{i}")
            builder.add_output(v, f"updated_value_cache.{i}")
        else:
            builder.add_output(k, f"present.{i}.key")
            builder.add_output(v, f"present.{i}.value")


def _make_gemma4_static_cache_inputs(
    builder: GraphBuilder,
    config: Gemma4Config,
    batch: ir.SymbolicDim,
    max_seq_len: int,
    past_seq_len: ir.SymbolicDim | None = None,
) -> list:
    """Create per-layer hybrid KV cache inputs for Gemma4 static cache mode.

    Full-attention layers get :class:`StaticCacheState` (TensorScatter).
    Sliding-attention layers get dynamic ``(past_key, past_value)`` tuples
    (GQA with ``local_window_size``).  KV-shared layers get ``None``.

    Why sliding window layers can't use static cache:
        Sliding window attention requires only attending to the most recent
        N tokens.  The standard ONNX Attention op lacks a
        ``local_window_size`` parameter to enforce this constraint.  With a
        static (pre-allocated) KV cache, stale entries beyond the window
        remain in the buffer and would be incorrectly attended to.
        ``GroupQueryAttention`` (GQA) supports ``local_window_size``
        natively and manages its own KV cache, so sliding window layers use
        GQA with dynamic cache while full-attention layers use
        TensorScatter with static cache.

    Args:
        past_seq_len: Symbolic dim for dynamic cache sequence length.
            Required when the config has sliding-attention layers.
    """
    from mobius.components._attention import StaticCacheState

    local_head_dim = config.head_dim
    global_head_dim = config.global_head_dim or config.head_dim
    num_kv_shared = config.num_kv_shared_layers or 0
    num_kv_layers = config.num_hidden_layers - num_kv_shared
    layer_types = config.layer_types or (["sliding_attention"] * config.num_hidden_layers)

    # Per non-shared layer: static or dynamic cache
    layer_entries: list[tuple[str, object]] = []  # ("static"|"dynamic", cache)
    for i in range(num_kv_layers):
        lt = layer_types[i] if i < len(layer_types) else "sliding_attention"
        hd = global_head_dim if lt == "full_attention" else local_head_dim
        is_full = lt == "full_attention"
        if is_full and config.num_global_key_value_heads is not None:
            kv_heads = config.num_global_key_value_heads
        else:
            kv_heads = config.num_key_value_heads

        if is_full:
            kv_hidden = kv_heads * hd
            key_cache = builder.input(
                f"key_cache.{i}",
                dtype=config.dtype,
                shape=[batch, max_seq_len, kv_hidden],
            )
            value_cache = builder.input(
                f"value_cache.{i}",
                dtype=config.dtype,
                shape=[batch, max_seq_len, kv_hidden],
            )
            layer_entries.append(("static", (key_cache, value_cache)))
        else:
            assert past_seq_len is not None, (
                "past_seq_len required for sliding-attention dynamic cache"
            )
            past_key = builder.input(
                f"past_key_values.{i}.key",
                dtype=config.dtype,
                shape=[batch, kv_heads, past_seq_len, hd],
            )
            past_value = builder.input(
                f"past_key_values.{i}.value",
                dtype=config.dtype,
                shape=[batch, kv_heads, past_seq_len, hd],
            )
            layer_entries.append(("dynamic", (past_key, past_value)))

    # Shared control inputs for static cache layers
    has_static = any(t == "static" for t, _ in layer_entries)
    write_indices = nonpad_kv_seqlen = None
    if has_static:
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

    # Build full per-layer list
    entry_iter = iter(layer_entries)
    full_list: list = []
    for i in range(config.num_hidden_layers):
        if i >= num_kv_layers:
            full_list.append(None)
        else:
            cache_type, pair = next(entry_iter)
            if cache_type == "static":
                k, v = pair
                full_list.append(
                    StaticCacheState(
                        key_cache=k,
                        value_cache=v,
                        write_indices=write_indices,
                        nonpad_kv_seqlen=nonpad_kv_seqlen,
                    )
                )
            else:
                full_list.append(pair)
    return full_list


def _make_gemma4_kv_cache_inputs(
    builder: GraphBuilder,
    config: Gemma4Config,
    batch: ir.SymbolicDim,
    past_seq_len: ir.SymbolicDim,
) -> list[tuple[ir.Value, ir.Value]]:
    """Create per-layer KV cache inputs accounting for dual head_dim and KV sharing.

    Uses ``builder.input()`` to create and register graph inputs directly.

    Local (sliding_attention) layers use ``config.head_dim``;
    global (full_attention) layers use ``config.global_head_dim``.

    The last ``config.num_kv_shared_layers`` layers share K,V from earlier
    layers and do NOT have their own cache entries — only
    ``num_hidden_layers - num_kv_shared_layers`` entries are created.

    All layers use the original ``num_key_value_heads`` except
    full-attention layers when ``num_global_key_value_heads`` is set
    (fewer KV heads, independent of the ``attention_k_eq_v`` flag).
    The Attention op supports GQA head counts natively.  CUDA EP limitations
    are tracked in microsoft/onnxruntime#28195 and #28196.
    """
    local_head_dim = config.head_dim
    global_head_dim = config.global_head_dim or config.head_dim
    # Number of layers with independent KV projections
    num_kv_shared = config.num_kv_shared_layers or 0
    num_kv_layers = config.num_hidden_layers - num_kv_shared
    layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"Gemma4Config.layer_types length ({len(layer_types)}) must match "
            f"num_hidden_layers ({config.num_hidden_layers})"
        )

    pairs: list[tuple[ir.Value, ir.Value]] = []
    for i in range(num_kv_layers):
        layer_type = layer_types[i] if i < len(layer_types) else "sliding_attention"
        hd = global_head_dim if layer_type == "full_attention" else local_head_dim
        # Full-attention layers use num_global_key_value_heads when set,
        # independent of the k_eq_v flag.
        is_full = layer_type == "full_attention"
        if is_full and config.num_global_key_value_heads is not None:
            kv_heads = config.num_global_key_value_heads
        else:
            kv_heads = config.num_key_value_heads
        past_key = builder.input(
            f"past_key_values.{i}.key",
            dtype=config.dtype,
            shape=[batch, kv_heads, past_seq_len, hd],
        )
        past_value = builder.input(
            f"past_key_values.{i}.value",
            dtype=config.dtype,
            shape=[batch, kv_heads, past_seq_len, hd],
        )
        pairs.append((past_key, past_value))
    return pairs


class Gemma4TextCausalLMTask(ModelTask):
    """Text-only causal LM task for Gemma4 with correct dual head_dim KV cache.

    Identical to :class:`~mobius.tasks.CausalLMTask` but uses
    :func:`_make_gemma4_kv_cache_inputs` for the KV cache so that:

    - global (full_attention) layers use ``config.global_head_dim`` (512)
    - local (sliding_attention) layers use ``config.head_dim`` (256)
    - the last ``config.num_kv_shared_layers`` layers share K,V and have no
      independent cache entries

    Supports ``static_cache=True`` for pre-allocated TensorScatter-based
    KV cache (requires ORT ≥ 1.25.0).

    Inputs (dynamic cache):
        - input_ids: [batch, sequence_len] INT64
        - attention_mask: [batch, past_seq_len + seq_len] INT64
        - position_ids: [batch, sequence_len] INT64
        - past_key_values.{i}.key / .value for i in 0..num_kv_layers-1
    Outputs:
        - logits: FLOAT
        - present.{i}.key / present.{i}.value for i in 0..num_kv_layers-1

    Inputs (static cache):
        - input_ids: [batch, sequence_len] INT64
        - position_ids: [batch, sequence_len] INT64
        - key_cache.{i} / value_cache.{i}: [batch, max_seq_len, kv_hidden]
        - write_indices: [batch] INT64
        - nonpad_kv_seqlen: [batch] INT64
    Outputs:
        - logits: FLOAT
        - updated_key_cache.{i} / updated_value_cache.{i}
    """

    def __init__(
        self,
        *,
        static_cache: bool = False,
        max_seq_len: int | None = None,
    ):
        self._static_cache = static_cache
        self._max_seq_len = max_seq_len

    def build(
        self,
        module: nn.Module,
        config: Gemma4Config,
    ) -> ModelPackage:
        from mobius.tasks._causal_lm import (
            _validate_static_cache_support,
        )

        static = self._static_cache

        if static:
            max_seq_len = self._max_seq_len
            if max_seq_len is None:
                max_seq_len = getattr(config, "max_position_embeddings", None)
            if max_seq_len is None or max_seq_len <= 0:
                raise ValueError("max_seq_len must be a positive integer for static cache.")
            _validate_static_cache_support(module)

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        graph, builder = _make_graph()
        op = builder.op

        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )

        if static:
            # Hybrid mode: sliding layers need attention_mask + dynamic cache,
            # full-attention layers use write_indices/nonpad_kv_seqlen.
            past_seq_len = ir.SymbolicDim("past_sequence_len")
            attention_mask = builder.input(
                "attention_mask",
                dtype=ir.DataType.INT64,
                shape=[batch, "past_seq_len + seq_len"],
            )
            position_ids = builder.input(
                "position_ids",
                dtype=ir.DataType.INT64,
                shape=[batch, seq_len],
            )
            past_key_values = _make_gemma4_static_cache_inputs(
                builder,
                config,
                batch,
                max_seq_len,
                past_seq_len,
            )
        else:
            past_seq_len = ir.SymbolicDim("past_sequence_len")
            attention_mask = builder.input(
                "attention_mask",
                dtype=ir.DataType.INT64,
                shape=[batch, "past_seq_len + seq_len"],
            )
            position_ids = builder.input(
                "position_ids",
                dtype=ir.DataType.INT64,
                shape=[batch, seq_len],
            )
            past_key_values = _make_gemma4_kv_cache_inputs(
                builder,
                config,
                batch,
                past_seq_len,
            )

        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        builder.add_output(logits, "logits")

        if static:
            _register_hybrid_cache_outputs(builder, present_key_values, config)
        else:
            _register_kv_cache_outputs(builder, present_key_values)

        return ModelPackage({"model": _make_model(graph)}, config=config)


class Gemma4Task(ModelTask):
    """Unified task for Gemma4 multimodal models (3- or 4-model split).

    Always builds:
    - ``decoder``: text decoder accepting ``inputs_embeds``
    - ``vision``: vision encoder accepting ``pixel_values, pixel_position_ids``
    - ``embedding``: embedding model fusing ``input_ids`` and multimodal features

    When ``config.audio is not None``, also builds:
    - ``audio``: Conformer audio encoder accepting ``input_features``
      (and adds ``audio_features`` as a third input to ``embedding``)

    Decoder KV cache is per-layer with the correct head_dim for each layer type
    (local vs global), unlike the uniform head_dim in :class:`VisionLanguageTask`.

    Supports ``static_cache=True`` for pre-allocated TensorScatter-based
    KV cache on the decoder (requires ORT ≥ 1.25.0).  Vision, audio, and
    embedding models are unaffected by the static cache setting.

    Batching strategies
    -------------------
    Each sub-model uses a different strategy for variable-size inputs:

    **Vision** — padded patches with sentinel position IDs.
        All images are patchified to a fixed ``max_soft_tokens`` (280)
        slots.  Images smaller than the maximum get ``(-1, -1)`` in their
        ``pixel_position_ids`` for unused patch slots, which the encoder
        ignores.  No explicit mask is needed.

    **Audio** — explicit contiguous bool mask.
        Audio clips of different lengths are padded to equal time
        dimension.  ``input_features_mask [B, T]`` marks valid frames
        (``True``) vs padding (``False``), always contiguous
        (right-padded).  The mask is downsampled through two conv layers
        (stride 2 each → ``T//4``) and used to zero out padding in
        Conformer attention. The export strips padding inside the ONNX
        graph and returns ``audio_features [num_valid, hidden_size]``.

    **Decoder** — standard ``attention_mask`` for KV cache padding
        (dynamic mode), or ``write_indices``/``nonpad_kv_seqlen`` for
        pre-allocated cache (static mode).
    """

    #: decoder + vision + embedding, plus audio when ``config.audio`` is set.
    #: ``audio_encoder`` is declared statically (it is config-gated at build time).
    model_roles: ClassVar[dict[str, str]] = {
        "decoder": "decoder",
        "vision_encoder": "encoder",
        "audio_encoder": "encoder",
        "embedding": "embedding",
    }

    def __init__(
        self,
        *,
        static_cache: bool = False,
        max_seq_len: int | None = None,
    ):
        self._static_cache = static_cache
        self._max_seq_len = max_seq_len

    def build(
        self,
        module: nn.Module,
        config: Gemma4Config,
    ) -> ModelPackage:
        # Decide whether to split the fused [V, L*D] per-layer embedding table
        # into L separate [V, D] tables.  Required when the fused table exceeds
        # the EP's max_buffer_size (e.g. WebGPU's 256 MiB limit).
        caps = ep_capabilities()
        per_layer_dim = getattr(config, "hidden_size_per_layer_input", 0)
        vocab_per_layer = getattr(config, "vocab_size_per_layer_input", 0)
        if caps.max_buffer_size and per_layer_dim and vocab_per_layer:
            dtype_bytes = int(config.dtype.itemsize)
            fused_bytes = (
                vocab_per_layer * config.num_hidden_layers * per_layer_dim * dtype_bytes
            )
            config.split_per_layer_embedding = fused_bytes > caps.max_buffer_size
        else:
            config.split_per_layer_embedding = False
        models: dict[str, ir.Model] = {}
        models["decoder"] = self._build_decoder(module.decoder, config)
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        if config.audio is not None:
            models["audio_encoder"] = self._build_audio(module.audio_encoder, config)
        models["embedding"] = self._build_embedding(module.embedding, config)
        return ModelPackage(models, config=config)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build text decoder: inputs_embeds [+ per_layer_inputs] -> logits + KV cache.

        When ``hidden_size_per_layer_input > 0`` (e.g. Gemma4 E2B), the decoder
        accepts precomputed ``per_layer_inputs`` from the embedding model instead
        of ``input_ids``.  This moves the per-layer embedding computation to the
        embedding model, simplifying the decoder graph.

        Exception: When the EP's ``max_buffer_size`` is set and the fused table
        would exceed it, split per-layer tables are used in the decoder instead,
        so ``input_ids`` is passed and ``per_layer_inputs`` is omitted.
        """
        from mobius.tasks._causal_lm import (
            _validate_static_cache_support,
        )

        static = self._static_cache
        if static:
            max_seq_len = self._max_seq_len
            if max_seq_len is None:
                max_seq_len = getattr(config, "max_position_embeddings", None)
            if max_seq_len is None or max_seq_len <= 0:
                raise ValueError("max_seq_len must be a positive integer for static cache.")
            _validate_static_cache_support(decoder)

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        graph, builder = _make_graph(name="decoder")
        op = builder.op

        inputs_embeds = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=[batch, seq_len, config.hidden_size],
        )

        if not static:
            past_seq_len = ir.SymbolicDim("past_sequence_len")
            attention_mask = builder.input(
                "attention_mask",
                dtype=ir.DataType.INT64,
                shape=[batch, "past_seq_len + seq_len"],
            )
        else:
            # Static cache still needs past_seq_len for sliding-window layers
            # that use dynamic cache within the hybrid static/dynamic scheme.
            past_seq_len = ir.SymbolicDim("past_sequence_len")

        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        per_layer_inputs_val: ir.Value | None = None
        per_layer_dim = getattr(config, "hidden_size_per_layer_input", 0)
        if per_layer_dim and not config.split_per_layer_embedding:
            total_per_layer = config.num_hidden_layers * per_layer_dim
            per_layer_inputs_val = builder.input(
                "per_layer_inputs",
                dtype=config.dtype,
                shape=[batch, seq_len, total_per_layer],
            )

        # Vision-block bidirectional attention: the decoder receives the raw
        # ``input_ids`` (alongside ``inputs_embeds``) and derives the block
        # overlay internally. This avoids a separate cross-model
        # ``block_sequence_ids`` tensor, which onnxruntime-genai cannot forward
        # between the embedding and decoder sub-models (it can forward
        # ``input_ids``). Only models with
        # ``use_bidirectional_attention == "vision"`` need it.
        # When split_per_layer_embedding is set, input_ids is also needed for
        # per-layer embedding lookups inside the decoder.
        input_ids_val: ir.Value | None = None
        if config.use_bidirectional_attention == "vision" or (
            per_layer_dim and config.split_per_layer_embedding
        ):
            input_ids_val = builder.input(
                "input_ids",
                dtype=ir.DataType.INT64,
                shape=[batch, seq_len],
            )

        if static:
            past_key_values = _make_gemma4_static_cache_inputs(
                builder,
                config,
                batch,
                max_seq_len,
                past_seq_len,
            )
            attention_mask = None  # Static cache uses position-based attention
        else:
            past_key_values = _make_gemma4_kv_cache_inputs(
                builder,
                config,
                batch,
                past_seq_len,
            )

        logits, present_key_values = decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            per_layer_inputs=per_layer_inputs_val,
            past_key_values=past_key_values,
            input_ids=input_ids_val,
        )

        builder.add_output(logits, "logits")
        if static:
            _register_hybrid_cache_outputs(builder, present_key_values, config)
        else:
            _register_kv_cache_outputs(builder, present_key_values)

        return _make_model(graph)

    def _build_vision(
        self,
        vision: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build vision encoder: pixel_values + pixel_position_ids -> image_features.

        Inputs:
        - ``pixel_values [B, N, 3*P^2]``: pre-patchified image data where
          ``B`` is the number of images and ``N`` is the number of patches
          (padded to ``max_soft_tokens``, typically 280).
        - ``pixel_position_ids [B, N, 2]``: (x, y) patch coordinates.
          Unused patch slots (from images smaller than the maximum) use
          ``(-1, -1)`` as a sentinel value — no explicit mask is needed.

        Output:
        - ``image_features [B*N, text_hidden_size]``: projected vision features
        """
        batch = ir.SymbolicDim("batch")
        num_patches = ir.SymbolicDim("num_patches")
        patch_size = config.vision.patch_size or 16 if config.vision else 16
        pixel_dim = 3 * patch_size * patch_size

        graph, builder = _make_graph(name="vision_encoder")
        op = builder.op

        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[batch, num_patches, pixel_dim],
        )
        pixel_position_ids = builder.input(
            "pixel_position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, num_patches, 2],
        )

        image_features = vision(
            op,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
        )

        builder.add_output(image_features, "image_features")

        declare_component_presence(graph, "image")
        return _make_model(graph)

    def _build_audio(
        self,
        audio: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build audio encoder: input_features + input_features_mask -> audio_features.

        Inputs:
        - ``input_features [batch, time, input_size]``: mel-spectrogram
        - ``input_features_mask [batch, time]``: BOOL mask indicating valid
          mel frames. Must be **contiguous**: all ``True`` entries precede
          all ``False`` entries (right-padded). The HuggingFace
          ``Gemma4AudioFeatureExtractor`` produces this layout by padding
          audio to equal batch lengths and marking padded frames as
          ``False``. The mask is downsampled through conv subsampling
          layers (stride 2 per stage) and used to zero out padded
          positions in Conformer attention.

        Outputs:
        - ``audio_features [num_valid, text_hidden_size]``: encoded tokens with
          padded rows removed in batch-major order
        - ``audio_features_mask [batch, time//4]``: BOOL downsampled mask
          retained for diagnostics and backwards compatibility
        """
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")
        input_size = (config.audio.input_size if config.audio else None) or 128

        graph, builder = _make_graph(name="audio_encoder")
        op = builder.op

        input_features = builder.input(
            "input_features",
            dtype=config.dtype,
            shape=[batch, time, input_size],
        )
        input_features_mask = builder.input(
            "input_features_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, time],
        )

        audio_features, downsampled_mask = audio(
            op,
            input_features,
            input_features_mask=input_features_mask,
        )
        if downsampled_mask is None:
            raise ValueError("Gemma4 audio encoder must return a downsampled validity mask")
        flattened_features = op.Reshape(
            audio_features,
            op.Constant(value_ints=[-1, config.hidden_size]),
        )
        flattened_mask = op.Reshape(
            downsampled_mask,
            op.Constant(value_ints=[-1]),
        )
        flattened_features_f32 = op.Cast(flattened_features, to=ir.DataType.FLOAT)
        audio_features = op.CastLike(
            op.Compress(flattened_features_f32, flattened_mask, axis=0),
            flattened_features,
        )
        builder.add_output(audio_features, "audio_features")
        builder.add_output(downsampled_mask, "audio_features_mask")

        declare_component_presence(graph, "audio")
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build embedding: input_ids + image_features [+ audio_features] -> inputs_embeds."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_image_tokens = ir.SymbolicDim("num_image_tokens")

        graph, builder = _make_graph(name="embedding")
        op = builder.op

        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        image_features = builder.input(
            "image_features",
            dtype=config.dtype,
            shape=[num_image_tokens, config.hidden_size],
        )
        declare_optional_input(
            image_features,
            presence="image",
            absent_shape=[0, config.hidden_size],
        )

        audio_features_val: ir.Value | None = None

        if config.audio is not None:
            num_audio_tokens = ir.SymbolicDim("num_audio_tokens")
            audio_features_val = builder.input(
                "audio_features",
                dtype=config.dtype,
                shape=[num_audio_tokens, config.hidden_size],
            )
            declare_optional_input(
                audio_features_val,
                presence="audio",
                absent_shape=[0, config.hidden_size],
            )

        result = embedding(
            op,
            input_ids=input_ids,
            image_features=image_features,
            audio_features=audio_features_val,
        )

        # ``embedding`` returns a dict of named outputs: always
        # ``inputs_embeds``; optionally ``per_layer_inputs`` (per-layer gating).
        builder.add_output(result["inputs_embeds"], "inputs_embeds")
        if "per_layer_inputs" in result:
            builder.add_output(result["per_layer_inputs"], "per_layer_inputs")
        return _make_model(graph)


class Gemma4UnifiedTask(Gemma4Task):
    """Task for ``gemma4_unified`` (gemma-4-12B) encoder-free multimodal models.

    Reuses :class:`Gemma4Task`'s ``decoder`` and ``embedding`` builders (the
    text decoder and multimodal fusion are identical to gemma4), but overrides
    the vision and audio builders for the encoder-free embedders:

    - **vision** — raw merged pixel patches ``pixel_values [B, N, P^2*3]`` and
      integer patch coordinates ``pixel_position_ids [B, N, 2]`` →
      ``image_features [num_valid_patches, text_hidden]`` (padding patches,
      with position ``-1``, are stripped inside the graph).
    - **audio** — raw waveform-frame features ``input_features [B, T, D_a]``
      and a validity mask ``input_features_mask [B, T]`` →
      ``audio_features [num_valid_frames, text_hidden]`` (padding frames are
      stripped inside the graph; no separate mask output is needed).
    """

    def _build_vision(
        self,
        vision: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build the encoder-free vision embedder.

        Inputs:
        - ``pixel_values [B, N, P^2*3]``: raw merged pixel patches, where
          ``P = patch_size * pooling_kernel_size`` (48 for gemma-4-12B).
        - ``pixel_position_ids [B, N, 2]``: integer (x, y) patch coordinates;
          ``(-1, -1)`` marks padding slots.

        Output:
        - ``image_features [num_valid_patches, text_hidden]``.
        """
        batch = ir.SymbolicDim("batch")
        num_patches = ir.SymbolicDim("num_patches")
        vc = config.vision
        patch_size = (vc.patch_size if vc else None) or 16
        pooling = (vc.pooling_kernel_size if vc else None) or 3
        model_patch_size = patch_size * pooling
        pixel_dim = 3 * model_patch_size * model_patch_size

        graph, builder = _make_graph(name="vision_encoder")
        op = builder.op

        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[batch, num_patches, pixel_dim],
        )
        pixel_position_ids = builder.input(
            "pixel_position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, num_patches, 2],
        )

        image_features = vision(
            op,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
        )
        builder.add_output(image_features, "image_features")
        declare_component_presence(graph, "image")
        return _make_model(graph)

    def _build_audio(
        self,
        audio: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build the encoder-free audio embedder.

        Inputs:
        - ``input_features [B, T, D_a]``: raw waveform-frame features
          (``D_a = audio_embed_dim``, 640 for gemma-4-12B).
        - ``input_features_mask [B, T]``: BOOL mask, ``True`` for valid frames.

        Output:
        - ``audio_features [num_valid_frames, text_hidden]``.
        """
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")
        audio_embed_dim = (
            getattr(config.audio, "hidden_size", None) if config.audio else None
        ) or 640

        graph, builder = _make_graph(name="audio_encoder")
        op = builder.op

        input_features = builder.input(
            "input_features",
            dtype=config.dtype,
            shape=[batch, time, audio_embed_dim],
        )
        input_features_mask = builder.input(
            "input_features_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, time],
        )

        audio_features, _ = audio(
            op,
            input_features,
            input_features_mask=input_features_mask,
        )
        builder.add_output(audio_features, "audio_features")
        declare_component_presence(graph, "audio")
        return _make_model(graph)
