# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-5.2 (``glm_moe_dsa``) MLA + DeepSeek Sparse Attention (DSA) + MoE.

Reference: ``transformers.models.glm_moe_dsa`` (``GlmMoeDsaForCausalLM``),
the authoritative ground truth for this architecture's config derivation
and forward semantics. GLM-5.2 reuses DeepSeek-V3's MLA/MoE building blocks
almost unchanged (identical fused-expert HF weight layout, identical
sigmoid + correction-bias + degenerate-group-of-one top-k routing) and adds
one new mechanism: DeepSeek Sparse Attention (DSA), where a lightweight
per-layer "indexer" selects the ``index_topk`` most relevant key positions
per query token, and only "full" layers run the indexer -- "shared" layers
reuse the nearest preceding "full" layer's selection unchanged.

Two export paths are supported:

- ``config.use_dsa=True`` (default): DSA is exported using the frozen
  ``pkg.nxrt::IndexShare`` custom op (see
  ``docs/architecture/INDEXSHARE_DESIGN.md`` in onnx-genai), which gathers
  only the indexer-selected key/value positions at runtime.
- ``config.use_dsa=False`` (the ``--glm-full-attention`` CLI feature):
  falls back to plain dense MLA -- literally ``DeepSeekV3TextModel``,
  unchanged -- which any ``Attention``-op-supporting runtime (including
  stock ORT) can already execute today.

Multi-token-prediction (MTP) is intentionally **not** exported this cycle:
the reference ``GlmMoeDsaForCausalLM`` implementation has no MTP module at
all and explicitly ignores ``model.layers.<num_hidden_layers>.*`` (MTP)
weights on load via ``_keys_to_ignore_on_load_unexpected``. This exporter
follows the same precedent: MTP weights are dropped with a clear, logged
capability message rather than shipping a from-scratch, unvalidated MTP
graph. See ``mobius.models.qwen35_mtp`` for the generic standalone-MTP-head
pattern a future slice should reuse to add GLM-5.2 MTP support.
"""

from __future__ import annotations

import dataclasses
import logging
import re

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    DeepSeekMLA,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.deepseek import (
    DeepSeekMoEGate,
    DeepSeekV3CausalLMModel,
    DeepSeekV3TextModel,
    _DeepSeekMoEFFN,
    _linear_factory,
)

logger = logging.getLogger(__name__)

# Matches "model.layers.<idx>.<rest>" HF state_dict keys so MTP-layer
# weights (idx >= num_hidden_layers) can be identified for dropping.
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def _indexer_types(config: ArchitectureConfig) -> list[str]:
    """Per-layer full/shared DSA indexer schedule.

    Mirrors HF ``GlmMoeDsaConfig.__post_init__`` exactly: an explicit
    ``indexer_types`` list always wins (and must have one entry per hidden
    layer); otherwise every layer is "full" except that layers are "shared"
    unless ``(i - offset + 1)`` clamped to >= 0 is a multiple of ``freq``.
    Defaults (``freq=1``, ``offset=2``) match upstream and make every layer
    "full" when the checkpoint config omits both fields.
    """
    if config.indexer_types is not None:
        indexer_types = list(config.indexer_types)
        if len(indexer_types) != config.num_hidden_layers:
            raise ValueError(
                "indexer_types must contain exactly one entry per hidden layer "
                f"(expected {config.num_hidden_layers}, got {len(indexer_types)})"
            )
        return indexer_types
    freq = max(config.index_topk_freq or 1, 1)
    offset = config.index_skip_topk_offset if config.index_skip_topk_offset is not None else 2
    return [
        "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
        for i in range(config.num_hidden_layers)
    ]


class GlmMoeDsaIndexer(nn.Module):
    """DeepSeek Sparse Attention (DSA) token indexer for "full" layers.

    Projects a lightweight, single-head key (``wk``) and a per-index-head
    query (``wq_b``, fed from the main attention's ``q_a_layernorm`` output)
    and scores every key position with a ReLU'd, head-weighted dot product.
    Only ``qk_rope_head_dim`` of ``index_head_dim`` is RoPE-rotated -- the
    remainder passes through unrotated -- matching
    ``transformers.models.glm_moe_dsa.modeling_glm_moe_dsa.GlmMoeDsaIndexer``
    exactly. The top ``index_topk`` scored positions (ascending, per
    ``pkg.nxrt::IndexShare``'s strictly-increasing-index requirement) are
    returned for the caller to either consume directly or thread to
    subsequent "shared" layers.
    """

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        assert config.q_lora_rank is not None
        assert config.index_head_dim is not None
        assert config.index_n_heads is not None
        assert config.qk_rope_head_dim is not None
        if linear_class is None:
            linear_class = Linear
        self.head_dim = config.index_head_dim
        self.n_heads = config.index_n_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = int(config.index_topk or 2048)
        self.rope_interleave = config.indexer_rope_interleave
        # Matches HF's ``softmax_scale = self.head_dim**-0.5`` -- note this
        # uses the *indexer's* head_dim (index_head_dim), not the main
        # attention's qk_head_dim.
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = linear_class(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = linear_class(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = linear_class(config.hidden_size, self.n_heads, bias=False)

    def _rope_split(
        self,
        op: OpBuilder,
        x: ir.Value,
        num_heads: int,
        position_embeddings: tuple,
    ) -> ir.Value:
        """RoPE-rotate only the leading ``qk_rope_head_dim`` slice of ``x``.

        ``x`` is ``(B, S, num_heads, head_dim)``. Unlike the main MLA
        attention (where the entire rope slice *is* the rotated portion),
        the indexer's ``index_head_dim`` is wider than
        ``qk_rope_head_dim``: only the first ``qk_rope_head_dim`` columns
        rotate and the remaining ``head_dim - qk_rope_head_dim`` columns
        pass through unchanged, then the two are re-concatenated.
        """
        pass_dim = self.head_dim - self.qk_rope_head_dim
        rot, pas = op.Split(x, [self.qk_rope_head_dim, pass_dim], axis=-1, _outputs=2)
        rot = op.Reshape(rot, [0, 0, -1])
        rot = apply_rotary_pos_emb(
            op,
            rot,
            position_embeddings,
            num_heads=num_heads,
            rotary_embedding_dim=0,
            interleaved=self.rope_interleave,
        )
        rot = op.Reshape(rot, [0, 0, num_heads, self.qk_rope_head_dim])
        return op.Concat(rot, pas, axis=-1)

    def project_key(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple,
    ) -> ir.Value:
        """Compute this token's (rope-split) indexer key: ``(B, S, head_dim)``."""
        key = self.k_norm(op, self.wk(op, hidden_states))
        key = op.Reshape(key, [0, 0, 1, self.head_dim])
        key = self._rope_split(op, key, num_heads=1, position_embeddings=position_embeddings)
        return op.Reshape(key, [0, 0, self.head_dim])

    def select(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        q_resid: ir.Value,
        position_embeddings: tuple,
        all_index_keys: ir.Value,
        attention_bias: ir.Value,
    ) -> ir.Value:
        query = self.wq_b(op, q_resid)
        query = op.Reshape(query, [0, 0, self.n_heads, self.head_dim])
        query = self._rope_split(op, query, self.n_heads, position_embeddings)
        query = op.Cast(query, to=ir.DataType.FLOAT)  # (B, S, H, D)

        keys = op.Cast(all_index_keys, to=ir.DataType.FLOAT)  # (B, T, D)
        keys_t = op.Transpose(keys, perm=[0, 2, 1])  # (B, D, T)
        # A plain 4D-query x 3D-key MatMul broadcasts the batch dims
        # right-aligned, so a (B, S) vs (B,) batch shape silently pairs the
        # query's S axis with the key's B axis whenever batch_size != seq_len.
        # Insert the query's singleton head axis explicitly (matching HF's
        # ``k.transpose(-1, -2).unsqueeze(1)``) so the (B, S, H, D) x
        # (B, 1, D, T) broadcast is unambiguous regardless of B vs S.
        keys_t = op.Unsqueeze(keys_t, [1])  # (B, 1, D, T)
        scores = op.MatMul(query, keys_t)  # (B, S, H, T)
        scores = op.Mul(scores, self.softmax_scale)
        scores = op.Relu(scores)

        weights = self.weights_proj(op, hidden_states)  # (B, S, H)
        weights = op.Cast(weights, to=ir.DataType.FLOAT)
        weights = op.Mul(weights, self.n_heads**-0.5)
        # Head-weighted sum: (B, S, H, T) -> (B, S, T).
        scores = op.ReduceSum(op.Mul(scores, op.Unsqueeze(weights, [3])), [2], keepdims=False)

        # The additive causal mask is sized to the attention key axis, which
        # the native decoder exposes at fixed KV capacity while the indexer
        # key cache grows with the logical prefix. Slice the mask down to
        # the score key length so the add stays broadcast-compatible under
        # both the logical (ORT) and fixed-capacity (native decode) shapes.
        key_length = op.Shape(scores, start=2, end=3)
        causal = op.Squeeze(attention_bias, [1])
        causal = op.Slice(
            causal, op.Constant(value_ints=[0]), key_length, op.Constant(value_ints=[2])
        )
        scores = op.Add(scores, op.Cast(causal, to=ir.DataType.FLOAT))

        k = op.Min(key_length, op.Constant(value_ints=[self.index_topk]))
        _, indices = op.TopK(scores, k, axis=-1, largest=1, sorted=0, _outputs=2)
        # ``pkg.nxrt::IndexShare`` requires the selected key positions to be
        # strictly increasing per row. TopK returns them in score order, so
        # sort the chosen positions ascending. The runtime TopK kernel is
        # f32-only, so round-trip through float before restoring Int64.
        indices = op.Cast(indices, to=ir.DataType.FLOAT)
        indices, _ = op.TopK(indices, k, axis=-1, largest=0, sorted=1, _outputs=2)
        return op.Cast(indices, to=ir.DataType.INT64)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        q_resid: ir.Value,
        position_embeddings: tuple,
        past_index_key: ir.Value | None,
        attention_bias: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        current_index_key = self.project_key(op, hidden_states, position_embeddings)
        all_index_keys = (
            current_index_key
            if past_index_key is None
            else op.Concat(past_index_key, current_index_key, axis=1)
        )
        indices = self.select(
            op, hidden_states, q_resid, position_embeddings, all_index_keys, attention_bias
        )
        return all_index_keys, indices


class GlmMoeDsaAttention(DeepSeekMLA):
    """MLA attention with a packed indexer-key cache and IndexShare output.

    Reuses ``DeepSeekMLA``'s exact Q/KV projection and RoPE math (duplicated
    here rather than factored out, since ``DeepSeekMLA.forward`` does not
    expose an overridable "run attention" seam); the only behavioral
    difference is the final attention call, which selects the
    indexer-chosen key/value positions via ``pkg.nxrt::IndexShare`` instead
    of a dense ``Attention`` op over the whole KV cache.

    The indexer's own key cache is packed into the *same* present-KV tuple
    as the main attention (index-key columns appended after the main key
    columns) so the exported graph keeps the usual 2-tensor present-KV
    convention per layer instead of doubling KV cache tensors.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        indexer_type: str,
        linear_class: type | None = None,
    ):
        super().__init__(config, linear_class=linear_class, split_kv_b=True)
        self.indexer_type = indexer_type
        self.dtype = config.dtype
        self.main_key_dim = self.num_heads * self.qk_head_dim
        self.main_value_dim = self.num_heads * self.v_head_dim
        self.index_head_dim = int(config.index_head_dim or 0)
        if indexer_type == "full":
            self.indexer = GlmMoeDsaIndexer(config, linear_class)

    def _unpack_past(self, op: OpBuilder, past_key_value: tuple):
        packed_key, packed_value = past_key_value
        key_tokens = op.Squeeze(packed_key, [1])
        main_key = op.Slice(key_tokens, [0], [self.main_key_dim], [2])
        main_key = op.Transpose(
            op.Reshape(main_key, [0, 0, self.num_heads, self.qk_head_dim]),
            perm=[0, 2, 1, 3],
        )
        value_tokens = op.Squeeze(packed_value, [1])
        main_value = op.Transpose(
            op.Reshape(value_tokens, [0, 0, self.num_heads, self.v_head_dim]),
            perm=[0, 2, 1, 3],
        )
        index_key = None
        if self.indexer_type == "full":
            index_key = op.Slice(
                key_tokens,
                [self.main_key_dim],
                [self.main_key_dim + self.index_head_dim],
                [2],
            )
        return (main_key, main_value), index_key

    def _pack_present(
        self,
        op: OpBuilder,
        present: tuple,
        index_keys: ir.Value | None,
    ) -> tuple:
        key = op.Reshape(
            op.Transpose(present[0], perm=[0, 2, 1, 3]), [0, 0, self.main_key_dim]
        )
        if index_keys is not None:
            key = op.Concat(key, index_keys, axis=-1)
        value = op.Reshape(
            op.Transpose(present[1], perm=[0, 2, 1, 3]),
            [0, 0, self.main_value_dim],
        )
        return op.Unsqueeze(key, [1]), op.Unsqueeze(value, [1])

    def _index_share_attention(
        self,
        op: OpBuilder,
        q: ir.Value,
        key: ir.Value,
        value: ir.Value,
        main_past: tuple | None,
        topk_indices: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """Device-resident selected-token attention via ``pkg.nxrt::IndexShare``.

        Replaces the dense additive-sparse-mask + ``Attention`` lowering
        with the frozen ``IndexShare`` op, which gathers only the indexer's
        selected key positions. Causality is carried entirely by the
        (already causal) ``topk_indices``, so no dense additive mask island
        is emitted here.

        ``IndexShare`` requires a homogeneous head size across query/key/
        value. MLA's value head (``v_head_dim``) is narrower than the
        query/key head (``qk_head_dim``), so the value is zero-padded up to
        ``qk_head_dim`` for the kernel and the padding columns are sliced
        back off the output. The returned present key/value stay in their
        native (unpadded) BNSH layout so the packed KV cache is unchanged.

        The frozen ``IndexShare`` schema pins ``query``/``key``/``value`` (and
        its ``output``) to f32 regardless of the model's compute dtype (see
        ``docs/architecture/INDEXSHARE_DESIGN.md``), so the query/key/padded
        value are cast to float right before the call and the result is cast
        back to ``self.dtype`` before returning -- the returned present
        key/value stay in their native dtype for the packed KV cache.
        """
        q_bnsh = op.Transpose(
            op.Reshape(q, [0, 0, self.num_heads, self.qk_head_dim]), perm=[0, 2, 1, 3]
        )
        cur_key = op.Transpose(
            op.Reshape(key, [0, 0, self.num_heads, self.qk_head_dim]), perm=[0, 2, 1, 3]
        )
        cur_value = op.Transpose(
            op.Reshape(value, [0, 0, self.num_heads, self.v_head_dim]), perm=[0, 2, 1, 3]
        )
        present_key = cur_key
        present_value = cur_value
        if main_past is not None:
            present_key = op.Concat(main_past[0], cur_key, axis=2)
            present_value = op.Concat(main_past[1], cur_value, axis=2)

        value_pad = self.qk_head_dim - self.v_head_dim
        padded_value = present_value
        if value_pad > 0:
            # Zero-pad the value head up to ``qk_head_dim`` without ``Pad``
            # (which the CUDA EP has no handler for). Build a device-resident
            # zero block of the value's own dtype and concatenate it onto
            # the head axis.
            pad_shape = op.Concat(
                op.Shape(present_value, start=0, end=3),
                op.Constant(value_ints=[value_pad]),
                axis=0,
            )
            zeros = op.Expand(op.Cast(op.Constant(value_float=0.0), to=self.dtype), pad_shape)
            padded_value = op.Concat(present_value, zeros, axis=3)

        # The frozen IndexShare schema pins query/key/value (and its output)
        # to f32 regardless of the model's compute dtype -- cast the inputs
        # here and cast the output back below, leaving the returned
        # present_key/present_value (packed KV cache) in their native dtype.
        needs_cast = self.dtype != ir.DataType.FLOAT
        q_f32 = op.Cast(q_bnsh, to=ir.DataType.FLOAT) if needs_cast else q_bnsh
        key_f32 = op.Cast(present_key, to=ir.DataType.FLOAT) if needs_cast else present_key
        value_f32 = op.Cast(padded_value, to=ir.DataType.FLOAT) if needs_cast else padded_value

        selected_indices = op.Unsqueeze(topk_indices, [1])
        # pkg.nxrt is a custom onnx-genai runtime domain; the ONNX checker
        # requires every domain used by a node to have a matching
        # opset_imports entry (not automatically added by the op call),
        # same as BlockQuantizedMatMul in _quantized_linear.py.
        op.builder.graph.opset_imports["pkg.nxrt"] = 1
        attn_output = op.IndexShare(
            q_f32,
            key_f32,
            value_f32,
            None,
            None,
            selected_indices,
            num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=self.scaling,
            _domain="pkg.nxrt",
            _outputs=1,
        )
        if needs_cast:
            attn_output = op.Cast(attn_output, to=self.dtype)
        if value_pad > 0:
            attn_output = op.Slice(attn_output, [0], [self.v_head_dim], [3])
        attn_output = op.Reshape(
            op.Transpose(attn_output, perm=[0, 2, 1, 3]),
            [0, 0, self.num_heads * self.v_head_dim],
        )
        return attn_output, present_key, present_value

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
        shared_topk_indices: ir.Value | None = None,
    ):
        q_resid = self.q_a_layernorm(op, self.q_a_proj(op, hidden_states))
        q = self.q_b_proj(op, q_resid)
        q = op.Reshape(q, [0, 0, self.num_heads, self.qk_head_dim])
        q_nope, q_rope = op.Split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1, _outputs=2
        )
        q_rope = op.Reshape(q_rope, [0, 0, -1])
        q_rope = apply_rotary_pos_emb(
            op,
            q_rope,
            position_embeddings,
            num_heads=self.num_heads,
            interleaved=self._rope_interleave,
        )
        q = op.Concat(
            op.Reshape(q_nope, [0, 0, self.num_heads, self.qk_nope_head_dim]),
            op.Reshape(q_rope, [0, 0, self.num_heads, self.qk_rope_head_dim]),
            axis=-1,
        )
        q = op.Reshape(q, [0, 0, self.num_heads * self.qk_head_dim])

        compressed_kv = self.kv_a_proj_with_mqa(op, hidden_states)
        kv_latent, k_rope = op.Split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1, _outputs=2
        )
        kv_latent = self.kv_a_layernorm(op, kv_latent)
        k_nope = op.Reshape(
            self.k_b_proj(op, kv_latent),
            [0, 0, self.num_heads, self.qk_nope_head_dim],
        )
        value = op.Reshape(
            self.v_b_proj(op, kv_latent),
            [0, 0, self.num_heads, self.v_head_dim],
        )
        value = op.Reshape(value, [0, 0, -1])
        k_rope = apply_rotary_pos_emb(
            op, k_rope, position_embeddings, num_heads=1, interleaved=self._rope_interleave
        )
        k_rope = op.Tile(k_rope, [1, 1, self.num_heads])
        key = op.Concat(
            op.Reshape(k_nope, [0, 0, self.num_heads, self.qk_nope_head_dim]),
            op.Reshape(k_rope, [0, 0, self.num_heads, self.qk_rope_head_dim]),
            axis=-1,
        )
        key = op.Reshape(key, [0, 0, self.num_heads * self.qk_head_dim])

        main_past = None
        past_index_key = None
        if past_key_value is not None:
            main_past, past_index_key = self._unpack_past(op, past_key_value)

        all_index_keys = None
        topk_indices = shared_topk_indices
        if self.indexer_type == "full":
            all_index_keys, topk_indices = self.indexer(
                op, hidden_states, q_resid, position_embeddings, past_index_key, attention_bias
            )
        if topk_indices is None:
            raise ValueError(
                "GLM-5.2 'shared' DSA layers require top-k indices threaded "
                "from a preceding 'full' indexer layer"
            )

        attn_output, present_key, present_value = self._index_share_attention(
            op, q, key, value, main_past, topk_indices
        )
        attn_output = self.o_proj(op, attn_output)
        present = self._pack_present(op, (present_key, present_value), all_index_keys)
        return attn_output, present, topk_indices


class GlmMoeDsaDecoderLayer(nn.Module):
    """DSA decoder layer: MLA+IndexShare attention plus MoE/dense FFN."""

    def __init__(
        self,
        config: ArchitectureConfig,
        indexer_type: str,
        is_moe: bool,
        linear_class: type | None = None,
    ):
        super().__init__()
        self.self_attn = GlmMoeDsaAttention(config, indexer_type, linear_class)
        if is_moe:
            self.mlp = _DeepSeekMoEFFN(
                config, DeepSeekMoEGate(config), linear_class=linear_class
            )
        else:
            self.mlp = MLP(config, linear_class=linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
        shared_topk_indices: ir.Value | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present, topk_indices = self.self_attn(
            op,
            hidden_states,
            attention_bias,
            position_embeddings,
            past_key_value,
            shared_topk_indices,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states, present, topk_indices


class GlmMoeDsaTextModel(nn.Module):
    """GLM-5.2 text backbone: embed -> N x (MLA+DSA/MoE) layers -> norm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        linear_class = _linear_factory(config)
        indexer_types = _indexer_types(config)
        self.layers = nn.ModuleList(
            [
                GlmMoeDsaDecoderLayer(
                    config,
                    indexer_types[i],
                    is_moe=i >= config.first_k_dense_replace,
                    linear_class=linear_class,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE applies only to the qk_rope_head_dim portion of Q/K (and, via
        # GlmMoeDsaIndexer._rope_split, the same-sized leading slice of the
        # indexer's wider index_head_dim) -- never the full head_dim. Build
        # the shared cos/sin cache from qk_rope_head_dim explicitly rather
        # than relying on the upstream HF config's own self-correcting
        # ``head_dim = qk_rope_head_dim`` (``GlmMoeDsaConfig.__post_init__``):
        # a raw config.json loaded without that Python class in the loop
        # would otherwise leave the un-fixed (larger) raw ``head_dim``, as
        # was the case for DeepSeek-V2/V3 (see ``DeepSeekV3TextModel``).
        if config.qk_rope_head_dim is not None and config.qk_rope_head_dim > 0:
            rope_config = dataclasses.replace(config, head_dim=config.qk_rope_head_dim)
        else:
            rope_config = config
        self.rotary_emb = initialize_rope(rope_config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
            attention_mask=attention_mask,
            dtype=self.config.dtype,
        )

        presents = []
        past_kvs = past_key_values or [None] * len(self.layers)
        shared_topk_indices = None
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present, shared_topk_indices = layer(
                op,
                hidden_states,
                attention_bias,
                position_embeddings,
                past_kv,
                shared_topk_indices,
            )
            presents.append(present)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, presents


class GlmMoeDsaCausalLMModel(DeepSeekV3CausalLMModel):
    """GLM-5.2 (``zai-org/GLM-5.2``) causal LM: MLA + DSA + MoE.

    model_type: glm_moe_dsa

    DSA (top-k sparse attention via the frozen ``pkg.nxrt::IndexShare`` op)
    is the default export path (``config.use_dsa=True``, the default).
    Setting ``config.use_dsa=False`` (the ``--glm-full-attention`` CLI
    feature) instead builds the plain dense-MLA ``DeepSeekV3TextModel``
    unchanged, which any ``Attention``-op-supporting runtime -- including
    stock ORT -- can already execute. ``preprocess_weights`` drops every DSA
    indexer weight in that case since nothing in the dense graph consumes it.

    Multi-token-prediction (MTP) is out of scope this cycle: see the module
    docstring. ``preprocess_weights`` drops ``model.layers.N.*`` weights for
    ``N >= num_hidden_layers`` (MTP) with a logged capability message rather
    than silently or accidentally feeding them into the base decoder.
    """

    default_task: str = "text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        if config.indexer_types is None:
            config = dataclasses.replace(config, indexer_types=_indexer_types(config))
        nn.Module.__init__(self)
        self.config = config
        if config.export_paged_attention and config.use_dsa:
            # Feature-on must error rather than silently exporting the DSA graph.
            from mobius.components._paged_mla import paged_attention_rejection

            raise ValueError(
                paged_attention_rejection(config)
                or "PagedAttention export requires dense MLA (--glm-full-attention)."
            )
        self.model = (
            GlmMoeDsaTextModel(config) if config.use_dsa else DeepSeekV3TextModel(config)
        )
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = dict(state_dict)
        kv_b_suffix = ".self_attn.kv_b_proj.weight"
        for key in tuple(state_dict):
            if not key.endswith(kv_b_suffix):
                continue
            tensor = state_dict.pop(key)
            expected_rows = self.config.num_attention_heads * (
                self.config.qk_nope_head_dim + self.config.v_head_dim
            )
            if tensor.dim() != 2 or tensor.shape != (
                expected_rows,
                self.config.kv_lora_rank,
            ):
                raise ValueError(
                    f"GLM-5.2 fused KV-B tensor {key!r} must have shape "
                    f"({expected_rows}, {self.config.kv_lora_rank}), got "
                    f"{tuple(tensor.shape)}"
                )
            per_head = tensor.reshape(
                self.config.num_attention_heads,
                self.config.qk_nope_head_dim + self.config.v_head_dim,
                self.config.kv_lora_rank,
            )
            key_rows, value_rows = per_head.split(
                [self.config.qk_nope_head_dim, self.config.v_head_dim],
                dim=1,
            )
            prefix = key[: -len("kv_b_proj.weight")]
            state_dict[f"{prefix}k_b_proj.weight"] = key_rows.reshape(
                -1, self.config.kv_lora_rank
            )
            state_dict[f"{prefix}v_b_proj.weight"] = value_rows.reshape(
                -1, self.config.kv_lora_rank
            )

        mtp_keys = []
        for key in state_dict:
            match = _LAYER_RE.match(key)
            if match is not None and int(match.group(1)) >= self.config.num_hidden_layers:
                mtp_keys.append(key)
        if mtp_keys:
            logger.warning(
                "Dropping %d GLM-5.2 multi-token-prediction (MTP) weight(s) "
                "under model.layers.%d.* and above: MTP export is out of "
                "scope this cycle. The reference transformers "
                "GlmMoeDsaForCausalLM implementation has no MTP module "
                "either and ignores these same keys on load "
                "(_keys_to_ignore_on_load_unexpected).",
                len(mtp_keys),
                self.config.num_hidden_layers,
            )
        filtered = {k: v for k, v in state_dict.items() if k not in mtp_keys}

        if not self.config.use_dsa:
            indexer_keys = [k for k in filtered if ".self_attn.indexer." in k]
            if indexer_keys:
                logger.info(
                    "Dropping %d GLM-5.2 DSA indexer weight(s): "
                    "config.use_dsa=False (--glm-full-attention) exports "
                    "plain dense MLA, which does not consume the indexer.",
                    len(indexer_keys),
                )
            filtered = {k: v for k, v in filtered.items() if k not in indexer_keys}

        return DeepSeekV3CausalLMModel.preprocess_weights(self, filtered)

    def dsa_kv_cache_specs(self) -> list[tuple[int, int]]:
        """Per-layer ``(key_head_dim, value_head_dim)`` for the packed DSA cache.

        Only meaningful when ``config.use_dsa`` (``self.model`` is a
        :class:`GlmMoeDsaTextModel`): ``GlmMoeDsaAttention._pack_present``
        packs the indexer's own key cache into the *same* present-KV tensor
        as the main attention (indexer columns appended after the main key
        columns), so the key head_dim varies per layer -- "full" indexer
        layers add ``index_head_dim`` extra columns, "shared" layers don't
        -- unlike the uniform per-layer shape every other registered task
        assumes. Reads the exact dims off the already-constructed attention
        modules (rather than recomputing from config) so this can never
        drift from what ``_pack_present``/``_unpack_past`` actually produce.
        Consumed by :class:`mobius.tasks._glm_moe_dsa.GlmMoeDsaTask`.
        """
        return [
            (
                layer.self_attn.main_key_dim
                + (
                    layer.self_attn.index_head_dim
                    if layer.self_attn.indexer_type == "full"
                    else 0
                ),
                layer.self_attn.main_value_dim,
            )
            for layer in self.model.layers
        ]
