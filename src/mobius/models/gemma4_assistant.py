# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mobius port of the Gemma4-Assistant speculative-decoding draft model.

The Gemma4-Assistant family (``google/gemma-4-{E2B,E4B,12B,26B,31B}-it-assistant``)
is a small Gemma4-style decoder used as the drafter in HuggingFace's
``assistant_model=`` speculative decoding API.  Architecturally::

    pre_projection (2 * backbone_hidden → assistant_hidden, no bias)
    →  N decoder layers, **every layer borrows K and V from the target's
       shared KV buffer for the matching layer_type** (full vs sliding)
    →  final RMSNorm
    →  (logits, projected_state):
         logits           = lm_head(last_hidden_state)              # [B, q, vocab]
         projected_state  = post_projection(last_hidden_state)      # [B, q, backbone]

Where:
- ``inputs_embeds`` is fed in by the target; the size is
  ``2 * backbone_hidden_size`` because the target concatenates the
  previous and current shared hidden states for the new draft position.
- ``shared_kv`` arrives as separate graph inputs per layer type
  (``shared_kv.full_attention.{key,value}`` and / or
  ``shared_kv.sliding_attention.{key,value}``) in
  ``[B, num_kv_heads, kv_len, head_dim]`` (BNSH) layout — i.e. the
  target's KV cache layout, already RoPE'd.
- The drafter has no KV cache of its own; every speculative step
  recomputes its full attention.  In the standard speculative-decoding
  loop ``q_len = 1`` (one draft token per call), in which case the
  bidirectional masks from the upstream HF implementation are no-ops
  (per upstream comment: "There is no difference for the edge case of
  ``q_len == 1`` as it acts as full attention no matter what").  Our
  implementation uses ``is_causal=0`` + no attention bias, which is
  plain full attention and stays correct for any ``q_len`` that does
  not require causality among Q positions.

Limitations of this implementation:
- ``q_len > 1`` cases that need causal masking among Q positions would
  require adding back the bidirectional masks.  Not exercised by the
  standard HF assisted-generation loop.
- Padded prompts (attention_mask with non-trivial padding) are not
  exercised; the implementation assumes the assistant runs with
  batch=1 and no padding (the typical spec-decoding setup).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import Gemma4AssistantConfig
from mobius.components import Linear, RMSNorm, initialize_rope
from mobius.components._gemma4_assistant import (
    Gemma4AssistantDecoderLayer,
    Gemma4AssistantMaskedEmbedder,
)

if TYPE_CHECKING:
    import onnx_ir as ir


class Gemma4AssistantCausalLMModel(nn.Module):
    """Gemma4-Assistant speculative-decoding draft model.

    See module docstring for the architecture overview and limitations.

    Weight module layout (mirrors HF ``Gemma4AssistantForCausalLM``):
    - ``self.pre_projection``  Linear[2*backbone → hidden] (no bias)
    - ``self.model.layers.{i}.{...}``  one Gemma4AssistantDecoderLayer per layer
    - ``self.model.norm``      final RMSNorm on hidden
    - ``self.lm_head``         Linear[hidden → vocab] (no bias; tied to embed_tokens upstream)
    - ``self.post_projection`` Linear[hidden → backbone] (no bias)
    """

    config_class: type = Gemma4AssistantConfig
    default_task: str = "gemma4-assistant"
    category: str = "Text Generation"

    def __init__(self, config: Gemma4AssistantConfig):
        super().__init__()
        self.config = config

        layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"Gemma4AssistantConfig.layer_types length ({len(layer_types)}) "
                f"must match num_hidden_layers ({config.num_hidden_layers})"
            )
        self.layer_types = layer_types

        # pre_projection consumes the target's concatenated hidden state.
        self.pre_projection = Linear(
            2 * config.backbone_hidden_size, config.hidden_size, bias=False
        )

        # Build the layer stack inside a sub-module ``self.model`` so the
        # state-dict keys match HF (model.layers.{i}.*, model.norm.weight).
        self.model = _Gemma4AssistantTextModel(config, layer_types)

        # Output side.
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_projection = Linear(
            config.hidden_size, config.backbone_hidden_size, bias=False
        )
        # Optional centroid-routed sparse LM head (used by the released
        # E2B-it-assistant checkpoint, which has use_ordered_embeddings=True).
        self.masked_embedding: Gemma4AssistantMaskedEmbedder | None = (
            Gemma4AssistantMaskedEmbedder(config) if config.use_ordered_embeddings else None
        )

    def preprocess_weights(self, state_dict):
        """Bridge HF state-dict naming to our module layout.

        The HF ``Gemma4AssistantForCausalLM`` wraps an inner ``Gemma4TextModel``
        (via ``AutoModel.from_config(text_config)``) which has its own
        ``embed_tokens`` table.  With ``tie_word_embeddings=True`` the HF
        checkpoint stores ONLY ``model.embed_tokens.weight`` (and not
        ``lm_head.weight``) — the LM head is reconstructed via the
        ``_tied_weights_keys`` aliasing.

        Our mobius assistant has no ``model.embed_tokens`` (it consumes
        ``inputs_embeds`` directly from the target), so we redirect the
        embedding weight to feed our ``lm_head``.  After the alias, the
        original ``model.embed_tokens.weight`` key is removed because no
        mobius module consumes it.

        When ``use_ordered_embeddings=False`` the ``masked_embedding.*``
        keys (if present in the state dict) are dropped because no mobius
        module would consume them.
        """
        if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        # The HF Gemma4TextModel's embed_tokens table is never consumed by
        # the mobius assistant — drop it after the alias.
        state_dict.pop("model.embed_tokens.weight", None)
        # When ordered-embeddings isn't built, also drop the unused
        # masked_embedding.* keys so the loader doesn't warn about them.
        if not self.config.use_ordered_embeddings:
            for unused in (
                "masked_embedding.centroids.weight",
                "masked_embedding.token_ordering",
            ):
                state_dict.pop(unused, None)
        return state_dict

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        position_ids: ir.Value,
        shared_kv: dict[str, tuple[ir.Value, ir.Value]],
        attention_mask: ir.Value | None = None,
    ) -> tuple[ir.Value, ir.Value]:
        """Build the Gemma4-Assistant ONNX graph.

        Args:
            op: onnxscript OpBuilder.
            inputs_embeds: ``[B, q_len, 2 * backbone_hidden_size]``.
            position_ids: ``[B, q_len]`` INT64.  Drives the RoPE on Q in the
                non-GQA path; unused under GQA fusion (positions are derived
                internally from seqlens_k).
            shared_kv: dict keyed by layer type (``"full_attention"``,
                ``"sliding_attention"``) → ``(key, value)`` ir.Values in
                BNSH layout ``[B, num_kv_heads, kv_len, head_dim]``.
            attention_mask: ``[B, kv_len]`` INT (0/1).  REQUIRED for the GQA
                fusion path (used to compute ``seqlens_k`` and
                ``total_seq_len``); ignored by the non-GQA fallback path.

        Returns:
            ``(logits, projected_state)``:
              - ``logits``: ``[B, q_len, vocab_size]``
              - ``projected_state``: ``[B, q_len, backbone_hidden_size]``
        """
        # pre_projection: 2*backbone → hidden.
        hidden_states = self.pre_projection(op, inputs_embeds)

        # Run the layer stack with external K/V.
        hidden_states = self.model(
            op,
            hidden_states=hidden_states,
            position_ids=position_ids,
            shared_kv=shared_kv,
            attention_mask=attention_mask,
        )

        # Heads.
        if self.masked_embedding is not None:
            # Centroid-routed sparse LM head: routes top_k centroids per
            # position, gathers their lm_head rows, computes dot products
            # with hidden_states, scatters into a vocab-sized buffer.
            #
            # We pass ``self.lm_head.weight`` straight into the Gather op
            # rather than calling ``self.lm_head.forward`` (which would
            # do a full dense MatMul we don't need).  Module.__call__
            # only realizes the parameters of the module being called
            # — see onnxscript/nn/_module.py:80-81 — and the qualified
            # initializer name is built from the module-stack context, so
            # we manually push the ``lm_head`` module here so the
            # registered initializer is named ``lm_head.weight`` to match
            # the HF state-dict key.
            op.builder.push_module("lm_head", "Linear")
            try:
                self.lm_head.weight._realize(op.builder)
            finally:
                op.builder.pop_module()
            logits = self.masked_embedding(op, hidden_states, self.lm_head.weight)
        else:
            logits = self.lm_head(op, hidden_states)
        projected_state = self.post_projection(op, hidden_states)
        return logits, projected_state


class _Gemma4AssistantTextModel(nn.Module):
    """The inner Gemma4-Assistant layer stack.

    Kept as a sub-module of :class:`Gemma4AssistantCausalLMModel` so that
    weight names match the HF state-dict layout (``model.layers.{i}.*``,
    ``model.norm.weight``).
    """

    def __init__(self, config: Gemma4AssistantConfig, layer_types: list[str]):
        super().__init__()
        self.layer_types = layer_types

        # Two RoPE flavours, exactly matching how Gemma4TextModel sets them up
        # (see mobius/models/gemma4.py:1515-1541).
        local_config = dataclasses.replace(
            config,
            rope_type="default",
            rope_scaling=None,
            partial_rotary_factor=1.0,
        )
        global_head_dim = config.global_head_dim or config.head_dim
        global_config = dataclasses.replace(
            config,
            head_dim=global_head_dim,
            rope_theta=config.global_rope_theta,
            partial_rotary_factor=config.global_partial_rotary_factor,
            rope_type="proportional",
            rope_scaling=None,
            sliding_window=None,
        )
        self.rotary_emb_local = initialize_rope(local_config)
        self.rotary_emb_global = initialize_rope(global_config)

        # Per-layer rotary_embedding_dim: matches Gemma4TextModel — both
        # DefaultRope (local, full rotation) and ProportionalRope (global,
        # zero-padded partial) handle partial rotation inside their cos/sin
        # caches, so the ONNX RotaryEmbedding op gets rotary_embedding_dim=0
        # (full head_dim) in both cases.
        self.layers = nn.ModuleList(
            [
                Gemma4AssistantDecoderLayer(
                    config, layer_type=layer_types[i], rotary_embedding_dim=0
                )
                for i in range(len(layer_types))
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_ids: ir.Value,
        shared_kv: dict[str, tuple[ir.Value, ir.Value]],
        attention_mask: ir.Value | None = None,
    ) -> ir.Value:
        import onnx_ir as ir_mod

        from mobius._build_context import ep_capabilities, get_build_dtype
        from mobius.components._attention import GQAContext

        caps = ep_capabilities()
        dtype = get_build_dtype()
        use_gqa = (
            attention_mask is not None
            and dtype in caps.gqa_dtypes
            and caps.supports_fused_rope
        )

        if use_gqa:
            # Materialize RoPE caches as graph initializers (GQA references
            # them directly).  Calling .forward() on the RoPE module is what
            # registers the cos/sin Parameter — without this they would be
            # absent from the graph.
            _ = self.rotary_emb_local(op, position_ids)
            _ = self.rotary_emb_global(op, position_ids)

            one_i32 = op.Constant(value_int=1)
            seqlens_k = op.Cast(
                op.Sub(
                    op.ReduceSum(attention_mask, [1], keepdims=0),
                    one_i32,
                ),
                to=ir_mod.DataType.INT32,
            )
            total_seq_len = op.Cast(
                op.Gather(op.Shape(attention_mask), 1),
                to=ir_mod.DataType.INT32,
            )
            gqa_ctx_by_type: dict[str, GQAContext] = {
                "sliding_attention": GQAContext(
                    seqlens_k=seqlens_k,
                    total_seq_len=total_seq_len,
                    cos_cache=self.rotary_emb_local.cos_cache,
                    sin_cache=self.rotary_emb_local.sin_cache,
                    local_window_size=(
                        getattr(self.layers[0].self_attn, "sliding_window", -1) or -1
                    ),
                ),
                "full_attention": GQAContext(
                    seqlens_k=seqlens_k,
                    total_seq_len=total_seq_len,
                    cos_cache=self.rotary_emb_global.cos_cache,
                    sin_cache=self.rotary_emb_global.sin_cache,
                ),
            }
            for layer in self.layers:
                shared_k, shared_v = shared_kv[layer.layer_type]
                hidden_states = layer(
                    op,
                    hidden_states=hidden_states,
                    shared_key=shared_k,
                    shared_value=shared_v,
                    position_embeddings=None,
                    gqa_ctx=gqa_ctx_by_type[layer.layer_type],
                )
            return self.norm(op, hidden_states)

        # Fallback (generic Attention) path.
        pos_emb_local = self.rotary_emb_local(op, position_ids)
        pos_emb_global = self.rotary_emb_global(op, position_ids)
        pos_emb_by_type = {
            "sliding_attention": pos_emb_local,
            "full_attention": pos_emb_global,
        }
        for layer in self.layers:
            shared_k, shared_v = shared_kv[layer.layer_type]
            hidden_states = layer(
                op,
                hidden_states=hidden_states,
                shared_key=shared_k,
                shared_value=shared_v,
                position_embeddings=pos_emb_by_type[layer.layer_type],
                gqa_ctx=None,
            )
        return self.norm(op, hidden_states)
