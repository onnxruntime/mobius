# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base causal language model for standard decoder-only transformers.

Provides TextModel (embedding + decoder layers + norm) and CausalLMModel
(TextModel + LM head). Directly used by Llama, Qwen2, Mistral, and other
architectures that follow the standard GQA + RoPE pattern.

Replicates HuggingFace's LlamaForCausalLM / MistralForCausalLM /
Qwen2ForCausalLM structure.
"""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities, get_build_dtype
from mobius._configs import ArchitectureConfig, CausalLMConfig
from mobius._flags import flags
from mobius._weight_utils import (
    preprocess_awq_weights,
    preprocess_gptq_weights,
    preprocess_olive_weights,
    tie_word_embeddings,
)
from mobius.components import (
    DecoderLayer,
    Embedding,
    FusedGateUpMLP,
    LayerNorm,
    Linear,
    QuantizedEmbedding,
    RMSNorm,
    TiedQuantizedLMHead,
    create_padding_mask,
    create_static_cache_attention_bias,
    initialize_rope,
    make_quantized_linear_factory,
)
from mobius.components._attention import GQAContext, StaticCacheState
from mobius.components._rotary_embedding import BaseRope, _MRopeBase


class TextModel(nn.Module):
    """Base text model with embedding, decoder layers, and final norm."""

    def __init__(self, config: ArchitectureConfig, mlp_class: type | None = None):
        super().__init__()
        self._dtype = config.dtype

        # If the config has quantization, swap Linear for QuantizedLinear
        # in all decoder layer projections (Attention Q/K/V/O + MLP).
        linear_class = None
        qc = getattr(config, "quantization", None)
        if qc is not None and qc.quant_method != "none":
            zp_dtype = (
                config.dtype if getattr(qc, "float_zero_point", False) else ir.DataType.UINT8
            )
            linear_class = make_quantized_linear_factory(
                bits=qc.bits,
                block_size=qc.group_size,
                has_zero_point=not qc.sym,
                zero_point_dtype=zp_dtype,
            )

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        if qc is not None and getattr(qc, "quantize_embeddings", False):
            # Olive RTN (embeds: true) quantizes the embedding table; look it
            # up with GatherBlockQuantized instead of a plain Gather.
            self.embed_tokens = QuantizedEmbedding(
                config.vocab_size,
                config.hidden_size,
                bits=qc.bits,
                block_size=qc.group_size,
                has_zero_point=not qc.sym,
                padding_idx=config.pad_token_id,
            )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, linear_class=linear_class, mlp_class=mlp_class)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

        # Sliding-window models declare a local-attention span; it drives the
        # optional static-cache float bias (flags.static_cache_bias). Standard
        # full-attention models leave this None, so the bias path is a no-op
        # for them even when the flag is set.
        self._sliding_window: int | None = getattr(config, "sliding_window", None)

    def _maybe_static_cache_bias(
        self,
        op: OpBuilder,
        seq_len_source: ir.Value,
        past_key_values: list | None,
    ) -> ir.Value | None:
        """Optionally build the static-cache float additive attention bias.

        Returns ``None`` (maskless ``is_causal=1`` default) unless ALL hold:
          * ``flags.static_cache_bias`` is set, AND
          * the model declares a bias need (``self._sliding_window`` is set), AND
          * the cache is the opset-24 external cache (``StaticCacheState``).

        When emitted, the bias is a ``(B, 1, S_q, max_seq_len)`` additive mask
        keyed on absolute query positions with KV validity
        ``slot < nonpad_kv_seqlen``; ``_apply_attention`` then pairs it with
        ``is_causal=0``. The ``write_indices`` / ``nonpad_kv_seqlen`` graph
        inputs are shared across all layers, so the first layer's cache state
        carries them.

        Args:
            seq_len_source: An always-present ``[B, S_q, ...]`` tensor (e.g.
                ``hidden_states``) whose dim 1 is the query length ``S_q``. Using
                this instead of ``input_ids`` keeps the bias enabled for
                ``inputs_embeds``-driven forwards (where ``input_ids`` is None).
        """
        if not flags.static_cache_bias or self._sliding_window is None:
            return None
        if not past_key_values:
            return None
        first = past_key_values[0]
        if not isinstance(first, StaticCacheState):
            return None

        # Static cache KV axis width is a concrete int: [B, max_seq_len, kv_hidden].
        # Guard against a symbolic dim, which would otherwise raise an opaque
        # TypeError downstream. Static-cache always allocates a fixed width today.
        max_seq_len = first.key_cache.shape[1]
        if not isinstance(max_seq_len, int):
            raise TypeError(
                "static-cache bias requires a concrete key_cache KV dimension "
                f"(axis 1), but got symbolic dim {max_seq_len!r}. The static "
                "cache must be allocated with a fixed max_seq_len."
            )
        # S_q lives at dim 1 of both input_ids ([B, S_q]) and hidden_states
        # ([B, S_q, hidden]), so the bias works for either forward entry point.
        seq_len = op.Shape(seq_len_source, start=1, end=2)  # (1,) int64 == [S_q]
        return create_static_cache_attention_bias(
            op,
            write_indices=first.write_indices,
            seq_len=seq_len,
            nonpad_kv_seqlen=first.nonpad_kv_seqlen,
            max_seq_len=max_seq_len,
            sliding_window=self._sliding_window,
            dtype=self._dtype,
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)

        # Determine whether to emit GroupQueryAttention directly.
        # Conditions:
        #  - attention_mask present: static-cache mode passes None; GQA requires seqlens_k.
        #  - EP gqa_dtypes: EP must declare GQA support for the build dtype (cuda/f16,
        #    cpu/f32, etc.). Default EP has gqa_dtypes={} so GQA is never emitted.
        #  - supports_fused_rope: EP must handle do_rotary=1 inside GQA. DML has
        #    gqa_dtypes={FLOAT16} but supports_fused_rope=False, so it uses the
        #    RotaryAttentionToGQA rewrite + SeparateRoPE path instead.
        #  - BaseRope (not _MRopeBase): standard 1D RoPE tables are required.
        #    _MRopeBase subclasses (ChunkedMRope for Qwen2.5-VL, InterleavedMRope for
        #    Qwen3-VL/Qwen3.5) use 3D position_ids; GQA do_rotary=1 only implements 1D
        #    RoPE, so those models must fall through to the RotaryAttentionToGQA rule.
        caps = ep_capabilities()
        dtype = get_build_dtype()
        use_gqa = (
            attention_mask is not None
            and dtype in caps.gqa_dtypes
            and caps.supports_fused_rope
            and isinstance(self.rotary_emb, BaseRope)
            and not isinstance(self.rotary_emb, _MRopeBase)
        )

        if use_gqa:
            # Call rotary_emb to realize cos_cache / sin_cache as ONNX graph
            # initializers (onnxscript registers parameters on module __call__).
            # The returned gathered embeddings are discarded — GroupQueryAttention
            # will index the full tables itself via do_rotary=1.
            self.rotary_emb(op, position_ids)

            # Build GQAContext from the cos/sin parameter tables and a
            # seqlens_k / total_seq_len pair derived from attention_mask.
            # Access cos_cache / sin_cache directly as ir.Value to avoid
            # creating dead Gather(cos_cache, position_ids) nodes.
            #
            # seqlens_k[b] = sum(attention_mask[b]) - 1 = last valid KV index.
            # total_seq_len = attention_mask.shape[1] = past + current len.
            one_i32 = op.Constant(value_int=1)
            seqlens_k = op.Cast(
                op.Sub(op.ReduceSum(attention_mask, [1], keepdims=0), one_i32),
                to=ir.DataType.INT32,
            )  # [batch] INT32
            total_seq_len = op.Cast(
                op.Gather(op.Shape(attention_mask), 1),
                to=ir.DataType.INT32,
            )  # scalar INT32

            attention_bias: GQAContext | ir.Value | None = GQAContext(
                seqlens_k=seqlens_k,
                total_seq_len=total_seq_len,
                cos_cache=self.rotary_emb.cos_cache,  # [max_seq, rotary_dim]
                sin_cache=self.rotary_emb.sin_cache,  # [max_seq, rotary_dim]
            )
            # position_embeddings not needed: GroupQueryAttention handles RoPE
            # internally via do_rotary=1. Passing None skips apply_rotary_pos_emb
            # in Attention.forward() (which checks `if position_embeddings is not None`).
            position_embeddings = None
        else:
            # NoPE models (e.g. NemotronH, GraniteMoeHybrid) have
            # ``rotary_emb = None`` because ``initialize_rope`` returned
            # ``None`` for ``config.rope_type is None``. Skip building
            # position_embeddings so that Attention.forward sees
            # ``position_embeddings=None`` and does not apply rotary encoding.
            if self.rotary_emb is not None:
                position_embeddings = self.rotary_emb(op, position_ids)
            else:
                position_embeddings = None

            # When attention_mask is None (static cache mode), skip mask
            # creation entirely — the Attention op uses is_causal=1 instead.
            # When present, create a bool padding mask. Causal masking is
            # handled by is_causal=1 on the Attention op (set in
            # _apply_attention), so we only need padding information here.
            if attention_mask is not None:
                attention_bias = create_padding_mask(
                    op,
                    input_ids=hidden_states if input_ids is None else input_ids,
                    attention_mask=attention_mask,
                )
            else:
                attention_bias = self._maybe_static_cache_bias(
                    op, hidden_states, past_key_values
                )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class CausalLMModel(nn.Module):
    """Standard causal language model with TextModel backbone and LM head.

    Compatible with Llama 2/3, Mistral, Qwen2/2.5, and other architectures
    that follow the standard decoder-only transformer pattern with GQA and RoPE.

    Replicates HuggingFace's ``LlamaForCausalLM``, ``MistralForCausalLM``,
    ``Qwen2ForCausalLM``, etc.

    Inputs: input_ids, attention_mask, position_ids, past_key_values.
    Outputs: logits (batch, seq_len, vocab_size), present_key_values.
    """

    default_task: str = "text-generation"
    category: str = "Text Generation"
    config_class: type = CausalLMConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)

        qc = getattr(config, "quantization", None)
        quantize_lm_head = qc is not None and getattr(qc, "quantize_lm_head", False)
        embed_quantized = qc is not None and getattr(qc, "quantize_embeddings", False)
        # Olive RTN may quantize+tie the head while clearing the model's
        # top-level tie flag; recover it from the quantization config.
        tie = config.tie_word_embeddings or (
            qc is not None and getattr(qc, "tie_word_embeddings", False)
        )

        if quantize_lm_head and embed_quantized and tie:
            # Tied quantized head: share the embedding's packed table and quant
            # params (one initializer each), reshaping to the MatMulNBits layout.
            self.lm_head = TiedQuantizedLMHead(
                self.model.embed_tokens, config.hidden_size, config.vocab_size
            )
        elif quantize_lm_head:
            # Untied quantized head (Olive RTN lm_head: true, not tied).
            zp_dtype = (
                config.dtype if getattr(qc, "float_zero_point", False) else ir.DataType.UINT8
            )
            lm_head_class = make_quantized_linear_factory(
                bits=qc.bits,
                block_size=qc.group_size,
                has_zero_point=not qc.sym,
                zero_point_dtype=zp_dtype,
            )
            self.lm_head = lm_head_class(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
            # Share a single ONNX initializer: lm_head and embed_tokens point
            # to the same nn.Parameter so only one ir.Value appears in the
            # graph. Only valid when both are unquantized float tables;
            # quantized embed/head use different packed layouts and are tied
            # by sharing Parameters in TiedQuantizedLMHead above.
            if config.tie_word_embeddings and not embed_quantized:
                self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess the state_dict to match the model's expected keys."""
        qc = getattr(self.config, "quantization", None)
        if qc is not None and qc.quant_method == "gptq":
            state_dict = preprocess_gptq_weights(
                state_dict, bits=qc.bits, group_size=qc.group_size
            )
        elif qc is not None and qc.quant_method == "awq":
            state_dict = preprocess_awq_weights(
                state_dict, bits=qc.bits, group_size=qc.group_size
            )
        elif qc is not None and qc.quant_method == "olive":
            # Olive-packed weights: also handles quantized embed/lm_head and
            # the float tied-head fallback, so return directly.
            tie = self.config.tie_word_embeddings or getattr(qc, "tie_word_embeddings", False)
            return preprocess_olive_weights(
                state_dict,
                bits=qc.bits,
                group_size=qc.group_size,
                quantize_embeddings=getattr(qc, "quantize_embeddings", False),
                quantize_lm_head=getattr(qc, "quantize_lm_head", False),
                tie_word_embeddings=tie,
            )
        if self.config.tie_word_embeddings:
            # Ensure both embed_tokens.weight and lm_head.weight are present so
            # apply_weights can assign each to its initializer.  For graph-level
            # tied models (standard CausalLMModel) both ir.Values are the same
            # object, so apply_weights' id()-dedup redirects lm_head uses to the
            # embed_tokens canonical and drops the duplicate initializer.  For
            # subclasses that override self.model after super().__init__ (e.g.
            # Cohere, GPT-2 family), the ir.Values differ but the dedup still
            # unifies them at load time via replace_all_uses_with.
            tie_word_embeddings(state_dict)
        return state_dict


class LayerNormTextModel(TextModel):
    """TextModel variant that uses LayerNorm (with bias) instead of RMSNorm.

    Used by models such as Cohere, StarCoder2, and StableLM where HuggingFace
    uses ``nn.LayerNorm`` (mean-centering + std-normalisation with learnable
    weight and bias) rather than the bias-free RMS normalisation.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Replace per-layer norms: DecoderLayer defaults to RMSNorm; override with LayerNorm.
        qc = getattr(config, "quantization", None)
        linear_class = None
        if qc is not None and qc.quant_method != "none":
            zp_dtype = (
                config.dtype if getattr(qc, "float_zero_point", False) else ir.DataType.UINT8
            )
            linear_class = make_quantized_linear_factory(
                bits=qc.bits,
                block_size=qc.group_size,
                has_zero_point=not qc.sym,
                zero_point_dtype=zp_dtype,
            )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, linear_class=linear_class, norm_class=LayerNorm)
                for _ in range(config.num_hidden_layers)
            ]
        )
        # Replace final norm with LayerNorm (weight + bias).
        self.norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)


class LayerNormCausalLMModel(CausalLMModel):
    """CausalLM variant that uses LayerNorm instead of RMSNorm.

    Drop-in replacement for ``CausalLMModel`` for architectures where
    HuggingFace uses standard ``nn.LayerNorm`` (weight + bias) in place of the
    bias-free RMSNorm used by most Llama-family models.

    Used by: Cohere, Cohere2, StarCoder2, StableLM.

    Replicates HuggingFace's ``CohereForCausalLM``, ``Starcoder2ForCausalLM``,
    and ``StableLmForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Replace TextModel with the LayerNorm-based variant.
        self.model = LayerNormTextModel(config)


class FusedGateUpCausalLMModel(CausalLMModel):
    """CausalLM variant that keeps gate_up_proj fused (no weight splitting).

    Use this instead of ``CausalLMModel`` for architectures where HuggingFace
    stores the gate and up projections as a single fused ``gate_up_proj``
    weight — e.g. Phi-3, Phi-4, and GLM.

    The MLP forward pass does a single ``gate_up_proj`` MatMul and splits the
    resulting activations, rather than splitting the weights at load time.
    This is robust to GPTQ int32-packed weights where dimension 0 is
    ``original / pack_factor`` and weight splitting would fail.

    ``preprocess_weights`` does NOT need to split ``gate_up_proj``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Parameterize TextModel to use FusedGateUpMLP for each decoder layer.
        self.model = TextModel(config, mlp_class=FusedGateUpMLP)
