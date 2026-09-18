# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3.6 multi-token-prediction (MTP) self-speculative head.

The dense ``Qwen/Qwen3.6-27B`` checkpoint ships an auxiliary MTP head under
the ``mtp.*`` weight prefix.  HuggingFace ``transformers`` discards those
weights on ``from_pretrained`` (the base model has no MTP module), so the
head is built here directly from the checkpoint tensors.

The GGUF may carry dedicated ``nextn.embed_tokens`` and
``nextn.shared_head_head`` tables. When present, this sidecar owns them and
consumes ``input_ids`` and/or emits ``logits``. When absent, it consumes the
target's ``inputs_embeds`` and/or emits ``mtp_hidden`` for the target's shared
LM head. ``nextn.shared_head_norm`` similarly falls back to the target's final
norm weights.

Architecturally::

    h'_i       = fc(concat[ pre_fc_norm_embedding(inputs_embeds),
                            pre_fc_norm_hidden(hidden_states) ])    # fc: 2H -> H
    h''_i      = Qwen35DecoderLayer(full_attention)(h'_i)          # one layer
    mtp_hidden = norm(h''_i)

where ``hidden_states`` is the target model's last hidden state ``h_i``
(post-final-norm — the tensor that feeds the target's ``lm_head``, exposed
by splitting the target's lm_head off the decoder body).

The single decoder layer is a ``full_attention`` :class:`Qwen35Attention`
block (doubled-Q output gating + per-head Q/K :class:`OffsetRMSNorm` +
partial mRoPE), identical to the target's full-attention layers — so it
reuses :class:`~mobius.models.qwen35.Qwen35DecoderLayer` unchanged.  The
three pre/post-fc norms are :class:`OffsetRMSNorm` (the ``1 + weight``
variant).

References:
- GenAI builder ``Qwen35MtpHead`` (microsoft/onnxruntime-genai#2218).
- vLLM ``Qwen3_5MultiTokenPredictor`` (``qwen3_5_mtp.py``) — the public
  PyTorch reference; ``transformers`` has no MTP head.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Qwen35MtpConfig
from mobius.components import Embedding, Linear, create_attention_bias, initialize_rope
from mobius.components._rms_norm import OffsetRMSNorm
from mobius.models.qwen35 import Qwen35DecoderLayer, _linear_factory

if TYPE_CHECKING:
    import onnx_ir as ir


class Qwen35MtpModel(nn.Module):
    """Qwen3.6 MTP self-speculative head (a single cross-conditioned full-attention block).

    Inputs (graph-level, set up by :class:`~mobius.tasks.Qwen35MtpTask`):
        input_ids or inputs_embeds: Dedicated embeddings consume token IDs;
            fallback embeddings arrive from the target as
            ``[batch, seq_len, hidden]``.
        hidden_states: ``[batch, seq_len, hidden]`` (model dtype) — the
            target model's last hidden state ``h_i`` (post-final-norm).
        attention_mask: ``[batch, total_seq_len]`` INT64.
        position_ids: ``[batch, seq_len]`` INT64.
        past_key_values: standard GQA KV cache for the single MTP layer.

    Outputs:
        logits or mtp_hidden: Dedicated heads emit vocabulary logits; fallback
            heads emit final hidden states for the target's shared ``lm_head``.
        present_key_values: updated KV cache for the single MTP layer.
    """

    config_class: type = Qwen35MtpConfig
    default_task: str = "qwen35-mtp"
    category: str = "Text Generation"

    def __init__(self, config: Qwen35MtpConfig):
        super().__init__()
        self.config = config
        self._dtype = config.dtype

        self.embed_tokens = (
            Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
            if config.use_dedicated_embeddings
            else None
        )

        # Input projection: fuse the token embedding with the target hidden.
        self.pre_fc_norm_embedding = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        linear_class = _linear_factory(config) or Linear
        self.fc = linear_class(2 * config.hidden_size, config.hidden_size, bias=False)

        # The single full-attention MTP decoder layer.
        self.layers = nn.ModuleList([Qwen35DecoderLayer(config, layer_idx=0)])
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = (
            Linear(config.hidden_size, config.vocab_size, bias=False)
            if config.use_dedicated_lm_head
            else None
        )
        self.rotary_emb = initialize_rope(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Strip the ``mtp.`` prefix and drop everything else.

        This is the HuggingFace shared-table path, so only ``mtp.*`` weights
        are consumed. GGUF dedicated tables are mapped directly to final
        sidecar names and intentionally bypass this preprocessor.

        Mapping::

            mtp.fc.weight                     -> fc.weight
            mtp.pre_fc_norm_embedding.weight  -> pre_fc_norm_embedding.weight
            mtp.pre_fc_norm_hidden.weight     -> pre_fc_norm_hidden.weight
            mtp.norm.weight                   -> norm.weight
            mtp.layers.0.*                    -> layers.0.*
        """
        return {
            key[len("mtp.") :]: value
            for key, value in state_dict.items()
            if key.startswith("mtp.")
        }

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value | None,
        input_ids: ir.Value | None,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        if self.embed_tokens is not None:
            if input_ids is None:
                raise ValueError("Dedicated MTP embeddings require input_ids")
            inputs_embeds = self.embed_tokens(op, input_ids)
        elif inputs_embeds is None:
            raise ValueError("Shared MTP embeddings require inputs_embeds")

        # h'_i = fc(concat[pre_fc_norm_embedding(inputs_embeds),
        #                  pre_fc_norm_hidden(h_i)])
        embeds = self.pre_fc_norm_embedding(op, inputs_embeds)
        target_hidden = self.pre_fc_norm_hidden(op, hidden_states)
        fused = op.Concat(embeds, target_hidden, axis=-1)
        hs = self.fc(op, fused)

        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=inputs_embeds,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values: list = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hs, present_kv = layer(
                op,
                hidden_states=hs,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        mtp_hidden = self.norm(op, hs)
        if self.lm_head is not None:
            return self.lm_head(op, mtp_hidden), present_key_values
        return mtp_hidden, present_key_values
