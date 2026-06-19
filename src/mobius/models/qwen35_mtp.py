# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3.6 multi-token-prediction (MTP) self-speculative head.

The dense ``Qwen/Qwen3.6-27B`` checkpoint ships an auxiliary MTP head under
the ``mtp.*`` weight prefix.  HuggingFace ``transformers`` discards those
weights on ``from_pretrained`` (the base model has no MTP module), so the
head is built here directly from the checkpoint tensors.

Architecturally the head predicts token ``t_{i+2}`` from the main model's
last hidden state ``h_i`` (post-final-norm — the same tensor that feeds the
target's ``lm_head``) and the just-emitted token ``t_{i+1}``::

    h'_i   = fc(concat[ pre_fc_norm_embedding(embed(t_{i+1})),
                        pre_fc_norm_hidden(h_i) ])          # fc: 2H -> H
    h''_i  = Qwen35DecoderLayer(full_attention)(h'_i)       # one layer
    logits = lm_head(norm(h''_i))

The single decoder layer is a ``full_attention`` :class:`Qwen35Attention`
block (doubled-Q output gating + per-head Q/K :class:`OffsetRMSNorm` +
partial mRoPE), identical to the target's full-attention layers — so it
reuses :class:`~mobius.models.qwen35.Qwen35DecoderLayer` unchanged.  The
three pre/post-fc norms are :class:`OffsetRMSNorm` (the ``1 + weight``
variant) and ``embed_tokens`` / ``lm_head`` are shared with the target
(``mtp_use_dedicated_embeddings = False``).

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
from mobius.components import Linear, initialize_rope
from mobius.components._common import Embedding, create_attention_bias
from mobius.components._rms_norm import OffsetRMSNorm
from mobius.models.qwen35 import Qwen35DecoderLayer

if TYPE_CHECKING:
    import onnx_ir as ir


class _Qwen35MtpTextModel(nn.Module):
    """Inner MTP layer stack — kept as ``self.model`` so weight names match
    the checkpoint after :meth:`Qwen35MtpModel.preprocess_weights` maps the
    ``mtp.*`` prefix onto ``model.*`` (e.g. ``mtp.layers.0.*`` ->
    ``model.layers.0.*``, ``mtp.norm`` -> ``model.norm``)."""

    def __init__(self, config: Qwen35MtpConfig):
        super().__init__()
        self._dtype = config.dtype

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        # Input projection: fuse the token embedding with the target hidden.
        self.pre_fc_norm_embedding = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.fc = Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # The single full-attention MTP decoder layer.
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(config, layer_idx=0)]
        )
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        # h'_i = fc(concat[pre_fc_norm_embedding(embed(t)),
        #                  pre_fc_norm_hidden(h_i)])
        embeds = self.embed_tokens(op, input_ids)
        embeds = self.pre_fc_norm_embedding(op, embeds)
        target_hidden = self.pre_fc_norm_hidden(op, hidden_states)
        fused = op.Concat(embeds, target_hidden, axis=-1)
        hs = self.fc(op, fused)

        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
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

        hs = self.norm(op, hs)
        return hs, present_key_values


class Qwen35MtpModel(nn.Module):
    """Qwen3.6 MTP self-speculative head.

    Inputs (graph-level, set up by
    :class:`~mobius.tasks.Qwen35MtpTask`):
        input_ids: ``[batch, seq_len]`` INT64 — the just-emitted token(s)
            ``t_{i+1}`` the head conditions on.
        hidden_states: ``[batch, seq_len, hidden]`` (model dtype) — the
            target model's last hidden state ``h_i`` (post-final-norm, i.e.
            the tensor the target feeds to its ``lm_head``).
        attention_mask: ``[batch, total_seq_len]`` INT64.
        position_ids: ``[batch, seq_len]`` INT64.
        past_key_values: standard GQA KV cache for the single MTP layer.

    Outputs:
        logits: ``[batch, seq_len, vocab_size]``.
        present_key_values: updated KV cache for the single MTP layer.
    """

    config_class: type = Qwen35MtpConfig
    default_task: str = "qwen35-mtp"
    category: str = "Text Generation"

    def __init__(self, config: Qwen35MtpConfig):
        super().__init__()
        self.config = config
        self.model = _Qwen35MtpTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map the checkpoint's ``mtp.*`` (and shared embed / lm_head) keys
        onto this module's layout, dropping everything else (the 64 main
        decoder layers, the vision tower, etc.).

        Mapping::

            mtp.fc.weight                     -> model.fc.weight
            mtp.pre_fc_norm_embedding.weight  -> model.pre_fc_norm_embedding.weight
            mtp.pre_fc_norm_hidden.weight     -> model.pre_fc_norm_hidden.weight
            mtp.norm.weight                   -> model.norm.weight
            mtp.layers.0.*                    -> model.layers.0.*
            (model.)?(language_model.)?embed_tokens.weight -> model.embed_tokens.weight
            lm_head.weight                    -> lm_head.weight
        """
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("mtp."):
                cleaned[f"model.{key[len('mtp.'):]}"] = value
            elif key == "lm_head.weight":
                cleaned["lm_head.weight"] = value
            elif key.endswith("embed_tokens.weight") and (
                key == "embed_tokens.weight"
                or key == "model.embed_tokens.weight"
                or key == "model.language_model.embed_tokens.weight"
                or key == "language_model.embed_tokens.weight"
            ):
                cleaned["model.embed_tokens.weight"] = value
            # Everything else (main decoder layers, visual tower, main norm)
            # is not consumed by the MTP head — drop it.
        return cleaned

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hs, present_key_values = self.model(
            op,
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hs)
        return logits, present_key_values
