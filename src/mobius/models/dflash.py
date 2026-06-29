# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DFlash speculative-decoding draft model.

Mobius port of ``DFlashDraftModel`` in ``z-lab/dflash:dflash/model.py``.
The drafter has no embedding table and no LM head — those are borrowed
from the target at inference time, so this module's forward signature
is ``(noise_embedding, target_hidden, position_ids, q_position_ids,
past_key_values) → (draft_hidden, present_key_values)``.

The two ``*_position_ids`` inputs cover the K and Q rotary positions
separately so the graph does not need any symbolic slicing of the RoPE
tables — see :class:`mobius.components._dflash.DFlashAttention` for the
attention-side wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import DFlashConfig
from mobius.components import Linear, RMSNorm, initialize_rope
from mobius.components._dflash import DFlashDecoderLayer

if TYPE_CHECKING:
    import onnx_ir as ir


class DFlashDraftModel(nn.Module):
    """DFlash drafter — a stack of cross-attending Qwen3-style blocks.

    Inputs (graph-level, set up by :class:`~mobius.tasks.DFlashDraftTask`):
        noise_embedding: ``[batch, q_len, hidden]`` — the target's
            ``embed_tokens(block_output_ids)`` where
            ``block_output_ids = [prev_token, mask, ..., mask]``.
        target_hidden: ``[batch, ctx_len, num_target_layers * hidden]`` —
            concatenated post-residual hidden states of selected target
            decoder layers, as configured by
            :attr:`DFlashConfig.target_layer_ids`.  Projected down to
            ``hidden`` by ``self.fc`` and RMSNorm'd by ``self.hidden_norm``.
        position_ids: ``[batch, ctx_len + q_len]`` — absolute positions
            for the K side (covers context tokens followed by noise tokens).
        q_position_ids: ``[batch, q_len]`` — absolute positions for the
            Q side (noise tokens only).
        past_key_values: per-layer ``(key, value)`` cache pairs from prior
            speculative steps.

    Outputs:
        draft_hidden: ``[batch, q_len, hidden]`` — final hidden states for
            each noise position.  The caller decodes them through the
            target's ``lm_head`` to obtain draft token logits.
        present_key_values: updated per-layer cache pairs.
    """

    config_class: type = DFlashConfig
    default_task: str = "dflash-draft"
    category: str = "Text Generation"

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not config.target_layer_ids:
            raise ValueError(
                "DFlashConfig.target_layer_ids must be a non-empty list — got "
                f"{config.target_layer_ids!r}.  Read it from the draft "
                "checkpoint's dflash_config or compute via "
                "dflash.model.build_target_layer_ids()."
            )

        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # Project the concatenated target hidden states down to hidden_size.
        # No bias — matches the reference implementation.
        self.fc = Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        noise_embedding: ir.Value,
        target_hidden: ir.Value,
        position_ids: ir.Value,
        q_position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        # Project + normalize the target context tokens once for all layers.
        target_hidden = self.fc(op, target_hidden)
        target_hidden = self.hidden_norm(op, target_hidden)

        # Gather RoPE embeddings for the two position streams independently
        # — this is cheaper than slicing inside every layer and avoids any
        # symbolic Slice on the cos/sin tables.
        k_position_embeddings = self.rotary_emb(op, position_ids)
        q_position_embeddings = self.rotary_emb(op, q_position_ids)

        hidden_states = noise_embedding
        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                q_position_embeddings=q_position_embeddings,
                k_position_embeddings=k_position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values
