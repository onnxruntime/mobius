# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unregistered explicit-phase variants of the Puzzle text and Omni decoders.

Replicates the ``NemotronHPuzzleForCausalLM`` layout with opt-in chunked
prefill: token IDs/embeddings, a 2D total mask, native caches and BOOL[] phase
produce logits and native caches. The ordinary attention route is unchanged.
"""

from __future__ import annotations

from typing import Any

import onnx_ir as ir
from onnxscript import OpBuilder

from mobius.components import (
    INT64_MAX,
    ChunkedPrefillMamba2Block,
    create_padding_mask,
    validate_cache_phase,
)
from mobius.models.nemotron_h import (
    NemotronHPuzzleCausalLMModel,
    NemotronHPuzzleMambaLayer,
    _NemotronHPuzzleTextModel,
)
from mobius.models.nemotron_h_omni import NemotronHOmniModel, _OmniDecoder


class _SourceChunkMambaLayer(NemotronHPuzzleMambaLayer):
    mamba_block_class = ChunkedPrefillMamba2Block

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[Any, ...] | None,
        past_key_value: tuple[ir.Value, ir.Value] | None,
        padding_mask: ir.Value | None = None,
        has_previous_state: ir.Value | None = None,
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value]]:
        del attention_bias, position_embeddings
        phase = validate_cache_phase(has_previous_state)
        residual = hidden_states
        normalized = self.norm(op, hidden_states)
        if padding_mask is not None:
            normalized = op.Mul(normalized, op.CastLike(padding_mask, normalized))
        if past_key_value is None:
            raise ValueError("Source-chunk decoder requires explicit cache tensors")
        conv, state = past_key_value
        output, conv, state = self.mamba(op, normalized, conv, state, padding_mask, phase)
        return op.Add(residual, output), (conv, state)


class _SourceChunkTextModel(_NemotronHPuzzleTextModel):
    mamba_layer_class = _SourceChunkMambaLayer

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[Any] | None = None,
        inputs_embeds: ir.Value | None = None,
        has_previous_state: ir.Value | None = None,
        cache_position: ir.Value | None = None,
    ) -> tuple[ir.Value, list[Any]]:
        del position_ids
        phase = validate_cache_phase(has_previous_state)
        if (
            attention_mask.shape is None
            or len(attention_mask.shape) != 2
            or attention_mask.dtype not in (ir.DataType.INT64, ir.DataType.BOOL)
        ):
            raise TypeError("Source-chunk attention_mask must be a 2D INT64/BOOL binary mask")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids and inputs_embeds is required")
        hidden = self.embed_tokens(op, input_ids) if inputs_embeds is None else inputs_embeds
        attention_bias = create_padding_mask(op, hidden, attention_mask)
        query_length = op.Shape(hidden, start=1, end=2)
        current_mask = op.Slice(attention_mask, op.Neg(query_length), [INT64_MAX], axes=[1])
        # Source mask offset is separate from cache validity. Ordinary callers
        # carry a total-length mask, so total-current is the physical cache
        # position. Direct source controls can explicitly supply cache_position.
        if cache_position is None:
            offset = op.Sub(op.Shape(attention_mask, start=1, end=2), query_length)
        else:
            if (
                cache_position.dtype != ir.DataType.INT64
                or cache_position.shape is None
                or len(cache_position.shape) != 1
            ):
                raise TypeError("cache_position must be an INT64 vector")
            offset = op.Reshape(op.Slice(cache_position, [0], [1], [0]), [1])
        current_mask = op.Where(
            op.Greater(offset, op.Constant(value_ints=[0])),
            op.CastLike(1, current_mask),
            current_mask,
        )
        mamba_mask = op.Unsqueeze(current_mask, [-1])
        if past_key_values is None or len(past_key_values) != len(self.layers):
            raise ValueError("Source-chunk decoder requires every mixed-cache slot")
        present = []
        for layer, past in zip(self.layers, past_key_values):
            kwargs = {}
            if isinstance(layer, _SourceChunkMambaLayer):
                kwargs = dict(padding_mask=mamba_mask, has_previous_state=phase)
            hidden, state = layer(op, hidden, attention_bias, None, past, **kwargs)
            present.append(state)
        return self.norm(op, hidden), present


class NemotronHPuzzleSourceChunkCausalLMModel(NemotronHPuzzleCausalLMModel):
    """Explicit opt-in candidate; construct with SourceChunkHybridCausalLMTask."""

    text_model_class = _SourceChunkTextModel

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[Any] | None = None,
        has_previous_state: ir.Value | None = None,
        cache_position: ir.Value | None = None,
    ) -> tuple[ir.Value, list[Any]]:
        hidden, present = self.model(
            op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            has_previous_state=has_previous_state,
            cache_position=cache_position,
        )
        return self.lm_head(op, hidden), present


class _SourceChunkOmniDecoder(_OmniDecoder):
    text_model_class = _SourceChunkTextModel

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[Any] | None = None,
        has_previous_state: ir.Value | None = None,
        cache_position: ir.Value | None = None,
    ) -> tuple[ir.Value, list[Any]]:
        hidden, present = self.model(
            op,
            None,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            has_previous_state,
            cache_position,
        )
        return self.lm_head(op, hidden), present


class NemotronHOmniSourceChunkModel(NemotronHOmniModel):
    """Same three components/weights, explicit-phase numerical-candidate decoder."""

    decoder_class = _SourceChunkOmniDecoder
