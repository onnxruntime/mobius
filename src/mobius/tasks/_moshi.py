# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Task builders for the Moshi full-duplex speech-to-speech LM."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import (
    _make_kv_cache_inputs,
    _register_kv_cache_outputs,
)


class MoshiTemporalTask(ModelTask):
    """Moshi temporal transformer with dynamic KV cache.

    The module's ``forward()`` must accept
    ``(op, input_frame, attention_mask, position_ids, past_key_values)`` and
    return ``(hidden, text_logits, list_of_(key, value)_tuples)``.

    Inputs:
        - input_frame: [batch, 17, sequence_len] INT64 (channel 0 = text,
          channels 1..16 = audio codebooks)
        - attention_mask: [batch, past_seq_len + sequence_len] INT64
        - position_ids: [batch, sequence_len] INT64
        - past_key_values.{i}.key / .value: [batch, num_heads, past_seq_len,
          head_dim] FLOAT
    Outputs:
        - hidden: [batch, sequence_len, hidden_size] FLOAT (post out_norm)
        - text_logits: [batch, sequence_len, text_vocab] FLOAT
        - present.{i}.key / .value: FLOAT
    """

    _NUM_CHANNELS = 17

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

        input_frame = builder.input(
            "input_frame",
            dtype=ir.DataType.INT64,
            shape=[batch, self._NUM_CHANNELS, seq_len],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_seq_len + seq_len"],
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )

        past_key_values = _make_kv_cache_inputs(
            builder,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )

        hidden, text_logits, present_key_values = module(
            op,
            input_frame=input_frame,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        builder.add_output(hidden, "hidden")
        builder.add_output(text_logits, "text_logits")
        _register_kv_cache_outputs(
            builder,
            present_key_values,
            batch=batch,
            num_kv_heads=config.num_key_value_heads,
            key_head_dim=config.head_dim,
            value_head_dim=config.head_dim,
            total_seq_len="past_sequence_len + sequence_len",
            dtype=config.dtype,
        )

        return ModelPackage({"model": _make_model(graph)}, config=config)


class MoshiDepformerTask(ModelTask):
    """Moshi/PersonaPlex depformer substep with intra-frame KV cache.

    The module's ``forward()`` must accept
    ``(op, hidden, prev_token, substep_index, past_key_values)`` and return
    ``(logits, list_of_(key, value)_tuples)``. The graph is invoked 8 times
    per frame for public Moshi/Moshiko or 16 times for PersonaPlex.

    Inputs:
        - hidden: [batch, 1, temporal_dim] FLOAT (temporal transformer output)
        - prev_token: [batch, 1] INT64 (token sampled at the previous substep)
        - substep_index: scalar INT64 in [0, config.max_position_embeddings - 1]
        - past_key_values.{i}.key / .value: [batch, num_heads, past_len,
          head_dim] FLOAT
    Outputs:
        - logits: [batch, 1, audio_card] FLOAT
        - present.{i}.key / .value: FLOAT
    """

    _TEMPORAL_DIM = 4096

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        past_seq_len = ir.SymbolicDim("past_substep_len")

        graph, builder = _make_graph()
        op = builder.op

        hidden = builder.input(
            "hidden", dtype=config.dtype, shape=[batch, 1, self._TEMPORAL_DIM]
        )
        prev_token = builder.input("prev_token", dtype=ir.DataType.INT64, shape=[batch, 1])
        substep_index = builder.input("substep_index", dtype=ir.DataType.INT64, shape=[])

        past_key_values = _make_kv_cache_inputs(
            builder,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )

        logits, present_key_values = module(
            op,
            hidden=hidden,
            prev_token=prev_token,
            substep_index=substep_index,
            past_key_values=past_key_values,
        )

        builder.add_output(logits, "logits")
        _register_kv_cache_outputs(
            builder,
            present_key_values,
            batch=batch,
            num_kv_heads=config.num_key_value_heads,
            key_head_dim=config.head_dim,
            value_head_dim=config.head_dim,
            total_seq_len="past_substep_len + 1",
            dtype=config.dtype,
        )

        return ModelPackage({"model": _make_model(graph)}, config=config)
