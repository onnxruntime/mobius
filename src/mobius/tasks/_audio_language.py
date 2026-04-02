# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio-language 3-model split task for audio understanding models.

Same structure as SpeechLanguageTask but uses 1D position_ids for
decoders with standard RoPE (not MRoPE), such as Qwen2.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius.tasks._base import (
    _make_graph,
    _make_kv_cache_inputs,
    _make_model,
    _register_kv_cache_outputs,
)
from mobius.tasks._speech_language import SpeechLanguageTask


class AudioLanguageTask(SpeechLanguageTask):
    """3-model split for audio-language models with standard 1D RoPE.

    Identical to :class:`SpeechLanguageTask` except the decoder uses
    standard 1D ``position_ids`` with shape ``[batch, seq_len]`` instead
    of MRoPE 3D ``[3, batch, seq_len]``.

    The module must expose three sub-module attributes:

    - ``audio_tower``: audio encoder (mel → audio features)
    - ``embedding``: embedding model (input_ids + audio_features → inputs_embeds)
    - ``decoder``: text decoder with standard 1D RoPE and KV cache

    Use this task for models whose text decoder is a standard 1D-RoPE
    transformer (e.g. Qwen2), in contrast to MRoPE-based decoders like
    Qwen3-ASR.
    """

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build decoder with standard 1D position_ids [batch, seq_len]."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        inputs_embeds = ir.Value(
            name="inputs_embeds",
            shape=ir.Shape([batch, seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        # Standard 1D position_ids — shape [batch, seq_len]
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [inputs_embeds, attention_mask, position_ids]

        kv_inputs, past_key_values = _make_kv_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(kv_inputs)

        graph, builder = _make_graph(graph_inputs, name="decoder")
        logits, present_key_values = decoder(
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)
        return _make_model(graph)
