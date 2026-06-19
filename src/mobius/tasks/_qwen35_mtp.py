# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-IO contract for the Qwen3.6 MTP self-speculative head.

The MTP head is a single ``full_attention`` decoder layer with a standard
GQA KV cache, plus an extra ``hidden_states`` input carrying the target
model's last hidden state (the tensor the target feeds to its ``lm_head``).

Graph inputs:
    - ``input_ids``      : ``[batch, seq_len]`` INT64 — the just-emitted
      token(s) ``t_{i+1}`` the head conditions on.
    - ``hidden_states``  : ``[batch, seq_len, hidden]`` (model dtype) — the
      target's last hidden state ``h_i`` (post-final-norm).
    - ``attention_mask`` : ``[batch, past_seq_len + seq_len]`` INT64.
    - ``position_ids``   : ``[batch, seq_len]`` INT64.
    - ``past_key_values.0.key`` / ``.value`` : standard GQA KV cache for the
      single MTP decoder layer, shape
      ``[batch, num_kv_heads, past_seq_len, head_dim]``.

Graph outputs:
    - ``logits``         : ``[batch, seq_len, vocab_size]`` — draft logits
      for ``t_{i+2}``.
    - ``present.0.key`` / ``.value`` : updated KV cache.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import Qwen35MtpConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import _make_kv_cache_inputs, _register_kv_cache_outputs


class Qwen35MtpTask(ModelTask):
    """Builds the ONNX graph for a :class:`~mobius.models.Qwen35MtpModel`."""

    def build(
        self,
        module: nn.Module,
        config: Qwen35MtpConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph()
        op = builder.op

        input_ids = builder.input(
            "input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        hidden_states = builder.input(
            "hidden_states",
            dtype=config.dtype,
            shape=[batch, seq_len, config.hidden_size],
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

        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
            total_seq_len="past_sequence_len + sequence_len",
            dtype=config.dtype,
        )

        return ModelPackage({"model": _make_model(graph)}, config=config)
