# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-IO contract for EAGLE-3 speculative-decoding draft models."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import Eagle3Config
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import _make_kv_cache_inputs, _register_kv_cache_outputs


class Eagle3DraftTask(ModelTask):
    """Builds the ONNX graph for :class:`~mobius.models.Eagle3DraftModel`."""

    def build(
        self,
        module: nn.Module,
        config: Eagle3Config,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph()

        input_ids = None
        inputs_embeds = None
        if config.use_draft_token_embedding:
            input_ids = builder.input(
                "input_ids",
                dtype=ir.DataType.INT64,
                shape=[batch, seq_len],
            )
        else:
            inputs_embeds = builder.input(
                "inputs_embeds",
                dtype=config.dtype,
                shape=[batch, seq_len, config.hidden_size],
            )
        fused_hidden = builder.input(
            "fused_hidden",
            dtype=config.dtype,
            shape=[
                batch,
                seq_len,
                3 * (config.target_hidden_size or config.hidden_size),
            ],
        )
        recycled_hidden = builder.input(
            "recycled_hidden",
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

        draft_output, recycled_hidden_out, present_key_values = module(
            builder.op,
            inputs_embeds=inputs_embeds,
            fused_hidden=fused_hidden,
            recycled_hidden=recycled_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_ids=input_ids,
        )

        builder.add_output(
            draft_output,
            "draft_hidden" if config.use_target_lm_head else "draft_logits",
        )
        builder.add_output(recycled_hidden_out, "next_hidden")
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
