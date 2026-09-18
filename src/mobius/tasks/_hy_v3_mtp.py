# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph I/O contract for the independently cached Hunyuan-V3 NextN head."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import HyV3MtpConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import _make_kv_cache_inputs, _register_kv_cache_outputs


class HyV3MtpTask(ModelTask):
    """Build the one-layer Hunyuan-V3 MTP sidecar and its independent KV cache."""

    def build(self, module: nn.Module, config: HyV3MtpConfig) -> ModelPackage:
        batch = "batch_size"
        seq_len = "sequence_len"
        past_seq_len = "past_sequence_len"
        batch_dim = ir.SymbolicDim(batch)
        past_seq_len_dim = ir.SymbolicDim(past_seq_len)
        graph, builder = _make_graph()

        input_ids = None
        inputs_embeds = None
        if config.use_dedicated_embeddings:
            input_ids = builder.input(
                "input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
            )
        else:
            inputs_embeds = builder.input(
                "inputs_embeds",
                dtype=config.dtype,
                shape=[batch, seq_len, config.hidden_size],
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
            1,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch_dim,
            past_seq_len_dim,
        )
        prediction, present_key_values = module(
            builder.op,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        builder.add_output(
            prediction,
            "logits" if config.use_dedicated_lm_head else "mtp_hidden",
        )
        _register_kv_cache_outputs(
            builder,
            present_key_values,
            batch=batch_dim,
            num_kv_heads=config.num_key_value_heads,
            key_head_dim=config.head_dim,
            value_head_dim=config.head_dim,
            total_seq_len="past_sequence_len + sequence_len",
            dtype=config.dtype,
        )
        return ModelPackage({"model": _make_model(graph)}, config=config)
