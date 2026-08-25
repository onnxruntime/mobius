# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-IO contract for the DFlash speculative-decoding drafter."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import DFlashConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import _make_kv_cache_inputs, _register_kv_cache_outputs


class DFlashDraftTask(ModelTask):
    """Builds the ONNX graph for a :class:`~mobius.models.DFlashDraftModel`.

    Graph inputs:
        - ``noise_embedding``  : ``[batch, q_len, hidden]`` (model dtype)
        - ``target_hidden``    : ``[batch, ctx_len,
          len(target_layer_ids) * hidden]`` (model dtype)
        - ``position_ids``     : ``[batch, ctx_len + q_len]`` INT64 — used
          for K's rotary embeddings (covers context tokens followed by
          noise tokens).
        - ``q_position_ids``   : ``[batch, q_len]`` INT64 — used for Q's
          rotary embeddings (noise tokens only).
        - ``past_key_values.{i}.key`` / ``.value`` : standard GQA KV cache
          shape ``[batch, num_kv_heads, past_seq_len, head_dim]``.

    Graph outputs:
        - ``draft_hidden`` : ``[batch, q_len, hidden]`` (model dtype) — the
          drafter's final hidden states.  Decode them through the target's
          ``lm_head`` to get draft token logits.
        - ``present.{i}.key`` / ``.value`` : updated KV cache with shape
          ``[batch, num_kv_heads, past_seq_len + ctx_len + q_len, head_dim]``.

    No ``attention_mask`` input — the drafter is non-causal and assumes
    no padding within the speculative block.
    """

    def build(
        self,
        module: nn.Module,
        config: DFlashConfig,
    ) -> ModelPackage:
        if not getattr(config, "target_layer_ids", None):
            raise ValueError(
                "DFlashDraftTask requires DFlashConfig.target_layer_ids to be a "
                "non-empty list."
            )

        batch = ir.SymbolicDim("batch")
        q_len = ir.SymbolicDim("q_len")
        ctx_len = ir.SymbolicDim("ctx_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph()
        op = builder.op

        num_target_features = len(config.target_layer_ids) * config.hidden_size

        noise_embedding = builder.input(
            "noise_embedding",
            dtype=config.dtype,
            shape=[batch, q_len, config.hidden_size],
        )
        target_hidden = builder.input(
            "target_hidden",
            dtype=config.dtype,
            shape=[batch, ctx_len, num_target_features],
        )
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, "ctx_len + q_len"],
        )
        q_position_ids = builder.input(
            "q_position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, q_len],
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

        draft_output, present_key_values = module(
            op,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=position_ids,
            q_position_ids=q_position_ids,
            past_key_values=past_key_values,
        )

        builder.add_output(
            draft_output,
            "draft_logits" if config.use_draft_lm_head else "draft_hidden",
        )
        _register_kv_cache_outputs(
            builder,
            present_key_values,
            batch=batch,
            num_kv_heads=config.num_key_value_heads,
            key_head_dim=config.head_dim,
            value_head_dim=config.head_dim,
            total_seq_len="past_sequence_len + ctx_len + q_len",
            dtype=config.dtype,
        )

        return ModelPackage({"model": _make_model(graph)}, config=config)
