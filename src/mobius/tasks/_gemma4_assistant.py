# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-IO contract for the Gemma4-Assistant draft model.

Inputs:
    - ``inputs_embeds``: ``[batch, q_len, 2 * backbone_hidden_size]``
      (model dtype) — concatenation of the target's previous and current
      shared hidden states.
    - ``position_ids``: ``[batch, q_len]`` INT64 — drives the RoPE on Q.
    - For each unique layer type that appears in ``config.layer_types``
      (any of ``"full_attention"``, ``"sliding_attention"``):
        - ``shared_kv.{layer_type}.key``: ``[batch, num_kv_heads,
          kv_len, head_dim]`` (model dtype, BNSH layout).
        - ``shared_kv.{layer_type}.value``: same shape.

Outputs:
    - ``logits``: ``[batch, q_len, vocab_size]`` (model dtype).
    - ``projected_state``: ``[batch, q_len, backbone_hidden_size]``
      (model dtype) — post_projection of last_hidden_state, fed back to
      the target for the next speculative step.

No ``past_key_values.*`` / ``present.*`` — the assistant has no KV cache.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import Gemma4AssistantConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class Gemma4AssistantTask(ModelTask):
    """Build the ONNX graph for a :class:`~mobius.models.Gemma4AssistantCausalLMModel`."""

    def build(
        self,
        module: nn.Module,
        config: Gemma4AssistantConfig,
    ) -> ModelPackage:
        if not config.layer_types:
            raise ValueError(
                "Gemma4AssistantTask requires config.layer_types to be populated."
            )

        batch = ir.SymbolicDim("batch")
        q_len = ir.SymbolicDim("q_len")
        # Per-layer-type kv_len symbolic dims — full and sliding caches may
        # have different lengths (sliding is bounded by the window).
        kv_lens: dict[str, ir.SymbolicDim] = {}

        graph, builder = _make_graph()
        op = builder.op

        inputs_embeds = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=[batch, q_len, 2 * config.backbone_hidden_size],
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, q_len]
        )

        # One (key, value) input pair per unique layer type that appears in
        # the assistant.  E2B-it-assistant uses both sliding and full.
        unique_layer_types: list[str] = []
        for lt in config.layer_types:
            if lt not in unique_layer_types:
                unique_layer_types.append(lt)

        global_head_dim = config.global_head_dim or config.head_dim
        shared_kv: dict[str, tuple[ir.Value, ir.Value]] = {}
        for lt in unique_layer_types:
            is_full = lt == "full_attention"
            head_dim = global_head_dim if is_full else config.head_dim
            if is_full and config.num_global_key_value_heads is not None:
                num_kv = config.num_global_key_value_heads
            else:
                num_kv = config.num_key_value_heads
            kv_lens[lt] = ir.SymbolicDim(f"kv_len_{lt}")
            shared_key = builder.input(
                f"shared_kv.{lt}.key",
                dtype=config.dtype,
                shape=[batch, num_kv, kv_lens[lt], head_dim],
            )
            shared_value = builder.input(
                f"shared_kv.{lt}.value",
                dtype=config.dtype,
                shape=[batch, num_kv, kv_lens[lt], head_dim],
            )
            shared_kv[lt] = (shared_key, shared_value)

        logits, projected_state = module(
            op,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            shared_kv=shared_kv,
        )

        builder.add_output(logits, "logits")
        builder.add_output(projected_state, "projected_state")

        return ModelPackage({"model": _make_model(graph)}, config=config)
