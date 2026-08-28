# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses

from mobius._builder import build_from_module
from mobius._testing import make_config
from mobius.models.base import CausalLMModel
from mobius.tasks._draft_target import DraftTargetCausalLMTask


def test_draft_target_task_emits_decoder_embedding_and_head() -> None:
    base = make_config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    fields = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    fields["output_layer_indices"] = [0, 1]
    config = type(base)(**fields)
    package = build_from_module(
        CausalLMModel(config),
        config,
        task=DraftTargetCausalLMTask(),
    )

    assert set(package) == {"model", "embedding", "lm_head"}
    assert [value.name for value in package["embedding"].graph.inputs] == ["input_ids"]
    assert [value.name for value in package["embedding"].graph.outputs] == ["inputs_embeds"]
    assert [value.name for value in package["lm_head"].graph.inputs] == ["hidden_states"]
    assert [value.name for value in package["lm_head"].graph.outputs] == ["logits"]
    decoder_outputs = {value.name for value in package["model"].graph.outputs}
    assert {"hidden_states.0", "hidden_states.1"}.issubset(decoder_outputs)
