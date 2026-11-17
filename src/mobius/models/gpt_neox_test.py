# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch

from mobius._testing import make_config
from mobius.models.gpt_neox import GPTNeoXCausalLMModel


def test_preprocess_weights_maps_lm_head_to_embed_out():
    config = make_config(
        attn_qkv_bias=True,
        attn_o_bias=True,
        mlp_bias=True,
        hidden_act="gelu",
        partial_rotary_factor=0.25,
    )
    model = GPTNeoXCausalLMModel(config)

    lm_head_weight = torch.zeros(config.vocab_size, config.hidden_size)
    state_dict = model.preprocess_weights({"lm_head.weight": lm_head_weight})

    assert "embed_out.weight" in state_dict
    assert state_dict["embed_out.weight"] is lm_head_weight
    assert "lm_head.weight" not in state_dict
