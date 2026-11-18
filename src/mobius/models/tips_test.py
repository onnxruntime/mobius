# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius.models.tips import TipsConfig, TipsModel


def tiny_config() -> TipsConfig:
    return TipsConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=128,
        image_size=28,
        patch_size=14,
        vision_hidden_size=32,
        vision_num_layers=2,
        vision_num_heads=4,
        max_position_embeddings=8,
    )


def test_tipsv2_build():
    config = tiny_config()
    package = build_from_module(TipsModel(config), config, task=TipsModel.default_task)
    graph = package["model"].graph
    assert [v.name for v in graph.inputs] == ["pixel_values", "input_ids", "padding_mask"]
    assert [v.name for v in graph.outputs] == [
        "cls_token",
        "register_tokens",
        "patch_tokens",
        "text_embeds",
    ]
    assert "vision_encoder.register_tokens" in graph.initializers
    assert "vision_encoder.pos_embed" in graph.initializers


def test_tipsv2_raw_and_trusted_config():
    raw = {
        "text_config": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 32000,
            "max_position_embeddings": 64,
        },
        "vision_config": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "image_size": 448,
            "patch_size": 14,
        },
        "temperature_init_value": 0.02,
    }
    trusted = SimpleNamespace(
        text_hidden_size=768,
        text_mlp_dim=3072,
        text_num_layers=12,
        text_num_heads=12,
        vocab_size=32000,
        max_len=64,
        vision_fn="vit_base",
        embed_dim=768,
        img_size=448,
        patch_size=14,
        ffn_layer="mlp",
        init_values=1.0,
        num_register_tokens=1,
        temperature=0.02,
        dtype=None,
    )
    assert TipsConfig.from_transformers(raw) == TipsConfig.from_transformers(trusted)
    assert TipsConfig.from_transformers(trusted).dtype == ir.DataType.FLOAT
    raw["vision_config"]["num_attention_heads"] = 6
    with pytest.raises(ValueError, match="num_attention_heads=12"):
        TipsConfig.from_transformers(raw)
    trusted.vision_fn = "vit_large"
    with pytest.raises(ValueError, match="vision_fn='vit_base'"):
        TipsConfig.from_transformers(trusted)
    with pytest.raises(ValueError, match="complete pinned"):
        TipsConfig.from_transformers({})


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_tipsv2_reject_unverified_dtype(dtype):
    with pytest.raises(ValueError, match="float32 only"):
        TipsModel(dataclasses.replace(tiny_config(), dtype=dtype))
    config = tiny_config()
    module = TipsModel(config)
    config.dtype = dtype
    with pytest.raises(ValueError, match="float32 only"):
        build_from_module(module, config, task=module.default_task)


def test_tipsv2_weight_alignment():
    config = tiny_config()
    module = TipsModel(config)
    package = build_from_module(module, config, task=module.default_task)
    state = {}
    for name, value in package["model"].graph.initializers.items():
        if ".attn.q_proj." in name:
            prefix, suffix = name.split(".q_proj.")
            if prefix.startswith("vision_encoder"):
                source = f"{prefix}.qkv.{suffix}"
            else:
                source = f"{prefix}.in_proj_{suffix}"
            shape = list(value.shape)
            shape[0] *= 3
            state[source] = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
        elif ".attn.k_proj." not in name and ".attn.v_proj." not in name:
            source = (
                name.replace(".attn.out_proj.", ".attn.proj.")
                if name.startswith("vision_encoder")
                else name
            )
            state[source] = torch.ones(tuple(value.shape))
    state["vision_encoder.mask_token"] = torch.zeros(1, config.vision_hidden_size)
    mapped = module.preprocess_weights(state)
    assert set(package["model"].graph.initializers) == set(mapped)
    for name, value in package["model"].graph.initializers.items():
        assert tuple(value.shape) == tuple(mapped[name].shape), name
    q, k, v = [
        mapped[f"vision_encoder.blocks.0.attn.{p}_proj.weight"] for p in ("q", "k", "v")
    ]
    assert not torch.equal(q, k)
    assert not torch.equal(k, v)
