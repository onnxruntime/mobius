# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for exact GGUF grouped-MoE module replacement."""

from __future__ import annotations

from onnxscript import nn

from mobius._testing import make_config
from mobius.components import (
    GroupedGatedSharedMoELayer,
    GroupedMoELayer,
    GroupedUngatedSharedMoELayer,
    MoELayer,
)
from mobius.integrations.gguf._builder import _replace_grouped_moe_layers
from mobius.models.moe import Qwen2MoELayer, UngatedSharedMoELayer


class _Root(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


def _config(**kwargs):
    return make_config(
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=48,
        **kwargs,
    )


def test_replaces_exact_supported_compositions() -> None:
    cases = [
        (MoELayer(_config(mlp_bias=True)), GroupedMoELayer),
        (Qwen2MoELayer(_config()), GroupedGatedSharedMoELayer),
        (UngatedSharedMoELayer(_config()), GroupedUngatedSharedMoELayer),
    ]
    for source, expected_type in cases:
        root = _Root(source)
        assert _replace_grouped_moe_layers(root) == ["mlp"]
        assert type(root.mlp) is expected_type


def test_does_not_blanket_replace_specialized_subclasses() -> None:
    class SpecializedMoE(MoELayer):
        pass

    root = _Root(SpecializedMoE(_config()))
    assert _replace_grouped_moe_layers(root) == []
    assert type(root.mlp) is SpecializedMoE
