# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for :class:`ArchitectureConfig.from_transformers`."""

from __future__ import annotations

from mobius._configs._base import ArchitectureConfig


class _FakeHFConfig:
    """Minimal HuggingFace-config stand-in with attribute access."""

    def __init__(self, model_type: str = "_unrelated_", **kwargs):
        self.model_type = model_type
        self.__dict__.update(kwargs)


def test_scalar_intermediate_size_passes_through():
    cfg = _FakeHFConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_attention_heads=8,
        num_hidden_layers=4,
        vocab_size=256,
    )
    out = ArchitectureConfig.from_transformers(cfg)
    assert out.intermediate_size == 8192


def test_list_intermediate_size_collapses_to_first_element():
    """Gemma 3n expresses intermediate_size as a per-layer list.

    The list is uniform in every shipped checkpoint, so collapsing to the
    first element yields the correct scalar MLP width. Without the coercion
    the list reached ``nn.Parameter``/``ir.Shape`` as a dim and raised.
    """
    cfg = _FakeHFConfig(
        hidden_size=2048,
        intermediate_size=[8192] * 4,
        num_attention_heads=8,
        num_hidden_layers=4,
        vocab_size=256,
    )
    out = ArchitectureConfig.from_transformers(cfg)
    assert out.intermediate_size == 8192
    assert isinstance(out.intermediate_size, int)
