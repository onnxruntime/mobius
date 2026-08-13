# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for architecture-specific configuration extraction."""

from __future__ import annotations

import types

from mobius._configs import ArchitectureConfig, NemotronParseConfig


def test_nemotron_parse_maps_raw_mbart_decoder_attention_heads():
    """The Hub's non-trusted config exposes MBART's decoder-specific aliases."""
    config = types.SimpleNamespace(
        model_type="nemotron_parse",
        decoder={
            "model_type": "nemotron_parse_text",
            "d_model": 1024,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 4096,
            "decoder_layers": 10,
            "num_hidden_layers": 12,
            "vocab_size": 72256,
            "pad_token_id": 1,
        },
        encoder={"patch_size": 16, "max_resolution": 2048},
        image_size=[2048, 1664],
        max_sequence_length=9000,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        tie_word_embeddings=True,
        decoder_start_token_id=2,
    )

    extracted = NemotronParseConfig.from_transformers(config)

    assert extracted.num_attention_heads == 16
    assert extracted.num_key_value_heads == 16
    assert extracted.head_dim == 64
    assert extracted.num_decoder_layers == 10


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
