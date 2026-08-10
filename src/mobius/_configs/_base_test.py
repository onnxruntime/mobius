# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for architecture-specific configuration extraction."""

from __future__ import annotations

import types

from mobius._configs import NemotronParseConfig


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
