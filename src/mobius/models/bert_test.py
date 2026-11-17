# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared tiny BERT-family configuration for model and metadata tests."""

from __future__ import annotations

from mobius._configs import ArchitectureConfig

PROTBERT_TINY_CONFIG = ArchitectureConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=128,
    vocab_size=30,
    max_position_embeddings=512,
    hidden_act="gelu",
    rms_norm_eps=1e-12,
    pad_token_id=0,
)
