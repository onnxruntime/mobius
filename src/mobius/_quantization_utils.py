# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared layout helpers for affine-quantized weights."""

from __future__ import annotations


def resolve_matmul_nbits_block_size(block_size: int, in_features: int) -> int:
    """Resolve per-channel ``-1`` to one block of exactly K elements."""
    if block_size == -1:
        if in_features <= 0:
            raise ValueError(f"Per-channel quantization requires K > 0, got {in_features}")
        return in_features
    return block_size
