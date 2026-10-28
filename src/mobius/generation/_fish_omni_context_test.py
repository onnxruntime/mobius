# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import pytest

from mobius.generation._fish_omni_context import validate_source_context


def _context(batch=2, length=65, start=30):
    return {
        "position_ids": np.broadcast_to(
            np.arange(start, start + length, dtype=np.int64), (batch, length)
        ).copy(),
        "attention_mask": np.ones((batch, start + length), np.int64),
        "input_shape": (batch, length, 32),
        "cache_shapes": {
            "past_key_values.0.conv_state": (batch, 512, 3),
            "past_key_values.0.recurrent_state": (batch, 2, 128, 128),
            "past_key_values.1.key": (batch, 2, start, 128),
            "past_key_values.1.value": (batch, 2, start, 128),
        },
    }


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("length", [1, 2, 65, 257])
@pytest.mark.parametrize("start", [0, 30])
def test_source_context_accepts_shared_contiguous_dynamic_append(batch, length, start):
    validate_source_context(**_context(batch, length, start))


@pytest.mark.parametrize(
    "invalid,match",
    [
        ("mixed", "shared contiguous"),
        ("gap", "shared contiguous"),
        ("negative", "negative"),
        ("overflow", "overflows"),
        ("position_rank", "INT64"),
        ("position_dtype", "INT64"),
        ("mask_padding", "unpadded"),
        ("mask_length", "unpadded"),
        ("mask_dtype", "unpadded"),
        ("static_kv", "past KV length"),
        ("stale_kv_initial", "past KV length"),
        ("cache_batch", "cache batch"),
        ("unpaired_kv", "matching key/value"),
        ("no_kv", "requires full-attention"),
        ("empty_time", "nonempty"),
        ("empty_batch", "nonempty"),
        ("input_rank", "rank 3"),
    ],
)
def test_source_context_rejects_unqualified_inputs(invalid, match):
    args = _context()
    if invalid == "mixed":
        args["position_ids"][1] += 1
    elif invalid == "gap":
        args["position_ids"][:, 1] += 1
    elif invalid == "negative":
        args["position_ids"] -= 31
    elif invalid == "overflow":
        args["position_ids"][:, 0] = np.iinfo(np.int64).max
    elif invalid == "position_rank":
        args["position_ids"] = args["position_ids"][0]
    elif invalid == "position_dtype":
        args["position_ids"] = args["position_ids"].astype(np.int32)
    elif invalid == "mask_padding":
        args["attention_mask"][1, 0] = 0
    elif invalid == "mask_length":
        args["attention_mask"] = args["attention_mask"][:, :-1]
    elif invalid == "mask_dtype":
        args["attention_mask"] = args["attention_mask"].astype(bool)
    elif invalid == "static_kv":
        args["cache_shapes"]["past_key_values.1.key"] = (2, 2, 512, 128)
    elif invalid == "stale_kv_initial":
        args["position_ids"] -= 30
        args["attention_mask"] = args["attention_mask"][:, :65]
    elif invalid == "cache_batch":
        args["cache_shapes"]["past_key_values.0.recurrent_state"] = (1, 2, 128, 128)
    elif invalid == "unpaired_kv":
        del args["cache_shapes"]["past_key_values.1.value"]
    elif invalid == "no_kv":
        args["cache_shapes"] = {}
    elif invalid == "empty_time":
        args["input_shape"] = (2, 0, 32)
    elif invalid == "empty_batch":
        args["input_shape"] = (0, 65, 32)
    elif invalid == "input_rank":
        args["input_shape"] = (2, 65)
    with pytest.raises(ValueError, match=match):
        validate_source_context(**args)
