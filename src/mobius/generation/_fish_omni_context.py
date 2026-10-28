# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Caller-side validation for the qualified source GDN continuation contract.

The ONNX graph keeps its existing ports. Graph execution alone does not reject
arbitrary positions; drivers must validate this contract before binding inputs.
No mixed-row positions, padding, static KV capacity, or capture is qualified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

SOURCE_CONTEXT_METADATA = "mobius.fish_omni.context_contract"
SOURCE_CONTEXT_CONTRACT = "shared-contiguous-dynamic-v1"


def validate_source_context(
    position_ids: np.ndarray,
    attention_mask: np.ndarray,
    input_shape: Sequence[int],
    cache_shapes: Mapping[str, Sequence[int]],
) -> None:
    """Validate shared contiguous positions and dynamic KV append without copying caches."""
    if len(input_shape) != 3:
        raise ValueError("Source Fish inputs_embeds must have rank 3")
    batch, length = input_shape[:2]
    if batch < 1 or length < 1:
        raise ValueError("Source Fish context requires a nonempty batch and sequence")
    if position_ids.dtype != np.int64 or position_ids.shape != (batch, length):
        raise ValueError("Source Fish position_ids must be INT64 [batch,sequence]")
    start = int(position_ids[0, 0])
    if start < 0 or start > np.iinfo(np.int64).max - length:
        raise ValueError("Source Fish position start is negative or overflows INT64")
    expected = np.arange(start, start + length, dtype=np.int64)
    if not np.all(position_ids == expected[None, :]):
        raise ValueError("Source Fish positions must be shared contiguous rows")
    if (
        attention_mask.dtype != np.int64
        or attention_mask.shape != (batch, start + length)
        or not np.all(attention_mask == 1)
    ):
        raise ValueError(
            "Source Fish requires an unpadded dynamic attention mask ending at start+T"
        )
    kv_shapes = {}
    for name, shape in cache_shapes.items():
        if not name.startswith("past_key_values."):
            continue
        if not shape or shape[0] != batch:
            raise ValueError(f"Source Fish cache batch does not match inputs: {name}")
        if name.endswith((".key", ".value")):
            if len(shape) != 4 or shape[2] != start:
                raise ValueError(
                    f"Source Fish past KV length must equal position start: {name}"
                )
            kv_shapes[name] = tuple(shape)
    if not kv_shapes:
        raise ValueError("Source Fish dynamic context requires full-attention KV caches")
    for name, shape in kv_shapes.items():
        other = name.rsplit(".", 1)[0] + (".value" if name.endswith(".key") else ".key")
        if kv_shapes.get(other) != shape:
            raise ValueError(f"Source Fish requires matching key/value cache shapes: {name}")
