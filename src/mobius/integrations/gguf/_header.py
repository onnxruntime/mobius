# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded structural validation for GGUF metadata headers."""

from __future__ import annotations

import struct
from typing import Any

_GGUF_ARCHITECTURE_KEY = b"general.architecture"
_GGUF_MAX_METADATA_ARRAY_DEPTH = 8
_GGUF_MAX_METADATA_ARRAY_ELEMENTS = 1_000_000
_GGUF_MAX_METADATA_ENTRIES = 1_000_000
_GGUF_SCALAR_WIDTHS = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


def _gguf_architecture_from_header(
    data: Any,
    *,
    source: str,
    require_architecture: bool = True,
) -> str | None:
    """Validate a GGUF metadata table and return its architecture when present."""
    size = len(data)
    if size < 24 or data[0:4] != b"GGUF":
        raise ValueError(f"{source!r} does not begin with a valid GGUF header.")

    little_version = struct.unpack_from("<I", data, 4)[0]
    if little_version in {2, 3}:
        byte_order = "<"
        version = little_version
    else:
        version = struct.unpack_from(">I", data, 4)[0]
        if version not in {2, 3}:
            raise ValueError(f"{source!r} uses unsupported GGUF version {little_version}.")
        byte_order = ">"

    def read_uint32(offset: int) -> tuple[int, int]:
        end = offset + 4
        if end > size:
            raise ValueError(f"{source!r} has a truncated GGUF metadata header.")
        return struct.unpack_from(f"{byte_order}I", data, offset)[0], end

    def read_uint64(offset: int) -> tuple[int, int]:
        end = offset + 8
        if end > size:
            raise ValueError(f"{source!r} has a truncated GGUF metadata header.")
        return struct.unpack_from(f"{byte_order}Q", data, offset)[0], end

    def read_string_span(offset: int) -> tuple[int, int, int]:
        length, offset = read_uint64(offset)
        end = offset + length
        if end > size:
            raise ValueError(
                f"{source!r} has a truncated GGUF metadata string: "
                f"declares {length} bytes with only {size - offset} remaining."
            )
        return offset, end, end

    def minimum_value_width(value_type: int) -> int:
        width = _GGUF_SCALAR_WIDTHS.get(value_type)
        if width is not None:
            return width
        if value_type == _GGUF_STRING:
            return 8
        if value_type == _GGUF_ARRAY:
            return 12
        raise ValueError(f"{source!r} uses unknown GGUF metadata type {value_type}.")

    def skip_value(value_type: int, offset: int, *, depth: int = 0) -> int:
        width = _GGUF_SCALAR_WIDTHS.get(value_type)
        if width is not None:
            end = offset + width
            if end > size:
                raise ValueError(f"{source!r} has a truncated GGUF metadata value.")
            return end
        if value_type == _GGUF_STRING:
            _, _, offset = read_string_span(offset)
            return offset
        if value_type != _GGUF_ARRAY:
            raise ValueError(f"{source!r} uses unknown GGUF metadata type {value_type}.")
        if depth >= _GGUF_MAX_METADATA_ARRAY_DEPTH:
            raise ValueError(f"{source!r} has excessively nested GGUF metadata arrays.")

        element_type, offset = read_uint32(offset)
        count, offset = read_uint64(offset)
        element_width = minimum_value_width(element_type)
        remaining = size - offset
        if count > remaining // element_width:
            raise ValueError(
                f"{source!r} has a truncated GGUF metadata array: declares {count} "
                f"elements requiring at least {count * element_width} bytes with only "
                f"{remaining} remaining."
            )
        if count > _GGUF_MAX_METADATA_ARRAY_ELEMENTS:
            raise ValueError(
                f"{source!r} declares {count} GGUF metadata array elements, exceeding "
                f"the safety limit of {_GGUF_MAX_METADATA_ARRAY_ELEMENTS}."
            )

        if element_type in _GGUF_SCALAR_WIDTHS:
            return offset + count * element_width
        for _ in range(count):
            offset = skip_value(element_type, offset, depth=depth + 1)
        return offset

    kv_count = struct.unpack_from(f"{byte_order}Q", data, 16)[0]
    if kv_count > _GGUF_MAX_METADATA_ENTRIES:
        raise ValueError(
            f"{source!r} declares {kv_count} GGUF metadata entries, exceeding "
            f"the safety limit of {_GGUF_MAX_METADATA_ENTRIES}."
        )
    # Every entry needs an 8-byte key length, a 4-byte type, and at least
    # one value byte. Reject impossible counts before entering the loop.
    if kv_count > (size - 24) // 13:
        raise ValueError(
            f"{source!r} has a truncated GGUF metadata table for {kv_count} entries."
        )

    offset = 24
    architecture_values: list[bytes] = []
    for _ in range(kv_count):
        key_start, key_end, offset = read_string_span(offset)
        value_type, offset = read_uint32(offset)
        is_architecture = (
            key_end - key_start == len(_GGUF_ARCHITECTURE_KEY)
            and data[key_start:key_end] == _GGUF_ARCHITECTURE_KEY
        )
        if is_architecture:
            if value_type != _GGUF_STRING:
                raise ValueError(
                    f"{source!r} encodes general.architecture with GGUF type "
                    f"{value_type}, expected string type {_GGUF_STRING}."
                )
            value_start, value_end, offset = read_string_span(offset)
            architecture_values.append(bytes(data[value_start:value_end]))
        else:
            offset = skip_value(value_type, offset)

    if not architecture_values and not require_architecture:
        return None
    if len(architecture_values) != 1:
        raise ValueError(
            f"{source!r} must contain exactly one general.architecture metadata entry, "
            f"found {len(architecture_values)}."
        )
    try:
        return architecture_values[0].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{source!r} has a non-UTF-8 general.architecture value.") from error
