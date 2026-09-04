# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded structural validation for GGUF metadata headers."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

_GGUF_ARCHITECTURE_KEY = b"general.architecture"
_GGUF_IDENTITY_STRING_KEYS = {
    b"general.name": "name",
}
_GGUF_IDENTITY_INTEGER_KEYS = {
    b"general.file_type": "file_type",
    b"general.quantization_version": "quantization_version",
}
_GGUF_SPLIT_KEYS = {
    b"split.no": "split_no",
    b"split.count": "split_count",
    b"split.tensors.count": "split_tensors_count",
}
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
_GGUF_INTEGER_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    10: "Q",
    11: "q",
}


class GGUFHeaderTruncatedError(ValueError):
    """A bounded GGUF header prefix ended before required metadata."""


@dataclass(frozen=True, slots=True)
class GGUFHeaderInfo:
    """Payload-free identity and split bookkeeping from one GGUF header."""

    architecture: str | None
    tensor_count: int
    split_no: int | None
    split_count: int | None
    split_tensors_count: int | None
    name: str | None = None
    file_type: int | None = None
    quantization_version: int | None = None
    tensor_type_ids: frozenset[int] = frozenset()


def _gguf_header_info_from_header(
    data: Any,
    *,
    source: str,
    require_architecture: bool = True,
    collect_tensor_type_ids: bool = False,
) -> GGUFHeaderInfo:
    """Validate a GGUF metadata table and return bounded preflight fields."""
    size = len(data)
    if size < 24 or data[0:4] != b"GGUF":
        raise ValueError(f"{source!r} does not begin with a valid GGUF header.")

    little_version = struct.unpack_from("<I", data, 4)[0]
    if little_version in {2, 3}:
        byte_order = "<"
    else:
        big_version = struct.unpack_from(">I", data, 4)[0]
        if big_version not in {2, 3}:
            raise ValueError(
                f"{source!r} uses an unsupported GGUF version "
                f"(little-endian={little_version}, big-endian={big_version})."
            )
        byte_order = ">"

    def read_uint32(offset: int) -> tuple[int, int]:
        end = offset + 4
        if end > size:
            raise GGUFHeaderTruncatedError(f"{source!r} has a truncated GGUF metadata header.")
        return struct.unpack_from(f"{byte_order}I", data, offset)[0], end

    def read_uint64(offset: int) -> tuple[int, int]:
        end = offset + 8
        if end > size:
            raise GGUFHeaderTruncatedError(f"{source!r} has a truncated GGUF metadata header.")
        return struct.unpack_from(f"{byte_order}Q", data, offset)[0], end

    def skip_bytes(offset: int, count: int, *, field_name: str) -> int:
        end = offset + count
        if end > size:
            raise GGUFHeaderTruncatedError(
                f"{source!r} has a truncated GGUF {field_name}: "
                f"requires {count} bytes with only {size - offset} remaining."
            )
        return end

    def read_string_span(offset: int) -> tuple[int, int, int]:
        length, offset = read_uint64(offset)
        end = offset + length
        if end > size:
            raise GGUFHeaderTruncatedError(
                f"{source!r} has a truncated GGUF metadata string: "
                f"declares {length} bytes with only {size - offset} remaining."
            )
        return offset, end, end

    def read_integer(value_type: int, offset: int, *, key_name: str) -> tuple[int, int]:
        format_char = _GGUF_INTEGER_FORMATS.get(value_type)
        if format_char is None:
            raise ValueError(
                f"{source!r} encodes {key_name} with non-integer GGUF type {value_type}."
            )
        width = _GGUF_SCALAR_WIDTHS[value_type]
        end = offset + width
        if end > size:
            raise GGUFHeaderTruncatedError(
                f"{source!r} has a truncated GGUF {key_name} metadata value."
            )
        return int(struct.unpack_from(f"{byte_order}{format_char}", data, offset)[0]), end

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
                raise GGUFHeaderTruncatedError(
                    f"{source!r} has a truncated GGUF metadata value."
                )
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
            raise GGUFHeaderTruncatedError(
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

    tensor_count = struct.unpack_from(f"{byte_order}Q", data, 8)[0]
    kv_count = struct.unpack_from(f"{byte_order}Q", data, 16)[0]
    if kv_count > _GGUF_MAX_METADATA_ENTRIES:
        raise ValueError(
            f"{source!r} declares {kv_count} GGUF metadata entries, exceeding "
            f"the safety limit of {_GGUF_MAX_METADATA_ENTRIES}."
        )
    # Every entry needs an 8-byte key length, a 4-byte type, and at least
    # one value byte. Reject impossible counts before entering the loop.
    if kv_count > (size - 24) // 13:
        raise GGUFHeaderTruncatedError(
            f"{source!r} has a truncated GGUF metadata table for {kv_count} entries."
        )

    offset = 24
    architecture_values: list[bytes] = []
    split_values: dict[str, list[int]] = {
        field_name: [] for field_name in _GGUF_SPLIT_KEYS.values()
    }
    identity_string_values: dict[str, list[bytes]] = {
        field_name: [] for field_name in _GGUF_IDENTITY_STRING_KEYS.values()
    }
    identity_integer_values: dict[str, list[int]] = {
        field_name: [] for field_name in _GGUF_IDENTITY_INTEGER_KEYS.values()
    }
    for _ in range(kv_count):
        key_start, key_end, offset = read_string_span(offset)
        value_type, offset = read_uint32(offset)
        key = bytes(data[key_start:key_end])
        is_architecture = (
            key_end - key_start == len(_GGUF_ARCHITECTURE_KEY)
            and key == _GGUF_ARCHITECTURE_KEY
        )
        if is_architecture:
            if value_type != _GGUF_STRING:
                raise ValueError(
                    f"{source!r} encodes general.architecture with GGUF type "
                    f"{value_type}, expected string type {_GGUF_STRING}."
                )
            value_start, value_end, offset = read_string_span(offset)
            architecture_values.append(bytes(data[value_start:value_end]))
        elif (field_name := _GGUF_IDENTITY_STRING_KEYS.get(key)) is not None:
            if value_type != _GGUF_STRING:
                raise ValueError(
                    f"{source!r} encodes general.{field_name} with GGUF type "
                    f"{value_type}, expected string type {_GGUF_STRING}."
                )
            value_start, value_end, offset = read_string_span(offset)
            identity_string_values[field_name].append(bytes(data[value_start:value_end]))
        elif (field_name := _GGUF_IDENTITY_INTEGER_KEYS.get(key)) is not None:
            value, offset = read_integer(
                value_type,
                offset,
                key_name=f"general.{field_name}",
            )
            identity_integer_values[field_name].append(value)
        elif (field_name := _GGUF_SPLIT_KEYS.get(key)) is not None:
            value, offset = read_integer(
                value_type, offset, key_name=field_name.replace("_", ".")
            )
            split_values[field_name].append(value)
        else:
            offset = skip_value(value_type, offset)

    if not architecture_values and not require_architecture:
        architecture = None
    elif len(architecture_values) != 1:
        raise ValueError(
            f"{source!r} must contain exactly one general.architecture metadata entry, "
            f"found {len(architecture_values)}."
        )
    else:
        try:
            architecture = architecture_values[0].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{source!r} has a non-UTF-8 general.architecture value."
            ) from error

    split_fields: dict[str, int | None] = {}
    for field_name, values in split_values.items():
        if len(values) > 1:
            raise ValueError(
                f"{source!r} contains duplicate {field_name.replace('_', '.')} metadata."
            )
        split_fields[field_name] = values[0] if values else None
    identity_strings: dict[str, str | None] = {}
    for field_name, values in identity_string_values.items():
        if len(values) > 1:
            raise ValueError(f"{source!r} contains duplicate general.{field_name} metadata.")
        if not values:
            identity_strings[field_name] = None
            continue
        try:
            identity_strings[field_name] = values[0].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{source!r} has a non-UTF-8 general.{field_name} value."
            ) from error
    identity_integers: dict[str, int | None] = {}
    for field_name, values in identity_integer_values.items():
        if len(values) > 1:
            raise ValueError(f"{source!r} contains duplicate general.{field_name} metadata.")
        identity_integers[field_name] = values[0] if values else None
    tensor_type_ids: frozenset[int] = frozenset()
    if collect_tensor_type_ids:
        types: set[int] = set()
        for tensor_index in range(tensor_count):
            _, _, offset = read_string_span(offset)
            dimensions, offset = read_uint32(offset)
            if dimensions > 4:
                raise ValueError(
                    f"{source!r} tensor {tensor_index} has {dimensions} dimensions; "
                    "GGUF tensor metadata permits at most 4."
                )
            offset = skip_bytes(
                offset,
                dimensions * 8,
                field_name=f"tensor {tensor_index} dimensions",
            )
            tensor_type, offset = read_uint32(offset)
            types.add(tensor_type)
            offset = skip_bytes(offset, 8, field_name=f"tensor {tensor_index} offset")
        tensor_type_ids = frozenset(types)
    return GGUFHeaderInfo(
        architecture=architecture,
        tensor_count=tensor_count,
        split_no=split_fields["split_no"],
        split_count=split_fields["split_count"],
        split_tensors_count=split_fields["split_tensors_count"],
        name=identity_strings["name"],
        file_type=identity_integers["file_type"],
        quantization_version=identity_integers["quantization_version"],
        tensor_type_ids=tensor_type_ids,
    )


def _gguf_architecture_from_header(
    data: Any,
    *,
    source: str,
    require_architecture: bool = True,
) -> str | None:
    """Validate a GGUF metadata table and return its architecture when present."""
    return _gguf_header_info_from_header(
        data,
        source=source,
        require_architecture=require_architecture,
    ).architecture
