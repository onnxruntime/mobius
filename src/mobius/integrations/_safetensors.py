# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

# SECURITY: This module parses safetensors files directly — no pickle,
# no arbitrary code execution. Only structured JSON headers and raw
# tensor bytes are read.

"""Memory-mapped safetensors loading for PyTorch tensors.

This module provides a custom safetensors parser that memory-maps weight
data into PyTorch tensors instead of eagerly copying file contents into
RAM. For large models the memory savings are significant: the OS pages
in tensor data on demand and can reclaim pages under memory pressure.

The implementation mirrors the `safetensors
<https://github.com/huggingface/safetensors>`_ wire format:

.. code-block:: text

    ┌──────────────┬──────────────────┬──────────────────────┐
    │ 8-byte LE u64│  JSON header     │  raw tensor bytes    │
    │ (header len) │  (tensor meta)   │  (contiguous)        │
    └──────────────┴──────────────────┴──────────────────────┘

Each tensor's metadata contains ``dtype``, ``shape``, and
``data_offsets`` (a ``[start, end)`` byte range into the raw data
region).

``torch.UntypedStorage.from_file()`` sets up a virtual memory mapping
without reading any data.  Sub-storage slicing (``storage[start:end]``)
returns a view — not a copy — into the same mapping.  The resulting
:class:`torch.Tensor` objects are therefore already lazy: the OS pages
in data only when the tensor is actually accessed.
"""

from __future__ import annotations

__all__ = [
    "load_safetensors_mmap",
]

import json
import os
import struct
from typing import Any

import torch

# The first 8 bytes of a safetensors file are a little-endian uint64
# encoding the length of the JSON header that follows.
_HEADER_SIZE_BYTES = 8

# Maximum header size (100 MB).  The reference safetensors Rust
# implementation enforces a similar cap.  Without this a crafted file
# could set header_size to gigabytes and cause an OOM on ``f.read()``.
_MAX_HEADER_SIZE = 100 * 1024 * 1024

# Safetensors dtype strings → PyTorch dtypes.
# Reference: https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs
# U16/U32/U64 require PyTorch 2.3+; gracefully omit if unavailable.
_SAFETENSORS_DTYPE_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

# torch.uint16/uint32/uint64 added in PyTorch 2.3
for _sf_name, _torch_name in (("U16", "uint16"), ("U32", "uint32"), ("U64", "uint64")):
    _dt = getattr(torch, _torch_name, None)
    if _dt is not None:
        _SAFETENSORS_DTYPE_TO_TORCH_DTYPE[_sf_name] = _dt


def _parse_header(
    path: str | os.PathLike,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Parse the safetensors JSON header from *path*.

    Returns:
        A tuple of ``(header_dict, header_byte_length)`` where
        *header_dict* maps tensor names to their metadata
        (``dtype``, ``shape``, ``data_offsets``) and may contain
        a ``__metadata__`` key with arbitrary string key-value pairs.
    """
    with open(path, "rb") as f:
        raw_size = f.read(_HEADER_SIZE_BYTES)
        if len(raw_size) < _HEADER_SIZE_BYTES:
            raise ValueError(f"Safetensors file too small to contain a valid header: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size > _MAX_HEADER_SIZE:
            raise ValueError(
                f"Safetensors header size {header_size} exceeds "
                f"maximum allowed {_MAX_HEADER_SIZE} bytes: {path}"
            )
        raw_header = f.read(header_size)
        if len(raw_header) < header_size:
            raise ValueError(
                f"Safetensors header truncated: expected {header_size} "
                f"bytes but got {len(raw_header)}: {path}"
            )
    header: dict[str, dict[str, Any]] = json.loads(raw_header.decode("utf-8"))
    return header, header_size


def load_safetensors_mmap(
    path: str | os.PathLike,
) -> dict[str, torch.Tensor]:
    """Load tensors from a safetensors file using memory-mapped I/O.

    This is a drop-in replacement for ``safetensors.torch.load_file()``
    that memory-maps the file instead of eagerly reading all tensor
    data into RAM.  The returned tensors share the memory-mapped
    storage; the OS pages data in on demand.

    Args:
        path: Path to the ``.safetensors`` file.

    Returns:
        A dictionary mapping tensor names to :class:`torch.Tensor`
        instances backed by memory-mapped storage.

    Raises:
        ValueError: If the file is corrupted or truncated.
        KeyError: If a tensor dtype is not supported.
    """
    path = os.fspath(path)
    header, header_size = _parse_header(path)

    # Offset from the start of the file where raw tensor data begins.
    data_offset = _HEADER_SIZE_BYTES + header_size

    file_size = os.path.getsize(path)
    # Memory-map the entire file into an UntypedStorage. The OS will
    # lazily page in only the regions that are actually accessed.
    storage = torch.UntypedStorage.from_file(path, shared=False, nbytes=file_size)

    tensors: dict[str, torch.Tensor] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue

        dtype_str = metadata["dtype"]
        if dtype_str not in _SAFETENSORS_DTYPE_TO_TORCH_DTYPE:
            raise KeyError(
                f"Unsupported safetensors dtype '{dtype_str}' for "
                f"tensor '{name}'. Supported: "
                f"{sorted(_SAFETENSORS_DTYPE_TO_TORCH_DTYPE)}"
            )
        torch_dtype = _SAFETENSORS_DTYPE_TO_TORCH_DTYPE[dtype_str]
        shape = metadata["shape"]
        start, end = metadata["data_offsets"]

        # Validate data offsets to prevent silent out-of-bounds reads.
        if start < 0 or end < start:
            raise ValueError(
                f"Invalid data_offsets [{start}, {end}) for tensor "
                f"'{name}': offsets must satisfy 0 <= start <= end"
            )
        if data_offset + end > file_size:
            raise ValueError(
                f"Tensor '{name}' data_offsets [{start}, {end}) "
                f"extend beyond file size "
                f"({data_offset + end} > {file_size}): {path}"
            )

        # Validate that byte size matches expected size from shape and dtype.
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        element_size = torch.tensor([], dtype=torch_dtype).element_size()
        expected_bytes = num_elements * element_size
        actual_bytes = end - start
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Tensor '{name}' byte size mismatch: data_offsets "
                f"span {actual_bytes} bytes but shape {shape} with "
                f"dtype {dtype_str} requires {expected_bytes} bytes"
            )

        byte_start = data_offset + start
        byte_end = data_offset + end

        # Create a tensor backed by a view into the mmap'd storage.
        # Sub-storage slicing returns a view (not a copy) — the tensor
        # data is only paged in by the OS when actually accessed.
        sub_storage = storage[byte_start:byte_end]
        tensors[name] = torch.empty(0, dtype=torch_dtype).set_(sub_storage).reshape(shape)

    return tensors
