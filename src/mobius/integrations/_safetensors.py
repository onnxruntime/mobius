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
"""

from __future__ import annotations

__all__ = [
    "MmapTensorDescriptor",
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
_SAFETENSORS_DTYPE_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "U16": torch.uint16,
    "U32": torch.uint32,
    "U64": torch.uint64,
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


class MmapTensorDescriptor:
    """Lightweight descriptor for a tensor stored in a memory-mapped file.

    Holds only the metadata (shape, dtype) and a reference to the
    parent :class:`torch.UntypedStorage` plus byte offsets.  The actual
    :class:`torch.Tensor` is created on demand via :meth:`materialize`,
    so for models whose ``preprocess_weights`` only renames keys the
    tensor data is never touched during weight loading.

    Attribute access other than :attr:`shape` and :attr:`dtype` is
    delegated to the materialized tensor, providing transparent
    compatibility with code that expects a :class:`torch.Tensor`.
    """

    __slots__ = (
        "_byte_end",
        "_byte_start",
        "_dtype",
        "_shape",
        "_storage",
        "_tensor",
    )

    def __init__(
        self,
        storage: torch.UntypedStorage,
        byte_start: int,
        byte_end: int,
        dtype: torch.dtype,
        shape: list[int],
    ) -> None:
        self._storage = storage
        self._byte_start = byte_start
        self._byte_end = byte_end
        self._dtype = dtype
        self._shape = torch.Size(shape)
        self._tensor: torch.Tensor | None = None

    @property
    def shape(self) -> torch.Size:
        """Tensor shape (available without materialization)."""
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """Tensor dtype (available without materialization)."""
        return self._dtype

    def is_materialized(self) -> bool:
        """Return True if the tensor has been materialized."""
        return self._tensor is not None

    def materialize(self) -> torch.Tensor:
        """Create a :class:`torch.Tensor` from the mmap'd storage.

        Each call creates a fresh tensor view into the memory-mapped
        file.  The result is **not** cached — use attribute delegation
        (via ``__getattr__``) for repeated access.
        """
        sub_storage = self._storage[self._byte_start : self._byte_end]
        return torch.empty(0, dtype=self._dtype).set_(sub_storage).reshape(list(self._shape))

    def __getattr__(self, name: str) -> Any:
        # Materialize once and cache for attribute delegation.
        # This path is hit by preprocess_weights methods that do
        # tensor operations (split, reshape, t, etc.).
        if self._tensor is None:
            self._tensor = self.materialize()
        return getattr(self._tensor, name)

    def __repr__(self) -> str:
        status = "materialized" if self._tensor is not None else "lazy"
        return (
            f"MmapTensorDescriptor(dtype={self._dtype}, shape={list(self._shape)}, {status})"
        )


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
    *,
    lazy: bool = False,
) -> dict[str, torch.Tensor | MmapTensorDescriptor]:
    """Load tensors from a safetensors file using memory-mapped I/O.

    This is a drop-in replacement for ``safetensors.torch.load_file()``
    that memory-maps the file instead of eagerly reading all tensor
    data into RAM.  The returned tensors share the memory-mapped
    storage; the OS pages data in on demand.

    Args:
        path: Path to the ``.safetensors`` file.
        lazy: If ``True``, return :class:`MmapTensorDescriptor` objects
            instead of :class:`torch.Tensor`.  Descriptors defer tensor
            creation until first attribute access (beyond ``.shape`` /
            ``.dtype``), enabling near-zero peak memory for weight
            pipelines that only rename keys.

    Returns:
        A dictionary mapping tensor names to :class:`torch.Tensor`
        instances (or :class:`MmapTensorDescriptor` when *lazy* is
        ``True``) backed by memory-mapped storage.

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

    tensors: dict[str, torch.Tensor | MmapTensorDescriptor] = {}
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

        byte_start = data_offset + start
        byte_end = data_offset + end

        if lazy:
            # Store a lightweight descriptor — no torch.Tensor created.
            tensors[name] = MmapTensorDescriptor(
                storage, byte_start, byte_end, torch_dtype, shape
            )
        else:
            # Eagerly create a tensor backed by mmap'd sub-storage.
            sub_storage = storage[byte_start:byte_end]
            tensors[name] = torch.empty(0, dtype=torch_dtype).set_(sub_storage).reshape(shape)

    return tensors
