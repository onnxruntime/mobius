# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF file reading and metadata/tensor extraction.

Wraps the ``gguf`` package's :class:`GGUFReader` to provide a clean
interface for reading GGUF model files. The main class is
:class:`GGUFModel`, which parses a ``.gguf`` file into structured
metadata and provides lazy tensor access with dequantization.

Example::

    from mobius.integrations.gguf._reader import GGUFModel

    model = GGUFModel("path/to/model.gguf")
    print(model.architecture)   # 'llama'
    print(model.metadata)       # {'llama.embedding_length': 4096, ...}

    for name, array in model.tensor_items():
        print(name, array.shape, array.dtype)
"""

from __future__ import annotations

__all__ = ["GGUFModel"]

import logging
from array import array
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _parse_field_value(field) -> Any:
    """Extract a Python value from a :class:`gguf.ReaderField`.

    Handles scalars (int, float, bool), strings, and arrays.
    Mirrors the logic in ``transformers.integrations.ggml._gguf_parse_value``.
    """
    from gguf import GGUFValueType

    types = field.types
    if not types:
        return None

    first_type = types[0]

    # Array: first type is ARRAY, element type is types[1]
    if first_type == GGUFValueType.ARRAY:
        element_type = types[1] if len(types) > 1 else None
        return _parse_array(field.data, field.parts, element_type)

    # Scalar / string
    return _parse_scalar(field.parts, first_type)


def _parse_scalar(parts: list, value_type) -> Any:
    """Parse a single scalar value from field parts."""
    from gguf import GGUFValueType

    # parts[-1] holds the value data as a numpy array
    value_data = parts[-1]

    if value_type in (
        GGUFValueType.UINT8,
        GGUFValueType.INT8,
        GGUFValueType.UINT16,
        GGUFValueType.INT16,
        GGUFValueType.UINT32,
        GGUFValueType.INT32,
        GGUFValueType.UINT64,
        GGUFValueType.INT64,
    ):
        return int(value_data[0])
    if value_type in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
        return float(value_data[0])
    if value_type == GGUFValueType.BOOL:
        return bool(value_data[0])
    if value_type == GGUFValueType.STRING:
        return array("B", list(value_data)).tobytes().decode()

    return value_data


def _parse_array(data: list[int], parts: list, element_type) -> list[Any]:
    """Parse an array field into a Python list."""
    from gguf import GGUFValueType

    if element_type == GGUFValueType.STRING:
        # String arrays: `data` holds the value-part indices (skipping the field
        # key name and length-prefix parts), exactly like the numeric path below.
        # Iterating parts directly would wrongly include the field key name as the
        # first element, shifting every string by one.
        result = []
        for idx in data:
            part = parts[idx]
            try:
                result.append(array("B", list(part)).tobytes().decode())
            except UnicodeDecodeError:
                result.append(
                    array("B", list(part)).tobytes().decode("utf-8", errors="replace")
                )
        return result

    # Numeric arrays: data indices point into the parts list
    if element_type in (
        GGUFValueType.UINT8,
        GGUFValueType.INT8,
        GGUFValueType.UINT16,
        GGUFValueType.INT16,
        GGUFValueType.UINT32,
        GGUFValueType.INT32,
        GGUFValueType.UINT64,
        GGUFValueType.INT64,
    ):
        return [int(parts[idx][0]) for idx in data]
    if element_type in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
        return [float(parts[idx][0]) for idx in data]
    if element_type == GGUFValueType.BOOL:
        return [bool(parts[idx][0]) for idx in data]

    return [parts[idx] for idx in data]


class GGUFModel:
    """Parsed GGUF file with metadata and tensor access.

    This class reads a ``.gguf`` file and provides:

    - :attr:`architecture` — the model architecture name (e.g. ``'llama'``)
    - :attr:`metadata` — all GGUF key-value metadata as a Python dict
    - :meth:`tensor_items` — iterate over ``(name, numpy_array)`` pairs
    - :meth:`get_tensor` — get a single dequantized tensor by name

    Tensors are dequantized lazily via ``gguf.dequantize()``. For
    unquantized types (F32, F16), the raw data is returned directly.
    BF16 tensors go through ``dequantize()`` and are returned as
    float32 (numpy has no native bfloat16 dtype).

    Args:
        path: Path to the ``.gguf`` file.
    """

    def __init__(self, path: str | Path) -> None:
        try:
            from gguf import GGUFReader
        except ImportError as e:
            raise ImportError(
                "The 'gguf' package is required for GGUF support. "
                "Install it with: pip install gguf"
            ) from e

        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"GGUF file not found: {self._path}")

        self._reader = GGUFReader(str(self._path))
        self._metadata: dict[str, Any] | None = None
        # Build tensor name → index map for O(1) lookup
        self._tensor_index: dict[str, int] = {
            t.name: i for i, t in enumerate(self._reader.tensors)
        }

    @property
    def architecture(self) -> str:
        """Model architecture name (e.g. ``'llama'``, ``'qwen2'``)."""
        field = self._reader.get_field("general.architecture")
        if field is None:
            raise ValueError("GGUF file missing 'general.architecture' metadata key")
        return _parse_field_value(field)

    @property
    def metadata(self) -> dict[str, Any]:
        """All GGUF metadata key-value pairs as a Python dict.

        Keys are the full GGUF key strings (e.g.
        ``'llama.embedding_length'``). Values are parsed into
        native Python types (int, float, str, bool, list).
        """
        if self._metadata is None:
            self._metadata = {}
            for key, field in self._reader.fields.items():
                try:
                    self._metadata[key] = _parse_field_value(field)
                except Exception:
                    logger.debug("Failed to parse GGUF field '%s'", key)
        return self._metadata

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a single metadata value by key.

        Args:
            key: The full GGUF metadata key
                (e.g. ``'llama.embedding_length'``).
            default: Value to return if the key is not found.

        Returns:
            The parsed metadata value, or *default* if not found.
        """
        field = self._reader.get_field(key)
        if field is None:
            return default
        try:
            return _parse_field_value(field)
        except Exception:
            logger.debug("Failed to parse GGUF field '%s'", key)
            return default

    @property
    def tensor_names(self) -> list[str]:
        """List of all tensor names in the GGUF file."""
        return list(self._tensor_index.keys())

    @property
    def num_tensors(self) -> int:
        """Number of tensors in the GGUF file."""
        return len(self._tensor_index)

    def reader_tensors(self):
        """Underlying ``gguf.ReaderTensor`` records (``name``/``shape``/``tensor_type``).

        Exposed so callers that previously reached into the private
        ``_reader.tensors`` list keep working across a single file *and* a
        multi-shard :class:`~mobius.integrations.gguf._shard_set.GgufShardSet`,
        which yields the records of every shard in order.
        """
        return list(self._reader.tensors)

    @property
    def is_little_endian(self) -> bool:
        """Whether raw tensor payloads use ONNX-compatible little-endian bytes."""
        return getattr(self._reader.endianess, "name", None) == "LITTLE"

    def _dequantize_tensor(self, tensor) -> np.ndarray:
        """Dequantize a single :class:`ReaderTensor` to a numpy array.

        For unquantized types (F32, F16), returns the raw data reshaped
        to numpy row-major order. For all other types (quantized and
        BF16), delegates to ``gguf.dequantize()`` which returns float32.

        Args:
            tensor: A ``gguf.ReaderTensor`` from the underlying reader.

        Returns:
            Numpy array in row-major shape order.
        """
        # GGUF stores shape in GGML (column-major) order; reverse
        # to get numpy (row-major) shape.
        np_shape = tuple(reversed(tensor.shape))
        return self.dequantize_raw_tensor(tensor.data, tensor.tensor_type, np_shape)

    def dequantize_raw_tensor(
        self,
        raw_data: np.ndarray,
        quant_type: Any,
        np_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Dequantize raw GGUF tensor data into its logical numpy shape.

        K-quant blocks are contiguous over the flattened tensor and may cross
        logical row boundaries. Delegating the complete raw array to gguf's
        reference dequantizer before reshaping preserves that layout.
        """
        from gguf import GGMLQuantizationType, dequantize

        if quant_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            return raw_data.reshape(np_shape)
        # All other types (quantized, BF16) go through dequantize().
        # BF16 is intentionally NOT in the fast path above because
        # numpy has no native bfloat16 dtype — the gguf library's
        # dequantize() converts BF16 bytes to float32.
        return dequantize(raw_data, quant_type).reshape(np_shape)

    def tensor_items(self) -> Iterator[tuple[str, np.ndarray]]:
        """Iterate over ``(name, dequantized_array)`` pairs.

        Each tensor is dequantized to float32 (for quantized and BF16
        types) or returned as-is (for F32/F16). Shapes follow the
        numpy (row-major) convention — GGUF's reversed shape is
        corrected automatically.

        Yields:
            Tuples of ``(tensor_name, numpy_array)``.
        """
        for tensor in self._reader.tensors:
            yield tensor.name, self._dequantize_tensor(tensor)

    def tensor_items_raw(
        self,
    ) -> Iterator[tuple[str, np.ndarray, Any, tuple[int, ...]]]:
        """Iterate over raw tensor data without dequantization.

        Used by the quantized import path to access raw GGUF block
        data for repacking into MatMulNBits format.

        Yields:
            Tuples of ``(name, raw_data, quant_type, np_shape)`` where:

            - *name*: GGUF tensor name
            - *raw_data*: Raw tensor storage. Quantized K-blocks are
              contiguous over the flattened logical tensor and are not
              independently padded at row boundaries.
            - *quant_type*: :class:`gguf.GGMLQuantizationType` value
            - *np_shape*: Logical shape in numpy (row-major) order
        """
        for tensor in self._reader.tensors:
            np_shape = tuple(reversed(tensor.shape))
            yield tensor.name, tensor.data, tensor.tensor_type, np_shape

    def get_tensor(self, name: str) -> np.ndarray:
        """Get a single dequantized tensor by name.

        Args:
            name: The GGUF tensor name
                (e.g. ``'blk.0.attn_q.weight'``).

        Returns:
            The dequantized tensor as a numpy array.

        Raises:
            KeyError: If the tensor name is not found.
        """
        if name not in self._tensor_index:
            raise KeyError(
                f"Tensor '{name}' not found in GGUF file. "
                f"Available: {self.tensor_names[:10]}..."
            )
        tensor = self._reader.get_tensor(self._tensor_index[name])
        return self._dequantize_tensor(tensor)

    def get_tensor_type(self, name: str) -> Any:
        """Get the quantization type of a tensor.

        Args:
            name: The GGUF tensor name.

        Returns:
            The :class:`gguf.GGMLQuantizationType` enum value.

        Raises:
            KeyError: If the tensor name is not found.
        """
        if name not in self._tensor_index:
            raise KeyError(f"Tensor '{name}' not found in GGUF file.")
        return self._reader.get_tensor(self._tensor_index[name]).tensor_type

    def tensor_storage_range(self, name: str) -> tuple[int, int, str]:
        """Return the exact file offset, byte length, and qtype for a tensor."""
        if name not in self._tensor_index:
            raise KeyError(f"Tensor '{name}' not found in GGUF file.")
        tensor = self._reader.get_tensor(self._tensor_index[name])
        return (
            int(tensor.data_offset),
            int(tensor.n_bytes),
            str(getattr(tensor.tensor_type, "name", tensor.tensor_type)),
        )

    def __repr__(self) -> str:
        arch = self.architecture if self._reader else "?"
        n_tensors = len(self._tensor_index)
        return f"GGUFModel(path='{self._path}', arch='{arch}', tensors={n_tensors})"
