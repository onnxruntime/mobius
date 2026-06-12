# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the shared initializer dtype helper."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from mobius._passes._dtype_utils import initializer_dtype


def _value(dtype: ir.DataType | None, const: np.ndarray | None) -> ir.Value:
    # Built via the ir.Value constructor rather than the ir.val() factory: these
    # fixtures deliberately construct degenerate initializers (a dropped declared
    # type alongside a const_value, and a declared type that contradicts the
    # const_value dtype) that ir.val() validates against and refuses to build.
    return ir.Value(
        name="w",
        type=ir.TensorType(dtype) if dtype is not None else None,
        const_value=ir.tensor(const) if const is not None else None,
    )


class TestInitializerDtype:
    def test_uses_declared_dtype_when_present(self):
        v = _value(ir.DataType.FLOAT16, np.ones((2,), np.float16))
        assert initializer_dtype(v) == ir.DataType.FLOAT16

    def test_falls_back_to_const_value_when_declared_missing(self):
        """The core fix: a dropped declared type must not hide the real dtype."""
        v = _value(None, np.ones((2,), np.float16))
        assert v.dtype is None
        assert initializer_dtype(v) == ir.DataType.FLOAT16

    def test_raises_on_dtype_contradiction(self):
        """Reject values whose declared dtype contradicts the serialized data.

        Such a value is corrupt metadata and must fail closed rather than
        silently pick one dtype.
        """
        v = _value(ir.DataType.FLOAT, np.ones((2,), np.float16))
        with pytest.raises(ValueError) as excinfo:
            initializer_dtype(v)
        message = str(excinfo.value)
        assert "FLOAT" in message and "FLOAT16" in message
        assert "w" in message

    def test_returns_none_when_nothing_available(self):
        v = _value(None, None)
        assert initializer_dtype(v) is None
