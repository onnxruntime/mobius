# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the shared initializer dtype helper."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from mobius._passes._dtype_utils import initializer_dtype


def _value(dtype: ir.DataType | None, const: np.ndarray | None) -> ir.Value:
    v = ir.Value(name="w")
    if dtype is not None:
        v.dtype = dtype
    if const is not None:
        v.const_value = ir.tensor(const)
    return v


class TestInitializerDtype:
    def test_uses_declared_dtype_when_present(self):
        v = _value(ir.DataType.FLOAT16, np.ones((2,), np.float16))
        assert initializer_dtype(v) == ir.DataType.FLOAT16

    def test_falls_back_to_const_value_when_declared_missing(self):
        """The core fix: a dropped declared type must not hide the real dtype."""
        v = _value(None, np.ones((2,), np.float16))
        assert v.dtype is None
        assert initializer_dtype(v) == ir.DataType.FLOAT16

    def test_const_value_wins_on_disagreement(self):
        """Stale declared metadata must not override the serialized data dtype."""
        v = _value(ir.DataType.FLOAT, np.ones((2,), np.float16))
        assert initializer_dtype(v) == ir.DataType.FLOAT16

    def test_returns_none_when_nothing_available(self):
        v = _value(None, None)
        assert initializer_dtype(v) is None
