# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for KV-cache I/O helpers in :mod:`mobius.tasks._cache_utils`."""

from __future__ import annotations

import logging

import onnx_ir as ir
import pytest

from mobius.tasks._base import _make_graph
from mobius.tasks._cache_utils import _register_kv_cache_outputs


def _present_pair(name: str, wrong_head_dim: int) -> tuple[ir.Value, ir.Value]:
    """A present key/value pair carrying a deliberately wrong inferred shape.

    Mimics ``com.microsoft::GroupQueryAttention`` shape inference, which
    mis-derives ``head_dim`` (e.g. 32 instead of 96) for the present outputs.
    """
    key = ir.Value(name=f"{name}_key")
    value = ir.Value(name=f"{name}_value")
    for v in (key, value):
        v.shape = ir.Shape(["batch", 32, "seq", wrong_head_dim])
        v.type = ir.TensorType(ir.DataType.FLOAT16)
    return key, value


def _dims(value: ir.Value) -> list[object]:
    return [d if isinstance(d, int) else str(d) for d in value.shape]


class TestRegisterKVCacheOutputs:
    def test_stamps_explicit_shapes_over_wrong_inference(self):
        """Explicit params must override mis-inferred present shapes (GQA bug)."""
        _, builder = _make_graph()
        pairs = [_present_pair("present.0", wrong_head_dim=32)]

        _register_kv_cache_outputs(
            builder,
            pairs,
            batch=ir.SymbolicDim("batch"),
            num_kv_heads=32,
            key_head_dim=96,
            value_head_dim=96,
            total_seq_len="past_sequence_len + sequence_len",
            dtype=ir.DataType.FLOAT16,
        )

        key, value = pairs[0]
        assert _dims(key) == ["batch", 32, "past_sequence_len + sequence_len", 96]
        assert _dims(value) == ["batch", 32, "past_sequence_len + sequence_len", 96]
        assert key.dtype == ir.DataType.FLOAT16
        assert value.dtype == ir.DataType.FLOAT16

    def test_distinct_key_value_head_dims(self):
        """MLA-style caches may use different key/value head dims."""
        _, builder = _make_graph()
        pairs = [_present_pair("present.0", wrong_head_dim=32)]

        _register_kv_cache_outputs(
            builder,
            pairs,
            batch=ir.SymbolicDim("batch"),
            num_kv_heads=16,
            key_head_dim=192,
            value_head_dim=128,
            total_seq_len="past_sequence_len + sequence_len",
            dtype=ir.DataType.FLOAT16,
        )

        key, value = pairs[0]
        assert _dims(key)[1] == 16 and _dims(key)[3] == 192
        assert _dims(value)[1] == 16 and _dims(value)[3] == 128

    def test_no_params_leaves_shapes_untouched(self, caplog):
        """Without shape params the helper must not stamp (inference path)."""
        _, builder = _make_graph()
        pairs = [_present_pair("present.0", wrong_head_dim=32)]

        with caplog.at_level(logging.WARNING, logger="mobius.tasks._cache_utils"):
            _register_kv_cache_outputs(builder, pairs)

        key, _ = pairs[0]
        # Unchanged: still the (wrong) pre-existing inferred shape.
        assert _dims(key) == ["batch", 32, "seq", 32]
        # Opting out (zero params) is intentional and must stay silent.
        assert caplog.text == ""

    def test_partial_params_raise(self):
        """A partial present-shape set is a wiring slip -> must raise.

        This is the fail-closed regression proof: the exact input that
        previously warned-and-proceeded (shipping a structurally-wrong model
        with mis-derived GroupQueryAttention ``head_dim``) now raises before
        any output is registered.
        """
        _, builder = _make_graph()
        pairs = [_present_pair("present.0", wrong_head_dim=32)]

        with pytest.raises(ValueError, match="partial set of present-shape parameters"):
            _register_kv_cache_outputs(
                builder,
                pairs,
                batch=ir.SymbolicDim("batch"),
                num_kv_heads=32,
                # key_head_dim / value_head_dim / total_seq_len / dtype omitted
            )

        # The message must name every omitted parameter so the slip is diagnosable.
        with pytest.raises(ValueError) as exc:
            _register_kv_cache_outputs(
                builder,
                pairs,
                batch=ir.SymbolicDim("batch"),
                num_kv_heads=32,
            )
        for missing in ("key_head_dim", "value_head_dim", "total_seq_len", "dtype"):
            assert missing in str(exc.value)

    def test_registers_named_outputs(self):
        """Outputs are registered with the conventional present.{i}.* names."""
        _, builder = _make_graph()
        pairs = [_present_pair("a", 32), _present_pair("b", 32)]

        _register_kv_cache_outputs(builder, pairs)

        names = [v.name for v in builder.graph.outputs]
        assert names == [
            "present.0.key",
            "present.0.value",
            "present.1.key",
            "present.1.value",
        ]
