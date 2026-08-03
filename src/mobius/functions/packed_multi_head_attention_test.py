# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the PackedMultiHeadAttention fallback ``ir.Function``.

These guard the formal-input arity/order of the standard-ONNX fallback for
``com.microsoft::PackedMultiHeadAttention``.  ORT's op has a 6-7 input
positional signature::

    query, key(opt), value(opt), bias(opt),
    token_offset, cumulative_sequence_length, attention_bias(opt)

Because ``token_offset`` / ``cumulative_sequence_length`` occupy positional
slots 5 and 6, the optional ``bias`` at slot 4 must exist as a formal input
even though the fallback body ignores it.  Call sites emit 6 inputs
``(q, k, v, "", token_offset, cu_seqlens)``; if the function declared only 5
formals, onnx-genai's function-inline admission rejects the call with a
FunctionArityMismatch (``call.input.len()=6 > func.input.len()=5``).
"""

from __future__ import annotations

from mobius.functions.packed_multi_head_attention import (
    packed_multi_head_attention,
)

EXPECTED_INPUT_ORDER = [
    "query",
    "key",
    "value",
    "bias",
    "token_offset",
    "cumulative_sequence_length",
]


def test_packed_mha_declares_six_positional_formal_inputs() -> None:
    func = packed_multi_head_attention()

    actual_order = [value.name for value in func.inputs]
    # Assert the full positional order, not just the count: the ``bias`` slot
    # must sit at index 3 (between ``value`` and ``token_offset``) to match the
    # ORT PackedMultiHeadAttention signature.
    assert actual_order == EXPECTED_INPUT_ORDER
    assert func.inputs[3].name == "bias"


def test_packed_mha_admits_six_input_call() -> None:
    func = packed_multi_head_attention()

    # A call site emits 6 inputs: (q, k, v, "", token_offset, cu_seqlens).
    # onnx-genai admits a call when len(call inputs) <= len(func inputs), so
    # the function must declare at least 6 formals for the call to be inlined.
    call_input_count = 6
    assert call_input_count <= len(func.inputs)


def test_packed_mha_function_identity() -> None:
    func = packed_multi_head_attention()

    assert func.domain == "com.microsoft"
    assert func.name == "PackedMultiHeadAttention"
