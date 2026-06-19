"""Unit tests for the static-cache Flash capability probe classifiers.

These cover the pure, CPU-only decision logic — the pre-#28958 reject
classifier and the known-answer output check — without needing a CUDA device
or running the ONNX probe session.
"""

from __future__ import annotations

import numpy as np
import pytest

from mobius._testing import ort_capabilities as cap


def test_notimplemented_is_always_expected_reject() -> None:
    # A NotImplemented from session.run is the confirmed pre-#28958 path and is
    # accepted regardless of message text.
    if cap._ort_capi is None:
        pytest.skip("onnxruntime pybind state unavailable")
    exc = cap._ort_capi.NotImplemented("any text at all")
    assert cap._is_expected_pre28958_reject(exc) is True


def test_fail_with_reject_signature_is_expected() -> None:
    # A Fail whose message carries the reject signature is treated as the
    # expected pre-#28958 reject.
    if cap._ort_capi is None:
        pytest.skip("onnxruntime pybind state unavailable")
    exc = cap._ort_capi.Fail(
        "Causal attention with TensorScatter (nonpad_kv_seqlen) and S_q != S_kv "
        "without past_key is not supported."
    )
    assert cap._is_expected_pre28958_reject(exc) is True


def test_genuine_fail_without_signature_is_not_expected() -> None:
    # A real Fail (e.g. CUDA OOM) must NOT be misclassified as "needs #28958",
    # so the probe surfaces it as a PROBE_ERROR instead of silently skipping.
    if cap._ort_capi is None:
        pytest.skip("onnxruntime pybind state unavailable")
    exc = cap._ort_capi.Fail("CUDA error: out of memory")
    assert cap._is_expected_pre28958_reject(exc) is False


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("graph uses nonpad_kv_seqlen", True),
        ("TensorScatter not supported", True),
        ("totally unrelated runtime failure", False),
    ],
)
def test_message_signature_matching_is_case_insensitive(message: str, expected: bool) -> None:
    # The fallback path (RuntimeError) relies purely on the message signature,
    # matched case-insensitively.
    assert cap._is_expected_pre28958_reject(RuntimeError(message)) is expected


def test_probe_output_is_correct_accepts_bottom_right_mean() -> None:
    # The spec-correct bottom-right kernel yields the mean of the value tags.
    correct = np.full(
        (cap._PROBE_BATCH, cap._PROBE_QUERY_LEN, cap._PROBE_HIDDEN),
        cap._PROBE_EXPECTED_OUTPUT,
        dtype=cap._PROBE_NP_DTYPE,
    )
    assert cap._probe_output_is_correct(correct) is True


def test_probe_output_is_correct_rejects_top_left_value() -> None:
    # A historical top-left kernel returns the first value tag, not the mean —
    # this must be rejected so a silent CPU fallback maps to NEEDS_FIX.
    wrong = np.full(
        (cap._PROBE_BATCH, cap._PROBE_QUERY_LEN, cap._PROBE_HIDDEN),
        cap._PROBE_VALUE_TAGS[0],
        dtype=cap._PROBE_NP_DTYPE,
    )
    assert cap._probe_output_is_correct(wrong) is False
