# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the static-cache Flash capability probe classifiers.

These cover the pure, CPU-only decision logic — the pre-#28958 reject
classifier, the caught-error → outcome mapping (NEEDS_FIX vs PROBE_ERROR), and
the known-answer output check — without needing a CUDA device or running the
ONNX probe session.
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


def test_classify_notimplemented_maps_to_needs_fix() -> None:
    # The confirmed pre-#28958 reject (NotImplemented) → NEEDS_FIX (fail-closed,
    # the static-cache suite SKIPs with the 'needs #28958' reason).
    if cap._ort_capi is None:
        pytest.skip("onnxruntime pybind state unavailable")
    exc = cap._ort_capi.NotImplemented("anything at all")
    assert cap._classify_run_error(exc) is cap._ProbeOutcome.NEEDS_FIX


def test_classify_signature_fail_maps_to_needs_fix() -> None:
    # A Fail carrying the reject signature is still the expected reject → NEEDS_FIX.
    if cap._ort_capi is None:
        pytest.skip("onnxruntime pybind state unavailable")
    exc = cap._ort_capi.Fail(
        "Causal attention with TensorScatter (nonpad_kv_seqlen) ... is not supported."
    )
    assert cap._classify_run_error(exc) is cap._ProbeOutcome.NEEDS_FIX


def test_classify_genuine_fail_maps_to_probe_error() -> None:
    # A real Fail whose message does NOT match the signature (e.g. CUDA OOM) must
    # map to PROBE_ERROR — NOT NEEDS_FIX — so it stays loud and does not silently
    # skip the whole suite by masquerading as 'needs #28958'.
    if cap._ort_capi is None:
        pytest.skip("onnxruntime pybind state unavailable")
    exc = cap._ort_capi.Fail("CUDA error: out of memory")
    assert cap._classify_run_error(exc) is cap._ProbeOutcome.PROBE_ERROR


def test_classify_runtimeerror_fallback_maps_to_probe_error() -> None:
    # The defensive RuntimeError fallback (no pybind types) without the signature
    # is also a genuine failure → PROBE_ERROR.
    assert (
        cap._classify_run_error(RuntimeError("unexpected segfault"))
        is cap._ProbeOutcome.PROBE_ERROR
    )


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


# --------------------------------------------------------------------------- #
# Fix ② observability, exercised end-to-end (not just at the classifier):     #
# drive _probe_static_cache_flash with a mocked ORT session so the            #
# PROBE_ERROR-vs-NEEDS_FIX decision and its WARNING log are covered on CPU,   #
# proving a genuine failure stays LOUD instead of masquerading as             #
# 'needs #28958' and silently skipping the suite.                             #
# --------------------------------------------------------------------------- #


class _FakeSession:
    """Minimal ``ort.InferenceSession`` stand-in whose ``run`` always raises."""

    def __init__(self, run_exc: BaseException) -> None:
        self._run_exc = run_exc

    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run(self, output_names, feeds):
        raise self._run_exc


@pytest.fixture
def probe_on_cuda(monkeypatch: pytest.MonkeyPatch):
    """Force the probe onto the CUDA path and feed it a session that raises.

    Yields an installer; clears the process-cached probe before and after so a
    mocked outcome never leaks into another test (or the real GPU probe).
    """

    def _install(run_exc: BaseException) -> None:
        monkeypatch.setattr(cap, "CUDA_AVAILABLE", True)
        cap._probe_static_cache_flash.cache_clear()
        monkeypatch.setattr(cap.ort, "InferenceSession", lambda *a, **k: _FakeSession(run_exc))

    yield _install
    cap._probe_static_cache_flash.cache_clear()


def test_probe_genuine_fail_is_loud_probe_error(
    probe_on_cuda, caplog: pytest.LogCaptureFixture
) -> None:
    # A signature-less Fail (e.g. CUDA OOM) raised by session.run must map to
    # PROBE_ERROR and log a WARNING — NOT a silent 'needs #28958' NEEDS_FIX skip.
    if cap._ort_capi is None:
        pytest.skip("onnxruntime pybind state unavailable")
    probe_on_cuda(cap._ort_capi.Fail("CUDA error: out of memory"))
    with caplog.at_level("WARNING", logger=cap.logger.name):
        outcome = cap._probe_static_cache_flash()
    assert outcome is cap._ProbeOutcome.PROBE_ERROR
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_probe_notimplemented_reject_is_needs_fix(probe_on_cuda) -> None:
    # Contrast: the genuine pre-#28958 NotImplemented reject still maps to
    # NEEDS_FIX end-to-end — the expected fail-closed skip, kept distinct from
    # the loud PROBE_ERROR above.
    if cap._ort_capi is None:
        pytest.skip("onnxruntime pybind state unavailable")
    probe_on_cuda(cap._ort_capi.NotImplemented("... is not supported."))
    assert cap._probe_static_cache_flash() is cap._ProbeOutcome.NEEDS_FIX
