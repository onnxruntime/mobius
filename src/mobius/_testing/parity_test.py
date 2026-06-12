# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the parity comparison utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mobius._testing.parity import (
    ParityReport,
    ParityResult,
    compare_golden,
    compare_synthetic,
)


class TestCompareSynthetic:
    """Tests for L3 synthetic parity comparison."""

    def test_identical_logits_pass(self):
        logits = np.random.default_rng(42).standard_normal((1, 100)).astype(np.float32)
        report = compare_synthetic(logits, logits.copy())
        assert report.result == ParityResult.PASS
        assert report.max_abs_diff == pytest.approx(0.0, abs=1e-12)
        assert report.cosine_similarity == pytest.approx(1.0, abs=1e-6)

    def test_small_diff_passes(self):
        rng = np.random.default_rng(42)
        hf = rng.standard_normal((1, 100)).astype(np.float32)
        onnx = hf + rng.uniform(-1e-4, 1e-4, hf.shape).astype(np.float32)
        report = compare_synthetic(onnx, hf, atol=1e-3, rtol=1e-3)
        assert report.result == ParityResult.PASS

    def test_large_diff_fails(self):
        rng = np.random.default_rng(42)
        hf = rng.standard_normal((1, 100)).astype(np.float32)
        onnx = hf + 1.0  # large shift
        report = compare_synthetic(onnx, hf, atol=1e-3, rtol=1e-3)
        assert report.result == ParityResult.FAIL
        assert report.max_abs_diff > 0.5

    def test_3d_logits_handled(self):
        """3-D (batch, seq, vocab) input extracts last-token metrics."""
        rng = np.random.default_rng(42)
        hf = rng.standard_normal((1, 5, 100)).astype(np.float32)
        report = compare_synthetic(hf, hf.copy())
        assert report.result == ParityResult.PASS

    def test_near_tie_diagnostic(self):
        """Near-tie adds diagnostic message but does not change result."""
        # Build logits where top-1 and top-2 are very close
        hf = np.zeros((1, 100), dtype=np.float32)
        hf[0, 10] = 1.0
        hf[0, 20] = 1.0 + 1e-4  # gap < 0.01 margin
        report = compare_synthetic(hf, hf.copy())
        assert report.near_tie is True
        assert "near-tie" in report.message

    def test_shape_mismatch_raises(self):
        a = np.zeros((1, 10), dtype=np.float32)
        b = np.zeros((1, 20), dtype=np.float32)
        with pytest.raises(AssertionError, match="Shape mismatch"):
            compare_synthetic(a, b)

    def test_report_is_dataclass(self):
        logits = np.zeros((1, 10), dtype=np.float32)
        report = compare_synthetic(logits, logits)
        assert isinstance(report, ParityReport)
        assert report.level == "L3"


class TestCompareGolden:
    """Tests for L4 golden comparison."""

    def test_argmax_match_passes(self):
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 42] = 10.0
        logits[0, 7] = 5.0
        report = compare_golden(
            logits,
            golden_top1_id=42,
            golden_top2_id=7,
            golden_top10_ids=list(range(10)),
        )
        assert report.result == ParityResult.PASS

    def test_argmax_mismatch_fails(self):
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 42] = 10.0
        report = compare_golden(
            logits,
            golden_top1_id=99,
            golden_top2_id=7,
            golden_top10_ids=[99, 7],
        )
        assert report.result == ParityResult.FAIL

    def test_near_tie_ambiguous(self):
        """Mismatch + near-tie + matches top2 → AMBIGUOUS."""
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 7] = 10.0
        logits[0, 42] = 10.0 - 1e-4  # near-tie: gap < 0.01
        # ONNX picks 7, golden top1=42, top2=7
        report = compare_golden(
            logits,
            golden_top1_id=42,
            golden_top2_id=7,
            golden_top10_ids=[42, 7],
        )
        assert report.result == ParityResult.AMBIGUOUS

    def test_3d_input(self):
        logits = np.zeros((1, 3, 50), dtype=np.float32)
        logits[0, -1, 5] = 10.0
        report = compare_golden(
            logits,
            golden_top1_id=5,
            golden_top2_id=0,
            golden_top10_ids=[5, 0],
        )
        assert report.result == ParityResult.PASS
        assert report.level == "L4"

    def test_high_jaccard_argmax_swap_ambiguous(self):
        """Top-10 identical, argmax swapped beyond near-tie margin → AMBIGUOUS.

        Models the phi4mm single-image CUDA float32 near-tie: the predicted
        token and golden top1 are swapped but their logit gap (~0.09) exceeds
        the float32 near_tie margin (0.01), while the full top-10 set matches.
        """
        top10 = [976, 32, 637, 5632, 2223, 2500, 3160, 51543, 6275, 2886]
        logits = np.full((1, 200064), -50.0, dtype=np.float32)
        # Assign descending logits to the golden top-10 ...
        for rank, tok in enumerate(top10):
            logits[0, tok] = 37.7 - rank * 0.5
        # ... but bump token 32 just above 976 so ONNX argmax is 32 (golden
        # top2), with a gap (~0.09) larger than the 0.01 float32 margin.
        logits[0, 32] = 37.76
        logits[0, 976] = 37.67
        report = compare_golden(
            logits,
            golden_top1_id=976,
            golden_top2_id=32,
            golden_top10_ids=top10,
        )
        assert report.result == ParityResult.AMBIGUOUS

    def test_low_jaccard_argmax_mismatch_still_fails(self):
        """Genuine divergence (low top-10 overlap) stays FAIL.

        Models phi4mm multi-image-audio: ONNX argmax equals the golden top2
        but the top-10 sets barely overlap (jaccard ≈ 0.3), so the Jaccard
        guard must NOT downgrade it to AMBIGUOUS.
        """
        golden_top10 = [38229, 976, 13145, 637, 2790, 19048, 1385, 15390, 5632, 90522]
        # ONNX strongly prefers a disjoint set of tokens; only 976 and 637
        # overlap with the golden top-10 (low jaccard).
        onnx_top = [976, 111111, 222222, 333333, 444444, 637, 555555, 666666, 777777, 888888]
        logits = np.full((1, 1000000), -50.0, dtype=np.float32)
        for rank, tok in enumerate(onnx_top):
            logits[0, tok] = 35.0 - rank * 0.5
        report = compare_golden(
            logits,
            golden_top1_id=38229,
            golden_top2_id=976,
            golden_top10_ids=golden_top10,
        )
        assert report.result == ParityResult.FAIL

    def test_nine_of_ten_overlap_tie_break_ambiguous(self):
        """Exactly 9/10 top-10 overlap with a #1/#2 swap → AMBIGUOUS.

        Boundary case for the count-based overlap gate: 9 of the golden top-10
        agree (Jaccard would be only 9/11 ≈ 0.818, below the old 0.9 ratio, so
        this case used to FAIL incorrectly).  The golden argmax stays in the
        ONNX top-2 and the predicted token is itself a golden top-10 token, so
        it is a genuine tie-break swap.
        """
        golden_top10 = [976, 32, 637, 5632, 2223, 2500, 3160, 51543, 6275, 2886]
        logits = np.full((1, 200064), -50.0, dtype=np.float32)
        for rank, tok in enumerate(golden_top10):
            logits[0, tok] = 37.7 - rank * 0.5
        # Swap #1/#2 beyond the float32 near-tie margin: ONNX argmax is 32
        # (golden top2), golden top1 (976) remains ONNX #2.
        logits[0, 32] = 37.76
        logits[0, 976] = 37.67
        # Replace one golden top-10 token (2886) with a non-golden token so the
        # overlap is exactly 9/10 (not identical sets).
        logits[0, 2886] = -50.0
        logits[0, 199999] = 30.0
        report = compare_golden(
            logits,
            golden_top1_id=976,
            golden_top2_id=32,
            golden_top10_ids=golden_top10,
        )
        assert report.result == ParityResult.AMBIGUOUS

    def test_identical_top10_low_token_promoted_still_fails(self):
        """Identical top-10 but golden argmax buried in ONNX → FAIL.

        Guards against the over-lenient case flagged in review: even with a
        perfect top-10 overlap, if the golden top1 is NOT in the ONNX top-2
        (a low-ranked token was promoted to #1 with a large gap), this is a
        real divergence and must stay FAIL — not be masked as AMBIGUOUS.
        """
        golden_top10 = [976, 32, 637, 5632, 2223, 2500, 3160, 51543, 6275, 2886]
        logits = np.full((1, 200064), -50.0, dtype=np.float32)
        # ONNX ranks the SAME 10 tokens but in a very different order: token
        # 2886 (golden #10) is promoted to #1 with a large gap, while golden #1
        # (976) is pushed down to ONNX rank ~5 (outside the top-2).
        onnx_order = [2886, 32, 637, 5632, 976, 2500, 3160, 51543, 6275, 2223]
        for rank, tok in enumerate(onnx_order):
            logits[0, tok] = 40.0 - rank * 1.0
        report = compare_golden(
            logits,
            golden_top1_id=976,
            golden_top2_id=32,
            golden_top10_ids=golden_top10,
        )
        assert report.result == ParityResult.FAIL
