# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the strict unified diff applier."""

from __future__ import annotations

import difflib
import unittest

from mobius.upstream_patches import PatchError, apply_unified_diff

_ORIGINAL = "alpha\nbravo\ncharlie\ndelta\n"

_DIFF = """--- a/f
+++ b/f
@@ -1,4 +1,4 @@
 alpha
-bravo
+BRAVO
 charlie
 delta
"""


class ApplyUnifiedDiffTest(unittest.TestCase):
    def test_applies_a_single_hunk(self):
        self.assertEqual(
            apply_unified_diff(_ORIGINAL, _DIFF),
            "alpha\nBRAVO\ncharlie\ndelta\n",
        )

    def test_applies_two_hunks(self):
        original = "".join(f"line{index}\n" for index in range(1, 21))
        modified = original.replace("line2\n", "two\n").replace("line18\n", "eighteen\n")
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                "a/f",
                "b/f",
            )
        )
        self.assertEqual(diff.count("@@ -"), 2)
        self.assertEqual(apply_unified_diff(original, diff), modified)

    def test_rejects_a_context_mismatch(self):
        with self.assertRaises(PatchError) as caught:
            apply_unified_diff("alpha\nBRAVO\ncharlie\ndelta\n", _DIFF)
        self.assertIn("context mismatch", str(caught.exception))

    def test_rejects_a_hunk_past_the_end_of_the_file(self):
        with self.assertRaises(PatchError):
            apply_unified_diff("alpha\n", _DIFF)

    def test_rejects_a_diff_without_hunks(self):
        with self.assertRaises(PatchError):
            apply_unified_diff(_ORIGINAL, "--- a/f\n+++ b/f\n")

    def test_preserves_a_missing_trailing_newline(self):
        original = "alpha\nbravo"
        diff = "@@ -1,2 +1,2 @@\n-alpha\n+ALPHA\n bravo\n\\ No newline at end of file\n"
        self.assertEqual(apply_unified_diff(original, diff), "ALPHA\nbravo")

    def test_handles_an_empty_context_line(self):
        original = "alpha\n\ncharlie\n"
        diff = "@@ -1,3 +1,3 @@\n alpha\n\n-charlie\n+CHARLIE\n"
        self.assertEqual(apply_unified_diff(original, diff), "alpha\n\nCHARLIE\n")


if __name__ == "__main__":
    unittest.main()
