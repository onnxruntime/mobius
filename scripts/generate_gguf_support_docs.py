# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Refresh or check the generated GGUF support census documentation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobius.integrations.gguf._docs import DOC_PATH, update_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = update_document()
    current = DOC_PATH.read_text(encoding="utf-8")
    if args.check:
        if current != generated:
            raise SystemExit(
                "docs/api/build_from_gguf.md is stale; run "
                "`python scripts/generate_gguf_support_docs.py`"
            )
        return
    DOC_PATH.write_text(generated, encoding="utf-8")


if __name__ == "__main__":
    main()
