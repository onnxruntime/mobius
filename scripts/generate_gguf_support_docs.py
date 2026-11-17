# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Refresh or check the generated GGUF capability catalog and evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobius.integrations.gguf._docs import DOC_PATH, update_document
from mobius.integrations.gguf._quant_capabilities import (
    CAPABILITY_MATRIX_PATH,
    render_quantization_capability_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = update_document()
    current = DOC_PATH.read_text(encoding="utf-8")
    capability_matrix = render_quantization_capability_matrix()
    current_capability_matrix = (
        CAPABILITY_MATRIX_PATH.read_text(encoding="utf-8")
        if CAPABILITY_MATRIX_PATH.exists()
        else ""
    )
    if args.check:
        if current != generated or current_capability_matrix != capability_matrix:
            raise SystemExit(
                "GGUF capability catalog or generated evidence is stale; run "
                "`python scripts/generate_gguf_support_docs.py`"
            )
        return
    DOC_PATH.write_text(generated, encoding="utf-8")
    CAPABILITY_MATRIX_PATH.write_text(capability_matrix, encoding="utf-8")


if __name__ == "__main__":
    main()
