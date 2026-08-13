#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compare equivalent native and metadata-driven workflow benchmark records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mobius.integrations.onnx_genai.performance import compare_performance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--max-regression-percent", type=float, default=5.0)
    args = parser.parse_args()

    with args.workflow.open(encoding="utf-8") as handle:
        workflow = json.load(handle)
    with args.native.open(encoding="utf-8") as handle:
        native = json.load(handle)
    result = compare_performance(
        workflow,
        native,
        max_regression_fraction=args.max_regression_percent / 100,
    )
    for observation in result.observations:
        print(f"MEASURED: {observation}")
    for failure in result.failures:
        print(f"FAIL: {failure}")
    if result.passed:
        print("PASS: workflow performance is competitive with native")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
