#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run the paired ONNX GenAI workflow benchmark for Muse Glimmer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmarks/muse_workflow_h200.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config: dict[str, Any] = json.loads(args.config.read_text())
    workload = config["workload"]
    sampling = config["sampling"]
    if sampling != {
        "algorithm": "greedy",
        "do_sample": False,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 0,
    }:
        raise ValueError("workflow runner currently requires the paired greedy policy")

    command = [
        str(args.runner),
        "--model",
        str(args.model),
        "--pipeline",
        "--backend",
        "ort",
        "--ep",
        "cuda",
        "--steady",
        "--tokens",
        str(workload["max_new_tokens"]),
        "--warmups",
        str(workload["warmups"]),
        "--runs",
        str(workload["runs"]),
        "--decode-skip",
        str(workload["decode_skip"]),
        "--prompt",
        workload["rendered_prompt"],
    ]
    if workload["image"]:
        command.extend(["--image", workload["image"]])
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    median = re.search(
        r"steady_median: prefill=([0-9.]+) ms decode=([0-9.]+) ms/token "
        r"throughput=([0-9.]+) tok/s",
        output,
    )
    tokens = re.search(r"generated_token_ids: (\[[^\n]+\])", output)
    if median is None or tokens is None:
        raise RuntimeError(f"workflow runner output is incomplete:\n{output}")

    record = {
        "kind": "workflow",
        "config": config,
        "package": {
            "metadata_sha256": _sha256(args.model / "inference_metadata.yaml"),
            "genai_config_sha256": _sha256(args.model / "genai_config.json"),
        },
        "metrics": {
            "ttft_ms": float(median.group(1)),
            "decode_ms_per_token": float(median.group(2)),
            "throughput_tok_s": float(median.group(3)),
        },
        "token_ids": json.loads(tokens.group(1)),
        "command": command,
        "runner_output": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
