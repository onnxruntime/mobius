#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run the paired ONNX GenAI workflow benchmark for Muse Glimmer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        "--runtime-repo",
        type=Path,
        default=Path(".contract-schema-latest"),
    )
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
    runtime_head = subprocess.run(
        ["git", "-C", str(args.runtime_repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    expected_head = config["runtime"]["onnxruntime_genai_commit"]
    if runtime_head != expected_head:
        raise ValueError(
            f"runtime source is {runtime_head}, paired benchmark requires {expected_head}"
        )
    git_index = Path(
        subprocess.run(
            [
                "git",
                "-C",
                str(args.runtime_repo),
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "index",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )
    if args.runner.stat().st_mtime < git_index.stat().st_mtime:
        raise ValueError(
            "runner predates the pinned runtime checkout; rebuild it before benchmarking"
        )
    if config["runtime"]["cudnn_flash_attention"]:
        raise ValueError("paired Muse benchmark requires cuDNN Flash Attention disabled")
    prompt_ids_path = Path(workload["prompt_ids_file"])
    prompt_ids = json.loads(prompt_ids_path.read_text())
    if len(prompt_ids) != int(workload["prompt_tokens"]):
        raise ValueError("prompt_ids_file length must equal prompt_tokens")
    if int(workload["prompt_tokens"]) + int(workload["max_new_tokens"]) != int(
        workload["request_max_length"]
    ):
        raise ValueError("request_max_length must equal prompt_tokens + max_new_tokens")
    if workload["stop_on_eos"] is not False:
        raise ValueError("paired Muse benchmark requires stop_on_eos=false")
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
        "--prompt-ids",
        str(prompt_ids_path),
    ]
    if workload["image"]:
        command.extend(["--image", workload["image"]])
    environment = os.environ.copy()
    environment["ONNX_GENAI_CUDA_GRAPH"] = "1"
    environment["ORT_ENABLE_CUDNN_FLASH_ATTENTION"] = "0"
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
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
        "runtime": {
            "source_head": runtime_head,
            "runner": str(args.runner.resolve()),
            "runner_sha256": _sha256(args.runner),
            "environment": {
                "ONNX_GENAI_CUDA_GRAPH": "1",
                "ORT_ENABLE_CUDNN_FLASH_ATTENTION": "0",
            },
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
