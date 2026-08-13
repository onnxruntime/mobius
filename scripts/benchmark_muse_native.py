#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark the native ORT GenAI path for the paired Muse workflow test."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("ORT_ENABLE_CUDNN_FLASH_ATTENTION", "0")

import onnxruntime as ort
import onnxruntime_genai as og

_CUDART = ctypes.CDLL("libcudart.so")


def _synchronize_cuda() -> None:
    status = _CUDART.cudaDeviceSynchronize()
    if status:
        raise RuntimeError(f"cudaDeviceSynchronize failed with status {status}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt(tokenizer: Any, text: str, *, include_image: bool) -> str:
    content: Any = [{"type": "image"}, {"type": "text", "text": text}]
    if not include_image:
        content = text
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)


def _run_once(
    model: Any,
    processor: Any,
    prompt: str,
    image: Any,
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    decode_skip: int,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    inputs = processor(prompt, images=image) if image is not None else processor(prompt)
    started = time.perf_counter()
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=prompt_tokens + max_new_tokens,
        min_length=prompt_tokens + max_new_tokens,
        do_sample=bool(sampling["do_sample"]),
        temperature=float(sampling["temperature"]),
        top_k=int(sampling["top_k"]),
        top_p=float(sampling["top_p"]),
    )
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    times: list[float] = []
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        generator.generate_next_token()
        _synchronize_cuda()
        times.append(time.perf_counter() - started)
        tokens.append(int(generator.get_next_tokens()[0]))
    if len(times) <= decode_skip:
        raise ValueError("max_new_tokens must exceed decode_skip")
    decode_seconds = times[-1] - times[decode_skip - 1]
    decode_tokens = len(times) - decode_skip
    return {
        "token_ids": tokens,
        "ttft_ms": times[0] * 1000,
        "decode_tokens": decode_tokens,
        "decode_seconds": decode_seconds,
        "throughput_tok_s": decode_tokens / decode_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmarks/muse_workflow_h200.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    workload = config["workload"]
    sampling = config["sampling"]

    if ort.__version__ != config["runtime"]["onnxruntime"]:
        raise RuntimeError(
            f"ORT version mismatch: {ort.__version__} != {config['runtime']['onnxruntime']}"
        )
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"CUDAExecutionProvider unavailable: {providers}")

    model = og.Model(str(args.model))
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    image_path = workload["image"]
    image = og.Images.open(image_path) if image_path else None
    prompt = (
        _prompt(tokenizer, workload["prompt"], include_image=True)
        if image is not None
        else workload["rendered_prompt"]
    )
    encoded_prompt = tokenizer.encode(prompt)
    prompt_ids = json.loads(Path(workload["prompt_ids_file"]).read_text())
    if encoded_prompt != prompt_ids:
        raise RuntimeError("native tokenizer output differs from the canonical prompt IDs")
    prompt_tokens = len(encoded_prompt)
    if prompt_tokens != int(workload["prompt_tokens"]):
        raise RuntimeError(
            f"prompt token count mismatch: {prompt_tokens} != {workload['prompt_tokens']}"
        )
    request_max_length = prompt_tokens + int(workload["max_new_tokens"])
    if request_max_length != int(workload["request_max_length"]):
        raise RuntimeError(
            "request max length mismatch: "
            f"{request_max_length} != {workload['request_max_length']}"
        )

    for _ in range(int(workload["warmups"])):
        _run_once(
            model,
            processor,
            prompt,
            image,
            prompt_tokens=prompt_tokens,
            max_new_tokens=int(workload["max_new_tokens"]),
            decode_skip=int(workload["decode_skip"]),
            sampling=sampling,
        )

    runs = [
        _run_once(
            model,
            processor,
            prompt,
            image,
            prompt_tokens=prompt_tokens,
            max_new_tokens=int(workload["max_new_tokens"]),
            decode_skip=int(workload["decode_skip"]),
            sampling=sampling,
        )
        for _ in range(int(workload["runs"]))
    ]
    reference = runs[0]["token_ids"]
    if any(run["token_ids"] != reference for run in runs[1:]):
        raise RuntimeError("native greedy token output changed across measured runs")

    record = {
        "kind": "native",
        "config": config,
        "environment": {
            "onnxruntime": ort.__version__,
            "onnxruntime_genai": getattr(og, "__version__", "unknown"),
            "providers": providers,
        },
        "package": {
            "metadata_sha256": _sha256(args.model / "inference_metadata.yaml"),
            "genai_config_sha256": _sha256(args.model / "genai_config.json"),
        },
        "prompt_tokens": prompt_tokens,
        "metrics": {
            "ttft_ms": statistics.median(run["ttft_ms"] for run in runs),
            "throughput_tok_s": statistics.median(run["throughput_tok_s"] for run in runs),
            "ttft_ms_runs": [run["ttft_ms"] for run in runs],
            "throughput_tok_s_runs": [run["throughput_tok_s"] for run in runs],
        },
        "token_ids": reference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
