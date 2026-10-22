#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Sample single-Engine GPU memory with exact validated-output checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from pathlib import Path

import onnxruntime_genai as og
import torch
from ornith_engine_benchmark import make_prompt, run_turn


def require_reference_output(result, expected, index):
    if result["output_ids"] != expected:
        raise RuntimeError(f"Output differs from validated reference at run {index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--decoder-file", type=Path, default=Path("decoder/model-packed.onnx"))
    parser.add_argument(
        "--arena-extend-strategy", choices=["kNextPowerOfTwo", "kSameAsRequested"]
    )
    parser.add_argument("--device-allocator-for-initializers", action="store_true")
    args = parser.parse_args()
    if args.context <= 0 or args.runs <= 0:
        parser.error("Require positive context and run counts")
    decoder = (args.model / args.decoder_file).resolve()
    if not decoder.is_relative_to(args.model.resolve()) or not decoder.is_file():
        parser.error("--decoder-file must identify a graph inside the model directory")
    reference = json.loads(args.reference_report.read_text())
    if reference["status"] != "passed" or reference["parity"]["status"] != "passed":
        parser.error("Reference report must have passed full greedy parity")
    cases = [case for case in reference["contexts"] if case["context_tokens"] == args.context]
    if len(cases) != 1:
        parser.error("Reference report must contain exactly one matching context")
    expected = cases[0]["runs"][0]["output_ids"]
    decoder_sha256 = hashlib.sha256(decoder.read_bytes()).hexdigest()
    if reference["decoder_sha256"] != decoder_sha256:
        parser.error("Reference report uses a different decoder graph")
    cuda_options = {"enable_cuda_graph": "1"}
    if args.arena_extend_strategy:
        cuda_options["arena_extend_strategy"] = str(
            ["kNextPowerOfTwo", "kSameAsRequested"].index(args.arena_extend_strategy)
        )
    overlay = {
        "model": {
            "decoder": {
                "filename": str(args.decoder_file),
                "session_options": {"provider_options": [{"cuda": cuda_options}]},
            }
        }
    }
    if args.device_allocator_for_initializers:
        overlay["model"]["decoder"]["session_options"][
            "session.use_device_allocator_for_initializers"
        ] = "1"
    speculative = reference.get("config_overlay", {}).get("speculative")
    if speculative is not None:
        overlay["speculative"] = speculative
    torch.cuda.init()
    stage, peaks, snapshots = "initial", {}, {}
    lock = threading.Lock()
    stopped = threading.Event()
    sampling_errors = []

    def sample():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        with lock:
            peaks[stage] = max(peaks.get(stage, 0), used)
        return used

    def sampling_loop():
        try:
            while not stopped.wait(0.01):
                sample()
        except Exception as error:
            sampling_errors.append(error)
            stopped.set()

    def mark(name):
        nonlocal stage
        torch.cuda.synchronize()
        snapshots[name] = sample()
        with lock:
            stage = name
        print(f"{name}: {snapshots[name] / 2**30:.3f} GiB", flush=True)

    report = {
        "status": "in_progress",
        "context_tokens": args.context,
        "generated_tokens": len(expected),
        "config_overlay": overlay,
        "decoder_sha256": decoder_sha256,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.mem_get_info()[1],
        "reference_report": str(args.reference_report.resolve()),
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "ORT_DISABLE_MOE_GEMV",
                "ORT_MOE_GEMV_FP32_ACCUM",
                "ORT_FPA_INTB_GEMM",
                "ORT_ENABLE_QMOE_INT2_PREFILL",
                "ORT_QMOE_INT2_PACKED_ONLY",
                "ORT_ENABLE_XQA",
                "ORT_PAGED_ATTENTION_DECODE_NUM_SPLITS",
            )
        },
        "sampling_interval_seconds": 0.01,
        "note": "Device-wide CUDA used memory, including Torch context; sampled, not exact allocator peak. Use an otherwise idle GPU.",
        "runs": [],
    }
    sampler = threading.Thread(target=sampling_loop)
    sampler.start()
    try:
        mark("before_model")
        config = og.Config(str(args.model))
        config.overlay(json.dumps(overlay))
        model = og.Model(config)
        mark("after_model")
        engine = og.Engine(model)
        mark("after_engine")
        prompt = make_prompt(reference["prompt_unit_token_ids"], args.context)
        for index in range(args.runs + 1):
            result = run_turn(engine, prompt, len(expected))
            require_reference_output(result, expected, index)
            report["runs"].append({"warmup": index == 0, **result})
            mark(f"after_run_{index}")
        report["output_matches_reference"] = True
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        stopped.set()
        sampler.join()
        report["stage_peak_device_memory_bytes"] = peaks
        report["stage_snapshot_device_memory_bytes"] = snapshots
        report["sampled_peak_device_memory_bytes"] = max(peaks.values(), default=0)
        report["runtime_library_sha256"] = {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libonnxruntime" in line
            }
        }
        if sampling_errors and report["status"] == "passed":
            report["status"] = "failed"
            report["error"] = f"GPU memory sampling failed: {sampling_errors[0]}"
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if sampling_errors and report.get("output_matches_reference"):
            raise RuntimeError("GPU memory sampling failed") from sampling_errors[0]


if __name__ == "__main__":
    main()
