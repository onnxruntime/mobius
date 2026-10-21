#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Batch-one native Engine benchmark; not a model accuracy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import torch


def make_prompt(unit, context):
    unit = list(unit)
    if not unit or context <= 0:
        raise ValueError("Require nonempty prompt tokens and a positive context length")
    return np.asarray((unit * ((context + len(unit) - 1) // len(unit)))[:context], np.int32)


def run_turn(engine, prompt, output_tokens, *, require_mtp=True):
    before = engine.get_speculative_stats()
    request = engine.create_request()
    options = og.TurnOptions(request)
    options.set_do_sample(False)
    options.set_min_generated_tokens(output_tokens)
    options.set_max_generated_tokens(output_tokens)
    buffer = engine.create_event_buffer(64)
    tokens, first, finished = [], None, False
    started = time.perf_counter()
    try:
        request.begin_turn(np.asarray(prompt, np.int32), options)
        steps = 0
        while engine.has_pending_requests():
            for event in engine.run(buffer):
                now = time.perf_counter()
                if event.flags & og.EngineEventFlags.FAILED:
                    if event.error_code == og.EngineErrorCode.ENGINE_EXECUTION_FAILURE:
                        # A subsequent Run rethrows the Engine's durable exception,
                        # including the original ORT error rather than only an enum.
                        engine.run(buffer)
                    raise RuntimeError(f"Engine turn failed: {event.error_code}")
                if event.flags & og.EngineEventFlags.CAPACITY_BLOCKED:
                    raise RuntimeError("Engine turn is capacity blocked")
                if event.flags & og.EngineEventFlags.TOKEN:
                    if first is None:
                        first = now
                    tokens.append(event.token)
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    finished = True
            steps += 1
            if steps > len(prompt) + 4 * output_tokens + 32:
                raise RuntimeError("Engine exceeded the bounded-progress limit")
        ended = time.perf_counter()
        if not finished or len(tokens) != output_tokens or first is None:
            raise RuntimeError(
                f"Expected {output_tokens} generated tokens; received {len(tokens)}"
            )
        after = engine.get_speculative_stats()
        counters = {
            name: after[name] - before[name]
            for name in (
                "draft_tokens_proposed",
                "draft_tokens_evaluated",
                "draft_tokens_accepted",
                "mtp_failures",
                "draft_forward_passes",
                "target_forward_passes",
                "zero_accept_rounds",
                "partial_accept_rounds",
                "full_accept_rounds",
            )
        }
        if require_mtp and (
            counters["draft_tokens_proposed"] <= 0 or counters["draft_tokens_evaluated"] <= 0
        ):
            raise RuntimeError(f"MTP did not execute: {counters}")
        if counters["mtp_failures"]:
            raise RuntimeError(f"MTP reported failures: {counters}")
        return {
            "context_tokens": len(prompt),
            "generated_tokens": len(tokens),
            "ttft_seconds": first - started,
            "total_seconds": ended - started,
            "decode_tokens_per_second": (len(tokens) - 1) / (ended - first),
            "end_to_end_output_tokens_per_second": len(tokens) / (ended - started),
            "speculative": counters,
            "output_ids": tokens,
        }
    finally:
        request.close()


def greedy_parity(engine, reference_engine, prompt, output_tokens=64):
    speculative = run_turn(engine, prompt, output_tokens)
    reference = run_turn(reference_engine, prompt, output_tokens, require_mtp=False)
    if reference["speculative"]["draft_tokens_proposed"]:
        raise RuntimeError("Parity reference unexpectedly enabled MTP")
    mismatch = next(
        (
            index
            for index, (actual, expected) in enumerate(
                zip(speculative["output_ids"], reference["output_ids"], strict=True)
            )
            if actual != expected
        ),
        None,
    )
    return {
        "status": "passed" if mismatch is None else "failed",
        "first_mismatch_index": mismatch,
        "speculative": speculative,
        "target_only": reference,
    }


def save_report(model_path, report_path, report, *, update_manifest=True):
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not update_manifest:
        return
    manifest_path = model_path / "ornith_engine_export.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["engine_validated"] = (
        report["status"] == "passed" and report["parity"]["status"] == "passed"
    )
    if manifest["engine_validated"]:
        manifest.pop("correctness_blocker", None)
    elif report["status"] == "failed_parity":
        manifest["correctness_blocker"] = (
            "MTP versus target-only greedy parity failed at output index "
            f"{report['parity']['first_mismatch_index']}"
        )
    manifest["engine_validation_report"] = str(report_path.resolve())
    manifest["accuracy_test"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192, 16384, 32768])
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--draft-tokens", type=int, help="Override the fixed MTP proposal width"
    )
    parser.add_argument(
        "--decoder-file",
        type=Path,
        help="Alternative decoder graph relative to the model directory",
    )
    parser.add_argument(
        "--cuda-graph", action="store_true", help="Enable native Engine CUDA graph capture"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Save an experimental report without replacing the export's validation record",
    )
    args = parser.parse_args()
    disk_config = json.loads((args.model / "genai_config.json").read_text())
    if args.draft_tokens is not None and not (
        0 < args.draft_tokens < disk_config["model"]["decoder"]["state_update_capacity"]
    ):
        parser.error("--draft-tokens must fit the decoder's speculative state capacity")
    if args.decoder_file is not None:
        if args.decoder_file.is_absolute() or not (
            args.model / args.decoder_file
        ).resolve().is_relative_to(args.model.resolve()):
            parser.error("--decoder-file must stay inside the model directory")
        if not (args.model / args.decoder_file).is_file():
            parser.error("--decoder-file does not exist")
    if (
        any(n <= 0 for n in args.contexts)
        or args.runs <= 0
        or args.warmup < 0
        or args.output_tokens < 2
    ):
        parser.error(
            "Require positive contexts/runs, nonnegative warmup, and at least 2 output tokens"
        )
    print("Loading native GenAI Engine", flush=True)
    config = og.Config(str(args.model))
    decoder_overlay = {}
    if args.cuda_graph:
        decoder_overlay["session_options"] = {
            "provider_options": [{"cuda": {"enable_cuda_graph": "1"}}]
        }
    if args.decoder_file is not None:
        decoder_overlay["filename"] = str(args.decoder_file)
    config_overlay = {"model": {"decoder": decoder_overlay}} if decoder_overlay else {}
    if args.draft_tokens is not None:
        config_overlay["speculative"] = {"max_draft_tokens": args.draft_tokens}
    if config_overlay:
        config.overlay(json.dumps(config_overlay))
    decoder_path = args.model / (
        args.decoder_file or disk_config["model"]["decoder"]["filename"]
    )
    model = og.Model(config)
    engine = og.Engine(model)
    tokenizer = og.Tokenizer(model)
    unit = tokenizer.encode(
        "A model processes a long document by keeping track of the preceding information. "
    )
    report = {
        "backend": "GenAI Engine",
        "genai_version": og.__version__,
        "genai_commit": og.__commit__,
        "model": str(args.model.resolve()),
        "config_sha256": hashlib.sha256(
            (args.model / "genai_config.json").read_bytes()
        ).hexdigest(),
        "config_overlay": config_overlay,
        "decoder_sha256": hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_runtime": torch.version.cuda,
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "ORT_DISABLE_MOE_GEMV",
                "ORT_MOE_GEMV_FP32_ACCUM",
                "ORT_ENABLE_XQA",
                "ORT_PAGED_ATTENTION_DECODE_NUM_SPLITS",
            )
        },
        "batch_size": 1,
        "greedy": True,
        "mtp_enabled": True,
        "warmup_runs_per_context": args.warmup,
        "measured_runs_per_context": args.runs,
        "accuracy_test": False,
        "vision_test": False,
        "prompt_kind": "Raw completion of repeated text tokens, truncated to the exact context length",
        "prompt_unit_token_ids": [int(token) for token in unit],
        "timing_note": "TTFT includes Engine admission/prefill/sampling; it is not isolated prefill latency.",
        "contexts": [],
        "requested_contexts": args.contexts,
        "status": "in_progress",
    }
    save_report(args.model, args.report, report, update_manifest=not args.report_only)
    reference_config = og.Config(str(args.model))
    if config_overlay:
        reference_config.overlay(json.dumps(config_overlay))
    reference_config.overlay('{"model":{"mtp":{"enabled":false}}}')
    reference_model = og.Model(reference_config)
    reference_engine = og.Engine(reference_model)
    print("Checking real-weight MTP versus target-only greedy parity", flush=True)
    report["parity"] = {"status": "in_progress", "cases": []}
    parity_cases = [
        (
            make_prompt(
                tokenizer.encode("Explain the difference between light and sound. "), 128
            ),
            64,
        )
    ] + [(make_prompt(unit, context), args.output_tokens) for context in args.contexts]
    for prompt, output_tokens in parity_cases:
        result = greedy_parity(engine, reference_engine, prompt, output_tokens)
        result["context_tokens"] = len(prompt)
        report["parity"]["cases"].append(result)
        if result["status"] != "passed":
            report["status"] = "failed_parity"
            report["parity"].update(
                status="failed",
                first_mismatch_index=result["first_mismatch_index"],
                context_tokens=len(prompt),
            )
            save_report(args.model, args.report, report, update_manifest=not args.report_only)
            raise RuntimeError(
                "MTP and target-only greedy decoding differ at output index "
                f"{result['first_mismatch_index']} with {len(prompt)} input tokens; "
                f"see {args.report}. Benchmarking is blocked until correctness is established."
            )
        save_report(args.model, args.report, report, update_manifest=not args.report_only)
        print(
            f"Greedy parity passed: {len(prompt)} input / {output_tokens} output", flush=True
        )
    report["parity"]["status"] = "passed"
    del reference_engine, reference_model, reference_config
    save_report(args.model, args.report, report, update_manifest=not args.report_only)
    expected_outputs = {
        case["context_tokens"]: case["target_only"]["output_ids"]
        for case in report["parity"]["cases"][1:]
    }
    for context in args.contexts:
        prompt = make_prompt(unit, context)
        if len(prompt) != context:
            raise RuntimeError(
                f"Prompt length mismatch: expected {context}, got {len(prompt)}"
            )
        if context + args.output_tokens > disk_config["model"]["context_length"]:
            raise ValueError(f"Context {context} plus output exceeds the model context limit")
        for _ in range(args.warmup):
            run_turn(engine, prompt, args.output_tokens)
        runs = []
        for index in range(args.runs):
            result = run_turn(engine, prompt, args.output_tokens)
            if result["output_ids"] != expected_outputs[context]:
                report["status"] = "failed_repeatability"
                report["failure"] = {"context_tokens": context, "run": index, **result}
                save_report(
                    args.model, args.report, report, update_manifest=not args.report_only
                )
                raise RuntimeError(
                    f"Timed output differs from target-only parity at context {context}, run {index}"
                )
            runs.append(result)
            print(json.dumps({"context": context, "run": index, **result}), flush=True)
        report["contexts"].append(
            {
                "context_tokens": context,
                "runs": runs,
                "mean_ttft_seconds": statistics.mean(run["ttft_seconds"] for run in runs),
                "mean_decode_tokens_per_second": statistics.mean(
                    run["decode_tokens_per_second"] for run in runs
                ),
                "mean_total_seconds": statistics.mean(run["total_seconds"] for run in runs),
            }
        )
        report["loaded_runtime_libraries"] = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libonnxruntime" in line
            }
        )
        if "runtime_library_sha256" not in report:
            report["runtime_library_sha256"] = {}
            for library in report["loaded_runtime_libraries"]:
                with Path(library).open("rb") as stream:
                    report["runtime_library_sha256"][library] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        if len(report["contexts"]) == len(args.contexts):
            report["status"] = "passed"
            report["timed_outputs_match_target_only"] = True
        save_report(args.model, args.report, report, update_manifest=not args.report_only)
    print(f"Saved {args.report}", flush=True)


if __name__ == "__main__":
    main()
