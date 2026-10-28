# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark Drama 3 requests with GPU-resident recurrent/KV state.

Supports the teacher-forced baseline and single-pass generation task. Report
initialization, prefill, first frame, steady frame latency and optional waveform
RTF separately. A contended or short run is diagnostic, never performance sign-off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from audit_drama3 import request as execute_greedy_request
from drama3_runtime import (
    BoundSession,
    codebook_view,
    gpu_snapshot,
    next_cache,
    require_idle,
    sequence_view,
)
from drama3_runtime import ort_to_numpy as _ort_value_to_numpy
from onnxscript import GraphBuilder


def greedy_model(config, dtype):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    logits = builder.input("logits", dtype, [1, config["text_config"]["vocab_size"]])
    candidates = builder.op.Constant(
        value_ints=sorted(
            [
                config["eos_token_id"],
                *range(config["semantic_start_token_id"], config["semantic_end_token_id"] + 1),
            ]
        )
    )
    selected = builder.op.Gather(logits, candidates, axis=-1)
    token = builder.op.Gather(
        candidates,
        builder.op.ArgMax(
            builder.op.Cast(selected, to=ir.DataType.FLOAT), axis=-1, keepdims=0
        ),
    )
    token.type, token.shape = ir.TensorType(ir.DataType.INT64), ir.Shape([1])
    builder.add_output(token, "token")
    return ir.Model(graph, ir_version=12)


def benchmark(args):
    conv_algo_search = getattr(args, "cudnn_conv_algo_search", "EXHAUSTIVE")
    if args.frames < 2 or args.warmup < 1:
        raise ValueError("Use at least two measured frames and one warmup request")
    if args.device == "cpu" and "cuda" in (args.embedding_device, args.codec_device):
        raise ValueError("CUDA component placement requires --device cuda")
    if args.codec_conv1d_pad_to_nc1d and (
        args.codec is None or (args.codec_device or args.device) != "cuda"
    ):
        raise ValueError("The cuDNN codec layout option requires a CUDA codec")
    if not np.isfinite(args.max_rtf) or args.max_rtf <= 0 or args.device_id < 0:
        raise ValueError("Use a finite positive RTF target and nonnegative device ID")
    idle = True
    before = None
    if args.device == "cuda":
        before = gpu_snapshot(args.device_id)
        idle = require_idle(before, args.allow_contended)
        ort.preload_dlls()
    config = json.loads(args.config.read_text())
    with np.load(args.prompt) as saved:
        prompt = saved["input_frames"]
    count = config["audio_decoder_config"]["num_codebooks"]
    if prompt.ndim != 2 or prompt.shape[1] != count + 1 or len(prompt) < 2:
        raise ValueError("Prompt must contain at least two [main,code0,...] frame rows")
    if len(prompt) + args.frames > config["text_config"]["max_position_embeddings"]:
        raise ValueError("Benchmark exceeds the backbone position capacity")
    fast = (args.package / "audio_generator" / "model.onnx").is_file()
    audio_name = "audio_generator" if fast else "audio_decoder"
    started = time.perf_counter()
    kwargs = {
        "device": args.device,
        "device_id": args.device_id,
        "profile": args.profile,
        "profile_directory": args.output.parent / "drama3-profiles",
        "cudnn_conv_algo_search": conv_algo_search,
    }
    sessions = {
        name: BoundSession(
            args.package / name / "model.onnx",
            **{
                **kwargs,
                "device": (args.embedding_device or args.device)
                if name == "embedding"
                else args.device,
            },
        )
        for name in ("embedding", "decoder", audio_name)
    }
    policy = {
        "temperature": np.array([args.temperature], np.float32),
        "top_k": np.array([args.top_k], np.int64),
        "top_p": np.array([args.top_p], np.float32),
        "seed": np.array([args.seed], np.int64),
    }
    controls = {
        **policy,
        "counter": np.zeros(1, np.int64),
        "generate": np.array(False),
        "do_sample": np.array(args.sample),
    }
    small_outputs = {"token", "frame_codes", "valid", "semantic_code", "next_counter"}
    for name in ("main_sampler", "codebook_sampler"):
        if args.sample:
            sessions[name] = BoundSession(args.package / "policies" / f"{name}.onnx", **kwargs)
    dtype = sessions["decoder"].session.get_outputs()[0].type
    runtime_features = (
        sessions["decoder"]
        .session.get_modelmeta()
        .custom_metadata_map.get("mobius.required_runtime_features", "")
    )
    element_type = {
        "tensor(float)": ir.DataType.FLOAT,
        "tensor(float16)": ir.DataType.FLOAT16,
        "tensor(bfloat16)": ir.DataType.BFLOAT16,
    }[dtype]

    def audio(feeds):
        fetches = [
            name
            for name in sessions[audio_name].outputs
            if name != "codebook_logits" or not fast
        ]
        result = sessions[audio_name].run(feeds, fetches=fetches, cpu_outputs=small_outputs)
        if fast and not _ort_value_to_numpy(result["valid"]).all():
            raise FloatingPointError(
                "Single-pass tower rejected a frame; caches were not committed"
            )
        return result

    with tempfile.TemporaryDirectory(prefix="drama3-benchmark-") as temporary:
        greedy_path = Path(temporary) / "greedy.onnx"
        ir.save(greedy_model(config, element_type), greedy_path)
        sessions["greedy"] = BoundSession(greedy_path, **kwargs)

        def request(limit):
            if fast and not args.sample:
                frames, timing = execute_greedy_request(
                    {
                        "embedding": sessions["embedding"],
                        "decoder": sessions["decoder"],
                        "audio_generator": sessions[audio_name],
                        "main_policy": sessions["greedy"],
                    },
                    config,
                    prompt,
                    limit,
                )
                return (
                    frames,
                    timing["frame_seconds"],
                    timing["prefill_seconds"],
                    timing["first_frame_seconds"],
                    timing["token_pipeline_seconds"],
                )
            origin = time.perf_counter()
            decoder_cache = sessions["decoder"].empty_cache()
            tower_cache = sessions[audio_name].empty_cache()
            counter = np.zeros(1, np.int64)
            history = np.zeros((1, 16), np.int64)
            positions = np.arange(len(prompt), dtype=np.int64)[None]
            mask = np.ones_like(positions)
            embedded = sessions["embedding"].run({"frames": prompt[None]})
            decoded = sessions["decoder"].run(
                {
                    **decoder_cache,
                    "inputs_embeds": embedded["inputs_embeds"],
                    "position_ids": positions,
                    "attention_mask": mask,
                }
            )
            decoder_cache = next_cache(decoded, args.device)
            prompt_semantic = (prompt[None, 1:, 0] >= config["semantic_start_token_id"]) & (
                prompt[None, 1:, 0] <= config["semantic_end_token_id"]
            )
            prefilled = audio(
                {
                    **tower_cache,
                    **(controls if fast else {}),
                    "hidden_states": sequence_view(
                        decoded["hidden_states"], 0, len(prompt) - 1
                    ),
                    "shared_input": sequence_view(
                        embedded["shared_input"], 0, len(prompt) - 1
                    ),
                    "codes": prompt[None, 1:, 1:],
                    "semantic_mask": prompt_semantic,
                    "position_ids": positions[:, :-1],
                    "attention_mask": mask[:, :-1],
                }
            )
            tower_cache = next_cache(prefilled, args.device)
            prefill_seconds = time.perf_counter() - origin
            durations, frames = [], []
            first_frame_seconds = None
            frame = prompt[None, -1:]
            for step in range(limit):
                tick = time.perf_counter()
                position = len(prompt) - 1 + step
                if step:
                    embedded = sessions["embedding"].run({"frames": frame})
                    positions = np.array([[position]], np.int64)
                    mask = np.ones((1, position + 1), np.int64)
                    decoded = sessions["decoder"].run(
                        {
                            **decoder_cache,
                            "inputs_embeds": embedded["inputs_embeds"],
                            "position_ids": positions,
                            "attention_mask": mask,
                        }
                    )
                    decoder_cache = next_cache(decoded, args.device)
                logits = sequence_view(
                    decoded["logits"], decoded["logits"].shape()[1] - 1, 1, squeeze_time=True
                )
                selection = sessions["main_sampler" if args.sample else "greedy"].run(
                    {
                        "logits": logits,
                        **(
                            {**policy, "counter": counter, "history": history}
                            if args.sample
                            else {}
                        ),
                    },
                    cpu_outputs=small_outputs,
                )
                if args.sample:
                    if not _ort_value_to_numpy(selection["valid"]).all():
                        raise FloatingPointError("Main sampling rejected invalid logits")
                    counter = _ort_value_to_numpy(selection["next_counter"])
                main = int(_ort_value_to_numpy(selection["token"])[0])
                if main == config["eos_token_id"]:
                    break
                history = np.concatenate(
                    (history[:, 1:], np.array([[main]], np.int64)), axis=1
                )
                codes = np.zeros((1, 1, count), np.int64)
                codes[0, 0, 0] = main - config["semantic_start_token_id"]
                tower_feeds = {
                    **tower_cache,
                    "hidden_states": sequence_view(
                        decoded["hidden_states"], decoded["hidden_states"].shape()[1] - 1, 1
                    ),
                    "shared_input": sequence_view(
                        embedded["shared_input"], embedded["shared_input"].shape()[1] - 1, 1
                    ),
                    "semantic_mask": np.ones((1, 1), bool),
                    "position_ids": np.array([[position]], np.int64),
                    "attention_mask": np.ones((1, position + 1), np.int64),
                }
                for index in range(1 if fast else count - 1):
                    output = audio(
                        {
                            **tower_feeds,
                            "codes": codes,
                            **(
                                {**controls, "generate": np.array(True), "counter": counter}
                                if fast
                                else {}
                            ),
                        }
                    )
                    if fast:
                        codes = _ort_value_to_numpy(output["frame_codes"])
                        counter = _ort_value_to_numpy(output["next_counter"])
                    elif args.sample:
                        sampled = sessions["codebook_sampler"].run(
                            {
                                **policy,
                                "counter": counter,
                                "logits": codebook_view(output["codebook_logits"], index),
                            },
                            cpu_outputs=small_outputs,
                        )
                        if not _ort_value_to_numpy(sampled["valid"]).all():
                            raise FloatingPointError(
                                "Codebook sampling rejected invalid logits"
                            )
                        codes[0, 0, index + 1] = _ort_value_to_numpy(sampled["token"])[0]
                        counter = _ort_value_to_numpy(sampled["next_counter"])
                    else:
                        picked = sessions["greedy_codebook"].run(
                            {"logits": codebook_view(output["codebook_logits"], index)},
                            cpu_outputs=small_outputs,
                        )
                        codes[0, 0, index + 1] = _ort_value_to_numpy(picked["token"])[0]
                tower_cache = next_cache(output, args.device)
                frame = np.concatenate((np.array([[[main]]], np.int64), codes), axis=-1)
                frames.append(frame[0, 0])
                durations.append(time.perf_counter() - tick)
                if first_frame_seconds is None:
                    first_frame_seconds = time.perf_counter() - origin
            elapsed = time.perf_counter() - origin
            if not frames:
                raise RuntimeError(
                    "No audio frames generated; no valid speech performance measurement"
                )
            return np.stack(frames), durations, prefill_seconds, first_frame_seconds, elapsed

        if not fast and not args.sample:
            vocabulary = config["audio_decoder_config"]["vocab_size"]
            graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
            builder = GraphBuilder(graph)
            logits = builder.input("logits", element_type, [1, vocabulary])
            token = builder.op.ArgMax(
                builder.op.Cast(logits, to=ir.DataType.FLOAT), axis=-1, keepdims=0
            )
            token.type, token.shape = ir.TensorType(ir.DataType.INT64), ir.Shape([1])
            builder.add_output(token, "token")
            codebook_path = Path(temporary) / "greedy_codebook.onnx"
            ir.save(ir.Model(graph, ir_version=12), codebook_path)
            sessions["greedy_codebook"] = BoundSession(codebook_path, **kwargs)
        if args.codec is not None:
            path = args.codec / "waveform_decoder" / "model.onnx"
            if not path.is_file():
                path = args.codec / "model.onnx"
            sessions["waveform_decoder"] = BoundSession(
                path,
                **{
                    **kwargs,
                    "device": args.codec_device or args.device,
                    "cudnn_conv1d_pad_to_nc1d": args.codec_conv1d_pad_to_nc1d,
                },
            )
            metadata = sessions["waveform_decoder"].session.get_modelmeta().custom_metadata_map
            if (
                metadata.get("sample_rate") != "44100"
                or metadata.get("samples_per_frame") != "2048"
            ):
                raise ValueError(
                    "Waveform graph must declare the Drama 3 44100Hz/2048-sample frame contract"
                )
        initialization = time.perf_counter() - started

        def decode_waveform(frames):
            tick = time.perf_counter()
            waveform = sessions["waveform_decoder"].run(
                {"codes": frames[None, :, 1:].transpose(0, 2, 1)}
            )
            samples = _ort_value_to_numpy(waveform["waveform"])
            seconds = time.perf_counter() - tick
            if samples.shape != (1, 1, len(frames) * 2048) or not np.isfinite(samples).all():
                raise RuntimeError("Invalid waveform output; refusing a performance pass")
            return seconds

        for _ in range(args.warmup):
            warm_frames, *_ = request(args.frames)
            if args.codec is not None:
                decode_waveform(warm_frames)
        after_warmup = gpu_snapshot(args.device_id) if args.device == "cuda" else None
        frames, durations, prefill, first_frame, elapsed = request(args.frames)
        waveform_seconds = decode_waveform(frames) if args.codec is not None else None
        after_measurement = gpu_snapshot(args.device_id) if args.device == "cuda" else None
        no_competitors = all(
            not snapshot.get("foreign_pids")
            for snapshot in (before, after_warmup, after_measurement)
            if snapshot is not None
        )
        steady = durations[1:]
        audio_seconds = len(frames) * 2048 / 44100
        result = {
            "ort_version": ort.__version__,
            "ort_build": ort.get_build_info(),
            "device": args.device,
            "embedding_device": sessions["embedding"].device,
            "codec_device": sessions["waveform_decoder"].device
            if args.codec is not None
            else None,
            "codec_provider_options": sessions[
                "waveform_decoder"
            ].session.get_provider_options()
            if args.codec is not None
            else None,
            "provider_options": (
                {
                    "device_id": args.device_id,
                    "use_tf32": "0",
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": conv_algo_search,
                }
                if args.device == "cuda"
                else {}
            ),
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "gpu_before": before,
            "gpu_after_warmup": after_warmup,
            "gpu_after_measurement": after_measurement,
            "dtype": dtype,
            "expert_execution": "dynamic_fp8"
            if "activation_quantization=dynamic_fp8" in runtime_features
            else "dense",
            "single_pass_tower": fast,
            "sampling": args.sample,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "requested_frames": args.frames,
            "warmup_requests": args.warmup,
            "prompt_rows": len(prompt),
            "prompt_sha256": hashlib.sha256(prompt.tobytes()).hexdigest(),
            "tower_calls_per_generated_frame": 1 if fast else count - 1,
            "graph_sha256": {
                name: hashlib.sha256(
                    (args.package / name / "model.onnx").read_bytes()
                ).hexdigest()
                for name in ("embedding", "decoder", audio_name)
            },
            "weights_hashed": False,
            "gpu_resident_caches": args.device == "cuda",
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "initialization_seconds": initialization,
            "prefill_seconds": prefill,
            "first_frame_seconds": first_frame,
            "generated_frames": len(frames),
            "generated_frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
            "steady_frames_per_second": len(steady) / sum(steady) if steady else None,
            "p50_frame_ms": statistics.median(steady) * 1000 if steady else None,
            "p95_frame_ms": float(np.percentile(steady, 95)) * 1000 if steady else None,
            "frame_ms": [seconds * 1000 for seconds in durations],
            "token_pipeline_rtf": elapsed / audio_seconds,
            "waveform_seconds": waveform_seconds,
            "time_to_waveform_seconds": elapsed + waveform_seconds
            if waveform_seconds is not None
            else None,
            "waveform_delivery": "Full sequence; first-frame latency is not streaming time-to-audio",
            "end_to_end_rtf": (elapsed + waveform_seconds) / audio_seconds
            if waveform_seconds is not None
            else None,
            "measurement_qualified": (
                args.device == "cuda"
                and idle
                and no_competitors
                and len(frames) >= 50
                and not args.profile
                and not args.allow_contended
            ),
            "quality_validation": "Must be established independently; this benchmark does not attest numerical parity",
        }
        if args.profile:
            result["profiles"] = {
                name: session.session.end_profiling() for name, session in sessions.items()
            }
        result["end_to_end_rtf_target"] = args.max_rtf
        result["meets_end_to_end_rtf_target"] = (
            result["end_to_end_rtf"] <= args.max_rtf
            if result["measurement_qualified"] and result["end_to_end_rtf"] is not None
            else None
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if result["meets_end_to_end_rtf_target"] is False:
            raise RuntimeError(f"End-to-end RTF exceeds the requested {args.max_rtf} target")
        return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--package", required=True, type=Path)
    result.add_argument(
        "--config", required=True, type=Path, help="Pinned Hugging Face config.json"
    )
    result.add_argument(
        "--prompt", required=True, type=Path, help="Independent state-00.npz with input_frames"
    )
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--codec", type=Path)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument(
        "--embedding-device",
        choices=("cpu", "cuda"),
        help="Override embedding placement to leave more GPU memory for the codec",
    )
    result.add_argument("--codec-device", choices=("cpu", "cuda"))
    result.add_argument(
        "--codec-conv1d-pad-to-nc1d",
        action="store_true",
        help="Use cuDNN's alternative 1-D codec layout; measure on the target GPU",
    )
    result.add_argument("--device-id", type=int, default=0)
    result.add_argument(
        "--cudnn-conv-algo-search",
        choices=("EXHAUSTIVE", "HEURISTIC", "DEFAULT"),
        default="EXHAUSTIVE",
        help="cuDNN convolution search for CUDA sessions (default: EXHAUSTIVE)",
    )
    result.add_argument("--frames", type=int, default=100)
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--sample", action="store_true")
    result.add_argument("--temperature", type=float, default=0.7)
    result.add_argument("--top-k", type=int, default=30)
    result.add_argument("--top-p", type=float, default=0.95)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--profile", action="store_true", help="Diagnostic run; timings are unqualified"
    )
    result.add_argument("--allow-contended", action="store_true", help="Diagnostic run only")
    result.add_argument("--max-rtf", type=float, default=1.0)
    return result


if __name__ == "__main__":
    benchmark(parser().parse_args())
