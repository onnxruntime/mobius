# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compare tensor-cache and experimental kv_cache model execution on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx

os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"

import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

_MODEL_REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "google/gemma-3-270m": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
}
_CACHE_INPUT = "past_key_values.{layer}.{kind}"
_CACHE_OUTPUT = "present.{layer}.{kind}"
_KV_CACHE_INPUT = "past_cache_{layer}"
_KV_CACHE_OUTPUT = "present_cache_{layer}"
_PAGE_SIZE = 16
_ONNX_REVISION = "4443045c53d13a8d3157fcbd2cd5e52450a0784b"
_ONNX_WHEEL_SHA256 = "03029ba7a488ee71bc1ad2f4647a695097e1eca95fb56000e52c89218482f744"
_ORT_REVISION = "11a522b835ee41b53c6d293a35c7c7135e2dfd5d"
_ORT_WHEEL_SHA256 = "ab1fc5ef214e69ce17ee566fec58ffe1a4a8bf4d0965cc97513cc1c9cd0205d4"


def _decoder_config(config: dict) -> dict:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _model_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while data := model_file.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def _input_tokens(tokenizer, prompt: str, count: int) -> np.ndarray:
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    repeated = tokenizer.encode(prompt, add_special_tokens=False)
    if not tokens or not repeated:
        raise ValueError("The prompt must produce at least one tokenizer token.")
    while len(tokens) < count:
        tokens.extend(repeated[: count - len(tokens)])
    return np.asarray(tokens[:count], dtype=np.int64)[None, :]


def _feeds(input_ids: np.ndarray, start: int) -> dict[str, np.ndarray]:
    end = start + input_ids.shape[1]
    return {
        "input_ids": input_ids,
        "attention_mask": np.ones((1, end), dtype=np.int64),
        "position_ids": np.arange(start, end, dtype=np.int64)[None, :],
    }


def _compare_logits(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> float:
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)
    max_abs = float(np.max(np.abs(actual - expected)))
    return max_abs


def _cache_backend(backend: str, window: int) -> str:
    if backend == "mixed":
        return "ring" if window else "paged"
    if backend == "ring" and not window:
        raise ValueError("The ring backend is only supported for sliding-window layers.")
    return backend


def _new_cache_sets(
    backends: list[str],
    layer_windows: list[int],
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, list[ort.OrtValue]]:
    caches: dict[str, list[ort.OrtValue]] = {}
    for backend in backends:
        caches[backend] = [
            ort.OrtValue.kv_cache(
                1,
                num_kv_heads,
                head_dim,
                head_dim,
                backend=_cache_backend(backend, window),
                page_size=_PAGE_SIZE,
                sliding_window=window,
            )
            for window in layer_windows
        ]
    return caches


def _run_tensor_cache(
    session: ort.InferenceSession,
    input_ids: np.ndarray,
    start: int,
    caches: list[np.ndarray],
    num_layers: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    inputs = _feeds(input_ids, start)
    for layer in range(num_layers):
        for kind_index, kind in enumerate(("key", "value")):
            inputs[_CACHE_INPUT.format(layer=layer, kind=kind)] = caches[
                layer * 2 + kind_index
            ]
    output_names = ["logits"] + [
        _CACHE_OUTPUT.format(layer=layer, kind=kind)
        for layer in range(num_layers)
        for kind in ("key", "value")
    ]
    outputs = session.run(output_names, inputs)
    return outputs[0], outputs[1:]


def _run_kv_cache(
    session: ort.InferenceSession,
    input_ids: np.ndarray,
    start: int,
    caches: list[ort.OrtValue],
) -> tuple[np.ndarray, list[ort.OrtValue]]:
    feeds = {
        name: ort.OrtValue.ortvalue_from_numpy(value)
        for name, value in _feeds(input_ids, start).items()
    }
    feeds.update({_KV_CACHE_INPUT.format(layer=i): cache for i, cache in enumerate(caches)})
    output_names = ["logits"] + [_KV_CACHE_OUTPUT.format(layer=i) for i in range(len(caches))]
    outputs = session.run_with_ort_values(output_names, feeds)
    return outputs[0].numpy(), outputs[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(_MODEL_REVISIONS), required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--prefill-chunks", help="Comma-separated token chunk sizes.")
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--backends", nargs="+", choices=("contiguous", "paged", "mixed"))
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--report",
        type=Path,
        help="Write machine-readable run evidence (default: <model-dir>/run_results.json).",
    )
    args = parser.parse_args()
    if args.decode_steps < 1:
        parser.error("--decode-steps must be positive.")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be nonnegative.")

    model_dir = args.model_dir.resolve()
    tensor_model_path = model_dir / "model.onnx"
    kv_model_path = model_dir / "model.kv_cache.onnx"
    config_path = Path(
        hf_hub_download(
            repo_id=args.model,
            filename="config.json",
            revision=_MODEL_REVISIONS[args.model],
        )
    )
    with config_path.open(encoding="utf-8") as config_file:
        config = _decoder_config(json.load(config_file))

    num_layers = int(config["num_hidden_layers"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    sliding_window = int(config.get("sliding_window") or 0)
    layer_types = config.get("layer_types")
    if sliding_window and (
        not isinstance(layer_types, list) or len(layer_types) != num_layers
    ):
        raise ValueError("A sliding-window model must expose one layer type per layer.")
    layer_windows = [
        sliding_window if layer_types and layer_types[i] == "sliding_attention" else 0
        for i in range(num_layers)
    ]
    if any(
        layer_type not in {"full_attention", "sliding_attention"}
        for layer_type in (layer_types or [])
    ):
        raise ValueError(f"Unsupported layer types in {args.model} config.")

    if args.prefill_chunks:
        chunks = [int(chunk) for chunk in args.prefill_chunks.split(",")]
        if any(chunk < 1 for chunk in chunks):
            parser.error("Every prefill chunk size must be positive.")
    elif sliding_window:
        chunks = [500, 24, 3]
    else:
        chunks = [8]
    backends = args.backends or (
        ["contiguous", "paged", "mixed"] if sliding_window else ["contiguous", "paged"]
    )
    if "mixed" in backends and not sliding_window:
        parser.error("The mixed backend requires a model with sliding-window layers.")
    if sliding_window and sum(chunks) <= sliding_window:
        raise ValueError(
            "The sliding-window example must prefill beyond one window; "
            "use chunked input totaling more than the window."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=_MODEL_REVISIONS[args.model],
    )
    token_ids = _input_tokens(tokenizer, args.prompt, sum(chunks))
    required_files = (tensor_model_path, kv_model_path, model_dir / "model.onnx.data")
    missing_files = [path.name for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing {', '.join(missing_files)} under {model_dir}; "
            "run build_model.py and keep the external ONNX data file beside the models."
        )

    session_options = ort.SessionOptions()
    session_options.add_session_config_entry("session.allow_released_opsets_only", "0")
    session_options.intra_op_num_threads = 2
    tensor_session = ort.InferenceSession(
        str(tensor_model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    kv_session = ort.InferenceSession(
        str(kv_model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    print(f"ORT version: {ort.__version__}")
    print(f"Model: {args.model}@{_MODEL_REVISIONS[args.model]}")
    print(f"ONNX SHA256: {_model_hash(kv_model_path)}")
    print(f"Original inputs: {[item.name for item in tensor_session.get_inputs()]}")
    print(f"kv_cache inputs: {[item.name for item in kv_session.get_inputs()]}")
    print(
        f"Layers: {num_layers} ({sum(bool(w) for w in layer_windows)} sliding, "
        f"{sum(not w for w in layer_windows)} full); chunks={chunks}; "
        f"tolerance atol={args.atol:g}, rtol={args.rtol:g}"
    )

    tensor_caches = [
        np.empty((1, num_kv_heads, 0, head_dim), dtype=np.float32)
        for _ in range(num_layers * 2)
    ]
    kv_caches = _new_cache_sets(backends, layer_windows, num_kv_heads, head_dim)
    generated_ids: list[int] = []
    last_logits: np.ndarray | None = None
    backend_steps: dict[str, list[dict]] = {backend: [] for backend in backends}
    start = 0
    steps = [token_ids[:, offset : offset + size] for offset, size in _chunk_offsets(chunks)]
    steps.extend([None] * args.decode_steps)

    for step_index, step_input in enumerate(steps):
        if step_input is None:
            if last_logits is None:
                raise RuntimeError("The model did not produce prefill logits.")
            step_input = np.asarray(
                [[int(np.argmax(last_logits[0, -1]))]],
                dtype=np.int64,
            )
            generated_ids.append(int(step_input[0, 0]))

        logits, tensor_caches = _run_tensor_cache(
            tensor_session, step_input, start, tensor_caches, num_layers
        )
        expected_next_token = int(np.argmax(logits[0, -1]))
        end = start + step_input.shape[1]
        max_diffs = {}
        cache_states = {}
        for backend in backends:
            previous_caches = kv_caches[backend]
            new_logits, next_caches = _run_kv_cache(
                kv_session, step_input, start, previous_caches
            )
            max_diffs[backend] = _compare_logits(
                new_logits,
                logits,
                atol=args.atol,
                rtol=args.rtol,
                label=f"{backend} logits differ at step {step_index}, positions {start}:{end}",
            )
            actual_next_token = int(np.argmax(new_logits[0, -1]))
            if actual_next_token != expected_next_token:
                raise AssertionError(
                    f"{backend} greedy token differs at step {step_index}: "
                    f"{actual_next_token} != {expected_next_token}."
                )
            retained_by_window = {}
            for layer, (previous_cache, cache) in enumerate(
                zip(previous_caches, next_caches, strict=True)
            ):
                previous_description = previous_cache.kv_cache_describe()
                expected_previous_begin = (
                    max(0, start - layer_windows[layer]) if layer_windows[layer] else 0
                )
                if previous_description["computed_tokens"] != start:
                    raise AssertionError(
                        f"{backend} layer {layer} input cache computed "
                        f"{previous_description['computed_tokens']} tokens, expected {start}."
                    )
                if previous_description["retained_begin"] != expected_previous_begin:
                    raise AssertionError(
                        f"{backend} layer {layer} input cache retained from "
                        f"{previous_description['retained_begin']}, "
                        f"expected {expected_previous_begin}."
                    )
                description = cache.kv_cache_describe()
                expected_begin = (
                    max(0, end - layer_windows[layer]) if layer_windows[layer] else 0
                )
                if description["computed_tokens"] != end:
                    raise AssertionError(
                        f"{backend} layer {layer} computed "
                        f"{description['computed_tokens']} tokens, expected {end}."
                    )
                if description["retained_begin"] != expected_begin:
                    raise AssertionError(
                        f"{backend} layer {layer} retained from "
                        f"{description['retained_begin']}, expected {expected_begin}."
                    )
                retained_by_window[layer_windows[layer]] = description["retained_begin"]
            cache_states[backend] = {
                "computed_tokens": end,
                "retained_begin_by_window": retained_by_window,
            }
            backend_steps[backend].append(
                {
                    "positions": [start, end],
                    "max_abs_error": max_diffs[backend],
                    **cache_states[backend],
                }
            )
            kv_caches[backend] = next_caches
        print(
            f"step={step_index} positions={start}:{end} "
            f"max_abs_diff={max_diffs} cache_state={cache_states}",
            flush=True,
        )
        start = end
        last_logits = logits

    print(f"Generated token IDs: {generated_ids}")
    generated_text = tokenizer.decode(generated_ids)
    print(f"Generated text: {generated_text}")
    report_path = (args.report or model_dir / "run_results.json").resolve()
    report = {
        "model": {
            "repo": args.model,
            "revision": _MODEL_REVISIONS[args.model],
            "config_sha256": _model_hash(model_dir / "config.json"),
            "baseline_onnx_sha256": _model_hash(tensor_model_path),
            "converted_onnx_sha256": _model_hash(kv_model_path),
            "external_data_sha256": _model_hash(model_dir / "model.onnx.data"),
            "layers": {
                "sliding": sum(bool(window) for window in layer_windows),
                "full": sum(not window for window in layer_windows),
                "sliding_window": sliding_window,
            },
        },
        "runtime": {
            "name": "onnxruntime",
            "version": ort.__version__,
            "source_revision": _ORT_REVISION,
            "wheel_sha256": _ORT_WHEEL_SHA256,
            "provider": "CPUExecutionProvider",
            "source_repository": "https://github.com/microsoft/onnxruntime",
        },
        "onnx": {
            "source_revision": _ONNX_REVISION,
            "wheel_version": onnx.__version__,
            "wheel_sha256": _ONNX_WHEEL_SHA256,
            "source_repository": "https://github.com/gramalingam/onnx",
        },
        "run": {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "prompt": args.prompt,
            "tokenizer_revision": _MODEL_REVISIONS[args.model],
            "tokenization": (
                "encode(add_special_tokens=True), then repeat "
                "encode(add_special_tokens=False) until the prefill token count; "
                "deterministic, no RNG seed"
            ),
            "input_token_ids": token_ids[0].tolist(),
            "prefill_chunks": chunks,
            "decode_steps": args.decode_steps,
            "backends": {
                backend: {
                    "full_layers": ("paged" if backend == "mixed" else backend),
                    "sliding_layers": ("ring" if backend == "mixed" else backend),
                    "max_abs_logit_error": max(
                        step["max_abs_error"] for step in backend_steps[backend]
                    ),
                }
                for backend in backends
            },
            "attention_masks": (
                "INT64 logical-history mask; positions are absolute; ring storage "
                "retains bounded physical K/V; same converted model for all backends"
            ),
            "tolerance": {"atol": args.atol, "rtol": args.rtol},
            "greedy_tokens_agree": True,
            "generated_token_ids": generated_ids,
            "generated_text": generated_text,
            "steps_by_backend": backend_steps,
            "command": "python " + " ".join(sys.argv),
            "python_executable": sys.executable,
            "input_token_ids_sha256": hashlib.sha256(token_ids.tobytes()).hexdigest(),
            "seed": None,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Run report: {report_path}")


def _chunk_offsets(chunks: list[int]):
    offset = 0
    for size in chunks:
        yield offset, size
        offset += size


if __name__ == "__main__":
    main()
