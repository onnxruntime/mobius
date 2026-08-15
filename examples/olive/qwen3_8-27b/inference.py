#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Direct ONNX Runtime generation for reduced Qwen3.8 hybrid-VL packages."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
_BFLOAT16_ONNX_TYPE = 16


def _dtype(ort_type: str):
    if ort_type == "tensor(float)":
        return np.float32
    if ort_type == "tensor(float16)":
        return np.float16
    if ort_type == "tensor(bfloat16)":
        import ml_dtypes

        return ml_dtypes.bfloat16
    raise TypeError(f"Unsupported ONNX input type: {ort_type}")


def _shape(shape: list[Any]) -> tuple[int, ...]:
    """Concretize dynamic hybrid-cache shapes for a one-item decode."""
    result = []
    for value in shape:
        if isinstance(value, int):
            result.append(value)
        elif "batch" in str(value):
            result.append(1)
        elif "past" in str(value) or "sequence" in str(value):
            result.append(0)
        else:
            raise ValueError(f"Cannot resolve state dimension {value!r}")
    return tuple(result)


def _create_session(model_path: Path, device: str, profile: bool = False):
    if device == "cuda":
        import torch  # noqa: F401  # Preloads matching CUDA DLLs on Windows.

    import onnxruntime as ort

    if device == "cuda" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    options = ort.SessionOptions()
    options.enable_profiling = profile
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(model_path), options, providers=providers)
    if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDAExecutionProvider was requested: {session.get_providers()}")
    return session


def _initial_states(session) -> dict[str, Any]:
    import onnxruntime as ort

    result: dict[str, Any] = {}
    for model_input in session.get_inputs():
        if not model_input.name.startswith("past_key_values."):
            continue
        zeros = np.zeros(_shape(model_input.shape), dtype=_dtype(model_input.type))
        if model_input.type == "tensor(bfloat16)":
            zeros = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                np.zeros(zeros.shape, dtype=np.uint16), _BFLOAT16_ONNX_TYPE
            )
        result[model_input.name] = zeros
    return result


def _run(session, output_names: list[str], feeds: dict[str, Any]) -> list[Any]:
    import onnxruntime as ort

    if any(isinstance(value, ort.OrtValue) for value in feeds.values()):
        values = {
            name: value
            if isinstance(value, ort.OrtValue)
            else ort.OrtValue.ortvalue_from_numpy(value)
            for name, value in feeds.items()
        }
        return list(session.run_with_ort_values(output_names, values))
    return session.run(output_names, feeds)


def _numpy(value: Any) -> np.ndarray:
    import onnxruntime as ort

    if not isinstance(value, ort.OrtValue):
        return value
    if value.data_type() == "tensor(bfloat16)":
        import torch

        return torch.from_dlpack(value).float().cpu().numpy()
    return value.numpy()


def _update_states(states: dict[str, Any], names: list[str], values: list[Any]) -> None:
    for name, value in zip(names, values):
        if name.startswith("present."):
            states[name.replace("present.", "past_key_values.", 1)] = value


def _embedding(session, token_ids: np.ndarray, hidden_size: int) -> np.ndarray:
    inputs = {item.name: item for item in session.get_inputs()}
    media_dtype = _dtype(inputs["image_features"].type)
    feeds = {
        "input_ids": token_ids,
        # Empty media is intentional for text generation; image/video paths
        # run this same model with actual vision features in the validator.
        "image_features": np.zeros((0, hidden_size), dtype=media_dtype),
    }
    outputs = _run(
        session,
        [item.name for item in session.get_outputs()],
        {name: value for name, value in feeds.items() if name in inputs},
    )
    return _numpy(outputs[0])


def run_token_ids(
    model_dir: str | Path,
    input_ids: list[int],
    *,
    hidden_size: int,
    max_new_tokens: int,
    device: str,
    profile: bool = False,
) -> tuple[list[int], list[np.ndarray], str | None]:
    """Generate exact-length greedy tokens through embedding and hybrid decoder."""
    root = Path(model_dir)
    decoder = _create_session(root / "decoder" / "model.onnx", device, profile)
    embedding = _create_session(root / "embedding" / "model.onnx", device)
    states = _initial_states(decoder)
    names = [item.name for item in decoder.get_outputs()]
    generated: list[int] = []
    logits_by_step: list[np.ndarray] = []
    past = 0
    logits: np.ndarray | None = None

    for token_id in input_ids:
        ids = np.array([[token_id]], dtype=np.int64)
        embeds = _embedding(embedding, ids, hidden_size)
        feeds: dict[str, Any] = {
            "inputs_embeds": embeds,
            "attention_mask": np.ones((1, past + 1), dtype=np.int64),
            # Qwen MRoPE uses three equal text positions.
            "position_ids": np.full((3, 1, 1), past, dtype=np.int64),
            **states,
        }
        outputs = _run(decoder, names, feeds)
        _update_states(states, names, outputs)
        logits = _numpy(outputs[names.index("logits")])[0, -1].astype(np.float32)
        past += 1
    if logits is None:
        raise ValueError("input_ids must not be empty")

    for _ in range(max_new_tokens):
        logits_by_step.append(logits.copy())
        token_id = int(np.argmax(logits))
        generated.append(token_id)
        ids = np.array([[token_id]], dtype=np.int64)
        embeds = _embedding(embedding, ids, hidden_size)
        outputs = _run(
            decoder,
            names,
            {
                "inputs_embeds": embeds,
                "attention_mask": np.ones((1, past + 1), dtype=np.int64),
                "position_ids": np.full((3, 1, 1), past, dtype=np.int64),
                **states,
            },
        )
        _update_states(states, names, outputs)
        logits = _numpy(outputs[names.index("logits")])[0, -1].astype(np.float32)
        past += 1
    return generated, logits_by_step, decoder.end_profiling() if profile else None


def summarize_profile(profile_path: str) -> dict[str, int]:
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    providers: Counter[str] = Counter()
    for event in events:
        provider = event.get("args", {}).get("provider")
        if event.get("cat") == "Node" and provider:
            providers[str(provider)] += 1
    return dict(sorted(providers.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--token-ids", nargs="+", type=int, default=[1, 42, 17])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    ids, _logits, profile = run_token_ids(
        args.model_dir,
        args.token_ids,
        hidden_size=args.hidden_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        profile=args.profile,
    )
    print(f"Generated token IDs: {ids}")
    if profile:
        assert profile is not None
        print(f"Provider placement: {summarize_profile(profile)}")


if __name__ == "__main__":
    main()
