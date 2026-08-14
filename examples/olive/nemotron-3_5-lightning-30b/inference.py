#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Direct ONNX Runtime generation for NemotronH hybrid-cache packages."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
REVISION = "d468880b6ad3c6e0d21377ce7242adaea4cc884d"
_BFLOAT16_ONNX_TYPE = 16


def _numpy_dtype(ort_type: str):
    if ort_type == "tensor(float)":
        return np.float32
    if ort_type == "tensor(float16)":
        return np.float16
    if ort_type == "tensor(bfloat16)":
        import ml_dtypes

        return ml_dtypes.bfloat16
    raise TypeError(f"Unsupported state input type: {ort_type}")


def _concrete_state_shape(shape: list[Any]) -> tuple[int, ...]:
    concrete: list[int] = []
    for dim in shape:
        if isinstance(dim, int):
            concrete.append(dim)
        elif "batch" in str(dim):
            concrete.append(1)
        elif "past" in str(dim):
            concrete.append(0)
        else:
            raise ValueError(f"Cannot resolve hybrid-cache dimension {dim!r}")
    return tuple(concrete)


def _initial_states(session) -> dict[str, Any]:
    import onnxruntime as ort

    states: dict[str, Any] = {}
    for model_input in session.get_inputs():
        if not model_input.name.startswith("past_key_values."):
            continue
        shape = _concrete_state_shape(model_input.shape)
        if model_input.type == "tensor(bfloat16)":
            states[model_input.name] = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                np.zeros(shape, dtype=np.uint16),
                _BFLOAT16_ONNX_TYPE,
            )
        else:
            states[model_input.name] = np.zeros(
                shape,
                dtype=_numpy_dtype(model_input.type),
            )
    return states


def _update_states(
    states: dict[str, Any],
    output_names: list[str],
    output_values: list[Any],
) -> None:
    for name, value in zip(output_names, output_values):
        if not name.startswith("present."):
            continue
        input_name = name.replace("present.", "past_key_values.", 1)
        if input_name in states:
            states[input_name] = value


def _token_feeds(
    session,
    token_ids: np.ndarray,
    *,
    total_length: int,
    position_ids: np.ndarray,
    states: dict[str, Any],
) -> dict[str, Any]:
    available = {model_input.name for model_input in session.get_inputs()}
    candidates = {
        "input_ids": token_ids,
        "attention_mask": np.ones((1, total_length), dtype=np.int64),
        "position_ids": position_ids,
        **states,
    }
    return {name: value for name, value in candidates.items() if name in available}


def _run_session(session, output_names: list[str], feeds: dict[str, Any]) -> list[Any]:
    import onnxruntime as ort

    if not any(isinstance(value, ort.OrtValue) for value in feeds.values()):
        return session.run(output_names, feeds)
    ort_feeds = {
        name: (
            value
            if isinstance(value, ort.OrtValue)
            else ort.OrtValue.ortvalue_from_numpy(value)
        )
        for name, value in feeds.items()
    }
    return list(session.run_with_ort_values(output_names, ort_feeds))


def _as_numpy(value: Any) -> np.ndarray:
    import onnxruntime as ort

    if not isinstance(value, ort.OrtValue):
        return value
    if value.data_type() == "tensor(bfloat16)":
        import torch

        return torch.from_dlpack(value).float().cpu().numpy()
    return value.numpy()


def _create_session(model_path: Path, device: str, profile: bool):
    if device == "cuda":
        # Importing PyTorch first preloads its matching CUDA/cuDNN DLLs on Windows.
        import torch  # noqa: F401

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
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
    )
    if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"CUDAExecutionProvider was requested but providers are {session.get_providers()}"
        )
    return session


def load_eos_token_ids(model_dir: str | Path) -> set[int]:
    """Load scalar or list EOS IDs from the assembled package metadata."""
    model_dir = Path(model_dir)
    eos_ids: set[int] = set()
    for filename in ("generation_config.json", "config.json"):
        path = model_dir / filename
        if not path.is_file():
            continue
        raw_eos = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
        if isinstance(raw_eos, int):
            eos_ids.add(raw_eos)
        elif isinstance(raw_eos, list):
            eos_ids.update(value for value in raw_eos if isinstance(value, int))
    return eos_ids


def run_token_ids(
    model_dir: str | Path,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    device: str,
    profile: bool = False,
    eos_token_ids: Collection[int] | None = None,
) -> tuple[list[int], list[np.ndarray], str | None]:
    """Run token-by-token hybrid-cache generation and return IDs plus logits."""
    model_path = Path(model_dir) / "model.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing ONNX model: {model_path}")
    if not input_ids:
        raise ValueError("input_ids must not be empty")

    session = _create_session(model_path, device, profile)
    states = _initial_states(session)
    output_names = [output.name for output in session.get_outputs()]
    generated: list[int] = []
    logits_by_step: list[np.ndarray] = []
    past_length = 0
    outputs: list[Any] | None = None

    for token_id in input_ids:
        feeds = _token_feeds(
            session,
            np.array([[token_id]], dtype=np.int64),
            total_length=past_length + 1,
            position_ids=np.array([[past_length]], dtype=np.int64),
            states=states,
        )
        outputs = _run_session(session, output_names, feeds)
        _update_states(states, output_names, outputs)
        past_length += 1

    assert outputs is not None
    logits = _as_numpy(outputs[output_names.index("logits")])[0, -1].astype(np.float32)
    eos_ids = set(eos_token_ids or ())
    for token_index in range(max_new_tokens):
        logits_by_step.append(logits.copy())
        token_id = int(np.argmax(logits))
        generated.append(token_id)
        if token_id in eos_ids or token_index + 1 == max_new_tokens:
            break
        feeds = _token_feeds(
            session,
            np.array([[token_id]], dtype=np.int64),
            total_length=past_length + 1,
            position_ids=np.array([[past_length]], dtype=np.int64),
            states=states,
        )
        outputs = _run_session(session, output_names, feeds)
        _update_states(states, output_names, outputs)
        past_length += 1
        logits = _as_numpy(outputs[output_names.index("logits")])[0, -1].astype(np.float32)

    profile_path = session.end_profiling() if profile else None
    return generated, logits_by_step, profile_path


def summarize_profile(profile_path: str) -> dict[str, int]:
    """Summarize actual node placement from an ORT profiling JSON file."""
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    providers: Counter[str] = Counter()
    for event in events:
        args = event.get("args", {})
        provider = args.get("provider")
        if event.get("cat") == "Node" and provider:
            providers[str(provider)] += 1
    return dict(sorted(providers.items()))


def _tokenize_prompt(model_dir: Path, prompt: str, use_chat: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        revision=None,
        local_files_only=True,
    )
    if use_chat:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        ids = tokenizer.encode(prompt)
    return tokenizer, list(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="What is 84 * 3 / 2?")
    parser.add_argument("--token-ids", nargs="+", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-chat", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = None
    if args.token_ids:
        input_ids = args.token_ids
    else:
        tokenizer, input_ids = _tokenize_prompt(model_dir, args.prompt, not args.no_chat)

    generated, _logits, profile_path = run_token_ids(
        model_dir,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        profile=args.profile,
        eos_token_ids=load_eos_token_ids(model_dir),
    )
    if not generated:
        raise RuntimeError("Generation produced no tokens")

    if tokenizer is None:
        print("Generated token IDs:", generated)
    else:
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if not text.strip():
            raise RuntimeError("Generation produced only empty/special-token text")
        print(text)

    if profile_path is not None:
        print(f"ORT profile: {profile_path}")
        print("Node placement:", summarize_profile(profile_path))


if __name__ == "__main__":
    main()
