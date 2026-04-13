#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluate Ministral-3-3B VLM (ONNX) vs PyTorch on AI2D.

AI2D is a multiple-choice visual QA benchmark on scientific diagrams.
Each sample has an image, a question, four answer options, and a
ground-truth answer. Accuracy is the fraction of questions answered
with the correct option letter.

Usage:
    # ONNX only (default)
    python eval.py --model_path cuda/models

    # Both ONNX and PyTorch side-by-side
    python eval.py --pytorch_model mistralai/Ministral-3-3B-Instruct-2512

    # Larger sample
    python eval.py --num_samples 200
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
import time

import onnxruntime_genai as og
from datasets import load_dataset
from PIL import Image

NUMBERS = ["1", "2", "3", "4"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise multiple-choice answering assistant. "
    "When given a question with numbered options, "
    "respond with ONLY a single digit (1, 2, 3, or 4). "
    "Do not include any explanation, reasoning, or other "
    "text -- just the digit."
)


def _build_content_text(question: str, options: list[str]) -> str:
    """Build the shared prompt content for evaluation."""
    option_text = "\n".join(f"{N}. {o}" for N, o in zip(NUMBERS, options))
    return (
        "Look at the diagram and answer the "
        "multiple-choice question.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_text}\n\n"
        "Reply with the number only (1, 2, 3, or 4)."
    )


def build_messages(
    question: str,
    options: list[str],
    system_prompt: str = "",
) -> str:
    """Return JSON-encoded chat messages for apply_chat_template."""
    content = _build_content_text(question, options)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Use [IMG] tag instead of structured content blocks —
    # Mistral's chat template uses sort(attribute='type')
    # which ORT GenAI's minja engine doesn't support
    messages.append({"role": "user", "content": "[IMG]" + content})
    return json.dumps(messages)


def parse_answer(text: str) -> str | None:
    """Extract the first 1/2/3/4 digit from a model response."""
    text = text.strip()
    m = re.search(r"\b([1-4])\b", text)
    if m:
        return m.group(1)
    for ch in text:
        if ch in NUMBERS:
            return ch
    return None


def ground_truth_number(sample: dict) -> str | None:
    """Normalise the dataset's answer field to a 1-based number string.

    AI2D stores answer as a 0-based integer index into the options
    list. We map: index 0 -> '1', 1 -> '2', 2 -> '3', 3 -> '4'.
    """
    answer = sample.get("answer", "")
    try:
        idx = int(answer)
        if 0 <= idx < 4:
            return NUMBERS[idx]
    except (ValueError, TypeError):
        pass  # answer field is not a valid integer index
    return None


def pil_from_sample(sample: dict) -> Image.Image | None:
    """Return PIL image from a dataset sample."""
    img = sample.get("image")
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    return None


def load_ai2d(num_samples: int):
    """Load a deterministic subset of AI2D test samples."""
    print(f"Loading AI2D dataset ({num_samples} samples)...")
    ds = load_dataset("lmms-lab/ai2d", split="test")
    ds = ds.select(range(min(num_samples, len(ds))))
    print(f"  Loaded {len(ds)} samples.")
    return ds


def build_onnx_runner(model_path: str):
    """Load ONNX model with ORT GenAI."""
    print(f"\nLoading ONNX model from: {model_path}")
    model = og.Model(model_path)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    print("  ONNX model loaded.")
    return model, processor, tokenizer


def run_onnx(
    model,
    processor,
    tokenizer,
    pil_image: Image.Image,
    messages_json: str,
) -> str:
    """Run a single inference with the ONNX GenAI model."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        pil_image.save(f, format="PNG")
        tmp_path = f.name

    try:
        images = og.Images.open(tmp_path)
        prompt = tokenizer.apply_chat_template(messages_json, add_generation_prompt=True)
        inputs = processor(prompt, images=images)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=2000, do_sample=False)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            tokens.append(generator.get_next_tokens()[0])
        del generator

        return tokenizer.decode(tokens)
    finally:
        os.unlink(tmp_path)


def build_pytorch_runner(model_id: str):
    """Load HuggingFace PyTorch model for comparison."""
    print(f"\nLoading PyTorch model: {model_id}")
    import torch
    from transformers import (
        AutoProcessor,
        Mistral3ForConditionalGeneration,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  Device: {device}, dtype: {dtype}")

    pt_model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True
    ).to(device)
    pt_proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("  PyTorch model loaded.")
    return pt_model, pt_proc, device


def run_pytorch(
    pt_model,
    pt_proc,
    pil_image: Image.Image,
    question: str,
    options: list[str],
    device: str,
    system_prompt: str = "",
) -> str:
    """Run a single inference with the HuggingFace model."""
    import torch

    content = _build_content_text(question, options)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": content},
            ],
        }
    )
    text = pt_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pt_proc(
        text=[text],
        images=[pil_image],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = pt_model.generate(**inputs, max_new_tokens=8, do_sample=False)

    out_ids = out[0][inputs["input_ids"].shape[-1] :]
    return pt_proc.decode(out_ids, skip_special_tokens=True)


def evaluate(dataset, runner_fn, label: str) -> dict:
    """Run evaluation on a dataset with the given runner."""
    correct = 0
    skipped = 0
    total = len(dataset)
    latencies = []

    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {label}  ({total} samples)")
    print(f"{'=' * 60}")

    for i, sample in enumerate(dataset):
        gt = ground_truth_number(sample)
        if gt is None:
            skipped += 1
            continue

        pil_image = pil_from_sample(sample)
        if pil_image is None:
            skipped += 1
            continue

        question = sample.get("question", "")
        options = sample.get("options", [])
        if len(options) < 2:
            skipped += 1
            continue

        try:
            t0 = time.perf_counter()
            raw = runner_fn(pil_image, question, options)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
        except Exception as e:
            print(f"  [WARN] sample {i}: {e}")
            skipped += 1
            continue

        pred = parse_answer(raw)
        hit = pred == gt
        if hit:
            correct += 1

        if (i + 1) % 10 == 0 or i == 0:
            denominator = i + 1 - skipped + 1e-9
            acc = correct / denominator
            print(
                f"  [{i + 1:4d}/{total}] "
                f"gt={gt} pred={pred} "
                f"raw={raw.strip()!r:20}  "
                f"{'OK' if hit else 'WRONG'}  "
                f"running_acc={acc:.3f}"
            )

    evaluated = total - skipped
    accuracy = correct / evaluated if evaluated > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print(
        f"\n  {label}: {correct}/{evaluated} correct  |  "
        f"accuracy = {accuracy:.4f} ({accuracy * 100:.2f}%)"
    )
    print(f"  avg latency per sample: {avg_lat:.2f}s  |  skipped: {skipped}")
    return {
        "label": label,
        "accuracy": accuracy,
        "correct": correct,
        "evaluated": evaluated,
        "avg_latency_s": avg_lat,
        "skipped": skipped,
    }


def main():
    """Entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description=("Eval ONNX vs PyTorch Ministral-3-3B VLM on AI2D")
    )
    parser.add_argument(
        "--model_path",
        default="models",
        help="Path to ONNX model dir",
    )
    parser.add_argument(
        "--pytorch_model",
        default=None,
        help="HuggingFace model ID for PyTorch comparison",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of AI2D test samples (default: 100)",
    )
    parser.add_argument(
        "--skip_onnx",
        action="store_true",
        help="Skip ONNX evaluation",
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help=("System prompt to suppress chain-of-thought. Pass empty string to disable."),
    )
    args = parser.parse_args()

    ds = load_ai2d(args.num_samples)
    results = []

    sys_prompt = args.system_prompt
    if sys_prompt:
        print(f"\nSystem prompt: {sys_prompt!r}")
    else:
        print("\nSystem prompt: (none)")

    if not args.skip_onnx:
        onnx_model, onnx_proc, onnx_tok = build_onnx_runner(args.model_path)

        def onnx_runner(pil_image, question, options):
            msgs = build_messages(question, options, sys_prompt)
            return run_onnx(
                onnx_model,
                onnx_proc,
                onnx_tok,
                pil_image,
                msgs,
            )

        results.append(
            evaluate(
                ds,
                onnx_runner,
                f"ONNX @ {args.model_path}",
            )
        )

    if args.pytorch_model:
        pt_model, pt_proc, device = build_pytorch_runner(args.pytorch_model)

        def pt_runner(pil_image, question, options):
            return run_pytorch(
                pt_model,
                pt_proc,
                pil_image,
                question,
                options,
                device,
                sys_prompt,
            )

        results.append(
            evaluate(
                ds,
                pt_runner,
                f"PyTorch @ {args.pytorch_model}",
            )
        )

    print(f"\n{'=' * 60}")
    print("  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print("  Model       : Ministral-3-3B-Instruct-2512 (VLM)")
    print("  Dataset     : AI2D (science diagram QA, multiple choice)")
    print(f"  Samples     : {args.num_samples}")
    if not sys_prompt:
        print("  System prompt: (none)")
    else:
        truncated = sys_prompt[:80]
        ellipsis = "..." if len(sys_prompt) > 80 else ""
        print(f"  System prompt: {truncated}{ellipsis}")
    print()
    for r in results:
        print(f"  {r['label']}")
        print(f"    Accuracy : {r['accuracy'] * 100:.2f}%  ({r['correct']}/{r['evaluated']})")
        print(f"    Avg lat  : {r['avg_latency_s']:.2f}s/sample")
        print()

    if len(results) == 2:
        delta = results[1]["accuracy"] - results[0]["accuracy"]
        onnx_lat = max(results[0]["avg_latency_s"], 1e-9)
        speedup = results[1]["avg_latency_s"] / onnx_lat
        print(f"  Accuracy delta (PyTorch - ONNX): {delta * 100:+.2f} pp")
        print(f"  Speedup (PyTorch lat / ONNX lat): {speedup:.2f}x")


if __name__ == "__main__":
    main()
