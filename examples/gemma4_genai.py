#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Gemma 4 text generation with ORT GenAI — build, export, and generate.

Builds an ONNX model for Gemma 4 (text-only) with mobius, saves it with
genai_config.json, and runs multi-step text generation via onnxruntime-genai.
Reports tokens/sec for benchmarking.

**Important:** Gemma 4 ``-it`` (instruction-tuned) models require chat-formatted
prompts. Raw text prompts produce degenerate repetitive output. This script
automatically wraps prompts in the Gemma chat template.

Requirements::

    pip install mobius-ai[transformers] onnxruntime-genai

Usage::

    # CPU generation (default):
    python examples/gemma4_genai.py

    # Custom prompt:
    python examples/gemma4_genai.py --prompt "Explain quantum computing"

    # Image + text (multimodal):
    python examples/gemma4_genai.py --image photo.jpg --prompt "Describe this image"

    # Audio + text (multimodal):
    python examples/gemma4_genai.py --audio speech.wav --prompt "Transcribe this audio"

    # CUDA generation:
    python examples/gemma4_genai.py --device cuda

    # Different model:
    python examples/gemma4_genai.py --model google/gemma-4-e4b-it

    # Half precision:
    python examples/gemma4_genai.py --dtype f16

    # Longer generation:
    python examples/gemma4_genai.py --max-new-tokens 200

    # Build and save only:
    python examples/gemma4_genai.py --save-to output/gemma4/

    # Use a pre-built model directory:
    python examples/gemma4_genai.py --model-dir output/gemma4/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_PROMPT = "What is the capital of France?"
MAX_NEW_TOKENS = 50


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------


def format_chat_prompt(user_message: str) -> str:
    """Wrap a user message in Gemma 4's chat template.

    Gemma 4 instruction-tuned models expect this format::

        <bos><start_of_turn>user
        {message}<end_of_turn>
        <start_of_turn>model

    The ``<bos>`` prefix is required — the Gemma4 tokenizer has
    ``add_bos_token=False``, so it must be included explicitly.
    Raw prompts without the template produce degenerate output.
    """
    return f"<bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"


# ---------------------------------------------------------------------------
# Export via mobius CLI
# ---------------------------------------------------------------------------


def build_and_export(
    model_id: str,
    output_dir: str,
    *,
    dtype: str = "f32",
) -> None:
    """Build ONNX model with mobius CLI and write ORT GenAI config."""
    cmd = [
        sys.executable,
        "-m",
        "mobius",
        "build",
        "--model",
        model_id,
        "--dtype",
        dtype,
        "--optimize",
        "--runtime",
        "ort-genai",
        output_dir,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print(f"Export complete → {output_dir}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    model_dir: str,
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = "cpu",
) -> tuple[str, float]:
    """Run text generation with ORT GenAI.  Returns (text, tokens_per_sec).

    The prompt should already be wrapped in the chat template via
    :func:`format_chat_prompt`.
    """
    import onnxruntime_genai as og

    print(f"Loading model from {model_dir!r} (device={device}) ...")
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)

    # Apply chat template so the -it model produces coherent output
    chat_prompt = format_chat_prompt(prompt)
    input_ids = tokenizer.encode(chat_prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_ids) + max_new_tokens)

    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)

    print(f"\nPrompt: {prompt}")
    print("-" * 60)

    tokenizer_stream = tokenizer.create_stream()
    generated_tokens: list[int] = []

    t_start = time.perf_counter()
    t_first_token = None

    for _ in range(max_new_tokens):
        if generator.is_done():
            break
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        generated_tokens.append(token)

        if t_first_token is None:
            t_first_token = time.perf_counter()

        print(tokenizer_stream.decode(token), end="", flush=True)

    t_end = time.perf_counter()
    print()
    print("-" * 60)

    total_time = t_end - t_start
    num_tokens = len(generated_tokens)
    tokens_per_sec = num_tokens / total_time if total_time > 0 else 0

    ttft = (t_first_token - t_start) if t_first_token else 0
    decode_time = t_end - t_first_token if t_first_token else total_time
    decode_tps = (num_tokens - 1) / decode_time if decode_time > 0 and num_tokens > 1 else 0

    print("\n📊 Performance:")
    print(f"   Tokens generated: {num_tokens}")
    print(f"   Total time:       {total_time:.2f}s")
    print(f"   Time to first:    {ttft:.3f}s")
    print(f"   Overall:          {tokens_per_sec:.1f} tok/s")
    print(f"   Decode:           {decode_tps:.1f} tok/s")

    del generator
    output = tokenizer.decode(generated_tokens)
    return output, tokens_per_sec


def generate_multimodal(
    model_dir: str,
    prompt: str,
    *,
    image_path: str | None = None,
    audio_path: str | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = "cpu",
) -> tuple[str, float]:
    """Run multimodal generation with ORT GenAI.  Returns (text, tokens_per_sec).

    Uses the GenAI MultiModalProcessor to handle image/audio inputs.
    The model directory must contain a VLM export (decoder + vision + embedding).
    """
    import onnxruntime_genai as og

    print(f"Loading VLM model from {model_dir!r} (device={device}) ...")
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)
    processor = model.create_multimodal_processor()

    # Build chat prompt with multimodal content
    content_parts = []
    if image_path:
        content_parts.append({"type": "image", "image": image_path})
    if audio_path:
        content_parts.append({"type": "audio", "audio": audio_path})
    content_parts.append({"type": "text", "text": prompt})

    # Use HF processor for chat template with multimodal placeholders
    from transformers import AutoProcessor as HFProcessor

    hf_proc = HFProcessor.from_pretrained(MODEL_ID)
    messages = [{"role": "user", "content": content_parts}]
    full_prompt = hf_proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Load media
    images = og.Images.open(image_path) if image_path else None
    audios = og.Audios.open(audio_path) if audio_path else None

    # Process inputs
    kwargs = {"images": images} if images else {}
    if audios:
        kwargs["audios"] = audios
    inputs = processor(full_prompt, **kwargs)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=8192)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    modality = []
    if image_path:
        modality.append(f"image={image_path}")
    if audio_path:
        modality.append(f"audio={audio_path}")
    print(f"\nPrompt: {prompt}")
    if modality:
        print(f"Media: {', '.join(modality)}")
    print("-" * 60)

    tokenizer_stream = tokenizer.create_stream()
    generated_tokens: list[int] = []

    t_start = time.perf_counter()
    t_first_token = None

    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        generated_tokens.append(token)

        if t_first_token is None:
            t_first_token = time.perf_counter()

        print(tokenizer_stream.decode(token), end="", flush=True)
        if len(generated_tokens) >= max_new_tokens:
            break

    t_end = time.perf_counter()
    print()
    print("-" * 60)

    total_time = t_end - t_start
    num_tokens = len(generated_tokens)
    tokens_per_sec = num_tokens / total_time if total_time > 0 else 0

    ttft = (t_first_token - t_start) if t_first_token else 0
    decode_time = t_end - t_first_token if t_first_token else total_time
    decode_tps = (num_tokens - 1) / decode_time if decode_time > 0 and num_tokens > 1 else 0

    print("\n📊 Performance:")
    print(f"   Tokens generated: {num_tokens}")
    print(f"   Total time:       {total_time:.2f}s")
    print(f"   Time to first:    {ttft:.3f}s")
    print(f"   Overall:          {tokens_per_sec:.1f} tok/s")
    print(f"   Decode:           {decode_tps:.1f} tok/s")

    del generator
    output = tokenizer.decode(generated_tokens)
    return output, tokens_per_sec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 text generation with ORT GenAI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Pre-built model directory (skips export).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Execution provider (default: cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="f32",
        choices=["f32", "f16", "bf16"],
        help="Model precision (default: f32).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text prompt (default: %(default)r).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to image file for multimodal generation.",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to audio file for multimodal generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Build and save model to DIR, then exit (no inference).",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with non-zero code on failure.",
    )
    args = parser.parse_args()

    is_multimodal = args.image or args.audio

    # ----- Export-only path -----
    if args.save_to:
        build_and_export(args.model, args.save_to, dtype=args.dtype)
        return

    # ----- Resolve model directory -----
    if args.model_dir:
        model_dir = args.model_dir
    else:
        suffix = "vlm" if is_multimodal else "text"
        default_dir = os.path.join("output", f"gemma4_{suffix}")
        model_dir = default_dir
        if not os.path.isfile(os.path.join(model_dir, "genai_config.json")):
            build_and_export(args.model, model_dir, dtype=args.dtype)

    # ----- Inference -----
    print("=" * 60)
    mode = "multimodal" if is_multimodal else "text"
    print(f"Gemma 4 — ORT GenAI ({mode}, device={args.device}, dtype={args.dtype})")
    print("=" * 60)

    if is_multimodal:
        output, _tps = generate_multimodal(
            model_dir,
            args.prompt,
            image_path=args.image,
            audio_path=args.audio,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    else:
        output, _tps = generate(
            model_dir,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )

    print(f"\n📝 Output: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if "--ci" in sys.argv:
            print(f"FAILED: {e}", file=sys.stderr)
            sys.exit(1)
        raise
