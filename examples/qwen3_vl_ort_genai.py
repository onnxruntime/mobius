#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-VL multimodal generation with onnxruntime-genai.

Builds the 3-model ONNX package (decoder, vision encoder, embedding)
using the mobius ORT GenAI integration, then runs multimodal inference.

Requirements::

    pip install mobius-ai[ort-genai]

Usage::

    # Build and run with an image:
    python examples/qwen3_vl_ort_genai.py --image testdata/pipeline-cat-chonk.jpeg

    # Build with specific dtype and EP:
    python examples/qwen3_vl_ort_genai.py --dtype bf16 --ep cuda --image <path>

    # Use a pre-built model directory:
    python examples/qwen3_vl_ort_genai.py --model-dir output/qwen3_vl/ --image <path>

    # Build and save (skip inference):
    python examples/qwen3_vl_ort_genai.py --save-to output/qwen3_vl/ --dtype f16 --ep cuda

    # Compare ORT GenAI output with HuggingFace transformers:
    python examples/qwen3_vl_ort_genai.py --image <path> --compare-hf

NOTE: Text-only generation is not currently supported with the 3-model
VLM split.  The decoder expects ``inputs_embeds``, and ORT GenAI does
not route ``input_ids`` through the embedding model for text-only prompts.
Always provide ``--image`` for inference.
"""

from __future__ import annotations

import argparse
import os
import sys

import onnxruntime_genai as og

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_PROMPT = "Describe this image in detail."
MAX_NEW_TOKENS = 50


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_and_export(
    model_id: str,
    output_dir: str,
    dtype: str = "f32",
    ep: str = "cpu",
) -> None:
    """Build the 3-model ONNX package via the mobius ORT GenAI integration.

    ``export_package`` handles everything: ONNX model save, genai_config.json,
    image_processor.json, tokenizer files, and chat template — all derived
    automatically from the HuggingFace config.
    """
    from mobius import build
    from mobius.integrations.ort_genai import export_package

    print(f"Building {model_id!r} (dtype={dtype}, ep={ep}) ...")
    pkg = build(model_id, dtype=dtype, execution_provider=ep, load_weights=True)
    print(f"Package components: {list(pkg.keys())}")

    print(f"Exporting to {output_dir!r} ...")
    export_package(pkg, output_dir, hf_model_id=model_id, ep=ep)
    print(f"Export complete → {output_dir}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_with_image(
    model_dir: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
) -> str:
    """Run multimodal generation with onnxruntime-genai.

    Uses the ORT GenAI multimodal processor + HF chat template.
    """
    from transformers import AutoProcessor

    print(f"Loading model from {model_dir!r} ...")
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)
    processor = model.create_multimodal_processor()

    # Apply chat template with image placeholder
    hf_proc = AutoProcessor.from_pretrained(MODEL_ID)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = hf_proc.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    images = og.Images.open(image_path)
    inputs = processor(text, images=images)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_new_tokens + 4096)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    print(f"\nPrompt: {prompt}")
    print(f"Image:  {image_path}")
    print("-" * 40)

    tokenizer_stream = tokenizer.create_stream()
    generated_tokens: list[int] = []
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        generated_tokens.append(token)
        print(tokenizer_stream.decode(token), end="", flush=True)
        if len(generated_tokens) >= max_new_tokens:
            break

    print()
    print("-" * 40)

    output = tokenizer.decode(generated_tokens)
    del generator
    return output


# ---------------------------------------------------------------------------
# HuggingFace comparison
# ---------------------------------------------------------------------------


def generate_with_image_hf(
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
) -> str:
    """Run multimodal generation with HuggingFace transformers."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[HF] Loading {model_id} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cpu")

    print(f"\n[HF] Prompt: {prompt}")
    print(f"[HF] Image:  {image_path}")
    print("-" * 40)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen_ids = output_ids[:, inputs.input_ids.shape[1] :]
    output = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print(output)
    print("-" * 40)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL multimodal generation with onnxruntime-genai.",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Pre-built model directory. Skip export if provided.",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Export the model to DIR and exit (no inference).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to image file for multimodal generation (required for inference).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt (default: %(default)s).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="Also run with HuggingFace transformers and compare outputs.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with non-zero code on failure (for CI pipelines).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (default: %(default)s).",
    )
    parser.add_argument(
        "--ep",
        default=None,
        help="Execution provider for ONNX build (default: matches --device).",
    )
    parser.add_argument(
        "--dtype",
        default="f32",
        help="Data type for ONNX model (default: %(default)s).",
    )
    args = parser.parse_args()
    ep = args.ep or args.device

    # ----- Export-only path -----
    if args.save_to:
        build_and_export(args.model_id, args.save_to, dtype=args.dtype, ep=ep)
        return

    # ----- Require --image for inference -----
    if not args.image:
        parser.error(
            "--image is required for inference. Qwen3-VL 3-model split "
            "requires the multimodal pipeline. Use --save-to for export-only."
        )

    # ----- Resolve model directory -----
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.path.join("output", "qwen3_vl")
        if not os.path.isfile(os.path.join(model_dir, "genai_config.json")):
            build_and_export(args.model_id, model_dir, dtype=args.dtype, ep=ep)

    # ----- Inference -----
    prompt = args.prompt or DEFAULT_PROMPT

    print("=" * 60)
    print("ORT GenAI")
    print("=" * 60)
    onnx_output = generate_with_image(
        model_dir,
        prompt,
        args.image,
        args.max_new_tokens,
    )

    if args.compare_hf:
        print("\n" + "=" * 60)
        print("HuggingFace Transformers")
        print("=" * 60)
        hf_output = generate_with_image_hf(
            args.model_id,
            prompt,
            args.image,
            args.max_new_tokens,
        )
        if onnx_output.strip() == hf_output.strip():
            print("\n✓ Outputs match exactly!")
        else:
            print("\n✗ Outputs differ!")
            print(f"  ONNX: {onnx_output!r}")
            print(f"  HF:   {hf_output!r}")
            if args.ci:
                sys.exit(1)


if __name__ == "__main__":
    main()
