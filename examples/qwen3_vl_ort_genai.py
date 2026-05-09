#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-VL generation with onnxruntime-genai.

Builds the 3-model ONNX package (decoder, vision encoder, embedding),
saves it in the flat layout expected by onnxruntime-genai, and runs
multimodal generation with an image.

Qwen3-VL uses the same 3-model I/O contract as Qwen2.5-VL, so it can
be loaded by onnxruntime-genai with ``model.type = "qwen2_5_vl"``.

NOTE: Text-only generation is not supported with the 3-model VLM split
because the decoder expects ``inputs_embeds`` and ORT GenAI does not
currently route ``input_ids`` through the embedding model for text-only.
Use ``--image`` for all generation.

Requirements::

    pip install mobius-ai[ort-genai]

Usage::

    # Multimodal generation (required — text-only not supported):
    python examples/qwen3_vl_ort_genai.py --image testdata/pipeline-cat-chonk.jpeg

    # Compare ORT GenAI output with HuggingFace transformers:
    python examples/qwen3_vl_ort_genai.py --image testdata/pipeline-cat-chonk.jpeg --compare-hf

    # Use a pre-built model directory:
    python examples/qwen3_vl_ort_genai.py --model-dir output/qwen3vl/ --image <path>

    # Build with specific dtype and EP:
    python examples/qwen3_vl_ort_genai.py --dtype bf16 --ep cuda --image <path>

    # Build and save (skip inference):
    python examples/qwen3_vl_ort_genai.py --save-to output/qwen3vl/ --dtype f16 --ep cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import onnxruntime_genai as og

# ---------------------------------------------------------------------------
# Model export helpers
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_PROMPT = "Describe this image in detail."
MAX_NEW_TOKENS = 50


def build_and_export(
    model_id: str,
    output_dir: str,
    dtype: str = "f32",
    ep: str = "cpu",
) -> None:
    """Build the 3-model ONNX package and save for onnxruntime-genai.

    Uses the mobius ORT GenAI integration to build the package,
    generate genai_config.json, and copy tokenizer/processor files
    from HuggingFace automatically.
    """
    from mobius import build
    from mobius.integrations.ort_genai import export_package

    print(f"Building {model_id!r} (dtype={dtype}, ep={ep}) ...")
    pkg = build(
        model_id,
        dtype=dtype,
        execution_provider=ep,
        load_weights=True,
    )
    print(f"Package components: {list(pkg.keys())}")

    print(f"Exporting to {output_dir} ...")
    export_package(
        pkg,
        output_dir,
        hf_model_id=model_id,
        ep=ep,
    )
    print("Export complete.")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(model_dir: str, prompt: str, max_new_tokens: int) -> str:
    """Run text-only generation with onnxruntime-genai.

    NOTE: Qwen3-VL 3-model split requires the full multimodal pipeline
    (embedding model converts input_ids → inputs_embeds). ORT GenAI
    does not currently support text-only generation without an image
    for this pipeline type. Use ``--image`` for multimodal generation,
    or use the ``qwen35_text_generation.py`` example for text-only.
    """
    print(f"Loading model from {model_dir} ...")
    print(
        "WARNING: Text-only generation is not supported with "
        "Qwen3-VL 3-model split. The decoder expects inputs_embeds, "
        "but ORT GenAI's append_tokens only provides input_ids.\n"
        "Use --image <path> for multimodal generation."
    )
    sys.exit(1)


def generate_with_image(
    model_dir: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
) -> str:
    """Run multimodal generation with onnxruntime-genai.

    Uses the ORT GenAI multimodal processor to encode the image
    into pixel_values + image_grid_thw alongside the tokenized prompt.
    """
    from transformers import AutoProcessor

    # The chat template encodes <|image_pad|> tokens for the image
    processor = AutoProcessor.from_pretrained(MODEL_ID)
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

    print(f"Loading model from {model_dir} ...")
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)
    ort_processor = model.create_multimodal_processor()

    # Load the image via ORT GenAI's image processor
    images = og.Images.open(image_path)
    inputs = ort_processor(text, images=images)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_new_tokens + 512)

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
# HuggingFace transformers generation (for comparison)
# ---------------------------------------------------------------------------


def generate_text_hf(model_id: str, prompt: str, max_new_tokens: int) -> str:
    """Run text-only generation with HuggingFace transformers."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[HF] Loading {model_id} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], return_tensors="pt").to("cpu")

    print(f"\n[HF] Prompt: {prompt}")
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
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to("cpu")

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


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL text generation with onnxruntime-genai.",
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
        "--prompt",
        default=None,
        help="Text prompt (default depends on whether --image is used).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to image file for multimodal generation.",
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
        help="Execution provider for ONNX build (e.g. cpu, cuda). "
        "Defaults to matching --device.",
    )
    parser.add_argument(
        "--dtype",
        default="f32",
        help="Data type for ONNX model (default: %(default)s).",
    )
    args = parser.parse_args()
    ep = args.ep or args.device

    if args.save_to:
        build_and_export(args.model_id, args.save_to, dtype=args.dtype, ep=ep)
        return

    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.path.join("output", "qwen3_vl")
        if not os.path.isfile(os.path.join(model_dir, "genai_config.json")):
            build_and_export(args.model_id, model_dir, dtype=args.dtype, ep=ep)

    if args.image:
        prompt = args.prompt or "Describe this image in detail."
        print("=" * 60)
        print("ORT GenAI")
        print("=" * 60)
        onnx_output = generate_with_image(model_dir, prompt, args.image, args.max_new_tokens)
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
                print("\n\u2713 Outputs match exactly!")
            else:
                print("\n\u2717 Outputs differ!")
                print(f"  ONNX: {onnx_output!r}")
                print(f"  HF:   {hf_output!r}")
                if args.ci:
                    sys.exit(1)
    else:
        prompt = args.prompt or DEFAULT_PROMPT
        print("=" * 60)
        print("ORT GenAI")
        print("=" * 60)
        onnx_output = generate(model_dir, prompt, args.max_new_tokens)
        if args.compare_hf:
            print("\n" + "=" * 60)
            print("HuggingFace Transformers")
            print("=" * 60)
            hf_output = generate_text_hf(args.model_id, prompt, args.max_new_tokens)
            if onnx_output.strip() == hf_output.strip():
                print("\n\u2713 Outputs match exactly!")
            else:
                print("\n\u2717 Outputs differ!")
                print(f"  ONNX: {onnx_output!r}")
                print(f"  HF:   {hf_output!r}")
                if args.ci:
                    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if "--ci" in sys.argv:
            print(f"FAILED: {e}", file=sys.stderr)
            sys.exit(1)
        raise
