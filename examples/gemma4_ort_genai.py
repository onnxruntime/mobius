#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma 4 generation with onnxruntime-genai.

Builds ONNX models for Gemma 4 (text-only or VLM), saves them in the flat
layout expected by onnxruntime-genai, and runs text generation.

**ORT GenAI model type support note:** The ``gemma4`` and ``gemma4_text``
model types are not yet in a released ORT GenAI build.  Until support lands,
text-only inference can be approximated with ``gemma3_text`` (missing: KV
sharing for the 20 shared layers, dual head_dim for global attention layers).
For VLM inference, there is no suitable fallback — this script requires a
build of ORT GenAI that supports the ``gemma4`` type.

Requirements::

    pip install mobius-ai[ort-genai]

Usage::

    # Text-only generation (text-decoder ONNX):
    python examples/gemma4_ort_genai.py

    # VLM generation (decoder + vision encoder + embedding):
    python examples/gemma4_ort_genai.py --mode vlm

    # With an image (requires --mode vlm):
    python examples/gemma4_ort_genai.py --mode vlm \\
        --image testdata/pipeline-cat-chonk.jpeg

    # Compare ORT GenAI output with HuggingFace transformers:
    python examples/gemma4_ort_genai.py --compare-hf

    # Use a pre-built model directory:
    python examples/gemma4_ort_genai.py --model-dir output/gemma4_text/

    # Build and save (skip inference):
    python examples/gemma4_ort_genai.py --save-to output/gemma4_text/
    python examples/gemma4_ort_genai.py --mode vlm --save-to output/gemma4_vlm/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import onnxruntime_genai as og

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_IMAGE_PROMPT = "Describe this image in detail."
MAX_NEW_TOKENS = 50

# Gemma 4 E2B-it architecture constants (google/gemma-4-E2B-it)
_VOCAB_SIZE = 262144
_HIDDEN_SIZE = 1536
_NUM_ATTENTION_HEADS = 8
_NUM_KEY_VALUE_HEADS = 1
_HEAD_SIZE = 256  # local/sliding attention head_dim
_NUM_TOTAL_LAYERS = 35
_NUM_KV_SHARED_LAYERS = 20
_NUM_KV_CACHE_LAYERS = _NUM_TOTAL_LAYERS - _NUM_KV_SHARED_LAYERS  # 15
_CONTEXT_LENGTH = 131072
_SLIDING_WINDOW_SIZE = 512
# Sliding-window layer indices within the 15-entry KV cache
# (layers 0-3, 5-8, 10-13 are local; layers 4, 9, 14 are global)
_SLIDING_LAYERS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
_BOS_TOKEN_ID = 2
_EOS_TOKEN_IDS = [1, 106]
_PAD_TOKEN_ID = 0
_IMAGE_TOKEN_ID = 255999  # boi_token_id

# SigLIP vision encoder constants
_VISION_IMAGE_SIZE = 448
_VISION_PATCH_SIZE = 16
_VISION_TOKENS_PER_IMAGE = 280


# ---------------------------------------------------------------------------
# genai_config.json writers
# ---------------------------------------------------------------------------


def _write_text_genai_config(output_dir: str) -> None:
    """Write genai_config.json for text-only decoder.

    NOTE: Uses model type ``gemma4_text``.  Until ORT GenAI ships this
    type, you can try ``gemma3_text`` as an approximation (no KV sharing,
    no dual head_dim support).
    """
    config = {
        "model": {
            "type": "gemma4_text",
            "vocab_size": _VOCAB_SIZE,
            "context_length": _CONTEXT_LENGTH,
            "bos_token_id": _BOS_TOKEN_ID,
            "eos_token_id": _EOS_TOKEN_IDS,
            "pad_token_id": _PAD_TOKEN_ID,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": "model.onnx",
                "hidden_size": _HIDDEN_SIZE,
                "head_size": _HEAD_SIZE,
                "num_attention_heads": _NUM_ATTENTION_HEADS,
                "num_key_value_heads": _NUM_KEY_VALUE_HEADS,
                # Only 15 layers produce independent KV cache entries;
                # the remaining 20 share KV projections from earlier layers.
                "num_hidden_layers": _NUM_KV_CACHE_LAYERS,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "sliding_window": {
                    "window_size": _SLIDING_WINDOW_SIZE,
                    "pad_value": 0,
                    "alignment": "right",
                    "slide_key_value_cache": True,
                    "slide_inputs": True,
                    "layers": _SLIDING_LAYERS,
                },
            },
        },
        "search": {
            "do_sample": False,
            "early_stopping": True,
            "max_length": 8192,
            "min_length": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    with open(os.path.join(output_dir, "genai_config.json"), "w") as f:
        json.dump(config, f, indent=4)


def _write_vlm_genai_config(output_dir: str) -> None:
    """Write genai_config.json for the 3-model VLM split.

    NOTE: Uses model type ``gemma4``.  This type is not yet in a released
    ORT GenAI build.  There is no suitable fallback type for VLM inference.
    """
    config = {
        "model": {
            "type": "gemma4",
            "vocab_size": _VOCAB_SIZE,
            "context_length": _CONTEXT_LENGTH,
            "bos_token_id": _BOS_TOKEN_ID,
            "eos_token_id": _EOS_TOKEN_IDS,
            "pad_token_id": _PAD_TOKEN_ID,
            "image_token_id": _IMAGE_TOKEN_ID,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": "model.onnx",
                "hidden_size": _HIDDEN_SIZE,
                "head_size": _HEAD_SIZE,
                "num_attention_heads": _NUM_ATTENTION_HEADS,
                "num_key_value_heads": _NUM_KEY_VALUE_HEADS,
                "num_hidden_layers": _NUM_KV_CACHE_LAYERS,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "sliding_window": {
                    "window_size": _SLIDING_WINDOW_SIZE,
                    "pad_value": 0,
                    "alignment": "right",
                    "slide_key_value_cache": True,
                    "slide_inputs": True,
                    "layers": _SLIDING_LAYERS,
                },
            },
            "vision": {
                "filename": "vision.onnx",
                "config_filename": "processor_config.json",
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                # Gemma 4 SigLIP takes pre-patchified inputs (not raw images):
                # pixel_values:      [batch, num_patches, 3 * patch_size^2]
                # pixel_position_ids:[batch, num_patches, 2]  (row, col coords)
                "inputs": {
                    "pixel_values": "pixel_values",
                    "pixel_position_ids": "pixel_position_ids",
                },
                "outputs": {
                    "image_features": "image_features",
                },
            },
            "embedding": {
                "filename": "embedding.onnx",
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "inputs": {
                    "input_ids": "input_ids",
                    "image_features": "image_features",
                },
                "outputs": {
                    "inputs_embeds": "inputs_embeds",
                },
            },
        },
        "search": {
            "do_sample": False,
            "early_stopping": True,
            "max_length": 8192,
            "min_length": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    with open(os.path.join(output_dir, "genai_config.json"), "w") as f:
        json.dump(config, f, indent=4)


def _write_processor_config(output_dir: str) -> None:
    """Write processor_config.json for the SigLIP vision encoder.

    Gemma 4 uses a SigLIP ViT (patch_size=16, image_size=448) with
    pan-and-scan tiling.  The ONNX vision model expects pre-patchified
    inputs; ORT GenAI's ``gemma4`` processor handles the tiling and
    patchification internally using these parameters.
    """
    processor_config = {
        "processor": {
            "name": "gemma4_image_processor",
            "image_size": _VISION_IMAGE_SIZE,
            "patch_size": _VISION_PATCH_SIZE,
            "tokens_per_image": _VISION_TOKENS_PER_IMAGE,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        }
    }
    with open(os.path.join(output_dir, "processor_config.json"), "w") as f:
        json.dump(processor_config, f, indent=4)


def _copy_tokenizer(model_id: str, output_dir: str) -> None:
    """Save tokenizer files from the HuggingFace processor."""
    from transformers import AutoTokenizer

    print(f"  Saving tokenizer from {model_id!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def build_and_export(model_id: str, output_dir: str, mode: str) -> None:
    """Build the ONNX model package and write ORT GenAI config files.

    Args:
        model_id: HuggingFace model ID.
        output_dir: Directory to write all outputs.
        mode: ``"text"`` for text-only decoder, ``"vlm"`` for 3-model VLM.
    """
    from mobius import build

    # Select the correct mobius model type
    hf_model_type = "gemma4_text" if mode == "text" else "gemma4"
    print(f"Building {model_id!r} (type={hf_model_type}) ...")
    pkg = build(model_id, dtype="f32", load_weights=True)
    print(f"Package components: {list(pkg.keys())}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving ONNX models to {output_dir!r} ...")
    pkg.save(output_dir)

    print("Writing genai_config.json ...")
    if mode == "text":
        _write_text_genai_config(output_dir)
    else:
        _write_vlm_genai_config(output_dir)
        _write_processor_config(output_dir)

    _copy_tokenizer(model_id, output_dir)
    print(f"Export complete → {output_dir}")


# ---------------------------------------------------------------------------
# Text-only generation
# ---------------------------------------------------------------------------


def generate_text(model_dir: str, prompt: str, max_new_tokens: int) -> str:
    """Run text-only generation with onnxruntime-genai.

    Requires a released ORT GenAI build that supports the ``gemma4_text``
    model type.  See the module docstring for the fallback workaround.
    """
    print(f"Loading model from {model_dir!r} ...")
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)

    input_ids = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_ids) + max_new_tokens)

    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)

    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    tokenizer_stream = tokenizer.create_stream()
    generated_tokens: list[int] = []
    for _ in range(max_new_tokens):
        if generator.is_done():
            break
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        generated_tokens.append(token)
        print(tokenizer_stream.decode(token), end="", flush=True)

    print()
    print("-" * 40)

    del generator
    return tokenizer.decode(generated_tokens)


# ---------------------------------------------------------------------------
# VLM generation (text + optional image)
# ---------------------------------------------------------------------------


def generate_vlm(
    model_dir: str,
    prompt: str,
    image_path: str | None,
    max_new_tokens: int,
) -> str:
    """Run VLM generation with onnxruntime-genai.

    Requires an ORT GenAI build that supports the ``gemma4`` model type
    with the Gemma 4 VLM pipeline (decoder + vision encoder + embedding).

    Gemma 4's vision encoder takes pre-patchified inputs:
    - ``pixel_values [batch, num_patches, 3*patch_size^2]``
    - ``pixel_position_ids [batch, num_patches, 2]``

    The ORT GenAI multimodal processor handles image tiling and
    patchification internally using ``processor_config.json``.
    """
    print(f"Loading model from {model_dir!r} ...")
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)

    if image_path is not None:
        processor = model.create_multimodal_processor()
        images = og.Images.open(image_path)

        # Build the chat-template prompt with image placeholder
        from transformers import AutoProcessor as HFProcessor

        hf_proc = HFProcessor.from_pretrained(MODEL_ID)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        full_prompt = hf_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(full_prompt, images=images)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=8192)
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        print(f"\nPrompt: {prompt}")
        print(f"Image:  {image_path}")
    else:
        # Text-only path through the VLM decoder (no image features)
        input_ids = tokenizer.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(input_ids) + max_new_tokens)
        generator = og.Generator(model, params)
        generator.append_tokens(input_ids)

        print(f"\nPrompt: {prompt}")

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

    del generator
    return tokenizer.decode(generated_tokens)


# ---------------------------------------------------------------------------
# HuggingFace transformers comparison
# ---------------------------------------------------------------------------


def generate_hf(
    model_id: str,
    prompt: str,
    image_path: str | None,
    max_new_tokens: int,
) -> str:
    """Run generation with HuggingFace transformers for comparison."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[HF] Loading {model_id!r} ...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    if image_path is not None:
        from PIL import Image

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
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cpu")
        print(f"\n[HF] Prompt: {prompt}")
        print(f"[HF] Image:  {image_path}")
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors="pt").to("cpu")
        print(f"\n[HF] Prompt: {prompt}")

    print("-" * 40)
    with torch.no_grad():
        output_ids = hf_model.generate(
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 generation with onnxruntime-genai.",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "vlm"],
        default="text",
        help="Export/run mode: 'text' (decoder only) or 'vlm' (3-model VLM). "
        "Default: text.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Pre-built model directory. Skips export when provided.",
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
        help="Path to image file for VLM generation (requires --mode vlm).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt override.",
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
    args = parser.parse_args()

    if args.image and args.mode == "text":
        parser.error("--image requires --mode vlm")

    # ----- Export-only path -----
    if args.save_to:
        build_and_export(args.model_id, args.save_to, mode=args.mode)
        return

    # ----- Resolve model directory -----
    if args.model_dir:
        model_dir = args.model_dir
    else:
        default_dir = os.path.join("output", f"gemma4_{args.mode}")
        model_dir = default_dir
        if not os.path.isfile(os.path.join(model_dir, "genai_config.json")):
            build_and_export(args.model_id, model_dir, mode=args.mode)

    # ----- Inference -----
    prompt = args.prompt or (DEFAULT_IMAGE_PROMPT if args.image else DEFAULT_PROMPT)

    print("=" * 60)
    print(f"ORT GenAI  (mode={args.mode})")
    print("=" * 60)

    if args.mode == "vlm":
        onnx_output = generate_vlm(model_dir, prompt, args.image, args.max_new_tokens)
    else:
        onnx_output = generate_text(model_dir, prompt, args.max_new_tokens)

    if args.compare_hf:
        print("\n" + "=" * 60)
        print("HuggingFace Transformers")
        print("=" * 60)
        hf_output = generate_hf(
            args.model_id, prompt, args.image, args.max_new_tokens
        )
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
