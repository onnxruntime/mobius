#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Gemma 4 generation with onnxruntime-genai.

Builds ONNX models for Gemma 4 (text-only or VLM), saves them in the flat
layout expected by onnxruntime-genai, and runs text generation.

**ORT GenAI model type support note:** The ``gemma4`` and ``gemma4_text``
model types are not yet in a released ORT GenAI build.  Until support lands,
text-only inference can be approximated with ``gemma3_text`` (missing: KV
sharing for the 20 shared layers, dual head_dim for global attention layers).
For VLM inference, there is no suitable fallback — this script requires a
build of ORT GenAI that supports the ``gemma4`` type.

Requirements::

    pip install mobius-onnx[ort-genai]

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
import sys

import onnxruntime_genai as og

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_IMAGE_PROMPT = "Describe this image in detail."
MAX_NEW_TOKENS = 50


# ---------------------------------------------------------------------------
# genai_config.json writer (derives all architecture values from config)
# ---------------------------------------------------------------------------


def _write_genai_config(
    config: object,
    mode: str,
    output_dir: str,
    *,
    bos_token_id: int | None,
    eos_token_id: int | list[int] | None,
    pad_token_id: int | None,
) -> None:
    """Write genai_config.json, deriving all values from the ArchitectureConfig.

    Args:
        config: Gemma4Config populated by ``build()``.
        mode: ``"text"`` for text-only decoder, ``"vlm"`` for 3-model VLM.
        output_dir: Directory to write ``genai_config.json``.
        bos_token_id: BOS token ID (from HuggingFace config).
        eos_token_id: EOS token ID(s) (from HuggingFace config).
        pad_token_id: PAD token ID (from HuggingFace config).

    NOTE: Uses model type ``gemma4_text`` (text) or ``gemma4`` (VLM).  These
    types are not yet in a released ORT GenAI build.  Until support lands,
    text-only inference can be approximated with ``gemma3_text`` (missing: KV
    sharing, dual head_dim).  For VLM there is no suitable fallback.
    """
    from mobius.integrations.ort_genai.genai_config import GenaiConfigGenerator

    # Gemma4 KV cache only covers layers with independent KV projections;
    # the last num_kv_shared_layers layers re-use KV from earlier layers.
    num_kv_shared: int = getattr(config, "num_kv_shared_layers", 0) or 0
    num_hidden_layers: int = getattr(config, "num_hidden_layers", 0)
    kv_cache_layers = num_hidden_layers - num_kv_shared

    model_type = "gemma4_text" if mode == "text" else "gemma4"
    generator = GenaiConfigGenerator(
        model_type,
        vocab_size=getattr(config, "vocab_size", 0),
        hidden_size=getattr(config, "hidden_size", 0),
        # Only layers with independent KV projections have cache entries.
        num_hidden_layers=kv_cache_layers,
        num_attention_heads=getattr(config, "num_attention_heads", 0),
        num_key_value_heads=getattr(config, "num_key_value_heads", 0),
        head_dim=getattr(config, "head_dim", 0),
        context_length=getattr(config, "max_position_embeddings", 4096),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    if mode == "vlm":
        vision_cfg = getattr(config, "vision", None)
        image_token_id: int = (
            getattr(vision_cfg, "image_token_id", None) or 255999  # boi_token_id
        )
        # Gemma4 SigLIP takes pre-patchified inputs (not raw images):
        #   pixel_values:       [batch, num_patches, 3 * patch_size^2]
        #   pixel_position_ids: [batch, num_patches, 2]  (row, col tile coords)
        # spatial_merge_size=None because Gemma4 doesn't use spatial merge.
        generator.with_vision(
            image_token_id=image_token_id,
            filename="vision.onnx",
            embedding_filename="embedding.onnx",
            spatial_merge_size=None,
            input_names={
                "pixel_values": "pixel_values",
                "pixel_position_ids": "pixel_position_ids",
            },
        )

    cfg = generator.generate()

    # Gemma4-specific: add sliding_window to decoder, derived from layer_types.
    # Layers at index i (within the kv_cache_layers range) whose type is
    # "sliding_attention" need the ORT GenAI sliding-window cache treatment.
    sliding_window: int | None = getattr(config, "sliding_window", None)
    layer_types: list[str] = getattr(config, "layer_types", None) or []
    if sliding_window:
        sliding_layer_indices = [
            i
            for i in range(kv_cache_layers)
            if i >= len(layer_types) or layer_types[i] == "sliding_attention"
        ]
        if sliding_layer_indices:
            cfg["model"]["decoder"]["sliding_window"] = {
                "window_size": sliding_window,
                "pad_value": 0,
                "alignment": "right",
                "slide_key_value_cache": True,
                "slide_inputs": True,
                "layers": sliding_layer_indices,
            }

    with open(os.path.join(output_dir, "genai_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)


def _write_processor_config(config: object, output_dir: str) -> None:
    """Write image_processor.json for the SigLIP vision encoder.

    Gemma 4 uses a SigLIP ViT with pan-and-scan tiling.  The ONNX vision
    model expects pre-patchified inputs; ORT GenAI's ``gemma4`` processor
    handles tiling and patchification internally using these parameters.
    All values are derived from the Gemma4Config vision sub-config.
    """
    vision_cfg = getattr(config, "vision", None)
    if vision_cfg is None:
        return
    processor_config = {
        "processor": {
            "name": "gemma4_image_processor",
            "image_size": getattr(vision_cfg, "image_size", 448),
            "patch_size": getattr(vision_cfg, "patch_size", 16),
            "tokens_per_image": getattr(vision_cfg, "mm_tokens_per_image", 256),
            # SigLIP normalisation constants (fixed for all SigLIP checkpoints).
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        }
    }
    with open(os.path.join(output_dir, "image_processor.json"), "w", encoding="utf-8") as f:
        json.dump(processor_config, f, indent=4)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def build_and_export(model_id: str, output_dir: str, mode: str) -> None:
    """Build the ONNX model package and write ORT GenAI config files.

    Derives all architecture constants from the Gemma4Config returned by
    ``build()``.  Token IDs (bos/eos/pad) are fetched from the HuggingFace
    config so they are never hardcoded in this script.

    Args:
        model_id: HuggingFace model ID.
        output_dir: Directory to write all outputs.
        mode: ``"text"`` for text-only decoder, ``"vlm"`` for 3-model VLM.
    """
    import transformers

    from mobius import build
    from mobius.integrations.ort_genai.auto_export import _copy_tokenizer_files

    print(f"Building {model_id!r} (mode={mode}) ...")
    pkg = build(model_id, dtype="f32", load_weights=True)
    print(f"Package components: {list(pkg.keys())}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving ONNX models to {output_dir!r} ...")
    pkg.save(output_dir)

    # Fetch token IDs from the HuggingFace config (not hardcoded).
    print(f"  Fetching token IDs from {model_id!r} ...")
    hf_config = transformers.AutoConfig.from_pretrained(model_id)
    bos_token_id = getattr(hf_config, "bos_token_id", None)
    eos_token_id = getattr(hf_config, "eos_token_id", None)
    pad_token_id = getattr(hf_config, "pad_token_id", None)

    print("Writing genai_config.json ...")
    _write_genai_config(
        pkg.config,
        mode,
        output_dir,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    if mode == "vlm":
        print("Writing image_processor.json ...")
        _write_processor_config(pkg.config, output_dir)

    print(f"  Copying tokenizer files from {model_id!r} ...")
    _copy_tokenizer_files(model_id, output_dir)
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
    patchification internally using ``image_processor.json``.
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
        help="Export/run mode: 'text' (decoder only) or 'vlm' (3-model VLM). Default: text.",
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
        hf_output = generate_hf(args.model_id, prompt, args.image, args.max_new_tokens)
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
