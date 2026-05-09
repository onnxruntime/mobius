#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-VL multimodal generation with ONNX Runtime.

Builds the 3-model ONNX package (decoder, vision encoder, embedding)
using the mobius ORT GenAI integration, then runs multimodal inference
using raw ONNX Runtime sessions with HuggingFace preprocessing.

Requirements::

    pip install mobius-ai[ort-genai] onnxruntime-gpu  # for CUDA
    pip install mobius-ai[ort-genai] onnxruntime       # for CPU only

Supported dtype/EP combinations::

    - CPU:  f32
    - CUDA: f32

    NOTE: f16/bf16 are NOT supported for this model. The Qwen3-VL-2B
    embedding weights have very small magnitudes (std≈0.01), causing f16
    underflow → NaN after 28 decoder layers. Use f32 for reliable results.

Usage::

    # Build and run with an image (CPU f32):
    python examples/qwen3_vl_ort_genai.py --image testdata/pipeline-cat-chonk.jpeg

    # CUDA f16:
    python examples/qwen3_vl_ort_genai.py \\
        --image testdata/pipeline-cat-chonk.jpeg --dtype f16 --ep cuda

    # Use a pre-built model directory:
    python examples/qwen3_vl_ort_genai.py \\
        --model-dir output/qwen3_vl/ --image <path> --ep cpu

    # Build and save (skip inference):
    python examples/qwen3_vl_ort_genai.py --save-to output/qwen3_vl/ --dtype f16 --ep cuda

    # Compare ORT output with HuggingFace transformers:
    python examples/qwen3_vl_ort_genai.py --image <path> --compare-hf
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

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
    processor_config.json, tokenizer files, and chat template — all derived
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
# ORT session helpers
# ---------------------------------------------------------------------------


def _ort_providers(ep: str) -> list[str]:
    """Return ORT execution providers for the given EP string."""
    if ep == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _ort_session(model_dir: str, subdir: str, ep: str) -> ort.InferenceSession:
    """Load an ONNX model as an ORT InferenceSession."""
    path = os.path.join(model_dir, subdir, "model.onnx")
    opts = ort.SessionOptions()
    # ORT 1.26 has a bug in EXTENDED graph optimizations for f16 CUDA
    # models (vector bounds assertion failure). Disable optimizations
    # for CUDA to avoid the crash.
    if ep == "cuda":
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        path,
        sess_options=opts,
        providers=_ort_providers(ep),
    )


def _run_session(
    session: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run an ORT session and return outputs as a dict."""
    output_names = [o.name for o in session.get_outputs()]
    results = session.run(output_names, feeds)
    return dict(zip(output_names, results))


def _numpy_dtype_for_ort(ort_type: str) -> np.dtype:
    """Map ORT type string to numpy dtype."""
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    return np.dtype(mapping.get(ort_type, np.float32))


def _get_feed_dtype(session: ort.InferenceSession, name: str) -> np.dtype:
    """Get the expected numpy dtype for a session input by name."""
    for inp in session.get_inputs():
        if inp.name == name:
            return _numpy_dtype_for_ort(inp.type)
    return np.dtype(np.float32)


# ---------------------------------------------------------------------------
# MRoPE position IDs
# ---------------------------------------------------------------------------


def _compute_mrope_position_ids(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray | None,
    image_token_id: int,
    spatial_merge_size: int = 2,
) -> np.ndarray:
    """Compute 3D MRoPE position IDs for the Qwen-VL decoder.

    Returns shape ``(3, batch, seq_len)`` where the three dimensions
    are (temporal, height, width).  For text tokens all three are
    identical (sequential position).  For image tokens, each dimension
    tracks the corresponding spatial/temporal position within the image
    grid.
    """
    batch_size, seq_len = input_ids.shape
    position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)

    for b in range(batch_size):
        text_pos = 0
        image_idx = 0
        i = 0
        while i < seq_len:
            if image_grid_thw is not None and input_ids[b, i] == image_token_id:
                # Span of consecutive image tokens
                img_start = i
                while i < seq_len and input_ids[b, i] == image_token_id:
                    i += 1

                t, h, w = image_grid_thw[image_idx]
                image_idx += 1
                merge_h = h // spatial_merge_size
                merge_w = w // spatial_merge_size

                idx = img_start
                for ti in range(t):
                    for hi in range(merge_h):
                        for wi in range(merge_w):
                            if idx < i:
                                position_ids[0, b, idx] = text_pos + ti
                                position_ids[1, b, idx] = text_pos + hi
                                position_ids[2, b, idx] = text_pos + wi
                                idx += 1

                text_pos += max(t, merge_h, merge_w)
            else:
                position_ids[0, b, i] = text_pos
                position_ids[1, b, i] = text_pos
                position_ids[2, b, i] = text_pos
                text_pos += 1
                i += 1

    return position_ids


# ---------------------------------------------------------------------------
# Generation (raw ORT sessions + HF preprocessing)
# ---------------------------------------------------------------------------


def generate_with_image(
    model_dir: str,
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
    ep: str = "cpu",
) -> str:
    """Run multimodal generation using raw ORT sessions.

    Steps:
    1. HF processor → pixel_values, image_grid_thw, input_ids
    2. Vision encoder(pixel_values, image_grid_thw) → image_features
    3. Embedding(input_ids, image_features) → inputs_embeds
    4. Compute 3D MRoPE position_ids
    5. Decoder prefill(inputs_embeds, position_ids, empty KV) → logits
    6. Autoregressive decode loop with KV cache
    """
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor

    # ----- HF preprocessing -----
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
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    image_grid_thw = inputs["image_grid_thw"].numpy().astype(np.int64)

    # ----- Load ORT sessions -----
    print(f"Loading model from {model_dir!r} (ep={ep}) ...")
    vision_sess = _ort_session(model_dir, "vision_encoder", ep)
    embed_sess = _ort_session(model_dir, "embedding", ep)
    decoder_sess = _ort_session(model_dir, "decoder", ep)

    # Derive the model dtype from decoder session inputs
    model_dtype = _get_feed_dtype(decoder_sess, "inputs_embeds")

    # ----- Load HF config for model dimensions -----
    config = AutoConfig.from_pretrained(model_id)
    text_config = getattr(config, "text_config", config)
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    hidden_size = text_config.hidden_size
    image_token_id = getattr(
        config,
        "image_token_id",
        processor.tokenizer.convert_tokens_to_ids(
            getattr(processor, "image_token", "<|image_pad|>")
        ),
    )
    spatial_merge_size = getattr(
        config,
        "spatial_merge_size",
        getattr(
            getattr(config, "vision_config", None),
            "spatial_merge_size",
            2,
        ),
    )
    eos_token_id = text_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    # ----- Step 1: Vision encoder -----
    vision_out = _run_session(
        vision_sess,
        {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        },
    )
    image_features = vision_out["image_features"]

    # Sanity check: image feature count matches image token count
    num_image_tokens = int((input_ids == image_token_id).sum())
    assert image_features.shape[0] == num_image_tokens, (
        f"image_features ({image_features.shape[0]}) != image tokens ({num_image_tokens})"
    )

    # ----- Step 2: Embedding -----
    embed_dtype = _get_feed_dtype(embed_sess, "image_features")
    embed_out = _run_session(
        embed_sess,
        {
            "input_ids": input_ids,
            "image_features": image_features.astype(embed_dtype),
        },
    )
    inputs_embeds = embed_out["inputs_embeds"].astype(model_dtype)

    # ----- Step 3: Compute MRoPE position IDs -----
    position_ids = _compute_mrope_position_ids(
        input_ids,
        image_grid_thw,
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
    )

    # ----- Step 4: Decoder prefill -----
    batch_size = 1
    seq_len = inputs_embeds.shape[1]
    kv_dtype = _get_feed_dtype(decoder_sess, "past_key_values.0.key")

    feeds: dict[str, np.ndarray] = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
        "position_ids": position_ids,
    }
    # Empty KV cache for prefill
    for i in range(num_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim),
            dtype=kv_dtype,
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim),
            dtype=kv_dtype,
        )

    outputs = _run_session(decoder_sess, feeds)

    print(f"\nPrompt: {prompt}")
    print(f"Image:  {image_path}")
    print("-" * 40)

    # ----- Step 5: Autoregressive decode -----
    logits = outputs["logits"].astype(np.float32)
    next_token = int(np.argmax(logits[0, -1]))
    generated_ids: list[int] = [next_token]

    # Update KV cache
    kv_cache: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        kv_cache[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
        kv_cache[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

    past_seq_len = seq_len
    max_pos = int(position_ids.max())

    # Use HF tokenizer for streaming output
    hf_tokenizer = processor.tokenizer

    for _ in range(max_new_tokens - 1):
        if next_token == eos_token_id:
            break

        # Embed the new token
        cur_ids = np.array([[next_token]], dtype=np.int64)
        embed_out = _run_session(
            embed_sess,
            {
                "input_ids": cur_ids,
                "image_features": np.zeros(
                    (0, hidden_size),
                    dtype=embed_dtype,
                ),
            },
        )
        cur_embed = embed_out["inputs_embeds"].astype(model_dtype)

        max_pos += 1
        pos = np.array(
            [[[max_pos]], [[max_pos]], [[max_pos]]],
            dtype=np.int64,
        )
        total_seq_len = past_seq_len + 1

        feeds = {
            "inputs_embeds": cur_embed,
            "attention_mask": np.ones(
                (batch_size, total_seq_len),
                dtype=np.int64,
            ),
            "position_ids": pos,
            **kv_cache,
        }
        outputs = _run_session(decoder_sess, feeds)

        # Update KV cache
        for i in range(num_layers):
            kv_cache[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
            kv_cache[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]
        past_seq_len = total_seq_len

        logits = outputs["logits"].astype(np.float32)
        next_token = int(np.argmax(logits[0, -1]))
        generated_ids.append(next_token)

    # Decode all generated tokens at once
    output = hf_tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(output)

    print(output)
    print("-" * 40)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL multimodal generation with ONNX Runtime.",
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
        help="Also run HuggingFace transformers and compare outputs.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with non-zero code on failure (for CI pipelines).",
    )
    parser.add_argument(
        "--ep",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Execution provider (default: %(default)s).",
    )
    parser.add_argument(
        "--dtype",
        default="f32",
        help="Data type for ONNX model (default: %(default)s).",
    )
    args = parser.parse_args()

    # ----- Export-only path -----
    if args.save_to:
        build_and_export(
            args.model_id,
            args.save_to,
            dtype=args.dtype,
            ep=args.ep,
        )
        return

    # ----- Require --image for inference -----
    if not args.image:
        parser.error(
            "--image is required for inference. Qwen3-VL 3-model split "
            "requires the multimodal pipeline. Use --save-to for "
            "export-only."
        )

    # ----- Resolve model directory -----
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.path.join("output", "qwen3_vl")
        if not os.path.isfile(os.path.join(model_dir, "genai_config.json")):
            build_and_export(
                args.model_id,
                model_dir,
                dtype=args.dtype,
                ep=args.ep,
            )

    # ----- Inference -----
    prompt = args.prompt or DEFAULT_PROMPT

    print("=" * 60)
    print("ONNX Runtime")
    print("=" * 60)
    onnx_output = generate_with_image(
        model_dir,
        args.model_id,
        prompt,
        args.image,
        args.max_new_tokens,
        ep=args.ep,
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
