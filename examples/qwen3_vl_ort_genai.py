#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Qwen3-VL multimodal generation with ONNX Runtime.

Builds the 3-model ONNX package (decoder, vision encoder, embedding)
using the mobius ORT GenAI integration, then runs multimodal inference
using raw ONNX Runtime sessions with HuggingFace preprocessing.

Requirements::

    pip install mobius-ai[ort-genai] onnxruntime-gpu  # for CUDA
    pip install mobius-ai[ort-genai] onnxruntime       # for CPU only

Supported dtype/EP combinations::

    - CPU:  f32
    - CUDA: f32, f16, bf16

    For bf16, torch IOBinding is used because numpy lacks bfloat16.
    Requires onnxruntime-gpu >= 1.27 for f16/bf16 CUDA support.

Usage::

    # CPU f32 (default):
    python examples/qwen3_vl_ort_genai.py \
        --image testdata/pipeline-cat-chonk.jpeg

    # CUDA f16:
    python examples/qwen3_vl_ort_genai.py \
        --image testdata/pipeline-cat-chonk.jpeg --dtype f16 --ep cuda

    # CUDA bf16:
    python examples/qwen3_vl_ort_genai.py \
        --image testdata/pipeline-cat-chonk.jpeg --dtype bf16 --ep cuda

    # Build and save (skip inference):
    python examples/qwen3_vl_ort_genai.py \
        --save-to output/qwen3_vl/ --dtype f16 --ep cuda

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
    """Build the 3-model ONNX package via the mobius ORT GenAI integration."""
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

    print(f"Exporting to {output_dir!r} ...")
    export_package(pkg, output_dir, hf_model_id=model_id)
    print(f"Export complete -> {output_dir}")


# ---------------------------------------------------------------------------
# ORT session helpers
# ---------------------------------------------------------------------------


_CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_CPU_PROVIDERS = ["CPUExecutionProvider"]


def _make_session(
    model_dir: str,
    subdir: str,
    providers: list[str],
) -> ort.InferenceSession:
    """Load an ONNX model as an ORT InferenceSession."""
    path = os.path.join(model_dir, subdir, "model.onnx")
    opts = ort.SessionOptions()
    # ORT 1.26 EXTENDED/ALL graph optimizations crash for f16/bf16 on
    # CUDA (vector bounds assertion in transformer_memcpy).  BASIC is
    # safe and still runs constant folding + redundant-node elimination.
    if any("CUDA" in p for p in providers):
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(
        path,
        sess_options=opts,
        providers=providers,
    )


def _run(
    sess: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run an ORT session and return outputs as a dict."""
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


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

    Returns shape ``(3, batch, seq_len)`` — (temporal, height, width).
    """
    batch_size, seq_len = input_ids.shape
    pos = np.zeros((3, batch_size, seq_len), dtype=np.int64)

    for b in range(batch_size):
        text_pos = 0
        img_idx = 0
        i = 0
        while i < seq_len:
            if image_grid_thw is not None and input_ids[b, i] == image_token_id:
                start = i
                while i < seq_len and input_ids[b, i] == image_token_id:
                    i += 1
                t, h, w = image_grid_thw[img_idx]
                img_idx += 1
                mh = h // spatial_merge_size
                mw = w // spatial_merge_size
                idx = start
                for ti in range(t):
                    for hi in range(mh):
                        for wi in range(mw):
                            if idx < i:
                                pos[0, b, idx] = text_pos + ti
                                pos[1, b, idx] = text_pos + hi
                                pos[2, b, idx] = text_pos + wi
                                idx += 1
                text_pos += max(t, mh, mw)
            else:
                pos[:, b, i] = text_pos
                text_pos += 1
                i += 1
    return pos


# ---------------------------------------------------------------------------
# Generation: f32 / f16 path (numpy session.run)
# ---------------------------------------------------------------------------


def _generate_numpy(
    model_dir: str,
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
    ep: str,
    dtype: str,
) -> str:
    """Generate text using numpy-based session.run().

    Works for f32 and f16.  All sessions run on the requested EP.
    """
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor

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

    config = AutoConfig.from_pretrained(model_id)
    tc = getattr(config, "text_config", config)
    image_token_id = config.image_token_id
    sms = config.vision_config.spatial_merge_size
    eos = tc.eos_token_id
    if isinstance(eos, list):
        eos = eos[0]

    # Vision encoder: CUDA if available (PackedMHA is CUDA-only)
    vision_provs = _CUDA_PROVIDERS if ep == "cuda" else _CPU_PROVIDERS
    decode_provs = _CUDA_PROVIDERS if ep == "cuda" else _CPU_PROVIDERS

    print(f"Loading model from {model_dir!r} (ep={ep}) ...")
    vsess = _make_session(model_dir, "vision_encoder", vision_provs)
    esess = _make_session(model_dir, "embedding", decode_provs)
    dsess = _make_session(model_dir, "decoder", decode_provs)

    # Detect model dtype from decoder inputs
    model_dt = np.float32
    for inp in dsess.get_inputs():
        if inp.name == "inputs_embeds":
            if inp.type == "tensor(float16)":
                model_dt = np.float16
            break

    # Vision encoder
    vout = _run(
        vsess,
        {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        },
    )
    img_feat = vout["image_features"].astype(model_dt)

    # Embedding
    eout = _run(
        esess,
        {
            "input_ids": input_ids,
            "image_features": img_feat,
        },
    )
    embeds = eout["inputs_embeds"]

    # MRoPE position IDs
    position_ids = _compute_mrope_position_ids(
        input_ids,
        image_grid_thw,
        image_token_id,
        sms,
    )

    # Decoder prefill
    seq_len = embeds.shape[1]
    feeds: dict[str, np.ndarray] = {
        "inputs_embeds": embeds,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": position_ids,
    }
    for i in range(tc.num_hidden_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, tc.num_key_value_heads, 0, tc.head_dim),
            dtype=model_dt,
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, tc.num_key_value_heads, 0, tc.head_dim),
            dtype=model_dt,
        )
    outputs = _run(dsess, feeds)

    print(f"\nPrompt: {prompt}")
    print(f"Image:  {image_path}")
    print("-" * 40)

    # Greedy decode
    logits = outputs["logits"].astype(np.float32)
    token = int(np.argmax(logits[0, -1]))
    generated: list[int] = [token]
    kv = {
        f"past_key_values.{i}.{t}": outputs[f"present.{i}.{t}"]
        for i in range(tc.num_hidden_layers)
        for t in ("key", "value")
    }
    past_len = seq_len
    max_pos = int(position_ids.max())

    for _ in range(max_new_tokens - 1):
        if token == eos:
            break
        eout = _run(
            esess,
            {
                "input_ids": np.array([[token]], dtype=np.int64),
                "image_features": np.zeros(
                    (0, tc.hidden_size),
                    dtype=model_dt,
                ),
            },
        )
        max_pos += 1
        feeds = {
            "inputs_embeds": eout["inputs_embeds"],
            "attention_mask": np.ones(
                (1, past_len + 1),
                dtype=np.int64,
            ),
            "position_ids": np.full(
                (3, 1, 1),
                max_pos,
                dtype=np.int64,
            ),
            **kv,
        }
        outputs = _run(dsess, feeds)
        kv = {
            f"past_key_values.{i}.{t}": outputs[f"present.{i}.{t}"]
            for i in range(tc.num_hidden_layers)
            for t in ("key", "value")
        }
        past_len += 1
        logits = outputs["logits"].astype(np.float32)
        token = int(np.argmax(logits[0, -1]))
        generated.append(token)

    output = processor.tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )
    print(output)
    print("-" * 40)
    return output


# ---------------------------------------------------------------------------
# Generation: bf16 path (torch IOBinding)
# ---------------------------------------------------------------------------


def _generate_bf16(
    model_dir: str,
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
) -> str:
    """Generate text for bf16 models using torch IOBinding.

    ORT's Python API cannot handle bfloat16 numpy arrays, so we use
    torch tensors + ``OrtValue.from_dlpack`` to feed bf16 data via
    IOBinding.
    """
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor

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

    config = AutoConfig.from_pretrained(model_id)
    tc = getattr(config, "text_config", config)
    image_token_id = config.image_token_id
    sms = config.vision_config.spatial_merge_size
    eos = tc.eos_token_id
    if isinstance(eos, list):
        eos = eos[0]

    print(f"Loading model from {model_dir!r} (ep=cuda, bf16) ...")
    vsess = _make_session(model_dir, "vision_encoder", _CUDA_PROVIDERS)
    esess = _make_session(model_dir, "embedding", _CUDA_PROVIDERS)
    dsess = _make_session(model_dir, "decoder", _CUDA_PROVIDERS)

    bf = torch.bfloat16

    def _run_io(
        sess: ort.InferenceSession,
        feed: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        io = sess.io_binding()
        for name, t in feed.items():
            io.bind_ortvalue_input(name, ort.OrtValue.from_dlpack(t))
        for out in sess.get_outputs():
            io.bind_output(out.name)
        sess.run_with_iobinding(io)
        return [torch.from_dlpack(o).clone() for o in io.get_outputs()]

    # Vision encoder
    v_outs = _run_io(
        vsess,
        {
            "pixel_values": torch.tensor(pixel_values),
            "image_grid_thw": torch.tensor(
                image_grid_thw,
                dtype=torch.int64,
            ),
        },
    )
    img_feat = v_outs[0]

    # Embedding
    e_outs = _run_io(
        esess,
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "image_features": img_feat,
        },
    )
    embeds = e_outs[0]

    # MRoPE position IDs
    position_ids = _compute_mrope_position_ids(
        input_ids,
        image_grid_thw,
        image_token_id,
        sms,
    )
    seq_len = embeds.shape[1]

    # Decoder prefill
    d_feed: dict[str, torch.Tensor] = {
        "inputs_embeds": embeds,
        "attention_mask": torch.ones(1, seq_len, dtype=torch.int64),
        "position_ids": torch.tensor(
            position_ids,
            dtype=torch.int64,
        ),
    }
    for i in range(tc.num_hidden_layers):
        empty = torch.zeros(
            1,
            tc.num_key_value_heads,
            0,
            tc.head_dim,
            dtype=bf,
        )
        d_feed[f"past_key_values.{i}.key"] = empty
        d_feed[f"past_key_values.{i}.value"] = empty

    d_outs = _run_io(dsess, d_feed)

    print(f"\nPrompt: {prompt}")
    print(f"Image:  {image_path}")
    print("-" * 40)

    logits = d_outs[0].float().cpu().numpy()
    token = int(np.argmax(logits[0, -1]))
    generated: list[int] = [token]

    kv: dict[str, torch.Tensor] = {}
    for i in range(tc.num_hidden_layers):
        kv[f"past_key_values.{i}.key"] = d_outs[1 + 2 * i]
        kv[f"past_key_values.{i}.value"] = d_outs[2 + 2 * i]

    past_len = seq_len
    max_pos = int(position_ids.max())

    for _ in range(max_new_tokens - 1):
        if token == eos:
            break

        e_outs = _run_io(
            esess,
            {
                "input_ids": torch.tensor(
                    [[token]],
                    dtype=torch.int64,
                ),
                "image_features": torch.zeros(
                    0,
                    tc.hidden_size,
                    dtype=bf,
                ),
            },
        )
        max_pos += 1
        d_feed = {
            "inputs_embeds": e_outs[0],
            "attention_mask": torch.ones(
                1,
                past_len + 1,
                dtype=torch.int64,
            ),
            "position_ids": torch.full(
                (3, 1, 1),
                max_pos,
                dtype=torch.int64,
            ),
            **kv,
        }
        d_outs = _run_io(dsess, d_feed)
        for i in range(tc.num_hidden_layers):
            kv[f"past_key_values.{i}.key"] = d_outs[1 + 2 * i]
            kv[f"past_key_values.{i}.value"] = d_outs[2 + 2 * i]
        past_len += 1

        logits = d_outs[0].float().cpu().numpy()
        token = int(np.argmax(logits[0, -1]))
        generated.append(token)

    output = processor.tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )
    print(output)
    print("-" * 40)
    return output


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def generate_with_image(
    model_dir: str,
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
    ep: str = "cpu",
    dtype: str = "f32",
) -> str:
    """Run multimodal generation, dispatching by dtype."""
    if dtype == "bf16":
        return _generate_bf16(
            model_dir,
            model_id,
            prompt,
            image_path,
            max_new_tokens,
        )
    return _generate_numpy(
        model_dir,
        model_id,
        prompt,
        image_path,
        max_new_tokens,
        ep,
        dtype,
    )


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
        help="Path to image file (required for inference).",
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
        help="Also run HuggingFace transformers and compare.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with non-zero code on failure.",
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

    if args.save_to:
        build_and_export(
            args.model_id,
            args.save_to,
            dtype=args.dtype,
            ep=args.ep,
        )
        return

    if not args.image:
        parser.error("--image is required for inference. Use --save-to for export-only.")

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
        dtype=args.dtype,
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
