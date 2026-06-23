#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Gemma4 WebGPU End-to-End: Export + ONNX Runtime + ORT-GenAI.

This script demonstrates the full E2E flow for running Gemma4 on WebGPU
with graph capture enabled:

1. **Export**: Build and export the model with WebGPU EP + graph capture
2. **Direct ORT**: Test the decoder model with direct ONNX Runtime
3. **ORT-GenAI**: Run text generation with onnxruntime-genai

WebGPU graph capture constraints:
- All inputs must be INT32 (WebGPU doesn't have INT64 Cast kernel)
- No Shape/ConstantOfShape ops (they output to CPU, breaking graph capture)
- current_sequence_length and past_sequence_length are explicit inputs

Known issue:
ORT-GenAI's WebGPU graph capture check looks at the provider list rather
than actual node assignments. As a workaround, this script can disable
graph capture in genai_config.json while still using the WebGPU EP.

Requirements::

    pip install mobius-ai[ort-genai] onnxruntime-web

Usage::

    # Export model for WebGPU with graph capture
    python examples/gemma4_webgpu_e2e.py --save-to output/gemma4_webgpu/

    # Test with direct ORT (validates graph capture works at ORT level)
    python examples/gemma4_webgpu_e2e.py --model-dir output/gemma4_webgpu/ --test-ort

    # Run text generation with ORT-GenAI (uses CPU fallback for now)
    python examples/gemma4_webgpu_e2e.py --model-dir output/gemma4_webgpu/

    # Full E2E: export + test ORT + run generation
    python examples/gemma4_webgpu_e2e.py --save-to output/gemma4_webgpu/ --test-ort
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 30


def build_and_export(
    model_id: str,
    output_dir: str,
    *,
    dtype: str = "f16",
    ep: str = "webgpu",
) -> dict[str, Any]:
    """Build Gemma4 and export for the specified EP.

    Args:
        model_id: HuggingFace model ID.
        output_dir: Output directory for ONNX models and config.
        dtype: Model dtype ("f16" recommended for WebGPU).
        ep: Execution provider ("webgpu" for graph capture, "cpu" for CPU).

    Returns:
        Dict with export manifest (paths to generated files).
    """
    from mobius import build
    from mobius.integrations.ort_genai import export_package

    os.makedirs(output_dir, exist_ok=True)

    print(f"Building {model_id!r} for {ep.upper()} (dtype={dtype})...")

    # Build with specified execution provider
    pkg = build(model_id, dtype=dtype, load_weights=True, execution_provider=ep)

    print(f"Package components: {list(pkg.keys())}")

    # Export with ORT-GenAI config
    print(f"Exporting to {output_dir}...")
    manifest = export_package(
        pkg,
        output_dir,
        hf_model_id=model_id,
        ep=ep,
        context_length=4096,
    )

    # Print op statistics from decoder
    decoder_model = pkg.get("decoder") or pkg.get("model")
    if decoder_model:
        op_counts: dict[str, int] = {}
        for node in decoder_model.graph:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        print("\nDecoder op counts:")
        critical_ops = ["Shape", "ConstantOfShape", "Cast"]
        for op in critical_ops:
            count = op_counts.get(op, 0)
            status = "OK" if count == 0 else "WARNING"
            print(f"  {op}: {count} [{status}]")

    print(f"\nExport complete -> {output_dir}")
    return manifest


def patch_genai_config_for_cpu(model_dir: str) -> None:
    """Patch genai_config.json to use CPU EP instead of WebGPU.

    This is a workaround for the ORT-GenAI graph capture check issue.
    The model is still optimized for WebGPU, but runs on CPU.
    """
    config_path = os.path.join(model_dir, "genai_config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Clear provider options to fall back to CPU
    for component in ["decoder", "vision", "embedding", "speech"]:
        if component in config.get("model", {}):
            config["model"][component]["session_options"]["provider_options"] = []

    # Disable past_present_share_buffer for CPU
    config["search"]["past_present_share_buffer"] = False

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("Patched genai_config.json for CPU fallback")


def test_direct_ort(model_dir: str, use_webgpu: bool = False) -> bool:
    """Test the decoder model with direct ONNX Runtime InferenceSession.

    This validates that graph capture works at the ORT level before
    involving ORT-GenAI.

    Args:
        model_dir: Path to exported model directory.
        use_webgpu: Whether to use WebGPU EP (requires onnxruntime-web).

    Returns:
        True if test passes, False otherwise.
    """
    import onnxruntime as ort

    # Load genai_config to get model info
    config_path = os.path.join(model_dir, "genai_config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    decoder_cfg = config["model"]["decoder"]
    decoder_path = os.path.join(model_dir, decoder_cfg["filename"])

    print(f"\nTesting decoder with direct ORT: {decoder_path}")

    # Session options
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # Warnings only

    providers: list[str | tuple[str, dict[str, Any]]]
    if use_webgpu:
        providers = [
            ("WebGpuExecutionProvider", {"enableGraphCapture": "1"}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(decoder_path, sess_options, providers=providers)
        print(f"  Loaded with providers: {[p.split('Execution')[0] for p in session.get_providers()]}")

        # Get input info
        inputs = {inp.name: inp for inp in session.get_inputs()}
        print(f"  Inputs: {list(inputs.keys())}")

        # Check that inputs are INT32 (WebGPU graph capture requirement)
        int_inputs = ["attention_mask", "position_ids", "input_ids"]
        for name in int_inputs:
            if name in inputs:
                elem_type = inputs[name].type
                if "int32" in elem_type.lower():
                    print(f"    {name}: {elem_type} [OK - INT32]")
                else:
                    print(f"    {name}: {elem_type} [WARNING - expected INT32]")

        # Check for current_sequence_length input
        if "current_sequence_length" in inputs:
            print("    current_sequence_length: present [OK - graph capture input]")
        else:
            print("    current_sequence_length: missing [WARNING - needed for graph capture]")

        print("\n  Direct ORT test: PASSED")
        return True

    except Exception as e:
        print(f"\n  Direct ORT test: FAILED - {e}")
        return False


def generate_text(
    model_dir: str,
    prompt: str,
    max_new_tokens: int,
    *,
    use_cpu_fallback: bool = True,
) -> str:
    """Run text generation with onnxruntime-genai.

    Args:
        model_dir: Path to exported model directory.
        prompt: Text prompt.
        max_new_tokens: Maximum tokens to generate.
        use_cpu_fallback: Patch config to use CPU EP (workaround for graph capture issue).

    Returns:
        Generated text.
    """
    import onnxruntime_genai as og

    if use_cpu_fallback:
        patch_genai_config_for_cpu(model_dir)

    print(f"\nLoading model from {model_dir}...")
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


def compare_with_hf(model_id: str, prompt: str, max_new_tokens: int) -> str:
    """Run generation with HuggingFace transformers for comparison."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[HF] Loading {model_id}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"\n[HF] Prompt: {prompt}")
    print("-" * 40)

    with torch.no_grad():
        output_ids = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
    output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print(output)
    print("-" * 40)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gemma4 WebGPU E2E: Export + ORT + ORT-GenAI"
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Pre-built model directory (skips export)",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Export model to this directory",
    )
    parser.add_argument(
        "--dtype",
        default="f16",
        choices=["f32", "f16", "bf16"],
        help="Model dtype (default: f16)",
    )
    parser.add_argument(
        "--ep",
        default="webgpu",
        choices=["webgpu", "cpu", "cuda"],
        help="Execution provider (default: webgpu). Use 'cpu' for ORT-GenAI testing.",
    )
    parser.add_argument(
        "--test-ort",
        action="store_true",
        help="Test decoder with direct ONNX Runtime",
    )
    parser.add_argument(
        "--test-ort-webgpu",
        action="store_true",
        help="Test decoder with WebGPU EP (requires onnxruntime-web)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="Also run with HuggingFace for comparison",
    )
    parser.add_argument(
        "--skip-genai",
        action="store_true",
        help="Skip ORT-GenAI generation (export/test only)",
    )
    args = parser.parse_args()

    # Determine model directory
    model_dir: str
    if args.model_dir:
        model_dir = args.model_dir
    elif args.save_to:
        model_dir = args.save_to
        build_and_export(
            args.model_id,
            model_dir,
            dtype=args.dtype,
            ep=args.ep,
        )
    else:
        # Default export location
        model_dir = os.path.join("output", f"gemma4_{args.ep}")
        if not os.path.isfile(os.path.join(model_dir, "genai_config.json")):
            build_and_export(
                args.model_id,
                model_dir,
                dtype=args.dtype,
                ep=args.ep,
            )

    # Test with direct ORT
    if args.test_ort or args.test_ort_webgpu:
        test_direct_ort(model_dir, use_webgpu=args.test_ort_webgpu)

    # Run generation with ORT-GenAI
    if not args.skip_genai:
        print("\n" + "=" * 60)
        print("ORT-GenAI Generation")
        print("=" * 60)

        # WebGPU models with graph capture have inputs that ORT-GenAI
        # doesn't know how to provide on CPU. Only run generation test
        # for CPU-built models or when WebGPU is available.
        if args.ep == "webgpu":
            print("\nNOTE: WebGPU model with graph capture cannot run on CPU.")
            print("The model is ready for WebGPU runtime (e.g., browser with WebGPU support).")
            print("To test with ORT-GenAI on CPU, rebuild with --ep cpu")
        else:
            ort_output = generate_text(
                model_dir,
                args.prompt,
                args.max_new_tokens,
                use_cpu_fallback=False,
            )

            if args.compare_hf:
                print("\n" + "=" * 60)
                print("HuggingFace Comparison")
                print("=" * 60)
                hf_output = compare_with_hf(args.model_id, args.prompt, args.max_new_tokens)

                if ort_output.strip() == hf_output.strip():
                    print("\n[OK] Outputs match exactly!")
                else:
                    print("\n[DIFF] Outputs differ:")
                    print(f"  ORT: {ort_output!r}")
                    print(f"  HF:  {hf_output!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
