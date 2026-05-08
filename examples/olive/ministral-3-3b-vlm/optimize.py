#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export Ministral-3-3B VLM to ONNX with optional Olive quantization.

Uses mobius for all 3 sub-model exports (text decoder, vision encoder,
embedding) and ORT GenAI config generation. Optionally applies Olive
ModelBuilder quantization (INT4/FP16) to the text decoder.

Usage::

    # Pure mobius export (FP16)
    python optimize.py

    # With Olive INT4 quantization for text decoder (CPU)
    python optimize.py --olive-config cpu_and_mobile/text.json

    # With Olive FP16 quantization for text decoder (CUDA)
    python optimize.py --olive-config cuda/text.json --ep cuda

    # Custom output directory
    python optimize.py --output-dir output/ministral3
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

DEFAULT_OUTPUT_DIR = "models"
DEFAULT_HF_MODEL = "mistralai/Ministral-3-3B-Instruct-2512"


def export_models(
    model_path: str,
    output_dir: str,
    dtype: str = "f16",
    ep: str = "cpu",
):
    """Build and save all 3 sub-models with mobius.

    Uses mobius.build() which constructs the ONNX graphs
    declaratively (no torch.onnx.export), avoiding dynamo
    issues with Pixtral's dynamic image dimensions.
    """
    from mobius import build
    from mobius.integrations.ort_genai import (
        write_ort_genai_config,
    )

    print(f"=== Building VLM from {model_path} ===")
    print(f"  dtype={dtype}, ep={ep}")

    pkg = build(model_path, dtype=dtype, load_weights=True)
    print(f"  Components: {list(pkg.keys())}")

    print(f"\n=== Saving to {output_dir} in Model Package layout ===")
    from mobius.integrations.ort_genai.auto_export import _resolve_component_map

    pkg.save_package_layout(output_dir, component_map=_resolve_component_map(pkg))

    print("\n=== Generating ORT GenAI Model Package config ===")
    write_ort_genai_config(
        pkg,
        output_dir,
        hf_model_id=model_path,
    )

    print("  Export complete")


def quantize_text_decoder(olive_config: str, output_dir: str):
    """Quantize text decoder using Olive ModelBuilder.

    Replaces the mobius-exported text decoder with an
    Olive-quantized version (INT4 or FP16 with GQA).

    Rewrites the Olive config's output_dir at runtime to
    match the user-provided output directory.
    """
    import json
    import tempfile

    try:
        from olive import run
    except ImportError:
        from olive.workflows import run

    config_path = Path(olive_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Olive config not found: {config_path}")

    # Rewrite output_dir in Olive config to match user's
    # --output-dir, so paths stay coordinated
    with open(config_path) as f:
        olive_cfg = json.load(f)
    olive_cfg["output_dir"] = str(Path(output_dir) / "text.onnx")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        dir=config_path.parent,
    ) as tmp:
        json.dump(olive_cfg, tmp, indent=4)
        tmp_path = tmp.name

    try:
        print(f"\n=== Olive quantization: {config_path} ===")
        run(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Move Olive output into decoder/ subdirectory
    olive_output = Path(output_dir) / "text.onnx"
    decoder_dir = Path(output_dir) / "decoder"
    if olive_output.is_dir():
        if decoder_dir.exists():
            shutil.rmtree(decoder_dir)
        decoder_dir.mkdir(exist_ok=True)
        for f in olive_output.iterdir():
            if f.is_file():
                shutil.move(str(f), str(decoder_dir / f.name))
        shutil.rmtree(olive_output)
        print(f"  Replaced decoder with Olive output in {decoder_dir}")


def main():
    """Run the export pipeline."""
    parser = argparse.ArgumentParser(
        description=("Export Ministral-3-3B VLM to ONNX with optional Olive quantization")
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_HF_MODEL,
        help=("HuggingFace model ID or local path to dequantized checkpoint"),
    )
    parser.add_argument(
        "--dtype",
        default="f16",
        choices=["f16", "f32", "bf16"],
        help="Model dtype (default: %(default)s)",
    )
    parser.add_argument(
        "--ep",
        default="cpu",
        choices=["cpu", "cuda", "dml"],
        help="Execution provider (default: %(default)s)",
    )
    parser.add_argument(
        "--olive-config",
        default=None,
        help=(
            "Path to Olive JSON config for text decoder "
            "quantization (e.g. cpu_and_mobile/text.json)"
        ),
    )
    args = parser.parse_args()

    # Step 1: Export all models with mobius
    export_models(
        args.model_path,
        args.output_dir,
        args.dtype,
        args.ep,
    )

    # Step 2: Optionally quantize text decoder with Olive
    if args.olive_config:
        quantize_text_decoder(args.olive_config, args.output_dir)

    print("\n=== Done ===")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
