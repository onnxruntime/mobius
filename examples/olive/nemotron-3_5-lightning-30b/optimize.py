#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned BF16 export and Olive INT4 packaging for Nemotron 3.5 Lightning."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from inference import MODEL_ID, REVISION, run_token_ids

_METADATA_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
}


def _require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _save_pinned_metadata(output_dir: Path) -> None:
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=False,
    )
    config.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tokenizer.save_pretrained(output_dir)
    generation = GenerationConfig.from_pretrained(MODEL_ID, revision=REVISION)
    generation.save_pretrained(output_dir)

    (output_dir / "source_manifest.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "runtime": "onnxruntime-direct",
                "ort_genai_supported": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def export_checkpoint(output_dir: str | Path, *, ep: str) -> Path:
    """Export the pinned BF16 checkpoint as a supported FP16 ONNX package."""
    from mobius import build
    from mobius._flags import override_flags

    output = Path(output_dir)
    _require_empty_output(output)
    with override_flags(ort_cuda_grouped_rmsnorm_workaround=ep == "cuda"):
        package = build(
            MODEL_ID,
            revision=REVISION,
            dtype="f16",
            load_weights=True,
            trust_remote_code=False,
            execution_provider=ep,
        )
    package.save(output, external_data="onnx")
    _save_pinned_metadata(output)
    manifest_path = output / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"source_dtype": "bf16", "dtype": "f16", "target_ep": ep})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def _olive_config(source_model: Path, output_dir: Path, precision: str) -> dict:
    if precision == "q4_k_m":
        pass_config = {
            "type": "OnnxKQuantQuantization",
            "bits": 4,
            "block_size": 32,
        }
    elif precision == "nf4":
        pass_config = {
            "type": "OnnxBnb4Quantization",
            "precision": "nf4",
        }
    else:
        raise ValueError(f"Unsupported quantization precision: {precision}")

    pass_config.update(
        {
            "save_as_external_data": True,
            "all_tensors_to_one_file": True,
            "external_data_name": "model.onnx.data",
            "size_threshold": 1024,
        }
    )
    return {
        "input_model": {
            "type": "OnnxModel",
            "model_path": str(source_model),
        },
        "passes": {precision: pass_config},
        "engine": {
            "target": {
                "type": "LocalSystem",
                "accelerators": [
                    {
                        "device": "cpu",
                        "execution_providers": ["CPUExecutionProvider"],
                    }
                ],
            }
        },
        "no_artifacts": True,
        "output_dir": str(output_dir),
    }


def _find_olive_model(output_dir: Path) -> Path:
    candidates = list(output_dir.rglob("*.onnx"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one Olive ONNX output under {output_dir}, got {candidates}"
        )
    return candidates[0]


def _copy_olive_model(model_path: Path, destination: Path) -> None:
    for child in model_path.parent.iterdir():
        if child.is_file():
            shutil.copy2(child, destination / child.name)
    copied_model = destination / model_path.name
    canonical_model = destination / "model.onnx"
    if copied_model != canonical_model:
        copied_model.replace(canonical_model)


def quantize_package(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    precision: str = "q4_k_m",
) -> Path:
    """Quantize model.onnx with a CPU-isolated Olive workflow."""
    import olive.systems.local as olive_local
    from olive.workflows import run as olive_run

    source = Path(source_dir)
    source_model = source / "model.onnx"
    if not source_model.is_file():
        raise FileNotFoundError(f"Missing source model: {source_model}")
    output = Path(output_dir)
    _require_empty_output(output)

    with tempfile.TemporaryDirectory(prefix="olive-nemotron-") as temp:
        olive_output = Path(temp) / "output"
        # Olive 0.13 auto-registers every DLL bundled in a GPU ORT wheel,
        # even for a CPU-only target. That makes an unrelated TensorRT DLL
        # failure abort weight-only quantization. Suppress registration for
        # this pass; the explicit workflow target remains CPU-only.
        register_ep_libraries = olive_local.maybe_register_ep_libraries
        olive_local.maybe_register_ep_libraries = lambda _paths: None
        try:
            olive_run(_olive_config(source_model, olive_output, precision))
        finally:
            olive_local.maybe_register_ep_libraries = register_ep_libraries
        _copy_olive_model(_find_olive_model(olive_output), output)

    for name in _METADATA_FILES | {"source_manifest.json"}:
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
    manifest_path = output / "source_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"model_id": MODEL_ID, "revision": REVISION}
    )
    manifest.update({"quantization": precision, "olive_provider": "CPUExecutionProvider"})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def smoke_test(model_dir: str | Path, *, device: str) -> list[int]:
    """Load the assembled package and perform cached multi-token generation."""
    import numpy as np

    generated, logits, _profile = run_token_ids(
        model_dir,
        [1, 42, 17],
        max_new_tokens=4,
        device=device,
    )
    if len(generated) != 4 or any(not np.isfinite(step).all() for step in logits):
        raise RuntimeError(f"Quantized generation smoke test failed: {generated}")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="output/f16/cuda")
    parser.add_argument("--output-dir", default="output/Q4_K_M/cuda")
    parser.add_argument("--ep", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--precision", choices=["q4_k_m", "nf4"], default="q4_k_m")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-quantization", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    if not args.skip_export:
        export_checkpoint(args.source_dir, ep=args.ep)
    result_dir = Path(args.source_dir)
    if not args.skip_quantization:
        result_dir = quantize_package(
            args.source_dir,
            args.output_dir,
            precision=args.precision,
        )
    if not args.skip_smoke:
        print("Generated token IDs:", smoke_test(result_dir, device=args.ep))


if __name__ == "__main__":
    main()
