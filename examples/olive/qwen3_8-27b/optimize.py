#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Assemble a complete Q4_K_M Qwen3.8 three-model package with Olive."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import onnx_ir as ir
from inference import MODEL_ID, REVISION

_ASSETS = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor.json",
}


def olive_config(
    decoder: Path,
    output: Path,
    *,
    nodes_to_exclude: list[str] | None = None,
) -> dict:
    """Return the CPU-isolated Olive Q4_K_M decoder-only workflow."""
    return {
        "input_model": {"type": "OnnxModel", "model_path": str(decoder)},
        "passes": {
            "q4_k_m": {
                "type": "OnnxKQuantQuantization",
                "bits": 4,
                "block_size": 32,
                "save_as_external_data": True,
                "all_tensors_to_one_file": True,
                "external_data_name": "decoder.onnx.data",
                "size_threshold": 1024,
                "nodes_to_exclude": nodes_to_exclude or [],
            }
        },
        "engine": {
            "target": {
                "type": "LocalSystem",
                "accelerators": [
                    {"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}
                ],
            }
        },
        "no_artifacts": True,
        "output_dir": str(output),
        # A global Olive cache can return a decoder produced with a different
        # exclusion set. Scope and clean it so recurrent-gate policy is exact.
        "cache_dir": str(output.parent / ".olive-cache"),
        "clean_cache": True,
    }


def _recurrent_gate_nodes(decoder: Path) -> list[str]:
    """Keep DeltaNet decay/time-step gates in f16 to preserve recurrent stability."""
    model = ir.load(decoder)
    return [
        node.name
        for node in model.graph.all_nodes()
        if node.op_type == "MatMul"
        and ("/linear_attn/in_proj_a/" in node.name or "/linear_attn/in_proj_b/" in node.name)
    ]


def quantize_package(source_dir: str | Path, output_dir: str | Path) -> Path:
    """Quantize only decoder and copy vision, embedding, metadata, and tokenizer."""
    import olive.systems.local as olive_local
    from olive.workflows import run as olive_run

    source, output = Path(source_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    decoder = source / "decoder" / "model.onnx"
    if not decoder.is_file():
        raise FileNotFoundError(decoder)
    olive_output = output / ".olive"
    # Olive can eagerly register unrelated GPU EP DLLs. K-quant is a
    # weight-only CPU pass, so suppress that registration only for this call.
    preserved_fp16_nodes = _recurrent_gate_nodes(decoder)
    register = olive_local.maybe_register_ep_libraries
    olive_local.maybe_register_ep_libraries = lambda _paths: None
    try:
        olive_run(
            olive_config(
                decoder,
                olive_output,
                nodes_to_exclude=preserved_fp16_nodes,
            )
        )
    finally:
        olive_local.maybe_register_ep_libraries = register
    models = list(olive_output.rglob("*.onnx"))
    if len(models) != 1:
        raise RuntimeError(f"Expected one Olive decoder, found {models}")
    decoder_dir = output / "decoder"
    decoder_dir.mkdir()
    for item in models[0].parent.iterdir():
        if item.is_file():
            shutil.copy2(item, decoder_dir / item.name)
    produced = decoder_dir / models[0].name
    if produced != decoder_dir / "model.onnx":
        produced.replace(decoder_dir / "model.onnx")
    shutil.rmtree(olive_output)
    olive_cache = output / ".olive-cache"
    if olive_cache.exists():
        shutil.rmtree(olive_cache)
    for name in ("embedding", "vision_encoder"):
        shutil.copytree(source / name, output / name)
    for name in _ASSETS:
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "quantization": "Q4_K_M",
        "quantized_component": "decoder",
        "olive_provider": "CPUExecutionProvider",
        "preserved_fp16_recurrent_gate_nodes": len(preserved_fp16_nodes),
        "components": [
            "decoder/model.onnx",
            "embedding/model.onnx",
            "vision_encoder/model.onnx",
        ],
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output
