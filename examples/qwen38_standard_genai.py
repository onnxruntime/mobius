#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qualify an existing standard Qwen3.8 VLM with the native GenAI pipeline.

Never rebuilds/saves ONNX weights. The destination is a fresh metadata overlay:
component directories are symlinks; mutable HF metadata/tokenizers are copies.
Generation is gated on the tiny native hybrid progression/reset receipt.
The proof is a local operator attestation; rerun after any runtime binary change.
CUDA is selected at execution through og.Config, not a Mobius CLI flag.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.integrations.ort_genai import write_ort_genai_config

_COMPONENTS = ("decoder", "vision_encoder", "embedding")
_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
_CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _disable_core_dumps() -> None:
    """Disable POSIX core dumps to avoid huge GPU crash artifacts."""
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def prepare_overlay(source: Path, assets: Path, output: Path) -> dict[str, Any]:
    """Reconstruct a lazy package without a manifest or tensor materialization."""
    from transformers import AutoConfig

    for path in (source, assets, output):
        if not path.is_absolute():
            raise ValueError(f"Use absolute paths, got {path}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Overlay must be fresh, refusing to overwrite {output}")
    source = source.resolve(strict=True)
    assets = assets.resolve(strict=True)
    output = output.resolve()
    if output == source or source in output.parents or assets in output.parents:
        raise ValueError("Overlay must not be inside an immutable source directory")
    if _hash(assets / "config.json") != _CONFIG_SHA256:
        raise ValueError("HF config does not match the pinned Qwen3.8-27B revision")
    hf_config = AutoConfig.from_pretrained(str(assets), local_files_only=True)
    config = ArchitectureConfig.from_transformers(
        hf_config.text_config, parent_config=hf_config
    )
    # ir.load keeps external tensors lazy. Do not call const_value.numpy(),
    # ModelPackage.load(), save(), or the HF model constructor here.
    models = {name: ir.load(source / name / "model.onnx") for name in _COMPONENTS}
    for name, model in models.items():
        if model.functions or any(
            node.domain not in {"", "ai.onnx"} for node in model.graph.all_nodes()
        ):
            raise ValueError(f"{name} is not function-free standard ONNX")
    decoder = models["decoder"]
    inputs = {value.name: value for value in decoder.graph.inputs}
    outputs = {value.name: value for value in decoder.graph.outputs}
    kv = sorted(int(name.split(".")[1]) for name in inputs if name and name.endswith(".key"))
    if kv != list(range(3, 64, 4)):
        raise ValueError(f"Unexpected global KV indices: {kv}")
    for index in range(64):
        roles = (
            (("key", ir.DataType.FLOAT16), ("value", ir.DataType.FLOAT16))
            if index % 4 == 3
            else (("conv_state", ir.DataType.FLOAT16), ("recurrent_state", ir.DataType.FLOAT))
        )
        for role, dtype in roles:
            if (
                inputs[f"past_key_values.{index}.{role}"].dtype != dtype
                or outputs[f"present.{index}.{role}"].dtype != dtype
            ):
                raise ValueError(f"Wrong dtype for global slot {index}.{role}")
    if inputs["position_ids"].shape[0] != 3:
        raise ValueError("Expected native 3D MRoPE positions")
    graph_stats = {
        name: {"bytes": (source / name / "model.onnx").stat().st_size, "functions": 0}
        for name in _COMPONENTS
    }
    asset_names = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    )
    asset_hashes = {name: _hash(assets / name) for name in asset_names}
    output.mkdir(parents=True)
    for name in _COMPONENTS:
        (output / name).symlink_to(source / name, target_is_directory=True)
    # Copy even large tokenizers (small relative to weights): asset patches and
    # the writer must never follow a symlink back into the pinned original.
    local_assets = output / "hf-assets"
    local_assets.mkdir()
    for filename in asset_hashes:
        shutil.copy2(assets / filename, local_assets / filename)
    package = ModelPackage(models, config=config)
    result = write_ort_genai_config(
        package,
        str(output),
        hf_model_id=str(local_assets),
        # This is runtime metadata, not graph construction: every graph above
        # remains standard-only. Explicit auxiliary CPU sessions do not follow
        # Config.append_provider in the installed native multimodal pipeline.
        ep="cuda",
    )
    payload = json.loads((output / "genai_config.json").read_text())
    if payload["model"]["type"] != "qwen3_5":
        raise ValueError("Expected native Qwen3.5 multimodal dispatch")
    if payload["model"]["decoder"]["num_hidden_layers"] != 64:
        raise ValueError("Runtime config compacted sparse global cache slots")
    if payload["search"]["past_present_share_buffer"] is not False:
        raise ValueError("Standard dynamic KV caches must not share buffers")
    for name in ("decoder", "vision", "embedding"):
        providers = payload["model"][name]["session_options"]["provider_options"]
        if providers[0]["cuda"]["enable_cuda_graph"] != "0":
            raise ValueError("CUDA runtime overlay must keep graph capture disabled")
    for filename, digest in asset_hashes.items():
        if _hash(assets / filename) != digest:
            raise RuntimeError(f"Immutable source asset changed: {filename}")
    receipt = {
        "revision": _REVISION,
        "source": str(source),
        "assets": str(assets),
        "overlay": str(output),
        "global_kv_indices": kv,
        "graphs": graph_stats,
        "original_asset_hashes": asset_hashes,
        "artifacts": result,
    }
    (output / "overlay-receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def _generate_native(
    model: Any,
    processor: Any,
    tokenizer: Any,
    prompt: str,
    images: Any,
    max_new_tokens: int,
) -> list[int]:
    import onnxruntime_genai as og

    inputs = processor(prompt, images=images) if images is not None else processor(prompt)
    params = og.GeneratorParams(model)
    # Count native-expanded image placeholders, not the unexpanded prompt.
    prompt_length = int(inputs["input_ids"].as_numpy().size)
    params.set_search_options(
        max_length=prompt_length + max_new_tokens,
        do_sample=False,
        past_present_share_buffer=False,
    )
    generator = og.Generator(model, params)
    print("Native prefill:", prompt_length, "tokens; image:", images is not None, flush=True)
    generator.set_inputs(inputs)
    print("Native prefill completed", flush=True)
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        if generator.is_done():
            break
        generator.generate_next_token()
        tokens.append(int(generator.get_next_tokens()[0]))
    if not tokens:
        raise RuntimeError("Native generation produced no tokens")
    print("Native tokens:", tokens, "decoded:", tokenizer.decode(tokens), flush=True)
    del generator, params, inputs
    gc.collect()
    return tokens


def qualify(
    output: Path,
    proof: Path,
    image: Path | None,
    max_new_tokens: int,
    second_image: Path | None = None,
) -> None:
    """Run native text/reset and optional single-/two-image decoding.

    ``second_image`` is a programmatic optional extension, not a new CLI flag.
    """
    if not 1 <= max_new_tokens <= 16:
        raise ValueError("max_new_tokens must be between 1 and 16")
    if second_image is not None and image is None:
        raise ValueError("second_image requires the first image")

    import onnxruntime_genai as og

    receipt = json.loads(proof.read_text())
    expected = {
        "genai_version": og.__version__,
        "provider": "CUDAExecutionProvider",
        "sparse_kv_indices": [3, 7],
        "conv_dtype": "float16",
        "recurrent_dtype": "float32",
        "past_present_share_buffer": False,
        "progression_and_reset": [[0, 4, 8], [0, 4, 8]],
    }
    if receipt != expected:
        raise ValueError(
            "A passing tiny native proof with matching reported runtime version is required"
        )
    config = og.Config(str(output))
    config.clear_providers()
    config.append_provider("CUDAExecutionProvider")
    print("Loading native real VLM:", output, flush=True)
    model = og.Model(config)
    print("Native real VLM loaded", flush=True)
    tokenizer = og.Tokenizer(model)
    processor = model.create_multimodal_processor()

    text = (
        "<|im_start|>user\nThe capital of France is<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    try:
        first = _generate_native(model, processor, tokenizer, text, None, max_new_tokens)
        reset = _generate_native(model, processor, tokenizer, text, None, max_new_tokens)
        if first != reset:
            raise AssertionError(f"Fresh-generator text reset differs: {first} vs {reset}")
        results: dict[str, Any] = {"text": first, "reset": reset}
        if image is not None:
            images = og.Images.open(str(image))
            prompt = (
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                "Describe this image briefly.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            results["image"] = _generate_native(
                model, processor, tokenizer, prompt, images, max_new_tokens
            )
            del images
            if second_image is not None:
                images = og.Images.open(str(image), str(second_image))
                prompt = (
                    "<|im_start|>user\n"
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    "Describe these two images briefly.<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                )
                results["multi_image"] = _generate_native(
                    model, processor, tokenizer, prompt, images, max_new_tokens
                )
                del images
        (output / "native-results.json").write_text(
            json.dumps({"genai_version": og.__version__, **results}, indent=2),
            encoding="utf-8",
        )
    finally:
        del processor, tokenizer, model
        gc.collect()


def main() -> None:
    import mobius

    _disable_core_dumps()
    print("Mobius:", inspect.getfile(mobius), flush=True)
    if not Path(inspect.getfile(mobius)).is_relative_to(
        Path(__file__).resolve().parents[1] / "src"
    ):
        raise RuntimeError("Mobius import must belong to this example's checkout")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("Set HF_HUB_OFFLINE=1; this example never downloads assets")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
        raise RuntimeError("Select one explicit GPU UUID through CUDA_VISIBLE_DEVICES")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--native-proof", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 16:
        parser.error("--max-new-tokens must be between 1 and 16")
    if not args.prepare_only and args.native_proof is None:
        parser.error("--native-proof is required before loading the real native model")
    print(json.dumps(prepare_overlay(args.source, args.assets, args.output), indent=2))
    if not args.prepare_only:
        qualify(args.output, args.native_proof, args.image, args.max_new_tokens)


if __name__ == "__main__":
    main()
