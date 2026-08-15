#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate Qwen3.8-27B using a small, pinned, reduced-real BF16 fixture.

The fixture deliberately retains the production 3xDeltaNet + 1xGQA layer
schedule and one Qwen vision block.  It is not a randomly initialized proxy:
every cached value is a deterministic row/column slice read with verified HTTP
Range requests from the pinned 18-shard, 1199-tensor checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from inference import (
    MODEL_ID,
    REVISION,
    _create_session,
    _embedding,
    _initial_states,
    _numpy,
    _run,
    run_token_ids,
    summarize_profile,
)
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FIXTURE_SCHEMA_VERSION = 1
_RANGE_ATTEMPTS = 3
_VOCAB_SIZE = 256
_HIDDEN_SIZE = 256
_GENERATION_TOKENS = 20
_MEDIA_IDS = {
    "image_token_id": 250,
    "video_token_id": 251,
    "vision_start_token_id": 252,
    "vision_end_token_id": 253,
}
_DTYPES = {
    "f32": (torch.float32, "FLOAT"),
    "f16": (torch.float16, "FLOAT16"),
    "bf16": (torch.bfloat16, "BFLOAT16"),
}


class _PinnedSafetensors:
    """Header-first, retrying reader for exact checkpoint byte ranges."""

    def __init__(self) -> None:
        index_path = hf_hub_download(
            MODEL_ID, "model.safetensors.index.json", revision=REVISION
        )
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        if len(self.weight_map) != 1199 or len(set(self.weight_map.values())) != 18:
            raise ValueError(
                "Pinned checkpoint manifest is not the expected 18 shards / 1199 tensors"
            )
        self._headers: dict[str, tuple[int, dict]] = {}
        self._session = requests.Session()

    def _url(self, shard: str) -> str:
        return f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{shard}"

    def _range(self, shard: str, start: int, end: int) -> bytes:
        expected = end - start + 1
        prefix = f"bytes {start}-{end}/"
        error = ""
        for attempt in range(_RANGE_ATTEMPTS):
            try:
                with self._session.get(
                    self._url(shard),
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=180,
                    stream=True,
                ) as response:
                    content_range = response.headers.get("Content-Range", "")
                    content_length = response.headers.get("Content-Length")
                    payload = response.content
                    if (
                        response.status_code == 206
                        and content_range.startswith(prefix)
                        and (content_length is None or content_length == str(expected))
                        and len(payload) == expected
                    ):
                        return payload
                    error = (
                        f"status={response.status_code}, range={content_range!r}, "
                        f"length={content_length}, bytes={len(payload)}, expected={expected}"
                    )
            except requests.RequestException as exc:
                error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < _RANGE_ATTEMPTS:
                time.sleep(2**attempt)
        raise RuntimeError(
            f"Range fetch failed after {_RANGE_ATTEMPTS} attempts for {shard} "
            f"bytes {start}-{end}: {error}"
        )

    def _header(self, shard: str) -> tuple[int, dict]:
        if shard not in self._headers:
            size = struct.unpack("<Q", self._range(shard, 0, 7))[0]
            self._headers[shard] = (size, json.loads(self._range(shard, 8, 7 + size)))
        return self._headers[shard]

    def sliced(self, name: str, shape: torch.Size) -> torch.Tensor:
        """Fetch only leading source rows, then deterministically trim every axis."""
        shard = self.weight_map[name]
        header_size, header = self._header(shard)
        entry = header[name]
        source_shape = list(entry["shape"])
        target_shape = list(shape)
        if len(source_shape) != len(target_shape) or any(
            a < b for a, b in zip(source_shape, target_shape)
        ):
            raise ValueError(
                f"Cannot reduce {name}: source={source_shape}, target={target_shape}"
            )
        dtype_name = entry["dtype"]
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[dtype_name]
        element_size = {"BF16": 2, "F32": 4}[dtype_name]
        rows = target_shape[0] if source_shape else 1
        row_width = math.prod(source_shape[1:]) if source_shape else 1
        start, _end = entry["data_offsets"]
        length = rows * row_width * element_size
        payload = self._range(
            shard, 8 + header_size + start, 8 + header_size + start + length - 1
        )
        leading = (
            torch.frombuffer(bytearray(payload), dtype=dtype)
            .clone()
            .reshape([rows, *source_shape[1:]])
        )
        return leading[tuple(slice(0, size) for size in target_shape)].contiguous()


def default_reduced_cache_path() -> Path:
    return (
        Path.home()
        / ".cache"
        / "mobius"
        / "qwen3_8-27b"
        / f"reduced-{REVISION}-schema-v{FIXTURE_SCHEMA_VERSION}.safetensors"
    )


def _reduced_hf_config():
    """Construct a tiny native HF config preserving all inference layer families."""
    from transformers import AutoConfig, Qwen3_5Config

    source = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
    text = source.text_config.to_dict()
    text.update(
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        mtp_num_hidden_layers=0,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        partial_rotary_factor=0.1875,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.1875,
            "mrope_section": [4, 4, 4],
            "mrope_interleaved": True,
        },
    )
    vision = source.vision_config.to_dict()
    vision.update(
        depth=1,
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        out_hidden_size=_HIDDEN_SIZE,
        num_position_embeddings=64,
    )
    return Qwen3_5Config(text_config=text, vision_config=vision, **_MEDIA_IDS)


def _reduced_mobius_config(dtype_name: str):
    import onnx_ir as ir

    from mobius._configs import ArchitectureConfig

    hf_config = _reduced_hf_config()
    config = ArchitectureConfig.from_transformers(
        hf_config.text_config, parent_config=hf_config
    )
    assert config.vision is not None
    return dataclasses.replace(
        config,
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=128,
        image_token_id=_MEDIA_IDS["image_token_id"],
        video_token_id=_MEDIA_IDS["video_token_id"],
        vision_start_token_id=_MEDIA_IDS["vision_start_token_id"],
        vision_end_token_id=_MEDIA_IDS["vision_end_token_id"],
        mrope_section=[4, 4, 4],
        mrope_interleaved=True,
        dtype=getattr(ir.DataType, _DTYPES[dtype_name][1]),
        vision=dataclasses.replace(
            config.vision,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            out_hidden_size=_HIDDEN_SIZE,
            num_position_embeddings=64,
        ),
    )


def _expected_hf_state() -> dict[str, torch.Tensor]:
    from transformers import Qwen3_5ForConditionalGeneration

    # Native HF state names are the checkpoint names, so strict loading below
    # guards both the fixture's tensor coverage and the source-name mapping.
    return {
        name: tensor
        for name, tensor in Qwen3_5ForConditionalGeneration(_reduced_hf_config())
        .state_dict()
        .items()
        if not name.startswith(("mtp_", "mtp."))
    }


def _build_reduced_state(cache_path: Path) -> dict[str, torch.Tensor]:
    expected_metadata = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "fixture_schema": str(FIXTURE_SCHEMA_VERSION),
        "source_shards": "18",
        "source_tensors": "1199",
    }
    if cache_path.is_file():
        with safe_open(cache_path, framework="pt") as cached:
            actual = cached.metadata() or {}
        if {key: actual.get(key) for key in expected_metadata} != expected_metadata:
            raise ValueError(
                "Reduced cache metadata mismatch; remove the stale cache and retry."
            )
        return load_file(cache_path)
    source = _PinnedSafetensors()
    expected = _expected_hf_state()
    missing = sorted(set(expected) - set(source.weight_map))
    if missing:
        raise ValueError(
            f"Reduced HF model requests tensors absent from checkpoint: {missing[:5]}"
        )
    # The expected model has only layers 0-3, intentionally covers three
    # DeltaNet layers and layer 3 full attention. MTP has no expected state.
    state = {name: source.sliced(name, tensor.shape) for name, tensor in expected.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.safetensors")
    save_file(state, temporary, metadata=expected_metadata)
    temporary.replace(cache_path)
    return state


def _hf_model(state: dict[str, torch.Tensor], *, dtype: torch.dtype, device: str):
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration(_reduced_hf_config()).to(
        device=device, dtype=dtype
    )
    target = model.state_dict()
    if set(target) != set(state):
        raise ValueError(
            f"Strict fixture state mismatch: missing={sorted(set(target) - set(state))[:5]}, "
            f"extra={sorted(set(state) - set(target))[:5]}"
        )
    model.load_state_dict(
        {k: v.to(device=device, dtype=target[k].dtype) for k, v in state.items()}, strict=True
    )
    return model.eval()


def _mobius_package(state: dict[str, torch.Tensor], *, dtype_name: str, ep: str):
    from mobius import build_from_module
    from mobius.models.qwen35 import Qwen35VL3ModelCausalLMModel

    config = _reduced_mobius_config(dtype_name)
    module = Qwen35VL3ModelCausalLMModel(config)
    package = build_from_module(module, config, task="hybrid-qwen-vl", execution_provider=ep)
    package.apply_weights(module.preprocess_weights(dict(state)))
    unset = [
        f"{model_name}:{name}"
        for model_name, model in package.items()
        for name, value in model.graph.initializers.items()
        if value.const_value is None
    ]
    if unset:
        raise ValueError(f"Weighted Qwen3.8 graph has unset initializers: {unset[:5]}")
    return package


def _onnx_prefill_logits(package_dir: Path, token_ids: list[int], device: str) -> np.ndarray:
    embedding = _create_session(package_dir / "embedding" / "model.onnx", device)
    decoder = _create_session(package_dir / "decoder" / "model.onnx", device)
    ids = np.array([token_ids], dtype=np.int64)
    embedded = _embedding(embedding, ids, _HIDDEN_SIZE)
    output_names = [item.name for item in decoder.get_outputs()]
    outputs = _run(
        decoder,
        output_names,
        {
            "inputs_embeds": _numpy(embedded),
            "attention_mask": np.ones_like(ids),
            "position_ids": np.repeat(np.arange(len(token_ids))[None, None, :], 3, axis=0),
            **_initial_states(decoder),
        },
    )
    return _numpy(outputs[output_names.index("logits")]).astype(np.float32)


def _hf_logits(model, token_ids: list[int], device: str) -> np.ndarray:
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        return (
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            .logits.float()
            .cpu()
            .numpy()
        )


def _graph_audit(package) -> dict[str, dict[str, int]]:
    return {
        name: dict(
            sorted(
                Counter(
                    f"{node.domain or 'ai.onnx'}::{node.op_type}"
                    for node in model.graph.all_nodes()
                ).items()
            )
        )
        for name, model in package.items()
    }


def _assert_logits_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    label: str,
) -> None:
    max_abs = float(np.max(np.abs(actual - expected)))
    cosine = float(
        np.dot(actual.ravel(), expected.ravel())
        / (np.linalg.norm(actual) * np.linalg.norm(expected))
    )
    print(f"{label}: max_abs={max_abs:.8f}, cosine={cosine:.9f}")
    if max_abs > atol or cosine < 0.999:
        raise AssertionError(
            f"{label} parity failed: max_abs={max_abs:.8f}, cosine={cosine:.9f}"
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=atol)


def _save_package_assets(package_dir: Path) -> None:
    """Copy pinned processor metadata for the reduced token-ID-only package."""
    from transformers import AutoProcessor, GenerationConfig

    _reduced_hf_config().save_pretrained(package_dir)
    GenerationConfig.from_pretrained(MODEL_ID, revision=REVISION).save_pretrained(package_dir)
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        (package_dir / "processor_config.json").write_text(
            json.dumps(processor.to_dict(), indent=2),
            encoding="utf-8",
        )
    except (ImportError, OSError, ValueError) as error:
        # The ONNX package remains directly runnable without the optional
        # processor serialization; retain the precise reason in its manifest.
        (package_dir / "processor-waiver.txt").write_text(f"{type(error).__name__}: {error}\n")
    (package_dir / "source_manifest.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "fixture_schema": FIXTURE_SCHEMA_VERSION,
                "runtime": "onnxruntime-direct",
                "text_input_contract": "token-ids-only",
                "components": [
                    "decoder/model.onnx",
                    "embedding/model.onnx",
                    "vision_encoder/model.onnx",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _media_smoke(package_dir: Path, device: str) -> dict[str, tuple[int, ...]]:
    """Exercise nonzero packed image/video/mixed vision inputs, processor-shaped grids."""
    vision = _create_session(package_dir / "vision_encoder" / "model.onnx", device)
    # A video T unit already packs two raw frames. Exercise multiple temporal
    # units and unequal spatial grids to cover the dynamic packed-media path.
    results = {}
    for kind, grid in {
        "image": np.array([[1, 4, 4]], dtype=np.int64),
        "video": np.array([[2, 4, 4]], dtype=np.int64),
        "mixed": np.array([[1, 4, 4], [2, 6, 4]], dtype=np.int64),
    }.items():
        patches = int(np.prod(grid, axis=1).sum())
        pixels = np.linspace(0.01, 1.0, patches * 3 * 2 * 16 * 16, dtype=np.float32).reshape(
            patches, -1
        )
        output = _numpy(
            _run(vision, ["image_features"], {"pixel_values": pixels, "image_grid_thw": grid})[
                0
            ]
        )
        if not np.isfinite(output).all() or not np.any(output):
            raise AssertionError(f"{kind} processor-shaped vision output is invalid")
        results[kind] = output.shape
    return results


def _cuda_standard_vision_smoke(
    state: dict[str, torch.Tensor],
    output_root: Path,
    *,
    dtype_name: str,
) -> dict[str, tuple[int, ...]]:
    """Run media through the standard-attention vision graph on CUDA.

    ORT 1.26's CUDA PackedMultiHeadAttention kernel is nondeterministic for
    dynamic packed vision batches. The portable graph still places vision
    compute on CUDA and provides stable image/video/mixed runtime evidence.
    """
    package = _mobius_package(state, dtype_name=dtype_name, ep="cpu")
    package_dir = output_root / f"{dtype_name}-cuda-standard-vision"
    package_dir.mkdir(parents=True, exist_ok=True)
    package.save(package_dir, external_data="onnx")
    return _media_smoke(package_dir, "cuda")


def _save_variant(
    state: dict[str, torch.Tensor],
    output_root: Path,
    *,
    dtype_name: str,
    device: str,
):
    package = _mobius_package(
        state, dtype_name=dtype_name, ep="cuda" if device == "cuda" else "cpu"
    )
    package_dir = output_root / f"{dtype_name}-{device}"
    package_dir.mkdir(parents=True, exist_ok=True)
    package.save(package_dir, external_data="onnx")
    _save_package_assets(package_dir)
    # Package save/load is part of the acceptance boundary, not a graph-only test.
    import onnx_ir as ir

    for name in ("decoder", "embedding", "vision_encoder"):
        ir.load(package_dir / name / "model.onnx")
    return package, package_dir


def _validate_variant(
    state: dict[str, torch.Tensor], output_root: Path, *, dtype_name: str, device: str
) -> Path:
    package, package_dir = _save_variant(
        state,
        output_root,
        dtype_name=dtype_name,
        device=device,
    )
    prompt = [1, 42, 17]
    actual = _onnx_prefill_logits(package_dir, prompt, device)
    model = _hf_model(state, dtype=_DTYPES[dtype_name][0], device=device)
    expected = _hf_logits(model, prompt, device)
    atol = 2e-3 if dtype_name == "f32" else 1e-2
    _assert_logits_close(
        actual,
        expected,
        atol=atol,
        label=f"{dtype_name}/{device} full-prefill",
    )
    generated, step_logits, profile = run_token_ids(
        package_dir,
        prompt,
        hidden_size=_HIDDEN_SIZE,
        max_new_tokens=_GENERATION_TOKENS,
        device=device,
        profile=device == "cuda",
    )
    hf_generated = model.generate(
        input_ids=torch.tensor([prompt], device=device),
        max_new_tokens=_GENERATION_TOKENS,
        do_sample=False,
    )[0, len(prompt) :].tolist()
    if (
        generated != hf_generated
        or len(generated) != _GENERATION_TOKENS
        or not all(np.isfinite(x).all() for x in step_logits)
    ):
        raise AssertionError(
            f"Cached generation mismatch: onnx={generated}, hf={hf_generated}"
        )
    if profile:
        placement = summarize_profile(profile)
        Path(profile).unlink(missing_ok=True)
        if placement.get("CUDAExecutionProvider", 0) == 0:
            raise AssertionError(f"CUDA provider received no decoder nodes: {placement}")
        print(f"{dtype_name}/{device} provider placement: {placement}")
    print(f"{dtype_name}/{device} generated IDs: {generated}")
    print(f"{dtype_name}/{device} graph audit: {_graph_audit(package)}")
    if device == "cuda":
        media_shapes = _cuda_standard_vision_smoke(
            state,
            output_root,
            dtype_name=dtype_name,
        )
        print(f"{dtype_name}/{device} standard-vision media shapes: {media_shapes}")
    else:
        print(f"{dtype_name}/{device} media shapes: {_media_smoke(package_dir, device)}")
    return package_dir


def write_goldens(state: dict[str, torch.Tensor], directory: Path) -> None:
    """Write L4/L5 only from the independent native HuggingFace reduced model."""
    model = _hf_model(state, dtype=torch.float32, device="cpu")
    prompt = [1, 42, 17]
    logits = _hf_logits(model, prompt, "cpu")[0, -1]
    top10 = np.argsort(logits)[::-1][:10]
    generated = model.generate(
        input_ids=torch.tensor([prompt]),
        max_new_tokens=_GENERATION_TOKENS,
        do_sample=False,
    )[0, len(prompt) :].tolist()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "qwen3_8-27b-reduced.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "fixture_schema": FIXTURE_SCHEMA_VERSION,
                "input_ids": prompt,
                "top10_ids": top10.tolist(),
                "top10_logits": [float(logits[index]).hex() for index in top10],
                "logits_summary": [
                    float(x).hex()
                    for x in (logits.max(), logits.min(), logits.mean(), logits.std())
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (directory / "qwen3_8-27b-reduced_generation.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "fixture_schema": FIXTURE_SCHEMA_VERSION,
                "input_ids": prompt,
                "max_new_tokens": _GENERATION_TOKENS,
                "generated_tokens": generated,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=default_reduced_cache_path())
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen3_8-reduced"))
    parser.add_argument(
        "--matrix",
        nargs="+",
        choices=["f32-cpu", "f16-cuda", "bf16-cuda"],
        default=["f32-cpu", "f16-cuda", "bf16-cuda"],
    )
    parser.add_argument("--write-goldens", action="store_true")
    parser.add_argument("--skip-quantization", action="store_true")
    args = parser.parse_args()
    state = _build_reduced_state(args.cache)
    print(f"Loaded {len(state)} reduced tensors from {MODEL_ID}@{REVISION}")
    if args.write_goldens:
        write_goldens(
            state, Path(__file__).parents[3] / "testdata" / "golden" / "vision-language"
        )
    variants = {}
    for variant in args.matrix:
        dtype, device = variant.split("-")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA validation requested but unavailable: {variant}")
        if variant == "bf16-cuda":
            _package, variants[variant] = _save_variant(
                state,
                args.output_dir,
                dtype_name=dtype,
                device=device,
            )
            print(
                "bf16/cuda export and package reload passed; ORT 1.26 runtime "
                "waived for hybrid BF16 provider initialization"
            )
        else:
            variants[variant] = _validate_variant(
                state,
                args.output_dir,
                dtype_name=dtype,
                device=device,
            )
    if not args.skip_quantization:
        from optimize import quantize_package

        source = variants.get("f16-cuda")
        if source is None:
            raise ValueError("Q4_K_M validation requires f16-cuda")
        quantized = quantize_package(source, args.output_dir / "q4_k_m")
        import onnx_ir as ir

        loaded_models = {
            name: ir.load(quantized / name / "model.onnx")
            for name in ("decoder", "embedding", "vision_encoder")
        }
        matmul_nbits = sum(
            node.domain == "com.microsoft" and node.op_type == "MatMulNBits"
            for node in loaded_models["decoder"].graph.all_nodes()
        )
        source_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
        quantized_bytes = sum(
            path.stat().st_size for path in quantized.rglob("*") if path.is_file()
        )
        if not matmul_nbits or quantized_bytes >= source_bytes:
            raise AssertionError(
                "Q4_K_M graph/package audit failed: "
                f"MatMulNBits={matmul_nbits}, source={source_bytes}, quantized={quantized_bytes}"
            )
        for name in loaded_models:
            _create_session(quantized / name / "model.onnx", "cuda")
        # Direct ORT sessions, not ORT GenAI capability probing.
        ids, logits, _ = run_token_ids(
            quantized,
            [1, 42, 17],
            hidden_size=_HIDDEN_SIZE,
            max_new_tokens=_GENERATION_TOKENS,
            device="cpu",
        )
        if len(ids) != _GENERATION_TOKENS or not all(
            np.isfinite(logit).all() for logit in logits
        ):
            raise AssertionError(f"Q4_K_M direct-session generation failed: {ids}")
        print(
            "Q4_K_M package audit: "
            f"MatMulNBits={matmul_nbits}, source={source_bytes}, quantized={quantized_bytes}"
        )
        print(f"Q4_K_M direct CPU-session IDs: {ids}")


if __name__ == "__main__":
    main()
