#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate a reduced model assembled from byte ranges of the pinned checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from inference import (
    MODEL_ID,
    REVISION,
    _create_session,
    _initial_states,
    _as_numpy,
    _run_session,
    _token_feeds,
    _update_states,
    run_token_ids,
    summarize_profile,
)
from optimize import quantize_package

_VOCAB_SIZE = 256
_NUM_EXPERTS = 4
_LAYER_REMAP = {0: 0, 1: 1, 5: 2}
_DTYPES = {
    "f32": (torch.float32, "FLOAT"),
    "f16": (torch.float16, "FLOAT16"),
    "bf16": (torch.bfloat16, "BFLOAT16"),
}


class _PinnedSafetensors:
    """Read selected tensors via HTTP Range without downloading 65.8 GB."""

    def __init__(self) -> None:
        import requests

        index_path = hf_hub_download(
            MODEL_ID,
            "model.safetensors.index.json",
            revision=REVISION,
        )
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}
        self._session = requests.Session()

    def _url(self, shard: str) -> str:
        return f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{shard}"

    def _range(self, shard: str, start: int, end: int) -> bytes:
        response = self._session.get(
            self._url(shard),
            headers={"Range": f"bytes={start}-{end}"},
            timeout=180,
        )
        expected = end - start + 1
        if response.status_code != 206 or len(response.content) != expected:
            raise RuntimeError(
                f"Range fetch failed for {shard} bytes {start}-{end}: "
                f"status={response.status_code}, bytes={len(response.content)}"
            )
        return response.content

    def _header(self, shard: str) -> tuple[int, dict]:
        if shard not in self._headers:
            header_length = struct.unpack("<Q", self._range(shard, 0, 7))[0]
            header = json.loads(self._range(shard, 8, 7 + header_length))
            self._headers[shard] = (header_length, header)
        return self._headers[shard]

    def tensor(self, name: str, *, rows: int | None = None) -> torch.Tensor:
        shard = self.weight_map[name]
        header_length, header = self._header(shard)
        entry = header[name]
        shape = list(entry["shape"])
        dtype_name = entry["dtype"]
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[dtype_name]
        element_size = {"BF16": 2, "F32": 4}[dtype_name]
        start, end = entry["data_offsets"]
        if rows is not None:
            if not shape or rows > shape[0]:
                raise ValueError(f"Invalid row slice {rows} for {name}: {shape}")
            row_elements = math.prod(shape[1:])
            end = start + rows * row_elements * element_size
            shape[0] = rows

        data_start = 8 + header_length
        payload = self._range(shard, data_start + start, data_start + end - 1)
        tensor = torch.frombuffer(bytearray(payload), dtype=dtype).clone()
        return tensor.reshape(shape)


def _source_to_target(name: str) -> str:
    if name == "backbone.embeddings.weight":
        return "model.embeddings.weight"
    if name == "backbone.norm_f.weight":
        return "model.norm_f.weight"
    if not name.startswith("backbone.layers."):
        return name
    parts = name.split(".")
    source_layer = int(parts[2])
    parts[2] = str(_LAYER_REMAP[source_layer])
    parts[0] = "model"
    return ".".join(parts)


def _build_reduced_state(cache_path: Path) -> dict[str, torch.Tensor]:
    if cache_path.is_file():
        with safe_open(cache_path, framework="pt") as cached:
            metadata = cached.metadata() or {}
        if metadata.get("revision") != REVISION:
            raise ValueError(
                f"Reduced cache revision mismatch: {metadata.get('revision')} != {REVISION}"
            )
        return load_file(cache_path)

    source = _PinnedSafetensors()
    state: dict[str, torch.Tensor] = {
        "model.embeddings.weight": source.tensor(
            "backbone.embeddings.weight",
            rows=_VOCAB_SIZE,
        ),
        "model.norm_f.weight": source.tensor("backbone.norm_f.weight"),
        "lm_head.weight": source.tensor("lm_head.weight", rows=_VOCAB_SIZE),
    }

    for source_layer in (0, 5):
        prefix = f"backbone.layers.{source_layer}."
        for name in sorted(source.weight_map):
            if name.startswith(prefix):
                state[_source_to_target(name)] = source.tensor(name)

    moe_prefix = "backbone.layers.1.mixer"
    state["model.layers.1.norm.weight"] = source.tensor("backbone.layers.1.norm.weight")
    state["model.layers.1.mixer.gate.weight"] = source.tensor(
        f"{moe_prefix}.gate.weight",
        rows=_NUM_EXPERTS,
    )
    state["model.layers.1.mixer.gate.e_score_correction_bias"] = source.tensor(
        f"{moe_prefix}.gate.e_score_correction_bias",
        rows=_NUM_EXPERTS,
    )
    for projection in ("up_proj", "down_proj"):
        state[f"model.layers.1.mixer.experts.{projection}"] = torch.stack(
            [
                source.tensor(f"{moe_prefix}.experts.{expert}.{projection}.weight")
                for expert in range(_NUM_EXPERTS)
            ]
        )
        state[f"model.layers.1.mixer.shared_experts.{projection}.weight"] = source.tensor(
            f"{moe_prefix}.shared_experts.{projection}.weight"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in state.items()},
        cache_path,
        metadata={"model_id": MODEL_ID, "revision": REVISION},
    )
    return state


def _hf_config():
    from transformers import NemotronHConfig

    return NemotronHConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=2688,
        layers_block_type=["linear_attention", "moe", "full_attention"],
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=1856,
        mamba_num_heads=64,
        mamba_head_dim=64,
        ssm_state_size=128,
        n_groups=8,
        conv_kernel=4,
        expand=2,
        use_mamba_kernels=False,
        moe_intermediate_size=1856,
        moe_shared_expert_intermediate_size=3712,
        n_routed_experts=_NUM_EXPERTS,
        num_experts_per_tok=2,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        layer_norm_epsilon=1e-5,
        rescale_prenorm_residual=False,
        max_position_embeddings=262144,
    )


def _hf_model(
    state: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    device: str,
):
    from transformers import NemotronHForCausalLM

    model = NemotronHForCausalLM(_hf_config()).to(device=device, dtype=dtype)
    target = model.state_dict()
    if set(target) != set(state):
        missing = sorted(set(target) - set(state))
        extra = sorted(set(state) - set(target))
        raise ValueError(f"Reduced state mismatch; missing={missing}, extra={extra}")
    converted = {
        name: tensor.to(device=device, dtype=target[name].dtype)
        for name, tensor in state.items()
    }
    model.load_state_dict(converted, strict=True)

    # The production loader keeps this selection-only bias in fp32.
    gate = model.model.layers[1].mixer.gate
    gate.e_score_correction_bias = state[
        "model.layers.1.mixer.gate.e_score_correction_bias"
    ].to(device=device, dtype=torch.float32)
    return model.eval()


def _mobius_package(
    state: dict[str, torch.Tensor],
    *,
    dtype_name: str,
    ep: str,
):
    import onnx_ir as ir

    from mobius import build_from_module
    from mobius._configs import NemotronHConfig
    from mobius._flags import override_flags
    from mobius.models.nemotron_h import NemotronHCausalLMModel

    config = NemotronHConfig.from_transformers(_hf_config())
    config.dtype = getattr(ir.DataType, _DTYPES[dtype_name][1])
    module = NemotronHCausalLMModel(config)
    with override_flags(ort_cuda_grouped_rmsnorm_workaround=ep == "cuda"):
        package = build_from_module(
            module,
            config,
            task="hybrid-text-generation",
            execution_provider=ep,
            trace_optimization=True,
        )
    package.apply_weights(module.preprocess_weights(dict(state)))
    unset = [
        name
        for name, value in package["model"].graph.initializers.items()
        if value.const_value is None
    ]
    if unset:
        raise ValueError(f"Weighted graph still has {len(unset)} unset parameters: {unset[:5]}")
    return package


def _full_prefill(session, token_ids: list[int]) -> np.ndarray:
    states = _initial_states(session)
    output_names = [output.name for output in session.get_outputs()]
    feeds = _token_feeds(
        session,
        np.array([token_ids], dtype=np.int64),
        total_length=len(token_ids),
        position_ids=np.arange(len(token_ids), dtype=np.int64)[None, :],
        states=states,
    )
    outputs = _run_session(session, output_names, feeds)
    return _as_numpy(outputs[output_names.index("logits")]).astype(np.float32)


def _hf_full_prefill(model, token_ids: list[int], device: str) -> np.ndarray:
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(len(token_ids), device=device)[None, :],
            use_cache=False,
        ).logits
    return logits.float().cpu().numpy()


def _hf_generate(
    model,
    token_ids: list[int],
    device: str,
    max_new_tokens: int,
) -> tuple[list[int], list[np.ndarray]]:
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    past_length = 0
    outputs = None
    with torch.no_grad():
        for token_id in token_ids:
            ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
            outputs = model(
                input_ids=ids,
                attention_mask=torch.ones((1, past_length + 1), dtype=torch.long, device=device),
                position_ids=torch.tensor([[past_length]], dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            past_length += 1

        assert outputs is not None
        generated: list[int] = []
        logits_by_step: list[np.ndarray] = []
        for _ in range(max_new_tokens):
            logits_by_step.append(outputs.logits[0, -1].float().cpu().numpy())
            token_id = int(outputs.logits[0, -1].argmax())
            generated.append(token_id)
            ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
            outputs = model(
                input_ids=ids,
                attention_mask=torch.ones((1, past_length + 1), dtype=torch.long, device=device),
                position_ids=torch.tensor([[past_length]], dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            past_length += 1
    return generated, logits_by_step


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
    print(f"{label}: max_abs={max_abs:.6g}, cosine={cosine:.9f}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=atol)


def _graph_audit(model) -> dict[str, int]:
    counts = Counter(
        f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.all_nodes()
    )
    return dict(sorted(counts.items()))


def _validate_variant(
    state: dict[str, torch.Tensor],
    output_root: Path,
    *,
    dtype_name: str,
    device: str,
) -> Path:
    torch_dtype = _DTYPES[dtype_name][0]
    ep = "cuda" if device == "cuda" else "cpu"
    package = _mobius_package(state, dtype_name=dtype_name, ep=ep)
    variant_dir = output_root / f"{dtype_name}-{ep}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    package.save(variant_dir, external_data="onnx")
    _hf_config().save_pretrained(variant_dir)

    profile = device == "cuda"
    session = _create_session(variant_dir / "model.onnx", device, profile)
    prompt_ids = [1, 42, 17]
    actual = _full_prefill(session, prompt_ids)
    hf_model = _hf_model(state, dtype=torch_dtype, device=device)
    expected = _hf_full_prefill(hf_model, prompt_ids, device)
    atol = 2e-3 if dtype_name == "f32" else 1e-2
    _assert_logits_close(actual, expected, atol=atol, label=f"{dtype_name}/{ep} prefill")

    generated, logits, profile_path = run_token_ids(
        variant_dir,
        prompt_ids,
        max_new_tokens=4,
        device=device,
        profile=profile,
    )
    if len(generated) != 4 or any(not np.isfinite(step).all() for step in logits):
        raise AssertionError(f"Invalid generation for {dtype_name}/{ep}: {generated}")
    hf_generated, hf_step_logits = _hf_generate(
        hf_model,
        prompt_ids,
        device,
        max_new_tokens=4,
    )
    if generated != hf_generated:
        raise AssertionError(
            f"{dtype_name}/{ep} generation mismatch: ONNX={generated}, HF={hf_generated}"
        )
    for index, (actual_step, expected_step) in enumerate(zip(logits, hf_step_logits)):
        _assert_logits_close(
            actual_step,
            expected_step,
            atol=atol,
            label=f"{dtype_name}/{ep} cached step {index}",
        )
    print(f"{dtype_name}/{ep} generated IDs: {generated}")
    print(f"{dtype_name}/{ep} weighted graph ops: {_graph_audit(package['model'])}")

    if profile_path is not None:
        placement = summarize_profile(profile_path)
        print(f"{dtype_name}/{ep} provider placement: {placement}")
        if placement.get("CUDAExecutionProvider", 0) == 0:
            raise AssertionError(f"No CUDA nodes found in profile: {placement}")
    del hf_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return variant_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default="cache/nemotron-3_5-reduced-real.safetensors",
    )
    parser.add_argument("--output-dir", default="output/reduced-validation")
    parser.add_argument(
        "--matrix",
        nargs="+",
        choices=["f32-cpu", "f16-cuda", "bf16-cuda"],
        default=["f32-cpu", "f16-cuda"],
    )
    parser.add_argument("--skip-quantization", action="store_true")
    args = parser.parse_args()

    state = _build_reduced_state(Path(args.cache))
    print(f"Loaded {len(state)} reduced real-weight tensors from revision {REVISION}")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Path] = {}
    for variant in args.matrix:
        dtype_name, device = variant.split("-")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA validation requested but unavailable: {variant}")
        variants[variant] = _validate_variant(
            state,
            output_root,
            dtype_name=dtype_name,
            device=device,
        )

    if not args.skip_quantization:
        source = variants.get("f16-cuda")
        if source is None:
            raise ValueError("Olive validation requires f16-cuda in --matrix")
        quantized = quantize_package(source, output_root / "q4_k_m-cuda")
        generated, logits, _profile = run_token_ids(
            quantized,
            [1, 42, 17],
            max_new_tokens=4,
            device="cuda",
            profile=False,
        )
        if len(generated) != 4 or any(not np.isfinite(step).all() for step in logits):
            raise AssertionError(f"Quantized generation failed: {generated}")
        print(f"Olive Q4_K_M package loaded and generated IDs: {generated}")


if __name__ == "__main__":
    main()
