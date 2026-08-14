# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned reduced-real-weight L4/L5 tests for Nemotron 3.5 Lightning."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).parents[1]
_EXAMPLE_DIR = _ROOT / "examples" / "olive" / "nemotron-3_5-lightning-30b"
_VALIDATOR_PATH = _EXAMPLE_DIR / "validate_reduced_checkpoint.py"
_L4_PATH = (
    _ROOT / "testdata" / "golden" / "causal-lm" / "nemotron-3_5-lightning-30b-reduced.json"
)
_L5_PATH = (
    _ROOT
    / "testdata"
    / "golden"
    / "causal-lm"
    / "nemotron-3_5-lightning-30b-reduced_generation.json"
)


def _load_validator():
    sys.path.insert(0, str(_EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "nemotron_reduced_validator", _VALIDATOR_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="module")
def reduced_real_state(tmp_path_factory):
    validator = _load_validator()
    configured_cache = os.environ.get("MOBIUS_NEMOTRON_REDUCED_CACHE")
    cache = (
        Path(configured_cache)
        if configured_cache
        else tmp_path_factory.mktemp("nemotron-real") / "reduced.safetensors"
    )
    state = validator._build_reduced_state(cache)
    return validator, state


@pytest.fixture(scope="module")
def reduced_real_outputs(reduced_real_state, tmp_path_factory):
    validator, state = reduced_real_state
    package = validator._mobius_package(state, dtype_name="f32", ep="cpu")
    output_dir = tmp_path_factory.mktemp("nemotron-onnx")
    package.save(output_dir, external_data="onnx")
    session = validator._create_session(output_dir / "model.onnx", "cpu", False)
    prompt_ids = [1, 42, 17]
    onnx_logits = validator._full_prefill(session, prompt_ids)
    onnx_tokens, _step_logits, _profile = validator.run_token_ids(
        output_dir,
        prompt_ids,
        max_new_tokens=4,
        device="cpu",
    )

    hf_model = validator._hf_model(state, dtype=torch.float32, device="cpu")
    hf_logits = validator._hf_full_prefill(hf_model, prompt_ids, "cpu")
    hf_tokens, _hf_step_logits = validator._hf_generate(hf_model, prompt_ids, "cpu", 4)
    return onnx_logits, onnx_tokens, hf_logits, hf_tokens


def _require_cuda() -> None:
    if os.environ.get("MOBIUS_TEST_DEVICE") != "cuda":
        pytest.skip("Set MOBIUS_TEST_DEVICE=cuda to run reduced-real CUDA coverage")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is unavailable")
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ONNX Runtime CUDAExecutionProvider is unavailable")


@pytest.fixture(scope="module")
def reduced_real_fp16_cuda(reduced_real_state, tmp_path_factory):
    validator, state = reduced_real_state
    _require_cuda()
    output_root = tmp_path_factory.mktemp("nemotron-fp16-cuda")
    package_dir = validator._validate_variant(
        state,
        output_root,
        dtype_name="f16",
        device="cuda",
    )
    return validator, state, package_dir


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.golden
@pytest.mark.parametrize("model_type", ["nemotron_h"])
def test_nemotron_h_3_5_reduced_real_l4(reduced_real_outputs, model_type):
    del model_type
    onnx_logits, _onnx_tokens, hf_logits, _hf_tokens = reduced_real_outputs
    golden = json.loads(_L4_PATH.read_text(encoding="utf-8"))

    np.testing.assert_allclose(onnx_logits, hf_logits, rtol=1e-3, atol=2e-3)
    last_logits = hf_logits[0, -1]
    top10 = np.argsort(last_logits)[::-1][:10]
    summary = [last_logits.max(), last_logits.min(), last_logits.mean(), last_logits.std()]

    assert top10.tolist() == golden["top10_ids"]
    np.testing.assert_allclose(
        last_logits[top10],
        [float.fromhex(value) for value in golden["top10_logits"]],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        summary,
        [float.fromhex(value) for value in golden["logits_summary"]],
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.generation
@pytest.mark.parametrize("model_type", ["nemotron_h"])
def test_nemotron_h_3_5_reduced_real_l5(reduced_real_outputs, model_type):
    del model_type
    _onnx_logits, onnx_tokens, _hf_logits, hf_tokens = reduced_real_outputs
    golden = json.loads(_L5_PATH.read_text(encoding="utf-8"))

    assert len(onnx_tokens) == golden["max_new_tokens"]
    assert len(hf_tokens) == golden["max_new_tokens"]
    assert hf_tokens == golden["generated_tokens"]
    assert onnx_tokens == golden["generated_tokens"]


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.golden
@pytest.mark.parametrize("model_type", ["nemotron_h"])
def test_nemotron_h_3_5_reduced_real_fp16_cuda(reduced_real_fp16_cuda, model_type):
    del model_type
    _validator, _state, package_dir = reduced_real_fp16_cuda

    assert (package_dir / "model.onnx").is_file()
    assert (package_dir / "model.onnx.data").is_file()


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.golden
@pytest.mark.parametrize("model_type", ["nemotron_h"])
def test_nemotron_h_3_5_bf16_rejection_evidence(
    reduced_real_state,
    tmp_path,
    model_type,
):
    del model_type
    validator, state = reduced_real_state
    _require_cuda()

    max_abs = validator._measure_bf16_rejection(state, tmp_path)

    assert max_abs > 1e-2


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.generation
@pytest.mark.quantization
@pytest.mark.parametrize("model_type", ["nemotron_h"])
def test_nemotron_h_3_5_olive_q4_final_package(
    reduced_real_fp16_cuda,
    tmp_path,
    model_type,
):
    del model_type
    import onnx_ir as ir

    validator, _state, source_dir = reduced_real_fp16_cuda
    quantized_dir = validator.quantize_package(source_dir, tmp_path / "q4_k_m-cuda")

    assert (quantized_dir / "model.onnx").is_file()
    assert (quantized_dir / "model.onnx.data").is_file()
    assert (quantized_dir / "config.json").is_file()
    assert sum(
        path.stat().st_size for path in quantized_dir.iterdir() if path.is_file()
    ) < sum(path.stat().st_size for path in source_dir.iterdir() if path.is_file())

    quantized_model = ir.load(quantized_dir / "model.onnx")
    assert (
        sum(
            node.domain == "com.microsoft" and node.op_type == "MatMulNBits"
            for node in quantized_model.graph.all_nodes()
        )
        == 15
    )

    generated, logits, profile_path = validator.run_token_ids(
        quantized_dir,
        [1, 42, 17],
        max_new_tokens=4,
        device="cuda",
        profile=True,
    )

    assert generated == [12, 13, 12, 12]
    assert all(np.isfinite(step).all() for step in logits)
    assert profile_path is not None
    assert validator.summarize_profile(profile_path).get("CUDAExecutionProvider", 0) > 0
    Path(profile_path).unlink(missing_ok=True)
