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


def test_load_eos_token_ids_unions_generation_and_model_config(tmp_path):
    validator = _load_validator()
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 11]}),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"eos_token_id": 2}),
        encoding="utf-8",
    )

    assert validator.load_eos_token_ids(tmp_path) == {2, 11}


def test_run_token_ids_stops_on_eos_without_unused_forward(tmp_path, monkeypatch):
    validator = _load_validator()
    inference_globals = validator.run_token_ids.__globals__
    (tmp_path / "model.onnx").touch()

    class _Output:
        name = "logits"

    class _Session:
        @staticmethod
        def get_inputs():
            return []

        @staticmethod
        def get_outputs():
            return [_Output()]

    calls = []

    def _run_session(_session, _output_names, _feeds):
        calls.append(1)
        logits = np.zeros((1, 1, 16), dtype=np.float32)
        logits[0, 0, 2] = 1.0
        return [logits]

    monkeypatch.setitem(inference_globals, "_create_session", lambda *_args: _Session())
    monkeypatch.setitem(inference_globals, "_initial_states", lambda _session: {})
    monkeypatch.setitem(inference_globals, "_run_session", _run_session)

    generated, logits, profile = validator.run_token_ids(
        tmp_path,
        [1],
        max_new_tokens=4,
        device="cpu",
        eos_token_ids={2},
    )

    assert generated == [2]
    assert len(logits) == 1
    assert len(calls) == 1
    assert profile is None


def test_pinned_range_fetch_retries_and_validates_headers(monkeypatch):
    validator = _load_validator()

    class _Response:
        def __init__(self, status, headers, content=b""):
            self.status_code = status
            self.headers = headers
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Session:
        def __init__(self):
            self.responses = [
                _Response(503, {}),
                _Response(206, {"Content-Range": "bytes 1-4/10", "Content-Length": "4"}),
                _Response(
                    206, {"Content-Range": "bytes 0-3/10", "Content-Length": "4"}, b"data"
                ),
            ]
            self.calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            return self.responses.pop(0)

    reader = object.__new__(validator._PinnedSafetensors)
    reader._session = _Session()
    sleeps = []
    monkeypatch.setattr(validator.time, "sleep", sleeps.append)

    assert reader._range("model.safetensors", 0, 3) == b"data"
    assert reader._session.calls == 3
    assert sleeps == [1, 2]


def test_pinned_range_fetch_fails_explicitly_after_retries(monkeypatch):
    validator = _load_validator()

    class _Response:
        def __init__(self):
            self.status_code = 503
            self.headers = {}
            self.content = b""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Session:
        calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            return _Response()

    reader = object.__new__(validator._PinnedSafetensors)
    reader._session = _Session()
    monkeypatch.setattr(validator.time, "sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match=r"failed after 3 attempts.*status=503"):
        reader._range("model.safetensors", 0, 3)
    assert reader._session.calls == 3


def test_reduced_cache_rejects_stale_fixture_schema(tmp_path):
    validator = _load_validator()
    from safetensors.torch import save_file

    cache_path = tmp_path / "stale.safetensors"
    save_file(
        {"placeholder": torch.zeros(1)},
        cache_path,
        metadata={
            "model_id": validator.MODEL_ID,
            "revision": validator.REVISION,
            "fixture_schema": "0",
        },
    )

    with pytest.raises(ValueError, match=r"fixture_schema.*Remove the stale cache"):
        validator._build_reduced_state(cache_path)


@pytest.fixture(scope="module")
def reduced_real_state():
    validator = _load_validator()
    configured_cache = os.environ.get("MOBIUS_NEMOTRON_REDUCED_CACHE")
    cache = (
        Path(configured_cache) if configured_cache else validator.default_reduced_cache_path()
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
        eos_token_ids=validator.load_eos_token_ids(output_dir),
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
        eos_token_ids=validator.load_eos_token_ids(quantized_dir),
    )

    assert generated == [12, 13, 12, 12]
    assert all(np.isfinite(step).all() for step in logits)
    assert profile_path is not None
    assert validator.summarize_profile(profile_path).get("CUDAExecutionProvider", 0) > 0
    Path(profile_path).unlink(missing_ok=True)
