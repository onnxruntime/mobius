# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned reduced-real Qwen3.8 L4/L5 and Olive recipe coverage."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parents[1]
_EXAMPLE = _ROOT / "examples" / "olive" / "qwen3_8-27b"
_L4 = _ROOT / "testdata" / "golden" / "vision-language" / "qwen3_8-27b-reduced.json"
_L5 = _ROOT / "testdata" / "golden" / "vision-language" / "qwen3_8-27b-reduced_generation.json"


def _load(name: str):
    sys.path.insert(0, str(_EXAMPLE))
    try:
        path = _EXAMPLE / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"qwen38_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_reduced_config_preserves_hybrid_layers_and_remaps_media_ids():
    validator = _load("validate_reduced_checkpoint")
    config = validator._reduced_hf_config()
    assert config.text_config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert config.text_config.mtp_num_hidden_layers == 0
    assert config.vision_config.depth == 1
    assert config.image_token_id < config.text_config.vocab_size
    assert config.video_token_id < config.text_config.vocab_size


def test_range_reader_retries_and_requires_exact_content_range(monkeypatch):
    validator = _load("validate_reduced_checkpoint")

    class Response:
        def __init__(self, status, headers, content=b""):
            self.status_code, self.headers, self.content = status, headers, content

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Session:
        def __init__(self):
            self.responses = [
                Response(503, {}),
                Response(
                    206, {"Content-Range": "bytes 2-5/10", "Content-Length": "4"}, b"fail"
                ),
                Response(
                    206, {"Content-Range": "bytes 0-3/10", "Content-Length": "4"}, b"pass"
                ),
            ]

        def get(self, *_args, **_kwargs):
            return self.responses.pop(0)

    reader = object.__new__(validator._PinnedSafetensors)
    reader._session = Session()
    monkeypatch.setattr(validator.time, "sleep", lambda _seconds: None)
    assert reader._range("shard", 0, 3) == b"pass"


def test_olive_recipe_is_q4_k_m_cpu_weight_only(tmp_path):
    optimize = _load("optimize")
    recipe = optimize.olive_config(tmp_path / "decoder.onnx", tmp_path / "out")
    config = recipe["passes"]["q4_k_m"]
    assert config["type"] == "OnnxKQuantQuantization"
    assert config["bits"] == 4
    assert recipe["clean_cache"] is True
    assert Path(recipe["cache_dir"]).name == ".olive-cache"
    assert recipe["engine"]["target"]["accelerators"][0]["execution_providers"] == [
        "CPUExecutionProvider"
    ]


def _require_real_fixture():
    if os.environ.get("MOBIUS_QWEN38_REDUCED_REAL") != "1":
        pytest.skip("Set MOBIUS_QWEN38_REDUCED_REAL=1 to enable pinned range fixture")


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.golden
def test_qwen38_reduced_real_l4(tmp_path):
    _require_real_fixture()
    validator = _load("validate_reduced_checkpoint")
    state = validator._build_reduced_state(validator.default_reduced_cache_path())
    package_dir = tmp_path / "qwen38-test-output"
    package_dir.mkdir(exist_ok=True)
    source = validator._validate_variant(state, package_dir, dtype_name="f32", device="cpu")
    golden = json.loads(_L4.read_text(encoding="utf-8"))
    logits = validator._onnx_prefill_logits(source, golden["input_ids"], "cpu")[0, -1]
    actual = np.argsort(logits)[::-1][:10].tolist()
    assert actual == golden["top10_ids"]
    np.testing.assert_allclose(
        logits[actual], [float.fromhex(value) for value in golden["top10_logits"]], atol=2e-3
    )


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.generation
def test_qwen38_reduced_real_l5(tmp_path):
    _require_real_fixture()
    validator = _load("validate_reduced_checkpoint")
    golden = json.loads(_L5.read_text(encoding="utf-8"))
    state = validator._build_reduced_state(validator.default_reduced_cache_path())
    package_dir = validator._validate_variant(
        state,
        tmp_path / "qwen38-test-output",
        dtype_name="f32",
        device="cpu",
    )
    generated, _logits, _profile = validator.run_token_ids(
        package_dir,
        golden["input_ids"],
        hidden_size=256,
        max_new_tokens=golden["max_new_tokens"],
        device="cpu",
    )
    assert generated == golden["generated_tokens"]


@pytest.mark.integration
@pytest.mark.integration_slow
def test_qwen38_reduced_real_bf16_export_and_reload(tmp_path):
    _require_real_fixture()
    import onnx_ir as ir

    validator = _load("validate_reduced_checkpoint")
    state = validator._build_reduced_state(validator.default_reduced_cache_path())
    package, package_dir = validator._save_variant(
        state,
        tmp_path / "qwen38-test-output",
        dtype_name="bf16",
        device="cuda",
    )
    for model in package.values():
        assert not [
            name
            for name, initializer in model.graph.initializers.items()
            if initializer.const_value is None
        ]
        assert any(
            initializer.dtype == ir.DataType.BFLOAT16
            for initializer in model.graph.initializers.values()
        )
    vision = ir.load(package_dir / "vision_encoder" / "model.onnx")
    assert vision.graph.inputs[0].name == "pixel_values"
    assert vision.graph.inputs[0].dtype == ir.DataType.FLOAT


@pytest.mark.integration
@pytest.mark.integration_slow
def test_qwen38_reduced_real_f16_cuda(tmp_path):
    _require_real_fixture()
    if os.environ.get("MOBIUS_TEST_DEVICE") != "cuda":
        pytest.skip("Set MOBIUS_TEST_DEVICE=cuda for CUDA parity")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    validator = _load("validate_reduced_checkpoint")
    state = validator._build_reduced_state(validator.default_reduced_cache_path())
    validator._validate_variant(
        state,
        tmp_path / "qwen38-test-output",
        dtype_name="f16",
        device="cuda",
    )


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.quantization
def test_qwen38_reduced_olive_q4_package(tmp_path):
    _require_real_fixture()
    if os.environ.get("MOBIUS_TEST_DEVICE") != "cuda":
        pytest.skip("Set MOBIUS_TEST_DEVICE=cuda for Q4_K_M validation")
    validator = _load("validate_reduced_checkpoint")
    optimize = _load("optimize")
    state = validator._build_reduced_state(validator.default_reduced_cache_path())
    output_root = tmp_path / "qwen38-test-output"
    source = validator._validate_variant(
        state,
        output_root,
        dtype_name="f16",
        device="cuda",
    )
    result = optimize.quantize_package(source, output_root / "q4_k_m")
    import onnx_ir as ir

    decoder = ir.load(result / "decoder" / "model.onnx")
    assert any(
        node.domain == "com.microsoft" and node.op_type == "MatMulNBits"
        for node in decoder.graph.all_nodes()
    )
    assert sum(path.stat().st_size for path in result.rglob("*") if path.is_file()) < sum(
        path.stat().st_size for path in source.rglob("*") if path.is_file()
    )
    for name in ("decoder", "embedding", "vision_encoder"):
        validator._create_session(result / name / "model.onnx", "cuda")
    ids, logits, _ = validator.run_token_ids(
        result,
        [1, 42, 17],
        hidden_size=256,
        max_new_tokens=20,
        device="cpu",
    )
    assert ids == [
        134,
        244,
        242,
        167,
        81,
        34,
        155,
        251,
        142,
        90,
        224,
        31,
        76,
        250,
        240,
        120,
        134,
        120,
        134,
        120,
    ]
    assert all(np.isfinite(value).all() for value in logits)
