# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import pytest
import torch

from mobius import ModelPackage, build_from_gguf
from mobius._flags import override_flags
from mobius.integrations.gguf._builder_test_utils import _write_quantized_gguf

pytestmark = [
    pytest.mark.opset29,
    pytest.mark.skipif(
        not all(onnx.defs.has(name, 29) for name in ("DequantizePacked", "GroupedMatMul")),
        reason="Official ONNX opset-29 packed/grouped schemas require current ONNX main.",
    ),
]


@pytest.mark.parametrize(
    "operation", [torch.log, lambda x: x - 1.0], ids=["inverse", "offset"]
)
def test_floating_reparameterization_fails_closed(operation) -> None:
    from mobius.integrations.gguf._packed import _LayoutGraph

    layout = _LayoutGraph()
    proxy = torch.empty([32], device="meta")
    layout.remember(
        proxy,
        ir.Value(name="weight", shape=ir.Shape([32]), type=ir.TensorType(ir.DataType.FLOAT)),
    )
    with layout, pytest.raises(ValueError, match="reparameterization"):
        operation(proxy)


@pytest.fixture(scope="module", autouse=True)
def packed_sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from mobius.integrations.gguf._builder_test_utils import _write_moe_gguf

    root = tmp_path_factory.mktemp("packed_models")
    paths = {}
    for name in [
        "qwen2",
        "llama",
        "split",
        "lossy",
        "optimize",
        "parity",
        "qwen2moe",
        "qwen3moe",
    ]:
        directory = root / name
        directory.mkdir()
        path = directory / "weights.gguf"
        if name in {"qwen2moe", "qwen3moe"}:
            _write_moe_gguf(path, name, "q4_0")
        elif name == "split":
            _write_quantized_gguf(
                path,
                architecture="qwen2",
                float_type="bf16",
                split_max_tensors=6,
                quantize_embedding=True,
            )
            path = sorted(directory.glob("*.gguf"))[0]
        else:
            _write_quantized_gguf(
                path,
                architecture="llama" if name in {"llama", "parity", "lossy"} else "qwen2",
                quantize_embedding=True,
                output_quantization="q8_0" if name in {"llama", "qwen2"} else None,
                projection_quantization="q5_1" if name == "lossy" else "q4_0",
            )
        paths[name] = path
    return paths


@pytest.mark.parametrize(
    "qtype",
    [
        "Q4_0",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "Q8_0",
        "Q2_K",
        "Q3_K",
        "Q4_K",
        "Q5_K",
        "Q6_K",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ3_XXS",
        "IQ1_S",
        "IQ4_NL",
        "IQ3_S",
        "IQ2_S",
        "IQ4_XS",
        "IQ1_M",
        "TQ1_0",
        "TQ2_0",
        "MXFP4",
        "NVFP4",
        "Q1_0",
    ],
)
def test_all_stored_reader_formats(tmp_path: Path, qtype: str) -> None:
    from types import SimpleNamespace

    import gguf
    from onnx.reference import ReferenceEvaluator

    from mobius.integrations.gguf._packed import bind_packed_weights
    from mobius.integrations.gguf._reader import GGUFModel

    kind = gguf.GGMLQuantizationType[qtype]
    block, size = gguf.GGML_QUANT_SIZES[kind]
    raw = np.random.default_rng(42).integers(0, 256, (2, size), dtype=np.uint8)
    half = np.array([0.125], dtype="<f2").view(np.uint8)
    if qtype in {"Q2_K", "Q3_K", "Q6_K"}:
        raw[:, -2:] = half
        if qtype == "Q2_K":
            raw[:, -4:-2] = half
    elif qtype in {"MXFP4", "NVFP4"}:
        raw[:, : 1 if qtype == "MXFP4" else 4] = 120 if qtype == "MXFP4" else 56
    elif qtype == "IQ1_M":
        scales = raw[:, 48:].copy().view("<u2")
        # FLOAT16 0.125 = 0x3000, distributed across the four high nibbles.
        scales &= 0x0FFF
        scales[:, 3] |= 0x3000
        raw[:, 48:] = scales.view(np.uint8)
    else:
        raw[:, :2] = half
        if qtype in {"Q4_1", "Q5_1", "Q4_K", "Q5_K"}:
            raw[:, 2:4] = half
    source = tmp_path / "weights.gguf"
    writer = gguf.GGUFWriter(str(source), "qwen2")
    writer.add_tensor("token_embd.weight", raw, raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    weight = ir.Value(
        name="model.embed_tokens.weight",
        shape=ir.Shape([2, block]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    ids = ir.Value(
        name="ids", shape=ir.Shape([None, None]), type=ir.TensorType(ir.DataType.INT64)
    )
    out = ir.Value(
        name="out", shape=ir.Shape([None, None, block]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    model = ir.Model(
        ir.Graph(
            [ids],
            [out],
            nodes=[ir.Node("", "Gather", [weight, ids], outputs=[out])],
            initializers=[weight],
            opset_imports={"": 29},
            name="packed_gather",
        ),
        ir_version=10,
    )
    package = ModelPackage({"model": model})
    reader = GGUFModel(source)
    bind_packed_weights(
        package,
        reader,
        SimpleNamespace(preprocess_weights=lambda state: state),
        SimpleNamespace(model_type="qwen2"),
    )
    assert [value.name for value in model.graph.outputs] == ["out"]
    package.save(str(tmp_path))
    restored = ModelPackage.load(str(tmp_path))
    assert [value.name for value in restored["model"].graph.outputs] == ["out"]
    onnx.checker.check_model(str(tmp_path / "model.onnx"), full_check=True)
    evaluator = ReferenceEvaluator(str(tmp_path / "model.onnx"))
    indices = np.array([[1, 0, 1], [-1, -2, 0]], dtype=np.int64)
    actual = evaluator.run(["out"], {"ids": indices})[0]
    if qtype == "Q4_0":
        empty = evaluator.run(None, {"ids": np.empty((1, 0), dtype=np.int64)})[0]
        assert empty.shape == (1, 0, block)
        for invalid in [-3, -4, -100, 2]:
            with pytest.raises((ValueError, RuntimeError)):
                evaluator.run(None, {"ids": np.array([[invalid]], dtype=np.int64)})
    if qtype == "Q1_0":
        bits = np.unpackbits(raw[:, 2:], axis=-1, bitorder="little")
        expected = (bits.astype(np.float32) * 2 - 1) * 0.125
    else:
        expected = gguf.dequantize(raw, kind)
    assert np.isfinite(expected).all() and np.count_nonzero(expected)
    np.testing.assert_allclose(actual, expected[indices], rtol=1e-6, atol=1e-7)
    payload = next(
        v.const_value
        for v in model.graph.initializers.values()
        if isinstance(v.const_value, ir.ExternalTensor)
    )
    assert payload.numpy().tobytes() == raw.tobytes()
    assert any(n.op_type == "Scan" for n in model.graph.all_nodes())


@pytest.mark.parametrize("architecture", ["qwen2", "llama"])
def test_packed_build_roundtrip(packed_sources: dict[str, Path], architecture: str) -> None:
    source = packed_sources[architecture]
    tmp_path = source.parent
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(source, execution_provider="onnx-standard")
    before = package.gguf_packed_sources[source.name]["sha256"]
    assert package["model"].opset_imports[""] == 29
    assert package.gguf_quantization_report.source_fidelity
    assert any(n.op_type == "DequantizePacked" for n in package["model"].graph)
    package.save(str(tmp_path))
    onnx.checker.check_model(str(tmp_path / "model.onnx"), full_check=True)
    restored = ModelPackage.load(str(tmp_path))
    restored.save(str(tmp_path))
    assert not (tmp_path / "model.onnx.data").exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    externals = [
        value.const_value
        for value in restored["model"].graph.initializers.values()
        if isinstance(value.const_value, ir.ExternalTensor)
    ]
    assert externals
    assert all(t.location == source.name for t in externals)


def test_split_shards_and_bf16_preserve_external_ranges(
    packed_sources: dict[str, Path],
) -> None:
    source = packed_sources["split"]
    tmp_path = source.parent
    shards = sorted(tmp_path.glob("*.gguf"))
    assert len(shards) > 1
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(shards[0], execution_provider="onnx-standard")
    package.save(str(tmp_path))
    assert set(package.gguf_packed_sources) == {p.name for p in shards}
    assert any(
        isinstance(v.const_value, ir.ExternalTensor)
        and v.const_value.dtype == ir.DataType.BFLOAT16
        for v in package["model"].graph.initializers.values()
    )
    onnx.checker.check_model(str(tmp_path / "model.onnx"), full_check=True)
    ModelPackage.load(str(tmp_path)).save(str(tmp_path))


def test_flag_off_rejects_previously_lossy_projection(packed_sources: dict[str, Path]) -> None:
    source = packed_sources["lossy"]
    with (
        override_flags(experimental_opset29=False),
        pytest.raises(ValueError, match=r"EXPERIMENTAL_OPSET29|opset29|lossy"),
    ):
        build_from_gguf(source)
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(source, execution_provider="onnx-standard")
    assert package.gguf_quantization_report.source_fidelity


def test_manifest_integrity_and_failed_save_preserve_model(
    packed_sources: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Python 3.10 has no file_digest; package hashing must not depend on it.
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    source = packed_sources["lossy"]
    root = source.parent
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(source, execution_provider="onnx-standard")
    package.save(str(root))
    original_model = (root / "model.onnx").read_bytes()
    manifest = root / "gguf-packed.json"
    original_manifest = manifest.read_text(encoding="utf-8")
    corrupt = json.loads(original_manifest)
    corrupt["sources"][source.name]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(corrupt), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="bytes changed"):
            ModelPackage.load(str(root))
    finally:
        manifest.write_text(original_manifest, encoding="utf-8")
    initializer = next(
        v
        for v in package["model"].graph.initializers.values()
        if isinstance(v.const_value, ir.ExternalTensor)
    )
    initializer.const_value = ir.tensor(np.zeros(1, dtype=np.uint8))
    with pytest.raises(ValueError, match="binding"):
        package.save(str(root))
    assert (root / "model.onnx").read_bytes() == original_model
    assert not (root / ".gguf-packed.lock").exists()


def test_packed_save_rejects_missing_generated_initializer(
    packed_sources: dict[str, Path],
) -> None:
    source = packed_sources["parity"]
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(source, execution_provider="onnx-standard")
    initializer = next(
        value
        for value in package["model"].graph.initializers.values()
        if value.const_value is not None
        and not isinstance(value.const_value, ir.ExternalTensor)
    )
    initializer.const_value = None
    with pytest.raises(ValueError, match="weight"):
        package.save(str(source.parent))


def test_optimization_cannot_fold_original_external_weights(
    packed_sources: dict[str, Path],
) -> None:
    from mobius._optimizations import fold_initializers_after_weights, optimize_model

    source = packed_sources["optimize"]
    tmp_path = source.parent
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(source, execution_provider="onnx-standard")
        optimize_model(package["model"], ep="onnx-standard")
        fold_initializers_after_weights(package["model"])
    package.save(str(tmp_path))
    assert not (tmp_path / "model.onnx.data").exists()


@pytest.mark.parametrize("architecture", ["qwen2moe", "qwen3moe"])
def test_packed_expert_layout_and_standard_moe(
    packed_sources: dict[str, Path], architecture: str
) -> None:
    source = packed_sources[architecture]
    tmp_path = source.parent
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(source, execution_provider="onnx-standard")
    package.save(str(tmp_path))
    if architecture == "qwen3moe":
        assert any(n.op_type == "GroupedMatMul" for n in package["model"].graph.all_nodes())
    assert any(
        v.name.endswith("ffn_up_exps.weight") and isinstance(v.const_value, ir.ExternalTensor)
        for v in package["model"].graph.initializers.values()
    )
    onnx.checker.check_model(str(tmp_path / "model.onnx"), full_check=True)


@pytest.mark.parametrize("architecture", ["parity", "qwen2moe", "qwen3moe"])
def test_packed_next_token_matches_float_source(
    packed_sources: dict[str, Path], architecture: str
) -> None:
    import onnxruntime as ort
    from onnx.reference import ReferenceEvaluator

    source = packed_sources[architecture]
    tmp_path = source.parent
    with override_flags(experimental_opset29=True):
        packed = build_from_gguf(source, execution_provider="onnx-standard")
    packed.save(str(tmp_path))
    dense = build_from_gguf(source, keep_quantized=False, execution_provider="onnx-standard")
    dense_dir = tmp_path / "dense"
    dense.save(str(dense_dir))
    reference = ReferenceEvaluator(str(tmp_path / "model.onnx"))
    dense_reference = ReferenceEvaluator(str(dense_dir / "model.onnx"))
    runtime = ort.InferenceSession(str(dense_dir / "model.onnx"))
    state = {}
    for tokens in [[1, 2, 3], [4], [5, 6]]:
        past_length = next(iter(state.values())).shape[2] if state else 0
        total = past_length + len(tokens)
        feeds = {
            "input_ids": np.array([tokens], dtype=np.int64),
            "attention_mask": np.ones((1, total), dtype=np.int64),
            "position_ids": np.arange(past_length, total, dtype=np.int64)[None],
        }
        for inp in runtime.get_inputs():
            if inp.name.startswith("past_key_values"):
                feeds[inp.name] = state.get(
                    inp.name, np.zeros((1, 2, 0, 16), dtype=np.float32)
                )
        actual = reference.run(None, feeds)
        expected = runtime.run(None, feeds)
        same_backend = dense_reference.run(None, feeds)
        for a, b in zip(actual, same_backend, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)
        state = {
            output.name.replace("present", "past_key_values"): array
            for output, array in zip(runtime.get_outputs(), expected, strict=True)
            if output.name != "logits"
        }
