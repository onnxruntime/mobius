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

pytestmark = pytest.mark.opset29


def _has_schema29(name: str) -> bool:
    return onnx.defs.has(name, 29) and onnx.defs.get_schema(name, 29).since_version == 29


requires_packed_schema = pytest.mark.skipif(
    not _has_schema29("DequantizePacked"),
    reason="DequantizePacked-29 is required for checking and reference execution.",
)
requires_grouped_schema = pytest.mark.skipif(
    not _has_schema29("GroupedMatMul"),
    reason="GroupedMatMul-29 is required for checking and reference execution.",
)


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
        "mtp_shared",
        "mtp_dedicated",
        "shared_components",
        "qwen3moe_8",
        "qwen3moe_64",
        "qwen35",
    ]:
        directory = root / name
        directory.mkdir()
        path = directory / "weights.gguf"
        if name == "qwen35":
            from mobius.integrations.gguf._builder_test_utils import _write_qwen35_gguf

            _write_qwen35_gguf(path, quantized=True, inner_size=64, quantized_embedding=True)
        elif name.startswith("mtp_"):
            from mobius.integrations.gguf._mtp_test import _write_qwen35_mtp_gguf

            dedicated = name == "mtp_dedicated"
            _write_qwen35_mtp_gguf(
                path,
                quantized=True,
                dedicated_embedding=dedicated,
                dedicated_head=dedicated,
                quantized_backbone_embedding=True,
            )
        elif name.startswith("qwen3moe_"):
            _write_moe_gguf(path, "qwen3moe", "q4_0", num_experts=int(name.split("_")[1]))
        elif name in {"qwen2moe", "qwen3moe"}:
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
        if name == "shared_components":
            second = directory / "other.gguf"
            _write_quantized_gguf(second, architecture="qwen2", quantize_embedding=True)
            paths["shared_second"] = second
        paths[name] = path
    return paths


@pytest.mark.parametrize(
    "validate_schema",
    [False, pytest.param(True, marks=requires_packed_schema)],
    ids=["binding-roundtrip", "reference"],
)
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
def test_all_stored_reader_formats(tmp_path: Path, qtype: str, validate_schema: bool) -> None:
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
    payload = next(
        v.const_value
        for v in restored["model"].graph.initializers.values()
        if isinstance(v.const_value, ir.ExternalTensor)
    )
    assert payload.numpy().tobytes() == raw.tobytes()
    assert any(n.op_type == "Scan" for n in model.graph.all_nodes())
    if not validate_schema:
        return
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


@requires_packed_schema
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


@pytest.mark.parametrize(
    "architecture",
    ["qwen2moe", pytest.param("qwen3moe", marks=requires_grouped_schema)],
)
@requires_grouped_schema
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
    for node in package["model"].graph.all_nodes():
        if node.op_type == "GroupedMatMul":
            bank = node.inputs[1]
            assert bank is not None
            producer = bank.producer()
            assert producer is not None and producer.op_type == "DequantizePacked", producer
    assert any(
        v.name.endswith("ffn_up_exps.weight") and isinstance(v.const_value, ir.ExternalTensor)
        for v in package["model"].graph.initializers.values()
    )


@requires_packed_schema
@pytest.mark.parametrize(
    "architecture",
    [
        "qwen2",
        "llama",
        "split",
        "qwen2moe",
        pytest.param("qwen3moe", marks=requires_grouped_schema),
    ],
)
def test_packed_model_schema(packed_sources: dict[str, Path], architecture: str) -> None:
    source = packed_sources[architecture]
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(source, execution_provider="onnx-standard")
    package.save(str(source.parent))
    onnx.checker.check_model(str(source.parent / "model.onnx"), full_check=True)


@requires_packed_schema
@pytest.mark.parametrize(
    "architecture",
    ["parity", "qwen2moe", pytest.param("qwen3moe", marks=requires_grouped_schema)],
)
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


@pytest.mark.parametrize("dedicated", [False, True])
@pytest.mark.parametrize(
    "validate_schema",
    [False, pytest.param(True, marks=requires_packed_schema)],
    ids=["binding-roundtrip", "reference"],
)
def test_original_mtp_composition_roundtrip(
    packed_sources: dict[str, Path], dedicated: bool, validate_schema: bool
) -> None:
    from onnx.reference import ReferenceEvaluator

    from mobius.integrations.gguf._reader import GGUFModel

    source = packed_sources["mtp_dedicated" if dedicated else "mtp_shared"]
    before = source.read_bytes()
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(source, execution_provider="onnx-standard")
    assert package.mtp_head is not None
    package.save(source.parent, progress_bar=False)
    restored = ModelPackage.load(str(source.parent))
    assert restored.mtp_head is not None
    assert set(restored) == {"model"}
    restored.save(source.parent, progress_bar=False)
    assert source.read_bytes() == before
    manifest = json.loads((source.parent / "gguf-packed.json").read_text())
    assert manifest["format"] == "mobius.gguf-packed.v2"
    assert len(manifest["sources"]) == 1
    assert {row["depth"] for row in manifest["models"]} == {0, 1}
    for model in (restored["model"], restored.mtp_head["model"]):
        assert any(
            isinstance(value.const_value, ir.ExternalTensor)
            for value in model.graph.initializers.values()
        )
    if not validate_schema:
        return
    for row in manifest["models"]:
        onnx.checker.check_model(str(source.parent / row["file"]), full_check=True)

    dense = build_from_gguf(source, keep_quantized=False, execution_provider="onnx-standard")
    dense_dir = source.parent / "dense"
    dense.save(dense_dir, progress_bar=False)
    dense = ModelPackage.load(str(dense_dir))
    assert dense.mtp_head is not None
    for model in (restored["model"], restored.mtp_head["model"]):
        assert any(
            isinstance(value.const_value, ir.ExternalTensor)
            for value in model.graph.initializers.values()
        )
    target = ReferenceEvaluator(str(source.parent / manifest["models"][0]["file"]))
    head = ReferenceEvaluator(str(source.parent / manifest["models"][1]["file"]))
    target_dense = ReferenceEvaluator(str(dense_dir / "model.onnx"))
    head_dense = ReferenceEvaluator(str(dense_dir / ".mobius-mtp" / "model.onnx"))
    reader = GGUFModel(source)
    table = reader.get_tensor("token_embd.weight")
    target_state = {}
    head_state = {}
    for tokens in [[1, 2], [3], [4, 5]]:
        past = next(iter(target_state.values())).shape[2] if target_state else 0
        ids = np.asarray([tokens], dtype=np.int64)
        common = {
            "attention_mask": np.ones((1, past + len(tokens)), dtype=np.int64),
            "position_ids": np.arange(past, past + len(tokens), dtype=np.int64)[None],
        }

        def caches(state):
            return {
                f"past_key_values.0.{kind}": state.get(
                    kind, np.zeros((1, 2, 0, 16), dtype=np.float32)
                )
                for kind in ("key", "value")
            }

        feeds = {**common, "input_ids": ids, **caches(target_state)}
        actual = target.run(None, feeds)
        expected = target_dense.run(None, feeds)
        for value, reference in zip(actual, expected, strict=True):
            assert np.isfinite(value).all()
            np.testing.assert_allclose(value, reference, rtol=2e-4, atol=2e-4)
        outputs = dict(zip(target.output_names, actual, strict=True))
        mtp_feeds = {
            **common,
            **caches(head_state),
            "hidden_states": outputs["mtp_seed"],
            **({"input_ids": ids} if dedicated else {"inputs_embeds": table[ids]}),
        }
        mtp_actual = head.run(None, mtp_feeds)
        mtp_expected = head_dense.run(None, mtp_feeds)
        for value, reference in zip(mtp_actual, mtp_expected, strict=True):
            assert np.isfinite(value).all()
            np.testing.assert_allclose(value, reference, rtol=2e-4, atol=2e-4)
        mtp_outputs = dict(zip(head.output_names, mtp_actual, strict=True))
        target_state = {kind: outputs[f"present.0.{kind}"] for kind in ("key", "value")}
        head_state = {kind: mtp_outputs[f"present.0.{kind}"] for kind in ("key", "value")}


def test_shared_source_components_validate_manifest(
    packed_sources: dict[str, Path],
) -> None:
    source = packed_sources["shared_components"]
    # Compose independently bound graphs without manually copying source registries.
    with override_flags(experimental_opset29=True):
        first = build_from_gguf(source, execution_provider="onnx-standard")
        second = build_from_gguf(
            packed_sources["shared_second"], execution_provider="onnx-standard"
        )
    package = ModelPackage(
        {
            "decoder": first["model"],
            "embedding": second["model"],
            "shared_decoder": first["model"],
        }
    )
    package.save(source.parent, progress_bar=False)
    loaded = ModelPackage.load(str(source.parent))
    assert set(loaded) == {"decoder", "embedding", "shared_decoder"}
    assert set(loaded.gguf_packed_sources) == {"weights.gguf", "other.gguf"}
    loaded.save(source.parent, progress_bar=False)
    manifest_path = source.parent / "gguf-packed.json"
    original = manifest_path.read_text()
    manifest = json.loads(original)
    manifest["models"][1]["file"] = manifest["models"][0]["file"]
    manifest_path.write_text(json.dumps(manifest))
    try:
        with pytest.raises(ValueError, match="missing/duplicate"):
            ModelPackage.load(str(source.parent))
    finally:
        manifest_path.write_text(original)


@requires_grouped_schema
def test_original_expert_bank_graph_size_is_independent_of_expert_count(
    packed_sources: dict[str, Path],
) -> None:
    counts = []
    for key in ("qwen3moe", "qwen3moe_8", "qwen3moe_64"):
        with override_flags(experimental_opset29=True):
            package = build_from_gguf(packed_sources[key], execution_provider="onnx-standard")
        graph = package["model"].graph
        counts.append(len(list(graph.all_nodes())))
        assert sum(node.op_type == "GroupedMatMul" for node in graph.all_nodes()) == 3
    assert counts[0] == counts[1] == counts[2]


@pytest.mark.parametrize(
    "validate_schema",
    [False, pytest.param(True, marks=requires_packed_schema)],
    ids=["binding-roundtrip", "reference"],
)
def test_original_deltanet_decay_needs_no_inverse_reparameterization(
    packed_sources: dict[str, Path], validate_schema: bool
) -> None:
    source = packed_sources["qwen35"]
    before = source.read_bytes()
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(source, execution_provider="onnx-standard")
    linear = [
        node
        for node in package["model"].graph
        if not node.domain and node.op_type == "LinearAttention"
    ]
    assert len(linear) == 3
    for node in linear:
        assert node.attributes["q_num_heads"].as_int() == 2
        assert node.attributes["kv_num_heads"].as_int() == 2
        assert node.attributes["v_num_heads"].as_int() == 4
    assert not any(node.op_type == "Exp" for node in package["model"].graph)
    package.save(source.parent, progress_bar=False)
    restored = ModelPackage.load(str(source.parent))
    restored.save(source.parent, progress_bar=False)
    assert source.read_bytes() == before
    if not validate_schema:
        return

    import onnxruntime as ort
    from onnx.reference import ReferenceEvaluator
    from onnx.reference.op_run import OpRun

    class LinearAttention(OpRun):
        def _run(
            self,
            query,
            key,
            value,
            past_state,
            decay,
            beta,
            *,
            q_num_heads=None,
            kv_num_heads=None,
            v_num_heads=0,
            scale=0.0,
            update_rule="gated_delta",
            chunk_size=64,
        ):
            assert update_rule == "gated_delta"
            assert q_num_heads == kv_num_heads == 2 and v_num_heads == 4
            batch, length, _ = query.shape
            q = np.repeat(query.reshape(batch, length, 2, 16), 2, axis=2)
            k = np.repeat(key.reshape(batch, length, 2, 16), 2, axis=2)
            v = value.reshape(batch, length, 4, 16)
            state = past_state.copy()
            result = []
            for step in range(length):
                state *= np.exp(decay[:, step, :, None, None])
                residual = v[:, step] - np.einsum("bhkm,bhk->bhm", state, k[:, step])
                state += np.einsum(
                    "bhk,bhm->bhkm", k[:, step], residual * beta[:, step, :, None]
                )
                result.append(np.einsum("bhk,bhkm->bhm", q[:, step], state) * scale)
            return np.stack(result, axis=1).reshape(batch, length, 64), state

    reference = ReferenceEvaluator(
        str(source.parent / "model.onnx"), new_ops=[LinearAttention]
    )
    dense = build_from_gguf(source, keep_quantized=False, execution_provider="onnx-standard")
    dense_dir = source.parent / "dense"
    dense.save(dense_dir, progress_bar=False)
    runtime = ort.InferenceSession(str(dense_dir / "model.onnx"))
    state = {
        f"past_key_values.{layer}.{kind}": np.zeros(shape, dtype=np.float32)
        for layer in range(3)
        for kind, shape in (
            ("conv_state", (1, 128, 2)),
            ("recurrent_state", (1, 4, 16, 16)),
        )
    }
    state.update(
        (f"past_key_values.3.{kind}", np.zeros((1, 2, 0, 8), dtype=np.float32))
        for kind in ("key", "value")
    )
    position = 0
    for tokens in [[1, 2, 3], [4], [5, 6]]:
        total = position + len(tokens)
        feeds = {
            "input_ids": np.asarray([tokens], dtype=np.int64),
            "attention_mask": np.ones((1, total), dtype=np.int64),
            "position_ids": np.arange(position, total, dtype=np.int64)[None],
            **state,
        }
        actual = reference.run(None, feeds)
        expected = runtime.run(None, feeds)
        for value, baseline in zip(actual, expected, strict=True):
            assert np.isfinite(value).all()
            np.testing.assert_allclose(value, baseline, rtol=2e-4, atol=2e-4)
        state = {
            name.replace("present.", "past_key_values."): value
            for name, value in zip(reference.output_names, actual, strict=True)
            if name.startswith("present.")
        }
        position = total
