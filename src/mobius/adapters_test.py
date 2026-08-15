# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for generic low-rank adapter artifacts and request state."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from safetensors.numpy import save_file

from mobius import (
    AdapterApplication,
    AdapterArtifact,
    AdapterBatchSelection,
    AdapterRowSelection,
    AdapterServiceOptions,
    AdapterSource,
    AdapterTarget,
    AdapterTargetDescriptor,
    AdapterTargetManifest,
    AdapterTargetSlice,
    AdapterWeights,
    ModelPackage,
    adapter_source_from_onnx_adapter,
    compose_adapter_deltas,
    fingerprint_model_weights,
    load_peft_adapter,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    add_adapter_service_to_workflow,
)


def _model(weight: np.ndarray | None = None) -> ir.Model:
    values = np.arange(12, dtype=np.float32).reshape(3, 4) if weight is None else weight
    initializer = ir.Value(
        name="projection.weight",
        const_value=ir.tensor(values),
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape(values.shape),
    )
    x = ir.val(
        "hidden_states",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([1, 4]),
    )
    projection = ir.Node("", "MatMul", [x, initializer], name="projection")
    output = projection.outputs[0]
    output.name = "projection.output"
    graph = ir.Graph(
        [x],
        [output],
        nodes=[projection],
        initializers=[initializer],
        name="adapter_test_model",
    )
    return ir.Model(graph, ir_version=10)


def _weights(
    *,
    component: str = "decoder",
    parameter: str = "projection.weight",
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    alpha: float = 4.0,
) -> AdapterWeights:
    a_values = np.arange(8, dtype=np.float32).reshape(2, 4) / 10 if a is None else a
    b_values = np.arange(6, dtype=np.float32).reshape(3, 2) / 10 if b is None else b
    return AdapterWeights(
        AdapterTarget(component, parameter),
        ir.tensor(a_values),
        ir.tensor(b_values),
        alpha,
    )


def _artifact(model: ir.Model, name: str = "style") -> AdapterArtifact:
    models = {"decoder": model}
    return AdapterArtifact(
        name=name,
        base_fingerprint=fingerprint_model_weights(models),
        weights=(_weights(),),
    )


def _manifest(model: ir.Model, *, include_second: bool = False) -> AdapterTargetManifest:
    targets = [
        AdapterTargetDescriptor(
            AdapterTarget("decoder", "projection.weight"),
            semantic_name="layers.0.self_attn.q_proj",
            node_name="projection",
            output_name="projection.output",
            input_size=4,
            output_size=3,
            layer_index=0,
            slices=(AdapterTargetSlice("q", 0, 3, rank=2, alpha=4.0),),
        )
    ]
    if include_second:
        targets.append(
            AdapterTargetDescriptor(
                AdapterTarget("decoder", "other.weight"),
                semantic_name="layers.0.self_attn.v_proj",
                node_name="other",
                output_name="other.output",
                input_size=4,
                output_size=3,
            )
        )
    return AdapterTargetManifest(fingerprint_model_weights({"decoder": model}), tuple(targets))


def test_artifact_validates_base_target_shape_dtype_and_checksum() -> None:
    model = _model()
    artifact = _artifact(model)

    artifact.validate_base({"decoder": model})
    assert artifact.checksum.startswith("sha256:")
    assert artifact.checksum == _artifact(model).checksum

    changed = _model(np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="base fingerprint mismatch"):
        artifact.validate_base({"decoder": changed})


def test_authoritative_target_manifest_validates_exact_graph_binding() -> None:
    model = _model()
    manifest = _manifest(model)
    manifest.validate({"decoder": model})
    assert manifest.bindings == {AdapterTarget("decoder", "projection.weight")}

    stale = AdapterTargetManifest(
        manifest.base_fingerprint,
        (
            AdapterTargetDescriptor(
                AdapterTarget("decoder", "projection.weight"),
                "layers.0.self_attn.q_proj",
                "projection",
                "stale.output",
                4,
                3,
            ),
        ),
    )
    with pytest.raises(ValueError, match="does not produce"):
        stale.validate({"decoder": model})


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (_weights(parameter="missing.weight"), "unknown parameter"),
        (
            _weights(a=np.ones((2, 5), dtype=np.float32)),
            "B @ A has shape",
        ),
    ],
)
def test_artifact_rejects_invalid_targets(weights: AdapterWeights, message: str) -> None:
    model = _model()
    artifact = AdapterArtifact(
        name="invalid",
        base_fingerprint=fingerprint_model_weights({"decoder": model}),
        weights=(weights,),
    )
    with pytest.raises(ValueError, match=message):
        artifact.validate_base({"decoder": model})


def test_adapter_weights_reject_mismatched_factor_dtype() -> None:
    with pytest.raises(ValueError, match="same dtype"):
        _weights(a=np.ones((2, 4), dtype=np.float16))


def test_lora_delta_matches_reference_math() -> None:
    weights = _weights(alpha=6.0)
    expected = weights.b.numpy() @ weights.a.numpy() * 3.0
    np.testing.assert_allclose(weights.delta(), expected, rtol=1e-6, atol=1e-6)


def test_composed_delta_matches_scaled_sum() -> None:
    model = _model()
    style = _artifact(model, "style")
    speaker = AdapterArtifact(
        "speaker",
        style.base_fingerprint,
        (_weights(alpha=2.0),),
    )
    row = AdapterRowSelection(
        100,
        1,
        (
            AdapterApplication("style", 0.25),
            AdapterApplication("speaker", 1.5),
        ),
    )
    actual = compose_adapter_deltas(row, {"style": style, "speaker": speaker})
    target = AdapterTarget("decoder", "projection.weight")
    expected = style.weights[0].delta() * 0.25 + speaker.weights[0].delta() * 1.5
    np.testing.assert_allclose(actual[target], expected, rtol=1e-6, atol=1e-6)


def test_zero_one_and_composed_per_row_adapters() -> None:
    batch = AdapterBatchSelection(
        (
            AdapterRowSelection(row_id=100, request_epoch=4),
            AdapterRowSelection(
                row_id=101,
                request_epoch=7,
                adapters=(AdapterApplication("style", 0.5),),
            ),
            AdapterRowSelection(
                row_id=102,
                request_epoch=2,
                adapters=(
                    AdapterApplication("style", 0.25),
                    AdapterApplication("speaker", 1.5),
                ),
            ),
        )
    )
    model = _model()
    catalog = {
        "style": _artifact(model, "style"),
        "speaker": _artifact(model, "speaker"),
    }
    batch.validate_catalog(catalog)

    assert batch.rows[0].adapters == ()
    assert [item.adapter for item in batch.rows[2].adapters] == ["style", "speaker"]


def test_compaction_preserves_semantic_rows_and_slot_reuse_uses_epoch() -> None:
    original = AdapterBatchSelection(
        (
            AdapterRowSelection(100, 1, (AdapterApplication("style"),)),
            AdapterRowSelection(101, 5, (AdapterApplication("speaker", 0.25),)),
        )
    )
    compacted = original.compact([1, 0])
    assert [row.row_id for row in compacted.rows] == [101, 100]
    assert compacted.rows[0].request_epoch == 5
    assert compacted.compact([1, 0]) == original
    assert compacted.referenced_adapters == {"style", "speaker"}

    reused_slot = AdapterRowSelection(
        row_id=200,
        request_epoch=2,
        adapters=(AdapterApplication("speaker"),),
    )
    assert reused_slot != original.rows[0]


def test_model_package_catalog_validates_and_rejects_duplicates() -> None:
    model = _model()
    artifact = _artifact(model)
    package = ModelPackage({"decoder": model}, adapter_target_manifest=_manifest(model))
    package.add_adapter_artifact(artifact)
    assert package.adapter_artifacts["style"] is artifact
    assert artifact.nbytes == 56

    with pytest.raises(ValueError, match="already attached"):
        package.add_adapter_artifact(artifact)

    with pytest.raises(ValueError, match="checksum mismatch"):
        artifact.validate_checksum("sha256:" + "0" * 64)


def test_model_package_requires_n_adapter_target_alignment() -> None:
    model = _model()
    other = ir.Value(
        name="other.weight",
        const_value=ir.tensor(np.ones((3, 4), dtype=np.float32)),
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([3, 4]),
    )
    x = model.graph.inputs[0]
    node = ir.Node("", "MatMul", [x, other], name="other")
    node.outputs[0].name = "other.output"
    model.graph.append(node)
    model.graph.initializers.add(other)
    package = ModelPackage(
        {"decoder": model}, adapter_target_manifest=_manifest(model, include_second=True)
    )
    package.add_adapter_artifact(_artifact(model, "style"))
    second = AdapterArtifact(
        "speaker",
        fingerprint_model_weights({"decoder": model}),
        (_weights(parameter="other.weight"),),
    )
    with pytest.raises(ValueError, match="does not align"):
        package.add_adapter_artifact(second)


def test_peft_migration_source_preserves_rank_alpha_and_provenance() -> None:
    directory = Path("artifacts") / f"adapter-peft-test-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        config = {
            "base_model_name_or_path": "synthetic/base",
            "revision": "producer-fixture",
            "r": 4,
            "lora_alpha": 8.0,
            "target_modules": ["q_proj"],
            "rank_pattern": {"layers.0.self_attn.q_proj": 2},
            "alpha_pattern": {"self_attn.q_proj": 6.0},
        }
        (directory / "adapter_config.json").write_text(json.dumps(config))
        module = "base_model.model.layers.0.self_attn.q_proj"
        a = np.arange(8, dtype=np.float32).reshape(2, 4)
        b = np.arange(6, dtype=np.float32).reshape(3, 2)
        save_file(
            {
                f"{module}.lora_A.weight": a,
                f"{module}.lora_B.weight": b,
            },
            directory / "adapter_model.safetensors",
        )
        model = _model()
        artifact = load_peft_adapter(
            directory,
            name="peft-style",
            base_fingerprint=fingerprint_model_weights({"decoder": model}),
            target_bindings={
                "layers.0.self_attn.q_proj": AdapterTarget("decoder", "projection.weight")
            },
        )
        artifact.validate_base({"decoder": model})
        assert artifact.weights[0].rank == 2
        assert artifact.weights[0].alpha == pytest.approx(6.0)
        assert artifact.source.format == "peft_safetensors"
        assert artifact.source.base_model == "synthetic/base"
        assert artifact.source.revision == "producer-fixture"
        np.testing.assert_allclose(
            artifact.weights[0].delta(), b @ a * 3.0, rtol=1e-6, atol=1e-6
        )
    finally:
        shutil.rmtree(directory)


def test_onnx_adapter_migration_source_is_optional_and_checksummed() -> None:
    directory = Path("artifacts") / f"adapter-ort-test-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        path = directory / "style.onnx_adapter"
        path.write_bytes(b"\x00\x00\x00\x00TORTsynthetic")
        source = adapter_source_from_onnx_adapter(path)
        assert source.format == "onnx_adapter"
        assert source.checksum is not None
        artifact = AdapterArtifact(
            "style",
            _artifact(_model()).base_fingerprint,
            (_weights(),),
            source=source,
        )
        assert artifact.source.path == str(path)
    finally:
        shutil.rmtree(directory)


def test_onnx_adapter_source_can_be_declared_for_native_capability() -> None:
    directory = Path("artifacts") / f"adapter-native-test-{uuid.uuid4().hex}"
    source_directory = directory / "source"
    output_directory = directory / "package"
    source_directory.mkdir(parents=True)
    output_directory.mkdir()
    try:
        source_path = source_directory / "style.onnx_adapter"
        source_path.write_bytes(b"\x00\x00\x00\x00TORTsynthetic")
        model = _model()
        package = ModelPackage(
            {"decoder": model},
            adapter_target_manifest=_manifest(model),
            adapter_service_options=AdapterServiceOptions(
                portable_fallback=False,
                preserve_source_format=True,
            ),
        )
        package.add_adapter_artifact(
            AdapterArtifact(
                "style",
                fingerprint_model_weights({"decoder": model}),
                (_weights(alpha=2.0),),
                source=adapter_source_from_onnx_adapter(source_path),
            )
        )
        catalog = package.save_adapter_artifacts(str(output_directory))
        declared = catalog["style"]["weights"][0]
        assert declared["format"] == "ort_genai"
        assert declared["location"] == "adapters/style.onnx_adapter"
        copied = output_directory / declared["location"]
        assert copied.read_bytes() == source_path.read_bytes()
        assert declared["sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    finally:
        shutil.rmtree(directory)


def test_adapter_source_rejects_unprovenanced_external_artifact() -> None:
    with pytest.raises(ValueError, match="requires a path"):
        AdapterSource("onnx_adapter")


def test_exact_onnx_genai_catalog_and_portable_bundle_serialization() -> None:
    directory = Path("artifacts") / f"adapter-export-test-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        model = _model()
        package = ModelPackage(
            {"decoder": model},
            adapter_target_manifest=_manifest(model),
            adapter_service_options=AdapterServiceOptions(
                row_ids="request.row_ids",
                active="request.active",
                cache_max_entries=2,
            ),
        )
        package.add_adapter_artifact(
            AdapterArtifact(
                "red",
                fingerprint_model_weights({"decoder": model}),
                (_weights(alpha=2.0),),
                identity="style-red",
                version="2026.08",
            )
        )
        metadata = {
            "pipeline": {
                "workflow": {
                    "manifest": {"capabilities": []},
                    "inputs": {
                        "request.row_ids": {
                            "contract": {
                                "dtype": "int64",
                                "rank": 1,
                                "shape": ["batch"],
                            }
                        },
                        "request.active": {
                            "contract": {
                                "dtype": "bool",
                                "rank": 1,
                                "shape": ["batch"],
                            }
                        },
                    },
                    "components": {"decoder": {"implementation": {"kind": "binding"}}},
                    "steps": [],
                }
            }
        }
        add_adapter_service_to_workflow(metadata, package, str(directory))
        service = metadata["pipeline"]["workflow"]["adapters"]
        assert service["base_model_fingerprint"].startswith("sha256:")
        assert service["row_ids"] == "request.row_ids"
        assert service["request_epochs"] == "request.request_epochs"
        assert service["active"] == "request.active"
        assert service["application_capability"] == "onnx-genai.adapters"
        assert service["cache"] == {"max_entries": 2, "eviction": "lru"}
        assert service["planning"] == {
            "bucket_by_adapter_set": True,
            "stable_buffers": True,
            "invalidate_capture_on_eviction": True,
        }
        assert metadata["pipeline"]["workflow"]["inputs"]["request.request_epochs"] == {
            "contract": {"dtype": "int64", "rank": 1, "shape": ["batch"]},
            "role": {
                "kind": "runtime",
                "version": "1.0",
                "role": "request_epochs",
            },
            "source": {"kind": "request"},
        }
        artifact = service["artifacts"]["red"]
        assert artifact["identity"] == "style-red"
        assert artifact["version"] == "2026.08"
        assert artifact["rank"] == 2
        assert artifact["alpha"] == pytest.approx(2.0)
        assert artifact["dtype"] == "float32"
        assert artifact["targets"] == [
            {
                "component": "decoder",
                "parameter": "projection.weight",
                "weight_key": "layers.0.self_attn.q_proj",
                "input_features": 4,
                "output_features": 3,
            }
        ]
        weight = artifact["weights"][0]
        payload = (directory / weight["location"]).read_bytes()
        assert weight["format"] == "json"
        assert len(weight["sha256"]) == 64
        assert weight["sha256"] == hashlib.sha256(payload).hexdigest()
        bundle = json.loads(payload)
        assert set(bundle["targets"]) == {"layers.0.self_attn.q_proj"}
        assert len(bundle["targets"]["layers.0.self_attn.q_proj"]["a"]) == 8
        assert len(bundle["targets"]["layers.0.self_attn.q_proj"]["b"]) == 6
    finally:
        shutil.rmtree(directory)


def test_wire_contract_rejects_heterogeneous_target_rank() -> None:
    model = _model()
    other = ir.Value(
        name="other.weight",
        const_value=ir.tensor(np.ones((3, 4), dtype=np.float32)),
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([3, 4]),
    )
    node = ir.Node("", "MatMul", [model.graph.inputs[0], other], name="other")
    node.outputs[0].name = "other.output"
    model.graph.append(node)
    model.graph.initializers.add(other)
    package = ModelPackage(
        {"decoder": model},
        adapter_target_manifest=_manifest(model, include_second=True),
    )
    package.add_adapter_artifact(
        AdapterArtifact(
            "mixed-rank",
            fingerprint_model_weights({"decoder": model}),
            (
                _weights(),
                _weights(
                    parameter="other.weight",
                    a=np.ones((1, 4), dtype=np.float32),
                    b=np.ones((3, 1), dtype=np.float32),
                ),
            ),
        )
    )
    directory = Path("artifacts") / f"adapter-rank-test-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        with pytest.raises(ValueError, match="heterogeneous target rank"):
            package.save_adapter_artifacts(str(directory))
    finally:
        shutil.rmtree(directory)


def test_application_scale_matches_runtime_bound() -> None:
    with pytest.raises(ValueError, match=r"within \[-16, 16\]"):
        AdapterApplication("style", 16.1)


def test_selection_rejects_unknown_adapter_and_invalid_permutation() -> None:
    batch = AdapterBatchSelection(
        (AdapterRowSelection(100, 0, (AdapterApplication("missing"),)),)
    )
    with pytest.raises(ValueError, match="unknown adapter"):
        batch.validate_catalog({})
    with pytest.raises(ValueError, match="permutation"):
        batch.compact([1])
