# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for generic low-rank adapter artifacts and request state."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from safetensors.numpy import load_file, save_file

from mobius import (
    AdapterApplication,
    AdapterArtifact,
    AdapterBatchSelection,
    AdapterServiceOptions,
    AdapterSlotSelection,
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
    add_adapter_service_to_metadata,
)


def _mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for nested in value.values() for key in _mapping_keys(nested)}
    if isinstance(value, list):
        return {key for nested in value for key in _mapping_keys(nested)}
    return set()


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
    weight_key: str | None = None,
    target_id: str | None = None,
) -> AdapterWeights:
    a_values = np.arange(8, dtype=np.float32).reshape(2, 4) / 10 if a is None else a
    b_values = np.arange(6, dtype=np.float32).reshape(3, 2) / 10 if b is None else b
    return AdapterWeights(
        AdapterTarget(component, parameter),
        ir.tensor(a_values),
        ir.tensor(b_values),
        alpha,
        weight_key=weight_key,
        target_id=target_id,
    )


def _artifact(model: ir.Model, name: str = "style") -> AdapterArtifact:
    models = {"decoder": model}
    weights = _weights()
    return AdapterArtifact(
        name=name,
        base_fingerprint=fingerprint_model_weights(models, (weights.target,)),
        weights=(weights,),
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
            activation_dtype=ir.DataType.FLOAT,
            graph_input_a="lora.q_proj.a",
            graph_input_b="lora.q_proj.b",
            graph_input_scale="lora.q_proj.scale",
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
    return AdapterTargetManifest(
        fingerprint_model_weights({"decoder": model}, tuple(targets)),
        tuple(targets),
    )


def test_artifact_validates_base_target_shape_dtype_and_checksum() -> None:
    model = _model()
    artifact = _artifact(model)

    artifact.validate_base({"decoder": model})
    assert artifact.checksum.startswith("sha256:")
    assert artifact.checksum == _artifact(model).checksum

    changed = _model(np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="base fingerprint mismatch"):
        artifact.validate_base({"decoder": changed})


def test_targeted_fingerprint_excludes_unrelated_weights_and_includes_consumers() -> None:
    model = _model()
    target = AdapterTarget("decoder", "projection.weight")
    fingerprint = fingerprint_model_weights({"decoder": model}, (target,))
    assert fingerprint.startswith("onnx-genai-targeted-base-v1:sha256:")

    unrelated = ir.Value(
        name="unrelated.weight",
        const_value=ir.tensor(np.ones((1,), dtype=np.float32)),
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([1]),
    )
    model.graph.initializers.add(unrelated)
    assert fingerprint_model_weights({"decoder": model}, (target,)) == fingerprint

    next(iter(model.graph)).attributes["producer_contract"] = ir.AttrInt64(
        "producer_contract", 1
    )
    assert fingerprint_model_weights({"decoder": model}, (target,)) != fingerprint


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
    row = AdapterSlotSelection(
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
            AdapterSlotSelection(slot_id=100, request_epoch=4),
            AdapterSlotSelection(
                slot_id=101,
                request_epoch=7,
                adapters=(AdapterApplication("style", 0.5),),
            ),
            AdapterSlotSelection(
                slot_id=102,
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

    assert batch.slots[0].adapters == ()
    assert [item.adapter for item in batch.slots[2].adapters] == ["style", "speaker"]


def test_compaction_preserves_semantic_rows_and_slot_reuse_uses_epoch() -> None:
    original = AdapterBatchSelection(
        (
            AdapterSlotSelection(100, 1, (AdapterApplication("style"),)),
            AdapterSlotSelection(101, 5, (AdapterApplication("speaker", 0.25),)),
        )
    )
    compacted = original.compact([1, 0])
    assert [slot.slot_id for slot in compacted.slots] == [101, 100]
    assert compacted.slots[0].request_epoch == 5
    assert compacted.compact([1, 0]) == original
    assert compacted.referenced_adapters == {"style", "speaker"}

    reused_slot = AdapterSlotSelection(
        slot_id=200,
        request_epoch=2,
        adapters=(AdapterApplication("speaker"),),
    )
    assert reused_slot != original.slots[0]


def test_selection_lowers_to_stable_fixed_shape_request_tensors() -> None:
    model = _model()
    artifact = _artifact(model)
    batch = AdapterBatchSelection(
        (
            AdapterSlotSelection(
                100,
                4,
                (
                    AdapterApplication("red", 0.5),
                    AdapterApplication("blue", -0.25),
                ),
            ),
            AdapterSlotSelection(101, 5, (AdapterApplication("blue", 1.0),)),
        )
    )
    tensors = batch.to_tensors(
        {"blue": artifact, "red": artifact},
        max_adapters=3,
        active=[True, False],
    )
    assert tensors.aliases == ("blue", "red")
    np.testing.assert_array_equal(tensors.slot_ids, [100, 101])
    np.testing.assert_array_equal(tensors.request_epochs, [4, 5])
    np.testing.assert_array_equal(tensors.segments, [[1, 0, -1], [-1, -1, -1]])
    np.testing.assert_array_equal(tensors.adapter_counts, [2, 0])
    np.testing.assert_array_equal(tensors.scales, [[0.5, -0.25, 0.0], [0.0, 0.0, 0.0]])
    assert tensors.segments.dtype == np.int64
    assert tensors.scales.dtype == np.float32

    compacted = batch.compact([1, 0]).to_tensors(
        {"blue": artifact, "red": artifact},
        max_adapters=3,
    )
    np.testing.assert_array_equal(compacted.slot_ids, [101, 100])
    np.testing.assert_array_equal(compacted.request_epochs, [5, 4])
    with pytest.raises(ValueError, match="exceeding max_adapters"):
        batch.to_tensors({"blue": artifact, "red": artifact}, max_adapters=1)


def test_model_package_catalog_validates_and_rejects_duplicates() -> None:
    model = _model()
    manifest = _manifest(model)
    artifact = AdapterArtifact("style", manifest.base_fingerprint, (_weights(),))
    package = ModelPackage({"decoder": model}, adapter_target_manifest=manifest)
    package.add_adapter_artifact(artifact)
    assert package.adapter_artifacts["style"] is artifact
    assert artifact.nbytes == 56

    with pytest.raises(ValueError, match="already attached"):
        package.add_adapter_artifact(artifact)

    with pytest.raises(ValueError, match="checksum mismatch"):
        artifact.validate_checksum("sha256:" + "0" * 64)


def test_model_package_allows_distinct_manifest_targets_per_adapter() -> None:
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
    manifest = _manifest(model, include_second=True)
    package = ModelPackage({"decoder": model}, adapter_target_manifest=manifest)
    package.add_adapter_artifact(
        AdapterArtifact("style", manifest.base_fingerprint, (_weights(),))
    )
    second = AdapterArtifact(
        "speaker",
        manifest.base_fingerprint,
        (_weights(parameter="other.weight"),),
    )
    package.add_adapter_artifact(second)
    assert package.adapter_artifacts["style"].target_bindings == {
        AdapterTarget("decoder", "projection.weight")
    }
    assert package.adapter_artifacts["speaker"].target_bindings == {
        AdapterTarget("decoder", "other.weight")
    }


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
        manifest = _manifest(model)
        artifact = load_peft_adapter(
            directory,
            name="peft-style",
            base_fingerprint=manifest.base_fingerprint,
            target_bindings={
                "layers.0.self_attn.q_proj": AdapterTarget("decoder", "projection.weight")
            },
        )
        artifact.validate_base({"decoder": model}, fingerprint_targets=manifest.targets)
        assert artifact.weights[0].rank == 2
        assert artifact.weights[0].alpha == pytest.approx(6.0)
        assert artifact.source.format == "peft_safetensors"
        assert artifact.source.base_model == "synthetic/base"
        assert artifact.source.revision == "producer-fixture"
        package = ModelPackage(
            {"decoder": model},
            adapter_target_manifest=manifest,
            adapter_service_options=AdapterServiceOptions(
                portable_fallback=False,
                preserve_source_format=True,
            ),
        )
        package.add_adapter_artifact(artifact)
        output = directory / "package"
        catalog = package.save_adapter_artifacts(str(output))
        declaration = catalog["peft-style"]["weights"][0]
        assert declaration["format"] == "hf_peft"
        assert declaration["loader_capability"] == "onnx-genai.adapters.hf-peft@1"
        assert declaration["scale_encoding"] == "alpha_over_rank"
        assert catalog["peft-style"]["provenance"] == {
            "producer": "mobius",
            "source": "synthetic/base",
            "revision": "producer-fixture",
        }
        saved = load_file(output / declaration["location"])
        assert set(saved) == {
            f"{module}.lora_A.weight",
            f"{module}.lora_B.weight",
        }
        assert (output / declaration["config_location"]).read_bytes() == (
            directory / "adapter_config.json"
        ).read_bytes()
        np.testing.assert_allclose(
            artifact.weights[0].delta(), b @ a * 3.0, rtol=1e-6, atol=1e-6
        )
    finally:
        shutil.rmtree(directory)


def test_peft_rank_and_alpha_patterns_emit_heterogeneous_binding_overrides() -> None:
    directory = Path("artifacts") / f"adapter-peft-pattern-test-{uuid.uuid4().hex}"
    source = directory / "source"
    source.mkdir(parents=True)
    try:
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
        manifest = _manifest(model, include_second=True)

        config = {
            "r": 1,
            "lora_alpha": 2.0,
            "target_modules": ["q_proj", "v_proj"],
            "rank_pattern": {"layers.0.self_attn.q_proj": 2},
            "alpha_pattern": {"layers.0.self_attn.q_proj": 6.0},
        }
        (source / "adapter_config.json").write_text(json.dumps(config))
        save_file(
            {
                "base_model.layers.0.self_attn.q_proj.lora_A.weight": np.ones(
                    (2, 4), dtype=np.float32
                ),
                "base_model.layers.0.self_attn.q_proj.lora_B.weight": np.ones(
                    (3, 2), dtype=np.float32
                ),
                "base_model.layers.0.self_attn.v_proj.lora_A.weight": np.ones(
                    (1, 4), dtype=np.float32
                ),
                "base_model.layers.0.self_attn.v_proj.lora_B.weight": np.ones(
                    (3, 1), dtype=np.float32
                ),
            },
            source / "adapter_model.safetensors",
        )
        artifact = load_peft_adapter(
            source,
            name="heterogeneous",
            base_fingerprint=manifest.base_fingerprint,
            target_bindings={
                "layers.0.self_attn.q_proj": AdapterTarget("decoder", "projection.weight"),
                "layers.0.self_attn.v_proj": AdapterTarget("decoder", "other.weight"),
            },
        )
        package = ModelPackage(
            {"decoder": model},
            adapter_target_manifest=manifest,
            adapter_service_options=AdapterServiceOptions(
                portable_fallback=False,
                preserve_source_format=True,
            ),
        )
        package.add_adapter_artifact(artifact)
        catalog = package.save_adapter_artifacts(str(directory / "package"))

        assert set(catalog) == {"heterogeneous"}
        declaration = catalog["heterogeneous"]
        assert declaration["rank"] == 1
        assert declaration["alpha"] == pytest.approx(2.0)
        bindings = {binding["target"]: binding for binding in declaration["bindings"]}
        assert bindings["layers.0.self_attn.q_proj"] == {
            "target": "layers.0.self_attn.q_proj",
            "weight_key": "layers.0.self_attn.q_proj",
            "rank": 2,
            "alpha": 6.0,
        }
        assert bindings["layers.0.self_attn.v_proj"] == {
            "target": "layers.0.self_attn.v_proj",
            "weight_key": "layers.0.self_attn.v_proj",
        }
        assert len(declaration["weights"]) == 1
        assert declaration["weights"][0]["format"] == "hf_peft"
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
        manifest = _manifest(model)
        package = ModelPackage(
            {"decoder": model},
            adapter_target_manifest=manifest,
            adapter_service_options=AdapterServiceOptions(
                portable_fallback=False,
                preserve_source_format=True,
            ),
        )
        package.add_adapter_artifact(
            AdapterArtifact(
                "style",
                manifest.base_fingerprint,
                (_weights(alpha=2.0),),
                source=adapter_source_from_onnx_adapter(source_path),
            )
        )
        catalog = package.save_adapter_artifacts(str(output_directory))
        declared = catalog["style"]["weights"][0]
        assert declared["format"] == "ort_genai"
        assert declared["loader_capability"] == "onnxruntime.lora-adapter@1"
        assert declared["scale_encoding"] == "baked"
        assert declared["location"] == "adapters/style/adapter.onnx_adapter"
        assert catalog["style"]["bindings"] == [
            {
                "target": "layers.0.self_attn.q_proj",
                "weight_key": "layers.0.self_attn.q_proj",
            }
        ]
        copied = output_directory / declared["location"]
        assert copied.read_bytes() == source_path.read_bytes()
        assert "sha256" not in declared
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
        manifest = _manifest(model)
        package = ModelPackage(
            {"decoder": model},
            adapter_target_manifest=manifest,
            adapter_service_options=AdapterServiceOptions(
                active="request.active",
                max_adapters=2,
                cache_max_entries=2,
            ),
        )
        package.add_adapter_artifact(
            AdapterArtifact(
                "red",
                manifest.base_fingerprint,
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
                        "request.slot_ids": {
                            "contract": {
                                "dtype": "int64",
                                "rank": 1,
                                "shape": ["batch"],
                            },
                            "role": {"kind": "opaque"},
                            "source": {
                                "kind": "application",
                                "name": "serving.slot_ids",
                            },
                        },
                        "request.active": {
                            "contract": {
                                "dtype": "bool",
                                "rank": 1,
                                "shape": ["batch"],
                            },
                            "role": {
                                "kind": "runtime",
                                "version": "1.0",
                                "role": "adapter_active",
                            },
                            "source": {"kind": "request"},
                        },
                    },
                    "components": {"decoder": {"implementation": {"kind": "binding"}}},
                    "steps": [],
                }
            }
        }
        add_adapter_service_to_metadata(metadata, package, str(directory))
        service = metadata["adapters"]
        assert "adapters" not in metadata["pipeline"]["workflow"]
        assert {
            "sha256",
            "config_sha256",
            "base_model_fingerprint",
        }.isdisjoint(_mapping_keys(metadata))
        assert service["selection"] == {
            "segments": "request.adapter_segments",
            "adapter_counts": "request.adapter_counts",
            "scales": "request.adapter_scales",
            "active": "request.active",
            "max_adapters": 2,
        }
        assert service["application_capability"] == "onnx-genai.adapters@1"
        assert service["discovery_fallback"] == "disabled"
        target = service["target_manifest"]["targets"][0]
        assert target == {
            "id": "layers.0.self_attn.q_proj",
            "component": "decoder",
            "initializer": "projection.weight",
            "layer_index": 0,
            "node_name": "projection",
            "output_name": "projection.output",
            "activation_dtype": "float32",
            "input_features": 4,
            "output_features": 3,
            "graph_inputs": {
                "a": "lora.q_proj.a",
                "b": "lora.q_proj.b",
                "scale": "lora.q_proj.scale",
            },
        }
        assert service["target_manifest"]["targets"][1]["output_slice"] == {
            "role": "q",
            "offset": 0,
            "width": 3,
            "rank": 2,
            "alpha": 4.0,
        }
        assert service["cache"] == {"max_entries": 2, "eviction": "lru"}
        assert service["planning"] == {
            "bucket_by_adapter_set": True,
            "stable_buffers": True,
            "invalidate_capture_on_eviction": True,
        }
        assert not any(
            "request_epochs" in name for name in metadata["pipeline"]["workflow"]["inputs"]
        )
        artifact = service["artifacts"]["red"]
        assert artifact["index"] == 0
        assert artifact["identity"] == "style-red"
        assert artifact["version"] == "2026.08"
        assert artifact["rank"] == 2
        assert artifact["alpha"] == pytest.approx(2.0)
        assert artifact["dtype"] == "float32"
        assert artifact["provenance"] == {"producer": "mobius"}
        assert artifact["bindings"] == [
            {
                "target": "layers.0.self_attn.q_proj",
                "weight_key": "layers.0.self_attn.q_proj",
            }
        ]
        weight = artifact["weights"][0]
        payload = (directory / weight["location"]).read_bytes()
        assert weight["format"] == "json"
        assert weight["loader_capability"] == "onnx-genai.adapters.json@1"
        assert weight["scale_encoding"] == "alpha_over_rank"
        assert weight["location"] == "adapters/red/adapter.json"
        bundle = json.loads(payload)
        assert set(bundle["targets"]) == {"layers.0.self_attn.q_proj"}
        assert len(bundle["targets"]["layers.0.self_attn.q_proj"]["a"]) == 8
        assert len(bundle["targets"]["layers.0.self_attn.q_proj"]["b"]) == 6
    finally:
        shutil.rmtree(directory)


def test_top_level_adapter_metadata_supports_bare_model_package() -> None:
    directory = Path("artifacts") / f"adapter-bare-test-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        model = _model()
        target = AdapterTarget("model", "projection.weight")
        descriptor = AdapterTargetDescriptor(
            target,
            semantic_name="projection",
            node_name="projection",
            output_name="projection.output",
            input_size=4,
            output_size=3,
            activation_dtype=ir.DataType.FLOAT,
        )
        fingerprint = fingerprint_model_weights({"model": model}, (descriptor,))
        package = ModelPackage(
            {"model": model},
            adapter_target_manifest=AdapterTargetManifest(fingerprint, (descriptor,)),
        )
        package.add_adapter_artifact(
            AdapterArtifact(
                "style",
                fingerprint,
                (_weights(component="model"),),
            )
        )
        metadata: dict[str, object] = {"schema_version": "v1"}
        add_adapter_service_to_metadata(metadata, package, str(directory))
        assert metadata["adapters"]["target_manifest"]["targets"][0]["component"] == "model"
        assert "pipeline" not in metadata
    finally:
        shutil.rmtree(directory)


def test_wire_contract_emits_per_target_rank_override() -> None:
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
    manifest = _manifest(model, include_second=True)
    package = ModelPackage({"decoder": model}, adapter_target_manifest=manifest)
    package.add_adapter_artifact(
        AdapterArtifact(
            "mixed-rank",
            manifest.base_fingerprint,
            (
                _weights(target_id="layers.0.self_attn.q_proj.q"),
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
        artifact = package.save_adapter_artifacts(str(directory))["mixed-rank"]
        assert artifact["rank"] == 1
        bindings = {binding["target"]: binding for binding in artifact["bindings"]}
        assert bindings["layers.0.self_attn.q_proj.q"]["rank"] == 2
        assert "rank" not in bindings["layers.0.self_attn.v_proj"]
    finally:
        shutil.rmtree(directory)


def test_wire_contract_rejects_manifest_rank_policy_violation() -> None:
    model = _model()
    descriptor = AdapterTargetDescriptor(
        AdapterTarget("decoder", "projection.weight"),
        semantic_name="projection",
        node_name="projection",
        output_name="projection.output",
        input_size=4,
        output_size=3,
        rank=1,
        alpha=4.0,
    )
    manifest = AdapterTargetManifest(
        fingerprint_model_weights({"decoder": model}, (descriptor,)),
        (descriptor,),
    )
    package = ModelPackage({"decoder": model}, adapter_target_manifest=manifest)
    package.add_adapter_artifact(
        AdapterArtifact("style", manifest.base_fingerprint, (_weights(),))
    )
    directory = Path("artifacts") / f"adapter-policy-test-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        with pytest.raises(ValueError, match="violates manifest policy"):
            package.save_adapter_artifacts(str(directory))
    finally:
        shutil.rmtree(directory)


def test_application_scale_matches_runtime_bound() -> None:
    with pytest.raises(ValueError, match=r"within \[-16, 16\]"):
        AdapterApplication("style", 16.1)


def test_selection_rejects_duplicate_adapter() -> None:
    application = AdapterApplication("style")
    with pytest.raises(ValueError, match="contains duplicate adapter"):
        AdapterSlotSelection(100, 0, (application, application))


def test_selection_rejects_unknown_adapter_and_invalid_permutation() -> None:
    batch = AdapterBatchSelection(
        (AdapterSlotSelection(100, 0, (AdapterApplication("missing"),)),)
    )
    with pytest.raises(ValueError, match="unknown adapter"):
        batch.validate_catalog({})
    with pytest.raises(ValueError, match="permutation"):
        batch.compact([1])
