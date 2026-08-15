# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for generic low-rank adapter artifacts and request state."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from mobius import (
    AdapterApplication,
    AdapterArtifact,
    AdapterBatchSelection,
    AdapterRowSelection,
    AdapterTarget,
    AdapterWeights,
    ModelPackage,
    compose_adapter_deltas,
    fingerprint_model_weights,
)


def _model(weight: np.ndarray | None = None) -> ir.Model:
    values = np.arange(12, dtype=np.float32).reshape(3, 4) if weight is None else weight
    initializer = ir.Value(
        name="projection.weight",
        const_value=ir.tensor(values),
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape(values.shape),
    )
    graph = ir.Graph([], [], nodes=[], initializers=[initializer], name="adapter_test_model")
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


def test_artifact_validates_base_target_shape_dtype_and_checksum() -> None:
    model = _model()
    artifact = _artifact(model)

    artifact.validate_base({"decoder": model})
    assert artifact.checksum.startswith("sha256:")
    assert artifact.checksum == _artifact(model).checksum

    changed = _model(np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="base fingerprint mismatch"):
        artifact.validate_base({"decoder": changed})


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

    reused_slot = AdapterRowSelection(
        row_id=200,
        request_epoch=2,
        adapters=(AdapterApplication("speaker"),),
    )
    assert reused_slot != original.rows[0]


def test_model_package_catalog_validates_and_rejects_duplicates() -> None:
    model = _model()
    artifact = _artifact(model)
    package = ModelPackage({"decoder": model})
    package.add_adapter_artifact(artifact)
    assert package.adapter_artifacts["style"] is artifact

    with pytest.raises(ValueError, match="already attached"):
        package.add_adapter_artifact(artifact)

    with pytest.raises(ValueError, match="checksum mismatch"):
        artifact.validate_checksum("sha256:" + "0" * 64)


def test_selection_rejects_unknown_adapter_and_invalid_permutation() -> None:
    batch = AdapterBatchSelection(
        (AdapterRowSelection(100, 0, (AdapterApplication("missing"),)),)
    )
    with pytest.raises(ValueError, match="unknown adapter"):
        batch.validate_catalog({})
    with pytest.raises(ValueError, match="permutation"):
        batch.compact([1])
