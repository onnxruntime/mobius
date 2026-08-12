# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import onnx_ir as ir
import pytest

from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_language_diffusion_pipeline_metadata,
)


def _value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _masked_denoiser_package() -> ModelPackage:
    input_ids = _value("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    logits = _value("logits", ir.DataType.FLOAT, ["batch", "sequence", 128])
    proposed = _value("proposed_tokens", ir.DataType.INT64, ["batch", "sequence"])
    graph = ir.Graph(
        inputs=[input_ids],
        outputs=[logits, proposed],
        nodes=[],
        name="masked_denoiser",
        opset_imports={"": 24},
    )
    return ModelPackage({"model": ir.Model(graph, ir_version=11)})


def test_language_diffusion_uses_exclusive_ssa_workflow():
    metadata = build_language_diffusion_pipeline_metadata(
        _masked_denoiser_package(),
        num_inference_steps=8,
    )
    pipeline = metadata["pipeline"]
    assert set(pipeline) == {"workflow"}

    workflow = pipeline["workflow"]
    assert workflow["state"]["tokens"]["recurrence"] == {"kind": "invariant"}
    assert workflow["state"]["tokens"]["contract"]["shape"] == ["batch", "sequence"]
    assert workflow["state"]["rng_offset"]["contract"]["shape"] == ["batch"]
    assert workflow["components"]["masked_update"]["policy"] == {
        "role": "masked_update",
        "state": "current_tokens",
        "proposal": "proposed_tokens",
        "mask": "masked",
        "step": "step",
        "next_state": "next_state",
        "next_mask": "next_mask",
        "rng": {
            "seed": "seed",
            "offset": "offset",
            "next_offset": "next_offset",
        },
        "effect": "update",
    }

    graph = workflow["graph"]
    assert graph["kind"] == "loop"
    assert graph["condition"] == "denoiser.body.continue"
    assert graph["max_iterations"] == "request.max_iterations"
    assert [node["component"] for node in graph["setup"]["nodes"]] == ["model"]
    assert [node["kind"] for node in graph["body"]["nodes"]] == [
        "invoke",
        "invoke",
        "invoke",
        "emit",
        "invoke",
    ]
    assert graph["body"]["nodes"][0]["inputs"]["total_steps"] == "package.num_steps"
    assert graph["body"]["nodes"][-2]["mode"] == "replace"


def test_language_diffusion_rejects_zero_steps():
    with pytest.raises(ValueError, match="num_inference_steps"):
        build_language_diffusion_pipeline_metadata(
            _masked_denoiser_package(),
            num_inference_steps=0,
        )


def test_language_diffusion_matches_pr_828_schema():
    schema_path = (
        Path(__file__).parents[4] / "tests" / "schemas" / "onnx_genai_4c3c4b6.schema.json"
    )
    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    metadata = build_language_diffusion_pipeline_metadata(
        _masked_denoiser_package(),
        num_inference_steps=8,
    )
    jsonschema.validate(instance=metadata, schema=schema)
