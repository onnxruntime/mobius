# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import onnx_ir as ir
import pytest

from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai.inference_metadata_test import (
    _model,
    _native_package,
    _VlmConfig,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_language_diffusion_pipeline_metadata,
    build_speculative_workflow_metadata,
    build_vlm_workflow_metadata,
    write_speculative_workflow_metadata,
)


def test_speculative_writer_saves_policy_artifacts(tmp_path):
    write_speculative_workflow_metadata(_speculative_package(), str(tmp_path))
    assert (tmp_path / "policies" / "speculative_acceptance.onnx").is_file()
    assert (tmp_path / "policies" / "branch_state.onnx").is_file()


def test_speculative_emit_uses_accepted_prefix_length():
    workflow = build_speculative_workflow_metadata(_speculative_package())["pipeline"][
        "workflow"
    ]
    emit = next(node for node in workflow["graph"]["body"]["nodes"] if node["kind"] == "emit")
    assert emit["valid_length"] == "acceptance.synchronized_length"
    assert "emit_valid_length" in workflow["manifest"]["capabilities"]
    assert workflow["outputs"]["tokens"]["contract"]["shape"][-1] == "accepted_sequence"
    assert workflow["state"]["cache_0"]["recurrence"] == {
        "kind": "bounded",
        "axis": 2,
        "max": "package.max_context",
    }
    assert workflow["inputs"]["package.max_context"]["default"] == 4096


def _value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def test_vlm_preprocessing_is_explicit_typed_ssa(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"use_hd_transform": True}), encoding="utf-8"
    )
    (source / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "dynamic_hd": 1,
                "crop_size": 16,
                "include_thumbnail": False,
                "thumbnail_order": "none",
                "mask_patch_size": 1,
            }
        ),
        encoding="utf-8",
    )
    vision = _model(
        "vision_encoder",
        [
            _value("pixel_values", ir.DataType.FLOAT, [1, 3, 16, 16]),
            _value("image_sizes", ir.DataType.INT64, [1, 2]),
            _value("image_attention_mask", ir.DataType.FLOAT, [1, 16, 16]),
        ],
        [("image_features", ir.DataType.FLOAT, [1, 64])],
    )

    metadata = build_vlm_workflow_metadata(
        _native_package(vision, _VlmConfig()),
        _VlmConfig(),
        source=str(source),
    )
    image = metadata["preprocessing"]["image"]
    declared = {name for transform in image["transforms"] for name in transform["outputs"]}
    assert "inputs" not in image["transforms"][0]
    assert all("outputs" in transform for transform in image["transforms"])
    assert all(output["source"] in declared for output in image["outputs"])


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


def _graph_model(
    name: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
) -> ir.Model:
    return ir.Model(
        ir.Graph(
            inputs=inputs,
            outputs=outputs,
            nodes=[],
            name=name,
            opset_imports={"": 24},
        ),
        ir_version=11,
    )


def _speculative_package() -> ModelPackage:
    proposer = _graph_model(
        "proposer",
        [_value("tokens", ir.DataType.INT64, ["batch", 4])],
        [
            _value("proposed_tokens", ir.DataType.INT64, ["batch", 4]),
            _value("proposal_scores", ir.DataType.FLOAT, ["batch", 4, 32]),
        ],
    )
    verifier = _graph_model(
        "verifier",
        [
            _value("proposed_tokens", ir.DataType.INT64, ["batch", 4]),
            _value(
                "past_key_values.0.key",
                ir.DataType.FLOAT,
                ["batch", 2, "past_sequence", 8],
            ),
        ],
        [
            _value("target_scores", ir.DataType.FLOAT, ["batch", 4, 32]),
            _value(
                "present.0.key",
                ir.DataType.FLOAT,
                ["batch", 2, "past_sequence + 4", 8],
            ),
        ],
    )
    return ModelPackage({"proposer": proposer, "verifier": verifier})


def test_speculative_workflow_uses_branch_phi_effect_join_and_rng():
    workflow = build_speculative_workflow_metadata(_speculative_package())["pipeline"][
        "workflow"
    ]
    body = workflow["graph"]["body"]["nodes"]
    branch = next(node for node in body if node["kind"] == "branch")
    assert branch["kind"] == "branch"
    assert branch["outputs"]["tokens.next"]["cases"] == {
        "true": "branch.accepted",
        "false": "branch.corrected",
    }
    assert branch["effects"]["state"]["produces"] == "branch.state.out"
    acceptance = body[2]
    assert acceptance["inputs"]["offset"] == "state.rng_offset.body"
    assert acceptance["outputs"]["next_offset"] == "rng_offset.body"
    rollback = next(node for node in body if node.get("component") == "rollback_cache_0")
    assert rollback["inputs"]["accepted_len"] == "acceptance.synchronized_length"
    assert branch["outputs"]["cache_0.next"]["cases"] == {
        "true": "branch.accepted.cache_0",
        "false": "branch.corrected.cache_0",
    }
    assert branch["effects"]["branch:cache_0"]["produces"] == "branch.cache_0.out"
    assert any(item["cell"].startswith("cache_") for item in workflow["graph"]["carried"])
