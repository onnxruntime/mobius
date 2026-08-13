# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path

import onnx_ir as ir
import pytest
import yaml

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
    write_vlm_workflow_metadata,
)


def test_speculative_writer_saves_policy_artifacts(tmp_path):
    write_speculative_workflow_metadata(_speculative_package(), str(tmp_path))
    assert (tmp_path / "policies" / "speculative_acceptance.onnx").is_file()
    assert (tmp_path / "policies" / "cache_length_update.onnx").is_file()


def test_speculative_writer_saves_guidance_and_adaptive_artifacts(tmp_path):
    write_speculative_workflow_metadata(
        _speculative_package(adaptive=True),
        str(tmp_path),
        grammar_guidance=True,
        adaptive_k_max=4,
    )
    assert (tmp_path / "policies" / "grammar_guidance.onnx").is_file()
    assert (tmp_path / "policies" / "adaptive_k.onnx").is_file()


def test_speculative_emit_uses_accepted_prefix_length():
    workflow = build_speculative_workflow_metadata(_speculative_package())["pipeline"][
        "workflow"
    ]
    emit = next(node for node in workflow["steps"][0]["steps"] if node["kind"] == "emit")
    assert emit["valid_length"] == "acceptance.length"
    assert "emit_valid_length" in workflow["manifest"]["capabilities"]
    assert workflow["outputs"]["tokens"]["contract"]["shape"][-1] == "accepted_sequence"
    assert workflow["state"]["cache_0"]["recurrence"] == {
        "kind": "bounded",
        "axis": 2,
        "max": "package.max_context",
    }
    assert workflow["state"]["cache_0"]["service_group"] == "verifier_cache"
    assert workflow["serving"]["accepted_len"] == "accepted_len"
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


def test_vlm_writer_derives_real_decoder_contract_from_artifact(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"use_hd_transform": True}), encoding="utf-8"
    )
    (source / "genai_config.json").write_text(
        json.dumps(
            {
                "model": {"eos_token_id": 200001, "context_length": 131072},
                "search": {"past_present_share_buffer": True},
            }
        ),
        encoding="utf-8",
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

    decoder_inputs = [
        _value("inputs_embeds", ir.DataType.BFLOAT16, ["batch", "sequence", 6656]),
        _value(
            "attention_mask",
            ir.DataType.INT64,
            ["batch", "past_sequence + sequence"],
        ),
    ]
    decoder_outputs = [("logits", ir.DataType.BFLOAT16, ["batch", "sequence", 202048])]
    for layer in range(52):
        cache_shape = ["batch", 2, "past_sequence", 128]
        present_shape = ["batch", 2, "total_sequence", 128]
        for kind in ("key", "value"):
            decoder_inputs.append(
                _value(
                    f"past_key_values.{layer}.{kind}",
                    ir.DataType.BFLOAT16,
                    cache_shape,
                )
            )
            decoder_outputs.append(
                (
                    f"present.{layer}.{kind}",
                    ir.DataType.BFLOAT16,
                    present_shape,
                )
            )
    decoder = _model("decoder", decoder_inputs, decoder_outputs)
    vision = _model(
        "vision_encoder",
        [
            _value("pixel_values", ir.DataType.FLOAT, [1, 3, 16, 16]),
            _value("image_sizes", ir.DataType.INT64, [1, 2]),
            _value("image_attention_mask", ir.DataType.FLOAT, [1, 16, 16]),
        ],
        [("image_features", ir.DataType.BFLOAT16, ["image_tokens", 6656])],
    )
    embedding = _model(
        "embedding",
        [
            _value("input_ids", ir.DataType.INT64, ["batch", "sequence"]),
            _value(
                "image_features",
                ir.DataType.BFLOAT16,
                ["image_tokens", 6656],
            ),
        ],
        [("inputs_embeds", ir.DataType.BFLOAT16, ["batch", "sequence", 6656])],
    )
    package = ModelPackage(
        {"decoder": decoder, "vision_encoder": vision, "embedding": embedding}
    )

    # Deliberately tiny config values must never override admitted artifact I/O.
    config = _VlmConfig()
    config.eos_token_id = 2
    path = write_vlm_workflow_metadata(
        package,
        str(tmp_path / "package"),
        config,
        source=str(source),
    )
    serialized = Path(path).read_text(encoding="utf-8")
    assert "&id" not in serialized
    assert "*id" not in serialized
    with open(path, encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)["pipeline"]["workflow"]
    decoder_invokes = []
    policy_invokes = {}

    def collect_decoder_invokes(node):
        if isinstance(node, dict):
            if node.get("kind") == "invoke" and node.get("component") == "decoder":
                decoder_invokes.append(node)
            elif node.get("kind") == "invoke":
                policy_invokes[node["component"]] = node
            for value in node.values():
                collect_decoder_invokes(value)
        elif isinstance(node, list):
            for value in node:
                collect_decoder_invokes(value)

    collect_decoder_invokes(workflow["steps"])
    assert len(decoder.graph.inputs) == 106
    assert len(decoder.graph.outputs) == 105
    assert len(decoder_invokes) == 2
    assert all(
        set(invoke["inputs"]) == {value.name for value in decoder.graph.inputs}
        for invoke in decoder_invokes
    )
    assert all("position_ids" not in invoke["inputs"] for invoke in decoder_invokes)
    assert (
        len([name for name in workflow["state"] if name.removeprefix("cache_").isdigit()])
        == 104
    )
    assert workflow["inputs"]["package.max_context"]["default"] == 131072
    assert workflow["inputs"]["package.eos_ids"]["default"] == 200001
    assert workflow["state"]["cache_103"]["contract"] == {
        "dtype": "bfloat16",
        "rank": 4,
        "shape": ["batch", 2, "past_sequence", 128],
    }
    assert workflow["state"]["logits"]["contract"] == {
        "dtype": "float32",
        "rank": 2,
        "shape": ["batch", 202048],
    }
    assert workflow["state"]["attention_mask"]["recurrence"] == {"kind": "invariant"}
    assert workflow["state"]["cache_103"]["recurrence"] == {
        "kind": "bounded",
        "axis": 2,
        "max": "package.max_context",
    }
    assert workflow["state"]["cache_lengths"]["initializer"] == "initializer.cache_lengths"
    assert policy_invokes["decoder_state_initializer"]["inputs"] == {
        "prompt_tokens": "request.prompt_tokens",
        "max_iterations": "request.max_iterations",
    }
    assert policy_invokes["decoder_step_update"]["inputs"]["logical_length"] == "cache_lengths"
    assert workflow["state"]["attention_mask"]["initializer"] == (
        "initializer.attention_mask"
    )
    assert any(
        invoke["inputs"]["attention_mask"] == "decoder_step.body_attention_mask"
        for invoke in decoder_invokes
    )
    assert workflow["inputs"]["request.image"]["required"] is False
    assert workflow["inputs"]["request.image"]["present_as"] == "request.image_present"
    assert "request.has_media" not in workflow["inputs"]
    assert "input_presence" in workflow["manifest"]["capabilities"]
    assert workflow["steps"][0]["termination"] == "generation_eos"
    media_branch = workflow["steps"][0]["setup"][0]
    assert media_branch["kind"] == "branch"
    assert media_branch["predicate"] == "request.image_present"
    assert set(media_branch["cases"]) == {"true", "false"}
    assert media_branch["cases"]["false"]["component"] == "empty_image_features"
    kv_service = workflow["serving"]["kv_service"]
    assert kv_service["paging"] == "none"
    assert kv_service["compaction"] is False
    decoder_cache = kv_service["groups"]["decoder_cache"]
    # Shared buffering is expressed by the admitted cache ports and runtime I/O
    # binding, even when the graph has no node-level share-buffer attribute.
    assert all("past_present_share_buffer" not in node.attributes for node in decoder.graph)
    assert decoder_cache["storage"] == "shared_buffer"
    kv_ports = decoder_cache["ports"]["decoder"]
    assert len(kv_ports) == 104
    assert kv_ports["cache_103"] == {
        "input": "past_key_values.51.value",
        "output": "present.51.value",
    }
    assert (tmp_path / "package" / "policies" / "last_token_logits.onnx").is_file()
    assert (tmp_path / "package" / "policies" / "empty_image_features.onnx").is_file()


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
    assert workflow["state"]["tokens_state"]["recurrence"] == {"kind": "invariant"}
    assert workflow["state"]["tokens_state"]["contract"]["shape"] == [
        "batch",
        "sequence",
    ]
    assert workflow["state"]["rng_offset"]["contract"]["shape"] == ["batch"]
    assert workflow["components"]["masked_update"]["contract"] == {
        "id": "onnx-genai.masked-update",
        "version": "1",
        "bindings": {
            "state": "current_tokens",
            "proposal": "proposed_tokens",
            "mask": "masked",
            "step": "step",
            "next_state": "next_state",
            "next_mask": "next_mask",
            "seed": "seed",
            "offset": "offset",
            "next_offset": "next_offset",
            "continue": "continue",
        },
    }

    graph = workflow["steps"][0]
    assert graph["kind"] == "loop"
    assert graph["continue_when"] == "loop_0_active"
    assert graph["max_iterations"] == "request.max_iterations"
    assert [node["component"] for node in graph["setup"]] == ["model"]
    assert [node["kind"] for node in graph["steps"]] == [
        "invoke",
        "emit",
        "invoke",
    ]
    assert graph["iteration"]["value"] == "loop.iteration"
    assert graph["steps"][0]["inputs"]["total_steps"] == "package.num_steps"
    assert graph["steps"][1]["mode"] == "replace"


def test_language_diffusion_rejects_zero_steps():
    with pytest.raises(ValueError, match="num_inference_steps"):
        build_language_diffusion_pipeline_metadata(
            _masked_denoiser_package(),
            num_inference_steps=0,
        )


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


def _speculative_package(
    *,
    adaptive: bool = False,
    budget_dtype: ir.DataType = ir.DataType.INT64,
) -> ModelPackage:
    proposer_inputs = [_value("tokens", ir.DataType.INT64, ["batch", 4])]
    if adaptive:
        proposer_inputs.append(_value("proposal_budget", budget_dtype, ["batch"]))
    proposer = _graph_model(
        "proposer",
        proposer_inputs,
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


def test_adaptive_speculative_requires_int64_budget_port():
    with pytest.raises(ValueError, match="rank-1 proposer budget input"):
        build_speculative_workflow_metadata(
            _speculative_package(adaptive=True, budget_dtype=ir.DataType.INT32),
            adaptive_k_max=4,
        )


def test_speculative_grammar_and_adaptive_k_use_typed_state_contracts():
    metadata = build_speculative_workflow_metadata(
        _speculative_package(adaptive=True),
        grammar_guidance=True,
        adaptive_k_max=4,
    )
    workflow = metadata["pipeline"]["workflow"]
    assert workflow["components"]["grammar_commit"]["contract"] == {
        "id": "onnx-genai.grammar-guidance",
        "version": "1",
        "bindings": {
            "state": "state",
            "tokens": "tokens",
            "valid_length": "valid_length",
            "transition_table": "transition_table",
            "next_state": "next_state",
            "consumed_length": "consumed_length",
            "logits_mask": "logits_mask",
            "forced_tokens": "forced_tokens",
            "forced_length": "forced_length",
        },
        "parameters": {"action": "commit"},
    }
    assert workflow["components"]["adaptive_k"]["contract"]["id"] == (
        "onnx-genai.adaptive-proposal-budget"
    )
    assert workflow["state"]["grammar"]["class"] == "semantic"
    assert workflow["state"]["proposal_k"]["class"] == "advisory"
    assert workflow["state"]["adaptive_estimates"]["class"] == "advisory"
    assert all(
        "ports" not in component and "effects" not in component
        for component in workflow["components"].values()
        if component["implementation"]["kind"] == "onnx"
    )
    assert "ports" in workflow["components"]["grammar_commit"]
    proposer = next(
        node for node in workflow["steps"][0]["steps"] if node.get("component") == "proposer"
    )
    assert proposer["inputs"]["proposal_budget"] == "proposal_k"
    grammar_emit = next(
        node
        for node in workflow["steps"][0]["steps"]
        if node.get("kind") == "emit" and node.get("value") == "grammar.token"
    )
    assert grammar_emit["valid_length"] == "grammar.forced_length"
    assert all("initial" not in carry for carry in workflow["steps"][0]["carried"])


def test_speculative_workflow_uses_per_row_ragged_state_and_rng():
    workflow = build_speculative_workflow_metadata(_speculative_package())["pipeline"][
        "workflow"
    ]
    body = workflow["steps"][0]["steps"]
    acceptance = body[2]
    assert acceptance["inputs"]["offset"] == "rng_offset"
    assert acceptance["outputs"]["next_offset"] == "rng_offset.body"
    assert acceptance["outputs"]["accepted_len"] == "acceptance.length"
    emit = next(node for node in body if node["kind"] == "emit")
    assert emit["valid_length"] == "acceptance.length"
    assert not any(node["kind"] == "branch" for node in body)
    assert workflow["serving"]["active"] == "active"
    assert workflow["serving"]["done"] == "done"
    assert workflow["serving"]["kv_service"]["groups"]["verifier_cache"]["ports"]["verifier"][
        "cache_0"
    ] == {
        "input": "past_key_values.0.key",
        "output": "present.0.key",
    }
    assert any(item["cell"].startswith("cache_") for item in workflow["steps"][0]["carried"])
