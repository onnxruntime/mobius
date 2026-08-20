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
    _kv_storage_contract,
    _static_cache_ports,
    build_language_diffusion_pipeline_metadata,
    build_speculative_workflow_metadata,
    build_video_diffusion_workflow_metadata,
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
    # Row identity is runtime-private: the published emit step must not name it.
    assert "row_ids" not in emit
    assert "emit_valid_length" in workflow["manifest"]["capabilities"]
    assert "emit_row_identity" not in workflow["manifest"]["capabilities"]
    assert not any("slot_ids" in name for name in workflow["inputs"])
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


def _decoder_with_capacity_addressable_attention(
    inputs: list[ir.Value],
    output_specs: list[tuple[str, ir.DataType, list[int | str]]],
) -> ir.Model:
    """Build a decoder whose only cache consumer is ``GroupQueryAttention``.

    Capacity-preallocated KV storage is a property of the attention operator,
    not of the port names: only an operator that takes the logical cache length
    as a separate input can ignore the unwritten tail of a capacity-sized
    buffer. A decoder wired through plain ``Identity`` nodes therefore describes
    a *dynamic* cache, so an artifact meant to stand in for a shared-buffer
    decoder has to name the operator that makes sharing sound.
    """
    outputs = [_value(*spec) for spec in output_specs]
    by_name = {output.name: output for output in outputs}
    logits = outputs[0]
    past_by_name = {value.name: value for value in inputs}
    nodes = [ir.Node("", "Identity", [inputs[0]], outputs=[logits], name="emit_logits")]
    layer = 0
    while f"past_key_values.{layer}.key" in past_by_name:
        nodes.append(
            ir.Node(
                "com.microsoft",
                "GroupQueryAttention",
                [
                    inputs[0],
                    past_by_name[f"past_key_values.{layer}.key"],
                    past_by_name[f"past_key_values.{layer}.value"],
                ],
                outputs=[
                    by_name[f"present.{layer}.key"],
                    by_name[f"present.{layer}.value"],
                ],
                name=f"attention_{layer}",
            )
        )
        layer += 1
    graph = ir.Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        name="decoder",
        opset_imports={"": 21, "com.microsoft": 1},
    )
    return ir.Model(graph, ir_version=10)


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
    decoder = _decoder_with_capacity_addressable_attention(decoder_inputs, decoder_outputs)
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
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    }
    assert workflow["state"]["logits"]["contract"] == {
        "dtype": "float32",
        "rank": 2,
        "shape": ["batch", 202048],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    }
    # A capacity-preallocated cache binds a mask sized once, up front.
    assert workflow["state"]["attention_mask"]["recurrence"] == {"kind": "invariant"}
    assert workflow["state"]["cache_103"]["recurrence"] == {
        "kind": "bounded",
        "axis": 2,
        "max": "package.max_context",
    }
    assert workflow["state"]["cache_lengths"]["initializer"] == "initializer.cache_lengths"
    assert policy_invokes["decoder_state_initializer"]["inputs"] == {
        "prompt_tokens": "request.prompt_tokens",
        "prompt_lengths": "request.prompt_lengths",
        "max_iterations": "request.max_iterations",
    }
    assert workflow["inputs"]["request.prompt_lengths"]["contract"]["shape"] == ["batch"]
    assert workflow["inputs"]["request.max_iterations"]["contract"]["shape"] == [1]
    assert workflow["steps"][0]["iteration"]["contract"] == {
        "dtype": "int64",
        "rank": 1,
        "shape": [1],
    }
    assert workflow["inputs"]["package.one"]["contract"]["shape"] == ["batch"]
    assert workflow["state"]["generated_lengths"]["initializer"] == (
        "initializer.generated_lengths"
    )
    assert policy_invokes["token_sampler"]["inputs"]["active"] == "active"
    assert policy_invokes["token_sampler"]["inputs"]["done"] == "done"
    assert set(policy_invokes["token_state_update"]["inputs"]) == {
        "current",
        "update",
        "active",
        "done",
    }
    assert policy_invokes["token_state_update"]["outputs"] == {"next": "token.body"}
    assert policy_invokes["generated_length_update"]["outputs"] == {
        "total": "token.emitted_length"
    }

    def collect_emits(node):
        if isinstance(node, dict):
            return ([node] if node.get("kind") == "emit" else []) + [
                emit for value in node.values() for emit in collect_emits(value)
            ]
        if isinstance(node, list):
            return [emit for value in node for emit in collect_emits(value)]
        return []

    emit = next(
        node for node in collect_emits(workflow["steps"]) if node["output"] == "tokens"
    )
    assert emit["when"] == "active"
    assert emit["valid_length"] == "token.emitted_length"
    assert "row_ids" not in emit
    assert "emit_row_identity" not in workflow["manifest"]["capabilities"]
    assert policy_invokes["decoder_step_update"]["inputs"]["logical_length"] == "cache_lengths"
    assert workflow["state"]["attention_mask"]["initializer"] == ("initializer.attention_mask")
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
    state_service = workflow["serving"]["state_service"]
    # Storage class, allocator, and compaction algorithm are the runtime's to
    # choose; the package only declares semantics.
    assert set(state_service) == {"groups"}
    decoder_group = state_service["groups"]["decoder_cache"]
    assert decoder_group["kind"] == "full_attention"
    assert decoder_group["aliasing"] == "permitted"
    assert decoder_group["reuse"] == {
        "prefix_reusable": True,
        "evictable_prefix": False,
    }
    assert "storage" not in decoder_group
    carried = {item["cell"] for item in workflow["steps"][0]["carried"]}
    assert not any("slot_ids" in cell for cell in carried)
    assert {
        "token",
        "logits",
        "generated_lengths",
        "rng_counter",
        "active",
        "done",
        "accepted_len",
        "cache_lengths",
        "attention_mask",
        "cache_0",
    } <= carried
    decoder_cache = state_service["groups"]["decoder_cache"]
    # Shared buffering is expressed by the admitted cache ports and runtime I/O
    # binding, even when the graph has no node-level share-buffer attribute.
    assert all("past_present_share_buffer" not in node.attributes for node in decoder.graph)
    kv_ports = decoder_cache["ports"]["decoder"]
    assert len(kv_ports) == 104
    assert kv_ports["cache_103"] == {
        "input": "past_key_values.51.value",
        "output": "present.51.value",
        "role": "value",
        "layer": 51,
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


def _video_package(*, cache_ports: bool = True, latent_rank: int = 5) -> ModelPackage:
    latent_shape: list[int | str] = ["batch", "num_frames", 4, "height", "width"]
    if latent_rank == 4:
        latent_shape = ["batch", 4, "height", "width"]
    sample = _value("sample", ir.DataType.FLOAT, latent_shape)
    timestep = _value("timestep", ir.DataType.INT64, ["batch"])
    conditioning = _value(
        "encoder_hidden_states", ir.DataType.FLOAT, ["batch", "prompt_sequence", 32]
    )
    noise_pred = _value("noise_pred", ir.DataType.FLOAT, latent_shape)
    denoiser = ir.Graph(
        inputs=[sample, timestep, conditioning],
        outputs=[noise_pred],
        nodes=[],
        name="transformer",
        opset_imports={"": 24},
    )

    latent_sample = _value(
        "latent_sample",
        ir.DataType.FLOAT,
        ["batch", 4, "latent_frames", "latent_height", "latent_width"],
    )
    frames = _value(
        "sample",
        ir.DataType.FLOAT,
        ["batch", 3, "frames", "2*latent_height", "2*latent_width"],
    )
    vae_inputs = [latent_sample]
    vae_outputs = [frames]
    if cache_ports:
        vae_inputs.append(
            _value(
                "conv_cache.conv_in",
                ir.DataType.FLOAT,
                ["batch", 4, "cache_frames", "latent_height", "latent_width"],
            )
        )
        vae_outputs.append(
            _value(
                "conv_cache_out.conv_in",
                ir.DataType.FLOAT,
                ["batch", 4, "cache_frames", "latent_height", "latent_width"],
            )
        )
    vae = ir.Graph(
        inputs=vae_inputs,
        outputs=vae_outputs,
        nodes=[],
        name="vae_decoder",
        opset_imports={"": 24},
    )
    vae_model = ir.Model(vae, ir_version=11)
    vae_model.metadata_props["mobius.conv_cache.spatial_scale.conv_cache.conv_in"] = "1"
    return ModelPackage(
        {
            "transformer": ir.Model(denoiser, ir_version=11),
            "vae_decoder": vae_model,
        }
    )


def test_video_diffusion_keeps_the_temporal_axis_through_every_stage():
    metadata = build_video_diffusion_workflow_metadata(
        _video_package(), num_inference_steps=2, solver="ddim"
    )
    workflow = metadata["pipeline"]["workflow"]

    published = workflow["outputs"]["video"]
    assert published["role"] == "video"
    assert published["contract"]["rank"] == 5

    latent = workflow["state"]["latent"]
    assert latent["contract"]["rank"] == 5
    assert latent["contract"]["shape"][1] == "num_frames"

    # The scheduler trajectory is state, not telemetry: a multistep video solver
    # reads it back.
    history = workflow["state"]["scheduler_history"]
    assert history["recurrence"]["kind"] == "growing"
    assert history["recurrence"]["axis"] == 1

    cache = workflow["state"]["conv_cache_conv_in"]
    assert cache["management"] == "runtime"
    assert cache["release_boundary"] == "invocation"
    assert cache["recurrence"] == {
        "kind": "bounded",
        "axis": 2,
        "max": "package.cache_frames",
    }

    denoise, decode = (step for step in workflow["steps"] if step["kind"] == "loop")
    assert denoise["max_iterations"] == "request.max_iterations"
    # The decoder runs once per causal chunk, and the chunk count is computed at
    # run time from the latent rather than fixed by the package.
    assert decode["max_iterations"] == "decode.chunks"
    emit = next(node for node in decode["steps"] if node["kind"] == "emit")
    # Frames append along time. Appending on the last axis -- the token-sequence
    # default -- would concatenate image columns instead.
    assert emit["mode"] == "append"
    assert emit["axis"] == 2

    assert "bounded_state_recurrence" in workflow["manifest"]["capabilities"]


def test_video_diffusion_rejects_an_image_latent():
    with pytest.raises(ValueError, match="rank-5"):
        build_video_diffusion_workflow_metadata(
            _video_package(latent_rank=4), num_inference_steps=2
        )


def test_video_diffusion_requires_paired_conv_caches():
    with pytest.raises(ValueError, match="conv_cache"):
        build_video_diffusion_workflow_metadata(
            _video_package(cache_ports=False), num_inference_steps=2
        )


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
        "ports" in component and "effects" not in component
        for component in workflow["components"].values()
        if component["implementation"]["kind"] == "onnx" and "contract" in component
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
    assert "row_ids" not in emit
    assert not any(node["kind"] == "branch" for node in body)
    assert workflow["serving"]["active"] == "active"
    assert workflow["serving"]["done"] == "done"
    assert workflow["serving"]["state_service"]["groups"]["verifier_cache"]["kind"] == (
        "full_attention"
    )
    assert "slot_ids" not in workflow["serving"]
    assert workflow["serving"]["state_service"]["groups"]["verifier_cache"]["ports"][
        "verifier"
    ]["cache_0"] == {
        "input": "past_key_values.0.key",
        "output": "present.0.key",
        "role": "key",
        "layer": 0,
    }
    assert any(item["cell"].startswith("cache_") for item in workflow["steps"][0]["carried"])


def _decoder_with_cache(domain: str, op_type: str) -> ir.Model:
    """Minimal decoder whose KV cache is consumed by ``domain::op_type``."""
    past_key = _value("past_key_values.0.key", ir.DataType.FLOAT, ["batch", 2, "past", 8])
    past_value = _value("past_key_values.0.value", ir.DataType.FLOAT, ["batch", 2, "past", 8])
    hidden = _value("hidden", ir.DataType.FLOAT, ["batch", "seq", 16])
    present_key = _value("present.0.key", ir.DataType.FLOAT, ["batch", 2, "total", 8])
    present_value = _value("present.0.value", ir.DataType.FLOAT, ["batch", 2, "total", 8])
    logits = _value("logits", ir.DataType.FLOAT, ["batch", "seq", 32])
    attention = ir.Node(
        domain,
        op_type,
        [hidden, past_key, past_value],
        outputs=[logits, present_key, present_value],
        name="attention",
    )
    graph = ir.Graph(
        inputs=[hidden, past_key, past_value],
        outputs=[logits, present_key, present_value],
        nodes=[attention],
        name="decoder",
        opset_imports={"": 21, "com.microsoft": 1},
    )
    return ir.Model(graph, ir_version=10)


def test_capacity_addressable_attention_declares_shared_buffer_storage():
    # GroupQueryAttention takes seqlens_k/total_sequence_length, so it can safely
    # read a capacity-sized buffer whose tail is unwritten.
    contract = _kv_storage_contract(
        _decoder_with_cache("com.microsoft", "GroupQueryAttention")
    )
    assert contract == {"paging": "none", "compaction": True, "storage": "shared_buffer"}


def test_standard_attention_declares_dynamic_storage():
    # The standard ONNX Attention operator derives the total sequence length from
    # the past tensor's own shape, so a preallocated buffer would both attend over
    # unwritten slots and contradict an exactly sized attention mask.
    contract = _kv_storage_contract(_decoder_with_cache("", "Attention"))
    assert contract == {"paging": "none", "compaction": True, "storage": "dynamic"}


def test_unconsumed_cache_ports_are_not_treated_as_capacity_addressable():
    # A graph that never reads its own cache cannot promise capacity-safe
    # behaviour it does not exercise.
    model = _decoder_with_cache("com.microsoft", "GroupQueryAttention")
    model.graph.node("attention").replace_input_with(1, None)
    model.graph.node("attention").replace_input_with(2, None)
    assert _kv_storage_contract(model)["storage"] == "dynamic"


def test_paged_cache_inputs_take_precedence_over_operator_derivation():
    model = _decoder_with_cache("", "Attention")
    model.graph.inputs.append(_value("block_tables", ir.DataType.INT32, ["batch", "blocks"]))
    contract = _kv_storage_contract(model)
    assert contract["paging"] == "paged"
    assert contract["storage"] == "paged"


def _static_cache_model(
    *,
    capacities: list[int] | None = None,
    scatter_axis: int | None = None,
    paired: bool = True,
    control_ports: bool = True,
) -> ir.Model:
    """A minimal graph shaped like a mobius static-cache decoder export."""
    capacities = capacities or [32, 32]
    inputs = [_value("input_ids", ir.DataType.INT64, ["batch", "sequence"])]
    for layer, capacity in enumerate(capacities):
        inputs.append(_value(f"key_cache.{layer}", ir.DataType.FLOAT, ["batch", capacity, 16]))
    if control_ports:
        inputs.append(_value("write_indices", ir.DataType.INT64, ["batch"]))
        inputs.append(_value("nonpad_kv_seqlen", ir.DataType.INT64, ["batch"]))
    outputs: list[tuple[str, ir.DataType, list[int | str]]] = [
        ("logits", ir.DataType.FLOAT, ["batch", "sequence", 128])
    ]
    if paired:
        for layer, capacity in enumerate(capacities):
            outputs.append(
                (f"updated_key_cache.{layer}", ir.DataType.FLOAT, ["batch", capacity, 16])
            )
    model = _model("decoder", inputs, outputs)
    if scatter_axis is not None:
        cache = model.graph.inputs[1]
        scattered = _value("scattered", ir.DataType.FLOAT, ["batch", capacities[0], 16])
        model.graph.append(
            ir.Node(
                "",
                "TensorScatter",
                [cache, cache, model.graph.inputs[-2]],
                outputs=[scattered],
                attributes=[ir.AttrInt64("axis", scatter_axis)],
                name="scatter",
            )
        )
    return model


class TestStaticCachePortDiscovery:
    """``_static_cache_ports`` reads the ABI from the graph or refuses to guess."""

    def test_returns_none_without_the_control_ports(self):
        # A dynamic decoder must not be mistaken for a fixed-capacity one.
        assert _static_cache_ports(_static_cache_model(control_ports=False)) is None

    def test_discovers_buffers_control_ports_and_capacity(self):
        ports = _static_cache_ports(_static_cache_model())
        assert ports["write_indices"] == "write_indices"
        assert ports["kv_sequence_length"] == "nonpad_kv_seqlen"
        assert ports["capacity"] == 32
        assert sorted(ports["buffers"]) == ["key_cache.0", "key_cache.1"]

    def test_rejects_control_ports_without_paired_buffers(self):
        # Nothing to scatter into: the runtime would have no output to carry.
        with pytest.raises(ValueError, match="no paired cache buffer"):
            _static_cache_ports(_static_cache_model(paired=False))

    def test_rejects_conflicting_capacities(self):
        # One write cursor cannot address buffers of different lengths.
        with pytest.raises(ValueError, match="conflicting capacities"):
            _static_cache_ports(_static_cache_model(capacities=[32, 64]))

    def test_rejects_a_symbolic_capacity(self):
        model = _static_cache_model()
        model.graph.inputs[1].shape = ir.Shape(["batch", "capacity", 16])
        with pytest.raises(ValueError, match="symbolic extent"):
            _static_cache_ports(model)

    def test_rejects_a_scatter_that_disagrees_with_the_declared_axis(self):
        # The capacity axis published in the metadata has to be the axis the
        # graph actually writes on, or a runtime sizes the wrong dimension.
        with pytest.raises(ValueError, match="declared capacity axis"):
            _static_cache_ports(_static_cache_model(scatter_axis=2))

    def test_accepts_a_scatter_on_the_declared_axis(self):
        assert _static_cache_ports(_static_cache_model(scatter_axis=1))["capacity"] == 32
