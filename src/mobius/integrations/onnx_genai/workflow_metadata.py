# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ONNX GenAI workflow-IR metadata production."""

from __future__ import annotations

import os
from typing import Any

import onnx_ir as ir
import yaml

from mobius._constants import OPSET_VERSION
from mobius.generation import (
    PolicyCapabilities,
    attach_policy_components,
    build_batch_minimum,
    build_boolean_not,
    build_code_frame_update,
    build_code_history_append,
    build_codec_layout_transpose,
    build_decoder_state_initializer,
    build_decoder_step_update,
    build_effectful_identity,
    build_euler_model_input,
    build_euler_solver_step,
    build_greedy_sampler,
    build_integer_increment,
    build_integer_minimum,
    build_last_token_logits,
    build_model_token_cast,
    build_proposal_metrics,
    build_schedule_constant,
    build_schedule_lookup,
    build_sequence_length,
    build_speculative_state_rollback,
    build_token_block_identity,
    build_token_to_slot,
    build_tts_decoder_state_initializer,
    build_tts_decoder_step_update,
    build_tts_state_initializer,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    _port,
    _shape_metadata,
    add_policy_components_to_workflow,
    build_native_vlm_package_metadata,
)


def _contract(value: ir.Value) -> dict[str, Any]:
    port = _port(value)
    dtype = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(
        port.dtype, port.dtype
    )
    return {
        "dtype": dtype,
        "rank": port.rank,
        "shape": _shape_metadata(port),
    }


def _component(
    model: ir.Model,
    artifact: str,
    *,
    effects: tuple[str, ...] = (),
) -> dict[str, Any]:
    del model, effects
    return {"implementation": {"kind": "onnx", "artifact": artifact}}


def _grammar_adapter_component(action: str) -> dict[str, Any]:
    """Declare one action of the versioned grammar-guidance adapter ABI."""

    def port(dtype: str, shape: list[int | str]) -> dict[str, Any]:
        return {"dtype": dtype, "rank": len(shape), "shape": shape}

    return {
        "implementation": {
            "kind": "adapter",
            "abi": "onnx-genai.grammar-guidance",
            "version": "1",
        },
        "ports": {
            "inputs": {
                "state": port("int64", ["batch"]),
                "tokens": port("int64", ["batch", "proposal"]),
                "valid_length": port("int64", ["batch"]),
                "transition_table": port("int64", ["grammar_states", "vocabulary"]),
            },
            "outputs": {
                "next_state": port("int64", ["batch"]),
                "consumed_length": port("int64", ["batch"]),
                "logits_mask": port("bool", ["batch", "vocabulary"]),
                "forced_tokens": port("int64", ["batch", 1]),
                "forced_length": port("int64", ["batch"]),
            },
        },
        "contract": {
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
            "parameters": {"action": action},
        },
        "effects": ["grammar"],
    }


def _effect(consumes: str, produces: str) -> dict[str, str]:
    return {"consumes": consumes, "produces": produces}


def _publish_workflow_v1(workflow: dict[str, Any]) -> dict[str, Any]:
    """Lower the producer's explicit-effect graph into the public v1 step IR."""
    graph = workflow.pop("graph")
    workflow.pop("initial_effects", None)
    substitutions: dict[str, str] = {}
    cell_aliases = {
        cell: f"{cell}_state" if cell in workflow.get("outputs", {}) else cell
        for cell in workflow.get("state", {})
    }
    if any(cell != alias for cell, alias in cell_aliases.items()):
        workflow["state"] = {
            cell_aliases[cell]: declaration for cell, declaration in workflow["state"].items()
        }

    def collect_carried(node: dict[str, Any]) -> None:
        if node["kind"] == "loop":
            for carry in node.get("carried", []):
                alias = cell_aliases.get(carry["cell"], carry["cell"])
                substitutions[carry["body_input"]] = alias
                substitutions[carry["next"]] = alias
            collect_carried(node["setup"])
            collect_carried(node["body"])
        elif node["kind"] == "sequence":
            for child in node["nodes"]:
                collect_carried(child)
        elif node["kind"] == "branch":
            for case in node["cases"].values():
                collect_carried(case)
            if "default" in node:
                collect_carried(node["default"])

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            return substitutions.get(value, value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    def convert(node: dict[str, Any]) -> dict[str, Any]:
        kind = node["kind"]
        if kind == "sequence":
            return {
                "kind": "sequence",
                "steps": [convert(child) for child in node["nodes"]],
            }
        if kind == "invoke":
            return {
                "kind": "invoke",
                "component": node["component"],
                "inputs": rewrite(node.get("inputs", {})),
                "outputs": rewrite(node.get("outputs", {})),
            }
        if kind == "emit":
            result = {
                "kind": "emit",
                "value": rewrite(node["value"]),
                "output": node["output"],
                "mode": node["mode"],
            }
            if "valid_length" in node:
                result["valid_length"] = rewrite(node["valid_length"])
            return result
        if kind == "branch":
            result = {
                "kind": "branch",
                "predicate": rewrite(node["predicate"]),
                "cases": {name: convert(case) for name, case in node["cases"].items()},
                "outputs": rewrite(node.get("outputs", {})),
            }
            if "default" in node:
                result["default"] = convert(node["default"])
            return result
        if kind == "loop":
            setup = node["setup"]
            body = node["body"]
            setup_steps = (
                [convert(child) for child in setup["nodes"]]
                if setup["kind"] == "sequence"
                else [convert(setup)]
            )
            body_steps = (
                [convert(child) for child in body["nodes"]]
                if body["kind"] == "sequence"
                else [convert(body)]
            )
            carried = []
            for carry in node.get("carried", []):
                cell = cell_aliases.get(carry["cell"], carry["cell"])
                published_carry = {
                    "cell": cell,
                    "next": rewrite(carry["body_output"]),
                }
                initial = rewrite(carry["current"])
                if workflow["state"][cell]["initializer"] != initial:
                    published_carry["initial"] = initial
                carried.append(published_carry)
            result = {
                "kind": "loop",
                "setup": setup_steps,
                "steps": body_steps,
                "condition": rewrite(node["condition"]),
                "max_iterations": rewrite(node["max_iterations"]),
                "carried": carried,
            }
            if "iteration" in node:
                result["iteration"] = node["iteration"]
            return result
        raise ValueError(f"unsupported workflow node kind {kind!r}")

    collect_carried(graph)
    published = convert(graph)
    workflow["steps"] = published["steps"] if published["kind"] == "sequence" else [published]
    return workflow


def _name_image_preprocessing_program(image: dict[str, Any]) -> None:
    """Convert structural preprocessing transforms into explicit typed SSA values."""
    transforms = image["transforms"]
    current: str | None = None
    decoded: str | None = None
    for index, transform in enumerate(transforms):
        name = f"image.transform_{index}"
        if transform["op"] in {"decode", "decode_rgb"}:
            transform.pop("inputs", None)
            decoded = name
        else:
            if current is None:
                raise ValueError("image preprocessing must decode before transforming")
            transform["inputs"] = [current]
        transform["outputs"] = [name]
        current = name
    if current is None:
        raise ValueError("image preprocessing must declare at least one transform")

    derived_ops = {
        "original_size": ("emit_original_size", decoded),
        "transformed_size": ("emit_transformed_size", current),
        "validity_mask": ("emit_validity_mask", current),
        "patch_coordinates": ("emit_patch_coordinates", current),
        "grid_dimensions": ("emit_grid_coordinates", current),
    }
    for output in image["outputs"]:
        content = output["content"]
        if content == "pixels":
            output["source"] = current
            continue
        if content not in derived_ops:
            raise ValueError(
                f"image preprocessing output content {content!r} has no typed SSA producer"
            )
        operation, source = derived_ops[content]
        if source is None:
            raise ValueError(
                f"image preprocessing output content {content!r} requires a decoded image"
            )
        name = f"image.output_{content}"
        transforms.append(
            {
                "op": operation,
                "inputs": [source],
                "outputs": [name],
            }
        )
        output["source"] = name


def _invoke(
    component: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    effects: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "invoke",
        "component": component,
        "inputs": inputs,
        "outputs": outputs,
        "effects": effects or {},
    }


def build_audio_codec_workflow_metadata(pkg: Any) -> dict[str, Any]:
    """Build typed SSA metadata for a waveform-to-codes-to-waveform codec."""
    names = set(pkg.keys())
    if names != {"encoder", "decoder"}:
        raise ValueError(
            "audio codec workflow requires exactly encoder and decoder components"
        )
    encoder = pkg["encoder"]
    decoder = pkg["decoder"]
    if len(encoder.graph.inputs) != 1 or len(encoder.graph.outputs) != 1:
        raise ValueError("codec encoder requires exactly one input and one output")
    if len(decoder.graph.inputs) != 1 or len(decoder.graph.outputs) != 1:
        raise ValueError("codec decoder requires exactly one input and one output")

    waveform_input = encoder.graph.inputs[0]
    codes_output = encoder.graph.outputs[0]
    codes_input = decoder.graph.inputs[0]
    waveform_output = decoder.graph.outputs[0]
    if codes_output.dtype != codes_input.dtype:
        raise ValueError("codec encoder output and decoder input dtypes must match")
    if _contract(codes_output) != _contract(codes_input):
        raise ValueError("codec encoder output and decoder input contracts must match")

    encode_effect = "codec_encode"
    decode_effect = "codec_decode"
    emit_effect = "audio_emit"
    workflow = {
        "manifest": {
            "ir_version": "1.0",
            "onnx_opsets": {"ai.onnx": OPSET_VERSION},
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "typed_emit",
            ],
        },
        "inputs": {
            "request.waveform": {
                "contract": _contract(waveform_input),
                "role": {"kind": "runtime", "version": "1.0", "role": "media"},
                "source": {"kind": "request", "field": "media"},
                "required": True,
            }
        },
        "outputs": {
            "waveform": {
                "contract": _contract(waveform_output),
                "role": "audio",
                "stage": "post_adapter",
            }
        },
        "components": {
            "encoder": _component(
                encoder,
                "encoder/model.onnx",
                effects=(encode_effect,),
            ),
            "decoder": _component(
                decoder,
                "decoder/model.onnx",
                effects=(decode_effect,),
            ),
        },
        "initial_effects": {
            encode_effect: f"{encode_effect}.0",
            decode_effect: f"{decode_effect}.0",
            emit_effect: f"{emit_effect}.0",
        },
        "graph": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "encoder",
                    {waveform_input.name: "request.waveform"},
                    {codes_output.name: "codec.codes"},
                    {encode_effect: _effect(f"{encode_effect}.0", f"{encode_effect}.1")},
                ),
                _invoke(
                    "decoder",
                    {codes_input.name: "codec.codes"},
                    {waveform_output.name: "codec.waveform"},
                    {decode_effect: _effect(f"{decode_effect}.0", f"{decode_effect}.1")},
                ),
                {
                    "kind": "emit",
                    "value": "codec.waveform",
                    "output": "waveform",
                    "mode": "replace",
                    "effect_name": emit_effect,
                    "effect": _effect(f"{emit_effect}.0", f"{emit_effect}.1"),
                },
            ],
        },
    }
    return {"schema_version": "v1", "pipeline": {"workflow": _publish_workflow_v1(workflow)}}


def write_audio_codec_workflow_metadata(pkg: Any, output_dir: str) -> str:
    """Write typed SSA metadata for an audio codec package."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_audio_codec_workflow_metadata(pkg)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def _model_cache_pairs(model: ir.Model) -> list[tuple[ir.Value, ir.Value]]:
    outputs = {value.name: value for value in model.graph.outputs}
    pairs = []
    for past in model.graph.inputs:
        present = next(
            (
                outputs.get(name)
                for name in (
                    past.name.replace("past_key_values", "present"),
                    past.name.replace("past.", "present."),
                )
                if name in outputs
            ),
            None,
        )
        if present is not None:
            pairs.append((past, present))
    return pairs


def _build_real_tts_workflow_metadata(pkg: Any, config: Any) -> dict[str, Any]:
    """Build the weight-bearing Qwen3-TTS talker/predictor/codec workflow."""
    talker = pkg["talker"]
    predictor = pkg["code_predictor"]
    prefill_embedder = pkg["talker_prefill_embedder"]
    predictor_indices = pkg["code_predictor_indices"]
    codec_name = next(
        (name for name in ("codec", "vocoder", "decoder") if name in pkg),
        None,
    )
    if codec_name is None:
        raise ValueError("real TTS workflow requires a merged codec/vocoder decoder component")
    codec = pkg[codec_name]
    num_groups = int(getattr(getattr(config, "tts", config), "num_code_groups", 16))
    if num_groups < 2:
        raise ValueError("real TTS workflow requires at least two code groups")

    prompt = next(iter(prefill_embedder.graph.inputs))
    prefill = _find_port(prefill_embedder.graph.outputs, "prefill")
    trailing = _find_port(prefill_embedder.graph.outputs, "trailing")
    talker_embed = _find_port(talker.graph.inputs, "inputs_embeds")
    talker_mask = _find_port(talker.graph.inputs, "attention_mask")
    talker_position = _find_port(talker.graph.inputs, "position_ids")
    talker_logits = _find_port(talker.graph.outputs, "logits")
    talker_hidden = _find_port(talker.graph.outputs, "hidden")
    predictor_embed = _find_port(predictor.graph.inputs, "inputs_embeds")
    predictor_step = _find_port(predictor.graph.inputs, "step_index")
    predictor_mask = _find_port(predictor.graph.inputs, "attention_mask")
    predictor_position = _find_port(predictor.graph.inputs, "position_ids")
    predictor_logits = _find_port(predictor.graph.outputs, "logits")
    codec_embeddings = _find_port(predictor.graph.outputs, "codec_embeddings")
    codec_input = next(iter(codec.graph.inputs), None)
    waveform = next(iter(codec.graph.outputs), None)
    required_ports = (
        prefill,
        trailing,
        talker_embed,
        talker_mask,
        talker_position,
        talker_logits,
        talker_hidden,
        predictor_embed,
        predictor_step,
        predictor_mask,
        predictor_position,
        predictor_logits,
        codec_embeddings,
        codec_input,
        waveform,
    )
    if any(value is None for value in required_ports):
        raise ValueError("real TTS package is missing a required typed transition port")
    assert prefill is not None and trailing is not None
    assert talker_embed is not None and talker_mask is not None
    assert talker_position is not None and talker_logits is not None
    assert talker_hidden is not None and predictor_embed is not None
    assert predictor_step is not None and predictor_mask is not None
    assert predictor_position is not None and predictor_logits is not None
    assert codec_embeddings is not None and codec_input is not None and waveform is not None

    talker_caches = _model_cache_pairs(talker)
    predictor_caches = _model_cache_pairs(predictor)
    attach_policy_components(pkg, PolicyCapabilities())
    pkg.add_policy_component("last_token_logits", build_last_token_logits())
    pkg.add_policy_component(
        "setup_talker_sampler", build_greedy_sampler(effect="setup_talker_sample")
    )
    pkg.add_policy_component(
        "setup_predictor_sampler",
        build_greedy_sampler(effect="setup_predictor_sample"),
    )
    pkg.add_policy_component("talker_sampler", build_greedy_sampler(effect="talker_sample"))
    pkg.add_policy_component(
        "predictor_prefill_sampler",
        build_greedy_sampler(effect="predictor_prefill_sample"),
    )
    pkg.add_policy_component(
        "predictor_body_sampler",
        build_greedy_sampler(effect="predictor_body_sample"),
    )
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    pkg.add_policy_component("tts_state_initializer", build_tts_state_initializer(num_groups))
    pkg.add_policy_component("token_to_slot", build_token_to_slot())
    pkg.add_policy_component(
        "code_frame_update", build_code_frame_update(num_groups, scalar_index=True)
    )
    pkg.add_policy_component("code_history_append", build_code_history_append(num_groups))
    pkg.add_policy_component(
        "talker_state_initializer",
        build_tts_decoder_state_initializer(
            talker,
            graph_name="talker_state_initializer",
            embedding_input=talker_embed.name,
            attention_mask_input=talker_mask.name,
            position_ids_input=talker_position.name,
            cache_inputs=[past.name for past, _ in talker_caches],
        ),
    )
    pkg.add_policy_component(
        "predictor_state_initializer",
        build_tts_decoder_state_initializer(
            predictor,
            graph_name="predictor_state_initializer",
            embedding_input=predictor_embed.name,
            attention_mask_input=predictor_mask.name,
            position_ids_input=predictor_position.name,
            cache_inputs=[past.name for past, _ in predictor_caches],
        ),
    )
    pkg.add_policy_component(
        "talker_step_update",
        build_tts_decoder_step_update(
            graph_name="talker_step_update",
            attention_dtype=talker_mask.dtype,
            position_dtype=talker_position.dtype,
            position_rank=len(talker_position.shape or []),
        ),
    )
    pkg.add_policy_component(
        "predictor_step_update",
        build_tts_decoder_step_update(
            graph_name="predictor_step_update",
            attention_dtype=predictor_mask.dtype,
            position_dtype=predictor_position.dtype,
            position_rank=len(predictor_position.shape or []),
        ),
    )
    codec_group_major = (
        codec_input.shape is not None
        and len(codec_input.shape) == 3
        and str(list(codec_input.shape)[1]) == str(num_groups)
    )
    if codec_group_major:
        pkg.add_policy_component("codec_layout", build_codec_layout_transpose(num_groups))

    batch = _contract(prompt)["shape"][0]
    batch_int = {"dtype": "int64", "rank": 1, "shape": [batch]}
    batch_bool = {"dtype": "bool", "rank": 1, "shape": [batch]}
    inputs = {
        "request.prompt_tokens": {
            "contract": _contract(prompt),
            "role": {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"},
            "source": {"kind": "request", "field": "prompt_tokens"},
            "required": True,
        },
        "request.max_iterations": {
            "contract": batch_int,
            "role": {"kind": "runtime", "version": "1.0", "role": "max_output_tokens"},
            "source": {"kind": "request", "field": "max_output_tokens"},
            "required": True,
        },
        "package.false": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": False,
        },
        "package.zero_scalar": {
            "contract": {"dtype": "int64", "rank": 0, "shape": []},
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0,
        },
        "package.one_scalar": {
            "contract": {"dtype": "int64", "rank": 0, "shape": []},
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.remaining_groups": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": num_groups - 2,
        },
        "package.predictor_context_limit": {
            "contract": {"dtype": "int64", "rank": 0, "shape": []},
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": num_groups,
        },
        "package.predictor_mask_limit": {
            "contract": {"dtype": "int64", "rank": 0, "shape": []},
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": num_groups + 1,
        },
        "package.talker_context_limit": {
            "contract": {"dtype": "int64", "rank": 0, "shape": []},
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": int(getattr(config, "max_position_embeddings", 4096)),
        },
    }
    for iteration in range(num_groups - 2):
        inputs[f"package.setup_predictor_iteration_{iteration}"] = {
            "contract": {"dtype": "int64", "rank": 0, "shape": []},
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": iteration,
        }

    talker_setup_inputs = {
        talker_embed.name: "tts.prefill_embeds",
        talker_mask.name: f"talker.initializer.{talker_mask.name}",
        talker_position.name: f"talker.initializer.{talker_position.name}",
        **{past.name: f"talker.initializer.{past.name}" for past, _ in talker_caches},
    }
    talker_body_inputs = {
        talker_embed.name: "talker.step_embeds",
        talker_mask.name: "state.talker_mask.body",
        talker_position.name: "state.talker_position.body",
        **{
            past.name: f"state.talker_cache_{i}.body"
            for i, (past, _) in enumerate(talker_caches)
        },
    }
    talker_setup_outputs = {
        talker_logits.name: "talker.setup.logits",
        talker_hidden.name: "talker.setup.hidden",
        **{present.name: f"talker.setup.{present.name}" for _, present in talker_caches},
    }
    talker_body_outputs = {
        talker_logits.name: "talker.body.logits",
        talker_hidden.name: "talker.body.hidden",
        **{present.name: f"talker.body.{present.name}" for _, present in talker_caches},
    }

    def predictor_outputs(prefix: str) -> dict[str, str]:
        return {
            predictor_logits.name: f"{prefix}.logits",
            codec_embeddings.name: f"{prefix}.codec_embeddings",
            **{present.name: f"{prefix}.{present.name}" for _, present in predictor_caches},
        }

    predictor_body_inputs = {
        predictor_embed.name: "predictor.body.inputs_embeds",
        predictor_step.name: "predictor.body.step_index",
        predictor_mask.name: "state.predictor_mask.inner",
        predictor_position.name: "state.predictor_position.inner",
        **{
            past.name: f"state.predictor_cache_{i}.inner"
            for i, (past, _) in enumerate(predictor_caches)
        },
    }

    def frame_generation_nodes(prefix: str, hidden: str, logits: str) -> list[dict[str, Any]]:
        if prefix == "setup":
            talker_sampler = "setup_talker_sampler"
            talker_effect = "setup_talker_sample"
            predictor_sampler = "setup_predictor_sampler"
            predictor_effect = "setup_predictor_sample"
        else:
            talker_sampler = "talker_sampler"
            talker_effect = "talker_sample"
            predictor_sampler = "predictor_prefill_sampler"
            predictor_effect = "predictor_prefill_sample"
        initializer = f"{prefix}.predictor.initializer"
        return [
            _invoke(
                "last_token_logits",
                {"logits": logits},
                {"last_logits": f"{prefix}.group0_logits"},
            ),
            _invoke(
                talker_sampler,
                {"logits": f"{prefix}.group0_logits"},
                {"token": f"{prefix}.group0"},
                {talker_effect: _effect(f"{talker_effect}.0", f"{talker_effect}.1")},
            ),
            _invoke(
                "token_to_slot",
                {
                    "token": f"{prefix}.group0",
                },
                {"slot": f"{prefix}.group0_slot"},
            ),
            _invoke(
                "embedding",
                {
                    "text_ids": "request.prompt_tokens",
                    "codec_ids": f"{prefix}.group0_slot",
                },
                {
                    "text_embeds": f"{prefix}.unused_text_embeds",
                    "codec_embeds": f"{prefix}.group0_embed",
                },
            ),
            _invoke(
                "code_predictor_prefill",
                {
                    "talker_hidden": hidden,
                    "group_0_embed": f"{prefix}.group0_embed",
                },
                {"inputs_embeds": f"{prefix}.predictor_prefill"},
            ),
            _invoke(
                "predictor_state_initializer",
                {"prefill_embeds": f"{prefix}.predictor_prefill"},
                {
                    predictor_mask.name: f"{initializer}.{predictor_mask.name}",
                    predictor_position.name: f"{initializer}.{predictor_position.name}",
                    "body_attention_mask": f"{initializer}.body_attention_mask",
                    "body_position_ids": f"{initializer}.body_position_ids",
                    **{
                        past.name: f"{initializer}.{past.name}" for past, _ in predictor_caches
                    },
                },
            ),
            _invoke(
                "code_predictor",
                {
                    predictor_embed.name: f"{prefix}.predictor_prefill",
                    predictor_step.name: "package.zero_scalar",
                    predictor_mask.name: f"{initializer}.{predictor_mask.name}",
                    predictor_position.name: f"{initializer}.{predictor_position.name}",
                    **{
                        past.name: f"{initializer}.{past.name}" for past, _ in predictor_caches
                    },
                },
                predictor_outputs(f"{prefix}.predictor"),
            ),
            _invoke(
                "last_token_logits",
                {"logits": f"{prefix}.predictor.logits"},
                {"last_logits": f"{prefix}.group1_logits"},
            ),
            _invoke(
                predictor_sampler,
                {"logits": f"{prefix}.group1_logits"},
                {"token": f"{prefix}.group1"},
                {
                    predictor_effect: _effect(
                        f"{predictor_effect}.0",
                        f"{predictor_effect}.1",
                    )
                },
            ),
            _invoke(
                "code_frame_update",
                {
                    "frame_codes": "initializer.frame_codes",
                    "token": f"{prefix}.group0",
                    "index": "package.zero_scalar",
                },
                {"next_frame": f"{prefix}.frame_group0"},
            ),
            _invoke(
                "code_frame_update",
                {
                    "frame_codes": f"{prefix}.frame_group0",
                    "token": f"{prefix}.group1",
                    "index": "package.one_scalar",
                },
                {"next_frame": f"{prefix}.frame_prefill"},
            ),
        ]

    setup_completion_nodes: list[dict[str, Any]] = []
    setup_frame = "setup.frame_prefill"
    setup_token = "setup.group1"
    setup_mask = "setup.predictor.initializer.body_attention_mask"
    setup_position = "setup.predictor.initializer.body_position_ids"
    setup_caches = [f"setup.predictor.{present.name}" for _, present in predictor_caches]
    for iteration in range(num_groups - 2):
        prefix = f"setup.predictor.remaining_{iteration}"
        setup_completion_nodes.extend(
            [
                _invoke(
                    "code_predictor_indices",
                    {"iteration": (f"package.setup_predictor_iteration_{iteration}")},
                    {
                        "embedding_index": f"{prefix}.embedding_index",
                        "step_index": f"{prefix}.step_index",
                        "frame_index": f"{prefix}.frame_index",
                    },
                ),
                _invoke(
                    "code_predictor_step_embedder",
                    {
                        "codec_embeddings": "setup.predictor.codec_embeddings",
                        "token": setup_token,
                        "embedding_index": f"{prefix}.embedding_index",
                    },
                    {"inputs_embeds": f"{prefix}.inputs_embeds"},
                ),
                _invoke(
                    "code_predictor",
                    {
                        predictor_embed.name: f"{prefix}.inputs_embeds",
                        predictor_step.name: f"{prefix}.step_index",
                        predictor_mask.name: setup_mask,
                        predictor_position.name: setup_position,
                        **{
                            past.name: setup_caches[index]
                            for index, (past, _) in enumerate(predictor_caches)
                        },
                    },
                    predictor_outputs(prefix),
                ),
                _invoke(
                    "last_token_logits",
                    {"logits": f"{prefix}.logits"},
                    {"last_logits": f"{prefix}.last_logits"},
                ),
                _invoke(
                    "predictor_body_sampler",
                    {"logits": f"{prefix}.last_logits"},
                    {"token": f"{prefix}.token"},
                    {
                        "predictor_body_sample": _effect(
                            f"predictor_body_sample.{iteration}",
                            f"predictor_body_sample.{iteration + 1}",
                        )
                    },
                ),
                _invoke(
                    "code_frame_update",
                    {
                        "frame_codes": setup_frame,
                        "token": f"{prefix}.token",
                        "index": f"{prefix}.frame_index",
                    },
                    {"next_frame": f"{prefix}.frame"},
                ),
                _invoke(
                    "predictor_step_update",
                    {
                        "attention_mask": setup_mask,
                        "position_ids": setup_position,
                    },
                    {
                        "next_attention_mask": f"{prefix}.mask",
                        "next_position_ids": f"{prefix}.position",
                    },
                ),
            ]
        )
        setup_frame = f"{prefix}.frame"
        setup_token = f"{prefix}.token"
        setup_mask = f"{prefix}.mask"
        setup_position = f"{prefix}.position"
        setup_caches = [f"{prefix}.{present.name}" for _, present in predictor_caches]

    inner_body = {
        "kind": "sequence",
        "nodes": [
            _invoke(
                "code_predictor_indices",
                {"iteration": "code.iteration"},
                {
                    "embedding_index": "predictor.body.embedding_index",
                    "step_index": "predictor.body.step_index",
                    "frame_index": "predictor.body.frame_index",
                },
            ),
            _invoke(
                "code_predictor_step_embedder",
                {
                    "codec_embeddings": "frame.predictor.codec_embeddings",
                    "token": "state.code_token.inner",
                    "embedding_index": "predictor.body.embedding_index",
                },
                {"inputs_embeds": "predictor.body.inputs_embeds"},
            ),
            _invoke(
                "code_predictor", predictor_body_inputs, predictor_outputs("predictor.body")
            ),
            _invoke(
                "last_token_logits",
                {"logits": "predictor.body.logits"},
                {"last_logits": "predictor.body.last_logits"},
            ),
            _invoke(
                "predictor_body_sampler",
                {"logits": "predictor.body.last_logits"},
                {"token": "code.token"},
                {
                    "predictor_body_sample": _effect(
                        f"predictor_body_sample.{num_groups - 2}",
                        f"predictor_body_sample.{num_groups - 1}",
                    )
                },
            ),
            _invoke(
                "code_frame_update",
                {
                    "frame_codes": "state.frame.inner",
                    "token": "code.token",
                    "index": "predictor.body.frame_index",
                },
                {"next_frame": "frame.inner"},
            ),
            _invoke(
                "predictor_step_update",
                {
                    "attention_mask": "state.predictor_mask.inner",
                    "position_ids": "state.predictor_position.inner",
                },
                {
                    "next_attention_mask": "predictor.mask.inner",
                    "next_position_ids": "predictor.position.inner",
                },
            ),
            _invoke(
                "continue_predicate",
                {"done": "package.false"},
                {"continue": "code.continue"},
            ),
        ],
    }
    inner_carried = [
        {
            "cell": "frame",
            "current": "frame.frame_prefill",
            "body_input": "state.frame.inner",
            "body_output": "frame.inner",
            "next": "frame.completed",
            "read_effect": _effect("state:frame.0", "state:frame.read"),
            "write_effect": _effect("state:frame.read", "state:frame.1"),
        },
        {
            "cell": "code_token",
            "current": "frame.group1",
            "body_input": "state.code_token.inner",
            "body_output": "code.token",
            "next": "code_token.final",
            "read_effect": _effect("state:code_token.0", "state:code_token.read"),
            "write_effect": _effect("state:code_token.read", "state:code_token.1"),
        },
        {
            "cell": "predictor_mask",
            "current": "frame.predictor.initializer.body_attention_mask",
            "body_input": "state.predictor_mask.inner",
            "body_output": "predictor.mask.inner",
            "next": "predictor.mask.final",
            "read_effect": _effect("state:predictor_mask.0", "state:predictor_mask.read"),
            "write_effect": _effect("state:predictor_mask.read", "state:predictor_mask.1"),
        },
        {
            "cell": "predictor_position",
            "current": "frame.predictor.initializer.body_position_ids",
            "body_input": "state.predictor_position.inner",
            "body_output": "predictor.position.inner",
            "next": "predictor.position.final",
            "read_effect": _effect(
                "state:predictor_position.0", "state:predictor_position.read"
            ),
            "write_effect": _effect(
                "state:predictor_position.read", "state:predictor_position.1"
            ),
        },
    ]
    for index, (_, present) in enumerate(predictor_caches):
        inner_carried.append(
            {
                "cell": f"predictor_cache_{index}",
                "current": f"frame.predictor.{present.name}",
                "body_input": f"state.predictor_cache_{index}.inner",
                "body_output": f"predictor.body.{present.name}",
                "next": f"predictor.cache_{index}.final",
                "read_effect": _effect(
                    f"state:predictor_cache_{index}.0", f"state:predictor_cache_{index}.read"
                ),
                "write_effect": _effect(
                    f"state:predictor_cache_{index}.read", f"state:predictor_cache_{index}.1"
                ),
            }
        )
    inner_loop = {
        "kind": "loop",
        "setup": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "continue_predicate",
                    {"done": "package.false"},
                    {"continue": "code.setup.continue"},
                )
            ],
        },
        "body": inner_body,
        "condition": "code.continue",
        "max_iterations": "package.remaining_groups",
        "iteration": {
            "value": "code.iteration",
            "contract": _contract(next(iter(predictor_indices.graph.inputs))),
        },
        "carried": inner_carried,
    }

    outer_body_nodes = [
        _invoke(
            "talker_text_step",
            {
                "trailing_text_embeds": "tts.trailing_text_embeds",
                "iteration": "talker.iteration",
            },
            {"text_embed": "talker.text_embed"},
        ),
        _invoke(
            "talker_step_embedder",
            {
                "frame_codes": "state.last_frame.outer",
                "text_embed": "talker.text_embed",
            },
            {"inputs_embeds": "talker.step_embeds"},
        ),
        _invoke("talker", talker_body_inputs, talker_body_outputs),
        *frame_generation_nodes("frame", "talker.body.hidden", "talker.body.logits"),
        inner_loop,
        _invoke(
            "code_history_append",
            {"history": "state.history.outer", "frame": "frame.completed"},
            {"next_history": "history.outer"},
        ),
        _invoke(
            "talker_step_update",
            {
                "attention_mask": "state.talker_mask.body",
                "position_ids": "state.talker_position.body",
            },
            {
                "next_attention_mask": "talker.mask.body",
                "next_position_ids": "talker.position.body",
            },
        ),
        _invoke(
            "continue_predicate",
            {"done": "package.false"},
            {"continue": "talker.continue"},
        ),
    ]

    setup_nodes = [
        _invoke(
            "tts_state_initializer",
            {"prompt_tokens": "request.prompt_tokens"},
            {
                "frame_codes": "initializer.frame_codes",
                "token_slot": "initializer.token_slot",
                "code_history": "initializer.code_history",
            },
        ),
        _invoke(
            "talker_prefill_embedder",
            {prompt.name: "request.prompt_tokens"},
            {
                prefill.name: "tts.prefill_embeds",
                trailing.name: "tts.trailing_text_embeds",
            },
        ),
        _invoke(
            "talker_state_initializer",
            {"prefill_embeds": "tts.prefill_embeds"},
            {
                talker_mask.name: f"talker.initializer.{talker_mask.name}",
                talker_position.name: f"talker.initializer.{talker_position.name}",
                "body_attention_mask": "talker.initializer.body_attention_mask",
                "body_position_ids": "talker.initializer.body_position_ids",
                **{past.name: f"talker.initializer.{past.name}" for past, _ in talker_caches},
            },
        ),
        _invoke("talker", talker_setup_inputs, talker_setup_outputs),
        *frame_generation_nodes("setup", "talker.setup.hidden", "talker.setup.logits"),
        *setup_completion_nodes,
        _invoke(
            "code_history_append",
            {"history": "initializer.code_history", "frame": setup_frame},
            {"next_history": "history.setup"},
        ),
        _invoke(
            "continue_predicate",
            {"done": "package.false"},
            {"continue": "talker.setup.continue"},
        ),
    ]

    state = {
        "last_frame": {
            "contract": {"dtype": "int64", "rank": 2, "shape": [batch, num_groups]},
            "scope": "invocation",
            "initializer": setup_frame,
            "recurrence": {"kind": "invariant"},
        },
        "history": {
            "contract": {"dtype": "int64", "rank": 3, "shape": [batch, "frames", num_groups]},
            "scope": "invocation",
            "initializer": "history.setup",
            "recurrence": {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one_scalar",
                "max": "package.talker_context_limit",
            },
        },
        "talker_mask": {
            "contract": {
                "dtype": _contract(talker_mask)["dtype"],
                "rank": 2,
                "shape": [batch, "talker_context"],
            },
            "scope": "invocation",
            "initializer": "talker.initializer.body_attention_mask",
            "recurrence": {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one_scalar",
                "max": "package.talker_context_limit",
            },
        },
        "talker_position": {
            "contract": _contract(
                next(
                    value
                    for value in pkg.policy_components[
                        "talker_state_initializer"
                    ].model.graph.outputs
                    if value.name == "body_position_ids"
                )
            ),
            "scope": "invocation",
            "initializer": "talker.initializer.body_position_ids",
            "recurrence": {"kind": "invariant"},
        },
        "frame": {
            "contract": {"dtype": "int64", "rank": 2, "shape": [batch, num_groups]},
            "scope": "invocation",
            "initializer": "frame.frame_prefill",
            "recurrence": {"kind": "invariant"},
        },
        "code_token": {
            "contract": batch_int,
            "scope": "invocation",
            "initializer": "frame.group1",
            "recurrence": {"kind": "invariant"},
        },
        "predictor_mask": {
            "contract": {
                "dtype": _contract(predictor_mask)["dtype"],
                "rank": 2,
                "shape": [batch, "predictor_context"],
            },
            "scope": "invocation",
            "initializer": "frame.predictor.initializer.body_attention_mask",
            "recurrence": {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one_scalar",
                "max": "package.predictor_mask_limit",
            },
        },
        "predictor_position": {
            "contract": _contract(
                next(
                    value
                    for value in pkg.policy_components[
                        "predictor_state_initializer"
                    ].model.graph.outputs
                    if value.name == "body_position_ids"
                )
            ),
            "scope": "invocation",
            "initializer": "frame.predictor.initializer.body_position_ids",
            "recurrence": {"kind": "invariant"},
        },
    }
    for index, (past, present) in enumerate(talker_caches):
        state[f"talker_cache_{index}"] = {
            "contract": _contract(past),
            "scope": "invocation",
            "initializer": f"talker.setup.{present.name}",
            "recurrence": {
                "kind": "growing",
                "axis": 2,
                "increment": "package.one_scalar",
                "max": "package.talker_context_limit",
            },
        }
    for index, (past, present) in enumerate(predictor_caches):
        state[f"predictor_cache_{index}"] = {
            "contract": _contract(past),
            "scope": "invocation",
            "initializer": f"frame.predictor.{present.name}",
            "recurrence": {
                "kind": "growing",
                "axis": 2,
                "increment": "package.one_scalar",
                "max": "package.predictor_context_limit",
            },
        }

    outer_carried = [
        {
            "cell": "last_frame",
            "current": "setup.frame_prefill",
            "body_input": "state.last_frame.outer",
            "body_output": "frame.completed",
            "next": "last_frame.final",
            "read_effect": _effect("state:last_frame.0", "state:last_frame.read"),
            "write_effect": _effect("state:last_frame.read", "state:last_frame.1"),
        },
        {
            "cell": "history",
            "current": "history.setup",
            "body_input": "state.history.outer",
            "body_output": "history.outer",
            "next": "history.final",
            "read_effect": _effect("state:history.0", "state:history.read"),
            "write_effect": _effect("state:history.read", "state:history.1"),
        },
        {
            "cell": "talker_mask",
            "current": "talker.initializer.body_attention_mask",
            "body_input": "state.talker_mask.body",
            "body_output": "talker.mask.body",
            "next": "talker.mask.final",
            "read_effect": _effect("state:talker_mask.0", "state:talker_mask.read"),
            "write_effect": _effect("state:talker_mask.read", "state:talker_mask.1"),
        },
        {
            "cell": "talker_position",
            "current": "talker.initializer.body_position_ids",
            "body_input": "state.talker_position.body",
            "body_output": "talker.position.body",
            "next": "talker.position.final",
            "read_effect": _effect("state:talker_position.0", "state:talker_position.read"),
            "write_effect": _effect("state:talker_position.read", "state:talker_position.1"),
        },
    ]
    for index, (_, present) in enumerate(talker_caches):
        outer_carried.append(
            {
                "cell": f"talker_cache_{index}",
                "current": f"talker.setup.{present.name}",
                "body_input": f"state.talker_cache_{index}.body",
                "body_output": f"talker.body.{present.name}",
                "next": f"talker.cache_{index}.final",
                "read_effect": _effect(
                    f"state:talker_cache_{index}.0", f"state:talker_cache_{index}.read"
                ),
                "write_effect": _effect(
                    f"state:talker_cache_{index}.read", f"state:talker_cache_{index}.1"
                ),
            }
        )
    initial_effects = {
        "setup_talker_sample": "setup_talker_sample.0",
        "setup_predictor_sample": "setup_predictor_sample.0",
        "talker_sample": "talker_sample.0",
        "predictor_prefill_sample": "predictor_prefill_sample.0",
        "predictor_body_sample": "predictor_body_sample.0",
        "emit": "emit.0",
    }
    for cell in state:
        initial_effects[f"state:{cell}"] = f"state:{cell}.0"

    codec_value = "history.final"
    final_nodes: list[dict[str, Any]] = []
    if codec_group_major:
        final_nodes.append(
            _invoke("codec_layout", {"history": "history.final"}, {"codes": "codec.codes"})
        )
        codec_value = "codec.codes"
    final_nodes.extend(
        [
            _invoke(
                codec_name, {codec_input.name: codec_value}, {waveform.name: "tts.waveform"}
            ),
            {
                "kind": "emit",
                "value": "tts.waveform",
                "output": "waveform",
                "mode": "replace",
                "effect_name": "emit",
                "effect": _effect("emit.0", "emit.1"),
            },
        ]
    )
    workflow = {
        "manifest": {
            "ir_version": "1.0",
            "onnx_opsets": {"ai.onnx": OPSET_VERSION},
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
            ],
        },
        "inputs": inputs,
        "outputs": {
            "waveform": {
                "contract": _contract(waveform),
                "role": "audio",
                "stage": "post_adapter",
            }
        },
        "components": {
            name: _component(model, _artifact(name, len(pkg))) for name, model in pkg.items()
        },
        "state": state,
        "initial_effects": initial_effects,
        "graph": {
            "kind": "sequence",
            "nodes": [
                {
                    "kind": "loop",
                    "setup": {"kind": "sequence", "nodes": setup_nodes},
                    "body": {"kind": "sequence", "nodes": outer_body_nodes},
                    "condition": "talker.continue",
                    "max_iterations": "request.max_iterations",
                    "iteration": {"value": "talker.iteration", "contract": batch_int},
                    "carried": outer_carried,
                },
                *final_nodes,
            ],
        },
    }
    metadata = {
        "schema_version": "v1",
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def build_tts_workflow_metadata(pkg: Any, config: Any) -> dict[str, Any]:
    """Build nested talker/code-predictor loops with lexical induction SSA."""
    real_transition_components = {
        "embedding",
        "code_predictor_prefill",
        "code_predictor_step_embedder",
        "code_predictor_indices",
        "talker_text_step",
    }
    if real_transition_components <= set(pkg.keys()):
        return _build_real_tts_workflow_metadata(pkg, config)
    required = {
        "talker",
        "code_predictor",
        "talker_step_embedder",
        "talker_prefill_embedder",
    }
    missing = sorted(required.difference(pkg.keys()))
    if missing:
        raise ValueError(f"TTS workflow is missing required components: {missing}")
    talker = pkg["talker"]
    predictor = pkg["code_predictor"]
    step_embedder = pkg["talker_step_embedder"]
    prefill_embedder = pkg["talker_prefill_embedder"]
    num_groups = int(getattr(getattr(config, "tts", config), "num_code_groups", 16))
    prompt_input = next(
        (
            value
            for value in prefill_embedder.graph.inputs
            if value.dtype in {ir.DataType.INT32, ir.DataType.INT64}
            and value.shape is not None
            and len(value.shape) == 2
        ),
        None,
    )
    prefill_output = _find_port(prefill_embedder.graph.outputs, "prefill", "embeds")
    step_frame_input = _find_port(step_embedder.graph.inputs, "frame", "codes")
    step_output = _find_port(step_embedder.graph.outputs, "embeds")
    talker_embed_input = _find_port(talker.graph.inputs, "embeds")
    talker_hidden = _find_port(talker.graph.outputs, "hidden")
    predictor_hidden_input = _find_port(predictor.graph.inputs, "hidden", "embeds")
    predictor_step_input = _find_port(predictor.graph.inputs, "step", "index")
    predictor_logits = _find_port(predictor.graph.outputs, "logits")
    if None in (
        prompt_input,
        prefill_output,
        step_frame_input,
        step_output,
        talker_embed_input,
        talker_hidden,
        predictor_hidden_input,
        predictor_step_input,
        predictor_logits,
    ):
        raise ValueError("TTS components do not expose the required typed ports")
    assert prompt_input is not None
    assert prefill_output is not None
    assert step_frame_input is not None
    assert step_output is not None
    assert talker_embed_input is not None
    assert talker_hidden is not None
    assert predictor_hidden_input is not None
    assert predictor_step_input is not None
    assert predictor_logits is not None
    if predictor_logits.shape is None or len(predictor_logits.shape) != 2:
        raise ValueError("TTS code predictor logits must be rank 2")
    predictor_step_contract = _contract(predictor_step_input)
    scalar_code_index = predictor_step_contract["rank"] == 0
    if predictor_step_contract["dtype"] != "int64" or predictor_step_contract["rank"] not in {
        0,
        1,
    }:
        raise ValueError("TTS code predictor step index must be scalar or batch int64")

    codec_name = next(
        (name for name in ("codec", "vocoder", "decoder") if name in pkg),
        None,
    )
    codec = pkg[codec_name] if codec_name is not None else None
    codec_input = next(iter(codec.graph.inputs), None) if codec is not None else None
    waveform_output = next(iter(codec.graph.outputs), None) if codec is not None else None
    if codec is None or codec_input is None or waveform_output is None:
        raise ValueError("TTS workflow requires a codec or vocoder component")

    attach_policy_components(pkg, PolicyCapabilities(sampler="greedy"))
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    pkg.add_policy_component("tts_state_initializer", build_tts_state_initializer(num_groups))
    pkg.add_policy_component(
        "code_frame_update",
        build_code_frame_update(num_groups, scalar_index=scalar_code_index),
    )
    pkg.add_policy_component("code_history_append", build_code_history_append(num_groups))
    codec_group_major = (
        codec_input.shape is not None
        and len(codec_input.shape) == 3
        and str(getattr(list(codec_input.shape)[1], "value", list(codec_input.shape)[1]))
        == str(num_groups)
    )
    if codec_group_major:
        pkg.add_policy_component("codec_layout", build_codec_layout_transpose(num_groups))

    batch = _contract(prompt_input)["shape"][0]
    batch_int = {"dtype": "int64", "rank": 1, "shape": [batch]}
    batch_bool = {"dtype": "bool", "rank": 1, "shape": [batch]}
    inputs: dict[str, Any] = {
        "request.prompt_tokens": {
            "contract": _contract(prompt_input),
            "role": {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"},
            "source": {"kind": "request", "field": "prompt_tokens"},
            "required": True,
        },
        "request.max_iterations": {
            "contract": batch_int,
            "role": {
                "kind": "runtime",
                "version": "1.0",
                "role": "max_output_tokens",
            },
            "source": {"kind": "request", "field": "max_output_tokens"},
            "required": True,
        },
        "package.code_groups": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": num_groups,
        },
        "package.false": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": False,
        },
        "package.one": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
    }

    def bind_remaining(
        component: str,
        values: Any,
        known: dict[str, str],
    ) -> dict[str, str]:
        result = dict(known)
        for value in values:
            if value.name in result:
                continue
            name = f"request.{component}.{value.name}"
            inputs[name] = {
                "contract": _contract(value),
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": f"{component}.{value.name}"},
                "required": True,
            }
            result[value.name] = name
        return result

    def bind_outputs(values: Any, bound: dict[str, str], prefix: str) -> dict[str, str]:
        result = dict(bound)
        for value in values:
            result.setdefault(value.name, f"{prefix}.{value.name}")
        return result

    prefill_inputs = bind_remaining(
        "talker_prefill_embedder",
        prefill_embedder.graph.inputs,
        {prompt_input.name: "request.prompt_tokens"},
    )
    talker_setup_inputs = bind_remaining(
        "talker",
        talker.graph.inputs,
        {talker_embed_input.name: "talker.prefill_embeds"},
    )
    step_inputs = bind_remaining(
        "talker_step_embedder",
        step_embedder.graph.inputs,
        {step_frame_input.name: "state.last_frame.outer"},
    )
    talker_body_inputs = bind_remaining(
        "talker",
        talker.graph.inputs,
        {talker_embed_input.name: "talker.step_embeds"},
    )
    predictor_inputs = bind_remaining(
        "code_predictor",
        predictor.graph.inputs,
        {
            predictor_hidden_input.name: "talker.body.hidden",
            predictor_step_input.name: "code.iteration",
        },
    )

    frame_contract = {
        "dtype": "int64",
        "rank": 2,
        "shape": [batch, num_groups],
    }
    history_contract = {
        "dtype": "int64",
        "rank": 3,
        "shape": [batch, "frames", num_groups],
    }
    initial_effects = {
        "sample": "sample.0",
        "emit": "emit.0",
        "state:last_frame": "state:last_frame.0",
        "state:frame": "state:frame.0",
        "state:history": "state:history.0",
    }
    inner_loop = {
        "kind": "loop",
        "setup": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "continue_predicate",
                    {"done": "package.false"},
                    {"continue": "code.setup.continue"},
                )
            ],
        },
        "body": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "code_predictor",
                    predictor_inputs,
                    bind_outputs(
                        predictor.graph.outputs,
                        {predictor_logits.name: "code.logits"},
                        "code_predictor.body",
                    ),
                ),
                _invoke(
                    "token_sampler",
                    {"logits": "code.logits"},
                    {"token": "code.token"},
                    {"sample": _effect("sample.0", "sample.1")},
                ),
                _invoke(
                    "code_frame_update",
                    {
                        "frame_codes": "state.frame.inner",
                        "token": "code.token",
                        "index": "code.iteration",
                    },
                    {"next_frame": "frame.inner"},
                ),
                _invoke(
                    "continue_predicate",
                    {"done": "package.false"},
                    {"continue": "code.continue"},
                ),
            ],
        },
        "condition": "code.continue",
        "max_iterations": "package.code_groups",
        "iteration": {
            "value": "code.iteration",
            "contract": predictor_step_contract,
        },
        "carried": [
            {
                "cell": "frame",
                "current": "initializer.frame_codes",
                "body_input": "state.frame.inner",
                "body_output": "frame.inner",
                "next": "frame.completed",
                "read_effect": _effect("state:frame.0", "state:frame.read"),
                "write_effect": _effect("state:frame.read", "state:frame.1"),
            }
        ],
    }
    outer_body = {
        "kind": "sequence",
        "nodes": [
            _invoke(
                "talker_step_embedder",
                step_inputs,
                bind_outputs(
                    step_embedder.graph.outputs,
                    {step_output.name: "talker.step_embeds"},
                    "talker_step_embedder.body",
                ),
            ),
            _invoke(
                "talker",
                talker_body_inputs,
                bind_outputs(
                    talker.graph.outputs,
                    {talker_hidden.name: "talker.body.hidden"},
                    "talker.body",
                ),
            ),
            inner_loop,
            _invoke(
                "code_history_append",
                {
                    "history": "state.history.outer",
                    "frame": "frame.completed",
                },
                {"next_history": "history.outer"},
            ),
            _invoke(
                "continue_predicate",
                {"done": "package.false"},
                {"continue": "talker.continue"},
            ),
        ],
    }
    setup_outputs = bind_outputs(
        prefill_embedder.graph.outputs,
        {prefill_output.name: "talker.prefill_embeds"},
        "talker_prefill_embedder.setup",
    )
    if talker_hidden.name in {value.name for value in talker.graph.outputs}:
        talker_setup_outputs = bind_outputs(
            talker.graph.outputs,
            {talker_hidden.name: "talker.prefill.hidden"},
            "talker.setup",
        )
    else:
        talker_setup_outputs = {}
    outer_loop = {
        "kind": "loop",
        "setup": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "tts_state_initializer",
                    {"prompt_tokens": "request.prompt_tokens"},
                    {
                        "frame_codes": "initializer.frame_codes",
                        "code_history": "initializer.code_history",
                    },
                ),
                _invoke("talker_prefill_embedder", prefill_inputs, setup_outputs),
                _invoke("talker", talker_setup_inputs, talker_setup_outputs),
            ],
        },
        "body": outer_body,
        "condition": "talker.continue",
        "max_iterations": "request.max_iterations",
        "iteration": {"value": "talker.iteration", "contract": batch_int},
        "carried": [
            {
                "cell": "last_frame",
                "current": "initializer.frame_codes",
                "body_input": "state.last_frame.outer",
                "body_output": "frame.completed",
                "next": "state.last_frame.final",
                "read_effect": _effect("state:last_frame.0", "state:last_frame.read"),
                "write_effect": _effect("state:last_frame.read", "state:last_frame.1"),
            },
            {
                "cell": "history",
                "current": "initializer.code_history",
                "body_input": "state.history.outer",
                "body_output": "history.outer",
                "next": "history.final",
                "read_effect": _effect("state:history.0", "state:history.read"),
                "write_effect": _effect("state:history.read", "state:history.1"),
            },
        ],
    }
    codec_value = "codec.codes"
    post_nodes: list[dict[str, Any]] = [outer_loop]
    if codec_group_major:
        post_nodes.append(
            _invoke(
                "codec_layout",
                {"history": "history.final"},
                {"codes": codec_value},
            )
        )
    else:
        codec_value = "history.final"
    post_nodes.extend(
        [
            _invoke(
                codec_name,
                {codec_input.name: codec_value},
                bind_outputs(
                    codec.graph.outputs,
                    {waveform_output.name: "tts.waveform"},
                    "codec.final",
                ),
            ),
            {
                "kind": "emit",
                "value": "tts.waveform",
                "output": "waveform",
                "mode": "replace",
                "effect_name": "emit",
                "effect": _effect("emit.0", "emit.1"),
            },
        ]
    )
    workflow = {
        "manifest": {
            "ir_version": "1.0",
            "onnx_opsets": {"ai.onnx": OPSET_VERSION},
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
            ],
        },
        "inputs": inputs,
        "outputs": {
            "waveform": {
                "contract": _contract(waveform_output),
                "role": "audio",
                "stage": "post_adapter",
            }
        },
        "components": {
            name: _component(model, _artifact(name, len(pkg))) for name, model in pkg.items()
        },
        "state": {
            "last_frame": {
                "contract": frame_contract,
                "scope": "invocation",
                "initializer": "initializer.frame_codes",
                "recurrence": {"kind": "invariant"},
            },
            "frame": {
                "contract": frame_contract,
                "scope": "invocation",
                "initializer": "initializer.frame_codes",
                "recurrence": {"kind": "invariant"},
            },
            "history": {
                "contract": history_contract,
                "scope": "invocation",
                "initializer": "initializer.code_history",
                "recurrence": {
                    "kind": "growing",
                    "axis": 1,
                    "increment": "package.one",
                    "max": "request.max_iterations",
                },
            },
        },
        "initial_effects": initial_effects,
        "graph": {"kind": "sequence", "nodes": post_nodes},
    }
    metadata = {
        "schema_version": "v1",
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def write_tts_workflow_metadata(pkg: Any, output_dir: str, config: Any) -> str:
    """Build and save an executable TTS workflow package."""
    metadata = build_tts_workflow_metadata(pkg, config)
    os.makedirs(output_dir, exist_ok=True)
    pkg.save_policy_components(output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def _artifact(name: str, package_size: int) -> str:
    return f"{name}/model.onnx" if package_size > 1 else "model.onnx"


def _find_port(values: Any, *fragments: str) -> ir.Value | None:
    return next(
        (value for value in values if any(part in value.name.lower() for part in fragments)),
        None,
    )


def _contracts_compatible(left: ir.Value, right: ir.Value) -> bool:
    """Return whether symbolic tensor contracts can unify by dtype and fixed dims."""
    if left.dtype != right.dtype or left.shape is None or right.shape is None:
        return False
    left_dims = list(left.shape)
    right_dims = list(right.shape)
    if len(left_dims) != len(right_dims):
        return False
    return all(
        not isinstance(left_dim, int)
        or not isinstance(right_dim, int)
        or left_dim == right_dim
        for left_dim, right_dim in zip(left_dims, right_dims)
    )


def build_diffusion_workflow_metadata(
    pkg: Any,
    *,
    num_inference_steps: int,
    schedule: list[float] | None = None,
    timesteps: list[float] | None = None,
) -> dict[str, Any]:
    """Build a fixed-schedule diffusion workflow with explicit latent state."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    names = set(pkg.keys())
    denoiser_name = next(
        (name for name in ("denoiser", "transformer", "unet") if name in names),
        None,
    )
    vae_name = next(
        (name for name in ("vae_decoder", "decoder", "vae") if name in names),
        None,
    )
    if denoiser_name is None or vae_name is None or denoiser_name == vae_name:
        raise ValueError("diffusion workflow requires distinct denoiser and VAE decoder")
    denoiser = pkg[denoiser_name]
    vae = pkg[vae_name]
    sample_input = _find_port(denoiser.graph.inputs, "sample", "latent", "hidden_states")
    timestep_input = _find_port(denoiser.graph.inputs, "timestep", "time")
    estimate_output = next(iter(denoiser.graph.outputs), None)
    vae_input = _find_port(vae.graph.inputs, "latent", "sample")
    vae_output = next(iter(vae.graph.outputs), None)
    if None in (sample_input, timestep_input, estimate_output, vae_input, vae_output):
        raise ValueError(
            "diffusion components do not expose sample/timestep/estimate/VAE ports"
        )
    assert sample_input is not None
    assert timestep_input is not None
    assert estimate_output is not None
    assert vae_input is not None
    assert vae_output is not None
    if len(sample_input.shape or []) != 4 or _contract(sample_input) != _contract(
        estimate_output
    ):
        raise ValueError("Euler diffusion workflow requires matching rank-4 latent/estimate")
    if _contract(vae_input) != _contract(sample_input):
        raise ValueError("VAE latent input must match the solver latent contract")

    text_name = next(
        (name for name in ("text_encoder", "text_encoder_2") if name in names),
        None,
    )
    text_encoder = pkg[text_name] if text_name is not None else None
    conditioning_input = next(
        (
            value
            for value in denoiser.graph.inputs
            if value is not sample_input
            and value is not timestep_input
            and ("encoder" in value.name or "context" in value.name)
        ),
        None,
    )
    conditioning_output = None
    if text_encoder is not None and conditioning_input is not None:
        conditioning_output = next(
            (
                value
                for value in text_encoder.graph.outputs
                if _contract(value) == _contract(conditioning_input)
            ),
            next(iter(text_encoder.graph.outputs), None),
        )

    attach_policy_components(pkg, PolicyCapabilities())
    pkg.add_policy_component("euler_model_input", build_euler_model_input(sample_input.dtype))
    pkg.add_policy_component("solver_step", build_euler_solver_step(sample_input.dtype))
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    schedule_values = schedule or [
        1.0 - index / num_inference_steps for index in range(num_inference_steps + 1)
    ]
    timestep_values = timesteps or schedule_values[:-1]
    if len(schedule_values) != num_inference_steps + 1:
        raise ValueError(
            "diffusion solver schedule must contain num_inference_steps + 1 values"
        )
    if len(timestep_values) != num_inference_steps:
        raise ValueError("diffusion timesteps must contain num_inference_steps values")
    pkg.add_policy_component("diffusion_schedule", build_schedule_constant(schedule_values))
    pkg.add_policy_component("diffusion_timesteps", build_schedule_constant(timestep_values))
    pkg.add_policy_component("schedule_lookup", build_schedule_lookup(timestep_input.dtype))

    batch = _contract(sample_input)["shape"][0]
    batch_int = {"dtype": "int64", "rank": 1, "shape": [batch]}
    batch_bool = {"dtype": "bool", "rank": 1, "shape": [batch]}
    inputs: dict[str, Any] = {
        "request.latent": {
            "contract": _contract(sample_input),
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "latent"},
            "required": True,
        },
        "request.max_iterations": {
            "contract": batch_int,
            "role": {"kind": "runtime", "version": "1.0", "role": "max_iterations"},
            "source": {"kind": "request", "field": "max_iterations"},
            "required": False,
            "default": num_inference_steps,
        },
        "package.false": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": False,
        },
    }
    setup_nodes: list[dict[str, Any]] = [
        _invoke("diffusion_schedule", {}, {"schedule": "diffusion.schedule"}),
        _invoke("diffusion_timesteps", {}, {"schedule": "diffusion.timesteps"}),
    ]
    conditioning_value = None
    if text_encoder is not None and conditioning_output is not None:
        text_inputs = {}
        for index, value in enumerate(text_encoder.graph.inputs):
            name = f"request.{value.name}"
            inputs[name] = {
                "contract": _contract(value),
                "role": (
                    {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"}
                    if index == 0
                    else {"kind": "opaque"}
                ),
                "source": {
                    "kind": "request" if index == 0 else "application",
                    "field": "prompt_tokens" if index == 0 else None,
                    "name": value.name if index else None,
                },
                "required": True,
            }
            inputs[name]["source"] = {
                key: item for key, item in inputs[name]["source"].items() if item is not None
            }
            text_inputs[value.name] = name
        conditioning_value = "conditioning.hidden_states"
        setup_nodes.append(
            _invoke(
                text_name,
                text_inputs,
                {conditioning_output.name: conditioning_value},
            )
        )
    setup_nodes.append(
        _invoke(
            "continue_predicate",
            {"done": "package.false"},
            {"continue": "setup.continue"},
        )
    )

    denoiser_inputs = {
        sample_input.name: "diffusion.model_input",
        timestep_input.name: "diffusion.timestep",
    }
    if conditioning_input is not None and conditioning_value is not None:
        denoiser_inputs[conditioning_input.name] = conditioning_value
    body_nodes: list[dict[str, Any]] = []
    body_nodes.append(
        _invoke(
            "schedule_lookup",
            {
                "schedule": "diffusion.timesteps",
                "step": "loop.iteration",
            },
            {"timestep": "diffusion.timestep"},
        )
    )
    body_nodes.append(
        _invoke(
            "euler_model_input",
            {
                "sample": "state.latent.body",
                "step": "loop.iteration",
                "schedule": "diffusion.schedule",
            },
            {"model_input": "diffusion.model_input"},
        )
    )
    body_nodes.extend(
        [
            _invoke(
                denoiser_name,
                denoiser_inputs,
                {estimate_output.name: "denoiser.estimate"},
            ),
            _invoke(
                "solver_step",
                {
                    "sample": "state.latent.body",
                    "derivative": "denoiser.estimate",
                    "step": "loop.iteration",
                    "schedule": "diffusion.schedule",
                },
                {"next_state": "latent.body"},
                {"solver": _effect("solver.0", "solver.1")},
            ),
            _invoke(
                "continue_predicate",
                {"done": "package.false"},
                {"continue": "loop.continue"},
            ),
        ]
    )
    latent_effect = "state:latent"
    workflow = {
        "manifest": {
            "ir_version": "1.0",
            "onnx_opsets": {"ai.onnx": OPSET_VERSION},
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
            ],
        },
        "inputs": inputs,
        "outputs": {
            "image": {
                "contract": _contract(vae_output),
                "role": "image",
                "stage": "pre_adapter",
            }
        },
        "components": {
            name: _component(model, _artifact(name, len(pkg))) for name, model in pkg.items()
        },
        "state": {
            "latent": {
                "contract": _contract(sample_input),
                "scope": "invocation",
                "initializer": "request.latent",
                "recurrence": {"kind": "invariant"},
            }
        },
        "initial_effects": {
            "solver": "solver.0",
            latent_effect: f"{latent_effect}.0",
            "emit": "emit.0",
        },
        "graph": {
            "kind": "sequence",
            "nodes": [
                {
                    "kind": "loop",
                    "setup": {"kind": "sequence", "nodes": setup_nodes},
                    "body": {"kind": "sequence", "nodes": body_nodes},
                    "condition": "loop.continue",
                    "max_iterations": "request.max_iterations",
                    "iteration": {"value": "loop.iteration", "contract": batch_int},
                    "carried": [
                        {
                            "cell": "latent",
                            "current": "request.latent",
                            "body_input": "state.latent.body",
                            "body_output": "latent.body",
                            "next": "latent.final",
                            "read_effect": _effect(
                                f"{latent_effect}.0", f"{latent_effect}.read"
                            ),
                            "write_effect": _effect(
                                f"{latent_effect}.read", f"{latent_effect}.1"
                            ),
                        }
                    ],
                },
                _invoke(
                    vae_name,
                    {vae_input.name: "latent.final"},
                    {vae_output.name: "vae.image"},
                ),
                {
                    "kind": "emit",
                    "value": "vae.image",
                    "output": "image",
                    "mode": "replace",
                    "effect_name": "emit",
                    "effect": _effect("emit.0", "emit.1"),
                },
            ],
        },
    }
    metadata = {
        "schema_version": "v1",
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def write_diffusion_workflow_metadata(
    pkg: Any,
    output_dir: str,
    *,
    num_inference_steps: int,
    schedule: list[float] | None = None,
    timesteps: list[float] | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_diffusion_workflow_metadata(
        pkg,
        num_inference_steps=num_inference_steps,
        schedule=schedule,
        timesteps=timesteps,
    )
    pkg.save_policy_components(output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def build_vlm_workflow_metadata(
    pkg: Any,
    config: Any,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Build a vision/audio-encoder to embedding to decoder SSA workflow."""
    required = {"vision_encoder", "embedding", "decoder"}
    missing = sorted(required.difference(pkg.keys()))
    if missing:
        raise ValueError(f"VLM workflow is missing required components: {missing}")
    vision = pkg["vision_encoder"]
    embedding = pkg["embedding"]
    decoder = pkg["decoder"]
    token_input = next(
        (
            value
            for value in embedding.graph.inputs
            if value.dtype in {ir.DataType.INT32, ir.DataType.INT64}
            and value.shape is not None
            and len(value.shape) == 2
        ),
        None,
    )
    embedding_output = next(
        (
            value
            for value in embedding.graph.outputs
            if value.shape is not None and len(value.shape) == 3
        ),
        None,
    )
    decoder_embed_input = (
        next(
            (
                value
                for value in decoder.graph.inputs
                if embedding_output is not None
                and _contracts_compatible(value, embedding_output)
            ),
            None,
        )
        if embedding_output is not None
        else None
    )
    logits_output = _find_port(decoder.graph.outputs, "logits")
    if None in (token_input, embedding_output, decoder_embed_input, logits_output):
        raise ValueError("VLM workflow requires token->embedding->decoder logits ports")
    assert token_input is not None
    assert embedding_output is not None
    assert decoder_embed_input is not None
    assert logits_output is not None
    embedding_output.shape = decoder_embed_input.shape
    embedding_inputs_by_name = {value.name: value for value in embedding.graph.inputs}
    for value in vision.graph.outputs:
        target = embedding_inputs_by_name.get(value.name)
        if target is not None and _contracts_compatible(value, target):
            value.shape = target.shape

    decoder_outputs = {value.name: value for value in decoder.graph.outputs}
    cache_pairs: list[tuple[ir.Value, ir.Value]] = []
    for value in decoder.graph.inputs:
        present = next(
            (
                decoder_outputs.get(name)
                for name in (
                    value.name.replace("past_key_values", "present"),
                    value.name.replace("past.", "present."),
                )
                if name in decoder_outputs
            ),
            None,
        )
        if present is not None:
            if present.shape is None:
                present.shape = value.shape
            cache_pairs.append((value, present))
    cache_names = {value.name for value, _ in cache_pairs}
    rank2_integer = [
        value
        for value in decoder.graph.inputs
        if value.name not in cache_names
        and value.dtype == ir.DataType.INT64
        and value.shape is not None
        and len(value.shape) == 2
    ]
    attention_input = _find_port(rank2_integer, "mask")
    position_input = _find_port(rank2_integer, "position")
    if attention_input is None:
        raise ValueError("VLM decoder requires an attention-mask input")

    legacy = build_native_vlm_package_metadata(pkg, config=config, source=source)
    preprocessing = legacy.get("preprocessing")
    if not preprocessing or "image" not in preprocessing:
        raise ValueError("VLM workflow requires declared image preprocessing")
    _name_image_preprocessing_program(preprocessing["image"])
    image_outputs = preprocessing["image"]["outputs"]
    vision_inputs = {value.name: value for value in vision.graph.inputs}
    adapter_outputs: dict[str, Any] = {}
    preprocessing_values: dict[str, str] = {}
    for output in image_outputs:
        endpoint = output["name"]
        port_name = endpoint.split(".", 1)[-1]
        if port_name not in vision_inputs:
            raise ValueError(f"preprocessing output {endpoint!r} has no vision input")
        output["contract"] = _contract(vision_inputs[port_name])
        output["dtype"] = output["contract"]["dtype"]
        output["name"] = f"image.{port_name}"
        adapter_outputs[port_name] = output["contract"]
        preprocessing_values[port_name] = output["name"]

    attach_policy_components(
        pkg,
        PolicyCapabilities(
            sampler="greedy",
            eos_termination=True,
            token_state_update=True,
        ),
    )
    pkg.add_policy_component("last_token_logits", build_last_token_logits())
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    pkg.add_policy_component(
        "decoder_state_initializer",
        build_decoder_state_initializer(
            decoder,
            token_input=None,
            prompt_dtype=token_input.dtype,
            attention_mask_input=attention_input.name,
            position_ids_input=position_input.name if position_input is not None else None,
            cache_inputs=sorted(cache_names),
        ),
    )
    pkg.add_policy_component(
        "decoder_step_update",
        build_decoder_step_update(
            attention_dtype=attention_input.dtype,
            position_dtype=position_input.dtype if position_input is not None else None,
        ),
    )

    batch = _contract(token_input)["shape"][0]
    batch_int = {"dtype": "int64", "rank": 1, "shape": [batch]}
    eos = getattr(config, "eos_token_id", 0)
    if isinstance(eos, list):
        eos = eos[0] if eos else 0
    inputs: dict[str, Any] = {
        "request.prompt_tokens": {
            "contract": _contract(token_input),
            "role": {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"},
            "source": {"kind": "request", "field": "prompt_tokens"},
            "required": True,
        },
        "request.image": {
            "contract": {"dtype": "uint8", "rank": 1, "shape": ["encoded_bytes"]},
            "role": {"kind": "runtime", "version": "1.0", "role": "media"},
            "source": {"kind": "request", "field": "media"},
            "required": True,
        },
        "request.max_iterations": {
            "contract": batch_int,
            "role": {
                "kind": "runtime",
                "version": "1.0",
                "role": "max_output_tokens",
            },
            "source": {"kind": "request", "field": "max_output_tokens"},
            "required": True,
        },
        "package.eos_ids": {
            "contract": {"dtype": "int64", "rank": 1, "shape": [1]},
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": int(eos or 0),
        },
        "package.max_context": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": int(getattr(config, "max_position_embeddings", 4096)),
        },
        "package.one": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
    }
    vision_invoke_inputs = {
        name: preprocessing_values[name]
        for name in vision_inputs
        if name in preprocessing_values
    }
    vision_outputs = {value.name: f"vision.{value.name}" for value in vision.graph.outputs}
    audio_setup_nodes: list[dict[str, Any]] = []
    audio_outputs: dict[str, str] = {}
    if "audio_encoder" in pkg:
        audio = pkg["audio_encoder"]
        audio_inputs = {}
        for value in audio.graph.inputs:
            name = f"request.audio.{value.name}"
            inputs[name] = {
                "contract": _contract(value),
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": f"audio.{value.name}"},
                "required": True,
            }
            audio_inputs[value.name] = name
        audio_outputs = {value.name: f"audio.{value.name}" for value in audio.graph.outputs}
        audio_setup_nodes.append(_invoke("audio_encoder", audio_inputs, audio_outputs))
    embedding_setup_inputs: dict[str, str] = {token_input.name: "request.prompt_tokens"}
    embedding_body_inputs: dict[str, str] = {token_input.name: "token.body"}
    produced_features = {value.name: f"vision.{value.name}" for value in vision.graph.outputs}
    produced_features.update(audio_outputs)
    for value in embedding.graph.inputs:
        if value is token_input:
            continue
        if value.name in produced_features:
            embedding_setup_inputs[value.name] = produced_features[value.name]
            embedding_body_inputs[value.name] = produced_features[value.name]
        else:
            name = f"request.{value.name}"
            inputs[name] = {
                "contract": _contract(value),
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": value.name},
                "required": True,
            }
            embedding_setup_inputs[value.name] = name
            embedding_body_inputs[value.name] = name

    setup_decoder_inputs = {
        decoder_embed_input.name: "embedding.setup.embeds",
        attention_input.name: f"initializer.{attention_input.name}",
    }
    body_decoder_inputs = {
        decoder_embed_input.name: "embedding.body.embeds",
        attention_input.name: "state.attention_mask.body",
    }
    if position_input is not None:
        setup_decoder_inputs[position_input.name] = f"initializer.{position_input.name}"
        body_decoder_inputs[position_input.name] = "state.position_ids.body"
    for past, _ in cache_pairs:
        setup_decoder_inputs[past.name] = f"initializer.{past.name}"
        body_decoder_inputs[past.name] = f"state.{past.name}.body"

    setup_decoder_outputs = {logits_output.name: "decoder.setup.logits"}
    body_decoder_outputs = {logits_output.name: "decoder.body.logits"}
    logits_contract = _contract(logits_output)
    last_logits_contract = {
        "dtype": logits_contract["dtype"],
        "rank": 2,
        "shape": [logits_contract["shape"][0], logits_contract["shape"][-1]],
    }
    state: dict[str, Any] = {
        "token": {
            "contract": {"dtype": "int64", "rank": 2, "shape": [batch, 1]},
            "scope": "invocation",
            "initializer": "initializer.token_slot",
            "recurrence": {"kind": "invariant"},
        },
        "logits": {
            "contract": last_logits_contract,
            "scope": "invocation",
            "initializer": "decoder.setup.last_logits",
            "recurrence": {"kind": "invariant"},
        },
        "attention_mask": {
            "contract": {
                "dtype": _contract(attention_input)["dtype"],
                "rank": 2,
                "shape": [batch, "context"],
            },
            "scope": "invocation",
            "initializer": "initializer.body_attention_mask",
            "recurrence": {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one",
                "max": "package.max_context",
            },
        },
    }
    state_specs = [
        (
            "token",
            "initializer.token_slot",
            "state.token.body",
            "token.body",
            "state.token.final",
        ),
        (
            "logits",
            "decoder.setup.last_logits",
            "state.logits.body",
            "decoder.body.last_logits",
            "state.logits.final",
        ),
        (
            "attention_mask",
            "initializer.body_attention_mask",
            "state.attention_mask.body",
            "decoder_step.body_attention_mask",
            "state.attention_mask.final",
        ),
    ]
    if position_input is not None:
        state["position_ids"] = {
            "contract": {
                "dtype": _contract(position_input)["dtype"],
                "rank": 2,
                "shape": [batch, 1],
            },
            "scope": "invocation",
            "initializer": "initializer.body_position_ids",
            "recurrence": {"kind": "invariant"},
        }
        state_specs.append(
            (
                "position_ids",
                "initializer.body_position_ids",
                "state.position_ids.body",
                "decoder_step.body_position_ids",
                "state.position_ids.final",
            )
        )
    for index, (past, present) in enumerate(cache_pairs):
        cell = f"cache_{index}"
        state[cell] = {
            "contract": _contract(past),
            "scope": "invocation",
            "initializer": f"decoder.setup.{present.name}",
            "recurrence": {
                "kind": "growing",
                "axis": next(
                    (
                        axis
                        for axis, dimension in enumerate(_contract(past)["shape"])
                        if "sequence" in str(dimension)
                    ),
                    2,
                ),
                "increment": "package.one",
                "max": "package.max_context",
            },
        }
        setup_decoder_outputs[present.name] = f"decoder.setup.{present.name}"
        body_decoder_outputs[present.name] = f"decoder.body.{present.name}"
        state_specs.append(
            (
                cell,
                f"decoder.setup.{present.name}",
                f"state.{past.name}.body",
                f"decoder.body.{present.name}",
                f"state.{past.name}.final",
            )
        )
    carried = []
    initial_effects = {
        "sample": "sample.0",
        "termination": "termination.0",
        "state": "state.0",
        "emit": "emit.0",
    }
    for cell, current, body_input, body_output, final in state_specs:
        effect = f"state:{cell}"
        initial_effects[effect] = f"{effect}.0"
        carried.append(
            {
                "cell": cell,
                "current": current,
                "body_input": body_input,
                "body_output": body_output,
                "next": final,
                "read_effect": _effect(f"{effect}.0", f"{effect}.read"),
                "write_effect": _effect(f"{effect}.read", f"{effect}.1"),
            }
        )

    setup = {
        "kind": "sequence",
        "nodes": [
            _invoke(
                "image_preprocess",
                {"encoded": "request.image"},
                dict(preprocessing_values),
            ),
            _invoke("vision_encoder", vision_invoke_inputs, vision_outputs),
            *audio_setup_nodes,
            _invoke(
                "decoder_state_initializer",
                {"prompt_tokens": "request.prompt_tokens"},
                {
                    attention_input.name: f"initializer.{attention_input.name}",
                    "body_attention_mask": "initializer.body_attention_mask",
                    "token_slot": "initializer.token_slot",
                    **(
                        {
                            position_input.name: f"initializer.{position_input.name}",
                            "body_position_ids": "initializer.body_position_ids",
                        }
                        if position_input is not None
                        else {}
                    ),
                    **{name: f"initializer.{name}" for name in sorted(cache_names)},
                },
            ),
            _invoke(
                "embedding",
                embedding_setup_inputs,
                {embedding_output.name: "embedding.setup.embeds"},
            ),
            _invoke("decoder", setup_decoder_inputs, setup_decoder_outputs),
            _invoke(
                "last_token_logits",
                {"logits": "decoder.setup.logits"},
                {"last_logits": "decoder.setup.last_logits"},
            ),
        ],
    }
    body = {
        "kind": "sequence",
        "nodes": [
            _invoke(
                "token_sampler",
                {"logits": "state.logits.body"},
                {"token": "sample.body"},
                {"sample": _effect("sample.0", "sample.1")},
            ),
            _invoke(
                "termination",
                {
                    "token_ids": "sample.body",
                    "eos_ids": "package.eos_ids",
                    "iteration": "loop.iteration",
                    "max_iterations": "request.max_iterations",
                },
                {"done": "loop.done"},
                {"termination": _effect("termination.0", "termination.1")},
            ),
            _invoke(
                "continue_predicate",
                {"done": "loop.done"},
                {"continue": "loop.continue"},
            ),
            {
                "kind": "emit",
                "value": "sample.body",
                "output": "tokens",
                "mode": "append",
                "effect_name": "emit",
                "effect": _effect("emit.0", "emit.1"),
            },
            _invoke(
                "token_state_update",
                {"current": "state.token.body", "update": "sample.body"},
                {"next": "token.body"},
                {"state": _effect("state.0", "state.1")},
            ),
            _invoke(
                "embedding",
                embedding_body_inputs,
                {embedding_output.name: "embedding.body.embeds"},
            ),
            _invoke("decoder", body_decoder_inputs, body_decoder_outputs),
            _invoke(
                "last_token_logits",
                {"logits": "decoder.body.logits"},
                {"last_logits": "decoder.body.last_logits"},
            ),
            _invoke(
                "decoder_step_update",
                {
                    "attention_mask": "state.attention_mask.body",
                    **(
                        {"position_ids": "state.position_ids.body"}
                        if position_input is not None
                        else {}
                    ),
                },
                {
                    "next_attention_mask": "decoder_step.body_attention_mask",
                    **(
                        {"next_position_ids": "decoder_step.body_position_ids"}
                        if position_input is not None
                        else {}
                    ),
                },
            ),
        ],
    }
    components = {
        name: _component(model, _artifact(name, len(pkg))) for name, model in pkg.items()
    }
    components["image_preprocess"] = {
        "implementation": {
            "kind": "adapter",
            "abi": "onnx-genai.image-preprocess",
            "version": "1",
        },
        "ports": {
            "inputs": {
                "encoded": {
                    "dtype": "uint8",
                    "rank": 1,
                    "shape": ["encoded_bytes"],
                }
            },
            "outputs": adapter_outputs,
        },
    }
    workflow = {
        "manifest": {
            "ir_version": "1.0",
            "onnx_opsets": {"ai.onnx": OPSET_VERSION},
            "adapter_abis": {"onnx-genai.image-preprocess": "1"},
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
            ],
        },
        "inputs": inputs,
        "outputs": {
            "tokens": {"contract": batch_int, "role": "tokens", "stage": "pre_adapter"}
        },
        "components": components,
        "state": state,
        "initial_effects": initial_effects,
        "graph": {
            "kind": "loop",
            "setup": setup,
            "body": body,
            "condition": "loop.continue",
            "max_iterations": "request.max_iterations",
            "iteration": {"value": "loop.iteration", "contract": batch_int},
            "carried": carried,
        },
    }
    metadata = {
        "schema_version": "v1",
        "preprocessing": preprocessing,
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def write_vlm_workflow_metadata(
    pkg: Any,
    output_dir: str,
    config: Any,
    *,
    source: str | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_vlm_workflow_metadata(pkg, config, source=source)
    pkg.save_policy_components(output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def build_speculative_workflow_metadata(
    pkg: Any,
    config: Any | None = None,
    *,
    grammar_guidance: bool = False,
    adaptive_k_max: int | None = None,
) -> dict[str, Any]:
    """Build proposer/verifier workflow with branch phi and effect joins."""
    config = config or getattr(pkg, "config", None)
    if not {"proposer", "verifier"} <= set(pkg.keys()):
        raise ValueError("speculative workflow requires proposer and verifier")
    proposer = pkg["proposer"]
    verifier = pkg["verifier"]
    proposer_input = next(
        (
            value
            for value in proposer.graph.inputs
            if value.dtype in {ir.DataType.INT32, ir.DataType.INT64}
            and value.shape is not None
            and len(value.shape) == 2
        ),
        None,
    )
    proposed_tokens = _find_port(proposer.graph.outputs, "proposed", "tokens")
    proposal_scores = _find_port(proposer.graph.outputs, "scores", "logits")
    verifier_token_input = next(
        (
            value
            for value in verifier.graph.inputs
            if proposed_tokens is not None and _contract(value) == _contract(proposed_tokens)
        ),
        None,
    )
    target_scores = _find_port(verifier.graph.outputs, "scores", "logits")
    if None in (proposer_input, proposed_tokens, verifier_token_input, target_scores):
        raise ValueError("speculative components do not expose compatible token/score ports")
    assert proposer_input is not None
    assert proposed_tokens is not None
    assert verifier_token_input is not None
    assert target_scores is not None
    proposal_budget_input = next(
        (
            value
            for value in proposer.graph.inputs
            if value is not proposer_input
            and value.dtype == ir.DataType.INT64
            and value.shape is not None
            and len(value.shape) == 1
            and any(term in value.name.lower() for term in ("budget", "proposal_k", "draft_k"))
        ),
        None,
    )
    if adaptive_k_max is not None and proposal_budget_input is None:
        raise ValueError(
            "adaptive speculative workflow requires a rank-1 proposer budget input"
        )
    if _contract(proposer_input) != _contract(proposed_tokens):
        raise ValueError(
            "representative speculative workflow requires fixed token-block shape"
        )

    attach_policy_components(
        pkg,
        PolicyCapabilities(
            speculative_acceptance=True,
            grammar_guidance=grammar_guidance,
            adaptive_k_max=adaptive_k_max,
        ),
    )
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    pkg.add_policy_component("branch_state", build_token_block_identity())
    if grammar_guidance:
        pkg.add_policy_component("grammar_length", build_integer_minimum())
        pkg.add_policy_component("grammar_emit_length", build_batch_minimum())
        pkg.add_policy_component("grammar_rollback_length", build_integer_minimum())
        pkg.add_policy_component("grammar_sampler_logits", build_last_token_logits())
        if adaptive_k_max is None:
            pkg.add_policy_component("proposal_length", build_sequence_length())
    if adaptive_k_max is not None:
        pkg.add_policy_component("proposal_metrics", build_proposal_metrics())
    batch = _contract(proposer_input)["shape"][0]
    batch_int = {"dtype": "int64", "rank": 1, "shape": [batch]}
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}
    control_bool = {"dtype": "bool", "rank": 1, "shape": [1]}
    inputs: dict[str, Any] = {
        "request.tokens": {
            "contract": _contract(proposer_input),
            "role": {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"},
            "source": {"kind": "request", "field": "prompt_tokens"},
            "required": True,
        },
        "request.seed": {
            "contract": batch_int,
            "role": {"kind": "runtime", "version": "1.0", "role": "seed"},
            "source": {"kind": "request", "field": "seed"},
            "required": False,
            "default": 0,
        },
        "request.max_iterations": {
            "contract": control_int,
            "role": {
                "kind": "runtime",
                "version": "1.0",
                "role": "max_output_tokens",
            },
            "source": {"kind": "request", "field": "max_output_tokens"},
            "required": True,
        },
        "package.zero": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0,
        },
        "package.one": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.max_context": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": int(getattr(config, "max_position_embeddings", 4096)),
        },
        "package.false": {
            "contract": control_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": False,
        },
    }
    if grammar_guidance:
        inputs.update(
            {
                "request.grammar_state": {
                    "contract": batch_int,
                    "role": {"kind": "opaque"},
                    "source": {
                        "kind": "application",
                        "name": "grammar.initial_state",
                    },
                    "required": True,
                },
                "request.grammar_transition_table": {
                    "contract": {
                        "dtype": "int64",
                        "rank": 2,
                        "shape": ["grammar_states", "vocabulary"],
                    },
                    "role": {"kind": "opaque"},
                    "source": {
                        "kind": "application",
                        "name": "grammar.transition_table",
                    },
                    "required": True,
                },
            }
        )
    if adaptive_k_max is not None:
        estimate_slots = 4 * (adaptive_k_max + 1) + 4
        inputs.update(
            {
                "request.adaptive_k": {
                    "contract": batch_int,
                    "role": {"kind": "opaque"},
                    "source": {"kind": "application", "name": "adaptive.current_k"},
                    "required": False,
                    "default": 1,
                },
                "request.adaptive_estimates": {
                    "contract": {
                        "dtype": "float32",
                        "rank": 2,
                        "shape": [batch, estimate_slots],
                    },
                    "role": {"kind": "opaque"},
                    "source": {
                        "kind": "application",
                        "name": "adaptive.estimates",
                    },
                    "required": False,
                    "default": 0.0,
                },
                "request.draft_ms": {
                    "contract": {
                        "dtype": "float32",
                        "rank": 1,
                        "shape": [batch],
                    },
                    "role": {"kind": "opaque"},
                    "source": {"kind": "application", "name": "telemetry.draft_ms"},
                    "required": True,
                },
                "request.target_ms": {
                    "contract": {
                        "dtype": "float32",
                        "rank": 1,
                        "shape": [batch],
                    },
                    "role": {"kind": "opaque"},
                    "source": {"kind": "application", "name": "telemetry.target_ms"},
                    "required": True,
                },
            }
        )

    proposer_inputs = {proposer_input.name: "state.tokens.body"}
    for value in proposer.graph.inputs:
        if value is proposer_input:
            continue
        if value is proposal_budget_input:
            proposer_inputs[value.name] = "state.proposal_k.body"
            continue
        name = f"request.proposer.{value.name}"
        inputs[name] = {
            "contract": _contract(value),
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": f"proposer.{value.name}"},
            "required": True,
        }
        proposer_inputs[value.name] = name
    verifier_inputs = {verifier_token_input.name: "proposal.tokens"}
    verifier_outputs = {target_scores.name: "target.scores"}
    cache_pairs: list[tuple[ir.Value, ir.Value]] = []
    verifier_output_map = {value.name: value for value in verifier.graph.outputs}
    for value in verifier.graph.inputs:
        if value is verifier_token_input:
            continue
        present = next(
            (
                verifier_output_map.get(name)
                for name in (
                    value.name.replace("past_key_values", "present"),
                    value.name.replace("past.", "present."),
                )
                if name in verifier_output_map
            ),
            None,
        )
        if present is not None:
            cell = f"cache_{len(cache_pairs)}"
            cache_pairs.append((value, present))
            name = f"request.verifier.{value.name}"
            inputs[name] = {
                "contract": _contract(value),
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": f"verifier.{value.name}"},
                "required": True,
            }
            verifier_inputs[value.name] = f"state.{cell}.body"
            verifier_outputs[present.name] = f"verifier.{present.name}"
        else:
            name = f"request.verifier.{value.name}"
            inputs[name] = {
                "contract": _contract(value),
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": f"verifier.{value.name}"},
                "required": True,
            }
            verifier_inputs[value.name] = name

    proposal_outputs = {proposed_tokens.name: "proposal.tokens"}
    acceptance_inputs = {
        "target_scores": "target.scores",
        "proposed_tokens": "proposal.tokens",
        "seed": "request.seed",
        "offset": "state.rng_offset.body",
    }
    if proposal_scores is not None:
        proposal_outputs[proposal_scores.name] = "proposal.scores"
    proposal_measure_nodes: list[dict[str, Any]] = []
    if adaptive_k_max is not None:
        proposal_measure_nodes.append(
            _invoke(
                "proposal_metrics",
                {
                    "proposed_tokens": "proposal.tokens",
                    "requested_k": "state.proposal_k.body",
                },
                {
                    "evaluated": "proposal.evaluated",
                    "filled_proposal_budget": "proposal.filled_budget",
                },
            )
        )
    elif grammar_guidance:
        proposal_measure_nodes.append(
            _invoke(
                "proposal_length",
                {"tokens": "proposal.tokens"},
                {"length": "proposal.evaluated"},
            )
        )
    grammar_pre_nodes: list[dict[str, Any]] = []
    if grammar_guidance:
        grammar_pre_nodes.extend(
            [
                _invoke(
                    "grammar_clone",
                    {
                        "state": "state.grammar.body",
                        "tokens": "proposal.tokens",
                        "valid_length": "package.zero",
                        "transition_table": "request.grammar_transition_table",
                    },
                    {
                        "next_state": "grammar.clone.state",
                        "consumed_length": "grammar.clone.consumed",
                        "logits_mask": "grammar.clone.mask",
                        "forced_tokens": "grammar.clone.forced",
                        "forced_length": "grammar.clone.forced_length",
                    },
                    {"grammar": _effect("grammar.0", "grammar.clone")},
                ),
                _invoke(
                    "grammar_lookahead",
                    {
                        "state": "grammar.clone.state",
                        "tokens": "proposal.tokens",
                        "valid_length": "proposal.evaluated",
                        "transition_table": "request.grammar_transition_table",
                    },
                    {
                        "next_state": "grammar.lookahead.state",
                        "consumed_length": "grammar.valid_length",
                        "logits_mask": "grammar.lookahead.mask",
                        "forced_tokens": "grammar.lookahead.forced",
                        "forced_length": "grammar.lookahead.forced_length",
                    },
                    {"grammar": _effect("grammar.clone", "grammar.lookahead")},
                ),
            ]
        )
    emit_length = "acceptance.synchronized_length"
    cache_rollback_length = "acceptance.rollback_length"
    grammar_post_nodes: list[dict[str, Any]] = []
    if grammar_guidance:
        emit_length = "grammar.synchronized_length"
        cache_rollback_length = "grammar.rollback_length"
        grammar_post_nodes.extend(
            [
                _invoke(
                    "grammar_length",
                    {
                        "left": "acceptance.length",
                        "right": "grammar.valid_length",
                    },
                    {"minimum": "grammar.committed_length"},
                ),
                _invoke(
                    "grammar_rollback_length",
                    {
                        "left": "acceptance.rollback_length",
                        "right": "grammar.valid_length",
                    },
                    {"minimum": "grammar.rollback_length"},
                ),
                _invoke(
                    "grammar_emit_length",
                    {"values": "grammar.committed_length"},
                    {"minimum": "grammar.synchronized_length"},
                ),
                _invoke(
                    "grammar_commit",
                    {
                        "state": "state.grammar.body",
                        "tokens": "acceptance.tokens",
                        "valid_length": "grammar.committed_length",
                        "transition_table": "request.grammar_transition_table",
                    },
                    {
                        "next_state": "grammar.next",
                        "consumed_length": "grammar.committed",
                        "logits_mask": "grammar.mask",
                        "forced_tokens": "grammar.forced",
                        "forced_length": "grammar.forced_length",
                    },
                    {"grammar": _effect("grammar.lookahead", "grammar.commit")},
                ),
                _invoke(
                    "grammar_sampler_logits",
                    {"logits": "target.scores"},
                    {"last_logits": "grammar.sampler_logits"},
                ),
                _invoke(
                    "grammar_guidance",
                    {
                        "logits": "grammar.sampler_logits",
                        "logits_mask": "grammar.mask",
                        "forced_tokens": "grammar.forced",
                        "forced_length": "grammar.forced_length",
                    },
                    {"token": "grammar.token"},
                ),
            ]
        )
    adaptive_nodes: list[dict[str, Any]] = []
    if adaptive_k_max is not None:
        committed_metric = (
            "grammar.committed_length" if grammar_guidance else "acceptance.length"
        )
        adaptive_nodes.append(
            _invoke(
                "adaptive_k",
                {
                    "current_k": "state.proposal_k.body",
                    "accepted": committed_metric,
                    "evaluated": "proposal.evaluated",
                    "committed_tokens": committed_metric,
                    "filled_proposal_budget": "proposal.filled_budget",
                    "draft_ms": "request.draft_ms",
                    "target_ms": "request.target_ms",
                    "estimates": "state.adaptive_estimates.body",
                },
                {
                    "next_k": "adaptive.next_k",
                    "next_estimates": "adaptive.next_estimates",
                },
                {"adaptive": _effect("adaptive.0", "adaptive.1")},
            )
        )
    rollback_nodes: list[dict[str, Any]] = []
    accepted_case_nodes = [
        _invoke(
            "branch_state",
            {"tokens": "acceptance.tokens"},
            {"next_tokens": "branch.accepted"},
            {"state": _effect("branch.state.in", "branch.state.accepted")},
        )
    ]
    corrected_case_nodes = [
        _invoke(
            "branch_state",
            {"tokens": "acceptance.tokens"},
            {"next_tokens": "branch.corrected"},
            {"state": _effect("branch.state.in", "branch.state.corrected")},
        )
    ]
    branch_outputs: dict[str, Any] = {
        "tokens.next": {
            "cases": {
                "true": "branch.accepted",
                "false": "branch.corrected",
            }
        }
    }
    branch_effects: dict[str, Any] = {
        "state": {
            "incoming": "branch.state.in",
            "cases": {
                "true": "branch.state.accepted",
                "false": "branch.state.corrected",
            },
            "produces": "branch.state.out",
        }
    }
    for index, (past, present) in enumerate(cache_pairs):
        cache_name = f"cache_{index}"
        cache_contract = _contract(past)
        sequence_axis = next(
            (
                axis
                for axis, dimension in enumerate(cache_contract["shape"])
                if "sequence" in str(dimension)
            ),
            2,
        )
        rollback_name = f"rollback_{cache_name}"
        publisher_name = f"publish_{cache_name}"
        branch_effect = f"branch:{cache_name}"
        pkg.add_policy_component(
            rollback_name,
            build_speculative_state_rollback(
                past.dtype,
                cache_contract["shape"],
                sequence_axis=sequence_axis,
                effect=rollback_name,
            ),
        )
        pkg.add_policy_component(
            publisher_name,
            build_effectful_identity(
                publisher_name,
                past.dtype,
                [
                    "branch_sequence" if axis == sequence_axis else dimension
                    for axis, dimension in enumerate(cache_contract["shape"])
                ],
                effect=branch_effect,
            ),
        )
        rollback_nodes.append(
            _invoke(
                rollback_name,
                {
                    "past_state": f"state.{cache_name}.body",
                    "tentative_state": f"verifier.{present.name}",
                    "accepted_len": cache_rollback_length,
                },
                {"corrected_state": f"rollback.{cache_name}"},
                {
                    rollback_name: _effect(
                        f"rollback.{cache_name}.0",
                        f"rollback.{cache_name}.1",
                    )
                },
            )
        )
        accepted_case_nodes.append(
            _invoke(
                publisher_name,
                {"value": f"verifier.{present.name}"},
                {"next_value": f"branch.accepted.{cache_name}"},
                {
                    branch_effect: _effect(
                        f"branch.{cache_name}.in",
                        f"branch.{cache_name}.accepted",
                    )
                },
            )
        )
        corrected_case_nodes.append(
            _invoke(
                publisher_name,
                {"value": f"rollback.{cache_name}"},
                {"next_value": f"branch.corrected.{cache_name}"},
                {
                    branch_effect: _effect(
                        f"branch.{cache_name}.in",
                        f"branch.{cache_name}.corrected",
                    )
                },
            )
        )
        branch_outputs[f"{cache_name}.next"] = {
            "cases": {
                "true": f"branch.accepted.{cache_name}",
                "false": f"branch.corrected.{cache_name}",
            }
        }
        branch_effects[branch_effect] = {
            "incoming": f"branch.{cache_name}.in",
            "cases": {
                "true": f"branch.{cache_name}.accepted",
                "false": f"branch.{cache_name}.corrected",
            },
            "produces": f"branch.{cache_name}.out",
        }
    branch = {
        "kind": "branch",
        "predicate": "acceptance.synchronized_done",
        "cases": {
            "true": {"kind": "sequence", "nodes": accepted_case_nodes},
            "false": {"kind": "sequence", "nodes": corrected_case_nodes},
        },
        "outputs": branch_outputs,
        "effects": branch_effects,
    }
    body_nodes = [
        _invoke("proposer", proposer_inputs, proposal_outputs),
        *proposal_measure_nodes,
        *grammar_pre_nodes,
        _invoke("verifier", verifier_inputs, verifier_outputs),
        _invoke(
            "speculative_acceptance",
            acceptance_inputs,
            {
                "accepted_tokens": "acceptance.tokens",
                "accepted_len": "acceptance.length",
                "done": "acceptance.done",
                "next_offset": "rng_offset.body",
                "synchronized_len": "acceptance.synchronized_length",
                "synchronized_done": "acceptance.synchronized_done",
                "rollback_len": "acceptance.rollback_length",
            },
            {"verify": _effect("verify.0", "verify.1")},
        ),
        *grammar_post_nodes,
        *adaptive_nodes,
        *rollback_nodes,
        branch,
        _invoke(
            "continue_predicate",
            {"done": "package.false"},
            {"continue": "speculative.continue"},
        ),
        {
            "kind": "emit",
            "value": "tokens.next",
            "valid_length": emit_length,
            "output": "tokens",
            "mode": "append",
            "effect_name": "emit",
            "effect": _effect("emit.0", "emit.1"),
        },
    ]
    if grammar_guidance:
        body_nodes.append(
            {
                "kind": "emit",
                "value": "grammar.token",
                "output": "tokens",
                "mode": "append",
                "effect_name": "emit",
                "effect": _effect("emit.1", "emit.2"),
            }
        )
    state = {
        "tokens": {
            "contract": _contract(proposer_input),
            "class": "semantic",
            "scope": "invocation",
            "initializer": "request.tokens",
            "recurrence": {"kind": "invariant"},
        },
        "rng_offset": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.zero",
            "recurrence": {"kind": "invariant"},
        },
    }
    state_specs = [
        (
            "tokens",
            "request.tokens",
            "state.tokens.body",
            "tokens.next",
            "state.tokens.final",
        ),
        (
            "rng_offset",
            "package.zero",
            "state.rng_offset.body",
            "rng_offset.body",
            "state.rng_offset.final",
        ),
    ]
    if grammar_guidance:
        state["grammar"] = {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "request.grammar_state",
            "recurrence": {"kind": "invariant"},
        }
        state_specs.append(
            (
                "grammar",
                "request.grammar_state",
                "state.grammar.body",
                "grammar.next",
                "state.grammar.final",
            )
        )
    if adaptive_k_max is not None:
        state["proposal_k"] = {
            "contract": batch_int,
            "class": "advisory",
            "scope": "invocation",
            "initializer": "request.adaptive_k",
            "recurrence": {"kind": "invariant"},
        }
        state["adaptive_estimates"] = {
            "contract": inputs["request.adaptive_estimates"]["contract"],
            "class": "advisory",
            "scope": "invocation",
            "initializer": "request.adaptive_estimates",
            "recurrence": {"kind": "invariant"},
        }
        state_specs.extend(
            [
                (
                    "proposal_k",
                    "request.adaptive_k",
                    "state.proposal_k.body",
                    "adaptive.next_k",
                    "state.proposal_k.final",
                ),
                (
                    "adaptive_estimates",
                    "request.adaptive_estimates",
                    "state.adaptive_estimates.body",
                    "adaptive.next_estimates",
                    "state.adaptive_estimates.final",
                ),
            ]
        )
    for index, (past, _present) in enumerate(cache_pairs):
        cell = f"cache_{index}"
        initializer = f"request.verifier.{past.name}"
        state[cell] = {
            "contract": _contract(past),
            "class": "semantic",
            "scope": "invocation",
            "initializer": initializer,
            "recurrence": {
                "kind": "bounded",
                "axis": next(
                    (
                        axis
                        for axis, dimension in enumerate(_contract(past)["shape"])
                        if "sequence" in str(dimension)
                    ),
                    2,
                ),
                "max": "package.max_context",
            },
        }
        state_specs.append(
            (
                cell,
                initializer,
                f"state.{cell}.body",
                f"{cell}.next",
                f"state.{cell}.final",
            )
        )
    initial_effects = {
        "verify": "verify.0",
        "emit": "emit.0",
        "state": "branch.state.in",
    }
    if grammar_guidance:
        initial_effects["grammar"] = "grammar.0"
    if adaptive_k_max is not None:
        initial_effects["adaptive"] = "adaptive.0"
    for index in range(len(cache_pairs)):
        initial_effects[f"rollback_cache_{index}"] = f"rollback.cache_{index}.0"
        initial_effects[f"branch:cache_{index}"] = f"branch.cache_{index}.in"
    carried = []
    for cell, current, body_input, body_output, final in state_specs:
        effect = f"state:{cell}"
        initial_effects[effect] = f"{effect}.0"
        carried.append(
            {
                "cell": cell,
                "current": current,
                "body_input": body_input,
                "body_output": body_output,
                "next": final,
                "read_effect": _effect(f"{effect}.0", f"{effect}.read"),
                "write_effect": _effect(f"{effect}.read", f"{effect}.1"),
            }
        )
    workflow = {
        "manifest": {
            "ir_version": "1.0",
            "onnx_opsets": {"ai.onnx": OPSET_VERSION},
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
                "emit_valid_length",
                "bounded_state_recurrence",
            ],
        },
        "inputs": inputs,
        "outputs": {
            "tokens": {
                "contract": {
                    **_contract(proposed_tokens),
                    "shape": [
                        *_contract(proposed_tokens)["shape"][:-1],
                        "accepted_sequence",
                    ],
                },
                "role": "tokens",
                "stage": "pre_adapter",
            },
        },
        "components": {
            name: _component(model, _artifact(name, len(pkg))) for name, model in pkg.items()
        },
        "state": state,
        "initial_effects": initial_effects,
        "graph": {
            "kind": "loop",
            "setup": {
                "kind": "sequence",
                "nodes": [
                    _invoke(
                        "continue_predicate",
                        {"done": "package.false"},
                        {"continue": "speculative.setup.continue"},
                    )
                ],
            },
            "body": {"kind": "sequence", "nodes": body_nodes},
            "condition": "speculative.continue",
            "max_iterations": "request.max_iterations",
            "iteration": {"value": "speculative.iteration", "contract": batch_int},
            "carried": carried,
        },
    }
    if grammar_guidance:
        workflow["manifest"]["adapter_abis"] = {"onnx-genai.grammar-guidance": "1"}
        workflow["manifest"]["capabilities"].append("grammar_guidance_adapter")
        workflow["components"].update(
            {
                "grammar_clone": _grammar_adapter_component("clone"),
                "grammar_lookahead": _grammar_adapter_component("lookahead"),
                "grammar_commit": _grammar_adapter_component("commit"),
            }
        )
    if adaptive_k_max is not None:
        workflow["manifest"]["capabilities"].extend(
            ["adaptive_proposal_budget", "advisory_state"]
        )
    metadata = {
        "schema_version": "v1",
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def write_speculative_workflow_metadata(
    pkg: Any,
    output_dir: str,
    *,
    grammar_guidance: bool = False,
    adaptive_k_max: int | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_speculative_workflow_metadata(
        pkg,
        grammar_guidance=grammar_guidance,
        adaptive_k_max=adaptive_k_max,
    )
    pkg.save_policy_components(output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def build_decoder_workflow_metadata(
    pkg: Any,
    config: Any,
    *,
    sampler: str = "greedy",
) -> dict[str, Any]:
    """Build the exact workflow-policy contract for an autoregressive decoder."""
    if len(pkg) != 1:
        raise ValueError("decoder workflow requires exactly one neural component")
    decoder_name, decoder = next(iter(pkg.items()))
    attach_policy_components(
        pkg,
        PolicyCapabilities(
            sampler=sampler,
            eos_termination=True,
            token_state_update=True,
        ),
    )
    pkg.add_policy_component("last_token_logits", build_last_token_logits())
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    pkg.add_policy_component("iteration_increment", build_integer_increment())

    inputs = list(decoder.graph.inputs)
    outputs = list(decoder.graph.outputs)
    token_input = next(
        (
            value
            for value in inputs
            if ("input_ids" in value.name or "token" in value.name)
            and value.dtype in {ir.DataType.INT32, ir.DataType.INT64}
            and value.shape is not None
            and len(value.shape) == 2
        ),
        next(
            (
                value
                for value in inputs
                if value.dtype in {ir.DataType.INT32, ir.DataType.INT64}
                and value.shape is not None
                and len(value.shape) == 2
            ),
            None,
        ),
    )
    logits_output = next(
        (
            value
            for value in outputs
            if value.dtype
            in {
                ir.DataType.FLOAT,
                ir.DataType.FLOAT16,
                ir.DataType.BFLOAT16,
                ir.DataType.DOUBLE,
            }
            and value.shape is not None
            and len(value.shape) == 3
        ),
        None,
    )
    if token_input is None or logits_output is None:
        raise ValueError(
            "decoder workflow requires rank-2 token input and rank-3 logits output"
        )

    output_by_suffix = {value.name: value for value in outputs}
    cache_pairs: list[tuple[ir.Value, ir.Value]] = []
    for value in inputs:
        candidates = [
            value.name.replace("past_key_values", "present"),
            value.name.replace("past.", "present."),
        ]
        present = next(
            (output_by_suffix.get(name) for name in candidates if name in output_by_suffix),
            None,
        )
        if present is not None:
            cache_pairs.append((value, present))
    cache_names = {past.name for past, _ in cache_pairs}
    integer_rank2 = [
        value
        for value in inputs
        if value is not token_input
        and value.name not in cache_names
        and value.dtype in {ir.DataType.INT32, ir.DataType.INT64}
        and value.shape is not None
        and len(value.shape) == 2
    ]
    attention_input = next(
        (
            value
            for value in integer_rank2
            if "mask" in value.name
            or "past" in str(getattr(list(value.shape)[1], "value", list(value.shape)[1]))
        ),
        None,
    )
    position_input = next(
        (
            value
            for value in integer_rank2
            if value is not attention_input and "position" in value.name
        ),
        next((value for value in integer_rank2 if value is not attention_input), None),
    )
    if attention_input is None:
        raise ValueError(
            "standard decoder workflow requires a derived rank-2 attention-mask input"
        )
    derived_names = cache_names | {attention_input.name}
    if position_input is not None:
        derived_names.add(position_input.name)
    unsupported = [
        value.name
        for value in inputs
        if value is not token_input and value.name not in derived_names
    ]
    if unsupported:
        raise ValueError(f"decoder workflow has unsupported non-request inputs: {unsupported}")
    pkg.add_policy_component(
        "decoder_state_initializer",
        build_decoder_state_initializer(
            decoder,
            token_input=token_input.name,
            attention_mask_input=attention_input.name,
            position_ids_input=position_input.name if position_input is not None else None,
            cache_inputs=sorted(cache_names),
        ),
    )
    pkg.add_policy_component(
        "decoder_step_update",
        build_decoder_step_update(
            attention_dtype=attention_input.dtype,
            position_dtype=position_input.dtype if position_input is not None else None,
        ),
    )
    needs_token_cast = token_input.dtype != ir.DataType.INT64
    if needs_token_cast:
        pkg.add_policy_component("model_token_cast", build_model_token_cast(token_input.dtype))

    workflow_inputs: dict[str, Any] = {}
    setup_decoder_inputs: dict[str, str] = {}
    body_decoder_inputs: dict[str, str] = {}
    for value in inputs:
        if value.name in derived_names:
            continue
        name = f"request.{value.name}"
        if value is token_input:
            role = {
                "kind": "runtime",
                "version": "1.0",
                "role": "prompt_tokens",
            }
            source = {"kind": "request", "field": "prompt_tokens"}
        else:
            role = {"kind": "opaque"}
            source = {"kind": "application", "name": value.name}
        workflow_inputs[name] = {
            "contract": _contract(value),
            "role": role,
            "source": source,
            "required": True,
        }
        setup_decoder_inputs[value.name] = name
        body_decoder_inputs[value.name] = name

    batch_dimension = _shape_metadata(_port(token_input))[0]
    batch_int = {"dtype": "int64", "rank": 1, "shape": [batch_dimension]}
    eos_token_id = getattr(config, "eos_token_id", 0)
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0] if eos_token_id else 0
    eos_token_id = int(eos_token_id or 0)
    workflow_inputs.update(
        {
            "request.max_iterations": {
                "contract": batch_int,
                "role": {
                    "kind": "runtime",
                    "version": "1.0",
                    "role": "max_output_tokens",
                },
                "source": {"kind": "request", "field": "max_output_tokens"},
                "required": True,
            },
            "package.eos_ids": {
                "contract": {"dtype": "int64", "rank": 1, "shape": ["E"]},
                "role": {"kind": "opaque"},
                "source": {"kind": "literal"},
                "required": True,
                "default": eos_token_id,
            },
            "package.zero_iteration": {
                "contract": batch_int,
                "role": {"kind": "opaque"},
                "source": {"kind": "literal"},
                "required": False,
                "default": 0,
            },
            "package.one_token": {
                "contract": batch_int,
                "role": {"kind": "opaque"},
                "source": {"kind": "literal"},
                "required": False,
                "default": 1,
            },
            "package.max_context": {
                "contract": batch_int,
                "role": {"kind": "opaque"},
                "source": {"kind": "literal"},
                "required": False,
                "default": int(getattr(config, "max_position_embeddings", 4096)),
            },
        }
    )
    stochastic_sampler = sampler != "greedy"
    if stochastic_sampler:
        workflow_inputs.update(
            {
                "request.temperature": {
                    "contract": {"dtype": "float32", "rank": 1, "shape": [1]},
                    "role": {
                        "kind": "runtime",
                        "version": "1.0",
                        "role": "sampling_temperature",
                    },
                    "source": {
                        "kind": "request",
                        "field": "sampling_temperature",
                    },
                    "required": False,
                    "default": 1.0,
                },
                "request.top_k": {
                    "contract": {"dtype": "int64", "rank": 1, "shape": [1]},
                    "role": {
                        "kind": "runtime",
                        "version": "1.0",
                        "role": "sampling_top_k",
                    },
                    "source": {"kind": "request", "field": "sampling_top_k"},
                    "required": False,
                    "default": 0,
                },
                "request.top_p": {
                    "contract": {"dtype": "float32", "rank": 1, "shape": [1]},
                    "role": {
                        "kind": "runtime",
                        "version": "1.0",
                        "role": "sampling_top_p",
                    },
                    "source": {"kind": "request", "field": "sampling_top_p"},
                    "required": False,
                    "default": 1.0,
                },
                "request.min_p": {
                    "contract": {"dtype": "float32", "rank": 1, "shape": [1]},
                    "role": {
                        "kind": "runtime",
                        "version": "1.0",
                        "role": "sampling_min_p",
                    },
                    "source": {
                        "kind": "request",
                        "field": "sampling_min_p",
                    },
                    "required": False,
                    "default": 0.0,
                },
                "request.seed": {
                    "contract": batch_int,
                    "role": {
                        "kind": "runtime",
                        "version": "1.0",
                        "role": "seed",
                    },
                    "source": {"kind": "request", "field": "seed"},
                    "required": True,
                },
                "request.grammar_mask": {
                    "contract": {
                        "dtype": "bool",
                        "rank": 2,
                        "shape": [
                            batch_dimension,
                            _contract(logits_output)["shape"][-1],
                        ],
                    },
                    "role": {"kind": "opaque"},
                    "source": {"kind": "application", "name": "grammar_mask"},
                    "required": True,
                },
                "request.rng_offset": {
                    "contract": batch_int,
                    "role": {"kind": "opaque"},
                    "source": {"kind": "application", "name": "rng_offset"},
                    "required": False,
                    "default": 0,
                },
            }
        )

    for value in inputs:
        if value is token_input:
            continue
        if value.name in cache_names:
            body_decoder_inputs[value.name] = f"state.{value.name}.body"
            setup_decoder_inputs[value.name] = f"initializer.{value.name}"
    setup_decoder_inputs[attention_input.name] = f"initializer.{attention_input.name}"
    body_decoder_inputs[attention_input.name] = "state.attention_mask.body"
    if position_input is not None:
        setup_decoder_inputs[position_input.name] = f"initializer.{position_input.name}"
        body_decoder_inputs[position_input.name] = "state.position_ids.body"
    body_decoder_inputs[token_input.name] = (
        "model_token.body" if needs_token_cast else "token.body"
    )

    setup_decoder_outputs = {logits_output.name: "decoder.setup.logits"}
    body_decoder_outputs = {logits_output.name: "decoder.body.logits"}
    logits_contract = _contract(logits_output)
    last_logits_contract = {
        "dtype": logits_contract["dtype"],
        "rank": 2,
        "shape": [logits_contract["shape"][0], logits_contract["shape"][-1]],
    }
    state: dict[str, Any] = {
        "token": {
            "contract": {
                "dtype": "int64",
                "rank": 2,
                "shape": [batch_dimension, 1],
            },
            "scope": "invocation",
            "initializer": "initializer.token_slot",
            "recurrence": {"kind": "invariant"},
        },
        "iteration": {
            "contract": batch_int,
            "scope": "invocation",
            "initializer": "package.zero_iteration",
            "recurrence": {"kind": "invariant"},
        },
        "logits": {
            "contract": last_logits_contract,
            "scope": "invocation",
            "initializer": "decoder.setup.last_logits",
            "recurrence": {"kind": "invariant"},
        },
    }
    initial_effects = {
        "sample": "sample.0",
        "termination": "termination.0",
        "state": "state.0",
        "emit": "emit.0",
        "state:token": "state:token.0",
        "state:iteration": "state:iteration.0",
        "state:logits": "state:logits.0",
    }
    carried = [
        {
            "cell": "token",
            "current": "initializer.token_slot",
            "body_input": "state.token.body",
            "body_output": "token.body",
            "next": "token.final",
            "read_effect": _effect("state:token.0", "state:token.read"),
            "write_effect": _effect("state:token.read", "state:token.1"),
        }
    ]
    carried.extend(
        [
            {
                "cell": "iteration",
                "current": "package.zero_iteration",
                "body_input": "state.iteration.body",
                "body_output": "iteration.body",
                "next": "state.iteration.final",
                "read_effect": _effect("state:iteration.0", "state:iteration.read"),
                "write_effect": _effect("state:iteration.read", "state:iteration.1"),
            },
            {
                "cell": "logits",
                "current": "decoder.setup.last_logits",
                "body_input": "state.logits.body",
                "body_output": "decoder.body.last_logits",
                "next": "state.logits.final",
                "read_effect": _effect("state:logits.0", "state:logits.read"),
                "write_effect": _effect("state:logits.read", "state:logits.1"),
            },
        ]
    )
    if stochastic_sampler:
        state["rng_offset"] = {
            "contract": batch_int,
            "scope": "invocation",
            "class": "semantic",
            "initializer": "request.rng_offset",
            "recurrence": {"kind": "invariant"},
        }
        initial_effects["state:rng_offset"] = "state:rng_offset.0"
        carried.append(
            {
                "cell": "rng_offset",
                "current": "request.rng_offset",
                "body_input": "state.rng_offset.body",
                "body_output": "sample.next_offset",
                "next": "state.rng_offset.final",
                "read_effect": _effect("state:rng_offset.0", "state:rng_offset.read"),
                "write_effect": _effect("state:rng_offset.read", "state:rng_offset.1"),
            }
        )
    decoder_state_specs = {
        "attention_mask": (
            {
                "dtype": _contract(attention_input)["dtype"],
                "rank": 2,
                "shape": [batch_dimension, "context"],
            },
            "initializer.body_attention_mask",
            "decoder_step.body_attention_mask",
            {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one_token",
                "max": "package.max_context",
            },
        ),
    }
    if position_input is not None:
        decoder_state_specs["position_ids"] = (
            {
                "dtype": _contract(position_input)["dtype"],
                "rank": 2,
                "shape": [batch_dimension, 1],
            },
            "initializer.body_position_ids",
            "decoder_step.body_position_ids",
            {"kind": "invariant"},
        )
    for cell, (contract, current, body_output, recurrence) in decoder_state_specs.items():
        effect_name = f"state:{cell}"
        initial_effects[effect_name] = f"{effect_name}.0"
        state[cell] = {
            "contract": contract,
            "scope": "invocation",
            "initializer": current,
            "recurrence": recurrence,
        }
        carried.append(
            {
                "cell": cell,
                "current": current,
                "body_input": f"state.{cell}.body",
                "body_output": body_output,
                "next": f"state.{cell}.final",
                "read_effect": _effect(f"{effect_name}.0", f"{effect_name}.read"),
                "write_effect": _effect(f"{effect_name}.read", f"{effect_name}.1"),
            }
        )
    for past, present in cache_pairs:
        cell = f"cache_{len(carried)}"
        setup_value = f"decoder.setup.{present.name}"
        body_value = f"decoder.body.{present.name}"
        setup_decoder_outputs[present.name] = setup_value
        body_decoder_outputs[present.name] = body_value
        state[cell] = {
            "contract": _contract(past),
            "scope": "invocation",
            "initializer": setup_value,
            "recurrence": {
                "kind": "growing",
                "axis": next(
                    (
                        index
                        for index, dimension in enumerate(_contract(past)["shape"])
                        if "sequence" in str(dimension)
                    ),
                    2,
                ),
                "increment": "package.one_token",
                "max": "package.max_context",
            },
        }
        effect_name = f"state:{cell}"
        initial_effects[effect_name] = f"{effect_name}.0"
        carried.append(
            {
                "cell": cell,
                "current": setup_value,
                "body_input": f"state.{past.name}.body",
                "body_output": body_value,
                "next": f"state.{past.name}.final",
                "read_effect": _effect(f"{effect_name}.0", f"{effect_name}.read"),
                "write_effect": _effect(f"{effect_name}.read", f"{effect_name}.1"),
            }
        )

    setup = {
        "kind": "sequence",
        "nodes": [
            _invoke(
                "decoder_state_initializer",
                {"prompt_tokens": f"request.{token_input.name}"},
                {
                    attention_input.name: f"initializer.{attention_input.name}",
                    "body_attention_mask": "initializer.body_attention_mask",
                    "token_slot": "initializer.token_slot",
                    **(
                        {
                            position_input.name: f"initializer.{position_input.name}",
                            "body_position_ids": "initializer.body_position_ids",
                        }
                        if position_input is not None
                        else {}
                    ),
                    **{name: f"initializer.{name}" for name in sorted(cache_names)},
                },
            ),
            _invoke(decoder_name, setup_decoder_inputs, setup_decoder_outputs),
            _invoke(
                "last_token_logits",
                {"logits": "decoder.setup.logits"},
                {"last_logits": "decoder.setup.last_logits"},
            ),
        ],
    }
    body = {
        "kind": "sequence",
        "nodes": [
            _invoke(
                "token_sampler",
                {
                    "logits": "state.logits.body",
                    **(
                        {
                            "temperature": "request.temperature",
                            "top_k": "request.top_k",
                            "top_p": "request.top_p",
                            "min_p": "request.min_p",
                            "grammar_mask": "request.grammar_mask",
                            "seed": "request.seed",
                            "offset": "state.rng_offset.body",
                        }
                        if stochastic_sampler
                        else {}
                    ),
                },
                {
                    "token": "sample.body",
                    **({"next_offset": "sample.next_offset"} if stochastic_sampler else {}),
                },
                {"sample": _effect("sample.0", "sample.1")},
            ),
            _invoke(
                "token_state_update",
                {"current": "state.token.body", "update": "sample.body"},
                {"next": "token.body"},
                {"state": _effect("state.0", "state.1")},
            ),
            *(
                [
                    _invoke(
                        "model_token_cast",
                        {"token": "token.body"},
                        {"model_token": "model_token.body"},
                    )
                ]
                if needs_token_cast
                else []
            ),
            _invoke(
                "termination",
                {
                    "token_ids": "sample.body",
                    "eos_ids": "package.eos_ids",
                    "iteration": "state.iteration.body",
                    "max_iterations": "request.max_iterations",
                },
                {"done": "loop.done"},
                {"termination": _effect("termination.0", "termination.1")},
            ),
            _invoke(
                "continue_predicate",
                {"done": "loop.done"},
                {"continue": "loop.continue"},
            ),
            {
                "kind": "emit",
                "value": "sample.body",
                "output": "tokens",
                "mode": "append",
                "effect_name": "emit",
                "effect": _effect("emit.0", "emit.1"),
            },
            _invoke(
                "iteration_increment",
                {"value": "state.iteration.body"},
                {"next_value": "iteration.body"},
            ),
            _invoke(decoder_name, body_decoder_inputs, body_decoder_outputs),
            _invoke(
                "last_token_logits",
                {"logits": "decoder.body.logits"},
                {"last_logits": "decoder.body.last_logits"},
            ),
            _invoke(
                "decoder_step_update",
                {
                    "attention_mask": "state.attention_mask.body",
                    **(
                        {"position_ids": "state.position_ids.body"}
                        if position_input is not None
                        else {}
                    ),
                },
                {
                    "next_attention_mask": "decoder_step.body_attention_mask",
                    **(
                        {"next_position_ids": "decoder_step.body_position_ids"}
                        if position_input is not None
                        else {}
                    ),
                },
            ),
        ],
    }

    use_subfolders = len(pkg) > 1
    artifact = f"{decoder_name}/model.onnx" if use_subfolders else "model.onnx"
    workflow = {
        "manifest": {
            "ir_version": "1.0",
            "onnx_opsets": {"ai.onnx": OPSET_VERSION},
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "typed_emit",
            ],
        },
        "inputs": workflow_inputs,
        "outputs": {
            "tokens": {
                "contract": batch_int,
                "role": "tokens",
                "stage": "pre_adapter",
            }
        },
        "components": {decoder_name: _component(decoder, artifact)},
        "state": state,
        "initial_effects": initial_effects,
        "graph": {
            "kind": "loop",
            "setup": setup,
            "body": body,
            "condition": "loop.continue",
            "max_iterations": "request.max_iterations",
            "carried": carried,
        },
    }
    metadata = {
        "schema_version": "1.0",
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def build_language_diffusion_pipeline_metadata(
    pkg: Any,
    *,
    num_inference_steps: int,
) -> dict[str, Any]:
    """Build a generic SSA workflow for a masked language-diffusion model."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    if len(pkg) != 1:
        raise ValueError("language-diffusion workflow requires exactly one neural component")
    denoiser_name, denoiser = next(iter(pkg.items()))
    if len(denoiser.graph.inputs) != 1:
        raise ValueError("language-diffusion denoiser requires exactly one token input")

    token_input = denoiser.graph.inputs[0]
    logits_output = next(
        (value for value in denoiser.graph.outputs if value.name == "logits"),
        None,
    )
    proposal_output = next(
        (value for value in denoiser.graph.outputs if value.name == "proposed_tokens"),
        None,
    )
    if (
        token_input.dtype not in {ir.DataType.INT32, ir.DataType.INT64}
        or token_input.shape is None
        or len(token_input.shape) != 2
        or logits_output is None
        or logits_output.shape is None
        or len(logits_output.shape) != 3
        or proposal_output is None
        or proposal_output.shape is None
        or len(proposal_output.shape) != 2
    ):
        raise ValueError(
            "language-diffusion workflow requires token [B,T], logits [B,T,V], "
            "and proposed_tokens [B,T] ports"
        )

    attach_policy_components(pkg, PolicyCapabilities(masked_update=True))
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    pkg.add_policy_component("iteration_increment", build_integer_increment())

    token_contract = _contract(token_input)
    mask_contract = {
        "dtype": "bool",
        "rank": 2,
        "shape": token_contract["shape"],
    }
    batch_dimension = token_contract["shape"][0]
    batch_int = {"dtype": "int64", "rank": 1, "shape": [batch_dimension]}
    inputs = {
        "request.input_ids": {
            "contract": token_contract,
            "role": {
                "kind": "runtime",
                "version": "1.0",
                "role": "prompt_tokens",
            },
            "source": {"kind": "request", "field": "prompt_tokens"},
            "required": True,
        },
        "request.mask": {
            "contract": mask_contract,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "masked_positions"},
            "required": True,
        },
        "request.seed": {
            "contract": batch_int,
            "role": {
                "kind": "runtime",
                "version": "1.0",
                "role": "seed",
            },
            "source": {"kind": "request", "field": "seed"},
            "required": False,
            "default": 0,
        },
        "request.rng_offset": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "rng_offset"},
            "required": False,
            "default": 0,
        },
        "request.max_iterations": {
            "contract": batch_int,
            "role": {
                "kind": "runtime",
                "version": "1.0",
                "role": "max_iterations",
            },
            "source": {"kind": "request", "field": "max_iterations"},
            "required": False,
            "default": num_inference_steps,
        },
        "package.zero_iteration": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0,
        },
        "package.num_steps": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": num_inference_steps,
        },
    }

    def denoiser_invoke(tokens: str, prefix: str) -> dict[str, Any]:
        return _invoke(
            denoiser_name,
            {token_input.name: tokens},
            {
                logits_output.name: f"{prefix}.logits",
                proposal_output.name: f"{prefix}.proposal",
            },
        )

    def update_invoke(
        tokens: str,
        mask: str,
        iteration: str,
        offset: str,
        logits: str,
        proposal: str,
        prefix: str,
        effect_in: str,
        effect_out: str,
    ) -> dict[str, Any]:
        return _invoke(
            "masked_update",
            {
                "current_tokens": tokens,
                "proposed_tokens": proposal,
                "logits": logits,
                "masked": mask,
                "step": iteration,
                "total_steps": "package.num_steps",
                "seed": "request.seed",
                "offset": offset,
            },
            {
                "next_state": f"{prefix}.tokens",
                "next_mask": f"{prefix}.mask",
                "next_offset": f"{prefix}.rng_offset",
                "done": f"{prefix}.done",
            },
            {"update": _effect(effect_in, effect_out)},
        )

    setup = {
        "kind": "sequence",
        "nodes": [denoiser_invoke("request.input_ids", "denoiser.setup")],
    }
    body = {
        "kind": "sequence",
        "nodes": [
            update_invoke(
                "state.tokens.body",
                "state.mask.body",
                "state.iteration.body",
                "state.rng_offset.body",
                "state.logits.body",
                "state.proposal.body",
                "denoiser.body",
                "update.0",
                "update.1",
            ),
            _invoke(
                "continue_predicate",
                {"done": "denoiser.body.done"},
                {"continue": "denoiser.body.continue"},
            ),
            _invoke(
                "iteration_increment",
                {"value": "state.iteration.body"},
                {"next_value": "denoiser.body.iteration"},
            ),
            {
                "kind": "emit",
                "value": "denoiser.body.tokens",
                "output": "tokens",
                "mode": "replace",
                "effect_name": "emit",
                "effect": _effect("emit.0", "emit.1"),
            },
            denoiser_invoke("denoiser.body.tokens", "denoiser.body"),
        ],
    }

    state_specs = {
        "tokens": (token_contract, "request.input_ids", "request.input_ids"),
        "mask": (mask_contract, "request.mask", "request.mask"),
        "rng_offset": (batch_int, "request.rng_offset", "request.rng_offset"),
        "iteration": (batch_int, "package.zero_iteration", "package.zero_iteration"),
        "logits": (
            _contract(logits_output),
            "denoiser.setup.logits",
            "denoiser.setup.logits",
        ),
        "proposal": (
            _contract(proposal_output),
            "denoiser.setup.proposal",
            "denoiser.setup.proposal",
        ),
    }
    state: dict[str, Any] = {}
    carried: list[dict[str, Any]] = []
    initial_effects = {"update": "update.0", "emit": "emit.0"}
    for name, (contract, initializer, current) in state_specs.items():
        effect_name = f"state:{name}"
        initial_effects[effect_name] = f"{effect_name}.0"
        state[name] = {
            "contract": contract,
            "scope": "invocation",
            "initializer": initializer,
            "recurrence": {"kind": "invariant"},
        }
        carried.append(
            {
                "cell": name,
                "current": current,
                "body_input": f"state.{name}.body",
                "body_output": f"denoiser.body.{name}",
                "next": f"state.{name}.final",
                "read_effect": _effect(f"{effect_name}.0", f"{effect_name}.read"),
                "write_effect": _effect(f"{effect_name}.read", f"{effect_name}.1"),
            }
        )

    workflow = {
        "manifest": {
            "ir_version": "1.0",
            "onnx_opsets": {"ai.onnx": OPSET_VERSION},
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "typed_emit",
            ],
        },
        "inputs": inputs,
        "outputs": {
            "tokens": {
                "contract": token_contract,
                "role": "tokens",
                "stage": "pre_adapter",
            }
        },
        "components": {denoiser_name: _component(denoiser, "model.onnx")},
        "state": state,
        "initial_effects": initial_effects,
        "graph": {
            "kind": "loop",
            "setup": setup,
            "body": body,
            "condition": "denoiser.body.continue",
            "max_iterations": "request.max_iterations",
            "carried": carried,
        },
    }
    metadata = {
        "schema_version": "1.0",
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def write_decoder_workflow_metadata(
    pkg: Any,
    output_dir: str,
    config: Any,
    *,
    sampler: str = "greedy",
) -> str:
    """Write decoder workflow metadata and policy artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_decoder_workflow_metadata(pkg, config, sampler=sampler)
    pkg.save_policy_components(output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def write_language_diffusion_workflow_metadata(
    pkg: Any,
    output_dir: str,
    *,
    num_inference_steps: int,
) -> str:
    """Write masked language-diffusion workflow metadata and policy artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_language_diffusion_pipeline_metadata(
        pkg,
        num_inference_steps=num_inference_steps,
    )
    pkg.save_policy_components(output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path
