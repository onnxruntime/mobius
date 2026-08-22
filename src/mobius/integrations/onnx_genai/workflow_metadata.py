# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ONNX GenAI workflow-IR metadata production."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from typing import Any

import onnx_ir as ir
import yaml

from mobius._constants import (
    STATIC_CACHE_KV_SEQUENCE_LENGTH,
    STATIC_CACHE_LAYOUT,
    STATIC_CACHE_SEQUENCE_AXIS,
    STATIC_CACHE_WRITE_INDICES,
)
from mobius.generation import (
    SOLVER_BUILDERS,
    PolicyCapabilities,
    attach_policy_components,
    build_acoustic_code_frame,
    build_autoregressive_audio_initializer,
    build_boolean_not,
    build_candidate_token_map,
    build_chunk_carry_update,
    build_chunk_overlap_prepare,
    build_chunk_plan,
    build_chunk_slice,
    build_code_frame_update,
    build_code_history_append,
    build_codebook_embedding_id,
    build_codec_layout_transpose,
    build_counter_rng_normal,
    build_ddim_solver_step,
    build_decoder_state_initializer,
    build_decoder_step_update,
    build_drop_first_frame,
    build_duplex_agent_frame_select,
    build_duplex_cell_to_frame,
    build_duplex_frame_assemble,
    build_duplex_frame_commit,
    build_duplex_stream_append,
    build_duplex_stream_tail,
    build_duplex_teacher_select,
    build_duplex_user_stream_merge,
    build_duplex_waveform_append,
    build_embedding_sum,
    build_empty_features,
    build_eos_termination,
    build_euler_model_input,
    build_euler_solver_step,
    build_flow_guidance,
    build_flow_match_solver_step,
    build_flow_model_inputs,
    build_frame_hidden_append,
    build_greedy_sampler,
    build_guidance_combine,
    build_guided_vocabulary_slice,
    build_identity_model_input,
    build_integer_add,
    build_integer_minimum,
    build_last_sequence_value,
    build_last_token_logits,
    build_local_codebook_select,
    build_local_rvq_append,
    build_local_rvq_initializer,
    build_model_token_cast,
    build_overlap_blend,
    build_pack_latents_2x2,
    build_proposal_metrics,
    build_request_continue,
    build_scalar_constant,
    build_scalar_integer_add,
    build_schedule_constant,
    build_schedule_history_append,
    build_schedule_lookup,
    build_seeded_categorical_sampler,
    build_selective_integer_add,
    build_sequence_concat,
    build_sequence_length,
    build_shape_constant,
    build_tensor_scale,
    build_termination_batch_initializer,
    build_token_state_update,
    build_token_to_slot,
    build_true_cfg,
    build_tts_decoder_state_initializer,
    build_tts_decoder_step_update,
    build_tts_state_initializer,
    build_unpack_latents_2x2,
    build_video_conv_cache_initializer,
    build_video_decode_chunk,
    build_video_decode_chunk_count,
    build_video_latent_initializer,
    build_video_latent_permute,
    build_video_latent_unscale,
    build_waveform_stitch,
    build_zeros_like,
    rotary_axis_count,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    _name_image_preprocessing_program,
    _port,
    _shape_metadata,
    _source_asset_path,
    add_adapter_service_to_metadata,
    add_policy_components_to_workflow,
    build_native_vlm_package_metadata,
    declare_request_alignment,
    request_batch_layout,
)
from mobius.tasks._ctc_asr import BATCH_PADDING_SENSITIVE_KEY


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _dump_yaml(metadata: dict[str, Any], handle: Any) -> None:
    yaml.dump(metadata, handle, Dumper=_NoAliasSafeDumper, sort_keys=False)


def _source_model_value(source: str | None, name: str, fallback: Any) -> Any:
    """Resolve a value from packaged runtime metadata when available."""
    candidates: list[tuple[str, tuple[str, ...]]] = []
    if source and os.path.isdir(source):
        candidates = [
            (os.path.join(source, "genai_config.json"), ("model", name)),
            (os.path.join(source, "tokenizer_config.json"), (name,)),
        ]
    for path, keys in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                value: Any = yaml.safe_load(handle)
            for key in keys:
                value = value[key]
        except (OSError, TypeError, KeyError):
            continue
        fallback = value
        break
    return fallback


def _source_token_id(source: str | None, name: str, fallback: Any) -> int:
    """Resolve a generation token ID from packaged runtime metadata when available."""
    fallback = _source_model_value(source, name, fallback)
    if isinstance(fallback, list):
        fallback = fallback[0] if fallback else 0
    return int(fallback or 0)


def _contract(value: ir.Value) -> dict[str, Any]:
    port = _port(value)
    dtype = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(
        port.dtype, port.dtype
    )
    shape = _shape_metadata(port)
    contract: dict[str, Any] = {
        "dtype": dtype,
        "rank": port.rank,
        "shape": shape,
    }
    layout = request_batch_layout(shape)
    if layout is not None:
        contract["batch_layout"] = layout
    return contract


def _request_aligned(contract: dict[str, Any], axis: int = 0) -> dict[str, Any]:
    """Mark a contract as carrying exactly one entry per in-flight request.

    This is a structural batching fact, not a row identity: it tells the runtime
    which axis to permute when it compacts the batch, while scheduler slots and
    sequence handles stay runtime-private.
    """
    return {**contract, "batch_layout": {"kind": "request_aligned", "axis": axis}}


# Translation between the port vocabulary this producer *mints* when it builds
# a graph and the runtime's architecture-neutral role vocabulary. Both sides are
# fixed vocabularies and Mobius owns one of them: the task builders in
# ``mobius.tasks`` choose these exact names, so reading them back here is a
# lookup, not an inference about a graph of unknown provenance. A port outside
# this vocabulary carries no role, because a workflow that guesses is worse than
# one that stays silent.
_PORT_ROLES: dict[str, str] = {
    "input_ids": "token_ids",
    "inputs_embeds": "inputs_embeds",
    "attention_mask": "attention_mask",
    "position_ids": "position_ids",
    "logits": "logits",
    "last_hidden_state": "hidden_states",
    "encoder_hidden_states": "encoder_hidden_states",
    "audio_features": "audio_features",
}


def _component(
    model: ir.Model,
    artifact: str,
    *,
    effects: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Declare one ONNX-backed workflow component: its artifact and port roles.

    A component declares only what its artifact cannot say about itself. The
    ``.onnx`` file is shipped inside the package and is authoritative for which
    ports exist and what dtype, rank and shape each one has, so transcribing
    that into YAML would create a second copy of a fact the package already
    carries — one that can drift from the graph and that nothing cross-checks
    at rest. The runtime resolves ports against the live session instead, which
    catches a name the graph does not expose rather than agreeing with a stale
    echo of it.

    What no graph carries is what a port *means*. ``input_ids`` and
    ``position_ids`` are both rank-2 ``int64``; nothing in the file says which
    one is the autoregressive sequence. An invocation binds an SSA value to a
    port, which records which value arrives but not whether it is tokens, a mask
    or logits — and that second fact is what a runtime needs before it can
    specialize a decode step. So ``roles`` is the whole declaration here.

    Only ports in this producer's own vocabulary get a role, and state ports
    never need one: the group that carries them already names its pairs, which
    is also where the fixed-capacity scatter ABI is stated.
    """
    del effects
    named = [str(value.name) for value in (*model.graph.inputs, *model.graph.outputs)]
    roles = {name: _PORT_ROLES[name] for name in named if name in _PORT_ROLES}
    declaration: dict[str, Any] = {"implementation": {"kind": "onnx", "artifact": artifact}}
    if roles:
        declaration["ports"] = {"roles": roles}
    return declaration


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
        # The adapter keeps one grammar FSM per in-flight request, and every
        # port is request-aligned on axis 0. Declaring the row scope is what
        # lets the runtime drive the mandatory compact(selection)/release(row)
        # ABI when the batch changes; without it the FSM rows would drift out
        # of correspondence with the sequences they guide.
        "row_scope": {"axis": 0, "stateful": True},
    }


def _effect(consumes: str, produces: str) -> dict[str, str]:
    return {"consumes": consumes, "produces": produces}


def _publish_workflow_v1(workflow: dict[str, Any]) -> dict[str, Any]:
    """Publish structured steps and logical carries without compiler bookkeeping."""
    graph = workflow.pop("graph")
    workflow.pop("initial_effects", None)
    for declaration in workflow.get("inputs", {}).values():
        source = declaration.get("source")
        if isinstance(source, dict) and source.get("kind") == "request":
            source.pop("field", None)

    # Every workflow value whose leading dimension is the batch symbol holds one
    # entry per in-flight request, so declare that structurally instead of leaving
    # a runtime to infer it. Graph-derived contracts already carry the layout;
    # this covers the hand-written declarations the runtime compares them against
    # when it validates a carry, a binding, or an emit.
    def _declare_row_alignment(contract: Any) -> Any:
        if (
            isinstance(contract, dict)
            and "batch_layout" not in contract
            and request_batch_layout(contract.get("shape")) is not None
        ):
            return _request_aligned(contract)
        return contract

    for section in ("inputs", "outputs", "state"):
        for declaration in workflow.get(section, {}).values():
            declaration["contract"] = _declare_row_alignment(declaration.get("contract"))
    for component in workflow.get("components", {}).values():
        for side in ("inputs", "outputs"):
            ports = component.get("ports", {}).get(side)
            if not ports:
                continue
            for port, contract in ports.items():
                ports[port] = _declare_row_alignment(contract)
    substitutions: dict[str, str] = {}
    loop_index = 0
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
        nonlocal loop_index
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
            if "axis" in node:
                result["axis"] = node["axis"]
            if "valid_length" in node:
                result["valid_length"] = rewrite(node["valid_length"])
            if "when" in node:
                result["when"] = rewrite(node["when"])
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
            current_loop = loop_index
            loop_index += 1
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
            active_cell = node.get("active_cell")
            if active_cell is None:
                active_cell = f"loop_{current_loop}_active"
                active_initializer = f"package.{active_cell}"
                workflow["inputs"][active_initializer] = {
                    "contract": {"dtype": "bool", "rank": 1, "shape": [1]},
                    "role": {"kind": "opaque"},
                    "source": {"kind": "literal"},
                    "required": False,
                    "default": True,
                }
                workflow["state"][active_cell] = {
                    "contract": {"dtype": "bool", "rank": 1, "shape": [1]},
                    "scope": "invocation",
                    "initializer": active_initializer,
                    "recurrence": {"kind": "invariant"},
                }
                carried.append(
                    {
                        "cell": active_cell,
                        "next": rewrite(node["condition"]),
                    }
                )
            result = {
                "kind": "loop",
                "setup": setup_steps,
                "steps": body_steps,
                "continue_when": active_cell,
                "max_iterations": rewrite(node["max_iterations"]),
                "carried": carried,
            }
            if "termination" in node:
                result["termination"] = node["termination"]
            if "iteration" in node:
                result["iteration"] = node["iteration"]
            return result
        raise ValueError(f"unsupported workflow node kind {kind!r}")

    collect_carried(graph)
    published = convert(graph)
    workflow["steps"] = published["steps"] if published["kind"] == "sequence" else [published]
    declare_request_alignment(workflow)
    return workflow


def _invoke(
    component: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    _effects: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "invoke",
        "component": component,
        "inputs": inputs,
        "outputs": outputs,
    }


def build_hierarchical_audio_workflow_metadata(pkg: Any) -> dict[str, Any]:
    """Build canonical nested AR/RVQ/flow/vocoder metadata for hierarchical audio."""
    required = {
        "language_model",
        "language_model_embedding",
        "language_model_semantic_embedding",
        "rvq_depth_decoder",
        "rvq_depth_decoder_projection",
        "rvq_depth_decoder_embedding",
        "rvq_depth_decoder_feedback_embedding",
        "rvq_depth_decoder_heads",
        "condition_encoder",
        "transformer",
        "vocoder",
    }
    missing = sorted(required.difference(pkg.keys()))
    if missing:
        raise ValueError(f"hierarchical audio workflow is missing components: {missing}")

    decoder = pkg["language_model"]
    decoder_inputs = {value.name: value for value in decoder.graph.inputs}
    decoder_outputs = {value.name: value for value in decoder.graph.outputs}
    cache_pairs = _model_cache_pairs(decoder)
    embeds = decoder_inputs["inputs_embeds"]
    hidden = decoder_outputs["last_hidden_state"]
    if embeds.dtype is None or embeds.shape is None or hidden.shape is None:
        raise ValueError("global decoder must expose typed embedding and hidden-state ports")
    hidden_size = list(hidden.shape)[-1]
    if not isinstance(hidden_size, int):
        raise TypeError("global decoder hidden size must be statically known")
    dtype = embeds.dtype
    dtype_name = _contract(embeds)["dtype"]
    fused_hidden_size = hidden_size * 8
    latent_channels = 128
    condition_size = 2048
    residual_codebooks = 7

    pkg.add_policy_component(
        "global_initializer",
        build_decoder_state_initializer(
            decoder,
            token_input=None,
            prompt_dtype=ir.DataType.INT64,
            attention_mask_input="attention_mask",
            position_ids_input="position_ids",
            cache_inputs=[past.name for past, _ in cache_pairs],
        ),
    )
    pkg.add_policy_component(
        "global_step_update",
        build_decoder_step_update(
            attention_dtype=decoder_inputs["attention_mask"].dtype,
            position_dtype=decoder_inputs["position_ids"].dtype,
        ),
    )
    pkg.add_policy_component(
        "ar_initializer",
        build_autoregressive_audio_initializer(dtype, fused_hidden_size=fused_hidden_size),
    )
    pkg.add_policy_component("length_step", build_integer_add())
    pkg.add_policy_component(
        "guided_vocabulary",
        build_guided_vocabulary_slice(
            vocabulary_start=151675,
            vocabulary_size=16384,
            stop_token_id=151670,
            guidance_scale=1.5,
            conditional_top_k=50,
            dtype=decoder_outputs["logits"].dtype,
        ),
    )
    pkg.add_policy_component("sampler", build_seeded_categorical_sampler())
    pkg.add_policy_component(
        "candidate_map",
        build_candidate_token_map(
            vocabulary_start=151675,
            vocabulary_size=16384,
            stop_token_id=151670,
        ),
    )
    pkg.add_policy_component("continue_predicate", build_request_continue())
    pkg.add_policy_component(
        "last_global_hidden",
        build_last_sequence_value(dtype, rows=2, channels=hidden_size),
    )
    pkg.add_policy_component(
        "local_initializer",
        build_local_rvq_initializer(dtype, hidden_size=hidden_size),
    )
    pkg.add_policy_component("codebook_index", build_scalar_integer_add())
    pkg.add_policy_component(
        "local_logits",
        build_local_codebook_select(dtype, guidance_scale=1.5),
    )
    pkg.add_policy_component(
        "embedding_id",
        build_codebook_embedding_id(codebook_size=1024),
    )
    pkg.add_policy_component(
        "local_append",
        build_local_rvq_append(dtype, hidden_size=hidden_size),
    )
    pkg.add_policy_component(
        "frame_append",
        build_frame_hidden_append(dtype, hidden_size=hidden_size, num_codebooks=8),
    )
    pkg.add_policy_component(
        "acoustic_frame",
        build_acoustic_code_frame(num_residual_codebooks=residual_codebooks),
    )
    pkg.add_policy_component(
        "feedback_sum",
        build_embedding_sum(dtype, hidden_size=hidden_size),
    )
    pkg.add_policy_component(
        "finalize_frames",
        build_drop_first_frame(dtype, fused_hidden_size=fused_hidden_size),
    )
    pkg.add_policy_component(
        "chunk_plan",
        build_chunk_plan(
            dtype,
            fused_hidden_size=fused_hidden_size,
            chunk_frames=200,
            chunk_hop=100,
            latent_channels=latent_channels,
            condition_size=condition_size,
        ),
    )
    pkg.add_policy_component(
        "chunk_slice",
        build_chunk_slice(
            dtype,
            fused_hidden_size=fused_hidden_size,
            chunk_frames=200,
            chunk_hop=100,
        ),
    )
    pkg.add_policy_component(
        "overlap_prepare",
        build_chunk_overlap_prepare(
            dtype,
            latent_channels=latent_channels,
            condition_size=condition_size,
        ),
    )
    pkg.add_policy_component(
        "latent_noise",
        build_counter_rng_normal(
            dtype,
            latent_dims=("batch", "channels", "latent_length"),
        ),
    )
    pkg.add_policy_component(
        "flow_schedule",
        build_schedule_constant([index / 30.0 for index in range(31)]),
    )
    pkg.add_policy_component("flow_timestep", build_schedule_lookup(dtype))
    pkg.add_policy_component(
        "overlap_blend",
        build_overlap_blend(dtype, latent_channels=latent_channels),
    )
    pkg.add_policy_component(
        "flow_inputs",
        build_flow_model_inputs(dtype, latent_channels=latent_channels),
    )
    pkg.add_policy_component(
        "flow_guidance",
        build_flow_guidance(dtype, latent_channels=latent_channels, guidance_scale=1.7),
    )
    pkg.add_policy_component(
        "flow_solver",
        build_flow_match_solver_step(dtype),
    )
    pkg.add_policy_component(
        "chunk_update",
        build_chunk_carry_update(
            dtype,
            latent_channels=latent_channels,
            condition_size=condition_size,
            carry_length=172,
        ),
    )
    pkg.add_policy_component(
        "waveform_stitch",
        build_waveform_stitch(
            dtype=next(iter(pkg["vocoder"].graph.outputs)).dtype,
            latent_hop_length=512,
            crop_left_latents=86,
            crop_right_latents=258,
        ),
    )

    batch_int = _request_aligned({"dtype": "int64", "rank": 1, "shape": ["batch"]})
    batch_float = _request_aligned({"dtype": "float32", "rank": 1, "shape": ["batch"]})
    batch_bool = _request_aligned({"dtype": "bool", "rank": 1, "shape": ["batch"]})
    scalar_int = {"dtype": "int64", "rank": 0, "shape": []}
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}
    prompt_contract = _contract(pkg["language_model_embedding"].graph.inputs[0])
    prompt_contract["batch_layout"] = {
        "kind": "request_expanded",
        "axis": 0,
        "factor": 2,
    }
    global_logits_contract = _contract(decoder_outputs["logits"])
    global_hidden_contract = _contract(hidden)
    global_mask_contract = _contract(decoder_inputs["attention_mask"])
    global_position_contract = _contract(decoder_inputs["position_ids"])
    for contract in (
        global_logits_contract,
        global_hidden_contract,
        global_mask_contract,
        global_position_contract,
    ):
        contract["batch_layout"] = {
            "kind": "request_expanded",
            "axis": 0,
            "factor": 2,
        }
    inputs: dict[str, Any] = {
        "request.prompt_tokens": {
            "contract": prompt_contract,
            "role": {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"},
            "source": {"kind": "request", "field": "prompt_tokens"},
            "required": True,
        },
        "request.max_frames_with_warmup": {
            "contract": control_int,
            "role": {"kind": "runtime", "version": "1.0", "role": "max_output_tokens"},
            "source": {"kind": "request", "field": "max_output_tokens"},
            "required": True,
        },
        "request.seed": {
            "contract": batch_int,
            "role": {"kind": "runtime", "version": "1.0", "role": "seed"},
            "source": {"kind": "request", "field": "seed"},
            "required": False,
            "default": 0,
        },
        "package.temperature": {
            "contract": batch_float,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1.0,
        },
        "package.top_k": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 50,
        },
        "package.top_p": {
            "contract": batch_float,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1.0,
        },
        "package.min_p": {
            "contract": batch_float,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0.0,
        },
        "package.one": {
            "contract": scalar_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.one_batch": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.local_steps": {
            "contract": scalar_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 7,
        },
        "package.local_context": {
            "contract": scalar_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 9,
        },
        "package.global_context": {
            "contract": scalar_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 10240,
        },
        "package.carry_length": {
            "contract": scalar_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 172,
        },
        "package.max_waveform_samples": {
            "contract": scalar_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 15_876_000,
        },
        "package.flow_steps": {
            "contract": scalar_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 30,
        },
        "package.flow_rng_offset": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0,
        },
    }

    def invoke_model(
        name: str,
        bindings: dict[str, str],
        prefix: str,
        output_bindings: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        model = pkg[name]
        for value in model.graph.inputs:
            if value.name not in bindings:
                raise ValueError(f"{name} input {value.name!r} is not bound")
        outputs = dict(output_bindings or {})
        for value in model.graph.outputs:
            outputs.setdefault(value.name, f"{prefix}.{value.name}")
        return _invoke(name, bindings, outputs)

    sample_inputs = {
        "logits": "frame.candidate_logits",
        "temperature": "package.temperature",
        "top_k": "package.top_k",
        "top_p": "package.top_p",
        "min_p": "package.min_p",
        "seed": "request.seed",
        "counter": "state.rng.outer",
        "active": "state.active.outer",
        "done": "state.done.outer",
    }
    local_sample_inputs = dict(sample_inputs)
    local_sample_inputs.update(
        logits="local.guided_logits",
        counter="state.local_rng.inner",
    )

    inner_loop = {
        "kind": "loop",
        "setup": {"kind": "sequence", "nodes": []},
        "body": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "codebook_index",
                    {"left": "local.iteration", "right": "package.one"},
                    {"total": "local.codebook_index"},
                ),
                invoke_model(
                    "rvq_depth_decoder",
                    {"inputs_embeds": "state.local_sequence.inner"},
                    "local.decoder",
                    {"hidden_states": "local.hidden_states"},
                ),
                invoke_model(
                    "rvq_depth_decoder_heads",
                    {"hidden_states": "local.hidden_states"},
                    "local.heads",
                    {"all_codebook_logits": "local.all_logits"},
                ),
                _invoke(
                    "local_logits",
                    {
                        "all_codebook_logits": "local.all_logits",
                        "codebook_index": "local.codebook_index",
                    },
                    {"logits": "local.guided_logits"},
                ),
                _invoke(
                    "sampler",
                    local_sample_inputs,
                    {"token": "local.token", "next_counter": "local.next_rng"},
                ),
                _invoke(
                    "embedding_id",
                    {
                        "token": "local.token",
                        "codebook_index": "local.codebook_index",
                    },
                    {"embedding_ids": "local.embedding_ids"},
                ),
                invoke_model(
                    "rvq_depth_decoder_embedding",
                    {"code_ids": "local.embedding_ids"},
                    "local.embedding",
                    {"code_embeddings": "local.code_embedding"},
                ),
                invoke_model(
                    "rvq_depth_decoder_projection",
                    {"hidden_states": "local.code_embedding"},
                    "local.code_projection",
                    {"projected_states": "local.projected_embedding"},
                ),
                _invoke(
                    "local_append",
                    {
                        "sequence": "state.local_sequence.inner",
                        "projected_embedding": "local.projected_embedding",
                        "acoustic_codes": "state.local_codes.inner",
                        "token": "local.token",
                        "hidden_states": "local.hidden_states",
                        "local_hidden_parts": "state.local_hidden.inner",
                    },
                    {
                        "next_sequence": "local.next_sequence",
                        "next_acoustic_codes": "local.next_codes",
                        "next_local_hidden_parts": "local.next_hidden",
                    },
                ),
            ],
        },
        "condition": "state.active.outer",
        "max_iterations": "package.local_steps",
        "iteration": {"value": "local.iteration", "contract": scalar_int},
        "carried": [
            {
                "cell": "local_sequence",
                "current": "local.initial_sequence",
                "body_input": "state.local_sequence.inner",
                "body_output": "local.next_sequence",
                "next": "local.sequence.final",
            },
            {
                "cell": "local_codes",
                "current": "local.initial_codes",
                "body_input": "state.local_codes.inner",
                "body_output": "local.next_codes",
                "next": "local.codes.final",
            },
            {
                "cell": "local_hidden",
                "current": "local.initial_hidden",
                "body_input": "state.local_hidden.inner",
                "body_output": "local.next_hidden",
                "next": "local.hidden.final",
            },
            {
                "cell": "local_rng",
                "current": "frame.next_rng",
                "body_input": "state.local_rng.inner",
                "body_output": "local.next_rng",
                "next": "local.rng.final",
            },
        ],
    }

    cache_setup_bindings = {
        "inputs_embeds": "global.prompt_embeds",
        "attention_mask": "global.initial.attention_mask",
        "position_ids": "global.initial.position_ids",
        **{past.name: f"global.initial.{past.name}" for past, _ in cache_pairs},
    }
    cache_body_bindings = {
        "inputs_embeds": "frame.feedback",
        "attention_mask": "state.global_mask.outer",
        "position_ids": "state.global_position.outer",
        **{
            past.name: f"state.global_cache_{index}.outer"
            for index, (past, _) in enumerate(cache_pairs)
        },
    }
    setup_decoder_outputs = {
        "logits": "global.setup.logits",
        "last_hidden_state": "global.setup.hidden",
        **{
            present.name: f"global.setup.cache_{index}"
            for index, (_, present) in enumerate(cache_pairs)
        },
    }
    body_decoder_outputs = {
        "logits": "global.next.logits",
        "last_hidden_state": "global.next.hidden",
        **{
            present.name: f"global.next.cache_{index}"
            for index, (_, present) in enumerate(cache_pairs)
        },
    }
    outer_body_nodes = [
        _invoke(
            "guided_vocabulary",
            {"logits": "state.global_logits.outer"},
            {"candidate_logits": "frame.candidate_logits"},
        ),
        _invoke(
            "sampler",
            sample_inputs,
            {"token": "frame.candidate", "next_counter": "frame.next_rng"},
        ),
        _invoke(
            "candidate_map",
            {"candidate": "frame.candidate"},
            {
                "token": "frame.token",
                "semantic_code": "frame.semantic_code",
                "semantic_token": "frame.semantic_token",
                "is_stop": "frame.is_stop",
            },
        ),
        _invoke(
            "continue_predicate",
            {"done": "frame.is_stop"},
            {"continue": "frame.continue"},
        ),
        _invoke(
            "last_global_hidden",
            {"value": "state.global_hidden.outer"},
            {"last": "frame.global_hidden"},
        ),
        invoke_model(
            "language_model_embedding",
            {"input_ids": "frame.semantic_token"},
            "frame.raw_semantic",
            {"inputs_embeds": "frame.raw_semantic_embedding"},
        ),
        invoke_model(
            "rvq_depth_decoder_projection",
            {"hidden_states": "frame.global_hidden"},
            "frame.global_projection",
            {"projected_states": "frame.projected_global"},
        ),
        invoke_model(
            "rvq_depth_decoder_projection",
            {"hidden_states": "frame.raw_semantic_embedding"},
            "frame.semantic_projection",
            {"projected_states": "frame.projected_semantic"},
        ),
        _invoke(
            "local_initializer",
            {
                "global_hidden": "frame.projected_global",
                "semantic_embedding": "frame.projected_semantic",
            },
            {
                "sequence": "local.initial_sequence",
                "acoustic_codes": "local.initial_codes",
                "local_hidden_parts": "local.initial_hidden",
            },
        ),
        inner_loop,
        _invoke(
            "frame_append",
            {
                "history": "state.frame_history.outer",
                "global_hidden": "frame.global_hidden",
                "local_hidden_parts": "local.hidden.final",
            },
            {"next_history": "frame.history.next"},
        ),
        _invoke(
            "acoustic_frame",
            {"acoustic_codes": "local.codes.final"},
            {"framed_acoustic_codes": "frame.acoustic_codes"},
        ),
        invoke_model(
            "language_model_semantic_embedding",
            {"semantic_codes": "frame.semantic_code"},
            "frame.semantic_feedback",
            {"semantic_feedback_embedding": "frame.semantic_feedback"},
        ),
        invoke_model(
            "rvq_depth_decoder_feedback_embedding",
            {"acoustic_codes": "frame.acoustic_codes"},
            "frame.acoustic_feedback",
            {"acoustic_feedback_embedding": "frame.acoustic_feedback"},
        ),
        _invoke(
            "feedback_sum",
            {
                "semantic": "frame.semantic_feedback",
                "acoustic": "frame.acoustic_feedback",
            },
            {"feedback": "frame.feedback"},
        ),
        invoke_model(
            "language_model",
            cache_body_bindings,
            "global.next",
            body_decoder_outputs,
        ),
        _invoke(
            "global_step_update",
            {
                "attention_mask": "state.global_mask.outer",
                "position_ids": "state.global_position.outer",
            },
            {
                "next_attention_mask": "global.mask.next",
                "next_position_ids": "global.position.next",
            },
        ),
        _invoke(
            "length_step",
            {
                "left": "state.global_length.outer",
                "right": "package.one_batch",
            },
            {"total": "global.length.next"},
        ),
    ]
    outer_carried = [
        {
            "cell": "global_logits",
            "current": "global.setup.logits",
            "body_input": "state.global_logits.outer",
            "body_output": "global.next.logits",
            "next": "global.logits.final",
        },
        {
            "cell": "global_hidden",
            "current": "global.setup.hidden",
            "body_input": "state.global_hidden.outer",
            "body_output": "global.next.hidden",
            "next": "global.hidden.final",
        },
        {
            "cell": "global_mask",
            "current": "global.initial.body_attention_mask",
            "body_input": "state.global_mask.outer",
            "body_output": "global.mask.next",
            "next": "global.mask.final",
        },
        {
            "cell": "global_position",
            "current": "global.initial.body_position_ids",
            "body_input": "state.global_position.outer",
            "body_output": "global.position.next",
            "next": "global.position.final",
        },
        {
            "cell": "frame_history",
            "current": "ar.initial.history",
            "body_input": "state.frame_history.outer",
            "body_output": "frame.history.next",
            "next": "frame.history.final",
        },
        {
            "cell": "rng",
            "current": "ar.initial.counter",
            "body_input": "state.rng.outer",
            "body_output": "local.rng.final",
            "next": "ar.rng.final",
        },
        {
            "cell": "active",
            "current": "ar.initial.active",
            "body_input": "state.active.outer",
            "body_output": "frame.continue",
            "next": "ar.active.final",
        },
        {
            "cell": "done",
            "current": "ar.initial.done",
            "body_input": "state.done.outer",
            "body_output": "frame.is_stop",
            "next": "ar.done.final",
        },
        {
            "cell": "global_length",
            "current": "ar.initial.counter",
            "body_input": "state.global_length.outer",
            "body_output": "global.length.next",
            "next": "global.length.final",
        },
    ]
    for index, (_, _present) in enumerate(cache_pairs):
        outer_carried.append(
            {
                "cell": f"global_cache_{index}",
                "current": f"global.setup.cache_{index}",
                "body_input": f"state.global_cache_{index}.outer",
                "body_output": f"global.next.cache_{index}",
                "next": f"global.cache_{index}.final",
            }
        )

    ar_loop = {
        "kind": "loop",
        "setup": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "global_initializer",
                    {"prompt_tokens": "request.prompt_tokens"},
                    {
                        "attention_mask": "global.initial.attention_mask",
                        "position_ids": "global.initial.position_ids",
                        "body_attention_mask": "global.initial.body_attention_mask",
                        "body_position_ids": "global.initial.body_position_ids",
                        "token_slot": "global.initial.token_slot",
                        **{
                            past.name: f"global.initial.{past.name}" for past, _ in cache_pairs
                        },
                    },
                ),
                invoke_model(
                    "language_model_embedding",
                    {"input_ids": "request.prompt_tokens"},
                    "global.embedding",
                    {"inputs_embeds": "global.prompt_embeds"},
                ),
                invoke_model(
                    "language_model",
                    cache_setup_bindings,
                    "global.setup",
                    setup_decoder_outputs,
                ),
                _invoke(
                    "ar_initializer",
                    {},
                    {
                        "frame_history": "ar.initial.history",
                        "rng_counter": "ar.initial.counter",
                        "active": "ar.initial.active",
                        "done": "ar.initial.done",
                    },
                ),
            ],
        },
        "body": {"kind": "sequence", "nodes": outer_body_nodes},
        "condition": "state.active.outer",
        "max_iterations": "request.max_frames_with_warmup",
        "iteration": {"value": "frame.iteration", "contract": scalar_int},
        "carried": outer_carried,
    }

    flow_loop = {
        "kind": "loop",
        "setup": {"kind": "sequence", "nodes": []},
        "body": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "flow_timestep",
                    {"schedule": "flow.schedule", "step": "flow.iteration"},
                    {"timestep": "flow.timestep.value"},
                ),
                _invoke(
                    "overlap_blend",
                    {
                        "latents": "state.flow_latents.inner",
                        "initial_noise": "chunk.initial_noise",
                        "previous_latent": "state.previous_latent.chunk",
                        "overlap": "chunk.overlap",
                        "timestep": "flow.timestep.value",
                    },
                    {"blended_latents": "flow.blended"},
                ),
                _invoke(
                    "flow_inputs",
                    {
                        "latents": "flow.blended",
                        "timestep": "flow.timestep.value",
                    },
                    {
                        "guided_latents": "flow.guided_latents",
                        "guided_timestep": "flow.guided_timestep",
                    },
                ),
                invoke_model(
                    "transformer",
                    {
                        "hidden_states": "flow.guided_latents",
                        "timestep": "flow.guided_timestep",
                        "encoder_hidden_states": "chunk.guided_condition",
                    },
                    "flow.transformer",
                    {"sample": "flow.sample"},
                ),
                _invoke(
                    "flow_guidance",
                    {"sample": "flow.sample"},
                    {"velocity": "flow.velocity"},
                ),
                _invoke(
                    "flow_solver",
                    {
                        "sample": "flow.blended",
                        "derivative": "flow.velocity",
                        "schedule": "flow.schedule",
                        "step": "flow.iteration",
                    },
                    {"next_state": "flow.next_latents"},
                ),
            ],
        },
        "condition": "state.active.outer",
        "max_iterations": "package.flow_steps",
        "iteration": {"value": "flow.iteration", "contract": batch_int},
        "carried": [
            {
                "cell": "flow_latents",
                "current": "chunk.initial_noise",
                "body_input": "state.flow_latents.inner",
                "body_output": "flow.next_latents",
                "next": "flow.latents.final",
            }
        ],
    }

    chunk_loop = {
        "kind": "loop",
        "setup": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "finalize_frames",
                    {
                        "history": "frame.history.final",
                        "stopped": "ar.done.final",
                    },
                    {"frame_hiddens": "audio.frame_hiddens"},
                ),
                _invoke(
                    "chunk_plan",
                    {"frame_hiddens": "audio.frame_hiddens"},
                    {
                        "chunk_count": "audio.chunk_count",
                        "waveform": "audio.initial_waveform",
                        "previous_latent": "audio.initial_previous_latent",
                        "previous_condition": "audio.initial_previous_condition",
                    },
                ),
                _invoke(
                    "flow_schedule",
                    {},
                    {"schedule": "flow.schedule"},
                ),
            ],
        },
        "body": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "chunk_slice",
                    {
                        "frame_hiddens": "audio.frame_hiddens",
                        "chunk_index": "chunk.iteration",
                    },
                    {"frame_chunk": "chunk.frame_hiddens"},
                ),
                invoke_model(
                    "condition_encoder",
                    {"hidden_states": "chunk.frame_hiddens"},
                    "chunk.condition_encoder",
                    {"encoder_hidden_states": "chunk.condition"},
                ),
                _invoke(
                    "overlap_prepare",
                    {
                        "condition": "chunk.condition",
                        "previous_condition": "state.previous_condition.chunk",
                        "previous_latent": "state.previous_latent.chunk",
                    },
                    {
                        "guided_condition": "chunk.guided_condition",
                        "spliced_condition": "chunk.spliced_condition",
                        "overlap": "chunk.overlap",
                        "noise_row_shape": "chunk.noise_shape",
                    },
                ),
                _invoke(
                    "latent_noise",
                    {
                        "seed": "request.seed",
                        "offset": "state.flow_rng.chunk",
                        "row_shape": "chunk.noise_shape",
                    },
                    {
                        "noise": "chunk.initial_noise",
                        "next_offset": "chunk.next_rng",
                    },
                ),
                flow_loop,
                _invoke(
                    "chunk_update",
                    {
                        "latents": "flow.latents.final",
                        "previous_latent": "state.previous_latent.chunk",
                        "condition": "chunk.spliced_condition",
                        "overlap": "chunk.overlap",
                    },
                    {
                        "restored_latents": "chunk.restored_latents",
                        "next_previous_latent": "chunk.next_previous_latent",
                        "next_previous_condition": "chunk.next_previous_condition",
                    },
                ),
                invoke_model(
                    "vocoder",
                    {"latents": "chunk.restored_latents"},
                    "chunk.vocoder",
                    {"waveform": "chunk.waveform"},
                ),
                _invoke(
                    "waveform_stitch",
                    {
                        "waveform": "chunk.waveform",
                        "history": "state.waveform.chunk",
                        "chunk_index": "chunk.iteration",
                        "chunk_count": "audio.chunk_count",
                    },
                    {"next_history": "chunk.waveform.next"},
                ),
            ],
        },
        "condition": "ar.initial.active",
        "max_iterations": "audio.chunk_count",
        "iteration": {"value": "chunk.iteration", "contract": scalar_int},
        "carried": [
            {
                "cell": "previous_latent",
                "current": "audio.initial_previous_latent",
                "body_input": "state.previous_latent.chunk",
                "body_output": "chunk.next_previous_latent",
                "next": "audio.previous_latent.final",
            },
            {
                "cell": "previous_condition",
                "current": "audio.initial_previous_condition",
                "body_input": "state.previous_condition.chunk",
                "body_output": "chunk.next_previous_condition",
                "next": "audio.previous_condition.final",
            },
            {
                "cell": "waveform",
                "current": "audio.initial_waveform",
                "body_input": "state.waveform.chunk",
                "body_output": "chunk.waveform.next",
                "next": "audio.waveform",
            },
            {
                "cell": "flow_rng",
                "current": "package.flow_rng_offset",
                "body_input": "state.flow_rng.chunk",
                "body_output": "chunk.next_rng",
                "next": "audio.flow_rng.final",
            },
        ],
    }

    state: dict[str, Any] = {
        "global_logits": {
            "contract": global_logits_contract,
            "scope": "invocation",
            "initializer": "global.setup.logits",
            "recurrence": {"kind": "bounded", "axis": 1, "max": "package.global_context"},
        },
        "global_hidden": {
            "contract": global_hidden_contract,
            "scope": "invocation",
            "initializer": "global.setup.hidden",
            "recurrence": {"kind": "bounded", "axis": 1, "max": "package.global_context"},
        },
        "global_mask": {
            "contract": global_mask_contract,
            "scope": "invocation",
            "initializer": "global.initial.body_attention_mask",
            "recurrence": {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one",
                "max": "package.global_context",
            },
        },
        "global_position": {
            "contract": {
                **global_position_contract,
                "shape": [global_position_contract["shape"][0], 1],
            },
            "scope": "invocation",
            "initializer": "global.initial.body_position_ids",
            "recurrence": {"kind": "invariant"},
        },
        "frame_history": {
            "contract": {
                "dtype": dtype_name,
                "rank": 3,
                "shape": [1, "frames", fused_hidden_size],
            },
            "scope": "invocation",
            "initializer": "ar.initial.history",
            "recurrence": {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one",
                "max": "request.max_frames_with_warmup",
            },
        },
        "rng": {
            "contract": batch_int,
            "scope": "invocation",
            "initializer": "ar.initial.counter",
            "recurrence": {"kind": "invariant"},
        },
        "active": {
            "contract": batch_bool,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "ar.initial.active",
            "recurrence": {"kind": "invariant"},
        },
        "done": {
            "contract": batch_bool,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "ar.initial.done",
            "recurrence": {"kind": "invariant"},
        },
        "global_length": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "ar.initial.counter",
            "recurrence": {"kind": "invariant"},
        },
        "local_sequence": {
            "contract": {"dtype": dtype_name, "rank": 3, "shape": [2, "steps", hidden_size]},
            "scope": "invocation",
            "initializer": "local.initial_sequence",
            "recurrence": {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one",
                "max": "package.local_context",
            },
        },
        "local_codes": {
            "contract": {"dtype": "int64", "rank": 2, "shape": [2, "codes"]},
            "scope": "invocation",
            "initializer": "local.initial_codes",
            "recurrence": {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one",
                "max": "package.local_steps",
            },
        },
        "local_hidden": {
            "contract": {"dtype": dtype_name, "rank": 3, "shape": [1, "parts", hidden_size]},
            "scope": "invocation",
            "initializer": "local.initial_hidden",
            "recurrence": {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one",
                "max": "package.local_steps",
            },
        },
        "local_rng": {
            "contract": batch_int,
            "scope": "invocation",
            "initializer": "frame.next_rng",
            "recurrence": {"kind": "invariant"},
        },
        "flow_latents": {
            "contract": {"dtype": dtype_name, "rank": 3, "shape": [1, 128, "latent_length"]},
            "scope": "invocation",
            "initializer": "chunk.initial_noise",
            "recurrence": {"kind": "invariant"},
        },
        "previous_latent": {
            "contract": {"dtype": dtype_name, "rank": 3, "shape": [1, 128, "carry_length"]},
            "scope": "invocation",
            "initializer": "audio.initial_previous_latent",
            "recurrence": {"kind": "bounded", "axis": 2, "max": "package.carry_length"},
        },
        "previous_condition": {
            "contract": {"dtype": dtype_name, "rank": 3, "shape": [1, "carry_length", 2048]},
            "scope": "invocation",
            "initializer": "audio.initial_previous_condition",
            "recurrence": {"kind": "bounded", "axis": 1, "max": "package.carry_length"},
        },
        "waveform": {
            "contract": _request_aligned(
                {"dtype": "float32", "rank": 3, "shape": ["batch", 2, "samples"]}
            ),
            "scope": "invocation",
            "initializer": "audio.initial_waveform",
            "recurrence": {
                "kind": "bounded",
                "axis": 2,
                "max": "package.max_waveform_samples",
            },
        },
        "flow_rng": {
            "contract": batch_int,
            "scope": "invocation",
            "initializer": "package.flow_rng_offset",
            "recurrence": {"kind": "invariant"},
        },
    }
    for index, (_, present) in enumerate(cache_pairs):
        cache_contract = _contract(present)
        cache_contract["batch_layout"] = {
            "kind": "request_expanded",
            "axis": 0,
            "factor": 2,
        }
        state[f"global_cache_{index}"] = {
            "contract": cache_contract,
            "class": "semantic",
            "scope": "invocation",
            "initializer": f"global.setup.cache_{index}",
            "recurrence": {"kind": "bounded", "axis": 2, "max": "package.global_context"},
            "service_group": "global_cache",
            "management": "runtime",
            "release_boundary": "invocation",
        }

    components = {
        name: _component(model, _artifact(name, len(pkg))) for name, model in pkg.items()
    }
    components["speech_text_assembly"] = {
        "implementation": {
            "kind": "adapter",
            "abi": "onnx-genai.text-assembly",
            "version": "1",
            "artifact": "speech_processor.json",
        },
        "contract": {"id": "onnx-genai.text-assembly", "version": "1"},
    }
    workflow = {
        "manifest": {
            "adapter_abis": {"onnx-genai.text-assembly": "1"},
            "capabilities": [
                "workflow_ssa",
                "nested_control_flow",
                "loop_induction_values",
                "loop_carried_state",
                "typed_emit",
                "serving_service_contract",
                "bounded_state_recurrence",
            ],
        },
        "inputs": inputs,
        "outputs": {
            "audio": {
                "contract": _request_aligned(
                    {"dtype": "float32", "rank": 3, "shape": ["batch", 2, "samples"]}
                ),
                "role": "audio",
                "stage": "pre_adapter",
                "media": {
                    "container": "wav",
                    "encoding": "pcm_s16_le",
                    "sample_rate_hz": 32000,
                    "source_sample_rate_hz": 44100,
                    "channels": 2,
                    "delivery": "buffered",
                },
            }
        },
        "components": components,
        "state": state,
        "serving": {
            "active": "active",
            "done": "done",
            "accepted_len": "global_length",
            "state_service": {
                "groups": {
                    "global_cache": {
                        "kind": "full_attention",
                        "sequence_axis": 2,
                        "layout": "bnsh",
                        "aliasing": "forbidden",
                        "reuse": {
                            "prefix_reusable": True,
                            "evictable_prefix": False,
                        },
                        "ports": {
                            "language_model": {
                                f"global_cache_{index}": {
                                    "input": past.name,
                                    "output": present.name,
                                    "role": (
                                        "key"
                                        if (past.name or "").endswith(".key")
                                        else "value"
                                    ),
                                    "layer": index // 2,
                                }
                                for index, (past, present) in enumerate(cache_pairs)
                            }
                        },
                    }
                }
            },
        },
        "graph": {
            "kind": "sequence",
            "nodes": [
                ar_loop,
                chunk_loop,
                {
                    "kind": "emit",
                    "value": "audio.waveform",
                    "output": "audio",
                    "mode": "replace",
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


def write_hierarchical_audio_workflow_metadata(pkg: Any, output_dir: str) -> str:
    """Write canonical hierarchical-audio workflow and exact prompt processor."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_hierarchical_audio_workflow_metadata(pkg)
    # Building the workflow registers its generic policy graphs. The CLI saves
    # neural components before metadata generation, so persist these newly
    # registered artifacts here rather than publishing dangling references.
    pkg.save_policy_components(output_dir)
    processor = {
        "max_input_tokens": 5000,
        "max_output_units": 9000,
        "state_advance_units": 1,
        "guidance_rows": {
            "unconditional_token_id": 151654,
            "replace_from": 1,
            "preserve_trailing": 2,
        },
        "segments": [
            {"literal": "<|im_start|><|caption_start|>"},
            {
                "field": "instructions",
                "transforms": [
                    {"kind": "rewrite_delimited_tags", "open": "<|", "close": "|>"},
                    {"kind": "strip_markdown"},
                    {"kind": "collapse_newlines"},
                ],
            },
            {"literal": "<|caption_end|><|lyrics_start|>[start]\n"},
            {
                "field": "input",
                "transforms": [
                    {"kind": "keep_leading_bracket_tags"},
                    {"kind": "replace", "from": "] ", "to": "]\n"},
                    {"kind": "replace", "from": " [", "to": "\n["},
                    {"kind": "replace", "from": " ^ ", "to": "\n"},
                    {"kind": "lowercase_bracket_tags"},
                ],
            },
            {"literal": "<|lyrics_end|><|im_end|><|audio_start|>"},
        ],
    }
    with open(
        os.path.join(output_dir, "speech_processor.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(processor, handle, indent=2)
        handle.write("\n")
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


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
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


def _model_cache_pairs(model: ir.Model) -> list[tuple[ir.Value, ir.Value]]:
    outputs = {value.name: value for value in model.graph.outputs}
    pairs = []
    for past in model.graph.inputs:
        present = next(
            (
                outputs.get(name)
                for name in _cache_output_candidates(past.name or "")
                if name in outputs
            ),
            None,
        )
        if present is not None:
            pairs.append((past, present))
    return pairs


def _cache_output_candidates(past_name: str) -> tuple[str, ...]:
    """Names an exporter may give the output that continues a cache input.

    An appending cache renames ``past`` to ``present``; a static, indexed cache
    keeps the buffer's name and prefixes the written result instead, because the
    output is the same buffer rather than a longer one.
    """
    return (
        past_name.replace("past_key_values", "present"),
        past_name.replace("past.", "present."),
        past_name.replace("past_", "present_"),
        f"updated_{past_name}",
    )


def _constant_extent(dimension: Any) -> int | None:
    """Return *dimension* as an ``int``, or ``None`` when it is symbolic.

    An :class:`ir.Shape` entry is a plain ``int`` exactly when the extent is
    known; otherwise it is a ``SymbolicDim`` whose value is its name.
    """
    return dimension if isinstance(dimension, int) else None


def _static_cache_ports(model: ir.Model) -> dict[str, Any] | None:
    """Return the declared static-cache ABI of *model*, or ``None``.

    The two control ports are per-row integer vectors and so are shape-indistin-
    guishable from each other; they are matched against the ABI mobius mints in
    :mod:`mobius._constants`, never guessed from the graph. The buffer ports are
    then whichever cache inputs the scatter addresses.
    """
    inputs = {value.name: value for value in model.graph.inputs}
    write_indices = inputs.get(STATIC_CACHE_WRITE_INDICES)
    kv_lengths = inputs.get(STATIC_CACHE_KV_SEQUENCE_LENGTH)
    if write_indices is None or kv_lengths is None:
        return None
    buffers = {
        past.name: past
        for past, present in _model_cache_pairs(model)
        if present.name == f"updated_{past.name}"
    }
    if not buffers:
        raise ValueError(
            "decoder declares the static-cache control ports "
            f"{STATIC_CACHE_WRITE_INDICES!r}/{STATIC_CACHE_KV_SEQUENCE_LENGTH!r} but exposes "
            "no paired cache buffer to scatter into; regenerate the package with "
            "updated_<buffer> outputs for every static cache input"
        )
    axes = {
        node.attributes.get_int("axis", 0)
        for node in ir.traversal.RecursiveGraphIterator(model.graph)
        if node.op_type == "TensorScatter"
    }
    if axes - {STATIC_CACHE_SEQUENCE_AXIS}:
        raise ValueError(
            f"static cache buffers are addressed on axes {sorted(axes)}, but the mobius "
            f"static-cache ABI scatters along axis {STATIC_CACHE_SEQUENCE_AXIS}; the "
            "declared capacity axis and the graph disagree"
        )
    capacities = set()
    for buffer in buffers.values():
        shape = list(buffer.shape or [])
        if len(shape) <= STATIC_CACHE_SEQUENCE_AXIS:
            raise ValueError(
                f"static cache buffer {buffer.name!r} has rank {len(shape)}, which cannot "
                f"carry a capacity on axis {STATIC_CACHE_SEQUENCE_AXIS}"
            )
        capacity = _constant_extent(shape[STATIC_CACHE_SEQUENCE_AXIS])
        if capacity is None:
            raise ValueError(
                f"static cache buffer {buffer.name!r} declares a symbolic extent "
                f"{shape[STATIC_CACHE_SEQUENCE_AXIS]!r} on its capacity axis; an "
                "indexed scatter is only meaningful against one constant capacity"
            )
        capacities.add(capacity)
    if len(capacities) != 1:
        raise ValueError(
            f"static cache buffers declare conflicting capacities {sorted(capacities)}; "
            "one write cursor cannot address buffers of different lengths"
        )
    return {
        "write_indices": STATIC_CACHE_WRITE_INDICES,
        "kv_sequence_length": STATIC_CACHE_KV_SEQUENCE_LENGTH,
        "buffers": buffers,
        "capacity": capacities.pop(),
    }


def _kv_storage_contract(model: ir.Model) -> dict[str, Any]:
    """Derive physical KV storage from the admitted model interface.

    Shared KV is a runtime I/O-binding contract: past and present ports bind the
    same full-capacity OrtValue. That is only sound when the graph's attention
    operator takes the *logical* cache length as a separate input, so it can
    ignore the unwritten tail of a capacity-sized buffer. ``GroupQueryAttention``
    does (``seqlens_k`` / ``total_sequence_length``), and a paged layout carries
    its lengths in the block tables.

    The standard ONNX ``Attention`` operator does not: it concatenates ``past_key``
    with the current key and *derives* ``total_sequence_length`` from the past
    tensor's own second-to-last dimension. Binding a capacity-sized buffer there
    would both attend over unwritten slots and make the attention mask (sized to
    the real length) disagree with the derived total length, which ORT rejects.
    Such a graph therefore grows its cache by concatenation and must be declared
    ``dynamic``.
    """
    input_names = {value.name.lower() for value in model.graph.inputs}
    paged = any(
        marker in name
        for name in input_names
        for marker in ("block_table", "block_tables", "page_table", "page_tables")
    )
    has_cache = bool(_model_cache_pairs(model))
    if paged:
        storage = "paged"
    elif has_cache and _consumes_explicit_cache_length(model):
        storage = "shared_buffer"
    else:
        storage = "dynamic"
    return {
        "paging": "paged" if paged else "none",
        # Row compaction is semantic for every batched KV layout: the runtime
        # applies one row permutation to slot identity, KV, RNG, and loop state.
        "compaction": has_cache,
        "storage": storage,
    }


#: Attention operators that accept a capacity-sized KV buffer plus an explicit
#: logical cache length, and so can be bound to a preallocated shared buffer.
_CAPACITY_ADDRESSABLE_ATTENTION = frozenset(
    {
        ("com.microsoft", "GroupQueryAttention"),
        ("com.microsoft", "PagedAttention"),
        ("com.microsoft", "SparseAttention"),
    }
)


def _consumes_explicit_cache_length(model: ir.Model) -> bool:
    """Report whether every cache consumer takes the logical cache length as input.

    Returns ``False`` when the model has no cache consumers at all, because a
    graph that never reads ``past_key_values.*`` cannot promise capacity-safe
    behaviour it does not exercise.
    """
    cache_values = {
        past.name for past, _ in _model_cache_pairs(model) if past.name is not None
    }
    # A static buffer's capacity safety comes from its declared write cursor and
    # logical lengths, not from the attention operator's signature, so its
    # scatter consumer must not veto the appending caches' storage class.
    static = _static_cache_ports(model)
    if static is not None:
        cache_values -= set(static["buffers"])
    if not cache_values:
        return False
    consumers = {
        (node.domain, node.op_type)
        for node in ir.traversal.RecursiveGraphIterator(model.graph)
        for value in node.inputs
        if value is not None and value.name in cache_values
    }
    return bool(consumers) and consumers <= _CAPACITY_ADDRESSABLE_ATTENTION


def _aliasing_for_storage(storage: str) -> str:
    """Translate a physical storage class into the semantic aliasing contract.

    Shared-buffer and paged layouts let the runtime bind ``present`` onto the
    same allocation as ``past``; the metadata only declares that doing so is
    *legal*, never that the runtime must.  A growable cache returns a fresh,
    longer tensor each step and must never be aliased onto its own input.
    """
    return "permitted" if storage in {"shared_buffer", "paged"} else "forbidden"


def _annotated_alias(alias: dict[str, Any]) -> dict[str, Any]:
    """Add the half and layer a state port pair carries, when the name states them.

    A layer's key buffer and its value buffer are the same shape and the same
    dtype, and a cell's label is producer-chosen so its lexicographic order is
    not the layer order (``cache_10`` sorts before ``cache_2``). A consumer that
    paired these positionally would silently transpose two layers' caches. Both
    facts are recoverable only here, from the port names this producer minted,
    so both are declared on the alias.

    They are declared together or not at all: a port outside the two attention
    cache ABIs — recurrent state, a convolution cache — has no half and no layer
    to state, and inventing an index for it would corrupt the very ordering the
    index exists to fix.
    """
    name = str(alias.get("input", ""))
    half = _cache_half(name)
    if half is None:
        return alias
    return {**alias, "role": half, "layer": _cache_layer_index(name, 0)}


def _state_group(
    *,
    ports: dict[str, Any],
    sequence_axis: int,
    logical_lengths: str | None = None,
    storage: str = "shared_buffer",
    kind: str = "full_attention",
    aliasing: str | None = None,
    layout: str = "bnsh",
) -> dict[str, Any]:
    """Describe one semantic state group of the serving state service.

    ``aliasing`` is a property of the admitted graph, not a runtime preference:
    a shared full-capacity buffer lets the component write ``present`` straight
    into the ``past`` binding, while a growable cache returns a fresh, longer
    tensor each step and must never be aliased onto its own input.
    """
    annotated_ports = {
        component: {cell: _annotated_alias(alias) for cell, alias in aliases.items()}
        for component, aliases in ports.items()
    }
    _validate_attention_alias_layer_sets({kind: annotated_ports})
    group: dict[str, Any] = {
        "kind": kind,
        "sequence_axis": sequence_axis,
        "layout": layout,
        "aliasing": (aliasing if aliasing is not None else _aliasing_for_storage(storage)),
        "reuse": {"prefix_reusable": True, "evictable_prefix": False},
        "capabilities": {"snapshot": True, "fork": True},
        "ports": annotated_ports,
    }
    if logical_lengths is not None:
        group["logical_lengths"] = logical_lengths
    return group


_LAYER_STATE_KINDS = {
    "sliding_attention": "sliding_attention",
    "full_attention": "full_attention",
    "chunked_attention": "sliding_attention",
}


def _state_aliasing(kv_contract: dict[str, Any]) -> str:
    """Translate a physical KV storage contract into the aliasing contract."""
    return _aliasing_for_storage(str(kv_contract["storage"]))


def _cache_layer_index(port_name: str, fallback: int) -> int:
    """Recover the decoder layer index from a cache port name.

    Both cache ABIs encode the layer in the port name — ``past_key_values.N.key``
    for an appending cache, ``key_cache.N`` for a static one — because a hybrid
    decoder's cache-owning layers are a subset of its layers, so a port's
    position in the port list is not its layer.
    """
    match = re.search(r"\.(\d+)\.(?:key|value)$|(?:key|value)_cache\.(\d+)$", port_name)
    if match is None:
        return fallback
    return int(match.group(1) or match.group(2))


def _cache_half(port_name: str) -> str | None:
    """Recover which half of a split attention cache a port carries.

    A layer's key buffer and its value buffer are the same shape and the same
    dtype, so nothing downstream can tell them apart once they are in a list.
    This producer minted both names, in the same two ABIs ``_cache_layer_index``
    reads, so it can say which is which; a port outside those two spellings gets
    no half, which is the right answer for recurrent and latent state that has
    no halves to distinguish.
    """
    match = re.search(r"\.\d+\.(key|value)$|(key|value)_cache\.\d+$", port_name)
    if match is None:
        return None
    return match.group(1) or match.group(2)


def _state_group_kinds(config: Any, cache_pairs: list[tuple[ir.Value, ir.Value]]) -> list[str]:
    """Return the semantic ``StateKind`` of every KV cache cell.

    Hybrid models interleave sliding-window and full-attention layers, and the
    two are not interchangeable: only a sliding layer's oldest positions may be
    evicted.  Cache-owning layers form a contiguous prefix of the decoder (a
    KV-sharing suffix owns no cache at all), so the port's own layer index —
    not its position in the port list — selects the layer type.
    """
    layer_types = list(getattr(config, "layer_types", None) or [])
    default_kind = (
        "sliding_attention"
        if not layer_types and getattr(config, "sliding_window", None)
        else "full_attention"
    )
    kinds = []
    for index, (past, _) in enumerate(cache_pairs):
        layer = _cache_layer_index(past.name or "", index // 2)
        layer_type = layer_types[layer] if layer < len(layer_types) else None
        kinds.append(_LAYER_STATE_KINDS.get(str(layer_type), default_kind))
    return kinds


def _validate_attention_alias_layer_sets(
    grouped_ports: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> None:
    """Require split key/value aliases to cover the same numeric layers."""
    for group_name, components in grouped_ports.items():
        for component, aliases in components.items():
            layers_by_role: dict[str, dict[int, str]] = {"key": {}, "value": {}}
            for alias_name, alias in aliases.items():
                role = alias.get("role")
                if role not in layers_by_role:
                    continue
                layer = alias.get("layer")
                if not isinstance(layer, int):
                    raise TypeError(
                        f"state group {group_name!r} component {component!r} alias "
                        f"{alias_name!r} declares role {role!r} without a numeric layer"
                    )
                previous = layers_by_role[role].get(layer)
                if previous is not None:
                    raise ValueError(
                        f"state group {group_name!r} component {component!r} declares "
                        f"duplicate {role} aliases for layer {layer}: "
                        f"{previous!r} and {alias_name!r}"
                    )
                layers_by_role[role][layer] = alias_name

            key_layers = set(layers_by_role["key"])
            value_layers = set(layers_by_role["value"])
            if not key_layers and not value_layers:
                continue
            if key_layers != value_layers:
                raise ValueError(
                    f"state group {group_name!r} component {component!r} must declare "
                    "one key and one value alias for the same attention layers; "
                    f"key layers are {sorted(key_layers)}, "
                    f"value layers are {sorted(value_layers)}, "
                    f"missing value layers are {sorted(key_layers - value_layers)}, "
                    f"missing key layers are {sorted(value_layers - key_layers)}"
                )


def _state_service_groups(
    *,
    config: Any,
    cache_pairs: list[tuple[ir.Value, ir.Value]],
    ports: dict[str, dict[str, dict[str, str]]],
    sequence_axis: int,
    logical_lengths: str | None,
    aliasing: str,
    base_name: str,
    indexed_scatter: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build ``serving.state_service.groups`` plus each cell's owning group.

    One group per semantic kind *and update discipline*: a hybrid decoder
    therefore publishes distinct ``sliding_attention`` and ``full_attention``
    groups whose per-cell contracts carry their own geometry (Gemma 4's global
    layers are double-wide), and a decoder that appends some caches while
    scattering others into fixed buffers keeps those apart too, because the
    valid region of an appended buffer is its shape while the valid region of a
    scattered one is a declared prefix.  The group declares *semantics* only —
    eviction legality, aliasing legality, layout — never a storage class,
    allocator, or compaction algorithm, which are the runtime's to choose.

    ``indexed_scatter`` describes the static, fixed-capacity buffers: which
    cache inputs they are, the constant capacity they were built against, the
    cell carrying each row's write cursor, and the port that receives it.
    """
    indexed_scatter = indexed_scatter or {}
    indexed_inputs = set(indexed_scatter.get("buffers", ()))
    kinds = _state_group_kinds(config, cache_pairs)
    scattered = [(past.name or "") in indexed_inputs for past, _ in cache_pairs]
    # Group identity is (semantic kind, update discipline); the suffix only
    # appears when more than one identity is present, so a homogeneous decoder
    # keeps publishing exactly one group under its base name.
    identities = [
        (kind, "indexed_scatter" if is_scattered else "append")
        for kind, is_scattered in zip(kinds, scattered)
    ]
    distinct = sorted(set(identities))
    names = {
        identity: (base_name if len(distinct) == 1 else f"{base_name}_{identity[0]}")
        for identity in distinct
    }
    if len({*names.values()}) != len(distinct):
        names = {identity: f"{base_name}_{identity[0]}_{identity[1]}" for identity in distinct}
    cell_group = {}
    grouped_ports: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        name: {} for name in names.values()
    }
    for index, identity in enumerate(identities):
        cell = f"cache_{index}"
        cell_group[cell] = names[identity]
    # A cell's label is producer-chosen and its lexicographic order is not the
    # layer order, and a layer's key and value buffers are shape-identical, so
    # each alias carries the half and layer its port name states.
    for component, aliases in ports.items():
        for cell, alias in aliases.items():
            grouped_ports[cell_group[cell]].setdefault(component, {})[cell] = _annotated_alias(
                alias
            )
    _validate_attention_alias_layer_sets(grouped_ports)
    groups = {}
    for kind, update in distinct:
        is_scattered = update == "indexed_scatter"
        name = names[(kind, update)]
        group: dict[str, Any] = {
            "kind": kind,
            "sequence_axis": (STATIC_CACHE_SEQUENCE_AXIS if is_scattered else sequence_axis),
            "layout": STATIC_CACHE_LAYOUT if is_scattered else "bnsh",
        }
        group_lengths = indexed_scatter["logical_lengths"] if is_scattered else logical_lengths
        if group_lengths:
            group["logical_lengths"] = group_lengths
        if is_scattered:
            group["update"] = {
                "kind": "indexed_scatter",
                "write_indices": indexed_scatter["write_indices"],
                "capacity": indexed_scatter["capacity"],
                "write_indices_ports": dict.fromkeys(
                    grouped_ports[name], indexed_scatter["port"]
                ),
                # The graph-visible valid length is a second rank-1 integer
                # vector sitting beside the destinations, so it is equally
                # shape-indistinguishable and equally must be named rather than
                # inferred. Together these two entries and ``ports`` below are
                # the whole scatter ABI, which is why the package needs no
                # second copy of it outside the workflow.
                "kv_length_ports": dict.fromkeys(
                    grouped_ports[name], indexed_scatter["kv_length_port"]
                ),
            }
        # A scatter writes through its buffer by construction: the written result
        # *is* the input allocation, so aliasing is legal for every static group
        # regardless of what the appending caches in the same graph can do.
        group["aliasing"] = "permitted" if is_scattered else aliasing
        group["reuse"] = {
            "prefix_reusable": True,
            # Dropping the oldest positions is only semantics-preserving
            # for a windowed layer; a full-attention layer that loses its
            # prefix silently answers a different question.
            "evictable_prefix": kind == "sliding_attention",
        }
        group["ports"] = grouped_ports[name]
        groups[name] = group
    return groups, cell_group


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
    talker_kv = _kv_storage_contract(talker)
    predictor_kv = _kv_storage_contract(predictor)
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
    if talker_caches or predictor_caches:
        pkg.add_policy_component("cache_length_update", build_integer_add())
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
    batch_int = _request_aligned({"dtype": "int64", "rank": 1, "shape": [batch]})
    batch_bool = _request_aligned({"dtype": "bool", "rank": 1, "shape": [batch]})
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}
    inputs = {
        "request.prompt_tokens": {
            "contract": _contract(prompt),
            "role": {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"},
            "source": {"kind": "request", "field": "prompt_tokens"},
            "required": True,
        },
        "request.max_iterations": {
            "contract": control_int,
            "role": {"kind": "runtime", "version": "1.0", "role": "max_output_tokens"},
            "source": {"kind": "request", "field": "max_output_tokens"},
            "required": True,
        },
        "package.false": {
            "contract": _request_aligned(batch_bool),
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
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": num_groups - 2,
        },
        "package.predictor_context_limit": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": num_groups,
        },
        "package.predictor_mask_limit": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": num_groups + 1,
        },
        "package.talker_context_limit": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": int(getattr(config, "max_position_embeddings", 4096)),
        },
        "package.one_control": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.zero_batch": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0,
        },
        "package.one_batch": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.true": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": True,
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
            *(
                [
                    _invoke(
                        "cache_length_update",
                        {
                            "left": "state.predictor_cache_lengths.inner",
                            "right": "package.one_batch",
                        },
                        {"total": "predictor_cache_lengths.next"},
                    )
                ]
                if predictor_caches
                else []
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
    if predictor_caches:
        inner_carried.append(
            {
                "cell": "predictor_cache_lengths",
                "current": "package.zero_batch",
                "body_input": "state.predictor_cache_lengths.inner",
                "body_output": "predictor_cache_lengths.next",
                "next": "predictor_cache_lengths.final",
            }
        )
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
        *(
            [
                _invoke(
                    "cache_length_update",
                    {
                        "left": "state.talker_cache_lengths.body",
                        "right": "package.one_batch",
                    },
                    {"total": "talker_cache_lengths.next"},
                ),
                _invoke(
                    "cache_length_update",
                    {"left": "package.zero_batch", "right": "package.one_batch"},
                    {"total": "accepted_len.next"},
                ),
            ]
            if talker_caches or predictor_caches
            else []
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
                "increment": "package.one_control",
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
                "increment": "package.one_control",
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
                "increment": "package.one_control",
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
        "active": {
            "contract": batch_bool,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.true",
            "recurrence": {"kind": "invariant"},
        },
        "done": {
            "contract": batch_bool,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.false",
            "recurrence": {"kind": "invariant"},
        },
        "accepted_len": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.zero_batch",
            "recurrence": {"kind": "invariant"},
        },
        "talker_cache_lengths": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.zero_batch",
            "recurrence": {"kind": "invariant"},
        },
        "predictor_cache_lengths": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.zero_batch",
            "recurrence": {"kind": "invariant"},
        },
    }
    for index, (past, present) in enumerate(talker_caches):
        state[f"talker_cache_{index}"] = {
            "contract": _request_aligned(_contract(past)),
            "class": "semantic",
            "scope": "invocation",
            "initializer": f"talker.setup.{present.name}",
            "recurrence": {
                "kind": "bounded",
                "axis": 2,
                "max": "package.talker_context_limit",
            },
            "service_group": "talker_cache",
            "management": "runtime",
            "release_boundary": "invocation",
        }
    for index, (past, present) in enumerate(predictor_caches):
        state[f"predictor_cache_{index}"] = {
            "contract": _request_aligned(_contract(past)),
            "class": "semantic",
            "scope": "invocation",
            "initializer": f"frame.predictor.{present.name}",
            "recurrence": {
                "kind": "bounded",
                "axis": 2,
                "max": "package.predictor_context_limit",
            },
            "service_group": "predictor_cache",
            "management": "runtime",
            "release_boundary": "invocation",
        }

    outer_carried = [
        {
            "cell": "active",
            "current": "package.true",
            "body_input": "state.active.body",
            "body_output": "state.active.body",
            "next": "state.active.final",
        },
        {
            "cell": "done",
            "current": "package.false",
            "body_input": "state.done.body",
            "body_output": "state.done.body",
            "next": "state.done.final",
        },
        {
            "cell": "accepted_len",
            "current": "package.zero_batch",
            "body_input": "state.accepted_len.body",
            "body_output": (
                "accepted_len.next"
                if talker_caches or predictor_caches
                else "state.accepted_len.body"
            ),
            "next": "state.accepted_len.final",
        },
        {
            "cell": "talker_cache_lengths",
            "current": "package.zero_batch",
            "body_input": "state.talker_cache_lengths.body",
            "body_output": (
                "talker_cache_lengths.next"
                if talker_caches or predictor_caches
                else "state.talker_cache_lengths.body"
            ),
            "next": "state.talker_cache_lengths.final",
        },
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
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
                *(
                    ["serving_service_contract", "bounded_state_recurrence"]
                    if talker_caches or predictor_caches
                    else []
                ),
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
        **(
            {
                "serving": {
                    "active": "active",
                    "done": "done",
                    "accepted_len": "accepted_len",
                    "state_service": {
                        "groups": {
                            **(
                                {
                                    "talker_cache": _state_group(
                                        sequence_axis=2,
                                        logical_lengths="talker_cache_lengths",
                                        storage=talker_kv["storage"],
                                        ports={
                                            "talker": {
                                                f"talker_cache_{index}": {
                                                    "input": past.name,
                                                    "output": present.name,
                                                }
                                                for index, (past, present) in enumerate(
                                                    talker_caches
                                                )
                                            }
                                        },
                                    )
                                }
                                if talker_caches
                                else {}
                            ),
                            **(
                                {
                                    "predictor_cache": _state_group(
                                        sequence_axis=2,
                                        logical_lengths="predictor_cache_lengths",
                                        storage=predictor_kv["storage"],
                                        ports={
                                            "code_predictor": {
                                                f"predictor_cache_{index}": {
                                                    "input": past.name,
                                                    "output": present.name,
                                                }
                                                for index, (past, present) in enumerate(
                                                    predictor_caches
                                                )
                                            }
                                        },
                                    )
                                }
                                if predictor_caches
                                else {}
                            ),
                        },
                    },
                }
            }
            if talker_caches or predictor_caches
            else {}
        ),
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
    batch_int = _request_aligned({"dtype": "int64", "rank": 1, "shape": [batch]})
    batch_bool = _request_aligned({"dtype": "bool", "rank": 1, "shape": [batch]})
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}
    inputs: dict[str, Any] = {
        "request.prompt_tokens": {
            "contract": _contract(prompt_input),
            "role": {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"},
            "source": {"kind": "request", "field": "prompt_tokens"},
            "required": True,
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
        "package.code_groups": {
            "contract": control_int,
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
            "contract": control_int,
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
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
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


def _accumulated_contract(contract: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Rename the trailing axis of an append-mode output to a private symbol.

    ``mode: append`` emits concatenate chunks along the last axis, so the final
    workflow value has ``steps * chunk`` entries there. Reusing the chunk's
    symbol (for example ``width``) would rebind it to the accumulated extent and
    contradict every other value that shares it, so the accumulating axis gets a
    symbol of its own.
    """
    shape = list(contract["shape"])
    shape[-1] = symbol
    return {**contract, "shape": shape}


def _cache_cell(port: str) -> str:
    """State-cell name for a ``conv_cache.<path>`` decoder port."""
    return "conv_cache_" + port[len("conv_cache.") :].replace(".", "_")


CONV_CACHE_SCALE_METADATA = "mobius.conv_cache.spatial_scale."


def _cache_spatial_scale(model: Any, value: ir.Value) -> int:
    """Spatial upsampling a conv-cache port has undergone relative to the latent.

    A causal video decoder caches activations at several resolutions, and the
    workflow has to allocate the empty first-chunk caches at exactly those
    resolutions or the decoder's concatenation fails. The producing task records
    the ratio on the model, because symbolic dimension names are not a reliable
    channel: shape inference is free to replace a declared ``8*latent_height``
    with an anonymous symbol when it unifies the port with an internal value.
    """
    recorded = model.metadata_props.get(f"{CONV_CACHE_SCALE_METADATA}{value.name}")
    if recorded is not None:
        return int(recorded)
    dimension = str(list(value.shape)[3])
    return int(dimension.split("*")[0]) if "*" in dimension else 1


def build_diffusion_workflow_metadata(
    pkg: Any,
    *,
    num_inference_steps: int,
    schedule: list[float] | None = None,
    timesteps: list[float] | None = None,
    solver: str = "euler",
    scale_model_input: bool = True,
    initial_state_scale: float = 1.0,
    decoder_input_scale: float = 1.0,
    guidance_scale: float | None = None,
    latent_source: str = "application",
    latent_row_shape: list[int] | None = None,
) -> dict[str, Any]:
    """Build a fixed-schedule diffusion workflow with explicit latent state.

    Everything the reference sampler hides in Python attributes becomes an
    explicit part of the workflow: the sigma schedule and timestep table are
    constant components, the step index is the loop induction value, a
    multistep solver's previous data estimate is a declared state cell, and the
    RNG counter is an ordinary integer tensor. Classifier-free guidance is two
    denoiser invocations plus a combine component rather than a hidden batch
    doubling, so every value stays request-aligned on axis 0.
    """
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    if solver not in SOLVER_BUILDERS:
        raise ValueError(f"unsupported diffusion solver {solver!r}")
    if latent_source not in {"application", "seed"}:
        raise ValueError(f"unsupported diffusion latent source {latent_source!r}")
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
        raise ValueError("diffusion workflow requires matching rank-4 latent/estimate")
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
    conditioned = text_encoder is not None and conditioning_output is not None
    if guidance_scale is not None and not conditioned:
        raise ValueError("classifier-free guidance requires a conditioned denoiser")
    if latent_source == "seed" and not latent_row_shape:
        raise ValueError("a seeded latent initializer requires an explicit row shape")

    solver_component = SOLVER_BUILDERS[solver](sample_input.dtype)
    solver_ports = {value.name for value in solver_component.model.graph.inputs}
    carries_history = "history" in solver_ports

    attach_policy_components(pkg, PolicyCapabilities())
    pkg.add_policy_component("solver_step", solver_component)
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    if scale_model_input:
        pkg.add_policy_component(
            "model_input_scale", build_euler_model_input(sample_input.dtype)
        )
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
    # A sampler whose state already lives in the denoiser's space, and a VAE whose
    # latents are unnormalized, need no rescaling step at all; only emit the
    # constant and the multiply that the pipeline actually performs.
    scales_initial_state = not math.isclose(initial_state_scale, 1.0)
    scales_decoder_input = not math.isclose(decoder_input_scale, 1.0)
    if scales_initial_state or scales_decoder_input:
        pkg.add_policy_component("tensor_scale", build_tensor_scale(sample_input.dtype))
    if scales_initial_state:
        pkg.add_policy_component(
            "initial_state_scale", build_scalar_constant(initial_state_scale)
        )
    if scales_decoder_input:
        pkg.add_policy_component(
            "decoder_input_scale", build_scalar_constant(decoder_input_scale)
        )
    if carries_history:
        pkg.add_policy_component("history_initializer", build_zeros_like(sample_input.dtype))
    if guidance_scale is not None:
        pkg.add_policy_component(
            "guidance_combine", build_guidance_combine(sample_input.dtype)
        )
    if latent_source == "seed":
        pkg.add_policy_component(
            "latent_row_shape", build_shape_constant(list(latent_row_shape or ()))
        )
        pkg.add_policy_component("latent_noise", build_counter_rng_normal(sample_input.dtype))

    batch = _contract(sample_input)["shape"][0]
    latent_contract = _contract(sample_input)
    row_int = _request_aligned({"dtype": "int64", "rank": 1, "shape": [batch]})
    row_float = _request_aligned({"dtype": "float32", "rank": 1, "shape": [batch]})
    batch_int = _request_aligned({"dtype": "int64", "rank": 1, "shape": [batch]})
    batch_bool = _request_aligned({"dtype": "bool", "rank": 1, "shape": [batch]})
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}
    inputs: dict[str, Any] = {
        "request.max_iterations": {
            "contract": control_int,
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
    if scales_initial_state:
        setup_nodes.append(
            _invoke("initial_state_scale", {}, {"value": "diffusion.initial_scale"})
        )
    if scales_decoder_input:
        setup_nodes.append(
            _invoke("decoder_input_scale", {}, {"value": "diffusion.decoder_scale"})
        )
    outputs: dict[str, Any] = {
        "image": {
            "contract": _contract(vae_output),
            "role": "image",
            "value_range": "negative_one_to_one",
            "stage": "pre_adapter",
        },
        "latent": {
            "contract": latent_contract,
            "role": "tensor",
            "stage": "pre_adapter",
        },
        "noise_estimate": {
            "contract": _accumulated_contract(latent_contract, "noise_estimate_width"),
            "role": "tensor",
            "stage": "pre_adapter",
        },
        "latent_trajectory": {
            "contract": _accumulated_contract(latent_contract, "trajectory_width"),
            "role": "tensor",
            "stage": "pre_adapter",
        },
    }

    if latent_source == "application":
        inputs["request.noise"] = {
            "contract": latent_contract,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "noise"},
            "required": True,
            "externally_suppliable": True,
        }
        noise_value = "request.noise"
    else:
        inputs["request.seed"] = {
            "contract": row_int,
            "role": {"kind": "runtime", "version": "1.0", "role": "seed"},
            "source": {"kind": "request", "field": "seed"},
            "required": False,
            "default": 0,
            "externally_suppliable": True,
        }
        inputs["package.rng_offset"] = {
            "contract": row_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0,
        }
        noise_value = "diffusion.noise"
        setup_nodes.append(
            _invoke("latent_row_shape", {}, {"shape": "diffusion.latent_row_shape"})
        )
        setup_nodes.append(
            _invoke(
                "latent_noise",
                {
                    "seed": "request.seed",
                    "offset": "package.rng_offset",
                    "row_shape": "diffusion.latent_row_shape",
                },
                {"noise": noise_value, "next_offset": "diffusion.rng_offset"},
            )
        )
        outputs["rng_offset"] = {
            "contract": row_int,
            "role": "tensor",
            "stage": "pre_adapter",
        }

    initial_state_value = noise_value
    if scales_initial_state:
        initial_state_value = "diffusion.initial_state"
        setup_nodes.append(
            _invoke(
                "tensor_scale",
                {"tensor": noise_value, "scale": "diffusion.initial_scale"},
                {"scaled": initial_state_value},
            )
        )

    conditioning_value = None
    unconditional_value = None
    if conditioned:
        assert text_encoder is not None
        assert conditioning_output is not None
        conditioning_value = "conditioning.hidden_states"
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
                "source": (
                    {"kind": "request", "field": "prompt_tokens"}
                    if index == 0
                    else {"kind": "application", "name": value.name}
                ),
                "required": True,
                "externally_suppliable": True,
            }
            text_inputs[value.name] = name
        setup_nodes.append(
            _invoke(
                text_name,
                text_inputs,
                {conditioning_output.name: conditioning_value},
            )
        )
        if guidance_scale is not None:
            unconditional_value = "conditioning.unconditional"
            negative_inputs = {}
            for value in text_encoder.graph.inputs:
                name = f"request.negative_{value.name}"
                inputs[name] = {
                    "contract": _contract(value),
                    "role": {"kind": "opaque"},
                    "source": {"kind": "application", "name": f"negative_{value.name}"},
                    "required": True,
                    "externally_suppliable": True,
                }
                negative_inputs[value.name] = name
            inputs["request.guidance_scale"] = {
                "contract": row_float,
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": "guidance_scale"},
                "required": False,
                "default": float(guidance_scale),
            }
            setup_nodes.append(
                _invoke(
                    text_name,
                    negative_inputs,
                    {conditioning_output.name: unconditional_value},
                )
            )

    state: dict[str, Any] = {
        "latent": {
            "contract": latent_contract,
            "scope": "invocation",
            "initializer": initial_state_value,
            "recurrence": {"kind": "invariant"},
        }
    }
    carried: list[dict[str, Any]] = [
        {
            "cell": "latent",
            "current": initial_state_value,
            "body_input": "state.latent.body",
            "body_output": "latent.body",
            "next": "latent.final",
            "read_effect": _effect("state:latent.0", "state:latent.read"),
            "write_effect": _effect("state:latent.read", "state:latent.1"),
        }
    ]
    if carries_history:
        setup_nodes.append(
            _invoke(
                "history_initializer",
                {"reference": initial_state_value},
                {"zeros": "diffusion.initial_history"},
            )
        )
        state["history"] = {
            "contract": latent_contract,
            "scope": "invocation",
            "initializer": "diffusion.initial_history",
            "recurrence": {"kind": "invariant"},
        }
        carried.append(
            {
                "cell": "history",
                "current": "diffusion.initial_history",
                "body_input": "state.history.body",
                "body_output": "history.body",
                "next": "history.final",
                "read_effect": _effect("state:history.0", "state:history.read"),
                "write_effect": _effect("state:history.read", "state:history.1"),
            }
        )
    setup_nodes.append(
        _invoke(
            "continue_predicate",
            {"done": "package.false"},
            {"continue": "setup.continue"},
        )
    )

    body_nodes: list[dict[str, Any]] = [
        _invoke(
            "schedule_lookup",
            {"schedule": "diffusion.timesteps", "step": "loop.iteration"},
            {"timestep": "diffusion.timestep"},
        )
    ]
    if scale_model_input:
        model_input_value = "diffusion.model_input"
        body_nodes.append(
            _invoke(
                "model_input_scale",
                {
                    "sample": "state.latent.body",
                    "step": "loop.iteration",
                    "schedule": "diffusion.schedule",
                },
                {"model_input": model_input_value},
            )
        )
    else:
        model_input_value = "state.latent.body"

    def denoiser_call(conditioning: str | None, estimate: str) -> dict[str, Any]:
        call_inputs = {
            sample_input.name: model_input_value,
            timestep_input.name: "diffusion.timestep",
        }
        if conditioning_input is not None and conditioning is not None:
            call_inputs[conditioning_input.name] = conditioning
        return _invoke(denoiser_name, call_inputs, {estimate_output.name: estimate})

    if guidance_scale is None:
        body_nodes.append(denoiser_call(conditioning_value, "denoiser.estimate"))
    else:
        body_nodes.append(denoiser_call(unconditional_value, "denoiser.unconditional"))
        body_nodes.append(denoiser_call(conditioning_value, "denoiser.conditional"))
        body_nodes.append(
            _invoke(
                "guidance_combine",
                {
                    "unconditional": "denoiser.unconditional",
                    "conditional": "denoiser.conditional",
                    "scale": "request.guidance_scale",
                },
                {"estimate": "denoiser.estimate"},
            )
        )

    solver_inputs = {
        "sample": "state.latent.body",
        "step": "loop.iteration",
        "schedule": "diffusion.schedule",
    }
    solver_inputs["estimate" if carries_history else "derivative"] = "denoiser.estimate"
    solver_outputs = {"next_state": "latent.body"}
    if carries_history:
        solver_inputs["history"] = "state.history.body"
        solver_outputs["next_history"] = "history.body"
    body_nodes.append(_invoke("solver_step", solver_inputs, solver_outputs))
    body_nodes.extend(
        [
            {
                "kind": "emit",
                "value": "denoiser.estimate",
                "output": "noise_estimate",
                "mode": "append",
                "effect_name": "emit",
                "effect": _effect("emit.0", "emit.1"),
            },
            {
                "kind": "emit",
                "value": "latent.body",
                "output": "latent_trajectory",
                "mode": "append",
                "effect_name": "emit",
                "effect": _effect("emit.1", "emit.2"),
            },
            _invoke(
                "continue_predicate",
                {"done": "package.false"},
                {"continue": "loop.continue"},
            ),
        ]
    )

    decoder_input_value = "latent.final"
    tail_nodes: list[dict[str, Any]] = []
    if scales_decoder_input:
        decoder_input_value = "diffusion.decoder_input"
        tail_nodes.append(
            _invoke(
                "tensor_scale",
                {"tensor": "latent.final", "scale": "diffusion.decoder_scale"},
                {"scaled": decoder_input_value},
            )
        )
    tail_nodes.extend(
        [
            _invoke(
                vae_name,
                {vae_input.name: decoder_input_value},
                {vae_output.name: "vae.image"},
            ),
            {
                "kind": "emit",
                "value": "latent.final",
                "output": "latent",
                "mode": "replace",
                "effect_name": "emit",
                "effect": _effect("emit.2", "emit.3"),
            },
            {
                "kind": "emit",
                "value": "vae.image",
                "output": "image",
                "mode": "replace",
                "effect_name": "emit",
                "effect": _effect("emit.3", "emit.4"),
            },
        ]
    )
    if latent_source == "seed":
        tail_nodes.append(
            {
                "kind": "emit",
                "value": "diffusion.rng_offset",
                "output": "rng_offset",
                "mode": "replace",
                "effect_name": "emit",
                "effect": _effect("emit.4", "emit.5"),
            }
        )

    workflow = {
        "manifest": {
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
            ],
        },
        "inputs": inputs,
        "outputs": outputs,
        "components": {
            name: _component(model, _artifact(name, len(pkg))) for name, model in pkg.items()
        },
        "state": state,
        "initial_effects": {
            "emit": "emit.0",
            **{f"state:{cell}": f"state:{cell}.0" for cell in state},
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
                    "carried": carried,
                },
                *tail_nodes,
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
    solver: str = "euler",
    scale_model_input: bool = True,
    initial_state_scale: float = 1.0,
    decoder_input_scale: float = 1.0,
    guidance_scale: float | None = None,
    latent_source: str = "application",
    latent_row_shape: list[int] | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_diffusion_workflow_metadata(
        pkg,
        num_inference_steps=num_inference_steps,
        schedule=schedule,
        timesteps=timesteps,
        solver=solver,
        scale_model_input=scale_model_input,
        initial_state_scale=initial_state_scale,
        decoder_input_scale=decoder_input_scale,
        guidance_scale=guidance_scale,
        latent_source=latent_source,
        latent_row_shape=latent_row_shape,
    )
    pkg.save_policy_components(output_dir)
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


def _application_input(value: ir.Value, name: str | None = None) -> dict[str, Any]:
    """Declare a workflow input supplied by the host application."""
    return {
        "contract": _contract(value),
        "role": {"kind": "opaque"},
        "source": {"kind": "application", "name": name or value.name},
        "required": True,
    }


def build_image_edit_workflow_metadata(
    pkg: Any,
    *,
    num_inference_steps: int,
    schedule: list[float],
    timesteps: list[float],
    guidance_scale: float,
) -> dict[str, Any]:
    """Build a flow-matching image-edit workflow with true classifier-free guidance.

    Emitted pipeline (Qwen Image Edit and any package with the same component
    shape)::

        vae_encoder(source pixels) -> pack -> source tokens          [setup]
        loop:
            timestep      = timesteps[i]
            model_input   = concat([target tokens, source tokens], 1)
            cond          = denoiser(model_input, positive prompt)
            uncond        = denoiser(model_input, negative prompt)
            estimate      = true_cfg(cond, uncond)
            target tokens = target + (sigma[i+1] - sigma[i]) * estimate
        unpack(target tokens) -> vae_decoder -> image

    The denoiser slices its own output back to the target token count using the
    ``target_sequence_length`` port, so the loop state stays rank-3 and the
    source tokens stay loop-invariant.
    """
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    if len(schedule) != num_inference_steps + 1:
        raise ValueError("image-edit schedule must contain num_inference_steps + 1 values")
    if len(timesteps) != num_inference_steps:
        raise ValueError("image-edit timesteps must contain num_inference_steps values")

    denoiser_name = "transformer"
    denoiser = pkg[denoiser_name]
    encoder = pkg["vae_encoder"]
    decoder = pkg["vae_decoder"]

    ports = {value.name: value for value in denoiser.graph.inputs}
    sample_input = ports["sample"]
    timestep_input = ports["timestep"]
    estimate_output = denoiser.graph.outputs[0]
    encoder_input = encoder.graph.inputs[0]
    encoder_output = encoder.graph.outputs[0]
    decoder_input = decoder.graph.inputs[0]
    decoder_output = decoder.graph.outputs[0]
    dtype = sample_input.dtype

    # Loop state is the target token block only; the denoiser input additionally
    # carries the source tokens, so the two contracts differ in sequence length.
    latent_contract = {
        "dtype": _contract(sample_input)["dtype"],
        "rank": 3,
        "shape": ["batch", "target_sequence_length", _contract(sample_input)["shape"][2]],
    }

    attach_policy_components(pkg, PolicyCapabilities())
    pkg.add_policy_component("solver_step", build_flow_match_solver_step(dtype))
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    pkg.add_policy_component("diffusion_schedule", build_schedule_constant(schedule))
    pkg.add_policy_component("diffusion_timesteps", build_schedule_constant(timesteps))
    pkg.add_policy_component("schedule_lookup", build_schedule_lookup(timestep_input.dtype))
    pkg.add_policy_component("pack_latents", build_pack_latents_2x2(dtype))
    pkg.add_policy_component("unpack_latents", build_unpack_latents_2x2(dtype))
    pkg.add_policy_component("sequence_concat", build_sequence_concat(dtype))
    pkg.add_policy_component("true_cfg", build_true_cfg(dtype, guidance_scale=guidance_scale))

    batch = latent_contract["shape"][0]
    batch_int = {"dtype": "int64", "rank": 1, "shape": [batch]}
    batch_bool = {"dtype": "bool", "rank": 1, "shape": [batch]}
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}

    conditioning_ports = ("encoder_hidden_states", "encoder_hidden_states_mask")
    rotary_ports = ("image_rotary_cos", "image_rotary_sin")
    text_rotary_ports = ("text_rotary_cos", "text_rotary_sin")

    inputs: dict[str, Any] = {
        "request.latent": {
            "contract": latent_contract,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "latent"},
            "required": True,
        },
        "request.source_pixels": _application_input(encoder_input, "source_pixels"),
        "request.target_sequence_length": _application_input(ports["target_sequence_length"]),
        "request.latent_height": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "latent_height"},
            "required": True,
        },
        "request.latent_width": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "latent_width"},
            "required": True,
        },
        "request.max_iterations": {
            "contract": control_int,
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
    for port_name in rotary_ports:
        inputs[f"request.{port_name}"] = _application_input(ports[port_name])
    # Positive and negative conditioning are separate application inputs: the two
    # prompts tokenize to different lengths, so they cannot share a contract dim.
    for prefix in ("positive", "negative"):
        for port_name in conditioning_ports + text_rotary_ports:
            contract = _contract(ports[port_name])
            contract["shape"] = [
                f"{prefix}_text_sequence_length"
                if isinstance(dim, str) and "text" in dim
                else dim
                for dim in contract["shape"]
            ]
            inputs[f"request.{prefix}_{port_name}"] = {
                "contract": contract,
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": f"{prefix}_{port_name}"},
                "required": True,
            }

    setup_nodes: list[dict[str, Any]] = [
        _invoke("diffusion_schedule", {}, {"schedule": "diffusion.schedule"}),
        _invoke("diffusion_timesteps", {}, {"schedule": "diffusion.timesteps"}),
        _invoke(
            "vae_encoder",
            {encoder_input.name: "request.source_pixels"},
            {encoder_output.name: "source.latent"},
        ),
        _invoke(
            "pack_latents",
            {"latent_sample": "source.latent"},
            {"packed_latent": "source.tokens"},
        ),
        _invoke(
            "continue_predicate",
            {"done": "package.false"},
            {"continue": "setup.continue"},
        ),
    ]

    def denoise(prefix: str, output: str) -> dict[str, Any]:
        feeds = {
            sample_input.name: "diffusion.model_input",
            timestep_input.name: "diffusion.timestep",
            "target_sequence_length": "request.target_sequence_length",
        }
        for port_name in rotary_ports:
            feeds[port_name] = f"request.{port_name}"
        for port_name in conditioning_ports + text_rotary_ports:
            feeds[port_name] = f"request.{prefix}_{port_name}"
        return _invoke(denoiser_name, feeds, {estimate_output.name: output})

    body_nodes: list[dict[str, Any]] = [
        _invoke(
            "schedule_lookup",
            {"schedule": "diffusion.timesteps", "step": "loop.iteration"},
            {"timestep": "diffusion.timestep"},
        ),
        _invoke(
            "sequence_concat",
            {"target": "state.latent.body", "source": "source.tokens"},
            {"sequence": "diffusion.model_input"},
        ),
        denoise("positive", "denoiser.conditional"),
        denoise("negative", "denoiser.unconditional"),
        _invoke(
            "true_cfg",
            {
                "conditional": "denoiser.conditional",
                "unconditional": "denoiser.unconditional",
            },
            {"estimate": "denoiser.estimate"},
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

    latent_effect = "state:latent"
    workflow = {
        "manifest": {
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
                "contract": _contract(decoder_output),
                "role": "image",
                "value_range": "negative_one_to_one",
                "stage": "pre_adapter",
            }
        },
        "components": {
            name: _component(model, _artifact(name, len(pkg))) for name, model in pkg.items()
        },
        "state": {
            "latent": {
                "contract": latent_contract,
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
                    "unpack_latents",
                    {
                        "packed_latent": "latent.final",
                        "height": "request.latent_height",
                        "width": "request.latent_width",
                    },
                    {"latent_sample": "vae.latent"},
                ),
                _invoke(
                    "vae_decoder",
                    {decoder_input.name: "vae.latent"},
                    {decoder_output.name: "vae.image"},
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


def write_image_edit_workflow_metadata(
    pkg: Any,
    output_dir: str,
    *,
    num_inference_steps: int,
    schedule: list[float],
    timesteps: list[float],
    guidance_scale: float,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_image_edit_workflow_metadata(
        pkg,
        num_inference_steps=num_inference_steps,
        schedule=schedule,
        timesteps=timesteps,
        guidance_scale=guidance_scale,
    )
    pkg.save_policy_components(output_dir)
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


def build_video_diffusion_workflow_metadata(
    pkg: Any,
    *,
    num_inference_steps: int,
    schedule: list[float] | None = None,
    timesteps: list[float] | None = None,
    init_noise_sigma: float = 1.0,
    scaling_factor: float = 1.0,
    latent_permutation: list[int] | None = None,
    solver: str = "euler",
    clip_sample_range: float | None = None,
) -> dict[str, Any]:
    """Build a text-to-video diffusion workflow over rank-5 temporal latents.

    The shape of the workflow differs from the image path in ways that are
    intrinsic to video rather than cosmetic:

    - the latent and the published frames carry a temporal axis, so every
      contract is rank 5 and no stage may assume a single frame;
    - the scheduler's timestep history is carried state, not telemetry;
    - the decoder is causal and is invoked once per latent-frame chunk, with the
      convolution caches as runtime-owned state released at the end of the
      invocation;
    - frames are published incrementally, appending on the temporal axis as each
      chunk is decoded, so a consumer sees frames before the clip is finished.
    """
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    names = set(pkg.keys())
    denoiser_name = next(
        (name for name in ("denoiser", "transformer", "unet") if name in names), None
    )
    vae_name = next(
        (name for name in ("vae_decoder", "decoder", "vae") if name in names), None
    )
    if denoiser_name is None or vae_name is None or denoiser_name == vae_name:
        raise ValueError("video workflow requires distinct denoiser and VAE decoder")
    denoiser = pkg[denoiser_name]
    vae = pkg[vae_name]
    sample_input = _find_port(denoiser.graph.inputs, "sample", "latent", "hidden_states")
    timestep_input = _find_port(denoiser.graph.inputs, "timestep", "time")
    estimate_output = next(iter(denoiser.graph.outputs), None)
    vae_input = _find_port(vae.graph.inputs, "latent_sample", "latent", "sample")
    vae_output = next(
        (value for value in vae.graph.outputs if not value.name.startswith("conv_cache")),
        None,
    )
    if None in (sample_input, timestep_input, estimate_output, vae_input, vae_output):
        raise ValueError("video components do not expose sample/timestep/estimate/VAE ports")
    assert sample_input is not None
    assert timestep_input is not None
    assert estimate_output is not None
    assert vae_input is not None
    assert vae_output is not None
    if len(sample_input.shape or []) != 5:
        raise ValueError(
            "video diffusion workflow requires a rank-5 [batch, frames, channels, "
            "height, width] latent; use build_diffusion_workflow_metadata for images"
        )
    if _contract(sample_input) != _contract(estimate_output):
        raise ValueError("video workflow requires matching latent/estimate contracts")
    if len(vae_input.shape or []) != 5 or len(vae_output.shape or []) != 5:
        raise ValueError("video VAE decode must be rank 5 on both the latent and the frames")

    cache_ports = [
        value.name for value in vae.graph.inputs if value.name.startswith("conv_cache.")
    ]
    cache_outputs = {
        value.name[len("conv_cache_out.") :]: value.name
        for value in vae.graph.outputs
        if value.name.startswith("conv_cache_out.")
    }
    if not cache_ports or set(cache_ports) != {f"conv_cache.{name}" for name in cache_outputs}:
        raise ValueError("causal video decode requires paired conv_cache/conv_cache_out ports")
    cache_entries = [
        (
            port,
            int(list(next(v for v in vae.graph.inputs if v.name == port).shape)[1]),
            _cache_spatial_scale(vae, next(v for v in vae.graph.inputs if v.name == port)),
        )
        for port in cache_ports
    ]

    text_name = next(
        (name for name in ("text_encoder", "text_encoder_2") if name in names), None
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

    if solver not in ("euler", "ddim"):
        raise ValueError("video solver must be 'euler' or 'ddim'")
    latent_dims = ["batch", "frames", "channels", "height", "width"]
    attach_policy_components(pkg, PolicyCapabilities())
    if solver == "ddim":
        # DDIM defines scale_model_input as the identity and consumes cumulative
        # alphas rather than sigmas.
        pkg.add_policy_component(
            "model_input", build_identity_model_input(sample_input.dtype, latent_dims)
        )
        pkg.add_policy_component(
            "solver_step",
            build_ddim_solver_step(
                sample_input.dtype, latent_dims, clip_sample_range=clip_sample_range
            ),
        )
    else:
        pkg.add_policy_component(
            "model_input", build_euler_model_input(sample_input.dtype, latent_dims)
        )
        pkg.add_policy_component(
            "solver_step", build_euler_solver_step(sample_input.dtype, latent_dims)
        )
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    pkg.add_policy_component(
        "video_latent_init",
        build_video_latent_initializer(sample_input.dtype, init_noise_sigma),
    )
    pkg.add_policy_component(
        "schedule_history_append", build_schedule_history_append(timestep_input.dtype)
    )
    pkg.add_policy_component(
        "video_latent_permute",
        build_video_latent_permute(latent_permutation or [0, 2, 1, 3, 4]),
    )
    pkg.add_policy_component(
        "video_latent_unscale", build_video_latent_unscale(scaling_factor)
    )
    pkg.add_policy_component("video_decode_chunks", build_video_decode_chunk_count())
    pkg.add_policy_component("video_decode_chunk", build_video_decode_chunk())
    pkg.add_policy_component(
        "video_conv_cache_init", build_video_conv_cache_initializer(cache_entries)
    )

    schedule_values = schedule or [
        1.0 - index / num_inference_steps for index in range(num_inference_steps + 1)
    ]
    timestep_values = timesteps or schedule_values[:-1]
    if len(schedule_values) != num_inference_steps + 1:
        raise ValueError("video solver schedule must contain num_inference_steps + 1 values")
    if len(timestep_values) != num_inference_steps:
        raise ValueError("video timesteps must contain num_inference_steps values")
    pkg.add_policy_component("diffusion_schedule", build_schedule_constant(schedule_values))
    pkg.add_policy_component("diffusion_timesteps", build_schedule_constant(timestep_values))
    pkg.add_policy_component("schedule_lookup", build_schedule_lookup(timestep_input.dtype))

    latent_contract = _request_aligned(_contract(sample_input))
    batch = latent_contract["shape"][0]
    batch_int = {"dtype": "int64", "rank": 1, "shape": [batch]}
    batch_bool = {"dtype": "bool", "rank": 1, "shape": [batch]}
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}
    history_contract = _request_aligned(
        {
            "dtype": _contract(timestep_input)["dtype"],
            "rank": 2,
            "shape": [batch, "scheduler_history"],
        }
    )

    inputs: dict[str, Any] = {
        "request.noise": {
            "contract": latent_contract,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "noise"},
            "required": True,
        },
        "request.max_iterations": {
            "contract": control_int,
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
        "package.one_control": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.history_limit": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": num_inference_steps,
        },
        "package.cache_frames": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 2,
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
                "contract": _request_aligned(_contract(value)),
                "role": (
                    {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"}
                    if index == 0
                    else {"kind": "opaque"}
                ),
                "source": (
                    {"kind": "request", "field": "prompt_tokens"}
                    if index == 0
                    else {"kind": "application", "name": value.name}
                ),
                "required": True,
            }
            text_inputs[value.name] = name
        conditioning_value = "conditioning.hidden_states"
        setup_nodes.append(
            _invoke(text_name, text_inputs, {conditioning_output.name: conditioning_value})
        )
    if conditioning_input is not None and conditioning_value is None:
        # No text encoder ships with the package, so the prompt embedding is
        # supplied by the application. Conditioning stays a declared input
        # rather than an implicit constant: an unconditioned video model would
        # simply have no such port on its denoiser.
        conditioning_value = f"request.{conditioning_input.name}"
        inputs[conditioning_value] = {
            "contract": _request_aligned(_contract(conditioning_input)),
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": conditioning_input.name},
            "required": True,
        }
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

    body_nodes: list[dict[str, Any]] = [
        _invoke(
            "schedule_lookup",
            {"schedule": "diffusion.timesteps", "step": "loop.iteration"},
            {"timestep": "diffusion.timestep"},
        ),
        _invoke(
            "model_input",
            {
                "sample": "state.latent.body",
                "step": "loop.iteration",
                "schedule": "diffusion.schedule",
            },
            {"model_input": "diffusion.model_input"},
        ),
        _invoke(denoiser_name, denoiser_inputs, {estimate_output.name: "denoiser.estimate"}),
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
            "schedule_history_append",
            {"history": "state.scheduler_history.body", "timestep": "diffusion.timestep"},
            {"next": "scheduler_history.body"},
        ),
        _invoke(
            "continue_predicate",
            {"done": "package.false"},
            {"continue": "loop.continue"},
        ),
    ]

    decode_body: list[dict[str, Any]] = [
        _invoke(
            "video_decode_chunk",
            {"latent": "decode.latent", "step": "decode.iteration"},
            {"chunk": "decode.chunk"},
        ),
        _invoke(
            vae_name,
            {
                vae_input.name: "decode.chunk",
                **{port: f"state.{_cache_cell(port)}.body" for port in cache_ports},
            },
            {
                vae_output.name: "decode.frames",
                **{
                    cache_outputs[name]: f"{_cache_cell(f'conv_cache.{name}')}.body"
                    for name in cache_outputs
                },
            },
        ),
        {
            "kind": "emit",
            "value": "decode.frames",
            "output": "video",
            "mode": "append",
            "axis": 2,
            "effect_name": "emit",
            "effect": _effect("emit.0", "emit.1"),
        },
        _invoke(
            "continue_predicate",
            {"done": "package.false"},
            {"continue": "decode.loop.continue"},
        ),
    ]

    frames_contract = _request_aligned(_contract(vae_output))
    workflow = {
        "manifest": {
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
                "bounded_state_recurrence",
            ],
        },
        "inputs": inputs,
        "outputs": {
            "video": {
                "contract": frames_contract,
                "role": "video",
                "stage": "pre_adapter",
            }
        },
        "components": {
            name: _component(model, _artifact(name, len(pkg))) for name, model in pkg.items()
        },
        "state": {
            "latent": {
                "contract": latent_contract,
                "scope": "invocation",
                "initializer": "latent.initial",
                "recurrence": {"kind": "invariant"},
            },
            "scheduler_history": {
                "contract": history_contract,
                "scope": "invocation",
                "initializer": "scheduler.history.initial",
                "recurrence": {
                    "kind": "growing",
                    "axis": 1,
                    "increment": "package.one_control",
                    "max": "package.history_limit",
                },
            },
            **{
                _cache_cell(port): {
                    "contract": _request_aligned(
                        _contract(next(v for v in vae.graph.inputs if v.name == port))
                    ),
                    "scope": "invocation",
                    "initializer": f"{_cache_cell(port)}.initial",
                    "recurrence": {
                        "kind": "bounded",
                        "axis": 2,
                        "max": "package.cache_frames",
                    },
                    "management": "runtime",
                    "release_boundary": "invocation",
                }
                for port in cache_ports
            },
        },
        "initial_effects": {
            "solver": "solver.0",
            "state:latent": "state:latent.0",
            "state:scheduler_history": "state:scheduler_history.0",
            **{
                f"state:{_cache_cell(port)}": f"state:{_cache_cell(port)}.0"
                for port in cache_ports
            },
            "emit": "emit.0",
        },
        "graph": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "video_latent_init",
                    {"noise": "request.noise"},
                    {"latent": "latent.initial", "history": "scheduler.history.initial"},
                ),
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
                            "current": "latent.initial",
                            "body_input": "state.latent.body",
                            "body_output": "latent.body",
                            "next": "latent.final",
                            "read_effect": _effect("state:latent.0", "state:latent.read"),
                            "write_effect": _effect("state:latent.read", "state:latent.1"),
                        },
                        {
                            "cell": "scheduler_history",
                            "current": "scheduler.history.initial",
                            "body_input": "state.scheduler_history.body",
                            "body_output": "scheduler_history.body",
                            "next": "scheduler_history.final",
                            "read_effect": _effect(
                                "state:scheduler_history.0", "state:scheduler_history.read"
                            ),
                            "write_effect": _effect(
                                "state:scheduler_history.read", "state:scheduler_history.1"
                            ),
                        },
                    ],
                },
                _invoke(
                    "video_latent_permute",
                    {"latent": "latent.final"},
                    {"permuted": "decode.latent_permuted"},
                ),
                _invoke(
                    "video_latent_unscale",
                    {"latent": "decode.latent_permuted"},
                    {"unscaled": "decode.latent"},
                ),
                _invoke(
                    "video_decode_chunks",
                    {"latent": "decode.latent"},
                    {"count": "decode.chunks"},
                ),
                _invoke(
                    "video_conv_cache_init",
                    {"latent": "decode.latent"},
                    {port: f"{_cache_cell(port)}.initial" for port in cache_ports},
                ),
                {
                    "kind": "loop",
                    "setup": {
                        "kind": "sequence",
                        "nodes": [
                            _invoke(
                                "continue_predicate",
                                {"done": "package.false"},
                                {"continue": "decode.setup.continue"},
                            )
                        ],
                    },
                    "body": {"kind": "sequence", "nodes": decode_body},
                    "condition": "decode.loop.continue",
                    "max_iterations": "decode.chunks",
                    "iteration": {"value": "decode.iteration", "contract": batch_int},
                    "carried": [
                        {
                            "cell": _cache_cell(port),
                            "current": f"{_cache_cell(port)}.initial",
                            "body_input": f"state.{_cache_cell(port)}.body",
                            "body_output": f"{_cache_cell(port)}.body",
                            "next": f"{_cache_cell(port)}.final",
                            "read_effect": _effect(
                                f"state:{_cache_cell(port)}.0",
                                f"state:{_cache_cell(port)}.read",
                            ),
                            "write_effect": _effect(
                                f"state:{_cache_cell(port)}.read",
                                f"state:{_cache_cell(port)}.1",
                            ),
                        }
                        for port in cache_ports
                    ],
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


def write_video_diffusion_workflow_metadata(
    pkg: Any,
    output_dir: str,
    **kwargs: Any,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_video_diffusion_workflow_metadata(pkg, **kwargs)
    pkg.save_policy_components(output_dir)
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
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
                for name in _cache_output_candidates(value.name or "")
                if name in decoder_outputs
            ),
            None,
        )
        if present is not None:
            if present.shape is None:
                present.shape = value.shape
            cache_pairs.append((value, present))
    cache_names = {value.name for value, _ in cache_pairs}
    # A multimodal decoder can be hybrid: sliding layers keep a growing cache
    # while full-attention layers scatter into fixed buffers. Both disciplines
    # are described side by side rather than one being folded into the other.
    static_cache = _static_cache_ports(decoder)
    decoder_kv = _kv_storage_contract(decoder)
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
    fixed_capacity = bool(cache_pairs) and decoder_kv["storage"] == "shared_buffer"
    # A static cache carries its own per-row length on a graph port, so the
    # prompt length must be materialized for it whether or not the shared-buffer
    # attention-mask discipline also applies.
    tracks_cache_lengths = fixed_capacity or static_cache is not None

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
    vision_feature_outputs = [
        value
        for value in vision.graph.outputs
        if value.name in embedding_inputs_by_name
        and value.shape is not None
        and len(value.shape) == 2
        and isinstance(list(value.shape)[-1], int)
    ]
    text_only_vision = vision_feature_outputs[0] if len(vision_feature_outputs) == 1 else None

    attach_policy_components(
        pkg,
        PolicyCapabilities(
            sampler="greedy",
            eos_termination=True,
            token_state_update=True,
        ),
    )
    pkg.add_policy_component("last_token_logits", build_last_token_logits(logits_output.dtype))
    if text_only_vision is not None:
        pkg.add_policy_component(
            "empty_image_features",
            build_empty_features(
                text_only_vision.dtype,
                int(list(text_only_vision.shape)[-1]),
            ),
        )
    pkg.add_policy_component(
        "decoder_state_initializer",
        build_decoder_state_initializer(
            decoder,
            token_input=None,
            prompt_dtype=token_input.dtype,
            attention_mask_input=attention_input.name,
            position_ids_input=position_input.name if position_input is not None else None,
            cache_inputs=sorted(cache_names),
            fixed_capacity=fixed_capacity,
            ragged=True,
            write_indices_output=(
                static_cache["write_indices"] if static_cache is not None else None
            ),
        ),
    )
    pkg.add_policy_component(
        "decoder_step_update",
        build_decoder_step_update(
            attention_dtype=attention_input.dtype,
            position_dtype=position_input.dtype if position_input is not None else None,
            fixed_capacity=fixed_capacity,
            position_sections=(
                rotary_axis_count(position_input) if position_input is not None else None
            ),
        ),
    )
    pkg.add_policy_component("token_sampler", build_seeded_categorical_sampler())
    pkg.add_policy_component("termination", build_eos_termination(row_selective=True))
    pkg.add_policy_component(
        "termination_batch_initializer",
        build_termination_batch_initializer(),
    )
    pkg.add_policy_component(
        "token_state_update",
        build_token_state_update(row_selective=True),
    )
    pkg.add_policy_component("token_to_slot", build_token_to_slot())
    pkg.add_policy_component("generated_length_update", build_selective_integer_add())
    if cache_pairs:
        pkg.add_policy_component("cache_length_update", build_selective_integer_add())

    batch = _contract(token_input)["shape"][0]
    batch_int = _request_aligned({"dtype": "int64", "rank": 1, "shape": [batch]})
    batch_bool = _request_aligned({"dtype": "bool", "rank": 1, "shape": [batch]})
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}
    eos = _source_token_id(source, "eos_token_id", getattr(config, "eos_token_id", 0))
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
            "required": text_only_vision is None,
            **(
                {"present_as": "request.image_present"} if text_only_vision is not None else {}
            ),
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
        "request.prompt_lengths": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "prompt_lengths"},
            "required": False,
            "default": -1,
        },
        "request.eos_ids": {
            "contract": {"dtype": "int64", "rank": 2, "shape": [batch, "num_eos"]},
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "eos_ids"},
            "required": False,
            "default": eos,
        },
        "request.eos_lengths": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "eos_lengths"},
            "required": False,
            "default": 1,
        },
        "request.row_max_iterations": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "row_max_iterations"},
            "required": False,
            "default": -1,
        },
        "package.eos_ids": {
            "contract": {"dtype": "int64", "rank": 1, "shape": [1]},
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": eos,
        },
        "package.max_context": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": int(
                _source_model_value(
                    source,
                    "context_length",
                    getattr(config, "max_position_embeddings", 4096),
                )
            ),
        },
        **(
            {
                # The capacity a static graph was built against is a graph fact,
                # not a deployment budget: it bounds legal write destinations.
                "package.cache_capacity": {
                    "contract": control_int,
                    "role": {"kind": "opaque"},
                    "source": {"kind": "literal"},
                    "required": False,
                    "default": int(static_cache["capacity"]),
                }
            }
            if static_cache is not None
            else {}
        ),
        "package.one": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.one_step": {
            # A growing-recurrence increment advances the whole invocation's
            # state axis by one, so it is an invocation-scoped control scalar,
            # not a per-row value like ``package.one``/``package.one_token``.
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.active": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": True,
        },
        "package.false": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": False,
        },
        "package.zero_batch": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0,
        },
    }
    inputs.update(
        {
            "request.temperature": {
                "contract": {"dtype": "float32", "rank": 1, "shape": [batch]},
                "role": {
                    "kind": "runtime",
                    "version": "1.0",
                    "role": "sampling_temperature",
                },
                "source": {"kind": "request", "field": "sampling_temperature"},
                "required": False,
                "default": 1.0,
            },
            "request.top_k": {
                "contract": batch_int,
                "role": {
                    "kind": "runtime",
                    "version": "1.0",
                    "role": "sampling_top_k",
                },
                "source": {"kind": "request", "field": "sampling_top_k"},
                "required": False,
                "default": 1,
            },
            "request.top_p": {
                "contract": {"dtype": "float32", "rank": 1, "shape": [batch]},
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
                "contract": {"dtype": "float32", "rank": 1, "shape": [batch]},
                "role": {
                    "kind": "runtime",
                    "version": "1.0",
                    "role": "sampling_min_p",
                },
                "source": {"kind": "request", "field": "sampling_min_p"},
                "required": False,
                "default": 0.0,
            },
            "request.seed": {
                "contract": batch_int,
                "role": {"kind": "runtime", "version": "1.0", "role": "seed"},
                "source": {"kind": "request", "field": "seed"},
                "required": False,
                "default": 0,
            },
            "request.rng_counter": {
                "contract": batch_int,
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": "rng_counter"},
                "required": False,
                "default": 0,
            },
        }
    )
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
        attention_input.name: (
            "decoder_step.body_attention_mask"
            if fixed_capacity
            else "state.attention_mask.body"
        ),
    }
    if position_input is not None:
        setup_decoder_inputs[position_input.name] = f"initializer.{position_input.name}"
        body_decoder_inputs[position_input.name] = "state.position_ids.body"
    if static_cache is not None:
        # Prefill scatters from slot zero; each decode step writes at the row's
        # current logical length, which is the same cursor the group declares.
        setup_decoder_inputs[static_cache["write_indices"]] = (
            f"initializer.{static_cache['write_indices']}"
        )
        setup_decoder_inputs[static_cache["kv_sequence_length"]] = "initializer.cache_lengths"
        body_decoder_inputs[static_cache["write_indices"]] = "state.cache_lengths.body"
        body_decoder_inputs[static_cache["kv_sequence_length"]] = "cache_lengths.next"
    for past, _ in cache_pairs:
        setup_decoder_inputs[past.name] = f"initializer.{past.name}"
        body_decoder_inputs[past.name] = f"state.{past.name}.body"

    setup_decoder_outputs = {logits_output.name: "decoder.setup.logits"}
    body_decoder_outputs = {logits_output.name: "decoder.body.logits"}
    logits_contract = _contract(logits_output)
    last_logits_contract = {
        "dtype": "float32",
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
        "generated_lengths": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "initializer.generated_lengths",
            "recurrence": {"kind": "invariant"},
        },
        "attention_mask": {
            "contract": {
                "dtype": _contract(attention_input)["dtype"],
                "rank": 2,
                "shape": [batch, "context"],
            },
            "scope": "invocation",
            "initializer": (
                f"initializer.{attention_input.name}"
                if fixed_capacity
                else "initializer.body_attention_mask"
            ),
            "recurrence": (
                {"kind": "invariant"}
                if fixed_capacity
                else {
                    "kind": "growing",
                    "axis": 1,
                    "increment": "package.one_step",
                    "max": "package.max_context",
                }
            ),
        },
        "active": {
            "contract": batch_bool,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.active",
            "recurrence": {"kind": "invariant"},
        },
        "done": {
            "contract": batch_bool,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.false",
            "recurrence": {"kind": "invariant"},
        },
        "accepted_len": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.zero_batch",
            "recurrence": {"kind": "invariant"},
        },
        "cache_lengths": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": (
                "initializer.cache_lengths" if tracks_cache_lengths else "package.zero_batch"
            ),
            "recurrence": {"kind": "invariant"},
        },
        "rng_counter": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "request.rng_counter",
            "recurrence": {"kind": "invariant"},
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
            (
                f"initializer.{attention_input.name}"
                if fixed_capacity
                else "initializer.body_attention_mask"
            ),
            "state.attention_mask.body",
            "decoder_step.body_attention_mask",
            "state.attention_mask.final",
        ),
        (
            "generated_lengths",
            "initializer.generated_lengths",
            "state.generated_lengths.body",
            "token.next_lengths",
            "state.generated_lengths.final",
        ),
        (
            "rng_counter",
            "request.rng_counter",
            "state.rng_counter.body",
            "sample.next_counter",
            "state.rng_counter.final",
        ),
        (
            "active",
            "package.active",
            "state.active.body",
            "loop.next_active",
            "state.active.final",
        ),
        ("done", "package.false", "state.done.body", "loop.done", "state.done.final"),
        (
            "accepted_len",
            "package.zero_batch",
            "state.accepted_len.body",
            "accepted_len.next",
            "state.accepted_len.final",
        ),
        (
            "cache_lengths",
            "initializer.cache_lengths" if tracks_cache_lengths else "package.zero_batch",
            "state.cache_lengths.body",
            "cache_lengths.next",
            "state.cache_lengths.final",
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
        # A scattered buffer overwrites cells inside a capacity fixed at export,
        # so its extent never changes; only an appending cache grows.
        scattered = static_cache is not None and past.name in static_cache["buffers"]
        state[cell] = {
            "contract": _request_aligned(_contract(past)),
            "scope": "invocation",
            "initializer": f"decoder.setup.{present.name}",
            "recurrence": (
                {"kind": "invariant"}
                if scattered
                else {
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
                }
            ),
            "management": "runtime",
            "release_boundary": "invocation",
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
    # Hybrid VL decoders (Gemma 3/4) publish one group per attention kind so a
    # global layer's prefix is never evicted with the sliding layers'.
    vlm_state_groups, vlm_cell_groups = _state_service_groups(
        config=getattr(config, "text", config),
        cache_pairs=cache_pairs,
        ports={
            "decoder": {
                f"cache_{index}": {"input": past.name, "output": present.name}
                for index, (past, present) in enumerate(cache_pairs)
            }
        },
        sequence_axis=next(
            (
                axis
                for axis, dimension in enumerate(_contract(cache_pairs[0][0])["shape"])
                if "sequence" in str(dimension)
            ),
            2,
        )
        if cache_pairs
        else 2,
        logical_lengths="cache_lengths",
        aliasing=_state_aliasing(decoder_kv),
        base_name="decoder_cache",
        indexed_scatter=(
            {
                "buffers": static_cache["buffers"],
                "capacity": "package.cache_capacity",
                # The write cursor and the logical length are one quantity: a
                # row's next write lands exactly where its valid prefix ends.
                "write_indices": "cache_lengths",
                "logical_lengths": "cache_lengths",
                "port": static_cache["write_indices"],
                "kv_length_port": static_cache["kv_sequence_length"],
            }
            if static_cache is not None
            else None
        ),
    )
    for cell, group_name in vlm_cell_groups.items():
        state[cell]["service_group"] = group_name
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

    if text_only_vision is not None:
        feature_name = text_only_vision.name
        vision_setup_nodes: list[dict[str, Any]] = [
            {
                "kind": "branch",
                "predicate": "request.image_present",
                "cases": {
                    "true": {
                        "kind": "sequence",
                        "nodes": [
                            _invoke(
                                "image_preprocess",
                                {"encoded": "request.image"},
                                dict(preprocessing_values),
                            ),
                            _invoke(
                                "vision_encoder",
                                vision_invoke_inputs,
                                {
                                    **vision_outputs,
                                    feature_name: f"vision.with_media.{feature_name}",
                                },
                            ),
                        ],
                    },
                    "false": _invoke(
                        "empty_image_features",
                        {},
                        {"features": f"vision.empty.{feature_name}"},
                    ),
                },
                "outputs": {
                    f"vision.{feature_name}": {
                        "cases": {
                            "true": f"vision.with_media.{feature_name}",
                            "false": f"vision.empty.{feature_name}",
                        }
                    }
                },
            }
        ]
    else:
        vision_setup_nodes = [
            _invoke(
                "image_preprocess",
                {"encoded": "request.image"},
                dict(preprocessing_values),
            ),
            _invoke("vision_encoder", vision_invoke_inputs, vision_outputs),
        ]

    setup = {
        "kind": "sequence",
        "nodes": [
            *vision_setup_nodes,
            *audio_setup_nodes,
            _invoke(
                "decoder_state_initializer",
                {
                    "prompt_tokens": "request.prompt_tokens",
                    "prompt_lengths": "request.prompt_lengths",
                    **({"max_iterations": "request.max_iterations"} if fixed_capacity else {}),
                },
                {
                    attention_input.name: f"initializer.{attention_input.name}",
                    "body_attention_mask": "initializer.body_attention_mask",
                    "token_slot": "initializer.token_slot",
                    "generated_lengths": "initializer.generated_lengths",
                    **(
                        {"cache_lengths": "initializer.cache_lengths"}
                        if tracks_cache_lengths
                        else {}
                    ),
                    **(
                        {
                            static_cache["write_indices"]: (
                                f"initializer.{static_cache['write_indices']}"
                            )
                        }
                        if static_cache is not None
                        else {}
                    ),
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
                "termination_batch_initializer",
                {
                    "input_eos_ids": "request.eos_ids",
                    "input_eos_lengths": "request.eos_lengths",
                    "input_max_iterations": "request.row_max_iterations",
                    "fallback_max_iterations": "request.max_iterations",
                    "active": "package.active",
                },
                {
                    "row_eos_ids": "termination.eos_ids",
                    "eos_lengths": "termination.eos_lengths",
                    "max_iterations": "termination.max_iterations",
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
    decoder_step_invoke = _invoke(
        "decoder_step_update",
        {
            "attention_mask": "state.attention_mask.body",
            **({"logical_length": "state.cache_lengths.body"} if fixed_capacity else {}),
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
    )
    body = {
        "kind": "sequence",
        "nodes": [
            _invoke(
                "token_sampler",
                {
                    "logits": "state.logits.body",
                    "temperature": "request.temperature",
                    "top_k": "request.top_k",
                    "top_p": "request.top_p",
                    "min_p": "request.min_p",
                    "seed": "request.seed",
                    "counter": "state.rng_counter.body",
                    "active": "state.active.body",
                    "done": "state.done.body",
                },
                {"token": "sample.body", "next_counter": "sample.next_counter"},
                {"sample": _effect("sample.0", "sample.1")},
            ),
            _invoke(
                "termination",
                {
                    "tokens": "sample.body",
                    "eos_ids": "termination.eos_ids",
                    "eos_lengths": "termination.eos_lengths",
                    "iteration": "loop.iteration",
                    "max_iterations": "termination.max_iterations",
                    "active": "state.active.body",
                },
                {
                    "done": "loop.done",
                    "next_active": "loop.next_active",
                    "continue": "loop.continue",
                },
                {"termination": _effect("termination.0", "termination.1")},
            ),
            *(
                [
                    _invoke(
                        "cache_length_update",
                        {
                            "left": "state.cache_lengths.body",
                            "right": "package.one",
                            "active": "state.active.body",
                            "done": "state.done.body",
                        },
                        {"total": "cache_lengths.next"},
                    ),
                    _invoke(
                        "cache_length_update",
                        {
                            "left": "package.zero_batch",
                            "right": "package.one",
                            "active": "state.active.body",
                            "done": "state.done.body",
                        },
                        {"total": "accepted_len.next"},
                    ),
                ]
                if cache_pairs
                else []
            ),
            _invoke("token_to_slot", {"token": "sample.body"}, {"slot": "sample.slot"}),
            _invoke(
                "generated_length_update",
                {
                    "left": "state.generated_lengths.body",
                    "right": "package.one",
                    "active": "state.active.body",
                    "done": "state.done.body",
                },
                {"total": "token.next_lengths"},
            ),
            _invoke(
                "generated_length_update",
                {
                    "left": "package.zero_batch",
                    "right": "package.one",
                    "active": "state.active.body",
                    "done": "state.done.body",
                },
                {"total": "token.emitted_length"},
            ),
            _invoke(
                "token_state_update",
                {
                    "current": "state.token.body",
                    "update": "sample.slot",
                    "active": "state.active.body",
                    "done": "state.done.body",
                },
                {"next": "token.body"},
                {"state": _effect("state.0", "state.1")},
            ),
            {
                "kind": "emit",
                "value": "token.body",
                "output": "tokens",
                "mode": "append",
                "when": "state.active.body",
                "valid_length": "token.emitted_length",
                "effect_name": "emit",
                "effect": _effect("emit.0", "emit.1"),
            },
            *([decoder_step_invoke] if fixed_capacity else []),
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
            *([] if fixed_capacity else [decoder_step_invoke]),
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
            "adapter_abis": {"onnx-genai.image-preprocess": "1"},
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
                "emit_valid_length",
                *(["input_presence"] if text_only_vision is not None else []),
                *(
                    ["serving_service_contract", "bounded_state_recurrence"]
                    if cache_pairs
                    else []
                ),
            ],
        },
        "inputs": inputs,
        "outputs": {
            "tokens": {
                "contract": _request_aligned(
                    {
                        "dtype": "int64",
                        "rank": 2,
                        "shape": [batch, "generated_sequence"],
                    }
                ),
                "role": "tokens",
                "stage": "pre_adapter",
            }
        },
        "components": components,
        "state": state,
        **(
            {
                "serving": {
                    "active": "active",
                    "done": "done",
                    "accepted_len": "accepted_len",
                    "state_service": {"groups": vlm_state_groups},
                }
            }
            if cache_pairs
            else {}
        ),
        "initial_effects": initial_effects,
        "graph": {
            "kind": "loop",
            "setup": setup,
            "body": body,
            "condition": "loop.continue",
            "termination": "generation_eos",
            "max_iterations": "request.max_iterations",
            "iteration": {
                "value": "loop.iteration",
                "contract": {"dtype": "int64", "rank": 1, "shape": [1]},
            },
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
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
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
    verifier_kv = _kv_storage_contract(verifier)
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
    if grammar_guidance:
        pkg.add_policy_component("grammar_length", build_integer_minimum())
        pkg.add_policy_component("grammar_sampler_logits", build_last_token_logits())
        if adaptive_k_max is None:
            pkg.add_policy_component("proposal_length", build_sequence_length())
    if adaptive_k_max is not None:
        pkg.add_policy_component("proposal_metrics", build_proposal_metrics())
    pkg.add_policy_component("cache_length_update", build_integer_add())
    batch = _contract(proposer_input)["shape"][0]
    batch_int = _request_aligned({"dtype": "int64", "rank": 1, "shape": [batch]})
    batch_bool = _request_aligned({"dtype": "bool", "rank": 1, "shape": [batch]})
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}
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
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": False,
        },
        "package.active": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": True,
        },
        "request.cache_lengths": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "serving.cache_lengths"},
            "required": False,
            "default": 0,
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
    emit_length = "acceptance.length"
    grammar_post_nodes: list[dict[str, Any]] = []
    if grammar_guidance:
        emit_length = "grammar.committed_length"
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
    cache_next_outputs = {
        f"cache_{index}.next": f"verifier.{present.name}"
        for index, (_past, present) in enumerate(cache_pairs)
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
                "continue": "acceptance.continue",
                "next_offset": "rng_offset.body",
                "rollback_len": "acceptance.rollback_length",
            },
            {"verify": _effect("verify.0", "verify.1")},
        ),
        *grammar_post_nodes,
        _invoke(
            "cache_length_update",
            {
                "left": "state.cache_lengths.body",
                "right": emit_length,
            },
            {"total": "cache_lengths.next"},
        ),
        *adaptive_nodes,
        {
            "kind": "emit",
            "value": "acceptance.tokens",
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
                "valid_length": "grammar.forced_length",
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
        "active": {
            "contract": batch_bool,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.active",
            "recurrence": {"kind": "invariant"},
        },
        "done": {
            "contract": batch_bool,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.false",
            "recurrence": {"kind": "invariant"},
        },
        "accepted_len": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "package.zero",
            "recurrence": {"kind": "invariant"},
        },
        "cache_lengths": {
            "contract": batch_int,
            "class": "semantic",
            "scope": "invocation",
            "initializer": "request.cache_lengths",
            "recurrence": {"kind": "invariant"},
        },
    }
    state_specs = [
        (
            "tokens",
            "request.tokens",
            "state.tokens.body",
            "acceptance.tokens",
            "state.tokens.final",
        ),
        (
            "rng_offset",
            "package.zero",
            "state.rng_offset.body",
            "rng_offset.body",
            "state.rng_offset.final",
        ),
        (
            "active",
            "package.active",
            "state.active.body",
            "acceptance.continue",
            "state.active.final",
        ),
        (
            "done",
            "package.false",
            "state.done.body",
            "acceptance.done",
            "state.done.final",
        ),
        (
            "accepted_len",
            "package.zero",
            "state.accepted_len.body",
            emit_length,
            "state.accepted_len.final",
        ),
        (
            "cache_lengths",
            "request.cache_lengths",
            "state.cache_lengths.body",
            "cache_lengths.next",
            "state.cache_lengths.final",
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
    kv_ports: dict[str, Any] = {}
    kv_sequence_axis = 2
    for index, (past, present) in enumerate(cache_pairs):
        cell = f"cache_{index}"
        initializer = f"request.verifier.{past.name}"
        kv_sequence_axis = next(
            (
                axis
                for axis, dimension in enumerate(_contract(past)["shape"])
                if "sequence" in str(dimension)
            ),
            2,
        )
        state[cell] = {
            "contract": _request_aligned(_contract(past)),
            "class": "semantic",
            "scope": "invocation",
            "initializer": initializer,
            "recurrence": {
                "kind": "bounded",
                "axis": kv_sequence_axis,
                "max": "package.max_context",
            },
            "service_group": "verifier_cache",
            "management": "runtime",
            "release_boundary": "invocation",
        }
        kv_ports[cell] = {"input": past.name, "output": present.name}
        state_specs.append(
            (
                cell,
                initializer,
                f"state.{cell}.body",
                cache_next_outputs[f"{cell}.next"],
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
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
                "emit_valid_length",
                "bounded_state_recurrence",
                "serving_service_contract",
            ],
        },
        "inputs": inputs,
        "outputs": {
            "tokens": {
                "contract": _request_aligned(
                    {
                        **_contract(proposed_tokens),
                        "shape": [
                            *_contract(proposed_tokens)["shape"][:-1],
                            "accepted_sequence",
                        ],
                    }
                ),
                "role": "tokens",
                "stage": "pre_adapter",
            },
        },
        "components": {
            name: _component(model, _artifact(name, len(pkg))) for name, model in pkg.items()
        },
        "state": state,
        "serving": {
            "active": "active",
            "done": "done",
            "accepted_len": "accepted_len",
            "state_service": {
                "groups": {
                    "verifier_cache": _state_group(
                        sequence_axis=kv_sequence_axis,
                        logical_lengths="cache_lengths",
                        storage=verifier_kv["storage"],
                        ports={"verifier": kv_ports},
                    )
                },
            },
        },
        "initial_effects": initial_effects,
        "graph": {
            "kind": "loop",
            "setup": {"kind": "sequence", "nodes": []},
            "body": {"kind": "sequence", "nodes": body_nodes},
            "condition": "acceptance.continue",
            "active_cell": "active",
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
        # The grammar adapter's three actions are a speculation protocol:
        # `clone` snapshots the FSM before a proposal, `lookahead` advances the
        # snapshot, and `commit` applies only the accepted prefix. Abandoning a
        # rejected proposal is therefore a transaction abort rather than an
        # unrecoverable side effect, and the explicit `clone` action is exactly
        # what makes the domain safe to enter speculatively.
        workflow["effects"] = {
            "grammar": {
                "retry": "transactional",
                "speculation_safety": {"kind": "clonable"},
            }
        }
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
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


def _cross_state_bindings(
    encoder: ir.Model, decoder: ir.Model
) -> dict[str, tuple[str, ir.Value]]:
    """Map decoder inputs that a conditioning encoder produces once per request.

    The mapping is purely structural. A decoder input is bound to an encoder
    output when the names match exactly (``encoder_hidden_states``) or when the
    decoder's ``past``/``pass``-side spelling rewrites to the encoder's
    ``present``-side spelling (``past_key_cross_0`` <- ``present_key_cross_0``),
    which is the same past/present rewrite already used for self-attention KV.
    No model family, tensor role, or vendor name is consulted.
    """
    encoder_outputs = {value.name: value for value in encoder.graph.outputs}
    bindings: dict[str, tuple[str, ir.Value]] = {}
    for value in decoder.graph.inputs:
        if value.name is None:
            continue
        candidates = (
            value.name,
            value.name.replace("past_key_values", "present"),
            value.name.replace("past.", "present."),
            value.name.replace("past_", "present_"),
        )
        produced = next(
            ((name, encoder_outputs[name]) for name in candidates if name in encoder_outputs),
            None,
        )
        if produced is not None:
            bindings[value.name] = produced
    return bindings


def build_speech_to_text_workflow_metadata(
    pkg: Any,
    config: Any,
    *,
    sampler: str = "greedy",
    audio_preprocessing: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the typed SSA workflow for an encoder-conditioned decoder package.

    The encoder runs once per request inside the loop ``setup``; every value it
    produces that the decoder consumes becomes a shape-invariant workflow state
    cell carried unchanged through the decode loop. That is what keeps encoder
    states and decoder rows aligned under batching: the cross state is
    request-aligned on the same axis as the tokens and is permuted by the same
    compaction that permutes the self-attention cache.
    """
    if set(pkg.keys()) != {"encoder", "decoder"}:
        raise ValueError(
            "speech-to-text workflow requires exactly encoder and decoder components, "
            f"got {sorted(pkg.keys())}"
        )
    return _build_autoregressive_workflow_metadata(
        pkg,
        config,
        sampler=sampler,
        encoder_name="encoder",
        audio_preprocessing=audio_preprocessing,
        artifacts=artifacts,
    )


def build_decoder_workflow_metadata(
    pkg: Any,
    config: Any,
    *,
    sampler: str = "greedy",
) -> dict[str, Any]:
    """Build the exact workflow-policy contract for an autoregressive decoder."""
    if len(pkg) != 1:
        raise ValueError("decoder workflow requires exactly one neural component")
    return _build_autoregressive_workflow_metadata(pkg, config, sampler=sampler)


def _build_autoregressive_workflow_metadata(
    pkg: Any,
    config: Any,
    *,
    sampler: str = "greedy",
    encoder_name: str | None = None,
    audio_preprocessing: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build an autoregressive decode loop, optionally conditioned by an encoder.

    ``artifacts`` overrides the package-relative ONNX path of a component, which
    lets an importer describe an existing on-disk layout without renaming files.
    """
    encoder = pkg[encoder_name] if encoder_name is not None else None
    decoder_items = [(name, model) for name, model in pkg.items() if name != encoder_name]
    if len(decoder_items) != 1:
        raise ValueError("workflow requires exactly one autoregressive component")
    decoder_name, decoder = decoder_items[0]
    cross_bindings = _cross_state_bindings(encoder, decoder) if encoder is not None else {}
    if encoder is not None and not cross_bindings:
        raise ValueError(
            "encoder-conditioned workflow requires at least one decoder input produced "
            "by the encoder"
        )
    inputs = list(decoder.graph.inputs)
    outputs = list(decoder.graph.outputs)
    decoder_kv_contract = _kv_storage_contract(decoder)
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
    attach_policy_components(
        pkg,
        PolicyCapabilities(
            sampler=sampler,
            eos_termination=True,
            token_state_update=True,
        ),
    )
    pkg.add_policy_component("last_token_logits", build_last_token_logits(logits_output.dtype))

    output_by_suffix = {value.name: value for value in outputs}
    cache_pairs: list[tuple[ir.Value, ir.Value]] = []
    for value in inputs:
        if value.name in cross_bindings:
            continue
        present = next(
            (
                output_by_suffix.get(name)
                for name in _cache_output_candidates(value.name or "")
                if name in output_by_suffix
            ),
            None,
        )
        if present is not None:
            cache_pairs.append((value, present))
    cache_names = {past.name for past, _ in cache_pairs}
    # A static cache is a fixed-capacity buffer the graph scatters into at
    # declared destinations, so it needs a write cursor and a per-row valid
    # length that no shape can carry. Its two control ports are integer vectors
    # and shape-indistinguishable, hence read from the exporter's declared ABI.
    static_cache = _static_cache_ports(decoder)
    static_control_names = (
        {static_cache["write_indices"], static_cache["kv_sequence_length"]}
        if static_cache is not None
        else set()
    )
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
            if value.name not in cross_bindings
            and (
                "mask" in value.name
                or "past" in str(getattr(list(value.shape)[1], "value", list(value.shape)[1]))
            )
        ),
        None,
    )
    position_input = next(
        (
            value
            for value in integer_rank2
            if value is not attention_input
            and value.name not in cross_bindings
            and "position" in value.name
        ),
        next(
            (
                value
                for value in integer_rank2
                if value is not attention_input and value.name not in cross_bindings
            ),
            None,
        ),
    )
    derived_names = cache_names | set(cross_bindings) | static_control_names
    if attention_input is not None:
        derived_names.add(attention_input.name)
    if position_input is not None:
        derived_names.add(position_input.name)
    unsupported = [
        value.name
        for value in inputs
        if value is not token_input and value.name not in derived_names
    ]
    if unsupported:
        raise ValueError(f"decoder workflow has unsupported non-request inputs: {unsupported}")
    # A shared full-capacity KV buffer needs an attention mask to convey each
    # row's logical length; a mask-free decoder must grow its cache instead.
    fixed_capacity = (
        bool(cache_pairs)
        and decoder_kv_contract["storage"] == "shared_buffer"
        and attention_input is not None
    )
    # A static cache carries its own per-row length on a graph port, so the
    # prompt length has to be materialized for it whether or not an attention
    # mask is also present.
    tracks_cache_lengths = fixed_capacity or static_cache is not None
    pkg.add_policy_component(
        "decoder_state_initializer",
        build_decoder_state_initializer(
            decoder,
            token_input=token_input.name,
            attention_mask_input=attention_input.name if attention_input is not None else None,
            position_ids_input=position_input.name if position_input is not None else None,
            cache_inputs=sorted(cache_names),
            fixed_capacity=fixed_capacity,
            ragged=bool(cache_pairs),
            write_indices_output=(
                static_cache["write_indices"] if static_cache is not None else None
            ),
        ),
    )
    if attention_input is not None:
        pkg.add_policy_component(
            "decoder_step_update",
            build_decoder_step_update(
                attention_dtype=attention_input.dtype,
                position_dtype=position_input.dtype if position_input is not None else None,
                fixed_capacity=fixed_capacity,
                position_sections=rotary_axis_count(position_input)
                if position_input is not None
                else None,
            ),
        )
    elif position_input is not None:
        pkg.add_policy_component(
            "decoder_step_update",
            build_decoder_step_update(
                attention_dtype=None,
                position_dtype=position_input.dtype,
                fixed_capacity=False,
                position_sections=rotary_axis_count(position_input),
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

    # Encoder inputs are request inputs of the whole workflow: the encoder runs
    # once in the loop setup and its results persist as invariant state.
    encoder_invoke_inputs: dict[str, str] = {}
    encoder_invoke_outputs: dict[str, str] = {}
    audio_adapter_outputs: dict[str, Any] = {}
    audio_values: dict[str, str] = {}
    audio_program: dict[str, Any] | None = None
    if encoder is not None:
        assert encoder_name is not None
        encoder_inputs_by_name = {value.name: value for value in encoder.graph.inputs}
        if audio_preprocessing is not None:
            audio_program = copy.deepcopy(audio_preprocessing)
            for binding in audio_program["outputs"]:
                port_name = binding["name"]
                if port_name not in encoder_inputs_by_name:
                    raise ValueError(
                        f"audio preprocessing output {port_name!r} has no encoder input"
                    )
                contract = _contract(encoder_inputs_by_name[port_name])
                binding["contract"] = contract
                binding["dtype"] = contract["dtype"]
                binding["name"] = f"audio.{port_name}"
                audio_adapter_outputs[port_name] = contract
                audio_values[port_name] = binding["name"]
        for value in encoder.graph.inputs:
            ssa = f"encoder.input.{value.name}"
            if value.name in audio_values:
                # The declared preprocessing program produces this encoder input,
                # so the workflow consumes decoded audio bytes instead of features.
                encoder_invoke_inputs[value.name] = audio_values[value.name]
                continue
            workflow_inputs[ssa] = {
                "contract": _contract(value),
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": value.name},
                "required": True,
                # An application may reuse a previously computed encoder input
                # instead of recomputing the feature extraction.
                "externally_suppliable": True,
            }
            encoder_invoke_inputs[value.name] = ssa
        for _decoder_input, (encoder_output, _) in sorted(cross_bindings.items()):
            encoder_invoke_outputs[encoder_output] = f"encoder.{encoder_output}"
    if audio_program is not None:
        # Raw encoded audio is the request-level input; the adapter decodes and
        # turns it into the encoder feature tensor declared by the program.
        workflow_inputs["request.audio"] = {
            "contract": {"dtype": "uint8", "rank": 1, "shape": ["encoded_bytes"]},
            "role": {"kind": "runtime", "version": "1.0", "role": "media"},
            "source": {"kind": "request", "field": "media"},
            "required": True,
        }
    batch_dimension = _shape_metadata(_port(token_input))[0]
    batch_int = _request_aligned({"dtype": "int64", "rank": 1, "shape": [batch_dimension]})
    batch_bool = _request_aligned({"dtype": "bool", "rank": 1, "shape": [batch_dimension]})
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}
    eos_token_id = getattr(config, "eos_token_id", 0)
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0] if eos_token_id else 0
    eos_token_id = int(eos_token_id or 0)
    workflow_inputs.update(
        {
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
            "package.eos_ids": {
                "contract": {"dtype": "int64", "rank": 1, "shape": ["E"]},
                "role": {"kind": "opaque"},
                "source": {"kind": "literal"},
                "required": True,
                "default": eos_token_id,
            },
            "package.one_token": {
                "contract": batch_int,
                "role": {"kind": "opaque"},
                "source": {"kind": "literal"},
                "required": False,
                "default": 1,
            },
            "package.one_step": {
                # A growing-recurrence increment advances the whole invocation's
                # state axis by one, so it is an invocation-scoped control scalar,
                # not a per-row value like ``package.one``/``package.one_token``.
                "contract": control_int,
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
        }
    )
    if static_cache is not None:
        # The capacity a static graph was built against is a graph fact, not a
        # deployment budget: it bounds legal write destinations and nothing else.
        workflow_inputs["package.cache_capacity"] = {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": int(static_cache["capacity"]),
        }
    if cache_pairs:
        workflow_inputs.update(
            {
                "request.prompt_lengths": {
                    "contract": batch_int,
                    "role": {"kind": "opaque"},
                    "source": {"kind": "application", "name": "prompt_lengths"},
                    "required": False,
                    "default": -1,
                },
                "request.eos_ids": {
                    "contract": {
                        "dtype": "int64",
                        "rank": 2,
                        "shape": [batch_dimension, "num_eos"],
                    },
                    "role": {"kind": "opaque"},
                    "source": {"kind": "application", "name": "eos_ids"},
                    "required": False,
                    "default": eos_token_id,
                },
                "request.eos_lengths": {
                    "contract": batch_int,
                    "role": {"kind": "opaque"},
                    "source": {"kind": "application", "name": "eos_lengths"},
                    "required": False,
                    "default": 1,
                },
                "request.row_max_iterations": {
                    "contract": batch_int,
                    "role": {"kind": "opaque"},
                    "source": {
                        "kind": "application",
                        "name": "row_max_iterations",
                    },
                    "required": False,
                    "default": -1,
                },
            }
        )
    stochastic_sampler = sampler != "greedy"
    sampler_with_rng = stochastic_sampler or bool(cache_pairs)
    if sampler_with_rng:
        workflow_inputs.update(
            {
                "request.temperature": {
                    "contract": {
                        "dtype": "float32",
                        "rank": 1,
                        "shape": [batch_dimension],
                    },
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
                    "contract": batch_int,
                    "role": {
                        "kind": "runtime",
                        "version": "1.0",
                        "role": "sampling_top_k",
                    },
                    "source": {"kind": "request", "field": "sampling_top_k"},
                    "required": False,
                    "default": 0 if stochastic_sampler else 1,
                },
                "request.top_p": {
                    "contract": {
                        "dtype": "float32",
                        "rank": 1,
                        "shape": [batch_dimension],
                    },
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
                    "contract": {
                        "dtype": "float32",
                        "rank": 1,
                        "shape": [batch_dimension],
                    },
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
                    "required": False,
                    "default": 0,
                },
                "request.rng_counter": {
                    "contract": batch_int,
                    "role": {"kind": "opaque"},
                    "source": {"kind": "application", "name": "rng_counter"},
                    "required": False,
                    "default": 0,
                },
            }
        )
    if cache_pairs:
        pkg.add_policy_component("cache_length_update", build_selective_integer_add())
        pkg.add_policy_component(
            "token_sampler",
            build_seeded_categorical_sampler(),
        )
        pkg.add_policy_component("termination", build_eos_termination(row_selective=True))
        pkg.add_policy_component(
            "termination_batch_initializer",
            build_termination_batch_initializer(),
        )
        pkg.add_policy_component(
            "token_state_update",
            build_token_state_update(row_selective=True),
        )
        pkg.add_policy_component("token_to_slot", build_token_to_slot())
        pkg.add_policy_component("generated_length_update", build_selective_integer_add())
        workflow_inputs.update(
            {
                "package.active": {
                    "contract": batch_bool,
                    "role": {"kind": "opaque"},
                    "source": {"kind": "literal"},
                    "required": False,
                    "default": True,
                },
                "package.not_done": {
                    "contract": batch_bool,
                    "role": {"kind": "opaque"},
                    "source": {"kind": "literal"},
                    "required": False,
                    "default": False,
                },
                "package.cache_lengths": {
                    "contract": batch_int,
                    "role": {"kind": "opaque"},
                    "source": {"kind": "literal"},
                    "required": False,
                    "default": 0,
                },
                "package.zero_batch": {
                    "contract": batch_int,
                    "role": {"kind": "opaque"},
                    "source": {"kind": "literal"},
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
    if attention_input is not None:
        setup_decoder_inputs[attention_input.name] = f"initializer.{attention_input.name}"
        body_decoder_inputs[attention_input.name] = (
            "decoder_step.body_attention_mask"
            if fixed_capacity
            else "state.attention_mask.body"
        )
    if position_input is not None:
        setup_decoder_inputs[position_input.name] = f"initializer.{position_input.name}"
        body_decoder_inputs[position_input.name] = "state.position_ids.body"
    if static_cache is not None:
        # Prefill scatters the whole prompt from slot zero and ends with
        # ``prompt_length`` valid entries. Each decode step then writes at the
        # row's current logical length and ends one entry longer — except for a
        # finished row, whose length does not advance, so the slot it just wrote
        # stays outside its valid prefix and is reclaimed by its next write.
        setup_decoder_inputs[static_cache["write_indices"]] = (
            f"initializer.{static_cache['write_indices']}"
        )
        setup_decoder_inputs[static_cache["kv_sequence_length"]] = "initializer.cache_lengths"
        body_decoder_inputs[static_cache["write_indices"]] = "state.cache_lengths.body"
        body_decoder_inputs[static_cache["kv_sequence_length"]] = "cache_lengths.next"
    # Cross state is produced once by the encoder and read unchanged by every
    # decode step, so setup binds the encoder result and the body binds the
    # invariant carried cell that holds it.
    for decoder_input, (encoder_output, _) in sorted(cross_bindings.items()):
        setup_decoder_inputs[decoder_input] = f"encoder.{encoder_output}"
        body_decoder_inputs[decoder_input] = f"state.cross.{decoder_input}.body"
    body_decoder_inputs[token_input.name] = (
        "model_token.body" if needs_token_cast else "token.body"
    )

    setup_decoder_outputs = {logits_output.name: "decoder.setup.logits"}
    body_decoder_outputs = {logits_output.name: "decoder.body.logits"}
    logits_contract = _contract(logits_output)
    last_logits_contract = {
        "dtype": "float32",
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
        "logits": {
            "contract": last_logits_contract,
            "scope": "invocation",
            "initializer": "decoder.setup.last_logits",
            "recurrence": {"kind": "invariant"},
        },
    }
    if cache_pairs:
        state.update(
            {
                "generated_lengths": {
                    "contract": batch_int,
                    "class": "semantic",
                    "scope": "invocation",
                    "initializer": "initializer.generated_lengths",
                    "recurrence": {"kind": "invariant"},
                },
                "active": {
                    "contract": batch_bool,
                    "class": "semantic",
                    "scope": "invocation",
                    "initializer": "package.active",
                    "recurrence": {"kind": "invariant"},
                },
                "done": {
                    "contract": batch_bool,
                    "class": "semantic",
                    "scope": "invocation",
                    "initializer": "package.not_done",
                    "recurrence": {"kind": "invariant"},
                },
                "accepted_len": {
                    "contract": batch_int,
                    "class": "semantic",
                    "scope": "invocation",
                    "initializer": "package.zero_batch",
                    "recurrence": {"kind": "invariant"},
                },
                "cache_lengths": {
                    "contract": batch_int,
                    "class": "semantic",
                    "scope": "invocation",
                    "initializer": (
                        "initializer.cache_lengths"
                        if tracks_cache_lengths
                        else "package.cache_lengths"
                    ),
                    "recurrence": {"kind": "invariant"},
                },
            }
        )
    initial_effects = {
        "sample": "sample.0",
        "termination": "termination.0",
        "state": "state.0",
        "emit": "emit.0",
        "state:token": "state:token.0",
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
    if cache_pairs:
        carried.extend(
            [
                {
                    "cell": "generated_lengths",
                    "current": "initializer.generated_lengths",
                    "body_input": "state.generated_lengths.body",
                    "body_output": "token.next_lengths",
                    "next": "state.generated_lengths.final",
                },
                {
                    "cell": "active",
                    "current": "package.active",
                    "body_input": "state.active.body",
                    "body_output": "loop.next_active",
                    "next": "state.active.final",
                },
                {
                    "cell": "done",
                    "current": "package.not_done",
                    "body_input": "state.done.body",
                    "body_output": "loop.done",
                    "next": "state.done.final",
                },
                {
                    "cell": "cache_lengths",
                    "current": (
                        "initializer.cache_lengths"
                        if tracks_cache_lengths
                        else "package.cache_lengths"
                    ),
                    "body_input": "state.cache_lengths.body",
                    "body_output": "cache_lengths.next",
                    "next": "state.cache_lengths.final",
                },
                {
                    "cell": "accepted_len",
                    "current": "package.zero_batch",
                    "body_input": "state.accepted_len.body",
                    "body_output": "accepted_len.next",
                    "next": "state.accepted_len.final",
                },
            ]
        )
    if sampler_with_rng:
        state["rng_counter"] = {
            "contract": batch_int,
            "scope": "invocation",
            "class": "semantic",
            "initializer": "request.rng_counter",
            "recurrence": {"kind": "invariant"},
        }
        initial_effects["state:rng_counter"] = "state:rng_counter.0"
        carried.append(
            {
                "cell": "rng_counter",
                "current": "request.rng_counter",
                "body_input": "state.rng_counter.body",
                "body_output": "sample.next_counter",
                "next": "state.rng_counter.final",
                "read_effect": _effect("state:rng_counter.0", "state:rng_counter.read"),
                "write_effect": _effect("state:rng_counter.read", "state:rng_counter.1"),
            }
        )
    decoder_state_specs: dict[str, tuple[dict[str, Any], str, str, dict[str, Any]]] = {}
    if attention_input is not None:
        decoder_state_specs["attention_mask"] = (
            {
                "dtype": _contract(attention_input)["dtype"],
                "rank": 2,
                "shape": [batch_dimension, "context"],
            },
            (
                f"initializer.{attention_input.name}"
                if fixed_capacity
                else "initializer.body_attention_mask"
            ),
            "decoder_step.body_attention_mask",
            (
                {"kind": "invariant"}
                if fixed_capacity
                else {
                    "kind": "growing",
                    "axis": 1,
                    "increment": "package.one_step",
                    "max": "package.max_context",
                }
            ),
        )
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
    decoder_kv_ports: dict[str, Any] = {}
    decoder_cache_cells: list[str] = []
    decoder_kv_axis = 2
    for cache_index, (past, present) in enumerate(cache_pairs):
        # Generated-length state is orthogonal to the admitted cache ABI and
        # must not renumber stable cache service cells.
        cell = f"cache_{cache_index}"
        setup_value = f"decoder.setup.{present.name}"
        body_value = f"decoder.body.{present.name}"
        setup_decoder_outputs[present.name] = setup_value
        body_decoder_outputs[present.name] = body_value
        decoder_kv_axis = next(
            (
                index
                for index, dimension in enumerate(_contract(past)["shape"])
                if "sequence" in str(dimension)
            ),
            2,
        )
        # A scattered buffer never changes shape: every step overwrites cells
        # inside a capacity fixed at export, so its extent is invariant and the
        # logical prefix is carried separately by ``cache_lengths``.
        scattered = static_cache is not None and past.name in static_cache["buffers"]
        state[cell] = {
            "contract": _request_aligned(_contract(past)),
            "scope": "invocation",
            "initializer": setup_value,
            "recurrence": (
                {"kind": "invariant"}
                if scattered
                else {
                    "kind": "bounded",
                    "axis": decoder_kv_axis,
                    "max": "package.max_context",
                }
            ),
            # Binding a cell to a state service group hands its storage to the
            # runtime, which then owns allocation, compaction, and release.
            "management": "runtime",
            "release_boundary": "invocation",
        }
        decoder_kv_ports[cell] = {"input": past.name, "output": present.name}
        decoder_cache_cells.append(cell)
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

    # Cross state: produced once by the encoder, shape-invariant, carried
    # unchanged. The identity carry keeps the cell request-aligned on the same
    # axis as the tokens so row compaction permutes encoder state and decoder
    # rows together.
    for decoder_input, (encoder_output, encoder_value) in sorted(cross_bindings.items()):
        cell = f"cross.{decoder_input}"
        setup_value = f"encoder.{encoder_output}"
        body_input = f"state.cross.{decoder_input}.body"
        contract = _contract(encoder_value)
        contract["batch_layout"] = {"kind": "request_aligned", "axis": 0}
        state[cell] = {
            "contract": contract,
            "class": "semantic",
            "scope": "invocation",
            "initializer": setup_value,
            "recurrence": {"kind": "invariant"},
        }
        effect_name = f"state:{cell}"
        initial_effects[effect_name] = f"{effect_name}.0"
        carried.append(
            {
                "cell": cell,
                "current": setup_value,
                "body_input": body_input,
                "body_output": body_input,
                "next": f"state.cross.{decoder_input}.final",
                "read_effect": _effect(f"{effect_name}.0", f"{effect_name}.read"),
                "write_effect": _effect(f"{effect_name}.read", f"{effect_name}.1"),
            }
        )
    # One semantic state group per attention kind. A hybrid decoder (Gemma 3/4,
    # Gemma 3n, ...) publishes distinct sliding and full-attention groups so the
    # runtime never evicts a global layer's prefix, and so each group's cells
    # carry their own geometry (Gemma 4's global layers are double-wide).
    decoder_state_groups, decoder_cell_groups = _state_service_groups(
        config=config,
        cache_pairs=cache_pairs,
        ports={decoder_name: decoder_kv_ports},
        sequence_axis=decoder_kv_axis,
        # A mask-free decoder grows its cache instead of writing into a shared
        # full-capacity buffer: it publishes no logical lengths and its present
        # ports must never be aliased onto the past bindings.
        logical_lengths="cache_lengths" if fixed_capacity else None,
        aliasing=_aliasing_for_storage(
            decoder_kv_contract["storage"] if fixed_capacity else "growable"
        ),
        base_name="decoder_cache",
        indexed_scatter=(
            {
                "buffers": static_cache["buffers"],
                "capacity": "package.cache_capacity",
                # The write cursor and the logical length are the same quantity:
                # a row's next write lands exactly where its valid prefix ends.
                "write_indices": "cache_lengths",
                "logical_lengths": "cache_lengths",
                "port": static_cache["write_indices"],
                "kv_length_port": static_cache["kv_sequence_length"],
            }
            if static_cache is not None
            else None
        ),
    )
    for cell in decoder_cache_cells:
        state[cell]["service_group"] = decoder_cell_groups[cell]

    setup = {
        "kind": "sequence",
        "nodes": [
            *(
                [
                    _invoke(
                        "audio_preprocess",
                        {"encoded": "request.audio"},
                        dict(audio_values),
                    )
                ]
                if audio_program is not None
                else []
            ),
            *(
                [_invoke(encoder_name, encoder_invoke_inputs, encoder_invoke_outputs)]
                if encoder is not None
                else []
            ),
            _invoke(
                "decoder_state_initializer",
                {
                    "prompt_tokens": f"request.{token_input.name}",
                    **({"prompt_lengths": "request.prompt_lengths"} if cache_pairs else {}),
                    **({"max_iterations": "request.max_iterations"} if fixed_capacity else {}),
                },
                {
                    **(
                        {
                            attention_input.name: f"initializer.{attention_input.name}",
                            "body_attention_mask": "initializer.body_attention_mask",
                        }
                        if attention_input is not None
                        else {}
                    ),
                    "token_slot": "initializer.token_slot",
                    **(
                        {"generated_lengths": "initializer.generated_lengths"}
                        if cache_pairs
                        else {}
                    ),
                    **(
                        {"cache_lengths": "initializer.cache_lengths"}
                        if tracks_cache_lengths
                        else {}
                    ),
                    **(
                        {
                            static_cache["write_indices"]: (
                                f"initializer.{static_cache['write_indices']}"
                            )
                        }
                        if static_cache is not None
                        else {}
                    ),
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
            *(
                [
                    _invoke(
                        "termination_batch_initializer",
                        {
                            "input_eos_ids": "request.eos_ids",
                            "input_eos_lengths": "request.eos_lengths",
                            "input_max_iterations": "request.row_max_iterations",
                            "fallback_max_iterations": "request.max_iterations",
                            "active": "package.active",
                        },
                        {
                            "row_eos_ids": "termination.eos_ids",
                            "eos_lengths": "termination.eos_lengths",
                            "max_iterations": "termination.max_iterations",
                        },
                    )
                ]
                if cache_pairs
                else []
            ),
            _invoke(
                "last_token_logits",
                {"logits": "decoder.setup.logits"},
                {"last_logits": "decoder.setup.last_logits"},
            ),
        ],
    }
    has_step_update = attention_input is not None or position_input is not None
    decoder_step_invoke = _invoke(
        "decoder_step_update",
        {
            **(
                {"attention_mask": "state.attention_mask.body"}
                if attention_input is not None
                else {}
            ),
            **({"logical_length": "state.cache_lengths.body"} if fixed_capacity else {}),
            **(
                {"position_ids": "state.position_ids.body"}
                if position_input is not None
                else {}
            ),
        },
        {
            **(
                {"next_attention_mask": "decoder_step.body_attention_mask"}
                if attention_input is not None
                else {}
            ),
            **(
                {"next_position_ids": "decoder_step.body_position_ids"}
                if position_input is not None
                else {}
            ),
        },
    )
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
                            "seed": "request.seed",
                            "counter": "state.rng_counter.body",
                        }
                        if sampler_with_rng
                        else {}
                    ),
                    **(
                        {
                            "active": "state.active.body",
                            "done": "state.done.body",
                        }
                        if cache_pairs
                        else {}
                    ),
                },
                {
                    "token": "sample.body",
                    **({"next_counter": "sample.next_counter"} if sampler_with_rng else {}),
                },
                {"sample": _effect("sample.0", "sample.1")},
            ),
            *(
                [
                    _invoke(
                        "token_to_slot",
                        {"token": "sample.body"},
                        {"slot": "sample.slot"},
                    ),
                    _invoke(
                        "generated_length_update",
                        {
                            "left": "state.generated_lengths.body",
                            "right": "package.one_token",
                            "active": "state.active.body",
                            "done": "state.done.body",
                        },
                        {"total": "token.next_lengths"},
                    ),
                    _invoke(
                        "generated_length_update",
                        {
                            "left": "package.zero_batch",
                            "right": "package.one_token",
                            "active": "state.active.body",
                            "done": "state.done.body",
                        },
                        {"total": "token.emitted_length"},
                    ),
                ]
                if cache_pairs
                else []
            ),
            _invoke(
                "token_state_update",
                {
                    "current": "state.token.body",
                    "update": "sample.slot" if cache_pairs else "sample.body",
                    **(
                        {
                            "active": "state.active.body",
                            "done": "state.done.body",
                        }
                        if cache_pairs
                        else {}
                    ),
                },
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
                    "tokens": "sample.body",
                    "eos_ids": ("termination.eos_ids" if cache_pairs else "package.eos_ids"),
                    **({"eos_lengths": "termination.eos_lengths"} if cache_pairs else {}),
                    "iteration": "loop.iteration",
                    "max_iterations": (
                        "termination.max_iterations"
                        if cache_pairs
                        else "request.max_iterations"
                    ),
                    **({"active": "state.active.body"} if cache_pairs else {}),
                },
                {
                    "done": "loop.done",
                    "continue": "loop.continue",
                    **({"next_active": "loop.next_active"} if cache_pairs else {}),
                },
                {"termination": _effect("termination.0", "termination.1")},
            ),
            *(
                [
                    _invoke(
                        "cache_length_update",
                        {
                            "left": "state.cache_lengths.body",
                            "right": "package.one_token",
                            "active": "state.active.body",
                            "done": "state.done.body",
                        },
                        {"total": "cache_lengths.next"},
                    ),
                    _invoke(
                        "cache_length_update",
                        {
                            "left": "package.zero_batch",
                            "right": "package.one_token",
                            "active": "state.active.body",
                            "done": "state.done.body",
                        },
                        {"total": "accepted_len.next"},
                    ),
                ]
                if cache_pairs
                else []
            ),
            {
                "kind": "emit",
                "value": "token.body",
                "output": "tokens",
                "mode": "append",
                **(
                    {
                        "when": "state.active.body",
                        "valid_length": "token.emitted_length",
                    }
                    if cache_pairs
                    else {}
                ),
                "effect_name": "emit",
                "effect": _effect("emit.0", "emit.1"),
            },
            *([decoder_step_invoke] if has_step_update and fixed_capacity else []),
            _invoke(decoder_name, body_decoder_inputs, body_decoder_outputs),
            _invoke(
                "last_token_logits",
                {"logits": "decoder.body.logits"},
                {"last_logits": "decoder.body.last_logits"},
            ),
            *([decoder_step_invoke] if has_step_update and not fixed_capacity else []),
        ],
    }

    artifacts = artifacts or {}
    use_subfolders = len(pkg) > 1
    artifact = artifacts.get(
        decoder_name, f"{decoder_name}/model.onnx" if use_subfolders else "model.onnx"
    )
    workflow = {
        "manifest": {
            **(
                {"adapter_abis": {"onnx-genai.audio-preprocess": "1"}}
                if audio_program is not None
                else {}
            ),
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "typed_emit",
                "emit_valid_length",
                "loop_induction_values",
                *(["audio_preprocessing_program"] if audio_program is not None else []),
                *(["serving_service_contract"] if cache_pairs else []),
                *(["bounded_state_recurrence"] if cache_pairs else []),
            ],
        },
        "inputs": workflow_inputs,
        "outputs": {
            "tokens": {
                "contract": _request_aligned(
                    {
                        "dtype": "int64",
                        "rank": 2,
                        "shape": [batch_dimension, "generated_sequence"],
                    }
                ),
                "role": "tokens",
                "stage": "pre_adapter",
            }
        },
        "components": {
            decoder_name: _component(decoder, artifact),
            **(
                {
                    encoder_name: _component(
                        encoder,
                        artifacts.get(encoder_name, f"{encoder_name}/model.onnx"),
                    )
                }
                if encoder is not None
                else {}
            ),
            **(
                {
                    "audio_preprocess": {
                        "implementation": {
                            "kind": "adapter",
                            "abi": "onnx-genai.audio-preprocess",
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
                            "outputs": audio_adapter_outputs,
                        },
                    }
                }
                if audio_program is not None
                else {}
            ),
        },
        "state": state,
        **(
            {
                "serving": {
                    "active": "active",
                    "done": "done",
                    "accepted_len": "accepted_len",
                    "state_service": {"groups": decoder_state_groups},
                }
            }
            if cache_pairs
            else {}
        ),
        "initial_effects": initial_effects,
        "graph": {
            "kind": "loop",
            "setup": setup,
            "body": body,
            "condition": "loop.continue",
            "termination": "generation_eos",
            **({"active_cell": "active"} if cache_pairs else {}),
            "max_iterations": "request.max_iterations",
            "iteration": {
                "value": "loop.iteration",
                "contract": {"dtype": "int64", "rank": 1, "shape": [1]},
            },
            "carried": carried,
        },
    }
    metadata = {
        "schema_version": "1.0",
        **({"preprocessing": {"audio": audio_program}} if audio_program is not None else {}),
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

    token_contract = _contract(token_input)
    mask_contract = {
        "dtype": "bool",
        "rank": 2,
        "shape": token_contract["shape"],
    }
    batch_dimension = token_contract["shape"][0]
    batch_int = _request_aligned({"dtype": "int64", "rank": 1, "shape": [batch_dimension]})
    control_int = {"dtype": "int64", "rank": 1, "shape": [1]}
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
            "contract": control_int,
            "role": {
                "kind": "runtime",
                "version": "1.0",
                "role": "max_iterations",
            },
            "source": {"kind": "request", "field": "max_iterations"},
            "required": False,
            "default": num_inference_steps,
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
                "continue": f"{prefix}.continue",
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
                "loop.iteration",
                "state.rng_offset.body",
                "state.logits.body",
                "state.proposal.body",
                "denoiser.body",
                "update.0",
                "update.1",
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
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "nested_control_flow",
                "typed_emit",
                "emit_valid_length",
                "loop_induction_values",
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
            "iteration": {
                "value": "loop.iteration",
                "contract": {"dtype": "int64", "rank": 1, "shape": [1]},
            },
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
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


def write_speech_to_text_workflow_metadata(
    pkg: Any,
    output_dir: str,
    config: Any,
    *,
    sampler: str = "greedy",
    audio_preprocessing: dict[str, Any] | None = None,
) -> str:
    """Write encoder-conditioned decode workflow metadata and policy artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_speech_to_text_workflow_metadata(
        pkg,
        config,
        sampler=sampler,
        audio_preprocessing=audio_preprocessing,
    )
    pkg.save_policy_components(output_dir)
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
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
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


_AUDIO_PREPROCESS_ABI = "onnx-genai.audio-preprocess"
_AUDIO_PREPROCESS_ABI_VERSION = "1"


def _audio_preprocess_component(
    values_contract: dict[str, Any],
    mask_contract: dict[str, Any],
) -> dict[str, Any]:
    """Declare the versioned audio-preprocessing adapter component.

    The adapter turns request-supplied encoded audio bytes into the encoder's
    waveform tensor and its sample-level validity mask.  Its ports are declared
    so the runtime can type-check the binding without knowing which model family
    produced the package.
    """
    return {
        "implementation": {
            "kind": "adapter",
            "abi": _AUDIO_PREPROCESS_ABI,
            "version": _AUDIO_PREPROCESS_ABI_VERSION,
        },
        "ports": {
            "inputs": {
                "encoded": {"dtype": "uint8", "rank": 1, "shape": ["bytes"]},
            },
            "outputs": {
                "input_values": values_contract,
                "attention_mask": mask_contract,
            },
        },
        "contract": {
            "id": _AUDIO_PREPROCESS_ABI,
            "version": _AUDIO_PREPROCESS_ABI_VERSION,
            "bindings": {
                "encoded": "encoded",
                "input_values": "input_values",
                "attention_mask": "attention_mask",
            },
        },
        "effects": ["audio_preprocess"],
    }


def _ctc_vocabulary(source: str | None, vocab_size: int) -> dict[str, Any]:
    """Describe the class-id → string table used to render a transcript.

    The table is inlined when the source checkpoint's ``vocab.json`` is
    reachable so the document is self-contained; otherwise the profile points at
    the packaged tokenizer.
    """
    vocabulary: dict[str, Any] = {"source": "tokenizer", "size": vocab_size}
    path = _source_asset_path(source, "vocab.json") if source else None
    if path is None:
        return vocabulary
    try:
        with open(path, encoding="utf-8") as handle:
            table = json.load(handle)
    except (OSError, ValueError):
        return vocabulary
    if not isinstance(table, dict) or not table:
        return vocabulary
    tokens = [""] * (max(int(index) for index in table.values()) + 1)
    for token, index in table.items():
        tokens[int(index)] = token
    if len(tokens) != vocab_size:
        return vocabulary
    vocabulary = {
        "source": "inline",
        "size": vocab_size,
        "tokens": tokens,
    }
    if "|" in table:
        vocabulary["word_delimiter"] = "|"
    ignored = [token for token in ("<pad>", "<s>", "</s>", "<unk>") if token in table]
    if ignored:
        vocabulary["ignored_tokens"] = ignored
    return vocabulary


def build_ctc_asr_workflow_metadata(
    pkg: Any,
    config: Any,
    *,
    source: str | None = None,
    artifact: str = "model.onnx",
) -> dict[str, Any]:
    """Build one-file metadata for a non-generative CTC ASR package.

    A CTC acoustic model is frame-synchronous: the encoder runs exactly once and
    emits one class distribution per frame.  The workflow is therefore a plain
    sequence with no loop and no carried state, and the transcript is recovered
    by the ``transcription`` profile's decoding contract rather than by a
    generation loop.

    Args:
        pkg: The built :class:`ModelPackage`; must hold a single ``model``.
        config: The resolved architecture config (supplies vocabulary size and
            the CTC blank id).
        source: HuggingFace model id or local directory used to inline the
            decoding vocabulary.
        artifact: Encoder artifact path relative to the package root.

    Returns:
        A metadata document with ``preprocessing.audio``, ``profiles`` and a
        single-step ``pipeline.workflow``.
    """
    if "model" not in pkg:
        raise ValueError("CTC ASR workflow requires a 'model' component")
    model = pkg["model"]

    graph_inputs = {value.name: value for value in model.graph.inputs}
    graph_outputs = {value.name: value for value in model.graph.outputs}
    for required in ("input_values", "attention_mask"):
        if required not in graph_inputs:
            raise ValueError(f"CTC ASR encoder must declare input '{required}'")
    if "logits" not in graph_outputs:
        raise ValueError("CTC ASR encoder must declare output 'logits'")
    has_frame_lengths = "frame_lengths" in graph_outputs

    values_contract = _contract(graph_inputs["input_values"])
    mask_contract = _contract(graph_inputs["attention_mask"])
    logits_contract = _contract(graph_outputs["logits"])

    sample_rate = int(getattr(getattr(config, "audio", None), "sampling_rate", 0) or 16_000)
    # Shape inference may leave the class axis unknown (or the whole shape
    # absent), so fall back to the config rather than emitting a vocabulary
    # whose declared size silently disagrees with the graph.
    logits_shape = logits_contract.get("shape") or []
    inferred_classes = logits_shape[-1] if logits_shape else None
    vocab_size = int(inferred_classes or getattr(config, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        raise ValueError(
            "CTC ASR metadata requires a vocabulary size; the 'logits' output "
            "declares no static class axis and the config has no vocab_size"
        )
    blank_id = int(getattr(config, "pad_token_id", 0) or 0)

    workflow_outputs = {
        "logits": {
            "contract": logits_contract,
            "role": "tensor",
            "stage": "post_adapter",
        }
    }
    emit_nodes = [
        {
            "kind": "emit",
            "value": "encoder.logits",
            "output": "logits",
            "mode": "replace",
        }
    ]
    profile_outputs = {"logits": "logits"}
    if has_frame_lengths:
        workflow_outputs["frame_lengths"] = {
            "contract": _contract(graph_outputs["frame_lengths"]),
            "role": "tensor",
            "stage": "post_adapter",
        }
        emit_nodes.append(
            {
                "kind": "emit",
                "value": "encoder.frame_lengths",
                "output": "frame_lengths",
                "mode": "replace",
            }
        )
        profile_outputs["frame_lengths"] = "frame_lengths"

    workflow = {
        "manifest": {
            "adapter_abis": {_AUDIO_PREPROCESS_ABI: _AUDIO_PREPROCESS_ABI_VERSION},
            "capabilities": ["workflow_ssa", "linear_effects", "typed_emit"],
        },
        "effects": {
            # Both steps are pure functions of their inputs: decoding audio and
            # running the encoder observe nothing external, so replay is always
            # safe.  Speculation safety is irrelevant here because a CTC
            # workflow has no speculative region, but it is declared explicitly
            # rather than left to a default.
            "audio_preprocess": {"retry": "pure", "speculation_safety": {"kind": "clonable"}},
            "encode": {"retry": "pure", "speculation_safety": {"kind": "clonable"}},
        },
        "inputs": {
            "request.audio": {
                "contract": {"dtype": "uint8", "rank": 1, "shape": ["bytes"]},
                "role": {"kind": "runtime", "version": "1.0", "role": "media"},
                "source": {"kind": "request", "field": "media"},
                "required": True,
            }
        },
        "outputs": workflow_outputs,
        "components": {
            "audio_preprocess": _audio_preprocess_component(values_contract, mask_contract),
            "encoder": _component(model, artifact, effects=("encode",)),
        },
        "initial_effects": {
            "audio_preprocess": "audio_preprocess.0",
            "encode": "encode.0",
        },
        "graph": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "audio_preprocess",
                    {"encoded": "request.audio"},
                    {
                        "input_values": "audio.input_values",
                        "attention_mask": "audio.attention_mask",
                    },
                ),
                _invoke(
                    "encoder",
                    {
                        "input_values": "audio.input_values",
                        "attention_mask": "audio.attention_mask",
                    },
                    {
                        name: f"encoder.{name}"
                        for name in ("logits", "frame_lengths")
                        if name in graph_outputs
                    },
                ),
                *emit_nodes,
            ],
        },
    }

    decoding: dict[str, Any] = {
        "kind": "ctc",
        "blank_id": blank_id,
        "collapse_repeats": True,
        "time_axis": 1,
        "class_axis": 2,
        "vocabulary": _ctc_vocabulary(source, vocab_size),
    }
    if has_frame_lengths:
        decoding["lengths"] = "frame_lengths"

    # A feature extractor that reduces over the padded time axis makes a row's
    # values depend on the width of the batch it was padded into.  The fact is
    # recorded by the task on the built graph; when nobody stated it we leave
    # the field absent rather than claim rows are independent.
    normalization = getattr(config, "feat_extract_norm", None)
    if normalization == "group":
        batch_invariance = "padding_sensitive"
    elif normalization == "layer":
        batch_invariance = "row_independent"
    else:
        recorded = model.metadata_props.get(BATCH_PADDING_SENSITIVE_KEY)
        batch_invariance = (
            None
            if recorded is None
            else ("padding_sensitive" if recorded == "true" else "row_independent")
        )

    profile: dict[str, Any] = {
        "kind": "transcription",
        "version": "1.0",
        "requirement": "required",
        "outputs": profile_outputs,
        "decoding": decoding,
    }
    # The claim is only checkable when the package also publishes per-row
    # lengths; without them a reader cannot isolate a row's valid region.
    if batch_invariance == "row_independent" or has_frame_lengths:
        if batch_invariance is not None:
            profile["batch_invariance"] = batch_invariance

    return {
        "schema_version": "v1",
        "preprocessing": {
            "audio": {
                "transforms": [
                    {"op": "decode", "outputs": ["samples"]},
                    {"op": "resample", "sample_rate": sample_rate},
                    {"op": "downmix", "channels": 1},
                    {"op": "zero_mean_unit_variance", "epsilon": 1e-7},
                    {
                        "op": "pad",
                        "mode": "right",
                        "pad_value": 0.0,
                        "outputs": ["values", "sample_mask"],
                    },
                ],
                "outputs": [
                    {
                        "name": "input_values",
                        "source": "values",
                        "content": "waveform",
                        "dtype": values_contract["dtype"],
                        "rank": values_contract["rank"],
                    },
                    {
                        "name": "attention_mask",
                        "source": "sample_mask",
                        "content": "validity_mask",
                        "dtype": mask_contract["dtype"],
                        "rank": mask_contract["rank"],
                    },
                ],
            }
        },
        "profiles": {
            "transcription": profile,
        },
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }


def write_ctc_asr_workflow_metadata(
    pkg: Any,
    output_dir: str,
    config: Any,
    *,
    source: str | None = None,
) -> str:
    """Write one-file CTC ASR metadata into *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_ctc_asr_workflow_metadata(pkg, config, source=source)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


_ENCODER_EMBEDDING_INPUT_ROLES: dict[str, str] = {
    "input_ids": "prompt_tokens",
    "attention_mask": "attention_mask",
    "token_type_ids": "token_type_ids",
    "position_ids": "position_ids",
}


def build_encoder_embedding_workflow_metadata(
    pkg: Any,
    config: Any = None,
    *,
    artifact: str = "model.onnx",
) -> dict[str, Any]:
    """Build one-file metadata for a bidirectional encoder that emits embeddings.

    An encoder such as BERT, ESM-2 or ProtBert is not generative: it reads the
    whole sequence at once and returns one hidden vector per position. There is
    no next-token step, no KV cache and nothing to sample, so the workflow is a
    plain sequence with a single invocation and no carried state. Describing
    such a package with decoder metadata would publish a generation loop that
    the artifact cannot execute -- it has no ``logits`` port to sample and
    re-feeding its output would be meaningless -- so this builder exists to
    keep the metadata a truthful description of the graph.

    Args:
        pkg: The built :class:`ModelPackage`; must hold a single ``model``.
        config: The resolved architecture config. Unused today; accepted so the
            dispatch site can pass it uniformly with the other builders.
        artifact: Encoder artifact path relative to the package root.

    Returns:
        A metadata document with an ``embedding`` profile and a single-step
        ``pipeline.workflow``.
    """
    del config
    if "model" not in pkg:
        raise ValueError("encoder embedding workflow requires a 'model' component")
    model = pkg["model"]

    graph_inputs = {str(value.name): value for value in model.graph.inputs}
    graph_outputs = {str(value.name): value for value in model.graph.outputs}
    if "input_ids" not in graph_inputs:
        raise ValueError("encoder embedding graph must declare input 'input_ids'")
    if "last_hidden_state" not in graph_outputs:
        raise ValueError("encoder embedding graph must declare output 'last_hidden_state'")

    # Declare exactly the ports the artifact exposes. ESM-2 has no token type
    # embedding, so its graph carries no ``token_type_ids``; BERT-family
    # encoders do. Reading the graph rather than the task signature is what
    # keeps the two packages describable by one builder.
    workflow_inputs: dict[str, Any] = {}
    invoke_inputs: dict[str, str] = {}
    for name, role in _ENCODER_EMBEDDING_INPUT_ROLES.items():
        if name not in graph_inputs:
            continue
        input_role: dict[str, Any]
        input_source: dict[str, Any]
        if role == "prompt_tokens":
            input_role = {"kind": "runtime", "version": "1.0", "role": role}
            input_source = {"kind": "request"}
        else:
            # The portable runtime-role vocabulary intentionally does not
            # encode architecture-specific auxiliary graph inputs.
            input_role = {"kind": "opaque"}
            input_source = {"kind": "application", "name": f"request.{name}"}
        declaration: dict[str, Any] = {
            "contract": _contract(graph_inputs[name]),
            "role": input_role,
            "source": input_source,
            # Every port here is a graph input of a single-invocation workflow,
            # so a runtime must bind all of them; none is optional.
            "required": True,
        }
        workflow_inputs[f"request.{name}"] = declaration
        invoke_inputs[name] = f"request.{name}"

    workflow_outputs: dict[str, Any] = {}
    invoke_outputs: dict[str, str] = {}
    emit_nodes: list[dict[str, Any]] = []
    profile_outputs: dict[str, str] = {}
    for name in ("last_hidden_state", "pooler_output"):
        if name not in graph_outputs:
            continue
        workflow_outputs[name] = {
            "contract": _contract(graph_outputs[name]),
            "role": "tensor",
            "stage": "post_adapter",
        }
        invoke_outputs[name] = f"encoder.{name}"
        emit_nodes.append(
            {
                "kind": "emit",
                "value": f"encoder.{name}",
                "output": name,
                "mode": "replace",
            }
        )
        profile_outputs[name] = name

    workflow = {
        "manifest": {
            "capabilities": ["workflow_ssa", "linear_effects", "typed_emit"],
        },
        "effects": {
            # One pure call: the encoder observes nothing outside its inputs,
            # so a retry replays it exactly and a speculative clone is safe.
            "encode": {"retry": "pure", "speculation_safety": {"kind": "clonable"}},
        },
        "inputs": workflow_inputs,
        "outputs": workflow_outputs,
        "components": {"encoder": _component(model, artifact, effects=("encode",))},
        "initial_effects": {"encode": "encode.0"},
        "graph": {
            "kind": "sequence",
            "nodes": [
                _invoke("encoder", invoke_inputs, invoke_outputs),
                *emit_nodes,
            ],
        },
    }

    profile: dict[str, Any] = {
        "kind": "embedding",
        "version": "1.0",
        "requirement": "required",
        "outputs": profile_outputs,
    }
    if "attention_mask" in graph_inputs:
        # The portable pooling profile describes the supported sequence
        # reduction; the application remains responsible for supplying the
        # architecture-specific attention mask to the graph.
        profile["pooling"] = {
            "kind": "mean",
            "axis": 1,
            "normalize": False,
        }
        profile["batch_invariance"] = "row_independent"

    return {
        "schema_version": "v1",
        "profiles": {"embedding": profile},
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }


def write_encoder_embedding_workflow_metadata(
    pkg: Any,
    output_dir: str,
    config: Any = None,
) -> str:
    """Write one-file encoder-embedding metadata into *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_encoder_embedding_workflow_metadata(pkg, config)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


def _ir_dtype(value: ir.Value) -> ir.DataType:
    """Return the exact ONNX element type of a graph port."""
    dtype = value.dtype
    if dtype is None:
        raise ValueError(f"port {value.name!r} has no element type")
    return dtype


def _duplex_delays(config: Any) -> list[int]:
    """Read the per-stream delay pattern of a Moshi-family full-duplex LM."""
    delays = getattr(config, "delays", None)
    if delays is None:
        raise ValueError("full-duplex workflow requires a delay pattern on the config")
    return [int(value) for value in delays]


def build_full_duplex_workflow_metadata(pkg: Any, config: Any) -> dict[str, Any]:
    """Build typed SSA metadata for one event of a full-duplex speech workflow.

    Full-duplex speech-to-speech models (Moshi, PersonaPlex) consume and produce
    audio simultaneously at a fixed frame rate. One invocation of this workflow is
    exactly one frame: it accepts one packed audio chunk plus a session ID, and it
    emits at most one packed audio chunk. Everything that must survive between
    frames -- the temporal transformer KV cache, the delay ring buffer, the frame
    offset, and the codec prefixes -- is declared as ``session``-scoped state with
    a ``session`` release boundary and an exclusive session lease, so a runtime can
    resume the conversation on the next invocation without replaying history.

    The acoustic (depformer) loop is the only loop in the graph. Its KV cache is
    ``invocation``-scoped because the upstream model resets it on every frame.

    Components:

    * ``encoder`` / ``decoder`` -- the Mimi-style neural audio codec.
    * ``temporal`` -- the frame-rate transformer over the interleaved token frame.
    * ``depformer`` -- the per-frame acoustic transformer over ``num_streams`` substeps.
    * ``frame_assemble`` / ``frame_commit`` -- the delay ring-buffer bookkeeping.
    * ``teacher_select`` -- prefers an externally supplied token over a sampled one.
    * ``token_sampler`` -- greedy sampling shared by the text and acoustic heads.
    """
    required = {"encoder", "decoder", "temporal", "depformer"}
    missing = required - set(pkg.keys())
    if missing:
        raise ValueError(
            f"full-duplex workflow requires components {sorted(required)}; "
            f"missing {sorted(missing)}"
        )
    encoder = pkg["encoder"]
    decoder = pkg["decoder"]
    temporal = pkg["temporal"]
    depformer = pkg["depformer"]

    delays = _duplex_delays(config)
    channels = len(delays)
    max_delay = max(delays)
    cache_length = max_delay + 3
    num_streams = int(getattr(config, "dep_q", channels // 2))
    audio_streams = int(getattr(config, "n_q", num_streams))
    frame_size = int(getattr(config, "frame_size", 1920))
    initial_tokens = [int(getattr(config, "text_initial_token_id", 32000))] + [
        int(getattr(config, "initial_token_id", 2048))
    ] * (channels - 1)

    waveform_input = _find_port(encoder.graph.inputs, "waveform")
    codes_output = _find_port(encoder.graph.outputs, "codes")
    codes_input = _find_port(decoder.graph.inputs, "codes")
    waveform_output = _find_port(decoder.graph.outputs, "waveform")
    input_frame = _find_port(temporal.graph.inputs, "input_frame")
    temporal_mask = _find_port(temporal.graph.inputs, "attention_mask")
    temporal_position = _find_port(temporal.graph.inputs, "position_ids")
    hidden = _find_port(temporal.graph.outputs, "hidden")
    text_logits = _find_port(temporal.graph.outputs, "text_logits")
    dep_hidden = _find_port(depformer.graph.inputs, "hidden")
    dep_prev = _find_port(depformer.graph.inputs, "prev_token")
    dep_index = _find_port(depformer.graph.inputs, "substep_index")
    dep_logits = _find_port(depformer.graph.outputs, "logits")
    ports = (
        waveform_input,
        codes_output,
        codes_input,
        waveform_output,
        input_frame,
        temporal_mask,
        temporal_position,
        hidden,
        text_logits,
        dep_hidden,
        dep_prev,
        dep_index,
        dep_logits,
    )
    if any(port is None for port in ports):
        raise ValueError("full-duplex workflow is missing a required component port")
    temporal_caches = _model_cache_pairs(temporal)
    depformer_caches = _model_cache_pairs(depformer)
    if not temporal_caches:
        raise ValueError("full-duplex workflow requires a temporal KV cache")
    if not depformer_caches:
        raise ValueError("full-duplex workflow requires a depformer KV cache")

    attach_policy_components(pkg, PolicyCapabilities(sampler="greedy"))
    pkg.add_policy_component(
        "frame_assemble",
        build_duplex_frame_assemble(channels=channels, cache_length=cache_length),
    )
    pkg.add_policy_component(
        "frame_commit",
        build_duplex_frame_commit(
            channels=channels, cache_length=cache_length, max_delay=max_delay
        ),
    )
    pkg.add_policy_component("teacher_select", build_duplex_teacher_select(channels=channels))
    pkg.add_policy_component(
        "frame_update", build_code_frame_update(channels, scalar_index=True)
    )
    pkg.add_policy_component(
        "waveform_append", build_duplex_waveform_append(dtype=_ir_dtype(waveform_input))
    )
    pkg.add_policy_component("codes_append", build_duplex_stream_append(streams=audio_streams))
    pkg.add_policy_component("codes_tail", build_duplex_stream_tail(streams=audio_streams))
    pkg.add_policy_component(
        "chunk_tail",
        build_duplex_stream_tail(streams=1, dtype=_ir_dtype(waveform_output)),
    )

    batch = _contract(input_frame)["shape"][0]
    request_aligned = {"kind": "request_aligned", "axis": 0}
    control_int = {"dtype": "int64", "rank": 0, "shape": []}
    scalar_bool = {"dtype": "bool", "rank": 0, "shape": []}
    batch_bool = {
        "dtype": "bool",
        "rank": 1,
        "shape": [batch],
        "batch_layout": request_aligned,
    }
    batch_int = {
        "dtype": "int64",
        "rank": 1,
        "shape": [batch],
        "batch_layout": request_aligned,
    }
    loop_flag = {"dtype": "bool", "rank": 1, "shape": [1]}
    frame_contract = {"dtype": "int64", "rank": 2, "shape": [batch, channels]}
    ring_contract = {"dtype": "int64", "rank": 3, "shape": [batch, channels, cache_length]}
    ring_flags = {"dtype": "bool", "rank": 3, "shape": [batch, channels, cache_length]}

    inputs: dict[str, Any] = {
        "request.audio_chunk": {
            "contract": _contract(waveform_input),
            "role": {"kind": "runtime", "version": "1.0", "role": "media"},
            "source": {"kind": "request", "field": "media"},
            "required": True,
        },
        "request.session_id": {
            "contract": {"dtype": "int64", "rank": 1, "shape": [batch]},
            "role": {"kind": "runtime", "version": "1.0", "role": "session_id"},
            "source": {"kind": "request", "field": "session_id"},
            "required": True,
        },
        "package.stream_tokens": {
            "contract": frame_contract,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            # -1 marks "this stream has nothing to contribute this frame", so the
            # broadcast default is a frame in which the model predicts everything.
            "default": -1,
        },
        "package.frames_per_invocation": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            # One event per invocation by default; a runtime that batches several
            # codec frames into a single call raises this without changing the graph.
            "default": 1,
        },
        "package.true_batch": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": True,
        },
        "package.false_batch": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": False,
        },
        "package.zero_batch": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0,
        },
        "package.one_batch": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.delays": {
            "contract": {"dtype": "int64", "rank": 1, "shape": [channels]},
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": delays,
        },
        "package.initial_tokens": {
            "contract": {"dtype": "int64", "rank": 1, "shape": [channels]},
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": initial_tokens,
        },
        "package.num_streams": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": num_streams,
        },
        "package.one_frame": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
        "package.frame_size": {
            "contract": control_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": frame_size,
        },
        "package.false": {
            "contract": scalar_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": False,
        },
    }

    session_lease = {"policy": "exclusive", "optimistic_metadata_version": False}

    def session_cell(
        contract: dict[str, Any],
        initializer: str,
        recurrence: dict[str, Any],
        *,
        release: str = "session",
    ) -> dict[str, Any]:
        return {
            "contract": contract,
            "scope": "session",
            "initializer": initializer,
            "recurrence": recurrence,
            "management": "runtime",
            "release_boundary": release,
            "session": session_lease,
        }

    invariant = {"kind": "invariant"}
    state: dict[str, Any] = {
        "token_cache": session_cell(ring_contract, "package.token_cache_init", invariant),
        "token_provided": session_cell(ring_flags, "package.token_provided_init", invariant),
        "offset": session_cell(control_int, "package.offset_init", invariant),
        "attention_mask": session_cell(
            {
                "dtype": _contract(temporal_mask)["dtype"],
                "rank": 2,
                "shape": [batch, "context"],
            },
            "package.attention_mask_init",
            {
                "kind": "growing",
                "axis": 1,
                "increment": "package.one_frame",
                "max": "package.context_limit",
            },
        ),
        "position_ids": session_cell(
            _contract(temporal_position), "package.position_ids_init", invariant
        ),
        "user_waveform": session_cell(
            {
                "dtype": _contract(waveform_input)["dtype"],
                "rank": 3,
                "shape": [batch, 1, "user_samples"],
            },
            "package.user_waveform_init",
            {
                "kind": "growing",
                "axis": 2,
                "increment": "package.frame_size",
                "max": "package.codec_prefix_limit",
            },
            release="invocation",
        ),
        "agent_codes": session_cell(
            {
                "dtype": _contract(codes_input)["dtype"],
                "rank": 3,
                "shape": [batch, audio_streams, "agent_frames"],
            },
            "package.agent_codes_init",
            {
                "kind": "growing",
                "axis": 2,
                "increment": "package.one_frame",
                "max": "package.codec_frame_limit",
            },
            release="invocation",
        ),
    }
    inputs["package.context_limit"] = {
        "contract": control_int,
        "role": {"kind": "opaque"},
        "source": {"kind": "literal"},
        "required": False,
        "default": int(getattr(config, "context", 3000)),
    }
    inputs["package.codec_prefix_limit"] = {
        "contract": control_int,
        "role": {"kind": "opaque"},
        "source": {"kind": "literal"},
        "required": False,
        "default": int(getattr(config, "context", 3000)) * frame_size,
    }
    inputs["package.codec_frame_limit"] = {
        "contract": control_int,
        "role": {"kind": "opaque"},
        "source": {"kind": "literal"},
        "required": False,
        "default": int(getattr(config, "context", 3000)),
    }
    zero_default = {"int64": 0, "int32": 0, "bool": False}
    for name, initial in (
        ("package.token_cache_init", ring_contract),
        ("package.token_provided_init", ring_flags),
        ("package.attention_mask_init", state["attention_mask"]["contract"]),
        ("package.position_ids_init", _contract(temporal_position)),
        ("package.user_waveform_init", state["user_waveform"]["contract"]),
        ("package.agent_codes_init", state["agent_codes"]["contract"]),
    ):
        inputs[name] = {
            "contract": initial,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            # A scalar default broadcasts across the whole contract, so an empty
            # session starts from an all-zero (or all-false) tensor.
            "default": zero_default.get(initial["dtype"], 0.0),
        }
    inputs["package.offset_init"] = {
        "contract": control_int,
        "role": {"kind": "opaque"},
        "source": {"kind": "literal"},
        "required": False,
        "default": 1,
    }

    for index, (past, present) in enumerate(temporal_caches):
        # A runtime that owns the cache must be able to permute and compact rows,
        # so a service-bound cell has to declare its request axis explicitly.
        cache_contract = {**_contract(past), "batch_layout": request_aligned}
        state[f"temporal_cache_{index}"] = session_cell(
            cache_contract,
            f"package.temporal_cache_{index}_init",
            {
                # The temporal cache is a sliding context window: it grows one
                # frame per invocation but never past ``context``, so it is
                # bounded rather than unboundedly growing.
                "kind": "bounded",
                "axis": 2,
                "max": "package.context_limit",
            },
        )
        state[f"temporal_cache_{index}"]["service_group"] = "temporal_cache"
        state[f"temporal_cache_{index}"]["class"] = "semantic"
        inputs[f"package.temporal_cache_{index}_init"] = {
            "contract": cache_contract,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0.0,
        }
        del present
    state["temporal_cache_lengths"] = {
        "contract": batch_int,
        "class": "semantic",
        "scope": "session",
        "initializer": "package.zero_batch",
        "recurrence": invariant,
        "session": session_lease,
    }
    state["active"] = {
        "contract": batch_bool,
        "class": "semantic",
        "scope": "invocation",
        "initializer": "package.true_batch",
        "recurrence": invariant,
    }
    state["done"] = {
        "contract": batch_bool,
        "class": "semantic",
        "scope": "invocation",
        "initializer": "package.false_batch",
        "recurrence": invariant,
    }
    state["accepted_len"] = {
        "contract": batch_int,
        "class": "semantic",
        "scope": "invocation",
        "initializer": "package.zero_batch",
        "recurrence": invariant,
    }
    for index, (past, _) in enumerate(depformer_caches):
        state[f"depformer_cache_{index}"] = {
            "contract": _contract(past),
            "scope": "invocation",
            "initializer": f"package.depformer_cache_{index}_init",
            "recurrence": {
                "kind": "growing",
                "axis": 2,
                "increment": "package.one_frame",
                "max": "package.num_streams",
            },
        }
        inputs[f"package.depformer_cache_{index}_init"] = {
            "contract": _contract(past),
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 0.0,
        }
    state["acoustic_frame"] = {
        "contract": frame_contract,
        "scope": "invocation",
        "initializer": "duplex.target_frame",
        "recurrence": invariant,
    }
    state["prev_token"] = {
        "contract": {"dtype": "int64", "rank": 1, "shape": [batch]},
        "scope": "invocation",
        "initializer": "duplex.text_token",
        "recurrence": invariant,
    }

    depformer_inputs = {
        dep_hidden.name: "duplex.hidden",
        dep_prev.name: "duplex.prev_token_slot",
        dep_index.name: "duplex.substep",
        **{
            past.name: f"state.depformer_cache_{index}.body"
            for index, (past, _) in enumerate(depformer_caches)
        },
    }
    depformer_outputs = {
        dep_logits.name: "duplex.acoustic_logits",
        **{
            present.name: f"duplex.depformer_cache_{index}"
            for index, (_, present) in enumerate(depformer_caches)
        },
    }

    acoustic_loop = {
        "kind": "loop",
        "setup": {"kind": "sequence", "nodes": []},
        "body": {
            "kind": "sequence",
            "nodes": [
                _invoke(
                    "token_to_slot",
                    {"token": "state.prev_token.body"},
                    {"slot": "duplex.prev_token_slot"},
                ),
                _invoke("depformer", depformer_inputs, depformer_outputs),
                _invoke(
                    "last_acoustic_logits",
                    {"logits": "duplex.acoustic_logits"},
                    {"last_logits": "duplex.acoustic_last"},
                ),
                _invoke(
                    "token_sampler",
                    {"logits": "duplex.acoustic_last"},
                    {"token": "duplex.acoustic_sampled"},
                ),
                _invoke(
                    "frame_update",
                    {
                        "frame_codes": "state.acoustic_frame.body",
                        "token": "duplex.acoustic_sampled",
                        "index": "duplex.acoustic_stream",
                    },
                    {"next_frame": "duplex.acoustic_frame_next"},
                ),
                _invoke(
                    "teacher_select",
                    {
                        "target": "duplex.target",
                        "target_provided": "duplex.target_provided",
                        "sampled": "duplex.acoustic_sampled",
                        "index": "duplex.acoustic_stream",
                    },
                    {"token": "duplex.acoustic_prev_next"},
                ),
            ],
        },
        "condition": "package.loop_active",
        "max_iterations": "package.num_streams",
        "iteration": {"value": "duplex.substep", "contract": control_int},
        "carried": [
            {
                "cell": "acoustic_frame",
                "current": "duplex.target_frame",
                "body_input": "state.acoustic_frame.body",
                "body_output": "duplex.acoustic_frame_next",
                "next": "duplex.acoustic_frame_final",
            },
            {
                "cell": "prev_token",
                "current": "duplex.text_token",
                "body_input": "state.prev_token.body",
                "body_output": "duplex.acoustic_prev_next",
                "next": "duplex.prev_token_final",
            },
            *[
                {
                    "cell": f"depformer_cache_{index}",
                    "current": f"package.depformer_cache_{index}_init",
                    "body_input": f"state.depformer_cache_{index}.body",
                    "body_output": f"duplex.depformer_cache_{index}",
                    "next": f"duplex.depformer_cache_{index}_final",
                }
                for index in range(len(depformer_caches))
            ],
        ],
    }
    inputs["package.loop_active"] = {
        "contract": loop_flag,
        "role": {"kind": "opaque"},
        "source": {"kind": "literal"},
        "required": False,
        "default": True,
    }
    inputs["package.acoustic_stream_offset"] = {
        "contract": control_int,
        "role": {"kind": "opaque"},
        "source": {"kind": "literal"},
        "required": False,
        "default": 1,
    }
    acoustic_loop["body"]["nodes"].insert(
        0,
        _invoke(
            "stream_index",
            {"left": "duplex.substep", "right": "package.acoustic_stream_offset"},
            {"total": "duplex.acoustic_stream"},
        ),
    )
    pkg.add_policy_component("stream_index", build_scalar_integer_add())
    pkg.add_policy_component("last_text_logits", build_last_token_logits())
    pkg.add_policy_component("last_acoustic_logits", build_last_token_logits())
    pkg.add_policy_component("token_to_slot", build_token_to_slot())

    temporal_inputs = {
        input_frame.name: "duplex.input_frame",
        temporal_mask.name: "state.attention_mask.body",
        temporal_position.name: "state.position_ids.body",
        **{
            past.name: f"state.temporal_cache_{index}.body"
            for index, (past, _) in enumerate(temporal_caches)
        },
    }
    temporal_outputs = {
        hidden.name: "duplex.hidden",
        text_logits.name: "duplex.text_logits",
        **{
            present.name: f"duplex.temporal_cache_{index}"
            for index, (_, present) in enumerate(temporal_caches)
        },
    }

    graph = {
        "kind": "sequence",
        "nodes": [],
    }
    frame_body = graph["nodes"]
    frame_body.extend(
        [
            # 1. packed audio in: grow the codec prefix and encode the newest frame.
            _invoke(
                "waveform_append",
                {"prefix": "state.user_waveform.body", "chunk": "request.audio_chunk"},
                {"next_prefix": "duplex.user_waveform_next"},
            ),
            _invoke(
                "encoder",
                {waveform_input.name: "duplex.user_waveform_next"},
                {codes_output.name: "duplex.user_codes"},
            ),
            _invoke(
                "codes_tail",
                {"prefix": "duplex.user_codes", "count": "package.one_frame"},
                {"tail": "duplex.user_frame_codes"},
            ),
            # 2. delay ring buffer: write every supplied stream, read one frame.
            _invoke(
                "user_stream_merge",
                {
                    "frame_codes": "package.stream_tokens",
                    "codes": "duplex.user_frame_codes",
                },
                {"stream_tokens": "duplex.stream_tokens"},
            ),
            _invoke(
                "frame_assemble",
                {
                    "token_cache": "state.token_cache.body",
                    "token_provided": "state.token_provided.body",
                    "offset": "state.offset.body",
                    "stream_tokens": "duplex.stream_tokens",
                    "delays": "package.delays",
                    "initial_tokens": "package.initial_tokens",
                },
                {
                    "next_token_cache": "duplex.cache_assembled",
                    "next_token_provided": "duplex.provided_assembled",
                    "input_frame": "duplex.input_frame",
                    "target": "duplex.target",
                    "target_provided": "duplex.target_provided",
                },
            ),
            # 3. frame-rate temporal transformer over the interleaved frame.
            _invoke("temporal", temporal_inputs, temporal_outputs),
            _invoke(
                "last_text_logits",
                {"logits": "duplex.text_logits"},
                {"last_logits": "duplex.text_last"},
            ),
            _invoke(
                "token_sampler",
                {"logits": "duplex.text_last"},
                {"token": "duplex.text_sampled"},
            ),
            _invoke(
                "teacher_select",
                {
                    "target": "duplex.target",
                    "target_provided": "duplex.target_provided",
                    "sampled": "duplex.text_sampled",
                    "index": "package.text_stream",
                },
                {"token": "duplex.text_token"},
            ),
            _invoke(
                "target_frame",
                {"target": "duplex.target"},
                {"frame": "duplex.target_frame"},
            ),
            # 4. acoustic loop: one substep per acoustic stream, KV reset per frame.
            acoustic_loop,
            _invoke(
                "text_frame_update",
                {
                    "frame_codes": "duplex.acoustic_frame_final",
                    "token": "duplex.text_token",
                    "index": "package.text_stream",
                },
                {"next_frame": "duplex.completed_frame"},
            ),
            # 5. commit the frame and undo the per-stream delays.
            _invoke(
                "frame_commit",
                {
                    "token_cache": "duplex.cache_assembled",
                    "token_provided": "duplex.provided_assembled",
                    "offset": "state.offset.body",
                    "frame": "duplex.completed_frame",
                    "delays": "package.delays",
                },
                {
                    "next_token_cache": "duplex.cache_committed",
                    "next_token_provided": "duplex.provided_committed",
                    "out_frame": "duplex.out_frame",
                    "next_offset": "duplex.next_offset",
                    "emit": "duplex.emit",
                },
            ),
            # 6. packed audio out: grow the agent code prefix and decode it.
            _invoke(
                "agent_frame_select",
                {"frame": "duplex.out_frame"},
                {"codes": "duplex.agent_frame"},
            ),
            _invoke(
                "codes_append",
                {"prefix": "state.agent_codes.body", "frame": "duplex.agent_frame"},
                {"next_prefix": "duplex.agent_codes_next"},
            ),
            _invoke(
                "decoder",
                {codes_input.name: "duplex.agent_codes_next"},
                {waveform_output.name: "duplex.agent_waveform"},
            ),
            _invoke(
                "chunk_tail",
                {"prefix": "duplex.agent_waveform", "count": "package.frame_size"},
                {"tail": "duplex.agent_chunk"},
            ),
            _invoke(
                "step_update",
                {
                    "attention_mask": "state.attention_mask.body",
                    "position_ids": "state.position_ids.body",
                },
                {
                    "next_attention_mask": "duplex.attention_mask_next",
                    "next_position_ids": "duplex.position_ids_next",
                },
            ),
            {
                "kind": "emit",
                "value": "duplex.agent_chunk",
                "output": "audio_chunk",
                "mode": "event",
                "when": "duplex.emit",
            },
        ]
    )
    # Session-resident cells are only readable inside a loop carry, so the whole
    # frame body is a loop. ``package.frames_per_invocation`` defaults to 1, which
    # makes one invocation exactly one duplex event; a runtime that hands several
    # codec frames to a single call raises it without changing the graph.
    session_carries = [
        # A duplex conversation has no generation-side termination predicate: it
        # runs until the session lease is released, so liveness is invariant.
        # These are carried first because the temporal cache recurrence quotes
        # ``accepted_len`` as its per-row growth increment.
        ("active", "package.true_batch", "package.true_batch"),
        ("done", "package.false_batch", "package.false_batch"),
        ("accepted_len", "package.zero_batch", "package.one_batch"),
        ("token_cache", "package.token_cache_init", "duplex.cache_committed"),
        ("token_provided", "package.token_provided_init", "duplex.provided_committed"),
        ("offset", "package.offset_init", "duplex.next_offset"),
        ("attention_mask", "package.attention_mask_init", "duplex.attention_mask_next"),
        ("position_ids", "package.position_ids_init", "duplex.position_ids_next"),
        ("user_waveform", "package.user_waveform_init", "duplex.user_waveform_next"),
        ("agent_codes", "package.agent_codes_init", "duplex.agent_codes_next"),
        *[
            (
                f"temporal_cache_{index}",
                f"package.temporal_cache_{index}_init",
                f"duplex.temporal_cache_{index}",
            )
            for index in range(len(temporal_caches))
        ],
        ("temporal_cache_lengths", "package.zero_batch", "duplex.temporal_cache_lengths"),
    ]
    frame_body.append(
        _invoke(
            "cache_length_update",
            {
                "left": "state.temporal_cache_lengths.body",
                "right": "package.one_batch",
            },
            {"total": "duplex.temporal_cache_lengths"},
        )
    )
    pkg.add_policy_component("cache_length_update", build_integer_add())
    frame_loop = {
        "kind": "loop",
        "setup": {"kind": "sequence", "nodes": []},
        "body": {"kind": "sequence", "nodes": frame_body},
        "condition": "package.loop_active",
        "max_iterations": "package.frames_per_invocation",
        "iteration": {"value": "duplex.frame_index", "contract": batch_int},
        "carried": [
            {
                "cell": cell,
                "current": current,
                "body_input": f"state.{cell}.body",
                "body_output": produced,
                "next": f"duplex.{cell}_final",
            }
            for cell, current, produced in session_carries
        ],
    }
    graph = {"kind": "sequence", "nodes": [frame_loop]}

    pkg.add_policy_component(
        "user_stream_merge",
        build_duplex_user_stream_merge(channels=channels, streams=audio_streams),
    )
    pkg.add_policy_component("target_frame", build_duplex_cell_to_frame(channels=channels))
    pkg.add_policy_component(
        "agent_frame_select",
        build_duplex_agent_frame_select(channels=channels, streams=audio_streams),
    )
    pkg.add_policy_component(
        "text_frame_update", build_code_frame_update(channels, scalar_index=True)
    )
    pkg.add_policy_component(
        "step_update",
        build_decoder_step_update(
            attention_dtype=_ir_dtype(temporal_mask),
            position_dtype=_ir_dtype(temporal_position),
        ),
    )
    inputs["package.text_stream"] = {
        "contract": control_int,
        "role": {"kind": "opaque"},
        "source": {"kind": "literal"},
        "required": False,
        "default": 0,
    }

    workflow = {
        "manifest": {
            "capabilities": [
                "workflow_ssa",
                "linear_effects",
                "typed_emit",
                "streaming_emit",
                "nested_control_flow",
                "loop_induction_values",
                "bounded_state_recurrence",
                "serving_service_contract",
                "session_state_lease",
            ],
        },
        "inputs": inputs,
        "outputs": {
            "audio_chunk": {
                "contract": {
                    **_contract(waveform_output),
                    "batch_layout": request_aligned,
                },
                "role": "audio",
                "stage": "post_adapter",
            }
        },
        "components": {
            "encoder": _component(encoder, "encoder/model.onnx"),
            "decoder": _component(decoder, "decoder/model.onnx"),
            "temporal": _component(temporal, "temporal/model.onnx"),
            "depformer": _component(depformer, "depformer/model.onnx"),
        },
        "state": state,
        "graph": graph,
    }
    metadata = {
        "schema_version": "v1",
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }
    add_policy_components_to_workflow(metadata, pkg)
    _annotate_duplex_state_service(metadata, config, temporal_caches)
    return metadata


def _annotate_duplex_state_service(
    metadata: dict[str, Any],
    config: Any,
    temporal_caches: list[tuple[ir.Value, ir.Value]],
) -> None:
    """Publish the semantic contract of the session-resident temporal KV group.

    A full-duplex conversation has no termination predicate: it runs until the
    session is released. The serving contract therefore points ``active``/``done``
    at the session-scoped liveness cells rather than at a generation-loop flag.
    """
    workflow = metadata["pipeline"]["workflow"]
    context = int(getattr(config, "context", 0))
    workflow["serving"] = {
        "active": "active",
        "done": "done",
        "accepted_len": "accepted_len",
        "state_service": {
            "groups": {
                "temporal_cache": {
                    "kind": "sliding_attention" if context else "full_attention",
                    "sequence_axis": 2,
                    "layout": "bnsh",
                    "logical_lengths": "temporal_cache_lengths",
                    "aliasing": "permitted",
                    "reuse": {"prefix_reusable": True, "evictable_prefix": False},
                    "capabilities": {"snapshot": True, "fork": False},
                    "ports": {
                        "temporal": {
                            f"temporal_cache_{index}": _annotated_alias(
                                {"input": past.name, "output": present.name}
                            )
                            for index, (past, present) in enumerate(temporal_caches)
                        }
                    },
                }
            }
        },
    }


def write_full_duplex_workflow_metadata(pkg: Any, config: Any, output_dir: str) -> str:
    """Write full-duplex workflow metadata and its policy artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_full_duplex_workflow_metadata(pkg, config)
    pkg.save_policy_components(output_dir)
    add_adapter_service_to_metadata(metadata, pkg, output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path
