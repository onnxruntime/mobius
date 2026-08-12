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
    build_boolean_not,
    build_decoder_state_initializer,
    build_decoder_step_update,
    build_integer_increment,
    build_last_token_logits,
    build_model_token_cast,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    _port,
    _shape_metadata,
    add_policy_components_to_workflow,
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
    component = {
        "implementation": {"kind": "onnx", "artifact": artifact},
        "ports": {
            "inputs": {value.name: _contract(value) for value in model.graph.inputs},
            "outputs": {value.name: _contract(value) for value in model.graph.outputs},
        },
    }
    if effects:
        component["effects"] = list(effects)
    return component


def _effect(consumes: str, produces: str) -> dict[str, str]:
    return {"consumes": consumes, "produces": produces}


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
    return {"schema_version": "v1", "pipeline": {"workflow": workflow}}


def write_audio_codec_workflow_metadata(pkg: Any, output_dir: str) -> str:
    """Write typed SSA metadata for an audio codec package."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_audio_codec_workflow_metadata(pkg)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def build_tts_workflow_metadata(pkg: Any, config: Any) -> dict[str, Any]:
    """Reject TTS until the generic nested-loop contract exposes induction SSA.

    Qwen3-TTS needs the inner loop index as the code predictor ``step_index`` and
    to select the next code embedding. The workflow ``loop`` node at producer
    commit 4c3c4b6 only accepts a condition and maximum value; it defines no
    iteration SSA value. Consequently the existing prefill and per-frame
    embedder artifacts can be invoked, but the code-predictor loop cannot be
    expressed without host preprocessing or a model-specific counter component.
    """
    required = {
        "talker",
        "code_predictor",
        "talker_step_embedder",
    }
    missing = sorted(required.difference(pkg.keys()))
    if missing:
        raise ValueError(f"TTS workflow is missing required components: {missing}")
    del config
    raise NotImplementedError(
        "generic TTS workflow requires a nested-loop induction SSA value: "
        "ONNX GenAI workflow Loop at producer commit 4c3c4b6 exposes neither an "
        "iteration output nor fixed-loop index, so code_predictor.step_index, "
        "position_ids, and per-group code embedding selection cannot be wired "
        "from the existing prefill/step embedder artifacts without host "
        "preprocessing"
    )


def write_tts_workflow_metadata(pkg: Any, output_dir: str, config: Any) -> str:
    """Build TTS workflow metadata, failing precisely on the producer defect."""
    metadata = build_tts_workflow_metadata(pkg, config)
    os.makedirs(output_dir, exist_ok=True)
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
            "contract": _contract(logits_output),
            "scope": "invocation",
            "initializer": "decoder.setup.logits",
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
                "current": "decoder.setup.logits",
                "body_input": "state.logits.body",
                "body_output": "decoder.body.logits",
                "next": "state.logits.final",
                "read_effect": _effect("state:logits.0", "state:logits.read"),
                "write_effect": _effect("state:logits.read", "state:logits.1"),
            },
        ]
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
        ],
    }
    body = {
        "kind": "sequence",
        "nodes": [
            _invoke(
                "last_token_logits",
                {"logits": "state.logits.body"},
                {"last_logits": "decoder.body.last_logits"},
            ),
            _invoke(
                "token_sampler",
                {"logits": "decoder.body.last_logits"},
                {"token": "sample.body"},
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
    metadata = {"schema_version": "1.0", "pipeline": {"workflow": workflow}}
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
    metadata = {"schema_version": "1.0", "pipeline": {"workflow": workflow}}
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def write_decoder_workflow_metadata(
    pkg: Any,
    output_dir: str,
    config: Any,
) -> str:
    """Write decoder workflow metadata and policy artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_decoder_workflow_metadata(pkg, config)
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
