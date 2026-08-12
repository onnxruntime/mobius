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

    inputs = list(decoder.graph.inputs)
    outputs = list(decoder.graph.outputs)
    token_input = next(
        (
            value
            for value in inputs
            if value.dtype in {ir.DataType.INT32, ir.DataType.INT64}
            and value.shape is not None
            and len(value.shape) == 2
        ),
        None,
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

    workflow_inputs: dict[str, Any] = {}
    setup_decoder_inputs: dict[str, str] = {}
    body_decoder_inputs: dict[str, str] = {}
    for value in inputs:
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
                "source": {"kind": "application", "name": "eos_token_ids"},
                "required": True,
            },
            "loop.iteration": {
                "contract": batch_int,
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": "iteration"},
                "required": True,
            },
            "loop.token_slot": {
                "contract": {
                    "dtype": "int64",
                    "rank": 2,
                    "shape": [batch_dimension, 1],
                },
                "role": {"kind": "opaque"},
                "source": {"kind": "application", "name": "token_slot"},
                "required": True,
            },
        }
    )

    cache_pairs: list[tuple[ir.Value, ir.Value]] = []
    output_by_suffix = {value.name: value for value in outputs}
    for value in inputs:
        if value is token_input:
            continue
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
            body_decoder_inputs[value.name] = f"state.{value.name}.body"
    body_decoder_inputs[token_input.name] = "state.token.body"

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
            "initializer": f"request.{token_input.name}",
            "recurrence": {"kind": "invariant"},
        }
    }
    initial_effects = {
        "sample": "sample.0",
        "termination": "termination.0",
        "state": "state.0",
        "emit": "emit.0",
        "state:token": "state:token.0",
    }
    carried = [
        {
            "cell": "token",
            "current": "token.setup",
            "body_input": "state.token.body",
            "body_output": "token.body",
            "next": "token.final",
            "read_effect": _effect("state:token.0", "state:token.read"),
            "write_effect": _effect("state:token.read", "state:token.1"),
        }
    ]
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
            "recurrence": {"kind": "invariant"},
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
            _invoke(decoder_name, setup_decoder_inputs, setup_decoder_outputs),
            _invoke(
                "token_sampler",
                {"logits": "decoder.setup.logits"},
                {"token": "sample.setup"},
                {"sample": _effect("sample.0", "sample.1")},
            ),
            _invoke(
                "token_state_update",
                {
                    "current": "loop.token_slot",
                    "update": "sample.setup",
                },
                {"next": "token.setup"},
                {"state": _effect("state.0", "state.1")},
            ),
        ],
    }
    body = {
        "kind": "sequence",
        "nodes": [
            _invoke(decoder_name, body_decoder_inputs, body_decoder_outputs),
            _invoke(
                "token_sampler",
                {"logits": "decoder.body.logits"},
                {"token": "sample.body"},
                {"sample": _effect("sample.1", "sample.2")},
            ),
            _invoke(
                "token_state_update",
                {"current": "state.token.body", "update": "sample.body"},
                {"next": "token.body"},
                {"state": _effect("state.1", "state.2")},
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
            {
                "kind": "emit",
                "value": "sample.body",
                "output": "tokens",
                "mode": "append",
                "effect_name": "emit",
                "effect": _effect("emit.0", "emit.1"),
            },
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
            "condition": "loop.done",
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
        "loop.iteration": {
            "contract": batch_int,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "iteration"},
            "required": True,
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
        offset: str,
        prefix: str,
        effect_in: str,
        effect_out: str,
    ) -> dict[str, Any]:
        return _invoke(
            "masked_update",
            {
                "current_tokens": tokens,
                "proposed_tokens": f"{prefix}.proposal",
                "masked": mask,
                "step": "loop.iteration",
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
        "nodes": [
            denoiser_invoke("request.input_ids", "denoiser.setup"),
            update_invoke(
                "request.input_ids",
                "request.mask",
                "request.rng_offset",
                "denoiser.setup",
                "update.0",
                "update.1",
            ),
        ],
    }
    body = {
        "kind": "sequence",
        "nodes": [
            denoiser_invoke("state.tokens.body", "denoiser.body"),
            update_invoke(
                "state.tokens.body",
                "state.mask.body",
                "state.rng_offset.body",
                "denoiser.body",
                "update.1",
                "update.2",
            ),
            {
                "kind": "emit",
                "value": "denoiser.body.tokens",
                "output": "tokens",
                "mode": "replace",
                "effect_name": "emit",
                "effect": _effect("emit.0", "emit.1"),
            },
        ],
    }

    state_specs = {
        "tokens": (token_contract, "request.input_ids", "denoiser.setup.tokens"),
        "mask": (mask_contract, "request.mask", "denoiser.setup.mask"),
        "rng_offset": (batch_int, "request.rng_offset", "denoiser.setup.rng_offset"),
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
            "condition": "denoiser.body.done",
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
