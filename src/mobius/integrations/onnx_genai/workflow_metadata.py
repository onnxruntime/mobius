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
    build_code_frame_update,
    build_code_history_append,
    build_codec_layout_transpose,
    build_decoder_state_initializer,
    build_decoder_step_update,
    build_integer_increment,
    build_iteration_cast,
    build_last_token_logits,
    build_model_token_cast,
    build_schedule_constant,
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
    """Build nested talker/code-predictor loops with lexical induction SSA."""
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
    pkg.add_policy_component("code_frame_update", build_code_frame_update(num_groups))
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
                    {predictor_logits.name: "code.logits"},
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
        "iteration": {"value": "code.iteration", "contract": batch_int},
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
                {step_output.name: "talker.step_embeds"},
            ),
            _invoke(
                "talker",
                talker_body_inputs,
                {talker_hidden.name: "talker.body.hidden"},
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
    setup_outputs = {prefill_output.name: "talker.prefill_embeds"}
    if talker_hidden.name in {value.name for value in talker.graph.outputs}:
        talker_setup_outputs = {talker_hidden.name: "talker.prefill.hidden"}
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
                {waveform_output.name: "tts.waveform"},
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
    metadata = {"schema_version": "v1", "pipeline": {"workflow": workflow}}
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def write_tts_workflow_metadata(pkg: Any, output_dir: str, config: Any) -> str:
    """Build TTS workflow metadata, failing precisely on the producer defect."""
    metadata = build_tts_workflow_metadata(pkg, config)
    os.makedirs(output_dir, exist_ok=True)
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


def build_diffusion_workflow_metadata(
    pkg: Any,
    *,
    num_inference_steps: int,
    schedule: list[float] | None = None,
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

    attach_policy_components(pkg, PolicyCapabilities(solver="euler"))
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    schedule_values = schedule or [
        1.0 - index / num_inference_steps for index in range(num_inference_steps + 1)
    ]
    pkg.add_policy_component("diffusion_schedule", build_schedule_constant(schedule_values))
    if timestep_input.dtype != ir.DataType.INT64:
        pkg.add_policy_component("iteration_cast", build_iteration_cast(timestep_input.dtype))

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
        _invoke("diffusion_schedule", {}, {"schedule": "diffusion.schedule"})
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
        sample_input.name: "state.latent.body",
        timestep_input.name: (
            "diffusion.timestep"
            if timestep_input.dtype != ir.DataType.INT64
            else "loop.iteration"
        ),
    }
    if conditioning_input is not None and conditioning_value is not None:
        denoiser_inputs[conditioning_input.name] = conditioning_value
    body_nodes: list[dict[str, Any]] = []
    if timestep_input.dtype != ir.DataType.INT64:
        body_nodes.append(
            _invoke(
                "iteration_cast",
                {"iteration": "loop.iteration"},
                {"timestep": "diffusion.timestep"},
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
    metadata = {"schema_version": "v1", "pipeline": {"workflow": workflow}}
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def write_diffusion_workflow_metadata(
    pkg: Any,
    output_dir: str,
    *,
    num_inference_steps: int,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_diffusion_workflow_metadata(pkg, num_inference_steps=num_inference_steps)
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
                and _contract(value) == _contract(embedding_output)
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
            cache_pairs.append((value, present))
    cache_names = {value.name for value, _ in cache_pairs}
    rank2_integer = [
        value
        for value in decoder.graph.inputs
        if value.name not in cache_names
        and value.dtype in {ir.DataType.INT32, ir.DataType.INT64}
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
        output["source"] = output["content"]
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
    state: dict[str, Any] = {
        "token": {
            "contract": {"dtype": "int64", "rank": 2, "shape": [batch, 1]},
            "scope": "invocation",
            "initializer": "initializer.token_slot",
            "recurrence": {"kind": "invariant"},
        },
        "logits": {
            "contract": _contract(logits_output),
            "scope": "invocation",
            "initializer": "decoder.setup.logits",
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
            "decoder.setup.logits",
            "state.logits.body",
            "decoder.body.logits",
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
        "pipeline": {"workflow": workflow},
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
