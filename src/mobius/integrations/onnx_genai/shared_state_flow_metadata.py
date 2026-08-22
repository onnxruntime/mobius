# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generic workflow metadata for pixel-flow generation over a frozen shared prefix."""

from __future__ import annotations

import os
from typing import Any

import onnx_ir as ir
import yaml

from mobius.generation import (
    PolicyCapabilities,
    attach_policy_components,
    build_boolean_not,
    build_decoder_state_initializer,
    build_empty_batched_features,
    build_euler_solver_step,
    build_guidance_combine,
    build_image_grid_positions,
    build_schedule_constant,
    build_schedule_lookup,
    build_tensor_cast,
    build_x0_flow_velocity,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    add_policy_components_to_workflow,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    _component,
    _contract,
    _effect,
    _invoke,
    _model_cache_pairs,
    _NoAliasSafeDumper,
    _publish_workflow_v1,
)


def _find_component(pkg: Any, *, inputs: set[str], outputs: set[str]) -> str:
    matches = []
    for name, model in pkg.items():
        graph_inputs = {value.name for value in model.graph.inputs}
        graph_outputs = {value.name for value in model.graph.outputs}
        if inputs <= graph_inputs and outputs <= graph_outputs:
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(
            "shared-prefix pixel-flow workflow requires one structural component "
            f"with inputs {sorted(inputs)} and outputs {sorted(outputs)}, got {matches}"
        )
    return matches[0]


def is_shared_state_pixel_flow_package(pkg: Any) -> bool:
    """Return whether *pkg* exposes the five generic component roles."""
    try:
        _find_component(pkg, inputs={"pixel_values"}, outputs={"image_features"})
        _find_component(
            pkg,
            inputs={"input_ids", "image_features", "image_mask"},
            outputs={"inputs_embeds"},
        )
        _find_component(pkg, inputs={"inputs_embeds"}, outputs={"logits"})
        _find_component(
            pkg,
            inputs={"latent", "timestep", "noise_scale"},
            outputs={"image_embeds"},
        )
        _find_component(
            pkg,
            inputs={"image_embeds", "position_ids", "token_grid"},
            outputs={"predicted_image"},
        )
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _image_preprocessing(vision_input: ir.Value) -> dict[str, Any]:
    """Exact upstream reference-image program for a plain RGB tensor."""
    contract = _contract(vision_input)
    # Component artifact symbols are local to that graph. The workflow also
    # carries an independently sized generated image, so publishing the generic
    # artifact names ``height``/``width`` here would incorrectly tie the source
    # image resize to the output latent geometry.
    contract["shape"] = [1, 3, "source_image_height", "source_image_width"]
    return {
        "image": {
            "transforms": [
                {"op": "decode_rgb", "outputs": ["image.decoded"]},
                {
                    "op": "resize",
                    "inputs": ["image.decoded"],
                    "outputs": ["image.resized"],
                    "mode": "pixel_area",
                    "interpolation": "lanczos3",
                    "min_pixels": 512 * 512,
                    "max_pixels": 2048 * 2048,
                    "size_multiple": 32,
                },
                {
                    "op": "rescale",
                    "inputs": ["image.resized"],
                    "outputs": ["image.rescaled"],
                    "scale": 1.0 / 255.0,
                },
                {
                    "op": "normalize",
                    "inputs": ["image.rescaled"],
                    "outputs": ["image.normalized"],
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ],
            "outputs": [
                {
                    "source": "image.normalized",
                    "name": "image.pixel_values",
                    "content": "pixels",
                    "dtype": contract["dtype"],
                    "contract": contract,
                }
            ],
        }
    }


def build_shared_state_pixel_flow_workflow_metadata(
    pkg: Any,
    config: Any,
    *,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,
    t_eps: float | None = None,
) -> dict[str, Any]:
    """Build an alternating-weight workflow over a frozen attention-state group."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    if not is_shared_state_pixel_flow_package(pkg):
        raise ValueError("package does not expose the shared-state pixel-flow component ABI")

    vision_name = _find_component(pkg, inputs={"pixel_values"}, outputs={"image_features"})
    embedding_name = _find_component(
        pkg,
        inputs={"input_ids", "image_features", "image_mask"},
        outputs={"inputs_embeds"},
    )
    decoder_name = _find_component(pkg, inputs={"inputs_embeds"}, outputs={"logits"})
    generation_embedding_name = _find_component(
        pkg,
        inputs={"latent", "timestep", "noise_scale"},
        outputs={"image_embeds"},
    )
    denoiser_name = _find_component(
        pkg,
        inputs={"image_embeds", "position_ids", "token_grid"},
        outputs={"predicted_image"},
    )

    vision = pkg[vision_name]
    embedding = pkg[embedding_name]
    decoder = pkg[decoder_name]
    generation_embedding = pkg[generation_embedding_name]
    denoiser = pkg[denoiser_name]
    decoder_pairs = _model_cache_pairs(decoder)
    denoiser_pairs = _model_cache_pairs(denoiser)
    if len(decoder_pairs) != len(denoiser_pairs) or not decoder_pairs:
        raise ValueError(
            "understanding and generation components must expose matching KV pairs"
        )

    prompt = next(value for value in embedding.graph.inputs if value.name == "input_ids")
    image_features = next(
        value for value in embedding.graph.inputs if value.name == "image_features"
    )
    image_mask = next(value for value in embedding.graph.inputs if value.name == "image_mask")
    inputs_embeds = next(
        value for value in embedding.graph.outputs if value.name == "inputs_embeds"
    )
    vision_input = next(value for value in vision.graph.inputs if value.name == "pixel_values")
    vision_output = next(
        value for value in vision.graph.outputs if value.name == "image_features"
    )
    decoder_outputs = {value.name: value for value in decoder.graph.outputs}
    latent = next(
        value for value in generation_embedding.graph.inputs if value.name == "latent"
    )
    timestep = next(
        value for value in generation_embedding.graph.inputs if value.name == "timestep"
    )
    noise_scale = next(
        value for value in generation_embedding.graph.inputs if value.name == "noise_scale"
    )
    image_embeds = next(
        value for value in generation_embedding.graph.outputs if value.name == "image_embeds"
    )
    predicted_image = next(
        value for value in denoiser.graph.outputs if value.name == "predicted_image"
    )
    generation_dtype = image_embeds.dtype

    attach_policy_components(pkg, PolicyCapabilities())
    cache_input_names = [past.name for past, _ in decoder_pairs]
    pkg.add_policy_component(
        "prefix_initializer",
        build_decoder_state_initializer(
            decoder,
            token_input=None,
            prompt_dtype=prompt.dtype,
            attention_mask_input="attention_mask",
            position_ids_input="position_ids",
            cache_inputs=cache_input_names,
        ),
    )
    pkg.add_policy_component(
        "empty_image_features",
        build_empty_batched_features(image_features.dtype, int(image_features.shape[-1])),
    )
    pkg.add_policy_component(
        "image_grid_positions",
        build_image_grid_positions(
            generation_dtype,
            pixels_per_token=int(getattr(config, "pixels_per_token", 32)),
        ),
    )
    schedule = [1.0 - index / num_inference_steps for index in range(num_inference_steps + 1)]
    pkg.add_policy_component("flow_schedule", build_schedule_constant(schedule))
    pkg.add_policy_component("schedule_lookup", build_schedule_lookup(timestep.dtype))
    pkg.add_policy_component("guidance_combine", build_guidance_combine(generation_dtype))
    pkg.add_policy_component(
        "x0_velocity",
        build_x0_flow_velocity(
            generation_dtype,
            t_eps=float(t_eps if t_eps is not None else getattr(config, "t_eps", 0.02)),
        ),
    )
    pkg.add_policy_component("solver_step", build_euler_solver_step(generation_dtype))
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    for index, ((decoder_past, _), (generation_past, _)) in enumerate(
        zip(decoder_pairs, denoiser_pairs)
    ):
        pkg.add_policy_component(
            f"cache_cast_{index}",
            build_tensor_cast(decoder_past.dtype, generation_past.dtype),
        )
        pkg.add_policy_component(
            f"cache_freeze_{index}",
            build_tensor_cast(generation_past.dtype, generation_past.dtype),
        )

    components = {
        name: _component(model, f"{name}/model.onnx")
        for name, model in pkg.items()
        if name
        in {
            vision_name,
            embedding_name,
            decoder_name,
            generation_embedding_name,
            denoiser_name,
        }
    }
    components["image_preprocess"] = {
        "implementation": {
            "kind": "adapter",
            "abi": "onnx-genai.image-preprocess",
            "version": "1",
        },
        "ports": {
            "inputs": {"encoded": {"dtype": "uint8", "rank": 1, "shape": ["encoded_bytes"]}},
            "outputs": {
                "pixel_values": {
                    **_contract(vision_input),
                    "shape": [1, 3, "source_image_height", "source_image_width"],
                }
            },
        },
        "contract": {
            "id": "onnx-genai.image-preprocess",
            "version": "1",
            "bindings": {"encoded": "encoded", "pixel_values": "pixel_values"},
        },
    }

    batch_bool = {
        "dtype": "bool",
        "rank": 1,
        "shape": ["batch"],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    }
    inputs: dict[str, Any] = {
        "request.prompt_tokens": {
            "contract": _contract(prompt),
            "role": {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"},
            "source": {"kind": "request"},
        },
        "request.negative_prompt_tokens": {
            "contract": _contract(prompt),
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "negative_prompt_tokens"},
        },
        "request.image": {
            "contract": {"dtype": "uint8", "rank": 1, "shape": ["encoded_bytes"]},
            "role": {"kind": "runtime", "version": "1.0", "role": "media"},
            "source": {"kind": "request"},
            "required": False,
            "present_as": "request.image_present",
        },
        "request.image_mask": {
            "contract": _contract(image_mask),
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "image_mask"},
            "required": False,
            "default": False,
        },
        "request.negative_image_mask": {
            "contract": _contract(image_mask),
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "negative_image_mask"},
            "required": False,
            "default": False,
        },
        "request.latent": {
            "contract": _contract(latent),
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "latent"},
            "externally_suppliable": True,
        },
        "request.noise_scale": {
            "contract": _contract(noise_scale),
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "noise_scale"},
            "required": False,
            "default": 1.0,
        },
        "request.guidance_scale": {
            "contract": {
                "dtype": "float32",
                "rank": 1,
                "shape": ["batch"],
                "batch_layout": {"kind": "request_aligned", "axis": 0},
            },
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "guidance_scale"},
            "required": False,
            "default": float(guidance_scale),
        },
        "request.max_iterations": {
            "contract": {"dtype": "int64", "rank": 1, "shape": [1]},
            "role": {"kind": "runtime", "version": "1.0", "role": "max_iterations"},
            "source": {"kind": "request"},
            "required": False,
            "default": num_inference_steps,
        },
        "request.text_only": {
            "contract": {"dtype": "bool", "rank": 1, "shape": [1]},
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "text_only"},
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
        "package.false": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": False,
        },
        "package.accepted_len": {
            "contract": {
                "dtype": "int64",
                "rank": 1,
                "shape": ["batch"],
                "batch_layout": {"kind": "request_aligned", "axis": 0},
            },
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": 1,
        },
    }
    outputs = {
        "logits": {
            "contract": (
                _contract(decoder_outputs["logits"])
                if decoder_outputs["logits"].shape is not None
                else {
                    "dtype": _contract(decoder_outputs["logits"])["dtype"],
                    "rank": 3,
                    "shape": ["batch", "sequence_len", int(config.vocab_size)],
                    "batch_layout": {"kind": "request_aligned", "axis": 0},
                }
            ),
            "role": "tensor",
            "stage": "pre_adapter",
        },
        "image": {
            "contract": _contract(predicted_image),
            "role": "image",
            "value_range": "negative_one_to_one",
            "stage": "pre_adapter",
        },
    }

    setup: list[dict[str, Any]] = [
        _invoke("flow_schedule", {}, {"schedule": "flow.schedule"}),
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
                            {"pixel_values": "image.pixel_values"},
                        ),
                        _invoke(
                            vision_name,
                            {vision_input.name: "image.pixel_values"},
                            {vision_output.name: "image.features.present"},
                        ),
                    ],
                },
                "false": _invoke(
                    "empty_image_features",
                    {},
                    {"features": "image.features.empty"},
                ),
            },
            "outputs": {
                "image.features": {
                    "cases": {
                        "true": "image.features.present",
                        "false": "image.features.empty",
                    }
                }
            },
        },
        _invoke(
            "empty_image_features",
            {},
            {"features": "image.features.unconditional"},
        ),
        _invoke(
            embedding_name,
            {
                "input_ids": "request.prompt_tokens",
                "image_features": "image.features",
                "image_mask": "request.image_mask",
            },
            {inputs_embeds.name: "prefix.conditional.embeds"},
        ),
        _invoke(
            embedding_name,
            {
                "input_ids": "request.negative_prompt_tokens",
                "image_features": "image.features.unconditional",
                "image_mask": "request.negative_image_mask",
            },
            {inputs_embeds.name: "prefix.unconditional.embeds"},
        ),
    ]

    state: dict[str, Any] = {}
    carried: list[dict[str, Any]] = []
    decoder_state_outputs: dict[str, str] = {
        "attention_mask": "prefix.attention_mask",
        "position_ids": "prefix.position_ids",
        "body_attention_mask": "prefix.body_attention_mask",
        "body_position_ids": "prefix.body_position_ids",
        "token_slot": "prefix.token_slot",
        **{name: f"prefix.empty.{name}" for name in cache_input_names},
    }
    setup.append(
        _invoke(
            "prefix_initializer",
            {"prompt_tokens": "request.prompt_tokens"},
            decoder_state_outputs,
        )
    )
    for branch, embeds in (
        ("conditional", "prefix.conditional.embeds"),
        ("unconditional", "prefix.unconditional.embeds"),
    ):
        decoder_call_inputs = {
            "inputs_embeds": embeds,
            "attention_mask": "prefix.attention_mask",
            "position_ids": "prefix.position_ids",
            **{name: f"prefix.empty.{name}" for name in cache_input_names},
        }
        decoder_call_outputs = {
            "logits": f"prefix.{branch}.logits",
            **{
                present.name: f"prefix.{branch}.{present.name}" for _, present in decoder_pairs
            },
        }
        setup.append(_invoke(decoder_name, decoder_call_inputs, decoder_call_outputs))
        for index, ((_, present), (generation_past, _)) in enumerate(
            zip(decoder_pairs, denoiser_pairs)
        ):
            initial = f"prefix.{branch}.cache.{index}"
            setup.append(
                _invoke(
                    f"cache_cast_{index}",
                    {"value": f"prefix.{branch}.{present.name}"},
                    {"cast": initial},
                )
            )
            cell = f"{branch}_cache_{index}"
            state[cell] = {
                "contract": _contract(generation_past),
                "scope": "invocation",
                "initializer": initial,
                "recurrence": {"kind": "invariant"},
                "management": "runtime",
                "release_boundary": "invocation",
                "service_group": f"{branch}_prefix",
            }
            carried.append(
                {
                    "cell": cell,
                    "current": initial,
                    "body_input": cell,
                    "body_output": f"{cell}.next",
                    "next": f"{cell}.final",
                }
            )

    body = [
        _invoke(
            "schedule_lookup",
            {"schedule": "flow.schedule", "step": "loop.iteration"},
            {"timestep": "flow.timestep"},
        ),
        _invoke(
            generation_embedding_name,
            {
                "latent": "latent",
                "timestep": "flow.timestep",
                "noise_scale": "request.noise_scale",
            },
            {"image_embeds": "flow.image_embeds"},
        ),
        _invoke(
            "image_grid_positions",
            {
                "latent": "latent",
                "prompt_tokens": "request.prompt_tokens",
            },
            {
                "position_ids": "flow.position_ids",
                "token_grid": "flow.token_grid",
            },
        ),
    ]
    for branch in ("conditional", "unconditional"):
        denoiser_inputs = {
            "image_embeds": "flow.image_embeds",
            "position_ids": "flow.position_ids",
            "token_grid": "flow.token_grid",
        }
        denoiser_outputs = {"predicted_image": f"flow.{branch}.x0"}
        for index, (generation_past, generation_present) in enumerate(denoiser_pairs):
            denoiser_inputs[generation_past.name] = f"{branch}_cache_{index}"
            denoiser_outputs[generation_present.name] = f"flow.discard.{branch}.{index}"
            body.append(
                _invoke(
                    f"cache_freeze_{index}",
                    {"value": f"{branch}_cache_{index}"},
                    {"cast": f"{branch}_cache_{index}.next"},
                )
            )
        body.append(_invoke(denoiser_name, denoiser_inputs, denoiser_outputs))
    body.extend(
        [
            _invoke(
                "guidance_combine",
                {
                    "unconditional": "flow.unconditional.x0",
                    "conditional": "flow.conditional.x0",
                    "scale": "request.guidance_scale",
                },
                {"estimate": "flow.x0"},
            ),
            _invoke(
                "x0_velocity",
                {
                    "sample": "latent",
                    "x0": "flow.x0",
                    "timestep": "flow.timestep",
                },
                {"velocity": "flow.velocity"},
            ),
            _invoke(
                "solver_step",
                {
                    "sample": "latent",
                    "derivative": "flow.velocity",
                    "step": "loop.iteration",
                    "schedule": "flow.schedule",
                },
                {"next_state": "latent.next"},
            ),
        ]
    )
    state["latent"] = {
        "contract": _contract(latent),
        "scope": "invocation",
        "initializer": "request.latent",
        "recurrence": {"kind": "invariant"},
    }
    carried.insert(
        0,
        {
            "cell": "latent",
            "current": "request.latent",
            "body_input": "latent",
            "body_output": "latent.next",
            "next": "latent.final",
        },
    )

    group_ports: dict[str, dict[str, Any]] = {
        branch: {decoder_name: {}, denoiser_name: {}}
        for branch in ("conditional", "unconditional")
    }
    for index, (
        (decoder_past, decoder_present),
        (generation_past, generation_present),
    ) in enumerate(zip(decoder_pairs, denoiser_pairs)):
        role = "key" if decoder_past.name.endswith(".key") else "value"
        for branch in ("conditional", "unconditional"):
            alias = f"{branch}_cache_{index}"
            group_ports[branch][decoder_name][alias] = {
                "input": decoder_past.name,
                "output": decoder_present.name,
                "role": role,
                "layer": index // 2,
            }
            group_ports[branch][denoiser_name][alias] = {
                "input": generation_past.name,
                "output": generation_present.name,
                "role": role,
                "layer": index // 2,
            }

    workflow = {
        "manifest": {
            "adapter_abis": {"onnx-genai.image-preprocess": "1"},
            "capabilities": [
                "workflow_ssa",
                "typed_emit",
                "nested_control_flow",
                "loop_induction_values",
                "serving_service_contract",
                "input_presence",
            ],
        },
        "inputs": inputs,
        "outputs": outputs,
        "components": components,
        "state": state,
        "initial_effects": {"emit": "emit.0"},
        "graph": {
            "kind": "sequence",
            "nodes": [
                *setup,
                {
                    "kind": "branch",
                    "predicate": "request.text_only",
                    "cases": {
                        "true": {
                            "kind": "emit",
                            "value": "prefix.conditional.logits",
                            "output": "logits",
                            "mode": "replace",
                            "effect_name": "emit",
                            "effect": _effect("emit.0", "emit.1"),
                        },
                        "false": {
                            "kind": "sequence",
                            "nodes": [
                                {
                                    "kind": "loop",
                                    "setup": {"kind": "sequence", "nodes": []},
                                    "body": {"kind": "sequence", "nodes": body},
                                    "condition": "package.active",
                                    "max_iterations": "request.max_iterations",
                                    "iteration": {
                                        "value": "loop.iteration",
                                        "contract": {
                                            "dtype": "int64",
                                            "rank": 1,
                                            "shape": [1],
                                        },
                                    },
                                    "carried": carried,
                                },
                                {
                                    "kind": "emit",
                                    "value": "latent",
                                    "output": "image",
                                    "mode": "replace",
                                    "effect_name": "emit",
                                    "effect": _effect("emit.0", "emit.1"),
                                },
                            ],
                        },
                    },
                },
            ],
        },
        "serving": {
            "active": "package.active",
            "done": "package.false",
            "accepted_len": "package.accepted_len",
            "state_service": {
                "groups": {
                    f"{branch}_prefix": {
                        "kind": "full_attention",
                        "sequence_axis": 2,
                        "layout": "bnsh",
                        "update": {"kind": "append"},
                        "ports": group_ports[branch],
                    }
                    for branch in ("conditional", "unconditional")
                }
            },
        },
    }
    metadata = {
        "schema_version": "v1",
        "preprocessing": _image_preprocessing(vision_input),
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def write_shared_state_pixel_flow_workflow_metadata(
    pkg: Any,
    config: Any,
    output_dir: str,
    **kwargs: Any,
) -> str:
    """Write metadata and attached policy graphs for the generic workflow."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_shared_state_pixel_flow_workflow_metadata(pkg, config, **kwargs)
    pkg.save_policy_components(output_dir)
    path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(metadata, handle, Dumper=_NoAliasSafeDumper, sort_keys=False)
    return path
