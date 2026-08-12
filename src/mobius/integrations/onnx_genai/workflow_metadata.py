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


def _component(model: ir.Model, artifact: str) -> dict[str, Any]:
    return {
        "implementation": {"kind": "onnx", "artifact": artifact},
        "ports": {
            "inputs": {value.name: _contract(value) for value in model.graph.inputs},
            "outputs": {value.name: _contract(value) for value in model.graph.outputs},
        },
    }


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
