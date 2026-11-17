# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Private builders shared by ONNX-GenAI tests and validation fixtures."""

from __future__ import annotations

import dataclasses
import os

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius._pipeline_contract import declare_optional_input


def _onnx_genai_schema_path() -> str:
    """Locate onnx-genai's published pipeline JSON schema.

    The vendored copy under ``_schema/`` is the default so conformance never
    skips. A developer checkout is deliberately not consulted implicitly: one
    that is ahead of or behind ``main`` makes the result machine-dependent. Set
    ``ONNX_GENAI_SCHEMA`` to validate against a specific revision.
    """
    override = os.environ.get("ONNX_GENAI_SCHEMA")
    if override:
        return override
    return os.path.join(os.path.dirname(__file__), "_schema", "inference_metadata.schema.json")


def _value(
    name: str,
    dtype: ir.DataType,
    shape: list[int | str],
) -> ir.Value:
    return ir.Value(
        name=name,
        type=ir.TensorType(dtype),
        shape=ir.Shape(shape),
    )


def _model(
    name: str,
    inputs: list[ir.Value],
    output_specs: list[tuple[str, ir.DataType, list[int | str]]],
) -> ir.Model:
    outputs = [_value(*spec) for spec in output_specs]
    nodes = [
        ir.Node("", "Identity", [inputs[0]], outputs=[output], name=f"emit_{output.name}")
        for output in outputs
    ]
    graph = ir.Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        name=name,
        opset_imports={"": 21},
    )
    model = ir.Model(graph, ir_version=10)
    assert ir.to_proto(model).graph.name == name
    return model


@dataclasses.dataclass
class _VisionConfig:
    image_size: int = 448
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    mm_tokens_per_image: int | None = None
    image_token_id: int = 200010


@dataclasses.dataclass
class _VlmConfig:
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 8
    hidden_size: int = 64
    vocab_size: int = 128
    max_position_embeddings: int = 4096
    image_token_id: int = 200010
    mm_tokens_per_image: int | None = None
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = False
    layer_types: list[str] | None = None
    vision: _VisionConfig = dataclasses.field(default_factory=_VisionConfig)


def _embedding_model(
    outputs: list[tuple[str, ir.DataType, list[int | str]]],
    *,
    optional_image: bool = False,
    include_audio: bool = False,
    optional_audio: bool = False,
) -> ir.Model:
    image_features = _value("image_features", ir.DataType.FLOAT, ["image_tokens", 64])
    if optional_image:
        declare_optional_input(
            image_features,
            presence="image",
            absent_shape=[0, 64],
        )
    inputs = [
        _value("input_ids", ir.DataType.INT64, ["batch", "sequence"]),
        image_features,
    ]
    if include_audio:
        audio_features = _value("audio_features", ir.DataType.FLOAT, ["audio_tokens", 64])
        if optional_audio:
            declare_optional_input(
                audio_features,
                presence="audio",
                absent_shape=[0, 64],
            )
        inputs.append(audio_features)
    return _model("embedding", inputs, outputs)


def _decoder_model(
    routed_inputs: list[tuple[str, ir.DataType, list[int | str]]],
    *,
    position_shape: list[int | str],
    raw_token_input: bool = False,
    fixed_state: bool = False,
    equal_kv_shape: bool = False,
    kv_head_dims: list[int] | None = None,
) -> ir.Model:
    inputs = [_value(name, dtype, shape) for name, dtype, shape in routed_inputs]
    inputs.extend(
        [
            _value(
                "attention_mask",
                ir.DataType.INT64,
                ["batch", "past_sequence + sequence"],
            ),
            _value("position_ids", ir.DataType.INT64, position_shape),
        ]
    )
    if raw_token_input:
        inputs.append(_value("input_ids", ir.DataType.INT64, ["batch", "sequence"]))
    kv_head_dims = kv_head_dims or [8]
    for layer, head_dim in enumerate(kv_head_dims):
        inputs.extend(
            [
                _value(
                    f"past_key_values.{layer}.key",
                    ir.DataType.FLOAT,
                    ["batch", 2, "past_sequence", head_dim],
                ),
                _value(
                    f"past_key_values.{layer}.value",
                    ir.DataType.FLOAT,
                    ["batch", 2, "past_sequence", head_dim],
                ),
            ]
        )
    output_specs = [("logits", ir.DataType.FLOAT, ["batch", "sequence", 128])]
    for layer, head_dim in enumerate(kv_head_dims):
        output_shape = (
            ["batch", 2, "past_sequence", head_dim]
            if equal_kv_shape
            else ["batch", 2, "total_sequence", head_dim]
        )
        output_specs.extend(
            [
                (f"present.{layer}.key", ir.DataType.FLOAT, output_shape),
                (f"present.{layer}.value", ir.DataType.FLOAT, output_shape),
            ]
        )
    if fixed_state:
        inputs.extend(
            [
                _value(
                    "past_key_values.3.conv_state",
                    ir.DataType.FLOAT,
                    ["batch", 16, 3],
                ),
                _value(
                    "past_key_values.3.recurrent_state",
                    ir.DataType.FLOAT,
                    ["batch", 2, 4, 8],
                ),
            ]
        )
        output_specs.extend(
            [
                (
                    "present.3.conv_state",
                    ir.DataType.FLOAT,
                    ["batch", 16, 3],
                ),
                (
                    "present.3.recurrent_state",
                    ir.DataType.FLOAT,
                    ["batch", 2, 4, 8],
                ),
            ]
        )
    return _model("decoder", inputs, output_specs)


def _native_package(
    vision_encoder: ir.Model,
    config: _VlmConfig,
    *,
    position_shape: list[int | str] | None = None,
    equal_kv_shape: bool = False,
) -> ModelPackage:
    return ModelPackage(
        {
            "decoder": _decoder_model(
                [
                    (
                        "inputs_embeds",
                        ir.DataType.FLOAT,
                        ["batch", "sequence", 64],
                    )
                ],
                position_shape=position_shape or ["batch", "sequence"],
                equal_kv_shape=equal_kv_shape,
            ),
            "vision_encoder": vision_encoder,
            "embedding": _embedding_model(
                [
                    (
                        "inputs_embeds",
                        ir.DataType.FLOAT,
                        ["batch", "sequence", 64],
                    )
                ]
            ),
        },
        config=config,
    )


@dataclasses.dataclass
class _Cfg:
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    hidden_size: int = 1024
    max_position_embeddings: int = 8192
    sliding_window: int | None = None
    model_type: str = "qwen"


@dataclasses.dataclass
class _VisionCfg:
    patch_size: int = 14
    temporal_patch_size: int = 2
    merge_size: int = 1
    spatial_merge_size: int = 1
    size: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"shortest_edge": 224, "longest_edge": 224}
    )


@dataclasses.dataclass
class _VlmCfg(_Cfg):
    vision: _VisionCfg = dataclasses.field(default_factory=_VisionCfg)
    image_token_id: int = 32000
    eos_token_id: int = 2


def _vlm_package(*, audio: bool = False):
    vision = _model(
        "vision_encoder",
        [
            _value("pixel_values", ir.DataType.FLOAT, ["patches", 1176]),
            _value("grid_thw", ir.DataType.INT64, ["images", 3]),
        ],
        [("image_features", ir.DataType.FLOAT, ["batch", 256, 32])],
    )
    embedding_inputs = [
        _value("input_ids", ir.DataType.INT64, ["batch", "sequence"]),
        _value("image_features", ir.DataType.FLOAT, ["batch", 256, 32]),
    ]
    components = {"vision_encoder": vision}
    if audio:
        components["audio_encoder"] = _model(
            "audio_encoder",
            [_value("input_features", ir.DataType.FLOAT, ["batch", 80, "frames"])],
            [("audio_features", ir.DataType.FLOAT, ["batch", 64, 32])],
        )
        embedding_inputs.append(_value("audio_features", ir.DataType.FLOAT, ["batch", 64, 32]))
    embedding = _model(
        "embedding",
        embedding_inputs,
        [("inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence", 32])],
    )
    decoder = _decoder_model(
        [("inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence", 32])],
        position_shape=["batch", "sequence"],
    )
    components.update({"embedding": embedding, "decoder": decoder})
    return ModelPackage(components, config=_VlmCfg())


def _decoder_package(config=None):
    model = _decoder_model(
        [],
        position_shape=["batch", "sequence"],
        raw_token_input=True,
    )
    return ModelPackage({"model": model}, config=config or _Cfg())


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
            _value(
                "past_key_values.0.value",
                ir.DataType.FLOAT,
                ["batch", 4, "past_sequence", 4],
            ),
        ],
        [
            _value("target_scores", ir.DataType.FLOAT, ["batch", 4, 32]),
            _value(
                "present.0.key",
                ir.DataType.FLOAT,
                ["batch", 2, "past_sequence + 4", 8],
            ),
            _value(
                "present.0.value",
                ir.DataType.FLOAT,
                ["batch", 4, "past_sequence + 4", 4],
            ),
        ],
    )
    return ModelPackage({"proposer": proposer, "verifier": verifier})
