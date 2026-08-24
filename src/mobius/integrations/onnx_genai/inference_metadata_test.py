# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for onnx-genai diffusion inference_metadata generation."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import re
from pathlib import Path

import jsonschema
import onnx_ir as ir
import pytest
import yaml

from mobius._model_package import ModelPackage
from mobius._pipeline_contract import (
    declare_component_presence,
    declare_optional_input,
)
from mobius.generation import build_greedy_sampler
from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    _decoder_io,
    _input_source_map,
    _match_max_token_grid,
    _port,
    _processor_values,
    add_policy_components_to_workflow,
    build_diffusion_pipeline_metadata,
    build_multimodal_pipeline_metadata,
    build_native_vlm_package_metadata,
    declare_input_admission,
    is_native_vlm_package,
    load_diffusers_scheduler_config,
    load_diffusers_vae_scaling_factor,
    published_value_references,
    validate_executable_closure,
    write_diffusion_pipeline_metadata,
    write_mtp_speculator_metadata,
    write_native_vlm_package_metadata,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_vlm_workflow_metadata,
)


def test_ort_extensions_processor_config_supplies_structural_values(tmp_path):
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor": {
                    "transforms": [
                        {
                            "operation": {
                                "type": "Resize",
                                "attrs": {
                                    "min_pixels": 784,
                                    "max_pixels": 2371600,
                                    "patch_size": 14,
                                    "merge_size": 2,
                                },
                            }
                        },
                        {
                            "operation": {
                                "type": "Rescale",
                                "attrs": {"rescale_factor": 1 / 255},
                            }
                        },
                        {
                            "operation": {
                                "type": "Normalize",
                                "attrs": {
                                    "mean": [0.5, 0.5, 0.5],
                                    "std": [0.5, 0.5, 0.5],
                                },
                            }
                        },
                        {
                            "operation": {
                                "type": "PatchImage",
                                "attrs": {"temporal_patch_size": 2},
                            }
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    values = _processor_values(str(tmp_path), object())

    assert values["size"] == {"shortest_edge": 784, "longest_edge": 2371600}
    assert values["patch_size"] == 14
    assert values["merge_size"] == 2
    assert values["temporal_patch_size"] == 2
    assert values["image_mean"] == [0.5, 0.5, 0.5]
    assert values["image_std"] == [0.5, 0.5, 0.5]


def test_max_token_packed_grid_derives_pixel_area_bounds():
    program = _match_max_token_grid(
        [
            _port(_value("pixel_values", ir.DataType.FLOAT, ["patches", 1176])),
            _port(_value("image_grid_thw", ir.DataType.INT64, ["images", 3])),
        ],
        {
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "max_image_tokens": 4096,
        },
    )

    assert program is not None
    resize = next(
        transform
        for transform in program.transforms(
            None,
            {
                "patch_size": 14,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "max_image_tokens": 4096,
            },
        )
        if transform["op"] == "resize"
    )
    assert resize["min_pixels"] == 784
    assert resize["max_pixels"] == 3_211_264


def test_workflow_policy_components_reference_saved_onnx_artifacts(tmp_path):
    package = ModelPackage({"model": _model("model", [], [])})
    package.add_policy_component("sample", build_greedy_sampler())
    package.save(str(tmp_path))
    metadata = {
        "pipeline": {
            "workflow": {
                "manifest": {},
                "components": {},
                "graph": {"kind": "sequence", "nodes": []},
            }
        }
    }

    add_policy_components_to_workflow(metadata, package)

    component = metadata["pipeline"]["workflow"]["components"]["sample"]
    assert component["implementation"] == {
        "kind": "onnx",
        "artifact": "policies/sample.onnx",
    }
    assert component["ports"] == {
        "inputs": {
            "logits": {
                "dtype": "float32",
                "rank": 2,
                "shape": ["batch", "vocabulary"],
                "batch_layout": {"kind": "request_aligned", "axis": 0},
            }
        },
        "outputs": {
            "token": {
                "dtype": "int64",
                "rank": 1,
                "shape": ["batch"],
                "batch_layout": {"kind": "request_aligned", "axis": 0},
            }
        },
    }
    assert component["contract"] == {
        "id": "onnx-genai.token-sampler",
        "version": "1",
        "bindings": {"logits": "logits", "token": "token"},
        "parameters": {"mode": "greedy"},
    }
    assert "effects" not in component
    assert (tmp_path / component["implementation"]["artifact"]).is_file()


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


def _static_cache_decoder_model() -> ir.Model:
    inputs = [
        _value("input_ids", ir.DataType.INT64, ["batch", "sequence"]),
        _value("attention_mask", ir.DataType.INT64, ["batch", 32]),
        _value("position_ids", ir.DataType.INT64, ["batch", "sequence"]),
        _value("key_cache.1", ir.DataType.FLOAT, ["batch", 32, 16]),
        _value("value_cache.1", ir.DataType.FLOAT, ["batch", 32, 16]),
        _value("key_cache.3", ir.DataType.FLOAT, ["batch", 32, 16]),
        _value("value_cache.3", ir.DataType.FLOAT, ["batch", 32, 16]),
        _value("write_indices", ir.DataType.INT64, ["batch"]),
        _value("nonpad_kv_seqlen", ir.DataType.INT64, ["batch"]),
    ]
    outputs = [
        ("logits", ir.DataType.FLOAT, ["batch", "sequence", 128]),
        ("updated_key_cache.1", ir.DataType.FLOAT, ["batch", 32, 16]),
        ("updated_value_cache.1", ir.DataType.FLOAT, ["batch", 32, 16]),
        ("updated_key_cache.3", ir.DataType.FLOAT, ["batch", 32, 16]),
        ("updated_value_cache.3", ir.DataType.FLOAT, ["batch", 32, 16]),
    ]
    return _model("decoder", inputs, outputs)


class TestCrossAttentionCacheSources:
    def test_cross_attention_cache_inputs_are_loop_state(self):
        inputs = [
            _value("input_ids", ir.DataType.INT64, ["batch", "sequence"]),
            _value(
                "encoder_hidden_states",
                ir.DataType.FLOAT,
                ["batch", "encoder_sequence", 64],
            ),
            _value(
                "past_key_values.0.self.key",
                ir.DataType.FLOAT,
                ["batch", 2, "past_sequence", 8],
            ),
            _value(
                "past_key_values.0.self.value",
                ir.DataType.FLOAT,
                ["batch", 2, "past_sequence", 8],
            ),
            _value(
                "past_key_values.0.cross.key",
                ir.DataType.FLOAT,
                ["batch", 2, "encoder_sequence", 8],
            ),
            _value(
                "past_key_values.0.cross.value",
                ir.DataType.FLOAT,
                ["batch", 2, "encoder_sequence", 8],
            ),
        ]
        outputs = [
            ("logits", ir.DataType.FLOAT, ["batch", "sequence", 128]),
            (
                "present.0.self.key",
                ir.DataType.FLOAT,
                ["batch", 2, "total_sequence", 8],
            ),
            (
                "present.0.self.value",
                ir.DataType.FLOAT,
                ["batch", 2, "total_sequence", 8],
            ),
            (
                "present.0.cross.key",
                ir.DataType.FLOAT,
                ["batch", 2, "encoder_sequence", 8],
            ),
            (
                "present.0.cross.value",
                ir.DataType.FLOAT,
                ["batch", 2, "encoder_sequence", 8],
            ),
        ]
        decoder = _model("decoder", inputs, outputs)
        decoder_io, _ = _decoder_io(decoder, {"encoder_hidden_states"}, _VlmConfig())
        ports = {
            "decoder": {
                "inputs": [_port(value) for value in decoder.graph.inputs],
                "outputs": [_port(value) for value in decoder.graph.outputs],
            }
        }
        models = {"decoder": {"io": decoder_io}}
        sources = _input_source_map(
            ports=ports,
            dataflow=[],
            models=models,
            decoder_name="decoder",
            image_endpoints=set(),
        )
        assert sources["decoder.past_key_values.0.cross.key"] == {
            "kind": "stateful",
            "from": "decoder.present.0.cross.key",
            "update": "append",
        }
        assert sources["decoder.past_key_values.0.cross.value"] == {
            "kind": "stateful",
            "from": "decoder.present.0.cross.value",
            "update": "append",
        }

    @staticmethod
    def _nested_package(inner_outputs):
        talker = _model(
            "talker",
            [_value("inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence", 64])],
            [("logits", ir.DataType.FLOAT, ["batch", "sequence", 128])],
        )
        code_predictor = _model(
            "code_predictor",
            [_value("inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence", 64])],
            inner_outputs,
        )
        return {"talker": talker, "code_predictor": code_predictor}

    @staticmethod
    def _nested_metadata(inner_embedding_output=None):
        strategy = {
            "kind": "nested_autoregressive",
            "outer": "talker",
            "inner": "code_predictor",
        }
        if inner_embedding_output is not None:
            strategy["inner_embedding_output"] = inner_embedding_output
        return {
            "pipeline": {
                "models": {
                    "talker": {"type": "decoder"},
                    "code_predictor": {"type": "decoder"},
                },
                "dataflow": [],
                "strategy": strategy,
            }
        }


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


def _assert_all_graph_ports_declared(
    package: ModelPackage,
    metadata: dict,
) -> None:
    models = metadata["pipeline"]["models"]
    assert set(models) == set(package)
    for name, model in package.items():
        io = models[name]["io"]
        assert [port["name"] for port in io["inputs"]] == [
            value.name for value in model.graph.inputs
        ]
        assert [port["name"] for port in io["outputs"]] == [
            value.name for value in model.graph.outputs
        ]
        for port, value in zip(io["inputs"], model.graph.inputs):
            assert (
                port["dtype"]
                == {
                    ir.DataType.FLOAT: "fp32",
                    ir.DataType.FLOAT16: "fp16",
                    ir.DataType.INT64: "int64",
                    ir.DataType.BOOL: "bool",
                }[value.dtype]
            )
            assert port["rank"] == len(value.shape)
            assert "source" in port
        for port, value in zip(io["outputs"], model.graph.outputs):
            assert port["rank"] == len(value.shape)


class TestNativeVlmPackageMetadata:
    def _validate(self, package, config, source=None) -> None:
        """Validate the *published* package contract against onnx-genai's schema.

        ``build_native_vlm_package_metadata`` returns mobius's internal
        structural descriptor; what a package publishes is the typed SSA
        workflow ``build_vlm_workflow_metadata`` derives from it.
        """
        published = build_vlm_workflow_metadata(package, config, source=source)
        assert set(published["pipeline"]) == {"workflow"}
        with open(_onnx_genai_schema_path(), encoding="utf-8") as handle:
            jsonschema.validate(instance=published, schema=json.load(handle))

    def test_gemma4_routes_all_embedding_outputs(self, tmp_path):
        config = _VlmConfig(
            image_token_id=258880,
            vision=_VisionConfig(
                image_size=896,
                patch_size=16,
                image_token_id=258880,
            ),
        )
        package = ModelPackage(
            {
                "decoder": _decoder_model(
                    [
                        (
                            "inputs_embeds",
                            ir.DataType.FLOAT,
                            ["batch", "sequence", 64],
                        ),
                        (
                            "per_layer_inputs",
                            ir.DataType.FLOAT,
                            ["batch", "sequence", 128],
                        ),
                    ],
                    position_shape=["batch", "sequence"],
                    raw_token_input=True,
                    kv_head_dims=[8, 16, 8],
                ),
                "vision_encoder": _model(
                    "vision_encoder",
                    [
                        _value(
                            "pixel_values",
                            ir.DataType.FLOAT,
                            ["images", 2520, 768],
                        ),
                        _value(
                            "pixel_position_ids",
                            ir.DataType.INT64,
                            ["images", 2520, 2],
                        ),
                    ],
                    [
                        (
                            "image_features",
                            ir.DataType.FLOAT,
                            ["image_tokens", 64],
                        )
                    ],
                ),
                "embedding": _embedding_model(
                    [
                        (
                            "inputs_embeds",
                            ir.DataType.FLOAT,
                            ["batch", "sequence", 64],
                        ),
                        (
                            "per_layer_inputs",
                            ir.DataType.FLOAT,
                            ["batch", "sequence", 128],
                        ),
                    ],
                    optional_image=True,
                    include_audio=True,
                    optional_audio=True,
                ),
                "audio_encoder": _model(
                    "audio_encoder",
                    [
                        _value(
                            "input_features",
                            ir.DataType.FLOAT,
                            ["batch", "time", 128],
                        ),
                        _value(
                            "input_features_mask",
                            ir.DataType.BOOL,
                            ["batch", "time"],
                        ),
                    ],
                    [
                        (
                            "audio_features",
                            ir.DataType.FLOAT,
                            ["audio_tokens", 64],
                        ),
                        (
                            "audio_features_mask",
                            ir.DataType.BOOL,
                            ["batch", "audio_sequence"],
                        ),
                    ],
                ),
            },
            config=config,
        )
        declare_component_presence(package["vision_encoder"].graph, "image")
        declare_component_presence(package["audio_encoder"].graph, "audio")

        source = tmp_path / "gemma"
        source.mkdir()
        (source / "processor_config.json").write_text(
            json.dumps(
                {
                    "image_processor": {
                        "do_normalize": False,
                        "do_rescale": True,
                        "do_resize": True,
                        "rescale_factor": 1 / 255,
                        "patch_size": 16,
                        "pooling_kernel_size": 3,
                        "max_soft_tokens": 280,
                    }
                }
            )
        )

        assert is_native_vlm_package(package)
        metadata = build_native_vlm_package_metadata(
            package, config=config, source=str(source)
        )
        emitted_yaml = yaml.safe_load(yaml.safe_dump(metadata, sort_keys=False))
        validate_executable_closure(package, metadata)
        self._validate(package, config, source=str(source))
        _assert_all_graph_ports_declared(package, metadata)
        assert metadata["schema_version"] == "v1"
        assert {
            "image_preprocessing_program",
            "packed_image_outputs",
            "position_program",
            "dual_sequence_inputs",
        } <= set(metadata["required_capabilities"])
        flow = metadata["pipeline"]["dataflow"]
        assert {
            "from": "embedding.inputs_embeds",
            "to": "decoder.inputs_embeds",
            "dtype": "fp32",
            "device_transfer": False,
        } in flow
        assert {
            "from": "embedding.per_layer_inputs",
            "to": "decoder.per_layer_inputs",
            "dtype": "fp32",
            "device_transfer": False,
        } in flow
        assert {
            "from": "audio_encoder.audio_features",
            "to": "embedding.audio_features",
            "dtype": "fp32",
            "device_transfer": False,
        } in flow
        embedding_audio = next(
            port
            for port in metadata["pipeline"]["models"]["embedding"]["io"]["inputs"]
            if port["name"] == "audio_features"
        )
        assert embedding_audio["source"] == {
            "kind": "dataflow",
            "from": "audio_encoder.audio_features",
        }
        assert emitted_yaml["pipeline"]["models"]["embedding"]["io"]["optional_inputs"] == {
            "image_features": {
                "presence": "image",
                "absent": {"kind": "zeros", "shape": [0, 64]},
            },
            "audio_features": {
                "presence": "audio",
                "absent": {"kind": "zeros", "shape": [0, 64]},
            },
        }
        assert "phases" not in emitted_yaml["pipeline"]
        assert "phases" not in metadata["pipeline"]
        assert metadata["pipeline"]["models"]["embedding"]["io"]["token_input"] == "input_ids"
        assert metadata["pipeline"]["vision"]["token_count_source"] == "from_coordinates"
        assert metadata["pipeline"]["vision"]["token_pooling_factor"] == 9
        transforms = metadata["preprocessing"]["image"]["transforms"]
        assert next(transform for transform in transforms if transform["op"] == "resize") == {
            "op": "resize",
            "mode": "aspect_ratio_patch_budget",
            "patch_size": 16,
            "max_patches": 2520,
            "pooling_kernel_size": 3,
            "interpolation": "bicubic",
            "inputs": ["image.transform_0"],
            "outputs": ["image.transform_1"],
        }
        assert (
            next(transform for transform in transforms if transform["op"] == "pad")[
                "target_length"
            ]
            == 2520
        )
        assert not any(transform["op"] == "normalize" for transform in transforms)
        assert "model" not in metadata or "io" not in metadata["model"]
        decoder_io = metadata["pipeline"]["models"]["decoder"]["io"]
        assert decoder_io["token_input"] == "input_ids"
        assert decoder_io["kv_inputs"] == [
            f"past_key_values.{layer}.{role}"
            for layer in range(3)
            for role in ("key", "value")
        ]
        assert decoder_io["kv_outputs"] == [
            f"present.{layer}.{role}" for layer in range(3) for role in ("key", "value")
        ]
        kv_inputs = {
            port["name"]: port
            for port in metadata["pipeline"]["models"]["decoder"]["io"]["inputs"]
            if port["name"].startswith("past_key_values.")
        }
        assert kv_inputs["past_key_values.1.key"]["shape"][-1] == 16
        image_outputs = metadata["preprocessing"]["image"]["outputs"]
        # Every output binds the processor-local value that produces it, so the
        # runtime never has to guess which transform an output came from.
        assert image_outputs == [
            {
                "name": "vision_encoder.pixel_values",
                "content": "pixels",
                "dtype": "fp32",
                "source": "image.transform_4",
            },
            {
                "name": "vision_encoder.pixel_position_ids",
                "content": "patch_coordinates",
                "dtype": "int64",
                "pad_value": -1,
                "source": "image.output_patch_coordinates",
            },
        ]
        broken = copy.deepcopy(metadata)
        broken["pipeline"]["dataflow"] = [
            edge
            for edge in broken["pipeline"]["dataflow"]
            if edge["to"] != "decoder.per_layer_inputs"
        ]
        with pytest.raises(
            ValueError,
            match=r"What:.*decoder\.per_layer_inputs.*Why:.*How to fix:",
        ):
            validate_executable_closure(package, broken)

    def test_qwen_packed_grid_rank3_positions_sparse_and_fixed_state(self, tmp_path):
        config = _VlmConfig(
            mrope_section=[16, 24, 24],
            mrope_interleaved=True,
            layer_types=[
                "full_attention",
                "full_attention",
                "full_attention",
                "linear_attention",
            ],
            vision=_VisionConfig(
                patch_size=14,
                temporal_patch_size=2,
                spatial_merge_size=2,
            ),
        )
        package = ModelPackage(
            {
                "decoder": _decoder_model(
                    [
                        (
                            "inputs_embeds",
                            ir.DataType.FLOAT,
                            ["batch", "sequence", 64],
                        )
                    ],
                    position_shape=[3, "batch", "sequence"],
                    fixed_state=True,
                ),
                "vision_encoder": _model(
                    "vision_encoder",
                    [
                        _value(
                            "pixel_values",
                            ir.DataType.FLOAT,
                            ["total_patches", 1176],
                        ),
                        _value(
                            "image_grid_thw",
                            ir.DataType.INT64,
                            ["images", 3],
                        ),
                    ],
                    [
                        (
                            "image_features",
                            ir.DataType.FLOAT,
                            ["image_tokens", 64],
                        )
                    ],
                ),
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

        source = tmp_path / "qwen"
        source.mkdir()
        (source / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "size": {
                        "shortest_edge": 65536,
                        "longest_edge": 16777216,
                    },
                    "patch_size": 14,
                    "temporal_patch_size": 2,
                    "merge_size": 2,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                }
            )
        )
        metadata = build_native_vlm_package_metadata(
            package, config=config, source=str(source)
        )
        self._validate(package, config, source=str(source))
        _assert_all_graph_ports_declared(package, metadata)
        assert {
            "position_program",
            "multi_axis_positions",
            "loop_carried_state",
        } <= set(metadata["required_capabilities"])
        positions = metadata["pipeline"]["positions"]
        assert positions == {
            "input": "position_ids",
            "rank": 3,
            "tensor_rank": 3,
            "dtype": "int64",
            "generation": "processor_coordinates",
            "continuation": "carry_max",
            "axes": ["temporal", "height", "width"],
            "sections": [16, 24, 24],
            "processor_summaries": ["vision_encoder.image_grid_thw"],
        }
        io = metadata["pipeline"]["models"]["decoder"]["io"]
        assert io["kv_inputs"] == [
            "past_key_values.0.key",
            "past_key_values.0.value",
        ]
        assert io["kv_outputs"] == ["present.0.key", "present.0.value"]
        assert io["state_pairs"] == [
            {
                "input": "past_key_values.3.conv_state",
                "output": "present.3.conv_state",
                "init": "zeros",
                "update": "replace",
            },
            {
                "input": "past_key_values.3.recurrent_state",
                "output": "present.3.recurrent_state",
                "init": "zeros",
                "update": "replace",
            },
        ]
        resize = next(
            transform
            for transform in metadata["preprocessing"]["image"]["transforms"]
            if transform["op"] == "resize"
        )
        assert resize["mode"] == "pixel_area"
        assert resize["min_pixels"] == 65536
        assert resize["max_pixels"] == 16777216
        assert "size" not in resize

    def test_phi_routes_both_modality_gates_and_mask_processor(self, tmp_path):
        config = _VlmConfig()
        package = ModelPackage(
            {
                "decoder": _decoder_model(
                    [
                        (
                            "inputs_embeds",
                            ir.DataType.FLOAT,
                            ["batch", "sequence", 64],
                        ),
                        ("vision_gate", ir.DataType.FLOAT, []),
                        ("speech_gate", ir.DataType.FLOAT, []),
                    ],
                    position_shape=["batch", "sequence"],
                ),
                "vision_encoder": _model(
                    "vision_encoder",
                    [
                        _value(
                            "pixel_values",
                            ir.DataType.FLOAT,
                            ["crops", 3, 448, 448],
                        ),
                        _value(
                            "image_sizes",
                            ir.DataType.INT64,
                            ["images", 2],
                        ),
                        _value(
                            "image_attention_mask",
                            ir.DataType.FLOAT,
                            ["crops", 32, 32],
                        ),
                    ],
                    [
                        (
                            "image_features",
                            ir.DataType.FLOAT,
                            ["image_tokens", 64],
                        )
                    ],
                ),
                "audio_encoder": _model(
                    "audio_encoder",
                    [
                        _value(
                            "audio_embeds",
                            ir.DataType.FLOAT,
                            ["batch", "audio_sequence", 80],
                        )
                    ],
                    [
                        (
                            "audio_features",
                            ir.DataType.FLOAT,
                            ["audio_tokens", 64],
                        )
                    ],
                ),
                "embedding": _embedding_model(
                    [
                        (
                            "inputs_embeds",
                            ir.DataType.FLOAT,
                            ["batch", "sequence", 64],
                        ),
                        ("vision_gate", ir.DataType.FLOAT, []),
                        ("speech_gate", ir.DataType.FLOAT, []),
                    ],
                    include_audio=True,
                ),
            },
            config=config,
        )

        source = tmp_path / "phi"
        source.mkdir()
        (source / "config.json").write_text(
            json.dumps(
                {
                    "embd_layer": {
                        "image_embd_layer": {
                            "crop_size": 448,
                            "hd_transform_order": "sub_glb",
                            "use_hd_transform": True,
                        }
                    }
                }
            )
        )
        (source / "preprocessor_config.json").write_text(json.dumps({"dynamic_hd": 36}))
        metadata = build_native_vlm_package_metadata(
            package, config=config, source=str(source)
        )
        self._validate(package, config, source=str(source))
        _assert_all_graph_ports_declared(package, metadata)
        flow = metadata["pipeline"]["dataflow"]
        for gate in ("vision_gate", "speech_gate"):
            assert {
                "from": f"embedding.{gate}",
                "to": f"decoder.{gate}",
                "dtype": "fp32",
                "device_transfer": False,
            } in flow
        outputs = metadata["preprocessing"]["image"]["outputs"]
        assert {output["content"] for output in outputs} == {
            "pixels",
            "transformed_size",
            "validity_mask",
        }
        tile = next(
            transform
            for transform in metadata["preprocessing"]["image"]["transforms"]
            if transform["op"] == "tile"
        )
        assert tile["mode"] == "dynamic_hd"
        assert tile["tile_size"] == 448
        assert tile["max_tiles"] == 36
        assert tile["include_thumbnail"] is True

    def test_equal_shape_key_value_ports_remain_declared_kv(self, tmp_path):
        config = _VlmConfig()
        vision = _model(
            "vision_encoder",
            [
                _value("pixel_values", ir.DataType.FLOAT, ["images", 2520, 768]),
                _value("pixel_position_ids", ir.DataType.INT64, ["images", 2520, 2]),
            ],
            [("image_features", ir.DataType.FLOAT, ["image_tokens", 64])],
        )
        package = _native_package(vision, config, equal_kv_shape=True)
        source = tmp_path / "processor"
        source.mkdir()
        (source / "processor_config.json").write_text(
            json.dumps(
                {
                    "image_processor": {
                        "do_normalize": False,
                        "do_rescale": True,
                        "rescale_factor": 1 / 255,
                        "patch_size": 16,
                        "pooling_kernel_size": 3,
                        "max_soft_tokens": 280,
                    }
                }
            )
        )

        metadata = build_native_vlm_package_metadata(
            package, config=config, source=str(source)
        )
        io = metadata["pipeline"]["models"]["decoder"]["io"]
        assert io["kv_update"] == "append"
        assert io["kv_inputs"] == [
            "past_key_values.0.key",
            "past_key_values.0.value",
        ]
        assert "state_pairs" not in io

    def test_rank3_positions_require_registry_declaration(self, tmp_path):
        config = _VlmConfig(mrope_section=[16, 24, 24], mrope_interleaved=False)
        vision = _model(
            "vision_encoder",
            [
                _value("pixel_values", ir.DataType.FLOAT, ["patches", 1176]),
                _value("image_grid_thw", ir.DataType.INT64, ["images", 3]),
            ],
            [("image_features", ir.DataType.FLOAT, ["image_tokens", 64])],
        )
        package = _native_package(vision, config, position_shape=[3, "batch", "sequence"])
        source = tmp_path / "processor"
        source.mkdir()
        (source / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "size": {
                        "shortest_edge": 65536,
                        "longest_edge": 16777216,
                    },
                    "patch_size": 14,
                    "temporal_patch_size": 2,
                    "merge_size": 2,
                }
            )
        )

        with pytest.raises(ValueError, match=r"Rank-3 axes.*never guessed"):
            build_native_vlm_package_metadata(package, config=config, source=str(source))

    def test_unsupported_vlm_signature_fails_actionably(self, tmp_path):
        config = _VlmConfig()
        vision = _model(
            "vision_encoder",
            [_value("pixel_values", ir.DataType.FLOAT, ["batch", 3, 224, 224])],
            [("image_features", ir.DataType.FLOAT, ["image_tokens", 64])],
        )
        package = _native_package(vision, config)
        source = tmp_path / "processor"
        source.mkdir()
        (source / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "size": {
                        "shortest_edge": 65536,
                        "longest_edge": 16777216,
                    },
                    "patch_size": 14,
                    "temporal_patch_size": 2,
                    "merge_size": 2,
                }
            )
        )

        assert is_native_vlm_package(package)
        with pytest.raises(
            ValueError,
            match=r"Generic fallback is unsafe.*Regenerate.*register",
        ):
            build_native_vlm_package_metadata(package, config=config, source=str(source))

    def test_missing_native_vlm_components_fail_actionably(self):
        config = _VlmConfig()
        vision_only = ModelPackage(
            {
                "vision_encoder": _model(
                    "vision_encoder",
                    [_value("pixel_values", ir.DataType.FLOAT, ["patches", 1536])],
                    [("image_features", ir.DataType.FLOAT, ["tokens", 64])],
                )
            },
            config=config,
        )

        assert is_native_vlm_package(vision_only)
        with pytest.raises(
            ValueError,
            match=r"missing required component.*decoder.*embedding.*Why:.*How to fix:",
        ):
            build_native_vlm_package_metadata(vision_only, config=config)

    def test_cached_qwen_processor_matches_emitted_area_program(self):
        np = pytest.importorskip("numpy")
        image_module = pytest.importorskip("PIL.Image")
        transformers = pytest.importorskip("transformers")
        model_id = "Qwen/Qwen3-VL-2B-Instruct"
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                model_id, local_files_only=True
            ).image_processor
        except Exception as error:
            pytest.skip(f"cached Qwen processor unavailable (offline): {error}")

        image = image_module.fromarray(np.zeros((300, 500, 3), dtype=np.uint8))
        reference = processor(images=[image], return_tensors="np")
        assert reference["pixel_values"].shape[0] == 576
        assert reference["image_grid_thw"].tolist() == [[1, 18, 32]]

        config = _VlmConfig(
            mrope_section=[24, 20, 20],
            mrope_interleaved=True,
            vision=_VisionConfig(
                image_size=448,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
            ),
        )
        vision = _model(
            "vision_encoder",
            [
                _value("pixel_values", ir.DataType.FLOAT, ["patches", 1536]),
                _value("image_grid_thw", ir.DataType.INT64, ["images", 3]),
            ],
            [("image_features", ir.DataType.FLOAT, ["image_tokens", 64])],
        )
        metadata = build_native_vlm_package_metadata(
            _native_package(vision, config, position_shape=[3, "batch", "sequence"]),
            config=config,
            source=model_id,
        )
        transforms = metadata["preprocessing"]["image"]["transforms"]
        resize = next(transform for transform in transforms if transform["op"] == "resize")
        height = round(300 / resize["size_multiple"]) * resize["size_multiple"]
        width = round(500 / resize["size_multiple"]) * resize["size_multiple"]
        if height * width < resize["min_pixels"]:
            scale = math.sqrt(resize["min_pixels"] / (300 * 500))
            height = math.ceil(300 * scale / resize["size_multiple"]) * resize["size_multiple"]
            width = math.ceil(500 * scale / resize["size_multiple"]) * resize["size_multiple"]
        elif height * width > resize["max_pixels"]:
            scale = math.sqrt((300 * 500) / resize["max_pixels"])
            height = (
                math.floor(300 / scale / resize["size_multiple"]) * resize["size_multiple"]
            )
            width = math.floor(500 / scale / resize["size_multiple"]) * resize["size_multiple"]
        patchify = next(transform for transform in transforms if transform["op"] == "patchify")
        emitted_patch_count = (height // patchify["patch_size"]) * (
            width // patchify["patch_size"]
        )
        assert emitted_patch_count == reference["pixel_values"].shape[0] == 576
        emitted_patch_width = (
            3
            * patchify["temporal_patch_size"]
            * patchify["patch_size"]
            * patchify["patch_size"]
        )
        assert emitted_patch_width == reference["pixel_values"].shape[1] == 1536
        assert patchify["channel_order"] == "channels_first"
        assert np.all(reference["pixel_values"] == -1)

    def test_cached_gemma_processor_matches_emitted_patch_budget(self):
        np = pytest.importorskip("numpy")
        image_module = pytest.importorskip("PIL.Image")
        pytest.importorskip("torchvision")
        from huggingface_hub import constants, scan_cache_dir
        from huggingface_hub.file_download import repo_folder_name
        from transformers.models.gemma4.image_processing_gemma4 import (
            Gemma4ImageProcessor,
        )

        model_id = "google/gemma-4-E2B-it-assistant"
        # Both artifacts below are read straight out of the local Hugging Face cache,
        # so this test only runs offline-style against whatever is already downloaded.
        cached_config = next(
            (
                str(file.file_path)
                for repo in scan_cache_dir().repos
                if repo.repo_id == model_id
                for revision in repo.revisions
                for file in revision.files
                if file.file_name == "config.json"
            ),
            None,
        )
        if cached_config is not None:
            # Optional cross-check: the assistant variant reuses the base checkpoint's
            # image processor, so only assert it when that repo happens to be cached.
            assert json.loads(Path(cached_config).read_text(encoding="utf-8"))[
                "model_type"
            ] == ("gemma4_assistant")
        processor_cache = (
            Path(constants.HF_HUB_CACHE)
            / repo_folder_name(repo_id="google/gemma-4-E2B-it", repo_type="model")
            / "snapshots"
        )
        processor_configs = sorted(
            processor_cache.glob("*/processor_config.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if not processor_configs:
            pytest.skip(
                "cached Gemma4 processor unavailable (offline): no "
                "google/gemma-4-E2B-it processor_config.json in the Hugging Face cache"
            )
        processor_path = processor_configs[-1]
        processor_values = json.loads(processor_path.read_text(encoding="utf-8"))[
            "image_processor"
        ]
        processor = Gemma4ImageProcessor(
            **{
                key: value
                for key, value in processor_values.items()
                if key != "image_processor_type"
            }
        )
        image = image_module.fromarray(np.zeros((300, 500, 3), dtype=np.uint8))
        reference = processor(images=[image], return_tensors="np")
        assert reference["pixel_values"].shape == (1, 2520, 768)
        assert int(reference["num_soft_tokens_per_image"][0]) == 252

        config = _VlmConfig(
            image_token_id=258880,
            vision=_VisionConfig(image_size=896, patch_size=16, image_token_id=258880),
        )
        vision = _model(
            "vision_encoder",
            [
                _value("pixel_values", ir.DataType.FLOAT, ["images", 2520, 768]),
                _value("pixel_position_ids", ir.DataType.INT64, ["images", 2520, 2]),
            ],
            [("image_features", ir.DataType.FLOAT, ["image_tokens", 64])],
        )
        metadata = build_native_vlm_package_metadata(
            _native_package(vision, config),
            config=config,
            source=str(processor_path.parent),
        )
        transforms = metadata["preprocessing"]["image"]["transforms"]
        resize = next(transform for transform in transforms if transform["op"] == "resize")
        patchify = next(transform for transform in transforms if transform["op"] == "patchify")
        pad = next(transform for transform in transforms if transform["op"] == "pad")
        assert resize == {
            "op": "resize",
            "mode": "aspect_ratio_patch_budget",
            "patch_size": 16,
            "max_patches": 2520,
            "pooling_kernel_size": 3,
            "interpolation": "bicubic",
            "inputs": ["image.transform_0"],
            "outputs": ["image.transform_1"],
        }
        assert pad["target_length"] == reference["pixel_values"].shape[1] == 2520
        assert patchify["channel_order"] == "channels_last"
        assert patchify["coordinate_order"] == "xy"
        assert not any(transform["op"] == "normalize" for transform in transforms)
        coordinate_output = next(
            output
            for output in metadata["preprocessing"]["image"]["outputs"]
            if output["content"] == "patch_coordinates"
        )
        assert coordinate_output["pad_value"] == -1
        assert np.all(reference["image_position_ids"][0, 2268:] == -1)
        assert np.all(reference["pixel_values"][0, 2268:] == 0)

    def test_cached_phi_processor_matches_emitted_dynamic_hd_program(self):
        np = pytest.importorskip("numpy")
        image_module = pytest.importorskip("PIL.Image")
        transformers = pytest.importorskip("transformers")
        model_id = "microsoft/Phi-4-multimodal-instruct"
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=True,
            ).image_processor
        except Exception as error:
            pytest.skip(f"cached Phi4MM processor unavailable (offline): {error}")

        image = image_module.fromarray(np.zeros((300, 500, 3), dtype=np.uint8))
        reference = processor(images=[image], return_tensors="np")
        assert reference["input_image_embeds"].shape == (1, 3, 3, 448, 448)
        assert reference["image_sizes"].tolist() == [[448, 896]]
        assert reference["image_attention_mask"].shape == (1, 3, 32, 32)

        config = _VlmConfig(
            vision=_VisionConfig(image_size=448, patch_size=14),
        )
        vision = _model(
            "vision_encoder",
            [
                _value("pixel_values", ir.DataType.FLOAT, ["crops", 3, 448, 448]),
                _value("image_sizes", ir.DataType.INT64, ["images", 2]),
                _value(
                    "image_attention_mask",
                    ir.DataType.FLOAT,
                    ["crops", 32, 32],
                ),
            ],
            [("image_features", ir.DataType.FLOAT, ["image_tokens", 64])],
        )
        metadata = build_native_vlm_package_metadata(
            _native_package(vision, config), config=config, source=model_id
        )
        transforms = metadata["preprocessing"]["image"]["transforms"]
        tile = next(transform for transform in transforms if transform["op"] == "tile")
        local_tiles = math.ceil(500 / tile["tile_size"]) * math.ceil(300 / tile["tile_size"])
        emitted_crops = local_tiles + int(tile["include_thumbnail"])
        assert emitted_crops == reference["input_image_embeds"].shape[1] == 3
        assert tile["mode"] == "dynamic_hd"
        assert tile["max_tiles"] == 36
        assert tile["mask_patch_size"] == 14
        assert tile["thumbnail_order"] == "prepend"
        assert tile["canvas_pad_value"] == 255
        assert tile["thumbnail_interpolation"] == "bicubic"
        assert reference["image_attention_mask"].shape[-1] == (
            tile["tile_size"] // tile["mask_patch_size"]
        )
        mask_output = next(
            output
            for output in metadata["preprocessing"]["image"]["outputs"]
            if output["content"] == "validity_mask"
        )
        assert mask_output["pad_value"] == 0
        size_output = next(
            output
            for output in metadata["preprocessing"]["image"]["outputs"]
            if output["content"] == "transformed_size"
        )
        assert size_output["name"] == "vision_encoder.image_sizes"
        assert float(reference["image_attention_mask"].min()) == pytest.approx(0)
        assert float(reference["image_attention_mask"].max()) == pytest.approx(1)

    def test_writer_copies_local_runtime_assets(self, tmp_path):
        config = _VlmConfig()
        source = tmp_path / "source"
        source.mkdir()
        (source / "tokenizer.json").write_text("{}", encoding="utf-8")
        (source / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ messages }}"}),
            encoding="utf-8",
        )
        (source / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "image_processor": {
                        "do_resize": True,
                        "do_rescale": True,
                        "do_normalize": False,
                        "rescale_factor": 1 / 255,
                        "patch_size": 16,
                        "pooling_kernel_size": 3,
                        "max_soft_tokens": 280,
                    }
                }
            ),
            encoding="utf-8",
        )
        package = ModelPackage(
            {
                "decoder": _decoder_model(
                    [
                        (
                            "inputs_embeds",
                            ir.DataType.FLOAT,
                            ["batch", "sequence", 64],
                        )
                    ],
                    position_shape=["batch", "sequence"],
                ),
                "vision_encoder": _model(
                    "vision_encoder",
                    [
                        _value(
                            "pixel_values",
                            ir.DataType.FLOAT,
                            ["images", 2520, 768],
                        ),
                        _value(
                            "pixel_position_ids",
                            ir.DataType.INT64,
                            ["images", 2520, 2],
                        ),
                    ],
                    [
                        (
                            "image_features",
                            ir.DataType.FLOAT,
                            ["image_tokens", 64],
                        )
                    ],
                ),
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

        output = tmp_path / "output"
        artifacts = write_native_vlm_package_metadata(
            package,
            str(output),
            config=config,
            source=str(source),
        )
        assert os.path.isfile(artifacts["inference_metadata"])
        assert (output / "tokenizer.json").is_file()
        assert (output / "tokenizer_config.json").is_file()
        assert (output / "chat_template.jinja").is_file()
        assert (output / "preprocessor_config.json").is_file()
        metadata = yaml.safe_load((output / "inference_metadata.yaml").read_text())
        transforms = metadata["preprocessing"]["image"]["transforms"]
        assert (
            next(transform for transform in transforms if transform["op"] == "pad")[
                "target_length"
            ]
            == 2520
        )


def _constant_values(component) -> list[float]:
    """Read the constant a schedule policy component materializes."""
    initializer = next(node for node in component.model.graph if node.op_type == "Constant")
    return initializer.attributes["value"].as_tensor().numpy().tolist()


def _invocations(workflow: dict, component: str) -> list[dict]:
    """Every ``invoke`` step of ``component``, at any nesting depth."""
    found: list[dict] = []

    def walk(step: dict) -> None:
        kind = step["kind"]
        if kind == "invoke":
            if step["component"] == component:
                found.append(step)
        elif kind == "loop":
            for child in (*step.get("setup", []), *step["steps"]):
                walk(child)
        elif kind == "sequence":
            for child in step["steps"]:
                walk(child)
        elif kind == "branch":
            for case in step["cases"].values():
                walk(case)
            if step.get("default"):
                walk(step["default"])

    for step in workflow["steps"]:
        walk(step)
    return found


class TestBuildDiffusionPipelineMetadata:
    """The published document is a typed SSA workflow.

    ``PipelineSpec`` declares ``workflow`` as its only property
    (``crates/onnx-genai-metadata/src/schema/pipeline.rs``), so the sampler is
    an executable component the package ships.
    """

    def _workflow(self, **kwargs) -> dict:
        meta = build_diffusion_pipeline_metadata(
            vae_filename="vae.onnx",
            **kwargs,
        )
        assert set(meta["pipeline"]) == {"workflow"}
        return meta["pipeline"]["workflow"]

    def test_a_latent_only_pipeline_is_not_publishable(self):
        # The workflow terminates in a decoded image, so a package with no
        # decoder has no executable result to declare.
        with pytest.raises(ValueError, match="without a VAE decoder"):
            build_diffusion_pipeline_metadata(num_inference_steps=20)

    def test_denoise_loop_carries_the_latent_through_the_solver(self):
        workflow = self._workflow(num_inference_steps=20)
        assert set(workflow["components"]) >= {
            "denoiser",
            "vae",
            "solver_step",
            "diffusion_schedule",
            "diffusion_timesteps",
            "schedule_lookup",
        }
        loop = next(step for step in workflow["steps"] if step["kind"] == "loop")
        assert loop["max_iterations"] == "request.max_iterations"
        assert workflow["inputs"]["request.max_iterations"]["default"] == 20
        # The latent is loop-carried state, advanced by the solver each step.
        # ``latent`` is also a workflow output, so the published cell is
        # disambiguated to ``latent_state``.
        latent_carry = next(
            carry for carry in loop["carried"] if carry["cell"] == "latent_state"
        )
        assert latent_carry["next"] == "latent.body"
        solver = _invocations(workflow, "solver_step")[0]
        assert solver["inputs"]["sample"] == "latent_state"
        assert solver["inputs"]["derivative"] == "denoiser.estimate"
        assert solver["outputs"]["next_state"] == "latent.body"
        # The step index is the loop induction value, not a timestep port name.
        assert loop["iteration"]["value"] == "loop.iteration"
        lookup = _invocations(workflow, "schedule_lookup")[0]
        assert lookup["inputs"] == {
            "schedule": "diffusion.timesteps",
            "step": "loop.iteration",
        }

    def test_full_pipeline_with_vae_and_text_encoder(self):
        workflow = self._workflow(
            num_inference_steps=4,
            text_encoder_filename="text_encoder.onnx",
            guidance_scale=7.5,
        )
        assert set(workflow["components"]) >= {
            "denoiser",
            "vae",
            "text_encoder",
            "guidance_combine",
        }
        # The text encoder runs once in the loop setup; the VAE decodes the
        # final latent after the loop.
        loop = next(step for step in workflow["steps"] if step["kind"] == "loop")
        encoder_calls = [
            step for step in loop["setup"] if step.get("component") == "text_encoder"
        ]
        assert len(encoder_calls) == 2  # conditional + unconditional
        assert encoder_calls[0]["outputs"]["last_hidden_state"] == (
            "conditioning.encoder_hidden_states"
        )
        vae = _invocations(workflow, "vae")[0]
        assert vae["inputs"]["latent"] == "latent_state"
        # CFG is two denoiser invocations plus a combine, not a hidden flag.
        denoiser_calls = _invocations(workflow, "denoiser")
        assert len(denoiser_calls) == 2
        assert denoiser_calls[0]["inputs"]["encoder_hidden_states"] == (
            "conditioning.unconditional.encoder_hidden_states"
        )
        assert denoiser_calls[1]["inputs"]["encoder_hidden_states"] == (
            "conditioning.encoder_hidden_states"
        )
        assert workflow["inputs"]["request.guidance_scale"]["default"] == pytest.approx(7.5)

    def test_sdxl_dual_conditioning_edges(self):
        workflow = self._workflow(
            num_inference_steps=4,
            text_encoder_filename="text_encoder.onnx",
            guidance_scale=7.5,
            text_encoder_edges=[
                ("encoder_hidden_states", "encoder_hidden_states"),
                ("text_embeds", "text_embeds"),
            ],
        )
        loop = next(step for step in workflow["steps"] if step["kind"] == "loop")
        encoder = next(
            step for step in loop["setup"] if step.get("component") == "text_encoder"
        )
        assert encoder["outputs"] == {
            "encoder_hidden_states": "conditioning.encoder_hidden_states",
            "text_embeds": "conditioning.text_embeds",
        }
        denoiser = _invocations(workflow, "denoiser")[-1]
        assert denoiser["inputs"]["encoder_hidden_states"] == (
            "conditioning.encoder_hidden_states"
        )
        assert denoiser["inputs"]["text_embeds"] == "conditioning.text_embeds"

    def test_guidance_scale_one_does_not_enable_cfg(self):
        workflow = self._workflow(
            num_inference_steps=2,
            text_encoder_filename="text_encoder.onnx",
            guidance_scale=1.0,
        )
        assert "guidance_combine" not in workflow["components"]
        assert "request.guidance_scale" not in workflow["inputs"]
        assert len(_invocations(workflow, "denoiser")) == 1

    def test_start_step_slices_the_schedule(self):
        # img2img "skip the noisiest steps" is a shorter schedule, not a knob.
        workflow = self._workflow(num_inference_steps=10, start_step=4)
        assert workflow["inputs"]["request.max_iterations"]["default"] == 6

    def test_schedule_is_derived_from_the_scheduler_not_invented(self):
        """``solver_step`` integrates the checkpoint's own noise schedule.

        The schedule component is what the solver reads as alpha_cumprod (DDIM)
        or sigma (Euler / DPM-Solver++), so a placeholder ramp would silently
        denoise along the wrong trajectory.
        """
        from mobius._model_package import ModelPackage
        from mobius.integrations.onnx_genai.auto_export import _ddim_alpha_schedule

        scheduler = SchedulerConfig(kind="ddim", beta_start=0.0001, beta_end=0.02)
        package = ModelPackage({})
        build_diffusion_pipeline_metadata(
            num_inference_steps=5,
            vae_filename="vae.onnx",
            scheduler=scheduler,
            package=package,
        )
        _, expected = _ddim_alpha_schedule(scheduler, 5)
        emitted = _constant_values(package.policy_components["diffusion_schedule"])
        assert emitted == pytest.approx(expected, rel=1e-6)
        # Guard the specific regression: a 1 - i/n ramp is not a beta schedule.
        assert emitted != pytest.approx([1.0 - index / 5 for index in range(6)])

    def test_vae_scaling_factor_unnormalizes_the_decoder_input(self):
        workflow = self._workflow(num_inference_steps=3, vae_scaling_factor=0.18215)
        assert {"tensor_scale", "decoder_input_scale"} <= set(workflow["components"])
        vae = _invocations(workflow, "vae")[0]
        assert vae["inputs"]["latent"] == "diffusion.decoder_input"

    def test_variance_preserving_solver_does_not_rescale_the_initial_latent(self):
        workflow = self._workflow(num_inference_steps=3)
        # DDIM starts from the unit-variance draw itself; only a sigma-space
        # sampler scales it by the largest sigma.
        assert "initial_state_scale" not in workflow["components"]
        assert workflow["state"]["latent_state"]["initializer"] == "request.noise"

    def test_sigma_space_solver_scales_the_initial_latent(self):
        workflow = self._workflow(
            num_inference_steps=3, scheduler=SchedulerConfig(kind="euler")
        )
        assert "initial_state_scale" in workflow["components"]
        assert workflow["state"]["latent_state"]["initializer"] == "diffusion.initial_state"

    def test_stochastic_scheduler_is_rejected(self):
        with pytest.raises(ValueError, match="euler_ancestral"):
            build_diffusion_pipeline_metadata(
                num_inference_steps=4,
                vae_filename="vae.onnx",
                scheduler=SchedulerConfig(kind="euler_ancestral"),
            )

    def test_scheduler_from_diffusers_config(self):
        sched = SchedulerConfig.from_diffusers(
            {
                "_class_name": "DDIMScheduler",
                "num_train_timesteps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "prediction_type": "epsilon",
            }
        )
        assert sched.kind == "ddim"
        assert sched.beta_end == pytest.approx(0.02)

    def test_scheduler_preserves_cogvideox_ddim_equation_fields(self):
        sched = SchedulerConfig.from_diffusers(
            {
                "_class_name": "CogVideoXDDIMScheduler",
                "prediction_type": "v_prediction",
                "clip_sample": False,
                "set_alpha_to_one": True,
                "timestep_spacing": "trailing",
                "rescale_betas_zero_snr": True,
                "snr_shift_scale": 3.0,
            }
        )
        assert sched.kind == "ddim"
        assert sched.prediction_type == "v_prediction"
        assert sched.timestep_spacing == "trailing"
        assert sched.rescale_betas_zero_snr
        assert sched.snr_shift_scale == pytest.approx(3.0)

    def test_scheduler_maps_euler_class(self):
        sched = SchedulerConfig.from_diffusers(
            {"_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear"}
        )
        assert sched.kind == "euler"
        assert sched.beta_schedule == "scaled_linear"

    def test_scheduler_maps_qwen_flow_match_fields(self):
        sched = SchedulerConfig.from_diffusers(
            {
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "num_train_timesteps": 1000,
                "base_image_seq_len": 256,
                "max_image_seq_len": 8192,
                "base_shift": 0.5,
                "max_shift": 0.9,
                "shift": 1.0,
                "shift_terminal": 0.02,
                "time_shift_type": "exponential",
                "use_dynamic_shifting": True,
            }
        )
        metadata = sched.to_metadata()
        assert metadata["kind"] == "flow_match_euler"
        assert metadata["prediction_type"] == "flow_prediction"
        assert metadata["max_image_seq_len"] == 8192
        assert metadata["shift_terminal"] == pytest.approx(0.02)
        assert metadata["use_dynamic_shifting"] is True
        assert "beta_start" not in metadata

    def test_scheduler_maps_dpmsolver_class(self):
        sched = SchedulerConfig.from_diffusers({"_class_name": "DPMSolverMultistepScheduler"})
        assert sched.kind == "dpmpp_2m"

    def test_scheduler_defaults_to_ddim_when_class_absent(self):
        assert SchedulerConfig.from_diffusers({}).kind == "ddim"

    def test_scheduler_maps_euler_ancestral(self):
        sched = SchedulerConfig.from_diffusers(
            {"_class_name": "EulerAncestralDiscreteScheduler"}
        )
        assert sched.kind == "euler_ancestral"

    def test_scheduler_rejects_ancestral(self):
        # Other ancestral / SDE samplers have no onnx-genai equivalent yet.
        with pytest.raises(ValueError, match="stochastic"):
            SchedulerConfig.from_diffusers({"_class_name": "DPMSolverSDEScheduler"})

    def test_scheduler_rejects_unsupported_class(self):
        with pytest.raises(ValueError, match="unsupported"):
            SchedulerConfig.from_diffusers({"_class_name": "LMSDiscreteScheduler"})

    def test_load_scheduler_from_local_checkpoint(self, tmp_path):
        import json

        sd = tmp_path / "scheduler"
        sd.mkdir()
        (sd / "scheduler_config.json").write_text(
            json.dumps({"_class_name": "EulerDiscreteScheduler", "beta_end": 0.015})
        )
        sched = load_diffusers_scheduler_config(str(tmp_path))
        assert sched is not None
        assert sched.kind == "euler"
        assert sched.beta_end == pytest.approx(0.015)

    def test_load_scheduler_none_when_absent(self, tmp_path):
        assert load_diffusers_scheduler_config(str(tmp_path)) is None
        assert load_diffusers_scheduler_config(None) is None

    def test_load_scheduler_falls_back_on_unsupported(self, tmp_path):
        import json

        sd = tmp_path / "scheduler"
        sd.mkdir()
        (sd / "scheduler_config.json").write_text(
            json.dumps({"_class_name": "LMSDiscreteScheduler"})
        )
        # Unsupported scheduler must not raise from the loader; returns None so
        # the caller falls back to the DDIM default.
        assert load_diffusers_scheduler_config(str(tmp_path)) is None

    def test_load_vae_scaling_factor_forwards_revision(self, tmp_path, monkeypatch):
        config = tmp_path / "vae_config.json"
        config.write_text(json.dumps({"scaling_factor": 0.13025}), encoding="utf-8")
        calls = []

        def fake_download(source, filename, *, revision=None):
            calls.append((source, filename, revision))
            return str(config)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

        factor = load_diffusers_vae_scaling_factor(
            "zai-org/CogVideoX-2b",
            revision="pinned-revision",
        )

        assert factor == pytest.approx(0.13025)
        assert calls == [("zai-org/CogVideoX-2b", "vae/config.json", "pinned-revision")]

    def test_rejects_zero_steps(self):
        with pytest.raises(ValueError):
            build_diffusion_pipeline_metadata(num_inference_steps=0)

    def test_write_roundtrips_yaml(self, tmp_path):
        path = write_diffusion_pipeline_metadata(
            str(tmp_path), num_inference_steps=3, vae_filename="vae.onnx"
        )
        with open(path) as handle:
            loaded = yaml.safe_load(handle)
        workflow = loaded["pipeline"]["workflow"]
        assert workflow["inputs"]["request.max_iterations"]["default"] == 3
        assert "vae" in workflow["components"]
        # The workflow declares the sampler components as ONNX artifacts, so
        # the writer has to ship them next to the document.
        solver = workflow["components"]["solver_step"]["implementation"]["artifact"]
        assert (tmp_path / solver).is_file()

    def test_matches_onnx_genai_json_schema(self):
        """The emitted metadata validates against onnx-genai's published schema."""
        import json

        import jsonschema

        with open(_onnx_genai_schema_path()) as handle:
            schema = json.load(handle)
        meta = build_diffusion_pipeline_metadata(
            num_inference_steps=4,
            vae_filename="vae.onnx",
            text_encoder_filename="text_encoder.onnx",
            guidance_scale=7.5,
        )
        # Validate the whole InferenceMetadata document.
        jsonschema.validate(instance=meta, schema=schema)


class TestBuildMultimodalPipelineMetadata:
    def test_vision_only_pipeline(self):
        metadata = build_multimodal_pipeline_metadata(
            vision_encoder_filename="vision_encoder.onnx"
        )

        assert metadata == {
            "pipeline": {
                "models": {
                    "vision_encoder": {
                        "filename": "vision_encoder.onnx",
                        "type": "vision_encoder",
                    },
                    "embedding": {"filename": "embedding.onnx", "type": "encoder"},
                    "decoder": {
                        "filename": "decoder.onnx",
                        "type": "decoder",
                        "tokenizer": "tokenizer.json",
                    },
                },
                "dataflow": [
                    {
                        "from": "vision_encoder.image_features",
                        "to": "embedding.image_features",
                        "dtype": "fp32",
                        "device_transfer": False,
                    },
                    {
                        "from": "embedding.inputs_embeds",
                        "to": "decoder.inputs_embeds",
                        "dtype": "fp32",
                        "device_transfer": False,
                    },
                ],
                "strategy": {
                    "kind": "composite",
                    "stages": [
                        {
                            "name": "encode_vision",
                            "strategy": {
                                "kind": "single_pass",
                                "model": "vision_encoder",
                            },
                        },
                        {
                            "name": "fuse_embeddings",
                            "strategy": {
                                "kind": "single_pass",
                                "model": "embedding",
                            },
                        },
                        {
                            "name": "decode",
                            "strategy": {
                                "kind": "autoregressive",
                                "decoder": "decoder",
                            },
                        },
                    ],
                },
            }
        }

    def test_vision_and_audio_pipeline(self):
        metadata = build_multimodal_pipeline_metadata(
            vision_encoder_filename="vision_encoder.onnx",
            audio_encoder_filename="audio_encoder.onnx",
        )
        pipeline = metadata["pipeline"]

        assert pipeline["models"] == {
            "vision_encoder": {
                "filename": "vision_encoder.onnx",
                "type": "vision_encoder",
            },
            "audio_encoder": {
                "filename": "audio_encoder.onnx",
                "type": "audio_encoder",
            },
            "embedding": {"filename": "embedding.onnx", "type": "encoder"},
            "decoder": {
                "filename": "decoder.onnx",
                "type": "decoder",
                "tokenizer": "tokenizer.json",
            },
        }
        assert pipeline["dataflow"] == [
            {
                "from": "vision_encoder.image_features",
                "to": "embedding.image_features",
                "dtype": "fp32",
                "device_transfer": False,
            },
            {
                "from": "audio_encoder.audio_features",
                "to": "embedding.audio_features",
                "dtype": "fp32",
                "device_transfer": False,
            },
            {
                "from": "embedding.inputs_embeds",
                "to": "decoder.inputs_embeds",
                "dtype": "fp32",
                "device_transfer": False,
            },
        ]
        assert pipeline["strategy"]["stages"] == [
            {
                "name": "encode_vision",
                "strategy": {"kind": "single_pass", "model": "vision_encoder"},
            },
            {
                "name": "encode_audio",
                "strategy": {"kind": "single_pass", "model": "audio_encoder"},
            },
            {
                "name": "fuse_embeddings",
                "strategy": {"kind": "single_pass", "model": "embedding"},
            },
            {
                "name": "decode",
                "strategy": {"kind": "autoregressive", "decoder": "decoder"},
            },
        ]


@dataclasses.dataclass
class _MtpBackboneConfig:
    num_hidden_layers: int = 64
    hidden_size: int = 5120
    vocab_size: int = 248320
    tie_word_embeddings: bool = False


def _seed_backbone_metadata(directory: Path) -> str:
    """Write a backbone inference_metadata.yaml for the MTP writer to extend.

    ``SpeculativeContract`` names workflow components and state cells, so the
    backbone must already publish a ``pipeline.workflow``. This is the shape
    ``write_onnx_genai_config`` emits for a single-component decoder package,
    reduced to what the speculator claim touches: one decoder component and one
    service-group-backed KV cell.
    """
    kv_contract = {
        "dtype": "float16",
        "rank": 4,
        "shape": ["batch", "kv_heads", "sequence", "head_dim"],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    }
    metadata = {
        "schema_version": "v1",
        "pipeline": {
            "workflow": {
                "manifest": {"capabilities": ["workflow_ssa", "serving_service_contract"]},
                "inputs": {
                    "request.input_ids": {
                        "contract": {
                            "dtype": "int64",
                            "rank": 2,
                            "shape": ["batch", "sequence"],
                            "batch_layout": {"kind": "request_aligned", "axis": 0},
                        },
                        "role": {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"},
                        "source": {"kind": "request"},
                        "required": True,
                    },
                    "request.decoder_cache": {
                        "contract": kv_contract,
                        "role": {"kind": "opaque"},
                        "source": {"kind": "application", "name": "decoder_cache"},
                        "required": True,
                    },
                    "request.active": {
                        "contract": {
                            "dtype": "bool",
                            "rank": 1,
                            "shape": ["batch"],
                            "batch_layout": {"kind": "request_aligned", "axis": 0},
                        },
                        "role": {"kind": "opaque"},
                        "source": {"kind": "application", "name": "active"},
                        "required": True,
                    },
                    "request.done": {
                        "contract": {
                            "dtype": "bool",
                            "rank": 1,
                            "shape": ["batch"],
                            "batch_layout": {"kind": "request_aligned", "axis": 0},
                        },
                        "role": {"kind": "opaque"},
                        "source": {"kind": "application", "name": "done"},
                        "required": True,
                    },
                },
                "components": {
                    "decoder": {
                        "implementation": {"kind": "onnx", "artifact": "model.onnx"},
                        "ports": {"roles": {"input_ids": "token_ids", "logits": "logits"}},
                    }
                },
                "state": {
                    "decoder_cache.0": {
                        "contract": kv_contract,
                        "scope": "invocation",
                        "initializer": "request.decoder_cache",
                        "recurrence": {"kind": "invariant"},
                        "service_group": "decoder_cache",
                    }
                },
                "steps": [
                    {
                        "kind": "invoke",
                        "component": "decoder",
                        "inputs": {
                            "input_ids": "request.input_ids",
                            "past_key_values.0.key": "decoder_cache.0",
                        },
                        "outputs": {"logits": "decoder.logits"},
                    }
                ],
                "serving": {
                    "active": "request.active",
                    "done": "request.done",
                    "state_service": {
                        "groups": {
                            "decoder_cache": {
                                "kind": "full_attention",
                                "sequence_axis": 2,
                                "layout": "bnsh",
                                "update": {"kind": "append"},
                                "capabilities": {"snapshot": True, "fork": True},
                                "ports": {
                                    "decoder": {
                                        "decoder_cache.0": {
                                            "input": "past_key_values.0.key",
                                            "output": "present.0.key",
                                        }
                                    }
                                },
                            }
                        }
                    },
                },
            }
        },
    }
    path = directory / "inference_metadata.yaml"
    path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return str(path)


class TestMtpSpeculatorMetadata:
    """The emitted ``speculative`` block conforms to the onnx-genai runtime schema.

    Authoritative source: onnx-genai
    ``crates/onnx-genai-metadata/src/schema/package.rs`` (``SpeculativeContract``,
    ``SpeculativeProposalExecution``, ``SpeculativeVocabulary``) plus
    ``validation.rs`` (``validate_speculative_rollback``).
    """

    def _write(self, tmp_path: Path) -> dict:
        _seed_backbone_metadata(tmp_path)
        out = write_mtp_speculator_metadata(
            str(tmp_path), backbone_config=_MtpBackboneConfig()
        )
        assert out is not None
        with open(out, encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_top_level_key_is_speculative(self, tmp_path):
        meta = self._write(tmp_path)
        # The runtime deserializes InferenceMetadata.speculative; a bare
        # ``speculator`` key is unknown and silently dropped.
        assert "speculative" in meta
        assert "speculator" not in meta

    def test_exact_schema_keys_and_values(self, tmp_path):
        spec = self._write(tmp_path)["speculative"]
        assert spec == {
            "proposer": "mtp",
            "target": "decoder",
            "proposal_execution": {"kind": "block"},
            "port_bindings": {"target_hidden_context": "hidden_states"},
            "shared_weights": ["lm_head.weight_t", "model.embed_tokens.weight"],
            "vocabulary": {"kind": "identical"},
            "max_proposal_width": 1,
            "distribution_preserving": True,
            "rollback_state": ["decoder_cache.0"],
        }

    def test_no_legacy_field_names(self, tmp_path):
        spec = self._write(tmp_path)["speculative"]
        # SpeculativeContract sets ``deny_unknown_fields``, so any of these
        # makes the whole package unparseable rather than being ignored.
        for banned in (
            "proposal_type",
            "num_speculative_tokens",
            "model",
            "model_path",
            "target_hidden_layout",
            "hc_mult",
            "mtp_hidden_output",
            "kv_mode",
            "embedding",
            "lm_head",
            "embedding_weights",
            "lm_head_weights",
            "target_hidden_output",
            "target_hidden_size",
            "hidden_size",
            "vocab_size",
        ):
            assert banned not in spec

    def test_proposer_is_registered_as_a_workflow_component(self, tmp_path):
        workflow = self._write(tmp_path)["pipeline"]["workflow"]
        # proposer/target are workflow component names, so the head has to be
        # declared before it can be referenced.
        assert workflow["components"]["mtp"]["implementation"] == {
            "kind": "onnx",
            "artifact": "mtp/model.onnx",
        }
        roles = workflow["components"]["mtp"]["ports"]["roles"]
        assert roles["hidden_states"] == "hidden_states"
        assert roles["inputs_embeds"] == "inputs_embeds"
        # Only the dedicated post-final-norm seed receives the hidden-state
        # role; hidden_states.N remains the pre-final-norm capture ABI.
        assert (
            workflow["components"]["decoder"]["ports"]["roles"]["mtp_seed"] == "hidden_states"
        )
        assert "hidden_states.63" not in workflow["components"]["decoder"]["ports"]["roles"]

    @pytest.mark.parametrize("dedicated_embedding", [False, True])
    @pytest.mark.parametrize("dedicated_head", [False, True])
    @pytest.mark.parametrize("tied_output", [False, True])
    def test_dedicated_tables_control_ports_and_shared_weights(
        self,
        tmp_path,
        dedicated_embedding,
        dedicated_head,
        tied_output,
    ):
        _seed_backbone_metadata(tmp_path)
        proposer = dataclasses.make_dataclass(
            "ProposerConfig",
            [
                ("use_dedicated_embeddings", bool),
                ("use_dedicated_lm_head", bool),
            ],
        )(dedicated_embedding, dedicated_head)
        out = write_mtp_speculator_metadata(
            str(tmp_path),
            backbone_config=_MtpBackboneConfig(tie_word_embeddings=tied_output),
            proposer_config=proposer,
        )
        assert out is not None
        with open(out, encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle)
        spec = metadata["speculative"]
        roles = metadata["pipeline"]["workflow"]["components"]["mtp"]["ports"]["roles"]

        assert ("input_ids" in roles) is dedicated_embedding
        assert ("inputs_embeds" in roles) is not dedicated_embedding
        assert ("logits" in roles) is dedicated_head
        assert ("mtp_hidden" in roles) is not dedicated_head
        expected_shared = []
        if not dedicated_embedding:
            expected_shared.append("model.embed_tokens.weight")
        if not dedicated_head:
            expected_shared.append(
                "model.embed_tokens.weight" if tied_output else "lm_head.weight_t"
            )
        if expected_shared:
            assert spec["shared_weights"] == sorted(set(expected_shared))
        else:
            assert "shared_weights" not in spec
        with open(_onnx_genai_schema_path()) as handle:
            jsonschema.validate(instance=metadata, schema=json.load(handle))

    def test_quantized_fallback_lists_every_required_initializer(self, tmp_path):
        from mobius._configs import QuantizationConfig

        _seed_backbone_metadata(tmp_path)
        backbone = _MtpBackboneConfig()
        backbone.quantization = QuantizationConfig(
            bits=4,
            group_size=32,
            quant_method="gguf",
            sym=False,
            quantize_embeddings=True,
            quantize_lm_head=True,
        )
        out = write_mtp_speculator_metadata(str(tmp_path), backbone_config=backbone)
        assert out is not None
        with open(out, encoding="utf-8") as handle:
            spec = yaml.safe_load(handle)["speculative"]
        assert spec["shared_weights"] == [
            "lm_head.scales",
            "lm_head.weight",
            "lm_head.zero_points",
            "model.embed_tokens.qweight",
            "model.embed_tokens.scales",
            "model.embed_tokens.zero_points",
        ]

    def test_explicit_lm_head_initializer_overrides_inferred_tying(self, tmp_path):
        _seed_backbone_metadata(tmp_path)
        out = write_mtp_speculator_metadata(
            str(tmp_path),
            backbone_config=_MtpBackboneConfig(tie_word_embeddings=True),
            lm_head_weights="decoder.shared_output.weight",
        )
        assert out is not None
        with open(out, encoding="utf-8") as handle:
            spec = yaml.safe_load(handle)["speculative"]
        assert spec["shared_weights"] == [
            "decoder.shared_output.weight",
            "model.embed_tokens.weight",
        ]

    @pytest.mark.parametrize("dedicated_embedding", [False, True])
    def test_tied_quantized_head_fallback_uses_embedding_initializers(
        self, tmp_path, dedicated_embedding
    ):
        from mobius._configs import QuantizationConfig

        _seed_backbone_metadata(tmp_path)
        backbone = _MtpBackboneConfig(tie_word_embeddings=True)
        backbone.quantization = QuantizationConfig(
            bits=4,
            group_size=32,
            quant_method="gguf",
            sym=False,
            quantize_embeddings=True,
            quantize_lm_head=True,
            tie_word_embeddings=True,
        )
        proposer = dataclasses.make_dataclass(
            "TiedProposerConfig",
            [
                ("use_dedicated_embeddings", bool),
                ("use_dedicated_lm_head", bool),
            ],
        )(dedicated_embedding, False)
        out = write_mtp_speculator_metadata(
            str(tmp_path),
            backbone_config=backbone,
            proposer_config=proposer,
        )
        assert out is not None
        with open(out, encoding="utf-8") as handle:
            spec = yaml.safe_load(handle)["speculative"]
        assert spec["shared_weights"] == [
            "model.embed_tokens.qweight",
            "model.embed_tokens.scales",
            "model.embed_tokens.zero_points",
        ]

    def test_rollback_capacity_covers_the_proposal_width(self, tmp_path):
        workflow = self._write(tmp_path)["pipeline"]["workflow"]
        group = workflow["serving"]["state_service"]["groups"]["decoder_cache"]
        # A rolled-back cell whose group declares no rollback_positions makes
        # the package unloadable, so attaching the speculator states the bound.
        assert group["capabilities"]["rollback_positions"] >= 1

    def test_anchors_to_a_real_mobius_decoder_workflow(self, tmp_path):
        """The target is picked out of a workflow mobius actually emits.

        A real decoder workflow ships a dozen generated policy graphs, every one
        of which is an ONNX component, so the verifier can only be identified by
        its declared ``logits`` role.
        """
        from mobius.integrations.onnx_genai.auto_export_test import _decoder_package
        from mobius.integrations.onnx_genai.workflow_metadata import (
            write_decoder_workflow_metadata,
        )

        package = _decoder_package()
        write_decoder_workflow_metadata(package, str(tmp_path), package.config)
        out = write_mtp_speculator_metadata(
            str(tmp_path), backbone_config=_MtpBackboneConfig()
        )
        assert out is not None
        with open(out, encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle)
        workflow = metadata["pipeline"]["workflow"]
        components = set(workflow["components"])
        assert len(components) > 2, "a real decoder workflow ships policy components"
        assert metadata["speculative"]["target"] == "model"
        assert metadata["speculative"]["proposer"] == "mtp"
        with open(_onnx_genai_schema_path()) as handle:
            jsonschema.validate(instance=metadata, schema=json.load(handle))

    def test_requires_a_workflow_to_anchor_against(self, tmp_path):
        (tmp_path / "inference_metadata.yaml").write_text(yaml.safe_dump({}), encoding="utf-8")
        with pytest.raises(ValueError, match=re.escape("pipeline.workflow")):
            write_mtp_speculator_metadata(str(tmp_path), backbone_config=_MtpBackboneConfig())

    def test_matches_onnx_genai_json_schema(self, tmp_path):
        """Emitted metadata validates against onnx-genai's published schema."""
        with open(_onnx_genai_schema_path()) as handle:
            schema = json.load(handle)
        meta = self._write(tmp_path)
        jsonschema.validate(instance=meta, schema=schema)


class TestInputAdmissionIsDerivedNotDefaulted:
    """A package states what a caller must attach; a reader never guesses it.

    ``required`` decides admission: an input a runtime believes is required and
    the caller did not attach rejects the request on *every* path, before any
    component runs. A consumer that reads a declaration with no ``required``
    key has to pick a default, and the one the schema picks is ``true`` -- the
    opposite of what omission means to a producer whose workflow computes,
    defaults, or presence-gates the value. These tests pin the derivation so a
    branch input can never again be published as a universal obligation.
    """

    @staticmethod
    def _workflow(**inputs):
        return {
            "inputs": inputs,
            "steps": [
                {
                    "kind": "branch",
                    "predicate": "request.thing_present",
                    "cases": {
                        "true": {
                            "kind": "invoke",
                            "component": "use",
                            "inputs": {"tensor": "request.thing"},
                            "outputs": {"out": "used"},
                        },
                        "false": {
                            "kind": "invoke",
                            "component": "make",
                            "inputs": {"seed": "request.seed"},
                            "outputs": {"out": "made"},
                        },
                    },
                    "outputs": {"value": {"cases": {"true": "used", "false": "made"}}},
                },
                {
                    "kind": "invoke",
                    "component": "head",
                    "inputs": {"value": "value", "prompt": "request.prompt"},
                    "outputs": {"out": "result"},
                },
                {"kind": "emit", "value": "result", "output": "result", "mode": "replace"},
            ],
        }

    def test_a_presence_gated_branch_input_is_not_a_universal_obligation(self):
        """The failure this whole invariant exists for.

        ``request.thing`` is read by exactly one branch case, and the other case
        builds the value instead. A caller on the generating path has nothing to
        attach, so admitting against it rejects a request the workflow can serve.
        """
        workflow = self._workflow(
            **{
                "request.prompt": {"source": {"kind": "request"}},
                "request.seed": {"source": {"kind": "request"}, "default": 0},
                "request.thing": {
                    "source": {"kind": "application", "name": "thing"},
                    "present_as": "request.thing_present",
                },
            }
        )
        declare_input_admission(workflow)
        assert workflow["inputs"]["request.thing"]["required"] is False
        assert workflow["inputs"]["request.seed"]["required"] is False
        assert workflow["inputs"]["request.prompt"]["required"] is True

    def test_every_declaration_publishes_admission_rather_than_implying_it(self):
        workflow = self._workflow(
            **{
                "request.prompt": {"source": {"kind": "request"}},
                "request.seed": {"source": {"kind": "literal"}, "default": 0},
                "request.thing": {
                    "source": {"kind": "application", "name": "thing"},
                    "present_as": "request.thing_present",
                },
            }
        )
        declare_input_admission(workflow)
        assert all("required" in declaration for declaration in workflow["inputs"].values())

    def test_a_package_supplied_source_is_never_a_caller_obligation(self):
        workflow = self._workflow(
            **{
                "request.prompt": {"source": {"kind": "request"}},
                "package.seed": {"source": {"kind": "literal"}, "default": 0},
                "request.thing": {
                    "source": {"kind": "application", "name": "thing"},
                    "present_as": "request.thing_present",
                },
            }
        )
        declare_input_admission(workflow)
        assert workflow["inputs"]["package.seed"]["required"] is False

    def test_a_package_literal_that_carries_no_value_fails_closed(self):
        """A literal source is bound from its own default and from nothing else.

        Declaring one without a default names a value the package does not hold
        and cannot ask a caller for, so it is neither required nor optional --
        it is unbindable, and saying so here beats discovering it mid-execution.
        """
        workflow = self._workflow(
            **{
                "request.prompt": {"source": {"kind": "request"}},
                "package.seed": {"source": {"kind": "literal"}},
            }
        )
        with pytest.raises(ValueError, match="nothing ever binds it"):
            declare_input_admission(workflow)

    def test_declaring_both_an_escape_and_an_obligation_fails_closed(self):
        workflow = self._workflow(
            **{
                "request.prompt": {"source": {"kind": "request"}},
                "request.seed": {"source": {"kind": "request"}, "default": 0},
                "request.thing": {
                    "source": {"kind": "application", "name": "thing"},
                    "present_as": "request.thing_present",
                    "required": True,
                },
            }
        )
        with pytest.raises(ValueError, match="already proceeds without it"):
            declare_input_admission(workflow)

    def test_relaxing_an_obligation_without_a_way_to_proceed_fails_closed(self):
        """The tempting non-fix -- flip ``required`` -- is refused at the source.

        Admission is the only place a missing value is reported cleanly. An
        input marked optional that the workflow has no default, package source
        or presence gate for does not become optional; the request is admitted
        and then reads an unbound value part-way through, which is a worse
        failure than the rejection it replaced.
        """
        workflow = self._workflow(
            **{
                "request.prompt": {"source": {"kind": "request"}},
                "request.thing": {
                    "source": {"kind": "application", "name": "thing"},
                    "required": False,
                },
            }
        )
        with pytest.raises(ValueError, match="no defined behaviour"):
            declare_input_admission(workflow)

    def test_a_presence_gate_no_step_reads_does_not_make_absence_executable(self):
        workflow = self._workflow(
            **{
                "request.prompt": {"source": {"kind": "request"}},
                "request.seed": {"source": {"kind": "request"}, "default": 0},
                "request.thing": {
                    "source": {"kind": "application", "name": "thing"},
                    "present_as": "request.thing_supplied",
                },
            }
        )
        with pytest.raises(ValueError, match="no step reads"):
            declare_input_admission(workflow)

    def test_derivation_reads_every_nested_position_a_value_can_occupy(self):
        """Positions this misses are values a package can silently declare dead.

        The set is deliberately exhaustive over the schema rather than over the
        walker: a bounded cell reads its own extent, and a state-service group
        reads its fixed update capacity, both of which are workflow values and
        neither of which appears in ``steps``.
        """
        workflow = {
            "inputs": {},
            "state": {
                "cell": {
                    "initializer": "package.zero",
                    "recurrence": {
                        "kind": "bounded",
                        "axis": 1,
                        "max": "package.limit",
                        "increment": "package.one_step",
                    },
                }
            },
            "serving": {
                "active": "package.active_rows",
                "state_service": {
                    "groups": {
                        "g": {
                            "ports": {"c": {"past": "past"}},
                            "update": {"capacity": "package.capacity"},
                        }
                    }
                },
            },
            "steps": [
                {
                    "kind": "loop",
                    "setup": [
                        {
                            "kind": "invoke",
                            "component": "c",
                            "inputs": {"a": "setup.value"},
                            "outputs": {"b": "seeded"},
                        }
                    ],
                    "steps": [
                        {
                            "kind": "emit",
                            "value": "row",
                            "output": "tokens",
                            "mode": "append",
                            "when": "package.active",
                            "valid_length": "row.length",
                        }
                    ],
                    "continue_when": "loop_active",
                    "max_iterations": "request.max_iterations",
                    "carried": [
                        {"cell": "cell", "next": "cell.next", "initial": "cell.first"}
                    ],
                }
            ],
        }
        assert published_value_references(workflow) == {
            "setup.value",
            "row",
            "package.active",
            "row.length",
            "loop_active",
            "request.max_iterations",
            "cell.next",
            "cell.first",
            "package.zero",
            "package.limit",
            "package.one_step",
            "package.active_rows",
            "package.capacity",
        }
        # Port names and the recurrence discriminator are not values.
        assert {"past", "bounded"}.isdisjoint(published_value_references(workflow))
