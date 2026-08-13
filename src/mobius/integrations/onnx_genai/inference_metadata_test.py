# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for onnx-genai diffusion inference_metadata generation."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
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
    add_explicit_package_io,
    add_policy_components_to_workflow,
    build_diffusion_pipeline_metadata,
    build_multimodal_pipeline_metadata,
    build_native_vlm_package_metadata,
    is_native_vlm_package,
    load_diffusers_scheduler_config,
    validate_executable_closure,
    write_diffusion_pipeline_metadata,
    write_native_vlm_package_metadata,
)


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
                "manifest": {"ir_version": "1.0"},
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
    assert set(component["ports"]["inputs"]) == {"logits"}
    assert set(component["ports"]["outputs"]) == {"token"}
    assert component["contract"] == {
        "id": "onnx-genai.token-sampler",
        "version": "1",
        "bindings": {"logits": "logits", "token": "token"},
        "parameters": {"mode": "greedy"},
    }
    assert component["effects"] == ["sample"]
    assert (tmp_path / component["implementation"]["artifact"]).is_file()


def _onnx_genai_schema_path() -> str | None:
    """Locate onnx-genai's committed pipeline JSON schema, if available."""
    candidates = [
        os.environ.get("ONNX_GENAI_SCHEMA"),
        os.path.join(
            os.path.dirname(__file__),
            "../../../../../onnx-genai/schema/inference_metadata.schema.json",
        ),
        "/home/justinchu/onnx-genai/schema/inference_metadata.schema.json",
        os.path.expanduser(
            "~/Documents/GitHub/onnx-genai/schema/inference_metadata.schema.json"
        ),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


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


class TestExplicitPackageIo:
    def test_emits_explicit_dynamic_decoder_roles(self):
        model = _decoder_model(
            [],
            position_shape=["batch", "sequence"],
            raw_token_input=True,
            kv_head_dims=[8, 16],
        )
        metadata = add_explicit_package_io({"model": {}}, {"model": model}, _VlmConfig())
        io = metadata["model"]["io"]
        assert io["token_input"] == "input_ids"
        assert io["sequence_source"] == "token_ids"
        assert io["logits_output"] == "logits"
        assert io["kv_ownership"] == "owned"
        assert io["kv_inputs"] == [
            "past_key_values.0.key",
            "past_key_values.0.value",
            "past_key_values.1.key",
            "past_key_values.1.value",
        ]
        assert io["kv_outputs"] == [
            "present.0.key",
            "present.0.value",
            "present.1.key",
            "present.1.value",
        ]

    def test_emits_explicit_static_cache_roles_in_layer_order(self):
        metadata = add_explicit_package_io(
            {"model": {}}, {"model": _static_cache_decoder_model()}, _VlmConfig()
        )
        io = metadata["model"]["io"]
        assert io["static_cache"] == {
            "write_indices_input": "write_indices",
            "kv_sequence_length_input": "nonpad_kv_seqlen",
            "key_cache_inputs": ["key_cache.1", "key_cache.3"],
            "value_cache_inputs": ["value_cache.1", "value_cache.3"],
            "key_cache_outputs": ["updated_key_cache.1", "updated_key_cache.3"],
            "value_cache_outputs": [
                "updated_value_cache.1",
                "updated_value_cache.3",
            ],
        }
        assert "kv_inputs" not in io

    @pytest.mark.parametrize(
        ("encoder_input", "role_field"),
        [
            (
                _value(
                    "mel_prompt",
                    ir.DataType.FLOAT,
                    ["batch", 80, "audio_sequence"],
                ),
                "audio_features_input",
            ),
            (
                _value("prompt_tokens", ir.DataType.INT64, ["batch", "sequence"]),
                "token_input",
            ),
        ],
    )
    def test_emits_explicit_encoder_prompt_role(self, encoder_input, role_field):
        encoder = _model(
            "encoder",
            [encoder_input],
            [
                (
                    "encoder_hidden_states",
                    ir.DataType.FLOAT,
                    ["batch", "encoder_sequence", 64],
                )
            ],
        )
        decoder = _model(
            "decoder",
            [
                _value("decoder_tokens", ir.DataType.INT64, ["batch", "sequence"]),
                _value(
                    "encoder_hidden_states",
                    ir.DataType.FLOAT,
                    ["batch", "encoder_sequence", 64],
                ),
            ],
            [("logits", ir.DataType.FLOAT, ["batch", "sequence", 128])],
        )
        metadata = {
            "pipeline": {
                "models": {
                    "encoder": {"type": "encoder"},
                    "decoder": {"type": "decoder"},
                },
                "dataflow": [],
                "strategy": {"kind": "autoregressive", "decoder": "decoder"},
            }
        }
        add_explicit_package_io(
            metadata, {"encoder": encoder, "decoder": decoder}, _VlmConfig()
        )
        encoder_io = metadata["pipeline"]["models"]["encoder"]["io"]
        assert encoder_io[role_field] == encoder_input.name
        decoder_io = metadata["pipeline"]["models"]["decoder"]["io"]
        assert decoder_io["token_input"] == "decoder_tokens"
        assert decoder_io["encoder_hidden_states_input"] == "encoder_hidden_states"
        assert decoder_io["logits_output"] == "logits"

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

    def test_explicit_inner_embedding_output_is_preserved(self):
        metadata = self._nested_metadata("declared_embedding")
        package = self._nested_package(
            [("logits", ir.DataType.FLOAT, ["batch", "sequence", 128])]
        )
        add_explicit_package_io(metadata, package, _VlmConfig())
        assert (
            metadata["pipeline"]["strategy"]["inner_embedding_output"] == "declared_embedding"
        )

    def test_missing_inner_embedding_output_is_derived(self):
        metadata = self._nested_metadata()
        package = self._nested_package(
            [
                ("logits", ir.DataType.FLOAT, ["batch", "sequence", 128]),
                ("codec_embeddings", ir.DataType.FLOAT, [16, 64]),
            ]
        )
        add_explicit_package_io(metadata, package, _VlmConfig())
        assert metadata["pipeline"]["strategy"]["inner_embedding_output"] == "codec_embeddings"

    def test_ambiguous_inner_embedding_output_fails_actionably(self):
        metadata = self._nested_metadata()
        package = self._nested_package(
            [
                ("logits", ir.DataType.FLOAT, ["batch", "sequence", 128]),
                ("first_embedding", ir.DataType.FLOAT, [16, 64]),
                ("second_embedding", ir.DataType.FLOAT, [16, 64]),
            ]
        )
        with pytest.raises(
            ValueError,
            match=r"pipeline\.strategy\.inner_embedding_output.*no unique",
        ):
            add_explicit_package_io(metadata, package, _VlmConfig())


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
    def _validate(self, metadata: dict) -> None:
        schema_path = _onnx_genai_schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found (set ONNX_GENAI_SCHEMA)")
        with open(schema_path, encoding="utf-8") as handle:
            jsonschema.validate(instance=metadata, schema=json.load(handle))

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
        self._validate(metadata)
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
        assert image_outputs == [
            {
                "name": "vision_encoder.pixel_values",
                "content": "pixels",
                "dtype": "fp32",
            },
            {
                "name": "vision_encoder.pixel_position_ids",
                "content": "patch_coordinates",
                "dtype": "int64",
                "pad_value": -1,
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
        self._validate(metadata)
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
        self._validate(metadata)
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
        assert cached_config is not None, "cached Gemma4 assistant config.json unavailable"
        assert json.loads(Path(cached_config).read_text(encoding="utf-8"))["model_type"] == (
            "gemma4_assistant"
        )
        processor_cache = (
            Path(constants.HF_HUB_CACHE)
            / repo_folder_name(repo_id="google/gemma-4-E2B-it", repo_type="model")
            / "snapshots"
        )
        processor_path = max(
            processor_cache.glob("*/processor_config.json"),
            key=lambda path: path.stat().st_mtime,
        )
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


class TestBuildDiffusionPipelineMetadata:
    def test_denoiser_only_minimal(self):
        meta = build_diffusion_pipeline_metadata(num_inference_steps=20)
        pipe = meta["pipeline"]
        assert set(pipe["models"]) == {"denoiser"}
        assert pipe["strategy"]["kind"] == "iterative"
        assert pipe["strategy"]["denoiser"] == "denoiser"
        assert pipe["strategy"]["num_steps"] == 20
        assert pipe["strategy"]["timestep_input"] == "timestep"
        # Loop-carried self-edge is present.
        assert {"from": "denoiser.noise_pred", "to": "denoiser.sample"} in pipe["dataflow"]
        # Default DDIM scheduler config.
        sched = pipe["strategy"]["scheduler_config"]
        assert sched["kind"] == "ddim"
        assert sched["num_train_timesteps"] == 1000

    def test_full_pipeline_with_vae_and_text_encoder(self):
        meta = build_diffusion_pipeline_metadata(
            num_inference_steps=4,
            vae_filename="vae.onnx",
            text_encoder_filename="text_encoder.onnx",
            guidance_scale=7.5,
        )
        pipe = meta["pipeline"]
        assert set(pipe["models"]) == {"denoiser", "vae", "text_encoder"}
        # Text encoder feeds conditioning (prompt-phase); VAE decodes final latent.
        assert {
            "from": "text_encoder.last_hidden_state",
            "to": "denoiser.encoder_hidden_states",
        } in pipe["dataflow"]
        assert {"from": "denoiser.sample", "to": "vae.latent"} in pipe["dataflow"]
        assert pipe["phases"]["text_encoder"] == {"run_on": "prompt_only"}
        assert pipe["phases"]["vae"] == {"run_on": "final_only"}
        # CFG enabled -> conditioning input declared for the unconditional pass.
        assert pipe["strategy"]["guidance_scale"] == pytest.approx(7.5)
        assert pipe["strategy"]["cfg_conditioning_input"] == "encoder_hidden_states"

    def test_sdxl_dual_conditioning_edges(self):
        meta = build_diffusion_pipeline_metadata(
            num_inference_steps=4,
            vae_filename="vae.onnx",
            text_encoder_filename="text_encoder.onnx",
            guidance_scale=7.5,
            text_encoder_edges=[
                ("encoder_hidden_states", "encoder_hidden_states"),
                ("text_embeds", "text_embeds"),
            ],
        )
        flow = meta["pipeline"]["dataflow"]
        assert {
            "from": "text_encoder.encoder_hidden_states",
            "to": "denoiser.encoder_hidden_states",
        } in flow
        assert {"from": "text_encoder.text_embeds", "to": "denoiser.text_embeds"} in flow

    def test_guidance_scale_one_does_not_enable_cfg(self):
        meta = build_diffusion_pipeline_metadata(num_inference_steps=2, guidance_scale=1.0)
        assert "cfg_conditioning_input" not in meta["pipeline"]["strategy"]

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

    def test_rejects_zero_steps(self):
        with pytest.raises(ValueError):
            build_diffusion_pipeline_metadata(num_inference_steps=0)

    def test_write_roundtrips_yaml(self, tmp_path):
        path = write_diffusion_pipeline_metadata(
            str(tmp_path), num_inference_steps=3, vae_filename="vae.onnx"
        )
        with open(path) as handle:
            loaded = yaml.safe_load(handle)
        assert loaded["pipeline"]["strategy"]["num_steps"] == 3
        assert "vae" in loaded["pipeline"]["models"]

    def test_matches_onnx_genai_json_schema(self):
        """The emitted metadata validates against onnx-genai's published schema."""
        schema_path = _onnx_genai_schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found (set ONNX_GENAI_SCHEMA)")
        import json

        import jsonschema

        with open(schema_path) as handle:
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
