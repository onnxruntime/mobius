# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the compositional pipeline core."""

from __future__ import annotations

import dataclasses
import json
import os
from unittest import mock

import onnx_ir as ir
import pytest

from mobius._model_package import ModelPackage
from mobius._pipeline import (
    LOOP_CARRIED_STATE_CAPABILITY,
    PIPELINE_FILENAME,
    GeneratedInputRule,
    InputSource,
    PipelineAsset,
    PipelineBuilder,
    PipelineComponent,
    PipelineConnection,
    PipelineInput,
    PipelineManifest,
    PipelineOutput,
    PipelinePackage,
    PipelinePort,
    PipelineProfile,
    PipelineStage,
    PipelineValidationError,
    register_generated_input,
    register_role,
    register_strategy,
    register_transform,
    transform_definition,
)

Dim = int | str


def _value(name: str, shape: list[Dim], dtype: ir.DataType = ir.DataType.FLOAT) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _make_model(
    inputs: dict[str, list[Dim]],
    outputs: dict[str, list[Dim]],
    *,
    dtype: ir.DataType = ir.DataType.FLOAT,
    name: str = "g",
) -> ir.Model:
    """Build a tiny well-formed ``ir.Model`` with the requested signature."""
    input_values = [_value(n, s, dtype) for n, s in inputs.items()]
    nodes = []
    output_values = []
    for out_name, shape in outputs.items():
        node = ir.Node("", "Identity", inputs=[input_values[0]], num_outputs=1)
        out = node.outputs[0]
        out.name = out_name
        out.type = ir.TensorType(dtype)
        out.shape = ir.Shape(shape)
        nodes.append(node)
        output_values.append(out)
    graph = ir.Graph(
        input_values, output_values, nodes=nodes, name=name, opset_imports={"": 24}
    )
    return ir.Model(graph, ir_version=10)


def _encoder() -> ir.Model:
    return _make_model(
        {"pixel_values": [1, 3, 8, 8]}, {"image_features": ["batch", "tokens", 16]}
    )


def _decoder() -> ir.Model:
    return _make_model(
        {"image_features": ["b", "t", 16], "position_ids": ["b", "t"]},
        {"logits": ["b", "t", 32]},
    )


def _simple_pipeline() -> PipelineBuilder:
    """Encoder -> decoder, one external input, one generated input."""
    builder = PipelineBuilder()
    builder.add_model("encoder", _encoder(), role="encoder")
    builder.add_model("decoder", _decoder(), role="decoder")
    builder.connect("encoder.image_features", "decoder.image_features")
    builder.declare_external("encoder.pixel_values")
    builder.declare_generated("decoder.position_ids", generator="zeros")
    builder.add_stage("encode", "single_pass", ["encoder"])
    builder.add_stage("generate", "autoregressive", ["decoder"])
    builder.add_public_output("decoder.logits")
    return builder


def _cosmos_style_pipeline() -> PipelineBuilder:
    """VAE moments -> sampled generator tokens, with an iterative generate stage."""
    builder = PipelineBuilder()
    builder.add_model(
        "vae",
        _make_model({"video": [1, 3, 8, 8]}, {"moments": ["b", 8, "h", "w"]}),
        role="encoder",
    )
    builder.add_model(
        "generator",
        _make_model({"tokens": ["b", "t", 16]}, {"latent": ["b", "t", 16]}),
        role="dynamics",
    )
    builder.connect("vae.moments", "generator.tokens", transform="sample")
    builder.declare_external("vae.video")
    builder.add_stage("encode", "single_pass", ["vae"])
    builder.add_stage("generate", "iterative", ["generator"])
    builder.add_public_output("generator.latent")
    return builder


class TestPipelinePort:
    def test_qualified_round_trip(self):
        port = PipelinePort("decoder", "logits")
        assert port.qualified == "decoder.logits"
        assert PipelinePort.parse("decoder.logits") == port

    def test_dotted_port_name_survives(self):
        port = PipelinePort.parse("decoder.past_key_values.0.key")
        assert port.component == "decoder"
        assert port.port == "past_key_values.0.key"

    def test_parse_requires_separator(self):
        with pytest.raises(PipelineValidationError, match=r"component\.port"):
            PipelinePort.parse("logits")

    @pytest.mark.parametrize(
        "name", ["", " ", "a b ".strip() + "/x", "a\\b", "..", "a..b", "a.b", "nul"]
    )
    def test_unsafe_component_names_rejected(self, name):
        with pytest.raises(PipelineValidationError):
            PipelinePort(name, "x")

    def test_blank_port_rejected(self):
        with pytest.raises(PipelineValidationError):
            PipelinePort("encoder", "")


class TestComposition:
    def test_multiple_tiny_models(self):
        pkg = _simple_pipeline().build()
        assert isinstance(pkg, ModelPackage)
        assert sorted(pkg) == ["decoder", "encoder"]
        assert pkg.manifest.component_names == ("decoder", "encoder")
        assert pkg.manifest.component("encoder").role == "encoder"
        assert [o.name for o in pkg.manifest.outputs] == ["logits"]

    def test_component_wraps_its_model(self):
        pkg = _simple_pipeline().build()
        component = pkg.manifest.component("encoder")
        assert component.model is pkg["encoder"]
        assert component.inputs[0].name == "pixel_values"
        assert component.inputs[0].dtype == "FLOAT"
        assert component.outputs[0].shape == ("batch", "tokens", 16)

    def test_input_source_classification(self):
        manifest = _simple_pipeline().build().manifest
        assert manifest.source_of("decoder.image_features") == InputSource.DATAFLOW
        assert manifest.source_of("encoder.pixel_values") == InputSource.EXTERNAL
        assert manifest.source_of("decoder.position_ids") == InputSource.GENERATED
        assert [i.port.qualified for i in manifest.external_inputs] == ["encoder.pixel_values"]

    def test_deterministic_serialization(self):
        first = _simple_pipeline().build().manifest
        builder = PipelineBuilder()
        # Declare everything in a different order.
        builder.add_model("decoder", _decoder(), role="decoder")
        builder.add_model("encoder", _encoder(), role="encoder")
        builder.declare_generated("decoder.position_ids", generator="zeros")
        builder.declare_external("encoder.pixel_values")
        builder.connect("encoder.image_features", "decoder.image_features")
        builder.add_stage("encode", "single_pass", ["encoder"])
        builder.add_stage("generate", "autoregressive", ["decoder"])
        builder.add_public_output("decoder.logits")
        assert builder.build().manifest.to_json() == first.to_json()

    def test_manifest_json_round_trip(self):
        manifest = _simple_pipeline().build().manifest
        assert PipelineManifest.from_json(manifest.to_json()) == manifest

    def test_builder_does_not_mutate_graphs(self):
        model = _encoder()
        before = len(model.graph)
        builder = PipelineBuilder()
        builder.add_model("encoder", model, role="encoder")
        builder.declare_external("encoder.pixel_values")
        builder.add_stage("encode", "single_pass", ["encoder"])
        builder.add_public_output("encoder.image_features")
        builder.build()
        assert len(model.graph) == before


class TestAddPackage:
    def test_multi_model_package_is_namespaced(self):
        package = ModelPackage(
            {
                "model": _make_model({"x": [1, 4]}, {"y": [1, 4]}),
                "vision": _make_model({"pixels": [1, 4]}, {"feats": [1, 4]}),
            }
        )
        builder = PipelineBuilder()
        created = builder.add_package(
            package, roles={"model": "decoder", "vision": "encoder"}, prefix="left"
        )
        assert [c.name for c in created] == ["left_model", "left_vision"]
        assert created[0].role == "decoder"
        assert created[0].metadata["package_key"] == "model"

    def test_two_packages_can_coexist(self):
        def package():
            return ModelPackage({"model": _make_model({"x": [1, 4]}, {"y": [1, 4]})})

        builder = PipelineBuilder()
        builder.add_package(package(), roles={"model": "encoder"}, prefix="left")
        builder.add_package(package(), roles={"model": "decoder"}, prefix="right")
        builder.connect("left_model.y", "right_model.x")
        builder.declare_external("left_model.x")
        builder.add_stage("run", "single_pass", ["left_model", "right_model"])
        builder.add_public_output("right_model.y")
        pkg = builder.build()
        assert pkg.manifest.component_names == ("left_model", "right_model")

    def test_missing_role_rejected(self):
        package = ModelPackage({"model": _make_model({"x": [1, 4]}, {"y": [1, 4]})})
        builder = PipelineBuilder()
        with pytest.raises(PipelineValidationError, match="No role declared"):
            builder.add_package(package, roles={"other": "encoder"})

    def test_role_callable_ignores_literal_key_semantics(self):
        package = ModelPackage(
            {
                "vision": _make_model({"x": [1, 4]}, {"y": [1, 4]}),
                "model": _make_model({"x": [1, 4]}, {"y": [1, 4]}),
            }
        )
        builder = PipelineBuilder()
        created = builder.add_package(package, roles=lambda key: "generic")
        assert {c.role for c in created} == {"generic"}
        assert [c.name for c in created] == ["model", "vision"]

    def test_duplicate_component_rejected(self):
        builder = PipelineBuilder()
        builder.add_model("encoder", _encoder(), role="encoder")
        with pytest.raises(PipelineValidationError, match="already registered"):
            builder.add_model("encoder", _encoder(), role="encoder")


class TestConnectionValidation:
    def test_fan_out_accepted(self):
        builder = PipelineBuilder()
        builder.add_model("src", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="decoder")
        builder.add_model("b", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="decoder")
        builder.connect("src.y", "a.x")
        builder.connect("src.y", "b.x")
        builder.declare_external("src.x")
        builder.add_stage("run", "single_pass", ["src", "a", "b"])
        builder.add_public_output("a.y", alias="a_out")
        builder.add_public_output("b.y", alias="b_out")
        manifest = builder.build().manifest
        targets = [c.target.qualified for c in manifest.connections]
        assert sorted(targets) == ["a.x", "b.x"]

    def test_duplicate_producer_rejected(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.add_model("b", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.add_model("c", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="decoder")
        builder.connect("a.y", "c.x")
        builder.connect("b.y", "c.x")
        builder.declare_external("a.x")
        builder.declare_external("b.x", alias="b_x")
        builder.add_stage("run", "single_pass", ["a", "b", "c"])
        with pytest.raises(PipelineValidationError, match=r"more than one .*producer"):
            builder.build()

    def test_unknown_component_endpoint_rejected(self):
        builder = _simple_pipeline()
        builder.connect("ghost.y", "decoder.position_ids")
        with pytest.raises(PipelineValidationError, match="unknown component 'ghost'"):
            builder.build()

    def test_unknown_port_rejected(self):
        builder = _simple_pipeline()
        builder.connect("encoder.nope", "decoder.position_ids")
        with pytest.raises(PipelineValidationError, match="unknown output 'nope'"):
            builder.build()

    def test_symbolic_dimension_name_mismatch_accepted(self):
        builder = PipelineBuilder()
        builder.add_model(
            "a", _make_model({"x": [1, 4]}, {"y": ["batch", "seq", 8]}), role="encoder"
        )
        builder.add_model(
            "b", _make_model({"h": ["n", "tokens", 8]}, {"y": [1, 4]}), role="decoder"
        )
        builder.connect("a.y", "b.h")
        builder.declare_external("a.x")
        builder.add_stage("run", "single_pass", ["a", "b"])
        builder.add_public_output("b.y")
        assert builder.build().manifest.source_of("b.h") == InputSource.DATAFLOW

    def test_concrete_dimension_mismatch_rejected(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": ["b", 8]}), role="encoder")
        builder.add_model("b", _make_model({"h": ["b", 16]}, {"y": [1, 4]}), role="decoder")
        builder.connect("a.y", "b.h")
        builder.declare_external("a.x")
        builder.add_stage("run", "single_pass", ["a", "b"])
        with pytest.raises(PipelineValidationError, match="dim 1: 8 != 16"):
            builder.build()

    def test_rank_mismatch_rejected(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": ["b", 8]}), role="encoder")
        builder.add_model("b", _make_model({"h": ["b", 1, 8]}, {"y": [1, 4]}), role="decoder")
        builder.connect("a.y", "b.h")
        builder.declare_external("a.x")
        builder.add_stage("run", "single_pass", ["a", "b"])
        with pytest.raises(PipelineValidationError, match="rank 2 != 3"):
            builder.build()

    def test_dtype_mismatch_rejected(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": ["b", 8]}), role="encoder")
        builder.add_model(
            "b",
            _make_model({"h": ["b", 8]}, {"y": [1, 4]}, dtype=ir.DataType.INT64),
            role="decoder",
        )
        builder.connect("a.y", "b.h")
        builder.declare_external("a.x")
        builder.add_stage("run", "single_pass", ["a", "b"])
        with pytest.raises(PipelineValidationError, match="dtype FLOAT != INT64"):
            builder.build()

    def test_registered_transform_allows_rank_and_dtype_change(self):
        """A VAE-style edge: FLOAT [b, 8, h, w] moments -> INT64 [b, t] tokens."""
        builder = PipelineBuilder()
        builder.add_model(
            "vae",
            _make_model({"x": [1, 4]}, {"moments": ["b", 8, "h", "w"]}),
            role="encoder",
        )
        builder.add_model(
            "generator",
            _make_model({"tokens": ["b", "t"]}, {"y": [1, 4]}, dtype=ir.DataType.INT64),
            role="decoder",
        )
        builder.connect("vae.moments", "generator.tokens", transform="patchify")
        builder.declare_external("vae.x")
        builder.add_stage("run", "single_pass", ["vae", "generator"])
        builder.add_public_output("generator.y")
        manifest = builder.build().manifest
        connection = manifest.connections[0]
        assert connection.transform == "patchify"
        assert connection.transform_capabilities == ("tensor_patchify",)
        assert "tensor_patchify" in manifest.required_capabilities

    def test_unknown_transform_rejected(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": ["b", 8]}), role="encoder")
        builder.add_model("b", _make_model({"h": ["b", 16]}, {"y": [1, 4]}), role="decoder")
        with pytest.raises(PipelineValidationError, match="Unknown transform 'tile'"):
            builder.connect("a.y", "b.h", transform="tile")

    def test_rank_mismatch_rejected_without_transform(self):
        """The same edge that a transform makes legal is illegal untransformed."""
        builder = PipelineBuilder()
        builder.add_model(
            "vae", _make_model({"x": [1, 4]}, {"moments": ["b", 8, "h", "w"]}), role="encoder"
        )
        builder.add_model(
            "generator", _make_model({"tokens": ["b", "t"]}, {"y": [1, 4]}), role="decoder"
        )
        builder.connect("vae.moments", "generator.tokens")
        builder.declare_external("vae.x")
        builder.add_stage("run", "single_pass", ["vae", "generator"])
        with pytest.raises(PipelineValidationError, match="rank 4 != 2"):
            builder.build()

    def test_transform_still_validates_endpoints(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": ["b", 8]}), role="encoder")
        builder.add_model("b", _make_model({"h": ["b", 16]}, {"y": [1, 4]}), role="decoder")
        builder.connect("a.ghost", "b.h", transform="reshape")
        builder.declare_external("a.x")
        builder.add_stage("run", "single_pass", ["a", "b"])
        with pytest.raises(PipelineValidationError, match="unknown output 'ghost'"):
            builder.build()

    def test_transform_still_enforces_single_producer(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.add_model("b", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.add_model("c", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="decoder")
        builder.connect("a.y", "c.x", transform="reshape")
        builder.connect("b.y", "c.x", transform="cast")
        builder.declare_external("a.x")
        builder.declare_external("b.x", alias="b_x")
        builder.add_stage("run", "single_pass", ["a", "b", "c"])
        with pytest.raises(PipelineValidationError, match=r"more than one .*producer"):
            builder.build()

    def test_transform_capabilities_must_be_declared_in_manifest(self):
        manifest = _cosmos_style_pipeline().build().manifest
        with pytest.raises(PipelineValidationError, match="requires capabilities"):
            dataclasses.replace(manifest, required_capabilities=())

    def test_recurrent_transform_edge(self):
        """A scheduler step feeding the next latent back into the denoiser."""
        model = _make_model(
            {"latent": ["b", 16], "timestep": ["b"]},
            {"noise_pred": ["b", 16]},
        )
        builder = PipelineBuilder()
        builder.add_model("denoiser", model, role="dynamics", run_on="step")
        builder.connect(
            "denoiser.noise_pred",
            "denoiser.latent",
            recurrent=True,
            transform="scheduler_step",
        )
        builder.declare_generated("denoiser.latent", generator="zeros")
        builder.declare_generated(
            "denoiser.timestep",
            generator="scheduler_timesteps",
            parameters={"stage": "denoise"},
        )
        builder.add_stage("denoise", "iterative", ["denoiser"], run_on="step")
        builder.add_public_output("denoiser.noise_pred")
        manifest = builder.build().manifest
        assert manifest.source_of("denoiser.latent") == InputSource.STATEFUL
        assert set(manifest.required_capabilities) == {
            LOOP_CARRIED_STATE_CAPABILITY,
            "iterative_scheduler",
        }

    def test_untyped_graph_rejected(self):
        model = _make_model({"x": [1, 4]}, {"y": [1, 4]})
        model.graph.inputs[0].shape = None
        builder = PipelineBuilder()
        with pytest.raises(PipelineValidationError, match="has no shape"):
            builder.add_model("a", model, role="encoder")


class TestInputSources:
    def _pipeline_without_position_ids(self) -> PipelineBuilder:
        """Encoder -> decoder where ``decoder.position_ids`` has no source yet."""
        builder = PipelineBuilder()
        builder.add_model("encoder", _encoder(), role="encoder")
        builder.add_model("decoder", _decoder(), role="decoder")
        builder.connect("encoder.image_features", "decoder.image_features")
        builder.declare_external("encoder.pixel_values")
        builder.add_stage("encode", "single_pass", ["encoder"])
        builder.add_stage("generate", "autoregressive", ["decoder"])
        builder.add_public_output("decoder.logits")
        return builder

    def test_missing_input_source_rejected(self):
        with pytest.raises(PipelineValidationError, match="has no initial source"):
            self._pipeline_without_position_ids().build()

    def test_connected_and_declared_is_conflicting(self):
        builder = _simple_pipeline()
        builder.declare_external("decoder.image_features")
        with pytest.raises(PipelineValidationError, match="exactly one initial source"):
            builder.build()

    def test_recurrent_input_requires_initial_source(self):
        model = _make_model({"state": [1, 4]}, {"next_state": [1, 4]})
        builder = PipelineBuilder()
        builder.add_model("dynamics", model, role="dynamics")
        builder.connect("dynamics.next_state", "dynamics.state", recurrent=True)
        builder.add_stage("imagine", "state_transition", ["dynamics"])
        builder.add_public_output("dynamics.next_state")

        with pytest.raises(PipelineValidationError, match=r"Recurrent input.*no initial"):
            builder.build()

    def test_dataflow_initializer_and_recurrent_update_can_share_input(self):
        builder = PipelineBuilder()
        builder.add_model(
            "encoder",
            _make_model({"observation": [1, 4]}, {"state": [1, 4]}),
            role="encoder",
        )
        builder.add_model(
            "dynamics",
            _make_model({"state": [1, 4]}, {"next_state": [1, 4]}),
            role="dynamics",
        )
        builder.connect("encoder.state", "dynamics.state")
        builder.connect("dynamics.next_state", "dynamics.state", recurrent=True)
        builder.declare_external("encoder.observation")
        builder.add_stage("encode", "single_pass", ["encoder"])
        builder.add_stage("imagine", "state_transition", ["dynamics"])
        builder.add_public_output("dynamics.next_state")

        manifest = builder.build().manifest

        assert manifest.source_of("dynamics.state") == InputSource.STATEFUL
        assert manifest.initial_source_of("dynamics.state") == InputSource.DATAFLOW

    def test_stateful_classification_does_not_depend_on_source_name_sorting(self):
        builder = PipelineBuilder()
        builder.add_model(
            "aencoder",
            _make_model({"observation": [1, 4]}, {"state": [1, 4]}),
            role="encoder",
        )
        builder.add_model(
            "dynamics",
            _make_model({"state": [1, 4]}, {"next_state": [1, 4]}),
            role="dynamics",
        )
        builder.connect("aencoder.state", "dynamics.state")
        builder.connect("dynamics.next_state", "dynamics.state", recurrent=True)
        builder.declare_external("aencoder.observation")
        builder.add_stage("encode", "single_pass", ["aencoder"])
        builder.add_stage("imagine", "state_transition", ["dynamics"])
        builder.add_public_output("dynamics.next_state")

        manifest = builder.build().manifest

        assert manifest.source_of("dynamics.state") == InputSource.STATEFUL
        assert manifest.initial_source_of("dynamics.state") == InputSource.DATAFLOW

    def test_double_declaration_rejected(self):
        builder = _simple_pipeline()
        builder.declare_generated("encoder.pixel_values", generator="zeros")
        with pytest.raises(PipelineValidationError, match="declared more than once"):
            builder.build()

    def test_defaulted_input_accepted_and_serialized(self):
        builder = self._pipeline_without_position_ids()
        builder.declare_default("decoder.position_ids", [[0, 1, 2]])
        manifest = builder.build().manifest
        assert manifest.source_of("decoder.position_ids") == InputSource.DEFAULTED
        entry = manifest.inputs_of_kind(InputSource.DEFAULTED)[0]
        assert entry.value == [[0, 1, 2]]
        assert PipelineManifest.from_json(manifest.to_json()) == manifest

    def test_defaulted_input_requires_json_safe_value(self):
        with pytest.raises(PipelineValidationError, match="JSON-safe"):
            PipelineInput(PipelinePort("a", "x"), InputSource.DEFAULTED, value=object())

    def test_defaulted_input_requires_a_value(self):
        with pytest.raises(PipelineValidationError, match="requires a value"):
            PipelineInput(PipelinePort("a", "x"), InputSource.DEFAULTED)

    def test_non_default_kind_rejects_value(self):
        with pytest.raises(PipelineValidationError, match="must not carry a default"):
            PipelineInput(PipelinePort("a", "x"), InputSource.EXTERNAL, value=3)

    def test_unknown_source_kind_rejected(self):
        with pytest.raises(PipelineValidationError, match="unknown source kind"):
            PipelineInput(PipelinePort("a", "x"), "magic")

    def test_duplicate_external_names_rejected(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.add_model("b", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.declare_external("a.x")
        builder.declare_external("b.x")
        builder.add_stage("run", "single_pass", ["a", "b"])
        with pytest.raises(PipelineValidationError, match="External input name"):
            builder.build()

    def test_alias_only_for_external(self):
        with pytest.raises(PipelineValidationError, match="Only external inputs"):
            PipelineInput(
                PipelinePort("a", "x"),
                InputSource.GENERATED,
                alias="z",
                generator=GeneratedInputRule("zeros"),
            )


class TestCycles:
    def _loop_builder(self, *, stage_kind: str) -> PipelineBuilder:
        model = _make_model(
            {"tokens": ["b", "s"], "state": ["b", 8]},
            {"logits": ["b", 32], "new_state": ["b", 8]},
        )
        builder = PipelineBuilder()
        builder.add_model("decoder", model, role="dynamics", run_on="decode")
        builder.connect("decoder.new_state", "decoder.state", recurrent=True)
        builder.declare_stateful("decoder.state")
        builder.declare_external("decoder.tokens")
        builder.add_stage("loop", stage_kind, ["decoder"], run_on="decode")
        builder.add_public_output("decoder.logits")
        return builder

    def test_legal_recurrent_cycle(self):
        manifest = self._loop_builder(stage_kind="autoregressive").build().manifest
        assert manifest.source_of("decoder.state") == InputSource.STATEFUL
        stage = manifest.stages[0]
        assert LOOP_CARRIED_STATE_CAPABILITY in stage.capabilities
        assert LOOP_CARRIED_STATE_CAPABILITY in manifest.required_capabilities

    def test_state_transition_stage_supports_loops(self):
        manifest = self._loop_builder(stage_kind="state_transition").build().manifest
        assert manifest.connections[0].recurrent

    def test_recurrent_edge_in_single_pass_stage_rejected(self):
        builder = self._loop_builder(stage_kind="single_pass")
        with pytest.raises(PipelineValidationError, match="loop-carried state"):
            builder.build()

    def test_illegal_non_recurrent_cycle(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.add_model("b", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="decoder")
        builder.connect("a.y", "b.x")
        builder.connect("b.y", "a.x")
        builder.add_stage("run", "iterative", ["a", "b"])
        builder.add_public_output("b.y")
        with pytest.raises(PipelineValidationError, match="cycle in non-recurrent"):
            builder.build()

    def test_non_recurrent_self_edge_rejected(self):
        model = _make_model({"x": [1, 4], "s": [1, 4]}, {"y": [1, 4]})
        builder = PipelineBuilder()
        builder.add_model("a", model, role="encoder")
        builder.connect("a.y", "a.s")
        builder.declare_external("a.x")
        builder.add_stage("run", "iterative", ["a"])
        with pytest.raises(PipelineValidationError, match="depend on itself"):
            builder.build()

    def test_recurrent_edge_across_stages_rejected(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.add_model("b", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="decoder")
        builder.connect("b.y", "a.x", recurrent=True)
        builder.declare_stateful("a.x")
        builder.declare_external("b.x")
        builder.add_stage("first", "iterative", ["a"])
        builder.add_stage("second", "iterative", ["b"])
        builder.add_public_output("a.y")
        with pytest.raises(PipelineValidationError, match="scoped to a single stage"):
            builder.build()


class TestStages:
    def test_unknown_component_in_stage_rejected(self):
        builder = _simple_pipeline()
        builder.add_stage("extra", "single_pass", ["ghost"])
        with pytest.raises(PipelineValidationError, match="unknown component 'ghost'"):
            builder.build()

    def test_component_without_stage_rejected(self):
        builder = PipelineBuilder()
        builder.add_model("a", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.declare_external("a.x")
        builder.add_public_output("a.y")
        with pytest.raises(PipelineValidationError, match="belong to no declared stage"):
            builder.build()

    def test_impossible_phase_combination_rejected(self):
        builder = PipelineBuilder()
        builder.add_model(
            "a", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder", run_on="prefill"
        )
        builder.declare_external("a.x")
        builder.add_stage("late", "single_pass", ["a"], run_on="decode")
        with pytest.raises(PipelineValidationError, match="could never execute"):
            builder.build()

    def test_on_demand_stage_with_presence(self):
        builder = PipelineBuilder()
        builder.add_model(
            "vision",
            _make_model({"pixels": [1, 4]}, {"feats": [1, 4]}),
            role="encoder",
            presence="has_image",
            run_on="on_demand",
        )
        builder.declare_external("vision.pixels")
        builder.add_stage(
            "maybe_vision",
            "on_demand",
            ["vision"],
            run_on="on_demand",
            options={"presence": "has_image"},
        )
        builder.add_public_output("vision.feats")
        manifest = builder.build().manifest
        assert manifest.component("vision").presence == "has_image"
        assert manifest.stages[0].options == {"presence": "has_image"}

    def test_on_demand_stage_without_presence_rejected(self):
        builder = PipelineBuilder()
        builder.add_model(
            "vision",
            _make_model({"pixels": [1, 4]}, {"feats": [1, 4]}),
            role="encoder",
        )
        builder.declare_external("vision.pixels")
        builder.add_stage("maybe_vision", "on_demand", ["vision"])
        builder.add_public_output("vision.feats")

        with pytest.raises(PipelineValidationError, match="no component presence"):
            builder.build()

    def test_composite_stage_kind(self):
        builder = _simple_pipeline()
        builder.add_stage("everything", "composite", ["encoder", "decoder"])
        manifest = builder.build().manifest
        assert [s.kind for s in manifest.stages] == [
            "single_pass",
            "autoregressive",
            "composite",
        ]

    def test_empty_stage_rejected(self):
        with pytest.raises(PipelineValidationError, match="must contain a component"):
            PipelineStage("s", "single_pass", ())

    def test_stage_options_must_be_json_safe(self):
        with pytest.raises(PipelineValidationError, match="JSON-safe"):
            PipelineStage("s", "single_pass", ("a",), options={"f": object()})

    def test_duplicate_stage_name_rejected(self):
        builder = _simple_pipeline()
        builder.add_stage("encode", "single_pass", ["decoder"])
        with pytest.raises(PipelineValidationError, match="Stage 'encode'"):
            builder.build()


class TestRegistries:
    def test_unknown_role_rejected(self):
        builder = PipelineBuilder()
        with pytest.raises(PipelineValidationError, match="Unknown role 'world_sim'"):
            builder.add_model("a", _make_model({"x": [1]}, {"y": [1]}), role="world_sim")

    def test_unknown_strategy_rejected(self):
        with pytest.raises(PipelineValidationError, match="Unknown strategy 'beam'"):
            PipelineStage("s", "beam", ("a",))

    def test_unknown_phase_rejected(self):
        builder = PipelineBuilder()
        with pytest.raises(PipelineValidationError, match="Unknown phase"):
            builder.add_model(
                "a", _make_model({"x": [1]}, {"y": [1]}), role="encoder", run_on="someday"
            )

    def test_registration_is_idempotent(self):
        first = register_role("test_role_idempotent", description="d")
        second = register_role("test_role_idempotent", description="d")
        assert first == second

    def test_transform_registration_is_idempotent(self):
        first = register_transform("test_transform_idempotent", capabilities=["cap"])
        second = register_transform("test_transform_idempotent", capabilities=["cap"])
        assert first == second
        assert first.capabilities == ("cap",)

    def test_conflicting_transform_registration_rejected(self):
        register_transform("test_transform_conflict", capabilities=["a"])
        with pytest.raises(PipelineValidationError, match="already registered"):
            register_transform("test_transform_conflict", capabilities=["b"])

    def test_registered_transform_usable_end_to_end(self):
        register_transform(
            "test_transform_usable",
            description="quantize latents into codebook ids",
            capabilities=["vector_quantization"],
        )
        builder = PipelineBuilder()
        builder.add_model(
            "a", _make_model({"x": [1, 4]}, {"y": ["b", "t", 16]}), role="encoder"
        )
        builder.add_model(
            "b",
            _make_model({"ids": ["b", "t"]}, {"y": [1, 4]}, dtype=ir.DataType.INT64),
            role="decoder",
        )
        builder.connect("a.y", "b.ids", transform="test_transform_usable")
        builder.declare_external("a.x")
        builder.add_stage("run", "single_pass", ["a", "b"])
        builder.add_public_output("b.y")
        manifest = builder.build().manifest
        assert manifest.required_capabilities == ("vector_quantization",)
        assert PipelineManifest.from_json(manifest.to_json()) == manifest

    def test_builtin_transform_capabilities(self):
        assert transform_definition("sample").capabilities == ("stochastic_sampling",)
        assert transform_definition("scheduler_step").capabilities == ("iterative_scheduler",)

    def test_unknown_transform_in_manifest_dict_rejected(self):
        data = _cosmos_style_pipeline().build().manifest.to_dict()
        data["connections"][0]["transform"] = "not_a_transform"
        with pytest.raises(PipelineValidationError, match="Unknown transform"):
            PipelineManifest.from_dict(data)

    def test_conflicting_registration_rejected(self):
        register_strategy("test_strategy_conflict", description="a")
        with pytest.raises(PipelineValidationError, match="already registered"):
            register_strategy("test_strategy_conflict", description="b")

    def test_registered_role_usable(self):
        register_role("test_role_usable", description="a world model rollout head")
        builder = PipelineBuilder()
        component = builder.add_model(
            "a", _make_model({"x": [1]}, {"y": [1]}), role="test_role_usable"
        )
        assert component.role == "test_role_usable"

    def test_unknown_role_in_manifest_dict_rejected(self):
        data = _simple_pipeline().build().manifest.to_dict()
        data["components"][0]["role"] = "not_a_role"
        with pytest.raises(PipelineValidationError, match="Unknown role"):
            PipelineManifest.from_dict(data)

    def test_unknown_strategy_in_manifest_dict_rejected(self):
        data = _simple_pipeline().build().manifest.to_dict()
        data["stages"][0]["kind"] = "not_a_strategy"
        with pytest.raises(PipelineValidationError, match="Unknown strategy"):
            PipelineManifest.from_dict(data)

    def test_generated_input_registration_is_closed_and_parameterized(self):
        register_generated_input(
            "test_generated_program",
            required_parameters=["source"],
            allowed_parameters=["source"],
        )
        with pytest.raises(PipelineValidationError, match="missing required"):
            GeneratedInputRule("test_generated_program")
        with pytest.raises(PipelineValidationError, match="unknown parameter"):
            GeneratedInputRule(
                "test_generated_program",
                {"source": "x", "extra": True},
            )
        assert GeneratedInputRule(
            "test_generated_program",
            {"source": "x"},
        ).parameters == {"source": "x"}

    def test_transform_parameters_are_validated(self):
        with pytest.raises(PipelineValidationError, match="unknown parameter"):
            PipelineConnection(
                PipelinePort("a", "y"),
                PipelinePort("b", "x"),
                transform="cast",
                parameters={"guess": "fp16"},
            )


class TestExecutableProfile:
    def _builder(self, tmp_path, *, include_state: bool = True) -> PipelineBuilder:
        builder = PipelineBuilder()
        builder.add_model(
            "decoder",
            _make_model(
                {"tokens": [1, 1], "state": [1, 4]},
                {"logits": [1, 8], "next_state": [1, 4]},
            ),
            role="decoder",
            preferred_execution_providers=["cuda", "cpu"],
            parameter_dtype="FLOAT",
        )
        builder.connect("decoder.next_state", "decoder.state", recurrent=True)
        builder.declare_external(
            "decoder.tokens",
            semantic="text.token_ids",
        )
        builder.declare_generated(
            "decoder.state",
            generator="zeros",
            semantic="kv_cache.initial",
        )
        builder.add_stage(
            "decode",
            "autoregressive",
            ["decoder"],
            options={
                "tokenizer_asset": "tokenizer.json",
                "sampling": {"do_sample": False},
                "stop": {"kind": "token_ids", "eos_token_ids": [2]},
                "state_names": ["cache"],
            },
        )
        if include_state:
            builder.add_state(
                "cache",
                kind="kv_cache",
                input="decoder.state",
                output="decoder.next_state",
                lifetime="sequence",
                release_after="decode",
                sequence_axis=1,
            )
            builder.add_public_state_output("cache", alias="final_cache")
        builder.add_public_output("decoder.logits")
        tokenizer = tmp_path / "tokenizer.json"
        tokenizer.write_text("{}", encoding="utf-8")
        scheduler_dir = tmp_path / "scheduler"
        scheduler_dir.mkdir(exist_ok=True)
        scheduler = scheduler_dir / "scheduler_config.json"
        scheduler.write_text("{}", encoding="utf-8")
        builder.add_asset("tokenizer.json", str(tokenizer))
        builder.add_asset("scheduler/scheduler_config.json", str(scheduler))
        builder.set_profile("test-world", "1.0")
        return builder

    def test_profile_round_trip_contains_executable_contract(self, tmp_path):
        manifest = self._builder(tmp_path).build().manifest
        restored = PipelineManifest.from_json(manifest.to_json())

        assert restored.profile is not None
        assert restored.profile.name == "test-world"
        assert restored.states[0].kind == "kv_cache"
        assert next(
            output for output in restored.outputs if output.name == "final_cache"
        ).state == ("cache")
        assert restored.inputs_of_kind(InputSource.GENERATED)[0].generator == (
            GeneratedInputRule("zeros")
        )
        assert restored.component("decoder").preferred_execution_providers == (
            "cuda",
            "cpu",
        )

    def test_profile_requires_state_lifecycle(self, tmp_path):
        with pytest.raises(PipelineValidationError, match="explicit state lifecycle"):
            self._builder(tmp_path, include_state=False).build()

    def test_profile_requires_input_semantics(self, tmp_path):
        builder = self._builder(tmp_path)
        entry = next(i for i in builder._inputs if i.kind == InputSource.EXTERNAL)
        builder._inputs[builder._inputs.index(entry)] = dataclasses.replace(
            entry,
            semantic=None,
        )
        with pytest.raises(PipelineValidationError, match="semantic names"):
            builder.build()

    def test_profile_version_is_independent_from_schema_major(self):
        assert PipelineProfile("future-runtime", "2.0").version == "2.0"

    def test_generated_program_port_references_are_validated(self):
        builder = PipelineBuilder()
        builder.add_model(
            "decoder",
            _make_model({"positions": [1, 1]}, {"logits": [1, 8]}),
            role="decoder",
        )
        builder.declare_generated(
            "decoder.positions",
            generator="multimodal_position_ids",
            parameters={"source": "ghost.tokens", "axes": 1},
        )
        builder.add_stage("run", "single_pass", ["decoder"])
        builder.add_public_output("decoder.logits")

        with pytest.raises(PipelineValidationError, match="unknown port"):
            builder.build()

    def test_generated_empty_tensor_axis_must_exist_on_port(self):
        builder = PipelineBuilder()
        builder.add_model(
            "decoder",
            _make_model({"cache": [1, "past"]}, {"logits": [1, 8]}),
            role="decoder",
        )
        builder.declare_generated(
            "decoder.cache",
            generator="empty_tensor",
            parameters={"dynamic_axes": {"past_sequence_length": 0}},
        )
        builder.add_stage("run", "single_pass", ["decoder"])
        builder.add_public_output("decoder.logits")

        with pytest.raises(PipelineValidationError, match="unknown dynamic axis"):
            builder.build()


class TestManifestSchema:
    def test_unknown_major_version_rejected(self):
        data = _simple_pipeline().build().manifest.to_dict()
        data["schema_version"] = "99.0"
        with pytest.raises(PipelineValidationError, match="Unsupported pipeline schema"):
            PipelineManifest.from_dict(data)

    def test_newer_minor_version_accepted(self):
        data = _simple_pipeline().build().manifest.to_dict()
        data["schema_version"] = "1.99"
        assert PipelineManifest.from_dict(data).schema_version == "1.99"

    def test_unknown_top_level_key_rejected(self):
        data = _simple_pipeline().build().manifest.to_dict()
        data["scheduler"] = {"algorithm": "continuous_batching"}
        with pytest.raises(PipelineValidationError, match="Unknown key"):
            PipelineManifest.from_dict(data)

    def test_metadata_bag_is_preserved(self):
        builder = _simple_pipeline()
        builder.set_metadata("provenance", {"tool": "mobius", "extra": [1, 2]})
        manifest = builder.build().manifest
        restored = PipelineManifest.from_json(manifest.to_json())
        assert restored.metadata == {"provenance": {"tool": "mobius", "extra": [1, 2]}}
        assert restored == manifest

    def test_component_metadata_preserved(self):
        data = _simple_pipeline().build().manifest.to_dict()
        data["components"][0]["metadata"] = {"future_field": True}
        restored = PipelineManifest.from_dict(data)
        assert restored.component("decoder").metadata == {"future_field": True}

    def test_public_output_must_exist(self):
        builder = _simple_pipeline()
        builder.add_public_output("decoder.ghost", alias="ghost")
        with pytest.raises(PipelineValidationError, match="unknown output 'ghost'"):
            builder.build()

    def test_required_capability_must_be_provided(self):
        builder = _simple_pipeline()
        builder.require_capability("streaming")
        with pytest.raises(PipelineValidationError, match="not provided by any"):
            builder.build()

    def test_required_capability_provided_by_component(self):
        builder = PipelineBuilder()
        builder.add_model(
            "a",
            _make_model({"x": [1, 4]}, {"y": [1, 4]}),
            role="encoder",
            capabilities=["streaming"],
        )
        builder.declare_external("a.x")
        builder.add_stage("run", "single_pass", ["a"])
        builder.add_public_output("a.y")
        builder.require_capability("streaming")
        assert builder.build().manifest.required_capabilities == ("streaming",)

    def test_manifest_equality_ignores_graph_identity(self):
        left = _simple_pipeline().build().manifest
        right = _simple_pipeline().build().manifest
        assert left == right
        assert left.component("encoder").model is not right.component("encoder").model


class TestPackagePersistence:
    def test_save_load_round_trip(self, tmp_path):
        pkg = _simple_pipeline().build()
        pkg.save(str(tmp_path), progress_bar=False)
        assert (tmp_path / PIPELINE_FILENAME).is_file()
        assert (tmp_path / "encoder" / "model.onnx").is_file()
        assert (tmp_path / "decoder" / "model.onnx").is_file()

        loaded = PipelinePackage.load(str(tmp_path))
        assert loaded.manifest == pkg.manifest
        assert sorted(loaded) == ["decoder", "encoder"]
        assert loaded.manifest.component("encoder").model is loaded["encoder"]

    def test_component_filenames_round_trip_exactly(self, tmp_path):
        pkg = _simple_pipeline().build()
        pkg.save(str(tmp_path), progress_bar=False)
        document = json.loads((tmp_path / PIPELINE_FILENAME).read_text(encoding="utf-8"))
        assert document["component_files"] == {
            "decoder": "decoder/model.onnx",
            "encoder": "encoder/model.onnx",
        }
        for relative in document["component_files"].values():
            assert (tmp_path / relative).is_file()

    def test_single_component_uses_flat_layout(self, tmp_path):
        builder = PipelineBuilder()
        builder.add_model("solo", _make_model({"x": [1, 4]}, {"y": [1, 4]}), role="encoder")
        builder.declare_external("solo.x")
        builder.add_stage("run", "single_pass", ["solo"])
        builder.add_public_output("solo.y")
        pkg = builder.build()
        pkg.save(str(tmp_path), progress_bar=False)
        assert (tmp_path / "model.onnx").is_file()
        assert pkg.component_files() == {"solo": "model.onnx"}
        assert PipelinePackage.load(str(tmp_path)).manifest == pkg.manifest

    def test_partial_save_rejected(self, tmp_path):
        pkg = _simple_pipeline().build()
        with pytest.raises(PipelineValidationError, match="partial saves"):
            pkg.save(str(tmp_path), components=lambda name: name == "encoder")

    def test_load_requires_pipeline_json(self, tmp_path):
        ModelPackage({"a": _make_model({"x": [1, 4]}, {"y": [1, 4]})}).save(
            str(tmp_path), progress_bar=False
        )
        with pytest.raises(PipelineValidationError, match="not found"):
            PipelinePackage.load(str(tmp_path))

    def test_load_rejects_missing_component_file(self, tmp_path):
        pkg = _simple_pipeline().build()
        pkg.save(str(tmp_path), progress_bar=False)
        (tmp_path / "encoder" / "model.onnx").unlink()
        with pytest.raises(PipelineValidationError, match="is missing"):
            PipelinePackage.load(str(tmp_path))

    def test_load_rejects_component_path_escape(self, tmp_path):
        pkg = _simple_pipeline().build()
        pkg.save(str(tmp_path), progress_bar=False)
        path = tmp_path / PIPELINE_FILENAME
        document = json.loads(path.read_text(encoding="utf-8"))
        document["component_files"]["encoder"] = "../outside/model.onnx"
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(PipelineValidationError, match="safe layout"):
            PipelinePackage.load(str(tmp_path))

    def test_load_wraps_corrupt_manifest_error(self, tmp_path):
        (tmp_path / PIPELINE_FILENAME).write_text('{"format":', encoding="utf-8")

        with pytest.raises(PipelineValidationError, match=r"valid 'pipeline\.json'"):
            PipelinePackage.load(str(tmp_path))

    def test_load_rejects_wrong_manifest_format(self, tmp_path):
        pkg = _simple_pipeline().build()
        pkg.save(str(tmp_path), progress_bar=False)
        path = tmp_path / PIPELINE_FILENAME
        document = json.loads(path.read_text(encoding="utf-8"))
        document["format"] = "other"
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(PipelineValidationError, match="unsupported format"):
            PipelinePackage.load(str(tmp_path))

    def test_failed_resave_leaves_no_completeness_marker(self, tmp_path):
        pkg = _simple_pipeline().build()
        pkg.save(str(tmp_path), progress_bar=False)

        with (
            mock.patch("mobius._pipeline.json.dump", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            pkg.save(str(tmp_path), progress_bar=False)

        assert not (tmp_path / PIPELINE_FILENAME).exists()
        assert not list(tmp_path.glob(".mobius-pipeline-*"))

    def test_desynchronized_models_rejected(self):
        pkg = _simple_pipeline().build()
        pkg["extra"] = _make_model({"x": [1, 4]}, {"y": [1, 4]})
        with pytest.raises(PipelineValidationError, match="do not match the manifest"):
            pkg.save("unused-directory", progress_bar=False)

    def test_package_requires_matching_models(self):
        manifest = _simple_pipeline().build().manifest
        with pytest.raises(PipelineValidationError, match="do not match the manifest"):
            PipelinePackage({"encoder": _encoder()}, manifest)


class TestAssets:
    def _asset(self, tmp_path, name: str, text: str) -> str:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _pipeline_with_assets(self, tmp_path) -> PipelineBuilder:
        builder = _simple_pipeline()
        builder.add_asset("tokenizer.json", self._asset(tmp_path, "tokenizer.json", '{"v":1}'))
        builder.add_asset(
            "scheduler/scheduler_config.json",
            self._asset(tmp_path, "scheduler_config.json", '{"steps":30}'),
        )
        return builder

    def test_assets_saved_and_recorded_by_destination_only(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        out = tmp_path / "out"
        pkg = self._pipeline_with_assets(source_dir).build()
        pkg.save(str(out), progress_bar=False)

        assert (out / "tokenizer.json").read_text(encoding="utf-8") == '{"v":1}'
        assert (out / "scheduler" / "scheduler_config.json").is_file()

        document = json.loads((out / PIPELINE_FILENAME).read_text(encoding="utf-8"))
        assets = document["manifest"]["assets"]
        assert assets == [
            {"path": "scheduler/scheduler_config.json"},
            {"path": "tokenizer.json"},
        ]
        # No machine-local source path may leak into the manifest.
        assert str(source_dir) not in json.dumps(document)

    def test_asset_round_trip_exposes_resolved_paths(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        out = tmp_path / "out"
        pkg = self._pipeline_with_assets(source_dir).build()
        pkg.save(str(out), progress_bar=False)

        loaded = PipelinePackage.load(str(out))
        assert loaded.manifest == pkg.manifest
        assert set(loaded.assets) == {"tokenizer.json", "scheduler/scheduler_config.json"}
        resolved = loaded.asset_path("tokenizer.json")
        assert os.path.isfile(resolved)
        with open(resolved, encoding="utf-8") as file:
            assert file.read() == '{"v":1}'

    def test_resaving_a_loaded_package_is_idempotent(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        first = tmp_path / "first"
        second = tmp_path / "second"
        self._pipeline_with_assets(source_dir).build().save(str(first), progress_bar=False)
        loaded = PipelinePackage.load(str(first))
        loaded.save(str(second), progress_bar=False)
        assert (second / "tokenizer.json").read_text(encoding="utf-8") == '{"v":1}'
        assert PipelinePackage.load(str(second)).manifest == loaded.manifest

    def test_save_in_place_preserves_assets(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        out = tmp_path / "out"
        self._pipeline_with_assets(source_dir).build().save(str(out), progress_bar=False)
        loaded = PipelinePackage.load(str(out))
        loaded.save(str(out), progress_bar=False)
        assert (out / "tokenizer.json").read_text(encoding="utf-8") == '{"v":1}'

    def test_missing_required_asset_fails_load(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        out = tmp_path / "out"
        self._pipeline_with_assets(source_dir).build().save(str(out), progress_bar=False)
        (out / "tokenizer.json").unlink()
        with pytest.raises(PipelineValidationError, match=r"Required asset 'tokenizer\.json'"):
            PipelinePackage.load(str(out))

    def test_optional_asset_may_be_absent(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        out = tmp_path / "out"
        builder = _simple_pipeline()
        builder.add_asset(
            "processor_config.json",
            self._asset(source_dir, "processor_config.json", "{}"),
            required=False,
        )
        builder.build().save(str(out), progress_bar=False)
        (out / "processor_config.json").unlink()
        loaded = PipelinePackage.load(str(out))
        assert loaded.assets == {}
        assert loaded.manifest.required_assets == ()

    def test_required_asset_without_source_rejected(self):
        manifest = _simple_pipeline().build().manifest
        manifest = dataclasses.replace(manifest, assets=(PipelineAsset("tokenizer.json"),))
        with pytest.raises(PipelineValidationError, match="have no source file"):
            PipelinePackage(
                {name: manifest.component(name).model for name in manifest.component_names},
                manifest,
            )

    def test_undeclared_asset_source_rejected(self, tmp_path):
        pkg = _simple_pipeline().build()
        with pytest.raises(PipelineValidationError, match="undeclared destination"):
            PipelinePackage(
                dict(pkg),
                pkg.manifest,
                assets={"tokenizer.json": self._asset(tmp_path, "t.json", "{}")},
            )

    def test_missing_source_file_rejected(self, tmp_path):
        builder = _simple_pipeline()
        with pytest.raises(PipelineValidationError, match="must be an existing file"):
            builder.add_asset("tokenizer.json", str(tmp_path / "nope.json"))

    def test_duplicate_asset_destination_rejected(self, tmp_path):
        builder = _simple_pipeline()
        source = self._asset(tmp_path, "t.json", "{}")
        builder.add_asset("tokenizer.json", source)
        with pytest.raises(PipelineValidationError, match="already registered"):
            builder.add_asset("tokenizer.json", source)

    @pytest.mark.parametrize(
        "destination",
        [
            "../escape.json",
            "a/../../escape.json",
            "/etc/passwd",
            "C:/Windows/system.ini",
            "c:tokenizer.json",
            "~/tokenizer.json",
            "sub\\tokenizer.json",
            "a//b.json",
            "./tokenizer.json",
            "sub/",
            "",
            "   ",
            " tokenizer.json",
            "nul.json",
            "sub/con",
            "bad\x00name.json",
        ],
    )
    def test_unsafe_asset_destinations_rejected(self, tmp_path, destination):
        builder = _simple_pipeline()
        source = self._asset(tmp_path, "t.json", "{}")
        with pytest.raises(PipelineValidationError):
            builder.add_asset(destination, source)

    @pytest.mark.parametrize(
        "destination",
        ["tokenizer.json", "sub/tokenizer.json", "a/b/c.txt", "chat_template.jinja"],
    )
    def test_safe_asset_destinations_accepted(self, tmp_path, destination):
        builder = _simple_pipeline()
        builder.add_asset(destination, self._asset(tmp_path, "t.json", "{}"))
        assert builder.build().manifest.assets[0].path == destination

    def test_asset_cannot_shadow_package_files(self, tmp_path):
        builder = _simple_pipeline()
        source = self._asset(tmp_path, "t.json", "{}")
        builder.add_asset(PIPELINE_FILENAME, source)
        with pytest.raises(PipelineValidationError, match="collides with a file written"):
            builder.build()

    def test_asset_cannot_shadow_component_graph(self, tmp_path):
        builder = _simple_pipeline()
        builder.add_asset("encoder/model.onnx", self._asset(tmp_path, "t.json", "{}"))
        with pytest.raises(PipelineValidationError, match="collides with a file written"):
            builder.build()

    def test_asset_cannot_shadow_component_graph_with_different_case(self, tmp_path):
        builder = _simple_pipeline()
        builder.add_asset("Encoder/model.onnx", self._asset(tmp_path, "t.json", "{}"))
        with pytest.raises(PipelineValidationError, match="collides with a file written"):
            builder.build()

    def test_asset_manifest_round_trips_through_json(self, tmp_path):
        builder = _simple_pipeline()
        builder.add_asset("tokenizer.json", self._asset(tmp_path, "t.json", "{}"))
        builder.add_asset("extra.txt", self._asset(tmp_path, "e.txt", "x"), required=False)
        manifest = builder.build().manifest
        restored = PipelineManifest.from_json(manifest.to_json())
        assert restored == manifest
        assert [(a.path, a.required) for a in restored.assets] == [
            ("extra.txt", False),
            ("tokenizer.json", True),
        ]

    def test_no_temporary_files_left_behind(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        out = tmp_path / "out"
        self._pipeline_with_assets(source_dir).build().save(str(out), progress_bar=False)
        leftovers = [p.name for p in out.rglob("*") if p.name.startswith(".mobius-asset-")]
        assert leftovers == []

    def test_deleted_source_between_build_and_save_rejected(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        builder = _simple_pipeline()
        source = self._asset(source_dir, "tokenizer.json", "{}")
        builder.add_asset("tokenizer.json", source)
        pkg = builder.build()
        os.remove(source)
        out = tmp_path / "out"
        with pytest.raises(PipelineValidationError, match="does not exist"):
            pkg.save(str(out), progress_bar=False)
        assert not (out / PIPELINE_FILENAME).exists()

    def test_asset_path_unknown_destination(self):
        pkg = _simple_pipeline().build()
        with pytest.raises(KeyError):
            pkg.asset_path("tokenizer.json")


class TestPackageConfigs:
    def test_primary_config_preserved(self):
        config = object()
        pkg = _simple_pipeline().build(config=config)
        assert pkg.config is config
        assert pkg.config_for("encoder") is config

    def test_per_component_config(self):
        primary = object()
        encoder_config = object()
        pkg = _simple_pipeline().build(
            config=primary, component_configs={"encoder": encoder_config}
        )
        assert pkg.config_for("encoder") is encoder_config
        assert pkg.config_for("decoder") is primary
        assert pkg.config is primary

    def test_config_for_unknown_component_rejected(self):
        with pytest.raises(PipelineValidationError, match="unknown component"):
            _simple_pipeline().build(component_configs={"ghost": object()})

    def test_component_config_must_be_json_safe(self):
        builder = PipelineBuilder()
        with pytest.raises(PipelineValidationError, match="JSON-safe"):
            builder.add_model(
                "a",
                _make_model({"x": [1, 4]}, {"y": [1, 4]}),
                role="encoder",
                config={"bad": object()},
            )

    def test_component_config_round_trips(self):
        builder = PipelineBuilder()
        builder.add_model(
            "a",
            _make_model({"x": [1, 4]}, {"y": [1, 4]}),
            role="encoder",
            config={"latent_dim": 8},
            source="acme/world-model",
        )
        builder.declare_external("a.x")
        builder.add_stage("run", "single_pass", ["a"])
        builder.add_public_output("a.y")
        manifest = builder.build().manifest
        restored = PipelineManifest.from_json(manifest.to_json())
        assert restored.component("a").config == {"latent_dim": 8}
        assert restored.component("a").source == "acme/world-model"


class TestDataclassSurface:
    def test_component_lookup_helpers(self):
        component = PipelineComponent.from_model("a", _encoder(), role="encoder")
        assert component.input("pixel_values") is not None
        assert component.input("missing") is None
        assert component.output("image_features") is not None

    def test_connection_serialization(self):
        connection = PipelineConnection(
            PipelinePort("a", "y"), PipelinePort("b", "x"), recurrent=True, transform="cast"
        )
        assert connection.to_dict() == {
            "source": "a.y",
            "target": "b.x",
            "recurrent": True,
            "transform": "cast",
        }
        assert PipelineConnection.from_dict(connection.to_dict()) == connection

    def test_transform_context_round_trips(self):
        connection = PipelineConnection(
            PipelinePort("denoiser", "velocity"),
            PipelinePort("decoder", "latent"),
            transform="scheduler_step",
            context=(PipelinePort("denoiser", "sample"),),
        )

        assert connection.to_dict()["context"] == ["denoiser.sample"]
        assert PipelineConnection.from_dict(connection.to_dict()) == connection

    def test_output_alias_defaults_to_port_name(self):
        assert PipelineOutput(PipelinePort("a", "logits")).name == "logits"
        assert PipelineOutput(PipelinePort("a", "logits"), "y").name == "y"

    def test_empty_manifest_is_valid(self):
        manifest = PipelineManifest()
        assert manifest.components == ()
        assert PipelineManifest.from_dict(manifest.to_dict()) == manifest

    def test_component_names_must_be_portable_across_case_sensitivity(self):
        first = PipelineComponent.from_model("Encoder", _encoder(), role="encoder")
        second = PipelineComponent.from_model("encoder", _encoder(), role="encoder")

        with pytest.raises(PipelineValidationError, match="case-insensitive"):
            PipelineManifest(components=(first, second))
