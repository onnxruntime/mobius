# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for multimodal components."""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius._testing import (
    create_test_builder,
    create_test_input,
)
from mobius.components._multimodal import (
    Gemma3MultiModalProjector,
    GGUFMLPProjector,
    GLMEdgeAdapterProjector,
    InputMixer,
    LinearMultiModalProjector,
    MiniCPMResamplerProjector,
    MLPMultiModalProjector,
    MobileLDPProjector,
    MobileLDPV2Projector,
)


class TestGemma3MultiModalProjector:
    def test_has_norm_and_projection(self):
        proj = Gemma3MultiModalProjector(
            vision_hidden_size=64,
            text_hidden_size=128,
            patches_per_image=4,
            tokens_per_image=4,
        )
        param_names = [n for n, _ in proj.named_parameters()]
        assert any("mm_soft_emb_norm" in n for n in param_names)
        assert any("mm_input_projection_weight" in n for n in param_names)

    def test_forward(self):
        proj = Gemma3MultiModalProjector(
            vision_hidden_size=64,
            text_hidden_size=128,
            patches_per_image=4,
            tokens_per_image=4,
        )
        b, op, graph = create_test_builder()
        features = create_test_input(b, "features", [1, 16, 64])
        result = proj(op, features)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0

    def test_forward_with_pooling(self):
        proj = Gemma3MultiModalProjector(
            vision_hidden_size=64,
            text_hidden_size=128,
            patches_per_image=8,
            tokens_per_image=4,
        )
        b, op, graph = create_test_builder()
        features = create_test_input(b, "features", [1, 64, 64])
        result = proj(op, features)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0


class TestMLPMultiModalProjector:
    def test_has_two_linear_layers(self):
        proj = MLPMultiModalProjector(
            vision_hidden_size=64,
            text_hidden_size=128,
        )
        param_names = [n for n, _ in proj.named_parameters()]
        assert any("linear_1" in n for n in param_names)
        assert any("linear_2" in n for n in param_names)

    def test_forward(self):
        proj = MLPMultiModalProjector(
            vision_hidden_size=64,
            text_hidden_size=128,
        )
        b, op, graph = create_test_builder()
        features = create_test_input(b, "features", [1, 16, 64])
        result = proj(op, features)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0


class TestGenericGGUFProjectors:
    def test_mlp_graph(self):
        projector = GGUFMLPProjector(vision_hidden_size=8, text_hidden_size=16)
        builder, op, graph = create_test_builder()
        features = create_test_input(builder, "features", [1, 16, 8])
        result = projector(op, features)
        builder._adapt_outputs([result], "")
        assert graph.num_nodes() > 0
        assert all(
            node.attributes["approximate"].value == "tanh"
            for node in graph
            if node.op_type == "Gelu"
        )

    def test_single_layer_mlp_graph(self):
        projector = GGUFMLPProjector(
            vision_hidden_size=8,
            text_hidden_size=16,
            has_second_layer=False,
        )
        builder, op, _graph = create_test_builder()
        features = create_test_input(builder, "features", [1, 16, 8])
        result = projector(op, features)
        builder._adapt_outputs([result], "")
        assert {name for name, _ in projector.named_parameters()} == {
            "linear_0.weight",
            "linear_0.bias",
        }

    def test_ldp_graph(self):
        projector = MobileLDPProjector(vision_hidden_size=8, text_hidden_size=16)
        builder, op, graph = create_test_builder()
        features = create_test_input(builder, "features", [1, 576, 8])
        result = projector(op, features)
        builder._adapt_outputs([result], "")
        assert graph.num_nodes() > 0
        hard_sigmoid = next(node for node in graph if node.op_type == "HardSigmoid")
        assert hard_sigmoid.attributes["alpha"].value == pytest.approx(1.0 / 6.0)
        assert hard_sigmoid.attributes["beta"].value == pytest.approx(0.5)
        assert all(
            node.attributes["approximate"].value == "tanh"
            for node in graph
            if node.op_type == "Gelu"
        )

    def test_ldpv2_graph(self):
        projector = MobileLDPV2Projector(vision_hidden_size=8, text_hidden_size=16)
        builder, op, graph = create_test_builder()
        features = create_test_input(builder, "features", [1, 576, 8])
        result = projector(op, features)
        builder._adapt_outputs([result], "")
        assert graph.num_nodes() > 0
        assert all(
            node.attributes["approximate"].value == "tanh"
            for node in graph
            if node.op_type == "Gelu"
        )

    def test_adapter_graph(self):
        projector = GLMEdgeAdapterProjector(
            vision_hidden_size=8,
            text_hidden_size=16,
            intermediate_size=48,
            grid_size=4,
        )
        builder, op, graph = create_test_builder()
        features = create_test_input(builder, "features", [1, 16, 8])
        result = projector(op, features)
        builder._adapt_outputs([result], "")
        assert graph.num_nodes() > 0
        assert all(
            node.attributes["approximate"].value == "tanh"
            for node in graph
            if node.op_type == "Gelu"
        )

    def test_resampler_graph(self):
        projector = MiniCPMResamplerProjector(
            vision_hidden_size=8,
            text_hidden_size=16,
            num_queries=4,
            grid_size=4,
            head_dim=8,
        )
        builder, op, graph = create_test_builder()
        features = create_test_input(builder, "features", [1, 16, 8])
        result = projector(op, features)
        builder._adapt_outputs([result], "")
        assert graph.num_nodes() > 0


class TestLinearMultiModalProjector:
    def test_has_linear_layer(self):
        proj = LinearMultiModalProjector(
            vision_hidden_size=64,
            text_hidden_size=128,
        )
        param_names = [n for n, _ in proj.named_parameters()]
        assert any("linear" in n for n in param_names)

    def test_forward(self):
        proj = LinearMultiModalProjector(
            vision_hidden_size=64,
            text_hidden_size=128,
        )
        b, op, graph = create_test_builder()
        features = create_test_input(b, "features", [1, 16, 64])
        result = proj(op, features)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0


class TestInputMixer:
    def test_forward(self):
        mixer = InputMixer(image_token_id=999)
        b, op, graph = create_test_builder()
        text_emb = create_test_input(b, "text_emb", [1, 10, 64])
        vision_emb = create_test_input(b, "vision_emb", [1, 4, 64])
        input_ids = create_test_input(b, "input_ids", [1, 10], dtype=ir.DataType.INT64)
        result = mixer(op, text_emb, vision_emb, input_ids)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0

    def test_image_token_id_stored(self):
        mixer = InputMixer(image_token_id=42)
        assert mixer.image_token_id == 42
