# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for multimodal components."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as torch_functional

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


def _run_projector(
    projector,
    features: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    builder, op, graph = create_test_builder()
    input_value = create_test_input(builder, "features", list(features.shape))
    result = projector(op, input_value)
    graph.outputs.append(result)

    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    for name, parameter in projector.named_parameters():
        shape = tuple(int(dim) for dim in parameter.shape)
        if name.endswith(".weight") and "norm" in name:
            values = (1.0 + rng.normal(0.0, 0.1, shape)).astype(np.float32)
        else:
            values = rng.normal(0.0, 0.1, shape).astype(np.float32)
        parameter.const_value = ir.tensor(values)
        state[name] = values

    proto = ir.serde.serialize_model(ir.Model(graph, ir_version=11))
    session = ort.InferenceSession(
        proto.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    (actual,) = session.run(None, {"features": features})
    return actual, state


def _linear(x: torch.Tensor, state: dict[str, np.ndarray], stem: str) -> torch.Tensor:
    weight = torch.from_numpy(state[f"{stem}.weight"])
    bias_value = state.get(f"{stem}.bias")
    bias = None if bias_value is None else torch.from_numpy(bias_value)
    return torch_functional.linear(x, weight, bias)


def _layer_norm(
    x: torch.Tensor,
    state: dict[str, np.ndarray],
    stem: str,
    eps: float,
) -> torch.Tensor:
    return torch_functional.layer_norm(
        x,
        (x.shape[-1],),
        torch.from_numpy(state[f"{stem}.weight"]),
        torch.from_numpy(state[f"{stem}.bias"]),
        eps,
    )


def _ldp_block(
    x: torch.Tensor,
    state: dict[str, np.ndarray],
    stem: str,
    *,
    stride: int,
    eps: float,
) -> torch.Tensor:
    residual = x
    x = torch_functional.conv2d(
        x,
        torch.from_numpy(state[f"{stem}.depthwise.weight"]),
        stride=stride,
        padding=1,
        groups=x.shape[1],
    )
    x = _layer_norm(
        x.permute(0, 2, 3, 1),
        state,
        f"{stem}.depthwise_norm",
        eps,
    ).permute(0, 3, 1, 2)
    x = torch_functional.hardswish(x)
    gate = x.mean(dim=(2, 3))
    gate = torch_functional.relu(_linear(gate, state, f"{stem}.se_fc1"))
    gate = torch_functional.hardsigmoid(_linear(gate, state, f"{stem}.se_fc2"))[
        :, :, None, None
    ]
    x = x * gate
    x = torch_functional.conv2d(x, torch.from_numpy(state[f"{stem}.pointwise.weight"]))
    x = _layer_norm(
        x.permute(0, 2, 3, 1),
        state,
        f"{stem}.pointwise_norm",
        eps,
    ).permute(0, 3, 1, 2)
    return x + residual if stride == 1 else x


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

    def test_mlp_matches_nonzero_reference(self):
        projector = GGUFMLPProjector(vision_hidden_size=8, text_hidden_size=16)
        features = np.random.default_rng(1).normal(size=(1, 7, 8)).astype(np.float32)

        actual, state = _run_projector(projector, features, seed=2)
        expected = _linear(torch.from_numpy(features), state, "linear_0")
        expected = torch_functional.gelu(expected, approximate="tanh")
        expected = _linear(expected, state, "linear_2").numpy()

        assert actual.shape == (1, 7, 16)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

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

    def test_ldp_matches_nonzero_reference_and_144_token_contract(self):
        projector = MobileLDPProjector(vision_hidden_size=8, text_hidden_size=16)
        features = np.random.default_rng(3).normal(size=(1, 576, 8)).astype(np.float32)

        actual, state = _run_projector(projector, features, seed=4)
        expected = _linear(torch.from_numpy(features), state, "mlp_1")
        expected = torch_functional.gelu(expected, approximate="tanh")
        expected = _linear(expected, state, "mlp_3")
        expected = expected.transpose(1, 2).reshape(1, 16, 24, 24)
        expected = _ldp_block(expected, state, "block_1", stride=1, eps=1e-5)
        expected = _ldp_block(expected, state, "block_2", stride=2, eps=1e-5)
        expected = expected.reshape(1, 16, 144).transpose(1, 2).numpy()

        assert actual.shape == (1, 144, 16)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

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

    def test_ldpv2_matches_nonzero_reference_and_144_token_contract(self):
        projector = MobileLDPV2Projector(vision_hidden_size=8, text_hidden_size=16)
        features = np.random.default_rng(5).normal(size=(1, 576, 8)).astype(np.float32)

        actual, state = _run_projector(projector, features, seed=6)
        expected = _linear(torch.from_numpy(features), state, "mlp_0")
        expected = torch_functional.gelu(expected, approximate="tanh")
        expected = _linear(expected, state, "mlp_2")
        expected = expected.transpose(1, 2).reshape(1, 16, 24, 24)
        expected = torch_functional.avg_pool2d(expected, kernel_size=2, stride=2)
        peg = torch_functional.conv2d(
            expected,
            torch.from_numpy(state["peg_0.weight"]),
            torch.from_numpy(state["peg_0.bias"]),
            padding=1,
            groups=16,
        )
        expected = (expected + peg).reshape(1, 16, 144).transpose(1, 2).numpy()

        assert actual.shape == (1, 144, 16)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

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

    def test_adapter_matches_nonzero_reference_and_boundary_rows(self):
        projector = GLMEdgeAdapterProjector(
            vision_hidden_size=8,
            text_hidden_size=16,
            intermediate_size=48,
            grid_size=4,
        )
        features = np.random.default_rng(7).normal(size=(1, 16, 8)).astype(np.float32)

        actual, state = _run_projector(projector, features, seed=8)
        expected = torch.from_numpy(features).reshape(1, 4, 4, 8).permute(0, 3, 1, 2)
        expected = torch_functional.conv2d(
            expected,
            torch.from_numpy(state["conv.weight"]),
            torch.from_numpy(state["conv.bias"]),
            stride=2,
        )
        expected = expected.flatten(2).transpose(1, 2)
        expected = _linear(expected, state, "linear")
        expected = _layer_norm(expected, state, "norm1", 1e-6)
        expected = torch_functional.gelu(expected, approximate="tanh")
        gate = _linear(expected, state, "gate")
        up = _linear(expected, state, "dense_h_to_4h")
        expected = _linear(torch_functional.silu(gate) * up, state, "dense_4h_to_h")
        expected = torch.cat(
            (
                torch.from_numpy(state["boi"])[None, None],
                expected,
                torch.from_numpy(state["eoi"])[None, None],
            ),
            dim=1,
        ).numpy()

        assert actual.shape == (1, 6, 16)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

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

    def test_resampler_matches_nonzero_reference_including_query_positions(self):
        projector = MiniCPMResamplerProjector(
            vision_hidden_size=8,
            text_hidden_size=128,
            num_queries=4,
            grid_size=2,
        )
        features = np.random.default_rng(9).normal(size=(1, 4, 8)).astype(np.float32)

        actual, state = _run_projector(projector, features, seed=10)
        query = torch.from_numpy(state["query"])[None]
        query = _layer_norm(query, state, "ln_q", 1e-6)
        query = query + torch.from_numpy(state["pos_embed"])[None]
        value = _linear(torch.from_numpy(features), state, "kv")
        value = _layer_norm(value, state, "ln_kv", 1e-6)

        positions = torch.arange(4)
        pos_h = torch.div(positions, 2, rounding_mode="floor").float()
        pos_w = torch.remainder(positions, 2).float()
        omega_index = torch.arange(32).float()
        omega = torch.reciprocal(torch.pow(10_000.0, omega_index / 32.0))
        theta_x = pos_w[:, None] * omega[None]
        theta_y = pos_h[:, None] * omega[None]
        position = torch.cat(
            (theta_x.sin(), theta_x.cos(), theta_y.sin(), theta_y.cos()),
            dim=1,
        )
        key = value + position[None]

        q = _linear(query, state, "attn_q").reshape(1, 4, 1, 128).transpose(1, 2)
        k = _linear(key, state, "attn_k").reshape(1, 4, 1, 128).transpose(1, 2)
        v = _linear(value, state, "attn_v").reshape(1, 4, 1, 128).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (128**-0.5)
        expected = torch.matmul(scores.softmax(dim=-1), v)
        expected = expected.transpose(1, 2).reshape(1, 4, 128)
        expected = _linear(expected, state, "attn_out")
        expected = _layer_norm(expected, state, "ln_post", 1e-6)
        expected = _linear(expected, state, "proj").numpy()

        assert actual.shape == (1, 4, 128)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


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
