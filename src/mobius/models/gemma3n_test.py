# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Gemma3n activation sparsity (``_gaussian_topk``)."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from mobius._configs import Gemma3nConfig
from mobius._constants import OPSET_VERSION
from mobius.models.gemma3n import Gemma3nMLP


def _tiny_gemma3n_config(**overrides) -> Gemma3nConfig:
    """Create a minimal Gemma3nConfig for MLP-level tests."""
    defaults = dict(
        model_type="gemma3n_text",
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
        hidden_act="gelu_pytorch_tanh",
        layer_types=["full_attention", "sliding_attention"],
        attn_qk_norm=True,
        altup_num_inputs=2,
        laurel_rank=16,
        hidden_size_per_layer_input=32,
        vocab_size_per_layer_input=256,
    )
    defaults.update(overrides)
    return Gemma3nConfig(**defaults)


def _hf_gaussian_topk(inputs: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Verbatim port of HF ``Gemma3nTextMLP._gaussian_topk``."""
    target = torch.tensor(sparsity, dtype=torch.float32, device=inputs.device)
    std_multiplier = torch.distributions.normal.Normal(0, 1).icdf(target)
    std_multiplier = std_multiplier.type(inputs.dtype)
    inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
    inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
    cutoff_x = inputs_mean + inputs_std * std_multiplier
    return torch.nn.functional.relu(inputs - cutoff_x)


def _run_gaussian_topk(mlp: Gemma3nMLP, x: np.ndarray) -> np.ndarray:
    """Build a single-op ONNX graph around ``mlp._gaussian_topk`` and run it."""
    from onnxscript import GraphBuilder

    x_input = ir.Value(
        name="x",
        shape=ir.Shape(list(x.shape)),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    graph = ir.Graph(
        inputs=[x_input],
        outputs=[],
        nodes=[],
        name="test_gaussian_topk",
        opset_imports={"": OPSET_VERSION},
    )
    gb = GraphBuilder(graph)
    result = mlp._gaussian_topk(gb.op, x_input)
    result.name = "output"
    graph.outputs.append(result)

    # Serialize in-memory (avoids Windows PermissionError from concurrent
    # tempfile access), matching the other component-level ORT tests.
    proto = ir.serde.serialize_model(ir.Model(graph, ir_version=11))
    sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x})[0]


class TestGemma3nActivationSparsity:
    """Gemma 3n sparsifies the gate projection on its early layers.

    E4B applies 0.95 sparsity to layers 0-9 and none to layers 10-34, so the
    per-layer pattern must select between the sparse and plain MLP paths.
    """

    @pytest.mark.parametrize("sparsity", [0.95, 0.5, 0.1])
    def test_std_multiplier_matches_torch_icdf(self, sparsity):
        """The folded Phi^-1(sparsity) constant matches HF's icdf call."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[sparsity, 0.0])
        mlp = Gemma3nMLP(config, layer_idx=0)

        expected = (
            torch.distributions.normal.Normal(0, 1)
            .icdf(torch.tensor(sparsity, dtype=torch.float32))
            .item()
        )
        assert mlp._std_multiplier == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("sparsity", [0.95, 0.5])
    def test_gaussian_topk_matches_hf(self, sparsity):
        """The ONNX cutoff graph matches HF's ``_gaussian_topk`` numerically."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[sparsity, 0.0])
        mlp = Gemma3nMLP(config, layer_idx=0)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 3, 128)).astype(np.float32)

        onnx_out = _run_gaussian_topk(mlp, x)
        hf_out = _hf_gaussian_topk(torch.from_numpy(x), sparsity).numpy()

        np.testing.assert_allclose(onnx_out, hf_out, atol=1e-5, rtol=1e-5)

    def test_gaussian_topk_zeroes_expected_fraction(self):
        """0.95 sparsity keeps roughly 5% of a Gaussian row's activations."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[0.95, 0.0])
        mlp = Gemma3nMLP(config, layer_idx=0)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((8, 1024)).astype(np.float32)

        kept = (_run_gaussian_topk(mlp, x) > 0).mean()
        assert kept == pytest.approx(0.05, abs=0.02)

    def test_pattern_selects_per_layer_sparsity(self):
        """Each layer reads its own entry from the pattern."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[0.95, 0.0])

        assert Gemma3nMLP(config, layer_idx=0).activation_sparsity == 0.95
        assert Gemma3nMLP(config, layer_idx=1).activation_sparsity == 0.0

    def test_no_pattern_disables_sparsity(self):
        """Without a pattern the MLP keeps the plain gated path."""
        mlp = Gemma3nMLP(_tiny_gemma3n_config(), layer_idx=0)

        assert mlp.activation_sparsity == 0.0
        assert mlp._std_multiplier == 0.0

    def test_dense_layer_emits_no_cutoff_ops(self):
        """A zero-sparsity layer must not pay for the mean/std subgraph."""
        from onnxscript import GraphBuilder

        config = _tiny_gemma3n_config(activation_sparsity_pattern=[0.95, 0.0])
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            name="test_dense",
            opset_imports={"": OPSET_VERSION},
        )
        gb = GraphBuilder(graph)
        x = ir.Value(
            name="x",
            shape=ir.Shape([1, 2, config.hidden_size]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        graph.inputs.append(x)

        Gemma3nMLP(config, layer_idx=1).forward(gb.op, x)

        assert "Relu" not in {node.op_type for node in graph}

    @pytest.mark.parametrize("sparsity", [1.0, -0.1, 1.5])
    def test_rejects_out_of_range_sparsity(self, sparsity):
        """Sparsity outside [0, 1) has no finite Gaussian cutoff."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[sparsity, 0.0])

        with pytest.raises(ValueError, match="activation_sparsity_pattern"):
            Gemma3nMLP(config, layer_idx=0)

    def test_rejects_short_pattern(self):
        """A pattern that does not cover every layer is a config error."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[0.95])

        with pytest.raises(ValueError, match="must cover every layer"):
            Gemma3nMLP(config, layer_idx=1)
