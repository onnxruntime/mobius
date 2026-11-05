# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build-graph tests for ``--features transposed-matmul``."""

from __future__ import annotations

import zlib

import numpy as np
import onnx_ir as ir
import pytest

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._optimizations import fold_initializers_after_weights

_LAYERS = 2


def _build(*, transposed_matmul: bool, execution_provider: str = "cuda") -> ir.Model:
    from mobius.models.base import CausalLMModel

    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10000.0,
        pad_token_id=0,
        num_hidden_layers=_LAYERS,
        dtype=ir.DataType.FLOAT16,
    )
    # cuda + fp16 fuses attention to GroupQueryAttention and packs Q/K/V.
    return build_from_module(
        CausalLMModel(config),
        config,
        execution_provider=execution_provider,
        transposed_matmul=transposed_matmul,
    )["model"]


def _load_weights(model: ir.Model) -> None:
    """Fill every weight from a seed of its name, then fold as loading does."""
    for name, init in model.graph.initializers.items():
        if init.const_value is None:
            rng = np.random.default_rng(zlib.crc32(name.encode()))
            shape = [int(d) for d in init.shape]
            init.const_value = ir.tensor((rng.standard_normal(shape) * 0.1).astype(np.float16))
    fold_initializers_after_weights(model)


def _ops(model: ir.Model, op_type: str) -> list[ir.Node]:
    return [n for n in model.graph.all_nodes() if n.op_type == op_type]


class TestTransposedMatMul:
    def test_projections_become_fused_matmul(self):
        model = _build(transposed_matmul=True)

        assert len(_ops(model, "GroupQueryAttention")) == _LAYERS
        # Every projection, the LM head included, keeps its [out, in] weight.
        assert not _ops(model, "MatMul")
        fused = _ops(model, "FusedMatMul")
        assert fused and all(n.attributes["transB"].value == 1 for n in fused)
        # The packed Q/K/V weight still reaches the op, as a Concat that
        # weight loading folds.
        packed = [n for n in fused if n.inputs[1].producer() is not None]
        assert len(packed) == _LAYERS
        assert all(n.inputs[1].producer().op_type == "Concat" for n in packed)

        _load_weights(model)
        assert not _ops(model, "Concat")
        assert not _ops(model, "Transpose")
        assert all(n.inputs[1].is_initializer() for n in _ops(model, "FusedMatMul"))

    def test_matches_matmul(self):
        from mobius._testing.ort_inference import OnnxModelSession

        seq = 5
        rng = np.random.default_rng(0)
        feeds = {
            "input_ids": rng.integers(1, 256, (1, seq), dtype=np.int64),
            "attention_mask": np.ones((1, seq), np.int64),
        }
        for i in range(_LAYERS):
            feeds[f"past_key_values.{i}.key"] = np.zeros((1, 2, 0, 16), np.float16)
            feeds[f"past_key_values.{i}.value"] = np.zeros((1, 2, 0, 16), np.float16)

        logits = {}
        for transposed in (False, True):
            model = _build(transposed_matmul=transposed)
            _load_weights(model)
            out = OnnxModelSession(model).run(dict(feeds))
            logits[transposed] = out["logits"].astype(np.float32)

        np.testing.assert_allclose(logits[True], logits[False], rtol=1e-2, atol=1e-3)

    def test_ignored_without_fused_matmul_kernel(self):
        with pytest.warns(UserWarning, match="transposed_matmul=True was requested"):
            model = _build(transposed_matmul=True, execution_provider="cpu")
        assert not _ops(model, "FusedMatMul")
