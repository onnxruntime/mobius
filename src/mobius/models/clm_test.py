# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU parity and fail-closed checkpoint tests for CLM ranking heads."""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from mobius import build, build_from_module
from mobius._configs.clm import CLMConfig
from mobius.integrations.transformers._clm import load_clm_checkpoint
from mobius.models.clm import CLMRankingModel


class _ReferenceHead(torch.nn.Module):
    """Independent Torch computation from pinned upstream make_head."""

    def __init__(self, config):
        super().__init__()
        self.inp = torch.nn.Linear(config.hidden_size, config.width)
        self.hidden = torch.nn.ModuleList(
            torch.nn.Linear(config.width, config.width) for _ in range(config.depth - 2)
        )
        self.norms = torch.nn.ModuleList(
            torch.nn.LayerNorm(config.width) if config.layernorm else torch.nn.Identity()
            for _ in self.hidden
        )
        self.out = torch.nn.Linear(config.width, config.projection_dim)
        self.act = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU, "silu": torch.nn.SiLU}[
            config.activation
        ]()
        self.residual = config.residual

    def forward(self, x):
        x = self.act(self.inp(x))
        for linear, norm in zip(self.hidden, self.norms, strict=True):
            h = self.act(norm(linear(x)))
            x = x + h if self.residual else h
        return torch.nn.functional.normalize(self.out(x), dim=-1)


@pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
@pytest.mark.parametrize(
    "layernorm,residual,depth", [(True, False, 3), (False, True, 4), (False, False, 2)]
)
def test_clm_synthetic_cpu_parity(tmp_path, activation, layernorm, residual, depth):
    torch.manual_seed(17)
    config = CLMConfig(
        hidden_size=8,
        width=6,
        projection_dim=4,
        depth=depth,
        layernorm=layernorm,
        residual=residual,
        activation=activation,
        scale=7.0,
    )
    states, actions = _ReferenceHead(config).eval(), _ReferenceHead(config).eval()
    weights = {
        f"{head}.{name}": value
        for head, module in (("state_head", states), ("action_head", actions))
        for name, value in module.state_dict().items()
    }
    package = build_from_module(
        CLMRankingModel(config),
        config,
        task="contrastive-ranking-heads",
        execution_provider="cpu",
    )
    package.apply_weights(weights)
    package.save(str(tmp_path))
    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    for nstates, ncandidates in [(2, 5), (1, 1), (3, 2)]:
        s, a = torch.randn(nstates, 8), torch.randn(ncandidates, 8)
        s[0] = 0
        a[0] *= 1e-14
        with torch.no_grad():
            zs, za = states(s), actions(a)
            logits = 7 * (zs @ za.T)
            expected = (zs, za, logits, logits.softmax(dim=-1))
        actual = session.run(
            None, {"state_embeddings": s.numpy(), "action_embeddings": a.numpy()}
        )
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, reference.numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(actual[-1].sum(-1), 1, atol=1e-6)


def _checkpoint(config):
    return {
        "cfg": {
            "width": config.width,
            "depth": config.depth,
            "hidden_size": config.hidden_size,
            "projection_dim": config.projection_dim,
            "activation": config.activation,
            "layernorm": config.layernorm,
            "residual": config.residual,
        },
        "state_head": _ReferenceHead(config).state_dict(),
        "action_head": _ReferenceHead(config).state_dict(),
        "logit_scale": torch.tensor(1000.0),
    }


def test_clm_checkpoint_validation(tmp_path):
    config = CLMConfig(hidden_size=8, width=6, projection_dim=4)
    checkpoint = _checkpoint(config)
    path = tmp_path / "heads.pt"
    torch.save(checkpoint, path)
    _, resolved, weights = load_clm_checkpoint(path, config)
    assert resolved.scale == pytest.approx(100.0)
    assert len(weights) == 16
    for corrupt in ("missing", "shape", "nan", "dtype", "scale", "width", "depth"):
        bad = _checkpoint(config)
        if corrupt == "missing":
            del bad["action_head"]["inp.bias"]
        elif corrupt == "shape":
            bad["state_head"]["inp.weight"] = torch.zeros(1)
        elif corrupt == "nan":
            bad["action_head"]["out.bias"][0] = float("nan")
        elif corrupt == "dtype":
            bad["state_head"]["out.bias"] = bad["state_head"]["out.bias"].half()
        elif corrupt == "scale":
            bad["logit_scale"] = torch.tensor(float("nan"))
        elif corrupt == "width":
            bad["cfg"]["width"] = -1
        else:
            bad["cfg"]["width"] = 1
            bad["cfg"]["depth"] = 10_000_000
        torch.save(bad, path)
        with pytest.raises(ValueError):
            load_clm_checkpoint(path, config)


def test_clm_local_public_build(tmp_path):
    config = CLMConfig(width=6, projection_dim=4)
    torch.save(_checkpoint(config), tmp_path / "CLM_v0.1-8B.pt")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "clm",
                "base_model": "Qwen/Qwen3-8B",
                "encoder_pooling": "last-token",
                "embedding_dim": 4096,
                "checkpoints": ["CLM_v0.1-8B.pt"],
            }
        )
    )
    package = build(str(tmp_path), execution_provider="cpu")
    assert package["model"].metadata_props["mobius.export_scope"] == "projection-head-only"
    assert [x.name for x in package["model"].graph.inputs] == [
        "state_embeddings",
        "action_embeddings",
    ]
    with pytest.raises(ValueError, match="not generation"):
        build(str(tmp_path), task="text-generation")
    with pytest.raises(ValueError, match="f32"):
        build(str(tmp_path), dtype="f16")


def test_clm_config_rejects_invalid_dtype():
    with pytest.raises(ValueError, match="f32"):
        dataclasses.replace(CLMConfig(), dtype=ir.DataType.BFLOAT16).validate()


def test_clm_finite_logit_scale_underflow_matches_uniform_upstream_probs(tmp_path):
    torch.manual_seed(29)
    config = CLMConfig(hidden_size=8, width=6, projection_dim=4, depth=2)
    checkpoint = _checkpoint(config)
    checkpoint["logit_scale"] = torch.tensor(-1000.0, dtype=torch.float32)
    path = tmp_path / "underflow.pt"
    torch.save(checkpoint, path)
    module, resolved, weights = load_clm_checkpoint(path, config)
    assert resolved.scale == pytest.approx(0.0)
    package = build_from_module(
        module, resolved, "contrastive-ranking-heads", execution_provider="cpu"
    )
    package.apply_weights(weights)
    package.save(str(tmp_path))
    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    s, a = torch.randn(2, 8), torch.randn(5, 8)
    states, actions = _ReferenceHead(config).eval(), _ReferenceHead(config).eval()
    states.load_state_dict(checkpoint["state_head"])
    actions.load_state_dict(checkpoint["action_head"])
    with torch.no_grad():
        scale = checkpoint["logit_scale"].float().exp().clamp(max=100)
        zs, za = states(s), actions(a)
        logits = scale * (zs @ za.T)
        expected = (zs, za, logits, logits.softmax(-1))
    actual = session.run(None, {"state_embeddings": s.numpy(), "action_embeddings": a.numpy()})
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference.numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(actual[2], np.zeros((2, 5), dtype=np.float32))
    np.testing.assert_array_equal(actual[3], np.full((2, 5), 0.2, dtype=np.float32))


@pytest.mark.parametrize("magnitude", [0.0, 1e-14, 1e-12, 1.0])
def test_clm_projection_normalization_boundary(tmp_path, magnitude):
    config = CLMConfig(hidden_size=8, width=6, projection_dim=4, depth=2)
    reference = _ReferenceHead(config).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.zero_()
        reference.out.bias[0] = magnitude
    weights = {
        f"{head}.{name}": value
        for head in ("state_head", "action_head")
        for name, value in reference.state_dict().items()
    }
    package = build_from_module(CLMRankingModel(config), config, "contrastive-ranking-heads")
    package.apply_weights(weights)
    package.save(str(tmp_path))
    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    x = torch.zeros(1, 8)
    actual = session.run(None, {"state_embeddings": x.numpy(), "action_embeddings": x.numpy()})
    with torch.no_grad():
        expected = reference(x).numpy()
    np.testing.assert_allclose(actual[0], expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(actual[1], expected, atol=1e-7, rtol=1e-6)


@pytest.mark.integration
def test_clm_real_checkpoint_cpu_parity(tmp_path):
    """Opt-in, no network: independently load the real heads via Torch."""
    checkpoint_path = os.environ.get("MOBIUS_CLM_CHECKPOINT")
    if checkpoint_path is None:
        pytest.skip("Set MOBIUS_CLM_CHECKPOINT to the pinned projection checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["cfg"]
    reference_config = CLMConfig(
        hidden_size=cfg["hidden_size"],
        width=cfg["width"],
        depth=cfg["depth"],
        projection_dim=checkpoint["projection_dim"],
        activation=cfg["activation"],
        layernorm=cfg["layernorm"],
        residual=cfg["residual"],
    )
    states, actions = (
        _ReferenceHead(reference_config).eval(),
        _ReferenceHead(reference_config).eval(),
    )
    states.load_state_dict(checkpoint["state_head"])
    actions.load_state_dict(checkpoint["action_head"])
    scale = float(checkpoint["logit_scale"].float().exp().clamp(max=100))
    module, config, weights = load_clm_checkpoint(checkpoint_path, CLMConfig())
    package = build_from_module(
        module, config, "contrastive-ranking-heads", execution_provider="cpu"
    )
    package.apply_weights(weights)
    package.save(str(tmp_path))
    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    torch.manual_seed(29)
    for ns, nc in [(2, 7), (1, 1), (3, 4)]:
        s, a = torch.randn(ns, 4096), torch.randn(nc, 4096)
        s[0] = 0
        a[0] *= 1e-14
        # This is the external upstream Embedder boundary, not graph behavior.
        s = s / (torch.linalg.vector_norm(s, dim=-1, keepdim=True) + 1e-12)
        a = a / (torch.linalg.vector_norm(a, dim=-1, keepdim=True) + 1e-12)
        with torch.no_grad():
            zs, za = states(s), actions(a)
            logits = scale * (zs @ za.T)
            expected = (zs, za, logits, logits.softmax(-1))
        actual = session.run(
            None, {"state_embeddings": s.numpy(), "action_embeddings": a.numpy()}
        )
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, reference.numpy(), atol=1e-4, rtol=1e-4)
        np.testing.assert_array_equal(
            np.argsort(-actual[-1], axis=-1), np.argsort(-expected[-1].numpy(), axis=-1)
        )
