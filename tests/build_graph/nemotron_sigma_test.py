# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network-free graph and host cache-contract checks for continuous Sigma."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from mobius import ModelPackage, build_from_module
from mobius._configs import RoPEConfig
from mobius._testing.golden import load_test_case
from mobius._testing.nemotron_sigma_golden import assert_golden
from mobius.generation.nemotron_sigma import NemotronSigmaGenerator
from mobius.models.nemotron_sigma import NemotronSigmaConfig, NemotronSigmaModel


def tiny_config(**overrides):
    fields = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        hidden_act="silu",
        max_position_embeddings=256,
        rope=RoPEConfig(rope_theta=1000000.0),
    )
    fields.update(overrides)
    return NemotronSigmaConfig(**fields)


def test_sigma_graph_preserves_bidirectional_attention_and_heads():
    config = tiny_config()
    package = build_from_module(NemotronSigmaModel(config), config, task="nemotron-sigma")
    model = package["model"]
    heads = {output.name for output in model.graph.outputs}
    assert {"ar_logits", "diffusion_logits", "denoised_mean", "gamma_bounds"} <= heads
    attentions = [node for node in model.graph if node.op_type == "Attention"]
    assert len(attentions) == config.num_hidden_layers
    assert all(node.attributes["is_causal"].as_int() == 0 for node in attentions)
    assert all(node.inputs[3] is not None for node in attentions)
    assert "replaid_selfcond_linear.weight" in model.graph.initializers
    assert "encoder.embed_tokens.weight" in model.graph.initializers
    assert "gamma_bounds.endpoint_values" in model.graph.initializers


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_sigma_rejects_unqualified_precision_at_construction_and_build(dtype):
    config = tiny_config()
    with pytest.raises(ValueError, match="FLOAT compute"):
        NemotronSigmaModel(dataclasses.replace(config, dtype=dtype))
    module = NemotronSigmaModel(config)
    config.dtype = dtype
    with pytest.raises(ValueError, match="FLOAT compute"):
        build_from_module(module, config, task="nemotron-sigma")


@pytest.mark.parametrize("accept_all", [True, False])
def test_sigma_roundtrip_acceptance_and_final_truncation(tmp_path, accept_all):
    config = tiny_config()
    package = build_from_module(NemotronSigmaModel(config), config, task="nemotron-sigma")
    # Equal heads deliberately accept every proposal. This exercises the
    # all-accepted/bonus branch and final truncation without mocking sessions.
    state = {
        name: torch.zeros(tuple(value.shape), dtype=torch.float32)
        for name, value in package["model"].graph.initializers.items()
        if value.const_value is None
    }
    state["gamma_bounds.endpoint_values"] = torch.tensor([-5.415, 4.8549])
    state["replaid_embedding_matrix.matrix"] = torch.ones(config.vocab_size, config.embed_dim)
    if not accept_all:
        # Positive noise proposes token 1, but the independent AR head picks 0.
        state["replaid_embedding_matrix.matrix"][0] *= -1
    package.apply_weights(state)
    package.save(str(tmp_path))
    restored = ModelPackage.load(str(tmp_path))
    assert restored["model"].metadata_props["mobius.sigma"]
    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    generator = NemotronSigmaGenerator(session)
    ids = np.asarray([[0, 1, 2]], dtype=np.int64)
    count = 65 if accept_all else 4
    result, stats = generator.generate_speculative(
        ids,
        attention_mask=np.asarray([[0, 1, 1]]),
        max_new_tokens=count,
        noise_sampler=lambda shape: np.ones(shape, np.float64),
        return_stats=True,
    )
    assert result.shape == (1, count + 3)
    np.testing.assert_array_equal(result[:, :3], ids)
    np.testing.assert_array_equal(result[:, 3:], 0)
    expected_stats = (
        {"rounds": 2, "accepted": 64, "committed": 65, "mean_accepted_length": 32.0}
        if accept_all
        else {"rounds": 4, "accepted": 0, "committed": 4, "mean_accepted_length": 0.0}
    )
    assert stats == expected_stats
    native, evaluations = generator.generate(
        ids, max_new_tokens=35, noise_sampler=lambda shape: np.ones(shape, np.float64)
    )
    assert native.shape == (1, 38)
    assert evaluations == 6
    with pytest.raises(ValueError, match="left padding"):
        generator.generate_speculative(ids, attention_mask=[[1, 0, 1]], max_new_tokens=1)
    with pytest.raises(ValueError, match="float64 noise"):
        generator.generate_speculative(
            ids, max_new_tokens=1, noise_sampler=lambda shape: np.ones(shape, np.float32)
        )


@pytest.mark.parametrize("corruption", ["truncated", "revision", "dtype"])
def test_sigma_golden_rejects_wrong_horizon_and_provenance(tmp_path, corruption):
    case = load_test_case(
        Path(__file__).parents[2] / "testdata" / "cases" / "causal-lm" / "nemotron-sigma.yaml"
    )
    golden = {
        "model_id": case.model_id,
        "revision": case.revision,
        "dtype": case.dtype,
        "speculative_tokens": [1] * 65,
        "diffusion_tokens": [1] * 65,
    }
    if corruption == "truncated":
        golden["diffusion_tokens"] = [1] * 32
    elif corruption == "revision":
        golden["revision"] = "0" * 40
    else:
        golden["dtype"] = "bfloat16"
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(golden))
    with pytest.raises(ValueError, match="regenerate"):
        assert_golden({}, path, case=case, level="L5")
