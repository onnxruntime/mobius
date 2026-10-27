# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network-free source-equivalent sampling math and explicit RNG-state tests."""

from __future__ import annotations

import copy

import numpy as np
import onnx_ir as ir
import pytest
import torch
from _test_configs import FISH_OMNI_CONFIG
from onnxscript import GraphBuilder

from mobius._configs.fish_omni import FishOmniConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.generation._fish_omni import (
    _race,
    build_fish_codebook_sampler,
    build_fish_main_sampler,
)
from mobius.integrations.transformers._config_resolver import _dict_to_pretrained_config


def _config():
    return FishOmniConfig.from_transformers(
        _dict_to_pretrained_config(copy.deepcopy(FISH_OMNI_CONFIG))
    )


def _reference(logits, temperature, top_k, top_p, noise):
    values, ids = torch.sort(
        torch.tensor(logits) / torch.tensor(temperature).clamp_min(1e-5)[:, None],
        descending=True,
    )
    ranks = torch.arange(values.shape[-1])
    dropped = (values.softmax(-1).cumsum(-1) > torch.tensor(top_p)[:, None]) | (
        ranks >= torch.tensor(top_k)[:, None]
    )
    dropped[:, 0] = False
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    chosen = (probabilities / torch.tensor(noise)).argmax(-1, keepdim=True)
    return ids.gather(-1, chosen).squeeze(-1).numpy()


@pytest.mark.parametrize("top_k,top_p", [([3, 3], [0.7, 1.0]), ([0, 1], [-1.0, 0.0])])
def test_exponential_race_drops_crossing_token(top_k, top_p):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    logits = builder.input("logits", ir.DataType.FLOAT, [2, 3])
    noise = builder.input("noise", ir.DataType.FLOAT, [2, 3])
    token, valid = _race(
        builder.op,
        logits,
        builder.op.Constant(value_floats=[1.0, 0.7]),
        builder.op.Constant(value_ints=top_k),
        builder.op.Constant(value_floats=top_p),
        noise,
    )
    for name, value, dtype in (
        ("token", token, ir.DataType.INT64),
        ("valid", valid, ir.DataType.BOOL),
    ):
        value.type, value.shape = ir.TensorType(dtype), ir.Shape([2])
        builder.add_output(value, name)
    scores = np.log(np.array([[0.45, 0.35, 0.2], [0.1, 0.65, 0.25]], dtype=np.float32))
    draws = np.array([[100, 1e-6, 1], [1, 0.01, 1]], dtype=np.float32)
    session = OnnxModelSession(ir.Model(graph, ir_version=12))
    try:
        result = session.run({"logits": scores, "noise": draws})
    finally:
        session.close()
    np.testing.assert_array_equal(
        result["token"], _reference(scores, [1.0, 0.7], top_k, top_p, draws)
    )
    assert result["token"][0] == 0
    assert result["valid"].all()


def _noise(seed, counter, width):
    modulus = 2147483647
    result = []
    for rank in range(width):
        state = (
            int(seed) % modulus + 1 + ((int(counter) + rank) % modulus) * 2654435761
        ) % modulus
        for _ in range(3):
            state = (state * 48271) % modulus
            state ^= state // 2048
            state = (state * 48271) % modulus
        uniform = np.float32(state + 1) / np.float32(modulus + 2)
        uniform = np.clip(uniform, np.float32(1.1754943508222875e-38), np.float32(1 - 2**-24))
        result.append(-np.log(uniform))
    return np.array(result, dtype=np.float32)


def test_main_sampler_rerolls_semantics_but_not_eos():
    config = _config()
    component = build_fish_main_sampler(config)
    candidates = sorted(
        [
            config.eos_token_id,
            *range(config.semantic_start_token_id, config.semantic_end_token_id + 1),
        ]
    )
    width = len(candidates)
    logits = np.random.default_rng(42).normal(size=(2, config.vocab_size)).astype(np.float32)
    logits[0, config.eos_token_id] = -100
    logits[1, config.eos_token_id] = 100
    seed, counter = np.array([42, 2026], np.int64), np.array([0, 17], np.int64)
    noise = np.stack([_noise(s, c, width) for s, c in zip(seed, counter)])
    retry_noise = np.stack([_noise(s, c + width, width) for s, c in zip(seed, counter)])
    params = {
        "temperature": np.array([0.8, 0.8], np.float32),
        "top_k": np.array([5, 5], np.int64),
        "top_p": np.array([0.9, 0.9], np.float32),
    }
    first = np.array(candidates)[
        _reference(
            logits[:, candidates],
            params["temperature"],
            params["top_k"],
            params["top_p"],
            noise,
        )
    ]
    retry = np.array(candidates)[
        _reference(logits[:, candidates], [1.0, 1.0], [30, 30], [0.95, 0.95], retry_noise)
    ]
    history = np.full((2, 16), -1, np.int64)
    history[:, -1] = first
    session = OnnxModelSession(component.model)
    try:
        feeds = {
            **params,
            "logits": logits,
            "seed": seed,
            "counter": counter,
            "history": history,
        }
        result = session.run(feeds)
        repeated = session.run(feeds)
    finally:
        session.close()
    np.testing.assert_array_equal(result["token"], [retry[0], config.eos_token_id])
    np.testing.assert_array_equal(result["next_counter"], counter + 2 * width)
    assert result["valid"].all()
    for name in result:
        np.testing.assert_array_equal(result[name], repeated[name])


@pytest.mark.parametrize("invalid", ["logits", "temperature", "top_p"])
def test_codebook_sampler_reports_invalid_inputs_without_advancing_rng(invalid):
    component = build_fish_codebook_sampler(_config())
    feeds = {
        "logits": np.arange(16, dtype=np.float32)[None],
        "temperature": np.array([1.0], np.float32),
        "top_k": np.array([16], np.int64),
        "top_p": np.array([1.0], np.float32),
        "seed": np.array([42], np.int64),
        "counter": np.array([17], np.int64),
    }
    feeds[invalid].flat[0] = np.nan
    session = OnnxModelSession(component.model)
    try:
        result = session.run(feeds)
    finally:
        session.close()
    assert result["token"].tolist() == [-1]
    assert result["valid"].tolist() == [False]
    assert result["next_counter"].tolist() == [17]


def test_main_sampler_rejects_nonfinite_noncandidate_logits():
    config = _config()
    logits = np.zeros((1, config.vocab_size), dtype=np.float32)
    logits[0, 0] = np.nan
    session = OnnxModelSession(build_fish_main_sampler(config).model)
    try:
        result = session.run(
            {
                "logits": logits,
                "temperature": np.array([1.0], np.float32),
                "top_k": np.array([16], np.int64),
                "top_p": np.array([1.0], np.float32),
                "seed": np.array([42], np.int64),
                "counter": np.array([17], np.int64),
                "history": np.full((1, 16), -1, np.int64),
            }
        )
    finally:
        session.close()
    assert result["token"].tolist() == [-1]
    assert result["valid"].tolist() == [False]
    assert result["next_counter"].tolist() == [17]
