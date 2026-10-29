# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Drama 3's drop-crossing nucleus sampler and semantic-only repetition reroll.

Policies use explicit counter RNG state, not Torch/Philox seed-bit identity.
They perform sampling in ONNX; the caller still orchestrates temporal decoding.
"""

from __future__ import annotations

from typing import NamedTuple

import onnx_ir as ir
from onnxscript import GraphBuilder, OpBuilder

from mobius._configs.fish_omni import FishOmniConfig
from mobius.generation._policy_components import (
    _RNG_MODULUS,
    PolicyComponent,
    _component,
    _counter_uniform,
    _make_graph,
    _set_public_shape,
)


class FishAudioSamplingInputs(NamedTuple):
    """Tensor controls for the single-pass temporal tower task."""

    generate: ir.Value
    do_sample: ir.Value
    temperature: ir.Value
    top_k: ir.Value
    top_p: ir.Value
    seed: ir.Value
    counter: ir.Value


def sample_codebook_or_greedy(
    op: OpBuilder,
    logits: ir.Value,
    sampling: FishAudioSamplingInputs,
    counter: ir.Value,
    *,
    index: int,
    vocabulary: int,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Skip RNG/sorting entirely for greedy generation and prompt ingestion."""
    active = op.And(sampling.generate, sampling.do_sample)

    def stochastic(branch_op):
        noises, _ = _noise(branch_op, logits, sampling.seed, counter, 1)
        return _race(
            branch_op,
            logits,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            noises[0],
        )

    def greedy(branch_op):
        float_logits = branch_op.Cast(logits, to=ir.DataType.FLOAT)
        return (
            branch_op.ArgMax(float_logits, axis=-1, keepdims=False),
            _all_rows(branch_op, _finite(branch_op, float_logits)),
        )

    branches = []
    for suffix, body in (("random", stochastic), ("greedy", greedy)):
        name = f"codebook_{index}_{suffix}"
        op.builder.push_module(name)
        branch = op.builder.subgraph(
            body,
            inputs=[],
            outputs=[ir.Value(name=f"{name}_token"), ir.Value(name=f"{name}_valid")],
            name=name,
        )
        op.builder.pop_module()
        branches.append(branch)
    token, valid = op.If(active, then_branch=branches[0], else_branch=branches[1], _outputs=2)
    advanced = op.Mod(
        op.Add(op.Mod(counter, _RNG_MODULUS, fmod=0), vocabulary), _RNG_MODULUS, fmod=0
    )
    return token, op.Where(active, advanced, counter), valid


def _finite(op: OpBuilder, value: ir.Value) -> ir.Value:
    return op.Not(op.Or(op.IsNaN(value), op.IsInf(value)))


def _all_rows(op: OpBuilder, value: ir.Value) -> ir.Value:
    return op.Equal(
        op.ReduceMin(op.Cast(value, to=ir.DataType.INT64), [-1], keepdims=False), 1
    )


def _race(
    op: OpBuilder,
    logits: ir.Value,
    temperature: ir.Value,
    top_k: ir.Value,
    top_p: ir.Value,
    noise: ir.Value,
) -> tuple[ir.Value, ir.Value]:
    """Fish Speech's sorted-probability exponential race with explicit noise."""
    scaled = op.Div(
        op.Cast(logits, to=ir.DataType.FLOAT),
        op.Unsqueeze(op.Max(temperature, 1e-5), [-1]),
    )
    width = op.Shape(scaled, start=1, end=2)
    values, ids = op.TopK(scaled, width, axis=-1, sorted=1, _outputs=2)
    ranks = op.Range(0, op.Squeeze(width, [0]), 1)
    cumulative = op.CumSum(op.Softmax(values, axis=-1), -1)
    dropped = op.And(
        op.Greater(ranks, 0),
        op.Or(
            op.Greater(cumulative, op.Unsqueeze(top_p, [-1])),
            op.GreaterOrEqual(ranks, op.Unsqueeze(top_k, [-1])),
        ),
    )
    probabilities = op.Softmax(op.Where(dropped, float("-inf"), values), axis=-1)
    choice = op.ArgMax(op.Div(probabilities, noise), axis=-1, keepdims=False)
    token = op.Squeeze(op.GatherElements(ids, op.Unsqueeze(choice, [-1]), axis=-1), [-1])
    valid = op.And(
        _all_rows(op, _finite(op, scaled)),
        op.And(
            op.And(_finite(op, temperature), _finite(op, top_p)),
            _all_rows(op, op.And(_finite(op, noise), op.Greater(noise, 0.0))),
        ),
    )
    return token, valid


def _noise(
    op: OpBuilder, logits: ir.Value, seed: ir.Value, counter: ir.Value, draws: int
) -> tuple[list[ir.Value], ir.Value]:
    width = op.Squeeze(op.Shape(logits, start=1, end=2), [0])
    ranks = op.Range(0, width, 1)
    # Bound integer multiplication in the shared RNG, including arbitrary
    # signed seeds/counters. Its counter stream is periodic modulo 2**31-1.
    seed = op.Mod(seed, _RNG_MODULUS, fmod=0)
    counter = op.Mod(counter, _RNG_MODULUS, fmod=0)
    noises = []
    for draw in range(draws):
        offsets = op.Mod(
            op.Add(op.Unsqueeze(op.Add(counter, op.Mul(draw, width)), [-1]), ranks),
            _RNG_MODULUS,
            fmod=0,
        )
        uniform = _counter_uniform(op, op.Unsqueeze(seed, [-1]), offsets)
        # Float32 conversion can round a uniform to 1; keep -log(U) positive.
        uniform = op.Clip(uniform, 1.1754943508222875e-38, 1.0 - 2**-24)
        noises.append(op.Neg(op.Log(uniform)))
    next_counter = op.Mod(op.Add(counter, op.Mul(draws, width)), _RNG_MODULUS, fmod=0)
    return noises, next_counter


def _inputs(
    builder: GraphBuilder, dtype: ir.DataType, width: int
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]:
    return (
        builder.input("logits", dtype, ["batch", width]),
        builder.input("temperature", ir.DataType.FLOAT, ["batch"]),
        builder.input("top_k", ir.DataType.INT64, ["batch"]),
        builder.input("top_p", ir.DataType.FLOAT, ["batch"]),
        builder.input("seed", ir.DataType.INT64, ["batch"]),
        builder.input("counter", ir.DataType.INT64, ["batch"]),
    )


def _output(builder: GraphBuilder, value: ir.Value, name: str, dtype: ir.DataType) -> None:
    value.type = ir.TensorType(dtype)
    _set_public_shape(value, ["batch"])
    builder.add_output(value, name)


_CONTRACT = {
    "sampling": "sorted-exponential-race",
    "top_p": "drop-crossing-token-except-rank-zero",
    "temperature_floor": 1e-5,
    "rng": "mobius.counter-uniform-exponential@1",
    "counter_modulus": _RNG_MODULUS,
    "torch_seed_identity": False,
    "invalid_input": "token=-1, valid=false, counter unchanged; caller must stop",
}


def build_fish_codebook_sampler(config: FishOmniConfig) -> PolicyComponent:
    """Sample one residual code; invoke after each conditioned tower projection."""
    config.validate()
    assert config.tower is not None
    graph, builder = _make_graph("fish_codebook_sampler")
    op = builder.op
    logits, temperature, top_k, top_p, seed, counter = _inputs(
        builder, config.dtype, config.tower.vocab_size
    )
    noises, advanced = _noise(op, logits, seed, counter, 1)
    token, valid = _race(op, logits, temperature, top_k, top_p, noises[0])
    _output(builder, op.Where(valid, token, -1), "token", ir.DataType.INT64)
    _output(builder, valid, "valid", ir.DataType.BOOL)
    _output(builder, op.Where(valid, advanced, counter), "next_counter", ir.DataType.INT64)
    return _component("mobius.policy.fish-codebook-sampler@1", graph, dict(_CONTRACT))


def build_fish_main_sampler(config: FishOmniConfig) -> PolicyComponent:
    """Draw main/retry every step and reroll only repeated semantic main tokens."""
    config.validate()
    assert config.tower is not None
    assert config.eos_token_id is not None
    graph, builder = _make_graph("fish_main_sampler")
    op = builder.op
    logits, temperature, top_k, top_p, seed, counter = _inputs(
        builder, config.dtype, config.vocab_size
    )
    history = builder.input("history", ir.DataType.INT64, ["batch", 16])
    candidates = op.Constant(
        value_ints=sorted(
            [
                config.eos_token_id,
                *range(config.semantic_start_token_id, config.semantic_end_token_id + 1),
            ]
        )
    )
    selected_logits = op.Gather(logits, candidates, axis=-1)
    noises, advanced = _noise(op, selected_logits, seed, counter, 2)
    first, first_valid = _race(op, selected_logits, temperature, top_k, top_p, noises[0])
    retry, retry_valid = _race(
        op,
        selected_logits,
        op.Expand(op.Constant(value_float=1.0), op.Shape(temperature)),
        op.Expand(op.Constant(value_int=30), op.Shape(top_k)),
        op.Expand(op.Constant(value_float=0.95), op.Shape(top_p)),
        noises[1],
    )
    first, retry = op.Gather(candidates, first), op.Gather(candidates, retry)
    repeated = op.Greater(
        op.ReduceSum(
            op.Cast(op.Equal(op.Unsqueeze(first, [-1]), history), to=ir.DataType.INT64),
            [-1],
            keepdims=False,
        ),
        0,
    )
    semantic = op.And(
        op.GreaterOrEqual(first, config.semantic_start_token_id),
        op.LessOrEqual(first, config.semantic_end_token_id),
    )
    main = op.Where(op.And(repeated, semantic), retry, first)
    valid = op.And(
        op.And(first_valid, retry_valid),
        _all_rows(op, _finite(op, op.Cast(logits, to=ir.DataType.FLOAT))),
    )
    _output(builder, op.Where(valid, main, -1), "token", ir.DataType.INT64)
    _output(
        builder,
        op.Where(
            valid,
            op.Clip(
                op.Sub(main, config.semantic_start_token_id), 0, config.tower.vocab_size - 1
            ),
            -1,
        ),
        "semantic_code",
        ir.DataType.INT64,
    )
    _output(builder, valid, "valid", ir.DataType.BOOL)
    _output(builder, op.Where(valid, advanced, counter), "next_counter", ir.DataType.INT64)
    return _component("mobius.policy.fish-main-sampler@1", graph, dict(_CONTRACT))
