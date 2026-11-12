# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mistral Timestral time-series forecasting, without tracing PyTorch.

Replicates ``timestral.model.TimestralModel.predict_block`` for target-only
univariate and multivariate series. Inputs are packed rows of context values,
observation masks, group IDs and a horizon measured in patches. The graph returns
all quantiles in the last context patch's normalization frame. Advancing-frame
refinement and median-feedback rollout live in :mod:`mobius.integrations.timestral`.
Known-future and past-only covariates are not part of this initial export contract.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._configs._base import _resolve_dtype_value
from mobius.components import Linear

TIMESTRAL_MODEL_ID = "mistralai/Timestral-5M"
TIMESTRAL_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"
TIMESTRAL_REFERENCE_REVISION = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"

_AXES = {
    "time": (True,),
    "variate": (False,),
    "time_variate": (True, False),
    "variate_time": (False, True),
}


@dataclasses.dataclass
class TimestralConfig(ArchitectureConfig):
    """Timestral inference geometry and normalization settings."""

    model_type: str = "timestral"
    patch_size: int = 32
    hidden_size: int = 192
    num_attention_heads: int = 3
    intermediate_size: int = 512
    block_schedule: tuple[str, ...] = ("time", "time_variate")
    quantile_knots: tuple[float, ...] = (0.1, 0.5, 0.9)
    block_size: int = 32
    context_length: int = 32768
    rope_theta: float = 10000.0
    rope_partial_factor: float = 0.5
    norm_eps: float = 1e-6
    attn_bias: bool = False
    mlp_bias: bool = False
    time_attn_gate: str = "sdpa_sigmoid"
    variate_attn_gate: str = "sdpa_sigmoid"
    minimum_scale: float = 1e-6
    use_asinh: bool = True
    stabilize_with_global: bool = False
    attend_missing_patches: bool = True
    quantile_width_factor: float = 2.0

    @classmethod
    def from_transformers(cls, config, parent_config=None):
        names = {
            "d_model": "hidden_size",
            "num_heads": "num_attention_heads",
            "mlp_hidden_dim": "intermediate_size",
        }
        fields = {}
        for field in dataclasses.fields(cls):
            if field.name == "dtype":
                continue
            source = next((k for k, v in names.items() if v == field.name), field.name)
            if hasattr(config, source):
                fields[field.name] = getattr(config, source)
        for name in ("block_schedule", "quantile_knots"):
            if name in fields:
                fields[name] = tuple(fields[name])
        source_dtype = getattr(config, "dtype", None)
        if source_dtype is not None:
            resolved_dtype = _resolve_dtype_value(source_dtype)
            if resolved_dtype is None:
                raise ValueError(f"Unknown Timestral dtype {source_dtype!r}; use float32.")
            fields["dtype"] = resolved_dtype
        result = cls(**fields)
        result.validate()
        return result

    def validate(self) -> None:
        """Reject unsupported checkpoint semantics rather than ignoring them."""
        if self.dtype != ir.DataType.FLOAT:
            raise ValueError(
                f"Timestral currently supports float32 only, got {self.dtype}; use dtype='f32'."
            )
        if self.quantization is not None or self.component_quantization is not None:
            raise ValueError(
                "Timestral quantized checkpoints are not supported; use float32 weights."
            )
        if (
            self.patch_size <= 0
            or self.block_size <= 0
            or self.context_length < self.patch_size
            or self.hidden_size <= 0
            or self.num_attention_heads <= 0
            or self.hidden_size % self.num_attention_heads
            or self.intermediate_size <= 0
        ):
            raise ValueError(
                "Invalid Timestral geometry; use positive patch/block/hidden sizes and divisible head counts."
            )
        rotary = int(self.hidden_size // self.num_attention_heads * self.rope_partial_factor)
        if rotary <= 0 or rotary % 2 or rotary > self.hidden_size // self.num_attention_heads:
            raise ValueError(
                "Invalid Timestral partial RoPE; its rotated head width must be positive, even, and at most head_dim."
            )
        if not self.block_schedule or any(kind not in _AXES for kind in self.block_schedule):
            raise ValueError(
                f"Unknown Timestral block_schedule {self.block_schedule!r}; expected layer kinds {tuple(_AXES)}."
            )
        if any(
            gate not in ("none", "sdpa_sigmoid")
            for gate in (self.time_attn_gate, self.variate_attn_gate)
        ):
            raise ValueError(
                "Unsupported Timestral attention gate; use 'none' or 'sdpa_sigmoid'."
            )
        if (
            0.5 not in self.quantile_knots
            or not all(0 < q < 1 and math.isfinite(q) for q in self.quantile_knots)
            or any(a >= b for a, b in zip(self.quantile_knots, self.quantile_knots[1:]))
        ):
            raise ValueError(
                "Timestral quantile_knots must be strictly increasing inside (0, 1) and contain 0.5."
            )
        if self.stabilize_with_global:
            raise ValueError(
                "Timestral advancing-frame inference requires stabilize_with_global=False."
            )
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.minimum_scale,
                self.norm_eps,
                self.rope_theta,
                self.quantile_width_factor,
            )
        ):
            raise ValueError(
                "Timestral scale floor, norm epsilon and RoPE base must be positive."
            )


def _norm(op, x, scale, eps):
    return op.RMSNormalization(x, scale, axis=-1, epsilon=eps, stash_type=ir.DataType.FLOAT)


class _Norm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = nn.Parameter([config.hidden_size])
        self.eps = config.norm_eps

    def forward(self, op, x):
        return _norm(op, x, self.scale, self.eps)


class _Residual(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias):
        super().__init__()
        self.hidden_layer = Linear(input_size, hidden_size, bias=bias)
        self.output_layer = Linear(hidden_size, output_size, bias=bias)
        self.residual_layer = Linear(input_size, output_size, bias=bias)

    def forward(self, op, x):
        h = self.hidden_layer(op, x)
        return op.Add(
            self.output_layer(op, op.Mul(h, op.Sigmoid(h))), self.residual_layer(op, x)
        )


class _Representation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_proj = _Residual(
            2 * config.patch_size, 4 * config.hidden_size, config.hidden_size, True
        )
        self.forecast_query = nn.Parameter([config.hidden_size])
        self.config = config

    def forward(self, op, scaled, observed, horizon):
        c = self.config
        values = op.Reshape(scaled, [0, -1, c.patch_size])
        missing = op.Reshape(
            op.Cast(op.Not(observed), to=ir.DataType.FLOAT), [0, -1, c.patch_size]
        )
        enc = self.patch_proj(op, op.Concat(values, missing, axis=-1))
        dec = op.Expand(
            op.Reshape(self.forecast_query, [1, 1, c.hidden_size]),
            op.Concat(op.Shape(scaled, start=0, end=1), horizon, [c.hidden_size], axis=0),
        )
        return enc, dec


class _Attention(nn.Module):
    def __init__(self, config, is_time):
        super().__init__()
        self.wQKV = Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attn_bias)
        self.wO = Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)
        gate = config.time_attn_gate if is_time else config.variate_attn_gate
        self._gate = (
            Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)
            if gate == "sdpa_sigmoid"
            else None
        )
        self.config = config
        self.is_time = is_time

    def forward(self, op, x, time_mask, variate_mask, cos, sin):
        c = self.config
        inputs = x if self.is_time else op.Transpose(x, perm=[1, 0, 2])
        hd = c.hidden_size // c.num_attention_heads
        # Checkpoint QKV layout is (qkv, head_dim, heads), not (qkv, heads, head_dim).
        packed = op.Reshape(self.wQKV(op, inputs), [0, 0, 3, hd, c.num_attention_heads])
        q, k, v = [
            op.Transpose(op.Gather(packed, i, axis=2), perm=[0, 3, 1, 2]) for i in range(3)
        ]
        if self.is_time:
            rotated = []
            for value in (q, k):
                half = op.Concat(
                    op.Neg(op.Slice(value, [hd // 2], [hd], [-1])),
                    op.Slice(value, [0], [hd // 2], [-1]),
                    axis=-1,
                )
                rotated.append(op.Add(op.Mul(value, cos), op.Mul(half, sin)))
            q, k = rotated
        out, _, _ = op.Attention(
            q,
            k,
            v,
            time_mask if self.is_time else variate_mask,
            q_num_heads=c.num_attention_heads,
            kv_num_heads=c.num_attention_heads,
            is_causal=0,
            _outputs=3,
        )
        out = op.Reshape(op.Transpose(out, perm=[0, 2, 1, 3]), [0, 0, c.hidden_size])
        if self._gate is not None:
            out = op.Mul(out, op.Sigmoid(self._gate(op, inputs)))
        if not self.is_time:
            out = op.Transpose(out, perm=[1, 0, 2])
        return self.wO(op, out)


class _MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w12 = Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=config.mlp_bias
        )
        self.w3 = Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, op, x):
        gate, up = op.Split(self.w12(op, x), axis=-1, num_outputs=2, _outputs=2)
        return self.w3(op, op.Mul(gate, op.Mul(up, op.Sigmoid(up))))


class _Block(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.attn_norms = nn.ModuleList([_Norm(config) for _ in _AXES[kind]])
        self.attns = nn.ModuleList([_Attention(config, axis) for axis in _AXES[kind]])
        self.mlp_norm = _Norm(config)
        self.mlp = _MLP(config)

    def forward(self, op, x, time_mask, variate_mask, cos, sin):
        for norm, attention in zip(self.attn_norms, self.attns):
            x = op.Add(x, attention(op, norm(op, x), time_mask, variate_mask, cos, sin))
        return op.Add(x, self.mlp(op, self.mlp_norm(op, x)))


class _Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([_Block(config, kind) for kind in config.block_schedule])
        self.out_norm = _Norm(config)
        self.config = config

    def forward(self, op, x, time_mask, variate_mask, positions):
        c = self.config
        hd = c.hidden_size // c.num_attention_heads
        rotary = int(hd * c.rope_partial_factor)
        inv = np.zeros(hd // 2, dtype=np.float32)
        inv[: rotary // 2] = 1.0 / c.rope_theta ** (
            np.arange(0, rotary, 2, dtype=np.float32) / rotary
        )
        # Full-head half-pair rotation with a zero frequency tail, not prefix-only RoPE.
        freq = op.Mul(
            op.Unsqueeze(op.Cast(positions, to=ir.DataType.FLOAT), [-1]),
            op.Constant(value=ir.tensor(inv)),
        )
        freq = op.Unsqueeze(op.Concat(freq, freq, axis=-1), [1])
        cos, sin = op.Cos(freq), op.Sin(freq)
        for layer in self.layers:
            x = layer(op, x, time_mask, variate_mask, cos, sin)
        return self.out_norm(op, x)


class _Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = _Residual(
            config.hidden_size,
            4 * config.hidden_size,
            config.patch_size * len(config.quantile_knots),
            False,
        )
        self.config = config

    def forward(self, op, x):
        c = self.config
        q = len(c.quantile_knots)
        raw = op.Reshape(self.projection(op, x), [0, -1, q])
        m = c.quantile_knots.index(0.5)
        median = op.Slice(raw, [m], [m + 1], [-1])
        outputs = []
        for i in range(q):
            if i == m:
                outputs.append(median)
                continue
            indices = range(i, m) if i < m else range(m + 1, i + 1)
            increments = []
            for j in indices:
                width = (
                    c.quantile_knots[j + 1] - c.quantile_knots[j]
                    if j < m
                    else c.quantile_knots[j] - c.quantile_knots[j - 1]
                )
                increments.append(
                    op.Mul(
                        op.Softplus(op.Slice(raw, [j], [j + 1], [-1])),
                        width * c.quantile_width_factor,
                    )
                )
            delta = increments[0]
            for increment in increments[1:]:
                delta = op.Add(delta, increment)
            outputs.append(op.Sub(median, delta) if i < m else op.Add(median, delta))
        return op.Concat(*outputs, axis=-1)


class TimestralModel(nn.Module):
    """Timestral target-only multivariate quantile forecasting (Mistral AI).

    Replicates ``timestral.model.TimestralModel.predict_block`` using native ONNX
    operators, including causal Welford scaling and time/variate attention.
    """

    default_task = "time-series-forecasting"
    category = "Time Series"
    config_class = TimestralConfig

    def __init__(self, config: TimestralConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.representation = _Representation(config)
        self.transformer = _Transformer(config)
        self.head = _Head(config)

    def forward(self, op, context, observed_mask, group_ids, horizon_patches):
        self.config.validate()
        c = self.config
        ps = c.patch_size
        clean = op.Where(observed_mask, context, 0.0)
        x = op.Cast(clean, to=ir.DataType.DOUBLE)
        weights = op.Cast(observed_mask, to=ir.DataType.DOUBLE)
        one = op.Constant(value=ir.tensor(np.array(1.0, dtype=np.float64)))
        zero = op.Constant(value=ir.tensor(np.array(0.0, dtype=np.float64)))
        floor = op.Constant(value=ir.tensor(np.array(c.minimum_scale, dtype=np.float64)))
        count = op.Max(op.CumSum(weights, -1), one)
        means = op.Div(op.CumSum(x, -1), count)
        previous = op.Pad(op.Slice(means, [0], [-1], [1]), [0, 1, 0, 0])
        increments = op.Mul(op.Mul(op.Sub(x, previous), op.Sub(x, means)), weights)
        variance = op.Div(op.CumSum(increments, -1), op.Max(op.Sub(count, one), one))
        scales = op.Max(op.Sqrt(op.Max(variance, zero)), floor)
        endpoint = op.Range(ps - 1, op.Squeeze(op.Shape(context, start=1, end=2), [0]), ps)
        loc = op.Cast(op.Gather(means, endpoint, axis=1), to=ir.DataType.FLOAT)
        scale = op.Cast(op.Gather(scales, endpoint, axis=1), to=ir.DataType.FLOAT)
        shape = op.Concat(op.Shape(loc), [ps], axis=0)
        loc_steps = op.Reshape(op.Expand(op.Unsqueeze(loc, [-1]), shape), op.Shape(context))
        scale_steps = op.Reshape(
            op.Expand(op.Unsqueeze(scale, [-1]), shape), op.Shape(context)
        )
        scaled = op.Where(
            observed_mask,
            op.Div(
                op.Sub(clean, loc_steps), op.Max(scale_steps, float(np.finfo(np.float32).eps))
            ),
            0.0,
        )
        if c.use_asinh:
            scaled = op.Asinh(scaled)
        enc, dec = self.representation(op, scaled, observed_mask, horizon_patches)
        n = op.Shape(enc, start=1, end=2)
        tokens = op.Concat(enc, dec, axis=1)  # (rows, context_patches + horizon_patches, D)
        positions = op.Range(0, op.Squeeze(op.Shape(tokens, start=1, end=2), [0]), 1)
        obs = op.Greater(
            op.ReduceMax(
                op.Reshape(op.Cast(observed_mask, to=ir.DataType.INT64), [0, -1, ps]),
                [-1],
                keepdims=0,
            ),
            0,
        )
        valid = (
            op.Greater(op.CumSum(op.Cast(obs, to=ir.DataType.INT64), -1), 0)
            if c.attend_missing_patches
            else obs
        )
        dec_valid = op.Cast(
            op.ConstantOfShape(
                op.Concat(op.Shape(context, start=0, end=1), horizon_patches, axis=0),
                value=ir.tensor([1], dtype=ir.DataType.INT64),
            ),
            to=ir.DataType.BOOL,
        )
        valid = op.Concat(valid, dec_valid, axis=1)
        row_equal = op.Equal(op.Unsqueeze(group_ids, [1]), op.Unsqueeze(group_ids, [0]))
        group_observed = op.MatMul(
            op.Cast(row_equal, to=ir.DataType.FLOAT), op.Cast(obs, to=ir.DataType.FLOAT)
        )
        first = op.ArgMax(group_observed, axis=-1, keepdims=1)
        position_ids = op.Sub(op.Unsqueeze(positions, [0]), first)
        query = op.Unsqueeze(positions, [1])
        key = op.Unsqueeze(positions, [0])
        diagonal = op.Equal(query, key)
        causal = op.GreaterOrEqual(query, key)
        time_mask = op.Unsqueeze(
            op.Or(op.And(causal, op.Unsqueeze(valid, [1])), diagonal), [1]
        )
        row_positions = op.Range(0, op.Squeeze(op.Shape(group_ids), [0]), 1)
        row_diagonal = op.Equal(
            op.Unsqueeze(row_positions, [1]), op.Unsqueeze(row_positions, [0])
        )
        variate_mask = op.Unsqueeze(
            op.Or(
                op.And(
                    op.Unsqueeze(row_equal, [0]),
                    op.Unsqueeze(op.Transpose(valid, perm=[1, 0]), [1]),
                ),
                row_diagonal,
            ),
            [1],
        )
        hidden = self.transformer(op, tokens, time_mask, variate_mask, position_ids)
        decoded = op.Slice(hidden, n, [np.iinfo(np.int64).max], [1])
        quantiles = self.head(op, decoded)
        if c.use_asinh:
            quantiles = op.Sinh(quantiles)
        anchor_loc = op.Unsqueeze(op.Slice(loc, [-1], [np.iinfo(np.int64).max], [1]), [-1])
        anchor_scale = op.Unsqueeze(op.Slice(scale, [-1], [np.iinfo(np.int64).max], [1]), [-1])
        quantiles = op.Add(op.Mul(quantiles, anchor_scale), anchor_loc)
        # Match the reference's sinh-overflow policy without hiding NaNs.
        finite = op.Not(op.Or(op.IsInf(quantiles), op.IsNaN(quantiles)))
        high = op.ReduceMax(op.Where(finite, quantiles, -float("inf")), [-1], keepdims=1)
        low = op.ReduceMin(op.Where(finite, quantiles, float("inf")), [-1], keepdims=1)
        quantiles = op.Where(
            op.IsInf(quantiles, detect_negative=0),
            high,
            op.Where(op.IsInf(quantiles, detect_positive=0), low, quantiles),
        )
        return op.Where(op.IsInf(quantiles), 0.0, quantiles)
