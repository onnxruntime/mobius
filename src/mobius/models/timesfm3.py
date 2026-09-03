# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""TimesFM 3 multivariate time-series forecasting model.

Replicates Google Research's ``TimesFM3Torch`` as a five-stage ONNX pipeline:
raw-series preparation, patched feature construction, transformer inference,
CPM/RevIN postprocessing, and forecast stitching.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import BaseModelConfig
from mobius.components import Linear
from mobius.components._scan_utils import create_body_graph, rename_subgraph_values

_INT64_MAX = 9223372036854775807
_PER_DIM_SCALE = 1.442695041
_REVIN_TOLERANCE = 1e-6
_FLOAT32_EPS = 1.1920928955078125e-7
_TARGET_ROLE = 0
_PAST_ONLY_ROLE = 1
_PAST_FUTURE_ROLE = 2


def _field(value, name: str, default):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclasses.dataclass
class TimesFM3Config(BaseModelConfig):
    """Configuration parsed from a TimesFM 3 ``config.json``."""

    input_patch_len: int = 32
    output_patch_len: int = 64
    quantiles: tuple[float, ...] = tuple(i / 10 for i in range(1, 10))
    num_layers: int = 20
    model_dims: int = 1280
    transformer_hidden_dims: int = 1280
    num_heads: int = 16
    max_variates: int = 32
    rms_norm_eps: float = _FLOAT32_EPS
    use_variate_attention: bool = True
    use_rope_seq: bool = True
    use_rope_var: bool = False
    value_clip: float = 1e20
    use_iterative_cpm_revin: bool = True
    use_linear_detrending: bool = True
    linear_detrending_threshold: float = 0.5
    model_type: str | None = "timesfm3"

    def __post_init__(self) -> None:
        self.hidden_size = self.model_dims
        self.intermediate_size = self.transformer_hidden_dims
        self.num_hidden_layers = self.num_layers
        self.num_attention_heads = self.num_heads
        self.num_key_value_heads = self.num_heads
        self.head_dim = self.model_dims // self.num_heads

    def validate(self) -> None:
        if (
            not isinstance(self.input_patch_len, int)
            or isinstance(self.input_patch_len, bool)
            or self.input_patch_len <= 0
            or not isinstance(self.output_patch_len, int)
            or isinstance(self.output_patch_len, bool)
            or self.output_patch_len <= 0
        ):
            raise ValueError("input_patch_len and output_patch_len must be positive integers")
        if self.output_patch_len % self.input_patch_len:
            raise ValueError("output_patch_len must be a multiple of input_patch_len")
        if self.model_dims % self.num_heads:
            raise ValueError("model_dims must be divisible by num_heads")
        if self.head_dim % 2:
            raise ValueError("TimesFM 3 RoPE requires an even head_dim")
        if not self.quantiles:
            raise ValueError("quantiles must not be empty")
        if self.output_patch_len <= self.input_patch_len:
            raise ValueError("TimesFM 3 stitching requires output_patch_len > input_patch_len")

    @classmethod
    def from_transformers(cls, config) -> TimesFM3Config:
        residual = _field(config, "residual_block_config", {})
        stack = _field(config, "transformer_config", {})
        transformer = _field(stack, "transformer", {})
        model_dims = int(
            _field(transformer, "model_dims", _field(residual, "output_dims", 1280))
        )
        return cls(
            input_patch_len=int(_field(config, "input_patch_len", 32)),
            output_patch_len=int(_field(config, "output_patch_len", 64)),
            quantiles=tuple(float(q) for q in _field(config, "quantiles", (0.5,))),
            num_layers=int(_field(stack, "num_layers", 20)),
            model_dims=model_dims,
            transformer_hidden_dims=int(_field(transformer, "hidden_dims", model_dims)),
            num_heads=int(_field(transformer, "num_heads", 16)),
            max_variates=int(_field(transformer, "max_variates", 32)),
            rms_norm_eps=_FLOAT32_EPS,
            use_variate_attention=bool(_field(config, "use_variate_attention", True)),
            use_rope_seq=bool(_field(transformer, "use_rope_seq", True)),
            use_rope_var=bool(_field(transformer, "use_rope_var", False)),
            value_clip=float(_field(config, "value_clip", 1e20)),
            use_iterative_cpm_revin=bool(_field(config, "use_iterative_cpm_revin", True)),
            use_linear_detrending=bool(_field(config, "use_linear_detrending", True)),
            linear_detrending_threshold=float(
                _field(config, "linear_detrending_threshold", 0.5)
            ),
            model_type="timesfm3",
        )


def _safe_divisor(op: OpBuilder, value: ir.Value) -> ir.Value:
    return op.Where(
        op.Less(value, op.CastLike(op.Constant(value_float=_REVIN_TOLERANCE), value)),
        op.CastLike(op.Constant(value_float=1.0), value),
        value,
    )


def _update_running_stats(
    op: OpBuilder,
    count: ir.Value,
    mean: ir.Value,
    std: ir.Value,
    values: ir.Value,
    masks: ir.Value,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    valid = op.Not(masks)
    valid_f = op.Cast(valid, to=ir.DataType.FLOAT)
    increment_count = op.ReduceSum(valid_f, axes=[-1], keepdims=False)
    safe_increment_count = op.Max(increment_count, op.Constant(value_float=1.0))

    valid_values = op.Where(valid, values, op.CastLike(op.Constant(value_float=0.0), values))
    increment_sum = op.ReduceSum(valid_values, axes=[-1], keepdims=False)
    increment_mean = op.Where(
        op.Equal(increment_count, op.Constant(value_float=0.0)),
        op.Mul(increment_sum, op.Constant(value_float=0.0)),
        op.Div(increment_sum, safe_increment_count),
    )

    centered = op.Sub(values, op.Unsqueeze(increment_mean, axes=[-1]))
    centered_sq = op.Where(
        valid,
        op.Mul(centered, centered),
        op.CastLike(op.Constant(value_float=0.0), values),
    )
    increment_variance = op.Div(
        op.ReduceSum(centered_sq, axes=[-1], keepdims=False),
        safe_increment_count,
    )
    new_count = op.Add(count, increment_count)
    safe_new_count = op.Max(new_count, op.Constant(value_float=1.0))
    new_mean = op.Div(
        op.Add(op.Mul(count, mean), op.Mul(increment_count, increment_mean)),
        safe_new_count,
    )
    merged_variance = op.Div(
        op.Add(
            op.Add(
                op.Mul(count, op.Mul(std, std)), op.Mul(increment_count, increment_variance)
            ),
            op.Add(
                op.Mul(count, op.Mul(op.Sub(mean, new_mean), op.Sub(mean, new_mean))),
                op.Mul(
                    increment_count,
                    op.Mul(
                        op.Sub(increment_mean, new_mean),
                        op.Sub(increment_mean, new_mean),
                    ),
                ),
            ),
        ),
        safe_new_count,
    )
    return new_count, new_mean, op.Sqrt(op.Max(merged_variance, op.Constant(value_float=0.0)))


def _running_stats_body() -> ir.Graph:
    float_type = ir.TensorType(ir.DataType.FLOAT)
    bool_type = ir.TensorType(ir.DataType.BOOL)
    count = ir.Value(name="count", type=float_type)
    mean = ir.Value(name="mean", type=float_type)
    std = ir.Value(name="std", type=float_type)
    values = ir.Value(name="values", type=float_type)
    masks = ir.Value(name="masks", type=bool_type)
    graph, builder = create_body_graph(
        state_inputs=[count, mean, std],
        scan_inputs=[values, masks],
        name="timesfm3_running_stats",
    )
    new_count, new_mean, new_std = _update_running_stats(
        builder.op, count, mean, std, values, masks
    )
    # Scan carry and per-step outputs must be distinct graph values. Use a
    # data-dependent copy because no-op Identities are removed during optimization.
    nonnegative_count = builder.op.GreaterOrEqual(
        new_count, builder.op.Constant(value_float=0.0)
    )
    scan_count = builder.op.Where(
        nonnegative_count, new_count, builder.op.Constant(value_float=0.0)
    )
    scan_mean = builder.op.Where(
        nonnegative_count, new_mean, builder.op.Constant(value_float=0.0)
    )
    scan_std = builder.op.Where(
        nonnegative_count, new_std, builder.op.Constant(value_float=0.0)
    )
    graph.outputs.extend([new_count, new_mean, new_std, scan_count, scan_mean, scan_std])
    rename_subgraph_values(graph, "timesfm3_stats_")
    return graph


def _get_running_stats(
    op: OpBuilder, values: ir.Value, masks: ir.Value
) -> tuple[ir.Value, ir.Value, ir.Value]:
    # Scan patch-major inputs so the recurrence exactly matches upstream's
    # population-variance merge, including its numerical evaluation order.
    patch_values = op.Transpose(op.Cast(values, to=ir.DataType.FLOAT), perm=[2, 0, 1, 3])
    patch_masks = op.Transpose(masks, perm=[2, 0, 1, 3])
    bv_shape = op.Shape(values, start=0, end=2)
    zeros = op.ConstantOfShape(bv_shape, value=ir.tensor([0.0], dtype=ir.DataType.FLOAT))
    _, _, _, count, mean, std = op.Scan(
        zeros,
        zeros,
        zeros,
        patch_values,
        patch_masks,
        body=_running_stats_body(),
        num_scan_inputs=2,
        _outputs=6,
    )
    return (
        op.Transpose(count, perm=[1, 2, 0]),
        op.Transpose(mean, perm=[1, 2, 0]),
        op.Transpose(std, perm=[1, 2, 0]),
    )


def _nearest_observed_body() -> ir.Graph:
    """Scan body carrying the nearest valid index seen so far."""
    int_type = ir.TensorType(ir.DataType.INT64)
    bool_type = ir.TensorType(ir.DataType.BOOL)
    previous = ir.Value(name="previous", type=int_type)
    valid = ir.Value(name="valid", type=bool_type)
    index = ir.Value(name="index", type=int_type)
    graph, builder = create_body_graph(
        state_inputs=[previous],
        scan_inputs=[valid, index],
        name="timesfm3_nearest_observed",
    )
    current = builder.op.Where(valid, index, previous)
    # Scan carry and scan output must be distinct values.
    scanned = builder.op.Where(
        builder.op.GreaterOrEqual(current, builder.op.Constant(value_int=-1)),
        current,
        builder.op.Constant(value_int=-1),
    )
    graph.outputs.extend([current, scanned])
    rename_subgraph_values(graph, "timesfm3_nearest_")
    return graph


def _interpolate_missing(
    op: OpBuilder,
    values: ir.Value,
    observed: ir.Value,
) -> ir.Value:
    """Match ``numpy.interp`` independently over the last axis."""
    values_f32 = op.Cast(values, to=ir.DataType.FLOAT)
    length = op.Squeeze(op.Shape(values, start=2, end=3))
    indices = op.Range(
        op.Constant(value_int=0),
        length,
        op.Constant(value_int=1),
    )
    valid_t = op.Transpose(observed, perm=[2, 0, 1])
    leading_shape = op.Shape(values, start=0, end=2)
    previous_init = op.ConstantOfShape(
        leading_shape, value=ir.tensor([-1], dtype=ir.DataType.INT64)
    )
    _, previous_t = op.Scan(
        previous_init,
        valid_t,
        indices,
        body=_nearest_observed_body(),
        num_scan_inputs=2,
        _outputs=2,
    )

    reverse_indices = op.Range(
        op.Sub(length, op.Constant(value_int=1)),
        op.Constant(value_int=-1),
        op.Constant(value_int=-1),
    )
    next_init = op.ConstantOfShape(
        leading_shape, value=ir.tensor([0], dtype=ir.DataType.INT64)
    )
    next_init = op.Add(next_init, length)
    _, next_reversed_t = op.Scan(
        next_init,
        op.Gather(valid_t, reverse_indices, axis=0),
        reverse_indices,
        body=_nearest_observed_body(),
        num_scan_inputs=2,
        _outputs=2,
    )
    next_t = op.Gather(next_reversed_t, reverse_indices, axis=0)

    previous = op.Transpose(previous_t, perm=[1, 2, 0])
    following = op.Transpose(next_t, perm=[1, 2, 0])
    zero = op.Constant(value_int=0)
    last = op.Sub(length, op.Constant(value_int=1))
    previous_values = op.GatherElements(
        values_f32, op.Min(op.Max(previous, zero), last), axis=2
    )
    following_values = op.GatherElements(
        values_f32, op.Min(op.Max(following, zero), last), axis=2
    )

    has_previous = op.GreaterOrEqual(previous, zero)
    has_following = op.Less(following, length)
    both = op.And(has_previous, has_following)
    span = op.Cast(op.Sub(following, previous), to=ir.DataType.FLOAT)
    safe_span = op.Where(
        op.Equal(span, op.CastLike(op.Constant(value_float=0.0), span)),
        op.CastLike(op.Constant(value_float=1.0), span),
        span,
    )
    position = op.Cast(
        op.Sub(op.Unsqueeze(indices, axes=[0, 1]), previous),
        to=ir.DataType.FLOAT,
    )
    interpolated = op.Add(
        previous_values,
        op.Mul(
            op.Div(position, safe_span),
            op.Sub(following_values, previous_values),
        ),
    )
    filled = op.Where(
        both,
        interpolated,
        op.Where(
            has_previous,
            previous_values,
            op.Where(
                has_following,
                following_values,
                op.Constant(value_float=0.0),
            ),
        ),
    )
    return op.CastLike(op.Where(observed, values_f32, filled), values)


def _roll_patches(
    op: OpBuilder, values: ir.Value, rolls: int, patch_len: int
) -> tuple[ir.Value, ir.Value]:
    shifted = values
    outputs = []
    for _ in range(rolls):
        shifted = op.Concat(
            op.Slice(shifted, starts=[1], ends=[_INT64_MAX], axes=[2]),
            op.Slice(shifted, starts=[0], ends=[1], axes=[2]),
            axis=2,
        )
        outputs.append(shifted)
    rolled = op.Concat(*outputs, axis=-1)

    num_patches = op.Shape(values, start=2, end=3)
    patch_index = op.Range(
        op.Constant(value_int=0), op.Squeeze(num_patches), op.Constant(value_int=1)
    )
    output_index = op.Constant(value_ints=list(range(rolls * patch_len)))
    source_patch = op.Add(
        op.Add(
            op.Unsqueeze(patch_index, axes=[1]),
            op.Constant(value_int=1),
        ),
        op.Div(
            op.Unsqueeze(output_index, axes=[0]),
            op.Constant(value_int=patch_len),
        ),
    )
    wrap_mask = op.GreaterOrEqual(source_patch, num_patches)
    return rolled, op.Unsqueeze(wrap_mask, axes=[0, 1])


class _ResidualBlock(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int):
        super().__init__()
        self.hidden_layer = Linear(input_dims, hidden_dims, bias=False)
        self.output_layer = Linear(hidden_dims, output_dims, bias=False)
        self.residual_layer = Linear(input_dims, output_dims, bias=False)

    def forward(self, op: OpBuilder, values: ir.Value) -> ir.Value:
        hidden = op.Relu(self.hidden_layer(op, values))
        return op.Add(self.output_layer(op, hidden), self.residual_layer(op, values))


class _PerDimScale(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.per_dim_scale = nn.Parameter([head_dim])
        self._factor = _PER_DIM_SCALE / math.sqrt(head_dim)

    def forward(self, op: OpBuilder, values: ir.Value) -> ir.Value:
        scale = op.Mul(op.Softplus(self.per_dim_scale), self._factor)
        return op.Mul(values, op.CastLike(scale, values))


class _TimesFMRMSNorm(nn.Module):
    def __init__(self, dimensions: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter([dimensions])
        self._eps = eps

    def forward(self, op: OpBuilder, values: ir.Value) -> ir.Value:
        values_f32 = op.Cast(values, to=ir.DataType.FLOAT)
        variance = op.ReduceMean(
            op.Mul(values_f32, values_f32),
            axes=[-1],
            keepdims=True,
        )
        normalized = op.Mul(
            values_f32,
            op.Reciprocal(
                op.Sqrt(
                    op.Add(
                        variance,
                        op.CastLike(op.Constant(value_float=self._eps), variance),
                    )
                )
            ),
        )
        output = op.Mul(normalized, op.CastLike(self.weight, normalized))
        return op.CastLike(output, values)


class _TimesFMAttention(nn.Module):
    def __init__(self, config: TimesFM3Config, *, causal: bool, use_rope: bool):
        super().__init__()
        self.query_proj = Linear(config.model_dims, config.model_dims, bias=False)
        self.key_proj = Linear(config.model_dims, config.model_dims, bias=False)
        self.value_proj = Linear(config.model_dims, config.model_dims, bias=False)
        self.out_proj = Linear(config.model_dims, config.model_dims, bias=False)
        self.query_ln = _TimesFMRMSNorm(config.head_dim, config.rms_norm_eps)
        self.key_ln = _TimesFMRMSNorm(config.head_dim, config.rms_norm_eps)
        self.per_dim_scale = _PerDimScale(config.head_dim)
        self._num_heads = config.num_heads
        self._head_dim = config.head_dim
        self._model_dims = config.model_dims
        self._causal = causal
        self._use_rope = use_rope
        self._mask_value = -65504.0 if config.dtype == ir.DataType.FLOAT16 else -1e9

    def _rope(self, op: OpBuilder, values: ir.Value) -> ir.Value:
        half_dim = self._head_dim // 2
        timescale = [10000.0 ** (2.0 * i / self._head_dim) for i in range(half_dim)]
        seq_len = op.Squeeze(op.Shape(values, start=1, end=2))
        positions = op.Cast(
            op.Range(
                op.Constant(value_int=0),
                seq_len,
                op.Constant(value_int=1),
            ),
            to=ir.DataType.FLOAT,
        )
        angles = op.Div(
            op.Unsqueeze(positions, axes=[0, 2, 3]),
            op.Constant(value_floats=timescale),
        )
        sin = op.Sin(angles)
        cos = op.Cos(angles)
        first = op.Slice(values, starts=[0], ends=[half_dim], axes=[-1])
        second = op.Slice(values, starts=[half_dim], ends=[self._head_dim], axes=[-1])
        sin = op.CastLike(sin, values)
        cos = op.CastLike(cos, values)
        return op.Concat(
            op.Sub(op.Mul(first, cos), op.Mul(second, sin)),
            op.Add(op.Mul(second, cos), op.Mul(first, sin)),
            axis=-1,
        )

    def forward(self, op: OpBuilder, values: ir.Value, patch_mask: ir.Value) -> ir.Value:
        shape = op.Shape(values)
        batch_like = op.Slice(shape, starts=[0], ends=[1])
        seq_len = op.Slice(shape, starts=[1], ends=[2])
        qkv_shape = op.Concat(
            batch_like,
            seq_len,
            op.Constant(value_ints=[self._num_heads, self._head_dim]),
            axis=0,
        )
        query = op.Reshape(self.query_proj(op, values), qkv_shape)
        key = op.Reshape(self.key_proj(op, values), qkv_shape)
        value = op.Reshape(self.value_proj(op, values), qkv_shape)

        if self._use_rope:
            query = self._rope(op, query)
            key = self._rope(op, key)
        query = self.per_dim_scale(op, self.query_ln(op, query))
        key = self.key_ln(op, key)

        query = op.Transpose(query, perm=[0, 2, 1, 3])
        key = op.Transpose(key, perm=[0, 2, 1, 3])
        value = op.Transpose(value, perm=[0, 2, 1, 3])
        scores = op.Mul(
            op.MatMul(query, op.Transpose(key, perm=[0, 1, 3, 2])),
            math.sqrt(self._head_dim),
        )

        key_valid = op.Unsqueeze(op.Not(patch_mask), axes=[1, 2])
        if self._causal:
            length = op.Squeeze(seq_len)
            indices = op.Range(op.Constant(value_int=0), length, op.Constant(value_int=1))
            causal = op.GreaterOrEqual(
                op.Unsqueeze(indices, axes=[1]), op.Unsqueeze(indices, axes=[0])
            )
            allowed = op.And(key_valid, op.Unsqueeze(causal, axes=[0, 1]))
        else:
            allowed = key_valid

        masked_scores = op.Where(
            allowed,
            scores,
            op.CastLike(op.Constant(value_float=self._mask_value), scores),
        )
        probabilities = op.Softmax(masked_scores, axis=-1)
        row_valid = op.Cast(
            op.ReduceMax(op.Cast(allowed, to=ir.DataType.INT64), axes=[-1], keepdims=True),
            to=ir.DataType.BOOL,
        )
        attended = op.Where(
            row_valid,
            op.MatMul(probabilities, value),
            op.CastLike(op.Constant(value_float=0.0), value),
        )
        attended = op.Transpose(attended, perm=[0, 2, 1, 3])
        output_shape = op.Concat(
            batch_like,
            seq_len,
            op.Constant(value_ints=[self._model_dims]),
            axis=0,
        )
        return self.out_proj(op, op.Reshape(attended, output_shape))


class _MixingTransformer(nn.Module):
    def __init__(self, config: TimesFM3Config):
        super().__init__()
        self.pre_seq_attn_ln = _TimesFMRMSNorm(config.model_dims, config.rms_norm_eps)
        self.post_seq_attn_ln = _TimesFMRMSNorm(config.model_dims, config.rms_norm_eps)
        self.seq_attn = _TimesFMAttention(config, causal=True, use_rope=config.use_rope_seq)
        if config.use_variate_attention:
            self.pre_var_attn_ln = _TimesFMRMSNorm(config.model_dims, config.rms_norm_eps)
            self.post_var_attn_ln = _TimesFMRMSNorm(config.model_dims, config.rms_norm_eps)
            self.var_attn = _TimesFMAttention(
                config, causal=False, use_rope=config.use_rope_var
            )
        else:
            self.pre_var_attn_ln = None
            self.post_var_attn_ln = None
            self.var_attn = None
        self.pre_ff_ln = _TimesFMRMSNorm(config.model_dims, config.rms_norm_eps)
        self.post_ff_ln = _TimesFMRMSNorm(config.model_dims, config.rms_norm_eps)
        self.ff0 = Linear(config.model_dims, config.transformer_hidden_dims, bias=False)
        self.ff1 = Linear(config.transformer_hidden_dims, config.model_dims, bias=False)
        self._model_dims = config.model_dims

    def forward(self, op: OpBuilder, values: ir.Value, patch_mask: ir.Value) -> ir.Value:
        shape = op.Shape(values)
        batch = op.Slice(shape, starts=[0], ends=[1])
        variates = op.Slice(shape, starts=[1], ends=[2])
        patches = op.Slice(shape, starts=[2], ends=[3])

        seq_shape = op.Concat(
            op.Mul(batch, variates),
            patches,
            op.Constant(value_ints=[self._model_dims]),
            axis=0,
        )
        seq_input = op.Reshape(self.pre_seq_attn_ln(op, values), seq_shape)
        seq_mask = op.Reshape(patch_mask, op.Concat(op.Mul(batch, variates), patches, axis=0))
        seq_output = self.seq_attn(op, seq_input, seq_mask)
        seq_output = op.Reshape(
            seq_output,
            op.Concat(
                batch,
                variates,
                patches,
                op.Constant(value_ints=[self._model_dims]),
                axis=0,
            ),
        )
        hidden = op.Add(self.post_seq_attn_ln(op, seq_output), values)

        if self.var_attn is not None:
            var_input = op.Transpose(self.pre_var_attn_ln(op, hidden), perm=[0, 2, 1, 3])
            var_shape = op.Concat(
                op.Mul(batch, patches),
                variates,
                op.Constant(value_ints=[self._model_dims]),
                axis=0,
            )
            var_input = op.Reshape(var_input, var_shape)
            var_mask = op.Reshape(
                op.Transpose(patch_mask, perm=[0, 2, 1]),
                op.Concat(op.Mul(batch, patches), variates, axis=0),
            )
            var_output = self.var_attn(op, var_input, var_mask)
            var_output = op.Transpose(
                op.Reshape(
                    var_output,
                    op.Concat(
                        batch,
                        patches,
                        variates,
                        op.Constant(value_ints=[self._model_dims]),
                        axis=0,
                    ),
                ),
                perm=[0, 2, 1, 3],
            )
            hidden = op.Add(self.post_var_attn_ln(op, var_output), hidden)

        ff = self.ff0(op, self.pre_ff_ln(op, hidden))
        ff = self.ff1(op, op.Relu(ff))
        return op.Add(self.post_ff_ln(op, ff), hidden)


class _StackedMixingTransformer(nn.Module):
    def __init__(self, config: TimesFM3Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [_MixingTransformer(config) for _ in range(config.num_layers)]
        )

    def forward(self, op: OpBuilder, values: ir.Value, patch_mask: ir.Value) -> ir.Value:
        for layer in self.layers:
            values = layer(op, values, patch_mask)
        return values


def _cpm_refinement_body(rolls: int, patch_len: int, value_clip: float) -> ir.Graph:
    float_type = ir.TensorType(ir.DataType.FLOAT)
    int_type = ir.TensorType(ir.DataType.INT64)
    bool_type = ir.TensorType(ir.DataType.BOOL)
    count = ir.Value(name="count", type=float_type)
    mean = ir.Value(name="mean", type=float_type)
    std = ir.Value(name="std", type=float_type)
    anchor = ir.Value(name="anchor", type=float_type)
    offset = ir.Value(name="offset", type=int_type)
    actual_count = ir.Value(name="actual_count", type=float_type)
    actual_mean = ir.Value(name="actual_mean", type=float_type)
    actual_std = ir.Value(name="actual_std", type=float_type)
    current_median = ir.Value(name="current_median", type=float_type)
    is_cpm = ir.Value(name="is_cpm", type=bool_type)
    graph, builder = create_body_graph(
        state_inputs=[count, mean, std, anchor, offset],
        scan_inputs=[actual_count, actual_mean, actual_std, current_median, is_cpm],
        name="timesfm3_cpm_refinement",
    )
    op = builder.op
    selector = op.Cast(
        op.Equal(
            op.Unsqueeze(offset, axes=[1]),
            op.Constant(value_ints=list(range(rolls))),
        ),
        to=ir.DataType.FLOAT,
    )
    predicted = op.ReduceSum(
        op.Mul(anchor, op.Unsqueeze(selector, axes=[1, 3])),
        axes=[2],
        keepdims=False,
    )
    prediction_mask = op.ConstantOfShape(
        op.Shape(predicted),
        value=ir.tensor([False], dtype=ir.DataType.BOOL),
    )
    candidate_count, candidate_mean, candidate_std = _update_running_stats(
        op, count, mean, std, predicted, prediction_mask
    )
    cpm_bv = op.Unsqueeze(is_cpm, axes=[1])
    output_count = op.Where(cpm_bv, candidate_count, actual_count)
    output_mean = op.Where(cpm_bv, candidate_mean, actual_mean)
    output_std = op.Where(cpm_bv, candidate_std, actual_std)
    new_offset = op.Where(
        is_cpm,
        op.Mod(op.Add(offset, op.Constant(value_int=1)), op.Constant(value_int=rolls)),
        op.Mul(offset, op.Constant(value_int=0)),
    )
    new_anchor = op.Clip(
        op.Add(
            op.Mul(
                current_median,
                op.Unsqueeze(output_std, axes=[-1, -2]),
            ),
            op.Unsqueeze(output_mean, axes=[-1, -2]),
        ),
        -value_clip,
        value_clip,
    )
    updated_anchor = op.Where(
        op.Unsqueeze(op.Equal(new_offset, op.Constant(value_int=0)), axes=[1, 2, 3]),
        new_anchor,
        anchor,
    )
    nonnegative_count = op.GreaterOrEqual(actual_count, op.Constant(value_float=0.0))
    scan_mean = op.Where(nonnegative_count, output_mean, op.Constant(value_float=0.0))
    scan_std = op.Where(nonnegative_count, output_std, op.Constant(value_float=0.0))
    graph.outputs.extend(
        [
            output_count,
            output_mean,
            output_std,
            updated_anchor,
            new_offset,
            scan_mean,
            scan_std,
        ]
    )
    rename_subgraph_values(graph, "timesfm3_cpm_")
    return graph


class TimesFM3Model(nn.Module):
    """TimesFM 3 padded-batch multivariate forecasting network."""

    default_task: str = "time-series-forecasting"
    category: str = "Time Series"
    config_class = TimesFM3Config

    def __init__(self, config: TimesFM3Config):
        super().__init__()
        config.validate()
        self.config = config
        feature_dims = 2 * (config.input_patch_len + config.output_patch_len)
        self.pre_transformer_resblock = _ResidualBlock(
            feature_dims, config.model_dims, config.model_dims
        )
        self.transformer_stack = _StackedMixingTransformer(config)
        self.output_head = Linear(
            config.model_dims,
            config.output_patch_len * len(config.quantiles),
            bias=True,
        )
        self._rolls = config.output_patch_len // config.input_patch_len
        self._value_clip = (
            min(config.value_clip, 65504.0)
            if config.dtype == ir.DataType.FLOAT16
            else config.value_clip
        )

    def prepare_raw_series(
        self,
        op: OpBuilder,
        context_values: ir.Value,
        context_observed: ir.Value,
        future_values: ir.Value,
        future_observed: ir.Value,
        context_lengths: ir.Value,
        horizon_lengths: ir.Value,
        variate_roles: ir.Value,
    ) -> tuple[ir.Value, ...]:
        """Convert right-aligned raw series to the upstream patched contract.

        ``context_values`` is right-aligned in ``[B, V, C]`` and
        ``future_values`` is left-aligned in ``[B, V, H]``. The corresponding
        observed tensors use True for supplied, finite observations. Roles are
        0=target, 1=past-only covariate, and 2=past-future covariate.
        """
        config = self.config
        patch_len = config.input_patch_len
        rolls = self._rolls
        extract_len = min(2 * patch_len, config.output_patch_len)
        overlap = extract_len - patch_len

        batch = op.Shape(context_values, start=0, end=1)
        variates = op.Shape(context_values, start=1, end=2)
        context_width = op.Squeeze(op.Shape(context_values, start=2, end=3))
        future_width = op.Squeeze(op.Shape(future_values, start=2, end=3))
        max_context = op.ReduceMax(context_lengths, keepdims=False)
        padded_context = op.Mul(
            op.Div(
                op.Add(max_context, op.Constant(value_int=patch_len - 1)),
                op.Constant(value_int=patch_len),
            ),
            op.Constant(value_int=patch_len),
        )
        padded_context = op.Max(padded_context, op.Constant(value_int=patch_len))
        context_patches = op.Div(padded_context, op.Constant(value_int=patch_len))

        forecast_numer = op.Max(
            op.Sub(horizon_lengths, op.Constant(value_int=overlap)),
            op.Constant(value_int=0),
        )
        forecast_patches = op.Div(
            op.Add(forecast_numer, op.Constant(value_int=patch_len - 1)),
            op.Constant(value_int=patch_len),
        )
        forecast_patches = op.Max(forecast_patches, op.Constant(value_int=1))
        max_forecast_patches = op.ReduceMax(forecast_patches, keepdims=False)
        horizon_patches = op.Add(max_forecast_patches, op.Constant(value_int=rolls - 1))
        padded_horizon = op.Mul(horizon_patches, op.Constant(value_int=patch_len))

        is_target = op.Equal(variate_roles, op.Constant(value_int=_TARGET_ROLE))
        is_past_only = op.Equal(variate_roles, op.Constant(value_int=_PAST_ONLY_ROLE))
        is_past_future = op.Equal(variate_roles, op.Constant(value_int=_PAST_FUTURE_ROLE))
        valid_role = op.Or(op.Or(is_target, is_past_only), is_past_future)

        # Right-align every row in the common dynamic context patch grid.
        context_position = op.Range(
            op.Constant(value_int=0),
            padded_context,
            op.Constant(value_int=1),
        )
        context_source = op.Add(context_position, op.Sub(context_width, padded_context))
        context_source = op.Min(
            op.Max(context_source, op.Constant(value_int=0)),
            op.Sub(context_width, op.Constant(value_int=1)),
        )
        context_shape = op.Concat(
            batch, variates, op.Unsqueeze(padded_context, axes=[0]), axis=0
        )
        context_indices = op.Expand(op.Unsqueeze(context_source, axes=[0, 1]), context_shape)
        aligned_context = op.GatherElements(context_values, context_indices, axis=2)
        aligned_context_observed = op.GatherElements(context_observed, context_indices, axis=2)
        context_active = op.GreaterOrEqual(
            op.Unsqueeze(context_position, axes=[0, 1]),
            op.Sub(
                padded_context,
                op.Unsqueeze(context_lengths, axes=[1, 2]),
            ),
        )
        context_active = op.And(context_active, op.Unsqueeze(valid_role, axes=[2]))

        # Future observations are left-aligned and only past-future variates
        # participate. Extra common horizon patches remain masked.
        horizon_position = op.Range(
            op.Constant(value_int=0),
            padded_horizon,
            op.Constant(value_int=1),
        )
        future_source = op.Min(
            horizon_position,
            op.Sub(future_width, op.Constant(value_int=1)),
        )
        horizon_shape = op.Concat(
            batch, variates, op.Unsqueeze(padded_horizon, axes=[0]), axis=0
        )
        future_indices = op.Expand(op.Unsqueeze(future_source, axes=[0, 1]), horizon_shape)
        aligned_future = op.GatherElements(future_values, future_indices, axis=2)
        aligned_future_observed = op.GatherElements(future_observed, future_indices, axis=2)
        future_active = op.And(
            op.Less(
                op.Unsqueeze(horizon_position, axes=[0, 1]),
                op.Unsqueeze(horizon_lengths, axes=[1, 2]),
            ),
            op.Unsqueeze(is_past_future, axes=[2]),
        )

        raw_values = op.Concat(aligned_context, aligned_future, axis=2)
        finite = op.Not(op.Or(op.IsNaN(raw_values), op.IsInf(raw_values)))
        raw_values = op.Where(
            finite,
            raw_values,
            op.CastLike(op.Constant(value_float=0.0), raw_values),
        )
        supplied = op.Concat(
            op.And(context_active, aligned_context_observed),
            op.And(future_active, aligned_future_observed),
            axis=2,
        )
        supplied = op.And(supplied, finite)
        interpolated = _interpolate_missing(op, raw_values, supplied)

        context = op.Slice(
            interpolated,
            starts=op.Constant(value_ints=[0]),
            ends=op.Unsqueeze(padded_context, axes=[0]),
            axes=op.Constant(value_ints=[2]),
        )
        future = op.Slice(
            interpolated,
            starts=op.Unsqueeze(padded_context, axes=[0]),
            ends=op.Unsqueeze(op.Add(padded_context, padded_horizon), axes=[0]),
            axes=op.Constant(value_ints=[2]),
        )

        # Fit y = m*t + c using each row's unpadded context length. The
        # right-aligned time grid is -(length-1)..0 for every row.
        context_f32 = op.Cast(context, to=ir.DataType.FLOAT)
        context_valid = context_active
        valid_f32 = op.Cast(context_valid, to=ir.DataType.FLOAT)
        length_f32 = op.Cast(op.Unsqueeze(context_lengths, axes=[1, 2]), to=ir.DataType.FLOAT)
        time = op.Sub(
            op.Cast(
                op.Unsqueeze(context_position, axes=[0, 1]),
                to=ir.DataType.FLOAT,
            ),
            op.Cast(
                op.Sub(padded_context, op.Constant(value_int=1)),
                to=ir.DataType.FLOAT,
            ),
        )
        normalized_time = op.Div(time, op.Max(length_f32, op.Constant(value_float=1.0)))
        zeros = op.ConstantOfShape(
            op.Shape(context_f32), value=ir.tensor([0.0], dtype=ir.DataType.FLOAT)
        )
        valid_values = op.Where(context_valid, context_f32, zeros)
        valid_time = op.Where(context_valid, normalized_time, zeros)
        count = op.ReduceSum(valid_f32, axes=[-1], keepdims=True)
        safe_count = op.Max(count, op.Constant(value_float=1.0))
        sum_time = op.ReduceSum(valid_time, axes=[-1], keepdims=True)
        sum_time2 = op.ReduceSum(
            op.Where(context_valid, op.Mul(normalized_time, normalized_time), zeros),
            axes=[-1],
            keepdims=True,
        )
        sum_values = op.ReduceSum(valid_values, axes=[-1], keepdims=True)
        sum_time_values = op.ReduceSum(
            op.Where(
                context_valid,
                op.Mul(normalized_time, context_f32),
                zeros,
            ),
            axes=[-1],
            keepdims=True,
        )
        determinant = op.Sub(op.Mul(count, sum_time2), op.Mul(sum_time, sum_time))
        determinant_zero = op.Equal(determinant, op.Constant(value_float=0.0))
        safe_determinant = op.Where(
            determinant_zero, op.Constant(value_float=1.0), determinant
        )
        trend_slope = op.Where(
            determinant_zero,
            op.Constant(value_float=0.0),
            op.Div(
                op.Sub(
                    op.Mul(count, sum_time_values),
                    op.Mul(sum_time, sum_values),
                ),
                safe_determinant,
            ),
        )
        trend_intercept = op.Where(
            determinant_zero,
            op.Where(
                op.Greater(count, op.Constant(value_float=0.0)),
                op.Div(sum_values, safe_count),
                op.Constant(value_float=0.0),
            ),
            op.Div(
                op.Sub(sum_values, op.Mul(trend_slope, sum_time)),
                safe_count,
            ),
        )
        detrended_context = op.Sub(
            context_f32,
            op.Add(op.Mul(trend_slope, normalized_time), trend_intercept),
        )
        mean = op.Div(sum_values, safe_count)
        sum_values2 = op.ReduceSum(
            op.Where(context_valid, op.Mul(context_f32, context_f32), zeros),
            axes=[-1],
            keepdims=True,
        )
        original_variance = op.Max(
            op.Sub(op.Div(sum_values2, safe_count), op.Mul(mean, mean)),
            op.Constant(value_float=0.0),
        )
        detrended_values = op.Where(context_valid, detrended_context, zeros)
        detrended_sum = op.ReduceSum(detrended_values, axes=[-1], keepdims=True)
        detrended_mean = op.Div(detrended_sum, safe_count)
        detrended_sum2 = op.ReduceSum(
            op.Mul(detrended_values, detrended_values),
            axes=[-1],
            keepdims=True,
        )
        detrended_variance = op.Max(
            op.Sub(
                op.Div(detrended_sum2, safe_count),
                op.Mul(detrended_mean, detrended_mean),
            ),
            op.Constant(value_float=0.0),
        )
        apply_detrend = op.Less(
            op.Sqrt(detrended_variance),
            op.Mul(
                op.Constant(value_float=config.linear_detrending_threshold),
                op.Sqrt(original_variance),
            ),
        )
        if not config.use_linear_detrending:
            apply_detrend = op.And(
                apply_detrend,
                op.ConstantOfShape(
                    op.Shape(apply_detrend),
                    value=ir.tensor([False], dtype=ir.DataType.BOOL),
                ),
            )
        context = op.CastLike(
            op.Where(apply_detrend, detrended_context, context_f32),
            context_values,
        )

        horizon_step = op.Cast(
            op.Add(horizon_position, op.Constant(value_int=1)),
            to=ir.DataType.FLOAT,
        )
        future_trend = op.Add(
            op.Mul(
                trend_slope,
                op.Div(
                    op.Unsqueeze(horizon_step, axes=[0, 1]),
                    op.Max(length_f32, op.Constant(value_float=1.0)),
                ),
            ),
            trend_intercept,
        )
        future_f32 = op.Cast(future, to=ir.DataType.FLOAT)
        future = op.CastLike(
            op.Where(
                apply_detrend,
                op.Sub(future_f32, future_trend),
                future_f32,
            ),
            future_values,
        )

        context_masks = op.Not(context_active)
        future_masks = op.Not(future_active)
        all_values = op.Concat(
            op.Where(
                context_masks,
                op.CastLike(op.Constant(value_float=0.0), context),
                context,
            ),
            op.Where(
                future_masks,
                op.CastLike(op.Constant(value_float=0.0), future),
                future,
            ),
            axis=2,
        )
        all_masks = op.Concat(context_masks, future_masks, axis=2)
        values = op.Reshape(
            all_values,
            op.Concat(
                batch,
                variates,
                op.Constant(value_ints=[-1, patch_len]),
                axis=0,
            ),
        )
        masks = op.Reshape(
            all_masks,
            op.Concat(
                batch,
                variates,
                op.Constant(value_ints=[-1, patch_len]),
                axis=0,
            ),
        )
        total_patches = op.Squeeze(op.Shape(values, start=2, end=3))
        patch_index = op.Range(
            op.Constant(value_int=0),
            total_patches,
            op.Constant(value_int=1),
        )
        patch_shape = op.Concat(batch, variates, op.Unsqueeze(total_patches, axes=[0]), axis=0)
        patch_is_target = op.Expand(
            op.Unsqueeze(op.Or(is_target, is_past_only), axes=[2]),
            patch_shape,
        )
        patch_cpm_mask = op.Expand(
            op.Unsqueeze(op.GreaterOrEqual(patch_index, context_patches), axes=[0]),
            op.Concat(batch, op.Unsqueeze(total_patches, axes=[0]), axis=0),
        )

        # Positivity policy is based on the original finite observations,
        # before interpolation or detrending.
        original_finite = op.Not(op.Or(op.IsNaN(aligned_context), op.IsInf(aligned_context)))
        original_valid = op.And(
            context_active,
            op.And(aligned_context_observed, original_finite),
        )
        original_count = op.ReduceSum(
            op.Cast(original_valid, to=ir.DataType.INT64),
            axes=[-1],
            keepdims=False,
        )
        original_nonnegative = op.ReduceMin(
            op.Cast(
                op.Or(
                    op.Not(original_valid),
                    op.GreaterOrEqual(
                        aligned_context,
                        op.CastLike(op.Constant(value_float=0.0), aligned_context),
                    ),
                ),
                to=ir.DataType.INT64,
            ),
            axes=[-1],
            keepdims=False,
        )
        nonnegative = op.And(
            op.Greater(original_count, op.Constant(value_int=0)),
            op.Cast(original_nonnegative, to=ir.DataType.BOOL),
        )
        nonnegative = op.And(nonnegative, is_target)

        return (
            values,
            masks,
            patch_is_target,
            patch_cpm_mask,
            op.Squeeze(trend_slope, axes=[-1]),
            op.Squeeze(trend_intercept, axes=[-1]),
            op.Squeeze(apply_detrend, axes=[-1]),
            is_target,
            nonnegative,
            context_lengths,
            horizon_lengths,
            context_patches,
            forecast_patches,
        )

    def _refine_cpm_stats(
        self,
        op: OpBuilder,
        raw_logits: ir.Value,
        running_count: ir.Value,
        running_mean: ir.Value,
        running_std: ir.Value,
        patch_cpm_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        config = self.config
        structured = op.Reshape(
            raw_logits,
            [0, 0, 0, self._rolls, config.input_patch_len, len(config.quantiles)],
        )
        median = op.Gather(structured, len(config.quantiles) // 2, axis=-1)
        median_t = op.Transpose(median, perm=[2, 0, 1, 3, 4])
        count_t = op.Transpose(running_count, perm=[2, 0, 1])
        mean_t = op.Transpose(running_mean, perm=[2, 0, 1])
        std_t = op.Transpose(running_std, perm=[2, 0, 1])
        cpm_t = op.Transpose(patch_cpm_mask, perm=[1, 0])

        bv_shape = op.Shape(running_count, start=0, end=2)
        zeros = op.ConstantOfShape(bv_shape, value=ir.tensor([0.0], dtype=ir.DataType.FLOAT))
        anchor_shape = op.Concat(
            bv_shape,
            op.Constant(value_ints=[self._rolls, config.input_patch_len]),
            axis=0,
        )
        anchor = op.ConstantOfShape(
            anchor_shape, value=ir.tensor([0.0], dtype=ir.DataType.FLOAT)
        )
        batch_shape = op.Shape(running_count, start=0, end=1)
        offset = op.ConstantOfShape(batch_shape, value=ir.tensor([0], dtype=ir.DataType.INT64))
        _, _, _, _, _, refined_mean, refined_std = op.Scan(
            zeros,
            zeros,
            zeros,
            anchor,
            offset,
            count_t,
            mean_t,
            std_t,
            median_t,
            cpm_t,
            body=_cpm_refinement_body(self._rolls, config.input_patch_len, self._value_clip),
            num_scan_inputs=5,
            _outputs=7,
        )
        return (
            op.Transpose(refined_mean, perm=[1, 2, 0]),
            op.Transpose(refined_std, perm=[1, 2, 0]),
        )

    def preprocess(
        self,
        op: OpBuilder,
        values: ir.Value,
        masks: ir.Value,
        patch_is_target: ir.Value,
        patch_cpm_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]:
        """Prepare patched inputs and RevIN statistics for the transformer."""
        config = self.config
        values = op.Where(
            op.IsNaN(values),
            op.CastLike(op.Constant(value_float=0.0), values),
            values,
        )
        values = op.Clip(values, -self._value_clip, self._value_clip)
        running_count, running_mean, running_std = _get_running_stats(op, values, masks)

        cpm_target = op.And(
            op.Unsqueeze(patch_cpm_mask, axes=[1, 3]),
            op.Unsqueeze(patch_is_target, axes=[3]),
        )
        effective_masks = op.Or(masks, cpm_target)
        model_mean = op.CastLike(running_mean, values)
        divisor = op.CastLike(_safe_divisor(op, running_std), values)
        current = op.Div(
            op.Sub(values, op.Unsqueeze(model_mean, axes=[-1])),
            op.Unsqueeze(divisor, axes=[-1]),
        )
        current = op.Where(
            effective_masks,
            op.CastLike(op.Constant(value_float=0.0), current),
            current,
        )

        future, wrap_mask = _roll_patches(op, values, self._rolls, config.input_patch_len)
        future = op.Div(
            op.Sub(future, op.Unsqueeze(model_mean, axes=[-1])),
            op.Unsqueeze(divisor, axes=[-1]),
        )
        rolled_masks, _ = _roll_patches(
            op,
            op.Cast(effective_masks, to=ir.DataType.FLOAT),
            self._rolls,
            config.input_patch_len,
        )
        future_masks = op.Or(
            op.Or(
                op.Cast(rolled_masks, to=ir.DataType.BOOL), op.Unsqueeze(patch_is_target, [3])
            ),
            wrap_mask,
        )
        future = op.Where(
            future_masks,
            op.CastLike(op.Constant(value_float=0.0), future),
            future,
        )

        values_cat = op.Concat(current, future, axis=-1)
        masks_cat = op.Concat(effective_masks, future_masks, axis=-1)
        residual_input = op.Concat(values_cat, op.CastLike(masks_cat, values_cat), axis=-1)
        patch_mask = op.Cast(
            op.ReduceMin(op.Cast(masks_cat, to=ir.DataType.INT64), axes=[-1], keepdims=False),
            to=ir.DataType.BOOL,
        )

        prefix_count = op.CumSum(
            op.Cast(patch_mask, to=ir.DataType.INT64), op.Constant(value_int=2)
        )
        num_patches = op.Squeeze(op.Shape(patch_mask, start=2, end=3))
        patch_ordinals = op.Add(
            op.Range(op.Constant(value_int=0), num_patches, op.Constant(value_int=1)),
            op.Constant(value_int=1),
        )
        effective_patch_mask = op.Equal(
            prefix_count, op.Unsqueeze(patch_ordinals, axes=[0, 1])
        )
        return (
            residual_input,
            effective_patch_mask,
            running_count,
            running_mean,
            running_std,
        )

    def forecast(
        self,
        op: OpBuilder,
        model_inputs: ir.Value,
        patch_mask: ir.Value,
    ) -> ir.Value:
        """Run every learned layer in the capture-friendly model component."""
        hidden = self.pre_transformer_resblock(op, model_inputs)
        hidden = self.transformer_stack(op, hidden, patch_mask)
        return self.output_head(op, hidden)

    def postprocess(
        self,
        op: OpBuilder,
        raw_logits: ir.Value,
        running_count: ir.Value,
        running_mean: ir.Value,
        running_std: ir.Value,
        patch_cpm_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """Apply CPM refinement, reverse RevIN, clipping, and output shaping."""
        config = self.config

        output_mean, output_std = running_mean, running_std
        if config.use_iterative_cpm_revin:
            refined_mean, refined_std = self._refine_cpm_stats(
                op,
                op.Cast(raw_logits, to=ir.DataType.FLOAT),
                running_count,
                running_mean,
                running_std,
                patch_cpm_mask,
            )
            cpm = op.Unsqueeze(patch_cpm_mask, axes=[1])
            output_mean = op.Where(cpm, refined_mean, running_mean)
            output_std = op.Where(cpm, refined_std, running_std)

        logits = op.Add(
            op.Mul(
                raw_logits,
                op.Unsqueeze(op.CastLike(output_std, raw_logits), axes=[-1]),
            ),
            op.Unsqueeze(op.CastLike(output_mean, raw_logits), axes=[-1]),
        )
        logits = op.Clip(logits, -self._value_clip, self._value_clip)
        logits = op.Reshape(
            logits,
            [0, 0, 0, config.output_patch_len, len(config.quantiles)],
        )
        return logits, running_mean, running_std

    def stitch_forecast(
        self,
        op: OpBuilder,
        logits: ir.Value,
        trend_slope: ir.Value,
        trend_intercept: ir.Value,
        apply_detrend: ir.Value,
        target_mask: ir.Value,
        nonnegative_mask: ir.Value,
        make_positive: ir.Value,
        context_lengths: ir.Value,
        horizon_lengths: ir.Value,
        context_patch_count: ir.Value,
        forecast_patch_counts: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """Stitch overlapping patches and apply the public forecast policies."""
        config = self.config
        patch_len = config.input_patch_len
        extract_len = min(2 * patch_len, config.output_patch_len)
        overlap = extract_len - patch_len
        num_quantiles = len(config.quantiles)

        max_forecast_patches = op.ReduceMax(forecast_patch_counts, keepdims=False)
        forecast_indices = op.Add(
            op.Range(
                op.Constant(value_int=0),
                max_forecast_patches,
                op.Constant(value_int=1),
            ),
            op.Sub(context_patch_count, op.Constant(value_int=1)),
        )
        patch_predictions = op.Gather(logits, forecast_indices, axis=2)
        patch_predictions = op.Slice(
            patch_predictions,
            starts=[0],
            ends=[extract_len],
            axes=[3],
        )

        # Express stitch_patches as indexed gathers so the single-patch case
        # needs no control flow. Output positions comprise P points per patch
        # plus the final overlap tail.
        stitched_length = op.Add(
            op.Mul(max_forecast_patches, op.Constant(value_int=patch_len)),
            op.Constant(value_int=overlap),
        )
        position = op.Range(
            op.Constant(value_int=0),
            stitched_length,
            op.Constant(value_int=1),
        )
        base_patch = op.Div(position, op.Constant(value_int=patch_len))
        offset = op.Mod(position, op.Constant(value_int=patch_len))
        patch_end = op.Mul(
            op.Unsqueeze(forecast_patch_counts, axes=[1]),
            op.Constant(value_int=patch_len),
        )
        position_b = op.Unsqueeze(position, axes=[0])
        base_patch_b = op.Unsqueeze(base_patch, axes=[0])
        offset_b = op.Unsqueeze(offset, axes=[0])
        is_tail = op.GreaterOrEqual(position_b, patch_end)
        current_patch = op.Min(
            base_patch_b,
            op.Unsqueeze(
                op.Sub(forecast_patch_counts, op.Constant(value_int=1)),
                axes=[1],
            ),
        )
        current_offset = op.Where(
            is_tail,
            op.Add(
                op.Constant(value_int=patch_len),
                op.Sub(position_b, patch_end),
            ),
            offset_b,
        )
        current_linear_index = op.Add(
            op.Mul(current_patch, op.Constant(value_int=extract_len)),
            current_offset,
        )
        previous_patch = op.Max(
            op.Sub(base_patch_b, op.Constant(value_int=1)),
            op.Constant(value_int=0),
        )
        previous_linear_index = op.Add(
            op.Mul(previous_patch, op.Constant(value_int=extract_len)),
            op.Add(op.Constant(value_int=patch_len), offset_b),
        )

        batch = op.Shape(logits, start=0, end=1)
        variates = op.Shape(logits, start=1, end=2)
        gather_shape = op.Concat(
            batch,
            variates,
            op.Unsqueeze(stitched_length, axes=[0]),
            op.Constant(value_ints=[num_quantiles]),
            axis=0,
        )
        flattened_predictions = op.Reshape(
            patch_predictions,
            op.Concat(
                batch,
                variates,
                op.Constant(value_ints=[-1, num_quantiles]),
                axis=0,
            ),
        )
        current = op.GatherElements(
            flattened_predictions,
            op.Expand(
                op.Unsqueeze(current_linear_index, axes=[1, 3]),
                gather_shape,
            ),
            axis=2,
        )
        previous = op.GatherElements(
            flattened_predictions,
            op.Expand(
                op.Unsqueeze(previous_linear_index, axes=[1, 3]),
                gather_shape,
            ),
            axis=2,
        )
        blend_position = op.Cast(offset, to=ir.DataType.FLOAT)
        blend_weight = op.Sub(
            op.Constant(value_float=1.0),
            op.Div(
                blend_position,
                op.Constant(value_float=float(max(overlap - 1, 1))),
            ),
        )
        blend_weight = op.CastLike(op.Unsqueeze(blend_weight, axes=[0, 1, 3]), current)
        blended = op.Add(
            op.Mul(blend_weight, previous),
            op.Mul(
                op.Sub(
                    op.CastLike(op.Constant(value_float=1.0), blend_weight),
                    blend_weight,
                ),
                current,
            ),
        )
        should_blend = op.And(
            op.And(
                op.GreaterOrEqual(position_b, op.Constant(value_int=patch_len)),
                op.Not(is_tail),
            ),
            op.Less(offset_b, op.Constant(value_int=overlap)),
        )
        stitched = op.Where(
            op.Unsqueeze(should_blend, axes=[1, 3]),
            blended,
            current,
        )

        max_horizon = op.ReduceMax(horizon_lengths, keepdims=False)
        stitched = op.Slice(
            stitched,
            starts=op.Constant(value_ints=[0]),
            ends=op.Unsqueeze(max_horizon, axes=[0]),
            axes=op.Constant(value_ints=[2]),
        )
        horizon_position = op.Range(
            op.Constant(value_int=0),
            max_horizon,
            op.Constant(value_int=1),
        )
        trend = op.Add(
            op.Mul(
                op.Unsqueeze(trend_slope, axes=[2]),
                op.Div(
                    op.Cast(
                        op.Unsqueeze(
                            op.Add(horizon_position, op.Constant(value_int=1)),
                            axes=[0, 1],
                        ),
                        to=ir.DataType.FLOAT,
                    ),
                    op.Cast(
                        op.Unsqueeze(context_lengths, axes=[1, 2]),
                        to=ir.DataType.FLOAT,
                    ),
                ),
            ),
            op.Unsqueeze(trend_intercept, axes=[2]),
        )
        trend = op.Where(
            op.Unsqueeze(apply_detrend, axes=[2]),
            trend,
            op.Constant(value_float=0.0),
        )
        forecasts = op.Add(
            stitched,
            op.Unsqueeze(op.CastLike(trend, stitched), axes=[3]),
        )

        # Upstream sorts quantiles before selecting the configured median.
        quantile_forecasts, _ = op.TopK(
            forecasts,
            op.Constant(value_ints=[num_quantiles]),
            axis=-1,
            largest=0,
            sorted=1,
            _outputs=2,
        )
        apply_positive = op.And(nonnegative_mask, make_positive)
        quantile_forecasts = op.Where(
            op.Unsqueeze(apply_positive, axes=[2, 3]),
            op.Max(
                quantile_forecasts,
                op.CastLike(op.Constant(value_float=0.0), quantile_forecasts),
            ),
            quantile_forecasts,
        )
        validity = op.And(
            op.Unsqueeze(target_mask, axes=[2]),
            op.Less(
                op.Unsqueeze(horizon_position, axes=[0, 1]),
                op.Unsqueeze(horizon_lengths, axes=[1, 2]),
            ),
        )
        quantile_forecasts = op.Where(
            op.Unsqueeze(validity, axes=[3]),
            quantile_forecasts,
            op.CastLike(op.Constant(value_float=0.0), quantile_forecasts),
        )
        point_forecast = op.Gather(quantile_forecasts, num_quantiles // 2, axis=-1)
        return point_forecast, quantile_forecasts, validity

    def forward(
        self,
        op: OpBuilder,
        values: ir.Value,
        masks: ir.Value,
        patch_is_target: ir.Value,
        patch_cpm_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """Compose the three export stages for direct module use."""
        model_inputs, patch_mask, running_count, running_mean, running_std = self.preprocess(
            op,
            values,
            masks,
            patch_is_target,
            patch_cpm_mask,
        )
        raw_logits = self.forecast(op, model_inputs, patch_mask)
        return self.postprocess(
            op,
            raw_logits,
            running_count,
            running_mean,
            running_std,
            patch_cpm_mask,
        )
