# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""TimesFM 3 multivariate time-series forecasting model.

Replicates the patched-tensor forward pass of Google Research's
``TimesFM3Torch``. Inputs are already split into fixed-width patches; host-side
interpolation, context padding, detrending, stitching, and quantile policy stay
outside this learned ONNX graph.
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
    model_type: str | None = "timesfm3"

    def __post_init__(self) -> None:
        self.hidden_size = self.model_dims
        self.intermediate_size = self.transformer_hidden_dims
        self.num_hidden_layers = self.num_layers
        self.num_attention_heads = self.num_heads
        self.num_key_value_heads = self.num_heads
        self.head_dim = self.model_dims // self.num_heads

    def validate(self) -> None:
        if self.output_patch_len % self.input_patch_len:
            raise ValueError("output_patch_len must be a multiple of input_patch_len")
        if self.model_dims % self.num_heads:
            raise ValueError("model_dims must be divisible by num_heads")
        if self.head_dim % 2:
            raise ValueError("TimesFM 3 RoPE requires an even head_dim")
        if not self.quantiles:
            raise ValueError("quantiles must not be empty")

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
    """TimesFM 3 patched-input multivariate forecasting network."""

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

    def forward(
        self,
        op: OpBuilder,
        values: ir.Value,
        masks: ir.Value,
        patch_is_target: ir.Value,
        patch_cpm_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
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
        hidden = self.pre_transformer_resblock(op, residual_input)
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
        hidden = self.transformer_stack(op, hidden, effective_patch_mask)
        raw_logits = self.output_head(op, hidden)

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
