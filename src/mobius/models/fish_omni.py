# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Drama 3 frame embeddings, hybrid text decoder, and temporal codebook decoder.

Replicates Fish Speech's FishOmniBackbone and FishOmniAudioDecoder. Graphs
consume token/code frames rather than waveforms. The task attaches ONNX sampling
policies; the waveform codec is exported independently by mobius.tasks.fish_codec.
The caller still formats prompts and orchestrates temporal decoding.

Reduced-precision qualification graphs use portable source-ordered row
reductions for offset/gated RMS norms. FP32 norms retain their existing graph;
these scoped substitutions do not open public reduced-precision admission.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._configs import ArchitectureConfig
from mobius._configs.fish_omni import FishOmniConfig
from mobius.components import (
    Embedding,
    Float32SiLU,
    Linear,
    OffsetRMSNorm,
    OutputMajorLinear,
    PostGatedRMSNorm,
    Qwen35Attention,
    SoftmaxTopKGate,
    apply_rotary_pos_emb,
    chunk_gated_delta_rule,
    create_attention_bias,
    float32_silu,
    initialize_rope,
    ordered_row_reduce,
    recurrent_gated_delta_rule,
)
from mobius.components._attention import _apply_attention
from mobius.components._gated_deltanet import _DepthwiseConv1d
from mobius.components._scan_utils import create_body_graph, rename_subgraph_values
from mobius.generation._fish_omni import (
    FishAudioSamplingInputs,
    _all_rows,
    _finite,
    sample_codebook_or_greedy,
)
from mobius.models.qwen35 import (
    Qwen35GatedDeltaNet,
    Qwen35MoEBlock,
    Qwen35MoEDecoderLayer,
    Qwen35MoETextModel,
)


def _fish_linear_class(config):
    # Keep the qualified FP32 projection selection. Reduced-precision source
    # projections require output-major GEMM rather than folded MatMul.
    return Linear if config.dtype == ir.DataType.FLOAT else OutputMajorLinear


def _fish_offset_norm_class(config):
    return OffsetRMSNorm if config.dtype == ir.DataType.FLOAT else _FishSourceOffsetNorm


def _float32_rope(config: ArchitectureConfig) -> nn.Module:
    rope = initialize_rope(dataclasses.replace(config, dtype=ir.DataType.FLOAT))
    assert rope is not None
    for parameter in rope.parameters():
        parameter._keep_float32 = True  # type: ignore[attr-defined]
    return rope


def _sigmoid_float32(op, value):
    # Avoid runtime sigmoid approximations losing small negative-tail values
    # before the source's single FP16/BF16 rounding boundary.
    value_f32 = op.Cast(value, to=ir.DataType.FLOAT)
    sigmoid = op.Div(1.0, op.Add(1.0, op.Exp(op.Neg(value_f32))))
    return op.CastLike(sigmoid, value)


def _apply_fish_attention(
    op, query, key, value, bias, past_key, past_value, *, dtype, **kwargs
):
    if dtype == ir.DataType.FLOAT or bias is None:
        return _apply_attention(op, query, key, value, bias, past_key, past_value, **kwargs)

    # The last causal query can see every key. An all-zero last bias row
    # therefore proves there is no padding; Attention supplies causality itself.
    # Omitting that redundant mask enables the source's fused BF16 attention
    # instead of rounding softmax probabilities in ORT's unfused fallback.
    last_row = op.Cast(op.Gather(bias, -1, axis=-2), to=ir.DataType.FLOAT)
    unpadded = op.Equal(
        op.ReduceMax(op.Abs(last_row), keepdims=False),
        op.CastLike(0.0, last_row),
    )

    def unmasked(branch_op):
        return _apply_attention(
            branch_op, query, key, value, None, past_key, past_value, **kwargs
        )

    def masked(branch_op):
        return _apply_attention(
            branch_op, query, key, value, bias, past_key, past_value, **kwargs
        )

    branches = []
    for name, body in (("unpadded_attention", unmasked), ("padded_attention", masked)):
        op.builder.push_module(name)
        branch = op.builder.subgraph(
            body,
            inputs=[],
            outputs=[
                ir.Value(name=f"{name}_{suffix}") for suffix in ("output", "key", "value")
            ],
            name=name,
        )
        op.builder.pop_module()
        branches.append(branch)
    return op.If(unpadded, then_branch=branches[0], else_branch=branches[1], _outputs=3)


def _rotate_float32(op, value, positions, *, heads, rotary_dim, interleaved):
    rotated = apply_rotary_pos_emb(
        op,
        op.Cast(value, to=ir.DataType.FLOAT),
        positions,
        num_heads=heads,
        rotary_embedding_dim=rotary_dim,
        interleaved=interleaved,
    )
    return op.CastLike(rotated, value)


class _FishGatedNorm(PostGatedRMSNorm):
    """Normalize in FP32, round activation/scale, then apply the FP32 SiLU gate."""

    def forward(self, op, hidden_states, gate):
        hidden_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        unit_scale = op.Expand(op.Constant(value_float=1.0), [self.hidden_size])
        normalized = op.RMSNormalization(
            hidden_f32,
            unit_scale,
            epsilon=self.variance_epsilon,
            axis=-1,
        )
        rounded = op.CastLike(normalized, hidden_states)
        scaled = op.Mul(rounded, self.weight)
        gated = op.Mul(
            op.Cast(scaled, to=ir.DataType.FLOAT),
            float32_silu(op, gate),
        )
        return op.CastLike(gated, hidden_states)


def _source_normalized(op, hidden_states, width, epsilon):
    wide = op.Cast(hidden_states, to=ir.DataType.FLOAT)
    variance = ordered_row_reduce(op, op.Mul(wide, wide), width, normalize=True)
    inverse = op.Pow(op.Add(variance, epsilon), -0.5)
    return op.Mul(wide, inverse)


class _FishSourceOffsetNorm(OffsetRMSNorm):
    """Source-ordered variance; FP32 offset and scale before the final cast."""

    def forward(self, op, hidden_states):
        normalized = _source_normalized(
            op, hidden_states, int(self.weight.shape[0]), self.variance_epsilon
        )
        gamma = op.Add(op.Cast(self.weight, to=ir.DataType.FLOAT), 1.0)
        result = op.CastLike(op.Mul(normalized, gamma), hidden_states)
        result.shape = hidden_states.shape
        return result


class _FishSourceGatedNorm(_FishGatedNorm):
    """Source-ordered variance with separate activation and learned-scale rounding."""

    def forward(self, op, hidden_states, gate):
        normalized = _source_normalized(
            op, hidden_states, self.hidden_size, self.variance_epsilon
        )
        rounded = op.CastLike(normalized, hidden_states)
        # A low-dtype product is exact in FP32 before narrowing. Spell out
        # that boundary: CPU lacks BF16 Mul, and FP16 promotion can erase it.
        scaled = op.CastLike(
            op.Mul(
                op.Cast(rounded, to=ir.DataType.FLOAT),
                op.Cast(self.weight, to=ir.DataType.FLOAT),
            ),
            hidden_states,
        )
        result = op.CastLike(
            op.Mul(op.Cast(scaled, to=ir.DataType.FLOAT), float32_silu(op, gate)),
            hidden_states,
        )
        result.shape = hidden_states.shape
        return result


class _FishSourceConv(_DepthwiseConv1d):
    """Unpadded depthwise convolution; the cache contains only real joined history."""

    def __init__(self, channels, kernel_size):
        super().__init__(channels, kernel_size)
        self._history = kernel_size - 1

    def forward(self, op, input_val, conv_state):
        joined = op.Concat(conv_state, input_val, axis=-1)
        convolved = op.Conv(joined, self.weight, group=self._channels)
        output = op.CastLike(float32_silu(op, convolved), input_val)
        present = (
            op.Slice(joined, [-self._history], [np.iinfo(np.int64).max], [2])
            if self._history
            else op.Slice(joined, [0], [0], [2])
        )
        return output, present


class _FishSourceGatedDeltaNet(Qwen35GatedDeltaNet):
    """Source-layout GDN with explicit sequence reset and chunk/recurrent dispatch.

    The portable solve does not reproduce cuBLAS bits. This qualification
    implementation leaves production reduced-precision admission closed.
    """

    def __init__(self, config):
        super().__init__(config, linear_class=OutputMajorLinear)
        assert self.head_v_dim is not None
        if (
            self.num_k_heads is None
            or self.num_k_heads < 1
            or self.num_v_heads is None
            or self.num_v_heads < 1
            or self.num_v_heads % self.num_k_heads != 0
        ):
            raise ValueError(
                "Source GDN requires value heads to be a positive multiple of key heads"
            )
        self.conv1d = _FishSourceConv(self.conv_dim, self.conv_kernel_size)
        self.norm = _FishSourceGatedNorm(self.head_v_dim, eps=config.rms_norm_eps)

    def forward(self, op, hidden_states, conv_state, recurrent_state, continue_state):
        assert self.num_v_heads is not None and self.num_k_heads is not None
        batch = op.Shape(hidden_states, start=0, end=1)
        length = op.Shape(hidden_states, start=1, end=2)
        # Where, rather than multiplication by zero, ignores stale NaN/Inf too.
        conv_state = op.Where(
            continue_state,
            conv_state,
            op.Expand(op.CastLike(0.0, conv_state), op.Shape(conv_state)),
        )
        recurrent_state = op.Where(
            continue_state, recurrent_state, op.Expand(0.0, op.Shape(recurrent_state))
        )
        mixed = op.Transpose(self.in_proj_qkv(op, hidden_states), perm=[0, 2, 1])
        mixed, present_conv = self.conv1d(op, mixed, conv_state)
        query, key, value = op.Split(
            op.Transpose(mixed, perm=[0, 2, 1]),
            [self.key_dim, self.key_dim, self.value_dim],
            axis=-1,
            _outputs=3,
        )
        q_shape = op.Concat(batch, length, [self.num_k_heads, self.head_k_dim], axis=0)
        v_shape = op.Concat(batch, length, [self.num_v_heads, self.head_v_dim], axis=0)
        query, key = [op.Reshape(x, q_shape) for x in (query, key)]
        if self.num_v_heads != self.num_k_heads:
            groups = self.num_v_heads // self.num_k_heads
            indices = np.repeat(np.arange(self.num_k_heads, dtype=np.int64), groups)
            repeats = op.Constant(value=ir.tensor(indices))
            query, key = [op.Gather(x, repeats, axis=2) for x in (query, key)]
        value = op.Reshape(value, v_shape)
        beta_input = op.Cast(self.in_proj_b(op, hidden_states), to=ir.DataType.FLOAT)
        beta = op.Div(1.0, op.Add(1.0, op.Exp(op.Neg(beta_input))))
        alpha = op.Add(
            op.Cast(self.in_proj_a(op, hidden_states), to=ir.DataType.FLOAT),
            op.Cast(self.dt_bias, to=ir.DataType.FLOAT),
        )
        # Torch softplus's threshold-20 branch is explicit. The portable
        # Softplus kernel is mathematical evidence, not native Log1p bit parity.
        softplus = op.Where(op.Greater(alpha, 20.0), alpha, op.Softplus(alpha))
        # A runtime-dependent shape prevents CPU folding of learned Exp from
        # replacing the source device's rounding in later native qualification.
        expanded_log = op.Expand(
            op.Cast(self.A_log, to=ir.DataType.FLOAT),
            op.Concat(batch, length, [self.num_v_heads], axis=0),
        )
        decay = op.Mul(op.Neg(op.Exp(expanded_log)), softplus)
        branches = []
        for name, rule in (
            ("source_recurrent", recurrent_gated_delta_rule),
            ("source_chunks", chunk_gated_delta_rule),
        ):

            def body(branch_op, rule=rule):
                dimensions = dict(key_width=self.head_k_dim, value_width=self.head_v_dim)
                if rule is chunk_gated_delta_rule:
                    dimensions["heads"] = self.num_v_heads
                return rule(
                    branch_op, query, key, value, decay, beta, recurrent_state, **dimensions
                )

            op.builder.push_module(name)
            branch = op.builder.subgraph(
                body,
                inputs=[],
                outputs=[
                    ir.Value(name=name + "_output", type=ir.TensorType(ir.DataType.FLOAT)),
                    ir.Value(name=name + "_state", type=ir.TensorType(ir.DataType.FLOAT)),
                ],
                name=name,
            )
            op.builder.pop_module()
            branches.append(branch)
        recurrent = op.And(continue_state, op.Equal(op.Squeeze(length, [0]), 1))
        core, present_state = op.If(
            recurrent, then_branch=branches[0], else_branch=branches[1], _outputs=2
        )
        core = op.Reshape(op.CastLike(core, hidden_states), [-1, self.head_v_dim])
        gate = op.Reshape(self.in_proj_z(op, hidden_states), [-1, self.head_v_dim])
        normalized = self.norm(op, core, gate)
        output = op.Reshape(normalized, op.Concat(batch, length, [self.value_dim], axis=0))
        return self.out_proj(op, output), present_conv, present_state


class _FishAttention(Qwen35Attention):
    """Gated GQA with FP32 partial rotation before rounding Q/K back."""

    def __init__(self, config):
        super().__init__(config, linear_class=_fish_linear_class(config))
        self._compute_dtype = config.dtype
        if config.dtype != ir.DataType.FLOAT:
            norm = _fish_offset_norm_class(config)
            self.q_norm = norm(config.head_dim, eps=config.rms_norm_eps)
            self.k_norm = norm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self, op, hidden_states, attention_bias, position_embeddings, past_key_value=None
    ):
        q_gate = op.Reshape(
            self.q_proj(op, hidden_states),
            [0, 0, self.num_attention_heads, self.head_dim * 2],
        )
        query, gate = op.Split(q_gate, num_outputs=2, axis=-1, _outputs=2)
        gate = op.Reshape(gate, [0, 0, -1])
        key = op.Reshape(self.k_proj(op, hidden_states), [0, 0, -1, self.head_dim])
        query = op.Reshape(self.q_norm(op, query), [0, 0, -1])
        key = op.Reshape(self.k_norm(op, key), [0, 0, -1])
        query = _rotate_float32(
            op,
            query,
            position_embeddings,
            heads=self.num_attention_heads,
            rotary_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )
        key = _rotate_float32(
            op,
            key,
            position_embeddings,
            heads=self.num_key_value_heads,
            rotary_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )
        attended, pk, pv = _apply_fish_attention(
            op,
            query,
            key,
            self.v_proj(op, hidden_states),
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
            dtype=self._compute_dtype,
        )
        return self.o_proj(op, op.Mul(attended, _sigmoid_float32(op, gate))), (pk, pv)


class _FishMoEBlock(Qwen35MoEBlock):
    """Accumulate weighted expert outputs in FP32 before returning model dtype."""

    def __init__(self, config, gate=None):
        if gate is None:
            assert config.num_local_experts is not None
            assert config.num_experts_per_tok is not None
            router = (
                _FishRouter if config.dtype == ir.DataType.FLOAT else _FishOutputMajorRouter
            )
            gate = router(
                config.hidden_size,
                config.num_local_experts,
                config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
            )
        super().__init__(config, gate=gate, linear_class=_fish_linear_class(config))
        self.packed_experts = (
            _FishFP8Experts(config) if config.native_fp8_experts else _FishExperts(config)
        )
        self.shared_expert.act_fn = Float32SiLU()

    def forward(self, op, hidden_states):
        routing, selected, logits = self.gate(op, hidden_states)
        routed = self.packed_experts(op, hidden_states, routing, selected, logits)
        return self._add_shared_expert(op, hidden_states, routed)

    def _add_shared_expert(self, op, hidden_states, routed):
        shared = self.shared_expert(op, hidden_states)
        gate = _sigmoid_float32(op, self.shared_expert_gate(op, hidden_states))
        return op.Add(routed, op.Mul(shared, gate))


class _FishExperts(nn.Module):
    """Selected-only expert execution with bounded portable prefill scratch."""

    def __init__(self, config):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.moe_intermediate_size is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.top_k = config.num_experts_per_tok
        self.normalize = int(config.norm_topk_prob)
        self.dtype = config.dtype
        self.act_fn = Float32SiLU()
        self.gate_up_proj = _FishExpertBank(
            [self.num_experts, 2 * self.intermediate_size, self.hidden_size]
        )
        self.down_proj = _FishExpertBank(
            [self.num_experts, self.hidden_size, self.intermediate_size]
        )

    def forward(self, op, hidden_states, weights, selected, router_logits):
        # Materialize banks in the root module scope, never inside If/Scan.
        caps = ep_capabilities()
        if self.dtype == ir.DataType.FLOAT and caps.name == "cuda" and caps.supports_fused_moe:
            return self._fused(
                op,
                hidden_states,
                router_logits,
                self.gate_up_proj(op),
                self.down_proj(op),
            )
        gate_up, down = self.gate_up_proj(op), self.down_proj(op)
        hidden = op.Reshape(hidden_states, [-1, self.hidden_size])
        weights = op.Reshape(weights, [-1, self.top_k])
        selected = op.Reshape(selected, [-1, self.top_k])
        length = op.Shape(hidden, start=0, end=1)
        count = op.Squeeze(length, [0])
        chunk_size = 16

        def direct(branch_op):
            return self._selected(branch_op, hidden, weights, selected, gate_up, down)

        def chunked(branch_op):
            padding = branch_op.Mod(branch_op.Neg(count), chunk_size, fmod=0)
            rows = branch_op.Unsqueeze(padding, [0])

            def pad(value, width, zero):
                tail = branch_op.Expand(zero, branch_op.Concat(rows, [width], axis=0))
                return branch_op.Reshape(
                    branch_op.Concat(value, tail, axis=0), [-1, chunk_size, width]
                )

            h_chunks = pad(hidden, self.hidden_size, branch_op.CastLike(0.0, hidden))
            w_chunks = pad(weights, self.top_k, branch_op.CastLike(0.0, weights))
            i_chunks = pad(selected, self.top_k, branch_op.Constant(value_int=0))
            assert gate_up.name is not None
            prefix = gate_up.name + "_selected_chunks_"
            h = ir.Value(
                name=prefix + "hidden",
                type=ir.TensorType(self.dtype),
                shape=ir.Shape([chunk_size, self.hidden_size]),
            )
            w = ir.Value(
                name=prefix + "weights",
                type=ir.TensorType(self.dtype),
                shape=ir.Shape([chunk_size, self.top_k]),
            )
            ids = ir.Value(
                name=prefix + "ids",
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape([chunk_size, self.top_k]),
            )
            body, builder = create_body_graph([], [h, w, ids], name=prefix + "body")
            output = self._selected(builder.op, h, w, ids, gate_up, down)
            output.type = ir.TensorType(self.dtype)
            output.shape = ir.Shape([chunk_size, self.hidden_size])
            output.name = prefix + "output"
            body.outputs.append(output)
            rename_subgraph_values(body, prefix)
            scanned = branch_op.Scan(
                h_chunks, w_chunks, i_chunks, body=body, num_scan_inputs=3
            )
            return branch_op.Slice(
                branch_op.Reshape(scanned, [-1, self.hidden_size]), [0], length, [0]
            )

        branches = []
        for name, body in (("selected_direct", direct), ("selected_chunks", chunked)):
            op.builder.push_module(name)
            branch = op.builder.subgraph(
                body,
                inputs=[],
                outputs=[ir.Value(name=name + "_output", type=ir.TensorType(self.dtype))],
                name=name,
            )
            op.builder.pop_module()
            branches.append(branch)
        result = op.If(
            op.LessOrEqual(count, chunk_size),
            then_branch=branches[0],
            else_branch=branches[1],
        )
        return op.Reshape(result, op.Shape(hidden_states))

    def _selected(self, op, hidden, weights, selected, gate_up_bank, down_bank):
        ids = op.Reshape(selected, [-1])
        pairs = op.Range(0, op.Size(ids), 1)
        inputs = op.Unsqueeze(op.Gather(hidden, op.Div(pairs, self.top_k), axis=0), [-1])
        projected = op.Squeeze(op.MatMul(op.Gather(gate_up_bank, ids, axis=0), inputs), [-1])
        gate, up = op.Split(
            projected, [self.intermediate_size, self.intermediate_size], axis=-1, _outputs=2
        )
        activated = op.Mul(self.act_fn(op, gate), up)
        projected = op.Squeeze(
            op.MatMul(op.Gather(down_bank, ids, axis=0), op.Unsqueeze(activated, [-1])),
            [-1],
        )
        # Both coefficient rounding and expert-output rounding precede FP32 weighting.
        contribution = op.Mul(
            op.Cast(projected, to=ir.DataType.FLOAT),
            op.Cast(op.Reshape(weights, [-1, 1]), to=ir.DataType.FLOAT),
        )
        summed = op.ReduceSum(
            op.Reshape(contribution, [-1, self.top_k, self.hidden_size]), [1], keepdims=0
        )
        return op.CastLike(summed, hidden)

    def _fused(self, op, hidden_states, router_logits, gate_up_bank, down_bank):
        # Legacy ABI shapes consume output-major bytes, not transposed storage.
        # Prepack [gate | up] offline: passing FC3 would copy the complete
        # 2 GiB expert bank on every layer invocation in the released model.
        gate_up = op.Reshape(
            gate_up_bank, [self.num_experts, self.hidden_size, 2 * self.intermediate_size]
        )
        down = op.Reshape(
            down_bank, [self.num_experts, self.intermediate_size, self.hidden_size]
        )
        output = op.MoE(
            op.Reshape(hidden_states, [-1, self.hidden_size]),
            op.Reshape(router_logits, [-1, self.num_experts]),
            gate_up,
            None,
            down,
            None,
            activation_type="swiglu",
            activation_alpha=1.0,
            activation_beta=0.0,
            swiglu_limit=float("inf"),
            k=self.top_k,
            normalize_routing_weights=self.normalize,
            swiglu_fusion=2,
            _domain="com.microsoft",
        )
        output.type = ir.TensorType(ir.DataType.FLOAT)
        return op.Reshape(output, op.Shape(hidden_states))


class _FishFP8ExpertBank(nn.Module):
    def __init__(self, experts, rows, columns):
        super().__init__()
        self.weight = nn.Parameter([experts, rows, columns], dtype=ir.DataType.FLOAT8E4M3FN)
        self.weight_scale_inv = nn.Parameter(
            [experts, rows // 128, columns // 128], dtype=ir.DataType.FLOAT
        )
        self.weight_scale_inv._keep_float32 = True  # type: ignore[attr-defined]

    def forward(self, op):
        return op.Identity(self.weight), op.Identity(self.weight_scale_inv)


class _FishFP8Experts(nn.Module):
    """Source dynamic W8A8 with unchanged E4M3 bytes and multiplicative FP32 grids."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.dtype = config.dtype
        self.gate_proj = _FishFP8ExpertBank(
            config.num_local_experts, config.moe_intermediate_size, config.hidden_size
        )
        self.up_proj = _FishFP8ExpertBank(
            config.num_local_experts, config.moe_intermediate_size, config.hidden_size
        )
        self.down_proj = _FishFP8ExpertBank(
            config.num_local_experts, config.hidden_size, config.moe_intermediate_size
        )

    def forward(self, op, hidden_states, weights, selected, router_logits):
        if ep_capabilities().name != "cuda":
            raise ValueError("Drama 3 native dynamic FP8 requires execution_provider='cuda'")
        gate, gate_scale = self.gate_proj(op)
        up, up_scale = self.up_proj(op)
        down, down_scale = self.down_proj(op)
        # QMoE owns softmax/top-k and the source's coefficient narrowing.
        # Unsupported devices/route counts must error in the opt-in runtime,
        # never silently fall back to weight-only FP8 or dense expert scratch.
        output = op.QMoE(
            op.Reshape(hidden_states, [-1, self.hidden_size]),
            op.Reshape(router_logits, [-1, self.num_experts]),
            gate,
            gate_scale,
            None,
            down,
            down_scale,
            None,
            up,
            up_scale,
            quant_type="fp8",
            expert_weight_bits=8,
            block_size=128,
            activation_quantization="dynamic_fp8",
            activation_type="silu",
            swiglu_fusion=0,
            k=self.top_k,
            normalize_routing_weights=1,
            _domain="com.microsoft",
        )
        output.type = ir.TensorType(self.dtype)
        return op.Reshape(output, op.Shape(hidden_states))


class _FishExpertBank(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(shape)

    def forward(self, op):
        return op.Identity(self.weight)


class _FishRouter(SoftmaxTopKGate):
    def _project_logits(self, op, hidden_states):
        return op.MatMul(hidden_states, op.Transpose(self.weight, perm=[1, 0]))

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        logits = self._project_logits(op, hidden_states)
        probabilities = op.Softmax(op.Cast(logits, to=ir.DataType.FLOAT), axis=-1)
        weights, indices = op.TopK(
            probabilities, op.Constant(value_ints=[self.top_k]), axis=-1, _outputs=2
        )
        if self.norm_topk_prob:
            weights = op.Div(weights, op.ReduceSum(weights, [-1], keepdims=True))
        return op.CastLike(weights, hidden_states), indices, logits


class _FishOutputMajorRouter(_FishRouter):
    """Source-layout router without adding a checkpoint weight-name segment."""

    def _project_logits(self, op, hidden_states):
        width, hidden = (int(dim) for dim in self.weight.shape)
        projected = op.Gemm(
            op.Reshape(hidden_states, op.Constant(value_ints=[-1, hidden])),
            self.weight,
            transB=1,
        )
        projected.shape = ir.Shape([None, width])
        shape = op.Concat(
            op.Shape(hidden_states, end=-1), op.Constant(value_ints=[width]), axis=0
        )
        return op.Reshape(projected, shape, allowzero=1)


class _FishDecoderLayer(Qwen35MoEDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._compute_dtype = config.dtype
        self.mlp = _FishMoEBlock(config)
        if config.dtype != ir.DataType.FLOAT:
            norm = _fish_offset_norm_class(config)
            self.input_layernorm = norm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = norm(config.hidden_size, eps=config.rms_norm_eps)
        if self.layer_type == "linear_attention":
            if config.dtype != ir.DataType.FLOAT:
                self.linear_attn = _FishSourceGatedDeltaNet(config)
            gated_norm = (
                _FishGatedNorm if config.dtype == ir.DataType.FLOAT else _FishSourceGatedNorm
            )
            self.linear_attn.norm = gated_norm(
                config.linear_value_head_dim,
                eps=config.rms_norm_eps,
            )
        else:
            self.self_attn = _FishAttention(config)

    def forward(
        self,
        op,
        hidden_states,
        attention_bias,
        position_embeddings,
        past_key_value,
        continue_state=None,
    ):
        if self._compute_dtype == ir.DataType.FLOAT:
            return super().forward(
                op, hidden_states, attention_bias, position_embeddings, past_key_value
            )
        if continue_state is None:
            raise ValueError(
                "Source Fish layers require explicit position-derived continuation"
            )
        # FP16 uses FP32 residuals; BF16 keeps the source's BF16 residual boundary.
        residual = (
            op.Cast(hidden_states, to=ir.DataType.FLOAT)
            if self._compute_dtype == ir.DataType.FLOAT16
            else hidden_states
        )
        normed = op.Cast(self.input_layernorm(op, residual), to=self._compute_dtype)
        if self.layer_type == "linear_attention":
            attended, conv, state = self.linear_attn(
                op, normed, past_key_value[0], past_key_value[1], continue_state
            )
            present = (conv, state)
        else:
            attended, present = self.self_attn(
                op, normed, attention_bias, position_embeddings, past_key_value
            )
        residual = op.Add(residual, op.CastLike(attended, residual))
        normed = op.Cast(self.post_attention_layernorm(op, residual), to=self._compute_dtype)
        output = self.mlp(op, normed)
        return op.Add(residual, op.CastLike(output, residual)), present


class _FishTextModel(Qwen35MoETextModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [_FishDecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.rotary_emb = _float32_rope(config)
        if config.dtype != ir.DataType.FLOAT:
            self.norm = _fish_offset_norm_class(config)(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward(
        self,
        op,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values=None,
        inputs_embeds=None,
        deepstack_embeds=None,
    ):
        if self._dtype == ir.DataType.FLOAT:
            return super().forward(
                op,
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                deepstack_embeds,
            )
        if past_key_values is None or len(past_key_values) != len(self.layers):
            raise ValueError(
                "Source Fish decoder requires one explicit dynamic cache per layer"
            )
        hidden = (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(op, input_ids)
        )
        assert self.rotary_emb is not None
        positions = self.rotary_emb(op, position_ids)
        bias = create_attention_bias(
            op,
            input_ids=hidden if input_ids is None else input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        # Qualified callers validate shared contiguous [B,T] rows and
        # past KV length == position_ids[0,0]. No mixed/static cache claim.
        first = op.Gather(op.Gather(position_ids, 0, axis=0), 0, axis=0)
        continuing = op.Greater(first, 0)
        present = []
        for index, (layer, cache) in enumerate(zip(self.layers, past_key_values)):
            hidden, current = layer(
                op, hidden, bias, positions, cache, continue_state=continuing
            )
            present.append(current)
            if deepstack_embeds is not None and index < len(deepstack_embeds):
                hidden = op.Add(hidden, deepstack_embeds[index])
        return self.norm(op, hidden), present


class FishOmniBackbone(nn.Module):
    """Hybrid Qwen3.5-MoE decoder producing both logits and normalized hidden states."""

    def __init__(self, config: FishOmniConfig):
        super().__init__()
        self.model = _FishTextModel(config)
        # The released head exceeds ORT's 1 GiB constant-folding limit;
        # retain output-major storage instead of transposing 2 GiB every step.
        self.lm_head = OutputMajorLinear(config.hidden_size, config.vocab_size, bias=False)
        self._compute_dtype = config.dtype

    def forward(self, op, inputs_embeds, attention_mask, position_ids, past_key_values):
        if self._compute_dtype == ir.DataType.FLOAT16:
            inputs_embeds = op.Cast(inputs_embeds, to=ir.DataType.FLOAT)
        hidden, present = self.model(
            op,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        if self._compute_dtype == ir.DataType.FLOAT16:
            hidden = op.Cast(hidden, to=ir.DataType.FLOAT16)
        return self.lm_head(op, hidden), hidden, present


class FishOmniEmbedding(nn.Module):
    """Mix frame codes into text embeddings and expose unseen codes to the shared tower."""

    def __init__(self, config: FishOmniConfig):
        super().__init__()
        assert config.tower is not None
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.codebook_embeddings = Embedding(
            config.num_codebooks * config.tower.vocab_size, config.hidden_size
        )

    def forward(self, op: OpBuilder, frames: ir.Value):
        cfg = self.config
        assert cfg.tower is not None
        tokens = op.Gather(frames, 0, axis=-1)
        semantic = op.And(
            op.GreaterOrEqual(tokens, cfg.semantic_start_token_id),
            op.LessOrEqual(tokens, cfg.semantic_end_token_id),
        )
        semantic = op.Unsqueeze(semantic, [-1])
        text = self.embed_tokens(op, tokens)  # (B, T, H)
        codebook_vocab = cfg.tower.vocab_size

        def code_sum(first: int, end: int):
            if first == end:
                return op.Mul(text, op.CastLike(0.0, text))
            codes = op.Slice(frames, [first + 1], [end + 1], [-1])
            offsets = op.Constant(value_ints=[i * codebook_vocab for i in range(first, end)])
            embeds = self.codebook_embeddings(op, op.Add(codes, offsets))
            return op.ReduceSum(embeds, [-2], keepdims=False)

        count = cfg.vq_backbone_codebooks
        mixed = op.Mul(
            op.Cast(op.Add(text, code_sum(1, count + 1)), to=ir.DataType.FLOAT),
            1.0 / math.sqrt(1 + count),
        )
        mixed = op.CastLike(mixed, text)
        unseen = code_sum(count + 1, cfg.num_codebooks)
        zero = op.CastLike(0.0, text)
        return op.Where(semantic, mixed, text), op.Where(semantic, unseen, zero)


class _TowerAttention(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        assert config.partial_rotary_factor is not None
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        self.q_width = config.num_attention_heads * config.head_dim
        self.kv_width = config.num_key_value_heads * config.head_dim
        linear = _fish_linear_class(config)
        self.wqkv = linear(config.hidden_size, 2 * (self.q_width + self.kv_width), bias=False)
        self.wo = linear(self.q_width, config.hidden_size, bias=False)
        # Released configs say attention_qk_norm=False, but the trained tower
        # always uses zero-centered per-head Q/K norms (Fish Speech TowerAttention).
        norm = _fish_offset_norm_class(config)
        self.q_norm = norm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = norm(config.head_dim, eps=config.rms_norm_eps)

    def forward(self, op, hidden, bias, positions, past):
        cfg = self.config
        q, gate, k, v = op.Split(
            self.wqkv(op, hidden),
            [self.q_width, self.q_width, self.kv_width, self.kv_width],
            axis=-1,
            _outputs=4,
        )
        q = self.q_norm(op, op.Reshape(q, [0, 0, -1, cfg.head_dim]))
        k = self.k_norm(op, op.Reshape(k, [0, 0, -1, cfg.head_dim]))
        q, k = op.Reshape(q, [0, 0, -1]), op.Reshape(k, [0, 0, -1])
        q = _rotate_float32(
            op,
            q,
            positions,
            heads=cfg.num_attention_heads,
            rotary_dim=self.rotary_dim,
            interleaved=True,
        )
        k = _rotate_float32(
            op,
            k,
            positions,
            heads=cfg.num_key_value_heads,
            rotary_dim=self.rotary_dim,
            interleaved=True,
        )
        attended, pk, pv = _apply_fish_attention(
            op,
            q,
            k,
            v,
            bias,
            past[0],
            past[1],
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            scale=cfg.head_dim**-0.5,
            dtype=cfg.dtype,
        )
        return self.wo(op, op.Mul(attended, _sigmoid_float32(op, gate))), (pk, pv)


class _FP32OutputLinear(OutputMajorLinear):
    """Native FP16-input GEMM with FP32 output, without promoting the weight bank."""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def _project(self, op, hidden):
        projected = op.GemmFloat8(
            hidden,
            self.weight,
            transA=0,
            transB=1,
            alpha=1.0,
            beta=0.0,
            dtype=int(ir.DataType.FLOAT),
            _domain="com.microsoft",
        )
        projected.type = ir.TensorType(ir.DataType.FLOAT)
        return projected


class _TowerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        linear = _fish_linear_class(config)
        self.w1 = linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = linear(config.hidden_size, config.intermediate_size, bias=False)
        if config.dtype == ir.DataType.FLOAT16:
            self.w2 = _FP32OutputLinear(config.intermediate_size, config.hidden_size)
        elif config.dtype == ir.DataType.BFLOAT16:
            self.w2 = OutputMajorLinear(
                config.intermediate_size, config.hidden_size, bias=False
            )
        else:
            self.w2 = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = Float32SiLU()

    def forward(self, op, hidden):
        gate = self.w1(op, hidden)
        return self.w2(op, op.Mul(self.act_fn(op, gate), self.w3(op, hidden)))


class _TowerTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = _TowerAttention(config)
        norm = _fish_offset_norm_class(config)
        self.attention_norm = norm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = norm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = _TowerMLP(config)
        self._compute_dtype = config.dtype

    def forward(self, op, hidden, bias, positions, past):
        if self._compute_dtype == ir.DataType.FLOAT16:
            hidden = op.Cast(hidden, to=ir.DataType.FLOAT)
            normed = op.Cast(self.attention_norm(op, hidden), to=ir.DataType.FLOAT16)
            attended, present = self.attention(op, normed, bias, positions, past)
            residual = op.Add(hidden, op.Cast(attended, to=ir.DataType.FLOAT))
            normed = op.Cast(self.ffn_norm(op, residual), to=ir.DataType.FLOAT16)
            return op.Add(residual, self.feed_forward(op, normed)), present
        attended, present = self.attention(
            op, self.attention_norm(op, hidden), bias, positions, past
        )
        hidden = op.Add(hidden, attended)
        return op.Add(hidden, self.feed_forward(op, self.ffn_norm(op, hidden))), present


class _TowerLayer(nn.Module):
    def __init__(self, config, *, shared: bool):
        super().__init__()
        self.shared = shared
        self.vocab_size = config.vocab_size
        if shared:
            self.semantic_embedding = Embedding(config.vocab_size, config.hidden_size)
        else:
            self.codebook_embedding = Embedding(config.vocab_size, config.hidden_size)
        norm = _fish_offset_norm_class(config)
        self.enorm = norm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = norm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = _fish_linear_class(config)(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        self.transformer_layer = _TowerTransformer(config)
        self.final_layernorm = norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op, hidden, codes, semantic, bias, positions, past, shared_input=None):
        table = self.semantic_embedding if self.shared else self.codebook_embedding
        embedded = table(op, op.Clip(codes, 0, self.vocab_size - 1))
        embedded = op.Where(op.Unsqueeze(semantic, [-1]), embedded, op.CastLike(0.0, hidden))
        if shared_input is not None:
            embedded = op.Add(embedded, shared_input)
        mixed = self.eh_proj(
            op, op.Concat(self.enorm(op, embedded), self.hnorm(op, hidden), axis=-1)
        )
        output, present = self.transformer_layer(op, mixed, bias, positions, past)
        return op.CastLike(self.final_layernorm(op, output), mixed), present


class FishOmniAudioDecoder(nn.Module):
    """Teacher-forced temporal codebook tower; caches persist across audio frames."""

    def __init__(self, config: FishOmniConfig):
        super().__init__()
        assert config.tower is not None
        self.config = config
        tower_config = dataclasses.replace(config.tower, dtype=config.dtype)
        self.shared_layers = nn.ModuleList(
            [
                _TowerLayer(tower_config, shared=True)
                for _ in range(config.shared_enrichment_layers)
            ]
        )
        self.residual_layers = nn.ModuleList(
            [_TowerLayer(tower_config, shared=False) for _ in range(config.num_codebooks - 1)]
        )
        self.codebook_heads = nn.ModuleList(
            [
                _fish_linear_class(config)(
                    config.hidden_size, config.tower.vocab_size, bias=False
                )
                for _ in range(config.num_codebooks - 1)
            ]
        )
        self.rotary_emb = _float32_rope(config.tower)

    def forward(
        self,
        op,
        hidden_states,
        codes,
        semantic_mask,
        shared_input,
        attention_mask,
        position_ids,
        past_key_values,
    ):
        hidden, bias, positions, present = self._shared_forward(
            op,
            hidden_states,
            codes,
            semantic_mask,
            shared_input,
            attention_mask,
            position_ids,
            past_key_values,
        )
        logits = []
        offset = len(self.shared_layers)
        for i, (layer, head) in enumerate(zip(self.residual_layers, self.codebook_heads)):
            hidden, kv = layer(
                op,
                hidden,
                op.Gather(codes, i, axis=-1),
                semantic_mask,
                bias,
                positions,
                past_key_values[offset + i],
            )
            present.append(kv)
            logits.append(op.Unsqueeze(head(op, hidden), [-2]))
        return op.Concat(*logits, axis=-2), present

    def _shared_forward(
        self,
        op,
        hidden_states,
        codes,
        semantic_mask,
        shared_input,
        attention_mask,
        position_ids,
        past_key_values,
    ):
        positions = self.rotary_emb(op, position_ids)
        bias = create_attention_bias(
            op, input_ids=semantic_mask, attention_mask=attention_mask, dtype=self.config.dtype
        )
        hidden = hidden_states
        present = []
        for i, layer in enumerate(self.shared_layers):
            hidden, kv = layer(
                op,
                hidden,
                op.Gather(codes, 0, axis=-1),
                semantic_mask,
                bias,
                positions,
                past_key_values[i],
                shared_input if i == 0 else None,
            )
            present.append(kv)
        return hidden, bias, positions, present

    def forward_sampled(
        self,
        op,
        hidden_states,
        codes,
        semantic_mask,
        shared_input,
        attention_mask,
        position_ids,
        past_key_values,
        sampling: FishAudioSamplingInputs,
    ):
        """Prefill teacher-forced rows or sample one frame in one tower traversal."""
        hidden, bias, positions, present = self._shared_forward(
            op,
            hidden_states,
            codes,
            semantic_mask,
            shared_input,
            attention_mask,
            position_ids,
            past_key_values,
        )
        last_codes = op.Gather(codes, -1, axis=1)
        sampled = [op.Gather(last_codes, 0, axis=-1)]
        valid = op.Expand(
            op.Or(
                op.Not(sampling.generate),
                op.Equal(op.Squeeze(op.Shape(hidden_states, start=1, end=2), [0]), 1),
            ),
            op.Shape(sampling.counter),
        )
        counter = sampling.counter
        logits = []
        offset = len(self.shared_layers)
        for i, (layer, head) in enumerate(zip(self.residual_layers, self.codebook_heads)):
            incoming = op.Where(
                sampling.generate,
                op.Unsqueeze(sampled[-1], [-1]),
                op.Gather(codes, i, axis=-1),
            )
            hidden, kv = layer(
                op,
                hidden,
                incoming,
                semantic_mask,
                bias,
                positions,
                past_key_values[offset + i],
            )
            present.append(kv)
            scores = head(op, hidden)
            logits.append(op.Unsqueeze(scores, [-2]))
            last_scores = op.Gather(scores, -1, axis=1)
            assert self.config.tower is not None
            token, counter, sample_valid = sample_codebook_or_greedy(
                op,
                last_scores,
                sampling,
                counter,
                index=i,
                vocabulary=self.config.tower.vocab_size,
            )
            finite = _all_rows(
                op,
                _finite(op, op.Reshape(op.Cast(scores, to=ir.DataType.FLOAT), [0, -1])),
            )
            valid = op.And(valid, op.And(finite, sample_valid))
            sampled.append(token)
        frame = op.Unsqueeze(
            op.Concat(*[op.Unsqueeze(code, [-1]) for code in sampled], axis=-1), [1]
        )
        frame = op.Where(sampling.generate, frame, op.Unsqueeze(last_codes, [1]))
        frame = op.Where(op.Unsqueeze(valid, [-1, -2]), frame, -1)
        return (
            op.Concat(*logits, axis=-2),
            present,
            frame,
            op.Where(valid, counter, sampling.counter),
            valid,
        )


class FishOmniModel(nn.Module):
    """Fish Audio Drama 3 token-generation pipeline with temporal codebook MTP.

    Exports frame embedding, hybrid MoE decoder, and teacher-forced audio decoder
    graphs with ONNX sampling policies. Reference-bracket prompt formatting and
    temporal orchestration remain caller-owned; the waveform codec is exported
    separately. FP8 experts require explicit streaming dequantization.
    """

    default_task: str = "fish-omni"
    category: str = "Audio"
    config_class = FishOmniConfig
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "decoder": ("model.language_model", "lm_head"),
        "embedding": (
            "model.language_model.embed_tokens",
            "audio_decoder.codebook_embeddings",
        ),
        "audio_decoder": ("audio_decoder_mtp_shared", "audio_decoder_mtp"),
        "audio_generator": ("audio_decoder_mtp_shared", "audio_decoder_mtp"),
    }

    def __init__(self, config: FishOmniConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.decoder = FishOmniBackbone(config)
        self.embedding = FishOmniEmbedding(config)
        self.audio_decoder = FishOmniAudioDecoder(config)
