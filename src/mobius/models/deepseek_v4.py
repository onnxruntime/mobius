# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSeek-V4 export with a sink-aware dense CSA fallback and MTP sidecar.

The released V4 architecture replaces V3 MLA with compressed sparse attention
and adds Hyper-Connections. This module implements the V4 projections,
Hyper-Connections, hash/sqrt-softplus MoE routing, and a dense causal-attention
fallback with the learned attention sinks. The official per-layer compression
schedule is represented by exported compressor/indexer tensors, and the
checkpoint's MTP block is exported as a standalone sidecar. Executing learned
KV compression and sparse selection still requires runtime cache/sparse ops;
until those land, the target and MTP graphs attend densely for correctness.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import (
    pack_qmoe_expert_weights,
    stack_per_expert_moe_weights,
    supported_qmoe_quantization,
)
from mobius.components import (
    Embedding,
    Linear,
    MoELayer,
    QuantizedEmbedding,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
    make_quantized_linear_factory,
)
from mobius.components._moe import _scatter_selected_to_full
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel

logger = logging.getLogger(__name__)


def _projection_class(config: ArchitectureConfig):
    quantization = config.quantization
    if quantization is None or quantization.quant_method == "none":
        return Linear
    zero_point_dtype = config.dtype if quantization.float_zero_point else ir.DataType.UINT8
    return make_quantized_linear_factory(
        bits=quantization.bits,
        block_size=quantization.group_size,
        has_zero_point=not quantization.sym,
        zero_point_dtype=zero_point_dtype,
    )


def _shape_anchor(op: OpBuilder, parameters: list[ir.Value]) -> ir.Value:
    """Reference one element of each deferred-runtime tensor and produce zero."""
    total = None
    for parameter in parameters:
        first = op.Gather(op.Reshape(parameter, [-1]), [0])
        present = op.Cast(
            op.Equal(first, first),
            to=ir.DataType.INT64,
        )
        total = present if total is None else op.Add(total, present)
    assert total is not None
    return op.ReduceSum(op.Mul(total, 0), [0], keepdims=False)


class DeepSeekV4DeferredProjection(nn.Module):
    """Projection parameters exported for a runtime path not yet executed."""

    def __init__(
        self,
        config: ArchitectureConfig,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        quantization = config.quantization
        self._gguf_quantized_linear = (
            quantization is not None and quantization.quant_method != "none"
        )
        if quantization is None or quantization.quant_method == "none":
            self.weight = nn.Parameter([out_features, in_features])
            self.scales = None
            self.zero_points = None
            return

        n_blocks = (in_features + quantization.group_size - 1) // quantization.group_size
        blob_size = quantization.group_size * quantization.bits // 8
        self.weight = nn.Parameter(
            [out_features, n_blocks, blob_size],
            dtype=ir.DataType.UINT8,
        )
        self.scales = nn.Parameter([out_features, n_blocks])
        if quantization.sym:
            self.zero_points = None
        elif quantization.float_zero_point:
            self.zero_points = nn.Parameter([out_features, n_blocks], dtype=config.dtype)
        else:
            packed_zero_points = (n_blocks * quantization.bits + 7) // 8
            self.zero_points = nn.Parameter(
                [out_features, packed_zero_points],
                dtype=ir.DataType.UINT8,
            )

    def forward(self, op: OpBuilder) -> ir.Value:
        return _shape_anchor(
            op,
            [
                value
                for value in (self.weight, self.scales, self.zero_points)
                if value is not None
            ],
        )


class DeepSeekV4DeferredNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])

    def forward(self, op: OpBuilder) -> ir.Value:
        return _shape_anchor(op, [self.weight])


def _validate_hash_routing_tables(state_dict: dict[str, torch.Tensor]) -> None:
    """Fail fast if a hash-routing table names a duplicate expert for a token.

    ``_scatter_selected_to_full`` (used by ``DeepSeekV4Gate.qmoe_routing`` to
    drive QMoE) requires ``top_k`` *distinct* experts per token:
    ``ScatterElements`` overwrites rather than accumulates duplicate indices,
    so a repeated expert would silently drop one of its contributions instead
    of raising an error. Real hash tables have no reason to route a token to
    the same expert twice, but this checks the actual checkpoint data rather
    than relying on that assumption alone.
    """
    for key, table in state_dict.items():
        if not key.endswith(".mlp.moe.gate.tid2eid") or table.shape[-1] <= 1:
            continue
        sorted_table, _ = torch.sort(table, dim=-1)
        duplicate_rows = torch.any(sorted_table[..., 1:] == sorted_table[..., :-1], dim=-1)
        if duplicate_rows.any():
            bad_tokens = torch.nonzero(duplicate_rows, as_tuple=False).flatten()[:5].tolist()
            raise ValueError(
                f"{key} routes token id(s) {bad_tokens} (showing up to 5 of "
                f"{int(duplicate_rows.sum())}) to a duplicate expert; QMoE "
                "export requires top_k distinct experts per token (see "
                "mobius.components._moe._scatter_selected_to_full)."
            )


class DeepSeekV4Gate(nn.Module):
    """V4 sqrt-softplus router with hash routing for the first layers."""

    def __init__(self, config: ArchitectureConfig, layer_id: int):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.score_func = config.scoring_func
        self.hash_routing = layer_id < config.num_hash_layers
        self.weight = nn.Parameter([self.num_experts, config.hidden_size])
        # Hash-routed layers do not consume this bias, but DeepSeek V4 GGUF
        # checkpoints provide it for every layer.
        self.bias = nn.Parameter([self.num_experts])
        if self.hash_routing:
            self.tid2eid = nn.Parameter(
                [config.vocab_size, self.top_k], dtype=ir.DataType.INT32
            )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
    ):
        logits = op.MatMul(
            op.Cast(hidden_states, to=ir.DataType.FLOAT.value),
            op.Transpose(self.weight, perm=[1, 0]),
        )
        if self.score_func == "softmax":
            scores = op.Softmax(logits, axis=-1)
        elif self.score_func == "sigmoid":
            scores = op.Sigmoid(logits)
        else:
            scores = op.Sqrt(op.Softplus(logits))

        if self.hash_routing:
            selected_experts = op.Cast(
                op.Gather(self.tid2eid, input_ids, axis=0),
                to=ir.DataType.INT64.value,
            )
        else:
            choice_scores = op.Add(scores, self.bias)
            _, selected_experts = op.TopK(
                choice_scores,
                op.Constant(value_ints=[self.top_k]),
                axis=-1,
                _outputs=2,
            )

        routing_weights = op.GatherElements(scores, selected_experts, axis=-1)
        if self.score_func != "softmax":
            weight_sum = op.ReduceSum(routing_weights, [-1], keepdims=True)
            routing_weights = op.Div(routing_weights, op.Add(weight_sum, 1e-20))
        if self.route_scale != 1.0:  # noqa: RUF069
            routing_weights = op.Mul(routing_weights, self.route_scale)
        return routing_weights, selected_experts

    def qmoe_routing(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
    ):
        """Adapt the already-selected (routing_weights, selected_experts) pair for QMoE.

        This does not alter their computation.

        Hash routing (``tid2eid``) is not expressible as "top-k of a
        per-expert score" (QMoE's only selection ABI), so this reuses
        ``forward()`` verbatim -- covering both hash and learned top-k
        routing identically -- and scatters its output into the
        full-``num_experts``-width tensors QMoE requires. See
        ``_scatter_selected_to_full`` for why this preserves the exact
        selection and weights, and for why -- like V3's ``DeepSeekMoEGate``
        -- this path is CPU-EP-correct only: CUDA QMoE ignores the gathered
        ``router_weights`` this adapter relies on (learned layers select via
        ``scores + bias`` but weight by ``scores`` alone, so raw-logit
        passthrough can't make CUDA's forced internal recompute agree either).
        """
        routing_weights, selected_experts = self.forward(op, hidden_states, input_ids)
        # QMoE's router_probs/router_weights share type constraint "T" with
        # hidden_states; routing_weights is computed in float32 (matching the
        # reference sqrt-softplus/softmax/sigmoid scoring), so cast back
        # before scattering.
        routing_weights = op.CastLike(routing_weights, hidden_states)
        router_probs, router_weights = _scatter_selected_to_full(
            op, routing_weights, selected_experts, self.num_experts
        )
        # route_scale (and, for non-softmax scoring, weight_sum renormalization)
        # is already folded into routing_weights above, so QMoE must not
        # renormalize or rescale again.
        return router_probs, router_weights, False, 1.0


class _DeepSeekV4Expert(nn.Module):
    def __init__(self, config: ArchitectureConfig, intermediate_size: int):
        super().__init__()
        projection = _projection_class(config)
        self.gate_proj = projection(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = projection(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = projection(intermediate_size, config.hidden_size, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        gate = self.gate_proj(op, hidden_states)
        up = self.up_proj(op, hidden_states)
        if self.limit > 0:
            gate = op.Clip(gate, None, self.limit)
            up = op.Clip(up, -self.limit, self.limit)
        return self.down_proj(op, op.Mul(op.Swish(gate), up))


class DeepSeekV4MoE(nn.Module):
    def __init__(self, config: ArchitectureConfig, layer_id: int):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.moe_intermediate_size is not None
        gate = DeepSeekV4Gate(config, layer_id)
        # QMoE's clipped-SwiGLU maps exactly onto _DeepSeekV4Expert's
        # activation (plain SiLU: alpha=1.0, beta=0.0). config.swiglu_limit<=0
        # means "no clipping" in _DeepSeekV4Expert.forward, but QMoE treats
        # swiglu_limit=0.0 as "clip to zero" -- math.inf is required to
        # disable clipping at the op level.
        swiglu_limit = config.swiglu_limit if config.swiglu_limit > 0 else math.inf
        self.moe = MoELayer(
            config,
            gate=gate,
            expert_factory=lambda expert_config, _linear_class: _DeepSeekV4Expert(
                expert_config, expert_config.intermediate_size
            ),
            activation_alpha=1.0,
            activation_beta=0.0,
            swiglu_limit=swiglu_limit,
        )
        shared_size = config.moe_intermediate_size * (config.n_shared_experts or 1)
        self.shared_experts = _DeepSeekV4Expert(config, shared_size)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
    ):
        moe_output = self.moe(op, hidden_states, input_ids)
        return op.Add(moe_output, self.shared_experts(op, hidden_states))


class DeepSeekV4CompressorTensors(nn.Module):
    """Official learned compressor tensors retained for sparse-runtime handoff."""

    def __init__(self, config: ArchitectureConfig, compress_ratio: int, head_dim: int):
        super().__init__()
        overlap_factor = 2 if compress_ratio == 4 else 1
        self.ape = nn.Parameter([compress_ratio, overlap_factor * head_dim])
        self.wkv = DeepSeekV4DeferredProjection(
            config, config.hidden_size, overlap_factor * head_dim
        )
        self.wgate = DeepSeekV4DeferredProjection(
            config, config.hidden_size, overlap_factor * head_dim
        )
        self.norm = DeepSeekV4DeferredNorm(head_dim)

    def forward(self, op: OpBuilder) -> ir.Value:
        anchor = _shape_anchor(
            op,
            [
                self.ape,
            ],
        )
        anchor = op.Add(anchor, self.wkv(op))
        anchor = op.Add(anchor, self.wgate(op))
        return op.Add(anchor, self.norm(op))


class DeepSeekV4IndexerTensors(nn.Module):
    """Official ratio-4 sparse indexer tensors retained in the dense graph."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.q_lora_rank is not None
        assert config.index_n_heads is not None
        assert config.index_head_dim is not None
        self.wq_b = DeepSeekV4DeferredProjection(
            config,
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
        )
        self.weights_proj = DeepSeekV4DeferredProjection(
            config, config.hidden_size, config.index_n_heads
        )
        self.compressor = DeepSeekV4CompressorTensors(
            config, compress_ratio=4, head_dim=config.index_head_dim
        )

    def forward(self, op: OpBuilder) -> ir.Value:
        own = op.Add(self.wq_b(op), self.weights_proj(op))
        return op.Add(own, self.compressor(op))


class DeepSeekV4Attention(nn.Module):
    """V4 MQA projections using sink-aware dense causal attention."""

    def __init__(self, config: ArchitectureConfig, layer_id: int):
        super().__init__()
        assert config.q_lora_rank is not None
        assert config.qk_rope_head_dim is not None
        assert config.o_lora_rank is not None
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.nope_dim = self.head_dim - self.rope_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        assert self.num_heads % self.o_groups == 0
        projection = _projection_class(config)

        self.q_a_proj = projection(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = projection(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.kv_proj = projection(config.hidden_size, self.head_dim, bias=False)
        self.kv_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        group_width = self.num_heads * self.head_dim // self.o_groups
        self.o_a_proj = projection(
            group_width,
            self.o_groups * self.o_lora_rank,
            bias=False,
        )
        self.o_b_proj = projection(
            self.o_groups * self.o_lora_rank,
            config.hidden_size,
            bias=False,
        )
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
        self.rope_interleave = config.rope_interleave
        ratios = config.compress_ratios or []
        self.compress_ratio = ratios[layer_id] if layer_id < len(ratios) else 0
        if self.compress_ratio not in (0, 4, 128):
            raise ValueError(
                "DeepSeek-V4 supports compression ratios 0, 4, and 128; "
                f"layer {layer_id} requested {self.compress_ratio}"
            )
        self.attn_sink = nn.Parameter([self.num_heads], dtype=ir.DataType.FLOAT)
        self.compressor = (
            DeepSeekV4CompressorTensors(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio
            else None
        )
        self.indexer = DeepSeekV4IndexerTensors(config) if self.compress_ratio == 4 else None

    def _rotate(
        self,
        op: OpBuilder,
        value: ir.Value,
        position_embeddings: tuple,
        num_heads: int,
        *,
        inverse: bool = False,
    ):
        value = op.Reshape(value, [0, 0, num_heads, self.head_dim])
        nope, rope = op.Split(value, [self.nope_dim, self.rope_dim], axis=-1, _outputs=2)
        rope = op.Reshape(rope, [0, 0, -1])
        if inverse:
            position_embeddings = (position_embeddings[0], op.Neg(position_embeddings[1]))
        rope = apply_rotary_pos_emb(
            op,
            rope,
            position_embeddings,
            num_heads=num_heads,
            rotary_embedding_dim=0,
            interleaved=self.rope_interleave,
        )
        rope = op.Reshape(rope, [0, 0, num_heads, self.rope_dim])
        return op.Reshape(op.Concat(nope, rope, axis=-1), [0, 0, -1])

    def _expand_kv(
        self,
        op: OpBuilder,
        value: ir.Value,
        batch: ir.Value,
        sequence_length: ir.Value,
    ) -> ir.Value:
        value = op.Unsqueeze(value, [2])
        value = op.Expand(
            value,
            op.Concat(
                batch,
                [1, self.num_heads],
                sequence_length,
                [self.head_dim],
                axis=0,
            ),
        )
        return op.Reshape(
            value,
            op.Concat(
                batch,
                [self.num_heads],
                sequence_length,
                [self.head_dim],
                axis=0,
            ),
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        query = self.q_b_proj(op, self.q_a_layernorm(op, self.q_a_proj(op, hidden_states)))
        query_4d = op.Reshape(query, [0, 0, self.num_heads, self.head_dim])
        query_rms = op.Sqrt(
            op.Add(
                op.ReduceMean(op.Mul(query_4d, query_4d), [-1], keepdims=True),
                self.eps,
            )
        )
        query = op.Reshape(op.Div(query_4d, query_rms), [0, 0, -1])
        query = self._rotate(op, query, position_embeddings, self.num_heads)

        kv = self.kv_layernorm(op, self.kv_proj(op, hidden_states))
        kv = self._rotate(op, kv, position_embeddings, 1)
        batch = op.Shape(query, start=0, end=1)
        query_length = op.Shape(query, start=1, end=2)
        query = op.Transpose(
            op.Reshape(query, [0, 0, self.num_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )
        key = op.Transpose(op.Reshape(kv, [0, 0, 1, self.head_dim]), perm=[0, 2, 1, 3])
        value = key
        if past_key_value is not None:
            key = op.Concat(past_key_value[0], key, axis=2)
            value = op.Concat(past_key_value[1], value, axis=2)
        present_key, present_value = key, value

        kv_length = op.Shape(key, start=2, end=3)
        key = self._expand_kv(op, key, batch, kv_length)
        value = self._expand_kv(op, value, batch, kv_length)
        scores = op.Mul(
            op.MatMul(query, op.Transpose(key, perm=[0, 1, 3, 2])),
            self.scale,
        )
        scores = op.Add(scores, attention_bias)
        sinks = op.Expand(
            op.Reshape(
                op.CastLike(self.attn_sink, scores),
                [1, self.num_heads, 1, 1],
            ),
            op.Concat(batch, [self.num_heads], query_length, [1], axis=0),
        )
        probabilities = op.Softmax(op.Concat(scores, sinks, axis=-1), axis=-1)
        probabilities = op.Slice(probabilities, [0], [-1], [3])
        output = op.Reshape(
            op.Transpose(op.MatMul(probabilities, value), perm=[0, 2, 1, 3]),
            [0, 0, -1],
        )
        output = self._rotate(op, output, position_embeddings, self.num_heads, inverse=True)

        if self.compressor is not None:
            anchor = self.compressor(op)
            if self.indexer is not None:
                anchor = op.Add(anchor, self.indexer(op))
            output = op.Add(output, op.CastLike(anchor, output))

        group_width = self.num_heads * self.head_dim // self.o_groups
        groups = op.Split(
            output,
            [group_width] * self.o_groups,
            axis=-1,
            _outputs=self.o_groups,
        )
        projected_groups = []
        for group_idx, group in enumerate(groups):
            projected = self.o_a_proj(op, group)
            projected_groups.append(
                op.Slice(
                    projected,
                    [group_idx * self.o_lora_rank],
                    [(group_idx + 1) * self.o_lora_rank],
                    [-1],
                )
            )
        output = self.o_b_proj(op, op.Concat(*projected_groups, axis=-1))
        return output, (present_key, present_value)


class DeepSeekV4DecoderLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig, layer_id: int):
        super().__init__()
        self.self_attn = DeepSeekV4Attention(config, layer_id)
        self.mlp = DeepSeekV4MoE(config, layer_id)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        hc_dim = config.hc_mult * config.hidden_size
        mix_dim = (2 + config.hc_mult) * config.hc_mult
        self.hc_attn_fn = Linear(hc_dim, mix_dim, bias=False)
        self.hc_attn_base = nn.Parameter([mix_dim])
        self.hc_attn_scale = nn.Parameter([3])
        self.hc_ffn_fn = Linear(hc_dim, mix_dim, bias=False)
        self.hc_ffn_base = nn.Parameter([mix_dim])
        self.hc_ffn_scale = nn.Parameter([3])

    def _hc_pre(self, op, states, fn, scale, base):
        flat = op.Reshape(states, [0, 0, -1])
        rms = op.Sqrt(op.Add(op.ReduceMean(op.Mul(flat, flat), [-1], keepdims=True), 1e-6))
        mixes = op.Div(fn(op, flat), rms)
        pre_raw, post_raw, comb_raw = op.Split(
            mixes,
            [self.hc_mult, self.hc_mult, self.hc_mult * self.hc_mult],
            axis=-1,
            _outputs=3,
        )
        base_pre, base_post, base_comb = op.Split(
            base,
            [self.hc_mult, self.hc_mult, self.hc_mult * self.hc_mult],
            axis=-1,
            _outputs=3,
        )
        scale_pre, scale_post, scale_comb = op.Split(scale, [1, 1, 1], axis=-1, _outputs=3)
        pre = op.Add(
            op.Sigmoid(op.Add(op.Mul(pre_raw, scale_pre), base_pre)),
            self.hc_eps,
        )
        post = op.Mul(op.Sigmoid(op.Add(op.Mul(post_raw, scale_post), base_post)), 2.0)
        comb = op.Reshape(
            op.Add(op.Mul(comb_raw, scale_comb), base_comb),
            [0, 0, self.hc_mult, self.hc_mult],
        )
        comb = op.Add(op.Softmax(comb, axis=-1), self.hc_eps)
        comb = op.Div(
            comb,
            op.Add(op.ReduceSum(comb, [-2], keepdims=True), self.hc_eps),
        )
        for _ in range(max(self.hc_iters - 1, 0)):
            comb = op.Div(
                comb,
                op.Add(op.ReduceSum(comb, [-1], keepdims=True), self.hc_eps),
            )
            comb = op.Div(
                comb,
                op.Add(op.ReduceSum(comb, [-2], keepdims=True), self.hc_eps),
            )
        reduced = op.ReduceSum(op.Mul(op.Unsqueeze(pre, [-1]), states), [2], keepdims=False)
        return reduced, post, comb

    @staticmethod
    def _hc_post(op, value, residual, post, comb):
        injected = op.Mul(op.Unsqueeze(post, [-1]), op.Unsqueeze(value, [-2]))
        mixed = op.ReduceSum(
            op.Mul(op.Unsqueeze(comb, [-1]), op.Unsqueeze(residual, [-2])),
            [2],
            keepdims=False,
        )
        return op.Add(injected, mixed)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        value, post, comb = self._hc_pre(
            op,
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        value, present = self.self_attn(
            op,
            self.input_layernorm(op, value),
            attention_bias,
            position_embeddings,
            past_key_value,
        )
        hidden_states = self._hc_post(op, value, residual, post, comb)

        residual = hidden_states
        value, post, comb = self._hc_pre(
            op,
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        value = self.mlp(op, self.post_attention_layernorm(op, value), input_ids)
        hidden_states = self._hc_post(op, value, residual, post, comb)
        return hidden_states, present


class DeepSeekV4TextModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        if config.quantization is not None and config.quantization.quantize_embeddings:
            self.embed_tokens = QuantizedEmbedding(
                config.vocab_size,
                config.hidden_size,
                bits=config.quantization.bits,
                block_size=config.quantization.group_size,
                has_zero_point=not config.quantization.sym,
                padding_idx=config.pad_token_id,
            )
        self.layers = nn.ModuleList(
            [
                DeepSeekV4DecoderLayer(config, layer_id)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = Linear(hc_dim, config.hc_mult, bias=False)
        self.hc_head_base = nn.Parameter([config.hc_mult])
        self.hc_head_scale = nn.Parameter([1])
        rope_config = config
        if config.qk_rope_head_dim is not None:
            rope_config = dataclasses.replace(config, head_dim=config.qk_rope_head_dim)
        self.rotary_emb = initialize_rope(
            dataclasses.replace(
                rope_config,
                rope_type="default",
                rope_scaling=None,
                original_max_position_embeddings=None,
            )
        )
        self.compressed_rotary_emb = initialize_rope(
            dataclasses.replace(
                rope_config,
                rope_theta=config.compress_rope_theta or config.rope_theta,
            )
        )
        self._dtype = config.dtype

    def _hc_head(self, op, states):
        flat = op.Reshape(states, [0, 0, -1])
        rms = op.Sqrt(op.Add(op.ReduceMean(op.Mul(flat, flat), [-1], keepdims=True), 1e-6))
        mixes = op.Div(self.hc_head_fn(op, flat), rms)
        weights = op.Add(
            op.Sigmoid(
                op.Add(
                    op.Mul(mixes, self.hc_head_scale),
                    self.hc_head_base,
                )
            ),
            self.hc_eps,
        )
        return op.ReduceSum(op.Mul(op.Unsqueeze(weights, [-1]), states), [2], keepdims=False)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(op, input_ids)
        )
        hidden_states = op.Expand(
            op.Unsqueeze(hidden_states, [-2]),
            [1, 1, self.hc_mult, 1],
        )
        position_embeddings = self.rotary_emb(op, position_ids)
        compressed_position_embeddings = self.compressed_rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        presents = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            layer_position_embeddings = (
                compressed_position_embeddings
                if layer.self_attn.compress_ratio
                else position_embeddings
            )
            hidden_states, present = layer(
                op,
                hidden_states,
                input_ids,
                attention_bias,
                layer_position_embeddings,
                past_kv,
            )
            presents.append(present)
        return self.norm(op, self._hc_head(op, hidden_states)), presents, hidden_states


class DeepSeekV4Mtp(DeepSeekV4DecoderLayer):
    """Official single MTP block, sharing target embeddings and LM head externally."""

    def __init__(self, config: ArchitectureConfig, layer_id: int):
        super().__init__(config, layer_id)
        projection = _projection_class(config)
        self.e_proj = projection(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = projection(config.hidden_size, config.hidden_size, bias=False)
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head_fn = Linear(
            config.hc_mult * config.hidden_size, config.hc_mult, bias=False
        )
        self.hc_head_base = nn.Parameter([config.hc_mult])
        self.hc_head_scale = nn.Parameter([1])
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        rope_config = dataclasses.replace(
            config,
            head_dim=config.qk_rope_head_dim,
            rope_type="default",
            rope_scaling=None,
            original_max_position_embeddings=None,
        )
        self.rotary_emb = initialize_rope(rope_config)
        self._dtype = config.dtype

    def _hc_head(self, op: OpBuilder, states: ir.Value) -> ir.Value:
        flat = op.Reshape(states, [0, 0, -1])
        rms = op.Sqrt(op.Add(op.ReduceMean(op.Mul(flat, flat), [-1], keepdims=True), 1e-6))
        mixes = op.Div(self.hc_head_fn(op, flat), rms)
        weights = op.Add(
            op.Sigmoid(op.Add(op.Mul(mixes, self.hc_head_scale), self.hc_head_base)),
            self.hc_eps,
        )
        return op.ReduceSum(op.Mul(op.Unsqueeze(weights, [-1]), states), [2], keepdims=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_value: tuple | None = None,
    ):
        hidden_states = op.Add(
            op.Unsqueeze(self.e_proj(op, self.enorm(op, inputs_embeds)), [2]),
            self.h_proj(op, self.hnorm(op, hidden_states)),
        )
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=inputs_embeds,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        hidden_states, present = super().forward(
            op,
            hidden_states,
            None,
            attention_bias,
            position_embeddings,
            past_key_value,
        )
        return self.norm(op, self._hc_head(op, hidden_states)), present


class DeepSeekV4CausalLMModel(CausalLMModel):
    """DeepSeek-V4 Causal LM with dense CSA fallback and an MTP sidecar."""

    default_task: str = "deepseek-v4"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        if any(config.compress_ratios or ()):
            logger.warning(
                "DeepSeek-V4 sparse cache execution requires runtime support; "
                "exporting sink-aware dense attention with CSA/HCA tensors retained."
            )
        self.config = config
        self.model = DeepSeekV4TextModel(config)
        if config.quantization is not None and config.quantization.quantize_lm_head:
            self.lm_head = _projection_class(config)(
                config.hidden_size, config.vocab_size, bias=False
            )
        else:
            self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.num_nextn_predict_layers not in (0, 1):
            raise ValueError("DeepSeek-V4 MTP export supports exactly one MTP layer")
        self.mtp = nn.ModuleList(
            [
                DeepSeekV4Mtp(config, config.num_hidden_layers + index)
                for index in range(config.num_nextn_predict_layers)
            ]
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, presents, _ = self.model(
            op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        )
        return self.lm_head(op, hidden_states), presents

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map the official DeepSeek checkpoint names to mobius modules."""
        # Same predicate as MoELayer/_supported_qmoe_quantization so the
        # repacked weights and the emitted graph never disagree.
        use_qmoe = supported_qmoe_quantization(self.config.quantization) is not None
        renamed: dict[str, torch.Tensor] = {}
        skipped = 0
        for key, value in state_dict.items():
            if key.startswith("mtp.") and len(self.mtp) == 0:
                skipped += 1
                continue
            new_key = key
            if new_key == "embed.weight":
                new_key = "model.embed_tokens.weight"
            elif new_key == "head.weight":
                new_key = "lm_head.weight"
            elif new_key == "norm.weight":
                new_key = "model.norm.weight"
            elif new_key.startswith("hc_head_"):
                new_key = (
                    f"model.{new_key}.weight"
                    if new_key == "hc_head_fn"
                    else f"model.{new_key}"
                )
            elif new_key.startswith("layers."):
                new_key = f"model.{new_key}"
            elif new_key.startswith("mtp."):
                try:
                    mtp_index = int(new_key.split(".", 2)[1])
                except (IndexError, ValueError):
                    skipped += 1
                    continue
                if mtp_index >= len(self.mtp):
                    skipped += 1
                    continue
            if new_key.startswith(("model.layers.", "mtp.")):
                new_key = new_key.replace(".attn.wq_a.", ".self_attn.q_a_proj.")
                new_key = new_key.replace(".attn.q_norm.", ".self_attn.q_a_layernorm.")
                new_key = new_key.replace(".attn.wq_b.", ".self_attn.q_b_proj.")
                new_key = new_key.replace(".attn.wkv.", ".self_attn.kv_proj.")
                new_key = new_key.replace(".attn.kv_norm.", ".self_attn.kv_layernorm.")
                new_key = new_key.replace(".attn.wo_a.", ".self_attn.o_a_proj.")
                new_key = new_key.replace(".attn.wo_b.", ".self_attn.o_b_proj.")
                new_key = new_key.replace(".attn.", ".self_attn.")
                new_key = new_key.replace(".attn_norm.", ".input_layernorm.")
                new_key = new_key.replace(".ffn_norm.", ".post_attention_layernorm.")
                # DeepSeekV4MoE composes the shared MoELayer, so the gate
                # lives at mlp.moe.gate.* (see DeepSeekV4MoE.__init__).
                new_key = new_key.replace(".ffn.gate.", ".mlp.moe.gate.")
                new_key = new_key.replace(".ffn.experts.", ".mlp.experts.")
                new_key = new_key.replace(".ffn.shared_experts.", ".mlp.shared_experts.")
                new_key = new_key.replace(".w1.", ".gate_proj.")
                new_key = new_key.replace(".w2.", ".down_proj.")
                new_key = new_key.replace(".w3.", ".up_proj.")
                if ".hc_" in new_key and new_key.endswith("_fn"):
                    new_key = f"{new_key}.weight"
                # Dense fallback (unquantized or non-QMoE-eligible): experts
                # are a plain ModuleList under moe.experts.{i}.*. Skipped for
                # the QMoE path -- stack_per_expert_moe_weights below expects
                # this same per-index ".mlp.experts.{i}.*" layout as input,
                # fusing it into the tensors pack_qmoe_expert_weights expects.
                if not use_qmoe:
                    new_key = new_key.replace(".mlp.experts.", ".mlp.moe.experts.")
            renamed[new_key] = value
        if skipped:
            logger.warning(
                "Skipped %d DeepSeek-V4 MTP tensors outside the configured "
                "num_nextn_predict_layers.",
                skipped,
            )
        processed = super().preprocess_weights(renamed)
        if use_qmoe:
            _validate_hash_routing_tables(processed)
            # DeepSeek-V4 checkpoints store routed experts per-index
            # (".mlp.experts.{i}.gate_proj/up_proj/down_proj.*"), unlike
            # DeepSeek-V3's already-fused HF format. Bridge to the fused
            # expert-major layout pack_qmoe_expert_weights expects.
            processed = stack_per_expert_moe_weights(processed, qmoe_target_path=".mlp")
            processed = pack_qmoe_expert_weights(processed)
        return processed
