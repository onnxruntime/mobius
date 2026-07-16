# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSeek-V4 dense-backbone export.

The released V4 architecture replaces V3 MLA with compressed sparse attention
and adds Hyper-Connections. This module implements the V4 projections,
Hyper-Connections, hash/sqrt-softplus MoE routing, and a dense causal-attention
fallback. Learned KV compression, sparse indexing, attention sinks, and MTP are
intentionally deferred until the runtime exposes suitable cache/sparse ops.
"""

from __future__ import annotations

import logging

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Embedding,
    Linear,
    QuantizedEmbedding,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
    make_quantized_linear_factory,
)
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
        if self.hash_routing:
            self.tid2eid = nn.Parameter(
                [config.vocab_size, self.top_k], dtype=ir.DataType.INT32
            )
        else:
            self.bias = nn.Parameter([self.num_experts])

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
            selected_experts = op.Gather(self.tid2eid, input_ids, axis=0)
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
        return self.down_proj(op, op.Mul(op.Mul(gate, op.Sigmoid(gate)), up))


class DeepSeekV4MoE(nn.Module):
    def __init__(self, config: ArchitectureConfig, layer_id: int):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.moe_intermediate_size is not None
        self.gate = DeepSeekV4Gate(config, layer_id)
        self.experts = nn.ModuleList(
            [
                _DeepSeekV4Expert(config, config.moe_intermediate_size)
                for _ in range(config.num_local_experts)
            ]
        )
        shared_size = config.moe_intermediate_size * (config.n_shared_experts or 1)
        self.shared_experts = _DeepSeekV4Expert(config, shared_size)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
    ):
        routing_weights, selected_experts = self.gate(op, hidden_states, input_ids)
        result = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(op, hidden_states)
            match = op.Equal(selected_experts, op.Constant(value_int=expert_idx))
            weight = op.ReduceSum(
                op.Mul(routing_weights, op.CastLike(match, routing_weights)),
                [-1],
                keepdims=True,
            )
            contribution = op.Mul(expert_output, weight)
            result = contribution if result is None else op.Add(result, contribution)
        return op.Add(result, self.shared_experts(op, hidden_states))


class DeepSeekV4Attention(nn.Module):
    """V4 MQA projections using dense causal attention as a safe fallback."""

    def __init__(self, config: ArchitectureConfig):
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
        output, present_key, present_value = op.Attention(
            query,
            kv,
            kv,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            q_num_heads=self.num_heads,
            kv_num_heads=1,
            scale=self.scale,
            _outputs=3,
        )
        output = self._rotate(op, output, position_embeddings, self.num_heads, inverse=True)

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
        self.self_attn = DeepSeekV4Attention(config)
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
            import dataclasses

            rope_config = dataclasses.replace(config, head_dim=config.qk_rope_head_dim)
        self.rotary_emb = initialize_rope(rope_config)
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
        attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        presents = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present = layer(
                op,
                hidden_states,
                input_ids,
                attention_bias,
                position_embeddings,
                past_kv,
            )
            presents.append(present)
        return self.norm(op, self._hc_head(op, hidden_states)), presents


class DeepSeekV4CausalLMModel(CausalLMModel):
    """DeepSeek-V4 Causal LM with dense HCA fallback and V4 MoE/HC blocks."""

    default_task: str = "text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        if any(config.compress_ratios or ()):
            logger.warning(
                "DeepSeek-V4 compressed sparse attention is not implemented; "
                "exporting the dense causal-attention preview backbone."
            )
        self.config = config
        self.model = DeepSeekV4TextModel(config)
        if config.quantization is not None and config.quantization.quantize_lm_head:
            self.lm_head = _projection_class(config)(
                config.hidden_size, config.vocab_size, bias=False
            )
        else:
            self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map the official DeepSeek checkpoint names to mobius modules."""
        renamed: dict[str, torch.Tensor] = {}
        skipped = 0
        for key, value in state_dict.items():
            if key.startswith("mtp.") or any(
                marker in key
                for marker in (
                    ".compressor.",
                    ".indexer.",
                    ".attn.attn_sink",
                )
            ):
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
                new_key = new_key.replace(".attn.wq_a.", ".self_attn.q_a_proj.")
                new_key = new_key.replace(".attn.q_norm.", ".self_attn.q_a_layernorm.")
                new_key = new_key.replace(".attn.wq_b.", ".self_attn.q_b_proj.")
                new_key = new_key.replace(".attn.wkv.", ".self_attn.kv_proj.")
                new_key = new_key.replace(".attn.kv_norm.", ".self_attn.kv_layernorm.")
                new_key = new_key.replace(".attn.wo_a.", ".self_attn.o_a_proj.")
                new_key = new_key.replace(".attn.wo_b.", ".self_attn.o_b_proj.")
                new_key = new_key.replace(".attn_norm.", ".input_layernorm.")
                new_key = new_key.replace(".ffn_norm.", ".post_attention_layernorm.")
                new_key = new_key.replace(".ffn.gate.", ".mlp.gate.")
                new_key = new_key.replace(".ffn.experts.", ".mlp.experts.")
                new_key = new_key.replace(".ffn.shared_experts.", ".mlp.shared_experts.")
                new_key = new_key.replace(".w1.", ".gate_proj.")
                new_key = new_key.replace(".w2.", ".down_proj.")
                new_key = new_key.replace(".w3.", ".up_proj.")
                if ".hc_" in new_key and new_key.endswith("_fn"):
                    new_key = f"{new_key}.weight"
            renamed[new_key] = value
        if skipped:
            logger.warning(
                "Skipped %d DeepSeek-V4 CSA/HCA/MTP tensors unsupported by "
                "the dense preview backbone.",
                skipped,
            )
        return super().preprocess_weights(renamed)
