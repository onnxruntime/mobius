# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Recursive Flint decoder blocks with learned sinks and sigmoid-routed experts.

Replicates the published ``flint_sglang`` attention and dense/MoE composition.
Blocks consume hidden states, explicit attention biases, rotary embeddings and
dynamic KV caches, and return hidden states with updated real-token KV caches.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._configs import FlintConfig
from mobius.components import (
    MLP,
    Attention,
    BaseRope,
    Embedding,
    Linear,
    RMSNorm,
    apply_rotary_pos_emb,
    create_attention_bias,
)
from mobius.models.base import CausalLMModel

FLINT_REVISION = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"
FLINT_CUDA_DOMAIN = "com.microsoft.source_compat"


class FlintRotaryEmbedding(BaseRope):
    """Publisher-compatible FP32 rotary tables and arithmetic before the native store."""

    def __init__(self, config: FlintConfig):
        assert config.rope_theta is not None
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        angles = torch.outer(
            torch.arange(config.max_position_embeddings, dtype=torch.float32), inv_freq
        )
        super().__init__(angles.cos().numpy(), angles.sin().numpy())
        self.cos_cache._keep_float32 = True  # type: ignore[attr-defined]
        self.sin_cache._keep_float32 = True  # type: ignore[attr-defined]


class FlintMLP(MLP):
    """Native projections with fused FP32 SwiGLU before its narrow store."""

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        gate = self.gate_proj(op, hidden_states)
        up = self.up_proj(op, hidden_states)
        combined = op.Mul(
            op.Swish(op.Cast(gate, to=ir.DataType.FLOAT)),
            op.Cast(up, to=ir.DataType.FLOAT),
        )
        return self.down_proj(op, op.CastLike(combined, hidden_states))


class FlintResidualRMSNorm(RMSNorm):
    """Normalize the unrounded FP32 residual sum, storing its native residual separately."""

    def forward(
        self, op: OpBuilder, hidden_states: ir.Value, residual: ir.Value | None = None
    ):
        if residual is None:
            return super().forward(op, hidden_states), hidden_states
        combined = op.Add(
            op.Cast(hidden_states, to=ir.DataType.FLOAT),
            op.Cast(residual, to=ir.DataType.FLOAT),
        )
        normalized = op.RMSNormalization(
            combined,
            op.Cast(self.weight, to=ir.DataType.FLOAT),
            epsilon=self.variance_epsilon,
            axis=-1,
        )
        return op.CastLike(normalized, hidden_states), op.CastLike(combined, residual)


class FlintGate(nn.Module):
    """BF16/native router projection followed by FP32 sigmoid and top-k normalization."""

    def __init__(self, config: FlintConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.weight = nn.Parameter([config.num_local_experts, config.hidden_size])
        self.top_k = config.num_experts_per_tok

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        logits = op.MatMul(hidden_states, op.Transpose(self.weight, perm=[1, 0]))
        # ORT's CPU Sigmoid rounds small probabilities to zero around logit -18.
        # exp(-softplus(-x)) retains those probabilities before top-k normalization.
        scores = op.Exp(op.Neg(op.Softplus(op.Neg(op.Cast(logits, to=ir.DataType.FLOAT)))))
        weights, indices = op.TopK(
            scores, op.Constant(value_ints=[self.top_k]), axis=-1, _outputs=2
        )
        denominator = op.Add(op.ReduceSum(weights, [-1], keepdims=True), 1e-20)
        return op.Div(weights, denominator), indices


class FlintMoE(nn.Module):
    """Externally routed SwiGLU experts and an ungated, always-on shared expert.

    Stock MoE kernels recompute softmax routing. Singleton dispatch instead
    executes each already-selected expert with weight one, then applies Flint's
    FP32 sigmoid weights outside the kernel. Graph size scales with top-k, not
    with the expert count, without rounding routing scores to the activation dtype.
    """

    def __init__(self, config: FlintConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        assert config.moe_intermediate_size is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.gate = FlintGate(config)
        self.fc1_experts_weights = nn.Parameter(
            [self.num_experts, 2 * self.intermediate_size, self.hidden_size]
        )
        self.fc2_experts_weights = nn.Parameter(
            [self.num_experts, self.hidden_size, self.intermediate_size]
        )
        shared_size = config.shared_expert_intermediate_size or 0
        self.shared_expert = (
            FlintMLP(dataclasses.replace(config, intermediate_size=shared_size))
            if shared_size
            else None
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        if not ep_capabilities().supports_fused_moe:
            raise NotImplementedError(
                "Flint requires fused MoE support; per-expert graph expansion is disabled"
            )
        flat = op.Reshape(hidden_states, [-1, self.hidden_size])
        weights, indices = self.gate(op, flat)  # (tokens, top_k), FP32 / INT64
        # Source banks store [gate rows | up rows]; the kernel consumes interleaved rows.
        fc1 = op.Reshape(
            op.Transpose(
                op.Reshape(
                    self.fc1_experts_weights,
                    [self.num_experts, 2, self.intermediate_size, self.hidden_size],
                ),
                perm=[0, 2, 1, 3],
            ),
            [self.num_experts, 2 * self.intermediate_size, self.hidden_size],
        )
        combined = None
        for route in range(self.top_k):
            expert_id = op.Gather(indices, route, axis=1)  # (tokens,)
            selection = op.CastLike(
                op.OneHot(
                    expert_id,
                    op.Constant(value_int=self.num_experts),
                    op.Constant(value_floats=[0.0, 1.0]),
                    axis=-1,
                ),
                flat,
            )  # (tokens, experts); singleton normalization is exactly one
            output = op.Cast(
                op.MoE(
                    flat,
                    selection,
                    fc1,
                    None,
                    self.fc2_experts_weights,
                    activation_type="swiglu",
                    activation_alpha=1.0,
                    activation_beta=0.0,
                    swiglu_fusion=1,
                    normalize_routing_weights=1,
                    k=1,
                    _domain="com.microsoft",
                ),
                to=ir.DataType.FLOAT,
            )
            weighted = op.Mul(output, op.Gather(weights, [route], axis=1))
            combined = weighted if combined is None else op.Add(combined, weighted)
        assert combined is not None
        routed = op.Reshape(op.CastLike(combined, hidden_states), op.Shape(hidden_states))
        if self.shared_expert is not None:
            routed = op.Add(routed, self.shared_expert(op, hidden_states))
        return routed


class FlintAttention(Attention):
    """Per-head normalized attention with a learned null-value softmax position."""

    def __init__(self, config: FlintConfig):
        super().__init__(config)
        self.softmax_offset = nn.Parameter([config.num_attention_heads])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
        attention_mask: ir.Value | None = None,
    ):
        query, key, value = self._project_and_normalize_qkv(op, hidden_states)
        if position_embeddings is not None:
            query = op.CastLike(
                apply_rotary_pos_emb(
                    op,
                    op.Cast(query, to=ir.DataType.FLOAT),
                    position_embeddings,
                    self.num_attention_heads,
                ),
                query,
            )
            key = op.CastLike(
                apply_rotary_pos_emb(
                    op,
                    op.Cast(key, to=ir.DataType.FLOAT),
                    position_embeddings,
                    self.num_key_value_heads,
                ),
                key,
            )
        query = op.Transpose(
            op.Reshape(query, [0, 0, self.num_attention_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )
        key = op.Transpose(
            op.Reshape(key, [0, 0, self.num_key_value_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )
        value = op.Transpose(
            op.Reshape(value, [0, 0, self.num_key_value_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )
        if past_key_value is not None:
            key = op.Concat(past_key_value[0], key, axis=2)
            value = op.Concat(past_key_value[1], value, axis=2)
        present = (key, value)
        mask_shape = op.Concat(op.Shape(query, end=3), op.Shape(key, start=2, end=3), axis=0)
        mask = op.Expand(op.Cast(attention_bias, to=ir.DataType.FLOAT), mask_shape)
        sink_shape = op.Concat(op.Shape(query, end=3), [1], axis=0)
        sink = op.Expand(
            op.Reshape(
                op.Cast(self.softmax_offset, to=ir.DataType.FLOAT),
                [1, self.num_attention_heads, 1, 1],
            ),
            sink_shape,
        )
        # Align the bias stride for CUDA memory-efficient attention. Only the
        # first artificial position is a sink; the rest have zero probability.
        padding = op.Sub([4], op.Mod(op.Shape(key, start=2, end=3), [4]))
        sink = op.Concat(
            sink,
            op.Expand(
                op.Constant(value_float=float("-inf")),
                op.Concat(op.Shape(query, end=3), op.Sub(padding, [1]), axis=0),
            ),
            axis=-1,
        )
        # A zero key contributes only its sink bias; a zero value discards its mass.
        zero = op.Expand(
            op.CastLike(0.0, key),
            op.Concat(op.Shape(key, end=2), padding, [self.head_dim], axis=0),
        )
        key = op.Concat(key, zero, axis=2)
        value = op.Concat(value, zero, axis=2)
        output = op.Attention(
            op.Reshape(op.Transpose(query, perm=[0, 2, 1, 3]), [0, 0, -1]),
            op.Reshape(op.Transpose(key, perm=[0, 2, 1, 3]), [0, 0, -1]),
            op.Reshape(op.Transpose(value, perm=[0, 2, 1, 3]), [0, 0, -1]),
            # CUDA's native kernel requires a bias with the Q/K/V dtype. All
            # finite entries are checkpoint sinks, already stored at that dtype.
            op.CastLike(op.Concat(mask, sink, axis=-1), query),
            q_num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            is_causal=0,
            scale=self.scaling,
            softmax_precision=int(ir.DataType.FLOAT),
        )
        return self.o_proj(op, output), present


class FlintCUDARMSNorm(RMSNorm):
    """Source-compatible BF16 CUDA reduction, with a standard-ONNX debug route."""

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        if ep_capabilities().name != "cuda":
            return super().forward(op, hidden_states)
        output = op.RMSNorm(
            hidden_states,
            self.weight,
            epsilon=self.variance_epsilon,
            _domain=FLINT_CUDA_DOMAIN,
            _version=1,
        )
        output.type = ir.TensorType(ir.DataType.BFLOAT16)
        output.shape = hidden_states.shape
        return output


class FlintCUDAResidualRMSNorm(FlintResidualRMSNorm):
    """Native fused normalization of the unrounded residual sum."""

    def forward(self, op, hidden_states, residual=None):
        if ep_capabilities().name != "cuda":
            return super().forward(op, hidden_states, residual)
        if residual is None:
            normalized = op.RMSNorm(
                hidden_states,
                self.weight,
                epsilon=self.variance_epsilon,
                _domain=FLINT_CUDA_DOMAIN,
                _version=1,
            )
            normalized.type = ir.TensorType(ir.DataType.BFLOAT16)
            normalized.shape = hidden_states.shape
            return normalized, hidden_states
        normalized, stored_residual = op.SkipRMSNorm(
            hidden_states,
            residual,
            self.weight,
            epsilon=self.variance_epsilon,
            _domain=FLINT_CUDA_DOMAIN,
            _version=1,
            _outputs=2,
        )
        normalized.type = ir.TensorType(ir.DataType.BFLOAT16)
        normalized.shape = hidden_states.shape
        stored_residual.type = ir.TensorType(ir.DataType.BFLOAT16)
        stored_residual.shape = residual.shape
        return normalized, stored_residual


class FlintCUDAAttention(FlintAttention):
    """Pinned FA3 BF16 sink attention for a single unpadded Hopper sequence."""

    def __init__(self, config: FlintConfig, layer_index: int):
        super().__init__(config)
        self.q_norm = FlintCUDARMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = FlintCUDARMSNorm(config.head_dim, eps=config.rms_norm_eps)
        layer_types = config.layer_types or ["full_attention"] * config.num_hidden_layers
        self.window_left = -1
        if layer_types[layer_index] == "sliding_attention":
            assert config.sliding_window is not None
            self.window_left = config.sliding_window - 1

    def forward(
        self,
        op,
        hidden_states,
        attention_bias,
        position_embeddings=None,
        past_key_value=None,
        attention_mask=None,
    ):
        if ep_capabilities().name != "cuda":
            return super().forward(
                op, hidden_states, attention_bias, position_embeddings, past_key_value
            )
        if attention_mask is None:
            raise NotImplementedError("Flint CUDA requires a dynamic-cache attention mask")
        query, key, value = self._project_and_normalize_qkv(op, hidden_states)
        if position_embeddings is not None:
            query = op.CastLike(
                apply_rotary_pos_emb(
                    op,
                    op.Cast(query, to=ir.DataType.FLOAT),
                    position_embeddings,
                    self.num_attention_heads,
                ),
                query,
            )
            key = op.CastLike(
                apply_rotary_pos_emb(
                    op,
                    op.Cast(key, to=ir.DataType.FLOAT),
                    position_embeddings,
                    self.num_key_value_heads,
                ),
                key,
            )
        key = op.Transpose(
            op.Reshape(key, [0, 0, self.num_key_value_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )
        value = op.Transpose(
            op.Reshape(value, [0, 0, self.num_key_value_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )
        if past_key_value is not None:
            key = op.Concat(past_key_value[0], key, axis=2)
            value = op.Concat(past_key_value[1], value, axis=2)
        output = op.SinkAttention(
            query,
            op.Reshape(op.Transpose(key, perm=[0, 2, 1, 3]), [0, 0, -1]),
            op.Reshape(op.Transpose(value, perm=[0, 2, 1, 3]), [0, 0, -1]),
            self.softmax_offset,
            attention_mask,
            q_num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            window_left=self.window_left,
            scale=self.scaling,
            _domain=FLINT_CUDA_DOMAIN,
            _version=1,
        )
        output.type = ir.TensorType(ir.DataType.BFLOAT16)
        output.shape = query.shape
        return self.o_proj(op, output), (key, value)


class FlintDecoderLayer(nn.Module):
    """Pre-norm attention and dense or sparse SwiGLU with two residual additions."""

    def __init__(self, config: FlintConfig, layer_index: int):
        super().__init__()
        self.self_attn = FlintAttention(config)
        self.input_layernorm = FlintResidualRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = FlintResidualRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = (
            FlintMLP(config) if layer_index in config.mlp_only_layers else FlintMoE(config)
        )

    def forward(
        self, op, hidden_states, attention_bias, position_embeddings=None, past_key_value=None
    ):
        feedforward, residual, present = self.forward_with_residual(
            op, hidden_states, None, attention_bias, position_embeddings, past_key_value
        )
        return op.Add(residual, feedforward), present

    def forward_with_residual(
        self,
        op,
        hidden_states,
        residual,
        attention_bias,
        position_embeddings=None,
        past_key_value=None,
        attention_mask=None,
    ):
        normalized, residual = self.input_layernorm(op, hidden_states, residual)
        attention, present = self.self_attn(
            op,
            normalized,
            attention_bias,
            position_embeddings,
            past_key_value,
            **({"attention_mask": attention_mask} if attention_mask is not None else {}),
        )
        normalized, residual = self.post_attention_layernorm(op, attention, residual)
        feedforward = self.mlp(op, normalized)
        return feedforward, residual, present


class FlintResidualDecoderLayer(FlintDecoderLayer):
    """A stacked decoder layer whose FFN output and residual remain separate."""

    def forward(
        self,
        op,
        hidden_states,
        residual,
        attention_bias,
        position_embeddings=None,
        past_key_value=None,
        attention_mask=None,
    ):
        return self.forward_with_residual(
            op,
            hidden_states,
            residual,
            attention_bias,
            position_embeddings,
            past_key_value,
            attention_mask,
        )


class FlintCUDAResidualDecoderLayer(FlintResidualDecoderLayer):
    """Residual-pair decoder using the source-compatible CUDA extension."""

    def __init__(self, config: FlintConfig, layer_index: int):
        super().__init__(config, layer_index)
        self.self_attn = FlintCUDAAttention(config, layer_index)
        self.input_layernorm = FlintCUDAResidualRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = FlintCUDAResidualRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class FlintTextModel(nn.Module):
    """Published Flint backbone, carrying the unmaterialized hidden/residual pair."""

    def __init__(self, config: FlintConfig):
        super().__init__()
        self.config = config
        self.layer_types = config.layer_types or ["full_attention"] * config.num_hidden_layers
        indices = config.output_layer_indices or []
        if len(indices) != len(set(indices)) or any(
            index < 0 or index >= config.num_hidden_layers for index in indices
        ):
            raise ValueError("output_layer_indices must be distinct valid Flint layer indices")
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        native_cuda = config.dtype == ir.DataType.BFLOAT16 and config.head_dim == 128
        layer_class = (
            FlintCUDAResidualDecoderLayer if native_cuda else FlintResidualDecoderLayer
        )
        norm_class = FlintCUDAResidualRMSNorm if native_cuda else FlintResidualRMSNorm
        self.layers = nn.ModuleList(
            [layer_class(config, index) for index in range(config.num_hidden_layers)]
        )
        self.norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = FlintRotaryEmbedding(config)

    def forward(self, op, input_ids, attention_mask, position_ids, past_key_values=None):
        hidden_states = self.embed_tokens(op, input_ids)
        embeddings = self.rotary_emb(op, position_ids)
        biases = {
            "full_attention": create_attention_bias(op, input_ids, attention_mask),
            "sliding_attention": create_attention_bias(
                op, input_ids, attention_mask, sliding_window=self.config.sliding_window
            ),
        }
        pasts = past_key_values or [None] * len(self.layers)
        if len(pasts) != len(self.layers):
            raise ValueError("Flint requires one KV-cache pair per decoder layer")
        present_key_values = []
        intermediates = {}
        residual = None
        for index, (layer, past) in enumerate(zip(self.layers, pasts, strict=True)):
            # Do not round hidden + residual before the next layer's fused input norm.
            hidden_states, residual, present = layer(
                op,
                hidden_states,
                residual,
                biases[self.layer_types[index]],
                None if index in (self.config.no_rope_layers or []) else embeddings,
                past,
                attention_mask=attention_mask,
            )
            present_key_values.append(present)
            if index in (self.config.output_layer_indices or []):
                intermediates[index] = op.Add(hidden_states, residual)
        hidden_states, _ = self.norm(op, hidden_states, residual)
        hidden_states.type = ir.TensorType(self.config.dtype)
        if input_ids.shape is not None:
            hidden_states.shape = ir.Shape([*input_ids.shape, self.config.hidden_size])
        if self.config.output_layer_indices:
            return (
                hidden_states,
                present_key_values,
                [intermediates[index] for index in self.config.output_layer_indices],
            )
        return hidden_states, present_key_values


class FlintLMHead(Linear):
    """The publisher's FP32 readout, irrespective of decoder activation precision."""

    def __init__(self, config: FlintConfig):
        super().__init__(config.hidden_size, config.vocab_size, bias=False)
        self.weight._keep_float32 = True  # type: ignore[attr-defined]

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        logits = super().forward(op, op.Cast(hidden_states, to=ir.DataType.FLOAT))
        if hidden_states.shape is not None:
            assert self.weight.shape is not None
            logits.shape = ir.Shape([*hidden_states.shape[:-1], self.weight.shape[0]])
        return logits


class FlintCausalLMModel(CausalLMModel):
    """RecursiveSI Flint: mixed full/sliding attention, sinks and sigmoid-routed MoE.

    Replicates the published ``flint_sglang`` decoder with paired residual state
    and an FP32 LM head. Inputs are token/position IDs, attention mask and dynamic
    KV caches; outputs are FP32 logits and the updated real-token KV caches.

    Deployment uses BF16 CUDA on Hopper with head dimension 128 and one
    unpadded sequence. Install ORT's standalone source-compatible CUDA library
    under ``lib/`` in the model package; its requirement is propagated to the
    ORT GenAI config. Other configurations are graph/reference-debug routes,
    not a claim of publisher-compatible execution.
    """

    category = "Mixture of Experts"
    config_class = FlintConfig
    custom_op_libraries: ClassVar[dict[str, tuple[str, ...]]] = {
        FLINT_CUDA_DOMAIN: ("lib/libonnxruntime_source_compat_cuda.so",),
    }

    def __init__(self, config: FlintConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = FlintTextModel(config)  # type: ignore[assignment]
        self.lm_head = FlintLMHead(config)  # type: ignore[assignment]
