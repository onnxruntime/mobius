# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Moonshot Kimi-K3 text decoder."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, KimiK3Config
from mobius.components import Linear, RMSNorm, create_attention_bias
from mobius.models.base import (
    CausalLMModel,
    embedding_for_config,
    linear_class_for_config,
)
from mobius.models.deepseek import DeepSeekMoEGate


class _DepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter([channels, kernel_size])
        self._channels = channels

    def forward(self, op: OpBuilder, x: ir.Value, state: ir.Value):
        weight = op.Unsqueeze(self.weight, [1])
        bias = op.Expand(op.CastLike(0.0, self.weight), [self._channels])
        return op.CausalConvWithState(
            x,
            weight,
            bias,
            state,
            activation="silu",
            _domain="com.microsoft",
            _outputs=2,
        )


class _GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self._eps = eps

    def forward(self, op: OpBuilder, x: ir.Value, gate: ir.Value):
        x_float = op.Cast(x, to=ir.DataType.FLOAT)
        normed = op.RMSNormalization(
            x_float,
            op.Cast(self.weight, to=ir.DataType.FLOAT),
            axis=-1,
            epsilon=self._eps,
            stash_type=1,
        )
        return op.Mul(normed, op.Sigmoid(op.Cast(gate, to=ir.DataType.FLOAT)))


class KimiK3DeltaAttention(nn.Module):
    """KDA with depthwise SiLU convolutions and an FP32 gated-delta state."""

    _QK_NORM_EPS = 1e-6

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        linear_class = linear_class or Linear
        self._heads = config.linear_num_key_heads
        self._head_dim = config.linear_key_head_dim
        assert self._heads is not None and self._head_dim is not None
        self._projection_size = self._heads * self._head_dim
        self._conv_history = config.linear_conv_kernel_dim - 1
        self._lower_bound = config.linear_gate_lower_bound
        assert self._lower_bound is not None

        self.q_proj = linear_class(config.hidden_size, self._projection_size, bias=False)
        self.k_proj = linear_class(config.hidden_size, self._projection_size, bias=False)
        self.v_proj = linear_class(config.hidden_size, self._projection_size, bias=False)
        self.q_conv1d = _DepthwiseConv1d(self._projection_size, config.linear_conv_kernel_dim)
        self.k_conv1d = _DepthwiseConv1d(self._projection_size, config.linear_conv_kernel_dim)
        self.v_conv1d = _DepthwiseConv1d(self._projection_size, config.linear_conv_kernel_dim)
        self.A_log = nn.Parameter([self._heads])
        self.f_a_proj = linear_class(config.hidden_size, self._head_dim, bias=False)
        self.f_b_proj = linear_class(self._head_dim, self._projection_size, bias=False)
        self.dt_bias = nn.Parameter([self._projection_size])
        self.b_proj = linear_class(config.hidden_size, self._heads, bias=False)
        self.g_proj = linear_class(config.hidden_size, self._projection_size, bias=False)
        self.o_norm = _GatedRMSNorm(self._head_dim, config.rms_norm_eps)
        self.o_proj = linear_class(self._projection_size, config.hidden_size, bias=False)

    def _project_conv(
        self,
        op: OpBuilder,
        projection: nn.Module,
        convolution: nn.Module,
        hidden_states: ir.Value,
        state: ir.Value,
        current_mask: ir.Value,
    ):
        # CausalConvWithState consumes (B, channels, sequence).
        projected = op.Transpose(projection(op, hidden_states), perm=[0, 2, 1])
        value, _ = convolution(op, projected, state)

        # Retain valid projected tokens rather than letting right padding evict history.
        conv_input = op.Concat(state, projected, axis=2)
        batch = op.Shape(current_mask, start=0, end=1)
        past_valid = op.Expand(
            op.CastLike(1, current_mask),
            op.Concat(batch, op.Constant(value_ints=[self._conv_history]), axis=0),
        )
        valid = op.Concat(past_valid, current_mask, axis=1)
        positions = op.Range(
            op.Constant(value_int=0),
            op.Gather(op.Shape(valid), op.Constant(value_int=1), axis=0),
            op.Constant(value_int=1),
        )
        positions = op.Expand(op.Unsqueeze(positions, [0]), op.Shape(valid))
        masked_positions = op.Where(
            op.Cast(valid, to=ir.DataType.BOOL),
            positions,
            op.Expand(op.Constant(value_int=-1), op.Shape(valid)),
        )
        selected, _ = op.TopK(
            masked_positions,
            op.Constant(value_ints=[self._conv_history]),
            axis=1,
            largest=1,
            sorted=1,
            _outputs=2,
        )
        selected = op.Gather(
            selected,
            op.Constant(value_ints=list(range(self._conv_history - 1, -1, -1))),
            axis=1,
        )
        selected = op.Expand(
            op.Unsqueeze(selected, [1]),
            op.Concat(
                op.Shape(projected, start=0, end=2),
                op.Constant(value_ints=[self._conv_history]),
                axis=0,
            ),
        )
        present = op.GatherElements(conv_input, selected, axis=2)
        return op.Transpose(value, perm=[0, 2, 1]), present

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        q_conv_state: ir.Value,
        k_conv_state: ir.Value,
        v_conv_state: ir.Value,
        recurrent_state: ir.Value,
    ):
        seq_len = op.Shape(hidden_states, start=1, end=2)
        current_mask = op.Slice(
            attention_mask,
            op.Neg(seq_len),
            op.Constant(value_ints=[9223372036854775807]),
            op.Constant(value_ints=[1]),
        )
        current_mask_3d = op.Unsqueeze(op.CastLike(current_mask, hidden_states), [-1])
        current_mask_float = op.Unsqueeze(
            op.Cast(current_mask, to=ir.DataType.FLOAT),
            [-1],
        )
        x = op.Mul(hidden_states, current_mask_3d)

        q, present_q = self._project_conv(
            op, self.q_proj, self.q_conv1d, x, q_conv_state, current_mask
        )
        k, present_k = self._project_conv(
            op, self.k_proj, self.k_conv1d, x, k_conv_state, current_mask
        )
        v, present_v = self._project_conv(
            op, self.v_proj, self.v_conv1d, x, v_conv_state, current_mask
        )

        head_shape = [0, 0, self._heads, self._head_dim]
        q4 = op.Cast(op.Reshape(q, head_shape), to=ir.DataType.FLOAT)
        k4 = op.Cast(op.Reshape(k, head_shape), to=ir.DataType.FLOAT)
        q4 = op.Div(
            q4,
            op.Sqrt(op.Add(op.ReduceSumSquare(q4, [-1], keepdims=True), self._QK_NORM_EPS)),
        )
        k4 = op.Div(
            k4,
            op.Sqrt(op.Add(op.ReduceSumSquare(k4, [-1], keepdims=True), self._QK_NORM_EPS)),
        )
        q = op.Reshape(q4, [0, 0, self._projection_size])
        k = op.Reshape(k4, [0, 0, self._projection_size])
        v = op.Cast(v, to=ir.DataType.FLOAT)

        # K3's safe gate is -L * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)).
        z = op.Add(
            op.Cast(self.f_b_proj(op, self.f_a_proj(op, x)), to=ir.DataType.FLOAT),
            op.Cast(self.dt_bias, to=ir.DataType.FLOAT),
        )
        a = op.Reshape(
            op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)),
            [1, 1, self._heads, 1],
        )
        decay = op.Reshape(z, head_shape)
        decay = op.Neg(op.Mul(float(self._lower_bound), op.Sigmoid(op.Mul(a, decay))))
        decay = op.Mul(op.Reshape(decay, [0, 0, self._projection_size]), current_mask_float)
        beta = op.Sigmoid(op.Cast(self.b_proj(op, x), to=ir.DataType.FLOAT))
        beta = op.Mul(beta, current_mask_float)

        output, present_recurrent = op.LinearAttention(
            q,
            k,
            v,
            recurrent_state,
            decay,
            beta,
            update_rule="gated_delta",
            scale=self._head_dim**-0.5,
            q_num_heads=self._heads,
            kv_num_heads=self._heads,
            _domain="com.microsoft",
            _outputs=2,
        )
        output = op.Reshape(output, head_shape)
        gate = op.Reshape(self.g_proj(op, x), head_shape)
        output = self.o_norm(op, output, gate)
        output = op.CastLike(op.Reshape(output, [0, 0, self._projection_size]), hidden_states)
        return self.o_proj(op, op.Mul(output, current_mask_3d)), (
            present_q,
            present_k,
            present_v,
            present_recurrent,
        )


class KimiK3MLAAttention(nn.Module):
    """NoPE MLA with Q-LoRA, expanded semantic KV cache, and output gating."""

    _LOW_RANK_NORM_EPS = 1e-6

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        linear_class = linear_class or Linear
        self._heads = config.num_attention_heads
        self._nope = config.qk_nope_head_dim
        self._extra = config.qk_rope_head_dim
        self._value_dim = config.v_head_dim
        self._kv_rank = config.kv_lora_rank
        self._q_rank = config.q_lora_rank
        assert None not in (
            self._nope,
            self._extra,
            self._value_dim,
            self._kv_rank,
            self._q_rank,
        )
        self._qk_dim = self._nope + self._extra
        self.q_a_proj = linear_class(config.hidden_size, self._q_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self._q_rank, eps=self._LOW_RANK_NORM_EPS)
        self.q_b_proj = linear_class(self._q_rank, self._heads * self._qk_dim, bias=False)
        self.kv_a_proj_with_mqa = linear_class(
            config.hidden_size, self._kv_rank + self._extra, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self._kv_rank, eps=self._LOW_RANK_NORM_EPS)
        self.k_b_proj = linear_class(self._kv_rank, self._heads * self._nope, bias=False)
        self.v_b_proj = linear_class(self._kv_rank, self._heads * self._value_dim, bias=False)
        self.g_proj = linear_class(
            config.hidden_size, self._heads * self._value_dim, bias=False
        )
        self.o_proj = linear_class(
            self._heads * self._value_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        past_key_value: tuple[ir.Value, ir.Value],
    ):
        # K3 keeps the historically named "rope" dimensions but applies no RoPE.
        query = self.q_b_proj(op, self.q_a_layernorm(op, self.q_a_proj(op, hidden_states)))
        compressed, key_extra = op.Split(
            self.kv_a_proj_with_mqa(op, hidden_states),
            [self._kv_rank, self._extra],
            axis=-1,
            _outputs=2,
        )
        compressed = self.kv_a_layernorm(op, compressed)
        key_nope = op.Reshape(
            self.k_b_proj(op, compressed),
            [0, 0, self._heads, self._nope],
        )
        value = op.Reshape(
            self.v_b_proj(op, compressed),
            [0, 0, self._heads, self._value_dim],
        )
        key_extra = op.Expand(
            op.Reshape(key_extra, [0, 0, 1, self._extra]),
            [1, 1, self._heads, 1],
        )
        key = op.Reshape(op.Concat(key_nope, key_extra, axis=-1), [0, 0, -1])
        value = op.Reshape(value, [0, 0, -1])
        output, present_key, present_value = op.Attention(
            query,
            key,
            value,
            attention_bias,
            past_key_value[0],
            past_key_value[1],
            q_num_heads=self._heads,
            kv_num_heads=self._heads,
            scale=self._qk_dim**-0.5,
            _outputs=3,
        )
        output = op.Mul(output, op.Sigmoid(self.g_proj(op, hidden_states)))
        return self.o_proj(op, output), (present_key, present_value)


class _SiTUMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        beta: float,
        linear_beta: float,
        linear_class: type | None,
    ):
        super().__init__()
        linear_class = linear_class or Linear
        self.gate_proj = linear_class(input_size, intermediate_size, bias=False)
        self.up_proj = linear_class(input_size, intermediate_size, bias=False)
        self.down_proj = linear_class(intermediate_size, output_size, bias=False)
        self._beta = beta
        self._linear_beta = linear_beta

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        gate = op.Cast(self.gate_proj(op, hidden_states), to=ir.DataType.FLOAT)
        up = op.Cast(self.up_proj(op, hidden_states), to=ir.DataType.FLOAT)
        situ = op.Mul(
            op.Mul(
                self._beta,
                op.Tanh(op.Div(gate, self._beta)),
            ),
            op.Sigmoid(gate),
        )
        up = op.Mul(self._linear_beta, op.Tanh(op.Div(up, self._linear_beta)))
        activated = op.CastLike(op.Mul(situ, up), hidden_states)
        return self.down_proj(op, activated)


class _KimiK3RoutedMoE(nn.Module):
    def __init__(self, config: ArchitectureConfig, linear_class: type | None):
        super().__init__()
        linear_class = linear_class or Linear
        assert config.routed_expert_hidden_size is not None
        assert config.moe_intermediate_size is not None
        assert config.num_local_experts is not None
        self.gate = DeepSeekMoEGate(config)
        self.experts = nn.ModuleList(
            [
                _SiTUMLP(
                    config.routed_expert_hidden_size,
                    config.moe_intermediate_size,
                    config.routed_expert_hidden_size,
                    config.activation_situ_beta,
                    config.activation_situ_linear_beta or 25.0,
                    linear_class,
                )
                for _ in range(config.num_local_experts)
            ]
        )

    def forward(self, op: OpBuilder, routing_input: ir.Value, latent: ir.Value) -> ir.Value:
        routing_weights, selected_experts = self.gate(op, routing_input)
        routed = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(op, latent)
            match = op.Equal(selected_experts, op.Constant(value_int=expert_idx))
            weight = op.ReduceSum(
                op.Mul(routing_weights, op.CastLike(match, routing_weights)),
                [-1],
                keepdims=True,
            )
            weight = op.CastLike(weight, expert_output)
            contribution = op.Mul(expert_output, weight)
            routed = contribution if routed is None else op.Add(routed, contribution)
        assert routed is not None
        return routed


class _KimiK3LatentMoE(nn.Module):
    def __init__(self, config: ArchitectureConfig, linear_class: type | None):
        super().__init__()
        linear_class = linear_class or Linear
        assert config.routed_expert_hidden_size is not None
        assert config.moe_intermediate_size is not None
        self.moe = _KimiK3RoutedMoE(config, linear_class)
        self.routed_down_proj = linear_class(
            config.hidden_size, config.routed_expert_hidden_size, bias=False
        )
        self.routed_norm = RMSNorm(config.routed_expert_hidden_size, eps=config.rms_norm_eps)
        self.routed_up_proj = linear_class(
            config.routed_expert_hidden_size, config.hidden_size, bias=False
        )
        self.shared_experts = _SiTUMLP(
            config.hidden_size,
            config.moe_intermediate_size * 2,
            config.hidden_size,
            config.activation_situ_beta,
            config.activation_situ_linear_beta or 25.0,
            linear_class,
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        latent = self.routed_down_proj(op, hidden_states)
        routed = self.moe(op, hidden_states, latent)
        routed = self.routed_up_proj(op, self.routed_norm(op, routed))
        return op.Add(routed, self.shared_experts(op, hidden_states))


class KimiK3DecoderLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__()
        linear_class = linear_class_for_config(config)
        self.layer_idx = layer_idx
        self._block_size = config.attn_res_block_size
        self._is_kda = config.layer_types[layer_idx] == "kimi_k3_attention"
        self.self_attn = (
            KimiK3DeltaAttention(config, linear_class)
            if self._is_kda
            else KimiK3MLAAttention(config, linear_class)
        )
        if layer_idx == 0:
            self.mlp = _SiTUMLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_size,
                config.activation_situ_beta,
                config.activation_situ_linear_beta or 25.0,
                linear_class,
            )
            self.block_sparse_moe = None
        else:
            self.mlp = None
            self.block_sparse_moe = _KimiK3LatentMoE(config, linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._residual_eps = config.rms_norm_eps
        # The reference's norm scale and projection vector are folded into one
        # float score weight, matching the Kimi-K3 GGUF tensor contract.
        self.attn_res_score = Linear(config.hidden_size, 1, bias=False)
        self.ffn_res_score = Linear(config.hidden_size, 1, bias=False)

    @staticmethod
    def _mix(
        op: OpBuilder,
        prefix_sum: ir.Value,
        bank: list[ir.Value],
        score: nn.Module,
        eps: float,
    ):
        sources = [op.Unsqueeze(value, [2]) for value in (*bank, prefix_sum)]
        values = op.Concat(*sources, axis=2)
        values_float = op.Cast(values, to=ir.DataType.FLOAT)
        rms = op.Sqrt(
            op.Add(
                op.ReduceMean(op.Mul(values_float, values_float), [-1], keepdims=True),
                eps,
            )
        )
        # The score weight contains reference norm.weight * projection.weight.
        scores = op.Squeeze(score(op, op.Div(values_float, rms)), [-1])
        probabilities = op.Unsqueeze(op.Softmax(scores, axis=-1), [-1])
        mixed = op.ReduceSum(op.Mul(probabilities, values_float), [2], keepdims=False)
        return op.CastLike(mixed, prefix_sum)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        bank: list[ir.Value],
        attention_bias: ir.Value,
        attention_mask: ir.Value,
        past_key_value: tuple[ir.Value, ...],
    ):
        prefix_sum = hidden_states
        if bank:
            hidden_states = self._mix(
                op,
                prefix_sum,
                bank,
                self.attn_res_score,
                self._residual_eps,
            )
        if self.layer_idx % self._block_size == 0:
            bank.append(prefix_sum)
            prefix_sum = None

        normed = self.input_layernorm(op, hidden_states)
        if self._is_kda:
            attention_output, present = self.self_attn(
                op, normed, attention_mask, *past_key_value
            )
        else:
            attention_output, present = self.self_attn(
                op, normed, attention_bias, past_key_value
            )
        prefix_sum = (
            attention_output if prefix_sum is None else op.Add(prefix_sum, attention_output)
        )
        hidden_states = self._mix(op, prefix_sum, bank, self.ffn_res_score, self._residual_eps)
        normed = self.post_attention_layernorm(op, hidden_states)
        feed_forward = (
            self.block_sparse_moe(op, normed)
            if self.block_sparse_moe is not None
            else self.mlp(op, normed)
        )
        return op.Add(prefix_sum, feed_forward), bank, present


class KimiK3TextModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [KimiK3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self._residual_eps = config.rms_norm_eps
        self.output_res_score = Linear(config.hidden_size, 1, bias=False)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._dtype = config.dtype

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]],
    ):
        del position_ids
        hidden_states = self.embed_tokens(op, input_ids)
        attention_bias = create_attention_bias(
            op, input_ids, attention_mask, dtype=self._dtype
        )
        bank: list[ir.Value] = []
        present = []
        for layer, past in zip(self.layers, past_key_values):
            hidden_states, bank, layer_present = layer(
                op, hidden_states, bank, attention_bias, attention_mask, past
            )
            present.append(layer_present)
        hidden_states = KimiK3DecoderLayer._mix(
            op,
            hidden_states,
            bank,
            self.output_res_score,
            self._residual_eps,
        )
        return self.norm(op, hidden_states), present


class KimiK3CausalLMModel(CausalLMModel):
    """Dedicated Kimi-K3 KDA/NoPE-MLA/attention-residual latent-MoE decoder."""

    default_task = "kimi-k3-text-generation"
    category = "Mixture of Experts"
    config_class = KimiK3Config

    def __init__(self, config: ArchitectureConfig):
        if config.tie_word_embeddings:
            raise ValueError("Kimi-K3 requires an untied LM head")
        # CausalLMModel owns the quantized/float LM-head construction. Its
        # temporary standard text model cannot resolve K3's custom "situ"
        # activation, so initialize it with an otherwise identical inert act.
        super().__init__(dataclasses.replace(config, hidden_act="silu"))
        self.config = config
        self._replace_text_model(KimiK3TextModel(config))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        residual_parts: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in state_dict.items():
            if key.startswith("language_model."):
                key = key.removeprefix("language_model.")
            if key.startswith(("vision_tower.", "mm_projector.")):
                continue
            if key.endswith(".self_attn.kv_b_proj.weight"):
                prefix = key.removesuffix("kv_b_proj.weight")
                reshaped = value.reshape(
                    self.config.num_attention_heads,
                    self.config.qk_nope_head_dim + self.config.v_head_dim,
                    self.config.kv_lora_rank,
                )
                key_weight, value_weight = torch.split(
                    reshaped,
                    [self.config.qk_nope_head_dim, self.config.v_head_dim],
                    dim=1,
                )
                renamed[prefix + "k_b_proj.weight"] = key_weight.reshape(
                    -1, self.config.kv_lora_rank
                )
                renamed[prefix + "v_b_proj.weight"] = value_weight.reshape(
                    -1, self.config.kv_lora_rank
                )
                continue
            residual_mapping = {
                ".self_attention_res_norm.weight": (".attn_res_score.weight", "norm"),
                ".self_attention_res_proj.weight": (".attn_res_score.weight", "proj"),
                ".mlp_res_norm.weight": (".ffn_res_score.weight", "norm"),
                ".mlp_res_proj.weight": (".ffn_res_score.weight", "proj"),
                "model.output_attn_res_norm.weight": (
                    "model.output_res_score.weight",
                    "norm",
                ),
                "model.output_attn_res_proj.weight": (
                    "model.output_res_score.weight",
                    "proj",
                ),
            }
            matched_residual = False
            for suffix, (target_suffix, part) in residual_mapping.items():
                if key.endswith(suffix):
                    target = key.removesuffix(suffix) + target_suffix
                    residual_parts.setdefault(target, {})[part] = value
                    matched_residual = True
                    break
            if matched_residual:
                continue
            key = key.replace(".w1.", ".gate_proj.")
            key = key.replace(".w2.", ".down_proj.")
            key = key.replace(".w3.", ".up_proj.")
            key = key.replace(
                ".block_sparse_moe.routed_expert_down_proj.",
                ".block_sparse_moe.routed_down_proj.",
            )
            key = key.replace(
                ".block_sparse_moe.routed_expert_norm.",
                ".block_sparse_moe.routed_norm.",
            )
            key = key.replace(
                ".block_sparse_moe.routed_expert_up_proj.",
                ".block_sparse_moe.routed_up_proj.",
            )
            key = key.replace(".block_sparse_moe.gate.", ".block_sparse_moe.moe.gate.")
            key = key.replace(".block_sparse_moe.experts.", ".block_sparse_moe.moe.experts.")
            renamed[key] = value
        for target, parts in residual_parts.items():
            if "norm" not in parts or "proj" not in parts:
                raise ValueError(f"Incomplete Kimi-K3 attention-residual weights for {target}")
            renamed[target] = parts["proj"].float() * parts["norm"].float().unsqueeze(0)
        return super().preprocess_weights(renamed)
