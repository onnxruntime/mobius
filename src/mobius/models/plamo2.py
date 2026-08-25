# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PLaMo2 alternating attention and selective-state-space language model."""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Plamo2Config
from mobius.components import (
    FusedGateUpMLP,
    Linear,
    RMSNorm,
    TiedQuantizedLMHead,
    apply_rotary_pos_emb,
    create_attention_bias,
    initialize_rope,
)
from mobius.models.base import (
    effective_tie_word_embeddings,
    embedding_for_config,
    linear_class_for_config,
)


class Plamo2Attention(nn.Module):
    """PLaMo2 fused-QKV grouped-query attention with per-head Q/K RMSNorm."""

    def __init__(self, config: Plamo2Config):
        super().__init__()
        linear_class = linear_class_for_config(config) or Linear
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_proj = linear_class(
            config.hidden_size,
            self.q_dim + 2 * self.kv_dim,
            bias=False,
        )
        self.q_weight = nn.Parameter([self.num_heads, self.head_dim])
        self.k_weight = nn.Parameter([self.num_kv_heads, self.head_dim])
        self.o_proj = linear_class(self.q_dim, config.hidden_size, bias=False)
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
        self.window_size = config.attention_window_size

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ...],
        past_key: ir.Value,
        past_value: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        # The serialized fused projection is ordered Q, K, V.
        qkv = self.qkv_proj(op, hidden_states)
        query, key, value = op.Split(
            qkv,
            [self.q_dim, self.kv_dim, self.kv_dim],
            axis=-1,
            _outputs=3,
        )
        query = op.Reshape(query, [0, 0, self.num_heads, self.head_dim])
        key = op.Reshape(key, [0, 0, self.num_kv_heads, self.head_dim])
        query = op.RMSNormalization(query, self.q_weight, axis=-1, epsilon=self.eps)
        key = op.RMSNormalization(key, self.k_weight, axis=-1, epsilon=self.eps)
        query = op.Reshape(query, [0, 0, self.q_dim])
        key = op.Reshape(key, [0, 0, self.kv_dim])
        query = apply_rotary_pos_emb(
            op,
            query,
            position_embeddings,
            num_heads=self.num_heads,
            rotary_embedding_dim=0,
            interleaved=False,
        )
        key = apply_rotary_pos_emb(
            op,
            key,
            position_embeddings,
            num_heads=self.num_kv_heads,
            rotary_embedding_dim=0,
            interleaved=False,
        )
        output, present_key, present_value = op.Attention(
            query,
            key,
            value,
            attention_bias,
            past_key,
            past_value,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_kv_heads,
            scale=self.scale,
            is_causal=0,
            _outputs=3,
        )
        present_key = op.Slice(
            present_key,
            [-self.window_size],
            [9223372036854775807],
            [2],
        )
        present_value = op.Slice(
            present_value,
            [-self.window_size],
            [9223372036854775807],
            [2],
        )
        return self.o_proj(op, output), present_key, present_value


class Plamo2Mamba(nn.Module):
    """PLaMo2 multi-head selective scan with projected and normalized B/C/dt."""

    def __init__(self, config: Plamo2Config):
        super().__init__()
        linear_class = linear_class_for_config(config) or Linear
        self.inner_size = config.mamba_inner_size
        self.num_heads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.state_size = config.mamba_d_state
        self.dt_rank = config.mamba_dt_rank
        self.conv_kernel = config.mamba_d_conv
        self.eps = config.rms_norm_eps

        self.in_proj = linear_class(config.hidden_size, 2 * self.inner_size, bias=False)
        self.conv1d = _Plamo2CausalConv(self.inner_size, self.conv_kernel)
        self.bcdt_proj = linear_class(
            self.inner_size,
            2 * self.state_size + self.dt_rank,
            bias=False,
        )
        # dt participates in Softplus and remains float even for quantized GGUF imports.
        self.dt_proj = Linear(self.dt_rank, self.num_heads, bias=False)
        self.dt_bias = nn.Parameter([self.num_heads])
        self.A_log = nn.Parameter([self.num_heads])
        self.D = nn.Parameter([self.num_heads])
        self.dt_norm_weight = nn.Parameter([self.dt_rank])
        self.B_norm_weight = nn.Parameter([self.state_size])
        self.C_norm_weight = nn.Parameter([self.state_size])
        self.out_proj = linear_class(self.inner_size, config.hidden_size, bias=False)

    def _repeat_for_heads(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        value = op.Unsqueeze(value, [2])  # (B, T, 1, state)
        value = op.Tile(value, [1, 1, self.num_heads, 1])
        return op.Reshape(value, [0, 0, self.num_heads * self.state_size])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
        padding_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        # Each head stores [gate_h | x_h], rather than two global contiguous halves.
        projected = op.Reshape(
            self.in_proj(op, hidden_states),
            [0, 0, self.num_heads, 2 * self.head_dim],
        )
        gate, x = op.Split(
            projected,
            [self.head_dim, self.head_dim],
            axis=-1,
            _outputs=2,
        )
        gate = op.Reshape(gate, [0, 0, self.inner_size])
        x = op.Reshape(x, [0, 0, self.inner_size])
        x = op.Mul(x, op.CastLike(padding_mask, x))
        x_t = op.Transpose(x, perm=[0, 2, 1])
        x_t, present_conv = self.conv1d(op, x_t, conv_state)
        x = op.Transpose(x_t, perm=[0, 2, 1])
        x = op.Mul(x, op.CastLike(padding_mask, x))

        # The learned projection is serialized in [B | C | dt] order.
        b_mat, c_mat, dt_raw = op.Split(
            self.bcdt_proj(op, x),
            [self.state_size, self.state_size, self.dt_rank],
            axis=-1,
            _outputs=3,
        )
        b_mat = op.RMSNormalization(b_mat, self.B_norm_weight, axis=-1, epsilon=self.eps)
        c_mat = op.RMSNormalization(c_mat, self.C_norm_weight, axis=-1, epsilon=self.eps)
        dt_raw = op.RMSNormalization(dt_raw, self.dt_norm_weight, axis=-1, epsilon=self.eps)

        dt = op.Softplus(
            op.Add(
                op.Cast(self.dt_proj(op, dt_raw), to=ir.DataType.FLOAT),
                op.Cast(self.dt_bias, to=ir.DataType.FLOAT),
            )
        )
        dt = op.Mul(dt, op.CastLike(padding_mask, dt))
        # Match the reference's float32 discretization at runtime rather than
        # baking a rounded -exp(A_log) into imported weights.
        a = op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)))
        decay = op.Mul(dt, a)
        x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
        x_heads = op.Reshape(x_f32, [0, 0, self.num_heads, self.head_dim])
        value = op.Reshape(
            op.Mul(op.Unsqueeze(dt, [-1]), x_heads),
            [0, 0, self.inner_size],
        )
        # The public ABI follows the reference [B, H, d_head, d_state].
        internal_state = op.Transpose(
            op.Cast(ssm_state, to=ir.DataType.FLOAT),
            perm=[0, 1, 3, 2],
        )
        output, present_state = op.LinearAttention(
            self._repeat_for_heads(op, op.Cast(c_mat, to=ir.DataType.FLOAT)),
            self._repeat_for_heads(op, op.Cast(b_mat, to=ir.DataType.FLOAT)),
            value,
            internal_state,
            decay,
            scale=1.0,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            update_rule="gated",
            _domain="com.microsoft",
            _outputs=2,
        )
        d = op.Reshape(op.Cast(self.D, to=ir.DataType.FLOAT), [1, 1, self.num_heads, 1])
        d_skip = op.Reshape(op.Mul(d, x_heads), [0, 0, self.inner_size])
        output = op.Add(output, d_skip)
        output = op.Mul(output, op.Swish(op.Cast(gate, to=ir.DataType.FLOAT)))
        output = self.out_proj(op, op.CastLike(output, hidden_states))
        present_state = op.Transpose(present_state, perm=[0, 1, 3, 2])
        return output, present_conv, present_state


class _Plamo2CausalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self.channels = channels

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        bias = op.Expand(
            op.CastLike(op.Constant(value_float=0.0), self.weight),
            [self.channels],
        )
        return op.CausalConvWithState(
            hidden_states,
            self.weight,
            bias,
            conv_state,
            activation="silu",
            _domain="com.microsoft",
            _outputs=2,
        )


class Plamo2DecoderLayer(nn.Module):
    """One exact PLaMo2 sandwich-normalized attention or Mamba layer."""

    def __init__(self, config: Plamo2Config, layer_type: str):
        super().__init__()
        self.pre_mixer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mixer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = Plamo2Mamba(config) if layer_type == "mamba" else Plamo2Attention(config)
        self.mlp = FusedGateUpMLP(config, linear_class=linear_class_for_config(config))
        self.layer_type = layer_type

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ...],
        padding_mask: ir.Value,
        state_a: ir.Value,
        state_b: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        residual = hidden_states
        normalized = self.pre_mixer_norm(op, hidden_states)
        if self.layer_type == "mamba":
            mixed, present_a, present_b = self.mixer(
                op, normalized, state_a, state_b, padding_mask
            )
        else:
            mixed, present_a, present_b = self.mixer(
                op,
                normalized,
                attention_bias,
                position_embeddings,
                state_a,
                state_b,
            )
        hidden_states = op.Add(residual, self.post_mixer_norm(op, mixed))
        residual = hidden_states
        hidden_states = self.mlp(op, self.pre_mlp_norm(op, hidden_states))
        hidden_states = op.Add(residual, self.post_mlp_norm(op, hidden_states))
        return hidden_states, present_a, present_b


class Plamo2Model(nn.Module):
    """PLaMo2 text backbone matching the pinned ``Plamo2Model`` layout."""

    def __init__(self, config: Plamo2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [Plamo2DecoderLayer(config, layer_type) for layer_type in config.layer_types or ()]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        position_ids: ir.Value,
        attention_mask: ir.Value,
        past_states: tuple[tuple[ir.Value, ir.Value], ...],
    ) -> tuple[ir.Value, tuple[tuple[ir.Value, ir.Value], ...]]:
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window=self.config.attention_window_size + 1,
            dtype=self.config.dtype,
        )
        current_length = op.Shape(input_ids, start=1, end=2)
        padding_mask = op.Unsqueeze(
            op.Slice(
                attention_mask,
                op.Neg(current_length),
                [9223372036854775807],
                [1],
            ),
            [-1],
        )
        present_states: list[tuple[ir.Value, ir.Value]] = []
        for layer, state in zip(self.layers, past_states):
            hidden_states, present_a, present_b = layer(
                op,
                hidden_states,
                attention_bias,
                position_embeddings,
                padding_mask,
                state[0],
                state[1],
            )
            present_states.append((present_a, present_b))
        return self.norm(op, hidden_states), tuple(present_states)


class Plamo2ForCausalLM(nn.Module):
    """Dedicated PLaMo2 causal LM with alternating KV and recurrent states."""

    default_task: str = "plamo2-text-generation"
    category: str = "Hybrid SSM+Attention"
    config_class: type = Plamo2Config

    def __init__(self, config: Plamo2Config):
        super().__init__()
        self.config = config
        self.model = Plamo2Model(config)
        quantization = getattr(config, "quantization", None)
        quantized_embedding = quantization is not None and bool(
            getattr(quantization, "quantize_embeddings", False)
        )
        quantized_head = quantization is not None and bool(
            getattr(quantization, "quantize_lm_head", False)
        )
        tie = effective_tie_word_embeddings(config)
        if tie and quantized_embedding != quantized_head:
            raise ValueError(
                "PLaMo2 tied embeddings require token embedding and LM head "
                "quantization to be enabled together"
            )
        if tie and quantized_embedding and quantized_head:
            self.lm_head = TiedQuantizedLMHead(
                self.model.embed_tokens,
                config.hidden_size,
                config.vocab_size,
            )
        else:
            head_class = linear_class_for_config(config) if quantized_head else None
            self.lm_head = (head_class or Linear)(
                config.hidden_size, config.vocab_size, bias=False
            )
        if tie and not quantized_embedding:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        position_ids: ir.Value,
        attention_mask: ir.Value,
        past_states: tuple[tuple[ir.Value, ir.Value], ...],
    ) -> tuple[ir.Value, tuple[tuple[ir.Value, ir.Value], ...]]:
        hidden_states, present_states = self.model(
            op, input_ids, position_ids, attention_mask, past_states
        )
        return self.lm_head(op, hidden_states), present_states

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert official and GGUF offset norms and Mamba decay names."""
        result: dict[str, torch.Tensor] = {}
        norm_offsets = {
            ".pre_mixer_norm.weight": 1.0,
            ".post_mixer_norm.weight": 1.0 / 5.0,
            ".pre_mlp_norm.weight": 1.0,
            ".post_mlp_norm.weight": 1.0 / (5.0**1.5),
        }
        norms_are_folded = bool(getattr(self.config, "_plamo2_norms_are_folded", False))
        tied_embeddings = effective_tie_word_embeddings(self.config)
        for name, value in state_dict.items():
            name = name.replace("model.layers.layers.", "model.layers.")
            if name == "model.embed_tokens.weight" and tied_embeddings:
                result[name] = value
                # onnxscript materializes the shared Parameter under both use
                # sites, while the official checkpoint stores only the embedding.
                result["lm_head.weight"] = value
                continue
            if name == "lm_head.weight" and tied_embeddings:
                continue
            if name == "model.norm.weight":
                result[name] = value if norms_are_folded else value + 1.0
                continue
            offset = next(
                (amount for suffix, amount in norm_offsets.items() if name.endswith(suffix)),
                None,
            )
            if offset is not None:
                result[name] = value if norms_are_folded else value + offset
            elif name.endswith(".mixer.A"):
                # llama.cpp serializes PLaMo2's A_log as -exp(A_log).
                result[name.removesuffix("A") + "A_log"] = torch.log(-value)
            else:
                result[name] = value
        return result
