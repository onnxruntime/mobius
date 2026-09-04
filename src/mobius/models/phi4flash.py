# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Phi-4 Flash Reasoning's SambaY causal language model.

Replicates the pinned remote-code ``Phi4FlashForCausalLM`` implementation:

```mermaid
flowchart LR
    E[Token embedding] --> P[Mamba / local differential attention]
    P --> M16[Layer 16 Mamba]
    M16 -->|raw SSM memory| CM[Cross-Mamba layers]
    M16 --> G17[Layer 17 global differential attention]
    G17 -->|shared global K/V| CA[Cross differential-attention layers]
    CM --> N[Final LayerNorm]
    CA --> N --> H[Tied LM head]
```

The model has no rotary position embedding. Its only persistent cache has
18 slots: nine Mamba convolution/SSM pairs, eight local differential-attention
KV pairs, and the global layer-17 KV pair. The later YOCO cross layers consume
transient layer-16 memory and layer-17 KV within the same graph invocation.
"""

from __future__ import annotations

import torch
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import Phi4FlashConfig
from mobius.components import (
    DifferentialGQAAttention,
    Embedding,
    FloatSwiGLU,
    GatedMemoryMixer,
    LayerNorm,
    Linear,
    StatefulMambaBlock,
)
from mobius.components._common import INT64_MAX

class _Phi4FlashMLP(nn.Module):
    """SambaY SwiGLU MLP with checkpoint-native fused ``fc1`` parameters."""

    def __init__(self, config: Phi4FlashConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = FloatSwiGLU()
        self._intermediate_size = config.intermediate_size

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        gate, value = op.Split(
            self.fc1(op, hidden_states),
            [self._intermediate_size, self._intermediate_size],
            axis=-1,
            _outputs=2,
        )
        return self.fc2(op, self.activation(op, gate, value))


class _Phi4FlashAttention(nn.Module):
    """Fused-QKV differential GQA with optional YOCO shared KV input."""

    def __init__(self, config: Phi4FlashConfig, layer_idx: int, *, cross: bool = False):
        super().__init__()
        self._cross = cross
        self._num_heads = config.num_attention_heads
        self._num_kv_heads = config.num_key_value_heads
        self._head_dim = config.head_dim
        self._query_width = self._num_heads * self._head_dim
        self._kv_width = self._num_kv_heads * self._head_dim
        projection_width = self._query_width if cross else self._query_width + 2 * self._kv_width

        # The remote checkpoint spells the fused projection Wqkv (capital W).
        self.Wqkv = Linear(config.hidden_size, projection_width, bias=True)
        self.out_proj = Linear(self._query_width, config.hidden_size, bias=True)
        self.inner_cross_attn = DifferentialGQAAttention(
            num_attention_heads=self._num_heads,
            num_key_value_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            depth=layer_idx,
            eps=config.layer_norm_eps,
            local_window_size=(
                config.local_attention_window if not cross and layer_idx < config.num_hidden_layers // 2 else None
            ),
        )

    def _to_query_heads(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        return op.Reshape(value, [0, 0, self._num_heads, self._head_dim])

    def _to_kv_cache(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        value = op.Reshape(value, [0, 0, self._num_kv_heads, self._head_dim])
        return op.Transpose(value, perm=[0, 2, 1, 3])  # (B, KV, T, D)

    @staticmethod
    def _current_and_past_shared_kv(
        op: OpBuilder,
        shared_key_value: tuple[ir.Value, ir.Value],
        hidden_states: ir.Value,
    ) -> tuple[ir.Value, ir.Value, tuple[ir.Value, ir.Value] | None]:
        """Split layer-17 chronological KV into this chunk and its causal past."""
        total_length = op.Shape(shared_key_value[0], start=2, end=3)
        current_length = op.Shape(hidden_states, start=1, end=2)
        past_length = op.Sub(total_length, current_length)
        current_key = op.Slice(
            shared_key_value[0], starts=past_length, ends=total_length, axes=[2]
        )
        current_value = op.Slice(
            shared_key_value[1], starts=past_length, ends=total_length, axes=[2]
        )
        past_key = op.Slice(shared_key_value[0], starts=[0], ends=past_length, axes=[2])
        past_value = op.Slice(shared_key_value[1], starts=[0], ends=past_length, axes=[2])
        return (
            op.Transpose(current_key, perm=[0, 2, 1, 3]),
            op.Transpose(current_value, perm=[0, 2, 1, 3]),
            (op.Transpose(past_key, perm=[0, 2, 1, 3]), op.Transpose(past_value, perm=[0, 2, 1, 3])),
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        past_key_value: tuple[ir.Value, ir.Value] | None = None,
        shared_key_value: tuple[ir.Value, ir.Value] | None = None,
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value] | None]:
        """Return layer output and the full self-attention KV cache when owned."""
        projected = self.Wqkv(op, hidden_states)
        if self._cross:
            if shared_key_value is None:
                raise ValueError("Phi4Flash cross differential attention requires layer-17 shared KV")
            query = self._to_query_heads(op, projected)
            key, value, attention_past_key_value = self._current_and_past_shared_kv(
                op, shared_key_value, hidden_states
            )
            present_key_value = None
        else:
            query_flat, key_flat, value_flat = op.Split(
                projected,
                [self._query_width, self._kv_width, self._kv_width],
                axis=-1,
                _outputs=3,
            )
            query = self._to_query_heads(op, query_flat)
            current_key = self._to_kv_cache(op, key_flat)
            current_value = self._to_kv_cache(op, value_flat)
            if past_key_value is None:
                key_cache, value_cache = current_key, current_value
            else:
                key_cache = op.Concat(past_key_value[0], current_key, axis=2)
                value_cache = op.Concat(past_key_value[1], current_value, axis=2)
            present_key_value = (key_cache, value_cache)
            key = op.Transpose(current_key, perm=[0, 2, 1, 3])
            value = op.Transpose(current_value, perm=[0, 2, 1, 3])
            attention_past_key_value = (
                (op.Transpose(past_key_value[0], perm=[0, 2, 1, 3]), op.Transpose(past_key_value[1], perm=[0, 2, 1, 3]))
                if past_key_value is not None
                else None
            )

        # The reference unconditionally invokes FlashAttention with bf16 Q/K/V.
        # Cast the result back before the projection so f32 graph construction
        # remains type-valid while bfloat16 exports preserve reference behavior.
        query = op.Cast(query, to=ir.DataType.BFLOAT16)
        key = op.Cast(key, to=ir.DataType.BFLOAT16)
        value = op.Cast(value, to=ir.DataType.BFLOAT16)
        attention = self.inner_cross_attn(
            op,
            query,
            key,
            value,
            attention_mask,
            attention_past_key_value,
        )
        attention = op.Reshape(attention, [0, 0, self._query_width])
        attention = op.CastLike(attention, hidden_states)
        return self.out_proj(op, attention), present_key_value


class _Phi4FlashDecoderLayer(nn.Module):
    """One SambaY decoder layer, including its source-defined residual ordering."""

    def __init__(self, config: Phi4FlashConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = (config.layer_types or [])[layer_idx]
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _Phi4FlashMLP(config)

        d_inner = config.hidden_size * config.mamba_expand
        if self.layer_type in {"mamba", "shared_memory_mamba"}:
            self.attn = StatefulMambaBlock(
                config.hidden_size,
                d_inner,
                d_state=config.mamba_d_state,
                dt_rank=config.mamba_dt_rank,
                conv_kernel=config.mamba_d_conv,
                # SambaY externally preserves K raw values, not the conventional
                # K-1 convolution carry.
                conv_state_width=config.mamba_d_conv,
                conv_bias=config.mamba_conv_bias,
                proj_bias=config.mamba_proj_bias,
            )
        elif self.layer_type == "cross_mamba":
            self.attn = GatedMemoryMixer(
                config.hidden_size,
                d_inner,
                bias=config.mamba_proj_bias,
            )
        elif self.layer_type in {
            "local_differential_attention",
            "global_differential_attention",
        }:
            self.attn = _Phi4FlashAttention(config, layer_idx)
        elif self.layer_type == "cross_differential_attention":
            self.attn = _Phi4FlashAttention(config, layer_idx, cross=True)
        else:
            raise ValueError(f"Unsupported Phi4Flash layer type {self.layer_type!r}")

    @staticmethod
    def _add_residual(op: OpBuilder, residual: ir.Value, update: ir.Value) -> ir.Value:
        """PyTorch promotes bf16 sublayer outputs into the fp32 Mamba residual."""
        return op.Add(residual, op.CastLike(update, residual))

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        *,
        attention_mask: ir.Value | None,
        padding_mask: ir.Value,
        past_state: tuple[ir.Value, ir.Value] | None,
        shared_memory: ir.Value | None,
        shared_key_value: tuple[ir.Value, ir.Value] | None,
    ) -> tuple[
        ir.Value,
        tuple[ir.Value, ir.Value] | None,
        ir.Value | None,
        tuple[ir.Value, ir.Value] | None,
    ]:
        """Return hidden state, owned cache state, transient memory, shared KV."""
        residual = hidden_states
        normalized = self.input_layernorm(op, op.CastLike(hidden_states, self.input_layernorm.weight))
        produced_memory = None
        produced_key_value = None

        if self.layer_type in {"mamba", "shared_memory_mamba"}:
            if past_state is None:
                raise ValueError(f"Phi4Flash Mamba layer {self.layer_idx} requires recurrent state")
            attn_output, conv_state, ssm_state, raw_ssm = self.attn(
                op,
                normalized,
                past_state[0],
                past_state[1],
                padding_mask,
            )
            present_state = (conv_state, ssm_state)
            if self.layer_type == "shared_memory_mamba":
                produced_memory = raw_ssm
            # The source explicitly casts only Mamba residuals to fp32.
            residual = op.Cast(residual, to=ir.DataType.FLOAT)
            attn_output = op.Cast(attn_output, to=ir.DataType.FLOAT)
        elif self.layer_type == "cross_mamba":
            if shared_memory is None:
                raise ValueError("Phi4Flash cross Mamba requires layer-16 shared memory")
            attn_output = self.attn(op, normalized, shared_memory)
            present_state = None
        elif self.layer_type in {
            "local_differential_attention",
            "global_differential_attention",
        }:
            if attention_mask is None or past_state is None:
                raise ValueError(
                    f"Phi4Flash differential attention layer {self.layer_idx} requires KV state and bias"
                )
            attn_output, present_state = self.attn(
                op,
                normalized,
                attention_mask,
                past_key_value=past_state,
            )
            if self.layer_type == "global_differential_attention":
                produced_key_value = present_state
        else:
            if attention_mask is None:
                raise ValueError("Phi4Flash cross differential attention requires an attention bias")
            attn_output, present_state = self.attn(
                op,
                normalized,
                attention_mask,
                shared_key_value=shared_key_value,
            )

        hidden_states = self._add_residual(op, residual, attn_output)
        residual = hidden_states
        normalized = self.post_attention_layernorm(
            op,
            op.CastLike(hidden_states, self.post_attention_layernorm.weight),
        )
        mlp_output = self.mlp(op, normalized)
        return (
            self._add_residual(op, residual, mlp_output),
            present_state,
            produced_memory,
            produced_key_value,
        )


class _Phi4FlashTextModel(nn.Module):
    """SambaY backbone with 18 cache slots and transient YOCO topology."""

    def __init__(self, config: Phi4FlashConfig):
        super().__init__()
        self.config = config
        self._dtype = config.dtype
        self._local_window = config.local_attention_window
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [_Phi4FlashDecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.final_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _tail_mask(
        self,
        op: OpBuilder,
        attention_mask: ir.Value,
        past_key_value: ir.Value,
        input_ids: ir.Value,
    ) -> ir.Value:
        """Align a local layer's mask to its bounded chronological cache."""
        key_length = op.Add(
            op.Shape(past_key_value, start=2, end=3),
            op.Shape(input_ids, start=1, end=2),
        )
        total_length = op.Shape(attention_mask, start=1, end=2)
        return op.Slice(attention_mask, op.Sub(total_length, key_length), total_length, [1])

    def _current_mask(self, op: OpBuilder, attention_mask: ir.Value, input_ids: ir.Value) -> ir.Value:
        current_length = op.Shape(input_ids, start=1, end=2)
        total_length = op.Shape(attention_mask, start=1, end=2)
        return op.Slice(attention_mask, op.Sub(total_length, current_length), total_length, [1])

    def _local_cache_tail(self, op: OpBuilder, cache: ir.Value) -> ir.Value:
        """Keep the reference local cache's last 512 chronological entries."""
        return op.Slice(
            cache,
            starts=[-self._local_window],
            ends=[INT64_MAX],
            axes=[2],
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        past_states: tuple[tuple[ir.Value, ir.Value], ...],
    ) -> tuple[ir.Value, list[tuple[ir.Value, ir.Value]], list[tuple[int, ir.Value]]]:
        """Run prefill or decode; all state inputs must be explicitly supplied."""
        if len(past_states) != 18 and len(past_states) != len(self.layers) // 2 + 2:
            raise ValueError(
                "Phi4Flash requires exactly the source cache slots 0..17 "
                "(or the equivalent derived tiny-config count)"
            )
        hidden_states = self.embed_tokens(op, input_ids)
        padding_mask = self._current_mask(op, attention_mask, input_ids)
        present_states: list[tuple[ir.Value, ir.Value]] = []
        captured_hidden_states: list[tuple[int, ir.Value]] = []
        shared_memory = None
        shared_key_value = None

        for layer in self.layers:
            layer_state = (
                past_states[layer.layer_idx]
                if layer.layer_type
                in {"mamba", "shared_memory_mamba", "local_differential_attention", "global_differential_attention"}
                else None
            )
            if layer.layer_type == "local_differential_attention":
                assert layer_state is not None
                layer_attention_mask = self._tail_mask(op, attention_mask, layer_state[0], input_ids)
            elif layer.layer_type in {
                "global_differential_attention",
                "cross_differential_attention",
            }:
                layer_attention_mask = attention_mask
            else:
                layer_attention_mask = None

            hidden_states, present_state, new_memory, new_key_value = layer(
                op,
                hidden_states,
                attention_mask=layer_attention_mask,
                padding_mask=padding_mask,
                past_state=layer_state,
                shared_memory=shared_memory,
                shared_key_value=shared_key_value,
            )
            if new_memory is not None:
                shared_memory = new_memory
            if new_key_value is not None:
                shared_key_value = new_key_value
            if present_state is not None:
                if layer.layer_type == "local_differential_attention":
                    present_state = (
                        self._local_cache_tail(op, present_state[0]),
                        self._local_cache_tail(op, present_state[1]),
                    )
                present_states.append(present_state)
            if self.config.output_layer_indices and layer.layer_idx in self.config.output_layer_indices:
                # Matches HuggingFace ``hidden_states[layer_idx + 1]``: this
                # is the post-residual output, before the next layer's norm.
                captured_hidden_states.append((layer.layer_idx, hidden_states))

        hidden_states = self.final_layernorm(
            op,
            op.CastLike(hidden_states, self.final_layernorm.weight),
        )
        return hidden_states, present_states, captured_hidden_states


class Phi4FlashCausalLMModel(nn.Module):
    """Phi-4 Flash reasoning causal LM with the source-compatible SambaY cache."""

    default_task: str = "phi4flash-text-generation"
    category: str = "SambaY SSM + differential attention"
    config_class: type = Phi4FlashConfig

    def __init__(self, config: Phi4FlashConfig):
        super().__init__()
        self.config = config
        self.model = _Phi4FlashTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        past_states: tuple[tuple[ir.Value, ir.Value], ...],
    ) -> tuple[ir.Value, list[tuple[ir.Value, ir.Value]], list[tuple[int, ir.Value]]]:
        hidden_states, present_states, captured_hidden_states = self.model(
            op, input_ids, attention_mask, past_states
        )
        return self.lm_head(op, hidden_states), present_states, captured_hidden_states

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Route all checkpoint-native paths and normalize the tied LM head."""
        if self.config.tie_word_embeddings:
            embedding_name = "model.embed_tokens.weight"
            head_name = "lm_head.weight"
            if embedding_name in state_dict and head_name in state_dict:
                embedding = state_dict[embedding_name]
                head = state_dict[head_name]
                if embedding.shape != head.shape or embedding.dtype != head.dtype or not torch.equal(
                    embedding, head
                ):
                    raise ValueError(
                        "Phi4Flash checkpoint has different tied model.embed_tokens.weight and "
                        "lm_head.weight tensors"
                    )
            if embedding_name not in state_dict:
                if head_name not in state_dict:
                    raise KeyError(
                        "Phi4Flash tied embeddings require model.embed_tokens.weight or lm_head.weight"
                    )
                state_dict[embedding_name] = state_dict[head_name]
            state_dict.pop(head_name, None)
        return state_dict
