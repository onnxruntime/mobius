# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-5.2 MLA+MoE with IndexShare deep sparse attention and MTP export."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._common import LayerNorm
from mobius.components._deepseek_mla import DeepSeekMLA
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.deepseek import (
    DeepSeekMoEGate,
    DeepSeekV3CausalLMModel,
    DeepSeekV3TextModel,
    _DeepSeekMoEFFN,
    _linear_class,
)

if TYPE_CHECKING:
    import torch


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def _indexer_types(config: ArchitectureConfig) -> list[str]:
    """Return the authoritative full/shared IndexShare schedule."""
    if config.indexer_types is not None:
        return list(config.indexer_types)
    full_layers = {0, 1, 2}
    full_layers.update(
        range(
            config.index_skip_topk_offset * 2,
            config.num_hidden_layers,
            config.index_topk_freq,
        )
    )
    return ["full" if i in full_layers else "shared" for i in range(config.num_hidden_layers)]


class GlmMoeDsaIndexer(nn.Module):
    """GLM-5.2 fp32 token indexer used by full IndexShare layers."""

    def __init__(self, config: ArchitectureConfig, linear_class: type):
        super().__init__()
        assert config.q_lora_rank is not None
        assert config.index_head_dim is not None
        assert config.index_n_heads is not None
        self.index_head_dim = config.index_head_dim
        self.index_n_heads = config.index_n_heads
        self.index_topk = int(config.index_topk or 2048)
        self.rope_interleave = config.indexer_rope_interleave
        self.wq_b = linear_class(
            config.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
        )
        self.wk = linear_class(config.hidden_size, self.index_head_dim, bias=False)
        self.k_norm = LayerNorm(self.index_head_dim, eps=1e-6)
        self.weights_proj = linear_class(config.hidden_size, self.index_n_heads, bias=False)

    def project_key(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple,
    ) -> ir.Value:
        key = self.k_norm(op, self.wk(op, hidden_states))
        return apply_rotary_pos_emb(
            op,
            key,
            position_embeddings,
            num_heads=1,
            rotary_embedding_dim=self.index_head_dim // 2,
            interleaved=self.rope_interleave,
        )

    def select(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        q_resid: ir.Value,
        all_index_keys: ir.Value,
        attention_bias: ir.Value,
    ) -> ir.Value:
        query = self.wq_b(op, q_resid)
        query = op.Reshape(query, [0, 0, self.index_n_heads, self.index_head_dim])
        query = op.Cast(query, to=ir.DataType.FLOAT)
        keys = op.Cast(all_index_keys, to=ir.DataType.FLOAT)
        scores = op.MatMul(query, op.Transpose(keys, perm=[0, 2, 1]))
        scores = op.Relu(scores)

        weights = self.weights_proj(op, hidden_states)
        weights = op.Cast(weights, to=ir.DataType.FLOAT)
        weights = op.Mul(weights, self.index_n_heads**-0.5)
        scores = op.ReduceSum(op.Mul(scores, op.Unsqueeze(weights, [3])), [2], keepdims=False)
        scores = op.Add(scores, op.Cast(op.Squeeze(attention_bias, [1]), to=ir.DataType.FLOAT))

        key_length = op.Shape(scores, start=2, end=3)
        k = op.Min(key_length, op.Constant(value_ints=[self.index_topk]))
        _, indices = op.TopK(scores, k, axis=-1, largest=1, sorted=0, _outputs=2)
        return op.Cast(indices, to=ir.DataType.INT32)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        q_resid: ir.Value,
        position_embeddings: tuple,
        past_index_key: ir.Value | None,
        attention_bias: ir.Value,
    ):
        current_index_key = self.project_key(op, hidden_states, position_embeddings)
        all_index_keys = (
            current_index_key
            if past_index_key is None
            else op.Concat(past_index_key, current_index_key, axis=1)
        )
        indices = self.select(
            op,
            hidden_states,
            q_resid,
            all_index_keys,
            attention_bias,
        )
        return all_index_keys, indices


class GlmMoeDsaAttention(DeepSeekMLA):
    """MLA attention with a packed indexer-key cache and sparse additive mask."""

    def __init__(
        self,
        config: ArchitectureConfig,
        indexer_type: str,
        linear_class: type,
    ):
        super().__init__(config, linear_class=linear_class)
        self.indexer_type = indexer_type
        self.dtype = config.dtype
        self.main_key_dim = self.num_heads * self.qk_head_dim
        self.main_value_dim = self.num_heads * self.v_head_dim
        self.index_head_dim = int(config.index_head_dim or 0)
        if indexer_type == "full":
            self.indexer = GlmMoeDsaIndexer(config, linear_class)

    def _unpack_past(self, op: OpBuilder, past_key_value: tuple):
        packed_key, packed_value = past_key_value
        key_tokens = op.Squeeze(packed_key, [1])
        main_key = op.Slice(key_tokens, [0], [self.main_key_dim], [2])
        main_key = op.Transpose(
            op.Reshape(main_key, [0, 0, self.num_heads, self.qk_head_dim]),
            perm=[0, 2, 1, 3],
        )
        value_tokens = op.Squeeze(packed_value, [1])
        main_value = op.Transpose(
            op.Reshape(value_tokens, [0, 0, self.num_heads, self.v_head_dim]),
            perm=[0, 2, 1, 3],
        )
        index_key = None
        if self.indexer_type == "full":
            index_key = op.Slice(
                key_tokens,
                [self.main_key_dim],
                [self.main_key_dim + self.index_head_dim],
                [2],
            )
        return (main_key, main_value), index_key

    def _pack_present(
        self,
        op: OpBuilder,
        present: tuple,
        index_keys: ir.Value | None,
    ) -> tuple:
        key = op.Reshape(op.Transpose(present[0], perm=[0, 2, 1, 3]), [0, 0, self.main_key_dim])
        if index_keys is not None:
            key = op.Concat(key, index_keys, axis=-1)
        value = op.Reshape(
            op.Transpose(present[1], perm=[0, 2, 1, 3]),
            [0, 0, self.main_value_dim],
        )
        return op.Unsqueeze(key, [1]), op.Unsqueeze(value, [1])

    def _sparse_bias(
        self,
        op: OpBuilder,
        indices: ir.Value,
        attention_bias: ir.Value,
    ) -> ir.Value:
        shape = op.Shape(op.Squeeze(attention_bias, [1]))
        masked = op.Expand(op.Constant(value_float=float(ir.DataType.FLOAT.min)), shape)
        updates = op.Mul(op.Cast(indices, to=ir.DataType.FLOAT), 0.0)
        sparse = op.ScatterElements(masked, indices, updates, axis=-1, reduction="none")
        sparse = op.Cast(op.Unsqueeze(sparse, [1]), to=self.dtype)
        return op.Add(attention_bias, sparse)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
        shared_topk_indices: ir.Value | None = None,
    ):
        q_resid = self.q_a_layernorm(op, self.q_a_proj(op, hidden_states))
        q = self.q_b_proj(op, q_resid)
        q = op.Reshape(q, [0, 0, self.num_heads, self.qk_head_dim])
        q_nope, q_rope = op.Split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            axis=-1,
            _outputs=2,
        )
        q_rope = op.Reshape(q_rope, [0, 0, -1])
        q_rope = apply_rotary_pos_emb(
            op,
            q_rope,
            position_embeddings,
            num_heads=self.num_heads,
            interleaved=self._rope_interleave,
        )
        q = op.Concat(
            op.Reshape(q_nope, [0, 0, self.num_heads, self.qk_nope_head_dim]),
            op.Reshape(q_rope, [0, 0, self.num_heads, self.qk_rope_head_dim]),
            axis=-1,
        )
        q = op.Reshape(q, [0, 0, self.num_heads * self.qk_head_dim])

        compressed_kv = self.kv_a_proj_with_mqa(op, hidden_states)
        kv_latent, k_rope = op.Split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
            _outputs=2,
        )
        kv_latent = self.kv_a_layernorm(op, kv_latent)
        kv = self.kv_b_proj(op, kv_latent)
        kv = op.Reshape(
            kv,
            [0, 0, self.num_heads, self.qk_nope_head_dim + self.v_head_dim],
        )
        k_nope, value = op.Split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            axis=-1,
            _outputs=2,
        )
        value = op.Reshape(value, [0, 0, -1])
        k_rope = apply_rotary_pos_emb(
            op,
            k_rope,
            position_embeddings,
            num_heads=1,
            interleaved=self._rope_interleave,
        )
        k_rope = op.Tile(k_rope, [1, 1, self.num_heads])
        key = op.Concat(
            op.Reshape(k_nope, [0, 0, self.num_heads, self.qk_nope_head_dim]),
            op.Reshape(k_rope, [0, 0, self.num_heads, self.qk_rope_head_dim]),
            axis=-1,
        )
        key = op.Reshape(key, [0, 0, self.num_heads * self.qk_head_dim])

        main_past = None
        past_index_key = None
        if past_key_value is not None:
            main_past, past_index_key = self._unpack_past(op, past_key_value)

        all_index_keys = None
        topk_indices = shared_topk_indices
        if self.indexer_type == "full":
            all_index_keys, topk_indices = self.indexer(
                op,
                hidden_states,
                q_resid,
                position_embeddings,
                past_index_key,
                attention_bias,
            )
        if topk_indices is None:
            raise ValueError("Shared GLM DSA layers require top-k indices from a preceding full layer")

        sparse_bias = self._sparse_bias(op, topk_indices, attention_bias)
        attn_output, present_key, present_value = op.Attention(
            q,
            key,
            value,
            attn_mask=sparse_bias,
            past_key=main_past[0] if main_past is not None else None,
            past_value=main_past[1] if main_past is not None else None,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=self.scaling,
            _outputs=3,
        )
        attn_output = self.o_proj(op, attn_output)
        present = self._pack_present(
            op,
            (present_key, present_value),
            all_index_keys,
        )
        return attn_output, present, topk_indices


class GlmMoeDsaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ArchitectureConfig,
        indexer_type: str,
        is_moe: bool,
        linear_class: type,
    ):
        super().__init__()
        self.self_attn = GlmMoeDsaAttention(config, indexer_type, linear_class)
        if is_moe:
            self.mlp = _DeepSeekMoEFFN(
                config,
                DeepSeekMoEGate(config),
                linear_class=linear_class,
            )
        else:
            self.mlp = MLP(config, linear_class=linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
        shared_topk_indices: ir.Value | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present, topk_indices = self.self_attn(
            op,
            hidden_states,
            attention_bias,
            position_embeddings,
            past_key_value,
            shared_topk_indices,
        )
        hidden_states = op.Add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states), present, topk_indices


class GlmMoeDsaTextModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = initialize_rope(config)
        linear_class = _linear_class(config)
        types = _indexer_types(config)
        self.layers = nn.ModuleList(
            [
                GlmMoeDsaDecoderLayer(
                    config,
                    types[i],
                    is_moe=i >= config.first_k_dense_replace,
                    linear_class=linear_class,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids,
            attention_mask,
            dtype=self.config.dtype,
        )
        presents = []
        shared_topk_indices = None
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            hidden_states, present, shared_topk_indices = layer(
                op,
                hidden_states,
                attention_bias,
                position_embeddings,
                past,
                shared_topk_indices,
            )
            presents.append(present)
        return self.norm(op, hidden_states), presents


class _SharedHead(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GlmMoeDsaMtp(nn.Module):
    """Improved GLM-5.2 MTP head: embed/hidden fusion plus a full DSA MoE block."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        linear_class = _linear_class(config)
        self.config = config
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = linear_class(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layer = GlmMoeDsaDecoderLayer(
            config,
            "full",
            is_moe=True,
            linear_class=linear_class,
        )
        self.shared_head = _SharedHead(config)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_value: tuple,
    ):
        fused = self.eh_proj(
            op,
            op.Concat(self.enorm(op, inputs_embeds), self.hnorm(op, hidden_states), axis=-1),
        )
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            position_ids,
            attention_mask,
            dtype=self.config.dtype,
        )
        output, present, topk_indices = self.layer(
            op,
            fused,
            attention_bias,
            position_embeddings,
            past_key_value,
            None,
        )
        return self.shared_head.norm(op, output), present, topk_indices


class GlmMoeDsaCausalLMModel(DeepSeekV3CausalLMModel):
    """GLM-5.2 target model with portable IndexShare and an exported MTP graph."""

    default_task = "glm-moe-dsa"

    def __init__(self, config: ArchitectureConfig):
        if config.indexer_types is None:
            config.indexer_types = _indexer_types(config)
        nn.Module.__init__(self)
        self.config = config
        self.model = (
            GlmMoeDsaTextModel(config) if config.use_dsa else DeepSeekV3TextModel(config)
        )
        qc = config.quantization
        lm_head_class = (
            _linear_class(config) if qc is not None and qc.quantize_lm_head else Linear
        )
        self.lm_head = lm_head_class(config.hidden_size, config.vocab_size, bias=False)
        if config.num_nextn_predict_layers > 0:
            if config.num_nextn_predict_layers != 1:
                raise ValueError("GLM MTP export currently supports exactly one NextN layer")
            if not config.use_dsa:
                raise ValueError(
                    "GLM-5.2 MTP export currently requires DSA; set num_nextn_predict_layers=0 "
                    "when using the full-attention fallback"
                )
            self.mtp = GlmMoeDsaMtp(config)
        else:
            self.mtp = None

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = DeepSeekV3CausalLMModel.preprocess_weights(self, state_dict)
        result: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            match = _LAYER_RE.match(key)
            if match is not None and int(match.group(1)) >= self.config.num_hidden_layers:
                layer_idx = int(match.group(1))
                if (
                    self.mtp is None
                    or layer_idx
                    >= self.config.num_hidden_layers + self.config.num_nextn_predict_layers
                ):
                    continue
                suffix = match.group(2)
                if suffix.startswith(("enorm.", "hnorm.", "eh_proj.", "shared_head.")):
                    key = f"mtp.{suffix}"
                else:
                    key = f"mtp.layer.{suffix}"
            elif not self.config.use_dsa and ".indexer." in key:
                continue
            result[key.replace(".mlp.experts.", ".mlp.moe.experts.")] = value
        return result
