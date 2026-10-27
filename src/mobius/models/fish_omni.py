# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Drama 3 frame embeddings, hybrid text decoder, and temporal codebook decoder.

Replicates Fish Speech's FishOmniBackbone and FishOmniAudioDecoder. Graphs
consume token/code frames rather than waveforms. The task attaches ONNX sampling
policies; the waveform codec is exported independently by mobius.tasks.fish_codec.
The caller still formats prompts and orchestrates temporal decoding.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._configs.fish_omni import FishOmniConfig
from mobius.components import (
    MLP,
    Embedding,
    Float32SiLU,
    Linear,
    OffsetRMSNorm,
    PostGatedRMSNorm,
    Qwen35Attention,
    SoftmaxTopKGate,
    apply_rotary_pos_emb,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import _apply_attention
from mobius.models.qwen35 import (
    Qwen35MoEBlock,
    Qwen35MoEDecoderLayer,
    Qwen35MoETextModel,
)


def _float32_rope(config: ArchitectureConfig) -> nn.Module:
    rope = initialize_rope(dataclasses.replace(config, dtype=ir.DataType.FLOAT))
    assert rope is not None
    return rope


def _sigmoid_float32(op, value):
    return op.CastLike(op.Sigmoid(op.Cast(value, to=ir.DataType.FLOAT)), value)


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
            op.Swish(op.Cast(gate, to=ir.DataType.FLOAT)),
        )
        return op.CastLike(gated, hidden_states)


class _FishAttention(Qwen35Attention):
    """Gated GQA with FP32 partial rotation before rounding Q/K back."""

    def __init__(self, config):
        super().__init__(config)
        self._compute_dtype = config.dtype

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
        super().__init__(config, gate=gate)
        if self.experts is not None:
            for expert in self.experts:
                assert isinstance(expert, MLP)
                expert.act_fn = Float32SiLU()
        self.shared_expert.act_fn = Float32SiLU()

    def forward(self, op, hidden_states):
        if self.experts is None:
            return super().forward(op, hidden_states)
        routing, selected = self.gate(op, hidden_states)
        combined = None
        for index, expert in enumerate(self.experts):
            output = expert(op, hidden_states)
            match = op.CastLike(op.Equal(selected, index), routing)
            weight = op.ReduceSum(op.Mul(routing, match), [-1], keepdims=True)
            contribution = op.Mul(
                op.Cast(output, to=ir.DataType.FLOAT),
                op.Cast(weight, to=ir.DataType.FLOAT),
            )
            combined = contribution if combined is None else op.Add(combined, contribution)
        assert combined is not None
        routed = op.CastLike(combined, hidden_states)
        shared = self.shared_expert(op, hidden_states)
        gate = _sigmoid_float32(op, self.shared_expert_gate(op, hidden_states))
        return op.Add(routed, op.Mul(shared, gate))


class _FishRouter(SoftmaxTopKGate):
    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        logits = op.MatMul(hidden_states, op.Transpose(self.weight, perm=[1, 0]))
        probabilities = op.Softmax(op.Cast(logits, to=ir.DataType.FLOAT), axis=-1)
        weights, indices = op.TopK(
            probabilities, op.Constant(value_ints=[self.top_k]), axis=-1, _outputs=2
        )
        if self.norm_topk_prob:
            weights = op.Div(weights, op.ReduceSum(weights, [-1], keepdims=True))
        return op.CastLike(weights, hidden_states), indices


class _FishDecoderLayer(Qwen35MoEDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mlp = _FishMoEBlock(
            config,
            gate=_FishRouter(
                config.hidden_size,
                config.num_local_experts,
                config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
            ),
        )
        if self.layer_type == "linear_attention":
            self.linear_attn.norm = _FishGatedNorm(
                config.linear_value_head_dim,
                eps=config.rms_norm_eps,
            )
        else:
            self.self_attn = _FishAttention(config)


class _FishTextModel(Qwen35MoETextModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [_FishDecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.rotary_emb = _float32_rope(config)


class FishOmniBackbone(nn.Module):
    """Hybrid Qwen3.5-MoE decoder producing both logits and normalized hidden states."""

    def __init__(self, config: FishOmniConfig):
        super().__init__()
        self.model = _FishTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, op, inputs_embeds, attention_mask, position_ids, past_key_values):
        hidden, present = self.model(
            op,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
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
        mixed = op.Div(
            op.Cast(op.Add(text, code_sum(1, count + 1)), to=ir.DataType.FLOAT),
            math.sqrt(1 + count),
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
        self.wqkv = Linear(config.hidden_size, 2 * (self.q_width + self.kv_width), bias=False)
        self.wo = Linear(self.q_width, config.hidden_size, bias=False)
        # Released configs say attention_qk_norm=False, but the trained tower
        # always uses zero-centered per-head Q/K norms (Fish Speech TowerAttention).
        self.q_norm = OffsetRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(config.head_dim, eps=config.rms_norm_eps)

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


class _TowerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = Float32SiLU()

    def forward(self, op, hidden):
        gate = self.w1(op, hidden)
        return self.w2(op, op.Mul(self.act_fn(op, gate), self.w3(op, hidden)))


class _TowerTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = _TowerAttention(config)
        self.attention_norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = _TowerMLP(config)

    def forward(self, op, hidden, bias, positions, past):
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
        self.enorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.transformer_layer = _TowerTransformer(config)
        self.final_layernorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        return self.final_layernorm(op, output), present


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
                Linear(config.hidden_size, config.tower.vocab_size, bias=False)
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
    }

    def __init__(self, config: FishOmniConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.decoder = FishOmniBackbone(config)
        self.embedding = FishOmniEmbedding(config)
        self.audio_decoder = FishOmniAudioDecoder(config)
