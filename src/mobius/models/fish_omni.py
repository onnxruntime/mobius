# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Drama 3 frame embeddings, hybrid text decoder, and temporal codebook decoder.

Replicates Fish Speech's FishOmniBackbone and FishOmniAudioDecoder. Graphs
consume token/code frames rather than waveforms; prompt formatting, sampling,
and the separately distributed audio codec remain caller responsibilities.
"""

from __future__ import annotations

import math
from typing import ClassVar

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._configs.fish_omni import FishOmniConfig
from mobius.components import (
    Embedding,
    Linear,
    OffsetRMSNorm,
    SoftmaxTopKGate,
    apply_rotary_pos_emb,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import _apply_attention
from mobius.models.qwen35 import Qwen35MoEDecoderLayer, Qwen35MoETextModel


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


class FishOmniBackbone(nn.Module):
    """Hybrid Qwen3.5-MoE decoder producing both logits and normalized hidden states."""

    def __init__(self, config: FishOmniConfig):
        super().__init__()
        self.model = Qwen35MoETextModel(config)
        assert config.num_local_experts is not None and config.num_experts_per_tok is not None
        for layer in self.model.layers:
            assert isinstance(layer, Qwen35MoEDecoderLayer)
            layer.mlp.gate = _FishRouter(
                config.hidden_size,
                config.num_local_experts,
                config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
            )
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
            op.Add(text, code_sum(1, count + 1)), op.CastLike(math.sqrt(1 + count), text)
        )
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
        q = apply_rotary_pos_emb(
            op,
            q,
            positions,
            num_heads=cfg.num_attention_heads,
            rotary_embedding_dim=self.rotary_dim,
            interleaved=True,
        )
        k = apply_rotary_pos_emb(
            op,
            k,
            positions,
            num_heads=cfg.num_key_value_heads,
            rotary_embedding_dim=self.rotary_dim,
            interleaved=True,
        )
        attended, pk, pv = _apply_attention(
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
        )
        return self.wo(op, op.Mul(attended, op.Sigmoid(gate))), (pk, pv)


class _TowerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, op, hidden):
        gate = self.w1(op, hidden)
        return self.w2(op, op.Mul(op.Mul(gate, op.Sigmoid(gate)), self.w3(op, hidden)))


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
        self.shared_layers = nn.ModuleList(
            [
                _TowerLayer(config.tower, shared=True)
                for _ in range(config.shared_enrichment_layers)
            ]
        )
        self.residual_layers = nn.ModuleList(
            [_TowerLayer(config.tower, shared=False) for _ in range(config.num_codebooks - 1)]
        )
        self.codebook_heads = nn.ModuleList(
            [
                Linear(config.hidden_size, config.tower.vocab_size, bias=False)
                for _ in range(config.num_codebooks - 1)
            ]
        )
        rotary_emb = initialize_rope(config.tower)
        assert rotary_emb is not None
        self.rotary_emb = rotary_emb

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
    graphs. Sampling, reference-bracket prompt formatting, and waveform codec
    execution are external. FP8 experts require explicit streaming dequantization.
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
