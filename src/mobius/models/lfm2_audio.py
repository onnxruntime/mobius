# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""LFM2-Audio: audio-to-audio model with hybrid conv+attention backbone.

Architecture (4-model ONNX split):
1. **audio_encoder**: ConformerEncoder + adapter MLP
   mel (B, n_mels, T) -> audio embeddings (B, T', hidden_size)
2. **embedding**: text token embed + audio codebook embed
   text_ids + audio_features -> inputs_embeds
3. **decoder**: LFM2 backbone (takes inputs_embeds, not input_ids)
   inputs_embeds -> text_logits + hybrid KV cache
4. **audio_decoder**: depthformer (per-codebook autoregressive transformer)
   backbone_hidden -> codebook_logits (one codebook at a time)

The decoder uses hybrid cache: "conv" layers carry conv_state,
"full_attention" layers carry standard KV cache.

HuggingFace weight name prefixes::

    lfm.              -> decoder sub-model (LFM2 backbone)
    conformer.        -> audio_encoder.encoder (ConformerEncoder)
    audio_adapter.    -> audio_encoder.adapter (projection MLP)
    audio_embedding.  -> embedding.audio_embedding
    depthformer.      -> audio_decoder.depthformer
    depth_linear.     -> audio_decoder.depth_linear
    depth_embeddings. -> audio_decoder.depth_embeddings
    embedding_norm.   -> audio_decoder.embedding_norm

Reference: ``liquid_audio.model.lfm2_audio.LFM2AudioModel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Lfm2AudioConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    FCMLP,
    MLP,
    Attention,
    ConformerEncoder,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.models.lfm2 import Lfm2AttentionDecoderLayer, Lfm2ConvDecoderLayer

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Audio Encoder sub-model
# ---------------------------------------------------------------------------


class _Lfm2AudioEncoder(nn.Module):
    """ConformerEncoder + adapter MLP for mel -> LFM hidden_size projection.

    The adapter is a 2-layer MLP:
        Linear(encoder_dim, hidden_size) -> GELU -> Linear(hidden_size, hidden_size)

    Weight names (HF)::

        conformer.* -> encoder.*
        audio_adapter.model.0.{weight,bias} -> adapter.up_proj.{weight,bias}
        audio_adapter.model.1.{weight,bias} -> (batch_norm, skipped)
        audio_adapter.model.3.{weight,bias} -> adapter.down_proj.{weight,bias}
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        audio = config.audio
        assert audio is not None

        self.encoder = ConformerEncoder(
            input_size=audio.num_mel_bins or 128,
            attention_dim=audio.attention_dim or audio.d_model or 512,
            attention_heads=audio.attention_heads or audio.encoder_attention_heads or 8,
            num_blocks=audio.num_blocks or audio.encoder_layers or 17,
            linear_units=(audio.linear_units or audio.encoder_ffn_dim or 2048),
            kernel_size=audio.kernel_size or 9,
            conv_channels=audio.conv_channels or 256,
            t5_bias_max_distance=audio.t5_bias_max_distance or 500,
        )
        # Adapter: encoder_dim -> hidden_size
        encoder_dim = audio.attention_dim or audio.d_model or 512
        self.adapter = FCMLP(
            hidden_size=encoder_dim,
            intermediate_size=config.hidden_size,
            activation="gelu",
            bias=True,
        )

    def forward(self, op: builder.OpBuilder, input_features: ir.Value):
        """Forward: mel (B, n_mels, T) -> (B, T', hidden_size)."""
        # ConformerEncoder expects (B, T, n_mels); transpose from (B, n_mels, T)
        input_features = op.Transpose(input_features, perm=[0, 2, 1])
        audio_features = self.encoder(op, input_features)
        # Adapter MLP: (B, T', encoder_dim) -> (B, T', hidden_size)
        return self.adapter(op, audio_features)


# ---------------------------------------------------------------------------
# Embedding sub-model
# ---------------------------------------------------------------------------


class _Lfm2AudioEmbedding(nn.Module):
    """Embedding model for LFM2-Audio.

    Combines text token embeddings with audio feature embeddings.
    In the actual model, a modality_flag tensor controls which positions
    get text embeddings vs audio-in vs audio-out embeddings.

    For ONNX export, this takes pre-computed audio features and text_ids,
    returning the combined inputs_embeds sequence.

    Weight names (HF)::

        lfm.embed_tokens.weight -> text_embed.weight
        audio_embedding.embedding.weight -> audio_embed.weight
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        self.text_embed = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # Audio codebook embedding: codebooks * audio_vocab_size entries
        audio_vocab = config.audio_vocab_size * config.num_codebooks
        self.audio_embed = Embedding(audio_vocab, config.hidden_size)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
    ):
        """Forward: text_ids -> inputs_embeds.

        Returns text embeddings only. Audio codebook embeddings are
        handled separately by the audio_decoder's depth_embeddings.
        Runtime assembles text + audio at the sequence level.
        """
        return self.text_embed(op, input_ids)


# ---------------------------------------------------------------------------
# Decoder sub-model (LFM2 backbone without embed_tokens)
# ---------------------------------------------------------------------------


# Reuse the decoder layers from the base LFM2 model — they are identical
# for the audio backbone. Lfm2AudioConfig inherits from Lfm2Config, so
# the constructors accept it directly.
_Lfm2AudioDecoderLayer = Lfm2AttentionDecoderLayer
_Lfm2AudioConvLayer = Lfm2ConvDecoderLayer


class _Lfm2AudioDecoder(nn.Module):
    """LFM2 decoder backbone: takes inputs_embeds -> logits + cache.

    This is the LFM2 model minus the embedding layer. It takes
    pre-assembled inputs_embeds (from the embedding model) and runs
    the hybrid conv+attention backbone, then projects to vocab logits.

    The text LM head shares weights with embed_tokens (tied).
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        self._dtype = config.dtype

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype == "conv":
                self.layers.append(_Lfm2AudioConvLayer(config))
            else:
                self.layers.append(_Lfm2AudioDecoderLayer(config))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        # LM head (tied with lfm.embed_tokens in preprocess_weights)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            # Use a dummy input_ids shape from inputs_embeds
            input_ids=position_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


# ---------------------------------------------------------------------------
# Audio decoder sub-model (depthformer)
# ---------------------------------------------------------------------------


class _DepthformerLayer(nn.Module):
    """Single depthformer layer with RMSNorm, Attention, and SwiGLU MLP.

    Architecture: RMSNorm -> Attention -> residual ->
    RMSNorm -> SwiGLU MLP -> residual.

    The depthformer uses the same StandardBlock structure as the LFM2
    backbone: operator_norm + operator + ffn_norm + feed_forward.
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        from mobius._configs import ArchitectureConfig

        # Create a mini-config for the depthformer attention
        depthformer_dim = config.depthformer_dim
        depthformer_heads = config.depthformer_heads
        head_dim = depthformer_dim // depthformer_heads

        # Build attention config for the depthformer
        attn_config = ArchitectureConfig(
            hidden_size=depthformer_dim,
            intermediate_size=depthformer_dim * 4,
            num_attention_heads=depthformer_heads,
            num_key_value_heads=depthformer_heads,
            head_dim=head_dim,
            hidden_act="silu",
            attn_qk_norm=True,
            rms_norm_eps=1e-5,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.self_attn = Attention(attn_config)
        self.operator_norm = RMSNorm(depthformer_dim, eps=1e-5)
        self.ffn_norm = RMSNorm(depthformer_dim, eps=1e-5)
        self.feed_forward = MLP(attn_config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        residual = hidden_states
        hidden_states = self.operator_norm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states, present_kv


class _Lfm2AudioDecoderModule(nn.Module):
    """Depthformer audio decoder for per-codebook token prediction.

    Takes backbone output + previous codebook embedding, runs through
    depthformer layers, and produces logits for the current codebook.

    Architecture::

        depth_linear(backbone_hidden) -> split by codebook_idx ->
        + prev_embedding -> depthformer layers -> embedding_norm ->
        codebook_head -> codebook_logits

    The codebook heads share weights with depth_embeddings (tied).
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        depthformer_dim = config.depthformer_dim

        # Project backbone hidden -> codebook inputs
        # depth_linear: (hidden_size) -> (codebooks * depthformer_dim)
        self.depth_linear = Linear(
            config.hidden_size,
            config.num_codebooks * depthformer_dim,
            bias=True,
        )

        # Depthformer layers
        self.layers = nn.ModuleList([])
        for _ in range(config.depthformer_layers):
            self.layers.append(_DepthformerLayer(config))

        # Output norm + per-codebook heads
        self.embedding_norm = RMSNorm(depthformer_dim, eps=1e-5)

        # Per-codebook logit projection
        # Each codebook has its own embedding + tied head
        self.depth_embeddings = nn.ModuleList([])
        for _ in range(config.num_codebooks):
            self.depth_embeddings.append(
                Linear(depthformer_dim, config.audio_vocab_size, bias=False)
            )

        # Stacked head weights for dynamic codebook selection via Gather.
        # Shape: (num_codebooks, audio_vocab_size, depthformer_dim)
        # In preprocess_weights, this is assembled from per-codebook weights.
        self.stacked_head_weights = nn.Parameter(
            [config.num_codebooks, config.audio_vocab_size, depthformer_dim]
        )

        self._depthformer_dim = depthformer_dim
        self._num_codebooks = config.num_codebooks

        # Build a separate RoPE for depthformer
        from mobius._configs import ArchitectureConfig

        rope_config = ArchitectureConfig(
            hidden_size=depthformer_dim,
            num_attention_heads=config.depthformer_heads,
            head_dim=depthformer_dim // config.depthformer_heads,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.rotary_emb = initialize_rope(rope_config)

    def forward(
        self,
        op: builder.OpBuilder,
        backbone_hidden: ir.Value,
        prev_embedding: ir.Value,
        codebook_idx: ir.Value,
        past_key_values: list | None = None,
    ):
        """Forward pass for single-codebook prediction.

        Args:
            backbone_hidden: (B, 1, hidden_size) from LFM2 decoder
            prev_embedding: (B, 1, depthformer_dim) from previous codebook
            codebook_idx: scalar int - which codebook to predict
            past_key_values: depthformer KV cache

        Returns:
            (codebook_logits, present_key_values)
        """
        # Project backbone hidden to all codebook inputs
        # (B, 1, hidden_size) -> (B, 1, codebooks * depthformer_dim)
        projected = self.depth_linear(op, backbone_hidden)

        # Reshape to (B, codebooks, depthformer_dim) for gathering
        # First squeeze the seq dim: (B, 1, C*D) -> (B, C*D)
        projected_2d = op.Squeeze(projected, [1])
        projected_3d = op.Reshape(
            projected_2d,
            op.Constant(
                value_ints=[
                    -1,
                    self._num_codebooks,
                    self._depthformer_dim,
                ]
            ),
        )
        # (B, codebooks, depthformer_dim)

        # Gather the codebook_idx slice: (B, 1, depthformer_dim)
        # Reshape idx to (1, 1, 1) then expand to (B, 1, depthformer_dim)
        idx_3d = op.Reshape(codebook_idx, op.Constant(value_ints=[1, 1, 1]))
        # Build expand shape dynamically to match batch size at runtime
        batch_dim = op.Shape(projected_3d, start=0, end=1)  # (1,) containing B
        expand_shape = op.Concat(
            batch_dim,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[self._depthformer_dim]),
            axis=0,
        )  # (3,) -> [B, 1, depthformer_dim]
        idx_expanded = op.Expand(idx_3d, expand_shape)
        depthformer_input = op.GatherElements(projected_3d, idx_expanded, axis=1)
        # (B, 1, depthformer_dim) - unsqueeze back seq dim is already there

        # Add previous codebook embedding
        hidden_states = op.Add(depthformer_input, prev_embedding)

        # Position IDs for depthformer (single step: just codebook_idx).
        # Shape (B, 1) — derive batch dim from hidden_states at runtime.
        batch_dim = op.Shape(hidden_states, start=0, end=1)  # (1,) containing B
        one_dim = op.Constant(value_ints=[1])
        position_shape = op.Concat(batch_dim, one_dim, axis=0)  # (2,) -> [B, 1]
        position_ids = op.Reshape(codebook_idx, position_shape)
        position_embeddings = self.rotary_emb(op, position_ids)

        # Run depthformer layers
        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=None,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        # Norm + logits for current codebook
        hidden_states = self.embedding_norm(op, hidden_states)

        # Dynamic codebook head selection:
        # stacked_head_weights: (num_codebooks, audio_vocab_size, dim)
        # Gather by codebook_idx -> (audio_vocab_size, dim)
        head_weight = op.Gather(self.stacked_head_weights, codebook_idx, axis=0)
        # (audio_vocab_size, depthformer_dim)
        head_weight_3d = op.Unsqueeze(head_weight, [0])
        # (1, audio_vocab_size, depthformer_dim)
        # hidden_states: (B, 1, depthformer_dim)
        # logits = hidden_states @ head_weight^T -> (B, 1, audio_vocab_size)
        logits = op.MatMul(hidden_states, op.Transpose(head_weight_3d, perm=[0, 2, 1]))

        return logits, present_key_values


# ---------------------------------------------------------------------------
# Composite model
# ---------------------------------------------------------------------------


class Lfm2AudioModel(nn.Module):
    """LFM2-Audio: audio-to-audio model.

    Exports as 4 ONNX models via AudioToAudioTask:
    - audio_encoder: ConformerEncoder + adapter
    - embedding: text + audio embedding fusion
    - decoder: LFM2 hybrid backbone
    - audio_decoder: depthformer per-codebook decoder

    HuggingFace reference: ``liquid_audio.model.lfm2_audio.LFM2AudioModel``.
    """

    default_task: str = "audio-to-audio"
    category: str = "Audio-to-Audio"
    config_class: type = Lfm2AudioConfig

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        self.config = config

        self.audio_encoder = _Lfm2AudioEncoder(config)
        self.embedding = _Lfm2AudioEmbedding(config)
        self.decoder = _Lfm2AudioDecoder(config)
        self.audio_decoder = _Lfm2AudioDecoderModule(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map LFM2-Audio weights to ONNX sub-model parameters.

        Routes weights to sub-models by prefix:
            lfm.embed_tokens.* -> embedding.text_embed.*
            lfm.* -> decoder.* (backbone layers)
            conformer.* -> audio_encoder.encoder.*
            audio_adapter.* -> audio_encoder.adapter.*
            audio_embedding.* -> embedding.audio_embed.*
            depthformer.* -> audio_decoder.depthformer.*
            depth_linear.* -> audio_decoder.depth_linear.*
            depth_embeddings.* -> audio_decoder.depth_embeddings.*
            embedding_norm.* -> audio_decoder.embedding_norm.*
        """
        if self.config.tie_word_embeddings:
            tie_word_embeddings(
                state_dict,
                embed_key="lfm.embed_tokens.weight",
                head_key="lfm.lm_head.weight",
            )

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_lfm2_audio_weight(key)
            if new_key is not None:
                new_state_dict[new_key] = value

        # Stack per-codebook head weights into stacked_head_weights
        # for dynamic codebook selection in the audio_decoder forward.
        head_weights = []
        for i in range(self.config.num_codebooks):
            wkey = f"audio_decoder.depth_embeddings.{i}.weight"
            if wkey in new_state_dict:
                head_weights.append(new_state_dict[wkey])
        if head_weights:
            import torch

            new_state_dict["audio_decoder.stacked_head_weights"] = torch.stack(
                head_weights, dim=0
            )

        return new_state_dict


def _rename_lfm2_audio_weight(key: str) -> str | None:
    """Rename a single HF weight key to ONNX module structure.

    Returns None if the weight should be skipped.
    """
    import re

    # LFM backbone embed_tokens -> embedding.text_embed
    if key.startswith("lfm.embed_tokens."):
        return key.replace("lfm.embed_tokens.", "embedding.text_embed.")

    # LFM backbone layers -> decoder.layers
    if key.startswith("lfm."):
        rest = key[len("lfm.") :]
        # model.layers.N patterns
        m = re.match(r"^layers\.(\d+)\.(.+)$", rest)
        if m:
            idx = m.group(1)
            layer_rest = m.group(2)
            # Conv weight nesting
            layer_rest = layer_rest.replace("conv.conv.weight", "conv.conv_weight")
            layer_rest = layer_rest.replace("conv.conv.bias", "conv.conv_bias")
            # MLP: w1->gate_proj, w3->up_proj, w2->down_proj
            layer_rest = layer_rest.replace("feed_forward.w1.", "feed_forward.gate_proj.")
            layer_rest = layer_rest.replace("feed_forward.w3.", "feed_forward.up_proj.")
            layer_rest = layer_rest.replace("feed_forward.w2.", "feed_forward.down_proj.")
            # Attention: out_proj->o_proj, layernorm->norm
            layer_rest = layer_rest.replace("self_attn.out_proj.", "self_attn.o_proj.")
            layer_rest = layer_rest.replace("self_attn.q_layernorm.", "self_attn.q_norm.")
            layer_rest = layer_rest.replace("self_attn.k_layernorm.", "self_attn.k_norm.")
            return f"decoder.layers.{idx}.{layer_rest}"

        # lfm.norm -> decoder.norm
        return f"decoder.{rest}"

    # Conformer -> audio_encoder.encoder
    if key.startswith("conformer."):
        return key.replace("conformer.", "audio_encoder.encoder.")

    # Audio adapter -> audio_encoder.adapter
    if key.startswith("audio_adapter."):
        rest = key[len("audio_adapter.") :]
        # audio_adapter.model.0.* -> adapter.up_proj.*
        rest = rest.replace("model.0.", "up_proj.")
        # audio_adapter.model.3.* -> adapter.down_proj.*
        rest = rest.replace("model.3.", "down_proj.")
        # Skip batch norm (model.1.*)
        if "model.1." in key or "model.2." in key:
            return None
        return f"audio_encoder.adapter.{rest}"

    # Audio embedding
    if key.startswith("audio_embedding."):
        rest = key[len("audio_embedding.") :]
        rest = rest.replace("embedding.", "audio_embed.")
        return f"embedding.{rest}"

    # Depthformer layers
    if key.startswith("depthformer.layers."):
        rest = key[len("depthformer.layers.") :]
        m = re.match(r"^(\d+)\.(.+)$", rest)
        if m:
            idx = m.group(1)
            layer_rest = m.group(2)
            # operator.qkv_proj -> self_attn.qkv_proj (if fused)
            # operator.out_proj -> self_attn.o_proj
            layer_rest = layer_rest.replace("operator.", "self_attn.")
            layer_rest = layer_rest.replace("self_attn.out_proj.", "self_attn.o_proj.")
            layer_rest = layer_rest.replace(
                "self_attn.bounded_attention.q_layernorm.",
                "self_attn.q_norm.",
            )
            layer_rest = layer_rest.replace(
                "self_attn.bounded_attention.k_layernorm.",
                "self_attn.k_norm.",
            )
            # MLP renames
            layer_rest = layer_rest.replace("feed_forward.w1.", "feed_forward.gate_proj.")
            layer_rest = layer_rest.replace("feed_forward.w3.", "feed_forward.up_proj.")
            layer_rest = layer_rest.replace("feed_forward.w2.", "feed_forward.down_proj.")
            return f"audio_decoder.layers.{idx}.{layer_rest}"
        return None

    # Depth linear
    if key.startswith("depth_linear."):
        return key.replace("depth_linear.", "audio_decoder.depth_linear.")

    # Depth embeddings
    if key.startswith("depth_embeddings."):
        return key.replace("depth_embeddings.", "audio_decoder.depth_embeddings.")

    # Embedding norm (for depthformer output)
    if key.startswith("embedding_norm."):
        return key.replace("embedding_norm.", "audio_decoder.embedding_norm.")

    return key
