# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
    depthformer.      -> audio_decoder.depthformer
    depth_linear.     -> audio_decoder.depth_linear
    depth_embeddings. -> audio_decoder.depth_embeddings
    embedding_norm.   -> audio_decoder.embedding_norm

Reference: ``liquid_audio.model.lfm2_audio.LFM2AudioModel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Lfm2AudioConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    MLP,
    Attention,
    ConformerEncoder,
    Embedding,
    LayerNorm,
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


class _Lfm2AudioAdapter(nn.Module):
    """Audio adapter MLP for LFM2-Audio.

    Matches the HuggingFace ``audio_adapter`` Sequential layout exactly::

        model.0 = LayerNorm(encoder_dim)        # weight + bias both [encoder_dim]
        model.1 = Linear(encoder_dim, hidden_size)
        model.2 = GELU                          # no parameters
        model.3 = Linear(hidden_size, hidden_size)

    Output dimension is ``hidden_size`` (the backbone hidden dim), not
    ``encoder_dim`` — i.e. this is *not* a residual MLP, it's a projection
    from the conformer's hidden width up to the LM backbone's hidden width
    followed by a hidden-size-square refinement Linear.
    """

    def __init__(self, encoder_dim: int, hidden_size: int):
        super().__init__()
        self.pre_norm = LayerNorm(encoder_dim, eps=1e-5)
        self.up_proj = Linear(encoder_dim, hidden_size, bias=False)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = self.pre_norm(op, x)
        x = self.up_proj(op, x)
        x = op.Gelu(x)
        return self.out_proj(op, x)


class _Lfm2AudioEncoder(nn.Module):
    """ConformerEncoder + adapter MLP for mel -> LFM hidden_size projection.

    HuggingFace weight mapping (handled by ``preprocess_weights``)::

        conformer.* -> encoder.*
        audio_adapter.model.0.{weight,bias} -> adapter.pre_norm.{weight,bias}
        audio_adapter.model.1.weight        -> adapter.up_proj.weight   (no bias in HF)
        audio_adapter.model.3.{weight,bias} -> adapter.out_proj.{weight,bias}
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        audio = config.audio
        assert audio is not None

        encoder_dim = audio.attention_dim or audio.d_model or 512
        self.encoder = ConformerEncoder(
            input_size=audio.num_mel_bins or 128,
            attention_dim=encoder_dim,
            attention_heads=audio.attention_heads or audio.encoder_attention_heads or 8,
            num_blocks=audio.num_blocks or audio.encoder_layers or 17,
            linear_units=(audio.linear_units or audio.encoder_ffn_dim or 2048),
            kernel_size=audio.kernel_size or 9,
            conv_channels=audio.conv_channels or 256,
            t5_bias_max_distance=audio.t5_bias_max_distance or 500,
        )
        # Adapter: encoder_dim -> hidden_size
        self.adapter = _Lfm2AudioAdapter(encoder_dim, config.hidden_size)

    def forward(self, op: OpBuilder, input_features: ir.Value):
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

    Returns text token embeddings for the backbone. Audio codebook embeddings
    are handled by the ``audio_decoder``'s ``depth_embeddings`` — not here.

    Weight names (HF)::

        lfm.embed_tokens.weight -> text_embed.weight
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        self.text_embed = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def forward(
        self,
        op: OpBuilder,
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
        op: OpBuilder,
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


def _depthformer_intermediate_size(config: Lfm2AudioConfig) -> int:
    """Return the depthformer SwiGLU intermediate size.

    Uses ``config.depthformer_intermediate_size`` when set; otherwise
    derives it from the depthformer hidden dim using the same
    ``block_auto_adjust_ff_dim`` formula as the LFM2 backbone:
    ``round_up(2 * 4 * dim / 3, 256)``.

    For ``depthformer_dim=1024`` this yields ``2816``, matching the
    LFM2-Audio-1.5B checkpoint's ``feed_forward.w*`` rows.
    """
    if config.depthformer_intermediate_size is not None:
        return int(config.depthformer_intermediate_size)
    dim = config.depthformer_dim
    intermediate = int(2 * (4 * dim) / 3)
    multiple_of = 256
    return multiple_of * ((intermediate + multiple_of - 1) // multiple_of)


class _DepthformerLayer(nn.Module):
    """Single depthformer layer with RMSNorm, GQA Attention, and SwiGLU MLP.

    Architecture: RMSNorm -> GQA Attention (head_dim=32, kv_heads=8) ->
    residual -> RMSNorm -> SwiGLU MLP -> residual.

    Mirrors the HuggingFace ``depthformer.layers.K`` block layout::

        operator_norm    (RMSNorm)             -> operator_norm
        operator         (BoundedAttention)    -> self_attn
          .qkv_proj      [num_q*hd + 2*num_kv*hd, dim]
                                                -> q_proj, k_proj, v_proj
          .out_proj      [dim, num_q*hd]       -> o_proj
          .bounded_attention.q_layernorm [hd]  -> q_norm  (per-head RMSNorm)
          .bounded_attention.k_layernorm [hd]  -> k_norm
        ffn_norm         (RMSNorm)             -> ffn_norm
        feed_forward     (SwiGLU MLP)          -> feed_forward
          .w1 [I, dim]                          -> gate_proj
          .w3 [I, dim]                          -> up_proj
          .w2 [dim, I]                          -> down_proj

    Note: ``head_dim`` is **not** ``depthformer_dim // depthformer_heads``.
    LFM2-Audio hardcodes ``head_dim=32`` (so ``num_q = dim // 32``) with
    GQA ``kv_heads=8``.
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        from mobius._configs import ArchitectureConfig

        depthformer_dim = config.depthformer_dim
        head_dim = config.depthformer_head_dim
        num_q_heads = depthformer_dim // head_dim
        num_kv_heads = config.depthformer_kv_heads
        intermediate = _depthformer_intermediate_size(config)

        attn_config = ArchitectureConfig(
            hidden_size=depthformer_dim,
            intermediate_size=intermediate,
            num_attention_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_act="silu",
            attn_qkv_bias=False,
            attn_o_bias=False,
            attn_qk_norm=True,
            attn_qk_norm_full=False,
            rms_norm_eps=1e-5,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.operator_norm = RMSNorm(depthformer_dim, eps=1e-5)
        self.self_attn = Attention(attn_config)
        self.ffn_norm = RMSNorm(depthformer_dim, eps=1e-5)
        self.feed_forward = MLP(attn_config)

    def forward(
        self,
        op: OpBuilder,
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


class _DepthCodebookHead(nn.Module):
    """Per-codebook embedding + norm + output head triple.

    Mirrors the HuggingFace ``depth_embeddings.K`` layout::

        embedding       [audio_vocab_size, dim]    Embedding layer
        embedding_norm  [dim]                      RMSNorm before to_logits
        to_logits       [audio_vocab_size, dim]    Linear (tied with embedding
                                                   when ``depthformer_tie=True``)

    The ``embedding`` is used externally (host code) to build the
    ``prev_embedding`` input fed into the depthformer. The ``embedding_norm``
    and ``to_logits`` weights are gathered at runtime through stacked tensors
    so the depthformer forward can select them by ``codebook_idx``.
    """

    def __init__(self, vocab_size: int, dim: int, eps: float = 1e-5):
        super().__init__()
        self.embedding = Embedding(vocab_size, dim)
        self.embedding_norm = RMSNorm(dim, eps=eps)
        self.to_logits = Linear(dim, vocab_size, bias=False)


class _Lfm2AudioDecoderModule(nn.Module):
    """Depthformer audio decoder for per-codebook token prediction.

    Takes the backbone hidden state + the previous codebook embedding,
    runs through depthformer layers, and produces logits for the current
    codebook.

    Architecture::

        depth_linear(backbone_hidden) -> split by codebook_idx ->
        + prev_embedding -> depthformer layers ->
        per-codebook embedding_norm -> per-codebook to_logits -> logits

    Each codebook has its own ``embedding`` / ``embedding_norm`` /
    ``to_logits`` triple (``depth_embeddings.K``), all stored as separate
    state-dict entries to match the HF checkpoint. At runtime, the
    per-codebook norm and head are selected by a single ``Gather`` against
    stacked tensors assembled in :meth:`preprocess_weights`.
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        depthformer_dim = config.depthformer_dim

        # Project backbone hidden -> per-codebook inputs.
        # depth_linear: (hidden_size) -> (codebooks * depthformer_dim).
        self.depth_linear = Linear(
            config.hidden_size,
            config.num_codebooks * depthformer_dim,
            bias=True,
        )

        # Depthformer layers.
        self.layers = nn.ModuleList([])
        for _ in range(config.depthformer_layers):
            self.layers.append(_DepthformerLayer(config))

        # Per-codebook embedding + norm + output head triples. These mirror
        # the HF ``depth_embeddings.K`` modules. The ``embedding`` weights
        # live here for host-side construction of ``prev_embedding`` even
        # though the audio_decoder forward never consumes them directly.
        self.depth_embeddings = nn.ModuleList(
            [
                _DepthCodebookHead(config.audio_vocab_size, depthformer_dim, eps=1e-5)
                for _ in range(config.num_codebooks)
            ]
        )

        # Stacked per-codebook tensors used by forward via Gather. Assembled
        # from the per-codebook triples in ``preprocess_weights``. Same
        # pattern as ``stacked_head_weights`` in :mod:`mobius.models.moshi`.
        self.stacked_norm_weights = nn.Parameter([config.num_codebooks, depthformer_dim])
        # Output head weights: tied with ``embedding.weight`` when
        # ``depthformer_tie=True``, but still shipped as a separate stacked
        # tensor so the ONNX graph remains tie-agnostic.
        self.stacked_head_weights = nn.Parameter(
            [config.num_codebooks, config.audio_vocab_size, depthformer_dim]
        )

        self._depthformer_dim = depthformer_dim
        self._num_codebooks = config.num_codebooks

        # Build a separate RoPE for depthformer (per-step head_dim=32).
        from mobius._configs import ArchitectureConfig

        head_dim = config.depthformer_head_dim
        num_q_heads = depthformer_dim // head_dim
        rope_config = ArchitectureConfig(
            hidden_size=depthformer_dim,
            num_attention_heads=num_q_heads,
            head_dim=head_dim,
            rope_theta=config.rope_theta,
            rope_type="default",
            max_position_embeddings=config.max_position_embeddings,
        )
        self.rotary_emb = initialize_rope(rope_config)

    def forward(
        self,
        op: OpBuilder,
        backbone_hidden: ir.Value,
        prev_embedding: ir.Value,
        codebook_idx: ir.Value,
        past_key_values: list | None = None,
    ):
        """Forward pass for single-codebook prediction.

        Args:
            backbone_hidden: (B, 1, hidden_size) from LFM2 decoder.
            prev_embedding: (B, 1, depthformer_dim) from previous codebook.
            codebook_idx: scalar int — which codebook to predict.
            past_key_values: depthformer KV cache.

        Returns:
            (codebook_logits, present_key_values)
        """
        # Project backbone hidden to all codebook inputs.
        # (B, 1, hidden_size) -> (B, 1, codebooks * depthformer_dim).
        projected = self.depth_linear(op, backbone_hidden)

        # Reshape to (B, codebooks, depthformer_dim) for gathering.
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

        # Gather the codebook_idx slice along axis 1: (B, 1, depthformer_dim).
        idx_3d = op.Reshape(codebook_idx, op.Constant(value_ints=[1, 1, 1]))
        batch_dim = op.Shape(projected_3d, start=0, end=1)
        expand_shape = op.Concat(
            batch_dim,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[self._depthformer_dim]),
            axis=0,
        )
        idx_expanded = op.Expand(idx_3d, expand_shape)
        depthformer_input = op.GatherElements(projected_3d, idx_expanded, axis=1)

        # Add previous codebook embedding (depth autoregressive context).
        hidden_states = op.Add(depthformer_input, prev_embedding)

        # Position IDs for depthformer (single step: just codebook_idx),
        # shape (B, 1) derived from the runtime batch dim.
        batch_dim = op.Shape(hidden_states, start=0, end=1)
        one_dim = op.Constant(value_ints=[1])
        position_shape = op.Concat(batch_dim, one_dim, axis=0)
        position_ids = op.Reshape(codebook_idx, position_shape)
        position_embeddings = self.rotary_emb(op, position_ids)

        # Run depthformer layers.
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

        # Per-codebook RMSNorm: gather weight (depthformer_dim,) by codebook_idx.
        norm_weight = op.Gather(self.stacked_norm_weights, codebook_idx, axis=0)
        hidden_states = op.RMSNormalization(
            hidden_states,
            norm_weight,
            epsilon=1e-5,
            axis=-1,
        )

        # Per-codebook output head: gather to_logits weight
        # (audio_vocab_size, depthformer_dim) and apply.
        head_weight = op.Gather(self.stacked_head_weights, codebook_idx, axis=0)
        head_weight_3d = op.Unsqueeze(head_weight, [0])
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
            lfm.embed_tokens.*  -> embedding.text_embed.*
            lfm.*               -> decoder.* (backbone layers)
            conformer.*         -> audio_encoder.encoder.*
            audio_adapter.*     -> audio_encoder.adapter.*
            audio_embedding.*   -> skipped (host code consumes raw HF tensor)
            depthformer.*       -> audio_decoder.* (depthformer layers)
            depth_linear.*      -> audio_decoder.depth_linear.*
            depth_embeddings.K.{embedding,embedding_norm,to_logits}.weight
                                -> audio_decoder.depth_embeddings.K.*

        Special-case transforms:
            * Each depthformer layer's fused ``operator.qkv_proj.weight``
              is split into ``self_attn.{q,k,v}_proj.weight`` along the row
              axis using ``num_q*head_dim`` / ``num_kv*head_dim`` chunks.
            * Per-codebook ``embedding_norm.weight`` tensors are stacked
              into ``audio_decoder.stacked_norm_weights``.
            * Per-codebook ``to_logits.weight`` tensors (which are tied to
              the corresponding ``embedding.weight`` in HF when
              ``depthformer_tie=True``) are stacked into
              ``audio_decoder.stacked_head_weights``.
        """
        if self.config.tie_word_embeddings:
            tie_word_embeddings(
                state_dict,
                embed_key="lfm.embed_tokens.weight",
                head_key="lfm.lm_head.weight",
            )

        # Split fused depthformer qkv_proj into q/k/v.
        head_dim = self.config.depthformer_head_dim
        num_q_heads = self.config.depthformer_dim // head_dim
        num_kv_heads = self.config.depthformer_kv_heads
        q_rows = num_q_heads * head_dim
        kv_rows = num_kv_heads * head_dim
        for i in range(self.config.depthformer_layers):
            qkv_key = f"depthformer.layers.{i}.operator.qkv_proj.weight"
            if qkv_key in state_dict:
                qkv = state_dict.pop(qkv_key)
                state_dict[f"depthformer.layers.{i}.operator.q_proj.weight"] = qkv[:q_rows]
                state_dict[f"depthformer.layers.{i}.operator.k_proj.weight"] = qkv[
                    q_rows : q_rows + kv_rows
                ]
                state_dict[f"depthformer.layers.{i}.operator.v_proj.weight"] = qkv[
                    q_rows + kv_rows : q_rows + 2 * kv_rows
                ]

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_lfm2_audio_weight(key)
            if new_key is not None:
                new_state_dict[new_key] = value

        # Stack per-codebook norm + head weights for Gather-by-codebook in
        # the audio_decoder forward.
        norm_weights = []
        head_weights = []
        for i in range(self.config.num_codebooks):
            nkey = f"audio_decoder.depth_embeddings.{i}.embedding_norm.weight"
            hkey = f"audio_decoder.depth_embeddings.{i}.to_logits.weight"
            if nkey in new_state_dict:
                norm_weights.append(new_state_dict[nkey])
            if hkey in new_state_dict:
                head_weights.append(new_state_dict[hkey])
        if norm_weights:
            new_state_dict["audio_decoder.stacked_norm_weights"] = torch.stack(
                norm_weights, dim=0
            )
        if head_weights:
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

    # Audio adapter: HF Sequential -> our named modules
    #   model.0 = LayerNorm(encoder_dim)              -> pre_norm
    #   model.1 = Linear(encoder_dim, hidden_size)    -> up_proj (no bias)
    #   model.2 = GELU (no params)
    #   model.3 = Linear(hidden_size, hidden_size)    -> out_proj
    if key.startswith("audio_adapter."):
        rest = key[len("audio_adapter.") :]
        if rest.startswith("model.2."):
            return None  # GELU has no params
        rest = rest.replace("model.0.", "pre_norm.")
        rest = rest.replace("model.1.", "up_proj.")
        rest = rest.replace("model.3.", "out_proj.")
        return f"audio_encoder.adapter.{rest}"

    # Audio embedding weights live in audio_decoder.depth_embeddings at runtime.
    # The embedding sub-model only handles text tokens — skip these.
    if key.startswith("audio_embedding."):
        return None

    # Depthformer layers
    if key.startswith("depthformer.layers."):
        rest = key[len("depthformer.layers.") :]
        m = re.match(r"^(\d+)\.(.+)$", rest)
        if m:
            idx = m.group(1)
            layer_rest = m.group(2)
            # operator.{q,k,v,out}_proj -> self_attn.{q,k,v,o}_proj
            # (qkv_proj was already split into q/k/v_proj earlier).
            layer_rest = layer_rest.replace("operator.out_proj.", "self_attn.o_proj.")
            layer_rest = layer_rest.replace(
                "operator.bounded_attention.q_layernorm.",
                "self_attn.q_norm.",
            )
            layer_rest = layer_rest.replace(
                "operator.bounded_attention.k_layernorm.",
                "self_attn.k_norm.",
            )
            layer_rest = layer_rest.replace("operator.q_proj.", "self_attn.q_proj.")
            layer_rest = layer_rest.replace("operator.k_proj.", "self_attn.k_proj.")
            layer_rest = layer_rest.replace("operator.v_proj.", "self_attn.v_proj.")
            # MLP renames
            layer_rest = layer_rest.replace("feed_forward.w1.", "feed_forward.gate_proj.")
            layer_rest = layer_rest.replace("feed_forward.w3.", "feed_forward.up_proj.")
            layer_rest = layer_rest.replace("feed_forward.w2.", "feed_forward.down_proj.")
            return f"audio_decoder.layers.{idx}.{layer_rest}"
        return None

    # Depth linear
    if key.startswith("depth_linear."):
        return key.replace("depth_linear.", "audio_decoder.depth_linear.")

    # Per-codebook depth embedding triples: depth_embeddings.K.{embedding,
    # embedding_norm, to_logits}.weight -> audio_decoder.depth_embeddings.K.*.
    if key.startswith("depth_embeddings."):
        return key.replace("depth_embeddings.", "audio_decoder.depth_embeddings.")

    # Legacy top-level embedding_norm (not present in LFM2-Audio-1.5B but
    # tolerated for older checkpoints). The current model moves the
    # depthformer output norm into each per-codebook triple, so this key
    # has nowhere to land — drop it rather than throwing.
    if key.startswith("embedding_norm."):
        return None

    return key
