# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HunYuan VL-MoT vision-language model — 3-model split.

Replicates ``tencent/HY-Embodied-0.5-X`` (HunYuanVLMoTForConditionalGeneration).

Architecture:
- **Vision encoder**: 27-block ViT (fused QKV, LayerNorm) with spatial merger
- **Embedding**: Token lookup + image feature scatter at placeholder positions
- **Decoder**: 32-layer GQA (16Q/4KV heads, head_dim=128) with QK-norm and
  Mixture-of-Tokens (MoT) dual-pathway routing

MoT dual pathway:
    Every decoder layer has two sets of projections — standard (text) and
    ``_v`` (vision).  During prefill, vision token positions are routed
    through ``_v`` weights while text tokens use standard weights.  Q/K/V
    are merged per-token **before** the attention operation so that all
    tokens attend to all tokens in a shared KV space.  The KV cache
    contains mixed text/vision K/V so that subsequent decode steps
    correctly attend to image context.

    QK norms (``query_layernorm`` / ``key_layernorm``) are shared between
    both pathways.

HuggingFace weight layout::

    model.visual.vision_tower.patch_embed.proj.{weight,bias}
    model.visual.vision_tower.pos_embed
    model.visual.vision_tower.blocks.{i}.attn.qkv.{weight,bias}
    model.visual.vision_tower.blocks.{i}.attn.proj.{weight,bias}
    model.visual.vision_tower.blocks.{i}.mlp.fc1/fc2.{weight,bias}
    model.visual.vision_tower.blocks.{i}.norm1/norm2.{weight,bias}
    model.visual.merger.proj1/proj2.{weight,bias}
    model.visual.merger.pooler.predictor.{0,2}.{weight,bias}
    model.language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj_v.weight
    model.language_model.model.layers.{i}.self_attn.query_layernorm.weight
    model.language_model.model.layers.{i}.self_attn.key_layernorm.weight
    model.language_model.model.layers.{i}.{input,post_attention}_layernorm.weight
    model.language_model.model.layers.{i}.{input,post_attention}_layernorm_v.weight
    model.language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.language_model.model.layers.{i}.mlp_v.{gate,up,down}_proj.weight
    model.language_model.model.embed_tokens.weight
    model.language_model.model.norm.weight
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Embedding,
    Linear,
    MLPMultiModalProjector,
    VisionModel,
)
from mobius.components._attention import _apply_attention
from mobius.components._mlp import MLP
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import (
    apply_rotary_pos_emb,
    initialize_rope,
)

if TYPE_CHECKING:
    import onnx_ir as ir


# ── Vision encoder ──────────────────────────────────────────────────────


class _HunYuanVLMoTVisionEncoderModel(nn.Module):
    """ViT vision tower + merger projector.

    The vision tower is a standard ViT (SigLIP-style) wrapped by
    :class:`VisionModel`.  The merger projects vision features from the
    vision hidden dimension to the text hidden dimension using a two-layer
    MLP (proj1 → GELU → proj2).

    HF weight prefix: ``model.visual.*``

    .. note::

       Weight renaming is handled entirely by
       :meth:`HunYuanVLMoTModel.preprocess_weights` — the sub-model's
       ``preprocess_weights`` is not called in the standard build flow.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "VisionConfig is required"
        self.vision_tower = VisionModel(config)
        # Merger: two linear projections that map
        # vision_hidden → text_hidden
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=vc.hidden_size or config.hidden_size,
            text_hidden_size=config.hidden_size,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        vision_features = self.vision_tower(op, pixel_values)
        return self.multi_modal_projector(op, vision_features)


# ── Embedding ───────────────────────────────────────────────────────────


class _HunYuanVLMoTEmbeddingModel(nn.Module):
    """Token embedding + image feature scatter.

    Replaces image placeholder tokens with projected vision features
    using the standard Gather + Where pattern.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = (
            config.image_token_id
            or (config.vision.image_token_id if config.vision else None)
            or 0
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ):
        text_embeds = self.embed_tokens(op, input_ids)

        # Build a mask of image-token positions
        image_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        # CumSum-based indexing into image_features
        mask_int = op.Cast(image_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, 1)
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        # Pad with one zero row so Gather is valid for text-only inputs
        pad_row = op.Expand(
            op.CastLike(0.0, image_features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(image_features, start=1, end=2),
                axis=0,
            ),
        )
        padded_features = op.Concat(image_features, pad_row, axis=0)

        gathered = op.Gather(padded_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)


# ── MoT Attention ───────────────────────────────────────────────────────


class _MoTAttention(nn.Module):
    """Mixture-of-Tokens attention with dual Q/K/V/O projections.

    Text tokens use standard projections; vision tokens use ``_v``
    projections.  Q/K/V are merged per-token **before** the single
    attention operation so that all tokens attend to all tokens in a
    shared KV space.  The QK norms (``q_norm`` / ``k_norm``) are shared
    between both pathways.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        h = config.hidden_size
        hd = config.head_dim
        nq = config.num_attention_heads
        nkv = config.num_key_value_heads

        self.num_attention_heads = nq
        self.num_key_value_heads = nkv
        self.head_dim = hd
        self.scaling = hd**-0.5

        bias = config.attn_qkv_bias
        o_bias = config.attn_o_bias

        # Text pathway projections
        self.q_proj = Linear(h, nq * hd, bias=bias)
        self.k_proj = Linear(h, nkv * hd, bias=bias)
        self.v_proj = Linear(h, nkv * hd, bias=bias)
        self.o_proj = Linear(nq * hd, h, bias=o_bias)

        # Vision pathway projections (_v)
        self.q_proj_v = Linear(h, nq * hd, bias=bias)
        self.k_proj_v = Linear(h, nkv * hd, bias=bias)
        self.v_proj_v = Linear(h, nkv * hd, bias=bias)
        self.o_proj_v = Linear(nq * hd, h, bias=o_bias)

        # Shared QK norms (applied to both pathways)
        self.q_norm = RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(hd, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states_text: ir.Value,
        hidden_states_vision: ir.Value,
        modality_mask_3d: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None,
        past_key_value: tuple | None,
    ):
        # Compute Q/K/V for both pathways
        q_text = self.q_proj(op, hidden_states_text)
        k_text = self.k_proj(op, hidden_states_text)
        v_text = self.v_proj(op, hidden_states_text)

        q_vision = self.q_proj_v(op, hidden_states_vision)
        k_vision = self.k_proj_v(op, hidden_states_vision)
        v_vision = self.v_proj_v(op, hidden_states_vision)

        # Merge Q/K/V per-token: vision tokens get _v projection output
        q = op.Where(modality_mask_3d, q_vision, q_text)
        k = op.Where(modality_mask_3d, k_vision, k_text)
        v = op.Where(modality_mask_3d, v_vision, v_text)

        # Apply shared QK norms (per-head: reshape to 4D, norm, reshape back)
        q = op.Reshape(q, [0, 0, -1, self.head_dim])
        k = op.Reshape(k, [0, 0, -1, self.head_dim])
        q = self.q_norm(op, q)
        k = self.k_norm(op, k)
        q = op.Reshape(q, [0, 0, -1])
        k = op.Reshape(k, [0, 0, -1])

        # Apply rotary position embeddings
        if position_embeddings is not None:
            q = apply_rotary_pos_emb(
                op,
                x=q,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=0,
            )
            k = apply_rotary_pos_emb(
                op,
                x=k,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=0,
            )

        # Single attention over merged Q/K/V
        past_k = past_key_value[0] if past_key_value else None
        past_v = past_key_value[1] if past_key_value else None
        attn_output, present_k, present_v = _apply_attention(
            op,
            q,
            k,
            v,
            attention_bias,
            past_k,
            past_v,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
        )

        # Route output through text or vision o_proj
        o_text = self.o_proj(op, attn_output)
        o_vision = self.o_proj_v(op, attn_output)
        output = op.Where(modality_mask_3d, o_vision, o_text)

        return output, (present_k, present_v)


# ── MoT Decoder Layer ──────────────────────────────────────────────────


class _MoTDecoderLayer(nn.Module):
    """Decoder layer with MoT dual-pathway routing.

    Each sub-layer (attention, MLP) has text and vision variants.
    Routing is controlled by ``modality_mask_3d`` (BOOL [B, S, 1]).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        h = config.hidden_size
        eps = config.rms_norm_eps

        # Dual input layernorms
        self.input_layernorm = RMSNorm(h, eps=eps)
        self.input_layernorm_v = RMSNorm(h, eps=eps)

        # MoT attention (dual projections, shared QK norms)
        self.self_attn = _MoTAttention(config)

        # Dual post-attention layernorms
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.post_attention_layernorm_v = RMSNorm(h, eps=eps)

        # Dual MLPs
        self.mlp = MLP(config)
        self.mlp_v = MLP(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        modality_mask_3d: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None,
        past_key_value: tuple | None,
    ):
        residual = hidden_states

        # Dual input norms, merge for attention input
        normed_text = self.input_layernorm(op, hidden_states)
        normed_vision = self.input_layernorm_v(op, hidden_states)

        # MoT attention: merges Q/K/V internally, single attention call
        attn_output, present_kv = self.self_attn(
            op,
            hidden_states_text=normed_text,
            hidden_states_vision=normed_vision,
            modality_mask_3d=modality_mask_3d,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, attn_output)

        # Dual post-attention norms → dual MLPs → merge
        residual = hidden_states
        normed_text = self.post_attention_layernorm(op, hidden_states)
        normed_vision = self.post_attention_layernorm_v(op, hidden_states)
        mlp_text = self.mlp(op, normed_text)
        mlp_vision = self.mlp_v(op, normed_vision)
        mlp_output = op.Where(modality_mask_3d, mlp_vision, mlp_text)
        hidden_states = op.Add(residual, mlp_output)

        return hidden_states, present_kv


# ── MoT Text Model ────────────────────────────────────────────────────


class _MoTTextModel(nn.Module):
    """Text model backbone with MoT decoder layers.

    Threads ``modality_mask`` through all layers for per-token routing.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [_MoTDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        modality_mask_3d: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(op, position_ids)
        # Pass attention_mask as a bool padding mask — _apply_attention
        # uses is_causal=1 so only padding information is needed.
        if attention_mask is not None:
            attention_bias = op.Cast(attention_mask, to=9)  # BOOL
        else:
            attention_bias = None

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                modality_mask_3d=modality_mask_3d,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


# ── Decoder ─────────────────────────────────────────────────────────────


class _HunYuanVLMoTDecoderModel(nn.Module):
    """MoT decoder with dual-pathway routing.

    Takes ``inputs_embeds`` and ``input_ids`` — the latter is used to
    derive the modality mask (vision tokens identified by image_token_id).

    Weight renaming is handled by
    :meth:`HunYuanVLMoTModel.preprocess_weights`.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.image_token_id = (
            config.image_token_id
            or (config.vision.image_token_id if config.vision else None)
            or 0
        )
        self.model = _MoTTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        input_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        # Derive modality mask from input_ids: True where vision tokens
        modality_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        modality_mask_3d = op.Unsqueeze(modality_mask, [-1])  # [B, S, 1]

        hidden_states, present_key_values = self.model(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            modality_mask_3d=modality_mask_3d,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


# ── Top-level model ─────────────────────────────────────────────────────


class HunYuanVLMoTModel(nn.Module):
    """HunYuan VL-MoT vision-language model (3-model split).

    Builds three ONNX models for ORT GenAI deployment:

    - **decoder**: text decoder taking ``inputs_embeds``
    - **vision_encoder**: ViT + merger projector
    - **embedding**: token embedding + image feature fusion
    """

    default_task: str = "hunyuan-vl-mot"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _HunYuanVLMoTDecoderModel(config)
        self.vision_encoder = _HunYuanVLMoTVisionEncoderModel(config)
        self.embedding = _HunYuanVLMoTEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "HunYuanVLMoTModel uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Top-level HF prefix: "model." wraps everything.
        # Strip it first: model.language_model.* → language_model.*
        #                  model.visual.*        → visual.*
        stripped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key[len("model.") :] if key.startswith("model.") else key
            stripped[new_key] = value

        result: dict[str, torch.Tensor] = {}

        for key, value in stripped.items():
            if key.startswith("visual."):
                self._route_vision_weight(key, value, result)
            elif key.startswith("language_model."):
                self._route_decoder_weight(key, value, result)

        # The generic VisionModel always creates a post_layernorm but the
        # HF model doesn't have one.  Provide identity LayerNorm weights
        # (weight=1, bias=0) so the layer is a no-op.
        pln_prefix = "vision_encoder.vision_tower.vision_model.post_layernorm"
        if f"{pln_prefix}.weight" not in result:
            hidden = self.config.vision.hidden_size or self.config.hidden_size
            result[f"{pln_prefix}.weight"] = torch.ones(hidden)
            result[f"{pln_prefix}.bias"] = torch.zeros(hidden)

        return result

    def _route_vision_weight(
        self,
        key: str,
        value: torch.Tensor,
        result: dict[str, torch.Tensor],
    ) -> None:
        """Route visual.* weights to vision_encoder sub-model."""
        suffix = key[len("visual.") :]

        if suffix.startswith("merger."):
            # merger.proj1 → multi_modal_projector.linear_1
            # merger.proj2 → multi_modal_projector.linear_2
            merger_key = suffix[len("merger.") :]
            merger_key = merger_key.replace("proj1.", "linear_1.")
            merger_key = merger_key.replace("proj2.", "linear_2.")
            # Skip pooler weights (not in standard MLP projector)
            if "pooler." in merger_key:
                return
            result[f"vision_encoder.multi_modal_projector.{merger_key}"] = value
            return

        # Vision tower weight renames:
        # vision_tower.blocks.N → vision_tower.vision_model.encoder.layers.N
        new_key = suffix.replace(
            "vision_tower.blocks.",
            "vision_tower.vision_model.encoder.layers.",
        )
        # attn.proj → self_attn.out_proj
        new_key = new_key.replace(".attn.proj.", ".self_attn.out_proj.")
        # norm1/norm2 → layer_norm1/layer_norm2
        new_key = new_key.replace(".norm1.", ".layer_norm1.")
        new_key = new_key.replace(".norm2.", ".layer_norm2.")
        # mlp.fc1/fc2 → mlp.up_proj/down_proj
        new_key = new_key.replace(".mlp.fc1.", ".mlp.up_proj.")
        new_key = new_key.replace(".mlp.fc2.", ".mlp.down_proj.")
        # patch_embed.proj → embeddings.patch_embedding
        new_key = new_key.replace(
            "vision_tower.patch_embed.proj.",
            "vision_tower.vision_model.embeddings.patch_embedding.",
        )
        # pos_embed → embeddings.position_embedding.weight
        # HF pos_embed is [1, num_patches, hidden] — squeeze batch dim
        if "vision_tower.pos_embed" in suffix:
            result[
                "vision_encoder.vision_tower.vision_model.embeddings.position_embedding.weight"
            ] = value.squeeze(0)
            return

        # Split fused QKV into separate Q, K, V
        if ".attn.qkv." in suffix:
            layer_prefix = new_key.split(".attn.qkv.")[0]
            param = "weight" if "weight" in key else "bias"
            chunks = torch.chunk(value, 3, dim=0)
            result[f"vision_encoder.{layer_prefix}.self_attn.q_proj.{param}"] = chunks[0]
            result[f"vision_encoder.{layer_prefix}.self_attn.k_proj.{param}"] = chunks[1]
            result[f"vision_encoder.{layer_prefix}.self_attn.v_proj.{param}"] = chunks[2]
            return

        result[f"vision_encoder.{new_key}"] = value

    def _route_decoder_weight(
        self,
        key: str,
        value: torch.Tensor,
        result: dict[str, torch.Tensor],
    ) -> None:
        """Route language_model.* weights to decoder and embedding."""
        # Strip language_model. prefix → model.layers.0.*, lm_head.*, etc.
        suffix = key[len("language_model.") :]

        # Duplicate embed_tokens to embedding sub-model
        if "embed_tokens" in suffix:
            embed_key = suffix[len("model.") :] if suffix.startswith("model.") else suffix
            result[f"embedding.{embed_key}"] = value

        # MoT _v pathway: HF names match our module names directly
        # e.g. self_attn.q_proj_v, mlp_v.gate_proj, input_layernorm_v

        # QK-norm rename: query_layernorm → q_norm, key_layernorm → k_norm
        renamed = suffix.replace(".query_layernorm.", ".q_norm.").replace(
            ".key_layernorm.", ".k_norm."
        )
        result[f"decoder.{renamed}"] = value

        # Weight tying: embed_tokens → lm_head
        if self.config.tie_word_embeddings and "embed_tokens.weight" in key:
            result["decoder.lm_head.weight"] = value
