# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Muse Glimmer text and vision-language models.

Replicates HuggingFace ``MuseGlimmerForConditionalGeneration`` as three ONNX
graphs: a standard-RoPE decoder, a dynamic packed vision encoder, and an
embedding/image-feature mixer.
"""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, MuseGlimmerConfig
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    MuseGlimmerVisionModel,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import StaticCacheState
from mobius.components._rotary_embedding import apply_rotary_pos_emb


class MuseGlimmerScaleFreeRMSNorm(nn.Module):
    """RMSNorm with no learned scale, evaluated in float32."""

    def __init__(self, eps: float):
        super().__init__()
        self._eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        hidden_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        variance = op.ReduceMean(
            op.Mul(hidden_f32, hidden_f32),
            [-1],
            keepdims=1,
        )
        normalized = op.Mul(
            hidden_f32,
            op.Pow(op.Add(variance, self._eps), -0.5),
        )
        return op.CastLike(normalized, hidden_states)


class MuseGlimmerCenteredRMSNorm(nn.Module):
    """Centered RMSNorm whose checkpoint multiplier is stored as ``weight + 1``."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self._eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        hidden_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        variance = op.ReduceMean(
            op.Mul(hidden_f32, hidden_f32),
            [-1],
            keepdims=1,
        )
        normalized = op.Mul(
            hidden_f32,
            op.Pow(op.Add(variance, self._eps), -0.5),
        )
        normalized = op.Mul(
            normalized,
            op.Add(op.Cast(self.weight, to=ir.DataType.FLOAT), 1.0),
        )
        return op.CastLike(normalized, hidden_states)


class MuseGlimmerTextAttention(nn.Module):
    """GQA with scale-free QK normalization and a sigmoid output gate."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._head_dim = config.head_dim
        self._num_heads = config.num_attention_heads
        self._num_kv_heads = config.num_key_value_heads
        self._scale = config.head_dim**-0.5
        self._qk_scale_factor = getattr(config, "qk_scale_factor", 1.0)
        self.q_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.gate_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.qk_norm = MuseGlimmerScaleFreeRMSNorm(config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple | None,
        past_key_value: tuple | StaticCacheState | None,
    ):
        query = self.q_proj(op, hidden_states)
        key = self.k_proj(op, hidden_states)
        value = self.v_proj(op, hidden_states)

        query = op.Reshape(query, [0, 0, self._num_heads, self._head_dim])
        key = op.Reshape(key, [0, 0, self._num_kv_heads, self._head_dim])
        query = op.Mul(self.qk_norm(op, query), self._qk_scale_factor)
        key = self.qk_norm(op, key)
        query = op.Reshape(query, [0, 0, -1])
        key = op.Reshape(key, [0, 0, -1])

        if position_embeddings is not None:
            query = apply_rotary_pos_emb(
                op,
                query,
                position_embeddings,
                self._num_heads,
            )
            key = apply_rotary_pos_emb(
                op,
                key,
                position_embeddings,
                self._num_kv_heads,
            )

        if isinstance(past_key_value, StaticCacheState):
            updated_key = op.TensorScatter(
                past_key_value.key_cache,
                key,
                past_key_value.write_indices,
                axis=1,
            )
            updated_value = op.TensorScatter(
                past_key_value.value_cache,
                value,
                past_key_value.write_indices,
                axis=1,
            )
            output, _, _ = op.Attention(
                query,
                updated_key,
                updated_value,
                attention_bias,
                None,
                None,
                past_key_value.nonpad_kv_seqlen,
                q_num_heads=self._num_heads,
                kv_num_heads=self._num_kv_heads,
                scale=self._scale,
                is_causal=0,
                _outputs=3,
            )
            present = (updated_key, updated_value)
        else:
            output, present_key, present_value = op.Attention(
                query,
                key,
                value,
                attention_bias,
                past_key_value[0] if past_key_value is not None else None,
                past_key_value[1] if past_key_value is not None else None,
                q_num_heads=self._num_heads,
                kv_num_heads=self._num_kv_heads,
                scale=self._scale,
                is_causal=0,
                _outputs=3,
            )
            present = (present_key, present_value)

        gate = op.Sigmoid(self.gate_proj(op, hidden_states))
        return self.o_proj(op, op.Mul(output, gate)), present


class MuseGlimmerTextDecoderLayer(nn.Module):
    """Four-norm Muse decoder block."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        post_norm_eps = getattr(config, "post_norm_eps", 1e-8)
        self.self_attn = MuseGlimmerTextAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = MuseGlimmerCenteredRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_attention_layernorm = MuseGlimmerCenteredRMSNorm(
            config.hidden_size,
            post_norm_eps,
        )
        self.pre_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
            config.hidden_size,
            post_norm_eps,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple | None,
        past_key_value: tuple | StaticCacheState | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_key_value = self.self_attn(
            op,
            hidden_states,
            attention_bias,
            position_embeddings,
            past_key_value,
        )
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.post_feedforward_layernorm(op, hidden_states)
        return op.Add(residual, hidden_states), present_key_value


class MuseGlimmerTextModel(nn.Module):
    """Muse language backbone with mixed sliding/full attention and NoPE layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self._layer_types = (
            config.layer_types or ["sliding_attention"] * config.num_hidden_layers
        )
        self._layer_rope_theta = getattr(
            config,
            "layer_rope_theta",
            [config.rope_theta] * config.num_hidden_layers,
        )
        self._sliding_window = config.sliding_window
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.embed_norm = MuseGlimmerScaleFreeRMSNorm(config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [MuseGlimmerTextDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is None:
            assert input_ids is not None
            hidden_states = self.embed_norm(op, self.embed_tokens(op, input_ids))
        else:
            hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(op, position_ids)
        full_attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        sliding_attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
            attention_mask=attention_mask,
            sliding_window=self._sliding_window,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_kvs)):
            is_sliding = self._layer_types[layer_idx] == "sliding_attention"
            layer_bias = sliding_attention_bias if is_sliding else full_attention_bias
            layer_position_embeddings = (
                position_embeddings if self._layer_rope_theta[layer_idx] else None
            )
            hidden_states, present_key_value = layer(
                op,
                hidden_states,
                layer_bias,
                layer_position_embeddings,
                past_key_value,
            )
            present_key_values.append(present_key_value)

        return self.norm(op, hidden_states), present_key_values


class MuseGlimmerDecoderModel(nn.Module):
    """Decoder graph consuming pre-computed multimodal embeddings."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.model = MuseGlimmerTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self._output_multiplier = getattr(config, "output_multiplier", 1.0)
        self._softcap = getattr(config, "final_logit_softcapping", 0.0)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = op.Mul(self.lm_head(op, hidden_states), self._output_multiplier)
        if self._softcap:
            logits = op.Mul(op.Tanh(op.Div(logits, self._softcap)), self._softcap)
        return logits, present_key_values

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return state_dict


class MuseGlimmerTextCausalLMModel(MuseGlimmerDecoderModel):
    """Standalone text sibling for ``model_type=muse_glimmer_text``."""

    default_task: str = "text-generation"
    category: str = "Text Generation"
    config_class: type = MuseGlimmerConfig

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Extract text weights from the composite Muse Glimmer checkpoint."""
        routed: dict[str, torch.Tensor] = {}
        language_prefix = "model.language_model."
        for key, value in state_dict.items():
            if key.startswith(language_prefix):
                routed[f"model.{key[len(language_prefix) :]}"] = value
            elif key.startswith("lm_head."):
                routed[key] = value
            elif key.startswith("model.") and not key.startswith(
                (
                    "model.vision_tower.",
                    "model.vision_adapter.",
                    "model.vision_projection.",
                )
            ):
                # Native MuseGlimmerForCausalLM checkpoints already use the
                # standalone graph's ``model.*`` namespace.
                routed[key] = value
        return routed

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = op.Mul(self.lm_head(op, hidden_states), self._output_multiplier)
        if self._softcap:
            logits = op.Mul(op.Tanh(op.Div(logits, self._softcap)), self._softcap)
        return logits, present_key_values


class MuseGlimmerVisionAdapter(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size, bias=False)
        self.fc2 = Linear(hidden_size, hidden_size, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        hidden_states = op.Gelu(self.fc1(op, hidden_states))
        return op.Gelu(self.fc2(op, hidden_states))


class MuseGlimmerVisionEncoderModel(nn.Module):
    """Vision tower, pixel-shuffle adapter, projection, and scale-free norm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vision = config.vision
        assert vision is not None
        assert vision.hidden_size is not None
        assert vision.intermediate_size is not None
        assert vision.num_hidden_layers is not None
        assert vision.num_attention_heads is not None
        merge_size = vision.spatial_merge_size
        position_grid_size = (
            getattr(vision, "position_embedding_height", None)
            or getattr(vision, "pos_emb_height", None)
            or 32
        )
        full_attention_layers = vision.fullatt_block_indexes or []
        self.vision_tower = MuseGlimmerVisionModel(
            depth=vision.num_hidden_layers,
            hidden_size=vision.hidden_size,
            intermediate_size=vision.intermediate_size,
            num_heads=vision.num_attention_heads,
            patch_size=vision.patch_size or 14,
            temporal_patch_size=vision.temporal_patch_size,
            in_channels=vision.in_channels,
            merge_size=merge_size,
            position_grid_size=position_grid_size,
            fullatt_block_indexes=full_attention_layers,
            norm_eps=vision.norm_eps,
            rope_theta=vision.rope_theta or 10_000.0,
        )
        shuffled_size = vision.hidden_size * merge_size * merge_size
        adapter_size = vision.projector_intermediate_size or 4096
        self.vision_adapter = MuseGlimmerVisionAdapter(
            shuffled_size,
            adapter_size,
        )
        self.vision_projection = Linear(
            adapter_size,
            config.hidden_size,
            bias=False,
        )
        self.perception_emb_norm = MuseGlimmerScaleFreeRMSNorm(config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ):
        hidden_states = self.vision_tower(
            op,
            pixel_values,
            image_grid_thw,
        )
        hidden_states = self.vision_adapter(op, hidden_states)
        hidden_states = self.vision_projection(op, hidden_states)
        return self.perception_emb_norm(op, hidden_states)


class MuseGlimmerEmbeddingModel(nn.Module):
    """Normalized token lookup with image/video feature replacement."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.embed_norm = MuseGlimmerScaleFreeRMSNorm(config.rms_norm_eps)
        self._hidden_size = config.hidden_size
        self._image_token_id = config.image_token_id or 200092
        self._video_token_id = getattr(config, "video_token_id", 200091)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ):
        text_embeddings = self.embed_norm(
            op,
            self.embed_tokens(op, input_ids),
        )
        feature_mask = op.Or(
            op.Equal(input_ids, self._image_token_id),
            op.Equal(input_ids, self._video_token_id),
        )
        # HF masked_scatter consumes packed features across the flattened batch.
        flat_mask = op.Reshape(feature_mask, [-1])
        feature_indices = op.Sub(
            op.CumSum(op.Cast(flat_mask, to=ir.DataType.INT64), 0),
            1,
        )
        feature_indices = op.Clip(feature_indices, 0)
        # Where evaluates both branches eagerly. Keep Gather valid when decode
        # steps provide no new image features.
        dummy_feature = op.Unsqueeze(
            op.CastLike(
                op.Constant(value_floats=[0.0] * self._hidden_size),
                image_features,
            ),
            [0],
        )
        safe_features = op.Concat(image_features, dummy_feature, axis=0)
        gathered_features = op.Gather(safe_features, feature_indices)
        gathered_features = op.Reshape(gathered_features, op.Shape(text_embeddings))
        return op.Where(
            op.Unsqueeze(feature_mask, [-1]),
            gathered_features,
            text_embeddings,
        )


class MuseGlimmerForConditionalGeneration(nn.Module):
    """Muse Glimmer vision-language model exposed through a 3-model task."""

    default_task: str = "muse-glimmer-vl"
    category: str = "Multimodal"
    config_class: type = MuseGlimmerConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = MuseGlimmerDecoderModel(config)
        self.vision_encoder = MuseGlimmerVisionEncoderModel(config)
        self.embedding = MuseGlimmerEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "MuseGlimmerForConditionalGeneration is built as decoder, "
            "vision_encoder, and embedding graphs."
        )

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        routed: dict[str, torch.Tensor] = {}
        language_prefix = "model.language_model."
        vision_prefix = "model."
        for key, value in state_dict.items():
            if key.startswith(language_prefix):
                language_key = key[len(language_prefix) :]
                routed[f"decoder.model.{language_key}"] = value
                if language_key == "embed_tokens.weight":
                    routed["embedding.embed_tokens.weight"] = value
            elif key.startswith(
                (
                    "model.vision_tower.",
                    "model.vision_adapter.",
                    "model.vision_projection.",
                )
            ):
                routed[f"vision_encoder.{key[len(vision_prefix) :]}"] = value
            elif key.startswith("lm_head."):
                routed[f"decoder.{key}"] = value
        return routed
