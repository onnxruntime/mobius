# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""EmbeddingGemma 2 bidirectional encoder and sentence embedding projection.

Replicates HuggingFace ``EmbeddingGemma2TextModel``: token or fused media
embeddings become projected token features. SentenceTransformers-style masked
mean pooling and L2 normalization produce retrieval-ready sentence embeddings.
"""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace
from typing import ClassVar

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Gemma4Config
from mobius.components import GatedMLP, Linear, RMSNorm, get_activation, initialize_rope
from mobius.models.gemma4 import (
    Gemma4Model,
    Gemma4ScaledWordEmbedding,
    Gemma4TextAttention,
    _Gemma4AudioEncoderModel,
    _Gemma4VisionEncoderModel,
)

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"


@dataclasses.dataclass
class EmbeddingGemma2Config(Gemma4Config):
    """Embedding encoder configuration; sliding_window is an inclusive radius."""

    embedding_dim: int = 768

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> EmbeddingGemma2Config:
        # Keep RoPE and sparse per-layer dictionaries as mappings.
        config = SimpleNamespace(**config) if isinstance(config, dict) else config
        parent = parent_config or config
        parent = SimpleNamespace(**parent) if isinstance(parent, dict) else parent
        text = getattr(config, "text_config", config)
        text = SimpleNamespace(**text) if isinstance(text, dict) else text
        # The generic extractor does not support this encoder's heterogeneous
        # per-layer schema. Resolve overrides explicitly, including sparse JSON.
        source = copy.copy(text)
        source.per_layer_config = None
        if getattr(source, "rope_scaling", None) is None:
            source.rope_scaling = getattr(source, "rope_parameters", None)
        base = Gemma4Config.from_transformers(source, parent)
        overrides = getattr(text, "per_layer_config", None) or {}
        if isinstance(overrides, SimpleNamespace):
            overrides = vars(overrides)
        dims: dict[str, set[tuple[int, int]]] = {}
        for i, layer_type in enumerate(base.layer_types or []):
            entry = (
                overrides.get(i, overrides.get(str(i), overrides.get(f"{i:02d}", {})))
                if isinstance(overrides, dict)
                else overrides[i]
            )
            hd = (
                entry.get("head_dim", base.head_dim)
                if isinstance(entry, dict)
                else getattr(entry, "head_dim", base.head_dim)
            )
            kv = (
                entry.get("num_key_value_heads", base.num_key_value_heads)
                if isinstance(entry, dict)
                else getattr(entry, "num_key_value_heads", base.num_key_value_heads)
            )
            dims.setdefault(layer_type, set()).add((hd, kv))
        if any(len(values) != 1 for values in dims.values()):
            raise ValueError(
                "EmbeddingGemma2 requires uniform head dimensions and KV heads within "
                "each attention type; use the pinned google/embeddinggemma-2 config."
            )
        values = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        # Checkpoint dtype describes storage, not the qualified export precision.
        values["dtype"] = ir.DataType.FLOAT
        if getattr(parent, "vision_config", None) is None:
            values["vision"] = None
        if getattr(parent, "audio_config", None) is None:
            values["audio"] = None
        if "full_attention" in dims:
            hd, kv = next(iter(dims["full_attention"]))
            values.update(global_head_dim=hd, num_global_key_value_heads=kv)
        if "sliding_attention" in dims:
            hd, kv = next(iter(dims["sliding_attention"]))
            values.update(head_dim=hd, num_key_value_heads=kv)
        values["global_partial_rotary_factor"] = 1.0
        values["partial_rotary_factor"] = 1.0
        return cls(**values, embedding_dim=getattr(text, "embedding_dim", 768))


class _ProjectionInputs(nn.Module):
    def __init__(self, config: EmbeddingGemma2Config):
        super().__init__()
        self.layers = config.num_hidden_layers
        self.dim = config.hidden_size_per_layer_input
        self.scale = config.hidden_size**-0.5
        self.per_layer_model_projection = Linear(
            config.hidden_size, self.layers * self.dim, bias=False
        )
        self.per_layer_projection_norm = RMSNorm(self.dim, eps=config.rms_norm_eps)

    def forward(self, op: OpBuilder, inputs_embeds: ir.Value) -> ir.Value:
        h = op.Mul(self.per_layer_model_projection(op, inputs_embeds), self.scale)
        h = op.Reshape(h, [0, 0, self.layers, self.dim])  # [B, S, L, P]
        return self.per_layer_projection_norm(op, h)


class _ProjectionBlock(nn.Module):
    def __init__(self, config: EmbeddingGemma2Config):
        super().__init__()
        self.per_layer_input_gate = Linear(
            config.hidden_size, config.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = Linear(
            config.hidden_size_per_layer_input, config.hidden_size, bias=False
        )
        self.post_per_layer_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.act = get_activation(config.hidden_act)

    def forward(self, op: OpBuilder, h: ir.Value, per_layer: ir.Value) -> ir.Value:
        gated = op.Mul(self.act(op, self.per_layer_input_gate(op, h)), per_layer)
        return op.Add(
            h, self.post_per_layer_input_norm(op, self.per_layer_projection(op, gated))
        )


class _EncoderLayer(nn.Module):
    def __init__(self, config: EmbeddingGemma2Config, index: int):
        super().__init__()
        assert config.layer_types is not None
        hd = (
            config.global_head_dim
            if config.layer_types[index] == "full_attention"
            else config.head_dim
        )
        assert hd is not None
        self.self_attn = Gemma4TextAttention(config, index, config.layer_types, 0, hd, 0)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GatedMLP(
            config.hidden_size, config.intermediate_size, activation=config.hidden_act
        )
        self.ple_block = _ProjectionBlock(config)
        self.layer_scalar = nn.Parameter([1])

    def forward(self, op, h, mask, rope, per_layer):
        a, _ = self.self_attn(op, self.input_layernorm(op, h), mask, rope, is_causal=0)
        h = op.Add(h, self.post_attention_layernorm(op, a))
        h = op.Add(
            h,
            self.post_feedforward_layernorm(
                op, self.mlp(op, self.pre_feedforward_layernorm(op, h))
            ),
        )
        return op.Mul(self.ple_block(op, h, per_layer), self.layer_scalar)


class EmbeddingGemma2TextModel(nn.Module):
    """Google EmbeddingGemma 2 encoder with normalized masked-mean embeddings.

    Inputs are token IDs, a padding mask and absolute positions. Outputs are
    projected token features and unit-length sentence embeddings, not logits.
    """

    default_task = "embedding-gemma2"
    category = "encoder-only"

    def __init__(self, config: EmbeddingGemma2Config):
        super().__init__()
        self.config = config
        self._validate()
        self.embed_tokens = Gemma4ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=config.hidden_size**0.5,
        )
        self.ple = _ProjectionInputs(config)
        self.layers = nn.ModuleList(
            [_EncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_projection = Linear(
            config.hidden_size, config.embedding_dim, bias=False
        )
        self.local_rope = initialize_rope(
            dataclasses.replace(config, rope_type="default", partial_rotary_factor=1.0)
        )
        self.global_rope = initialize_rope(
            dataclasses.replace(
                config,
                head_dim=config.global_head_dim or config.head_dim,
                rope_theta=config.global_rope_theta,
                rope_type="default",
                partial_rotary_factor=1.0,
            )
        )

    def _validate(self) -> None:
        c = self.config
        if c.dtype != ir.DataType.FLOAT:
            raise ValueError(
                f"EmbeddingGemma2 currently supports float32 only, got {c.dtype}. "
                "Export with dtype='f32'; float16 overflows this architecture."
            )
        if not c.layer_types or len(c.layer_types) != c.num_hidden_layers:
            raise ValueError("EmbeddingGemma2 needs one layer_types entry per encoder layer.")
        if set(c.layer_types) - {"sliding_attention", "full_attention"}:
            raise ValueError(
                "EmbeddingGemma2 supports sliding_attention and full_attention only."
            )
        if c.hidden_size_per_layer_input <= 0 or not c.sliding_window:
            raise ValueError(
                "EmbeddingGemma2 requires positive PLE dimension and window radius."
            )

    def forward(
        self,
        op,
        input_ids,
        attention_mask,
        position_ids,
        image_features=None,
        video_features=None,
        audio_features=None,
    ):
        self._validate()
        h = self.embed_tokens(op, input_ids)
        for token_id, features in (
            (self.config.image_token_id, image_features),
            (self.config.video_token_id, video_features),
            (self.config.audio.audio_token_id if self.config.audio else None, audio_features),
        ):
            if features is not None and token_id is not None:
                # Feature rows follow placeholder order over the flattened batch.
                # Keep image and video streams separate even though they share a tower.
                mask = op.Equal(input_ids, token_id)
                flat_mask = op.Reshape(op.Cast(mask, to=ir.DataType.INT64), [-1])
                index = op.CumSum(flat_mask, 0)
                index = op.Where(op.Cast(flat_mask, to=ir.DataType.BOOL), index, 0)
                padded = op.Concat(
                    op.Constant(
                        value=ir.tensor(torch.zeros(1, self.config.hidden_size).numpy())
                    ),
                    features,
                    axis=0,
                )
                media = op.Reshape(op.Gather(padded, index, axis=0), op.Shape(h))
                h = op.Where(op.Unsqueeze(mask, [-1]), media, h)
        per_layer = self.ple(op, h)
        local_rope = self.local_rope(op, position_ids)
        global_rope = self.global_rope(op, position_ids)
        length = op.Squeeze(op.Shape(input_ids, start=1, end=2), [0])
        indices = op.Range(0, length, 1)
        distance = op.Abs(op.Sub(op.Unsqueeze(indices, [1]), op.Unsqueeze(indices, [0])))
        local = op.LessOrEqual(distance, self.config.sliding_window)
        padding = op.Unsqueeze(op.Cast(attention_mask, to=ir.DataType.BOOL), [1, 2])
        full = op.Expand(
            padding,
            op.Concat(
                op.Shape(input_ids, start=0, end=1),
                [1],
                op.Shape(input_ids, start=1, end=2),
                op.Shape(input_ids, start=1, end=2),
                axis=0,
            ),
        )
        masks = {
            "full_attention": full,
            "sliding_attention": op.And(padding, op.Unsqueeze(local, [0, 1])),
        }
        for i, layer in enumerate(self.layers):
            kind = self.config.layer_types[i]
            # Projection-only PLE has no auxiliary vocabulary table or blend.
            ple_i = op.Squeeze(op.Slice(per_layer, [i], [i + 1], [2]), [2])
            h = layer(
                op,
                h,
                masks[kind],
                local_rope if kind == "sliding_attention" else global_rope,
                ple_i,
            )
        tokens = self.embedding_projection(op, self.norm(op, h))  # [B, S, E]
        weights = op.Unsqueeze(op.CastLike(attention_mask, tokens), [-1])
        summed = op.ReduceSum(op.Mul(tokens, weights), [1], keepdims=0)
        count = op.Max(op.ReduceSum(weights, [1], keepdims=0), 1e-9)
        mean = op.Div(summed, count)  # [B, E]
        norm = op.Sqrt(op.ReduceSum(op.Mul(mean, mean), [-1], keepdims=1))
        return tokens, op.Div(mean, op.Max(norm, 1e-12))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            key.removeprefix("language_model."): value
            for key, value in state_dict.items()
            if key.startswith("language_model.")
        }


class EmbeddingGemma2Model(Gemma4Model):
    """Multimodal EmbeddingGemma 2 with stateless sentence embedding output.

    The package includes a bidirectional embedding encoder and optional Gemma4
    image/video and audio towers. Media features replace placeholders before
    projection-only PLE; there is no decoder, LM head, or KV cache. Only float32
    export is qualified. The shared vision graph processes one image or video
    frame per call; callers concatenate its feature rows in processor order.
    """

    default_task = "embedding-gemma2"
    category = "Multimodal"
    config_class = EmbeddingGemma2Config
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "model": ("language_model",),
        "vision_encoder": ("vision_tower", "embed_vision"),
        "audio_encoder": ("audio_tower", "embed_audio"),
    }
    HF_COMPONENT_MODULE_ALIASES: ClassVar[dict[str, dict[str, str]]] = {
        "model": {"encoder": "language_model"},
        "vision_encoder": {
            "encoder": "vision_tower.encoder",
            "projector": "embed_vision.embedding_projection",
        },
        "audio_encoder": {
            "encoder": "audio_tower",
            "projector": "embed_audio.embedding_projection",
        },
    }

    def __init__(self, config: EmbeddingGemma2Config):
        nn.Module.__init__(self)
        self.config = config
        self.encoder = EmbeddingGemma2TextModel(config)
        self.vision_encoder = (
            _Gemma4VisionEncoderModel(config) if config.vision is not None else None
        )
        self.audio_encoder = (
            _Gemma4AudioEncoderModel(config) if config.audio is not None else None
        )

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "EmbeddingGemma2Model exports separate stateless encoder and media graphs; "
            "build it with task='embedding-gemma2'."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Reuse the Gemma4 media tower mappings without introducing decoder weights.
        mapped = super().preprocess_weights(state_dict)
        renamed = {
            key: value
            for key, value in mapped.items()
            if key.startswith(("vision_encoder.", "audio_encoder."))
        }
        source = {key.removeprefix("model."): value for key, value in state_dict.items()}
        renamed.update(
            {
                "encoder." + key: value
                for key, value in self.encoder.preprocess_weights(source).items()
            }
        )
        return renamed
