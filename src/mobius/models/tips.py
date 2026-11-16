# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Google TIPS spatial vision-language embeddings.

Replicates ``TIPSv2Model`` from the pinned ``google/tipsv1-b14`` checkpoint.
Images are RGB floats in [0, 1], without ImageNet normalization. Text uses
lowercased SentencePiece ids without BOS/EOS; padding_mask is 1 for padding.
The output preserves the unnormalized CLS, register, spatial patch, and pooled
text embeddings. This initial implementation exports the native image grid.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import rename_weight_keys
from mobius.components import Conv2d, Embedding, EncoderAttention, LayerNorm, Linear

TIPS_REVISION = "424f5e0ee56557079d99419c729a655e2cbcc7b1"


def _field(config, name: str, default):
    return (
        config.get(name, default)
        if isinstance(config, Mapping)
        else getattr(config, name, default)
    )


@dataclasses.dataclass
class TipsConfig(ArchitectureConfig):
    """Native-grid TIPS configuration, accepting raw and trusted Hub configs."""

    model_type: str | None = "tipsv2"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    vocab_size: int = 32000
    image_size: int = 448
    patch_size: int = 14
    vision_hidden_size: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    max_position_embeddings: int = 64
    temperature: float = 0.00397537462413311

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> TipsConfig:
        # Composite resolution unwraps text_config. The ArchitectureConfig
        # interface forwards the parent, which retains the vision geometry.
        config = parent_config if parent_config is not None else config
        # Upstream's trusted config flattens and discards both nested configs.
        text = _field(config, "text_config", {})
        vision = _field(config, "vision_config", {})
        raw = bool(text) and bool(vision)
        required = (
            (
                (
                    text,
                    {
                        "hidden_size": 768,
                        "intermediate_size": 3072,
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "vocab_size": 32000,
                        "max_position_embeddings": 64,
                    },
                ),
                (
                    vision,
                    {
                        "hidden_size": 768,
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "image_size": 448,
                        "patch_size": 14,
                    },
                ),
            )
            if raw
            else (
                (
                    config,
                    {
                        "vision_fn": "vit_base",
                        "embed_dim": 768,
                        "img_size": 448,
                        "patch_size": 14,
                        "ffn_layer": "mlp",
                        "init_values": 1.0,
                        "num_register_tokens": 1,
                        "text_hidden_size": 768,
                        "text_mlp_dim": 3072,
                        "text_num_heads": 12,
                        "text_num_layers": 12,
                        "vocab_size": 32000,
                        "max_len": 64,
                    },
                ),
            )
        )
        for section, expected in required:
            for name, value in expected.items():
                actual = _field(section, name, None)
                if actual != value:
                    raise ValueError(
                        f"TIPS b14 requires {name}={value!r}, got {actual!r}; "
                        "load the complete pinned google/tipsv1-b14 configuration."
                    )
        for section, expected in (
            (
                text,
                {
                    "hidden_act": "relu",
                    "layer_norm_eps": 1e-5,
                    "pooling_epsilon": 1e-8,
                    "scale_sqrt_depth": True,
                    "pad_token_id": 0,
                },
            ),
            (
                vision,
                {
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "num_channels": 3,
                    "hidden_act": "gelu",
                    "layer_norm_eps": 1e-6,
                    "mlp_ratio": 4,
                    "qkv_bias": True,
                    "apply_layernorm": True,
                },
            ),
        ):
            for name, value in expected.items():
                actual = _field(section, name, value)
                if actual != value:
                    raise ValueError(
                        f"TIPS b14 requires {name}={value!r}, got {actual!r}; "
                        "use the pinned google/tipsv1-b14 configuration."
                    )
        if _field(config, "vision_fn", "vit_base") != "vit_base":
            raise ValueError("Only TIPS vit_base is supported; use google/tipsv1-b14.")
        if _field(config, "init_values", _field(vision, "layerscale_value", 1.0)) != 1:
            raise ValueError("TIPS b14 requires learned LayerScale; use layerscale_value=1.0.")
        if (
            _field(vision, "use_swiglu_ffn", False)
            or _field(config, "ffn_layer", "mlp") != "mlp"
        ):
            raise ValueError("TIPS SwiGLU checkpoints are not supported; use tipsv1-b14.")
        if _field(vision, "num_register_tokens", 1) != 1:
            raise ValueError(
                "TIPS requires one register token; use the b14 checkpoint config."
            )
        return cls(
            temperature=_field(
                config,
                "temperature_init_value",
                _field(config, "temperature", cls.temperature),
            ),
        )

    def validate(self) -> None:
        if self.dtype != ir.DataType.FLOAT:
            raise ValueError("TIPS currently supports float32 only; select --dtype f32.")
        for width, heads in (
            (self.hidden_size, self.num_attention_heads),
            (self.vision_hidden_size, self.vision_num_heads),
        ):
            if width <= 0 or heads <= 0 or width % heads:
                raise ValueError(
                    f"TIPS hidden width {width} must be positive and divisible by heads {heads}."
                )
        if self.hidden_size < 4 or self.hidden_size % 2:
            raise ValueError("TIPS text hidden_size must be even and at least 4.")
        if self.patch_size <= 0 or self.image_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("TIPS image_size must be a positive multiple of patch_size.")
        if (
            min(
                self.num_hidden_layers,
                self.vision_num_layers,
                self.vocab_size,
                self.intermediate_size,
                self.max_position_embeddings,
            )
            <= 0
        ):
            raise ValueError(
                "TIPS layer counts, vocabulary, MLP width and text length must be positive."
            )


class _Scale(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gamma = nn.Parameter((width,))

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.Mul(x, self.gamma)


class _VisionMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = Linear(width, width * 4)
        self.fc2 = Linear(width * 4, width)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return self.fc2(op, op.Gelu(self.fc1(op, x), approximate="none"))


class _VisionBlock(nn.Module):
    def __init__(self, config: TipsConfig):
        super().__init__()
        width = config.vision_hidden_size
        self.norm1 = LayerNorm(width, eps=1e-6)
        self.attn = EncoderAttention(width, config.vision_num_heads)
        self.ls1 = _Scale(width)
        self.norm2 = LayerNorm(width, eps=1e-6)
        self.mlp = _VisionMLP(width)
        self.ls2 = _Scale(width)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = op.Add(x, self.ls1(op, self.attn(op, self.norm1(op, x))))
        return op.Add(x, self.ls2(op, self.mlp(op, self.norm2(op, x))))


class _PatchEmbed(nn.Module):
    def __init__(self, config: TipsConfig):
        super().__init__()
        self.proj = Conv2d(3, config.vision_hidden_size, config.patch_size, config.patch_size)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = self.proj(op, x)  # (B, D, grid, grid)
        x = op.Reshape(x, [0, 0, -1])
        return op.Transpose(x, perm=[0, 2, 1])  # (B, grid*grid, D)


class _VisionEncoder(nn.Module):
    def __init__(self, config: TipsConfig):
        super().__init__()
        self.width = config.vision_hidden_size
        self.patch_embed = _PatchEmbed(config)
        self.cls_token = nn.Parameter((1, 1, self.width))
        self.register_tokens = nn.Parameter((1, 1, self.width))
        self.pos_embed = nn.Parameter(
            (1, (config.image_size // config.patch_size) ** 2 + 1, self.width)
        )
        self.blocks = nn.ModuleList(
            [_VisionBlock(config) for _ in range(config.vision_num_layers)]
        )
        self.norm = LayerNorm(self.width, eps=1e-6)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        x = self.patch_embed(op, pixel_values)
        token_shape = op.Concat(op.Shape(x, start=0, end=1), [1, self.width], axis=0)
        x = op.Add(
            op.Concat(op.Expand(self.cls_token, token_shape), x, axis=1), self.pos_embed
        )
        # Register tokens are inserted AFTER position addition, not given a position.
        x = op.Concat(
            op.Slice(x, [0], [1], [1]),
            op.Expand(self.register_tokens, token_shape),
            op.Slice(x, [1], [2**63 - 1], [1]),
            axis=1,
        )
        for block in self.blocks:
            x = block(op, x)
        x = self.norm(op, x)
        return (
            op.Slice(x, [0], [1], [1]),
            op.Slice(x, [1], [2], [1]),
            op.Slice(x, [2], [2**63 - 1], [1]),
        )


class _TextMLP(nn.Module):
    def __init__(self, config: TipsConfig):
        super().__init__()
        self.c_fc = Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = Linear(config.intermediate_size, config.hidden_size)

    def forward(self, op: OpBuilder, x: ir.Value, mask: ir.Value) -> ir.Value:
        x = op.Mul(op.Relu(self.c_fc(op, x)), mask)
        return op.Mul(self.c_proj(op, x), mask)


class _TextBlock(nn.Module):
    def __init__(self, config: TipsConfig):
        super().__init__()
        self.attn = EncoderAttention(config.hidden_size, config.num_attention_heads)
        self.ln_1 = LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = _TextMLP(config)
        self.ln_2 = LayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, op: OpBuilder, x: ir.Value, mask: ir.Value, bias: ir.Value):
        x = op.Add(x, self.attn(op, self.ln_1(op, x), bias))
        return op.Add(x, self.mlp(op, self.ln_2(op, x), mask))


class _TextTransformer(nn.Module):
    def __init__(self, config: TipsConfig):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [_TextBlock(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, op: OpBuilder, x: ir.Value, mask: ir.Value, bias: ir.Value):
        for block in self.resblocks:
            x = block(op, x, mask, bias)
        return x


class _TextEncoder(nn.Module):
    def __init__(self, config: TipsConfig):
        super().__init__()
        self.width = config.hidden_size
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.transformer = _TextTransformer(config)
        self.ln_final = LayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, op: OpBuilder, input_ids: ir.Value, padding_mask: ir.Value):
        valid = op.Equal(padding_mask, 0)
        mask = op.Unsqueeze(op.Cast(valid, to=ir.DataType.FLOAT), [2])
        # ORT's rank-3 Attention requires an explicit query-length mask axis.
        bias = op.Expand(
            op.Unsqueeze(valid, [1, 2]),
            op.Concat(
                op.Shape(input_ids, start=0, end=1),
                [1],
                op.Shape(input_ids, start=1, end=2),
                op.Shape(input_ids, start=1, end=2),
                axis=0,
            ),
        )  # (B, 1, S, S), True = attend
        length = op.Squeeze(op.Shape(input_ids, start=1, end=2))
        positions = op.Cast(op.Range(0, length, 1), to=ir.DataType.FLOAT)
        half = self.width // 2
        frequencies = op.Exp(
            op.Mul(
                op.Cast(op.Range(0, half, 1), to=ir.DataType.FLOAT),
                -math.log(10000.0) / (half - 1),
            )
        )
        phase = op.Mul(op.Unsqueeze(positions, [1]), op.Unsqueeze(frequencies, [0]))
        positions = op.Concat(op.Sin(phase), op.Cos(phase), axis=1)  # (S, D)
        x = op.Add(
            op.Mul(self.token_embedding(op, input_ids), math.sqrt(self.width)), positions
        )
        x = self.ln_final(op, self.transformer(op, x, mask, bias))
        # Where, rather than multiply, also removes NaNs for an all-padding row.
        x = op.Where(op.Unsqueeze(valid, [2]), x, 0.0)
        count = op.Add(op.ReduceSum(mask, [1], keepdims=0), 1e-8)
        return op.Div(op.ReduceSum(x, [1], keepdims=0), count)


class TipsModel(nn.Module):
    """Google TIPS dual-tower encoder with dense spatial image features.

    Implements upstream ``TIPSv2Model`` for the ``google/tipsv1-b14`` native
    448px grid. Returns unnormalized CLS/register/patch image features and
    masked-mean text embeddings, preserving both global and spatial retrieval.
    """

    default_task = "tips-feature-extraction"
    category = "Multimodal"
    config_class = TipsConfig

    def __init__(self, config: TipsConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.vision_encoder = _VisionEncoder(config)
        self.text_encoder = _TextEncoder(config)

    def forward(self, op: OpBuilder, pixel_values, input_ids, padding_mask):
        self.config.validate()
        return (
            *self.vision_encoder(op, pixel_values),
            self.text_encoder(op, input_ids, padding_mask),
        )

    def preprocess_weights(self, state_dict: Mapping[str, torch.Tensor]):
        # The ~784 MB f32 checkpoint fits eagerly; split QKV tensors are views.
        weights = rename_weight_keys(
            state_dict,
            [(".attn.proj.", ".attn.out_proj.")],
        )
        result = {}
        for name, value in weights.items():
            if name == "vision_encoder.mask_token":
                continue  # Used only for masked-image training, not encode_image.
            if ".attn.qkv." in name or ".attn.in_proj_" in name:
                prefix, suffix = (
                    name.split(".qkv.") if ".qkv." in name else name.split(".in_proj_")
                )
                for projection, part in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3, 0)):
                    result[f"{prefix}.{projection}.{suffix}"] = part
            else:
                result[name] = value
        return result
