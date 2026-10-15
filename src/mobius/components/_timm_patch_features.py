from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import VisionConfig
from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d


class _PatchEmbed(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int):
        super().__init__()
        self.proj = Conv2d(3, hidden_size, patch_size, patch_size)

    def forward(self, op: OpBuilder, pixels: ir.Value):
        patches = self.proj(op, pixels)
        patches = op.Reshape(
            patches,
            op.Concat(op.Shape(patches, start=0, end=2), [-1], axis=0),
        )
        return op.Transpose(patches, perm=[0, 2, 1])


class _TimmAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.qkv = Linear(hidden_size, 3 * hidden_size)
        self.proj = Linear(hidden_size, hidden_size)
        self.num_heads = num_heads

    def forward(self, op: OpBuilder, hidden: ir.Value):
        query, key, value = op.Split(self.qkv(op, hidden), axis=-1, num_outputs=3, _outputs=3)
        attended = op.Attention(
            query,
            key,
            value,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            _outputs=1,
        )
        return self.proj(op, attended)


class _TimmMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, approximation: str):
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size)
        self.fc2 = Linear(intermediate_size, hidden_size)
        self.approximation = approximation

    def forward(self, op: OpBuilder, hidden: ir.Value):
        return self.fc2(op, op.Gelu(self.fc1(op, hidden), approximate=self.approximation))


class _TimmBlock(nn.Module):
    def __init__(self, config: VisionConfig, approximation: str):
        super().__init__()
        assert config.hidden_size is not None
        assert config.intermediate_size is not None
        assert config.num_attention_heads is not None
        self.norm1 = LayerNorm(config.hidden_size, config.norm_eps)
        self.attn = _TimmAttention(config.hidden_size, config.num_attention_heads)
        self.norm2 = LayerNorm(config.hidden_size, config.norm_eps)
        self.mlp = _TimmMLP(config.hidden_size, config.intermediate_size, approximation)

    def forward(self, op: OpBuilder, hidden: ir.Value):
        hidden = op.Add(hidden, self.attn(op, self.norm1(op, hidden)))
        return op.Add(hidden, self.mlp(op, self.norm2(op, hidden)))


class _LayerScale(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale_factor = nn.Parameter([hidden_size])

    def forward(self, op: OpBuilder, hidden: ir.Value):
        return op.Mul(hidden, self.scale_factor)


class _DinoBlock(_TimmBlock):
    def __init__(self, config: VisionConfig):
        super().__init__(config, "none")
        assert config.hidden_size is not None
        self.ls1 = _LayerScale(config.hidden_size)
        self.ls2 = _LayerScale(config.hidden_size)

    def forward(self, op: OpBuilder, hidden: ir.Value):
        hidden = op.Add(hidden, self.ls1(op, self.attn(op, self.norm1(op, hidden))))
        return op.Add(hidden, self.ls2(op, self.mlp(op, self.norm2(op, hidden))))


class SiglipPatchFeatures(nn.Module):
    """Unnormalized timm ViT patch features at an explicit intermediate block."""

    def __init__(self, config: VisionConfig, *, num_layers: int):
        super().__init__()
        assert config.hidden_size is not None
        assert config.patch_size is not None
        assert config.image_size is not None
        self.patch_embed = _PatchEmbed(config.hidden_size, config.patch_size)
        self.pos_embed = nn.Parameter(
            [1, (config.image_size // config.patch_size) ** 2, config.hidden_size]
        )
        self.blocks = nn.ModuleList([_TimmBlock(config, "none") for _ in range(num_layers)])

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        hidden = op.Add(self.patch_embed(op, pixel_values), self.pos_embed)
        for block in self.blocks:
            hidden = block(op, hidden)
        return hidden


class DinoV2PatchFeatures(nn.Module):
    """timm DINOv2 register-token features without the unused final block/norm."""

    def __init__(self, config: VisionConfig, *, num_layers: int, register_tokens: int = 4):
        super().__init__()
        assert config.hidden_size is not None
        assert config.patch_size is not None
        assert config.image_size is not None
        self.patch_embed = _PatchEmbed(config.hidden_size, config.patch_size)
        patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter([1, 1, config.hidden_size])
        self.reg_token = nn.Parameter([1, register_tokens, config.hidden_size])
        self.pos_embed = nn.Parameter([1, patches, config.hidden_size])
        self.blocks = nn.ModuleList([_DinoBlock(config) for _ in range(num_layers)])
        self.register_tokens = register_tokens
        self.hidden_size = config.hidden_size

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        patches = op.Add(self.patch_embed(op, pixel_values), self.pos_embed)
        batch = op.Shape(patches, start=0, end=1)
        cls = op.Expand(
            self.cls_token, op.Concat(batch, [1, self.hidden_size], axis=0)
        )
        registers = op.Expand(
            self.reg_token,
            op.Concat(batch, [self.register_tokens, self.hidden_size], axis=0),
        )
        hidden = op.Concat(
            cls,
            registers,
            patches,
            axis=1,
        )
        for block in self.blocks:
            hidden = block(op, hidden)
        return op.Slice(hidden, [1 + self.register_tokens], [9223372036854775807], [1])
