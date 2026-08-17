# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Faithful PyTorch reference for the public NVIDIA Cosmos3-Edge Reasoner.

This module re-implements the *published* Cosmos3-Edge vision tower, merger
projector, interleaved M-RoPE and text decoder so mobius' ONNX graphs can be
compared numerically against a trusted implementation.  Every routine below is
a direct transcription of authoritative upstream sources:

- ``huggingface/transformers`` —
  ``src/transformers/models/cosmos3_edge/modular_cosmos3_edge.py``
  (``Cosmos3EdgeVisionEmbeddings.resize_positional_embeddings``,
  ``Cosmos3EdgeVisionAttention``, ``Cosmos3EdgePatchMerger``,
  ``Cosmos3EdgeTextRotaryEmbedding``, ``Cosmos3EdgeImageProcessor.patchify``,
  ``Cosmos3EdgeVideoProcessor.patchify``).
- ``vllm-project/vllm`` — ``vllm/model_executor/models/cosmos3_edge.py``
  (independent confirmation of the projector shape contract).

The reference is intentionally dependency-light: it consumes plain tensors and
a small config dataclass so it can run against the real checkpoint without
requiring a ``transformers`` release that ships ``cosmos3_edge``.

.. note::
   Only the *understanding* (Reasoner) tower is reproduced.  The Cosmos3-Edge
   Generator/Action towers that share the checkpoint are proprietary
   rectified-flow components and are out of scope here.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as functional

__all__ = [
    "EdgeRefConfig",
    "merge_patches",
    "patchify_images",
    "patchify_videos",
    "ref_interleaved_mrope_cos_sin",
    "ref_projector",
    "ref_text_decoder_logits",
    "ref_vision_features",
    "ref_vision_tower",
    "resize_positional_embeddings",
    "smart_resize",
]


@dataclasses.dataclass
class EdgeRefConfig:
    """The subset of ``config.json`` the reference needs."""

    # Vision tower (``vision_config``).
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_layers: int = 27
    vision_num_heads: int = 16
    patch_size: int = 16
    num_channels: int = 3
    num_patches: int = 256
    layer_norm_eps: float = 1e-6
    spatial_merge_size: int = 2
    # Projector (``projector_config``).
    projector_hidden_size: int = 11520
    use_postshuffle_norm: bool = False
    # Text tower (``text_config``).
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100_000_000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    vocab_size: int = 131072

    @classmethod
    def from_hf_config(cls, config: dict) -> EdgeRefConfig:
        """Build from a raw ``config.json`` mapping."""
        vision = config["vision_config"]
        projector = config["projector_config"]
        text = config["text_config"]
        rope = text["rope_parameters"]
        return cls(
            vision_hidden_size=vision["hidden_size"],
            vision_intermediate_size=vision["intermediate_size"],
            vision_num_layers=vision["num_hidden_layers"],
            vision_num_heads=vision["num_attention_heads"],
            patch_size=vision["patch_size"],
            num_channels=vision["num_channels"],
            num_patches=vision["num_patches"],
            layer_norm_eps=vision["layer_norm_eps"],
            spatial_merge_size=vision["spatial_merge_size"],
            projector_hidden_size=projector["merger_intermediate_size"],
            use_postshuffle_norm=projector.get("use_postshuffle_norm", False),
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            rms_norm_eps=text["rms_norm_eps"],
            rope_theta=rope["rope_theta"],
            mrope_section=tuple(rope["mrope_section"]),
            vocab_size=text["vocab_size"],
        )


# ──────────────────────────────────────────────────────────────────────────
# Preprocessing (Cosmos3EdgeImageProcessor / Cosmos3EdgeVideoProcessor)
# ──────────────────────────────────────────────────────────────────────────


def smart_resize(
    height: int,
    width: int,
    *,
    num_frames: int = 1,
    temporal_factor: int = 1,
    factor: int = 32,
    min_pixels: int = 256 * 256,
    max_pixels: int = 4096 * 4096,
) -> tuple[int, int]:
    """Exact transcription of ``Cosmos3EdgeImageProcessor.smart_resize``.

    The output side lengths are multiples of ``factor``
    (``patch_size * merge_size`` = 32 for Cosmos3-Edge) and the total
    ``num_frames * height * width`` volume is clamped to the processor's
    ``size.shortest_edge`` / ``size.longest_edge``, which are *areas*
    (``256*256`` / ``4096*4096`` for images, ``64*64`` / ``24*1024*1024`` for
    videos), not edge lengths.

    The processor resamples with **bicubic** interpolation, then rescales by
    ``1/255`` and normalises with ``mean = std = 0.5``.
    """
    if num_frames < temporal_factor:
        raise ValueError(
            f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}"
        )
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def patchify_images(
    images: torch.Tensor,
    *,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int = 1,
) -> tuple[torch.Tensor, int, int]:
    """``Cosmos3EdgeImageProcessor.patchify``: block-major, HWC-in-patch.

    ``images`` is ``[B, C, H, W]``.  Returns ``([B, grid_h*grid_w, patch_dim],
    grid_h, grid_w)`` where ``patch_dim = patch*patch*C*temporal_patch_size``
    and the values inside a patch are ordered ``(patch_h, patch_w, channel)``
    (channel-**last**), not the usual ``(channel, patch_h, patch_w)``.
    """
    batch_size, channel, height, width = images.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = images.reshape(
        batch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    # (B, gh/m, gw/m, m, m, ph, pw, C) — 2x2 blocks are contiguous.
    patches = patches.permute(0, 2, 5, 3, 6, 4, 7, 1)
    flatten_patches = (
        patches.unsqueeze(-1)
        .expand(-1, -1, -1, -1, -1, -1, -1, -1, temporal_patch_size)
        .reshape(
            batch_size,
            grid_h * grid_w,
            patch_size * patch_size * channel * temporal_patch_size,
        )
    )
    return flatten_patches, grid_h, grid_w


def patchify_videos(
    videos: torch.Tensor,
    *,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int = 1,
) -> tuple[torch.Tensor, int, int, int]:
    """``Cosmos3EdgeVideoProcessor.patchify``: ``[B, T, C, H, W]`` → packed.

    Cosmos3-Edge only supports ``temporal_patch_size=1``, so every frame
    contributes its own ``grid_h*grid_w`` block-major patch run.
    """
    batch_size, num_frames, channel, height, width = videos.shape
    grid_t = num_frames // temporal_patch_size
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = videos.view(
        batch_size,
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 6, 9, 3, 2)
    flatten_patches = patches.reshape(
        batch_size,
        grid_t * grid_h * grid_w,
        patch_size * patch_size * channel * temporal_patch_size,
    )
    return flatten_patches, grid_t, grid_h, grid_w


# ──────────────────────────────────────────────────────────────────────────
# Vision tower
# ──────────────────────────────────────────────────────────────────────────


def resize_positional_embeddings(
    positional_embeddings: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """``Cosmos3EdgeVisionEmbeddings.resize_positional_embeddings``.

    ``positional_embeddings`` is the learned ``(g, g, D)`` reference grid.  It
    is bilinearly resampled (antialiased, ``align_corners=False``) to each
    item's ``(H, W)`` grid and then reordered into the processor's block-major
    2x2 layout before being repeated ``T`` times.
    """
    positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
    source_dtype = positional_embeddings.dtype
    if positional_embeddings.device.type == "cpu":
        positional_embeddings = positional_embeddings.float()

    position_chunks = []
    for temporal, height, width in grid_thw.tolist():
        resized = functional.interpolate(
            positional_embeddings,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        resized = resized.squeeze(0).permute(1, 2, 0).to(source_dtype)
        resized = resized.reshape(
            height // spatial_merge_size,
            spatial_merge_size,
            width // spatial_merge_size,
            spatial_merge_size,
            -1,
        )
        resized = resized.transpose(1, 2).reshape(height * width, -1)
        position_chunks.append(resized.repeat(temporal, 1))

    return torch.cat(position_chunks, dim=0)


def _vision_attention(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    prefix: str,
    num_heads: int,
) -> torch.Tensor:
    """Packed SigLIP2 attention; ``hidden`` is ``[frames, seq, D]``."""
    frames, seq, dim = hidden.shape
    head_dim = dim // num_heads
    scale = head_dim**-0.5

    def linear(name: str, x: torch.Tensor) -> torch.Tensor:
        return functional.linear(
            x, weights[f"{prefix}.{name}.weight"], weights[f"{prefix}.{name}.bias"]
        )

    query = linear("q_proj", hidden).view(frames, seq, num_heads, head_dim).transpose(1, 2)
    key = linear("k_proj", hidden).view(frames, seq, num_heads, head_dim).transpose(1, 2)
    value = linear("v_proj", hidden).view(frames, seq, num_heads, head_dim).transpose(1, 2)
    attn = functional.scaled_dot_product_attention(query, key, value, scale=scale)
    attn = attn.transpose(1, 2).reshape(frames, seq, dim)
    return linear("out_proj", attn)


def _layer_norm(
    hidden: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str, eps: float
) -> torch.Tensor:
    return functional.layer_norm(
        hidden,
        (hidden.shape[-1],),
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.bias"],
        eps,
    )


def ref_vision_tower(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: EdgeRefConfig,
) -> torch.Tensor:
    """SigLIP2 tower: packed ``[N, patch_dim]`` → ``[N, vision_hidden]``.

    ``grid_thw`` must describe a single item (``[1, 3]``); attention runs
    independently per frame, exactly as ``Cosmos3EdgeEncoder`` does with
    per-frame ``cu_seqlens``.
    """
    assert grid_thw.shape[0] == 1, "reference handles one packed item at a time"
    temporal, height, width = (int(v) for v in grid_thw[0].tolist())

    hidden = functional.linear(
        pixel_values,
        weights["visual.embeddings.patch_embedding.weight"],
        weights["visual.embeddings.patch_embedding.bias"],
    )
    grid = math.isqrt(config.num_patches)
    position_table = weights["visual.embeddings.position_embedding.weight"].reshape(
        grid, grid, -1
    )
    hidden = hidden + resize_positional_embeddings(
        position_table, grid_thw, config.spatial_merge_size
    )

    # Per-frame attention == batched attention over equal-length sequences.
    hidden = hidden.reshape(temporal, height * width, -1)
    for index in range(config.vision_num_layers):
        prefix = f"visual.encoder.layers.{index}"
        residual = hidden
        normed = _layer_norm(hidden, weights, f"{prefix}.layer_norm1", config.layer_norm_eps)
        hidden = residual + _vision_attention(
            normed, weights, f"{prefix}.self_attn", config.vision_num_heads
        )
        residual = hidden
        normed = _layer_norm(hidden, weights, f"{prefix}.layer_norm2", config.layer_norm_eps)
        normed = functional.linear(
            normed,
            weights[f"{prefix}.mlp.fc1.weight"],
            weights[f"{prefix}.mlp.fc1.bias"],
        )
        normed = functional.gelu(normed, approximate="tanh")
        normed = functional.linear(
            normed,
            weights[f"{prefix}.mlp.fc2.weight"],
            weights[f"{prefix}.mlp.fc2.bias"],
        )
        hidden = residual + normed

    hidden = _layer_norm(hidden, weights, "visual.post_layernorm", config.layer_norm_eps)
    return hidden.reshape(temporal * height * width, -1)


def merge_patches(features: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Group ``spatial_merge_size**2`` consecutive (block-major) patches."""
    return features.reshape(-1, spatial_merge_size**2, features.shape[-1])


def ref_projector(
    features: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: EdgeRefConfig,
) -> torch.Tensor:
    """``Cosmos3EdgePatchMerger``: LayerNorm → shuffle → fc1 → GELU → fc2."""
    merged_dim = config.vision_hidden_size * config.spatial_merge_size**2
    hidden = merge_patches(features, config.spatial_merge_size)
    if config.use_postshuffle_norm:
        hidden = _layer_norm(hidden.reshape(-1, merged_dim), weights, "projector.norm", 1e-6)
    else:
        hidden = _layer_norm(hidden, weights, "projector.norm", 1e-6).reshape(-1, merged_dim)
    hidden = functional.linear(
        hidden, weights["projector.linear_fc1.weight"], weights["projector.linear_fc1.bias"]
    )
    hidden = functional.gelu(hidden)
    return functional.linear(
        hidden, weights["projector.linear_fc2.weight"], weights["projector.linear_fc2.bias"]
    )


def ref_vision_features(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: EdgeRefConfig,
) -> torch.Tensor:
    """Full ``pixel_values`` → projected ``[num_vision_tokens, text_hidden]``."""
    return ref_projector(
        ref_vision_tower(pixel_values, grid_thw, weights, config), weights, config
    )


# ──────────────────────────────────────────────────────────────────────────
# Text tower
# ──────────────────────────────────────────────────────────────────────────


def ref_interleaved_mrope_cos_sin(
    position_ids: torch.Tensor, config: EdgeRefConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Cosmos3EdgeTextRotaryEmbedding``: axis-interleaved 3D M-RoPE.

    ``position_ids`` is ``[3, batch, seq]``.  Frequency channel ``i`` is driven
    by the height axis when ``i % 3 == 1 and i < 3*mrope_section[1]``, by the
    width axis when ``i % 3 == 2 and i < 3*mrope_section[2]``, and by the
    temporal axis otherwise.
    """
    dim = config.head_dim
    inv_freq = 1.0 / (
        config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    )
    indices = torch.arange(inv_freq.shape[0])
    height_mask = (indices % 3 == 1) & (indices < config.mrope_section[1] * 3)
    width_mask = (indices % 3 == 2) & (indices < config.mrope_section[2] * 3)
    temporal_mask = ~(height_mask | width_mask)
    inv_freq = torch.stack(
        (inv_freq * temporal_mask, inv_freq * height_mask, inv_freq * width_mask)
    )
    freqs = position_ids.permute(1, 2, 0).double() @ inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().float(), emb.sin().float()


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = hidden.float().pow(2).mean(-1, keepdim=True)
    return weight * (hidden.float() * torch.rsqrt(variance + eps)).to(hidden.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def ref_text_decoder_logits(
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: EdgeRefConfig,
) -> torch.Tensor:
    """Causal GQA decoder with squared-ReLU FFN and interleaved M-RoPE.

    Mirrors ``Cosmos3EdgeTextModel`` (a ``LlamaModel`` with
    ``Cosmos3EdgeTextMLP`` = ``fc1 → relu² → fc2`` and no QK-norm).
    ``inputs_embeds`` is ``[batch, seq, hidden]`` and ``position_ids`` is
    ``[3, batch, seq]``.
    """
    batch, seq, _ = inputs_embeds.shape
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    cos, sin = ref_interleaved_mrope_cos_sin(position_ids, config)
    cos = cos.unsqueeze(1).to(inputs_embeds.dtype)
    sin = sin.unsqueeze(1).to(inputs_embeds.dtype)

    hidden = inputs_embeds
    for index in range(config.num_hidden_layers):
        prefix = f"layers.{index}"
        residual = hidden
        normed = _rms_norm(
            hidden, weights[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps
        )
        query = functional.linear(normed, weights[f"{prefix}.self_attn.to_q.weight"])
        key = functional.linear(normed, weights[f"{prefix}.self_attn.to_k.weight"])
        value = functional.linear(normed, weights[f"{prefix}.self_attn.to_v.weight"])
        query = query.view(batch, seq, heads, head_dim).transpose(1, 2)
        key = key.view(batch, seq, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch, seq, kv_heads, head_dim).transpose(1, 2)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        attn = functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=True
        )
        attn = attn.transpose(1, 2).reshape(batch, seq, heads * head_dim)
        hidden = residual + functional.linear(
            attn, weights[f"{prefix}.self_attn.to_out.weight"]
        )

        residual = hidden
        normed = _rms_norm(
            hidden, weights[f"{prefix}.post_attention_layernorm.weight"], config.rms_norm_eps
        )
        normed = functional.linear(normed, weights[f"{prefix}.mlp.up_proj.weight"])
        normed = torch.square(functional.relu(normed))
        hidden = residual + functional.linear(
            normed, weights[f"{prefix}.mlp.down_proj.weight"]
        )

    hidden = _rms_norm(hidden, weights["norm.weight"], config.rms_norm_eps)
    return functional.linear(hidden, weights["lm_head.weight"])
