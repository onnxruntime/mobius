# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact standalone import helpers for the remaining vision projector routes."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn

from mobius._configs import ArchitectureConfig, Lfm2VlConfig, VisionConfig
from mobius.components import (
    CogVLMClipSidecar,
    Exaone45VisionSidecar,
    FixedResolutionSiglipMLPSidecar,
    HunyuanVLClipSidecar,
    KimiK25VisionSidecar,
    KimiVLVisionSidecar,
    MiMoVLVisionSidecar,
    MiniMaxM3VisionSidecar,
    NemotronV2VLClipSidecar,
    Step3VLClipSidecar,
    Yasa2VisionSidecar,
    map_fixed_siglip_sidecar_weight,
)
from mobius.models.clip import ClipVisionConfigView, SigLIPVisionModel
from mobius.models.lfm2_vl import _Lfm2VlVisionEncoderModel
from mobius.models.minicpmv4_6 import _MiniCPMVisionEncoderModel

TensorShapes = Mapping[str, tuple[int, ...]]

REMAINING_VISION_PROJECTOR_TYPES = frozenset(
    {
        "cogvlm",
        "exaone4_5",
        "hunyuanvl",
        "janus_pro",
        "kimik25",
        "kimivl",
        "lfm2",
        "mimovl",
        "minicpmv4_6",
        "minimax_m3",
        "nemotron_v2_vl",
        "step3vl",
        "yasa2",
    }
)

_BLOCK = re.compile(r"^v\.blk\.(\d+)\.(.+)$")
_YASA_BLOCK = re.compile(r"^v\.stage\.(\d+)\.blk\.(\d+)\.(.+)$")
_YASA_DOWN = re.compile(r"^v\.stage\.(\d+)\.down\.(.+)$")


def _metadata_int(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}.")
    return int(value)


def _metadata_float(metadata: Mapping[str, object], key: str) -> float:
    value = metadata.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{key} must be a positive finite number, got {value!r}.")
    return float(value)


def _optional_metadata_int(
    metadata: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    value = metadata.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}.")
    return int(value)


def _shape(shapes: TensorShapes, name: str, rank: int | None = None) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in shapes[name])
    except KeyError as exc:
        raise ValueError(f"GGUF projector is missing tensor {name!r}.") from exc
    if rank is not None and len(shape) != rank:
        raise ValueError(
            f"GGUF projector tensor {name!r} has shape {shape}, expected rank {rank}."
        )
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"GGUF projector tensor {name!r} has non-positive shape {shape}.")
    return shape


def _attach_contract(
    module: nn.Module,
    input_schema: tuple[tuple[str, ir.DataType, tuple[Any, ...]], ...],
    *,
    squeeze_batch_dim: bool = False,
) -> nn.Module:
    module.input_schema = input_schema  # type: ignore[attr-defined]
    module.squeeze_batch_dim = squeeze_batch_dim  # type: ignore[attr-defined]
    return module


def _standard_vision_config(metadata: Mapping[str, object]) -> VisionConfig:
    return VisionConfig(
        hidden_size=_metadata_int(metadata, "clip.vision.embedding_length"),
        intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
        num_hidden_layers=_metadata_int(metadata, "clip.vision.block_count"),
        num_attention_heads=_metadata_int(metadata, "clip.vision.attention.head_count"),
        image_size=_metadata_int(metadata, "clip.vision.image_size"),
        patch_size=_metadata_int(metadata, "clip.vision.patch_size"),
        norm_eps=_metadata_float(metadata, "clip.vision.attention.layer_norm_epsilon"),
        hidden_act="gelu_pytorch_tanh",
    )


def _create_janus(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    vision = _standard_vision_config(metadata)
    hidden = int(vision.hidden_size or 0)
    first = _shape(shapes, "mm.0.weight", 2)
    second = _shape(shapes, "mm.1.weight", 2)
    if first[1] != hidden or second != (first[0], first[0]):
        raise ValueError(f"janus_pro projector shapes {first}/{second} are inconsistent.")
    tower = SigLIPVisionModel(ClipVisionConfigView(vision))
    module = FixedResolutionSiglipMLPSidecar(tower, hidden, first[0], second[0])
    image_size = int(vision.image_size or 0)
    return _attach_contract(
        module,
        (("pixel_values", ir.DataType.FLOAT, (1, 3, image_size, image_size)),),
        squeeze_batch_dim=True,
    )


def _create_lfm2(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    hidden = _metadata_int(metadata, "clip.vision.embedding_length")
    patch = _metadata_int(metadata, "clip.vision.patch_size")
    output = _metadata_int(metadata, "clip.vision.projection_dim")
    positions = _shape(shapes, "v.position_embd.weight", 2)[0]
    first = _shape(shapes, "mm.1.weight", 2)
    scale = _metadata_int(metadata, "clip.vision.projector.scale_factor")
    config = Lfm2VlConfig(
        model_type="lfm2_vl",
        hidden_size=output,
        intermediate_size=output * 4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=output,
        vocab_size=1,
        max_position_embeddings=1,
        vision=VisionConfig(
            model_type="siglip2_naflex",
            hidden_size=hidden,
            intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
            num_hidden_layers=_metadata_int(metadata, "clip.vision.block_count"),
            num_attention_heads=_metadata_int(metadata, "clip.vision.attention.head_count"),
            patch_size=patch,
            num_position_embeddings=positions,
            norm_eps=_metadata_float(
                metadata,
                "clip.vision.attention.layer_norm_epsilon",
            ),
            hidden_act="gelu_pytorch_tanh",
        ),
        downsample_factor=scale,
        projector_hidden_size=first[0],
        projector_hidden_act="gelu",
        projector_bias=True,
        projector_use_layernorm="mm.input_norm.weight" in shapes,
    )
    module = _Lfm2VlVisionEncoderModel(config)
    images = ir.SymbolicDim("num_images")
    max_patches = ir.SymbolicDim("max_patches")
    return _attach_contract(
        module,
        (
            (
                "pixel_values",
                ir.DataType.FLOAT,
                (images, max_patches, 3 * patch * patch),
            ),
            ("pixel_attention_mask", ir.DataType.INT32, (images, max_patches)),
            ("spatial_shapes", ir.DataType.INT64, (images, 2)),
        ),
    )


def _create_minicpm(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    hidden = _metadata_int(metadata, "clip.vision.embedding_length")
    patch = _metadata_int(metadata, "clip.vision.patch_size")
    output = _metadata_int(metadata, "clip.vision.projection_dim")
    positions = _shape(shapes, "v.position_embd.weight", 2)[0]
    position_side = math.isqrt(positions)
    if position_side * position_side != positions:
        raise ValueError("minicpmv4_6 position table must be square.")
    wa_layers = metadata.get("clip.vision.wa_layer_indexes")
    if (
        not isinstance(wa_layers, list)
        or len(wa_layers) != 1
        or isinstance(wa_layers[0], bool)
        or not isinstance(wa_layers[0], int)
    ):
        raise ValueError("minicpmv4_6 requires one integer wa_layer_indexes entry.")
    scale = _metadata_int(metadata, "clip.vision.projector.scale_factor")
    if scale not in (2, 4):
        raise ValueError(
            f"minicpmv4_6 clip.vision.projector.scale_factor must be 2 or 4, got {scale}."
        )
    config = ArchitectureConfig(
        model_type="minicpmv4_6",
        hidden_size=output,
        intermediate_size=output * 4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=output,
        vocab_size=1,
        max_position_embeddings=1,
        downsample_mode="4x" if scale == 2 else "16x",
        vision=VisionConfig(
            hidden_size=hidden,
            intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
            num_hidden_layers=_metadata_int(metadata, "clip.vision.block_count"),
            num_attention_heads=_metadata_int(metadata, "clip.vision.attention.head_count"),
            image_size=position_side * patch,
            patch_size=patch,
            norm_eps=_metadata_float(
                metadata,
                "clip.vision.attention.layer_norm_epsilon",
            ),
            num_position_embeddings=positions,
            insert_layer_id=int(wa_layers[0]),
            window_kernel_size=(2, 2),
            merge_kernel_size=(2, 2),
            merger_times=1,
        ),
    )
    module = _MiniCPMVisionEncoderModel(config)
    packed_width = ir.SymbolicDim("packed_width")
    visual_units = ir.SymbolicDim("num_visual_units")
    return _attach_contract(
        module,
        (
            ("pixel_values", ir.DataType.FLOAT, (1, 3, patch, packed_width)),
            ("target_sizes", ir.DataType.INT32, (visual_units, 2)),
        ),
    )


def _create_yasa(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    block_indices: dict[int, set[int]] = {}
    hidden_sizes = [_shape(shapes, "v.patch_embd.weight", 4)[0]]
    for name in shapes:
        block = _YASA_BLOCK.fullmatch(name)
        if block is not None:
            block_indices.setdefault(int(block.group(1)), set()).add(int(block.group(2)))
    if not block_indices or sorted(block_indices) != list(range(len(block_indices))):
        raise ValueError("yasa2 stage indices must be contiguous.")
    depths = []
    for stage in range(len(block_indices)):
        indices = block_indices[stage]
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"yasa2 stage {stage} block indices must be contiguous.")
        depths.append(len(indices))
        if stage:
            hidden_sizes.append(_shape(shapes, f"v.stage.{stage}.down.conv.weight", 4)[0])
    first = _shape(shapes, "mm.0.weight", 2)
    second = _shape(shapes, "mm.2.weight", 2)
    module = Yasa2VisionSidecar(
        depths,
        hidden_sizes,
        first[0],
        second[0],
        image_size=_metadata_int(metadata, "clip.vision.image_size"),
        eps=_metadata_float(metadata, "clip.vision.attention.layer_norm_epsilon"),
    )
    image_size = _metadata_int(metadata, "clip.vision.image_size")
    return _attach_contract(
        module,
        (("pixel_values", ir.DataType.FLOAT, (1, 3, image_size, image_size)),),
        squeeze_batch_dim=True,
    )


def _create_hunyuan(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    image = _metadata_int(metadata, "clip.vision.image_size")
    patch = _metadata_int(metadata, "clip.vision.patch_size")
    grid = image // patch
    position_count = _shape(shapes, "v.position_embd.weight", 2)[0]
    position_grid_size = math.isqrt(position_count)
    if position_grid_size * position_grid_size != position_count:
        raise ValueError("hunyuanvl position table must contain a square patch grid.")
    conv = _shape(shapes, "mm.0.weight", 4)
    output = _shape(shapes, "mm.model.fc.weight", 2)[0]
    module = HunyuanVLClipSidecar(
        vision_hidden_size=_metadata_int(metadata, "clip.vision.embedding_length"),
        intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
        num_heads=_metadata_int(metadata, "clip.vision.attention.head_count"),
        num_layers=_metadata_int(metadata, "clip.vision.block_count"),
        patch_size=patch,
        grid_height=grid,
        grid_width=grid,
        position_grid_size=position_grid_size,
        projector_hidden_size=conv[0],
        output_size=output,
        merge_size=_metadata_int(metadata, "clip.vision.spatial_merge_size"),
        eps=_metadata_float(metadata, "clip.vision.attention.layer_norm_epsilon"),
    )
    height = ir.SymbolicDim("height")
    width = ir.SymbolicDim("width")
    return _attach_contract(
        module,
        (("pixel_values", ir.DataType.FLOAT, (1, 3, height, width)),),
        squeeze_batch_dim=True,
    )


def _create_step3(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    image = _metadata_int(metadata, "clip.vision.image_size")
    patch = _metadata_int(metadata, "clip.vision.patch_size")
    grid = image // patch
    positions = _shape(shapes, "v.position_embd.weight", 2)[0]
    position_side = math.isqrt(positions)
    if position_side * position_side != positions:
        raise ValueError("step3vl position table must be square.")
    downsample = _shape(shapes, "mm.0.weight", 4)[0]
    output = _shape(shapes, "mm.model.fc.weight", 2)[0]
    module = Step3VLClipSidecar(
        vision_hidden_size=_metadata_int(metadata, "clip.vision.embedding_length"),
        intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
        num_heads=_metadata_int(metadata, "clip.vision.attention.head_count"),
        num_layers=_metadata_int(metadata, "clip.vision.block_count"),
        patch_size=patch,
        grid_height=grid,
        grid_width=grid,
        position_grid_size=position_side,
        downsample_hidden_size=downsample,
        output_size=output,
        eps=_metadata_float(metadata, "clip.vision.attention.layer_norm_epsilon"),
    )
    height = ir.SymbolicDim("height")
    width = ir.SymbolicDim("width")
    patches = ir.SymbolicDim("num_patches")
    return _attach_contract(
        module,
        (
            ("pixel_values", ir.DataType.FLOAT, (1, 3, height, width)),
            ("pos_h", ir.DataType.INT64, (patches,)),
            ("pos_w", ir.DataType.INT64, (patches,)),
        ),
        squeeze_batch_dim=True,
    )


def _create_cog(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    image = _metadata_int(metadata, "clip.vision.image_size")
    patch = _metadata_int(metadata, "clip.vision.patch_size")
    projector = _shape(shapes, "mm.model.fc.weight", 2)[0]
    projector_intermediate = _shape(shapes, "mm.up.weight", 2)[0]
    output = _shape(shapes, "mm.down.weight", 2)[0]
    module = CogVLMClipSidecar(
        image_size=image,
        patch_size=patch,
        hidden_size=_metadata_int(metadata, "clip.vision.embedding_length"),
        intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
        num_layers=_metadata_int(metadata, "clip.vision.block_count"),
        num_heads=_metadata_int(metadata, "clip.vision.attention.head_count"),
        projector_hidden_size=projector,
        projector_intermediate_size=projector_intermediate,
        output_size=output,
        norm_eps=_metadata_float(metadata, "clip.vision.attention.layer_norm_epsilon"),
    )
    return _attach_contract(
        module,
        (("pixel_values", ir.DataType.FLOAT, (1, 3, image, image)),),
        squeeze_batch_dim=True,
    )


def _create_nemotron(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    image = _metadata_int(metadata, "clip.vision.image_size")
    patch = _metadata_int(metadata, "clip.vision.patch_size")
    registers = _shape(shapes, "v.class_embd", 2)[0]
    first = _shape(shapes, "mm.model.mlp.1.weight", 2)
    output = _shape(shapes, "mm.model.mlp.3.weight", 2)[0]
    module = NemotronV2VLClipSidecar(
        image_size=image,
        patch_size=patch,
        hidden_size=_metadata_int(metadata, "clip.vision.embedding_length"),
        intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
        num_layers=_metadata_int(metadata, "clip.vision.block_count"),
        num_heads=_metadata_int(metadata, "clip.vision.attention.head_count"),
        num_register_tokens=registers,
        projector_hidden_size=first[0],
        output_size=output,
        norm_eps=_metadata_float(metadata, "clip.vision.attention.layer_norm_epsilon"),
    )
    return _attach_contract(
        module,
        (("pixel_values", ir.DataType.FLOAT, (1, 3, image, image)),),
        squeeze_batch_dim=True,
    )


def _create_kimi(
    projector_type: str,
    metadata: Mapping[str, object],
    shapes: TensorShapes,
) -> nn.Module:
    hidden = _metadata_int(metadata, "clip.vision.embedding_length")
    patch = _metadata_int(metadata, "clip.vision.patch_size")
    position = _shape(shapes, "v.position_embd.weight")
    first = _shape(shapes, "mm.1.weight", 2)
    second = _shape(shapes, "mm.2.weight", 2)
    common = dict(
        depth=_metadata_int(metadata, "clip.vision.block_count"),
        hidden_size=hidden,
        intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
        num_heads=_metadata_int(metadata, "clip.vision.attention.head_count"),
        patch_size=patch,
        in_channels=_shape(shapes, "v.patch_embd.weight", 4)[1],
        projector_hidden_size=first[0],
        output_size=second[0],
        merge_size=_metadata_int(metadata, "clip.vision.projector.scale_factor"),
    )
    if projector_type == "kimik25":
        if len(position) != 3 or position[2] != hidden:
            raise ValueError("kimik25 position table must have shape [height,width,hidden].")
        module: nn.Module = KimiK25VisionSidecar(
            **common,
            stored_height=position[0],
            stored_width=position[1],
            norm_eps=_metadata_float(
                metadata,
                "clip.vision.attention.layer_norm_epsilon",
            ),
        )
    else:
        if len(position) != 2 or position[1] != hidden:
            raise ValueError("kimivl position table must have shape [positions,hidden].")
        side = math.isqrt(position[0])
        if side * side != position[0]:
            raise ValueError("kimivl position table must be square.")
        module = KimiVLVisionSidecar(**common, stored_height=side, stored_width=side)
    height = ir.SymbolicDim("height")
    width = ir.SymbolicDim("width")
    return _attach_contract(
        module,
        (("pixel_values", ir.DataType.FLOAT, (1, 3, height, width)),),
    )


def _create_exaone(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    pattern = _metadata_int(metadata, "clip.vision.n_wa_pattern")
    depth = _metadata_int(metadata, "clip.vision.block_count")
    full_attention = [index for index in range(depth) if (index + 1) % pattern == 0]
    patch = _metadata_int(metadata, "clip.vision.patch_size")
    module = Exaone45VisionSidecar(
        depth=depth,
        hidden_size=_metadata_int(metadata, "clip.vision.embedding_length"),
        intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
        num_heads=_metadata_int(metadata, "clip.vision.attention.head_count"),
        num_kv_heads=_metadata_int(metadata, "clip.vision.attention.head_count_kv"),
        patch_size=patch,
        in_channels=_shape(shapes, "v.patch_embd.weight", 4)[1],
        output_size=_shape(shapes, "mm.2.weight", 2)[0],
        fullatt_block_indexes=full_attention,
        window_size=_metadata_int(metadata, "clip.vision.window_size"),
        norm_eps=_metadata_float(metadata, "clip.vision.attention.layer_norm_epsilon"),
    )
    patches = ir.SymbolicDim("total_patches")
    images = ir.SymbolicDim("num_images")
    return _attach_contract(
        module,
        (
            ("pixel_values", ir.DataType.FLOAT, (patches, 2 * 3 * patch * patch)),
            ("image_grid_thw", ir.DataType.INT64, (images, 3)),
        ),
    )


def _create_mimovl(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    modes = metadata.get("clip.vision.wa_pattern_mode")
    depth = _metadata_int(metadata, "clip.vision.block_count")
    if (
        not isinstance(modes, list)
        or len(modes) != depth
        or any(isinstance(mode, bool) or mode not in (-1, 0, 1) for mode in modes)
    ):
        raise ValueError("mimovl wa_pattern_mode must contain one -1/0/1 entry per layer.")
    hidden = _metadata_int(metadata, "clip.vision.embedding_length")
    q_heads = _metadata_int(metadata, "clip.vision.attention.head_count")
    kv_heads = _metadata_int(metadata, "clip.vision.attention.head_count_kv")
    qkv = _shape(shapes, "v.blk.0.attn_qkv.weight", 2)
    head_dim, remainder = divmod(qkv[0], q_heads + 2 * kv_heads)
    if remainder:
        raise ValueError("mimovl fused QKV rows do not match its Q/KV head counts.")
    patch = _metadata_int(metadata, "clip.vision.patch_size")
    module = MiMoVLVisionSidecar(
        hidden_size=hidden,
        intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        patch_size=patch,
        window_modes=[int(mode) for mode in modes],
        projector_hidden_size=_shape(shapes, "mm.0.weight", 2)[0],
        output_size=_shape(shapes, "mm.2.weight", 2)[0],
    )
    tokens = ir.SymbolicDim("total_patches")
    merged = ir.SymbolicDim("merged_patches")
    return _attach_contract(
        module,
        (
            ("pixel_values", ir.DataType.FLOAT, (tokens, 2 * 3 * patch * patch)),
            ("row_position_ids", ir.DataType.INT64, (tokens, 2)),
            ("column_position_ids", ir.DataType.INT64, (tokens, 2)),
            ("window_bias", ir.DataType.FLOAT, (tokens, tokens)),
            ("column_indices", ir.DataType.INT64, (merged,)),
            ("inverse_column_indices", ir.DataType.INT64, (merged,)),
        ),
    )


def _create_minimax(metadata: Mapping[str, object], shapes: TensorShapes) -> nn.Module:
    hidden = _metadata_int(metadata, "clip.vision.embedding_length")
    patch = _metadata_int(metadata, "clip.vision.patch_size")
    image = _metadata_int(metadata, "clip.vision.image_size")
    merge = _optional_metadata_int(
        metadata,
        "clip.vision.spatial_merge_size",
        2,
    )
    first = _shape(shapes, "mm.1.weight", 2)
    second = _shape(shapes, "mm.2.weight", 2)
    merger_first = _shape(shapes, "mm.merger.fc1.weight", 2)
    merger_second = _shape(shapes, "mm.merger.fc2.weight", 2)
    module = MiniMaxM3VisionSidecar(
        hidden_size=hidden,
        intermediate_size=_metadata_int(metadata, "clip.vision.feed_forward_length"),
        num_heads=_metadata_int(metadata, "clip.vision.attention.head_count"),
        num_layers=_metadata_int(metadata, "clip.vision.block_count"),
        patch_size=patch,
        grid_height=image // patch,
        grid_width=image // patch,
        patch_mlp_size=first[0],
        projected_size=second[0],
        merger_mlp_size=merger_first[0],
        output_size=merger_second[0],
        merge_size=merge,
        norm_eps=_metadata_float(metadata, "clip.vision.attention.layer_norm_epsilon"),
    )
    patches = ir.SymbolicDim("total_patches")
    return _attach_contract(
        module,
        (
            ("pixel_values", ir.DataType.FLOAT, (patches, 2 * 3 * patch * patch)),
            ("grid_size", ir.DataType.INT64, (2,)),
        ),
    )


def create_remaining_vision_projector(
    projector_type: str,
    metadata: Mapping[str, object],
    tensor_shapes: TensorShapes,
) -> nn.Module:
    """Construct one exact route-specific sidecar encoder."""
    if projector_type == "janus_pro":
        return _create_janus(metadata, tensor_shapes)
    if projector_type == "lfm2":
        return _create_lfm2(metadata, tensor_shapes)
    if projector_type == "minicpmv4_6":
        return _create_minicpm(metadata, tensor_shapes)
    if projector_type == "yasa2":
        return _create_yasa(metadata, tensor_shapes)
    if projector_type == "hunyuanvl":
        return _create_hunyuan(metadata, tensor_shapes)
    if projector_type == "step3vl":
        return _create_step3(metadata, tensor_shapes)
    if projector_type == "cogvlm":
        return _create_cog(metadata, tensor_shapes)
    if projector_type == "nemotron_v2_vl":
        return _create_nemotron(metadata, tensor_shapes)
    if projector_type in {"kimik25", "kimivl"}:
        return _create_kimi(projector_type, metadata, tensor_shapes)
    if projector_type == "exaone4_5":
        return _create_exaone(metadata, tensor_shapes)
    if projector_type == "mimovl":
        return _create_mimovl(metadata, tensor_shapes)
    if projector_type == "minimax_m3":
        return _create_minimax(metadata, tensor_shapes)
    raise ValueError(f"Unknown remaining vision projector type {projector_type!r}.")


_STANDARD_BLOCK_MAP = {
    "ln1.weight": "layer_norm1.weight",
    "ln1.bias": "layer_norm1.bias",
    "ln2.weight": "layer_norm2.weight",
    "ln2.bias": "layer_norm2.bias",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_out.bias": "self_attn.out_proj.bias",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_up.bias": "mlp.up_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_down.bias": "mlp.down_proj.bias",
}


def _map_yasa(name: str) -> str | None:
    block = _YASA_BLOCK.fullmatch(name)
    if block is not None:
        stage, index, suffix = block.groups()
        mapped = {
            "dw.weight": "depthwise.weight",
            "dw.bias": "depthwise.bias",
            "ln.weight": "layer_norm.weight",
            "ln.bias": "layer_norm.bias",
            "pw1.weight": "pointwise_up.weight",
            "pw1.bias": "pointwise_up.bias",
            "grn.weight": "grn.weight",
            "grn.bias": "grn.bias",
            "pw2.weight": "pointwise_down.weight",
            "pw2.bias": "pointwise_down.bias",
        }.get(suffix)
        return None if mapped is None else f"stages.{stage}.blocks.{index}.{mapped}"
    down = _YASA_DOWN.fullmatch(name)
    if down is not None:
        stage, suffix = down.groups()
        mapped = {
            "ln.weight": "downsample_norm.weight",
            "ln.bias": "downsample_norm.bias",
            "conv.weight": "downsample_conv.weight",
            "conv.bias": "downsample_conv.bias",
        }.get(suffix)
        return None if mapped is None else f"stages.{stage}.{mapped}"
    return {
        "v.patch_embd.weight": "patch_embedding.weight",
        "v.patch_embd.bias": "patch_embedding.bias",
        "v.patch_ln.weight": "patch_layer_norm.weight",
        "v.patch_ln.bias": "patch_layer_norm.bias",
        "v.vision_pos_embed": "vision_position_embedding",
        "mm.0.weight": "projector_up.weight",
        "mm.0.bias": "projector_up.bias",
        "mm.2.weight": "projector_down.weight",
        "mm.2.bias": "projector_down.bias",
    }.get(name)


def map_remaining_projector_weight(name: str, projector_type: str) -> str | None:
    """Map one GGUF tensor to the route component's ONNX initializer name."""
    local: str | None
    block = _BLOCK.fullmatch(name)
    index = block.group(1) if block is not None else ""
    suffix = block.group(2) if block is not None else ""
    if projector_type == "cogvlm":
        if block is not None:
            local_suffix = {
                "ln1.weight": "input_layernorm.weight",
                "ln1.bias": "input_layernorm.bias",
                "ln2.weight": "post_attention_layernorm.weight",
                "ln2.bias": "post_attention_layernorm.bias",
                "attn_out.weight": "attention.proj.weight",
                "attn_out.bias": "attention.proj.bias",
                "ffn_up.weight": "mlp.up.weight",
                "ffn_up.bias": "mlp.up.bias",
                "ffn_down.weight": "mlp.down.weight",
                "ffn_down.bias": "mlp.down.bias",
            }.get(suffix)
            local = None if local_suffix is None else f"blocks.{index}.{local_suffix}"
        else:
            local = {
                "v.patch_embd.weight": "patch_embedding.weight",
                "v.patch_embd.bias": "patch_embedding.bias",
                "v.class_embd": "class_embedding",
                "v.position_embd.weight": "position_embedding",
                "mm.model.fc.weight": "projection.weight",
                "mm.post_fc_norm.weight": "projector_norm.weight",
                "mm.post_fc_norm.bias": "projector_norm.bias",
                "mm.up.weight": "projector_up.weight",
                "mm.gate.weight": "projector_gate.weight",
                "mm.down.weight": "projector_down.weight",
                "v.boi": "boi",
                "v.eoi": "eoi",
            }.get(name)
    elif projector_type == "nemotron_v2_vl":
        if block is not None:
            local_suffix = {
                "ln1.weight": "norm1.weight",
                "ln1.bias": "norm1.bias",
                "ln2.weight": "norm2.weight",
                "ln2.bias": "norm2.bias",
                "attn_qkv.weight": "attention.qkv.weight",
                "attn_qkv.bias": "attention.qkv.bias",
                "attn_out.weight": "attention.proj.weight",
                "attn_out.bias": "attention.proj.bias",
                "ffn_up.weight": "mlp.fc1.weight",
                "ffn_up.bias": "mlp.fc1.bias",
                "ffn_down.weight": "mlp.fc2.weight",
                "ffn_down.bias": "mlp.fc2.bias",
            }.get(suffix)
            local = None if local_suffix is None else f"blocks.{index}.{local_suffix}"
        else:
            local = {
                "v.patch_embd.weight": "patch_embedding.weight",
                "v.class_embd": "class_embedding",
                "v.position_embd.weight": "position_embedding",
                "mm.model.mlp.0.weight": "projector_norm_weight",
                "mm.model.mlp.1.weight": "projector_up_weight",
                "mm.model.mlp.3.weight": "projector_down_weight",
            }.get(name)
    elif projector_type == "janus_pro":
        local = map_fixed_siglip_sidecar_weight(name)
    elif projector_type == "yasa2":
        local = _map_yasa(name)
    else:
        local = None
        if projector_type == "hunyuanvl":
            if block is not None:
                local_suffix = {
                    "ln1.weight": "norm1.weight",
                    "ln1.bias": "norm1.bias",
                    "ln2.weight": "norm2.weight",
                    "ln2.bias": "norm2.bias",
                    "attn_out.weight": "attn.out_proj.weight",
                    "attn_out.bias": "attn.out_proj.bias",
                    "ffn_up.weight": "mlp_up.weight",
                    "ffn_up.bias": "mlp_up.bias",
                    "ffn_down.weight": "mlp_down.weight",
                    "ffn_down.bias": "mlp_down.bias",
                }.get(suffix)
                local = None if local_suffix is None else f"layers.{index}.{local_suffix}"
            else:
                local = {
                    "v.patch_embd.weight": "patch_embedding.proj.weight",
                    "v.patch_embd.bias": "patch_embedding.proj.bias",
                    "v.position_embd.weight": "position_embedding",
                    "mm.pre_norm.weight": "pre_projector_norm.weight",
                    "mm.0.weight": "projector_conv1.weight",
                    "mm.0.bias": "projector_conv1.bias",
                    "mm.2.weight": "projector_conv2.weight",
                    "mm.2.bias": "projector_conv2.bias",
                    "v.image_newline": "image_newline",
                    "mm.model.fc.weight": "projector.weight",
                    "mm.model.fc.bias": "projector.bias",
                    "mm.image_begin": "image_begin",
                    "mm.image_end": "image_end",
                    "mm.post_norm.weight": "post_projector_norm.weight",
                }.get(name)
        elif projector_type == "step3vl":
            if block is not None:
                local_suffix = {
                    "attn_qkv.weight": "attn.in_proj.weight",
                    "attn_qkv.bias": "attn.in_proj.bias",
                    "attn_out.weight": "attn.out_proj.weight",
                    "attn_out.bias": "attn.out_proj.bias",
                    "ln1.weight": "norm1.weight",
                    "ln1.bias": "norm1.bias",
                    "ln2.weight": "norm2.weight",
                    "ln2.bias": "norm2.bias",
                    "ls1.weight": "ls_1",
                    "ls2.weight": "ls_2",
                    "ffn_up.weight": "mlp_up.weight",
                    "ffn_up.bias": "mlp_up.bias",
                    "ffn_down.weight": "mlp_down.weight",
                    "ffn_down.bias": "mlp_down.bias",
                }.get(suffix)
                local = None if local_suffix is None else f"layers.{index}.{local_suffix}"
            else:
                local = {
                    "v.patch_embd.weight": "patch_embedding.proj.weight",
                    "v.position_embd.weight": "position_embedding",
                    "v.pre_ln.weight": "pre_layer_norm.weight",
                    "v.pre_ln.bias": "pre_layer_norm.bias",
                    "mm.0.weight": "downsample1.weight",
                    "mm.0.bias": "downsample1.bias",
                    "mm.1.weight": "downsample2.weight",
                    "mm.1.bias": "downsample2.bias",
                    "mm.model.fc.weight": "projector.weight",
                }.get(name)
        elif projector_type == "lfm2":
            if block is not None:
                mapped = _STANDARD_BLOCK_MAP.get(suffix)
                local = (
                    None if mapped is None else f"vision_tower.encoder.layers.{index}.{mapped}"
                )
            else:
                local = {
                    "v.patch_embd.weight": "vision_tower.embeddings.patch_embedding.weight",
                    "v.patch_embd.bias": "vision_tower.embeddings.patch_embedding.bias",
                    "v.position_embd.weight": "vision_tower.embeddings.position_embedding.weight",
                    "v.post_ln.weight": "vision_tower.post_layernorm.weight",
                    "v.post_ln.bias": "vision_tower.post_layernorm.bias",
                    "mm.input_norm.weight": "multi_modal_projector.layer_norm.weight",
                    "mm.input_norm.bias": "multi_modal_projector.layer_norm.bias",
                    "mm.1.weight": "multi_modal_projector.linear_1.weight",
                    "mm.1.bias": "multi_modal_projector.linear_1.bias",
                    "mm.2.weight": "multi_modal_projector.linear_2.weight",
                    "mm.2.bias": "multi_modal_projector.linear_2.bias",
                }.get(name)
        elif projector_type == "minicpmv4_6":
            if block is not None:
                mapped = _STANDARD_BLOCK_MAP.get(suffix)
                local = (
                    None if mapped is None else f"vision_tower.encoder.layers.{index}.{mapped}"
                )
            elif name.startswith("v.vit_merger."):
                suffix = name.removeprefix("v.vit_merger.")
                mapped = {
                    "ln1.weight": "layer_norm1.weight",
                    "ln1.bias": "layer_norm1.bias",
                    "attn_q.weight": "self_attn.q_proj.weight",
                    "attn_q.bias": "self_attn.q_proj.bias",
                    "attn_k.weight": "self_attn.k_proj.weight",
                    "attn_k.bias": "self_attn.k_proj.bias",
                    "attn_v.weight": "self_attn.v_proj.weight",
                    "attn_v.bias": "self_attn.v_proj.bias",
                    "attn_out.weight": "self_attn.out_proj.weight",
                    "attn_out.bias": "self_attn.out_proj.bias",
                    "ds_ln.weight": "pre_norm.weight",
                    "ds_ln.bias": "pre_norm.bias",
                    "ds_ffn_up.weight": "linear_1.weight",
                    "ds_ffn_up.bias": "linear_1.bias",
                    "ds_ffn_down.weight": "linear_2.weight",
                    "ds_ffn_down.bias": "linear_2.bias",
                }.get(suffix)
                local = None if mapped is None else f"vision_tower.encoder.vit_merger.{mapped}"
            else:
                local = {
                    "v.patch_embd.weight": "vision_tower.embeddings.patch_embedding.weight",
                    "v.patch_embd.bias": "vision_tower.embeddings.patch_embedding.bias",
                    "v.position_embd.weight": "vision_tower.embeddings.position_embedding.weight",
                    "v.post_ln.weight": "vision_tower.post_layernorm.weight",
                    "v.post_ln.bias": "vision_tower.post_layernorm.bias",
                    "mm.input_norm.weight": "merger.mlp.0.pre_norm.weight",
                    "mm.input_norm.bias": "merger.mlp.0.pre_norm.bias",
                    "mm.up.weight": "merger.mlp.0.linear_1.weight",
                    "mm.up.bias": "merger.mlp.0.linear_1.bias",
                    "mm.down.weight": "merger.mlp.0.linear_2.weight",
                    "mm.down.bias": "merger.mlp.0.linear_2.bias",
                }.get(name)
        elif projector_type in {"kimik25", "kimivl"}:
            if block is not None:
                attn_prefix = "qkv" if projector_type == "kimik25" else None
                local_suffix = {
                    "ln1.weight": "norm1.weight",
                    "ln1.bias": "norm1.bias",
                    "ln2.weight": "norm2.weight",
                    "ln2.bias": "norm2.bias",
                    "attn_qkv.weight": f"attn.{attn_prefix}.weight" if attn_prefix else None,
                    "attn_qkv.bias": f"attn.{attn_prefix}.bias" if attn_prefix else None,
                    "attn_q.weight": "attn.q_proj.weight",
                    "attn_q.bias": "attn.q_proj.bias",
                    "attn_k.weight": "attn.k_proj.weight",
                    "attn_k.bias": "attn.k_proj.bias",
                    "attn_v.weight": "attn.v_proj.weight",
                    "attn_v.bias": "attn.v_proj.bias",
                    "attn_out.weight": "attn.proj.weight",
                    "attn_out.bias": "attn.proj.bias",
                    "ffn_up.weight": "mlp.up_proj.weight",
                    "ffn_up.bias": "mlp.up_proj.bias",
                    "ffn_down.weight": "mlp.down_proj.weight",
                    "ffn_down.bias": "mlp.down_proj.bias",
                }.get(suffix)
                local = None if local_suffix is None else f"layers.{index}.{local_suffix}"
            else:
                local = {
                    "v.patch_embd.weight": "patch_embed.proj",
                    "v.patch_embd.bias": "patch_embed.bias",
                    "v.position_embd.weight": "position_embedding.position_embeddings",
                    "v.post_ln.weight": "final_layernorm.weight",
                    "v.post_ln.bias": "final_layernorm.bias",
                    "mm.input_norm.weight": "projector.input_norm.weight",
                    "mm.input_norm.bias": "projector.input_norm.bias",
                    "mm.1.weight": "projector.linear_1.weight",
                    "mm.1.bias": "projector.linear_1.bias",
                    "mm.2.weight": "projector.linear_2.weight",
                    "mm.2.bias": "projector.linear_2.bias",
                }.get(name)
        elif projector_type == "exaone4_5":
            if block is not None:
                local_suffix = {
                    "ln1.weight": "norm1.weight",
                    "ln2.weight": "norm2.weight",
                    "attn_qkv.weight": "attn.qkv.weight",
                    "attn_qkv.bias": "attn.qkv.bias",
                    "attn_out.weight": "attn.proj.weight",
                    "attn_out.bias": "attn.proj.bias",
                    "ffn_gate.weight": "mlp.gate_proj.weight",
                    "ffn_gate.bias": "mlp.gate_proj.bias",
                    "ffn_up.weight": "mlp.up_proj.weight",
                    "ffn_up.bias": "mlp.up_proj.bias",
                    "ffn_down.weight": "mlp.down_proj.weight",
                    "ffn_down.bias": "mlp.down_proj.bias",
                }.get(suffix)
                local = None if local_suffix is None else f"blocks.{index}.{local_suffix}"
            else:
                local = {
                    "v.patch_embd.weight": "patch_embed.weight_0",
                    "v.patch_embd.weight.1": "patch_embed.weight_1",
                    "v.post_ln.weight": "merger.post_layernorm.weight",
                    "mm.0.weight": "merger.linear_1.weight",
                    "mm.0.bias": "merger.linear_1.bias",
                    "mm.2.weight": "merger.linear_2.weight",
                    "mm.2.bias": "merger.linear_2.bias",
                }.get(name)
        elif projector_type == "mimovl":
            if block is not None:
                local_suffix = {
                    "ln1.weight": "norm1.weight",
                    "ln2.weight": "norm2.weight",
                    "attn_qkv.weight": "attn.qkv.weight",
                    "attn_qkv.bias": "attn.qkv.bias",
                    "attn_out.weight": "attn.proj.weight",
                    "attn_out.bias": "attn.proj.bias",
                    "attn_sinks": "attn.attn_sinks",
                    "ffn_gate.weight": "mlp.gate_proj.weight",
                    "ffn_gate.bias": "mlp.gate_proj.bias",
                    "ffn_up.weight": "mlp.up_proj.weight",
                    "ffn_up.bias": "mlp.up_proj.bias",
                    "ffn_down.weight": "mlp.down_proj.weight",
                    "ffn_down.bias": "mlp.down_proj.bias",
                }.get(suffix)
                local = None if local_suffix is None else f"blocks.{index}.{local_suffix}"
            else:
                local = {
                    "v.patch_embd.weight": "patch_embed.weight_0",
                    "v.patch_embd.weight.1": "patch_embed.weight_1",
                    "v.post_ln.weight": "projector.post_ln.weight",
                    "mm.0.weight": "projector.fc1.weight",
                    "mm.2.weight": "projector.fc2.weight",
                }.get(name)
        elif projector_type == "minimax_m3":
            if block is not None:
                local_suffix = {
                    "ln1.weight": "norm1.weight",
                    "ln1.bias": "norm1.bias",
                    "ln2.weight": "norm2.weight",
                    "ln2.bias": "norm2.bias",
                    "attn_q.weight": "attn.q_proj.weight",
                    "attn_q.bias": "attn.q_proj.bias",
                    "attn_k.weight": "attn.k_proj.weight",
                    "attn_k.bias": "attn.k_proj.bias",
                    "attn_v.weight": "attn.v_proj.weight",
                    "attn_v.bias": "attn.v_proj.bias",
                    "attn_out.weight": "attn.out_proj.weight",
                    "attn_out.bias": "attn.out_proj.bias",
                    "ffn_up.weight": "mlp.fc1.weight",
                    "ffn_up.bias": "mlp.fc1.bias",
                    "ffn_down.weight": "mlp.fc2.weight",
                    "ffn_down.bias": "mlp.fc2.bias",
                }.get(suffix)
                local = None if local_suffix is None else f"blocks.{index}.{local_suffix}"
            else:
                local = {
                    "v.patch_embd.weight": "patch_embed.weight_0",
                    "v.patch_embd.weight.1": "patch_embed.weight_1",
                    "mm.1.weight": "projector.patch_mlp.fc1.weight",
                    "mm.1.bias": "projector.patch_mlp.fc1.bias",
                    "mm.2.weight": "projector.patch_mlp.fc2.weight",
                    "mm.2.bias": "projector.patch_mlp.fc2.bias",
                    "mm.merger.fc1.weight": "projector.merger_mlp.fc1.weight",
                    "mm.merger.fc1.bias": "projector.merger_mlp.fc1.bias",
                    "mm.merger.fc2.weight": "projector.merger_mlp.fc2.weight",
                    "mm.merger.fc2.bias": "projector.merger_mlp.fc2.bias",
                }.get(name)
    return None if local is None else f"vision_encoder.{local}"


def _fused_hunyuan_state(mmproj_gguf: Any) -> tuple[dict[str, torch.Tensor], set[str]]:
    state: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    layers = _metadata_int(mmproj_gguf.metadata, "clip.vision.block_count")
    for layer in range(layers):
        for kind in ("weight", "bias"):
            names = [f"v.blk.{layer}.attn_{part}.{kind}" for part in ("q", "k", "v")]
            values = [
                np.asarray(mmproj_gguf.get_tensor(name), dtype=np.float32) for name in names
            ]
            state[f"vision_encoder.layers.{layer}.attn.in_proj.{kind}"] = torch.from_numpy(
                np.concatenate(values, axis=0).copy()
            )
            consumed.update(names)
    return state, consumed


def _fused_cog_state(mmproj_gguf: Any) -> tuple[dict[str, torch.Tensor], set[str]]:
    state: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    layers = _metadata_int(mmproj_gguf.metadata, "clip.vision.block_count")
    for layer in range(layers):
        for kind in ("weight", "bias"):
            names = [f"v.blk.{layer}.attn_{part}.{kind}" for part in ("q", "k", "v")]
            values = [
                np.asarray(mmproj_gguf.get_tensor(name), dtype=np.float32) for name in names
            ]
            target = f"vision_encoder.blocks.{layer}.attention.qkv.{kind}"
            state[target] = torch.from_numpy(np.concatenate(values, axis=0).copy())
            consumed.update(names)
    return state, consumed


def remaining_projector_state_dict(mmproj_gguf: Any, projector_type: str) -> dict:
    """Load every mapped sidecar tensor under its graph-local initializer name."""
    state: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    if projector_type == "hunyuanvl":
        state, consumed = _fused_hunyuan_state(mmproj_gguf)
    elif projector_type == "cogvlm":
        state, consumed = _fused_cog_state(mmproj_gguf)
    for name in mmproj_gguf.tensor_names:
        if name in consumed:
            continue
        if projector_type == "nemotron_v2_vl" and name.startswith(("a.", "mm.a.")):
            continue
        if projector_type == "mimovl" and name.startswith(("a.", "mm.a.")):
            continue
        mapped = map_remaining_projector_weight(name, projector_type)
        if mapped is None:
            continue
        values = np.asarray(mmproj_gguf.get_tensor(name), dtype=np.float32)
        if projector_type == "lfm2" and name == "v.patch_embd.weight":
            values = values.reshape(values.shape[0], -1)
        elif projector_type == "kimik25" and name == "v.position_embd.weight":
            values = values.transpose(2, 1, 0)
        elif projector_type == "yasa2" and ".grn." in name:
            values = values.reshape(-1)
        state[mapped] = torch.from_numpy(values.copy())
    return state


def validate_remaining_projector_shapes(mmproj_gguf: Any, projector_type: str) -> None:
    """Validate metadata-derived module dimensions against every mapped source tensor."""
    shapes = {
        name: tuple(int(dim) for dim in mmproj_gguf.get_tensor_shape(name))
        for name in mmproj_gguf.tensor_names
    }
    module = create_remaining_vision_projector(projector_type, mmproj_gguf.metadata, shapes)
    from mobius.tasks import GGUFVisionProjectorModel
    from mobius.tasks._base import _make_graph, _make_model

    wrapper = GGUFVisionProjectorModel(module)
    graph, builder = _make_graph(name=f"{projector_type}_shape_validation")
    input_schema = getattr(module, "input_schema", None)
    if not isinstance(input_schema, tuple):
        raise TypeError(f"{projector_type} projector does not declare an input_schema.")
    inputs = {
        name: builder.input(name, dtype=dtype, shape=list(shape))
        for name, dtype, shape in input_schema
    }
    image_features = wrapper.vision_encoder(builder.op, **inputs)
    if bool(getattr(module, "squeeze_batch_dim", False)):
        image_features = builder.op.Squeeze(image_features, [0])
    builder.add_output(image_features, "image_features")
    graph_model = _make_model(graph)
    parameters: dict[str, tuple[int, ...]] = {}
    for name, initializer in graph_model.graph.initializers.items():
        if initializer.const_value is not None:
            continue
        shape = initializer.shape
        if shape is None or any(not isinstance(dim, int) for dim in shape):
            raise ValueError(f"{projector_type} parameter {name!r} has no static shape.")
        parameters[name] = cast(tuple[int, ...], tuple(shape))
    mapped_targets: set[str] = set()
    for name, source_shape in shapes.items():
        mapped = map_remaining_projector_weight(name, projector_type)
        if mapped is None:
            continue
        mapped_targets.add(mapped)
        expected = parameters.get(mapped)
        if expected is None:
            raise ValueError(
                f"{projector_type} tensor {name!r} maps to unknown parameter {mapped!r}."
            )
        actual = source_shape
        if projector_type == "lfm2" and name == "v.patch_embd.weight":
            actual = (source_shape[0], math.prod(source_shape[1:]))
        elif projector_type == "kimik25" and name == "v.position_embd.weight":
            actual = (source_shape[2], source_shape[1], source_shape[0])
        elif projector_type == "yasa2" and ".grn." in name:
            actual = (math.prod(source_shape),)
        if actual != expected:
            raise ValueError(
                f"{projector_type} tensor {name!r} has shape {source_shape}, "
                f"but {mapped!r} expects {expected}."
            )
    if projector_type == "hunyuanvl":
        for layer in range(_metadata_int(mmproj_gguf.metadata, "clip.vision.block_count")):
            for kind in ("weight", "bias"):
                mapped_targets.add(f"vision_encoder.layers.{layer}.attn.in_proj.{kind}")
    elif projector_type == "cogvlm":
        for layer in range(_metadata_int(mmproj_gguf.metadata, "clip.vision.block_count")):
            for kind in ("weight", "bias"):
                mapped_targets.add(f"vision_encoder.blocks.{layer}.attention.qkv.{kind}")
    missing = sorted(set(parameters) - mapped_targets)
    if missing:
        raise ValueError(
            f"{projector_type} sidecar does not initialize graph parameter(s): {missing}"
        )


def validate_remaining_projector_state_dict(
    model: ir.Model,
    state_dict: Mapping[str, torch.Tensor],
    projector_type: str,
) -> None:
    """Require production mapping to initialize every non-constant graph parameter."""
    required = {
        name
        for name, initializer in model.graph.initializers.items()
        if initializer.const_value is None
    }
    provided = set(state_dict)
    missing = sorted(required - provided)
    unknown = sorted(provided - required)
    if missing or unknown:
        raise ValueError(
            f"{projector_type} weight mapping does not close over the exported graph: "
            f"missing={missing}, unknown={unknown}"
        )
