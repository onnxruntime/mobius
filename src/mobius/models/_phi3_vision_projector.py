# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Host-side HD feature transform and projector for Phi-3.5-Vision.

The mobius ``vision_encoder`` ONNX graph for ``phi3_v`` emits the raw CLIP
patch features ``(num_crops, num_patches, image_dim_out)`` — exactly the output
of HuggingFace ``Phi3ImageEmbedding.get_img_features``. Everything that follows
(the 2x2 spatial "HD" patch merge, the learnable ``sub_GN``/``glb_GN`` line
separators, and the ``img_projection`` MLP) is image-size dependent: the crop
grid, the separator interleaving, and the final token count all vary with the
input image resolution, so they cannot be expressed as a static ONNX graph.

This module reproduces that host-side tail faithfully with NumPy and PyTorch, mirroring
``modeling_phi3_v.py`` (``hd_feature_transform`` → ``reshape_hd_patches_2x2merge``
→ ``add_image_newline`` → ``img_projection``). The output is the flat sequence
of projected image embeddings ``(total_image_tokens, hidden_size)`` that the
mobius ``embedding`` ONNX model scatters into the ``<|image|>`` token positions.

The projector weights (``img_projection`` MLP and the ``sub_GN``/``glb_GN``
separators) are intentionally dropped from the ONNX graphs (see
``phi3_v._rename_phi3v_vision_weight``); :func:`load_phi3_vision_projector_weights`
loads them directly from the checkpoint so the transform can run host-side.
"""

from __future__ import annotations

import dataclasses
import glob
import os

import numpy as np
import torch

# HF checkpoint key prefix for the projector / separators (outside the CLIP
# ``img_processor`` vision tower, which is exported to ONNX separately).
_PROJECTOR_PREFIX = "model.vision_embed_tokens."

_GLOBAL_SEPARATOR_KEY = _PROJECTOR_PREFIX + "glb_GN"
_SUBLAYER_SEPARATOR_KEY = _PROJECTOR_PREFIX + "sub_GN"
_PROJECTION_FIRST_WEIGHT_KEY = _PROJECTOR_PREFIX + "img_projection.0.weight"
_PROJECTION_FIRST_BIAS_KEY = _PROJECTOR_PREFIX + "img_projection.0.bias"
_PROJECTION_SECOND_WEIGHT_KEY = _PROJECTOR_PREFIX + "img_projection.2.weight"
_PROJECTION_SECOND_BIAS_KEY = _PROJECTOR_PREFIX + "img_projection.2.bias"

# CLIP ViT-L/14-336 produces a 24x24 patch grid (576 patches).
_PATCH_GRID_SIDE = 24
# Each crop is a 336x336 tile.
_CROP_PIXEL_SIZE = 336


@dataclasses.dataclass(frozen=True)
class Phi3VisionProjectorWeights:
    """Host-side projector weights for the Phi-3.5-Vision HD feature transform.

    All arrays are stored in ``float32`` for numerically faithful comparison
    against the HuggingFace reference.

    Attributes:
        global_separator: ``glb_GN`` separator, shape ``(1, 1, image_dim_out * 4)``.
        sublayer_separator: ``sub_GN`` separator, shape ``(1, 1, 1, image_dim_out * 4)``.
        projection_first_weight: First ``img_projection`` Linear weight,
            shape ``(hidden_size, image_dim_out * 4)``.
        projection_first_bias: First Linear bias, shape ``(hidden_size,)``.
        projection_second_weight: Second Linear weight,
            shape ``(hidden_size, hidden_size)``.
        projection_second_bias: Second Linear bias, shape ``(hidden_size,)``.
    """

    global_separator: np.ndarray
    sublayer_separator: np.ndarray
    projection_first_weight: np.ndarray
    projection_first_bias: np.ndarray
    projection_second_weight: np.ndarray
    projection_second_bias: np.ndarray


def _resolve_checkpoint_directory(model_id_or_directory: str) -> str:
    """Return a local directory that contains the checkpoint safetensors.

    Accepts either a local path or a HuggingFace hub model id (which is
    resolved via the local cache; the weights must already be downloaded).
    """
    if os.path.isdir(model_id_or_directory):
        return model_id_or_directory

    from huggingface_hub import snapshot_download

    return snapshot_download(model_id_or_directory, local_files_only=True)


def load_phi3_vision_projector_weights(
    model_id_or_directory: str,
) -> Phi3VisionProjectorWeights:
    """Load the Phi-3.5-Vision projector + separator weights from a checkpoint.

    These tensors are omitted from the exported ONNX graphs (they belong to the
    host-side HD feature transform), so they are read directly from the
    checkpoint's safetensors shards here.

    Args:
        model_id_or_directory: Local checkpoint directory or a HuggingFace hub
            model id whose weights are already present in the local cache.

    Returns:
        The projector weights as ``float32`` NumPy arrays.

    Raises:
        FileNotFoundError: If no safetensors shards are found.
        KeyError: If an expected projector tensor is missing.
    """
    from safetensors import safe_open

    directory = _resolve_checkpoint_directory(model_id_or_directory)
    shard_paths = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No .safetensors shards found in checkpoint directory: {directory}"
        )

    wanted_keys = {
        _GLOBAL_SEPARATOR_KEY,
        _SUBLAYER_SEPARATOR_KEY,
        _PROJECTION_FIRST_WEIGHT_KEY,
        _PROJECTION_FIRST_BIAS_KEY,
        _PROJECTION_SECOND_WEIGHT_KEY,
        _PROJECTION_SECOND_BIAS_KEY,
    }
    collected: dict[str, np.ndarray] = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="numpy") as shard:
            shard_keys = set(shard.keys())
            for key in wanted_keys & shard_keys:
                collected[key] = shard.get_tensor(key).astype(np.float32)

    missing = wanted_keys - collected.keys()
    if missing:
        raise KeyError(
            f"Missing Phi-3.5-Vision projector weights in {directory}: {sorted(missing)}"
        )

    return Phi3VisionProjectorWeights(
        global_separator=collected[_GLOBAL_SEPARATOR_KEY],
        sublayer_separator=collected[_SUBLAYER_SEPARATOR_KEY],
        projection_first_weight=collected[_PROJECTION_FIRST_WEIGHT_KEY],
        projection_first_bias=collected[_PROJECTION_FIRST_BIAS_KEY],
        projection_second_weight=collected[_PROJECTION_SECOND_WEIGHT_KEY],
        projection_second_bias=collected[_PROJECTION_SECOND_BIAS_KEY],
    )


def _reshape_hd_patches_2x2_merge(
    patch_features: np.ndarray, height_crops: int, width_crops: int
) -> np.ndarray:
    """Merge each 2x2 block of patches into one channel-concatenated token.

    Mirrors ``modeling_phi3_v.reshape_hd_patches_2x2merge``.

    Args:
        patch_features: ``(num_images * num_crops, 576, image_dim_out)``.
        height_crops: Number of crop tiles stacked vertically.
        width_crops: Number of crop tiles stacked horizontally.

    Returns:
        ``(num_images, height_crops * 12, width_crops * 12, image_dim_out * 4)``.
    """
    total_crops, patch_count, channels = patch_features.shape
    if patch_count != _PATCH_GRID_SIDE * _PATCH_GRID_SIDE:
        raise ValueError(
            f"Expected {_PATCH_GRID_SIDE * _PATCH_GRID_SIDE} patches, got {patch_count}"
        )
    if total_crops % (height_crops * width_crops) != 0:
        raise ValueError(
            f"Crop count {total_crops} is not divisible by {height_crops} * {width_crops}"
        )
    num_images = total_crops // (height_crops * width_crops)
    grid_side = _PATCH_GRID_SIDE
    half_side = grid_side // 2
    merged = (
        patch_features.reshape(total_crops, grid_side, grid_side, channels)
        .reshape(total_crops, half_side, 2, half_side, 2, channels)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(total_crops, half_side * half_side, 4 * channels)
        .reshape(num_images, height_crops, width_crops, half_side, half_side, 4 * channels)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(num_images, height_crops * half_side, width_crops * half_side, 4 * channels)
    )
    return merged


def _add_image_newline(
    merged_features: np.ndarray, sublayer_separator: np.ndarray
) -> np.ndarray:
    """Append the learnable ``sub_GN`` separator to each patch row.

    Mirrors ``modeling_phi3_v.add_image_newline``.

    Args:
        merged_features: ``(num_images, height, width, hidden_dim)``.
        sublayer_separator: ``sub_GN`` of shape ``(1, 1, 1, hidden_dim)``.

    Returns:
        ``(num_images, height * (width + 1), hidden_dim)``.
    """
    num_images, height, _width, hidden_dim = merged_features.shape
    separator_column = np.broadcast_to(sublayer_separator, (num_images, height, 1, hidden_dim))
    with_newline = np.concatenate([merged_features, separator_column], axis=2)
    return with_newline.reshape(num_images, -1, hidden_dim)


def _apply_image_projection(
    tokens: np.ndarray, weights: Phi3VisionProjectorWeights
) -> np.ndarray:
    """Apply the ``img_projection`` MLP: Linear -> GELU -> Linear.

    Uses the exact (erf-based) GELU to match ``nn.GELU()``.

    Args:
        tokens: ``(num_tokens, image_dim_out * 4)``.
        weights: Projector weights.

    Returns:
        ``(num_tokens, hidden_size)``.
    """
    hidden = tokens @ weights.projection_first_weight.T + weights.projection_first_bias
    # Exact GELU (erf form), matching torch.nn.GELU()'s default approximate="none".
    hidden_tensor = torch.from_numpy(hidden)
    activated = (0.5 * hidden_tensor * (1.0 + torch.erf(hidden_tensor / np.sqrt(2.0)))).numpy()
    projected = activated @ weights.projection_second_weight.T + weights.projection_second_bias
    return projected.astype(np.float32)


def phi3_vision_hd_feature_transform(
    image_features: np.ndarray,
    image_sizes: np.ndarray,
    weights: Phi3VisionProjectorWeights,
) -> np.ndarray:
    """Reproduce Phi-3.5-Vision ``hd_feature_transform`` + ``img_projection``.

    Mirrors ``modeling_phi3_v.Phi3ImageEmbedding.hd_feature_transform`` with the
    fixed ``hd_transform_order == 'sub_glb'`` ordering used by Phi-3.5-Vision:
    for each image the token layout is ``[sub-crop features + newlines,
    glb_GN separator, global features + newlines]``.

    Args:
        image_features: Raw CLIP patch features of shape
            ``(num_images, num_crops + 1, 576, image_dim_out)`` — index 0 along
            axis 1 is the global (thumbnail) crop, the rest are the HD sub-crops.
        image_sizes: ``(num_images, 2)`` padded ``(height, width)`` in pixels
            per image, as produced by the HuggingFace image processor.
        weights: Host-side projector + separator weights.

    Returns:
        Projected image embeddings of shape ``(total_image_tokens, hidden_size)``
        ready to be scattered into ``<|image|>`` token positions.
    """
    image_features = np.asarray(image_features, dtype=np.float32)
    image_sizes = np.asarray(image_sizes)
    if image_features.ndim != 4:
        raise ValueError(
            "image_features must be (num_images, num_crops+1, 576, image_dim_out); "
            f"got shape {image_features.shape}"
        )
    num_images, num_crops_including_global, _, _ = image_features.shape
    if image_sizes.shape != (num_images, 2):
        raise ValueError(
            "image_sizes must be (num_images, 2) matching image_features; "
            f"got shape {image_sizes.shape} for {num_images} images"
        )

    global_separator = weights.global_separator.reshape(1, -1)  # (1, image_dim_out*4)
    sublayer_separator = weights.sublayer_separator  # (1, 1, 1, image_dim_out*4)

    # The global (thumbnail) crop is a special 1x1 HD case, shared across images.
    global_features = image_features[:, 0]  # (num_images, 576, image_dim_out)
    global_merged = _reshape_hd_patches_2x2_merge(global_features, 1, 1)
    global_with_newline = _add_image_newline(global_merged, sublayer_separator)

    per_image_token_blocks: list[np.ndarray] = []
    for image_index, (height, width) in enumerate(image_sizes):
        if height <= 0 or width <= 0 or height % _CROP_PIXEL_SIZE or width % _CROP_PIXEL_SIZE:
            raise ValueError(
                f"image_sizes[{image_index}] must contain positive multiples of "
                f"{_CROP_PIXEL_SIZE}; got ({height}, {width})"
            )
        height_crops = int(height) // _CROP_PIXEL_SIZE
        width_crops = int(width) // _CROP_PIXEL_SIZE
        crop_count = height_crops * width_crops
        available_sub_crops = num_crops_including_global - 1
        if crop_count > available_sub_crops:
            raise ValueError(
                f"image_sizes[{image_index}] requires {crop_count} HD crops, but "
                f"image_features provides only {available_sub_crops} sub-crops"
            )

        sub_features = image_features[image_index, 1 : 1 + crop_count]
        sub_merged = _reshape_hd_patches_2x2_merge(sub_features, height_crops, width_crops)
        sub_with_newline = _add_image_newline(sub_merged, sublayer_separator)

        per_image_token_blocks.extend(
            [
                sub_with_newline[0],
                global_separator,
                global_with_newline[image_index],
            ]
        )

    all_tokens = np.concatenate(per_image_token_blocks, axis=0)
    return _apply_image_projection(all_tokens, weights)
