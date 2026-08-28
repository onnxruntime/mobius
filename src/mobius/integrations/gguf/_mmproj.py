# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build a full Gemma4 multimodal ONNX package from GGUF (text + mmproj).

Gemma4's text backbone ships in one ``*.gguf`` file while its vision (and
audio) encoders live in a companion ``mmproj-*.gguf`` whose
``general.architecture`` is ``clip``.  :func:`build_gemma4_vlm_from_gguf`
assembles both into a single onnx-genai-ready :class:`ModelPackage`:

- **decoder** — the Gemma4 text decoder (from the text GGUF), taking
  ``inputs_embeds``.
- **vision_encoder** — the Gemma4 SigLIP vision encoder + projector (from the
  mmproj), taking ``pixel_values`` + ``pixel_position_ids``.
- **embedding** — scaled token lookup that fuses text + image features (built
  from the text config, reusing :class:`Gemma4EmbeddingModel`).

The mmproj ``clip.vision.*`` metadata is read into a :class:`VisionConfig`
(:func:`read_mmproj_vision_config`) and merged onto the text
:class:`Gemma4Config`; the mmproj ``v.*``/``mm.*`` tensors are mapped to their
HF names (:mod:`_mmproj_mapping`) so they flow through the same tested
``Gemma4Model.preprocess_weights`` path as a real HF checkpoint.

Audio: :func:`read_mmproj_audio_config` extracts the Conformer
:class:`Gemma4AudioConfig`; active ``gemma4a`` and encoder-free ``gemma4ua``
roles are retained by default. Callers may explicitly request a vision-only
package with ``include_audio=False``.
"""

from __future__ import annotations

__all__ = [
    "build_audio_projector_from_gguf",
    "build_generic_projector_vlm_from_gguf",
    "build_gemma3_vlm_from_gguf",
    "build_gemma4_vlm_from_gguf",
    "build_mmproj_from_gguf",
    "build_qwen_glm_projector_from_gguf",
    "build_remaining_vision_projector_from_gguf",
    "build_qwen_vlm_from_gguf",
    "build_vlm_from_gguf",
    "build_muse_glimmer_vlm_from_gguf",
    "read_mmproj_audio_config",
    "read_mmproj_gemma3_vision_config",
    "read_mmproj_generic_vision_config",
    "read_mmproj_qwen_vision_config",
    "read_mmproj_muse_glimmer_vision_config",
    "read_mmproj_vision_config",
]

import dataclasses
import json
import logging
import math
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mobius._model_package import ModelPackage
from mobius.integrations.gguf._arch_registry import MMPROJ_ARCHITECTURE
from mobius.integrations.gguf._mmproj_registry import (
    LLAMA_CPP_MMPROJ_SHA,
    MMProjModality,
    ProjectorSpec,
    get_projector_spec,
    projector_type_for_modality,
)
from mobius.integrations.gguf._spec import Support

if TYPE_CHECKING:
    from mobius._configs._sub_configs import VisionConfig
    from mobius.integrations.gguf._quantization_report import (
        GGUFQuantizationReport,
        QuantizationTensorRecord,
    )

logger = logging.getLogger(__name__)

# Gemma4 vision RoPE base frequency. The ``clip`` mmproj metadata does not store
# it, so use the HF Gemma4VisionAttention default. Source: HF Gemma4 vision.
_GEMMA4_VISION_ROPE_THETA = 100.0
# Gemma4 vision pooler spatial average pooling kernel (k x k). Not present in
# mmproj metadata; the SigLIP encoder pools N patches to N/k^2 soft tokens.
_DEFAULT_POOLING_KERNEL_SIZE = 3
_QWEN_VISION_WINDOW_SIZE = 112
_GEMMA3_POOLING_KERNEL_SIZE = 4

# Muse Glimmer's vision tower uses ordinary 2D RoPE. The mmproj stores no
# rope.freq_base, so use the published vision config value.
_MUSE_GLIMMER_VISION_ROPE_THETA = 10_000.0
# Full-attention stride: every 4th block is global, and so is the last block.
_MUSE_GLIMMER_VISION_FULL_ATTENTION_STRIDE = 4


def _with_muse_glimmer_media_token_ids(config: Any) -> Any:
    """Fill in the media placeholder ids a GGUF cannot carry.

    They belong to the tokenizer, not to the model metadata, so a converted
    config arrives with both unset. Leaving them that way is not cosmetic: the
    embedding graph compares ``input_ids`` against them, and the ort-genai
    exporter omits the field from ``genai_config`` entirely when it is ``None``,
    producing a package that cannot address video at all.
    """
    from mobius.models.muse_glimmer import IMAGE_TOKEN_ID, VIDEO_TOKEN_ID

    updates = {}
    if config.image_token_id is None:
        updates["image_token_id"] = IMAGE_TOKEN_ID
    if config.video_token_id is None:
        updates["video_token_id"] = VIDEO_TOKEN_ID
    return dataclasses.replace(config, **updates) if updates else config


def _muse_glimmer_temporal_patch_size(patch_embd: Any, *, patch_size: int) -> int:
    """Recover the temporal patch depth from the patch-embedding weight.

    ``clip.vision.*`` metadata carries no temporal depth. HF checkpoints fold
    ``temporal_patch_size`` frames into a single Conv3d kernel, but a converter
    is free to collapse that axis (llama.cpp emits a plain Conv2d kernel of
    ``[out, in_channels, patch, patch]``, which is exactly the image-only case
    with both frames summed). Trusting a hard-coded 2 therefore builds a tower
    whose patch embedding is twice as wide as the weight it is about to load,
    so derive the depth from the weight itself.
    """
    shape = tuple(int(dim) for dim in patch_embd.shape)
    if len(shape) < 2:
        raise ValueError(
            f"Muse Glimmer patch embedding must be at least 2D, got shape {shape}."
        )
    in_channels = shape[1]
    per_frame = in_channels * patch_size * patch_size
    elements = math.prod(shape[1:])
    if per_frame == 0 or elements % per_frame != 0:
        raise ValueError(
            f"Muse Glimmer patch embedding of shape {shape} is not divisible into "
            f"{in_channels}x{patch_size}x{patch_size} frames."
        )
    return elements // per_frame


def read_mmproj_muse_glimmer_vision_config(gguf_model: Any):
    """Extract a Muse Glimmer :class:`VisionConfig` from ``clip.vision.*``.

    Four things the metadata does not carry are recovered from the tensors or
    from the published vision config:

    ``position_embedding_height`` / ``width``
        The learned table is ``[grid**2, hidden]``; the grid is square, so the
        side is ``sqrt(rows)``.
    ``projector_intermediate_size``
        The width of the two adapter matrices, read from ``mm.0``.
    ``fullatt_block_indexes``
        Muse Glimmer alternates window and full attention on a stride of 4
        (blocks 3, 7, 11 …) *and* makes the final block full attention.
    ``temporal_patch_size`` / ``rope_theta``
        Architectural constants with no clip key; see the module constants.

    Args:
        gguf_model: A :class:`GGUFModel` for a ``clip`` mmproj file.

    Returns:
        A populated :class:`VisionConfig`, or ``None`` if the file has no
        vision encoder.
    """
    import math

    from mobius._configs._sub_configs import VisionConfig

    md = gguf_model.metadata
    if not md.get("clip.has_vision_encoder"):
        return None

    num_layers = int(md["clip.vision.block_count"])
    position_rows = int(gguf_model.get_tensor("v.position_embd.weight").shape[0])
    grid = math.isqrt(position_rows)
    if grid * grid != position_rows:
        raise ValueError(
            f"Muse Glimmer position embedding table has {position_rows} rows, "
            "which is not a square grid."
        )
    patch_embd = gguf_model.get_tensor("v.patch_embd.weight")
    projector_intermediate_size = int(gguf_model.get_tensor("mm.0.weight").shape[0])
    hidden_size = int(md["clip.vision.embedding_length"])
    merge_size = int(md.get("clip.vision.spatial_merge_size", 2))

    stride = _MUSE_GLIMMER_VISION_FULL_ATTENTION_STRIDE
    full_attention = sorted(
        {index for index in range(num_layers) if (index + 1) % stride == 0} | {num_layers - 1}
    )

    return VisionConfig(
        hidden_size=hidden_size,
        intermediate_size=int(md["clip.vision.feed_forward_length"]),
        num_hidden_layers=num_layers,
        num_attention_heads=int(md["clip.vision.attention.head_count"]),
        image_size=int(md["clip.vision.image_size"]),
        patch_size=int(md["clip.vision.patch_size"]),
        norm_eps=float(md.get("clip.vision.attention.layer_norm_epsilon", 1e-5)),
        in_channels=int(patch_embd.shape[1]),
        spatial_merge_size=merge_size,
        temporal_patch_size=_muse_glimmer_temporal_patch_size(
            patch_embd, patch_size=int(md["clip.vision.patch_size"])
        ),
        position_embedding_size=position_rows,
        position_embedding_height=grid,
        position_embedding_width=grid,
        fullatt_block_indexes=full_attention,
        projector_intermediate_size=projector_intermediate_size,
        # HF stores the pixel-shuffled width here (hidden * merge**2), not the
        # text hidden size that clip.vision.projection_dim reports.
        out_hidden_size=hidden_size * merge_size * merge_size,
        hidden_act="gelu",
        rope_theta=_MUSE_GLIMMER_VISION_ROPE_THETA,
    )


def read_mmproj_vision_config(gguf_model: Any):
    """Extract a Gemma4 :class:`VisionConfig` from ``clip.vision.*`` metadata.

    Args:
        gguf_model: A :class:`GGUFModel` for a ``clip`` mmproj file.

    Returns:
        A populated :class:`VisionConfig`, or ``None`` if the file has no
        vision encoder (``clip.has_vision_encoder`` is false/absent).
    """
    from mobius._configs._sub_configs import VisionConfig

    md = gguf_model.metadata
    if not md.get("clip.has_vision_encoder"):
        return None

    # Position embedding table size is authoritative from the tensor itself
    # ([2, pos_emb_size, hidden]); metadata carries no equivalent key.
    pos_emb_size = int(gguf_model.get_tensor("v.position_embd.weight").shape[1])

    return VisionConfig(
        hidden_size=int(md["clip.vision.embedding_length"]),
        intermediate_size=int(md["clip.vision.feed_forward_length"]),
        num_hidden_layers=int(md["clip.vision.block_count"]),
        num_attention_heads=int(md["clip.vision.attention.head_count"]),
        image_size=int(md["clip.vision.image_size"]),
        patch_size=int(md["clip.vision.patch_size"]),
        norm_eps=float(md.get("clip.vision.attention.layer_norm_epsilon", 1e-6)),
        # The F16 Gemma4 sidecar carries learned activation bounds for every
        # attention/MLP projection; they are part of ClippableLinear semantics.
        use_clipped_linears=True,
        position_embedding_size=pos_emb_size,
        pooling_kernel_size=_DEFAULT_POOLING_KERNEL_SIZE,
        hidden_act="gelu_pytorch_tanh",
        rope_theta=_GEMMA4_VISION_ROPE_THETA,
    )


def read_mmproj_gemma3_vision_config(gguf_model: Any):
    """Extract the pinned Gemma3 SigLIP configuration from ``clip.vision.*``."""
    from mobius._configs._sub_configs import VisionConfig

    md = gguf_model.metadata
    if not md.get("clip.has_vision_encoder"):
        return None
    image_size = int(md["clip.vision.image_size"])
    patch_size = int(md["clip.vision.patch_size"])
    patches_per_side = image_size // patch_size
    if image_size % patch_size or patches_per_side % _GEMMA3_POOLING_KERNEL_SIZE:
        raise ValueError(
            "Gemma3 image/patch grid must be divisible by the 4x4 projector pool."
        )
    position_rows = int(gguf_model.get_tensor_shape("v.position_embd.weight")[0])
    if position_rows != patches_per_side**2:
        raise ValueError(
            "Gemma3 position embedding rows do not match the image patch grid: "
            f"{position_rows} != {patches_per_side**2}."
        )
    return VisionConfig(
        hidden_size=int(md["clip.vision.embedding_length"]),
        intermediate_size=int(md["clip.vision.feed_forward_length"]),
        num_hidden_layers=int(md["clip.vision.block_count"]),
        num_attention_heads=int(md["clip.vision.attention.head_count"]),
        image_size=image_size,
        patch_size=patch_size,
        norm_eps=float(md["clip.vision.attention.layer_norm_epsilon"]),
        mm_tokens_per_image=(patches_per_side // _GEMMA3_POOLING_KERNEL_SIZE) ** 2,
        position_embedding_size=position_rows,
        hidden_act="gelu_pytorch_tanh",
        pooling_kernel_size=_GEMMA3_POOLING_KERNEL_SIZE,
    )


def read_mmproj_qwen_vision_config(gguf_model: Any, projector_type: str):
    """Recover the exact Qwen2/Qwen2.5-VL tower configuration."""
    from mobius._configs._sub_configs import VisionConfig

    md = gguf_model.metadata
    if not md.get("clip.has_vision_encoder"):
        return None
    if projector_type not in {"qwen2vl_merger", "qwen2.5vl_merger"}:
        raise ValueError(f"Unsupported Qwen VL projector type {projector_type!r}.")

    hidden_size = int(md["clip.vision.embedding_length"])
    num_layers = int(md["clip.vision.block_count"])
    patch_size = int(md["clip.vision.patch_size"])
    patch0 = tuple(gguf_model.get_tensor_shape("v.patch_embd.weight"))
    patch1 = tuple(gguf_model.get_tensor_shape("v.patch_embd.weight.1"))
    if patch0 != patch1 or len(patch0) != 4:
        raise ValueError(
            "Qwen VL temporal patch halves must have equal [out, channels, H, W] shapes."
        )
    if patch0[0] != hidden_size or patch0[2:] != (patch_size, patch_size):
        raise ValueError(
            f"Qwen VL patch tensors {patch0} do not match hidden={hidden_size}, "
            f"patch={patch_size}."
        )

    ffn_shape = tuple(gguf_model.get_tensor_shape("v.blk.0.ffn_up.weight"))
    if len(ffn_shape) != 2 or hidden_size not in ffn_shape:
        raise ValueError(
            f"Qwen VL FFN-up shape {ffn_shape} does not contain hidden size {hidden_size}."
        )
    intermediate_size = ffn_shape[0] if ffn_shape[1] == hidden_size else ffn_shape[1]

    full_attention: list[int] | None = None
    if projector_type == "qwen2.5vl_merger":
        pattern = int(md["clip.vision.n_wa_pattern"])
        if pattern <= 0:
            raise ValueError("clip.vision.n_wa_pattern must be positive.")
        full_attention = list(range(pattern - 1, num_layers, pattern))

    return VisionConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=int(md["clip.vision.attention.head_count"]),
        image_size=int(md["clip.vision.image_size"]),
        patch_size=patch_size,
        norm_eps=float(md["clip.vision.attention.layer_norm_epsilon"]),
        in_channels=int(patch0[1]),
        out_hidden_size=int(md["clip.vision.projection_dim"]),
        spatial_merge_size=2,
        temporal_patch_size=2,
        fullatt_block_indexes=full_attention,
        window_size=_QWEN_VISION_WINDOW_SIZE,
    )


def read_mmproj_generic_vision_config(mmproj_gguf: Any) -> VisionConfig | None:
    """Extract the common CLIP/SigLIP tower metadata used by generic sidecars."""
    from mobius._configs._sub_configs import VisionConfig

    md = mmproj_gguf.metadata
    if not bool(md.get("clip.has_vision_encoder", False)):
        return None
    required = (
        "clip.vision.image_size",
        "clip.vision.patch_size",
        "clip.vision.embedding_length",
        "clip.vision.feed_forward_length",
        "clip.vision.attention.head_count",
        "clip.vision.attention.layer_norm_epsilon",
        "clip.vision.block_count",
    )
    missing = [key for key in required if key not in md]
    if missing:
        raise ValueError(
            "Generic GGUF projector is missing required vision metadata: " + ", ".join(missing)
        )
    projector_type = projector_type_for_modality(md, MMProjModality.VISION)
    feature_layers = md.get("clip.vision.feature_layer")
    feature_layer: int | None = None
    if feature_layers is not None:
        if not isinstance(feature_layers, list) or any(
            isinstance(layer, bool) or not isinstance(layer, int) for layer in feature_layers
        ):
            raise ValueError("clip.vision.feature_layer must be an array of integer indices.")
        if not feature_layers:
            raise ValueError("clip.vision.feature_layer must contain exactly one index.")
        if len(feature_layers) > 1:
            raise NotImplementedError(
                "Generic GGUF projectors do not support concatenating multiple "
                "clip.vision.feature_layer outputs."
            )
        if projector_type not in {"mlp", "ldp", "ldpv2"}:
            raise NotImplementedError(
                f"{projector_type} does not support explicit clip.vision.feature_layer."
            )
        feature_layer = feature_layers[0]
        block_count = int(md["clip.vision.block_count"])
        if not 0 <= feature_layer <= block_count:
            raise ValueError(
                f"clip.vision.feature_layer index {feature_layer} is outside the "
                f"valid hidden-state range 0..{block_count}."
            )
    return VisionConfig(
        image_size=int(md["clip.vision.image_size"]),
        patch_size=int(md["clip.vision.patch_size"]),
        hidden_size=int(md["clip.vision.embedding_length"]),
        intermediate_size=int(md["clip.vision.feed_forward_length"]),
        num_attention_heads=int(md["clip.vision.attention.head_count"]),
        num_hidden_layers=int(md["clip.vision.block_count"]),
        norm_eps=float(md["clip.vision.attention.layer_norm_epsilon"]),
        hidden_act=(
            "gelu_pytorch_tanh" if bool(md.get("clip.use_gelu", False)) else "quick_gelu"
        ),
        feature_layer=feature_layer,
    )


def read_mmproj_audio_config(gguf_model: Any):
    """Extract a Gemma4 :class:`Gemma4AudioConfig` from ``clip.audio.*``.

    Args:
        gguf_model: A :class:`GGUFModel` for a ``clip`` mmproj file.

    Returns:
        A populated :class:`Gemma4AudioConfig`, or ``None`` if the file has no
        audio encoder (``clip.has_audio_encoder`` is false/absent).
    """
    from mobius._configs._sub_configs import Gemma4AudioConfig

    md = gguf_model.metadata
    if not md.get("clip.has_audio_encoder"):
        return None

    # Subsampling conv channel sizes are read from the conv tensors:
    # a.conv1d.0.weight [c0, 1, 3, 3], a.conv1d.1.weight [c1, c0, 3, 3].
    conv0 = int(gguf_model.get_tensor("a.conv1d.0.weight").shape[0])
    conv1 = int(gguf_model.get_tensor("a.conv1d.1.weight").shape[0])
    # Projector output dim = mm.a.input_projection input width.
    output_proj_dims = int(gguf_model.get_tensor("mm.a.input_projection.weight").shape[1])

    return Gemma4AudioConfig(
        input_size=int(md["clip.audio.num_mel_bins"]),
        hidden_size=int(md["clip.audio.embedding_length"]),
        num_layers=int(md["clip.audio.block_count"]),
        attention_heads=int(md["clip.audio.attention.head_count"]),
        linear_units=int(md["clip.audio.feed_forward_length"]),
        subsampling_conv_channels=[conv0, conv1],
        output_proj_dims=output_proj_dims,
        # The converter historically serialized 1e-5 here, while every
        # Gemma4 audio graph and the upstream loader use 1e-6.
        rms_norm_eps=1e-6,
    )


def _resolve_local_path(path: str | Path) -> Any:
    from mobius.integrations.gguf._builder import _resolve_gguf_path

    return _resolve_gguf_path(path)


def _open_text_gguf(path: Any) -> Any:
    from mobius.integrations.gguf._shard_set import open_gguf_model

    return open_gguf_model(path)


def _resolve_mmproj_companion_path(path: str | Path) -> Any:
    from mobius.integrations.gguf._builder import (
        _resolve_mmproj_companion_path as resolve_companion,
    )

    return resolve_companion(path)


_CLIPPING_BOUND_SUFFIXES = (".input_min", ".input_max", ".output_min", ".output_max")
_FLOAT_MMPROJ_QTYPES = frozenset({"F32", "F16", "BF16"})


def _canonical_text_architecture(architecture: str) -> str:
    from mobius.integrations.gguf._arch_registry import try_get_arch_spec

    arch_spec = try_get_arch_spec(architecture)
    return architecture if arch_spec is None else arch_spec.gguf_arch


def _validate_mmproj_container_type(mmproj_gguf: Any) -> None:
    """Validate only the serialized sidecar role, never publisher identity labels.

    Real GGUF pairs commonly carry paths, download labels, case differences, or
    omit ``general.name`` entirely. Compatibility is instead established below
    from the text architecture, projector type, exact metadata/tensor closure,
    tensor shapes, output width, and tokenizer media-token content.
    """
    general_type = mmproj_gguf.metadata.get("general.type")
    if general_type not in (None, "mmproj", "clip-vision"):
        raise ValueError(
            "clip sidecar general.type must be 'mmproj' or 'clip-vision', "
            f"got {general_type!r}."
        )


def _collect_block_tensors(
    names: set[str],
    *,
    prefix: str,
    suffixes: tuple[str, ...],
) -> tuple[dict[int, set[str]], set[str]]:
    blocks: dict[int, set[str]] = {}
    matched: set[str] = set()
    allowed_suffixes = set(suffixes)
    for name in names:
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :]
        index_text, separator, suffix = remainder.partition(".")
        if separator and index_text.isdecimal() and suffix in allowed_suffixes:
            blocks.setdefault(int(index_text), set()).add(suffix)
            matched.add(name)
    return blocks, matched


def _validate_block_tensor_set(
    *,
    projector_type: str,
    modality: MMProjModality,
    blocks: dict[int, set[str]],
    block_count: int,
    required_suffixes: tuple[str, ...],
    suffix_variants: tuple[tuple[str, ...], ...] = (),
) -> None:
    expected_layers = set(range(block_count))
    if set(blocks) != expected_layers:
        raise ValueError(
            f"{projector_type} {modality.value} block indices are "
            f"{sorted(blocks)}, expected {sorted(expected_layers)}."
        )
    if suffix_variants:
        variants = tuple(set(variant) for variant in suffix_variants)
        mismatched = {
            layer: sorted(blocks[layer])
            for layer in sorted(blocks)
            if not any(blocks[layer] == variant for variant in variants)
        }
        if mismatched:
            raise ValueError(
                f"{projector_type} mmproj has unsupported {modality.value} block "
                f"suffix variants: {mismatched}"
            )
        return

    required = set(required_suffixes)
    missing = {
        layer: sorted(required - blocks[layer])
        for layer in sorted(blocks)
        if required - blocks[layer]
    }
    if missing:
        raise ValueError(
            f"{projector_type} mmproj is missing required {modality.value} block "
            f"suffixes: {missing}"
        )


def _validate_gemma4_audio_metadata(metadata: dict[str, Any]) -> None:
    """Validate every pinned field required by an active Gemma4 audio encoder."""
    if metadata["clip.has_audio_encoder"] is not True:
        raise ValueError("clip.has_audio_encoder must be boolean true for gemma4a.")
    positive_integer_keys = (
        "clip.audio.embedding_length",
        "clip.audio.projection_dim",
        "clip.audio.attention.head_count",
        "clip.audio.num_mel_bins",
    )
    for key in positive_integer_keys:
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{key} must be a positive integer, got {value!r}.")
    for key in ("clip.audio.feed_forward_length", "clip.audio.block_count"):
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer, got {value!r}.")
    epsilon = metadata["clip.audio.attention.layer_norm_epsilon"]
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float, np.integer, np.floating))
        or not math.isfinite(float(epsilon))
        or epsilon <= 0
    ):
        raise ValueError(
            "clip.audio.attention.layer_norm_epsilon must be a positive finite "
            f"number, got {epsilon!r}."
        )


def _validate_mmproj_tensor_closure(mmproj_gguf: Any, spec: ProjectorSpec) -> None:
    """Validate the complete sidecar inventory, including deferred companions."""
    names = set(mmproj_gguf.tensor_names)
    top = set(spec.required_top_tensors) | set(spec.optional_top_tensors)
    if spec.block_prefix is None:
        block_names: dict[int, set[str]] = {}
        matched: set[str] = set()
    else:
        allowed_suffixes = (
            tuple(
                sorted(
                    {suffix for variant in spec.block_suffix_variants for suffix in variant}
                )
            )
            if spec.block_suffix_variants
            else spec.block_suffixes
        )
        block_names, matched = _collect_block_tensors(
            names,
            prefix=spec.block_prefix,
            suffixes=allowed_suffixes,
        )
    for pattern in spec.auxiliary_tensor_patterns:
        expression = re.compile(pattern)
        matched.update(name for name in names if expression.fullmatch(name))
    matched.update(top & names)

    active_companions = []
    for companion in spec.companion_tensors:
        presence_key = f"clip.has_{companion.modality.value.replace('.', '_')}_encoder"
        if not mmproj_gguf.metadata.get(presence_key):
            continue
        actual_type = projector_type_for_modality(mmproj_gguf.metadata, companion.modality)
        if actual_type != companion.projector_type:
            raise ValueError(
                f"{spec.projector_type} sidecar declares companion "
                f"{companion.modality.value} projector {actual_type!r}, expected "
                f"{companion.projector_type!r}."
            )
        missing_metadata = [
            key for key in companion.required_metadata if key not in mmproj_gguf.metadata
        ]
        if missing_metadata:
            raise ValueError(
                f"{companion.projector_type} companion is missing required metadata: "
                f"{missing_metadata}"
            )
        if companion.projector_type == "gemma4a":
            _validate_gemma4_audio_metadata(mmproj_gguf.metadata)
        companion_top = set(companion.required_top_tensors)
        companion_blocks, companion_matched = _collect_block_tensors(
            names,
            prefix=companion.block_prefix,
            suffixes=companion.block_suffixes,
        )
        missing_top = sorted(companion_top - names)
        if missing_top:
            raise ValueError(
                f"{companion.projector_type} companion is missing required tensor(s): "
                f"{missing_top}"
            )
        block_count_key = f"clip.{companion.modality.value}.block_count"
        _validate_block_tensor_set(
            projector_type=companion.projector_type,
            modality=companion.modality,
            blocks=companion_blocks,
            block_count=int(mmproj_gguf.metadata[block_count_key]),
            required_suffixes=companion.block_suffixes,
        )
        matched.update(companion_top & names)
        matched.update(companion_matched)
        active_companions.append(companion)

    quarantined_names: set[str] = set()
    for deferred in spec.deferred_companions:
        presence_key = f"clip.has_{deferred.modality.value.replace('.', '_')}_encoder"
        if not mmproj_gguf.metadata.get(presence_key):
            continue
        actual_type = projector_type_for_modality(
            mmproj_gguf.metadata,
            deferred.modality,
        )
        if actual_type != deferred.projector_type:
            raise ValueError(
                f"{spec.projector_type} sidecar declares deferred "
                f"{deferred.modality.value} projector {actual_type!r}, expected "
                f"{deferred.projector_type!r}."
            )
        for prefix in deferred.tensor_prefixes:
            prefixed = {name for name in names if name.startswith(prefix)}
            if not prefixed:
                raise ValueError(
                    f"{deferred.projector_type} deferred companion has no tensors "
                    f"under required namespace {prefix!r}."
                )
            quarantined_names.update(prefixed)
        matched.update(quarantined_names)

    unexpected = sorted(names - matched)
    if unexpected:
        raise ValueError(
            f"{spec.projector_type} mmproj has tensors outside the pinned suffix-exact "
            f"loader closure: {unexpected}. Projector tensors are never dropped."
        )

    missing_top = sorted(set(spec.required_top_tensors) - names)
    if missing_top:
        raise ValueError(
            f"{spec.projector_type} mmproj is missing required tensor(s): {missing_top}"
        )
    if spec.block_prefix is not None:
        primary_modality = spec.primary_modality
        layers = int(mmproj_gguf.metadata[f"clip.{primary_modality.value}.block_count"])
        _validate_block_tensor_set(
            projector_type=spec.projector_type,
            modality=primary_modality,
            blocks=block_names,
            block_count=layers,
            required_suffixes=spec.block_suffixes,
            suffix_variants=spec.block_suffix_variants,
        )

    for name in sorted(names):
        if name in quarantined_names:
            continue
        qtype = mmproj_gguf.get_tensor_type(name).name
        is_calibration = name.endswith(_CLIPPING_BOUND_SUFFIXES)
        if is_calibration and qtype != "F32":
            raise ValueError(
                f"{name} is a clipping-bound tensor and must be F32, got {qtype}."
            )
        if not is_calibration and qtype not in _FLOAT_MMPROJ_QTYPES:
            raise NotImplementedError(
                f"{spec.projector_type} mmproj tensor {name!r} uses packed {qtype}. "
                "Mobius does not preserve packed vision/audio/projector tensors; use "
                "an F16, BF16, or F32 mmproj so every role is explicitly dequantized."
            )

    if spec.primary_modality is MMProjModality.VISION:
        _validate_supported_mmproj_shapes(mmproj_gguf, spec)
    elif spec.primary_modality is MMProjModality.AUDIO:
        _validate_gemma4_audio_metadata(mmproj_gguf.metadata)
    for companion in active_companions:
        if companion.projector_type == "gemma4a":
            _validate_gemma4_audio_companion_shapes(mmproj_gguf, companion)


def _expect_mmproj_shape(mmproj_gguf: Any, name: str, expected: tuple[int, ...]) -> None:
    actual = mmproj_gguf.get_tensor_shape(name)
    if actual != expected:
        raise ValueError(f"mmproj tensor {name!r} has shape {actual}, expected {expected}.")


def _validate_gemma4_audio_companion_shapes(mmproj_gguf: Any, companion: Any) -> None:
    """Validate the deferred Gemma4 audio inventory without claiming graph support."""
    md = mmproj_gguf.metadata
    hidden = int(md["clip.audio.embedding_length"])
    intermediate = int(md["clip.audio.feed_forward_length"])
    layers = int(md["clip.audio.block_count"])
    heads = int(md["clip.audio.attention.head_count"])
    projection = int(md["clip.audio.projection_dim"])
    if hidden <= 0 or heads <= 0 or hidden % heads:
        raise ValueError(
            "gemma4a audio hidden/head dimensions are invalid: "
            f"embedding_length={hidden}, head_count={heads}."
        )
    head_dim = hidden // heads

    conv0_shape = mmproj_gguf.get_tensor_shape("a.conv1d.0.weight")
    if len(conv0_shape) != 4 or conv0_shape[1:] != (1, 3, 3):
        raise ValueError(
            f"a.conv1d.0.weight must have shape [channels, 1, 3, 3], got {conv0_shape}."
        )
    conv1_shape = mmproj_gguf.get_tensor_shape("a.conv1d.1.weight")
    if len(conv1_shape) != 4 or conv1_shape[1:] != (conv0_shape[0], 3, 3):
        raise ValueError(
            "a.conv1d.1.weight must have shape [channels, conv0_channels, 3, 3], "
            f"got {conv1_shape}."
        )
    _expect_mmproj_shape(mmproj_gguf, "a.conv1d.0.norm.weight", (conv0_shape[0],))
    _expect_mmproj_shape(mmproj_gguf, "a.conv1d.1.norm.weight", (conv1_shape[0],))
    _expect_mmproj_shape(mmproj_gguf, "a.input_projection.weight", (hidden, hidden))
    _expect_mmproj_shape(mmproj_gguf, "a.pre_encode.out.weight", (projection, hidden))
    _expect_mmproj_shape(mmproj_gguf, "a.pre_encode.out.bias", (projection,))
    _expect_mmproj_shape(mmproj_gguf, "mm.a.input_projection.weight", (projection, projection))

    matrix_shapes = {
        "attn_q.weight": (hidden, hidden),
        "attn_k.weight": (hidden, hidden),
        "attn_v.weight": (hidden, hidden),
        "attn_out.weight": (hidden, hidden),
        "attn_k_rel.weight": (hidden, hidden),
        "conv_pw1.weight": (2 * hidden, hidden),
        "conv_dw.weight": (hidden, 5),
        "conv_pw2.weight": (hidden, hidden),
        "ffn_up.weight": (intermediate, hidden),
        "ffn_down.weight": (hidden, intermediate),
        "ffn_up_1.weight": (intermediate, hidden),
        "ffn_down_1.weight": (hidden, intermediate),
    }
    vector_shapes = {
        "ffn_norm.weight": (hidden,),
        "ffn_post_norm.weight": (hidden,),
        "ffn_norm_1.weight": (hidden,),
        "ffn_post_norm_1.weight": (hidden,),
        "attn_pre_norm.weight": (hidden,),
        "attn_post_norm.weight": (hidden,),
        "ln2.weight": (hidden,),
        "conv_norm.weight": (hidden,),
        "norm_conv.weight": (hidden,),
        "per_dim_scale.weight": (head_dim,),
    }
    for layer in range(layers):
        prefix = f"{companion.block_prefix}{layer}."
        for suffix, shape in (*matrix_shapes.items(), *vector_shapes.items()):
            _expect_mmproj_shape(mmproj_gguf, prefix + suffix, shape)
        for suffix in companion.block_suffixes:
            if suffix.endswith(_CLIPPING_BOUND_SUFFIXES):
                _expect_mmproj_shape(mmproj_gguf, prefix + suffix, (1,))


def _validate_supported_mmproj_shapes(mmproj_gguf: Any, spec: ProjectorSpec) -> None:
    if spec.sidecar_builder == "core_vlm_projector":
        from mobius.integrations.gguf._core_vlm_projector import (
            validate_core_vlm_projector_shapes,
        )

        validate_core_vlm_projector_shapes(mmproj_gguf, spec.projector_type)
        return
    if spec.sidecar_builder == "qwen_glm_projector":
        from mobius.integrations.gguf._qwen_glm_projector import (
            validate_qwen_glm_projector_shapes,
        )

        validate_qwen_glm_projector_shapes(mmproj_gguf, spec.projector_type)
        return
    if spec.sidecar_builder == "remaining_vision_projector":
        from mobius.integrations.gguf._remaining_projectors import (
            validate_remaining_projector_shapes,
        )

        validate_remaining_projector_shapes(mmproj_gguf, spec.projector_type)
        return
    if spec.primary_modality is not MMProjModality.VISION:
        raise RuntimeError(
            f"{spec.projector_type} declares a non-vision primary modality without "
            "an architecture-specific shape validator."
        )
    md = mmproj_gguf.metadata
    hidden = int(md["clip.vision.embedding_length"])
    intermediate = int(md["clip.vision.feed_forward_length"])
    layers = int(md["clip.vision.block_count"])
    heads = int(md["clip.vision.attention.head_count"])
    patch = int(md["clip.vision.patch_size"])
    projection = int(md["clip.vision.projection_dim"])
    if hidden <= 0 or heads <= 0 or hidden % heads:
        raise ValueError(
            f"{spec.projector_type} vision hidden/head dimensions are invalid: "
            f"embedding_length={hidden}, head_count={heads}."
        )

    patch_shape = mmproj_gguf.get_tensor_shape("v.patch_embd.weight")
    if spec.projector_type == "gemma3":
        expected_patch_shape = (hidden, 3, patch, patch)
        if patch_shape != expected_patch_shape:
            raise ValueError(
                "Gemma3 v.patch_embd.weight must have graph-compatible shape "
                f"{expected_patch_shape}, got {patch_shape}."
            )
        patches_per_side = int(md["clip.vision.image_size"]) // patch
        _expect_mmproj_shape(
            mmproj_gguf,
            "v.position_embd.weight",
            (patches_per_side**2, hidden),
        )
        for name in (
            "v.patch_embd.bias",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.soft_emb_norm.weight",
        ):
            _expect_mmproj_shape(mmproj_gguf, name, (hidden,))
        _expect_mmproj_shape(mmproj_gguf, "mm.input_projection.weight", (hidden, projection))
        for layer in range(layers):
            prefix = f"v.blk.{layer}."
            for stem in ("ln1", "ln2"):
                for kind in ("weight", "bias"):
                    _expect_mmproj_shape(mmproj_gguf, prefix + stem + "." + kind, (hidden,))
            for stem in ("attn_q", "attn_k", "attn_v", "attn_out"):
                _expect_mmproj_shape(mmproj_gguf, prefix + stem + ".weight", (hidden, hidden))
                _expect_mmproj_shape(mmproj_gguf, prefix + stem + ".bias", (hidden,))
            _expect_mmproj_shape(mmproj_gguf, prefix + "ffn_up.weight", (hidden, intermediate))
            _expect_mmproj_shape(mmproj_gguf, prefix + "ffn_up.bias", (hidden,))
            _expect_mmproj_shape(
                mmproj_gguf, prefix + "ffn_down.weight", (intermediate, hidden)
            )
            _expect_mmproj_shape(mmproj_gguf, prefix + "ffn_down.bias", (intermediate,))
        return

    if spec.projector_type in {"qwen2vl_merger", "qwen2.5vl_merger"}:
        expected_patch_shape = (hidden, 3, patch, patch)
        for name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
            _expect_mmproj_shape(mmproj_gguf, name, expected_patch_shape)
        merged = hidden * 4
        _expect_mmproj_shape(mmproj_gguf, "v.post_ln.weight", (hidden,))
        _expect_mmproj_shape(mmproj_gguf, "mm.0.weight", (merged, merged))
        _expect_mmproj_shape(mmproj_gguf, "mm.0.bias", (merged,))
        _expect_mmproj_shape(mmproj_gguf, "mm.2.weight", (projection, merged))
        _expect_mmproj_shape(mmproj_gguf, "mm.2.bias", (projection,))

        qwen2 = spec.projector_type == "qwen2vl_merger"
        ffn_shape = mmproj_gguf.get_tensor_shape("v.blk.0.ffn_up.weight")
        vision_intermediate = (
            ffn_shape[1] if qwen2 and ffn_shape[0] == hidden else ffn_shape[0]
        )
        for layer in range(layers):
            prefix = f"v.blk.{layer}."
            for stem in ("attn_q", "attn_k", "attn_v", "attn_out"):
                _expect_mmproj_shape(mmproj_gguf, prefix + stem + ".weight", (hidden, hidden))
                _expect_mmproj_shape(mmproj_gguf, prefix + stem + ".bias", (hidden,))
            if qwen2:
                for stem in ("ln1", "ln2"):
                    _expect_mmproj_shape(mmproj_gguf, prefix + stem + ".weight", (hidden,))
                    _expect_mmproj_shape(mmproj_gguf, prefix + stem + ".bias", (hidden,))
                _expect_mmproj_shape(
                    mmproj_gguf,
                    prefix + "ffn_up.weight",
                    (hidden, vision_intermediate),
                )
                _expect_mmproj_shape(mmproj_gguf, prefix + "ffn_up.bias", (hidden,))
                _expect_mmproj_shape(
                    mmproj_gguf,
                    prefix + "ffn_down.weight",
                    (vision_intermediate, hidden),
                )
                _expect_mmproj_shape(
                    mmproj_gguf,
                    prefix + "ffn_down.bias",
                    (vision_intermediate,),
                )
            else:
                for stem in ("ln1", "ln2"):
                    _expect_mmproj_shape(mmproj_gguf, prefix + stem + ".weight", (hidden,))
                for stem in ("ffn_gate", "ffn_up"):
                    _expect_mmproj_shape(
                        mmproj_gguf,
                        prefix + stem + ".weight",
                        (vision_intermediate, hidden),
                    )
                    _expect_mmproj_shape(
                        mmproj_gguf,
                        prefix + stem + ".bias",
                        (vision_intermediate,),
                    )
                _expect_mmproj_shape(
                    mmproj_gguf,
                    prefix + "ffn_down.weight",
                    (hidden, vision_intermediate),
                )
                _expect_mmproj_shape(
                    mmproj_gguf, prefix + "ffn_up.bias", (vision_intermediate,)
                )
                _expect_mmproj_shape(mmproj_gguf, prefix + "ffn_down.bias", (hidden,))
        if qwen2:
            _expect_mmproj_shape(mmproj_gguf, "v.post_ln.bias", (hidden,))
        return

    if spec.projector_type == "gemma4v":
        expected_patch_shape = (hidden, 3, patch, patch)
        if patch_shape != expected_patch_shape:
            raise ValueError(
                "Gemma4 v.patch_embd.weight must have graph-compatible shape "
                f"{expected_patch_shape}, got {patch_shape}."
            )
        position_shape = mmproj_gguf.get_tensor_shape("v.position_embd.weight")
        if len(position_shape) != 3 or position_shape[0] != 2 or position_shape[2] != hidden:
            raise ValueError(
                "Gemma4 position embeddings must have shape [2, positions, hidden], "
                f"got {position_shape}."
            )
        _expect_mmproj_shape(mmproj_gguf, "mm.input_projection.weight", (projection, hidden))
        head_dim = hidden // heads
        shapes = {
            "ln1.weight": (hidden,),
            "ln2.weight": (hidden,),
            "attn_post_norm.weight": (hidden,),
            "ffn_post_norm.weight": (hidden,),
            "attn_q.weight": (hidden, hidden),
            "attn_k.weight": (hidden, hidden),
            "attn_v.weight": (hidden, hidden),
            "attn_out.weight": (hidden, hidden),
            "attn_q_norm.weight": (head_dim,),
            "attn_k_norm.weight": (head_dim,),
            "ffn_gate.weight": (intermediate, hidden),
            "ffn_up.weight": (intermediate, hidden),
            "ffn_down.weight": (hidden, intermediate),
        }
        shapes.update(dict.fromkeys(_CLIPPING_BOUND_SUFFIXES, ()))
        for layer in range(layers):
            for suffix, expected in shapes.items():
                if suffix in _CLIPPING_BOUND_SUFFIXES:
                    continue
                _expect_mmproj_shape(mmproj_gguf, f"v.blk.{layer}.{suffix}", expected)
            for stem in (
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_out",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ):
                for bound in ("input_min", "input_max", "output_min", "output_max"):
                    _expect_mmproj_shape(mmproj_gguf, f"v.blk.{layer}.{stem}.{bound}", (1,))
        return

    if spec.projector_type == "muse-glimmer":
        if (
            len(patch_shape) not in (4, 5)
            or patch_shape[0] != hidden
            or patch_shape[-2:] != (patch, patch)
        ):
            raise ValueError(
                "Muse Glimmer v.patch_embd.weight must be rank 4/5 with output "
                f"{hidden} and a {patch}x{patch} spatial kernel, got {patch_shape}."
            )
        merge = int(md["clip.vision.spatial_merge_size"])
        position_shape = mmproj_gguf.get_tensor_shape("v.position_embd.weight")
        if len(position_shape) != 2 or position_shape[1] != hidden:
            raise ValueError(
                "Muse Glimmer position embeddings must have shape [positions, hidden], "
                f"got {position_shape}."
            )
        projector_width = mmproj_gguf.get_tensor_shape("mm.0.weight")[0]
        _expect_mmproj_shape(
            mmproj_gguf, "mm.0.weight", (projector_width, hidden * merge * merge)
        )
        _expect_mmproj_shape(mmproj_gguf, "mm.1.weight", (projector_width, projector_width))
        _expect_mmproj_shape(mmproj_gguf, "mm.2.weight", (projection, projector_width))
        for name in ("v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias"):
            _expect_mmproj_shape(mmproj_gguf, name, (hidden,))
        for layer in range(layers):
            for stem in ("ln1", "ln2"):
                for kind in ("weight", "bias"):
                    _expect_mmproj_shape(
                        mmproj_gguf, f"v.blk.{layer}.{stem}.{kind}", (hidden,)
                    )
            for stem in ("attn_q", "attn_k", "attn_v", "attn_out"):
                _expect_mmproj_shape(
                    mmproj_gguf, f"v.blk.{layer}.{stem}.weight", (hidden, hidden)
                )
                _expect_mmproj_shape(mmproj_gguf, f"v.blk.{layer}.{stem}.bias", (hidden,))
            _expect_mmproj_shape(
                mmproj_gguf, f"v.blk.{layer}.ffn_up.weight", (intermediate, hidden)
            )
            _expect_mmproj_shape(mmproj_gguf, f"v.blk.{layer}.ffn_up.bias", (intermediate,))
            _expect_mmproj_shape(
                mmproj_gguf, f"v.blk.{layer}.ffn_down.weight", (hidden, intermediate)
            )
            _expect_mmproj_shape(mmproj_gguf, f"v.blk.{layer}.ffn_down.bias", (hidden,))
        return

    if spec.builder == "generic_projector":
        grid = int(md["clip.vision.image_size"]) // patch
        has_class_token = "v.class_embd" in mmproj_gguf.tensor_names
        _expect_mmproj_shape(mmproj_gguf, "v.patch_embd.weight", (hidden, 3, patch, patch))
        _expect_mmproj_shape(
            mmproj_gguf,
            "v.position_embd.weight",
            (grid * grid + int(has_class_token), hidden),
        )
        if has_class_token:
            _expect_mmproj_shape(mmproj_gguf, "v.class_embd", (hidden,))
            for name in ("v.pre_ln.weight", "v.pre_ln.bias"):
                _expect_mmproj_shape(mmproj_gguf, name, (hidden,))
        else:
            for name in ("v.patch_embd.bias", "v.post_ln.weight", "v.post_ln.bias"):
                _expect_mmproj_shape(mmproj_gguf, name, (hidden,))
        for layer in range(layers):
            prefix = f"v.blk.{layer}."
            for stem in ("ln1", "ln2"):
                for kind in ("weight", "bias"):
                    _expect_mmproj_shape(mmproj_gguf, prefix + stem + "." + kind, (hidden,))
            for stem in ("attn_q", "attn_k", "attn_v", "attn_out"):
                _expect_mmproj_shape(mmproj_gguf, prefix + stem + ".weight", (hidden, hidden))
                _expect_mmproj_shape(mmproj_gguf, prefix + stem + ".bias", (hidden,))
            _expect_mmproj_shape(
                mmproj_gguf, prefix + "ffn_down.weight", (intermediate, hidden)
            )
            _expect_mmproj_shape(mmproj_gguf, prefix + "ffn_down.bias", (intermediate,))
            _expect_mmproj_shape(mmproj_gguf, prefix + "ffn_up.weight", (hidden, intermediate))
            _expect_mmproj_shape(mmproj_gguf, prefix + "ffn_up.bias", (hidden,))
        vision = read_mmproj_generic_vision_config(mmproj_gguf)
        if vision is None:
            raise ValueError("Generic GGUF projector has no vision configuration.")
        _generic_projector_dimensions(mmproj_gguf, spec.projector_type, vision)


def _validate_supported_mmproj_metadata(mmproj_gguf: Any, spec: ProjectorSpec) -> None:
    if spec.sidecar_builder == "qwen_glm_projector":
        from mobius.integrations.gguf._qwen_glm_projector import (
            validate_qwen_glm_projector_metadata,
        )

        validate_qwen_glm_projector_metadata(mmproj_gguf, spec.projector_type)


def _preflight_mmproj_pair(
    text_gguf: Any,
    mmproj_gguf: Any,
    *,
    modalities: tuple[MMProjModality, ...],
) -> dict[MMProjModality, ProjectorSpec]:
    """Validate the pair and return supported specs before graph construction."""
    if mmproj_gguf.architecture != MMPROJ_ARCHITECTURE:
        raise ValueError(
            f"Expected a {MMPROJ_ARCHITECTURE!r} mmproj GGUF, got architecture "
            f"{mmproj_gguf.architecture!r}."
        )
    _validate_mmproj_container_type(mmproj_gguf)
    text_arch = _canonical_text_architecture(text_gguf.architecture)
    resolved: dict[MMProjModality, ProjectorSpec] = {}
    for modality in modalities:
        presence_key = f"clip.has_{modality.value.replace('.', '_')}_encoder"
        if not mmproj_gguf.metadata.get(presence_key):
            raise ValueError(
                f"mmproj GGUF has no {modality.value} encoder ({presence_key} is unset)."
            )
        projector_type = projector_type_for_modality(mmproj_gguf.metadata, modality)
        spec = get_projector_spec(projector_type)
        if modality not in spec.modalities:
            raise ValueError(
                f"Projector {projector_type!r} is not a {modality.value} projector."
            )
        if text_arch not in spec.target_architectures:
            raise ValueError(
                f"clip projector {projector_type!r} targets "
                f"{sorted(spec.target_architectures)}, not text architecture "
                f"{text_gguf.architecture!r}."
            )
        missing_metadata = [
            key for key in spec.required_metadata if key not in mmproj_gguf.metadata
        ]
        if missing_metadata:
            raise ValueError(
                f"{projector_type} mmproj is missing required metadata: {missing_metadata}"
            )
        _validate_supported_mmproj_metadata(mmproj_gguf, spec)
        _validate_mmproj_tensor_closure(mmproj_gguf, spec)
        if not spec.is_importable:
            blocked = ", ".join(
                name
                for name, verdict in spec.verdicts.items()
                if name != "runtime" and verdict is not Support.SUPPORTED
            )
            raise NotImplementedError(
                f"clip projector {projector_type!r} is known at llama.cpp "
                f"{LLAMA_CPP_MMPROJ_SHA} but is deferred/rejected "
                f"({blocked} unavailable): {spec.reason}"
            )
        resolved[modality] = spec
    return resolved


def _preflight_standalone_mmproj(
    mmproj_gguf: Any,
    *,
    projector_type: str,
    target_architecture: str,
) -> ProjectorSpec:
    """Validate one explicitly selected standalone sidecar graph route."""
    if mmproj_gguf.architecture != MMPROJ_ARCHITECTURE:
        raise ValueError(
            f"Expected a {MMPROJ_ARCHITECTURE!r} mmproj GGUF, got architecture "
            f"{mmproj_gguf.architecture!r}."
        )
    _validate_mmproj_container_type(mmproj_gguf)
    spec = get_projector_spec(projector_type)
    canonical_target = _canonical_text_architecture(target_architecture)
    if canonical_target not in spec.target_architectures:
        raise ValueError(
            f"clip projector {projector_type!r} targets "
            f"{sorted(spec.target_architectures)}, not text architecture "
            f"{target_architecture!r}."
        )

    selected_modalities = []
    for modality in spec.modalities:
        presence_key = f"clip.has_{modality.value.replace('.', '_')}_encoder"
        if not bool(mmproj_gguf.metadata.get(presence_key)):
            continue
        if projector_type_for_modality(mmproj_gguf.metadata, modality) == projector_type:
            selected_modalities.append(modality)
    if spec.primary_modality not in selected_modalities:
        raise ValueError(
            f"mmproj GGUF does not declare {projector_type!r} for its "
            f"{spec.primary_modality.value} encoder."
        )

    missing_metadata = [
        key for key in spec.required_metadata if key not in mmproj_gguf.metadata
    ]
    if missing_metadata:
        raise ValueError(
            f"{projector_type} mmproj is missing required metadata: {missing_metadata}"
        )
    _validate_supported_mmproj_metadata(mmproj_gguf, spec)
    _validate_mmproj_tensor_closure(mmproj_gguf, spec)
    if not spec.is_importable or spec.sidecar_builder is None:
        blocked = ", ".join(
            name
            for name, verdict in spec.verdicts.items()
            if name != "runtime" and verdict is not Support.SUPPORTED
        )
        raise NotImplementedError(
            f"clip projector {projector_type!r} has no standalone graph importer at "
            f"llama.cpp {LLAMA_CPP_MMPROJ_SHA} ({blocked or 'dispatch'} unavailable): "
            f"{spec.reason}"
        )
    return spec


def build_mmproj_from_gguf(
    mmproj_gguf_path: str | Path,
    *,
    projector_type: str,
    target_architecture: str,
    dtype: str | None = None,
    execution_provider: str = "default",
    _mmproj_gguf_model: Any | None = None,
) -> ModelPackage:
    """Build explicit standalone encoder/projector components from one sidecar.

    This entry point never synthesizes a text decoder or runtime assembly. The
    caller names both the serialized projector route and its paired text
    architecture so a vision, audio, merger, or speaker encoder cannot be
    mistaken for a generic VLM package.
    """
    from mobius.integrations.gguf._builder import _validate_gguf_model
    from mobius.integrations.gguf._reader import GGUFModel

    resolved_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
    mmproj_gguf = (
        _mmproj_gguf_model if _mmproj_gguf_model is not None else GGUFModel(resolved_path)
    )
    _validate_gguf_model(
        mmproj_gguf,
        source=str(mmproj_gguf_path),
        allow_mmproj_companion=True,
    )
    spec = _preflight_standalone_mmproj(
        mmproj_gguf,
        projector_type=projector_type,
        target_architecture=target_architecture,
    )
    if spec.runtime is not Support.SUPPORTED:
        warnings.warn(
            f"clip projector {projector_type!r} exports an exact ONNX component, "
            "but downstream runtime execution remains unvalidated.",
            RuntimeWarning,
            stacklevel=2,
        )
    builder_attribute = _MMPROJ_BUILDERS.get(spec.sidecar_builder or "")
    if builder_attribute is None:
        raise RuntimeError(
            f"Projector registry references unknown standalone builder "
            f"{spec.sidecar_builder!r}."
        )
    builder: Callable[..., ModelPackage] = globals()[builder_attribute]
    package = builder(
        resolved_path,
        projector_type=projector_type,
        target_architecture=target_architecture,
        dtype=dtype,
        execution_provider=execution_provider,
        _mmproj_gguf_model=mmproj_gguf,
    )
    expected_roles = {role.value for role in spec.model_roles}
    if set(package) != expected_roles:
        raise RuntimeError(
            f"{projector_type} standalone builder produced components "
            f"{sorted(package)}, expected {sorted(expected_roles)}."
        )
    if spec.runtime is not Support.SUPPORTED:
        logger.warning(
            "Built standalone %s graph component(s) for clip projector %r; "
            "downstream runtime orchestration is %s: %s",
            ", ".join(sorted(expected_roles)),
            projector_type,
            spec.runtime.value,
            spec.reason,
        )
    return package


def _token_id(text_gguf: Any, token: str) -> int | None:
    tokens = text_gguf.metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list):
        return None
    try:
        return tokens.index(token)
    except ValueError:
        return None


def _validate_projector_output_and_media_tokens(
    config: Any,
    text_gguf: Any,
    mmproj_gguf: Any,
    *,
    require_audio: bool = False,
) -> None:
    """Validate dimensions and tokenizer-owned media IDs before graph build."""
    projection = int(mmproj_gguf.metadata["clip.vision.projection_dim"])
    if projection != int(config.hidden_size):
        raise ValueError(
            f"mmproj projection_dim={projection} does not match text hidden_size="
            f"{config.hidden_size}."
        )
    if int(config.vocab_size) <= 0:
        raise ValueError(f"Text GGUF has invalid vocab_size={config.vocab_size}.")
    media_ids = {"image": config.image_token_id}
    if require_audio:
        audio_projection = int(mmproj_gguf.metadata["clip.audio.projection_dim"])
        if audio_projection != int(config.hidden_size):
            raise ValueError(
                f"audio mmproj projection_dim={audio_projection} does not match text "
                f"hidden_size={config.hidden_size}."
            )
        media_ids["audio"] = config.audio_token_id
    for modality, token_id in media_ids.items():
        if token_id is None:
            raise ValueError(
                f"The paired text GGUF does not identify its {modality} media token. "
                f"Embed tokenizer.ggml.tokens with '<|{modality}|>' or pass the "
                "architecture-specific explicit token id."
            )
        if not 0 <= int(token_id) < int(config.vocab_size):
            raise ValueError(
                f"{modality}_token_id={token_id} is outside text vocab_size="
                f"{config.vocab_size}."
            )


# Text-GGUF tensors whose names are not covered by the block ``gemma4`` text
# mapping; they route to their HF language-model names directly.
_PER_LAYER_TOP_HF_NAMES = {
    "per_layer_token_embd.weight": "language_model.embed_tokens_per_layer.weight",
    "per_layer_model_proj.weight": "language_model.per_layer_model_projection.weight",
    "per_layer_proj_norm.weight": "language_model.per_layer_projection_norm.weight",
}


def _text_gguf_name_to_hf_multimodal(gguf_name: str) -> str | None:
    """Map one text-GGUF tensor name to its HF multimodal (``language_model.*``) name.

    Returns ``None`` for tensors that have no multimodal counterpart.
    """
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    if gguf_name in _PER_LAYER_TOP_HF_NAMES:
        return _PER_LAYER_TOP_HF_NAMES[gguf_name]
    text_hf = map_gguf_to_hf_names(gguf_name, "gemma4")
    if text_hf is None:
        return None
    # gemma4 text mapping yields ``model.*`` / ``lm_head.*``; nest under the
    # multimodal ``language_model.`` namespace (HF Gemma4 stores the decoder
    # layers directly under language_model, so strip the ``model.`` prefix).
    if text_hf.startswith("model."):
        return "language_model." + text_hf[len("model.") :]
    return "language_model." + text_hf


def _text_gguf_to_hf_multimodal(text_gguf: Any) -> dict:
    """Load text-backbone GGUF tensors as HF multimodal (``language_model.*``).

    Reuses the text ``gemma4`` GGUF→HF name mapping, then rewrites names into
    the multimodal ``language_model.*`` namespace that
    ``Gemma4Model.preprocess_weights`` expects.
    """
    import torch

    state_dict: dict[str, torch.Tensor] = {}
    for gguf_name, array in text_gguf.tensor_items():
        hf_name = _text_gguf_name_to_hf_multimodal(gguf_name)
        if hf_name is None:
            continue

        values = np.array(array).astype(np.float32)
        # layer_scalar is an nn.Parameter (no ``.weight`` module suffix, shape [1]).
        if hf_name.endswith(".layer_scalar.weight"):
            hf_name = hf_name[: -len(".weight")]
            values = values.reshape(-1)[:1]
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict


# HF projection-weight name suffixes (under ``language_model.layers.N.``) that
# become MatMulNBits QuantizedLinear layers in the quantized text decoder.
_QUANTIZED_LINEAR_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)

# HF token-embedding weight names that become GatherBlockQuantized tables.
_QUANTIZED_EMBEDDING_NAMES = (
    "language_model.embed_tokens.weight",
    "language_model.embed_tokens_per_layer.weight",
)


def _text_gguf_to_hf_multimodal_quantized(
    text_gguf: Any,
    config: Any,
    *,
    bits: int,
    block_size: int,
    symmetric: bool,
) -> dict:
    """Load text GGUF tensors, quantizing the decoder and compatible embeddings.

    Mirrors :func:`_text_gguf_to_hf_multimodal` but keeps the text decoder
    projections in MatMulNBits form (``.weight`` uint8 + ``.scales`` [+
    ``.zero_points``]) and compatible token-embedding tables in
    GatherBlockQuantized form (``.qweight`` + ``.scales`` [+
    ``.zero_points``]). Incompatible token embeddings, norms, and float
    per-layer projections stay dequantized. Vision/audio weights are loaded
    separately and always stay float.

    Names are the HF multimodal ``language_model.*`` names that
    :meth:`Gemma4Model.preprocess_weights` expects.
    """
    import torch

    from mobius.integrations.gguf._builder import repack_gguf_weight_to_target
    from mobius.integrations.gguf._quant_registry import float_storage_type_ids
    from mobius.integrations.gguf._spec import TensorRole

    quantize_embeddings = bool(getattr(config.quantization, "quantize_embeddings", False))
    quantize_lm_head = bool(getattr(config.quantization, "quantize_lm_head", False))
    tie_word_embeddings = bool(config.tie_word_embeddings)
    float_type_ids = float_storage_type_ids()

    def _emit_linear(state_dict: dict, stem: str, repacked) -> None:
        state_dict[f"{stem}.weight"] = torch.from_numpy(repacked.weight)
        state_dict[f"{stem}.scales"] = torch.from_numpy(repacked.scales)
        if repacked.zero_points is not None:
            state_dict[f"{stem}.zero_points"] = torch.from_numpy(repacked.zero_points)

    def _emit_embedding(state_dict: dict, stem: str, repacked) -> None:
        weight = repacked.weight
        state_dict[f"{stem}.qweight"] = torch.from_numpy(weight.reshape(weight.shape[0], -1))
        state_dict[f"{stem}.scales"] = torch.from_numpy(repacked.scales)
        if repacked.zero_points is not None:
            state_dict[f"{stem}.zero_points"] = torch.from_numpy(repacked.zero_points)

    state_dict: dict[str, torch.Tensor] = {}
    for gguf_name, raw, qtype, np_shape in text_gguf.tensor_items_raw():
        hf_name = _text_gguf_name_to_hf_multimodal(gguf_name)
        if hf_name is None:
            continue

        is_quant_linear = hf_name.endswith(_QUANTIZED_LINEAR_SUFFIXES) or (
            quantize_lm_head and hf_name == "language_model.lm_head.weight"
        )
        is_quant_embedding = quantize_embeddings and hf_name in _QUANTIZED_EMBEDDING_NAMES

        if (is_quant_linear or is_quant_embedding) and len(np_shape) == 2:
            qtype_id = getattr(qtype, "value", qtype)
            if qtype_id in float_type_ids:
                raise ValueError(
                    "Quantization-preserving Gemma4 GGUF import would quantize float "
                    f"projection {gguf_name} ({getattr(qtype, 'name', qtype)}) to "
                    f"the graph's {bits}-bit/block-{block_size} MatMulNBits contract. "
                    "Use keep_quantized=False (API) for explicit float import."
                )
            repacked = repack_gguf_weight_to_target(
                text_gguf,
                raw,
                qtype,
                np_shape,
                target_bits=bits,
                target_block_size=block_size,
                target_symmetric=symmetric,
                tensor_name=hf_name,
                tensor_role=(
                    TensorRole.EMBEDDING
                    if is_quant_embedding
                    else TensorRole.AFFINE_PROJECTION
                ),
            )
            stem = hf_name[: -len(".weight")]
            if is_quant_embedding:
                _emit_embedding(state_dict, stem, repacked)
                # A tied quantized LM head shares the token-embedding table but,
                # because the decoder is a separate ONNX graph, needs its own
                # MatMulNBits copy (same repacked bytes, linear layout).
                if (
                    hf_name == "language_model.embed_tokens.weight"
                    and quantize_lm_head
                    and tie_word_embeddings
                ):
                    _emit_linear(state_dict, "language_model.lm_head", repacked)
            else:
                _emit_linear(state_dict, stem, repacked)
            continue

        # Everything else (norms, float per-layer projections) stays float.
        qtype_id = getattr(qtype, "value", qtype)
        if qtype_id not in float_type_ids:
            raise ValueError(
                "Quantization-preserving Gemma4 GGUF import cannot retain packed tensor "
                f"{gguf_name} ({getattr(qtype, 'name', qtype)}) because its graph "
                "target is float. Use keep_quantized=False (API) for explicit float import."
            )
        values = np.array(text_gguf.dequantize_raw_tensor(raw, qtype, np_shape)).astype(
            np.float32
        )
        if hf_name.endswith(".layer_scalar.weight"):
            hf_name = hf_name[: -len(".weight")]
            values = values.reshape(-1)[:1]
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict


def _mmproj_vision_to_hf(mmproj_gguf: Any) -> dict:
    """Load mmproj vision tensors as HF names (``vision_tower.*``/``embed_vision.*``)."""
    import torch

    from mobius.integrations.gguf._mmproj_mapping import map_mmproj_vision_to_hf

    state_dict: dict[str, torch.Tensor] = {}
    for name in mmproj_gguf.tensor_names:
        if not (name.startswith("v.") or name == "mm.input_projection.weight"):
            continue
        hf_name = map_mmproj_vision_to_hf(name)
        if hf_name is None:
            continue
        values = np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
        if name == "v.patch_embd.weight":
            # Conv patch embed [out, in_ch, kh, kw] → Linear [out, in_ch*kh*kw].
            # The flattening order (in_ch, kh, kw) row-major matches the
            # pre-patchified pixel_values layout consumed by the encoder.
            values = values.reshape(values.shape[0], -1)
        elif name.endswith(_CLIPPING_BOUND_SUFFIXES):
            values = values.reshape(())
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict


def _mmproj_gemma3_vision_to_hf(mmproj_gguf: Any) -> dict:
    """Load the exact Gemma3 vision/projector closure under HF names."""
    import torch

    from mobius.integrations.gguf._mmproj_mapping import (
        map_mmproj_gemma3_vision_to_hf,
    )

    state_dict: dict[str, torch.Tensor] = {}
    for name in mmproj_gguf.tensor_names:
        hf_name = map_mmproj_gemma3_vision_to_hf(name)
        if hf_name is None:
            continue
        values = np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
        if name == "mm.soft_emb_norm.weight":
            # llama.cpp bakes OffsetRMSNorm's +1 into GGUF norm scales.
            values = values - 1.0
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict


def _mmproj_qwen_vision_to_hf(mmproj_gguf: Any, projector_type: str) -> dict:
    """Load Qwen tower weights, fusing serialized QKV and temporal patch halves."""
    import torch

    from mobius.integrations.gguf._mmproj_mapping import (
        map_mmproj_qwen_vision_to_hf,
    )

    state_dict: dict[str, torch.Tensor] = {}
    qwen2 = projector_type == "qwen2vl_merger"
    num_layers = int(mmproj_gguf.metadata["clip.vision.block_count"])

    patch_halves = [
        np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
        for name in ("v.patch_embd.weight", "v.patch_embd.weight.1")
    ]
    state_dict["visual.patch_embed.proj.weight"] = torch.from_numpy(
        np.stack(patch_halves, axis=2).copy()
    )

    for layer in range(num_layers):
        prefix = f"v.blk.{layer}."
        for kind in ("weight", "bias"):
            parts = [
                np.array(mmproj_gguf.get_tensor(prefix + f"attn_{name}.{kind}")).astype(
                    np.float32
                )
                for name in ("q", "k", "v")
            ]
            state_dict[f"visual.blocks.{layer}.attn.qkv.{kind}"] = torch.from_numpy(
                np.concatenate(parts, axis=0).copy()
            )

    fused_sources = {
        "v.patch_embd.weight",
        "v.patch_embd.weight.1",
        *(
            f"v.blk.{layer}.attn_{name}.{kind}"
            for layer in range(num_layers)
            for name in ("q", "k", "v")
            for kind in ("weight", "bias")
        ),
    }
    for name in mmproj_gguf.tensor_names:
        if name in fused_sources:
            continue
        hf_name = map_mmproj_qwen_vision_to_hf(name)
        if hf_name is None:
            continue
        values = np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
        if qwen2:
            # llama.cpp's Qwen2-VL clip loader names the projection producing
            # hidden width "ffn_up" and the expansion "ffn_down"; route by
            # shape/bias semantics rather than transposing either matrix.
            if ".ffn_up." in name:
                hf_name = hf_name.replace(".mlp.up_proj.", ".mlp.down_proj.")
            elif ".ffn_down." in name:
                hf_name = hf_name.replace(".mlp.down_proj.", ".mlp.up_proj.")
        state_dict[hf_name] = torch.from_numpy(values.copy())
    return state_dict


def _mmproj_generic_to_onnx(mmproj_gguf: Any, projector_type: str) -> dict:
    """Load a generic CLIP/SigLIP tower and projector under graph-local names."""
    import torch

    from mobius.integrations.gguf._mmproj_mapping import (
        map_generic_projector_to_onnx,
        map_generic_vision_to_onnx,
    )

    state_dict: dict[str, torch.Tensor] = {}
    compatibility_only = {"resampler.pos_embed_k"}
    for name in mmproj_gguf.tensor_names:
        if name in compatibility_only:
            continue
        mapped = (
            map_generic_vision_to_onnx(name)
            if name.startswith("v.")
            else map_generic_projector_to_onnx(name, projector_type)
        )
        if mapped is None:
            continue
        values = np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
        if (projector_type == "ldp" and ".depthwise.weight" in mapped) or (
            projector_type == "ldpv2" and mapped == "projector.peg_0.weight"
        ):
            values = values[:, None, :, :]
        elif projector_type == "ldp" and ".pointwise.weight" in mapped:
            values = values[:, :, None, None]
        state_dict[f"vision_encoder.{mapped}"] = torch.from_numpy(values.copy())
    return state_dict


def _generic_projector_dimensions(
    mmproj_gguf: Any,
    projector_type: str,
    vision: VisionConfig,
) -> tuple[int, int | None, int | None]:
    """Validate exact projector shapes and return output/intermediate/query sizes."""

    def shape(name: str) -> tuple[int, ...]:
        return tuple(int(dim) for dim in mmproj_gguf.get_tensor_shape(name))

    if vision.hidden_size is None or vision.image_size is None or vision.patch_size is None:
        raise ValueError("Generic GGUF projector vision dimensions must be defined.")
    vision_width = int(vision.hidden_size)
    image_size = int(vision.image_size)
    patch_size = int(vision.patch_size)
    grid = image_size // patch_size
    if image_size % patch_size:
        raise ValueError("Generic GGUF projector image size must divide by patch size.")

    if projector_type == "mlp":
        first = shape("mm.0.weight")
        if any(name.startswith("mm.3.") for name in mmproj_gguf.tensor_names):
            raise ValueError(
                "MLP sidecar contains mm.3 tensors and therefore selects llama.cpp's "
                "distinct MLP_NORM topology, which this generic route does not implement."
            )
        has_second_weight = "mm.2.weight" in mmproj_gguf.tensor_names
        has_second_bias = "mm.2.bias" in mmproj_gguf.tensor_names
        if has_second_weight != has_second_bias:
            raise ValueError("MLP projector mm.2 weight and bias must be present together.")
        second = shape("mm.2.weight") if has_second_weight else None
        if len(first) != 2 or first[1] != vision_width:
            raise ValueError(
                f"MLP projector shapes {first}/{second} do not form "
                f"{vision_width}->hidden->hidden."
            )
        width = first[0]
        expected = {
            "mm.0.bias": (width,),
            **(
                {"mm.2.weight": (width, width), "mm.2.bias": (width,)}
                if second is not None
                else {}
            ),
        }
        for name, expected_shape in expected.items():
            _expect_mmproj_shape(mmproj_gguf, name, expected_shape)
        return first[0], None, None
    if projector_type == "ldp":
        if grid != 24:
            raise ValueError(f"LDP requires a 24x24 patch grid, got {grid}x{grid}.")
        first = shape("mm.model.mlp.1.weight")
        second = shape("mm.model.mlp.3.weight")
        if len(first) != 2 or first[1] != vision_width or second != (first[0], first[0]):
            raise ValueError(f"LDP MLP shapes {first}/{second} are inconsistent.")
        width = first[0]
        squeeze = width // 4
        for name in ("mm.model.mlp.1.bias", "mm.model.mlp.3.bias"):
            _expect_mmproj_shape(mmproj_gguf, name, (width,))
        for block in (1, 2):
            prefix = f"mm.model.mb_block.{block}.block."
            expected = {
                "0.0.weight": (width, 3, 3),
                "0.1.weight": (width,),
                "0.1.bias": (width,),
                "1.fc1.weight": (squeeze, width),
                "1.fc1.bias": (squeeze,),
                "1.fc2.weight": (width, squeeze),
                "1.fc2.bias": (width,),
                "2.0.weight": (width, width),
                "2.1.weight": (width,),
                "2.1.bias": (width,),
            }
            for suffix, expected_shape in expected.items():
                _expect_mmproj_shape(mmproj_gguf, prefix + suffix, expected_shape)
        return first[0], None, None
    if projector_type == "ldpv2":
        if grid != 24:
            raise ValueError(f"LDPv2 requires a 24x24 patch grid, got {grid}x{grid}.")
        first = shape("mm.model.mlp.0.weight")
        second = shape("mm.model.mlp.2.weight")
        peg = shape("mm.model.peg.0.weight")
        if (
            len(first) != 2
            or first[1] != vision_width
            or second != (first[0], first[0])
            or peg != (first[0], 3, 3)
        ):
            raise ValueError(
                f"LDPv2 projector shapes {first}/{second}/{peg} are inconsistent."
            )
        width = first[0]
        for name in ("mm.model.mlp.0.bias", "mm.model.mlp.2.bias", "mm.model.peg.0.bias"):
            _expect_mmproj_shape(mmproj_gguf, name, (width,))
        return first[0], None, None
    if projector_type == "adapter":
        conv = shape("adapter.conv.weight")
        up = shape("adapter.linear.dense_h_to_4h.weight")
        down = shape("adapter.linear.dense_4h_to_h.weight")
        if (
            len(conv) != 4
            or len(up) != 2
            or conv[1:] != (vision_width, 2, 2)
            or up[1] != conv[0]
            or down != (conv[0], up[0])
        ):
            raise ValueError(f"Adapter projector shapes {conv}/{up}/{down} are inconsistent.")
        width = conv[0]
        intermediate = up[0]
        expected = {
            "adapter.boi": (width,),
            "adapter.eoi": (width,),
            "adapter.conv.bias": (width,),
            "adapter.linear.linear.weight": (width, width),
            "adapter.linear.norm1.weight": (width,),
            "adapter.linear.norm1.bias": (width,),
            "adapter.linear.gate.weight": (intermediate, width),
        }
        for name, expected_shape in expected.items():
            _expect_mmproj_shape(mmproj_gguf, name, expected_shape)
        return conv[0], up[0], None
    if projector_type == "resampler":
        query = shape("resampler.query")
        query_position = shape("resampler.pos_embed")
        kv = shape("resampler.kv.weight")
        proj = shape("resampler.proj.weight")
        if (
            len(query) != 2
            or query_position != query
            or kv != (query[1], vision_width)
            or proj
            != (
                query[1],
                query[1],
            )
        ):
            raise ValueError(
                f"Resampler projector shapes {query}/{kv}/{proj} are inconsistent."
            )
        if grid * grid != shape("v.position_embd.weight")[0]:
            raise ValueError("Resampler vision position rows do not match its patch grid.")
        width = query[1]
        expected = {
            "resampler.attn.q.weight": (width, width),
            "resampler.attn.k.weight": (width, width),
            "resampler.attn.v.weight": (width, width),
            "resampler.attn.out.weight": (width, width),
            "resampler.attn.q.bias": (width,),
            "resampler.attn.k.bias": (width,),
            "resampler.attn.v.bias": (width,),
            "resampler.attn.out.bias": (width,),
            "resampler.ln_q.weight": (width,),
            "resampler.ln_q.bias": (width,),
            "resampler.ln_kv.weight": (width,),
            "resampler.ln_kv.bias": (width,),
            "resampler.ln_post.weight": (width,),
            "resampler.ln_post.bias": (width,),
        }
        if "resampler.pos_embed_k" in mmproj_gguf.tensor_names:
            expected["resampler.pos_embed_k"] = (grid * grid, width)
        for name, expected_shape in expected.items():
            _expect_mmproj_shape(mmproj_gguf, name, expected_shape)
        return query[1], None, query[0]
    raise ValueError(f"Unknown generic GGUF projector type {projector_type!r}")


def build_generic_projector_vlm_from_gguf(
    text_gguf_path: str | Path,
    mmproj_gguf_path: str | Path,
    *,
    dtype: str | None = None,
    execution_provider: str = "default",
    image_token_id: int | None = None,
    keep_quantized: bool = True,
    _text_gguf_model: Any | None = None,
    _mmproj_gguf_model: Any | None = None,
) -> ModelPackage:
    """Build the generic CLIP/SigLIP projector cohort as a three-model package."""
    import dataclasses

    import torch

    from mobius._builder import build_from_module, resolve_dtype
    from mobius._configs import QuantizationConfig
    from mobius._registry import registry
    from mobius.integrations.gguf._arch_registry import get_arch_spec
    from mobius.integrations.gguf._builder import (
        _can_quantize_embedding,
        _can_quantize_lm_head,
        _detect_quant_params,
        _has_quantized_weights,
        _load_dequantized_state_dict,
        _load_quantized_state_dict,
        _normalize_gguf_weights,
        _reject_unsupported_quantization_preservation,
        _replace_native_block_linears,
        _validate_gguf_model,
    )
    from mobius.integrations.gguf._config_mapping import gguf_to_config
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._tensor_processors import process_tensors
    from mobius.models.gguf_projector import GenericGGUFProjectorModel
    from mobius.tasks import GGUFProjectorVisionLanguageTask

    resolved_text_path = _resolve_local_path(text_gguf_path)
    text_gguf = (
        _text_gguf_model
        if _text_gguf_model is not None
        else _open_text_gguf(resolved_text_path)
    )
    _validate_gguf_model(text_gguf, source=str(text_gguf_path))

    resolved_mmproj_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
    mmproj_gguf = (
        _mmproj_gguf_model
        if _mmproj_gguf_model is not None
        else GGUFModel(resolved_mmproj_path)
    )
    _validate_gguf_model(
        mmproj_gguf,
        source=str(mmproj_gguf_path),
        allow_mmproj_companion=True,
    )
    specs = _preflight_mmproj_pair(
        text_gguf,
        mmproj_gguf,
        modalities=(MMProjModality.VISION,),
    )
    projector_type = specs[MMProjModality.VISION].projector_type
    if projector_type not in {"mlp", "ldp", "ldpv2", "adapter", "resampler"}:
        raise ValueError(f"Unsupported generic GGUF projector type {projector_type!r}.")

    arch_spec = get_arch_spec(text_gguf.architecture)
    if arch_spec.vlm_builder != "generic_projector":
        raise ValueError(
            f"Text architecture {text_gguf.architecture!r} does not declare the "
            "generic_projector VLM builder."
        )
    config = gguf_to_config(text_gguf)
    vision = read_mmproj_generic_vision_config(mmproj_gguf)
    if vision is None:
        raise ValueError("Generic projector sidecar has no vision encoder.")
    output_width, intermediate_width, num_queries = _generic_projector_dimensions(
        mmproj_gguf,
        projector_type,
        vision,
    )
    if output_width != int(config.hidden_size):
        raise ValueError(
            f"{projector_type} projector output width {output_width} does not match "
            f"text hidden size {config.hidden_size}."
        )

    image_start_token_id: int | None = None
    image_end_token_id: int | None = None
    resolved_image_token_id = image_token_id
    if projector_type == "resampler":
        image_start_token_id = _token_id(text_gguf, "<image>")
        image_end_token_id = _token_id(text_gguf, "</image>")
        if image_start_token_id is None or image_end_token_id is None:
            raise ValueError(
                "The paired MiniCPM text GGUF must contain <image> and </image> "
                "boundary tokens."
            )
        if resolved_image_token_id is None:
            unknown_id = text_gguf.metadata.get("tokenizer.ggml.unknown_token_id")
            resolved_image_token_id = (
                int(unknown_id) if unknown_id is not None else _token_id(text_gguf, "<unk>")
            )
    else:
        if resolved_image_token_id is None:
            resolved_image_token_id = config.image_token_id
        if resolved_image_token_id is None:
            for token in ("<image>", "<|image|>", "<|image_pad|>", "<|begin_of_image|>"):
                resolved_image_token_id = _token_id(text_gguf, token)
                if resolved_image_token_id is not None:
                    break
    if resolved_image_token_id is None:
        raise ValueError(
            "The paired text GGUF has no recognized image placeholder token; "
            "pass image_token_id explicitly."
        )
    config = dataclasses.replace(
        config,
        vision=vision,
        image_token_id=resolved_image_token_id,
    )
    if dtype is not None:
        resolved_dtype = resolve_dtype(dtype)
        if resolved_dtype is not None:
            config = dataclasses.replace(config, dtype=resolved_dtype)
    config._gguf_arch = text_gguf.architecture  # type: ignore[attr-defined]

    preserve_quantization = keep_quantized and _has_quantized_weights(
        text_gguf,
        text_gguf.architecture,
    )
    _reject_unsupported_quantization_preservation(
        text_gguf,
        text_gguf.architecture,
        preserve_quantization=preserve_quantization,
    )
    if preserve_quantization:
        bits, block_size, symmetric = _detect_quant_params(
            text_gguf,
            text_gguf.architecture,
        )
        quantize_embeddings = _can_quantize_embedding(
            text_gguf,
            text_gguf.architecture,
            bits=bits,
            block_size=block_size,
        )
        quantize_lm_head = (
            quantize_embeddings
            if config.tie_word_embeddings
            else _can_quantize_lm_head(text_gguf, text_gguf.architecture)
        )
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=block_size,
                quant_method="gguf",
                sym=symmetric,
                quantize_embeddings=quantize_embeddings,
                quantize_lm_head=quantize_lm_head,
                tie_word_embeddings=quantize_lm_head and config.tie_word_embeddings,
            ),
        )
        config._gguf_arch = text_gguf.architecture  # type: ignore[attr-defined]

    model_type = arch_spec.module_type or arch_spec.model_type
    if model_type is None:
        raise RuntimeError(f"{text_gguf.architecture!r} has no registered model type.")
    module_class: Any = registry.get(model_type)
    causal_lm = module_class(config)
    if preserve_quantization:
        _replace_native_block_linears(
            causal_lm,
            text_gguf,
            text_gguf.architecture,
        )
    module = GenericGGUFProjectorModel(
        config,
        causal_lm,
        projector_type=projector_type,
        projector_hidden_size=output_width,
        projector_intermediate_size=intermediate_width,
        num_queries=num_queries,
        mlp_has_second_layer="mm.2.weight" in mmproj_gguf.tensor_names,
        image_token_id=resolved_image_token_id,
        image_start_token_id=image_start_token_id,
        image_end_token_id=image_end_token_id,
    )
    pkg = build_from_module(
        module,
        config,
        task=GGUFProjectorVisionLanguageTask(),
        execution_provider=execution_provider,
    )

    text_state = (
        _load_quantized_state_dict(
            text_gguf,
            text_gguf.architecture,
            causal_lm,
            config,
        )
        if preserve_quantization
        else _load_dequantized_state_dict(text_gguf, text_gguf.architecture)
    )
    float_state = {
        key: value
        for key, value in text_state.items()
        if not key.endswith((".scales", ".zero_points", ".qweight"))
        and value.dtype != torch.uint8
    }
    retained_state = {
        key: value for key, value in text_state.items() if key not in float_state
    }
    float_state = _normalize_gguf_weights(
        process_tensors(float_state, config),
        text_gguf.architecture,
        config,
    )
    text_state = causal_lm.preprocess_weights({**float_state, **retained_state})

    # The embedding graph owns an independent copy of the token table.
    package_state = {f"decoder.{key}": value for key, value in text_state.items()}
    for suffix in ("weight", "qweight", "scales", "zero_points"):
        source = f"model.embed_tokens.{suffix}"
        if source in text_state:
            package_state[f"embedding.embed_tokens.{suffix}"] = text_state[source]
    state_dict = {
        **package_state,
        **_mmproj_generic_to_onnx(mmproj_gguf, projector_type),
    }
    pkg.apply_weights(state_dict)

    from mobius.integrations.gguf._tokenizer import inspect_gguf_tokenizer

    pkg.gguf_source_path = str(Path(resolved_text_path).resolve())  # type: ignore[attr-defined]
    pkg.gguf_tokenizer_verdict = inspect_gguf_tokenizer(  # type: ignore[attr-defined]
        text_gguf.metadata, source=str(resolved_text_path)
    )
    return pkg


def build_qwen_vlm_from_gguf(
    text_gguf_path: str | Path,
    mmproj_gguf_path: str | Path,
    *,
    dtype: str | None = None,
    execution_provider: str = "default",
    image_token_id: int | None = None,
    keep_quantized: bool = True,
    _text_gguf_model: Any | None = None,
    _mmproj_gguf_model: Any | None = None,
) -> ModelPackage:
    """Build Qwen2/Qwen2.5-VL decoder, vision, and multimedia embedding graphs."""
    import dataclasses

    import torch

    from mobius._builder import build_from_module, resolve_dtype
    from mobius._configs import QuantizationConfig
    from mobius.integrations.gguf._builder import (
        _can_quantize_embedding,
        _can_quantize_lm_head,
        _detect_quant_params,
        _has_quantized_weights,
        _load_dequantized_state_dict,
        _load_quantized_state_dict,
        _normalize_gguf_weights,
        _reject_unsupported_quantization_preservation,
        _replace_native_block_linears,
        _validate_gguf_model,
    )
    from mobius.integrations.gguf._config_mapping import gguf_to_config
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._tensor_processors import process_tensors
    from mobius.models.qwen_vl import Qwen2VLCausalLMModel, Qwen25VLCausalLMModel
    from mobius.tasks._vision_language_3model import Qwen2VLMultimediaTask

    resolved_text_path = _resolve_local_path(text_gguf_path)
    text_gguf = (
        _text_gguf_model
        if _text_gguf_model is not None
        else _open_text_gguf(resolved_text_path)
    )
    if text_gguf.architecture != "qwen2vl":
        raise ValueError(
            "Qwen VL package construction requires a qwen2vl text GGUF; "
            f"got {text_gguf.architecture!r}"
        )
    _validate_gguf_model(text_gguf, source=str(text_gguf_path))

    resolved_mmproj_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
    mmproj_gguf = (
        _mmproj_gguf_model
        if _mmproj_gguf_model is not None
        else GGUFModel(resolved_mmproj_path)
    )
    _validate_gguf_model(
        mmproj_gguf,
        source=str(mmproj_gguf_path),
        allow_mmproj_companion=True,
    )
    specs = _preflight_mmproj_pair(text_gguf, mmproj_gguf, modalities=(MMProjModality.VISION,))
    projector_type = specs[MMProjModality.VISION].projector_type

    config = gguf_to_config(text_gguf)
    vision = read_mmproj_qwen_vision_config(mmproj_gguf, projector_type)
    if vision is None:
        raise ValueError(
            "mmproj GGUF has no vision encoder (clip.has_vision_encoder is unset)."
        )
    resolved_image_token_id = (
        image_token_id
        if image_token_id is not None
        else config.image_token_id
        if config.image_token_id is not None
        else _token_id(text_gguf, "<|image_pad|>")
    )
    config = dataclasses.replace(
        config,
        vision=vision,
        image_token_id=resolved_image_token_id,
        video_token_id=(
            config.video_token_id
            if config.video_token_id is not None
            else _token_id(text_gguf, "<|video_pad|>")
        ),
        vision_start_token_id=(
            config.vision_start_token_id
            if config.vision_start_token_id is not None
            else _token_id(text_gguf, "<|vision_start|>")
        ),
        vision_end_token_id=(
            config.vision_end_token_id
            if config.vision_end_token_id is not None
            else _token_id(text_gguf, "<|vision_end|>")
        ),
        spatial_merge_size=vision.spatial_merge_size,
        temporal_patch_size=vision.temporal_patch_size,
        fullatt_block_indexes=vision.fullatt_block_indexes,
        window_size=vision.window_size or _QWEN_VISION_WINDOW_SIZE,
    )
    if dtype is not None:
        resolved_dtype = resolve_dtype(dtype)
        if resolved_dtype is not None:
            config = dataclasses.replace(config, dtype=resolved_dtype)
    _validate_projector_output_and_media_tokens(config, text_gguf, mmproj_gguf)
    if config.video_token_id is None:
        raise ValueError("The paired Qwen text GGUF has no <|video_pad|> tokenizer token.")
    if config.vision_start_token_id is None or config.vision_end_token_id is None:
        raise ValueError("The paired Qwen text GGUF lacks vision boundary tokenizer tokens.")

    preserve_quantization = keep_quantized and _has_quantized_weights(text_gguf, "qwen2vl")
    _reject_unsupported_quantization_preservation(
        text_gguf,
        "qwen2vl",
        preserve_quantization=preserve_quantization,
    )
    if preserve_quantization:
        bits, block_size, symmetric = _detect_quant_params(text_gguf, "qwen2vl")
        quantize_embeddings = _can_quantize_embedding(
            text_gguf,
            "qwen2vl",
            bits=bits,
            block_size=block_size,
        )
        quantize_lm_head = (
            quantize_embeddings
            if config.tie_word_embeddings
            else _can_quantize_lm_head(text_gguf, "qwen2vl")
        )
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=block_size,
                quant_method="gguf",
                sym=symmetric,
                quantize_embeddings=quantize_embeddings,
                quantize_lm_head=quantize_lm_head,
                tie_word_embeddings=quantize_lm_head and config.tie_word_embeddings,
            ),
        )

    module_class = (
        Qwen2VLCausalLMModel if projector_type == "qwen2vl_merger" else Qwen25VLCausalLMModel
    )
    module = module_class(config)
    if preserve_quantization:
        _replace_native_block_linears(module.decoder, text_gguf, "qwen2vl")
    pkg = build_from_module(
        module,
        config,
        task=Qwen2VLMultimediaTask(),
        execution_provider=execution_provider,
    )

    text_state = (
        _load_quantized_state_dict(text_gguf, "qwen2vl", module.decoder, config)
        if preserve_quantization
        else _load_dequantized_state_dict(text_gguf, "qwen2vl")
    )
    float_state = {
        key: value
        for key, value in text_state.items()
        if not key.endswith((".scales", ".zero_points", ".qweight"))
        and value.dtype != torch.uint8
    }
    retained_state = {
        key: value for key, value in text_state.items() if key not in float_state
    }
    config._gguf_arch = text_gguf.architecture
    float_state = _normalize_gguf_weights(
        process_tensors(float_state, config), "qwen2vl", config
    )
    state_dict = {**float_state, **retained_state}
    state_dict.update(_mmproj_qwen_vision_to_hf(mmproj_gguf, projector_type))
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)

    from mobius.integrations.gguf._tokenizer import inspect_gguf_tokenizer

    pkg.gguf_source_path = str(Path(resolved_text_path).resolve())
    pkg.gguf_tokenizer_verdict = inspect_gguf_tokenizer(
        text_gguf.metadata, source=str(resolved_text_path)
    )
    return pkg


def _gemma3_multimodal_name(hf_name: str) -> str:
    """Nest Gemma3 text-only HF names under the composite HF namespace."""
    if hf_name.startswith("model."):
        return "language_model.model." + hf_name[len("model.") :]
    if hf_name.startswith("lm_head."):
        return "language_model." + hf_name
    return hf_name


def build_gemma3_vlm_from_gguf(
    text_gguf_path: str | Path,
    mmproj_gguf_path: str | Path,
    *,
    dtype: str | None = None,
    execution_provider: str = "default",
    image_token_id: int | None = None,
    keep_quantized: bool = True,
    _text_gguf_model: Any | None = None,
    _mmproj_gguf_model: Any | None = None,
) -> ModelPackage:
    """Build Gemma3 decoder, vision encoder, and embedding graphs from GGUFs.

    This is graph-import support only. It does not assert that a downstream
    multimodal runtime can execute the resulting package.
    """
    import dataclasses

    import torch

    from mobius._builder import build_from_module, resolve_dtype
    from mobius._configs import QuantizationConfig
    from mobius.integrations.gguf._builder import (
        _detect_quant_params,
        _has_quantized_weights,
        _load_dequantized_state_dict,
        _load_quantized_state_dict,
        _normalize_gguf_weights,
        _reject_unsupported_quantization_preservation,
        _replace_native_block_linears,
        _validate_gguf_model,
    )
    from mobius.integrations.gguf._config_mapping import gguf_to_config
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._tensor_processors import process_tensors
    from mobius.models.gemma3 import Gemma3MultiModalModel
    from mobius.tasks import Gemma3VisionLanguageTask

    resolved_text_path = _resolve_local_path(text_gguf_path)
    text_gguf = (
        _text_gguf_model
        if _text_gguf_model is not None
        else _open_text_gguf(resolved_text_path)
    )
    if text_gguf.architecture != "gemma3":
        raise ValueError(
            "Gemma3 VLM package construction requires a gemma3 text GGUF; "
            f"got {text_gguf.architecture!r}"
        )
    _validate_gguf_model(text_gguf, source=str(text_gguf_path))

    resolved_mmproj_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
    mmproj_gguf = (
        _mmproj_gguf_model
        if _mmproj_gguf_model is not None
        else GGUFModel(resolved_mmproj_path)
    )
    _validate_gguf_model(
        mmproj_gguf,
        source=str(mmproj_gguf_path),
        allow_mmproj_companion=True,
    )
    _preflight_mmproj_pair(text_gguf, mmproj_gguf, modalities=(MMProjModality.VISION,))

    config = gguf_to_config(text_gguf)
    vision_config = read_mmproj_gemma3_vision_config(mmproj_gguf)
    if vision_config is None:
        raise ValueError(
            "mmproj GGUF has no vision encoder (clip.has_vision_encoder is unset)."
        )
    resolved_image_token_id = (
        image_token_id
        if image_token_id is not None
        else config.image_token_id
        if config.image_token_id is not None
        else _token_id(text_gguf, "<image_soft_token>")
    )
    config = dataclasses.replace(
        config,
        vision=vision_config,
        image_token_id=resolved_image_token_id,
    )
    if dtype is not None:
        resolved_dtype = resolve_dtype(dtype)
        if resolved_dtype is not None:
            config = dataclasses.replace(config, dtype=resolved_dtype)
    _validate_projector_output_and_media_tokens(config, text_gguf, mmproj_gguf)

    preserve_quantization = keep_quantized and _has_quantized_weights(text_gguf, "gemma3")
    _reject_unsupported_quantization_preservation(
        text_gguf,
        "gemma3",
        preserve_quantization=preserve_quantization,
        allow_quantized_embeddings=False,
        allow_quantized_lm_head=False,
    )
    if preserve_quantization:
        bits, block_size, symmetric = _detect_quant_params(text_gguf, "gemma3")
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=block_size,
                quant_method="gguf",
                sym=symmetric,
                quantize_embeddings=False,
                quantize_lm_head=False,
                tie_word_embeddings=False,
            ),
        )

    module = Gemma3MultiModalModel(config)
    if preserve_quantization:
        _replace_native_block_linears(module.decoder, text_gguf, "gemma3")
    pkg = build_from_module(
        module,
        config,
        task=Gemma3VisionLanguageTask(),
        execution_provider=execution_provider,
    )

    if preserve_quantization:
        text_state = _load_quantized_state_dict(text_gguf, "gemma3", module.decoder, config)
    else:
        text_state = _load_dequantized_state_dict(text_gguf, "gemma3")
    float_state = {
        key: value
        for key, value in text_state.items()
        if not key.endswith((".scales", ".zero_points", ".qweight"))
        and value.dtype != torch.uint8
    }
    retained_state = {
        key: value for key, value in text_state.items() if key not in float_state
    }
    config._gguf_arch = text_gguf.architecture
    float_state = _normalize_gguf_weights(process_tensors(float_state, config))
    text_state = {**float_state, **retained_state}

    state_dict = {_gemma3_multimodal_name(key): value for key, value in text_state.items()}
    state_dict.update(_mmproj_gemma3_vision_to_hf(mmproj_gguf))
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)

    from mobius.integrations.gguf._tokenizer import inspect_gguf_tokenizer

    pkg.gguf_source_path = str(Path(resolved_text_path).resolve())
    pkg.gguf_tokenizer_verdict = inspect_gguf_tokenizer(
        text_gguf.metadata, source=str(resolved_text_path)
    )
    return pkg


def _preflight_mmproj_quantization_report(
    mmproj_gguf: Any,
    *,
    include_audio: bool,
) -> GGUFQuantizationReport:
    """Classify every mapped mmproj vision/(audio) tensor before conversion.

    The Gemma4 vision (and optional audio) encoder always builds float
    parameters regardless of the mmproj source qtype -- see
    ``build_gemma4_vlm_from_gguf``'s "Mixed precision" note -- so every mapped
    tensor here lands on an explicit float disposition: a tensor already
    stored as float (F32/F16/BF16) is ``SOURCE_FLOAT``; anything else is
    unconditionally dequantized on load (:meth:`GGUFModel.get_tensor` always
    dequantizes) and is ``DEQUANTIZED_FLOAT``. There is no lossy-requantize
    path for mmproj tensors, so this never contributes to the fidelity
    warning.

    Record names are qualified with an ``"mmproj:"`` prefix so they cannot
    collide with the text GGUF's own tensor names when the two component
    reports are merged (see :func:`_merge_component_quantization_reports`)
    into the package-level report.
    """
    from mobius.integrations.gguf._mmproj_mapping import (
        map_mmproj_audio_to_hf,
        map_mmproj_vision_to_hf,
    )
    from mobius.integrations.gguf._quant_registry import (
        float_storage_type_ids,
        get_quant_spec,
    )
    from mobius.integrations.gguf._quantization_report import (
        GGUFQuantizationReport,
        QuantizationDisposition,
        QuantizationTensorRecord,
    )
    from mobius.integrations.gguf._spec import Support

    float_type_ids = float_storage_type_ids()
    metadata = getattr(mmproj_gguf, "metadata", {})
    vision_projector_type = (
        projector_type_for_modality(metadata, MMProjModality.VISION) if metadata else "gemma4v"
    )
    audio_projector_type = (
        (
            projector_type_for_modality(metadata, MMProjModality.AUDIO)
            if metadata
            else "gemma4a"
        )
        if include_audio
        else None
    )
    source_qtypes: list[tuple[str, int]] = []
    records: list[QuantizationTensorRecord] = []
    rejected: list[QuantizationTensorRecord] = []
    for tensor in mmproj_gguf.reader_tensors():
        qtype = tensor.tensor_type
        qtype_id = getattr(qtype, "value", qtype)
        quant_spec = get_quant_spec(qtype)
        qtype_name = (
            quant_spec.name if quant_spec is not None else str(getattr(qtype, "name", qtype))
        )
        source_bytes = int(tensor.n_bytes)
        source_qtypes.append((qtype_name, source_bytes))

        hf_name = None
        if tensor.name.startswith("v.") or tensor.name == "mm.input_projection.weight":
            if vision_projector_type == "gemma4uv":
                from mobius.integrations.gguf._core_vlm_projector import (
                    map_core_vlm_projector_tensor,
                )

                hf_name = map_core_vlm_projector_tensor(tensor.name, "gemma4uv")
            else:
                hf_name = map_mmproj_vision_to_hf(tensor.name)
        elif include_audio and (
            tensor.name.startswith("a.") or tensor.name == "mm.a.input_projection.weight"
        ):
            if audio_projector_type == "gemma4ua":
                from mobius.integrations.gguf._core_vlm_projector import (
                    map_core_vlm_projector_tensor,
                )

                hf_name = map_core_vlm_projector_tensor(tensor.name, "gemma4ua")
            else:
                hf_name = map_mmproj_audio_to_hf(tensor.name)
        if hf_name is None:
            # Unmapped tensor (e.g. a different modality, or an unused
            # metadata tensor): still counted in the source qtype census
            # above, but not a mapped-weight record.
            continue

        if qtype_id in float_type_ids:
            disposition = QuantizationDisposition.SOURCE_FLOAT
            reason = (
                "The mmproj vision/audio tensor is already stored as float and "
                "the encoder always builds float parameters."
            )
        elif quant_spec is None:
            disposition = QuantizationDisposition.REJECTED
            reason = "The qtype is outside the pinned llama.cpp census."
        elif quant_spec.dequantize is not Support.SUPPORTED:
            disposition = QuantizationDisposition.REJECTED
            reason = "The mapped mmproj tensor has no trusted float dequantizer."
        else:
            disposition = QuantizationDisposition.DEQUANTIZED_FLOAT
            reason = (
                "The mmproj vision/audio encoder always builds float parameters; "
                "the quantized source tensor is unconditionally dequantized on load."
            )
        record = QuantizationTensorRecord(
            name=f"mmproj:{tensor.name}",
            qtype=qtype_name,
            source_bytes=source_bytes,
            disposition=disposition,
            target_storage=(
                "rejected" if disposition is QuantizationDisposition.REJECTED else "float"
            ),
            reason=reason,
        )
        records.append(record)
        if disposition is QuantizationDisposition.REJECTED:
            rejected.append(record)

    if rejected:
        details = "; ".join(
            f"{record.name} ({record.qtype}): {record.reason}" for record in rejected[:5]
        )
        suffix = "" if len(rejected) <= 5 else f"; and {len(rejected) - 5} more"
        raise ValueError(
            "GGUF quantization preflight could not determine a safe disposition for "
            f"{len(rejected)} mapped mmproj tensor(s): {details}{suffix}"
        )

    return GGUFQuantizationReport.create(
        source_qtypes=source_qtypes,
        tensor_records=records,
        target_storage_format="float",
        compute_mode="float operators",
        compute_capability="Storage is float and executes through float operators.",
    )


def _merge_component_quantization_reports(
    *reports: GGUFQuantizationReport,
) -> GGUFQuantizationReport:
    """Merge independently-preflighted reports from *different* GGUF sources.

    :meth:`GGUFQuantizationReport.combine` requires every component to share
    one GGUF source qtype census -- it's designed to reassemble one file's
    main graph plus its MTP-head sidecar. The Gemma4 multimodal package
    instead draws from two independent files (the text backbone GGUF and the
    companion mmproj GGUF), each with its own census, so this recomputes the
    merged census/dispositions directly from every component's raw per-tensor
    statistics via :meth:`GGUFQuantizationReport.create` rather than asserting
    the censuses match.

    Tensor records must already use source-qualified names where collisions
    are possible (see :func:`_preflight_mmproj_quantization_report`'s
    ``"mmproj:"`` prefix); a name collision with conflicting dispositions
    raises, mirroring :meth:`GGUFQuantizationReport.combine`.
    """
    from mobius.integrations.gguf._quantization_report import (
        GGUFQuantizationReport,
    )

    if not reports:
        raise ValueError("At least one GGUF quantization report is required")
    records_by_name: dict[str, QuantizationTensorRecord] = {}
    for report in reports:
        for record in report.tensor_records:
            previous = records_by_name.setdefault(record.name, record)
            if previous != record:
                raise ValueError(
                    f"Conflicting GGUF quantization dispositions for {record.name!r}"
                )
    source_qtypes = [
        (stat.qtype, stat.source_bytes if index == 0 else 0)
        for report in reports
        for stat in report.source_qtype_census
        for index in range(stat.tensor_count)
    ]
    target_formats = {
        target for report in reports for target in report.target_storage_format.split(" + ")
    }
    compute_modes = {report.compute_mode for report in reports}
    compute_capabilities = {report.compute_capability for report in reports}
    return GGUFQuantizationReport.create(
        source_qtypes=source_qtypes,
        tensor_records=records_by_name.values(),
        target_storage_format=" + ".join(sorted(target_formats)),
        compute_mode=" + ".join(sorted(compute_modes)),
        compute_capability=" ".join(sorted(compute_capabilities)),
    )


def build_gemma4_vlm_from_gguf(
    text_gguf_path: str | Path,
    mmproj_gguf_path: str | Path,
    *,
    dtype: str | None = None,
    execution_provider: str = "default",
    image_token_id: int | None = None,
    include_audio: bool | None = None,
    keep_quantized: bool = True,
    _text_gguf_model: Any | None = None,
    _mmproj_gguf_model: Any | None = None,
) -> ModelPackage:
    """Build a full Gemma4 multimodal ONNX package from text + mmproj GGUFs.

    Args:
        text_gguf_path: Path (or HF ref) to the Gemma4 text-backbone GGUF.
        mmproj_gguf_path: Path (or HF ref) to the companion ``clip`` mmproj GGUF.
        dtype: Optional dtype override (e.g. ``"f16"``); defaults to float32.
        execution_provider: Target EP for EP-aware optimisations.
        image_token_id: Vocabulary id of the image soft-token placeholder used
            to scatter image features into text embeddings. When ``None``, the
            value carried by the text config is used (if any).
        include_audio: Whether to build an audio role when the sidecar carries
            one. ``None`` (the default) preserves every active supported role.
        keep_quantized: Preserve the text backbone's GGUF quantization when
            present as quantized target storage. This is the default: decoder projections become
            MatMulNBits and compatible token-embedding tables become
            GatherBlockQuantized. Incompatible embedding qtypes or shapes stay
            float. Quantized projection source types, including native
            IQ/MXFP4 blocks, are normalized to the common affine layout rather
            than retained byte-for-byte. Lossy normalization emits one warning
            and is recorded in ``quantization_report.json``. Set to ``False`` to dequantize all
            text weights. The vision (and audio) encoder always stays float
            because its weights come from the mmproj as F16 — see the "Mixed
            precision" note below. ``quantization_report.json`` also censuses
            the mmproj vision/(audio) source tensors and records their
            (always-float) dispositions alongside the text backbone's, under
            source-qualified ``"mmproj:"``-prefixed tensor names.

    Returns:
        A :class:`ModelPackage` with ``decoder`` + ``vision_encoder`` +
        ``embedding`` components (plus ``audio_encoder`` if ``include_audio``).

    Mixed precision:
        Quantized text weights yield a mixed-precision package. Only the
        Gemma4 *text* components read ``config.quantization`` (see
        :func:`mobius.models.gemma4._text_linear_class`); the vision/audio
        encoder modules always build float ``Linear`` layers, so a single
        module-global :class:`QuantizationConfig` quantizes the decoder +
        embedding while leaving the mmproj-sourced vision encoder float — no
        per-module quantization opt-out is required.
    """
    import dataclasses

    from mobius._builder import resolve_dtype
    from mobius._configs import Gemma4Config
    from mobius.integrations.gguf._builder import (
        _has_quantized_weights,
        _preflight_quantization_report,
        _reject_unsupported_quantization_preservation,
        _validate_gguf_model,
    )
    from mobius.integrations.gguf._config_mapping import gguf_to_config
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.models.gemma4 import Gemma4Model, Gemma4UnifiedModel
    from mobius.tasks._gemma4 import Gemma4Task, Gemma4UnifiedTask

    resolved_text_path = _resolve_local_path(text_gguf_path)
    text_gguf = (
        _text_gguf_model
        if _text_gguf_model is not None
        else _open_text_gguf(resolved_text_path)
    )
    if text_gguf.architecture != "gemma4":
        raise ValueError(
            "Gemma4 VLM package construction requires a gemma4 text GGUF; "
            f"got {text_gguf.architecture!r}"
        )
    _validate_gguf_model(text_gguf, source=str(text_gguf_path))

    preserve_quantization = keep_quantized and _has_quantized_weights(text_gguf, "gemma4")
    _reject_unsupported_quantization_preservation(
        text_gguf,
        "gemma4",
        preserve_quantization=preserve_quantization,
        allow_native_blocks=False,
    )
    if keep_quantized and not preserve_quantization:
        logger.info(
            "Text GGUF contains no mapped quantized weights; using the float import path"
        )

    resolved_mmproj_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
    mmproj_gguf = (
        _mmproj_gguf_model
        if _mmproj_gguf_model is not None
        else GGUFModel(resolved_mmproj_path)
    )
    _validate_gguf_model(
        mmproj_gguf,
        source=str(mmproj_gguf_path),
        allow_mmproj_companion=True,
    )
    vision_projector_type = projector_type_for_modality(
        mmproj_gguf.metadata,
        MMProjModality.VISION,
    )
    is_unified = vision_projector_type == "gemma4uv"
    if include_audio is None:
        include_audio = bool(mmproj_gguf.metadata.get("clip.has_audio_encoder"))
    modalities = (
        (MMProjModality.VISION, MMProjModality.AUDIO)
        if include_audio
        else (MMProjModality.VISION,)
    )
    _preflight_mmproj_pair(text_gguf, mmproj_gguf, modalities=modalities)
    logger.info("Building Gemma4 VLM from text=%s mmproj=%s", text_gguf_path, mmproj_gguf_path)

    # 1. Text config + merged vision/audio sub-configs.
    config = cast(Gemma4Config, gguf_to_config(text_gguf))
    if is_unified:
        from mobius.integrations.gguf._core_vlm_projector import (
            read_core_vlm_projector_config,
        )

        vision_config = read_core_vlm_projector_config(
            mmproj_gguf,
            "gemma4uv",
        ).vision
    else:
        vision_config = read_mmproj_vision_config(mmproj_gguf)
    if vision_config is None:
        raise ValueError(
            "mmproj GGUF has no vision encoder (clip.has_vision_encoder is unset)."
        )
    config = dataclasses.replace(config, vision=vision_config)

    if include_audio:
        if is_unified:
            audio_config = read_core_vlm_projector_config(
                mmproj_gguf,
                "gemma4ua",
            ).audio
        else:
            audio_config = read_mmproj_audio_config(mmproj_gguf)
        config = dataclasses.replace(config, audio=audio_config)
    else:
        # Vision-only VLM: drop any audio sub-config so the package is the
        # 3-component (decoder + vision + embedding) multimodal shape.
        config = dataclasses.replace(config, audio=None)

    resolved_image_token_id = (
        image_token_id
        if image_token_id is not None
        else config.image_token_id
        if config.image_token_id is not None
        else _token_id(text_gguf, "<|image|>")
    )
    updates: dict[str, Any] = {
        "image_token_id": resolved_image_token_id,
        "model_type": "gemma4_unified" if is_unified else config.model_type,
    }
    if include_audio:
        resolved_audio_token_id = (
            config.audio_token_id
            if config.audio_token_id is not None
            else _token_id(text_gguf, "<|audio|>")
        )
        updates["audio_token_id"] = resolved_audio_token_id
        if config.audio is not None:
            updates["audio"] = dataclasses.replace(
                config.audio,
                audio_token_id=resolved_audio_token_id,
            )
    config = dataclasses.replace(config, **updates)
    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None:
            config = dataclasses.replace(config, dtype=resolved)
    _validate_projector_output_and_media_tokens(
        config, text_gguf, mmproj_gguf, require_audio=include_audio
    )

    # 1b. Quantized mode: set the module-global quantization config from the
    # text GGUF BEFORE building so the text graph emits MatMulNBits and, when
    # compatible, GatherBlockQuantized. The vision/audio encoders stay float.
    quant_params: tuple[int, int, bool] | None = None
    if preserve_quantization:
        from mobius._configs import QuantizationConfig
        from mobius.integrations.gguf._builder import (
            _can_quantize_embedding,
            _can_quantize_lm_head,
            _detect_quant_params,
        )

        bits, block_size, is_symmetric = _detect_quant_params(text_gguf, "gemma4")
        quant_params = (bits, block_size, is_symmetric)
        quantize_embeddings = _can_quantize_embedding(
            text_gguf, "gemma4", bits=bits, block_size=block_size
        )
        quantize_lm_head = (
            quantize_embeddings
            if config.tie_word_embeddings
            else _can_quantize_lm_head(text_gguf, "gemma4")
        )
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=block_size,
                quant_method="gguf",
                sym=is_symmetric,
                quantize_embeddings=quantize_embeddings,
                quantize_lm_head=quantize_lm_head,
                tie_word_embeddings=quantize_lm_head and config.tie_word_embeddings,
            ),
        )
        logger.info(
            "Quantized multimodal mode: bits=%d, block_size=%d, symmetric=%s, "
            "embedding=%s, lm_head=%s (vision/audio stay float)",
            bits,
            block_size,
            is_symmetric,
            quantize_embeddings,
            quantize_lm_head,
        )

    # 2. Build the multimodal graph (decoder + vision + embedding [+ audio]).
    #    Route through build_from_module so each component gets the EP-aware
    #    optimize_model passes (GQA fusion, etc.) — the same pipeline the
    #    text-only build_from_gguf path uses; calling Gemma4Task().build()
    #    directly would skip those optimizations.
    from mobius._builder import build_from_module

    module = Gemma4UnifiedModel(config) if is_unified else Gemma4Model(config)
    task = Gemma4UnifiedTask() if is_unified else Gemma4Task()

    def target_name(hf_name: str) -> str:
        if hf_name.startswith("language_model.lm_head."):
            return "decoder.lm_head." + hf_name.removeprefix("language_model.lm_head.")
        if hf_name.startswith("language_model.embed_tokens"):
            return "embedding." + hf_name.removeprefix("language_model.")
        if hf_name.startswith("language_model."):
            return "decoder.model." + hf_name.removeprefix("language_model.")
        return hf_name

    # The text-only preflight below already emits the single deterministic
    # fidelity warning (via its default emit_warning=True) when the text
    # backbone has lossy tensors. The mmproj vision/(audio) report merged in
    # afterwards can never introduce a lossy-requantize tensor (mmproj
    # weights always stay float — see _preflight_mmproj_quantization_report),
    # so it cannot change that warning; the merge below must not re-emit it.
    quantization = config.quantization
    quantization_report = _preflight_quantization_report(
        text_gguf,
        "gemma4",
        module,
        config,
        preserve_quantization=preserve_quantization,
        target_bits=(
            quantization.bits if preserve_quantization and quantization is not None else None
        ),
        target_block_size=(
            quantization.group_size
            if preserve_quantization and quantization is not None
            else None
        ),
        execution_provider=execution_provider,
        name_mapper=lambda name, _architecture: _text_gguf_name_to_hf_multimodal(name),
        target_name_mapper=target_name,
    )
    mmproj_quantization_report = _preflight_mmproj_quantization_report(
        mmproj_gguf, include_audio=include_audio
    )
    quantization_report = _merge_component_quantization_reports(
        quantization_report, mmproj_quantization_report
    )
    pkg = build_from_module(module, config, task=task, execution_provider=execution_provider)
    pkg.gguf_quantization_report = quantization_report
    logger.info("Built Gemma4 VLM graph (%d components: %s)", len(pkg), list(pkg))

    # 3. Assemble the combined HF-multimodal state dict from both GGUFs. The
    #    text backbone is quantized when requested; vision/audio always float.
    if preserve_quantization:
        if quant_params is None:
            raise RuntimeError("Quantized Gemma4 import is missing quantization parameters.")
        bits, block_size, is_symmetric = quant_params
        state_dict = _text_gguf_to_hf_multimodal_quantized(
            text_gguf,
            config,
            bits=bits,
            block_size=block_size,
            symmetric=is_symmetric,
        )
    else:
        state_dict = _text_gguf_to_hf_multimodal(text_gguf)
    if is_unified:
        from mobius.integrations.gguf._core_vlm_projector import (
            _load_core_vlm_projector_weights,
        )

        state_dict = module.preprocess_weights(state_dict)
        state_dict.update(_load_core_vlm_projector_weights(mmproj_gguf, "gemma4uv"))
        if include_audio:
            state_dict.update(_load_core_vlm_projector_weights(mmproj_gguf, "gemma4ua"))
    else:
        state_dict.update(_mmproj_vision_to_hf(mmproj_gguf))
        if include_audio:
            state_dict.update(_mmproj_audio_to_hf(mmproj_gguf))
        # Run the tested HF->ONNX preprocessing for the non-unified layout.
        state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)
    logger.info("Applied %d mapped weights to the Gemma4 VLM package", len(state_dict))
    from mobius.integrations.gguf._tokenizer import inspect_gguf_tokenizer

    pkg.gguf_source_path = str(Path(resolved_text_path).resolve())
    pkg.gguf_tokenizer_verdict = inspect_gguf_tokenizer(
        text_gguf.metadata, source=str(resolved_text_path)
    )

    return pkg


def _mmproj_muse_glimmer_vision_to_hf(mmproj_gguf: Any) -> dict:
    """Load Muse Glimmer mmproj tensors under their HF ``model.vision_*`` names."""
    import torch

    from mobius.integrations.gguf._mmproj_mapping import (
        map_mmproj_muse_glimmer_vision_to_hf,
    )

    state_dict: dict[str, torch.Tensor] = {}
    for name in mmproj_gguf.tensor_names:
        hf_name = map_mmproj_muse_glimmer_vision_to_hf(name)
        if hf_name is None:
            continue
        values = np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
        if name == "v.patch_embd.weight":
            # Conv patch embed [out, in_ch, kh, kw] → Linear [out, in_ch*kh*kw].
            # The encoder consumes pre-patchified pixels in that row-major order.
            values = values.reshape(values.shape[0], -1)
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict


def build_muse_glimmer_vlm_from_gguf(
    text_gguf_path: str | Path,
    mmproj_gguf_path: str | Path,
    *,
    dtype: str | None = None,
    execution_provider: str = "default",
    image_token_id: int | None = None,
    keep_quantized: bool = True,
    _text_gguf_model: Any | None = None,
    _mmproj_gguf_model: Any | None = None,
) -> ModelPackage:
    """Build a Muse Glimmer decoder + vision + embedding package from GGUFs.

    Args:
        text_gguf_path: Path (or HF ref) to the Muse Glimmer text GGUF.
        mmproj_gguf_path: Path (or HF ref) to the companion ``clip`` mmproj.
        dtype: Optional dtype override (e.g. ``"bf16"``).
        execution_provider: Target EP for EP-aware optimisations.
        image_token_id: Vocabulary id of the image placeholder token. GGUF
            carries no such key, so when ``None`` the model's published default
            is used.
        keep_quantized: Preserve the text GGUF's quantization for the decoder
            projections. The token-embedding table and LM head always stay
            float here: the multimodal split shares one embedding table between
            the decoder and the separate embedding graph, and that routing keys
            off the exact float weight name.

    Returns:
        A :class:`ModelPackage` with ``decoder`` + ``vision_encoder`` +
        ``embedding`` components.

    The vision tower always stays float — the mmproj ships it as F32/BF16.
    """
    import dataclasses

    import torch

    from mobius._builder import build_from_module, resolve_dtype
    from mobius.integrations.gguf._builder import (
        _has_quantized_weights,
        _load_dequantized_state_dict,
        _load_quantized_state_dict,
        _normalize_gguf_weights,
        _reject_unsupported_quantization_preservation,
        _replace_native_block_linears,
        _validate_gguf_model,
    )
    from mobius.integrations.gguf._config_mapping import gguf_to_config
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._tensor_processors import process_tensors
    from mobius.models.muse_glimmer import MuseGlimmerForConditionalGeneration
    from mobius.tasks import MuseGlimmerVLTask

    resolved_text_path = _resolve_local_path(text_gguf_path)
    text_gguf = (
        _text_gguf_model
        if _text_gguf_model is not None
        else _open_text_gguf(resolved_text_path)
    )
    _validate_gguf_model(text_gguf, source=str(text_gguf_path))
    text_arch = text_gguf.architecture

    resolved_mmproj_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
    mmproj_gguf = (
        _mmproj_gguf_model
        if _mmproj_gguf_model is not None
        else GGUFModel(resolved_mmproj_path)
    )
    _validate_gguf_model(
        mmproj_gguf,
        source=str(mmproj_gguf_path),
        allow_mmproj_companion=True,
    )
    _preflight_mmproj_pair(text_gguf, mmproj_gguf, modalities=(MMProjModality.VISION,))
    logger.info(
        "Building Muse Glimmer VLM from text=%s mmproj=%s",
        text_gguf_path,
        mmproj_gguf_path,
    )

    # 1. Text config + vision sub-config.
    config = gguf_to_config(text_gguf)
    vision_config = read_mmproj_muse_glimmer_vision_config(mmproj_gguf)
    if vision_config is None:
        raise ValueError(
            "mmproj GGUF has no vision encoder (clip.has_vision_encoder is unset)."
        )
    config = dataclasses.replace(config, vision=vision_config, audio=None)
    if image_token_id is not None:
        config = dataclasses.replace(config, image_token_id=image_token_id)
    config = _with_muse_glimmer_media_token_ids(config)
    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None:
            config = dataclasses.replace(config, dtype=resolved)
    _validate_projector_output_and_media_tokens(config, text_gguf, mmproj_gguf)

    preserve_quantization = keep_quantized and _has_quantized_weights(text_gguf, text_arch)
    _reject_unsupported_quantization_preservation(
        text_gguf,
        text_arch,
        preserve_quantization=preserve_quantization,
        allow_quantized_embeddings=False,
        allow_quantized_lm_head=False,
    )
    if keep_quantized and not preserve_quantization:
        logger.info(
            "Text GGUF contains no mapped quantized weights; using the float import path"
        )
    if preserve_quantization:
        from mobius._configs import QuantizationConfig
        from mobius.integrations.gguf._builder import _detect_quant_params

        bits, block_size, is_symmetric = _detect_quant_params(text_gguf, text_arch)
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=block_size,
                quant_method="gguf",
                sym=is_symmetric,
                quantize_embeddings=False,
                quantize_lm_head=False,
                tie_word_embeddings=False,
            ),
        )
        logger.info(
            "Quantized multimodal mode: bits=%d, block_size=%d, symmetric=%s "
            "(embedding, LM head and vision stay float)",
            bits,
            block_size,
            is_symmetric,
        )

    # 2. Build the three-component graph.
    module = MuseGlimmerForConditionalGeneration(config)
    if preserve_quantization:
        _replace_native_block_linears(module.decoder, text_gguf, text_arch)
    pkg = build_from_module(
        module, config, MuseGlimmerVLTask(), execution_provider=execution_provider
    )
    logger.info("Built Muse Glimmer VLM graph (%d components: %s)", len(pkg), list(pkg))

    # 3. Text weights under plain HF names. The quantized loader matches GGUF
    #    tensors against module paths, so it is given the decoder sub-module
    #    whose paths are the ``model.*`` HF names it expects.
    if preserve_quantization:
        text_state = _load_quantized_state_dict(text_gguf, text_arch, module.decoder, config)
    else:
        text_state = _load_dequantized_state_dict(text_gguf, text_arch)

    # 4. Architecture fix-ups (Muse Glimmer's centered block norms) run on the
    #    plain HF names, exactly as in the text-only path, before the names are
    #    moved into the multimodal namespace.
    float_state = {
        key: value
        for key, value in text_state.items()
        if not key.endswith((".scales", ".zero_points", ".qweight"))
        and value.dtype != torch.uint8
    }
    rest = {key: value for key, value in text_state.items() if key not in float_state}
    # ``dataclasses.replace`` above drops the plain instance attribute that
    # ``process_tensors`` dispatches on, so restore it before processing.
    config._gguf_arch = text_gguf.architecture
    float_state = _normalize_gguf_weights(process_tensors(float_state, config))
    text_state = {**float_state, **rest}

    # 5. Move the text backbone under ``model.language_model.`` and merge in
    #    the vision tower, then run the model's own HF→ONNX routing.
    state_dict = {
        _muse_glimmer_multimodal_name(key): value for key, value in text_state.items()
    }
    state_dict.update(_mmproj_muse_glimmer_vision_to_hf(mmproj_gguf))
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)
    logger.info("Applied %d mapped weights to the Muse Glimmer VLM package", len(state_dict))
    from mobius.integrations.gguf._tokenizer import inspect_gguf_tokenizer

    pkg.gguf_source_path = str(Path(resolved_text_path).resolve())
    pkg.gguf_tokenizer_verdict = inspect_gguf_tokenizer(
        text_gguf.metadata, source=str(resolved_text_path)
    )

    return pkg


def build_qwen_glm_projector_from_gguf(
    mmproj_gguf_path: str | Path,
    *,
    projector_type: str,
    target_architecture: str,
    dtype: str | None = None,
    execution_provider: str = "default",
    _mmproj_gguf_model: Any | None = None,
) -> ModelPackage:
    """Build the exact standalone Qwen/GLM vision, audio, or speaker roles."""
    from mobius.integrations.gguf._qwen_glm_projector import (
        build_qwen_glm_projector_package,
    )

    resolved_path = Path(mmproj_gguf_path)
    mmproj_gguf = _mmproj_gguf_model
    if mmproj_gguf is None:
        from mobius.integrations.gguf._builder import _validate_gguf_model
        from mobius.integrations.gguf._reader import GGUFModel

        resolved_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
        mmproj_gguf = GGUFModel(resolved_path)
        _validate_gguf_model(
            mmproj_gguf,
            source=str(mmproj_gguf_path),
            allow_mmproj_companion=True,
        )
        _preflight_standalone_mmproj(
            mmproj_gguf,
            projector_type=projector_type,
            target_architecture=target_architecture,
        )
    return build_qwen_glm_projector_package(
        mmproj_gguf,
        resolved_path=resolved_path,
        projector_type=projector_type,
        dtype=dtype,
        execution_provider=execution_provider,
    )


def _mmproj_audio_projector_to_onnx(
    mmproj_gguf: Any,
    projector_type: str,
) -> dict[str, Any]:
    import torch

    from mobius.integrations.gguf._mmproj_mapping import (
        map_mmproj_audio_projector_to_onnx,
    )

    state_dict: dict[str, torch.Tensor] = {}
    for name in mmproj_gguf.tensor_names:
        mapped = map_mmproj_audio_projector_to_onnx(name, projector_type)
        if mapped is None:
            continue
        values = np.asarray(mmproj_gguf.get_tensor(name), dtype=np.float32)
        state_dict[mapped] = torch.from_numpy(values.copy())
    return state_dict


def build_audio_projector_from_gguf(
    mmproj_gguf_path: str | Path,
    *,
    projector_type: str,
    target_architecture: str,
    dtype: str | None = None,
    execution_provider: str = "default",
    _mmproj_gguf_model: Any | None = None,
) -> ModelPackage:
    """Build the standalone MERaLiON audio encoder/projector sidecar."""
    from mobius._builder import build_from_module, resolve_dtype
    from mobius._configs import ArchitectureConfig
    from mobius.integrations.gguf._builder import _validate_gguf_model
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.models.gguf_audio_projector import (
        AUDIO_PROCESSOR_ABIS,
        create_gguf_audio_projector,
    )
    from mobius.tasks import GGUFAudioProjectorModel, GGUFAudioProjectorTask

    resolved_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
    mmproj_gguf = (
        _mmproj_gguf_model if _mmproj_gguf_model is not None else GGUFModel(resolved_path)
    )
    _validate_gguf_model(
        mmproj_gguf,
        source=str(mmproj_gguf_path),
        allow_mmproj_companion=True,
    )
    spec = _preflight_standalone_mmproj(
        mmproj_gguf,
        projector_type=projector_type,
        target_architecture=target_architecture,
    )
    hidden_size = int(mmproj_gguf.metadata["clip.audio.embedding_length"])
    num_heads = int(mmproj_gguf.metadata["clip.audio.attention.head_count"])
    config = ArchitectureConfig(
        model_type=f"gguf_{projector_type}",
        vocab_size=1,
        hidden_size=hidden_size,
        intermediate_size=int(mmproj_gguf.metadata["clip.audio.feed_forward_length"]),
        num_hidden_layers=int(mmproj_gguf.metadata["clip.audio.block_count"]),
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=hidden_size // num_heads,
        max_position_embeddings=65_536,
        hidden_act="gelu",
    )
    if dtype is not None:
        resolved_dtype = resolve_dtype(dtype)
        if resolved_dtype is not None:
            config = dataclasses.replace(config, dtype=resolved_dtype)
    tensor_shapes = {
        name: tuple(int(dim) for dim in mmproj_gguf.get_tensor_shape(name))
        for name in mmproj_gguf.tensor_names
    }
    audio_encoder = create_gguf_audio_projector(
        spec.projector_type,
        mmproj_gguf.metadata,
        tensor_shapes,
    )
    module = GGUFAudioProjectorModel(audio_encoder)
    package = build_from_module(
        module,
        config,
        task=GGUFAudioProjectorTask(),
        execution_provider=execution_provider,
    )
    package.apply_weights(_mmproj_audio_projector_to_onnx(mmproj_gguf, projector_type))
    processor_abi = AUDIO_PROCESSOR_ABIS[projector_type]
    serialized_processor_abi = json.dumps(
        dataclasses.asdict(processor_abi),
        sort_keys=True,
        separators=(",", ":"),
    )
    for model in package.values():
        model.metadata_props["mobius.gguf_projector_type"] = projector_type
        model.metadata_props["mobius.gguf_target_architecture"] = target_architecture
        model.metadata_props["mobius.gguf_audio_processor_abi"] = serialized_processor_abi
        model.metadata_props["mobius.runtime_support"] = (
            "standalone-sidecar-only; paired multimodal runtime unvalidated"
        )
    package.gguf_source_path = str(Path(resolved_path).resolve())  # type: ignore[attr-defined]
    package.gguf_projector_type = projector_type  # type: ignore[attr-defined]
    package.gguf_audio_processor_abi = processor_abi  # type: ignore[attr-defined]
    runtime_warning = (
        "Standalone projector graph only; paired text insertion and downstream "
        "multimodal runtime execution are not validated."
    )
    package.gguf_runtime_warning = runtime_warning  # type: ignore[attr-defined]
    logger.warning("%s", runtime_warning)
    return package


def build_remaining_vision_projector_from_gguf(
    mmproj_gguf_path: str | Path,
    *,
    projector_type: str,
    target_architecture: str,
    dtype: str | None = None,
    execution_provider: str = "default",
    _mmproj_gguf_model: Any | None = None,
) -> ModelPackage:
    """Build one exact standalone vision encoder/projector sidecar."""
    from mobius._builder import build_from_module, resolve_dtype
    from mobius._configs import ArchitectureConfig
    from mobius.integrations.gguf._builder import _validate_gguf_model
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._remaining_projectors import (
        create_remaining_vision_projector,
        remaining_projector_state_dict,
    )
    from mobius.tasks import GGUFVisionProjectorModel, GGUFVisionProjectorTask

    resolved_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
    mmproj_gguf = (
        _mmproj_gguf_model if _mmproj_gguf_model is not None else GGUFModel(resolved_path)
    )
    _validate_gguf_model(
        mmproj_gguf,
        source=str(mmproj_gguf_path),
        allow_mmproj_companion=True,
    )
    _preflight_standalone_mmproj(
        mmproj_gguf,
        projector_type=projector_type,
        target_architecture=target_architecture,
    )
    hidden_size = int(mmproj_gguf.metadata["clip.vision.embedding_length"])
    num_heads = int(mmproj_gguf.metadata["clip.vision.attention.head_count"])
    config = ArchitectureConfig(
        model_type=f"gguf_{projector_type}",
        vocab_size=1,
        hidden_size=hidden_size,
        intermediate_size=int(mmproj_gguf.metadata["clip.vision.feed_forward_length"]),
        num_hidden_layers=int(mmproj_gguf.metadata["clip.vision.block_count"]),
        num_attention_heads=num_heads,
        num_key_value_heads=int(
            mmproj_gguf.metadata.get("clip.vision.attention.head_count_kv", num_heads)
        ),
        head_dim=hidden_size // num_heads,
        max_position_embeddings=65_536,
        hidden_act="gelu",
    )
    if dtype is not None:
        resolved_dtype = resolve_dtype(dtype)
        if resolved_dtype is not None:
            config = dataclasses.replace(config, dtype=resolved_dtype)
    tensor_shapes = {
        name: tuple(int(dim) for dim in mmproj_gguf.get_tensor_shape(name))
        for name in mmproj_gguf.tensor_names
    }
    vision_encoder = create_remaining_vision_projector(
        projector_type,
        mmproj_gguf.metadata,
        tensor_shapes,
    )
    vision_input_schema = cast(Any, vision_encoder).input_schema
    input_schema = [
        {
            "name": name,
            "dtype": str(dtype),
            "shape": [str(dim) for dim in shape],
        }
        for name, dtype, shape in vision_input_schema
    ]
    package = build_from_module(
        GGUFVisionProjectorModel(vision_encoder),
        config,
        task=GGUFVisionProjectorTask(),
        execution_provider=execution_provider,
    )
    for model in package.values():
        model.metadata_props["mobius.gguf_projector_type"] = projector_type
        model.metadata_props["mobius.gguf_target_architecture"] = target_architecture
        model.metadata_props["mobius.gguf_input_schema"] = json.dumps(
            input_schema,
            sort_keys=True,
            separators=(",", ":"),
        )
        model.metadata_props["mobius.runtime_support"] = (
            "standalone-sidecar-only; paired multimodal runtime unvalidated"
        )
    package.apply_weights(remaining_projector_state_dict(mmproj_gguf, projector_type))
    package.gguf_source_path = str(Path(resolved_path).resolve())  # type: ignore[attr-defined]
    package.gguf_projector_type = projector_type  # type: ignore[attr-defined]
    package.gguf_input_schema = input_schema  # type: ignore[attr-defined]
    runtime_warning = (
        "Standalone projector graph only; paired text insertion and downstream "
        "multimodal runtime execution are not validated."
    )
    package.gguf_runtime_warning = runtime_warning  # type: ignore[attr-defined]
    logger.warning("%s", runtime_warning)
    return package


def build_vlm_from_gguf(
    text_gguf_path: str | Path,
    mmproj_gguf_path: str | Path,
    *,
    dtype: str | None = None,
    execution_provider: str = "default",
    image_token_id: int | None = None,
    keep_quantized: bool = True,
    _text_gguf_model: Any | None = None,
) -> ModelPackage:
    """Route a text + mmproj pair to the architecture-specific VLM builder.

    The mmproj itself is always ``general.architecture = clip``, so the text
    backbone is what decides how the pair is assembled.
    """
    from mobius.integrations.gguf._builder import _validate_gguf_model
    from mobius.integrations.gguf._reader import GGUFModel

    resolved_text_path = _resolve_local_path(text_gguf_path)
    text_gguf = (
        _text_gguf_model
        if _text_gguf_model is not None
        else _open_text_gguf(resolved_text_path)
    )
    _validate_gguf_model(text_gguf, source=str(text_gguf_path))
    resolved_mmproj_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
    mmproj_gguf = GGUFModel(resolved_mmproj_path)
    _validate_gguf_model(
        mmproj_gguf,
        source=str(mmproj_gguf_path),
        allow_mmproj_companion=True,
    )
    specs = _preflight_mmproj_pair(text_gguf, mmproj_gguf, modalities=(MMProjModality.VISION,))
    builder = _resolve_vlm_builder(
        text_gguf.architecture, specs[MMProjModality.VISION].projector_type
    )
    pkg = builder(
        resolved_text_path,
        resolved_mmproj_path,
        dtype=dtype,
        execution_provider=execution_provider,
        image_token_id=image_token_id,
        keep_quantized=keep_quantized,
        _text_gguf_model=text_gguf,
        _mmproj_gguf_model=mmproj_gguf,
    )
    return pkg


#: Named multimodal assembly entry points that
#: :attr:`GGUFArchitectureSpec.vlm_builder` selects, held as module attribute
#: names so dispatch resolves at call time rather than capturing the function
#: objects at import.
#:
#: Every name here must be referenced by a spec and every name a spec references
#: must exist here; ``_arch_registry_test`` checks both directions.
_VLM_BUILDERS: dict[str, str] = {
    "generic_projector": "build_generic_projector_vlm_from_gguf",
    "gemma3": "build_gemma3_vlm_from_gguf",
    "gemma4": "build_gemma4_vlm_from_gguf",
    "muse_glimmer": "build_muse_glimmer_vlm_from_gguf",
    "qwen_vl": "build_qwen_vlm_from_gguf",
}


#: Standalone sidecar graph entry points selected by
#: :attr:`ProjectorSpec.sidecar_builder`. Unlike ``_VLM_BUILDERS``, these
#: functions never create or silently omit a paired text decoder.
def _build_core_vlm_projector_mmproj(*args, **kwargs) -> ModelPackage:
    from mobius.integrations.gguf._core_vlm_projector import (
        build_core_vlm_projector_mmproj,
    )

    return build_core_vlm_projector_mmproj(*args, **kwargs)


_MMPROJ_BUILDERS: dict[str, str] = {
    "core_vlm_projector": "_build_core_vlm_projector_mmproj",
    "qwen_glm_projector": "build_qwen_glm_projector_from_gguf",
    "audio_projector": "build_audio_projector_from_gguf",
    "remaining_vision_projector": "build_remaining_vision_projector_from_gguf",
}


def _resolve_vlm_builder(text_arch: str, projector_type: str) -> Callable[..., ModelPackage]:
    """Resolve dispatch only from an exact supported projector/target pair."""
    from mobius.integrations.gguf._arch_registry import try_get_arch_spec

    projector_spec = get_projector_spec(projector_type)
    if not projector_spec.is_importable:
        raise NotImplementedError(
            f"clip projector {projector_type!r} cannot build: {projector_spec.reason}"
        )
    text_spec = try_get_arch_spec(text_arch)
    canonical_text_arch = text_arch if text_spec is None else text_spec.gguf_arch
    if canonical_text_arch not in projector_spec.target_architectures:
        raise ValueError(
            f"clip projector {projector_type!r} targets "
            f"{sorted(projector_spec.target_architectures)}, not {text_arch!r}."
        )
    if text_spec is None or text_spec.vlm_builder != projector_spec.builder:
        raise ValueError(
            f"Text architecture {text_arch!r} and clip projector "
            f"{projector_type!r} do not declare the same VLM builder."
        )
    builder_name = projector_spec.builder
    if builder_name is None:
        raise RuntimeError(
            f"Importable projector {projector_type!r} has no registered VLM builder."
        )
    attribute = _VLM_BUILDERS.get(builder_name)
    if attribute is None:
        raise RuntimeError(
            f"Projector registry references unknown VLM builder {projector_spec.builder!r}."
        )
    builder: Callable[..., ModelPackage] = globals()[attribute]
    return builder


def _muse_glimmer_multimodal_name(hf_name: str) -> str:
    """Nest a text-backbone HF name under the multimodal namespace.

    ``MuseGlimmerForConditionalGeneration.preprocess_weights`` expects the
    decoder weights as ``model.language_model.*`` and leaves ``lm_head.*``
    at the top level.
    """
    if hf_name.startswith("model."):
        return "model.language_model." + hf_name[len("model.") :]
    return hf_name


def _mmproj_audio_to_hf(mmproj_gguf: Any) -> dict:
    """Load mmproj audio tensors as HF names (experimental; see module docstring)."""
    import torch

    from mobius.integrations.gguf._mmproj_mapping import map_mmproj_audio_to_hf

    state_dict: dict[str, torch.Tensor] = {}
    for name in mmproj_gguf.tensor_names:
        if not (name.startswith("a.") or name == "mm.a.input_projection.weight"):
            continue
        hf_name = map_mmproj_audio_to_hf(name)
        if hf_name is None:
            continue
        values = np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
        if hf_name.endswith(".per_dim_scale"):
            # Gemma converters store softplus(raw); the model applies softplus.
            if np.any(values <= 0):
                raise ValueError(f"{name} must contain positive baked softplus values.")
            values = np.log(np.expm1(values.astype(np.float64))).astype(np.float32)
        elif name.endswith(".conv_dw.weight") and values.ndim == 2:
            values = values[:, None, :]
        elif name.endswith(_CLIPPING_BOUND_SUFFIXES):
            values = values.reshape(())
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict
