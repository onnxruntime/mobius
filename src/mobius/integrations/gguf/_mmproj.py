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
:class:`Gemma4AudioConfig` and the audio tensor mapping exists
(:func:`map_mmproj_audio_to_hf`), but the audio encoder is **not** yet wired
into the assembled package — its Conformer forward-pass weight layout still
needs validation against llama.cpp's ``clip.cpp`` gemma4a reference.  Pass
``include_audio=True`` to opt in to the (experimental) audio path.
"""

from __future__ import annotations

__all__ = [
    "build_gemma4_vlm_from_gguf",
    "build_vlm_from_gguf",
    "build_muse_glimmer_vlm_from_gguf",
    "read_mmproj_audio_config",
    "read_mmproj_muse_glimmer_vision_config",
    "read_mmproj_vision_config",
]

import dataclasses
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

logger = logging.getLogger(__name__)

# Gemma4 vision RoPE base frequency. The ``clip`` mmproj metadata does not store
# it, so use the HF Gemma4VisionAttention default. Source: HF Gemma4 vision.
_GEMMA4_VISION_ROPE_THETA = 100.0
# Gemma4 vision pooler spatial average pooling kernel (k x k). Not present in
# mmproj metadata; the SigLIP encoder pools N patches to N/k^2 soft tokens.
_DEFAULT_POOLING_KERNEL_SIZE = 3

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
        rms_norm_eps=float(md.get("clip.audio.attention.layer_norm_epsilon", 1e-6)),
    )


def _resolve_local_path(path: str | Path) -> str:
    from mobius.integrations.gguf._builder import _resolve_gguf_path

    return _resolve_gguf_path(path)


_CLIPPING_BOUND_SUFFIXES = (".input_min", ".input_max", ".output_min", ".output_max")
_FLOAT_MMPROJ_QTYPES = frozenset({"F32", "F16", "BF16"})


def _canonical_text_architecture(architecture: str) -> str:
    from mobius.integrations.gguf._arch_registry import try_get_arch_spec

    arch_spec = try_get_arch_spec(architecture)
    return architecture if arch_spec is None else arch_spec.gguf_arch


def _normalized_identity(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _validate_mmproj_source_identity(text_gguf: Any, mmproj_gguf: Any) -> None:
    """Reject a sidecar whose declared base model contradicts the text GGUF."""
    text_name = text_gguf.metadata.get("general.name")
    sidecar_base = mmproj_gguf.metadata.get("general.base_model.0.name")
    if text_name and sidecar_base:
        text_identity = _normalized_identity(text_name)
        sidecar_identity = _normalized_identity(sidecar_base)
        if text_identity != sidecar_identity:
            raise ValueError(
                "mmproj source identity does not match the text GGUF: "
                f"general.base_model.0.name={sidecar_base!r}, "
                f"text general.name={text_name!r}."
            )
    general_type = mmproj_gguf.metadata.get("general.type")
    if general_type not in (None, "mmproj"):
        raise ValueError(f"clip sidecar general.type must be 'mmproj', got {general_type!r}.")


def _mmproj_tensor_names_for_spec(mmproj_gguf: Any, spec: ProjectorSpec) -> set[str]:
    prefixes = tuple({role_prefix for role_prefix, _ in spec.tensor_roles})
    return {
        name
        for name in mmproj_gguf.tensor_names
        if name.startswith(prefixes)
        or name in spec.required_top_tensors
        or name in spec.optional_top_tensors
    }


def _validate_mmproj_tensor_closure(mmproj_gguf: Any, spec: ProjectorSpec) -> None:
    """Validate exact names, layers, shapes, and float storage before graph build."""
    names = _mmproj_tensor_names_for_spec(mmproj_gguf, spec)
    top = set(spec.required_top_tensors) | set(spec.optional_top_tensors)
    block_names: dict[int, set[str]] = {}
    unexpected: list[str] = []
    prefix = spec.block_prefix
    for name in names:
        if name in top:
            continue
        if prefix is None or not name.startswith(prefix):
            unexpected.append(name)
            continue
        remainder = name[len(prefix) :]
        index_text, separator, suffix = remainder.partition(".")
        if not separator or not index_text.isdecimal() or suffix not in spec.block_suffixes:
            unexpected.append(name)
            continue
        block_names.setdefault(int(index_text), set()).add(suffix)
    if unexpected:
        raise ValueError(
            f"{spec.projector_type} mmproj has tensors outside the pinned suffix-exact "
            f"loader closure: {sorted(unexpected)}. Projector tensors are never dropped."
        )

    missing_top = sorted(set(spec.required_top_tensors) - names)
    if missing_top:
        raise ValueError(
            f"{spec.projector_type} mmproj is missing required tensor(s): {missing_top}"
        )
    layers = int(mmproj_gguf.metadata["clip.vision.block_count"])
    expected_layers = set(range(layers))
    if set(block_names) != expected_layers:
        raise ValueError(
            f"{spec.projector_type} mmproj block indices are "
            f"{sorted(block_names)}, expected {sorted(expected_layers)}."
        )
    required_suffixes = set(spec.block_suffixes)
    missing_by_layer = {
        layer: sorted(required_suffixes - block_names[layer])
        for layer in sorted(block_names)
        if required_suffixes - block_names[layer]
    }
    if missing_by_layer:
        raise ValueError(
            f"{spec.projector_type} mmproj is missing required block suffixes: "
            f"{missing_by_layer}"
        )

    for name in sorted(names):
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

    _validate_supported_mmproj_shapes(mmproj_gguf, spec)


def _expect_mmproj_shape(mmproj_gguf: Any, name: str, expected: tuple[int, ...]) -> None:
    actual = mmproj_gguf.get_tensor_shape(name)
    if actual != expected:
        raise ValueError(f"mmproj tensor {name!r} has shape {actual}, expected {expected}.")


def _validate_supported_mmproj_shapes(mmproj_gguf: Any, spec: ProjectorSpec) -> None:
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
    if len(patch_shape) not in (4, 5) or patch_shape[0] != hidden:
        raise ValueError(
            f"v.patch_embd.weight must be rank 4/5 with output {hidden}, got {patch_shape}."
        )
    if patch_shape[-2:] != (patch, patch):
        raise ValueError(
            f"v.patch_embd.weight spatial kernel {patch_shape[-2:]} does not match "
            f"clip.vision.patch_size={patch}."
        )

    if spec.projector_type == "gemma4v":
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
    _validate_mmproj_source_identity(text_gguf, mmproj_gguf)
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
        if not spec.is_supported:
            blocked = ", ".join(
                name
                for name, verdict in spec.verdicts.items()
                if verdict is not Support.SUPPORTED
            )
            raise NotImplementedError(
                f"clip projector {projector_type!r} is known at llama.cpp "
                f"{LLAMA_CPP_MMPROJ_SHA} but is deferred/rejected "
                f"({blocked} unavailable): {spec.reason}"
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
        _validate_mmproj_tensor_closure(mmproj_gguf, spec)
        resolved[modality] = spec
    return resolved


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
    from mobius.integrations.gguf._spec import TensorRole

    quantize_embeddings = bool(getattr(config.quantization, "quantize_embeddings", False))
    quantize_lm_head = bool(getattr(config.quantization, "quantize_lm_head", False))
    tie_word_embeddings = bool(config.tie_word_embeddings)

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


def build_gemma4_vlm_from_gguf(
    text_gguf_path: str | Path,
    mmproj_gguf_path: str | Path,
    *,
    dtype: str | None = None,
    execution_provider: str = "default",
    image_token_id: int | None = None,
    include_audio: bool = False,
    keep_quantized: bool = True,
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
        include_audio: When ``True``, also build the (experimental) audio
            encoder. Off by default — see the module docstring.
        keep_quantized: Preserve the text backbone's GGUF quantization when
            present. This is the default: decoder projections become
            MatMulNBits and compatible token-embedding tables become
            GatherBlockQuantized. Incompatible embedding qtypes or shapes stay
            float. Quantized projection source types, including native
            IQ/MXFP4 blocks, are normalized to the common affine layout rather
            than retained byte-for-byte. Set to ``False`` to dequantize all
            text weights. The vision (and audio) encoder always stays float
            because its weights come from the mmproj as F16 — see the "Mixed
            precision" note below.

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
    from mobius.integrations.gguf._builder import (
        _has_quantized_weights,
        _validate_gguf_model,
    )
    from mobius.integrations.gguf._config_mapping import gguf_to_config
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.models.gemma4 import Gemma4Model
    from mobius.tasks._gemma4 import Gemma4Task

    text_gguf = GGUFModel(_resolve_local_path(text_gguf_path))
    _validate_gguf_model(text_gguf, source=str(text_gguf_path))

    preserve_quantization = keep_quantized and _has_quantized_weights(text_gguf, "gemma4")
    if keep_quantized and not preserve_quantization:
        logger.info(
            "Text GGUF contains no mapped quantized weights; using the float import path"
        )

    mmproj_gguf = GGUFModel(_resolve_local_path(mmproj_gguf_path))
    _validate_gguf_model(mmproj_gguf, source=str(mmproj_gguf_path))
    modalities = (
        (MMProjModality.VISION, MMProjModality.AUDIO)
        if include_audio
        else (MMProjModality.VISION,)
    )
    _preflight_mmproj_pair(text_gguf, mmproj_gguf, modalities=modalities)
    logger.info("Building Gemma4 VLM from text=%s mmproj=%s", text_gguf_path, mmproj_gguf_path)

    # 1. Text config + merged vision/audio sub-configs.
    config = gguf_to_config(text_gguf)
    vision_config = read_mmproj_vision_config(mmproj_gguf)
    if vision_config is None:
        raise ValueError(
            "mmproj GGUF has no vision encoder (clip.has_vision_encoder is unset)."
        )
    config = dataclasses.replace(config, vision=vision_config)

    if include_audio:
        config = dataclasses.replace(config, audio=read_mmproj_audio_config(mmproj_gguf))
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
    updates: dict[str, Any] = {"image_token_id": resolved_image_token_id}
    if include_audio and config.audio_token_id is None:
        updates["audio_token_id"] = _token_id(text_gguf, "<|audio|>")
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

    module = Gemma4Model(config)
    pkg = build_from_module(
        module, config, task=Gemma4Task(), execution_provider=execution_provider
    )
    logger.info("Built Gemma4 VLM graph (%d components: %s)", len(pkg), list(pkg))

    # 3. Assemble the combined HF-multimodal state dict from both GGUFs. The
    #    text backbone is quantized when requested; vision/audio always float.
    if preserve_quantization:
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
    state_dict.update(_mmproj_vision_to_hf(mmproj_gguf))
    if include_audio:
        state_dict.update(_mmproj_audio_to_hf(mmproj_gguf))

    # 4. Run the tested HF→ONNX preprocessing and apply. Names are already
    #    component-qualified after preprocessing, so no prefix_map is needed.
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)
    logger.info("Applied %d mapped weights to the Gemma4 VLM package", len(state_dict))

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
        _replace_native_block_linears,
        _validate_gguf_model,
    )
    from mobius.integrations.gguf._config_mapping import gguf_to_config
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._tensor_processors import process_tensors
    from mobius.models.muse_glimmer import MuseGlimmerForConditionalGeneration
    from mobius.tasks import MuseGlimmerVLTask

    text_gguf = GGUFModel(_resolve_local_path(text_gguf_path))
    _validate_gguf_model(text_gguf, source=str(text_gguf_path))
    text_arch = text_gguf.architecture

    mmproj_gguf = GGUFModel(_resolve_local_path(mmproj_gguf_path))
    _validate_gguf_model(mmproj_gguf, source=str(mmproj_gguf_path))
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

    return pkg


def build_vlm_from_gguf(
    text_gguf_path: str | Path,
    mmproj_gguf_path: str | Path,
    *,
    dtype: str | None = None,
    execution_provider: str = "default",
    keep_quantized: bool = True,
) -> ModelPackage:
    """Route a text + mmproj pair to the architecture-specific VLM builder.

    The mmproj itself is always ``general.architecture = clip``, so the text
    backbone is what decides how the pair is assembled.
    """
    from mobius.integrations.gguf._builder import _validate_gguf_model
    from mobius.integrations.gguf._reader import GGUFModel

    text_gguf = GGUFModel(_resolve_local_path(text_gguf_path))
    _validate_gguf_model(text_gguf, source=str(text_gguf_path))
    mmproj_gguf = GGUFModel(_resolve_local_path(mmproj_gguf_path))
    _validate_gguf_model(mmproj_gguf, source=str(mmproj_gguf_path))
    specs = _preflight_mmproj_pair(text_gguf, mmproj_gguf, modalities=(MMProjModality.VISION,))
    builder = _resolve_vlm_builder(
        text_gguf.architecture, specs[MMProjModality.VISION].projector_type
    )
    return builder(
        text_gguf_path,
        mmproj_gguf_path,
        dtype=dtype,
        execution_provider=execution_provider,
        keep_quantized=keep_quantized,
    )


#: Named multimodal assembly entry points that
#: :attr:`GGUFArchitectureSpec.vlm_builder` selects, held as module attribute
#: names so dispatch resolves at call time rather than capturing the function
#: objects at import.
#:
#: Every name here must be referenced by a spec and every name a spec references
#: must exist here; ``_arch_registry_test`` checks both directions.
_VLM_BUILDERS: dict[str, str] = {
    "gemma4": "build_gemma4_vlm_from_gguf",
    "muse_glimmer": "build_muse_glimmer_vlm_from_gguf",
}


def _resolve_vlm_builder(text_arch: str, projector_type: str) -> Callable[..., ModelPackage]:
    """Resolve dispatch only from an exact supported projector/target pair."""
    from mobius.integrations.gguf._arch_registry import try_get_arch_spec

    projector_spec = get_projector_spec(projector_type)
    if not projector_spec.is_supported:
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
    attribute = _VLM_BUILDERS.get(projector_spec.builder)
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
            values = values.reshape(-1)
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict
