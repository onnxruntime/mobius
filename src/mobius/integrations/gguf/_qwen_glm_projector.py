# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact Qwen/GLM ``clip`` sidecar configuration, mapping, and graph helpers."""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn

from mobius._configs import (
    ArchitectureConfig,
    AudioConfig,
    SpeakerEncoderConfig,
    TTSConfig,
    VisionConfig,
)
from mobius._model_package import ModelPackage
from mobius.components import (
    GGUFLegacyGlmAudioProjector,
    GGUFQwen2AudioProjector,
)
from mobius.integrations.gguf._mmproj_mapping import (
    map_mmproj_glm4v_vision_to_onnx,
    map_mmproj_glma_audio_to_onnx,
    map_mmproj_qwen2_audio_to_onnx,
    map_mmproj_qwen3_audio_to_onnx,
    map_mmproj_qwen3_speaker_to_onnx,
    map_mmproj_qwen3_vision_to_onnx,
)
from mobius.models.gguf_qwen_glm_projector import (
    GGUFGlm4VVisionProjector,
    GGUFGlmOcrVisionProjector,
    GGUFQwen3AudioProjector,
    GGUFQwen3TTSSpeakerProjector,
    GGUFQwen3VLProjector,
    GGUFQwen25OmniVisionProjector,
)

_FLOAT_DTYPES = {
    ir.DataType.FLOAT,
    ir.DataType.FLOAT16,
    ir.DataType.BFLOAT16,
}

QWEN_GLM_PROCESSOR_ABIS: dict[str, dict[str, str | int]] = {
    "qwen2a": {
        "sample_rate": 16_000,
        "input_features": "float32[1,128,3000]",
        "output": "750 audio rows per 30-second chunk",
        "preprocessing": "Whisper log10 mel; fixed 3000-frame chunks",
        "empty_media": "do not invoke the audio graph",
    },
    "qwen3a": {
        "sample_rate": 16_000,
        "input_features": "float32[1,128,frames_multiple_of_100]",
        "output": "13 * (frames / 100) audio rows",
        "preprocessing": "Qwen3 Whisper mel; <=800-frame windows padded to 100",
        "empty_media": "do not invoke the audio graph",
    },
    "glma": {
        "sample_rate": 16_000,
        "input_features": "float32[1,num_mel_bins,3000]",
        "output": "frames / (2 * stack_factor) + BOI/EOI rows",
        "preprocessing": "legacy Whisper log10 mel; fixed 3000-frame chunks",
        "empty_media": "do not invoke the audio graph",
    },
    "qwen3tts_spkenc": {
        "sample_rate": 24_000,
        "mel_features": "float32[frames,128]",
        "output": "one speaker-conditioning row",
        "preprocessing": "natural-log magnitude mel; n_fft=1024, hop=256",
        "empty_media": "speaker conditioning omitted; do not invoke graph",
    },
    "qwen3vl_merger": {
        "pixel_values": "float32[total_patches,3*2*16*16]",
        "image_grid_thw": "int64[num_media,3]",
        "output": "sum(T*H*W/4) rows, width=(1+deepstack_layers)*text_hidden",
        "ordering": "batch-major media; merge-block-major 2x2 patches",
        "empty_media": "do not invoke the vision graph",
    },
    "glm4v": {
        "pixel_values": "float32[total_patches,3*2*patch_size*patch_size]",
        "image_grid_thw": "int64[num_media,3]",
        "output": "sum(T*H*W/4) text-width rows",
        "ordering": "batch-major media; merge-block-major 2x2 patches",
        "empty_media": "do not invoke the vision graph",
    },
    "qwen2.5o": {
        "vision": "qwen2.5vl_merger component ABI",
        "audio": "qwen2a component ABI",
        "output": "separate vision_encoder and audio_encoder components",
        "empty_media": "invoke only components for present modalities",
    },
}


def qwen3vl_decoder_mrope_positions(
    *,
    merged_height: int,
    merged_width: int,
    start_position: int,
) -> tuple[np.ndarray, int]:
    """Return llama.cpp's four decoder-position sections for one image grid.

    Section order is ``t, y, x, z``. Unlike the vision tower's internal
    merge-block rotary coordinates, these positions index the *merged* image
    rows inserted into the text sequence. The next text position advances by
    ``max(merged_height, merged_width)`` rather than by the image token count.
    """
    if merged_height <= 0 or merged_width <= 0 or start_position < 0:
        raise ValueError("Qwen3-VL merged dimensions must be positive")
    token_count = merged_height * merged_width
    indices: np.ndarray = np.arange(token_count, dtype=np.int64)
    positions = np.stack(
        (
            np.full(token_count, start_position, dtype=np.int64),
            start_position + indices // merged_width,
            start_position + indices % merged_width,
            np.zeros(token_count, dtype=np.int64),
        ),
        axis=0,
    )
    return positions, start_position + max(merged_height, merged_width)


def _base_config(
    *,
    model_type: str,
    hidden_size: int,
    dtype: ir.DataType,
    vision: VisionConfig | None = None,
    audio: AudioConfig | None = None,
    tts: TTSConfig | None = None,
) -> ArchitectureConfig:
    if dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{model_type} sidecar dtype must be floating point")
    return ArchitectureConfig(
        model_type=model_type,
        vocab_size=1,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden_size,
        max_position_embeddings=1,
        hidden_act="silu",
        dtype=dtype,
        vision=vision,
        audio=audio,
        tts=tts,
    )


def _shape(model: Any, name: str) -> tuple[int, ...]:
    return tuple(int(dim) for dim in model.get_tensor_shape(name))


def _expect_shape(model: Any, name: str, expected: tuple[int, ...]) -> None:
    actual = _shape(model, name)
    if actual != expected:
        raise ValueError(f"mmproj tensor {name!r} has shape {actual}, expected {expected}.")


def read_qwen3vl_sidecar_config(model: Any, *, dtype: ir.DataType) -> ArchitectureConfig:
    """Read the Qwen3-VL tower and DeepStack merger configuration."""
    metadata = model.metadata
    hidden = int(metadata["clip.vision.embedding_length"])
    projection = int(metadata["clip.vision.projection_dim"])
    patch_size = int(metadata["clip.vision.patch_size"])
    merge = int(metadata.get("clip.vision.spatial_merge_size", 2))
    if merge != 2:
        raise ValueError(
            "qwen3vl_merger requires clip.vision.spatial_merge_size=2 because "
            "the final GGUF merger hardcodes four patches per output token."
        )
    position_rows = _shape(model, "v.position_embd.weight")[0]
    position_grid = math.isqrt(position_rows)
    if position_grid * position_grid != position_rows:
        raise ValueError("Qwen3-VL position embeddings must form a square grid")

    deepstack_layers = sorted(
        {
            int(match.group(1))
            for name in model.tensor_names
            if (
                match := re.match(
                    r"^v\.deepstack\.(\d+)\.(?:norm|fc1|fc2)\.(?:weight|bias)$",
                    name,
                )
            )
        }
    )
    declared = metadata.get("clip.vision.is_deepstack_layers")
    if declared is not None:
        if not isinstance(declared, list) or len(declared) != int(
            metadata["clip.vision.block_count"]
        ):
            raise ValueError(
                "clip.vision.is_deepstack_layers must be one bool per vision block"
            )
        declared_layers = [index for index, enabled in enumerate(declared) if bool(enabled)]
        if declared_layers != deepstack_layers:
            raise ValueError(
                "Qwen3-VL DeepStack metadata and tensor layer indices disagree: "
                f"{declared_layers} != {deepstack_layers}."
            )

    vision = VisionConfig(
        hidden_size=hidden,
        intermediate_size=int(metadata["clip.vision.feed_forward_length"]),
        num_hidden_layers=int(metadata["clip.vision.block_count"]),
        num_attention_heads=int(metadata["clip.vision.attention.head_count"]),
        image_size=int(metadata["clip.vision.image_size"]),
        patch_size=patch_size,
        norm_eps=float(metadata["clip.vision.attention.layer_norm_epsilon"]),
        in_channels=_shape(model, "v.patch_embd.weight")[1],
        out_hidden_size=projection,
        spatial_merge_size=merge,
        temporal_patch_size=2,
        num_position_embeddings=position_rows,
        deepstack_visual_indexes=deepstack_layers,
        hidden_act=(
            "silu" if bool(metadata.get("clip.use_silu", False)) else "gelu_pytorch_tanh"
        ),
    )
    return _base_config(
        model_type="gguf_qwen3vl_merger",
        hidden_size=projection,
        dtype=dtype,
        vision=vision,
    )


def read_glm4v_sidecar_config(model: Any, *, dtype: ir.DataType) -> ArchitectureConfig:
    """Read GLM4V's packed ViT, learned downsampler, and gated projector."""
    metadata = model.metadata
    hidden = int(metadata["clip.vision.embedding_length"])
    projection = int(metadata["clip.vision.projection_dim"])
    merge = int(metadata.get("clip.vision.spatial_merge_size", 2))
    if merge != 2:
        raise ValueError("glm4v requires clip.vision.spatial_merge_size=2")
    position_rows = (
        _shape(model, "v.position_embd.weight")[0]
        if "v.position_embd.weight" in model.tensor_names
        else None
    )
    projector_intermediate = _shape(model, "mm.up.weight")[0]
    block_intermediate = _shape(model, "v.blk.0.ffn_up.weight")[0]
    vision = VisionConfig(
        hidden_size=hidden,
        intermediate_size=block_intermediate,
        num_hidden_layers=int(metadata["clip.vision.block_count"]),
        num_attention_heads=int(metadata["clip.vision.attention.head_count"]),
        image_size=int(metadata["clip.vision.image_size"]),
        patch_size=int(metadata["clip.vision.patch_size"]),
        norm_eps=float(metadata["clip.vision.attention.layer_norm_epsilon"]),
        in_channels=_shape(model, "v.patch_embd.weight")[1],
        out_hidden_size=projection,
        spatial_merge_size=merge,
        temporal_patch_size=2,
        num_position_embeddings=position_rows,
        projector_intermediate_size=projector_intermediate,
        hidden_act="silu" if bool(metadata.get("clip.use_silu", False)) else "gelu",
    )
    return _base_config(
        model_type="gguf_glm4v",
        hidden_size=projection,
        dtype=dtype,
        vision=vision,
    )


def read_qwen25o_vision_config(
    model: Any,
    *,
    dtype: ir.DataType,
) -> ArchitectureConfig:
    """Resolve qwen2.5o's vision context as an exact Qwen2.5-VL merger."""
    metadata = model.metadata
    hidden = int(metadata["clip.vision.embedding_length"])
    layers = int(metadata["clip.vision.block_count"])
    intermediate = _shape(model, "v.blk.0.ffn_up.weight")[0]
    pattern = int(metadata["clip.vision.n_wa_pattern"])
    if pattern <= 0:
        raise ValueError("clip.vision.n_wa_pattern must be positive")
    vision = VisionConfig(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=int(metadata["clip.vision.attention.head_count"]),
        image_size=int(metadata["clip.vision.image_size"]),
        patch_size=int(metadata["clip.vision.patch_size"]),
        norm_eps=float(metadata["clip.vision.attention.layer_norm_epsilon"]),
        in_channels=_shape(model, "v.patch_embd.weight")[1],
        out_hidden_size=int(metadata["clip.vision.projection_dim"]),
        spatial_merge_size=2,
        temporal_patch_size=2,
        fullatt_block_indexes=list(range(pattern - 1, layers, pattern)),
        window_size=112,
        hidden_act="silu",
    )
    return _base_config(
        model_type="gguf_qwen25o_vision",
        hidden_size=int(vision.out_hidden_size or 0),
        dtype=dtype,
        vision=vision,
    )


def _audio_config(model: Any) -> AudioConfig:
    metadata = model.metadata
    return AudioConfig(
        d_model=int(metadata["clip.audio.embedding_length"]),
        encoder_layers=int(metadata["clip.audio.block_count"]),
        encoder_attention_heads=int(metadata["clip.audio.attention.head_count"]),
        encoder_ffn_dim=int(metadata["clip.audio.feed_forward_length"]),
        encoder_layer_norm_eps=float(metadata["clip.audio.attention.layer_norm_epsilon"]),
        num_mel_bins=int(metadata["clip.audio.num_mel_bins"]),
        max_source_positions=_shape(model, "a.position_embd.weight")[0],
        output_dim=int(metadata["clip.audio.projection_dim"]),
        activation_function="gelu",
    )


def read_qwen2a_sidecar_config(model: Any, *, dtype: ir.DataType) -> ArchitectureConfig:
    audio = _audio_config(model)
    return _base_config(
        model_type="gguf_qwen2a",
        hidden_size=int(audio.output_dim or 0),
        dtype=dtype,
        audio=audio,
    )


def read_qwen3a_sidecar_config(model: Any, *, dtype: ir.DataType) -> ArchitectureConfig:
    audio = dataclasses.replace(
        _audio_config(model),
        downsample_hidden_size=_shape(model, "a.conv2d.1.weight")[0],
        n_window=50,
        n_window_infer=int(model.metadata.get("clip.audio.projector.window_size", 800)),
    )
    return _base_config(
        model_type="gguf_qwen3a",
        hidden_size=int(audio.output_dim or 0),
        dtype=dtype,
        audio=audio,
    )


def read_glma_sidecar_config(model: Any, *, dtype: ir.DataType) -> ArchitectureConfig:
    audio = _audio_config(model)
    return _base_config(
        model_type="gguf_glma",
        hidden_size=int(audio.output_dim or 0),
        dtype=dtype,
        audio=audio,
    )


def read_qwen3tts_speaker_config(
    model: Any,
    *,
    dtype: ir.DataType,
) -> ArchitectureConfig:
    """Infer the non-metadata ECAPA dimensions from the exact tensor closure."""
    stem = _shape(model, "a.conv1d.0.weight")
    mfa = _shape(model, "a.conv_out.weight")
    asp_tdnn = _shape(model, "a.asp_tdnn.weight")
    se = _shape(model, "a.blk.1.se_conv1.weight")
    final = _shape(model, "mm.a.fc.weight")
    speaker = SpeakerEncoderConfig(
        mel_dim=stem[1],
        enc_dim=final[0],
        enc_channels=[stem[0], stem[0], stem[0], stem[0], mfa[0]],
        enc_kernel_sizes=[stem[2], 3, 3, 3, mfa[2]],
        enc_dilations=[1, 2, 3, 4, 1],
        enc_attention_channels=asp_tdnn[0],
        enc_res2net_scale=8,
        enc_se_channels=se[0],
    )
    return _base_config(
        model_type="gguf_qwen3tts_spkenc",
        hidden_size=speaker.enc_dim,
        # llama.cpp forces speaker TDNN matrix products to float32.
        dtype=ir.DataType.FLOAT,
        tts=TTSConfig(speaker_encoder=speaker),
    )


def create_qwen_glm_projector(
    projector_type: str,
    model: Any,
    *,
    dtype: ir.DataType,
) -> tuple[nn.Module, ArchitectureConfig]:
    """Create one exact standalone component and its graph-only config."""
    component: nn.Module
    if projector_type == "glm4v":
        config = read_glm4v_sidecar_config(model, dtype=dtype)
        if "v.blk.0.attn_q_norm.weight" in model.tensor_names:
            return GGUFGlmOcrVisionProjector(config), config
        return GGUFGlm4VVisionProjector(config), config
    if projector_type == "qwen3vl_merger":
        config = read_qwen3vl_sidecar_config(model, dtype=dtype)
        return GGUFQwen3VLProjector(config), config
    if projector_type == "qwen2a":
        config = read_qwen2a_sidecar_config(model, dtype=dtype)
        audio = config.audio
        assert audio is not None
        component = GGUFQwen2AudioProjector(
            num_mel_bins=int(audio.num_mel_bins or 0),
            hidden_size=int(audio.d_model or 0),
            intermediate_size=int(audio.encoder_ffn_dim or 0),
            num_hidden_layers=int(audio.encoder_layers or 0),
            num_attention_heads=int(audio.encoder_attention_heads or 0),
            max_source_positions=int(audio.max_source_positions or 0),
            output_size=int(audio.output_dim or 0),
            norm_eps=float(audio.encoder_layer_norm_eps or 1e-5),
        )
        return component, config
    if projector_type == "qwen3a":
        config = read_qwen3a_sidecar_config(model, dtype=dtype)
        return GGUFQwen3AudioProjector(config), config
    if projector_type == "glma":
        config = read_glma_sidecar_config(model, dtype=dtype)
        audio = config.audio
        assert audio is not None
        stack_factor = int(model.metadata["clip.audio.projector.stack_factor"])
        component = GGUFLegacyGlmAudioProjector(
            num_mel_bins=int(audio.num_mel_bins or 0),
            hidden_size=int(audio.d_model or 0),
            intermediate_size=int(audio.encoder_ffn_dim or 0),
            num_hidden_layers=int(audio.encoder_layers or 0),
            num_attention_heads=int(audio.encoder_attention_heads or 0),
            max_source_positions=int(audio.max_source_positions or 0),
            stack_factor=stack_factor,
            projector_intermediate_size=_shape(model, "mm.a.mlp.1.weight")[0],
            output_size=int(audio.output_dim or 0),
            norm_eps=float(audio.encoder_layer_norm_eps or 1e-5),
            activation="silu" if bool(model.metadata.get("clip.use_silu")) else "gelu",
        )
        return component, config
    if projector_type == "qwen3tts_spkenc":
        config = read_qwen3tts_speaker_config(model, dtype=dtype)
        return GGUFQwen3TTSSpeakerProjector(config), config
    raise ValueError(f"Unsupported Qwen/GLM projector type {projector_type!r}")


def _mapped_state(
    model: Any,
    projector_type: str,
    *,
    component_prefix: str,
    deepstack_layers: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    mappers = {
        "glm4v": map_mmproj_glm4v_vision_to_onnx,
        "qwen2a": map_mmproj_qwen2_audio_to_onnx,
        "qwen3a": map_mmproj_qwen3_audio_to_onnx,
        "glma": map_mmproj_glma_audio_to_onnx,
        "qwen3tts_spkenc": map_mmproj_qwen3_speaker_to_onnx,
    }
    mapper = mappers.get(projector_type)
    state: dict[str, torch.Tensor] = {}
    fused_sources: set[str] = set()

    if projector_type in {"glm4v", "qwen3vl_merger"}:
        patch_names = ("v.patch_embd.weight", "v.patch_embd.weight.1")
        patch_halves = [
            np.asarray(model.get_tensor(name), dtype=np.float32) for name in patch_names
        ]
        state[f"{component_prefix}.visual.patch_embed.proj.weight"] = torch.from_numpy(
            np.stack(patch_halves, axis=2).copy()
        )
        fused_sources.update(patch_names)

    if projector_type == "qwen3vl_merger":
        mapper = lambda name: map_mmproj_qwen3_vision_to_onnx(  # noqa: E731
            name,
            deepstack_layers=deepstack_layers,
        )

    for name in model.tensor_names:
        if name in fused_sources or name.startswith("a.gen."):
            continue
        if mapper is None:
            continue
        mapped = mapper(name)
        if mapped is None:
            continue
        values = np.asarray(model.get_tensor(name), dtype=np.float32)
        target = f"{component_prefix}.{mapped}"
        initializer_shape = values.shape
        if name.endswith(".bias") and values.ndim > 1:
            values = values.reshape(-1)
        if (
            projector_type == "qwen3tts_spkenc"
            and name.endswith(".weight")
            and values.ndim == 2
            and name.startswith(("a.conv", "a.asp", "mm.a.fc"))
        ):
            values = values[:, :, None]
        if not values.flags.c_contiguous:
            values = values.copy()
        state[target] = torch.from_numpy(values)
        del initializer_shape
    return state


def qwen_glm_projector_state(
    model: Any,
    projector_type: str,
    *,
    component_prefix: str,
) -> dict[str, torch.Tensor]:
    """Load and transform the selected sidecar component into ONNX names."""
    deepstack_layers: tuple[int, ...] = ()
    if projector_type == "qwen3vl_merger":
        vision = read_qwen3vl_sidecar_config(
            model,
            dtype=ir.DataType.FLOAT,
        ).vision
        assert vision is not None
        deepstack_layers = tuple(vision.deepstack_visual_indexes or ())
    return _mapped_state(
        model,
        projector_type,
        component_prefix=component_prefix,
        deepstack_layers=deepstack_layers,
    )


def qwen25o_vision_state(
    model: Any,
    *,
    component_prefix: str,
) -> dict[str, torch.Tensor]:
    """Transform the legacy alias's vision tensors through Qwen2.5-VL rules."""
    from mobius.integrations.gguf._mmproj import _mmproj_qwen_vision_to_hf

    transformed = _mmproj_qwen_vision_to_hf(model, "qwen2.5vl_merger")
    state = {}
    for name, value in transformed.items():
        name = name.replace(".merger.mlp.0.", ".merger.mlp_0.")
        name = name.replace(".merger.mlp.2.", ".merger.mlp_2.")
        state[f"{component_prefix}.{name}"] = value
    return state


def validate_qwen_glm_projector_shapes(model: Any, projector_type: str) -> None:
    """Reject any shape contract that the exact route graph cannot consume."""
    metadata = model.metadata
    if projector_type == "qwen2.5o":
        # The legacy selector has no graph of its own. Its audio context is
        # exactly qwen2a, while its vision context is qwen2.5vl_merger.
        validate_qwen_glm_projector_shapes(model, "qwen2a")
        hidden = int(metadata["clip.vision.embedding_length"])
        intermediate = _shape(model, "v.blk.0.ffn_up.weight")[0]
        projection = int(metadata["clip.vision.projection_dim"])
        patch = int(metadata["clip.vision.patch_size"])
        layers = int(metadata["clip.vision.block_count"])
        heads = int(metadata["clip.vision.attention.head_count"])
        if hidden <= 0 or heads <= 0 or hidden % heads:
            raise ValueError("Qwen2.5-Omni vision hidden size must divide by heads")
        for name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
            _expect_shape(model, name, (hidden, 3, patch, patch))
        merged = hidden * 4
        _expect_shape(model, "v.post_ln.weight", (hidden,))
        _expect_shape(model, "mm.0.weight", (merged, merged))
        _expect_shape(model, "mm.0.bias", (merged,))
        _expect_shape(model, "mm.2.weight", (projection, merged))
        _expect_shape(model, "mm.2.bias", (projection,))
        for layer in range(layers):
            prefix = f"v.blk.{layer}."
            for norm in ("ln1", "ln2"):
                _expect_shape(model, prefix + norm + ".weight", (hidden,))
            for stem in ("attn_q", "attn_k", "attn_v", "attn_out"):
                _expect_shape(model, prefix + stem + ".weight", (hidden, hidden))
                _expect_shape(model, prefix + stem + ".bias", (hidden,))
            for stem in ("ffn_gate", "ffn_up"):
                _expect_shape(model, prefix + stem + ".weight", (intermediate, hidden))
                _expect_shape(model, prefix + stem + ".bias", (intermediate,))
            _expect_shape(model, prefix + "ffn_down.weight", (hidden, intermediate))
            _expect_shape(model, prefix + "ffn_down.bias", (hidden,))
        return

    if projector_type in {"glm4v", "qwen3vl_merger"}:
        hidden = int(metadata["clip.vision.embedding_length"])
        intermediate = int(metadata["clip.vision.feed_forward_length"])
        projection = int(metadata["clip.vision.projection_dim"])
        patch = int(metadata["clip.vision.patch_size"])
        layers = int(metadata["clip.vision.block_count"])
        heads = int(metadata["clip.vision.attention.head_count"])
        if hidden <= 0 or heads <= 0 or hidden % heads:
            raise ValueError("vision hidden size must be divisible by its head count")
        if projector_type == "glm4v":
            intermediate = _shape(model, "v.blk.0.ffn_up.weight")[0]
        for name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
            _expect_shape(model, name, (hidden, 3, patch, patch))
        _expect_shape(model, "v.patch_embd.bias", (hidden,))

        if projector_type == "qwen3vl_merger":
            position = _shape(model, "v.position_embd.weight")
            if len(position) != 2 or position[1] != hidden:
                raise ValueError("Qwen3-VL position embeddings must be [positions, hidden]")
            merged = hidden * 4
            _expect_shape(model, "v.post_ln.weight", (hidden,))
            _expect_shape(model, "v.post_ln.bias", (hidden,))
            _expect_shape(model, "mm.0.weight", (merged, merged))
            _expect_shape(model, "mm.0.bias", (merged,))
            _expect_shape(model, "mm.2.weight", (projection, merged))
            _expect_shape(model, "mm.2.bias", (projection,))
            for layer in range(layers):
                prefix = f"v.blk.{layer}."
                for norm in ("ln1", "ln2"):
                    _expect_shape(model, prefix + norm + ".weight", (hidden,))
                    _expect_shape(model, prefix + norm + ".bias", (hidden,))
                _expect_shape(model, prefix + "attn_qkv.weight", (3 * hidden, hidden))
                _expect_shape(model, prefix + "attn_qkv.bias", (3 * hidden,))
                _expect_shape(model, prefix + "attn_out.weight", (hidden, hidden))
                _expect_shape(model, prefix + "attn_out.bias", (hidden,))
                _expect_shape(model, prefix + "ffn_up.weight", (intermediate, hidden))
                _expect_shape(model, prefix + "ffn_up.bias", (intermediate,))
                _expect_shape(model, prefix + "ffn_down.weight", (hidden, intermediate))
                _expect_shape(model, prefix + "ffn_down.bias", (hidden,))
            config = read_qwen3vl_sidecar_config(model, dtype=ir.DataType.FLOAT)
            deepstack = config.vision.deepstack_visual_indexes if config.vision else []
            for layer in deepstack or []:
                prefix = f"v.deepstack.{layer}."
                _expect_shape(model, prefix + "norm.weight", (merged,))
                _expect_shape(model, prefix + "norm.bias", (merged,))
                _expect_shape(model, prefix + "fc1.weight", (merged, merged))
                _expect_shape(model, prefix + "fc1.bias", (merged,))
                _expect_shape(model, prefix + "fc2.weight", (projection, merged))
                _expect_shape(model, prefix + "fc2.bias", (projection,))
            return

        is_glm_ocr = "v.blk.0.attn_q_norm.weight" in model.tensor_names
        if is_glm_ocr and (
            "v.norm_embd.weight" in model.tensor_names
            or "v.position_embd.weight" in model.tensor_names
        ):
            raise ValueError(
                "The GLM-OCR qk-norm variant cannot carry GLM4V post-conv or "
                "learned-position tensors."
            )
        if not is_glm_ocr and "v.norm_embd.weight" not in model.tensor_names:
            raise ValueError("The GLM4V variant requires v.norm_embd.weight")
        if "v.norm_embd.weight" in model.tensor_names:
            _expect_shape(model, "v.norm_embd.weight", (hidden,))
        if "v.position_embd.weight" in model.tensor_names:
            position = _shape(model, "v.position_embd.weight")
            if len(position) != 2 or position[1] != hidden:
                raise ValueError("GLM4V position embeddings must be [positions, hidden]")
        _expect_shape(model, "v.post_ln.weight", (hidden,))
        _expect_shape(model, "mm.patch_merger.weight", (projection, hidden, 2, 2))
        _expect_shape(model, "mm.patch_merger.bias", (projection,))
        _expect_shape(model, "mm.model.fc.weight", (projection, projection))
        _expect_shape(model, "mm.post_norm.weight", (projection,))
        _expect_shape(model, "mm.post_norm.bias", (projection,))
        projector_intermediate = _shape(model, "mm.up.weight")[0]
        _expect_shape(model, "mm.up.weight", (projector_intermediate, projection))
        _expect_shape(model, "mm.gate.weight", (projector_intermediate, projection))
        _expect_shape(model, "mm.down.weight", (projection, projector_intermediate))
        for layer in range(layers):
            prefix = f"v.blk.{layer}."
            for norm in ("ln1", "ln2"):
                _expect_shape(model, prefix + norm + ".weight", (hidden,))
            _expect_shape(model, prefix + "attn_qkv.weight", (3 * hidden, hidden))
            _expect_shape(model, prefix + "attn_out.weight", (hidden, hidden))
            _expect_shape(model, prefix + "ffn_gate.weight", (intermediate, hidden))
            _expect_shape(model, prefix + "ffn_up.weight", (intermediate, hidden))
            _expect_shape(model, prefix + "ffn_down.weight", (hidden, intermediate))
            if is_glm_ocr:
                _expect_shape(model, prefix + "attn_qkv.bias", (3 * hidden,))
                _expect_shape(model, prefix + "attn_out.bias", (hidden,))
                _expect_shape(
                    model,
                    prefix + "attn_q_norm.weight",
                    (hidden // heads,),
                )
                _expect_shape(
                    model,
                    prefix + "attn_k_norm.weight",
                    (hidden // heads,),
                )
                _expect_shape(model, prefix + "ffn_gate.bias", (intermediate,))
                _expect_shape(model, prefix + "ffn_up.bias", (intermediate,))
                _expect_shape(model, prefix + "ffn_down.bias", (hidden,))
        return

    if projector_type in {"qwen2a", "qwen3a", "glma"}:
        hidden = int(metadata["clip.audio.embedding_length"])
        intermediate = int(metadata["clip.audio.feed_forward_length"])
        projection = int(metadata["clip.audio.projection_dim"])
        layers = int(metadata["clip.audio.block_count"])
        heads = int(metadata["clip.audio.attention.head_count"])
        mel = int(metadata["clip.audio.num_mel_bins"])
        if hidden <= 0 or heads <= 0 or hidden % heads:
            raise ValueError("audio hidden size must be divisible by its head count")
        position = _shape(model, "a.position_embd.weight")
        if len(position) != 2 or position[1] != hidden:
            raise ValueError("audio position embeddings must be [positions, hidden]")
        for layer in range(layers):
            prefix = f"a.blk.{layer}."
            for norm in ("ln1", "ln2"):
                _expect_shape(model, prefix + norm + ".weight", (hidden,))
                _expect_shape(model, prefix + norm + ".bias", (hidden,))
            for stem in ("attn_q", "attn_k", "attn_v", "attn_out"):
                _expect_shape(model, prefix + stem + ".weight", (hidden, hidden))
            for stem in ("attn_q", "attn_v", "attn_out"):
                _expect_shape(model, prefix + stem + ".bias", (hidden,))
            if projector_type == "qwen3a":
                _expect_shape(model, prefix + "attn_k.bias", (hidden,))
            _expect_shape(model, prefix + "ffn_up.weight", (intermediate, hidden))
            _expect_shape(model, prefix + "ffn_up.bias", (intermediate,))
            _expect_shape(model, prefix + "ffn_down.weight", (hidden, intermediate))
            _expect_shape(model, prefix + "ffn_down.bias", (hidden,))
        _expect_shape(model, "a.post_ln.weight", (hidden,))
        _expect_shape(model, "a.post_ln.bias", (hidden,))

        if projector_type in {"qwen2a", "glma"}:
            _expect_shape(model, "a.conv1d.1.weight", (hidden, mel, 3))
            _expect_shape(model, "a.conv1d.2.weight", (hidden, hidden, 3))
            for index in (1, 2):
                bias = _shape(model, f"a.conv1d.{index}.bias")
                if math.prod(bias) != hidden:
                    raise ValueError(f"a.conv1d.{index}.bias must contain {hidden} values")
        if projector_type == "qwen2a":
            _expect_shape(model, "mm.a.fc.weight", (projection, hidden))
            _expect_shape(model, "mm.a.fc.bias", (projection,))
            return
        if projector_type == "glma":
            stack = int(metadata["clip.audio.projector.stack_factor"])
            projector_intermediate = _shape(model, "mm.a.mlp.1.weight")[0]
            _expect_shape(
                model,
                "mm.a.mlp.1.weight",
                (projector_intermediate, hidden * stack),
            )
            _expect_shape(model, "mm.a.mlp.1.bias", (projector_intermediate,))
            _expect_shape(
                model,
                "mm.a.mlp.2.weight",
                (projection, projector_intermediate),
            )
            _expect_shape(model, "mm.a.mlp.2.bias", (projection,))
            _expect_shape(model, "mm.a.norm_pre.weight", (hidden,))
            _expect_shape(model, "mm.a.norm_pre.bias", (hidden,))
            _expect_shape(model, "v.boi", (projection,))
            _expect_shape(model, "v.eoi", (projection,))
            return

        downsample = _shape(model, "a.conv2d.1.weight")[0]
        _expect_shape(model, "a.conv2d.1.weight", (downsample, 1, 3, 3))
        _expect_shape(model, "a.conv2d.2.weight", (downsample, downsample, 3, 3))
        _expect_shape(model, "a.conv2d.3.weight", (downsample, downsample, 3, 3))
        for index in (1, 2, 3):
            bias = _shape(model, f"a.conv2d.{index}.bias")
            if math.prod(bias) != downsample:
                raise ValueError(f"a.conv2d.{index}.bias must contain {downsample} values")
        mel_after = (mel + 1) // 2
        mel_after = (mel_after + 1) // 2
        mel_after = (mel_after + 1) // 2
        _expect_shape(model, "a.conv_out.weight", (hidden, downsample * mel_after))
        _expect_shape(model, "mm.a.mlp.1.weight", (hidden, hidden))
        _expect_shape(model, "mm.a.mlp.1.bias", (hidden,))
        _expect_shape(model, "mm.a.mlp.2.weight", (projection, hidden))
        _expect_shape(model, "mm.a.mlp.2.bias", (projection,))
        return

    if projector_type == "qwen3tts_spkenc":
        if int(metadata["clip.audio.block_count"]) != 3:
            raise ValueError("qwen3tts_spkenc requires exactly three SE-Res2Net blocks")
        stem_shape = _shape(model, "a.conv1d.0.weight")
        if len(stem_shape) != 3 or stem_shape[2] != 5:
            raise ValueError("Qwen3-TTS speaker stem must be [channels, mel_bins, 5]")
        channels = stem_shape[0]
        _expect_shape(model, "a.conv1d.0.bias", (channels,))
        for block in range(1, 4):
            prefix = f"a.blk.{block}."
            _expect_shape(model, prefix + "conv_pw1.weight", (channels, channels, 1))
            _expect_shape(model, prefix + "conv_pw1.bias", (channels,))
            _expect_shape(model, prefix + "conv_pw2.weight", (channels, channels, 1))
            _expect_shape(model, prefix + "conv_pw2.bias", (channels,))
            se_channels = _shape(model, prefix + "se_conv1.weight")[0]
            _expect_shape(
                model,
                prefix + "se_conv1.weight",
                (se_channels, channels, 1),
            )
            _expect_shape(model, prefix + "se_conv1.bias", (se_channels,))
            _expect_shape(
                model,
                prefix + "se_conv2.weight",
                (channels, se_channels, 1),
            )
            _expect_shape(model, prefix + "se_conv2.bias", (channels,))
            for branch in range(7):
                _expect_shape(
                    model,
                    prefix + f"res2.{branch}.weight",
                    (channels // 8, channels // 8, 3),
                )
                _expect_shape(
                    model,
                    prefix + f"res2.{branch}.bias",
                    (channels // 8,),
                )
        mfa = _shape(model, "a.conv_out.weight")
        if len(mfa) != 3 or mfa[1:] != (3 * channels, 1):
            raise ValueError("Qwen3-TTS MFA weight must consume all three block outputs")
        _expect_shape(model, "a.conv_out.bias", (mfa[0],))
        asp_tdnn = _shape(model, "a.asp_tdnn.weight")
        if len(asp_tdnn) != 3 or asp_tdnn[1:] != (3 * mfa[0], 1):
            raise ValueError("Qwen3-TTS ASP TDNN must consume hidden/mean/std")
        _expect_shape(model, "a.asp_tdnn.bias", (asp_tdnn[0],))
        _expect_shape(model, "a.asp_attn.weight", (mfa[0], asp_tdnn[0], 1))
        _expect_shape(model, "a.asp_attn.bias", (mfa[0],))
        final = _shape(model, "mm.a.fc.weight")
        if len(final) != 3 or final[1:] != (2 * mfa[0], 1):
            raise ValueError("Qwen3-TTS speaker FC must consume weighted mean and std")
        _expect_shape(model, "mm.a.fc.bias", (final[0],))
        return

    raise ValueError(f"No shape validator for projector type {projector_type!r}")


def build_qwen_glm_projector_package(
    model: Any,
    *,
    resolved_path: str | Path,
    projector_type: str,
    dtype: str | None,
    execution_provider: str,
) -> ModelPackage:
    """Build and load the exact standalone component set for this cohort."""
    from mobius._builder import build_from_module, resolve_dtype
    from mobius.tasks import (
        GGUFAudioProjectorModel,
        GGUFAudioProjectorTask,
        GGUFSpeakerProjectorModel,
        GGUFSpeakerProjectorTask,
        GGUFVisionProjectorModel,
        GGUFVisionProjectorTask,
    )

    resolved_dtype = resolve_dtype(dtype) or ir.DataType.FLOAT
    if projector_type == "qwen3tts_spkenc" and resolved_dtype != ir.DataType.FLOAT:
        raise ValueError(
            "qwen3tts_spkenc executes its TDNN matrix products in float32; "
            "reduced-precision graph conversion is not exact."
        )

    if projector_type == "qwen2.5o":
        vision_config = read_qwen25o_vision_config(model, dtype=resolved_dtype)
        vision_component = GGUFQwen25OmniVisionProjector(vision_config)
        vision_package = build_from_module(
            GGUFVisionProjectorModel(vision_component),
            vision_config,
            task=GGUFVisionProjectorTask(),
            execution_provider=execution_provider,
        )
        vision_package.apply_weights(
            qwen25o_vision_state(model, component_prefix="vision_encoder")
        )

        audio_config = read_qwen2a_sidecar_config(model, dtype=resolved_dtype)
        audio_component, _ = create_qwen_glm_projector(
            "qwen2a",
            model,
            dtype=resolved_dtype,
        )
        audio_package = build_from_module(
            GGUFAudioProjectorModel(audio_component),
            audio_config,
            task=GGUFAudioProjectorTask(),
            execution_provider=execution_provider,
        )
        audio_package.apply_weights(
            qwen_glm_projector_state(
                model,
                "qwen2a",
                component_prefix="audio_encoder",
            )
        )
        combined_config = dataclasses.replace(
            vision_config,
            audio=audio_config.audio,
        )
        package = ModelPackage(
            {
                "vision_encoder": vision_package["vision_encoder"],
                "audio_encoder": audio_package["audio_encoder"],
            },
            config=combined_config,
        )
    else:
        component, config = create_qwen_glm_projector(
            projector_type,
            model,
            dtype=resolved_dtype,
        )
        if projector_type in {"glm4v", "qwen3vl_merger"}:
            package = build_from_module(
                GGUFVisionProjectorModel(component),
                config,
                task=GGUFVisionProjectorTask(),
                execution_provider=execution_provider,
            )
            component_prefix = "vision_encoder"
        elif projector_type == "qwen3tts_spkenc":
            package = build_from_module(
                GGUFSpeakerProjectorModel(component),
                config,
                task=GGUFSpeakerProjectorTask(),
                execution_provider=execution_provider,
            )
            component_prefix = "speaker_encoder"
        else:
            package = build_from_module(
                GGUFAudioProjectorModel(component),
                config,
                task=GGUFAudioProjectorTask(),
                execution_provider=execution_provider,
            )
            component_prefix = "audio_encoder"
        package.apply_weights(
            qwen_glm_projector_state(
                model,
                projector_type,
                component_prefix=component_prefix,
            )
        )

    package.gguf_source_path = str(Path(resolved_path).resolve())  # type: ignore[attr-defined]
    package.gguf_projector_type = projector_type  # type: ignore[attr-defined]
    package.gguf_processor_abi = QWEN_GLM_PROCESSOR_ABIS[  # type: ignore[attr-defined]
        projector_type
    ]
    return package
