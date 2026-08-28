# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone GGUF builders for OCR and document projector sidecars."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn

from mobius._builder import build_from_module, resolve_dtype
from mobius._configs import ArchitectureConfig
from mobius._configs._sub_configs import VisionConfig
from mobius.components import (
    DeepSeekOCR2FullImageEncoder,
    DeepSeekOCRFullImageEncoder,
    Dots3NoteAudioEncoder,
    DotsVisionEncoder,
    Granite4VisionEncoder,
    LightOnOCRVisionEncoder,
    PaddleOCRVisionEncoder,
    YouTuVLVisionEncoder,
)
from mobius.integrations.gguf._mmproj_mapping import map_ocr_projector_to_onnx
from mobius.tasks import (
    GGUFVisionAudioProjectorModel,
    GGUFVisionAudioProjectorTask,
    GGUFVisionProjectorModel,
    GGUFVisionProjectorTask,
    ModelTask,
)

logger = logging.getLogger(__name__)

_OCR_PROJECTOR_TYPES = frozenset(
    {
        "deepseekocr",
        "deepseekocr2",
        "dots3note_a",
        "dots3note_v",
        "dots_ocr",
        "granite4_vision",
        "lightonocr",
        "paddleocr",
        "youtuvl",
    }
)


def _standalone_config(
    output_size: int,
    *,
    dtype: str | None,
    vision: VisionConfig | None = None,
) -> ArchitectureConfig:
    resolved_dtype = resolve_dtype(dtype) if dtype is not None else None
    return ArchitectureConfig(
        vocab_size=1,
        hidden_size=output_size,
        intermediate_size=max(4, output_size * 2),
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=output_size,
        max_position_embeddings=1,
        hidden_act="silu",
        dtype=resolved_dtype or ir.DataType.FLOAT,
        vision=vision,
    )


def _dots_vision(mmproj: Any, *, qk_norm: bool) -> DotsVisionEncoder:
    md = mmproj.metadata
    depth = int(md["clip.vision.block_count"])
    expert_counts = []
    expert_intermediate = None
    for layer in range(depth):
        name = f"v.blk.{layer}.ffn_gate_exps.weight"
        if name not in mmproj.tensor_names:
            expert_counts.append(0)
            continue
        shape = mmproj.get_tensor_shape(name)
        expert_counts.append(int(shape[0]))
        expert_intermediate = int(shape[1])
    merge_key = (
        "clip.vision.spatial_merge_size" if qk_norm else "clip.vision.projector.scale_factor"
    )
    return DotsVisionEncoder(
        depth=depth,
        hidden_size=int(md["clip.vision.embedding_length"]),
        intermediate_size=int(md["clip.vision.feed_forward_length"]),
        num_heads=int(md["clip.vision.attention.head_count"]),
        patch_size=int(md["clip.vision.patch_size"]),
        output_size=int(md["clip.vision.projection_dim"]),
        projector_intermediate_size=int(mmproj.get_tensor_shape("mm.0.weight")[0]),
        spatial_merge_size=int(md[merge_key]),
        norm_eps=float(md["clip.vision.attention.layer_norm_epsilon"]),
        qk_norm=qk_norm,
        expert_counts=expert_counts,
        expert_intermediate_size=expert_intermediate,
        top_k=int(md.get("clip.vision.expert_used_count", 0)),
    )


def _dots_audio(mmproj: Any) -> Dots3NoteAudioEncoder:
    md = mmproj.metadata
    return Dots3NoteAudioEncoder(
        num_mel_bins=int(md["clip.audio.num_mel_bins"]),
        conv_channels=int(mmproj.get_tensor_shape("a.conv2d.1.weight")[0]),
        depth=int(md["clip.audio.block_count"]),
        hidden_size=int(md["clip.audio.embedding_length"]),
        intermediate_size=int(md["clip.audio.feed_forward_length"]),
        num_heads=int(md["clip.audio.attention.head_count"]),
        output_size=int(md["clip.audio.projection_dim"]),
        norm_eps=float(md["clip.audio.attention.layer_norm_epsilon"]),
    )


def _lighton(
    mmproj: Any, *, dtype: str | None
) -> tuple[LightOnOCRVisionEncoder, ArchitectureConfig]:
    md = mmproj.metadata
    hidden = int(md["clip.vision.embedding_length"])
    output = int(md["clip.vision.projection_dim"])
    vision = VisionConfig(
        hidden_size=hidden,
        intermediate_size=int(md["clip.vision.feed_forward_length"]),
        num_hidden_layers=int(md["clip.vision.block_count"]),
        num_attention_heads=int(md["clip.vision.attention.head_count"]),
        head_dim=hidden // int(md["clip.vision.attention.head_count"]),
        image_size=int(md["clip.vision.image_size"]),
        patch_size=int(md["clip.vision.patch_size"]),
        norm_eps=float(md["clip.vision.attention.layer_norm_epsilon"]),
        spatial_merge_size=int(md["clip.vision.spatial_merge_size"]),
        rope_theta=10_000.0,
        hidden_act="silu",
    )
    config = _standalone_config(output, dtype=dtype, vision=vision)
    return (
        LightOnOCRVisionEncoder(
            config,
            first_bias="mm.1.bias" in mmproj.tensor_names,
            second_bias="mm.2.bias" in mmproj.tensor_names,
        ),
        config,
    )


def _vision_module(
    mmproj: Any,
    projector_type: str,
    *,
    dtype: str | None,
) -> tuple[Any, ArchitectureConfig]:
    md = mmproj.metadata
    output = int(md["clip.vision.projection_dim"])
    if projector_type == "dots_ocr":
        return _dots_vision(mmproj, qk_norm=False), _standalone_config(output, dtype=dtype)
    if projector_type == "dots3note_v":
        return _dots_vision(mmproj, qk_norm=True), _standalone_config(output, dtype=dtype)
    if projector_type == "paddleocr":
        return (
            PaddleOCRVisionEncoder(
                depth=int(md["clip.vision.block_count"]),
                hidden_size=int(md["clip.vision.embedding_length"]),
                intermediate_size=int(md["clip.vision.feed_forward_length"]),
                num_heads=int(md["clip.vision.attention.head_count"]),
                patch_size=int(md["clip.vision.patch_size"]),
                position_size=int(mmproj.get_tensor_shape("v.position_embd.weight")[0]),
                output_size=output,
                projector_intermediate_size=int(mmproj.get_tensor_shape("mm.1.weight")[0]),
                norm_eps=float(md["clip.vision.attention.layer_norm_epsilon"]),
            ),
            _standalone_config(output, dtype=dtype),
        )
    if projector_type == "youtuvl":
        patch_size = int(md["clip.vision.patch_size"])
        return (
            YouTuVLVisionEncoder(
                depth=int(md["clip.vision.block_count"]),
                hidden_size=int(md["clip.vision.embedding_length"]),
                intermediate_size=int(md["clip.vision.feed_forward_length"]),
                num_heads=int(md["clip.vision.attention.head_count"]),
                pixel_size=3 * patch_size * patch_size,
                patch_size=patch_size,
                output_size=output,
                projector_intermediate_size=int(mmproj.get_tensor_shape("mm.0.weight")[0]),
                spatial_merge_size=int(md["clip.vision.spatial_merge_size"]),
                window_size=int(md["clip.vision.window_size"]),
                full_attention_layers=tuple(md["clip.vision.wa_layer_indexes"]),
                norm_eps=float(md["clip.vision.attention.layer_norm_epsilon"]),
            ),
            _standalone_config(output, dtype=dtype),
        )
    if projector_type == "lightonocr":
        return _lighton(mmproj, dtype=dtype)
    if projector_type == "granite4_vision":
        return (
            Granite4VisionEncoder(
                depth=int(md["clip.vision.block_count"]),
                hidden_size=int(md["clip.vision.embedding_length"]),
                intermediate_size=int(md["clip.vision.feed_forward_length"]),
                num_heads=int(md["clip.vision.attention.head_count"]),
                image_size=int(md["clip.vision.image_size"]),
                patch_size=int(md["clip.vision.patch_size"]),
                feature_layers=tuple(md["clip.vision.feature_layer"]),
                spatial_offsets=tuple(md["clip.vision.projector.spatial_offsets"]),
                query_side=int(md["clip.vision.projector.query_side"]),
                window_side=int(md["clip.vision.projector.window_side"]),
                output_size=output,
                qformer_intermediate_size=int(
                    mmproj.get_tensor_shape("v.proj_blk.0.ffn_up.weight")[0]
                ),
                norm_eps=float(md["clip.vision.attention.layer_norm_epsilon"]),
            ),
            _standalone_config(
                output * len(md["clip.vision.feature_layer"]),
                dtype=dtype,
            ),
        )
    if projector_type == "deepseekocr":
        hidden = int(md["clip.vision.embedding_length"])
        return (
            DeepSeekOCRFullImageEncoder(
                sam_hidden_size=int(md["clip.vision.sam.embedding_length"]),
                sam_num_heads=int(md["clip.vision.sam.head_count"]),
                sam_depth=int(md["clip.vision.sam.block_count"]),
                sam_window_size=int(md["clip.vision.window_size"]),
                clip_hidden_size=hidden,
                clip_intermediate_size=int(
                    mmproj.get_tensor_shape("v.blk.0.ffn_up.weight")[0]
                ),
                clip_num_heads=int(md["clip.vision.attention.head_count"]),
                clip_depth=int(md["clip.vision.block_count"]),
                output_size=output,
            ),
            _standalone_config(output, dtype=dtype),
        )
    if projector_type == "deepseekocr2":
        return (
            DeepSeekOCR2FullImageEncoder(
                sam_hidden_size=int(md["clip.vision.sam.embedding_length"]),
                sam_num_heads=int(md["clip.vision.sam.head_count"]),
                sam_depth=int(md["clip.vision.sam.block_count"]),
                sam_window_size=int(md["clip.vision.window_size"]),
                hidden_size=int(md["clip.vision.embedding_length"]),
                intermediate_size=int(md["clip.vision.feed_forward_length"]),
                depth=int(md["clip.vision.block_count"]),
                num_heads=int(md["clip.vision.attention.head_count"]),
                num_kv_heads=int(md["clip.vision.attention.head_count_kv"]),
                output_size=output,
                norm_eps=float(md["clip.vision.attention.layer_norm_epsilon"]),
            ),
            _standalone_config(output, dtype=dtype),
        )
    raise ValueError(f"Unsupported OCR projector type {projector_type!r}.")


def _map_state(
    mmproj: Any,
    projector_type: str,
    *,
    mixed: bool,
) -> dict[str, torch.Tensor]:
    from mobius.integrations.gguf._tensor_processors import _reverse_permute

    state: dict[str, torch.Tensor] = {}
    for name in mmproj.tensor_names:
        route = projector_type
        role = "vision_encoder"
        if projector_type in {"dots3note_v", "dots3note_a"}:
            if name.startswith(("a.", "mm.a.")):
                route = "dots3note_a"
                role = "audio_encoder"
            else:
                route = "dots3note_v"
        mapped = map_ocr_projector_to_onnx(name, route)
        if mapped is None:
            continue
        values = np.asarray(mmproj.get_tensor(name), dtype=np.float32)
        if name == "v.view_seperator":
            values = values.reshape(-1)
        if route == "lightonocr" and name.endswith(("attn_q.weight", "attn_k.weight")):
            values = _reverse_permute(
                torch.from_numpy(values),
                int(mmproj.metadata["clip.vision.attention.head_count"]),
            ).numpy()
        if route in {"deepseekocr", "deepseekocr2"}:
            tensor = torch.from_numpy(values.copy())
            if mapped in {"image_newline", "view_separator"}:
                state[f"vision_encoder.{mapped}"] = tensor
            else:
                state[f"vision_encoder.global_encoder.{mapped}"] = tensor
                state[f"vision_encoder.local_encoder.{mapped}"] = tensor
            continue
        prefix = f"{role}." if mixed or role == "vision_encoder" else "audio_encoder."
        state[prefix + mapped] = torch.from_numpy(values.copy())
    return state


def build_ocr_projector_from_gguf(
    _mmproj_path: Any,
    *,
    projector_type: str,
    target_architecture: str,
    dtype: str | None,
    execution_provider: str,
    _mmproj_gguf_model: Any,
):
    """Build exact standalone OCR/document sidecar components."""
    if projector_type not in _OCR_PROJECTOR_TYPES:
        raise ValueError(f"Unsupported OCR projector type {projector_type!r}.")
    mmproj = _mmproj_gguf_model
    mixed = projector_type in {"dots3note_v", "dots3note_a"}
    module: nn.Module
    task: ModelTask
    if mixed:
        vision, config = _vision_module(mmproj, "dots3note_v", dtype=dtype)
        audio = _dots_audio(mmproj)
        module = GGUFVisionAudioProjectorModel(vision, audio)
        task = GGUFVisionAudioProjectorTask()
    else:
        vision, config = _vision_module(mmproj, projector_type, dtype=dtype)
        module = GGUFVisionProjectorModel(vision)
        task = GGUFVisionProjectorTask()
    package = build_from_module(
        module,
        config,
        task=task,
        execution_provider=execution_provider,
    )
    package.apply_weights(_map_state(mmproj, projector_type, mixed=mixed))
    from mobius.integrations.gguf._mmproj import (
        _preflight_mmproj_quantization_report,
        _require_loaded_projector_initializers,
    )

    _require_loaded_projector_initializers(package, projector_type)
    package.gguf_quantization_report = _preflight_mmproj_quantization_report(
        mmproj,
        include_audio=mixed,
        standalone_projector_type=projector_type,
    )
    package.gguf_source_path = str(Path(_mmproj_path).resolve())  # type: ignore[attr-defined]
    package.gguf_projector_type = projector_type  # type: ignore[attr-defined]
    package.gguf_target_architecture = target_architecture  # type: ignore[attr-defined]
    package.gguf_projector_output_size = config.hidden_size  # type: ignore[attr-defined]
    runtime_warning = (
        "Standalone projector graph only; paired text insertion and downstream "
        "multimodal runtime execution are not validated."
    )
    package.gguf_runtime_warning = runtime_warning  # type: ignore[attr-defined]
    package_input_schema = {}
    for component, model in package.items():
        input_schema = [
            {
                "name": value.name,
                "dtype": str(value.dtype),
                "shape": [str(dim) for dim in value.shape],
            }
            for value in model.graph.inputs
        ]
        package_input_schema[component] = input_schema
        model.metadata_props["mobius.gguf_projector_type"] = projector_type
        model.metadata_props["mobius.gguf_target_architecture"] = target_architecture
        model.metadata_props["mobius.gguf_input_schema"] = json.dumps(
            input_schema,
            sort_keys=True,
            separators=(",", ":"),
        )
        model.metadata_props["mobius.projector_output_size"] = str(config.hidden_size)
        model.metadata_props["mobius.runtime_support"] = (
            "standalone-sidecar-only; paired multimodal runtime unvalidated"
        )
    package.gguf_input_schema = package_input_schema  # type: ignore[attr-defined]
    logger.warning("%s", runtime_warning)
    return package
