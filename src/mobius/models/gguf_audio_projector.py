# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone audio-sidecar container for exact GGUF projector graphs."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components import MeralionAudioSidecar

TensorShapes = Mapping[str, tuple[int, ...]]


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFAudioProcessorABI:
    """Revision-neutral processor-to-graph contract for one audio route."""

    sample_rate: int
    channels: int
    graph_input: str
    graph_layout: str
    feature_contract: str
    n_fft: int | None = None
    window_length: int | None = None
    hop_length: int | None = None
    chunk_seconds: int | None = None
    frame_multiple: int | None = None
    max_seconds: int | None = None


AUDIO_PROCESSOR_ABIS: Mapping[str, GGUFAudioProcessorABI] = {
    "meralion": GGUFAudioProcessorABI(
        16_000,
        1,
        "input_features",
        "float32[3000,128]",
        "Whisper log10 mel for one right-padded 30-second chunk",
        n_fft=400,
        window_length=400,
        hop_length=160,
        chunk_seconds=30,
        max_seconds=300,
    ),
}


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
        or value <= 0
    ):
        raise ValueError(f"{key} must be a positive finite number, got {value!r}.")
    return float(value)


def _shape(shapes: TensorShapes, name: str, rank: int) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in shapes[name])
    except KeyError as exc:
        raise ValueError(f"GGUF audio sidecar is missing tensor {name!r}.") from exc
    if len(shape) != rank or any(dim <= 0 for dim in shape):
        raise ValueError(f"GGUF audio tensor {name!r} has invalid shape {shape}.")
    return shape


class GGUFAudioProjectorModel(nn.Module):
    """One standalone audio sidecar component with no implied text runtime."""

    default_task = "gguf-audio-projector"
    category = "Audio"

    def __init__(self, audio_encoder: nn.Module):
        super().__init__()
        self.audio_encoder = audio_encoder

    def forward(self, op: OpBuilder, **kwargs):
        del op, kwargs
        raise NotImplementedError("GGUF audio projectors are built as an audio_encoder graph.")


def create_gguf_audio_projector(
    projector_type: str,
    metadata: Mapping[str, object],
    tensor_shapes: TensorShapes,
) -> GGUFAudioProjectorModel:
    """Create the exact reusable graph for one supported audio route."""
    if projector_type != "meralion":
        raise NotImplementedError(
            f"GGUF audio projector {projector_type!r} is not implemented in this cohort."
        )

    hidden_size = _metadata_int(metadata, "clip.audio.embedding_length")
    stack_factor = _metadata_int(metadata, "clip.audio.projector.stack_factor")
    position_shape = _shape(tensor_shapes, "a.position_embd.weight", 2)
    first_shape = _shape(tensor_shapes, "mm.a.mlp.0.weight", 2)
    gate_shape = _shape(tensor_shapes, "mm.a.mlp.1.weight", 2)
    pool_shape = _shape(tensor_shapes, "mm.a.mlp.2.weight", 2)
    output_shape = _shape(tensor_shapes, "mm.a.mlp.3.weight", 2)
    if (
        position_shape[1] != hidden_size
        or first_shape[1] != hidden_size * stack_factor
        or gate_shape != (first_shape[0], first_shape[0])
        or pool_shape != gate_shape
        or output_shape[1] != first_shape[0]
    ):
        raise ValueError(
            "meralion position/projector shapes do not form the pinned "
            "Whisper -> stack -> gated adapter contract"
        )
    encoder = MeralionAudioSidecar(
        num_mel_bins=_metadata_int(metadata, "clip.audio.num_mel_bins"),
        d_model=hidden_size,
        encoder_layers=_metadata_int(metadata, "clip.audio.block_count"),
        encoder_heads=_metadata_int(metadata, "clip.audio.attention.head_count"),
        encoder_ffn_dim=_metadata_int(metadata, "clip.audio.feed_forward_length"),
        max_source_positions=position_shape[0],
        projector_hidden_size=first_shape[0],
        output_size=output_shape[0],
        stack_factor=stack_factor,
        eps=_metadata_float(metadata, "clip.audio.attention.layer_norm_epsilon"),
    )
    return GGUFAudioProjectorModel(encoder)
