# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent FP32 Drama 3 waveform graph/export integration helper."""

from __future__ import annotations

from pathlib import Path

import onnx_ir as ir
import torch
from huggingface_hub import hf_hub_download

from mobius._configs.fish_codec import (
    DRAMA3_CODEC_REPO,
    DRAMA3_CODEC_REVISION,
    FishCodecConfig,
)
from mobius._model_package import ModelPackage
from mobius.integrations._weight_loading import apply_weights
from mobius.models.fish_codec import FishCodecDecoder
from mobius.tasks._base import _make_graph, _make_model


def build_fish_codec_decoder(
    config: FishCodecConfig | None = None,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> ModelPackage:
    """Build ``pkg["waveform_decoder"]`` without downloads or tracing.

    Contract: ``codes`` INT64 [batch,10,frames] -> ``waveform`` FP32
    [batch,1,frames*2048], 44100 Hz. Batch and frame count are dynamic;
    1 <= frames <= config.max_frames. Codes are nonnegative. Values above
    each native vocabulary are max-clamped exactly as upstream does.
    A graph built without ``state_dict`` has unbound learned initializers.
    Only FP32 is currently validated; no reduced-precision claim is implied.
    """
    config = config or FishCodecConfig()
    module = FishCodecDecoder(config)
    graph, builder = _make_graph(name="fish_codec_decoder")
    codes = builder.input(
        "codes", dtype=ir.DataType.INT64, shape=["batch", config.num_codebooks, "frames"]
    )
    waveform = module(builder.op, codes)
    waveform.type = ir.TensorType(ir.DataType.FLOAT)
    waveform.shape = ir.Shape(["batch", 1, "samples"])
    builder.add_output(waveform, "waveform")
    model = _make_model(graph)
    model.metadata_props.update(
        {
            "sample_rate": str(config.sample_rate),
            "samples_per_frame": str(config.samples_per_frame),
            "max_frames": str(config.max_frames),
            "codec_repo": DRAMA3_CODEC_REPO,
            "codec_revision": DRAMA3_CODEC_REVISION,
        }
    )
    if state_dict is not None:
        apply_weights(model, module.preprocess_weights(state_dict))
    return ModelPackage({"waveform_decoder": model})


def load_drama3_codec_decoder(
    checkpoint_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> ModelPackage:
    """Load the pinned codec (or a caller-owned local copy) safely, then build.

    The 1.87 GB checkpoint includes unused encoding weights. Loading is
    CPU memory-mapped and always ``weights_only=True``; arbitrary pickle
    globals are never allowlisted. Local paths are explicitly caller-owned.
    Save the resulting package with ``pkg.save(output_directory)`` or add
    its ``waveform_decoder`` graph to the parent Drama 3 package.
    """
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(
            DRAMA3_CODEC_REPO,
            "codec.pth",
            revision=DRAMA3_CODEC_REVISION,
            cache_dir=cache_dir,
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("Expected codec.pth to contain a tensor state dictionary")
    return build_fish_codec_decoder(state_dict=state_dict)
