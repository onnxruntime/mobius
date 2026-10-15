# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Three-model Omni interfaces with mixed KV/Mamba state and media retention."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    _make_graph,
    _make_model,
    build_decoder_from_embeds,
    build_embedding_from_features,
)
from mobius.tasks._vision_language_3model import VisionLanguageTask


class NemotronHOmniTask(VisionLanguageTask):
    """Split Puzzle decoding, flat media embedding and source-exact EVS vision."""

    def build(self, module: nn.Module, config: ArchitectureConfig) -> ModelPackage:
        self._validate_components(module)
        decoder: object = getattr(module, "decoder", None)
        vision_encoder: object = getattr(module, "vision_encoder", None)
        embedding: object = getattr(module, "embedding", None)
        if (
            not isinstance(decoder, nn.Module)
            or not isinstance(vision_encoder, nn.Module)
            or not isinstance(embedding, nn.Module)
        ):
            raise TypeError("Omni decoder, vision encoder and embedding must be modules")
        return ModelPackage(
            {
                "decoder": build_decoder_from_embeds(decoder, config, hybrid=True),
                "vision_encoder": self._build_vision(vision_encoder, config),
                "embedding": build_embedding_from_features(
                    embedding,
                    config,
                    feature_name="image_features",
                    feature_dim=config.hidden_size,
                ),
            },
            config=config,
        )

    def _build_vision(self, vision: nn.Module, config: ArchitectureConfig) -> ir.Model:
        graph, builder = _make_graph(name="vision_encoder")
        # Processor output is already normalized. Images or one video's frames
        # are processed per call; no padded dummy media or second normalization.
        pixels = builder.input(
            "pixel_values",
            dtype=ir.DataType.FLOAT,
            shape=["num_images_or_frames", 3, "image_height", "image_width"],
        )
        is_video = builder.input("is_video", dtype=ir.DataType.BOOL, shape=[])
        outputs = vision(builder.op, pixel_values=pixels, is_video=is_video)
        for value, name in zip(outputs, ("image_features", "retention_mask", "kept_indices")):
            builder.add_output(value, name)
        return _make_model(graph)
