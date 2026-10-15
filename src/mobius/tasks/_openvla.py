from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import OpenVLAConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
    build_decoder_from_embeds,
)


class OpenVLATask(ModelTask):
    """OpenVLA vision/projector, BOS insertion and own-cache Llama decoder graphs."""

    components = ComponentSpec(
        decoder="decoder",
        vision_encoder="vision_encoder",
        embedding="embedding",
    )
    model_roles: ClassVar[dict[str, str]] = {
        "decoder": "decoder",
        "vision_encoder": "vision",
        "embedding": "embedding",
    }

    def build(self, module: nn.Module, config: OpenVLAConfig) -> ModelPackage:
        from mobius.models.openvla import OpenVLAModel

        if not isinstance(module, OpenVLAModel):
            raise TypeError("OpenVLATask requires an OpenVLAModel")
        self._validate_components(module)
        vision, builder = _make_graph("vision_encoder")
        pixels = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=["batch", 6, config.dino.image_size, config.dino.image_size],
        )
        builder.add_output(
            module.vision_encoder(builder.op, pixel_values=pixels), "image_features"
        )

        embedding, builder = _make_graph("embedding")
        input_ids = builder.input(
            "input_ids", dtype=ir.DataType.INT64, shape=["batch", "text_sequence"]
        )
        features = builder.input(
            "image_features",
            dtype=config.dtype,
            shape=["batch", "image_sequence", config.hidden_size],
        )
        builder.add_output(
            module.embedding(builder.op, input_ids=input_ids, image_features=features),
            "inputs_embeds",
        )
        return ModelPackage(
            {
                "decoder": build_decoder_from_embeds(module.decoder, config),
                "vision_encoder": _make_model(vision),
                "embedding": _make_model(embedding),
            },
            config=config,
        )
