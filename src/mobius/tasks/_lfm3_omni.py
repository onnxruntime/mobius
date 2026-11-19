# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Five-component offline LFM3-Omni export: LFM3-Audio's graphs plus LFM3-VL's vision encoder."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import GraphBuilder, nn

from mobius._configs.lfm3_omni import Lfm3OmniConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ComponentConfig, ComponentSpec
from mobius.tasks._causal_lm import _LFM3_INPUT_CONTRACT
from mobius.tasks._lfm3_audio import Lfm3AudioTask
from mobius.tasks._vision_language_3model import Lfm3VlTask


class Lfm3OmniTask(Lfm3AudioTask, Lfm3VlTask):
    """LFM3-Audio's four graphs and LFM3-VL's one-image vision encoder.

    The embedding also takes ``image_features``, the concatenated
    ``vision_encoder`` outputs of the prompt's images in placeholder order.
    """

    components = ComponentSpec(
        vision_encoder="vision_encoder",
        audio_encoder="audio_encoder",
        embedding="embedding",
        decoder="decoder",
        depth_decoder=ComponentConfig("depth_decoder", "head"),
    )
    model_roles: ClassVar[dict[str, str]] = {
        "vision_encoder": "encoder",
        **Lfm3AudioTask.model_roles,
    }

    def build(self, module: nn.Module, config: Lfm3OmniConfig) -> ModelPackage:
        package = super().build(module, config)
        package["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        package["decoder"].metadata_props["mobius.input_contract"] = _LFM3_INPUT_CONTRACT
        return package

    def _embedding_features(
        self, builder: GraphBuilder, config: Lfm3OmniConfig
    ) -> list[ir.Value]:
        image = builder.input(
            "image_features", dtype=config.dtype, shape=["image_tokens", config.hidden_size]
        )
        return [image, *super()._embedding_features(builder, config)]
