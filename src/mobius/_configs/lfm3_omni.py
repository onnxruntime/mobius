# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM3-Omni configuration: the LFM3-Audio speech parts and the LFM3-VL vision tower."""

from __future__ import annotations

import dataclasses
import inspect

from mobius._configs._base import _shallow_fields
from mobius._configs.lfm3_audio import Lfm3AudioConfig
from mobius._configs.lfm3_vl import Lfm3VlConfig


@dataclasses.dataclass
class Lfm3OmniConfig(Lfm3AudioConfig, Lfm3VlConfig):
    """Configuration for LiquidAI LFM3-Omni.

    The checkpoint's ``config.json`` holds the LFM3-Audio layout (``text_config``,
    ``audio_encoder_config``, ``depth_config`` and the speech ids) plus the LFM3-VL
    ``vision_config`` and projector fields, so both parsers read it unchanged.
    """

    model_type: str = "lfm3_omni"

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Lfm3OmniConfig:
        source = parent_config if parent_config is not None else config
        fields = _shallow_fields(Lfm3VlConfig.from_transformers(source.text_config, source))
        # The audio parser fills only the fields its class declares; it never sees the vision tower.
        audio = Lfm3AudioConfig.from_transformers(source)
        fields.update(
            {name: getattr(audio, name) for name in inspect.get_annotations(Lfm3AudioConfig)}
        )
        fields["model_type"] = "lfm3_omni"
        return cls(**fields)
