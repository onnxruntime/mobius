# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM3-VL configuration: LFM3 decoder, conv/attention vision tower, MLP projector."""

from __future__ import annotations

import dataclasses

from mobius._configs._base import _shallow_fields
from mobius._configs.lfm3 import Lfm3Config


@dataclasses.dataclass
class Lfm3VlConfig(Lfm3Config):
    """Configuration for LiquidAI LFM3-VL.

    The HuggingFace config is composite: the LFM3 decoder fields live under
    ``text_config`` (which is what ``config`` refers to here), the projector
    fields on the top level, and the vision tower under ``vision_config``.
    :attr:`vision` holds the tower geometry; the three vision fields
    :class:`VisionConfig` has no slot for are kept here. Defaults follow the
    checkpoint's ``configuration_lfm3_vl.py``.
    """

    model_type: str = "lfm3_vl"
    downsample_factor: int = 2
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 4096
    projector_bias: bool = True
    projector_layer_norm_eps: float = 1e-6
    vision_num_key_value_heads: int = 8
    vision_layer_types: list[str] | None = None
    vision_conv_kernel_size: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        vision = self.vision
        if vision is None:
            raise ValueError("LFM3-VL requires a vision config")
        if self.vision_layer_types is None:
            self.vision_layer_types = [
                "conv" if i < 2 or i % 2 else "full_attention"
                for i in range(vision.num_hidden_layers)
            ]
        # Transformers 5.16 (the CI pin) renames custom-code "conv" to "linear_attention".
        self.vision_layer_types = [
            "conv" if kind == "linear_attention" else kind for kind in self.vision_layer_types
        ]
        if len(self.vision_layer_types) != vision.num_hidden_layers:
            raise ValueError("LFM3-VL vision_layer_types must match the vision layer count")
        unknown = sorted(set(self.vision_layer_types) - {"conv", "full_attention"})
        if unknown:
            raise ValueError(f"Unsupported LFM3-VL vision layer type(s): {unknown}")
        if vision.hidden_size % vision.num_attention_heads or (
            vision.num_attention_heads % self.vision_num_key_value_heads
        ):
            raise ValueError(
                "LFM3-VL vision head counts must divide hidden_size and each other"
            )
        if self.downsample_factor != vision.spatial_merge_size:
            raise ValueError("LFM3-VL downsample_factor must equal vision spatial_merge_size")

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Lfm3VlConfig:
        source = parent_config if parent_config is not None else config
        vision = source.vision_config
        fields = _shallow_fields(Lfm3Config.from_transformers(config, parent_config))
        fields.update(
            model_type="lfm3_vl",
            **{
                name: getattr(source, name, getattr(cls, name))
                for name in (
                    "downsample_factor",
                    "projector_hidden_act",
                    "projector_hidden_size",
                    "projector_bias",
                    "projector_layer_norm_eps",
                )
            },
            vision_num_key_value_heads=getattr(
                vision, "num_key_value_heads", cls.vision_num_key_value_heads
            ),
            vision_layer_types=getattr(vision, "layer_types", None),
            vision_conv_kernel_size=getattr(
                vision, "conv_kernel_size", cls.vision_conv_kernel_size
            ),
        )
        return cls(**fields)
