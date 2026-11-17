# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Nested Omni configuration and native mixed-precision storage contracts."""

from __future__ import annotations

import copy
import dataclasses
import math

from mobius._configs._base import _as_attribute_config, _shallow_fields
from mobius._configs.per_model._nemotron_h_puzzle import NemotronHPuzzleConfig


@dataclasses.dataclass(frozen=True)
class NemotronHQuantizedProjection:
    """One source projection's storage format, not a global INT4 scheme."""

    quant_algo: str
    group_size: int
    block_payload_bytes: int | None


@dataclasses.dataclass
class NemotronHOmniConfig(NemotronHPuzzleConfig):
    """C-RADIOv4 + projector + nested Puzzle decoder configuration.

    ``quantized_projections`` preserves native IQ2_XXS/Q8_0/NVFP4 semantics.
    It must not be interpreted as affine integer quantization or silently
    expanded into a full BF16 checkpoint.
    """

    quantized_projections: dict[str, NemotronHQuantizedProjection] = dataclasses.field(
        default_factory=dict
    )
    radio_max_img_size: int = 2048
    radio_num_cls_tokens: int = 3
    radio_num_registers: int = 7
    radio_summary_idxs: tuple[int, ...] = (0, 1)
    radio_norm_mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)
    radio_norm_std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)
    video_temporal_patch_size: int = 2
    projector_hidden_size: int = 20480
    projector_norm_eps: float = 1e-5
    video_pruning_rate: float = 0.0

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> NemotronHOmniConfig:
        source = _as_attribute_config(parent_config if parent_config is not None else config)
        text = getattr(source, "llm_config", None)
        vision = _as_attribute_config(getattr(source, "vision_config", None))
        if text is None or getattr(_as_attribute_config(text), "model_type", None) != (
            "nemotron_h_puzzle"
        ):
            raise ValueError("Nemotron-H Omni requires a nested Nemotron-H Puzzle llm_config")
        if vision is None or getattr(vision, "model_type", None) != "radio":
            raise ValueError("Nemotron-H Omni requires a RADIO vision_config")
        for field, expected in (
            ("hidden_act", "gelu"),
            ("use_swiglu_ffn", False),
            ("qkv_bias", True),
            ("layerscale_value", 1.0),
            ("num_channels", 3),
        ):
            if getattr(vision, field, expected) != expected:
                raise NotImplementedError(
                    f"Nemotron-H Omni RADIO requires {field}={expected!r}"
                )
        if getattr(source, "sound_config", None) is not None:
            raise NotImplementedError("Nemotron-H Omni sound encoders are not supported")
        if getattr(source, "ps_version", "v2") != "v2" or not math.isclose(
            getattr(source, "downsample_ratio", 0.5), 0.5, rel_tol=0.0, abs_tol=0.0
        ):
            raise NotImplementedError("Nemotron-H Omni requires v2 spatial downsampling by 2")
        if getattr(source, "vision_final_layernorm", False):
            raise NotImplementedError(
                "Nemotron-H Omni final vision layer norm is not supported"
            )
        mean = tuple(getattr(vision, "norm_mean", (0.48145466, 0.4578275, 0.40821073)))
        std = tuple(getattr(vision, "norm_std", (0.26862954, 0.26130258, 0.27577711)))
        if (
            len(mean) != 3
            or len(std) != 3
            or not all(math.isfinite(value) for value in (*mean, *std))
            or any(value <= 0 for value in std)
        ):
            raise ValueError(
                "Omni requires finite RGB means and positive RGB standard deviations"
            )

        projections = {}
        quantization = _as_attribute_config(getattr(source, "quantization_config", None))
        if quantization is not None:
            if (
                getattr(quantization, "quant_method", None) != "modelopt"
                or getattr(quantization, "quant_algo", None) != "MIXED_PRECISION"
            ):
                raise ValueError("Omni native quantization requires ModelOpt MIXED_PRECISION")
            layers = _as_attribute_config(getattr(quantization, "quantized_layers", None))
            if layers is None or not vars(layers):
                raise ValueError("Omni mixed quantization requires explicit quantized_layers")
            contracts = {"IQ2_XXS": (256, 66), "Q8_0": (32, 34), "W4A16_NVFP4": (16, None)}
            for name, specification in vars(layers).items():
                specification = _as_attribute_config(specification)
                kind = getattr(specification, "quant_algo", None)
                if kind not in contracts:
                    raise NotImplementedError(f"Unsupported Omni projection format {kind!r}")
                group_size, payload_bytes = contracts[kind]
                if getattr(specification, "group_size", None) != group_size:
                    raise ValueError(f"Invalid {kind} group size for {name}")
                if payload_bytes is not None and (
                    getattr(specification, "packing", None) != "ggml"
                    or getattr(specification, "block_payload_bytes", None) != payload_bytes
                ):
                    raise ValueError(f"Invalid GGML block packing for {name}")
                if not name.startswith("language_model."):
                    raise NotImplementedError(f"Unsupported quantized Omni component {name}")
                projections[name] = NemotronHQuantizedProjection(
                    kind, group_size, payload_bytes
                )

        # The common reader handles dense topology/vision, while the validated
        # per-projection plan above exclusively owns mixed storage semantics.
        common = copy.copy(source)
        setattr(common, "quantization_config", None)  # noqa: B010 - dynamic HF attributes
        text = copy.copy(_as_attribute_config(text))
        setattr(text, "quantization_config", None)  # noqa: B010 - dynamic HF attributes
        base = NemotronHPuzzleConfig.from_transformers(text, common)
        fields = _shallow_fields(base)
        fields.update(
            model_type="nemotron_h_omni",
            component_quantization={
                "decoder": None,
                "vision_encoder": None,
                "embedding": None,
            },
            image_token_id=getattr(source, "img_context_token_id", None),
            eos_token_id=getattr(source, "eos_token_id", base.eos_token_id),
        )
        if fields["image_token_id"] is None:
            raise ValueError("Nemotron-H Omni requires img_context_token_id")
        return cls(
            **fields,
            quantized_projections=projections,
            radio_max_img_size=getattr(vision, "max_img_size", 2048),
            radio_num_cls_tokens=getattr(vision, "num_cls_tokens", 3),
            radio_num_registers=getattr(vision, "num_registers", 7),
            radio_summary_idxs=tuple(getattr(vision, "summary_idxs", [0, 1])),
            radio_norm_mean=mean,
            radio_norm_std=std,
            video_temporal_patch_size=getattr(
                source,
                "video_temporal_patch_size",
                getattr(vision, "video_temporal_patch_size", 2),
            ),
            projector_hidden_size=getattr(source, "projector_hidden_size", 20480),
            projector_norm_eps=1e-5,
            video_pruning_rate=getattr(source, "video_pruning_rate", 0.0),
        )
