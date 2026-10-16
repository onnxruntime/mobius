# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fail-closed, bounded loading of Omni IQ2_XXS/Q8_0/NVFP4 checkpoints."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from functools import partial

import onnx_ir as ir
import torch

from mobius._configs import NemotronHOmniConfig
from mobius._model_package import ModelPackage
from mobius.integrations._weight_loading import (
    StreamingTransformedWeightSource,
    StreamingWeightPlan,
    StreamingWeightSource,
    stream_preprocessed_safetensors_to_package,
)
from mobius.models.nemotron_h import _rename_nemotron_h_weight


def _identity(tensor: torch.Tensor, _name: str) -> torch.Tensor:
    return tensor


def _validate_ggml_scale(tensor: torch.Tensor, name: str) -> None:
    scales = tensor[..., :2].contiguous().view(torch.float16)
    if not torch.isfinite(scales).all().item():
        raise ValueError(f"Nonfinite GGML block scale in {name}")


def _nvfp4_scale_bytes(tensor: torch.Tensor, _name: str) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _validate_nvfp4_scales(tensor: torch.Tensor, name: str) -> None:
    values = tensor.to(torch.float32)
    if not torch.isfinite(values).all().item() or (values < 0).any().item():
        raise ValueError(f"Invalid NVFP4 scale in {name}")


def _scalar_to_vector(tensor: torch.Tensor, _name: str) -> torch.Tensor:
    return tensor.reshape(1)


def _fixed_shape(shape: ir.Shape) -> tuple[int, ...]:
    dimensions = []
    for dimension in shape:
        if not isinstance(dimension, int):
            raise TypeError("Native Omni graph initializers require fixed dimensions")
        dimensions.append(dimension)
    return tuple(dimensions)


def _target_name(source: str, config: NemotronHOmniConfig) -> str | None:
    if source in {
        "language_model.backbone.embeddings.weight",
        "language_model.model.embeddings.weight",
    }:
        return "embedding.embed_tokens.weight"
    if source.startswith("language_model."):
        return "decoder." + _rename_nemotron_h_weight(
            source.removeprefix("language_model."), config.layer_types or []
        )
    if source.startswith("vision_model.radio_model.model."):
        return "vision_encoder.vision_model." + source.removeprefix(
            "vision_model.radio_model.model."
        )
    if source.startswith("mlp1."):
        return "vision_encoder.projector." + source
    return None


def build_nemotron_h_omni_streaming_plan(
    config: NemotronHOmniConfig,
    key_index: Mapping[str, tuple[str, list[int], str]],
    initializers: Mapping[str, ir.Value],
) -> StreamingWeightPlan:
    """Classify every source from authoritative per-projection storage metadata."""
    targets: dict[str, StreamingWeightSource | StreamingTransformedWeightSource] = {}

    def transformed(
        target_name: str,
        source_name: str,
        source_shape: tuple[int, ...],
        source_dtype: str,
        transform: Callable[[torch.Tensor, str], torch.Tensor],
        *,
        validator: Callable[[torch.Tensor, str], None] | None = None,
        scratch_bytes: int = 0,
    ) -> None:
        value = initializers.get(target_name)
        if value is None or value.shape is None or value.dtype is None:
            raise ValueError(f"Missing native Omni graph initializer {target_name!r}")
        if target_name in targets:
            raise ValueError(f"Duplicate Omni binding for {target_name}")
        targets[target_name] = StreamingTransformedWeightSource(
            source_name=source_name,
            expected_source_shape=source_shape,
            expected_source_dtype=source_dtype,
            expected_target_shape=_fixed_shape(value.shape),
            expected_target_dtype=value.dtype,
            transform=transform,
            validate_tensor=validator,
            scratch_bytes=scratch_bytes,
        )

    for source_root, specification in config.quantized_projections.items():
        target_root = _target_name(source_root, config)
        if target_root is None:
            raise ValueError(f"Unsupported Omni source projection {source_root}")
        weight_name = target_root + ".weight"
        weight = initializers.get(weight_name)
        if weight is None or weight.shape is None or weight.dtype != ir.DataType.UINT8:
            raise ValueError(f"Missing packed graph weight for {source_root}")
        shape = _fixed_shape(weight.shape)
        source_name = source_root + ".weight"
        if specification.quant_algo in {"IQ2_XXS", "Q8_0"}:
            transformed(
                weight_name,
                source_name,
                shape,
                "U8",
                _identity,
                validator=_validate_ggml_scale,
            )
        elif specification.quant_algo == "W4A16_NVFP4":
            rows, packed_k = shape
            transformed(weight_name, source_name, shape, "U8", _identity)
            transformed(
                target_root + ".weight_scale",
                source_root + ".weight_scale",
                (rows, packed_k // 8),
                "F8_E4M3",
                _nvfp4_scale_bytes,
                validator=_validate_nvfp4_scales,
                scratch_bytes=rows * (packed_k // 8) * 5,
            )
            scalar_source = source_root + ".weight_scale_2"
            metadata = key_index.get(scalar_source)
            if metadata is None or tuple(metadata[1]) not in {(), (1,)}:
                raise ValueError(f"NVFP4 global scale must be scalar: {scalar_source}")
            transformed(
                target_root + ".weight_scale_2",
                scalar_source,
                tuple(metadata[1]),
                "F32",
                _scalar_to_vector,
                validator=_validate_nvfp4_scales,
            )
        else:
            raise ValueError(f"Unsupported Omni format {specification.quant_algo}")

    for source_name, (_path, _shape, dtype) in key_index.items():
        target_name = _target_name(source_name, config)
        if target_name is None or target_name not in initializers:
            continue
        if target_name in targets:
            continue
        if dtype not in {"BF16", "F16", "F32"}:
            raise ValueError(
                f"Unclassified packed Omni tensor {source_name!r}; refusing a dense fallback"
            )
        targets[target_name] = StreamingWeightSource(source_name)

    # The pinned processor normalizes pixels. Validate these externalized
    # conditioner buffers rather than silently dropping or applying them twice.
    constants = {
        "vision_model.radio_model.input_conditioner.norm_mean": torch.tensor(
            config.radio_norm_mean, dtype=torch.float32
        ).reshape(3, 1, 1),
        "vision_model.radio_model.input_conditioner.norm_std": torch.tensor(
            config.radio_norm_std, dtype=torch.float32
        ).reshape(3, 1, 1),
        "vision_model.summary_idxs": torch.tensor(
            config.radio_summary_idxs, dtype=torch.int64
        ),
    }
    return StreamingWeightPlan(
        targets=targets,
        constants=constants,
        report={
            "streaming_external_data": True,
            "output_weight_format": (
                "native-mixed-precision" if config.quantized_projections else "dense"
            ),
            "native_iq2_storage": any(
                item.quant_algo == "IQ2_XXS" for item in config.quantized_projections.values()
            ),
            "full_checkpoint_dequantization": False,
            "projection_formats": dict(
                Counter(item.quant_algo for item in config.quantized_projections.values())
            ),
            "externalized_conditioner_tensors": sorted(
                name for name in constants if ".input_conditioner." in name
            ),
            "validated_summary_indices": list(config.radio_summary_idxs),
            "ggml_reconstructed_weight_dtype": str(config.dtype),
        },
    )


def stream_nemotron_h_omni_safetensors_to_package(
    package: ModelPackage,
    model_id: str,
    config: NemotronHOmniConfig,
    *,
    revision: str | None = None,
) -> dict[str, object]:
    """Bind one native checkpoint copy, preserving source/target accountability."""
    return stream_preprocessed_safetensors_to_package(
        package,
        model_id,
        partial(build_nemotron_h_omni_streaming_plan, config),
        revision=revision,
    )
