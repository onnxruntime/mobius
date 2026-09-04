# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit onnx-genai ``inference_metadata`` for multi-model pipelines.

Mobius builds the neural components of a diffusion model (denoiser transformer,
VAE, and — externally — a text encoder) as separate ONNX graphs. The document
this module produces wires them into onnx-genai's ``pipeline.workflow``: a typed
SSA graph in which the sampler is an executable component the package ships, so
the sigma schedule and timestep table are constant components, the step index is
the loop induction value, and classifier-free guidance is two denoiser
invocations plus a combine component.

It builds that document from the component filenames plus a scheduler config,
materializing the sampler components from mobius's policy library. It reads no
torch/diffusers state — only plain values — so it is cheap to unit-test and safe
to call anywhere.

Not everything here is publishable: :func:`build_native_vlm_package_metadata`
returns mobius's *internal* structural descriptor of a VLM package, from which
:func:`~mobius.integrations.onnx_genai.workflow_metadata.build_vlm_workflow_metadata`
derives the published contract.

Autoregressive decoder-only LLM metadata (``model.attention`` + ``kv_cache``)
lives in the sibling :mod:`mobius.integrations.onnx_genai.decoder_metadata`
module.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import math
import os
import re
import shutil
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from mobius._constants import (
    STATIC_CACHE_KV_SEQUENCE_LENGTH,
    STATIC_CACHE_WRITE_INDICES,
)
from mobius._pipeline_contract import component_presence, optional_input_contract
from mobius.integrations.onnx_genai._metadata_io import _dump_yaml
from mobius.integrations.onnx_genai._workflow_contract import (
    _invoke,
    _Port,
    _port,
    _publish_workflow_v1,
    _request_aligned,
    _shape_metadata,
    add_policy_components_to_workflow,
    declare_input_admission,
    declare_request_alignment,
)
from mobius.upstream_patches import apply_asset_patches

_LOGGER = logging.getLogger(__name__)

_RUNTIME_ASSET_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "image_processor.json",
)

#: Assets a text-only package needs. Excludes the image/audio processor
#: contracts, which would advertise media preprocessing a text package's
#: graphs cannot consume. ``chat_template.jinja`` is required, not optional:
#: instruction-tuned decoders (Gemma 4, Llama-3-Instruct, Qwen-Instruct)
#: depend on their turn markers and leading BOS, and degenerate into
#: repetition when a raw prompt reaches the model instead.
_TEXT_RUNTIME_ASSET_NAMES = tuple(
    name
    for name in _RUNTIME_ASSET_NAMES
    if name
    not in {"processor_config.json", "preprocessor_config.json", "image_processor.json"}
)


@dataclasses.dataclass(frozen=True)
class _ImageProgram:
    """Registry result for one declared image processor contract."""

    name: str
    bindings: tuple[tuple[_Port, str, float | None], ...]
    transforms: Callable[[Any, dict[str, Any]], list[dict[str, Any]]]
    token_count_source: str
    summary_contents: tuple[str, ...] = ()
    vision_properties: Callable[[dict[str, Any]], dict[str, Any]] | None = None


def _name_image_preprocessing_program(image: dict[str, Any]) -> None:
    """Convert structural preprocessing transforms into explicit typed SSA values."""
    transforms = image["transforms"]
    if all("source" in output for output in image["outputs"]) and all(
        "outputs" in transform for transform in transforms
    ):
        # Already named: re-running would append duplicate derived transforms.
        return
    current: str | None = None
    decoded: str | None = None
    for index, transform in enumerate(transforms):
        name = f"image.transform_{index}"
        if transform["op"] in {"decode", "decode_rgb"}:
            transform.pop("inputs", None)
            decoded = name
        else:
            if current is None:
                raise ValueError("image preprocessing must decode before transforming")
            transform["inputs"] = [current]
        transform["outputs"] = [name]
        current = name
    if current is None:
        raise ValueError("image preprocessing must declare at least one transform")

    derived_ops = {
        "original_size": ("emit_original_size", decoded),
        "transformed_size": ("emit_transformed_size", current),
        "validity_mask": ("emit_validity_mask", current),
        "patch_coordinates": ("emit_patch_coordinates", current),
        "grid_dimensions": ("emit_grid_coordinates", current),
    }
    for output in image["outputs"]:
        content = output["content"]
        if content == "pixels":
            output["source"] = current
            continue
        if content not in derived_ops:
            raise ValueError(
                f"image preprocessing output content {content!r} has no typed SSA producer"
            )
        operation, source = derived_ops[content]
        if source is None:
            raise ValueError(
                f"image preprocessing output content {content!r} requires a decoded image"
            )
        name = f"image.output_{content}"
        transforms.append(
            {
                "op": operation,
                "inputs": [source],
                "outputs": [name],
            }
        )
        output["source"] = name


def _port_metadata(port: _Port) -> dict[str, Any]:
    """Serialize one exact graph port for the executable component contract."""
    return {
        "name": port.name,
        "dtype": port.dtype,
        "rank": port.rank,
        "shape": _shape_metadata(port),
    }


def _same_dim(left: Any, right: Any) -> bool:
    if isinstance(left, int) or isinstance(right, int):
        return left == right
    return getattr(left, "value", None) == getattr(right, "value", None)


def _ports_match_for_dataflow(source: _Port, target: _Port) -> bool:
    return source.dtype == target.dtype and source.rank == target.rank


def _is_float(port: _Port) -> bool:
    return port.dtype in {
        "fp32",
        "fp16",
        "bf16",
        "float8_e4m3fn",
        "float8_e5m2",
    }


def _is_integer(port: _Port) -> bool:
    return port.dtype in {"int64", "int32", "int8", "uint8"}


def _static_dim(port: _Port, index: int) -> int | None:
    if port.rank is None or not -port.rank <= index < port.rank:
        return None
    dim = port.dims[index]
    return dim if isinstance(dim, int) else None


def _select_one(
    ports: Iterable[_Port],
    predicate: Callable[[_Port], bool],
) -> _Port | None:
    matches = [port for port in ports if predicate(port)]
    return matches[0] if len(matches) == 1 else None


def _resample_name(value: Any) -> str:
    if isinstance(value, int):
        return {
            0: "nearest",
            1: "lanczos",
            2: "bilinear",
            3: "bicubic",
            4: "box",
            5: "hamming",
        }.get(value, str(value))
    return str(getattr(value, "name", value) or "bicubic").lower()


def _enabled(values: dict[str, Any], name: str, default: bool) -> bool:
    value = values.get(name)
    return default if value is None else bool(value)


def _pixel_value_transforms(
    values: dict[str, Any],
    *,
    default_rescale: bool,
    default_normalize: bool,
    default_rescale_factor: float | None = None,
    default_mean: tuple[float, ...] | None = None,
    default_std: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    transforms: list[dict[str, Any]] = []
    if _enabled(values, "do_rescale", default_rescale):
        scale = values.get("rescale_factor", default_rescale_factor)
        if scale is None:
            raise ValueError(
                "Image processor declares do_rescale=true but no rescale_factor. "
                "Regenerate the processor assets with an explicit factor or add it "
                "to the structural processor registry entry."
            )
        transforms.append({"op": "rescale", "scale": float(scale)})
    if _enabled(values, "do_normalize", default_normalize):
        mean = values.get("image_mean", default_mean)
        std = values.get("image_std", default_std)
        if mean is None or std is None:
            raise ValueError(
                "Image processor declares do_normalize=true but no image_mean/image_std. "
                "Regenerate the processor assets with explicit normalization values or "
                "add them to the structural processor registry entry."
            )
        transforms.append(
            {
                "op": "normalize",
                "mean": [float(value) for value in mean],
                "std": [float(value) for value in std],
            }
        )
    return transforms


def _area_grid_transforms(config: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    del config
    size = values["size"]
    patch_size = int(values["patch_size"])
    merge_size = int(values["merge_size"])
    transforms: list[dict[str, Any]] = [{"op": "decode_rgb"}]
    if _enabled(values, "do_resize", True):
        transforms.append(
            {
                "op": "resize",
                "mode": "pixel_area",
                "interpolation": _resample_name(values.get("resample", "bicubic")),
                "min_pixels": int(size["shortest_edge"]),
                "max_pixels": int(size["longest_edge"]),
                "size_multiple": patch_size * merge_size,
            }
        )
    transforms.extend(
        _pixel_value_transforms(
            values,
            default_rescale=True,
            default_normalize=True,
            default_rescale_factor=1 / 255,
            default_mean=(0.5, 0.5, 0.5),
            default_std=(0.5, 0.5, 0.5),
        )
    )
    transforms.append(
        {
            "op": "patchify",
            "patch_size": patch_size,
            "flatten": True,
            "temporal_patch_size": int(values["temporal_patch_size"]),
            "merge_size": merge_size,
            "channel_order": "channels_first",
        }
    )
    return transforms


def _patch_budget_transforms(config: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    del config
    patch_size = int(values["patch_size"])
    pooling = int(values["pooling_kernel_size"])
    max_soft_tokens = int(values["max_soft_tokens"])
    transforms: list[dict[str, Any]] = [{"op": "decode_rgb"}]
    if _enabled(values, "do_resize", True):
        transforms.append(
            {
                "op": "resize",
                "mode": "aspect_ratio_patch_budget",
                "patch_size": patch_size,
                "max_patches": max_soft_tokens * pooling**2,
                "pooling_kernel_size": pooling,
                "interpolation": _resample_name(values.get("resample", "bicubic")),
            }
        )
    transforms.extend(
        _pixel_value_transforms(
            values,
            default_rescale=True,
            default_normalize=False,
            default_rescale_factor=1 / 255,
        )
    )
    transforms.append(
        {
            "op": "patchify",
            "patch_size": patch_size,
            "channel_order": "channels_last",
            "coordinate_order": "xy",
            "flatten": True,
        }
    )
    transforms.append(
        {
            "op": "pad",
            "pad_value": 0,
            "target_length": max_soft_tokens * pooling**2,
        }
    )
    return transforms


def _dynamic_hd_transforms(config: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    del config
    transforms: list[dict[str, Any]] = [
        {"op": "decode_rgb"},
        {
            "op": "tile",
            "mode": "dynamic_hd",
            "tile_size": int(values["crop_size"]),
            "max_tiles": int(values["dynamic_hd"]),
            "include_thumbnail": bool(values["include_thumbnail"]),
            "thumbnail_order": values["thumbnail_order"],
            "interpolation": _resample_name(values.get("resample", "bilinear")),
            "thumbnail_interpolation": _resample_name(
                values.get("thumbnail_resample", "bicubic")
            ),
            "canvas_pad_value": float(values.get("canvas_pad_value", 255)),
            "mask_patch_size": int(values.get("mask_patch_size", 14)),
        },
    ]
    transforms.extend(
        _pixel_value_transforms(
            values,
            default_rescale=True,
            default_normalize=True,
            default_rescale_factor=1 / 255,
            default_mean=(0.5, 0.5, 0.5),
            default_std=(0.5, 0.5, 0.5),
        )
    )
    return transforms


def _match_packed_coordinates(
    ports: list[_Port],
) -> tuple[tuple[_Port, str, float | None], ...] | None:
    pixels = _select_one(ports, lambda p: _is_float(p) and p.rank == 3)
    coordinates = _select_one(
        ports,
        lambda p: _is_integer(p) and p.rank == 3 and _static_dim(p, -1) == 2,
    )
    if pixels is None or coordinates is None:
        return None
    return ((pixels, "pixels", None), (coordinates, "patch_coordinates", -1))


def _match_packed_grid(
    ports: list[_Port],
) -> tuple[tuple[_Port, str, float | None], ...] | None:
    pixels = _select_one(ports, lambda p: _is_float(p) and p.rank == 2)
    grid = _select_one(
        ports,
        lambda p: _is_integer(p) and p.rank == 2 and _static_dim(p, -1) == 3,
    )
    if pixels is None or grid is None:
        return None
    return ((pixels, "pixels", None), (grid, "grid_dimensions", None))


def _match_crop_mask(
    ports: list[_Port],
) -> tuple[tuple[_Port, str, float | None], ...] | None:
    pixels = _select_one(ports, lambda p: _is_float(p) and p.rank == 4)
    transformed_size = _select_one(
        ports,
        lambda p: _is_integer(p) and p.rank == 2 and _static_dim(p, -1) == 2,
    )
    validity_mask = _select_one(ports, lambda p: _is_float(p) and p.rank == 3)
    if pixels is None or transformed_size is None or validity_mask is None:
        return None
    return (
        (pixels, "pixels", None),
        (transformed_size, "transformed_size", None),
        (validity_mask, "validity_mask", 0),
    )


def _match_area_grid(ports: list[_Port], values: dict[str, Any]) -> _ImageProgram | None:
    bindings = _match_packed_grid(ports)
    size = values.get("size")
    if (
        bindings is None
        or not isinstance(size, dict)
        or not isinstance(size.get("shortest_edge"), int)
        or not isinstance(size.get("longest_edge"), int)
        or not all(
            isinstance(values.get(key), int)
            for key in ("patch_size", "temporal_patch_size", "merge_size")
        )
    ):
        return None
    return _ImageProgram(
        name="area_bounded_packed_grid",
        bindings=bindings,
        transforms=_area_grid_transforms,
        token_count_source="from_grid",
        summary_contents=("grid_dimensions",),
    )


def _max_token_grid_transforms(config: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    patch_size = int(values["patch_size"])
    merge_size = int(values["merge_size"])
    token_pixels = (patch_size * merge_size) ** 2
    declared = dict(values)
    declared["size"] = {
        "shortest_edge": token_pixels,
        "longest_edge": int(values["max_image_tokens"]) * token_pixels,
    }
    return _area_grid_transforms(config, declared)


def _match_max_token_grid(ports: list[_Port], values: dict[str, Any]) -> _ImageProgram | None:
    bindings = _match_packed_grid(ports)
    if bindings is None or not all(
        isinstance(values.get(key), int)
        for key in (
            "patch_size",
            "temporal_patch_size",
            "merge_size",
            "max_image_tokens",
        )
    ):
        return None
    return _ImageProgram(
        name="max_token_packed_grid",
        bindings=bindings,
        transforms=_max_token_grid_transforms,
        token_count_source="from_grid",
        summary_contents=("grid_dimensions",),
    )


def _match_patch_budget(ports: list[_Port], values: dict[str, Any]) -> _ImageProgram | None:
    bindings = _match_packed_coordinates(ports)
    if (
        bindings is None
        or values.get("size") is not None
        or not all(
            isinstance(values.get(key), int)
            for key in ("patch_size", "pooling_kernel_size", "max_soft_tokens")
        )
    ):
        return None
    return _ImageProgram(
        name="aspect_ratio_patch_budget",
        bindings=bindings,
        transforms=_patch_budget_transforms,
        token_count_source="from_coordinates",
        summary_contents=("patch_coordinates",),
        vision_properties=lambda declared: {
            "token_pooling_factor": int(declared["pooling_kernel_size"]) ** 2,
            "max_tokens_per_image": int(declared["max_soft_tokens"]),
        },
    )


def _match_dynamic_hd(ports: list[_Port], values: dict[str, Any]) -> _ImageProgram | None:
    bindings = _match_crop_mask(ports)
    if bindings is None or not all(
        isinstance(values.get(key), int) for key in ("dynamic_hd", "crop_size")
    ):
        return None
    return _ImageProgram(
        name="dynamic_hd_crop_mask",
        bindings=bindings,
        transforms=_dynamic_hd_transforms,
        token_count_source="from_validity_mask",
        summary_contents=("transformed_size", "validity_mask"),
        vision_properties=lambda declared: {
            "thumbnail_order": declared["thumbnail_order"],
        },
    )


_IMAGE_PROCESSOR_REGISTRY: tuple[
    Callable[[list[_Port], dict[str, Any]], _ImageProgram | None], ...
] = (
    _match_area_grid,
    _match_max_token_grid,
    _match_patch_budget,
    _match_dynamic_hd,
)


def _resolve_image_program(model: Any, values: dict[str, Any]) -> _ImageProgram:
    ports = [_port(value) for value in model.graph.inputs]
    for resolve in _IMAGE_PROCESSOR_REGISTRY:
        program = resolve(ports, values)
        if program is not None:
            return program
    signature = [(port.name, port.dtype, port.rank, port.dims) for port in ports]
    declared = sorted(key for key, value in values.items() if value is not None)
    raise ValueError(
        "Cannot emit native VLM preprocessing metadata: the vision_encoder input "
        f"signature {signature} and declared processor keys {declared} do not match "
        "a registered structural processor contract. Generic fallback is unsafe "
        "because resize/tiling/normalization semantics would be guessed. Regenerate "
        "the package with complete config.json plus processor_config.json or "
        "preprocessor_config.json, or register this structural signature in "
        "_IMAGE_PROCESSOR_REGISTRY."
    )


@lru_cache(maxsize=16)
def _cached_source_assets(source: str) -> dict[str, str]:
    """Index all files cached for a Hub model, across cached revisions."""
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
    except Exception:
        return {}
    repos = [repo for repo in cache.repos if repo.repo_id == source]
    if not repos:
        return {}
    assets: dict[str, str] = {}
    revisions = sorted(
        (revision for repo in repos for revision in repo.revisions),
        key=lambda revision: revision.last_modified,
        reverse=True,
    )
    for revision in revisions:
        for file in revision.files:
            assets.setdefault(file.file_name, str(file.file_path))
    return assets


def _source_asset_path(
    source: str,
    filename: str,
    *,
    revision: str | None = None,
) -> str | None:
    if os.path.isdir(source):
        path = os.path.join(source, filename)
        return path if os.path.isfile(path) else None
    if revision is None:
        cached = _cached_source_assets(source).get(filename)
        if cached is not None and os.path.isfile(cached):
            return cached
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(source, filename, revision=revision)
    except Exception:
        return None


def _processor_values(
    source: str | None,
    config: Any,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Load plain processor parameters without architecture dispatch."""
    values: dict[str, Any] = {}
    if source:
        for filename in (
            "config.json",
            "processor_config.json",
            "preprocessor_config.json",
            "image_processor.json",
        ):
            path = _source_asset_path(source, filename, revision=revision)
            if path is not None:
                try:
                    with open(path, encoding="utf-8") as handle:
                        values.update(json.load(handle))
                except (OSError, ValueError):
                    _LOGGER.warning("Could not read processor config %s", path)
    image_processor = values.get("image_processor")
    if isinstance(image_processor, dict):
        values.update(image_processor)
    processor = values.get("processor")
    if isinstance(processor, dict):
        for transform in processor.get("transforms", []):
            operation = transform.get("operation", {}) if isinstance(transform, dict) else {}
            operation_type = operation.get("type")
            attrs = operation.get("attrs", {})
            if not isinstance(attrs, dict):
                continue
            for key, value in attrs.items():
                values.setdefault(key, value)
            if operation_type == "Resize":
                values.setdefault("do_resize", True)
                if "min_pixels" in attrs and "max_pixels" in attrs:
                    values.setdefault(
                        "size",
                        {
                            "shortest_edge": attrs["min_pixels"],
                            "longest_edge": attrs["max_pixels"],
                        },
                    )
            elif operation_type == "Rescale":
                values.setdefault("do_rescale", True)
                values.setdefault("rescale_factor", attrs.get("rescale_factor"))
            elif operation_type == "Normalize":
                values.setdefault("do_normalize", True)
                values.setdefault("image_mean", attrs.get("mean"))
                values.setdefault("image_std", attrs.get("std"))

    embedding = values.get("embd_layer")
    if isinstance(embedding, dict):
        image_embedding = embedding.get("image_embd_layer")
        if isinstance(image_embedding, dict):
            values.update(image_embedding)

    vision = getattr(config, "vision", None)
    for name in (
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
        "image_crop_size",
        "size",
    ):
        value = getattr(vision, name, None)
        if value is None:
            value = getattr(config, name, None)
        if value is not None:
            values.setdefault(name, value)
    values.setdefault("merge_size", values.get("spatial_merge_size"))
    values.setdefault("crop_size", values.get("image_crop_size"))
    if "dynamic_hd" in values:
        if values.get("use_hd_transform") is not True:
            raise ValueError(
                "Processor declares dynamic_hd but config.json does not explicitly enable "
                "use_hd_transform. Regenerate the package with the complete model config "
                "or register the missing dynamic-HD contract; fixed-resize fallback is unsafe."
            )
        values.setdefault("include_thumbnail", True)
        values.setdefault("thumbnail_order", "prepend")
        values.setdefault("mask_patch_size", values.get("patch_size", 14))
        values.setdefault("canvas_pad_value", 255)
    return values


_STATE_INPUT = re.compile(
    r"^past_key_values\.(?P<layer>\d+)\."
    r"(?:(?P<scope>self|cross)\.)?"
    r"(?P<role>key|value|conv_state|recurrent_state|ssm_state|"
    r"index_key|ple_conv_state|ple_context)$"
)
_STATIC_CACHE_PORT = re.compile(
    r"^(?P<updated>updated_)?(?P<role>key|value)_cache\.(?P<layer>\d+)$"
)
_REPLACE_ROLES = {
    "lightning_attention": {"recurrent_state"},
    "linear_attention": {
        "conv_state",
        "recurrent_state",
        "ple_conv_state",
        "ple_context",
    },
    "qwen_sparse_attention": {"index_key"},
    "conv": {"conv_state"},
    "mamba": {"conv_state", "ssm_state"},
    "mamba2": {"conv_state", "ssm_state"},
}
_NO_KV_LAYER_TYPES = set(_REPLACE_ROLES) - {"qwen_sparse_attention"}
_STATELESS_LAYER_TYPES = {"mlp", "moe"}


def _state_and_kv_pairs(
    decoder_inputs: list[_Port],
    decoder_outputs: list[_Port],
    config: Any,
) -> tuple[list[str], list[str], list[str], list[str], list[dict[str, str]]]:
    """Pair decoder state by declared port role and config layer type."""
    outputs = {port.name: port for port in decoder_outputs}
    layer_types = getattr(config, "layer_types", None)
    kv_inputs: list[str] = []
    kv_outputs: list[str] = []
    cross_kv_inputs: list[str] = []
    cross_kv_outputs: list[str] = []
    state_pairs: list[dict[str, str]] = []
    consumed_outputs: set[str] = set()
    for input_port in decoder_inputs:
        match = _STATE_INPUT.fullmatch(input_port.name)
        if match is None:
            raise ValueError(
                "Cannot classify decoder loop state port "
                f"{input_port.name!r}: it is neither routed data nor a registered "
                "past_key_values.<layer>.<role> contract. Regenerate the ONNX package "
                "with declared state port roles or register this decoder signature."
            )
        layer = int(match.group("layer"))
        scope = match.group("scope")
        role = match.group("role")
        output_name = f"present.{layer}.{scope + '.' if scope else ''}{role}"
        output_port = outputs.get(output_name)
        if output_port is None:
            raise ValueError(
                f"Decoder state input {input_port.name!r} declares role {role!r}, but "
                f"matching output {output_name!r} is absent. Regenerate the package "
                "with paired state I/O or register an explicit output mapping."
            )
        if input_port.dtype != output_port.dtype:
            raise ValueError(
                f"Decoder state pair {input_port.name!r} -> {output_name!r} has "
                f"dtype mismatch {input_port.dtype} vs {output_port.dtype}; regenerate "
                "the ONNX graph with a stable loop-state dtype."
            )
        consumed_outputs.add(output_name)

        layer_type = None
        if layer_types is not None:
            if layer >= len(layer_types):
                raise ValueError(
                    f"Decoder state port {input_port.name!r} references layer {layer}, "
                    f"but config.layer_types declares only {len(layer_types)} layers. "
                    "Regenerate the package from the matching config."
                )
            layer_type = str(layer_types[layer])

        if role in {"key", "value"}:
            if layer_type in _NO_KV_LAYER_TYPES or layer_type in _STATELESS_LAYER_TYPES:
                raise ValueError(
                    f"Decoder port {input_port.name!r} declares KV role {role!r}, but "
                    f"config.layer_types[{layer}]={layer_type!r} does not declare KV "
                    "append state. Regenerate the graph/config pair or register the "
                    "decoder state contract explicitly."
                )
            if scope == "cross":
                cross_kv_inputs.append(input_port.name)
                cross_kv_outputs.append(output_name)
            else:
                kv_inputs.append(input_port.name)
                kv_outputs.append(output_name)
        else:
            allowed = _REPLACE_ROLES.get(layer_type or "", set())
            if role not in allowed:
                why = (
                    "config.layer_types is absent"
                    if layer_types is None
                    else f"config.layer_types[{layer}]={layer_type!r}"
                )
                raise ValueError(
                    f"Decoder port {input_port.name!r} declares recurrent role {role!r}, "
                    f"but {why} does not register replace-state semantics. Regenerate "
                    "with explicit layer_types or add a structural decoder-state "
                    "registry entry; equal tensor shapes are not used to guess."
                )
            state_pairs.append(
                {
                    "input": input_port.name,
                    "output": output_name,
                    "init": "zeros",
                    "update": "replace",
                }
            )

    unpaired = [port.name for port in decoder_outputs if port.name not in consumed_outputs]
    if unpaired:
        raise ValueError(
            "Decoder exposes unpaired loop-state outputs "
            f"{unpaired}. Regenerate the package with present.<layer>.<role> outputs "
            "matching declared past_key_values inputs, or register an explicit mapping."
        )
    return kv_inputs, kv_outputs, cross_kv_inputs, cross_kv_outputs, state_pairs


def _static_cache_io(
    decoder_inputs: list[_Port],
    decoder_outputs: list[_Port],
) -> dict[str, Any] | None:
    """Return the explicit TensorScatter static-cache ABI from exported ports."""
    inputs: dict[tuple[int, str], str] = {}
    outputs: dict[tuple[int, str], str] = {}
    for port in decoder_inputs:
        match = _STATIC_CACHE_PORT.fullmatch(port.name)
        if match is not None and match.group("updated") is None:
            inputs[(int(match.group("layer")), match.group("role"))] = port.name
    for port in decoder_outputs:
        match = _STATIC_CACHE_PORT.fullmatch(port.name)
        if match is not None and match.group("updated") is not None:
            outputs[(int(match.group("layer")), match.group("role"))] = port.name
    if not inputs and not outputs:
        return None

    layers = sorted({layer for layer, _ in inputs} | {layer for layer, _ in outputs})
    missing = [
        f"{kind}.{layer}.{role}"
        for layer in layers
        for role in ("key", "value")
        for kind, ports in (("input", inputs), ("output", outputs))
        if (layer, role) not in ports
    ]
    input_names = {port.name for port in decoder_inputs}
    for control in (STATIC_CACHE_WRITE_INDICES, STATIC_CACHE_KV_SEQUENCE_LENGTH):
        if control not in input_names:
            missing.append(f"input.{control}")
    if missing:
        raise ValueError(
            "Cannot describe the static-cache ABI because the exported "
            f"TensorScatter ports are incomplete: {missing}"
        )
    return {
        "write_indices_input": STATIC_CACHE_WRITE_INDICES,
        "kv_sequence_length_input": STATIC_CACHE_KV_SEQUENCE_LENGTH,
        "key_cache_inputs": [inputs[(layer, "key")] for layer in layers],
        "value_cache_inputs": [inputs[(layer, "value")] for layer in layers],
        "key_cache_outputs": [outputs[(layer, "key")] for layer in layers],
        "value_cache_outputs": [outputs[(layer, "value")] for layer in layers],
    }


@dataclasses.dataclass(frozen=True)
class _PositionProgram:
    rank: int
    axes: tuple[str, ...]
    generation: str
    continuation: str
    matches: Callable[[Any], bool]
    sections_attribute: str | None = None


_POSITION_PROGRAM_REGISTRY = (
    _PositionProgram(
        rank=1,
        axes=("sequence",),
        generation="linear",
        continuation="linear_increment",
        matches=lambda config: True,
    ),
    _PositionProgram(
        rank=3,
        axes=("temporal", "height", "width"),
        generation="processor_coordinates",
        continuation="carry_max",
        matches=lambda config: (
            bool(getattr(config, "mrope_interleaved", False))
            and bool(getattr(config, "mrope_section", None))
        ),
        sections_attribute="mrope_section",
    ),
    _PositionProgram(
        rank=4,
        axes=("text", "temporal", "height", "width"),
        generation="processor_coordinates",
        continuation="carry_state",
        matches=lambda config: (
            getattr(config, "model_type", None) in {"qwen4_exp", "qwen4_exp_text"}
            and bool(getattr(config, "mrope_interleaved", False))
            and bool(getattr(config, "mrope_section", None))
        ),
        sections_attribute="mrope_section",
    ),
)


def _positions_from_registry(position: _Port, config: Any) -> dict[str, Any]:
    if position.rank == 3:
        semantic_rank = 4 if position.dims[0] == 4 else 3
    elif position.rank == 2:
        semantic_rank = 1
    else:
        raise ValueError(
            f"Cannot emit position metadata for decoder port {position.name!r}: "
            f"expected a registered rank-2 or rank-3 tensor, got shape {position.dims}. "
            "Regenerate the decoder with a declared position contract or register "
            "the new layout."
        )
    for program in _POSITION_PROGRAM_REGISTRY:
        if program.rank != semantic_rank or not program.matches(config):
            continue
        positions: dict[str, Any] = {
            "input": position.name,
            "rank": program.rank,
            "tensor_rank": position.rank,
            "dtype": position.dtype,
            "generation": program.generation,
            "continuation": program.continuation,
            "axes": list(program.axes),
        }
        if program.sections_attribute is not None:
            sections = getattr(config, program.sections_attribute)
            positions["sections"] = [int(section) for section in sections]
        return positions
    raise ValueError(
        f"Cannot emit position metadata for decoder port {position.name!r} with "
        f"shape {position.dims}: no position registry entry matches the explicit "
        "config. Rank-3 axes and multi-axis continuation are never guessed. Regenerate with "
        "mrope_interleaved/mrope_section declarations or register this position contract."
    )


def _decoder_io(
    decoder: Any,
    routed_inputs: set[str],
    config: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    inputs = [_port(value) for value in decoder.graph.inputs]
    outputs = [_port(value) for value in decoder.graph.outputs]
    io: dict[str, Any] = {
        "inputs": [_port_metadata(port) for port in inputs],
        "outputs": [_port_metadata(port) for port in outputs],
        "kv_ownership": "owned",
    }

    routed = [port for port in inputs if port.name in routed_inputs]
    embedded = next(
        (
            port
            for port in routed
            if _is_float(port)
            and port.rank == 3
            and port.name != "encoder_hidden_states"
            and _STATIC_CACHE_PORT.fullmatch(port.name) is None
        ),
        None,
    )
    if embedded is None:
        embedded = _select_one(
            inputs,
            lambda port: (
                _is_float(port)
                and port.rank == 3
                and port.name != "encoder_hidden_states"
                and _STATIC_CACHE_PORT.fullmatch(port.name) is None
            ),
        )
    if embedded is not None:
        io["inputs_embeds_input"] = embedded.name
        io["sequence_source"] = "inputs_embeds"

    input_by_name = {port.name: port for port in inputs}
    output_by_name = {port.name: port for port in outputs}
    attention_mask = input_by_name.get("attention_mask")
    if attention_mask is not None:
        io["attention_mask_input"] = attention_mask.name

    position = input_by_name.get("position_ids")
    if position is not None:
        io["position_ids_input"] = position.name

    token = input_by_name.get("input_ids") or _select_one(
        inputs,
        lambda port: (
            _is_integer(port)
            and port.rank == 2
            and port.name not in {"attention_mask", "position_ids"}
            and _STATE_INPUT.fullmatch(port.name) is None
        ),
    )
    if token is not None:
        io["token_input"] = token.name
        io.setdefault("sequence_source", "token_ids")

    logits = output_by_name.get("logits")
    if logits is None:
        raise ValueError(
            "Cannot emit native VLM decoder metadata because the graph has no "
            "'logits' output. Regenerate the Mobius decoder with its declared "
            "logits role or register an explicit decoder I/O contract."
        )
    io["logits_output"] = logits.name
    encoder_hidden_states = input_by_name.get("encoder_hidden_states")
    if encoder_hidden_states is not None:
        io["encoder_hidden_states_input"] = encoder_hidden_states.name

    static_cache = _static_cache_io(inputs, outputs)
    if static_cache is not None:
        io["static_cache"] = static_cache
    static_names = (
        {
            static_cache["write_indices_input"],
            static_cache["kv_sequence_length_input"],
            *static_cache["key_cache_inputs"],
            *static_cache["value_cache_inputs"],
            *static_cache["key_cache_outputs"],
            *static_cache["value_cache_outputs"],
        }
        if static_cache is not None
        else set()
    )
    core_inputs = routed_inputs | {
        port.name
        for port in (attention_mask, position, token, embedded, encoder_hidden_states)
        if port is not None
    }
    state_inputs = [
        port
        for port in inputs
        if port.name not in core_inputs
        and port.name not in static_names
        and _STATE_INPUT.fullmatch(port.name) is not None
    ]
    state_outputs = [
        port
        for port in outputs
        if port.name != logits.name
        and port.name not in static_names
        and port.name.startswith("present.")
    ]
    (
        kv_inputs,
        kv_outputs,
        cross_kv_inputs,
        cross_kv_outputs,
        state_pairs,
    ) = _state_and_kv_pairs(state_inputs, state_outputs, config)
    if kv_inputs:
        io["kv_inputs"] = kv_inputs
        io["kv_outputs"] = kv_outputs
        io["kv_update"] = "append"
    if cross_kv_inputs:
        io["cross_kv_inputs"] = cross_kv_inputs
        io["cross_kv_outputs"] = cross_kv_outputs
    if state_pairs:
        io["state_pairs"] = state_pairs
    past_position_ids = input_by_name.get("past_position_ids")
    present_position_ids = output_by_name.get("present_position_ids")
    if (past_position_ids is None) != (present_position_ids is None):
        raise ValueError(
            "Decoder position history must expose paired past_position_ids and "
            "present_position_ids ports"
        )
    if past_position_ids is not None and present_position_ids is not None:
        if past_position_ids.dtype != present_position_ids.dtype:
            raise ValueError(
                "Decoder position history must preserve dtype across decode steps"
            )
        io.setdefault("state_pairs", []).append(
            {
                "input": past_position_ids.name,
                "output": present_position_ids.name,
                "init": "zeros",
                "update": "replace",
            }
        )
    consumed_outputs = set(kv_outputs) | set(cross_kv_outputs)
    hidden_outputs = [
        port
        for port in outputs
        if port.name != logits.name
        and port.name not in static_names
        and port.name not in consumed_outputs
        and _is_float(port)
    ]
    if len(hidden_outputs) == 1:
        io["hidden_output"] = hidden_outputs[0].name

    positions = _positions_from_registry(position, config) if position is not None else None
    return io, positions


def _component_filenames(pkg: Any) -> dict[str, str]:
    multiple = len(pkg) > 1
    return {name: f"{name}/model.onnx" if multiple else "model.onnx" for name in pkg}


def _sequence_decoder_inputs(decoder_ports: dict[str, list[_Port]]) -> set[str]:
    """Find decoder inputs whose leading dimensions track the logits sequence."""
    logits = next(
        (
            port
            for port in decoder_ports["outputs"]
            if port.name == "logits" and port.rank is not None and port.rank >= 2
        ),
        None,
    )
    if logits is None:
        return set()
    return {
        port.name
        for port in decoder_ports["inputs"]
        if port.rank is not None
        and port.rank >= 2
        and _same_dim(port.dims[0], logits.dims[0])
        and _same_dim(port.dims[1], logits.dims[1])
        and _is_float(port)
    }


def _component_token_input(component_ports: dict[str, list[_Port]]) -> _Port | None:
    """Resolve a single structural token stream for an upstream step component."""
    candidates = [
        port for port in component_ports["inputs"] if _is_integer(port) and port.rank == 2
    ]
    return candidates[0] if len(candidates) == 1 else None


def _input_source_map(
    *,
    ports: dict[str, dict[str, list[_Port]]],
    dataflow: list[dict[str, Any]],
    models: dict[str, Any],
    decoder_name: str,
    image_endpoints: set[str],
) -> dict[str, dict[str, Any]]:
    incoming = {edge["to"]: edge for edge in dataflow}
    sources: dict[str, dict[str, Any]] = {}
    for component, component_ports in ports.items():
        for port in component_ports["inputs"]:
            endpoint = f"{component}.{port.name}"
            edge = incoming.get(endpoint)
            if edge is not None:
                sources[endpoint] = {"kind": "dataflow", "from": edge["from"]}
            elif endpoint in image_endpoints:
                sources[endpoint] = {
                    "kind": "generated",
                    "generator": "image_preprocessing",
                }
            else:
                sources[endpoint] = {"kind": "external", "input": "request"}

    decoder_io = models[decoder_name]["io"]
    generated_ports = {
        decoder_io.get("attention_mask_input"): "attention_mask",
        decoder_io.get("position_ids_input"): "positions",
    }
    for port, generator in generated_ports.items():
        if port is not None:
            sources[f"{decoder_name}.{port}"] = {
                "kind": "generated",
                "generator": generator,
            }

    for input_field, output_field in (
        ("kv_inputs", "kv_outputs"),
        ("cross_kv_inputs", "cross_kv_outputs"),
    ):
        for input_name, output_name in zip(
            decoder_io.get(input_field, []),
            decoder_io.get(output_field, []),
        ):
            sources[f"{decoder_name}.{input_name}"] = {
                "kind": "stateful",
                "from": f"{decoder_name}.{output_name}",
                "update": decoder_io.get("kv_update", "append"),
            }
    for pair in decoder_io.get("state_pairs", []):
        sources[f"{decoder_name}.{pair['input']}"] = {
            "kind": "stateful",
            "from": f"{decoder_name}.{pair['output']}",
            "update": pair["update"],
        }
    static_cache = decoder_io.get("static_cache")
    if static_cache is not None:
        sources[f"{decoder_name}.{static_cache['write_indices_input']}"] = {
            "kind": "generated",
            "generator": "static_cache_write_indices",
        }
        sources[f"{decoder_name}.{static_cache['kv_sequence_length_input']}"] = {
            "kind": "generated",
            "generator": "kv_sequence_length",
        }
        for input_name, output_name in zip(
            static_cache["key_cache_inputs"] + static_cache["value_cache_inputs"],
            static_cache["key_cache_outputs"] + static_cache["value_cache_outputs"],
        ):
            sources[f"{decoder_name}.{input_name}"] = {
                "kind": "stateful",
                "from": f"{decoder_name}.{output_name}",
                "update": "shared_buffer",
            }
    return sources


def _annotate_component_inputs(
    models: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> None:
    for component, model in models.items():
        for port in model["io"]["inputs"]:
            port["source"] = sources[f"{component}.{port['name']}"]


def _closure_error(endpoint: str, why: str, how: str) -> ValueError:
    return ValueError(
        f"What: executable metadata closure is invalid at '{endpoint}'. "
        f"Why: {why} How to fix: {how}"
    )


def validate_executable_closure(pkg: Any, metadata: dict[str, Any]) -> None:
    """Reject a native sidecar that cannot bind every real graph input exactly once."""
    pipeline = metadata.get("pipeline")
    if not isinstance(pipeline, dict):
        raise _closure_error(
            "pipeline.models",
            "the metadata has no pipeline object.",
            "emit the native multi-model pipeline contract before packaging.",
        )
    declared_models = pipeline.get("models")
    if not isinstance(declared_models, dict):
        raise _closure_error(
            "pipeline.models",
            "the metadata has no component declarations.",
            "emit one pipeline.models entry for every ONNX graph.",
        )

    graph_ports = {
        component: {
            "inputs": {_port(value).name: _port(value) for value in model.graph.inputs},
            "outputs": {_port(value).name: _port(value) for value in model.graph.outputs},
        }
        for component, model in pkg.items()
    }
    incoming: dict[str, list[dict[str, Any]]] = {}
    for edge in pipeline.get("dataflow", []):
        source_endpoint = edge.get("from")
        target_endpoint = edge.get("to")
        if not isinstance(source_endpoint, str) or not isinstance(target_endpoint, str):
            raise _closure_error(
                "pipeline.dataflow",
                "a dataflow edge does not declare string from/to endpoints.",
                "emit every edge as exact component.output and component.input endpoints.",
            )
        source_component, separator, source_name = source_endpoint.partition(".")
        target_component, target_separator, target_name = target_endpoint.partition(".")
        if not separator or not target_separator:
            raise _closure_error(
                target_endpoint,
                "the edge endpoint is not in component.port form.",
                "emit exact component.output and component.input endpoints.",
            )
        source = graph_ports.get(source_component, {}).get("outputs", {}).get(source_name)
        target = graph_ports.get(target_component, {}).get("inputs", {}).get(target_name)
        if source is None:
            raise _closure_error(
                source_endpoint,
                "the declared producer is not an output of its ONNX graph.",
                "regenerate dataflow from the saved graph outputs.",
            )
        if target is None:
            raise _closure_error(
                target_endpoint,
                "the declared consumer is not an input of its ONNX graph.",
                "regenerate dataflow from the saved graph inputs.",
            )
        if source.dtype != target.dtype or source.rank != target.rank:
            raise _closure_error(
                target_endpoint,
                f"edge {source_endpoint} has dtype/rank {source.dtype}/{source.rank}, "
                f"but the consumer requires {target.dtype}/{target.rank}.",
                "insert an explicit typed transform or remove the incompatible edge.",
            )
        if edge.get("dtype") != source.dtype:
            raise _closure_error(
                target_endpoint,
                f"the edge declares dtype {edge.get('dtype')}, "
                f"but the ONNX ports use {source.dtype}.",
                "derive the edge dtype directly from the matched graph ports.",
            )
        incoming.setdefault(target_endpoint, []).append(edge)

    for component, component_ports in graph_ports.items():
        model = declared_models.get(component)
        if not isinstance(model, dict) or not isinstance(model.get("io"), dict):
            first_port = next(iter(component_ports["inputs"]), "<io>")
            raise _closure_error(
                f"{component}.{first_port}",
                "the component has no explicit io contract.",
                "emit typed inputs and outputs from ONNX graph introspection.",
            )
        io = model["io"]
        input_specs = io.get("inputs")
        output_specs = io.get("outputs")
        if not isinstance(input_specs, list) or not isinstance(output_specs, list):
            first_port = next(iter(component_ports["inputs"]), "<io>")
            raise _closure_error(
                f"{component}.{first_port}",
                "the component io contract does not contain typed input/output lists.",
                "emit name, dtype, rank, and shape for every real graph port.",
            )
        declared_inputs = {
            spec.get("name"): spec for spec in input_specs if isinstance(spec, dict)
        }
        declared_outputs = {
            spec.get("name"): spec for spec in output_specs if isinstance(spec, dict)
        }
        for direction, real_ports, declared in (
            ("input", component_ports["inputs"], declared_inputs),
            ("output", component_ports["outputs"], declared_outputs),
        ):
            if set(declared) != set(real_ports):
                missing = sorted(set(real_ports) - set(declared))
                extra = sorted(set(declared) - set(real_ports))
                port = (missing or extra or ["<io>"])[0]
                raise _closure_error(
                    f"{component}.{port}",
                    f"declared {direction} ports differ from the ONNX graph "
                    f"(missing={missing}, extra={extra}).",
                    "regenerate component io from the packaged ONNX graph.",
                )
            for name, real in real_ports.items():
                spec = declared[name]
                if (
                    spec.get("dtype") != real.dtype
                    or spec.get("rank") != real.rank
                    or spec.get("shape") != _shape_metadata(real)
                ):
                    raise _closure_error(
                        f"{component}.{name}",
                        "the declared dtype, rank, or shape does not match the ONNX graph.",
                        "regenerate the typed port declaration from graph introspection.",
                    )

        for name in component_ports["inputs"]:
            endpoint = f"{component}.{name}"
            spec = declared_inputs[name]
            edges = incoming.get(endpoint, [])
            source = spec.get("source")
            if not isinstance(source, dict):
                raise _closure_error(
                    endpoint,
                    "the required graph input has no declared source classification.",
                    "classify it as external, generated, stateful, defaulted, or dataflow.",
                )
            kind = source.get("kind")
            if kind == "dataflow":
                if len(edges) != 1 or source.get("from") != edges[0]["from"]:
                    raise _closure_error(
                        endpoint,
                        f"the input declares dataflow source {source.get('from')!r}, "
                        f"but {len(edges)} matching edge(s) exist.",
                        "emit exactly one compatible edge and reference its producer.",
                    )
            elif kind in {"external", "generated", "stateful", "defaulted"}:
                if edges:
                    raise _closure_error(
                        endpoint,
                        f"the input is classified as {kind!r} but also has a dataflow edge.",
                        "keep exactly one source category for every required graph input.",
                    )
                if kind == "defaulted" and "value" not in source:
                    raise _closure_error(
                        endpoint,
                        "a defaulted input does not declare its default value.",
                        "emit the explicit typed default value or use another source category.",
                    )
            else:
                raise _closure_error(
                    endpoint,
                    f"the source category {kind!r} is not executable.",
                    "use external, generated, stateful, defaulted, or dataflow.",
                )


def add_adapter_service_to_metadata(
    metadata: dict[str, Any],
    pkg: Any,
    output_dir: str,
) -> dict[str, Any]:
    """Attach the exact generic adapter catalog and saved artifact references."""
    artifacts = getattr(pkg, "adapter_artifacts", {})
    if not artifacts:
        return metadata
    workflow_value = metadata.get("pipeline", {}).get("workflow")
    workflow = workflow_value if isinstance(workflow_value, dict) else None
    manifest = getattr(pkg, "adapter_target_manifest", None)
    if manifest is None:
        raise ValueError("parameter adapters require an authoritative adapter target manifest")
    options = pkg.adapter_service_options
    inputs = workflow.setdefault("inputs", {}) if workflow is not None else {}

    def compatible_input(
        name: str,
        *,
        dtype: str,
        shape: list[str | int],
        role: str | None,
    ) -> bool:
        declaration = inputs.get(name)
        if not isinstance(declaration, dict):
            return False
        contract = declaration.get("contract", {})
        semantic_role = declaration.get("role", {})
        return (
            contract.get("dtype") == dtype
            and contract.get("shape") == shape
            # A selection tensor the service reads for every batch it plans has
            # no package-side escape, so admission derivation stamps it required
            # and this stays a required-input check either way.
            and declaration.get("required", True)
            and declaration.get("source", {}).get("kind") in {"request", "application"}
            and (
                role is None
                or semantic_role == {"kind": "runtime", "version": "1.0", "role": role}
            )
        )

    def ensure_input(
        name: str,
        *,
        dtype: str,
        shape: list[str | int],
        role: str | None,
        source: dict[str, str] | None = None,
    ) -> None:
        if name not in inputs:
            inputs[name] = {
                "contract": {"dtype": dtype, "shape": shape},
                "role": (
                    {"kind": "runtime", "version": "1.0", "role": role}
                    if role is not None
                    else {"kind": "opaque"}
                ),
                "source": source or {"kind": "request"},
                # A selection tensor has no package-side default and no
                # presence gate: the service reads it for every batch it plans,
                # so the caller supplies it or the request is not admissible.
                "required": True,
            }
        if not compatible_input(name, dtype=dtype, shape=shape, role=role):
            raise ValueError(
                f"adapter {role} must reference a required "
                "request/application-sourced "
                f"{dtype}{shape} workflow input"
            )

    active = options.active
    if workflow is not None:
        ensure_input(
            options.segments,
            dtype="int64",
            shape=["batch", options.max_adapters],
            role="adapter_segments",
        )
        ensure_input(
            options.adapter_counts,
            dtype="int64",
            shape=["batch"],
            role="adapter_counts",
        )
        ensure_input(
            options.scales,
            dtype="float32",
            shape=["batch", options.max_adapters],
            role="adapter_scales",
        )
        if active is not None:
            ensure_input(
                active,
                dtype="bool",
                shape=["batch"],
                role="adapter_active",
            )

    catalog = pkg.save_adapter_artifacts(output_dir)
    if workflow is not None:
        workflow.pop("adapters", None)
    metadata["adapters"] = {
        "target_manifest": pkg.adapter_target_manifest_metadata(),
        "discovery_fallback": options.discovery_fallback,
        "selection": {
            "segments": options.segments,
            "adapter_counts": options.adapter_counts,
            "scales": options.scales,
            **({"active": active} if active is not None else {}),
            "max_adapters": options.max_adapters,
        },
        "application_capability": options.application_capability,
        "portable_fallback": options.portable_fallback,
        "cache": {
            "max_entries": options.cache_max_entries,
            "eviction": "lru",
        },
        "planning": {
            "bucket_by_adapter_set": options.bucket_by_adapter_set,
            "stable_buffers": options.stable_buffers,
            "invalidate_capture_on_eviction": (options.invalidate_capture_on_eviction),
        },
        "artifacts": catalog,
    }
    if workflow is not None:
        capabilities = workflow.setdefault("manifest", {}).setdefault("capabilities", [])
        for capability in ("parameter_adapters", "heterogeneous_adapter_batching"):
            if capability not in capabilities:
                capabilities.append(capability)
        # Selection inputs carry exactly one entry per in-flight request, so
        # they are request-aligned by construction. Stamping here rather than
        # inside `ensure_input` also covers the declarations a producer wrote
        # by hand before attaching the adapter service.
        declare_request_alignment(workflow)
        declare_input_admission(workflow)
    return metadata


def _topological_order(
    names: Iterable[str],
    edges: list[dict[str, Any]],
) -> list[str]:
    names = list(names)
    incoming = dict.fromkeys(names, 0)
    outgoing: dict[str, list[str]] = {name: [] for name in names}
    for edge in edges:
        source = edge["from"].split(".", 1)[0]
        target = edge["to"].split(".", 1)[0]
        if source == target:
            continue
        outgoing[source].append(target)
        incoming[target] += 1
    ready = [name for name in names if incoming[name] == 0]
    ordered: list[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for target in outgoing[name]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if len(ordered) != len(names):
        raise ValueError("Component dataflow contains a cycle")
    return ordered


def is_native_vlm_package(pkg: Any) -> bool:
    """Return whether a package must use native VLM emission.

    Processor support is deliberately not checked here. A VLM-shaped package
    must enter the native emitter so an unsupported signature fails clearly
    instead of silently receiving generic decoder metadata.
    """
    try:
        names = set(pkg)
    except (AttributeError, TypeError):
        return False
    return "vision_encoder" in names


def build_native_vlm_package_metadata(
    pkg: Any,
    *,
    config: Any,
    source: str | None = None,
    revision: str | None = None,
    decoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe a native VLM package by inspecting every component graph.

    This is mobius's **internal** structural descriptor, not a publishable
    document: its ``pipeline`` key is a ``models``/``dataflow``/``strategy``
    view used to reason about component wiring and validate the executable
    closure. It is consumed by
    :func:`~mobius.integrations.onnx_genai.workflow_metadata.build_vlm_workflow_metadata`,
    which publishes the workflow and the ``preprocessing`` program. Use
    :func:`~mobius.integrations.onnx_genai.workflow_metadata.write_native_vlm_package_metadata`
    to emit the package contract.

    Processor selection is registry-driven from graph rank/dtype/shape
    signatures. No model type, architecture name, or model-name branch
    participates in dispatch.
    """
    try:
        component_names = set(pkg)
    except (AttributeError, TypeError):
        component_names = set()
    required_components = {"vision_encoder", "embedding", "decoder"}
    missing_components = sorted(required_components - component_names)
    if missing_components:
        raise ValueError(
            "Cannot emit native VLM metadata: missing required component(s) "
            f"{missing_components}. Why: executable native VLM metadata requires a "
            "vision_encoder to produce image features, an embedding component to route "
            "them into token embeddings, and a decoder to consume those embeddings; "
            "partial packages cannot define complete graph I/O or dataflow. How to fix: "
            "regenerate the package with the native multimodal task so all three "
            "components are present, or use a component-specific non-VLM exporter."
        )

    filenames = _component_filenames(pkg)
    ports = {
        name: {
            "inputs": [_port(value) for value in model.graph.inputs],
            "outputs": [_port(value) for value in model.graph.outputs],
        }
        for name, model in pkg.items()
    }
    dataflow: list[dict[str, Any]] = []
    for target_name, target_ports in ports.items():
        for target_port in target_ports["inputs"]:
            matches = [
                (source_name, source_port)
                for source_name, source_ports in ports.items()
                if source_name != target_name
                for source_port in source_ports["outputs"]
                if source_port.name == target_port.name
                and _ports_match_for_dataflow(source_port, target_port)
            ]
            if len(matches) > 1:
                producers = [f"{name}.{port.name}" for name, port in matches]
                raise _closure_error(
                    f"{target_name}.{target_port.name}",
                    f"multiple compatible graph outputs could feed this input: {producers}.",
                    "declare a unique structural producer or rename the ambiguous graph ports.",
                )
            if matches:
                source_name, source_port = matches[0]
                dataflow.append(
                    {
                        "from": f"{source_name}.{source_port.name}",
                        "to": f"{target_name}.{target_port.name}",
                        "dtype": source_port.dtype,
                        "device_transfer": False,
                    }
                )

    decoder_name = "decoder"
    routed_decoder_inputs = {
        edge["to"].split(".", 1)[1]
        for edge in dataflow
        if edge["to"].startswith(f"{decoder_name}.")
    }
    processor_values = _processor_values(source, config, revision=revision)
    image_program = _resolve_image_program(pkg["vision_encoder"], processor_values)
    decoder_io, positions = _decoder_io(pkg[decoder_name], routed_decoder_inputs, config)

    preprocessing_outputs = []
    processor_summaries = []
    for port, content, pad_value in image_program.bindings:
        endpoint = f"vision_encoder.{port.name}"
        output: dict[str, Any] = {
            "name": endpoint,
            "content": content,
            "dtype": port.dtype,
        }
        if pad_value is not None:
            output["pad_value"] = pad_value
        preprocessing_outputs.append(output)
        if content in image_program.summary_contents:
            processor_summaries.append(endpoint)

    if positions is not None and positions["rank"] > 1 and processor_summaries:
        positions["processor_summaries"] = processor_summaries

    sequence_decoder_inputs = _sequence_decoder_inputs(ports[decoder_name])
    downstream_to_decoder = {
        edge["from"].split(".", 1)[0]
        for edge in dataflow
        if edge["to"].startswith(f"{decoder_name}.")
        and edge["to"].split(".", 1)[1] in sequence_decoder_inputs
    }
    models: dict[str, Any] = {}
    phases: dict[str, Any] = {}
    for name in pkg:
        if name == decoder_name:
            role = "decoder"
            run_on = "every_step"
            io = decoder_io
        elif name == "vision_encoder":
            role = "vision_encoder"
            run_on = "prompt_only"
            io = {
                "inputs": [_port_metadata(port) for port in ports[name]["inputs"]],
                "outputs": [_port_metadata(port) for port in ports[name]["outputs"]],
            }
        else:
            role = "audio_encoder" if name == "audio_encoder" else "encoder"
            run_on = "every_step" if name in downstream_to_decoder else "prompt_only"
            io = {
                "inputs": [_port_metadata(port) for port in ports[name]["inputs"]],
                "outputs": [_port_metadata(port) for port in ports[name]["outputs"]],
            }
            if run_on == "every_step":
                token_input = _component_token_input(ports[name])
                if token_input is None:
                    raise _closure_error(
                        f"{name}.<token_input>",
                        "the sequence-dependent component must run every step, but its "
                        "graph does not expose one structurally unique rank-2 integer token input.",
                        "declare a unique token-stream input or add a structural registry entry.",
                    )
                io["token_input"] = token_input.name
                io["sequence_source"] = "token_ids"
        models[name] = {
            "filename": filenames[name],
            "type": role,
            "io": io,
        }
        optional_inputs = {
            value.name: contract
            for value in pkg[name].graph.inputs
            if (contract := optional_input_contract(value)) is not None
        }
        if optional_inputs:
            io["optional_inputs"] = optional_inputs
        if name == decoder_name:
            models[name]["tokenizer"] = "tokenizer.json"
        phases[name] = {"run_on": run_on}
        if presence := component_presence(pkg[name].graph):
            phases[name]["when_present"] = presence

    image_endpoints = {
        output["name"] for output in preprocessing_outputs if not output.get("optional", False)
    }
    sources = _input_source_map(
        ports=ports,
        dataflow=dataflow,
        models=models,
        decoder_name=decoder_name,
        image_endpoints=image_endpoints,
    )
    _annotate_component_inputs(models, sources)

    stages = []
    for name in _topological_order(pkg.keys(), dataflow):
        strategy = (
            {"kind": "autoregressive", "decoder": name}
            if name == decoder_name
            else {"kind": "single_pass", "model": name}
        )
        stages.append(
            {
                "name": f"run_{name}",
                "strategy": strategy,
            }
        )

    vision_config: dict[str, Any] = {
        "image_placeholder_token_id": getattr(config, "image_token_id", None)
        or getattr(getattr(config, "vision", None), "image_token_id", None),
        "image_token_id": getattr(config, "image_token_id", None)
        or getattr(getattr(config, "vision", None), "image_token_id", None),
        "placeholder_per_image": True,
        "token_count_source": image_program.token_count_source,
    }
    if processor_summaries:
        vision_config["processor_summaries"] = processor_summaries
    if image_program.vision_properties is not None:
        vision_config.update(image_program.vision_properties(processor_values))
    vision_config = {key: value for key, value in vision_config.items() if value is not None}

    metadata = dict(decoder_metadata or {})
    metadata.setdefault("schema_version", "v1")
    capabilities = list(metadata.get("required_capabilities", []))
    for capability in (
        "image_preprocessing_program",
        "autoregressive_every_step_components",
    ):
        if capability not in capabilities:
            capabilities.append(capability)
    if len(preprocessing_outputs) > 1 and "packed_image_outputs" not in capabilities:
        capabilities.append("packed_image_outputs")
    if positions is not None and "position_program" not in capabilities:
        capabilities.append("position_program")
    if positions is not None and positions["rank"] > 1:
        capabilities.append("multi_axis_positions")
    if decoder_io.get("state_pairs"):
        capabilities.append("loop_carried_state")
    if decoder_io.get("token_input") and decoder_io.get("inputs_embeds_input"):
        capabilities.append("dual_sequence_inputs")
    metadata["required_capabilities"] = capabilities
    metadata["preprocessing"] = {
        "image": {
            "transforms": image_program.transforms(config, processor_values),
            "outputs": preprocessing_outputs,
        }
    }
    # Bind every declared output to the processor-local value that produces it,
    # so the runtime never has to guess which transform an output came from.
    _name_image_preprocessing_program(metadata["preprocessing"]["image"])
    metadata["pipeline"] = {
        "models": models,
        "dataflow": dataflow,
        "strategy": {"kind": "composite", "stages": stages},
        "vision": vision_config,
    }
    if positions is not None:
        metadata["pipeline"]["positions"] = positions
    validate_executable_closure(pkg, metadata)
    return metadata


def _copy_runtime_assets(
    output_dir: str,
    source: str | None,
    names: Sequence[str] = _RUNTIME_ASSET_NAMES,
    *,
    revision: str | None = None,
) -> dict[str, str]:
    if not source:
        return {}
    os.makedirs(output_dir, exist_ok=True)
    for filename in names:
        source_path = _source_asset_path(source, filename, revision=revision)
        if source_path is not None:
            shutil.copy2(source_path, os.path.join(output_dir, filename))

    template_path = os.path.join(output_dir, "chat_template.jinja")
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.isfile(template_path) and os.path.isfile(tokenizer_config_path):
        try:
            with open(tokenizer_config_path, encoding="utf-8") as handle:
                chat_template = json.load(handle).get("chat_template")
            if chat_template:
                Path(template_path).write_text(chat_template, encoding="utf-8")
        except (OSError, ValueError):
            _LOGGER.warning("Could not extract chat_template from %s", tokenizer_config_path)

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_path):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                source,
                use_fast=True,
                revision=revision,
            )
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is not None:
                backend.save(tokenizer_path)
            chat_template = getattr(tokenizer, "chat_template", None)
            if chat_template and not os.path.isfile(template_path):
                Path(template_path).write_text(chat_template, encoding="utf-8")
        except Exception as error:
            _LOGGER.warning(
                "Could not materialize tokenizer assets from %r: %s", source, error
            )

    apply_asset_patches(output_dir)

    return {
        Path(filename).stem: os.path.join(output_dir, filename)
        for filename in names
        if os.path.isfile(os.path.join(output_dir, filename))
    }


@dataclasses.dataclass(frozen=True)
class SchedulerConfig:
    """Diffusion noise-schedule parameters for an onnx-genai scheduler."""

    kind: str = "ddim"
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"
    clip_sample: bool = False
    clip_sample_range: float = 1.0
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    timestep_spacing: str = "leading"
    rescale_betas_zero_snr: bool = False
    snr_shift_scale: float = 1.0
    use_karras_sigmas: bool = False
    use_exponential_sigmas: bool = False
    shift: float | None = None
    base_image_seq_len: int | None = None
    max_image_seq_len: int | None = None
    base_shift: float | None = None
    max_shift: float | None = None
    shift_terminal: float | None = None
    use_dynamic_shifting: bool = False
    time_shift_type: str | None = None
    invert_sigmas: bool = False
    stochastic_sampling: bool = False
    algorithm_type: str = "dpmsolver++"
    solver_order: int = 2
    solver_type: str = "midpoint"
    lower_order_final: bool = True
    final_sigmas_type: str = "zero"

    def to_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "kind": self.kind,
            "num_train_timesteps": self.num_train_timesteps,
            "prediction_type": self.prediction_type,
        }
        if self.kind != "flow_match_euler":
            meta.update(
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                beta_schedule=self.beta_schedule,
                clip_sample=self.clip_sample,
                clip_sample_range=self.clip_sample_range,
                set_alpha_to_one=self.set_alpha_to_one,
                steps_offset=self.steps_offset,
                timestep_spacing=self.timestep_spacing,
                rescale_betas_zero_snr=self.rescale_betas_zero_snr,
                snr_shift_scale=self.snr_shift_scale,
            )
        if self.use_karras_sigmas:
            meta["use_karras_sigmas"] = True
        if self.use_exponential_sigmas:
            meta["use_exponential_sigmas"] = True
        for name in (
            "shift",
            "base_image_seq_len",
            "max_image_seq_len",
            "base_shift",
            "max_shift",
            "shift_terminal",
            "time_shift_type",
        ):
            value = getattr(self, name)
            if value is not None:
                meta[name] = value
        if self.use_dynamic_shifting:
            meta["use_dynamic_shifting"] = True
        if self.invert_sigmas:
            meta["invert_sigmas"] = True
        if self.stochastic_sampling:
            meta["stochastic_sampling"] = True
        return meta

    @classmethod
    def from_diffusers(cls, config: dict[str, Any]) -> SchedulerConfig:
        """Build from a diffusers ``scheduler/scheduler_config.json`` dict.

        Unknown/absent schedule parameters fall back to the (Stable Diffusion)
        defaults. The diffusers scheduler class name (``_class_name``) is mapped
        to an onnx-genai scheduler ``kind``:

        * ``DDIMScheduler``  -> ``ddim``
        * ``EulerDiscreteScheduler`` (non-ancestral) -> ``euler``

        Ancestral samplers (which inject fresh noise every step) have no
        deterministic onnx-genai equivalent and are rejected, as are scheduler
        classes onnx-genai does not implement, so a Mobius-built package never
        silently runs the wrong denoise dynamics.
        """
        raw_name = str(config.get("_class_name", ""))
        name = raw_name.lower()
        if "flowmatcheuler" in name:
            kind = "flow_match_euler"
        elif "eulerancestral" in name:
            kind = "euler_ancestral"
        elif "ancestral" in name or "sde" in name:
            raise ValueError(
                f"onnx-genai has no equivalent for the stochastic diffusers scheduler "
                f"{raw_name!r}; supported: DDIMScheduler, EulerDiscreteScheduler, "
                f"EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler"
            )
        elif not name or "ddim" in name:
            kind = "ddim"
        elif "dpmsolvermultistep" in name or "dpm++" in name or "dpmpp" in name:
            kind = "dpmpp_2m"
        elif "euler" in name:
            kind = "euler"
        else:
            raise ValueError(
                f"unsupported diffusers scheduler {raw_name!r} for onnx-genai; "
                f"supported kinds: ddim (DDIMScheduler), euler (EulerDiscreteScheduler)"
            )
        return cls(
            kind=kind,
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            beta_start=float(config.get("beta_start", 0.00085)),
            beta_end=float(config.get("beta_end", 0.012)),
            beta_schedule=str(config.get("beta_schedule", "scaled_linear")),
            prediction_type=str(
                config.get(
                    "prediction_type",
                    "flow_prediction" if kind == "flow_match_euler" else "epsilon",
                )
            ),
            clip_sample=bool(config.get("clip_sample")),
            clip_sample_range=float(config.get("clip_sample_range", 1.0)),
            set_alpha_to_one=bool(config.get("set_alpha_to_one", True)),
            steps_offset=int(config.get("steps_offset", 0)),
            timestep_spacing=str(config.get("timestep_spacing", "leading")),
            rescale_betas_zero_snr=bool(config.get("rescale_betas_zero_snr")),
            snr_shift_scale=float(config.get("snr_shift_scale", 1.0)),
            use_karras_sigmas=bool(config.get("use_karras_sigmas")),
            use_exponential_sigmas=bool(config.get("use_exponential_sigmas")),
            shift=float(config["shift"]) if config.get("shift") is not None else None,
            base_image_seq_len=(
                int(config["base_image_seq_len"])
                if config.get("base_image_seq_len") is not None
                else None
            ),
            max_image_seq_len=(
                int(config["max_image_seq_len"])
                if config.get("max_image_seq_len") is not None
                else None
            ),
            base_shift=(
                float(config["base_shift"]) if config.get("base_shift") is not None else None
            ),
            max_shift=(
                float(config["max_shift"]) if config.get("max_shift") is not None else None
            ),
            shift_terminal=(
                float(config["shift_terminal"])
                if config.get("shift_terminal") is not None
                else None
            ),
            use_dynamic_shifting=bool(config.get("use_dynamic_shifting")),
            time_shift_type=(
                str(config["time_shift_type"])
                if config.get("time_shift_type") is not None
                else None
            ),
            invert_sigmas=bool(config.get("invert_sigmas")),
            stochastic_sampling=bool(config.get("stochastic_sampling")),
            algorithm_type=str(config.get("algorithm_type", cls.algorithm_type)),
            solver_order=int(config.get("solver_order", cls.solver_order)),
            solver_type=str(config.get("solver_type", cls.solver_type)),
            lower_order_final=bool(config.get("lower_order_final", cls.lower_order_final)),
            final_sigmas_type=str(config.get("final_sigmas_type", cls.final_sigmas_type)),
        )


def load_diffusers_scheduler_config(
    source: str | None,
    *,
    revision: str | None = None,
) -> SchedulerConfig | None:
    """Best-effort load of a diffusers ``scheduler/scheduler_config.json``.

    ``source`` may be a local diffusers checkpoint directory or a Hugging Face
    model id. Returns a :class:`SchedulerConfig` on success, or ``None`` when the
    config cannot be found or names a scheduler onnx-genai does not implement
    (in which case a warning is logged and the caller should fall back to the
    DDIM default). This never raises for a missing/unsupported scheduler so a
    model build is not blocked by scheduler-metadata resolution.
    """
    if not source:
        return None
    raw: dict[str, Any] | None = None
    local = os.path.join(source, "scheduler", "scheduler_config.json")
    if os.path.isfile(local):
        try:
            with open(local, encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as err:
            _LOGGER.warning("could not read %s: %s", local, err)
            return None
    else:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                source,
                "scheduler/scheduler_config.json",
                revision=revision,
            )
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as err:
            _LOGGER.info("no diffusers scheduler config for %r (%s)", source, err)
            return None
    try:
        return SchedulerConfig.from_diffusers(raw)
    except ValueError as err:
        _LOGGER.warning(
            "%s; falling back to onnx-genai's default DDIM scheduler metadata", err
        )
        return None


def load_diffusers_vae_scaling_factor(
    source: str | None,
    *,
    revision: str | None = None,
) -> float | None:
    """Best-effort load of a diffusers ``vae/config.json`` ``scaling_factor``.

    The latent a diffusion sampler carries is scaled by this factor before the
    VAE decodes it, so the workflow needs the real value rather than a guess.
    Returns ``None`` when the config cannot be read, letting the caller decide.
    """
    if not source:
        return None
    raw: dict[str, Any] | None = None
    local = os.path.join(source, "vae", "config.json")
    if os.path.isfile(local):
        try:
            with open(local, encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as err:
            _LOGGER.warning("could not read %s: %s", local, err)
            return None
    else:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(source, "vae/config.json", revision=revision)
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as err:
            _LOGGER.info("no diffusers VAE config for %r (%s)", source, err)
            return None
    # ``AutoencoderKL`` defaults this to 0.18215, and diffusers checkpoints that
    # accept the default omit the key entirely.
    factor = (raw or {}).get("scaling_factor", 0.18215)
    return float(factor) if factor else None


#: Diffusion solvers mobius can materialize as an ONNX policy component, keyed
#: by the scheduler ``kind`` a diffusers config or a ComfyUI sampler resolves
#: to, with whether the solver consumes the latent pre-scaled by the current
#: sigma. A stochastic sampler injects fresh noise every step and has no
#: deterministic solver here, so it is rejected rather than silently lowered to
#: the closest deterministic one.
_SCHEDULER_SOLVERS: dict[str, tuple[str, bool]] = {
    "ddim": ("ddim", False),
    "euler": ("euler", True),
    "dpmpp_2m": ("multistep", False),
}

#: Latent axis names shared by the solver, guidance and clamp policy components
#: (``mobius.generation._policy_components._IMAGE_LATENT_DIMS``). The workflow's
#: latent contract reuses them so a declared value and the component port it
#: binds to can never disagree about an axis.
_LATENT_DIMS: tuple[str, ...] = ("batch", "channels", "height", "width")

_WORKFLOW_DTYPES: dict[str, str] = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


def _diffusion_solver(scheduler: SchedulerConfig) -> tuple[str, bool]:
    """Resolve the solver component and whether it pre-scales the model input."""
    resolved = _SCHEDULER_SOLVERS.get(scheduler.kind)
    if resolved is None:
        raise ValueError(
            f"Cannot publish a diffusion workflow for scheduler kind {scheduler.kind!r}. "
            "Why: the runtime executes the sampler as a declared solver component, and "
            f"only {sorted(_SCHEDULER_SOLVERS)} have one; a stochastic sampler injects "
            "fresh noise per step and has no deterministic equivalent. How to fix: export "
            "with a deterministic scheduler (DDIM, Euler or DPMSolverMultistep)."
        )
    return resolved


def _diffusion_schedule_values(
    scheduler: SchedulerConfig,
    num_inference_steps: int,
    timesteps: list[float] | None,
    schedule: list[float] | None,
    start_step: int | None,
) -> tuple[list[float], list[float], int]:
    """Resolve the solver schedule, the timestep table and the executed step count.

    The schedule is what ``solver_step`` integrates: a variance-preserving DDIM
    solver reads cumulative alphas, a sigma-space Euler / DPM-Solver++ one reads
    sigmas. Both use the same diffusers-compatible derivations the package
    exporter does, so a ComfyUI conversion and a package export of one
    checkpoint describe the same dynamics.

    ``start_step`` is img2img's "skip the noisiest steps", which diffusers
    implements by starting from a later entry of the same table, so it lowers to
    a sliced schedule plus a smaller loop bound.
    """
    from mobius.integrations.onnx_genai.auto_export import (
        _ddim_alpha_schedule,
        _diffusion_schedule,
    )

    if timesteps is not None and len(timesteps) != num_inference_steps:
        raise ValueError(
            f"timesteps has {len(timesteps)} entries but num_inference_steps is "
            f"{num_inference_steps}"
        )
    if schedule is not None and len(schedule) != num_inference_steps + 1:
        raise ValueError(
            f"schedule has {len(schedule)} entries but num_inference_steps + 1 is "
            f"{num_inference_steps + 1}"
        )
    derive = _ddim_alpha_schedule if scheduler.kind == "ddim" else _diffusion_schedule
    derived_timesteps, derived_schedule = derive(scheduler, num_inference_steps)
    schedule = (
        [float(value) for value in schedule] if schedule is not None else derived_schedule
    )
    table = (
        [float(value) for value in timesteps] if timesteps is not None else derived_timesteps
    )
    if start_step:
        if not 0 < start_step < num_inference_steps:
            raise ValueError(
                f"start_step ({start_step}) must be in 1..{num_inference_steps - 1}"
            )
        schedule = schedule[start_step:]
        table = table[start_step:]
    return schedule, table, len(table)


def build_diffusion_pipeline_metadata(
    *,
    num_inference_steps: int,
    denoiser_filename: str = "denoiser.onnx",
    denoiser_sample_input: str = "sample",
    denoiser_timestep_input: str = "timestep",
    denoiser_conditioning_input: str = "encoder_hidden_states",
    denoiser_output: str = "noise_pred",
    scheduler: SchedulerConfig | None = None,
    timesteps: list[float] | None = None,
    schedule: list[float] | None = None,
    guidance_scale: float | None = None,
    start_step: int | None = None,
    vae_filename: str | None = None,
    vae_latent_input: str = "latent",
    vae_output: str = "sample",
    text_encoder_filename: str | None = None,
    text_encoder_input: str = "input_ids",
    text_encoder_output: str = "last_hidden_state",
    text_encoder_edges: list[tuple[str, str]] | None = None,
    vae_scaling_factor: float | None = None,
    activation_dtype: str = "fp32",
    package: Any | None = None,
) -> dict[str, Any]:
    """Build the onnx-genai ``inference_metadata`` document for a diffusion pipeline.

    The denoise loop is emitted as an explicit ``pipeline.workflow``: the sigma
    schedule and timestep table are constant components, the step index is the
    loop induction value, and classifier-free guidance is two denoiser
    invocations plus a combine component.

    Only the ONNX components a caller *names* are described by artifact; their
    graphs stay authoritative for which ports exist. The solver, schedule,
    lookup, guidance and clamp components are built here from mobius's policy
    library and are attached to ``package`` so a writer can save them next to
    the metadata.

    Args:
        num_inference_steps: Number of denoise steps.
        denoiser_*: Denoiser component filename and I/O port names.
        scheduler: Noise-schedule config (defaults to DDIM defaults). Its
            ``kind`` selects the solver component.
        timesteps: Explicit per-step timestep table; derived from ``scheduler``
            otherwise.
        schedule: Explicit solver schedule (``num_inference_steps + 1`` values);
            derived from ``scheduler`` otherwise.
        guidance_scale: When set and != 1.0, enables classifier-free guidance.
        start_step: img2img skip count; lowered to a sliced schedule.
        vae_filename: VAE decoder producing the image output.
        text_encoder_filename: Optional text encoder feeding the conditioning.
        text_encoder_edges: Every ``(encoder_output, denoiser_input)`` edge to
            route; defaults to the single primary conditioning edge.
        vae_scaling_factor: The diffusers ``scaling_factor`` the VAE latents are
            normalized by; the decoder input is divided by it before decoding.
        activation_dtype: Declared dtype of the latent and image values.
        package: Optional :class:`~mobius._model_package.ModelPackage` the
            generated policy components are attached to.

    Returns:
        A dict with ``schema_version`` and a top-level ``pipeline.workflow``.
    """
    from mobius._model_package import ModelPackage
    from mobius.generation import (
        SOLVER_BUILDERS,
        build_boolean_not,
        build_euler_model_input,
        build_guidance_combine,
        build_scalar_constant,
        build_schedule_constant,
        build_schedule_lookup,
        build_tensor_clamp,
        build_tensor_scale,
        build_zeros_like,
    )

    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    if vae_filename is None:
        raise ValueError(
            "Cannot publish a diffusion workflow without a VAE decoder. Why: the "
            "workflow terminates in a decoded image output, so a latent-only pipeline "
            "has no executable result to declare. How to fix: pass vae_filename for the "
            "decoder the package ships."
        )
    if activation_dtype not in _WORKFLOW_DTYPES:
        raise ValueError(
            f"unsupported diffusion activation dtype {activation_dtype!r}; "
            f"expected one of {sorted(_WORKFLOW_DTYPES)}"
        )
    scheduler = scheduler or SchedulerConfig()
    solver, scales_model_input = _diffusion_solver(scheduler)
    schedule_values, timestep_values, executed_steps = _diffusion_schedule_values(
        scheduler, num_inference_steps, timesteps, schedule, start_step
    )
    conditioned = text_encoder_filename is not None
    guided = guidance_scale is not None and not math.isclose(guidance_scale, 1.0)
    if guided and not conditioned:
        raise ValueError("classifier-free guidance requires a text encoder to condition on")

    dtype = _ir_dtype_for(activation_dtype)
    pkg = package if package is not None else ModelPackage({})
    solver_builder = SOLVER_BUILDERS[solver]
    # A solver that fixes its own latent axis names takes only a dtype.
    solver_component = (
        solver_builder(dtype, _LATENT_DIMS)
        if "latent_dims" in inspect.signature(solver_builder).parameters
        else solver_builder(dtype)
    )
    # A multistep solver keeps the previous data estimate; a single-step one
    # does not, so only then is a history cell part of the loop.
    carries_history = "history" in {
        value.name for value in solver_component.model.graph.inputs
    }
    pkg.add_policy_component("solver_step", solver_component)
    pkg.add_policy_component("diffusion_schedule", build_schedule_constant(schedule_values))
    pkg.add_policy_component("diffusion_timesteps", build_schedule_constant(timestep_values))
    pkg.add_policy_component("schedule_lookup", build_schedule_lookup(dtype))
    pkg.add_policy_component("continue_predicate", build_boolean_not())
    if scales_model_input:
        pkg.add_policy_component(
            "model_input_scale", build_euler_model_input(dtype, _LATENT_DIMS)
        )
    if carries_history:
        pkg.add_policy_component("history_initializer", build_zeros_like(dtype))
    if guided:
        pkg.add_policy_component(
            "guidance_combine", build_guidance_combine(dtype, _LATENT_DIMS)
        )
    pkg.add_policy_component(
        "image_output_clamp",
        build_tensor_clamp(dtype, _LATENT_DIMS, minimum=-1.0, maximum=1.0),
    )
    # A sigma-space sampler starts from noise scaled by the largest sigma; a
    # variance-preserving one starts from the unit-variance draw itself. And a
    # VAE whose latents are normalized needs them un-normalized before decoding.
    # Only emit the constant and the multiply the pipeline actually performs.
    initial_state_scale = schedule_values[0] if scales_model_input else 1.0
    decoder_input_scale = 1.0 / vae_scaling_factor if vae_scaling_factor else 1.0
    scales_initial_state = not math.isclose(initial_state_scale, 1.0)
    scales_decoder_input = not math.isclose(decoder_input_scale, 1.0)
    if scales_initial_state or scales_decoder_input:
        pkg.add_policy_component("tensor_scale", build_tensor_scale(dtype))
    if scales_initial_state:
        pkg.add_policy_component(
            "initial_state_scale", build_scalar_constant(initial_state_scale)
        )
    if scales_decoder_input:
        pkg.add_policy_component(
            "decoder_input_scale", build_scalar_constant(decoder_input_scale)
        )

    workflow_dtype = _WORKFLOW_DTYPES[activation_dtype]
    latent_contract = _request_aligned({"dtype": workflow_dtype, "shape": list(_LATENT_DIMS)})
    row_float = _request_aligned({"dtype": "float32", "shape": ["batch"]})
    batch_bool = _request_aligned({"dtype": "bool", "shape": ["batch"]})
    prompt_contract = _request_aligned(
        {"dtype": "int64", "shape": ["batch", "prompt_sequence"]}
    )

    inputs: dict[str, Any] = {
        "request.max_iterations": {
            "contract": {"dtype": "int64", "shape": [1]},
            "role": {"kind": "runtime", "version": "1.0", "role": "max_iterations"},
            "source": {"kind": "request", "field": "max_iterations"},
            "required": False,
            "default": executed_steps,
        },
        "package.false": {
            "contract": batch_bool,
            "role": {"kind": "opaque"},
            "source": {"kind": "literal"},
            "required": False,
            "default": False,
        },
        "request.noise": {
            "contract": latent_contract,
            "role": {"kind": "opaque"},
            "source": {"kind": "application", "name": "noise"},
            "required": True,
            "externally_suppliable": True,
        },
    }
    outputs: dict[str, Any] = {
        "image": {
            "contract": latent_contract,
            "role": "image",
            "value_range": "negative_one_to_one",
            "stage": "pre_adapter",
        },
        "latent": {
            "contract": latent_contract,
            "role": "tensor",
            "stage": "pre_adapter",
        },
    }

    components: dict[str, Any] = {
        "denoiser": {"implementation": {"kind": "onnx", "artifact": denoiser_filename}},
        "vae": {"implementation": {"kind": "onnx", "artifact": vae_filename}},
    }

    setup_nodes: list[dict[str, Any]] = [
        _invoke("diffusion_schedule", {}, {"schedule": "diffusion.schedule"}),
        _invoke("diffusion_timesteps", {}, {"schedule": "diffusion.timesteps"}),
    ]
    if scales_initial_state:
        setup_nodes.append(
            _invoke("initial_state_scale", {}, {"value": "diffusion.initial_scale"})
        )
    if scales_decoder_input:
        setup_nodes.append(
            _invoke("decoder_input_scale", {}, {"value": "diffusion.decoder_scale"})
        )
    initial_state_value = "request.noise"
    if scales_initial_state:
        initial_state_value = "diffusion.initial_state"
        setup_nodes.append(
            _invoke(
                "tensor_scale",
                {"tensor": "request.noise", "scale": "diffusion.initial_scale"},
                {"scaled": initial_state_value},
            )
        )

    # Each (encoder_output, denoiser_input) edge becomes one SSA value routed
    # from the text encoder into the denoiser. SDXL routes two (concatenated
    # hidden states + pooled text_embeds); SD routes one.
    edges = list(text_encoder_edges or [(text_encoder_output, denoiser_conditioning_input)])
    conditional_values: dict[str, str] = {}
    unconditional_values: dict[str, str] = {}
    if conditioned:
        components["text_encoder"] = {
            "implementation": {"kind": "onnx", "artifact": text_encoder_filename}
        }
        inputs["request.prompt_tokens"] = {
            "contract": prompt_contract,
            "role": {"kind": "runtime", "version": "1.0", "role": "prompt_tokens"},
            "source": {"kind": "request", "field": "prompt_tokens"},
            "required": True,
            "externally_suppliable": True,
        }
        conditional_values = {
            denoiser_in: f"conditioning.{denoiser_in}" for _, denoiser_in in edges
        }
        setup_nodes.append(
            _invoke(
                "text_encoder",
                {text_encoder_input: "request.prompt_tokens"},
                {
                    encoder_out: conditional_values[denoiser_in]
                    for encoder_out, denoiser_in in edges
                },
            )
        )
        if guided:
            assert guidance_scale is not None
            inputs["request.negative_prompt_tokens"] = {
                "contract": prompt_contract,
                "role": {
                    "kind": "runtime",
                    "version": "1.0",
                    "role": "negative_prompt_tokens",
                },
                "source": {"kind": "request", "field": "negative_prompt_tokens"},
                "required": True,
                "externally_suppliable": True,
            }
            inputs["request.guidance_scale"] = {
                "contract": row_float,
                "role": {"kind": "runtime", "version": "1.0", "role": "guidance_scale"},
                "source": {"kind": "request", "field": "guidance_scale"},
                "required": False,
                "default": float(guidance_scale),
            }
            unconditional_values = {
                denoiser_in: f"conditioning.unconditional.{denoiser_in}"
                for _, denoiser_in in edges
            }
            setup_nodes.append(
                _invoke(
                    "text_encoder",
                    {text_encoder_input: "request.negative_prompt_tokens"},
                    {
                        encoder_out: unconditional_values[denoiser_in]
                        for encoder_out, denoiser_in in edges
                    },
                )
            )

    state: dict[str, Any] = {
        "latent": {
            "contract": latent_contract,
            "scope": "invocation",
            "initializer": initial_state_value,
            "recurrence": {"kind": "invariant"},
        }
    }
    carried: list[dict[str, Any]] = [
        {
            "cell": "latent",
            "current": initial_state_value,
            "body_input": "state.latent.body",
            "body_output": "latent.body",
            "next": "latent.final",
        }
    ]
    if carries_history:
        setup_nodes.append(
            _invoke(
                "history_initializer",
                {"reference": initial_state_value},
                {"zeros": "diffusion.initial_history"},
            )
        )
        state["history"] = {
            "contract": latent_contract,
            "scope": "invocation",
            "initializer": "diffusion.initial_history",
            "recurrence": {"kind": "invariant"},
        }
        carried.append(
            {
                "cell": "history",
                "current": "diffusion.initial_history",
                "body_input": "state.history.body",
                "body_output": "history.body",
                "next": "history.final",
            }
        )
    setup_nodes.append(
        _invoke(
            "continue_predicate", {"done": "package.false"}, {"continue": "setup.continue"}
        )
    )

    body_nodes: list[dict[str, Any]] = [
        _invoke(
            "schedule_lookup",
            {"schedule": "diffusion.timesteps", "step": "loop.iteration"},
            {"timestep": "diffusion.timestep"},
        )
    ]
    model_input_value = "state.latent.body"
    if scales_model_input:
        model_input_value = "diffusion.model_input"
        body_nodes.append(
            _invoke(
                "model_input_scale",
                {
                    "sample": "state.latent.body",
                    "step": "loop.iteration",
                    "schedule": "diffusion.schedule",
                },
                {"model_input": model_input_value},
            )
        )

    def denoiser_call(conditioning: dict[str, str], estimate: str) -> dict[str, Any]:
        call_inputs = {
            denoiser_sample_input: model_input_value,
            denoiser_timestep_input: "diffusion.timestep",
            **conditioning,
        }
        return _invoke("denoiser", call_inputs, {denoiser_output: estimate})

    if guided:
        body_nodes.append(denoiser_call(unconditional_values, "denoiser.unconditional"))
        body_nodes.append(denoiser_call(conditional_values, "denoiser.conditional"))
        body_nodes.append(
            _invoke(
                "guidance_combine",
                {
                    "unconditional": "denoiser.unconditional",
                    "conditional": "denoiser.conditional",
                    "scale": "request.guidance_scale",
                },
                {"estimate": "denoiser.estimate"},
            )
        )
    else:
        body_nodes.append(denoiser_call(conditional_values, "denoiser.estimate"))

    solver_inputs = {
        "sample": "state.latent.body",
        "step": "loop.iteration",
        "schedule": "diffusion.schedule",
        "estimate" if carries_history else "derivative": "denoiser.estimate",
    }
    solver_outputs = {"next_state": "latent.body"}
    if carries_history:
        solver_inputs["history"] = "state.history.body"
        solver_outputs["next_history"] = "history.body"
    body_nodes.append(_invoke("solver_step", solver_inputs, solver_outputs))
    body_nodes.append(
        _invoke("continue_predicate", {"done": "package.false"}, {"continue": "loop.continue"})
    )

    decoder_input_value = "latent.final"
    tail_nodes: list[dict[str, Any]] = []
    if scales_decoder_input:
        decoder_input_value = "diffusion.decoder_input"
        tail_nodes.append(
            _invoke(
                "tensor_scale",
                {"tensor": "latent.final", "scale": "diffusion.decoder_scale"},
                {"scaled": decoder_input_value},
            )
        )
    tail_nodes += [
        _invoke("vae", {vae_latent_input: decoder_input_value}, {vae_output: "vae.raw_image"}),
        _invoke("image_output_clamp", {"tensor": "vae.raw_image"}, {"clamped": "vae.image"}),
        {"kind": "emit", "value": "latent.final", "output": "latent", "mode": "replace"},
        {"kind": "emit", "value": "vae.image", "output": "image", "mode": "replace"},
    ]

    workflow = {
        "manifest": {
            "capabilities": [
                "workflow_ssa",
                "nested_control_flow",
                "loop_induction_values",
                "typed_emit",
            ]
        },
        "inputs": inputs,
        "outputs": outputs,
        "components": components,
        "state": state,
        "graph": {
            "kind": "sequence",
            "nodes": [
                {
                    "kind": "loop",
                    "setup": {"kind": "sequence", "nodes": setup_nodes},
                    "body": {"kind": "sequence", "nodes": body_nodes},
                    "condition": "loop.continue",
                    "max_iterations": "request.max_iterations",
                    "iteration": {
                        "value": "loop.iteration",
                        "contract": {"dtype": "int64", "shape": ["batch"]},
                    },
                    "carried": carried,
                },
                *tail_nodes,
            ],
        },
    }
    metadata = {
        "schema_version": "v1",
        "pipeline": {"workflow": _publish_workflow_v1(workflow)},
    }
    add_policy_components_to_workflow(metadata, pkg)
    return metadata


def _ir_dtype_for(activation_dtype: str) -> Any:
    import onnx_ir as ir

    return {
        "fp32": ir.DataType.FLOAT,
        "fp16": ir.DataType.FLOAT16,
        "bf16": ir.DataType.BFLOAT16,
    }[activation_dtype]


def build_multimodal_pipeline_metadata(
    *,
    decoder_filename: str = "decoder.onnx",
    embedding_filename: str = "embedding.onnx",
    vision_encoder_filename: str | None = None,
    audio_encoder_filename: str | None = None,
    tokenizer_filename: str = "tokenizer.json",
    activation_dtype: str = "fp32",
    decoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata for an encoder-to-fusion-to-decoder multimodal pipeline.

    At least one modality encoder is required. Each encoder and the embedding
    fusion model runs once for the prompt; the decoder then runs
    autoregressively for every generation step.

    Args:
        decoder_filename: Decoder ONNX filename relative to the package root.
        embedding_filename: Embedding fusion ONNX filename.
        vision_encoder_filename: Optional vision encoder ONNX filename.
        audio_encoder_filename: Optional audio encoder ONNX filename.
        tokenizer_filename: Tokenizer filename used by the decoder.
        decoder_metadata: Optional output from
            :func:`decoder_metadata_from_config`. Its decoder capabilities are
            retained at the document top level.

    Returns:
        A dict with a top-level ``pipeline`` key and any decoder capabilities.
    """
    if vision_encoder_filename is None and audio_encoder_filename is None:
        raise ValueError("a multimodal pipeline requires a vision or audio encoder")

    models: dict[str, Any] = {}
    dataflow: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    phases: dict[str, Any] = {}

    def add_encoder(
        name: str,
        filename: str,
        model_type: str,
        output_name: str,
        stage_name: str,
    ) -> None:
        models[name] = {"filename": filename, "type": model_type}
        dataflow.append(
            {
                "from": f"{name}.{output_name}",
                "to": f"embedding.{output_name}",
                "dtype": activation_dtype,
                "device_transfer": False,
            }
        )
        stages.append(
            {
                "name": stage_name,
                "strategy": {"kind": "single_pass", "model": name},
            }
        )
        phases[name] = {"run_on": "prompt_only"}

    if vision_encoder_filename is not None:
        add_encoder(
            "vision_encoder",
            vision_encoder_filename,
            "vision_encoder",
            "image_features",
            "encode_vision",
        )
    if audio_encoder_filename is not None:
        add_encoder(
            "audio_encoder",
            audio_encoder_filename,
            "audio_encoder",
            "audio_features",
            "encode_audio",
        )

    models["embedding"] = {"filename": embedding_filename, "type": "encoder"}
    models["decoder"] = {
        "filename": decoder_filename,
        "type": "decoder",
        "tokenizer": tokenizer_filename,
    }
    dataflow.append(
        {
            "from": "embedding.inputs_embeds",
            "to": "decoder.inputs_embeds",
            "dtype": activation_dtype,
            "device_transfer": False,
        }
    )
    stages.extend(
        [
            {
                "name": "fuse_embeddings",
                "strategy": {"kind": "single_pass", "model": "embedding"},
            },
            {
                "name": "decode",
                "strategy": {"kind": "autoregressive", "decoder": "decoder"},
            },
        ]
    )
    phases["embedding"] = {"run_on": "prompt_only"}
    phases["decoder"] = {"run_on": "every_step"}

    metadata = dict(decoder_metadata or {})
    metadata["pipeline"] = {
        "models": models,
        "dataflow": dataflow,
        "strategy": {"kind": "composite", "stages": stages},
    }
    return metadata


def write_multimodal_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write composite multimodal metadata into ``directory``."""
    metadata = build_multimodal_pipeline_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


def build_speech_to_text_pipeline_metadata(
    *,
    encoder_filename: str = "encoder/model.onnx",
    decoder_filename: str = "decoder/model.onnx",
    tokenizer_filename: str = "tokenizer.json",
    activation_dtype: str = "fp32",
    encoder_attention_mask: bool = False,
    decoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata for a cross-attention encoder-decoder ASR pipeline.

    This is the Whisper-style speech-to-text shape (DESIGN.md §20): the audio
    encoder runs once for the prompt and produces ``encoder_hidden_states``,
    which the autoregressive decoder consumes via cross-attention (distinct from
    the multimodal ``inputs_embeds`` fusion shape). The decoder then runs for
    every generation step.

    Args:
        encoder_filename: Audio encoder ONNX filename relative to the package
            root.
        decoder_filename: Decoder ONNX filename relative to the package root.
        tokenizer_filename: Tokenizer filename used by the decoder.
        decoder_metadata: Optional output from
            :func:`decoder_metadata_from_config`; its decoder capabilities are
            retained at the document top level.
        encoder_attention_mask: Whether to route the encoder's downsampled
            attention mask into decoder cross-attention.

    Returns:
        A dict with a top-level ``pipeline`` key and any decoder capabilities.
    """
    metadata = dict(decoder_metadata or {})
    dataflow = [
        {
            "from": "encoder.encoder_hidden_states",
            "to": "decoder.encoder_hidden_states",
            "dtype": activation_dtype,
            "device_transfer": False,
        }
    ]
    if encoder_attention_mask:
        dataflow.append(
            {
                "from": "encoder.encoder_attention_mask",
                "to": "decoder.encoder_attention_mask",
                "dtype": "int64",
                "device_transfer": False,
            }
        )
    metadata["pipeline"] = {
        "models": {
            "encoder": {"filename": encoder_filename, "type": "encoder"},
            "decoder": {
                "filename": decoder_filename,
                "type": "decoder",
                "tokenizer": tokenizer_filename,
            },
        },
        "dataflow": dataflow,
        "strategy": {
            "kind": "composite",
            "stages": [
                {
                    "name": "encode_audio",
                    "strategy": {"kind": "single_pass", "model": "encoder"},
                },
                {
                    "name": "decode_transcript",
                    "strategy": {"kind": "autoregressive", "decoder": "decoder"},
                },
            ],
        },
    }
    return metadata


def write_speech_to_text_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write composite speech-to-text metadata into ``directory``."""
    metadata = build_speech_to_text_pipeline_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


def write_diffusion_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write ``inference_metadata.yaml`` into ``directory``.

    The generated sampler policy components (solver, schedule constants,
    lookup, guidance, clamp) are saved alongside it, because the emitted
    workflow references them as ONNX artifacts.

    Extra keyword arguments are forwarded to
    :func:`build_diffusion_pipeline_metadata`. Returns the written path.
    """
    from mobius._model_package import ModelPackage

    package = ModelPackage({})
    metadata = build_diffusion_pipeline_metadata(package=package, **kwargs)
    os.makedirs(directory, exist_ok=True)
    package.save_policy_components(directory)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


#: ``SpeculativeContract.port_bindings`` role for the proposer input port that
#: receives the target's per-token hidden state.
_TARGET_HIDDEN_CONTEXT_ROLE = "target_hidden_context"

#: Port roles the Qwen3.5/3.8 MTP sidecar exposes. The graph is authoritative
#: for which ports exist and their contracts; only what a port *means* is
#: declared here (see ``workflow_metadata._component``).
_MTP_PROPOSER_PORT_ROLES: dict[str, str] = {
    "hidden_states": "hidden_states",
    "attention_mask": "attention_mask",
    "position_ids": "position_ids",
}


def _speculative_target_candidates(workflow: dict[str, Any], exclude: str) -> list[str]:
    """Workflow components that can verify a speculative proposal.

    A verifier scores the next token, so it is selected by its declared
    ``logits`` output role. Selecting on ``implementation.kind == "onnx"`` would
    not narrow anything: the generated policy graphs a workflow ships (sampler,
    termination predicate, state updates) are ONNX components too, and a real
    decoder package declares about a dozen of them.
    """
    return [
        name
        for name, declaration in (workflow.get("components") or {}).items()
        if name != exclude
        and isinstance(declaration, dict)
        and declaration.get("implementation", {}).get("kind") == "onnx"
        and "logits" in (declaration.get("ports", {}).get("roles") or {}).values()
    ]


def write_mtp_speculator_metadata(
    directory: str,
    *,
    backbone_config: Any | None = None,
    proposer_config: Any | None = None,
    filename: str = "inference_metadata.yaml",
    model_path: str = "mtp/model.onnx",
    num_speculative_tokens: int = 1,
    embedding_weights: str = "model.embed_tokens.weight",
    lm_head_weights: str | None = None,
    proposer_name: str = "mtp",
) -> str | None:
    """Declare the exported MTP head as the backbone's speculative proposer.

    The Qwen3.5/3.8 MTP head is a self-speculative drafter saved next to the
    backbone (``mtp/model.onnx``). It borrows target embedding / LM-head weights
    only when the GGUF omits dedicated tables, and is seeded by the backbone's
    dedicated post-final-norm ``mtp_seed`` output.

    ``SpeculativeContract`` is expressed in terms of the workflow — ``proposer``
    and ``target`` are component names, ``rollback_state`` names state cells —
    so this writer also registers the head as a workflow component and completes
    the rollback capabilities the claim depends on. Notably:

    - ``proposal_execution: block``, not ``chained``: one invocation emits the
      whole proposal. A sidecar without a dedicated LM head exposes
      ``mtp_hidden`` for decoding through the target head; a dedicated head
      exposes ``logits`` directly.
    - ``port_bindings.target_hidden_context`` names the proposer input port the
      target's hidden state lands in; its source is declared as a
      ``hidden_states`` port role on the target component.
    - ``vocabulary: identical`` — sharing the target's LM head means scoring the
      target's own vocabulary axis.

    The backbone ``inference_metadata.yaml`` must already exist and declare a
    ``pipeline.workflow`` to anchor against. Returns the metadata path, or
    ``None`` when the file is missing.
    """
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}

    workflow = metadata.get("pipeline", {}).get("workflow")
    if not isinstance(workflow, dict) or not workflow.get("components"):
        raise ValueError(
            "Cannot declare an MTP speculator: the backbone "
            f"{filename!r} has no pipeline.workflow components. Why: a "
            "SpeculativeContract names its proposer and target as workflow "
            "components, so there is nothing to anchor the claim to. How to "
            "fix: write the backbone workflow metadata (write_onnx_genai_config) "
            "before attaching the MTP head."
        )
    candidates = _speculative_target_candidates(workflow, exclude=proposer_name)
    if len(candidates) != 1:
        raise ValueError(
            "Cannot declare an MTP speculator: expected exactly one workflow "
            "component declaring a 'logits' output role to verify proposals, found "
            f"{sorted(candidates)}. Why: the speculative target must be unambiguous. "
            "How to fix: pass a backbone package whose workflow declares a single "
            "decoder component."
        )
    target_name = candidates[0]

    from mobius.models.base import effective_tie_word_embeddings

    dedicated_embeddings = bool(getattr(proposer_config, "use_dedicated_embeddings", False))
    dedicated_lm_head = bool(getattr(proposer_config, "use_dedicated_lm_head", False))
    quantization = getattr(backbone_config, "quantization", None)
    quantized_embedding = bool(
        quantization is not None and getattr(quantization, "quantize_embeddings", False)
    )
    quantized_lm_head = bool(
        quantization is not None and getattr(quantization, "quantize_lm_head", False)
    )
    effective_tie = effective_tie_word_embeddings(backbone_config)
    tied_quantized_head = bool(quantized_embedding and quantized_lm_head and effective_tie)
    tied_float_head = bool(not quantized_embedding and not quantized_lm_head and effective_tie)
    has_zero_point = bool(quantization is not None and not getattr(quantization, "sym", True))
    embedding_initializers = [embedding_weights]
    if quantized_embedding:
        embedding_stem = embedding_weights.removesuffix(".weight")
        embedding_initializers = [
            f"{embedding_stem}.qweight",
            f"{embedding_stem}.scales",
            *([f"{embedding_stem}.zero_points"] if has_zero_point else []),
        ]
    if lm_head_weights is not None:
        lm_head_initializers = [lm_head_weights]
    elif tied_quantized_head or tied_float_head:
        # The target LM head consumes the embedding initializer directly; the
        # canonical text-model rebinder guarantees there is no lm_head-owned
        # duplicate in the graph.
        lm_head_initializers = embedding_initializers
    elif quantized_lm_head:
        lm_head_initializers = [
            "lm_head.weight",
            "lm_head.scales",
            *(["lm_head.zero_points"] if has_zero_point else []),
        ]
    else:
        lm_head_initializers = ["lm_head.weight_t"]
    shared_weights: list[str] = []
    if not dedicated_embeddings:
        shared_weights.extend(embedding_initializers)
    if not dedicated_lm_head:
        shared_weights.extend(lm_head_initializers)
    shared_weights = sorted(set(shared_weights))

    proposer_roles = dict(_MTP_PROPOSER_PORT_ROLES)
    proposer_roles["input_ids" if dedicated_embeddings else "inputs_embeds"] = (
        "token_ids" if dedicated_embeddings else "inputs_embeds"
    )
    proposer_roles["logits" if dedicated_lm_head else "mtp_hidden"] = (
        "logits" if dedicated_lm_head else "hidden_states"
    )

    components = workflow["components"]
    components[proposer_name] = {
        "implementation": {"kind": "onnx", "artifact": model_path},
        "ports": {"roles": proposer_roles},
    }
    # The target seed is explicitly post-final-norm. Per-layer hidden_states.N
    # remain pre-final-norm and must never receive this runtime role.
    target_roles = components[target_name].setdefault("ports", {}).setdefault("roles", {})
    target_roles["mtp_seed"] = "hidden_states"

    # A rejected proposal must rewind every state cell the target advances.
    rollback_cells = sorted(
        cell
        for cell, declaration in (workflow.get("state") or {}).items()
        if isinstance(declaration, dict) and declaration.get("service_group")
    )
    _declare_rollback_capacity(workflow, rollback_cells, int(num_speculative_tokens))

    metadata["speculative"] = {
        "proposer": proposer_name,
        "target": target_name,
        # One invocation yields the whole k-token proposal.
        "proposal_execution": {"kind": "block"},
        "port_bindings": {_TARGET_HIDDEN_CONTEXT_ROLE: "hidden_states"},
        **({"shared_weights": shared_weights} if shared_weights else {}),
        # The head scores the target's own vocabulary through the shared LM head.
        "vocabulary": {"kind": "identical"},
        "max_proposal_width": int(num_speculative_tokens),
        # Standard rejection sampling against the target keeps the target's
        # output distribution exact.
        "distribution_preserving": True,
        **({"rollback_state": rollback_cells} if rollback_cells else {}),
    }
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return path


def _declare_rollback_capacity(
    workflow: dict[str, Any],
    cells: Sequence[str],
    positions: int,
) -> None:
    """Guarantee every group reached by ``cells`` can rewind ``positions``.

    A package is rejected when a rolled-back cell resolves to a group declaring
    no ``rollback_positions``, or fewer than the maximum proposal width.
    Attaching the speculator creates that requirement, so it also states the
    bound.
    """
    groups = (workflow.get("serving") or {}).get("state_service", {}).get("groups", {})
    state = workflow.get("state") or {}
    pending = [
        state[cell]["service_group"]
        for cell in cells
        if isinstance(state.get(cell), dict) and state[cell].get("service_group")
    ]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        contract = groups.get(name)
        if not isinstance(contract, dict):
            continue
        capabilities = contract.setdefault("capabilities", {})
        declared = capabilities.get("rollback_positions")
        if not isinstance(declared, int) or declared < positions:
            capabilities["rollback_positions"] = positions
        pending.extend(capabilities.get("cascade") or [])
