# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit onnx-genai ``inference_metadata`` for multi-model pipelines.

Mobius builds the neural components of a diffusion model (denoiser transformer,
VAE, and — externally — a text encoder) as separate ONNX graphs, but does not
itself carry a scheduler loop. onnx-genai's *iterative* pipeline supplies that
loop declaratively: given an ``inference_metadata`` document describing the
components, the loop-carried dataflow, a timestep input, a scheduler, and
(optionally) classifier-free guidance, it drives the denoise loop and returns
the decoded output.

This module produces that document from the component filenames + a scheduler
config. It reads no torch/diffusers state — only plain values — so it is cheap
to unit-test and safe to call anywhere.

The emitted contract matches onnx-genai's pipeline schema:
``schema/inference_metadata.schema.json`` (kind ``iterative`` with
``denoiser`` / ``num_steps`` / ``timestep_input`` / ``scheduler_config`` /
``cfg_conditioning_input`` and denoiser self-edge loop-carried dataflow).

Autoregressive decoder-only LLM metadata (``model.attention`` + ``kv_cache``)
lives in the sibling :mod:`mobius.integrations.onnx_genai.decoder_metadata`
module. Composite multimodal pipelines retain those decoder properties while
declaring their encoder, fusion, and decoder execution stages here.
"""

from __future__ import annotations

import dataclasses
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
class _Port:
    """Structural description of one ONNX graph port."""

    value: Any
    name: str
    dtype: str
    rank: int | None
    dims: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class _ImageProgram:
    """Registry result for one declared image processor contract."""

    name: str
    bindings: tuple[tuple[_Port, str, float | None], ...]
    transforms: Callable[[Any, dict[str, Any]], list[dict[str, Any]]]
    token_count_source: str
    summary_contents: tuple[str, ...] = ()
    vision_properties: Callable[[dict[str, Any]], dict[str, Any]] | None = None


_DTYPE_TAGS = {
    "FLOAT": "fp32",
    "FLOAT16": "fp16",
    "BFLOAT16": "bf16",
    "FLOAT8E4M3FN": "float8_e4m3fn",
    "FLOAT8E5M2": "float8_e5m2",
    "INT64": "int64",
    "INT32": "int32",
    "INT8": "int8",
    "UINT8": "uint8",
    "BOOL": "bool",
    "STRING": "string",
}


def _port(value: Any) -> _Port:
    shape = getattr(value, "shape", None)
    dims = tuple(shape) if shape is not None else ()
    dtype = getattr(getattr(value, "dtype", None), "name", "")
    return _Port(
        value=value,
        name=str(value.name),
        dtype=_DTYPE_TAGS.get(str(dtype).upper(), str(dtype).lower() or "fp32"),
        rank=len(dims) if shape is not None else None,
        dims=dims,
    )


_BATCH_DIMENSION = "batch"
"""Symbolic leading dimension mobius uses for per-request batching."""


def _shape_metadata(port: _Port) -> list[int | str]:
    """Return a YAML-safe graph shape without losing symbolic dimensions."""
    shape: list[int | str] = []
    for axis, dim in enumerate(port.dims):
        if isinstance(dim, int):
            shape.append(dim)
            continue
        value = getattr(dim, "value", None)
        # Metadata dimensions cannot be null. Preserve named graph dimensions;
        # give anonymous dynamic dimensions a stable, port-local name instead
        # of pretending they are static or serializing an invalid null.
        shape.append(str(value) if value is not None else f"{port.name}_dim_{axis}")
    return shape


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
    r"(?P<role>key|value|conv_state|recurrent_state|ssm_state)$"
)
_STATIC_CACHE_PORT = re.compile(
    r"^(?P<updated>updated_)?(?P<role>key|value)_cache\.(?P<layer>\d+)$"
)
_REPLACE_ROLES = {
    "lightning_attention": {"recurrent_state"},
    "linear_attention": {"conv_state", "recurrent_state"},
    "conv": {"conv_state"},
    "mamba": {"conv_state", "ssm_state"},
    "mamba2": {"conv_state", "ssm_state"},
}
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
            if layer_type in _REPLACE_ROLES or layer_type in _STATELESS_LAYER_TYPES:
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
)


def _positions_from_registry(position: _Port, config: Any) -> dict[str, Any]:
    if position.rank == 3:
        semantic_rank = 3
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
        "config. Rank-3 axes and continuation are never guessed. Regenerate with "
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


#: Symbolic leading dimension Mobius emits for every batched ONNX port. A port
#: that opens with it holds exactly one entry per in-flight request, which is
#: the structural fact a runtime needs to permute or drop rows.
REQUEST_AXIS_SYMBOL = "batch"


def request_batch_layout(shape: list[Any] | None) -> dict[str, Any] | None:
    """Return the request-aligned batch layout implied by a port's shape."""
    if shape and shape[0] == REQUEST_AXIS_SYMBOL:
        return {"kind": "request_aligned", "axis": 0}
    return None


def add_policy_components_to_workflow(
    metadata: dict[str, Any],
    pkg: Any,
) -> dict[str, Any]:
    """Reference attached ONNX policy artifacts from an existing workflow.

    This helper intentionally does not synthesize a workflow or guess bindings.
    It only adds schema-defined component declarations when a producer has
    already emitted the exact workflow contract.
    """
    policy_components = getattr(pkg, "policy_components", {})
    if not policy_components:
        return metadata
    workflow = metadata.get("pipeline", {}).get("workflow")
    if not isinstance(workflow, dict):
        return metadata
    components = workflow.setdefault("components", {})

    def semantic_contract(component: Any) -> dict[str, Any]:
        contract = component.contract
        contract_name, version = component.contract_id.rsplit("@", 1)
        bindings = {
            key: value
            for key, value in contract.items()
            if key
            not in {
                "role",
                "mode",
                "effect",
                "rng",
                "state_class",
                "batching",
                "inactive_rows",
            }
            and isinstance(value, str)
        }
        rng = contract.get("rng")
        if isinstance(rng, dict):
            bindings.update(
                {key: value for key, value in rng.items() if isinstance(value, str)}
            )
        declaration: dict[str, Any] = {
            "id": contract_name,
            "version": version,
            "bindings": bindings,
        }
        parameters = {
            key: contract[key]
            for key in ("mode", "batching", "inactive_rows")
            if key in contract
        }
        if parameters:
            declaration["parameters"] = parameters
        return declaration

    def tensor_contract(value: Any) -> dict[str, Any]:
        port = _port(value)
        dtype = {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
        }.get(port.dtype, port.dtype)
        shape = _shape_metadata(port)
        contract: dict[str, Any] = {
            "dtype": dtype,
            "rank": port.rank,
            "shape": shape,
        }
        layout = request_batch_layout(shape)
        if layout is not None:
            contract["batch_layout"] = layout
        return contract

    for name, component in policy_components.items():
        # A policy graph is synthesized by this producer to realize the
        # workflow's own control flow, so its port contracts are not a
        # transcription of an external interface: they are the type annotations
        # of the workflow's dataflow. A workflow value acquires its dtype, rank
        # and request axis from the port that produces it, and the validator
        # reads metadata without the artifacts, so a policy output that states
        # no contract leaves every value derived from it untyped.
        declaration = {
            "implementation": {
                "kind": "onnx",
                "artifact": f"policies/{name}.onnx",
            },
            "ports": {
                "inputs": {
                    value.name: tensor_contract(value)
                    for value in component.model.graph.inputs
                },
                "outputs": {
                    value.name: tensor_contract(value)
                    for value in component.model.graph.outputs
                },
            },
        }
        if component.contract:
            declaration["contract"] = semantic_contract(component)
            if component.contract.get("role") == "token_sampler":
                declaration["application_overridable"] = True
        components[name] = declaration
    declare_request_alignment(workflow)
    return metadata


_BATCH_DIMENSION_NAMES = frozenset({"batch", "batch_size", "batch_dim", "b"})


def declare_request_alignment(workflow: dict[str, Any]) -> None:
    """Stamp the request-aligned row axis onto every batch-leading contract.

    The runtime compacts finished rows out of a batch by applying one row
    permutation to every request-aligned tensor. A contract whose leading axis
    is the batch symbol but that does not say so is unpermutable, so state,
    component ports, and outputs would silently drift apart after the first
    eviction. Deriving the declaration from the admitted graph's own batch
    symbol keeps alignment a property of the model interface rather than an
    annotation every workflow builder has to remember.
    """

    def stamp(contract: Any) -> None:
        if not isinstance(contract, dict) or "batch_layout" in contract:
            return
        shape = contract.get("shape") or []
        if shape and str(shape[0]) in _BATCH_DIMENSION_NAMES:
            contract["batch_layout"] = {"kind": "request_aligned", "axis": 0}

    for section in ("inputs", "outputs", "state"):
        for declaration in (workflow.get(section) or {}).values():
            if isinstance(declaration, dict):
                stamp(declaration.get("contract"))
    for component in (workflow.get("components") or {}).values():
        ports = component.get("ports", {}) if isinstance(component, dict) else {}
        for side in ("inputs", "outputs"):
            for contract in (ports.get(side) or {}).values():
                stamp(contract)
    # A cell backed by a state-service group is stored by the runtime, not by
    # the workflow: the group owns the buffer and the eviction policy, so the
    # cell also needs an explicit boundary at which the runtime may free it.
    for declaration in (workflow.get("state") or {}).values():
        if isinstance(declaration, dict) and declaration.get("service_group"):
            declaration.setdefault("management", "runtime")
            declaration.setdefault("release_boundary", declaration.get("scope", "invocation"))


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
            and contract.get("rank") == len(shape)
            and contract.get("shape") == shape
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
                "contract": {"dtype": dtype, "rank": len(shape), "shape": shape},
                "role": (
                    {"kind": "runtime", "version": "1.0", "role": role}
                    if role is not None
                    else {"kind": "opaque"}
                ),
                "source": source or {"kind": "request"},
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
    """Emit a native VLM contract by inspecting every component graph.

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


def write_native_vlm_package_metadata(
    pkg: Any,
    directory: str,
    *,
    config: Any,
    source: str | None = None,
    revision: str | None = None,
    filename: str = "inference_metadata.yaml",
) -> dict[str, str]:
    """Write native VLM metadata and the runtime's tokenizer/processor assets."""
    from mobius.integrations.onnx_genai.decoder_metadata import (
        decoder_metadata_from_config,
    )

    metadata = build_native_vlm_package_metadata(
        pkg,
        config=config,
        source=source,
        decoder_metadata=decoder_metadata_from_config(config),
    )
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    artifacts = {"inference_metadata": path}
    artifacts.update(_copy_runtime_assets(directory, source, revision=revision))
    return artifacts


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
    guidance_scale: float | None = None,
    start_step: int | None = None,
    vae_filename: str | None = None,
    vae_latent_input: str = "latent",
    text_encoder_filename: str | None = None,
    text_encoder_output: str = "last_hidden_state",
    text_encoder_edges: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the onnx-genai ``inference_metadata`` dict for a diffusion pipeline.

    The denoiser runs an iterative loop: its ``denoiser_output`` (a noise
    prediction) is fed back to ``denoiser_sample_input`` each step (a
    loop-carried self-edge), the scheduler combines it with the current latent,
    the per-step timestep is injected into ``denoiser_timestep_input``, and the
    conditioning is supplied on ``denoiser_conditioning_input``.

    Args:
        num_inference_steps: Number of denoise steps (``strategy.num_steps``).
        denoiser_*: Denoiser component filename and I/O port names.
        scheduler: Noise-schedule config (defaults to DDIM defaults).
        guidance_scale: When set and != 1.0, enables classifier-free guidance
            (the conditioning input is zeroed on the unconditional pass).
        vae_filename: Optional VAE decoder; runs ``final_only`` on the final
            latent (``denoiser_sample_input``).
        vae_latent_input: VAE latent input port name.
        text_encoder_filename: Optional text encoder; runs ``prompt_only`` and
            feeds ``denoiser_conditioning_input``.
        text_encoder_output: Text encoder output port name.

    Returns:
        A dict with a top-level ``pipeline`` key, ready to serialize to
        ``inference_metadata.yaml``.
    """
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    scheduler = scheduler or SchedulerConfig()

    models: dict[str, Any] = {
        "denoiser": {"filename": denoiser_filename, "type": "denoiser"},
    }
    dataflow: list[dict[str, Any]] = [
        # Loop-carried self-edge: previous step's prediction seeds the next.
        {
            "from": f"denoiser.{denoiser_output}",
            "to": f"denoiser.{denoiser_sample_input}",
        },
    ]
    phases: dict[str, Any] = {}

    if text_encoder_filename is not None:
        models["text_encoder"] = {
            "filename": text_encoder_filename,
            "type": "encoder",
        }
        # Route each text-encoder output to its denoiser conditioning input. SD
        # has one edge (hidden states -> encoder_hidden_states); SDXL has two
        # (concatenated hidden states + pooled text_embeds). `time_ids` is not
        # routed here — it is an external denoiser input the caller supplies.
        edges = text_encoder_edges or [(text_encoder_output, denoiser_conditioning_input)]
        for enc_out, denoiser_in in edges:
            dataflow.append(
                {"from": f"text_encoder.{enc_out}", "to": f"denoiser.{denoiser_in}"}
            )
        phases["text_encoder"] = {"run_on": "prompt_only"}

    if vae_filename is not None:
        models["vae"] = {"filename": vae_filename, "type": "vae"}
        # The VAE decodes the final post-scheduler latent (the sample port).
        dataflow.append(
            {
                "from": f"denoiser.{denoiser_sample_input}",
                "to": f"vae.{vae_latent_input}",
            }
        )
        phases["vae"] = {"run_on": "final_only"}

    strategy: dict[str, Any] = {
        "kind": "iterative",
        "denoiser": "denoiser",
        "num_steps": num_inference_steps,
        "timestep_input": denoiser_timestep_input,
        "scheduler_config": scheduler.to_metadata(),
    }
    if timesteps is not None:
        if len(timesteps) != num_inference_steps:
            raise ValueError(
                f"timesteps has {len(timesteps)} entries but num_inference_steps is "
                f"{num_inference_steps}"
            )
        strategy["timesteps"] = [float(t) for t in timesteps]
    if guidance_scale is not None:
        strategy["guidance_scale"] = guidance_scale
        if not math.isclose(guidance_scale, 1.0):
            strategy["cfg_conditioning_input"] = denoiser_conditioning_input
    if start_step:
        if not 0 < start_step < num_inference_steps:
            raise ValueError(
                f"start_step ({start_step}) must be in 1..{num_inference_steps - 1}"
            )
        strategy["start_step"] = start_step

    pipeline: dict[str, Any] = {
        "models": models,
        "dataflow": dataflow,
        "strategy": strategy,
    }
    if phases:
        pipeline["phases"] = phases
    return {"pipeline": pipeline}


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
        yaml.safe_dump(metadata, handle, sort_keys=False)
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
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def write_diffusion_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write ``inference_metadata.yaml`` into ``directory``.

    Extra keyword arguments are forwarded to
    :func:`build_diffusion_pipeline_metadata`. Returns the written path.
    """
    metadata = build_diffusion_pipeline_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path
