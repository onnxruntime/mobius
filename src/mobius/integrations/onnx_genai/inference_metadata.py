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
import shutil
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

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
    """Registry result for one generic image-input category."""

    name: str
    bindings: tuple[tuple[_Port, str, float | None], ...]
    transforms: Callable[[Any, dict[str, Any]], list[dict[str, Any]]]
    token_count_source: str


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


def _base_image_transforms(config: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    vision = getattr(config, "vision", None)
    size = values.get("size") or getattr(vision, "image_size", None)
    transforms: list[dict[str, Any]] = [{"op": "decode_rgb"}]
    if size:
        transforms.append(
            {
                "op": "resize",
                "size": int(size) if isinstance(size, (int, float)) else size,
                "interpolation": str(values.get("resample", "bicubic")).lower(),
            }
        )
    scale = values.get("rescale_factor")
    if scale is not None:
        transforms.append({"op": "rescale", "scale": float(scale)})
    mean = values.get("image_mean")
    std = values.get("image_std")
    if mean is not None and std is not None:
        transforms.append(
            {
                "op": "normalize",
                "mean": [float(value) for value in mean],
                "std": [float(value) for value in std],
            }
        )
    return transforms


def _packed_coordinate_transforms(config: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    vision = getattr(config, "vision", None)
    patch_size = values.get("patch_size") or getattr(vision, "patch_size", None)
    transforms = _base_image_transforms(config, values)
    if patch_size:
        transforms.append({"op": "patchify", "patch_size": int(patch_size), "flatten": True})
    transforms.append({"op": "pad", "pad_value": -1})
    return transforms


def _packed_grid_transforms(config: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    vision = getattr(config, "vision", None)
    patch_size = values.get("patch_size") or getattr(vision, "patch_size", None)
    transforms = _base_image_transforms(config, values)
    if patch_size:
        transforms.append({"op": "patchify", "patch_size": int(patch_size), "flatten": True})
    return transforms


def _crop_mask_transforms(config: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    transforms = _base_image_transforms(config, values)
    transforms.append({"op": "pad", "pad_value": 0})
    return transforms


def _match_packed_coordinates(ports: list[_Port]) -> _ImageProgram | None:
    pixels = _select_one(ports, lambda p: _is_float(p) and p.rank == 3)
    coordinates = _select_one(
        ports,
        lambda p: _is_integer(p) and p.rank == 3 and _static_dim(p, -1) == 2,
    )
    if pixels is None or coordinates is None:
        return None
    return _ImageProgram(
        name="packed_patch_coordinates",
        bindings=((pixels, "pixels", None), (coordinates, "patch_coordinates", -1)),
        transforms=_packed_coordinate_transforms,
        token_count_source="per_tile",
    )


def _match_packed_grid(ports: list[_Port]) -> _ImageProgram | None:
    pixels = _select_one(ports, lambda p: _is_float(p) and p.rank == 2)
    grid = _select_one(
        ports,
        lambda p: _is_integer(p) and p.rank == 2 and _static_dim(p, -1) == 3,
    )
    if pixels is None or grid is None:
        return None
    return _ImageProgram(
        name="packed_patch_grid",
        bindings=((pixels, "pixels", None), (grid, "grid_dimensions", None)),
        transforms=_packed_grid_transforms,
        token_count_source="from_grid",
    )


def _match_crop_mask(ports: list[_Port]) -> _ImageProgram | None:
    pixels = _select_one(ports, lambda p: _is_float(p) and p.rank == 4)
    original_size = _select_one(
        ports,
        lambda p: _is_integer(p) and p.rank == 2 and _static_dim(p, -1) == 2,
    )
    validity_mask = _select_one(ports, lambda p: _is_float(p) and p.rank == 3)
    if pixels is None or original_size is None or validity_mask is None:
        return None
    return _ImageProgram(
        name="cropped_image_with_mask",
        bindings=(
            (pixels, "pixels", None),
            (original_size, "original_size", None),
            (validity_mask, "validity_mask", 0),
        ),
        transforms=_crop_mask_transforms,
        token_count_source="per_patch",
    )


_IMAGE_PROCESSOR_REGISTRY: tuple[Callable[[list[_Port]], _ImageProgram | None], ...] = (
    _match_packed_coordinates,
    _match_packed_grid,
    _match_crop_mask,
)


def _resolve_image_program(model: Any) -> _ImageProgram:
    ports = [_port(value) for value in model.graph.inputs]
    for resolve in _IMAGE_PROCESSOR_REGISTRY:
        program = resolve(ports)
        if program is not None:
            return program
    signature = [(port.dtype, port.rank, port.dims) for port in ports]
    raise ValueError(
        "No registered generic image processor matches the vision component "
        f"input signature {signature}. Add a structural registry entry rather "
        "than branching on model_type or model name."
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


def _source_asset_path(source: str, filename: str) -> str | None:
    if os.path.isdir(source):
        path = os.path.join(source, filename)
        return path if os.path.isfile(path) else None
    cached = _cached_source_assets(source).get(filename)
    if cached is not None and os.path.isfile(cached):
        return cached
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(source, filename)
    except Exception:
        return None


def _processor_values(source: str | None, config: Any) -> dict[str, Any]:
    """Load plain processor parameters without architecture dispatch."""
    values: dict[str, Any] = {}
    if source:
        for filename in (
            "preprocessor_config.json",
            "processor_config.json",
            "image_processor.json",
        ):
            path = _source_asset_path(source, filename)
            if path is not None:
                try:
                    with open(path, encoding="utf-8") as handle:
                        values.update(json.load(handle))
                except (OSError, ValueError):
                    _LOGGER.warning("Could not read processor config %s", path)
                break
    image_processor = values.get("image_processor")
    if isinstance(image_processor, dict):
        values.update(image_processor)

    vision = getattr(config, "vision", None)
    for name in (
        "image_size",
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
    ):
        value = getattr(vision, name, None)
        if value is None:
            value = getattr(config, name, None)
        if value is not None:
            values.setdefault(name, value)
    size = values.get("size")
    if isinstance(size, dict):
        values["size"] = (
            size.get("height") or size.get("shortest_edge") or size.get("longest_edge")
        )
    values.setdefault("size", values.get("image_size"))
    return values


def _shape_key(port: _Port) -> tuple[str, tuple[str, ...]]:
    return port.dtype, tuple(str(dim) for dim in port.dims)


def _state_and_kv_pairs(
    decoder_inputs: list[_Port],
    decoder_outputs: list[_Port],
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Pair actual remaining state ports positionally and classify by shape."""
    if len(decoder_inputs) != len(decoder_outputs):
        raise ValueError(
            "Decoder state I/O is not pairable after explicit data/core port "
            f"classification: {len(decoder_inputs)} inputs versus "
            f"{len(decoder_outputs)} outputs."
        )
    kv_inputs: list[str] = []
    kv_outputs: list[str] = []
    state_pairs: list[dict[str, str]] = []
    for input_port, output_port in zip(decoder_inputs, decoder_outputs, strict=True):
        if _shape_key(input_port) == _shape_key(output_port):
            state_pairs.append(
                {
                    "input": input_port.name,
                    "output": output_port.name,
                    "init": "zeros",
                    "update": "replace",
                }
            )
        else:
            kv_inputs.append(input_port.name)
            kv_outputs.append(output_port.name)
    return kv_inputs, kv_outputs, state_pairs


def _decoder_io(
    decoder: Any,
    routed_inputs: set[str],
    config: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    inputs = [_port(value) for value in decoder.graph.inputs]
    outputs = [_port(value) for value in decoder.graph.outputs]
    io: dict[str, Any] = {
        "inputs": [port.name for port in inputs],
        "outputs": [port.name for port in outputs],
    }

    routed = [port for port in inputs if port.name in routed_inputs]
    embedded = next((port for port in routed if _is_float(port) and port.rank == 3), None)
    if embedded is not None:
        io["inputs_embeds_input"] = embedded.name

    integer_inputs = [port for port in inputs if _is_integer(port)]
    rank3_position = next((port for port in integer_inputs if port.rank == 3), None)
    attention_mask = next(
        (port for port in integer_inputs if any("+" in str(dim) for dim in port.dims)),
        None,
    )
    if attention_mask is not None:
        io["attention_mask_input"] = attention_mask.name

    position = rank3_position
    if position is None:
        remaining_rank2 = [
            port for port in integer_inputs if port.rank == 2 and port is not attention_mask
        ]
        position = remaining_rank2[0] if remaining_rank2 else None
    if position is not None:
        io["position_ids_input"] = position.name

    token = next(
        (
            port
            for port in integer_inputs
            if port.rank == 2 and port is not attention_mask and port is not position
        ),
        None,
    )
    if token is not None:
        io["token_input"] = token.name

    vocab_size = getattr(config, "vocab_size", None)
    logits = next(
        (
            port
            for port in outputs
            if port.rank == 3
            and isinstance(vocab_size, int)
            and _static_dim(port, -1) == vocab_size
        ),
        None,
    )
    if logits is None:
        logits = next((port for port in outputs if port.rank == 3), None)
    if logits is None and outputs:
        logits = outputs[0]
    if logits is not None:
        io["logits_output"] = logits.name

    core_inputs = routed_inputs | {
        port.name for port in (attention_mask, position, token) if port is not None
    }
    state_inputs = [port for port in inputs if port.name not in core_inputs]
    state_outputs = [port for port in outputs if logits is None or port.name != logits.name]
    kv_inputs, kv_outputs, state_pairs = _state_and_kv_pairs(state_inputs, state_outputs)
    if kv_inputs:
        io["kv_inputs"] = kv_inputs
        io["kv_outputs"] = kv_outputs
        io["kv_update"] = "append"
    if state_pairs:
        io["state_pairs"] = state_pairs

    positions = None
    if position is not None:
        rank = 3 if position.rank == 3 else 1
        positions = {
            "input": position.name,
            "rank": rank,
            "dtype": position.dtype,
            "continuation": "from_grid" if rank > 1 else "linear_increment",
        }
        if rank == 3:
            positions["axes"] = ["temporal", "height", "width"]
            sections = getattr(config, "mrope_section", None)
            if sections:
                positions["sections"] = [int(section) for section in sections]
    return io, positions


def _component_filenames(pkg: Any) -> dict[str, str]:
    multiple = len(pkg) > 1
    return {name: f"{name}/model.onnx" if multiple else "model.onnx" for name in pkg}


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
    """Return whether a package has the structural encoder/fusion/decoder shape."""
    try:
        names = set(pkg)
        if not {"vision_encoder", "embedding", "decoder"} <= names:
            return False
        _resolve_image_program(pkg["vision_encoder"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    else:
        return True


def build_native_vlm_package_metadata(
    pkg: Any,
    *,
    config: Any,
    source: str | None = None,
    decoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a native VLM contract by inspecting every component graph.

    Processor selection is registry-driven from graph rank/dtype/shape
    signatures. No model type, architecture name, or model-name branch
    participates in dispatch.
    """
    if not is_native_vlm_package(pkg):
        raise ValueError(
            "Native VLM emission requires vision_encoder, embedding, and decoder "
            "components with a registered structural image-input signature."
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
    for source_name, source_ports in ports.items():
        output_by_name = {port.name: port for port in source_ports["outputs"]}
        for target_name, target_ports in ports.items():
            if source_name == target_name:
                continue
            for target_port in target_ports["inputs"]:
                source_port = output_by_name.get(target_port.name)
                if source_port is None or source_port.dtype != target_port.dtype:
                    continue
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
    decoder_io, positions = _decoder_io(pkg[decoder_name], routed_decoder_inputs, config)

    image_program = _resolve_image_program(pkg["vision_encoder"])
    processor_values = _processor_values(source, config)
    preprocessing_outputs = []
    grid_summary = None
    for port, content, pad_value in image_program.bindings:
        output: dict[str, Any] = {
            "name": port.name,
            "content": content,
            "dtype": port.dtype,
        }
        if pad_value is not None:
            output["pad_value"] = pad_value
        preprocessing_outputs.append(output)
        if content == "grid_dimensions":
            grid_summary = port.name

    if positions is not None and positions["rank"] > 1 and grid_summary is not None:
        positions["processor_summaries"] = [grid_summary]

    downstream_to_decoder = {
        edge["from"].split(".", 1)[0]
        for edge in dataflow
        if edge["to"].startswith(f"{decoder_name}.")
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
                "inputs": [port.name for port in ports[name]["inputs"]],
                "outputs": [port.name for port in ports[name]["outputs"]],
            }
        else:
            role = "encoder"
            run_on = "every_step" if name in downstream_to_decoder else "prompt_only"
            io = {
                "inputs": [port.name for port in ports[name]["inputs"]],
                "outputs": [port.name for port in ports[name]["outputs"]],
            }
        models[name] = {
            "filename": filenames[name],
            "type": role,
            "io": io,
        }
        if name == decoder_name:
            models[name]["tokenizer"] = "tokenizer.json"
        phases[name] = {"run_on": run_on}

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
                "run_on": phases[name]["run_on"],
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
    if image_program.token_count_source == "per_tile":
        tokens = (
            processor_values.get("image_seq_length")
            or processor_values.get("max_soft_tokens")
            or getattr(config, "mm_tokens_per_image", None)
            or getattr(getattr(config, "vision", None), "mm_tokens_per_image", None)
        )
        if not tokens:
            pixel_port = next(
                (
                    port
                    for port, content, _pad_value in image_program.bindings
                    if content == "pixels"
                ),
                None,
            )
            if pixel_port is not None and pixel_port.rank == 3:
                tokens = _static_dim(pixel_port, 1)
        if tokens:
            vision_config["tokens_per_tile"] = int(tokens)
    elif image_program.token_count_source == "per_patch":
        vision_config["tokens_per_patch"] = 1
    vision_config = {key: value for key, value in vision_config.items() if value is not None}

    metadata = dict(decoder_metadata or {})
    capabilities = list(metadata.get("required_capabilities", []))
    for capability in (
        "multimodal_image_preprocessing",
        "autoregressive_every_step_components",
    ):
        if capability not in capabilities:
            capabilities.append(capability)
    if positions is not None and positions["rank"] > 1:
        capabilities.append("multiaxis_positions")
    if decoder_io.get("state_pairs"):
        capabilities.append("loop_state")
    metadata["required_capabilities"] = capabilities
    metadata.setdefault("model", {})["io"] = decoder_io
    metadata["preprocessing"] = {
        "image": {
            "transforms": image_program.transforms(config, processor_values),
            "outputs": preprocessing_outputs,
        }
    }
    metadata["pipeline"] = {
        "models": models,
        "dataflow": dataflow,
        "strategy": {"kind": "composite", "stages": stages},
        "phases": phases,
        "vision": vision_config,
    }
    if positions is not None:
        metadata["pipeline"]["positions"] = positions
    return metadata


def _copy_runtime_assets(
    output_dir: str,
    source: str | None,
) -> dict[str, str]:
    if not source:
        return {}
    os.makedirs(output_dir, exist_ok=True)
    for filename in _RUNTIME_ASSET_NAMES:
        source_path = _source_asset_path(source, filename)
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

            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
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

    return {
        Path(filename).stem: os.path.join(output_dir, filename)
        for filename in _RUNTIME_ASSET_NAMES
        if os.path.isfile(os.path.join(output_dir, filename))
    }


def write_native_vlm_package_metadata(
    pkg: Any,
    directory: str,
    *,
    config: Any,
    source: str | None = None,
    kv_native_dtype: str | None = None,
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
        decoder_metadata=decoder_metadata_from_config(config, kv_native_dtype=kv_native_dtype),
    )
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    artifacts = {"inference_metadata": path}
    artifacts.update(_copy_runtime_assets(directory, source))
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
    use_karras_sigmas: bool = False
    use_exponential_sigmas: bool = False

    def to_metadata(self) -> dict[str, Any]:
        meta = {
            "kind": self.kind,
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "prediction_type": self.prediction_type,
        }
        if self.use_karras_sigmas:
            meta["use_karras_sigmas"] = True
        if self.use_exponential_sigmas:
            meta["use_exponential_sigmas"] = True
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
        if "eulerancestral" in name:
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
            prediction_type=str(config.get("prediction_type", "epsilon")),
            use_karras_sigmas=bool(config.get("use_karras_sigmas")),
            use_exponential_sigmas=bool(config.get("use_exponential_sigmas")),
        )


def load_diffusers_scheduler_config(source: str | None) -> SchedulerConfig | None:
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

            path = hf_hub_download(source, "scheduler/scheduler_config.json")
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


def build_language_diffusion_pipeline_metadata(
    *,
    mask_token_id: int,
    num_inference_steps: int,
    model_filename: str = "model.onnx",
    input_ids_port: str = "input_ids",
    logits_port: str = "logits",
    block_length: int | None = None,
    temperature: float | None = None,
    guidance_scale: float | None = None,
) -> dict[str, Any]:
    """Build the onnx-genai ``inference_metadata`` for a masked language-diffusion model.

    For a masked (discrete) language-diffusion model (e.g. LLaDA / Dream).

    The model is a mask predictor: it takes an int64 token sequence on
    ``input_ids_port`` (prompt tokens plus a masked generation region) and emits
    ``[B, S, V]`` logits on ``logits_port``. onnx-genai's ``masked_diffusion``
    scheduler drives the reverse process — each step commits the highest-confidence
    still-masked positions (LLaDA low-confidence remasking) via a loop-carried
    ``logits -> input_ids`` self-edge, unmasking progressively.

    Args:
        mask_token_id: The ``[MASK]`` token id (e.g. 126336 for LLaDA-8B).
        num_inference_steps: Total reverse-process steps (``strategy.num_steps``).
        model_filename: The mask-predictor ONNX filename.
        input_ids_port / logits_port: Model I/O port names.
        block_length: Semi-autoregressive block length in tokens. When set, the
            generation region is decoded in contiguous left-to-right blocks and
            ``num_inference_steps`` must be divisible by the block count.
        temperature: Gumbel-max sampling temperature (default 0 = argmax).
        guidance_scale: Unsupervised classifier-free guidance multiplier. LLaDA's
            effective multiplier is ``cfg_scale + 1``, so pass ``cfg_scale + 1``.

    Returns:
        A dict with a top-level ``pipeline`` key, ready to serialize to
        ``inference_metadata.yaml``.
    """
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    if block_length is not None and block_length < 1:
        raise ValueError("block_length must be >= 1")

    scheduler_config: dict[str, Any] = {
        "kind": "masked_diffusion",
        "mask_token_id": int(mask_token_id),
    }
    if temperature is not None:
        scheduler_config["temperature"] = float(temperature)
    if block_length is not None:
        scheduler_config["block_length"] = int(block_length)

    strategy: dict[str, Any] = {
        "kind": "iterative",
        "denoiser": "denoiser",
        "num_steps": num_inference_steps,
        "scheduler_config": scheduler_config,
    }
    if guidance_scale is not None and not math.isclose(guidance_scale, 1.0):
        strategy["guidance_scale"] = guidance_scale

    pipeline: dict[str, Any] = {
        "models": {"denoiser": {"filename": model_filename, "type": "denoiser"}},
        # Loop-carried self-edge: the emitted logits refine the token sequence.
        "dataflow": [{"from": f"denoiser.{logits_port}", "to": f"denoiser.{input_ids_port}"}],
        "strategy": strategy,
    }
    return {"pipeline": pipeline}


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
                "run_on": "prompt_only",
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
                "run_on": "prompt_only",
            },
            {
                "name": "decode",
                "strategy": {"kind": "autoregressive", "decoder": "decoder"},
                "run_on": "every_step",
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
        "phases": phases,
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

    Returns:
        A dict with a top-level ``pipeline`` key and any decoder capabilities.
    """
    metadata = dict(decoder_metadata or {})
    metadata["pipeline"] = {
        "models": {
            "encoder": {"filename": encoder_filename, "type": "encoder"},
            "decoder": {
                "filename": decoder_filename,
                "type": "decoder",
                "tokenizer": tokenizer_filename,
            },
        },
        "dataflow": [
            {
                "from": "encoder.encoder_hidden_states",
                "to": "decoder.encoder_hidden_states",
                "dtype": activation_dtype,
                "device_transfer": False,
            }
        ],
        "strategy": {
            "kind": "composite",
            "stages": [
                {
                    "name": "encode_audio",
                    "strategy": {"kind": "single_pass", "model": "encoder"},
                    "run_on": "prompt_only",
                },
                {
                    "name": "decode_transcript",
                    "strategy": {"kind": "autoregressive", "decoder": "decoder"},
                    "run_on": "every_step",
                },
            ],
        },
        "phases": {
            "encoder": {"run_on": "prompt_only"},
            "decoder": {"run_on": "every_step"},
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


def build_audio_codec_pipeline_metadata(
    *,
    encoder_filename: str = "encoder/model.onnx",
    decoder_filename: str = "decoder/model.onnx",
    codes_dtype: str = "int64",
) -> dict[str, Any]:
    """Build metadata for an audio-to-audio neural codec pipeline.

    This is the pure single-pass composite shape (DESIGN.md §20): an audio
    encoder maps a waveform to ``codes``, and a decoder reconstructs a waveform
    from those codes. Both stages run once over the shared tensor pool (there is
    no autoregressive decode and no tokenizer), wired ``encoder.codes ->
    decoder.codes``.

    Args:
        encoder_filename: Waveform-to-codes encoder ONNX filename.
        decoder_filename: Codes-to-waveform decoder ONNX filename.
        codes_dtype: Metadata dtype of the ``codes`` tensor exchanged between the
            two stages (neural codecs typically emit ``int64`` code indices).

    Returns:
        A dict with a top-level ``pipeline`` key. No decoder capabilities are
        emitted because the pipeline produces tensors, not tokens.
    """
    return {
        "pipeline": {
            "models": {
                "encoder": {"filename": encoder_filename, "type": "audio_encoder"},
                "decoder": {"filename": decoder_filename, "type": "vocoder"},
            },
            "dataflow": [
                {
                    "from": "encoder.codes",
                    "to": "decoder.codes",
                    "dtype": codes_dtype,
                    "device_transfer": False,
                }
            ],
            "strategy": {
                "kind": "composite",
                "stages": [
                    {
                        "name": "encode_waveform",
                        "strategy": {"kind": "single_pass", "model": "encoder"},
                        "run_on": "prompt_only",
                    },
                    {
                        "name": "decode_waveform",
                        "strategy": {"kind": "single_pass", "model": "decoder"},
                        "run_on": "prompt_only",
                    },
                ],
            },
            "phases": {
                "encoder": {"run_on": "prompt_only"},
                "decoder": {"run_on": "prompt_only"},
            },
        }
    }


def write_audio_codec_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write composite audio-codec metadata into ``directory``."""
    metadata = build_audio_codec_pipeline_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def build_tts_pipeline_metadata(
    *,
    num_code_groups: int,
    max_frames: int = 2000,
    talker_filename: str = "talker/model.onnx",
    code_predictor_filename: str = "code_predictor/model.onnx",
    pre_embedder_filename: str = "talker_step_embedder/model.onnx",
    prefill_embedder_filename: str | None = "talker_prefill_embedder/model.onnx",
    tokenizer_filename: str = "tokenizer.json",
    activation_dtype: str = "fp32",
    decoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata for a pre-embedder-driven multi-decoder TTS pipeline.

    This is the real Qwen3-TTS shape (DESIGN.md §20.3, ``nested_autoregressive``
    with the optional ``pre_embedder`` extension): an OUTER ``talker`` AR loop
    where each frame drives an INNER ``code_predictor`` AR loop of
    ``num_code_groups`` steps (seeded by the talker's ``last_hidden_state``).
    Unlike the plain nested shape, the talker is **not** driven by ``input_ids``:
    each frame its ``inputs_embeds`` is materialized from the previous frame's
    codes by the ``talker_step_embedder`` pre-embedder (``frame_codes
    [+ text_embed] -> inputs_embeds``), keeping the engine generic.

    When ``prefill_embedder_filename`` is set (the default), a
    ``talker_prefill_embedder`` prompt-phase component is also emitted. It maps
    the tokenized prompt ``text_ids -> prefill_embeds + trailing_text_embeds``:
    the runtime feeds ``prefill_embeds`` to the talker on frame 0 and threads
    ``trailing_text_embeds[:, k-1, :]`` as the pre-embedder's ``text_embed`` on
    frames k>=1 (see the ``prefill_embedder`` field). Pass ``None`` to emit the
    prefill-less shape (talker frame 0 + ``text_embed`` fed zeros).

    The engine-driven components are emitted (``talker``, ``code_predictor``,
    ``talker_step_embedder``, and ``talker_prefill_embedder`` when present). The
    package's ``embedding`` and optional ``speaker_encoder`` models are internal
    weight sources already folded into the pre-/prefill-embedders, so they are
    not declared as pipeline models. There is **no in-package vocoder** — the
    assembled ``talker.output_codes`` are decoded by a separate codec model.

    Args:
        num_code_groups: Codes collected per outer frame (RVQ residual count).
        max_frames: Maximum number of outer talker frames to generate.
        talker_filename: Outer decoder (talker) ONNX filename.
        code_predictor_filename: Inner decoder ONNX filename.
        pre_embedder_filename: ``talker_step_embedder`` ONNX filename.
        prefill_embedder_filename: ``talker_prefill_embedder`` ONNX filename, or
            ``None`` to omit the prefill/trailing-text path.
        tokenizer_filename: Tokenizer filename used by the talker.
        decoder_metadata: Optional output from
            :func:`decoder_metadata_from_config`; its decoder capabilities are
            retained at the document top level.

    Returns:
        A dict with a top-level ``pipeline`` key and any decoder capabilities.
    """
    if num_code_groups < 1:
        raise ValueError("num_code_groups must be at least 1")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    models: dict[str, Any] = {
        "talker": {
            "filename": talker_filename,
            "type": "decoder",
            "tokenizer": tokenizer_filename,
        },
        "talker_step_embedder": {
            "filename": pre_embedder_filename,
            "type": "embedding",
        },
        "code_predictor": {
            "filename": code_predictor_filename,
            "type": "decoder",
        },
    }
    dataflow: list[dict[str, Any]] = [
        {
            "from": "talker_step_embedder.inputs_embeds",
            "to": "talker.inputs_embeds",
            "dtype": activation_dtype,
            "device_transfer": False,
        },
        {
            "from": "talker.last_hidden_state",
            "to": "code_predictor.inputs_embeds",
            "dtype": activation_dtype,
            "device_transfer": False,
        },
    ]
    stage_strategy: dict[str, Any] = {
        "kind": "nested_autoregressive",
        "outer": "talker",
        "inner": "code_predictor",
        "pre_embedder": {
            "component": "talker_step_embedder",
            "frame_codes_input": "frame_codes",
            "text_embed_input": "text_embed",
        },
        "num_code_groups": num_code_groups,
        "max_tokens": max_frames,
    }
    phases: dict[str, Any] = {
        "talker": {"run_on": "every_step"},
        "talker_step_embedder": {"run_on": "on_demand"},
        "code_predictor": {"run_on": "every_step"},
    }

    if prefill_embedder_filename is not None:
        models["talker_prefill_embedder"] = {
            "filename": prefill_embedder_filename,
            "type": "embedding",
        }
        # Runs once in the prompt phase; the runtime seeds the declared
        # `prompt_input` with the tokenized prompt and reads the two named
        # outputs from the pool. Every port is declared explicitly (the engine
        # never guesses tensor names).
        stage_strategy["prefill_embedder"] = {
            "component": "talker_prefill_embedder",
            "prompt_input": "text_ids",
            "prefill_output": "prefill_embeds",
            "trailing_output": "trailing_text_embeds",
        }
        phases["talker_prefill_embedder"] = {"run_on": "prompt_only"}

    metadata = dict(decoder_metadata or {})
    metadata["pipeline"] = {
        "models": models,
        "dataflow": dataflow,
        "strategy": {
            "kind": "composite",
            "stages": [
                {
                    "name": "generate_codes",
                    "strategy": stage_strategy,
                    "run_on": "every_step",
                },
            ],
        },
        "phases": phases,
    }
    return metadata


def write_tts_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write pre-embedder-driven TTS metadata into ``directory``."""
    metadata = build_tts_pipeline_metadata(**kwargs)
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
