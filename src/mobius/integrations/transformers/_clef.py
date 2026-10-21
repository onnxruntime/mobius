# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned Clef checkpoint detection and bounded-memory joint-head loading."""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
from collections.abc import Mapping

import onnx_ir as ir
import torch
from huggingface_hub import hf_hub_download
from onnx_ir import tensor_adapters

from mobius._builder import build_from_module, resolve_dtype
from mobius.integrations._weight_loading import (
    StreamingTransformedWeightSource,
    StreamingWeightPlan,
    StreamingWeightSource,
    _resolve_shard_paths,
    stream_preprocessed_safetensors_to_model,
)
from mobius.models.clef import (
    CLEF_FLASH_MODEL_ID,
    CLEF_FLASH_REVISION,
    ClefConfig,
    ClefFlashModel,
    clef_head_source_name,
)
from mobius.tasks._clef import ClefDecisionTask


def is_clef_checkpoint(model_id: str) -> bool:
    """Detect the official checkpoint or a local release with its head sidecar."""
    return model_id == CLEF_FLASH_MODEL_ID or (
        pathlib.Path(model_id).is_dir()
        and (pathlib.Path(model_id) / "joint_head_config.json").is_file()
    )


def _asset(model_id: str, filename: str, revision: str | None) -> str:
    if pathlib.Path(model_id).is_dir():
        path = pathlib.Path(model_id) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Clef checkpoint is missing {path}")
        return str(path)
    return hf_hub_download(model_id, filename, revision=revision)


def _source_name(component: str, name: str) -> tuple[str, int | None]:
    name = name.removeprefix(f"{component}.")
    if component == "decision_head":
        if name == "output_embedding.weight":
            return "lm_head.weight", None
        return clef_head_source_name(name)
    if component == "decoder":
        return "model.language_model." + name.removeprefix("model."), None
    if component == "embedding":
        return "model.language_model." + name, None
    name = "model." + name
    return (
        name.replace(".mlp.up_proj.", ".mlp.linear_fc1.").replace(
            ".mlp.down_proj.", ".mlp.linear_fc2."
        ),
        None,
    )


def _split_qkv(
    tensor: torch.Tensor, _name: str, *, index: int, dtype: torch.dtype
) -> torch.Tensor:
    return tensor.chunk(3, dim=0)[index].to(dtype=dtype).contiguous()


def _plan(
    index: Mapping[str, tuple[str, list[int], str]],
    initializers: Mapping[str, ir.Value],
    *,
    component: str,
    recognized: set[str],
    tied_embeddings: bool,
) -> StreamingWeightPlan:
    targets: dict[str, StreamingWeightSource | StreamingTransformedWeightSource] = {}
    used = set()
    for name, value in initializers.items():
        if value.const_value is not None:
            continue
        source, split = _source_name(component, name)
        if source == "lm_head.weight" and source not in index and tied_embeddings:
            source = "model.language_model.embed_tokens.weight"
        if source not in index:
            raise ValueError(f"Clef checkpoint is missing {source!r} for {component}/{name}")
        _, source_shape, source_dtype = index[source]
        if source_dtype not in {"BF16", "F16", "F32"}:
            raise ValueError(f"Clef requires floating-point weights; {source}: {source_dtype}")
        used.add(source)
        if split is None:
            targets[name] = StreamingWeightSource(source)
        else:
            assert value.shape is not None and value.dtype is not None
            if any(not isinstance(dim, int) for dim in value.shape):
                raise ValueError(f"Clef parameter {name!r} must have a static shape")
            target_shape = tuple(dim for dim in value.shape if isinstance(dim, int))
            expected_shape = (target_shape[0] * 3, *target_shape[1:])
            if tuple(source_shape) != expected_shape:
                raise ValueError(f"Invalid fused Clef QKV shape for {source}: {source_shape}")
            targets[name] = StreamingTransformedWeightSource(
                source_name=source,
                expected_source_shape=expected_shape,
                expected_source_dtype=source_dtype,
                expected_target_shape=target_shape,
                expected_target_dtype=value.dtype,
                transform=functools.partial(
                    _split_qkv,
                    index=split,
                    dtype=tensor_adapters.to_torch_dtype(value.dtype),
                ),
            )
    ignored = {}
    for source in index.keys() - used:
        if source in recognized:
            ignored[source] = "Owned by another Clef component"
        elif source.startswith(("mtp.", "mtp_")):
            ignored[source] = "Training-only multi-token prediction head"
        else:
            raise ValueError(f"Unrecognized Clef checkpoint tensor: {source}")
    return StreamingWeightPlan(targets=targets, ignored=ignored)


def build_clef_model(
    model_id: str,
    *,
    revision: str | None,
    dtype: str | ir.DataType | None,
    execution_provider: str,
    load_weights: bool,
):
    """Build all four decision components, pinning every config/weight read."""
    from mobius.integrations.transformers._builder import (
        _load_transformers_config,
        _select_primary_config,
    )

    if model_id == CLEF_FLASH_MODEL_ID and revision is None:
        revision = CLEF_FLASH_REVISION
    parent, _ = _load_transformers_config(model_id, revision=revision, trust_remote_code=False)
    if parent is None:
        raise ValueError(f"Clef checkpoint {model_id!r} has no backbone config")
    text, parent, model_type = _select_primary_config(parent)
    if model_type != "qwen3_5":
        raise ValueError(f"Clef-Flash requires a dense Qwen3.5 backbone, got {model_type}")
    config = ClefConfig.from_transformers(text, parent_config=parent)
    assert isinstance(config, ClefConfig)
    head = json.loads(
        pathlib.Path(_asset(model_id, "joint_head_config.json", revision)).read_text(
            encoding="utf-8"
        )
    )
    expected = {"hidden_size", "width", "routing_layers", "layers", "heads", "feedforward"}
    if (
        not isinstance(head, dict)
        or set(head) != expected
        or head["hidden_size"] != config.hidden_size
    ):
        raise ValueError("Clef joint-head config does not match the backbone/topology")
    if any(type(value) is not int for value in head.values()):
        raise ValueError("Clef joint-head dimensions must be integers")
    resolved_dtype = resolve_dtype(dtype) if dtype is not None else config.dtype
    assert resolved_dtype is not None
    config = dataclasses.replace(
        config,
        model_type="clef_flash",
        head_width=head["width"],
        head_routing_layers=head["routing_layers"],
        head_layers=head["layers"],
        head_heads=head["heads"],
        head_feedforward=head["feedforward"],
        dtype=resolved_dtype,
    )
    if config.quantization is not None or config.component_quantization is not None:
        raise ValueError("Clef-Flash currently requires an unquantized source checkpoint")
    module = ClefFlashModel(config)
    package = build_from_module(
        module, config, ClefDecisionTask(), execution_provider=execution_provider
    )
    for model in package.values():
        model.metadata_props["mobius.source_revision"] = revision or "local"
        model.metadata_props["mobius.task"] = "clef-decision"
    if load_weights:
        paths = _resolve_shard_paths(model_id, revision)
        head_path = _asset(model_id, "joint_head.safetensors", revision)
        if head_path not in paths:
            paths.append(head_path)
        recognized = {
            _source_name(component, name)[0]
            for component, model in package.items()
            for name, value in model.graph.initializers.items()
            if value.const_value is None
        }
        recognized.add("model.language_model.embed_tokens.weight")
        reports = {}
        for component, model in package.items():
            reports[component] = stream_preprocessed_safetensors_to_model(
                model,
                model_id,
                functools.partial(
                    _plan,
                    component=component,
                    recognized=recognized,
                    tied_embeddings=config.tie_word_embeddings,
                ),
                revision=revision,
                _resolved_paths=paths,
            )
        package.weight_loading_report = {
            "format": "mobius.weight-loading-report.v1",
            "components": reports,
        }
    return package
