# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded-memory safetensors loading for native GPT-OSS MXFP4 experts."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping

import onnx_ir as ir
import torch
from huggingface_hub.utils import EntryNotFoundError

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.integrations._weight_loading import (
    StreamingTransformedWeightSource,
    StreamingWeightPlan,
    StreamingWeightSource,
    _local_weight_paths,
    _resolve_shard_paths,
    stream_preprocessed_safetensors_to_model,
)
from mobius.models.gptoss import (
    _native_mxfp4_projection_specs,
    _reinterpret_mxfp4_scales_unchecked,
    _validate_mxfp4_scale_bytes,
    repack_gptoss_mxfp4_blocks,
)

_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})


def _lazy_safetensors_source_parent_aliases(paths: list[str]) -> frozenset[pathlib.Path]:
    """Return both directory identities from which lazy shards may be read."""
    # HF snapshots can contain shard symlinks into the blob cache. Resolving
    # the parent preserves the snapshot directory, while resolving the whole
    # shard path follows the symlink and identifies the blob directory.
    return frozenset(
        {pathlib.Path(path).parent.resolve() for path in paths}
        | {pathlib.Path(path).resolve().parent for path in paths}
    )


def _repack_blocks(tensor: torch.Tensor, _source_name: str) -> torch.Tensor:
    return repack_gptoss_mxfp4_blocks(tensor)


def _reinterpret_scales(tensor: torch.Tensor, _source_name: str) -> torch.Tensor:
    return _reinterpret_mxfp4_scales_unchecked(tensor)


def _validate_scales(tensor: torch.Tensor, source_name: str) -> None:
    _validate_mxfp4_scale_bytes(tensor, source_name)


def _require_header(
    key_index: Mapping[str, tuple[str, list[int], str]],
    source_name: str,
    expected_shape: tuple[int, ...],
    *,
    expected_dtypes: frozenset[str],
    description: str,
) -> None:
    metadata = key_index.get(source_name)
    if metadata is None:
        raise ValueError(
            f"Malformed GPT-OSS MXFP4 checkpoint: missing {description} "
            f"tensor {source_name!r}."
        )
    _path, shape, dtype = metadata
    if tuple(shape) != expected_shape:
        raise ValueError(
            f"GPT-OSS {description} {source_name!r} must have shape "
            f"{expected_shape}, got {tuple(shape)}"
        )
    if dtype not in expected_dtypes:
        raise ValueError(
            f"GPT-OSS {description} {source_name!r} must have dtype "
            f"{sorted(expected_dtypes)}, got {dtype}"
        )


def build_gptoss_mxfp4_streaming_plan(
    config: ArchitectureConfig,
    key_index: Mapping[str, tuple[str, list[int], str]],
    initializers: Mapping[str, ir.Value],
) -> StreamingWeightPlan:
    """Classify and validate an entire native checkpoint from safetensors headers."""
    specs = _native_mxfp4_projection_specs(config)
    num_experts = config.num_local_experts
    assert num_experts is not None

    expected_expert_keys: set[str] = set()
    for mlp_root in sorted(specs):
        for projection in specs[mlp_root]:
            base = f"{mlp_root}.experts.{projection}"
            expected_expert_keys.update((f"{base}_blocks", f"{base}_scales"))
        expected_expert_keys.update(
            (
                f"{mlp_root}.experts.gate_up_proj_bias",
                f"{mlp_root}.experts.down_proj_bias",
            )
        )
    actual_expert_keys = {key for key in key_index if ".mlp.experts." in key}
    if actual_expert_keys != expected_expert_keys:
        raise ValueError(
            "Malformed GPT-OSS MXFP4 checkpoint: every expected MoE layer must "
            "contain exactly gate_up_proj/down_proj blocks, scales, and biases. "
            f"Missing: {sorted(expected_expert_keys - actual_expert_keys)}; "
            f"unexpected: {sorted(actual_expert_keys - expected_expert_keys)}."
        )

    targets: dict[
        str,
        StreamingWeightSource | StreamingTransformedWeightSource,
    ] = {}
    target_constants: dict[str, torch.Tensor] = {}
    claimed_targets: set[str] = set()

    for mlp_root in sorted(specs):
        for projection, (block_shape, target) in specs[mlp_root].items():
            base = f"{mlp_root}.experts.{projection}"
            block_name = f"{base}_blocks"
            scale_name = f"{base}_scales"
            scale_shape = block_shape[:-1]
            _require_header(
                key_index,
                block_name,
                block_shape,
                expected_dtypes=frozenset({"U8"}),
                description="MXFP4 blocks",
            )
            _require_header(
                key_index,
                scale_name,
                scale_shape,
                expected_dtypes=frozenset({"U8"}),
                description="MXFP4 scales",
            )

            output_pairs = block_shape[1] // 2
            scratch_bytes = (
                block_shape[0] * block_shape[2] * block_shape[3] * min(64, output_pairs)
            )
            weight_target = f"{mlp_root}.{target}_experts_weights"
            scale_target = f"{mlp_root}.{target}_scales"
            global_scale_target = f"{mlp_root}.{target}_global_scales"
            targets[weight_target] = StreamingTransformedWeightSource(
                source_name=block_name,
                expected_source_shape=block_shape,
                expected_source_dtype="U8",
                transform=_repack_blocks,
                scratch_bytes=scratch_bytes,
            )
            targets[scale_target] = StreamingTransformedWeightSource(
                source_name=scale_name,
                expected_source_shape=scale_shape,
                expected_source_dtype="U8",
                transform=_reinterpret_scales,
                validate_tensor=_validate_scales,
            )
            target_constants[global_scale_target] = torch.ones(
                num_experts, dtype=torch.float32
            )
            claimed_targets.update((weight_target, scale_target, global_scale_target))

        source_biases: dict[str, tuple[str, tuple[int, ...]]] = {
            f"{mlp_root}.experts.gate_up_proj_bias": (
                f"{mlp_root}.fc1_experts_bias",
                (num_experts, 2 * config.intermediate_size),
            ),
            f"{mlp_root}.experts.down_proj_bias": (
                f"{mlp_root}.fc2_experts_bias",
                (num_experts, config.hidden_size),
            ),
        }
        for source_name, (target_name, shape) in source_biases.items():
            _require_header(
                key_index,
                source_name,
                shape,
                expected_dtypes=_FLOAT_DTYPES,
                description="expert bias",
            )
            targets[target_name] = StreamingWeightSource(source_name)
            claimed_targets.add(target_name)

        router_sources: dict[str, tuple[str, tuple[int, ...]]] = {
            f"{mlp_root}.router.weight": (
                f"{mlp_root}.gate.weight",
                (num_experts, config.hidden_size),
            ),
            f"{mlp_root}.router.bias": (
                f"{mlp_root}.gate.bias",
                (num_experts,),
            ),
        }
        for source_name, (target_name, shape) in router_sources.items():
            _require_header(
                key_index,
                source_name,
                shape,
                expected_dtypes=_FLOAT_DTYPES,
                description="router tensor",
            )
            targets[target_name] = StreamingWeightSource(source_name)
            claimed_targets.add(target_name)

    # Everything outside the native experts is the existing strict
    # pass-through path. Iterating graph names makes omissions fail locally,
    # while the generic loader rejects any unclassified checkpoint sidecars.
    for target_name, initializer in sorted(initializers.items()):
        if initializer.const_value is not None or target_name in claimed_targets:
            continue
        if target_name in key_index:
            targets[target_name] = StreamingWeightSource(target_name)

    return StreamingWeightPlan(
        targets=targets,
        target_constants=target_constants,
        report={
            "output_weight_format": "mxfp4",
            "native_mxfp4": True,
            "streaming_external_data": True,
            "streaming_unit": "one_moe_projection",
            "num_moe_layers": config.num_hidden_layers,
        },
    )


def stream_gptoss_mxfp4_safetensors_to_package(
    package: ModelPackage,
    model_id: str,
    config: ArchitectureConfig,
    *,
    revision: str | None = None,
) -> dict[str, object]:
    """Bind native MXFP4 weights without ever constructing a checkpoint dict."""
    if len(package) != 1:
        raise ValueError(
            "Native GPT-OSS MXFP4 streaming requires a single text model component."
        )

    try:
        local = _local_weight_paths(pathlib.Path(model_id))
    except FileNotFoundError as exc:
        raise ValueError(
            "Native GPT-OSS MXFP4 export requires a streamable safetensors "
            "checkpoint containing model.safetensors or "
            "model.safetensors.index.json; eager loading is intentionally disabled."
        ) from exc
    if local is not None and local[1] != "safetensors":
        raise ValueError(
            "Native GPT-OSS MXFP4 export requires a safetensors checkpoint "
            "(model.safetensors or model.safetensors.index.json). Legacy "
            "PyTorch weights cannot be streamed; convert the checkpoint to "
            "safetensors instead of attempting eager 120B loading."
        )
    model = next(iter(package.values()))

    def planner(key_index, initializers):
        return build_gptoss_mxfp4_streaming_plan(config, key_index, initializers)

    try:
        paths = local[0] if local is not None else _resolve_shard_paths(model_id, revision)
        report = stream_preprocessed_safetensors_to_model(
            model,
            model_id,
            planner,
            revision=revision,
            _resolved_paths=paths,
        )
    except EntryNotFoundError as exc:
        raise ValueError(
            "Native GPT-OSS MXFP4 export requires streamable safetensors, but "
            "the repository publishes no model.safetensors index/file. Convert "
            "the checkpoint to safetensors; eager loading is intentionally disabled."
        ) from exc
    package._native_streaming_source_directories = _lazy_safetensors_source_parent_aliases(
        paths
    )
    package._native_streaming_source_files = frozenset(
        pathlib.Path(path).resolve() for path in paths
    )
    if local is not None:
        # Local filesystem paths are needed by LazyTensor closures and the
        # transient overlap guard only; reports and ONNX metadata stay portable.
        report["source"] = "local-safetensors-checkpoint"
        model.metadata_props["mobius.weight_loading"] = json.dumps(report, sort_keys=True)
    package.weight_loading_report = report
    return report


__all__ = [
    "build_gptoss_mxfp4_streaming_plan",
    "stream_gptoss_mxfp4_safetensors_to_package",
]
