# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fail-closed bounded-memory loading for Drama 3 BF16/block-FP8 checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import onnx_ir as ir
import torch

from mobius._configs.fish_omni import FishOmniConfig
from mobius._model_package import ModelPackage
from mobius.integrations._weight_loading import (
    StreamingExpertBankSource,
    StreamingWeightPlan,
    StreamingWeightSource,
    stream_preprocessed_safetensors_to_model,
)


def _classify_source(source, index, config, *, expert, saturate_float16=False):
    if source not in index:
        raise ValueError(f"Missing Drama 3 checkpoint tensor {source!r}")
    source_dtype = index[source][2]
    if config.fp8_experts and expert:
        if source_dtype != "F8_E4M3":
            raise ValueError(f"Drama 3 FP8 expert {source!r} must have dtype F8_E4M3")
        return StreamingWeightSource(
            source,
            mode="fp8_block_128",
            scale_name=source.removesuffix(".weight") + ".weight_scale_inv",
            reconstruction_dtype=torch.float32,
            saturate_float16=saturate_float16,
        )
    if source_dtype not in {"F32", "BF16", "F16"}:
        raise ValueError(f"Unexpected Drama 3 weight dtype {source_dtype}: {source}")
    return StreamingWeightSource(source, saturate_float16=saturate_float16)


def _source_name(target: str) -> str:
    if target == "embedding.embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if target == "embedding.codebook_embeddings.weight":
        return "audio_decoder.codebook_embeddings.weight"
    if target.startswith("decoder.model."):
        source = "model.language_model." + target.removeprefix("decoder.model.")
        source = source.replace(".mlp.packed_experts.", ".mlp.experts.")
        return source.replace(".linear_attn.conv1d_weight", ".linear_attn.conv1d.weight")
    if target == "decoder.lm_head.weight":
        return "lm_head.weight"
    for local, source in (
        ("audio_decoder.shared_layers.", "audio_decoder_mtp_shared.layers."),
        ("audio_decoder.residual_layers.", "audio_decoder_mtp.layers."),
        ("audio_decoder.codebook_heads.", "audio_decoder_mtp.heads."),
    ):
        if target.startswith(local):
            return source + target.removeprefix(local)
    raise ValueError(f"Unrecognized Drama 3 graph parameter {target!r}")


def build_fish_omni_streaming_plan(
    index: Mapping[str, tuple[str, list[int], str]],
    initializers: Mapping[str, ir.Value],
    config: FishOmniConfig,
) -> StreamingWeightPlan:
    """Map every learned parameter and classify each independently scaled expert."""
    targets: dict[str, StreamingWeightSource | StreamingExpertBankSource] = {}
    for name, value in initializers.items():
        if value.const_value is not None:
            continue
        source = _source_name(name)
        saturate = value.dtype == ir.DataType.FLOAT16 and (
            source.startswith("model.language_model.") or source == "lm_head.weight"
        )
        if any(
            source.endswith(f".experts.{projection}.weight")
            for projection in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
        ):
            assert config.num_local_experts is not None
            prefix, projection = source.rsplit(".experts.", 1)
            projections = (
                ("gate_proj.weight", "up_proj.weight")
                if projection == "gate_up_proj.weight"
                else (projection,)
            )
            targets[name] = StreamingExpertBankSource(
                tuple(
                    tuple(
                        _classify_source(
                            f"{prefix}.experts.{expert}.{part}",
                            index,
                            config,
                            expert=True,
                            saturate_float16=saturate,
                        )
                        for part in projections
                    )
                    for expert in range(config.num_local_experts)
                )
            )
        else:
            targets[name] = _classify_source(
                source,
                index,
                config,
                expert=".mlp.experts." in source,
                saturate_float16=saturate,
            )
    return StreamingWeightPlan(
        targets=targets,
        report={"architecture": "fish_omni", "codec_included": False},
    )


def stream_fish_omni_safetensors_to_package(
    package: ModelPackage,
    model_id: str,
    config: FishOmniConfig,
    *,
    revision: str | None,
) -> None:
    """Validate the entire package before binding any lazy weight sources.

    The aggregate graph is only a binding inventory and is never executed or
    serialized. One generic planner checks all three components transactionally, including unexpected
    source keys, rather than ignoring another component's unchecked tensors.
    """
    parameters: dict[str, ir.Value] = {}
    for model in package.values():
        for name, value in model.graph.initializers.items():
            if value.const_value is not None:
                continue
            if name in parameters:
                raise ValueError(f"Duplicate Drama 3 parameter {name!r}")
            parameters[name] = value
    inventory = ir.Model(
        ir.Graph(
            [],
            [],
            nodes=[],
            initializers=[
                ir.Value(name=name, shape=value.shape, type=value.type)
                for name, value in parameters.items()
            ],
            opset_imports={"": 24},
        ),
        ir_version=12,
    )
    report = stream_preprocessed_safetensors_to_model(
        inventory,
        model_id,
        lambda index, initializers: build_fish_omni_streaming_plan(
            index, initializers, config
        ),
        revision=revision,
    )
    for name, value in parameters.items():
        value.const_value = inventory.graph.initializers[name].const_value
    package.weight_loading_report = report
    for model in package.values():
        model.metadata_props.update(inventory.metadata_props)
