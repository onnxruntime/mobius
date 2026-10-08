# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fail-closed streaming plans for Nemotron-H BF16 and reconstructed ModelOpt weights."""

from __future__ import annotations

from collections.abc import Mapping

import onnx_ir as ir
import torch

from mobius._configs import NemotronHConfig
from mobius.integrations._weight_loading import (
    StreamingWeightPlan,
    StreamingWeightSource,
    _load_indexed_tensor,
)
from mobius.integrations.modelopt._config import ModelOptConfig
from mobius.integrations.modelopt._weights import validate_modelopt_scale
from mobius.models.nemotron_h import _rename_nemotron_h_weight


def build_nemotron_h_streaming_plan(
    config: NemotronHConfig,
    key_index: Mapping[str, tuple[str, list[int], str]],
    initializers: Mapping[str, ir.Value],
    *,
    modelopt: ModelOptConfig | None = None,
    target_decoder_only: bool = False,
) -> StreamingWeightPlan:
    """Map exact per-expert checkpoint keys and account for every source tensor."""
    mtp_names = sorted(name for name in key_index if name.startswith("mtp."))
    if (config.num_nextn_predict_layers or mtp_names) and not target_decoder_only:
        raise NotImplementedError(
            "Nemotron-H NextN/MTP tensors require a validated MTP export contract; "
            "they must not be silently discarded. Use explicit target_decoder_only=True "
            "for the separately labeled target decoder variant."
        )
    if target_decoder_only and config.num_nextn_predict_layers and not mtp_names:
        raise ValueError("Source config declares NextN/MTP but no mtp.* tensors were found")
    omitted = dict.fromkeys(
        mtp_names, "explicit target-decoder-only variant excludes NextN/MTP"
    )
    targets: dict[str, StreamingWeightSource] = {}
    constants: dict[str, torch.Tensor] = {}
    ignored = omitted
    quantized_modules: set[str] = set()
    activation_modules = modelopt.activation_modules if modelopt is not None else frozenset()
    cache_scales = set()
    if modelopt is not None and modelopt.kv_cache_fp8:
        for layer_index, kind in enumerate(config.layer_types or []):
            if kind == "full_attention":
                for projection, suffix in (("k_proj", "k_scale"), ("v_proj", "v_scale")):
                    cache_scales.add(
                        f"backbone.layers.{layer_index}.mixer.{projection}.{suffix}"
                    )
        for name in sorted(cache_scales):
            metadata = key_index.get(name)
            if metadata is None or metadata[1] != [1] or metadata[2] != "F32":
                raise ValueError(f"Missing/malformed ModelOpt KV-cache scale {name!r}")
            scale = _load_indexed_tensor(name, key_index)
            validate_modelopt_scale(scale, name, scalar=True)
            constants[name] = scale
    for source_name in key_index:
        if source_name in omitted:
            continue
        if source_name in cache_scales:
            continue
        if source_name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            continue
        target_name = _rename_nemotron_h_weight(source_name, config.layer_types or [])
        if target_name in targets:
            raise ValueError(f"Multiple Nemotron-H sources map to {target_name!r}")
        if target_name not in initializers:
            raise ValueError(f"Unmapped Nemotron-H checkpoint tensor {source_name!r}")
        module_name = source_name.removesuffix(".weight")
        mode = modelopt.modules.get(module_name) if modelopt is not None else None
        if mode is not None:
            if not source_name.endswith(".weight"):
                raise ValueError(f"ModelOpt target {module_name!r} is not a projection weight")
            quantized_modules.add(module_name)
            targets[target_name] = StreamingWeightSource(
                source_name,
                mode=mode,
                scale_name=module_name + ".weight_scale",
                global_scale_name=(
                    module_name + ".weight_scale_2" if mode == "modelopt_nvfp4" else None
                ),
            )
            if module_name in activation_modules:
                input_name = module_name + ".input_scale"
                metadata = key_index.get(input_name)
                if metadata is None or metadata[1] not in ([], [1]) or metadata[2] != "F32":
                    raise ValueError(f"Missing/malformed ModelOpt input scale {input_name!r}")
                scale = _load_indexed_tensor(input_name, key_index)
                validate_modelopt_scale(scale, input_name, scalar=True)
                # Explicitly validated, but not used: this route reconstructs
                # weights for dense BF16 execution, not FP8 activations.
                constants[input_name] = scale
        else:
            if key_index[source_name][2] not in {"BF16", "F16", "F32"}:
                raise ValueError(f"Unclassified quantized Nemotron-H weight {source_name!r}")
            targets[target_name] = StreamingWeightSource(source_name)
    if modelopt is not None and quantized_modules != set(modelopt.modules):
        raise ValueError(
            "ModelOpt projection inventory mismatch; missing "
            f"{sorted(set(modelopt.modules) - quantized_modules)[:5]}"
        )
    return StreamingWeightPlan(
        targets,
        ignored=ignored,
        constants=constants,
        report={
            "export_variant": "target-decoder-only" if target_decoder_only else "standard",
            "source_num_nextn_predict_layers": config.num_nextn_predict_layers,
            "mtp_preserved": False,
            "mtp_inventory_status": "inspected",
            "omitted_mtp_tensor_count": len(omitted),
            "omitted_mtp_tensors": {
                name: {
                    "reason": reason,
                    "shape": key_index[name][1],
                    "dtype": key_index[name][2],
                }
                for name, reason in omitted.items()
            },
            "source_weight_format": modelopt.source_format if modelopt else "floating",
            "storage_policy": "explicit-dense-bf16-reconstruction" if modelopt else "dense",
            "native_nvfp4": False,
            "activation_quantization_preserved": False,
            "kv_cache_quantization_preserved": False,
            "validated_kv_cache_scale_count": len(cache_scales),
            "reconstructed_projection_count": len(quantized_modules),
        },
    )
