#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export Ornith-1.5-122B: INT2 decoder, FP8 LM head/MTP/KV, FP16 vision.

Requires this Mobius checkout and Olive on PYTHONPATH. This exports the existing
batched Mobius graph ABI, NOT GenAI Engine's packed/paged decoder ABI. The
manifest explicitly records this runtime integration gap; do not benchmark this
package as Engine-ready until packed/paged export and image admission are wired.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from contextlib import ExitStack
from pathlib import Path

import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import TorchTensor
from onnxscript import nn
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, QuantizationConfig, Qwen35MtpConfig
from mobius._model_package import ModelPackage
from mobius.components import Linear, QuantizedLinear
from mobius.components._fp8 import Fp8Linear, Fp8QwenMoEBlock, quantize_fp8
from mobius.components._moe import MoELayer
from mobius.integrations.ort_genai import write_ort_genai_config
from mobius.models.qwen35 import Qwen35MoEVL3ModelCausalLMModel
from mobius.models.qwen35_mtp import Qwen35MtpModel

MODEL_ID = "ornith-ai/Ornith-1.5-122B"
REVISION = "7681ef608624e7532d8164128bcffdbdbaf1b1af"
logger = logging.getLogger(__name__)


def validate_checkpoint_revision(checkpoint: Path) -> None:
    """Require pinned Hub download metadata rather than mislabel local weights."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    sources = set(index["weight_map"].values()) | {
        "config.json",
        "model.safetensors.index.json",
    }
    for source in sorted(sources):
        metadata = checkpoint / ".cache/huggingface/download" / f"{source}.metadata"
        if not metadata.is_file():
            raise ValueError(f"Missing pinned Hub download metadata for {source}")
        lines = metadata.read_text().splitlines()
        if not lines or lines[0] != REVISION:
            raise ValueError(f"Checkpoint revision mismatch for {source}: expected {REVISION}")


def _source_name(name: str) -> str:
    if name.startswith("decoder.model."):
        return "model.language_model." + name.removeprefix("decoder.model.")
    if name.startswith("decoder."):
        return name.removeprefix("decoder.")
    if name.startswith("embedding."):
        return "model.language_model." + name.removeprefix("embedding.")
    if name.startswith("vision_encoder."):
        source = (
            ("model." + name.removeprefix("vision_encoder."))
            .replace(".mlp.up_proj.", ".mlp.linear_fc1.")
            .replace(".mlp.down_proj.", ".mlp.linear_fc2.")
            .replace(".patch_embed.", ".patch_embed.proj.")
        )
        return source + ".weight" if source.endswith(".pos_embed") else source
    if name.startswith("mtp.embed_tokens."):
        return "model.language_model." + name.removeprefix("mtp.")
    if name.startswith("mtp.lm_head."):
        return name.removeprefix("mtp.")
    return name


def _replace_linears(module: nn.Module, *, fp8: bool = False, prefix: str = "") -> None:
    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, (Linear, QuantizedLinear)) and not path.endswith(
            ("shared_expert_gate", ".gate")
        ):
            if isinstance(child, QuantizedLinear):
                n, k = child._n, child._k
            else:
                n, k = map(int, child.weight.shape)
            replacement = (
                Fp8Linear(k, n, bias=child.bias is not None)
                if fp8 or path.endswith("lm_head")
                else QuantizedLinear(k, n, bits=2, block_size=128, bias=child.bias is not None)
            )
            setattr(module, name, replacement)
        else:
            _replace_linears(child, fp8=fp8, prefix=path)


def create_modules(hf_config):
    """Build the mixed-precision modules without allocating checkpoint weights."""
    if hf_config.model_type != "qwen3_5_moe" or not hasattr(hf_config, "vision_config"):
        raise ValueError("Expected a Qwen3.5-MoE vision-language checkpoint")
    text = hf_config.text_config
    if text.mtp_num_hidden_layers != 1:
        raise ValueError("Ornith export currently requires exactly one MTP layer")
    base = ArchitectureConfig.from_transformers(text, parent_config=hf_config)
    config = dataclasses.replace(
        base,
        dtype=ir.DataType.FLOAT16,
        quantization=QuantizationConfig(bits=2, group_size=128, quant_method="olive"),
        num_nextn_predict_layers=text.mtp_num_hidden_layers,
    )
    target = Qwen35MoEVL3ModelCausalLMModel(config)
    _replace_linears(target.decoder)
    mtp_config = dataclasses.replace(
        Qwen35MtpConfig.from_transformers(hf_config),
        dtype=ir.DataType.FLOAT16,
        use_dedicated_embeddings=True,
        use_dedicated_lm_head=True,
        num_nextn_predict_layers=text.mtp_num_hidden_layers,
        quantization=None,
    )
    mtp = Qwen35MtpModel(mtp_config)
    mtp.layers[0].mlp = Fp8QwenMoEBlock(mtp_config)
    _replace_linears(mtp, fp8=True)
    return target, config, mtp, mtp_config


def _bind(parameter, tensor: torch.Tensor) -> None:
    expected = tuple(parameter.shape)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"Weight shape mismatch: expected {expected}, got {tuple(tensor.shape)}"
        )
    parameter.const_value = TorchTensor(tensor.cpu().contiguous())


def expose_mtp_seed(package: ModelPackage, config: ArchitectureConfig) -> None:
    """Expose the exact post-final-norm input to the target LM head."""
    graph = package["decoder"].graph
    if any(value.name == "mtp_seed" for value in graph.outputs):
        raise ValueError("Target already exposes mtp_seed")
    heads = [
        node
        for node in graph
        if node.op_type == "MatMulBlockQuantizedFp8Weight"
        and node.inputs[1] is not None
        and node.inputs[1].shape == ir.Shape([config.vocab_size, config.hidden_size])
    ]
    if len(heads) != 1:
        raise ValueError(f"Expected one target FP8 LM head, found {len(heads)}")
    hidden = heads[0].inputs[0]
    seed = ir.Value(
        name="mtp_seed",
        type=ir.TensorType(config.dtype),
        shape=ir.Shape(["batch_size", "sequence_len", config.hidden_size]),
    )
    graph.append(ir.Node("", "Identity", [hidden], outputs=[seed]))
    graph.outputs.append(seed)


def load_weights(module: nn.Module, checkpoint: Path, *, prefix: str, device: str) -> None:
    """Read one source tensor at a time; retain only quantized target weights."""
    from olive.common.quant.utils import WeightQuantizer, pack_to_uint8

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    quantizer = WeightQuantizer(bits=2, symmetric=True, group_size=128)

    with ExitStack() as stack:
        shards = {
            name: stack.enter_context(
                safe_open(checkpoint / name, framework="pt", device="cpu")
            )
            for name in sorted(set(weight_map.values()))
        }

        def read(name: str) -> torch.Tensor:
            if name not in weight_map:
                raise KeyError(f"Checkpoint is missing required tensor {name!r}")
            return shards[weight_map[name]].get_tensor(name)

        def int2(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            weight = weight.to(device=device, dtype=torch.float32)
            scales, zeros = quantizer.find_qparams(weight)
            # Store scales in the same dtype consumed by MatMulNBits/QMoE.
            minimum = torch.nextafter(
                torch.zeros((), dtype=torch.float16, device=device),
                torch.ones((), dtype=torch.float16, device=device),
            ).float()
            scales = scales.clamp_min(minimum).to(torch.float16)
            if not torch.isfinite(scales).all() or torch.any(scales <= 0):
                raise ValueError("INT2 scales cannot be represented as positive finite FP16")
            codes = quantizer.quantize(weight, scales.float(), zeros)
            return pack_to_uint8(codes, 2).cpu(), scales.cpu()

        for path, child in module.named_modules():
            name = ".".join(part for part in (prefix, path) if part)
            if isinstance(child, (Fp8QwenMoEBlock, MoELayer)) and (
                isinstance(child, Fp8QwenMoEBlock) or child.experts is None
            ):
                root = _source_name(name)
                count = (
                    child.gate.num_experts
                    if isinstance(child, Fp8QwenMoEBlock)
                    else child.num_experts
                )
                banks: dict[str, list[torch.Tensor]] = {
                    key: []
                    for key in (
                        "fc1_experts_weights",
                        "fc1_scales",
                        "fc2_experts_weights",
                        "fc2_scales",
                    )
                }
                logger.info("Quantizing routed experts: %s", root)
                for expert in range(count):
                    stem = f"{root}.experts.{expert}"
                    fused_key = f"{root}.experts.gate_up_proj"
                    down_key = f"{root}.experts.down_proj"
                    if fused_key in weight_map:
                        gate_up = shards[weight_map[fused_key]].get_slice(fused_key)[expert]
                        gate, up = gate_up.chunk(2, dim=0)
                        down = shards[weight_map[down_key]].get_slice(down_key)[expert]
                    else:
                        gate = read(f"{stem}.gate_proj.weight")
                        up = read(f"{stem}.up_proj.weight")
                        down = read(f"{stem}.down_proj.weight")
                    for label, tensor in (
                        ("fc1", torch.cat([gate, up], dim=0)),
                        ("fc2", down),
                    ):
                        if isinstance(child, Fp8QwenMoEBlock):
                            packed, scales = quantize_fp8(
                                tensor.to(device), square_blocks=True
                            )
                        else:
                            packed, scales = int2(tensor)
                        banks[f"{label}_experts_weights"].append(packed.cpu())
                        banks[f"{label}_scales"].append(scales.cpu())
                for key, tensors in banks.items():
                    _bind(getattr(child, key), torch.stack(tensors))
                continue
            if isinstance(child, (QuantizedLinear, Fp8Linear)):
                weight = read(_source_name(f"{name}.weight"))
                if isinstance(child, Fp8Linear):
                    packed, scales = quantize_fp8(weight.to(device))
                else:
                    packed, scales = int2(weight)
                    packed = packed.reshape(tuple(child.weight.shape))
                _bind(child.weight, packed)
                _bind(child.scales, scales)
                if child.bias is not None:
                    _bind(child.bias, read(_source_name(f"{name}.bias")).half())
                continue
            for key, parameter in child.named_parameters(recurse=False):
                if parameter.const_value is not None:
                    continue
                if key.startswith(("fc1_", "fc2_")):
                    raise ValueError(f"Unrecognized expert-bank parameter: {name}.{key}")
                tensor = read(_source_name(f"{name}.{key}"))
                if name.endswith("patch_embed") and key == "weight":
                    tensor = tensor.reshape(tuple(parameter.shape))
                if (
                    parameter.dtype == ir.DataType.FLOAT
                    or parameter.dtype == ir.DataType.FLOAT16
                ):
                    tensor = (
                        tensor.float()
                        if getattr(parameter, "_keep_float32", False)
                        else tensor.half()
                    )
                _bind(parameter, tensor)


def write_export_metadata(package: ModelPackage, checkpoint: Path, output: Path) -> None:
    AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    write_ort_genai_config(
        package,
        str(output),
        local_config_dir=str(checkpoint),
        ep="cuda",
        context_length=32768 + 512,
    )
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "decoder_weights": "int2, symmetric RTN, block_size=128",
        "lm_head": "fp8_e4m3fn W8A16, block_size=128",
        "mtp": "fp8_e4m3fn W8A16, square expert blocks=128",
        "kv_cache": "fp8_e4m3fn, unit scales (not calibrated)",
        "vision": "float16",
        "engine_ready": False,
        "component_validation": "not_run",
        "blockers": [
            "Mobius decoder/MTP use batched tensors, not Engine packed/paged I/O.",
            "GenAI Engine currently accepts decoder-only models, not the vision pipeline.",
        ],
    }
    (output / "ornith_export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.warning("Export saved; Engine integration is NOT complete. See ornith_export.json.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite export directory: {args.output}")
    validate_checkpoint_revision(args.checkpoint)
    AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    target, target_config, mtp, mtp_config = create_modules(config)
    with torch.inference_mode():
        load_weights(target, args.checkpoint, prefix="", device=args.device)
        package = build_from_module(
            target,
            target_config,
            target.default_task,
            execution_provider="cuda",
            fp8_kv_cache=True,
        )
        expose_mtp_seed(package, target_config)
        load_weights(mtp, args.checkpoint, prefix="mtp", device=args.device)
        package.mtp_head = build_from_module(
            mtp, mtp_config, mtp.default_task, execution_provider="cuda", fp8_kv_cache=True
        )
        package.save(str(args.output), external_data="onnx", max_workers=1)
    write_export_metadata(package, args.checkpoint, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
