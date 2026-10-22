#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export an opt-in FP8 Ornith DFlash2 drafter using GenAI's native block-drafter ABI.

Add onnxruntime-genai/src/python/py to PYTHONPATH to reuse its DFlash2 builder.
The accepted decoder, weights and default config are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import onnx_ir as ir
import torch
from huggingface_hub import snapshot_download
from models.builders.dflash2 import DFlash2Builder
from safetensors import safe_open

from mobius.components._fp8 import quantize_fp8

MODEL_ID = "incoai/Ornith-1.5-122B-DFlash2"
REVISION = "9baefc1bec85c08193ab442e0ca5f7057669b61d"
TARGET_DECODER = "decoder/model-fused-shared-swiglu-fp16-state-gated-rms.onnx"


def add_target_taps(model, layer_ids, hidden_size):
    """Tap post-residual layer outputs via the following layer's input norm."""
    if not layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise ValueError("Target tap IDs must be nonempty and unique")
    if any(not isinstance(i, int) or not 0 <= i < 47 for i in layer_ids):
        raise ValueError("Ornith target tap IDs must be in [0, 47)")
    if any(value.name == "aux_hidden_states" for value in model.graph.outputs):
        raise ValueError("Target already has an aux_hidden_states output")
    taps = []
    for index in layer_ids:
        prefix = f"decoder/model/layers.{index + 1}/input_layernorm/Cast"
        matches = [
            node.inputs[0]
            for node in model.graph
            if node.name
            and node.name.startswith(prefix)
            and not node.domain
            and node.op_type == "Cast"
            and not node.inputs[0].is_initializer()
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one unfused input norm for layer {index + 1}")
        tap = matches[0]
        if tap.dtype != ir.DataType.FLOAT16 or tap.shape != ir.Shape(
            ["num_tokens", hidden_size]
        ):
            raise ValueError(f"Target tap {index} has an incompatible shape or dtype")
        taps.append(tap)
    output = ir.Value(
        name="aux_hidden_states",
        type=ir.TensorType(ir.DataType.FLOAT16),
        shape=ir.Shape(["num_tokens", hidden_size * len(taps)]),
    )
    model.graph.append(
        ir.Node("", "Concat", taps, {"axis": ir.AttrInt64("axis", -1)}, outputs=[output])
    )
    model.graph.outputs.append(output)
    return taps


class OrnithDFlash2Builder(DFlash2Builder):
    """Reuse GenAI's architecture with Mobius block-FP8 projection quantization."""

    def __init__(
        self, draft_dir, target_model, *, fp8=True, num_draft_tokens=7, fp32_conv=False
    ):
        self.target_model = target_model
        self.fp8 = fp8
        super().__init__(
            str(draft_dir),
            "",
            ir.DataType.FLOAT16,
            paged_block_size=256,
            max_position_embeddings=262144,
            filename="dflash2-fp8.onnx" if fp8 else "dflash2-bf16.onnx",
            num_draft_tokens=num_draft_tokens,
            compute_dtype=ir.DataType.BFLOAT16,
            fuse_gate_up=True,
            fuse_qkv=True,
            conv_fp32_accumulation=fp32_conv,
        )
        if self.block_size != 8:
            self.filename = self.filename.replace(".onnx", f"-block{self.block_size}.onnx")
        if fp32_conv:
            self.filename = self.filename.replace(".onnx", "-fp32conv.onnx")

    def load_weights(self):
        weights = {}
        for filename in ("model.safetensors", "mask_embedding.safetensors"):
            with safe_open(Path(self.draft_dir) / filename, framework="pt") as handle:
                weights.update(
                    {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
                )
        self.validate_weight_shapes(
            weights,
            {
                "fc.weight": (self.hidden_size, self.aux_hidden_size),
                "mask_embedding": (self.hidden_size,),
            },
        )
        self.weights = weights
        return weights

    def matmul(
        self,
        name,
        root_input,
        weight_tensor,
        in_features,
        out_features,
        rows,
        weight_name=None,
        quantize=True,
    ):
        if not self.fp8 or not quantize:
            return super().matmul(
                name,
                root_input,
                weight_tensor,
                in_features,
                out_features,
                rows,
                weight_name=weight_name,
                quantize=quantize,
            )
        if tuple(weight_tensor.shape) != (out_features, in_features):
            raise ValueError(f"Invalid projection weight shape for {name}")
        initializer = weight_name or (name[1:].replace("/", ".") + ".weight")
        scales_name = initializer + ".scales"
        if initializer not in self.values:
            weight, scales = quantize_fp8(weight_tensor, block_size=128)
            self.make_initializer(weight, initializer)
            self.make_initializer(scales, scales_name, to=ir.DataType.FLOAT)
        output = self.out(name)
        self.make_node(
            "MatMulBlockQuantizedFp8Weight",
            [root_input, initializer, scales_name],
            [output],
            name=name,
            domain="com.microsoft",
            block_size=128,
        )
        self.make_value(output, self.io_dtype, [rows, out_features])
        return output

    def _adopt(self, value):
        if value.name not in self.values:
            adopted = self.make_value(value.name, value.dtype, value.shape)
            adopted.const_value = value.const_value
            self.graph.register_initializer(adopted)
        return value.name

    def make_embedding(self, name, rows):
        embeddings = [
            node
            for node in self.target_model.graph
            if node.op_type == "Gather"
            and not node.domain
            and node.inputs[0].is_initializer()
            and node.inputs[0].shape == ir.Shape([self.vocab_size, self.hidden_size])
        ]
        if len(embeddings) != 1 or embeddings[0].inputs[0].dtype != self.external_dtype:
            raise ValueError("Expected one FP16 target embedding table")
        output = self.out(name)
        self.make_node(
            "Gather", [self._adopt(embeddings[0].inputs[0]), "input_ids"], [output], name=name
        )
        self.make_value(output, self.external_dtype, [rows, self.hidden_size])
        return output

    def make_lm_head(self, root, rows="num_block"):
        heads = [
            node
            for node in self.target_model.graph
            if node.op_type == "MatMulBlockQuantizedFp8Weight"
            and node.domain == "com.microsoft"
            and node.outputs[0].name == "logits"
        ]
        if len(heads) != 1 or any(
            value is not None and not value.is_initializer() for value in heads[0].inputs[1:]
        ):
            raise ValueError("Expected one FP8 target LM head with constant parameters")
        head = heads[0]
        root = self.unary(
            "Cast",
            "/lm_head/Cast",
            root,
            self.external_dtype,
            [rows, self.hidden_size],
            to=self.external_dtype,
        )
        output = self.out("/lm_head/MatMul")
        node = self.make_node(
            head.op_type,
            [root, *[self._adopt(x) if x is not None else "" for x in head.inputs[1:]]],
            [output],
            name="/lm_head/MatMul",
            domain=head.domain,
        )
        node.attributes.update(head.attributes)
        self.make_value(output, self.external_dtype, [rows, self.vocab_size])
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--num-draft-tokens", type=int, default=7, choices=range(1, 16))
    parser.add_argument(
        "--fp32-conv",
        action="store_true",
        help="Match compiled reference convolution accumulation",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Export a nonquantized draft control"
    )
    args = parser.parse_args()
    torch.set_num_threads(12)
    draft_dir = Path(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            allow_patterns=["config.json", "model.safetensors", "mask_embedding.safetensors"],
        )
    )
    target_path = args.model / TARGET_DECODER
    target = ir.load(target_path)
    builder = OrnithDFlash2Builder(
        draft_dir,
        target,
        fp8=not args.bf16,
        num_draft_tokens=args.num_draft_tokens,
        fp32_conv=args.fp32_conv,
    )
    config = json.loads((args.model / "genai_config.json").read_text())
    capacity = config["model"]["decoder"]["state_update_capacity"]
    if capacity < 2:
        raise ValueError("Target recurrent rollback capacity must allow at least one draft")
    verification_width = min(builder.num_draft_tokens, capacity - 1)
    output_dir = args.model / "dflash2"
    output_dir.mkdir(exist_ok=True)
    target_variant = target_path.with_name("model-gated-rms-dflash2.onnx")
    add_target_taps(target, builder.target_layer_ids, builder.hidden_size)
    target_proto = ir.serde.serialize_model(target)
    if (
        target_variant.exists()
        and target_variant.read_bytes() != target_proto.SerializeToString()
    ):
        raise FileExistsError("Existing tapped decoder differs from the requested target")
    if (output_dir / builder.filename).exists():
        raise FileExistsError(
            "DFlash2 output already exists; choose a clean experiment directory"
        )
    builder.make_model()
    builder.save_model(str(output_dir))
    # Serialize without resolving external tensors: keep the accepted INT2 weight bytes.
    if not target_variant.exists():
        onnx.save_model(target_proto, target_variant)
    section = builder.genai_config_section()
    section["filename"] = f"dflash2/{builder.filename}"
    section["aux_hidden_state_layers"] = [i + 1 for i in builder.target_layer_ids]
    overlay = {
        "model": {
            "decoder": {
                "filename": str(target_variant.relative_to(args.model)),
                "outputs": {"aux_hidden_states": "aux_hidden_states"},
            },
            "mtp": {"enabled": False},
            "dflash2": section,
        },
        "speculative": {"max_draft_tokens": verification_width},
    }
    name = "bf16" if args.bf16 else "fp8"
    if builder.block_size != 8:
        name += f"-block{builder.block_size}"
    if args.fp32_conv:
        name += "-fp32conv"
    (args.model / f"dflash2-{name}-config.json").write_text(
        json.dumps(overlay, indent=2) + "\n"
    )
    provenance = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "target_decoder": TARGET_DECODER,
        "target_decoder_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "draft_block_size": builder.block_size,
        "verification_draft_tokens": verification_width,
        "conv_fp32_accumulation": args.fp32_conv,
        "target_capture_capacity": capacity,
        "precision": {
            "projections": "BF16" if args.bf16 else "FP8 E4M3FN, row-block 128, FP32 scales",
            "activations_and_draft_kv": "BF16",
            "dynamic_conv_and_selector": "BF16 weights; unquantized",
            "mask_embedding": "supplied BF16 row, restored after embedding cast",
            "lm_head": "exact accepted target FP8 weights/scales",
        },
        "status": "exported_not_validated",
        "accuracy_test": False,
    }
    (output_dir / f"export-{name}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported {output_dir / builder.filename}; validation and benchmarking required.")


if __name__ == "__main__":
    main()
