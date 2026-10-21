#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export Ornith's packed/paged text decoder and FP8 MTP for GenAI Engine.

Vision components remain in the component export; Engine image admission is
not implemented by this text-only integration.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from contextlib import ExitStack
from pathlib import Path

import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import TorchTensor, to_torch_dtype
from onnxscript import nn
from ornith_export import (
    MODEL_ID,
    REVISION,
    _source_name,
    create_modules,
    validate_checkpoint_revision,
)
from safetensors import safe_open
from transformers import AutoConfig

from mobius._builder import build_from_module
from mobius._model_package import ModelPackage
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.tasks import ModelTask
from mobius.tasks._base import _make_graph, _make_model

logger = logging.getLogger(__name__)


def bind_component_weights(model, component, checkpoint, *, mtp=False):
    """Reuse packed initializers; recover unfused small parameters from source."""
    previous = ir.load(component).graph.initializers
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    reused = recovered = 0
    with ExitStack() as stack:
        shards = {}
        for name, value in model.graph.initializers.items():
            if value.const_value is not None:
                continue
            if name in previous:
                source = previous[name]
                if source.shape != value.shape or source.dtype != value.dtype:
                    raise ValueError(f"Component initializer ABI mismatch for {name}")
                if source.const_value is None:
                    raise ValueError(f"Component initializer has no data: {name}")
                value.const_value = source.const_value
                reused += 1
                continue
            source_name = _source_name(f"mtp.{name}" if mtp else name)
            if source_name not in index:
                raise ValueError(f"No component or checkpoint initializer for {name}")
            shard_name = index[source_name]
            if shard_name not in shards:
                shards[shard_name] = stack.enter_context(
                    safe_open(checkpoint / shard_name, framework="pt", device="cpu")
                )
            tensor = shards[shard_name].get_tensor(source_name)
            if tuple(tensor.shape) != tuple(value.shape):
                raise ValueError(f"Checkpoint initializer shape mismatch for {name}")
            value.const_value = TorchTensor(
                tensor.to(to_torch_dtype(value.dtype)).contiguous()
            )
            recovered += 1
    logger.info("Reused %d component tensors; recovered %d source tensors", reused, recovered)


class PackedConv(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.channels = original._channels

    def forward(self, op, x, state, cumulative, capture, capacity):
        bias = op.Expand(op.CastLike(0.0, x), [self.channels])
        return op.VarlenCausalConvWithState(
            x,
            self.weight,
            cumulative,
            bias,
            state,
            capture,
            activation="silu",
            state_update_capacity=capacity,
            _domain="com.microsoft",
            _outputs=3,
        )


class PackedDeltaNet(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name, child in original.named_children():
            setattr(self, name, PackedConv(child) if name == "conv1d" else child)
        self.A_log = original.A_log
        self.dt_bias = original.dt_bias
        for name in (
            "num_v_heads",
            "num_k_heads",
            "head_k_dim",
            "head_v_dim",
            "key_dim",
            "value_dim",
        ):
            setattr(self, name, getattr(original, name))

    def forward(self, op, hidden, state, metadata, capacity):
        cumulative, _, _, _, capture, active = metadata
        qkv = self.in_proj_qkv(op, hidden)
        z = self.in_proj_z(op, hidden)
        beta = self.in_proj_b(op, hidden)
        decay = self.in_proj_a(op, hidden)
        conv, present_conv, conv_values = self.conv1d(
            op, qkv, state[0], cumulative, capture, capacity
        )
        q, k, v = op.Split(
            conv, [self.key_dim, self.key_dim, self.value_dim], axis=-1, _outputs=3
        )
        q = op.Reshape(q, [-1, self.num_k_heads, self.head_k_dim])
        k = op.Reshape(k, [-1, self.num_k_heads, self.head_k_dim])
        v = op.Reshape(v, [-1, self.num_v_heads, self.head_v_dim])
        output, present_recurrent, capsule = op.GatedDeltaNet(
            q,
            k,
            v,
            cumulative,
            op.Cast(decay, to=ir.DataType.FLOAT),
            op.Cast(beta, to=ir.DataType.FLOAT),
            state[1],
            op.Cast(self.A_log, to=ir.DataType.FLOAT),
            op.Cast(self.dt_bias, to=ir.DataType.FLOAT),
            capture,
            active,
            gate_activation="qwen",
            beta_activation="sigmoid",
            qk_l2_norm=1,
            update_rule="gated_delta",
            scale=0.0,
            state_update_capacity=capacity,
            _domain="com.microsoft",
            _outputs=3,
        )
        output.type = ir.TensorType(hidden.dtype)
        output = op.Reshape(output, [-1, self.head_v_dim])
        z = op.Reshape(z, [-1, self.head_v_dim])
        output = self.norm(op, output, z)
        output = self.out_proj(op, op.Reshape(output, [-1, self.value_dim]))
        return output, (present_conv, present_recurrent), (conv_values, capsule)


class PackedAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name, child in original.named_children():
            setattr(self, name, child)
        for name in (
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rotary_embedding_dim",
            "_rope_interleave",
            "scaling",
        ):
            setattr(self, name, getattr(original, name))

    def forward(self, op, hidden, state, positions, metadata):
        cumulative, past_lengths, blocks, attention_metadata, _, _ = metadata
        q_gate = op.Reshape(
            self.q_proj(op, hidden), [-1, self.num_attention_heads, 2 * self.head_dim]
        )
        q, gate = op.Split(q_gate, num_outputs=2, axis=-1, _outputs=2)
        k = op.Reshape(self.k_proj(op, hidden), [-1, self.num_key_value_heads, self.head_dim])
        q, k = self.q_norm(op, q), self.k_norm(op, k)
        rotated = []
        for value, heads in ((q, self.num_attention_heads), (k, self.num_key_value_heads)):
            value = op.Reshape(value, [1, -1, heads * self.head_dim])
            value = apply_rotary_pos_emb(
                op, value, positions, heads, self.rotary_embedding_dim, self._rope_interleave
            )
            rotated.append(op.Reshape(value, [-1, heads * self.head_dim]))
        v = self.v_proj(op, hidden)
        scale = op.Constant(value_floats=[1.0])
        output, present_k, present_v = op.PagedAttention(
            rotated[0],
            rotated[1],
            v,
            state[0],
            state[1],
            cumulative,
            past_lengths,
            blocks,
            None,
            None,
            None,
            None,
            None,
            None,
            scale,
            scale,
            attention_metadata,
            num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            scale=self.scaling,
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_TENSOR",
            _domain="com.microsoft",
            _outputs=3,
        )
        output.type = ir.TensorType(hidden.dtype)
        gate = op.Reshape(gate, [-1, self.num_attention_heads * self.head_dim])
        output = self.o_proj(op, op.Mul(output, op.Sigmoid(gate)))
        return output, (present_k, present_v), ()


class PackedLayer(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.layer_type = original.layer_type
        for name, child in original.named_children():
            if name == "linear_attn":
                child = PackedDeltaNet(child)
            elif name == "self_attn":
                child = PackedAttention(child)
            setattr(self, name, child)

    def forward(self, op, hidden, state, positions, metadata, capacity):
        normed = self.input_layernorm(op, hidden)
        if self.layer_type == "linear_attention":
            output, present, updates = self.linear_attn(op, normed, state, metadata, capacity)
        else:
            output, present, updates = self.self_attn(op, normed, state, positions, metadata)
        hidden = op.Add(hidden, output)
        hidden = op.Add(hidden, self.mlp(op, self.post_attention_layernorm(op, hidden)))
        return hidden, present, updates


class PackedBackbone(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.embed_tokens = original.embed_tokens
        self.layers = nn.ModuleList([PackedLayer(layer) for layer in original.layers])
        self.norm = original.norm
        self.rotary_emb = original.rotary_emb

    def forward(self, op, ids, position_ids, states, metadata, capacity):
        hidden = self.embed_tokens(op, ids)
        positions = self.rotary_emb(op, op.Unsqueeze(position_ids, [1]))
        present, updates = [], []
        for layer, state in zip(self.layers, states, strict=True):
            hidden, current, update = layer(op, hidden, state, positions, metadata, capacity)
            present.append(current)
            updates.append(update)
        return self.norm(op, hidden), present, updates


class PackedDecoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.model = PackedBackbone(original.model)
        self.lm_head = original.lm_head

    def forward(self, op, ids, position_ids, states, metadata, capacity, logits_indices):
        hidden, present, updates = self.model(
            op, ids, position_ids, states, metadata, capacity
        )
        logits = self.lm_head(op, op.Gather(hidden, logits_indices, axis=0))
        return logits, hidden, present, updates


class PackedMtp(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name, child in original.named_children():
            if name == "layers":
                child = nn.ModuleList([PackedLayer(layer) for layer in child])
            setattr(self, name, child)
        self.multimodal_rope = original.config.mrope_section is not None

    def forward(self, op, ids, seed, position_ids, states, metadata, capacity, logits_indices):
        embeds = self.pre_fc_norm_embedding(op, self.embed_tokens(op, ids))
        seed = self.pre_fc_norm_hidden(op, seed)
        hidden = self.fc(op, op.Concat(embeds, seed, axis=-1))
        rope_ids = (
            op.Unsqueeze(position_ids, [1])
            if self.multimodal_rope
            else op.Gather(position_ids, [0], axis=0)
        )
        positions = self.rotary_emb(op, rope_ids)
        hidden, present, updates = self.layers[0](
            op, hidden, states[0], positions, metadata, capacity
        )
        hidden = self.norm(op, hidden)
        logits = self.lm_head(op, op.Gather(hidden, logits_indices, axis=0))
        return logits, hidden, [present], [updates]


class PackedOrnithTask(ModelTask):
    def __init__(self, *, mtp=False, block_size=256, capacity=8):
        self.mtp = mtp
        self.block_size = block_size
        self.capacity = capacity

    def build(self, module, config):
        graph, builder = _make_graph()
        op = builder.op
        ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=["num_tokens"])
        positions = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[3, "num_tokens"]
        )
        cumulative = builder.input(
            "cumulative_sequence_lengths", dtype=ir.DataType.INT32, shape=["batch_size + 1"]
        )
        past_lengths = builder.input(
            "past_sequence_lengths", dtype=ir.DataType.INT32, shape=["batch_size"]
        )
        blocks = builder.input(
            "block_table", dtype=ir.DataType.INT32, shape=["batch_size", "max_blocks"]
        )
        attention_metadata = builder.input(
            "attention_metadata", dtype=ir.DataType.INT32, shape=[3]
        )
        indices = builder.input(
            "logits_indices", dtype=ir.DataType.INT32, shape=["logits_rows"]
        )
        capture = active = None
        if not self.mtp:
            capture = builder.input(
                "state_update_capture_count", dtype=ir.DataType.INT32, shape=["batch_size"]
            )
            active = builder.input("state_update_active", dtype=ir.DataType.INT32, shape=[1])
        metadata = (cumulative, past_lengths, blocks, attention_metadata, capture, active)
        states, shapes, update_shapes = [], [], []
        for index, kind in enumerate(config.layer_types):
            if kind == "linear_attention":
                channels = 2 * config.linear_num_key_heads * config.linear_key_head_dim
                channels += config.linear_num_value_heads * config.linear_value_head_dim
                conv = ["batch_size", channels, config.linear_conv_kernel_dim - 1]
                recurrent = [
                    "batch_size",
                    config.linear_num_value_heads,
                    config.linear_value_head_dim,
                    config.linear_key_head_dim,
                ]
                state = (
                    builder.input(f"past.{index}.conv", dtype=config.dtype, shape=conv),
                    builder.input(
                        f"past.{index}.recurrent", dtype=ir.DataType.FLOAT, shape=recurrent
                    ),
                )
                shapes.append((conv, recurrent))
                capsule_width = self.capacity * (
                    config.linear_num_value_heads
                    + config.linear_num_key_heads * config.linear_key_head_dim
                    + config.linear_num_value_heads * config.linear_value_head_dim
                )
                update_shapes.append(
                    (["batch_size", self.capacity, channels], ["batch_size", capsule_width])
                )
            else:
                cache = [
                    "num_blocks",
                    self.block_size,
                    config.num_key_value_heads,
                    config.head_dim,
                ]
                state = tuple(
                    builder.input(
                        f"past_key_values.{index}.{side}",
                        dtype=ir.DataType.FLOAT8E4M3FN,
                        shape=cache,
                    )
                    for side in ("key", "value")
                )
                shapes.append((cache, cache))
                update_shapes.append(())
            states.append(state)
        if self.mtp:
            seed = builder.input(
                "hidden_states", dtype=config.dtype, shape=["num_tokens", config.hidden_size]
            )
            logits, hidden, present, updates = module(
                op, ids, seed, positions, states, metadata, self.capacity, indices
            )
        else:
            logits, hidden, present, updates = module.decoder(
                op, ids, positions, states, metadata, self.capacity, indices
            )
        logits.type, logits.shape = (
            ir.TensorType(config.dtype),
            ir.Shape(["logits_rows", config.vocab_size]),
        )
        hidden.type, hidden.shape = (
            ir.TensorType(config.dtype),
            ir.Shape(["num_tokens", config.hidden_size]),
        )
        builder.add_output(logits, "logits")
        builder.add_output(hidden, "hidden_states_out" if self.mtp else "hidden_states")
        for index, (state, kind) in enumerate(zip(present, config.layer_types, strict=True)):
            sides = ("conv", "recurrent") if kind == "linear_attention" else ("key", "value")
            for value, side, shape in zip(state, sides, shapes[index], strict=True):
                dtype = (
                    ir.DataType.FLOAT
                    if side == "recurrent"
                    else (config.dtype if side == "conv" else ir.DataType.FLOAT8E4M3FN)
                )
                value.type, value.shape = ir.TensorType(dtype), ir.Shape(shape)
                builder.add_output(value, f"present.{index}.{side}")
            for value, side, shape in zip(
                updates[index],
                ("conv_value", "recurrent_capsule") if updates[index] else (),
                update_shapes[index],
                strict=True,
            ):
                dtype = config.dtype if side == "conv_value" else ir.DataType.FLOAT
                value.type, value.shape = ir.TensorType(dtype), ir.Shape(shape)
                builder.add_output(value, f"state_update.{index}.{side}")
        graph.opset_imports["com.microsoft"] = 1
        return ModelPackage(
            {"decoder" if not self.mtp else "model": _make_model(graph)}, config=config
        )


def engine_config(component_config, config, *, block_size=256, capacity=8):
    model = component_config["model"]
    model.pop("vision", None)
    model.pop("embedding", None)
    model["type"] = "qwen3_5_moe_text"
    decoder = model["decoder"]
    inputs = {
        name: name
        for name in (
            "input_ids",
            "position_ids",
            "block_table",
            "cumulative_sequence_lengths",
            "past_sequence_lengths",
            "attention_metadata",
            "logits_indices",
            "state_update_capture_count",
            "state_update_active",
        )
    }
    inputs.update(
        past_key_names="past_key_values.%d.key",
        past_value_names="past_key_values.%d.value",
        past_conv_names="past.%d.conv",
        past_recurrent_names="past.%d.recurrent",
    )
    outputs = {
        "logits": "logits",
        "hidden_states": "hidden_states",
        "present_key_names": "present.%d.key",
        "present_value_names": "present.%d.value",
        "present_conv_names": "present.%d.conv",
        "present_recurrent_names": "present.%d.recurrent",
        "state_update_conv_value_names": "state_update.%d.conv_value",
        "state_update_recurrent_capsule_names": "state_update.%d.recurrent_capsule",
    }
    full = [i for i, kind in enumerate(config.layer_types) if kind == "full_attention"]
    linear = [i for i, kind in enumerate(config.layer_types) if kind == "linear_attention"]
    decoder.update(
        filename="decoder/model.onnx",
        inputs=inputs,
        outputs=outputs,
        state_update_capacity=capacity,
        state_groups=[
            {"kind": "paged_kv", "layer_ids": full},
            {
                "kind": "fixed_conv",
                "layer_ids": linear,
                "state_update": {"capacity": capacity},
            },
            {
                "kind": "fixed_recurrent",
                "layer_ids": linear,
                "state_update": {
                    "capacity": capacity,
                    "key_head_count": config.linear_num_key_heads,
                },
            },
        ],
    )
    decoder["session_options"]["provider_options"] = [{"cuda": {"enable_cuda_graph": "0"}}]
    # Packed scheduling bindings are inherited from the target by CreateMtpDecoderConfig.
    mtp_inputs = {
        key: inputs[key]
        for key in ("input_ids", "position_ids", "past_key_names", "past_value_names")
    }
    mtp_inputs["hidden_states"] = "hidden_states"
    model["mtp"] = {
        "enabled": True,
        "filename": "mtp.onnx",
        "num_hidden_layers": 1,
        "num_key_value_heads": config.num_key_value_heads,
        "head_size": config.head_dim,
        "main_hidden_states": "hidden_states",
        "inputs": mtp_inputs,
        "outputs": {
            "logits": "logits",
            "hidden_states": "hidden_states_out",
            "present_key_names": "present.%d.key",
            "present_value_names": "present.%d.value",
        },
    }
    component_config["search"].update(do_sample=False, top_k=1, chunk_size=1024)
    component_config["engine"] = {
        "dynamic_batching": {
            "block_size": block_size,
            "num_blocks": 512,
            "max_batch_size": 1,
            "max_scheduled_tokens": 1024,
            "prefix_caching": False,
        }
    }
    component_config["speculative"] = {"max_draft_tokens": 1}
    return component_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    validate_checkpoint_revision(args.checkpoint)
    provenance = json.loads((args.components / "ornith_export.json").read_text())
    if provenance["model_id"] != MODEL_ID or provenance["revision"] != REVISION:
        raise ValueError("Component export provenance does not match the pinned checkpoint")
    target, config, mtp, mtp_config = create_modules(
        AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    )
    target.decoder = PackedDecoder(target.decoder)
    mtp = PackedMtp(mtp)
    with torch.inference_mode():
        package = build_from_module(
            target, config, PackedOrnithTask(), execution_provider="cuda"
        )
        bind_component_weights(
            package["decoder"], args.components / "decoder/model.onnx", args.checkpoint
        )
        (args.output / "decoder").mkdir(parents=True)
        ir.save(
            package["decoder"],
            args.output / "decoder/model.onnx",
            external_data="model.onnx.data",
            max_workers=1,
        )
        head = build_from_module(
            mtp, mtp_config, PackedOrnithTask(mtp=True), execution_provider="cuda"
        )
        bind_component_weights(
            head["model"],
            args.components / ".mobius-mtp/model.onnx",
            args.checkpoint,
            mtp=True,
        )
        ir.save(
            head["model"],
            args.output / "mtp.onnx",
            external_data="mtp.onnx.data",
            max_workers=1,
        )
    runtime_config = engine_config(
        json.loads((args.components / "genai_config.json").read_text()), config
    )
    (args.output / "genai_config.json").write_text(json.dumps(runtime_config, indent=2) + "\n")
    for source in args.components.iterdir():
        if source.is_file() and (
            "tokenizer" in source.name
            or source.name in {"vocab.json", "merges.txt", "added_tokens.json"}
        ):
            shutil.copy2(source, args.output / source.name)
    (args.output / "ornith_engine_export.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "engine_validated": False,
                "greedy_parity_runtime_environment": {
                    "ORT_DISABLE_MOE_GEMV": "1",
                    "ORT_ENABLE_XQA": "0",
                    "ORT_PAGED_ATTENTION_DISABLE_SPLIT_KV": "1",
                },
                "vision_engine_ready": False,
                "kv_cache": "fp8_e4m3fn, unit scales",
                "state_update_capacity": 8,
                "max_draft_tokens": 1,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("Packed text/MTP export saved; Engine execution remains to be validated")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
