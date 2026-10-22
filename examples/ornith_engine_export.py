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
from collections import defaultdict
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
from mobius._passes._fold_concat import FoldConcatInitializersPass
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.tasks import ModelTask
from mobius.tasks._base import _make_graph, _make_model

logger = logging.getLogger(__name__)


def fuse_moe_routers(model, *, use_matmul=False):
    """Integrate FP16 router projections into compatible group-128 INT2 QMoE."""
    graph = model.graph
    expected = {
        "activation_type": "swiglu",
        "normalize_routing_weights": 1,
        "k": 8,
        "expert_weight_bits": 2,
        "block_size": 128,
        "swiglu_fusion": 1,
        "quant_type": "int",
        "weights_prepacked": 0,
    }

    def single_use(value):
        return value not in graph.outputs and len(tuple(value.uses())) == 1

    fused = 0
    for moe in list(graph):
        if (
            moe.domain != "com.microsoft"
            or moe.op_type != "QMoE"
            or set(moe.attributes) != set(expected)
            or any(moe.attributes[name].value != value for name, value in expected.items())
            or len(moe.inputs) != 7
            or moe.inputs[4] is not None
        ):
            continue
        activation, scores = moe.inputs[:2]
        if (
            activation.dtype != ir.DataType.FLOAT16
            or activation.shape is None
            or activation.shape.rank() != 2
            or (isinstance(activation.shape[0], int) and activation.shape[0] <= 0)
        ):
            continue
        removed = [moe]
        projection_output = scores
        reshape = scores.producer()
        if reshape is not None and not reshape.domain and reshape.op_type == "Reshape":
            if reshape.attributes or not single_use(scores):
                continue
            removed.append(reshape)
            projection_output = reshape.inputs[0]
        matmul = projection_output.producer()
        if (
            matmul is None
            or matmul.domain
            or matmul.op_type != "MatMul"
            or matmul.attributes
            or matmul.inputs[0] is not activation
            or not single_use(projection_output)
        ):
            continue
        transpose = matmul.inputs[1].producer()
        if (
            transpose is None
            or transpose.domain
            or transpose.op_type != "Transpose"
            or set(transpose.attributes) != {"perm"}
            or tuple(transpose.attributes["perm"].as_ints()) != (1, 0)
        ):
            continue
        weight = transpose.inputs[0]
        if (
            not weight.is_initializer()
            or weight.dtype != ir.DataType.FLOAT16
            or weight.shape is None
            or weight.shape.rank() != 2
            or not isinstance(weight.shape[0], int)
            or weight.shape[0] < 8
            or not isinstance(weight.shape[1], int)
            or weight.shape[1] <= 0
            or activation.shape[1] != weight.shape[1]
            or any(
                value.shape != ir.Shape([activation.shape[0], weight.shape[0]])
                for value in (projection_output, scores)
            )
            or any(
                value is None
                or value.shape is None
                or value.shape.rank() != 3
                or value.shape[0] != weight.shape[0]
                for value in (moe.inputs[2], moe.inputs[5])
            )
        ):
            continue
        if reshape is not None and not reshape.domain and reshape.op_type == "Reshape":
            shape = reshape.inputs[1].const_value
            if (
                shape is None
                or shape.dtype != ir.DataType.INT64
                or shape.shape != ir.Shape([2])
                or tuple(shape.numpy().tolist()) != (-1, weight.shape[0])
            ):
                continue
        removed.append(matmul)
        output = ir.Value(
            name=moe.outputs[0].name, shape=moe.outputs[0].shape, type=moe.outputs[0].type
        )
        attributes = dict(moe.attributes)
        if use_matmul:
            attributes["router_use_matmul"] = ir.AttrInt64("router_use_matmul", 1)
        node = ir.Node(
            "com.microsoft",
            "QMoEWithRouter",
            [activation, weight, *moe.inputs[2:]],
            attributes,
            outputs=[output],
        )
        ir.convenience.replace_nodes_and_values(
            graph,
            moe,
            removed,
            [node],
            [moe.outputs[0]],
            [output],
        )
        fused += 1
    if fused:
        ir.passes.common.RemoveUnusedNodesPass()(model)
    return fused


def fuse_shared_expert_gates(model):
    """Replace scalar gate MatMul/Sigmoid/Mul/Add/Add with one FP16 CUDA op."""
    graph = model.graph

    def sole_user(value, op_type):
        uses = tuple(value.uses())
        if value in graph.outputs or len(uses) != 1:
            return None
        user = uses[0].node
        if user.domain or user.op_type != op_type or user.attributes:
            return None
        return user

    fused = 0
    for matmul in list(graph):
        if matmul.domain or matmul.op_type != "MatMul" or matmul.attributes:
            continue
        x, transposed = matmul.inputs
        transpose = transposed.producer()
        if (
            transpose is None
            or transpose.domain
            or transpose.op_type != "Transpose"
            or set(transpose.attributes) != {"perm"}
            or tuple(transpose.attributes["perm"].as_ints()) != (1, 0)
        ):
            continue
        weight = transpose.inputs[0]
        if (
            x.dtype != ir.DataType.FLOAT16
            or weight.dtype != ir.DataType.FLOAT16
            or not weight.is_initializer()
            or weight.shape is None
            or weight.shape.rank() != 2
            or weight.shape[0] != 1
            or not isinstance(weight.shape[1], int)
            or weight.shape[1] <= 0
            or x.shape is None
            or x.shape.rank() != 2
            or x.shape[1] != weight.shape[1]
        ):
            continue
        sigmoid = sole_user(matmul.outputs[0], "Sigmoid")
        mul = sole_user(sigmoid.outputs[0], "Mul") if sigmoid else None
        add = sole_user(mul.outputs[0], "Add") if mul else None
        final = sole_user(add.outputs[0], "Add") if add else None
        if final is None:
            continue
        shared = next((value for value in mul.inputs if value is not sigmoid.outputs[0]), None)
        routed = next((value for value in add.inputs if value is not mul.outputs[0]), None)
        residual = next((value for value in final.inputs if value is not add.outputs[0]), None)
        if any(
            value is None
            or value.dtype != ir.DataType.FLOAT16
            or value.shape is None
            or value.shape.rank() != 2
            or not isinstance(value.shape[1], int)
            or value.shape[1] <= 0
            or value.shape != shared.shape
            or value.shape[0] != x.shape[0]
            for value in (shared, routed, residual)
        ):
            continue
        output = ir.Value(
            name=final.outputs[0].name,
            shape=final.outputs[0].shape,
            type=final.outputs[0].type,
        )
        node = ir.Node(
            "com.microsoft",
            "MatMulSigmoidGatedResidual",
            [x, weight, shared, routed, residual],
            outputs=[output],
        )
        ir.convenience.replace_nodes_and_values(
            graph,
            final,
            [matmul, sigmoid, mul, add, final],
            [node],
            [final.outputs[0]],
            [output],
        )
        fused += 1
    if fused:
        graph.opset_imports.setdefault("com.microsoft", 1)
        ir.passes.common.RemoveUnusedNodesPass()(model)
    return fused


def fuse_int2_swiglu(model):
    """Fuse INT2 gate/up projections using ORT's optimized QuickGelu SiLU arithmetic."""
    graph = model.graph

    def sole_user(value, op_type, domain=""):
        uses = tuple(value.uses())
        if value in graph.outputs or len(uses) != 1:
            return None
        node = uses[0].node
        return node if node.domain == domain and node.op_type == op_type else None

    fused = 0
    for projection in list(graph):
        if (
            projection.domain != "com.microsoft"
            or projection.op_type != "MatMulNBits"
            or len(projection.inputs) != 3
            or set(projection.attributes) != {"K", "N", "bits", "block_size"}
            or projection.attributes["bits"].as_int() != 2
            or projection.attributes["block_size"].as_int() != 128
        ):
            continue
        x, weight, scales = projection.inputs
        k = projection.attributes["K"].as_int()
        n = projection.attributes["N"].as_int()
        if (
            k <= 0
            or k % 128
            or k > 2**31 - 1
            or n <= 0
            or n % 16
            or n > 2**31 - 1
            or x is None
            or x.dtype != ir.DataType.FLOAT16
            or x.shape is None
            or x.shape.rank() != 2
            or x.shape[1] != k
            or weight is None
            or not weight.is_initializer()
            or weight.dtype != ir.DataType.UINT8
            or weight.shape != ir.Shape([n, k // 128, 32])
            or scales is None
            or not scales.is_initializer()
            or scales.dtype != ir.DataType.FLOAT16
            or scales.shape != ir.Shape([n, k // 128])
        ):
            continue
        split = sole_user(projection.outputs[0], "Split")
        if (
            split is None
            or len(split.inputs) != 2
            or len(split.outputs) != 2
            or set(split.attributes) != {"axis"}
            or split.attributes["axis"].as_int() not in (-1, 1)
            or split.inputs[1] is None
            or not split.inputs[1].is_initializer()
            or split.inputs[1].dtype != ir.DataType.INT64
            or split.inputs[1].const_value is None
            or split.inputs[1].shape != ir.Shape([2])
            or tuple(split.inputs[1].const_value.numpy()) != (n // 2, n // 2)
        ):
            continue
        gate, up = split.outputs
        swish = sole_user(gate, "Swish")
        if swish is None:
            swish = sole_user(gate, "QuickGelu", "com.microsoft")
        mul = sole_user(up, "Mul")
        if (
            swish is None
            or mul is None
            or set(swish.attributes) - {"alpha"}
            or (swish.op_type == "QuickGelu" and "alpha" not in swish.attributes)
            or (
                "alpha" in swish.attributes and swish.attributes["alpha"].as_float() != 1.0  # noqa: RUF069 - exact activation semantics
            )
            or mul.attributes
            or sole_user(swish.outputs[0], "Mul") is not mul
            or set(mul.inputs) != {swish.outputs[0], up}
            or any(
                value.dtype != ir.DataType.FLOAT16
                or value.shape != ir.Shape([x.shape[0], n // 2])
                for value in (gate, up, swish.outputs[0], mul.outputs[0])
            )
            or projection.outputs[0].dtype != ir.DataType.FLOAT16
            or projection.outputs[0].shape != ir.Shape([x.shape[0], n])
        ):
            continue
        output = ir.Value(
            name=mul.outputs[0].name,
            shape=mul.outputs[0].shape,
            type=mul.outputs[0].type,
        )
        node = ir.Node(
            "com.microsoft",
            "MatMulNBitsSwiGLU",
            [x, weight, scales],
            [ir.AttrInt64("K", k), ir.AttrInt64("N", n)],
            outputs=[output],
        )
        ir.convenience.replace_nodes_and_values(
            graph, mul, [projection, split, swish, mul], [node], [mul.outputs[0]], [output]
        )
        fused += 1
    if fused:
        ir.passes.common.RemoveUnusedNodesPass()(model)
    return fused


def pack_int2_projections(model):
    """Pack compatible shared-input projections without requantizing their weights."""
    graph = model.graph
    groups = defaultdict(list)
    for node in graph:
        if (
            node.domain != "com.microsoft"
            or node.op_type != "MatMulNBits"
            or len(node.inputs) != 3
            or set(node.attributes) != {"K", "N", "bits", "block_size"}
            or node.attributes["bits"].as_int() != 2
            or node.attributes["block_size"].as_int() != 128
            or any(
                value is None or not value.is_initializer() or value.const_value is None
                for value in node.inputs[1:]
            )
        ):
            continue
        k = node.attributes["K"].as_int()
        n = node.attributes["N"].as_int()
        if (
            k % 128
            or node.inputs[1].shape != ir.Shape([n, k // 128, 32])
            or node.inputs[2].shape != ir.Shape([n, k // 128])
            or node.outputs[0].shape is None
            or node.outputs[0].shape.rank() != 2
        ):
            continue
        groups[
            (
                node.inputs[0],
                k,
                tuple(value.dtype for value in node.inputs[1:]),
            )
        ].append(node)
    removed = 0
    for nodes in groups.values():
        if len(nodes) < 2:
            continue
        widths = [node.attributes["N"].as_int() for node in nodes]
        # Keep every original column aligned to the same decode-kernel warp tile.
        if any(width % 16 for width in widths):
            continue
        new_nodes = []
        packed_inputs = [nodes[0].inputs[0]]
        for index in (1, 2):
            parts = [node.inputs[index] for node in nodes]
            shape = list(parts[0].shape)
            shape[0] = sum(widths)
            value = ir.Value(shape=ir.Shape(shape), type=parts[0].type)
            new_nodes.append(
                ir.Node("", "Concat", parts, [ir.AttrInt64("axis", 0)], outputs=[value])
            )
            packed_inputs.append(value)
        attributes = dict(nodes[0].attributes)
        attributes["N"] = ir.AttrInt64("N", sum(widths))
        packed_shape = list(nodes[0].outputs[0].shape)
        packed_shape[-1] = sum(widths)
        packed = ir.Value(shape=ir.Shape(packed_shape), type=nodes[0].outputs[0].type)
        new_nodes.append(
            ir.Node(
                "com.microsoft", "MatMulNBits", packed_inputs, attributes, outputs=[packed]
            )
        )
        sizes = ir.Value(
            name=nodes[0].outputs[0].name + ".packed_split_sizes",
            shape=ir.Shape([len(widths)]),
            type=ir.TensorType(ir.DataType.INT64),
            const_value=ir.tensor(widths, dtype=ir.DataType.INT64),
        )
        graph.register_initializer(sizes)
        outputs = [
            ir.Value(
                name=node.outputs[0].name,
                shape=node.outputs[0].shape,
                type=node.outputs[0].type,
            )
            for node in nodes
        ]
        new_nodes.append(
            ir.Node("", "Split", [packed, sizes], [ir.AttrInt64("axis", -1)], outputs=outputs)
        )
        ir.convenience.replace_nodes_and_values(
            graph, nodes[0], nodes, new_nodes, [node.outputs[0] for node in nodes], outputs
        )
        removed += len(nodes) - 1
    if removed:
        FoldConcatInitializersPass()(model)
        ir.passes.common.RemoveUnusedNodesPass()(model)
    return removed


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
    def __init__(self, *, mtp=False, block_size=256, capacity=8, recurrent_state_fp16=False):
        self.mtp = mtp
        self.block_size = block_size
        self.capacity = capacity
        self.recurrent_dtype = (
            ir.DataType.FLOAT16 if recurrent_state_fp16 else ir.DataType.FLOAT
        )

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
                        f"past.{index}.recurrent", dtype=self.recurrent_dtype, shape=recurrent
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
                    self.recurrent_dtype
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
    parser.add_argument(
        "--fuse-moe-routers",
        action="store_true",
        help="Integrate FP16 router MatMuls into group-128 INT2 QMoE",
    )
    parser.add_argument(
        "--moe-router-use-matmul",
        action="store_true",
        help="Use CUDA MatMul inside integrated routers at every row count; requires --fuse-moe-routers",
    )
    parser.add_argument(
        "--recurrent-state-fp16",
        action="store_true",
        help="Store target/MTP recurrent states in FP16, retaining FP32 updates and capsules",
    )
    parser.add_argument(
        "--fuse-shared-expert-gates",
        action="store_true",
        help="Fuse FP16 shared gate MatMul, sigmoid, expert sum and residual",
    )
    parser.add_argument(
        "--fuse-shared-expert-swiglu",
        action="store_true",
        help="Fuse packed FP16 INT2 gate/up projection and SwiGLU; requires --pack-projections",
    )
    parser.add_argument(
        "--pack-projections",
        action="store_true",
        help="Pack shared-input INT2 projections to reduce decode kernel launches",
    )
    args = parser.parse_args()
    if args.moe_router_use_matmul and not args.fuse_moe_routers:
        parser.error("--moe-router-use-matmul requires --fuse-moe-routers")
    if args.fuse_shared_expert_swiglu and not args.pack_projections:
        parser.error("--fuse-shared-expert-swiglu requires --pack-projections")
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
            target,
            config,
            PackedOrnithTask(recurrent_state_fp16=args.recurrent_state_fp16),
            execution_provider="cuda",
        )
        bind_component_weights(
            package["decoder"], args.components / "decoder/model.onnx", args.checkpoint
        )
        packed_projections = (
            pack_int2_projections(package["decoder"]) if args.pack_projections else 0
        )
        fused_swiglu = (
            fuse_int2_swiglu(package["decoder"]) if args.fuse_shared_expert_swiglu else 0
        )
        fused_gates = (
            fuse_shared_expert_gates(package["decoder"])
            if args.fuse_shared_expert_gates
            else 0
        )
        fused_routers = (
            fuse_moe_routers(package["decoder"], use_matmul=args.moe_router_use_matmul)
            if args.fuse_moe_routers
            else 0
        )
        (args.output / "decoder").mkdir(parents=True)
        ir.save(
            package["decoder"],
            args.output / "decoder/model.onnx",
            external_data="model.onnx.data",
            max_workers=1,
        )
        head = build_from_module(
            mtp,
            mtp_config,
            PackedOrnithTask(mtp=True, recurrent_state_fp16=args.recurrent_state_fp16),
            execution_provider="cuda",
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
                "packed_projection_nodes_removed": packed_projections,
                "shared_expert_gates_fused": fused_gates,
                "shared_expert_swiglu_fused": fused_swiglu,
                "moe_routers_fused": fused_routers,
                "moe_router_use_matmul": args.moe_router_use_matmul,
                "recurrent_state_dtype": "float16" if args.recurrent_state_fp16 else "float32",
                "greedy_parity_runtime_environment": {
                    "ORT_DISABLE_MOE_GEMV": "1",
                    "ORT_ENABLE_XQA": "0",
                    "ORT_PAGED_ATTENTION_DECODE_NUM_SPLITS": "8",
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
