# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the quantized GGUF → ONNX build pipeline."""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import httpx
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from huggingface_hub.utils import OfflineModeIsEnabled


def _gguf_header_prefix(*architectures: str) -> bytes:
    key = b"general.architecture"
    entries: list[bytes] = []
    for architecture in architectures:
        value = architecture.encode("utf-8")
        entries.extend(
            [
                struct.pack("<Q", len(key)),
                key,
                struct.pack("<I", 8),
                struct.pack("<Q", len(value)),
                value,
            ]
        )
    return b"".join(
        [
            b"GGUF",
            struct.pack("<I", 3),
            struct.pack("<Q", 0),
            struct.pack("<Q", len(architectures)),
            *entries,
        ]
    )


def _run_gather_block_quantized(
    tmp_path: Path,
    *,
    zero_point: int,
) -> np.ndarray:
    """Run a tiny GatherBlockQuantized graph with a controlled zero point."""
    qweight = np.full((2, 16), 0xAA, dtype=np.uint8)
    scales = np.array([[0.5], [0.25]], dtype=np.float16)
    zero_points = np.full((2, 1), zero_point, dtype=np.uint8)

    def _const(name: str, arr: np.ndarray) -> ir.Value:
        value = ir.Value(name=name)
        tensor = ir.tensor(arr)
        value.const_value = tensor
        value.shape = ir.Shape(arr.shape)
        value.dtype = tensor.dtype
        return value

    input_ids = ir.Value(
        name="input_ids",
        shape=ir.Shape([2]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    output = ir.Value(
        name="output",
        shape=ir.Shape([2, 32]),
        type=ir.TensorType(ir.DataType.FLOAT16),
    )
    qweight_init = _const("qweight", qweight)
    scales_init = _const("scales", scales)
    zero_points_init = _const("zero_points", zero_points)
    node = ir.Node(
        "com.microsoft",
        "GatherBlockQuantized",
        inputs=[
            qweight_init,
            input_ids,
            scales_init,
            zero_points_init,
        ],
        outputs=[output],
        attributes=ir.convenience.convert_attributes(
            {"bits": 4, "block_size": 32, "gather_axis": 0, "quantize_axis": 1}
        ),
    )
    graph = ir.Graph(
        inputs=[input_ids],
        outputs=[output],
        nodes=[node],
        initializers=[qweight_init, scales_init, zero_points_init],
        opset_imports={"": 18, "com.microsoft": 1},
        name="gbq_zero_point",
    )
    model = ir.Model(graph, ir_version=10)
    path = tmp_path / f"gbq_zp_{zero_point}.onnx"
    ir.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (result,) = session.run(None, {"input_ids": np.array([0, 1], dtype=np.int64)})
    return result


def _write_quantized_gguf(
    path: Path,
    *,
    architecture: str = "llama",
    hidden_size: int = 64,
    num_layers: int = 1,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    intermediate_size: int = 128,
    vocab_size: int = 256,
    quantize_embedding: bool = False,
    embedding_quantization: str | None = None,
    projection_quantization: str = "q4_0",
    value_projection_quantization: str | None = None,
    output_quantization: str | None = None,
    tie_embeddings: bool = False,
    float_type: str = "f32",
    fused_qkv: bool = False,
    fused_qkv_float: bool = False,
) -> None:
    """Write a GGUF file with quantized projection weights.

    Norms are float32; all linear-layer weights in decoder blocks are
    encoded with *projection_quantization*. The embedding can optionally
    be Q4_0 and tied to the LM head.
    """
    from gguf import GGMLQuantizationType, GGUFWriter

    writer = GGUFWriter(str(path), architecture)
    writer.add_context_length(512)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_block_count(num_layers)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_rope_freq_base(10000.0)
    if architecture in {"olmo", "cohere2"}:
        writer.add_layer_norm_eps(1e-5)
    else:
        writer.add_layer_norm_rms_eps(1e-5)
    if architecture == "cohere2":
        writer.add_sliding_window(128)
        writer.add_logit_scale(0.0625)
        writer.add_rope_dimension_count(head_dim := hidden_size // num_heads)
    elif architecture == "granitemoe":
        writer.add_logit_scale(16.0)
    writer.add_vocab_size(vocab_size)
    if architecture in {"dream", "llada", "llada-moe", "rnd1"}:
        writer.add_uint32("tokenizer.ggml.mask_token_id", vocab_size - 1)

    head_dim = hidden_size // num_heads

    def _add_f32(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    def _add_f16(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float16))

    def _add_bf16(name: str, shape: tuple[int, ...]) -> None:
        values = np.random.randn(*shape).astype(np.float32)
        raw = (values.view(np.uint32) >> 16).astype(np.uint16)
        writer.add_tensor(
            name,
            raw,
            raw_shape=shape,
            raw_dtype=GGMLQuantizationType.BF16,
        )

    add_float = {"f32": _add_f32, "f16": _add_f16, "bf16": _add_bf16}[float_type]

    def _add_q4_0(name: str, n_out: int, k_in: int) -> None:
        """Write a Q4_0-quantized weight tensor."""
        block_size = 32
        block_bytes = 18  # 2B scale + 16B quants
        n_blocks = k_in // block_size
        bytes_per_row = n_blocks * block_bytes
        raw = np.zeros((n_out, bytes_per_row), dtype=np.uint8)
        for row in range(n_out):
            for b in range(n_blocks):
                off = b * block_bytes
                # Random fp16 scale
                scale = np.random.uniform(0.01, 1.0)
                raw[row, off : off + 2] = np.array([scale], dtype=np.float16).view(np.uint8)
                # Random packed nibbles
                raw[row, off + 2 : off + 18] = np.random.randint(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    def _add_q8_0(name: str, n_out: int, k_in: int) -> None:
        """Write a Q8_0-quantized weight tensor."""
        block_size = 32
        block_bytes = 34  # 2B scale + 32B quants
        n_blocks = k_in // block_size
        bytes_per_row = n_blocks * block_bytes
        raw = np.zeros((n_out, bytes_per_row), dtype=np.uint8)
        for row in range(n_out):
            for b in range(n_blocks):
                off = b * block_bytes
                scale = np.random.uniform(0.01, 1.0)
                raw[row, off : off + 2] = np.array([scale], dtype=np.float16).view(np.uint8)
                raw[row, off + 2 : off + 34] = (
                    np.random.randint(-127, 128, size=32, dtype=np.int16)
                    .astype(np.int8, copy=False)
                    .view(np.uint8)
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q8_0)

    def _add_q5_1(name: str, n_out: int, k_in: int) -> None:
        """Write a Q5_1-quantized weight tensor."""
        block_size = 32
        block_bytes = 24  # 2B scale + 2B min + 4B high bits + 16B low nibbles
        n_blocks = k_in // block_size
        bytes_per_row = n_blocks * block_bytes
        raw = np.zeros((n_out, bytes_per_row), dtype=np.uint8)
        for row in range(n_out):
            for b in range(n_blocks):
                off = b * block_bytes
                scale = np.random.uniform(0.01, 1.0)
                minimum = np.random.uniform(-0.5, 0.5)
                raw[row, off : off + 2] = np.array([scale], dtype=np.float16).view(np.uint8)
                raw[row, off + 2 : off + 4] = np.array([minimum], dtype=np.float16).view(
                    np.uint8
                )
                raw[row, off + 4 : off + 8] = np.random.randint(0, 256, size=4, dtype=np.uint8)
                raw[row, off + 8 : off + 24] = np.random.randint(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q5_1)

    def _add_native(name: str, n_out: int, k_in: int, format_name: str) -> None:
        """Write deterministic native blocks with embedded scales."""
        qtype, block_elements, block_bytes = {
            "mxfp4": (GGMLQuantizationType.MXFP4, 32, 17),
            "iq4_nl": (GGMLQuantizationType.IQ4_NL, 32, 18),
            "iq4_xs": (GGMLQuantizationType.IQ4_XS, 256, 136),
            "iq3_s": (GGMLQuantizationType.IQ3_S, 256, 110),
            "iq3_xxs": (GGMLQuantizationType.IQ3_XXS, 256, 98),
            "iq2_xxs": (GGMLQuantizationType.IQ2_XXS, 256, 66),
            "iq2_xs": (GGMLQuantizationType.IQ2_XS, 256, 74),
            "iq2_s": (GGMLQuantizationType.IQ2_S, 256, 82),
            "iq1_s": (GGMLQuantizationType.IQ1_S, 256, 50),
            "iq1_m": (GGMLQuantizationType.IQ1_M, 256, 56),
        }[format_name]
        n_blocks = (k_in + block_elements - 1) // block_elements
        raw = np.arange(n_out * n_blocks * block_bytes, dtype=np.uint8).reshape(
            n_out, n_blocks * block_bytes
        )
        writer.add_tensor(name, raw, raw_dtype=qtype)

    if quantize_embedding:
        _add_q4_0("token_embd.weight", vocab_size, hidden_size)
    elif embedding_quantization == "q8_0":
        _add_q8_0("token_embd.weight", vocab_size, hidden_size)
    elif embedding_quantization == "q5_1":
        _add_q5_1("token_embd.weight", vocab_size, hidden_size)
    elif embedding_quantization == "iq4_nl":
        block_bytes = 18
        n_blocks = (hidden_size + 31) // 32
        raw = np.zeros((vocab_size, n_blocks * block_bytes), dtype=np.uint8)
        scale = np.array([1.0], dtype=np.float16).view(np.uint8)
        for block_index in range(n_blocks):
            offset = block_index * block_bytes
            raw[:, offset : offset + 2] = scale
        writer.add_tensor(
            "token_embd.weight",
            raw,
            raw_dtype=GGMLQuantizationType.IQ4_NL,
        )
    elif embedding_quantization is not None:
        _add_native("token_embd.weight", vocab_size, hidden_size, embedding_quantization)
    else:
        add_float("token_embd.weight", (vocab_size, hidden_size))

    if projection_quantization in {"q4_0", "q5_1", "q8_0"}:
        add_projection = {
            "q4_0": _add_q4_0,
            "q5_1": _add_q5_1,
            "q8_0": _add_q8_0,
        }[projection_quantization]
    elif projection_quantization in {"f32", "f16", "bf16"}:

        def add_projection(name: str, n_out: int, k_in: int) -> None:
            add_float(name, (n_out, k_in))

    else:

        def add_projection(name: str, n_out: int, k_in: int) -> None:
            _add_native(name, n_out, k_in, projection_quantization)

    add_value_projection = (
        _add_q5_1 if value_projection_quantization == "q5_1" else add_projection
    )

    for i in range(num_layers):
        if fused_qkv:
            fused_rows = (num_heads + 2 * num_kv_heads) * head_dim
            if fused_qkv_float:
                add_float(f"blk.{i}.attn_qkv.weight", (fused_rows, hidden_size))
            else:
                add_projection(f"blk.{i}.attn_qkv.weight", fused_rows, hidden_size)
        else:
            add_projection(
                f"blk.{i}.attn_q.weight",
                num_heads * head_dim,
                hidden_size,
            )
            add_projection(
                f"blk.{i}.attn_k.weight",
                num_kv_heads * head_dim,
                hidden_size,
            )
            add_value_projection(
                f"blk.{i}.attn_v.weight",
                num_kv_heads * head_dim,
                hidden_size,
            )
        add_projection(
            f"blk.{i}.attn_output.weight",
            hidden_size,
            num_heads * head_dim,
        )
        if architecture != "arcee":
            add_projection(f"blk.{i}.ffn_gate.weight", intermediate_size, hidden_size)
        add_projection(f"blk.{i}.ffn_up.weight", intermediate_size, hidden_size)
        add_projection(f"blk.{i}.ffn_down.weight", hidden_size, intermediate_size)
        if architecture == "olmo2":
            add_float(f"blk.{i}.post_attention_norm.weight", (hidden_size,))
            add_float(f"blk.{i}.post_ffw_norm.weight", (hidden_size,))
            add_float(f"blk.{i}.attn_q_norm.weight", (num_heads * head_dim,))
            add_float(f"blk.{i}.attn_k_norm.weight", (num_kv_heads * head_dim,))
        elif architecture == "cohere2":
            add_float(f"blk.{i}.attn_norm.weight", (hidden_size,))
        elif architecture == "seed_oss":
            add_float(f"blk.{i}.attn_norm.weight", (hidden_size,))
            add_float(f"blk.{i}.post_attention_norm.weight", (hidden_size,))
        elif architecture != "olmo":
            add_float(f"blk.{i}.attn_norm.weight", (hidden_size,))
            add_float(f"blk.{i}.ffn_norm.weight", (hidden_size,))

    # Output norm + optional untied lm_head
    if architecture != "olmo":
        add_float("output_norm.weight", (hidden_size,))
    if architecture != "cohere2" and not tie_embeddings:
        if output_quantization == "q4_0":
            _add_q4_0("output.weight", vocab_size, hidden_size)
        elif output_quantization == "q8_0":
            _add_q8_0("output.weight", vocab_size, hidden_size)
        else:
            add_float("output.weight", (vocab_size, hidden_size))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_encoder_gguf(
    path: Path,
    architecture: str,
    *,
    quantized: bool = False,
    fused_qkv: bool = False,
    fused_qkv_float: bool = False,
    fused_width_delta: int = 0,
    kv_heads: int | None = 4,
    pooling_type: int = 0,
    include_head: bool = False,
    omit: str | None = None,
    malformed: str | None = None,
    auxiliary_suffix: str | None = None,
) -> None:
    """Write a one-layer BERT or ModernBERT backbone with pinned tensor names."""
    from gguf import GGMLQuantizationType, GGUFWriter, PoolingType

    hidden = 64
    intermediate = 64
    vocab = 64
    context = 32
    writer = GGUFWriter(str(path), architecture)
    writer.add_context_length(context)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(1)
    writer.add_head_count(4)
    if kv_heads is not None:
        writer.add_head_count_kv(kv_heads)
    writer.add_vocab_size(vocab)
    writer.add_causal_attention(False)
    writer.add_pooling_type(PoolingType(pooling_type))
    writer.add_tokenizer_model("bert")
    writer.add_token_list([f"token-{index}" for index in range(vocab)])

    if architecture == "bert":
        writer.add_layer_norm_eps(1e-5)
        writer.add_token_type_count(2)
    else:
        writer.add_layer_norm_eps(1e-5)
        writer.add_rope_dimension_count(hidden // 4)
        writer.add_rope_freq_base(10000.0)
        writer.add_string(f"{architecture}.hidden_activation", "gelu")

    rng = np.random.default_rng(0)

    def add_float(name: str, shape: tuple[int, ...]) -> None:
        if name == omit:
            return
        if name == malformed:
            shape = (*shape, 1)
        values = (
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            if name.endswith("attn_rel_b.weight")
            else rng.normal(0, 0.02, shape).astype(np.float32)
        )
        writer.add_tensor(name, values)

    def add_q4(name: str, n_out: int, k_in: int) -> None:
        if name == omit:
            return
        if name == malformed:
            n_out += 1
        raw = np.zeros((n_out, (k_in // 32) * 18), dtype=np.uint8)
        for block in range(k_in // 32):
            offset = block * 18
            raw[:, offset] = 0
            raw[:, offset + 1] = 60  # fp16 scale of one
            raw[:, offset + 2 : offset + 18] = rng.integers(
                0, 256, (n_out, 16), dtype=np.uint8
            )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    def add_matrix(name: str, shape: tuple[int, int]) -> None:
        if quantized:
            add_q4(name, *shape)
        else:
            add_float(name, shape)

    add_matrix("token_embd.weight", (vocab, hidden))
    if architecture == "bert":
        add_float("token_types.weight", (2, hidden))
        add_float("position_embd.weight", (context, hidden))
        add_float("token_embd_norm.weight", (hidden,))
        add_float("token_embd_norm.bias", (hidden,))
        if fused_qkv:
            fused_shape = (3 * hidden + fused_width_delta, hidden)
            if fused_qkv_float:
                add_float("blk.0.attn_qkv.weight", fused_shape)
            else:
                add_matrix("blk.0.attn_qkv.weight", fused_shape)
            add_float("blk.0.attn_qkv.bias", (3 * hidden + fused_width_delta,))
            projections = ("attn_output",)
        else:
            projections = ("attn_q", "attn_k", "attn_v", "attn_output")
        for projection in projections:
            add_matrix(f"blk.0.{projection}.weight", (hidden, hidden))
            add_float(f"blk.0.{projection}.bias", (hidden,))
        for norm in ("attn_output_norm", "layer_output_norm"):
            add_float(f"blk.0.{norm}.weight", (hidden,))
            add_float(f"blk.0.{norm}.bias", (hidden,))
        add_matrix("blk.0.ffn_up.weight", (intermediate, hidden))
        add_float("blk.0.ffn_up.bias", (intermediate,))
        add_matrix("blk.0.ffn_down.weight", (hidden, intermediate))
        add_float("blk.0.ffn_down.bias", (hidden,))
    else:
        add_float("token_embd_norm.weight", (hidden,))
        add_float("output_norm.weight", (hidden,))
        add_matrix("blk.0.attn_qkv.weight", (3 * hidden, hidden))
        add_matrix("blk.0.attn_output.weight", (hidden, hidden))
        add_matrix("blk.0.ffn_up.weight", (2 * intermediate, hidden))
        add_matrix("blk.0.ffn_down.weight", (hidden, intermediate))
        add_float("blk.0.ffn_norm.weight", (hidden,))

    if include_head:
        add_float("cls.weight", (hidden, hidden))
    if auxiliary_suffix is not None:
        add_float(f"blk.0.attn_output.{auxiliary_suffix}", (1,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def _write_t5_gguf(
    path: Path,
    architecture: str,
    *,
    quantized: bool = False,
    gated: bool = False,
    encoder_layers: int = 1,
    decoder_layers: int = 1,
    omit: str | None = None,
    malformed: str | None = None,
    auxiliary_suffix: str | None = None,
    extra_layer_tensor: str | None = None,
    include_ignored_tensor: bool = False,
) -> None:
    """Write a tiny pinned-layout T5 or T5-encoder GGUF."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 32
    intermediate = 64
    vocab = 64
    heads = 4
    head_dim = hidden // heads
    buckets = 8
    writer = GGUFWriter(str(path), architecture)
    writer.add_context_length(32)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(encoder_layers)
    writer.add_head_count(heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_relative_attn_buckets_count(buckets)
    writer.add_vocab_size(vocab)
    writer.add_tokenizer_model("t5")
    writer.add_token_list([f"token-{index}" for index in range(vocab)])
    writer.add_pad_token_id(0)
    writer.add_eos_token_id(1)
    if architecture == "t5":
        writer.add_decoder_block_count(decoder_layers)
        writer.add_decoder_start_token_id(0)

    rng = np.random.default_rng(0)

    def add_float(name: str, shape: tuple[int, ...]) -> None:
        if name == omit:
            return
        if name == malformed:
            shape = (*shape, 1)
        values = (
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            if name.endswith("attn_rel_b.weight")
            else rng.normal(0, 0.02, shape).astype(np.float32)
        )
        writer.add_tensor(name, values)

    def add_q4(name: str, shape: tuple[int, int]) -> None:
        if name == omit:
            return
        n_out, k_in = shape
        if name == malformed:
            n_out += 1
        raw = np.zeros((n_out, (k_in // 32) * 18), dtype=np.uint8)
        for block in range(k_in // 32):
            offset = block * 18
            raw[:, offset] = 0
            raw[:, offset + 1] = 60
            raw[:, offset + 2 : offset + 18] = rng.integers(
                0, 256, (n_out, 16), dtype=np.uint8
            )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    def add_matrix(name: str, shape: tuple[int, int]) -> None:
        if quantized:
            add_q4(name, shape)
        else:
            add_float(name, shape)

    add_matrix("token_embd.weight", (vocab, hidden))
    add_float("enc.output_norm.weight", (hidden,))
    if include_ignored_tensor and architecture == "t5encoder":
        add_matrix("output.weight", (vocab, hidden))

    def add_stack(prefix: str, layers: int, *, decoder: bool) -> None:
        for layer in range(layers):
            base = f"{prefix}.blk.{layer}"
            add_float(f"{base}.attn_norm.weight", (hidden,))
            for projection in ("q", "k", "v"):
                add_matrix(f"{base}.attn_{projection}.weight", (hidden, hidden))
            add_matrix(f"{base}.attn_o.weight", (hidden, hidden))
            add_float(f"{base}.ffn_norm.weight", (hidden,))
            if gated:
                add_matrix(f"{base}.ffn_gate.weight", (intermediate, hidden))
            add_matrix(f"{base}.ffn_up.weight", (intermediate, hidden))
            add_matrix(f"{base}.ffn_down.weight", (hidden, intermediate))
            if layer == 0:
                add_float(f"{base}.attn_rel_b.weight", (buckets, heads))
            if decoder:
                add_float(f"{base}.cross_attn_norm.weight", (hidden,))
                for projection in ("q", "k", "v"):
                    add_matrix(
                        f"{base}.cross_attn_{projection}.weight",
                        (hidden, hidden),
                    )
                add_matrix(f"{base}.cross_attn_o.weight", (hidden, hidden))

    add_stack("enc", encoder_layers, decoder=False)
    if architecture == "t5":
        add_float("dec.output_norm.weight", (hidden,))
        add_stack("dec", decoder_layers, decoder=True)
        if include_ignored_tensor:
            add_float("dec.blk.0.cross_attn_rel_b.weight", (buckets, heads))
    if auxiliary_suffix is not None:
        add_float(f"enc.blk.0.attn_q.{auxiliary_suffix}", (1,))
    if extra_layer_tensor is not None:
        add_matrix(extra_layer_tensor, (hidden, hidden))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def _write_recurrent_gguf(
    path: Path,
    architecture: str,
    *,
    quantized: bool,
    malformed_conv: bool = False,
    auxiliary_scale: bool = False,
    malformed_suffix: bool = False,
) -> None:
    """Write an exact one-layer Mamba/Mamba2 GGUF tensor set."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden_size = 32
    inner_size = 64
    state_size = 8
    conv_kernel = 4
    dt_rank_or_heads = 4
    vocab_size = 64
    groups = 1
    rng = np.random.default_rng(7)

    writer = GGUFWriter(str(path), architecture)
    writer.add_context_length(64)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(0)
    writer.add_block_count(1)
    writer.add_head_count(0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(vocab_size)
    writer.add_ssm_conv_kernel(conv_kernel)
    writer.add_ssm_inner_size(inner_size)
    writer.add_ssm_state_size(state_size)
    writer.add_ssm_time_step_rank(dt_rank_or_heads)
    if architecture == "mamba":
        writer.add_ssm_dt_b_c_rms(False)
    else:
        writer.add_ssm_group_count(groups)

    def add_float(name: str, shape: tuple[int, ...], *, negative: bool = False) -> None:
        values = rng.normal(0.0, 0.05, size=shape).astype(np.float32)
        if negative:
            values = -np.exp(values)
        writer.add_tensor(name, values)

    def add_q4(name: str, shape: tuple[int, int]) -> None:
        rows, columns = shape
        assert columns % 32 == 0
        raw = np.zeros((rows, columns // 32 * 18), dtype=np.uint8)
        for row in range(rows):
            for block in range(columns // 32):
                offset = block * 18
                raw[row, offset : offset + 2] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[row, offset + 2 : offset + 18] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    add_projection = add_q4 if quantized else add_float
    add_float("token_embd.weight", (vocab_size, hidden_size))
    add_float("output_norm.weight", (hidden_size,))
    add_float("output.weight", (vocab_size, hidden_size))
    add_float("blk.0.attn_norm.weight", (hidden_size,))
    if architecture == "mamba":
        add_projection("blk.0.ssm_in.weight", (2 * inner_size, hidden_size))
        conv_width = conv_kernel + 1 if malformed_conv else conv_kernel
        add_float("blk.0.ssm_conv1d.weight", (inner_size, conv_width))
        add_float("blk.0.ssm_conv1d.bias", (inner_size,))
        add_projection(
            "blk.0.ssm_x.weight",
            (dt_rank_or_heads + 2 * state_size, inner_size),
        )
        # The small dt projection is deliberately float: its input dimension
        # cannot satisfy Q4_0's 32-value block ABI.
        add_float("blk.0.ssm_dt.weight", (inner_size, dt_rank_or_heads))
        add_float("blk.0.ssm_dt.bias", (inner_size,))
        add_float("blk.0.ssm_a", (inner_size, state_size), negative=True)
        add_float("blk.0.ssm_d", (inner_size,))
    else:
        conv_dim = inner_size + 2 * groups * state_size
        projection_size = 2 * inner_size + 2 * groups * state_size + dt_rank_or_heads
        add_projection("blk.0.ssm_in.weight", (projection_size, hidden_size))
        conv_width = conv_kernel + 1 if malformed_conv else conv_kernel
        add_float("blk.0.ssm_conv1d.weight", (conv_dim, conv_width))
        add_float("blk.0.ssm_conv1d.bias", (conv_dim,))
        add_float("blk.0.ssm_dt.bias", (dt_rank_or_heads,))
        add_float("blk.0.ssm_a", (dt_rank_or_heads, 1), negative=True)
        add_float("blk.0.ssm_d", (dt_rank_or_heads, 1))
        add_float("blk.0.ssm_norm.weight", (inner_size,))
    add_projection("blk.0.ssm_out.weight", (hidden_size, inner_size))
    if auxiliary_scale:
        add_float("blk.0.ssm_in.scale", (1,))
    if malformed_suffix:
        add_float("blk.0.ssm_a.weight", (dt_rank_or_heads,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_falcon_h1_gguf(
    path: Path,
    *,
    quantized: bool,
    omit: str | None = None,
    biases: bool = False,
    partial_bias: bool = False,
    invalid_decay: bool = False,
) -> None:
    """Write a complete one-layer Falcon-H1 GGUF at tiny dimensions."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 32
    intermediate = 64
    vocab = 64
    attn_heads = 4
    kv_heads = 2
    head_dim = 8
    ssm_inner = 32
    ssm_heads = 4
    groups = 1
    state = 8
    kernel = 4
    conv_dim = ssm_inner + 2 * groups * state
    projection_size = 2 * ssm_inner + 2 * groups * state + ssm_heads
    rng = np.random.default_rng(606)

    writer = GGUFWriter(str(path), "falcon-h1")
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(1)
    writer.add_head_count(attn_heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10_000.0)
    writer.add_vocab_size(vocab)
    writer.add_ssm_conv_kernel(kernel)
    writer.add_ssm_inner_size(ssm_inner)
    writer.add_ssm_state_size(state)
    writer.add_ssm_time_step_rank(ssm_heads)
    writer.add_ssm_group_count(groups)

    def add_float(name: str, shape: tuple[int, ...], *, negative: bool = False) -> None:
        if name == omit:
            return
        values = rng.normal(0.0, 0.05, size=shape).astype(np.float32)
        if negative:
            values = -np.exp(values)
        writer.add_tensor(name, values)

    def add_q4(name: str, shape: tuple[int, int]) -> None:
        if name == omit:
            return
        rows, columns = shape
        assert columns % 32 == 0
        raw = np.zeros((rows, columns // 32 * 18), dtype=np.uint8)
        for row in range(rows):
            for block in range(columns // 32):
                offset = block * 18
                raw[row, offset : offset + 2] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[row, offset + 2 : offset + 18] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    projection = add_q4 if quantized else add_float
    add_float("token_embd.weight", (vocab, hidden))
    add_float("output_norm.weight", (hidden,))
    projection("output.weight", (vocab, hidden))
    add_float("blk.0.attn_norm.weight", (hidden,))
    projection("blk.0.attn_q.weight", (hidden, hidden))
    projection("blk.0.attn_k.weight", (kv_heads * head_dim, hidden))
    projection("blk.0.attn_v.weight", (kv_heads * head_dim, hidden))
    projection("blk.0.attn_output.weight", (hidden, hidden))
    if biases or partial_bias:
        add_float("blk.0.attn_q.bias", (hidden,))
    if biases:
        add_float("blk.0.attn_k.bias", (kv_heads * head_dim,))
        add_float("blk.0.attn_v.bias", (kv_heads * head_dim,))
        add_float("blk.0.attn_output.bias", (hidden,))
    add_float("blk.0.ffn_norm.weight", (hidden,))
    projection("blk.0.ffn_gate.weight", (intermediate, hidden))
    projection("blk.0.ffn_up.weight", (intermediate, hidden))
    projection("blk.0.ffn_down.weight", (hidden, intermediate))
    if biases:
        add_float("blk.0.ffn_gate.bias", (intermediate,))
        add_float("blk.0.ffn_up.bias", (intermediate,))
        add_float("blk.0.ffn_down.bias", (hidden,))
    # Recurrent tensors are deliberately float even for a quantized source.
    add_float("blk.0.ssm_in.weight", (projection_size, hidden))
    add_float("blk.0.ssm_conv1d.weight", (conv_dim, kernel))
    add_float("blk.0.ssm_conv1d.bias", (conv_dim,))
    add_float("blk.0.ssm_dt.bias", (ssm_heads,))
    add_float("blk.0.ssm_a", (ssm_heads, 1), negative=not invalid_decay)
    add_float("blk.0.ssm_d", (ssm_heads, 1))
    add_float("blk.0.ssm_out.weight", (hidden, ssm_inner))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_plamo2_gguf(
    path: Path,
    *,
    quantized: bool,
    omit: str | None = None,
    extra: str | None = None,
    invalid_decay: bool = False,
    group_count: int = 0,
    epsilon: float = 1e-6,
    activation: str | None = None,
    predefined_state: bool = False,
    head_counts: list[int] | None = None,
    kv_head_counts: list[int] | None = None,
    legacy_scalar_heads: bool = False,
    quantized_embedding: bool = False,
    include_output: bool = False,
    rope_theta: float = 10_000.0,
) -> None:
    """Write a complete tiny alternating PLaMo2 GGUF."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 32
    intermediate = 64
    vocab = 64
    heads = 4
    kv_heads = 2
    head_dim = 8
    inner = 32
    ssm_heads = 4
    state = 4
    kernel = 4
    dt_rank = 64
    rng = np.random.default_rng(607)

    writer = GGUFWriter(str(path), "plamo2")
    writer.add_context_length(32)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(2)
    writer.add_head_count(heads if legacy_scalar_heads else head_counts or [0, heads])
    writer.add_head_count_kv(
        kv_heads if legacy_scalar_heads else kv_head_counts or [0, kv_heads]
    )
    writer.add_layer_norm_rms_eps(epsilon)
    writer.add_rope_freq_base(rope_theta)
    writer.add_vocab_size(vocab)
    writer.add_ssm_conv_kernel(kernel)
    writer.add_ssm_inner_size(inner)
    writer.add_ssm_state_size(state)
    writer.add_ssm_time_step_rank(ssm_heads)
    writer.add_ssm_group_count(group_count)
    if activation is not None:
        writer.add_string("plamo2.feed_forward.activation", activation)
    if predefined_state:
        writer.add_bool("plamo2.ssm.use_predefined_initial_state", True)

    def add_float(name: str, shape: tuple[int, ...], *, negative: bool = False) -> None:
        if name == omit:
            return
        values = rng.normal(0.0, 0.03, size=shape).astype(np.float32)
        if negative:
            values = -np.exp(values)
        writer.add_tensor(name, values)

    def add_q4(name: str, shape: tuple[int, int]) -> None:
        if name == omit:
            return
        rows, columns = shape
        assert columns % 32 == 0
        raw = np.zeros((rows, columns // 32 * 18), dtype=np.uint8)
        for row in range(rows):
            for block in range(columns // 32):
                offset = block * 18
                raw[row, offset : offset + 2] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[row, offset + 2 : offset + 18] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    projection = add_q4 if quantized else add_float
    (add_q4 if quantized_embedding else add_float)("token_embd.weight", (vocab, hidden))
    if include_output:
        projection("output.weight", (vocab, hidden))
    add_float("output_norm.weight", (hidden,))
    for layer in range(2):
        prefix = f"blk.{layer}."
        add_float(prefix + "attn_norm.weight", (hidden,))
        add_float(prefix + "post_attention_norm", (hidden,))
        add_float(prefix + "ffn_norm.weight", (hidden,))
        add_float(prefix + "post_ffw_norm", (hidden,))
        projection(prefix + "ffn_up.weight", (2 * intermediate, hidden))
        projection(prefix + "ffn_down.weight", (hidden, intermediate))
    projection(
        "blk.0.ssm_in.weight",
        (2 * inner, hidden),
    )
    add_float("blk.0.ssm_conv1d.weight", (inner, kernel))
    projection("blk.0.ssm_x.weight", (2 * state + dt_rank, inner))
    add_float("blk.0.ssm_dt.weight", (ssm_heads, dt_rank))
    add_float("blk.0.ssm_dt.bias", (ssm_heads,))
    add_float("blk.0.ssm_a", (ssm_heads,), negative=not invalid_decay)
    add_float("blk.0.ssm_d", (ssm_heads,))
    add_float("blk.0.ssm_dt_norm", (dt_rank,))
    add_float("blk.0.ssm_b_norm", (state,))
    add_float("blk.0.ssm_c_norm", (state,))
    projection("blk.0.ssm_out.weight", (hidden, inner))
    projection(
        "blk.1.attn_qkv.weight",
        (heads * head_dim + 2 * kv_heads * head_dim, hidden),
    )
    add_float("blk.1.attn_q_norm.weight", (heads, head_dim))
    add_float("blk.1.attn_k_norm.weight", (kv_heads, head_dim))
    projection("blk.1.attn_output.weight", (hidden, heads * head_dim))
    if extra is not None:
        add_float(extra, (1,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_jamba_gguf(
    path: Path,
    *,
    quantized: bool,
    omit: str | None = None,
    extra: str | None = None,
    expert_count: int = 2,
    expert_used_count: int = 1,
    malformed_shape: str | None = None,
    invalid_decay: bool = False,
) -> None:
    """Write a tiny mixed Jamba GGUF with one dense and one routed layer."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 32
    intermediate = 64
    vocab = 64
    heads = 4
    kv_heads = 2
    inner = 64
    state = 4
    rank = 2
    kernel = 4
    rng = np.random.default_rng(612)

    writer = GGUFWriter(str(path), "jamba")
    writer.add_context_length(32)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(2)
    writer.add_head_count(heads)
    writer.add_head_count_kv([0, kv_heads])
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_vocab_size(vocab)
    writer.add_ssm_conv_kernel(kernel)
    writer.add_ssm_inner_size(inner)
    writer.add_ssm_state_size(state)
    writer.add_ssm_time_step_rank(rank)
    writer.add_expert_count(expert_count)
    writer.add_expert_used_count(expert_used_count)
    writer.add_string("jamba.feed_forward.activation", "silu")

    def adjusted_shape(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
        if name == malformed_shape:
            return (*shape[:-1], shape[-1] + 1)
        return shape

    def add_float(
        name: str,
        shape: tuple[int, ...],
        *,
        expert_order: bool = False,
        negative: bool = False,
    ) -> None:
        if name == omit:
            return
        shape = adjusted_shape(name, shape)
        values = rng.normal(0.0, 0.03, size=shape).astype(np.float32)
        if negative:
            values = -np.exp(values)
            if invalid_decay:
                values.flat[0] = -np.inf
        if expert_order:
            for expert in range(shape[0]):
                values[expert].fill(expert + 1)
        writer.add_tensor(name, values)

    def add_q4(name: str, shape: tuple[int, ...]) -> None:
        if name == omit:
            return
        shape = adjusted_shape(name, shape)
        assert shape[-1] % 32 == 0
        byte_shape = (*shape[:-1], shape[-1] // 32 * 18)
        raw = np.zeros(byte_shape, dtype=np.uint8)
        for index in np.ndindex(shape[:-1]):
            for block in range(shape[-1] // 32):
                offset = block * 18
                raw[(*index, slice(offset, offset + 2))] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[(*index, slice(offset + 2, offset + 18))] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    projection = add_q4 if quantized else add_float
    add_float("token_embd.weight", (vocab, hidden))
    add_float("output_norm.weight", (hidden,))
    for layer in range(2):
        prefix = f"blk.{layer}."
        add_float(prefix + "attn_norm.weight", (hidden,))
        add_float(prefix + "ffn_norm.weight", (hidden,))

    projection("blk.0.ffn_gate.weight", (intermediate, hidden))
    projection("blk.0.ffn_up.weight", (intermediate, hidden))
    projection("blk.0.ffn_down.weight", (hidden, intermediate))
    (add_q4 if quantized else add_float)("blk.0.ssm_in.weight", (2 * inner, hidden))
    add_float("blk.0.ssm_conv1d.weight", (inner, kernel))
    add_float("blk.0.ssm_conv1d.bias", (inner,))
    (add_q4 if quantized else add_float)(
        "blk.0.ssm_x.weight",
        (rank + 2 * state, inner),
    )
    add_float("blk.0.ssm_dt_norm.weight", (rank,))
    add_float("blk.0.ssm_dt.weight", (inner, rank))
    add_float("blk.0.ssm_dt.bias", (inner,))
    add_float("blk.0.ssm_b_norm.weight", (state,))
    add_float("blk.0.ssm_c_norm.weight", (state,))
    add_float("blk.0.ssm_a", (inner, state), negative=True)
    add_float("blk.0.ssm_d", (inner,))
    (add_q4 if quantized else add_float)("blk.0.ssm_out.weight", (hidden, inner))

    projection("blk.1.attn_q.weight", (hidden, hidden))
    projection("blk.1.attn_k.weight", (kv_heads * hidden // heads, hidden))
    projection("blk.1.attn_v.weight", (kv_heads * hidden // heads, hidden))
    projection("blk.1.attn_output.weight", (hidden, hidden))
    add_float("blk.1.ffn_gate_inp.weight", (expert_count, hidden))
    if quantized:
        projection(
            "blk.1.ffn_gate_exps.weight",
            (expert_count, intermediate, hidden),
        )
        projection(
            "blk.1.ffn_up_exps.weight",
            (expert_count, intermediate, hidden),
        )
        projection(
            "blk.1.ffn_down_exps.weight",
            (expert_count, hidden, intermediate),
        )
    else:
        add_float(
            "blk.1.ffn_gate_exps.weight",
            (expert_count, intermediate, hidden),
            expert_order=True,
        )
        add_float(
            "blk.1.ffn_up_exps.weight",
            (expert_count, intermediate, hidden),
            expert_order=True,
        )
        add_float(
            "blk.1.ffn_down_exps.weight",
            (expert_count, hidden, intermediate),
            expert_order=True,
        )
    if extra is not None:
        add_float(extra, (1,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_granitehybrid_moe_gguf(
    path: Path,
    *,
    quantized: bool,
    omit: str | None = None,
    extra: str | None = None,
    expert_count: int = 2,
    expert_used_count: int = 1,
    shared_width: int = 16,
    attention_biases: bool = True,
    malformed_shape: str | None = None,
) -> None:
    """Write a tiny mixed GraniteHybrid GGUF with routed and shared experts."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 32
    expert_width = 32
    vocab = 64
    heads = 4
    kv_heads = 2
    inner = 64
    ssm_heads = 4
    groups = 1
    state = 4
    kernel = 4
    rng = np.random.default_rng(614)

    writer = GGUFWriter(str(path), "granitehybrid")
    writer.add_context_length(32)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(expert_width)
    writer.add_block_count(2)
    writer.add_head_count(heads)
    writer.add_head_count_kv([0, kv_heads])
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(vocab)
    writer.add_ssm_conv_kernel(kernel)
    writer.add_ssm_inner_size(inner)
    writer.add_ssm_state_size(state)
    writer.add_ssm_time_step_rank(ssm_heads)
    writer.add_ssm_group_count(groups)
    writer.add_expert_count(expert_count)
    writer.add_expert_used_count(expert_used_count)
    if shared_width:
        writer.add_expert_shared_feed_forward_length(shared_width)
    writer.add_embedding_scale(12.0)
    writer.add_residual_scale(0.5)
    writer.add_attention_scale(0.125)
    writer.add_logit_scale(16.0)
    writer.add_bool("granitehybrid.rope.scaling.finetuned", False)
    writer.add_string("granitehybrid.feed_forward.activation", "silu")

    def adjusted_shape(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
        if name == malformed_shape:
            return (*shape[:-1], shape[-1] + 1)
        return shape

    def add_float(
        name: str,
        shape: tuple[int, ...],
        *,
        fill: float | None = None,
        expert_base: float | None = None,
        negative: bool = False,
    ) -> None:
        if name == omit:
            return
        shape = adjusted_shape(name, shape)
        values = rng.normal(0.0, 0.03, size=shape).astype(np.float32)
        if fill is not None:
            values.fill(fill)
        if expert_base is not None:
            for expert in range(shape[0]):
                values[expert].fill(expert_base + expert)
        if negative:
            values = -np.exp(values)
        writer.add_tensor(name, values)

    def add_q4(name: str, shape: tuple[int, ...]) -> None:
        if name == omit:
            return
        shape = adjusted_shape(name, shape)
        assert shape[-1] % 32 == 0
        byte_shape = (*shape[:-1], shape[-1] // 32 * 18)
        raw = np.zeros(byte_shape, dtype=np.uint8)
        for index in np.ndindex(shape[:-1]):
            for block in range(shape[-1] // 32):
                offset = block * 18
                raw[(*index, slice(offset, offset + 2))] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[(*index, slice(offset + 2, offset + 18))] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    add_float("token_embd.weight", (vocab, hidden))
    add_float("output_norm.weight", (hidden,))
    for layer in range(2):
        prefix = f"blk.{layer}."
        add_float(prefix + "attn_norm.weight", (hidden,))
        add_float(prefix + "ffn_norm.weight", (hidden,))
        add_float(prefix + "ffn_gate_inp.weight", (expert_count, hidden))
        expert_projection = add_q4 if quantized else add_float
        if quantized:
            expert_projection(
                prefix + "ffn_gate_exps.weight",
                (expert_count, expert_width, hidden),
            )
            expert_projection(
                prefix + "ffn_up_exps.weight",
                (expert_count, expert_width, hidden),
            )
            expert_projection(
                prefix + "ffn_down_exps.weight",
                (expert_count, hidden, expert_width),
            )
        else:
            add_float(
                prefix + "ffn_gate_exps.weight",
                (expert_count, expert_width, hidden),
                expert_base=11.0,
            )
            add_float(
                prefix + "ffn_up_exps.weight",
                (expert_count, expert_width, hidden),
                expert_base=21.0,
            )
            add_float(
                prefix + "ffn_down_exps.weight",
                (expert_count, hidden, expert_width),
                expert_base=31.0,
            )
        if shared_width:
            add_float(prefix + "ffn_gate_shexp.weight", (shared_width, hidden))
            add_float(prefix + "ffn_up_shexp.weight", (shared_width, hidden))
            add_float(prefix + "ffn_down_shexp.weight", (hidden, shared_width))

    add_float(
        "blk.0.ssm_in.weight",
        (2 * inner + 2 * groups * state + ssm_heads, hidden),
    )
    add_float("blk.0.ssm_conv1d.weight", (inner + 2 * groups * state, kernel))
    add_float("blk.0.ssm_conv1d.bias", (inner + 2 * groups * state,))
    add_float("blk.0.ssm_dt.bias", (ssm_heads,))
    add_float("blk.0.ssm_a", (ssm_heads, 1), negative=True)
    add_float("blk.0.ssm_d", (ssm_heads, 1))
    add_float("blk.0.ssm_norm.weight", (groups, inner // groups))
    add_float("blk.0.ssm_out.weight", (hidden, inner))

    head_dim = hidden // heads
    add_float("blk.1.attn_q.weight", (heads * head_dim, hidden))
    add_float("blk.1.attn_k.weight", (kv_heads * head_dim, hidden))
    add_float("blk.1.attn_v.weight", (kv_heads * head_dim, hidden))
    if attention_biases:
        add_float("blk.1.attn_q.bias", (heads * head_dim,))
        add_float("blk.1.attn_k.bias", (kv_heads * head_dim,))
        add_float("blk.1.attn_v.bias", (kv_heads * head_dim,))
    add_float("blk.1.attn_output.weight", (hidden, heads * head_dim))
    if extra is not None:
        add_float(extra, (1,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_minimax_gguf(
    path: Path,
    *,
    quantized: bool,
    quantized_embedding: bool = False,
    omit: str | None = None,
    extra: str | None = None,
    malformed_shape: str | None = None,
    recurrent_layers: list[bool] | None = None,
    norm_eps: float = 1e-5,
    rope_freq_base: float = 10_000_000.0,
) -> None:
    """Write a tiny MiniMax-01 GGUF with one Lightning and one full-attention layer."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 64
    intermediate = 32
    vocab = 64
    heads = 4
    kv_heads = 2
    head_dim = 16
    experts = 2
    rng = np.random.default_rng(601)

    writer = GGUFWriter(str(path), "minimax-01")
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(2)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(norm_eps)
    writer.add_rope_freq_base(rope_freq_base)
    writer.add_rope_dimension_count(8)
    writer.add_expert_count(experts)
    writer.add_expert_used_count(1)
    writer.add_residual_scale(3.5565588200778455)
    writer.add_vocab_size(vocab)
    writer.add_array(
        "minimax-01.attention.recurrent_layers",
        recurrent_layers if recurrent_layers is not None else [True, False],
    )

    def shape_for(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
        if name == malformed_shape:
            return (*shape[:-1], shape[-1] + 1)
        return shape

    def add_float(
        name: str,
        shape: tuple[int, ...],
        *,
        expert_base: float | None = None,
    ) -> None:
        if name == omit:
            return
        shape = shape_for(name, shape)
        values = rng.normal(0.0, 0.02, shape).astype(np.float32)
        if expert_base is not None:
            for expert in range(shape[0]):
                values[expert].fill(expert_base + expert)
        writer.add_tensor(name, values)

    def add_q4(name: str, shape: tuple[int, ...]) -> None:
        if name == omit:
            return
        shape = shape_for(name, shape)
        assert shape[-1] % 32 == 0
        raw = np.zeros((*shape[:-1], shape[-1] // 32 * 18), dtype=np.uint8)
        for index in np.ndindex(shape[:-1]):
            for block in range(shape[-1] // 32):
                offset = block * 18
                raw[(*index, slice(offset, offset + 2))] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[(*index, slice(offset + 2, offset + 18))] = rng.integers(
                    0, 256, 16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    (add_q4 if quantized_embedding else add_float)("token_embd.weight", (vocab, hidden))
    add_float("output_norm.weight", (hidden,))
    add_float("output.weight", (vocab, hidden))
    projection = add_q4 if quantized else add_float
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    for layer in range(2):
        prefix = f"blk.{layer}."
        add_float(prefix + "attn_norm.weight", (hidden,))
        add_float(prefix + "ffn_norm.weight", (hidden,))
        add_float(prefix + "ffn_gate_inp.weight", (experts, hidden))
        if quantized:
            projection(prefix + "ffn_gate_exps.weight", (experts, intermediate, hidden))
            projection(prefix + "ffn_up_exps.weight", (experts, intermediate, hidden))
            projection(prefix + "ffn_down_exps.weight", (experts, hidden, intermediate))
        else:
            add_float(
                prefix + "ffn_gate_exps.weight",
                (experts, intermediate, hidden),
                expert_base=11.0,
            )
            add_float(
                prefix + "ffn_up_exps.weight",
                (experts, intermediate, hidden),
                expert_base=21.0,
            )
            add_float(
                prefix + "ffn_down_exps.weight",
                (experts, hidden, intermediate),
                expert_base=31.0,
            )
        projection(prefix + "attn_output.weight", (hidden, q_width))
        if layer == 0:
            projection(prefix + "attn_qkv.weight", (3 * q_width, hidden))
            projection(prefix + "attn_gate.weight", (q_width, hidden))
            add_float(prefix + "attn_norm_2.weight", (q_width,))
        else:
            projection(prefix + "attn_q.weight", (q_width, hidden))
            projection(prefix + "attn_k.weight", (kv_width, hidden))
            projection(prefix + "attn_v.weight", (kv_width, hidden))
    if extra is not None:
        add_float(extra, (1,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_nemotron_h_moe_gguf(
    path: Path,
    *,
    quantized: bool,
    latent: bool = False,
    omit: str | None = None,
    extra: str | None = None,
    malformed_shape: str | None = None,
    mtp: bool = False,
    quantized_only: str | None = None,
) -> None:
    """Write a tiny exact Nemotron-H backbone covering all four layer kinds."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 32
    dense_width = 64
    expert_width = 64
    shared_width = 64
    latent_width = 32
    vocab = 64
    heads = 2
    kv_heads = 1
    ssm_heads = 4
    head_dim = hidden // heads
    inner = 64
    groups = 1
    state = 4
    kernel = 4
    experts = 2
    rng = np.random.default_rng(613)

    writer = GGUFWriter(str(path), "nemotron_h_moe")
    writer.add_context_length(32)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(
        [0, dense_width, 0, dense_width] + ([dense_width] if mtp else [])
    )
    writer.add_block_count(5 if mtp else 4)
    writer.add_head_count([0, 0, heads, 0] + ([heads] if mtp else []))
    writer.add_head_count_kv([0, 0, kv_heads, 0] + ([kv_heads] if mtp else []))
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_vocab_size(vocab)
    writer.add_ssm_conv_kernel(kernel)
    writer.add_ssm_inner_size(inner)
    writer.add_ssm_state_size(state)
    writer.add_ssm_time_step_rank(ssm_heads)
    writer.add_ssm_group_count(groups)
    writer.add_expert_count(experts)
    writer.add_expert_used_count(1)
    writer.add_expert_feed_forward_length(expert_width)
    writer.add_expert_shared_count(1)
    writer.add_expert_shared_feed_forward_length(shared_width)
    writer.add_expert_weights_norm(True)
    writer.add_expert_weights_scale(2.5)
    if latent:
        writer.add_uint32("nemotron_h_moe.moe_latent_size", latent_width)
    if mtp:
        writer.add_uint32("nemotron_h_moe.nextn_predict_layers", 1)

    def adjusted_shape(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
        if name == malformed_shape:
            return (*shape[:-1], shape[-1] + 1)
        return shape

    def add_float(
        name: str,
        shape: tuple[int, ...],
        *,
        expert_order: bool = False,
        negative: bool = False,
    ) -> None:
        if name == omit:
            return
        shape = adjusted_shape(name, shape)
        values = rng.normal(0.0, 0.03, size=shape).astype(np.float32)
        if negative:
            values = -np.exp(values)
        if expert_order:
            for expert in range(shape[0]):
                values[expert].fill(expert + 1)
        writer.add_tensor(name, values)

    def add_q4(name: str, shape: tuple[int, ...]) -> None:
        if name == omit:
            return
        shape = adjusted_shape(name, shape)
        assert shape[-1] % 32 == 0
        byte_shape = (*shape[:-1], shape[-1] // 32 * 18)
        raw = np.zeros(byte_shape, dtype=np.uint8)
        for index in np.ndindex(shape[:-1]):
            for block in range(shape[-1] // 32):
                offset = block * 18
                raw[(*index, slice(offset, offset + 2))] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[(*index, slice(offset + 2, offset + 18))] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    def projection(name: str, shape: tuple[int, ...]) -> None:
        if quantized and (quantized_only is None or name == quantized_only):
            add_q4(name, shape)
        else:
            add_float(name, shape)

    add_float("token_embd.weight", (vocab, hidden))
    add_float("output_norm.weight", (hidden,))
    projection("output.weight", (vocab, hidden))
    for layer in range(4):
        add_float(f"blk.{layer}.attn_norm.weight", (hidden,))

    conv_width = inner + 2 * groups * state
    projection(
        "blk.0.ssm_in.weight",
        (2 * inner + 2 * groups * state + ssm_heads, hidden),
    )
    add_float("blk.0.ssm_conv1d.weight", (conv_width, kernel))
    add_float("blk.0.ssm_conv1d.bias", (conv_width,))
    add_float("blk.0.ssm_dt.bias", (ssm_heads,))
    add_float("blk.0.ssm_a", (ssm_heads, 1), negative=True)
    add_float("blk.0.ssm_d", (ssm_heads, 1))
    add_float("blk.0.ssm_norm.weight", (groups, inner // groups))
    projection("blk.0.ssm_out.weight", (hidden, inner))

    expert_input = latent_width if latent else hidden
    add_float("blk.1.ffn_gate_inp.weight", (experts, hidden))
    add_float("blk.1.exp_probs_b.bias", (experts,))
    if quantized:
        projection("blk.1.ffn_up_exps.weight", (experts, expert_width, expert_input))
        projection("blk.1.ffn_down_exps.weight", (experts, expert_input, expert_width))
    else:
        add_float(
            "blk.1.ffn_up_exps.weight",
            (experts, expert_width, expert_input),
            expert_order=True,
        )
        add_float(
            "blk.1.ffn_down_exps.weight",
            (experts, expert_input, expert_width),
            expert_order=True,
        )
    projection("blk.1.ffn_up_shexp.weight", (shared_width, hidden))
    projection("blk.1.ffn_down_shexp.weight", (hidden, shared_width))
    if latent:
        projection("blk.1.ffn_latent_down.weight", (latent_width, hidden))
        projection("blk.1.ffn_latent_up.weight", (hidden, latent_width))

    projection("blk.2.attn_q.weight", (heads * head_dim, hidden))
    projection("blk.2.attn_k.weight", (kv_heads * head_dim, hidden))
    projection("blk.2.attn_v.weight", (kv_heads * head_dim, hidden))
    projection("blk.2.attn_output.weight", (hidden, heads * head_dim))

    projection("blk.3.ffn_up.weight", (dense_width, hidden))
    projection("blk.3.ffn_down.weight", (hidden, dense_width))
    if mtp:
        projection("blk.4.nextn.eh_proj.weight", (hidden, 2 * hidden))
    if extra is not None:
        add_float(extra, (1,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_lfm2_gguf(path: Path, *, quantized: bool) -> None:
    """Write a tiny two-layer LFM2 GGUF with one conv and one attention layer."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 32
    intermediate = 64
    vocab = 64
    heads = 4
    kv_heads = 2
    head_dim = hidden // heads
    kernel = 3
    rng = np.random.default_rng(23)

    writer = GGUFWriter(str(path), "lfm2")
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(2)
    writer.add_head_count(heads)
    writer.add_head_count_kv([0, kv_heads])
    writer.add_rope_freq_base(10_000.0)
    writer.add_rope_dimension_count(head_dim)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(vocab)
    writer.add_uint32("lfm2.shortconv.l_cache", kernel)

    def add_float(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, rng.normal(0, 0.03, shape).astype(np.float32))

    def add_q4(name: str, shape: tuple[int, int]) -> None:
        rows, columns = shape
        assert columns % 32 == 0
        raw = np.zeros((rows, columns // 32 * 18), dtype=np.uint8)
        for row in range(rows):
            for block in range(columns // 32):
                offset = block * 18
                raw[row, offset : offset + 2] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[row, offset + 2 : offset + 18] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    add_projection = add_q4 if quantized else add_float
    add_float("token_embd.weight", (vocab, hidden))
    add_float("token_embd_norm.weight", (hidden,))
    for layer in range(2):
        prefix = f"blk.{layer}."
        add_float(prefix + "attn_norm.weight", (hidden,))
        add_float(prefix + "ffn_norm.weight", (hidden,))
        add_projection(prefix + "ffn_gate.weight", (intermediate, hidden))
        add_projection(prefix + "ffn_up.weight", (intermediate, hidden))
        add_projection(prefix + "ffn_down.weight", (hidden, intermediate))

    add_float("blk.0.shortconv.conv.weight", (hidden, kernel))
    add_projection("blk.0.shortconv.in_proj.weight", (3 * hidden, hidden))
    add_projection("blk.0.shortconv.out_proj.weight", (hidden, hidden))

    add_projection("blk.1.attn_q.weight", (heads * head_dim, hidden))
    add_projection("blk.1.attn_k.weight", (kv_heads * head_dim, hidden))
    add_projection("blk.1.attn_v.weight", (kv_heads * head_dim, hidden))
    add_projection("blk.1.attn_output.weight", (hidden, heads * head_dim))
    add_float("blk.1.attn_q_norm.weight", (head_dim,))
    add_float("blk.1.attn_k_norm.weight", (head_dim,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_lfm2moe_gguf(path: Path, *, quantized: bool) -> None:
    """Write a tiny LFM2MoE GGUF with dense-conv and routed-attention layers."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 32
    intermediate = 64
    expert_intermediate = 32
    experts = 4
    vocab = 64
    heads = 4
    kv_heads = 2
    head_dim = hidden // heads
    kernel = 3
    rng = np.random.default_rng(31)

    writer = GGUFWriter(str(path), "lfm2moe")
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(2)
    writer.add_head_count(heads)
    writer.add_head_count_kv([0, kv_heads])
    writer.add_rope_freq_base(10_000.0)
    writer.add_rope_dimension_count(head_dim)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(vocab)
    writer.add_expert_count(experts)
    writer.add_expert_used_count(2)
    writer.add_expert_feed_forward_length(expert_intermediate)
    writer.add_uint32("lfm2moe.shortconv.l_cache", kernel)
    writer.add_uint32("lfm2moe.leading_dense_block_count", 1)
    writer.add_uint32("lfm2moe.expert_gating_func", 2)

    def add_float(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, rng.normal(0, 0.03, shape).astype(np.float32))

    def add_q4(name: str, shape: tuple[int, ...]) -> None:
        rows = int(np.prod(shape[:-1]))
        columns = shape[-1]
        assert columns % 32 == 0
        byte_shape = (*shape[:-1], columns // 32 * 18)
        raw = np.zeros((rows, byte_shape[-1]), dtype=np.uint8)
        for row in range(rows):
            for block in range(columns // 32):
                offset = block * 18
                raw[row, offset : offset + 2] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[row, offset + 2 : offset + 18] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(
            name,
            raw.reshape(byte_shape),
            raw_dtype=GGMLQuantizationType.Q4_0,
        )

    add_projection = add_q4 if quantized else add_float
    add_float("token_embd.weight", (vocab, hidden))
    add_float("token_embd_norm.weight", (hidden,))
    for layer in range(2):
        prefix = f"blk.{layer}."
        add_float(prefix + "attn_norm.weight", (hidden,))
        add_float(prefix + "ffn_norm.weight", (hidden,))

    add_projection("blk.0.ffn_gate.weight", (intermediate, hidden))
    add_projection("blk.0.ffn_up.weight", (intermediate, hidden))
    add_projection("blk.0.ffn_down.weight", (hidden, intermediate))
    add_float("blk.0.shortconv.conv.weight", (hidden, kernel))
    add_projection("blk.0.shortconv.in_proj.weight", (3 * hidden, hidden))
    add_projection("blk.0.shortconv.out_proj.weight", (hidden, hidden))

    add_float("blk.1.ffn_gate_inp.weight", (experts, hidden))
    add_projection(
        "blk.1.ffn_gate_exps.weight",
        (experts, expert_intermediate, hidden),
    )
    add_projection(
        "blk.1.ffn_up_exps.weight",
        (experts, expert_intermediate, hidden),
    )
    add_projection(
        "blk.1.ffn_down_exps.weight",
        (experts, hidden, expert_intermediate),
    )
    add_float("blk.1.exp_probs_b.bias", (experts,))
    add_projection("blk.1.attn_q.weight", (heads * head_dim, hidden))
    add_projection("blk.1.attn_k.weight", (kv_heads * head_dim, hidden))
    add_projection("blk.1.attn_v.weight", (kv_heads * head_dim, hidden))
    add_projection("blk.1.attn_output.weight", (hidden, heads * head_dim))
    add_float("blk.1.attn_q_norm.weight", (head_dim,))
    add_float("blk.1.attn_k_norm.weight", (head_dim,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_qwen35_gguf(
    path: Path,
    *,
    quantized: bool,
    inner_size: int = 256,
    quantized_embedding: bool = False,
) -> None:
    """Write a tiny Qwen3.5 GGUF with three DeltaNet layers and one attention layer."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 32
    intermediate = 64
    vocab = 64
    heads = 4
    kv_heads = 2
    head_dim = 8
    key_heads = 2
    value_heads = 4
    state_size = inner_size // value_heads
    kernel = 3
    key_dim = key_heads * state_size
    value_dim = inner_size
    conv_dim = 2 * key_dim + value_dim
    rng = np.random.default_rng(29)

    writer = GGUFWriter(str(path), "qwen35")
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(4)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_rope_freq_base(10_000.0)
    writer.add_rope_dimension_count(head_dim)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(vocab)
    writer.add_ssm_conv_kernel(kernel)
    writer.add_ssm_inner_size(inner_size)
    writer.add_ssm_state_size(state_size)
    writer.add_ssm_time_step_rank(value_heads)
    writer.add_ssm_group_count(key_heads)
    writer.add_uint32("qwen35.full_attention_interval", 4)
    writer.add_array("qwen35.rope.dimension_sections", [2, 2, 0, 0])

    def add_float(name: str, shape: tuple[int, ...], *, negative_exp: bool = False) -> None:
        values = rng.normal(0, 0.03, shape).astype(np.float32)
        if negative_exp:
            values = -np.exp(values)
        writer.add_tensor(name, values)

    def add_q4(name: str, shape: tuple[int, int]) -> None:
        rows, columns = shape
        assert columns % 32 == 0
        raw = np.zeros((rows, columns // 32 * 18), dtype=np.uint8)
        for row in range(rows):
            for block in range(columns // 32):
                offset = block * 18
                raw[row, offset : offset + 2] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[row, offset + 2 : offset + 18] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    add_projection = add_q4 if quantized else add_float
    (add_q4 if quantized_embedding else add_float)("token_embd.weight", (vocab, hidden))
    add_float("output_norm.weight", (hidden,))
    for layer in range(4):
        prefix = f"blk.{layer}."
        add_float(prefix + "attn_norm.weight", (hidden,))
        add_float(prefix + "post_attention_norm.weight", (hidden,))
        add_projection(prefix + "ffn_gate.weight", (intermediate, hidden))
        add_projection(prefix + "ffn_up.weight", (intermediate, hidden))
        add_projection(prefix + "ffn_down.weight", (hidden, intermediate))
        if layer == 3:
            add_projection(prefix + "attn_q.weight", (2 * heads * head_dim, hidden))
            add_projection(prefix + "attn_k.weight", (kv_heads * head_dim, hidden))
            add_projection(prefix + "attn_v.weight", (kv_heads * head_dim, hidden))
            add_projection(prefix + "attn_output.weight", (hidden, heads * head_dim))
            add_float(prefix + "attn_q_norm.weight", (head_dim,))
            add_float(prefix + "attn_k_norm.weight", (head_dim,))
        else:
            add_projection(prefix + "attn_qkv.weight", (conv_dim, hidden))
            add_projection(prefix + "attn_gate.weight", (value_dim, hidden))
            add_float(prefix + "ssm_conv1d.weight", (conv_dim, kernel))
            add_float(prefix + "ssm_dt.bias", (value_heads,))
            add_float(prefix + "ssm_a", (value_heads,), negative_exp=True)
            add_projection(prefix + "ssm_beta.weight", (value_heads, hidden))
            add_projection(prefix + "ssm_alpha.weight", (value_heads, hidden))
            add_float(prefix + "ssm_norm.weight", (inner_size // value_heads,))
            add_projection(prefix + "ssm_out.weight", (hidden, value_dim))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_moe_gguf(
    path: Path,
    architecture: str,
    projection_quantization: str,
    *,
    phi_fused_qkv: bool = False,
    diffusion_fused_qkv: bool = False,
    fused_qkv_float: bool = False,
    quantize_tied_embedding: bool = False,
    expert_scale_suffix: str | None = None,
    malformed_expert_scale: bool = False,
) -> None:
    """Write a one-layer conventional-attention MoE GGUF with exact families."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden_size = 64
    intermediate_size = 128
    expert_size = 64 if architecture in {"qwen2moe", "qwen3moe", "llada-moe", "rnd1"} else 128
    shared_size = 128 if architecture == "qwen2moe" else 32
    num_experts = 4
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    vocab_size = 256

    writer = GGUFWriter(str(path), architecture)
    writer.add_context_length(512)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_block_count(1)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_rope_freq_base(10_000.0)
    writer.add_rope_dimension_count(head_dim)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(vocab_size)
    writer.add_expert_count(num_experts)
    writer.add_expert_used_count(2)
    if architecture in {"llada-moe", "rnd1"}:
        writer.add_uint32("tokenizer.ggml.mask_token_id", vocab_size - 1)
    if architecture == "llada-moe":
        writer.add_bool("diffusion.shift_logits", False)
    if architecture in {"qwen2moe", "qwen3moe"}:
        writer.add_expert_feed_forward_length(expert_size)
    if architecture == "qwen2moe":
        writer.add_expert_shared_feed_forward_length(shared_size)
    if architecture == "granitemoe":
        writer.add_logit_scale(16.0)
        writer.add_embedding_scale(12.0)
        writer.add_residual_scale(0.5)
        writer.add_attention_scale(0.125)
        writer.add_expert_shared_feed_forward_length(shared_size)

    rng = np.random.default_rng(0)

    def add_float(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, rng.normal(size=shape).astype(np.float32))

    def add_q4(name: str, shape: tuple[int, ...]) -> None:
        rows = int(np.prod(shape[:-1]))
        k_in = shape[-1]
        assert k_in % 32 == 0
        byte_shape = (*shape[:-1], (k_in // 32) * 18)
        raw = np.zeros((rows, byte_shape[-1]), dtype=np.uint8)
        for row in range(rows):
            for block in range(k_in // 32):
                offset = block * 18
                raw[row, offset : offset + 2] = np.array(
                    [rng.uniform(0.01, 0.1)], dtype=np.float16
                ).view(np.uint8)
                raw[row, offset + 2 : offset + 18] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(
            name,
            raw.reshape(byte_shape),
            raw_dtype=GGMLQuantizationType.Q4_0,
        )

    def add_mxfp4(name: str, shape: tuple[int, ...]) -> None:
        assert shape[-1] % 32 == 0
        byte_shape = (*shape[:-1], (shape[-1] // 32) * 17)
        writer.add_tensor(
            name,
            np.zeros(byte_shape, dtype=np.uint8),
            raw_dtype=GGMLQuantizationType.MXFP4,
        )

    add_projection = {
        "f32": add_float,
        "mxfp4": add_mxfp4,
        "q4_0": add_q4,
    }[projection_quantization]

    if quantize_tied_embedding:
        assert architecture in {"qwen3moe", "granitemoe"}
        add_q4("token_embd.weight", (vocab_size, hidden_size))
    else:
        add_float("token_embd.weight", (vocab_size, hidden_size))
    if architecture == "phimoe" and phi_fused_qkv:
        add_projection(
            "blk.0.attn_qkv.weight",
            ((num_heads + 2 * num_kv_heads) * head_dim, hidden_size),
        )
    elif architecture in {"llada-moe", "rnd1"} and diffusion_fused_qkv:
        add = add_float if fused_qkv_float else add_projection
        add(
            "blk.0.attn_qkv.weight",
            ((num_heads + 2 * num_kv_heads) * head_dim, hidden_size),
        )
    else:
        add_projection("blk.0.attn_q.weight", (num_heads * head_dim, hidden_size))
        add_projection("blk.0.attn_k.weight", (num_kv_heads * head_dim, hidden_size))
        add_projection("blk.0.attn_v.weight", (num_kv_heads * head_dim, hidden_size))
    add_projection("blk.0.attn_output.weight", (hidden_size, num_heads * head_dim))
    add_float("blk.0.attn_norm.weight", (hidden_size,))
    add_float("blk.0.ffn_norm.weight", (hidden_size,))
    add_float("blk.0.ffn_gate_inp.weight", (num_experts, hidden_size))
    add_projection(
        "blk.0.ffn_gate_exps.weight",
        (num_experts, expert_size, hidden_size),
    )
    add_projection(
        "blk.0.ffn_up_exps.weight",
        (num_experts, expert_size, hidden_size),
    )
    add_projection(
        "blk.0.ffn_down_exps.weight",
        (num_experts, hidden_size, expert_size),
    )
    if expert_scale_suffix is not None:
        assert expert_scale_suffix in {"scale", "input_scale"}
        scale_count = num_experts - 1 if malformed_expert_scale else num_experts
        add_float(f"blk.0.ffn_gate_exps.{expert_scale_suffix}", (scale_count,))

    if architecture == "olmoe":
        add_float("blk.0.attn_q_norm.weight", (num_heads * head_dim,))
        add_float("blk.0.attn_k_norm.weight", (num_kv_heads * head_dim,))
    elif architecture in {"qwen3moe", "llada-moe", "rnd1"}:
        add_float("blk.0.attn_q_norm.weight", (head_dim,))
        add_float("blk.0.attn_k_norm.weight", (head_dim,))
    elif architecture == "qwen2moe":
        add_float("blk.0.ffn_gate_inp_shexp.weight", (hidden_size,))
        add_projection("blk.0.ffn_gate_shexp.weight", (shared_size, hidden_size))
        add_projection("blk.0.ffn_up_shexp.weight", (shared_size, hidden_size))
        add_projection("blk.0.ffn_down_shexp.weight", (hidden_size, shared_size))
    elif architecture == "granitemoe":
        add_projection("blk.0.ffn_gate_shexp.weight", (shared_size, hidden_size))
        add_projection("blk.0.ffn_up_shexp.weight", (shared_size, hidden_size))
        add_projection("blk.0.ffn_down_shexp.weight", (hidden_size, shared_size))
    elif architecture == "phimoe":
        attention_biases = (
            (("attn_qkv", (num_heads + 2 * num_kv_heads) * head_dim),)
            if phi_fused_qkv
            else (
                ("attn_q", num_heads * head_dim),
                ("attn_k", num_kv_heads * head_dim),
                ("attn_v", num_kv_heads * head_dim),
            )
        )
        for name, size in (*attention_biases, ("attn_output", hidden_size)):
            add_float(f"blk.0.{name}.bias", (size,))
        add_float("blk.0.attn_norm.bias", (hidden_size,))
        add_float("blk.0.ffn_norm.bias", (hidden_size,))

    add_float("output_norm.weight", (hidden_size,))
    if architecture == "phimoe":
        add_float("output_norm.bias", (hidden_size,))
    if architecture not in {"qwen3moe", "granitemoe"}:
        add_projection("output.weight", (vocab_size, hidden_size))
        if architecture == "phimoe":
            add_float("output.bias", (vocab_size,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def q4_0_gguf(tmp_path: Path) -> Path:
    """Create a Q4_0 quantized GGUF test file."""
    path = tmp_path / "test_q4_0.gguf"
    _write_quantized_gguf(path)
    return path


@pytest.fixture(params=["f32", "f16", "bf16"])
def float_only_gguf(tmp_path: Path, request) -> Path:
    """Create a GGUF with only unquantized tensor types."""
    float_type = request.param
    path = tmp_path / f"test_{float_type}.gguf"
    _write_quantized_gguf(
        path,
        projection_quantization=float_type,
        float_type=float_type,
    )
    return path


@pytest.fixture
def q4_0_embedding_gguf(tmp_path: Path) -> Path:
    """Create a GGUF with a Q4_0 token embedding."""
    path = tmp_path / "test_q4_0_embedding.gguf"
    _write_quantized_gguf(path, quantize_embedding=True)
    return path


@pytest.fixture
def q4_0_tied_embedding_gguf(tmp_path: Path) -> Path:
    """Create a GGUF with one Q4_0 table shared by embedding and LM head."""
    path = tmp_path / "test_q4_0_tied_embedding.gguf"
    _write_quantized_gguf(path, quantize_embedding=True, tie_embeddings=True)
    return path


@pytest.fixture
def iq4_nl_embedding_gguf(tmp_path: Path) -> Path:
    """Create a native-IQ embedding with ordinary Q4_0 projections."""
    path = tmp_path / "test_iq4_nl_embedding.gguf"
    _write_quantized_gguf(path, embedding_quantization="iq4_nl")
    return path


@pytest.fixture
def q4_0_embedding_q8_head_gguf(tmp_path: Path) -> Path:
    """Create a GGUF with a Q4 embedding and an untied Q8 output head."""
    path = tmp_path / "test_q4_0_embedding_q8_head.gguf"
    _write_quantized_gguf(
        path,
        quantize_embedding=True,
        output_quantization="q8_0",
    )
    return path


@pytest.fixture
def q8_0_projection_q4_head_gguf(tmp_path: Path) -> Path:
    """Create a GGUF whose Q4 output head would require Q8 requantization."""
    path = tmp_path / "test_q8_0_projection_q4_head.gguf"
    _write_quantized_gguf(
        path,
        projection_quantization="q8_0",
        output_quantization="q4_0",
    )
    return path


@pytest.fixture(
    params=[
        ("mxfp4", 32, 17),
        ("iq4_nl", 32, 18),
        ("iq4_xs", 256, 136),
        ("iq3_s", 256, 110),
        ("iq3_xxs", 256, 98),
        ("iq2_xxs", 256, 66),
        ("iq2_xs", 256, 74),
        ("iq2_s", 256, 82),
        ("iq1_s", 256, 50),
        ("iq1_m", 256, 56),
    ]
)
def native_block_gguf(tmp_path: Path, request) -> tuple[Path, str, int, int]:
    """Create a GGUF whose projection weights use a runtime-native block format."""
    format_name, block_elements, block_bytes = request.param
    path = tmp_path / f"test_{format_name}.gguf"
    _write_quantized_gguf(
        path,
        hidden_size=256,
        num_kv_heads=4,
        intermediate_size=256,
        projection_quantization=format_name,
    )
    return path, format_name, block_elements, block_bytes


@pytest.fixture
def mixed_native_q5_q8_gguf(tmp_path: Path) -> Path:
    """Create native IQ projections with a Q5_1 value projection and Q8 embedding."""
    path = tmp_path / "test_mixed_native_q5_q8.gguf"
    _write_quantized_gguf(
        path,
        hidden_size=256,
        num_kv_heads=4,
        intermediate_size=256,
        embedding_quantization="q8_0",
        projection_quantization="iq4_xs",
        value_projection_quantization="q5_1",
    )
    return path


class TestReuseGgufWeights:
    """Tests for mixed GGUF references plus converted ONNX sidecar weights."""

    def test_mixed_save_preserves_ranges_and_runs(self, tmp_path: Path):
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        package.save(str(tmp_path), progress_bar=False)

        verify_gguf_reuse_manifest(tmp_path)
        manifest = json.loads((tmp_path / "gguf-reuse.json").read_text())
        converted = manifest["converted_tensors"]
        assert len(converted) == len(set(converted))
        assert "model.layers.0.self_attn.q_proj.weight" not in converted
        assert "model.embed_tokens.weight" not in converted
        q_route = next(
            route
            for route in manifest["reused_tensors"]
            if route["initializer"] == "model.layers.0.self_attn.q_proj.weight"
        )
        assert q_route["transform"] == "llama_qk_permute"
        reloaded = ModelPackage.load(str(tmp_path))
        initializers = reloaded["model"].graph.initializers
        embedding = initializers["model.embed_tokens.weight"].const_value
        q_proj = initializers["model.layers.0.self_attn.q_proj.weight"].const_value
        assert isinstance(embedding, ir.ExternalTensor)
        assert embedding.location == "model.gguf"
        assert embedding.offset is not None
        assert embedding.length == 256 * 64 * 4
        # Llama Q weights keep their GGUF bytes; ONNX performs the row permutation.
        assert isinstance(q_proj, ir.ExternalTensor)
        assert q_proj.location == "model.gguf"
        assert any(
            node.name == "model.layers.0.self_attn.q_proj.weight.gguf_reuse.Transpose"
            for node in reloaded["model"].graph
        )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(tmp_path / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        feeds = {
            "input_ids": np.zeros((1, 2), dtype=np.int64),
            "attention_mask": np.zeros((1, 2), dtype=np.int64),
            "position_ids": np.zeros((1, 2), dtype=np.int64),
            "past_key_values.0.key": np.zeros((1, 2, 0, 16), dtype=np.float32),
            "past_key_values.0.value": np.zeros((1, 2, 0, 16), dtype=np.float32),
        }
        outputs = session.run(None, feeds)
        assert outputs[0].shape == (1, 2, 256)
        assert np.isfinite(outputs[0]).all()

        reference_dir = tmp_path / "reference"
        build_from_gguf(gguf_path).save(str(reference_dir), progress_bar=False)
        reference_session = ort.InferenceSession(
            str(reference_dir / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        reference_outputs = reference_session.run(None, feeds)
        np.testing.assert_allclose(outputs[0], reference_outputs[0], rtol=1e-5, atol=1e-5)

    def test_native_projection_bytes_are_not_copied_to_sidecar(self, tmp_path: Path):
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(
            gguf_path,
            architecture="qwen2",
            hidden_size=256,
            num_heads=4,
            num_kv_heads=4,
            intermediate_size=256,
            projection_quantization="iq4_nl",
        )
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        package.save(str(tmp_path), progress_bar=False)

        source = GGUFModel(gguf_path)
        offset, length, _ = source.tensor_storage_range("blk.0.attn_q.weight")
        direct_payload = gguf_path.read_bytes()[offset : offset + length]
        sidecar = (tmp_path / "model.onnx.data").read_bytes()
        assert direct_payload not in sidecar

        reloaded = ir.load(tmp_path / "model.onnx")
        q_proj = reloaded.graph.initializers[
            "model.layers.0.self_attn.q_proj.weight"
        ].const_value
        assert isinstance(q_proj, ir.ExternalTensor)
        assert (q_proj.location, q_proj.offset, q_proj.length) == (
            "model.gguf",
            offset,
            length,
        )

    def test_rejects_non_flat_source_and_detects_identity_change(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        gguf_path = source_dir / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)

        with pytest.raises(ValueError, match="flat same-directory packaging"):
            package.save(str(tmp_path / "output"), progress_bar=False)

        package.save(str(source_dir), progress_bar=False)
        with gguf_path.open("r+b") as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0] ^ 0xFF]))
        with pytest.raises(ValueError, match="identity mismatch"):
            verify_gguf_reuse_manifest(source_dir)

    def test_does_not_reuse_same_size_dtype_cast(self, tmp_path: Path):
        from mobius.integrations.gguf import build_from_gguf

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(
            gguf_path,
            projection_quantization="f16",
            float_type="f16",
        )
        with pytest.raises(ValueError, match="no byte-compatible tensors"):
            build_from_gguf(
                gguf_path,
                dtype="bf16",
                reuse_gguf_weights=True,
            )

    @pytest.mark.parametrize("artifact_name", ["model.onnx.data", ".gguf-reuse.lock"])
    def test_rejects_generated_artifact_name_collision(
        self, tmp_path: Path, artifact_name: str
    ):
        from mobius.integrations.gguf import build_from_gguf

        gguf_path = tmp_path / artifact_name
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        with pytest.raises(ValueError, match="collides"):
            package.save(str(tmp_path), progress_bar=False)
        # Validation happens before any generated artifact can truncate the source.
        assert gguf_path.stat().st_size == package.gguf_reuse_plan.size

    @pytest.mark.parametrize("artifact_name", ["model.onnx.data", ".gguf-reuse.lock"])
    def test_rejects_hardlink_to_generated_artifact(self, tmp_path: Path, artifact_name: str):
        from mobius.integrations.gguf import build_from_gguf

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        os.link(gguf_path, tmp_path / artifact_name)

        with pytest.raises(ValueError, match="hard-linked"):
            package.save(str(tmp_path), progress_bar=False)
        assert gguf_path.stat().st_size == package.gguf_reuse_plan.size

    def test_generated_looking_files_without_journal_are_preserved(self, tmp_path: Path):
        from mobius.integrations.gguf import build_from_gguf

        token = "0" * 32
        gguf_path = tmp_path / f".model.onnx.{token}.tmp"
        unrelated = tmp_path / f".gguf-reuse.json.{token}.tmp"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        source_bytes = gguf_path.read_bytes()
        unrelated.write_bytes(b"user-owned temporary data")

        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )

        assert gguf_path.read_bytes() == source_bytes
        assert unrelated.read_bytes() == b"user-owned temporary data"

    def test_ordinary_resave_removes_stale_reuse_manifest(self, tmp_path: Path):
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )

        loaded = ModelPackage.load(str(tmp_path))
        loaded.save(str(tmp_path), progress_bar=False)
        assert not (tmp_path / "gguf-reuse.json").exists()

    def test_verifier_rejects_unmanifested_external_initializer(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        model_path = tmp_path / "model.onnx"
        model = ir.load(model_path)
        initializer = next(
            value
            for value in model.graph.initializers.values()
            if not isinstance(value.const_value, ir.ExternalTensor)
        )
        tensor = initializer.const_value
        assert tensor is not None
        initializer.const_value = ir.ExternalTensor(
            "model.onnx.data",
            0,
            tensor.nbytes,
            tensor.dtype,
            shape=tensor.shape,
            name=tensor.name or initializer.name,
            base_dir=tmp_path,
        )
        ir.save(model, model_path)

        with pytest.raises(ValueError, match="Unmanifested sidecar initializer"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_verifier_rejects_wrong_external_dtype(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        model_path = tmp_path / "model.onnx"
        model = ir.load(model_path)
        initializer = model.graph.initializers["model.embed_tokens.weight"]
        external = initializer.const_value
        assert isinstance(external, ir.ExternalTensor)
        initializer.const_value = ir.ExternalTensor(
            external.location,
            external.offset,
            external.length,
            ir.DataType.UINT8,
            shape=ir.Shape([external.length]),
            name=external.name,
            base_dir=tmp_path,
        )
        initializer.dtype = ir.DataType.UINT8
        initializer.shape = ir.Shape([external.length])
        ir.save(model, model_path)

        with pytest.raises(ValueError, match="incompatible dtype"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_verifier_rejects_wrong_manifest_qtype(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        manifest_path = tmp_path / "gguf-reuse.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["reused_tensors"][0]["qtype"] = "F16"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="does not match source tensor"):
            verify_gguf_reuse_manifest(tmp_path)

    @pytest.mark.parametrize("length_delta", [-1, 1])
    def test_verifier_rejects_wrong_sidecar_byte_length(
        self, tmp_path: Path, length_delta: int
    ):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        model_path = tmp_path / "model.onnx"
        model = ir.load(model_path)
        initializer = next(
            value
            for value in model.graph.initializers.values()
            if isinstance(value.const_value, ir.ExternalTensor)
            and value.const_value.location == "model.onnx.data"
        )
        external = initializer.const_value
        assert isinstance(external, ir.ExternalTensor)
        assert external.length is not None
        initializer.const_value = ir.ExternalTensor(
            external.location,
            external.offset,
            external.length + length_delta,
            external.dtype,
            shape=external.shape,
            name=external.name,
            base_dir=tmp_path,
        )
        ir.save(model, model_path)

        with pytest.raises(ValueError, match="byte length"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_save_and_verifier_do_not_run_behind_active_writer(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            _reuse,
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        package.save(str(tmp_path), progress_bar=False)

        with _reuse._package_lock(tmp_path):
            operations = (
                lambda: verify_gguf_reuse_manifest(tmp_path),
                lambda: package.save(str(tmp_path), progress_bar=False),
            )
            for operation in operations:
                with pytest.raises(ValueError, match="locked by active writer"):
                    operation()

    @pytest.mark.parametrize(
        "artifact_name", [".gguf-reuse.lock", ".gguf-reuse.transaction.json"]
    )
    def test_rejects_dangling_control_artifact_symlink(
        self, tmp_path: Path, artifact_name: str
    ):
        from mobius.integrations.gguf import (
            _reuse,
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        artifact = tmp_path / artifact_name
        if artifact_name == _reuse._LOCK_NAME:
            artifact.unlink()
        artifact.symlink_to(tmp_path / "missing-target")

        with pytest.raises(ValueError, match="Unsafe GGUF"):
            if artifact_name == _reuse._LOCK_NAME:
                verify_gguf_reuse_manifest(tmp_path)
            else:
                build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
                    str(tmp_path), progress_bar=False
                )
        assert artifact.is_symlink()

    @pytest.mark.parametrize(
        "artifact_name", ["model.onnx", "model.onnx.data", "gguf-reuse.json"]
    )
    def test_verifier_rejects_symlinked_package_artifact(
        self, tmp_path: Path, artifact_name: str
    ):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        artifact = tmp_path / artifact_name
        moved = tmp_path / f"{artifact_name}.moved"
        artifact.replace(moved)
        artifact.symlink_to(moved)

        with pytest.raises(ValueError, match="Unsafe GGUF"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_failed_rerun_restores_existing_package(self, tmp_path: Path, monkeypatch):
        from mobius.integrations.gguf import _reuse, build_from_gguf

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        artifact_names = ("model.onnx", "model.onnx.data", "gguf-reuse.json")
        original = {name: (tmp_path / name).read_bytes() for name in artifact_names}

        real_replace = _reuse.os.replace
        injected = False

        def fail_manifest_install(source, destination):
            nonlocal injected
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                not injected
                and source_path.name.startswith(".gguf-reuse.json.")
                and destination_path.name == "gguf-reuse.json"
            ):
                injected = True
                raise OSError("injected manifest install failure")
            return real_replace(source, destination)

        monkeypatch.setattr(_reuse.os, "replace", fail_manifest_install)
        rerun = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        with pytest.raises(OSError, match="injected"):
            rerun.save(str(tmp_path), progress_bar=False)

        assert injected
        assert {name: (tmp_path / name).read_bytes() for name in artifact_names} == original
        assert not list(tmp_path.glob(".*.tmp"))
        assert not list(tmp_path.glob(".*.backup"))

    def test_interrupted_rerun_recovers_from_transaction_journal(
        self, tmp_path: Path, monkeypatch
    ):
        from mobius.integrations.gguf import (
            _reuse,
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        artifact_names = ("model.onnx", "model.onnx.data", "gguf-reuse.json")
        original = {name: (tmp_path / name).read_bytes() for name in artifact_names}

        real_replace = _reuse.os.replace
        interrupted = False

        def interrupt_manifest_install(source, destination):
            nonlocal interrupted
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                not interrupted
                and source_path.name.startswith(".gguf-reuse.json.")
                and destination_path.name == "gguf-reuse.json"
            ):
                interrupted = True
                raise KeyboardInterrupt
            return real_replace(source, destination)

        monkeypatch.setattr(_reuse.os, "replace", interrupt_manifest_install)
        rerun = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        with pytest.raises(KeyboardInterrupt):
            rerun.save(str(tmp_path), progress_bar=False)
        monkeypatch.setattr(_reuse.os, "replace", real_replace)

        verify_gguf_reuse_manifest(tmp_path)
        assert {name: (tmp_path / name).read_bytes() for name in artifact_names} == original
        assert not (tmp_path / ".gguf-reuse.transaction.json").exists()
        assert not list(tmp_path.glob(".*.backup"))

    def test_committed_transaction_recovery_keeps_new_artifacts(self, tmp_path: Path):
        from mobius.integrations.gguf import _reuse

        token = "0" * 32
        managed = []
        for name in ("model.onnx", "model.onnx.data", "gguf-reuse.json"):
            final = tmp_path / name
            final.write_bytes(f"new {name}".encode())
            backup_name = f".{name}.{token}.backup"
            (tmp_path / backup_name).write_bytes(f"old {name}".encode())
            managed.append({"final": name, "backup": backup_name, "had_existing": True})
        journal = {
            "phase": "committed",
            "managed": managed,
            "staged": [
                f".model.onnx.{token}.tmp",
                f".gguf-reuse.json.{token}.tmp",
            ],
        }
        (tmp_path / ".gguf-reuse.transaction.json").write_text(json.dumps(journal))

        _reuse._recover_transaction(tmp_path)

        for name in ("model.onnx", "model.onnx.data", "gguf-reuse.json"):
            assert (tmp_path / name).read_bytes() == f"new {name}".encode()
        assert not list(tmp_path.glob(".*.backup"))
        assert not (tmp_path / ".gguf-reuse.transaction.json").exists()

    def test_verifier_rejects_wrong_transform_parameter(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        manifest_path = tmp_path / "gguf-reuse.json"
        manifest = json.loads(manifest_path.read_text())
        q_route = next(
            route
            for route in manifest["reused_tensors"]
            if route["transform"] == "llama_qk_permute"
        )
        q_route["transform_parameter"] = 3
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="Invalid Q/K head count"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_verifier_rejects_wrong_transform_permutation(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        model_path = tmp_path / "model.onnx"
        model = ir.load(model_path)
        transpose = next(
            node
            for node in model.graph
            if node.name == "model.layers.0.self_attn.q_proj.weight.gguf_reuse.Transpose"
        )
        transpose.attributes["perm"] = ir.AttrInt64s("perm", [0, 1, 2, 3])
        ir.save(model, model_path)

        with pytest.raises(ValueError, match="Q/K permutation shapes are wrong"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_overwrite_rejects_filesystem_without_hardlinks(self, tmp_path: Path, monkeypatch):
        from mobius.integrations.gguf import _reuse, build_from_gguf

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        package.save(str(tmp_path), progress_bar=False)
        original = (tmp_path / "model.onnx").read_bytes()

        def reject_link(source, destination):
            raise OSError("hard links unavailable")

        monkeypatch.setattr(_reuse.os, "link", reject_link)
        with pytest.raises(ValueError, match="requires same-directory hard-link"):
            package.save(str(tmp_path), progress_bar=False)
        assert (tmp_path / "model.onnx").read_bytes() == original
        assert not (tmp_path / ".gguf-reuse.transaction.json").exists()

    def test_recovery_rejects_unsafe_journal_paths(self, tmp_path: Path):
        from mobius.integrations.gguf import _reuse

        victim = tmp_path.parent / "victim"
        victim.write_bytes(b"keep")
        journal = {
            "managed": [
                {
                    "final": "../victim",
                    "backup": ".victim.00000000000000000000000000000000.backup",
                    "had_existing": True,
                },
                {
                    "final": "model.onnx.data",
                    "backup": ".model.onnx.data.00000000000000000000000000000000.backup",
                    "had_existing": False,
                },
                {
                    "final": "gguf-reuse.json",
                    "backup": ".gguf-reuse.json.00000000000000000000000000000000.backup",
                    "had_existing": False,
                },
            ],
            "staged": [
                ".model.onnx.00000000000000000000000000000000.tmp",
                ".gguf-reuse.json.00000000000000000000000000000000.tmp",
            ],
        }
        (tmp_path / ".gguf-reuse.transaction.json").write_text(json.dumps(journal))

        with pytest.raises(ValueError, match="Unsafe GGUF transaction"):
            _reuse._recover_transaction(tmp_path)
        assert victim.read_bytes() == b"keep"

    def test_transaction_removes_obsolete_sidecar(self, tmp_path: Path):
        from mobius.integrations.gguf import _reuse

        final_model = tmp_path / "model.onnx"
        final_sidecar = tmp_path / "model.onnx.data"
        final_manifest = tmp_path / "gguf-reuse.json"
        for path in (final_model, final_sidecar, final_manifest):
            path.write_bytes(b"old")
        staged_model = tmp_path / f".model.onnx.{'0' * 32}.tmp"
        staged_manifest = tmp_path / f".gguf-reuse.json.{'1' * 32}.tmp"
        staged_model.write_bytes(b"new model")
        staged_manifest.write_bytes(b"new manifest")

        _reuse._replace_artifacts(
            {
                final_model: staged_model,
                final_manifest: staged_manifest,
            },
            (final_model, final_sidecar, final_manifest),
        )

        assert final_model.read_bytes() == b"new model"
        assert final_manifest.read_bytes() == b"new manifest"
        assert not final_sidecar.exists()


@pytest.fixture
def q5_1_gguf(tmp_path: Path) -> Path:
    """Create a GGUF whose projections require dequantize/requantize."""
    path = tmp_path / "test_q5_1.gguf"
    _write_quantized_gguf(path, projection_quantization="q5_1")
    return path


class TestBuildQuantizedGguf:
    """Tests for the default quantization-preserving GGUF build."""

    def test_produces_model_package(self, q4_0_gguf: Path):
        """Quantized build returns a valid ModelPackage."""
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf)
        assert "model" in pkg
        assert pkg["model"].graph is not None

    def test_model_has_matmulnbits_ops(self, q4_0_gguf: Path):
        """The API default uses MatMulNBits instead of float MatMul weights."""
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf)
        model = pkg["model"]

        op_types = {node.op_type for node in model.graph if node.op_type}
        assert "MatMulNBits" in op_types, (
            f"Expected MatMulNBits in ops, got: {sorted(op_types)}"
        )

    def test_default_quantized_package_save_reload(self, q4_0_gguf: Path, tmp_path: Path):
        """Default quantized ops and weights survive ModelPackage persistence."""
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        output_dir = tmp_path / "saved"
        build_from_gguf(q4_0_gguf).save(str(output_dir), progress_bar=False)
        reloaded = ModelPackage.load(str(output_dir))

        op_types = {node.op_type for node in reloaded["model"].graph}
        assert "MatMulNBits" in op_types
        assert (
            reloaded["model"]
            .graph.initializers["model.layers.0.self_attn.q_proj.weight"]
            .const_value
            is not None
        )

    def test_decoder_backed_qtype_build_save_reload(
        self, q5_1_gguf: Path, tmp_path: Path
    ) -> None:
        """A pure Q5_1 file uses the declared 4-bit requantization route."""
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        output_dir = tmp_path / "saved_q5_1"
        package = build_from_gguf(q5_1_gguf)
        package.save(str(output_dir), progress_bar=False)
        model = ModelPackage.load(str(output_dir))["model"]

        quantized_nodes = [node for node in model.graph if node.op_type == "MatMulNBits"]
        assert quantized_nodes
        assert all(node.attributes["bits"].value == 4 for node in quantized_nodes)
        assert all(node.attributes["block_size"].value == 32 for node in quantized_nodes)

    def test_float_only_default_uses_float_path(self, float_only_gguf: Path):
        """F32/BF16-only GGUFs do not fail when preservation is the default."""
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(float_only_gguf)["model"]
        op_types = {node.op_type for node in model.graph}
        assert "MatMulNBits" not in op_types
        assert "GatherBlockQuantized" not in op_types
        assert "BlockQuantizedMatMul" not in op_types

    @pytest.mark.parametrize(
        "architecture",
        ["olmo", "olmo2", "cohere2", "arcee", "smollm3", "exaone"],
    )
    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_dense_cohort_builds_complete_graphs(
        self,
        architecture: str,
        projection_quantization: str,
        tmp_path: Path,
    ) -> None:
        """Exact per-architecture tensor sets satisfy the full graph."""
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{projection_quantization}.gguf"
        _write_quantized_gguf(
            path,
            architecture=architecture,
            projection_quantization=projection_quantization,
        )

        model = build_from_gguf(path)["model"]
        op_types = {node.op_type for node in model.graph}
        if projection_quantization == "q4_0":
            assert "MatMulNBits" in op_types
        else:
            assert "MatMulNBits" not in op_types

        output_dir = tmp_path / f"{architecture}-{projection_quantization}-onnx"
        build_from_gguf(path).save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        feeds = {
            "input_ids": np.array([[1, 2]], dtype=np.int64),
            "attention_mask": np.ones((1, 2), dtype=np.int64),
            "position_ids": np.array([[0, 1]], dtype=np.int64),
            "past_key_values.0.key": np.empty((1, 2, 0, 16), dtype=np.float32),
            "past_key_values.0.value": np.empty((1, 2, 0, 16), dtype=np.float32),
        }
        first = session.run(["logits"], feeds)[0]
        second = session.run(["logits"], feeds)[0]
        assert first.shape == (1, 2, 256)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_dream_masked_diffusion_float_and_quantized_forward(
        self, projection_quantization: str, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"dream-{projection_quantization}.gguf"
        quantized = projection_quantization == "q4_0"
        _write_quantized_gguf(
            path,
            architecture="dream",
            projection_quantization=projection_quantization,
            quantize_embedding=quantized,
            output_quantization=projection_quantization,
        )

        preserved = build_from_gguf(path)
        model = preserved["model"]
        op_types = {node.op_type for node in model.graph}
        assert ("MatMulNBits" in op_types) is quantized
        assert [value.name for value in model.graph.inputs] == ["input_ids"]
        assert not any(
            token in value.name
            for value in (*model.graph.inputs, *model.graph.outputs)
            for token in ("past", "present", "cache")
        )

        output_dir = tmp_path / f"dream-{projection_quantization}-onnx"
        preserved.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        logits, proposed = session.run(
            None, {"input_ids": np.array([[1, 2, 255]], dtype=np.int64)}
        )
        assert logits.shape == (1, 3, 256)
        assert proposed.shape == (1, 3)
        assert np.isfinite(logits).all()
        np.testing.assert_array_equal(proposed, np.argmax(logits, axis=-1))

        if quantized:
            dequantized_dir = tmp_path / "dream-dequantized-onnx"
            build_from_gguf(path, keep_quantized=False).save(
                dequantized_dir, progress_bar=False
            )
            float_session = ort.InferenceSession(
                str(dequantized_dir / "model.onnx"),
                providers=["CPUExecutionProvider"],
            )
            float_logits = float_session.run(
                ["logits"], {"input_ids": np.array([[1, 2, 255]], dtype=np.int64)}
            )[0]
            np.testing.assert_allclose(logits, float_logits, rtol=0, atol=2e-2)

    @pytest.mark.parametrize("architecture", ["dream", "llada-moe", "rnd1"])
    @pytest.mark.parametrize(
        ("source_kind", "writer_kwargs"),
        [
            ("affine", {"projection_quantization": "q4_0"}),
            (
                "native",
                {"projection_quantization": "mxfp4"},
            ),
            (
                "mixed_float_fused",
                {
                    "projection_quantization": "q4_0",
                    "fused_qkv_float": True,
                },
            ),
        ],
    )
    def test_quantized_diffusion_fused_qkv_rejects_before_graph_build(
        self,
        architecture: str,
        source_kind: str,
        writer_kwargs: dict,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        import mobius._builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-fused-{source_kind}.gguf"
        if architecture == "dream":
            _write_quantized_gguf(
                path,
                architecture=architecture,
                fused_qkv=True,
                **writer_kwargs,
            )
        else:
            _write_moe_gguf(
                path,
                architecture,
                writer_kwargs["projection_quantization"],
                diffusion_fused_qkv=True,
                fused_qkv_float=writer_kwargs.get("fused_qkv_float", False),
            )
        monkeypatch.setattr(
            mobius._builder,
            "build_from_module",
            lambda *_args, **_kwargs: pytest.fail("graph construction must not run"),
        )

        with pytest.raises(
            ValueError,
            match=rf"Quantization-preserving import of fused QKV.*{architecture}",
        ) as error:
            build_from_gguf(path)
        assert "keep_quantized=False" in str(error.value)

    @pytest.mark.parametrize("architecture", ["dream", "llada-moe", "rnd1"])
    @pytest.mark.parametrize(
        ("projection_quantization", "keep_quantized"),
        [("f32", True), ("q4_0", False)],
    )
    def test_float_diffusion_fused_qkv_still_builds_and_runs(
        self,
        architecture: str,
        projection_quantization: str,
        keep_quantized: bool,
        tmp_path: Path,
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-fused-{projection_quantization}.gguf"
        if architecture == "dream":
            _write_quantized_gguf(
                path,
                architecture=architecture,
                projection_quantization=projection_quantization,
                fused_qkv=True,
            )
        else:
            _write_moe_gguf(
                path,
                architecture,
                projection_quantization,
                diffusion_fused_qkv=True,
                fused_qkv_float=projection_quantization == "f32",
            )

        package = build_from_gguf(path, keep_quantized=keep_quantized)
        model = package["model"]
        assert "MatMulNBits" not in {node.op_type for node in model.graph}
        output_dir = tmp_path / f"{architecture}-fused-{projection_quantization}-onnx"
        package.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        logits, proposed = session.run(
            None, {"input_ids": np.array([[1, 2, 255]], dtype=np.int64)}
        )
        assert logits.shape == (1, 3, 256)
        np.testing.assert_array_equal(proposed, np.argmax(logits, axis=-1))

    @pytest.mark.parametrize(
        "architecture", ["olmoe", "phimoe", "qwen2moe", "qwen3moe", "granitemoe"]
    )
    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_moe_cohort_builds_complete_graphs(
        self,
        architecture: str,
        projection_quantization: str,
        tmp_path: Path,
    ) -> None:
        """All routed/shared expert tensors survive build, save, load, and ORT."""
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{projection_quantization}.gguf"
        _write_moe_gguf(path, architecture, projection_quantization)
        package = build_from_gguf(path)
        model = package["model"]
        initializer_names = set(model.graph.initializers)

        def has_weight(stem: str) -> bool:
            return (
                f"{stem}.weight" in initializer_names
                or f"{stem}.weight_t" in initializer_names
            )

        for expert in range(4):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                assert has_weight(f"model.layers.0.mlp.experts.{expert}.{projection}")
        assert has_weight("model.layers.0.mlp.gate")
        if architecture in {"qwen2moe", "granitemoe"}:
            assert has_weight("model.layers.0.mlp.shared_expert.gate_proj")
            assert has_weight("model.layers.0.mlp.shared_expert.up_proj")
            assert has_weight("model.layers.0.mlp.shared_expert.down_proj")

        op_types = {node.op_type for node in model.graph}
        if projection_quantization == "q4_0":
            assert "MatMulNBits" in op_types
            assert not any("fc1_experts" in name for name in initializer_names)
        else:
            assert "MatMulNBits" not in op_types

        output_dir = tmp_path / f"{architecture}-{projection_quantization}-onnx"
        package.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        feeds = {
            "input_ids": np.array([[1, 2]], dtype=np.int64),
            "attention_mask": np.ones((1, 2), dtype=np.int64),
            "position_ids": np.array([[0, 1]], dtype=np.int64),
            "past_key_values.0.key": np.empty((1, 2, 0, 16), dtype=np.float32),
            "past_key_values.0.value": np.empty((1, 2, 0, 16), dtype=np.float32),
        }
        first = session.run(["logits"], feeds)[0]
        second = session.run(["logits"], feeds)[0]
        assert first.shape == (1, 2, 256)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("architecture", ["llada-moe", "rnd1"])
    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_diffusion_moe_cohort_builds_and_runs_masked_forward(
        self,
        architecture: str,
        projection_quantization: str,
        tmp_path: Path,
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{projection_quantization}.gguf"
        _write_moe_gguf(path, architecture, projection_quantization)
        package = build_from_gguf(path)
        model = package["model"]
        op_types = {node.op_type for node in model.graph}
        assert ("MatMulNBits" in op_types) is (projection_quantization == "q4_0")
        assert [value.name for value in model.graph.inputs] == ["input_ids"]
        assert not any(
            token in value.name
            for value in (*model.graph.inputs, *model.graph.outputs)
            for token in ("past", "present", "cache")
        )

        output_dir = tmp_path / f"{architecture}-{projection_quantization}-onnx"
        package.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        feeds = {"input_ids": np.array([[1, 2, 255]], dtype=np.int64)}
        first_logits, first_proposal = session.run(None, feeds)
        second_logits, second_proposal = session.run(None, feeds)
        assert first_logits.shape == (1, 3, 256)
        assert first_proposal.shape == (1, 3)
        assert np.isfinite(first_logits).all()
        np.testing.assert_array_equal(first_logits, second_logits)
        np.testing.assert_array_equal(first_proposal, second_proposal)

    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_granitemoe_qk_rows_are_reverse_permuted_by_value(
        self, projection_quantization: str, tmp_path: Path
    ) -> None:
        import torch

        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.integrations.gguf._repacker import repack_gguf_tensor
        from mobius.integrations.gguf._tensor_processors import _reverse_permute

        path = tmp_path / f"granitemoe-qk-{projection_quantization}.gguf"
        _write_moe_gguf(path, "granitemoe", projection_quantization)
        source = GGUFModel(path)
        model = build_from_gguf(path)["model"]

        raw_tensors = {
            name: (raw, qtype, tuple(int(dim) for dim in shape))
            for name, raw, qtype, shape in source.tensor_items_raw()
        }
        for gguf_name, projection, heads in (
            ("blk.0.attn_q.weight", "q_proj", 4),
            ("blk.0.attn_k.weight", "k_proj", 2),
        ):
            raw, qtype, shape = raw_tensors[gguf_name]
            if projection_quantization == "f32":
                unpermuted = torch.from_numpy(np.array(source.get_tensor(gguf_name)))
            else:
                packed = repack_gguf_tensor(
                    raw.ravel().view(np.uint8),
                    qtype.value,
                    shape,
                )
                unpermuted = torch.from_numpy(packed.weight)
            expected = _reverse_permute(unpermuted, heads)
            stem = f"model.layers.0.self_attn.{projection}"
            if projection_quantization == "f32":
                actual = model.graph.initializers[f"{stem}.weight_t"].const_value.numpy().T
            else:
                actual = model.graph.initializers[f"{stem}.weight"].const_value.numpy()
            np.testing.assert_array_equal(actual, expected.numpy())
            assert not torch.equal(expected, unpermuted)

    def test_granitemoe_zero_experts_selects_dense_graph(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitemoe-dense.gguf"
        _write_quantized_gguf(
            path,
            architecture="granitemoe",
            projection_quantization="q4_0",
            tie_embeddings=True,
            quantize_embedding=True,
        )
        model = build_from_gguf(path, keep_quantized=False)["model"]
        names = set(model.graph.initializers)
        assert any(".mlp.gate_proj.weight" in name for name in names)
        assert not any(".mlp.experts." in name for name in names)
        assert not any(".mlp.gate.weight" in name for name in names)
        assert "model.embed_tokens.weight" in names
        assert not any(name.startswith("lm_head.") for name in names)

    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_phimoe_fused_qkv_is_split_without_loss(
        self, projection_quantization: str, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"phimoe-fused-{projection_quantization}.gguf"
        _write_moe_gguf(
            path,
            "phimoe",
            projection_quantization,
            phi_fused_qkv=True,
        )
        model = build_from_gguf(path)["model"]
        names = set(model.graph.initializers)
        for projection in ("q_proj", "k_proj", "v_proj"):
            assert any(
                f"model.layers.0.self_attn.{projection}.weight" in name for name in names
            )
            assert any(f"model.layers.0.self_attn.{projection}.bias" in name for name in names)
        assert not any("qkv_proj" in name for name in names)

    @pytest.mark.parametrize("architecture", ["qwen3moe", "granitemoe"])
    def test_tied_quantized_embedding_is_shared_with_output_head(
        self, architecture: str, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-tied-q4.gguf"
        _write_moe_gguf(
            path,
            architecture,
            "q4_0",
            quantize_tied_embedding=True,
        )
        package = build_from_gguf(path)
        model = package["model"]
        op_types = [node.op_type for node in model.graph]
        assert op_types.count("GatherBlockQuantized") == 1
        assert "MatMulNBits" in op_types
        assert "model.embed_tokens.qweight" in model.graph.initializers
        assert not any(name.startswith("lm_head.") for name in model.graph.initializers)
        output_dir = tmp_path / f"{architecture}-tied-q4-onnx"
        package.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        logits = session.run(
            ["logits"],
            {
                "input_ids": np.array([[1, 2]], dtype=np.int64),
                "attention_mask": np.ones((1, 2), dtype=np.int64),
                "position_ids": np.array([[0, 1]], dtype=np.int64),
                "past_key_values.0.key": np.empty((1, 2, 0, 16), dtype=np.float32),
                "past_key_values.0.value": np.empty((1, 2, 0, 16), dtype=np.float32),
            },
        )[0]
        assert logits.shape == (1, 2, 256)
        assert np.isfinite(logits).all()

    @pytest.mark.parametrize("suffix", ["scale", "input_scale"])
    def test_qwen3moe_auxiliary_expert_scales_are_rejected_before_build(
        self, suffix: str, tmp_path: Path, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"qwen3moe-{suffix}.gguf"
        _write_moe_gguf(path, "qwen3moe", "q4_0", expert_scale_suffix=suffix)

        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match="cannot represent GGUF scale/input_scale"):
            build_from_gguf(path)
        assert not graph_build_started

    def test_malformed_qwen3moe_expert_scale_is_rejected(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "qwen3moe-malformed-scale.gguf"
        _write_moe_gguf(
            path,
            "qwen3moe",
            "q4_0",
            expert_scale_suffix="scale",
            malformed_expert_scale=True,
        )
        with pytest.raises(ValueError, match=r"expected shape \(4,\), got \(3,\)"):
            build_from_gguf(path)

    def test_qwen3moe_optional_expert_scales_may_be_absent(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "qwen3moe-no-scale.gguf"
        _write_moe_gguf(path, "qwen3moe", "q4_0")
        assert build_from_gguf(path)["model"].graph.num_nodes() > 0

    def test_q4_0_matmulnbits_has_explicit_zero_points(self, q4_0_gguf: Path):
        """GGUF Q4_0 projections explicitly encode zp=8 instead of EP defaults."""
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q4_0_gguf, keep_quantized=True)["model"]
        nodes = [node for node in model.graph if node.op_type == "MatMulNBits"]
        assert nodes
        for node in nodes:
            assert len(node.inputs) == 4
            zero_point_name = node.inputs[3].name
            assert zero_point_name.endswith(".zero_points")
            zero_points = model.graph.initializers[zero_point_name]
            np.testing.assert_array_equal(zero_points.const_value.numpy(), 0x88)

    def test_native_blocks_emit_block_quantized_matmul_and_preserve_bytes(
        self,
        native_block_gguf: tuple[Path, str, int, int],
    ):
        """Runtime-native IQ/MXFP4 projections retain their exact GGUF bytes."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        path, format_name, block_elements, block_bytes = native_block_gguf
        model = build_from_gguf(path, keep_quantized=True)["model"]
        nodes = [node for node in model.graph if node.op_type == "BlockQuantizedMatMul"]
        assert len(nodes) == 7
        assert all(node.domain == "pkg.nxrt" for node in nodes)
        for node in nodes:
            attrs = {attribute.name: attribute.value for attribute in node.attributes.values()}
            assert attrs["format"] == format_name
            assert attrs["block_layout_version"] == 1
            assert attrs["K"] == 256
            assert attrs["N"] == 256

        weight = model.graph.initializers["model.layers.0.self_attn.o_proj.weight"]
        assert weight.dtype == ir.DataType.UINT8
        n_blocks = (256 + block_elements - 1) // block_elements
        assert list(weight.shape) == [256, n_blocks, block_bytes]
        expected = np.arange(256 * n_blocks * block_bytes, dtype=np.uint8).reshape(
            256, n_blocks, block_bytes
        )
        np.testing.assert_array_equal(weight.const_value.numpy(), expected)
        assert "model.layers.0.self_attn.o_proj.scales" not in model.graph.initializers

        assert model.graph.opset_imports["pkg.nxrt"] == 1
        proto = ir.serde.serialize_model(model)
        imports = {opset.domain: opset.version for opset in proto.opset_import}
        assert imports["pkg.nxrt"] == 1

    def test_mixed_native_quantization_uses_q4_scaffold(self, mixed_native_q5_q8_gguf: Path):
        """Native IQ tensors force Q5_1 fallback weights onto a Q4 scaffold."""
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(mixed_native_q5_q8_gguf, keep_quantized=True)["model"]
        native_nodes = [node for node in model.graph if node.op_type == "BlockQuantizedMatMul"]
        assert len(native_nodes) == 6

        value_nodes = [
            node
            for node in model.graph
            if node.op_type == "MatMulNBits"
            and node.inputs[1].name == "model.layers.0.self_attn.v_proj.weight"
        ]
        assert len(value_nodes) == 1
        assert value_nodes[0].attributes["bits"].value == 4
        assert value_nodes[0].attributes["block_size"].value == 32
        assert "model.layers.0.self_attn.v_proj.zero_points" in model.graph.initializers

        native_weight = model.graph.initializers["model.layers.0.self_attn.o_proj.weight"]
        expected = np.arange(256 * 136, dtype=np.uint8).reshape(256, 1, 136)
        np.testing.assert_array_equal(native_weight.const_value.numpy(), expected)

    def test_quantized_embedding_uses_gatherblockquantized(self, q4_0_embedding_gguf: Path):
        """A quantized GGUF embedding remains packed in the ONNX graph."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q4_0_embedding_gguf, keep_quantized=True)["model"]
        gather_nodes = [node for node in model.graph if node.op_type == "GatherBlockQuantized"]
        assert len(gather_nodes) == 1
        assert gather_nodes[0].domain == "com.microsoft"
        assert len(gather_nodes[0].inputs) == 4

        qweight = model.graph.initializers["model.embed_tokens.qweight"]
        assert qweight.dtype == ir.DataType.UINT8
        assert list(qweight.shape) == [256, 32]
        assert list(model.graph.initializers["model.embed_tokens.scales"].shape) == [
            256,
            2,
        ]
        zero_points = model.graph.initializers["model.embed_tokens.zero_points"]
        assert zero_points.dtype == ir.DataType.UINT8
        assert list(zero_points.shape) == [256, 1]
        np.testing.assert_array_equal(zero_points.const_value.numpy(), 0x88)
        assert "model.embed_tokens.weight" not in model.graph.initializers

    def test_gatherblockquantized_zero_point_dequantizes_q4_0(self, tmp_path: Path):
        """GatherBlockQuantized output must match GGUF Q4_0's ``(q - 8) * scale``."""
        actual = _run_gather_block_quantized(tmp_path, zero_point=0x08).astype(np.float32)
        expected = np.stack(
            [
                np.full(32, (10 - 8) * 0.5, dtype=np.float32),
                np.full(32, (10 - 8) * 0.25, dtype=np.float32),
            ]
        )
        np.testing.assert_allclose(actual, expected)
        wrong = _run_gather_block_quantized(tmp_path, zero_point=0x00).astype(np.float32)
        assert not np.allclose(wrong, expected)

    def test_native_projection_abi_embedding_is_converted_to_gather(
        self, iq4_nl_embedding_gguf: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(iq4_nl_embedding_gguf)["model"]
        gather_nodes = [node for node in model.graph if node.op_type == "GatherBlockQuantized"]

        assert len(gather_nodes) == 1
        assert "model.embed_tokens.qweight" in model.graph.initializers
        assert "model.embed_tokens.weight" not in model.graph.initializers

    def test_tied_quantized_embedding_drives_matmulnbits_head(
        self, q4_0_tied_embedding_gguf: Path
    ):
        """Tied embedding/head share one packed table across both contrib ops."""
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q4_0_tied_embedding_gguf, keep_quantized=True)["model"]
        op_types = [node.op_type for node in model.graph]
        assert op_types.count("GatherBlockQuantized") == 1
        assert "MatMulNBits" in op_types
        assert "model.embed_tokens.qweight" in model.graph.initializers
        assert "model.embed_tokens.zero_points" in model.graph.initializers
        assert not any(name.startswith("lm_head.") for name in model.graph.initializers)

    def test_untied_quantized_head_uses_q4_matmulnbits(
        self, q4_0_embedding_q8_head_gguf: Path
    ):
        """An untied quantized output is requantized to the graph's Q4 layout."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q4_0_embedding_q8_head_gguf, keep_quantized=True)["model"]
        head_nodes = [
            node
            for node in model.graph
            if node.op_type == "MatMulNBits" and node.outputs[0].name == "logits"
        ]
        assert len(head_nodes) == 1
        assert head_nodes[0].attributes["bits"].value == 4
        assert head_nodes[0].attributes["block_size"].value == 32

        qweight = model.graph.initializers["lm_head.weight"]
        assert qweight.dtype == ir.DataType.UINT8
        assert list(qweight.shape) == [256, 2, 16]
        assert list(model.graph.initializers["lm_head.scales"].shape) == [256, 2]
        assert "lm_head.weight_t" not in model.graph.initializers

    def test_unsupported_requantization_target_has_clear_error(
        self, q8_0_projection_q4_head_gguf: Path
    ):
        """Mixed targets outside 4-bit/block-32 fail before the Q4 repacker."""
        from mobius.integrations.gguf import build_from_gguf

        with pytest.raises(
            ValueError,
            match=(
                "keep_quantized MatMulNBits requantization currently supports only "
                r"4-bit/block-32 targets; got bits=8 block=32 for tensor lm_head\.weight"
            ),
        ):
            build_from_gguf(q8_0_projection_q4_head_gguf, keep_quantized=True)

    def test_norms_are_float(self, q4_0_gguf: Path):
        """Norm weights remain float, not quantized."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf, keep_quantized=True)
        model = pkg["model"]

        for init in model.graph.initializers.values():
            name = init.name or ""
            if "norm" in name and "weight" in name:
                assert init.dtype != ir.DataType.UINT8, (
                    f"Norm {name} should be float, not uint8"
                )

    def test_dequantized_path_no_matmulnbits(self, q4_0_gguf: Path):
        """Explicit API dequantization emits no quantized projection ops."""
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf, keep_quantized=False)
        model = pkg["model"]

        op_types = {node.op_type for node in model.graph if node.op_type}
        assert "MatMulNBits" not in op_types
        assert "BlockQuantizedMatMul" not in op_types

    def test_detect_quant_params(self, q4_0_gguf: Path):
        """_detect_quant_params finds Q4_0 as dominant type."""
        from mobius.integrations.gguf._builder import (
            _detect_quant_params,
        )
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_model = GGUFModel(q4_0_gguf)
        bits, block_size, is_sym = _detect_quant_params(gguf_model, gguf_model.architecture)
        assert bits == 4
        assert block_size == 32
        assert is_sym is False

    def test_embedding_quantization_check_is_metadata_only(self, monkeypatch):
        """Embedding compatibility does not read or repack tensor data."""
        from types import SimpleNamespace

        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf import _tencent_q1_0
        from mobius.integrations.gguf._builder import _can_quantize_embedding

        monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: False)
        tensor = SimpleNamespace(
            name="token_embd.weight",
            tensor_type=GGMLQuantizationType.Q4_0,
            shape=(64, 256),
        )
        model = SimpleNamespace(
            _reader=SimpleNamespace(tensors=[tensor]),
            reader_tensors=lambda: [tensor],
        )

        assert _can_quantize_embedding(model, "llama", bits=4, block_size=32)

    def test_tencent_q1_0_embedding_is_not_quantized(self, monkeypatch):
        """Tencent Q1_0 detection short-circuits before inspecting tensors."""
        from types import SimpleNamespace

        from mobius.integrations.gguf import _tencent_q1_0
        from mobius.integrations.gguf._builder import _can_quantize_embedding

        monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: True)
        model = SimpleNamespace()

        assert not _can_quantize_embedding(model, "llama", bits=4, block_size=128)

    def test_decoder_backed_embedding_uses_affine_gather_target(self, monkeypatch):
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf import _tencent_q1_0
        from mobius.integrations.gguf._builder import _can_quantize_embedding

        monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: False)
        tensor = SimpleNamespace(
            name="token_embd.weight",
            tensor_type=GGMLQuantizationType.Q5_1,
            shape=(64, 256),
        )
        model = SimpleNamespace(
            _reader=SimpleNamespace(tensors=[tensor]),
            reader_tensors=lambda: [tensor],
        )

        assert _can_quantize_embedding(model, "llama", bits=4, block_size=32)

    def test_decoder_backed_output_head_stays_quantized(self):
        from mobius.integrations.gguf._builder import _can_quantize_lm_head

        class _OutputModel:
            def tensor_items_raw(self):
                yield (
                    "output.weight",
                    np.empty(0, dtype=np.uint8),
                    SimpleNamespace(name="TQ1_0"),
                    (256, 64),
                )

        assert _can_quantize_lm_head(_OutputModel(), "llama")

    def test_detect_q4_k_m_mixed_profile(self):
        """Q4_K presence selects a 4-bit target despite more Q5_0 tensors."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import _detect_quant_params

        class _MixedModel:
            def tensor_items_raw(self):
                for i in range(5):
                    yield (
                        f"blk.{i}.attn_q.weight",
                        np.empty(0, dtype=np.uint8),
                        GGMLQuantizationType.Q5_0,
                        (64, 64),
                    )
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_K,
                    (64, 128),
                )

        bits, block_size, is_sym = _detect_quant_params(_MixedModel(), "llama")
        assert (bits, block_size, is_sym) == (4, 32, False)

    def test_decoder_backed_qtype_selects_explicit_requantization_target(self):
        """A decoder-backed qtype takes the declared 4-bit requantization route."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import _detect_quant_params

        class _UnsupportedModel:
            def tensor_items_raw(self):
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q5_K,
                    (64, 128),
                )

        assert _detect_quant_params(_UnsupportedModel(), "llama") == (4, 32, False)

    def test_qtype_without_decoder_or_kernel_is_rejected_actionably(self):
        import enum

        from mobius.integrations.gguf._builder import _detect_quant_params

        class _PinnedQType(enum.IntEnum):
            Q2_0 = 42

        class _Q20Model:
            def tensor_items_raw(self):
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    _PinnedQType.Q2_0,
                    (64, 128),
                )

        with pytest.raises(
            ValueError,
            match=r"Q2_0: gguf-py ships no Python dequantizer.*Re-quantize",
        ):
            _detect_quant_params(_Q20Model(), "llama")

    def test_out_of_census_qtype_is_rejected_before_route_selection(self):
        import enum

        from mobius.integrations.gguf._builder import _detect_quant_params

        class _FutureQType(enum.IntEnum):
            FUTURE = 99

        class _FutureModel:
            def tensor_items_raw(self):
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    _FutureQType.FUTURE,
                    (64, 128),
                )

        with pytest.raises(ValueError, match=r"outside the pinned llama\.cpp census"):
            _detect_quant_params(_FutureModel(), "llama")

    def test_architecture_without_quantized_modules_rejects_preservation(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "internlm2_q4_0.gguf"
        _write_quantized_gguf(path, architecture="internlm2")

        with pytest.raises(
            ValueError,
            match=r"does not support keep_quantized=True.*floating Linear modules",
        ):
            build_from_gguf(path)

    def test_q6_k_selects_the_asymmetric_four_bit_repack_target(self):
        """Q6_K repacks to the same 4-bit/32 target as Q4_K, with zero points.

        Q6_K's source form is symmetric around 32, but it reaches MatMulNBits
        through the asymmetric affine requantizer, so zero points are required.
        A missing entry in `type_can_omit_zero_points` raises `KeyError` here
        rather than producing a wrong model.
        """
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import _detect_quant_params

        class _Q6KModel:
            def tensor_items_raw(self):
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q6_K,
                    (64, 128),
                )

        bits, block_size, is_sym = _detect_quant_params(_Q6KModel(), "llama")

        assert (bits, block_size) == (4, 32)
        assert is_sym is False

    def test_runtime_unsupported_format_does_not_select_native_op(self):
        """A GGUF type outside the runtime contract remains on the fallback."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import _native_block_format

        assert _native_block_format(GGMLQuantizationType.Q5_0) is None

    @pytest.mark.parametrize("moe_container", ["experts", "moe.experts"])
    def test_native_moe_tensor_maps_to_each_expert(self, moe_container: str):
        """Stacked GGUF MoE blocks route to standard and DeepSeek expert paths."""
        from mobius.integrations.gguf._builder import _native_block_target_stems

        available = {f"model.layers.0.mlp.{moe_container}.{i}.gate_proj" for i in range(3)}
        assert _native_block_target_stems(
            "model.layers.0.mlp.experts.gate_proj.weight",
            (3, 64, 64),
            available,
        ) == sorted(available, key=lambda name: int(name.split(".")[-2]))

    def test_float_moe_tensor_preserves_expert_order_and_rejects_bad_shape(self):
        import torch

        from mobius._configs import ArchitectureConfig
        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        config = ArchitectureConfig(
            vocab_size=32,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=3,
            num_experts_per_tok=2,
            moe_intermediate_size=6,
        )
        key = "model.layers.0.mlp.experts.gate_proj.weight"
        stacked = torch.arange(3 * 6 * 4, dtype=torch.float32).reshape(3, 6, 4)
        normalized = _normalize_gguf_weights({key: stacked}, config=config)

        assert key not in normalized
        for expert in range(3):
            target = f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"
            assert torch.equal(normalized[target], stacked[expert])

        with pytest.raises(ValueError, match="Invalid stacked expert shape"):
            _normalize_gguf_weights({key: stacked[:, :-1]}, config=config)
        with pytest.raises(ValueError, match="Invalid stacked expert shape"):
            _normalize_gguf_weights({key: stacked.reshape(18, 4)}, config=config)

        router_key = "model.layers.0.mlp.gate.weight"
        with pytest.raises(ValueError, match="Invalid router shape"):
            _normalize_gguf_weights(
                {router_key: torch.empty(2, 4)},
                config=config,
            )

    @pytest.mark.parametrize("architecture", ["dream", "llada-moe", "rnd1"])
    def test_float_diffusion_fused_qkv_splits_by_gqa_width(self, architecture: str):
        import torch

        from mobius._configs import ArchitectureConfig
        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        config = ArchitectureConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
        stem = "model.layers.0.self_attn"
        fused_weight = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
        fused_bias = torch.arange(16, dtype=torch.float32)
        normalized = _normalize_gguf_weights(
            {
                f"{stem}.qkv_proj.weight": fused_weight,
                f"{stem}.qkv_proj.bias": fused_bias,
            },
            gguf_arch=architecture,
            config=config,
        )

        expected_widths = {"q_proj": (0, 8), "k_proj": (8, 12), "v_proj": (12, 16)}
        for projection, (start, end) in expected_widths.items():
            assert torch.equal(
                normalized[f"{stem}.{projection}.weight"],
                fused_weight[start:end],
            )
            assert torch.equal(
                normalized[f"{stem}.{projection}.bias"],
                fused_bias[start:end],
            )

        with pytest.raises(ValueError, match=r"Invalid fused .* QKV weight width"):
            _normalize_gguf_weights(
                {f"{stem}.qkv_proj.weight": fused_weight[:-1]},
                gguf_arch=architecture,
                config=config,
            )


class TestLanguageDiffusionDispatch:
    @staticmethod
    def _patch_dream_reader(monkeypatch) -> None:
        class _DreamGGUF:
            architecture = "dream"
            metadata: ClassVar[dict] = {
                "dream.embedding_length": 8,
                "dream.feed_forward_length": 16,
                "dream.block_count": 1,
                "dream.attention.head_count": 2,
                "dream.attention.head_count_kv": 1,
                "dream.context_length": 32,
                "dream.rope.dimension_count": 4,
                "dream.vocab_size": 16,
                "dream.attention.layer_norm_rms_epsilon": 1e-5,
                "tokenizer.ggml.mask_token_id": 15,
            }
            tensor_names: ClassVar[list[str]] = [
                "token_embd.weight",
                "output_norm.weight",
                "output.weight",
                "blk.0.attn_norm.weight",
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.attn_output.weight",
                "blk.0.ffn_norm.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_down.weight",
                "blk.0.ffn_up.weight",
            ]

            def __init__(self, _path):
                pass

            def get_metadata(self, key, default=None):
                return self.metadata.get(key, default)

        import mobius._builder
        import mobius.integrations.gguf._builder as builder
        import mobius.integrations.gguf._shard_set as shard_set

        monkeypatch.setattr(builder, "_resolve_gguf_path", lambda path: path)
        monkeypatch.setattr(builder, "_validate_gguf_model", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(builder, "_has_quantized_weights", lambda *_args: False)
        monkeypatch.setattr(shard_set, "open_gguf_model", _DreamGGUF)
        monkeypatch.setattr(
            mobius._builder,
            "build_from_module",
            lambda *_args, **_kwargs: pytest.fail("graph construction must not run"),
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"task": "text-generation"}, "only supports task='masked-diffusion'"),
            ({"static_cache": True}, "has no KV cache"),
        ],
    )
    def test_invalid_generation_contract_is_rejected_before_graph_build(
        self, monkeypatch, kwargs, message
    ):
        from mobius.integrations.gguf import build_from_gguf

        self._patch_dream_reader(monkeypatch)
        with pytest.raises(ValueError, match=message):
            build_from_gguf("unused.gguf", **kwargs)


class TestEncoderGGUFBuild:
    """Encoder GGUFs dispatch to feature extraction and preserve token outputs."""

    @staticmethod
    def _run(model, sequence_length: int, masked: bool = False) -> np.ndarray:
        from mobius._testing.ort_inference import OnnxModelSession

        mask = np.ones((1, sequence_length), dtype=np.int64)
        if masked:
            mask[0, -1] = 0
        feeds = {
            "input_ids": np.arange(sequence_length, dtype=np.int64)[None, :],
            "attention_mask": mask,
            "token_type_ids": np.zeros((1, sequence_length), dtype=np.int64),
        }
        session = OnnxModelSession(model)
        try:
            outputs = session.run(feeds)
        finally:
            session.close()
        assert set(outputs) == {"last_hidden_state"}
        return outputs["last_hidden_state"]

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    def test_float_build_save_load_and_variable_masks(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-f32.gguf"
        _write_encoder_gguf(path, architecture)
        package = build_from_gguf(path)
        model = package["model"]
        assert {value.name for value in model.graph.outputs} == {"last_hidden_state"}
        assert not any(
            marker in value.name
            for value in (*model.graph.inputs, *model.graph.outputs)
            for marker in ("past_key_values", "present", "logits")
        )

        saved = tmp_path / f"{architecture}-saved"
        package.save(saved, progress_bar=False)
        reloaded = ModelPackage.load(saved)
        for sequence_length in (1, 3, 7):
            output = self._run(reloaded["model"], sequence_length, masked=sequence_length > 1)
            assert output.shape == (1, sequence_length, 64)
            assert np.isfinite(output).all()

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    def test_quantized_source_preserves_linears_but_dequantizes_embedding(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / f"{architecture}-q4.gguf"
        _write_encoder_gguf(path, architecture, quantized=True)
        source = GGUFModel(path)
        token_qtype = next(
            qtype
            for name, _raw, qtype, _shape in source.tensor_items_raw()
            if name == "token_embd.weight"
        )
        assert token_qtype == GGMLQuantizationType.Q4_0

        preserved = build_from_gguf(path, keep_quantized=True)["model"]
        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]

        op_types = {node.op_type for node in preserved.graph}
        assert "MatMulNBits" in op_types
        assert "GatherBlockQuantized" not in op_types
        actual = self._run(preserved, 5, masked=True)
        expected = self._run(explicit_float, 5, masked=True)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-2)

    def test_float_fused_bert_qkv_splits_losslessly_and_runs(self, tmp_path: Path) -> None:
        import torch

        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._builder import (
            _load_dequantized_state_dict,
            _normalize_gguf_weights,
        )
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / "bert-fused-qkv.gguf"
        _write_encoder_gguf(path, "bert", fused_qkv=True)
        source = GGUFModel(path)
        mapped = _load_dequantized_state_dict(source, "bert")
        fused_weight = mapped["bert.encoder.layer.0.attention.self.qkv.weight"]
        fused_bias = mapped["bert.encoder.layer.0.attention.self.qkv.bias"]
        normalized = _normalize_gguf_weights(
            mapped,
            "bert",
            SimpleNamespace(hidden_size=64),
        )

        for index, projection in enumerate(("query", "key", "value")):
            stem = f"bert.encoder.layer.0.attention.self.{projection}"
            assert torch.equal(
                normalized[f"{stem}.weight"],
                fused_weight[index * 64 : (index + 1) * 64],
            )
            assert torch.equal(
                normalized[f"{stem}.bias"],
                fused_bias[index * 64 : (index + 1) * 64],
            )
        assert not any(".qkv." in name for name in normalized)

        model = build_from_gguf(path)["model"]
        output = self._run(model, 5, masked=True)
        assert output.shape == (1, 5, 64)
        assert np.isfinite(output).all()

    def test_float_fused_bert_qkv_repacks_in_mixed_quantized_file(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "bert-fused-qkv-mixed.gguf"
        _write_encoder_gguf(
            path,
            "bert",
            quantized=True,
            fused_qkv=True,
            fused_qkv_float=True,
        )
        preserved = build_from_gguf(path, keep_quantized=True)["model"]
        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]

        assert "MatMulNBits" in {node.op_type for node in preserved.graph}
        actual = self._run(preserved, 5, masked=True)
        expected = self._run(explicit_float, 5, masked=True)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-2)

    @pytest.mark.parametrize(
        ("quantized", "width_delta", "message"),
        [
            (False, 1, "invalid encoder tensor shape"),
            (True, 0, "cannot be split losslessly"),
        ],
    )
    def test_invalid_fused_bert_qkv_fails_before_graph_build(
        self,
        tmp_path: Path,
        quantized: bool,
        width_delta: int,
        message: str,
        monkeypatch,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "bert-invalid-fused-qkv.gguf"
        _write_encoder_gguf(
            path,
            "bert",
            quantized=quantized,
            fused_qkv=True,
            fused_width_delta=width_delta,
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=message):
            build_from_gguf(path)

    def test_fused_bert_qkv_requires_matching_bias(self, tmp_path: Path, monkeypatch) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "bert-fused-qkv-missing-bias.gguf"
        _write_encoder_gguf(
            path,
            "bert",
            fused_qkv=True,
            omit="blk.0.attn_qkv.bias",
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"attn_qkv.bias"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    @pytest.mark.parametrize("kv_heads", [None, 4])
    def test_encoder_kv_heads_default_to_query_heads_or_accept_equality(
        self, tmp_path: Path, architecture: str, kv_heads: int | None
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / f"{architecture}-kv-{kv_heads}.gguf"
        _write_encoder_gguf(path, architecture, kv_heads=kv_heads)
        config = gguf_to_config(GGUFModel(path))
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 4

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    def test_encoder_gqa_rejected_before_graph_build(
        self, tmp_path: Path, architecture: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-gqa.gguf"
        _write_encoder_gguf(path, architecture, kv_heads=2)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match="grouped-query attention is not supported"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    def test_pooling_classifier_and_task_overrides_are_rejected(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        pooled = tmp_path / f"{architecture}-pooled.gguf"
        _write_encoder_gguf(pooled, architecture, pooling_type=1)
        with pytest.raises(ValueError, match="pooling_type"):
            build_from_gguf(pooled)

        headed = tmp_path / f"{architecture}-head.gguf"
        _write_encoder_gguf(headed, architecture, include_head=True)
        with pytest.raises(ValueError, match="silently discard encoder heads"):
            build_from_gguf(headed)

        plain = tmp_path / f"{architecture}-task.gguf"
        _write_encoder_gguf(plain, architecture)
        with pytest.raises(ValueError, match="feature-extraction"):
            build_from_gguf(plain, task="text-generation")
        with pytest.raises(ValueError, match="encoder-only"):
            build_from_gguf(plain, static_cache=True)

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    @pytest.mark.parametrize("failure", ["missing", "rank"])
    def test_missing_and_malformed_tensors_fail_before_graph_build(
        self, tmp_path: Path, architecture: str, failure: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        target = "blk.0.attn_q.weight" if architecture == "bert" else "blk.0.attn_qkv.weight"
        path = tmp_path / f"{architecture}-{failure}.gguf"
        _write_encoder_gguf(
            path,
            architecture,
            omit=target if failure == "missing" else None,
            malformed=target if failure == "rank" else None,
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"missing required|invalid encoder tensor shape"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    @pytest.mark.parametrize("suffix", ["scale", "input_scale"])
    def test_auxiliary_quantization_sidecars_are_never_dropped(
        self, tmp_path: Path, architecture: str, suffix: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{suffix}.gguf"
        _write_encoder_gguf(path, architecture, auxiliary_suffix=suffix)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"scale/input_scale"):
            build_from_gguf(path)


class TestT5GGUFBuild:
    """T5 GGUFs preserve encoder-only and seq2seq task contracts."""

    @staticmethod
    def _run_encoder(model, sequence_length: int, masked: bool = False) -> np.ndarray:
        from mobius._testing.ort_inference import OnnxModelSession

        mask = np.ones((1, sequence_length), dtype=np.int64)
        if masked:
            mask[0, -1] = 0
        session = OnnxModelSession(model)
        try:
            outputs = session.run(
                {
                    "input_ids": np.arange(sequence_length, dtype=np.int64)[None, :],
                    "attention_mask": mask,
                }
            )
        finally:
            session.close()
        assert set(outputs) == {"last_hidden_state"}
        return outputs["last_hidden_state"]

    @staticmethod
    def _run_seq2seq(package) -> tuple[np.ndarray, np.ndarray]:
        from mobius._testing.ort_inference import OnnxModelSession

        encoder = OnnxModelSession(package["encoder"])
        decoder = OnnxModelSession(package["decoder"])
        try:
            encoder_mask = np.array([[1, 1, 0]], dtype=np.int64)
            encoder_outputs = encoder.run(
                {
                    "input_ids": np.array([[2, 3, 0]], dtype=np.int64),
                    "attention_mask": encoder_mask,
                }
            )
            hidden = encoder_outputs["last_hidden_state"]
            prefill_feeds = {
                "input_ids": np.array([[0]], dtype=np.int64),
                "encoder_hidden_states": hidden,
                "attention_mask": np.ones((1, 1), dtype=np.int64),
                "encoder_attention_mask": encoder_mask,
            }
            for layer in range(2):
                for attention in ("self", "cross"):
                    for kind in ("key", "value"):
                        prefill_feeds[f"past_key_values.{layer}.{attention}.{kind}"] = (
                            np.zeros((1, 4, 0, 8), dtype=np.float32)
                        )
            prefill = decoder.run(prefill_feeds)

            decode_feeds = {
                "input_ids": np.array([[1]], dtype=np.int64),
                "encoder_hidden_states": np.zeros((1, 0, 32), dtype=np.float32),
                "attention_mask": np.ones((1, 2), dtype=np.int64),
                "encoder_attention_mask": encoder_mask,
            }
            for layer in range(2):
                for attention in ("self", "cross"):
                    for kind in ("key", "value"):
                        decode_feeds[f"past_key_values.{layer}.{attention}.{kind}"] = prefill[
                            f"present.{layer}.{attention}.{kind}"
                        ]
            decode = decoder.run(decode_feeds)
        finally:
            encoder.close()
            decoder.close()

        assert prefill["present.1.self.key"].shape == (1, 4, 1, 8)
        assert prefill["present.1.cross.key"].shape == (1, 4, 3, 8)
        assert decode["present.1.self.key"].shape == (1, 4, 2, 8)
        assert decode["present.1.cross.key"].shape == (1, 4, 3, 8)
        return prefill["logits"], decode["logits"]

    def test_t5encoder_float_save_load_and_variable_masks(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "t5encoder-f32.gguf"
        _write_t5_gguf(path, "t5encoder")
        package = build_from_gguf(path)
        assert set(package) == {"model"}
        assert {value.name for value in package["model"].graph.outputs} == {
            "last_hidden_state"
        }
        assert not any(
            marker in value.name
            for value in (*package["model"].graph.inputs, *package["model"].graph.outputs)
            for marker in ("past_key_values", "present", "logits")
        )

        saved = tmp_path / "t5encoder-saved"
        package.save(saved, progress_bar=False)
        reloaded = ModelPackage.load(saved)
        for sequence_length in (1, 3, 7):
            output = self._run_encoder(
                reloaded["model"], sequence_length, masked=sequence_length > 1
            )
            assert output.shape == (1, sequence_length, 32)
            assert np.isfinite(output).all()

    def test_t5encoder_quantized_projection_matches_dequantized(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "t5encoder-q4.gguf"
        _write_t5_gguf(path, "t5encoder", quantized=True)
        preserved = build_from_gguf(path, keep_quantized=True)["model"]
        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]

        op_types = {node.op_type for node in preserved.graph}
        assert "MatMulNBits" in op_types
        assert "GatherBlockQuantized" not in op_types
        actual = self._run_encoder(preserved, 5, masked=True)
        expected = self._run_encoder(explicit_float, 5, masked=True)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-2)

    def test_relative_bias_values_keep_bucket_head_orientation(self, tmp_path: Path) -> None:
        import torch

        from mobius.integrations.gguf._builder import _load_dequantized_state_dict
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / "t5encoder-relative-bias.gguf"
        _write_t5_gguf(path, "t5encoder")
        state_dict = _load_dequantized_state_dict(GGUFModel(path), "t5encoder")
        expected = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        assert torch.equal(
            state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
            expected,
        )

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    def test_gated_activation_ambiguity_fails_before_graph_build(
        self, tmp_path: Path, architecture: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-gated.gguf"
        _write_t5_gguf(path, architecture, gated=True)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"gated FFNs are ambiguous"):
            build_from_gguf(path)

    def test_t5_builds_distinct_encoder_and_decoder_contracts(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "t5-f32.gguf"
        _write_t5_gguf(path, "t5", encoder_layers=1, decoder_layers=2)
        package = build_from_gguf(path)
        assert set(package) == {"encoder", "decoder"}
        decoder_inputs = {value.name for value in package["decoder"].graph.inputs}
        assert {"encoder_hidden_states", "encoder_attention_mask"} <= decoder_inputs
        assert "past_key_values.1.cross.key" in decoder_inputs
        assert {value.name for value in package["decoder"].graph.outputs} >= {
            "logits",
            "present.1.self.key",
            "present.1.cross.key",
        }
        prefill_logits, decode_logits = self._run_seq2seq(package)
        assert prefill_logits.shape == decode_logits.shape == (1, 1, 64)
        assert np.isfinite(prefill_logits).all()
        assert np.isfinite(decode_logits).all()

    def test_t5_quantized_prefill_and_decode_match_dequantized(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "t5-q4.gguf"
        _write_t5_gguf(
            path,
            "t5",
            quantized=True,
            encoder_layers=1,
            decoder_layers=2,
        )
        preserved = build_from_gguf(path, keep_quantized=True)
        explicit_float = build_from_gguf(path, keep_quantized=False)
        assert "MatMulNBits" in {
            node.op_type for model in preserved.values() for node in model.graph
        }

        actual_prefill, actual_decode = self._run_seq2seq(preserved)
        expected_prefill, expected_decode = self._run_seq2seq(explicit_float)
        np.testing.assert_allclose(actual_prefill, expected_prefill, rtol=0, atol=1e-2)
        np.testing.assert_allclose(actual_decode, expected_decode, rtol=0, atol=1e-2)

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    def test_task_and_static_cache_misdispatch_is_rejected(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-task.gguf"
        _write_t5_gguf(path, architecture)
        with pytest.raises(ValueError, match="only supports task"):
            build_from_gguf(path, task="text-generation")
        with pytest.raises(ValueError, match="static_cache"):
            build_from_gguf(path, static_cache=True)

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    @pytest.mark.parametrize("failure", ["missing", "rank"])
    def test_missing_and_malformed_tensors_fail_before_graph_build(
        self,
        tmp_path: Path,
        architecture: str,
        failure: str,
        monkeypatch,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        target = "enc.blk.0.attn_q.weight"
        path = tmp_path / f"{architecture}-{failure}.gguf"
        _write_t5_gguf(
            path,
            architecture,
            omit=target if failure == "missing" else None,
            malformed=target if failure == "rank" else None,
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"missing required|invalid T5 tensor shape"):
            build_from_gguf(path)

    @pytest.mark.parametrize(
        ("architecture", "tensor_name"),
        [
            ("t5encoder", "enc.blk.1.attn_q.weight"),
            ("t5", "dec.blk.1.cross_attn_q.weight"),
            ("t5", "enc.blk.-1.attn_q.weight"),
            ("t5", "dec.blk.+1.cross_attn_q.weight"),
            ("t5encoder", "enc.blk.01.attn_q.weight"),
            ("t5encoder", "enc.blk.1\u0660.attn_q.weight"),
        ],
    )
    def test_invalid_layer_indices_fail_before_graph_build(
        self,
        tmp_path: Path,
        architecture: str,
        tensor_name: str,
        monkeypatch,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-invalid-layer.gguf"
        _write_t5_gguf(path, architecture, extra_layer_tensor=tensor_name)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"invalid T5 layer tensor index"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    def test_exact_pinned_ignored_tensors_remain_accepted(
        self, tmp_path: Path, architecture: str, caplog
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-ignored.gguf"
        _write_t5_gguf(path, architecture, include_ignored_tensor=True)
        build_from_gguf(path)
        assert "Ignoring pinned llama.cpp T5 tensor" in caplog.text

    def test_cross_relative_bias_ignore_is_not_a_suffix_wildcard(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "t5-broad-ignore.gguf"
        _write_t5_gguf(
            path,
            "t5",
            extra_layer_tensor="dec.blk.0.extra.cross_attn_rel_b.weight",
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"suffixes do not match"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    @pytest.mark.parametrize("suffix", ["scale", "input_scale"])
    def test_auxiliary_quantization_sidecars_are_never_dropped(
        self, tmp_path: Path, architecture: str, suffix: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{suffix}.gguf"
        _write_t5_gguf(path, architecture, auxiliary_suffix=suffix)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"scale/input_scale"):
            build_from_gguf(path)


class TestHybridGGUFBuild:
    @staticmethod
    def _run_lfm2(model) -> list[dict[str, np.ndarray]]:
        from mobius._testing.ort_inference import OnnxModelSession

        session = OnnxModelSession(model)
        states = {
            "past_key_values.0.conv_state": np.zeros((1, 32, 2), dtype=np.float32),
            "past_key_values.1.key": np.zeros((1, 2, 0, 8), dtype=np.float32),
            "past_key_values.1.value": np.zeros((1, 2, 0, 8), dtype=np.float32),
        }
        outputs = []
        try:
            for tokens in ([1, 2, 3], [4]):
                total = states["past_key_values.1.key"].shape[2] + len(tokens)
                output = session.run(
                    {
                        "input_ids": np.asarray([tokens], dtype=np.int64),
                        "attention_mask": np.ones((1, total), dtype=np.int64),
                        "position_ids": np.arange(total - len(tokens), total, dtype=np.int64)[
                            None
                        ],
                        **states,
                    }
                )
                outputs.append(output)
                states = {
                    "past_key_values.0.conv_state": output["present.0.conv_state"],
                    "past_key_values.1.key": output["present.1.key"],
                    "past_key_values.1.value": output["present.1.value"],
                }
        finally:
            session.close()
        return outputs

    @staticmethod
    def _run_qwen35(model) -> list[dict[str, np.ndarray]]:
        from mobius._testing.ort_inference import OnnxModelSession

        session = OnnxModelSession(model)
        states = {}
        for layer in range(3):
            states[f"past_key_values.{layer}.conv_state"] = np.zeros(
                (1, 512, 2), dtype=np.float32
            )
            states[f"past_key_values.{layer}.recurrent_state"] = np.zeros(
                (1, 4, 64, 64), dtype=np.float32
            )
        states["past_key_values.3.key"] = np.zeros((1, 2, 0, 8), dtype=np.float32)
        states["past_key_values.3.value"] = np.zeros((1, 2, 0, 8), dtype=np.float32)
        outputs = []
        try:
            for tokens in ([1, 2, 3], [4]):
                total = states["past_key_values.3.key"].shape[2] + len(tokens)
                output = session.run(
                    {
                        "input_ids": np.asarray([tokens], dtype=np.int64),
                        "attention_mask": np.ones((1, total), dtype=np.int64),
                        "position_ids": np.arange(total - len(tokens), total, dtype=np.int64)[
                            None
                        ],
                        **states,
                    }
                )
                outputs.append(output)
                for layer in range(3):
                    states[f"past_key_values.{layer}.conv_state"] = output[
                        f"present.{layer}.conv_state"
                    ]
                    states[f"past_key_values.{layer}.recurrent_state"] = output[
                        f"present.{layer}.recurrent_state"
                    ]
                states["past_key_values.3.key"] = output["present.3.key"]
                states["past_key_values.3.value"] = output["present.3.value"]
        finally:
            session.close()
        return outputs

    def test_lfm2_float_prefill_decode_threads_only_mixer_specific_state(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2-f32.gguf"
        _write_lfm2_gguf(path, quantized=False)
        model = build_from_gguf(path)["model"]

        assert [value.name for value in model.graph.inputs if "past_" in value.name] == [
            "past_key_values.0.conv_state",
            "past_key_values.1.key",
            "past_key_values.1.value",
        ]
        outputs = self._run_lfm2(model)
        assert outputs[0]["present.0.conv_state"].shape == (1, 32, 2)
        assert outputs[0]["present.1.key"].shape == (1, 2, 3, 8)
        assert outputs[1]["present.1.key"].shape == (1, 2, 4, 8)
        assert all(np.isfinite(output["logits"]).all() for output in outputs)

    def test_lfm2_quantized_preservation_fails_closed_and_float_import_executes(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2-q4.gguf"
        _write_lfm2_gguf(path, quantized=True)
        with pytest.raises(
            ValueError,
            match=r"Cannot keep Q4_0 projection .*MatMulNBits",
        ):
            build_from_gguf(path, keep_quantized=True)

        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]

        assert "MatMulNBits" not in {node.op_type for node in explicit_float.graph}
        outputs = self._run_lfm2(explicit_float)
        assert outputs[0]["present.0.conv_state"].shape == (1, 32, 2)
        assert outputs[1]["present.1.key"].shape == (1, 2, 4, 8)
        assert all(np.isfinite(output["logits"]).all() for output in outputs)

    def test_lfm2moe_float_prefill_decode_threads_mixed_state(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2moe-f32.gguf"
        _write_lfm2moe_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]

        assert [value.name for value in model.graph.inputs if "past_" in value.name] == [
            "past_key_values.0.conv_state",
            "past_key_values.1.key",
            "past_key_values.1.value",
        ]
        outputs = self._run_lfm2(model)
        assert outputs[0]["present.0.conv_state"].shape == (1, 32, 2)
        assert outputs[0]["present.1.key"].shape == (1, 2, 3, 8)
        assert outputs[1]["present.1.key"].shape == (1, 2, 4, 8)
        assert all(np.isfinite(output["logits"]).all() for output in outputs)

        saved = tmp_path / "saved-lfm2moe"
        package.save(str(saved), progress_bar=False, check_weights=True)
        reloaded = ModelPackage.load(str(saved))["model"]
        reloaded_outputs = self._run_lfm2(reloaded)
        for actual, expected in zip(reloaded_outputs, outputs):
            for name in actual:
                np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=0)

    def test_lfm2moe_quantized_source_requires_explicit_dequantization(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2moe-q4.gguf"
        _write_lfm2moe_gguf(path, quantized=True)
        with pytest.raises(ValueError, match="keep_quantized=False"):
            build_from_gguf(path)

        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]
        outputs = self._run_lfm2(explicit_float)
        assert all(np.isfinite(output["logits"]).all() for output in outputs)

    def test_lfm2moe_state_rollback_and_batch_reorder(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2moe-state.gguf"
        _write_lfm2moe_gguf(path, quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        try:
            prefill = session.run(
                {
                    "input_ids": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
                    "attention_mask": np.ones((2, 2), dtype=np.int64),
                    "position_ids": np.asarray([[0, 1], [0, 1]], dtype=np.int64),
                    "past_key_values.0.conv_state": np.zeros((2, 32, 2), dtype=np.float32),
                    "past_key_values.1.key": np.zeros((2, 2, 0, 8), dtype=np.float32),
                    "past_key_values.1.value": np.zeros((2, 2, 0, 8), dtype=np.float32),
                }
            )
            snapshot = {
                "past_key_values.0.conv_state": prefill["present.0.conv_state"],
                "past_key_values.1.key": prefill["present.1.key"],
                "past_key_values.1.value": prefill["present.1.value"],
            }

            def decode(tokens: list[list[int]], states: dict[str, np.ndarray]):
                return session.run(
                    {
                        "input_ids": np.asarray(tokens, dtype=np.int64),
                        "attention_mask": np.ones((2, 3), dtype=np.int64),
                        "position_ids": np.asarray([[2], [2]], dtype=np.int64),
                        **states,
                    }
                )

            first = decode([[5], [6]], snapshot)
            replayed = decode([[5], [6]], snapshot)
            reordered = decode(
                [[6], [5]],
                {name: value[[1, 0]] for name, value in snapshot.items()},
            )
        finally:
            session.close()

        for name in first:
            np.testing.assert_allclose(replayed[name], first[name], rtol=0, atol=0)
            np.testing.assert_allclose(reordered[name], first[name][[1, 0]], rtol=0, atol=0)

    def test_qwen35_float_and_quantized_prefill_decode_thread_mixed_state(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        float_path = tmp_path / "qwen35-f32.gguf"
        quantized_path = tmp_path / "qwen35-q4.gguf"
        _write_qwen35_gguf(float_path, quantized=False)
        _write_qwen35_gguf(quantized_path, quantized=True)

        float_steps = self._run_qwen35(build_from_gguf(float_path)["model"])
        quantized_model = build_from_gguf(quantized_path, keep_quantized=True)["model"]
        quantized_steps = self._run_qwen35(quantized_model)

        assert "MatMulNBits" in {node.op_type for node in quantized_model.graph}
        for steps in (float_steps, quantized_steps):
            assert steps[0]["present.0.conv_state"].shape == (1, 512, 2)
            assert steps[0]["present.0.recurrent_state"].shape == (1, 4, 64, 64)
            assert steps[0]["present.3.key"].shape == (1, 2, 3, 8)
            assert steps[1]["present.3.key"].shape == (1, 2, 4, 8)
            assert all(np.isfinite(output["logits"]).all() for output in steps)

    def test_qwen35_tied_quantized_embedding_owns_head_and_saves(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "qwen35-tied-embedding-q4.gguf"
        _write_qwen35_gguf(path, quantized=True, quantized_embedding=True)
        package = build_from_gguf(path, keep_quantized=True)
        model = package["model"]

        initializer_names = set(model.graph.initializers)
        assert {
            "model.embed_tokens.qweight",
            "model.embed_tokens.scales",
            "model.embed_tokens.zero_points",
        } <= initializer_names
        assert not any(name.startswith("lm_head.") for name in initializer_names)
        assert all(
            value.const_value is not None for value in model.graph.initializers.values()
        )
        op_types = {node.op_type for node in model.graph}
        assert "GatherBlockQuantized" in op_types
        assert "MatMulNBits" in op_types

        saved = tmp_path / "saved-qwen35"
        package.save(str(saved), progress_bar=False, check_weights=True)
        reloaded = ModelPackage.load(str(saved))["model"]
        assert set(reloaded.graph.initializers) == initializer_names
        assert all(
            value.const_value is not None for value in reloaded.graph.initializers.values()
        )

    def test_qwen35_packed_v_head_reorder_fails_closed_when_not_repackable(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "qwen35-narrow-q4.gguf"
        _write_qwen35_gguf(path, quantized=True, inner_size=32)
        with pytest.raises(ValueError, match=r"cannot map .* quant blocks"):
            build_from_gguf(path, keep_quantized=True)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"static_cache": True}, "architecture-specific"),
            ({"task": "text-generation"}, "hybrid-text-generation"),
        ],
    )
    def test_lfm2_cache_task_misdispatch_is_rejected(
        self, tmp_path: Path, kwargs: dict, message: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2-misdispatch.gguf"
        _write_lfm2_gguf(path, quantized=False)
        with pytest.raises(ValueError, match=re.escape(message)):
            build_from_gguf(path, **kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"static_cache": True}, "architecture-specific"),
            ({"task": "text-generation"}, "hybrid-text-generation"),
        ],
    )
    def test_lfm2moe_cache_task_misdispatch_is_rejected(
        self, tmp_path: Path, kwargs: dict, message: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2moe-misdispatch.gguf"
        _write_lfm2moe_gguf(path, quantized=False)
        with pytest.raises(ValueError, match=re.escape(message)):
            build_from_gguf(path, **kwargs)


class TestRecurrentGGUFBuild:
    """Mamba GGUF imports preserve recurrent state and tensor-role semantics."""

    @staticmethod
    def _run_steps(model, architecture: str) -> list[dict[str, np.ndarray]]:
        from mobius._testing.ort_inference import OnnxModelSession

        conv_channels = 64 if architecture == "mamba" else 80
        ssm_shape = (1, 64, 8) if architecture == "mamba" else (1, 4, 8, 16)
        states = {
            "past_states.0.conv_state": np.zeros((1, conv_channels, 3), dtype=np.float32),
            "past_states.0.ssm_state": np.zeros(ssm_shape, dtype=np.float32),
        }
        token_groups = [[1], [2], [3], [4]]
        if architecture == "mamba2":
            token_groups = [[1, 2, 3], [4]]

        session = OnnxModelSession(model)
        outputs = []
        try:
            for tokens in token_groups:
                out = session.run(
                    {
                        "input_ids": np.asarray([tokens], dtype=np.int64),
                        **states,
                    }
                )
                outputs.append(out)
                states = {
                    "past_states.0.conv_state": out["present.0.conv_state"],
                    "past_states.0.ssm_state": out["present.0.ssm_state"],
                }
        finally:
            session.close()
        return outputs

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_float_prefill_decode_state_threading_and_save_load(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-f32.gguf"
        _write_recurrent_gguf(path, architecture, quantized=False)
        package = build_from_gguf(path)

        output_dir = tmp_path / f"{architecture}-saved"
        package.save(output_dir, progress_bar=False)
        reloaded = ModelPackage.load(output_dir)
        outputs = self._run_steps(reloaded["model"], architecture)

        assert all(np.isfinite(out["logits"]).all() for out in outputs)
        assert np.count_nonzero(outputs[-1]["present.0.conv_state"]) > 0
        assert np.count_nonzero(outputs[-1]["present.0.ssm_state"]) > 0

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_quantized_projection_preservation_is_rejected_and_float_import_executes(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-q4.gguf"
        _write_recurrent_gguf(path, architecture, quantized=True)
        with pytest.raises(ValueError, match="does not support keep_quantized=True"):
            build_from_gguf(path, keep_quantized=True)

        # Current Mamba projection consumers are ordinary Linear modules.
        # Explicit float import dequantizes the GGUF source before stateful execution.
        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]
        assert "MatMulNBits" not in {node.op_type for node in explicit_float.graph}
        float_steps = self._run_steps(explicit_float, architecture)
        assert all(np.isfinite(out["logits"]).all() for out in float_steps)
        assert np.count_nonzero(float_steps[-1]["present.0.conv_state"]) > 0
        assert np.count_nonzero(float_steps[-1]["present.0.ssm_state"]) > 0

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_static_kv_cache_is_rejected(self, tmp_path: Path, architecture: str) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-static.gguf"
        _write_recurrent_gguf(path, architecture, quantized=False)
        with pytest.raises(ValueError, match="conv_state and ssm_state"):
            build_from_gguf(path, static_cache=True)

    def test_mamba1_graph_rejects_multi_token_input(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "mamba-multitoken.gguf"
        _write_recurrent_gguf(path, "mamba", quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        try:
            with pytest.raises(
                ort.capi.onnxruntime_pybind11_state.InvalidArgument,
                match=r"dimension|Expected",
            ):
                session.run(
                    {
                        "input_ids": np.asarray([[1, 2]], dtype=np.int64),
                        "past_states.0.conv_state": np.zeros((1, 64, 3), np.float32),
                        "past_states.0.ssm_state": np.zeros((1, 64, 8), np.float32),
                    }
                )
        finally:
            session.close()

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_malformed_conv_shape_is_not_silently_loaded(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-bad-conv.gguf"
        _write_recurrent_gguf(path, architecture, quantized=False, malformed_conv=True)
        with pytest.raises(ValueError, match="shape"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_malformed_recurrent_suffix_is_rejected_before_graph_build(
        self, tmp_path: Path, architecture: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-bad-suffix.gguf"
        _write_recurrent_gguf(
            path,
            architecture,
            quantized=False,
            malformed_suffix=True,
        )

        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match="suffixes do not match"):
            build_from_gguf(path)
        assert not graph_build_started

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_auxiliary_quantization_sidecar_is_rejected_before_graph_build(
        self, tmp_path: Path, architecture: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-sidecar.gguf"
        _write_recurrent_gguf(path, architecture, quantized=False, auxiliary_scale=True)
        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match="scale/input_scale"):
            build_from_gguf(path)
        assert not graph_build_started


class TestFalconH1GGUFBuild:
    """Falcon-H1 GGUF import preserves its parallel graph and four-state ABI."""

    @staticmethod
    def _initial_inputs(batch: int, tokens: list[list[int]]) -> dict[str, np.ndarray]:
        sequence = len(tokens[0])
        return {
            "input_ids": np.asarray(tokens, dtype=np.int64),
            "position_ids": np.broadcast_to(
                np.arange(sequence, dtype=np.int64),
                (batch, sequence),
            ).copy(),
            "attention_mask": np.ones((batch, sequence), dtype=np.int64),
            "past_key_values.0.key": np.zeros((batch, 2, 0, 8), np.float32),
            "past_key_values.0.value": np.zeros((batch, 2, 0, 8), np.float32),
            "past_key_values.0.conv_state": np.zeros((batch, 48, 3), np.float32),
            "past_key_values.0.ssm_state": np.zeros((batch, 4, 8, 8), np.float32),
        }

    def test_float_prefill_decode_replay_batch_reorder_and_save_load(
        self, tmp_path: Path
    ) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "falcon-h1-f32.gguf"
        _write_falcon_h1_gguf(path, quantized=False)
        package = build_from_gguf(path)
        output_dir = tmp_path / "saved"
        package.save(output_dir, progress_bar=False)
        model = ModelPackage.load(output_dir)["model"]
        assert [value.name for value in model.graph.outputs] == [
            "logits",
            "present.0.key",
            "present.0.value",
            "present.0.conv_state",
            "present.0.ssm_state",
        ]

        session = OnnxModelSession(model)
        try:
            prefill = session.run(self._initial_inputs(2, [[1, 2, 3], [4, 5, 6]]))
            decode_inputs = {
                "input_ids": np.asarray([[7], [8]], np.int64),
                "position_ids": np.asarray([[3], [3]], np.int64),
                "attention_mask": np.ones((2, 4), np.int64),
                **{
                    name.replace("present.", "past_key_values."): value
                    for name, value in prefill.items()
                    if name.startswith("present.")
                },
            }
            first = session.run(decode_inputs)
            replay = session.run(decode_inputs)
            np.testing.assert_array_equal(first["logits"], replay["logits"])

            reordered = {
                name: value[::-1].copy() if value.shape[0] == 2 else value
                for name, value in decode_inputs.items()
            }
            reordered["attention_mask"] = decode_inputs["attention_mask"][::-1].copy()
            swapped = session.run(reordered)
            np.testing.assert_allclose(swapped["logits"], first["logits"][::-1])
        finally:
            session.close()

    def test_quantized_source_dequantizes_and_keep_quantized_fails_closed(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "falcon-h1-q4.gguf"
        _write_falcon_h1_gguf(path, quantized=True)
        model = build_from_gguf(path, keep_quantized=False)["model"]
        assert "MatMulNBits" not in {node.op_type for node in model.graph}
        with pytest.raises(ValueError, match="keep_quantized=True"):
            build_from_gguf(path, keep_quantized=True)

    def test_complete_optional_bias_families_select_the_biased_graph(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "falcon-h1-biased.gguf"
        _write_falcon_h1_gguf(path, quantized=False, biases=True)
        model = build_from_gguf(path)["model"]
        assert "model.layers.0.self_attn.q_proj.bias" in model.graph.initializers
        assert "model.layers.0.self_attn.o_proj.bias" in model.graph.initializers
        assert "model.layers.0.feed_forward.gate_proj.bias" in model.graph.initializers

    def test_cli_builds_dequantized_graph_package(self, tmp_path: Path) -> None:
        from mobius.__main__ import main

        path = tmp_path / "falcon-h1-cli.gguf"
        output_dir = tmp_path / "falcon-h1-cli"
        _write_falcon_h1_gguf(path, quantized=True)
        main(
            [
                "build-gguf",
                str(path),
                "--output",
                str(output_dir),
                "--dequantize",
            ]
        )
        assert (output_dir / "model.onnx").is_file()

    def test_static_cache_and_incomplete_closure_fail_before_graph(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        complete = tmp_path / "falcon-h1-static.gguf"
        _write_falcon_h1_gguf(complete, quantized=False)
        with pytest.raises(ValueError, match="four-state"):
            build_from_gguf(complete, static_cache=True)

        incomplete = tmp_path / "falcon-h1-incomplete.gguf"
        _write_falcon_h1_gguf(
            incomplete,
            quantized=False,
            omit="blk.0.ssm_out.weight",
        )
        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match="tensor closure"):
            build_from_gguf(incomplete)
        assert not graph_build_started

        partial_bias = tmp_path / "falcon-h1-partial-bias.gguf"
        _write_falcon_h1_gguf(
            partial_bias,
            quantized=False,
            partial_bias=True,
        )
        with pytest.raises(ValueError, match="biases must be present"):
            build_from_gguf(partial_bias)
        assert not graph_build_started


class TestPlamo2GGUFBuild:
    """PLaMo2 GGUF import preserves exact fused tensors and mixed state."""

    @staticmethod
    def _inputs(batch: int) -> dict[str, np.ndarray]:
        return {
            "input_ids": np.asarray([[1, 2], [3, 4]][:batch], np.int64),
            "position_ids": np.asarray([[0, 1], [0, 1]][:batch], np.int64),
            "attention_mask": np.ones((batch, 2), np.int64),
            "past_key_values.0.conv_state": np.zeros((batch, 32, 3), np.float32),
            "past_key_values.0.recurrent_state": np.zeros((batch, 4, 8, 4), np.float32),
            "past_key_values.1.key": np.zeros((batch, 2, 0, 8), np.float32),
            "past_key_values.1.value": np.zeros((batch, 2, 0, 8), np.float32),
        }

    def test_float_import_executes_and_round_trips(self, tmp_path: Path) -> None:
        from gguf import GGUFReader

        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-f32.gguf"
        _write_plamo2_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        source_tensors = {tensor.name: tensor.data for tensor in GGUFReader(path).tensors}
        norm_pairs = {
            "output_norm.weight": "model.norm.weight",
            "blk.0.attn_norm.weight": "model.layers.0.pre_mixer_norm.weight",
            "blk.0.post_attention_norm": "model.layers.0.post_mixer_norm.weight",
            "blk.0.ffn_norm.weight": "model.layers.0.pre_mlp_norm.weight",
            "blk.0.post_ffw_norm": "model.layers.0.post_mlp_norm.weight",
        }
        for source_name, initializer_name in norm_pairs.items():
            np.testing.assert_array_equal(
                model.graph.initializers[initializer_name].const_value.numpy(),
                source_tensors[source_name],
            )
        assert model.metadata_props["mobius.runtime_support"] == (
            "ORT GenAI 0.15.2 state ABI; package requires GQA-specialized attention"
        )
        assert [value.name for value in model.graph.outputs] == [
            "logits",
            "present.0.conv_state",
            "present.0.recurrent_state",
            "present.1.key",
            "present.1.value",
        ]
        output_dir = tmp_path / "saved"
        package.save(output_dir, progress_bar=False)
        session = OnnxModelSession(ModelPackage.load(output_dir)["model"])
        outputs = session.run(self._inputs(2))
        assert outputs["logits"].shape == (2, 2, 64)
        assert outputs["present.0.recurrent_state"].dtype == np.float32

    def test_legacy_scalar_heads_infer_exact_tensor_schedule(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-legacy-scalar-heads.gguf"
        _write_plamo2_gguf(path, quantized=False, legacy_scalar_heads=True)
        package = build_from_gguf(path)
        assert package.config.layer_types == ["mamba", "full_attention"]
        assert package.config.attention_head_counts == (0, 4)
        assert package.config.attention_kv_head_counts == (0, 2)

    def test_legacy_million_base_restores_reference_local_rope(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-legacy-rope.gguf"
        _write_plamo2_gguf(path, quantized=False, rope_theta=1_000_000.0)
        package = build_from_gguf(path)
        assert package.config.rope_theta == pytest.approx(10_000.0)

    def test_quantized_source_preserves_exact_projection_roles(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-q4.gguf"
        _write_plamo2_gguf(path, quantized=True)
        model = build_from_gguf(path, keep_quantized=True)["model"]
        assert sum(node.op_type == "MatMulNBits" for node in model.graph) == 9
        assert (
            model.graph.initializers["model.layers.0.mixer.A_log"].const_value.dtype
            == ir.DataType.FLOAT
        )
        assert (
            model.graph.initializers["model.layers.0.mixer.dt_proj.weight_t"].const_value.dtype
            == ir.DataType.FLOAT
        )
        assert (
            model.graph.initializers["model.layers.0.mixer.B_norm_weight"].const_value.dtype
            == ir.DataType.FLOAT
        )

    def test_quantized_tied_embedding_is_preserved_once(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-q4-embedding.gguf"
        _write_plamo2_gguf(path, quantized=True, quantized_embedding=True)
        model = build_from_gguf(path, keep_quantized=True)["model"]
        op_types = [node.op_type for node in model.graph]
        assert op_types.count("GatherBlockQuantized") == 1
        assert op_types.count("MatMulNBits") == 10
        assert (
            sum(name.endswith("embed_tokens.qweight") for name in model.graph.initializers)
            == 1
        )

    def test_quantized_untied_output_head_is_preserved(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-q4-output.gguf"
        _write_plamo2_gguf(path, quantized=True, include_output=True)
        model = build_from_gguf(path, keep_quantized=True)["model"]
        assert sum(node.op_type == "MatMulNBits" for node in model.graph) == 10
        assert "lm_head.weight" in model.graph.initializers
        assert "lm_head.scales" in model.graph.initializers

    def test_cli_builds_dequantized_package(self, tmp_path: Path) -> None:
        from mobius.__main__ import main

        source = tmp_path / "plamo2-cli.gguf"
        output = tmp_path / "plamo2-cli"
        _write_plamo2_gguf(source, quantized=True)
        main(["build-gguf", str(source), "--output", str(output), "--dequantize"])
        assert (output / "model.onnx").is_file()

    def test_static_cache_is_rejected(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-static.gguf"
        _write_plamo2_gguf(path, quantized=False)
        with pytest.raises(ValueError, match="heterogeneous"):
            build_from_gguf(path, static_cache=True)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omit": "blk.0.ssm_out.weight"}, "tensor closure"),
            ({"extra": "blk.0.unknown.weight"}, "tensor"),
            ({"invalid_decay": True}, "finite negative"),
            ({"group_count": 1}, "group_count=0"),
            ({"epsilon": 1e-5}, "rms_epsilon=1e-6"),
            ({"activation": "gelu"}, "fused SwiGLU"),
            ({"predefined_state": True}, "predefined initial state"),
            ({"head_counts": [4, 4]}, "head_count=4"),
            ({"kv_head_counts": [0]}, "match block_count"),
        ],
    )
    def test_malformed_sources_fail_before_graph(
        self,
        tmp_path: Path,
        monkeypatch,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-invalid.gguf"
        _write_plamo2_gguf(path, quantized=False, **kwargs)
        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path)
        assert not graph_build_started


class TestJambaGGUFBuild:
    """Jamba GGUF import preserves mixed mixers, routed experts, and state."""

    @staticmethod
    def _inputs(tokens: np.ndarray) -> dict[str, np.ndarray]:
        batch, sequence = tokens.shape
        return {
            "input_ids": tokens,
            "position_ids": np.broadcast_to(
                np.arange(sequence, dtype=np.int64),
                (batch, sequence),
            ).copy(),
            "attention_mask": np.ones((batch, sequence), np.int64),
            "past_key_values.0.conv_state": np.zeros((batch, 64, 3), np.float32),
            "past_key_values.0.ssm_state": np.zeros((batch, 64, 4), np.float32),
            "past_key_values.1.key": np.zeros((batch, 2, 0, 8), np.float32),
            "past_key_values.1.value": np.zeros((batch, 2, 0, 8), np.float32),
        }

    def test_float_import_preserves_expert_order_and_round_trips(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "jamba-f32.gguf"
        _write_jamba_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        assert model.metadata_props["mobius.runtime_support"].endswith(
            "onnxruntime/mobius#605"
        )
        assert [value.name for value in model.graph.outputs] == [
            "logits",
            "present.0.conv_state",
            "present.0.ssm_state",
            "present.1.key",
            "present.1.value",
        ]
        for expert in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                value = model.graph.initializers[
                    f"model.layers.1.feed_forward.experts.{expert}.{projection}.weight_t"
                ].const_value.numpy()
                np.testing.assert_array_equal(value, expert + 1)

        output_dir = tmp_path / "saved-jamba"
        package.save(output_dir, progress_bar=False)
        session = OnnxModelSession(ModelPackage.load(output_dir)["model"])
        outputs = session.run(self._inputs(np.asarray([[1, 2], [3, 4]], np.int64)))
        assert outputs["logits"].shape == (2, 2, 64)
        assert outputs["present.0.ssm_state"].dtype == np.float32

    def test_prefill_decode_reorder_and_snapshot_replay(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "jamba-state.gguf"
        _write_jamba_gguf(path, quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        histories = np.asarray([[1, 2], [3, 4]], np.int64)
        prefill = session.run(self._inputs(histories))

        order = np.asarray([1, 0], np.int64)
        next_tokens = np.asarray([[5], [6]], np.int64)
        decode_inputs = {
            "input_ids": next_tokens,
            "position_ids": np.full((2, 1), 2, np.int64),
            "attention_mask": np.ones((2, 3), np.int64),
            "past_key_values.0.conv_state": prefill["present.0.conv_state"][order],
            "past_key_values.0.ssm_state": prefill["present.0.ssm_state"][order],
            "past_key_values.1.key": prefill["present.1.key"][order],
            "past_key_values.1.value": prefill["present.1.value"][order],
        }
        decoded = session.run(decode_inputs)
        replayed = session.run(decode_inputs)
        for name in decoded:
            np.testing.assert_array_equal(decoded[name], replayed[name])

        full_tokens = np.concatenate([histories[order], next_tokens], axis=1)
        full = session.run(self._inputs(full_tokens))
        np.testing.assert_allclose(
            decoded["logits"][:, -1],
            full["logits"][:, -1],
            atol=2e-5,
            rtol=2e-5,
        )

    def test_quantized_source_keeps_only_compatible_matmul_roles(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "jamba-q4.gguf"
        _write_jamba_gguf(path, quantized=True)
        model = build_from_gguf(path, keep_quantized=True)["model"]
        # 4 attention + 3 dense FFN + 2 experts * 3 expert projections.
        assert sum(node.op_type == "MatMulNBits" for node in model.graph) == 13
        for stem in (
            "model.layers.0.mamba.in_proj.weight_t",
            "model.layers.0.mamba.conv1d.weight",
            "model.layers.0.mamba.ssm.x_proj.weight_t",
            "model.layers.0.mamba.ssm.dt_proj.weight_t",
            "model.layers.0.mamba.out_proj.weight_t",
        ):
            assert model.graph.initializers[stem].const_value.dtype == ir.DataType.FLOAT

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omit": "blk.1.ffn_up_exps.weight"}, "tensor closure"),
            ({"extra": "blk.0.ffn_gate_inp.weight"}, "tensor closure"),
            (
                {"malformed_shape": "blk.1.ffn_gate_exps.weight"},
                "tensor shape",
            ),
            ({"expert_count": 0, "expert_used_count": 1}, "expert_count"),
            ({"expert_count": 1, "expert_used_count": 1}, "not a routed-MoE"),
            ({"expert_count": 2, "expert_used_count": 3}, "expert_used_count"),
            ({"invalid_decay": True}, "finite negative"),
            ({"extra": "blk.0.ssm_in.scale"}, "auxiliary|tensor closure"),
        ],
    )
    def test_malformed_sources_fail_before_graph(
        self,
        tmp_path: Path,
        monkeypatch,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "jamba-invalid.gguf"
        _write_jamba_gguf(path, quantized=False, **kwargs)
        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path)
        assert not graph_build_started


class TestMiniMaxGGUFBuild:
    """MiniMax-01 GGUF import preserves hybrid state and expert tensor order."""

    @staticmethod
    def _inputs(tokens: np.ndarray, states: dict[str, np.ndarray] | None = None):
        batch, sequence = tokens.shape
        if states is None:
            states = {
                "past_key_values.0.recurrent_state": np.zeros(
                    (batch, 4, 16, 16), dtype=np.float32
                ),
                "past_key_values.1.key": np.zeros((batch, 2, 0, 16), dtype=np.float32),
                "past_key_values.1.value": np.zeros((batch, 2, 0, 16), dtype=np.float32),
            }
        past = states["past_key_values.1.key"].shape[2]
        return {
            "input_ids": tokens,
            "attention_mask": np.ones((batch, past + sequence), dtype=np.int64),
            "position_ids": np.broadcast_to(
                np.arange(past, past + sequence, dtype=np.int64), (batch, sequence)
            ).copy(),
            **states,
        }

    def test_float_import_runtime_save_reload_and_expert_order(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "minimax-f32.gguf"
        _write_minimax_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        assert [value.name for value in model.graph.inputs if "past_" in value.name] == [
            "past_key_values.0.recurrent_state",
            "past_key_values.1.key",
            "past_key_values.1.value",
        ]
        for layer in range(2):
            for expert in range(2):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}"
                np.testing.assert_array_equal(
                    model.graph.initializers[
                        f"{prefix}.gate_proj.weight_t"
                    ].const_value.numpy(),
                    11.0 + expert,
                )
                np.testing.assert_array_equal(
                    model.graph.initializers[f"{prefix}.up_proj.weight_t"].const_value.numpy(),
                    21.0 + expert,
                )
                np.testing.assert_array_equal(
                    model.graph.initializers[
                        f"{prefix}.down_proj.weight_t"
                    ].const_value.numpy(),
                    31.0 + expert,
                )

        session = OnnxModelSession(model)
        try:
            prefill = session.run(self._inputs(np.asarray([[1, 2], [3, 4]], np.int64)))
            snapshot = {
                "past_key_values.0.recurrent_state": prefill["present.0.recurrent_state"],
                "past_key_values.1.key": prefill["present.1.key"],
                "past_key_values.1.value": prefill["present.1.value"],
            }
            first = session.run(self._inputs(np.asarray([[5], [6]], np.int64), snapshot))
            replay = session.run(self._inputs(np.asarray([[5], [6]], np.int64), snapshot))
            reordered = session.run(
                self._inputs(
                    np.asarray([[6], [5]], np.int64),
                    {name: value[[1, 0]] for name, value in snapshot.items()},
                )
            )
        finally:
            session.close()
        for name in first:
            np.testing.assert_allclose(replay[name], first[name], rtol=0, atol=0)
            np.testing.assert_allclose(reordered[name], first[name][[1, 0]], rtol=0, atol=0)

        saved = tmp_path / "saved-minimax"
        package.save(saved, progress_bar=False, check_weights=True)
        reloaded = ModelPackage.load(saved)["model"]
        assert [value.name for value in reloaded.graph.outputs] == [
            value.name for value in model.graph.outputs
        ]

    def test_quantized_source_preserves_exact_projection_roles(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "minimax-q4.gguf"
        _write_minimax_gguf(path, quantized=True, quantized_embedding=True)
        model = build_from_gguf(path, keep_quantized=True)["model"]
        assert sum(node.op_type == "GatherBlockQuantized" for node in model.graph) == 1
        expected_projection_weights = {
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.output_gate.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            *{
                f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
                for layer in range(2)
                for expert in range(2)
                for projection in ("gate", "up", "down")
            },
        }
        assert {
            node.inputs[1].name for node in model.graph if node.op_type == "MatMulNBits"
        } == expected_projection_weights
        assert all(
            "norm" not in node.inputs[1].name
            for node in model.graph
            if node.op_type == "MatMulNBits"
        )
        session = OnnxModelSession(model)
        try:
            outputs = session.run(self._inputs(np.asarray([[1, 2]], np.int64)))
        finally:
            session.close()
        assert np.isfinite(outputs["logits"]).all()

    def test_tied_output_uses_embedding_storage(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "minimax-tied.gguf"
        _write_minimax_gguf(path, quantized=False, omit="output.weight")
        model = build_from_gguf(path)["model"]
        assert "lm_head.weight" not in model.graph.initializers
        assert "model.embed_tokens.weight" in model.graph.initializers

    def test_quantized_tied_output_uses_embedding_storage(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "minimax-q4-tied.gguf"
        _write_minimax_gguf(
            path,
            quantized=True,
            quantized_embedding=True,
            omit="output.weight",
        )
        model = build_from_gguf(path, keep_quantized=True)["model"]
        op_types = [node.op_type for node in model.graph]
        assert op_types.count("GatherBlockQuantized") == 1
        assert op_types.count("MatMulNBits") == 20
        assert (
            sum(name.endswith("embed_tokens.qweight") for name in model.graph.initializers)
            == 1
        )
        assert not any(name.startswith("lm_head.") for name in model.graph.initializers)
        tied_head = next(
            node for node in reversed(model.graph) if node.op_type == "MatMulNBits"
        )
        assert tied_head.inputs[2].name == "model.embed_tokens.scales"
        assert tied_head.inputs[3].name == "model.embed_tokens.zero_points"
        session = OnnxModelSession(model)
        try:
            outputs = session.run(self._inputs(np.asarray([[1, 2]], np.int64)))
        finally:
            session.close()
        assert np.isfinite(outputs["logits"]).all()

    def test_cli_build(self, tmp_path: Path) -> None:
        from mobius.__main__ import main
        from mobius._model_package import ModelPackage

        path = tmp_path / "minimax-cli.gguf"
        output = tmp_path / "minimax-cli-output"
        _write_minimax_gguf(path, quantized=False)
        main(["build-gguf", str(path), "--output", str(output), "--dequantize"])
        assert "model" in ModelPackage.load(output)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"omit": "blk.0.attn_gate.weight"},
                "missing=.*blk.0.attn_gate.weight",
            ),
            (
                {"extra": "blk.0.attn_q.weight"},
                "unexpected=.*blk.0.attn_q.weight",
            ),
            (
                {"extra": "blk.2.attn_q.weight"},
                "out_of_range=.*blk.2.attn_q.weight",
            ),
            (
                {"malformed_shape": "blk.1.attn_k.weight"},
                "invalid tensor shape",
            ),
            (
                {"recurrent_layers": [True]},
                "must contain exactly 2 entries",
            ),
            (
                {"norm_eps": 0.0},
                "inconsistent architecture metadata",
            ),
            (
                {"rope_freq_base": float("nan")},
                "inconsistent architecture metadata",
            ),
        ],
    )
    def test_invalid_contract_rejected_before_graph(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        kwargs: dict,
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "minimax-invalid.gguf"
        _write_minimax_gguf(path, quantized=False, **kwargs)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path)
        graph_build.assert_not_called()


class TestGraniteHybridMoEGGUFBuild:
    """GraniteHybrid GGUF import preserves mixed state and routed expert order."""

    @staticmethod
    def _inputs(tokens: np.ndarray) -> dict[str, np.ndarray]:
        batch, sequence = tokens.shape
        return {
            "input_ids": tokens,
            "position_ids": np.broadcast_to(
                np.arange(sequence, dtype=np.int64),
                (batch, sequence),
            ).copy(),
            "attention_mask": np.ones((batch, sequence), np.int64),
            "past_key_values.0.conv_state": np.zeros((batch, 72, 3), np.float32),
            "past_key_values.0.ssm_state": np.zeros((batch, 4, 4, 16), np.float32),
            "past_key_values.1.key": np.zeros((batch, 2, 0, 8), np.float32),
            "past_key_values.1.value": np.zeros((batch, 2, 0, 8), np.float32),
        }

    def test_float_import_preserves_expert_order_and_round_trips(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-f32.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        assert [value.name for value in model.graph.outputs] == [
            "logits",
            "present.0.conv_state",
            "present.0.ssm_state",
            "present.1.key",
            "present.1.value",
        ]
        for layer in range(2):
            fused = model.graph.initializers[
                f"model.layers.{layer}.block_sparse_moe.input_linear.weight"
            ].const_value.numpy()
            down = model.graph.initializers[
                f"model.layers.{layer}.block_sparse_moe.output_linear.weight"
            ].const_value.numpy()
            for expert in range(2):
                np.testing.assert_array_equal(fused[expert, :32], 11.0 + expert)
                np.testing.assert_array_equal(fused[expert, 32:], 21.0 + expert)
                np.testing.assert_array_equal(down[expert], 31.0 + expert)

        output_dir = tmp_path / "saved-granitehybrid"
        package.save(output_dir, progress_bar=False)
        session = OnnxModelSession(ModelPackage.load(output_dir)["model"])
        outputs = session.run(self._inputs(np.asarray([[1, 2], [3, 4]], np.int64)))
        assert outputs["logits"].shape == (2, 2, 64)
        assert outputs["present.0.conv_state"].shape == (2, 72, 3)
        assert outputs["present.0.ssm_state"].shape == (2, 4, 4, 16)

    def test_prefill_decode_reorder_rollback_and_replay(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-state.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        histories = np.asarray([[1, 2], [3, 4]], np.int64)
        prefill = session.run(self._inputs(histories))
        snapshot = {name: np.array(value, copy=True) for name, value in prefill.items()}

        order = np.asarray([1, 0], np.int64)
        next_tokens = np.asarray([[5], [6]], np.int64)
        decode_inputs = {
            "input_ids": next_tokens,
            "position_ids": np.full((2, 1), 2, np.int64),
            "attention_mask": np.ones((2, 3), np.int64),
            "past_key_values.0.conv_state": snapshot["present.0.conv_state"][order],
            "past_key_values.0.ssm_state": snapshot["present.0.ssm_state"][order],
            "past_key_values.1.key": snapshot["present.1.key"][order],
            "past_key_values.1.value": snapshot["present.1.value"][order],
        }
        decoded = session.run(decode_inputs)
        replayed = session.run(decode_inputs)
        for name in decoded:
            np.testing.assert_array_equal(decoded[name], replayed[name])

        full_tokens = np.concatenate([histories[order], next_tokens], axis=1)
        full = session.run(self._inputs(full_tokens))
        np.testing.assert_allclose(
            decoded["logits"][:, -1],
            full["logits"][:, -1],
            atol=2e-5,
            rtol=2e-5,
        )

    def test_optional_shared_expert_can_be_absent(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-no-shared.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=False, shared_width=0)
        model = build_from_gguf(path)["model"]
        assert not any("shared_mlp" in name for name in model.graph.initializers)

    def test_attention_biases_can_be_absent(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-no-attention-bias.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=False, attention_biases=False)
        model = build_from_gguf(path)["model"]
        assert not any(
            name.startswith("model.layers.1.self_attn.")
            and name.endswith((".q_proj.bias", ".k_proj.bias", ".v_proj.bias"))
            for name in model.graph.initializers
        )

    def test_quantized_source_requires_explicit_dequantization(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-q4.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=True)
        with pytest.raises(ValueError, match=r"keep_quantized=False"):
            build_from_gguf(path, keep_quantized=True)

        model = build_from_gguf(path, keep_quantized=False)["model"]
        assert all(node.op_type != "MatMulNBits" for node in model.graph)
        assert model.graph.initializers[
            "model.layers.0.block_sparse_moe.input_linear.weight"
        ].shape == [2, 64, 32]

    def test_cli_builds_dequantized_package(self, tmp_path: Path) -> None:
        from mobius.__main__ import main
        from mobius._model_package import ModelPackage

        path = tmp_path / "granitehybrid-moe-cli.gguf"
        output_dir = tmp_path / "cli-output"
        _write_granitehybrid_moe_gguf(path, quantized=False)

        main(["build-gguf", str(path), "--output", str(output_dir), "--dequantize"])

        package = ModelPackage.load(output_dir)
        assert set(package) == {"model"}
        assert (output_dir / "model.onnx").is_file()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omit": "blk.1.ffn_up_exps.weight"}, "tensor closure"),
            ({"extra": "blk.0.ffn_gate.weight"}, "mixes dense and routed"),
            (
                {"malformed_shape": "blk.1.ffn_gate_exps.weight"},
                "tensor shape",
            ),
            ({"expert_count": 0, "expert_used_count": 1}, "both be zero or both positive"),
            ({"expert_count": 1, "expert_used_count": 1}, "not a routed-MoE"),
            ({"expert_count": 2, "expert_used_count": 3}, "must be in"),
            ({"omit": "blk.1.attn_q.bias"}, "partial attention Q/K/V projection"),
            ({"extra": "blk.0.ffn_gate_up_exps.weight"}, "unexpected|Malformed"),
            ({"extra": "blk.0.ffn_gate_exps.scale"}, "auxiliary quantization"),
            ({"extra": "blk.2.ffn_gate_inp.weight"}, "out_of_range"),
        ],
    )
    def test_malformed_sources_fail_before_graph(
        self,
        tmp_path: Path,
        monkeypatch,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-invalid.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=False, **kwargs)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path)
        graph_build.assert_not_called()


class TestNemotronHMoEGGUFBuild:
    """Nemotron-H GGUF import preserves hybrid state and exact expert semantics."""

    @staticmethod
    def _inputs(tokens: np.ndarray) -> dict[str, np.ndarray]:
        batch, sequence = tokens.shape
        return {
            "input_ids": tokens,
            "position_ids": np.broadcast_to(
                np.arange(sequence, dtype=np.int64),
                (batch, sequence),
            ).copy(),
            "attention_mask": np.ones((batch, sequence), np.int64),
            "past_key_values.0.conv_state": np.zeros((batch, 72, 3), np.float32),
            "past_key_values.0.ssm_state": np.zeros((batch, 4, 4, 16), np.float32),
            "past_key_values.2.key": np.zeros((batch, 1, 0, 16), np.float32),
            "past_key_values.2.value": np.zeros((batch, 1, 0, 16), np.float32),
        }

    def test_float_import_preserves_expert_order_state_and_roundtrip(
        self, tmp_path: Path
    ) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-f32.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        assert [value.name for value in model.graph.outputs] == [
            "logits",
            "present.0.conv_state",
            "present.0.ssm_state",
            "present.2.key",
            "present.2.value",
        ]
        for expert in range(2):
            for projection in ("up_proj", "down_proj"):
                value = model.graph.initializers[
                    f"model.layers.1.moe.experts.{expert}.{projection}.weight_t"
                ].const_value.numpy()
                np.testing.assert_array_equal(value, expert + 1)

        output_dir = tmp_path / "saved-nemotron-h-moe"
        package.save(output_dir, progress_bar=False)
        session = OnnxModelSession(ModelPackage.load(output_dir)["model"])
        outputs = session.run(self._inputs(np.asarray([[1, 2], [3, 4]], np.int64)))
        assert outputs["logits"].shape == (2, 2, 64)
        assert outputs["present.0.conv_state"].shape == (2, 72, 3)
        assert outputs["present.0.ssm_state"].shape == (2, 4, 4, 16)
        session.close()

    def test_latent_projection_imports_and_executes(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-latent.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False, latent=True)
        model = build_from_gguf(path)["model"]
        assert model.graph.initializers[
            "model.layers.1.moe.fc1_latent_proj.weight_t"
        ].shape == [32, 32]
        session = OnnxModelSession(model)
        outputs = session.run(self._inputs(np.asarray([[1, 2]], np.int64)))
        assert outputs["logits"].shape == (1, 2, 64)
        session.close()

    def test_prefill_decode_threads_all_state_families(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-state.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        histories = np.asarray([[1, 2], [3, 4]], np.int64)
        prefill = session.run(self._inputs(histories))
        order = np.asarray([1, 0], np.int64)
        next_tokens = np.asarray([[5], [6]], np.int64)
        decode_inputs = {
            "input_ids": next_tokens,
            "position_ids": np.full((2, 1), 2, np.int64),
            "attention_mask": np.ones((2, 3), np.int64),
            "past_key_values.0.conv_state": prefill["present.0.conv_state"][order],
            "past_key_values.0.ssm_state": prefill["present.0.ssm_state"][order],
            "past_key_values.2.key": prefill["present.2.key"][order],
            "past_key_values.2.value": prefill["present.2.value"][order],
        }
        decoded = session.run(decode_inputs)
        replayed = session.run(decode_inputs)
        for name in decoded:
            np.testing.assert_array_equal(decoded[name], replayed[name])

        full_tokens = np.concatenate([histories[order], next_tokens], axis=1)
        full = session.run(self._inputs(full_tokens))
        np.testing.assert_allclose(
            decoded["logits"][:, -1],
            full["logits"][:, -1],
            atol=2e-5,
            rtol=2e-5,
        )
        session.close()

    def test_quantized_source_requires_explicit_dequantization(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-q4.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=True)
        with pytest.raises(ValueError, match=r"keep_quantized=False"):
            build_from_gguf(path, keep_quantized=True)

        model = build_from_gguf(path, keep_quantized=False)["model"]
        assert all(node.op_type != "MatMulNBits" for node in model.graph)
        assert (
            model.graph.initializers[
                "model.layers.1.moe.experts.0.up_proj.weight_t"
            ].const_value.dtype
            == ir.DataType.FLOAT
        )

    def test_quantized_expert_only_requires_explicit_dequantization(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-expert-q4.gguf"
        _write_nemotron_h_moe_gguf(
            path,
            quantized=True,
            quantized_only="blk.1.ffn_up_exps.weight",
        )
        with pytest.raises(ValueError, match=r"keep_quantized=False"):
            build_from_gguf(path, keep_quantized=True)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omit": "blk.1.ffn_up_exps.weight"}, "tensor closure"),
            ({"omit": "blk.1.exp_probs_b.bias"}, "tensor closure"),
            ({"extra": "blk.1.ffn_exp_probs_b.bias"}, "tensor closure"),
            ({"extra": "blk.1.ffn_up_exps.scale"}, "auxiliary quantization"),
            (
                {"latent": True, "omit": "blk.1.ffn_latent_up.weight"},
                "latent projection",
            ),
            (
                {"malformed_shape": "blk.1.ffn_down_exps.weight"},
                "tensor shape",
            ),
        ],
    )
    def test_malformed_sources_fail_before_graph(
        self,
        tmp_path: Path,
        monkeypatch,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-invalid.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False, **kwargs)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path)
        graph_build.assert_not_called()

    def test_mtp_sidecar_fails_before_graph(self, tmp_path: Path, monkeypatch) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-mtp.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False, mtp=True)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)
        with pytest.raises(NotImplementedError, match="routed experts"):
            build_from_gguf(path)
        graph_build.assert_not_called()

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"static_cache": True}, "static_cache=True"),
            ({"task": "text-generation"}, "hybrid-text-generation"),
        ],
    )
    def test_incompatible_task_options_fail_closed(
        self,
        tmp_path: Path,
        options: dict[str, object],
        match: str,
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-options.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path, **options)


class TestBuildGgufStaticCache:
    """Tests for build_from_gguf(static_cache=True).

    Static cache mode replaces the dynamic concat-grow KV cache with
    pre-allocated fixed-width buffers (written in place via TensorScatter),
    producing a fully static-shaped graph required by fixed-shape runtimes
    such as the QNN HTP backend. Llama uses the base ``DecoderLayer`` which
    supports the StaticCacheState dispatch.
    """

    def test_static_cache_emits_fixed_width_cache_io(self, q4_0_gguf: Path):
        """Static cache build exposes fixed-width key_cache/value_cache I/O."""
        from mobius.integrations.gguf import build_from_gguf

        max_seq_len = 128
        model = build_from_gguf(
            q4_0_gguf, keep_quantized=True, static_cache=True, max_seq_len=max_seq_len
        )["model"]

        input_names = {i.name for i in model.graph.inputs}
        # Static cache uses key_cache.N / value_cache.N inputs, not the
        # dynamic past_key_values.N.key / .value pair.
        assert any(n and n.startswith("key_cache.") for n in input_names), (
            f"Expected key_cache.* inputs, got {sorted(input_names)}"
        )
        assert not any(n and n.startswith("past_key_values.") for n in input_names), (
            f"Static cache must not emit past_key_values.* inputs, got {sorted(input_names)}"
        )

        # The KV axis of every cache buffer must be a concrete int == max_seq_len,
        # i.e. fully static (no symbolic dims).
        for inp in model.graph.inputs:
            name = inp.name or ""
            if name.startswith(("key_cache.", "value_cache.")):
                assert inp.shape is not None
                assert inp.shape[1] == max_seq_len, (
                    f"{name} KV axis {inp.shape[1]!r} != max_seq_len {max_seq_len}"
                )

    def test_static_cache_rejects_explicit_task(self, q4_0_gguf: Path):
        """static_cache=True with an explicit task override is a ValueError."""
        from mobius.integrations.gguf import build_from_gguf

        with pytest.raises(ValueError, match="static_cache"):
            build_from_gguf(
                q4_0_gguf,
                keep_quantized=True,
                static_cache=True,
                task="text-generation",
            )


class TestMultimodalQuantizationDefault:
    @pytest.mark.parametrize("keep_quantized", [True, False])
    def test_build_from_gguf_forwards_quantization_policy_to_mmproj(
        self, keep_quantized: bool
    ):
        from mobius.integrations.gguf import build_from_gguf

        expected = mock.sentinel.package
        with mock.patch(
            "mobius.integrations.gguf._mmproj.build_vlm_from_gguf",
            return_value=expected,
        ) as build_multimodal:
            kwargs = {} if keep_quantized else {"keep_quantized": False}
            actual = build_from_gguf("text.gguf", mmproj="mmproj.gguf", **kwargs)

        assert actual is expected
        build_multimodal.assert_called_once_with(
            "text.gguf",
            "mmproj.gguf",
            dtype=None,
            execution_provider="default",
            keep_quantized=keep_quantized,
        )


class TestRawTensorIterator:
    """Tests for GGUFModel.tensor_items_raw()."""

    def test_yields_raw_data(self, q4_0_gguf: Path):
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(q4_0_gguf)
        items = list(model.tensor_items_raw())

        # Should have tensors
        assert len(items) > 0

        # Check a quantized tensor
        q_items = [(n, d, qt, s) for n, d, qt, s in items if qt == GGMLQuantizationType.Q4_0]
        assert len(q_items) > 0
        _name, raw, _qtype, shape = q_items[0]
        assert raw.dtype == np.uint8
        assert len(shape) == 2

    def test_float_tensors_have_correct_type(self, q4_0_gguf: Path):
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(q4_0_gguf)

        f32_items = [
            (n, d, qt, s)
            for n, d, qt, s in model.tensor_items_raw()
            if qt == GGMLQuantizationType.F32
        ]
        assert len(f32_items) > 0
        for _name, _raw, qtype, _shape in f32_items:
            assert qtype == GGMLQuantizationType.F32

    def test_dequantize_raw_tensor_matches_get_tensor(self, q4_0_gguf: Path):
        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(q4_0_gguf)
        name, raw, qtype, shape = next(
            item for item in model.tensor_items_raw() if len(item[3]) == 2
        )
        expected = model.get_tensor(name)
        actual = model.dequantize_raw_tensor(raw, qtype, shape)
        np.testing.assert_array_equal(actual, expected)


class TestGGUFPreflightGuards:
    """Unsupported layouts fail before graph construction or large downloads."""

    def test_nemotron_layout_excludes_combined_mtp_block(self):
        from mobius.integrations.gguf._builder import (
            _summarize_nemotron_h_moe_layout,
        )

        # Pinned NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 schedule:
        # 52 backbone layers followed by one combined attention+MoE MTP block.
        backbone_schedule = (
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
        )
        assert len(backbone_schedule) == 52

        representative_tensor = {
            "mamba": "ssm_in.weight",
            "moe": "ffn_up_exps.weight",
            "attention": "attn_q.weight",
        }
        tensor_names = [
            f"blk.{index}.{representative_tensor[layer_type]}"
            for index, layer_type in enumerate(backbone_schedule)
        ]
        tensor_names.extend(
            [
                "blk.52.nextn.eh_proj.weight",
                "blk.52.attn_q.weight",
                "blk.52.ffn_up_exps.weight",
            ]
        )

        counts, mtp_blocks, mtp_kinds = _summarize_nemotron_h_moe_layout(tensor_names)

        assert dict(counts) == {"mamba": 23, "moe": 23, "attention": 6}
        assert mtp_blocks == (52,)
        assert mtp_kinds == {52: frozenset({"attention", "moe"})}

    @pytest.mark.parametrize(
        ("architecture", "projection_quantization"),
        [
            pytest.param(architecture, quantization, id=f"{architecture}-{quantization}")
            for architecture in ("pockettts", "qwen3tts", "talkie", "wavtokenizer-dec")
            for quantization in ("f32", "q4_0")
        ],
    )
    @pytest.mark.parametrize(
        "build_options",
        [
            pytest.param({}, id="default"),
            pytest.param({"dtype": "f16"}, id="dtype"),
            pytest.param({"static_cache": True}, id="static-cache"),
            pytest.param({"keep_quantized": False}, id="dequantize"),
        ],
    )
    def test_deferred_audio_architectures_fail_before_all_downstream_stages(
        self,
        architecture: str,
        projection_quantization: str,
        build_options: dict[str, object],
        tmp_path: Path,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius import _registry as core_registry
        from mobius.integrations.gguf import _builder as gguf_builder
        from mobius.integrations.gguf import _config_mapping, build_from_gguf
        from mobius.integrations.gguf._errors import UnsupportedGGUFArchitectureError

        path = tmp_path / f"{architecture}-{projection_quantization}.gguf"
        _write_quantized_gguf(
            path,
            architecture=architecture,
            projection_quantization=projection_quantization,
        )

        downstream = AssertionError("deferred architecture reached a downstream stage")
        with (
            mock.patch.object(
                gguf_builder, "_has_quantized_weights", side_effect=downstream
            ) as quantization_probe,
            mock.patch.object(
                _config_mapping, "gguf_to_config", side_effect=downstream
            ) as config_extraction,
            mock.patch.object(
                core_registry.registry, "get", side_effect=downstream
            ) as module_lookup,
            mock.patch.object(
                core_builder, "build_from_module", side_effect=downstream
            ) as graph_build,
            pytest.raises(
                UnsupportedGGUFArchitectureError,
                match=rf"{re.escape(architecture)}.*before config extraction",
            ),
        ):
            build_from_gguf(path, **build_options)

        quantization_probe.assert_not_called()
        config_extraction.assert_not_called()
        module_lookup.assert_not_called()
        graph_build.assert_not_called()

    def test_invalid_decay_fails_before_graph(self, tmp_path: Path, monkeypatch) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "falcon-h1-invalid-decay.gguf"
        _write_falcon_h1_gguf(path, quantized=False, invalid_decay=True)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)

        with pytest.raises(ValueError, match="finite negative"):
            build_from_gguf(path)
        graph_build.assert_not_called()

    @pytest.mark.parametrize(
        "architecture",
        [
            "bailingmoe3",
            "deepseek4",
            "kimi-k3",
            "kimi-linear",
            "arctic",
            "dbrx",
            "gpt-oss",
            "grok",
            "grovemoe",
            "smallthinker",
            "chameleon",
            "cogvlm",
            "deepseek2-ocr",
            "gemma3n",
            "hunyuan_vl",
            "llama4",
            "mistral3",
            "paddleocr",
            "qwen2vl",
            "qwen3vl",
            "qwen3vlmoe",
        ],
    )
    def test_deferred_audited_cohorts_fail_before_graph_construction(
        self, architecture: str, tmp_path: Path
    ) -> None:
        from mobius import _builder as core_builder
        from mobius import _registry as core_registry
        from mobius.integrations.gguf import _builder as gguf_builder
        from mobius.integrations.gguf import _config_mapping, build_from_gguf
        from mobius.integrations.gguf._errors import UnsupportedGGUFArchitectureError

        path = tmp_path / f"{architecture}.gguf"
        _write_quantized_gguf(path, architecture=architecture)
        downstream = AssertionError("deferred architecture reached graph construction")
        with (
            mock.patch.object(
                gguf_builder, "_has_quantized_weights", side_effect=downstream
            ) as quantization_probe,
            mock.patch.object(
                _config_mapping, "gguf_to_config", side_effect=downstream
            ) as config_extraction,
            mock.patch.object(
                core_registry.registry, "get", side_effect=downstream
            ) as module_lookup,
            mock.patch.object(
                core_builder, "build_from_module", side_effect=downstream
            ) as graph_build,
            pytest.raises(UnsupportedGGUFArchitectureError, match=architecture),
        ):
            build_from_gguf(path)

        quantization_probe.assert_not_called()
        config_extraction.assert_not_called()
        module_lookup.assert_not_called()
        graph_build.assert_not_called()

    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_standalone_clip_fails_before_all_downstream_stages(
        self, projection_quantization: str, tmp_path: Path
    ) -> None:
        from mobius import _builder as core_builder
        from mobius import _registry as core_registry
        from mobius.integrations.gguf import _builder as gguf_builder
        from mobius.integrations.gguf import _config_mapping, build_from_gguf
        from mobius.integrations.gguf._errors import DisabledGGUFArchitectureError

        path = tmp_path / f"standalone-clip-{projection_quantization}.gguf"
        _write_quantized_gguf(
            path,
            architecture="clip",
            projection_quantization=projection_quantization,
        )

        downstream = AssertionError("standalone clip reached a downstream stage")
        with (
            mock.patch.object(
                gguf_builder, "_has_quantized_weights", side_effect=downstream
            ) as quantization_probe,
            mock.patch.object(
                _config_mapping, "gguf_to_config", side_effect=downstream
            ) as config_extraction,
            mock.patch.object(
                core_registry.registry, "get", side_effect=downstream
            ) as module_lookup,
            mock.patch.object(
                core_builder, "build_from_module", side_effect=downstream
            ) as graph_build,
            pytest.raises(
                DisabledGGUFArchitectureError,
                match=r"clip.*intentionally disabled",
            ),
        ):
            build_from_gguf(path)

        quantization_probe.assert_not_called()
        config_extraction.assert_not_called()
        module_lookup.assert_not_called()
        graph_build.assert_not_called()

    def test_remote_deferred_audio_architecture_fails_before_download(self):
        from mobius.integrations.gguf._builder import _resolve_gguf_path
        from mobius.integrations.gguf._errors import UnsupportedGGUFArchitectureError

        filename = "talkie-f16.gguf"
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file",
                side_effect=UnsupportedGGUFArchitectureError(
                    "talkie before config extraction"
                ),
            ),
            mock.patch("mobius.integrations.gguf._builder.hf_hub_download") as download,
            pytest.raises(
                UnsupportedGGUFArchitectureError,
                match=r"talkie.*before config extraction",
            ),
        ):
            _resolve_gguf_path(f"example/talkie:{filename}")

        download.assert_not_called()

    def test_remote_standalone_clip_fails_before_download(self):
        from mobius.integrations.gguf._builder import _resolve_gguf_path
        from mobius.integrations.gguf._errors import DisabledGGUFArchitectureError

        filename = "mmproj-f16.gguf"
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file",
                side_effect=DisabledGGUFArchitectureError("clip intentionally disabled"),
            ),
            mock.patch("mobius.integrations.gguf._builder.hf_hub_download") as download,
            pytest.raises(
                DisabledGGUFArchitectureError,
                match=r"clip.*intentionally disabled",
            ),
        ):
            _resolve_gguf_path(f"example/mmproj:{filename}")

        download.assert_not_called()

    def test_exact_selected_clip_header_returns_immutable_revision(self) -> None:
        from mobius.integrations.gguf._builder import (
            _GGUF_HEADER_RANGE_BYTES,
            _preflight_hf_mmproj_companion_file,
        )

        response = mock.MagicMock()
        response.iter_bytes.return_value = [_gguf_header_prefix("clip")]
        response_context = mock.MagicMock()
        response_context.__enter__.return_value = response
        session = mock.MagicMock()
        session.stream.return_value = response_context
        commit_hash = "c" * 40

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_url",
                return_value="https://huggingface.co/exact-file",
            ) as hub_url,
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash=commit_hash,
                    location="https://huggingface.co/mmproj.gguf",
                ),
            ) as file_metadata,
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.build_hf_headers",
                return_value={"authorization": "Bearer test-token"},
            ),
        ):
            actual = _preflight_hf_mmproj_companion_file(
                "example/gemma4",
                "nested/mmproj-F16.gguf",
                revision="pinned-revision",
            )

        assert actual == commit_hash
        hub_url.assert_called_once_with(
            "example/gemma4",
            "nested/mmproj-F16.gguf",
            revision="pinned-revision",
        )
        file_metadata.assert_called_once_with("https://huggingface.co/exact-file")
        session.stream.assert_called_once()
        stream_args = session.stream.call_args
        assert stream_args.args == ("GET", "https://huggingface.co/mmproj.gguf")
        assert stream_args.kwargs["headers"]["Range"] == (
            f"bytes=0-{_GGUF_HEADER_RANGE_BYTES - 1}"
        )
        assert stream_args.kwargs["headers"]["authorization"] == "Bearer test-token"
        response.raise_for_status.assert_called_once_with()

    def test_exact_selected_primary_header_pins_download_revision(self) -> None:
        from mobius.integrations.gguf._builder import (
            _preflight_hf_gguf_file,
            _resolve_gguf_path,
        )

        response = mock.MagicMock()
        response.iter_bytes.return_value = [_gguf_header_prefix("qwen35")]
        response_context = mock.MagicMock()
        response_context.__enter__.return_value = response
        session = mock.MagicMock()
        session.stream.return_value = response_context
        commit_hash = "a" * 40
        with (
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_url",
                return_value="https://huggingface.co/exact-primary",
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash=commit_hash,
                    location="https://huggingface.co/model.gguf",
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
        ):
            assert (
                _preflight_hf_gguf_file(
                    "example/qwen35",
                    "model.gguf",
                    revision="requested-revision",
                )
                == commit_hash
            )

        with (
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file",
                return_value=commit_hash,
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_download",
                return_value="cached-model.gguf",
            ) as download,
        ):
            assert _resolve_gguf_path("example/qwen35:model.gguf") == "cached-model.gguf"
        download.assert_called_once_with(
            repo_id="example/qwen35",
            filename="model.gguf",
            revision=commit_hash,
        )

    def test_explicit_hub_revision_and_filename_are_preflighted_exactly(self) -> None:
        from mobius.integrations.gguf._builder import _resolve_gguf_path

        commit_hash = "b" * 40
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file",
                return_value=commit_hash,
            ) as preflight,
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_download",
                return_value="cached-model.gguf",
            ) as download,
        ):
            assert (
                _resolve_gguf_path("example/qwen35@release-v1:nested/model.gguf")
                == "cached-model.gguf"
            )

        preflight.assert_called_once_with(
            "example/qwen35", "nested/model.gguf", revision="release-v1"
        )
        download.assert_called_once_with(
            repo_id="example/qwen35",
            filename="nested/model.gguf",
            revision=commit_hash,
        )

    def test_nested_hub_filename_is_preserved_for_runtime_identity(self):
        from mobius.integrations.gguf._builder import _logical_source_filename

        assert (
            _logical_source_filename(
                "example/model:nested/model.gguf",
                "/cache/blobs/abcdef",
            )
            == "nested/model.gguf"
        )
        assert (
            _logical_source_filename(
                "example/model",
                "/cache/models--example--model/snapshots/abc/nested/model.gguf",
            )
            == "nested/model.gguf"
        )

    def test_explicit_hub_revision_is_used_for_filename_discovery(self) -> None:
        from mobius.integrations.gguf._builder import _resolve_gguf_path

        commit_hash = "c" * 40
        with (
            mock.patch("mobius.integrations.gguf._builder.HfApi") as api_type,
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file",
                return_value=commit_hash,
            ) as preflight,
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_download",
                return_value="cached-historical.gguf",
            ) as download,
        ):
            api_type.return_value.list_repo_files.return_value = ["historical.gguf"]
            assert _resolve_gguf_path("example/qwen35@historical") == "cached-historical.gguf"

        api_type.return_value.list_repo_files.assert_called_once_with(
            "example/qwen35", revision="historical"
        )
        preflight.assert_called_once_with(
            "example/qwen35", "historical.gguf", revision="historical"
        )
        download.assert_called_once_with(
            repo_id="example/qwen35",
            filename="historical.gguf",
            revision=commit_hash,
        )

    def test_exact_selected_non_clip_header_rejects_without_payload(self) -> None:
        from mobius.integrations.gguf._builder import (
            _preflight_hf_mmproj_companion_file,
        )

        response = mock.MagicMock()
        response.iter_bytes.return_value = [_gguf_header_prefix("gemma4")]
        response_context = mock.MagicMock()
        response_context.__enter__.return_value = response
        session = mock.MagicMock()
        session.stream.return_value = response_context

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_url",
                return_value="https://huggingface.co/selected-text-file",
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash="d" * 40,
                    location="https://cdn.example/model.gguf",
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.build_hf_headers",
                return_value={"authorization": "Bearer must-not-leak"},
            ),
            pytest.raises(
                ValueError,
                match=r"Expected a 'clip' mmproj.*architecture 'gemma4'.*No payload",
            ),
        ):
            _preflight_hf_mmproj_companion_file(
                "example/mixed-repo",
                "selected-text.gguf",
                revision="exact-revision",
            )

        assert "authorization" not in session.stream.call_args.kwargs["headers"]

    def test_structural_header_parser_rejects_duplicate_architecture_entries(self) -> None:
        from mobius.integrations.gguf._builder import (
            _gguf_architecture_from_header_prefix,
        )

        with pytest.raises(ValueError, match=r"exactly one.*found 2"):
            _gguf_architecture_from_header_prefix(
                _gguf_header_prefix("clip", "llama"),
                source="duplicate.gguf",
            )

    def test_exact_companion_preflight_rejects_unresolved_mutable_revision(self) -> None:
        from mobius.integrations.gguf._builder import (
            _preflight_hf_mmproj_companion_file,
        )

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                side_effect=OfflineModeIsEnabled("offline"),
            ),
            pytest.raises(RuntimeError, match="immutable revision"),
        ):
            (
                _preflight_hf_mmproj_companion_file(
                    "example/offline",
                    "mmproj.gguf",
                    revision="main",
                )
            )

    def test_exact_companion_range_read_supports_requests_hub_sessions(self) -> None:
        from mobius.integrations.gguf._builder import (
            _preflight_hf_mmproj_companion_file,
        )

        class RequestsResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

            def iter_content(self, *, chunk_size: int):
                assert chunk_size == 64 * 1024
                yield _gguf_header_prefix("clip")

        session = mock.Mock()
        session.stream = False
        session.get.return_value = RequestsResponse()
        with (
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash="e" * 40,
                    location="https://cdn.example/mmproj.gguf",
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
        ):
            assert (
                _preflight_hf_mmproj_companion_file(
                    "example/legacy-hub",
                    "mmproj.gguf",
                    revision="main",
                )
                == "e" * 40
            )

        session.get.assert_called_once()
        assert session.get.call_args.kwargs["stream"] is True

    def test_exact_companion_http_status_falls_back_to_pinned_download(self) -> None:
        from mobius.integrations.gguf._builder import (
            _preflight_hf_mmproj_companion_file,
        )

        request = httpx.Request("GET", "https://cdn.example/mmproj.gguf")
        failed_response = httpx.Response(403, request=request)
        response = mock.MagicMock()
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "forbidden",
            request=request,
            response=failed_response,
        )
        response_context = mock.MagicMock()
        response_context.__enter__.return_value = response
        session = mock.MagicMock()
        session.stream.return_value = response_context
        commit_hash = "f" * 40

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash=commit_hash,
                    location="https://cdn.example/mmproj.gguf",
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
        ):
            assert (
                _preflight_hf_mmproj_companion_file(
                    "example/status-fallback",
                    "mmproj.gguf",
                    revision="main",
                )
                == commit_hash
            )

    @pytest.mark.parametrize(
        "preflight_error",
        [
            pytest.param(OfflineModeIsEnabled("offline"), id="offline-cache"),
            pytest.param(httpx.ConnectError("disconnected"), id="transport-error"),
        ],
    )
    def test_unavailable_exact_file_preflight_rejects_mutable_download(self, preflight_error):
        from mobius.integrations.gguf._builder import _resolve_gguf_path

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                side_effect=preflight_error,
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_download",
                return_value="cached-model.gguf",
            ) as download,
            pytest.raises(RuntimeError, match="immutable revision"),
        ):
            _resolve_gguf_path("owner/repo:model.gguf")

        download.assert_not_called()

    def test_remote_shard_fails_before_hub_calls(self):
        from mobius.integrations.gguf._builder import _resolve_gguf_path

        filename = "BF16/model-00001-of-00002.gguf"
        with (
            mock.patch("mobius.integrations.gguf._builder.HfApi") as api_type,
            mock.patch("mobius.integrations.gguf._builder.hf_hub_download") as download,
            pytest.raises(NotImplementedError, match="shard 1 of 2"),
        ):
            _resolve_gguf_path(f"owner/repo:{filename}")

        api_type.return_value.model_info.assert_not_called()
        download.assert_not_called()

    def test_local_split_metadata_is_rejected(self):
        from mobius.integrations.gguf._builder import _raise_for_sharded_gguf

        with pytest.raises(NotImplementedError, match="cannot assemble split tensor tables"):
            _raise_for_sharded_gguf(source="model-00001-of-00002.gguf", split_count=2)


class TestHybridTensorContract:
    class _FakeGGUF:
        def __init__(self, architecture, metadata, tensor_names):
            self.architecture = architecture
            self.metadata = metadata
            self.tensor_names = tensor_names

    @staticmethod
    def _lfm2_names() -> list[str]:
        names = ["token_embd.weight", "token_embd_norm.weight"]
        common = [
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ]
        mixers = [
            [
                "shortconv.conv.weight",
                "shortconv.in_proj.weight",
                "shortconv.out_proj.weight",
            ],
            [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
            ],
        ]
        for layer, conditional in enumerate(mixers):
            names.extend(f"blk.{layer}.{suffix}" for suffix in [*common, *conditional])
        return names

    @classmethod
    def _lfm2moe_names(cls) -> list[str]:
        names = cls._lfm2_names()
        dense_layer_1 = {
            "blk.1.ffn_gate.weight",
            "blk.1.ffn_up.weight",
            "blk.1.ffn_down.weight",
        }
        names = [name for name in names if name not in dense_layer_1]
        names.extend(
            f"blk.1.{suffix}"
            for suffix in (
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "exp_probs_b.bias",
            )
        )
        return names

    def test_lfm2_exact_mixer_closure_passes(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        model = self._FakeGGUF(
            "lfm2",
            {
                "lfm2.block_count": 2,
                "lfm2.attention.head_count_kv": [0, 2],
            },
            self._lfm2_names(),
        )
        _raise_for_invalid_hybrid_tensor_contract(model)

    def test_lfm2moe_exact_dense_and_routed_closure_passes(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        _raise_for_invalid_hybrid_tensor_contract(
            self._FakeGGUF(
                "lfm2moe",
                {
                    "lfm2moe.block_count": 2,
                    "lfm2moe.attention.head_count_kv": [0, 2],
                    "lfm2moe.leading_dense_block_count": 1,
                },
                self._lfm2moe_names(),
            )
        )

    @pytest.mark.parametrize(
        "wrong_tensor",
        ["blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate.weight"],
    )
    def test_lfm2moe_wrong_ffn_family_is_rejected(self, wrong_tensor: str) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        with pytest.raises(ValueError, match=r"wrong .* FFN family"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(
                    "lfm2moe",
                    {
                        "lfm2moe.block_count": 2,
                        "lfm2moe.attention.head_count_kv": [0, 2],
                        "lfm2moe.leading_dense_block_count": 1,
                    },
                    [*self._lfm2moe_names(), wrong_tensor],
                )
            )

    def test_wrong_mixer_tensor_is_rejected(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        names = self._lfm2_names()
        names.append("blk.0.attn_q.weight")
        model = self._FakeGGUF(
            "lfm2",
            {
                "lfm2.block_count": 2,
                "lfm2.attention.head_count_kv": [0, 2],
            },
            names,
        )
        with pytest.raises(ValueError, match="wrong conv mixer family"):
            _raise_for_invalid_hybrid_tensor_contract(model)

    def test_missing_recurrent_tensor_and_extra_layer_are_rejected(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        metadata = {
            "lfm2.block_count": 2,
            "lfm2.attention.head_count_kv": [0, 2],
        }
        missing = [
            name for name in self._lfm2_names() if name != "blk.0.shortconv.conv.weight"
        ]
        with pytest.raises(ValueError, match=r"shortconv\.conv\.weight"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF("lfm2", metadata, missing)
            )

        with pytest.raises(ValueError, match="out-of-range"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(
                    "lfm2", metadata, [*self._lfm2_names(), "blk.2.ffn_norm.weight"]
                )
            )

    @staticmethod
    def _second_cohort_names(architecture: str) -> list[str]:
        names = ["token_embd.weight", "output_norm.weight"]
        if architecture == "jamba":
            common = [
                "attn_norm.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ]
            mixers = [
                [
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_conv1d.bias",
                    "ssm_x.weight",
                    "ssm_dt_norm.weight",
                    "ssm_dt.weight",
                    "ssm_dt.bias",
                    "ssm_b_norm.weight",
                    "ssm_c_norm.weight",
                    "ssm_a",
                    "ssm_d",
                    "ssm_out.weight",
                ],
                [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                ],
            ]
        elif architecture == "nemotron_h":
            common = ["attn_norm.weight"]
            mixers = [
                [
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ],
                ["ffn_up.weight", "ffn_down.weight"],
            ]
        else:
            common = [
                "attn_norm.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ]
            mixers = [
                [
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ],
                [
                    "attn_q.weight",
                    "attn_q.bias",
                    "attn_k.weight",
                    "attn_k.bias",
                    "attn_v.weight",
                    "attn_v.bias",
                    "attn_output.weight",
                ],
            ]
        for layer, mixer in enumerate(mixers):
            names.extend(f"blk.{layer}.{suffix}" for suffix in [*common, *mixer])
        return names

    @pytest.mark.parametrize("architecture", ["jamba", "nemotron_h", "granitehybrid"])
    def test_second_cohort_exact_closure_and_mutations(self, architecture: str) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        metadata = {
            f"{architecture}.block_count": 2,
            f"{architecture}.attention.head_count_kv": [0, 2],
        }
        if architecture == "nemotron_h":
            metadata[f"{architecture}.feed_forward_length"] = [0, 128]
        names = self._second_cohort_names(architecture)
        _raise_for_invalid_hybrid_tensor_contract(
            self._FakeGGUF(architecture, metadata, names)
        )

        with pytest.raises(ValueError, match="missing"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(architecture, metadata, names[:-1])
            )
        with pytest.raises(ValueError, match="unexpected"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(
                    architecture,
                    metadata,
                    [*names, "blk.0.attn_q.weight"],
                )
            )
        mtp_metadata = {
            **metadata,
            f"{architecture}.nextn_predict_layers": 1,
        }
        with pytest.raises(ValueError, match=r"auxiliary|folded MTP"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(architecture, mtp_metadata, names)
            )
        with pytest.raises(ValueError, match="out_of_range"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(
                    architecture,
                    metadata,
                    [*names, "blk.2.attn_norm.weight"],
                )
            )
        partial_bias = {
            "nemotron_h": "blk.1.ffn_up.bias",
        }.get(architecture)
        if partial_bias is not None:
            with pytest.raises(ValueError, match=r"partial .* bias family"):
                _raise_for_invalid_hybrid_tensor_contract(
                    self._FakeGGUF(
                        architecture,
                        metadata,
                        [*names, partial_bias],
                    )
                )

    def test_partial_fused_and_separate_experts_are_rejected(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        model = self._FakeGGUF(
            "qwen35moe",
            {
                "qwen35moe.block_count": 1,
                "qwen35moe.attention.recurrent_layers": [False],
            },
            [
                "token_embd.weight",
                "output_norm.weight",
                "blk.0.ffn_gate_up_exps.weight",
                "blk.0.ffn_gate_exps.weight",
            ],
        )
        with pytest.raises(ValueError, match="mixes fused and separate"):
            _raise_for_invalid_hybrid_tensor_contract(model)

    def test_partial_modern_and_legacy_recurrent_inputs_are_rejected(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        model = self._FakeGGUF(
            "qwen3next",
            {
                "qwen3next.block_count": 1,
                "qwen3next.attention.recurrent_layers": [True],
            },
            [
                "token_embd.weight",
                "output_norm.weight",
                "blk.0.ffn_gate_up_exps.weight",
                "blk.0.ssm_in.weight",
                "blk.0.attn_qkv.weight",
            ],
        )
        with pytest.raises(ValueError, match="mixes legacy ssm_in with modern"):
            _raise_for_invalid_hybrid_tensor_contract(model)


class TestNormalizeGgufWeights:
    """Tests for GGUF-specific weight shape/value normalization."""

    def test_deltanet_a_log_is_inverted_from_neg_exp(self):
        """GGUF ssm_a = -exp(A_log); normalize must recover raw A_log = log(-ssm_a).

        The GatedDeltaNet module re-applies ``-exp(A_log)`` at runtime, so the
        round-trip ``-exp(normalize(ssm_a))`` must reproduce the original
        ``ssm_a`` the converter stored.
        """
        import torch

        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        # A representative raw A_log, and the value the GGUF converter stores.
        a_log_raw = torch.tensor([-3.4688, -1.0703, -5.0, -0.5], dtype=torch.float32)
        ssm_a = -torch.exp(a_log_raw)  # what llama.cpp writes to blk.N.ssm_a
        assert bool((ssm_a < 0).all())  # sanity: pre-transformed value is negative

        key = "model.layers.0.linear_attn.A_log"
        out = _normalize_gguf_weights({key: ssm_a})

        # The stored parameter must be the raw A_log again ...
        assert torch.allclose(out[key], a_log_raw, atol=1e-5)
        # ... so that the module's runtime -exp(A_log) recovers ssm_a exactly.
        assert torch.allclose(-torch.exp(out[key]), ssm_a, atol=1e-6)

    def test_non_deltanet_a_log_is_untouched(self):
        """Mamba/PLaMo SSM ``A_log`` (consumed as ``A`` directly) must not be inverted."""
        import torch

        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        ssm_a = torch.tensor([-0.04, -0.5], dtype=torch.float32)
        key = "backbone.layers.0.mixer.A_log"
        out = _normalize_gguf_weights({key: ssm_a})
        assert torch.allclose(out[key], ssm_a)

    def test_zero_centered_norm_offset_removed_for_qwen35(self):
        """qwen35 GGUF bakes +1 into transformer norms; normalize must strip it.

        mobius applies the ``1 +`` at runtime via OffsetRMSNorm, so the stored
        weight must be the raw zero-centered value again.
        """
        import torch

        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        sd = {
            "model.layers.0.input_layernorm.weight": torch.tensor([1.5, 2.0]),
            "model.layers.3.self_attn.q_norm.weight": torch.tensor([1.25]),
            "model.layers.3.self_attn.k_norm.weight": torch.tensor([1.1]),
            "model.norm.weight": torch.tensor([1.94]),
            # DeltaNet internal gated norm — converter did NOT add +1.
            "model.layers.0.linear_attn.norm.weight": torch.tensor([0.87]),
        }
        out = _normalize_gguf_weights(dict(sd), gguf_arch="qwen35")

        assert torch.allclose(
            out["model.layers.0.input_layernorm.weight"], torch.tensor([0.5, 1.0])
        )
        assert torch.allclose(
            out["model.layers.3.self_attn.q_norm.weight"], torch.tensor([0.25])
        )
        assert torch.allclose(out["model.norm.weight"], torch.tensor([0.94]))
        # linear_attn.norm is a plain gated RMSNorm — must be left untouched.
        assert torch.allclose(
            out["model.layers.0.linear_attn.norm.weight"], torch.tensor([0.87])
        )

    def test_norm_offset_not_applied_for_non_offset_arch(self):
        """Standard-RMSNorm archs (e.g. llama/qwen2) must not have norms shifted."""
        import torch

        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        sd = {
            "model.layers.0.input_layernorm.weight": torch.tensor([1.0, 1.0]),
            "model.norm.weight": torch.tensor([1.0]),
        }
        out = _normalize_gguf_weights(dict(sd), gguf_arch="qwen2")
        assert torch.allclose(
            out["model.layers.0.input_layernorm.weight"], torch.tensor([1.0, 1.0])
        )
        assert torch.allclose(out["model.norm.weight"], torch.tensor([1.0]))

    @pytest.mark.parametrize(
        ("suffix", "shape"),
        [
            ("weight", (2, 6, 4)),
            ("qweight", (2, 6, 4, 2)),
            ("scales", (2, 6, 4)),
            ("zero_points", (2, 6, 4)),
        ],
    )
    def test_fused_experts_are_split_without_tensor_loss(self, suffix, shape):
        import torch

        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        key = f"model.layers.0.mlp.experts.gate_up_proj.{suffix}"
        value = torch.arange(np.prod(shape)).reshape(shape)
        out = _normalize_gguf_weights({key: value}, gguf_arch="qwen35moe")

        assert key not in out
        assert len(out) == 4
        gate, up = value.chunk(2, dim=1)
        for expert in range(2):
            assert torch.equal(
                out[f"model.layers.0.mlp.experts.{expert}.gate_proj.{suffix}"],
                gate[expert],
            )
            assert torch.equal(
                out[f"model.layers.0.mlp.experts.{expert}.up_proj.{suffix}"],
                up[expert],
            )


class TestReorderDeltaNetVHeads:
    """Undo of the GGUF converter's grouped→tiled Gated-DeltaNet V-head order.

    The llama.cpp converter reorders every V-indexed ``linear_attn`` tensor from
    HuggingFace *grouped* order into ggml *tiled* order (see
    ``_LinearAttentionVReorderBase._reorder_v_heads``). mobius consumes grouped
    order, so ``_reorder_deltanet_v_heads`` must be the exact inverse.
    """

    # Small grouped linear-attention geometry: 2 K-heads, 6 V-heads (v_per_k=3).
    CFG = SimpleNamespace(
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
    )

    @staticmethod
    def _converter_reorder(tensor, dim, num_k_heads, num_v_per_k, head_dim):
        """Reference grouped→tiled reorder copied from llama.cpp's converter."""
        import torch  # noqa: F401

        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        new_shape = [*shape[:dim], num_k_heads, num_v_per_k, head_dim, *shape[dim + 1 :]]
        tensor = tensor.reshape(*new_shape)
        perm = list(range(len(new_shape)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return tensor.permute(*perm).contiguous().reshape(*shape)

    def test_row_tensors_roundtrip(self):
        """Grouped weights survive tile→untile for every V-row projection."""
        import torch

        from mobius.integrations.gguf._builder import _reorder_deltanet_v_heads

        cfg = self.CFG
        n_k, n_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        v_per_k = n_v // n_k
        hd_k, hd_v = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        key_dim, value_dim = hd_k * n_k, hd_v * n_v
        hidden = 5
        torch.manual_seed(0)

        p = "model.layers.0.linear_attn."
        grouped = {
            f"{p}in_proj_z.weight": torch.randn(value_dim, hidden),
            f"{p}in_proj_a.weight": torch.randn(n_v, hidden),
            f"{p}in_proj_b.weight": torch.randn(n_v, hidden),
            f"{p}A_log": torch.randn(n_v),
            f"{p}dt_bias": torch.randn(n_v),
            f"{p}conv1d.weight": torch.randn(2 * key_dim + value_dim, 1, 4),
        }
        # in_proj_qkv: only the V rows (after 2*key_dim) are reordered.
        qkv = torch.randn(2 * key_dim + value_dim, hidden)
        grouped[f"{p}in_proj_qkv.weight"] = qkv

        # Build the tiled (GGUF) state by applying the converter's reorder.
        tiled = {k: v.clone() for k, v in grouped.items()}
        tiled[f"{p}in_proj_z.weight"] = self._converter_reorder(
            grouped[f"{p}in_proj_z.weight"], 0, n_k, v_per_k, hd_v
        )
        for name in ("in_proj_a", "in_proj_b"):
            tiled[f"{p}{name}.weight"] = self._converter_reorder(
                grouped[f"{p}{name}.weight"], 0, n_k, v_per_k, 1
            )
        for name in ("A_log", "dt_bias"):
            tiled[f"{p}{name}"] = self._converter_reorder(
                grouped[f"{p}{name}"], 0, n_k, v_per_k, 1
            )
        # V portion of qkv / conv1d.
        v0 = 2 * key_dim
        qv = self._converter_reorder(qkv[v0:], 0, n_k, v_per_k, hd_v)
        tiled[f"{p}in_proj_qkv.weight"] = torch.cat([qkv[:v0], qv], dim=0)
        conv = grouped[f"{p}conv1d.weight"]
        cv = self._converter_reorder(conv[v0:], 0, n_k, v_per_k, hd_v)
        tiled[f"{p}conv1d.weight"] = torch.cat([conv[:v0], cv], dim=0)

        out = _reorder_deltanet_v_heads({k: v.clone() for k, v in tiled.items()}, cfg)

        for k in grouped:
            assert torch.allclose(out[k], grouped[k]), k

    def test_quantized_out_proj_columns_roundtrip(self):
        """out_proj's quantized K axis (blocks + packed zero-points) round-trips."""
        import torch

        from mobius.integrations.gguf._builder import _reorder_deltanet_v_heads

        cfg = self.CFG
        n_k, n_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        v_per_k = n_v // n_k
        hd_v = cfg.linear_value_head_dim  # 4
        value_dim = hd_v * n_v  # 24
        block = 2  # 2 elems/block -> head_v_dim(4) = 2 blocks (even -> byte aligned)
        n_blocks = value_dim // block  # 12
        hidden = 5
        torch.manual_seed(1)

        p = "model.layers.0.linear_attn."
        # Grouped quantized out_proj triplet: [hidden, K/block, block/2], etc.
        gw = torch.randint(0, 255, (hidden, n_blocks, block // 2 + 7), dtype=torch.uint8)
        gs = torch.randn(hidden, n_blocks, dtype=torch.float16)
        gz = torch.randint(0, 255, (hidden, n_blocks // 2), dtype=torch.uint8)
        # Provide a grouped row tensor so the head geometry is exercised too.
        grouped = {
            f"{p}out_proj.weight": gw,
            f"{p}out_proj.scales": gs,
            f"{p}out_proj.zero_points": gz,
        }
        blocks_per_head = n_blocks // n_v  # 2
        tiled = {
            f"{p}out_proj.weight": self._converter_reorder(
                gw, 1, n_k, v_per_k, blocks_per_head
            ),
            f"{p}out_proj.scales": self._converter_reorder(
                gs, 1, n_k, v_per_k, blocks_per_head
            ),
            f"{p}out_proj.zero_points": self._converter_reorder(
                gz, 1, n_k, v_per_k, blocks_per_head // 2
            ),
        }

        out = _reorder_deltanet_v_heads({k: v.clone() for k, v in tiled.items()}, cfg)

        for k in grouped:
            assert torch.equal(out[k], grouped[k]), k

    def test_no_reorder_when_heads_equal(self):
        """Ungrouped linear attention (num_v == num_k) is left untouched."""
        import torch

        from mobius.integrations.gguf._builder import _reorder_deltanet_v_heads

        cfg = SimpleNamespace(
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
        )
        p = "model.layers.0.linear_attn."
        sd = {f"{p}in_proj_z.weight": torch.randn(16, 5)}
        ref = sd[f"{p}in_proj_z.weight"].clone()
        out = _reorder_deltanet_v_heads({k: v.clone() for k, v in sd.items()}, cfg)
        assert torch.equal(out[f"{p}in_proj_z.weight"], ref)


class TestGgufArchSurvivesToWeightProcessing:
    """``_gguf_arch`` must reach ``process_tensors`` on every build path.

    It is a plain instance attribute, not a dataclass field, so every
    ``dataclasses.replace`` in the builder drops it. It is also the key the
    weight-processor dispatch is built on, so losing it silently demotes
    dispatch to the ``model_type`` fallback — which is exactly the indirection
    the architecture registry exists to remove. A regression here would be
    invisible until a spec's processor stopped agreeing with its model_type,
    and would then affect only non-float32 and quantized imports.
    """

    @staticmethod
    def _recorded_arches(monkeypatch, gguf_path: Path, **build_kwargs) -> list[object]:
        from mobius.integrations.gguf import _builder as builder_module
        from mobius.integrations.gguf import _tensor_processors

        seen: list[object] = []
        real = _tensor_processors.process_tensors

        def spy(state_dict, config):
            seen.append(getattr(config, "_gguf_arch", None))
            return real(state_dict, config)

        monkeypatch.setattr(_tensor_processors, "process_tensors", spy)
        builder_module.build_from_gguf(gguf_path, **build_kwargs)
        return seen

    def test_float_path_keeps_the_architecture(self, monkeypatch, q4_0_gguf: Path):
        seen = self._recorded_arches(monkeypatch, q4_0_gguf, keep_quantized=False)
        assert seen, "process_tensors was never called"
        assert all(arch == "llama" for arch in seen), seen

    def test_dtype_override_keeps_the_architecture(self, monkeypatch, q4_0_gguf: Path):
        """``dtype`` triggers a ``dataclasses.replace`` that drops the attribute."""
        seen = self._recorded_arches(
            monkeypatch, q4_0_gguf, keep_quantized=False, dtype="float16"
        )
        assert seen, "process_tensors was never called"
        assert all(arch == "llama" for arch in seen), seen

    def test_quantized_path_keeps_the_architecture(self, monkeypatch, q4_0_gguf: Path):
        """The preserve-quantization path replaces the config as well."""
        seen = self._recorded_arches(monkeypatch, q4_0_gguf, keep_quantized=True)
        assert seen, "process_tensors was never called"
        assert all(arch == "llama" for arch in seen), seen
