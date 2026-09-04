# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Private synthetic GGUF writers and fixtures shared by builder tests."""

from __future__ import annotations

import errno
import os
import struct
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest


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


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        if os.name == "nt" and (
            getattr(error, "winerror", None) in {1, 50, 1314}
            or error.errno in {errno.EPERM, errno.EACCES, errno.ENOSYS}
        ):
            pytest.skip(f"Windows runner cannot create test symlinks: {error}")
        raise


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
    split_max_tensors: int = 0,
    small_first_shard: bool = False,
) -> None:
    """Write a GGUF file with quantized projection weights.

    Norms are float32; all linear-layer weights in decoder blocks are
    encoded with *projection_quantization*. The embedding can optionally
    be Q4_0 and tied to the LM head.
    """
    from gguf import GGMLQuantizationType, GGUFWriter

    writer_path = path.with_suffix("") if split_max_tensors else path
    writer = GGUFWriter(
        str(writer_path),
        architecture,
        split_max_tensors=split_max_tensors,
        small_first_shard=small_first_shard,
    )
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

    def _add_k_quant(name: str, n_out: int, k_in: int, format_name: str) -> None:
        """Write finite K-quant blocks over the flattened logical tensor."""
        qtype, block_bytes = {
            "q4_k": (GGMLQuantizationType.Q4_K, 144),
            "q5_k": (GGMLQuantizationType.Q5_K, 176),
            "q6_k": (GGMLQuantizationType.Q6_K, 210),
        }[format_name]
        if k_in % 256:
            raise ValueError(f"{format_name} test tensors require K divisible by 256")
        raw = np.zeros((n_out, (k_in // 256) * block_bytes), dtype=np.uint8)
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
    elif embedding_quantization in {"q4_k", "q5_k", "q6_k"}:
        _add_k_quant(
            "token_embd.weight",
            vocab_size,
            hidden_size,
            embedding_quantization,
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
    elif projection_quantization in {"q4_k", "q5_k", "q6_k"}:

        def add_projection(name: str, n_out: int, k_in: int) -> None:
            _add_k_quant(name, n_out, k_in, projection_quantization)

    elif projection_quantization in {"f32", "f16", "bf16"}:

        def add_projection(name: str, n_out: int, k_in: int) -> None:
            add_float(name, (n_out, k_in))

    else:

        def add_projection(name: str, n_out: int, k_in: int) -> None:
            _add_native(name, n_out, k_in, projection_quantization)

    if value_projection_quantization == "q5_1":
        add_value_projection = _add_q5_1
    elif value_projection_quantization in {"q4_k", "q5_k", "q6_k"}:

        def add_value_projection(name: str, n_out: int, k_in: int) -> None:
            assert value_projection_quantization is not None
            _add_k_quant(name, n_out, k_in, value_projection_quantization)

    else:
        add_value_projection = add_projection

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
        elif output_quantization in {"q4_k", "q5_k", "q6_k"}:
            _add_k_quant(
                "output.weight",
                vocab_size,
                hidden_size,
                output_quantization,
            )
        else:
            add_float("output.weight", (vocab_size, hidden_size))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_tencent_q1_0_gguf(path: Path) -> None:
    """Write a tiny Llama file using Tencent's 130-byte Q1_0 blocks."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = intermediate = 512
    vocab = 32
    writer = GGUFWriter(str(path), "llama")
    writer.add_context_length(32)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(1)
    writer.add_head_count(8)
    writer.add_head_count_kv(2)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(vocab)

    def add_float(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.ones(shape, dtype=np.float32))

    def add_tencent_q1(name: str, n_out: int, k_in: int) -> None:
        assert k_in % 512 == 0
        blocks_per_row = k_in // 512
        raw = np.zeros((n_out, blocks_per_row * 130), dtype=np.uint8)
        raw.reshape(n_out, blocks_per_row, 130)[:, :, :2] = np.array(
            [1.0], dtype=np.float16
        ).view(np.uint8)
        writer.add_tensor_info(
            name,
            (n_out, (k_in // 128) * 18),
            raw.dtype,
            raw.nbytes,
            raw_dtype=GGMLQuantizationType.Q1_0,
        )
        writer.tensors[-1][name].tensor = raw

    add_float("token_embd.weight", (vocab, hidden))
    add_tencent_q1("blk.0.attn_q.weight", hidden, hidden)
    add_tencent_q1("blk.0.attn_k.weight", hidden // 4, hidden)
    add_tencent_q1("blk.0.attn_v.weight", hidden // 4, hidden)
    add_tencent_q1("blk.0.attn_output.weight", hidden, hidden)
    add_tencent_q1("blk.0.ffn_gate.weight", intermediate, hidden)
    add_tencent_q1("blk.0.ffn_up.weight", intermediate, hidden)
    add_tencent_q1("blk.0.ffn_down.weight", hidden, intermediate)
    add_float("blk.0.attn_norm.weight", (hidden,))
    add_float("blk.0.ffn_norm.weight", (hidden,))
    add_float("output_norm.weight", (hidden,))
    add_float("output.weight", (vocab, hidden))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
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
    mlp_biases: bool = False,
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
        expert_projection = add_q4 if quantized else add_float
        if not expert_count:
            expert_projection(prefix + "ffn_gate.weight", (expert_width, hidden))
            expert_projection(prefix + "ffn_up.weight", (expert_width, hidden))
            expert_projection(prefix + "ffn_down.weight", (hidden, expert_width))
            if mlp_biases:
                add_float(prefix + "ffn_gate.bias", (expert_width,), fill=41.0)
                add_float(prefix + "ffn_up.bias", (expert_width,), fill=42.0)
                add_float(prefix + "ffn_down.bias", (hidden,), fill=43.0)
        else:
            add_float(prefix + "ffn_gate_inp.weight", (expert_count, hidden))
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
        if expert_count and shared_width:
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


def _write_kimi_linear_gguf(
    path: Path,
    *,
    quantized: bool,
    native_quantized: bool = False,
    omit: str | None = None,
    extra: str | None = None,
    malformed_shape: str | None = None,
    kv_heads: list[int] | None = None,
    gating: int = 2,
    conv: int = 4,
) -> None:
    """Write a tiny pinned-format Kimi Linear GGUF with one KDA and one MLA layer."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 64
    dense_intermediate = 64
    expert_intermediate = 32
    vocab = 64
    heads = 2
    kda_dim = 32
    qk_dim = 48
    extra_dim = 16
    value_dim = 32
    kv_rank = 32
    experts = 2
    rng = np.random.default_rng(617)

    writer = GGUFWriter(str(path), "kimi-linear")
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(dense_intermediate)
    writer.add_block_count(2)
    writer.add_head_count(heads)
    writer.add_array(
        "kimi-linear.attention.head_count_kv",
        kv_heads if kv_heads is not None else [0, 1],
    )
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_uint32("kimi-linear.attention.key_length_mla", qk_dim)
    writer.add_uint32("kimi-linear.attention.value_length_mla", value_dim)
    writer.add_uint32("kimi-linear.attention.kv_lora_rank", kv_rank)
    writer.add_rope_dimension_count(extra_dim)
    writer.add_uint32("kimi-linear.ssm.conv_kernel", conv)
    writer.add_uint32("kimi-linear.kda.head_dim", kda_dim)
    writer.add_expert_count(experts)
    writer.add_expert_used_count(1)
    writer.add_uint32("kimi-linear.expert_feed_forward_length", expert_intermediate)
    writer.add_uint32("kimi-linear.expert_shared_count", 1)
    writer.add_uint32("kimi-linear.leading_dense_block_count", 1)
    writer.add_float32("kimi-linear.expert_weights_scale", 2.446)
    writer.add_uint32("kimi-linear.expert_gating_func", gating)
    writer.add_vocab_size(vocab)

    def shape_for(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
        if name == malformed_shape:
            return (*shape[:-1], shape[-1] + 1)
        return shape

    def add_float(name: str, shape: tuple[int, ...]) -> None:
        if name == omit:
            return
        shape = shape_for(name, shape)
        values = rng.normal(0.0, 0.02, shape).astype(np.float32)
        if name.endswith("ssm_a"):
            values = -np.exp(values)
        elif name.endswith(("output_norm.weight", "attn_norm.weight", "ffn_norm.weight")):
            values.fill(1.0)
        writer.add_tensor(name, values)

    def add_quantized(name: str, shape: tuple[int, ...]) -> None:
        if name == omit:
            return
        shape = shape_for(name, shape)
        assert shape[-1] % 32 == 0
        raw = np.zeros((*shape[:-1], shape[-1] // 32 * 18), dtype=np.uint8)
        qtype = GGMLQuantizationType.IQ4_NL if native_quantized else GGMLQuantizationType.Q4_0
        if not native_quantized:
            for index in np.ndindex(shape[:-1]):
                for block in range(shape[-1] // 32):
                    offset = block * 18
                    raw[(*index, slice(offset, offset + 2))] = np.array(
                        [rng.uniform(0.01, 0.05)], dtype=np.float16
                    ).view(np.uint8)
                    raw[(*index, slice(offset + 2, offset + 18))] = rng.integers(
                        0, 256, 16, dtype=np.uint8
                    )
        writer.add_tensor(name, raw, raw_dtype=qtype)

    add_float("token_embd.weight", (vocab, hidden))
    add_float("output_norm.weight", (hidden,))
    projection = add_quantized if quantized else add_float
    projection("output.weight", (vocab, hidden))
    projection_width = heads * kda_dim

    for layer in range(2):
        prefix = f"blk.{layer}."
        add_float(prefix + "attn_norm.weight", (hidden,))
        add_float(prefix + "ffn_norm.weight", (hidden,))
        projection(prefix + "attn_output.weight", (hidden, projection_width))
        if layer == 0:
            projection(prefix + "attn_q.weight", (projection_width, hidden))
            projection(prefix + "attn_k.weight", (projection_width, hidden))
            projection(prefix + "attn_v.weight", (projection_width, hidden))
            add_float(prefix + "ssm_conv1d_q.weight", (1, projection_width, 1, conv))
            add_float(prefix + "ssm_conv1d_k.weight", (1, projection_width, 1, conv))
            add_float(prefix + "ssm_conv1d_v.weight", (1, projection_width, 1, conv))
            projection(prefix + "ssm_f_a.weight", (kda_dim, hidden))
            projection(prefix + "ssm_f_b.weight", (projection_width, kda_dim))
            projection(prefix + "ssm_beta.weight", (heads, hidden))
            add_float(prefix + "ssm_a", (1, 1, heads, 1))
            add_float(prefix + "ssm_dt.bias", (projection_width,))
            projection(prefix + "ssm_g_a.weight", (kda_dim, hidden))
            projection(prefix + "ssm_g_b.weight", (projection_width, kda_dim))
            add_float(prefix + "ssm_norm.weight", (kda_dim,))
            projection(prefix + "ffn_gate.weight", (dense_intermediate, hidden))
            projection(prefix + "ffn_up.weight", (dense_intermediate, hidden))
            projection(prefix + "ffn_down.weight", (hidden, dense_intermediate))
        else:
            projection(prefix + "attn_q.weight", (heads * qk_dim, hidden))
            projection(prefix + "attn_kv_a_mqa.weight", (kv_rank + extra_dim, hidden))
            add_float(prefix + "attn_kv_a_norm.weight", (kv_rank,))
            projection(prefix + "attn_k_b.weight", (heads, kv_rank, qk_dim - extra_dim))
            projection(prefix + "attn_v_b.weight", (heads, value_dim, kv_rank))
            add_float(prefix + "ffn_gate_inp.weight", (experts, hidden))
            add_float(prefix + "exp_probs_b.bias", (experts,))
            projection(
                prefix + "ffn_gate_exps.weight",
                (experts, expert_intermediate, hidden),
            )
            projection(
                prefix + "ffn_up_exps.weight",
                (experts, expert_intermediate, hidden),
            )
            projection(
                prefix + "ffn_down_exps.weight",
                (experts, hidden, expert_intermediate),
            )
            projection(prefix + "ffn_gate_shexp.weight", (expert_intermediate, hidden))
            projection(prefix + "ffn_up_shexp.weight", (expert_intermediate, hidden))
            projection(prefix + "ffn_down_shexp.weight", (hidden, expert_intermediate))
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
        permute_heads: int | None = None,
    ) -> None:
        if name == omit:
            return
        shape = adjusted_shape(name, shape)
        values = rng.normal(0.0, 0.03, size=shape).astype(np.float32)
        if permute_heads is not None:
            values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            values = (
                values.reshape(
                    permute_heads,
                    2,
                    shape[0] // permute_heads // 2,
                    *shape[1:],
                )
                .swapaxes(1, 2)
                .reshape(shape)
            )
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

    if quantized:
        projection("blk.2.attn_q.weight", (heads * head_dim, hidden))
        projection("blk.2.attn_k.weight", (kv_heads * head_dim, hidden))
    else:
        add_float(
            "blk.2.attn_q.weight",
            (heads * head_dim, hidden),
            permute_heads=heads,
        )
        add_float(
            "blk.2.attn_k.weight",
            (kv_heads * head_dim, hidden),
            permute_heads=kv_heads,
        )
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
    output_quantization: str | None = None,
    include_output: bool = True,
    expert_scale_suffix: str | None = None,
    malformed_expert_scale: bool = False,
) -> None:
    """Write a one-layer conventional-attention MoE GGUF with exact families."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden_size = 64
    intermediate_size = 128
    expert_size = (
        64
        if architecture
        in {"bailingmoe", "deepseek", "dots1", "qwen2moe", "qwen3moe", "llada-moe", "rnd1"}
        else 128
    )
    shared_size = 128 if architecture == "qwen2moe" else 32
    num_experts = 4
    num_heads = 4
    num_kv_heads = num_heads if architecture == "dots1" else 2
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
    if architecture == "dots1":
        writer.add_uint32("dots1.expert_gating_func", 1)
    if architecture in {"llada-moe", "rnd1"}:
        writer.add_uint32("tokenizer.ggml.mask_token_id", vocab_size - 1)
    if architecture == "llada-moe":
        writer.add_bool("diffusion.shift_logits", False)
    if architecture in {"bailingmoe", "deepseek", "dots1", "qwen2moe", "qwen3moe"}:
        writer.add_expert_feed_forward_length(expert_size)
    if architecture in {"bailingmoe", "deepseek", "dots1"}:
        writer.add_expert_shared_count(1)
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
    add_output = {
        "f32": add_float,
        "mxfp4": add_mxfp4,
        "q4_0": add_q4,
    }[output_quantization or projection_quantization]

    if quantize_tied_embedding:
        assert architecture in {"deepseek", "dots1", "qwen3moe", "granitemoe"}
        add_q4("token_embd.weight", (vocab_size, hidden_size))
    else:
        add_float("token_embd.weight", (vocab_size, hidden_size))
    if architecture in {"bailingmoe", "deepseek", "dots1", "phimoe"} and phi_fused_qkv:
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
    elif architecture in {"dots1", "qwen3moe", "llada-moe", "rnd1"}:
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
    if architecture in {"bailingmoe", "deepseek", "dots1"}:
        add_projection("blk.0.ffn_gate_shexp.weight", (expert_size, hidden_size))
        add_projection("blk.0.ffn_up_shexp.weight", (expert_size, hidden_size))
        add_projection("blk.0.ffn_down_shexp.weight", (hidden_size, expert_size))
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
    if architecture in {"bailingmoe", "deepseek", "dots1"} and phi_fused_qkv:
        add_float(
            "blk.0.attn_qkv.bias",
            ((num_heads + 2 * num_kv_heads) * head_dim,),
        )

    add_float("output_norm.weight", (hidden_size,))
    if architecture == "phimoe":
        add_float("output_norm.bias", (hidden_size,))
    if architecture not in {"qwen3moe", "granitemoe"} and include_output:
        add_output("output.weight", (vocab_size, hidden_size))
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


@pytest.fixture
def q5_1_gguf(tmp_path: Path) -> Path:
    """Create a GGUF whose projections require dequantize/requantize."""
    path = tmp_path / "test_q5_1.gguf"
    _write_quantized_gguf(path, projection_quantization="q5_1")
    return path
