# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Focused exactness tests for dense MiniCPM and MiniCPM3 GGUF support."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._configs import ArchitectureConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.gguf._builder import (
    _needs_qk_permute,
    _raise_for_invalid_minicpm_tensor_contract,
)
from mobius.integrations.gguf._config_mapping import (
    _infer_tie_embeddings,
    _minicpm3_postprocess,
    _minicpm_postprocess,
)
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.integrations.gguf._tensor_processors import _reverse_permute, process_tensors
from mobius.models.minicpm import MiniCPM3CausalLMModel, MiniCPMCausalLMModel
from mobius.tasks import CausalLMTask


class _FakeGGUF:
    def __init__(
        self,
        architecture: str,
        metadata: dict[str, object],
        tensors: dict[str, np.ndarray],
        qtypes: dict[str, int] | None = None,
    ):
        self.architecture = architecture
        self.metadata = metadata
        self._tensors = tensors
        self._qtypes = qtypes or {}
        self.tensor_names = tuple(tensors)

    def get_tensor(self, name: str) -> np.ndarray:
        return self._tensors[name]

    def tensor_items_raw(self):
        for name, value in self._tensors.items():
            yield name, memoryview(value), self._qtypes.get(name, 0), value.shape


def _base_config(**overrides: object) -> ArchitectureConfig:
    values: dict[str, object] = {
        "hidden_size": 8,
        "head_dim": 4,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 16,
        "max_position_embeddings": 16,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "rope_type": "default",
    }
    values.update(overrides)
    return ArchitectureConfig(**values)


def _metadata(architecture: str) -> dict[str, object]:
    return {
        f"{architecture}.context_length": 16,
        f"{architecture}.embedding_length": 8,
        f"{architecture}.feed_forward_length": 16,
        f"{architecture}.block_count": 1,
        f"{architecture}.attention.head_count": 2,
        f"{architecture}.attention.head_count_kv": 2,
        f"{architecture}.vocab_size": 16,
    }


def _dense_tensors(architecture: str, *, output: bool = True) -> dict[str, np.ndarray]:
    tensors = {
        "token_embd.weight": np.zeros((16, 8), dtype=np.float32),
        "output_norm.weight": np.ones(8, dtype=np.float32),
        "blk.0.attn_norm.weight": np.ones(8, dtype=np.float32),
        "blk.0.attn_output.weight": np.zeros((8, 8), dtype=np.float32),
        "blk.0.ffn_norm.weight": np.ones(8, dtype=np.float32),
        "blk.0.ffn_gate.weight": np.zeros((16, 8), dtype=np.float32),
        "blk.0.ffn_up.weight": np.zeros((16, 8), dtype=np.float32),
        "blk.0.ffn_down.weight": np.zeros((8, 16), dtype=np.float32),
    }
    if output:
        tensors["output.weight"] = np.zeros((16, 8), dtype=np.float32)
    if architecture == "minicpm":
        tensors.update(
            {
                "blk.0.attn_q.weight": np.zeros((8, 8), dtype=np.float32),
                "blk.0.attn_k.weight": np.zeros((8, 8), dtype=np.float32),
                "blk.0.attn_v.weight": np.zeros((8, 8), dtype=np.float32),
            }
        )
    else:
        tensors.update(
            {
                "blk.0.attn_q_a.weight": np.zeros((8, 8), dtype=np.float32),
                "blk.0.attn_q_a_norm.weight": np.ones(8, dtype=np.float32),
                "blk.0.attn_q_b.weight": np.zeros((16, 8), dtype=np.float32),
                "blk.0.attn_kv_a_mqa.weight": np.zeros((8, 8), dtype=np.float32),
                "blk.0.attn_kv_a_norm.weight": np.ones(4, dtype=np.float32),
                "blk.0.attn_kv_b.weight": np.zeros((16, 4), dtype=np.float32),
            }
        )
    return tensors


def _minicpm3_config(**overrides: object) -> ArchitectureConfig:
    values: dict[str, object] = {
        "head_dim": 8,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "rope_interleave": False,
        "embedding_multiplier": 12.0,
        "residual_multiplier": 1.4,
        "logits_scaling": 8 / 256,
    }
    values.update(overrides)
    return _base_config(
        **values,
    )


def _write_tiny_gguf(path: Path, architecture: str) -> None:
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), architecture)
    writer.add_context_length(16)
    writer.add_embedding_length(8)
    writer.add_feed_forward_length(16)
    writer.add_block_count(1)
    writer.add_head_count(2)
    writer.add_head_count_kv(2)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(16)
    if architecture == "minicpm":
        writer.add_float32("minicpm.embedding_scale", 12.0)
        writer.add_float32("minicpm.residual_scale", 1.0)
        writer.add_float32("minicpm.logit_scale", 1.0)
        writer.add_rope_dimension_count(4)
    else:
        writer.add_uint32("minicpm3.attention.key_length", 8)
        writer.add_uint32("minicpm3.attention.q_lora_rank", 8)
        writer.add_uint32("minicpm3.attention.kv_lora_rank", 4)
        writer.add_rope_dimension_count(4)
    for name, values in _dense_tensors(architecture).items():
        writer.add_tensor(name, values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_quantized_minicpm(path: Path) -> None:
    from gguf import GGMLQuantizationType, GGUFWriter

    writer = GGUFWriter(str(path), "minicpm")
    writer.add_context_length(16)
    writer.add_embedding_length(32)
    writer.add_feed_forward_length(32)
    writer.add_block_count(1)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(32)
    writer.add_float32("minicpm.embedding_scale", 12.0)
    writer.add_float32("minicpm.residual_scale", 1.4)
    writer.add_float32("minicpm.logit_scale", 8.0)
    writer.add_rope_dimension_count(8)

    def add_float(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.ones(shape, dtype=np.float32))

    def add_q4(name: str, shape: tuple[int, int]) -> None:
        raw = np.zeros((shape[0], shape[1] // 32 * 18), dtype=np.uint8)
        for row in range(shape[0]):
            raw[row, :2] = np.array([row + 1], dtype=np.float16).view(np.uint8)
            raw[row, 2:] = row
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    add_float("token_embd.weight", (32, 32))
    add_float("output_norm.weight", (32,))
    add_q4("output.weight", (32, 32))
    add_float("blk.0.attn_norm.weight", (32,))
    add_q4("blk.0.attn_q.weight", (32, 32))
    add_q4("blk.0.attn_k.weight", (16, 32))
    add_q4("blk.0.attn_v.weight", (16, 32))
    add_q4("blk.0.attn_output.weight", (32, 32))
    add_float("blk.0.ffn_norm.weight", (32,))
    add_q4("blk.0.ffn_gate.weight", (32, 32))
    add_q4("blk.0.ffn_up.weight", (32, 32))
    add_q4("blk.0.ffn_down.weight", (32, 32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_minicpm_scales_and_longrope_are_exact() -> None:
    metadata = _metadata("minicpm")
    metadata.update(
        {
            "minicpm.embedding_scale": 12.0,
            "minicpm.residual_scale": 0.7,
            "minicpm.logit_scale": 10.0,
            "minicpm.rope.dimension_count": 4,
            "minicpm.rope.scaling.original_context_length": 8,
        }
    )
    tensors = _dense_tensors("minicpm")
    tensors["rope_factors_long.weight"] = np.array([2.0, 3.0], dtype=np.float32)
    tensors["rope_factors_short.weight"] = np.array([1.0, 1.5], dtype=np.float32)
    config = _minicpm_postprocess(
        _base_config(),
        metadata,
        _FakeGGUF("minicpm", metadata, tensors),
    )

    assert config.embedding_multiplier == pytest.approx(12.0)
    assert config.residual_multiplier == pytest.approx(0.7)
    assert config.logits_scaling == pytest.approx(10.0)
    assert config.rope_type == "longrope"
    assert config.original_max_position_embeddings == 8
    assert config.rope_scaling == {
        "long_factor": [2.0, 3.0],
        "short_factor": [1.0, 1.5],
    }


def test_minicpm_quantized_longrope_factors_fail_closed() -> None:
    metadata = _metadata("minicpm")
    metadata.update(
        {
            "minicpm.embedding_scale": 12.0,
            "minicpm.residual_scale": 0.7,
            "minicpm.logit_scale": 10.0,
            "minicpm.rope.dimension_count": 4,
            "minicpm.rope.scaling.original_context_length": 8,
        }
    )
    tensors = _dense_tensors("minicpm")
    tensors["rope_factors_long.weight"] = np.array([2.0, 3.0], dtype=np.float32)
    tensors["rope_factors_short.weight"] = np.array([1.0, 1.5], dtype=np.float32)
    model = _FakeGGUF(
        "minicpm",
        metadata,
        tensors,
        qtypes={"rope_factors_long.weight": 2},
    )

    with pytest.raises(ValueError, match="F32/F16/BF16"):
        _minicpm_postprocess(_base_config(), metadata, model)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_minicpm_invalid_scales_fail_closed(bad: float) -> None:
    metadata = _metadata("minicpm")
    metadata.update(
        {
            "minicpm.embedding_scale": bad,
            "minicpm.residual_scale": 1.0,
            "minicpm.logit_scale": 1.0,
        }
    )
    model = _FakeGGUF("minicpm", metadata, _dense_tensors("minicpm"))
    with pytest.raises(ValueError, match="scales must be finite positive"):
        _minicpm_postprocess(_base_config(), metadata, model)


def test_minicpm3_postprocessor_derives_pinned_mla_geometry_and_scales() -> None:
    metadata = _metadata("minicpm3")
    metadata.update(
        {
            "minicpm3.attention.key_length": 8,
            "minicpm3.attention.q_lora_rank": 8,
            "minicpm3.attention.kv_lora_rank": 4,
            "minicpm3.rope.dimension_count": 4,
        }
    )
    model = _FakeGGUF("minicpm3", metadata, _dense_tensors("minicpm3"))
    config = _minicpm3_postprocess(_base_config(), metadata, model)

    assert config.q_lora_rank == 8
    assert config.kv_lora_rank == 4
    assert config.qk_nope_head_dim == 4
    assert config.qk_rope_head_dim == 4
    assert config.v_head_dim == 4
    assert config.embedding_multiplier == pytest.approx(12.0)
    assert config.residual_multiplier == pytest.approx(1.4)
    assert config.logits_scaling == pytest.approx(8 / 256)


@pytest.mark.parametrize("architecture", ["minicpm", "minicpm3"])
def test_tiny_gguf_build_closes_config_mapping_and_weight_loading(
    tmp_path: Path, architecture: str
) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = tmp_path / f"{architecture}.gguf"
    _write_tiny_gguf(path, architecture)
    package = build_from_gguf(path)
    assert package.config.model_type == architecture
    assert all(
        initializer.const_value is not None
        for initializer in package["model"].graph.initializers.values()
    )
    if architecture == "minicpm3":
        assert str(package["model"].graph.outputs[1].shape).endswith(",8]")
        assert str(package["model"].graph.outputs[2].shape).endswith(",4]")


def test_quantized_minicpm_loader_permutes_packed_rows_scales_and_zero_points(
    tmp_path: Path,
) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = tmp_path / "minicpm-q4.gguf"
    _write_quantized_minicpm(path)
    model = build_from_gguf(path, keep_quantized=True)["model"]
    expected_q_scales = _reverse_permute(
        torch.arange(1, 33, dtype=torch.float32).reshape(32, 1), 4
    ).numpy()
    expected_k_scales = _reverse_permute(
        torch.arange(1, 17, dtype=torch.float32).reshape(16, 1), 2
    ).numpy()

    for projection, expected_scales in (
        ("q_proj", expected_q_scales),
        ("k_proj", expected_k_scales),
    ):
        stem = f"model.layers.0.self_attn.{projection}"
        packed = model.graph.initializers[f"{stem}.weight"].const_value
        scales = model.graph.initializers[f"{stem}.scales"].const_value
        zero_points = model.graph.initializers[f"{stem}.zero_points"].const_value
        assert packed is not None
        assert scales is not None
        assert zero_points is not None
        np.testing.assert_array_equal(scales.numpy(), expected_scales)
        np.testing.assert_array_equal(
            zero_points.numpy(),
            np.full(expected_scales.shape, 8, dtype=np.uint8),
        )
        # Every source row has a unique packed nibble pattern, so this verifies
        # that packed values follow the same row order as their affine metadata.
        expected_rows = _reverse_permute(
            torch.arange(expected_scales.shape[0]), 4 if projection == "q_proj" else 2
        ).numpy()
        np.testing.assert_array_equal(
            packed.numpy()[:, 0, 0],
            ((expected_rows & 0x0F) * 0x11).astype(np.uint8),
        )


@pytest.mark.parametrize("architecture", ["minicpm", "minicpm3"])
def test_dense_tensor_closure_and_output_ownership(architecture: str) -> None:
    metadata = _metadata(architecture)
    if architecture == "minicpm":
        metadata["minicpm.rope.dimension_count"] = 4
    else:
        metadata.update(
            {
                "minicpm3.attention.key_length": 8,
                "minicpm3.attention.q_lora_rank": 8,
                "minicpm3.attention.kv_lora_rank": 4,
                "minicpm3.rope.dimension_count": 4,
            }
        )
    untied = _FakeGGUF(architecture, metadata, _dense_tensors(architecture))
    tied = _FakeGGUF(architecture, metadata, _dense_tensors(architecture, output=False))
    _raise_for_invalid_minicpm_tensor_contract(untied)
    _raise_for_invalid_minicpm_tensor_contract(tied)
    assert not _infer_tie_embeddings(untied)
    assert _infer_tie_embeddings(tied)


def test_minicpm_moe_and_malformed_closures_fail_before_graph_build() -> None:
    metadata = _metadata("minicpm")
    metadata["minicpm.expert_count"] = 8
    model = _FakeGGUF("minicpm", metadata, _dense_tensors("minicpm"))
    with pytest.raises(ValueError, match="outside the exact dense graph subset"):
        _raise_for_invalid_minicpm_tensor_contract(model)

    metadata["minicpm.expert_count"] = 0
    model._tensors["blk.0.attn_q.weight"] = np.zeros((7, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=r"malformed=.*attn_q"):
        _raise_for_invalid_minicpm_tensor_contract(model)

    model._tensors["blk.0.attn_q.weight"] = np.zeros((8, 8), dtype=np.float32)
    model._tensors["rope_freqs.weight"] = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError, match=r"serialized rope_freqs\.weight"):
        _raise_for_invalid_minicpm_tensor_contract(model)


def test_minicpm_float_and_quantized_qk_rows_use_distinct_exact_permutations() -> None:
    config = SimpleNamespace(
        _gguf_arch="minicpm",
        model_type="minicpm",
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    q = torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3)
    k = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3) + 1_000

    def pinned_permute(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        return (
            tensor.reshape(heads, 2, -1, *tensor.shape[1:]).swapaxes(1, 2).reshape_as(tensor)
        )

    result = process_tensors(
        {
            "model.layers.0.self_attn.q_proj.weight": pinned_permute(q, 4),
            "model.layers.0.self_attn.k_proj.weight": pinned_permute(k, 2),
        },
        config,
    )
    torch.testing.assert_close(result["model.layers.0.self_attn.q_proj.weight"], q)
    torch.testing.assert_close(result["model.layers.0.self_attn.k_proj.weight"], k)

    assert _needs_qk_permute(
        "model.layers.0.self_attn.q_proj.weight", 4, 2, gguf_arch="minicpm"
    )
    for rows, heads in ((32, 4), (16, 2)):
        packed = torch.arange(rows * 2).reshape(rows, 2)
        scales = torch.arange(rows).reshape(rows, 1)
        zero_points = torch.arange(rows).reshape(rows, 1) + 100
        permutation = [
            _reverse_permute(value, heads) for value in (packed, scales, zero_points)
        ]
        expected_rows = _reverse_permute(torch.arange(rows), heads)
        torch.testing.assert_close(permutation[0], packed.index_select(0, expected_rows))
        torch.testing.assert_close(permutation[1], scales.index_select(0, expected_rows))
        torch.testing.assert_close(permutation[2], zero_points.index_select(0, expected_rows))


def test_minicpm3_tensor_mapping_is_exact() -> None:
    assert (
        map_gguf_to_hf_names("blk.7.attn_q_a.weight", "minicpm3")
        == "model.layers.7.self_attn.q_a_proj.weight"
    )
    assert (
        map_gguf_to_hf_names("blk.7.attn_kv_b.weight", "minicpm3")
        == "model.layers.7.self_attn.kv_b_proj.weight"
    )


def test_tiny_minicpm_graph_contains_model_owned_scaling() -> None:
    config = _base_config(
        embedding_multiplier=12.0,
        residual_multiplier=0.7,
        logits_scaling=10.0,
    )
    model = CausalLMTask().build(MiniCPMCausalLMModel(config), config)["model"]
    constants = [
        float(node.attributes["value_float"].as_float())
        for node in model.graph
        if node.op_type == "Constant" and "value_float" in node.attributes
    ]
    constants.extend(
        float(value.const_value.numpy())
        for node in model.graph
        for value in node.inputs
        if value is not None
        and value.const_value is not None
        and value.const_value.numpy().shape == ()
    )
    assert any(value == pytest.approx(12.0) for value in constants)
    assert any(value == pytest.approx(0.7) for value in constants)
    assert any(value == pytest.approx(0.1) for value in constants)


def test_minicpm3_asymmetric_expanded_cache_prefill_decode_parity() -> None:
    config = _minicpm3_config()
    graph = CausalLMTask().build(MiniCPM3CausalLMModel(config), config)["model"]
    assert tuple(graph.graph.inputs[3].shape) == ("batch", 2, "past_sequence_len", 8)
    assert tuple(graph.graph.inputs[4].shape) == ("batch", 2, "past_sequence_len", 4)
    assert tuple(graph.graph.outputs[1].shape) == (
        "batch",
        2,
        "past_sequence_len + sequence_len",
        8,
    )
    assert tuple(graph.graph.outputs[2].shape) == (
        "batch",
        2,
        "past_sequence_len + sequence_len",
        4,
    )

    rng = np.random.default_rng(17)
    for initializer in graph.graph.initializers.values():
        if initializer.const_value is None:
            values = (rng.standard_normal(initializer.shape) * 0.05).astype(np.float32)
            if "norm.weight" in initializer.name:
                values.fill(1.0)
            initializer.const_value = ir.tensor(values, name=initializer.name)

    session = OnnxModelSession(graph)
    tokens = np.array([[2, 5, 7]], dtype=np.int64)

    def run(input_ids: np.ndarray, past_key: np.ndarray, past_value: np.ndarray):
        total = past_key.shape[2] + input_ids.shape[1]
        return session.run(
            {
                "input_ids": input_ids,
                "attention_mask": np.ones((1, total), dtype=np.int64),
                "position_ids": np.arange(past_key.shape[2], total, dtype=np.int64).reshape(
                    1, -1
                ),
                "past_key_values.0.key": past_key,
                "past_key_values.0.value": past_value,
            }
        )

    empty_key = np.empty((1, 2, 0, 8), dtype=np.float32)
    empty_value = np.empty((1, 2, 0, 4), dtype=np.float32)
    full = run(tokens, empty_key, empty_value)
    prefill = run(tokens[:, :2], empty_key, empty_value)
    decoded = run(
        tokens[:, 2:],
        prefill["present.0.key"],
        prefill["present.0.value"],
    )
    np.testing.assert_allclose(
        decoded["logits"],
        full["logits"][:, -1:],
        rtol=1e-4,
        atol=1e-5,
    )


def test_minicpm3_static_and_paged_cache_fail_closed() -> None:
    config = _minicpm3_config()
    with pytest.raises(TypeError, match="Static cache mode requires decoder layers"):
        CausalLMTask(static_cache=True, max_seq_len=16).build(
            MiniCPM3CausalLMModel(config), config
        )
    paged = _minicpm3_config(export_paged_attention=True)
    with pytest.raises(ValueError, match="expanded standard K/V cache"):
        MiniCPM3CausalLMModel(paged)
