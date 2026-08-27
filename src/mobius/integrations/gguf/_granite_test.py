# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exactness tests for llama.cpp ``general.architecture = "granite"``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import _raise_for_invalid_granite_tensor_contract
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.integrations.gguf._tensor_processors import _reverse_permute, process_tensors

_HIDDEN = 32
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_INTERMEDIATE = 32
_EXPERTS = 4
_TOP_K = 2
_SHARED = 32
_LAYERS = 1
_VOCAB = 32


class _FakeGGUF:
    def __init__(
        self,
        metadata: dict[str, object],
        tensors: dict[str, np.ndarray],
        *,
        qtypes: dict[str, int] | None = None,
    ):
        self.architecture = "granite"
        self.metadata = metadata
        self._tensors = tensors
        self.tensor_names = tuple(tensors)
        self._qtypes = qtypes or {}

    def get_metadata(self, key: str, default=None):
        return self.metadata.get(key, default)

    def get_tensor(self, name: str) -> np.ndarray:
        return self._tensors[name]

    def tensor_items_raw(self):
        for name, tensor in self._tensors.items():
            yield (
                name,
                memoryview(tensor),
                SimpleNamespace(value=self._qtypes.get(name, 0)),
                tensor.shape,
            )


def _metadata(*, moe: bool = False) -> dict[str, object]:
    metadata: dict[str, object] = {
        "granite.context_length": 64,
        "granite.embedding_length": _HIDDEN,
        "granite.feed_forward_length": _INTERMEDIATE,
        "granite.block_count": _LAYERS,
        "granite.attention.head_count": _HEADS,
        "granite.attention.head_count_kv": _KV_HEADS,
        "granite.attention.layer_norm_rms_epsilon": 1e-5,
        "granite.rope.freq_base": 10_000.0,
        "granite.rope.dimension_count": _HEAD_DIM,
        "granite.vocab_size": _VOCAB,
        "granite.logit_scale": 16.0,
    }
    if moe:
        metadata.update(
            {
                "granite.expert_count": _EXPERTS,
                "granite.expert_used_count": _TOP_K,
                "granite.expert_shared_feed_forward_length": _SHARED,
            }
        )
    return metadata


def _tensors(
    *,
    moe: bool = False,
    fused_qkv: bool = False,
    biases: bool = False,
) -> dict[str, np.ndarray]:
    tensors = {
        "token_embd.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        "output_norm.weight": np.ones((_HIDDEN,), np.float32),
        "output.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
    }
    for layer in range(_LAYERS):
        prefix = f"blk.{layer}."
        tensors.update(
            {
                prefix + "attn_norm.weight": np.ones((_HIDDEN,), np.float32),
                prefix + "attn_output.weight": np.zeros((_HIDDEN, _HIDDEN), np.float32),
                prefix + "ffn_norm.weight": np.ones((_HIDDEN,), np.float32),
            }
        )
        if fused_qkv:
            tensors[prefix + "attn_qkv.weight"] = np.zeros(
                (_HIDDEN + 2 * _KV_HEADS * _HEAD_DIM, _HIDDEN), np.float32
            )
            if biases:
                tensors[prefix + "attn_qkv.bias"] = np.zeros(
                    (_HIDDEN + 2 * _KV_HEADS * _HEAD_DIM,), np.float32
                )
        else:
            tensors.update(
                {
                    prefix + "attn_q.weight": np.zeros((_HIDDEN, _HIDDEN), np.float32),
                    prefix + "attn_k.weight": np.zeros(
                        (_KV_HEADS * _HEAD_DIM, _HIDDEN), np.float32
                    ),
                    prefix + "attn_v.weight": np.zeros(
                        (_KV_HEADS * _HEAD_DIM, _HIDDEN), np.float32
                    ),
                }
            )
            if biases:
                tensors.update(
                    {
                        prefix + "attn_q.bias": np.zeros((_HIDDEN,), np.float32),
                        prefix + "attn_k.bias": np.zeros((_KV_HEADS * _HEAD_DIM,), np.float32),
                        prefix + "attn_v.bias": np.zeros((_KV_HEADS * _HEAD_DIM,), np.float32),
                    }
                )
        if biases:
            tensors[prefix + "attn_output.bias"] = np.zeros((_HIDDEN,), np.float32)

        if moe:
            tensors.update(
                {
                    prefix + "ffn_gate_inp.weight": np.zeros((_EXPERTS, _HIDDEN), np.float32),
                    prefix + "ffn_gate_exps.weight": np.zeros(
                        (_EXPERTS, _INTERMEDIATE, _HIDDEN), np.float32
                    ),
                    prefix + "ffn_up_exps.weight": np.zeros(
                        (_EXPERTS, _INTERMEDIATE, _HIDDEN), np.float32
                    ),
                    prefix + "ffn_down_exps.weight": np.zeros(
                        (_EXPERTS, _HIDDEN, _INTERMEDIATE), np.float32
                    ),
                    prefix + "ffn_gate_shexp.weight": np.zeros((_SHARED, _HIDDEN), np.float32),
                    prefix + "ffn_up_shexp.weight": np.zeros((_SHARED, _HIDDEN), np.float32),
                    prefix + "ffn_down_shexp.weight": np.zeros((_HIDDEN, _SHARED), np.float32),
                }
            )
        else:
            tensors.update(
                {
                    prefix + "ffn_gate.weight": np.zeros((_INTERMEDIATE, _HIDDEN), np.float32),
                    prefix + "ffn_up.weight": np.zeros((_INTERMEDIATE, _HIDDEN), np.float32),
                    prefix + "ffn_down.weight": np.zeros((_HIDDEN, _INTERMEDIATE), np.float32),
                }
            )
            if biases:
                tensors.update(
                    {
                        prefix + "ffn_gate.bias": np.zeros((_INTERMEDIATE,), np.float32),
                        prefix + "ffn_up.bias": np.zeros((_INTERMEDIATE,), np.float32),
                        prefix + "ffn_down.bias": np.zeros((_HIDDEN,), np.float32),
                    }
                )
    return tensors


def _write_tiny_granite(
    path: Path,
    *,
    moe: bool,
    fused_qkv: bool,
    quantized: bool,
) -> None:
    from gguf import GGMLQuantizationType, GGUFWriter

    writer = GGUFWriter(str(path), "granite")
    writer.add_context_length(64)
    writer.add_embedding_length(_HIDDEN)
    writer.add_feed_forward_length(_INTERMEDIATE)
    writer.add_block_count(_LAYERS)
    writer.add_head_count(_HEADS)
    writer.add_head_count_kv(_KV_HEADS)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10_000.0)
    writer.add_rope_dimension_count(_HEAD_DIM)
    writer.add_vocab_size(_VOCAB)
    writer.add_logit_scale(16.0)
    writer.add_embedding_scale(12.0)
    writer.add_residual_scale(0.5)
    writer.add_attention_scale(0.125)
    if moe:
        writer.add_expert_count(_EXPERTS)
        writer.add_expert_used_count(_TOP_K)
        writer.add_expert_shared_feed_forward_length(_SHARED)

    rng = np.random.default_rng(0)

    def add_tensor(name: str, shape: tuple[int, ...]) -> None:
        if (
            not quantized
            or len(shape) == 1
            or name
            in {
                "token_embd.weight",
                "output.weight",
            }
        ):
            writer.add_tensor(name, rng.normal(0.0, 0.02, shape).astype(np.float32))
            return
        assert shape[-1] % 32 == 0
        raw_shape = (*shape[:-1], shape[-1] // 32 * 18)
        raw = np.zeros(raw_shape, dtype=np.uint8)
        raw[..., :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    for name, tensor in _tensors(moe=moe, fused_qkv=fused_qkv).items():
        add_tensor(name, tensor.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_granite_registry_is_exact_and_runtime_deferred() -> None:
    spec = get_arch_spec("granite")
    assert spec.model_type == "granite"
    assert spec.tensor_map_recipe == ("llama", "diffusion_fused_qkv", "moe_extras")
    assert spec.config_postprocessor == "granite"
    assert spec.tensor_processor == "llama"
    assert spec.llama_qk_permute
    assert spec.quantized_import.value == "supported"
    assert spec.runtime.value == "deferred"


def test_granite_dense_config_restores_scales_biases_and_no_rope() -> None:
    metadata = _metadata()
    metadata.update(
        {
            "granite.embedding_scale": 12.0,
            "granite.residual_scale": 0.5,
            "granite.attention.scale": 0.125,
            "granite.rope.scaling.finetuned": False,
        }
    )
    config = gguf_to_config(_FakeGGUF(metadata, _tensors(biases=True)))

    assert config.model_type == "granite"
    assert config.embedding_multiplier == pytest.approx(12.0)
    assert config.residual_multiplier == pytest.approx(0.5)
    assert config.attention_multiplier == pytest.approx(0.125)
    assert config.logits_scaling == pytest.approx(16.0)
    assert config.attn_qkv_bias and config.attn_o_bias and config.mlp_bias
    assert config.rope_type is None


def test_granite_zero_optional_scales_mean_no_override() -> None:
    metadata = _metadata()
    metadata.update(
        {
            "granite.embedding_scale": 0.0,
            "granite.residual_scale": 0.0,
            "granite.attention.scale": 0.0,
        }
    )
    config = gguf_to_config(_FakeGGUF(metadata, _tensors()))

    assert config.embedding_multiplier == pytest.approx(1.0)
    assert config.residual_multiplier == pytest.approx(1.0)
    assert config.attention_multiplier is None


def test_granite_missing_output_uses_tied_embeddings() -> None:
    tensors = _tensors()
    tensors.pop("output.weight")
    model = _FakeGGUF(_metadata(), tensors)

    _raise_for_invalid_granite_tensor_contract(model)
    assert gguf_to_config(model).tie_word_embeddings


def test_granite_moe_config_selects_shared_expert_graph() -> None:
    model = _FakeGGUF(_metadata(moe=True), _tensors(moe=True, fused_qkv=True))
    _raise_for_invalid_granite_tensor_contract(model)
    config = gguf_to_config(model)

    assert config.model_type == "granitemoe"
    assert config.num_local_experts == _EXPERTS
    assert config.num_experts_per_tok == _TOP_K
    assert config.moe_intermediate_size == _INTERMEDIATE
    assert config.shared_expert_intermediate_size == _SHARED
    assert config.norm_topk_prob


def test_granite_longrope_factors_are_consumed_exactly() -> None:
    metadata = _metadata()
    metadata.update(
        {
            "granite.rope.scaling.type": "longrope",
            "granite.rope.scaling.original_context_length": 32,
        }
    )
    tensors = _tensors()
    tensors["rope_factors_short.weight"] = np.asarray([1.0, 1.5, 2.0, 2.5], np.float32)
    tensors["rope_factors_long.weight"] = np.asarray([3.0, 3.5, 4.0, 4.5], np.float32)
    model = _FakeGGUF(metadata, tensors)

    _raise_for_invalid_granite_tensor_contract(model)
    config = gguf_to_config(model)
    assert config.rope_type == "longrope"
    assert config.original_max_position_embeddings == 32
    assert config.rope_scaling == {
        "short_factor": [1.0, 1.5, 2.0, 2.5],
        "long_factor": [3.0, 3.5, 4.0, 4.5],
    }


def test_granite_fused_qkv_is_split_then_reverse_permuted_by_value() -> None:
    q_width = _HIDDEN
    kv_width = _KV_HEADS * _HEAD_DIM
    fused = torch.arange((q_width + 2 * kv_width) * _HIDDEN, dtype=torch.float32).reshape(
        q_width + 2 * kv_width, _HIDDEN
    )
    fused_bias = torch.arange(q_width + 2 * kv_width, dtype=torch.float32)
    config = gguf_to_config(_FakeGGUF(_metadata(), _tensors(fused_qkv=True, biases=True)))
    state = {
        "model.layers.0.self_attn.qkv_proj.weight": fused.clone(),
        "model.layers.0.self_attn.qkv_proj.bias": fused_bias.clone(),
    }

    result = process_tensors(state, config)
    for suffix, start, width, heads in (
        ("q_proj", 0, q_width, _HEADS),
        ("k_proj", q_width, kv_width, _KV_HEADS),
        ("v_proj", q_width + kv_width, kv_width, _KV_HEADS),
    ):
        expected_weight = fused[start : start + width]
        expected_bias = fused_bias[start : start + width]
        if suffix != "v_proj":
            expected_weight = _reverse_permute(expected_weight, heads)
            expected_bias = _reverse_permute(expected_bias, heads)
        torch.testing.assert_close(
            result[f"model.layers.0.self_attn.{suffix}.weight"],
            expected_weight,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result[f"model.layers.0.self_attn.{suffix}.bias"],
            expected_bias,
            rtol=0,
            atol=0,
        )
    assert not any(".qkv_proj." in name for name in result)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata, tensors: metadata.update({"granite.logit_scale": 0.0}), "scaling"),
        (
            lambda metadata, tensors: metadata.update(
                {"granite.rope.dimension_count": _HEAD_DIM // 2}
            ),
            "rotary dimensions",
        ),
        (
            lambda metadata, tensors: metadata.update({"granite.deepstack_mapping": [0]}),
            "deep-stack",
        ),
        (
            lambda metadata, tensors: tensors.pop("blk.0.attn_v.weight"),
            "complete QKV",
        ),
        (
            lambda metadata, tensors: tensors.update(
                {"blk.0.ffn_gate.bias": np.zeros((_INTERMEDIATE,), np.float32)}
            ),
            "partial dense FFN bias",
        ),
        (
            lambda metadata, tensors: tensors.update(
                {"blk.0.ffn_gate_inp.weight": np.zeros((_EXPERTS, _HIDDEN), np.float32)}
            ),
            "unexpected",
        ),
        (
            lambda metadata, tensors: tensors.update(
                {"rope_freqs.weight": np.ones((_HEAD_DIM // 2,), np.float32)}
            ),
            "rope_freqs",
        ),
    ],
)
def test_granite_unsupported_or_malformed_contracts_fail_closed(mutation, message) -> None:
    metadata = _metadata()
    tensors = _tensors()
    mutation(metadata, tensors)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        _raise_for_invalid_granite_tensor_contract(_FakeGGUF(metadata, tensors))


def test_granite_tensor_names_close_for_dense_moe_and_fused_qkv() -> None:
    for moe, fused in ((False, False), (False, True), (True, False), (True, True)):
        model = _FakeGGUF(_metadata(moe=moe), _tensors(moe=moe, fused_qkv=fused))
        _raise_for_invalid_granite_tensor_contract(model)
        for name in model.tensor_names:
            mapped = map_gguf_to_hf_names(name, "granite")
            assert mapped is not None, name


@pytest.mark.parametrize(("moe", "fused_qkv"), [(False, False), (True, True)])
@pytest.mark.parametrize("quantized", [False, True])
def test_granite_tiny_gguf_builds_complete_graph(
    tmp_path: Path,
    moe: bool,
    fused_qkv: bool,
    quantized: bool,
) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = (
        tmp_path / f"granite-{'moe' if moe else 'dense'}-{'q4' if quantized else 'f32'}.gguf"
    )
    _write_tiny_granite(
        path,
        moe=moe,
        fused_qkv=fused_qkv,
        quantized=quantized,
    )
    package = build_from_gguf(path, keep_quantized=quantized)
    model = package["model"]
    names = set(model.graph.initializers)

    packed_ops = {"MatMulNBits", "BlockQuantizedMatMul"}
    assert bool(packed_ops & {node.op_type for node in model.graph}) is quantized
    assert any("self_attn.q_proj.weight" in name for name in names)
    assert any("self_attn.k_proj.weight" in name for name in names)
    assert any("self_attn.v_proj.weight" in name for name in names)
    assert not any("qkv_proj" in name for name in names)
    if moe:
        assert any(".mlp.experts.0.gate_proj.weight" in name for name in names)
        assert any(".mlp.shared_expert.gate_proj.weight" in name for name in names)
    else:
        assert any(".mlp.gate_proj.weight" in name for name in names)
        assert not any(".mlp.experts." in name for name in names)
