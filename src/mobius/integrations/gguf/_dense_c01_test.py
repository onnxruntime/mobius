# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mobius._configs import ArchitectureConfig
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import (
    _raise_for_invalid_dense_c01_tensor_contract,
    _reject_unsupported_quantization_preservation,
    _validate_gguf_model,
    build_from_gguf,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._errors import UnsupportedGGUFArchitectureError
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.integrations.gguf._tensor_processors import process_tensors
from mobius.models.chatglm import ChatGLMCausalLMModel
from mobius.models.phi import PhiCausalLMModel


class _FakeGGUF:
    def __init__(
        self,
        architecture: str,
        metadata: dict,
        tensors: dict[str, tuple[int, ...]],
        *,
        quantized: set[str] = frozenset(),
    ):
        self.architecture = architecture
        self.metadata = metadata
        self.tensor_names = list(tensors)
        self._tensors = tensors
        self._quantized = quantized

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        for name, shape in self._tensors.items():
            yield name, None, SimpleNamespace(value=2 if name in self._quantized else 0), shape

    def get_tensor(self, name):
        return np.ones(self._tensors[name], dtype=np.float32)


def _metadata(
    architecture: str,
    *,
    hidden: int = 8,
    intermediate: int = 16,
    layers: int = 2,
    heads: int = 2,
    kv_heads: int | None = None,
) -> dict:
    return {
        f"{architecture}.embedding_length": hidden,
        f"{architecture}.feed_forward_length": intermediate,
        f"{architecture}.block_count": layers,
        f"{architecture}.attention.head_count": heads,
        f"{architecture}.attention.head_count_kv": heads if kv_heads is None else kv_heads,
        f"{architecture}.context_length": 128,
        f"{architecture}.rope.freq_base": 10_000.0,
        f"{architecture}.rope.dimension_count": hidden // heads,
        f"{architecture}.vocab_size": 32,
    }


def _chatglm_tensors(metadata: dict) -> dict[str, tuple[int, ...]]:
    hidden = metadata["chatglm.embedding_length"]
    intermediate = metadata["chatglm.feed_forward_length"]
    layers = metadata["chatglm.block_count"]
    heads = metadata["chatglm.attention.head_count"]
    kv_heads = metadata["chatglm.attention.head_count_kv"]
    head_dim = hidden // heads
    tensors = {
        "token_embd.weight": (32, hidden),
        "output_norm.weight": (hidden,),
    }
    for layer in range(layers):
        prefix = f"blk.{layer}."
        tensors.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_qkv.weight": (
                    (heads + 2 * kv_heads) * head_dim,
                    hidden,
                ),
                prefix + "attn_output.weight": (hidden, hidden),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "ffn_up.weight": (2 * intermediate, hidden),
                prefix + "ffn_down.weight": (hidden, intermediate),
            }
        )
    return tensors


def _phi2_tensors(metadata: dict) -> dict[str, tuple[int, ...]]:
    hidden = metadata["phi2.embedding_length"]
    intermediate = metadata["phi2.feed_forward_length"]
    q_dim = hidden
    tensors = {
        "token_embd.weight": (32, hidden),
        "output_norm.weight": (hidden,),
        "output_norm.bias": (hidden,),
        "output.weight": (32, hidden),
        "output.bias": (32,),
    }
    for layer in range(metadata["phi2.block_count"]):
        prefix = f"blk.{layer}."
        tensors.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_norm.bias": (hidden,),
                prefix + "attn_q.weight": (q_dim, hidden),
                prefix + "attn_q.bias": (q_dim,),
                prefix + "attn_k.weight": (q_dim, hidden),
                prefix + "attn_k.bias": (q_dim,),
                prefix + "attn_v.weight": (q_dim, hidden),
                prefix + "attn_v.bias": (q_dim,),
                prefix + "attn_output.weight": (hidden, q_dim),
                prefix + "attn_output.bias": (hidden,),
                prefix + "ffn_up.weight": (intermediate, hidden),
                prefix + "ffn_up.bias": (intermediate,),
                prefix + "ffn_down.weight": (hidden, intermediate),
                prefix + "ffn_down.bias": (hidden,),
            }
        )
    return tensors


def _seed_oss_tensors(
    metadata: dict, *, include_qkv_bias: bool = False
) -> dict[str, tuple[int, ...]]:
    hidden = metadata["seed_oss.embedding_length"]
    intermediate = metadata["seed_oss.feed_forward_length"]
    layers = metadata["seed_oss.block_count"]
    heads = metadata["seed_oss.attention.head_count"]
    kv_heads = metadata["seed_oss.attention.head_count_kv"]
    head_dim = hidden // heads
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    tensors = {
        "token_embd.weight": (32, hidden),
        "output_norm.weight": (hidden,),
    }
    for layer in range(layers):
        prefix = f"blk.{layer}."
        tensors.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_q.weight": (q_dim, hidden),
                prefix + "attn_k.weight": (kv_dim, hidden),
                prefix + "attn_v.weight": (kv_dim, hidden),
                prefix + "attn_output.weight": (hidden, q_dim),
                prefix + "post_attention_norm.weight": (hidden,),
                prefix + "ffn_gate.weight": (intermediate, hidden),
                prefix + "ffn_up.weight": (intermediate, hidden),
                prefix + "ffn_down.weight": (hidden, intermediate),
            }
        )
        if include_qkv_bias:
            tensors.update(
                {
                    prefix + "attn_q.bias": (q_dim,),
                    prefix + "attn_k.bias": (kv_dim,),
                    prefix + "attn_v.bias": (kv_dim,),
                }
            )
    return tensors


@pytest.mark.parametrize(
    ("architecture", "model_type", "quantized_import"),
    [
        ("baichuan", "baichuan", "supported"),
        ("chatglm", "chatglm", "rejected"),
        ("phi2", "phi", "rejected"),
        ("seed_oss", "seed_oss", "supported"),
    ],
)
def test_dense_cohort_registry_runtime_is_deferred(
    architecture, model_type, quantized_import
) -> None:
    spec = get_arch_spec(architecture)
    assert spec.model_type == model_type
    assert spec.is_importable
    assert spec.runtime.value == "deferred"
    assert spec.quantized_import.value == quantized_import
    assert not spec.aliases


@pytest.mark.parametrize("architecture", ["apertus", "minicpm3"])
def test_dense_graph_mismatches_fail_before_config(architecture) -> None:
    model = _FakeGGUF(architecture, {}, {})
    with pytest.raises(UnsupportedGGUFArchitectureError, match="before config extraction"):
        _validate_gguf_model(model, source="synthetic.gguf")


@pytest.mark.parametrize(
    ("architecture", "layers", "intermediate"),
    [
        ("baichuan", 32, 16),
        ("chatglm", 2, 16),
        ("phi2", 2, 32),
        ("seed_oss", 64, 16),
    ],
)
def test_supported_dense_configs_build_registered_graphs(
    architecture, layers, intermediate
) -> None:
    from mobius._registry import registry
    from mobius.tasks import CausalLMTask

    metadata = _metadata(architecture, layers=layers, intermediate=intermediate)
    epsilon = (
        "attention.layer_norm_epsilon"
        if architecture == "phi2"
        else "attention.layer_norm_rms_epsilon"
    )
    metadata[f"{architecture}.{epsilon}"] = 1e-5
    config = gguf_to_config(_FakeGGUF(architecture, metadata, {"token_embd.weight": (32, 8)}))
    module = registry.get(config.model_type)(config)
    package = CausalLMTask().build(module, config)
    assert "model" in package


@pytest.mark.parametrize("architecture", ["chatglm", "phi2"])
def test_partial_rope_does_not_shrink_attention_head_width(architecture) -> None:
    intermediate = 32 if architecture == "phi2" else 16
    metadata = _metadata(architecture, layers=1, intermediate=intermediate)
    metadata[f"{architecture}.rope.dimension_count"] = 2
    epsilon = (
        "attention.layer_norm_epsilon"
        if architecture == "phi2"
        else "attention.layer_norm_rms_epsilon"
    )
    metadata[f"{architecture}.{epsilon}"] = 1e-5
    config = gguf_to_config(_FakeGGUF(architecture, metadata, {}))
    assert config.head_dim == 4
    assert config.partial_rotary_factor == pytest.approx(0.5)
    if architecture == "phi2":
        assert config.hidden_act == "gelu_new"


def test_baichuan_processor_exactly_inverts_converter_permutation() -> None:
    config = SimpleNamespace(
        _gguf_arch="baichuan",
        model_type="baichuan",
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    original = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    converted = original.reshape(2, 2, 2, 8).swapaxes(1, 2).reshape_as(original)
    result = process_tensors(
        {"model.layers.0.self_attn.q_proj.weight": converted.clone()}, config
    )
    torch.testing.assert_close(result["model.layers.0.self_attn.q_proj.weight"], original)


def test_chatglm_fused_projection_order_is_q_then_k_then_v_and_gate_then_up() -> None:
    config = ArchitectureConfig(
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=128,
        rope_type="default",
        hidden_act="silu",
    )
    module = ChatGLMCausalLMModel(config)
    qkv = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    gate_up = torch.arange(12 * 8, dtype=torch.float32).reshape(12, 8)
    result = module.preprocess_weights(
        {
            "transformer.encoder.layers.0.self_attention.query_key_value.weight": qkv,
            "transformer.encoder.layers.0.mlp.dense_h_to_4h.weight": gate_up,
        }
    )
    torch.testing.assert_close(result["model.layers.0.self_attn.q_proj.weight"], qkv[:8])
    torch.testing.assert_close(result["model.layers.0.self_attn.k_proj.weight"], qkv[8:12])
    torch.testing.assert_close(result["model.layers.0.self_attn.v_proj.weight"], qkv[12:])
    torch.testing.assert_close(result["model.layers.0.mlp.gate_proj.weight"], gate_up[:6])
    torch.testing.assert_close(result["model.layers.0.mlp.up_proj.weight"], gate_up[6:])


def test_chatglm_accepts_complete_split_qkv_layout() -> None:
    metadata = _metadata("chatglm", layers=1, kv_heads=1)
    metadata["chatglm.attention.layer_norm_rms_epsilon"] = 1e-5
    tensors = _chatglm_tensors(metadata)
    tensors.pop("blk.0.attn_qkv.weight")
    tensors.update(
        {
            "blk.0.attn_q.weight": (8, 8),
            "blk.0.attn_k.weight": (4, 8),
            "blk.0.attn_v.weight": (4, 8),
            "blk.0.attn_q.bias": (8,),
            "blk.0.attn_k.bias": (4,),
            "blk.0.attn_v.bias": (4,),
        }
    )
    model = _FakeGGUF("chatglm", metadata, tensors)
    _raise_for_invalid_dense_c01_tensor_contract(model)
    assert gguf_to_config(model).attn_qkv_bias


def test_phi2_fused_qkv_is_split_for_the_float_graph() -> None:
    metadata = _metadata("phi2", intermediate=32, layers=1)
    metadata["phi2.attention.layer_norm_epsilon"] = 1e-5
    tensors = _phi2_tensors(metadata)
    for suffix in ("q", "k", "v"):
        tensors.pop(f"blk.0.attn_{suffix}.weight")
        tensors.pop(f"blk.0.attn_{suffix}.bias")
    tensors["blk.0.attn_qkv.weight"] = (24, 8)
    tensors["blk.0.attn_qkv.bias"] = (24,)
    _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("phi2", metadata, tensors))

    config = gguf_to_config(_FakeGGUF("phi2", metadata, tensors))
    module = PhiCausalLMModel(config)
    qkv_weight = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
    qkv_bias = torch.arange(24, dtype=torch.float32)
    result = module.preprocess_weights(
        {
            "model.layers.0.self_attn.qkv_proj.weight": qkv_weight,
            "model.layers.0.self_attn.qkv_proj.bias": qkv_bias,
        }
    )
    for projection in ("q", "k", "v"):
        assert f"model.layers.0.self_attn.{projection}_proj.weight" in result
        assert f"model.layers.0.self_attn.{projection}_proj.bias" in result
    assert not any("qkv_proj" in name for name in result)


def test_quantized_chatglm_fused_tensors_can_only_dequantize() -> None:
    metadata = _metadata("chatglm", kv_heads=1)
    metadata["chatglm.attention.layer_norm_rms_epsilon"] = 1e-5
    tensors = _chatglm_tensors(metadata)
    model = _FakeGGUF(
        "chatglm",
        metadata,
        tensors,
        quantized={"blk.0.attn_qkv.weight"},
    )
    _raise_for_invalid_dense_c01_tensor_contract(model)
    with pytest.raises(ValueError, match=r"fused QKV and gate/up"):
        _reject_unsupported_quantization_preservation(
            model, "chatglm", preserve_quantization=True
        )
    _reject_unsupported_quantization_preservation(
        model, "chatglm", preserve_quantization=False
    )


def test_mixed_quantized_chatglm_is_rejected_before_graph() -> None:
    model = _FakeGGUF(
        "chatglm",
        _metadata("chatglm", layers=1),
        {
            "blk.0.attn_qkv.weight": (24, 8),
            "blk.0.ffn_up.weight": (32, 8),
            "blk.0.attn_output.weight": (8, 8),
        },
        quantized={"blk.0.attn_output.weight"},
    )
    with pytest.raises(ValueError, match=r"fused QKV and gate/up"):
        _reject_unsupported_quantization_preservation(
            model, "chatglm", preserve_quantization=True
        )
    _reject_unsupported_quantization_preservation(
        model, "chatglm", preserve_quantization=False
    )


def test_phi2_rejects_quantization_preservation() -> None:
    model = _FakeGGUF(
        "phi2",
        _metadata("phi2", layers=1),
        {"blk.0.attn_q.weight": (8, 8)},
        quantized={"blk.0.attn_q.weight"},
    )
    with pytest.raises(ValueError, match="float-only linear modules"):
        _reject_unsupported_quantization_preservation(
            model, "phi2", preserve_quantization=True
        )
    _reject_unsupported_quantization_preservation(model, "phi2", preserve_quantization=False)


def test_phi2_requires_complete_bias_closure() -> None:
    metadata = _metadata("phi2", intermediate=32, layers=1)
    metadata["phi2.attention.layer_norm_epsilon"] = 1e-5
    tensors = _phi2_tensors(metadata)
    _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("phi2", metadata, tensors))
    tensors.pop("blk.0.ffn_down.bias")
    with pytest.raises(ValueError, match=r"ffn_down\.bias"):
        _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("phi2", metadata, tensors))


def test_dense_cohort_rejects_mismatched_key_value_widths_before_graph() -> None:
    metadata = _metadata("chatglm", layers=1)
    metadata["chatglm.attention.layer_norm_rms_epsilon"] = 1e-5
    metadata["chatglm.attention.key_length"] = 4
    metadata["chatglm.attention.value_length"] = 2
    with pytest.raises(ValueError, match=r"equal positive key/value head widths"):
        _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("chatglm", metadata, {}))


def test_dense_cohort_missing_geometry_is_actionable_before_graph() -> None:
    metadata = _metadata("phi2", layers=1, intermediate=32)
    metadata.pop("phi2.embedding_length")
    with pytest.raises(ValueError, match=r"missing required dense metadata"):
        _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("phi2", metadata, {}))


def test_dense_cohort_known_skip_tensors_do_not_break_exact_closure() -> None:
    metadata = _metadata("phi2", intermediate=32, layers=1)
    metadata["phi2.attention.layer_norm_epsilon"] = 1e-5
    tensors = _phi2_tensors(metadata)
    tensors["blk.0.rope_freqs.weight"] = (4,)
    tensors["tokenizer.ggml.extra"] = (1,)
    _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("phi2", metadata, tensors))


def test_dense_cohort_still_rejects_unknown_extra_tensors() -> None:
    metadata = _metadata("phi2", intermediate=32, layers=1)
    metadata["phi2.attention.layer_norm_epsilon"] = 1e-5
    tensors = _phi2_tensors(metadata)
    tensors["blk.0.unowned.weight"] = (8,)
    with pytest.raises(ValueError, match=r"unexpected=.*unowned"):
        _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("phi2", metadata, tensors))


@pytest.mark.parametrize("context", [0, -1])
def test_dense_cohort_rejects_nonpositive_context_before_graph(context) -> None:
    metadata = _metadata("chatglm", layers=1)
    metadata["chatglm.context_length"] = context
    with pytest.raises(ValueError, match=r"invalid dense model geometry"):
        _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("chatglm", metadata, {}))


def test_seed_oss_rejects_unconsumed_attention_scale() -> None:
    metadata = _metadata("seed_oss", layers=64)
    metadata["seed_oss.attention.scale"] = 0.125
    with pytest.raises(ValueError, match=r"not consumed by the pinned llama\.cpp loader"):
        _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("seed_oss", metadata, {}))


def test_baichuan_13b_is_rejected_before_graph() -> None:
    metadata = _metadata("baichuan", layers=40)
    metadata["baichuan.attention.layer_norm_rms_epsilon"] = 1e-5
    with pytest.raises(ValueError, match=r"hardcoded ALiBi"):
        _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("baichuan", metadata, {}))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "blk.3.post_attention_norm.weight",
            "model.layers.3.post_attention_layernorm.weight",
        ),
        (
            "blk.3.attn_qkv.weight",
            "transformer.encoder.layers.3.self_attention.query_key_value.weight",
        ),
        ("output.bias", "lm_head.bias"),
    ],
)
def test_dense_cohort_tensor_mappings(name, expected) -> None:
    architecture = (
        "seed_oss" if "post_attention" in name else "chatglm" if "qkv" in name else "phi2"
    )
    assert map_gguf_to_hf_names(name, architecture) == expected


def test_seed_oss_complete_qkv_bias_family_builds_bias_graph() -> None:
    from mobius._registry import registry
    from mobius.tasks import CausalLMTask

    metadata = _metadata("seed_oss", layers=64, kv_heads=1)
    metadata["seed_oss.attention.layer_norm_rms_epsilon"] = 1e-5
    tensors = _seed_oss_tensors(metadata, include_qkv_bias=True)
    model = _FakeGGUF("seed_oss", metadata, tensors)
    _raise_for_invalid_dense_c01_tensor_contract(model)

    config = gguf_to_config(model)
    assert config.attn_qkv_bias is True
    package = CausalLMTask().build(registry.get(config.model_type)(config), config)
    initializers = set(package["model"].graph.initializers)
    assert "model.layers.0.self_attn.q_proj.bias" in initializers
    assert "model.layers.63.self_attn.v_proj.bias" in initializers


@pytest.mark.parametrize(
    "missing_bias",
    ["blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.63.attn_v.bias"],
)
def test_seed_oss_rejects_partial_qkv_bias_family(missing_bias) -> None:
    metadata = _metadata("seed_oss", layers=64, kv_heads=1)
    metadata["seed_oss.attention.layer_norm_rms_epsilon"] = 1e-5
    tensors = _seed_oss_tensors(metadata, include_qkv_bias=True)
    tensors.pop(missing_bias)
    with pytest.raises(ValueError, match=r"present for every projection in every layer"):
        _raise_for_invalid_dense_c01_tensor_contract(_FakeGGUF("seed_oss", metadata, tensors))


def test_seed_oss_output_presence_controls_effective_tie() -> None:
    metadata = _metadata("seed_oss", layers=64)
    metadata["seed_oss.attention.layer_norm_rms_epsilon"] = 1e-5
    tied = gguf_to_config(_FakeGGUF("seed_oss", metadata, {"token_embd.weight": (32, 8)}))
    untied = gguf_to_config(
        _FakeGGUF(
            "seed_oss",
            metadata,
            {"token_embd.weight": (32, 8), "output.weight": (32, 8)},
        )
    )
    assert tied.tie_word_embeddings is True
    assert untied.tie_word_embeddings is False


def test_seed_oss_packed_tied_head_has_one_canonical_owner(tmp_path) -> None:
    from mobius.integrations.gguf._builder_test import _write_quantized_gguf

    path = tmp_path / "seed-oss-tied-q4.gguf"
    _write_quantized_gguf(
        path,
        architecture="seed_oss",
        num_layers=64,
        num_heads=4,
        num_kv_heads=2,
        quantize_embedding=True,
        tie_embeddings=True,
    )

    model = build_from_gguf(path, keep_quantized=True)["model"]
    initializers = set(model.graph.initializers)
    assert {
        "model.embed_tokens.qweight",
        "model.embed_tokens.scales",
        "model.embed_tokens.zero_points",
    } <= initializers
    assert not any(name.startswith("lm_head.") for name in initializers)
