# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact contract and executable parity tests for the remaining GGUF MoE cohort."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch
from torch.nn import functional

from mobius._configs import ArchitectureConfig, GrokGGUFConfig, GroveMoEGGUFConfig
from mobius._registry import registry
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.gguf._arch_registry import Support, get_arch_spec
from mobius.integrations.gguf._builder import (
    _normalize_gguf_weights,
    _serialize_route_graph_config,
    build_from_gguf,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._remaining_moe import validate_remaining_moe_tensor_contract
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.models.moe import (
    GrokGGUFDecoderLayer,
    GroveMoEBlock,
    _PostRoPEQKNormAttention,
)
from mobius.tasks import CausalLMTask

_HIDDEN = 8
_DENSE = 12
_EXPERT_WIDTH = 6
_CHUNK_WIDTH = 4
_EXPERTS = 4
_TOP_K = 2
_HEADS = 2
_KV_HEADS = 1
_HEAD_DIM = 4
_LAYERS = 2
_VOCAB = 24

_IMPORT_EVIDENCE = {
    "grok": {
        "repository": "unsloth/grok-2-GGUF",
        "revision": "7b560aeb31bbd8ff7a026f3cfd981c0364024cbf",
        "filename": "Q2_K/grok-2-Q2_K-00001-of-00003.gguf",
        "size": 49_852_700_608,
        "logical_size": 100_050_489_536,
        "sha256": "64f6de31f793bb8a5788f55981834a5e252f532a2a1556a38ace873160ca9cff",
        "downloaded": 8 * 1024**2,
        "header": (477, 0, 3, 963),
    },
    "grovemoe": {
        "repository": "mradermacher/GroveMoE-Inst-GGUF",
        "revision": "4dec2e9eaf01d5150d5b45a7d7c95c6220aaad7f",
        "filename": "GroveMoE-Inst.Q2_K.gguf",
        "size": 12_240_087_808,
        "logical_size": 12_240_087_808,
        "sha256": "564172a63ed5b9d734dbbd5e82a07b1cab5d216a614c2e8d1312f168766d2973",
        "downloaded": 8 * 1024**2,
        "header": (723, None, None, None),
    },
    "hunyuan-moe": {
        "repository": "gabriellarson/Hunyuan-A13B-Instruct-GGUF",
        "revision": "1ab8a37a314dfa05d4dc11cd899a5dded62c0578",
        "filename": "Hunyuan-A13B-Instruct-IQ1_S.gguf",
        "size": 16_298_101_888,
        "logical_size": 16_298_101_888,
        "sha256": "b82ce44d0521dc75e419aed987e638a4d3b6f03bede3ab23c04923bf4e1a0b84",
        "downloaded": 8 * 1024**2,
        "header": (482, None, None, None),
    },
}


class _FakeGGUF:
    def __init__(
        self,
        architecture: str,
        metadata: dict[str, object],
        tensors: dict[str, tuple[int, ...]],
    ):
        self.architecture = architecture
        self.metadata = metadata
        self._tensors = tensors

    @property
    def tensor_names(self) -> list[str]:
        return list(self._tensors)

    def get_metadata(self, key: str, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        qtype = SimpleNamespace(value=0, name="F32")
        for name, shape in self._tensors.items():
            yield name, None, qtype, shape


def _attention_shapes(layer: int) -> dict[str, tuple[int, ...]]:
    prefix = f"blk.{layer}."
    return {
        prefix + "attn_q.weight": (_HEADS * _HEAD_DIM, _HIDDEN),
        prefix + "attn_k.weight": (_KV_HEADS * _HEAD_DIM, _HIDDEN),
        prefix + "attn_v.weight": (_KV_HEADS * _HEAD_DIM, _HIDDEN),
        prefix + "attn_output.weight": (_HIDDEN, _HEADS * _HEAD_DIM),
    }


def _fixture(architecture: str) -> _FakeGGUF:
    metadata: dict[str, object] = {
        f"{architecture}.context_length": 32,
        f"{architecture}.embedding_length": _HIDDEN,
        f"{architecture}.feed_forward_length": _DENSE,
        f"{architecture}.block_count": _LAYERS,
        f"{architecture}.attention.head_count": _HEADS,
        f"{architecture}.attention.head_count_kv": _KV_HEADS,
        f"{architecture}.attention.key_length": _HEAD_DIM,
        f"{architecture}.attention.value_length": _HEAD_DIM,
        f"{architecture}.attention.layer_norm_rms_epsilon": 1e-5,
        f"{architecture}.rope.dimension_count": _HEAD_DIM,
        f"{architecture}.rope.freq_base": 10_000.0,
        f"{architecture}.expert_count": _EXPERTS,
        f"{architecture}.expert_used_count": _TOP_K,
        f"{architecture}.vocab_size": _VOCAB,
    }
    tensors: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (_VOCAB, _HIDDEN),
        "output_norm.weight": (_HIDDEN,),
        "output.weight": (_VOCAB, _HIDDEN),
    }

    if architecture == "grok":
        metadata.update(
            {
                "grok.expert_feed_forward_length": _EXPERT_WIDTH,
                "grok.attention.layer_norm_epsilon": 1e-12,
                "grok.embedding_scale": 2.0,
                "grok.attention.output_scale": 0.5,
                "grok.logit_scale": 0.25,
                "grok.attn_logit_softcapping": 3.0,
                "grok.router_logit_softcapping": 7.0,
                "grok.final_logit_softcapping": 4.0,
            }
        )
        for layer in range(_LAYERS):
            prefix = f"blk.{layer}."
            tensors.update(
                {
                    **_attention_shapes(layer),
                    prefix + "attn_norm.weight": (_HIDDEN,),
                    prefix + "attn_output_norm.weight": (_HIDDEN,),
                    prefix + "ffn_norm.weight": (_HIDDEN,),
                    prefix + "ffn_gate.weight": (_DENSE, _HIDDEN),
                    prefix + "ffn_up.weight": (_DENSE, _HIDDEN),
                    prefix + "ffn_down.weight": (_HIDDEN, _DENSE),
                    prefix + "ffn_gate_inp.weight": (_EXPERTS, _HIDDEN),
                    prefix + "ffn_gate_exps.weight": (
                        _EXPERTS,
                        _EXPERT_WIDTH,
                        _HIDDEN,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        _EXPERTS,
                        _EXPERT_WIDTH,
                        _HIDDEN,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        _EXPERTS,
                        _HIDDEN,
                        _EXPERT_WIDTH,
                    ),
                    prefix + "layer_output_norm.weight": (_HIDDEN,),
                }
            )
        return _FakeGGUF(architecture, metadata, tensors)

    if architecture == "grovemoe":
        metadata.update(
            {
                "grovemoe.expert_feed_forward_length": _EXPERT_WIDTH,
                "grovemoe.expert_chunk_feed_forward_length": _CHUNK_WIDTH,
                "grovemoe.experts_per_group": 2,
                "grovemoe.expert_group_scale": 0.05,
            }
        )
        for layer in range(_LAYERS):
            prefix = f"blk.{layer}."
            tensors.update(
                {
                    **_attention_shapes(layer),
                    prefix + "attn_norm.weight": (_HIDDEN,),
                    prefix + "attn_q_norm.weight": (_HEAD_DIM,),
                    prefix + "attn_k_norm.weight": (_HEAD_DIM,),
                    prefix + "ffn_norm.weight": (_HIDDEN,),
                    prefix + "ffn_gate_inp.weight": (_EXPERTS, _HIDDEN),
                    prefix + "ffn_gate_exps.weight": (
                        _EXPERTS,
                        _EXPERT_WIDTH,
                        _HIDDEN,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        _EXPERTS,
                        _EXPERT_WIDTH,
                        _HIDDEN,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        _EXPERTS,
                        _HIDDEN,
                        _EXPERT_WIDTH,
                    ),
                    prefix + "ffn_gate_chexps.weight": (
                        _EXPERTS // 2,
                        _CHUNK_WIDTH,
                        _HIDDEN,
                    ),
                    prefix + "ffn_up_chexps.weight": (
                        _EXPERTS // 2,
                        _CHUNK_WIDTH,
                        _HIDDEN,
                    ),
                    prefix + "ffn_down_chexps.weight": (
                        _EXPERTS // 2,
                        _HIDDEN,
                        _CHUNK_WIDTH,
                    ),
                }
            )
        return _FakeGGUF(architecture, metadata, tensors)

    assert architecture == "hunyuan-moe"
    metadata.update(
        {
            "hunyuan-moe.expert_feed_forward_length": _DENSE,
            "hunyuan-moe.expert_shared_feed_forward_length": _DENSE,
            "hunyuan-moe.expert_shared_count": 1,
        }
    )
    for layer in range(_LAYERS):
        prefix = f"blk.{layer}."
        tensors.update(
            {
                **_attention_shapes(layer),
                prefix + "attn_norm.weight": (_HIDDEN,),
                prefix + "attn_q_norm.weight": (_HEAD_DIM,),
                prefix + "attn_k_norm.weight": (_HEAD_DIM,),
                prefix + "ffn_norm.weight": (_HIDDEN,),
                prefix + "ffn_gate_inp.weight": (_EXPERTS, _HIDDEN),
                prefix + "ffn_gate_exps.weight": (_EXPERTS, _DENSE, _HIDDEN),
                prefix + "ffn_up_exps.weight": (_EXPERTS, _DENSE, _HIDDEN),
                prefix + "ffn_down_exps.weight": (_EXPERTS, _HIDDEN, _DENSE),
                prefix + "ffn_gate_shexp.weight": (_DENSE, _HIDDEN),
                prefix + "ffn_up_shexp.weight": (_DENSE, _HIDDEN),
                prefix + "ffn_down_shexp.weight": (_HIDDEN, _DENSE),
            }
        )
    return _FakeGGUF(architecture, metadata, tensors)


def _write_fixture(
    path: Path,
    architecture: str,
    *,
    source: _FakeGGUF | None = None,
) -> None:
    from gguf import GGUFWriter

    source = source or _fixture(architecture)
    if source.architecture != architecture:
        raise ValueError("Synthetic GGUF source architecture does not match the writer")
    writer = GGUFWriter(str(path), architecture)
    for key, value in source.metadata.items():
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_float32(key, float(value))
    rng = np.random.default_rng(31)
    for name, shape in source._tensors.items():
        writer.add_tensor(
            name,
            rng.normal(0.0, 0.02, shape).astype(np.float32),
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _fill_parameters(module, model: ir.Model, *, seed: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    weights: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        values = rng.normal(0.0, 0.2, tuple(parameter.shape)).astype(np.float32)
        if name.endswith("norm.weight"):
            values += 1.0
        model.graph.initializers[name].const_value = ir.tensor(values)
        weights[name] = torch.from_numpy(values)
    return weights


def _rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return normalized * weight


def _linear(value: torch.Tensor, weights: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return value @ weights[f"{name}.weight"].T


def _mlp(
    value: torch.Tensor,
    weights: dict[str, torch.Tensor],
    prefix: str,
    *,
    activation: str,
) -> torch.Tensor:
    gate = _linear(value, weights, f"{prefix}.gate_proj")
    up = _linear(value, weights, f"{prefix}.up_proj")
    if activation == "gelu":
        gate = functional.gelu(gate, approximate="tanh")
    else:
        gate = functional.silu(gate)
    return _linear(gate * up, weights, f"{prefix}.down_proj")


def _causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    softcap: float = 0.0,
) -> torch.Tensor:
    batch, sequence, _ = query.shape
    query = query.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, sequence, kv_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, sequence, kv_heads, head_dim).transpose(1, 2)
    repeats = heads // kv_heads
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    causal = torch.triu(
        torch.ones(sequence, sequence, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal, float("-inf"))
    output = torch.matmul(torch.softmax(scores, dim=-1), value)
    return output.transpose(1, 2).reshape(batch, sequence, heads * head_dim)


@pytest.mark.parametrize("architecture", ["grok", "grovemoe", "hunyuan-moe"])
def test_remaining_moe_routes_are_float_only_and_runtime_deferred(
    architecture: str,
) -> None:
    spec = get_arch_spec(architecture)
    assert spec.is_importable
    assert spec.module_type == architecture.replace("-", "_") + "_gguf"
    assert spec.quantized_import is Support.REJECTED
    assert spec.runtime is Support.DEFERRED
    assert "keep_quantized=False" in (spec.reason or "")


def test_import_evidence_is_immutable_and_bounded() -> None:
    assert set(_IMPORT_EVIDENCE) == {"grok", "grovemoe", "hunyuan-moe"}
    assert sum(record["downloaded"] for record in _IMPORT_EVIDENCE.values()) < 16 * 1024**3
    for record in _IMPORT_EVIDENCE.values():
        assert len(record["revision"]) == 40
        assert len(record["sha256"]) == 64
        int(record["sha256"], 16)
        assert record["downloaded"] <= record["size"]
        assert len(record["header"]) == 4
    assert _IMPORT_EVIDENCE["grok"]["logical_size"] > 16 * 1024**3
    assert _IMPORT_EVIDENCE["grovemoe"]["logical_size"] < 16 * 1024**3
    assert _IMPORT_EVIDENCE["hunyuan-moe"]["logical_size"] < 16 * 1024**3


@pytest.mark.parametrize("architecture", ["grok", "grovemoe", "hunyuan-moe"])
def test_exact_tensor_closure_config_and_graph(architecture: str) -> None:
    source = _fixture(architecture)
    validate_remaining_moe_tensor_contract(source)
    config = gguf_to_config(source)
    spec = get_arch_spec(architecture)
    module = registry.get(spec.module_type)(config)
    graph = CausalLMTask().build(module, config)["model"].graph

    mapped = {
        map_gguf_to_hf_names(name, architecture)
        for name in source.tensor_names
        if "_exps." not in name and "_chexps." not in name
    }
    parameters = {name for name, _ in module.named_parameters()}
    assert mapped <= parameters
    assert graph.outputs[0].name == "logits"
    assert not any("com.microsoft" in node.domain for node in graph)
    if architecture == "grok":
        attention = next(node for node in graph if node.op_type == "Attention")
        assert attention.attributes["scale"].value == pytest.approx(0.5)
        assert attention.attributes["softcap"].value == pytest.approx(3.0)
        assert all(
            node.attributes["epsilon"].value == pytest.approx(1e-5)
            for node in graph
            if node.op_type == "RMSNormalization"
        )
    elif architecture == "grovemoe":
        assert any("chunk_experts.0" in name for name in graph.initializers)
    else:
        nodes = list(graph)
        rotary_index = next(
            i for i, node in enumerate(nodes) if node.op_type == "RotaryEmbedding"
        )
        q_norm_index = next(
            i
            for i, node in enumerate(nodes)
            if node.op_type == "RMSNormalization"
            and any(
                value.name and value.name.endswith("self_attn.q_norm.weight")
                for value in node.inputs
                if value is not None
            )
        )
        assert rotary_index < q_norm_index


def test_grok_forward_preserves_optional_base_outputs() -> None:
    config = dataclasses.replace(
        gguf_to_config(_fixture("grok")),
        output_final_hidden_state=True,
    )
    module = registry.get(get_arch_spec("grok").module_type)(config)
    graph = CausalLMTask().build(module, config)["model"].graph
    assert "mtp_seed" in {output.name for output in graph.outputs}


def _run_synthetic_package(package) -> np.ndarray:
    session = OnnxModelSession(package["model"], device="cpu")
    feeds = {
        "input_ids": np.array([[1, 2]], dtype=np.int64),
        "attention_mask": np.ones((1, 2), dtype=np.int64),
        "position_ids": np.array([[0, 1]], dtype=np.int64),
    }
    for layer in range(_LAYERS):
        shape = (1, _KV_HEADS, 0, _HEAD_DIM)
        feeds[f"past_key_values.{layer}.key"] = np.empty(shape, dtype=np.float32)
        feeds[f"past_key_values.{layer}.value"] = np.empty(shape, dtype=np.float32)
    try:
        return session.run(feeds)["logits"]
    finally:
        session.close()


@pytest.mark.parametrize("architecture", ["grok", "grovemoe", "hunyuan-moe"])
def test_synthetic_gguf_build_executes_end_to_end(
    architecture: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"{architecture}.gguf"
    _write_fixture(path, architecture)
    package = build_from_gguf(path, keep_quantized=False)
    logits = _run_synthetic_package(package)
    assert logits.shape == (1, 2, _VOCAB)
    assert np.isfinite(logits).all()


def test_grok_ungated_dense_ffn_builds_applies_weights_and_executes(
    tmp_path: Path,
) -> None:
    source = _fixture("grok")
    for layer in range(_LAYERS):
        del source._tensors[f"blk.{layer}.ffn_gate.weight"]
    validate_remaining_moe_tensor_contract(source)
    config = gguf_to_config(source)
    assert config.has_dense_ffn
    assert not config.has_gated_dense_ffn

    path = tmp_path / "grok-ungated-dense.gguf"
    _write_fixture(path, "grok", source=source)
    package = build_from_gguf(path, keep_quantized=False)
    initializers = package["model"].graph.initializers
    for layer in range(_LAYERS):
        prefix = f"model.layers.{layer}.residual_mlp."
        assert not any(name.startswith(prefix + "gate_proj.") for name in initializers)
        assert initializers[prefix + "up_proj.weight_t"].const_value is not None
        assert initializers[prefix + "down_proj.weight_t"].const_value is not None
    logits = _run_synthetic_package(package)
    assert logits.shape == (1, 2, _VOCAB)
    assert np.isfinite(logits).all()


def test_grok_dense_ffn_rejects_partial_and_mixed_gating() -> None:
    partial = _fixture("grok")
    del partial._tensors["blk.0.ffn_down.weight"]
    with pytest.raises(ValueError, match="partial dense FFN tensor family"):
        validate_remaining_moe_tensor_contract(partial)

    gate_only = _fixture("grok")
    del gate_only._tensors["blk.0.ffn_up.weight"]
    del gate_only._tensors["blk.0.ffn_down.weight"]
    with pytest.raises(ValueError, match="gate without complete up/down"):
        validate_remaining_moe_tensor_contract(gate_only)

    mixed = _fixture("grok")
    del mixed._tensors["blk.0.ffn_gate.weight"]
    with pytest.raises(ValueError, match="dense FFN gating topology"):
        validate_remaining_moe_tensor_contract(mixed)


@pytest.mark.parametrize("architecture", ["grok", "grovemoe", "hunyuan-moe"])
def test_stacked_experts_unpack_to_graph_parameters(architecture: str) -> None:
    source = _fixture(architecture)
    config = gguf_to_config(source)
    module = registry.get(get_arch_spec(architecture).module_type)(config)
    stacked = {
        map_gguf_to_hf_names(name, architecture): torch.zeros(shape)
        for name, shape in source._tensors.items()
        if "_exps." in name or "_chexps." in name
    }
    normalized = _normalize_gguf_weights(stacked, architecture, config)
    expected = {
        name
        for name, _ in module.named_parameters()
        if ".experts." in name or ".chunk_experts." in name
    }
    assert set(normalized) == expected


@pytest.mark.parametrize("architecture", ["grok", "grovemoe", "hunyuan-moe"])
@pytest.mark.parametrize("mutation", ["missing", "unexpected", "malformed"])
def test_tensor_contract_rejects_mutations(architecture: str, mutation: str) -> None:
    source = _fixture(architecture)
    if mutation == "missing":
        del source._tensors["blk.0.attn_norm.weight"]
    elif mutation == "unexpected":
        source._tensors["blk.0.unowned.weight"] = (_HIDDEN, _HIDDEN)
    else:
        source._tensors["blk.0.ffn_gate_inp.weight"] = (1, _HIDDEN)
    with pytest.raises(ValueError, match=r"missing|unexpected|malformed"):
        validate_remaining_moe_tensor_contract(source)


def test_suffix_exact_mapping_rejects_legacy_and_cross_architecture_tensors() -> None:
    assert map_gguf_to_hf_names("blk.0.ffn_gate_exps.scale", "grok") == (
        "model.layers.0.mlp.experts.gate_proj.scale"
    )
    assert map_gguf_to_hf_names("blk.0.ffn_gate.0.weight", "grok") is None
    assert map_gguf_to_hf_names("blk.0.ffn_gate_chexps.scale", "grovemoe") is None
    assert map_gguf_to_hf_names("blk.0.ffn_gate_shexp.weight", "grovemoe") is None
    assert map_gguf_to_hf_names("blk.0.ffn_gate_chexps.weight", "hunyuan-moe") is None


def test_auxiliary_sidecars_are_owned_by_the_fail_closed_preflight(tmp_path: Path) -> None:
    source = _fixture("grok")
    source._tensors["blk.0.ffn_up_exps.scale"] = (_EXPERTS,)
    validate_remaining_moe_tensor_contract(source)

    path = tmp_path / "grok-sidecar.gguf"
    _write_fixture(path, "grok", source=source)
    with pytest.raises(ValueError, match="cannot represent GGUF scale/input_scale sidecars"):
        build_from_gguf(path, keep_quantized=False)

    unowned = _fixture("grok")
    unowned._tensors["blk.0.unowned.scale"] = (1,)
    with pytest.raises(ValueError, match="unexpected"):
        validate_remaining_moe_tensor_contract(unowned)


def test_config_closure_preserves_architecture_semantics() -> None:
    grok = gguf_to_config(_fixture("grok"))
    assert isinstance(grok, GrokGGUFConfig)
    assert grok.rms_norm_eps == pytest.approx(1e-5)
    assert grok.embedding_scale == pytest.approx(2.0)
    assert grok.attention_output_scale == pytest.approx(0.5)
    assert grok.logit_output_scale == pytest.approx(0.25)
    assert grok.attn_logit_softcapping == pytest.approx(3.0)
    assert grok.final_logit_softcapping == pytest.approx(4.0)
    assert grok.has_dense_ffn and grok.has_gated_dense_ffn and grok.has_gated_experts

    grove = gguf_to_config(_fixture("grovemoe"))
    assert isinstance(grove, GroveMoEGGUFConfig)
    assert grove.attn_qk_norm and not grove.attn_qk_norm_full
    assert grove.chunk_expert_intermediate_size == _CHUNK_WIDTH
    assert grove.experts_per_group == 2
    assert grove.expert_group_scale == pytest.approx(0.05)

    hunyuan = gguf_to_config(_fixture("hunyuan-moe"))
    assert hunyuan.attn_qk_norm and not hunyuan.attn_qk_norm_full
    assert hunyuan.moe_intermediate_size == _DENSE
    assert hunyuan.shared_expert_intermediate_size == _DENSE
    assert hunyuan.norm_topk_prob


def test_metadata_defaults_match_pinned_loaders() -> None:
    grok_source = _fixture("grok")
    for key in (
        "grok.expert_feed_forward_length",
        "grok.embedding_scale",
        "grok.attention.output_scale",
        "grok.logit_scale",
        "grok.attn_logit_softcapping",
        "grok.router_logit_softcapping",
        "grok.final_logit_softcapping",
    ):
        del grok_source.metadata[key]
    grok = gguf_to_config(grok_source)
    assert grok.moe_intermediate_size == _DENSE
    assert grok.embedding_scale == pytest.approx(78.38367176906169)
    assert grok.attention_output_scale == pytest.approx(0.08838834764831845)
    assert grok.logit_output_scale == pytest.approx(0.5773502691896257)
    assert grok.attn_logit_softcapping == pytest.approx(30.0)
    assert grok.router_logit_softcapping == pytest.approx(30.0)
    assert grok.final_logit_softcapping == pytest.approx(0.0)

    grove_source = _fixture("grovemoe")
    del grove_source.metadata["grovemoe.expert_chunk_feed_forward_length"]
    assert gguf_to_config(grove_source).chunk_expert_intermediate_size == _HEAD_DIM

    hunyuan_source = _fixture("hunyuan-moe")
    del hunyuan_source.metadata["hunyuan-moe.expert_shared_feed_forward_length"]
    assert gguf_to_config(hunyuan_source).shared_expert_intermediate_size == _DENSE


def test_inconsistent_metadata_fails_closed() -> None:
    grok = _fixture("grok")
    grok.metadata["grok.attn_logit_softcapping"] = 0.0
    with pytest.raises(ValueError, match="attn_logit_softcapping must be positive"):
        gguf_to_config(grok)

    grove = _fixture("grovemoe")
    grove.metadata["grovemoe.experts_per_group"] = 3
    with pytest.raises(ValueError, match="must divide expert_count"):
        validate_remaining_moe_tensor_contract(grove)

    hunyuan = _fixture("hunyuan-moe")
    hunyuan.metadata["hunyuan-moe.expert_feed_forward_length"] = _EXPERT_WIDTH
    with pytest.raises(ValueError, match="must equal feed_forward_length"):
        validate_remaining_moe_tensor_contract(hunyuan)


def test_grok_yarn_defaults_and_fingerprint_isolation() -> None:
    source = _fixture("grok")
    source.metadata.update(
        {
            "grok.rope.scaling.type": "yarn",
            "grok.rope.scaling.factor": 16.0,
            "grok.rope.scaling.original_context_length": 8192,
        }
    )
    config = gguf_to_config(source)
    assert config.rope_scaling["beta_fast"] == pytest.approx(8.0)
    assert config.rope_scaling["beta_slow"] == pytest.approx(1.0)

    baseline = _serialize_route_graph_config(config, "grok")
    assert json.loads(baseline)["rms_norm_eps"] == pytest.approx(1e-5)
    changed_ignored = _serialize_route_graph_config(
        dataclasses.replace(config, router_logit_softcapping=99.0),
        "grok",
    )
    changed_consumed = _serialize_route_graph_config(
        dataclasses.replace(config, attention_output_scale=0.25),
        "grok",
    )
    assert changed_ignored == baseline
    assert (
        hashlib.sha256(changed_consumed.encode()).digest()
        != hashlib.sha256(baseline.encode()).digest()
    )


def test_grok_decoder_matches_pinned_synthetic_reference() -> None:
    parsed = gguf_to_config(_fixture("grok"))
    assert isinstance(parsed, GrokGGUFConfig)
    config = dataclasses.replace(
        parsed,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_local_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=3,
        hidden_act="gelu_new",
        attention_output_scale=0.5,
        attn_logit_softcapping=2.0,
        has_dense_ffn=True,
        has_gated_dense_ffn=True,
        has_gated_experts=True,
    )
    eps = config.rms_norm_eps
    layer = GrokGGUFDecoderLayer(config)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 4])
    output, _ = layer(op, hidden, None, None, None)
    output.name = "output"
    graph.outputs.append(output)
    model = ir.Model(graph, ir_version=10)
    weights = _fill_parameters(layer, model, seed=11)
    hidden_value = torch.tensor(
        [[[0.3, -0.2, 0.4, 0.7], [-0.5, 0.8, 0.1, -0.4]]],
        dtype=torch.float32,
    )

    normalized = _rms_norm(hidden_value, weights["input_layernorm.weight"], eps)
    query = _linear(normalized, weights, "self_attn.q_proj")
    key = _linear(normalized, weights, "self_attn.k_proj")
    value = _linear(normalized, weights, "self_attn.v_proj")
    attention = _causal_attention(
        query,
        key,
        value,
        heads=2,
        kv_heads=1,
        head_dim=2,
        scale=0.5,
        softcap=2.0,
    )
    attention = _linear(attention, weights, "self_attn.o_proj")
    attention = _rms_norm(
        attention,
        weights["attention_output_layernorm.weight"],
        eps,
    )
    ffn_input = hidden_value + attention
    normalized = _rms_norm(
        ffn_input,
        weights["pre_feedforward_layernorm.weight"],
        eps,
    )
    probabilities = torch.softmax(_linear(normalized, weights, "mlp.gate"), dim=-1)
    routing_weights, selected = torch.topk(probabilities, 2, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routed = torch.zeros_like(normalized)
    for expert in range(3):
        expert_output = _mlp(
            normalized,
            weights,
            f"mlp.experts.{expert}",
            activation="gelu",
        )
        weight = (routing_weights * (selected == expert)).sum(dim=-1, keepdim=True)
        routed += expert_output * weight
    dense = _mlp(normalized, weights, "residual_mlp", activation="gelu")
    ffn_output = (dense + routed) * (2.0**0.5 / 2.0)
    ffn_output = _rms_norm(
        ffn_output,
        weights["post_feedforward_layernorm.weight"],
        eps,
    )
    expected = ffn_input + ffn_output

    session = OnnxModelSession(model, device="cpu")
    try:
        actual = session.run({"hidden": hidden_value.numpy()})["output"]
    finally:
        session.close()
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)


def test_grovemoe_block_matches_pinned_two_bank_reference() -> None:
    config = GroveMoEGGUFConfig(
        hidden_size=4,
        intermediate_size=6,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=3,
        chunk_expert_intermediate_size=2,
        experts_per_group=2,
        expert_group_scale=0.2,
        hidden_act="silu",
    )
    block = GroveMoEBlock(config)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 4])
    output = block(op, hidden)
    output.name = "output"
    graph.outputs.append(output)
    model = ir.Model(graph, ir_version=10)
    weights = _fill_parameters(block, model, seed=17)
    hidden_value = torch.tensor(
        [[[0.4, -0.3, 0.8, 0.1], [-0.2, 0.7, -0.5, 0.6]]],
        dtype=torch.float32,
    )

    logits = _linear(hidden_value, weights, "gate")
    probabilities = torch.softmax(logits, dim=-1)
    selected = torch.topk(torch.sigmoid(logits), 2, dim=-1).indices
    routing_weights = probabilities.gather(-1, selected)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True).clamp_min(6.103515625e-5)
    main = torch.zeros_like(hidden_value)
    for expert in range(4):
        expert_output = _mlp(
            hidden_value,
            weights,
            f"experts.{expert}",
            activation="silu",
        )
        weight = (routing_weights * (selected == expert)).sum(dim=-1, keepdim=True)
        main += expert_output * weight

    chunk_source = torch.topk(torch.sigmoid(logits), 2, dim=-1).indices
    chunk_selected = (chunk_source.float() * 0.5).to(torch.int64)
    chunk_weights = probabilities.gather(-1, chunk_selected)
    chunk_weights /= chunk_weights.sum(dim=-1, keepdim=True).clamp_min(6.103515625e-5)
    chunk = torch.zeros_like(main)
    for expert in range(2):
        expert_output = _mlp(
            main,
            weights,
            f"chunk_experts.{expert}",
            activation="silu",
        )
        weight = (chunk_weights * (chunk_selected == expert)).sum(dim=-1, keepdim=True)
        chunk += expert_output * weight
    expected = main + 0.2 * chunk

    session = OnnxModelSession(model, device="cpu")
    try:
        actual = session.run({"hidden": hidden_value.numpy()})["output"]
    finally:
        session.close()
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)


def test_hunyuan_attention_matches_post_rope_qk_norm_reference() -> None:
    config = ArchitectureConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attn_qk_norm=True,
        attn_qk_norm_full=False,
        rms_norm_eps=1e-5,
        rope_interleave=False,
    )
    attention = _PostRoPEQKNormAttention(config)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 8])
    cos = create_test_input(builder, "cos", [1, 2, 2])
    sin = create_test_input(builder, "sin", [1, 2, 2])
    output, _ = attention(op, hidden, None, (cos, sin))
    output.name = "output"
    graph.outputs.append(output)
    model = ir.Model(graph, ir_version=10)
    weights = _fill_parameters(attention, model, seed=23)
    hidden_value = torch.tensor(
        [
            [
                [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8],
                [-0.4, 0.2, 0.5, -0.1, 0.8, -0.7, 0.3, 0.6],
            ]
        ],
        dtype=torch.float32,
    )
    angles = torch.tensor([[[0.2, 0.5], [0.7, 1.1]]], dtype=torch.float32)
    cos_value, sin_value = torch.cos(angles), torch.sin(angles)

    def rotate(value: torch.Tensor, heads: int) -> torch.Tensor:
        value = value.reshape(1, 2, heads, 4)
        first, second = value[..., :2], value[..., 2:]
        cos_heads = cos_value.unsqueeze(2)
        sin_heads = sin_value.unsqueeze(2)
        return torch.cat(
            (first * cos_heads - second * sin_heads, second * cos_heads + first * sin_heads),
            dim=-1,
        )

    query = rotate(_linear(hidden_value, weights, "q_proj"), 2)
    key = rotate(_linear(hidden_value, weights, "k_proj"), 1)
    query = _rms_norm(query, weights["q_norm.weight"], 1e-5).reshape(1, 2, 8)
    key = _rms_norm(key, weights["k_norm.weight"], 1e-5).reshape(1, 2, 4)
    value = _linear(hidden_value, weights, "v_proj")
    expected = _causal_attention(
        query,
        key,
        value,
        heads=2,
        kv_heads=1,
        head_dim=4,
        scale=0.5,
    )
    expected = _linear(expected, weights, "o_proj")

    session = OnnxModelSession(model, device="cpu")
    try:
        actual = session.run(
            {
                "hidden": hidden_value.numpy(),
                "cos": cos_value.numpy(),
                "sin": sin_value.numpy(),
            }
        )["output"]
    finally:
        session.close()
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)
