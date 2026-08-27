# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact GGUF contract and end-to-end graph tests for Hunyuan-V3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from mobius.integrations.gguf._hy_v3 import validate_hy_v3_tensor_contract

_H = 16
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 4
_DENSE = 24
_EXPERTS = 4
_TOP_K = 2
_EXPERT_WIDTH = 8
_VOCAB = 32
_BLOCKS = 3


def _metadata() -> dict[str, object]:
    return {
        "hy_v3.context_length": 32,
        "hy_v3.embedding_length": _H,
        "hy_v3.feed_forward_length": _DENSE,
        "hy_v3.block_count": _BLOCKS,
        "hy_v3.attention.head_count": _HEADS,
        "hy_v3.attention.head_count_kv": _KV_HEADS,
        "hy_v3.attention.key_length": _HEAD_DIM,
        "hy_v3.attention.value_length": _HEAD_DIM,
        "hy_v3.attention.layer_norm_rms_epsilon": 1e-5,
        "hy_v3.rope.freq_base": 10_000.0,
        "hy_v3.rope.dimension_count": _HEAD_DIM,
        "hy_v3.expert_count": _EXPERTS,
        "hy_v3.expert_used_count": _TOP_K,
        "hy_v3.expert_feed_forward_length": _EXPERT_WIDTH,
        "hy_v3.expert_shared_feed_forward_length": _EXPERT_WIDTH,
        "hy_v3.expert_gating_func": 2,
        "hy_v3.expert_weights_norm": True,
        "hy_v3.expert_weights_scale": 2.826,
        "hy_v3.nextn_predict_layers": 1,
        "hy_v3.vocab_size": _VOCAB,
    }


def _tensor_shapes(
    *, fused_experts: bool = True, qkv_bias: bool = False
) -> dict[str, tuple[int, ...]]:
    shapes = {
        "token_embd.weight": (_VOCAB, _H),
        "output_norm.weight": (_H,),
        "output.weight": (_VOCAB, _H),
    }
    for layer in range(_BLOCKS):
        prefix = f"blk.{layer}."
        shapes.update(
            {
                prefix + "attn_norm.weight": (_H,),
                prefix + "attn_q_norm.weight": (_HEAD_DIM,),
                prefix + "attn_k_norm.weight": (_HEAD_DIM,),
                prefix + "attn_q.weight": (_HEADS * _HEAD_DIM, _H),
                prefix + "attn_k.weight": (_KV_HEADS * _HEAD_DIM, _H),
                prefix + "attn_v.weight": (_KV_HEADS * _HEAD_DIM, _H),
                prefix + "attn_output.weight": (_H, _HEADS * _HEAD_DIM),
                prefix + "ffn_norm.weight": (_H,),
            }
        )
        if qkv_bias:
            shapes.update(
                {
                    prefix + "attn_q.bias": (_HEADS * _HEAD_DIM,),
                    prefix + "attn_k.bias": (_KV_HEADS * _HEAD_DIM,),
                    prefix + "attn_v.bias": (_KV_HEADS * _HEAD_DIM,),
                }
            )
        if layer == 0:
            shapes.update(
                {
                    prefix + "ffn_gate.weight": (_DENSE, _H),
                    prefix + "ffn_up.weight": (_DENSE, _H),
                    prefix + "ffn_down.weight": (_H, _DENSE),
                }
            )
        else:
            shapes.update(
                {
                    prefix + "ffn_gate_inp.weight": (_EXPERTS, _H),
                    prefix + "exp_probs_b": (_EXPERTS,),
                    prefix + "ffn_down_exps.weight": (
                        _EXPERTS,
                        _H,
                        _EXPERT_WIDTH,
                    ),
                    prefix + "ffn_gate_shexp.weight": (_EXPERT_WIDTH, _H),
                    prefix + "ffn_up_shexp.weight": (_EXPERT_WIDTH, _H),
                    prefix + "ffn_down_shexp.weight": (_H, _EXPERT_WIDTH),
                }
            )
            if fused_experts:
                shapes[prefix + "ffn_gate_up_exps.weight"] = (
                    _EXPERTS,
                    2 * _EXPERT_WIDTH,
                    _H,
                )
            else:
                shapes[prefix + "ffn_gate_exps.weight"] = (
                    _EXPERTS,
                    _EXPERT_WIDTH,
                    _H,
                )
                shapes[prefix + "ffn_up_exps.weight"] = (
                    _EXPERTS,
                    _EXPERT_WIDTH,
                    _H,
                )
    mtp = "blk.2.nextn."
    shapes.update(
        {
            mtp + "eh_proj.weight": (_H, 2 * _H),
            mtp + "enorm.weight": (_H,),
            mtp + "hnorm.weight": (_H,),
        }
    )
    return shapes


def _write_hy_v3(
    path: Path,
    *,
    include_mtp: bool = True,
    fused_experts: bool = True,
    qkv_bias: bool = False,
) -> None:
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "hy_v3")
    for key, value in _metadata().items():
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_float32(key, value)
    rng = np.random.default_rng(0)
    for name, shape in _tensor_shapes(fused_experts=fused_experts, qkv_bias=qkv_bias).items():
        if not include_mtp and name.startswith("blk.2."):
            continue
        writer.add_tensor(name, rng.normal(0.0, 0.02, shape).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@dataclass
class _FakeGGUF:
    metadata: dict[str, object]
    tensors: dict[str, tuple[object, tuple[int, ...]]]
    architecture: str = "hy_v3"

    @property
    def tensor_names(self):
        return list(self.tensors)

    def tensor_items_raw(self):
        for name, (qtype, shape) in self.tensors.items():
            yield name, np.empty(0, dtype=np.uint8), qtype, shape


def _fake() -> _FakeGGUF:
    from gguf import GGMLQuantizationType

    return _FakeGGUF(
        _metadata(),
        {name: (GGMLQuantizationType.F32, shape) for name, shape in _tensor_shapes().items()},
    )


def test_hy_v3_exact_contract_accepts_complete_combined_file() -> None:
    validate_hy_v3_tensor_contract(_fake())


def test_hy_v3_exact_contract_accepts_target_only_split_file() -> None:
    model = _fake()
    model.tensors = {
        name: value for name, value in model.tensors.items() if not name.startswith("blk.2.")
    }
    validate_hy_v3_tensor_contract(model)


def test_hy_v3_exact_contract_accepts_zero_routing_scale_sentinel() -> None:
    model = _fake()
    model.metadata["hy_v3.expert_weights_scale"] = 0.0
    validate_hy_v3_tensor_contract(model)


def test_hy_v3_exact_contract_rejects_unknown_tensor() -> None:
    model = _fake()
    model.tensors["blk.1.hyper_connection.weight"] = (
        next(iter(model.tensors.values()))[0],
        (_H,),
    )
    with pytest.raises(ValueError, match=r"unexpected=.*hyper_connection"):
        validate_hy_v3_tensor_contract(model)


def test_hy_v3_exact_contract_rejects_quantized_selection_bias() -> None:
    from gguf import GGMLQuantizationType

    model = _fake()
    model.tensors["blk.1.exp_probs_b"] = (
        GGMLQuantizationType.Q4_0,
        (_EXPERTS,),
    )
    with pytest.raises(ValueError, match="must use explicit float storage"):
        validate_hy_v3_tensor_contract(model)


def test_hy_v3_exact_contract_rejects_quantized_router() -> None:
    from gguf import GGMLQuantizationType

    model = _fake()
    model.tensors["blk.1.ffn_gate_inp.weight"] = (
        GGMLQuantizationType.Q4_0,
        (_EXPERTS, _H),
    )
    with pytest.raises(ValueError, match="must use explicit float storage"):
        validate_hy_v3_tensor_contract(model)


@pytest.mark.parametrize("qtype_name", ["I8", "I16", "Q8_1", "Q8_K"])
def test_hy_v3_exact_contract_rejects_non_weight_storage(qtype_name: str) -> None:
    from gguf import GGMLQuantizationType

    model = _fake()
    model.tensors["blk.1.attn_q.weight"] = (
        getattr(GGMLQuantizationType, qtype_name),
        (_HEADS * _HEAD_DIM, _H),
    )
    with pytest.raises(ValueError, match="must use float, F64, or quantized weight storage"):
        validate_hy_v3_tensor_contract(model)


def test_hy_v3_exact_contract_accepts_f64_weight_storage() -> None:
    from gguf import GGMLQuantizationType

    model = _fake()
    model.tensors["blk.1.attn_norm.weight"] = (GGMLQuantizationType.F64, (_H,))
    validate_hy_v3_tensor_contract(model)


def test_hy_v3_exact_contract_rejects_noncontiguous_dense_schedule() -> None:
    model = _fake()
    for suffix in (
        "ffn_gate_inp.weight",
        "exp_probs_b",
        "ffn_gate_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    ):
        del model.tensors[f"blk.1.{suffix}"]
    model.tensors.update(
        {
            "blk.1.ffn_gate.weight": (
                next(iter(model.tensors.values()))[0],
                (_DENSE, _H),
            ),
            "blk.1.ffn_up.weight": (
                next(iter(model.tensors.values()))[0],
                (_DENSE, _H),
            ),
            "blk.1.ffn_down.weight": (
                next(iter(model.tensors.values()))[0],
                (_H, _DENSE),
            ),
        }
    )
    with pytest.raises(ValueError, match="contiguous leading dense prefix"):
        validate_hy_v3_tensor_contract(model)


@pytest.mark.parametrize("fused_experts", [False, True])
def test_hy_v3_gguf_builds_trunk_and_independent_mtp(
    tmp_path: Path, fused_experts: bool
) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = tmp_path / "tiny-hy-v3.gguf"
    _write_hy_v3(path, fused_experts=fused_experts)
    package = build_from_gguf(path, keep_quantized=False)

    assert package.mtp_head is not None
    trunk = package["model"]
    mtp = package.mtp_head["model"]
    assert sum(node.op_type == "Attention" for node in trunk.graph) == 2
    assert sum(node.op_type == "Attention" for node in mtp.graph) == 1
    assert len([value for value in trunk.graph.inputs if "past_key_values" in value.name]) == 4
    assert len([value for value in mtp.graph.inputs if "past_key_values" in value.name]) == 2
    assert "model.layers.1.mlp.e_score_correction_bias" in trunk.graph.initializers
    assert "layers.0.mlp.e_score_correction_bias" in mtp.graph.initializers


def test_hy_v3_target_only_split_builds_without_sidecar(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = tmp_path / "tiny-hy-v3-target.gguf"
    _write_hy_v3(path, include_mtp=False, qkv_bias=True)
    package = build_from_gguf(path, keep_quantized=False, static_cache=True, max_seq_len=32)

    assert package.mtp_head is None
    assert sum(node.op_type == "Attention" for node in package["model"].graph) == 2
    assert "model.layers.1.self_attn.q_proj.bias" in package["model"].graph.initializers
