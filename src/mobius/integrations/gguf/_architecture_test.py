# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType

from mobius._configs import NemotronHConfig
from mobius.integrations.gguf._architecture import (
    GGUFArchitectureAdapter,
    GGUFMappingAudit,
    create_architecture_adapter,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config

_LAYER_TYPES = (
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
)
_SUFFIXES = {
    "mamba2": (
        "attn_norm.weight",
        "ssm_a",
        "ssm_conv1d.bias",
        "ssm_conv1d.weight",
        "ssm_d",
        "ssm_dt.bias",
        "ssm_in.weight",
        "ssm_norm.weight",
        "ssm_out.weight",
    ),
    "moe": (
        "attn_norm.weight",
        "exp_probs_b.bias",
        "ffn_down_exps.weight",
        "ffn_down_shexp.weight",
        "ffn_gate_inp.weight",
        "ffn_up_exps.weight",
        "ffn_up_shexp.weight",
    ),
    "full_attention": (
        "attn_k.weight",
        "attn_norm.weight",
        "attn_output.weight",
        "attn_q.weight",
        "attn_v.weight",
    ),
}
_MTP_SUFFIXES = (
    "attn_k.weight",
    "attn_norm.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_v.weight",
    "exp_probs_b.bias",
    "ffn_down_exps.weight",
    "ffn_down_shexp.weight",
    "ffn_gate_inp.weight",
    "ffn_up_exps.weight",
    "ffn_up_shexp.weight",
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.shared_head_norm.weight",
    "post_attention_norm.weight",
)
_Q8_SUFFIXES = {
    "ssm_in.weight",
    "ssm_out.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
}


def _source_shape(layer_type: str, suffix: str) -> tuple[int, ...]:
    h, q, kv = 64, 64, 32
    experts, moe_inner, shared_inner = 128, 32, 64
    d_inner, conv_dim, mamba_heads = 64, 128, 2
    shapes = {
        "attn_norm.weight": (h,),
        "ssm_a": (mamba_heads, 1),
        "ssm_conv1d.bias": (conv_dim,),
        "ssm_conv1d.weight": (conv_dim, 4),
        "ssm_d": (mamba_heads, 1),
        "ssm_dt.bias": (mamba_heads,),
        "ssm_in.weight": (d_inner + conv_dim + mamba_heads, h),
        "ssm_norm.weight": (2, 32),
        "ssm_out.weight": (h, d_inner),
        "exp_probs_b.bias": (experts,),
        "ffn_down_exps.weight": (experts, h, moe_inner),
        "ffn_down_shexp.weight": (h, shared_inner),
        "ffn_gate_inp.weight": (experts, h),
        "ffn_up_exps.weight": (experts, moe_inner, h),
        "ffn_up_shexp.weight": (shared_inner, h),
        "attn_k.weight": (kv, h),
        "attn_output.weight": (h, q),
        "attn_q.weight": (q, h),
        "attn_v.weight": (kv, h),
    }
    assert suffix in _SUFFIXES[layer_type]
    return shapes[suffix]


def _record(name: str, qtype, shape: tuple[int, ...]):
    return SimpleNamespace(
        name=name,
        tensor_type=qtype,
        # ReaderTensor shapes use GGML order.
        shape=np.asarray(tuple(reversed(shape)), dtype=np.uint64),
    )


def _synthetic_pinned_header():
    records = [
        _record("output.weight", GGMLQuantizationType.Q8_0, (256, 64)),
        _record("output_norm.weight", GGMLQuantizationType.F32, (64,)),
        _record("token_embd.weight", GGMLQuantizationType.Q8_0, (256, 64)),
    ]
    for index, layer_type in enumerate(_LAYER_TYPES):
        for suffix in _SUFFIXES[layer_type]:
            qtype = (
                GGMLQuantizationType.Q8_0
                if suffix in _Q8_SUFFIXES
                else GGMLQuantizationType.F32
            )
            records.append(
                _record(
                    f"blk.{index}.{suffix}",
                    qtype,
                    _source_shape(layer_type, suffix),
                )
            )

    mtp_shapes = {
        suffix: _source_shape(
            (
                "full_attention"
                if suffix.startswith("attn_") and suffix != "attn_norm.weight"
                else "moe"
            ),
            suffix,
        )
        for suffix in _MTP_SUFFIXES
        if suffix in _Q8_SUFFIXES
        or suffix in {"attn_norm.weight", "exp_probs_b.bias", "ffn_gate_inp.weight"}
    }
    mtp_shapes.update(
        {
            "nextn.eh_proj.weight": (64, 128),
            "nextn.enorm.weight": (64,),
            "nextn.hnorm.weight": (64,),
            "nextn.shared_head_norm.weight": (64,),
            "post_attention_norm.weight": (64,),
        }
    )
    for suffix in _MTP_SUFFIXES:
        if suffix == "ffn_gate_inp.weight":
            qtype = GGMLQuantizationType.BF16
        elif suffix in _Q8_SUFFIXES or suffix == "nextn.eh_proj.weight":
            qtype = GGMLQuantizationType.Q8_0
        else:
            qtype = GGMLQuantizationType.F32
        records.append(_record(f"blk.52.{suffix}", qtype, mtp_shapes[suffix]))

    metadata = {
        "nemotron_h_moe.attention.head_count": 2,
        "nemotron_h_moe.attention.head_count_kv": [
            1 if layer_type == "full_attention" else 0 for layer_type in _LAYER_TYPES
        ]
        + [1],
        "nemotron_h_moe.attention.key_length": 32,
        "nemotron_h_moe.attention.layer_norm_rms_epsilon": 1e-5,
        "nemotron_h_moe.block_count": 53,
        "nemotron_h_moe.context_length": 128,
        "nemotron_h_moe.embedding_length": 64,
        "nemotron_h_moe.expert_count": 128,
        "nemotron_h_moe.expert_feed_forward_length": 32,
        "nemotron_h_moe.expert_group_count": 1,
        "nemotron_h_moe.expert_group_used_count": 1,
        "nemotron_h_moe.expert_shared_count": 1,
        "nemotron_h_moe.expert_shared_feed_forward_length": 64,
        "nemotron_h_moe.expert_used_count": 1,
        "nemotron_h_moe.expert_weights_norm": True,
        "nemotron_h_moe.expert_weights_scale": 2.5,
        "nemotron_h_moe.nextn_predict_layers": 1,
        "nemotron_h_moe.ssm.conv_kernel": 4,
        "nemotron_h_moe.ssm.group_count": 2,
        "nemotron_h_moe.ssm.inner_size": 64,
        "nemotron_h_moe.ssm.state_size": 16,
        "nemotron_h_moe.ssm.time_step_rank": 2,
        "nemotron_h_moe.vocab_size": 256,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "pixtral",
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 11,
        "tokenizer.ggml.padding_token_id": 999,
    }
    return SimpleNamespace(
        architecture="nemotron_h_moe",
        metadata=metadata,
        tensor_names=[record.name for record in records],
        _reader=SimpleNamespace(tensors=records),
        _path=Path("synthetic-nemotron-q8.gguf"),
    )


def test_architecture_adapter_requires_source_validation() -> None:
    adapter = GGUFArchitectureAdapter(SimpleNamespace())

    with pytest.raises(NotImplementedError):
        adapter.validate_model(source="synthetic")


def test_nemotron_adapter_derives_exact_backbone_and_mapping() -> None:
    model = _synthetic_pinned_header()
    adapter = create_architecture_adapter(model.architecture, model)
    assert adapter is not None
    adapter.validate_model(source="synthetic")

    config = gguf_to_config(model, adapter=adapter)
    assert isinstance(config, NemotronHConfig)
    assert config.model_type == "nemotron_h"
    assert config.layer_types == list(_LAYER_TYPES)
    assert config.layer_types.count("mamba2") == 23
    assert config.layer_types.count("moe") == 23
    assert config.layer_types.count("full_attention") == 6
    assert config.num_hidden_layers == 52
    assert config.bos_token_id == 1
    assert config.eos_token_id == 2
    assert config.pad_token_id == 0
    assert config.num_key_value_heads == 1
    assert config.mamba_n_heads == 2
    assert config.mamba_d_head == 32

    audit = GGUFMappingAudit()
    for record in model._reader.tensors:
        shape = tuple(int(dim) for dim in reversed(record.shape))
        audit.record(record.name, adapter.map_tensor(record.name, shape))
    adapter.validate_mapping_audit(audit)

    assert len(audit.mapped_sources) == 401
    assert len(audit.excluded_sources) == 16
    assert set(audit.excluded_sources) == {f"blk.52.{suffix}" for suffix in _MTP_SUFFIXES}
    assert len(audit.target_sources) == 6243


def test_nemotron_adapter_expands_experts_and_transforms_mamba_values() -> None:
    model = _synthetic_pinned_header()
    adapter = create_architecture_adapter(model.architecture, model)
    assert adapter is not None

    mapping = adapter.map_tensor("blk.1.ffn_up_exps.weight", (128, 32, 64))
    assert mapping is not None
    assert [target.source_index for target in mapping.targets] == list(range(128))
    assert mapping.targets[0].state_dict_name == (
        "backbone.layers.1.mixer.experts.0.up_proj.weight"
    )
    assert mapping.targets[1].initializer_name == (
        "model.layers.1.moe.experts.1.up_proj.weight"
    )

    target = adapter.map_tensor("blk.0.ssm_a", (2, 1)).targets[0]
    transformed = adapter.transform_tensor(
        "blk.0.ssm_a",
        target,
        torch.tensor([[-1.0], [-np.e]], dtype=torch.float32),
    )
    torch.testing.assert_close(transformed, torch.tensor([0.0, 1.0]))

    conv_target = adapter.map_tensor("blk.0.ssm_conv1d.weight", (128, 4)).targets[0]
    conv = adapter.transform_tensor(
        "blk.0.ssm_conv1d.weight",
        conv_target,
        torch.zeros(128, 4),
    )
    assert conv.shape == (128, 1, 4)

    with pytest.raises(ValueError, match="expected"):
        adapter.map_tensor("blk.0.ssm_out.weight", (63, 64))


def test_nemotron_adapter_rejects_unpreserved_base_qtype_with_evidence() -> None:
    model = _synthetic_pinned_header()
    record = next(
        record for record in model._reader.tensors if record.name == "blk.1.ffn_up_exps.weight"
    )
    record.tensor_type = GGMLQuantizationType.Q5_1
    adapter = create_architecture_adapter(model.architecture, model)
    assert adapter is not None

    with pytest.raises(NotImplementedError, match=r"Q5_1=262,144 parameters"):
        adapter.validate_model(source="synthetic")


def test_nemotron_adapter_rejects_qtype_location_swap() -> None:
    model = _synthetic_pinned_header()
    q8_record = next(
        record for record in model._reader.tensors if record.name == "blk.1.ffn_up_exps.weight"
    )
    float_record = next(
        record
        for record in model._reader.tensors
        if record.name == "blk.1.ffn_gate_inp.weight"
    )
    q8_record.tensor_type, float_record.tensor_type = (
        float_record.tensor_type,
        q8_record.tensor_type,
    )
    adapter = create_architecture_adapter(model.architecture, model)
    assert adapter is not None

    with pytest.raises(ValueError, match="exact Q8 preservation"):
        adapter.validate_model(source="synthetic")


@pytest.mark.integration
def test_pinned_nemotron_q8_header_and_mapping_from_real_artifact() -> None:
    path_value = os.environ.get("MOBIUS_NEMOTRON_Q8_GGUF")
    if not path_value:
        pytest.skip("Set MOBIUS_NEMOTRON_Q8_GGUF to the pinned Q8_0 artifact")
    path = Path(path_value)
    assert path.stat().st_size == 35_004_643_392

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    assert digest.hexdigest() == (
        "dc5276dd0619c04e277504d2358a793e31ccbe39e894d767d0d14f2a221e2ca4"
    )

    from mobius.integrations.gguf._reader import GGUFModel

    model = GGUFModel(path)
    adapter = create_architecture_adapter(model.architecture, model)
    assert adapter is not None
    adapter.validate_model(source=str(path))
    audit = GGUFMappingAudit()
    for record in model._reader.tensors:
        source_shape = tuple(int(dim) for dim in reversed(record.shape))
        audit.record(record.name, adapter.map_tensor(record.name, source_shape))
    adapter.validate_mapping_audit(audit)
