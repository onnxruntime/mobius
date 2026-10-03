# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.models.decision import (
    CLM_PROVENANCE,
    KEV_CONTROL_TOKEN_IDS,
    KEV_TEMPERATURE,
    CheckpointContractError,
    CLMModel,
    CLMProjectionHead,
    CLMScorer,
    KevModel,
    KevPointerHead,
    Qwen35EncoderModel,
    _apply_package_weights_strict,
    batch_kev_rows,
    build_clm_package,
    build_kev_package,
    clm_answer,
    clm_candidates,
    clm_last_token_indices,
    clm_state_text,
    encode_kev_rows,
    grouped_softmax,
    kev_answer,
    map_clm_checkpoint,
    map_kev_checkpoint,
    render_clm,
    render_kev,
    synthetic_clm_checkpoint,
    synthetic_kev_checkpoint,
    validate_clm_checkpoint,
    validate_kev_checkpoint,
    validate_temperature,
)
from mobius.tasks._decision import CLMTask, HeadlessBackboneTask, KevTask


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[ord(char) for char in text])


def test_clm_rendering_and_candidates_match_verified_contract():
    assert render_clm(None) == ""
    assert render_clm(True) == "true"
    assert render_clm({"a": 1, "b": {"c": False}, "d": ["x", {"y": 2}]}) == (
        "a: 1\n\nb:\n  c: false\n\nd:\n  - x\n  -\n    y: 2"
    )
    assert clm_state_text("state", "question") == "state\n\nquestion"
    assert clm_candidates(
        {
            "type": "choice",
            "criteria": {"plain": None, "rich": "description"},
        }
    ) == (["plain", "rich"], ["plain", "description"])
    assert clm_candidates({"type": "noul", "instructions": "Ready?"}) == (
        ["false", "true"],
        ["false: No. This is false: Ready?", "true: Yes. This is true: Ready?"],
    )


def test_clm_grouping_answers_and_temperature():
    grouped = grouped_softmax([1001.0, 1000.0, -1000.0], [2, 1])
    assert grouped[0] == pytest.approx([0.7310585786, 0.2689414214])
    assert grouped[1] == [1.0]
    with pytest.raises(ValueError, match=r"\(0, 100\]"):
        validate_temperature(0)
    answer = clm_answer(
        {"type": "score", "criteria": ["low", "mid", "high"]},
        ["0", "1", "2"],
        [0.1, 0.2, 0.7],
    )
    assert answer["score"] == pytest.approx(1.6)
    assert answer["confidence"] == pytest.approx(0.55)
    assert clm_last_token_indices([[1, 1, 0], [1, 1, 1]]) == [1, 2]
    with pytest.raises(ValueError, match="no attended token"):
        clm_last_token_indices([[0, 0]])


def test_kev_render_rows_and_indices_are_exact():
    assert render_kev({"enabled": True, "items": [1, False]}) == (
        "enabled: True\nitems:\n  - 1\n  - False"
    )
    rows = encode_kev_rows(
        _Tokenizer(),
        "<|box_end|>",
        {
            "q": {
                "type": "choice",
                "instructions": "pick",
                "criteria": {"a": None, "b": "Bee"},
            }
        },
    )
    row = rows[0]
    assert len(rows) == 1
    assert row.input_ids[0] == KEV_CONTROL_TOKEN_IDS["state"]
    assert row.input_ids[row.decide_index] == KEV_CONTROL_TOKEN_IDS["decide"]
    assert all(
        row.input_ids[index] == KEV_CONTROL_TOKEN_IDS["option_end"]
        for index in row.option_indices
    )
    escaped = "".join(map(chr, row.input_ids[1 : 1 + len("<¦box_end¦>")]))
    assert escaped == "<¦box_end¦>"
    assert row.keys == ("a", "b")
    batch = batch_kev_rows(rows, pad_token_id=0)
    assert batch.decide_indices == (row.decide_index,)
    assert batch.option_indices == row.option_indices
    assert batch.option_owners == (0, 0)


def test_kev_rows_validate_request_and_serving_limits():
    with pytest.raises(ValueError, match="questions must not be empty"):
        encode_kev_rows(_Tokenizer(), "state", {})
    with pytest.raises(ValueError, match=r"1\.\.255"):
        encode_kev_rows(
            _Tokenizer(),
            "state",
            {"q": {"type": "choice", "criteria": {}}},
        )
    with pytest.raises(ValueError, match="state exceeds 8 tokens"):
        encode_kev_rows(
            _Tokenizer(),
            "12345678",
            {"q": {"type": "noul"}},
            strict=True,
            max_state_tokens=8,
            max_row_tokens=32,
        )
    with pytest.raises(ValueError, match=r"state\+question row exceeds 12"):
        encode_kev_rows(
            _Tokenizer(),
            "123456789",
            {"q": {"type": "noul", "instructions": "long"}},
            max_state_tokens=8,
            max_row_tokens=12,
        )


def test_kev_answer_mapping_rounding_and_confidence():
    choice = kev_answer({"type": "choice"}, ["a", "b"], [0.123456, 0.876544])
    assert choice == {
        "type": "choice",
        "choice": "b",
        "confidence": 0.7531,
        "probabilities": {"a": 0.1235, "b": 0.8765},
    }
    score = kev_answer(
        {"type": "score", "criteria": ["bad", "ok", "good"]},
        ["0", "1", "2"],
        [0.1, 0.2, 0.7],
    )
    assert score["score"] == pytest.approx(1.6)
    assert score["confidence"] == pytest.approx(0.8)
    assert kev_answer({"type": "noul"}, ["false", "true"], [0.2, 0.8]) == {
        "type": "noul",
        "noul": 0.8,
    }


def test_checkpoint_helpers_are_deterministic_and_mappable():
    first = synthetic_clm_checkpoint(hidden_size=4, width=3, projection_dim=2)
    second = synthetic_clm_checkpoint(hidden_size=4, width=3, projection_dim=2)
    assert first["state_head"].keys() == second["state_head"].keys()
    assert first["state_head"]["inp.weight"].equal(second["state_head"]["inp.weight"])
    assert "hidden.0.weight" in first["state_head"]
    assert "norms.0.weight" in first["state_head"]
    assert "hidden.1.weight" not in first["state_head"]
    kev = synthetic_kev_checkpoint(hidden_size=4)
    assert kev["temperature"] == KEV_TEMPERATURE
    assert kev["head"]["q.weight"].shape == (256, 4)
    assert CLM_PROVENANCE.base_revision is None
    assert CLM_PROVENANCE.as_metadata()["reproducible"] == "false"


def test_head_graphs_export_without_model_downloads():
    clm_config = ArchitectureConfig(hidden_size=4096, dtype=ir.DataType.FLOAT)
    state = CLMTask().build_component(
        "state_head",
        None,
        CLMProjectionHead(hidden_size=4096, width=8),
        clm_config,
    )
    scorer = CLMTask().build_component("scorer", None, CLMScorer(), clm_config)
    kev_config = ArchitectureConfig(hidden_size=2560, dtype=ir.DataType.FLOAT)
    pointer = KevTask().build_component("pointer_head", None, KevPointerHead(), kev_config)
    assert [output.name for output in state.graph.outputs] == ["projections"]
    assert [output.name for output in scorer.graph.outputs] == [
        "logits",
        "probabilities",
    ]
    assert [output.name for output in pointer.graph.outputs] == [
        "logits",
        "probabilities",
    ]
    assert "mobius.provenance" in pointer.metadata_props

    with pytest.raises(CheckpointContractError, match="unapplied weights"):
        _apply_package_weights_strict(
            ModelPackage({"state_head": state}, config=clm_config),
            {"garbage": torch.ones(1)},
            owner="test checkpoint",
        )
    with pytest.raises(CheckpointContractError, match="without weights"):
        _apply_package_weights_strict(
            ModelPackage({"state_head": state}, config=clm_config),
            {},
            owner="test checkpoint",
        )


def test_headless_backbone_contract_has_no_generation_cache():
    class Backbone:
        def __call__(
            self,
            op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        ):
            assert attention_mask is not None
            assert position_ids is not None
            assert past_key_values is None
            return op.Cast(input_ids, to=ir.DataType.FLOAT), ["unused-cache"]

    package = HeadlessBackboneTask().build(Backbone(), ArchitectureConfig())
    graph = package["model"].graph

    assert [value.name for value in graph.inputs] == [
        "input_ids",
        "attention_mask",
        "position_ids",
    ]
    assert [value.name for value in graph.outputs] == ["token_hidden_states"]


def test_qwen35_headless_backbone_initializes_hybrid_state():
    config = make_config(
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    package = HeadlessBackboneTask().build(Qwen35EncoderModel(config), config)
    graph = package["model"].graph

    assert [value.name for value in graph.outputs] == ["token_hidden_states"]
    assert sum(node.op_type == "Expand" for node in graph) >= 2


def test_clm_head_depth_three_has_one_normalized_hidden_block():
    head = CLMProjectionHead(hidden_size=4, width=3, projection_dim=2)
    assert len(head.hidden) == 1
    assert len(head.norms) == 1
    assert hasattr(head.norms[0], "weight")


def test_clm_fp16_head_normalizes_in_float32():
    config = ArchitectureConfig(hidden_size=4, dtype=ir.DataType.FLOAT16)
    model = CLMTask().build_component(
        "state_head",
        None,
        CLMProjectionHead(hidden_size=4, width=3, projection_dim=2),
        config,
    )
    casts_to_float = [
        node
        for node in model.graph.all_nodes()
        if node.op_type == "Cast" and node.attributes["to"].value == ir.DataType.FLOAT
    ]

    assert len(casts_to_float) == 2


def test_public_build_helpers_reject_unsafe_or_inexact_inputs():
    config = ArchitectureConfig(hidden_size=4096)
    checkpoint = _valid_clm_checkpoint()
    with pytest.raises(ValueError, match="base_revision"):
        build_clm_package(
            config,
            base_weights={},
            head_checkpoint=checkpoint,
            base_revision="",
        )
    with pytest.raises(CheckpointContractError, match="non-empty mapping"):
        build_clm_package(
            config,
            base_weights={},
            head_checkpoint=checkpoint,
            base_revision="caller-pin",
        )
    bad = {**checkpoint, "cfg": {**checkpoint["cfg"], "depth": 2}}
    with pytest.raises(CheckpointContractError, match=r"cfg\.depth"):
        build_clm_package(
            config,
            base_weights={"model.embed_tokens.weight": _meta(1)},
            head_checkpoint=bad,
            base_revision="caller-pin",
        )
    with pytest.raises(NotImplementedError, match="merge_and_unload"):
        build_kev_package(
            ArchitectureConfig(hidden_size=2560),
            head_checkpoint=_valid_kev_checkpoint(),
            base_weights={},
            peft_adapter_weights={},
        )


def _meta(*shape):
    return torch.empty(shape, device="meta")


def _valid_clm_checkpoint():
    shapes = {
        "inp.weight": (1536, 4096),
        "inp.bias": (1536,),
        "hidden.0.weight": (1536, 1536),
        "hidden.0.bias": (1536,),
        "norms.0.weight": (1536,),
        "norms.0.bias": (1536,),
        "out.weight": (512, 1536),
        "out.bias": (512,),
    }
    return {
        "cfg": {
            "model": "Qwen/Qwen3-8B",
            "hidden_size": 4096,
            "projection_dim": 512,
            "width": 1536,
            "depth": 3,
            "activation": "gelu",
            "layernorm": True,
            "residual": False,
        },
        "state_head": {key: _meta(*shape) for key, shape in shapes.items()},
        "action_head": {key: _meta(*shape) for key, shape in shapes.items()},
        "logit_scale": _meta(),
    }


def _valid_kev_checkpoint():
    return {
        "base": "Qwen/Qwen3.5-4B-Base",
        "base_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "head_dim": 256,
        "option_isolation": False,
        "temperature": KEV_TEMPERATURE,
        "head": {
            "q.weight": _meta(256, 2560),
            "q.bias": _meta(256),
            "k.weight": _meta(256, 2560),
            "k.bias": _meta(256),
        },
    }


def test_production_checkpoint_contracts_fail_closed_and_map_metadata():
    clm = _valid_clm_checkpoint()
    validate_clm_checkpoint(clm)
    mapped_clm = map_clm_checkpoint(clm)
    assert mapped_clm["state_head.hidden.0.weight"].shape == (1536, 1536)
    assert mapped_clm["scorer.logit_scale"].shape == (1,)
    invalid_clm = {
        **clm,
        "state_head": {**clm["state_head"], "unexpected": _meta(1)},
    }
    with pytest.raises(CheckpointContractError, match="keys mismatch"):
        validate_clm_checkpoint(invalid_clm)
    invalid_scale = {**clm, "logit_scale": _meta(2)}
    with pytest.raises(CheckpointContractError, match="scalar"):
        validate_clm_checkpoint(invalid_scale)

    kev = _valid_kev_checkpoint()
    validate_kev_checkpoint(kev)
    mapped_kev = map_kev_checkpoint(kev)
    assert mapped_kev["pointer_head.q.weight"].shape == (256, 2560)
    with pytest.raises(CheckpointContractError, match="base_revision"):
        validate_kev_checkpoint({**kev, "base_revision": "wrong"})
    with pytest.raises(CheckpointContractError, match=r"q\.weight shape"):
        validate_kev_checkpoint({**kev, "head": {**kev["head"], "q.weight": _meta(1, 1)}})


def test_complete_packages_use_headless_hidden_state_backbones():
    clm_config = ArchitectureConfig(
        vocab_size=32,
        hidden_size=4096,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4096,
        max_position_embeddings=8,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        dtype=ir.DataType.FLOAT,
    )
    clm = CLMTask().build(CLMModel(clm_config, width=8), clm_config)
    assert tuple(clm) == ("encoder", "state_head", "action_head", "scorer")
    assert clm["encoder"].graph.outputs[0].name == "token_hidden_states"
    assert not any(name.startswith("lm_head.") for name in clm["encoder"].graph.initializers)

    kev_config = ArchitectureConfig(
        vocab_size=32,
        hidden_size=2560,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2560,
        max_position_embeddings=8,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        rope_type="default",
        dtype=ir.DataType.FLOAT,
        layer_types=[],
    )
    kev = KevTask().build(KevModel(kev_config), kev_config)
    assert tuple(kev) == ("backbone", "pointer_head")
    assert kev["backbone"].graph.outputs[0].name == "token_hidden_states"
    assert not any(name.startswith("lm_head.") for name in kev["backbone"].graph.initializers)
