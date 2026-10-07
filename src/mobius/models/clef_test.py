# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius._configs import VisionConfig
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.clef import encode_clef_record
from mobius.models.clef import (
    CLEF_FLASH_MODEL_ID,
    CLEF_FLASH_REVISION,
    ClefConfig,
    ClefFlashModel,
    ClefJointHead,
    _span_mean,
    clef_head_source_name,
)
from mobius.tasks._base import ModelTask, _make_graph, _make_model


def tiny_config(**overrides):
    base = make_config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=256,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        partial_rotary_factor=0.5,
        mrope_section=[1, 1, 0],
        mrope_interleaved=True,
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
        temporal_patch_size=2,
        spatial_merge_size=2,
        vision=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=1,
            patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
        ),
    )
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    fields.update(
        model_type="clef_flash",
        head_width=16,
        head_heads=4,
        head_routing_layers=2,
        head_layers=2,
        head_feedforward=32,
    )
    fields.update(overrides)
    return ClefConfig(**fields)


class _HeadTask(ModelTask):
    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(self, module, config):
        graph, builder = _make_graph()
        specs = [
            ("hidden_states", config.dtype, [1, "sequence", config.hidden_size]),
            ("input_ids", ir.DataType.INT64, [1, "sequence"]),
            ("question_spans", ir.DataType.INT64, ["questions", 2]),
            ("option_spans", ir.DataType.INT64, ["options", 2]),
            ("option_question_ids", ir.DataType.INT64, ["options"]),
            ("question_types", ir.DataType.INT64, ["questions"]),
        ]
        inputs = [
            builder.input(name, dtype=dtype, shape=shape) for name, dtype, shape in specs
        ]
        logits, probabilities = module(builder.op, *inputs)
        builder.add_output(logits, "logits")
        builder.add_output(probabilities, "probabilities")
        return ModelPackage({"model": _make_model(graph)}, config=config)


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_full_package_builds(dtype):
    config = tiny_config(dtype=dtype)
    module = ClefFlashModel(config)
    package = build_from_module(module, config, "clef-decision")
    assert set(package) == {"decoder", "vision_encoder", "embedding", "decision_head"}
    assert [v.name for v in package["decoder"].graph.outputs] == ["hidden_states"]
    assert not any("past" in v.name for v in package["decoder"].graph.inputs)
    assert "lm_head.weight" not in package["decoder"].graph.initializers
    assert [v.name for v in package["decision_head"].graph.outputs] == [
        "logits",
        "probabilities",
    ]
    assert len(package["decoder"].functions) > 0


def test_preprocessing_splits_qkv_and_uses_output_embeddings():
    config = tiny_config()
    module = ClefFlashModel(config)
    width = config.head_width
    tensor = torch.arange(3 * width * width).reshape(3 * width, width).float()
    output_table = torch.randn(config.vocab_size, config.hidden_size)
    result = module.preprocess_weights(
        {
            "lm_head.weight": output_table,
            "model.language_model.embed_tokens.weight": torch.zeros_like(output_table),
            "head.evidence_layers.0.attention.in_proj_weight": tensor,
        }
    )
    assert result["decision_head.output_embedding.weight"] is output_table
    for index, projection in enumerate(("q_proj", "k_proj", "v_proj")):
        torch.testing.assert_close(
            result[f"decision_head.evidence_layers.0.attention.{projection}.weight"],
            tensor.chunk(3)[index],
        )


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_late_schema_spans_are_pooled_in_float32(dtype):
    from onnxscript import nn

    class PoolTask(ModelTask):
        model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

        def build(self, module, config):
            graph, builder = _make_graph()
            values = builder.input("values", dtype=config.dtype, shape=["sequence", 2])
            spans = builder.input("spans", dtype=ir.DataType.INT64, shape=["spans", 2])
            builder.add_output(_span_mean(builder.op, values, spans), "means")
            return ModelPackage({"model": _make_model(graph)}, config=config)

    package = build_from_module(nn.Module(), tiny_config(dtype=dtype), PoolTask())
    cumsum = next(node for node in package["model"].graph if node.op_type == "CumSum")
    # Check the pre-runtime graph too: CPU ORT may promote FP16 automatically.
    assert cumsum.inputs[0].dtype == ir.DataType.FLOAT
    if dtype == ir.DataType.FLOAT16:
        actual = OnnxModelSession(package["model"]).run(
            {
                "values": np.ones((16384, 2), dtype=np.float16),
                "spans": np.array([[16000, 16003], [16380, 16384]], dtype=np.int64),
            }
        )["means"]
        np.testing.assert_array_equal(actual, np.ones((2, 2), dtype=np.float16))


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        return SimpleNamespace(input_ids=[ord(c) % 256 for c in text])


def _record():
    return {
        "state": {"message": "An overdue invoice", "amount": 200},
        "questions": {
            "urgent": {"type": "noul"},
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"sales": "Sales", "billing": "Billing", "support": None},
            },
            "priority": {"type": "score", "criteria": ["low", "high"]},
        },
    }


def test_encoder_preserves_ragged_option_labels_and_truncation():
    record = encode_clef_record(_Tokenizer(), _record(), max_state_tokens=3)
    assert record.option_ids == (
        ("true", "false"),
        ("billing", "sales", "support"),
        ("0", "1"),
    )
    assert record.option_question_ids.tolist() == [0, 0, 1, 1, 1, 2, 2]
    assert record.question_types.tolist() == [0, 1, 2]
    assert np.all(record.option_spans[:, 1] > record.option_spans[:, 0])
    probs = np.array([0.3, 0.7, 0.2, 0.3, 0.5, 0.4, 0.6])
    assert record.probabilities_by_question(probs)["team"]["support"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="exactly"):
        record.probabilities_by_question(probs[:-1])
    with pytest.raises(ValueError, match="sum to one"):
        record.probabilities_by_question(probs * 0.5)
    with pytest.raises(ValueError, match="hidden states"):
        record.decision_feeds(np.zeros((2, record.input_ids.shape[1], 32)))
    with pytest.raises(ValueError, match="schema/media"):
        encode_clef_record(_Tokenizer(), _record(), max_length=5)


@pytest.mark.parametrize(
    "record, message",
    [
        ({"questions": {"a": {"type": "noul"}}}, "requires a state"),
        ({"state": "", "questions": {}}, "nonempty questions"),
        (
            {"state": "", "questions": {1: {"type": "noul"}, "1": {"type": "noul"}}},
            "question labels",
        ),
        (
            {
                "state": "",
                "questions": {"a": {"type": "choice", "criteria": {1: "", "1": ""}}},
            },
            "option labels",
        ),
        (
            {"state": "", "questions": {"a": {"type": "noul", "criteria": []}}},
            "boolean criteria",
        ),
    ],
)
def test_encoder_rejects_invalid_records(record, message):
    with pytest.raises(ValueError, match=message):
        encode_clef_record(_Tokenizer(), record)


def test_encoder_preserves_processor_media_and_full_prompt_token_types():
    def processor(**kwargs):
        assert kwargs["images"] == ["image"]
        assert kwargs["videos"] == ["video"]
        return {
            "input_ids": np.array([[252, 250, 253, 252, 251, 253]]),
            "attention_mask": np.ones((1, 6), dtype=np.int64),
            "mm_token_type_ids": np.array([[0, 1, 0, 0, 2, 0]]),
            "pixel_values": np.ones((4, 24), dtype=np.float32),
            "pixel_values_videos": np.ones((4, 24), dtype=np.float32),
            "image_grid_thw": np.array([[1, 2, 2]]),
            "video_grid_thw": np.array([[1, 2, 2]]),
        }

    source = {**_record(), "images": ["image"], "videos": ["video"]}
    record = encode_clef_record(_Tokenizer(), source, processor=processor)
    assert record.media["mm_token_type_ids"].shape == record.input_ids.shape
    assert np.count_nonzero(record.media["mm_token_type_ids"]) == 2
    assert np.all(record.media["mm_token_type_ids"][0, record.question_spans[:, 0]] == 0)
    assert record.media["pixel_values"].shape == (4, 24)
    with pytest.raises(ValueError, match="require a processor"):
        encode_clef_record(_Tokenizer(), source)


@pytest.mark.parametrize("revision", [None, "explicit-release"])
def test_checkpoint_revision_is_forwarded_to_every_asset(monkeypatch, tmp_path, revision):
    from mobius.integrations.transformers import _builder, _clef

    config = tiny_config()
    head = {
        "hidden_size": 32,
        "width": 16,
        "routing_layers": 2,
        "layers": 2,
        "heads": 4,
        "feedforward": 32,
    }
    path = tmp_path / "joint_head_config.json"
    path.write_text(json.dumps(head))
    expected = CLEF_FLASH_REVISION if revision is None else revision
    calls = []

    def load_config(model_id, **kwargs):
        assert model_id == CLEF_FLASH_MODEL_ID
        assert kwargs == {"revision": expected, "trust_remote_code": False}
        return SimpleNamespace(model_type="qwen3_5", text_config=SimpleNamespace()), None

    def asset(model_id, filename, asset_revision):
        assert model_id == CLEF_FLASH_MODEL_ID
        assert asset_revision == expected
        calls.append(filename)
        return str(path) if filename.endswith(".json") else "head.safetensors"

    def shards(model_id, shard_revision):
        assert model_id == CLEF_FLASH_MODEL_ID and shard_revision == expected
        return ["backbone.safetensors"]

    def stream(model, model_id, planner, **kwargs):
        assert model_id == CLEF_FLASH_MODEL_ID
        assert kwargs["revision"] == expected
        assert kwargs["_resolved_paths"] == ["backbone.safetensors", "head.safetensors"]
        calls.append("stream")
        return {"assigned_tensors": 1}

    monkeypatch.setattr(_builder, "_load_transformers_config", load_config)
    monkeypatch.setattr(
        ClefConfig, "from_transformers", classmethod(lambda cls, *a, **k: config)
    )
    monkeypatch.setattr(_clef, "_asset", asset)
    monkeypatch.setattr(_clef, "_resolve_shard_paths", shards)
    monkeypatch.setattr(_clef, "stream_preprocessed_safetensors_to_model", stream)
    package = _clef.build_clef_model(
        CLEF_FLASH_MODEL_ID,
        revision=revision,
        dtype="f32",
        execution_provider="default",
        load_weights=True,
    )
    assert calls == ["joint_head_config.json", "joint_head.safetensors"] + ["stream"] * 4
    assert all(
        model.metadata_props["mobius.source_revision"] == expected
        for model in package.values()
    )


@pytest.mark.parametrize("head", [[], {}, {"hidden_size": 99}])
def test_checkpoint_rejects_malformed_head_config(monkeypatch, tmp_path, head):
    from mobius.integrations.transformers import _builder, _clef

    path = tmp_path / "joint_head_config.json"
    path.write_text(json.dumps(head))
    parent = SimpleNamespace(model_type="qwen3_5", text_config=SimpleNamespace())
    monkeypatch.setattr(_builder, "_load_transformers_config", lambda *a, **k: (parent, None))
    monkeypatch.setattr(
        ClefConfig, "from_transformers", classmethod(lambda cls, *a, **k: tiny_config())
    )
    monkeypatch.setattr(_clef, "_asset", lambda *a: str(path))
    with pytest.raises(ValueError, match="joint-head config"):
        _clef.build_clef_model(
            str(tmp_path),
            revision=None,
            dtype="f32",
            execution_provider="default",
            load_weights=False,
        )


def test_streaming_plan_is_fail_closed_and_supports_tied_embeddings():
    from mobius.integrations.transformers._clef import _plan

    name = "decision_head.output_embedding.weight"
    initializer = ir.Value(
        name=name, shape=ir.Shape([256, 32]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    params = {
        "initializers": {name: initializer},
        "component": "decision_head",
        "recognized": {"lm_head.weight", "model.language_model.embed_tokens.weight"},
        "tied_embeddings": False,
    }
    with pytest.raises(ValueError, match="missing"):
        _plan({}, **params)
    with pytest.raises(ValueError, match="floating-point"):
        _plan({"lm_head.weight": ("head", [256, 32], "I8")}, **params)
    with pytest.raises(ValueError, match="Unrecognized"):
        _plan(
            {
                "lm_head.weight": ("head", [256, 32], "F32"),
                "unrecognized.weight": ("head", [1], "F32"),
            },
            **params,
        )
    plan = _plan(
        {"model.language_model.embed_tokens.weight": ("backbone", [256, 32], "F32")},
        **{**params, "tied_embeddings": True},
    )
    assert plan.targets[name].source_name == "model.language_model.embed_tokens.weight"
    with pytest.raises(FileNotFoundError, match="missing"):
        from mobius.integrations.transformers._clef import _asset

        _asset(str(pathlib.Path(__file__).parent), "does_not_exist.safetensors", None)


@pytest.mark.parametrize("option", ["text_only", "export_paged_attention", "fp8_kv_cache"])
def test_autoregressive_build_options_are_rejected(option):
    from mobius.integrations.transformers._builder import build_transformers_model

    with pytest.raises(ValueError, match=option):
        build_transformers_model(CLEF_FLASH_MODEL_ID, **{option: True})


def test_runtime_contract_does_not_claim_autoregressive_support(tmp_path):
    from mobius.integrations.onnx_genai.auto_export import write_onnx_genai_config
    from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

    config = tiny_config()
    package = build_from_module(ClefFlashModel(config), config, "clef-decision")
    with pytest.raises(ValueError, match="not an autoregressive"):
        write_ort_genai_config(package, str(tmp_path))
    paths = write_onnx_genai_config(package, str(tmp_path))
    metadata = json.loads(pathlib.Path(paths["runtime_compatibility"]).read_text())
    assert metadata["runtime_validation_status"] == "unsupported-by-tested-runtime"
    assert set(metadata["components"]) == set(package)
    assert metadata["components"]["decision_head"]["outputs"] == ["logits", "probabilities"]
    assert not (tmp_path / "genai_config.json").exists()


def test_cli_export_writes_advisory_contract_for_clef(tmp_path):
    from mobius.__main__ import _save_package, build_parser

    config = tiny_config()
    package = build_from_module(ClefFlashModel(config), config, "clef-decision")
    args = build_parser().parse_args(
        [
            "build",
            "--model",
            CLEF_FLASH_MODEL_ID,
            str(tmp_path),
            "--no-weights",
            "--runtime",
            "onnx-genai",
        ]
    )
    _save_package(package, str(tmp_path), args, None, None)
    metadata = json.loads((tmp_path / "runtime_compatibility.json").read_text())
    assert metadata["runtime_validation_status"] == "unsupported-by-tested-runtime"
    assert all((tmp_path / name / "model.onnx").exists() for name in package)
    assert not (tmp_path / "genai_config.json").exists()


@pytest.mark.integration
def test_published_checkpoint_builds_all_decision_components():
    import mobius

    package = mobius.build(CLEF_FLASH_MODEL_ID, load_weights=False, dtype="f32")
    assert set(package) == {"decoder", "vision_encoder", "embedding", "decision_head"}
    assert package.config.hidden_size == 4096
    assert package.config.head_width == 1024
    assert all(
        model.metadata_props["mobius.source_revision"] == CLEF_FLASH_REVISION
        for model in package.values()
    )


@pytest.fixture
def upstream():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        CLEF_FLASH_MODEL_ID, "joint_schema_model.py", revision=CLEF_FLASH_REVISION
    )
    spec = importlib.util.spec_from_file_location("_clef_test_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.integration
def test_encoder_matches_pinned_publisher(upstream):
    record = _record()
    actual = encode_clef_record(_Tokenizer(), record, max_state_tokens=3)
    expected = upstream.encode_record(_Tokenizer(), record, max_state_tokens=3)
    assert actual.input_ids[0].tolist() == list(expected.input_ids)
    assert actual.question_spans.tolist() == [
        list(q.question_span) for q in expected.questions
    ]
    assert actual.option_spans.tolist() == [
        list(span) for q in expected.questions for span in q.option_spans
    ]


@pytest.mark.integration
def test_encoder_matches_publisher_with_pinned_image_processor(upstream):
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        CLEF_FLASH_MODEL_ID, revision=CLEF_FLASH_REVISION, trust_remote_code=False
    )
    source = {**_record(), "images": [Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))]}
    actual = encode_clef_record(processor.tokenizer, source, processor=processor)
    expected = upstream.encode_record(processor.tokenizer, source, processor=processor)
    expected_batch = upstream.collate_records(
        [expected], processor.tokenizer.pad_token_id, torch.device("cpu")
    )
    np.testing.assert_array_equal(actual.input_ids, expected_batch["input_ids"].numpy())
    np.testing.assert_array_equal(
        actual.media["mm_token_type_ids"], expected_batch["media"]["mm_token_type_ids"].numpy()
    )
    np.testing.assert_array_equal(
        actual.media["image_grid_thw"], expected_batch["media"]["image_grid_thw"].numpy()
    )
    np.testing.assert_allclose(
        actual.media["pixel_values"],
        expected_batch["media"]["pixel_values"].numpy(),
        atol=1e-6,
    )
    assert actual.question_spans.tolist() == [
        list(question.question_span) for question in expected.questions
    ]


@pytest.mark.integration
@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_head_matches_pinned_publisher(upstream, dtype):
    import onnxruntime as ort

    if (
        dtype == ir.DataType.BFLOAT16
        and "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        pytest.skip("ORT CPU does not provide the BF16 arithmetic kernels used by this head")
    config = tiny_config(dtype=dtype)
    torch_dtype = {
        ir.DataType.FLOAT: torch.float32,
        ir.DataType.FLOAT16: torch.float16,
        ir.DataType.BFLOAT16: torch.bfloat16,
    }[dtype]
    torch.manual_seed(42)
    reference = (
        upstream.JointSchemaHead(
            hidden_size=config.hidden_size,
            width=config.head_width,
            routing_layers=config.head_routing_layers,
            layers=config.head_layers,
            heads=config.head_heads,
            feedforward=config.head_feedforward,
        )
        .to(dtype=torch_dtype)
        .eval()
    )
    # Nonzero gates/scales ensure every scoring branch contributes.
    with torch.no_grad():
        reference.prior_logit_scale.fill_(0.5)
        reference.joint_logit_scale.fill_(0.3)
        reference.residual_gate.fill_(0.7)
    state = reference.state_dict()
    output_table = torch.randn(config.vocab_size, config.hidden_size, dtype=torch_dtype)
    head = ClefJointHead(config)
    package = build_from_module(head, config, _HeadTask())
    weights = {"output_embedding.weight": output_table}
    for name, _ in head.named_parameters():
        if name == "output_embedding.weight":
            continue
        source, split = clef_head_source_name(name)
        weights[name] = state[source] if split is None else state[source].chunk(3)[split]
    package.apply_weights(weights)
    session = OnnxModelSession(
        package["model"], device="cuda" if dtype == ir.DataType.BFLOAT16 else "cpu"
    )
    for spans, owners, types in (
        ([(2, 4), (4, 6), (6, 9), (9, 11), (11, 13)], [0, 0, 1, 1, 1], [0, 1]),
        ([(3, 6)], [0], [2]),
    ):
        hidden = torch.randn(1, 16, config.hidden_size, dtype=torch_dtype)
        ids = torch.randint(config.vocab_size, (1, 16))
        question_spans = [(0, 2)] * len(types)
        questions = tuple(
            upstream.EncodedQuestion(
                question_id=str(i),
                question_type=kind,
                question_span=question_spans[i],
                option_spans=tuple(
                    s for s, owner in zip(spans, owners, strict=True) if owner == i
                ),
                option_ids=tuple(str(j) for j, owner in enumerate(owners) if owner == i),
            )
            for i, kind in enumerate(types)
        )
        record = upstream.EncodedRecord(tuple(ids[0].tolist()), questions, "test")
        with torch.no_grad():
            expected = (
                torch.cat(
                    reference(hidden, ids, torch.ones_like(ids), [record], output_table)[0]
                )
                .float()
                .numpy()
            )
        # ml_dtypes preserves bf16 at the ORT boundary.
        import ml_dtypes

        np_dtype = {
            ir.DataType.FLOAT: np.float32,
            ir.DataType.FLOAT16: np.float16,
            ir.DataType.BFLOAT16: ml_dtypes.bfloat16,
        }[dtype]
        feeds = {
            "hidden_states": hidden.float().numpy().astype(np_dtype),
            "input_ids": ids.numpy(),
            "question_spans": np.array(question_spans, dtype=np.int64),
            "option_spans": np.array(spans, dtype=np.int64),
            "option_question_ids": np.array(owners, dtype=np.int64),
            "question_types": np.array(types, dtype=np.int64),
        }
        actual = session.run(feeds)
        tolerance = 1e-4 if dtype == ir.DataType.FLOAT else 1e-2
        np.testing.assert_allclose(
            actual["logits"].astype(np.float32), expected, atol=tolerance, rtol=tolerance
        )
        for i in range(len(types)):
            np.testing.assert_allclose(
                actual["probabilities"][np.array(owners) == i].astype(np.float32).sum(),
                1.0,
                atol=0.01,
            )


@pytest.mark.integration
@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize(
    "layer_types",
    [
        ["linear_attention", "full_attention"],
        ["full_attention", "full_attention"],
    ],
)
def test_streamed_full_pipeline_matches_upstream_for_text_image_and_video(
    upstream,
    tmp_path,
    layer_types,
    dtype,
):
    from safetensors.torch import save_file
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    import mobius

    torch_dtype = torch.float32 if dtype == "f32" else torch.float16
    np_dtype = np.float32 if dtype == "f32" else np.float16
    tolerance = 1e-4 if dtype == "f32" else 1e-2
    torch.manual_seed(7)
    hf_config = Qwen3_5Config(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 256,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "layer_types": layer_types,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "depth": 1,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
    )
    reference = Qwen3_5ForConditionalGeneration(hf_config).to(torch_dtype).eval()
    head_config = {
        "hidden_size": 32,
        "width": 16,
        "routing_layers": 2,
        "layers": 2,
        "heads": 4,
        "feedforward": 32,
    }
    head = upstream.JointSchemaHead(**head_config).to(torch_dtype).eval()
    hf_config.save_pretrained(tmp_path)
    save_file(reference.state_dict(), str(tmp_path / "model.safetensors"))
    save_file(head.state_dict(), str(tmp_path / "joint_head.safetensors"))
    (tmp_path / "joint_head_config.json").write_text(json.dumps(head_config))
    package = mobius.build(str(tmp_path), dtype=dtype)
    assert package.config.partial_rotary_factor == pytest.approx(0.5)
    assert package.config.rope_theta == 10000
    assert package.config.mrope_section == [1, 1, 0]
    assert package.weight_loading_report["components"]["decision_head"]["assigned_tensors"] > 0
    # Serialize the lazy bindings, then execute each preceding ONNX stage.
    package.save(str(tmp_path / "onnx"), progress_bar=False, max_workers=1)
    sessions = {name: OnnxModelSession(model) for name, model in package.items()}
    grid = torch.tensor([[1, 4, 4]])
    for modalities in ((), ("image",), ("video",), ("video", "image")):
        ids = [1, 2]
        media_types = [0, 0]
        media = {}
        for kind in modalities:
            token = 250 if kind == "image" else 251
            ids += [252] + [token] * 4 + [253, 3]
            media_types += [0] + [1 if kind == "image" else 2] * 4 + [0, 0]
            if kind == "image":
                media["pixel_values"] = torch.randn(16, 24)
                media["image_grid_thw"] = grid
            else:
                media["pixel_values_videos"] = torch.randn(16, 24)
                media["video_grid_thw"] = grid
        ids += [4, 5, 6, 7, 8, 9, 10, 11]
        media_types += [0] * 8
        input_ids = torch.tensor([ids])
        mask = torch.ones_like(input_ids)
        positions, _ = reference.model.get_rope_index(
            input_ids,
            torch.tensor([media_types]),
            image_grid_thw=media.get("image_grid_thw"),
            video_grid_thw=media.get("video_grid_thw"),
            attention_mask=mask,
        )
        with torch.no_grad():
            expected_hidden = reference.model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
                **media,
            ).last_hidden_state
        features = {}
        for kind in ("image", "video"):
            pixel_key = "pixel_values" if kind == "image" else "pixel_values_videos"
            if pixel_key in media:
                features[f"{kind}_features"] = sessions["vision_encoder"].run(
                    {
                        "pixel_values": media[pixel_key].numpy(),
                        "image_grid_thw": grid.numpy(),
                    }
                )["image_features"]
            else:
                features[f"{kind}_features"] = np.zeros((0, 32), dtype=np_dtype)
        embeds = sessions["embedding"].run(
            {
                "input_ids": input_ids.numpy(),
                **features,
            }
        )["inputs_embeds"]
        if not modalities:
            with torch.no_grad():
                expected_embeds = reference.model.language_model.embed_tokens(input_ids)
            np.testing.assert_allclose(embeds, expected_embeds.numpy(), atol=0, rtol=0)
        hidden = sessions["decoder"].run(
            {
                "inputs_embeds": embeds,
                "attention_mask": mask.numpy(),
                "position_ids": positions.numpy(),
            }
        )["hidden_states"]
        np.testing.assert_allclose(
            hidden, expected_hidden.numpy(), atol=tolerance, rtol=tolerance
        )
        questions = (
            upstream.EncodedQuestion("a", 0, (0, 2), ((2, 4), (4, 6)), ("true", "false")),
            upstream.EncodedQuestion("b", 1, (0, 2), ((6, 8),), ("one",)),
        )
        record = upstream.EncodedRecord(tuple(ids), questions, "test")
        with torch.no_grad():
            expected = torch.cat(
                head(expected_hidden, input_ids, mask, [record], reference.lm_head.weight)[0]
            ).numpy()
        actual = sessions["decision_head"].run(
            {
                "hidden_states": hidden,
                "input_ids": input_ids.numpy(),
                "question_spans": np.array([[0, 2], [0, 2]], dtype=np.int64),
                "option_spans": np.array([[2, 4], [4, 6], [6, 8]], dtype=np.int64),
                "option_question_ids": np.array([0, 0, 1], dtype=np.int64),
                "question_types": np.array([0, 1], dtype=np.int64),
            }
        )
        np.testing.assert_allclose(actual["logits"], expected, atol=tolerance, rtol=tolerance)
        assert actual["probabilities"][2] == pytest.approx(1.0)
