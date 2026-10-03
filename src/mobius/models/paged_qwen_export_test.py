# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Packed, separate-K/V Qwen PagedAttention graph and config contracts."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import onnx_ir as ir
import pytest

from mobius.__main__ import _save_package
from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._testing import make_config
from mobius.integrations.onnx_genai.auto_export import write_onnx_genai_config
from mobius.integrations.ort_genai.auto_export import (
    _preflight_dense_paged_decoder,
    export_package,
    write_ort_genai_config,
)
from mobius.integrations.ort_genai.genai_config import GenaiConfigGenerator
from mobius.models.base import CausalLMModel
from mobius.models.qwen import Qwen3CausalLMModel
from mobius.tasks import CausalLMTask
from mobius.tasks._causal_lm import dense_paged_rejection


def _config(model_type="qwen2", **changes):
    args = dict(
        model_type=model_type,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rope_type="default",
        dtype=ir.DataType.FLOAT16,
        attn_qk_norm=model_type == "qwen3",
        export_paged_attention=True,
    )
    args.update(changes)
    return make_config(**args)


def _build(config):
    model = (
        Qwen3CausalLMModel(config) if config.model_type == "qwen3" else CausalLMModel(config)
    )
    return build_from_module(
        model, config, task=CausalLMTask(paged_cache=True), execution_provider="cuda"
    )


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_packed_paged_abi(model_type):
    config = _config(model_type)
    graph = _build(config)["model"].graph
    inputs = {v.name: v for v in graph.inputs}
    outputs = {v.name: v for v in graph.outputs}
    assert inputs["input_ids"].shape.rank() == 1
    assert inputs["attention_metadata"].shape.rank() == 1
    assert "position_ids" not in inputs and "attention_mask" not in inputs
    assert "slot_mapping" not in inputs
    assert "past_key_values.0.value" in inputs
    assert "present.1.value" in outputs
    for i in range(config.num_hidden_layers):
        for kind in ("key", "value"):
            inp = inputs[f"past_key_values.{i}.{kind}"]
            out = outputs[f"present.{i}.{kind}"]
            assert out.shape == inp.shape
            assert out.dtype == inp.dtype == ir.DataType.FLOAT16
    nodes = [n for n in graph if n.op_type == "PagedAttention"]
    assert len(nodes) == 2
    for i, node in enumerate(nodes):
        assert node.domain == "com.microsoft"
        assert len(node.outputs) == 3
        assert len(node.inputs) == 17
        assert node.inputs[3].name == f"past_key_values.{i}.key"
        assert node.inputs[4].name == f"past_key_values.{i}.value"
        assert node.inputs[5].name == "cumulative_sequence_lengths"
        assert node.inputs[6].name == "past_sequence_lengths"
        assert node.inputs[7].name == "block_table"
        assert node.inputs[8] is not None and node.inputs[9] is not None
        assert node.inputs[10] is None and all(v is None for v in node.inputs[11:16])
        assert node.inputs[16].name == "attention_metadata"
        assert "kv_cache_layout" not in node.attributes
        assert node.attributes["do_rotary"].value == 1
        assert outputs[f"present.{i}.key"] is node.outputs[1]
        assert outputs[f"present.{i}.value"] is node.outputs[2]
    assert outputs["logits"].dtype == ir.DataType.FLOAT
    if model_type == "qwen3":
        assert any("q_norm" in v.name for v in graph.initializers.values())
        assert any("k_norm" in v.name for v in graph.initializers.values())
        # The operator consumes the already-normalized Q and K (no fused
        # norm weights in slots 12/13); the operator itself applies RoPE.
        for node in nodes:
            for projection in node.inputs[:2]:
                normalized = projection.producer().inputs[0]
                assert normalized.producer().op_type == "Reshape"
                assert normalized.producer().inputs[0].producer().op_type == "RMSNormalization"
    assert not any(n.op_type == "RotaryEmbedding" for n in graph)


def test_genai_engine_mapping():
    config = _config()
    graph = _build(config)["model"].graph
    inputs = {v.name for v in graph.inputs}
    gen = GenaiConfigGenerator.from_config(
        config,
        "qwen2",
        ep="cuda",
        decoder_inputs={
            **{n: n for n in inputs if not n.startswith("past_key_values.")},
            "past_key_names": "past_key_values.%d.key",
            "past_value_names": "past_key_values.%d.value",
        },
        decoder_outputs={
            "logits": "logits",
            "present_key_names": "present.%d.key",
            "present_value_names": "present.%d.value",
        },
    ).generate()
    assert gen["engine"]["dynamic_batching"]["block_size"] == 256
    assert "state_groups" not in gen["model"]["decoder"]
    assert gen["search"]["past_present_share_buffer"] is True
    assert gen["model"]["context_length"] == config.max_position_embeddings
    assert gen["search"]["max_length"] <= config.max_position_embeddings
    assert "position_ids" not in gen["model"]["decoder"]["inputs"]


@pytest.mark.parametrize(
    "model_type,config_type",
    [
        ("qwen2", "Qwen2Config"),
        ("qwen3", "Qwen3Config"),
    ],
)
def test_real_transformers_full_attention_schedule(model_type, config_type):
    transformers = pytest.importorskip("transformers")
    hf_config = getattr(transformers, config_type)(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    assert hf_config.layer_types == ["full_attention"] * 2
    config = dataclasses.replace(
        ArchitectureConfig.from_transformers(hf_config),
        dtype=ir.DataType.FLOAT16,
        export_paged_attention=True,
    )
    assert config.model_type == model_type
    assert dense_paged_rejection(config) is None
    assert (
        len([n for n in _build(config)["model"].graph if n.op_type == "PagedAttention"]) == 2
    )


def test_paged_engine_rejects_context_longer_than_rope_cache():
    with pytest.raises(ValueError, match="RoPE cache length"):
        GenaiConfigGenerator.from_config(_config(), "qwen2", ep="cuda", context_length=65)
    generator = GenaiConfigGenerator.from_config(_config(), "qwen2", ep="cuda")
    generator._search_overrides = {"max_length": 65}
    with pytest.raises(ValueError, match=r"search\.max_length"):
        generator.generate()


def test_paged_generator_requires_graph_abi():
    with pytest.raises(ValueError, match="graph-derived decoder inputs and outputs"):
        GenaiConfigGenerator.from_config(_config(), "qwen2", ep="cuda").generate()
    generator = GenaiConfigGenerator.from_config(
        _config(),
        "qwen2",
        ep="cuda",
        decoder_inputs={"input_ids": "input_ids"},
        decoder_outputs={"logits": "logits"},
    )
    with pytest.raises(ValueError, match="complete graph-derived"):
        generator.generate()


def test_paged_generator_rejects_webgpu_engine():
    with pytest.raises(ValueError, match="WebGPU page allocation is unsupported"):
        GenaiConfigGenerator.from_config(_config(), "qwen2", ep="webgpu")


@pytest.mark.parametrize(
    "ep,dtype,reason",
    [
        ("webgpu", ir.DataType.FLOAT16, "WebGPU page allocation is unsupported"),
        ("cuda", None, "requires FP16"),
        ("cuda", ir.DataType.FLOAT, "requires FP16"),
        ("cuda", ir.DataType.BFLOAT16, "BF16 is disabled"),
    ],
)
def test_direct_paged_generator_requires_supported_engine(ep, dtype, reason):
    generator = GenaiConfigGenerator(
        "qwen2",
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        ep=ep,
        dense_paged=True,
        dtype=dtype,
    )
    with pytest.raises(ValueError, match=reason):
        generator.generate()


def test_direct_paged_generator_accepts_cuda_with_graph_mapping():
    generator = GenaiConfigGenerator(
        "qwen2",
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        ep="cuda",
        dense_paged=True,
        dtype=ir.DataType.FLOAT16,
        decoder_inputs={
            "input_ids": "input_ids",
            "block_table": "block_table",
            "cumulative_sequence_lengths": "cumulative_sequence_lengths",
            "past_sequence_lengths": "past_sequence_lengths",
            "attention_metadata": "attention_metadata",
            "past_key_names": "past_key_values.%d.key",
            "past_value_names": "past_key_values.%d.value",
        },
        decoder_outputs={
            "logits": "logits",
            "present_key_names": "present.%d.key",
            "present_value_names": "present.%d.value",
        },
    )
    assert generator.generate()["engine"]["dynamic_batching"]["block_size"] == 256


def test_paged_generator_rechecks_provider_before_generation():
    generator = GenaiConfigGenerator.from_config(_config(), "qwen2", ep="cuda")
    generator.ep = "webgpu"
    with pytest.raises(ValueError, match="WebGPU page allocation is unsupported"):
        generator.generate()


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_bfloat16_paged_export_rejected(model_type):
    config = _config(model_type, dtype=ir.DataType.BFLOAT16)
    with pytest.raises(ValueError, match="bfloat16 is disabled"):
        _build(config)
    with pytest.raises(ValueError, match="BF16 is disabled"):
        GenaiConfigGenerator.from_config(config, model_type, ep="cuda")


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"dtype": ir.DataType.FLOAT}, "float16"),
        ({"dtype": ir.DataType.BFLOAT16}, "bfloat16 is disabled"),
        ({"sliding_window": 32}, "sliding"),
        ({"head_dim": 15}, "divisible"),
        ({"qk_rope_head_dim": 8}, "full standard RoPE"),
        ({"rope_type": "yarn"}, "RoPE"),
        ({"num_nextn_predict_layers": 1}, "Multi-Token"),
        ({"model_type": "qwen2_vl_text"}, "text-only"),
        ({"model_type": "qwen3_moe"}, "text-only"),
        ({"layer_types": ["full_attention"]}, "hybrid"),
        ({"layer_types": ["full_attention", "sliding_attention"]}, "hybrid"),
    ],
)
def test_rejected_configs(change, reason):
    config = _config(**change)
    assert reason in (dense_paged_rejection(config) or "")


def test_flag_off_keeps_standard_graph():
    config = dataclasses.replace(_config(), export_paged_attention=False)
    graph = build_from_module(CausalLMModel(config), config, task=CausalLMTask())[
        "model"
    ].graph
    assert not any(n.op_type == "PagedAttention" for n in graph)


@pytest.mark.parametrize(
    "model_type,dtype",
    [
        ("qwen2", ir.DataType.FLOAT16),
        ("qwen3", ir.DataType.FLOAT16),
    ],
)
def test_ort_config_from_real_graph(tmp_path, model_type, dtype):
    pkg = _build(_config(model_type, dtype=dtype))
    path = write_ort_genai_config(pkg, str(tmp_path), ep="cuda")["genai_config"]
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    inputs = result["model"]["decoder"]["inputs"]
    assert inputs["block_table"] == "block_table"
    assert inputs["cumulative_sequence_lengths"] == "cumulative_sequence_lengths"
    assert inputs["past_sequence_lengths"] == "past_sequence_lengths"
    assert inputs["attention_metadata"] == "attention_metadata"
    assert inputs["past_value_names"] == "past_key_values.%d.value"
    assert result["engine"]["dynamic_batching"]["block_size"] == 256
    assert result["model"]["decoder"]["outputs"]["present_value_names"] == "present.%d.value"


def test_webgpu_engine_export_rejected_before_writing(tmp_path):
    pkg = _build(_config())
    with pytest.raises(ValueError, match="WebGPU page allocation is unsupported"):
        write_ort_genai_config(pkg, str(tmp_path), ep="webgpu")
    assert not list(tmp_path.iterdir())


def test_package_export_rejects_webgpu_before_saving(tmp_path):
    pkg = _build(_config())
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="WebGPU page allocation is unsupported"):
        export_package(pkg, str(output), ep="webgpu")
    assert not output.exists()


def test_cli_export_rejects_webgpu_before_saving(tmp_path):
    pkg = _build(_config())
    args = SimpleNamespace(
        runtime="ort-genai",
        release=False,
        max_shard_size=None,
        external_data="onnx",
        execution_provider="webgpu",
    )
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="WebGPU page allocation is unsupported"):
        _save_package(pkg, str(output), args, None, None)
    assert not output.exists()


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"export_paged_attention": False}, "disagrees"),
        ({"dtype": ir.DataType.BFLOAT16}, "BF16 is disabled"),
        ({"model_type": "llama"}, "incompatible with config.model_type"),
        ({"model_type": "qwen3_5_text"}, "incompatible with config.model_type"),
        ({"max_position_embeddings": 128}, "RoPE cache length"),
        ({"num_hidden_layers": 1}, "scheduler or cache I/O"),
        ({"vocab_size": 129}, "packed"),
        ({"num_attention_heads": 8}, "matching head counts"),
    ],
)
def test_paged_config_must_match_graph(tmp_path, change, reason):
    pkg = _build(_config())
    pkg.config = dataclasses.replace(pkg.config, **change)
    with pytest.raises(ValueError, match=reason):
        write_ort_genai_config(pkg, str(tmp_path), ep="cuda")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model_type", ["qwen3_5_text", "qwen3_5_vl_text"])
def test_dense_preflight_does_not_claim_qwen35_hybrid_graph(model_type):
    pkg = _build(_config())
    pkg.config = dataclasses.replace(
        pkg.config, model_type=model_type, export_paged_attention=False
    )
    graph = pkg["model"].graph
    graph.inputs.append(ir.Value(name="past_key_values.0.recurrent_state"))
    graph.append(
        ir.Node(
            op_type="LinearAttention",
            domain="com.microsoft",
            inputs=[],
            num_outputs=1,
        )
    )
    _preflight_dense_paged_decoder(pkg, "cuda")


def test_paged_graph_must_have_expected_page_shape(tmp_path):
    pkg = _build(_config())
    cache = next(
        value for value in pkg["model"].graph.inputs if value.name == "past_key_values.0.key"
    )
    cache.shape = ir.Shape(["num_blocks", 128, 2, 16])
    with pytest.raises(ValueError, match=r"incompatible past_key_values\.0\.key pages"):
        write_ort_genai_config(pkg, str(tmp_path), ep="cuda")
    assert not list(tmp_path.iterdir())


def test_paged_large_rope_tables_remain_exportable(tmp_path):
    pkg = _build(_config(max_position_embeddings=2048))
    assert write_ort_genai_config(pkg, str(tmp_path), ep="cuda")["genai_config"] == str(
        tmp_path / "genai_config.json"
    )


def test_invalid_ep_and_task_rejected():
    config = _config()
    with pytest.raises(ValueError, match="CUDA FP16"):
        build_from_module(
            CausalLMModel(config),
            config,
            task=CausalLMTask(paged_cache=True),
            execution_provider="cpu",
        )
    with pytest.raises(ValueError, match="paged_cache=True"):
        build_from_module(
            CausalLMModel(config),
            config,
            task=CausalLMTask(),
            execution_provider="cuda",
        )
    with pytest.raises(ValueError, match="fp8_kv_cache"):
        build_from_module(
            CausalLMModel(config),
            config,
            task=CausalLMTask(paged_cache=True),
            execution_provider="cuda",
            fp8_kv_cache=True,
        )
    with pytest.raises(ValueError, match="prune_prefill_prefix"):
        build_from_module(
            CausalLMModel(config),
            config,
            task=CausalLMTask(paged_cache=True),
            execution_provider="cuda",
            prune_prefill_prefix=True,
        )


def test_native_workflow_rejects_packed_page_contract(tmp_path):
    with pytest.raises(ValueError, match="ORT GenAI Engine"):
        write_onnx_genai_config(_build(_config()), str(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_native_cli_rejects_packed_pages_before_saving(tmp_path, model_type):
    pkg = _build(_config(model_type))
    args = SimpleNamespace(
        runtime="onnx-genai",
        release=False,
        max_shard_size=None,
        external_data="onnx",
        execution_provider="cuda",
    )
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="ORT GenAI Engine"):
        _save_package(pkg, str(output), args, None, None)
    assert not output.exists()


@pytest.mark.integration
@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
@pytest.mark.parametrize("metadata_with_lower_bound", [False, True])
@pytest.mark.parametrize("head_dim", [16, 64, 128])
def test_cuda_prefill_and_decode_match_dynamic_gqa(
    model_type, metadata_with_lower_bound, head_dim
):
    """Exercise in-place page binding and compare logits/cache over two invocations."""
    import numpy as np

    ort = pytest.importorskip("onnxruntime")

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ORT CUDA provider is unavailable")
    config = _config(
        model_type,
        num_hidden_layers=1,
        hidden_size=4 * head_dim,
        intermediate_size=8 * head_dim,
        head_dim=head_dim,
    )
    paged = _build(config)["model"]
    baseline_config = dataclasses.replace(config, export_paged_attention=False)
    model_class = Qwen3CausalLMModel if model_type == "qwen3" else CausalLMModel
    baseline = build_from_module(
        model_class(baseline_config),
        baseline_config,
        task=CausalLMTask(),
        execution_provider="cuda",
    )["model"]

    # Both graphs must use exactly the same weights. Real rotary tables are
    # already initialized; projection weights use reproducible synthetic values.
    for model in (paged, baseline):
        for name, value in model.graph.initializers.items():
            if value.const_value is not None:
                continue
            shape = tuple(value.shape)
            if "norm.weight" in name or "layernorm.weight" in name:
                data = np.ones(shape, dtype=np.float16)
            else:
                data = (
                    np.random.default_rng(sum(name.encode()))
                    .normal(0, 0.04, size=shape)
                    .astype(np.float16)
                )
            value.const_value = ir.tensor(data)
    paged_session = ort.InferenceSession(
        ir.serde.serialize_model(paged).SerializeToString(),
        providers=["CUDAExecutionProvider"],
    )
    baseline_session = ort.InferenceSession(
        ir.serde.serialize_model(baseline).SerializeToString(),
        providers=["CUDAExecutionProvider"],
    )
    if paged_session.get_providers()[0] != "CUDAExecutionProvider":
        pytest.skip("CUDA device is unavailable")

    pools = {
        kind: ort.OrtValue.ortvalue_from_numpy(
            np.zeros((1, 256, 2, head_dim), dtype=np.float16), "cuda", 0
        )
        for kind in ("key", "value")
    }
    baseline_cache = {kind: np.zeros((1, 2, 0, head_dim), dtype=np.float16) for kind in pools}
    # Preserve the stricter existing small-head gate; use the FP16 full-logit
    # budget for the representative Qwen head widths.
    rtol, atol = (0.02, 0.002) if head_dim == 16 else (0.01, 0.01)
    for tokens, past in (([1, 2, 3], 0), ([4], 3)):
        binding = paged_session.io_binding()
        metadata = [len(tokens), past + len(tokens)]
        if metadata_with_lower_bound:
            metadata.append(past)
        for name, array in {
            "input_ids": np.array(tokens, dtype=np.int64),
            "block_table": np.array([[0]], dtype=np.int32),
            "cumulative_sequence_lengths": np.array([0, len(tokens)], dtype=np.int32),
            "past_sequence_lengths": np.array([past], dtype=np.int32),
            "attention_metadata": np.array(metadata, dtype=np.int32),
        }.items():
            binding.bind_cpu_input(name, array)
        for kind, pool in pools.items():
            binding.bind_ortvalue_input(f"past_key_values.0.{kind}", pool)
            binding.bind_ortvalue_output(f"present.0.{kind}", pool)
        binding.bind_output("logits", "cuda")
        paged_session.run_with_iobinding(binding)
        packed_logits = binding.get_outputs()[-1].numpy()

        feed = {
            "input_ids": np.array([tokens], dtype=np.int64),
            "attention_mask": np.ones((1, past + len(tokens)), dtype=np.int64),
            **{f"past_key_values.0.{kind}": cache for kind, cache in baseline_cache.items()},
        }
        outputs = baseline_session.run(None, feed)
        np.testing.assert_allclose(packed_logits, outputs[0][0], rtol=rtol, atol=atol)
        for index, (kind, pool) in enumerate(pools.items(), start=1):
            baseline_cache[kind] = outputs[index]
            page_slice = pool.numpy()[:, : past + len(tokens)]
            np.testing.assert_allclose(
                np.transpose(page_slice, (0, 2, 1, 3)),
                baseline_cache[kind],
                rtol=rtol,
                atol=atol,
            )


@pytest.mark.integration
@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
@pytest.mark.parametrize("head_dim", [16, 64, 128])
def test_cuda_packed_batch_crosses_page_boundary(model_type, head_dim):
    import numpy as np

    ort = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ORT CUDA provider is unavailable")

    config = _config(
        model_type,
        num_hidden_layers=1,
        max_position_embeddings=512,
        hidden_size=4 * head_dim,
        intermediate_size=8 * head_dim,
        head_dim=head_dim,
    )
    baseline_config = dataclasses.replace(config, export_paged_attention=False)
    model_class = Qwen3CausalLMModel if model_type == "qwen3" else CausalLMModel
    paged = _build(config)["model"]
    baseline = build_from_module(
        model_class(baseline_config),
        baseline_config,
        task=CausalLMTask(),
        execution_provider="cuda",
    )["model"]
    for model in (paged, baseline):
        for name, initializer in model.graph.initializers.items():
            if initializer.const_value is not None:
                continue
            rng = np.random.default_rng(sum(name.encode()))
            shape = tuple(initializer.shape)
            if "norm.weight" in name or "layernorm.weight" in name:
                data = rng.uniform(0.8, 1.2, size=shape).astype(np.float16)
            else:
                data = rng.normal(0, 0.04, size=shape).astype(np.float16)
            initializer.const_value = ir.tensor(data)

    paged_session = ort.InferenceSession(
        ir.serde.serialize_model(paged).SerializeToString(),
        providers=["CUDAExecutionProvider"],
    )
    baseline_session = ort.InferenceSession(
        ir.serde.serialize_model(baseline).SerializeToString(),
        providers=["CUDAExecutionProvider"],
    )
    pools = {
        kind: ort.OrtValue.ortvalue_from_numpy(
            np.zeros((4, 256, 2, head_dim), dtype=np.float16), "cuda", 0
        )
        for kind in ("key", "value")
    }
    baseline_caches = [
        {kind: np.zeros((1, 2, 0, head_dim), dtype=np.float16) for kind in pools}
        for _ in range(2)
    ]
    rtol, atol = (0.02, 0.002) if head_dim == 16 else (0.01, 0.01)
    block_table = np.array([[2, 0], [3, 1]], dtype=np.int32)
    token_batches = [
        [[i % 120 + 1 for i in range(254)], [2, 3, 4]],
        [[5, 6, 7], [8, 9]],
    ]
    past_lengths = [0, 0]
    for tokens_by_row in token_batches:
        lengths = [len(tokens) for tokens in tokens_by_row]
        cumulative = np.array([0, lengths[0], sum(lengths)], dtype=np.int32)
        metadata = np.array(
            [
                max(lengths),
                max(p + n for p, n in zip(past_lengths, lengths)),
                min(past_lengths),
            ],
            dtype=np.int32,
        )
        binding = paged_session.io_binding()
        for name, data in {
            "input_ids": np.array(
                [token for tokens in tokens_by_row for token in tokens], dtype=np.int64
            ),
            "block_table": block_table,
            "cumulative_sequence_lengths": cumulative,
            "past_sequence_lengths": np.array(past_lengths, dtype=np.int32),
            "attention_metadata": metadata,
        }.items():
            binding.bind_cpu_input(name, data)
        for kind, pool in pools.items():
            binding.bind_ortvalue_input(f"past_key_values.0.{kind}", pool)
            binding.bind_ortvalue_output(f"present.0.{kind}", pool)
        binding.bind_output("logits", "cuda")
        paged_session.run_with_iobinding(binding)
        packed_logits = binding.get_outputs()[-1].numpy()

        for row, tokens in enumerate(tokens_by_row):
            past = past_lengths[row]
            outputs = baseline_session.run(
                None,
                {
                    "input_ids": np.array([tokens], dtype=np.int64),
                    "attention_mask": np.ones((1, past + len(tokens)), dtype=np.int64),
                    **{
                        f"past_key_values.0.{kind}": cache
                        for kind, cache in baseline_caches[row].items()
                    },
                },
            )
            np.testing.assert_allclose(
                packed_logits[cumulative[row] : cumulative[row + 1]],
                outputs[0][0],
                rtol=rtol,
                atol=atol,
            )
            for index, kind in enumerate(pools, start=1):
                baseline_caches[row][kind] = outputs[index]
                physical_pages = pools[kind].numpy()[block_table[row]]
                linear_cache = physical_pages.reshape(-1, 2, head_dim)[: past + len(tokens)]
                np.testing.assert_allclose(
                    linear_cache.transpose(1, 0, 2)[None],
                    baseline_caches[row][kind],
                    rtol=rtol,
                    atol=atol,
                )
            past_lengths[row] += len(tokens)
