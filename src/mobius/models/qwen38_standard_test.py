# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline Qwen3.8-27B text export and reduced Qwen3_5ForCausalLM parity.

The pinned official config exercises the public text-only builder. Numerical
tests use random weights and reduced dimensions, not the 27B checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
import transformers

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.integrations._weight_loading import apply_weights
from mobius.integrations.transformers._builder import build_transformers_model

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "testdata/configs/qwen3_8-27b.json"


def _hf_config(*, reduced: bool = False):
    config = transformers.Qwen3_5Config.from_dict(json.loads(_CONFIG_PATH.read_text()))
    if reduced:
        text = config.text_config
        text.hidden_size = 64
        text.intermediate_size = 128
        text.num_hidden_layers = 4
        text.layer_types = text.layer_types[:4]
        text.num_attention_heads = 6
        text.num_key_value_heads = 1
        text.head_dim = 32
        text.vocab_size = 256
        text.linear_num_key_heads = 2
        text.linear_num_value_heads = 6
        text.linear_key_head_dim = 8
        text.linear_value_head_dim = 12
        text.max_position_embeddings = 128
        text.rope_parameters["mrope_section"] = [2, 1, 1]
        text.bos_token_id = 1
        text.eos_token_id = 2
    return config


def _assert_standard(model: ir.Model) -> None:
    assert not model.functions, "standard-only export must inline local functions"
    nodes = list(model.graph.all_nodes())
    assert any(node.op_type == "Scan" for node in nodes)
    for node in nodes:
        assert node.domain in {"", "ai.onnx"}, (node.domain, node.op_type)
        onnx.defs.get_schema(
            node.op_type, model.opset_imports.get(node.domain, model.opset_imports[""]), ""
        )


def _build(config, path: Path, dtype: str):
    config.save_pretrained(path)
    package = build_transformers_model(
        str(path),
        text_only=True,
        load_weights=False,
        dtype=dtype,
        execution_provider="onnx-standard",
    )
    assert set(package) == {"model"}
    assert package.config.model_type == "qwen3_5_text"
    _assert_standard(package["model"])
    return package


def test_qwen38_official_text_config():
    config = ArchitectureConfig.from_transformers(_hf_config().text_config)
    assert config.num_hidden_layers == 64
    assert (
        config.layer_types
        == ["linear_attention"] * 3
        + ["full_attention"] * 1
        + (["linear_attention"] * 3 + ["full_attention"]) * 15
    )
    assert (config.hidden_size, config.intermediate_size, config.vocab_size) == (
        5120,
        17408,
        248320,
    )
    assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim) == (
        24,
        4,
        256,
    )
    assert config.mrope_interleaved
    assert config.mrope_section == [11, 11, 10]
    assert getattr(config, "mamba_ssm_dtype", None) == ir.DataType.FLOAT


@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize("text_config_only", [False, True])
def test_qwen38_standard_text_only_build(tmp_path: Path, dtype: str, text_config_only: bool):
    config = _hf_config(reduced=True)
    if text_config_only:
        config = config.text_config
    package = _build(config, tmp_path / "config", dtype)
    graph = package["model"].graph
    assert len([value for value in graph.inputs if "past_key_values." in value.name]) == 8
    for value in [*graph.inputs, *graph.outputs]:
        if value.name.endswith(".recurrent_state"):
            assert value.dtype == ir.DataType.FLOAT


@pytest.mark.parametrize("invalid_dtype", ["not-a-dtype", "int64"])
def test_qwen38_rejects_invalid_state_dtype(invalid_dtype: str):
    config = _hf_config(reduced=True).text_config
    config.mamba_ssm_dtype = invalid_dtype
    with pytest.raises(ValueError, match="mamba_ssm_dtype"):
        ArchitectureConfig.from_transformers(config)


@pytest.mark.parametrize("state_dtype", [None, "auto"])
def test_qwen38_unspecified_state_dtype_preserves_existing_behavior(
    tmp_path: Path, state_dtype: str | None
):
    config = _hf_config(reduced=True)
    if state_dtype is None:
        del config.text_config.mamba_ssm_dtype
    else:
        config.text_config.mamba_ssm_dtype = state_dtype
    package = _build(config, tmp_path / "config", "f16")
    assert package.config.mamba_ssm_dtype is None
    states = [
        value
        for value in package["model"].graph.inputs
        if value.name.endswith(".recurrent_state")
    ]
    assert states
    assert all(value.dtype == ir.DataType.FLOAT16 for value in states)


def test_qwen38_standard_check_rejects_nested_custom_op(tmp_path: Path):
    model = _build(_hf_config(reduced=True), tmp_path / "config", "f32")["model"]
    scan = next(node for node in model.graph if node.op_type == "Scan")
    body = scan.attributes["body"].value
    next(node for node in body if node.op_type == "MatMul").domain = "com.microsoft"
    with pytest.raises(AssertionError, match=r"com\.microsoft"):
        _assert_standard(model)


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_qwen38_full_config_standard_build(tmp_path: Path, dtype: str):
    """L2 config/graph evidence only: all 64 layers, without weight allocation."""
    package = _build(_hf_config(), tmp_path / "config", dtype)
    graph = package["model"].graph
    assert len([value for value in graph.inputs if value.name.endswith(".key")]) == 16
    assert len([value for value in graph.inputs if value.name.endswith(".conv_state")]) == 48


def _initial_feeds(model: ir.Model, batch: int) -> dict[str, np.ndarray]:
    feeds = {}
    for value in model.graph.inputs:
        if not value.name.startswith("past_key_values."):
            continue
        shape = [
            dim if isinstance(dim, int) else batch if i == 0 else 0
            for i, dim in enumerate(value.shape)
        ]
        feeds[value.name] = np.zeros(shape, dtype=value.dtype.numpy())
    return feeds


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize("batch,prefill", [(1, 1), (2, 6)])
def test_qwen38_standard_cuda_hf_parity(tmp_path: Path, dtype: str, batch: int, prefill: int):
    """Compare full logits and every carried state at prefill and four decode steps."""
    if (
        "CUDAExecutionProvider" not in ort.get_available_providers()
        or not torch.cuda.is_available()
    ):
        pytest.skip("CUDA ORT and PyTorch are required")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(38)
    config = _hf_config(reduced=True)
    config.text_config._attn_implementation = "eager"
    torch_dtype = torch.float32 if dtype == "f32" else torch.float16
    reference = (
        Qwen3_5ForCausalLM._from_config(config.text_config, dtype=torch_dtype).eval().cuda()
    )
    package = _build(config, tmp_path / "config", dtype)
    model = package["model"]
    processed = reference.state_dict()
    # Exercise the actual standalone model's HF name adapter.
    from mobius.models import Qwen35CausalLMModel

    processed = Qwen35CausalLMModel(package.config).preprocess_weights(dict(processed))
    apply_weights(model, processed)
    _assert_standard(model)
    package.save(str(tmp_path / "package"), progress_bar=False)
    reloaded = ModelPackage.load(str(tmp_path / "package"))
    model = reloaded["model"]
    _assert_standard(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "cuda-profile")
    session = ort.InferenceSession(
        str(tmp_path / "package/model.onnx"),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    rng = np.random.default_rng(38)
    tokens = rng.integers(3, 256, size=(batch, prefill), dtype=np.int64)
    state_feeds = _initial_feeds(model, batch)
    cache = None
    seen = 0
    output_names = [value.name for value in model.graph.outputs]
    tolerance = 1e-3 if dtype == "f32" else 1e-2
    for step in range(5):
        length = tokens.shape[1]
        mask = np.ones((batch, seen + length), dtype=np.int64)
        positions = np.broadcast_to(np.arange(seen, seen + length), tokens.shape).copy()
        with torch.no_grad():
            expected = reference(
                input_ids=torch.from_numpy(tokens).cuda(),
                attention_mask=torch.from_numpy(mask).cuda(),
                position_ids=torch.from_numpy(positions).cuda(),
                past_key_values=cache,
                use_cache=True,
            )
        cache = expected.past_key_values
        actual = dict(
            zip(
                output_names,
                session.run(
                    None,
                    {
                        "input_ids": tokens,
                        "attention_mask": mask,
                        "position_ids": positions,
                        **state_feeds,
                    },
                ),
                strict=True,
            )
        )
        target_logits = expected.logits.cpu().numpy()
        np.testing.assert_allclose(
            actual["logits"],
            target_logits,
            rtol=tolerance,
            atol=tolerance,
            err_msg=f"logits step {step}",
            strict=True,
        )
        expected_tokens = expected.logits[:, -1].argmax(-1).cpu().numpy()
        np.testing.assert_array_equal(actual["logits"][:, -1].argmax(-1), expected_tokens)
        for layer_idx, layer_type in enumerate(config.text_config.layer_types):
            layer = cache.layers[layer_idx]
            if layer_type == "linear_attention":
                expected_states = {
                    "conv_state": layer.conv_states[0][
                        ..., -(config.text_config.linear_conv_kernel_dim - 1) :
                    ],
                    "recurrent_state": layer.recurrent_states[0],
                }
            else:
                expected_states = {"key": layer.keys, "value": layer.values}
            for role, value in expected_states.items():
                name = f"present.{layer_idx}.{role}"
                target = value.cpu().numpy()
                np.testing.assert_allclose(
                    actual[name],
                    target,
                    rtol=tolerance,
                    atol=tolerance,
                    err_msg=f"{name} step {step}",
                    strict=True,
                )
                state_feeds[f"past_key_values.{layer_idx}.{role}"] = actual[name]
        seen += length
        tokens = expected_tokens[:, None]
    profile = json.loads(Path(session.end_profiling()).read_text())
    assert any(
        event.get("args", {}).get("provider") == "CUDAExecutionProvider"
        and event["args"].get("op_name") in {"MatMul", "FusedMatMul"}
        for event in profile
    ), "provider registration alone is not evidence of CUDA computation"
