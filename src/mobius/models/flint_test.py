# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent eager comparisons for Flint's three decoder-block families."""

from __future__ import annotations

import dataclasses
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnx_ir import tensor_adapters
from onnxscript import nn
from torch.nn import functional

from mobius._build_context import build_context
from mobius._builder import _cast_module_dtype, build_from_module
from mobius._configs import FlintConfig
from mobius._execution_providers import EpCapabilities, get_ep
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import create_attention_bias
from mobius.models.flint import (
    FLINT_CUDA_DOMAIN,
    FlintCausalLMModel,
    FlintCUDALinear,
    FlintDecoderLayer,
    FlintGate,
    FlintMLP,
    FlintMoE,
    FlintResidualRMSNorm,
    FlintRotaryEmbedding,
)


def _config(**overrides) -> FlintConfig:
    options = dict(
        model_type="flint",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=1024,
        pad_token_id=0,
        hidden_act="silu",
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=32,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layers=[0, 2],
        mlp_only_layers=(0,),
        sliding_window=513,
    )
    options.update(overrides)
    return FlintConfig(**options)


class _Block(nn.Module):
    def __init__(self, config: FlintConfig, layer_index: int):
        super().__init__()
        self.layer = FlintDecoderLayer(config, layer_index)
        self.rotary_emb = FlintRotaryEmbedding(config)
        self.config = config
        self.layer_index = layer_index

    def forward(self, op, hidden, mask, positions, past_key, past_value):
        window = (
            self.config.sliding_window
            if self.config.layer_types[self.layer_index] == "sliding_attention"
            else None
        )
        bias = create_attention_bias(op, hidden, mask, sliding_window=window)
        embeddings = (
            self.rotary_emb(op, positions)
            if self.layer_index not in (self.config.no_rope_layers or [])
            else None
        )
        output, present = self.layer(op, hidden, bias, embeddings, (past_key, past_value))
        return output, *present


def _initialize(
    module: nn.Module, dtype: ir.DataType, std: float = 0.08
) -> dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(57)
    state = {}
    for name, parameter in module.named_parameters():
        if parameter.const_value is not None:
            continue
        shape = tuple(parameter.shape)
        tensor = torch.randn(shape, generator=rng) * std
        if "norm.weight" in name:
            tensor += 1
        if name.endswith("softmax_offset"):
            tensor = torch.linspace(-0.7, 0.9, shape[0])
        parameter.const_value = tensor_adapters.TorchTensor(tensor)
        state[name] = tensor.to(
            torch.float32
            if getattr(parameter, "_keep_float32", False)
            else tensor_adapters.to_torch_dtype(dtype)
        )
    _cast_module_dtype(module, dtype)
    return state


def _build_block(
    config: FlintConfig, layer_index: int, ep: str = "default", *, std: float = 0.08
):
    module = _Block(config, layer_index)
    state = _initialize(module, config.dtype, std)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(
        builder, "hidden", ["batch", "query", config.hidden_size], config.dtype
    )
    mask = create_test_input(builder, "mask", ["batch", "total"], ir.DataType.INT64)
    positions = create_test_input(builder, "positions", ["batch", "query"], ir.DataType.INT64)
    cache_shape = ["batch", config.num_key_value_heads, "past", config.head_dim]
    past_key = create_test_input(builder, "past_key", cache_shape, config.dtype)
    past_value = create_test_input(builder, "past_value", cache_shape, config.dtype)
    with build_context(get_ep(ep), config.dtype):
        outputs = module(op, hidden, mask, positions, past_key, past_value)
    for name, output in zip(("output", "key", "value"), outputs, strict=True):
        output.name = name
        graph.outputs.append(output)
    graph.opset_imports["com.microsoft"] = 1
    return ir.Model(graph, ir_version=10), state


def _norm(x, weight, eps):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(x.dtype)


def _mlp(x, state, prefix):
    gate = functional.linear(x, state[f"{prefix}.gate_proj.weight"])
    up = functional.linear(x, state[f"{prefix}.up_proj.weight"])
    combined = (functional.silu(gate.float()) * up.float()).to(x.dtype)
    return functional.linear(combined, state[f"{prefix}.down_proj.weight"])


def _reference_moe(x, state, config):
    prefix = "layer.mlp"
    flat = x.reshape(-1, config.hidden_size)
    scores = functional.linear(flat, state[f"{prefix}.gate.weight"]).float().sigmoid()
    weights, indices = scores.topk(config.num_experts_per_tok, dim=-1, sorted=False)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    fc1 = state[f"{prefix}.fc1_experts_weights"]
    fc2 = state[f"{prefix}.fc2_experts_weights"]
    output = torch.zeros_like(flat, dtype=torch.float32)
    for expert in range(config.num_local_experts):
        tokens, slots = torch.where(indices == expert)
        gate, up = functional.linear(flat[tokens], fc1[expert]).chunk(2, dim=-1)
        combined = (functional.silu(gate.float()) * up.float()).to(x.dtype)
        expert_output = functional.linear(combined, fc2[expert])
        output.index_add_(0, tokens, expert_output.float() * weights[tokens, slots, None])
    output = output.to(x.dtype).reshape_as(x)
    if config.shared_expert_intermediate_size:
        output = output + _mlp(x, state, f"{prefix}.shared_expert")
    return output


def _reference_block(
    x,
    mask,
    positions,
    cache,
    state,
    config,
    layer_index,
    *,
    residual=None,
    keep_residual=False,
):
    prefix = "layer.self_attn"
    combined = x.float() if residual is None else x.float() + residual.float()
    normalized = _norm(
        combined, state["layer.input_layernorm.weight"], config.rms_norm_eps
    ).to(x.dtype)
    residual = combined.to(x.dtype)
    q = functional.linear(normalized, state[f"{prefix}.q_proj.weight"])
    k = functional.linear(normalized, state[f"{prefix}.k_proj.weight"])
    v = functional.linear(normalized, state[f"{prefix}.v_proj.weight"])
    q = q.reshape(*x.shape[:2], config.num_attention_heads, config.head_dim)
    k = k.reshape(*x.shape[:2], config.num_key_value_heads, config.head_dim)
    q = _norm(q, state[f"{prefix}.q_norm.weight"], config.rms_norm_eps)
    k = _norm(k, state[f"{prefix}.k_norm.weight"], config.rms_norm_eps)
    if layer_index not in (config.no_rope_layers or []):
        inv = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        angles = positions.float().cpu()[..., None] * inv
        cos = torch.cat((angles.cos(), angles.cos()), dim=-1).unsqueeze(2).to(x.device)
        sin = torch.cat((angles.sin(), angles.sin()), dim=-1).unsqueeze(2).to(x.device)
        q, k = q.float(), k.float()
        q_left, q_right = q.chunk(2, dim=-1)
        k_left, k_right = k.chunk(2, dim=-1)
        q = (q * cos + torch.cat((-q_right, q_left), dim=-1) * sin).to(x.dtype)
        k = (k * cos + torch.cat((-k_right, k_left), dim=-1) * sin).to(x.dtype)
    q = q.transpose(1, 2)
    k = torch.cat((cache[0], k.transpose(1, 2)), dim=2)
    v = torch.cat(
        (
            cache[1],
            v.reshape(*x.shape[:2], config.num_key_value_heads, config.head_dim).transpose(
                1, 2
            ),
        ),
        dim=2,
    )
    present = (k, v)
    groups = config.num_attention_heads // config.num_key_value_heads
    scores = q.float() @ k.repeat_interleave(groups, dim=1).float().transpose(-1, -2)
    scores = scores * config.head_dim**-0.5
    logical_positions = mask.cumsum(-1)
    distance = logical_positions[:, -x.shape[1] :, None] - logical_positions[:, None, :]
    allowed = (distance >= 0) & mask[:, None, :].bool()
    if config.layer_types[layer_index] == "sliding_attention":
        allowed &= distance < config.sliding_window
    scores = scores.masked_fill(~allowed[:, None], float("-inf"))
    sinks = state[f"{prefix}.softmax_offset"].float()[None, :, None, None]
    sinks = sinks.expand(*scores.shape[:-1], 1)
    probs = torch.cat((scores, sinks), dim=-1).softmax(-1)[..., :-1].to(x.dtype)
    output = probs @ v.repeat_interleave(groups, dim=1)
    output = output.transpose(1, 2).reshape(*x.shape[:2], -1)
    projected = functional.linear(output, state[f"{prefix}.o_proj.weight"])
    residual_float = residual.float() + projected.float()
    residual = residual_float.to(x.dtype)
    normalized = _norm(
        residual_float, state["layer.post_attention_layernorm.weight"], config.rms_norm_eps
    ).to(x.dtype)
    feedforward = (
        _mlp(normalized, state, "layer.mlp")
        if layer_index in config.mlp_only_layers
        else _reference_moe(normalized, state, config)
    )
    return ((feedforward, residual) if keep_residual else residual + feedforward), present


def _numpy(tensor):
    return tensor_adapters.TorchTensor(tensor.contiguous().cpu()).numpy()


@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", ir.DataType.FLOAT), ("cuda", ir.DataType.FLOAT)],
)
def test_flint_complete_lm_prefill_and_own_cache_decode(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = _config(dtype=dtype)
    module = FlintCausalLMModel(config)
    state = _initialize(module, dtype)
    state = {name: value.to(device) for name, value in state.items()}
    builder, op, graph = create_test_builder()
    tokens = create_test_input(builder, "input_ids", [1, "query"], ir.DataType.INT64)
    mask = create_test_input(builder, "attention_mask", [1, "total"], ir.DataType.INT64)
    positions = create_test_input(builder, "position_ids", [1, "query"], ir.DataType.INT64)
    cache_shape = [1, config.num_key_value_heads, "past", config.head_dim]
    pasts = [
        tuple(
            create_test_input(builder, f"past.{index}.{kind}", cache_shape, dtype)
            for kind in ("key", "value")
        )
        for index in range(config.num_hidden_layers)
    ]
    with build_context(get_ep(device), dtype):
        logits, presents = module(op, tokens, mask, positions, pasts)
    logits.name = "logits"
    graph.outputs.append(logits)
    for index, present in enumerate(presents):
        for kind, value in zip(("key", "value"), present, strict=True):
            value.name = f"present.{index}.{kind}"
            graph.outputs.append(value)
    graph.opset_imports["com.microsoft"] = 1
    model = ir.Model(graph, ir_version=10)
    assert model.graph.initializers["lm_head.weight"].dtype == ir.DataType.FLOAT
    torch_dtype = tensor_adapters.to_torch_dtype(dtype)
    empty_shape = (1, config.num_key_value_heads, 0, config.head_dim)
    reference_cache = [
        (
            torch.empty(empty_shape, dtype=torch_dtype, device=device),
            torch.empty(empty_shape, dtype=torch_dtype, device=device),
        )
        for _ in range(config.num_hidden_layers)
    ]
    onnx_cache = [tuple(_numpy(value) for value in pair) for pair in reference_cache]
    input_ids = torch.tensor([[3, 4, 5, 6, 7]], device=device)
    tolerance = 1e-2 if dtype == ir.DataType.BFLOAT16 else 1e-5
    with closing(
        OnnxModelSession(
            model,
            device=device,
            intra_op_num_threads=4,
            providers=(
                [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
                if device == "cuda"
                else ["CPUExecutionProvider"]
            ),
        )
    ) as session:
        offset = 0
        for width in (3, 1, 1):
            chunk = input_ids[:, offset : offset + width]
            attention_mask = torch.ones((1, offset + width), dtype=torch.int64, device=device)
            position_ids = torch.arange(offset, offset + width, device=device).unsqueeze(0)
            hidden = functional.embedding(chunk, state["model.embed_tokens.weight"])
            residual = None
            for index in range(config.num_hidden_layers):
                prefix = f"model.layers.{index}."
                layer_state = {
                    "layer." + name.removeprefix(prefix): value
                    for name, value in state.items()
                    if name.startswith(prefix)
                }
                (hidden, residual), reference_cache[index] = _reference_block(
                    hidden,
                    attention_mask,
                    position_ids,
                    reference_cache[index],
                    layer_state,
                    config,
                    index,
                    residual=residual,
                    keep_residual=True,
                )
            normalized = _norm(
                hidden.float() + residual.float(),
                state["model.norm.weight"],
                config.rms_norm_eps,
            ).to(torch_dtype)
            expected = functional.linear(normalized.float(), state["lm_head.weight"])
            feeds = {
                "input_ids": chunk.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
                "position_ids": position_ids.cpu().numpy(),
            }
            for index, pair in enumerate(onnx_cache):
                for kind, value in zip(("key", "value"), pair, strict=True):
                    feeds[f"past.{index}.{kind}"] = value
            actual = session.run(feeds)
            np.testing.assert_allclose(
                actual["logits"],
                expected.cpu().numpy(),
                atol=tolerance,
                rtol=tolerance,
            )
            for index, pair in enumerate(reference_cache):
                onnx_cache[index] = tuple(
                    actual[f"present.{index}.{kind}"] for kind in ("key", "value")
                )
                for kind, expected_cache in zip(("key", "value"), pair, strict=True):
                    np.testing.assert_allclose(
                        actual[f"present.{index}.{kind}"].astype(np.float32),
                        expected_cache.float().cpu().numpy(),
                        atol=tolerance,
                        rtol=tolerance,
                        err_msg=f"layer {index}, {kind}, offset {offset}",
                    )
            offset += width


@pytest.mark.parametrize("default_layer_types", [False, True])
def test_flint_bf16_cuda_graph_preserves_native_contract_and_fp32_logits(default_layer_types):
    config = _config(dtype=ir.DataType.BFLOAT16, head_dim=128)
    if default_layer_types:
        config.layer_types = None
    package = build_from_module(FlintCausalLMModel(config), config, execution_provider="cuda")
    graph = package["model"].graph
    attention = [
        node
        for node in graph
        if node.domain == FLINT_CUDA_DOMAIN and node.op_type == "SinkAttention"
    ]
    expected_windows = [-1, -1, -1] if default_layer_types else [-1, 512, -1]
    assert [node.attributes["window_left"].as_int() for node in attention] == expected_windows
    mask = next(value for value in graph.inputs if value.name == "attention_mask")
    assert all(node.inputs[4] is mask for node in attention)
    assert graph.opset_imports[FLINT_CUDA_DOMAIN] == 1
    assert graph.outputs[0].dtype == ir.DataType.FLOAT
    assert graph.outputs[0].shape is not None
    assert len(graph.outputs[0].shape) == 3


@pytest.mark.parametrize(
    "layer_index", [0, 1, 2], ids=["dense-full", "moe-sliding", "moe-full"]
)
@pytest.mark.parametrize("ep", ["default", "cpu"])
def test_flint_block_prefill_and_own_cache_decode(layer_index, ep):
    config = _config()
    model, state = _build_block(config, layer_index, ep)
    rng = torch.Generator().manual_seed(104)
    hidden = torch.randn(2, 8, config.hidden_size, generator=rng)
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])
    cache_shape = (2, config.num_key_value_heads, 0, config.head_dim)
    reference_cache = (torch.empty(cache_shape), torch.empty(cache_shape))
    onnx_cache = tuple(_numpy(tensor) for tensor in reference_cache)
    with closing(OnnxModelSession(model, intra_op_num_threads=4)) as session:
        offset = 0
        for width in (5, 1, 2):
            if offset:
                mask = torch.cat((mask, torch.ones(2, width, dtype=torch.int64)), dim=1)
            positions = mask.cumsum(-1)[:, -width:] - 1
            positions = positions.clamp_min(0)
            chunk = hidden[:, offset : offset + width]
            expected, reference_cache = _reference_block(
                chunk, mask, positions, reference_cache, state, config, layer_index
            )
            actual = session.run(
                {
                    "hidden": _numpy(chunk),
                    "mask": mask.numpy(),
                    "positions": positions.numpy(),
                    "past_key": onnx_cache[0],
                    "past_value": onnx_cache[1],
                }
            )
            np.testing.assert_allclose(
                actual["output"], expected.numpy(), atol=1e-5, rtol=1e-5
            )
            for name, tensor in zip(("key", "value"), reference_cache, strict=True):
                np.testing.assert_allclose(actual[name], tensor.numpy(), atol=1e-5, rtol=1e-5)
            onnx_cache = actual["key"], actual["value"]
            offset += width


@pytest.mark.parametrize("head_dim", [8, 128])
def test_flint_sliding_window_boundaries(head_dim):
    config = _config(head_dim=head_dim)
    model, state = _build_block(config, 1)
    hidden = torch.randn(
        1, 514, config.hidden_size, generator=torch.Generator().manual_seed(2)
    )
    shape = (1, config.num_key_value_heads, 0, config.head_dim)
    cache = (torch.empty(shape), torch.empty(shape))
    onnx_cache = tuple(_numpy(tensor) for tensor in cache)
    with closing(OnnxModelSession(model, intra_op_num_threads=4)) as session:
        start = 0
        for end in (512, 513, 514):
            chunk = hidden[:, start:end]
            mask = torch.ones(1, end, dtype=torch.int64)
            positions = torch.arange(start, end).unsqueeze(0)
            expected, cache = _reference_block(chunk, mask, positions, cache, state, config, 1)
            actual = session.run(
                {
                    "hidden": _numpy(chunk),
                    "mask": mask.numpy(),
                    "positions": positions.numpy(),
                    "past_key": onnx_cache[0],
                    "past_value": onnx_cache[1],
                }
            )
            np.testing.assert_allclose(
                actual["output"], expected.numpy(), atol=1e-5, rtol=1e-5
            )
            for name, tensor in zip(("key", "value"), cache, strict=True):
                np.testing.assert_allclose(actual[name], tensor.numpy(), atol=1e-5, rtol=1e-5)
            onnx_cache = actual["key"], actual["value"]
            start = end


def test_flint_rotary_tables_keep_fp32_after_body_dtype_conversion():
    config = _config(head_dim=128, dtype=ir.DataType.BFLOAT16)
    module = FlintRotaryEmbedding(config)
    _cast_module_dtype(module, config.dtype)
    builder, op, graph = create_test_builder()
    positions = create_test_input(builder, "positions", [1, 3], ir.DataType.INT64)
    for name, output in zip(("cos", "sin"), module(op, positions), strict=True):
        output.name = name
        graph.outputs.append(output)
    values = torch.tensor([[511, 512, 513]])
    inv = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
    )
    angles = values.float()[..., None] * inv
    with closing(OnnxModelSession(ir.Model(graph, ir_version=10))) as session:
        actual = session.run({"positions": values.numpy()})
    np.testing.assert_array_equal(actual["cos"], angles.cos().numpy(), strict=True)
    np.testing.assert_array_equal(actual["sin"], angles.sin().numpy(), strict=True)


def test_flint_sink_alignment_preserves_real_cache():
    config = _config()
    model, state = _build_block(config, 1)
    attention = next(node for node in model.graph if node.op_type == "Attention")
    for name, value in (
        ("attention_keys", attention.inputs[1]),
        ("attention_bias", attention.inputs[3]),
    ):
        value.name = name
        model.graph.outputs.append(value)
    hidden = torch.randn(
        1, 4, config.hidden_size, generator=torch.Generator().manual_seed(108)
    )
    shape = (1, config.num_key_value_heads, 0, config.head_dim)
    cache = (torch.empty(shape), torch.empty(shape))
    onnx_cache = tuple(_numpy(tensor) for tensor in cache)
    with closing(OnnxModelSession(model, intra_op_num_threads=4)) as session:
        for total in range(1, 5):
            chunk = hidden[:, total - 1 : total]
            mask = torch.ones(1, total, dtype=torch.int64)
            positions = torch.tensor([[total - 1]])
            expected, cache = _reference_block(chunk, mask, positions, cache, state, config, 1)
            actual = session.run(
                {
                    "hidden": _numpy(chunk),
                    "mask": mask.numpy(),
                    "positions": positions.numpy(),
                    "past_key": onnx_cache[0],
                    "past_value": onnx_cache[1],
                }
            )
            keys, bias = actual["attention_keys"], actual["attention_bias"]
            assert keys.shape[1] == bias.shape[-1] == (total // 4 + 1) * 4
            np.testing.assert_array_equal(keys[:, total:], 0.0)
            np.testing.assert_array_equal(
                bias[0, :, 0, total], state["layer.self_attn.softmax_offset"].numpy()
            )
            assert np.isneginf(bias[..., total + 1 :]).all()
            np.testing.assert_allclose(
                actual["output"], expected.numpy(), atol=1e-5, rtol=1e-5
            )
            for name, tensor in zip(("key", "value"), cache, strict=True):
                np.testing.assert_allclose(actual[name], tensor.numpy(), atol=1e-5, rtol=1e-5)
                assert actual[name].shape[2] == total
            onnx_cache = actual["key"], actual["value"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
@pytest.mark.parametrize(
    "layer_index", [0, 1, 2], ids=["dense-full", "moe-sliding", "moe-full"]
)
def test_flint_cuda_blocks(dtype, layer_index, tmp_path):
    config = _config(dtype=dtype)
    model, state = _build_block(config, layer_index, "cuda")
    torch_dtype = tensor_adapters.to_torch_dtype(dtype)
    hidden = torch.randn(
        2, 7, config.hidden_size, generator=torch.Generator().manual_seed(81)
    ).to(torch_dtype)
    shape = (2, config.num_key_value_heads, 0, config.head_dim)
    reference_cache = (
        torch.empty(shape, dtype=torch_dtype),
        torch.empty(shape, dtype=torch_dtype),
    )
    onnx_cache = tuple(_numpy(tensor) for tensor in reference_cache)
    with closing(
        OnnxModelSession(
            model,
            device="cuda",
            intra_op_num_threads=4,
            enable_profiling=True,
            profile_file_prefix=str(tmp_path / "flint"),
        )
    ) as session:
        start = 0
        mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
        for end in (4, 5, 7):
            if start:
                mask = torch.cat((mask, torch.ones(2, end - start, dtype=torch.int64)), dim=1)
            positions = (mask.cumsum(-1)[:, -(end - start) :] - 1).clamp_min(0)
            chunk = hidden[:, start:end]
            expected, reference_cache = _reference_block(
                chunk, mask, positions, reference_cache, state, config, layer_index
            )
            actual = session.run(
                {
                    "hidden": _numpy(chunk),
                    "mask": mask.numpy(),
                    "positions": positions.numpy(),
                    "past_key": onnx_cache[0],
                    "past_value": onnx_cache[1],
                }
            )
            np.testing.assert_allclose(
                actual["output"].astype(np.float32),
                expected.float().numpy(),
                atol=1e-2,
                rtol=1e-2,
            )
            for name, tensor in zip(("key", "value"), reference_cache, strict=True):
                np.testing.assert_allclose(
                    actual[name].astype(np.float32),
                    tensor.float().numpy(),
                    atol=1e-2,
                    rtol=1e-2,
                )
            onnx_cache = actual["key"], actual["value"]
            start = end
        events = json.loads(Path(session._session.end_profiling()).read_text())
    critical = [
        event["args"]
        for event in events
        if event.get("args", {}).get("op_name") in {"Attention", "MoE", "MatMul", "Gemm"}
        and "provider" in event.get("args", {})
    ]
    assert critical
    required_ops = {"Attention", "MatMul", "Gemm"}
    if layer_index not in config.mlp_only_layers:
        required_ops.add("MoE")
    assert required_ops <= {event["op_name"] for event in critical}
    assert {event["provider"] for event in critical} == {"CUDAExecutionProvider"}


def test_flint_router_preserves_small_sigmoid_weights():
    config = _config(hidden_size=1, num_local_experts=3, num_experts_per_tok=2)
    gate = FlintGate(config)
    gate.weight.const_value = ir.tensor(np.array([[-20.0], [-21.0], [-22.0]], np.float32))
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 1])
    for name, output in zip(("weights", "indices"), gate(op, x), strict=True):
        output.name = name
        graph.outputs.append(output)
    with closing(
        OnnxModelSession(ir.Model(graph, ir_version=10), intra_op_num_threads=4)
    ) as session:
        actual = session.run({"x": np.ones((1, 1), np.float32)})
    expected = torch.tensor([[-20.0, -21.0]]).sigmoid()
    expected /= expected.sum(-1, keepdim=True) + 1e-20
    np.testing.assert_allclose(actual["weights"], expected.numpy(), atol=1e-7)
    np.testing.assert_array_equal(actual["indices"], [[0, 1]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_flint_cuda_down_projection_preserves_source_gemm_layout():
    module = FlintCUDALinear(6144, 2048, bias=False)
    state = _initialize(module, ir.DataType.BFLOAT16)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, "query", 6144], ir.DataType.BFLOAT16)
    with build_context(get_ep("cuda"), ir.DataType.BFLOAT16):
        output = module(op, hidden)
    output.name = "output"
    graph.outputs.append(output)
    rng = torch.Generator().manual_seed(81)
    weight = state["weight"].cuda()
    with closing(
        OnnxModelSession(ir.Model(graph, ir_version=10), device="cuda", intra_op_num_threads=4)
    ) as session:
        for query in (16, 1):
            x = torch.randn((1, query, 6144), generator=rng).bfloat16()
            expected = functional.linear(x.cuda(), weight).cpu()
            actual = session.run({"hidden": _numpy(x)})["output"]
            np.testing.assert_array_equal(actual.astype(np.float32), expected.float().numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_flint_mlp_rounds_only_after_fused_swiglu():
    config = _config(hidden_size=4, intermediate_size=4, dtype=ir.DataType.BFLOAT16)
    module = FlintMLP(config)
    _cast_module_dtype(module, config.dtype)
    for _name, parameter in module.named_parameters():
        parameter.const_value = tensor_adapters.TorchTensor(torch.eye(4).bfloat16())
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 1, 4], config.dtype)
    result = module(op, hidden)
    result.name = "output"
    graph.outputs.append(result)
    x = torch.tensor([[[-3.1875, -0.515625, 0.7890625, 2.75]]]).bfloat16()
    with closing(OnnxModelSession(ir.Model(graph, ir_version=10), device="cuda")) as session:
        actual = session.run({"hidden": _numpy(x)})["output"]
    expected = (functional.silu(x.float()) * x.float()).to(x.dtype)
    assert not torch.equal(expected, functional.silu(x) * x)
    np.testing.assert_array_equal(actual.astype(np.float32), expected.float().numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_flint_residual_norm_uses_unrounded_sum():
    module = FlintResidualRMSNorm(4, eps=1e-6)
    module.weight.const_value = tensor_adapters.TorchTensor(torch.ones(4).bfloat16())
    module.weight.type = ir.TensorType(ir.DataType.BFLOAT16)
    builder, op, graph = create_test_builder()
    x_input = create_test_input(builder, "x", [1, 4], ir.DataType.BFLOAT16)
    residual_input = create_test_input(builder, "residual", [1, 4], ir.DataType.BFLOAT16)
    for name, value in zip(
        ("normalized", "residual"), module(op, x_input, residual_input), strict=True
    ):
        value.name = f"output_{name}"
        graph.outputs.append(value)
    x = torch.tensor([[1.0, 0.25, -0.5, 2.0]]).bfloat16()
    residual = torch.tensor([[0.00390625, 0.001953125, 0.00390625, 0.0078125]]).bfloat16()
    combined = x.float() + residual.float()
    expected = _norm(combined, torch.ones(4), 1e-6).bfloat16()
    rounded = _norm(x + residual, torch.ones(4), 1e-6)
    assert not torch.equal(expected, rounded)
    with closing(OnnxModelSession(ir.Model(graph, ir_version=10), device="cuda")) as session:
        actual = session.run({"x": _numpy(x), "residual": _numpy(residual)})
    np.testing.assert_array_equal(
        actual["output_normalized"].astype(np.float32), expected.float().numpy()
    )
    np.testing.assert_array_equal(
        actual["output_residual"].astype(np.float32), combined.bfloat16().float().numpy()
    )


def test_flint_moe_refuses_per_expert_expansion():
    builder, op, _graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 2, 32])
    with (
        build_context(
            EpCapabilities(name="unfused", supports_fused_moe=False), ir.DataType.FLOAT
        ),
        pytest.raises(NotImplementedError, match="per-expert graph expansion"),
    ):
        FlintMoE(_config())(op, x)


def test_flint_moe_graph_scales_with_topk_not_expert_count():
    histograms = []
    for expert_count in (8, 256):
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 4, 32])
        module = FlintMoE(_config(num_local_experts=expert_count, num_experts_per_tok=6))
        graph.outputs.append(module(op, hidden))
        histogram = {}
        for node in graph:
            histogram[node.op_type] = histogram.get(node.op_type, 0) + 1
        histograms.append(histogram)
    assert histograms[0] == histograms[1]
    assert histograms[1]["MoE"] == 6


def test_flint_raw_config_semantics():
    raw = dict(
        model_type="flint",
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=3,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=150144,
        max_position_embeddings=262144,
        hidden_act="silu",
        torch_dtype="bfloat16",
        rope_theta=10000.0,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0, 2],
        mlp_only_layers=[0],
        sliding_window=513,
        num_experts=256,
        num_experts_per_tok=6,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=1024,
    )
    config = FlintConfig.from_transformers(raw)
    config.validate()
    assert config.dtype == ir.DataType.BFLOAT16
    assert config.no_rope_layers == [0, 2]
    assert config.mlp_only_layers == (0,)
    assert config.attn_qk_norm
    assert config.head_dim * config.num_attention_heads == 4096
    assert config.num_local_experts == 256
    assert config.rope_theta == pytest.approx(10000.0)
    disabled = FlintConfig.from_transformers({**raw, "use_sliding_window": False})
    assert disabled.sliding_window is None


def test_flint_raw_config_preserves_defaults_and_nested_rope():
    raw = dict(
        model_type="flint",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        rope_parameters={"rope_type": "default", "rope_theta": 25000.0},
    )
    config = FlintConfig.from_transformers(raw)
    config.validate()
    assert config.num_experts_per_tok == 6
    assert config.moe_intermediate_size == 512
    assert config.shared_expert_intermediate_size == 1024
    assert config.mlp_only_layers == ()
    assert config.layer_types == ["full_attention"] * 3
    assert config.rope_theta == pytest.approx(25000.0)
    assert config.hidden_act == "silu"
    disabled = FlintConfig.from_transformers(
        {**raw, "shared_expert_intermediate_size": 0, "mlp_only_layers": []}
    )
    disabled.validate()
    assert disabled.shared_expert_intermediate_size == 0
    with pytest.raises(ValueError, match="one layer_types"):
        FlintConfig.from_transformers({**raw, "layer_types": []}).validate()
    with pytest.raises(ValueError, match="full-head default RoPE"):
        FlintConfig.from_transformers({**raw, "partial_rotary_factor": 0.0}).validate()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"layer_types": ["full_attention"]}, "one layer_types"),
        ({"no_rope_layers": [3]}, "valid layer indices"),
        ({"mlp_only_layers": (0, 0)}, "distinct valid"),
        ({"num_experts_per_tok": 5}, "valid top-k"),
        ({"sliding_window": 0}, "positive window"),
        ({"router_score_function": "softmax"}, "sigmoid routing"),
        ({"norm_topk_prob": False}, "normalizes top-k"),
        ({"partial_rotary_factor": 0.5}, "full-head default RoPE"),
        ({"rope_type": "linear"}, "full-head default RoPE"),
    ],
)
def test_flint_rejects_unsupported_semantics(overrides, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(_config(), **overrides).validate()
