# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exercise the Ornith loader with small, local safetensors checkpoints."""

import json

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnxscript import nn
from ornith_export import (
    REVISION,
    _source_name,
    expose_mtp_seed,
    load_weights,
    validate_checkpoint_revision,
)
from safetensors.torch import save_file

from mobius._builder import _cast_module_dtype
from mobius._configs import QuantizationConfig
from mobius._model_package import ModelPackage
from mobius._testing import create_test_builder, create_test_input, make_config
from mobius.components import QuantizedLinear
from mobius.components._fp8 import Fp8Linear, Fp8QwenMoEBlock
from mobius.components._moe import MoELayer


def _checkpoint(directory, tensors):
    save_file(tensors, directory / "weights.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "weights.safetensors")})
    )
    return directory


@pytest.mark.parametrize(
    "name,expected",
    [
        ("decoder.model.embed_tokens.weight", "model.language_model.embed_tokens.weight"),
        ("decoder.lm_head.weight", "lm_head.weight"),
        ("vision_encoder.visual.pos_embed", "model.visual.pos_embed.weight"),
        ("vision_encoder.visual.patch_embed.weight", "model.visual.patch_embed.proj.weight"),
        (
            "vision_encoder.visual.blocks.0.mlp.up_proj.weight",
            "model.visual.blocks.0.mlp.linear_fc1.weight",
        ),
        ("mtp.embed_tokens.weight", "model.language_model.embed_tokens.weight"),
        ("mtp.lm_head.weight", "lm_head.weight"),
        ("mtp.layers.0.mlp.gate.weight", "mtp.layers.0.mlp.gate.weight"),
    ],
)
def test_source_names(name, expected):
    assert _source_name(name) == expected


def test_revision_validation_checks_every_shard(tmp_path):
    _checkpoint(tmp_path, {"projection.weight": torch.ones(1)})
    metadata = tmp_path / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    for name in ("config.json", "model.safetensors.index.json", "weights.safetensors"):
        (metadata / f"{name}.metadata").write_text(f"{REVISION}\netag\n0\n")
    validate_checkpoint_revision(tmp_path)
    shard_metadata = metadata / "weights.safetensors.metadata"
    shard_metadata.write_text("different-revision\netag\n0\n")
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_checkpoint_revision(tmp_path)
    shard_metadata.unlink()
    with pytest.raises(ValueError, match="Missing pinned"):
        validate_checkpoint_revision(tmp_path)


def test_mtp_seed_exposes_lm_head_input():
    config = make_config(hidden_size=128, vocab_size=256, dtype=ir.DataType.FLOAT16)
    layer = Fp8Linear(128, 256)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 128], ir.DataType.FLOAT16)
    logits = layer(op, hidden)
    builder.add_output(logits, "logits")
    package = ModelPackage({"decoder": ir.Model(graph, ir_version=10)}, config=config)
    expose_mtp_seed(package, config)
    seed = graph.outputs[-1]
    assert seed.name == "mtp_seed"
    assert seed.shape[-1] == 128
    assert seed.producer().inputs[0] is hidden
    with pytest.raises(ValueError, match="already exposes"):
        expose_mtp_seed(package, config)


def test_int2_loader_preserves_zero_and_subnormal_blocks(tmp_path):
    module = nn.Module()
    module.projection = QuantizedLinear(128, 3, bits=2, block_size=128)
    weight = torch.stack(
        [torch.zeros(128), torch.full((128,), 1e-22), torch.linspace(-0.1, 0.1, 128)]
    )
    load_weights(
        module,
        _checkpoint(tmp_path, {"projection.weight": weight}),
        prefix="",
        device="cpu",
    )
    scales = module.projection.scales.const_value.raw
    packed = module.projection.weight.const_value.raw.reshape(3, 32)
    assert scales.dtype == torch.float16
    assert torch.isfinite(scales).all() and torch.all(scales > 0)
    codes = torch.stack([(packed >> shift) & 3 for shift in (0, 2, 4, 6)], -1)
    reconstructed = (codes.reshape(3, 128).float() - 2) * scales.float()
    assert torch.count_nonzero(reconstructed[:2]) == 0
    assert torch.max(torch.abs(reconstructed[2] - weight[2])) < 0.034


def _expert_module(fp8):
    config = make_config(
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        quantization=None
        if fp8
        else QuantizationConfig(bits=2, group_size=128, quant_method="olive"),
    )
    module = nn.Module()
    module.mlp = Fp8QwenMoEBlock(config) if fp8 else MoELayer(config)
    return module


def _expert_weights(module, fused):
    tensors = {}
    for name, child in module.named_modules():
        if isinstance(child, Fp8Linear):
            tensors[f"mtp.{name}.weight"] = torch.randn(*child.weight.shape) * 0.02
        else:
            for key, parameter in child.named_parameters(recurse=False):
                if not key.startswith(("fc1_", "fc2_")) and key != "scales":
                    tensors[f"mtp.{name}.{key}"] = torch.randn(*parameter.shape) * 0.02
    gate_up = torch.randn(4, 256, 128) * 0.02
    down = torch.randn(4, 128, 128) * 0.02
    if fused:
        tensors["mtp.mlp.experts.gate_up_proj"] = gate_up
        tensors["mtp.mlp.experts.down_proj"] = down
    else:
        for index in range(4):
            stem = f"mtp.mlp.experts.{index}"
            tensors[f"{stem}.gate_proj.weight"] = gate_up[index, :128].contiguous()
            tensors[f"{stem}.up_proj.weight"] = gate_up[index, 128:].contiguous()
            tensors[f"{stem}.down_proj.weight"] = down[index]
    return tensors


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_expert_loader_accepts_decoder_and_mtp_layouts(tmp_path, fused, fp8):
    module = _expert_module(fp8)
    tensors = _expert_weights(module, fused)
    load_weights(module, _checkpoint(tmp_path, tensors), prefix="mtp", device="cpu")
    assert all(parameter.const_value is not None for _, parameter in module.named_parameters())
    assert module.mlp.fc1_experts_weights.const_value.raw.dtype == (
        torch.float8_e4m3fn if fp8 else torch.uint8
    )
    assert module.mlp.fc1_scales.const_value.raw.dtype == (
        torch.float32 if fp8 else torch.float16
    )
    assert module.mlp.fc1_experts_weights.dtype == (
        ir.DataType.FLOAT8E4M3FN if fp8 else ir.DataType.UINT8
    )


def _dequantize(weight, scales, fp8):
    if fp8:
        expanded = scales.repeat_interleave(128, -2).repeat_interleave(128, -1)
        return weight.float() * expanded[..., : weight.shape[-2], : weight.shape[-1]]
    codes = torch.stack([(weight >> shift) & 3 for shift in (0, 2, 4, 6)], -1)
    codes = codes.reshape(*weight.shape[:-1], weight.shape[-1] * 4)
    return (codes.float() - 2) * scales.float().repeat_interleave(128, -1)


@pytest.mark.integration
@pytest.mark.parametrize("tokens", [1, 8])
@pytest.mark.parametrize("fp8", [False, True])
def test_quantized_experts_cuda_match_reference(tmp_path, tokens, fp8):
    import onnxruntime as ort

    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    ort.preload_dlls()
    torch.manual_seed(17)
    module = _expert_module(fp8)
    _cast_module_dtype(module, ir.DataType.FLOAT16)
    load_weights(
        module,
        _checkpoint(tmp_path, _expert_weights(module, fused=True)),
        prefix="mtp",
        device="cpu",
    )
    layer = module.mlp
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, tokens, 128], ir.DataType.FLOAT16)
    output = layer(op, hidden)
    output.type = ir.TensorType(ir.DataType.FLOAT16)
    output.shape = ir.Shape([1, tokens, 128])
    builder.add_output(output, "output")
    graph.opset_imports["com.microsoft"] = 1
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        ir.serde.serialize_model(ir.Model(graph, ir_version=10)).SerializeToString(),
        sess_options=options,
        providers=["CUDAExecutionProvider"],
    )
    x = torch.randn(1, tokens, 128).half()
    actual = session.run(None, {"hidden": x.numpy()})[0]
    fc1 = _dequantize(
        layer.fc1_experts_weights.const_value.raw,
        layer.fc1_scales.const_value.raw,
        fp8,
    )
    fc2 = _dequantize(
        layer.fc2_experts_weights.const_value.raw,
        layer.fc2_scales.const_value.raw,
        fp8,
    )
    scores = torch.nn.functional.linear(x.float(), layer.gate.weight.const_value.raw.float())
    probabilities, experts = scores.softmax(-1).topk(2, dim=-1)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(4):
        gate, up = torch.nn.functional.linear(x.float(), fc1[expert]).chunk(2, -1)
        values = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, fc2[expert])
        routing = ((experts == expert).float() * probabilities).sum(-1, keepdim=True)
        expected += values * routing
    if fp8:

        def linear(projection, value):
            weight = projection.weight.const_value.raw.float()
            scales = projection.scales.const_value.raw.repeat_interleave(128, -1)
            return torch.nn.functional.linear(value, weight * scales)

        shared = layer.shared_expert
        gate = torch.nn.functional.silu(linear(shared.gate_proj, x.float()))
        values = linear(shared.down_proj, gate * linear(shared.up_proj, x.float()))
        mixing = torch.nn.functional.linear(
            x.float(), layer.shared_expert_gate.weight.const_value.raw.float()
        ).sigmoid()
        expected += values * mixing
    np.testing.assert_allclose(actual, expected.numpy(), rtol=0.03, atol=0.0001)
