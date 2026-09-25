# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Opt-in, synthetic full-logits CUDA parity for Olive mixed-width Qwen3 MoE."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
import torch
import torch.nn.functional as functional

from mobius._configs import QuantizationConfig, QuantizationOverride
from mobius._testing import make_config
from mobius.models.moe import MoECausalLMModel

_H = _INTER = 64
_BLOCK = 32
_EXPERTS = 4
_VOCAB = 32
_ORT_COMMIT = "ee5f6e7"
_DATA_ROOT = Path("/datadisks/disk5/titaiwang")


def _packed_weights(
    rng: np.random.Generator, bits: int, rows: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Olive K-last format: low nibble precedes high nibble; no zero-point
    # sidecar means the implicit midpoint (8 for INT4, 128 for INT8).
    codes = (
        rng.integers(4, 12, (_EXPERTS, rows, k), dtype=np.uint8)
        if bits == 4
        else (rng.integers(120, 136, (_EXPERTS, rows, k), dtype=np.uint8))
    )
    packed = codes[..., ::2] | (codes[..., 1::2] << 4) if bits == 4 else codes
    scales = rng.uniform(0.012, 0.035, (_EXPERTS, rows, k // _BLOCK)).astype(np.float16)
    return torch.from_numpy(packed.copy()), torch.from_numpy(scales)


def _dequantize(weight: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """Independent symmetric affine dequantization of Olive's K-last bytes."""
    codes = weight.to(torch.int32)
    if bits == 4:
        codes = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2)
    return (codes - (1 << (bits - 1))).float() * scale.float().repeat_interleave(
        _BLOCK, dim=-1
    )


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)


def _reference(
    input_ids: np.ndarray,
    state: dict[str, torch.Tensor],
    layouts: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Dense Qwen pre-norm residual blocks; attention is explicitly zero.

    Unlike QMoE this executes each selected expert's dequantized gate, up and
    down matmuls separately, including sigmoid-SwiGLU and top-2 routing.
    """
    hidden = state["model.embed_tokens.weight"][torch.from_numpy(input_ids)].float()
    for layer, (fc1_bits, fc2_bits) in enumerate(layouts):
        p = f"model.layers.{layer}.mlp."
        x = _rms(hidden)  # zero attention projection leaves the residual intact
        router = x @ state[p + "gate.weight"].float().T
        indices = router.topk(2, dim=-1).indices
        probabilities = router.softmax(dim=-1).gather(-1, indices)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        gate_up = _dequantize(
            state[p + "experts.gate_up_proj_qweight"],
            state[p + "experts.gate_up_proj_scales"],
            fc1_bits,
        )
        down = _dequantize(
            state[p + "experts.down_proj_qweight"],
            state[p + "experts.down_proj_scales"],
            fc2_bits,
        )
        routed = torch.zeros_like(x)
        for slot in range(2):
            for expert in range(_EXPERTS):
                # The unfused dense graph runs all experts then masks their
                # outputs; doing so here also tests multi-token routing.
                gate = x @ gate_up[expert, :_INTER].T
                up = x @ gate_up[expert, _INTER:].T
                value = (functional.silu(gate) * up) @ down[expert].T
                routed += (
                    value
                    * (indices[..., slot] == expert).unsqueeze(-1)
                    * (probabilities[..., slot].unsqueeze(-1))
                )
        hidden = hidden + routed
    logits = _rms(hidden) @ state["lm_head.weight"].float().T
    return logits.numpy()


def _checkpoint(layouts: tuple[tuple[int, int], ...]) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(744)
    state = {
        "model.embed_tokens.weight": torch.from_numpy(
            rng.normal(0, 0.65, (_VOCAB, _H)).astype(np.float16)
        ),
        "model.norm.weight": torch.ones(_H, dtype=torch.float16),
        "lm_head.weight": torch.from_numpy(
            rng.normal(0, 0.16, (_VOCAB, _H)).astype(np.float16)
        ),
    }
    for layer, (fc1_bits, fc2_bits) in enumerate(layouts):
        prefix = f"model.layers.{layer}."
        expert_prefix = prefix + "mlp.experts."
        for projection, bits, rows, k in (
            ("gate_up_proj", fc1_bits, 2 * _INTER, _H),
            ("down_proj", fc2_bits, _H, _INTER),
        ):
            weight, scales = _packed_weights(rng, bits, rows, k)
            state[expert_prefix + projection + "_qweight"] = weight
            state[expert_prefix + projection + "_scales"] = scales
        # An unquantized router is the Olive Qwen3-MoE convention.
        state[prefix + "mlp.gate.weight"] = torch.from_numpy(
            rng.normal(0, 0.13, (_EXPERTS, _H)).astype(np.float16)
        )
        for norm in ("input_layernorm", "post_attention_layernorm"):
            state[prefix + norm + ".weight"] = torch.ones(_H, dtype=torch.float16)
        for proj, rows in (("q_proj", _H), ("k_proj", 32), ("v_proj", 32), ("o_proj", _H)):
            base = prefix + "self_attn." + proj + ".weight_"
            # Symmetric INT4 midpoint: zero attention, without leaving any
            # initializer unset or introducing attention numerical noise.
            state[base + "qweight"] = torch.full((rows, _H // 2), 0x88, dtype=torch.uint8)
            state[base + "scales"] = torch.ones(rows, _H // _BLOCK, dtype=torch.float16)
    return state


@pytest.mark.integration
@pytest.mark.parametrize(
    "layouts",
    [((4, 8), (8, 4)), ((8, 8),)],
    ids=["adjacent-4x8-8x4", "uniform-8x8"],
)
def test_olive_qwen3_moe_full_logits_on_cuda(layouts):
    """Run the exported model, not a standalone hand-written QMoE node."""
    if os.environ.get("MOBIUS_QMOE_CUDA_TEST") != "1":
        pytest.skip("set MOBIUS_QMOE_CUDA_TEST=1 with the ee5f6e7c ORT CUDA wheel")

    import onnxruntime as ort

    assert f"git-commit-id={_ORT_COMMIT}" in ort.get_build_info(), (
        f"ORT CUDA wheel from {_ORT_COMMIT} required, got {ort.get_build_info()}"
    )
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    assert _DATA_ROOT.is_dir(), f"model/profiling output directory missing: {_DATA_ROOT}"

    overrides = {
        f"model.layers.{layer}.mlp.experts.{projection}": QuantizationOverride(bits=bits)
        for layer, pair in enumerate(layouts)
        for projection, bits in zip(("gate_up_proj", "down_proj"), pair)
        if bits != 4
    }
    config = make_config(
        dtype=ir.DataType.FLOAT16,
        hidden_size=_H,
        intermediate_size=_INTER,
        moe_intermediate_size=_INTER,
        num_hidden_layers=len(layouts),
        num_local_experts=_EXPERTS,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=_VOCAB,
        quantization=QuantizationConfig(
            bits=4,
            group_size=_BLOCK,
            quant_method="olive",
            sym=True,
            overrides=overrides,
        ),
    )
    state = _checkpoint(layouts)
    module = MoECausalLMModel(config)
    assert all(layer.mlp.experts is None for layer in module.model.layers)
    processed = module.preprocess_weights(state.copy())
    # Module plans defer ordinary packed linears to the component loader. Do
    # that conversion explicitly here; routed experts were already consumed
    # by the QMoE preprocessor and must not be reshaped as MatMulNBits weights.
    from mobius._weight_utils import preprocess_olive_weights

    attention_sidecars = {
        key: processed.pop(key)
        for key in list(processed)
        if ".self_attn." in key and key.endswith(("_qweight", "_scales"))
    }
    processed.update(preprocess_olive_weights(attention_sidecars, bits=4, group_size=_BLOCK))
    from mobius import build_from_module

    package = build_from_module(module, config)
    package.apply_weights(processed, fold_constants=False)
    model = package["model"]
    qmoes = [node for node in model.graph if node.op_type == "QMoE"]
    assert len(qmoes) == len(layouts)
    for layer, (node, (fc1, fc2)) in enumerate(zip(qmoes, layouts)):
        assert node.domain == "com.microsoft"
        assert node.attributes["weights_prepacked"].value == 0
        assert node.attributes["block_size"].value == _BLOCK
        assert _BLOCK & (_BLOCK - 1) == 0 and 16 <= _BLOCK <= 256
        assert _H % _BLOCK == _INTER % _BLOCK == 0
        assert node.attributes["quant_type"].value == "int"
        assert node.attributes["activation_type"].value == "swiglu"
        assert node.attributes["swiglu_fusion"].value == 1
        # ee5f6e7c's mixed CUDA fallback requires 3-D scales matching the
        # FP16/BF16 activation type, and raw uint8-packed FC1/FC2 weights.
        assert node.inputs[0].dtype == ir.DataType.FLOAT16
        assert node.inputs[2].dtype == node.inputs[5].dtype == ir.DataType.UINT8
        assert node.inputs[3].dtype == node.inputs[6].dtype == node.inputs[0].dtype
        for label in ("fc1", "fc2"):
            prefix = f"model.layers.{layer}.mlp.{label}_scales"
            assert processed[prefix].ndim == 3
            assert processed[prefix].dtype == torch.float16
        assert node.attributes["expert_weight_bits"].value == (4 if fc1 != fc2 else 8)
        if fc1 != fc2:
            assert tuple(
                node.attributes[f"fc{i}_expert_weight_bits"].value for i in (1, 2, 3)
            ) == (fc1, fc2, fc1)
    assert all(init.const_value is not None for init in model.graph.initializers.values())

    input_ids = np.array([[1, 9, 3]], dtype=np.int64)
    expected = _reference(input_ids, state, layouts)
    # Catch accidentally vacuous checkpoints even if the CUDA path regresses
    # to returning only the embedding residual.
    residual_logits = (
        _rms(state["model.embed_tokens.weight"][torch.from_numpy(input_ids)].float())
        @ state["lm_head.weight"].float().T
    ).numpy()
    assert np.max(np.abs(expected - residual_logits)) > 0.02
    swapped = state.copy()
    gate_up_key = "model.layers.0.mlp.experts.gate_up_proj_qweight"
    swapped[gate_up_key] = state[gate_up_key].roll(_INTER, dims=1)
    assert np.max(np.abs(expected - _reference(input_ids, swapped, layouts))) > 0.02

    with tempfile.TemporaryDirectory(prefix="mobius-qmoe-", dir=_DATA_ROOT) as directory:
        directory = Path(directory)
        model_path = directory / "model.onnx"
        ir.save(model, model_path, external_data="model.onnx.data")
        options = ort.SessionOptions()
        # The full decoder has CPU shape/control nodes; profile the QMoE nodes
        # rather than rejecting their legitimate host-side shape work.
        options.enable_profiling = True
        options.profile_file_prefix = str(directory / "profile")
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CUDAExecutionProvider"]
        )
        assert session.get_providers()[0] == "CUDAExecutionProvider"
        feeds = {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids),
            "position_ids": np.arange(input_ids.shape[1], dtype=np.int64)[None, :],
        }
        for layer in range(len(layouts)):
            for cache in ("key", "value"):
                feeds[f"past_key_values.{layer}.{cache}"] = np.zeros(
                    (1, 2, 0, 16), dtype=np.float16
                )
        actual = session.run(["logits"], feeds)[0]
        profile = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
        qmoe_events = [
            event
            for event in profile
            if event.get("cat") == "Node"
            and (
                event.get("args", {}).get("op_name") == "QMoE"
                or "QMoE" in event.get("name", "")
            )
        ]
        assert len(qmoe_events) >= len(layouts), "QMoE execution missing from ORT profile"
        assert all(
            event.get("args", {}).get("provider") == "CUDAExecutionProvider"
            for event in qmoe_events
        ), qmoe_events
    np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=0.01, atol=0.01)
