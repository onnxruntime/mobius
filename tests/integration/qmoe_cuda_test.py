# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic CUDA parity and path qualification for mixed-width Qwen3 MoE.

Install the pinned CUDA 12 wheel from requirements/ci/qmoe-cuda.txt and set
MOBIUS_QMOE_CUDA_TEST=1 to run the network-free synthetic tests. INT2 decode
excludes dense weight scratch; prefill deliberately disables native INT2
prefill to qualify the bounded dense fallback. Real-slice experiments remain
separately opt-in and require their original custom wheel and local data.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
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
_SYNTHETIC_ORT_VERSION = "1.31.0.dev20261008002"
_SYNTHETIC_ORT_COMMIT = "b131c79"
_SCRATCH_KEY = "ep.cuda.qmoe_int_dequant_max_scratch_bytes"
_FALLBACK_SCRATCH_BYTES = 8 * 1024 * 1024
_DATA_ROOT = Path("/datadisks/disk5/titaiwang")


def _packed_weights(
    rng: np.random.Generator, bits: int, rows: int, k: int, group_size: int = _BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    # Independent K-last fixture: low-bit codes come first within each byte.
    if bits == 2:
        codes = rng.integers(0, 4, (_EXPERTS, rows, k), dtype=np.uint8)
        packed = (
            codes[..., 0::4]
            | (codes[..., 1::4] << 2)
            | (codes[..., 2::4] << 4)
            | (codes[..., 3::4] << 6)
        )
    elif bits == 4:
        codes = rng.integers(4, 12, (_EXPERTS, rows, k), dtype=np.uint8)
        packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    elif bits == 8:
        packed = rng.integers(120, 136, (_EXPERTS, rows, k), dtype=np.uint8)
    else:
        raise ValueError(f"unsupported synthetic weight width: {bits}")
    scales = (
        rng.uniform(0.012, 0.035, (_EXPERTS, rows, k // group_size)) * np.sqrt(_H / k)
    ).astype(np.float16)
    return torch.from_numpy(packed.copy()), torch.from_numpy(scales)


def _dequantize(
    weight: torch.Tensor, scale: torch.Tensor, bits: int, group_size: int = _BLOCK
) -> torch.Tensor:
    """Independent symmetric affine dequantization of Olive's K-last bytes."""
    codes = weight.to(torch.int32)
    if bits == 2:
        codes = torch.stack(tuple((codes >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1)
        codes = codes.flatten(-2)
    elif bits == 4:
        codes = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2)
    elif bits != 8:
        raise ValueError(f"unsupported synthetic weight width: {bits}")
    return (codes - (1 << (bits - 1))).float() * scale.float().repeat_interleave(
        group_size, dim=-1
    )


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)


def _reference(
    input_ids: np.ndarray,
    state: dict[str, torch.Tensor],
    layouts: tuple[tuple[int, int], ...],
    group_size: int = _BLOCK,
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
            group_size,
        )
        down = _dequantize(
            state[p + "experts.down_proj_qweight"],
            state[p + "experts.down_proj_scales"],
            fc2_bits,
            group_size,
        )
        routed = torch.zeros_like(x)
        for slot in range(2):
            for expert in range(_EXPERTS):
                # The unfused dense graph runs all experts then masks their
                # outputs; doing so here also tests multi-token routing.
                intermediate_size = gate_up.shape[1] // 2
                gate = x @ gate_up[expert, :intermediate_size].T
                up = x @ gate_up[expert, intermediate_size:].T
                value = (functional.silu(gate) * up) @ down[expert].T
                routed += (
                    value
                    * (indices[..., slot] == expert).unsqueeze(-1)
                    * (probabilities[..., slot].unsqueeze(-1))
                )
        hidden = hidden + routed
    logits = _rms(hidden) @ state["lm_head.weight"].float().T
    return logits.numpy()


def _checkpoint(
    layouts: tuple[tuple[int, int], ...],
    hidden_size: int = _H,
    intermediate_size: int = _INTER,
    group_size: int = _BLOCK,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(744)
    state = {
        "model.embed_tokens.weight": torch.from_numpy(
            rng.normal(0, 0.65, (_VOCAB, hidden_size)).astype(np.float16)
        ),
        "model.norm.weight": torch.ones(hidden_size, dtype=torch.float16),
        "lm_head.weight": torch.from_numpy(
            rng.normal(0, 0.16, (_VOCAB, hidden_size)).astype(np.float16)
        ),
    }
    for layer, (fc1_bits, fc2_bits) in enumerate(layouts):
        prefix = f"model.layers.{layer}."
        expert_prefix = prefix + "mlp.experts."
        for projection, bits, rows, k in (
            ("gate_up_proj", fc1_bits, 2 * intermediate_size, hidden_size),
            ("down_proj", fc2_bits, hidden_size, intermediate_size),
        ):
            weight, scales = _packed_weights(rng, bits, rows, k, group_size)
            state[expert_prefix + projection + "_qweight"] = weight
            state[expert_prefix + projection + "_scales"] = scales
        # An unquantized router is the Olive Qwen3-MoE convention.
        state[prefix + "mlp.gate.weight"] = torch.from_numpy(
            rng.normal(0, 0.13, (_EXPERTS, hidden_size)).astype(np.float16)
        )
        for norm in ("input_layernorm", "post_attention_layernorm"):
            state[prefix + norm + ".weight"] = torch.ones(hidden_size, dtype=torch.float16)
        for proj, rows in (
            ("q_proj", hidden_size),
            ("k_proj", hidden_size // 2),
            ("v_proj", hidden_size // 2),
            ("o_proj", hidden_size),
        ):
            base = prefix + "self_attn." + proj + ".weight_"
            # Symmetric INT4 midpoint: zero attention, without leaving any
            # initializer unset or introducing attention numerical noise.
            state[base + "qweight"] = torch.full(
                (rows, hidden_size // 2), 0x88, dtype=torch.uint8
            )
            state[base + "scales"] = torch.ones(
                rows, hidden_size // group_size, dtype=torch.float16
            )
    return state


def test_int2_fixture_preserves_lsb_first_codes_and_offset_two():
    packed = torch.tensor([[[0xE4, 0x1B]]], dtype=torch.uint8)
    scale = torch.ones(1, 1, 1, dtype=torch.float16)
    torch.testing.assert_close(
        _dequantize(packed, scale, bits=2, group_size=8),
        torch.tensor([[[-2, -1, 0, 1, 1, 0, -1, -2]]], dtype=torch.float32),
    )
    rng = np.random.default_rng(744)
    codes = rng.integers(0, 4, (_EXPERTS, 2, 8), dtype=np.uint8)
    packed, scales = _packed_weights(np.random.default_rng(744), 2, 2, 8, 8)
    expected = torch.from_numpy(codes.astype(np.float32) - 2) * scales.float()
    torch.testing.assert_close(_dequantize(packed, scales, 2, 8), expected)


@pytest.mark.integration
@pytest.mark.parametrize(
    "layouts,hidden_size,intermediate_size,group_size,tokens,route",
    [
        pytest.param(((4, 8), (8, 4)), 64, 64, 32, 3, "dense", id="adjacent-4x8-8x4"),
        pytest.param(((8, 8),), 64, 64, 32, 3, "dense", id="uniform-8x8"),
        pytest.param(((2, 4),), 512, 512, 64, 1, "packed", id="int2x4-block64-decode"),
        pytest.param(((2, 4),), 512, 512, 128, 1, "packed", id="int2x4-block128-decode"),
        pytest.param(((2, 4),), 512, 512, 64, 257, "dense", id="int2x4-block64-prefill"),
        pytest.param(((2, 4),), 512, 512, 128, 257, "dense", id="int2x4-block128-prefill"),
    ],
)
def test_olive_qwen3_moe_full_logits_on_cuda(
    layouts,
    hidden_size,
    intermediate_size,
    group_size,
    tokens,
    route,
    tmp_path,
    monkeypatch,
    record_property,
):
    """Run the exported model, not a standalone hand-written QMoE node."""
    if os.environ.get("MOBIUS_QMOE_CUDA_TEST") != "1":
        pytest.skip("set MOBIUS_QMOE_CUDA_TEST=1 with requirements/ci/qmoe-cuda.txt")

    import onnxruntime as ort

    assert ort.__version__ == _SYNTHETIC_ORT_VERSION, ort.__version__
    assert f"git-commit-id={_SYNTHETIC_ORT_COMMIT}" in ort.get_build_info(), (
        f"pinned CUDA wheel required, got {ort.get_build_info()}"
    )
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    record_property("ort_version", ort.__version__)
    record_property("ort_build", ort.get_build_info())
    record_property("qmoe_route", route)
    # The pin enables native INT2 prefill by default. Keep it off to isolate
    # packed GEMV from dense fallback, rather than accidentally testing prefill.
    monkeypatch.setenv("ORT_ENABLE_QMOE_INT2_PREFILL", "0")
    monkeypatch.setenv("ORT_DISABLE_MOE_GEMV", "0" if route == "packed" else "1")
    monkeypatch.setenv("ORT_ENABLE_QMOE_KERNEL_DEBUG_INFO", "1")
    head_dim = hidden_size // 4

    overrides = {
        f"model.layers.{layer}.mlp.experts.{projection}": QuantizationOverride(bits=bits)
        for layer, pair in enumerate(layouts)
        for projection, bits in zip(("gate_up_proj", "down_proj"), pair)
        if bits != 4
    }
    config = make_config(
        dtype=ir.DataType.FLOAT16,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=intermediate_size,
        num_hidden_layers=len(layouts),
        num_local_experts=_EXPERTS,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        vocab_size=_VOCAB,
        quantization=QuantizationConfig(
            bits=4,
            group_size=group_size,
            quant_method="olive",
            sym=True,
            overrides=overrides,
        ),
    )
    state = _checkpoint(layouts, hidden_size, intermediate_size, group_size)
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
    processed.update(
        preprocess_olive_weights(attention_sidecars, bits=4, group_size=group_size)
    )
    from mobius import build_from_module

    package = build_from_module(module, config)
    package.apply_weights(processed, fold_constants=False)
    model = package["model"]
    qmoes = [node for node in model.graph if node.op_type == "QMoE"]
    assert len(qmoes) == len(layouts)
    for layer, (node, (fc1, fc2)) in enumerate(zip(qmoes, layouts)):
        assert node.domain == "com.microsoft"
        assert node.attributes["weights_prepacked"].value == 0
        assert node.attributes["block_size"].value == group_size
        assert hidden_size % group_size == intermediate_size % group_size == 0
        assert node.attributes["quant_type"].value == "int"
        assert node.attributes["activation_type"].value == "swiglu"
        assert node.attributes["swiglu_fusion"].value == 1
        # Raw packed weights and 3-D scales matching the activation dtype.
        assert node.inputs[0].dtype == ir.DataType.FLOAT16
        assert node.inputs[2].dtype == node.inputs[5].dtype == ir.DataType.UINT8
        assert node.inputs[3].dtype == node.inputs[6].dtype == node.inputs[0].dtype
        for label in ("fc1", "fc2"):
            prefix = f"model.layers.{layer}.mlp.{label}_scales"
            assert processed[prefix].ndim == 3
            assert processed[prefix].dtype == torch.float16
        assert node.attributes["expert_weight_bits"].value == (4 if fc1 != fc2 else fc1)
        if fc1 != fc2:
            assert tuple(
                node.attributes[f"fc{i}_expert_weight_bits"].value for i in (1, 2, 3)
            ) == (fc1, fc2, fc1)
    assert all(init.const_value is not None for init in model.graph.initializers.values())

    input_ids = np.resize(np.array([1, 9, 3], dtype=np.int64), (1, tokens))
    expected = _reference(input_ids, state, layouts, group_size)
    # Catch accidentally vacuous checkpoints even if the CUDA path regresses
    # to returning only the embedding residual.
    residual_logits = (
        _rms(state["model.embed_tokens.weight"][torch.from_numpy(input_ids)].float())
        @ state["lm_head.weight"].float().T
    ).numpy()
    assert np.max(np.abs(expected - residual_logits)) > 0.02
    assert not np.allclose(expected, residual_logits, rtol=0.01, atol=0.01)
    swapped = state.copy()
    gate_up_key = "model.layers.0.mlp.experts.gate_up_proj_qweight"
    swapped[gate_up_key] = state[gate_up_key].roll(intermediate_size, dims=1)
    swapped_logits = _reference(input_ids, swapped, layouts, group_size)
    assert np.max(np.abs(expected - swapped_logits)) > 0.02
    assert not np.allclose(expected, swapped_logits, rtol=0.01, atol=0.01)

    model_path = tmp_path / "model.onnx"
    ir.save(model, model_path, external_data="model.onnx.data")
    options = ort.SessionOptions()
    options.add_session_config_entry(
        _SCRATCH_KEY, str(1 if route == "packed" else _FALLBACK_SCRATCH_BYTES)
    )
    # CPU shape/control nodes are legitimate; require CUDA specifically for QMoE.
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "profile")
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CUDAExecutionProvider"]
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    feeds = {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids),
        "position_ids": np.arange(tokens, dtype=np.int64)[None, :],
    }
    for layer in range(len(layouts)):
        for cache in ("key", "value"):
            feeds[f"past_key_values.{layer}.{cache}"] = np.zeros(
                (1, 2, 0, head_dim), dtype=np.float16
            )
    actual = session.run(["logits"], feeds)[0]
    profile = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
    qmoe_events = [
        event
        for event in profile
        if event.get("cat") == "Node" and event.get("args", {}).get("op_name") == "QMoE"
    ]
    assert len(qmoe_events) >= len(layouts), "QMoE execution missing from ORT profile"
    assert all(
        event.get("args", {}).get("provider") == "CUDAExecutionProvider"
        for event in qmoe_events
    ), qmoe_events
    np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=0.01, atol=0.01)
    if layouts[0][0] == 2:
        # A successful 1-byte-budget decode cannot be dense dequantization.
        # Verify that the guard really rejects the same model when packing is
        # disabled, rather than trusting a CUDA provider label as path proof.
        del session
        monkeypatch.setenv("ORT_DISABLE_MOE_GEMV", "1")
        guard_options = ort.SessionOptions()
        guard_options.add_session_config_entry(_SCRATCH_KEY, "1")
        fallback = ort.InferenceSession(
            str(model_path), sess_options=guard_options, providers=["CUDAExecutionProvider"]
        )
        assert fallback.get_providers()[0] == "CUDAExecutionProvider"
        with pytest.raises(
            ort.capi.onnxruntime_pybind11_state.Fail,
            match=r"INT2 or mixed-width CUDA QMoE dense fallback requires .*"
            r"exceeding the configured limit of 1 bytes",
        ):
            fallback.run(["logits"], feeds)


_REAL_H = 2048
_REAL_INTER = 768
_REAL_BLOCK = 128
_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
_SNAPSHOT = (
    _DATA_ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots" / _REVISION
)


def _real_slice() -> dict[str, torch.Tensor]:
    """Select only the approved rows; never materialize the full embedding."""
    from safetensors import safe_open

    shard_name = "model-00001-of-00016.safetensors"
    index_path = _SNAPSHOT / "model.safetensors.index.json"
    shard_path = _SNAPSHOT / shard_name
    assert index_path.is_file() and shard_path.is_file(), "pinned Qwen3 shard/index missing"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    shapes = {
        "model.embed_tokens.weight": (151936, _REAL_H),
        "model.layers.0.mlp.gate.weight": (128, _REAL_H),
    }
    for expert in range(_EXPERTS):
        for proj, shape in (
            ("gate_proj", (_REAL_INTER, _REAL_H)),
            ("up_proj", (_REAL_INTER, _REAL_H)),
            ("down_proj", (_REAL_H, _REAL_INTER)),
        ):
            shapes[f"model.layers.0.mlp.experts.{expert}.{proj}.weight"] = shape
    assert all(index.get(key) == shard_name for key in shapes), "unexpected shard mapping"
    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        available = set(shard.keys())
        assert all(
            key in available and tuple(shard.get_slice(key).get_shape()) == shape
            for key, shape in shapes.items()
        ), "unexpected checkpoint tensor shape"
        weights = {
            key: shard.get_slice(key)[: 32 if "embed_tokens" in key else 4]
            .clone()
            .to(torch.float16)
            if key in ("model.embed_tokens.weight", "model.layers.0.mlp.gate.weight")
            else shard.get_slice(key)[:].clone().to(torch.float16)
            for key in shapes
        }
    return weights


def _olive_sidecars(
    weights: dict[str, torch.Tensor], fc1_bits: int, fc2_bits: int, directory: Path
) -> dict[str, torch.Tensor]:
    """Save and reload actual Olive state_dict buffers, not reconstructed packed bytes."""
    from olive.common.quant.state_dict import install_quant_tensor_param
    from olive.common.quant.tensor import QuantTensor
    from safetensors.torch import load_file, save_file

    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    mlp = torch.nn.Module()
    root.model.layers[0].mlp = mlp
    mlp.experts = torch.nn.Module()
    prefix = "model.layers.0.mlp.experts."
    gate_up = torch.stack(
        [
            torch.cat(
                (
                    weights[f"{prefix}{expert}.gate_proj.weight"],
                    weights[f"{prefix}{expert}.up_proj.weight"],
                )
            )
            for expert in range(_EXPERTS)
        ]
    )
    down = torch.stack(
        [weights[f"{prefix}{expert}.down_proj.weight"] for expert in range(_EXPERTS)]
    )
    for name, tensor, bits in (
        ("gate_up_proj", gate_up, fc1_bits),
        ("down_proj", down, fc2_bits),
    ):
        install_quant_tensor_param(
            mlp.experts,
            name,
            QuantTensor.from_float(tensor, bits=bits, symmetric=True, group_size=_REAL_BLOCK),
        )
    sidecars = dict(root.state_dict())
    expected = {
        prefix + "gate_up_proj_qweight": (_EXPERTS, 2 * _REAL_INTER, _REAL_H * fc1_bits // 8),
        prefix + "gate_up_proj_scales": (_EXPERTS, 2 * _REAL_INTER, _REAL_H // _REAL_BLOCK),
        prefix + "down_proj_qweight": (_EXPERTS, _REAL_H, _REAL_INTER * fc2_bits // 8),
        prefix + "down_proj_scales": (_EXPERTS, _REAL_H, _REAL_INTER // _REAL_BLOCK),
    }
    assert set(sidecars) == set(expected), f"unexpected Olive buffers: {set(sidecars)}"
    for key, shape in expected.items():
        assert tuple(sidecars[key].shape) == shape
        assert sidecars[key].dtype == (
            torch.uint8 if key.endswith("qweight") else torch.float16
        )
    path = directory / "olive-experts.safetensors"
    save_file({key: value.contiguous() for key, value in sidecars.items()}, path)
    reloaded = load_file(path)
    assert all(torch.equal(sidecars[key], reloaded[key]) for key in expected)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "source_revision": _REVISION,
                "source_shard": "model-00001-of-00016.safetensors",
                "experts": list(range(_EXPERTS)),
                "layout": [fc1_bits, fc2_bits],
                "quantization": {
                    "producer": "olive.common.quant.tensor.QuantTensor.from_float",
                    "symmetric": True,
                    "group_size": _REAL_BLOCK,
                    "input_dtype": "float16",
                },
                "selected_source_fp16_sha256": {
                    key: hashlib.sha256(weights[key].numpy().tobytes()).hexdigest()
                    for key in sorted(weights)
                },
                "sidecars_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tensor_sha256": {
                    key: hashlib.sha256(reloaded[key].numpy().tobytes()).hexdigest()
                    for key in sorted(expected)
                },
                "keys": sorted(expected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return reloaded


def _real_reference(
    input_ids: np.ndarray, state: dict[str, torch.Tensor], bits: tuple[int, int]
) -> np.ndarray:
    p = "model.layers.0.mlp."
    x = _rms(state["model.embed_tokens.weight"][torch.from_numpy(input_ids)].float())
    scores = x @ state[p + "gate.weight"].float().T
    indices = scores.topk(2, dim=-1).indices
    probs = scores.softmax(dim=-1).gather(-1, indices)
    probs /= probs.sum(dim=-1, keepdim=True)
    fc1 = _dequantize(
        state[p + "experts.gate_up_proj_qweight"],
        state[p + "experts.gate_up_proj_scales"],
        bits[0],
        _REAL_BLOCK,
    )
    fc2 = _dequantize(
        state[p + "experts.down_proj_qweight"],
        state[p + "experts.down_proj_scales"],
        bits[1],
        _REAL_BLOCK,
    )
    routed = torch.zeros_like(x)
    for slot in range(2):
        for expert in range(_EXPERTS):
            gate = x @ fc1[expert, :_REAL_INTER].T
            up = x @ fc1[expert, _REAL_INTER:].T
            value = (functional.silu(gate) * up) @ fc2[expert].T
            routed += (
                value
                * (indices[..., slot] == expert).unsqueeze(-1)
                * probs[..., slot].unsqueeze(-1)
            )
    return (
        _rms(state["model.embed_tokens.weight"][torch.from_numpy(input_ids)].float() + routed)
        @ state["lm_head.weight"].float().T
    ).numpy()


@pytest.mark.integration
@pytest.mark.parametrize("bits", [(4, 8), (8, 4)], ids=["real-4x8", "real-8x4"])
def test_real_qwen3_slice_olive_qmoe_cuda(bits):
    """Bounded checkpoint experiment, not original-model HF or full Olive parity."""
    if os.environ.get("MOBIUS_QMOE_REAL_SLICE_CUDA_TEST") != "1":
        pytest.skip("set MOBIUS_QMOE_REAL_SLICE_CUDA_TEST=1 for pinned local CUDA experiment")

    import onnxruntime as ort

    assert f"git-commit-id={_ORT_COMMIT}" in ort.get_build_info()
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    assert _SNAPSHOT.is_dir(), f"pinned snapshot missing: {_SNAPSHOT}"
    directory = _DATA_ROOT / "qmoe-real-slice-744" / f"{bits[0]}x{bits[1]}"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    weights = _real_slice()
    sidecars = _olive_sidecars(weights, *bits, directory)
    p = "model.layers.0.mlp."
    state = {
        "model.embed_tokens.weight": weights["model.embed_tokens.weight"],
        "model.norm.weight": torch.ones(_REAL_H, dtype=torch.float16),
        "lm_head.weight": torch.eye(32, _REAL_H, dtype=torch.float16) * 2,
        p + "gate.weight": weights[p + "gate.weight"],
        "model.layers.0.input_layernorm.weight": torch.ones(_REAL_H, dtype=torch.float16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(
            _REAL_H, dtype=torch.float16
        ),
        **sidecars,
    }
    # The attention branch is explicitly zero, using Olive-produced sidecars
    # even for its ordinary MatMulNBits projections.
    from olive.common.quant.tensor import QuantTensor

    from mobius import build_from_module
    from mobius._weight_utils import preprocess_olive_weights

    for proj, rows, columns in (
        ("q_proj", 4096, _REAL_H),
        ("k_proj", 512, _REAL_H),
        ("v_proj", 512, _REAL_H),
        ("o_proj", _REAL_H, 4096),
    ):
        qt = QuantTensor.from_float(
            torch.zeros(rows, columns, dtype=torch.float16),
            bits=4,
            symmetric=True,
            group_size=_REAL_BLOCK,
        )
        key = f"model.layers.0.self_attn.{proj}.weight_"
        state[key + "qweight"] = qt.qweight
        state[key + "scales"] = qt.scales
    overrides = {
        f"model.layers.0.mlp.experts.{name}": QuantizationOverride(bits=width)
        for name, width in zip(("gate_up_proj", "down_proj"), bits)
        if width != 4
    }
    config = make_config(
        dtype=ir.DataType.FLOAT16,
        hidden_size=_REAL_H,
        intermediate_size=_REAL_INTER,
        moe_intermediate_size=_REAL_INTER,
        num_hidden_layers=1,
        num_local_experts=_EXPERTS,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=32,
        quantization=QuantizationConfig(
            bits=4,
            group_size=_REAL_BLOCK,
            quant_method="olive",
            sym=True,
            overrides=overrides,
        ),
    )
    module = MoECausalLMModel(config)
    processed = module.preprocess_weights(state.copy())
    attention = {
        key: processed.pop(key)
        for key in list(processed)
        if ".self_attn." in key and key.endswith(("_qweight", "_scales"))
    }
    processed.update(preprocess_olive_weights(attention, bits=4, group_size=_REAL_BLOCK))
    package = build_from_module(module, config)
    package.apply_weights(processed, fold_constants=False)
    model = package["model"]
    qmoes = [node for node in model.graph if node.op_type == "QMoE"]
    assert len(qmoes) == 1
    node = qmoes[0]
    assert node.domain == "com.microsoft"
    assert node.attributes["block_size"].value == _REAL_BLOCK
    assert node.attributes["quant_type"].value == "int"
    assert node.attributes["activation_type"].value == "swiglu"
    assert node.attributes["swiglu_fusion"].value == 1
    assert node.attributes["weights_prepacked"].value == 0
    assert node.attributes["expert_weight_bits"].value == 4
    assert tuple(node.attributes[f"fc{i}_expert_weight_bits"].value for i in (1, 2, 3)) == (
        bits[0],
        bits[1],
        bits[0],
    )
    assert node.inputs[0].dtype == ir.DataType.FLOAT16
    assert node.inputs[2].dtype == node.inputs[5].dtype == ir.DataType.UINT8
    assert node.inputs[3].dtype == node.inputs[6].dtype == ir.DataType.FLOAT16
    for name, shape in (
        ("fc1_experts_weights", (_EXPERTS, 2 * _REAL_INTER, _REAL_H * bits[0] // 8)),
        ("fc1_scales", (_EXPERTS, 2 * _REAL_INTER, _REAL_H // _REAL_BLOCK)),
        ("fc2_experts_weights", (_EXPERTS, _REAL_H, _REAL_INTER * bits[1] // 8)),
        ("fc2_scales", (_EXPERTS, _REAL_H, _REAL_INTER // _REAL_BLOCK)),
    ):
        assert tuple(processed[p + name].shape) == shape
    assert all(init.const_value is not None for init in model.graph.initializers.values())
    input_ids = np.array([[1, 9, 3]], dtype=np.int64)
    expected = _real_reference(input_ids, state, bits)
    assert np.isfinite(expected).all()
    residual = _rms(state["model.embed_tokens.weight"][torch.from_numpy(input_ids)].float())
    baseline = (residual @ state["lm_head.weight"].float().T).numpy()
    assert np.max(np.abs(expected - baseline)) > 0.02
    swapped = state.copy()
    key = p + "experts.gate_up_proj_qweight"
    swapped[key] = state[key].roll(_REAL_INTER, dims=1)
    assert np.max(np.abs(expected - _real_reference(input_ids, swapped, bits))) > 0.02
    model_path = directory / "model.onnx"
    ir.save(model, model_path, external_data="model.onnx.data")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(directory / "profile")
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CUDAExecutionProvider"]
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    actual = session.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids),
            "position_ids": np.arange(3, dtype=np.int64)[None, :],
            "past_key_values.0.key": np.zeros((1, 4, 0, 128), dtype=np.float16),
            "past_key_values.0.value": np.zeros((1, 4, 0, 128), dtype=np.float16),
        },
    )[0]
    profile = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
    events = [
        event
        for event in profile
        if event.get("cat") == "Node"
        and (event.get("args", {}).get("op_name") == "QMoE" or "QMoE" in event.get("name", ""))
    ]
    assert len(events) == 1 and events[0]["args"]["provider"] == "CUDAExecutionProvider"
    assert np.isfinite(actual).all()
    error = np.abs(actual.astype(np.float32) - expected)
    print(f"real {bits}: max_abs={error.max():.6f}, seconds={time.monotonic() - started:.2f}")
    np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=0.01, atol=0.01)
