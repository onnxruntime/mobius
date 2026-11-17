# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import (
    _raise_for_invalid_bitnet_tensor_contract,
    build_from_gguf,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.integrations.gguf._tensor_processors import process_tensors
from mobius.models.bitnet import BitNetCausalLMModel
from mobius.tasks import CausalLMTask


class _FakeBitNetGGUF:
    architecture = "bitnet"

    def __init__(self):
        self.metadata = {
            "bitnet.context_length": 32,
            "bitnet.embedding_length": 8,
            "bitnet.feed_forward_length": 16,
            "bitnet.block_count": 1,
            "bitnet.attention.head_count": 2,
            "bitnet.attention.head_count_kv": 1,
            "bitnet.attention.layer_norm_rms_epsilon": 1e-5,
            "bitnet.rope.freq_base": 10_000.0,
            "bitnet.rope.dimension_count": 4,
            "bitnet.vocab_size": 24,
        }
        self.tensors = {
            "token_embd.weight": (24, 8),
            "output_norm.weight": (8,),
            "blk.0.attn_norm.weight": (8,),
            "blk.0.attn_sub_norm.weight": (8,),
            "blk.0.attn_q.weight": (8, 8),
            "blk.0.attn_k.weight": (4, 8),
            "blk.0.attn_v.weight": (4, 8),
            "blk.0.attn_output.weight": (8, 8),
            "blk.0.ffn_norm.weight": (8,),
            "blk.0.ffn_sub_norm.weight": (16,),
            "blk.0.ffn_gate.weight": (16, 8),
            "blk.0.ffn_up.weight": (16, 8),
            "blk.0.ffn_down.weight": (8, 16),
        }
        self.qtypes = {}
        self.tensor_names = list(self.tensors)

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        return [
            (name, None, self.qtypes.get(name), shape) for name, shape in self.tensors.items()
        ]


def test_bitnet_is_dedicated_registered_architecture() -> None:
    assert registry.get("bitnet") is BitNetCausalLMModel
    spec = get_arch_spec("bitnet")
    assert spec.model_type == "bitnet"
    assert spec.is_importable
    assert spec.quantized_import is Support.REJECTED
    assert spec.runtime is Support.DEFERRED


def test_bitnet_gguf_config_tensor_and_graph_closure() -> None:
    source = _FakeBitNetGGUF()
    _raise_for_invalid_bitnet_tensor_contract(source)
    config = gguf_to_config(source)
    assert config.model_type == "bitnet"
    assert config.hidden_act == "silu"
    assert config.tie_word_embeddings
    assert config.rms_norm_eps == pytest.approx(1e-5)

    model = registry.get(config.model_type)(config)
    graph = CausalLMTask().build(model, config)["model"]
    initializers = graph.graph.initializers
    assert "model.layers.0.self_attn.attn_sub_norm.weight" in initializers
    assert "model.layers.0.mlp.ffn_sub_norm.weight" in initializers
    assert "lm_head.weight" not in initializers

    mapped = {map_gguf_to_hf_names(name, "bitnet") for name in source.tensor_names}
    graph_weights = {
        name.removesuffix("_t")
        for name in initializers
        if not name.startswith("const_") and ".rotary_emb." not in name
    }
    assert None not in mapped
    # The tied output projection reuses the embedding initializer. Linear
    # transpose folding may rename any matrix initializer with ``_t``.
    assert graph_weights == mapped


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_attn_subnorm", r"missing=.*attn_sub_norm"),
        ("missing_ffn_subnorm", r"missing=.*ffn_sub_norm"),
        ("unexpected_output", r"unexpected=.*output\.weight"),
        ("malformed_projection", r"malformed=.*attn_k\.weight"),
        ("malformed_scale", r"malformed=.*attn_q\.scale"),
    ],
)
def test_bitnet_tensor_contract_rejects_mutations(
    mutation: str, expected: str, monkeypatch
) -> None:
    from mobius import _builder as core_builder

    source = _FakeBitNetGGUF()
    if mutation == "missing_attn_subnorm":
        del source.tensors["blk.0.attn_sub_norm.weight"]
    elif mutation == "missing_ffn_subnorm":
        del source.tensors["blk.0.ffn_sub_norm.weight"]
    elif mutation == "unexpected_output":
        source.tensors["output.weight"] = (24, 8)
    elif mutation == "malformed_projection":
        source.tensors["blk.0.attn_k.weight"] = (8, 8)
    else:
        source.tensors["blk.0.attn_q.scale"] = (2,)
    source.tensor_names = list(source.tensors)
    monkeypatch.setattr(
        core_builder,
        "build_from_module",
        lambda *args, **kwargs: pytest.fail("graph construction must not start"),
    )

    with pytest.raises(ValueError, match=expected):
        build_from_gguf(
            f"bitnet-{mutation}.gguf",
            keep_quantized=False,
            _gguf_model=source,
        )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", r"missing required architecture metadata.*head_count_kv"),
        ("geometry", r"equal full attention/RoPE head widths"),
    ],
)
def test_bitnet_tensor_contract_rejects_invalid_metadata(mutation: str, expected: str) -> None:
    source = _FakeBitNetGGUF()
    if mutation == "missing":
        del source.metadata["bitnet.attention.head_count_kv"]
    else:
        source.metadata["bitnet.rope.dimension_count"] = 2

    with pytest.raises(ValueError, match=expected):
        _raise_for_invalid_bitnet_tensor_contract(source)


def test_bitnet_scalar_scale_folds_dequantized_ternary_values() -> None:
    config = dataclasses.replace(
        ArchitectureConfig(),
        model_type="bitnet",
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=4,
        head_dim=2,
    )
    config._gguf_arch = "bitnet"
    packed_values = torch.tensor(
        [
            [-1.0, 0.0, 1.0, -1.0],
            [1.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
        ]
    )
    state = {
        "model.layers.0.self_attn.o_proj.weight": packed_values.clone(),
        "model.layers.0.self_attn.o_proj.scale": torch.tensor([0.125]),
    }
    result = process_tensors(state, config)
    assert "model.layers.0.self_attn.o_proj.scale" not in result
    torch.testing.assert_close(
        result["model.layers.0.self_attn.o_proj.weight"],
        packed_values * 0.125,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("qtype_name", ["TQ1_0", "TQ2_0"])
def test_bitnet_pinned_ternary_qtypes_dequantize_by_value(qtype_name: str) -> None:
    from gguf import GGMLQuantizationType, dequantize, quantize

    qtype = getattr(GGMLQuantizationType, qtype_name)
    codes = np.resize(np.array([-1.0, 0.0, 1.0], dtype=np.float32), 256).reshape(1, 256)
    represented = codes * np.float32(0.5)
    packed = quantize(represented, qtype)
    assert packed.shape == ((1, 54) if qtype_name == "TQ1_0" else (1, 66))
    dequantized = dequantize(packed, qtype)
    np.testing.assert_array_equal(dequantized, represented)


def _write_synthetic_bitnet(
    path: Path, qtype_name: str | None = None
) -> dict[str, np.ndarray]:
    from gguf import GGMLQuantizationType, GGUFWriter, quantize

    hidden = 8 if qtype_name is None else 256
    heads = 2 if qtype_name is None else 1
    kv_hidden = hidden // heads
    intermediate = hidden * 2
    writer = GGUFWriter(str(path), "bitnet")
    writer.add_context_length(32)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(1)
    writer.add_head_count(heads)
    writer.add_head_count_kv(1)
    writer.add_rope_freq_base(10_000.0)
    writer.add_rope_dimension_count(hidden // heads)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(24)

    rng = np.random.default_rng(0)
    ternary = rng.integers(-1, 2, size=(hidden, hidden)).astype(np.float32)
    tensors = {
        "token_embd.weight": rng.normal(size=(24, hidden)).astype(np.float32),
        "output_norm.weight": np.linspace(0.75, 1.25, hidden, dtype=np.float32),
        "blk.0.attn_norm.weight": np.linspace(0.8, 1.2, hidden, dtype=np.float32),
        "blk.0.attn_sub_norm.weight": np.linspace(0.9, 1.1, hidden, dtype=np.float32),
        # Repeated rows are invariant under GGUF's Q/K head permutation, so the
        # independently loaded Hugging Face reference sees exactly these values.
        "blk.0.attn_q.weight": np.repeat(ternary[:1], hidden, axis=0),
        "blk.0.attn_k.weight": np.repeat(ternary[1:2], kv_hidden, axis=0),
        "blk.0.attn_v.weight": ternary[:kv_hidden],
        "blk.0.attn_output.weight": ternary,
        "blk.0.ffn_norm.weight": np.linspace(0.7, 1.3, hidden, dtype=np.float32),
        "blk.0.ffn_sub_norm.weight": np.linspace(0.85, 1.15, intermediate, dtype=np.float32),
        "blk.0.ffn_gate.weight": rng.integers(-1, 2, size=(intermediate, hidden)).astype(
            np.float32
        ),
        "blk.0.ffn_up.weight": rng.integers(-1, 2, size=(intermediate, hidden)).astype(
            np.float32
        ),
        "blk.0.ffn_down.weight": rng.integers(-1, 2, size=(hidden, intermediate)).astype(
            np.float32
        ),
    }
    for name, value in tensors.items():
        if qtype_name is not None and name == "blk.0.attn_output.weight":
            qtype = getattr(GGMLQuantizationType, qtype_name)
            writer.add_tensor(name, quantize(value, qtype), raw_dtype=qtype)
        else:
            writer.add_tensor(name, value)
    writer.add_tensor("blk.0.attn_output.scale", np.array([0.25], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


def test_bitnet_quantization_preservation_fails_closed() -> None:
    from gguf import GGMLQuantizationType

    source = _FakeBitNetGGUF()
    source.qtypes["token_embd.weight"] = GGMLQuantizationType.TQ1_0

    with pytest.raises(
        ValueError,
        match=r"bitnet.*does not support keep_quantized=True.*ternary codebook",
    ):
        build_from_gguf("bitnet-tq.gguf", _gguf_model=source)


@pytest.mark.parametrize("qtype_name", [None, "TQ1_0", "TQ2_0"])
def test_bitnet_synthetic_gguf_build_folds_projection_scale(
    tmp_path: Path, qtype_name: str | None
) -> None:
    path = tmp_path / f"bitnet-{qtype_name or 'f32'}.gguf"
    tensors = _write_synthetic_bitnet(path, qtype_name)

    model = build_from_gguf(path, keep_quantized=False)["model"]
    initializers = model.graph.initializers
    name = "model.layers.0.self_attn.o_proj.weight_t"
    if name in initializers:
        actual = initializers[name].const_value.numpy().T
    else:
        actual = initializers["model.layers.0.self_attn.o_proj.weight"].const_value.numpy()
    np.testing.assert_array_equal(actual, tensors["blk.0.attn_output.weight"] * 0.25)


def test_bitnet_nonzero_ort_prefill_and_cached_decode_match_huggingface(
    tmp_path: Path,
) -> None:
    from transformers import BitNetConfig, BitNetForCausalLM

    from mobius._testing.ort_inference import OnnxModelSession

    path = tmp_path / "bitnet-runtime.gguf"
    tensors = _write_synthetic_bitnet(path)
    onnx_model = build_from_gguf(path, keep_quantized=False)["model"]

    reference = BitNetForCausalLM(
        BitNetConfig(
            vocab_size=24,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            rope_theta=10_000.0,
            rms_norm_eps=1e-5,
            hidden_act="silu",
            tie_word_embeddings=True,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()
    state = {}
    for gguf_name, value in tensors.items():
        hf_name = map_gguf_to_hf_names(gguf_name, "bitnet")
        assert hf_name is not None
        if gguf_name == "blk.0.attn_output.weight":
            value = value * np.float32(0.25)
        state[hf_name] = torch.from_numpy(value.copy())
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    reference.load_state_dict(state, strict=True)

    prompt = np.array([[2, 5, 7]], dtype=np.int64)
    prefill_mask = np.ones_like(prompt)
    prefill_positions = np.arange(prompt.shape[1], dtype=np.int64)[None, :]
    with torch.no_grad():
        reference_prefill = reference(
            input_ids=torch.from_numpy(prompt),
            attention_mask=torch.from_numpy(prefill_mask),
            position_ids=torch.from_numpy(prefill_positions),
            use_cache=True,
        )

    session = OnnxModelSession(onnx_model)
    try:
        onnx_prefill = session.run(
            {
                "input_ids": prompt,
                "attention_mask": prefill_mask,
                "position_ids": prefill_positions,
                "past_key_values.0.key": np.zeros((1, 1, 0, 4), dtype=np.float32),
                "past_key_values.0.value": np.zeros((1, 1, 0, 4), dtype=np.float32),
            }
        )
        np.testing.assert_allclose(
            onnx_prefill["logits"],
            reference_prefill.logits.numpy(),
            rtol=2e-4,
            atol=2e-4,
        )

        next_token = np.array([[11]], dtype=np.int64)
        decode_mask = np.ones((1, 4), dtype=np.int64)
        decode_position = np.array([[3]], dtype=np.int64)
        with torch.no_grad():
            reference_decode = reference(
                input_ids=torch.from_numpy(next_token),
                attention_mask=torch.from_numpy(decode_mask),
                position_ids=torch.from_numpy(decode_position),
                past_key_values=reference_prefill.past_key_values,
                use_cache=True,
            )
        onnx_decode = session.run(
            {
                "input_ids": next_token,
                "attention_mask": decode_mask,
                "position_ids": decode_position,
                "past_key_values.0.key": onnx_prefill["present.0.key"],
                "past_key_values.0.value": onnx_prefill["present.0.value"],
            }
        )
        np.testing.assert_allclose(
            onnx_decode["logits"],
            reference_decode.logits.numpy(),
            rtol=2e-4,
            atol=2e-4,
        )
    finally:
        session.close()


@pytest.mark.parametrize(
    "scale",
    [torch.tensor([1.0, 2.0]), torch.tensor([float("nan")])],
)
def test_bitnet_rejects_non_scalar_or_nonfinite_scale(scale: torch.Tensor) -> None:
    config = SimpleNamespace(
        model_type="bitnet",
        _gguf_arch="bitnet",
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
    )
    state = {
        "model.layers.0.mlp.down_proj.weight": torch.ones((2, 2)),
        "model.layers.0.mlp.down_proj.scale": scale,
    }
    with pytest.raises(ValueError, match="one finite scalar"):
        process_tensors(state, config)
