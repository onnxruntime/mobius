# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Weight-preprocessing tests for the generic MoE family (``qwen3_moe`` & co.).

Focus: packed *fused* expert tensors must survive
:func:`~mobius.models.moe._rename_moe_expert_weights` untouched so
``pack_qmoe_expert_weights`` can map them onto ``com.microsoft::QMoE``
``fc1``/``fc2`` parameters.

An Olive-quantized Qwen3-MoE checkpoint stores each layer's routed experts as
expert-major *packed* sidecars — ``experts.gate_up_proj_qweight``
``[E, 2*moe_inter, hidden*bits/8]`` uint8 next to ``experts.gate_up_proj_scales``
``[E, 2*moe_inter, hidden/group]`` bf16 (and ``_qzeros`` when asymmetric).
Every one of those keys contains ``.experts.gate_up_proj`` and is 3-D, so the
fused-expert *splitter* used to claim them: it reinterpreted packed bytes as
float rows **and** wrote every sidecar of a projection to the same
``experts.{i}.gate_proj.weight`` key, so only the last one survived and the QMoE
parameters were filled with garbage (or nothing at all).

All configs are tiny and synthetic — no checkpoint download.
"""

from __future__ import annotations

import dataclasses
import math

import onnx_ir as ir
import pytest
import torch

from mobius._configs import (
    QuantizationConfig,
    QuantizationOverride,
    QuantizedWeightFormat,
)
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.models.moe import MoECausalLMModel, _rename_moe_expert_weights

# Tiny mirror of the real Olive Qwen3-MoE ABI (hidden=2048, moe_inter=768,
# bits=4, group=128): every shape below keeps the same relationships.
_E, _H, _INT, _BLK, _BITS = 4, 64, 32, 16, 4
_FC1_OUT = 2 * _INT  # gate rows then up rows
# Packed/blocked column counts are derived from the *input* dim of a projection:
# gate_up_proj and the attention projections consume ``hidden``, down_proj
# consumes ``moe_intermediate``.
_HIDDEN_PACKED = _H * _BITS // 8
_INTER_PACKED = _INT * _BITS // 8
_HIDDEN_BLOCKS = _H // _BLK
_INTER_BLOCKS = _INT // _BLK
_LAYER = "model.layers.0."


def _quantization(*, sym: bool = True, **overrides) -> QuantizationConfig:
    return QuantizationConfig(
        bits=_BITS, group_size=_BLK, quant_method="olive", sym=sym, **overrides
    )


def _moe_config(quantization: QuantizationConfig | None) -> object:
    return make_config(
        num_hidden_layers=1,
        hidden_size=_H,
        intermediate_size=64,
        moe_intermediate_size=_INT,
        num_local_experts=_E,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        quantization=quantization,
    )


def _packed_expert_state_dict(*, sym: bool = True) -> dict[str, torch.Tensor]:
    """Fused expert-major Olive sidecars, as found in the Qwen3-MoE checkpoint.

    Olive suffixes the *parameter* name with an underscore
    (``gate_up_proj_qweight``), not a dotted sibling buffer — see Olive's
    ``olive/common/quant/state_dict.py``. Scales are bf16 in the real
    checkpoint.
    """
    p = f"{_LAYER}mlp."
    state_dict = {
        p + "experts.gate_up_proj_qweight": torch.randint(
            0, 256, (_E, _FC1_OUT, _HIDDEN_PACKED), dtype=torch.uint8
        ),
        p + "experts.gate_up_proj_scales": torch.rand(
            _E, _FC1_OUT, _HIDDEN_BLOCKS, dtype=torch.bfloat16
        ),
        p + "experts.down_proj_qweight": torch.randint(
            0, 256, (_E, _H, _INTER_PACKED), dtype=torch.uint8
        ),
        p + "experts.down_proj_scales": torch.rand(
            _E, _H, _INTER_BLOCKS, dtype=torch.bfloat16
        ),
        p + "gate.weight": torch.rand(_E, _H),
    }
    if not sym:
        state_dict[p + "experts.gate_up_proj_qzeros"] = torch.randint(
            0, 256, (_E, _FC1_OUT, math.ceil(_HIDDEN_BLOCKS * _BITS / 8)), dtype=torch.uint8
        )
        state_dict[p + "experts.down_proj_qzeros"] = torch.randint(
            0, 256, (_E, _H, math.ceil(_INTER_BLOCKS * _BITS / 8)), dtype=torch.uint8
        )
    return state_dict


def _packed_decoder_state_dict(*, sym: bool = True) -> dict[str, torch.Tensor]:
    """Full single-layer checkpoint: packed attention + packed routed experts.

    Attention projections are packed per-linear (``<proj>.weight_qweight``
    ``[N, hidden*bits/8]``); norms, router, embedding and LM head stay float,
    matching what Olive emits for Qwen3-MoE.
    """
    state_dict = _packed_expert_state_dict(sym=sym)
    kv_out = 2 * 16  # num_key_value_heads * head_dim
    for proj, out_features in (
        ("q_proj", 4 * 16),
        ("k_proj", kv_out),
        ("v_proj", kv_out),
        ("o_proj", _H),
    ):
        base = f"{_LAYER}self_attn.{proj}."
        state_dict[base + "weight_qweight"] = torch.randint(
            0, 256, (out_features, _HIDDEN_PACKED), dtype=torch.uint8
        )
        state_dict[base + "weight_scales"] = torch.rand(
            out_features, _HIDDEN_BLOCKS, dtype=torch.bfloat16
        )
    state_dict[f"{_LAYER}input_layernorm.weight"] = torch.rand(_H)
    state_dict[f"{_LAYER}post_attention_layernorm.weight"] = torch.rand(_H)
    state_dict["model.norm.weight"] = torch.rand(_H)
    state_dict["model.embed_tokens.weight"] = torch.rand(32, _H)
    state_dict["lm_head.weight"] = torch.rand(32, _H)
    return state_dict


class TestRenameMoEExpertWeightsPacked:
    """Packed fused sidecars must pass through the HF→ONNX renamer untouched."""

    def test_packed_olive_sidecars_stay_distinct_and_unsplit(self):
        """``_qweight`` and ``_scales`` keep their own keys, shapes and dtypes.

        Regression guard: the splitter matched on the ``.experts.gate_up_proj``
        substring, so both sidecars were split into per-expert
        ``experts.{i}.gate_proj.weight`` keys — the same key for both — and the
        packed payload was lost.
        """
        state_dict = _packed_expert_state_dict()
        original = {k: (tuple(v.shape), v.dtype) for k, v in state_dict.items()}

        out = _rename_moe_expert_weights(dict(state_dict))

        assert set(out) == set(state_dict), "packed keys must pass through unchanged"
        for key, tensor in out.items():
            assert (tuple(tensor.shape), tensor.dtype) == original[key]
            assert tensor is state_dict[key]
        # No per-expert float-style keys were fabricated from packed tensors.
        assert not any(f".experts.{i}." in k for k in out for i in range(_E))

    @pytest.mark.parametrize("suffix", ["_qweight", "_scales", "_qzeros"])
    def test_packed_granite_fused_linears_pass_through(self, suffix):
        """GraniteMoE's fused ``input_linear``/``output_linear`` are guarded too.

        Their branches keyed off the ``.input_linear.weight`` /
        ``.output_linear.weight`` substring, which a packed
        ``...input_linear.weight_qweight`` key also contains.

        Scope: this only pins the *generic renamer's* preservation of packed
        keys. ``GraniteMoECausalLMModel`` preprocesses through the base
        ``CausalLMModel`` (``qmoe_target_path=None``), so it never reaches
        ``pack_qmoe_expert_weights`` — quantized GraniteMoE export stays
        unwired, and this test does not claim otherwise.
        """
        state_dict = {
            f"{_LAYER}block_sparse_moe.input_linear.weight{suffix}": torch.zeros(
                _E, _FC1_OUT, _HIDDEN_PACKED, dtype=torch.uint8
            ),
            f"{_LAYER}block_sparse_moe.output_linear.weight{suffix}": torch.zeros(
                _E, _H, _INTER_PACKED, dtype=torch.uint8
            ),
        }

        out = _rename_moe_expert_weights(state_dict)

        # ``block_sparse_moe`` → ``mlp`` still applies (path rename only).
        assert set(out) == {
            f"{_LAYER}mlp.input_linear.weight{suffix}",
            f"{_LAYER}mlp.output_linear.weight{suffix}",
        }

    @pytest.mark.parametrize("suffix", [".qweight", ".scales", ".qzeros"])
    def test_packed_dotted_fused_experts_pass_through(self, suffix):
        """GPTQ/AWQ dotted sidecars on a fused expert parameter are guarded too."""
        state_dict = {
            f"{_LAYER}mlp.experts.gate_up_proj{suffix}": torch.zeros(
                _E, _FC1_OUT, _HIDDEN_PACKED, dtype=torch.uint8
            ),
            f"{_LAYER}mlp.experts.down_proj{suffix}": torch.zeros(
                _E, _H, _INTER_PACKED, dtype=torch.uint8
            ),
        }

        out = _rename_moe_expert_weights(dict(state_dict))

        assert set(out) == set(state_dict)

    def test_unquantized_fused_experts_still_split(self):
        """Float fused experts keep the dense per-expert unfusing behaviour."""
        p = f"{_LAYER}mlp."
        state_dict = {
            p + "experts.gate_up_proj": torch.rand(_E, _FC1_OUT, _H),
            p + "experts.down_proj": torch.rand(_E, _H, _INT),
            p + "gate.weight": torch.rand(_E, _H),
        }

        out = _rename_moe_expert_weights(state_dict)

        assert p + "experts.gate_up_proj" not in out
        assert p + "experts.down_proj" not in out
        for i in range(_E):
            assert out[f"{p}experts.{i}.gate_proj.weight"].shape == (_INT, _H)
            assert out[f"{p}experts.{i}.up_proj.weight"].shape == (_INT, _H)
            assert out[f"{p}experts.{i}.down_proj.weight"].shape == (_H, _INT)
        assert out[p + "gate.weight"].shape == (_E, _H)

    def test_unquantized_granite_fused_linears_still_split(self):
        """Float GraniteMoE fused linears keep splitting (and the router rename)."""
        p = f"{_LAYER}block_sparse_moe."
        state_dict = {
            p + "input_linear.weight": torch.rand(_E, _FC1_OUT, _H),
            p + "output_linear.weight": torch.rand(_E, _H, _INT),
            p + "router.layer.weight": torch.rand(_E, _H),
        }

        out = _rename_moe_expert_weights(state_dict)

        assert f"{_LAYER}mlp.gate.weight" in out
        for i in range(_E):
            assert out[f"{_LAYER}mlp.experts.{i}.gate_proj.weight"].shape == (_INT, _H)
            assert out[f"{_LAYER}mlp.experts.{i}.down_proj.weight"].shape == (_H, _INT)

    def test_router_rename_still_applies_to_packed_keys(self):
        """Module-path renames are suffix-preserving, so packed routers rename too."""
        out = _rename_moe_expert_weights(
            {
                f"{_LAYER}mlp.router.weight_qweight": torch.zeros(
                    _E, _HIDDEN_PACKED, dtype=torch.uint8
                )
            }
        )

        assert set(out) == {f"{_LAYER}mlp.gate.weight_qweight"}


class TestMoECausalLMPackedQMoEExport:
    """End-to-end ``preprocess_weights`` for an Olive-packed Qwen3-MoE layer."""

    def test_packed_experts_become_qmoe_parameters(self):
        model = MoECausalLMModel(_moe_config(_quantization()))

        out = model.preprocess_weights(_packed_expert_state_dict())

        p = f"{_LAYER}mlp."
        fc1, fc2 = out[p + "fc1_experts_weights"], out[p + "fc2_experts_weights"]
        assert fc1.shape == (_E, _FC1_OUT, _HIDDEN_PACKED)
        assert fc1.dtype == torch.uint8
        assert fc2.shape == (_E, _H, _INTER_PACKED)
        assert fc2.dtype == torch.uint8
        # Scales stay a separate tensor (they used to collide with qweight).
        assert out[p + "fc1_scales"].shape == (_E, _FC1_OUT, _HIDDEN_BLOCKS)
        assert out[p + "fc2_scales"].shape == (_E, _H, _INTER_BLOCKS)
        assert out[p + "fc1_scales"].dtype == torch.bfloat16
        # No dense per-expert fallback keys leaked through.
        assert not any(".mlp.experts." in k for k in out)

    def test_symmetric_quantization_emits_no_zero_points(self):
        model = MoECausalLMModel(_moe_config(_quantization(sym=True)))

        out = model.preprocess_weights(_packed_expert_state_dict())

        assert model.model.layers[0].mlp.fc1_experts_zero_points is None
        assert model.model.layers[0].mlp.fc2_experts_zero_points is None
        assert not any("zero_point" in k for k in out)

    def test_asymmetric_quantization_packs_zero_points(self):
        """The ``_qzeros`` sidecar survives the renamer and lands on QMoE too."""
        model = MoECausalLMModel(_moe_config(_quantization(sym=False)))

        out = model.preprocess_weights(_packed_expert_state_dict(sym=False))

        p = f"{_LAYER}mlp."
        params = dict(model.named_parameters())
        for key in (p + "fc1_experts_zero_points", p + "fc2_experts_zero_points"):
            assert out[key].dtype == torch.uint8
            assert tuple(params[key].shape) == tuple(out[key].shape)

    def test_every_produced_key_binds_to_a_named_parameter(self):
        """Whole-layer round trip: names *and* shapes match the built module."""
        model = MoECausalLMModel(_moe_config(_quantization()))
        params = dict(model.named_parameters())

        out = model.preprocess_weights(_packed_decoder_state_dict())

        assert out, "preprocess_weights returned an empty state dict"
        for name, tensor in out.items():
            assert name in params, f"{name} does not bind to any model parameter"
            assert tuple(params[name].shape) == tuple(tensor.shape), (
                f"shape mismatch for {name}: model expects "
                f"{tuple(params[name].shape)}, got {tuple(tensor.shape)}"
            )
        # The fused QMoE parameters are actually populated, not skipped.
        assert f"{_LAYER}mlp.fc1_experts_weights" in out
        assert f"{_LAYER}mlp.fc2_experts_weights" in out

    def test_attention_is_quantized_while_router_and_tables_stay_float(self):
        """Only the checkpoint's packed modules become ``QuantizedLinear``."""
        model = MoECausalLMModel(_moe_config(_quantization()))
        layer = model.model.layers[0]

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert type(getattr(layer.self_attn, proj)).__name__ == "QuantizedLinear"
        # Routed experts go through fused QMoE, not per-expert dense MLPs.
        assert layer.mlp.experts is None
        assert type(layer.mlp.gate).__name__ == "SoftmaxTopKGate"
        assert type(layer.input_layernorm).__name__ == "RMSNorm"
        assert type(model.lm_head).__name__ == "Linear"
        assert type(model.model.embed_tokens).__name__ == "Embedding"
        # Router, norms and the embedding/LM-head tables keep float parameters
        # (Olive leaves them unquantized), unlike the uint8 QMoE payloads.
        params = dict(model.named_parameters())
        for name in (
            f"{_LAYER}mlp.gate.weight",
            f"{_LAYER}input_layernorm.weight",
            f"{_LAYER}post_attention_layernorm.weight",
            "model.norm.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ):
            assert params[name].dtype.is_floating_point(), f"{name} must stay float"
        assert params[f"{_LAYER}mlp.fc1_experts_weights"].dtype == ir.DataType.UINT8
        assert params[f"{_LAYER}self_attn.q_proj.weight"].dtype == ir.DataType.UINT8

    def test_unquantized_model_uses_dense_expert_fallback(self):
        """Without quantization the fused float experts still un-fuse and bind."""
        model = MoECausalLMModel(_moe_config(None))
        params = dict(model.named_parameters())
        p = f"{_LAYER}mlp."

        out = model.preprocess_weights(
            {
                p + "experts.gate_up_proj": torch.rand(_E, _FC1_OUT, _H),
                p + "experts.down_proj": torch.rand(_E, _H, _INT),
                p + "gate.weight": torch.rand(_E, _H),
            }
        )

        assert model.model.layers[0].mlp.experts is not None
        assert not any("fc1_experts_weights" in k for k in out)
        for name, tensor in out.items():
            assert tuple(params[name].shape) == tuple(tensor.shape)


def _mixed_quantization(
    *,
    pattern: str = f"{_LAYER}mlp.experts.gate_up_proj",
    fc1_sym: bool = True,
    fc2_sym: bool = True,
    fc1_group_size: int = _BLK,
) -> QuantizationConfig:
    return QuantizationConfig(
        bits=4,
        group_size=_BLK,
        quant_method="olive",
        sym=fc2_sym,
        overrides={
            pattern: QuantizationOverride(
                bits=2,
                group_size=fc1_group_size,
                sym=fc1_sym,
            )
        },
    )


def _mixed_expert_state_dict(
    *,
    num_experts: int = _E,
    hidden_size: int = _H,
    intermediate_size: int = _INT,
    block_size: int = _BLK,
    fc1_sym: bool = True,
    fc2_sym: bool = True,
) -> dict[str, torch.Tensor]:
    prefix = f"{_LAYER}mlp.experts."
    fc1_out = 2 * intermediate_size
    state = {
        prefix + "gate_up_proj_qweight": torch.arange(
            num_experts * fc1_out * (hidden_size * 2 // 8),
            dtype=torch.int64,
        )
        .remainder(256)
        .to(torch.uint8)
        .reshape(num_experts, fc1_out, hidden_size * 2 // 8),
        prefix + "gate_up_proj_scales": torch.rand(
            num_experts,
            fc1_out,
            hidden_size // block_size,
            dtype=torch.bfloat16,
        ),
        prefix + "down_proj_qweight": torch.arange(
            num_experts * hidden_size * (intermediate_size * 4 // 8),
            dtype=torch.int64,
        )
        .remainder(256)
        .to(torch.uint8)
        .reshape(num_experts, hidden_size, intermediate_size * 4 // 8),
        prefix + "down_proj_scales": torch.rand(
            num_experts,
            hidden_size,
            intermediate_size // block_size,
            dtype=torch.bfloat16,
        ),
    }
    if not fc1_sym:
        state[prefix + "gate_up_proj_qzeros"] = torch.randint(
            0,
            256,
            (
                num_experts,
                fc1_out,
                math.ceil((hidden_size // block_size) * 2 / 8),
            ),
            dtype=torch.uint8,
        )
    if not fc2_sym:
        state[prefix + "down_proj_qzeros"] = torch.randint(
            0,
            256,
            (
                num_experts,
                hidden_size,
                math.ceil((intermediate_size // block_size) * 4 / 8),
            ),
            dtype=torch.uint8,
        )
    return state


class TestMixedWidthQMoEExport:
    @pytest.mark.parametrize(
        "pattern",
        [
            f"{_LAYER}mlp.experts.gate_up_proj",
            r"re:model\.layers\.\d+\.mlp\.experts\.gate_up_proj",
        ],
    )
    def test_exact_and_regex_fc1_override_emit_mixed_qmoe(self, pattern):
        config = _moe_config(_mixed_quantization(pattern=pattern))
        model = MoECausalLMModel(config)
        block = model.model.layers[0].mlp

        assert block.experts is None
        assert block.fc1_experts_weights.shape == ir.Shape([_E, _FC1_OUT, _H * 2 // 8])
        assert block.fc2_experts_weights.shape == ir.Shape([_E, _H, _INT * 4 // 8])

        from mobius._builder import build_from_module

        graph = build_from_module(model, config)["model"].graph
        qmoe = next(node for node in graph if node.op_type == "QMoE")
        assert qmoe.attributes["expert_weight_bits"].value == 4
        assert qmoe.attributes["fc1_expert_weight_bits"].value == 2
        assert qmoe.attributes["fc2_expert_weight_bits"].value == 4
        assert qmoe.attributes["fc3_expert_weight_bits"].value == 2

    @pytest.mark.parametrize("exact_first", [True, False])
    def test_plain_parent_override_does_not_shadow_exact_regex(self, exact_first):
        parent = (f"{_LAYER}mlp", QuantizationOverride(bits=8))
        regex = (
            r"re:model\.layers\.\d+\.mlp\.experts\.gate_up_proj",
            QuantizationOverride(bits=2),
        )
        overrides = dict((parent, regex) if exact_first else (regex, parent))
        quantization = QuantizationConfig(
            bits=4,
            group_size=_BLK,
            quant_method="olive",
            overrides=overrides,
        )

        block = MoECausalLMModel(_moe_config(quantization)).model.layers[0].mlp

        assert block._qmoe_quantization is not None
        assert block._qmoe_quantization.fc1.bits == 2
        assert block._qmoe_quantization.fc2.bits == 4

    def test_mixed_olive_payload_is_bound_byte_exactly(self):
        model = MoECausalLMModel(_moe_config(_mixed_quantization()))
        raw = _mixed_expert_state_dict()

        out = model.preprocess_weights(raw)

        prefix = f"{_LAYER}mlp."
        torch.testing.assert_close(
            out[prefix + "fc1_experts_weights"],
            raw[f"{_LAYER}mlp.experts.gate_up_proj_qweight"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            out[prefix + "fc2_experts_weights"],
            raw[f"{_LAYER}mlp.experts.down_proj_qweight"],
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize(
        ("fc1_sym", "fc2_sym"),
        [(False, True), (True, False), (False, False)],
    )
    def test_independent_zero_point_presence(self, fc1_sym, fc2_sym):
        model = MoECausalLMModel(
            _moe_config(_mixed_quantization(fc1_sym=fc1_sym, fc2_sym=fc2_sym))
        )
        out = model.preprocess_weights(
            _mixed_expert_state_dict(fc1_sym=fc1_sym, fc2_sym=fc2_sym)
        )
        prefix = f"{_LAYER}mlp."

        assert (prefix + "fc1_experts_zero_points" in out) is not fc1_sym
        assert (prefix + "fc2_experts_zero_points" in out) is not fc2_sym

    def test_odd_zero_point_byte_ceiling(self):
        hidden_size, intermediate_size, num_experts = 80, 48, 3
        quantization = _mixed_quantization(fc1_sym=False, fc2_sym=False)
        config = make_config(
            num_hidden_layers=1,
            hidden_size=hidden_size,
            intermediate_size=64,
            moe_intermediate_size=intermediate_size,
            num_local_experts=num_experts,
            num_experts_per_tok=2,
            num_attention_heads=5,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=32,
            quantization=quantization,
        )
        model = MoECausalLMModel(config)
        out = model.preprocess_weights(
            _mixed_expert_state_dict(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                fc1_sym=False,
                fc2_sym=False,
            )
        )
        prefix = f"{_LAYER}mlp."

        assert out[prefix + "fc1_experts_zero_points"].shape[-1] == 2
        assert out[prefix + "fc2_experts_zero_points"].shape[-1] == 2

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            ("missing_scale", "incomplete"),
            ("wrong_rank", "rank 3"),
            ("wrong_dtype", "must be uint8"),
            ("wrong_scale_dtype", "must be float16, bfloat16, or float32"),
            ("wrong_experts", "shape"),
            ("wrong_packed_bytes", "shape"),
            ("wrong_scale_geometry", "shape"),
        ],
    )
    def test_invalid_mixed_sidecars_fail_closed(self, mutation, message):
        state = _mixed_expert_state_dict()
        gate_weight = f"{_LAYER}mlp.experts.gate_up_proj_qweight"
        gate_scales = f"{_LAYER}mlp.experts.gate_up_proj_scales"
        if mutation == "missing_scale":
            del state[gate_scales]
        elif mutation == "wrong_rank":
            state[gate_weight] = state[gate_weight].flatten(1)
        elif mutation == "wrong_dtype":
            state[gate_weight] = state[gate_weight].to(torch.int8)
        elif mutation == "wrong_scale_dtype":
            state[gate_scales] = state[gate_scales].to(torch.float64)
        elif mutation == "wrong_experts":
            state[gate_weight] = state[gate_weight][:-1]
        elif mutation == "wrong_packed_bytes":
            state[gate_weight] = state[gate_weight][..., :-1]
        elif mutation == "wrong_scale_geometry":
            state[gate_scales] = state[gate_scales][..., :-1]

        model = MoECausalLMModel(_moe_config(_mixed_quantization()))
        with pytest.raises(ValueError, match=message):
            model.preprocess_weights(state)

    def test_unknown_packed_sidecar_under_recognized_root_fails_closed(self):
        state = _mixed_expert_state_dict()
        state[f"{_LAYER}mlp.experts.gate_proj_qweight"] = torch.zeros(
            _E, _INT, _H // 4, dtype=torch.uint8
        )
        model = MoECausalLMModel(_moe_config(_mixed_quantization()))

        with pytest.raises(ValueError, match="unknown or stale packed expert sidecar"):
            model.preprocess_weights(state)

    def test_unknown_packed_sidecar_root_fails_closed(self):
        state = _mixed_expert_state_dict()
        state["model.layers.9.mlp.experts.gate_up_proj_qweight"] = torch.zeros(
            _E, 2 * _INT, _H // 4, dtype=torch.uint8
        )
        model = MoECausalLMModel(_moe_config(_mixed_quantization()))

        with pytest.raises(ValueError, match="only supports fused Olive K-last"):
            model.preprocess_weights(state)

    def test_olive_substring_exclusion_prevents_partial_qmoe(self):
        quantization = dataclasses.replace(
            _mixed_quantization(),
            modules_to_not_convert=("experts.gate_up_proj",),
        )

        with pytest.raises(ValueError, match="both routed expert projections"):
            MoECausalLMModel(_moe_config(quantization))

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            ("missing", "incomplete"),
            ("shape", "qzeros shape"),
            ("dtype", "qzeros must be uint8"),
        ],
    )
    def test_invalid_zero_point_sidecars_fail_closed(self, mutation, message):
        quantization = _mixed_quantization(fc1_sym=False)
        state = _mixed_expert_state_dict(fc1_sym=False)
        key = f"{_LAYER}mlp.experts.gate_up_proj_qzeros"
        if mutation == "missing":
            del state[key]
        elif mutation == "shape":
            state[key] = torch.cat((state[key], state[key]), dim=-1)
        elif mutation == "dtype":
            state[key] = state[key].to(torch.int8)

        model = MoECausalLMModel(_moe_config(quantization))
        with pytest.raises(ValueError, match=message):
            model.preprocess_weights(state)

    def test_mismatched_group_size_fails_at_construction(self):
        with pytest.raises(ValueError, match="common FC1/FC2 group_size"):
            MoECausalLMModel(_moe_config(_mixed_quantization(fc1_group_size=32)))

    def test_uniform_int4_omits_projection_specific_attributes(self):
        config = _moe_config(_quantization())
        model = MoECausalLMModel(config)

        from mobius._builder import build_from_module

        graph = build_from_module(model, config)["model"].graph
        qmoe = next(node for node in graph if node.op_type == "QMoE")
        assert "fc1_expert_weight_bits" not in qmoe.attributes
        assert "fc2_expert_weight_bits" not in qmoe.attributes
        assert "fc3_expert_weight_bits" not in qmoe.attributes

    def test_uniform_int4_experts_do_not_bypass_unsupported_global_fallback(self):
        quantization = QuantizationConfig(
            bits=2,
            group_size=_BLK,
            quant_method="olive",
            overrides={
                f"{_LAYER}mlp.experts.gate_up_proj": QuantizationOverride(bits=4),
                f"{_LAYER}mlp.experts.down_proj": QuantizationOverride(bits=4),
            },
        )

        block = MoECausalLMModel(_moe_config(quantization)).model.layers[0].mlp

        assert block._qmoe_quantization is None
        assert block.experts is not None

    def test_model_package_roundtrip_preserves_mixed_attrs_and_bytes(self, tmp_path):
        from mobius._builder import build_from_module

        config = _moe_config(_mixed_quantization())
        module = MoECausalLMModel(config)
        package = build_from_module(module, config)
        processed = module.preprocess_weights(_mixed_expert_state_dict())
        package.apply_weights(processed, fold_constants=False)
        names = (
            f"{_LAYER}mlp.fc1_experts_weights",
            f"{_LAYER}mlp.fc2_experts_weights",
        )
        expected = {
            name: package["model"].graph.initializers[name].const_value.numpy().tobytes()
            for name in names
        }

        package.save(str(tmp_path), check_weights=False, progress_bar=False)
        reloaded = ModelPackage.load(str(tmp_path))["model"]
        qmoe = next(node for node in reloaded.graph if node.op_type == "QMoE")

        assert qmoe.attributes["fc1_expert_weight_bits"].value == 2
        assert qmoe.attributes["fc2_expert_weight_bits"].value == 4
        assert qmoe.attributes["fc3_expert_weight_bits"].value == 2
        for name, payload in expected.items():
            assert reloaded.graph.initializers[name].const_value.numpy().tobytes() == payload

    @pytest.mark.parametrize(
        ("quantization", "message"),
        [
            (
                QuantizationConfig(
                    bits=4,
                    group_size=_BLK,
                    quant_method="gptq",
                    overrides={
                        f"{_LAYER}mlp.experts.gate_up_proj": QuantizationOverride(bits=2)
                    },
                ),
                "requires Olive integer-affine",
            ),
            (
                QuantizationConfig(
                    bits=4,
                    group_size=_BLK,
                    quant_method="olive",
                    weight_format=QuantizedWeightFormat.MXFP4,
                    overrides={
                        f"{_LAYER}mlp.experts.gate_up_proj": QuantizationOverride(bits=2)
                    },
                ),
                "requires Olive integer-affine",
            ),
            (
                QuantizationConfig(
                    bits=4,
                    group_size=_BLK,
                    quant_method="olive",
                    overrides={
                        f"{_LAYER}mlp.experts.gate_up_proj": QuantizationOverride(bits=8)
                    },
                ),
                "expected model-wide INT4 with FC1 INT2 and FC2 INT4",
            ),
        ],
    )
    def test_unsupported_mixed_layout_fails_at_construction(self, quantization, message):
        with pytest.raises(ValueError, match=message):
            MoECausalLMModel(_moe_config(quantization))

    def test_missing_entire_routed_layer_fails_closed(self):
        quantization = _mixed_quantization(
            pattern=r"re:model\.layers\.\d+\.mlp\.experts\.gate_up_proj"
        )
        config = dataclasses.replace(
            _moe_config(quantization),
            num_hidden_layers=2,
        )
        model = MoECausalLMModel(config)

        with pytest.raises(ValueError, match=r"model\.layers\.1\.mlp.*incomplete"):
            model.preprocess_weights(_mixed_expert_state_dict())

    def test_missing_all_routed_expert_sidecars_fails_closed(self):
        model = MoECausalLMModel(_moe_config(_mixed_quantization()))

        with pytest.raises(ValueError, match=r"model\.layers\.0\.mlp.*incomplete"):
            model.preprocess_weights({})
