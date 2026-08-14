# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build/weight tests for Qwen3.5-MoE export.

Focuses on the ``com.microsoft::QMoE`` emission path: when the quantization
config matches the native QMoE ABI (blk32 int4 Olive/GPTQ/AWQ),
:meth:`Qwen35MoECausalLMModel.preprocess_weights` keeps the fused expert-major
tensors and repacks them into ``fc1``/``fc2`` QMoE parameters (mirroring
DeepSeek-V3), instead of un-fusing into a per-expert dense fallback. All tiny
random configs -- no checkpoint download.
"""

from __future__ import annotations

import torch

from mobius._configs import QuantizationConfig
from mobius._testing import make_config
from mobius.models.qwen35 import Qwen35MoECausalLMModel

_E, _H, _INT, _BLK, _BITS = 8, 32, 16, 16, 4
_FC1_OUT = 2 * _INT


def _moe_config(quantization: QuantizationConfig | None) -> object:
    return make_config(
        num_hidden_layers=1,
        hidden_size=_H,
        intermediate_size=64,
        moe_intermediate_size=_INT,
        shared_expert_intermediate_size=_INT,
        num_local_experts=_E,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        quantization=quantization,
    )


def _olive_expert_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic HF-style fused Olive-quantized MoE expert tensors.

    Olive's on-disk suffix convention is an *underscore* suffix directly on
    the parameter name (``<pname>_qweight``/``_scales``/``_qzeros``), not a
    dotted one -- see ``olive/common/quant/state_dict.py``. For a fused MoE
    parameter like ``gate_up_proj`` (no nested ``nn.Linear``), this means
    ``experts.gate_up_proj_qweight``, not ``experts.gate_up_proj.qweight``.
    """
    p = "model.language_model.layers.0.mlp."
    return {
        p + "experts.gate_up_proj_qweight": torch.randint(
            0, 256, (_E, _FC1_OUT, _H * _BITS // 8), dtype=torch.uint8
        ),
        p + "experts.gate_up_proj_scales": torch.rand(_E, _FC1_OUT, _H // _BLK),
        p + "experts.gate_up_proj_qzeros": torch.randint(
            0, 256, (_E, _FC1_OUT, 1), dtype=torch.uint8
        ),
        p + "experts.down_proj_qweight": torch.randint(
            0, 256, (_E, _H, _INT * _BITS // 8), dtype=torch.uint8
        ),
        p + "experts.down_proj_scales": torch.rand(_E, _H, _INT // _BLK),
        p + "experts.down_proj_qzeros": torch.randint(0, 256, (_E, _H, 1), dtype=torch.uint8),
        p + "gate.weight": torch.rand(_E, _H),
    }


class TestQwen35MoEQMoEExport:
    def test_moe_block_uses_qmoe_when_quantized(self):
        model = _moe_config(
            QuantizationConfig(bits=4, group_size=_BLK, quant_method="olive", sym=False)
        )
        model = Qwen35MoECausalLMModel(model)
        block = model.model.layers[0].mlp
        # Fused QMoE mode: no per-expert MLP ModuleList.
        assert block.experts is None
        assert hasattr(block, "fc1_experts_weights")

    def test_olive_preprocess_packs_qmoe_and_binds(self):
        config = _moe_config(
            QuantizationConfig(bits=4, group_size=_BLK, quant_method="olive", sym=False)
        )
        model = Qwen35MoECausalLMModel(config)
        out = model.preprocess_weights(_olive_expert_state_dict())

        prefix = "model.layers.0.mlp."
        assert out[prefix + "fc1_experts_weights"].shape == (_E, _FC1_OUT, _H * _BITS // 8)
        assert out[prefix + "fc1_scales"].shape == (_E, _FC1_OUT, _H // _BLK)
        assert out[prefix + "fc1_experts_zero_points"].shape == (_E, _FC1_OUT, 1)
        assert out[prefix + "fc2_experts_weights"].shape == (_E, _H, _INT * _BITS // 8)
        assert out[prefix + "fc2_scales"].shape == (_E, _H, _INT // _BLK)
        assert out[prefix + "fc2_experts_zero_points"].shape == (_E, _H, 1)

        # No per-expert dense-fallback storm leaked through.
        assert not any(".mlp.experts." in k for k in out)

        # Packed keys bind to real model parameters (weights actually load).
        param_names = {n for n, _ in model.named_parameters()}
        for suffix in (
            "fc1_experts_weights",
            "fc1_scales",
            "fc1_experts_zero_points",
            "fc2_experts_weights",
            "fc2_scales",
            "fc2_experts_zero_points",
        ):
            assert prefix + suffix in param_names

    def test_unquantized_preprocess_unfuses_dense_fallback(self):
        config = _moe_config(None)
        model = Qwen35MoECausalLMModel(config)
        p = "model.language_model.layers.0.mlp."
        fused = {
            p + "experts.gate_up_proj": torch.rand(_E, _FC1_OUT, _H),
            p + "experts.down_proj": torch.rand(_E, _H, _INT),
        }
        out = model.preprocess_weights(fused)

        # Dense fallback un-fuses into per-expert gate/up/down tensors.
        assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (_INT, _H)
        assert out["model.layers.0.mlp.experts.0.up_proj.weight"].shape == (_INT, _H)
        assert out[f"model.layers.0.mlp.experts.{_E - 1}.down_proj.weight"].shape == (
            _H,
            _INT,
        )
        assert not any("fc1_experts_weights" in k for k in out)
