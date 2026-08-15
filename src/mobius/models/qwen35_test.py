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

import numpy as np
import onnx_ir as ir
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

from mobius._configs import ArchitectureConfig, QuantizationConfig, Qwen35MtpConfig
from mobius._registry import registry
from mobius._testing import make_config
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.qwen35 import Qwen35MoECausalLMModel, Qwen35VL3ModelCausalLMModel
from mobius.models.qwen_vl import Qwen3VLEmbeddingModel
from mobius.tasks import build_embedding_from_features

_E, _H, _INT, _BLK, _BITS = 8, 32, 16, 16, 4
_FC1_OUT = 2 * _INT
_QWEN38_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def _qwen38_config() -> Qwen3_5Config:
    layer_types = [
        layer_type
        for _ in range(16)
        for layer_type in (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        )
    ]
    return Qwen3_5Config(
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        text_config={
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
            "mtp_num_hidden_layers": 1,
            "tie_word_embeddings": False,
        },
        vision_config={
            "model_type": "qwen3_5",
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 5120,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": [],
        },
    )


class TestQwen38Alias:
    def test_exact_config_extracts_dense_hybrid_vl_architecture(self):
        hf_config = _qwen38_config()
        config = ArchitectureConfig.from_transformers(
            hf_config.text_config,
            parent_config=hf_config,
        )

        assert _QWEN38_REVISION == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        assert registry.get("qwen3_5") is Qwen35VL3ModelCausalLMModel
        assert registry.get("qwen3_5_vl") is Qwen35VL3ModelCausalLMModel
        assert registry.get_registration("qwen3_5").test_model_id == "Qwen/Qwen3.8-27B"
        assert registry.get_registration("qwen3_5_vl").test_model_id == "Qwen/Qwen3.5-2B"
        assert config.hidden_size == 5120
        assert config.intermediate_size == 17408
        assert config.num_hidden_layers == 64
        assert config.layer_types == hf_config.text_config.layer_types
        assert config.layer_types.count("linear_attention") == 48
        assert config.layer_types.count("full_attention") == 16
        assert config.num_attention_heads == 24
        assert config.num_key_value_heads == 4
        assert config.head_dim == 256
        assert np.isclose(config.partial_rotary_factor, 0.25)
        assert config.mrope_interleaved is True
        assert config.mrope_section == [11, 11, 10]
        assert config.linear_num_key_heads == 16
        assert config.linear_num_value_heads == 48
        assert config.linear_key_head_dim == 128
        assert config.linear_value_head_dim == 128
        assert config.linear_conv_kernel_dim == 4
        assert config.vocab_size == 248320
        assert config.image_token_id == 248056
        assert config.video_token_id == 248057
        assert config.vision_start_token_id == 248053
        assert config.vision_end_token_id == 248054
        assert config.vision is not None
        assert config.vision.num_hidden_layers == 27
        assert config.vision.hidden_size == 1152
        assert config.vision.intermediate_size == 4304
        assert config.vision.num_attention_heads == 16
        assert config.vision.patch_size == 16
        assert config.vision.temporal_patch_size == 2
        assert config.vision.spatial_merge_size == 2
        assert config.vision.out_hidden_size == 5120
        assert config.vision.deepstack_visual_indexes == []

    def test_one_layer_mtp_is_classified_as_separate_optional_drafter(self):
        hf_config = _qwen38_config()
        assert hf_config.text_config.mtp_num_hidden_layers == 1

        mtp_config = Qwen35MtpConfig.from_transformers(hf_config)
        assert mtp_config.num_hidden_layers == 1
        assert mtp_config.layer_types == ["full_attention"]
        assert registry.get_registration("Qwen35MtpModel").task == "qwen35-mtp"

    def test_weight_routing_excludes_separately_packaged_mtp(self):
        config = ArchitectureConfig.from_transformers(
            _qwen38_config().text_config,
            parent_config=_qwen38_config(),
        )
        model = Qwen35VL3ModelCausalLMModel(config)
        state_dict = {
            "model.language_model.embed_tokens.weight": torch.ones(2, 2),
            "model.language_model.layers.0.linear_attn.A_log": torch.ones(2),
            "model.language_model.layers.3.self_attn.q_proj.weight": torch.ones(2, 2),
            "model.visual.blocks.0.mlp.linear_fc1.weight": torch.ones(2, 2),
            "lm_head.weight": torch.ones(2, 2),
            "mtp.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
            "mtp.fc.weight": torch.ones(2, 2),
        }

        result = model.preprocess_weights(state_dict)

        assert "decoder.model.embed_tokens.weight" in result
        assert "embedding.embed_tokens.weight" in result
        assert "decoder.model.layers.0.linear_attn.A_log" in result
        assert "decoder.model.layers.3.self_attn.q_proj.weight" in result
        assert "vision_encoder.visual.blocks.0.mlp.up_proj.weight" in result
        assert "decoder.lm_head.weight" in result
        assert not any(key.startswith("mtp") or ".mtp." in key for key in result)

    def test_qwen_vl_processor_boundary_stays_float32_for_bf16_export(self):
        hf_config = _qwen38_config()
        config = ArchitectureConfig.from_transformers(
            hf_config.text_config,
            parent_config=hf_config,
        )
        config.dtype = ir.DataType.BFLOAT16
        package = Qwen35VL3ModelCausalLMModel(config)
        task = registry.get_registration("qwen3_5").task

        from mobius.tasks import get_task

        vision_model = get_task(task).build(package, config)["vision_encoder"]

        assert vision_model.graph.inputs[0].name == "pixel_values"
        assert vision_model.graph.inputs[0].dtype == ir.DataType.FLOAT
        assert any(node.op_type == "Cast" for node in vision_model.graph)

    def test_embedding_scatter_matches_separate_image_then_video_streams(self):
        config = ArchitectureConfig(
            vocab_size=16,
            hidden_size=4,
            pad_token_id=0,
            image_token_id=10,
            video_token_id=11,
            dtype=ir.DataType.FLOAT,
        )
        graph = build_embedding_from_features(
            Qwen3VLEmbeddingModel(config),
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
        )
        embedding_weight = np.arange(
            config.vocab_size * config.hidden_size,
            dtype=np.float32,
        ).reshape(config.vocab_size, config.hidden_size)
        for name, initializer in graph.graph.initializers.items():
            if name.endswith("embed_tokens.weight"):
                initializer.const_value = ir.tensor(embedding_weight)

        input_ids = np.array(
            [
                [config.video_token_id, 1, config.image_token_id],
                [2, config.image_token_id, config.video_token_id],
            ],
            dtype=np.int64,
        )
        # HF scatters the two image rows first, then the two video rows.
        media_features = np.arange(100, 116, dtype=np.float32).reshape(4, 4)
        session = OnnxModelSession(graph)
        result = session.run(
            {
                "input_ids": input_ids,
                "image_features": media_features,
            }
        )["inputs_embeds"]

        expected = embedding_weight[input_ids].copy()
        expected[0, 2] = media_features[0]
        expected[1, 1] = media_features[1]
        expected[0, 0] = media_features[2]
        expected[1, 2] = media_features[3]
        np.testing.assert_array_equal(result, expected)

        decode_ids = np.array([[3], [4]], dtype=np.int64)
        decode = session.run(
            {
                "input_ids": decode_ids,
                "image_features": np.empty((0, config.hidden_size), dtype=np.float32),
            }
        )["inputs_embeds"]
        session.close()
        np.testing.assert_array_equal(decode, embedding_weight[decode_ids])


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

    def test_unsupported_qmoe_quantization_with_packed_experts_raises(self):
        """Packed quantized expert weights + an unsupported QMoE ABI must raise.

        Regression test: ``group_size=24`` (not a power of two) makes
        ``_supported_qmoe_quantization`` reject the config, so
        ``preprocess_weights`` takes the dense-fallback branch. But that
        branch's fused-tensor unfuser only knows how to split *unquantized*
        float ``experts.gate_up_proj``/``experts.down_proj`` tensors -- there
        is no code path that splits packed ``_qweight``/``_scales``/
        ``_qzeros`` fused-expert tensors into per-expert quantized Linear
        initializers. Previously this silently fell through to
        ``cleaned[key] = value`` and produced a graph whose per-expert
        Linear modules expect keys that were never generated.
        """
        config = _moe_config(
            QuantizationConfig(bits=8, group_size=_BLK, quant_method="olive", sym=False)
        )
        model = Qwen35MoECausalLMModel(config)
        try:
            model.preprocess_weights(_olive_expert_state_dict())
        except ValueError as e:
            assert "QMoE ABI" in str(e)
        else:
            raise AssertionError("expected ValueError for unsupported QMoE + packed experts")
