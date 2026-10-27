# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Gemma4 preprocess_weights — MoE expert rename and router scale."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest
import torch

from mobius._component_quantization import normalize_component_quantized_weights
from mobius._configs import (
    AudioConfig,
    Gemma4AudioConfig,
    Gemma4Config,
    QuantizationConfig,
    QuantizationOverride,
)
from mobius.models.gemma4 import Gemma4CausalLMModel, Gemma4EmbeddingModel, Gemma4Model
from mobius.weights import adapt_model_weights


def _tiny_gemma4_config(**overrides) -> Gemma4Config:
    """Create a minimal Gemma4Config for preprocess_weights tests."""
    from mobius._configs import VisionConfig

    defaults = dict(
        model_type="gemma4",
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
        global_head_dim=32,
        hidden_act="gelu",
        enable_moe_block=True,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        layer_types=["sliding_attention", "full_attention"],
        attention_k_eq_v=True,
        num_global_key_value_heads=1,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=4,
        ),
    )
    defaults.update(overrides)
    return Gemma4Config(**defaults)


class TestGemma4HfPrecisionAlignment:
    @pytest.mark.parametrize(
        "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
    )
    @pytest.mark.parametrize("quantized", [False, True])
    @pytest.mark.parametrize("optimized", [False, True])
    def test_vision_accepts_float32_processor_pixels(self, dtype, quantized, optimized):
        from mobius._builder import _cast_module_dtype, build_from_module
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(
            dtype=dtype,
            enable_moe_block=False,
            quantization=(
                QuantizationConfig(
                    quant_method="olive", bits=4, group_size=16, quantize_vision=True
                )
                if quantized
                else None
            ),
        )
        module = Gemma4Model(config)
        _cast_module_dtype(module, dtype)
        package = (
            build_from_module(module, config, task="gemma4")
            if optimized
            else Gemma4Task().build(module, config)
        )
        graph = package["vision_encoder"].graph
        assert graph.inputs[0].name == "pixel_values"
        assert graph.inputs[0].dtype == ir.DataType.FLOAT
        projection = next(node for node in graph if node.op_type in {"MatMul", "MatMulNBits"})
        assert projection.inputs[0].dtype == dtype

    @pytest.mark.parametrize(
        ("dtype", "torch_dtype"),
        [
            (ir.DataType.FLOAT, torch.float32),
            (ir.DataType.FLOAT16, torch.float16),
            (ir.DataType.BFLOAT16, torch.bfloat16),
        ],
    )
    def test_pixel_normalization_before_projection_cast(self, dtype, torch_dtype):
        import numpy as np
        from onnx_ir import tensor_adapters

        from mobius._builder import _cast_module_dtype
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import _Gemma4VisionPatchEmbedder
        from mobius.tasks._base import _make_graph

        module = _Gemma4VisionPatchEmbedder(1, 4, 1)
        _cast_module_dtype(module, dtype)
        graph, builder = _make_graph()
        pixels = builder.input("pixel_values", ir.DataType.FLOAT, [1, 256, 3])
        positions = builder.input("positions", ir.DataType.INT64, [1, 256, 2])
        module(builder.op, pixels, positions)
        projection = next(node for node in graph if node.op_type == "MatMul")
        normalized = projection.inputs[0]
        # Execute only normalization: no CPU BF16 MatMul kernel is needed.
        normalized_f32 = builder.op.Cast(normalized, to=ir.DataType.FLOAT)
        builder.add_output(normalized_f32, "normalized")
        module.position_embedding_table.const_value = tensor_adapters.TorchTensor(
            torch.zeros(2, 1, 4, dtype=torch_dtype)
        )
        prefix = ir.convenience.extract(graph, [pixels], [normalized_f32])
        values = (torch.arange(256).float() / 255).reshape(1, 256, 1).expand(1, 256, 3)
        expected = (2 * (values - 0.5)).to(torch_dtype).float().numpy()
        actual = OnnxModelSession(ir.Model(prefix, ir_version=10), device="cpu").run(
            {"pixel_values": values.numpy().copy()}
        )["normalized"]
        np.testing.assert_array_equal(actual, expected)
        if dtype == ir.DataType.BFLOAT16:
            old = (values.to(torch_dtype) * 2 - 1).float().numpy()
            assert np.count_nonzero(old != expected) == 128 * 3

    @pytest.mark.parametrize(
        ("dtype", "torch_dtype"),
        [
            (ir.DataType.FLOAT, torch.float32),
            (ir.DataType.FLOAT16, torch.float16),
            (ir.DataType.BFLOAT16, torch.bfloat16),
        ],
    )
    def test_router_scalar_after_learned_scale(self, dtype, torch_dtype):
        import ml_dtypes
        import numpy as np

        from mobius._builder import _cast_module_dtype
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import _Gemma4MoeRouter
        from mobius.tasks._base import _make_graph

        # A non-power-of-two hidden size exposes BF16 scalar-rounding differences.
        generator = torch.Generator().manual_seed(7)
        normed = torch.randn(4, 2816, generator=generator).to(torch_dtype)
        scale = (0.2 + torch.rand(2816, generator=generator)).to(torch_dtype)
        learned = normed * scale
        expected = (learned * (2816**-0.5)).float().numpy()
        module = _Gemma4MoeRouter(2816, 4)
        _cast_module_dtype(module, dtype)
        graph, builder = _make_graph()
        hidden = builder.input("hidden", dtype, [4, 2816])
        module(builder.op, hidden)
        projection = next(node for node in graph if node.op_type == "MatMul")
        scaled = projection.inputs[0]
        assert scaled.producer().op_type == "CastLike"
        learned_value = scaled.producer().inputs[1]
        assert learned_value.producer().op_type == "Mul"
        assert learned_value.producer().inputs[0].producer().op_type == "RMSNormalization"
        output = builder.op.Cast(scaled, to=ir.DataType.FLOAT)
        builder.add_output(output, "scaled")
        prefix = ir.convenience.extract(graph, [learned_value], [output])
        np_dtype = {
            ir.DataType.FLOAT: np.float32,
            ir.DataType.FLOAT16: np.float16,
            ir.DataType.BFLOAT16: ml_dtypes.bfloat16,
        }[dtype]
        actual = OnnxModelSession(ir.Model(prefix, ir_version=10), device="cpu").run(
            {learned_value.name: learned.float().numpy().astype(np_dtype)}
        )["scaled"]
        np.testing.assert_array_equal(actual, expected)
        if dtype == ir.DataType.BFLOAT16:
            folded = normed * (scale * (2816**-0.5))
            assert np.count_nonzero(folded.float().numpy() != expected) > 0


class TestGemma4CausalLMPreprocessWeights:
    """Test Gemma4CausalLMModel.preprocess_weights."""

    def test_expert_weight_rename(self):
        config = _tiny_gemma4_config()
        model = Gemma4CausalLMModel(config)

        fake_sd = {
            "model.layers.0.experts.gate_up_proj": torch.zeros(4, 64, 64),
            "model.layers.0.experts.down_proj": torch.zeros(4, 64, 32),
        }
        result = model.preprocess_weights(fake_sd)

        assert "model.layers.0.fc1_experts_weights" in result
        assert "model.layers.0.fc2_experts_weights" in result
        assert "model.layers.0.experts.gate_up_proj" not in result

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_router_scale_preserved(self, dtype):
        config = _tiny_gemma4_config()
        model = Gemma4CausalLMModel(config)

        scale_val = torch.tensor([2.0], dtype=dtype)
        fake_sd = {"model.layers.0.router.scale": scale_val.clone()}
        result = model.preprocess_weights(fake_sd)

        torch.testing.assert_close(result["model.layers.0.router.scale"], scale_val)

    def test_quantized_moe_experts_fail_closed(self):
        config = _tiny_gemma4_config(
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
            )
        )
        state_dict = {
            "model.layers.0.experts.gate_up_proj_qweight": torch.zeros(
                4, 64, 32, dtype=torch.uint8
            )
        }

        with pytest.raises(NotImplementedError, match="Quantized Gemma4 MoE experts"):
            Gemma4CausalLMModel(config).preprocess_weights(state_dict)


class TestGemma4ModelPreprocessWeights:
    """Test Gemma4Model.preprocess_weights (multimodal path)."""

    def test_expert_weight_rename(self):
        config = _tiny_gemma4_config()
        model = Gemma4Model(config)

        fake_sd = {
            "model.language_model.layers.0.experts.gate_up_proj": torch.zeros(4, 64, 64),
            "model.language_model.layers.0.experts.down_proj": torch.zeros(4, 64, 32),
        }
        result = model.preprocess_weights(fake_sd)

        assert "decoder.model.layers.0.fc1_experts_weights" in result
        assert "decoder.model.layers.0.fc2_experts_weights" in result

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_router_scale_preserved(self, dtype):
        config = _tiny_gemma4_config()
        model = Gemma4Model(config)

        scale_val = torch.tensor([2.0], dtype=dtype)
        fake_sd = {"model.language_model.layers.0.router.scale": scale_val.clone()}
        result = model.preprocess_weights(fake_sd)

        key = "decoder.model.layers.0.router.scale"
        assert key in result
        torch.testing.assert_close(result[key], scale_val)

    def test_quantized_moe_experts_fail_closed(self):
        config = _tiny_gemma4_config(
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
            )
        )
        state_dict = {
            "model.language_model.layers.0.experts.down_proj.weight_scales": torch.ones(
                4, 64, 2
            )
        }

        with pytest.raises(NotImplementedError, match="Quantized Gemma4 MoE experts"):
            Gemma4Model(config).preprocess_weights(state_dict)

    def test_olive_quantized_decoder_sidecars_are_preprocessed(self):
        config = _tiny_gemma4_config(
            enable_moe_block=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
            ),
        )
        model = Gemma4Model(config)
        fake_sd = {
            "model.language_model.layers.0.self_attn.q_proj.weight_qweight": torch.zeros(
                64, 32, dtype=torch.uint8
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight_scales": torch.ones(64, 4),
        }

        result = model.preprocess_weights(fake_sd)

        weight_key = "decoder.model.layers.0.self_attn.q_proj.weight"
        scales_key = "decoder.model.layers.0.self_attn.q_proj.scales"
        assert result[weight_key].shape == (64, 4, 8)
        assert result[weight_key].dtype == torch.uint8
        assert (
            result[scales_key]
            is fake_sd["model.language_model.layers.0.self_attn.q_proj.weight_scales"]
        )

    def test_olive_quantized_vision_sidecars_are_preprocessed(self):
        config = _tiny_gemma4_config(
            enable_moe_block=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
                quantize_vision=True,
            ),
        )
        model = Gemma4Model(config)
        fake_sd = {
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_qweight": torch.zeros(
                32, 16, dtype=torch.uint8
            ),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_scales": torch.ones(
                32, 2
            ),
            "model.embed_vision.embedding_projection.weight_qweight": torch.zeros(
                64, 16, dtype=torch.uint8
            ),
            "model.embed_vision.embedding_projection.weight_scales": torch.ones(64, 2),
        }

        result = model.preprocess_weights(fake_sd)

        assert result["vision_encoder.encoder.layers.0.self_attn.q_proj.weight"].shape == (
            32,
            2,
            8,
        )
        assert result["vision_encoder.encoder.layers.0.self_attn.q_proj.scales"].shape == (
            32,
            2,
        )
        assert result["vision_encoder.projector.weight"].shape == (64, 2, 8)
        assert result["vision_encoder.projector.scales"].shape == (64, 2)

    @pytest.mark.parametrize("quant_method", ["gptq", "awq"])
    def test_gptq_awq_clippable_vision_sidecars_drop_linear_wrapper(self, quant_method):
        config = _tiny_gemma4_config(
            enable_moe_block=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method=quant_method,
                sym=True,
                quantize_vision=True,
            ),
        )
        model = Gemma4Model(config)
        fake_sd = {
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.qweight": torch.zeros(
                4, 32, dtype=torch.int32
            ),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.scales": torch.ones(
                2, 32
            ),
        }

        result = model.preprocess_weights(fake_sd)

        assert result["vision_encoder.encoder.layers.0.self_attn.q_proj.weight"].shape == (
            32,
            2,
            8,
        )
        assert result["vision_encoder.encoder.layers.0.self_attn.q_proj.scales"].shape == (
            32,
            2,
        )
        assert not any(".linear." in name for name in result)

    def test_top_level_lm_head_routes_to_decoder(self):
        model = Gemma4Model(_tiny_gemma4_config(tie_word_embeddings=False))
        weight = torch.randn(256, 64)

        result = model.preprocess_weights({"lm_head.weight": weight})

        assert result["decoder.lm_head.weight"] is weight

    @pytest.mark.parametrize(
        ("quantize_embeddings", "quantize_lm_head", "state_dict"),
        [
            (
                True,
                False,
                {
                    "model.language_model.embed_tokens.weight_qweight": torch.zeros(
                        256, 32, dtype=torch.uint8
                    )
                },
            ),
            (
                False,
                True,
                {"lm_head.weight_qweight": torch.zeros(256, 32, dtype=torch.uint8)},
            ),
        ],
    )
    def test_quantized_embedding_or_lm_head_fails_closed(
        self,
        quantize_embeddings,
        quantize_lm_head,
        state_dict,
    ):
        config = _tiny_gemma4_config(
            tie_word_embeddings=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
                quantize_embeddings=quantize_embeddings,
                quantize_lm_head=quantize_lm_head,
            ),
        )

        with pytest.raises(
            NotImplementedError,
            match="Quantized embeddings and LM heads are not yet supported",
        ):
            Gemma4Model(config).preprocess_weights(state_dict)

    def test_gguf_quantized_tables_keep_canonical_weight_path(self):
        config = _tiny_gemma4_config(
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="gguf",
                sym=True,
                quantize_embeddings=True,
                quantize_lm_head=True,
            ),
        )

        assert Gemma4Model(config).preprocess_weights({}) == {}

    def test_per_expert_scale_not_folded(self):
        """router.per_expert_scale should NOT be multiplied by scale_factor."""
        config = _tiny_gemma4_config()
        model = Gemma4Model(config)

        fake_sd = {
            "model.language_model.layers.0.router.per_expert_scale": torch.ones(4),
        }
        result = model.preprocess_weights(fake_sd)

        key = "decoder.model.layers.0.router.per_expert_scale"
        assert key in result
        assert torch.allclose(result[key], torch.ones(4))


class TestGemma4EmbeddingModel:
    def test_reuses_token_id_fields_for_masking(self):
        config = _tiny_gemma4_config(
            image_token_id=200010,
            audio=AudioConfig(audio_token_id=200011),
        )
        model = Gemma4EmbeddingModel(config)

        assert model.image_token_id == 200010
        assert model.audio_token_id == 200011
        assert not hasattr(model, "_image_token_id_mask")
        assert not hasattr(model, "_audio_token_id_mask")


class TestGemma4PerLayerInputLayout:
    @pytest.mark.parametrize(
        ("execution_provider", "expected_rank"),
        [("default", 3), ("onnx-standard", 3), ("qnn", 3), ("openvino", 4)],
    )
    def test_layout_matches_execution_provider(self, execution_provider, expected_rank):
        from mobius._builder import build_from_module
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(
            enable_moe_block=False,
            hidden_size_per_layer_input=8,
            vocab_size_per_layer_input=256,
            num_kv_shared_layers=0,
        )
        package = build_from_module(
            Gemma4Model(config),
            config,
            task=Gemma4Task(),
            execution_provider=execution_provider,
        )
        decoder_input = next(
            value
            for value in package["decoder"].graph.inputs
            if value.name == "per_layer_inputs"
        )
        embedding_output = next(
            value
            for value in package["embedding"].graph.outputs
            if value.name == "per_layer_inputs"
        )

        assert len(decoder_input.shape) == expected_rank
        assert len(embedding_output.shape) == expected_rank
        if expected_rank == 4:
            assert list(decoder_input.shape[-2:]) == [
                config.num_hidden_layers,
                config.hidden_size_per_layer_input,
            ]
            assert list(embedding_output.shape[-2:]) == [
                config.num_hidden_layers,
                config.hidden_size_per_layer_input,
            ]
            assert not any(
                node.op_type == "Reshape"
                and any(value is decoder_input for value in node.inputs if value is not None)
                for node in package["decoder"].graph
            )
        else:
            assert (
                decoder_input.shape[-1]
                == config.num_hidden_layers * config.hidden_size_per_layer_input
            )
            assert (
                embedding_output.shape[-1]
                == config.num_hidden_layers * config.hidden_size_per_layer_input
            )

        producer = {
            id(value): node for node in package["decoder"].graph for value in node.outputs
        }
        slice_squeezes = [
            node
            for node in package["decoder"].graph
            if node.op_type == "Squeeze"
            and node.inputs
            and node.inputs[0] is not None
            and (parent := producer.get(id(node.inputs[0]))) is not None
            and parent.op_type == "Slice"
        ]
        if execution_provider == "openvino":
            assert not slice_squeezes
        elif execution_provider == "qnn":
            assert len(slice_squeezes) >= config.num_hidden_layers


class TestGemma4VisionQuantization:
    def test_quantize_vision_emits_matmulnbits_and_keeps_activation_clipping(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(
            enable_moe_block=False,
            vision=dataclasses.replace(
                _tiny_gemma4_config().vision,
                use_clipped_linears=True,
            ),
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
                quantize_vision=True,
            ),
        )

        graph = Gemma4Task().build(Gemma4Model(config), config)["vision_encoder"].graph
        op_types = [node.op_type for node in graph]

        assert op_types.count("MatMulNBits") == 9
        assert op_types.count("Clip") >= 16

    def test_quantized_decoder_does_not_quantize_vision_by_default(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(
            enable_moe_block=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
            ),
        )

        graph = Gemma4Task().build(Gemma4Model(config), config)["vision_encoder"].graph

        assert not any(node.op_type == "MatMulNBits" for node in graph)

    def test_unrelated_module_plan_does_not_quantize_vision(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(
            enable_moe_block=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
                modules_to_not_convert=("lm_head",),
            ),
        )

        graph = Gemma4Task().build(Gemma4Model(config), config)["vision_encoder"].graph

        assert not any(node.op_type == "MatMulNBits" for node in graph)

    def test_top_level_vision_override_does_not_enable_vision_quantization(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(
            enable_moe_block=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
                quantize_vision=False,
                overrides={
                    "model.vision_tower": {
                        "bits": 4,
                        "group_size": 16,
                        "sym": True,
                    }
                },
            ),
        )

        graph = Gemma4Task().build(Gemma4Model(config), config)["vision_encoder"].graph

        assert not any(node.op_type == "MatMulNBits" for node in graph)


class TestGemma4PerComponentQuantization:
    @staticmethod
    def _config() -> Gemma4Config:
        decoder = QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="olive",
            sym=True,
        )
        return _tiny_gemma4_config(
            enable_moe_block=False,
            hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=256,
            audio=Gemma4AudioConfig(
                input_size=32,
                num_layers=1,
                hidden_size=32,
                output_proj_dims=64,
                subsampling_conv_channels=[16, 8],
                audio_token_id=254,
            ),
            quantization=decoder,
            component_quantization={
                "decoder": decoder,
                "vision_encoder": QuantizationConfig(
                    bits=8,
                    group_size=32,
                    quant_method="olive",
                    sym=True,
                ),
                "audio_encoder": QuantizationConfig(
                    bits=2,
                    group_size=16,
                    quant_method="olive",
                    sym=True,
                ),
                "embedding": QuantizationConfig(
                    bits=8,
                    group_size=16,
                    quant_method="olive",
                    sym=True,
                ),
            },
        )

    @staticmethod
    def _layouts(graph) -> set[tuple[int, int]]:
        return {
            (
                node.attributes["bits"].as_int(),
                node.attributes["block_size"].as_int(),
            )
            for node in graph
            if node.op_type == "MatMulNBits"
        }

    def test_each_component_emits_its_own_layout(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config()
        package = Gemma4Task().build(Gemma4Model(config), config)

        assert self._layouts(package["decoder"].graph) == {(4, 16)}
        assert self._layouts(package["vision_encoder"].graph) == {(8, 32)}
        assert self._layouts(package["audio_encoder"].graph) == {(2, 16)}
        assert self._layouts(package["embedding"].graph) == {(8, 16)}

    def test_preprocess_preserves_raw_component_sidecars(self):
        model = Gemma4Model(self._config())
        state_dict = {
            "model.language_model.layers.0.self_attn.q_proj.weight_qweight": torch.zeros(
                64, 32, dtype=torch.uint8
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight_scales": torch.ones(64, 4),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_qweight": torch.zeros(
                32, 32, dtype=torch.uint8
            ),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_scales": torch.ones(
                32, 1
            ),
            "model.audio_tower.layers.0.self_attn.q_proj.linear.weight_qweight": torch.zeros(
                32, 8, dtype=torch.uint8
            ),
            "model.audio_tower.layers.0.self_attn.q_proj.linear.weight_scales": torch.ones(
                32, 2
            ),
            "model.language_model.per_layer_model_projection.weight_qweight": torch.zeros(
                32, 64, dtype=torch.uint8
            ),
            "model.language_model.per_layer_model_projection.weight_scales": torch.ones(32, 4),
        }

        result = model.preprocess_weights(state_dict)

        source_to_target = {
            "model.language_model.layers.0.self_attn.q_proj": (
                "decoder.model.layers.0.self_attn.q_proj"
            ),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear": (
                "vision_encoder.encoder.layers.0.self_attn.q_proj"
            ),
            "model.audio_tower.layers.0.self_attn.q_proj.linear": (
                "audio_encoder.encoder.layers.0.self_attn.q_proj"
            ),
            "model.language_model.per_layer_model_projection": (
                "embedding.per_layer_model_projection"
            ),
        }
        for source, target in source_to_target.items():
            for suffix in ("weight_qweight", "weight_scales"):
                assert result[f"{target}.{suffix}"] is state_dict[f"{source}.{suffix}"]
            assert f"{target}.weight" not in result

    def test_component_override_uses_same_layout_for_graph_and_weights(self):
        from mobius.tasks._gemma4 import Gemma4Task

        override = QuantizationOverride(bits=8, group_size=32)
        vision_quantization = QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="olive",
            sym=True,
            overrides={
                "model.vision_tower": override,
                "model.embed_vision": override,
            },
        )
        config = self._config()
        config.component_quantization["vision_encoder"] = vision_quantization
        model = Gemma4Model(config)
        task = Gemma4Task()
        package = task.build(model, config)
        graph = package["vision_encoder"].graph

        result = model.preprocess_weights(
            {
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_qweight": torch.zeros(
                    32, 32, dtype=torch.uint8
                ),
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_scales": torch.ones(
                    32, 1
                ),
            }
        )
        result = normalize_component_quantized_weights(
            result, model, config, package.keys(), task=task
        )

        assert self._layouts(graph) == {(8, 32)}
        assert result["vision_encoder.encoder.layers.0.self_attn.q_proj.weight"].shape == (
            32,
            1,
            32,
        )


def test_component_regex_keeps_per_layer_decoder_projections_float():
    from mobius._component_quantization import configure_component_quantization
    from mobius.components import Linear, QuantizedLinear
    from mobius.tasks._gemma4 import Gemma4Task

    quantization = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        sym=True,
        modules_to_not_convert=(
            "lm_head",
            r"re:.*\.per_layer_input_gate",
            r"re:.*\.per_layer_projection",
        ),
    )
    config = _tiny_gemma4_config(
        enable_moe_block=False,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=256,
        quantization=quantization,
        component_quantization={"decoder": quantization},
    )
    module = Gemma4Model(config)

    configure_component_quantization(module, config, Gemma4Task())

    layer = module.decoder.model.layers[0]
    assert isinstance(layer.self_attn.q_proj, QuantizedLinear)
    assert type(layer.per_layer_input_gate) is Linear
    assert type(layer.per_layer_projection) is Linear


class TestGemma4ComponentWeightAdapters:
    @staticmethod
    def _config() -> Gemma4Config:
        decoder = QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="olive",
            sym=True,
            modules_to_not_convert=(
                r"re:.*\.per_layer_input_gate",
                r"re:.*\.per_layer_projection",
            ),
        )
        return _tiny_gemma4_config(
            enable_moe_block=False,
            hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=256,
            audio=Gemma4AudioConfig(
                input_size=32,
                num_layers=1,
                hidden_size=32,
                output_proj_dims=64,
                subsampling_conv_channels=[16, 8],
                audio_token_id=254,
            ),
            quantization=decoder,
            component_quantization={
                "decoder": decoder,
                "vision_encoder": QuantizationConfig(
                    bits=8,
                    group_size=32,
                    quant_method="olive",
                    sym=True,
                ),
                "audio_encoder": QuantizationConfig(
                    bits=2,
                    group_size=16,
                    quant_method="olive",
                    sym=True,
                ),
                "embedding": QuantizationConfig(
                    bits=8,
                    group_size=16,
                    quant_method="olive",
                    sym=True,
                ),
            },
        )

    @staticmethod
    def _layouts(graph) -> set[tuple[int, int]]:
        return {
            (
                node.attributes["bits"].as_int(),
                node.attributes["block_size"].as_int(),
            )
            for node in graph
            if node.op_type == "MatMulNBits"
        }

    def test_builds_each_component_with_its_declared_layout(self):
        from mobius._builder import build_from_module

        config = self._config()
        package = build_from_module(
            Gemma4Model(config),
            config,
            task="gemma4",
        )

        assert self._layouts(package["decoder"].graph) == {(4, 16)}
        assert self._layouts(package["vision_encoder"].graph) == {(8, 32)}
        assert self._layouts(package["audio_encoder"].graph) == {(2, 16)}
        assert self._layouts(package["embedding"].graph) == {(8, 16)}

    def test_vision_full_hf_linear_exclusion_matches_local_module(self):
        from mobius._component_quantization import configure_component_quantization
        from mobius.components import ClippableLinear, ClippableQuantizedLinear
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config()
        vision_quantization = dataclasses.replace(
            config.component_quantization["vision_encoder"],
            modules_to_not_convert=(
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
            ),
        )
        config = dataclasses.replace(
            config,
            vision=dataclasses.replace(
                config.vision,
                use_clipped_linears=True,
            ),
            component_quantization={
                **config.component_quantization,
                "vision_encoder": vision_quantization,
            },
        )
        module = Gemma4Model(config)

        configure_component_quantization(module, config, Gemma4Task())

        attention = module.vision_encoder.encoder.layers[0].self_attn
        assert type(attention.q_proj) is ClippableLinear
        assert isinstance(attention.k_proj, ClippableQuantizedLinear)

    def test_routes_raw_sidecars_to_each_component_codec(self):
        from mobius._component_quantization import (
            configure_component_quantization,
            normalize_component_quantized_weights,
        )
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config()
        module = Gemma4Model(config)
        task = Gemma4Task()
        manifest = configure_component_quantization(module, config, task)
        state_dict = {
            "model.language_model.layers.0.self_attn.q_proj.weight_qweight": torch.zeros(
                64, 32, dtype=torch.uint8
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight_scales": torch.ones(64, 4),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_qweight": torch.zeros(
                32, 32, dtype=torch.uint8
            ),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_scales": torch.ones(
                32, 1
            ),
            "model.audio_tower.layers.0.self_attn.q_proj.linear.weight_qweight": torch.zeros(
                32, 8, dtype=torch.uint8
            ),
            "model.audio_tower.layers.0.self_attn.q_proj.linear.weight_scales": torch.ones(
                32, 2
            ),
            "model.language_model.per_layer_model_projection.weight_qweight": torch.zeros(
                32, 64, dtype=torch.uint8
            ),
            "model.language_model.per_layer_model_projection.weight_scales": torch.ones(32, 4),
        }

        renamed = module.preprocess_weights(state_dict)
        result = normalize_component_quantized_weights(
            renamed,
            module,
            config,
            ("decoder", "vision_encoder", "audio_encoder", "embedding"),
            manifest=manifest,
            task=task,
        )

        assert result["decoder.model.layers.0.self_attn.q_proj.weight"].shape == (
            64,
            4,
            8,
        )
        assert result["vision_encoder.encoder.layers.0.self_attn.q_proj.weight"].shape == (
            32,
            1,
            32,
        )
        assert result["audio_encoder.encoder.layers.0.self_attn.q_proj.weight"].shape == (
            32,
            2,
            4,
        )
        assert result["embedding.per_layer_model_projection.weight"].shape == (
            32,
            4,
            16,
        )

    def test_quantized_embedding_sidecars_route_only_to_embedding_component(self):
        from mobius._component_quantization import (
            configure_component_quantization,
            normalize_component_quantized_weights,
        )
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config()
        embedding_quantization = dataclasses.replace(
            config.component_quantization["embedding"],
            quantize_embeddings=True,
        )
        config = dataclasses.replace(
            config,
            component_quantization={
                **config.component_quantization,
                "embedding": embedding_quantization,
            },
        )
        module = Gemma4Model(config)
        task = Gemma4Task()
        manifest = configure_component_quantization(module, config, task)
        renamed = module.preprocess_weights(
            {
                "model.language_model.embed_tokens.weight_qweight": torch.zeros(
                    256, 64, dtype=torch.uint8
                ),
                "model.language_model.embed_tokens.weight_scales": torch.ones(256, 4),
            }
        )

        assert not any(key.startswith("decoder.") for key in renamed)
        result = normalize_component_quantized_weights(
            renamed,
            module,
            config,
            ("decoder", "vision_encoder", "audio_encoder", "embedding"),
            manifest=manifest,
            task=task,
        )

        assert result["embedding.embed_tokens.qweight"].shape == (256, 64)
        assert result["embedding.embed_tokens.scales"].shape == (256, 4)

    def test_tied_quantized_embedding_materializes_decoder_lm_head(self):
        from mobius._component_quantization import (
            configure_component_quantization,
            normalize_component_quantized_weights,
        )
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config()
        decoder_quantization = dataclasses.replace(
            config.component_quantization["decoder"],
            quantize_lm_head=True,
            overrides={"lm_head": QuantizationOverride(bits=8)},
        )
        embedding_quantization = dataclasses.replace(
            config.component_quantization["embedding"],
            quantize_embeddings=True,
        )
        config = dataclasses.replace(
            config,
            tie_word_embeddings=True,
            component_quantization={
                **config.component_quantization,
                "decoder": decoder_quantization,
                "embedding": embedding_quantization,
            },
        )
        module = Gemma4Model(config)
        task = Gemma4Task()
        manifest = configure_component_quantization(module, config, task)
        qweight = torch.zeros(256, 64, dtype=torch.uint8)
        scales = torch.ones(256, 4)

        renamed = adapt_model_weights(
            module,
            {
                "model.language_model.embed_tokens.weight_qweight": qweight,
                "model.language_model.embed_tokens.weight_scales": scales,
            },
            config=config,
            manifest=manifest,
        )

        assert renamed["embedding.embed_tokens.weight_qweight"] is qweight
        assert renamed["embedding.embed_tokens.weight_scales"] is scales
        assert renamed["decoder.lm_head.weight_qweight"] is qweight
        assert renamed["decoder.lm_head.weight_scales"] is scales

        result = normalize_component_quantized_weights(
            renamed,
            module,
            config,
            ("decoder", "vision_encoder", "audio_encoder", "embedding"),
            manifest=manifest,
            task=task,
        )

        assert "embedding.embed_tokens.qweight" in result
        assert "embedding.embed_tokens.scales" in result
        assert "decoder.lm_head.weight" in result
        assert "decoder.lm_head.scales" in result


class TestScaleFreeRMSNormOverflow:
    """V norm should handle FP16 overflow from squaring large values."""

    def test_vnorm_fp16_no_nan(self):
        """Values ~888 overflow FP16 when squared (888²=788K > 65504).

        The scale-free RMSNorm must use stash_type=1 (float32 accumulation)
        to avoid inf/NaN from the variance computation.
        """
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.components import ScaleFreeRMSNorm

        dim = 64
        norm = ScaleFreeRMSNorm(dim, eps=1e-6)

        # Build a minimal ONNX graph for the norm
        from mobius.tasks._base import _make_graph, _make_model

        graph, builder = _make_graph()
        op = builder.op
        x = builder.input("x", dtype=ir.DataType.FLOAT16, shape=[1, 4, dim])
        y = norm(op, x)
        builder.add_output(y, "y")
        model = _make_model(graph)

        session = OnnxModelSession(model, device="cpu")

        # Values that overflow FP16 when squared: 888² = 788,544 > 65504
        test_input = np.full((1, 4, dim), 888.0, dtype=np.float16)
        result = session.run({"x": test_input})
        output = result["y"]

        assert not np.any(np.isnan(output)), "V norm produced NaN for input 888"
        assert not np.any(np.isinf(output)), "V norm produced Inf for input 888"
        # RMSNorm of a constant vector: x/rms(x) = sign(x) ≈ 1.0
        np.testing.assert_allclose(
            output.astype(np.float32),
            np.ones_like(output, dtype=np.float32),
            atol=0.01,
        )


class TestGemma4BlockSequenceIds:
    """Vision-block bidirectional attention wiring (use_bidirectional_attention)."""

    def test_compute_block_sequence_ids_values(self):
        """``_compute_block_sequence_ids`` matches HF get_block_sequence_ids_for_mask."""
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import _compute_block_sequence_ids
        from mobius.tasks._base import _make_graph, _make_model

        graph, builder = _make_graph()
        op = builder.op
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[1, "S"])
        out = _compute_block_sequence_ids(op, input_ids, image_token_id=255)
        builder.add_output(out, "block_sequence_ids")
        session = OnnxModelSession(_make_model(graph), device="cpu")

        # text  img img  text  aud aud  text  img  text
        ids = np.array([[10, 255, 255, 11, 254, 254, 11, 255, 12]], dtype=np.int64)
        result = session.run({"input_ids": ids})["block_sequence_ids"]
        # Only image runs form blocks (groups 0, 1). Audio (254) and text -> -1,
        # matching HF where audio is token-type 3 and excluded from is_vision.
        expected = np.array([[-1, 0, 0, -1, -1, -1, -1, 1, -1]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_unsupported_bidirectional_mode_raises(self):
        """``use_bidirectional_attention='all'`` is rejected, not silently causal."""
        import pytest

        from mobius.models.gemma4 import Gemma4TextModel

        config = _tiny_gemma4_config(use_bidirectional_attention="all")
        with pytest.raises(NotImplementedError, match="use_bidirectional_attention"):
            Gemma4TextModel(config)

    def test_audio_tokens_excluded_from_blocks(self):
        """Audio placeholders never join a vision block (HF parity).

        HF derives ``is_vision`` from ``mm_token_type_ids`` as ``(==1)|(==2)``
        (image or video); audio is token-type ``3`` and is excluded, so an audio
        run adjacent to an image run does NOT extend the block.
        """
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import _compute_block_sequence_ids
        from mobius.tasks._base import _make_graph, _make_model

        graph, builder = _make_graph()
        op = builder.op
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[1, "S"])
        out = _compute_block_sequence_ids(op, input_ids, image_token_id=255)
        builder.add_output(out, "block_sequence_ids")
        session = OnnxModelSession(_make_model(graph), device="cpu")

        ids = np.array([[10, 255, 255, 254, 254, 11]], dtype=np.int64)
        result = session.run({"input_ids": ids})["block_sequence_ids"]
        # Image run -> block 0; the adjacent audio run (254) stays -1 (causal).
        expected = np.array([[-1, 0, 0, -1, -1, -1]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_package_wires_block_sequence_ids_end_to_end(self):
        """Decoder takes input_ids and derives the vision-block overlay itself.

        With ``use_bidirectional_attention='vision'`` the embedding no longer
        emits ``block_sequence_ids``; instead the decoder receives ``input_ids``
        (alongside ``inputs_embeds``) and computes the overlay internally. This
        avoids a cross-model tensor that onnxruntime-genai cannot forward.
        """
        from mobius.models.gemma4 import Gemma4Model
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(use_bidirectional_attention="vision", image_token_id=255)
        pkg = Gemma4Task().build(Gemma4Model(config), config)

        emb_outputs = {o.name for o in pkg["embedding"].graph.outputs}
        dec_inputs = {i.name for i in pkg["decoder"].graph.inputs}
        assert "block_sequence_ids" not in emb_outputs
        assert "block_sequence_ids" not in dec_inputs
        assert "input_ids" in dec_inputs

        # Only local layers disable the built-in causal mask for the image overlay.
        dec = pkg["decoder"].graph
        assert not any(n.op_type == "GroupQueryAttention" for n in dec)
        attn_nodes = [n for n in dec if n.op_type == "Attention"]
        assert len(attn_nodes) == len(config.layer_types)
        for node, layer_type in zip(attn_nodes, config.layer_types, strict=True):
            expected = 0 if layer_type == "sliding_attention" else 1
            assert node.attributes["is_causal"].as_int() == expected
        # A global mask must have no data dependency on the image block overlay.
        global_bias = attn_nodes[1].inputs[3]
        pending = [global_bias]
        visited = set()
        while pending:
            value = pending.pop()
            if value is None or value.producer() is None:
                continue
            producer = value.producer()
            if producer in visited:
                continue
            visited.add(producer)
            assert producer.op_type != "Or"
            pending.extend(producer.inputs)

    def test_no_block_sequence_ids_when_causal(self):
        """Without bidirectional attention, no input_ids/overlay is wired."""
        from mobius.models.gemma4 import Gemma4Model
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(use_bidirectional_attention=None)
        pkg = Gemma4Task().build(Gemma4Model(config), config)

        emb_outputs = {o.name for o in pkg["embedding"].graph.outputs}
        dec_inputs = {i.name for i in pkg["decoder"].graph.inputs}
        assert "block_sequence_ids" not in emb_outputs
        assert "block_sequence_ids" not in dec_inputs
        assert "input_ids" not in dec_inputs

    @pytest.mark.parametrize("mask", [[1, 0, 1, 1], [1, 1, 1, 0]])
    @pytest.mark.parametrize("text_task", [False, True], ids=["split", "text"])
    def test_cuda_export_preserves_global_cached_mask_holes(self, mask, text_task):
        import numpy as np

        from mobius import build_from_module
        from mobius._pipeline_contract import requires_arbitrary_attention_mask
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.tasks._gemma4 import Gemma4Task, Gemma4TextCausalLMTask

        config = _tiny_gemma4_config(
            dtype=ir.DataType.FLOAT16,
            enable_moe_block=False,
            use_bidirectional_attention="vision",
            image_token_id=255,
        )
        module = Gemma4CausalLMModel(config) if text_task else Gemma4Model(config)
        task = Gemma4TextCausalLMTask() if text_task else Gemma4Task()
        package = build_from_module(module, config, task=task, execution_provider="cuda")
        decoder = package["model" if text_task else "decoder"]
        assert requires_arbitrary_attention_mask(decoder.graph)
        assert not any(node.op_type == "GroupQueryAttention" for node in decoder.graph)
        global_attention = next(
            node
            for node in decoder.graph
            if node.op_type == "Attention"
            and node.attributes["kv_num_heads"].as_int() == config.num_global_key_value_heads
        )
        assert global_attention.attributes["is_causal"].as_int() == 1
        bias = global_attention.inputs[3]
        assert bias is not None and bias.name is not None
        # Execute the actual optimized bias with one query and three past slots.
        # Equal token counts must not collapse these distinct visible-key sets.
        inputs = [
            value
            for value in decoder.graph.inputs
            if value.name in {"input_ids", "attention_mask"}
        ]
        graph = ir.convenience.extract(decoder.graph, inputs, [bias])
        session = OnnxModelSession(ir.Model(graph, ir_version=10), device="cpu")
        try:
            actual = session.run(
                {
                    "input_ids": np.array([[5]], dtype=np.int64),
                    "attention_mask": np.array([mask], dtype=np.int64),
                }
            )[bias.name]
        finally:
            session.close()
        np.testing.assert_array_equal(actual[0, 0, 0] > -1, np.array(mask, dtype=bool))

    def test_text_only_cuda_export_keeps_gqa_eligibility(self):
        from mobius import build_from_module
        from mobius._pipeline_contract import requires_arbitrary_attention_mask

        config = _tiny_gemma4_config(
            model_type="gemma4_text",
            dtype=ir.DataType.FLOAT16,
            enable_moe_block=False,
            vision=None,
            image_token_id=None,
            use_bidirectional_attention=None,
        )
        package = build_from_module(
            Gemma4CausalLMModel(config),
            config,
            task="gemma4-text-generation",
            execution_provider="cuda",
        )
        graph = package["model"].graph
        assert not requires_arbitrary_attention_mask(graph)
        assert (
            sum(node.op_type == "GroupQueryAttention" for node in graph)
            == config.num_hidden_layers
        )


def _tiny_gemma4_unified_config(**overrides) -> Gemma4Config:
    """Minimal gemma4_unified config (dense decoder + encoder-free embedders)."""
    from mobius._configs import Gemma4AudioConfig, VisionConfig

    base = dict(
        model_type="gemma4_unified",
        enable_moe_block=False,
        tie_word_embeddings=True,
        use_bidirectional_attention="vision",
        image_token_id=255,
        vision=VisionConfig(
            hidden_size=32,
            position_embedding_size=1120,
            patch_size=4,
            pooling_kernel_size=3,
            out_hidden_size=32,
            norm_eps=1e-6,
        ),
        audio=Gemma4AudioConfig(hidden_size=16, output_proj_dims=16, audio_token_id=254),
    )
    base.update(overrides)
    return _tiny_gemma4_config(**base)


class TestGemma4UnifiedPreprocessWeights:
    """Gemma4UnifiedModel.preprocess_weights — checkpoint name mapping."""

    def test_full_checkpoint_rename(self):
        from mobius.models.gemma4 import Gemma4UnifiedModel

        config = _tiny_gemma4_unified_config()
        model = Gemma4UnifiedModel(config)

        # pos_embedding stored as [posemb, 2, mm_embed_dim]; x-axis = [:, 0, :],
        # y-axis = [:, 1, :]. Use distinct constants to verify the split.
        pos_embedding = torch.empty(1120, 2, 32)
        pos_embedding[:, 0, :] = 1.0
        pos_embedding[:, 1, :] = 2.0

        fake_sd = {
            "model.language_model.embed_tokens.weight": torch.zeros(256, 64),
            "model.language_model.layers.0.input_layernorm.weight": torch.zeros(64),
            "model.vision_embedder.patch_ln1.weight": torch.zeros(432),
            "model.vision_embedder.patch_dense.weight": torch.zeros(32, 432),
            "model.vision_embedder.pos_embedding": pos_embedding,
            "model.vision_embedder.pos_norm.weight": torch.zeros(32),
            "model.embed_vision.embedding_projection.weight": torch.zeros(64, 32),
            "model.embed_audio.embedding_projection.weight": torch.zeros(64, 16),
            # Scale-free RMSNorms have no learnable weight in the checkpoint, but
            # assert they are dropped even if a stray key appears.
            "model.embed_vision.embedding_pre_projection_norm.weight": torch.zeros(32),
            "model.embed_audio.embedding_pre_projection_norm.weight": torch.zeros(16),
        }
        result = model.preprocess_weights(fake_sd)

        # Text backbone → decoder.model.* and tied embedding/lm_head.
        assert "decoder.model.embed_tokens.weight" in result
        assert "embedding.embed_tokens.weight" in result
        assert "decoder.lm_head.weight" in result  # synthesized from tied embed
        assert "decoder.model.layers.0.input_layernorm.weight" in result

        # Vision front-end → vision_encoder.*; pos_embedding split into x/y.
        assert "vision_encoder.patch_ln1.weight" in result
        assert "vision_encoder.patch_dense.weight" in result
        assert "vision_encoder.pos_norm.weight" in result
        assert result["vision_encoder.pos_emb_x.weight"].shape == (1120, 32)
        assert result["vision_encoder.pos_emb_y.weight"].shape == (1120, 32)
        assert torch.allclose(result["vision_encoder.pos_emb_x.weight"], torch.tensor(1.0))
        assert torch.allclose(result["vision_encoder.pos_emb_y.weight"], torch.tensor(2.0))

        # Projections → vision_encoder.projector / audio_encoder.projector.
        assert "vision_encoder.projector.weight" in result
        assert "audio_encoder.projector.weight" in result

        # Scale-free pre-projection norms must be dropped (no graph initializer).
        assert not any("embedding_pre_projection_norm" in k for k in result)
        # The raw checkpoint module prefixes must not leak through.
        assert not any(k.startswith("vision_embedder.") for k in result)
        assert not any(k.startswith("embed_vision.") for k in result)
        assert not any(k.startswith("language_model.") for k in result)

    def test_vision_embedder_preprocess_standalone(self):
        from mobius.models.gemma4 import _Gemma4UnifiedVisionEmbedderModel

        config = _tiny_gemma4_unified_config()
        embedder = _Gemma4UnifiedVisionEmbedderModel(config)

        pos_embedding = torch.empty(1120, 2, 32)
        pos_embedding[:, 0, :] = 3.0
        pos_embedding[:, 1, :] = 4.0
        fake_sd = {
            "vision_embedder.patch_ln1.weight": torch.zeros(432),
            "vision_embedder.pos_embedding": pos_embedding,
            "embed_vision.embedding_projection.weight": torch.zeros(64, 32),
            "embed_vision.embedding_pre_projection_norm.weight": torch.zeros(32),
        }
        result = embedder.preprocess_weights(fake_sd)

        assert "patch_ln1.weight" in result
        assert "projector.weight" in result
        assert torch.allclose(result["pos_emb_x.weight"], torch.tensor(3.0))
        assert torch.allclose(result["pos_emb_y.weight"], torch.tensor(4.0))
        assert not any("embedding_pre_projection_norm" in k for k in result)

    def test_audio_embedder_preprocess_standalone(self):
        from mobius.models.gemma4 import _Gemma4UnifiedAudioEmbedderModel

        config = _tiny_gemma4_unified_config()
        embedder = _Gemma4UnifiedAudioEmbedderModel(config)

        fake_sd = {
            "embed_audio.embedding_projection.weight": torch.zeros(64, 16),
            "embed_audio.embedding_pre_projection_norm.weight": torch.zeros(16),
        }
        result = embedder.preprocess_weights(fake_sd)

        assert result["projector.weight"].shape == (64, 16)
        assert not any("embedding_pre_projection_norm" in k for k in result)


class TestGemma4UnifiedConfigHooks:
    """Config extraction hooks map unified vision/audio sub-configs."""

    def test_vision_hook_maps_fields(self):
        from types import SimpleNamespace

        from mobius._configs.per_model._gemma4_unified_vision import (
            _gemma4_unified_vision,
        )

        composite = SimpleNamespace(
            model_type="gemma4_unified",
            image_token_id=258880,
            vision_config=SimpleNamespace(
                mm_embed_dim=3840,
                patch_size=16,
                pooling_kernel_size=3,
                mm_posemb_size=1120,
                output_proj_dims=3840,
                rms_norm_eps=1e-6,
            ),
        )
        fields: dict = {}
        _gemma4_unified_vision(composite, None, "gemma4_unified", fields)

        assert fields["hidden_size"] == 3840
        assert fields["patch_size"] == 16
        assert fields["pooling_kernel_size"] == 3
        assert fields["position_embedding_size"] == 1120
        assert fields["out_hidden_size"] == 3840
        assert fields["image_token_id"] == 258880

    def test_vision_hook_skips_unrelated_model(self):
        from types import SimpleNamespace

        from mobius._configs.per_model._gemma4_unified_vision import (
            _gemma4_unified_vision,
        )

        composite = SimpleNamespace(model_type="qwen2_vl", vision_config=object())
        fields: dict = {}
        result = _gemma4_unified_vision(composite, None, "qwen2_vl", fields)
        assert result is None
        assert fields == {}

    def test_audio_hook_maps_fields(self):
        from types import SimpleNamespace

        from mobius._configs.per_model._gemma4_unified_audio import (
            _gemma4_unified_audio,
        )

        composite = SimpleNamespace(
            model_type="gemma4_unified",
            audio_token_id=258881,
            audio_config=SimpleNamespace(audio_embed_dim=640),
        )
        result = _gemma4_unified_audio(composite, None, "gemma4_unified", {})

        assert result is not None
        audio_cfg = result["audio"]
        assert audio_cfg.hidden_size == 640
        assert audio_cfg.output_proj_dims == 640
        assert audio_cfg.audio_token_id == 258881


def _tiny_gemma4_text_moe_config(**overrides) -> Gemma4Config:
    """Minimal text-only Gemma4 config with the parallel MoE block enabled.

    The graph builds on CPU, which has no fused ``com.microsoft::MoE`` op, so
    ``Gemma4DecoderLayer`` emits the vectorized unfused fallback — the path that
    applies the exact expert activation.
    """
    defaults = dict(
        model_type="gemma4",
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=256,
        rms_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        attn_qk_norm=True,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
        global_head_dim=16,
        global_rope_theta=10_000.0,
        global_partial_rotary_factor=0.25,
        final_logit_softcapping=0.0,
        hidden_size_per_layer_input=0,
        pad_token_id=0,
        tie_word_embeddings=True,
        enable_moe_block=True,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
    )
    defaults.update(overrides)
    return Gemma4Config(**defaults)


def _build_gemma4_text(config: Gemma4Config) -> ir.Model:
    from mobius._registry import registry
    from mobius.integrations.transformers._config_resolver import _default_task_for_model
    from mobius.tasks import get_task

    module = registry.get("gemma4_text")(config)
    task = get_task(_default_task_for_model("gemma4_text"))
    return task.build(module, config)["model"]


class TestGemma4NativeMoe:
    @pytest.fixture(autouse=True)
    def custom_runtime_opt_in(self):
        from mobius._flags import override_flags

        with override_flags(gemma4_custom_geglu=True):
            yield

    @staticmethod
    def _quantization(**overrides):
        return QuantizationConfig(
            quant_method="olive", bits=4, group_size=16, quantize_moe=True, **overrides
        )

    @pytest.mark.parametrize("quantized", [False, True])
    @pytest.mark.parametrize(
        "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
    )
    def test_native_geglu_uses_raw_logits_and_separate_expert_norm(
        self, quantized, dtype, tmp_path
    ):
        import numpy as np

        from mobius._build_context import build_context
        from mobius._builder import _cast_module_dtype
        from mobius._constants import GEMMA4_CUSTOM_GEGLU_METADATA_KEY
        from mobius._execution_providers import get_ep
        from mobius.tasks._gemma4 import Gemma4TextCausalLMTask

        config = _tiny_gemma4_text_moe_config(
            dtype=dtype, quantization=self._quantization() if quantized else None
        )
        if quantized and dtype == ir.DataType.FLOAT:
            with pytest.raises(NotImplementedError, match="FLOAT16 or BFLOAT16"):
                Gemma4CausalLMModel(config)
            return
        module = Gemma4CausalLMModel(config)
        _cast_module_dtype(module, dtype)
        with build_context(get_ep("cuda"), dtype):
            graph = Gemma4TextCausalLMTask().build(module, config)["model"].graph
        nodes = [node for node in graph if node.op_type == ("QMoE" if quantized else "MoE")]
        assert len(nodes) == 2
        for node in nodes:
            assert node.metadata_props[GEMMA4_CUSTOM_GEGLU_METADATA_KEY] == (
                "custom-cuda-geglu-v1"
            )
            assert node.attributes["activation_type"].value == "geglu"
            assert node.attributes["swiglu_fusion"].value == 1
            router = node.inputs[1]
            while router.producer().op_type == "Reshape":
                router = router.producer().inputs[0]
            assert router.producer().op_type == "MatMul"
            scaled = router.producer().inputs[0]
            assert scaled.producer().op_type == "CastLike"
            learned = scaled.producer().inputs[1]
            norm = learned.producer().inputs[0].producer()
            assert norm.op_type == "RMSNormalization"
            expert_norm = node.inputs[0].producer()
            assert expert_norm.op_type == "RMSNormalization"
            assert norm.inputs[0] is expert_norm.inputs[0]
            assert norm is not expert_norm
        if quantized:
            assert graph.initializers["model.layers.0.moe.fc1_experts_weights"].dtype == (
                ir.DataType.UINT8
            )
            assert graph.initializers["model.layers.0.moe.gate.proj.weight"].dtype == dtype
        # The requirement belongs to the exported operator, not transient flags.
        for initializer in graph.initializers.values():
            if initializer.const_value is None:
                assert initializer.shape is not None and initializer.dtype is not None
                initializer.const_value = ir.tensor(
                    np.zeros(list(initializer.shape), dtype=initializer.dtype.numpy())
                )
        ir.save(ir.Model(graph, ir_version=10), tmp_path / "native.onnx")
        restored = ir.load(tmp_path / "native.onnx")
        restored_nodes = [
            node for node in restored.graph if node.op_type == ("QMoE" if quantized else "MoE")
        ]
        assert len(restored_nodes) == 2
        assert all(
            node.metadata_props[GEMMA4_CUSTOM_GEGLU_METADATA_KEY] == "custom-cuda-geglu-v1"
            for node in restored_nodes
        )

    def test_default_cuda_is_portable_and_quantized_experts_require_opt_in(self):
        from mobius._builder import build_from_module
        from mobius._flags import override_flags

        with override_flags(gemma4_custom_geglu=False):
            config = _tiny_gemma4_text_moe_config(dtype=ir.DataType.BFLOAT16)
            package = build_from_module(
                Gemma4CausalLMModel(config), config, execution_provider="cuda"
            )
            assert not any(node.op_type in {"MoE", "QMoE"} for node in package["model"].graph)
            assert any(node.op_type == "Softmax" for node in package["model"].graph)
            with pytest.raises(NotImplementedError, match="MOBIUS_GEMMA4_CUSTOM_GEGLU"):
                Gemma4CausalLMModel(
                    dataclasses.replace(config, quantization=self._quantization())
                )

    def test_native_experts_fail_closed_on_unsupported_ep_and_layout(self):
        config = _tiny_gemma4_text_moe_config(
            dtype=ir.DataType.BFLOAT16, quantization=self._quantization()
        )
        with pytest.raises(NotImplementedError, match="GEGLU-capable"):
            _build_gemma4_text(config)
        for hidden_act in ("gelu", "silu"):
            with pytest.raises(NotImplementedError, match="tanh-GELU"):
                Gemma4CausalLMModel(dataclasses.replace(config, hidden_act=hidden_act))
        quantization = dataclasses.replace(
            config.quantization,
            overrides={
                "model.language_model.layers.0.experts.down_proj": QuantizationOverride(bits=8)
            },
        )
        with pytest.raises(NotImplementedError, match="INT4"):
            Gemma4CausalLMModel(dataclasses.replace(config, quantization=quantization))

    @pytest.mark.parametrize("multimodal", [False, True])
    @pytest.mark.parametrize("symmetric", [False, True])
    @pytest.mark.parametrize("component_plan", [False, True])
    def test_expert_payloads_and_module_layouts(self, multimodal, symmetric, component_plan):
        from mobius._component_quantization import configure_component_quantization
        from mobius.tasks._gemma4 import Gemma4Task

        quantization = self._quantization(
            sym=symmetric,
            overrides=(
                {
                    "model.language_model.layers.0.self_attn.q_proj": QuantizationOverride(
                        bits=8
                    )
                }
                if component_plan
                else {}
            ),
        )
        config = _tiny_gemma4_config(
            dtype=ir.DataType.BFLOAT16,
            hidden_act="gelu_pytorch_tanh",
            quantization=quantization,
            component_quantization={"decoder": quantization} if component_plan else None,
        )
        prefix = "model.language_model.layers.0" if multimodal else "model.layers.0"
        target = "decoder.model.layers.0.moe" if multimodal else "model.layers.0.moe"
        gate = torch.arange(4 * 64 * 32).to(torch.uint8).reshape(4, 64, 32)
        down = torch.arange(4 * 64 * 16).to(torch.uint8).reshape(4, 64, 16)
        output_scale = torch.tensor([-2.0, 0.0, 0.5, 1.5], dtype=torch.bfloat16)
        state = {
            f"{prefix}.experts.gate_up_proj_qweight": gate,
            f"{prefix}.experts.gate_up_proj_scales": torch.ones(
                4, 64, 4, dtype=torch.bfloat16
            ),
            f"{prefix}.experts.down_proj_qweight": down,
            f"{prefix}.experts.down_proj_scales": torch.ones(4, 64, 2, dtype=torch.bfloat16),
            f"{prefix}.router.scale": torch.ones(64, dtype=torch.bfloat16),
            f"{prefix}.router.proj.weight": torch.ones(4, 64, dtype=torch.bfloat16),
            f"{prefix}.router.per_expert_scale": output_scale,
        }
        if not symmetric:
            state[f"{prefix}.experts.gate_up_proj_qzeros"] = torch.full(
                (4, 64, 2), 0x93, dtype=torch.uint8
            )
            state[f"{prefix}.experts.down_proj_qzeros"] = torch.full(
                (4, 64, 1), 0x57, dtype=torch.uint8
            )
        module = (Gemma4Model if multimodal else Gemma4CausalLMModel)(config)
        result = module.preprocess_weights(dict(state))
        torch.testing.assert_close(result[f"{target}.fc1_experts_weights"], gate)
        torch.testing.assert_close(result[f"{target}.fc2_experts_weights"], down)
        torch.testing.assert_close(
            result[f"{target}.fc2_scales"],
            torch.ones(4, 64, 2, dtype=torch.bfloat16) * output_scale[:, None, None],
        )
        torch.testing.assert_close(
            result[f"{target}.gate.scale"], torch.ones(64, dtype=torch.bfloat16)
        )
        assert (f"{target}.fc1_experts_zero_points" in result) is not symmetric
        if not symmetric:
            assert torch.all(result[f"{target}.fc1_experts_zero_points"] == 0x93)
        if component_plan and multimodal:
            from mobius.components import Linear

            task = Gemma4Task()
            manifest = configure_component_quantization(module, config, task)
            normalized = normalize_component_quantized_weights(
                result,
                module,
                config,
                ("decoder", "vision_encoder", "embedding"),
                manifest=manifest,
                task=task,
            )
            assert normalized.keys() == result.keys()
            torch.testing.assert_close(normalized[f"{target}.fc1_experts_weights"], gate)
            assert module.decoder.model.layers[0].self_attn.q_proj._bits == 8
            assert isinstance(module.decoder.model.layers[0].moe.gate.proj, Linear)
        invalid = dict(state)
        invalid[f"{prefix}.experts.down_proj_qweight"] = down[..., :-1]
        with pytest.raises(ValueError, match="expected shape"):
            module.preprocess_weights(invalid)
        invalid = dict(state)
        invalid.pop(f"{prefix}.experts.down_proj_scales")
        with pytest.raises(ValueError, match="expected sidecars"):
            module.preprocess_weights(invalid)

    @pytest.mark.parametrize("key", ["moe", "quantize_moe"])
    def test_quantization_metadata_opt_in(self, key):
        quantization = QuantizationConfig.from_value(
            {"quant_method": "olive", "bits": 4, "group_size": 16, key: True}
        )
        assert quantization.quantize_moe

    def test_canonical_quantize_moe_takes_precedence_over_legacy_alias(self):
        quantization = QuantizationConfig.from_value(
            {"quant_method": "olive", "quantize_moe": False, "moe": True}
        )
        assert not quantization.quantize_moe


class TestGemma4VisionStandardization:
    def test_bf16_kernel_three_pool_matches_actual_hf_rounding(self):
        import numpy as np
        from transformers import Gemma4VisionConfig as HFVisionConfig
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4VisionPooler as HFVisionPooler,
        )

        from mobius._testing import create_test_builder, create_test_input
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import Gemma4VisionPooler

        hidden = torch.ones(1, 9, 4, dtype=torch.bfloat16)
        hidden[:, -1] = 1.03125
        positions = torch.tensor([[(x, y) for y in range(3) for x in range(3)]])
        hf_pooler = HFVisionPooler(HFVisionConfig(hidden_size=4))
        expected, _ = hf_pooler(
            hidden, positions, torch.zeros(1, 9, dtype=torch.bool), output_length=1
        )
        # Actual pinned HF rounds the FP32 mean to 1, then scales by sqrt(4).
        torch.testing.assert_close(expected, torch.full((1, 1, 4), 2.0), rtol=0, atol=0)
        rounded_coefficients = torch.full((1, 1, 9), 1 / 9, dtype=torch.bfloat16)
        old_mean = (rounded_coefficients.float() @ hidden.float()).to(torch.bfloat16)
        # Sensitive to premature coefficient cast; compare exactly in BF16.
        torch.testing.assert_close(
            old_mean, torch.full_like(old_mean, 1.0078125), rtol=0, atol=0
        )

        builder, op, graph = create_test_builder()
        features = create_test_input(builder, "features", [1, 9, 4], dtype=ir.DataType.FLOAT)
        pos = create_test_input(builder, "positions", [1, 9, 2], dtype=ir.DataType.INT64)
        output = Gemma4VisionPooler(4, kernel_size=3)(
            op, op.Cast(features, to=ir.DataType.BFLOAT16), pos
        )
        output.name = "pooled"
        graph.outputs.append(output)
        matmul = next(node for node in graph if node.op_type == "MatMul")
        assert all(value.dtype == ir.DataType.FLOAT for value in matmul.inputs)
        actual = OnnxModelSession(ir.Model(graph, ir_version=10), device="cpu").run(
            {"features": hidden.float().numpy(), "positions": positions.numpy()}
        )["pooled"]
        np.testing.assert_array_equal(actual, expected.numpy())

    @pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
    def test_bf16_pool_zeroes_padded_nonfinite_like_actual_hf(self, nonfinite):
        import numpy as np
        from transformers import Gemma4VisionConfig as HFVisionConfig
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4VisionPooler as HFVisionPooler,
        )

        from mobius._testing import create_test_builder, create_test_input
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import Gemma4VisionPooler

        hidden = torch.ones(1, 9, 4, dtype=torch.bfloat16)
        hidden[:, -1] = nonfinite
        positions = torch.tensor([[(x, y) for y in range(3) for x in range(3)]])
        positions[:, -1] = -1
        padding = (positions == -1).all(dim=-1)
        expected, _ = HFVisionPooler(HFVisionConfig(hidden_size=4))(
            hidden, positions, padding, output_length=1
        )
        assert torch.isfinite(expected).all()
        rounded_mean = torch.tensor(8 / 9, dtype=torch.bfloat16).float()
        torch.testing.assert_close(
            expected, (rounded_mean * 2).expand(1, 1, 4), rtol=0, atol=0
        )

        builder, op, graph = create_test_builder()
        features = create_test_input(builder, "features", [1, 9, 4], dtype=ir.DataType.FLOAT)
        pos = create_test_input(builder, "positions", [1, 9, 2], dtype=ir.DataType.INT64)
        output = Gemma4VisionPooler(4, kernel_size=3)(
            op, op.Cast(features, to=ir.DataType.BFLOAT16), pos
        )
        output.name = "pooled"
        graph.outputs.append(output)
        actual = OnnxModelSession(ir.Model(graph, ir_version=10), device="cpu").run(
            {"features": hidden.float().numpy(), "positions": positions.numpy()}
        )["pooled"]
        np.testing.assert_array_equal(actual, expected.numpy())

    @pytest.mark.parametrize("missing", ["vision", "hidden_size"])
    def test_vision_encoder_requires_concrete_hidden_size(self, missing):
        from mobius._configs import VisionConfig
        from mobius.models.gemma4 import _Gemma4VisionEncoderModel

        with pytest.raises(ValueError, match=r"vision\.hidden_size"):
            _Gemma4VisionEncoderModel(
                _tiny_gemma4_config(
                    vision=None if missing == "vision" else VisionConfig(hidden_size=None)
                )
            )

    @pytest.mark.parametrize("standardize", [False, True])
    def test_actual_hf_vision_entrypoint_with_padding(self, standardize):
        import numpy as np
        from transformers import Gemma4VisionConfig as HFVisionConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        from mobius import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import _Gemma4VisionEncoderModel
        from mobius.tasks._gemma4 import Gemma4Task

        torch.manual_seed(42)
        hf_config = HFVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            patch_size=4,
            pooling_kernel_size=2,
            position_embedding_size=8,
            standardize=standardize,
            use_clipped_linears=False,
        )
        hf_config._attn_implementation = "eager"
        reference = Gemma4VisionModel(hf_config).eval()
        if standardize:
            reference.std_bias.copy_(torch.linspace(-5.0, 5.0, 32))
            reference.std_scale.copy_(torch.linspace(0.1, 2.0, 32))
        vision = dataclasses.replace(
            _tiny_gemma4_config().vision,
            pooling_kernel_size=2,
            position_embedding_size=8,
            standardize=standardize,
        )
        config = _tiny_gemma4_config(vision=vision)
        module = _Gemma4VisionEncoderModel(config)
        package = ModelPackage({"vision_encoder": Gemma4Task()._build_vision(module, config)})
        projection = torch.randn(64, 32) * 0.1
        weights = {
            "vision_tower." + name: value for name, value in reference.state_dict().items()
        }
        weights["embed_vision.embedding_projection.weight"] = projection
        package.apply_weights(module.preprocess_weights(weights))
        positions = torch.tensor(
            [[*[[x, y] for y in range(2) for x in range(4)], *[[-1, -1]] * 4]]
        )
        pixels = torch.rand(1, 12, 48)
        with torch.inference_mode():
            features = reference(
                pixel_values=pixels, pixel_position_ids=positions
            ).last_hidden_state
            expected = torch.nn.functional.linear(
                torch.nn.functional.rms_norm(features, (32,), eps=vision.norm_eps), projection
            )
        actual = OnnxModelSession(package["vision_encoder"], device="cpu").run(
            {"pixel_values": pixels.numpy(), "pixel_position_ids": positions.numpy()}
        )["image_features"]
        assert actual.shape == (2, 64)
        np.testing.assert_allclose(actual, expected.numpy(), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("standardize", [False, True])
    def test_config_and_weight_mapping(self, standardize):
        from types import SimpleNamespace

        from mobius._configs._vision_defaults import apply_vision_defaults
        from mobius.models.gemma4 import _Gemma4VisionEncoderModel

        fields = {}
        apply_vision_defaults(
            SimpleNamespace(vision_config={"standardize": standardize}),
            None,
            "gemma4",
            fields,
        )
        assert fields["standardize"] is standardize
        config = _tiny_gemma4_config(
            vision=dataclasses.replace(_tiny_gemma4_config().vision, standardize=standardize)
        )
        module = _Gemma4VisionEncoderModel(config)
        assert (module.std_bias is not None) is standardize
        if standardize:
            bias, scale = torch.arange(32).float(), torch.ones(32)
            state = {"vision_tower.std_bias": bias, "vision_tower.std_scale": scale}
            assert module.preprocess_weights(state) == {"std_bias": bias, "std_scale": scale}
            result = Gemma4Model(config).preprocess_weights(
                {"model." + key: value for key, value in state.items()}
            )
            assert result["vision_encoder.std_bias"] is bias
            assert result["vision_encoder.std_scale"] is scale


class TestGemma4MoeFallbackActivation:
    """The unfused MoE fallback must apply the EXACT ``hidden_act`` (ACT2FN).

    ``gelu`` maps to the erf GELU while only the tanh/fast/new variants use the
    tanh approximation, so ``hidden_act="gelu"`` must NOT silently become
    ``Gelu(approximate="tanh")`` on the fallback path.
    """

    def _activation_nodes(self, model: ir.Model):
        acts = []
        for node in model.graph:
            if node.op_type in ("Gelu", "Swish"):
                attrs = {a.name: a.value for a in node.attributes.values()}
                acts.append((node.op_type, attrs.get("approximate")))
        return acts

    def test_silu_uses_swish(self):
        model = _build_gemma4_text(_tiny_gemma4_text_moe_config(hidden_act="silu"))
        acts = self._activation_nodes(model)
        assert acts, "MoE fallback must emit an activation op"
        assert all(op == "Swish" for op, _ in acts), acts

    def test_gelu_uses_erf_not_tanh(self):
        model = _build_gemma4_text(_tiny_gemma4_text_moe_config(hidden_act="gelu"))
        acts = self._activation_nodes(model)
        assert acts, "MoE fallback must emit an activation op"
        assert all(op == "Gelu" for op, _ in acts), acts
        # erf GELU => the ``approximate`` attribute must be absent/"none".
        assert all(approx in (None, "none") for _, approx in acts), acts

    def test_gelu_pytorch_tanh_uses_tanh_approx(self):
        model = _build_gemma4_text(
            _tiny_gemma4_text_moe_config(hidden_act="gelu_pytorch_tanh")
        )
        acts = self._activation_nodes(model)
        assert acts, "MoE fallback must emit an activation op"
        assert all(op == "Gelu" and approx == "tanh" for op, approx in acts), acts

    @pytest.mark.parametrize("selected_expert", [0, 1])
    @pytest.mark.parametrize("expert_scale", [0.0, 1.0])
    def test_nonfinite_expert_is_masked_by_selection_not_weight(
        self, selected_expert, expert_scale
    ):
        import numpy as np

        from mobius._testing import create_test_builder, create_test_input
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import Gemma4DecoderLayer

        class FallbackOnly(Gemma4DecoderLayer):
            def forward(
                self,
                op,
                hidden_states,
                attention_bias,
                position_embeddings=None,
                shared_kv_states=None,
                per_layer_input=None,
                past_key_value=None,
                is_causal=1,
                static_kv_seqlen=None,
            ):
                self.router(op, hidden_states)
                return self._dispatch_moe_fallback(op, hidden_states, attention_bias)

        config = _tiny_gemma4_text_moe_config(
            dtype=ir.DataType.FLOAT,
            num_local_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=16,
        )
        module = FallbackOnly(config, 0)
        fc1 = np.zeros((2, 32, 64), dtype=np.float32)
        fc1[1] = np.finfo(np.float32).max
        fc2 = np.zeros((2, 64, 16), dtype=np.float32)
        fc2[1] = 1
        module.fc1_experts_weights.const_value = ir.tensor(fc1)
        module.fc2_experts_weights.const_value = ir.tensor(fc2)
        module.router.per_expert_scale.const_value = ir.tensor(
            np.full(2, expert_scale, dtype=np.float32)
        )
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "normed_flat", [1, 64])
        probs = create_test_input(builder, "router_probs", [1, 2])
        output = module(op, x, probs)
        output.name = "fallback"
        output.shape = ir.Shape([1, 64])
        output.dtype = ir.DataType.FLOAT
        graph.outputs.append(output)
        graph = ir.convenience.extract(graph, [x, probs], [output])
        session = OnnxModelSession(ir.Model(graph, ir_version=10), device="cpu")
        try:
            actual = session.run(
                {
                    "normed_flat": np.ones((1, 64), dtype=np.float32),
                    "router_probs": np.eye(2, dtype=np.float32)[
                        selected_expert : selected_expert + 1
                    ],
                }
            )["fallback"]
        finally:
            session.close()
        if selected_expert == 0:
            np.testing.assert_array_equal(actual, np.zeros((1, 64), dtype=np.float32))
        else:
            # Even a zero learned scale must not hide a selected expert's NaNs.
            assert not np.isfinite(actual).any()


class TestGemma4OutputLayerIndices:
    """``output_layer_indices`` taps the post-final-norm hidden as ``hidden_states.{idx}``.

    Gemma4 exposes ONLY the final post-final-norm hidden (== HF
    ``hidden_states[-1]``, the lm_head input), which a borrowed-KV speculative
    drafter consumes as its folded-carry seed. It is faithful only for the last
    decoder layer, so exactly one index == ``num_hidden_layers - 1`` is allowed;
    duplicate/out-of-range/negative/multiple/non-last indices are rejected.
    ``None``/empty preserves the legacy output contract (no extra port).
    """

    def test_none_keeps_legacy_output_contract(self):
        model = _build_gemma4_text(_tiny_gemma4_text_moe_config())
        names = {o.name for o in model.graph.outputs}
        assert "logits" in names
        assert not any(n.startswith("hidden_states.") for n in names)

    def test_empty_list_keeps_legacy_output_contract(self):
        model = _build_gemma4_text(_tiny_gemma4_text_moe_config(output_layer_indices=[]))
        names = {o.name for o in model.graph.outputs}
        assert "logits" in names
        assert not any(n.startswith("hidden_states.") for n in names)

    def test_last_index_emits_hidden_states_port(self):
        # Last decoder layer index (num_hidden_layers - 1) — the post-norm slot.
        config = _tiny_gemma4_text_moe_config(output_layer_indices=[1])
        model = _build_gemma4_text(config)
        names = {o.name for o in model.graph.outputs}
        assert "hidden_states.1" in names
        assert "logits" in names

    def test_hidden_states_port_is_identity_of_lm_head_input(self):
        """The emitted hidden is a DISTINCT Identity of the exact lm_head input.

        Distinctness prevents the in-place-rename collapse that would occur if a
        single shared Value were passed to ``add_output`` more than once, while
        the Identity's input being the lm_head input proves the post-final-norm
        faithfulness.
        """
        config = _tiny_gemma4_text_moe_config(output_layer_indices=[1])
        model = _build_gemma4_text(config)
        hs_out = next(o for o in model.graph.outputs if o.name == "hidden_states.1")
        logits_out = next(o for o in model.graph.outputs if o.name == "logits")
        lm_head_node = logits_out.producer()
        assert lm_head_node is not None
        # The output port is its own Identity node (unique name, no in-place
        # rename of the internal lm_head input value).
        id_node = hs_out.producer()
        assert id_node is not None and id_node.op_type == "Identity"
        # ...and its input is the exact post-final-norm value fed into lm_head.
        assert id_node.inputs[0] in set(lm_head_node.inputs), (
            "hidden_states port must be an Identity of the lm_head input (post-final-norm)"
        )

    def test_multiple_indices_rejected(self):
        with pytest.raises(ValueError, match="exactly"):
            Gemma4CausalLMModel(_tiny_gemma4_text_moe_config(output_layer_indices=[0, 1]))

    def test_non_last_index_rejected(self):
        # num_hidden_layers=2 => only [1] is valid; [0] is a non-last index.
        with pytest.raises(ValueError, match="exactly"):
            Gemma4CausalLMModel(_tiny_gemma4_text_moe_config(output_layer_indices=[0]))

    def test_duplicate_indices_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            Gemma4CausalLMModel(_tiny_gemma4_text_moe_config(output_layer_indices=[1, 1]))

    def test_out_of_range_index_rejected(self):
        # num_hidden_layers=2 => valid indices are [0, 2); 2 is out of range.
        with pytest.raises(ValueError, match="out of range"):
            Gemma4CausalLMModel(_tiny_gemma4_text_moe_config(output_layer_indices=[2]))

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            Gemma4CausalLMModel(_tiny_gemma4_text_moe_config(output_layer_indices=[-1]))

    def test_valid_index_scales_with_num_hidden_layers(self):
        # The accepted index tracks the config (no hardcoded layer count/name).
        config = _tiny_gemma4_text_moe_config(
            num_hidden_layers=3,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
            output_layer_indices=[2],
        )
        model = _build_gemma4_text(config)
        names = {o.name for o in model.graph.outputs}
        assert "hidden_states.2" in names


class TestGemma4PerExpertScalePartialStateDict:
    """Neutralization of ``router.per_expert_scale`` is scoped to folded prefixes.

    A partial state_dict that supplies ``router.per_expert_scale`` without the
    matching ``fc2_experts_weights`` must keep its scale intact (no silent drop).
    """

    def test_scale_preserved_when_fc2_absent(self):
        config = _tiny_gemma4_config()
        model = Gemma4CausalLMModel(config)
        pes = torch.tensor([1.0, 2.0, 3.0, 4.0])
        fake_sd = {"model.layers.0.router.per_expert_scale": pes.clone()}
        result = model.preprocess_weights(fake_sd)
        key = "model.layers.0.router.per_expert_scale"
        assert key in result
        assert torch.allclose(result[key], pes), (
            "per_expert_scale must be preserved when its fc2 pair is missing"
        )

    def test_scale_folded_and_neutralized_when_fc2_present(self):
        config = _tiny_gemma4_config()
        model = Gemma4CausalLMModel(config)
        pes = torch.tensor([1.0, 2.0, 3.0, 4.0])
        fc2 = torch.ones(4, 64, 32)
        fake_sd = {
            "model.layers.0.router.per_expert_scale": pes.clone(),
            "model.layers.0.experts.down_proj": fc2.clone(),
        }
        result = model.preprocess_weights(fake_sd)
        # Router copy neutralized to ones (scale now baked into fc2).
        neutralized = result["model.layers.0.router.per_expert_scale"]
        assert torch.allclose(neutralized, torch.ones(4))
        # fc2[e] scaled by per_expert_scale[e].
        folded = result["model.layers.0.fc2_experts_weights"]
        expected = fc2 * pes.reshape(-1, 1, 1)
        assert torch.allclose(folded, expected)
