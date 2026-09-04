# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Core text, encoder, seq2seq, vision, detection, and graph-option L1 tests.

Run the complete L1 suite with ``pytest tests/build_graph``.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from _test_configs import (
    ALL_CAUSAL_LM_CONFIGS,
    ALL_CONFIGS,
    AUTO_GENERATED_CONFIGS,
    DETECTION_CONFIGS,
    ENCODER_CONFIGS,
    LONGROPE_FACTORS,
    SEQ2SEQ_CONFIGS,
    VISION_CONFIGS,
    _base_config,
)

from build_graph._support import (
    _assert_outputs_have_shapes_and_dtypes,
    _make_params,
    _run_onnx_checker,
    known_untested_model_types,
    specialized_test_model_types,
)
from mobius._builder import DTYPE_MAP, build_from_module
from mobius._configs import AudioConfig, VisionConfig
from mobius._registry import registry
from mobius.integrations.transformers._config_resolver import _default_task_for_model
from mobius.tasks import CausalLMTask, Phi4MMMultiModalTask, get_task

_MODEL_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, _ in ALL_CAUSAL_LM_CONFIGS]
_MODEL_PARAMS = _make_params(ALL_CAUSAL_LM_CONFIGS)
_ENCODER_MODEL_PARAMS = _make_params(ENCODER_CONFIGS)
_SEQ2SEQ_MODEL_PARAMS = _make_params(SEQ2SEQ_CONFIGS)
_VISION_MODEL_PARAMS = _make_params(VISION_CONFIGS)
_DETECTION_MODEL_PARAMS = _make_params(DETECTION_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _MODEL_PARAMS)
class TestBuildGraph:
    """Verify that each model type builds a valid ONNX graph."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        """Build a model graph from a tiny config and verify basic structure."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        # Basic structure checks
        assert model.graph is not None
        assert len(model.graph.inputs) > 0, "Model should have inputs"
        assert len(model.graph.outputs) > 0, "Model should have outputs"

        # Check expected inputs exist
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names

        # Check outputs include logits and KV cache
        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names
        if task_name == "masked-diffusion":
            assert input_names == {"input_ids"}
            assert output_names == {"logits", "proposed_tokens"}
            return

        assert "attention_mask" in input_names
        assert "position_ids" in input_names

        # Check KV cache / hybrid cache outputs.  Models whose trailing layers
        # borrow K,V from an earlier layer (Gemma 3n's num_kv_shared_layers)
        # expose fewer cache entries than they have layers; the non-shared
        # layers are the leading ones, so truncating the range is enough.
        count_fn = getattr(module, "kv_cache_layer_count", None)
        num_layers = count_fn() if callable(count_fn) else config.num_hidden_layers
        layer_types = config.layer_types or []
        for i in range(num_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype in ("mlp", "moe"):
                continue  # MLP and MoE layers are stateless — no cache outputs
            if ltype == "lightning_attention":
                # Lightning Attention: single recurrent state only (no conv_state)
                assert f"present.{i}.recurrent_state" in output_names, (
                    f"Missing present.{i}.recurrent_state"
                )
            elif ltype in ("linear_attention",):
                assert f"present.{i}.conv_state" in output_names, (
                    f"Missing present.{i}.conv_state"
                )
                assert f"present.{i}.recurrent_state" in output_names, (
                    f"Missing present.{i}.recurrent_state"
                )
            elif ltype in ("kimi_linear_attention", "kimi_k3_attention"):
                for state_name in (
                    "q_conv_state",
                    "k_conv_state",
                    "v_conv_state",
                    "recurrent_state",
                ):
                    assert f"present.{i}.{state_name}" in output_names, (
                        f"Missing present.{i}.{state_name}"
                    )
            elif ltype in ("mamba", "mamba2"):
                assert f"present.{i}.conv_state" in output_names, (
                    f"Missing present.{i}.conv_state"
                )
                state_name = "recurrent_state" if model_type == "plamo2" else "ssm_state"
                assert f"present.{i}.{state_name}" in output_names, (
                    f"Missing present.{i}.{state_name}"
                )
            elif ltype == "conv":
                assert f"present.{i}.conv_state" in output_names, (
                    f"Missing present.{i}.conv_state"
                )
            else:
                assert f"present.{i}.key" in output_names, f"Missing present.{i}.key"
                assert f"present.{i}.value" in output_names, f"Missing present.{i}.value"

    def test_graph_has_initializers(self, model_type: str, config_overrides: dict):
        """Verify the graph has initializers (parameters) even without weight values."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0, "Model should have initializers"

        # Check for expected parameter patterns (allow model-specific naming)
        has_embed = any(
            "embed_tokens" in n or "word_embeddings" in n or "wte" in n or "embed_in" in n
            for n in init_names
        )
        has_attn = any(
            "self_attn" in n
            or "self_attention" in n
            or "attention" in n
            or ".attn." in n
            or "qkv_proj" in n
            for n in init_names
        )
        has_mlp = any("mlp" in n or "expert" in n or "feed_forward" in n for n in init_names)
        assert has_embed, "Should have embedding parameters"
        assert has_attn, "Should have attention parameters"
        assert has_mlp, "Should have MLP parameters"

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


class TestTextDecoderBatchGreaterThanOne:
    """Text-only causal LM decoders must run with ``batch_size > 1``.

    Regression guard for the attention-mask head-dim bug: ``create_padding_mask``
    / ``create_sliding_window_mask`` previously returned a 3-D ``(B, q, total)``
    mask whose batch axis was right-aligned onto ``q_num_heads`` by the ONNX
    Attention op — it worked for ``batch == 1`` but ORT rejected ``batch > 1``.
    Covers a plain decoder, GQA, and sliding-window architectures, with ragged
    padding across rows.
    """

    @pytest.mark.parametrize("model_type", ["qwen2", "llama", "mistral", "gemma2"])
    def test_batch2_prefill_runs(self, model_type: str):
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.rewrite_rules._testing_utils import fill_random_weights

        overrides = dict(_MODEL_CONFIGS)[model_type]
        config = _base_config(**overrides)
        module = registry.get(model_type)(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        model = pkg["model"]
        fill_random_weights(model)
        sess = OnnxModelSession(model)

        batch, seq = 2, 6
        rng = np.random.default_rng(0)
        input_ids = rng.integers(1, config.vocab_size, size=(batch, seq)).astype(np.int64)
        attention_mask = np.ones((batch, seq), dtype=np.int64)
        attention_mask[1, :2] = 0  # row 1 has two leading padding tokens
        position_ids = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for i in range(config.num_hidden_layers):
            feeds[f"past_key_values.{i}.key"] = np.zeros(
                (batch, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
            )
            feeds[f"past_key_values.{i}.value"] = np.zeros(
                (batch, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
            )

        out = sess.run(feeds)
        sess.close()
        logits = out["logits"]
        assert logits.shape[0] == batch
        assert logits.shape[1] == seq
        # Independent rows (different inputs) must produce different logits.
        assert not np.allclose(logits[0], logits[1])


@pytest.mark.parametrize("model_type,config_overrides", _ENCODER_MODEL_PARAMS)
class TestBuildEncoderGraph:
    """Verify that encoder-only model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        if model_type not in {"gemma_embedding_gguf", "llama_embed_gguf"}:
            assert "token_type_ids" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names
        # No KV cache outputs for encoder-only models
        assert not any(n.startswith("present.") for n in output_names)

    def test_graph_has_initializers(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_embed = any("word_embeddings" in n or "embed" in n for n in init_names)
        has_attn = any(
            "self_attn" in n or "self_attention" in n or "attention" in n or ".attn." in n
            for n in init_names
        )
        has_mlp = any(
            "mlp" in n or "ffn" in n or "feed_forward" in n or "intermediate" in n
            for n in init_names
        )
        assert has_embed, "Should have word embedding parameters"
        assert has_attn, "Should have attention parameters"
        assert has_mlp, "Should have MLP parameters"

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


@pytest.mark.parametrize("model_type,config_overrides", _SEQ2SEQ_MODEL_PARAMS)
class TestBuildSeq2SeqGraph:
    """Verify that encoder-decoder model types build valid ONNX graphs."""

    def test_encoder_graph_builds(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["encoder"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names

    def test_package_has_encoder_and_decoder(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert "encoder" in pkg
        assert "decoder" in pkg

        dec_outputs = {out.name for out in pkg["decoder"].graph.outputs}
        assert "logits" in dec_outputs

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


@pytest.mark.parametrize("model_type,config_overrides", _VISION_MODEL_PARAMS)
class TestBuildVisionGraph:
    """Verify that vision model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "pixel_values" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


@pytest.mark.parametrize("model_type,config_overrides", _DETECTION_MODEL_PARAMS)
class TestBuildDetectionGraph:
    """Verify that object detection model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "pixel_values" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names
        assert "pred_boxes" in output_names

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


class TestBuildGraphLoRA:
    """Verify LoRA-specific structure in Phi4MM graph."""

    def _phi4mm_config(self):
        return _base_config(
            partial_rotary_factor=0.5,
            rope_type="longrope",
            rope_scaling={
                "short_factor": LONGROPE_FACTORS,
                "long_factor": LONGROPE_FACTORS,
            },
            original_max_position_embeddings=128,
            vision=VisionConfig(
                lora={"r": 4, "lora_alpha": 8},
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            audio=AudioConfig(
                lora={"r": 8, "lora_alpha": 16},
                attention_dim=32,
                attention_heads=2,
                num_blocks=1,
                linear_units=64,
                kernel_size=3,
                input_size=16,
                conv_channels=32,
                t5_bias_max_distance=10,
            ),
            image_token_id=200010,
        )

    def test_lora_initializers_present(self):
        config = self._phi4mm_config()
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = Phi4MMMultiModalTask()
        pkg = task.build(module, config)
        # LoRA adapters live in the decoder model (pkg["decoder"])
        decoder = pkg["decoder"]

        init_names = list(decoder.graph.initializers)
        lora_names = [n for n in init_names if "lora" in n]
        assert len(lora_names) > 0, "Phi4MM should have LoRA initializers"

        # Each layer should have LoRA for q/k/v/o_proj and gate/up/down_proj
        # Each proj has lora_A and lora_B for both vision and speech adapters
        vision_a = [n for n in lora_names if "lora_A.vision" in n]
        vision_b = [n for n in lora_names if "lora_B.vision" in n]
        speech_a = [n for n in lora_names if "lora_A.speech" in n]
        speech_b = [n for n in lora_names if "lora_B.speech" in n]
        assert len(vision_a) > 0, "Should have vision lora_A"
        assert len(vision_b) > 0, "Should have vision lora_B"
        assert len(speech_a) > 0, "Should have speech lora_A"
        assert len(speech_b) > 0, "Should have speech lora_B"


class TestBuildGraphQuantized:
    """Verify quantized model graphs use MatMulNBits."""

    TINY_LAYERS = 2
    NUM_PROJECTIONS_PER_LAYER = 7  # q, k, v, o, gate, up, down

    def _quantized_config(self, sym=True):
        from mobius._configs import QuantizationConfig

        qc = QuantizationConfig(bits=4, group_size=32, quant_method="gptq", sym=sym)
        return _base_config(
            num_hidden_layers=self.TINY_LAYERS,
            quantization=qc,
        )

    def test_matmulnbits_count(self):
        """Each layer has 7 projections → 2 layers = 14 MatMulNBits ops."""
        config = self._quantized_config()
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        matmulnbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        expected = self.TINY_LAYERS * self.NUM_PROJECTIONS_PER_LAYER
        assert len(matmulnbits) == expected, (
            f"Expected {expected} MatMulNBits, got {len(matmulnbits)}"
        )

    def test_scales_initializers_present(self):
        """Quantized projections should have scales initializers."""
        config = self._quantized_config()
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        scales_names = [n for n in init_names if ".scales" in n]
        expected = self.TINY_LAYERS * self.NUM_PROJECTIONS_PER_LAYER
        assert len(scales_names) == expected, (
            f"Expected {expected} scales initializers, got {len(scales_names)}"
        )

    def test_asymmetric_has_zero_points(self):
        """Asymmetric quantization should have zero_points initializers."""
        config = self._quantized_config(sym=False)
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        zp_names = [n for n in init_names if ".zero_points" in n]
        expected = self.TINY_LAYERS * self.NUM_PROJECTIONS_PER_LAYER
        assert len(zp_names) == expected, (
            f"Expected {expected} zero_points, got {len(zp_names)}"
        )

    def test_lm_head_stays_fp(self):
        """lm_head should remain a standard Linear (MatMul), not quantized."""
        config = self._quantized_config()
        model_cls = registry.get("llama")
        module = model_cls(config)
        assert type(module.lm_head).__name__ == "Linear"

    def test_no_quantization_no_matmulnbits(self):
        """Without quantization config, no MatMulNBits ops should exist."""
        config = _base_config(num_hidden_layers=1)
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        matmulnbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        assert len(matmulnbits) == 0, "Non-quantized model should have no MatMulNBits"

    def test_awq_produces_matmulnbits(self):
        """AWQ quantization should also produce MatMulNBits ops."""
        from mobius._configs import QuantizationConfig

        qc = QuantizationConfig(bits=4, group_size=32, quant_method="awq", sym=False)
        config = _base_config(num_hidden_layers=1, quantization=qc)
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        matmulnbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        expected = 1 * self.NUM_PROJECTIONS_PER_LAYER
        assert len(matmulnbits) == expected

    def _shared_moe_config(self, model_type, quantization=None):
        overrides = dict(
            num_hidden_layers=1,
            num_local_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            intermediate_size=32,
            shared_expert_intermediate_size=32,
            hidden_size=64,
        )
        if quantization is not None:
            overrides["quantization"] = quantization
        return _base_config(**overrides)

    def test_qwen2_moe_int4_quantizes_shared_expert(self):
        """int4 Qwen2-MoE builds shared-expert projections as MatMulNBits (mobius#513).

        The GPTQ Qwen1.5-MoE-A2.7B-Int4 checkpoint quantizes
        ``mlp.shared_expert.{gate,up,down}_proj`` (they appear in
        ``modules_in_block_to_quantize``). Before the fix ``Qwen2MoEDecoderLayer``
        built ``Qwen2MoELayer`` without a ``linear_class``, so the shared expert was
        a dense ``MLP`` whose ``Linear`` layers expected an unpacked ``[hidden,inter]``
        weight and failed to load the packed GPTQ tensor. The shared-expert MLP must
        use the quantization-aware factory (three MatMulNBits: gate/up/down), while
        the tiny ``shared_expert_gate`` (hidden -> 1) stays dense.
        """
        from mobius._configs import QuantizationConfig

        qc = QuantizationConfig(bits=4, group_size=32, quant_method="gptq", sym=False)
        module = registry.get("qwen2_moe")(self._shared_moe_config("qwen2_moe", qc))
        layer = module.model.layers[0]
        # Attention projections must also use the quantized factory: the MoE
        # decoder path previously built dense Attention, so a GPTQ checkpoint's
        # packed self_attn weights could not load (mobius#513).
        assert type(layer.self_attn.q_proj).__name__ == "QuantizedLinear"
        assert type(layer.mlp.shared_expert.gate_proj).__name__ == "QuantizedLinear"
        assert type(layer.mlp.shared_expert.up_proj).__name__ == "QuantizedLinear"
        assert type(layer.mlp.shared_expert.down_proj).__name__ == "QuantizedLinear"
        # Routing projection stays dense (excluded from quantization in the source).
        assert type(layer.mlp.shared_expert_gate).__name__ == "Linear"

        pkg = CausalLMTask().build(module, module.config)
        model = pkg["model"]
        qmoe = [n for n in model.graph if n.op_type == "QMoE"]
        nbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        assert len(qmoe) == 1, f"routed experts must fuse to one QMoE, got {len(qmoe)}"
        assert len(nbits) >= 3, (
            f"shared expert must emit >=3 MatMulNBits (gate/up/down), got {len(nbits)}"
        )

    def test_qwen2_moe_dense_shared_expert_stays_linear(self):
        """Without quantization the Qwen2-MoE shared expert stays a dense MLP."""
        module = registry.get("qwen2_moe")(self._shared_moe_config("qwen2_moe"))
        layer = module.model.layers[0]
        assert type(layer.mlp.shared_expert.down_proj).__name__ == "Linear"
        assert type(layer.mlp.shared_expert_gate).__name__ == "Linear"

    def test_glm4_moe_int4_quantizes_shared_expert(self):
        """int4 GLM4-MoE (ungated shared expert) quantizes its shared expert too.

        Same mobius#513 class of bug in ``UngatedSharedMoELayer`` (Ernie4.5 / GLM4),
        which built the shared expert with a dense ``MLP`` regardless of quantization.
        """
        from mobius._configs import QuantizationConfig

        qc = QuantizationConfig(bits=4, group_size=32, quant_method="gptq", sym=False)
        module = registry.get("glm4_moe")(self._shared_moe_config("glm4_moe", qc))
        layer = module.model.layers[0]
        assert type(layer.mlp.shared_expert.down_proj).__name__ == "QuantizedLinear"

    def test_qwen3_moe_olive_int4_emits_qmoe_and_matmulnbits(self):
        """Olive-int4 Qwen3-MoE fuses routed experts into QMoE, attention into MatMulNBits.

        Graph/parameter-ABI check only — no weights are loaded and
        ``preprocess_weights`` is not exercised here (see
        ``src/mobius/models/moe_test.py`` for the state-dict side). It pins the
        shapes the fused expert path *expects*: one ``com.microsoft::QMoE`` per
        layer with expert-major ``[experts, 2*moe_inter, hidden*bits/8]``
        weights, plus MatMulNBits for the four quantized attention projections.
        """
        from mobius._configs import QuantizationConfig

        qc = QuantizationConfig(bits=4, group_size=32, quant_method="olive", sym=True)
        config = self._shared_moe_config("qwen3_moe", qc)
        module = registry.get("qwen3_moe")(config)
        pkg = CausalLMTask().build(module, config)
        model = pkg["model"]

        qmoe = [n for n in model.graph if n.op_type == "QMoE"]
        nbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        assert len(qmoe) == 1, f"routed experts must fuse to one QMoE, got {len(qmoe)}"
        assert len(nbits) == 4, f"attention q/k/v/o must emit 4 MatMulNBits, got {len(nbits)}"

        initializers = model.graph.initializers
        experts = config.num_local_experts
        assert tuple(initializers["model.layers.0.mlp.fc1_experts_weights"].shape) == (
            experts,
            2 * config.moe_intermediate_size,
            config.hidden_size * qc.bits // 8,
        )
        assert tuple(initializers["model.layers.0.mlp.fc2_experts_weights"].shape) == (
            experts,
            config.hidden_size,
            config.moe_intermediate_size * qc.bits // 8,
        )
        assert tuple(initializers["model.layers.0.mlp.fc1_scales"].shape) == (
            experts,
            2 * config.moe_intermediate_size,
            config.hidden_size // qc.group_size,
        )
        # Symmetric quantization carries no zero points for the routed experts.
        moe_initializers = [n for n in initializers if n.startswith("model.layers.0.mlp.")]
        assert not any("zero_point" in name for name in moe_initializers)


class TestBuildGraphDtype:
    """Verify dtype casting for model initializers."""

    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("f16", "FLOAT16"),
            ("bf16", "BFLOAT16"),
        ],
    )
    def test_dtype_casts_float_initializers(self, dtype_str, expected):
        """Build with dtype and verify Parameter-derived initializers are cast."""
        config = _base_config()
        config.dtype = DTYPE_MAP[dtype_str]
        model_cls = registry.get("llama")
        module = model_cls(config)
        model = build_from_module(module, config)["model"]

        expected_dtype = ir.DataType[expected]
        for name, init in model.graph.initializers.items():
            if init.dtype == ir.DataType.INT64:
                continue
            # Lifted scalar constants (e.g. const_1.0_f32) stay f32
            if name.startswith("const_"):
                continue
            assert init.dtype == expected_dtype, (
                f"Initializer '{name}' dtype is {init.dtype}, expected {expected_dtype}"
            )

    @pytest.mark.parametrize(
        "dtype_str",
        ["f16", "bf16"],
    )
    def test_multimodal_encoder_inputs_match_model_dtype(self, dtype_str):
        """Vision/audio encoder graph inputs use the model's compute dtype.

        ORT GenAI's image/audio processor output doesn't have to be f32 —
        the runtime can deliver inputs at the model's compute dtype
        directly. Declaring the graph input as ``config.dtype`` removes
        an extra Cast node at graph entry.
        """
        # Use a VL model with 3-model split (vision_encoder is separate)
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=32000,
        )
        config.dtype = DTYPE_MAP[dtype_str]
        model_cls = registry.get("llava")
        module = model_cls(config)
        task = get_task("vision-language")
        pkg = task.build(module, config)

        # Vision encoder pixel_values input must match config.dtype
        vision_model = pkg["vision_encoder"]
        pixel_values_input = vision_model.graph.inputs[0]
        assert pixel_values_input.name == "pixel_values"
        assert pixel_values_input.dtype == config.dtype, (
            f"Vision encoder input dtype is {pixel_values_input.dtype}, "
            f"expected {config.dtype} (no Cast at graph entry)"
        )

        # No Cast as the first node — inputs already arrive at the right dtype
        first_node = next(iter(vision_model.graph))
        assert first_node.op_type != "Cast" or first_node.inputs[0].name != "pixel_values", (
            f"Unexpected Cast on pixel_values: graph input dtype {pixel_values_input.dtype} "
            f"should already match the encoder's compute dtype."
        )

    @pytest.mark.parametrize(
        "dtype_str",
        ["f16", "bf16"],
    )
    def test_gemma4_encoder_inputs_match_model_dtype(self, dtype_str):
        """Gemma4 vision/audio encoder inputs match the model's compute dtype."""
        from mobius._configs import Gemma4AudioConfig, Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "sliding_attention"],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            num_kv_shared_layers=1,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                patch_size=16,
                norm_eps=1e-6,
            ),
            audio=Gemma4AudioConfig(
                input_size=16,
                hidden_size=32,
                num_layers=1,
                output_dim=64,
                output_proj_dims=64,
                audio_token_id=255998,
            ),
            dtype=DTYPE_MAP[dtype_str],
        )
        model_cls = registry.get("gemma4")
        module = model_cls(config)
        task = get_task("gemma4")
        pkg = task.build(module, config)

        # Vision encoder pixel_values must match config.dtype
        vision_model = pkg["vision_encoder"]
        pv_input = vision_model.graph.inputs[0]
        assert pv_input.name == "pixel_values"
        assert pv_input.dtype == config.dtype

        # Audio encoder input_features must match config.dtype
        audio_model = pkg["audio_encoder"]
        af_input = audio_model.graph.inputs[0]
        assert af_input.name == "input_features"
        assert af_input.dtype == config.dtype


class TestRegistryCompleteness:
    """Ensure every registered model type has a test config entry."""

    def test_all_registered_models_have_test_coverage(self):
        """Every model_type in the registry must be accounted for.

        A model_type is *covered* if it appears in parametrized test
        configs, auto-generated configs, a specialized test class, or
        the known-untested allowlist.  New registrations that aren't
        covered anywhere will cause this test to fail.
        """
        specialized = specialized_test_model_types()
        known_untested = known_untested_model_types()
        all_covered = (
            {mt for mt, _, _ in ALL_CONFIGS}
            | {mt for mt, _, _ in AUTO_GENERATED_CONFIGS}
            | specialized
            | known_untested
        )
        registered = set(registry.architectures())
        missing = registered - all_covered
        assert not missing, (
            f"Registered model types without test coverage: "
            f"{sorted(missing)}. Add a test config to "
            "tests/_test_configs.py, a specialized test class, or "
            "acknowledge in _KNOWN_UNTESTED_MODEL_TYPES."
        )

    def test_known_untested_is_minimal(self):
        """Entries in _KNOWN_UNTESTED should still be registered.

        If a model_type is removed from the registry, it should also
        be removed from _KNOWN_UNTESTED_MODEL_TYPES.  If a test is
        added for it, it should move to the appropriate config list or
        _SPECIALIZED_TEST_MODEL_TYPES.
        """
        registered = set(registry.architectures())
        stale = known_untested_model_types() - registered
        assert not stale, (
            f"Entries in _KNOWN_UNTESTED_MODEL_TYPES that are no longer "
            f"registered: {sorted(stale)}. Remove them."
        )
