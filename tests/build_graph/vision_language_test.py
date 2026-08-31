# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision-language and multimodal L1 graph-construction tests.

Run the complete L1 suite with ``pytest tests/build_graph``.
"""

from __future__ import annotations

import re

import ml_dtypes
import numpy as np
import onnx_ir as ir
import pytest
from _test_configs import (
    LONGROPE_FACTORS,
    VL_CONFIGS,
    _base_config,
    vl_overrides,
)

from build_graph._support import (
    _assert_outputs_have_shapes_and_dtypes,
    _make_params,
    _run_onnx_checker,
)
from mobius._builder import DTYPE_MAP, build_from_module
from mobius._configs import (
    AudioConfig,
    VisionConfig,
)
from mobius._pipeline_contract import component_presence, optional_input_contract
from mobius._registry import registry
from mobius.integrations.transformers._config_resolver import _default_task_for_model
from mobius.tasks import (
    Phi4MMMultiModalTask,
    Qwen3VLVisionLanguageTask,
    get_task,
)

_VL_MODEL_PARAMS = _make_params(VL_CONFIGS)
_VL_SINGLE_MODEL_TASKS = {"qwen3-vl-vision-language"}
_VL_TWO_MODEL_TASKS = {"vision-encoder-decoder"}


class TestBuildGraphVisionLanguage:
    """Verify multimodal models build correctly."""

    def test_phi4mm_multimodal_graph(self):
        """Build Phi4MM with Phi4MMMultiModalTask and verify 4-model split."""
        config = _base_config(
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
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = Phi4MMMultiModalTask()
        pkg = task.build(module, config)

        # Verify 4-model package structure
        assert "vision_encoder" in pkg, "Should have vision model"
        assert "audio_encoder" in pkg, "Should have audio model"
        assert "embedding" in pkg, "Should have embedding model"
        assert "decoder" in pkg, "Should have decoder model"

        # Vision model: pixel_values + image_sizes → image_features
        vision = pkg["vision_encoder"]
        v_inputs = {inp.name for inp in vision.graph.inputs}
        v_outputs = {out.name for out in vision.graph.outputs}
        assert "pixel_values" in v_inputs
        assert "image_sizes" in v_inputs
        assert "image_features" in v_outputs
        v_inits = list(vision.graph.initializers)
        assert any("img_processor" in n for n in v_inits), (
            "Vision model should have SigLIP initializers"
        )

        # Speech model: audio_embeds + metadata → audio_features (single output)
        speech = pkg["audio_encoder"]
        s_inputs = {inp.name for inp in speech.graph.inputs}
        s_outputs = {out.name for out in speech.graph.outputs}
        assert "audio_embeds" in s_inputs
        assert "audio_sizes" in s_inputs
        assert "audio_projection_mode" in s_inputs
        assert "audio_features" in s_outputs

        # Embedding model: input_ids + features → inputs_embeds
        emb = pkg["embedding"]
        e_inputs = {inp.name for inp in emb.graph.inputs}
        e_outputs = {out.name for out in emb.graph.outputs}
        assert "input_ids" in e_inputs
        assert "image_features" in e_inputs
        assert "audio_features" in e_inputs
        assert "inputs_embeds" in e_outputs

        # Decoder model (pkg["decoder"]): inputs_embeds → logits + KV cache
        decoder = pkg["decoder"]
        d_inputs = {inp.name for inp in decoder.graph.inputs}
        d_outputs = {out.name for out in decoder.graph.outputs}
        assert "inputs_embeds" in d_inputs
        assert "attention_mask" in d_inputs
        assert "position_ids" in d_inputs
        assert "logits" in d_outputs

    def test_llava_vision_language_graph(self):
        """Build LLaVA with 3-model split and verify all components."""
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
        model_cls = registry.get("llava")
        module = model_cls(config)
        task_name = _default_task_for_model("llava")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_internvl2_vision_language_graph(self):
        """Build InternVL2 with 3-model split and verify all components."""
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
        model_cls = registry.get("internvl_chat")
        module = model_cls(config)
        task_name = _default_task_for_model("internvl_chat")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

        # Verify aliases also resolve to InternVL2Model
        from mobius.models.internvl import InternVL2Model

        for alias in ("internvl2", "internvl"):
            alias_cls = registry.get(alias)
            assert alias_cls is InternVL2Model, f"{alias} should map to InternVL2Model"

    def test_qwen2_5_vl_graph(self):
        """Build Qwen2.5-VL with its auto-detected 3-model task."""
        config = _base_config(
            attn_qkv_bias=True,
            mrope_section=[8, 12, 12],
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=14,
                in_channels=3,
                out_hidden_size=64,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            fullatt_block_indexes=[1],
            image_token_id=151655,
        )
        model_cls = registry.get("qwen2_5_vl")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen2_5_vl")
        task = get_task(task_name)
        pkg = task.build(module, config)

        # 3-model split: decoder, vision, embedding
        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        # Decoder: inputs_embeds → logits + KV cache
        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        # Vision: pixel_values → image_features
        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        # Embedding: input_ids + image_features → inputs_embeds
        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_glm_ocr_graph(self):
        """Build GLM-OCR's dedicated vision tower and GLM text decoder."""
        config = _base_config(
            hidden_size=64,
            intermediate_size=192,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            attn_qkv_bias=False,
            mrope_section=[2, 3, 3],
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=14,
                in_channels=3,
                out_hidden_size=64,
                norm_eps=1e-5,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            image_token_id=59280,
            vision_start_token_id=59256,
            vision_end_token_id=59257,
        )
        module = registry.get("glm_ocr")(config)
        task = get_task(_default_task_for_model("glm_ocr"))
        pkg = task.build(module, config)

        assert set(pkg) == {"decoder", "vision_encoder", "embedding"}
        assert {value.name for value in pkg["vision_encoder"].graph.inputs} == {
            "pixel_values",
            "image_grid_thw",
        }
        assert {value.name for value in pkg["vision_encoder"].graph.outputs} == {
            "image_features"
        }
        assert "inputs_embeds" in {value.name for value in pkg["decoder"].graph.inputs}
        assert "logits" in {value.name for value in pkg["decoder"].graph.outputs}
        assert {value.name for value in pkg["embedding"].graph.inputs} == {
            "input_ids",
            "image_features",
        }

    def test_qwen2_5_vl_text_graph(self):
        """Build Qwen2.5-VL text-only model."""
        config = _base_config(attn_qkv_bias=True, mrope_section=[8, 12, 12])
        model_cls = registry.get("qwen2_5_vl_text")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen2_5_vl_text")
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]
        assert model.graph is not None
        assert "logits" in {out.name for out in model.graph.outputs}

    def test_qwen_image_edit_text_encoder_graphs(self):
        """Build the Qwen2.5-VL hidden-state encoder split used by image edit."""
        from mobius.tasks import QwenImageTextEncoderTask

        config = _base_config(
            attn_qkv_bias=True,
            mrope_section=[8, 12, 12],
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=14,
                in_channels=3,
                out_hidden_size=64,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            fullatt_block_indexes=[1],
            image_token_id=151655,
        )
        module = registry.get("qwen2_5_vl")(config)
        package = QwenImageTextEncoderTask().build(module, config)

        assert set(package) == {"model", "vision_encoder", "embedding"}
        assert {output.name for output in package["model"].graph.outputs} == {
            "prompt_embeds",
            "prompt_embeds_mask",
        }

    def test_qwen3_vl_graph(self):
        """Build Qwen3-VL with its auto-detected 3-model task."""
        config = _base_config(
            attn_qk_norm=True,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=16,
                in_channels=3,
                out_hidden_size=64,
                num_position_embeddings=16,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            image_token_id=151655,
            mrope_section=[8, 12, 12],
        )
        model_cls = registry.get("qwen3_vl")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen3_vl")
        pkg = build_from_module(module, config, task=task_name, execution_provider="cpu")

        # 3-model split produces decoder, vision, embedding
        assert "decoder" in pkg
        assert "vision_encoder" in pkg
        assert "embedding" in pkg

        # Decoder should have logits output and inputs_embeds input
        decoder = pkg["decoder"]
        assert "logits" in {out.name for out in decoder.graph.outputs}
        assert "inputs_embeds" in {inp.name for inp in decoder.graph.inputs}
        assert "per_layer_inputs" in {inp.name for inp in decoder.graph.inputs}
        assert "deepstack_embeds" not in {inp.name for inp in decoder.graph.inputs}
        num_deepstack = len(config.deepstack_visual_indexes)
        per_layer_input = next(
            inp for inp in decoder.graph.inputs if inp.name == "per_layer_inputs"
        )
        assert per_layer_input.shape[-1] == num_deepstack * config.hidden_size

        vision_outputs = {out.name for out in pkg["vision_encoder"].graph.outputs}
        assert vision_outputs == {"image_features"}
        assert (
            pkg["vision_encoder"].graph.outputs[0].shape[-1]
            == (num_deepstack + 1) * config.hidden_size
        )
        embedding_inputs = {inp.name for inp in pkg["embedding"].graph.inputs}
        embedding_outputs = {out.name for out in pkg["embedding"].graph.outputs}
        assert embedding_inputs == {"input_ids", "image_features"}
        assert embedding_outputs == {"inputs_embeds", "per_layer_inputs"}
        image_features = next(
            inp for inp in pkg["embedding"].graph.inputs if inp.name == "image_features"
        )
        per_layer_output = next(
            out for out in pkg["embedding"].graph.outputs if out.name == "per_layer_inputs"
        )
        assert image_features.shape[-1] == (num_deepstack + 1) * config.hidden_size
        assert per_layer_output.shape[-1] == num_deepstack * config.hidden_size

        vision_encoder = pkg["vision_encoder"]
        node_order = {id(node): index for index, node in enumerate(vision_encoder.graph)}
        for index, node in enumerate(vision_encoder.graph):
            for input_value in node.inputs:
                producer = input_value.producer() if input_value is not None else None
                if producer is not None and producer.graph is vision_encoder.graph:
                    assert node_order[id(producer)] < index

    def test_qwen35_vl_graph(self):
        """Build Qwen3.5-VL with its auto-detected 3-model task."""
        config = _base_config(
            attn_qk_norm=True,
            partial_rotary_factor=0.5,
            layer_types=["linear_attention", "full_attention"],
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=16,
                in_channels=3,
                out_hidden_size=64,
                num_position_embeddings=16,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            image_token_id=248056,
            mrope_section=[8, 12, 12],
        )
        model_cls = registry.get("qwen3_5_vl")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen3_5_vl")
        task = get_task(task_name)
        pkg = task.build(module, config)

        # 3-model split produces decoder, vision, embedding
        assert "decoder" in pkg
        assert "vision_encoder" in pkg
        assert "embedding" in pkg

        # Decoder should have logits output and inputs_embeds input
        decoder = pkg["decoder"]
        assert "logits" in {out.name for out in decoder.graph.outputs}
        assert "inputs_embeds" in {inp.name for inp in decoder.graph.inputs}
        assert "per_layer_inputs" in {inp.name for inp in decoder.graph.inputs}
        assert {out.name for out in pkg["vision_encoder"].graph.outputs} == {"image_features"}
        assert {out.name for out in pkg["embedding"].graph.outputs} == {
            "inputs_embeds",
            "per_layer_inputs",
        }

        # Verify hybrid cache: linear_attention layer gets conv_state/recurrent_state,
        # full_attention layer gets key/value
        output_names = {out.name for out in decoder.graph.outputs}
        assert "present.0.conv_state" in output_names
        assert "present.0.recurrent_state" in output_names
        assert "present.1.key" in output_names
        assert "present.1.value" in output_names

    def test_qwen3_vl_single_model_graph(self):
        """Build Qwen3-VL with single-model Qwen3VLVisionLanguageTask."""
        config = _base_config(
            attn_qk_norm=True,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=16,
                in_channels=3,
                out_hidden_size=64,
                num_position_embeddings=16,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            image_token_id=151655,
            mrope_section=[8, 12, 12],
        )
        model_cls = registry.get("qwen3_vl_single")
        module = model_cls(config)
        task = Qwen3VLVisionLanguageTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        assert model.graph is not None
        assert "logits" in {out.name for out in model.graph.outputs}

    def test_gemma3_multimodal_graph(self):
        """Build Gemma3 multimodal model with 3-model split."""
        config = _base_config(
            attn_qk_norm=True,
            rope_local_base_freq=10_000.0,
            layer_types=["full_attention", "sliding_attention"],
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            mm_tokens_per_image=4,
            image_token_id=255999,
        )
        model_cls = registry.get("gemma3")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma3")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}
        assert "pixel_values" in {i.name for i in pkg["vision_encoder"].graph.inputs}
        assert "logits" in {o.name for o in pkg["decoder"].graph.outputs}

    def test_gemma4_multimodal_graph(self):
        """Build Gemma4 vision-language model via registry (3-model split: decoder+vision+embedding).

        The ``gemma4`` model type maps to Gemma4Model.  Without an audio config,
        the package has three models: decoder, vision, embedding.
        """
        from mobius._configs import Gemma4Config

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
            # Dual layer types: 1 local + 1 global
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            # Global attention config (same head_dim in test for simplicity)
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                patch_size=16,
                norm_eps=1e-6,
            ),
        )
        model_cls = registry.get("gemma4")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma4")
        task = get_task(task_name)
        pkg = build_from_module(module, config, task=task)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}, (
            f"Vision-only Gemma4 should produce 3 models, got: {set(pkg.keys())}"
        )
        # Decoder: inputs_embeds -> logits + per-layer KV cache
        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}
        # Vision: pixel_values + pixel_position_ids -> image_features
        vision = pkg["vision_encoder"]
        vision_input_names = {i.name for i in vision.graph.inputs}
        assert "pixel_values" in vision_input_names
        assert "pixel_position_ids" in vision_input_names
        assert "image_features" in {o.name for o in vision.graph.outputs}
        assert component_presence(vision.graph) == "image"
        # Embedding: input_ids + image_features (no audio) -> inputs_embeds
        embedding = pkg["embedding"]
        emb_input_names = {i.name for i in embedding.graph.inputs}
        assert "input_ids" in emb_input_names
        assert "image_features" in emb_input_names
        assert "audio_features" not in emb_input_names
        embedding_image = next(i for i in embedding.graph.inputs if i.name == "image_features")
        assert optional_input_contract(embedding_image) == {
            "presence": "image",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        assert "inputs_embeds" in {o.name for o in embedding.graph.outputs}

    def test_gemma4_kv_shared_fallback_attention_is_causal_zero(self):
        """KV-shared layers must use is_causal=0 in the non-GQA fallback.

        The shared-KV fallback feeds the full borrowed K/V as key/value with
        no past (so q_len < kv_len during decode) and relies on the float
        causal bias from ``create_attention_bias`` for masking.  It must NOT
        also set is_causal=1: the ONNX Attention op's built-in causal mask is
        upper-left aligned on CUDA but bottom-right on CPU, so double-masking
        diverges across EPs.  Source/non-shared layers keep is_causal=1.
        """
        from mobius._configs import Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=4,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            pad_token_id=0,
            tie_word_embeddings=True,
            num_kv_shared_layers=2,
        )
        module = registry.get("gemma4_text")(config)
        task = get_task(_default_task_for_model("gemma4_text"))
        pkg = task.build(module, config)
        model = pkg["model"]

        # The fp32 build (no EP) takes the ONNX Attention fallback path.
        is_causal_by_layer: dict[int, int] = {}
        for node in model.graph:
            if node.op_type != "Attention":
                continue
            m = re.search(r"layers\.(\d+)/self_attn", node.name)
            assert m is not None, node.name
            layer_idx = int(m.group(1))
            attr = next(a for a in node.attributes.values() if a.name == "is_causal")
            is_causal_by_layer[layer_idx] = attr.value

        # Source/non-shared layers (0,1) keep is_causal=1; the last
        # num_kv_shared_layers layers (2,3) must use is_causal=0.
        assert is_causal_by_layer == {0: 1, 1: 1, 2: 0, 3: 0}, is_causal_by_layer

    def test_gemma4_moe_graph(self):
        """Build Gemma4 text-only model with enable_moe_block=True (MoE path)."""
        from mobius._configs import Gemma4Config

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
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            pad_token_id=0,
            tie_word_embeddings=True,
            # MoE config: every layer has a parallel MoE block
            enable_moe_block=True,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
        )
        model_cls = registry.get("gemma4_text")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma4_text")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert "model" in pkg
        model = pkg["model"]
        input_names = {i.name for i in model.graph.inputs}
        output_names = {o.name for o in model.graph.outputs}
        assert "input_ids" in input_names
        assert "logits" in output_names

    def test_gemma4_unified_text_graph(self):
        """Build the gemma4_unified (gemma-4-12B) text backbone via Gemma4CausalLMModel.

        ``gemma4_unified_text`` reuses Gemma4CausalLMModel.  This exercises the
        12B-family text architecture: dual head_dim (local 16 / global 32),
        ``attention_k_eq_v`` with a single global KV head, vision-block
        bidirectional attention, and final-logit softcapping.
        """
        from mobius._configs import Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=32,
            global_rope_theta=1_000_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=30.0,
            hidden_size_per_layer_input=0,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            use_bidirectional_attention="vision",
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        model_cls = registry.get("gemma4_unified_text")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma4_unified_text")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert "model" in pkg
        model = pkg["model"]
        input_names = {i.name for i in model.graph.inputs}
        output_names = {o.name for o in model.graph.outputs}
        assert "input_ids" in input_names
        assert "logits" in output_names
        # Full-attention layer uses the single global KV head, so its cache
        # entry has a different head_dim than the sliding layer.
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.1.key" in input_names

    def test_gemma4_unified_text_only_emits_gqa(self):
        """text_only build of gemma-4-12B emits GroupQueryAttention on CUDA.

        The multimodal ``gemma4_unified`` decoder uses the bidirectional
        vision-block overlay (float-bias ``Attention``), but the text-only
        export strips ``image_token_id`` / ``use_bidirectional_attention`` so
        the decoder is pure causal and fuses to ``GroupQueryAttention`` on a
        GQA-capable execution provider. This mirrors what
        ``build(text_only=True)`` produces, without network access.
        """
        from collections import Counter

        from mobius._configs import Gemma4Config
        from mobius.integrations.transformers._builder import _strip_to_text_only
        from mobius.tasks import get_task

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=32,
            global_rope_theta=1_000_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=30.0,
            hidden_size_per_layer_input=0,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            use_bidirectional_attention="vision",
            image_token_id=258880,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        config = _strip_to_text_only(config, "gemma4_unified_text")
        config.dtype = DTYPE_MAP["f16"]
        assert config.image_token_id is None
        assert config.use_bidirectional_attention is None

        model_cls = registry.get("gemma4_unified_text")
        module = model_cls(config)
        task = get_task(_default_task_for_model("gemma4_unified_text"))
        pkg = build_from_module(module, config, task=task, execution_provider="cuda")

        counts = Counter(n.op_type for n in pkg["model"].graph)
        assert counts.get("GroupQueryAttention", 0) == 2, dict(counts)
        assert counts.get("Attention", 0) == 0, dict(counts)

    def test_strip_to_text_only(self):
        """``_strip_to_text_only`` nulls multimodal fields and sets model_type."""
        from mobius._configs import Gemma4AudioConfig, Gemma4Config
        from mobius.integrations.transformers._builder import _strip_to_text_only

        config = Gemma4Config(
            model_type="gemma4_unified",
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            hidden_act="gelu_pytorch_tanh",
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=32,
            use_bidirectional_attention="vision",
            image_token_id=258880,
            boa_token_id=256000,
            audio=Gemma4AudioConfig(),
            pad_token_id=0,
        )
        out = _strip_to_text_only(config, "gemma4_unified_text")

        assert out.model_type == "gemma4_unified_text"
        assert out.image_token_id is None
        assert out.use_bidirectional_attention is None
        assert out.boa_token_id is None
        assert out.audio is None
        assert out.vision is None
        # Original config is untouched (dataclasses.replace returns a copy).
        assert config.image_token_id == 258880

    def test_build_text_only_unsupported_model_type_raises(self):
        """``build(text_only=True)`` rejects model types with no text sibling."""
        from unittest import mock

        from mobius.integrations.transformers import build

        fake_hf = type("HF", (), {"model_type": "llama"})()
        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf),
            pytest.raises(ValueError, match="text_only=True is not supported"),
        ):
            build("meta-llama/Llama-3.2-1B", load_weights=False, text_only=True)

    def test_build_text_only_remaps_and_strips(self):
        """``build(text_only=True)`` remaps to the text sibling and strips config.

        Happy path: the model_type is remapped to its text-only registry
        sibling and the multimodal config is stripped before building.
        """
        from unittest import mock

        from mobius.integrations.transformers import _builder as transformers_builder

        fake_hf = type("HF", (), {"model_type": "gemma4_unified"})()
        raw_config = mock.MagicMock(name="raw_config")
        stripped_config = mock.MagicMock(name="stripped_config")
        fake_pkg = mock.MagicMock()
        fake_pkg.items.return_value = []
        fake_module_cls = mock.MagicMock(name="Gemma4CausalLMModel")

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf),
            mock.patch.object(
                transformers_builder.registry, "get", return_value=fake_module_cls
            ) as mock_get,
            mock.patch(
                "mobius.integrations.transformers._config_resolver._config_from_hf",
                return_value=raw_config,
            ),
            mock.patch(
                "mobius.integrations.transformers._builder._strip_to_text_only",
                return_value=stripped_config,
            ) as mock_strip,
            mock.patch(
                "mobius.integrations.transformers._config_resolver._default_task_for_model",
                return_value="text-generation",
            ),
            mock.patch(
                "mobius.integrations.transformers._builder.build_from_module",
                return_value=fake_pkg,
            ) as mock_build_mod,
        ):
            pkg = transformers_builder.build_transformers_model(
                "google/gemma-4-12B", load_weights=False, text_only=True
            )

        # model_type was remapped to the text sibling before module lookup
        mock_get.assert_called_once_with("gemma4_unified_text")
        # config stripping invoked with the remapped (text) model_type
        mock_strip.assert_called_once_with(raw_config, "gemma4_unified_text")
        # the stripped config (not the raw multimodal one) is what gets built
        assert mock_build_mod.call_args.args[1] is stripped_config
        assert pkg is fake_pkg

    def test_build_text_only_diffusers_path_raises(self):
        """``build(text_only=True)`` errors on the diffusers/unsupported path.

        When AutoConfig fails and the model is not in the registry, ``build``
        normally falls through to ``build_diffusers_pipeline``. With
        ``text_only=True`` it must raise instead of silently ignoring the flag.
        """
        from unittest import mock

        from mobius.integrations.transformers import build

        with (
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=ValueError("no such model_type"),
            ),
            mock.patch(
                "mobius.integrations.transformers._config_resolver._try_load_config_json",
                return_value=None,
            ),
            pytest.raises(ValueError, match="does not resolve to a registered"),
        ):
            build("some/diffusion-pipeline", load_weights=False, text_only=True)

    def test_gemma4_any_to_any_graph(self):
        """Build Gemma4 Any-to-Any model (4-model split: decoder+vision+speech+embedding).

        When ``config.audio`` is set, Gemma4Model adds a ``speech`` model and a
        3-input embedding (input_ids + image_features + audio_features).
        """
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
            # num_kv_shared_layers=1 → layer 1 shares KV from layer 0 (same type)
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
        )
        model_cls = registry.get("gemma4")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma4")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {
            "decoder",
            "vision_encoder",
            "audio_encoder",
            "embedding",
        }, f"AnyToAny Gemma4 should produce 4 models (with 'audio'), got: {set(pkg.keys())}"
        # Decoder KV cache: num_hidden_layers - num_kv_shared_layers = 1 entry
        decoder = pkg["decoder"]
        decoder_input_names = {i.name for i in decoder.graph.inputs}
        assert "inputs_embeds" in decoder_input_names
        assert "past_key_values.0.key" in decoder_input_names
        assert "past_key_values.1.key" not in decoder_input_names  # shared layer
        assert "logits" in {o.name for o in decoder.graph.outputs}
        # Vision
        vision = pkg["vision_encoder"]
        vision_input_names = {i.name for i in vision.graph.inputs}
        assert "pixel_values" in vision_input_names
        assert "pixel_position_ids" in vision_input_names
        assert "image_features" in {o.name for o in vision.graph.outputs}
        assert component_presence(vision.graph) == "image"
        # Audio encoder
        audio = pkg["audio_encoder"]
        audio_input_names = {i.name for i in audio.graph.inputs}
        assert "input_features" in audio_input_names
        assert "input_features_mask" in audio_input_names
        audio_features = next(o for o in audio.graph.outputs if o.name == "audio_features")
        assert len(audio_features.shape) == 2
        assert audio_features.shape[-1] == config.hidden_size
        assert component_presence(audio.graph) == "audio"
        # Embedding: all three inputs
        embedding = pkg["embedding"]
        emb_input_names = {i.name for i in embedding.graph.inputs}
        assert "input_ids" in emb_input_names
        assert "image_features" in emb_input_names
        assert "audio_features" in emb_input_names
        embedding_image = next(i for i in embedding.graph.inputs if i.name == "image_features")
        assert optional_input_contract(embedding_image) == {
            "presence": "image",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        embedding_audio = next(i for i in embedding.graph.inputs if i.name == "audio_features")
        assert optional_input_contract(embedding_audio) == {
            "presence": "audio",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        assert "inputs_embeds" in {o.name for o in embedding.graph.outputs}
        # KV cache outputs: num_kv_layers = num_hidden_layers - num_kv_shared_layers = 1
        decoder_output_names = {o.name for o in decoder.graph.outputs}
        assert "present.0.key" in decoder_output_names
        assert "present.0.value" in decoder_output_names
        assert "present.1.key" not in decoder_output_names  # shared layer excluded
        assert "present.1.value" not in decoder_output_names  # shared layer excluded

    @pytest.mark.parametrize(
        ("dtype", "np_dtype"),
        [
            (ir.DataType.FLOAT, np.float32),
            (ir.DataType.FLOAT16, np.float16),
            (ir.DataType.BFLOAT16, ml_dtypes.bfloat16),
        ],
    )
    def test_gemma4_audio_encoder_strips_padding_in_graph(self, dtype, np_dtype):
        """The exported audio graph produces ordered rank-2 valid feature rows."""
        from onnxscript import nn

        from mobius._configs import Gemma4AudioConfig, Gemma4Config
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.tasks._gemma4 import Gemma4Task

        class IdentityAudio(nn.Module):
            def forward(self, op, input_features, input_features_mask=None):
                return op.Identity(input_features), op.Identity(input_features_mask)

        config = Gemma4Config(
            hidden_size=4,
            dtype=dtype,
            audio=Gemma4AudioConfig(input_size=4),
        )
        model = Gemma4Task()._build_audio(IdentityAudio(), config)
        features = np.arange(24, dtype=np_dtype).reshape(2, 3, 4)
        mask = np.array([[True, True, False], [True, False, False]])

        session = OnnxModelSession(model)
        outputs = session.run(
            {
                "input_features": features,
                "input_features_mask": mask,
            }
        )
        session.close()

        np.testing.assert_array_equal(
            outputs["audio_features"],
            np.concatenate([features[0, :2], features[1, :1]], axis=0),
        )
        assert outputs["audio_features"].dtype == np.dtype(np_dtype)

    def test_gemma4_unified_multimodal_graph(self):
        """Build gemma4_unified (gemma-4-12B) encoder-free multimodal model.

        Produces a 4-model split (decoder + vision_encoder + audio_encoder +
        embedding).  The vision/audio encoders are encoder-free embedders
        (no SigLIP/Conformer tower); the decoder uses vision-block
        bidirectional attention, which it derives internally from
        ``input_ids`` (the embedding model does *not* emit
        ``block_sequence_ids``).
        """
        from mobius._configs import Gemma4AudioConfig, Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=32,
            global_rope_theta=1_000_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=30.0,
            hidden_size_per_layer_input=0,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            use_bidirectional_attention="vision",
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            vision=VisionConfig(
                hidden_size=48,
                patch_size=4,
                pooling_kernel_size=2,
                position_embedding_size=64,
                out_hidden_size=48,
                norm_eps=1e-6,
            ),
            audio=Gemma4AudioConfig(
                hidden_size=40,
                output_proj_dims=40,
                audio_token_id=255998,
            ),
        )
        model_cls = registry.get("gemma4_unified")
        module = model_cls(config)
        task = get_task(_default_task_for_model("gemma4_unified"))
        pkg = build_from_module(module, config, task=task)

        assert set(pkg.keys()) == {
            "decoder",
            "vision_encoder",
            "audio_encoder",
            "embedding",
        }, f"gemma4_unified should produce 4 models, got: {set(pkg.keys())}"

        # Vision embedder: raw patches (no encoder layers) → image_features
        vision = pkg["vision_encoder"]
        v_inputs = {i.name for i in vision.graph.inputs}
        assert v_inputs == {"pixel_values", "pixel_position_ids"}
        assert "image_features" in {o.name for o in vision.graph.outputs}
        assert component_presence(vision.graph) == "image"

        # Audio embedder: raw frames + mask → audio_features
        audio = pkg["audio_encoder"]
        a_inputs = {i.name for i in audio.graph.inputs}
        assert a_inputs == {"input_features", "input_features_mask"}
        assert "audio_features" in {o.name for o in audio.graph.outputs}
        assert component_presence(audio.graph) == "audio"

        # Embedding: fuses both modalities → inputs_embeds (no block_sequence_ids;
        # the decoder derives the bidirectional overlay from input_ids itself)
        embedding = pkg["embedding"]
        e_inputs = {i.name for i in embedding.graph.inputs}
        assert {"input_ids", "image_features", "audio_features"} <= e_inputs
        embedding_image = next(i for i in embedding.graph.inputs if i.name == "image_features")
        assert optional_input_contract(embedding_image) == {
            "presence": "image",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        embedding_audio = next(i for i in embedding.graph.inputs if i.name == "audio_features")
        assert optional_input_contract(embedding_audio) == {
            "presence": "audio",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        e_outputs = {o.name for o in embedding.graph.outputs}
        assert "inputs_embeds" in e_outputs
        assert "block_sequence_ids" not in e_outputs

        # Decoder: consumes inputs_embeds + input_ids (for the vision-block
        # bidirectional overlay, derived internally)
        decoder = pkg["decoder"]
        d_inputs = {i.name for i in decoder.graph.inputs}
        assert "inputs_embeds" in d_inputs
        assert "input_ids" in d_inputs
        assert "block_sequence_ids" not in d_inputs
        assert "logits" in {o.name for o in decoder.graph.outputs}

    def test_gemma4_kv_shared_layer_tracing(self):
        """Verify all num_hidden_layers are traced and KV outputs = num_kv_layers.

        With num_kv_shared_layers=1 and num_hidden_layers=2:
        - Both layers must be traced (Attention op count = 2)
        - KV cache inputs: 1 entry (only layer 0 has its own K/V)
        - KV cache outputs: 1 entry (shared layer excluded from present_key_values)
        """
        from mobius._configs import Gemma4AudioConfig, Gemma4Config
        from mobius.tasks._gemma4 import Gemma4Task

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
        )
        model_cls = registry.get("gemma4")
        module = model_cls(config)
        task = Gemma4Task()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]

        # All num_hidden_layers=2 layers must be traced: each produces one Attention op.
        attention_nodes = [n for n in decoder.graph if n.op_type == "Attention"]
        assert len(attention_nodes) == config.num_hidden_layers, (
            f"Expected {config.num_hidden_layers} Attention ops (all layers traced), "
            f"got {len(attention_nodes)}"
        )

        # KV cache inputs: exactly num_kv_layers = 1 (shared layer has no own KV)
        input_names = {i.name for i in decoder.graph.inputs}
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.1.key" not in input_names

        # KV cache outputs: exactly num_kv_layers = 1 (shared layer excluded)
        output_names = {o.name for o in decoder.graph.outputs}
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names
        assert "present.1.key" not in output_names
        assert "present.1.value" not in output_names

    def test_gemma4_k_eq_v_with_global_kv_heads(self):
        """Verify attention_k_eq_v removes v_proj and num_global_key_value_heads sets KV cache shapes.

        Config: attention_k_eq_v=True, num_key_value_heads=4 (sliding),
        num_global_key_value_heads=2 (full). Full-attention layers should:
        - Have no v_proj initializer (V=K)
        - Use num_global_key_value_heads=2 for KV cache shapes
        Sliding layers should use num_key_value_heads=4.
        """
        from mobius._configs import Gemma4Config
        from mobius.models.gemma4 import Gemma4CausalLMModel
        from mobius.tasks._gemma4 import Gemma4TextCausalLMTask

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            # Layer 0: sliding, Layer 1: full (k_eq_v + global heads)
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            attention_k_eq_v=True,
            num_global_key_value_heads=2,
        )
        module = Gemma4CausalLMModel(config)
        task = Gemma4TextCausalLMTask()
        pkg = task.build(module, config)
        decoder = pkg["model"]

        # Check initializer names: full-attention layer (1) should have no v_proj
        init_names = set(decoder.graph.initializers)
        # Sliding layer 0 has k_proj, v_proj
        assert "model.layers.0.self_attn.k_proj.weight" in init_names
        assert "model.layers.0.self_attn.v_proj.weight" in init_names
        # Full layer 1 has k_proj but NO v_proj (k_eq_v: V=K)
        assert "model.layers.1.self_attn.k_proj.weight" in init_names
        assert "model.layers.1.self_attn.v_proj.weight" not in init_names

        # KV cache shapes:
        # Layer 0 (sliding): num_key_value_heads=4
        # Layer 1 (full): num_global_key_value_heads=2
        input_shapes = {i.name: list(i.shape) for i in decoder.graph.inputs}
        # Layer 0: kv_heads=4
        layer0_key_shape = input_shapes["past_key_values.0.key"]
        assert layer0_key_shape[1] == 4, (
            f"Sliding layer 0 should have 4 KV heads, got {layer0_key_shape[1]}"
        )
        # Layer 1: kv_heads=2 (num_global_key_value_heads)
        layer1_key_shape = input_shapes["past_key_values.1.key"]
        assert layer1_key_shape[1] == 2, (
            f"Full layer 1 should have 2 KV heads "
            f"(num_global_key_value_heads), got {layer1_key_shape[1]}"
        )

    def test_gemma3n_multimodal_graph(self):
        """Build Gemma 3n via the registry (4-model split, audio configured).

        The tiny config is taken from ``VL_CONFIGS`` so this test and the
        parametrized suite can never drift apart.
        """
        config = _base_config(**vl_overrides("gemma3n"))
        module = registry.get("gemma3n")(config)
        task_name = _default_task_for_model("gemma3n")
        assert task_name == "gemma3n"
        pkg = get_task(task_name).build(module, config)

        assert set(pkg.keys()) == {
            "decoder",
            "vision_encoder",
            "audio_encoder",
            "embedding",
        }, f"Gemma3n with audio should produce 4 models, got: {set(pkg.keys())}"

        # --- decoder: inputs_embeds + per_layer_inputs -> logits + KV cache.
        # The per-layer embedding tables live in the embedding sub-model, so
        # the decoder takes their combined output as a graph input and never
        # sees input_ids.
        decoder = pkg["decoder"]
        decoder_inputs = {i.name: i for i in decoder.graph.inputs}
        assert "input_ids" not in decoder_inputs
        assert "inputs_embeds" in decoder_inputs
        assert "per_layer_inputs" in decoder_inputs
        assert list(decoder_inputs["per_layer_inputs"].shape)[-1] == (
            config.num_hidden_layers * config.hidden_size_per_layer_input
        )
        decoder_outputs = {o.name for o in decoder.graph.outputs}
        assert "logits" in decoder_outputs
        # num_kv_shared_layers=0 here, so every layer owns a cache entry.
        assert module.decoder.kv_cache_layer_count() == config.num_hidden_layers
        for i in range(config.num_hidden_layers):
            assert f"past_key_values.{i}.key" in decoder_inputs
            assert f"present.{i}.key" in decoder_outputs

        # --- vision: fixed-size pixels -> [B*256, hidden], no mask.
        vision = pkg["vision_encoder"]
        vision_inputs = {i.name: i for i in vision.graph.inputs}
        assert set(vision_inputs) == {"pixel_values"}
        image_size = config.vision.image_size
        assert list(vision_inputs["pixel_values"].shape)[1:] == [3, image_size, image_size]
        image_features = next(o for o in vision.graph.outputs if o.name == "image_features")
        assert len(image_features.shape) == 2
        assert image_features.shape[-1] == config.hidden_size
        assert component_presence(vision.graph) == "image"

        # --- audio: mel frames + bool mask -> fixed-count [B*188, hidden].
        # Unlike Gemma 4, padded rows are not stripped, so there is no
        # companion audio_features_mask output.
        audio = pkg["audio_encoder"]
        audio_inputs = {i.name: i for i in audio.graph.inputs}
        assert set(audio_inputs) == {"input_features", "input_features_mask"}
        assert list(audio_inputs["input_features"].shape)[-1] == (config.audio.input_feat_size)
        assert audio_inputs["input_features_mask"].dtype == ir.DataType.BOOL
        assert {o.name for o in audio.graph.outputs} == {"audio_features"}
        audio_features = next(o for o in audio.graph.outputs if o.name == "audio_features")
        assert len(audio_features.shape) == 2
        assert audio_features.shape[-1] == config.hidden_size
        assert component_presence(audio.graph) == "audio"

        # --- embedding: ids + both feature sets -> inputs_embeds + per_layer.
        embedding = pkg["embedding"]
        emb_inputs = {i.name: i for i in embedding.graph.inputs}
        assert set(emb_inputs) == {"input_ids", "image_features", "audio_features"}
        for name, presence in (("image_features", "image"), ("audio_features", "audio")):
            assert optional_input_contract(emb_inputs[name]) == {
                "presence": presence,
                "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
            }, name
        assert {o.name for o in embedding.graph.outputs} == {
            "inputs_embeds",
            "per_layer_inputs",
        }
        # The embedding's per_layer_inputs must match what the decoder expects.
        emb_per_layer = next(
            o for o in embedding.graph.outputs if o.name == "per_layer_inputs"
        )
        assert (
            list(emb_per_layer.shape)[-1]
            == (list(decoder_inputs["per_layer_inputs"].shape)[-1])
        )
        # The 4.7 GB per-layer table belongs to the embedding model only.
        assert "embedding.embed_tokens_per_layer.weight" in embedding.graph.initializers
        assert not any("embed_tokens_per_layer" in n for n in decoder.graph.initializers)
        # Only the *hard* embedder path is built here, so the soft-path norm
        # (which the towers own) must not add a dangling initializer.
        assert "embedding.embed_vision.hard_embedding_norm.weight" in (
            embedding.graph.initializers
        )
        assert not any("soft_embedding_norm" in n for n in embedding.graph.initializers)

    def test_gemma3n_multimodal_graph_without_audio(self):
        """With ``config.audio=None`` the package drops the audio encoder.

        The embedding model must then also drop its ``audio_features`` input,
        or the runtime would be asked for a tensor no component produces.
        """
        overrides = vl_overrides("gemma3n")
        overrides["audio"] = None
        overrides["audio_token_id"] = None
        config = _base_config(**overrides)
        module = registry.get("gemma3n")(config)
        pkg = get_task("gemma3n").build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}
        assert module.audio_encoder is None
        emb_input_names = {i.name for i in pkg["embedding"].graph.inputs}
        assert "image_features" in emb_input_names
        assert "audio_features" not in emb_input_names
        # No audio embedder weights should be built either.
        assert not any("embed_audio" in n for n in pkg["embedding"].graph.initializers)

    def test_blip2_vision_language_graph(self):
        """Build BLIP-2 with ViT + Q-Former + LLM 3-model split."""
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
            image_token_id=50265,
            # Q-Former config
            num_query_tokens=4,
            qformer_hidden_size=32,
            qformer_num_hidden_layers=1,
            qformer_num_attention_heads=2,
            qformer_intermediate_size=64,
        )
        model_cls = registry.get("blip-2")
        module = model_cls(config)
        task_name = _default_task_for_model("blip-2")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        # Decoder: inputs_embeds → logits + KV cache
        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        # Vision: pixel_values → image_features (via ViT + Q-Former)
        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        # Embedding: input_ids + image_features → inputs_embeds
        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_llava_aliases_build(self):
        """LLaVA aliases (llava_next, llava_onevision, video_llava, etc.) all build."""
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
        for model_type in (
            "aya_vision",
            "cohere2_vision",
            "deepseek_vl",
            "deepseek_vl_hybrid",
            "glm4v",
            "glm4v_moe",
            "got_ocr2",
            "instructblipvideo",
            "janus",
            "llava_next",
            "llava_next_video",
            "llava_onevision",
            "ovis2",
            "smolvlm",
            "video_llava",
            "vipllava",
            "chameleon",
            "florence2",
            "fuyu",
            "idefics2",
            "idefics3",
            "instructblip",
            "molmo",
            "paligemma",
            "pixtral",
        ):
            model_cls = registry.get(model_type)
            module = model_cls(config)
            task_name = _default_task_for_model(model_type)
            task = get_task(task_name)
            pkg = task.build(module, config)

            assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}, (
                f"{model_type} should produce 3 models"
            )
            assert "logits" in {o.name for o in pkg["decoder"].graph.outputs}, (
                f"{model_type} decoder missing logits"
            )
            assert "pixel_values" in {i.name for i in pkg["vision_encoder"].graph.inputs}, (
                f"{model_type} vision missing pixel_values"
            )

    def test_mistral3_pixtral_vision_build(self):
        """Build Mistral-3 with Pixtral vision encoder (2D RoPE + patch merge)."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
                model_type="pixtral",
            ),
            image_token_id=32000,
        )
        model_cls = registry.get("mistral3")
        module = model_cls(config)
        task_name = _default_task_for_model("mistral3")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}
        assert "logits" in {o.name for o in pkg["decoder"].graph.outputs}
        assert "pixel_values" in {i.name for i in pkg["vision_encoder"].graph.inputs}

    def test_pixtral_preprocess_weights_remapping(self):
        """Verify _preprocess_pixtral_weights remaps HF weight names correctly."""
        import torch

        from mobius.models.llava import _preprocess_pixtral_weights

        state_dict = {
            "vision_tower.patch_conv.weight": torch.zeros(1),
            "multi_modal_projector.norm.weight": torch.zeros(1),
            "language_model.model.embed_tokens.weight": torch.zeros(1),
            "language_model.lm_head.weight": torch.zeros(1),
            "language_model.model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        }
        result = _preprocess_pixtral_weights(state_dict, tie_word_embeddings=False)

        # Vision/projector keys get vision_encoder. prefix
        assert "vision_encoder.vision_tower.patch_conv.weight" in result
        assert "vision_encoder.multi_modal_projector.norm.weight" in result
        # embed_tokens duplicated to decoder and embedding
        assert "decoder.model.embed_tokens.weight" in result
        assert "embedding.embed_tokens.weight" in result
        # lm_head remapped under decoder
        assert "decoder.lm_head.weight" in result
        # Other language_model keys remapped under decoder
        assert "decoder.model.layers.0.self_attn.q_proj.weight" in result
        # Original keys should not be present
        for original_key in state_dict:
            assert original_key not in result

        # tie_word_embeddings=True creates decoder.lm_head.weight from embed_tokens
        state_dict_tied = {
            "language_model.model.embed_tokens.weight": torch.zeros(1),
        }
        result_tied = _preprocess_pixtral_weights(state_dict_tied, tie_word_embeddings=True)
        assert "decoder.lm_head.weight" in result_tied
        assert "decoder.model.embed_tokens.weight" in result_tied
        assert "embedding.embed_tokens.weight" in result_tied

    def test_pixtral_preprocess_weights_model_prefix_strip(self):
        """Verify outer model. prefix is stripped (Mistral3ForConditionalGeneration)."""
        import torch

        from mobius.models.llava import _preprocess_pixtral_weights

        state_dict = {
            "model.vision_tower.patch_conv.weight": torch.zeros(1),
            "model.language_model.model.layers.0.mlp.gate_proj.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
        }
        result = _preprocess_pixtral_weights(state_dict, tie_word_embeddings=False)

        # model. prefix stripped, then vision gets vision_encoder. prefix
        assert "vision_encoder.vision_tower.patch_conv.weight" in result
        # model. prefix stripped, then language_model remapped to decoder
        assert "decoder.model.layers.0.mlp.gate_proj.weight" in result
        # bare lm_head gets decoder. prefix
        assert "decoder.lm_head.weight" in result

    def test_mllama_vision_language_graph(self):
        """Build Mllama (Llama 3.2 Vision) with cross-attention decoder."""
        from mobius._configs import MllamaConfig

        config = _base_config(
            config_cls=MllamaConfig,
            num_hidden_layers=3,
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
            cross_attention_layers=[1],
        )
        model_cls = registry.get("mllama")
        module = model_cls(config)
        task_name = _default_task_for_model("mllama")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        dec_inputs = {i.name for i in decoder.graph.inputs}
        assert "inputs_embeds" in dec_inputs
        assert "logits" in {o.name for o in decoder.graph.outputs}

        # Cross-attention states must be a decoder input
        assert "cross_attention_states" in dec_inputs

        # Cross-attention layers (layer 1) should use a different
        # past-sequence-length dim than self-attention layers (0, 2)
        kv_shapes = {}
        for inp in decoder.graph.inputs:
            if inp.name.startswith("past_key_values."):
                kv_shapes[inp.name] = str(inp.shape)
        assert kv_shapes["past_key_values.1.key"] != kv_shapes["past_key_values.0.key"]
        assert kv_shapes["past_key_values.0.key"] == kv_shapes["past_key_values.2.key"]

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_deepseek_ocr2_graph(self):
        """Build DeepSeek-OCR-2 with 3-model VL split."""
        config = _base_config(
            # LLM decoder: DeepSeek-V2 non-MLA + MoE
            qk_nope_head_dim=0,
            qk_rope_head_dim=0,
            v_head_dim=0,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            scoring_func="softmax",
            topk_method="greedy",
            first_k_dense_replace=1,
            n_shared_experts=2,
            image_token_id=100015,
        )
        model_cls = registry.get("deepseek_vl_v2")
        module = model_cls(config)
        task_name = _default_task_for_model("deepseek_vl_v2")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_vl_aliases_resolve(self):
        """Verify VL alias model_types resolve to the same class and task."""
        from mobius.models.qwen35 import (
            Qwen35VL3ModelCausalLMModel,
        )
        from mobius.models.qwen_vl import (
            Qwen2VLCausalLMModel,
            Qwen25VLCausalLMModel,
            Qwen25VLTextModel,
        )

        # Qwen2-VL has its own model class (LayerNorm + FCMLP vision)
        assert registry.get("qwen2_vl") is Qwen2VLCausalLMModel
        assert _default_task_for_model("qwen2_vl") == "qwen-vl"

        # Qwen2.5-VL is separate (RMSNorm + GatedMLP vision)
        assert registry.get("qwen2_5_vl") is Qwen25VLCausalLMModel

        assert registry.get("qwen2_vl_text") is Qwen25VLTextModel
        assert registry.get("qwen2_vl_text") is registry.get("qwen2_5_vl_text")

        assert registry.get("qwen3_5") is Qwen35VL3ModelCausalLMModel
        assert registry.get("qwen3_5") is registry.get("qwen3_5_vl")
        assert _default_task_for_model("qwen3_5") == "hybrid-qwen-vl"

    def test_qwen35_vl_preprocess_weights_model_prefix(self):
        """Qwen3.5-VL preprocess handles model.language_model.* style keys."""
        import torch

        from mobius.models.qwen35 import Qwen35VL3ModelCausalLMModel

        vision_config = VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=16,
            in_channels=3,
            out_hidden_size=64,
            num_position_embeddings=16,
        )
        config = _base_config(vision=vision_config)
        module = Qwen35VL3ModelCausalLMModel(config)
        embed_weight = torch.zeros(config.vocab_size, config.hidden_size)
        state_dict = {
            "model.language_model.embed_tokens.weight": embed_weight,
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(
                config.hidden_size, config.hidden_size
            ),
            "model.language_model.lm_head.weight": torch.zeros(
                config.vocab_size, config.hidden_size
            ),
            "model.visual.blocks.0.mlp.linear_fc1.weight": torch.zeros(
                vision_config.intermediate_size, vision_config.hidden_size
            ),
            "mtp_head.weight": torch.zeros(1),
        }

        result = module.preprocess_weights(state_dict)

        assert "decoder.model.embed_tokens.weight" in result
        assert "embedding.embed_tokens.weight" in result
        assert (
            result["decoder.model.embed_tokens.weight"]
            is result["embedding.embed_tokens.weight"]
        )
        assert "decoder.model.layers.0.self_attn.q_proj.weight" in result
        assert "decoder.lm_head.weight" in result
        assert "vision_encoder.visual.blocks.0.mlp.up_proj.weight" in result
        assert "mtp_head.weight" not in result


class TestBuildGraphMultiModal:
    """Verify Phi4MM builds with Phi4MMMultiModalTask (4-model split)."""

    def test_phi4mm_multimodal_graph(self):
        """Build Phi4MM 4-model split and verify all components."""
        config = _base_config(
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
                token_id=200011,
            ),
            image_token_id=200010,
        )
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = Phi4MMMultiModalTask()
        pkg = task.build(module, config)

        # Verify 4 models in package
        assert len(pkg) == 4, f"Expected 4 models, got {len(pkg)}"
        for key in ("vision_encoder", "audio_encoder", "embedding", "decoder"):
            assert key in pkg, f"Missing model: {key}"

        # Vision model has SigLIP encoder initializers
        vision_inits = list(pkg["vision_encoder"].graph.initializers)
        assert any("img_processor" in n for n in vision_inits), (
            "Vision model should have SigLIP initializers"
        )

        # Speech model has Conformer encoder initializers
        speech_inits = list(pkg["audio_encoder"].graph.initializers)
        assert any("encoder" in n for n in speech_inits), (
            "Speech model should have Conformer initializers"
        )

        # Decoder model (pkg["decoder"]) has LoRA initializers
        decoder_inits = list(pkg["decoder"].graph.initializers)
        assert any("lora" in n for n in decoder_inits), (
            "Decoder model should have LoRA initializers"
        )

    def test_phi4_multimodal_alias_resolves(self):
        """Verify phi4_multimodal alias resolves to same class as phi4mm."""
        from mobius.models.phi import Phi4MMMultiModalModel

        assert registry.get("phi4_multimodal") is Phi4MMMultiModalModel
        assert registry.get("phi4_multimodal") is registry.get("phi4mm")
        assert _default_task_for_model("phi4_multimodal") == "phi4mm-multimodal"

    def test_phi3_v_vision_language_graph(self):
        """Build Phi-3-Vision with 3-model split and verify all components."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-5,
            ),
            image_token_id=32044,
        )
        model_cls = registry.get("phi3_v")
        module = model_cls(config)
        task_name = _default_task_for_model("phi3_v")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_phi4_siglip_vision_language_graph(self):
        """Build Phi-4-Reasoning-Vision (phi4-siglip) with 3-model split and verify components."""
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
            image_token_id=-200,
        )
        model_cls = registry.get("phi4-siglip")
        module = model_cls(config)
        task_name = _default_task_for_model("phi4-siglip")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_phi3_v_decoder_excludes_vision_weights(self):
        """Decoder weight preprocessing must not retain vision-only checkpoint tensors."""
        import torch

        from mobius.models.phi3_v import _Phi3VDecoderModel

        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
            ),
            image_token_id=32044,
        )
        weights = {
            "model.layers.0.self_attn.qkv_proj.weight": torch.zeros(128, 64),
            "model.vision_embed_tokens.img_processor.vision_model.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(256, 64),
        }
        remapped = _Phi3VDecoderModel(config).preprocess_weights(weights)

        assert "model.vision_embed_tokens.img_processor.vision_model.weight" not in remapped
        assert "lm_head.weight" in remapped


@pytest.mark.parametrize("model_type,config_overrides", _VL_MODEL_PARAMS)
class TestBuildVLGraph:
    """Verify vision-language models build valid multi-model ONNX packages."""

    def test_package_builds(self, model_type: str, config_overrides: dict):
        """Build a VL model and verify it produces the expected sub-models."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        if task_name in _VL_SINGLE_MODEL_TASKS:
            assert "model" in pkg, f"{model_type} should produce 'model'"
            model = pkg["model"]
            assert model.graph is not None
            output_names = {o.name for o in model.graph.outputs}
            assert "logits" in output_names
        elif task_name in _VL_TWO_MODEL_TASKS:
            assert set(pkg) == {"decoder", "vision_encoder"}
            decoder = pkg["decoder"]
            assert "encoder_hidden_states" in {i.name for i in decoder.graph.inputs}
            assert "logits" in {o.name for o in decoder.graph.outputs}
            vision = pkg["vision_encoder"]
            pixel_values = next(i for i in vision.graph.inputs if i.name == "pixel_values")
            assert pixel_values.dtype == ir.DataType.FLOAT
        else:
            assert "decoder" in pkg, f"{model_type} should produce 'decoder'"
            assert "vision_encoder" in pkg, f"{model_type} should produce 'vision_encoder'"
            assert "embedding" in pkg, f"{model_type} should produce 'embedding'"

            decoder = pkg["decoder"]
            assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
            assert "logits" in {o.name for o in decoder.graph.outputs}

            vision = pkg["vision_encoder"]
            pixel_values = next(i for i in vision.graph.inputs if i.name == "pixel_values")
            assert pixel_values.dtype == ir.DataType.FLOAT

    def test_has_initializers(self, model_type: str, config_overrides: dict):
        """Verify all sub-models have non-empty initializers."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        for name, model in pkg.items():
            init_names = list(model.graph.initializers)
            assert len(init_names) > 0, f"{model_type}/{name} should have initializers"

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run ONNX CheckerPass on all sub-models."""
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
