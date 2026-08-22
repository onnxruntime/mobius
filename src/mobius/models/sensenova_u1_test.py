# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the SenseNova-U1.5 (``neo_chat``) NEO-unify model."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from mobius._configs import SenseNovaU1Config
from mobius._configs._sub_configs import VisionConfig
from mobius.models.sensenova_u1 import SenseNovaU1Model
from mobius.tasks import get_task

HIDDEN = 64
LAYERS = 2
Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
VOCAB = 256
PATCH = 4
VISION_HIDDEN = 32
FREQ_EMBED = 8
MERGE = 2
PIXELS_PER_TOKEN = PATCH * MERGE


def _tiny_config(dtype: ir.DataType = ir.DataType.FLOAT) -> SenseNovaU1Config:
    config = SenseNovaU1Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=Q_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rope_theta=5e6,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        hidden_act="silu",
        dtype=dtype,
        attn_qk_norm=True,
        rope_type="default",
        rope_theta_hw=1e4,
        max_position_embeddings_hw=512,
        patch_size=PATCH,
        downsample_ratio=0.5,
        frequency_embedding_size=FREQ_EMBED,
    )
    config.vision = VisionConfig(
        hidden_size=VISION_HIDDEN,
        patch_size=PATCH,
        in_channels=3,
        spatial_merge_size=MERGE,
        out_hidden_size=HIDDEN,
        rope_theta=1e4,
        num_position_embeddings=512,
        num_hidden_layers=0,
        num_attention_heads=0,
    )
    return config


def _build(dtype: ir.DataType = ir.DataType.FLOAT):
    config = _tiny_config(dtype)
    module = SenseNovaU1Model(config)
    package = get_task("sensenova-u1").build(module, config)
    return config, module, package


# ── L1: graph construction ──────────────────────────────────────────────


class TestGraphBuild:
    def test_package_has_five_components(self):
        _, _, package = _build()
        assert set(package.keys()) == {
            "decoder",
            "vision_encoder",
            "embedding",
            "image_gen_embedding",
            "image_gen_denoiser",
        }

    def test_decoder_io_contract(self):
        config, _, package = _build()
        graph = package["decoder"].graph
        names = [value.name for value in graph.inputs]
        assert names[:3] == ["inputs_embeds", "attention_mask", "position_ids"]
        # Three stacked rotary axes (temporal, height, width).
        assert graph.inputs[2].shape[0] == 3
        outputs = [value.name for value in graph.outputs]
        assert outputs[0] == "logits"
        assert len(outputs) == 1 + 2 * config.num_hidden_layers

    def test_generation_components_io(self):
        _, _, package = _build()
        gen_embed = package["image_gen_embedding"].graph
        assert [v.name for v in gen_embed.inputs] == [
            "latent",
            "timestep",
            "noise_scale",
        ]
        assert [v.name for v in gen_embed.outputs] == ["image_embeds"]

        denoiser = package["image_gen_denoiser"].graph
        assert [v.name for v in denoiser.inputs][:3] == [
            "image_embeds",
            "position_ids",
            "token_grid",
        ]
        outputs = [v.name for v in denoiser.outputs]
        assert outputs[0] == "predicted_image"
        # ORT's CUDA Attention kernel requires present_key/present_value
        # outputs whenever past_key/past_value are provided, even though the
        # sampler discards them (upstream denoises with update_cache=False).
        assert outputs[1:] == [
            name
            for layer in range(LAYERS)
            for name in (f"present.{layer}.key", f"present.{layer}.value")
        ]

    @pytest.mark.parametrize(
        "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
    )
    def test_builds_for_each_dtype(self, dtype: ir.DataType):
        _, _, package = _build(dtype)
        assert package["decoder"].graph.inputs[0].dtype == dtype

    def test_no_protobuf_apis_used(self):
        # ir.Model is the only representation the task returns.
        _, _, package = _build()
        assert all(isinstance(model, ir.Model) for model in package.values())


# ── Config extraction ───────────────────────────────────────────────────


class TestConfigExtraction:
    """``neo_chat`` is composite: text fields live under ``llm_config``."""

    @staticmethod
    def _hf_config():
        from mobius.integrations.transformers._config_resolver import (
            _dict_to_pretrained_config,
        )

        return _dict_to_pretrained_config(
            {
                "model_type": "neo_chat",
                "architectures": ["NEOChatModel"],
                "downsample_ratio": 0.5,
                "patch_size": 16,
                "use_pixel_head": True,
                "fm_head_dim": 1536,
                "fm_head_layers": 2,
                "add_noise_scale_embedding": True,
                "noise_scale_max_value": 16.0,
                "noise_scale_mode": "resolution",
                "t_eps": 0.05,
                "timestep_shift": 1.0,
                "tie_word_embeddings": False,
                "llm_config": {
                    "architectures": ["Qwen3ForCausalLM"],
                    "model_type": "qwen3",
                    "hidden_size": 4096,
                    "intermediate_size": 12288,
                    "num_hidden_layers": 42,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "vocab_size": 151936,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 5000000.0,
                    "rope_theta_hw": 10000.0,
                    "max_position_embeddings": 262144,
                    "max_position_embeddings_hw": 10000,
                    "hidden_act": "silu",
                    "attention_bias": False,
                },
                "vision_config": {
                    "model_type": "neo_vision",
                    "hidden_size": 1024,
                    "llm_hidden_size": 4096,
                    "downsample_ratio": 0.5,
                    "patch_size": 16,
                    "num_channels": 3,
                    "rope_theta_vision": 10000.0,
                    "max_position_embeddings_vision": 10000,
                },
            }
        )

    def test_lifts_text_fields_from_llm_config(self):
        hf_config = self._hf_config()
        config = SenseNovaU1Config.from_transformers(hf_config, parent_config=hf_config)
        assert config.num_hidden_layers == 42
        assert config.hidden_size == 4096
        assert (config.num_attention_heads, config.num_key_value_heads) == (32, 8)
        assert config.head_dim == 128
        assert config.vocab_size == 151936
        assert config.rope_theta == pytest.approx(5_000_000.0)

    def test_extracts_spatial_rope_and_flow_matching_fields(self):
        hf_config = self._hf_config()
        config = SenseNovaU1Config.from_transformers(hf_config, parent_config=hf_config)
        assert config.rope_theta_hw == pytest.approx(10_000.0)
        assert config.max_position_embeddings_hw == 10_000
        assert config.use_pixel_head is True
        assert config.noise_scale_max_value == pytest.approx(16.0)
        assert config.t_eps == pytest.approx(0.05)

    def test_derived_patch_geometry(self):
        hf_config = self._hf_config()
        config = SenseNovaU1Config.from_transformers(hf_config, parent_config=hf_config)
        assert config.merge_size == 2
        # One LLM token covers a 32x32 pixel tile.
        assert config.pixels_per_token == 32

    def test_vision_hook_populates_llm_projection_width(self):
        hf_config = self._hf_config()
        config = SenseNovaU1Config.from_transformers(hf_config, parent_config=hf_config)
        assert config.vision.hidden_size == 1024
        assert config.vision.out_hidden_size == 4096
        assert config.vision.spatial_merge_size == 2
        # The NEO vision tower has no transformer blocks at all.
        assert config.vision.num_hidden_layers == 0


# ── Weight-name alignment ───────────────────────────────────────────────


def _hf_state_dict_names(config: SenseNovaU1Config) -> list[str]:
    """Reproduce the released checkpoint's key set."""
    names = [
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.patch_embedding.bias",
        "vision_model.embeddings.dense_embedding.weight",
        "vision_model.embeddings.dense_embedding.bias",
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias",
        "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight",
        "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.bias",
        "fm_modules.fm_head.conv1.weight",
        "fm_modules.fm_head.conv1.bias",
        "fm_modules.fm_head.conv2.weight",
        "fm_modules.fm_head.conv2.bias",
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "language_model.model.norm.weight",
        "language_model.model.norm_mot_gen.weight",
    ]
    for embedder in ("timestep_embedder", "noise_scale_embedder"):
        for index in (0, 2):
            names.append(f"fm_modules.{embedder}.mlp.{index}.weight")
            names.append(f"fm_modules.{embedder}.mlp.{index}.bias")
    for layer in range(config.num_hidden_layers):
        prefix = f"language_model.model.layers.{layer}"
        for branch in ("", "_mot_gen"):
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                names.append(f"{prefix}.self_attn.{proj}{branch}.weight")
            for norm in ("q_norm", "q_norm_hw", "k_norm", "k_norm_hw"):
                names.append(f"{prefix}.self_attn.{norm}{branch}.weight")
            for proj in ("gate_proj", "up_proj", "down_proj"):
                names.append(f"{prefix}.mlp{branch}.{proj}.weight")
            names.append(f"{prefix}.input_layernorm{branch}.weight")
            names.append(f"{prefix}.post_attention_layernorm{branch}.weight")
    return names


class TestWeightAlignment:
    def test_every_hf_key_maps_to_an_initializer(self):
        config, module, package = _build()
        initializers = {
            name for model in package.values() for name in model.graph.initializers
        }
        unmapped = []
        for key in _hf_state_dict_names(config):
            targets = module._targets_for(key)
            assert targets, f"no ONNX target for {key}"
            for target in targets:
                if target not in initializers:
                    unmapped.append((key, target))
        assert not unmapped, f"HF keys mapped to unknown initializers: {unmapped[:5]}"

    def test_mot_gen_weights_route_to_the_generation_graph(self):
        _, module, _ = _build()
        und = module._targets_for("language_model.model.layers.0.mlp.gate_proj.weight")
        gen = module._targets_for("language_model.model.layers.0.mlp_mot_gen.gate_proj.weight")
        assert und == ["decoder.model.layers.0.mlp.gate_proj.weight"]
        assert gen == ["image_gen_denoiser.model.layers.0.mlp_mot_gen.gate_proj.weight"]

    def test_final_norms_split_across_branches(self):
        _, module, _ = _build()
        assert module._targets_for("language_model.model.norm.weight") == [
            "decoder.model.norm.weight"
        ]
        assert module._targets_for("language_model.model.norm_mot_gen.weight") == [
            "image_gen_denoiser.model.norm_mot_gen.weight"
        ]

    def test_fm_head_and_gen_tower_split(self):
        _, module, _ = _build()
        assert module._targets_for("fm_modules.fm_head.conv1.weight") == [
            "image_gen_denoiser.fm_head.conv1.weight"
        ]
        assert module._targets_for(
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight"
        ) == ["image_gen_embedding.vision_model_mot_gen.embeddings.patch_embedding.weight"]

    def test_only_derived_tables_lack_an_hf_source(self):
        config, module, package = _build()
        mapped = {
            target
            for key in _hf_state_dict_names(config)
            for target in module._targets_for(key)
        }
        initializers = {
            name for model in package.values() for name in model.graph.initializers
        }
        extra = sorted(initializers - mapped)
        # Only precomputed constant tables may be missing from the checkpoint.
        assert all(name.endswith((".cos_cache", ".sin_cache", ".freqs")) for name in extra), (
            extra
        )


# ── L4: executable graphs on CPU ────────────────────────────────────────


def _fill(model: ir.Model) -> None:
    from mobius.rewrite_rules._testing_utils import fill_random_weights

    fill_random_weights(model)


class TestExecution:
    def _session(self, model: ir.Model):
        from mobius._testing.ort_inference import OnnxModelSession

        _fill(model)
        return OnnxModelSession(model)

    def test_vision_tower_handles_non_square_grids(self):
        _, _, package = _build()
        session = self._session(package["vision_encoder"])
        try:
            for grid_h, grid_w in [(4, 4), (2, 6)]:
                height = grid_h * PATCH
                width = grid_w * PATCH
                pixel_values = np.random.randn(1, 3, height, width).astype(np.float32)
                out = session.run({"pixel_values": pixel_values})["image_features"]
                # 2x2 patch merge halves each grid axis.
                assert out.shape == (1, (grid_h // MERGE) * (grid_w // MERGE), HIDDEN)
        finally:
            session.close()

    def test_decoder_runs_with_batch_two_and_ragged_padding(self):
        config, _, package = _build()
        session = self._session(package["decoder"])
        try:
            batch, seq = 2, 6
            rng = np.random.default_rng(0)
            attention_mask = np.ones((batch, seq), dtype=np.int64)
            attention_mask[1, :2] = 0
            positions = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
            position_ids = np.stack(
                [positions, np.zeros_like(positions), np.zeros_like(positions)]
            )
            feeds: dict[str, np.ndarray] = {
                "inputs_embeds": rng.standard_normal((batch, seq, HIDDEN)).astype(np.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            for layer in range(config.num_hidden_layers):
                feeds[f"past_key_values.{layer}.key"] = np.zeros(
                    (batch, KV_HEADS, 0, HEAD_DIM), dtype=np.float32
                )
                feeds[f"past_key_values.{layer}.value"] = np.zeros(
                    (batch, KV_HEADS, 0, HEAD_DIM), dtype=np.float32
                )
            out = session.run(feeds)
            assert out["logits"].shape == (batch, seq, VOCAB)
            assert not np.allclose(out["logits"][0], out["logits"][1])
        finally:
            session.close()

    def test_image_tokens_attend_bidirectionally_within_a_block(self):
        """Tokens sharing a temporal index must see each other.

        A plain causal mask would make the *first* image token's output
        independent of later image tokens; block-causal masking must not.
        """
        config, _, package = _build()
        session = self._session(package["decoder"])
        try:
            text_len, n_image = 3, 4
            seq = text_len + n_image
            rng = np.random.default_rng(1)
            embeds = rng.standard_normal((1, seq, HIDDEN)).astype(np.float32)
            temporal = np.concatenate(
                [np.arange(text_len), np.full(n_image, text_len)]
            ).astype(np.int64)[None]
            index = np.arange(n_image)
            height = np.concatenate([np.zeros(text_len), index // 2]).astype(np.int64)
            width = np.concatenate([np.zeros(text_len), index % 2]).astype(np.int64)
            position_ids = np.stack([temporal, height[None], width[None]])
            feeds: dict[str, np.ndarray] = {
                "inputs_embeds": embeds,
                "attention_mask": np.ones((1, seq), dtype=np.int64),
                "position_ids": position_ids,
            }
            for layer in range(config.num_hidden_layers):
                feeds[f"past_key_values.{layer}.key"] = np.zeros(
                    (1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32
                )
                feeds[f"past_key_values.{layer}.value"] = np.zeros(
                    (1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32
                )
            baseline = session.run(feeds)["logits"][0, text_len].copy()

            perturbed = embeds.copy()
            perturbed[0, -1] += 5.0  # change the LAST image token
            feeds["inputs_embeds"] = perturbed
            changed = session.run(feeds)["logits"][0, text_len]
            assert not np.allclose(baseline, changed, atol=1e-5)

            # A text token before the image block must stay unaffected.
            feeds["inputs_embeds"] = embeds
            text_baseline = session.run(feeds)["logits"][0, text_len - 1].copy()
            feeds["inputs_embeds"] = perturbed
            text_changed = session.run(feeds)["logits"][0, text_len - 1]
            np.testing.assert_allclose(text_baseline, text_changed, atol=1e-5)
        finally:
            session.close()

    def test_image_generation_stages_run_end_to_end(self):
        config, _, package = _build()
        grid_h, grid_w = 4, 4
        height = grid_h * PATCH
        width = grid_w * PATCH
        token_h = grid_h // MERGE
        token_w = grid_w // MERGE

        embed_session = self._session(package["image_gen_embedding"])
        try:
            latent = np.random.randn(1, 3, height, width).astype(np.float32)
            image_embeds = embed_session.run(
                {
                    "latent": latent,
                    "timestep": np.array([0.3], dtype=np.float32),
                    "noise_scale": np.array([0.0625], dtype=np.float32),
                }
            )["image_embeds"]
        finally:
            embed_session.close()
        assert image_embeds.shape == (1, token_h * token_w, HIDDEN)

        denoise_session = self._session(package["image_gen_denoiser"])
        try:
            n_tokens = token_h * token_w
            prefix = 5
            index = np.arange(n_tokens)
            position_ids = np.stack(
                [
                    np.full((1, n_tokens), prefix, dtype=np.int64),
                    (index // token_w)[None].astype(np.int64),
                    (index % token_w)[None].astype(np.int64),
                ]
            )
            feeds: dict[str, np.ndarray] = {
                "image_embeds": image_embeds,
                "position_ids": position_ids,
                "token_grid": np.array([token_h, token_w], dtype=np.int64),
            }
            for layer in range(config.num_hidden_layers):
                feeds[f"past_key_values.{layer}.key"] = np.random.randn(
                    1, KV_HEADS, prefix, HEAD_DIM
                ).astype(np.float32)
                feeds[f"past_key_values.{layer}.value"] = np.random.randn(
                    1, KV_HEADS, prefix, HEAD_DIM
                ).astype(np.float32)
            predicted = denoise_session.run(feeds)["predicted_image"]
        finally:
            denoise_session.close()
        # The pixel head reconstructs the full-resolution RGB image.
        assert predicted.shape == (1, 3, height, width)


# ── Metadata contract status ────────────────────────────────────────────


class TestInferenceMetadataStatus:
    """Pin the canonical shared-state workflow contract."""

    def test_package_is_recognised_as_a_native_vlm_shape(self):
        from mobius.integrations.onnx_genai.inference_metadata import (
            is_native_vlm_package,
        )

        _, _, package = _build()
        assert is_native_vlm_package(package)

    def test_emits_complete_hashless_shared_state_workflow(self):
        from mobius.integrations.onnx_genai.shared_state_flow_metadata import (
            build_shared_state_pixel_flow_workflow_metadata,
        )

        config, _, package = _build()
        metadata = build_shared_state_pixel_flow_workflow_metadata(
            package, config, num_inference_steps=2
        )
        workflow = metadata["pipeline"]["workflow"]
        assert "linear_effects" in workflow["manifest"]["capabilities"]
        assert set(workflow["components"]) >= {
            "embedding",
            "vision_encoder",
            "decoder",
            "image_gen_embedding",
            "image_gen_denoiser",
            "x0_velocity",
            "guidance_combine",
            "solver_step",
            "image_dimensions",
            "image_noise_geometry",
            "image_noise",
            "latent_scale",
            "image_output_clamp",
        }
        group_ports = workflow["serving"]["state_service"]["groups"]["conditional_prefix"][
            "ports"
        ]
        assert "image_gen_denoiser" not in group_ports
        assert workflow["inputs"]["request.seed"]["role"]["role"] == "seed"
        assert workflow["inputs"]["request.width"]["role"]["role"] == "width"
        assert workflow["inputs"]["request.height"]["role"]["role"] == "height"
        assert workflow["inputs"]["request.negative_prompt_tokens"]["contract"]["shape"] == [
            "batch",
            "negative_sequence_len",
        ]
        assert workflow["inputs"]["request.latent"]["required"] is False
        assert workflow["inputs"]["request.latent"]["present_as"] == "request.latent_present"
        latent_branch = workflow["steps"][2]
        assert latent_branch["predicate"] == "request.latent_present"
        generated_latent_scale = latent_branch["cases"]["false"]["steps"][1]
        assert generated_latent_scale["inputs"]["scale"] == "noise.noise_scale"
        image_dimensions = next(
            step for step in workflow["steps"] if step.get("component") == "image_dimensions"
        )
        assert image_dimensions["inputs"]["tensor"] == "latent.initial"
        resolved_geometry = [
            step
            for step in workflow["steps"]
            if step.get("component") == "image_noise_geometry"
        ][1]
        assert resolved_geometry["inputs"] == {
            "height": "image.actual_height",
            "width": "image.actual_width",
        }
        assert resolved_geometry["outputs"]["noise_scale"] == "image.noise_scale"
        prefix_initializers = [
            step for step in workflow["steps"] if step.get("component") == "prefix_initializer"
        ]
        assert len(prefix_initializers) == 2
        generation_steps = workflow["steps"][-1]["cases"]["false"]["steps"]
        loop_body = generation_steps[0]["steps"]
        grid_invocations = [
            step for step in loop_body if step.get("component") == "image_grid_positions"
        ]
        assert len(grid_invocations) == 2
        assert {
            step["inputs"]["prompt_tokens"]: step["outputs"]["position_ids"]
            for step in grid_invocations
        } == {
            "request.prompt_tokens": "flow.conditional.position_ids",
            "request.negative_prompt_tokens": "flow.unconditional.position_ids",
        }
        denoiser_invocations = [
            step for step in loop_body if step.get("component") == "image_gen_denoiser"
        ]
        denoisers_by_output = {
            step["outputs"]["predicted_image"]: step for step in denoiser_invocations
        }
        assert (
            denoisers_by_output["flow.conditional.x0"]["inputs"]["position_ids"]
            == "flow.conditional.position_ids"
        )
        assert (
            denoisers_by_output["flow.unconditional.x0"]["inputs"]["position_ids"]
            == "flow.unconditional.position_ids"
        )
        generation_embedding = next(
            step for step in loop_body if step.get("component") == "image_gen_embedding"
        )
        assert generation_embedding["inputs"]["noise_scale"] == "image.noise_scale"
        np.testing.assert_array_equal(
            package.policy_components["flow_schedule"]
            .model.graph.outputs[0]
            .const_value.numpy(),
            [0.0, 0.5, 1.0],
        )
        assert workflow["outputs"]["image"]["value_range"] == "negative_one_to_one"
        transforms = metadata["preprocessing"]["image"]["transforms"]
        assert [transform["op"] for transform in transforms] == [
            "decode_rgb",
            "resize",
            "rescale",
            "normalize",
        ]
        assert transforms[1]["size_multiple"] == 32
        serialized = str(metadata)
        for forbidden in (
            "sha256",
            "config_sha256",
            "base_model_fingerprint",
            "ir_version",
            "onnx_opsets",
        ):
            assert forbidden not in serialized

    def test_auto_export_writes_exact_synthesized_processor_asset(self, tmp_path):
        import json

        from mobius.integrations.onnx_genai import write_onnx_genai_config

        config, _, package = _build()
        artifacts = write_onnx_genai_config(
            package,
            str(tmp_path),
            config=config,
            num_inference_steps=2,
        )
        processor = json.loads((tmp_path / "preprocessor_config.json").read_text())
        assert processor["size_multiple"] == 32
        assert processor["min_pixels"] == 512 * 512
        assert processor["max_pixels"] == 2048 * 2048
        assert processor["rescale_factor"] == pytest.approx(1 / 255)
        assert processor["image_mean"] == [0.485, 0.456, 0.406]
        assert processor["image_std"] == [0.229, 0.224, 0.225]
        assert artifacts["inference_metadata"].endswith("inference_metadata.yaml")

    def test_tied_embeddings_populate_the_lm_head(self):
        """Tied variants ship no ``lm_head`` tensor of their own."""
        import torch

        config = _tiny_config()
        config.tie_word_embeddings = True
        module = SenseNovaU1Model(config)
        weights = module.preprocess_weights(
            {"language_model.model.embed_tokens.weight": torch.zeros(VOCAB, HIDDEN)}
        )
        assert "decoder.lm_head.weight" in weights
        assert weights["decoder.lm_head.weight"] is weights["embedding.embed_tokens.weight"]

    def test_untied_checkpoint_keeps_its_own_lm_head(self):
        import torch

        _, module, _ = _build()
        head = torch.ones(VOCAB, HIDDEN)
        weights = module.preprocess_weights(
            {
                "language_model.model.embed_tokens.weight": torch.zeros(VOCAB, HIDDEN),
                "language_model.lm_head.weight": head,
            }
        )
        assert weights["decoder.lm_head.weight"] is head
