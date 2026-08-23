# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Diffusers integration builder."""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import mock_open, patch

import onnx_ir as ir
import pytest

from mobius._model_package import ModelPackage
from mobius.integrations.diffusers._builder import (
    _DIFFUSERS_CLASS_MAP,
    _download_diffusers_component_weights,
    _init_diffusers_class_map,
    _load_diffusers_component_config,
    _load_diffusers_pipeline_index,
    _load_optional_diffusers_json,
    _resolve_diffusers_component_source,
    _resolve_hierarchical_workflow_config,
    build_diffusers_pipeline,
)
from mobius.integrations.diffusers._configs import (
    MINIMAX_MUSIC3_AUDIO_CODE_OFFSET,
    MINIMAX_MUSIC3_AUDIO_END_TOKEN_ID,
    MINIMAX_MUSIC3_GLOBAL_CONTEXT,
    MINIMAX_MUSIC3_SEMANTIC_VOCAB_SIZE,
    MINIMAX_MUSIC3_UNCONDITIONAL_TOKEN_ID,
    MiniMaxMusic3WorkflowConfig,
)
from mobius.integrations.onnx_genai import HierarchicalAudioWorkflowConfig

# ── Helpers ──────────────────────────────────────────────────────────────


def _hierarchical_workflow_config(**overrides) -> HierarchicalAudioWorkflowConfig:
    """Construct a valid hierarchical-audio workflow config for tests.

    The values are illustrative structural/semantic facts, not a checked-in
    model contract; callers override individual fields to prove the metadata
    producer follows its typed input rather than any default.
    """
    fields = {
        "components": {
            "global_decoder": "decoder",
            "global_embedding": "embedding",
            "semantic_embedding": "semantic",
            "local_decoder": "local",
            "local_projection": "projection",
            "local_embedding": "local_embedding",
            "local_feedback_embedding": "feedback",
            "local_heads": "heads",
            "condition_encoder": "condition",
            "flow_transformer": "flow",
            "vocoder": "vocoder",
        },
        "semantic_vocabulary_start": 901,
        "semantic_vocabulary_size": 16,
        "stop_token_id": 4,
        "unconditional_token_id": 5,
        "semantic_guidance_scale": 1.5,
        "local_guidance_scale": 1.5,
        "flow_guidance_scale": 1.7,
        "sampling_top_k": 50,
        "chunk_frames": 200,
        "chunk_hop": 100,
        "flow_steps": 30,
        "carry_length": 172,
        "crop_left_latents": 86,
        "crop_right_latents": 258,
        "max_prompt_tokens": 5000,
        "max_audio_frames": 9000,
        "global_context": 128,
        "target_sample_rate": 32000,
        "unconditional_replace_from": 1,
        "unconditional_preserve_trailing": 2,
        "prompt_segments": [{"literal": "<prompt>"}],
    }
    fields.update(overrides)
    return HierarchicalAudioWorkflowConfig(**fields)


@pytest.mark.parametrize(
    ("semantic_start", "flow_steps", "target_sample_rate"),
    [(901, 30, 32000), (2048, 12, 44100)],
)
def test_hierarchical_workflow_config_records_explicit_inputs(
    semantic_start, flow_steps, target_sample_rate
):
    """Two explicit configs prove emitted policy/template fields follow input."""
    config = _hierarchical_workflow_config(
        semantic_vocabulary_start=semantic_start,
        flow_steps=flow_steps,
        target_sample_rate=target_sample_rate,
    )
    assert config.semantic_vocabulary_start == semantic_start
    assert config.flow_steps == flow_steps
    assert config.target_sample_rate == target_sample_rate
    # asdict is the boundary the producer consumes; every field must survive it.
    as_dict = dataclasses.asdict(config)
    assert as_dict["semantic_vocabulary_start"] == semantic_start
    assert as_dict["flow_steps"] == flow_steps
    assert as_dict["target_sample_rate"] == target_sample_rate
    assert as_dict["prompt_segments"] == [{"literal": "<prompt>"}]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"components": {"global_decoder": "decoder"}}, "missing component roles"),
        ({"flow_steps": 0}, "flow_steps"),
        ({"chunk_hop": 500}, "chunk_hop cannot exceed chunk_frames"),
        ({"semantic_guidance_scale": 0.0}, "semantic_guidance_scale"),
        ({"prompt_segments": []}, "prompt_segments"),
    ],
)
def test_hierarchical_workflow_config_fails_closed_on_invalid_input(overrides, match):
    """The typed config refuses to describe an under-specified workflow."""
    with pytest.raises((ValueError, TypeError), match=match):
        _hierarchical_workflow_config(**overrides)


def test_minimax_workflow_adapter_applies_owned_defaults():
    """The mobius adapter fills the typed config from Music 3 defaults it owns."""
    roles = _hierarchical_workflow_config().components
    config = MiniMaxMusic3WorkflowConfig.from_diffusers(
        components=dict(roles),
        component_configs={"decoder": {"max_position_embeddings": 4096}},
    )
    assert isinstance(config, HierarchicalAudioWorkflowConfig)
    # Semantic constants come from the mobius-owned defaults, not the writer.
    assert config.semantic_vocabulary_start == MINIMAX_MUSIC3_AUDIO_CODE_OFFSET
    assert config.semantic_vocabulary_size == MINIMAX_MUSIC3_SEMANTIC_VOCAB_SIZE
    assert config.stop_token_id == MINIMAX_MUSIC3_AUDIO_END_TOKEN_ID
    assert config.unconditional_token_id == MINIMAX_MUSIC3_UNCONDITIONAL_TOKEN_ID
    # The context window is derived from the source language config.
    assert config.global_context == 4096
    assert config.components == dict(roles)
    assert config.prompt_segments  # non-empty prompt template


@pytest.mark.parametrize("max_positions", [2048, 8192])
def test_minimax_workflow_adapter_generalizes_context(max_positions):
    """A divergent language config flows through to the derived context window."""
    roles = _hierarchical_workflow_config().components
    config = MiniMaxMusic3WorkflowConfig.from_diffusers(
        components=dict(roles),
        component_configs={"decoder": {"max_position_embeddings": max_positions}},
    )
    assert config.global_context == max_positions


def test_minimax_workflow_adapter_context_fallback():
    """When the source omits the context window the adapter uses its named default."""
    roles = _hierarchical_workflow_config().components
    config = MiniMaxMusic3WorkflowConfig.from_diffusers(
        components=dict(roles),
        component_configs={},
    )
    assert config.global_context == MINIMAX_MUSIC3_GLOBAL_CONTEXT


def test_resolve_workflow_config_explicit_argument_overrides_defaults():
    """An explicit caller config wins over the mobius adapter defaults."""
    roles = _hierarchical_workflow_config().components
    override = _hierarchical_workflow_config(
        semantic_vocabulary_start=2048,
        stop_token_id=7,
        flow_steps=12,
        target_sample_rate=44100,
    )
    resolved = _resolve_hierarchical_workflow_config(
        "MiniMaxMusic3ModularPipeline",
        dict(roles),
        {"decoder": {"max_position_embeddings": 4096}},
        override,
    )
    assert resolved is not None
    # The caller's divergent values survive; the adapter defaults are not used.
    assert resolved.semantic_vocabulary_start == 2048
    assert resolved.stop_token_id == 7
    assert resolved.flow_steps == 12
    assert resolved.target_sample_rate == 44100
    # The built graphs' structural role map is authoritative.
    assert resolved.components == dict(roles)


def test_resolve_workflow_config_uses_adapter_defaults():
    """Without an explicit config, mobius supplies the pipeline's owned defaults."""
    roles = _hierarchical_workflow_config().components
    resolved = _resolve_hierarchical_workflow_config(
        "MiniMaxMusic3ModularPipeline",
        dict(roles),
        {"decoder": {"max_position_embeddings": 4096}},
        None,
    )
    assert resolved is not None
    assert resolved.semantic_vocabulary_start == MINIMAX_MUSIC3_AUDIO_CODE_OFFSET
    assert resolved.global_context == 4096
    assert resolved.components == dict(roles)


def test_resolve_workflow_config_fails_closed_for_unknown_pipeline():
    """A hierarchical topology with no registered adapter fails closed (None)."""
    roles = _hierarchical_workflow_config().components
    resolved = _resolve_hierarchical_workflow_config(
        "SomeUnknownAudioPipeline",
        dict(roles),
        {"decoder": {"max_position_embeddings": 4096}},
        None,
    )
    assert resolved is None


def _fake_pipeline_index(
    components: dict[str, list[str]] | None = None,
) -> dict:
    """Build a fake model_index.json dict.

    Args:
        components: Mapping of component name to [library, class_name].
            Defaults to a single FluxTransformer2DModel component.
    """
    index: dict = {"_class_name": "FluxPipeline"}
    if components is None:
        components = {
            "transformer": ["diffusers", "FluxTransformer2DModel"],
        }
    index.update(components)
    return index


# ── _init_diffusers_class_map ────────────────────────────────────────────


class TestInitDiffusersClassMap:
    """Tests for lazy initialization of the diffusers class map."""

    def test_populates_expected_classes(self):
        _init_diffusers_class_map()
        expected_keys = {
            "DiTTransformer2DModel",
            "HunyuanDiT2DModel",
            "PixArtTransformer2DModel",
            "FluxTransformer2DModel",
            "SD3Transformer2DModel",
            "QwenImageTransformer2DModel",
            "Qwen2_5_VLForConditionalGeneration",
            "UNet2DConditionModel",
            "CLIPTextModel",
            "T5EncoderModel",
            "AutoencoderKL",
            "AutoencoderKLQwenImage",
            "AutoencoderKLCogVideoX",
            "CogVideoXTransformer3DModel",
            "MiniMaxMusic3ConditionEncoder",
            "MiniMaxMusic3RVQDepthDecoder",
            "MiniMaxMusic3Transformer1DModel",
            "MiniMaxMusic3Vocoder",
            "Qwen3ForCausalLM",
        }
        assert expected_keys == set(_DIFFUSERS_CLASS_MAP.keys())

    def test_each_entry_is_three_tuple(self):
        _init_diffusers_class_map()
        for class_name, entry in _DIFFUSERS_CLASS_MAP.items():
            assert len(entry) == 3, (
                f"Entry for {class_name} should be (module_class, config_class, task_name)"
            )
            module_class, config_class, task_name = entry
            assert callable(module_class)
            assert callable(config_class)
            assert isinstance(task_name, str)

    def test_task_names_are_valid(self):
        _init_diffusers_class_map()
        # Classic Stable Diffusion adds the CLIP text encoder ("feature-extraction")
        # and the UNet denoiser ("denoising").
        valid_tasks = {
            "denoising",
            "vae",
            "qwen-image-vae",
            "qwen-image-denoising",
            "qwen-image-text-encoding",
            "video-denoising",
            "video-vae",
            "feature-extraction",
            "t5-text-encoding",
            "minimax-music3-condition",
            "minimax-music3-denoising",
            "minimax-music3-language",
            "minimax-music3-rvq",
            "minimax-music3-vocoder",
        }
        for class_name, (_, _, task_name) in _DIFFUSERS_CLASS_MAP.items():
            assert task_name in valid_tasks, f"Unknown task '{task_name}' for {class_name}"

    def test_idempotent(self):
        """Calling _init_diffusers_class_map twice does not duplicate entries."""
        _init_diffusers_class_map()
        count_before = len(_DIFFUSERS_CLASS_MAP)
        _init_diffusers_class_map()
        assert len(_DIFFUSERS_CLASS_MAP) == count_before


# ── build_diffusers_pipeline error handling ──────────────────────────────


class TestBuildDiffusersPipelineErrors:
    """Tests for error paths in build_diffusers_pipeline."""

    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
        return_value=None,
    )
    def test_raises_when_no_model_index(self, _mock_load):
        """ValueError when model_index.json is not found."""
        with pytest.raises(ValueError, match="does not appear to be a diffusers pipeline"):
            build_diffusers_pipeline("fake/no-index-model", load_weights=False)

    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
        return_value={"_class_name": "SomePipeline"},
    )
    def test_raises_when_no_supported_components(self, _mock_load):
        """ValueError when pipeline has no registered neural network components."""
        with pytest.raises(ValueError, match="No supported neural network components"):
            build_diffusers_pipeline("fake/empty-pipeline", load_weights=False)

    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
        return_value={
            "_class_name": "SomePipeline",
            "scheduler": ["diffusers", "EulerDiscreteScheduler"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
        },
    )
    def test_raises_when_only_non_nn_components(self, _mock_load):
        """ValueError when pipeline only has non-NN components (scheduler, tokenizer)."""
        with pytest.raises(ValueError, match="No supported neural network components"):
            build_diffusers_pipeline("fake/scheduler-only", load_weights=False)


# ── Hub revision propagation ─────────────────────────────────────────────


class TestDiffusersHubRevision:
    @patch("huggingface_hub.hf_hub_download", return_value="model_index.json")
    def test_pipeline_index_download_uses_revision(self, mock_download):
        with patch("builtins.open", mock_open(read_data='{"_class_name": "FakePipeline"}')):
            result = _load_diffusers_pipeline_index(
                "fake/model",
                revision="pinned-revision",
            )

        assert result == {"_class_name": "FakePipeline"}
        mock_download.assert_called_once_with(
            repo_id="fake/model",
            filename="model_index.json",
            revision="pinned-revision",
        )


class TestLocalDiffusersPackage:
    def test_loads_modular_index_without_hub_access(self, tmp_path):
        index = {"_class_name": "LocalModularPipeline"}
        (tmp_path / "modular_model_index.json").write_text(json.dumps(index))

        with patch("huggingface_hub.hf_hub_download") as download:
            assert _load_diffusers_pipeline_index(str(tmp_path)) == index

        download.assert_not_called()

    def test_resolves_root_component_to_local_subfolder(self, tmp_path):
        component = tmp_path / "transformer"
        component.mkdir()
        (component / "config.json").write_text("{}")
        info = [
            "diffusers",
            "Transformer",
            {
                "pretrained_model_name_or_path": "upstream/model",
                "revision": "upstream-revision",
                "subfolder": "transformer",
            },
        ]

        assert _resolve_diffusers_component_source(
            str(tmp_path), "ignored-local-revision", "transformer", info
        ) == (str(tmp_path), None, "transformer")

    def test_loads_local_component_and_optional_configs(self, tmp_path):
        component = tmp_path / "transformer"
        scheduler = tmp_path / "scheduler"
        component.mkdir()
        scheduler.mkdir()
        (component / "config.json").write_text('{"width": 64}')
        (scheduler / "scheduler_config.json").write_text('{"steps": 30}')

        with patch("huggingface_hub.hf_hub_download") as download:
            assert _load_diffusers_component_config(str(tmp_path), "transformer") == {
                "width": 64
            }
            assert _load_optional_diffusers_json(
                str(tmp_path), "scheduler/scheduler_config.json"
            ) == {"steps": 30}
            assert _load_optional_diffusers_json(str(tmp_path), "missing.json") == {}

        download.assert_not_called()

    @patch("huggingface_hub.hf_hub_download", return_value="config.json")
    def test_component_config_download_uses_revision(self, mock_download):
        with patch("builtins.open", mock_open(read_data='{"in_channels": 3}')):
            result = _load_diffusers_component_config(
                "fake/model",
                "vae",
                revision="pinned-revision",
            )

        assert result == {"in_channels": 3}
        mock_download.assert_called_once_with(
            repo_id="fake/model",
            filename="vae/config.json",
            revision="pinned-revision",
        )

    @patch("huggingface_hub.hf_hub_download", return_value="config.json")
    def test_component_config_uses_resolved_external_subfolder(self, mock_download):
        with patch("builtins.open", mock_open(read_data='{"in_channels": 3}')):
            _load_diffusers_component_config(
                "external/component-repo",
                "transformer",
                revision="component-revision",
                subfolder="nested/transformer",
            )
        mock_download.assert_called_once_with(
            repo_id="external/component-repo",
            filename="nested/transformer/config.json",
            revision="component-revision",
        )

    @patch("huggingface_hub.hf_hub_download", return_value="scheduler_config.json")
    def test_optional_metadata_download_uses_revision(self, mock_download):
        with patch("builtins.open", mock_open(read_data='{"beta_start": 0.001}')):
            result = _load_optional_diffusers_json(
                "fake/model",
                "scheduler/scheduler_config.json",
                revision="pinned-revision",
            )

        assert result == {"beta_start": 0.001}
        mock_download.assert_called_once_with(
            repo_id="fake/model",
            filename="scheduler/scheduler_config.json",
            revision="pinned-revision",
        )

    @patch("mobius.integrations.diffusers._builder._parallel_download", return_value=[])
    @patch("huggingface_hub.hf_hub_download", return_value="weights.index.json")
    def test_component_weight_downloads_use_revision(
        self,
        mock_download,
        mock_parallel_download,
    ):
        index = '{"weight_map": {"weight": "model-00001-of-00001.safetensors"}}'
        with patch("builtins.open", mock_open(read_data=index)):
            result = _download_diffusers_component_weights(
                "fake/model",
                "vae",
                revision="pinned-revision",
            )

        assert result == {}
        mock_download.assert_called_once_with(
            repo_id="fake/model",
            filename="vae/diffusion_pytorch_model.safetensors.index.json",
            revision="pinned-revision",
        )
        mock_parallel_download.assert_called_once_with(
            "fake/model",
            ["vae/model-00001-of-00001.safetensors"],
            revision="pinned-revision",
            desc="vae weights",
        )

    @patch("mobius.integrations.diffusers._builder._parallel_download", return_value=[])
    @patch("huggingface_hub.hf_hub_download", return_value="weights.index.json")
    def test_component_weights_use_resolved_external_subfolder(
        self,
        mock_download,
        mock_parallel_download,
    ):
        index = '{"weight_map": {"weight": "model-00001-of-00001.safetensors"}}'
        with patch("builtins.open", mock_open(read_data=index)):
            _download_diffusers_component_weights(
                "external/component-repo",
                "transformer",
                revision="component-revision",
                subfolder="nested/transformer",
            )
        mock_download.assert_called_once_with(
            repo_id="external/component-repo",
            filename=("nested/transformer/diffusion_pytorch_model.safetensors.index.json"),
            revision="component-revision",
        )
        mock_parallel_download.assert_called_once_with(
            "external/component-repo",
            ["nested/transformer/model-00001-of-00001.safetensors"],
            revision="component-revision",
            desc="transformer weights",
        )

    def test_component_source_revision_precedence(self):
        root_entry = [
            "diffusers",
            "MiniMaxMusic3ConditionEncoder",
            {
                "pretrained_model_name_or_path": "root/music",
                "revision": "metadata-revision",
                "subfolder": "condition_encoder",
            },
        ]
        assert _resolve_diffusers_component_source(
            "root/music", "caller-revision", "condition_encoder", root_entry
        ) == ("root/music", "caller-revision", "condition_encoder")
        assert _resolve_diffusers_component_source(
            "root/music", None, "condition_encoder", root_entry
        ) == ("root/music", "metadata-revision", "condition_encoder")

        external_entry = [
            "diffusers",
            "MiniMaxMusic3ConditionEncoder",
            {
                "pretrained_model_name_or_path": "external/components",
                "revision": "external-revision",
                "subfolder": None,
            },
        ]
        assert _resolve_diffusers_component_source(
            "root/music", "caller-revision", "condition_encoder", external_entry
        ) == ("external/components", "external-revision", "")

    @patch("huggingface_hub.hf_hub_download")
    def test_falls_back_to_modular_model_index(self, mock_download):
        from huggingface_hub.utils import EntryNotFoundError

        mock_download.side_effect = [
            EntryNotFoundError("missing"),
            "modular_model_index.json",
        ]
        with patch(
            "builtins.open",
            mock_open(read_data='{"_class_name": "MiniMaxMusic3ModularPipeline"}'),
        ):
            result = _load_diffusers_pipeline_index("fake/model", revision="pinned-revision")
        assert result["_class_name"] == "MiniMaxMusic3ModularPipeline"


# ── build_diffusers_pipeline component filtering ─────────────────────────


class TestBuildDiffusersPipelineFiltering:
    """Tests for how build_diffusers_pipeline filters pipeline components."""

    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_skips_underscore_prefixed_keys(self, mock_load_index, mock_load_config):
        """Keys starting with '_' (like _class_name) are skipped."""
        mock_load_index.return_value = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
        }
        # Should raise because no valid components remain
        with pytest.raises(ValueError, match="No supported neural network"):
            build_diffusers_pipeline("fake/model", load_weights=False)
        # _load_diffusers_component_config should never be called
        mock_load_config.assert_not_called()

    @patch(
        "mobius.integrations.diffusers._builder._load_optional_diffusers_json",
        return_value={},
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
        return_value={
            "condition_hidden_dim": 16,
            "num_condition_layers": 2,
            "out_dim": 8,
            "input_sampling_rate": 24000,
            "input_hop_length": 960,
            "output_sampling_rate": 44100,
            "output_hop_length": 512,
        },
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
        return_value={
            "_class_name": "MiniMaxMusic3ModularPipeline",
            "condition_encoder": [
                "diffusers",
                "MiniMaxMusic3ConditionEncoder",
                {
                    "pretrained_model_name_or_path": "fake/music3",
                    "revision": None,
                    "subfolder": "condition_encoder",
                },
            ],
            "scheduler": [
                "diffusers",
                "FlowMatchEulerDiscreteScheduler",
                {
                    "pretrained_model_name_or_path": "external/scheduler",
                    "revision": "scheduler-revision",
                    "subfolder": "flow",
                },
            ],
        },
    )
    def test_builds_modular_index_entry_with_metadata(
        self, _mock_index, _mock_config, _mock_optional
    ):
        package = build_diffusers_pipeline(
            "fake/music3", revision="root-revision", load_weights=False
        )
        assert set(package) == {"condition_encoder"}
        assert package.config.model_type == "minimax_music3"
        assert package.config.pipeline_class == "MiniMaxMusic3ModularPipeline"
        _mock_config.assert_called_once_with(
            "fake/music3",
            "condition_encoder",
            revision="root-revision",
            subfolder="condition_encoder",
        )
        _mock_optional.assert_called_once_with(
            "external/scheduler",
            "flow/scheduler_config.json",
            revision="scheduler-revision",
        )

    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_skips_non_list_entries(self, mock_load_index, mock_load_config):
        """Non-list entries (e.g. strings, dicts) are skipped."""
        mock_load_index.return_value = {
            "_class_name": "FluxPipeline",
            "some_string": "not a list",
            "some_dict": {"key": "value"},
            "some_int": 42,
        }
        with pytest.raises(ValueError, match="No supported neural network"):
            build_diffusers_pipeline("fake/model", load_weights=False)
        mock_load_config.assert_not_called()

    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_skips_lists_with_wrong_length(self, mock_load_index, mock_load_config):
        """Lists that don't have two elements or two plus metadata are skipped."""
        mock_load_index.return_value = {
            "_class_name": "FluxPipeline",
            "single": ["only_one"],
            "quadruple": ["a", "b", {}, "extra"],
            "empty": [],
        }
        with pytest.raises(ValueError, match="No supported neural network"):
            build_diffusers_pipeline("fake/model", load_weights=False)
        mock_load_config.assert_not_called()

    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_skips_unregistered_class_names(self, mock_load_index, mock_load_config):
        """Components with unregistered class names are skipped with a log message."""
        mock_load_index.return_value = {
            "_class_name": "FluxPipeline",
            "scheduler": ["diffusers", "EulerDiscreteScheduler"],
            "text_encoder": ["transformers", "UnsupportedTextEncoderModel"],
        }
        with pytest.raises(ValueError, match="No supported neural network"):
            build_diffusers_pipeline("fake/model", load_weights=False)
        mock_load_config.assert_not_called()


# ── build_diffusers_pipeline successful build ────────────────────────────


class TestBuildDiffusersPipelineSuccess:
    """Tests for successful build_diffusers_pipeline calls."""

    def _mock_build_for_vae(self, mock_load_index, mock_load_config, mock_build_from_module):
        """Set up mocks for a minimal VAE component build."""
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        # Minimal diffusers VAE config
        mock_load_config.return_value = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
        }
        # Return a fake ModelPackage with a minimal model
        graph = ir.Graph([], [], nodes=[], name="fake_vae")
        model = ir.Model(graph, ir_version=10)
        mock_build_from_module.return_value = ModelPackage({"model": model})

    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_returns_model_package(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """Successful build returns a ModelPackage."""
        self._mock_build_for_vae(mock_load_index, mock_load_config, mock_build_from_module)
        result = build_diffusers_pipeline("fake/vae-model", load_weights=False)
        assert isinstance(result, ModelPackage)

    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_single_model_subpackage_flattened(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """When sub-package has one 'model' entry, it's flattened to component name."""
        self._mock_build_for_vae(mock_load_index, mock_load_config, mock_build_from_module)
        result = build_diffusers_pipeline("fake/vae-model", load_weights=False)
        # The "model" key from sub-package becomes "vae" in the top-level package
        assert "vae" in result
        assert "model" not in result

    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_graph_name_set_to_model_id_component(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """Graph name is set to '{model_id}/{component_name}'."""
        self._mock_build_for_vae(mock_load_index, mock_load_config, mock_build_from_module)
        result = build_diffusers_pipeline("fake/vae-model", load_weights=False)
        assert result["vae"].graph.name == "fake/vae-model/vae"

    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_multi_model_subpackage_prefixed(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """When sub-package has multiple entries, they're prefixed with component name."""
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        mock_load_config.return_value = {"in_channels": 3}
        # Return a sub-package with multiple models
        graph_a = ir.Graph([], [], nodes=[], name="encoder")
        graph_b = ir.Graph([], [], nodes=[], name="decoder")
        mock_build_from_module.return_value = ModelPackage(
            {
                "encoder": ir.Model(graph_a, ir_version=10),
                "decoder": ir.Model(graph_b, ir_version=10),
            }
        )
        result = build_diffusers_pipeline("fake/multi", load_weights=False)
        assert "vae_encoder" in result
        assert "vae_decoder" in result
        assert result["vae_encoder"].graph.name == "fake/multi/vae_encoder"
        assert result["vae_decoder"].graph.name == "fake/multi/vae_decoder"

    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_multiple_components_built(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """Pipeline with multiple supported components builds all of them."""
        mock_load_index.return_value = _fake_pipeline_index(
            {
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                # Classic-SD CLIP text encoder is now a supported component too.
                "text_encoder": ["transformers", "CLIPTextModel"],
            }
        )
        mock_load_config.return_value = {}

        def fake_build(module, config, task_name, **kwargs):
            graph = ir.Graph([], [], nodes=[], name="g")
            return ModelPackage({"model": ir.Model(graph, ir_version=10)})

        mock_build_from_module.side_effect = fake_build

        result = build_diffusers_pipeline("fake/flux", load_weights=False)
        assert "transformer" in result
        assert "vae" in result
        assert "text_encoder" in result

    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_dtype_string_resolved(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """String dtype is resolved to ir.DataType and passed to config."""
        self._mock_build_for_vae(mock_load_index, mock_load_config, mock_build_from_module)
        # Should not raise — dtype string "f16" is resolved by resolve_dtype()
        build_diffusers_pipeline("fake/vae-model", dtype="f16", load_weights=False)
        # Verify build_from_module was called (string dtype resolved without error)
        mock_build_from_module.assert_called_once()

    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_dtype_ir_datatype_passthrough(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """ir.DataType dtype is passed through without conversion."""
        self._mock_build_for_vae(mock_load_index, mock_load_config, mock_build_from_module)
        build_diffusers_pipeline(
            "fake/vae-model",
            dtype=ir.DataType.FLOAT16,
            load_weights=False,
        )
        # Verify build_from_module was called (ir.DataType accepted without error)
        mock_build_from_module.assert_called_once()

    @patch("mobius.integrations.diffusers._builder._download_diffusers_component_weights")
    @patch("mobius.integrations.diffusers._builder.apply_weights")
    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch("mobius.integrations.diffusers._builder._load_diffusers_component_config")
    @patch("mobius.integrations.diffusers._builder._load_diffusers_pipeline_index")
    def test_revision_propagates_to_all_pipeline_artifacts(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
        mock_apply_weights,
        mock_download_weights,
    ):
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        mock_load_config.return_value = {}
        graph = ir.Graph([], [], nodes=[], name="vae")
        model = ir.Model(graph, ir_version=10)
        mock_build_from_module.return_value = ModelPackage({"model": model})
        mock_download_weights.return_value = {}

        build_diffusers_pipeline("fake/model", revision="pinned-revision")

        mock_load_index.assert_called_once_with("fake/model", revision="pinned-revision")
        mock_load_config.assert_called_once_with(
            "fake/model", "vae", revision="pinned-revision"
        )
        mock_download_weights.assert_called_once_with(
            "fake/model", "vae", revision="pinned-revision"
        )
        mock_apply_weights.assert_called_once()

    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch("mobius.integrations.diffusers._builder._load_diffusers_component_config")
    @patch("mobius.integrations.diffusers._builder._load_diffusers_pipeline_index")
    def test_qwen_edit_uses_normalized_vae_task(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        mock_load_index.return_value = {
            "_class_name": "QwenImageEditPlusPipeline",
            "vae": ["diffusers", "AutoencoderKLQwenImage"],
        }
        mock_load_config.return_value = {
            "base_dim": 8,
            "z_dim": 4,
            "dim_mult": [1, 2],
            "num_res_blocks": 1,
            "temperal_downsample": [False],
            "latents_mean": [0.0] * 4,
            "latents_std": [1.0] * 4,
        }
        graph = ir.Graph([], [], nodes=[], name="vae")
        mock_build_from_module.return_value = ModelPackage(
            {"model": ir.Model(graph, ir_version=10)}
        )

        result = build_diffusers_pipeline("fake/qwen-edit", load_weights=False)

        assert "vae" in result
        assert mock_build_from_module.call_args.args[2] == "qwen-image-edit-vae"
        assert result.config.model_type == "qwen_image_edit"

    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch("mobius.integrations.diffusers._builder._load_diffusers_component_config")
    @patch("mobius.integrations.diffusers._builder._load_diffusers_pipeline_index")
    def test_component_allowlist_avoids_building_other_components(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        mock_load_index.return_value = _fake_pipeline_index(
            {
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
        mock_load_config.return_value = {}
        graph = ir.Graph([], [], nodes=[], name="transformer")
        mock_build_from_module.return_value = ModelPackage(
            {"model": ir.Model(graph, ir_version=10)}
        )

        result = build_diffusers_pipeline(
            "fake/filtered",
            load_weights=False,
            components={"transformer"},
        )

        assert set(result) == {"transformer"}
        mock_load_config.assert_called_once_with("fake/filtered", "transformer", revision=None)


# ── build_diffusers_pipeline weight loading ──────────────────────────────


class TestBuildDiffusersPipelineWeights:
    """Tests for weight loading paths in build_diffusers_pipeline."""

    @patch("mobius.integrations.diffusers._builder.fold_initializers_after_weights")
    @patch("mobius.integrations.diffusers._builder.apply_weights")
    @patch(
        "mobius.integrations.diffusers._builder._download_diffusers_component_weights",
    )
    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_load_weights_true_downloads_and_applies(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
        mock_download_weights,
        mock_apply_weights,
        mock_fold_initializers,
    ):
        """When load_weights=True, weights are downloaded and applied."""
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        mock_load_config.return_value = {}
        graph = ir.Graph([], [], nodes=[], name="vae")
        model = ir.Model(graph, ir_version=10)
        mock_build_from_module.return_value = ModelPackage({"model": model})
        mock_download_weights.return_value = {}

        build_diffusers_pipeline("fake/model", load_weights=True)

        mock_download_weights.assert_called_once_with("fake/model", "vae", revision=None)
        mock_apply_weights.assert_called_once()
        mock_fold_initializers.assert_called_once_with(model)

    @patch(
        "mobius.integrations.diffusers._builder._download_diffusers_component_weights",
    )
    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_load_weights_false_skips_download(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
        mock_download_weights,
    ):
        """When load_weights=False, no weight download occurs."""
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        mock_load_config.return_value = {}
        graph = ir.Graph([], [], nodes=[], name="vae")
        mock_build_from_module.return_value = ModelPackage(
            {"model": ir.Model(graph, ir_version=10)}
        )

        build_diffusers_pipeline("fake/model", load_weights=False)
        mock_download_weights.assert_not_called()

    @patch("mobius.integrations.diffusers._builder.apply_weights")
    @patch(
        "mobius.integrations.diffusers._builder._download_diffusers_component_weights",
    )
    @patch("mobius.integrations.diffusers._builder.build_from_module")
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_component_config",
    )
    @patch(
        "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
    )
    def test_preprocess_weights_called_when_available(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
        mock_download_weights,
        mock_apply_weights,
    ):
        """preprocess_weights is called on the module when it has the method."""
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        mock_load_config.return_value = {}
        graph = ir.Graph([], [], nodes=[], name="vae")
        model = ir.Model(graph, ir_version=10)
        mock_build_from_module.return_value = ModelPackage({"model": model})
        raw_weights = {"weight.data": "raw"}
        processed_weights = {"weight.data": "processed"}
        mock_download_weights.return_value = raw_weights

        # The module class will have preprocess_weights set by AutoencoderKLModel
        # We patch it at the module instance level via build_from_module's first arg
        def capture_build(module, config, task_name, **kwargs):
            module.preprocess_weights = lambda sd: processed_weights
            return ModelPackage({"model": model})

        mock_build_from_module.side_effect = capture_build

        build_diffusers_pipeline("fake/model", load_weights=True)
        # apply_weights should receive the processed weights
        call_args = mock_apply_weights.call_args
        assert call_args[0][1] is processed_weights


def test_prepare_unet_loras_infers_rank_and_merges(tmp_path):
    import torch
    from safetensors.torch import save_file

    from mobius.integrations.diffusers._builder import _prepare_unet_loras

    path = tmp_path / "style.safetensors"
    save_file(
        {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight": torch.zeros(
                8, 32
            ),
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.up.weight": torch.zeros(
                32, 8
            ),
        },
        str(path),
    )
    adapters, merged = _prepare_unet_loras({"style": str(path)})
    assert adapters == (("style", 8, 1.0),)  # rank inferred from lora_A [rank, in]
    assert set(merged) == {
        "down_blocks.0.attentions.0.attn1.to_q.lora_A.style.weight",
        "down_blocks.0.attentions.0.attn1.to_q.lora_B.style.weight",
    }
