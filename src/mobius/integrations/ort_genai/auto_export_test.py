# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the ORT-GenAI auto-export pipeline."""

from __future__ import annotations

import json
import os
import types
from unittest import mock

import numpy as np
import onnx_ir as ir
import pytest

from mobius.integrations.ort_genai.auto_export import (
    _copy_tokenizer_files,
    _copy_tokenizer_files_from_local,
    _count_cache_layer_slots,
    _fix_chat_template,
    _fix_tokenizer_config,
    _graph_input_names,
    _introspect_outputs,
    _resolve_ort_genai_model_type,
    _select_ort_model_type,
    _write_audio_processor_config,
    _write_genai_config,
    _write_vision_processor_config,
    auto_export,
    export_package,
    write_ort_genai_config,
)


def _mock_model(
    *, inputs: list[str] | None = None, outputs: list[str] | None = None
) -> ir.Model:
    """Create a minimal ir.Model for package and graph-introspection tests."""
    graph = ir.Graph(
        inputs=[ir.Value(name=name) for name in inputs or []],
        outputs=[ir.Value(name=name) for name in outputs or []],
        nodes=[],
        name="mock_model",
    )
    return ir.Model(graph, ir_version=10)


def _mock_model_with_inputs(names: list[str]) -> ir.Model:
    """Create a minimal ir.Model whose graph inputs have the given names."""
    return _mock_model(inputs=names)


def _mock_model_with_outputs(names: list[str]) -> ir.Model:
    """Create a minimal ir.Model whose graph outputs have the given names."""
    return _mock_model(outputs=names)


def test_moonshine_native_runtime_is_rejected(tmp_path):
    from mobius._model_package import ModelPackage

    config = mock.MagicMock()
    config.model_type = "moonshine"
    package = ModelPackage(
        {"encoder": _mock_model(), "decoder": _mock_model()},
        config=config,
    )

    with pytest.raises(
        NotImplementedError,
        match="variable-length raw-waveform encoder",
    ) as error:
        write_ort_genai_config(package, str(tmp_path))
    assert "onnx-genai" not in str(error.value)
    assert "ONNX Runtime" in str(error.value)


def _make_fake_llm_pkg(model_type: str = "qwen2"):
    """Build a minimal LLM-only ModelPackage with a fake config."""
    import dataclasses

    from mobius._model_package import ModelPackage

    @dataclasses.dataclass
    class FakeConfig:
        model_type: str = "qwen2"
        vocab_size: int = 256
        hidden_size: int = 64
        num_hidden_layers: int = 2
        num_attention_heads: int = 4
        num_key_value_heads: int = 2
        head_dim: int = 16
        max_position_embeddings: int = 128

    return ModelPackage(
        {"model": _mock_model()},
        config=FakeConfig(model_type=model_type),
    )


class TestResolveOrtGenaiModelType:
    def test_known_model_type(self):
        assert _resolve_ort_genai_model_type("qwen3") == "qwen2"
        assert _resolve_ort_genai_model_type("gemma2") == "gemma"
        assert _resolve_ort_genai_model_type("llama") == "llama"

    def test_hunyuan_v1_dense_maps_to_decoder(self):
        # ORT GenAI accepts "decoder" as a generic LLM type for any
        # decoder-only causal LM not in its built-in registry.
        assert _resolve_ort_genai_model_type("hunyuan_v1_dense") == "decoder"

    def test_unknown_model_type_passthrough(self):
        assert _resolve_ort_genai_model_type("my_custom") == "my_custom"

    def test_mistral3_model_type(self):
        assert _resolve_ort_genai_model_type("mistral3") == "mistral3"
        # Text-only mistral is a separate mapping
        assert _resolve_ort_genai_model_type("mistral") == "mistral"

    def test_phi4mm_model_types(self):
        assert _resolve_ort_genai_model_type("phi4mm") == "phi4mm"
        assert _resolve_ort_genai_model_type("phi4_multimodal") == "phi4mm"
        assert _resolve_ort_genai_model_type("phi") == "phi"

    def test_gemma4_unified_model_types(self):
        # The gemma-4-12B unified checkpoint (model_type "gemma4_unified")
        # reuses the multimodal "gemma4" ORT GenAI pipeline; its standalone
        # text decoder ("gemma4_unified_text") maps to "gemma4_text".
        assert _resolve_ort_genai_model_type("gemma4_unified") == "gemma4"
        assert _resolve_ort_genai_model_type("gemma4_unified_text") == "gemma4_text"
        # Released gemma4 mappings remain unchanged.
        assert _resolve_ort_genai_model_type("gemma4") == "gemma4"
        assert _resolve_ort_genai_model_type("gemma4_text") == "gemma4_text"


class TestSelectOrtModelType:
    """Text-only / multimodal ORT model type selection (PR: text_only export)."""

    def test_decoder_only_prefers_config_type(self):
        # Text-only gemma-4-12B: package config carries the text sibling, HF
        # reports the multimodal type. Decoder-only -> follow the package.
        assert (
            _select_ort_model_type(
                "gemma4_unified_text", "gemma4_unified", is_decoder_only=True
            )
            == "gemma4_text"
        )

    def test_multimodal_keeps_hf_type(self):
        # Full multimodal export: build() unwraps the composite to its text
        # sub-config, so config.model_type may be a text type even though the
        # package is multimodal. Must keep the HF parent type -> gemma4.
        assert (
            _select_ort_model_type(
                "gemma4_unified_text", "gemma4_unified", is_decoder_only=False
            )
            == "gemma4"
        )

    def test_decoder_only_falls_back_to_hf_when_config_missing(self):
        assert _select_ort_model_type(None, "qwen3", is_decoder_only=True) == "qwen2"

    def test_decoder_only_unknown_config_falls_back_to_hf(self):
        # An unrecognised config.model_type (not in _ORT_GENAI_MODEL_TYPE) must
        # not pass straight through as an invalid ORT type; fall back to the
        # known HF-derived mapping instead.
        assert (
            _select_ort_model_type("not_a_real_type", "qwen3", is_decoder_only=True) == "qwen2"
        )


class TestWriteProcessorConfig:
    def test_no_vision_returns_none(self, tmp_path):
        config = mock.MagicMock(spec=[])
        del config.vision  # ensure no vision attribute
        assert _write_vision_processor_config(config, str(tmp_path)) is None

    def test_writes_transform_pipeline(self, tmp_path):
        """Generates full transform pipeline for generic VLMs."""
        vision = mock.MagicMock()
        vision.image_size = 448
        vision.patch_size = 14
        vision.spatial_merge_size = 2
        vision.model_type = None
        config = mock.MagicMock()
        config.vision = vision
        config.spatial_merge_size = 2

        path = _write_vision_processor_config(config, str(tmp_path))
        assert path is not None
        with open(path) as f:
            data = json.load(f)

        proc = data["processor"]
        assert proc["name"] == "image_processor"
        transforms = proc["transforms"]
        assert len(transforms) == 4
        assert transforms[0]["operation"]["type"] == "DecodeImage"
        assert transforms[1]["operation"]["type"] == "Resize"
        assert transforms[2]["operation"]["type"] == "Rescale"
        assert transforms[3]["operation"]["type"] == "Normalize"

        # Check resize attrs
        resize_attrs = transforms[1]["operation"]["attrs"]
        assert resize_attrs["patch_size"] == 14
        assert resize_attrs["merge_size"] == 2

        # Check normalization defaults (CLIP-standard)
        norm_attrs = transforms[3]["operation"]["attrs"]
        assert len(norm_attrs["mean"]) == 3
        assert len(norm_attrs["std"]) == 3

    def test_mage_vl_writes_packed_patch_processor(self, tmp_path):
        vision = types.SimpleNamespace(
            image_size=448,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=1,
            model_type="mage_vl_vision",
        )
        config = types.SimpleNamespace(
            vision=vision,
            model_type="mage_vl",
            spatial_merge_size=2,
            temporal_patch_size=1,
        )

        path = _write_vision_processor_config(config, str(tmp_path))
        assert path is not None
        assert path.endswith("image_processor.json")
        with open(path) as f:
            data = json.load(f)

        assert data["processor"]["name"] == "qwen2_5_image_processor"
        transforms = data["processor"]["transforms"]
        assert transforms[-1]["operation"] == {
            "name": "patch_image",
            "type": "PatchImage",
            "attrs": {
                "patch_size": 16,
                "temporal_patch_size": 1,
                "merge_size": 2,
            },
        }
        normalize = next(
            transform["operation"]
            for transform in transforms
            if transform["operation"]["type"] == "Normalize"
        )
        assert normalize["attrs"]["qwen2_5_vl"] == 1

    def test_mage_vl_processor_propagates_trust_remote_code(self, tmp_path):
        vision = types.SimpleNamespace(
            image_size=448,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=1,
            model_type="mage_vl_vision",
        )
        config = types.SimpleNamespace(
            vision=vision,
            model_type="mage_vl",
            spatial_merge_size=2,
            temporal_patch_size=1,
        )
        hf_processor = mock.MagicMock()
        hf_processor.image_processor = None
        with mock.patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=hf_processor,
        ) as from_pretrained:
            _write_vision_processor_config(
                config,
                str(tmp_path),
                hf_model_id="microsoft/Mage-VL",
                trust_remote_code=True,
            )

        from_pretrained.assert_called_once_with(
            "microsoft/Mage-VL",
            trust_remote_code=True,
        )

    def test_gemma4_unified_skips_image_processor(self, tmp_path):
        """Encoder-free gemma4_unified has no native transform: no image_processor.json."""
        vision = mock.MagicMock()
        vision.model_type = None
        config = mock.MagicMock()
        config.vision = vision
        config.model_type = "gemma4_unified"
        assert _write_vision_processor_config(config, str(tmp_path)) is None

    def test_pixtral_vision_config(self, tmp_path):
        """Generates pixtral-specific processor config with 7 transforms."""
        vision = mock.MagicMock()
        vision.image_size = 1540
        vision.patch_size = 14
        vision.spatial_merge_size = 2
        vision.model_type = "pixtral"
        config = mock.MagicMock()
        config.vision = vision
        config.spatial_merge_size = 2
        config.model_type = "mistral3"

        path = _write_vision_processor_config(config, str(tmp_path))
        assert path is not None
        assert path.endswith("processor_config.json")
        with open(path) as f:
            data = json.load(f)

        proc = data["processor"]
        assert proc["name"] == "pixtral_image_processor"
        transforms = proc["transforms"]
        assert len(transforms) == 6

        # Verify all 6 transform types in order
        types = [t["operation"]["type"] for t in transforms]
        assert types == [
            "DecodeImage",
            "Resize",
            "Rescale",
            "Normalize",
            "Permute3D",
            "PixtralImageSizes",
        ]

        resize = transforms[1]["operation"]["attrs"]
        assert resize["height"] == 1540
        assert resize["width"] == 1540

        # Permute3D has correct dims
        permute = transforms[4]["operation"]["attrs"]
        assert permute["dims"] == [2, 0, 1]

    def test_muse_glimmer_uses_packed_qwen_image_pipeline(self, tmp_path):
        """Muse vision consumes flattened patches and image grid dimensions.

        No ``ConvertRGB``: it unconditionally swaps R and B, so pairing it with
        ``DecodeImage(color_space="RGB")`` would hand the encoder BGR. The
        ``qwen2_5_vl`` flag therefore lands on ``Normalize`` at index 3.
        """
        vision = mock.MagicMock()
        vision.image_size = 448
        vision.patch_size = 14
        vision.spatial_merge_size = 2
        vision.model_type = "muse_glimmer_vision"
        config = mock.MagicMock()
        config.vision = vision
        # Composite builds unwrap to the text config before processor export.
        config.model_type = "muse_glimmer_text"
        config.spatial_merge_size = 2
        config.temporal_patch_size = 2

        path = _write_vision_processor_config(config, str(tmp_path))
        assert path is not None
        with open(path) as f:
            data = json.load(f)

        proc = data["processor"]
        assert proc["name"] == "qwen2_5_image_processor"
        transforms = proc["transforms"]
        assert [t["operation"]["type"] for t in transforms] == [
            "DecodeImage",
            "Resize",
            "Rescale",
            "Normalize",
            "PatchImage",
        ]
        assert transforms[3]["operation"]["attrs"]["qwen2_5_vl"] == 1
        assert transforms[4]["operation"]["attrs"] == {
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
        }

    def test_qwen35_moe_text_uses_packed_qwen_image_pipeline(self, tmp_path):
        """Qwen3.6 VL builds unwrap to the MoE text config before processor export."""
        vision = mock.MagicMock()
        vision.image_size = 16_777_216
        vision.patch_size = 16
        vision.spatial_merge_size = 2
        config = mock.MagicMock()
        config.vision = vision
        config.model_type = "qwen3_5_moe_text"
        config.spatial_merge_size = 2
        config.temporal_patch_size = 2

        path = _write_vision_processor_config(config, str(tmp_path))
        assert path is not None
        with open(path) as f:
            data = json.load(f)

        proc = data["processor"]
        assert proc["name"] == "qwen2_5_image_processor"
        transforms = proc["transforms"]
        assert [t["operation"]["type"] for t in transforms] == [
            "DecodeImage",
            "Resize",
            "Rescale",
            "Normalize",
            "PatchImage",
        ]
        assert transforms[3]["operation"]["attrs"]["qwen2_5_vl"] == 1
        assert transforms[4]["operation"]["attrs"] == {
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
        }

    def test_gemma3_vision_config(self, tmp_path):
        """Gemma3 gets a fixed-size resize + Permute3D (not the generic branch).

        Regression guard: gemma3's config unwraps to text_config, so
        ``config.model_type`` is "gemma3_text". The generic-VLM branch would
        emit smart_resize (variable HxW) with min_pixels/max_pixels and no
        Permute3D, producing a variable-size HWC tensor that fails the SigLIP
        encoder's fixed NCHW [batch, 3, 896, 896] input.
        """
        vision = mock.MagicMock()
        vision.image_size = 896
        vision.model_type = "siglip_vision_model"
        config = mock.MagicMock()
        config.vision = vision
        # Unwrapped text-config model_type — NOT "gemma3".
        config.model_type = "gemma3_text"

        # No hf_model_id → uses gemma3 defaults (image_size=896, mean/std=0.5).
        path = _write_vision_processor_config(config, str(tmp_path))
        assert path is not None
        assert path.endswith("processor_config.json")
        with open(path) as f:
            data = json.load(f)

        proc = data["processor"]
        transforms = proc["transforms"]

        # 5-step pipeline ending in Permute3D (HWC→CHW).
        types = [t["operation"]["type"] for t in transforms]
        assert types == [
            "DecodeImage",
            "Resize",
            "Rescale",
            "Normalize",
            "Permute3D",
        ]

        # Fixed-size resize: smart_resize disabled, no variable-pixel bounds.
        resize = transforms[1]["operation"]["attrs"]
        assert resize["smart_resize"] == 0
        assert resize["height"] == 896
        assert resize["width"] == 896
        assert "min_pixels" not in resize
        assert "max_pixels" not in resize
        # SigLIP resamples bilinear; ort-extensions would default to CUBIC.
        assert resize["interpolation"] == "LINEAR"

        # Trailing Permute3D matches the encoder's channels-first contract.
        assert transforms[4]["operation"]["attrs"]["dims"] == [2, 0, 1]

    def test_gemma3n_vision_config_omits_normalize(self, tmp_path):
        """Gemma3n shares gemma3's fixed resize but skips Normalize.

        Its ``SiglipImageProcessorFast`` sets ``do_normalize=False``, so the
        MobileNet-V5 tower is trained on [0, 1] pixels.  Emitting a mean/std-0.5
        Normalize would map them to [-1, 1] and silently degrade every caption.
        """
        vision = mock.MagicMock()
        vision.image_size = 768
        vision.model_type = "gemma3n_vision"
        config = mock.MagicMock()
        config.vision = vision
        # Unwrapped text-config model_type — NOT "gemma3n".
        config.model_type = "gemma3n_text"

        path = _write_vision_processor_config(config, str(tmp_path))
        assert path is not None
        assert path.endswith("processor_config.json")
        with open(path) as f:
            data = json.load(f)

        transforms = data["processor"]["transforms"]
        types = [t["operation"]["type"] for t in transforms]
        assert types == [
            "DecodeImage",
            "Resize",
            "Rescale",
            "Permute3D",
        ]

        # Fixed 768x768 resize (MobileNet-V5 has no dynamic-resolution path).
        resize = transforms[1]["operation"]["attrs"]
        assert resize["smart_resize"] == 0
        assert resize["height"] == 768
        assert resize["width"] == 768
        assert transforms[2]["operation"]["attrs"]["rescale_factor"] == pytest.approx(
            1.0 / 255.0
        )
        assert transforms[-1]["operation"]["attrs"]["dims"] == [2, 0, 1]

    def test_gemma3n_vision_config_honours_hf_do_normalize(self, tmp_path):
        """A checkpoint that *does* normalize gets the Normalize step back."""
        vision = mock.MagicMock()
        vision.image_size = 768
        config = mock.MagicMock()
        config.vision = vision
        config.model_type = "gemma3n_text"

        image_processor = mock.MagicMock()
        image_processor.image_mean = [0.5, 0.5, 0.5]
        image_processor.image_std = [0.5, 0.5, 0.5]
        image_processor.rescale_factor = 1.0 / 255.0
        image_processor.do_normalize = True
        image_processor.resample = 2
        image_processor.size = {"height": 768, "width": 768}
        hf_proc = mock.MagicMock()
        hf_proc.image_processor = image_processor

        with mock.patch("transformers.AutoProcessor.from_pretrained", return_value=hf_proc):
            path = _write_vision_processor_config(
                config, str(tmp_path), hf_model_id="google/gemma-3n-E4B-it"
            )

        with open(path) as f:
            transforms = json.load(f)["processor"]["transforms"]
        types = [t["operation"]["type"] for t in transforms]
        assert types == [
            "DecodeImage",
            "Resize",
            "Rescale",
            "Normalize",
            "Permute3D",
        ]

    @pytest.mark.parametrize(
        "model_type,vision_model_type",
        [
            ("gemma3n_text", "gemma3n_vision"),
            ("gemma3_text", "siglip_vision_model"),
            ("mistral3", "pixtral"),
            ("paligemma", "siglip_vision_model"),
        ],
    )
    def test_no_pipeline_emits_convert_rgb(self, tmp_path, model_type, vision_model_type):
        """ConvertRGB after DecodeImage(RGB) hands the encoder BGR.

        ort-extensions' ``convert_to_rgb`` swaps R and B *unconditionally* — it
        is the fix-up for a BGR decode, not a no-op assertion of RGB-ness.
        Chaining it onto ``DecodeImage(color_space="RGB")`` therefore inverted
        the red and blue channels of every image fed to every exported VLM.
        Measured against HF's own processor on a 1920x1242 JPEG, that put the
        pixel tensor 22% (relative L2) away from the reference; dropping the
        step brings it to 0.01%.
        """
        vision = mock.MagicMock()
        vision.image_size = 768
        vision.patch_size = 14
        vision.spatial_merge_size = 2
        vision.model_type = vision_model_type
        config = mock.MagicMock()
        config.vision = vision
        config.model_type = model_type
        config.spatial_merge_size = 2

        path = _write_vision_processor_config(config, str(tmp_path))
        with open(path) as f:
            transforms = json.load(f)["processor"]["transforms"]

        types = [t["operation"]["type"] for t in transforms]
        assert "ConvertRGB" not in types
        assert types[0] == "DecodeImage"
        assert transforms[0]["operation"]["attrs"]["color_space"] == "RGB"

    @pytest.mark.parametrize(
        "resample,expected",
        [(0, "NEAREST"), (1, "LANCZOS"), (2, "LINEAR"), (3, "CUBIC")],
    )
    def test_resize_interpolation_follows_hf_resample(self, tmp_path, resample, expected):
        """The HF processor's PIL ``resample`` must reach the Resize step.

        ort-extensions defaults to CUBIC while HF image processors
        overwhelmingly use PIL BILINEAR (``resample=2``), so leaving the
        attribute off silently resamples with the wrong kernel.
        """
        vision = mock.MagicMock()
        vision.image_size = 768
        vision.model_type = "gemma3n_vision"
        config = mock.MagicMock()
        config.vision = vision
        config.model_type = "gemma3n_text"

        image_processor = mock.MagicMock()
        image_processor.image_mean = [0.5, 0.5, 0.5]
        image_processor.image_std = [0.5, 0.5, 0.5]
        image_processor.rescale_factor = 1.0 / 255.0
        image_processor.do_normalize = False
        image_processor.resample = resample
        image_processor.size = {"height": 768, "width": 768}
        hf_proc = mock.MagicMock()
        hf_proc.image_processor = image_processor

        with mock.patch("transformers.AutoProcessor.from_pretrained", return_value=hf_proc):
            path = _write_vision_processor_config(
                config, str(tmp_path), hf_model_id="google/gemma-3n-E4B-it"
            )

        with open(path) as f:
            transforms = json.load(f)["processor"]["transforms"]
        resize = next(t["operation"] for t in transforms if t["operation"]["type"] == "Resize")
        assert resize["attrs"]["interpolation"] == expected

    def test_unsupported_resample_omits_interpolation(self, tmp_path):
        """PIL BOX/HAMMING have no ort-extensions filter: fall back, don't crash."""
        vision = mock.MagicMock()
        vision.image_size = 768
        vision.model_type = "gemma3n_vision"
        config = mock.MagicMock()
        config.vision = vision
        config.model_type = "gemma3n_text"

        image_processor = mock.MagicMock()
        image_processor.image_mean = [0.5, 0.5, 0.5]
        image_processor.image_std = [0.5, 0.5, 0.5]
        image_processor.rescale_factor = 1.0 / 255.0
        image_processor.do_normalize = False
        image_processor.resample = 4  # PIL BOX
        image_processor.size = {"height": 768, "width": 768}
        hf_proc = mock.MagicMock()
        hf_proc.image_processor = image_processor

        with mock.patch("transformers.AutoProcessor.from_pretrained", return_value=hf_proc):
            path = _write_vision_processor_config(
                config, str(tmp_path), hf_model_id="google/gemma-3n-E4B-it"
            )

        with open(path) as f:
            transforms = json.load(f)["processor"]["transforms"]
        resize = next(t["operation"] for t in transforms if t["operation"]["type"] == "Resize")
        assert "interpolation" not in resize["attrs"]

    def test_size_mapping_reads_transformers_v5_size_dict(self, tmp_path):
        """``image_processor.size`` is a SizeDict, not a dict, in transformers >= 5.

        An ``isinstance(size, dict)`` guard silently discards it and falls back
        to the hardcoded per-family default, so a checkpoint at any other
        resolution would export a processor config that disagrees with the
        encoder's own input shape.
        """

        class SizeDict:  # mirrors transformers.image_utils.SizeDict: not a dict
            height = 512
            width = 512
            longest_edge = None
            shortest_edge = None

            def get(self, key, default=None):
                return getattr(self, key, default)

        vision = mock.MagicMock()
        vision.image_size = 768
        vision.model_type = "gemma3n_vision"
        config = mock.MagicMock()
        config.vision = vision
        config.model_type = "gemma3n_text"

        image_processor = mock.MagicMock()
        image_processor.image_mean = [0.5, 0.5, 0.5]
        image_processor.image_std = [0.5, 0.5, 0.5]
        image_processor.rescale_factor = 1.0 / 255.0
        image_processor.do_normalize = False
        image_processor.resample = 2
        image_processor.size = SizeDict()
        hf_proc = mock.MagicMock()
        hf_proc.image_processor = image_processor

        with mock.patch("transformers.AutoProcessor.from_pretrained", return_value=hf_proc):
            path = _write_vision_processor_config(
                config, str(tmp_path), hf_model_id="google/gemma-3n-E4B-it"
            )

        with open(path) as f:
            transforms = json.load(f)["processor"]["transforms"]
        resize = next(t["operation"] for t in transforms if t["operation"]["type"] == "Resize")
        assert resize["attrs"]["height"] == 512
        assert resize["attrs"]["width"] == 512

    def test_gemma3_vision_config_keeps_normalize(self, tmp_path):
        """Sharing the branch with gemma3n must not drop gemma3's Normalize."""
        vision = mock.MagicMock()
        vision.image_size = 896
        vision.model_type = "siglip_vision_model"
        config = mock.MagicMock()
        config.vision = vision
        config.model_type = "gemma3_text"

        path = _write_vision_processor_config(config, str(tmp_path))

        with open(path) as f:
            transforms = json.load(f)["processor"]["transforms"]
        normalize = next(
            t["operation"] for t in transforms if t["operation"]["type"] == "Normalize"
        )
        assert normalize["attrs"]["mean"] == [0.5, 0.5, 0.5]
        assert normalize["attrs"]["std"] == [0.5, 0.5, 0.5]

    def test_siglip_vision_config_non_gemma3_uses_generic_branch(self, tmp_path):
        """A SigLIP vision tower alone is not enough to select Gemma3 preprocessing."""
        vision = mock.MagicMock()
        vision.image_size = 448
        vision.patch_size = 14
        vision.spatial_merge_size = 2
        vision.model_type = "siglip_vision_model"
        config = mock.MagicMock()
        config.vision = vision
        config.model_type = "paligemma"
        config.spatial_merge_size = 2

        path = _write_vision_processor_config(config, str(tmp_path))
        assert path is not None
        with open(path) as f:
            data = json.load(f)

        transforms = data["processor"]["transforms"]
        types = [t["operation"]["type"] for t in transforms]
        assert types == ["DecodeImage", "Resize", "Rescale", "Normalize"]
        resize = transforms[1]["operation"]["attrs"]
        assert resize["smart_resize"] == 1
        assert "min_pixels" in resize
        assert "max_pixels" in resize

    def test_hf_processor_fallback_to_clip_defaults(self, tmp_path):
        """Falls back to CLIP-standard defaults when HF processor can't be loaded."""
        vision = mock.MagicMock()
        vision.image_size = 448
        vision.patch_size = 14
        vision.spatial_merge_size = 2
        vision.model_type = None
        config = mock.MagicMock()
        config.vision = vision
        config.spatial_merge_size = 2
        config.model_type = "qwen2"

        with mock.patch(
            "transformers.AutoProcessor.from_pretrained",
            side_effect=OSError("model not found"),
        ):
            path = _write_vision_processor_config(
                config, str(tmp_path), hf_model_id="nonexistent/model"
            )

        assert path is not None
        with open(path) as f:
            data = json.load(f)

        proc = data["processor"]
        # Should use CLIP-standard normalization defaults
        normalize = proc["transforms"][3]["operation"]["attrs"]
        assert normalize["mean"] == pytest.approx([0.48145466, 0.4578275, 0.40821073])
        assert normalize["std"] == pytest.approx([0.26862954, 0.26130258, 0.27577711])


class TestFixTokenizerConfig:
    def test_remaps_tokenizers_backend(self, tmp_path):
        tc = {"tokenizer_class": "TokenizersBackend"}
        tc_path = tmp_path / "tokenizer_config.json"
        tc_path.write_text(json.dumps(tc))

        assert _fix_tokenizer_config(str(tmp_path)) is True

        fixed = json.loads(tc_path.read_text())
        assert fixed["tokenizer_class"] == "LlamaTokenizer"

    def test_no_fix_needed(self, tmp_path):
        tc = {"tokenizer_class": "LlamaTokenizer"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
        assert _fix_tokenizer_config(str(tmp_path)) is False

    def test_no_tokenizer_config(self, tmp_path):
        assert _fix_tokenizer_config(str(tmp_path)) is False


class TestFixChatTemplate:
    def test_adds_chat_template(self, tmp_path):
        """Adds chat_template from HF tokenizer when missing."""
        tc = {"tokenizer_class": "LlamaTokenizer"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        fake_tokenizer = mock.MagicMock()
        fake_tokenizer.chat_template = "{{ bos_token }}"

        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ):
            result = _fix_chat_template(str(tmp_path), "fake/model")

        assert result is True
        fixed = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert fixed["chat_template"] == "{{ bos_token }}"

    def test_skips_when_template_exists(self, tmp_path):
        tc = {
            "tokenizer_class": "LlamaTokenizer",
            "chat_template": "existing",
        }
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
        assert _fix_chat_template(str(tmp_path), "fake/model") is False

    def test_skips_without_model_id(self, tmp_path):
        tc = {"tokenizer_class": "LlamaTokenizer"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
        assert _fix_chat_template(str(tmp_path), None) is False

    def test_skips_without_file(self, tmp_path):
        assert _fix_chat_template(str(tmp_path), "fake/model") is False

    def test_audio_no_audio_returns_none(self, tmp_path):
        config = mock.MagicMock(spec=[])
        del config.audio
        assert _write_audio_processor_config(config, str(tmp_path)) is None

    def test_audio_non_gemma4_returns_none(self, tmp_path):
        config = mock.MagicMock()
        config.audio = mock.MagicMock()
        config.model_type = "whisper"
        assert _write_audio_processor_config(config, str(tmp_path)) is None

    def test_audio_gemma4_unified_skips_audio_processor(self, tmp_path):
        """Encoder-free gemma4_unified has no native transform: no audio_processor.json."""
        config = mock.MagicMock()
        config.audio = mock.MagicMock()
        config.model_type = "gemma4_unified"
        assert _write_audio_processor_config(config, str(tmp_path)) is None

    def test_audio_gemma4_writes_feature_extraction_json(self, tmp_path):
        config = mock.MagicMock()
        config.audio = mock.MagicMock()
        config.model_type = "gemma4"

        path = _write_audio_processor_config(config, str(tmp_path))
        assert path is not None
        assert path.endswith("audio_feature_extraction.json")
        with open(path) as f:
            data = json.load(f)

        # Must use feature_extraction.sequence, not processor.transforms
        assert "feature_extraction" in data
        seq = data["feature_extraction"]["sequence"]
        assert len(seq) == 2
        assert seq[0]["operation"]["type"] == "AudioDecoder"
        assert seq[1]["operation"]["type"] == "Gemma4LogMel"
        attrs = seq[1]["operation"]["attrs"]
        assert attrs["feature_size"] == 128
        assert attrs["sampling_rate"] == 16000
        assert attrs["frame_length_ms"] == 20.0  # noqa: RUF069
        assert attrs["hop_length_ms"] == 10.0  # noqa: RUF069
        assert attrs["mel_floor"] == 0.001  # noqa: RUF069

    def test_handles_tokenizer_load_error(self, tmp_path):
        """Gracefully handles AutoTokenizer.from_pretrained raising."""
        tc = {"tokenizer_class": "LlamaTokenizer"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("model not available"),
        ):
            result = _fix_chat_template(str(tmp_path), "fake/model")

        assert result is False
        # Original config is unchanged
        fixed = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert "chat_template" not in fixed


class TestCopyTokenizerFiles:
    def test_copies_available_files(self, tmp_path):
        # Create a fake tokenizer file to "download"
        fake_src = tmp_path / "src"
        fake_src.mkdir()
        (fake_src / "tokenizer.json").write_text('{"test": true}')
        (fake_src / "chat_template.jinja").write_text("{{ messages }}")

        with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
            mock_dl.side_effect = lambda model_id, filename: (
                str(fake_src / filename)
                if (fake_src / filename).exists()
                else (_ for _ in ()).throw(OSError("not found"))
            )

            dst = tmp_path / "output"
            dst.mkdir()
            copied = _copy_tokenizer_files("fake/model", str(dst))

        assert "tokenizer.json" in copied
        assert (dst / "tokenizer.json").exists()
        assert "chat_template.jinja" in copied
        assert (dst / "chat_template.jinja").exists()


class TestCopyTokenizerFilesFromLocal:
    """Tests for _copy_tokenizer_files_from_local."""

    def test_copies_present_files(self, tmp_path):
        """Copies tokenizer files that exist in the source directory."""
        src = tmp_path / "model"
        src.mkdir()
        (src / "tokenizer.json").write_text('{"test": true}')
        (src / "tokenizer_config.json").write_text('{"model_type": "llama"}')
        (src / "chat_template.jinja").write_text("{{ messages }}")

        dst = tmp_path / "output"
        dst.mkdir()
        copied = _copy_tokenizer_files_from_local(str(src), str(dst))

        assert set(copied) == {
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        }
        assert (dst / "tokenizer.json").read_text() == '{"test": true}'
        assert (dst / "chat_template.jinja").read_text() == "{{ messages }}"

    def test_skips_absent_files(self, tmp_path):
        """Files not present in the source directory are silently skipped."""
        src = tmp_path / "model"
        src.mkdir()
        # Only tokenizer.json present — tokenizer.model (SentencePiece), etc. absent

        (src / "tokenizer.json").write_text("{}")

        dst = tmp_path / "output"
        dst.mkdir()
        copied = _copy_tokenizer_files_from_local(str(src), str(dst))

        assert copied == ["tokenizer.json"]
        assert not (dst / "tokenizer.model").exists()

    def test_empty_source_returns_empty_list(self, tmp_path):
        """No tokenizer files in source returns an empty list."""
        src = tmp_path / "model"
        src.mkdir()
        dst = tmp_path / "output"
        dst.mkdir()
        copied = _copy_tokenizer_files_from_local(str(src), str(dst))
        assert copied == []

    def test_missing_source_dir_warns_and_returns_empty(self, tmp_path, caplog):
        """Non-existent source_dir emits a warning and returns empty list."""
        import logging

        dst = tmp_path / "output"
        dst.mkdir()
        with caplog.at_level(
            logging.WARNING, logger="mobius.integrations.ort_genai.auto_export"
        ):
            copied = _copy_tokenizer_files_from_local(str(tmp_path / "nonexistent"), str(dst))
        assert copied == []
        assert "does not exist" in caplog.text


class TestWriteOrtGenaiConfigLocalDir:
    """Tests for write_ort_genai_config with local_config_dir."""

    @staticmethod
    def _make_pkg():
        return _make_fake_llm_pkg("llama")

    def test_local_config_dir_copies_tokenizer_files(self, tmp_path):
        """When local_config_dir is set, tokenizer files are copied from it."""
        src = tmp_path / "local_model"
        src.mkdir()
        (src / "tokenizer.json").write_text('{"local": true}')
        (src / "tokenizer_config.json").write_text('{"type": "llama"}')

        out = tmp_path / "output"
        out.mkdir()
        pkg = self._make_pkg()

        result = write_ort_genai_config(
            pkg,
            str(out),
            local_config_dir=str(src),
        )

        assert "tokenizer.json" in result
        assert (out / "tokenizer.json").read_text() == '{"local": true}'
        assert "tokenizer_config.json" in result

    def test_hf_model_id_takes_precedence_over_local_dir(self, tmp_path):
        """When both hf_model_id and local_config_dir are set, HF takes precedence."""
        src = tmp_path / "local_model"
        src.mkdir()
        (src / "tokenizer.json").write_text('{"local": true}')

        out = tmp_path / "output"
        out.mkdir()
        pkg = self._make_pkg()

        with (
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files",
                return_value=["tokenizer.json"],
            ) as mock_hf,
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files_from_local",
                return_value=[],
            ) as mock_local,
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock.MagicMock(
                    model_type="llama",
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                ),
            ),
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._load_generation_config",
                return_value=None,
            ),
        ):
            (out / "tokenizer.json").write_text("{}")  # pretend HF copy happened
            write_ort_genai_config(
                pkg,
                str(out),
                hf_model_id="meta-llama/Llama-3-8B",
                local_config_dir=str(src),
            )

        mock_hf.assert_called_once()
        mock_local.assert_not_called()

    def test_local_hf_model_id_uses_local_tokenizer_copy(self, tmp_path):
        """A local hf_model_id should copy tokenizer files locally, not call the Hub."""
        src = tmp_path / "local_model"
        src.mkdir()
        (src / "config.json").write_text(
            '{"model_type": "llama", "bos_token_id": 1, "eos_token_id": 2}'
        )
        (src / "tokenizer.json").write_text('{"local": true}')

        out = tmp_path / "output"
        out.mkdir()
        pkg = self._make_pkg()

        with (
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files",
                return_value=[],
            ) as mock_hub_copy,
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files_from_local",
                wraps=_copy_tokenizer_files_from_local,
            ) as mock_local_copy,
        ):
            result = write_ort_genai_config(pkg, str(out), hf_model_id=str(src))

        mock_hub_copy.assert_not_called()
        mock_local_copy.assert_called_once_with(str(src), str(out))
        assert "tokenizer.json" in result
        assert (out / "tokenizer.json").read_text() == '{"local": true}'


class TestExportForOrtGenai:
    """Unit tests for write_ort_genai_config()."""

    @staticmethod
    def _make_pkg():
        return _make_fake_llm_pkg("qwen2")

    def test_genai_config_json_is_written(self, tmp_path):
        """genai_config.json is always written to the output directory."""
        pkg = self._make_pkg()
        result = write_ort_genai_config(pkg, str(tmp_path))

        assert "genai_config" in result
        assert os.path.isfile(result["genai_config"])
        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert "model" in data
        assert data["model"]["type"] == "qwen2"

    def test_rejects_generic_vision_encoder_decoder_package(self, tmp_path):
        import dataclasses

        from mobius._model_package import ModelPackage

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "nemotron_parse"

        pkg = ModelPackage(
            {
                "vision_encoder": _mock_model(),
                "decoder": _mock_model(),
            },
            config=FakeConfig(),
        )
        with pytest.raises(
            NotImplementedError,
            match="does not support generic vision encoder-decoder",
        ):
            write_ort_genai_config(pkg, str(tmp_path))
        assert not (tmp_path / "genai_config.json").exists()

    def test_processor_config_written_with_vision(self, tmp_path):
        """image_processor.json is written when pkg.config.vision is set."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 448
            patch_size: int = 14
            spatial_merge_size: int = 2
            model_type: str | None = None

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "qwen2"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            spatial_merge_size: int = 2
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        pkg = ModelPackage(
            {
                "model": _mock_model(),
                "vision_encoder": _mock_model(),
                "embedding": _mock_model(),
            },
            config=FakeConfig(),
        )
        result = write_ort_genai_config(pkg, str(tmp_path), ep="cuda")

        assert "processor_config" in result
        assert os.path.isfile(result["processor_config"])
        with open(result["processor_config"]) as f:
            data = json.load(f)
        # New format: transform pipeline under "processor"
        assert "processor" in data
        transforms = data["processor"]["transforms"]
        assert len(transforms) >= 4
        # Verify resize uses config values
        resize = transforms[1]["operation"]["attrs"]
        assert resize["patch_size"] == 14

    def test_qwen3_vl_writes_qwen3_vl_model_type_and_vision_fields(self, tmp_path):
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 448
            patch_size: int = 16
            spatial_merge_size: int = 2
            window_size: int = 64
            model_type: str | None = None

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "qwen3_vl"
            vocab_size: int = 151936
            hidden_size: int = 2048
            num_hidden_layers: int = 1
            num_attention_heads: int = 16
            num_key_value_heads: int = 8
            head_dim: int = 128
            image_token_id: int = 151655
            vision_start_token_id: int = 151652
            video_token_id: int = 151656
            tokens_per_second: float = 2.0
            temporal_patch_size: int = 2
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        pkg = ModelPackage(
            {
                "decoder": _mock_model(),
                "vision_encoder": _mock_model(),
                "embedding": _mock_model(),
            },
            config=FakeConfig(),
        )

        result = write_ort_genai_config(pkg, str(tmp_path), ep="cuda")

        with open(result["genai_config"]) as f:
            data = json.load(f)
        model = data["model"]
        assert model["type"] == "qwen3_vl"
        assert model["vision_start_token_id"] == 151652
        assert model["video_token_id"] == 151656
        assert model["vision"]["tokens_per_second"] == pytest.approx(2.0)
        assert model["vision"]["patch_size"] == 16
        assert model["vision"]["window_size"] == 64

    def test_glm_ocr_writes_qwen_runtime_and_packed_processor_contract(self, tmp_path):
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 336
            patch_size: int = 14
            spatial_merge_size: int = 2
            model_type: str = "glm_ocr_vision"

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "glm_ocr_text"
            vocab_size: int = 59392
            hidden_size: int = 1536
            num_hidden_layers: int = 16
            num_attention_heads: int = 16
            num_key_value_heads: int = 8
            head_dim: int = 128
            image_token_id: int = 59280
            vision_start_token_id: int = 59256
            temporal_patch_size: int = 2
            spatial_merge_size: int = 2
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        pkg = ModelPackage(
            {
                "decoder": _mock_model(
                    inputs=["inputs_embeds", "attention_mask", "position_ids"],
                    outputs=["logits"],
                ),
                "vision_encoder": _mock_model(
                    inputs=["pixel_values", "image_grid_thw"],
                    outputs=["image_features"],
                ),
                "embedding": _mock_model(
                    inputs=["input_ids", "image_features"],
                    outputs=["inputs_embeds"],
                ),
            },
            config=FakeConfig(),
        )

        result = write_ort_genai_config(pkg, str(tmp_path), ep="cuda")

        with open(result["genai_config"]) as f:
            model = json.load(f)["model"]
        assert model["type"] == "qwen2_5_vl"
        assert model["image_token_id"] == 59280
        assert model["vision_start_token_id"] == 59256
        assert model["vision"]["spatial_merge_size"] == 2
        assert model["vision"]["config_filename"] == "processor_config.json"
        assert model["vision"]["inputs"] == {
            "pixel_values": "pixel_values",
            "image_grid_thw": "image_grid_thw",
        }

        with open(result["processor_config"]) as f:
            processor = json.load(f)["processor"]
        assert processor["name"] == "qwen2_5_image_processor"
        patch_image = processor["transforms"][-1]["operation"]
        assert patch_image == {
            "name": "patch_image",
            "type": "PatchImage",
            "attrs": {
                "patch_size": 14,
                "temporal_patch_size": 2,
                "merge_size": 2,
            },
        }

    def test_glm_ocr_pixel_bounds_are_not_used_as_image_dimensions(self, tmp_path):
        """GLM-OCR's longest_edge is a pixel-count ceiling, not a side length."""
        import dataclasses

        from mobius.integrations.ort_genai.auto_export import (
            _write_vision_processor_config,
        )

        revision = "ca5d8b3e287e52589e37c28385d9655ee4372f9d"

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 336
            patch_size: int = 14
            spatial_merge_size: int = 2

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "glm_ocr_text"
            temporal_patch_size: int = 2
            spatial_merge_size: int = 2
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        image_processor = mock.MagicMock()
        image_processor.image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_processor.image_std = [0.26862954, 0.26130258, 0.27577711]
        image_processor.rescale_factor = 1.0 / 255.0
        image_processor.resample = 3
        image_processor.size = {
            "shortest_edge": 12544,
            "longest_edge": 9633792,
        }
        hf_processor = mock.MagicMock(image_processor=image_processor)

        with mock.patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=hf_processor,
        ) as mock_from_pretrained:
            path = _write_vision_processor_config(
                FakeConfig(),
                str(tmp_path),
                hf_model_id="zai-org/GLM-OCR",
                revision=revision,
            )

        mock_from_pretrained.assert_called_once_with(
            "zai-org/GLM-OCR",
            revision=revision,
            trust_remote_code=False,
        )
        assert path is not None
        with open(path) as f:
            transforms = json.load(f)["processor"]["transforms"]
        resize = transforms[1]["operation"]["attrs"]
        assert resize["height"] == 336
        assert resize["width"] == 336
        assert resize["min_pixels"] == 12544
        assert resize["max_pixels"] == 9633792

    def test_mage_vl_is_rejected_before_writing_runtime_artifacts(self, tmp_path):
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 448
            patch_size: int = 16
            spatial_merge_size: int = 2

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "mage_vl"
            vocab_size: int = 151936
            hidden_size: int = 2560
            num_hidden_layers: int = 1
            num_attention_heads: int = 32
            num_key_value_heads: int = 8
            head_dim: int = 128
            image_token_id: int = 151655
            temporal_patch_size: int = 1
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        pkg = ModelPackage(
            {
                "decoder": _mock_model(),
                "vision_encoder": _mock_model(),
                "embedding": _mock_model(),
            },
            config=FakeConfig(),
        )

        output_dir = tmp_path / "ort-genai"
        with pytest.raises(
            ValueError,
            match=r"Mage-VL.*patch_positions.*1D decoder position_ids",
        ):
            write_ort_genai_config(pkg, str(output_dir))
        assert not output_dir.exists()

    def test_processor_config_not_written_without_vision(self, tmp_path):
        """image_processor.json is NOT written when pkg.config has no vision attr."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        result = write_ort_genai_config(pkg, str(tmp_path))

        assert "processor_config" not in result
        assert not os.path.exists(os.path.join(str(tmp_path), "processor_config.json"))

    def test_gemma4_image_processor_json_written(self, tmp_path):
        """Gemma4 writes image_processor.json with onnxruntime-extensions transforms pipeline."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 448
            patch_size: int = 16
            mm_tokens_per_image: int = 260
            pooling_kernel_size: int = 3

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma4"
            vocab_size: int = 262144
            hidden_size: int = 1536
            num_hidden_layers: int = 35
            num_attention_heads: int = 8
            num_key_value_heads: int = 1
            head_dim: int = 256
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        pkg = ModelPackage(
            {
                "model": _mock_model(),
                "vision": _mock_model(),
                "embedding": _mock_model(),
            },
            config=FakeConfig(),
        )
        result = write_ort_genai_config(pkg, str(tmp_path))

        # Should write image_processor.json, not processor_config.json
        assert "processor_config" in result
        proc_path = result["processor_config"]
        assert proc_path.endswith("image_processor.json")
        assert os.path.isfile(proc_path)
        assert not os.path.exists(os.path.join(str(tmp_path), "processor_config.json"))

        with open(proc_path) as f:
            data = json.load(f)

        # Verify onnxruntime-extensions transforms pipeline structure
        assert "processor" in data
        assert "transforms" in data["processor"]
        transforms = data["processor"]["transforms"]
        assert len(transforms) == 2

        # First op: DecodeImage
        op0 = transforms[0]["operation"]
        assert op0["type"] == "DecodeImage"
        assert op0["attrs"]["color_space"] == "RGB"

        # Second op: Gemma4ImageTransform with correct attrs from config
        op1 = transforms[1]["operation"]
        assert op1["type"] == "Gemma4ImageTransform"
        assert op1["attrs"]["patch_size"] == 16
        assert op1["attrs"]["max_soft_tokens"] == 260
        assert op1["attrs"]["pooling_kernel_size"] == 3

    def test_gemma4_audio_processor_json_written(self, tmp_path):
        """Gemma4 writes audio_feature_extraction.json with feature_extraction.sequence schema."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeAudio:
            feature_size: int = 128

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma4"
            vocab_size: int = 262144
            hidden_size: int = 1536
            num_hidden_layers: int = 35
            num_attention_heads: int = 8
            num_key_value_heads: int = 1
            head_dim: int = 256
            audio: FakeAudio = dataclasses.field(default_factory=FakeAudio)

        pkg = ModelPackage(
            {
                "model": _mock_model(),
                "audio_encoder": _mock_model(),
                "embedding": _mock_model(),
            },
            config=FakeConfig(),
        )
        result = write_ort_genai_config(pkg, str(tmp_path))

        # Should write audio_feature_extraction.json
        assert "audio_processor" in result
        audio_path = result["audio_processor"]
        assert audio_path.endswith("audio_feature_extraction.json")
        assert os.path.isfile(audio_path)

        with open(audio_path) as f:
            data = json.load(f)

        # Verify feature_extraction.sequence schema (not processor.transforms)
        assert "feature_extraction" in data
        assert "processor" not in data
        seq = data["feature_extraction"]["sequence"]
        assert len(seq) == 2

        # First op: AudioDecoder
        op0 = seq[0]["operation"]
        assert op0["type"] == "AudioDecoder"

        # Second op: Gemma4LogMel with expected attrs
        op1 = seq[1]["operation"]
        assert op1["type"] == "Gemma4LogMel"
        assert op1["attrs"]["feature_size"] == 128
        assert op1["attrs"]["sampling_rate"] == 16000
        assert op1["attrs"]["mel_floor"] == 0.001  # noqa: RUF069

    def test_gemma4_audio_processor_not_written_without_audio(self, tmp_path):
        """No audio_feature_extraction.json when config has no audio attr."""
        import dataclasses

        from mobius._model_package import ModelPackage

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma4"
            vocab_size: int = 262144
            hidden_size: int = 1536
            num_hidden_layers: int = 35
            num_attention_heads: int = 8
            num_key_value_heads: int = 1
            head_dim: int = 256

        pkg = ModelPackage({"model": _mock_model()}, config=FakeConfig())
        result = write_ort_genai_config(pkg, str(tmp_path))

        assert "audio_processor" not in result
        assert not os.path.exists(os.path.join(str(tmp_path), "audio_feature_extraction.json"))

    def test_gemma4_genai_config_speech_section(self, tmp_path):
        """Gemma4 with audio_encoder writes speech section with correct config_filename and input mapping."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeAudio:
            feature_size: int = 128
            audio_token_id: int = 255999

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma4"
            vocab_size: int = 262144
            hidden_size: int = 1536
            num_hidden_layers: int = 35
            num_attention_heads: int = 8
            num_key_value_heads: int = 1
            head_dim: int = 256
            boa_token_id: int = 255998
            audio: FakeAudio = dataclasses.field(default_factory=FakeAudio)

        pkg = ModelPackage(
            {
                "model": _mock_model(),
                "audio_encoder": _mock_model(),
                "embedding": _mock_model(),
            },
            config=FakeConfig(),
        )
        result = write_ort_genai_config(pkg, str(tmp_path))

        with open(result["genai_config"]) as f:
            data = json.load(f)

        # Speech section must exist
        assert "speech" in data["model"], (
            "genai_config.json should have a speech section for Gemma4 with audio_encoder"
        )
        speech = data["model"]["speech"]

        # config_filename must point to the onnxruntime-extensions audio config
        assert speech["config_filename"] == "audio_feature_extraction.json"

        # filename must point to the audio encoder ONNX model
        assert speech["filename"] == "audio_encoder/model.onnx"

        # input_names mapping: genai internal name -> ONNX model input name
        assert speech["inputs"]["audio_embeds"] == "input_features"
        assert speech["inputs"]["attention_mask"] == "input_features_mask"

    def test_tokenizer_not_copied_without_model_id(self, tmp_path):
        """No tokenizer files copied when hf_model_id=None."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        with mock.patch(
            "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files"
        ) as mock_copy:
            write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)
        mock_copy.assert_not_called()

    def test_tokenizer_copied_when_model_id_provided(self, tmp_path):
        """Tokenizer files are copied when hf_model_id is provided."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        with (
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files",
                return_value=["tokenizer.json"],
            ) as mock_copy,
            mock.patch("transformers.AutoConfig.from_pretrained") as mock_hf,
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._load_generation_config",
                return_value=None,
            ),
        ):
            mock_hf.return_value = mock.MagicMock(
                model_type="qwen2", bos_token_id=1, eos_token_id=2, pad_token_id=0
            )
            result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id="fake/model")

        mock_copy.assert_called_once_with("fake/model", str(tmp_path))
        assert "tokenizer.json" in result

    def test_hf_config_propagates_trust_remote_code(self, tmp_path):
        """Remote-code models can resolve their HuggingFace configuration."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        with (
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files",
                return_value=[],
            ),
            mock.patch("transformers.AutoConfig.from_pretrained") as mock_hf,
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._load_generation_config",
                return_value=None,
            ),
        ):
            mock_hf.return_value = mock.MagicMock(
                model_type="mage_vl", bos_token_id=1, eos_token_id=2, pad_token_id=0
            )
            write_ort_genai_config(
                pkg,
                str(tmp_path),
                hf_model_id="microsoft/Mage-VL",
                trust_remote_code=True,
            )

        mock_hf.assert_called_once_with("microsoft/Mage-VL", trust_remote_code=True)

    def test_generation_config_multi_eos_overrides_model_config(self, tmp_path):
        """generation_config.json stop tokens take precedence over the model config."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        hf_config = mock.MagicMock(
            model_type="qwen3_5_moe",
            bos_token_id=248044,
            eos_token_id=248044,
            pad_token_id=None,
        )
        generation_config = mock.MagicMock(
            bos_token_id=248044,
            eos_token_id=[248046, 248044],
            pad_token_id=248044,
        )
        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=hf_config),
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._load_generation_config",
                return_value=generation_config,
            ),
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files",
                return_value=[],
            ),
        ):
            result = write_ort_genai_config(
                pkg,
                str(tmp_path),
                hf_model_id="Qwen/Qwen3.6-35B-A3B",
            )

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["eos_token_id"] == [248046, 248044]
        assert data["model"]["pad_token_id"] == 248044

    def test_hf_revision_is_used_for_config_and_tokenizer_assets(self, tmp_path):
        """A pinned export never mixes config and tokenizer revisions."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        revision = "0123456789abcdef"
        with (
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files",
                return_value=[],
            ) as mock_copy,
            mock.patch("transformers.AutoConfig.from_pretrained") as mock_hf,
        ):
            mock_hf.return_value = mock.MagicMock(
                model_type="qwen2", bos_token_id=1, eos_token_id=2, pad_token_id=0
            )
            write_ort_genai_config(
                pkg,
                str(tmp_path),
                hf_model_id="fake/model",
                revision=revision,
            )

        mock_hf.assert_called_once_with(
            "fake/model",
            revision=revision,
            trust_remote_code=False,
        )
        mock_copy.assert_called_once_with(
            "fake/model",
            str(tmp_path),
            revision=revision,
        )

    def test_ep_default_normalizes_to_cpu(self, tmp_path):
        """ep='default' is normalized to cpu (provider_options=[])."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        result = write_ort_genai_config(pkg, str(tmp_path), ep="default")

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["decoder"]["session_options"]["provider_options"] == []

    def test_ep_onnx_standard_normalizes_to_cpu(self, tmp_path):
        """ep='onnx-standard' is normalized to cpu (provider_options=[])."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        result = write_ort_genai_config(pkg, str(tmp_path), ep="onnx-standard")

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["decoder"]["session_options"]["provider_options"] == []

    def test_ep_cuda_passes_through(self, tmp_path):
        """ep='cuda' passes through to session_options with CUDA provider."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        result = write_ort_genai_config(pkg, str(tmp_path), ep="cuda")

        with open(result["genai_config"]) as f:
            data = json.load(f)
        provider_opts = data["model"]["decoder"]["session_options"]["provider_options"]
        assert len(provider_opts) == 1
        assert "cuda" in provider_opts[0]

    def test_raises_when_pkg_config_is_none(self, tmp_path):
        """ValueError is raised when pkg.config is None."""
        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = ModelPackage({"model": _mock_model()}, config=None)
        with pytest.raises(ValueError, match="config"):
            write_ort_genai_config(pkg, str(tmp_path))

    def test_does_not_require_onnxruntime_genai(self, tmp_path):
        """write_ort_genai_config works without onnxruntime-genai installed."""
        import sys

        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        # Remove onnxruntime_genai from sys.modules if present, then restore
        saved = sys.modules.pop("onnxruntime_genai", None)
        try:
            # Should not raise ImportError — ort-genai runtime is not needed
            result = write_ort_genai_config(pkg, str(tmp_path))
            assert "genai_config" in result
        finally:
            if saved is not None:
                sys.modules["onnxruntime_genai"] = saved

    def test_config_mode_model_type_propagated(self, tmp_path):
        """model.type in genai_config.json is correct when hf_model_id=None."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeConfig:
            # model_type now stored on ArchitectureConfig for --config mode
            model_type: str = "gemma2"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128

        pkg = ModelPackage({"model": _mock_model()}, config=FakeConfig())
        result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)

        with open(result["genai_config"]) as f:
            data = json.load(f)
        # "gemma2" maps to "gemma" in _ORT_GENAI_MODEL_TYPE
        assert data["model"]["type"] == "gemma"

    def test_config_mode_gemma3_text_vlm_uses_multimodal_model_type(self, tmp_path):
        """Gemma3 VLM --config exports use ORT's multimodal gemma3 type."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 896
            patch_size: int = 14
            spatial_merge_size: int = 2
            model_type: str = "siglip_vision_model"

        @dataclasses.dataclass
        class FakeConfig:
            # build() stores the unwrapped text sub-config type on Gemma3 VLMs.
            model_type: str = "gemma3_text"
            vocab_size: int = 262144
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128
            image_token_id: int = 255999
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        pkg = ModelPackage(
            {
                "decoder": _mock_model_with_inputs(["inputs_embeds", "attention_mask"]),
                "vision_encoder": _mock_model_with_inputs(["pixel_values"]),
                "embedding": _mock_model_with_inputs(["input_ids", "image_features"]),
            },
            config=FakeConfig(),
        )
        result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["type"] == "gemma3"

    def test_config_mode_gemma3n_text_vlm_uses_multimodal_model_type(self, tmp_path):
        """Gemma3n unwraps to "gemma3n_text" too, and must not alias to gemma3.

        Its package threads ``per_layer_inputs`` (and optional audio) that
        gemma3's ORT pipeline does not bind, so borrowing that type would
        mis-wire the graph.
        """
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 768
            model_type: str = "gemma3n_vision"

        @dataclasses.dataclass
        class FakeConfig:
            # build() stores the unwrapped text sub-config type on Gemma3n VLMs.
            model_type: str = "gemma3n_text"
            vocab_size: int = 262400
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128
            image_token_id: int = 262145
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        pkg = ModelPackage(
            {
                "decoder": _mock_model_with_inputs(
                    ["inputs_embeds", "attention_mask", "per_layer_inputs"]
                ),
                "vision_encoder": _mock_model_with_inputs(["pixel_values"]),
                "embedding": _mock_model_with_inputs(["input_ids", "image_features"]),
            },
            config=FakeConfig(),
        )
        result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["type"] == "gemma3n"

    def test_gemma3n_references_only_processor_files_that_exist(self, tmp_path):
        """Every processor file genai_config.json names must be on disk.

        Gemma3n is the first model to reach the ``elif has_speech`` vision
        branch, which sets no ``config_filename``; it used to inherit
        ``with_vision``'s "image_processor.json" default (written only for
        gemma4) while ``_write_vision_processor_config`` wrote
        processor_config.json. The audio side had no writer at all, so
        ``with_audio``'s "audio_processor.json" default dangled too. Both
        references pointed at files that were never created.

        Vision and audio are deliberately two different files: ORT-GenAI loads
        them via ``OrtxCreateProcessor`` and
        ``OrtxCreateSpeechFeatureExtractor`` respectively, which parse
        different schemas. Dropping either reference is not an option — the
        runtime throws when ``speech.filename`` is set without
        ``speech.config_filename``.
        """
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 768
            model_type: str = "gemma3n_vision"

        @dataclasses.dataclass
        class FakeAudio:
            model_type: str = "gemma3n_audio"
            input_feat_size: int = 128

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma3n_text"
            vocab_size: int = 262400
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128
            image_token_id: int = 262145
            audio_token_id: int = 262273
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)
            audio: FakeAudio = dataclasses.field(default_factory=FakeAudio)

        pkg = ModelPackage(
            {
                "decoder": _mock_model_with_inputs(
                    ["inputs_embeds", "attention_mask", "per_layer_inputs"]
                ),
                "vision_encoder": _mock_model_with_inputs(["pixel_values"]),
                "audio_encoder": _mock_model_with_inputs(
                    ["input_features", "input_features_mask"]
                ),
                "embedding": _mock_model_with_inputs(
                    ["input_ids", "image_features", "audio_features"]
                ),
            },
            config=FakeConfig(),
        )
        result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)

        with open(result["genai_config"]) as f:
            model = json.load(f)["model"]

        # Each modality must name the file its own writer actually produced.
        assert model["vision"]["config_filename"] == "processor_config.json"
        assert model["speech"]["config_filename"] == "audio_feature_extraction.json"

        # Both must be on disk. Every processor reference is checked, so a new
        # section that names a file nothing writes fails here too.
        for section in ("vision", "speech"):
            named = model[section]["config_filename"]
            assert os.path.exists(os.path.join(str(tmp_path), named)), (
                f"genai_config.json model.{section} references {named!r}, "
                "which was never written"
            )

    def test_gemma3n_speech_inputs_use_runtime_schema_keys(self, tmp_path):
        """``model.speech.inputs`` keys are a closed set, not graph names.

        Unlike the decoder and vision sections — where the keys happen to equal
        the graph input names, so an introspected identity map works — ORT-GenAI
        defines exactly four accepted keys for the speech section
        (``audio_embeds``, ``attention_mask``, ``audio_sizes``,
        ``audio_projection_mode``) and maps each to a graph name. Passing
        ``_introspect_inputs``' identity map made the config unparseable:
        ``model:speech:inputs: Unknown value "input_features"``.
        """
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 768
            model_type: str = "gemma3n_vision"

        @dataclasses.dataclass
        class FakeAudio:
            model_type: str = "gemma3n_audio"
            input_feat_size: int = 128

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma3n_text"
            vocab_size: int = 262400
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128
            image_token_id: int = 262145
            audio_token_id: int = 262273
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)
            audio: FakeAudio = dataclasses.field(default_factory=FakeAudio)

        pkg = ModelPackage(
            {
                "decoder": _mock_model_with_inputs(
                    ["inputs_embeds", "attention_mask", "per_layer_inputs"]
                ),
                "vision_encoder": _mock_model_with_inputs(["pixel_values"]),
                "audio_encoder": _mock_model_with_inputs(
                    ["input_features", "input_features_mask"]
                ),
                "embedding": _mock_model_with_inputs(
                    ["input_ids", "image_features", "audio_features"]
                ),
            },
            config=FakeConfig(),
        )
        result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)

        with open(result["genai_config"]) as f:
            speech = json.load(f)["model"]["speech"]

        # Schema key -> graph name, same as the gemma4 branch above.
        assert speech["inputs"] == {
            "audio_embeds": "input_features",
            "attention_mask": "input_features_mask",
        }

        # No key may be a graph name the runtime doesn't recognise. This is the
        # assertion that fails if someone reintroduces the identity map.
        assert set(speech["inputs"]) <= {
            "audio_embeds",
            "attention_mask",
            "audio_sizes",
            "audio_projection_mode",
        }

    def test_gemma3n_audio_processor_uses_gemma3n_mel_params(self, tmp_path):
        """Gemma3n reuses gemma4's op but not its filterbank values.

        Gemma3nAudioFeatureExtractor is a different mel filterbank from
        gemma4's; copying gemma4's attrs would silently degrade transcription
        rather than fail. Values are from the E4B preprocessor_config.json.
        """
        import dataclasses

        @dataclasses.dataclass
        class FakeAudio:
            model_type: str = "gemma3n_audio"
            input_feat_size: int = 128

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma3n_text"
            audio: FakeAudio = dataclasses.field(default_factory=FakeAudio)

        path = _write_audio_processor_config(FakeConfig(), str(tmp_path))
        assert path is not None
        assert path.endswith("audio_feature_extraction.json")

        with open(path) as f:
            sequence = json.load(f)["feature_extraction"]["sequence"]

        # OrtxCreateSpeechFeatureExtractor requires the feature_extraction
        # .sequence schema, decoder first.
        assert sequence[0]["operation"]["type"] == "AudioDecoder"
        op = sequence[1]["operation"]
        assert op["type"] == "Gemma4LogMel"

        # frame_length 512 / hop_length 160 samples @ 16 kHz, in milliseconds.
        assert op["attrs"] == {
            "feature_size": 128,
            "sampling_rate": 16000,
            "frame_length_ms": 32.0,
            "hop_length_ms": 10.0,
            "min_frequency": 125.0,
            "max_frequency": 7600.0,
            "preemphasis": 0.97,
            "preemphasis_htk_flavor": 1,
            "fft_overdrive": 1,
            "mel_floor": 1e-05,
        }

    def test_config_mode_token_ids_propagated(self, tmp_path):
        """bos/eos token IDs in genai_config.json come from config fields in --config mode."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "llama"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128
            bos_token_id: int = 1
            eos_token_id: int = 2
            pad_token_id: int = 0

        pkg = ModelPackage({"model": _mock_model()}, config=FakeConfig())
        result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["bos_token_id"] == 1
        assert data["model"]["eos_token_id"] == 2
        assert data["model"]["pad_token_id"] == 0

    def test_config_mode_eos_token_id_as_list(self, tmp_path):
        """eos_token_id can be a list[int] (e.g. Gemma multi-stop tokens)."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma2"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128
            bos_token_id: int = 2
            eos_token_id: list = dataclasses.field(default_factory=lambda: [1, 106])
            pad_token_id: int = 0

        pkg = ModelPackage({"model": _mock_model()}, config=FakeConfig())
        result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["eos_token_id"] == [1, 106]


class TestExportPackage:
    """Tests for export_package() — the save+config integration helper."""

    @staticmethod
    def _make_pkg():
        return _make_fake_llm_pkg("qwen2")

    def test_writes_both_onnx_and_genai_config(self, tmp_path, monkeypatch):
        """export_package calls pkg.save AND writes genai_config.json."""
        from mobius.integrations.ort_genai.auto_export import export_package

        pkg = self._make_pkg()
        save_calls = []

        def fake_save(self, directory, **kwargs):
            save_calls.append((directory, kwargs))

        monkeypatch.setattr(pkg.__class__, "save", fake_save)

        result = export_package(pkg, str(tmp_path))

        # pkg.save called exactly once with the output dir
        assert len(save_calls) == 1
        assert save_calls[0][0] == str(tmp_path)
        # genai_config artifact is in the manifest
        assert "genai_config" in result
        assert os.path.isfile(result["genai_config"])
        # ONNX path is in the manifest (single-component package)
        assert result["model"] == os.path.join(str(tmp_path), "model.onnx")

    def test_mage_vl_is_rejected_before_saving_onnx(self, tmp_path):
        pkg = self._make_pkg()
        pkg.config.model_type = "mage_vl"

        with (
            mock.patch.object(pkg, "save") as save,
            pytest.raises(ValueError, match=r"Mage-VL.*patch_positions"),
        ):
            export_package(pkg, str(tmp_path))

        save.assert_not_called()

    def test_propagates_save_kwargs(self, tmp_path, monkeypatch):
        """external_data and progress_bar are forwarded to pkg.save."""
        from mobius.integrations.ort_genai.auto_export import export_package

        pkg = self._make_pkg()
        save_calls = []

        def fake_save(self, directory, **kwargs):
            save_calls.append(kwargs)

        monkeypatch.setattr(pkg.__class__, "save", fake_save)

        export_package(
            pkg,
            str(tmp_path),
            external_data="safetensors",
            progress_bar=False,
        )

        assert save_calls[0]["external_data"] == "safetensors"
        assert save_calls[0]["progress_bar"] is False

    def test_propagates_genai_config_kwargs(self, tmp_path, monkeypatch):
        """The ep and context_length kwargs reach the generated genai_config.json."""
        from mobius.integrations.ort_genai.auto_export import export_package

        pkg = self._make_pkg()
        monkeypatch.setattr(pkg.__class__, "save", lambda self, d, **kw: None)

        result = export_package(
            pkg,
            str(tmp_path),
            ep="cuda",
            context_length=8192,
        )

        with open(result["genai_config"]) as f:
            data = json.load(f)
        # ep="cuda" should produce CUDA provider_options
        provider_opts = data["model"]["decoder"]["session_options"]["provider_options"]
        assert any("cuda" in po for po in provider_opts)
        # context_length should bump max_length
        assert data["search"]["max_length"] == 8192

    def test_preflights_missing_config(self, tmp_path, monkeypatch):
        """Raises ValueError BEFORE writing any ONNX when pkg.config is None.

        This avoids leaving a half-exported directory with model.onnx but
        no genai_config.json.
        """
        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import export_package

        pkg = ModelPackage({"model": _mock_model()}, config=None)
        save_called = []

        def fake_save(self, *a, **kw):
            save_called.append(True)

        monkeypatch.setattr(pkg.__class__, "save", fake_save)

        with pytest.raises(ValueError, match="config"):
            export_package(pkg, str(tmp_path))

        # save was NOT called — preflight failed before any I/O
        assert save_called == []

    def test_returns_manifest_with_all_artifacts(self, tmp_path, monkeypatch):
        """Returned manifest contains ONNX paths AND config artifacts."""
        from mobius.integrations.ort_genai.auto_export import export_package

        pkg = self._make_pkg()
        monkeypatch.setattr(pkg.__class__, "save", lambda self, d, **kw: None)

        result = export_package(pkg, str(tmp_path))

        # Manifest is non-empty and includes both kinds of artifacts
        assert isinstance(result, dict)
        assert "model" in result
        assert "genai_config" in result


class TestGemma4GenaiConfig:
    """Tests for Gemma4-specific genai_config generation via graph introspection."""

    @staticmethod
    def _make_gemma4_pkg():
        """Build a mock Gemma4 VLM package with graph inputs."""
        import dataclasses

        from mobius._model_package import ModelPackage

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 448
            patch_size: int = 16
            mm_tokens_per_image: int = 256

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma4"
            vocab_size: int = 262144
            hidden_size: int = 2048
            num_hidden_layers: int = 26
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            head_dim: int = 256
            max_position_embeddings: int = 8192
            image_token_id: int = 255999
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        # Mock graph inputs for each sub-model
        decoder = _mock_model_with_inputs(
            [
                "inputs_embeds",
                "per_layer_inputs",
                "attention_mask",
                "position_ids",
                "past_key_values.0.key",
                "past_key_values.0.value",
            ]
        )
        vision = _mock_model_with_inputs(
            [
                "pixel_values",
                "pixel_position_ids",
            ]
        )
        embedding = _mock_model_with_inputs(
            [
                "input_ids",
                "image_features",
            ]
        )

        return ModelPackage(
            {
                "decoder": decoder,
                "vision_encoder": vision,
                "embedding": embedding,
            },
            config=FakeConfig(),
        )

    def test_gemma4_vision_inputs(self, tmp_path):
        """Gemma4 vision uses pixel_values + pixel_position_ids."""
        pkg = self._make_gemma4_pkg()
        path = _write_genai_config(
            pkg.config,
            str(tmp_path),
            pkg=pkg,
            ort_model_type="gemma4",
            ep="cpu",
            context_length=4096,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            is_vlm=True,
            has_speech=False,
        )
        with open(path) as f:
            data = json.load(f)
        vision_inputs = data["model"]["vision"]["inputs"]
        assert "pixel_values" in vision_inputs
        assert "pixel_position_ids" in vision_inputs
        assert "image_grid_thw" not in vision_inputs
        assert data["model"]["vision"]["spatial_merge_size"] == 2

    def test_gemma4_decoder_has_per_layer_inputs_and_inputs_embeds(self, tmp_path):
        """Gemma4 decoder has inputs_embeds and per_layer_inputs."""
        pkg = self._make_gemma4_pkg()
        path = _write_genai_config(
            pkg.config,
            str(tmp_path),
            pkg=pkg,
            ort_model_type="gemma4",
            ep="cpu",
            context_length=4096,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            is_vlm=True,
            has_speech=False,
        )
        with open(path) as f:
            data = json.load(f)
        decoder_inputs = data["model"]["decoder"]["inputs"]
        assert "inputs_embeds" in decoder_inputs
        assert "per_layer_inputs" in decoder_inputs
        assert "input_ids" not in decoder_inputs
        # KV cache templates are present
        assert decoder_inputs["past_key_names"] == "past_key_values.%d.key"

    def test_gemma4_embedding_inputs(self, tmp_path):
        """Gemma4 embedding inputs discovered from graph."""
        pkg = self._make_gemma4_pkg()
        path = _write_genai_config(
            pkg.config,
            str(tmp_path),
            pkg=pkg,
            ort_model_type="gemma4",
            ep="cpu",
            context_length=4096,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            is_vlm=True,
            has_speech=False,
        )
        with open(path) as f:
            data = json.load(f)
        emb_inputs = data["model"]["embedding"]["inputs"]
        assert "input_ids" in emb_inputs
        assert "image_features" in emb_inputs


class TestPixtralGenaiConfig:
    """Tests for Pixtral/Ministral-3-specific genai_config generation."""

    @staticmethod
    def _make_pixtral_pkg():
        """Build a mock Pixtral VLM package with graph inputs."""
        import dataclasses

        from mobius._model_package import ModelPackage

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 1540
            patch_size: int = 14
            spatial_merge_size: int = 2
            model_type: str = "pixtral"

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "mistral3"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128
            image_token_id: int = 10
            spatial_merge_size: int = 2
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        decoder = _mock_model_with_inputs(
            ["input_ids", "attention_mask", "past_key_values.0.key"]
        )
        vision = _mock_model_with_inputs(["pixel_values"])
        embedding = _mock_model_with_inputs(["input_ids", "image_features"])

        return ModelPackage(
            {
                "model": decoder,
                "vision_encoder": vision,
                "embedding": embedding,
            },
            config=FakeConfig(),
        )

    def test_pixtral_config_filename_is_processor_config(self, tmp_path):
        """Pixtral genai_config.json references processor_config.json, not image_processor.json."""
        pkg = self._make_pixtral_pkg()
        path = _write_genai_config(
            pkg.config,
            str(tmp_path),
            pkg=pkg,
            ort_model_type="mistral3",
            ep="cpu",
            context_length=4096,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            is_vlm=True,
            has_speech=False,
        )
        with open(path) as f:
            data = json.load(f)
        vision = data["model"]["vision"]
        assert vision["config_filename"] == "processor_config.json"
        assert vision["spatial_merge_size"] == 2
        assert data["model"]["image_token_id"] == 10


class TestHybridAttentionShareBufferGuard:
    """Tests for the LinearAttention/GQA past_present_share_buffer guard.

    See the comment above ``supports_in_place_kv_cache`` in
    ``_write_genai_config``: recurrent-state layers (LinearAttention)
    mandate ``past_present_share_buffer=True``, but standard (non-GQA)
    Attention is incompatible with it. A hybrid graph with both, and no
    GQA node to lower the standard Attention layers to, must raise a clear
    build-time error rather than silently emit a broken config.
    """

    @staticmethod
    def _make_pkg(node_op_types: list[tuple[str, str]]):
        """Build a fake decoder pkg whose graph has the given (op_type, domain) nodes."""
        import dataclasses

        from mobius._model_package import ModelPackage

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "qwen35_moe"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128

        nodes = [
            ir.Node(op_type=op_type, domain=domain, inputs=[], num_outputs=1)
            for op_type, domain in node_op_types
        ]
        graph = ir.Graph(
            inputs=[ir.Value(name="input_ids")],
            outputs=[ir.Value(name="logits")],
            nodes=nodes,
            name="decoder",
        )
        decoder = ir.Model(graph, ir_version=10)
        return ModelPackage({"model": decoder}, config=FakeConfig())

    def _write(self, pkg, tmp_path):
        return _write_genai_config(
            pkg.config,
            str(tmp_path),
            pkg=pkg,
            ort_model_type="qwen35_moe",
            ep="cpu",
            context_length=4096,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            is_vlm=False,
            has_speech=False,
        )

    def test_recurrent_state_with_standard_attention_and_no_gqa_raises(self, tmp_path):
        """LinearAttention + standard Attention + no GQA is an unrunnable config."""
        pkg = self._make_pkg(
            [("LinearAttention", "com.microsoft"), ("Attention", "")],
        )
        with pytest.raises(ValueError, match="past_present_share_buffer"):
            self._write(pkg, tmp_path)

    def test_recurrent_state_with_gqa_does_not_raise(self, tmp_path):
        """LinearAttention + GQA (no standard Attention) is a valid hybrid config."""
        pkg = self._make_pkg(
            [
                ("LinearAttention", "com.microsoft"),
                ("GroupQueryAttention", "com.microsoft"),
            ],
        )
        path = self._write(pkg, tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["search"]["past_present_share_buffer"] is True

    def test_recurrent_state_with_standard_attention_and_gqa_raises(self, tmp_path):
        """Partial GQA fusion still leaves an incompatible standard Attention node.

        Regression test: the guard previously read
        ``has_recurrent_state and has_standard_attention and not has_gqa``, so
        a GQA node present *anywhere* in the graph would short-circuit the
        check even though a separate, unfused standard Attention node
        coexists. A GQA node on one layer doesn't make a standard Attention
        node on another layer safe for ``past_present_share_buffer=True``.
        """
        pkg = self._make_pkg(
            [
                ("LinearAttention", "com.microsoft"),
                ("GroupQueryAttention", "com.microsoft"),
                ("Attention", ""),
            ],
        )
        with pytest.raises(ValueError, match="past_present_share_buffer"):
            self._write(pkg, tmp_path)

    def test_recurrent_state_only_does_not_raise(self, tmp_path):
        """LinearAttention with no full-attention layers at all is unaffected."""
        pkg = self._make_pkg([("LinearAttention", "com.microsoft")])
        path = self._write(pkg, tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["search"]["past_present_share_buffer"] is True


class TestGraphInputNames:
    """Tests for _graph_input_names() helper."""

    def test_filters_kv_cache_inputs(self):
        """KV cache inputs (past_key_values.*) are filtered out."""
        model = _mock_model_with_inputs(
            [
                "input_ids",
                "attention_mask",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "past_key_values.1.key",
                "past_key_values.1.value",
            ]
        )
        result = _graph_input_names(model)
        assert result == ["input_ids", "attention_mask"]

    def test_filters_past_prefix(self):
        """Inputs starting with 'past_' are also filtered out."""
        model = _mock_model_with_inputs(
            [
                "input_ids",
                "past_something",
            ]
        )
        result = _graph_input_names(model)
        assert result == ["input_ids"]

    def test_skips_none_names(self):
        """Inputs with name=None are skipped."""
        inp_good = mock.MagicMock()
        inp_good.name = "input_ids"
        inp_none = mock.MagicMock()
        inp_none.name = None
        m = mock.MagicMock()
        m.graph.inputs = [inp_good, inp_none]
        result = _graph_input_names(m)
        assert result == ["input_ids"]

    def test_returns_all_semantic_inputs(self):
        """All non-KV-cache inputs are returned in order."""
        model = _mock_model_with_inputs(
            [
                "inputs_embeds",
                "input_ids",
                "attention_mask",
                "position_ids",
            ]
        )
        result = _graph_input_names(model)
        assert result == [
            "inputs_embeds",
            "input_ids",
            "attention_mask",
            "position_ids",
        ]


class TestIntrospectVisionOutputs:
    """_introspect_outputs surfaces extra vision outputs (DeepStack)."""

    def test_vision_deepstack_output_is_surfaced(self):
        """A Qwen3-VL vision encoder's deepstack_features output is mapped."""
        from mobius._model_package import ModelPackage

        pkg = ModelPackage(
            {
                "vision_encoder": _mock_model_with_outputs(
                    ["image_features", "deepstack_features"]
                ),
            },
            config=mock.MagicMock(),
        )
        mapping = _introspect_outputs(pkg, "vision_encoder")
        assert mapping == {
            "image_features": "image_features",
            "deepstack_features": "deepstack_features",
        }

    def test_missing_key_returns_none(self):
        from mobius._model_package import ModelPackage

        pkg = ModelPackage({}, config=mock.MagicMock())
        assert _introspect_outputs(pkg, "vision_encoder") is None


class TestCountCacheLayerSlots:
    """Tests for _count_cache_layer_slots() helper."""

    def test_counts_key_inputs(self):
        """Counts past_key_values.{i}.key inputs, not the value pairs."""
        model = _mock_model_with_inputs(
            [
                "input_ids",
                "attention_mask",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "past_key_values.1.key",
                "past_key_values.1.value",
            ]
        )
        assert _count_cache_layer_slots(model) == 2

    def test_uses_global_indices_across_hybrid_cache_types(self):
        model = _mock_model_with_inputs(
            [
                "past_key_values.0.conv_state",
                "past_key_values.1.key",
                "past_key_values.1.value",
                "past_key_values.3.recurrent_state",
            ]
        )
        assert _count_cache_layer_slots(model) == 4

    def test_returns_none_without_kv_cache(self):
        """A static-cache export (key_cache.{i}) falls back to the config."""
        model = _mock_model_with_inputs(["input_ids", "key_cache.0", "value_cache.0"])
        assert _count_cache_layer_slots(model) is None

    def test_returns_none_for_missing_model(self):
        assert _count_cache_layer_slots(None) is None


class TestGemma4RealModel:
    """Build a real tiny Gemma4 model and verify genai config inputs."""

    def test_gemma4_genai_config_from_real_model(self, tmp_path):
        """Build tiny Gemma4 VLM, generate genai config, verify inputs."""
        from mobius._builder import build_from_module
        from mobius._configs import Gemma4Config, VisionConfig
        from mobius._registry import registry
        from mobius.integrations.transformers._config_resolver import (
            _default_task_for_model,
        )
        from mobius.tasks import get_task

        config = Gemma4Config(
            model_type="gemma4",
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
            image_token_id=255999,
            bos_token_id=2,
            boa_token_id=256000,
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
        pkg.config = config

        result = write_ort_genai_config(pkg, str(tmp_path))
        with open(result["genai_config"]) as f:
            data = json.load(f)

        # Decoder inputs introspected from graph
        decoder_inputs = data["model"]["decoder"]["inputs"]
        assert "inputs_embeds" in decoder_inputs
        assert "input_ids" not in decoder_inputs
        assert "attention_mask" in decoder_inputs
        assert "position_ids" in decoder_inputs
        assert decoder_inputs["past_key_names"] == ("past_key_values.%d.key")

        # Vision inputs introspected from graph
        vision_inputs = data["model"]["vision"]["inputs"]
        assert "pixel_values" in vision_inputs
        assert "pixel_position_ids" in vision_inputs
        assert "image_grid_thw" not in vision_inputs

        # Embedding inputs introspected from graph
        emb_inputs = data["model"]["embedding"]["inputs"]
        assert "input_ids" in emb_inputs
        assert "image_features" in emb_inputs

        # Config-level properties are still present
        assert data["model"]["image_token_id"] == 255999
        assert data["model"]["bos_token_id"] == 2
        assert data["model"]["vision"]["spatial_merge_size"] == 2
        assert data["model"]["vision"]["config_filename"] == "image_processor.json"

    def test_text_only_genai_config_is_decoder_only(self, tmp_path):
        """text_only gemma4_unified export -> decoder-only genai config.

        Reproduces ``auto_export(text_only=True)``: a single-"model" package
        whose ``config.model_type`` is the text sibling ``gemma4_unified_text``.
        Verifies the ORT-GenAI ``type`` is resolved from ``pkg.config.model_type``
        (``gemma4_text``), NOT the multimodal HF config (``gemma4_unified``),
        and that no vision/audio sections or processor files are written.
        """
        from mobius._builder import build_from_module
        from mobius._configs import Gemma4Config
        from mobius._registry import registry
        from mobius.integrations.transformers._builder import _strip_to_text_only
        from mobius.integrations.transformers._config_resolver import (
            _default_task_for_model,
        )
        from mobius.tasks import get_task

        # Start from a multimodal-flavoured config and strip to text-only, the
        # same transformation build(text_only=True) applies.
        config = Gemma4Config(
            model_type="gemma4_unified",
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
            bos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        config = _strip_to_text_only(config, "gemma4_unified_text")
        assert config.model_type == "gemma4_unified_text"
        assert config.image_token_id is None
        assert config.use_bidirectional_attention is None

        model_cls = registry.get("gemma4_unified_text")
        module = model_cls(config)
        task = get_task(_default_task_for_model("gemma4_unified_text"))
        pkg = build_from_module(module, config, task=task)
        pkg.config = config

        # HF config reports the multimodal model_type; the built package's
        # config carries the text sibling. ORT type must follow the package.
        fake_hf = types.SimpleNamespace(
            model_type="gemma4_unified",
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
        )
        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf),
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._load_generation_config",
                return_value=None,
            ),
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files",
                return_value=[],
            ),
        ):
            result = write_ort_genai_config(
                pkg, str(tmp_path), hf_model_id="google/gemma-4-12B"
            )

        with open(result["genai_config"]) as f:
            data = json.load(f)

        # ORT-GenAI type resolved from pkg.config.model_type (gemma4_text),
        # NOT the multimodal HF gemma4_unified -> gemma4.
        assert data["model"]["type"] == "gemma4_text"
        # Decoder-only: input_ids decoder, no multimodal sections.
        assert "vision" not in data["model"]
        assert "audio" not in data["model"]
        assert "input_ids" in data["model"]["decoder"]["inputs"]
        # No multimodal processor artifacts.
        assert "processor_config" not in result
        assert "audio_processor" not in result
        assert not os.path.exists(os.path.join(str(tmp_path), "image_processor.json"))

    def test_auto_export_forwards_text_only_and_build_ep(self, tmp_path):
        """auto_export(text_only, ep) threads through to build() correctly.

        - ``text_only`` is forwarded to ``build``.
        - the runtime ``ep`` drives the build EP for non-CPU providers
          (``cuda`` -> GroupQueryAttention fusion).
        - ``ep="cpu"`` maps to the portable ``"default"`` build (unchanged).
        """
        captured: dict[str, object] = {}

        def fake_build(model_id, **kwargs):
            captured.update(kwargs)
            return _make_fake_llm_pkg("qwen2")

        def fake_export_package(pkg, output_dir, **kwargs):
            return {"genai_config": os.path.join(output_dir, "genai_config.json")}

        with (
            mock.patch("mobius.integrations.transformers.build", side_effect=fake_build),
            mock.patch(
                "mobius.integrations.ort_genai.auto_export.export_package",
                side_effect=fake_export_package,
            ),
        ):
            auto_export("google/gemma-4-12B", str(tmp_path), ep="cuda", text_only=True)
        assert captured["text_only"] is True
        assert captured["execution_provider"] == "cuda"

        captured.clear()
        with (
            mock.patch("mobius.integrations.transformers.build", side_effect=fake_build),
            mock.patch(
                "mobius.integrations.ort_genai.auto_export.export_package",
                side_effect=fake_export_package,
            ),
        ):
            auto_export("Qwen/Qwen2.5-0.5B", str(tmp_path), ep="cpu")
        # cpu maps to the portable default build (backward compatible).
        assert captured["execution_provider"] == "default"
        assert captured["text_only"] is False

    def test_auto_export_rejects_mage_vl_before_saving(self, tmp_path):
        pkg = _make_fake_llm_pkg("mage_vl")

        with (
            mock.patch("mobius.integrations.transformers.build", return_value=pkg),
            mock.patch.object(pkg, "save") as save,
            pytest.raises(ValueError, match=r"Mage-VL.*patch_positions"),
        ):
            auto_export("microsoft/Mage-VL", str(tmp_path))

        save.assert_not_called()

    def test_auto_export_produces_genai_config(self, tmp_path):
        """Mock build() to return a tiny package, verify genai_config."""
        import onnx_ir as ir

        from mobius._builder import build_from_module
        from mobius._configs import ArchitectureConfig
        from mobius._registry import registry
        from mobius.integrations.ort_genai.genai_config import (
            GenaiConfigGenerator,
        )

        # Build a tiny model
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
        )
        module = registry.get("qwen2")(config)
        pkg = build_from_module(module, config)

        # Fill with random weights
        rng = np.random.default_rng(42)
        for model in pkg.values():
            for init in model.graph.initializers.values():
                if init.const_value is None:
                    shape = list(init.shape)
                    init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

        # Generate genai_config from the config
        gen = GenaiConfigGenerator.from_config(config, "qwen2")
        genai_config = gen.generate()

        assert "model" in genai_config
        assert genai_config["model"]["type"] == "qwen2"
        assert genai_config["model"]["vocab_size"] == 256
        assert genai_config["model"]["decoder"]["num_hidden_layers"] == 2

        # Save models and config
        output_dir = str(tmp_path / "export")
        os.makedirs(output_dir)
        pkg.save(output_dir, progress_bar=False)
        gen.write(output_dir)

        assert os.path.exists(os.path.join(output_dir, "model.onnx"))
        assert os.path.exists(os.path.join(output_dir, "genai_config.json"))

        with open(os.path.join(output_dir, "genai_config.json")) as f:
            saved = json.load(f)
        assert saved["model"]["type"] == "qwen2"

    def test_phi4mm_detection_and_config(self, tmp_path):
        """Simulate phi4mm auto-export: verify detection and config."""
        import onnx_ir as ir

        from mobius._builder import build_from_module
        from mobius._configs import ArchitectureConfig, AudioConfig, VisionConfig
        from mobius.integrations.ort_genai.genai_config import (
            GenaiConfigGenerator,
        )
        from mobius.models.phi import Phi4MMMultiModalModel
        from mobius.tasks import Phi4MMMultiModalTask

        # Build a tiny phi4mm model
        # LongRoPE requires rope_scaling with long/short factors
        # inv_freq has int(head_dim * partial_rotary_factor) // 2 elements
        rope_dim = int(16 * 0.75) // 2  # head_dim=16, partial_rotary_factor=0.75
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=1,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="longrope",
            rope_theta=10000.0,
            partial_rotary_factor=0.75,
            original_max_position_embeddings=128,
            rope_scaling={
                "long_factor": [1.0] * rope_dim,
                "short_factor": [1.0] * rope_dim,
            },
            pad_token_id=0,
            image_token_id=200010,
            vision=VisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=28,
                patch_size=14,
                lora={"r": 4, "lora_alpha": 8},
            ),
            audio=AudioConfig(
                attention_dim=64,
                attention_heads=4,
                num_blocks=1,
                linear_units=128,
                kernel_size=3,
                input_size=80,
                token_id=200011,
                lora={"r": 4, "lora_alpha": 8},
            ),
        )
        module = Phi4MMMultiModalModel(config)
        pkg = build_from_module(module, config, task=Phi4MMMultiModalTask())

        # Verify 4-model split
        assert "vision_encoder" in pkg
        assert "audio_encoder" in pkg
        assert "embedding" in pkg
        assert "decoder" in pkg

        # Simulate auto_export detection logic
        is_vlm = "vision_encoder" in pkg and "embedding" in pkg
        has_speech = "audio_encoder" in pkg
        ort_model_type = "phi"  # HF model_type for phi4mm
        if ort_model_type == "phi" and has_speech:
            ort_model_type = "phi4mm"

        assert ort_model_type == "phi4mm"
        assert is_vlm
        assert has_speech

        # Build genai_config using the same logic as auto_export
        generator = GenaiConfigGenerator.from_config(config, ort_model_type)
        vision_kwargs = {
            "spatial_merge_size": None,
            "config_filename": "image_processor.json",
            "input_names": {
                "pixel_values": "pixel_values",
                "image_sizes": "image_sizes",
            },
        }
        generator.with_vision(image_token_id=config.image_token_id, **vision_kwargs)
        generator.with_audio(audio_token_id=config.audio.token_id)

        genai_config = generator.generate()

        # Verify config structure
        model = genai_config["model"]
        assert model["type"] == "phi4mm"
        assert model["image_token_id"] == 200010
        assert model["audio_token_id"] == 200011

        # All 4 model sections present
        assert "decoder" in model
        assert "vision" in model
        assert "speech" in model
        assert "embedding" in model

        # Vision uses phi4mm-specific inputs
        assert model["vision"]["inputs"]["pixel_values"] == "pixel_values"
        assert model["vision"]["inputs"]["image_sizes"] == "image_sizes"
        assert "image_grid_thw" not in model["vision"]["inputs"]
        assert "spatial_merge_size" not in model["vision"]
        assert model["vision"]["config_filename"] == "image_processor.json"

        # Audio section
        assert model["speech"]["inputs"]["audio_embeds"] == "audio_embeds"
        assert model["speech"]["inputs"]["audio_sizes"] == "audio_sizes"
        assert model["speech"]["inputs"]["audio_projection_mode"] == "audio_projection_mode"

        # Embedding includes audio_features
        assert model["embedding"]["inputs"]["audio_features"] == "audio_features"

        # Decoder uses inputs_embeds (multimodal)
        assert "inputs_embeds" in model["decoder"]["inputs"]

        # Save and verify files
        output_dir = str(tmp_path / "phi4mm_export")
        os.makedirs(output_dir)

        # Fill with random weights so save() doesn't complain
        rng = np.random.default_rng(42)
        for model in pkg.values():
            for init in model.graph.initializers.values():
                if init.const_value is None:
                    shape = list(init.shape)
                    init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

        pkg.save(output_dir, progress_bar=False)
        generator.write(output_dir)

        assert os.path.exists(os.path.join(output_dir, "genai_config.json"))
        # 4-model split produces subdirectories
        assert os.path.exists(os.path.join(output_dir, "vision_encoder"))
        assert os.path.exists(os.path.join(output_dir, "audio_encoder"))
        assert os.path.exists(os.path.join(output_dir, "embedding"))

        with open(os.path.join(output_dir, "genai_config.json")) as f:
            saved = json.load(f)
        assert saved["model"]["type"] == "phi4mm"
        assert "speech" in saved["model"]
