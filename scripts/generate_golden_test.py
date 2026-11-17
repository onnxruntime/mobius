# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for scripts/generate_golden.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Import the generate_golden module via path manipulation since
# scripts/ is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_golden


@pytest.fixture
def audio_stub(monkeypatch):
    """Loader probes must not depend on optional audio decoding packages."""
    audio = SimpleNamespace(load=lambda *args, **kwargs: (np.zeros(16), 16000))
    monkeypatch.setitem(sys.modules, "librosa", audio)
    return audio


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    ("generator", "loader"),
    [
        ("_generate_causal_lm", "load_torch_model"),
        ("_generate_encoder", "load_torch_encoder_model"),
        ("_generate_seq2seq", "load_torch_seq2seq_model"),
        ("_generate_vision_language", "load_torch_multimodal_model"),
        ("_generate_audio_feature_extraction", "load_torch_audio_model"),
        ("_generate_image_classification", "load_torch_vision_model"),
    ],
)
def test_ordinary_loader_precision(
    monkeypatch, tmp_path, audio_stub, generator, loader, dtype, device
):
    import torch

    from mobius._testing import torch_reference

    calls = []

    def fail_at_loader(model_id, **kwargs):
        calls.append(kwargs)
        assert kwargs["dtype"] == getattr(torch, dtype)
        assert kwargs["device"] == device
        raise RuntimeError("selected precision unsupported")

    monkeypatch.setattr(torch_reference, loader, fail_at_loader)
    case = SimpleNamespace(
        model_id="offline/model",
        dtype=dtype,
        reference_loader=None,
        revision=None,
        trust_remote_code=False,
    )
    with pytest.raises(RuntimeError, match="selected precision unsupported"):
        getattr(generate_golden, generator)(case, tmp_path / "unused.json", device)
    assert len(calls) == 1  # Backend failures must not trigger an FP32 retry.


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    ("generator", "loader", "keyword"),
    [
        ("_generate_image_to_text", "AutoModel", "torch_dtype"),
        ("_generate_speech_to_text", "AutoModelForSpeechSeq2Seq", "dtype"),
        ("_generate_ctc_asr", "AutoModelForCTC", "torch_dtype"),
        ("_generate_object_detection", "AutoModelForObjectDetection", "torch_dtype"),
    ],
)
def test_direct_loader_precision(
    monkeypatch, tmp_path, audio_stub, generator, loader, keyword, dtype, device
):
    import torch

    calls = []

    def fail_at_loader(model_id, **kwargs):
        calls.append(kwargs)
        assert kwargs[keyword] == getattr(torch, dtype)
        if "device_map" in kwargs:
            assert kwargs["device_map"] == device
        raise RuntimeError("selected precision unsupported")

    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: None),
        AutoImageProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: None),
    )
    setattr(transformers, loader, SimpleNamespace(from_pretrained=fail_at_loader))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    case = SimpleNamespace(
        model_id="offline/model",
        model_type="ordinary",
        dtype=dtype,
        revision=None,
        trust_remote_code=False,
        generation_params={},
    )
    with pytest.raises(RuntimeError, match="selected precision unsupported"):
        getattr(generate_golden, generator)(case, tmp_path / "unused.json", device)
    assert len(calls) == 1


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0", "auto"])
@pytest.mark.parametrize("model_type", ["glmasr", "qwen3_asr", "gemma4"])
def test_speech_language_loader_precision(monkeypatch, dtype, device, model_type):
    import torch

    from mobius._testing import torch_reference

    calls = []

    def fail_at_loader(model_id, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("dtype", kwargs.get("torch_dtype")) == getattr(torch, dtype)
        raise RuntimeError("selected precision unsupported")

    monkeypatch.setattr(generate_golden, "_try_register_qwen3_asr", lambda: None)
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: SimpleNamespace(model_type=model_type)
        ),
        GlmAsrForConditionalGeneration=SimpleNamespace(from_pretrained=fail_at_loader),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr.core.transformers_backend.modeling_qwen3_asr",
        SimpleNamespace(
            Qwen3ASRForConditionalGeneration=SimpleNamespace(from_pretrained=fail_at_loader)
        ),
    )
    monkeypatch.setattr(torch_reference, "load_torch_multimodal_model", fail_at_loader)
    case = SimpleNamespace(
        model_id="offline/model", dtype=dtype, revision=None, trust_remote_code=False
    )
    with pytest.raises(RuntimeError, match="selected precision unsupported"):
        generate_golden._load_speech_language_model(case, device)
    assert len(calls) == 1


def test_unknown_reference_precision_is_rejected():
    with pytest.raises(ValueError, match="Unsupported reference dtype"):
        generate_golden._reference_dtype("float8")


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_processor_float_inputs_follow_precision_without_casting_ids_or_masks(dtype):
    import torch

    processed = {
        "pixel_values": torch.tensor([1.25]),
        "input_features": torch.tensor([2.5]),
        "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
        "attention_mask": torch.tensor([[True, False]]),
    }
    result = generate_golden._prepare_reference_inputs(
        processed, torch.device("cpu"), getattr(torch, dtype)
    )
    for name in ("pixel_values", "input_features"):
        assert result[name].dtype == getattr(torch, dtype)
    assert result["input_ids"].dtype == torch.int64
    assert result["attention_mask"].dtype == torch.bool
    for name in processed:
        torch.testing.assert_close(result[name].float(), processed[name].float())


def test_processor_non_tensor_metadata_is_preserved_by_identity():
    import torch

    metadata = {
        "sampling_rate": 16000,
        "audio_path": "unused.wav",
        "image_sizes": [(32, 64)],
        "image_grid": np.array([[1, 2, 3]], dtype=np.int64),
        "options": {"padding": False},
        "optional": None,
        "custom": SimpleNamespace(),
    }
    processed = {
        "input_features": torch.tensor([1.25]),
        "input_ids": torch.tensor([[1]], dtype=torch.int64),
        **metadata,
    }
    result = generate_golden._prepare_reference_inputs(
        processed, torch.device("cpu"), torch.bfloat16
    )
    assert result["input_features"].dtype == torch.bfloat16
    assert result["input_ids"].dtype == torch.int64
    for name, value in metadata.items():
        if isinstance(value, list):
            assert result[name] == value
            assert type(result[name]) is type(value)
        else:
            assert result[name] is value


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_processor_nested_list_tuple_tensor_leaves_are_moved(dtype, device):
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    selected_dtype = getattr(torch, dtype)
    opaque = SimpleNamespace()
    mapping = {"tensor": torch.tensor([3.5], dtype=torch.float32)}
    nested = [
        torch.tensor([1.25], dtype=torch.float32),
        (
            torch.tensor([1, 2], dtype=torch.int64),
            [torch.tensor([True, False]), torch.tensor([2.5], dtype=torch.float32)],
            opaque,
            mapping,
            "metadata",
        ),
    ]
    processed = BatchFeature({"nested": nested})
    result = generate_golden._prepare_reference_inputs(
        processed, torch.device(device), selected_dtype
    )
    moved = result["nested"]
    assert type(moved) is list
    assert type(moved[1]) is tuple
    assert type(moved[1][1]) is list
    for tensor, original in (
        (moved[0], nested[0]),
        (moved[1][0], nested[1][0]),
        (moved[1][1][0], nested[1][1][0]),
        (moved[1][1][1], nested[1][1][1]),
    ):
        expected_dtype = selected_dtype if original.is_floating_point() else original.dtype
        assert tensor.dtype == expected_dtype
        assert tensor.device == torch.device(device)
        assert tensor.shape == original.shape
        if device == "cpu":
            torch.testing.assert_close(tensor.float(), original.float())
    assert moved[1][2] is opaque
    assert moved[1][3] is mapping
    assert mapping["tensor"].dtype == torch.float32
    assert moved[1][4] == "metadata"
    assert nested[0].dtype == torch.float32


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(
    "generator",
    [
        "_generate_vision_language",
        "_generate_image_to_text",
        "_generate_speech_to_text",
        "_generate_speech_language",
        "_generate_ctc_asr",
        "_generate_object_detection",
    ],
)
def test_media_generators_execute_selected_precision(
    monkeypatch, tmp_path, audio_stub, generator, dtype
):
    """Exercise real floating tensors through forward, generation, and metrics."""
    import torch
    from PIL import Image

    from mobius._testing import golden, torch_reference

    selected_dtype = getattr(torch, dtype)
    calls = []
    saved = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=selected_dtype))
            self.config = SimpleNamespace(decoder_start_token_id=1)
            self.generation_config = SimpleNamespace(eos_token_id=9)

        def check_inputs(self, kwargs):
            assert any(t.is_floating_point() for t in kwargs.values())
            for name, tensor in kwargs.items():
                if tensor.is_floating_point():
                    assert tensor.dtype == selected_dtype
                elif "ids" in name:
                    assert tensor.dtype == torch.int64
                elif "mask" in name:
                    assert tensor.dtype == torch.bool

        def forward(self, **kwargs):
            self.check_inputs(kwargs)
            calls.append("forward")
            logits = self.weight * torch.tensor([[[1.25, -2.5, 0.5]]], dtype=selected_dtype)
            return SimpleNamespace(logits=logits)

        def generate(self, **kwargs):
            kwargs.pop("max_new_tokens")
            kwargs.pop("do_sample")
            self.check_inputs(kwargs)
            calls.append("generate")
            return torch.tensor([[1, 2, 3]], dtype=torch.int64)

    class Processor:
        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
                "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
                "input_features": torch.ones((1, 4), dtype=torch.float32),
                "attention_mask": torch.tensor([[True, False]]),
            }

        def decode(self, *args, **kwargs):
            return "decoded"

        def batch_decode(self, *args, **kwargs):
            return ["decoded"]

    model = Model()
    processor = Processor()

    def load_model(*args, **kwargs):
        assert kwargs.get("dtype", kwargs.get("torch_dtype")) == selected_dtype
        return model

    model_loader = SimpleNamespace(from_pretrained=load_model)
    processor_loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=model_loader,
            AutoModelForSpeechSeq2Seq=model_loader,
            AutoModelForCTC=model_loader,
            AutoModelForObjectDetection=model_loader,
            AutoProcessor=processor_loader,
            AutoImageProcessor=processor_loader,
        ),
    )
    monkeypatch.setattr(
        torch_reference,
        "load_torch_multimodal_model",
        lambda *args, **kwargs: (load_model(*args, **kwargs), processor, processor),
    )
    monkeypatch.setattr(
        generate_golden,
        "_load_speech_language_model",
        lambda *args: (model, processor, model),
    )
    monkeypatch.setattr(generate_golden, "_detection_forced_size", lambda *args: None)
    image = SimpleNamespace(convert=lambda *args: None)
    monkeypatch.setattr(Image, "open", lambda *args: image)
    monkeypatch.setattr(golden, "save_golden_ref", lambda path, **kwargs: saved.append(kwargs))
    monkeypatch.setattr(golden, "save_generation_json", lambda *args, **kwargs: None)
    case = SimpleNamespace(
        model_id="offline/model",
        model_type="ordinary",
        dtype=dtype,
        revision=None,
        trust_remote_code=False,
        prompts=["prompt"],
        images=["unused.png"],
        videos=[],
        audio=["unused.wav"],
        decoder_prompt="prompt",
        media_max_pixels=None,
        generation_params={},
        level=["L4", "L5"],
    )
    getattr(generate_golden, generator)(case, tmp_path / "unused.json", "cpu")
    expected_calls = (
        ["forward"]
        if generator in ("_generate_ctc_asr", "_generate_object_detection")
        else ["forward", "generate"]
    )
    assert calls == expected_calls
    assert len(saved) == 1
    assert saved[0]["top10_logits"] == [1.25, 0.5, -2.5]
    assert saved[0]["logits_summary"].dtype == np.float64


class TestExtractLogitsGolden:
    """Tests for _extract_logits_golden() helper."""

    def test_basic_extraction(self):
        """Top-1/2/10 IDs and logits are correctly extracted."""
        rng = np.random.default_rng(42)
        logits = rng.standard_normal(100).astype(np.float32)

        result = generate_golden._extract_logits_golden(logits)

        assert result["top1_id"] == int(np.argmax(logits))
        assert len(result["top10_ids"]) == 10
        assert len(result["top10_logits"]) == 10
        # Top-10 IDs should be sorted by descending logit
        assert result["top10_logits"][0] >= result["top10_logits"][-1]
        # top1_id should be the first in top10
        assert result["top10_ids"][0] == result["top1_id"]

    def test_summary_statistics(self):
        """logits_summary contains [max, min, mean, std]."""
        logits = np.array([1.0, -2.0, 3.0, 0.0, -1.0], dtype=np.float32)

        result = generate_golden._extract_logits_golden(logits)
        summary = result["logits_summary"]

        assert summary[0] == pytest.approx(3.0)  # max
        assert summary[1] == pytest.approx(-2.0)  # min
        assert summary[2] == pytest.approx(0.2)  # mean
        assert summary[3] > 0  # std > 0

    def test_small_vocab(self):
        """Works with vocab smaller than 10 tokens."""
        logits = np.array([5.0, 3.0, 1.0], dtype=np.float32)

        result = generate_golden._extract_logits_golden(logits)

        assert result["top1_id"] == 0
        assert result["top2_id"] == 1
        assert len(result["top10_ids"]) == 3

    def test_dtype_promotion(self):
        """float32 input produces float64 summary."""
        logits = np.ones(50, dtype=np.float32)
        result = generate_golden._extract_logits_golden(logits)

        assert result["logits_summary"].dtype == np.float64


def test_temporary_processor_max_pixels_restores_none_and_missing_attributes():
    class ProcessorPart:
        pass

    image_processor = ProcessorPart()
    image_processor.max_pixels = None
    image_processor.size = ProcessorPart()
    image_processor.size.longest_edge = None
    video_processor = ProcessorPart()
    processor = ProcessorPart()
    processor.image_processor = image_processor
    processor.video_processor = video_processor

    with generate_golden._temporary_processor_max_pixels(processor, 1234):
        assert image_processor.max_pixels == 1234
        assert image_processor.size.longest_edge == 1234
        assert video_processor.max_pixels == 1234

    assert image_processor.max_pixels is None
    assert image_processor.size.longest_edge is None
    assert not hasattr(video_processor, "max_pixels")


class TestDryRun:
    """Tests for main() with --dry-run (no HF inference needed)."""

    def test_dry_run_all_cases(self):
        """--dry-run lists cases without running inference."""
        result = subprocess.run(
            [sys.executable, "scripts/generate_golden.py", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "Found" in result.stdout
        assert "DRY-RUN" in result.stdout

    def test_dry_run_with_filter(self):
        """--dry-run --filter limits output to matching cases."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_golden.py",
                "--dry-run",
                "--filter",
                "gpt2",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "gpt2" in result.stdout
        # Should not contain other models
        assert "qwen" not in result.stdout.lower()

    def test_dry_run_with_task_type(self):
        """--dry-run --task-type filters by task."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_golden.py",
                "--dry-run",
                "--task-type",
                "encoder",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "bert" in result.stdout.lower()

    def test_dry_run_no_matches(self):
        """--filter with no matches exits cleanly."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_golden.py",
                "--dry-run",
                "--filter",
                "nonexistent_model_xyz",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "No test cases found" in result.stdout


class TestGeneratorDispatch:
    """Tests for the _GENERATORS dispatch table."""

    def test_all_task_types_have_generators(self):
        """All task types used in YAML cases have generators."""
        expected_types = {
            "text-generation",
            "feature-extraction",
            "seq2seq",
            "image-text-to-text",
            "speech-to-text",
            "audio-feature-extraction",
        }
        assert expected_types <= set(generate_golden._GENERATORS.keys())

    def test_generator_functions_are_callable(self):
        for name, func in generate_golden._GENERATORS.items():
            assert callable(func), f"Generator for {name} is not callable"
