# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the onnx-genai write_onnx_genai_config dispatcher."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from mobius.integrations.onnx_genai import write_onnx_genai_config


@dataclasses.dataclass
class _Cfg:
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    hidden_size: int = 1024
    max_position_embeddings: int = 8192
    sliding_window: int | None = None
    model_type: str = "qwen"


class _DiffusionPkg(dict):
    pass


class _MultimodalPkg(dict):
    config = _Cfg()


def test_dispatch_decoder(tmp_path):
    arts = write_onnx_genai_config(
        object(), str(tmp_path), config=_Cfg(), kv_native_dtype="bf16"
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    assert meta["model"]["attention"]["type"] == "grouped_query"
    assert meta["kv_cache"]["native_dtype"] == "bf16"


def test_dispatch_diffusion(tmp_path):
    pkg = _DiffusionPkg({"denoiser": object(), "vae": object()})
    arts = write_onnx_genai_config(
        pkg,
        str(tmp_path),
        num_inference_steps=20,
        vae_filename="vae.onnx",
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    assert meta["pipeline"]["strategy"]["kind"] == "iterative"
    assert "vae" in meta["pipeline"]["models"]


def test_dispatch_diffusion_emits_clip_tokenizer(tmp_path, monkeypatch):
    """Emit tokenizer.json for a text-conditioned diffusion package.

    The onnx-genai runners can then tokenize prompts from the package alone.
    """
    import os

    class _Backend:
        def save(self, path):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{}")

    class _Tokenizer:
        backend_tokenizer = _Backend()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: _Tokenizer()
    )
    pkg = _DiffusionPkg({"denoiser": object(), "text_encoder": object(), "vae": object()})
    arts = write_onnx_genai_config(
        pkg,
        str(tmp_path),
        num_inference_steps=20,
        vae_filename="vae.onnx",
        text_encoder_filename="text_encoder.onnx",
        source="fake/model",
    )
    assert "tokenizer" in arts
    assert os.path.basename(arts["tokenizer"]) == "tokenizer.json"
    assert os.path.isfile(arts["tokenizer"])


def test_dispatch_diffusion_tokenizer_skip_is_non_fatal(tmp_path, monkeypatch):
    """Skip the tokenizer artifact without failing when none can be loaded.

    If the source has no CLIP tokenizer (or transformers can't load one), the
    build still succeeds and simply omits the tokenizer artifact.
    """

    def _boom(*args, **kwargs):
        raise OSError("no tokenizer here")

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _boom)
    pkg = _DiffusionPkg({"denoiser": object(), "text_encoder": object()})
    arts = write_onnx_genai_config(
        pkg,
        str(tmp_path),
        num_inference_steps=20,
        text_encoder_filename="text_encoder.onnx",
        source="fake/model",
    )
    assert "inference_metadata" in arts
    assert "tokenizer" not in arts


def test_dispatch_diffusion_auto_reads_scheduler_from_source(tmp_path):
    import json

    src = tmp_path / "ckpt"
    (src / "scheduler").mkdir(parents=True)
    (src / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear"})
    )
    out = tmp_path / "out"
    pkg = _DiffusionPkg({"denoiser": object()})
    arts = write_onnx_genai_config(
        pkg,
        str(out),
        num_inference_steps=15,
        source=str(src),
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    assert meta["pipeline"]["strategy"]["scheduler_config"]["kind"] == "euler"


def test_dispatch_vision_multimodal_pipeline(tmp_path):
    pkg = _MultimodalPkg(
        {
            "decoder": object(),
            "vision_encoder": object(),
            "embedding": object(),
        }
    )
    artifacts = write_onnx_genai_config(pkg, str(tmp_path), kv_native_dtype="bf16")

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    assert metadata["required_capabilities"] == [
        "kv_cache",
        "grouped_query_attention",
    ]
    assert metadata["kv_cache"] == {"native_dtype": "bf16"}
    pipeline = metadata["pipeline"]
    assert pipeline["models"] == {
        "vision_encoder": {
            "filename": "vision_encoder/model.onnx",
            "type": "vision_encoder",
        },
        "embedding": {
            "filename": "embedding/model.onnx",
            "type": "encoder",
        },
        "decoder": {
            "filename": "decoder/model.onnx",
            "type": "decoder",
            "tokenizer": "tokenizer.json",
        },
    }
    assert pipeline["strategy"]["kind"] == "composite"


def test_dispatch_audio_only_multimodal_pipeline(tmp_path):
    # The audio-only fusion shape used by speech-language ASR models such as
    # qwen3_asr and fun_asr: audio_encoder -> embedding fusion -> AR decoder.
    pkg = _MultimodalPkg(
        {
            "decoder": object(),
            "audio_encoder": object(),
            "embedding": object(),
        }
    )
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    pipeline = metadata["pipeline"]
    assert pipeline["models"] == {
        "audio_encoder": {
            "filename": "audio_encoder/model.onnx",
            "type": "audio_encoder",
        },
        "embedding": {"filename": "embedding/model.onnx", "type": "encoder"},
        "decoder": {
            "filename": "decoder/model.onnx",
            "type": "decoder",
            "tokenizer": "tokenizer.json",
        },
    }
    assert pipeline["dataflow"] == [
        {
            "from": "audio_encoder.audio_features",
            "to": "embedding.audio_features",
            "dtype": "fp32",
            "device_transfer": False,
        },
        {
            "from": "embedding.inputs_embeds",
            "to": "decoder.inputs_embeds",
            "dtype": "fp32",
            "device_transfer": False,
        },
    ]
    assert [stage["name"] for stage in pipeline["strategy"]["stages"]] == [
        "encode_audio",
        "fuse_embeddings",
        "decode",
    ]


def test_dispatch_vision_and_audio_multimodal_pipeline(tmp_path):
    pkg = _MultimodalPkg(
        {
            "decoder": object(),
            "vision_encoder": object(),
            "audio_encoder": object(),
            "embedding": object(),
        }
    )
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    pipeline = metadata["pipeline"]
    assert pipeline["dataflow"] == [
        {
            "from": "vision_encoder.image_features",
            "to": "embedding.image_features",
            "dtype": "fp32",
            "device_transfer": False,
        },
        {
            "from": "audio_encoder.audio_features",
            "to": "embedding.audio_features",
            "dtype": "fp32",
            "device_transfer": False,
        },
        {
            "from": "embedding.inputs_embeds",
            "to": "decoder.inputs_embeds",
            "dtype": "fp32",
            "device_transfer": False,
        },
    ]
    assert [stage["name"] for stage in pipeline["strategy"]["stages"]] == [
        "encode_vision",
        "encode_audio",
        "fuse_embeddings",
        "decode",
    ]


class _FakeValue:
    def __init__(self, name: str, dtype: object | None = None) -> None:
        self.name = name
        self.dtype = dtype


class _FakeGraph:
    def __init__(self, input_names: list[str], output_names: list[str] | None = None) -> None:
        self.inputs = [_FakeValue(name) for name in input_names]
        self.outputs = [_FakeValue(name) for name in (output_names or [])]


class _FakeModel:
    """Minimal stand-in for an ir.Model exposing graph input/output names."""

    def __init__(self, input_names: list[str], output_names: list[str] | None = None) -> None:
        self.graph = _FakeGraph(input_names, output_names)


class _EncoderDecoderPkg(dict):
    config = _Cfg()


def test_dispatch_speech_to_text_pipeline(tmp_path):
    # Whisper-style ASR: the decoder consumes encoder_hidden_states (cross-attn).
    pkg = _EncoderDecoderPkg(
        {
            "encoder": _FakeModel(["input_features"], ["encoder_hidden_states"]),
            "decoder": _FakeModel(["decoder_input_ids", "encoder_hidden_states"]),
        }
    )
    artifacts = write_onnx_genai_config(pkg, str(tmp_path), kv_native_dtype="bf16")

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    assert metadata["kv_cache"] == {"native_dtype": "bf16"}
    pipeline = metadata["pipeline"]
    assert pipeline["models"] == {
        "encoder": {"filename": "encoder/model.onnx", "type": "encoder"},
        "decoder": {
            "filename": "decoder/model.onnx",
            "type": "decoder",
            "tokenizer": "tokenizer.json",
        },
    }
    assert pipeline["dataflow"] == [
        {
            "from": "encoder.encoder_hidden_states",
            "to": "decoder.encoder_hidden_states",
            "dtype": "fp32",
            "device_transfer": False,
        }
    ]
    stages = pipeline["strategy"]["stages"]
    assert pipeline["strategy"]["kind"] == "composite"
    assert [stage["name"] for stage in stages] == ["encode_audio", "decode_transcript"]
    assert [stage["strategy"]["kind"] for stage in stages] == [
        "single_pass",
        "autoregressive",
    ]


def test_dispatch_audio_codec_pipeline(tmp_path):
    # A neural codec: encoder outputs codes consumed by a single-pass decoder,
    # with no cross-attention. It is a pure tensor pipeline (no decoder config).
    pkg = _EncoderDecoderPkg(
        {
            "encoder": _FakeModel(["waveform"], ["codes"]),
            "decoder": _FakeModel(["codes"], ["waveform"]),
        }
    )
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    # No decoder capabilities (produces tensors, not tokens).
    assert "model" not in metadata
    pipeline = metadata["pipeline"]
    assert pipeline["models"] == {
        "encoder": {"filename": "encoder/model.onnx", "type": "audio_encoder"},
        "decoder": {"filename": "decoder/model.onnx", "type": "vocoder"},
    }
    assert pipeline["dataflow"] == [
        {
            "from": "encoder.codes",
            "to": "decoder.codes",
            "dtype": "int64",
            "device_transfer": False,
        }
    ]
    stages = pipeline["strategy"]["stages"]
    assert [stage["strategy"]["kind"] for stage in stages] == [
        "single_pass",
        "single_pass",
    ]


def test_multi_decoder_tts_raises_precise_not_implemented(tmp_path):
    # A nested multi-decoder TTS stack (talker + code_predictor) is a designed
    # but unimplemented shape and must fail with a precise, actionable error.
    pkg = _EncoderDecoderPkg(
        {
            "talker": _FakeModel(["inputs_embeds"]),
            "code_predictor": _FakeModel(["inputs_embeds"]),
            "embedding": _FakeModel(["text_ids"]),
        }
    )
    with pytest.raises(NotImplementedError, match="nested_autoregressive"):
        write_onnx_genai_config(pkg, str(tmp_path))


def test_unrecognized_multi_component_package_fails_loudly(tmp_path):
    # A multi-component package matching no known shape must not be silently
    # emitted as a bare decoder.
    pkg = _EncoderDecoderPkg(
        {
            "widget": _FakeModel(["x"]),
            "gadget": _FakeModel(["y"]),
        }
    )
    with pytest.raises(ValueError, match="multi-component"):
        write_onnx_genai_config(pkg, str(tmp_path))


def test_decoder_emits_tokenizer_from_source(tmp_path):
    # A text-producing package emits tokenizer.json from its HF source so the
    # onnx-genai package is self-contained.
    from unittest import mock

    saved = {}

    class _FakeBackend:
        def save(self, path):
            saved["path"] = path
            with open(path, "w") as handle:
                handle.write("{}")

    fake_tok = mock.Mock()
    fake_tok.backend_tokenizer = _FakeBackend()
    fake_tf = mock.Mock()
    fake_tf.AutoTokenizer.from_pretrained.return_value = fake_tok

    with mock.patch.dict("sys.modules", {"transformers": fake_tf}):
        artifacts = write_onnx_genai_config(
            object(), str(tmp_path), config=_Cfg(), source="some/model-id"
        )

    assert artifacts.get("tokenizer") == str(tmp_path / "tokenizer.json")
    assert (tmp_path / "tokenizer.json").exists()
    fake_tf.AutoTokenizer.from_pretrained.assert_called_once()


def test_decoder_without_source_skips_tokenizer(tmp_path):
    artifacts = write_onnx_genai_config(object(), str(tmp_path), config=_Cfg())
    assert "tokenizer" not in artifacts
    assert not (tmp_path / "tokenizer.json").exists()
