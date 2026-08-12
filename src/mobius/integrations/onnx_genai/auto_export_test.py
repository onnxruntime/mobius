# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the onnx-genai write_onnx_genai_config dispatcher."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import yaml

from mobius._configs import QuantizationConfig
from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai import write_onnx_genai_config
from mobius.integrations.onnx_genai.inference_metadata_test import (
    _decoder_model,
    _model,
    _value,
)


@dataclasses.dataclass
class _Cfg:
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    hidden_size: int = 1024
    max_position_embeddings: int = 8192
    sliding_window: int | None = None
    model_type: str = "qwen"


@dataclasses.dataclass
class _Int4Cfg(_Cfg):
    dtype: ir.DataType = ir.DataType.FLOAT16
    quantization: QuantizationConfig = dataclasses.field(
        default_factory=lambda: QuantizationConfig(bits=4, quant_method="rtn")
    )


class _DiffusionPkg(dict):
    pass


def _diffusion_package(*, text: bool = False):
    latent = ["batch", 4, "height", "width"]
    denoiser_inputs = [
        _value("sample", ir.DataType.FLOAT, latent),
        _value("timestep", ir.DataType.FLOAT, ["batch"]),
    ]
    components = {}
    if text:
        denoiser_inputs.append(
            _value(
                "encoder_hidden_states",
                ir.DataType.FLOAT,
                ["batch", "prompt_sequence", 32],
            )
        )
        components["text_encoder"] = _model(
            "text_encoder",
            [_value("input_ids", ir.DataType.INT64, ["batch", "prompt_sequence"])],
            [
                (
                    "encoder_hidden_states",
                    ir.DataType.FLOAT,
                    ["batch", "prompt_sequence", 32],
                )
            ],
        )
    denoiser = _model(
        "denoiser",
        denoiser_inputs,
        [("noise_pred", ir.DataType.FLOAT, latent)],
    )
    vae = _model(
        "vae_decoder",
        [_value("latent", ir.DataType.FLOAT, latent)],
        [("image", ir.DataType.FLOAT, ["batch", 3, "image_height", "image_width"])],
    )
    components.update({"denoiser": denoiser, "vae_decoder": vae})
    return ModelPackage(components)


class _MultimodalPkg(dict):
    config = _Cfg()


def _decoder_package(config=None):
    model = _decoder_model(
        [],
        position_shape=["batch", "sequence"],
        raw_token_input=True,
    )
    return ModelPackage({"model": model}, config=config or _Cfg())


def test_dispatch_decoder(tmp_path):
    package = _decoder_package(_Int4Cfg())
    arts = write_onnx_genai_config(package, str(tmp_path), config=_Int4Cfg())
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    workflow = meta["pipeline"]["workflow"]
    assert workflow["manifest"]["ir_version"] == "1.0"
    assert workflow["components"]["token_sampler"]["policy"]["role"] == "token_sampler"
    assert workflow["components"]["termination"]["policy"]["role"] == ("termination_predicate")
    assert workflow["graph"]["kind"] == "loop"
    assert all(
        value["source"]["kind"] != "application" for value in workflow["inputs"].values()
    )
    assert [node["component"] for node in workflow["graph"]["setup"]["nodes"]] == [
        "decoder_state_initializer",
        "model",
    ]
    body = workflow["graph"]["body"]["nodes"]
    assert [node["kind"] for node in body].count("emit") == 1
    assert next(node for node in body if node["kind"] == "emit")["value"] == "sample.body"
    assert workflow["state"]["iteration"]["initializer"] == "package.zero_iteration"
    assert workflow["state"]["token"]["initializer"] == "initializer.token_slot"
    assert (tmp_path / "policies" / "token_sampler.onnx").is_file()


def test_dispatch_language_diffusion(tmp_path):
    package = ModelPackage(
        {
            "model": _model(
                "masked_denoiser",
                [_value("input_ids", ir.DataType.INT64, ["batch", "sequence"])],
                [
                    ("logits", ir.DataType.FLOAT, ["batch", "sequence", 128]),
                    ("proposed_tokens", ir.DataType.INT64, ["batch", "sequence"]),
                ],
            )
        },
        config=_Cfg(model_type="llada"),
    )
    artifacts = write_onnx_genai_config(
        package,
        str(tmp_path),
        num_inference_steps=12,
    )
    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)
    pipeline = metadata["pipeline"]
    assert set(pipeline) == {"workflow"}
    assert pipeline["workflow"]["inputs"]["request.max_iterations"]["default"] == 12
    assert (tmp_path / "policies" / "masked_update.onnx").is_file()


def test_dispatch_diffusion(tmp_path):
    pkg = _diffusion_package()
    arts = write_onnx_genai_config(
        pkg,
        str(tmp_path),
        num_inference_steps=20,
        vae_filename="vae.onnx",
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    workflow = meta["pipeline"]["workflow"]
    assert workflow["graph"]["nodes"][0]["iteration"]["value"] == "loop.iteration"
    assert workflow["graph"]["nodes"][1]["component"] == "vae_decoder"
    assert "strategy" not in meta["pipeline"]


def test_single_diffusion_component_uses_flat_model_path(tmp_path):
    pkg = _DiffusionPkg({"transformer": object()})
    artifacts = write_onnx_genai_config(pkg, str(tmp_path), num_inference_steps=2)
    with open(artifacts["inference_metadata"], encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    assert metadata["pipeline"]["models"]["denoiser"]["filename"] == "model.onnx"


def test_rejects_unsupported_qwen_image_edit_runtime_export(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "scheduler").mkdir(parents=True)
    (source / "processor").mkdir()
    (source / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "base_image_seq_len": 256,
                "max_image_seq_len": 8192,
                "base_shift": 0.5,
                "max_shift": 0.9,
                "use_dynamic_shifting": True,
            }
        ),
        encoding="utf-8",
    )
    for filename in (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    ):
        (source / "processor" / filename).write_text("{}", encoding="utf-8")

    pkg = _DiffusionPkg(
        {
            "transformer": object(),
            "text_encoder": object(),
            "text_encoder_vision_encoder": object(),
            "text_encoder_embedding": object(),
            "vae_encoder": object(),
            "vae_decoder": object(),
        }
    )
    pkg.config = SimpleNamespace(
        model_type="qwen_image_edit",
        processor_config={"patch_size": 14, "merge_size": 2},
    )
    with pytest.raises(ValueError, match="cannot execute Qwen Image Edit"):
        write_onnx_genai_config(
            pkg,
            str(output),
            source=str(source),
            num_inference_steps=3,
        )


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
    pkg = _diffusion_package(text=True)
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
    pkg = _diffusion_package(text=True)
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
    pkg = _diffusion_package()
    arts = write_onnx_genai_config(
        pkg,
        str(out),
        num_inference_steps=15,
        source=str(src),
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    components = meta["pipeline"]["workflow"]["components"]
    assert components["diffusion_schedule"]["ports"]["outputs"]["schedule"]["shape"] == [16]


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
    assert metadata["kv_cache"] == {"native_dtype": "bfloat16"}
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


def test_dispatch_audio_only_multimodal_pipeline(tmp_path, monkeypatch):
    # The audio-only fusion shape used by speech-language ASR models such as
    # qwen3_asr and fun_asr: audio_encoder -> embedding fusion -> AR decoder.
    pkg = _MultimodalPkg(
        {
            "decoder": object(),
            "audio_encoder": object(),
            "embedding": object(),
        }
    )
    audio_processor = tmp_path / "audio_processor.json"
    audio_processor.write_text("{}")
    calls: list[tuple[str | None, str | None]] = []

    def fake_audio_processor(output_dir, source, *, revision=None):
        calls.append((source, revision))
        return str(audio_processor)

    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.auto_export._write_hf_audio_processor",
        fake_audio_processor,
    )
    artifacts = write_onnx_genai_config(
        pkg,
        str(tmp_path),
        source="zai-org/GLM-ASR-Nano-2512",
        revision="pinned-revision",
    )
    assert artifacts["audio_processor"] == str(audio_processor)
    assert calls == [("zai-org/GLM-ASR-Nano-2512", "pinned-revision")]

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
            "decoder": _FakeModel(["decoder_input_ids", "encoder_hidden_states"], ["logits"]),
        }
    )
    artifacts = write_onnx_genai_config(pkg, str(tmp_path), kv_native_dtype="bf16")

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    assert metadata["kv_cache"] == {"native_dtype": "bfloat16"}
    pipeline = metadata["pipeline"]
    assert pipeline["models"]["encoder"]["filename"] == "encoder/model.onnx"
    assert pipeline["models"]["encoder"]["type"] == "encoder"
    decoder_model = pipeline["models"]["decoder"]
    assert decoder_model["filename"] == "decoder/model.onnx"
    assert decoder_model["type"] == "decoder"
    assert decoder_model["tokenizer"] == "tokenizer.json"
    assert decoder_model["io"]["logits_output"] == "logits"
    assert decoder_model["io"]["kv_ownership"] == "owned"
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


def test_dispatch_speech_to_text_routes_encoder_mask(tmp_path):
    pkg = _EncoderDecoderPkg(
        {
            "encoder": _FakeModel(
                ["input_values", "attention_mask"],
                ["encoder_hidden_states", "encoder_attention_mask"],
            ),
            "decoder": _FakeModel(
                [
                    "decoder_input_ids",
                    "encoder_hidden_states",
                    "encoder_attention_mask",
                ],
                ["logits"],
            ),
        }
    )
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    assert metadata["pipeline"]["dataflow"][1] == {
        "from": "encoder.encoder_attention_mask",
        "to": "decoder.encoder_attention_mask",
        "dtype": "int64",
        "device_transfer": False,
    }


def test_dispatch_audio_codec_pipeline(tmp_path):
    encoder = _model(
        "encoder",
        [_value("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
        [("codes", ir.DataType.INT64, ["batch", 16, "frames"])],
    )
    decoder = _model(
        "decoder",
        [_value("codes", ir.DataType.INT64, ["batch", 16, "frames"])],
        [("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
    )
    pkg = ModelPackage({"encoder": encoder, "decoder": decoder})
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    assert "model" not in metadata
    assert not {"models", "dataflow", "strategy", "phases"}.intersection(metadata["pipeline"])
    workflow = metadata["pipeline"]["workflow"]
    assert workflow["graph"]["nodes"][0]["outputs"] == {"codes": "codec.codes"}
    assert workflow["graph"]["nodes"][1]["inputs"] == {"codes": "codec.codes"}
    assert workflow["outputs"]["waveform"]["stage"] == "post_adapter"


def test_multi_decoder_tts_without_pre_embedder_raises_precise_not_implemented(tmp_path):
    # A nested multi-decoder TTS stack lacking the talker_step_embedder
    # pre-embedder cannot yet be mapped and must fail with a precise error.
    pkg = _EncoderDecoderPkg(
        {
            "talker": _FakeModel(["inputs_embeds"]),
            "code_predictor": _FakeModel(["inputs_embeds"]),
            "embedding": _FakeModel(["text_ids"]),
        }
    )
    with pytest.raises(NotImplementedError, match="nested generic workflow loops"):
        write_onnx_genai_config(pkg, str(tmp_path))


@dataclasses.dataclass
class _TTSSubCfg:
    num_code_groups: int = 16


@dataclasses.dataclass
class _TTSCfg(_Cfg):
    tts: _TTSSubCfg = dataclasses.field(default_factory=_TTSSubCfg)


class _TTSPkg(dict):
    config = _TTSCfg()


def test_dispatch_multi_decoder_tts_with_pre_embedder(tmp_path):
    pkg = _TTSPkg(
        {
            "talker": _FakeModel(["inputs_embeds"], ["logits", "last_hidden_state"]),
            "code_predictor": _FakeModel(["inputs_embeds"], ["logits", "codec_embeddings"]),
            "talker_step_embedder": _FakeModel(["frame_codes"], ["inputs_embeds"]),
            "talker_prefill_embedder": _FakeModel(
                ["text_ids"], ["prefill_embeds", "trailing_text_embeds"]
            ),
            "embedding": _FakeModel(["text_ids"]),
        }
    )
    with pytest.raises(NotImplementedError, match="nested-loop induction SSA value"):
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
            _decoder_package(), str(tmp_path), config=_Cfg(), source="some/model-id"
        )

    assert artifacts.get("tokenizer") == str(tmp_path / "tokenizer.json")
    assert (tmp_path / "tokenizer.json").exists()
    fake_tf.AutoTokenizer.from_pretrained.assert_called_once()


def test_decoder_without_source_skips_tokenizer(tmp_path):
    artifacts = write_onnx_genai_config(_decoder_package(), str(tmp_path), config=_Cfg())
    assert "tokenizer" not in artifacts
    assert not (tmp_path / "tokenizer.json").exists()
