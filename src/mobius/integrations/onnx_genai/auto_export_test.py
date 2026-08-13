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
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_decoder_workflow_metadata,
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


@dataclasses.dataclass
class _VisionCfg:
    patch_size: int = 14
    temporal_patch_size: int = 2
    merge_size: int = 1
    spatial_merge_size: int = 1
    size: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"shortest_edge": 224, "longest_edge": 224}
    )


@dataclasses.dataclass
class _VlmCfg(_Cfg):
    vision: _VisionCfg = dataclasses.field(default_factory=_VisionCfg)
    image_token_id: int = 32000
    eos_token_id: int = 2


def _vlm_package(*, audio: bool = False):
    vision = _model(
        "vision_encoder",
        [
            _value("pixel_values", ir.DataType.FLOAT, ["patches", 1176]),
            _value("grid_thw", ir.DataType.INT64, ["images", 3]),
        ],
        [("image_features", ir.DataType.FLOAT, ["batch", 256, 32])],
    )
    embedding_inputs = [
        _value("input_ids", ir.DataType.INT64, ["batch", "sequence"]),
        _value("image_features", ir.DataType.FLOAT, ["batch", 256, 32]),
    ]
    components = {"vision_encoder": vision}
    if audio:
        components["audio_encoder"] = _model(
            "audio_encoder",
            [_value("input_features", ir.DataType.FLOAT, ["batch", 80, "frames"])],
            [("audio_features", ir.DataType.FLOAT, ["batch", 64, 32])],
        )
        embedding_inputs.append(_value("audio_features", ir.DataType.FLOAT, ["batch", 64, 32]))
    embedding = _model(
        "embedding",
        embedding_inputs,
        [("inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence", 32])],
    )
    decoder = _decoder_model(
        [("inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence", 32])],
        position_shape=["batch", "sequence"],
    )
    components.update({"embedding": embedding, "decoder": decoder})
    return ModelPackage(components, config=_VlmCfg())


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
    assert workflow["components"]["token_sampler"]["contract"]["id"] == (
        "onnx-genai.token-sampler"
    )
    assert workflow["components"]["termination"]["contract"]["id"] == (
        "onnx-genai.termination-predicate"
    )
    assert workflow["steps"][0]["kind"] == "loop"
    assert all(
        value["source"]["kind"] != "application" for value in workflow["inputs"].values()
    )
    assert [node["component"] for node in workflow["steps"][0]["setup"]] == [
        "decoder_state_initializer",
        "model",
        "last_token_logits",
    ]
    body = workflow["steps"][0]["steps"]
    assert [node["kind"] for node in body].count("emit") == 1
    assert next(node for node in body if node["kind"] == "emit")["value"] == "sample.body"
    assert workflow["state"]["iteration"]["initializer"] == "package.zero_iteration"
    assert workflow["state"]["token"]["initializer"] == "initializer.token_slot"
    assert workflow["state"]["logits"] == {
        "contract": {"dtype": "float32", "rank": 2, "shape": ["batch", 128]},
        "scope": "invocation",
        "initializer": "decoder.setup.last_logits",
        "recurrence": {"kind": "invariant"},
    }
    assert (tmp_path / "policies" / "token_sampler.onnx").is_file()


def test_seeded_decoder_sampler_uses_request_controls_and_direct_kv_carry():
    workflow = build_decoder_workflow_metadata(
        _decoder_package(), _Cfg(), sampler="seeded_categorical"
    )["pipeline"]["workflow"]
    sampler = workflow["components"]["token_sampler"]
    assert sampler["application_overridable"] is True
    assert sampler["contract"]["bindings"] == {
        "logits": "logits",
        "token": "token",
        "temperature": "temperature",
        "top_k": "top_k",
        "top_p": "top_p",
        "grammar_mask": "grammar_mask",
        "rng_seed": "seed",
        "rng_offset": "offset",
        "rng_next_offset": "next_offset",
    }
    sampler_step = next(
        step
        for step in workflow["steps"][0]["steps"]
        if step.get("component") == "token_sampler"
    )
    assert sampler_step["inputs"] == {
        "logits": "logits",
        "temperature": "request.temperature",
        "top_k": "request.top_k",
        "top_p": "request.top_p",
        "grammar_mask": "request.grammar_mask",
        "seed": "request.seed",
        "offset": "rng_offset",
    }
    assert workflow["state"]["rng_offset"]["class"] == "semantic"
    assert workflow["state"]["rng_offset"]["initializer"] == "request.rng_offset"
    assert not any("kv_update" in name for name in workflow["components"])


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
    assert workflow["steps"][0]["iteration"]["value"] == "loop.iteration"
    assert workflow["steps"][1]["component"] == "vae_decoder"
    assert "strategy" not in meta["pipeline"]
    assert (tmp_path / "policies" / "solver_step.onnx").is_file()
    assert (tmp_path / "policies" / "schedule_lookup.onnx").is_file()


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
        guidance_scale=1.0,
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
        guidance_scale=1.0,
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
    assert "ports" not in components["diffusion_schedule"]
    schedule = ir.load(out / "policies" / "diffusion_schedule.onnx")
    assert list(schedule.graph.outputs[0].shape) == [16]


def test_dispatch_vision_multimodal_pipeline(tmp_path):
    pkg = _vlm_package()
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    pipeline = metadata["pipeline"]
    assert set(pipeline) == {"workflow"}
    workflow = pipeline["workflow"]
    assert workflow["manifest"]["adapter_abis"] == {"onnx-genai.image-preprocess": "1"}
    assert workflow["steps"][0]["setup"][0]["component"] == "image_preprocess"
    assert workflow["steps"][0]["setup"][1]["component"] == "vision_encoder"
    assert workflow["steps"][0]["setup"][3]["component"] == "embedding"
    assert workflow["steps"][0]["iteration"]["value"] == "loop.iteration"
    assert workflow["state"]["logits"]["contract"] == {
        "dtype": "float32",
        "rank": 2,
        "shape": ["batch", 128],
    }
    assert workflow["state"]["logits"]["initializer"] == "decoder.setup.last_logits"
    assert (tmp_path / "policies" / "token_sampler.onnx").is_file()


def test_workflow_vlm_rejects_kv_dtype_override(tmp_path):
    with pytest.raises(ValueError, match="kv_native_dtype overrides are unsupported"):
        write_onnx_genai_config(
            _vlm_package(),
            str(tmp_path),
            kv_native_dtype="bf16",
        )


def test_text_diffusion_requires_explicit_unguided_mode(tmp_path):
    with pytest.raises(ValueError, match=r"pass guidance_scale=1\.0 explicitly"):
        write_onnx_genai_config(
            _diffusion_package(text=True),
            str(tmp_path),
        )


def test_dispatch_audio_only_multimodal_pipeline(tmp_path):
    # The audio-only fusion shape used by speech-language ASR models such as
    # qwen3_asr and fun_asr: audio_encoder -> embedding fusion -> AR decoder.
    pkg = _vlm_package(audio=True)
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    setup = metadata["pipeline"]["workflow"]["steps"][0]["setup"]
    assert [node["component"] for node in setup[:3]] == [
        "image_preprocess",
        "vision_encoder",
        "audio_encoder",
    ]
    embedding = next(node for node in setup if node.get("component") == "embedding")
    assert embedding["inputs"]["audio_features"] == "audio.audio_features"


def test_dispatch_vision_and_audio_multimodal_pipeline(tmp_path):
    pkg = _vlm_package(audio=True)
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    workflow = metadata["pipeline"]["workflow"]
    assert set(workflow["components"]) >= {
        "vision_encoder",
        "audio_encoder",
        "embedding",
        "decoder",
    }


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
    assert workflow["steps"][0]["outputs"] == {"codes": "codec.codes"}
    assert workflow["steps"][1]["inputs"] == {"codes": "codec.codes"}
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
    pkg = ModelPackage(
        {
            "talker": _model(
                "talker",
                [_value("inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence", 16])],
                [("last_hidden_state", ir.DataType.FLOAT, ["batch", 16])],
            ),
            "code_predictor": _model(
                "code_predictor",
                [
                    _value("last_hidden_state", ir.DataType.FLOAT, ["batch", 16]),
                    _value("step_index", ir.DataType.INT64, ["batch"]),
                ],
                [("logits", ir.DataType.FLOAT, ["batch", 64])],
            ),
            "talker_step_embedder": _model(
                "talker_step_embedder",
                [_value("frame_codes", ir.DataType.INT64, ["batch", 16])],
                [("inputs_embeds", ir.DataType.FLOAT, ["batch", 1, 16])],
            ),
            "talker_prefill_embedder": _model(
                "talker_prefill_embedder",
                [_value("text_ids", ir.DataType.INT64, ["batch", "sequence"])],
                [("prefill_embeds", ir.DataType.FLOAT, ["batch", "sequence", 16])],
            ),
            "codec": _model(
                "codec",
                [_value("codes", ir.DataType.INT64, ["batch", 16, "frames"])],
                [("waveform", ir.DataType.FLOAT, ["batch", 1, "samples"])],
            ),
        },
        config=_TTSCfg(),
    )
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))
    with open(artifacts["inference_metadata"]) as handle:
        workflow = yaml.safe_load(handle)["pipeline"]["workflow"]
    outer = workflow["steps"][0]
    assert outer["iteration"]["value"] == "talker.iteration"
    assert outer["steps"][2]["iteration"]["value"] == "code.iteration"
    assert (tmp_path / "policies" / "code_frame_update.onnx").is_file()


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
