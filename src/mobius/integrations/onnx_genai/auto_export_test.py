# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the onnx-genai write_onnx_genai_config dispatcher."""

from __future__ import annotations

import dataclasses
import os

import onnx_ir as ir
import pytest
import yaml

from mobius._configs import QuantizationConfig
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


@dataclasses.dataclass
class _Int4Cfg(_Cfg):
    dtype: ir.DataType = ir.DataType.FLOAT16
    quantization: QuantizationConfig = dataclasses.field(
        default_factory=lambda: QuantizationConfig(bits=4, quant_method="rtn")
    )


class _DiffusionPkg(dict):
    pass


class _MultimodalPkg(dict):
    config = _Cfg()


def test_dispatch_decoder(tmp_path):
    arts = write_onnx_genai_config(object(), str(tmp_path), config=_Int4Cfg())
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    assert meta["model"]["attention"]["type"] == "grouped_query_attention"
    assert meta["kv_cache"]["native_dtype"] == "float16"


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
    with pytest.raises(NotImplementedError, match="nested_autoregressive"):
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
    # The real Qwen3-TTS shape: talker + code_predictor + talker_step_embedder
    # (+ talker_prefill_embedder) emits the pre_embedder/prefill-driven
    # nested_autoregressive contract.
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
    artifacts = write_onnx_genai_config(pkg, str(tmp_path))

    with open(artifacts["inference_metadata"]) as handle:
        metadata = yaml.safe_load(handle)

    pipeline = metadata["pipeline"]
    assert set(pipeline["models"]) == {
        "talker",
        "talker_step_embedder",
        "code_predictor",
        "talker_prefill_embedder",
    }
    stage = pipeline["strategy"]["stages"][0]["strategy"]
    assert stage["kind"] == "nested_autoregressive"
    assert stage["inner_embedding_output"] == "codec_embeddings"
    assert stage["pre_embedder"]["component"] == "talker_step_embedder"
    assert stage["prefill_embedder"]["component"] == "talker_prefill_embedder"
    assert stage["num_code_groups"] == 16
    assert pipeline["phases"]["talker_prefill_embedder"]["run_on"] == "prompt_only"
    assert {
        "from": "talker_step_embedder.inputs_embeds",
        "to": "talker.inputs_embeds",
        "dtype": "fp32",
        "device_transfer": False,
    } in pipeline["dataflow"]


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


class _GraphlessModel:
    """Stand-in for a built component whose ONNX graph was streamed to disk.

    Large decoders built with external-data / streamed weights do not retain
    the ``ir.Graph`` in memory, so the in-memory package value exposes no
    ``.graph`` attribute. The port contract must instead be derived from the
    ``model.onnx`` written next to the sidecar.
    """


class _GraphlessPkg(dict):
    pass


def _ir_value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _ir_decoder_model(
    inputs: list[ir.Value],
    output_specs: list[tuple[str, ir.DataType, list[int | str]]],
) -> ir.Model:
    outputs = [_ir_value(*spec) for spec in output_specs]
    nodes = [
        ir.Node("", "Identity", [inputs[0]], outputs=[out], name=f"emit_{out.name}")
        for out in outputs
    ]
    graph = ir.Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        name="decoder",
        opset_imports={"": 21},
    )
    return ir.Model(graph, ir_version=10)


@dataclasses.dataclass
class _HybridCfg(_Cfg):
    # A hybrid decoder mirroring Qwen3.6-27B: linear-attention layers carrying
    # conv_state + recurrent_state interleaved with grouped-query-attention KV
    # layers. This is the shape that shipped thin metadata (io block dropped).
    layer_types: list[str] | None = dataclasses.field(
        default_factory=lambda: [
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
    )


def _write_hybrid_decoder_onnx(directory: str) -> None:
    inputs = [
        _ir_value("input_ids", ir.DataType.INT64, ["batch", "sequence"]),
        _ir_value("attention_mask", ir.DataType.INT64, ["batch", "past_sequence + sequence"]),
        _ir_value("position_ids", ir.DataType.INT64, ["batch", "sequence"]),
    ]
    output_specs = [("logits", ir.DataType.FLOAT, ["batch", "sequence", 128])]
    # Layers 0-1: linear attention (conv_state + recurrent_state replace-state).
    for layer in (0, 1):
        inputs.extend(
            [
                _ir_value(
                    f"past_key_values.{layer}.conv_state",
                    ir.DataType.FLOAT,
                    ["batch", 10240, 3],
                ),
                _ir_value(
                    f"past_key_values.{layer}.recurrent_state",
                    ir.DataType.FLOAT,
                    ["batch", 48, 128, 128],
                ),
            ]
        )
        output_specs.extend(
            [
                (f"present.{layer}.conv_state", ir.DataType.FLOAT, ["batch", 10240, 3]),
                (
                    f"present.{layer}.recurrent_state",
                    ir.DataType.FLOAT,
                    ["batch", 48, 128, 128],
                ),
            ]
        )
    # Layer 2: grouped-query attention (KV append cache).
    inputs.extend(
        [
            _ir_value(
                "past_key_values.2.key", ir.DataType.FLOAT, ["batch", 4, "past_sequence", 256]
            ),
            _ir_value(
                "past_key_values.2.value",
                ir.DataType.FLOAT,
                ["batch", 4, "past_sequence", 256],
            ),
        ]
    )
    output_specs.extend(
        [
            ("present.2.key", ir.DataType.FLOAT, ["batch", 4, "total_sequence", 256]),
            ("present.2.value", ir.DataType.FLOAT, ["batch", 4, "total_sequence", 256]),
        ]
    )
    model = _ir_decoder_model(inputs, output_specs)
    ir.save(model, os.path.join(directory, "model.onnx"))


def test_graphless_decoder_reloads_io_from_disk(tmp_path):
    # Regression: when the in-memory package value lacks `.graph` (external-data /
    # streamed build), the sidecar MUST still gain its explicit `model.io` port
    # contract by reloading `model.onnx` from disk — never ship thin metadata.
    _write_hybrid_decoder_onnx(str(tmp_path))
    pkg = _GraphlessPkg({"model": _GraphlessModel()})
    assert not hasattr(pkg["model"], "graph")

    arts = write_onnx_genai_config(pkg, str(tmp_path), config=_HybridCfg())
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)

    io = meta["model"]["io"]
    assert io["kv_inputs"] == ["past_key_values.2.key", "past_key_values.2.value"]
    assert io["kv_outputs"] == ["present.2.key", "present.2.value"]
    # The linear-attention layers' conv/recurrent state are replace-state pairs,
    # locking the 27B hybrid scenario.
    state_inputs = {pair["input"] for pair in io["state_pairs"]}
    assert state_inputs == {
        "past_key_values.0.conv_state",
        "past_key_values.0.recurrent_state",
        "past_key_values.1.conv_state",
        "past_key_values.1.recurrent_state",
    }
    assert all(pair["update"] == "replace" for pair in io["state_pairs"])
    assert io["token_input"] == "input_ids"
    assert io["logits_output"] == "logits"


def test_graphless_decoder_without_disk_model_warns_and_skips(tmp_path, caplog):
    # If the graph is unavailable in memory AND absent on disk, we must not ship
    # thin metadata silently — emit a loud warning and leave the sidecar without
    # the `io` block (downstream runtimes derive it from the graph at load).
    import logging

    pkg = _GraphlessPkg({"model": _GraphlessModel()})
    with caplog.at_level(logging.WARNING):
        arts = write_onnx_genai_config(pkg, str(tmp_path), config=_Cfg())

    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    assert "io" not in meta.get("model", {})
    assert any("model.io" in rec.message for rec in caplog.records)


def test_in_memory_graph_is_not_reloaded_from_disk(tmp_path, monkeypatch):
    # When `.graph` is present in memory, we must derive io directly and never
    # touch the disk reload path (preserving existing fast-path behavior).
    from mobius.integrations.onnx_genai import auto_export

    def _boom(_model_path):
        raise AssertionError("disk reload must not run when .graph is present")

    monkeypatch.setattr(auto_export, "_load_graph_from_disk", _boom)

    inputs = [
        _ir_value("input_ids", ir.DataType.INT64, ["batch", "sequence"]),
        _ir_value("attention_mask", ir.DataType.INT64, ["batch", "past_sequence + sequence"]),
        _ir_value("position_ids", ir.DataType.INT64, ["batch", "sequence"]),
        _ir_value(
            "past_key_values.0.key", ir.DataType.FLOAT, ["batch", 4, "past_sequence", 256]
        ),
        _ir_value(
            "past_key_values.0.value", ir.DataType.FLOAT, ["batch", 4, "past_sequence", 256]
        ),
    ]
    output_specs = [
        ("logits", ir.DataType.FLOAT, ["batch", "sequence", 128]),
        ("present.0.key", ir.DataType.FLOAT, ["batch", 4, "total_sequence", 256]),
        ("present.0.value", ir.DataType.FLOAT, ["batch", 4, "total_sequence", 256]),
    ]
    pkg = _GraphlessPkg({"model": _ir_decoder_model(inputs, output_specs)})

    arts = write_onnx_genai_config(pkg, str(tmp_path), config=_Cfg())
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    assert meta["model"]["io"]["kv_inputs"] == [
        "past_key_values.0.key",
        "past_key_values.0.value",
    ]
