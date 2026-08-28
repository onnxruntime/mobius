# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnxscript import OpBuilder, nn

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.integrations.gguf._mmproj_mapping import (
    map_mmproj_audio_projector_to_onnx,
)
from mobius.integrations.gguf._mmproj_registry import get_projector_spec
from mobius.integrations.onnx_genai import write_onnx_genai_config
from mobius.integrations.ort_genai import write_ort_genai_config
from mobius.models.gguf_audio_projector import (
    AUDIO_PROCESSOR_ABIS,
    _GeluProjector,
    _GraniteSpeechAttention,
    _LFM2AudioAdapter,
    _MimoRVQBridge,
    _PocketCausalConv1d,
    _SquaredReLUProjector,
    _UltravoxProjector,
    create_gguf_audio_projector,
)
from mobius.tasks import (
    GGUFAudioProjectorModel,
    GGUFAudioProjectorTask,
    GGUFSpeakerProjectorModel,
    GGUFSpeakerProjectorTask,
)


@dataclass(frozen=True)
class _RouteCase:
    metadata: dict[str, object]
    shapes: dict[str, tuple[int, ...]]
    inputs: dict[str, np.ndarray]
    expected_shape: tuple[int, int]


def _base_metadata(
    *,
    hidden: int,
    intermediate: int,
    layers: int,
    heads: int,
    mel_bins: int,
) -> dict[str, object]:
    return {
        "clip.audio.embedding_length": hidden,
        "clip.audio.feed_forward_length": intermediate,
        "clip.audio.block_count": layers,
        "clip.audio.attention.head_count": heads,
        "clip.audio.attention.layer_norm_epsilon": 1e-5,
        "clip.audio.num_mel_bins": mel_bins,
    }


def _whisper_case(projector_type: str) -> _RouteCase:
    metadata = _base_metadata(
        hidden=8,
        intermediate=16,
        layers=1,
        heads=2,
        mel_bins=4,
    )
    if projector_type in {"ultravox", "voxtral"}:
        metadata["clip.audio.projector.stack_factor"] = 2
    projector_input = 16 if projector_type in {"ultravox", "voxtral"} else 8
    first_output = 24 if projector_type == "ultravox" else 10
    second_input = first_output // 2 if projector_type == "ultravox" else first_output
    shapes = {
        "a.conv1d.1.weight": (8, 4, 3),
        "a.conv1d.2.weight": (8, 8, 3),
        "a.position_embd.weight": (32, 8),
        "mm.a.mlp.1.weight": (first_output, projector_input),
        "mm.a.mlp.2.weight": (5, second_input),
    }
    if projector_type == "musicflamingo":
        shapes["mm.a.mlp.1.bias"] = (first_output,)
        shapes["mm.a.mlp.2.bias"] = (5,)
    output_frames = {
        "ultravox": 4,
        "voxtral": 2,
        "musicflamingo": 4,
    }[projector_type]
    return _RouteCase(
        metadata,
        shapes,
        {"input_features": np.linspace(-1, 1, 64, dtype=np.float32).reshape(16, 4)},
        (output_frames, 5),
    )


def _lfm2a_case() -> _RouteCase:
    metadata = _base_metadata(
        hidden=16,
        intermediate=16,
        layers=1,
        heads=2,
        mel_bins=8,
    )
    shapes = {
        "a.conv1d.0.weight": (4, 1, 3, 3),
        "a.blk.0.ffn_up.weight": (32, 16),
        "a.blk.0.conv_dw.weight": (16, 5),
        "a.position_embd.weight": (64, 20),
        "mm.a.mlp.1.weight": (24, 16),
        "mm.a.mlp.3.weight": (20, 24),
    }
    return _RouteCase(
        metadata,
        shapes,
        {"input_features": np.linspace(-1, 1, 128, dtype=np.float32).reshape(16, 8)},
        (2, 20),
    )


def _parakeet_case() -> _RouteCase:
    metadata = _base_metadata(
        hidden=16,
        intermediate=32,
        layers=1,
        heads=2,
        mel_bins=8,
    )
    metadata.update(
        {
            "clip.audio.subsampling_factor": 8,
            "clip.audio.conv_kernel_size": 5,
        }
    )
    shapes = {
        "a.conv1d.0.weight": (4, 1, 3, 3),
        "mm.a.mlp.1.weight": (24, 16),
        "mm.a.mlp.2.weight": (20, 24),
    }
    return _RouteCase(
        metadata,
        shapes,
        {"input_features": np.linspace(-1, 1, 128, dtype=np.float32).reshape(16, 8)},
        (2, 20),
    )


def _granite_case() -> _RouteCase:
    metadata = _base_metadata(
        hidden=8,
        intermediate=16,
        layers=2,
        heads=2,
        mel_bins=4,
    )
    metadata.update(
        {
            "clip.audio.chunk_size": 4,
            "clip.audio.max_pos_emb": 4,
            "clip.audio.projector.window_size": 4,
            "clip.audio.projector.downsample_rate": 2,
            "clip.audio.projector.head_count": 2,
        }
    )
    shapes = {
        "a.input_projection.weight": (8, 4),
        "a.blk.0.ffn_up.weight": (16, 8),
        "a.blk.0.attn_rel_pos_emb": (9, 4),
        "a.blk.0.conv_dw.weight": (12, 3),
        "a.enc_ctc_out.weight": (5, 8),
        "a.enc_ctc_out_mid.weight": (8, 5),
        "a.proj_query": (1, 2, 8),
        "a.proj_blk.0.ffn_up.weight": (16, 8),
        "a.proj_blk.0.self_attn_q.weight": (8, 8),
        "a.proj_blk.1.self_attn_q.weight": (8, 8),
        "a.proj_linear.weight": (6, 8),
    }
    return _RouteCase(
        metadata,
        shapes,
        {"input_features": np.linspace(-1, 1, 20, dtype=np.float32).reshape(5, 4)},
        (4, 6),
    )


def _mimo_case() -> _RouteCase:
    metadata = _base_metadata(
        hidden=8,
        intermediate=16,
        layers=3,
        heads=2,
        mel_bins=4,
    )
    metadata.update(
        {
            "clip.audio.window_size": 4,
            "clip.audio.wa_pattern_mode": [0, 0, -1],
            "clip.audio.local_group_size": 2,
            "clip.audio.local_block_count": 1,
            "clip.audio.rvq.num_quantizers": 2,
            "clip.audio.rvq.codebook_size": [5, 6],
        }
    )
    shapes = {
        "a.conv1d.1.weight": (8, 4, 3),
        "a.conv1d.2.weight": (8, 8, 3),
        "a.blk.0.ffn_up.weight": (16, 8),
        "a.downsample.conv.weight": (8, 8, 2),
        "a.rvq.codebook.weight": (2, 6, 8),
        "mm.a.code_embd.weight": (2, 6, 8),
        "mm.a.local_blk.0.ffn_up.weight": (12, 8),
        "mm.a.mlp.1.weight": (10, 16),
        "mm.a.mlp.2.weight": (6, 10),
    }
    for layer in range(3):
        for stem in ("attn_q", "attn_k", "attn_v", "attn_out"):
            shapes[f"a.blk.{layer}.{stem}.weight"] = (8, 8)
        for stem in ("attn_q", "attn_v", "attn_out"):
            shapes[f"a.blk.{layer}.{stem}.bias"] = (8,)
    for stem in ("attn_q", "attn_k", "attn_v"):
        shapes[f"mm.a.local_blk.0.{stem}.weight"] = (8, 8)
        shapes[f"mm.a.local_blk.0.{stem}.bias"] = (8,)
    shapes.update(
        {
            "mm.a.local_blk.0.attn_out.weight": (8, 8),
            "mm.a.local_blk.0.ffn_gate.weight": (12, 8),
            "mm.a.local_blk.0.ffn_down.weight": (8, 12),
            "mm.a.local_blk.0.ln1.weight": (8,),
            "mm.a.local_blk.0.ln2.weight": (8,),
        }
    )
    return _RouteCase(
        metadata,
        shapes,
        {"input_features": np.linspace(-1, 1, 64, dtype=np.float32).reshape(16, 4)},
        (2, 6),
    )


def _pockettts_case() -> _RouteCase:
    metadata = _base_metadata(
        hidden=64,
        intermediate=128,
        layers=2,
        heads=1,
        mel_bins=1,
    )
    shapes = {
        "a.seanet.conv_in.weight": (8, 1, 7),
        "a.seanet.conv_in.bias": (8,),
        "a.seanet.conv_out.weight": (64, 64, 3),
        "a.seanet.conv_out.bias": (64,),
        "a.downsample.conv.weight": (32, 64, 32),
        "a.speaker_proj.weight": (48, 32),
    }
    for index, (channels, kernel) in enumerate(zip((8, 16, 32), (8, 10, 12))):
        shapes.update(
            {
                f"a.seanet.blk.{index}.res_conv1.weight": (
                    channels // 2,
                    channels,
                    3,
                ),
                f"a.seanet.blk.{index}.res_conv1.bias": (channels // 2,),
                f"a.seanet.blk.{index}.res_conv2.weight": (
                    channels,
                    channels // 2,
                    1,
                ),
                f"a.seanet.blk.{index}.res_conv2.bias": (channels,),
                f"a.seanet.blk.{index}.scale_conv.weight": (
                    channels * 2,
                    channels,
                    kernel,
                ),
                f"a.seanet.blk.{index}.scale_conv.bias": (channels * 2,),
            }
        )
    return _RouteCase(
        metadata,
        shapes,
        {"input_values": np.linspace(-0.1, 0.1, 1920, dtype=np.float32)},
        (1, 48),
    )


def _case(projector_type: str) -> _RouteCase:
    if projector_type in {"ultravox", "voxtral", "musicflamingo"}:
        return _whisper_case(projector_type)
    return {
        "lfm2a": _lfm2a_case,
        "parakeet": _parakeet_case,
        "granite_speech": _granite_case,
        "mimo_audio": _mimo_case,
        "pockettts_spkenc": _pockettts_case,
    }[projector_type]()


def _build(
    projector_type: str,
    *,
    dtype: ir.DataType = ir.DataType.FLOAT,
):
    case = _case(projector_type)
    module = create_gguf_audio_projector(
        projector_type,
        case.metadata,
        case.shapes,
    )
    hidden = int(case.metadata["clip.audio.embedding_length"])
    heads = int(case.metadata["clip.audio.attention.head_count"])
    config = ArchitectureConfig(
        model_type=f"gguf_{projector_type}",
        vocab_size=1,
        hidden_size=hidden,
        intermediate_size=max(
            hidden,
            int(case.metadata["clip.audio.feed_forward_length"]),
        ),
        num_hidden_layers=int(case.metadata["clip.audio.block_count"]),
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=hidden // heads,
        max_position_embeddings=65_536,
        dtype=dtype,
    )
    if projector_type == "pockettts_spkenc":
        speaker_module = GGUFSpeakerProjectorModel(
            module,
            output_name="speaker_features",
        )
        return case, build_from_module(
            speaker_module,
            config,
            task=GGUFSpeakerProjectorTask(),
        )["speaker_encoder"]
    return case, build_from_module(
        GGUFAudioProjectorModel(module),
        config,
        task=GGUFAudioProjectorTask(),
    )["audio_encoder"]


_PROJECTOR_TYPES = (
    "granite_speech",
    "lfm2a",
    "mimo_audio",
    "musicflamingo",
    "parakeet",
    "ultravox",
    "voxtral",
    "pockettts_spkenc",
)


@pytest.mark.parametrize("projector_type", _PROJECTOR_TYPES)
def test_audio_projector_graph_builds_with_route_specific_abi(projector_type: str):
    case, model = _build(projector_type)

    assert [value.name for value in model.graph.inputs] == list(case.inputs)
    assert model.graph.inputs[0].dtype == ir.DataType.FLOAT
    output_name = (
        "speaker_features" if projector_type == "pockettts_spkenc" else "audio_features"
    )
    assert [value.name for value in model.graph.outputs] == [output_name]
    assert model.graph.outputs[0].dtype == ir.DataType.FLOAT


@pytest.mark.parametrize("projector_type", _PROJECTOR_TYPES)
def test_reduced_precision_graph_keeps_float32_processor_boundary(
    projector_type: str,
):
    _, model = _build(projector_type, dtype=ir.DataType.FLOAT16)

    assert model.graph.inputs[0].dtype == ir.DataType.FLOAT
    parameter_dtypes = {
        initializer.dtype
        for name, initializer in model.graph.initializers.items()
        if name.startswith(("audio_encoder.", "speaker_encoder."))
        and not name.endswith(("cos_cache", "sin_cache"))
    }
    assert parameter_dtypes == {ir.DataType.FLOAT16}


@pytest.mark.parametrize("projector_type", _PROJECTOR_TYPES)
def test_audio_projector_tensor_map_covers_every_graph_parameter(projector_type: str):
    case = _case(projector_type)
    module = create_gguf_audio_projector(
        projector_type,
        case.metadata,
        case.shapes,
    )
    if projector_type == "pockettts_spkenc":
        mapped_module: nn.Module = GGUFSpeakerProjectorModel(module)
    else:
        mapped_module = GGUFAudioProjectorModel(module)
    parameters = {
        name
        for name, _ in mapped_module.named_parameters()
        if not name.endswith(("rotary_emb.cos_cache", "rotary_emb.sin_cache"))
    }
    spec = get_projector_spec(projector_type)
    source_names = set(spec.required_top_tensors)
    source_names.update(name for name in spec.optional_top_tensors if name in case.shapes)
    source_names.update(
        f"{spec.block_prefix}{layer}.{suffix}"
        for layer in range(int(case.metadata["clip.audio.block_count"]))
        for suffix in spec.block_suffixes
    )
    if projector_type == "mimo_audio":
        source_names.update(name for name in case.shapes if name.startswith("mm.a.local_blk."))
    mapped = {
        target
        for name in source_names
        if (
            target := map_mmproj_audio_projector_to_onnx(
                name,
                projector_type,
            )
        )
        is not None
    }

    assert mapped == parameters


@pytest.mark.parametrize("projector_type", _PROJECTOR_TYPES)
def test_audio_projector_graph_executes_nonzero_features(projector_type: str):
    case, session = _materialized_session(projector_type)
    (actual,) = session.run(None, case.inputs)

    assert actual.shape == case.expected_shape
    assert np.isfinite(actual).all()
    assert np.count_nonzero(actual) > 0


def _materialized_session(
    projector_type: str,
) -> tuple[_RouteCase, ort.InferenceSession]:
    case, model = _build(projector_type)
    rng = np.random.default_rng(sum(projector_type.encode()))
    for name, initializer in model.graph.initializers.items():
        if initializer.const_value is not None:
            continue
        shape = tuple(int(dim) for dim in initializer.shape)
        if name.endswith("running_var"):
            values = np.ones(shape, dtype=np.float32)
        elif any(
            token in name
            for token in (
                "layernorm.weight",
                "layer_norm.weight",
                "norm.weight",
                "norm1.weight",
                "norm2.weight",
                "norm_pre.weight",
                "norm_mid.weight",
                "scale",
            )
        ):
            values = np.ones(shape, dtype=np.float32)
        else:
            values = (rng.standard_normal(shape) * 0.02).astype(np.float32)
        initializer.const_value = ir.tensor(values)

    session = ort.InferenceSession(
        ir.serde.serialize_model(model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return case, session


_ORT_INPUT_ERRORS = (
    ort.capi.onnxruntime_pybind11_state.Fail,
    ort.capi.onnxruntime_pybind11_state.InvalidArgument,
    ort.capi.onnxruntime_pybind11_state.RuntimeException,
)


@pytest.mark.parametrize(
    ("projector_type", "feeds"),
    [
        (
            "ultravox",
            {"input_features": np.zeros((65, 4), dtype=np.float32)},
        ),
        (
            "lfm2a",
            {"input_features": np.zeros((513, 8), dtype=np.float32)},
        ),
        (
            "pockettts_spkenc",
            {"input_values": np.zeros((1919,), dtype=np.float32)},
        ),
        (
            "pockettts_spkenc",
            {"input_values": np.zeros((30 * 24_000 + 1,), dtype=np.float32)},
        ),
    ],
)
def test_audio_projector_rejects_unsupported_frame_contracts(
    projector_type: str,
    feeds: dict[str, np.ndarray],
):
    _, session = _materialized_session(projector_type)

    with pytest.raises(_ORT_INPUT_ERRORS, match="indices element out of data bounds"):
        session.run(None, feeds)


def test_audio_processor_abis_preserve_sample_channel_and_frame_contracts():
    assert tuple(AUDIO_PROCESSOR_ABIS) == (
        "ultravox",
        "voxtral",
        "musicflamingo",
        "lfm2a",
        "granite_speech",
        "parakeet",
        "mimo_audio",
        "pockettts_spkenc",
        "meralion",
    )
    assert AUDIO_PROCESSOR_ABIS["ultravox"].sample_rate == 16_000
    assert AUDIO_PROCESSOR_ABIS["voxtral"].chunk_seconds == 30
    assert AUDIO_PROCESSOR_ABIS["musicflamingo"].n_fft == 400
    assert AUDIO_PROCESSOR_ABIS["lfm2a"].n_fft == 512
    assert AUDIO_PROCESSOR_ABIS["granite_speech"].graph_layout.endswith(",160]")
    assert AUDIO_PROCESSOR_ABIS["parakeet"].channels == 1
    assert AUDIO_PROCESSOR_ABIS["mimo_audio"].sample_rate == 24_000
    assert AUDIO_PROCESSOR_ABIS["pockettts_spkenc"].frame_multiple == 1_920
    assert AUDIO_PROCESSOR_ABIS["pockettts_spkenc"].max_seconds == 30
    with pytest.raises(TypeError):
        AUDIO_PROCESSOR_ABIS["ultravox"] = AUDIO_PROCESSOR_ABIS["voxtral"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("projector_type", "component"),
    [
        ("ultravox", "audio_encoder"),
        ("pockettts_spkenc", "speaker_encoder"),
    ],
)
def test_standalone_runtime_exports_are_advisory(
    tmp_path,
    projector_type: str,
    component: str,
):
    _, model = _build(projector_type)
    config = ArchitectureConfig(
        model_type=f"gguf_{projector_type}",
        vocab_size=1,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
    )
    package = ModelPackage({component: model}, config=config)
    package.gguf_projector_type = projector_type  # type: ignore[attr-defined]
    model.metadata_props["mobius.processor_abi"] = json.dumps(
        asdict(AUDIO_PROCESSOR_ABIS[projector_type]),
        sort_keys=True,
        separators=(",", ":"),
    )

    onnx_paths = write_onnx_genai_config(package, str(tmp_path / "onnx-genai"))
    onnx_metadata = json.loads(
        (tmp_path / "onnx-genai" / "inference_metadata.yaml").read_text()
    )
    assert set(onnx_paths) == {"inference_metadata", "runtime_compatibility"}
    assert onnx_metadata["components"][component]["inputs"] == [
        value.name for value in model.graph.inputs
    ]
    assert onnx_metadata["components"][component]["outputs"] == [
        value.name for value in model.graph.outputs
    ]
    assert "mobius.processor_abi" in onnx_metadata["components"][component]["metadata"]

    ort_paths = write_ort_genai_config(package, str(tmp_path / "ort-genai"))
    ort_metadata = json.loads(
        (tmp_path / "ort-genai" / "runtime_compatibility.json").read_text()
    )
    assert set(ort_paths) == {"runtime_compatibility"}
    assert not (tmp_path / "ort-genai" / "genai_config.json").exists()
    assert ort_metadata["runtime_validation_status"] == "unsupported-by-tested-runtime"
    assert set(ort_metadata["graph_contract"]) == {component}


def test_granite_feature_capture_precedes_midpoint_ctc_injection():
    case = _granite_case()
    case.metadata["clip.audio.feature_layer"] = [1]
    module = create_gguf_audio_projector(
        "granite_speech",
        case.metadata,
        case.shapes,
    )
    config = ArchitectureConfig(
        model_type="gguf_granite_feature_capture_test",
        vocab_size=1,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    graph = build_from_module(
        GGUFAudioProjectorModel(module),
        config,
        task=GGUFAudioProjectorTask(),
    )["audio_encoder"].graph
    feature_concat = next(
        node
        for node in graph
        if node.op_type == "Concat"
        and node.attributes["axis"].value == -1
        and any("output_norm" in value.name for value in node.inputs)
    )

    assert "layers.0.output_norm" in feature_concat.inputs[0].name


class _ComponentEncoder(nn.Module):
    def __init__(self, component: nn.Module, input_size: int):
        super().__init__()
        self.component = component
        self.input_schema = (
            (
                "input_features",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("frames"), input_size),
            ),
        )

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        return self.component(op, input_features)


class _BatchedComponentEncoder(_ComponentEncoder):
    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        output = self.component(op, op.Unsqueeze(input_features, [0]))
        return op.Squeeze(output, [0])


class _WaveConvEncoder(nn.Module):
    def __init__(self, component: nn.Module):
        super().__init__()
        self.component = component
        self.input_schema = (
            (
                "input_values",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("samples"),),
            ),
        )

    def forward(self, op: OpBuilder, input_values: ir.Value) -> ir.Value:
        values = op.Unsqueeze(op.Unsqueeze(input_values, [0]), [0])
        return op.Squeeze(self.component(op, values), [0, 1])


def _run_component(
    encoder: nn.Module,
    feeds: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    model = GGUFAudioProjectorModel(encoder)
    config = ArchitectureConfig(
        model_type="gguf_audio_component_test",
        vocab_size=1,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
    )
    graph_model = build_from_module(
        model,
        config,
        task=GGUFAudioProjectorTask(),
    )["audio_encoder"]
    prefix = "audio_encoder.component."
    for name, initializer in graph_model.graph.initializers.items():
        if initializer.const_value is not None:
            continue
        local_name = name.removeprefix(prefix)
        if local_name not in weights:
            raise AssertionError(f"Missing independent test weight for {name}")
        initializer.const_value = ir.tensor(weights[local_name].astype(np.float32))
    session = ort.InferenceSession(
        ir.serde.serialize_model(graph_model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return session.run(None, feeds)[0]


def _rms_norm(value: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps) * weight


def test_ultravox_projector_matches_independent_swapped_swiglu_reference():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((3, 4), dtype=np.float32)
    weights = {
        "norm_pre.weight": rng.standard_normal(4, dtype=np.float32),
        "linear_1.weight": rng.standard_normal((6, 4), dtype=np.float32),
        "norm_mid.weight": rng.standard_normal(3, dtype=np.float32),
        "linear_2.weight": rng.standard_normal((2, 3), dtype=np.float32),
    }
    actual = _run_component(
        _ComponentEncoder(_UltravoxProjector(4, 6, 2), 4),
        {"input_features": values},
        weights,
    )

    hidden = _rms_norm(values, weights["norm_pre.weight"])
    expanded = hidden @ weights["linear_1.weight"].T
    first, second = np.split(expanded, 2, axis=-1)
    hidden = first * (second / (1.0 + np.exp(-second)))
    hidden = _rms_norm(hidden, weights["norm_mid.weight"])
    expected = hidden @ weights["linear_2.weight"].T
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("projector_type", "with_bias"),
    [("voxtral", False), ("musicflamingo", True)],
)
def test_whisper_gelu_projectors_match_independent_torch_reference(
    projector_type: str,
    with_bias: bool,
):
    rng = np.random.default_rng(sum(projector_type.encode()))
    values = rng.standard_normal((3, 4), dtype=np.float32)
    weights = {
        "linear_1.weight": rng.standard_normal((5, 4), dtype=np.float32),
        "linear_2.weight": rng.standard_normal((2, 5), dtype=np.float32),
    }
    if with_bias:
        weights["linear_1.bias"] = rng.standard_normal(5, dtype=np.float32)
        weights["linear_2.bias"] = rng.standard_normal(2, dtype=np.float32)
    actual = _run_component(
        _ComponentEncoder(
            _GeluProjector(
                4,
                5,
                2,
                first_bias=with_bias,
                second_bias=with_bias,
            ),
            4,
        ),
        {"input_features": values},
        weights,
    )

    hidden = values @ weights["linear_1.weight"].T
    if with_bias:
        hidden += weights["linear_1.bias"]
    hidden = torch.nn.functional.gelu(torch.from_numpy(hidden)).numpy()
    expected = hidden @ weights["linear_2.weight"].T
    if with_bias:
        expected += weights["linear_2.bias"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_parakeet_projector_matches_independent_squared_relu_reference():
    rng = np.random.default_rng(3)
    values = rng.standard_normal((3, 4), dtype=np.float32)
    weights = {
        "norm_pre.weight": rng.standard_normal(4, dtype=np.float32),
        "linear_1.weight": rng.standard_normal((5, 4), dtype=np.float32),
        "linear_2.weight": rng.standard_normal((2, 5), dtype=np.float32),
    }
    actual = _run_component(
        _ComponentEncoder(
            _SquaredReLUProjector(
                4,
                5,
                2,
                first_bias=False,
                second_bias=False,
            ),
            4,
        ),
        {"input_features": values},
        weights,
    )
    hidden = _rms_norm(values, weights["norm_pre.weight"])
    hidden = np.maximum(hidden @ weights["linear_1.weight"].T, 0.0) ** 2
    expected = hidden @ weights["linear_2.weight"].T
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_lfm2a_adapter_matches_independent_layernorm_gelu_reference():
    rng = np.random.default_rng(4)
    values = rng.standard_normal((3, 4), dtype=np.float32)
    weights = {
        "norm.weight": rng.standard_normal(4, dtype=np.float32),
        "norm.bias": rng.standard_normal(4, dtype=np.float32),
        "linear_1.weight": rng.standard_normal((5, 4), dtype=np.float32),
        "linear_1.bias": rng.standard_normal(5, dtype=np.float32),
        "linear_2.weight": rng.standard_normal((2, 5), dtype=np.float32),
        "linear_2.bias": rng.standard_normal(2, dtype=np.float32),
    }
    actual = _run_component(
        _ComponentEncoder(_LFM2AudioAdapter(4, 5, 2, 1e-5), 4),
        {"input_features": values},
        weights,
    )
    expected = torch.nn.functional.layer_norm(
        torch.from_numpy(values),
        (4,),
        torch.from_numpy(weights["norm.weight"]),
        torch.from_numpy(weights["norm.bias"]),
        1e-5,
    )
    expected = torch.nn.functional.gelu(
        expected @ torch.from_numpy(weights["linear_1.weight"]).T
        + torch.from_numpy(weights["linear_1.bias"])
    )
    expected = (
        expected @ torch.from_numpy(weights["linear_2.weight"]).T
        + torch.from_numpy(weights["linear_2.bias"])
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_granite_speech_shaw_attention_matches_independent_numpy_reference():
    rng = np.random.default_rng(5)
    values = rng.standard_normal((2, 4), dtype=np.float32)
    weights = {
        "relative_positions": rng.standard_normal((5, 2), dtype=np.float32),
        "q_proj.weight": rng.standard_normal((4, 4), dtype=np.float32),
        "k_proj.weight": rng.standard_normal((4, 4), dtype=np.float32),
        "v_proj.weight": rng.standard_normal((4, 4), dtype=np.float32),
        "out_proj.weight": rng.standard_normal((4, 4), dtype=np.float32),
        "out_proj.bias": rng.standard_normal(4, dtype=np.float32),
    }
    actual = _run_component(
        _BatchedComponentEncoder(
            _GraniteSpeechAttention(4, 2, 2, 2, 5),
            4,
        ),
        {"input_features": values},
        weights,
    )

    query = values @ weights["q_proj.weight"].T
    key = values @ weights["k_proj.weight"].T
    value = values @ weights["v_proj.weight"].T
    query = query.reshape(1, 1, 2, 2, 2).transpose(0, 1, 3, 2, 4)
    key = key.reshape(1, 1, 2, 2, 2).transpose(0, 1, 3, 2, 4)
    value = value.reshape(1, 1, 2, 2, 2).transpose(0, 1, 3, 2, 4)
    distance_indices = np.array([[2, 1], [3, 2]])
    relative = weights["relative_positions"][distance_indices]
    scores = query @ key.transpose(0, 1, 2, 4, 3)
    scores += np.einsum("bnhqd,qkd->bnhqk", query, relative)
    scores *= 2**-0.5
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = probabilities @ value
    expected = expected.transpose(0, 1, 3, 2, 4).reshape(2, 4)
    expected = expected @ weights["out_proj.weight"].T + weights["out_proj.bias"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_mimo_rvq_bridge_matches_independent_residual_nearest_neighbor_reference():
    values = np.array([[0.9, 0.1], [-0.1, 1.1]], dtype=np.float32)
    codebook = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 0.0], [0.2, 0.2], [-0.2, -0.2]],
        ],
        dtype=np.float32,
    )
    embeddings = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    actual = _run_component(
        _ComponentEncoder(_MimoRVQBridge((2, 3, 2), (2, 3, 2), (3, 3)), 2),
        {"input_features": values},
        {"codebook": codebook, "code_embeddings": embeddings},
    )

    residual = values.copy()
    expected = np.zeros_like(values)
    for quantizer in range(2):
        scores = 2.0 * residual @ codebook[quantizer].T
        scores -= np.sum(codebook[quantizer] ** 2, axis=-1)
        codes = np.argmax(scores, axis=-1)
        residual -= codebook[quantizer][codes]
        expected += embeddings[quantizer][codes]
    np.testing.assert_array_equal(actual, expected)


def test_pockettts_causal_convolution_matches_independent_torch_reference():
    values = np.arange(8, dtype=np.float32)
    weight = np.array([[[1.0, -0.5, 0.25]]], dtype=np.float32)
    bias = np.array([0.1], dtype=np.float32)
    actual = _run_component(
        _WaveConvEncoder(_PocketCausalConv1d((1, 1, 3), stride=2, bias=True)),
        {"input_values": values},
        {"weight": weight, "bias": bias},
    )
    padded = torch.nn.functional.pad(torch.from_numpy(values)[None, None], (1, 0))
    expected = torch.nn.functional.conv1d(
        padded,
        torch.from_numpy(weight),
        torch.from_numpy(bias),
        stride=2,
    ).numpy()[0, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
