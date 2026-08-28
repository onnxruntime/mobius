# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for spectral speech-enhancement onnx-genai metadata."""

from __future__ import annotations

import json
import os

import numpy as np
import onnx_ir as ir
import pytest

from mobius import build_from_module
from mobius.integrations.onnx_genai import (
    build_speech_enhancement_workflow_metadata,
    write_onnx_genai_config,
    write_speech_enhancement_workflow_metadata,
)
from mobius.integrations.onnx_genai.auto_export import _looks_like_speech_enhancement
from mobius.integrations.onnx_genai.inference_metadata_test import (
    _onnx_genai_schema_path,
)
from mobius.models.reuse import ReUseConfig, SEMambaSpeechEnhancementModel

_TINY = {
    "model_cfg": {
        "hid_feature": 8,
        "num_tfmamba": 1,
        "d_state": 4,
        "expand": 2,
        "compress_factor": "relu_log1p",
    },
    "stft_cfg": {"n_fft": 32, "hop_size": 4, "win_size": 32, "sampling_rate": 8000},
}


def _package(**config_overrides):
    config = ReUseConfig.from_json(_TINY)
    for key, value in config_overrides.items():
        setattr(config, key, value)
    module = SEMambaSpeechEnhancementModel(config)
    return build_from_module(module, config, task="speech-enhancement"), config


def _runtime_package(**config_overrides):
    config = ReUseConfig(
        hid_feature=4,
        num_tfmamba=1,
        d_state=2,
        d_conv=2,
        expand=2,
        n_fft=320,
        hop_size=40,
        win_size=320,
        sampling_rate=8000,
        **config_overrides,
    )
    module = SEMambaSpeechEnhancementModel(config)
    rng = np.random.default_rng(0)
    for _name, parameter in module.named_parameters():
        shape = [
            dimension if isinstance(dimension, int) else 1 for dimension in parameter.shape
        ]
        parameter.const_value = ir.tensor(
            (rng.standard_normal(shape) * 0.02).astype(np.float32)
        )
    package = build_from_module(module, config, task="speech-enhancement")
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(
        ir.to_proto(package["model"]).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return package, config, session


def _even_scaled(value: int, sample_rate: int, reference_rate: int) -> int:
    scaled = value * sample_rate // reference_rate
    return scaled if scaled % 2 == 0 else scaled + 1


def _run_declared_workflow(package, config, session, waveform, source_rate):
    """Execute the declared native/BWE transform, graph, and inverse contract."""
    torch = pytest.importorskip("torch")
    metadata = build_speech_enhancement_workflow_metadata(package, config)
    transforms = metadata["preprocessing"]["audio"]["transforms"]
    samples = np.asarray(waveform, dtype=np.float32)
    sample_rate = source_rate

    for transform in transforms:
        op = transform["op"]
        if op in {"decode", "downmix", "log1p"}:
            continue
        if op == "require_sample_rate":
            assert sample_rate == transform["sample_rate"]
            continue
        if op == "resample":
            target_rate = transform["sample_rate"]
            target_length = round(samples.shape[-1] * target_rate / sample_rate)
            source_positions = np.arange(samples.shape[-1], dtype=np.float64)
            target_positions = np.linspace(
                0.0, samples.shape[-1] - 1, target_length, dtype=np.float64
            )
            samples = np.interp(target_positions, source_positions, samples).astype(np.float32)
            sample_rate = target_rate
            continue
        if op in {"spectrogram", "scaled_spectrogram"}:
            if op == "scaled_spectrogram":
                reference_rate = transform["sample_rate"]
                geometry = tuple(
                    _even_scaled(transform[name], sample_rate, reference_rate)
                    for name in ("n_fft", "hop_length", "win_length")
                )
            else:
                geometry = (
                    transform["n_fft"],
                    transform["hop_length"],
                    transform["win_length"],
                )
            break
    else:
        raise AssertionError("workflow declares no spectrogram transform")

    n_fft, hop_length, win_length = geometry
    tensor = torch.from_numpy(samples[None])
    spectrum = torch.stft(
        tensor,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    noisy_mag = torch.log1p(torch.abs(spectrum)).numpy()
    noisy_pha = torch.angle(spectrum).numpy()
    denoised_mag, denoised_pha, _denoised_com = session.run(
        None,
        {"noisy_mag": noisy_mag, "noisy_pha": noisy_pha},
    )

    # NVIDIA suppresses frames whose decompressed magnitude is zero in more
    # than half the bins, then performs ISTFT and pads/trims to the input.
    decompressed = np.expm1(np.maximum(denoised_mag, 0.0))
    bad_frames = np.mean(np.equal(decompressed, 0.0), axis=1) > 0.5
    denoised_mag = denoised_mag.copy()
    denoised_mag[:, :, bad_frames[0]] = 0.0
    magnitude = torch.from_numpy(np.expm1(np.maximum(denoised_mag, 0.0)))
    phase = torch.from_numpy(denoised_pha)
    enhanced = torch.istft(
        torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase)),
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=True,
    ).numpy()[0]
    target_length = samples.shape[-1]
    if enhanced.shape[-1] < target_length:
        enhanced = np.pad(
            enhanced,
            (0, target_length - enhanced.shape[-1]),
            constant_values=1e-8,
        )
    else:
        enhanced = enhanced[:target_length]
    return {
        "audio": enhanced,
        "sample_rate": sample_rate,
        "sample_length": target_length,
        "geometry": geometry,
        "input_shape": noisy_mag.shape,
    }


class TestDetection:
    """The dispatcher must recognise the package structurally."""

    def test_detects_enhancement_package(self):
        pkg, _config = _package()
        assert _looks_like_speech_enhancement(pkg) is True

    def test_rejects_a_decoder_package(self):
        from mobius._testing import make_config
        from mobius.models import CausalLMModel

        config = make_config()
        pkg = build_from_module(CausalLMModel(config), config)

        assert _looks_like_speech_enhancement(pkg) is False


class TestMetadata:
    """The emitted document must describe the graph truthfully."""

    def test_declares_a_single_pure_invocation(self):
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        workflow = metadata["pipeline"]["workflow"]
        # No generation loop: nothing to sample, nothing carried between calls.
        invokes = [s for s in workflow["steps"] if s["kind"] == "invoke"]
        assert [s["component"] for s in invokes] == [
            "audio_preprocess",
            "enhancer",
            "audio_postprocess",
        ]
        emits = [s for s in workflow["steps"] if s["kind"] == "emit"]
        assert len(emits) == 6
        assert not any(s["kind"] == "loop" for s in workflow["steps"])
        for effect in workflow["effects"].values():
            assert effect["retry"] == "pure"

    def test_publishes_every_graph_output(self):
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        outputs = metadata["pipeline"]["workflow"]["outputs"]
        assert set(outputs) == {
            "denoised_mag",
            "denoised_pha",
            "denoised_com",
            "audio",
            "sample_rate",
            "sample_lengths",
        }
        # The complex spectrogram carries a trailing (real, imag) pair.
        assert outputs["denoised_com"]["contract"]["rank"] == 4
        assert outputs["denoised_com"]["contract"]["shape"][-1] == 2
        assert outputs["denoised_mag"]["contract"]["rank"] == 3
        profile = metadata["profiles"]["speech_enhancement"]
        assert set(profile["outputs"]) == set(outputs)

    def test_declares_the_stft_front_end(self):
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        transforms = metadata["preprocessing"]["audio"]["transforms"]
        by_op = {t["op"]: t for t in transforms}
        assert "resample" not in by_op
        spectrogram = by_op["scaled_spectrogram"]
        assert spectrogram["n_fft"] == config.n_fft
        assert spectrogram["hop_length"] == config.hop_size
        assert spectrogram["win_length"] == config.win_size
        assert spectrogram["sample_rate"] == config.sampling_rate
        assert spectrogram["mode"] == (
            "native_scaled_floor_then_even_center_reflect_unnormalized"
        )
        # The model is trained on log1p-compressed magnitudes.
        assert "log1p" in by_op
        assert by_op["log1p"]["inputs"] == ["magnitude"]

    def test_audio_outputs_bind_to_the_graph_inputs(self):
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        bindings = {b["name"]: b for b in metadata["preprocessing"]["audio"]["outputs"]}
        assert set(bindings) == {
            "noisy_mag",
            "noisy_pha",
            "reference_audio",
            "sample_rate",
            "sample_lengths",
        }
        assert bindings["noisy_mag"]["source"] == "magnitude"
        assert bindings["noisy_pha"]["source"] == "phase"
        assert bindings["reference_audio"]["source"] == "samples"
        assert bindings["sample_rate"]["source"] == "sample_rate"
        assert bindings["sample_lengths"]["source"] == "sample_lengths"
        assert bindings["noisy_mag"]["contract"]["rank"] == 3
        assert bindings["reference_audio"]["contract"]["rank"] == 2

    def test_declares_the_adapter_abi_when_the_front_end_ships(self):
        """A runtime must be able to version-check the STFT adapter it has to supply.

        The component already names the ABI, but a consumer reads the manifest to
        decide whether it can run the package at all — the other audio workflows
        declare it there, and omitting it hides the requirement until binding time.
        """
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        manifest = metadata["pipeline"]["workflow"]["manifest"]
        components = metadata["pipeline"]["workflow"]["components"]
        for name in ("audio_preprocess", "audio_postprocess"):
            component = components[name]
            abi = component["implementation"]["abi"]
            assert manifest["adapter_abis"][abi] == component["implementation"]["version"]

    def test_omits_the_adapter_abi_when_no_front_end_ships(self):
        """No adapter in the package means nothing for a runtime to version-check.

        Declaring the ABI unconditionally would advertise a requirement the caller
        does not have to satisfy, since it supplies the spectra itself.
        """
        pkg, config = _package()
        config.sampling_rate = None  # type: ignore[assignment]

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        manifest = metadata["pipeline"]["workflow"]["manifest"]
        assert "adapter_abis" not in manifest
        assert "audio_preprocess" not in metadata["pipeline"]["workflow"]["components"]
        assert "audio_postprocess" not in metadata["pipeline"]["workflow"]["components"]

    def test_omits_the_program_when_geometry_is_unknown(self):
        """Never invent a transform program the config cannot justify."""
        pkg, config = _package()
        config.sampling_rate = None  # type: ignore[assignment]

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        assert "preprocessing" not in metadata
        # The caller then supplies the spectra directly.
        inputs = metadata["pipeline"]["workflow"]["inputs"]
        assert set(inputs) == {"request.noisy_mag", "request.noisy_pha"}

    def test_fixed_native_rate_requires_that_rate_without_resampling(self):
        pkg, config = _package(input_sampling_rate=16_000)

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)
        transforms = metadata["preprocessing"]["audio"]["transforms"]
        by_op = {transform["op"]: transform for transform in transforms}

        assert by_op["require_sample_rate"]["sample_rate"] == 16_000
        assert "resample" not in by_op
        assert by_op["spectrogram"] == {
            "op": "spectrogram",
            "n_fft": 64,
            "hop_length": 8,
            "win_length": 64,
            "window": "hann",
            "mode": "center_reflect_unnormalized",
            "inputs": ["samples"],
            "outputs": ["magnitude", "phase"],
        }

    def test_bwe_rate_resamples_and_scales_static_geometry(self):
        pkg, config = _package(bwe_sampling_rate=48_000)

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)
        transforms = metadata["preprocessing"]["audio"]["transforms"]
        by_op = {transform["op"]: transform for transform in transforms}

        assert by_op["resample"]["sample_rate"] == 48_000
        assert "require_sample_rate" not in by_op
        assert by_op["spectrogram"]["n_fft"] == 192
        assert by_op["spectrogram"]["hop_length"] == 24
        assert by_op["spectrogram"]["win_length"] == 192

    @pytest.mark.parametrize("bwe_sampling_rate", [16_000, 48_000])
    def test_rejects_native_and_bwe_rates_together(self, bwe_sampling_rate):
        pkg, config = _package()
        config.input_sampling_rate = 16_000
        config.bwe_sampling_rate = bwe_sampling_rate

        with pytest.raises(ValueError, match="mutually exclusive"):
            build_speech_enhancement_workflow_metadata(pkg, config)

    @pytest.mark.parametrize(
        "missing_field", ["sampling_rate", "n_fft", "hop_size", "win_size"]
    )
    @pytest.mark.parametrize("bwe_sampling_rate", [16_000, 48_000])
    def test_rejects_dual_rates_before_incomplete_geometry(
        self, missing_field, bwe_sampling_rate
    ):
        pkg, config = _package()
        setattr(config, missing_field, None)
        config.input_sampling_rate = 16_000
        config.bwe_sampling_rate = bwe_sampling_rate

        with pytest.raises(ValueError, match="mutually exclusive"):
            build_speech_enhancement_workflow_metadata(pkg, config)

    @pytest.mark.parametrize("rate_name", ["input_sampling_rate", "bwe_sampling_rate"])
    @pytest.mark.parametrize("value", ["16000", 16_000.0, True, False, 0, -1])
    def test_rejects_invalid_rate_before_incomplete_geometry(self, rate_name, value):
        pkg, config = _package()
        config.sampling_rate = None  # type: ignore[assignment]
        setattr(config, rate_name, value)

        with pytest.raises(ValueError, match=f"{rate_name} must be a positive integer"):
            build_speech_enhancement_workflow_metadata(pkg, config)

    @pytest.mark.parametrize(
        ("rate_name", "rate_value"),
        [(None, None), ("input_sampling_rate", 16_000), ("bwe_sampling_rate", 48_000)],
    )
    @pytest.mark.parametrize(
        "missing_field", ["sampling_rate", "n_fft", "hop_size", "win_size"]
    )
    def test_valid_rates_with_incomplete_geometry_use_spectrum_inputs(
        self, missing_field, rate_name, rate_value
    ):
        pkg, config = _package()
        setattr(config, missing_field, None)
        if rate_name is not None:
            setattr(config, rate_name, rate_value)

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        assert "preprocessing" not in metadata
        assert set(metadata["pipeline"]["workflow"]["inputs"]) == {
            "request.noisy_mag",
            "request.noisy_pha",
        }

    @pytest.mark.parametrize("bwe_sampling_rate", [16_000, 48_000])
    def test_builder_validates_rates_before_graph_inspection(self, bwe_sampling_rate):
        _, config = _package()
        config.sampling_rate = None  # type: ignore[assignment]
        config.input_sampling_rate = 16_000
        config.bwe_sampling_rate = bwe_sampling_rate

        with pytest.raises(ValueError, match="mutually exclusive"):
            build_speech_enhancement_workflow_metadata({}, config)

    def test_postprocess_declares_inverse_and_reference_length_contract(self):
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)
        component = metadata["pipeline"]["workflow"]["components"]["audio_postprocess"]
        parameters = component["contract"]["parameters"]

        assert parameters["geometry_mode"] == "native_scaled"
        assert parameters["rounding"] == "floor_then_even"
        assert parameters["magnitude_decompression"] == "relu_log1p"
        assert parameters["zero_frame_fraction_threshold"] == pytest.approx(0.5)
        assert parameters["length_alignment"] == "pad_or_trim_to_reference"
        assert parameters["pad_value"] == pytest.approx(1e-8)

    def test_rejects_a_graph_without_the_expected_ports(self):
        pkg, config = _package()
        pkg["model"].graph.inputs[0].name = "something_else"

        with pytest.raises(ValueError, match="noisy_mag"):
            build_speech_enhancement_workflow_metadata(pkg, config)


class TestNativeRateRuntime:
    """Execute the pinned source's rate scaling and length alignment contract."""

    @pytest.fixture(scope="class")
    def native_runtime(self):
        return _runtime_package()

    @pytest.mark.parametrize(
        ("sample_rate", "geometry"),
        [
            (8_000, (320, 40, 320)),
            (16_000, (640, 80, 640)),
            (48_000, (1920, 240, 1920)),
        ],
    )
    @pytest.mark.parametrize("length_delta", [0, 1])
    def test_native_rate_preserves_rate_and_odd_even_length(
        self,
        native_runtime,
        sample_rate,
        geometry,
        length_delta,
    ):
        package, config, session = native_runtime
        length = sample_rate // 20 + length_delta
        time = np.arange(length, dtype=np.float32) / sample_rate
        waveform = 0.1 * np.sin(2 * np.pi * 440 * time)

        result = _run_declared_workflow(
            package,
            config,
            session,
            waveform,
            sample_rate,
        )

        assert result["sample_rate"] == sample_rate
        assert result["sample_length"] == length
        assert result["audio"].shape == (length,)
        assert result["geometry"] == geometry
        assert result["input_shape"][1] == geometry[0] // 2 + 1
        assert np.isfinite(result["audio"]).all()

    def test_explicit_bwe_resamples_and_returns_target_rate(self):
        package, config, session = _runtime_package(bwe_sampling_rate=16_000)
        source_rate = 8_000
        source_length = 401
        time = np.arange(source_length, dtype=np.float32) / source_rate
        waveform = 0.1 * np.sin(2 * np.pi * 440 * time)

        result = _run_declared_workflow(
            package,
            config,
            session,
            waveform,
            source_rate,
        )

        assert result["sample_rate"] == 16_000
        assert result["sample_length"] == 802
        assert result["audio"].shape == (802,)
        assert result["geometry"] == (640, 80, 640)
        assert result["input_shape"][1] == 321


class TestAutoExport:
    """`write_onnx_genai_config` must route the package here."""

    def test_dispatches_to_the_enhancement_writer(self, tmp_path):
        pkg, config = _package()

        artifacts = write_onnx_genai_config(pkg, str(tmp_path), config=config)

        assert os.path.isfile(artifacts["inference_metadata"])

    def test_written_document_round_trips(self, tmp_path):
        pkg, config = _package()
        yaml = pytest.importorskip("yaml")

        path = write_speech_enhancement_workflow_metadata(pkg, str(tmp_path), config)

        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        assert document["schema_version"] == "v1"
        assert "speech_enhancement" in document["profiles"]

    def test_writer_validates_rates_before_output_or_graph_inspection(self, tmp_path):
        _, config = _package()
        config.n_fft = None  # type: ignore[assignment]
        config.input_sampling_rate = 16_000
        config.bwe_sampling_rate = 48_000
        output_dir = tmp_path / "metadata"

        with pytest.raises(ValueError, match="mutually exclusive"):
            write_speech_enhancement_workflow_metadata({}, str(output_dir), config)

        assert not output_dir.exists()


class TestSchema:
    """The document must validate against onnx-genai's committed schema."""

    def test_matches_onnx_genai_json_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema_path = _onnx_genai_schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found (set ONNX_GENAI_SCHEMA)")
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        with open(schema_path, encoding="utf-8") as handle:
            schema = json.load(handle)
        jsonschema.validate(metadata, schema)

    def test_matches_schema_without_preprocessing(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema_path = _onnx_genai_schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found (set ONNX_GENAI_SCHEMA)")
        pkg, config = _package()
        config.n_fft = None  # type: ignore[assignment]

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        with open(schema_path, encoding="utf-8") as handle:
            schema = json.load(handle)
        jsonschema.validate(metadata, schema)
