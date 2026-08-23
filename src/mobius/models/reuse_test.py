# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the RE-USE / SEMamba speech-enhancement model."""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import pytest

from mobius import build_from_module
from mobius.models.reuse import (
    ReUseConfig,
    SEMambaSpeechEnhancementModel,
    _atan2,
    build_reuse,
)
from mobius.tasks import SpeechEnhancementTask, get_task

# The real nvidia/RE-USE config, scaled down: same structure, tiny widths.
_TINY_CONFIG = {
    "model_cfg": {
        "hid_feature": 8,
        "num_tfmamba": 2,
        "d_state": 4,
        "d_conv": 4,
        "expand": 2,
        "input_channel": 2,
        "output_channel": 1,
        "norm_epsilon": 1e-5,
        "compress_factor": "relu_log1p",
    },
    "stft_cfg": {"n_fft": 32, "hop_size": 4, "win_size": 32, "sampling_rate": 8000},
}


def _tiny_config() -> ReUseConfig:
    return ReUseConfig.from_json(_TINY_CONFIG)


def _build():
    config = _tiny_config()
    module = SEMambaSpeechEnhancementModel(config)
    return config, build_from_module(module, config, task=SpeechEnhancementTask())


class TestReUseConfig:
    """Config extraction from the nvidia/RE-USE config.json layout."""

    def test_from_json_reads_both_sections(self):
        config = _tiny_config()
        assert config.hid_feature == 8
        assert config.num_tfmamba == 2
        assert config.d_state == 4
        assert config.expand == 2
        assert config.n_fft == 32
        assert config.hop_size == 4
        assert config.sampling_rate == 8000
        assert config.model_type == "reuse"

    def test_real_checkpoint_shape_parameters(self):
        """The published config yields the checkpoint's real dimensions."""
        config = ReUseConfig.from_json(
            {
                "model_cfg": {
                    "hid_feature": 64,
                    "num_tfmamba": 30,
                    "d_state": 16,
                    "d_conv": 4,
                    "expand": 4,
                },
                "stft_cfg": {"n_fft": 320, "hop_size": 40, "win_size": 320},
            }
        )
        # in_proj is (2 * d_inner, hid_feature) = (512, 64) in the checkpoint.
        assert config.d_inner == 256
        # x_proj is (dt_rank + 2 * d_state, d_inner) = (36, 256).
        assert config.dt_rank == 4
        assert config.dt_rank + 2 * config.d_state == 36
        assert config.num_freq_bins == 161

    def test_dt_rank_follows_mamba_convention(self):
        for hid in (8, 64, 100):
            config = ReUseConfig(hid_feature=hid)
            assert config.dt_rank == math.ceil(hid / 16)

    def test_validate_rejects_odd_n_fft(self):
        with pytest.raises(ValueError, match="n_fft"):
            ReUseConfig(n_fft=33).validate()


class TestBuildGraphReUse:
    """Graph construction for the SEMamba generator."""

    def test_task_registry_lookup(self):
        assert isinstance(get_task("speech-enhancement"), SpeechEnhancementTask)

    def test_default_task(self):
        assert SEMambaSpeechEnhancementModel.default_task == "speech-enhancement"

    def test_package_has_single_model(self):
        _config, pkg = _build()
        assert set(pkg) == {"model"}

    def test_model_io(self):
        config, pkg = _build()
        graph = pkg["model"].graph

        assert [inp.name for inp in graph.inputs] == ["noisy_mag", "noisy_pha"]
        assert [out.name for out in graph.outputs] == [
            "denoised_mag",
            "denoised_pha",
            "denoised_com",
        ]
        # Frequency is fixed by the STFT size; batch and time stay symbolic.
        for value in (*graph.inputs, *graph.outputs[:2]):
            assert value.shape[1] == config.num_freq_bins
        assert graph.outputs[2].shape[3] == 2

    def test_initializer_names_match_checkpoint_layout(self):
        """Parameter names line up with the nvidia/RE-USE checkpoint."""
        config = _tiny_config()
        module = SEMambaSpeechEnhancementModel(config)
        names = {name for name, _ in module.named_parameters()}

        # nn.Sequential stages keep their integer indices.
        assert "dense_encoder.dense_conv_1.0.weight" in names
        assert "dense_encoder.dense_conv_1.1.bias" in names
        assert "dense_encoder.dense_conv_1.2.weight" in names
        assert "dense_encoder.dense_block.dense_block.3.0.weight" in names
        assert "mask_decoder.up_conv1.0.conv.weight" in names
        assert "mask_decoder.final_conv.weight" in names
        assert "phase_decoder.phase_conv_r.weight" in names
        assert "phase_decoder.phase_conv_i.weight" in names
        # Mamba parameters keep the checkpoint's block names.
        assert "TSMamba.0.time_mamba.forward_blocks.in_proj.weight" in names
        assert "TSMamba.0.freq_mamba.backward_blocks.conv1d.bias" in names
        assert "TSMamba.1.time_mamba.output_proj.bias" in names
        assert "TSMamba.1.freq_mamba.norm.weight" in names

    def test_dense_block_dilation_grows_by_powers_of_two(self):
        """The i-th dense conv reads i+1 stacked feature maps at dilation 2**i."""
        config = _tiny_config()
        module = SEMambaSpeechEnhancementModel(config)
        block = module.dense_encoder.dense_block
        for i in range(config.dense_depth):
            conv = block.dense_block[i][0]
            assert list(conv.weight.shape) == [
                config.hid_feature,
                config.hid_feature * (i + 1),
                3,
                3,
            ]
            assert conv._dilations == (2**i, 1)
            assert conv._pads == [2**i, 1, 2**i, 1]

    def test_preprocess_weights_nests_ssm_parameters(self):
        """Flat checkpoint SSM parameters move under the ``ssm`` submodule."""
        config = _tiny_config()
        module = SEMambaSpeechEnhancementModel(config)
        prefix = "TSMamba.0.time_mamba.forward_blocks"
        state_dict = {
            f"{prefix}.A_log": "a",
            f"{prefix}.D": "d",
            f"{prefix}.x_proj.weight": "x",
            f"{prefix}.dt_proj.weight": "w",
            f"{prefix}.dt_proj.bias": "b",
            f"{prefix}.in_proj.weight": "i",
            f"{prefix}.conv1d.bias": "c",
        }

        result = module.preprocess_weights(state_dict)

        assert result[f"{prefix}.ssm.A_log"] == "a"
        assert result[f"{prefix}.ssm.dt_proj.bias"] == "b"
        # Non-SSM parameters are untouched.
        assert result[f"{prefix}.in_proj.weight"] == "i"
        assert result[f"{prefix}.conv1d.bias"] == "c"
        assert f"{prefix}.A_log" not in result

    def test_preprocess_weights_is_idempotent(self):
        """Already-nested names must survive a second pass unchanged.

        Renaming on suffix alone would turn ``ssm.A_log`` into
        ``ssm.ssm.A_log``, silently dropping every SSM parameter when a
        converted state dict is preprocessed again.
        """
        config = _tiny_config()
        module = SEMambaSpeechEnhancementModel(config)
        onnx_names = {name for name, _ in module.named_parameters()}
        checkpoint_names = {name.replace(".ssm.", ".") for name in onnx_names}

        once = module.preprocess_weights(dict.fromkeys(checkpoint_names, 0))
        twice = module.preprocess_weights(dict(once))

        assert set(once) == onnx_names
        assert set(twice) == onnx_names
        assert not any(".ssm.ssm." in name for name in twice)

    def test_preprocess_weights_covers_every_parameter(self):
        """Renaming a checkpoint-shaped state dict yields exactly our names."""
        config = _tiny_config()
        module = SEMambaSpeechEnhancementModel(config)
        onnx_names = {name for name, _ in module.named_parameters()}
        # Reconstruct the checkpoint's flat naming from ours.
        checkpoint_names = {name.replace(".ssm.", ".") for name in onnx_names}

        renamed = module.preprocess_weights(dict.fromkeys(checkpoint_names, 0))

        assert set(renamed) == onnx_names

    def test_scan_count_matches_mamba_module_count(self):
        """One Scan per Mamba module: blocks x 2 axes x 2 directions."""
        config = _tiny_config()
        _config, pkg = _build()
        scans = sum(1 for node in pkg["model"].graph if node.op_type == "Scan")
        assert scans == config.num_tfmamba * 4


class TestReUseRuntime:
    """End-to-end ONNX Runtime execution with random weights."""

    def _session(self):
        ort = pytest.importorskip("onnxruntime")
        config = _tiny_config()
        module = SEMambaSpeechEnhancementModel(config)

        rng = np.random.default_rng(0)
        for _name, param in module.named_parameters():
            shape = [d if isinstance(d, int) else 1 for d in param.shape]
            param.const_value = ir.tensor(
                (rng.standard_normal(shape) * 0.05).astype(np.float32)
            )

        pkg = build_from_module(module, config, task=SpeechEnhancementTask())
        session = ort.InferenceSession(
            ir.to_proto(pkg["model"]).SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        return config, session

    @pytest.mark.parametrize("time_steps", [1, 5, 17])
    def test_output_shapes_track_input_length(self, time_steps):
        """The model is spectrally shape preserving for any input length."""
        config, session = self._session()
        rng = np.random.default_rng(1)
        shape = (2, config.num_freq_bins, time_steps)
        mag = np.abs(rng.standard_normal(shape)).astype(np.float32)
        pha = (rng.standard_normal(shape) * np.pi).astype(np.float32)

        denoised_mag, denoised_pha, denoised_com = session.run(
            None, {"noisy_mag": mag, "noisy_pha": pha}
        )

        assert denoised_mag.shape == shape
        assert denoised_pha.shape == shape
        assert denoised_com.shape == (*shape, 2)
        assert np.isfinite(denoised_mag).all()
        assert np.isfinite(denoised_pha).all()

    def test_phase_is_wrapped_and_consistent_with_complex_output(self):
        """Phase stays in (-pi, pi] and denoised_com is its polar form."""
        config, session = self._session()
        rng = np.random.default_rng(2)
        shape = (1, config.num_freq_bins, 7)
        mag = np.abs(rng.standard_normal(shape)).astype(np.float32)
        pha = (rng.standard_normal(shape) * np.pi).astype(np.float32)

        denoised_mag, denoised_pha, denoised_com = session.run(
            None, {"noisy_mag": mag, "noisy_pha": pha}
        )

        assert np.abs(denoised_pha).max() <= np.pi + 1e-5
        np.testing.assert_allclose(
            denoised_com[..., 0], denoised_mag * np.cos(denoised_pha), atol=1e-5
        )
        np.testing.assert_allclose(
            denoised_com[..., 1], denoised_mag * np.sin(denoised_pha), atol=1e-5
        )


class TestAtan2:
    """The hand-rolled atan2 (ONNX has no Atan2 op)."""

    def test_matches_numpy_over_all_quadrants(self):
        ort = pytest.importorskip("onnxruntime")
        session = self._session()

        # Cover all four quadrants plus both axes and the origin.
        grid = np.array([-2.0, -1.0, 0.0, 0.5, 2.0], dtype=np.float32)
        ys, xs = (a.ravel() for a in np.meshgrid(grid, grid))
        (got,) = session.run(None, {"y": ys, "x": xs})

        np.testing.assert_allclose(got, np.arctan2(ys, xs), atol=1e-6)
        assert ort is not None

    def test_negative_zero_numerator_picks_the_positive_branch(self):
        """Signed zeros collapse: -0.0 is treated as +0.0.

        IEEE distinguishes ``atan2(-0.0, -1) == -pi`` from
        ``atan2(+0.0, -1) == +pi``.  This implementation returns ``+pi`` for
        both, which is the same angle and therefore indistinguishable to the
        ``cos``/``sin`` consumers downstream.
        """
        pytest.importorskip("onnxruntime")
        session = self._session()

        ys = np.array([-0.0, 0.0], dtype=np.float32)
        xs = np.array([-1.0, -1.0], dtype=np.float32)
        (got,) = session.run(None, {"y": ys, "x": xs})

        np.testing.assert_allclose(got, [np.pi, np.pi], atol=1e-6)
        # Same point on the unit circle either way.
        np.testing.assert_allclose(np.cos(got), np.cos(np.arctan2(ys, xs)), atol=1e-6)
        np.testing.assert_allclose(np.sin(got), np.sin(np.arctan2(ys, xs)), atol=1e-6)

    def _session(self):
        import onnxruntime as ort
        from onnxscript import GraphBuilder

        from mobius._constants import OPSET_VERSION

        graph = ir.Graph([], [], nodes=[], name="g", opset_imports={"": OPSET_VERSION})
        builder = GraphBuilder(graph)
        y = builder.input("y", dtype=ir.DataType.FLOAT, shape=["n"])
        x = builder.input("x", dtype=ir.DataType.FLOAT, shape=["n"])
        builder.add_output(_atan2(builder.op, y, x), "out")

        return ort.InferenceSession(
            ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString(),
            providers=["CPUExecutionProvider"],
        )


class TestBuildReUse:
    """The bespoke Hub/directory loader (RE-USE has no transformers config)."""

    def _checkpoint_dir(self, tmp_path):
        """Write a tiny RE-USE-shaped repo: config.json + model.safetensors."""
        import json

        import torch
        from safetensors.torch import save_file

        (tmp_path / "config.json").write_text(json.dumps(_TINY_CONFIG))

        # The checkpoint keeps SSM parameters flat, so undo our ``ssm`` nesting.
        module = SEMambaSpeechEnhancementModel(_tiny_config())
        state = {
            name.replace(".ssm.", "."): torch.zeros(
                [d if isinstance(d, int) else 1 for d in param.shape]
            )
            for name, param in module.named_parameters()
        }
        save_file(state, str(tmp_path / "model.safetensors"))
        return tmp_path

    def test_reads_config_from_a_directory(self, tmp_path):
        pytest.importorskip("safetensors")
        config = ReUseConfig.from_pretrained(str(self._checkpoint_dir(tmp_path)))

        assert config.hid_feature == 8
        assert config.num_tfmamba == 2
        assert config.n_fft == 32

    def test_builds_and_fills_every_initializer(self, tmp_path):
        """A full local build leaves no initializer unpopulated."""
        pytest.importorskip("safetensors")
        pkg = build_reuse(str(self._checkpoint_dir(tmp_path)))

        graph = pkg["model"].graph
        unfilled = [
            name for name, value in graph.initializers.items() if value.const_value is None
        ]
        assert unfilled == []
        assert [inp.name for inp in graph.inputs] == ["noisy_mag", "noisy_pha"]

    def test_structure_only_build_skips_weights(self, tmp_path):
        pytest.importorskip("safetensors")
        pkg = build_reuse(str(self._checkpoint_dir(tmp_path)), load_weights=False)

        assert "model" in pkg


class TestExecutionProviderPartitioning:
    """Lock in graph shapes that plugin EPs need in order to claim the SSM.

    Both assertions here look like arbitrary spelling choices but were
    measured on the MLX plugin EP (Apple M1 Max, 2s of audio): getting
    either wrong costs between 2x and 80x.
    """

    def test_scan_iterates_axis_zero(self):
        """Scan must iterate axis 0, not name a `scan_input_axes` of 1.

        The MLX EP rejects any Scan whose scan axis is not 0
        ("only scan_input_axes=0 is supported"), which sends all 120
        recurrences back to CPU and erases the speedup.
        """
        _config, pkg = _build()

        scans = [node for node in pkg["model"].graph if node.op_type == "Scan"]
        assert scans
        for scan in scans:
            assert "scan_input_axes" not in scan.attributes
            assert "scan_output_axes" not in scan.attributes

    def test_sequence_reverse_uses_negative_step_slice(self):
        """The backward branch reverses with a negative-step Slice.

        ReverseSequence would also reverse the sequence, but it needs an
        extra Expand to build per-row lengths and implies padding semantics
        this model does not have — nothing here is padded, so every row is
        reversed in full.
        """
        _config, pkg = _build()

        reverse_slices = [
            node
            for node in pkg["model"].graph
            if node.op_type == "Slice"
            and len(node.inputs) == 5
            and node.inputs[4] is not None
            and node.inputs[4].const_value is not None
            and node.inputs[4].const_value.numpy().tolist() == [-1]
        ]
        # Two reversals (in and out) per bidirectional block, and each
        # TFMambaBlock has a time-axis and a frequency-axis block.
        assert len(reverse_slices) == _tiny_config().num_tfmamba * 2 * 2
        assert not any(node.op_type == "ReverseSequence" for node in pkg["model"].graph)
