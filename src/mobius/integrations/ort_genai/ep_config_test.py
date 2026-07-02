# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for _genai_config module."""

from __future__ import annotations

from mobius.integrations.ort_genai.ep_config import (
    make_genai_decoder_config,
    make_kv_cache_dim_name,
    make_provider_options,
    make_sliding_window_config,
)


class TestMakeProviderOptions:
    def test_cpu_returns_empty(self):
        assert make_provider_options("cpu") == []

    def test_cuda_default(self):
        result = make_provider_options("cuda")
        assert len(result) == 1
        assert "cuda" in result[0]
        assert result[0]["cuda"]["enable_cuda_graph"] == "0"

    def test_cuda_graph_capture_from_capability_flag(self):
        # A CUDA-like EP registered with enable_graph_capture=True emits
        # enable_cuda_graph="1"; the flag is the single source of truth.
        from mobius._execution_providers import EpCapabilities, ep_registry

        original = ep_registry.get("cuda")
        ep_registry.register(
            EpCapabilities(name="cuda", enable_graph_capture=True),
            overwrite=True,
        )
        try:
            result = make_provider_options("cuda")
            assert result[0]["cuda"]["enable_cuda_graph"] == "1"
        finally:
            ep_registry.register(original, overwrite=True)

    def test_dml(self):
        result = make_provider_options("dml")
        assert len(result) == 1
        assert "dml" in result[0]

    def test_webgpu_default(self):
        # WebGPU registers enable_graph_capture=True, so graph capture is on by
        # default and propagates to the genai_config provider options.
        result = make_provider_options("webgpu")
        assert result[0]["webgpu"]["enableGraphCapture"] == "1"
        assert result[0]["webgpu"]["validationMode"] == "disabled"

    def test_webgpu_graph_capture_disabled_when_flag_false(self):
        # When the capability flag is False, WebGPU emits the disabled defaults.
        from mobius._execution_providers import EpCapabilities, ep_registry

        original = ep_registry.get("webgpu")
        ep_registry.register(
            EpCapabilities(name="webgpu", enable_graph_capture=False),
            overwrite=True,
        )
        try:
            result = make_provider_options("webgpu")
            assert result[0]["webgpu"]["enableGraphCapture"] == "0"
            assert result[0]["webgpu"]["validationMode"] == "basic"
        finally:
            ep_registry.register(original, overwrite=True)

    def test_trt_rtx(self):
        result = make_provider_options("trt-rtx")
        assert result[0]["NvTensorRtRtx"]["enable_cuda_graph"] == "1"

    def test_openvino_defaults_to_npu(self):
        # The OpenVINO graph is device-independent; mobius emits a default
        # device_type ("NPU") that can be overridden downstream in genai_config.
        result = make_provider_options("openvino")
        assert result == [{"OpenVINO": {"device_type": "NPU"}}]


class TestMakeSlidingWindowConfig:
    def test_disabled_when_zero(self):
        assert make_sliding_window_config(window_size=0, num_layers=32) is None

    def test_disabled_when_negative(self):
        assert make_sliding_window_config(window_size=-1, num_layers=32) is None

    def test_all_layers_by_default(self):
        result = make_sliding_window_config(window_size=4096, num_layers=4)
        assert result is not None
        assert result["window_size"] == 4096
        assert result["layers"] == [0, 1, 2, 3]
        assert result["slide_key_value_cache"] is False

    def test_selective_layers(self):
        """Only even layers use sliding window."""
        result = make_sliding_window_config(
            window_size=2048,
            num_layers=4,
            is_local_fn=lambda i: i % 2 == 0,
        )
        assert result is not None
        assert result["layers"] == [0, 2]


class TestMakeKvCacheDimName:
    def test_non_trt_unchanged(self):
        assert (
            make_kv_cache_dim_name("past_sequence_length", ep="cuda", is_sliding_layer=False)
            == "past_sequence_length"
        )

    def test_trt_non_sliding_unchanged(self):
        assert (
            make_kv_cache_dim_name(
                "past_sequence_length", ep="trt-rtx", is_sliding_layer=False
            )
            == "past_sequence_length"
        )

    def test_trt_sliding_replaces_sequence(self):
        assert (
            make_kv_cache_dim_name("past_sequence_length", ep="trt-rtx", is_sliding_layer=True)
            == "past_sliding_length"
        )

    def test_trt_total_sequence(self):
        assert (
            make_kv_cache_dim_name(
                "total_sequence_length", ep="trt-rtx", is_sliding_layer=True
            )
            == "total_sliding_length"
        )


class TestMakeGenaiDecoderConfig:
    def test_basic_structure(self):
        result = make_genai_decoder_config(
            "cuda",
            head_size=64,
            hidden_size=2048,
            num_attention_heads=32,
            num_hidden_layers=24,
            num_key_value_heads=8,
        )
        assert result["filename"] == "model.onnx"
        assert result["head_size"] == 64
        assert result["hidden_size"] == 2048
        assert result["num_attention_heads"] == 32
        assert result["num_hidden_layers"] == 24
        assert result["num_key_value_heads"] == 8
        assert "session_options" in result
        assert len(result["session_options"]["provider_options"]) == 1

    def test_cpu_no_provider_options(self):
        result = make_genai_decoder_config(
            "cpu",
            head_size=64,
            hidden_size=2048,
            num_attention_heads=32,
            num_hidden_layers=24,
            num_key_value_heads=8,
        )
        assert result["session_options"]["provider_options"] == []

    def test_trt_rtx_with_sliding_window(self):
        result = make_genai_decoder_config(
            "trt-rtx",
            head_size=64,
            hidden_size=2048,
            num_attention_heads=32,
            num_hidden_layers=4,
            num_key_value_heads=8,
            sliding_window_size=4096,
        )
        assert "sliding_window" in result
        assert result["sliding_window"]["window_size"] == 4096
        assert result["sliding_window"]["layers"] == [0, 1, 2, 3]

    def test_trt_rtx_no_sliding_window(self):
        result = make_genai_decoder_config(
            "trt-rtx",
            head_size=64,
            hidden_size=2048,
            num_attention_heads=32,
            num_hidden_layers=4,
            num_key_value_heads=8,
        )
        assert "sliding_window" not in result

    def test_io_name_templates(self):
        result = make_genai_decoder_config(
            "cuda",
            head_size=64,
            hidden_size=2048,
            num_attention_heads=32,
            num_hidden_layers=24,
            num_key_value_heads=8,
        )
        assert result["inputs"]["past_key_names"] == "past_key_values.%d.key"
        assert result["outputs"]["present_key_names"] == "present.%d.key"
