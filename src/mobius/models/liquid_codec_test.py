# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph, synthesis, and weight-contract tests for LiquidCodec."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.liquid_codec import (
    LiquidCodecConfig,
    LiquidCodecDecoder,
    LiquidCodecISTFT,
    build_liquid_codec_decoder,
)


def test_decoder_builds_independent_codes_to_waveform_graph():
    model = build_liquid_codec_decoder()

    assert [value.name for value in model.graph.inputs] == ["codes"]
    assert model.graph.inputs[0].dtype == ir.DataType.INT64
    assert [value.name for value in model.graph.outputs] == ["waveform"]
    assert any(node.op_type == "DFT" for node in model.graph)
    assert any(node.op_type == "Col2Im" for node in model.graph)
    assert any(node.op_type == "Attention" for node in model.graph)
    assert any(node.op_type == "CausalConvWithState" for node in model.graph)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_decoder_rejects_unverified_reduced_precision(dtype):
    with pytest.raises(ValueError, match="supports float32 only"):
        LiquidCodecDecoder(LiquidCodecConfig(dtype=dtype))


def test_preprocess_weights_keeps_only_standalone_synthesis_tensors():
    weights = {
        "codec.quantizer.project_out.weight": object(),
        "model.from_quantizer.bias": object(),
        "decoder.lfm.layers.0.operator_norm.weight": object(),
        "frontend.in_proj.weight": object(),
        "encoder.lfm.layers.0.operator_norm.weight": object(),
        "to_quantizer.weight": object(),
    }

    result = LiquidCodecDecoder.preprocess_weights(weights)

    assert set(result) == {
        "quantizer.project_out.weight",
        "from_quantizer.bias",
        "decoder.lfm.layers.0.operator_norm.weight",
    }


def test_istft_matches_torch_inverse_fft_and_overlap_add():
    config = LiquidCodecConfig()
    module = LiquidCodecISTFT(config)
    graph = ir.Graph([], [], nodes=[], opset_imports={"": OPSET_VERSION}, name="istft")
    builder = GraphBuilder(graph)
    real = builder.input("real", dtype=ir.DataType.FLOAT, shape=[1, 641, 2])
    imag = builder.input("imag", dtype=ir.DataType.FLOAT, shape=[1, 641, 2])
    audio, envelope = module(builder.op, real, imag)
    builder.add_output(audio, "audio")
    builder.add_output(envelope, "envelope")
    model = ir.Model(graph, ir_version=10)

    torch.manual_seed(7)
    spectrum = torch.randn(1, 641, 2, dtype=torch.complex64)
    window = torch.hann_window(config.stft_win_length)
    frames = torch.fft.irfft(spectrum, config.stft_n_fft, dim=1) * window[None, :, None]
    expected_audio = torch.nn.functional.fold(
        frames,
        output_size=(1, 1600),
        kernel_size=(1, config.stft_win_length),
        stride=(1, config.stft_hop_length),
    )[:, 0, 0][:, 480:-480]

    session = OnnxModelSession(model)
    try:
        actual = session.run({"real": spectrum.real.numpy(), "imag": spectrum.imag.numpy()})
    finally:
        session.close()

    assert actual["audio"].shape == (1, 640)
    assert np.all(actual["envelope"] > 0)
    np.testing.assert_allclose(actual["audio"], expected_audio.numpy(), rtol=2e-5, atol=1e-5)
