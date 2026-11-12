# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned upstream numerical checks for LiquidAI LFM3 Audio components."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnxscript import GraphBuilder
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mobius._configs import Lfm2Config
from mobius._constants import OPSET_VERSION
from mobius._optimizations import optimize_model
from mobius.models.lfm2 import rename_lfm2_weight_key
from mobius.models.lfm3_audio import Lfm3AudioDownsampler, Lfm3AudioEncoder
from tests.integration._support import _make_session

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
REVISION = "2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1"


@pytest.mark.integration
@pytest.mark.parametrize("frames,lengths", [(25, [25, 17]), (32, [32, 9]), (1, [1, 1])])
def test_downsampler_matches_pinned_upstream(frames, lengths):
    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    ).audio_encoder_config
    config.conv_channels = 4
    config.hidden_size = 32
    reference_class = get_class_from_dynamic_module(
        "modeling_lfm3_audio.Lfm3AudioDownsampler", MODEL_ID, revision=REVISION
    )
    torch.manual_seed(42)
    reference = reference_class(config).eval()
    module = Lfm3AudioDownsampler(
        config.num_mel_bins, config.conv_channels, config.hidden_size
    )

    graph = ir.Graph([], [], nodes=[], opset_imports={"": OPSET_VERSION}, name="downsampler")
    builder = GraphBuilder(graph)
    features = builder.input("input_features", dtype=ir.DataType.FLOAT, shape=[2, 128, frames])
    feature_lengths = builder.input("feature_lengths", dtype=ir.DataType.INT64, shape=[2])
    output = module(builder.op, features, feature_lengths)
    builder.add_output(output, "hidden_states")
    model = ir.Model(graph, ir_version=10)
    for name, value in reference.state_dict().items():
        model.graph.initializers[name].const_value = ir.Tensor(value.numpy())
    inputs = torch.randn(2, 128, frames)
    valid_lengths = torch.tensor(lengths, dtype=torch.int64)
    with torch.no_grad():
        expected = reference(inputs, valid_lengths).numpy()
    session = _make_session(model)
    try:
        actual = session.run(
            {"input_features": inputs.numpy(), "feature_lengths": valid_lengths.numpy()},
        )["hidden_states"]
    finally:
        session.close()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("frames,lengths", [(25, [25, 17]), (65, [65, 33])])
def test_encoder_matches_pinned_upstream(frames, lengths):
    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    ).audio_encoder_config
    config.conv_channels = 4
    config.hidden_size = 32
    config.sliding_window = 3
    text = config.text_config
    text.hidden_size = 32
    text.intermediate_size = 64
    text.block_auto_adjust_ff_dim = False
    text.num_hidden_layers = 2
    text.layer_types = ["conv", "full_attention"]
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text._attn_implementation = "eager"
    reference_class = get_class_from_dynamic_module(
        "modeling_lfm3_audio.Lfm3AudioEncoder", MODEL_ID, revision=REVISION
    )
    torch.manual_seed(42)
    reference = reference_class(config).eval()
    module = Lfm3AudioEncoder(
        Lfm2Config.from_transformers(text),
        num_mel_bins=config.num_mel_bins,
        conv_channels=config.conv_channels,
        sliding_window=config.sliding_window,
    )
    graph = ir.Graph([], [], nodes=[], opset_imports={"": OPSET_VERSION}, name="audio_encoder")
    builder = GraphBuilder(graph)
    features = builder.input("input_features", dtype=ir.DataType.FLOAT, shape=[2, 128, frames])
    feature_lengths = builder.input("feature_lengths", dtype=ir.DataType.INT64, shape=[2])
    output, output_lengths = module(builder.op, features, feature_lengths)
    builder.add_output(output, "audio_features")
    builder.add_output(output_lengths, "audio_feature_lengths")
    model = ir.Model(graph, ir_version=10)
    for name, value in reference.state_dict().items():
        name = rename_lfm2_weight_key(name)
        model.graph.initializers[name].const_value = ir.Tensor(value.numpy())
    optimize_model(model, ep="cpu", model_role="encoder")
    inputs = torch.randn(2, 128, frames)
    valid_lengths = torch.tensor(lengths, dtype=torch.int64)
    with torch.no_grad():
        expected, expected_lengths = reference(inputs, valid_lengths)
    session = _make_session(model)
    try:
        actual = session.run(
            {"input_features": inputs.numpy(), "feature_lengths": valid_lengths.numpy()}
        )
    finally:
        session.close()
    np.testing.assert_array_equal(actual["audio_feature_lengths"], expected_lengths.numpy())
    np.testing.assert_allclose(
        actual["audio_features"], expected.numpy(), rtol=1e-4, atol=1e-5
    )
