# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Numerical parity of the from-scratch SD UNet against diffusers.

These build the mobius module + the matching diffusers module with the *same*
random weights (no checkpoint download) and compare outputs — the pattern used
by ``tests/integration_test.py`` for the QwenImage VAE. Guarded on diffusers.
"""

from __future__ import annotations

import re
import tempfile

import numpy as np
import pytest


def _remap_transformer(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        new_key = re.sub(r"transformer_blocks\.\d+\.", "", key)
        new_key = new_key.replace("ff.net.0.proj.", "ff.proj_in.").replace(
            "ff.net.2.", "ff.proj_out."
        )
        out[new_key] = value
    return out


def test_cross_attention_block_matches_diffusers():
    pytest.importorskip("diffusers")
    import onnx_ir
    import onnxruntime as ort
    import torch
    from diffusers.models.transformers.transformer_2d import Transformer2DModel

    from mobius._weight_loading import apply_weights
    from mobius.models.unet import _CrossAttentionBlock
    from mobius.tasks._base import _make_graph, _make_model

    torch.manual_seed(0)
    channels, heads, head_dim, cross_dim, groups = 32, 2, 16, 16, 32
    hf = Transformer2DModel(
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        in_channels=channels,
        cross_attention_dim=cross_dim,
        use_linear_projection=False,
        norm_num_groups=groups,
    ).eval()
    hidden = torch.randn(1, channels, 4, 4)
    context = torch.randn(1, 5, cross_dim)
    with torch.no_grad():
        expected = hf(hidden, encoder_hidden_states=context).sample.numpy()

    graph, builder = _make_graph()
    op = builder.op
    hs = builder.input("hidden", dtype=onnx_ir.DataType.FLOAT, shape=[1, channels, 4, 4])
    ehs = builder.input("context", dtype=onnx_ir.DataType.FLOAT, shape=[1, 5, cross_dim])
    block = _CrossAttentionBlock(channels, cross_dim, num_heads=heads, norm_num_groups=groups)
    builder.add_output(block(op, hs, ehs), "out")
    model = _make_model(graph)
    apply_weights(model, _remap_transformer(hf.state_dict()))

    with tempfile.NamedTemporaryFile(suffix=".onnx") as handle:
        onnx_ir.save(model, handle.name)
        session = ort.InferenceSession(handle.name)
        actual = session.run(
            None, {"hidden": hidden.numpy(), "context": context.numpy()}
        )[0]
    assert np.abs(actual - expected).max() < 1e-4


def test_unet_matches_diffusers():
    pytest.importorskip("diffusers")
    import onnx_ir
    import onnxruntime as ort
    import torch
    from diffusers import UNet2DConditionModel as HFUNet

    from mobius._diffusers_configs import UNet2DConfig
    from mobius._weight_loading import apply_weights
    from mobius.models.unet import UNet2DConditionModel
    from mobius.tasks._denoising import DenoisingTask

    torch.manual_seed(0)
    hf = HFUNet(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        cross_attention_dim=16,
        attention_head_dim=8,
        norm_num_groups=32,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
    ).eval()
    sample = torch.randn(1, 4, 8, 8)
    timestep = torch.tensor([1])
    encoder_hidden_states = torch.randn(1, 4, 16)
    with torch.no_grad():
        expected = hf(sample, timestep, encoder_hidden_states).sample.numpy()

    config = UNet2DConfig(
        in_channels=4,
        out_channels=4,
        block_out_channels=(32, 64),
        layers_per_block=1,
        norm_num_groups=32,
        cross_attention_dim=16,
        attention_head_dim=8,
        use_linear_projection=False,
    )
    module = UNet2DConditionModel(config)
    model = DenoisingTask().build(module, config)["model"]
    apply_weights(model, module.preprocess_weights(dict(hf.state_dict())))

    with tempfile.NamedTemporaryFile(suffix=".onnx") as handle:
        onnx_ir.save(model, handle.name)
        session = ort.InferenceSession(handle.name)
        actual = session.run(
            None,
            {
                "sample": sample.numpy(),
                "timestep": timestep.numpy().astype(np.int64),
                "encoder_hidden_states": encoder_hidden_states.numpy(),
            },
        )[0]
    assert np.abs(actual - expected).max() < 2e-4
