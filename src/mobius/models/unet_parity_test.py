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
from pathlib import Path

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

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.onnx"
        onnx_ir.save(model, model_path)
        session = ort.InferenceSession(model_path)
        actual = session.run(None, {"hidden": hidden.numpy(), "context": context.numpy()})[0]
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

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.onnx"
        onnx_ir.save(model, model_path)
        session = ort.InferenceSession(model_path)
        actual = session.run(
            None,
            {
                "sample": sample.numpy(),
                "timestep": timestep.numpy().astype(np.int64),
                "encoder_hidden_states": encoder_hidden_states.numpy(),
            },
        )[0]
    assert np.abs(actual - expected).max() < 2e-4


def test_unet_sd1x_mixed_block_types_matches_diffusers():
    """Parity for the classic SD 1.x block pattern (plain last-down/first-up).

    Stable Diffusion 1.x uses ``CrossAttnDownBlock2D`` for the first down blocks
    and a plain ``DownBlock2D`` for the last, mirrored on the up path. This
    verifies the from-scratch UNet honors ``down_block_types`` / ``up_block_types``
    so cross-attention is present only where diffusers places it.
    """
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
    down_block_types = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
    up_block_types = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
    hf = HFUNet(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64, 64),
        cross_attention_dim=16,
        attention_head_dim=8,
        norm_num_groups=32,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    ).eval()
    sample = torch.randn(1, 4, 8, 8)
    timestep = torch.tensor([1])
    encoder_hidden_states = torch.randn(1, 4, 16)
    with torch.no_grad():
        expected = hf(sample, timestep, encoder_hidden_states).sample.numpy()

    config = UNet2DConfig(
        in_channels=4,
        out_channels=4,
        block_out_channels=(32, 64, 64),
        layers_per_block=1,
        norm_num_groups=32,
        cross_attention_dim=16,
        attention_head_dim=8,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        use_linear_projection=False,
    )
    module = UNet2DConditionModel(config)
    model = DenoisingTask().build(module, config)["model"]
    apply_weights(model, module.preprocess_weights(dict(hf.state_dict())))

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.onnx"
        onnx_ir.save(model, model_path)
        session = ort.InferenceSession(model_path)
        actual = session.run(
            None,
            {
                "sample": sample.numpy(),
                "timestep": timestep.numpy().astype(np.int64),
                "encoder_hidden_states": encoder_hidden_states.numpy(),
            },
        )[0]
    assert np.abs(actual - expected).max() < 2e-4


def test_unet_lora_gate_parity():
    """Runtime LoRA parity: gate=0 == diffusers base, gate=1 == diffusers+LoRA."""
    pytest.importorskip("diffusers")
    pytest.importorskip("peft")
    import onnx_ir
    import onnxruntime as ort
    import torch
    from diffusers import UNet2DConditionModel as HFUNet
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    from mobius._diffusers_configs import UNet2DConfig
    from mobius._weight_loading import apply_weights
    from mobius.models.unet import (
        UNet2DConditionModel,
        remap_diffusers_unet_lora,
    )
    from mobius.tasks._denoising import DenoisingTask

    torch.manual_seed(0)
    unet_kwargs = dict(
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
    )
    hf = HFUNet(**unet_kwargs).eval()
    base_state = dict(hf.state_dict())  # clean base weights (pre-adapter)

    sample = torch.randn(1, 4, 8, 8)
    timestep = torch.tensor([1])
    encoder_hidden_states = torch.randn(1, 4, 16)
    with torch.no_grad():
        base_out = hf(sample, timestep, encoder_hidden_states).sample.numpy()

    rank = 4
    hf.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,  # scale = alpha/rank = 1.0
            init_lora_weights=False,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    with torch.no_grad():
        lora_out = hf(sample, timestep, encoder_hidden_states).sample.numpy()
    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(hf))

    config = UNet2DConfig(
        in_channels=4,
        out_channels=4,
        block_out_channels=(32, 64),
        layers_per_block=1,
        norm_num_groups=32,
        cross_attention_dim=16,
        attention_head_dim=8,
        use_linear_projection=False,
        lora_adapters=(("test", rank, 1.0),),
    )
    module = UNet2DConditionModel(config)
    model = DenoisingTask().build(module, config)["model"]
    weights = module.preprocess_weights(base_state)
    weights.update(remap_diffusers_unet_lora(lora_state, "test"))
    apply_weights(model, weights)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.onnx"
        onnx_ir.save(model, model_path)
        session = ort.InferenceSession(model_path)
        feed = {
            "sample": sample.numpy(),
            "timestep": timestep.numpy().astype(np.int64),
            "encoder_hidden_states": encoder_hidden_states.numpy(),
        }
        off = session.run(None, {**feed, "lora_gate.test": np.array(0.0, dtype=np.float32)})[0]
        on = session.run(None, {**feed, "lora_gate.test": np.array(1.0, dtype=np.float32)})[0]

    # gate=0 disables the adapter (base); gate=1 applies it (diffusers+LoRA).
    assert np.abs(off - base_out).max() < 2e-4, np.abs(off - base_out).max()
    assert np.abs(on - lora_out).max() < 2e-4, np.abs(on - lora_out).max()
    # And the LoRA must actually change the output (non-trivial adapter).
    assert np.abs(base_out - lora_out).max() > 1e-3
