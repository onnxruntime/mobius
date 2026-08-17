# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._configs import VisionConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius._weight_loading import apply_weights
from mobius.components import Siglip2NaFlexVisionModel
from mobius.tasks._base import _make_graph, _make_model

_HIDDEN_SIZE = 32
_PATCH_SIZE = 4
_NUM_PATCHES = 16


def _vision_config() -> VisionConfig:
    return VisionConfig(
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        patch_size=_PATCH_SIZE,
        norm_eps=1e-6,
        in_channels=3,
        num_position_embeddings=_NUM_PATCHES,
        hidden_act="gelu_pytorch_tanh",
    )


def _build_tower() -> tuple[Siglip2NaFlexVisionModel, ir.Model]:
    tower = Siglip2NaFlexVisionModel(_vision_config())
    graph, builder = _make_graph(name="siglip2_naflex")
    num_images = ir.SymbolicDim("num_images")
    max_patches = ir.SymbolicDim("max_num_patches")
    pixel_values = builder.input(
        "pixel_values",
        dtype=ir.DataType.FLOAT,
        shape=[num_images, max_patches, 3 * _PATCH_SIZE * _PATCH_SIZE],
    )
    pixel_attention_mask = builder.input(
        "pixel_attention_mask",
        dtype=ir.DataType.INT64,
        shape=[num_images, max_patches],
    )
    spatial_shapes = builder.input(
        "spatial_shapes",
        dtype=ir.DataType.INT64,
        shape=[num_images, 2],
    )
    hidden_states = tower(
        builder.op,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
        spatial_shapes=spatial_shapes,
    )
    builder.add_output(hidden_states, "last_hidden_state")
    return tower, _make_model(graph)


def test_parameter_names_follow_huggingface_siglip2():
    _, model = _build_tower()
    names = set(model.graph.initializers)
    assert "embeddings.patch_embedding.weight" in names
    assert "embeddings.patch_embedding.bias" in names
    # The learned table must survive even though it is only consumed inside
    # the Scan body that resizes it.
    assert "embeddings.position_embedding.weight" in names
    assert "encoder.layers.0.layer_norm1.weight" in names
    assert "encoder.layers.1.self_attn.out_proj.weight" in names
    assert "post_layernorm.bias" in names


def test_square_position_grid_is_required():
    config = _vision_config()
    config.num_position_embeddings = 15
    with pytest.raises(ValueError, match="square position grid"):
        Siglip2NaFlexVisionModel(config)


@pytest.mark.parametrize("shapes", [[(4, 4)], [(2, 6), (6, 2), (3, 5)]])
def test_naflex_tower_matches_transformers(shapes):
    """L3: padded variable-resolution tower matches HF on random weights."""
    from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

    torch.manual_seed(0)
    hf_config = Siglip2VisionConfig(
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        patch_size=_PATCH_SIZE,
        num_patches=_NUM_PATCHES,
        layer_norm_eps=1e-6,
        vision_use_head=False,
    )
    hf_model = Siglip2VisionModel(hf_config).eval()

    _, model = _build_tower()
    weights = {
        name.replace(".mlp.fc1.", ".mlp.up_proj.").replace(
            ".mlp.fc2.", ".mlp.down_proj."
        ): value
        for name, value in hf_model.state_dict().items()
    }
    parameters = {name for name in model.graph.initializers if not name.startswith("const")}
    assert set(weights) == parameters
    apply_weights(model, weights)

    rng = np.random.default_rng(3)
    max_patches = max(h * w for h, w in shapes)
    patch_dim = 3 * _PATCH_SIZE * _PATCH_SIZE
    pixel_values = np.zeros((len(shapes), max_patches, patch_dim), dtype=np.float32)
    mask = np.zeros((len(shapes), max_patches), dtype=np.int64)
    for index, (height, width) in enumerate(shapes):
        count = height * width
        pixel_values[index, :count] = rng.standard_normal((count, patch_dim))
        mask[index, :count] = 1
    spatial_shapes = np.array(shapes, dtype=np.int64)

    with torch.no_grad():
        expected = hf_model(
            pixel_values=torch.from_numpy(pixel_values),
            pixel_attention_mask=torch.from_numpy(mask),
            spatial_shapes=torch.from_numpy(spatial_shapes),
        ).last_hidden_state.numpy()

    session = OnnxModelSession(model)
    actual = session.run(
        {
            "pixel_values": pixel_values,
            "pixel_attention_mask": mask,
            "spatial_shapes": spatial_shapes,
        }
    )["last_hidden_state"]
    session.close()

    # Padded positions are masked out of attention and dropped downstream, so
    # only the valid patches are required to agree.
    valid = mask.astype(bool)
    np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-4, atol=1e-4)
