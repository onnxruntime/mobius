# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import torch
import torch.nn.functional as torch_functional

from mobius._testing import create_test_builder, create_test_input
from mobius.components._fixed_siglip_sidecar import (
    ExactGELUMLPProjector,
    FixedResolutionSiglipMLPSidecar,
    map_fixed_siglip_sidecar_weight,
)
from mobius.models.clip import SigLIPVisionModel


def _siglip_config(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    image_size: int,
    patch_size: int,
):
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        rms_norm_eps=1e-6,
        hidden_act="gelu",
    )


def _run_module(module: torch.nn.Module, input_name: str, values: np.ndarray, seed: int):
    builder, op, graph = create_test_builder()
    input_value = create_test_input(builder, input_name, list(values.shape))
    output = module(op, input_value)
    graph.outputs.append(output)

    rng = np.random.default_rng(seed)
    state = {}
    for name, parameter in module.named_parameters():
        parameter_values = rng.normal(0.0, 0.08, tuple(parameter.shape)).astype(np.float32)
        if "layer_norm" in name or "layernorm" in name:
            if name.endswith(".weight"):
                parameter_values += 1.0
        parameter.const_value = ir.tensor(parameter_values)
        state[name] = parameter_values

    model = ir.Model(graph, ir_version=11)
    session = ort.InferenceSession(
        ir.serde.serialize_model(model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    (actual,) = session.run(None, {input_name: values})
    return actual, state, graph


def test_exact_gelu_projector_executes_and_matches_torch():
    projector = ExactGELUMLPProjector(4, 6, 5)
    features = np.linspace(-1.2, 1.3, 24, dtype=np.float32).reshape(2, 3, 4)

    actual, state, graph = _run_module(projector, "features", features, seed=11)

    reference = torch_functional.linear(
        torch.from_numpy(features),
        torch.from_numpy(state["linear_0.weight"]),
        torch.from_numpy(state["linear_0.bias"]),
    )
    reference = torch_functional.gelu(reference, approximate="none")
    reference = torch_functional.linear(
        reference,
        torch.from_numpy(state["linear_1.weight"]),
        torch.from_numpy(state["linear_1.bias"]),
    )
    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-5, atol=1e-6)
    gelu = next(node for node in graph if node.op_type == "Gelu")
    assert "approximate" not in gelu.attributes


def test_full_fixed_siglip_sidecar_executes_nonzero_pixels():
    tower = SigLIPVisionModel(
        _siglip_config(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=4,
            patch_size=2,
        )
    )
    sidecar = FixedResolutionSiglipMLPSidecar(tower, 4, 6)
    pixels = np.linspace(-0.9, 1.1, 48, dtype=np.float32).reshape(1, 3, 4, 4)

    actual, _, _ = _run_module(sidecar, "pixel_values", pixels, seed=19)

    assert actual.shape == (1, 4, 6)
    assert np.isfinite(actual).all()
    assert np.count_nonzero(actual) == actual.size


def test_janus_header_weight_names_have_exact_production_shapes():
    sidecar = FixedResolutionSiglipMLPSidecar(
        SigLIPVisionModel(
            _siglip_config(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
                image_size=384,
                patch_size=16,
            )
        ),
        vision_hidden_size=1024,
        projector_hidden_size=2048,
        output_hidden_size=2048,
    )
    parameters = {
        name: tuple(parameter.shape) for name, parameter in sidecar.named_parameters()
    }
    expected_source_shapes = {
        "v.patch_embd.weight": (1024, 3, 16, 16),
        "v.patch_embd.bias": (1024,),
        "v.position_embd.weight": (576, 1024),
        "v.post_ln.weight": (1024,),
        "v.post_ln.bias": (1024,),
        "mm.0.weight": (2048, 1024),
        "mm.0.bias": (2048,),
        "mm.1.weight": (2048, 2048),
        "mm.1.bias": (2048,),
    }
    for layer in (0, 23):
        prefix = f"v.blk.{layer}."
        expected_source_shapes.update(
            {
                prefix + "ln1.weight": (1024,),
                prefix + "ln1.bias": (1024,),
                prefix + "ln2.weight": (1024,),
                prefix + "ln2.bias": (1024,),
                prefix + "attn_q.weight": (1024, 1024),
                prefix + "attn_q.bias": (1024,),
                prefix + "attn_k.weight": (1024, 1024),
                prefix + "attn_k.bias": (1024,),
                prefix + "attn_v.weight": (1024, 1024),
                prefix + "attn_v.bias": (1024,),
                prefix + "attn_out.weight": (1024, 1024),
                prefix + "attn_out.bias": (1024,),
                prefix + "ffn_up.weight": (4096, 1024),
                prefix + "ffn_up.bias": (4096,),
                prefix + "ffn_down.weight": (1024, 4096),
                prefix + "ffn_down.bias": (1024,),
            }
        )

    for source_name, expected_shape in expected_source_shapes.items():
        target_name = map_fixed_siglip_sidecar_weight(source_name)
        assert target_name is not None
        assert parameters[target_name] == expected_shape

    assert map_fixed_siglip_sidecar_weight("v.blk.0.attn_qkv.weight") is None
    assert map_fixed_siglip_sidecar_weight("mm.2.weight") is None
