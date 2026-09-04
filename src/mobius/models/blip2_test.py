# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import torch

from mobius.models.blip2 import Blip2Model


def test_component_quantization_routes_packed_opt_namespaces() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            component_quantization={"decoder": object()},
            quantization=None,
            tie_word_embeddings=False,
        )
    )
    qweight = torch.zeros(32, 8, dtype=torch.uint8)

    result = Blip2Model.preprocess_weights(
        model,
        {
            "language_model.model.decoder.layers.0.self_attn.q_proj.weight_qweight": (
                torch.zeros(16, 8, dtype=torch.uint8)
            ),
            "language_model.model.decoder.embed_tokens.weight_qweight": qweight,
            "vision_model.encoder.layers.0.mlp.fc1.weight_qweight": torch.zeros(
                32, 8, dtype=torch.uint8
            ),
            "qformer.encoder.layer.0.attention.output.dense.weight_qweight": torch.zeros(
                32, 8, dtype=torch.uint8
            ),
        },
    )

    assert "decoder.model.layers.0.self_attn.q_proj.weight_qweight" in result
    assert result["embedding.embed_tokens.weight_qweight"] is qweight
    assert (
        "vision_encoder.vision_model.vision_model.encoder.layers.0.mlp.up_proj.weight_qweight"
        in result
    )
    assert "vision_encoder.qformer.layers.0.self_attn.out_proj.weight_qweight" in result
    assert not any(name.startswith("language_model.") for name in result)
