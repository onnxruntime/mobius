# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch

from mobius.models.mage_vl import MageVLForConditionalGeneration


def test_packed_token_table_routes_to_embedding_component() -> None:
    model = object.__new__(MageVLForConditionalGeneration)
    qweight = torch.zeros(32, 8, dtype=torch.uint8)
    scales = torch.ones(32, 1)

    result = model.preprocess_weights(
        {
            "model.language_model.embed_tokens.weight_qweight": qweight,
            "model.language_model.embed_tokens.weight_scales": scales,
        }
    )

    assert result["embedding.embed_tokens.weight_qweight"] is qweight
    assert result["embedding.embed_tokens.weight_scales"] is scales
    assert not any(name.startswith("decoder.") for name in result)
