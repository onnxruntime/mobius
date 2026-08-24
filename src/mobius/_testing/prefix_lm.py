# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PrefixLM ``token_type_ids`` contract shared by every test/golden path.

Models whose exported graph declares ``requires_token_type_ids`` were
pre-trained with a PrefixLM mask: prompt tokens attend bidirectionally to each
other, generated tokens attend causally.  ``sapientinc/HRM-Text-1B``'s model
card spells the correct call out explicitly — mark the *entire* prompt as one
bidirectional prefix block (``token_type_ids = ones_like(input_ids)``) and
leave generated positions at ``0``.

Both sides of every comparison must use the same rule or the numbers are
meaningless, so the rule lives here exactly once: the HuggingFace reference is
driven from :func:`model_type_uses_prefix_lm` and the exported graph from
:func:`graph_uses_prefix_lm`, and both hand out the same tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnx_ir as ir

TOKEN_TYPE_IDS = "token_type_ids"


def model_type_uses_prefix_lm(model_type: str) -> bool:
    """Whether *model_type*'s registered class consumes ``token_type_ids``."""
    from mobius._registry import registry

    try:
        model_cls = registry.get(model_type)
    except Exception:
        return False
    return bool(getattr(model_cls, "requires_token_type_ids", False))


def graph_uses_prefix_lm(model: ir.Model) -> bool:
    """Whether an exported graph declares a ``token_type_ids`` input."""
    return any(value.name == TOKEN_TYPE_IDS for value in model.graph.inputs)


def prompt_token_type_ids(input_ids: np.ndarray) -> np.ndarray:
    """``token_type_ids`` for a prompt that is one bidirectional prefix block."""
    return np.ones_like(input_ids, dtype=np.int64)


def generated_token_type_ids(input_ids: np.ndarray) -> np.ndarray:
    """``token_type_ids`` for generated positions: causal, never prefix."""
    return np.zeros_like(input_ids, dtype=np.int64)
