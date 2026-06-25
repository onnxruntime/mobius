# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Rewrite rules for graph transformations.

These rules are applied automatically by :func:`~mobius._optimizations.optimize_model`
for the relevant execution provider and dtype. They can also be applied
manually after model export:

Example::

    from mobius.rewrite_rules import (
        bias_gelu_rules,
        gelu_fusion_rules,
        layer_norm_fusion_rules,
        packed_attention_rules,
        skip_layer_norm_rules,
        skip_norm_rules,
    )
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen3-0.6B")
    rewrite(model, pattern_rewrite_rules=packed_attention_rules())
    rewrite(model, pattern_rewrite_rules=skip_norm_rules())

    gpt2_model = build("openai-community/gpt2")
    rewrite(gpt2_model, pattern_rewrite_rules=skip_layer_norm_rules())
    rewrite(gpt2_model, pattern_rewrite_rules=bias_gelu_rules())
"""

__all__ = [
    "bias_gelu_rules",
    "gelu_fusion_rules",
    "group_query_attention_rules",
    "htp_rank4_rmsnorm_rules",
    "layer_norm_fusion_rules",
    "pack_qkv_for_gqa_rules",
    "packed_attention_rules",
    "separate_rope_rules",
    "skip_layer_norm_rules",
    "skip_norm_rules",
    "unpack_qkv_rules",
]

from mobius.rewrite_rules._bias_gelu import bias_gelu_rules
from mobius.rewrite_rules._gelu_fusion import gelu_fusion_rules
from mobius.rewrite_rules._group_query_attention import (
    group_query_attention_rules,
    pack_qkv_for_gqa_rules,
)
from mobius.rewrite_rules._htp_rank4_rmsnorm import htp_rank4_rmsnorm_rules
from mobius.rewrite_rules._layer_norm_fusion import (
    layer_norm_fusion_rules,
)
from mobius.rewrite_rules._packed_attention import packed_attention_rules
from mobius.rewrite_rules._separate_rope import separate_rope_rules
from mobius.rewrite_rules._skip_layer_norm import skip_layer_norm_rules
from mobius.rewrite_rules._skip_norm import skip_norm_rules
from mobius.rewrite_rules._unpack_qkv import unpack_qkv_rules
