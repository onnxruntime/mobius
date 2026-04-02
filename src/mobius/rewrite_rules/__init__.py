# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rules for optional graph transformations.

These rules are **not applied by default**. Users can apply them after
model export to replace standard ONNX patterns with optimised custom ops.

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
    "cast_int64_to_int32_rules",
    "decompose_if_pass",
    "decompose_simplified_layer_norm_rules",
    "decompose_skip_layer_norm_rules",
    "eliminate_shape_rules",
    "fused_matmul_rules",
    "gelu_fusion_rules",
    "group_query_attention_rules",
    "layer_norm_fusion_rules",
    "packed_attention_rules",
    "separate_rope_rules",
    "skip_layer_norm_rules",
    "skip_norm_rules",
    "unpack_qkv_rules",
]

from mobius.rewrite_rules._bias_gelu import bias_gelu_rules
from mobius.rewrite_rules._cast_int64_to_int32 import cast_int64_to_int32_rules
from mobius.rewrite_rules._decompose_if import decompose_if_pass
from mobius.rewrite_rules._decompose_layer_norm import (
    decompose_simplified_layer_norm_rules,
)
from mobius.rewrite_rules._decompose_skip_layer_norm import (
    decompose_skip_layer_norm_rules,
)
from mobius.rewrite_rules._eliminate_shape import eliminate_shape_rules
from mobius.rewrite_rules._fused_matmul import fused_matmul_rules
from mobius.rewrite_rules._gelu_fusion import gelu_fusion_rules
from mobius.rewrite_rules._group_query_attention import (
    group_query_attention_rules,
)
from mobius.rewrite_rules._layer_norm_fusion import (
    layer_norm_fusion_rules,
)
from mobius.rewrite_rules._packed_attention import packed_attention_rules
from mobius.rewrite_rules._separate_rope import separate_rope_rules
from mobius.rewrite_rules._skip_layer_norm import skip_layer_norm_rules
from mobius.rewrite_rules._skip_norm import skip_norm_rules
from mobius.rewrite_rules._unpack_qkv import unpack_qkv_rules
