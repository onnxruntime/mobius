# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exercise production registration and its standard InlinePass boundary."""

from __future__ import annotations

import onnx_ir as ir

from mobius._optimizations import _inline_function_bodies
from mobius.functions import register_function_bodies


def inline_native_calls(model: ir.Model) -> ir.Model:
    register_function_bodies(model, moe_fallback=True)
    _inline_function_bodies(
        model, lambda fn: fn.domain == "com.microsoft" and fn.name in {"MoE", "QMoE"}
    )
    return model
