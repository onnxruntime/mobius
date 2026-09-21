# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicitly apply the legacy Mobius optimizer in rewrite-rule tests."""

from __future__ import annotations

from collections.abc import Callable

import onnx_ir as ir
import pytest

from mobius._optimizations import optimize_model

_MODEL_ROLE_MAP = {
    "model": "decoder",
    "decoder": "decoder",
    "vision_encoder": "vision",
    "embedding": "embedding",
    "encoder": "encoder",
    "audio_encoder": "encoder",
    "vision": "vision",
    "audio": "encoder",
    "speech": "encoder",
}


def _with_explicit_optimization(builder: Callable):
    def wrapped(*args, **kwargs):
        execution_provider = kwargs.get("execution_provider", "default")
        package = builder(*args, **kwargs)
        dtype = getattr(package.config, "dtype", ir.DataType.FLOAT)
        for name, model in package.items():
            optimize_model(
                model,
                ep=execution_provider,
                dtype=dtype,
                model_role=_MODEL_ROLE_MAP.get(name, "decoder"),
            )
        return package

    return wrapped


@pytest.fixture(autouse=True)
def apply_rewrites_explicitly(monkeypatch, request):
    """Keep rewrite tests focused on rules now that export no longer runs them."""
    for name in ("build", "build_from_module"):
        builder = getattr(request.module, name, None)
        if builder is not None:
            monkeypatch.setattr(request.module, name, _with_explicit_optimization(builder))
