# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Every ONNX graph this producer emits carries a modern IR and opset floor.

Mobius constructs graphs declaratively with ``onnx_ir`` and targets a single
opset (``mobius._constants.OPSET_VERSION``) and IR version. That is only worth
anything if it cannot quietly lapse: a component built through a path that
never set ``opset_imports`` would inherit an *empty* default domain, and a
model built without an explicit ``ir_version`` would fall back to whatever the
serializer picks — either would still serialize, still load, and still pass
every feature test while silently shipping a stale contract to the runtime.

So this test regenerates every conformance package — decoder, static-cache,
VLM, shared-state pixel flow, diffusion (both variants), TTS, speculative,
masked, video, codec, the two protein encoders, the adapter package, and every
attached policy artifact — and asserts, for each ``.onnx`` file, that it
declares IR version >= 11 and a default-domain opset >= 24. It loads through
``onnx_ir`` only; it uses no protobuf APIs and carries no allowlist. A single
graph built through a path that inherits empty defaults fails here rather than
reaching a runtime.
"""

from __future__ import annotations

import os

import onnx_ir as ir

from mobius._constants import OPSET_VERSION

# The IR version every Mobius model declares. Set once at construction in
# ``mobius.tasks._base._make_model`` and ``mobius.generation._policy_components``
# (both ``ir.Model(graph, ir_version=11)``); restated here as the enforced floor
# so a regression to an older, serializer-chosen default is a failure.
MINIMUM_IR_VERSION = 11

# The default ("") domain floor. ``OPSET_VERSION`` is the single opset the whole
# package targets; a graph below it — or one that never imported the default
# domain at all, and so reports ``None`` — is a stale or empty contract.
MINIMUM_DEFAULT_OPSET = OPSET_VERSION


def _onnx_models(root: str) -> list[str]:
    """Every serialized ONNX graph in the materialized package tree."""
    return sorted(
        os.path.join(directory, name)
        for directory, _subdirs, files in os.walk(root)
        for name in files
        if name.endswith(".onnx")
    )


def test_every_generated_onnx_declares_modern_ir_and_opset(materialized_workflow_packages):
    paths = _onnx_models(materialized_workflow_packages)
    # A tree with no graphs would pass every per-file assertion vacuously; the
    # generator emits well over a hundred, so a near-empty tree is itself a bug.
    assert len(paths) > 100, f"expected the generator to emit many graphs, found {len(paths)}"

    violations: list[str] = []
    for path in paths:
        model = ir.load(path)
        relative = os.path.relpath(path, materialized_workflow_packages)
        ir_version = model.ir_version
        default_opset = model.opset_imports.get("")
        if ir_version is None or ir_version < MINIMUM_IR_VERSION:
            violations.append(f"{relative}: ir_version={ir_version} < {MINIMUM_IR_VERSION}")
        if default_opset is None or default_opset < MINIMUM_DEFAULT_OPSET:
            violations.append(
                f"{relative}: default opset={default_opset} < {MINIMUM_DEFAULT_OPSET}"
            )

    assert not violations, "ONNX graphs below the IR/opset floor:\n" + "\n".join(violations)
