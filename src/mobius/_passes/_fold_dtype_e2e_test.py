# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end regression tests for fp16 dtype preservation through the fold passes.

These complement the unit-level coverage in ``_dtype_utils_test.py`` and the
pass-level coverage in ``_fold_concat_test.py`` / ``_fold_transpose_test.py`` by
driving the *real* export pipeline (``build_from_module`` + ``apply_weights``)
rather than hand-built single-pass graphs.

The df203cc fix has two parts, and this module guards both at the export level:

1. **dtype stamping (type-stamping path).** The fold passes stamp the *resolved*
   fp16 dtype on the new packed / transposed initializer (its ``TensorType`` and
   its ``LazyTensor``) instead of inheriting the unset type of the rewrite-produced
   ``Concat`` intermediate or defaulting to ``FLOAT``. Guarded by
   ``test_packed_qkv_weights_stay_fp16_after_export`` and
   ``test_all_folded_matmul_weights_stay_fp16_after_export``: in the real fp16 GQA
   export the PackQKV rewrite emits a ``MatMul(hidden, Transpose(Concat(W_q, W_k,
   W_v)))`` whose ``Concat`` output carries no declared dtype, so without the
   stamping the chained FoldConcat -> FoldTranspose widens the packed-QKV weight
   to fp32.

2. **const_value fallback.** ``_dtype_utils.initializer_dtype`` resolves the dtype
   from ``const_value`` when an initializer's declared ``type`` was dropped during
   graph building. Guarded by
   ``test_dropped_declared_dtype_falls_back_to_fp16_const``, which reproduces that
   dropped-dtype condition on the packed-QKV weights (as the original regression
   exhibited) so the const_value fallback is the only thing keeping the folded
   weights fp16.

Either failure silently widens an fp16 model to fp32 packed weights, which
onnxruntime rejects at load time with a fp16/fp32 ``MatMul`` type-mismatch
(``Type parameter (T) of Optype (MatMul) bound to different types``).

The in-memory dtype assertions read the IR ``const_value`` metadata. The
regression's *ground-truth* symptom, however, is a serialize-time corruption:
the fold passes write the fp16 *bytes* under a fp32 declaration, so the persisted
initializer is a fp32-declared tensor backed by half-width data. An in-memory
``const_value.numpy()`` can stay fp16 and miss this (only the declared dtype
widens), so ``test_packed_qkv_survives_serialization_roundtrip`` and the
serialize check in ``test_dropped_declared_dtype_falls_back_to_fp16_const`` drive
the real export artifact (``ir.save`` with external data — the same
``model.onnx`` + ``model.onnx.data`` layout the fp16 export produces), reload it,
and assert the folded weights reload as fp16 with their bytes intact.

The models are tiny and fully synthetic (no HuggingFace download, no GPU and no
onnxruntime execution — only the ONNX graph is constructed, serialized and
inspected), so the tests fit the per-PR CI tier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._optimizations import fold_initializers_after_weights
from mobius._registry import registry
from mobius._weight_loading import apply_weights


def _make_fp16_llama_config() -> ArchitectureConfig:
    """A tiny llama config whose Q/K/V weights are packable (no QK norm).

    ``dtype=FLOAT16`` routes the build through the fp16 packed-QKV path that the
    fold passes must preserve.
    """
    return ArchitectureConfig(
        # Structural invariants that make the fp16 PackQKV fold fire:
        #   num_attention_heads * head_dim == hidden_size  (4 * 16 == 64), and
        #   num_key_value_heads < num_attention_heads       (2 < 4, i.e. GQA).
        # The remaining fields are arbitrary small values kept lightweight.
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10000.0,
        pad_token_id=0,
        dtype=ir.DataType.FLOAT16,
    )


def _build_fp16_decoder(config: ArchitectureConfig) -> ir.Model:
    """Build an fp16 llama decoder graph with packed QKV, weights not yet loaded.

    Uses the ``cuda`` execution provider so the GQA + PackQKV rewrites pack
    Q/K/V into a single ``MatMul(hidden, Transpose(Concat(...)))``. Only the ONNX
    graph is constructed — no GPU or onnxruntime execution is required. The fold
    passes have not run yet (they run when weights are loaded).
    """
    return build_from_module(
        registry.get("llama")(config),
        config,
        execution_provider="cuda",
    )["model"]


def _fp16_weight_tensors(model: ir.Model) -> dict[str, torch.Tensor]:
    """fp16 tensors for every uninitialised parameter, mirroring an fp16 checkpoint.

    Weight initializers always carry fully-static integer shapes (they are
    concrete parameter tensors, never symbolic), so ``int(d)`` is safe here.
    """
    return {
        name: torch.randn(*[int(d) for d in init.shape]).to(torch.float16)
        for name, init in model.graph.initializers.items()
        if init.const_value is None
    }


def _matmul_weight_initializers(model: ir.Model) -> list[ir.Value]:
    """Return every ``MatMul`` second input that is a graph initializer."""
    initializers = model.graph.initializers
    weights: list[ir.Value] = []
    for node in model.graph:
        if node.op_type != "MatMul":
            continue
        weight = node.inputs[1]
        if weight is not None and weight.name in initializers:
            weights.append(weight)
    return weights


def _assert_matmul_weights_roundtrip_as_fp16(model: ir.Model, tmp_path: Path) -> None:
    """Assert every folded MatMul weight reloads as fp16 with its bytes intact.

    Serializes ``model`` through the production external-data path. This is the
    ground-truth df203cc guard. The regression writes the packed / transposed fp16 *bytes* under a fp32 declaration, a corruption an in-memory
    ``const_value`` check can mask (the backing array stays fp16 while only the
    declared dtype widens) but that surfaces on serialize -> reload as a
    fp32-declared initializer whose bytes no longer match the fp16 source.
    ``ir.save`` with ``external_data`` mirrors the real fp16 export's
    ``model.onnx`` + ``model.onnx.data`` layout, exactly where the dtype bug
    corrupts the weight bytes.
    """
    before = {w.name: w.const_value.numpy() for w in _matmul_weight_initializers(model)}
    assert before, "Expected the export to contain MatMul weight initializers to round-trip"

    model_path = tmp_path / "model.onnx"
    ir.save(model, model_path, external_data="model.onnx.data")
    assert (tmp_path / "model.onnx.data").exists(), (
        "fp16 weights should be externalized, exercising the real export's "
        "model.onnx + model.onnx.data save path where the dtype bug corrupts bytes"
    )

    reloaded = ir.load(model_path)
    for weight in _matmul_weight_initializers(reloaded):
        assert weight.dtype == ir.DataType.FLOAT16, (
            f"Reloaded MatMul weight {weight.name!r} serialized as {weight.dtype} "
            f"(expected FLOAT16): fp16 bytes were written under a fp32 declaration "
            f"— the df203cc serialize-time corruption."
        )
        np.testing.assert_array_equal(
            weight.const_value.numpy(),
            before[weight.name],
            err_msg=(
                f"Reloaded MatMul weight {weight.name!r} bytes changed across "
                f"serialize -> reload; the fp16 packed weight was silently corrupted "
                f"(fp16 data written under a fp32 dtype)."
            ),
        )


@pytest.fixture(scope="module")
def fp16_export() -> tuple[ArchitectureConfig, ir.Model]:
    """A real fp16 packed-QKV export: build + ``apply_weights`` (folds inside).

    Module-scoped so the (read-only) realistic-export assertions share a single
    build instead of rebuilding per test.
    """
    config = _make_fp16_llama_config()
    model = _build_fp16_decoder(config)
    # apply_weights assigns the fp16 weights and then runs
    # fold_initializers_after_weights, folding the Transpose/Concat weight nodes.
    apply_weights(model, _fp16_weight_tensors(model))
    return config, model


class TestFp16FoldDtypeE2E:
    def test_packed_qkv_weights_stay_fp16_after_export(
        self, fp16_export: tuple[ArchitectureConfig, ir.Model]
    ) -> None:
        """An fp16 packed-QKV export keeps fp16 packed/transposed weights.

        Guards the df203cc dtype-stamping path: the chained
        ``FoldConcatInitializersPass`` (the Q/K/V pack) and
        ``FoldTransposedInitializerPass`` (the weight transpose) must carry the
        fp16 dtype through the rewrite-produced ``Concat`` intermediate instead
        of defaulting it to ``FLOAT``.
        """
        config, model = fp16_export

        # The packed path must actually have run, otherwise the dtype assertion
        # below would pass vacuously (e.g. if the build stopped packing QKV).
        gqa_nodes = [n for n in model.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == config.num_hidden_layers, (
            "Expected one packed GroupQueryAttention per layer; the fp16 GQA "
            "PackQKV path was not exercised, so this regression guard is vacuous."
        )

        for gqa in gqa_nodes:
            matmul = gqa.inputs[0].producer()
            assert matmul is not None and matmul.op_type == "MatMul", (
                "Packed GQA projection should be produced by a MatMul"
            )
            weight = matmul.inputs[1]
            assert weight is not None, "Packed projection MatMul is missing its weight input"
            # FoldConcat + FoldTranspose ran inside apply_weights, so the packed
            # weight is now a plain folded initializer (no producer node left).
            assert weight.producer() is None, (
                "Packed QKV weight should be a folded initializer after export, "
                "not the output of a residual Transpose/Concat node"
            )
            assert weight.const_value is not None
            assert weight.const_value.dtype == ir.DataType.FLOAT16, (
                f"Packed QKV weight {weight.name!r} widened to "
                f"{weight.const_value.dtype} (expected FLOAT16); the fp16 fold "
                f"dtype fix (df203cc) has regressed."
            )
            assert weight.dtype == ir.DataType.FLOAT16

    def test_all_folded_matmul_weights_stay_fp16_after_export(
        self, fp16_export: tuple[ArchitectureConfig, ir.Model]
    ) -> None:
        """Every folded MatMul weight in an fp16 export stays fp16.

        Broader guard than the packed-QKV case: it also covers the standalone
        ``o_proj`` and MLP weight transposes folded by
        ``FoldTransposedInitializerPass``. A single fp32-widened weight here is
        exactly what makes onnxruntime reject the fp16 model at load time.
        """
        _, model = fp16_export

        weights = _matmul_weight_initializers(model)
        assert weights, "Expected the export to contain MatMul weight initializers"
        # Anti-vacuity: at least one weight must be a folded transposed initializer
        # (FoldTransposedInitializerPass names them ``..._t``). Without this, the
        # test could pass simply because no folding ran.
        assert any(weight.name.endswith("_t") for weight in weights), (
            "Expected at least one folded transposed weight (``..._t``); the fold "
            "passes did not run, so this regression guard would be vacuous."
        )

        # fp32 widening (FLOAT) is the specific documented failure mode of the
        # df203cc regression — the fold passes defaulting a dropped declared dtype
        # to FLOAT — so we flag exactly that rather than any non-fp16 dtype.
        widened = [
            weight.name
            for weight in weights
            if weight.const_value is not None and weight.const_value.dtype == ir.DataType.FLOAT
        ]
        assert not widened, (
            f"fp16 export produced fp32 MatMul weights after folding: {widened}. "
            f"The fold passes must preserve fp16 (df203cc)."
        )

    def test_packed_qkv_survives_serialization_roundtrip(
        self, fp16_export: tuple[ArchitectureConfig, ir.Model], tmp_path: Path
    ) -> None:
        """The fp16 export's folded weights survive serialize -> reload as fp16.

        Ground-truth guard for the dtype-stamping path. The in-memory assertions
        above read the IR ``const_value`` metadata; this drives the real export
        artifact (``ir.save`` with external data, like the production
        ``model.onnx`` + ``model.onnx.data``) and reloads it, the only check that
        catches the df203cc serialize-time fp16-under-fp32 byte corruption — an
        in-memory ``const_value.numpy()`` can stay fp16 and miss it.
        """
        _, model = fp16_export
        _assert_matmul_weights_roundtrip_as_fp16(model, tmp_path)

    def test_dropped_declared_dtype_falls_back_to_fp16_const(self, tmp_path: Path) -> None:
        """Folding fp16 weights whose declared dtype was dropped stays fp16.

        Guards the df203cc ``initializer_dtype`` const_value fallback. The
        original regression arose because graph building left the packed-QKV
        weights with fp16 ``const_value`` but no declared ``type``; the fold
        passes then defaulted to ``FLOAT``. Here we reproduce that exact
        condition on a real fp16 export — load fp16 weights, clear the declared
        type on the packed-QKV ``Concat`` inputs, then run the real fold
        orchestration — so the const_value fallback is the *only* thing that can
        keep the folded weights fp16.

        ``fold_initializers_after_weights`` is invoked directly (rather than via
        ``apply_weights``) so the dropped-dtype condition can be injected between
        loading the weights and folding; it is the same orchestration
        ``apply_weights`` runs internally.
        """
        config = _make_fp16_llama_config()
        model = _build_fp16_decoder(config)

        # Load fp16 weights onto the initializers (mirrors apply_weights' assign
        # step) without yet folding.
        for name, tensor in _fp16_weight_tensors(model).items():
            model.graph.initializers[name].const_value = ir.tensor(tensor.numpy())

        # Reproduce the dropped-declared-dtype condition: clear the declared type
        # on the packed-QKV weight initializers (the Concat inputs) while keeping
        # their fp16 const_value.
        dropped: list[str] = []
        for node in model.graph:
            if node.op_type != "Concat":
                continue
            for inp in node.inputs:
                if (
                    inp is not None
                    and inp.name in model.graph.initializers
                    and inp.const_value is not None
                ):
                    inp.type = None
                    dropped.append(inp.name)
        assert dropped, (
            "Expected packed-QKV Concat inputs to drop declared dtype on; the "
            "fp16 PackQKV path was not exercised, so this guard would be vacuous."
        )

        fold_initializers_after_weights(model)

        gqa_nodes = [n for n in model.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == config.num_hidden_layers
        for gqa in gqa_nodes:
            matmul = gqa.inputs[0].producer()
            assert matmul is not None and matmul.op_type == "MatMul", (
                "Packed GQA projection should be produced by a MatMul"
            )
            weight = matmul.inputs[1]
            assert weight is not None, "Packed projection MatMul is missing its weight input"
            assert weight.const_value is not None
            assert weight.const_value.dtype == ir.DataType.FLOAT16, (
                f"Packed QKV weight {weight.name!r} widened to "
                f"{weight.const_value.dtype} when its declared dtype was dropped; "
                f"the initializer_dtype const_value fallback (df203cc) regressed."
            )

        # Ground-truth check: the const_value fallback must also hold through the
        # real serialize -> reload path. Reverting only the ``initializer_dtype``
        # call-sites widens these dropped-dtype weights and corrupts their bytes
        # here even though the type-stamp keeps the realistic-export tests green.
        _assert_matmul_weights_roundtrip_as_fp16(model, tmp_path)
