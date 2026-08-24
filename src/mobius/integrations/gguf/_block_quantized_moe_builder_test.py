# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder-side wiring for the native-block sparse-MoE fusion + honesty gate.

These tests exercise the ``build_from_gguf`` post-export integration points
(:func:`_fuse_native_block_moe`, :func:`_assert_sparse_moe_graph`, and the
:func:`_perproj_bqmoe_v2_runtime_available` runtime-capability probe) on synthetic
exported packages -- i.e. the graph state the builder hands to the fusion after
``pkg.apply_weights``. They are distinct from the rewrite-rule unit tests: the
focus here is the *builder's* honesty policy (default fail-closed, per-projection
v2 runtime gate, opt-in dense retention, and the final-graph-state backstop),
byte-preserving expert banks, and deterministic external data.

Native IQ/GGUF blocks are codebook-based and cannot be dequantized in NumPy, so
correctness is proven structurally (the routed expert storm collapses; the shared
expert is untouched) and by byte-preservation (the native blocks are stacked
verbatim). GLM-5.2 UD-IQ1 layers mix native formats across their fc1/fc2/fc3
banks, so they require the ``block_layout_version=2`` per-projection ABI; until
that ships in the runtime the builder must typed-reject them rather than emit an
unrunnable node.
"""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius._constants import OPSET_VERSION
from mobius._model_package import ModelPackage
from mobius.integrations.gguf import SparseMoEExportError
from mobius.integrations.gguf._builder import (
    _assert_sparse_moe_graph,
    _fuse_native_block_moe,
    _perproj_bqmoe_v2_runtime_available,
    _routed_dense_block_matmul_nodes,
)
from mobius.rewrite_rules._block_quantized_moe_fusion import _NATIVE_BLOCK_FORMATS
from mobius.rewrite_rules._block_quantized_moe_fusion_test import (
    E,
    H,
    _build_dense_graph,
)

_V2_ENV = "MOBIUS_ENABLE_BQMOE_PERPROJ_V2"


def _pkg(**kwargs) -> tuple[ModelPackage, dict]:
    model, weights = _build_dense_graph(**kwargs)
    return ModelPackage({"model": model}), weights


def _count(pkg: ModelPackage, op_type: str) -> int:
    return sum(1 for model in pkg.values() for n in model.graph if n.op_type == op_type)


def _moe(pkg: ModelPackage) -> ir.Node:
    return next(
        n for model in pkg.values() for n in model.graph if n.op_type == "BlockQuantizedMoE"
    )


# --------------------------------------------------------------------------- #
# Runtime-capability probe                                                    #
# --------------------------------------------------------------------------- #
def test_perproj_v2_runtime_probe_defaults_false(monkeypatch) -> None:
    monkeypatch.delenv(_V2_ENV, raising=False)
    assert _perproj_bqmoe_v2_runtime_available() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_perproj_v2_runtime_probe_env_opt_in(monkeypatch, value) -> None:
    monkeypatch.setenv(_V2_ENV, value)
    assert _perproj_bqmoe_v2_runtime_available() is True


# --------------------------------------------------------------------------- #
# Fusion wiring on the final graph state                                      #
# --------------------------------------------------------------------------- #
def test_builder_fuses_uniform_native_moe(monkeypatch) -> None:
    """A uniform-format routed storm collapses to one v1 BQMoE per layer.

    The always-active shared expert (3 native BlockQuantizedMatMul projections)
    must stay dense -- it is not routed and must not be swept into the bank.
    """
    monkeypatch.delenv(_V2_ENV, raising=False)
    pkg, _ = _pkg(gate_fmt="iq4_xs", up_fmt="iq4_xs", down_fmt="iq4_xs")
    fused = _fuse_native_block_moe(pkg, allow_dense=False)
    _assert_sparse_moe_graph(pkg, source="uniform.gguf", allow_dense=False)

    assert fused == 1
    assert _count(pkg, "BlockQuantizedMoE") == 1
    # 4 routed experts * 3 projections collapse; only the shared expert's 3
    # projections survive.
    assert _count(pkg, "BlockQuantizedMatMul") == 3
    attrs = _moe(pkg).attributes
    assert "block_layout_version" not in attrs  # v1 (uniform)
    assert attrs["format"].value == "iq4_xs"


def test_builder_mixed_format_typed_rejects_without_v2_runtime(monkeypatch) -> None:
    """GLM-5.2-style mixed formats need v2; with no v2 runtime, fail closed.

    The reject must be atomic: the dense graph is left untouched so the caller
    sees the original storm, never a half-rewritten graph.
    """
    monkeypatch.delenv(_V2_ENV, raising=False)
    pkg, _ = _pkg(gate_fmt="iq1_s", up_fmt="iq1_s", down_fmt="iq4_xs")
    equal_before = _count(pkg, "Equal")

    with pytest.raises(SparseMoEExportError, match=r"block_layout_version=2"):
        _fuse_native_block_moe(pkg, allow_dense=False)

    assert _count(pkg, "BlockQuantizedMoE") == 0
    assert _count(pkg, "Equal") == equal_before  # untouched


def test_builder_mixed_format_fuses_with_v2_runtime(monkeypatch) -> None:
    """With the v2 runtime opted in, a mixed-format layer fuses to a v2 node."""
    monkeypatch.setenv(_V2_ENV, "1")
    pkg, _ = _pkg(gate_fmt="iq1_s", up_fmt="iq1_s", down_fmt="iq4_xs")
    fused = _fuse_native_block_moe(pkg, allow_dense=False)
    _assert_sparse_moe_graph(pkg, source="mixed.gguf", allow_dense=False)

    assert fused == 1
    attrs = _moe(pkg).attributes
    assert attrs["block_layout_version"].value == 2
    assert attrs["fc1_format"].value == "iq1_s"
    assert attrs["fc2_format"].value == "iq4_xs"


def test_builder_allow_dense_retains_dense_fallback(monkeypatch) -> None:
    """Opting into the dense fallback keeps the runnable per-expert storm."""
    monkeypatch.delenv(_V2_ENV, raising=False)
    pkg, _ = _pkg(gate_fmt="iq1_s", up_fmt="iq1_s", down_fmt="iq4_xs")
    equal_before = _count(pkg, "Equal")

    fused = _fuse_native_block_moe(pkg, allow_dense=True)
    # The gate is a no-op under the opt-in (the fusion already warned).
    _assert_sparse_moe_graph(pkg, source="mixed.gguf", allow_dense=True)

    assert fused == 0
    assert _count(pkg, "BlockQuantizedMoE") == 0
    assert _count(pkg, "Equal") == equal_before  # dense fallback preserved


def test_builder_bank_bytes_are_byte_preserved(monkeypatch) -> None:
    """The emitted fc2 expert bank is the source down blocks stacked verbatim."""
    import numpy as np

    monkeypatch.delenv(_V2_ENV, raising=False)
    pkg, weights = _pkg(gate_fmt="iq4_xs", up_fmt="iq4_xs", down_fmt="iq4_xs")
    _fuse_native_block_moe(pkg, allow_dense=False)

    fc2_bank = _moe(pkg).inputs[4].const_value.numpy()
    expected = np.stack([weights[f"d{e}"] for e in range(E)], axis=0)
    assert fc2_bank.shape == expected.shape
    assert np.array_equal(fc2_bank, expected)  # byte-for-byte, no requantization


def test_builder_fusion_is_deterministic(monkeypatch, tmp_path) -> None:
    """Two independent fusions produce byte-identical external data on save."""
    monkeypatch.delenv(_V2_ENV, raising=False)

    def _save(sub: str) -> bytes:
        pkg, _ = _pkg(gate_fmt="iq4_xs", up_fmt="iq4_xs", down_fmt="iq4_xs")
        _fuse_native_block_moe(pkg, allow_dense=False)
        out = tmp_path / sub
        pkg.save(str(out), progress_bar=False, check_weights=False)
        return (out / "model.onnx.data").read_bytes()

    assert _save("a") == _save("b")


# --------------------------------------------------------------------------- #
# Final-graph-state honesty gate (backstop)                                   #
# --------------------------------------------------------------------------- #
def _graph_with_named_bqmatmul(weight_name: str) -> ModelPackage:
    """One ``pkg.nxrt::BlockQuantizedMatMul`` whose weight carries *weight_name*."""
    block_elements, block_bytes = _NATIVE_BLOCK_FORMATS["iq4_xs"]
    x = ir.Value(name="x", shape=ir.Shape(["T", H]), type=ir.TensorType(ir.DataType.FLOAT))
    weight = ir.Value(
        name=weight_name,
        shape=ir.Shape([H, (H + block_elements - 1) // block_elements, block_bytes]),
        type=ir.TensorType(ir.DataType.UINT8),
    )
    node = ir.node(
        "BlockQuantizedMatMul",
        inputs=[x, weight],
        attributes={"K": H, "N": H, "format": "iq4_xs", "block_layout_version": 1},
        domain="pkg.nxrt",
        num_outputs=1,
        name="expert_proj",
    )
    node.outputs[0].name = "y"
    node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    graph = ir.Graph([x], [node.outputs[0]], nodes=[node], name="storm")
    model = ir.Model(graph, ir_version=10, producer_name="test")
    model.opset_imports[""] = OPSET_VERSION
    model.opset_imports["pkg.nxrt"] = 1
    return ModelPackage({"model": model})


def test_gate_flags_leftover_routed_storm() -> None:
    """A routed ``.experts.`` BlockQuantizedMatMul that fusion left is a storm."""
    pkg = _graph_with_named_bqmatmul("model.layers.0.mlp.experts.2.down_proj.weight")
    assert len(_routed_dense_block_matmul_nodes(pkg["model"])) == 1
    with pytest.raises(SparseMoEExportError, match=r"remain as"):
        _assert_sparse_moe_graph(pkg, source="storm.gguf", allow_dense=False)


def test_gate_ignores_shared_expert() -> None:
    """A ``shared_expert`` projection is not routed and must not trip the gate."""
    pkg = _graph_with_named_bqmatmul("model.layers.0.mlp.shared_expert.down_proj.weight")
    assert _routed_dense_block_matmul_nodes(pkg["model"]) == []
    _assert_sparse_moe_graph(pkg, source="shared.gguf", allow_dense=False)  # no raise


def test_gate_opt_in_allows_leftover_storm() -> None:
    """Under the dense opt-in the backstop stays silent (fusion already warned)."""
    pkg = _graph_with_named_bqmatmul("model.layers.0.mlp.experts.0.gate_proj.weight")
    _assert_sparse_moe_graph(pkg, source="storm.gguf", allow_dense=True)  # no raise


# --------------------------------------------------------------------------- #
# End-to-end: synthetic native-block MoE GGUF -> build_from_gguf              #
# --------------------------------------------------------------------------- #
# These drive the *whole* builder pipeline (metadata -> config bridge -> module
# build -> weight application -> step 9b fusion + gate) on a synthetic sharded-
# style GGUF, not just the post-export helpers. GLM-5.2 (glm_moe_dsa) needs
# Batty's full MLA/DSA machinery to reach the MoE block; qwen3_moe / qwen2_moe
# exercise the identical routed dense-loop -> BlockQuantizedMatMul storm and the
# same step 9b authority point with a tiny, self-contained model, so they are
# the honest end-to-end proof for the fusion wiring. GLM-5.2's own layers are
# per-projection mixed-format, so end-to-end they can only typed-reject until the
# v2 runtime ships (covered by the mixed-format case below).

_E2E_HID = 64
_E2E_MOE_INTER = 32
_E2E_N_EXP = 4
_E2E_TOPK = 2
_E2E_HEADS = 4
_E2E_KV = 2
_E2E_VOCAB = 256

_E2E_NATIVE = {"iq4_xs": "IQ4_XS", "iq1_s": "IQ1_S"}


def _write_native_moe_gguf(
    path, *, down_fmt: str, arch: str = "qwen3_moe", shared: bool = False
) -> None:
    """Write a tiny single-layer native-block MoE GGUF (routed + optional shared).

    Routed ``fc1``/``fc3`` banks are ``iq1_s``; the routed ``fc2`` (down) bank is
    *down_fmt* (``iq1_s`` = uniform, ``iq4_xs`` = GLM-5.2-style per-projection
    mix). Attention projections are ``iq4_xs`` native blocks so they survive as
    ordinary BlockQuantizedMatMul and let the test prove *only* the routed storm
    collapsed.
    """
    import numpy as np
    from gguf import GGMLQuantizationType, GGUFWriter

    native_bytes = {
        "iq4_xs": (GGMLQuantizationType.IQ4_XS, 256, 136),
        "iq1_s": (GGMLQuantizationType.IQ1_S, 256, 50),
    }

    def add_native(name, shape, fmt):
        qtype, be, bb = native_bytes[fmt]
        *lead, k_in = shape
        nb = (k_in + be - 1) // be
        cols = nb * bb
        total = int(np.prod(lead)) * cols
        raw = np.arange(total, dtype=np.uint8).reshape(*lead, cols)
        w.add_tensor(name, raw, raw_dtype=qtype)

    def add_f32(name, shape):
        w.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    w = GGUFWriter(str(path), arch)
    w.add_context_length(512)
    w.add_embedding_length(_E2E_HID)
    w.add_feed_forward_length(_E2E_HID * 2)
    w.add_block_count(1)
    w.add_head_count(_E2E_HEADS)
    w.add_head_count_kv(_E2E_KV)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(_E2E_VOCAB)
    w.add_expert_count(_E2E_N_EXP)
    w.add_expert_used_count(_E2E_TOPK)
    w.add_expert_feed_forward_length(_E2E_MOE_INTER)
    if shared:
        w.add_expert_shared_feed_forward_length(_E2E_MOE_INTER)

    head_dim = _E2E_HID // _E2E_HEADS
    add_f32("token_embd.weight", (_E2E_VOCAB, _E2E_HID))
    add_native("blk.0.attn_q.weight", (_E2E_HEADS * head_dim, _E2E_HID), "iq4_xs")
    add_native("blk.0.attn_k.weight", (_E2E_KV * head_dim, _E2E_HID), "iq4_xs")
    add_native("blk.0.attn_v.weight", (_E2E_KV * head_dim, _E2E_HID), "iq4_xs")
    add_native("blk.0.attn_output.weight", (_E2E_HID, _E2E_HEADS * head_dim), "iq4_xs")
    add_f32("blk.0.attn_norm.weight", (_E2E_HID,))
    add_f32("blk.0.ffn_norm.weight", (_E2E_HID,))
    add_f32("blk.0.ffn_gate_inp.weight", (_E2E_N_EXP, _E2E_HID))
    add_native("blk.0.ffn_gate_exps.weight", (_E2E_N_EXP, _E2E_MOE_INTER, _E2E_HID), "iq1_s")
    add_native("blk.0.ffn_up_exps.weight", (_E2E_N_EXP, _E2E_MOE_INTER, _E2E_HID), "iq1_s")
    add_native("blk.0.ffn_down_exps.weight", (_E2E_N_EXP, _E2E_HID, _E2E_MOE_INTER), down_fmt)
    if shared:
        add_f32("blk.0.ffn_gate_inp_shexp.weight", (1, _E2E_HID))
        add_native("blk.0.ffn_gate_shexp.weight", (_E2E_MOE_INTER, _E2E_HID), "iq1_s")
        add_native("blk.0.ffn_up_shexp.weight", (_E2E_MOE_INTER, _E2E_HID), "iq1_s")
        add_native("blk.0.ffn_down_shexp.weight", (_E2E_HID, _E2E_MOE_INTER), "iq1_s")
    add_f32("output_norm.weight", (_E2E_HID,))
    add_f32("output.weight", (_E2E_VOCAB, _E2E_HID))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _e2e_ops(pkg: ModelPackage) -> tuple[int, int]:
    return _count(pkg, "BlockQuantizedMoE"), _count(pkg, "BlockQuantizedMatMul")


def test_e2e_uniform_native_moe_fuses_through_builder(monkeypatch, tmp_path) -> None:
    """A uniform routed native-block MoE GGUF builds into one sparse BQMoE.

    Structural proof the routed expert matmuls collapse: 4 experts * 3
    projections (12 BlockQuantizedMatMul) become one BlockQuantizedMoE, and the
    only BlockQuantizedMatMul left are the 4 attention projections -- i.e. no
    routed expert storm and only the selected experts execute at run time.
    """
    from mobius.integrations.gguf import build_from_gguf

    monkeypatch.delenv(_V2_ENV, raising=False)
    path = tmp_path / "uniform-moe.gguf"
    _write_native_moe_gguf(path, down_fmt="iq1_s")

    pkg = build_from_gguf(path, keep_quantized=True)
    moe, matmul = _e2e_ops(pkg)
    assert moe == 1
    assert matmul == 4  # attn q/k/v/o only; the 12 routed expert matmuls collapsed
    attrs = _moe(pkg).attributes
    assert "block_layout_version" not in attrs  # uniform -> v1
    assert attrs["format"].value == "iq1_s"


def test_e2e_mixed_format_typed_rejects_through_builder(monkeypatch, tmp_path) -> None:
    """A GLM-5.2-style per-projection mixed GGUF fails closed end-to-end.

    Without the v2 runtime the builder must raise ``SparseMoEExportError`` rather
    than emit an unrunnable v2 node -- this is exactly the honest path GLM-5.2
    UD-IQ1 takes until the per-projection runtime ships.
    """
    from mobius.integrations.gguf import build_from_gguf

    monkeypatch.delenv(_V2_ENV, raising=False)
    path = tmp_path / "mixed-moe.gguf"
    _write_native_moe_gguf(path, down_fmt="iq4_xs")

    with pytest.raises(SparseMoEExportError, match=r"block_layout_version=2"):
        build_from_gguf(path, keep_quantized=True)


def test_e2e_mixed_format_fuses_with_v2_runtime(monkeypatch, tmp_path) -> None:
    """Opting the v2 runtime in emits a single per-projection v2 BQMoE end-to-end."""
    from mobius.integrations.gguf import build_from_gguf

    monkeypatch.setenv(_V2_ENV, "1")
    path = tmp_path / "mixed-moe-v2.gguf"
    _write_native_moe_gguf(path, down_fmt="iq4_xs")

    pkg = build_from_gguf(path, keep_quantized=True)
    moe, matmul = _e2e_ops(pkg)
    assert moe == 1
    assert matmul == 4
    attrs = _moe(pkg).attributes
    assert attrs["block_layout_version"].value == 2
    assert attrs["fc1_format"].value == "iq1_s"
    assert attrs["fc2_format"].value == "iq4_xs"


def test_e2e_allow_dense_retains_storm_through_builder(monkeypatch, tmp_path) -> None:
    """``allow_dense_moe=True`` keeps the runnable per-expert dense fallback.

    It must warn+retain, never count as a fused success: the 12 routed expert
    matmuls stay, so no BlockQuantizedMoE is produced.
    """
    from mobius.integrations.gguf import build_from_gguf

    monkeypatch.delenv(_V2_ENV, raising=False)
    path = tmp_path / "mixed-moe-dense.gguf"
    _write_native_moe_gguf(path, down_fmt="iq4_xs")

    pkg = build_from_gguf(path, keep_quantized=True, allow_dense_moe=True)
    moe, matmul = _e2e_ops(pkg)
    assert moe == 0
    # 4 attention + 4 experts * 3 projections all remain as dense BlockQuantizedMatMul.
    assert matmul == 16


def test_e2e_shared_expert_survives_fusion_through_builder(monkeypatch, tmp_path) -> None:
    """A shared-expert MoE (qwen2_moe) keeps the shared expert dense after fusion.

    The always-active shared expert must stay semantically correct: its 3 native
    projections remain as BlockQuantizedMatMul while the routed storm collapses
    into one BlockQuantizedMoE.
    """
    from mobius.integrations.gguf import build_from_gguf

    monkeypatch.delenv(_V2_ENV, raising=False)
    path = tmp_path / "shared-moe.gguf"
    _write_native_moe_gguf(path, down_fmt="iq1_s", arch="qwen2_moe", shared=True)

    pkg = build_from_gguf(path, keep_quantized=True)
    moe, matmul = _e2e_ops(pkg)
    assert moe == 1
    # 4 attention + 3 shared-expert projections survive; routed 12 collapsed.
    assert matmul == 7
