# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Paged / block-table KV cache tests (onnx-genai DESIGN §39.4 Option C).

These tests guard the paged-attention KV cache layout mobius emits with
``CausalLMTask(paged_cache=True)``.  KV lives in a shared *page pool* of
fixed-size pages that are non-contiguous per sequence; a per-sequence
``block_table`` maps logical page slots to physical pages, and a
``slot_mapping`` gives the flat physical slot for each newly written token.
This is the vLLM PagedAttention layout, and — because sequences can list the
*same* physical page in their ``block_table`` — it also expresses SGLang
RadixAttention (shared prefix pages) with no graph change.

The in-graph paging is built from standard ONNX ops
(``Reshape``/``Shape``/``Unsqueeze``/``ScatterND``/``Gather``/``Attention``);
see :class:`mobius.components._attention.PagedCacheState`.

Two levels of coverage live here:

* :class:`TestPagedPagingOpsCpuParity` — runs the paging ops
  (``ScatterND`` write + ``Gather`` page assembly) on the **CPU** EP and
  checks them against a NumPy reference, including a RadixAttention
  shared-page case.  These ops are EP-agnostic, so this runs anywhere.

* The full-model ``Attention`` op with ``nonpad_kv_seqlen`` (Attention input
  #6) is CUDA-only until onnxruntime#28958 ships (same constraint as the
  static cache), so end-to-end paged *execution* parity is intentionally not
  asserted here — the graph-construction guarantees are covered by
  ``build_graph_test.py::TestBuildPagedCacheGraph``.

Run::

    pytest tests/paged_cache_test.py -v
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION


def _build_paging_probe_graph(
    *,
    num_pages: int,
    page_size: int,
    kv_hidden: int,
    seq_len: int,
    num_blocks: int,
) -> bytes:
    """Build a standalone graph mirroring the paged write+assemble sub-graph.

    Replicates :func:`mobius.components._attention._apply_paged_attention` up
    to (but excluding) the ``Attention`` op: ``ScatterND`` the new K rows into
    the flattened pool at ``slot_mapping``, restore the pool shape, then
    ``Gather`` the sequence's physical pages via ``block_table`` and flatten to
    a contiguous ``[1, num_blocks * page_size, kv_hidden]`` KV sequence.

    Inputs:  ``key_pool`` ``[num_pages, page_size, kv_hidden]``, ``key``
             ``[1, seq_len, kv_hidden]``, ``block_table`` ``[num_blocks]`` and
             ``slot_mapping`` ``[seq_len]`` int64.
    Outputs: ``updated_pool`` (same shape as ``key_pool``) and ``gathered``
             ``[1, num_blocks * page_size, kv_hidden]``.
    """

    def _value(name: str, dims: list[int], dt: ir.DataType) -> ir.Value:
        return ir.Value(name=name, shape=ir.Shape(dims), type=ir.TensorType(dt))

    key_pool = _value("key_pool", [num_pages, page_size, kv_hidden], ir.DataType.FLOAT)
    key = _value("key", [1, seq_len, kv_hidden], ir.DataType.FLOAT)
    block_table = _value("block_table", [num_blocks], ir.DataType.INT64)
    slot_mapping = _value("slot_mapping", [seq_len], ir.DataType.INT64)

    graph = ir.Graph(
        inputs=[key_pool, key, block_table, slot_mapping],
        outputs=[],
        nodes=[],
        name="paged_paging_probe",
        opset_imports={"": OPSET_VERSION},
    )
    op = GraphBuilder(graph).op

    pool_shape = op.Shape(key_pool)
    pool_flat = op.Reshape(key_pool, [-1, kv_hidden])
    key_rows = op.Reshape(key, [-1, kv_hidden])
    slot_idx = op.Unsqueeze(slot_mapping, [-1])
    updated_flat = op.ScatterND(pool_flat, slot_idx, key_rows)
    updated_pool = op.Reshape(updated_flat, pool_shape)
    gathered = op.Gather(updated_pool, block_table, axis=0)
    gathered = op.Reshape(gathered, [1, -1, kv_hidden])

    updated_pool.name = "updated_pool"
    gathered.name = "gathered"
    graph.outputs.extend([updated_pool, gathered])

    model = ir.Model(graph, ir_version=10)
    return ir.serde.serialize_model(model).SerializeToString()


class TestPagedPagingOpsCpuParity:
    """The paged write+assemble sub-graph matches a NumPy reference on CPU."""

    def _run(self, proto: bytes, feeds: dict[str, np.ndarray]):
        sess = ort.InferenceSession(proto, providers=["CPUExecutionProvider"])
        return sess.run(None, feeds)

    def test_scatter_then_gather_matches_numpy(self):
        """Write new tokens to physical slots, then gather contiguous pages."""
        num_pages, page_size, kv_hidden = 6, 4, 8
        seq_len, num_blocks = 3, 2
        proto = _build_paging_probe_graph(
            num_pages=num_pages,
            page_size=page_size,
            kv_hidden=kv_hidden,
            seq_len=seq_len,
            num_blocks=num_blocks,
        )

        rng = np.random.default_rng(0)
        pool0 = rng.standard_normal((num_pages, page_size, kv_hidden)).astype(np.float32)
        new_k = rng.standard_normal((1, seq_len, kv_hidden)).astype(np.float32)
        # Sequence uses physical pages 4 then 1; write 3 tokens into page 4.
        block = np.array([4, 1], dtype=np.int64)
        slots = np.array(
            [4 * page_size + 0, 4 * page_size + 1, 4 * page_size + 2], dtype=np.int64
        )

        out_pool, out_gathered = self._run(
            proto,
            {"key_pool": pool0, "key": new_k, "block_table": block, "slot_mapping": slots},
        )

        ref_flat = pool0.copy().reshape(-1, kv_hidden)
        ref_flat[slots] = new_k.reshape(-1, kv_hidden)
        ref_pool = ref_flat.reshape(num_pages, page_size, kv_hidden)
        ref_gathered = ref_pool[block].reshape(1, -1, kv_hidden)

        np.testing.assert_allclose(out_pool, ref_pool, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out_gathered, ref_gathered, rtol=1e-6, atol=1e-6)
        # The written tokens are visible at the front of the first gathered page.
        np.testing.assert_allclose(out_gathered[0, 0:seq_len], new_k[0], rtol=1e-6, atol=1e-6)
        assert out_gathered.shape == (1, num_blocks * page_size, kv_hidden)

    def test_radix_shared_page_is_read_by_two_block_tables(self):
        """A physical page shared by two sequences' block_tables reads back once.

        This is the SGLang RadixAttention property: two sequences that share a
        common prefix point their ``block_table`` at the *same* physical page,
        and both gather identical KV for that page — no duplication, no graph
        change.
        """
        num_pages, page_size, kv_hidden = 5, 2, 4
        proto = _build_paging_probe_graph(
            num_pages=num_pages,
            page_size=page_size,
            kv_hidden=kv_hidden,
            seq_len=2,
            num_blocks=2,
        )
        rng = np.random.default_rng(7)
        pool0 = rng.standard_normal((num_pages, page_size, kv_hidden)).astype(np.float32)
        # Write a fresh "prefix" page into physical page 3.
        prefix_kv = rng.standard_normal((1, 2, kv_hidden)).astype(np.float32)
        slots = np.array([3 * page_size + 0, 3 * page_size + 1], dtype=np.int64)

        # Sequence A: [shared page 3, own page 0]; Sequence B: [shared page 3, own page 1].
        block_a = np.array([3, 0], dtype=np.int64)
        block_b = np.array([3, 1], dtype=np.int64)

        pool_a, gathered_a = self._run(
            proto,
            {
                "key_pool": pool0,
                "key": prefix_kv,
                "block_table": block_a,
                "slot_mapping": slots,
            },
        )
        # Sequence B reads the already-written shared page (no new write needed);
        # feed a no-op write that rewrites the same slots with the same values.
        _, gathered_b = self._run(
            proto,
            {
                "key_pool": pool_a,
                "key": prefix_kv,
                "block_table": block_b,
                "slot_mapping": slots,
            },
        )

        # Both sequences see identical KV for the shared prefix page (first page).
        np.testing.assert_allclose(
            gathered_a[0, 0:page_size], gathered_b[0, 0:page_size], rtol=1e-6, atol=1e-6
        )
        # And that shared KV equals what was written.
        np.testing.assert_allclose(
            gathered_a[0, 0:page_size], prefix_kv[0], rtol=1e-6, atol=1e-6
        )


def test_paged_cache_task_builds_valid_onnx():
    """End-to-end: CausalLMTask(paged_cache=True) builds a checker-valid graph."""
    from _test_configs import _base_config
    from onnx_ir.passes.common import CheckerPass

    from mobius._optimizations import SymbolicShapeInferencePass
    from mobius._registry import registry
    from mobius.tasks import CausalLMTask

    config = _base_config()
    module = registry.get("qwen2")(config)
    task = CausalLMTask(paged_cache=True, page_size=8, num_pages=16)
    pkg = task.build(module, config)
    model = pkg["model"]

    # Fill dummy weights so the checker can serialize.
    for init in model.graph.initializers.values():
        if init.const_value is None:
            dims = [d if isinstance(d, int) else 1 for d in (init.shape or [1])]
            dtype = init.dtype or ir.DataType.FLOAT
            init.const_value = ir.Tensor(np.zeros(dims, dtype=dtype.numpy()))

    CheckerPass(True)(model)
    SymbolicShapeInferencePass()(model)

    kv_hidden = config.num_key_value_heads * config.head_dim
    out_shapes = {o.name: list(o.shape) for o in model.graph.outputs}
    for i in range(config.num_hidden_layers):
        assert out_shapes[f"updated_key_pool.{i}"] == [16, 8, kv_hidden]
        assert out_shapes[f"updated_value_pool.{i}"] == [16, 8, kv_hidden]


def test_paged_validation_message_names_paged_mode():
    """An unsupported decoder layer reports the *paged* mode in the error."""
    import pytest
    from onnxscript import nn

    from mobius.tasks._causal_lm import _validate_static_cache_support

    class _BadLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()

    class _BadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_BadLayer()])

    with pytest.raises(TypeError, match="Paged cache mode"):
        _validate_static_cache_support(_BadModel(), mode="Paged cache")


def test_paged_cache_rejects_attention_bias():
    """Paged attention has no additive-bias path, so it must fail fast."""
    import pytest
    from _test_configs import _base_config

    from mobius._registry import registry
    from mobius.components._attention import PagedCacheState
    from mobius.tasks._base import _make_graph

    config = _base_config()
    module = registry.get("qwen2")(config)
    attn = module.model.layers[0].self_attn

    _graph, builder = _make_graph()
    op = builder.op
    hidden = builder.input("hidden", dtype=config.dtype, shape=[1, 1, config.hidden_size])
    bt = builder.input("block_table", dtype=ir.DataType.INT64, shape=[1, 4])
    sm = builder.input("slot_mapping", dtype=ir.DataType.INT64, shape=[1])
    kp = builder.input("kpool", dtype=config.dtype, shape=[16, 8, config.head_dim])
    vp = builder.input("vpool", dtype=config.dtype, shape=[16, 8, config.head_dim])
    seqlen = builder.input("nonpad_kv_seqlen", dtype=ir.DataType.INT64, shape=[1])
    paged = PagedCacheState(
        key_pool=kp,
        value_pool=vp,
        block_table=bt,
        slot_mapping=sm,
        nonpad_kv_seqlen=seqlen,
    )

    with pytest.raises(ValueError, match="attention_bias"):
        attn(
            op,
            hidden,
            attention_bias=op.Identity(hidden),
            paged_cache=paged,
        )
