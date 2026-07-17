# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for :class:`DecomposeAttentionPass`.

The pass rewrites the fused opset-24 ``Attention`` op into scaled-dot-product
primitives for EPs without an ``Attention`` kernel (QNN HTP). The critical
invariants:

1. **Numerical parity** — the decomposed graph must produce identical output to
   the fused op for prefill / decode (with past KV) / GQA / softcap / mask.
2. **Output rewiring** — the ``Attention`` op has three outputs
   (Y, present_key, present_value); the pass must preserve those that are graph
   outputs (cache-producing layers) and drop those with no consumers (shared-KV
   layers). This mixed arity is why it is an IR pass, not a pattern rule.
3. **EP gating** — a ``qnn`` build decomposes; a ``cpu`` build keeps the fused op.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius.rewrite_rules import decompose_attention_pass


def _run(model: ir.Model, feeds: dict) -> list[np.ndarray]:
    path = os.path.join(tempfile.mkdtemp(), "a.onnx")
    ir.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def _build_attention_model(
    batch: int,
    seq_q: int,
    seq_past: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    softcap: float = 0.0,
    is_causal: int = 1,
    with_mask: bool = True,
    with_past: bool = True,
) -> ir.Model:
    """Single fused Attention node with rank-3 Q/K/V and optional mask/past."""
    q_hidden = q_heads * head_dim
    kv_hidden = kv_heads * head_dim
    q = ir.Value(
        name="q",
        shape=ir.Shape([batch, seq_q, q_hidden]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    k = ir.Value(
        name="k",
        shape=ir.Shape([batch, seq_q, kv_hidden]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    v = ir.Value(
        name="v",
        shape=ir.Shape([batch, seq_q, kv_hidden]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    inputs = [q, k, v]
    graph_inputs = [q, k, v]
    if with_mask:
        mask = ir.Value(
            name="mask",
            shape=ir.Shape([batch, 1, seq_q, seq_past + seq_q]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        inputs.append(mask)
        graph_inputs.append(mask)
    else:
        inputs.append(None)
    if with_past:
        pk = ir.Value(
            name="pk",
            shape=ir.Shape([batch, kv_heads, seq_past, head_dim]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        pv = ir.Value(
            name="pv",
            shape=ir.Shape([batch, kv_heads, seq_past, head_dim]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        inputs.extend([pk, pv])
        graph_inputs.extend([pk, pv])

    y = ir.Value(name="y")
    present_k = ir.Value(name="present_k")
    present_v = ir.Value(name="present_v")
    node = ir.Node(
        "",
        "Attention",
        inputs=inputs,
        outputs=[y, present_k, present_v],
        attributes=ir.convenience.convert_attributes(
            {
                "q_num_heads": q_heads,
                "kv_num_heads": kv_heads,
                "scale": 1.0,
                "softcap": softcap,
                "is_causal": is_causal,
            }
        ),
    )
    graph = ir.Graph(
        inputs=graph_inputs,
        outputs=[y, present_k, present_v],
        nodes=[node],
        initializers=[],
        opset_imports={"": 24},
        name="attn",
    )
    return ir.Model(graph, ir_version=10)


def _feeds(
    batch,
    seq_q,
    seq_past,
    q_heads,
    kv_heads,
    head_dim,
    *,
    with_mask=True,
    with_past=True,
    seed=0,
):
    rng = np.random.default_rng(seed)
    f = {
        "q": rng.standard_normal((batch, seq_q, q_heads * head_dim)).astype(np.float32),
        "k": rng.standard_normal((batch, seq_q, kv_heads * head_dim)).astype(np.float32),
        "v": rng.standard_normal((batch, seq_q, kv_heads * head_dim)).astype(np.float32),
    }
    if with_mask:
        f["mask"] = (rng.standard_normal((batch, 1, seq_q, seq_past + seq_q)) * 0.1).astype(
            np.float32
        )
    if with_past:
        f["pk"] = rng.standard_normal((batch, kv_heads, seq_past, head_dim)).astype(np.float32)
        f["pv"] = rng.standard_normal((batch, kv_heads, seq_past, head_dim)).astype(np.float32)
    return f


# (seq_q, seq_past, q_heads, kv_heads, softcap, is_causal, with_mask, with_past, atol, tag)
_CASES = [
    (4, 0, 8, 8, 0.0, 1, True, False, 0.0, "prefill_mha"),
    (4, 0, 8, 2, 0.0, 1, True, False, 0.0, "prefill_gqa"),
    (1, 3, 8, 1, 0.0, 1, True, True, 0.0, "decode_gqa_past"),
    (4, 0, 8, 2, 30.0, 1, True, False, 1e-5, "softcap"),
    (4, 0, 8, 2, 0.0, 0, True, False, 0.0, "mask_only_noncausal"),
    (1, 5, 8, 1, 30.0, 1, True, True, 1e-5, "decode_softcap_past"),
]


class TestDecomposeAttentionParity:
    @pytest.mark.parametrize(
        "seq_q,seq_past,q_heads,kv_heads,softcap,is_causal,with_mask,with_past,atol,tag",
        _CASES,
        ids=[c[-1] for c in _CASES],
    )
    def test_matches_fused_op(
        self,
        seq_q,
        seq_past,
        q_heads,
        kv_heads,
        softcap,
        is_causal,
        with_mask,
        with_past,
        atol,
        tag,
    ):
        batch, head_dim = 1, 16
        feeds = _feeds(
            batch,
            seq_q,
            seq_past,
            q_heads,
            kv_heads,
            head_dim,
            with_mask=with_mask,
            with_past=with_past,
        )

        ref_model = _build_attention_model(
            batch,
            seq_q,
            seq_past,
            q_heads,
            kv_heads,
            head_dim,
            softcap=softcap,
            is_causal=is_causal,
            with_mask=with_mask,
            with_past=with_past,
        )
        reference = _run(ref_model, feeds)

        dec_model = _build_attention_model(
            batch,
            seq_q,
            seq_past,
            q_heads,
            kv_heads,
            head_dim,
            softcap=softcap,
            is_causal=is_causal,
            with_mask=with_mask,
            with_past=with_past,
        )
        decompose_attention_pass()(dec_model)
        assert Counter(n.op_type for n in dec_model.graph).get("Attention", 0) == 0
        got = _run(dec_model, feeds)

        # Y and present_key/present_value all match.
        for ref, out in zip(reference, got):
            np.testing.assert_allclose(out, ref, rtol=0, atol=atol)


class TestDecomposeAttentionRewiring:
    def test_preserves_kv_cache_graph_outputs(self):
        """present_key/value that are graph outputs must be rewired, not dropped."""
        batch, seq_q, seq_past, q_heads, kv_heads, head_dim = 1, 1, 3, 8, 1, 16
        model = _build_attention_model(batch, seq_q, seq_past, q_heads, kv_heads, head_dim)
        decompose_attention_pass()(model)
        out_names = {o.name for o in model.graph.outputs}
        assert {"present_k", "present_v", "y"} <= out_names
        feeds = _feeds(batch, seq_q, seq_past, q_heads, kv_heads, head_dim)
        outs = _run(model, feeds)
        # present_key has shape (batch, kv_heads, seq_past + seq_q, head_dim)
        assert outs[1].shape == (batch, kv_heads, seq_past + seq_q, head_dim)

    def test_leaves_rank4_attention_untouched(self):
        """Non-rank-3 Q/K/V is not decomposable and must be left as-is."""
        q = ir.Value(
            name="q", shape=ir.Shape([1, 4, 8, 16]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        k = ir.Value(
            name="k", shape=ir.Shape([1, 4, 8, 16]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        v = ir.Value(
            name="v", shape=ir.Shape([1, 4, 8, 16]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        y = ir.Value(name="y")
        node = ir.Node(
            "",
            "Attention",
            inputs=[q, k, v],
            outputs=[y],
            attributes=ir.convenience.convert_attributes(
                {"q_num_heads": 4, "kv_num_heads": 4}
            ),
        )
        graph = ir.Graph(
            inputs=[q, k, v],
            outputs=[y],
            nodes=[node],
            initializers=[],
            opset_imports={"": 24},
            name="r4",
        )
        model = ir.Model(graph, ir_version=10)
        result = decompose_attention_pass()(model)
        assert not result.modified
        assert Counter(n.op_type for n in model.graph).get("Attention", 0) == 1


class TestDecomposeAttentionEpGating:
    def _build(self, ep: str, dtype: ir.DataType) -> Counter:
        import dataclasses

        from mobius._builder import build_from_module
        from mobius._config_resolver import _default_task_for_model
        from mobius._configs import Gemma4Config, QuantizationConfig
        from mobius._registry import registry

        cfg = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=1_000_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=30.0,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            tie_word_embeddings=True,
            max_position_embeddings=64,
            quantization=QuantizationConfig(
                bits=4, group_size=32, quant_method="gguf", sym=False
            ),
        )
        cfg = dataclasses.replace(cfg, dtype=dtype)
        module = registry.get("gemma4_text")(cfg)
        model = build_from_module(
            module, cfg, task=_default_task_for_model("gemma4_text"), execution_provider=ep
        )["model"]
        return Counter(n.op_type for n in model.graph)

    def test_cpu_keeps_fused_attention(self):
        # CPU EP fuses GQA only for float32 (its gqa_dtypes = {FLOAT}); the fused
        # attention op is kept, never decomposed to Softmax-based SDPA.
        ops = self._build("cpu", ir.DataType.FLOAT)
        assert ops.get("GroupQueryAttention", 0) > 0
        assert ops.get("Softmax", 0) == 0

    def test_qnn_decomposes_attention(self):
        ops = self._build("qnn", ir.DataType.FLOAT16)
        # No fused Attention nor GQA; decomposed to Softmax-based SDPA.
        assert ops.get("Attention", 0) == 0
        assert ops.get("GroupQueryAttention", 0) == 0
        assert ops.get("Softmax", 0) > 0
