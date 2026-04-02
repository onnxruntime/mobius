# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius.models.base import CausalLMModel
from mobius.rewrite_rules._testing_utils import count_ops
from mobius.rewrite_rules._unpack_qkv import unpack_qkv_rules

# Tiny config with standard GQA (no QK norm so PackQKV fires)
_CONFIG = ArchitectureConfig(
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_hidden_layers=1,
    vocab_size=256,
    max_position_embeddings=128,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    rope_type="default",
    rope_theta=10000.0,
    pad_token_id=0,
)


def _build_packed_model():
    """Build model with GQA fusion + QKV packing applied."""
    mod = CausalLMModel(_CONFIG)
    pkg = build_from_module(mod, _CONFIG)
    model = pkg["model"]
    # GQA fusion (with PackQKV) runs as part of default CPU build
    return model


class TestUnpackQKVRules:
    def test_returns_rule_set(self):
        rules = unpack_qkv_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_unpacks_packed_qkv(self):
        """Packed GQA (k=None, v=None) is split into 3 separate MatMuls."""
        model = _build_packed_model()

        # Verify the model has packed GQA before unpacking
        gqa_nodes = [n for n in model.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == 1
        gqa_node = gqa_nodes[0]

        # If the GQA was packed (k=None, v=None), apply unpack
        if gqa_node.inputs[1] is None and gqa_node.inputs[2] is None:
            counts_before = count_ops(model)
            matmul_before = counts_before.get("MatMul", 0)

            rewrite(model, pattern_rewrite_rules=unpack_qkv_rules())

            counts_after = count_ops(model)
            # Should still have one GQA node
            assert counts_after.get("GroupQueryAttention", 0) == 1
            # GQA should now have separate Q/K/V (inputs[1] and inputs[2] not None)
            gqa_after = next(n for n in model.graph if n.op_type == "GroupQueryAttention")
            assert gqa_after.inputs[1] is not None, "K should be separate after unpack"
            assert gqa_after.inputs[2] is not None, "V should be separate after unpack"
            # Should have 2 additional MatMul nodes (was 1 packed → 3 separate)
            matmul_after = counts_after.get("MatMul", 0)
            assert matmul_after == matmul_before + 2
        else:
            # PackQKV didn't fire (e.g., QK norm present), skip this test
            pass

    def test_unpack_preserves_num_matmul_outputs(self):
        """After unpack, Q/K/V projections output the correct feature sizes."""
        model = _build_packed_model()
        gqa_nodes = [n for n in model.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == 1
        gqa_node = gqa_nodes[0]

        if gqa_node.inputs[1] is not None:
            # Not packed — skip
            return

        rewrite(model, pattern_rewrite_rules=unpack_qkv_rules())

        gqa_after = next(n for n in model.graph if n.op_type == "GroupQueryAttention")
        q_in = gqa_after.inputs[0]
        k_in = gqa_after.inputs[1]
        v_in = gqa_after.inputs[2]

        # q, k, v should be produced by MatMul nodes
        assert q_in is not None and q_in.producer().op_type == "MatMul"
        assert k_in is not None and k_in.producer().op_type == "MatMul"
        assert v_in is not None and v_in.producer().op_type == "MatMul"

    def test_no_match_when_already_separate(self):
        """UnpackQKV does not match GQA with separate Q/K/V inputs."""
        model = _build_packed_model()
        gqa_nodes = [n for n in model.graph if n.op_type == "GroupQueryAttention"]
        gqa_node = gqa_nodes[0]

        # If already separate, unpack should be a no-op
        if gqa_node.inputs[1] is not None:
            counts_before = dict(count_ops(model))
            rewrite(model, pattern_rewrite_rules=unpack_qkv_rules())
            assert dict(count_ops(model)) == counts_before
