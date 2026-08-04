# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: GatedDeltaNet numerical parity against HuggingFace.

Verifies that the mobius GatedDeltaNet ONNX model produces the same output
logits and recurrent state as HuggingFace's Qwen3_5GatedDeltaNet when
processing a single token (decode-step mode).

Run with::

    pytest tests/parity/gated_deltanet_integration_test.py -m integration -sv
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from mobius import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._testing.comparison import assert_logits_close
from mobius._testing.ort_inference import OnnxModelSession
from mobius._weight_loading import apply_weights
from mobius.models.qwen35 import Qwen35CausalLMModel

# ---------------------------------------------------------------------------
# Tiny config — built locally, no HF Hub download required.
# Two layers: one DeltaNet (linear_attention) + one full attention.
# A mixed config is needed because HF's cache requires at least one
# attention layer to compute sequence length correctly.
# ---------------------------------------------------------------------------
_HF_CONFIG = Qwen3_5TextConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rms_norm_eps=1e-6,
    max_position_embeddings=128,
    rope_parameters={
        "rope_type": "default",
        "partial_rotary_factor": 0.5,
        "rope_theta": 10000.0,
    },
    layer_types=["linear_attention", "full_attention"],
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
)


def _build_cache_feeds(
    onnx_model: ir.Model,
    base_feeds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return zero-valued feeds for all cache inputs not already in base_feeds.

    For KV-cache inputs (``past_key_values.*.key`` / ``past_key_values.*.value``),
    the batch dim (index 0) is set to 1 and any remaining symbolic dims (i.e.
    the past-sequence dim) are set to 0 for an empty initial cache.
    For all other inputs (DeltaNet conv/recurrent states), any symbolic dim is
    mapped to 1 (batch dimension).
    """
    feeds = dict(base_feeds)
    for inp in onnx_model.graph.inputs:
        name = inp.name
        if name in feeds:
            continue
        is_kv_cache = name.endswith((".key", ".value"))
        # KV caches: batch (dim 0) → 1, past-sequence (non-0 symbolic) → 0.
        # DeltaNet states / others: all symbolic dims → 1 (batch only).
        shape = tuple(
            d if isinstance(d, int) else (0 if (is_kv_cache and i > 0) else 1)
            for i, d in enumerate(inp.shape)
        )
        feeds[name] = np.zeros(shape, dtype=np.float32)
    return feeds


@pytest.fixture(scope="class")
def parity_outputs():
    """Build models once and run a single-token decode step.

    Shared across all tests in ``TestGatedDeltaNetParity`` so that the
    expensive model construction + weight transfer happens only once per
    test-class invocation.

    Returns a dict with keys:
      - ``hf_logits``: ``np.ndarray`` — HF model output logits
      - ``hf_rec``:    ``np.ndarray`` — HF recurrent state for layer 0 (DeltaNet)
      - ``onnx_out``:  ``dict[str, np.ndarray]`` — full ONNX output dict
    """
    torch.manual_seed(42)
    hf_model = Qwen3_5ForCausalLM._from_config(_HF_CONFIG).float().eval()

    # Build ONNX model via the hybrid task so that CausalConv1DWithState and
    # LinearAttention are embedded as ONNX local functions in the model.
    # (Building a bare component graph omits these definitions and ORT fails.)
    arch_config = ArchitectureConfig.from_transformers(_HF_CONFIG)
    arch_config.dtype = ir.DataType.FLOAT
    onnx_module = Qwen35CausalLMModel(arch_config)
    pkg = build_from_module(onnx_module, arch_config, task="hybrid-text-generation")
    onnx_model = pkg["model"]

    # Transfer HF weights → ONNX via preprocess_weights.
    preprocessed = onnx_module.preprocess_weights(dict(hf_model.state_dict()))
    apply_weights(onnx_model, preprocessed)

    rng = np.random.default_rng(42)
    input_ids = rng.integers(1, _HF_CONFIG.vocab_size, size=(1, 1)).astype(np.int64)
    attention_mask = np.ones((1, 1), dtype=np.int64)
    position_ids = np.zeros((1, 1), dtype=np.int64)

    # HF forward — zero initial DeltaNet state; capture logits and recurrent state.
    with torch.no_grad():
        hf_out = hf_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
        )
    hf_logits = hf_out.logits.numpy()
    # transformers >=5.x stores recurrent state per cache layer
    # (``cache.layers[idx].recurrent_states``) instead of a top-level list.
    hf_rec = hf_out.past_key_values.layers[0].recurrent_states.numpy()

    # ONNX forward — build zero-valued cache feeds from graph inputs.
    base_feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    feeds = _build_cache_feeds(onnx_model, base_feeds)

    sess = OnnxModelSession(onnx_model)
    onnx_out = sess.run(feeds)
    sess.close()

    return {"hf_logits": hf_logits, "hf_rec": hf_rec, "onnx_out": onnx_out}


@pytest.mark.integration
@pytest.mark.integration_fast
class TestGatedDeltaNetParity:
    """Numerical parity: ONNX GatedDeltaNet vs HuggingFace Qwen3_5GatedDeltaNet."""

    def test_logits_match(self, parity_outputs):
        """Single-token decode: output logits match HF Qwen3_5ForCausalLM."""
        assert_logits_close(
            parity_outputs["onnx_out"]["logits"],
            parity_outputs["hf_logits"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_recurrent_state_matches(self, parity_outputs):
        """Single-token decode: DeltaNet recurrent state matches HF after one step."""
        np.testing.assert_allclose(
            parity_outputs["onnx_out"]["present.0.recurrent_state"],
            parity_outputs["hf_rec"],
            rtol=1e-3,
            atol=1e-3,
            err_msg="GatedDeltaNet recurrent_state mismatch vs HF Qwen3_5GatedDeltaNet",
        )
