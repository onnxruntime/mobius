# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end regression test for the static (TensorScatter) KV cache.

mobius used to emit the static-cache ``Attention`` op with ``is_causal=1``
alongside ``nonpad_kv_seqlen``.  The opset-24 ONNX ``Attention`` CUDA kernel
rejects that combination whenever the query length differs from the total KV
length and there is no ``past_key`` (the ``causal_cross_no_past`` guard in
``onnxruntime/core/providers/cuda/llm/attention.cc``).  Because the static
cache is pre-allocated to ``max_seq_len``, that guard fires in **both**
prefill (``S_q = N``) and decode (``S_q = 1``), raising ``NOT_IMPLEMENTED``.

The fix sets ``is_causal=0`` and supplies an explicit causal mask
(:func:`mobius.components._common.create_static_cache_causal_mask`).  This test
exercises the actual ONNX Runtime kernel for both phases so the regression
cannot silently come back.  It requires the CUDA Execution Provider because
``TensorScatter`` and the external-cache ``Attention`` path are CUDA-only.

The model is built fp32: the ``is_causal`` guard fires in ORT *before* kernel
dtype dispatch, so fp32 exercises the same external-cache code path as a
production fp16 export while avoiding the cos/sin-cache dtype casting that
only the full CLI build pipeline applies.  A raw ``InferenceSession`` is used
(instead of the ``OnnxModelSession`` test helper) to keep this regression
guard free of the optional ``onnxruntime-easy`` dependency.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from _test_configs import _base_config

from mobius._registry import registry
from mobius.tasks import CausalLMTask

pytestmark = pytest.mark.skipif(
    "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="static-cache TensorScatter / external-cache Attention are CUDA-only",
)

_MAX_SEQ_LEN = 16
_MODEL_TYPE = "qwen2"
_CACHE_DTYPE = np.float32


def _fill_random_weights(model: ir.Model, rng: np.random.Generator) -> None:
    """Fill empty initializers with small random values of their dtype.

    The graph is built without real weights; ORT still needs concrete
    initializers to run.  Small values keep logits finite and well-scaled.
    """
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = initializer.shape
        dims = [d if isinstance(d, int) else 1 for d in shape] if shape else [1]
        dtype = initializer.dtype or ir.DataType.FLOAT
        np_dtype = dtype.numpy()
        if np.issubdtype(np_dtype, np.floating):
            data = (rng.standard_normal(dims) * 0.02).astype(np_dtype)
        else:
            data = np.zeros(dims, dtype=np_dtype)
        initializer.const_value = ir.Tensor(data)


def _build_static_cache_session(
    tmp_dir: str,
) -> tuple[ort.InferenceSession, object]:
    """Build a tiny static-cache qwen2 graph and load it on CUDA."""
    config = _base_config()
    module = registry.get(_MODEL_TYPE)(config)
    task = CausalLMTask(static_cache=True, max_seq_len=_MAX_SEQ_LEN)
    model = task.build(module, config)["model"]
    _fill_random_weights(model, np.random.default_rng(0))

    model_path = str(Path(tmp_dir) / "model.onnx")
    ir.save(model, model_path, external_data="model.onnx.data")
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert "CUDAExecutionProvider" in session.get_providers(), (
        "static-cache regression test must run on CUDA"
    )
    return session, config


def _empty_caches(num_layers: int, kv_hidden: int) -> dict[str, np.ndarray]:
    """Zeroed ``[1, max_seq, kv_hidden]`` cache buffers for every layer."""
    feeds: dict[str, np.ndarray] = {}
    for layer in range(num_layers):
        zeros = np.zeros((1, _MAX_SEQ_LEN, kv_hidden), dtype=_CACHE_DTYPE)
        feeds[f"key_cache.{layer}"] = zeros.copy()
        feeds[f"value_cache.{layer}"] = zeros.copy()
    return feeds


def _carry_caches(
    outputs: dict[str, np.ndarray], num_layers: int
) -> dict[str, np.ndarray]:
    """Feed the prefill ``updated_*`` caches back in as decode inputs."""
    feeds: dict[str, np.ndarray] = {}
    for layer in range(num_layers):
        feeds[f"key_cache.{layer}"] = outputs[f"updated_key_cache.{layer}"]
        feeds[f"value_cache.{layer}"] = outputs[f"updated_value_cache.{layer}"]
    return feeds


def test_static_cache_prefill_and_decode_run_on_cuda():
    """Prefill (S_q>1) and decode (S_q=1) both run without NOT_IMPLEMENTED.

    This is the regression guard the codebase previously lacked: it loads
    a real static-cache graph and runs it on ``CUDAExecutionProvider`` for
    both phases.  Reverting to ``is_causal=1`` (no mask) makes ORT raise
    ``NOT_IMPLEMENTED`` here, failing the test.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session, config = _build_static_cache_session(tmp_dir)
        num_layers = config.num_hidden_layers
        kv_hidden = config.num_key_value_heads * config.head_dim
        vocab = config.vocab_size
        output_names = [out.name for out in session.get_outputs()]
        rng = np.random.default_rng(1)

        # --- Prefill: write N tokens from slot 0 (S_q = N != max_seq). ---
        prefill_len = 4
        prefill_feeds: dict[str, np.ndarray] = {
            "input_ids": rng.integers(
                0, vocab, size=(1, prefill_len), dtype=np.int64
            ),
            "position_ids": np.arange(prefill_len, dtype=np.int64)[None, :],
            "write_indices": np.array([0], dtype=np.int64),
            "nonpad_kv_seqlen": np.array([prefill_len], dtype=np.int64),
        }
        prefill_feeds.update(_empty_caches(num_layers, kv_hidden))

        prefill_out = dict(
            zip(output_names, session.run(output_names, prefill_feeds))
        )
        prefill_logits = prefill_out["logits"]
        assert prefill_logits.shape == (1, prefill_len, vocab)
        assert np.isfinite(prefill_logits).all(), "prefill logits must be finite"

        # --- Decode: one token at slot N (S_q = 1 != max_seq). ---
        decode_feeds: dict[str, np.ndarray] = {
            "input_ids": rng.integers(0, vocab, size=(1, 1), dtype=np.int64),
            "position_ids": np.array([[prefill_len]], dtype=np.int64),
            "write_indices": np.array([prefill_len], dtype=np.int64),
            "nonpad_kv_seqlen": np.array([prefill_len + 1], dtype=np.int64),
        }
        decode_feeds.update(_carry_caches(prefill_out, num_layers))

        decode_out = dict(
            zip(output_names, session.run(output_names, decode_feeds))
        )
        decode_logits = decode_out["logits"]
        assert decode_logits.shape == (1, 1, vocab)
        assert np.isfinite(decode_logits).all(), "decode logits must be finite"

        # The decode step must have scattered its key into slot N (the
        # previously-empty tail), confirming the in-place cache advanced.
        advanced_key = decode_out["updated_key_cache.0"]
        assert advanced_key.shape == (1, _MAX_SEQ_LEN, kv_hidden)
        assert np.any(advanced_key[0, prefill_len] != 0), (
            "decode should scatter the new key into cache slot N"
        )
