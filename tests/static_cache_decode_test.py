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


def _carry_caches(outputs: dict[str, np.ndarray], num_layers: int) -> dict[str, np.ndarray]:
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
            "input_ids": rng.integers(0, vocab, size=(1, prefill_len), dtype=np.int64),
            "position_ids": np.arange(prefill_len, dtype=np.int64)[None, :],
            "write_indices": np.array([0], dtype=np.int64),
            "nonpad_kv_seqlen": np.array([prefill_len], dtype=np.int64),
        }
        prefill_feeds.update(_empty_caches(num_layers, kv_hidden))

        prefill_out = dict(zip(output_names, session.run(output_names, prefill_feeds)))
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

        decode_out = dict(zip(output_names, session.run(output_names, decode_feeds)))
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


def test_static_cache_decode_mask_bounds_attention_to_frontier_on_cuda():
    """Always-masked decode attends only to cache slots within the causal frontier.

    Option Y exports the static cache with ``is_causal=0`` plus an explicit
    :func:`mobius.components._common.create_static_cache_causal_mask`, which keeps
    key slot ``j`` for a query at absolute position ``p = write_indices[b] + t``
    iff ``j <= p``.  Because the op runs ``is_causal=0``, at a non-zero decode
    offset this explicit mask is the *only* thing bounding attention: it must zero
    out every slot beyond the frontier ``j <= write_indices + t`` — equivalently
    every padding slot ``j >= nonpad_kv_seqlen`` — so out-of-range cache garbage
    can never reach the softmax.

    This is the masked-semantics counterpart of the (dropped) phase-split
    maskless-nonpad guard: here the bound is enforced by the explicit mask, not by
    a maskless decode reading exactly ``nonpad`` keys.  It is checked two ways:

    * **Out-of-range (negative) control:** poisoning every slot at or beyond the
      frontier with large garbage must NOT change the decode logits — those keys
      are masked out, so the result is bit-identical to the clean-cache decode.
    * **In-range (positive) control:** poisoning a slot strictly inside the
      frontier MUST change the decode logits — proving the decode genuinely
      attends to in-range keys, so the negative control is meaningful rather than
      passing vacuously because decode ignores the cache.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session, config = _build_static_cache_session(tmp_dir)
        num_layers = config.num_hidden_layers
        kv_hidden = config.num_key_value_heads * config.head_dim
        vocab = config.vocab_size
        output_names = [out.name for out in session.get_outputs()]
        rng = np.random.default_rng(7)

        # Prefill four real tokens into slots 0..3 to populate the cache.
        prefill_len = 4
        prefill_feeds: dict[str, np.ndarray] = {
            "input_ids": rng.integers(0, vocab, size=(1, prefill_len), dtype=np.int64),
            "position_ids": np.arange(prefill_len, dtype=np.int64)[None, :],
            "write_indices": np.array([0], dtype=np.int64),
            "nonpad_kv_seqlen": np.array([prefill_len], dtype=np.int64),
        }
        prefill_feeds.update(_empty_caches(num_layers, kv_hidden))
        prefill_out = dict(zip(output_names, session.run(output_names, prefill_feeds)))

        # Decode one token into slot 4 at offset>0: write_indices=4, so the
        # causal frontier is j <= 4 (slots 0..4 valid; nonpad=5 marks the tail).
        nonpad = prefill_len + 1
        decode_inputs: dict[str, np.ndarray] = {
            "input_ids": rng.integers(0, vocab, size=(1, 1), dtype=np.int64),
            "position_ids": np.array([[prefill_len]], dtype=np.int64),
            "write_indices": np.array([prefill_len], dtype=np.int64),
            "nonpad_kv_seqlen": np.array([nonpad], dtype=np.int64),
        }

        clean_feeds = {**decode_inputs, **_carry_caches(prefill_out, num_layers)}
        baseline = dict(zip(output_names, session.run(output_names, clean_feeds)))

        # Negative control: poison every slot at/beyond the frontier (j >= nonpad).
        # The explicit mask zeroes these, so the decode logits must be unchanged.
        out_of_range = _carry_caches(prefill_out, num_layers)
        for layer in range(num_layers):
            for name in (f"key_cache.{layer}", f"value_cache.{layer}"):
                buf = out_of_range[name].copy()
                buf[:, nonpad:, :] = _CACHE_DTYPE(50.0)
                out_of_range[name] = buf
        out_of_range_feeds = {**decode_inputs, **out_of_range}
        masked_out = dict(zip(output_names, session.run(output_names, out_of_range_feeds)))

        assert np.array_equal(baseline["logits"], masked_out["logits"]), (
            "decode logits changed when cache slots beyond the causal frontier "
            "(j >= nonpad_kv_seqlen) were poisoned — the explicit static-cache "
            "mask is not bounding attention to j <= write_indices + t"
        )

        # Positive control: poison an in-range slot (0, strictly inside the
        # frontier). It is attended, so the decode logits MUST change — proving
        # the negative control above is a live guard, not a no-op.
        in_range = _carry_caches(prefill_out, num_layers)
        for layer in range(num_layers):
            for name in (f"key_cache.{layer}", f"value_cache.{layer}"):
                buf = in_range[name].copy()
                buf[:, 0, :] = _CACHE_DTYPE(50.0)
                in_range[name] = buf
        in_range_feeds = {**decode_inputs, **in_range}
        attended = dict(zip(output_names, session.run(output_names, in_range_feeds)))

    assert not np.array_equal(baseline["logits"], attended["logits"]), (
        "decode logits were unchanged when an in-frontier cache slot (0) was "
        "poisoned — decode is not attending to valid in-range keys, so the "
        "out-of-range guard would pass vacuously"
    )
