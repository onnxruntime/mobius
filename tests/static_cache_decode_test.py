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

The fix sets ``is_causal=0`` and phase-splits the attention behind an ``If``
keyed on ``Shape(query)[1] > 1``: the multi-token (prefill) branch supplies an
explicit causal mask (:func:`mobius.components._common.create_static_cache_causal_mask`,
memory-efficient path), while the single-token decode branch omits the mask so
ORT keeps it on Flash/XQA — the same kernel the GQA variant uses, so the
profiling comparison stays apples-to-apples.  This test exercises the actual
ONNX Runtime kernel for both phases so the regression cannot silently come
back.  It requires the CUDA Execution Provider because ``TensorScatter`` and
the external-cache ``Attention`` path are CUDA-only.

The runnability test (:func:`test_static_cache_prefill_and_decode_run_on_cuda`)
is built fp32: the ``is_causal`` guard fires in ORT *before* kernel dtype
dispatch, so fp32 exercises the same external-cache code path as a production
fp16 export while avoiding the cos/sin-cache dtype casting that only the full
CLI build pipeline applies.  The Flash-eligibility guard
(:func:`test_static_cache_decode_runs_maskless_on_cuda`) builds fp16 — the
production precision — and asserts the decode ``If`` branch executes without an
``attn_mask`` input (the structural precondition for ORT to keep decode on
Flash).  A raw ``InferenceSession`` is used (instead of the
``OnnxModelSession`` test helper) to keep this regression guard free of the
optional ``onnxruntime-easy`` dependency.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from _test_configs import _base_config

from mobius._builder import build_from_module
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
    *,
    ir_dtype: ir.DataType = ir.DataType.FLOAT,
    enable_profiling: bool = False,
) -> tuple[ort.InferenceSession, object]:
    """Build a tiny static-cache qwen2 graph and load it on CUDA.

    Uses the full ``build_from_module`` export path (not bare ``task.build``)
    so the phase-split ``If`` subgraphs are exercised through the real
    ``optimize_model`` pipeline — that is where a structural regression in
    the static-cache attention would surface.

    Args:
        tmp_dir: Directory for the saved ONNX model + external data.
        ir_dtype: Weight/activation precision.  Defaults to fp32 (the
            ``is_causal`` guard fires before kernel dtype dispatch, so fp32
            exercises the same external-cache path while avoiding cos/sin
            cache casting).  Pass ``ir.DataType.FLOAT16`` to build the
            production-precision graph used for the Flash-eligibility guard.
        enable_profiling: Turn on ORT op-level profiling so callers can
            inspect which ``If`` branch executed and with which inputs.
    """
    config = _base_config()
    config = dataclasses.replace(config, dtype=ir_dtype)
    module = registry.get(_MODEL_TYPE)(config)
    task = CausalLMTask(static_cache=True, max_seq_len=_MAX_SEQ_LEN)
    package = build_from_module(
        module, config, task=task, execution_provider="default"
    )
    model = package["model"]
    _fill_random_weights(model, np.random.default_rng(0))

    model_path = str(Path(tmp_dir) / "model.onnx")
    ir.save(model, model_path, external_data="model.onnx.data")
    session_options = ort.SessionOptions()
    if enable_profiling:
        session_options.enable_profiling = True
        session_options.profile_file_prefix = str(Path(tmp_dir) / "prof")
    session = ort.InferenceSession(
        model_path,
        session_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert "CUDAExecutionProvider" in session.get_providers(), (
        "static-cache regression test must run on CUDA"
    )
    return session, config


def _empty_caches(
    num_layers: int, kv_hidden: int, np_dtype: np.dtype = _CACHE_DTYPE
) -> dict[str, np.ndarray]:
    """Zeroed ``[1, max_seq, kv_hidden]`` cache buffers for every layer."""
    feeds: dict[str, np.ndarray] = {}
    for layer in range(num_layers):
        zeros = np.zeros((1, _MAX_SEQ_LEN, kv_hidden), dtype=np_dtype)
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


def _executed_attention_events(profile_path: str) -> list[dict]:
    """Op-level ``Attention`` events from an ORT profiling JSON.

    Each returned dict has ``name`` (carries the ``static_cache_prefill`` /
    ``static_cache_decode`` branch tag) and ``has_mask`` — True when the
    executed node received a rank-4 ``attn_mask`` input.  ORT 1.27's Python
    profiler emits only op-level ``Node`` events (no CUDA ``Kernel`` events),
    so the *internal* attention kernel (Flash vs memory-efficient) is not
    observable here; the mask-input signature is, and the presence of an
    ``attn_mask`` is exactly what makes ORT ineligible for Flash.
    """
    with open(profile_path) as handle:
        events = json.load(handle)

    attention_events: list[dict] = []
    for event in events:
        args = event.get("args", {})
        if args.get("op_name") != "Attention":
            continue
        input_shapes = args.get("input_type_shape", [])
        has_mask = any(
            len(next(iter(shape.values()))) == 4 for shape in input_shapes
        )
        attention_events.append({"name": event["name"], "has_mask": has_mask})
    return attention_events


def test_static_cache_decode_runs_maskless_on_cuda():
    """The decode branch executes maskless at runtime (Flash-eligible).

    Build-time tests assert the *graph* phase-splits the static-cache
    attention behind an ``If`` (decode branch omits ``attn_mask``).  This
    test closes the loop at runtime on the production fp16 path: it profiles
    a single-token decode and a multi-token prefill on CUDA and asserts the
    ``If`` routed correctly and that the executed decode ``Attention`` carries
    **no** ``attn_mask`` input.

    Why this matters: ORT disables Flash whenever ``attn_mask`` is present
    (by pointer, not content), so a regression that wired the mask onto the
    decode branch would silently push the hot decode path onto the slower
    memory-efficient kernel and invalidate the Attention-vs-GQA decode
    comparison.  ORT 1.27's Python profiler does not surface the internal
    kernel name, so we assert the structural precondition (mask absence)
    that deterministically governs Flash eligibility.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session, config = _build_static_cache_session(
            tmp_dir, ir_dtype=ir.DataType.FLOAT16, enable_profiling=True
        )
        num_layers = config.num_hidden_layers
        kv_hidden = config.num_key_value_heads * config.head_dim
        vocab = config.vocab_size
        output_names = [out.name for out in session.get_outputs()]
        rng = np.random.default_rng(2)

        # Single-token decode (S_q = 1): the If must take the maskless branch.
        decode_feeds: dict[str, np.ndarray] = {
            "input_ids": rng.integers(0, vocab, size=(1, 1), dtype=np.int64),
            "position_ids": np.array([[4]], dtype=np.int64),
            "write_indices": np.array([4], dtype=np.int64),
            "nonpad_kv_seqlen": np.array([5], dtype=np.int64),
        }
        decode_feeds.update(_empty_caches(num_layers, kv_hidden, np.float16))
        session.run(output_names, decode_feeds)

        # Multi-token prefill (S_q = 4): the If must take the masked branch.
        prefill_len = 4
        prefill_feeds: dict[str, np.ndarray] = {
            "input_ids": rng.integers(
                0, vocab, size=(1, prefill_len), dtype=np.int64
            ),
            "position_ids": np.arange(prefill_len, dtype=np.int64)[None, :],
            "write_indices": np.array([0], dtype=np.int64),
            "nonpad_kv_seqlen": np.array([prefill_len], dtype=np.int64),
        }
        prefill_feeds.update(_empty_caches(num_layers, kv_hidden, np.float16))
        session.run(output_names, prefill_feeds)

        events = _executed_attention_events(session.end_profiling())

    decode_events = [e for e in events if "static_cache_decode" in e["name"]]
    prefill_events = [e for e in events if "static_cache_prefill" in e["name"]]

    # Decode took the maskless branch on every layer (Flash-eligible).
    assert len(decode_events) == num_layers, (
        f"expected {num_layers} decode-branch Attention executions, "
        f"got {len(decode_events)}"
    )
    assert all(not e["has_mask"] for e in decode_events), (
        "decode-branch Attention must run WITHOUT an attn_mask input so ORT "
        "keeps it on Flash; a mask here forces the slower memory-efficient "
        "path and breaks the decode-latency comparison"
    )

    # Prefill took the masked branch on every layer (memory-efficient path).
    assert len(prefill_events) == num_layers, (
        f"expected {num_layers} prefill-branch Attention executions, "
        f"got {len(prefill_events)}"
    )
    assert all(e["has_mask"] for e in prefill_events), (
        "prefill-branch Attention must carry the explicit causal mask"
    )
