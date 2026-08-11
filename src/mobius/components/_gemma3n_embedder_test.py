# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the Gemma 3n multimodal embedder.

HF's ``Gemma3nMultimodalEmbedder`` is importable, so these tests diff against
it numerically rather than re-deriving the arithmetic.  What that does *not*
cover, and what is therefore also asserted here: the checkpoint weight-name
contract, and that both the soft and hard paths register their initializers
under qualified names (reaching into a submodule's parameters from outside
``Module.__call__`` silently produces unregistered, unqualified initializers).
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius.components._gemma3n_embedder import Gemma3nMultimodalEmbedder

# Small but non-degenerate: the multimodal and text widths differ so a
# transposed projection would not silently pass, and 8 vocab entries with a
# non-zero offset exercise the id rebasing.
_MM_HIDDEN = 16
_TEXT_HIDDEN = 24
_VOCAB_SIZE = 8
_VOCAB_OFFSET = 100
_EPS = 1e-6

# The four tensors the checkpoint actually ships per modality (verified against
# model.embed_vision.* / model.embed_audio.* in google/gemma-3n-E4B-it).
_CHECKPOINT_WEIGHTS = {
    "embedding.weight",
    "embedding_projection.weight",
    "hard_embedding_norm.weight",
    "soft_embedding_norm.weight",
}


def _make_embedder() -> Gemma3nMultimodalEmbedder:
    return Gemma3nMultimodalEmbedder(
        _MM_HIDDEN,
        _TEXT_HIDDEN,
        vocab_size=_VOCAB_SIZE,
        vocab_offset=_VOCAB_OFFSET,
        eps=_EPS,
    )


def _hf_reference():
    """Build the HF embedder with randomized weights, or skip if unavailable."""
    torch = pytest.importorskip("torch")
    config_mod = pytest.importorskip("transformers.models.gemma3n.configuration_gemma3n")
    modeling = pytest.importorskip("transformers.models.gemma3n.modeling_gemma3n")

    torch.manual_seed(0)
    hf = modeling.Gemma3nMultimodalEmbedder(
        config_mod.Gemma3nVisionConfig(
            hidden_size=_MM_HIDDEN,
            vocab_size=_VOCAB_SIZE,
            vocab_offset=_VOCAB_OFFSET,
            rms_norm_eps=_EPS,
        ),
        # num_attention_heads must divide hidden_size for the config validator.
        config_mod.Gemma3nTextConfig(hidden_size=_TEXT_HIDDEN, num_attention_heads=4),
    ).eval()
    for param in hf.parameters():
        with torch.no_grad():
            # Scale down so RMSNorm output stays O(1) and float32 comparison
            # is meaningful.
            param.copy_(torch.randn_like(param) * 0.5)
    state = {n: p.detach().numpy().astype(np.float32) for n, p in hf.named_parameters()}
    return hf, state, torch


def _build_graph(inputs, build_outputs) -> ir.Graph:
    """Build a graph from ``build_outputs(op, values) -> [(name, ir.Value)]``.

    No weights and no session: enough for the initializer-registration
    assertions, which is the level the name contract lives at.
    """
    graph = ir.Graph(
        inputs=list(inputs),
        outputs=[],
        nodes=[],
        name="test_gemma3n_embedder",
        opset_imports={"": OPSET_VERSION},
    )
    gb = GraphBuilder(graph)
    for name, value in build_outputs(gb.op, inputs):
        value.name = name
        graph.outputs.append(value)
    return graph


def _build_session(embedder, inputs, build_outputs, state):
    """Build ``embedder`` into an in-memory ONNX session with ``state`` weights.

    The scale-free post-projection norm keeps its pre-populated constant, so it
    is absent from ``state``.  Serialising in memory (rather than via a
    tempfile) avoids Windows PermissionError under concurrent tests.
    """
    graph = _build_graph(inputs, build_outputs)
    for name, param in embedder.named_parameters():
        if name in state:
            param.const_value = ir.tensor(state[name])

    proto = ir.serde.serialize_model(ir.Model(graph, ir_version=11))
    return ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])


def _feature_input(name: str = "inputs_embeds", tokens: int = 5) -> ir.Value:
    return ir.Value(
        name=name,
        shape=ir.Shape([2, tokens, _MM_HIDDEN]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )


def _id_input(name: str = "input_ids", tokens: int = 5) -> ir.Value:
    return ir.Value(
        name=name,
        shape=ir.Shape([2, tokens]),
        type=ir.TensorType(ir.DataType.INT64),
    )


# ---------------------------------------------------------------------------
# Weight contract
# ---------------------------------------------------------------------------


def test_parameter_names_match_the_checkpoint():
    """Names must match ``model.embed_vision.*`` verbatim — no renaming hook.

    ``embedding_post_projection_norm.weight`` is the extra one: HF builds that
    norm with ``with_scale=False`` so the checkpoint ships no tensor for it,
    and it is materialized here as a constant all-ones initializer.
    """
    names = {n for n, _ in _make_embedder().named_parameters()}
    assert names == _CHECKPOINT_WEIGHTS | {"embedding_post_projection_norm.weight"}


def test_parameter_shapes_match_the_checkpoint():
    """``embedding_projection`` maps multimodal width to text width, not back."""
    shapes = {n: [int(d) for d in p.shape] for n, p in _make_embedder().named_parameters()}
    assert shapes["embedding.weight"] == [_VOCAB_SIZE, _MM_HIDDEN]
    assert shapes["embedding_projection.weight"] == [_TEXT_HIDDEN, _MM_HIDDEN]
    assert shapes["hard_embedding_norm.weight"] == [_MM_HIDDEN]
    assert shapes["soft_embedding_norm.weight"] == [_MM_HIDDEN]
    assert shapes["embedding_post_projection_norm.weight"] == [_TEXT_HIDDEN]


def test_post_projection_norm_scale_is_a_populated_constant():
    """Scale-free means all-ones and pre-populated — nothing to load."""
    embedder = _make_embedder()
    weight = embedder.embedding_post_projection_norm.weight
    assert weight.const_value is not None
    np.testing.assert_array_equal(
        weight.const_value.numpy(), np.ones(_TEXT_HIDDEN, dtype=np.float32)
    )


def test_projection_has_no_bias():
    """HF's ``embedding_projection`` is ``nn.Linear(..., bias=False)``."""
    assert _make_embedder().embedding_projection.bias is None


# ---------------------------------------------------------------------------
# Argument dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="neither"),
        pytest.param({"inputs_embeds": True, "input_ids": True}, id="both"),
    ],
)
def test_forward_requires_exactly_one_input(kwargs):
    """Mirrors HF's XOR check; ambiguity here would silently pick a path."""
    embedder = _make_embedder()
    graph = ir.Graph(
        inputs=[], outputs=[], nodes=[], name="t", opset_imports={"": OPSET_VERSION}
    )
    op = GraphBuilder(graph).op
    resolved = {k: (_feature_input() if k == "inputs_embeds" else _id_input()) for k in kwargs}
    with pytest.raises(ValueError, match="exactly one"):
        embedder(op, **resolved)


def test_soft_path_emits_no_hard_path_weights():
    """A vision graph needing only soft tokens must not carry the 128-row table.

    The branch is resolved at build time, so the unused path contributes no
    nodes and no initializers.
    """
    embedder = _make_embedder()
    graph = _build_graph(
        [_feature_input()],
        lambda op, values: [("out", embedder(op, inputs_embeds=values[0]))],
    )
    assert set(graph.initializers) == {
        "soft_embedding_norm.weight",
        "embedding_projection.weight",
        "embedding_post_projection_norm.weight",
    }


def test_hard_path_registers_qualified_initializers():
    """Every hard-path weight must be registered under its qualified name.

    Building the graph outside ``Module.__call__`` would leave the submodule
    parameters unqualified (bare ``weight``) and unregistered, which serialises
    into a model that loads but computes with uninitialized values.
    """
    embedder = _make_embedder()
    graph = _build_graph(
        [_id_input()],
        lambda op, values: [("out", embedder(op, input_ids=values[0]))],
    )
    assert set(graph.initializers) == {
        "embedding.weight",
        "hard_embedding_norm.weight",
        "embedding_projection.weight",
        "embedding_post_projection_norm.weight",
    }


def test_one_instance_serves_both_paths_in_one_graph():
    """HF's audio path uses soft features *and* hard padding embeddings together.

    Both invocations must share the same initializers rather than duplicating
    them under a second qualified prefix.
    """
    embedder = _make_embedder()

    def outputs(op, values):
        return [
            ("soft", embedder(op, inputs_embeds=values[0])),
            ("hard", embedder(op, input_ids=values[1])),
        ]

    graph = _build_graph([_feature_input(), _id_input()], outputs)

    assert set(graph.initializers) == _CHECKPOINT_WEIGHTS | {
        "embedding_post_projection_norm.weight"
    }


# ---------------------------------------------------------------------------
# Numerical parity with HuggingFace
# ---------------------------------------------------------------------------


def test_soft_path_matches_huggingface():
    """soft_embedding_norm -> projection -> scale-free post-norm."""
    hf, state, torch = _hf_reference()
    embedder = _make_embedder()
    session = _build_session(
        embedder,
        [_feature_input()],
        lambda op, values: [("out", embedder(op, inputs_embeds=values[0]))],
        state,
    )

    x = np.random.default_rng(1).standard_normal((2, 5, _MM_HIDDEN)).astype(np.float32)
    got = session.run(None, {"inputs_embeds": x})[0]

    expected = hf(inputs_embeds=torch.from_numpy(x)).detach().numpy()
    assert got.shape == (2, 5, _TEXT_HIDDEN)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_hard_path_matches_huggingface():
    """Token ids are rebased by ``vocab_offset`` before the lookup.

    Skipping the subtraction would index the 128-row table out of range (or
    wrap), so this is the check that pins the offset.
    """
    hf, state, torch = _hf_reference()
    embedder = _make_embedder()
    session = _build_session(
        embedder,
        [_id_input(tokens=3)],
        lambda op, values: [("out", embedder(op, input_ids=values[0]))],
        state,
    )

    # Spans the full reserved range, including the first and last valid ids.
    token_ids = np.array(
        [
            [_VOCAB_OFFSET, _VOCAB_OFFSET + 1, _VOCAB_OFFSET + _VOCAB_SIZE - 1],
            [_VOCAB_OFFSET + 3, _VOCAB_OFFSET + 3, _VOCAB_OFFSET + 5],
        ],
        dtype=np.int64,
    )
    got = session.run(None, {"input_ids": token_ids})[0]

    expected = hf(input_ids=torch.from_numpy(token_ids)).detach().numpy()
    assert got.shape == (2, 3, _TEXT_HIDDEN)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_hard_path_distinguishes_token_ids():
    """Distinct ids must give distinct embeddings.

    A dropped ``Sub`` that collapsed every id to the same row would still match
    shapes and still pass a same-id comparison.
    """
    _, state, _ = _hf_reference()
    embedder = _make_embedder()
    session = _build_session(
        embedder,
        [_id_input(tokens=2)],
        lambda op, values: [("out", embedder(op, input_ids=values[0]))],
        state,
    )

    token_ids = np.array(
        [[_VOCAB_OFFSET, _VOCAB_OFFSET + 1], [_VOCAB_OFFSET + 2, _VOCAB_OFFSET + 3]],
        dtype=np.int64,
    )
    got = session.run(None, {"input_ids": token_ids})[0]

    rows = got.reshape(-1, _TEXT_HIDDEN)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            assert not np.allclose(rows[i], rows[j], rtol=1e-3, atol=1e-3)


def test_soft_and_hard_paths_differ():
    """The two norms are separate weights, so the paths must not alias.

    Wiring ``soft_embedding_norm`` into the hard branch (an easy copy-paste
    slip) would leave every other assertion here green.
    """
    _, state, _ = _hf_reference()
    embedder = _make_embedder()

    def outputs(op, values):
        return [
            ("soft", embedder(op, inputs_embeds=values[0])),
            ("hard", embedder(op, input_ids=values[1])),
        ]

    session = _build_session(
        embedder, [_feature_input(tokens=3), _id_input(tokens=3)], outputs, state
    )

    # Feed the hard path's own embedding rows through the soft path: identical
    # inputs to the projection tail, so any difference is the norm weights.
    rows = state["embedding.weight"][:3][None].repeat(2, axis=0)
    token_ids = np.array([[100, 101, 102], [100, 101, 102]], dtype=np.int64)
    soft, hard = session.run(None, {"inputs_embeds": rows, "input_ids": token_ids})

    assert not np.allclose(soft, hard, rtol=1e-3, atol=1e-3)
