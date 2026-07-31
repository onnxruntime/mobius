# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the Gemma 3n USM audio encoder.

HF's ``Gemma3nAudioEncoder`` is importable, so the arithmetic is diffed against
it end to end rather than re-derived.  That parity check is the load-bearing
one: it is what pins the cumulative group norm, the reverse-causal SSCP
padding, and the claim that flattening HF's chunked attention to full T×T
attention is exactly equivalent offline.

It is run over a sweep of sequence lengths and context configurations, because
a single length can hide all three of the interesting edge cases: a ``T`` that
is not a multiple of the chunk size, a non-zero right context, and a
zero-length history window.

What parity does *not* cover, and is therefore also asserted here:

* the 269-tensor checkpoint name and shape contract (a renamed weight fails to
  load, but only at export time against the real checkpoint);
* that the sinusoidal ``sin_emb`` constants are pre-populated, since nothing in
  the checkpoint will fill them;
* initializer registration under qualified names — building nodes from a
  submodule's parameters outside ``Module.__call__`` silently yields
  unqualified, unregistered initializers that serialise into a model which
  loads but computes with garbage;
* that the mask is honoured at all, and with the documented polarity, which
  parity alone would miss if both sides inverted it together.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius.components._gemma3n_audio import (
    Gemma3nAudioEncoder,
    _sscp_freq_out_dim,
    _timing_signal,
)

# Small but non-degenerate. The frequency flow 16 -> 8 -> 4 keeps two real
# subsampling stages, and asymmetric channels [8, 4] mean a transposed or
# channel-major flatten into input_proj_linear would not silently pass.
_HIDDEN = 32
_HEADS = 4
_LAYERS = 2
_FEAT = 16
_CHANNELS = [8, 4]
_CHUNK = 4
_LEFT = 5
_RIGHT = 0
_REDUCTION = 2

# Non-default alternatives, to check the config fields are actually threaded
# rather than the defaults being silently reused.
_ALT_CONTEXTS = [
    pytest.param(3, 4, 2, 1, id="right-context"),
    pytest.param(2, 1, 0, 1, id="no-history"),
    pytest.param(12, 13, 0, 4, id="e4b-defaults"),
]

_SEQ_LENS = [
    (20, 16),  # padded tail
    (7, 7),  # not a multiple of the chunk size
    (33, 20),  # odd length, padded tail
    (4, 1),  # almost entirely padding
    (64, 64),  # several full chunks, no padding
]


def _make_encoder(
    chunk: int = _CHUNK,
    left: int = _LEFT,
    right: int = _RIGHT,
    reduction: int = _REDUCTION,
    layers: int = _LAYERS,
    gradient_clipping: float = 1e10,
) -> Gemma3nAudioEncoder:
    return Gemma3nAudioEncoder(
        input_feat_size=_FEAT,
        hidden_size=_HIDDEN,
        num_heads=_HEADS,
        num_layers=layers,
        conv_channel_size=_CHANNELS,
        attention_chunk_size=chunk,
        attention_context_left=left,
        attention_context_right=right,
        reduction_factor=reduction,
        gradient_clipping=gradient_clipping,
    )


def _hf_reference(
    chunk: int = _CHUNK,
    left: int = _LEFT,
    right: int = _RIGHT,
    reduction: int = _REDUCTION,
    layers: int = _LAYERS,
    gradient_clipping: float = 1e10,
    seed: int = 0,
):
    """Build the HF encoder with randomized weights, or skip if unavailable."""
    torch = pytest.importorskip("torch")
    config_mod = pytest.importorskip(
        "transformers.models.gemma3n.configuration_gemma3n"
    )
    modeling = pytest.importorskip("transformers.models.gemma3n.modeling_gemma3n")

    torch.manual_seed(seed)
    hf = modeling.Gemma3nAudioEncoder(
        config_mod.Gemma3nAudioConfig(
            hidden_size=_HIDDEN,
            conf_num_attention_heads=_HEADS,
            conf_num_hidden_layers=layers,
            input_feat_size=_FEAT,
            sscp_conv_channel_size=_CHANNELS,
            conf_attention_chunk_size=chunk,
            conf_attention_context_left=left,
            conf_attention_context_right=right,
            conf_reduction_factor=reduction,
            gradient_clipping=gradient_clipping,
        )
    ).eval()
    for param in hf.parameters():
        with torch.no_grad():
            # Randomize away from the all-ones norms and zeroed per_dim_scale:
            # at their initial values several weights are identity-like and a
            # dropped norm or scale would go unnoticed.
            param.copy_(torch.randn_like(param) * 0.3)
    state = {n: p.detach().numpy().astype(np.float32) for n, p in hf.named_parameters()}
    return hf, state, torch


def _graph_inputs(batch: int = 1) -> list[ir.Value]:
    return [
        ir.Value(
            name="input_features",
            shape=ir.Shape([batch, "T", _FEAT]),
            type=ir.TensorType(ir.DataType.FLOAT),
        ),
        ir.Value(
            name="input_features_mask",
            shape=ir.Shape([batch, "T"]),
            type=ir.TensorType(ir.DataType.BOOL),
        ),
    ]


def _build_graph(encoder: Gemma3nAudioEncoder, batch: int = 1) -> ir.Graph:
    """Build ``encoder`` into a graph with no weights loaded.

    Enough for the initializer-registration assertions, which is the level the
    name contract lives at.
    """
    inputs = _graph_inputs(batch)
    graph = ir.Graph(
        inputs=inputs,
        outputs=[],
        nodes=[],
        name="test_gemma3n_audio",
        opset_imports={"": OPSET_VERSION},
    )
    gb = GraphBuilder(graph)
    encodings, mask = encoder(gb.op, *inputs)
    for name, value in (("encodings", encodings), ("mask", mask)):
        value.name = name
        graph.outputs.append(value)
    return graph


def _build_session(encoder: Gemma3nAudioEncoder, state: dict, batch: int = 1):
    """Build ``encoder`` into an in-memory ONNX session with ``state`` weights.

    The sinusoidal ``sin_emb`` constants keep their pre-populated values, so
    they are absent from ``state``.  Serialising in memory (rather than via a
    tempfile) avoids Windows PermissionError under concurrent tests.
    """
    graph = _build_graph(encoder, batch)
    for name, param in encoder.named_parameters():
        if name in state:
            param.const_value = ir.tensor(state[name])

    options = ort.SessionOptions()
    # The float32 CastLike on the sin_emb constant has no CPU kernel to fold,
    # which is a warning, not an error; it would otherwise bury the output.
    options.log_severity_level = 3
    proto = ir.serde.serialize_model(ir.Model(graph, ir_version=11))
    return ort.InferenceSession(
        proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _mel_inputs(seq_len: int, num_valid: int, batch: int = 1) -> dict:
    """Random mel features plus a mobius-polarity (True = valid) mask."""
    features = (
        np.random.default_rng(seq_len)
        .standard_normal((batch, seq_len, _FEAT))
        .astype(np.float32)
    )
    valid = np.zeros((batch, seq_len), dtype=bool)
    valid[:, :num_valid] = True
    return {"input_features": features, "input_features_mask": valid}


# ---------------------------------------------------------------------------
# Weight contract
# ---------------------------------------------------------------------------


def test_parameter_names_match_huggingface():
    """Names must match ``model.audio_tower.*`` verbatim — no renaming hook.

    ``sin_emb`` is the only extra: it is derived from hyperparameters, not
    shipped in the checkpoint.
    """
    _, state, _ = _hf_reference()
    names = {n for n, _ in _make_encoder().named_parameters()}

    extra = names - set(state)
    assert names - extra == set(state)
    assert extra == {
        f"conformer.{i}.attention.attn.relative_position_embedding.sin_emb"
        for i in range(_LAYERS)
    }


def test_parameter_shapes_match_huggingface():
    """Every shared tensor must agree on shape, not just on name."""
    _, state, _ = _hf_reference()
    for name, param in _make_encoder().named_parameters():
        if name not in state:
            continue
        assert [int(d) for d in param.shape] == list(state[name].shape), name


def test_checkpoint_tensor_count_matches_the_real_model():
    """E4B ships 269 audio tensors: 5 subsample + 22 per conformer block."""
    encoder = Gemma3nAudioEncoder()  # published E4B defaults
    names = {
        n
        for n, _ in encoder.named_parameters()
        if not n.endswith("relative_position_embedding.sin_emb")
    }
    assert len(names) == 269
    assert sum(1 for n in names if n.startswith("subsample_conv_projection.")) == 5
    assert sum(1 for n in names if n.startswith("conformer.0.")) == 22


def test_projections_have_no_bias():
    """Every audio-tower ``nn.Linear`` in HF is ``bias=False``."""
    encoder = _make_encoder()
    assert encoder.subsample_conv_projection.input_proj_linear.bias is None
    block = encoder.conformer[0]
    assert block.attention.post.bias is None
    assert block.attention.attn.q_proj.bias is None
    assert block.attention.attn.relative_position_embedding.pos_proj.bias is None
    assert block.lconv1d.linear_start.bias is None
    assert block.ffw_layer_start.ffw_layer_1.bias is None


def test_no_activation_clipping_bounds():
    """Gemma 3n reuses Gemma 4's blocks but ships no ``ClippableLinear`` bounds.

    Leaving the ``ClippableLinear`` default in place would add four
    initializers per projection that the checkpoint cannot fill.
    """
    names = {n for n, _ in _make_encoder().named_parameters()}
    assert not any(
        n.endswith((".input_min", ".input_max", ".output_min", ".output_max"))
        for n in names
    )


def test_sin_emb_is_a_populated_constant():
    """Nothing in the checkpoint fills ``sin_emb``, so it must be pre-populated."""
    rpe = _make_encoder().conformer[0].attention.attn.relative_position_embedding
    weight = rpe.sin_emb
    assert weight.const_value is not None
    # span = L + R + 1, with L = attention_context_left - 1.
    assert [int(d) for d in weight.shape] == [_LEFT + _RIGHT, _HIDDEN]
    assert np.isfinite(weight.const_value.numpy()).all()


def test_initializers_are_registered_under_qualified_names():
    """Every parameter must reach the graph under its qualified name.

    A bare ``weight`` here would mean nodes were built from a submodule's
    parameters outside ``Module.__call__``: the model still serialises and
    still loads, but computes with uninitialized values.  The graph also holds
    hoisted ``const_*`` scalars, which are not parameters.
    """
    encoder = _make_encoder()
    graph = _build_graph(encoder)
    names = {n for n, _ in encoder.named_parameters()}
    assert names <= set(graph.initializers)
    assert not any(
        n in {"weight", "bias", "per_dim_scale", "sin_emb"} for n in graph.initializers
    )


# ---------------------------------------------------------------------------
# Derived dimensions
# ---------------------------------------------------------------------------


def test_sscp_frequency_flow_matches_the_checkpoint():
    """E4B's ``input_proj_linear`` is [1536, 1024], i.e. 32 channels x 32 bins.

    That 1024 is the one hard external check on the frequency arithmetic: get
    the padding wrong and the projection no longer matches the checkpoint.
    """
    assert _sscp_freq_out_dim(128, 3, 2) == 64
    assert _sscp_freq_out_dim(64, 3, 2) == 32
    subsample = Gemma3nAudioEncoder().subsample_conv_projection
    assert subsample.input_proj_in_features == 1024
    assert [int(d) for d in subsample.input_proj_linear.weight.shape] == [1536, 1024]


def test_sscp_time_padding_is_reverse_causal():
    """ONNX pads are ``[t_begin, f_begin, t_end, f_end]``.

    Time is padded only at the end (JAX ``reverse_causal``); frequency is
    padded symmetrically. Swapping the two would still produce plausible
    shapes for a square kernel.
    """
    conv = Gemma3nAudioEncoder().subsample_conv_projection.conv_0.conv
    assert conv._pads == [0, 1, 2, 1]  # kernel_h 3 -> t_end 2


def test_timing_signal_matches_huggingface():
    """The sinusoid must use HF's concatenated (not interleaved) layout."""
    _, _, torch = _hf_reference()
    modeling = pytest.importorskip("transformers.models.gemma3n.modeling_gemma3n")
    config_mod = pytest.importorskip(
        "transformers.models.gemma3n.configuration_gemma3n"
    )

    hf_rpe = modeling.Gemma3nAudioRelativePositionEmbedding(
        config_mod.Gemma3nAudioConfig(
            hidden_size=_HIDDEN, conf_num_attention_heads=_HEADS
        )
    )
    positions = np.arange(4, -3, -1)
    expected = hf_rpe._get_timing_signal_1d_pos(
        torch.from_numpy(positions).unsqueeze(0), dtype=torch.float32
    )
    np.testing.assert_allclose(
        _timing_signal(positions, _HIDDEN), expected.squeeze(0).numpy(), atol=1e-6
    )


def test_non_square_sscp_kernels_are_rejected():
    """``Conv2dNoBias`` takes one int per axis pair, so this must not pass silently."""
    with pytest.raises(NotImplementedError, match="square kernels"):
        Gemma3nAudioEncoder(conv_kernel_size_2d=[[3, 5], [3, 3]])


# ---------------------------------------------------------------------------
# Numerical parity with HuggingFace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("seq_len", "num_valid"), _SEQ_LENS)
def test_matches_huggingface(seq_len, num_valid):
    """Full-encoder parity over lengths that stress the chunking and padding."""
    hf, state, torch = _hf_reference()
    encoder = _make_encoder()
    session = _build_session(encoder, state)

    feeds = _mel_inputs(seq_len, num_valid)
    encodings, mask = session.run(None, feeds)

    # HF's audio_mel_mask is True for *padded* frames — the inverse convention.
    expected = hf(
        torch.from_numpy(feeds["input_features"]),
        torch.from_numpy(~feeds["input_features_mask"]),
    )
    np.testing.assert_allclose(
        encodings, expected.last_hidden_state.detach().numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_array_equal(mask, ~expected.audio_mel_mask.numpy())


@pytest.mark.parametrize(("chunk", "left", "right", "reduction"), _ALT_CONTEXTS)
def test_matches_huggingface_for_other_contexts(chunk, left, right, reduction):
    """Parity must hold for a right context and for a zero-length history.

    ``right > 0`` is the case where flattening HF's blocked attention could
    plausibly diverge, and ``left = 1`` (history window 0) is where an
    off-by-one in the window bound shows up as a wrong result rather than a
    shape error.
    """
    kwargs = dict(chunk=chunk, left=left, right=right, reduction=reduction, layers=1)
    hf, state, torch = _hf_reference(**kwargs)
    encoder = _make_encoder(**kwargs)
    session = _build_session(encoder, state)

    feeds = _mel_inputs(20, 16)
    encodings, _ = session.run(None, feeds)

    expected = hf(
        torch.from_numpy(feeds["input_features"]),
        torch.from_numpy(~feeds["input_features_mask"]),
    )
    np.testing.assert_allclose(
        encodings, expected.last_hidden_state.detach().numpy(), rtol=1e-4, atol=1e-4
    )


def test_matches_huggingface_with_interior_padding():
    """A mask with a *hole* in it, not just a padded tail.

    Real ``audio_mel_mask``s only ever pad the tail, which makes the mask gate
    in front of the light conv unobservable: that conv is causal, so a valid
    frame never reads a later padded one.  An interior hole is what forces the
    gate to matter, and dropping it here diverges from HF.
    """
    hf, state, torch = _hf_reference()
    encoder = _make_encoder()
    session = _build_session(encoder, state)

    feeds = _mel_inputs(24, 24)
    feeds["input_features_mask"][:, 8:16] = False

    encodings, _ = session.run(None, feeds)
    expected = hf(
        torch.from_numpy(feeds["input_features"]),
        torch.from_numpy(~feeds["input_features_mask"]),
    )
    np.testing.assert_allclose(
        encodings, expected.last_hidden_state.detach().numpy(), rtol=1e-4, atol=1e-4
    )


def test_matches_huggingface_when_gradient_clipping_binds():
    """Parity with a clamp small enough to actually clip.

    At the shipped ``1e10`` every clamp is the identity, so their placement is
    untested: in particular the attention block's residual is the *unclipped*
    input, and reading the clipped value instead would go unnoticed.
    """
    kwargs = dict(gradient_clipping=0.1, layers=1)
    hf, state, torch = _hf_reference(**kwargs)
    encoder = _make_encoder(**kwargs)
    session = _build_session(encoder, state)

    feeds = _mel_inputs(20, 16)
    encodings, _ = session.run(None, feeds)
    expected = hf(
        torch.from_numpy(feeds["input_features"]),
        torch.from_numpy(~feeds["input_features_mask"]),
    )
    np.testing.assert_allclose(
        encodings, expected.last_hidden_state.detach().numpy(), rtol=1e-4, atol=1e-4
    )


def test_matches_huggingface_batched():
    """Two utterances with different valid lengths in one batch.

    The mask subsample uses a batch-shared ``Gather`` where HF expands the
    indices per row; equivalent only because the indices do not depend on the
    batch, which this pins.
    """
    hf, state, torch = _hf_reference()
    encoder = _make_encoder()
    session = _build_session(encoder, state, batch=2)

    feeds = _mel_inputs(24, 24, batch=2)
    feeds["input_features_mask"][1, 9:] = False

    encodings, mask = session.run(None, feeds)
    expected = hf(
        torch.from_numpy(feeds["input_features"]),
        torch.from_numpy(~feeds["input_features_mask"]),
    )
    np.testing.assert_allclose(
        encodings, expected.last_hidden_state.detach().numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_array_equal(mask, ~expected.audio_mel_mask.numpy())


# ---------------------------------------------------------------------------
# Mask behaviour
# ---------------------------------------------------------------------------


def test_padded_outputs_are_zero():
    """HF ``masked_fill``s padded frames after the stride reduction."""
    _, state, _ = _hf_reference()
    session = _build_session(_make_encoder(), state)

    encodings, mask = session.run(None, _mel_inputs(24, 8))
    assert not mask.all(), "test needs some padded frames to be meaningful"
    np.testing.assert_array_equal(encodings[~mask], 0.0)
    assert np.abs(encodings[mask]).max() > 0


def test_mask_changes_valid_frame_outputs():
    """Padding must be excluded from attention, not merely zeroed afterwards.

    Zeroing the output alone would leave valid frames unchanged, so this needs a
    lookahead window (``right = 2``) to be a real test: with the default
    ``right = 0`` everything downstream of the mask is causal and shortening the
    mask genuinely cannot move an earlier frame.
    """
    kwargs = dict(chunk=3, left=4, right=2, reduction=1, layers=1)
    _, state, _ = _hf_reference(**kwargs)
    session = _build_session(_make_encoder(**kwargs), state)

    feeds = _mel_inputs(24, 24)
    all_valid, _ = session.run(None, feeds)

    feeds["input_features_mask"] = feeds["input_features_mask"].copy()
    feeds["input_features_mask"][:, 12:] = False
    partial, mask = session.run(None, feeds)

    assert mask.any(), "expected some frames valid under both masks"
    assert not np.allclose(all_valid[mask], partial[mask], rtol=1e-3, atol=1e-3)


def test_causal_config_does_not_see_the_padded_tail():
    """With ``right = 0`` a shorter mask must leave the valid prefix bit-identical.

    The counterpart to the lookahead test above: every mask-sensitive step
    (attention window, the pre-conv gate, the causal depthwise conv) only ever
    looks backwards, so truncating the mask must not perturb earlier frames.
    A leaky window or a non-causal conv would show up here.
    """
    _, state, _ = _hf_reference()
    session = _build_session(_make_encoder(), state)

    feeds = _mel_inputs(24, 24)
    all_valid, _ = session.run(None, feeds)

    feeds["input_features_mask"] = feeds["input_features_mask"].copy()
    feeds["input_features_mask"][:, 12:] = False
    partial, mask = session.run(None, feeds)

    assert mask.any() and not mask.all()
    np.testing.assert_array_equal(all_valid[mask], partial[mask])


def test_mask_polarity_is_true_for_valid():
    """An all-False mask means "nothing valid", so the output must be all zero.

    If the polarity were inverted, this would instead be the fully-valid case
    and produce non-zero output.
    """
    _, state, _ = _hf_reference()
    session = _build_session(_make_encoder(), state)

    feeds = _mel_inputs(16, 0)
    encodings, mask = session.run(None, feeds)
    assert not mask.any()
    np.testing.assert_array_equal(encodings, 0.0)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [16, 24, 33, 64])
def test_output_length_is_input_reduced_by_16(seq_len):
    """Time is halved twice by the SSCP, then strided by ``reduction_factor``.

    E4B's factor is 4, so the total is 16x; ceil at each stage.
    """
    _, state, _ = _hf_reference(chunk=12, left=13, right=0, reduction=4, layers=1)
    encoder = _make_encoder(chunk=12, left=13, right=0, reduction=4, layers=1)
    session = _build_session(encoder, state)

    encodings, mask = session.run(None, _mel_inputs(seq_len, seq_len))
    subsampled = seq_len
    for stride in (2, 2, 4):
        subsampled = -(-subsampled // stride)  # ceil
    assert encodings.shape == (1, subsampled, _HIDDEN)
    assert mask.shape == (1, subsampled)
