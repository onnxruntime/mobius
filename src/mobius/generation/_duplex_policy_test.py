# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the full-duplex delay-cache policy components.

The reference implementations below are a direct transcription of the upstream
``moshi.models.lm.LMGen.prepare_step_input`` / ``_step`` delay bookkeeping used by
Moshi-family full-duplex models (Moshi, PersonaPlex). The ONNX components must
reproduce them exactly, because a single misplaced ring slot silently corrupts
the interleaved text/agent/user token streams.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius.generation import (
    build_duplex_frame_assemble,
    build_duplex_frame_commit,
    build_duplex_stream_append,
    build_duplex_stream_tail,
    build_duplex_teacher_select,
    build_duplex_waveform_append,
)

# PersonaPlex / Moshi channel layout: text, 8 agent streams, 8 user streams.
DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
CHANNELS = len(DELAYS)
MAX_DELAY = max(DELAYS)
CACHE_T = MAX_DELAY + 3
TEXT_INITIAL = 32000
AUDIO_INITIAL = 2048
INITIAL = [TEXT_INITIAL] + [AUDIO_INITIAL] * (CHANNELS - 1)


def _session(component) -> ort.InferenceSession:
    proto = ir.to_proto(component.model)
    return ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])


def _reference_assemble(cache, provided, offset, stream_tokens):
    """Upstream ``prepare_step_input`` for ``offset >= 1``."""
    cache = cache.copy()
    provided = provided.copy()
    for k, delay in enumerate(DELAYS):
        token = int(stream_tokens[0, k])
        if token >= 0:
            pos = (offset + delay) % CACHE_T
            cache[0, k, pos] = token
            provided[0, k, pos] = True
    for k, delay in enumerate(DELAYS):
        if offset <= delay:
            cache[0, k, offset % CACHE_T] = INITIAL[k]
            provided[0, k, offset % CACHE_T] = True
    input_pos = (offset - 1) % CACHE_T
    target_pos = offset % CACHE_T
    return (
        cache,
        provided,
        cache[:, :, input_pos : input_pos + 1].copy(),
        cache[:, :, target_pos : target_pos + 1].copy(),
        provided[:, :, target_pos : target_pos + 1].copy(),
    )


def _reference_commit(cache, provided, offset, frame):
    """Upstream cache commit plus delay-compensated read-out."""
    cache = cache.copy()
    provided = provided.copy()
    input_pos = (offset - 1) % CACHE_T
    target_pos = offset % CACHE_T
    provided[0, :, input_pos] = False
    for k in range(CHANNELS):
        if not provided[0, k, target_pos]:
            cache[0, k, target_pos] = int(frame[0, k])
    out = np.array(
        [cache[0, k, (offset - MAX_DELAY + DELAYS[k]) % CACHE_T] for k in range(CHANNELS)],
        np.int64,
    )[None]
    return cache, provided, out, offset + 1, offset > MAX_DELAY


def _random_state(rng):
    cache = rng.integers(0, 2048, size=(1, CHANNELS, CACHE_T)).astype(np.int64)
    provided = rng.random((1, CHANNELS, CACHE_T)) < 0.5
    return cache, provided


@pytest.mark.parametrize("offset", [1, 2, 3, 4, 5, 8, 123])
def test_duplex_frame_assemble_matches_reference(offset: int) -> None:
    rng = np.random.default_rng(offset)
    session = _session(build_duplex_frame_assemble(channels=CHANNELS, cache_length=CACHE_T))
    for trial in range(4):
        cache, provided = _random_state(rng)
        stream_tokens = rng.integers(-1, 2048, size=(1, CHANNELS)).astype(np.int64)
        if trial == 0:  # live phase: only the user streams carry tokens
            stream_tokens[:, :9] = -1
        got = session.run(
            None,
            {
                "token_cache": cache,
                "token_provided": provided,
                "offset": np.array(offset, np.int64),
                "stream_tokens": stream_tokens,
                "delays": np.array(DELAYS, np.int64),
                "initial_tokens": np.array(INITIAL, np.int64),
            },
        )
        want = _reference_assemble(cache, provided, offset, stream_tokens)
        for index, (actual, expected) in enumerate(zip(got, want)):
            np.testing.assert_array_equal(actual, expected, err_msg=f"output {index}")


@pytest.mark.parametrize("offset", [1, 2, 3, 4, 7, 122])
def test_duplex_frame_commit_matches_reference(offset: int) -> None:
    rng = np.random.default_rng(1000 + offset)
    session = _session(
        build_duplex_frame_commit(channels=CHANNELS, cache_length=CACHE_T, max_delay=MAX_DELAY)
    )
    for _ in range(4):
        cache, provided = _random_state(rng)
        frame = rng.integers(0, 2048, size=(1, CHANNELS)).astype(np.int64)
        got = session.run(
            None,
            {
                "token_cache": cache,
                "token_provided": provided,
                "offset": np.array(offset, np.int64),
                "frame": frame,
                "delays": np.array(DELAYS, np.int64),
            },
        )
        want = _reference_commit(cache, provided, offset, frame)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
        np.testing.assert_array_equal(got[2], want[2])
        assert int(got[3]) == want[3]
        assert bool(got[4]) == want[4]


def test_duplex_assemble_commit_round_trip_preserves_streams() -> None:
    """A full delayed round trip returns the tokens that were fed in.

    Streams with delay 1 are emitted one frame late, so feeding a known user
    stream and reading it back through the ring must reproduce it exactly.
    """
    assemble = _session(build_duplex_frame_assemble(channels=CHANNELS, cache_length=CACHE_T))
    commit = _session(
        build_duplex_frame_commit(channels=CHANNELS, cache_length=CACHE_T, max_delay=MAX_DELAY)
    )
    rng = np.random.default_rng(7)
    cache = np.full((1, CHANNELS, CACHE_T), -1, np.int64)
    provided = np.zeros((1, CHANNELS, CACHE_T), bool)
    cache[0, :, 0] = INITIAL
    offset = 1
    fed: list[list[int]] = []
    emitted: list[list[int]] = []
    for _ in range(12):
        user = rng.integers(0, 2048, size=8).astype(np.int64)
        fed.append(user.tolist())
        stream_tokens = np.full((1, CHANNELS), -1, np.int64)
        stream_tokens[0, 9:] = user
        cache, provided, _, _target, _target_provided = assemble.run(
            None,
            {
                "token_cache": cache,
                "token_provided": provided,
                "offset": np.array(offset, np.int64),
                "stream_tokens": stream_tokens,
                "delays": np.array(DELAYS, np.int64),
                "initial_tokens": np.array(INITIAL, np.int64),
            },
        )
        frame = rng.integers(0, 2048, size=(1, CHANNELS)).astype(np.int64)
        cache, provided, out, next_offset, emit = commit.run(
            None,
            {
                "token_cache": cache,
                "token_provided": provided,
                "offset": np.array(offset, np.int64),
                "frame": frame,
                "delays": np.array(DELAYS, np.int64),
            },
        )
        if bool(emit):
            emitted.append(out[0, 9:].tolist())
        offset = int(next_offset)
    # user streams have delay 1 and the read-out subtracts max_delay 1, so the
    # emitted user stream is exactly what was fed on the same frame.
    assert emitted == fed[: len(emitted)]
    assert len(emitted) == 11


def test_duplex_teacher_select_prefers_supplied_tokens() -> None:
    session = _session(build_duplex_teacher_select(channels=CHANNELS))
    target = np.arange(CHANNELS, dtype=np.int64).reshape(1, CHANNELS, 1) + 100
    provided = np.zeros((1, CHANNELS, 1), bool)
    provided[0, 3, 0] = True
    for index in (0, 3, 16):
        token = session.run(
            None,
            {
                "target": target,
                "target_provided": provided,
                "sampled": np.array([7], np.int64),
                "index": np.array(index, np.int64),
            },
        )[0]
        assert token.tolist() == ([103] if index == 3 else [7])


def test_duplex_stream_append_and_tail() -> None:
    append = _session(build_duplex_stream_append(streams=8))
    tail = _session(build_duplex_stream_tail(streams=8))
    prefix = np.zeros((1, 8, 0), np.int64)
    frames = []
    for step in range(5):
        frame = np.arange(8, dtype=np.int64).reshape(1, 8) + step * 8
        frames.append(frame)
        prefix = append.run(None, {"prefix": prefix, "frame": frame})[0]
    assert prefix.shape == (1, 8, 5)
    np.testing.assert_array_equal(prefix[:, :, -1], frames[-1])
    got = tail.run(None, {"prefix": prefix, "count": np.array(2, np.int64)})[0]
    np.testing.assert_array_equal(got, prefix[:, :, -2:])


def test_duplex_waveform_append_grows_packed_audio() -> None:
    append = _session(build_duplex_waveform_append())
    tail = _session(build_duplex_stream_tail(streams=1, dtype=ir.DataType.FLOAT))
    prefix = np.zeros((1, 1, 0), np.float32)
    for step in range(3):
        chunk = np.full((1, 1, 1920), float(step), np.float32)
        prefix = append.run(None, {"prefix": prefix, "chunk": chunk})[0]
    assert prefix.shape == (1, 1, 5760)
    got = tail.run(None, {"prefix": prefix, "count": np.array(1920, np.int64)})[0]
    np.testing.assert_allclose(got, np.full((1, 1, 1920), 2.0, np.float32))
