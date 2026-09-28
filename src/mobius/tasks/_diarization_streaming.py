# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Streaming (per-chunk, stateful) speaker-diarization task.

Builds a single ONNX graph that processes one audio chunk per call, carrying
the Arrival-Order Speaker Cache (AOSC) + FIFO queue state across calls as
fixed-size cache buffers plus scalar occupancy counters — the same "static
cache" convention used for KV caches elsewhere in mobius (see
``tasks/_cache_utils.py``).
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs import Nemotron3DiarizationConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class DiarizationStreamingTask(ModelTask):
    """Build an incremental, per-chunk ONNX graph for streaming speaker diarization.

    Input:
        ``input_features`` — ``[batch, feat, chunk_length + chunk_right_context]``
            raw (pre-subsampling) mel-spectrogram window: the chunk plus its
            look-ahead frames, at ``config.subsampling_factor`` frames per
            encoder frame.
        ``num_lookahead_frames`` — scalar ``int64``: how many trailing encoder
            frames of the window are look-ahead only (attended to, but not
            emitted or pushed to the FIFO). ``0`` for the last chunk of a
            recording.
        ``past_cache_embeds`` / ``past_cache_probs`` / ``past_fifo`` — fixed
            capacity AOSC + FIFO state buffers from the previous call (all
            zeros for the first chunk of a stream).
        ``past_num_cache_frames`` / ``past_num_fifo_frames`` — scalar
            ``int64`` occupancy counters for the two buffers above.
        ``past_is_compressed`` — scalar ``bool``: whether the AOSC has ever
            been compressed (governs how cached-frame probabilities are
            re-estimated on the next call).

    Output:
        ``speaker_probs`` — ``[batch, frames, num_spks]`` sigmoid
        probabilities for this chunk only (``frames = chunk_length *
        subsampling_factor``, trimmed to the input's raw length).
        ``present_cache_embeds`` / ``present_cache_probs`` / ``present_fifo``
        / ``present_num_cache_frames`` / ``present_num_fifo_frames`` /
        ``present_is_compressed`` — updated state, fed back as the ``past_*``
        inputs of the next call.
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config: Nemotron3DiarizationConfig,
    ) -> ModelPackage:
        graph, builder = _make_graph(name="nemotron3_diarization_streaming")
        op = builder.op

        window = config.chunk_length + config.chunk_right_context
        raw_window = window * config.subsampling_factor
        cache_capacity = config.streaming_speaker_cache_length
        fifo_capacity = config.streaming_fifo_length

        input_features = builder.input(
            "input_features",
            dtype=config.dtype,
            shape=["batch", config.feat_in, raw_window],
        )
        num_lookahead_frames = builder.input(
            "num_lookahead_frames", dtype=ir.DataType.INT64, shape=[]
        )
        past_cache_embeds = builder.input(
            "past_cache_embeds",
            dtype=config.dtype,
            shape=["batch", cache_capacity, config.hidden_size],
        )
        past_cache_probs = builder.input(
            "past_cache_probs",
            dtype=config.dtype,
            shape=["batch", cache_capacity, config.num_speakers],
        )
        past_fifo = builder.input(
            "past_fifo",
            dtype=config.dtype,
            shape=["batch", fifo_capacity, config.hidden_size],
        )
        past_num_cache_frames = builder.input(
            "past_num_cache_frames", dtype=ir.DataType.INT64, shape=[]
        )
        past_num_fifo_frames = builder.input(
            "past_num_fifo_frames", dtype=ir.DataType.INT64, shape=[]
        )
        past_is_compressed = builder.input(
            "past_is_compressed", dtype=ir.DataType.BOOL, shape=[]
        )

        (
            speaker_probs,
            present_cache_embeds,
            present_cache_probs,
            present_fifo,
            present_num_cache_frames,
            present_num_fifo_frames,
            present_is_compressed,
        ) = module.forward_streaming(
            op,
            input_features,
            num_lookahead_frames,
            past_cache_embeds,
            past_cache_probs,
            past_fifo,
            past_num_cache_frames,
            past_num_fifo_frames,
            past_is_compressed,
        )

        builder.add_output(speaker_probs, "speaker_probs")
        builder.add_output(present_cache_embeds, "present_cache_embeds")
        builder.add_output(present_cache_probs, "present_cache_probs")
        builder.add_output(present_fifo, "present_fifo")
        builder.add_output(present_num_cache_frames, "present_num_cache_frames")
        builder.add_output(present_num_fifo_frames, "present_num_fifo_frames")
        builder.add_output(present_is_compressed, "present_is_compressed")

        return ModelPackage({"model": _make_model(graph)}, config=config)
