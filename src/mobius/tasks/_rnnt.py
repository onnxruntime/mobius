# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""RNN-T (transducer) task — builds ONNX graphs for FastConformer-RNNT.

Produces these models:
  - ``"encoder"``: mel features ``(B, feat_in, T)`` + ``length`` ``(B,)`` →
                   encoded ``(B, d_model, T')`` + ``encoder_length`` ``(B,)``
  - ``"encoder_streaming"``: one feature chunk + running attention/conv caches →
                   chunk encoding + updated caches (cache-aware streaming step)
  - ``"decoder"``: token ids ``(B, U)`` + LSTM state → prediction ``(B, d_pred, U)``
                   + next LSTM state
  - ``"joint"``:   encoder + prediction → logits ``(B, T', U, vocab+1)``

The transducer greedy-decode loop (alternating decoder/joint steps with the
encoder run once) is performed by the runtime, not inside any single graph.

Contract:
  - **Offline full-context** (``"encoder"``): consumes the full mel-feature
    sequence with causal convolutions and zero left-padding.  Ragged batches
    are supported via the ``length`` input (padded frames masked out of
    attention and zeroed in the output; ``encoder_length`` reports the
    subsampled valid length per sample).
  - **Cache-aware streaming** (``"encoder_streaming"``): consumes one feature
    chunk plus NeMo's per-layer ``cache_last_channel`` / ``cache_last_time`` /
    ``cache_last_channel_len`` state and returns the updated caches, enabling
    chunk-by-chunk inference.  The pre-encode (subsampling) left context is
    supplied by chunk overlap at the caller (NeMo ``pre_encode_cache_size``)
    and the leading ``drop_extra_pre_encoded`` subsampled frames are dropped.
    Like NeMo's streaming, this targets a single stream of equal-length chunks
    (batch size 1 / homogeneous, no intra-chunk padding); use the offline
    ``"encoder"`` for ragged batches.
"""

from __future__ import annotations

import copy
from typing import ClassVar

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model


class RNNTTask(ModelTask):
    """Build encoder/decoder/joint ONNX graphs for a FastConformer-RNNT model."""

    name = "fastconformer-rnnt"
    components: ClassVar[ComponentSpec] = ComponentSpec(
        encoder="encoder", decoder="prediction", joint="joint"
    )
    model_roles: ClassVar[dict[str, str]] = {
        "encoder": "encoder",
        "encoder_streaming": "encoder",
        "decoder": "encoder",
        "joint": "encoder",
    }

    def build(self, module, config: ArchitectureConfig) -> ModelPackage:
        self._validate_components(module)
        # The streaming encoder shares the offline encoder's weights (identical
        # initializer names) but needs its own parameter instances: onnxscript
        # realizes each Parameter into a single graph, so a deep copy lets the
        # streaming graph register the same-named initializers independently.
        stream_module = copy.deepcopy(module)
        return ModelPackage(
            {
                "encoder": self._build_encoder(module.encoder, config),
                "encoder_streaming": self._build_streaming_encoder(
                    stream_module.encoder, config
                ),
                "decoder": self._build_decoder(module.prediction, config),
                "joint": self._build_joint(module.joint, config),
            },
            config=config,
        )

    def _build_encoder(self, encoder, config: ArchitectureConfig) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")
        graph, builder = _make_graph(name="encoder")
        audio_signal = builder.input(
            "audio_signal",
            dtype=config.dtype,
            shape=[batch, config.fastconformer_feat_in, time],
        )
        # Per-sample valid feature-frame counts; drives padding-aware attention
        # masking and the returned subsampled lengths.
        length = builder.input("length", dtype=ir.DataType.INT64, shape=[batch])
        encoder_output, encoder_length = encoder(
            builder.op, audio_signal=audio_signal, length=length
        )
        builder.add_output(encoder_output, "encoder_output")
        builder.add_output(encoder_length, "encoder_length")
        return _make_model(graph)

    def _build_streaming_encoder(self, encoder, config: ArchitectureConfig) -> ir.Model:
        """Build the cache-aware streaming encoder graph.

        Consumes one feature chunk plus the running attention/conv caches and
        returns the chunk's encoded frames together with the updated caches, so
        a runtime can drive chunk-by-chunk streaming inference.

        Streaming contract: like NeMo's cache-aware streaming, the per-layer
        cache update keeps the physical tail of the chunk and grows
        ``cache_last_channel_len`` by the chunk's subsampled frame count. This is
        well-defined for a single stream of equal-length chunks (batch size 1,
        or a homogeneous batch with no intra-chunk padding). Ragged/padded
        batches whose caches are reused across chunks are not supported here —
        use the offline ``encoder`` for ragged batches.
        """
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")
        layers = config.num_hidden_layers
        d = config.hidden_size
        cache_size = config.fastconformer_streaming_cache_size
        conv_cache = config.fastconformer_conv_kernel_size - 1
        graph, builder = _make_graph(name="encoder_streaming")
        audio_signal = builder.input(
            "audio_signal",
            dtype=config.dtype,
            shape=[batch, config.fastconformer_feat_in, time],
        )
        length = builder.input("length", dtype=ir.DataType.INT64, shape=[batch])
        # Per-layer running caches (NeMo cache-aware streaming state).
        cache_last_channel = builder.input(
            "cache_last_channel",
            dtype=config.dtype,
            shape=[layers, batch, cache_size, d],
        )
        cache_last_time = builder.input(
            "cache_last_time",
            dtype=config.dtype,
            shape=[layers, batch, d, conv_cache],
        )
        cache_last_channel_len = builder.input(
            "cache_last_channel_len", dtype=ir.DataType.INT64, shape=[batch]
        )
        (
            encoder_output,
            encoder_length,
            cache_channel_next,
            cache_time_next,
            cache_len_next,
        ) = encoder(
            builder.op,
            audio_signal=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        builder.add_output(encoder_output, "encoder_output")
        builder.add_output(encoder_length, "encoder_length")
        builder.add_output(cache_channel_next, "cache_last_channel_next")
        builder.add_output(cache_time_next, "cache_last_time_next")
        builder.add_output(cache_len_next, "cache_last_channel_len_next")
        return _make_model(graph)

    def _build_decoder(self, prediction, config: ArchitectureConfig) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        labels = ir.SymbolicDim("labels")
        layers = config.rnnt_pred_rnn_layers
        hidden = config.rnnt_pred_hidden
        graph, builder = _make_graph(name="decoder")
        targets = builder.input("targets", dtype=ir.DataType.INT64, shape=[batch, labels])
        state_h = builder.input("state_h", dtype=config.dtype, shape=[layers, batch, hidden])
        state_c = builder.input("state_c", dtype=config.dtype, shape=[layers, batch, hidden])
        g, new_h, new_c = prediction(
            builder.op, targets=targets, state_h=state_h, state_c=state_c
        )
        builder.add_output(g, "decoder_output")
        builder.add_output(new_h, "state_h_out")
        builder.add_output(new_c, "state_c_out")
        return _make_model(graph)

    def _build_joint(self, joint, config: ArchitectureConfig) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")
        labels = ir.SymbolicDim("labels")
        graph, builder = _make_graph(name="joint")
        encoder_outputs = builder.input(
            "encoder_outputs",
            dtype=config.dtype,
            shape=[batch, config.hidden_size, time],
        )
        decoder_outputs = builder.input(
            "decoder_outputs",
            dtype=config.dtype,
            shape=[batch, config.rnnt_pred_hidden, labels],
        )
        logits = joint(
            builder.op,
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
        )
        builder.add_output(logits, "logits")
        return _make_model(graph)
