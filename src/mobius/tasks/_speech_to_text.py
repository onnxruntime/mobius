# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build configurable encoder and cached-decoder graphs for speech-to-text models."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, SpeechToTextConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._cache_utils import (
    _make_kv_cache_inputs,
    _register_kv_cache_outputs,
)


class SpeechToTextTask(ModelTask):
    """Encoder-decoder task shared by Whisper- and Moonshine-style ASR models.

    This task builds **two** separate ONNX models via :meth:`build`:

    - **encoder**: the audio input named by
      :attr:`SpeechToTextConfig.encoder_input_name` and, when
      :attr:`SpeechToTextConfig.encoder_uses_attention_mask` is true, an
      ``attention_mask``. It returns ``encoder_hidden_states`` and, for masked
      encoders such as Moonshine, ``encoder_attention_mask``.
    - **decoder**: ``decoder_input_ids``, ``encoder_hidden_states``,
      ``position_ids``, and ``past_key_values``. When
      :attr:`SpeechToTextConfig.decoder_uses_encoder_attention_mask` is true,
      the encoder's mask is also routed to the decoder. It returns ``logits``
      and ``present_key_values``.

    The module must expose ``model.encoder`` and ``model.decoder`` sub-modules
    following this configurable contract. Whisper uses ``input_features``
    without an encoder mask; Moonshine uses ``input_values`` and routes its
    downsampled encoder mask into cross-attention.
    """

    model_roles: ClassVar[dict[str, str]] = {"encoder": "encoder", "decoder": "decoder"}
    components: ClassVar[ComponentSpec] = ComponentSpec(
        encoder="model.encoder", decoder="model.decoder"
    )

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        if not isinstance(config, SpeechToTextConfig):
            raise TypeError(
                f"SpeechToTextTask requires SpeechToTextConfig, got {type(config).__name__}"
            )

        encoder_model = self._build_encoder(module.model.encoder, config)
        decoder_model = self._build_decoder(module.model.decoder, config)
        return ModelPackage(
            {"encoder": encoder_model, "decoder": decoder_model},
            config=config,
        )

    def _build_encoder(
        self,
        encoder: nn.Module,
        config: SpeechToTextConfig,
    ) -> ir.Model:
        """Build the encoder graph from the configured audio input and mask contract."""
        batch = ir.SymbolicDim("batch")
        audio_seq_len = ir.SymbolicDim("audio_seq_len")

        graph, builder = _make_graph(name="encoder")
        op = builder.op

        input_shape = [batch, audio_seq_len]
        if config.encoder_input_channels is not None:
            input_shape.insert(1, config.encoder_input_channels)
        encoder_input = builder.input(
            config.encoder_input_name,
            dtype=config.dtype,
            shape=input_shape,
        )

        if config.encoder_uses_attention_mask:
            attention_mask = builder.input(
                "attention_mask",
                dtype=ir.DataType.INT64,
                shape=[batch, audio_seq_len],
            )
            encoder_hidden_states, encoder_attention_mask = encoder(
                op,
                **{
                    config.encoder_input_name: encoder_input,
                    "attention_mask": attention_mask,
                },
            )
            builder.add_output(encoder_attention_mask, "encoder_attention_mask")
        else:
            encoder_hidden_states = encoder(op, **{config.encoder_input_name: encoder_input})

        builder.add_output(encoder_hidden_states, "encoder_hidden_states")

        return _make_model(graph)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: SpeechToTextConfig,
    ) -> ir.Model:
        """Build the cached decoder, routing the encoder mask only when configured."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")
        encoder_seq_len = ir.SymbolicDim("encoder_sequence_len")

        graph, builder = _make_graph()
        op = builder.op

        decoder_input_ids = builder.input(
            "decoder_input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        encoder_hidden_states = builder.input(
            "encoder_hidden_states",
            dtype=config.dtype,
            shape=[batch, encoder_seq_len, config.encoder_output_size],
        )
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )

        past_key_values = _make_kv_cache_inputs(
            builder,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )

        decoder_kwargs = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
        }
        if config.decoder_uses_encoder_attention_mask:
            decoder_kwargs["encoder_attention_mask"] = builder.input(
                "encoder_attention_mask",
                dtype=ir.DataType.INT64,
                shape=[batch, encoder_seq_len],
            )
        logits, present_key_values = decoder(op, **decoder_kwargs)

        builder.add_output(logits, "logits")
        _register_kv_cache_outputs(builder, present_key_values)

        return _make_model(graph)
