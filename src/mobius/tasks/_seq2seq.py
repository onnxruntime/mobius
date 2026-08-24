# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Seq2Seq task for encoder-decoder models (T5, BART, etc.)."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
)


class Seq2SeqTask(ModelTask):
    """Encoder-decoder model for seq2seq generation.

    Produces a ModelPackage with two components:
    - "encoder": input_ids, attention_mask → last_hidden_state
    - "decoder": input_ids, encoder_hidden_states, attention_mask,
                 past_key_values → logits, present_key_values

    The module must have ``encoder`` and ``decoder`` attributes.
    """

    model_roles: ClassVar[dict[str, str]] = {"encoder": "encoder", "decoder": "decoder"}
    components: ClassVar[ComponentSpec] = ComponentSpec(encoder="encoder", decoder="decoder")

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        encoder_model = self._build_encoder_graph(module, config)
        decoder_model = self._build_decoder_graph(module, config)
        return ModelPackage(
            {"encoder": encoder_model, "decoder": decoder_model},
            config=config,
        )

    def _build_encoder_graph(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        graph, builder = _make_graph(name="encoder")
        op = builder.op

        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])
        attention_mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )

        encoder_hidden_states = module.encoder(
            op, input_ids=input_ids, attention_mask=attention_mask
        )

        builder.add_output(encoder_hidden_states, "last_hidden_state")

        return _make_model(graph)

    def _build_decoder_graph(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        dec_seq_len = ir.SymbolicDim("decoder_sequence_len")
        enc_seq_len = ir.SymbolicDim("encoder_sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph()
        op = builder.op

        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, dec_seq_len],
        )
        encoder_hidden_states = builder.input(
            "encoder_hidden_states",
            dtype=config.dtype,
            shape=[batch, enc_seq_len, config.hidden_size],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_seq_len + dec_seq_len"],
        )
        encoder_attention_mask = None
        if getattr(module, "uses_encoder_attention_mask", False):
            encoder_attention_mask = builder.input(
                "encoder_attention_mask",
                dtype=ir.DataType.INT64,
                shape=[batch, "cross_past_sequence_len + encoder_sequence_len"],
            )

        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        num_decoder_layers = config.num_decoder_layers or config.num_hidden_layers

        # Self-attention KV cache (named past_key_values.{i}.self.key/value)
        past_self_kvs: list[tuple[ir.Value, ir.Value]] = []
        for i in range(num_decoder_layers):
            past_key = builder.input(
                f"past_key_values.{i}.self.key",
                dtype=config.dtype,
                shape=[batch, num_heads, past_seq_len, head_dim],
            )
            past_value = builder.input(
                f"past_key_values.{i}.self.value",
                dtype=config.dtype,
                shape=[batch, num_heads, past_seq_len, head_dim],
            )
            past_self_kvs.append((past_key, past_value))

        # Cross-attention KV cache (named past_key_values.{i}.cross.key/value)
        cross_past_kvs: list[tuple[ir.Value, ir.Value]] = []
        for i in range(num_decoder_layers):
            past_key = builder.input(
                f"past_key_values.{i}.cross.key",
                dtype=config.dtype,
                shape=[batch, num_heads, "cross_past_sequence_len", head_dim],
            )
            past_value = builder.input(
                f"past_key_values.{i}.cross.value",
                dtype=config.dtype,
                shape=[batch, num_heads, "cross_past_sequence_len", head_dim],
            )
            cross_past_kvs.append((past_key, past_value))

        decoder_kwargs = dict(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_self_kvs,
            cross_past_key_values=cross_past_kvs,
        )
        if encoder_attention_mask is not None:
            decoder_kwargs["encoder_attention_mask"] = encoder_attention_mask
        logits, present_self_kvs, present_cross_kvs = module.decoder(
            op,
            **decoder_kwargs,
        )

        builder.add_output(logits, "logits")

        for i, (k, v) in enumerate(present_self_kvs):
            builder.add_output(k, f"present.{i}.self.key")
            builder.add_output(v, f"present.{i}.self.value")

        for i, (k, v) in enumerate(present_cross_kvs):
            builder.add_output(k, f"present.{i}.cross.key")
            builder.add_output(v, f"present.{i}.cross.value")

        return _make_model(graph)
