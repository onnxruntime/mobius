# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision encoder-decoder task for image-to-text generation."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ComponentSpec, _make_graph, _make_model
from mobius.tasks._seq2seq import Seq2SeqTask


class VisionEncoderDecoderTask(Seq2SeqTask):
    """Split an image encoder and cross-attentive text decoder into two models."""

    model_roles: ClassVar[dict[str, str]] = {
        "vision_encoder": "vision",
        "decoder": "decoder",
    }
    components: ClassVar[ComponentSpec] = ComponentSpec(
        vision_encoder="vision_encoder",
        decoder="decoder",
    )

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        return ModelPackage(
            {
                "vision_encoder": self._build_vision_encoder_graph(module, config),
                "decoder": self._build_decoder_graph(module, config),
            },
            config=config,
        )

    def _build_vision_encoder_graph(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        image_height = getattr(config, "image_height", None) or config.image_size
        image_width = getattr(config, "image_width", None) or config.image_size

        graph, builder = _make_graph(name="vision_encoder")
        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[batch, 3, image_height, image_width],
        )
        encoder_hidden_states = module.vision_encoder(
            builder.op,
            pixel_values=pixel_values,
        )
        builder.add_output(encoder_hidden_states, "last_hidden_state")
        return _make_model(graph)

    def _build_decoder_graph(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build the image-to-text decoder with self-attention caches only.

        The compressed visual sequence remains constant during decoding. The
        decoder projects it for cross-attention on each step, so exporting
        unused cross-cache inputs would force callers to allocate invalid dummy
        tensors and would expose outputs that cannot be consumed.
        """
        batch = ir.SymbolicDim("batch")
        dec_seq_len = ir.SymbolicDim("decoder_sequence_len")
        enc_seq_len = ir.SymbolicDim("encoder_sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")
        total_seq_len = ir.SymbolicDim("total_sequence_len")

        graph, builder = _make_graph(name="decoder")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, dec_seq_len],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, total_seq_len],
        )
        encoder_hidden_states = builder.input(
            "encoder_hidden_states",
            dtype=config.dtype,
            shape=[batch, enc_seq_len, config.hidden_size],
        )

        past_self_kvs: list[tuple[ir.Value, ir.Value]] = []
        num_decoder_layers = getattr(config, "num_decoder_layers", config.num_hidden_layers)
        for i in range(num_decoder_layers):
            past_key = builder.input(
                f"past_key_values.{i}.self.key",
                dtype=config.dtype,
                shape=[
                    batch,
                    config.num_key_value_heads,
                    past_seq_len,
                    config.head_dim,
                ],
            )
            past_value = builder.input(
                f"past_key_values.{i}.self.value",
                dtype=config.dtype,
                shape=[
                    batch,
                    config.num_key_value_heads,
                    past_seq_len,
                    config.head_dim,
                ],
            )
            past_self_kvs.append((past_key, past_value))

        logits, present_self_kvs = module.decoder(
            builder.op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_self_kvs,
        )
        builder.add_output(logits, "logits")
        for i, (key, value) in enumerate(present_self_kvs):
            builder.add_output(key, f"present.{i}.self.key")
            builder.add_output(value, f"present.{i}.self.value")
        return _make_model(graph)
