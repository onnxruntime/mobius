# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SenseNova-U1.5 (NEO-unify) task — 5-model unified any-to-any split.

The upstream reference implementation runs two structurally different
passes over one backbone, so the export mirrors them one-for-one instead
of forcing a single graph:

``embedding`` + ``vision`` + ``model``
    The *understanding* pass.  Text (and reference-image) tokens are
    embedded, run through the understanding branch, and produce both
    ``logits`` (for text / think decoding) and the KV cache that
    conditions image generation.

``image_gen_embedding`` + ``image_gen_denoiser``
    The *generation* pass, executed once per flow-matching step.  The
    noisy image is patchified by a second vision tower, offset by the
    timestep / noise-scale embeddings, then run through the ``_mot_gen``
    branch while attending over the cached understanding prefix.  The
    pixel head returns the ``x0`` estimate.

Because upstream raises ``NotImplementedError`` for a mixed
understanding/generation forward, splitting at this boundary is the only
faithful decomposition — not merely a convenience.
"""

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
from mobius.tasks._cache_utils import (
    _make_kv_cache_inputs,
    _register_kv_cache_outputs,
)


class SenseNovaU1Task(ModelTask):
    """Build the five NEO-unify ONNX graphs."""

    model_roles: ClassVar[dict[str, str]] = {
        "model": "decoder",
        "vision": "encoder",
        "embedding": "embedding",
        "image_gen_embedding": "embedding",
        "image_gen_denoiser": "decoder",
    }
    components = ComponentSpec(
        model="model",
        vision="vision",
        embedding="embedding",
        image_gen_embedding="image_gen_embedding",
        image_gen_denoiser="image_gen_denoiser",
    )

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {
            "model": self._build_decoder(module.model, config),
            "vision": self._build_vision(module.vision, config),
            "embedding": self._build_embedding(module.embedding, config),
            "image_gen_embedding": self._build_image_gen_embedding(
                module.image_gen_embedding, config
            ),
            "image_gen_denoiser": self._build_image_gen_denoiser(
                module.image_gen_denoiser, config
            ),
        }
        return ModelPackage(models, config=config)

    # ── Understanding branch ────────────────────────────────────────────

    def _build_decoder(self, decoder: nn.Module, config: ArchitectureConfig) -> ir.Model:
        """``inputs_embeds`` + 3-axis positions + KV cache → ``logits``."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph()
        op = builder.op

        inputs_embeds = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=[batch, seq_len, config.hidden_size],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_seq_len + seq_len"],
        )
        # Three stacked rotary axes: temporal, image height, image width.
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[3, batch, seq_len],
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

        logits, present_key_values = decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        builder.add_output(logits, "logits")
        _register_kv_cache_outputs(builder, present_key_values)
        return _make_model(graph)

    def _build_vision(self, vision: nn.Module, config: ArchitectureConfig) -> ir.Model:
        """Reference image ``(1, 3, H, W)`` → ``image_features``."""
        height = ir.SymbolicDim("height")
        width = ir.SymbolicDim("width")

        graph, builder = _make_graph(name="vision")
        op = builder.op
        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[1, 3, height, width],
        )
        image_features = vision(op, pixel_values)
        builder.add_output(image_features, "image_features")
        return _make_model(graph)

    def _build_embedding(self, embedding: nn.Module, config: ArchitectureConfig) -> ir.Model:
        """``input_ids`` (+ optional image features) → ``inputs_embeds``."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_features = ir.SymbolicDim("num_image_tokens")

        graph, builder = _make_graph(name="embedding")
        op = builder.op
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        image_features = builder.input(
            "image_features",
            dtype=config.dtype,
            shape=[1, num_features, config.hidden_size],
        )
        image_mask = builder.input(
            "image_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, seq_len],
        )
        inputs_embeds = embedding(
            op,
            input_ids,
            image_features=image_features,
            image_mask=image_mask,
        )
        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)

    # ── Generation branch ───────────────────────────────────────────────

    def _build_image_gen_embedding(
        self, module: nn.Module, config: ArchitectureConfig
    ) -> ir.Model:
        """Noisy latent + timestep (+ noise scale) → generation embeds."""
        height = ir.SymbolicDim("height")
        width = ir.SymbolicDim("width")

        graph, builder = _make_graph(name="image_gen_embedding")
        op = builder.op
        latent = builder.input(
            "latent",
            dtype=config.dtype,
            shape=[1, 3, height, width],
        )
        timestep = builder.input("timestep", dtype=config.dtype, shape=[1])
        noise_scale = builder.input("noise_scale", dtype=config.dtype, shape=[1])
        image_embeds = module(op, latent, timestep, noise_scale)
        builder.add_output(image_embeds, "image_embeds")
        return _make_model(graph)

    def _build_image_gen_denoiser(
        self, module: nn.Module, config: ArchitectureConfig
    ) -> ir.Model:
        """Generation-branch transformer + pixel head → ``x0`` prediction."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("image_tokens")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph(name="image_gen_denoiser")
        op = builder.op

        image_embeds = builder.input(
            "image_embeds",
            dtype=config.dtype,
            shape=[batch, seq_len, config.hidden_size],
        )
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[3, batch, seq_len],
        )
        token_grid = builder.input(
            "token_grid",
            dtype=ir.DataType.INT64,
            shape=[2],
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
        predicted_image = module(
            op,
            image_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_grid=token_grid,
        )
        builder.add_output(predicted_image, "predicted_image")
        return _make_model(graph)
