# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph contracts for MiniMax Music 3 neural components."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._diffusers_configs import (
    MINIMAX_MUSIC3_AUDIO_CODE_OFFSET,
    MINIMAX_MUSIC3_FEEDBACK_SCALE,
)
from mobius._model_package import ModelPackage
from mobius.components import Embedding
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import (
    _make_kv_cache_inputs,
    _register_kv_cache_outputs,
)


class _EmbeddingHolder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embed_tokens = embedding

    def forward(self, op, input_ids):
        return self.embed_tokens(op, input_ids)


class _LanguageEmbeddingModel(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.model = _EmbeddingHolder(embedding)

    def forward(self, op, input_ids):
        return self.model(op, input_ids)


class _AudioEmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.audio_embeddings = Embedding(vocab_size, hidden_size)

    def forward(self, op, code_ids):
        return self.audio_embeddings(op, code_ids)


class MiniMaxMusic3LanguageTask(ModelTask):
    """Build Qwen3 embedding and decoder graphs required by the autoregressive loop."""

    model_roles: ClassVar[dict[str, str]] = {
        "model": "decoder",
        "embedding": "embedding",
        "semantic_embedding": "embedding",
    }

    def build(self, module, config) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_length")
        past_len = ir.SymbolicDim("past_sequence_length")

        graph, builder = _make_graph()
        inputs_embeds = builder.input(
            "inputs_embeds", dtype=config.dtype, shape=[batch, seq_len, config.hidden_size]
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_sequence_length + sequence_length"],
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        past = _make_kv_cache_inputs(
            builder,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_len,
        )
        logits, hidden_states, present = module(
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
        )
        builder.add_output(logits, "logits")
        builder.add_output(hidden_states, "last_hidden_state")
        _register_kv_cache_outputs(builder, present)
        decoder = _make_model(graph)

        graph, builder = _make_graph(name="embedding")
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])
        embeddings = _LanguageEmbeddingModel(Embedding(config.vocab_size, config.hidden_size))(
            builder.op, input_ids
        )
        builder.add_output(embeddings, "inputs_embeds")
        embedding = _make_model(graph)

        # Semantic frame codes occupy a dedicated 16,384-token range beginning
        # at 151675. Scale this contribution by 1/sqrt(8); the RVQ feedback
        # graph applies the same scale to the sum of seven acoustic embeddings.
        graph, builder = _make_graph(name="semantic_embedding")
        semantic_codes = builder.input(
            "semantic_codes", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        semantic_ids = builder.op.Add(
            semantic_codes, builder.op.Constant(value_int=MINIMAX_MUSIC3_AUDIO_CODE_OFFSET)
        )
        semantic_embedding = _LanguageEmbeddingModel(
            Embedding(config.vocab_size, config.hidden_size)
        )(builder.op, semantic_ids)
        semantic_embedding = builder.op.Mul(semantic_embedding, MINIMAX_MUSIC3_FEEDBACK_SCALE)
        builder.add_output(semantic_embedding, "semantic_feedback_embedding")
        return ModelPackage(
            {
                "model": decoder,
                "embedding": embedding,
                "semantic_embedding": _make_model(graph),
            },
            config=config,
        )


class MiniMaxMusic3RVQTask(ModelTask):
    """Build the local depth decoder and expose its pipeline-owned heads/tables."""

    model_roles: ClassVar[dict[str, str]] = {
        "model": "decoder",
        "projection": "embedding",
        "embedding": "embedding",
        "feedback_embedding": "embedding",
        "heads": "decoder",
    }

    def build(self, module, config) -> ModelPackage:
        graph, builder = _make_graph()
        inputs_embeds = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=["batch", "steps", config.hidden_size],
        )
        hidden_states = module(builder.op, inputs_embeds)
        builder.add_output(hidden_states, "hidden_states")
        decoder = _make_model(graph)

        graph, builder = _make_graph(name="projection")
        projection_input = builder.input(
            "hidden_states",
            dtype=config.dtype,
            shape=["batch", "steps", config.hidden_size],
        )
        builder.add_output(module.projection(builder.op, projection_input), "projected_states")
        projection = _make_model(graph)

        graph, builder = _make_graph(name="embedding")
        code_ids = builder.input("code_ids", dtype=ir.DataType.INT64, shape=["batch", "steps"])
        embedding_module = _AudioEmbeddingModel(
            config.audio_vocab_size * (config.num_codebooks - 1), config.hidden_size
        )
        builder.add_output(embedding_module(builder.op, code_ids), "code_embeddings")
        embedding = _make_model(graph)

        # Complete-frame feedback sums all seven residual-codebook embeddings.
        # Each codebook owns a contiguous audio_vocab_size slice in the table.
        graph, builder = _make_graph(name="feedback_embedding")
        acoustic_codes = builder.input(
            "acoustic_codes",
            dtype=ir.DataType.INT64,
            shape=["batch", "frames", config.num_codebooks - 1],
        )
        offsets = builder.op.Constant(
            value_ints=[
                index * config.audio_vocab_size for index in range(config.num_codebooks - 1)
            ]
        )
        acoustic_ids = builder.op.Add(acoustic_codes, offsets)
        feedback_embedding_module = _AudioEmbeddingModel(
            config.audio_vocab_size * (config.num_codebooks - 1), config.hidden_size
        )
        acoustic_embeddings = feedback_embedding_module(builder.op, acoustic_ids)
        acoustic_embeddings = builder.op.ReduceSum(
            acoustic_embeddings,
            builder.op.Constant(value_ints=[2]),
            keepdims=0,
        )
        acoustic_embeddings = builder.op.Mul(
            acoustic_embeddings, MINIMAX_MUSIC3_FEEDBACK_SCALE
        )
        builder.add_output(acoustic_embeddings, "acoustic_feedback_embedding")
        feedback_embedding = _make_model(graph)

        graph, builder = _make_graph(name="heads")
        head_input = builder.input(
            "hidden_states",
            dtype=config.dtype,
            shape=["batch", "steps", config.hidden_size],
        )
        all_logits = [
            builder.op.Unsqueeze(head(builder.op, head_input), [0])
            for head in module.audio_heads
        ]
        builder.add_output(builder.op.Concat(*all_logits, axis=0), "all_codebook_logits")
        heads = _make_model(graph)

        return ModelPackage(
            {
                "model": decoder,
                "projection": projection,
                "embedding": embedding,
                "feedback_embedding": feedback_embedding,
                "heads": heads,
            },
            config=config,
        )


class MiniMaxMusic3ConditionTask(ModelTask):
    """Mix global-final + seven RVQ-step slices and resample with floor rounding."""

    def build(self, module, config) -> ModelPackage:
        graph, builder = _make_graph()
        hidden_states = builder.input(
            "hidden_states",
            dtype=config.dtype,
            shape=[
                "batch",
                "frames",
                config.num_condition_layers * config.condition_hidden_dim,
            ],
        )
        condition = module(builder.op, hidden_states)
        builder.add_output(condition, "encoder_hidden_states")
        return ModelPackage({"model": _make_model(graph)}, config=config)


class MiniMaxMusic3DenoisingTask(ModelTask):
    """Build the 1D flow-matching velocity predictor."""

    def build(self, module, config) -> ModelPackage:
        graph, builder = _make_graph()
        hidden_states = builder.input(
            "hidden_states",
            dtype=config.dtype,
            shape=["batch", config.in_channels, "latent_length"],
        )
        timestep = builder.input("timestep", dtype=config.dtype, shape=["batch"])
        encoder_hidden_states = builder.input(
            "encoder_hidden_states",
            dtype=config.dtype,
            shape=["batch", "latent_length", config.condition_dim],
        )
        sample = module(builder.op, hidden_states, timestep, encoder_hidden_states)
        builder.add_output(sample, "sample")
        return ModelPackage({"model": _make_model(graph)}, config=config)


class MiniMaxMusic3VocoderTask(ModelTask):
    """Build the stereo latent-to-waveform decoder."""

    model_roles: ClassVar[dict[str, str]] = {"model": "decoder"}

    def build(self, module, config) -> ModelPackage:
        graph, builder = _make_graph()
        latents = builder.input(
            "latents",
            dtype=config.dtype,
            shape=["batch", config.latent_channels, "latent_length"],
        )
        waveform = module(builder.op, latents)
        builder.add_output(waveform, "waveform")
        return ModelPackage({"model": _make_model(graph)}, config=config)
