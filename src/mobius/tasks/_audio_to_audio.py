# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio-to-audio task for end-to-end speech models.

Builds separate ONNX models for audio-to-audio pipelines like
LFM2-Audio and Moshi/PersonaPlex. These models take audio in and
produce audio out, with an intermediate language model backbone.

Typical model split:
1. **audio_encoder**: mel/waveform -> audio features (Conformer/encoder)
2. **embedding**: text + audio token fusion -> inputs_embeds
3. **decoder**: inputs_embeds -> logits + KV cache (hybrid conv+attention LM)
4. **audio_decoder**: backbone hidden -> per-codebook logits (depthformer)
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_hybrid_cache_inputs,
    _make_kv_cache_inputs,
    _make_model,
    _register_hybrid_cache_outputs,
    _register_kv_cache_outputs,
)


class AudioToAudioTask(ModelTask):
    """Multi-model split for audio-to-audio models.

    The module must provide sub-modules as attributes:

    - ``audio_encoder``: audio encoder taking mel/waveform input
    - ``embedding``: embedding model fusing text + audio features
    - ``decoder``: language model backbone with KV cache
    - ``audio_decoder`` (optional): depthformer or codec decoder

    Each sub-module is wired into its own ONNX graph.
    Supports both standard KV cache and hybrid (conv+attention) cache
    for the decoder, selected automatically based on ``config.layer_types``.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        models: dict[str, ir.Model] = {}

        models["audio_encoder"] = self._build_audio_encoder(module.audio_encoder, config)
        models["embedding"] = self._build_embedding(module.embedding, config)
        models["decoder"] = self._build_decoder(module.decoder, config)

        if hasattr(module, "audio_decoder"):
            models["audio_decoder"] = self._build_audio_decoder(module.audio_decoder, config)

        return ModelPackage(models, config=config)

    def _build_audio_encoder(
        self,
        audio_encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build audio encoder: mel (batch, n_mels, time) -> audio features."""
        batch = ir.SymbolicDim("batch")
        mel_seq = ir.SymbolicDim("mel_sequence_len")
        n_mels = config.audio.num_mel_bins or 128 if config.audio else 128

        input_features = ir.Value(
            name="input_features",
            shape=ir.Shape([batch, n_mels, mel_seq]),
            type=ir.TensorType(config.dtype),
        )

        graph, builder = _make_graph([input_features], name="audio_encoder")
        audio_features = audio_encoder(builder.op, input_features)

        audio_features.name = "audio_features"
        graph.outputs.append(audio_features)
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build embedding: text_ids -> inputs_embeds."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph([input_ids], name="embedding")
        inputs_embeds = embedding(
            builder.op,
            input_ids=input_ids,
        )

        inputs_embeds.name = "inputs_embeds"
        graph.outputs.append(inputs_embeds)
        return _make_model(graph)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build decoder: inputs_embeds -> logits + cache.

        Automatically uses hybrid cache (conv+attention) when
        ``config.layer_types`` is set, otherwise standard KV cache.
        """
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        inputs_embeds = ir.Value(
            name="inputs_embeds",
            shape=ir.Shape([batch, seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [inputs_embeds, attention_mask, position_ids]

        # Select cache type based on config
        use_hybrid = config.layer_types is not None and any(
            lt != "full_attention" for lt in config.layer_types
        )
        if use_hybrid:
            cache_inputs, past_key_values = _make_hybrid_cache_inputs(
                config, config.dtype, batch, past_seq_len
            )
        else:
            cache_inputs, past_key_values = _make_kv_cache_inputs(
                config.num_hidden_layers,
                config.num_key_value_heads,
                config.head_dim,
                config.dtype,
                batch,
                past_seq_len,
            )
        graph_inputs.extend(cache_inputs)

        graph, builder = _make_graph(graph_inputs, name="decoder")
        logits, present_key_values = decoder(
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        if use_hybrid:
            _register_hybrid_cache_outputs(
                graph,
                present_key_values,
                config.layer_types or [],
            )
        else:
            _register_kv_cache_outputs(graph, present_key_values)
        return _make_model(graph)

    def _build_audio_decoder(
        self,
        audio_decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build audio decoder: backbone hidden -> per-codebook logits.

        The audio decoder (depthformer) takes a backbone hidden state and
        produces logits for one codebook at a time. Runtime handles the
        autoregressive loop over codebooks.

        Inputs:
            backbone_hidden: (batch, 1, hidden_size) - LM output embedding
            prev_embedding: (batch, 1, depthformer_dim) - previous codebook
            codebook_idx: scalar int64 - which codebook to predict
            past KV cache for depthformer layers

        Outputs:
            codebook_logits: (batch, 1, audio_vocab_size)
            present KV cache
        """
        depthformer_dim = getattr(config, "depthformer_dim", config.hidden_size)
        depthformer_layers = getattr(config, "depthformer_layers", 6)
        depthformer_heads = getattr(config, "depthformer_heads", 16)
        depthformer_head_dim = depthformer_dim // depthformer_heads

        batch = ir.SymbolicDim("batch")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        backbone_hidden = ir.Value(
            name="backbone_hidden",
            shape=ir.Shape([batch, 1, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        prev_embedding = ir.Value(
            name="prev_embedding",
            shape=ir.Shape([batch, 1, depthformer_dim]),
            type=ir.TensorType(config.dtype),
        )
        codebook_idx = ir.Value(
            name="codebook_idx",
            shape=ir.Shape([]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [backbone_hidden, prev_embedding, codebook_idx]

        # Depthformer KV cache (all attention layers)
        kv_inputs, past_key_values = _make_kv_cache_inputs(
            depthformer_layers,
            depthformer_heads,
            depthformer_head_dim,
            config.dtype,
            batch,
            past_seq_len,
            prefix="past_key_values",
        )
        graph_inputs.extend(kv_inputs)

        graph, builder = _make_graph(graph_inputs, name="audio_decoder")
        codebook_logits, present_kv = audio_decoder(
            builder.op,
            backbone_hidden=backbone_hidden,
            prev_embedding=prev_embedding,
            codebook_idx=codebook_idx,
            past_key_values=past_key_values,
        )

        codebook_logits.name = "codebook_logits"
        graph.outputs.append(codebook_logits)
        _register_kv_cache_outputs(graph, present_kv)
        return _make_model(graph)
