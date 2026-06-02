# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._cache_utils import (
    _make_hybrid_cache_inputs,
    _make_kv_cache_inputs,
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

    model_roles: ClassVar[dict[str, str]] = {
        "audio_encoder": "encoder",
        "embedding": "embedding",
        "decoder": "decoder",
        "audio_decoder": "decoder",
    }

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
        n_mels = (config.audio.num_mel_bins or 128) if config.audio else 128

        graph, builder = _make_graph(name="audio_encoder")
        input_features = builder.input(
            "input_features",
            dtype=config.dtype,
            shape=[batch, n_mels, mel_seq],
        )
        audio_features = audio_encoder(builder.op, input_features)
        builder.add_output(audio_features, "audio_features")
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build embedding: text_ids -> inputs_embeds."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        graph, builder = _make_graph(name="embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        inputs_embeds = embedding(
            builder.op,
            input_ids=input_ids,
        )
        builder.add_output(inputs_embeds, "inputs_embeds")
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

        graph, builder = _make_graph(name="decoder")
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
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )

        # Select cache type based on config
        use_hybrid = config.layer_types is not None and any(
            lt != "full_attention" for lt in config.layer_types
        )
        if use_hybrid:
            past_key_values = _make_hybrid_cache_inputs(
                builder, config, config.dtype, batch, past_seq_len
            )
        else:
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
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        builder.add_output(logits, "logits")
        if use_hybrid:
            _register_hybrid_cache_outputs(
                builder,
                present_key_values,
                config.layer_types or [],
            )
        else:
            _register_kv_cache_outputs(builder, present_key_values)
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

        graph, builder = _make_graph(name="audio_decoder")
        backbone_hidden = builder.input(
            "backbone_hidden",
            dtype=config.dtype,
            shape=[batch, 1, config.hidden_size],
        )
        prev_embedding = builder.input(
            "prev_embedding",
            dtype=config.dtype,
            shape=[batch, 1, depthformer_dim],
        )
        codebook_idx = builder.input(
            "codebook_idx",
            dtype=ir.DataType.INT64,
            shape=[],
        )

        # Depthformer KV cache (all attention layers)
        past_key_values = _make_kv_cache_inputs(
            builder,
            depthformer_layers,
            depthformer_heads,
            depthformer_head_dim,
            config.dtype,
            batch,
            past_seq_len,
            prefix="past_key_values",
        )

        codebook_logits, present_kv = audio_decoder(
            builder.op,
            backbone_hidden=backbone_hidden,
            prev_embedding=prev_embedding,
            codebook_idx=codebook_idx,
            past_key_values=past_key_values,
        )

        builder.add_output(codebook_logits, "codebook_logits")
        _register_kv_cache_outputs(builder, present_kv)
        return _make_model(graph)


class MoshiTask(AudioToAudioTask):
    """Multi-model split for Moshi/PersonaPlex audio-to-audio models.

    Differs from :class:`AudioToAudioTask`:

    - No ``audio_encoder``: Moshi consumes audio as codec token IDs,
      not mel spectrograms, so no waveform encoder is needed.
    - ``embedding`` accepts both ``input_ids`` (text) and ``audio_codes``
      (shape: batch x seq x num_codebooks) and returns ``inputs_embeds``.
    - ``audio_decoder`` KV cache is sized with ``head_dim = depformer_dim``
      (one full-dimensioned head per codebook) rather than
      ``depformer_dim // depformer_num_heads``.
    """

    model_roles: ClassVar[dict[str, str]] = {
        "embedding": "embedding",
        "decoder": "decoder",
        "audio_decoder": "decoder",
    }

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        models: dict[str, ir.Model] = {}

        # Moshi has no audio encoder — audio input is codec token IDs.
        models["embedding"] = self._build_embedding(module.embedding, config)
        models["decoder"] = self._build_decoder(module.decoder, config)

        if hasattr(module, "audio_decoder"):
            models["audio_decoder"] = self._build_audio_decoder(module.audio_decoder, config)

        return ModelPackage(models, config=config)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build Moshi embedding: (text_ids, audio_codes) -> inputs_embeds."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_codebooks = getattr(config, "num_codebooks", 16)

        graph, builder = _make_graph(name="embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        audio_codes = builder.input(
            "audio_codes",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len, num_codebooks],
        )
        inputs_embeds = embedding(builder.op, input_ids=input_ids, audio_codes=audio_codes)
        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)

    def _build_audio_decoder(
        self,
        audio_decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build Moshi depformer: backbone_hidden -> codebook_logits.

        Uses ``head_dim = depformer_dim`` (one full-size head per codebook),
        matching PersonaPlex's packed-QKV depformer attention weights.
        """
        depformer_dim = getattr(config, "depformer_dim", 1024)
        depformer_layers = getattr(config, "depformer_layers", 6)
        # Each head in the depformer covers one full depformer dimension.
        depformer_heads = getattr(
            config, "depformer_num_heads", getattr(config, "num_codebooks", 16)
        )
        depformer_head_dim = depformer_dim  # full head_dim = depformer_dim

        batch = ir.SymbolicDim("batch")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph(name="audio_decoder")
        backbone_hidden = builder.input(
            "backbone_hidden",
            dtype=config.dtype,
            shape=[batch, 1, config.hidden_size],
        )
        prev_embedding = builder.input(
            "prev_embedding",
            dtype=config.dtype,
            shape=[batch, 1, depformer_dim],
        )
        codebook_idx = builder.input(
            "codebook_idx",
            dtype=ir.DataType.INT64,
            shape=[],
        )

        past_key_values = _make_kv_cache_inputs(
            builder,
            depformer_layers,
            depformer_heads,
            depformer_head_dim,
            config.dtype,
            batch,
            past_seq_len,
            prefix="past_key_values",
        )

        codebook_logits, present_kv = audio_decoder(
            builder.op,
            backbone_hidden=backbone_hidden,
            prev_embedding=prev_embedding,
            codebook_idx=codebook_idx,
            past_key_values=past_key_values,
        )

        builder.add_output(codebook_logits, "codebook_logits")
        _register_kv_cache_outputs(builder, present_kv)
        return _make_model(graph)
