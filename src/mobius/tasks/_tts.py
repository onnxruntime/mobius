# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""TTS 4-model split task for Qwen3-TTS.

Builds four separate ONNX models:
1. **talker**: inputs_embeds → logits (first code group) + last_hidden_state + KV cache
2. **code_predictor**: inputs_embeds → hidden_states + KV cache (1D RoPE)
3. **embedding**: text_ids + codec_ids → text_embeds + codec_embeds
4. **speaker_encoder**: mel_input → speaker_embedding

Used by Qwen3TTSForConditionalGeneration.
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


class TTSTask(ModelTask):
    """4-model split for Qwen3-TTS.

    The module must provide three required sub-modules and one optional:

    - ``talker``: Decoder producing logits + last_hidden_state
    - ``code_predictor``: Small decoder for remaining code groups
    - ``embedding``: Text + codec embedding model
    - ``speaker_encoder``: ECAPA-TDNN speaker encoder *(optional — omitted
      when the model uses a pre-computed speaker embedding instead)*

    Each sub-module is wired into its own ONNX graph.
    """

    model_roles: ClassVar[dict[str, str]] = {
        "talker": "decoder",
        "code_predictor": "decoder",
        "embedding": "embedding",
        "speaker_encoder": "encoder",
    }
    components: ClassVar[ComponentSpec] = ComponentSpec(
        talker="talker",
        code_predictor="code_predictor",
        embedding="embedding",
    )

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {}

        models["talker"] = self._build_talker(module.talker, config)
        models["code_predictor"] = self._build_code_predictor(module.code_predictor, config)
        models["embedding"] = self._build_embedding(module.embedding, config)
        if module.speaker_encoder is not None:
            models["speaker_encoder"] = self._build_speaker_encoder(
                module.speaker_encoder, config
            )

        return ModelPackage(models, config=config)

    def _build_talker(
        self,
        talker: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build talker: inputs_embeds → logits + last_hidden_state + KV cache.

        Uses MRoPE 3D position_ids (3, batch, seq_len).
        """
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph(name="talker")

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
        # MRoPE: 3D position_ids (3, batch, seq_len)
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

        logits, last_hidden_state, present_key_values = talker(
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        builder.add_output(logits, "logits")
        builder.add_output(last_hidden_state, "last_hidden_state")
        _register_kv_cache_outputs(builder, present_key_values)
        return _make_model(graph, builder.functions.values())

    def _build_code_predictor(
        self,
        code_predictor: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build code predictor: inputs_embeds → logits + KV cache.

        The generation loop constructs inputs_embeds in **talker_hidden**
        space (e.g. 2048 for 1.7B, 1024 for 0.6B):
          - Step 0 (prefill): concat(talker_hidden, talker_embed(code_0))
            → 2 tokens. Matches HF's code predictor prefill.
          - Steps 1-14: CP_embed[step-1](code_i) → 1 token.
            CP codec embeddings are stored in talker_hidden space.

        The model projects to cp_hidden internally via
        ``small_to_mtp_projection`` (Identity when dims match).

        Uses standard 1D RoPE (2D position_ids).
        """
        # Read code predictor config directly to avoid importing the model class.
        # Defaults match Qwen3TTSCodePredictorModel._make_cp_config().
        tts = config.tts
        cp = tts.code_predictor if tts else None
        cp_num_hidden_layers = cp.num_hidden_layers if cp else 5
        cp_num_key_value_heads = cp.num_key_value_heads if cp else 8
        cp_head_dim = cp.head_dim if cp else 128

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph(name="code_predictor")

        # Pre-embedded input in talker_hidden space (constructed by
        # generation loop). The model projects to cp_hidden internally.
        inputs_embeds = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=[batch, seq_len, config.hidden_size],
        )
        # Step index: selects which lm_head to use (0..14)
        step_index = builder.input(
            "step_index",
            dtype=ir.DataType.INT64,
            shape=[],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_seq_len + seq_len"],
        )
        # 1D RoPE: 2D position_ids (batch, seq_len)
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )

        past_key_values = _make_kv_cache_inputs(
            builder,
            cp_num_hidden_layers,
            cp_num_key_value_heads,
            cp_head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )

        logits, present_key_values, codec_embeddings = code_predictor(
            builder.op,
            inputs_embeds=inputs_embeds,
            step_index=step_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        builder.add_output(logits, "logits")
        # Expose stacked codec embeddings for generation loop to extract.
        # The Identity node ensures renaming the output doesn't affect
        # the initializer name used for weight loading.
        builder.add_output(codec_embeddings, "codec_embeddings")
        _register_kv_cache_outputs(builder, present_key_values)
        return _make_model(graph, builder.functions.values())

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build embedding: text_ids + codec_ids → text_embeds + codec_embeds."""
        batch = ir.SymbolicDim("batch")
        text_seq = ir.SymbolicDim("text_sequence_len")
        codec_seq = ir.SymbolicDim("codec_sequence_len")

        graph, builder = _make_graph(name="embedding")

        text_ids = builder.input(
            "text_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, text_seq],
        )
        codec_ids = builder.input(
            "codec_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, codec_seq],
        )

        text_embeds, codec_embeds = embedding(
            builder.op,
            text_ids=text_ids,
            codec_ids=codec_ids,
        )

        builder.add_output(text_embeds, "text_embeds")
        builder.add_output(codec_embeds, "codec_embeds")
        return _make_model(graph, builder.functions.values())

    def _build_speaker_encoder(
        self,
        speaker_encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build speaker encoder: mel_input → speaker_embedding."""
        batch = ir.SymbolicDim("batch")
        mel_seq = ir.SymbolicDim("mel_sequence_len")
        tts = config.tts
        se = tts.speaker_encoder if tts else None
        mel_dim = se.mel_dim if se else 128

        graph, builder = _make_graph(name="speaker_encoder")

        mel_input = builder.input(
            "mel_input",
            dtype=config.dtype,
            shape=[batch, mel_seq, mel_dim],
        )

        speaker_embedding = speaker_encoder(builder.op, mel_input)

        builder.add_output(speaker_embedding, "speaker_embedding")
        return _make_model(graph, builder.functions.values())
