# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""TTS multi-model split task for Qwen3-TTS.

Builds separate ONNX models (five required + one optional speaker encoder):
1. **talker**: inputs_embeds → logits (first code group) + last_hidden_state + KV cache
2. **code_predictor**: inputs_embeds → hidden_states + KV cache (1D RoPE)
3. **embedding**: text_ids + codec_ids → text_embeds + codec_embeds
4. **talker_step_embedder**: frame_codes + text_embed → inputs_embeds
5. **talker_prefill_embedder**: text_ids → prefill_embeds + trailing_text_embeds
6. **speaker_encoder**: mel_input → speaker_embedding
7. **code_predictor_prefill**: talker hidden + group-0 embedding → predictor input
8. **code_predictor_step_embedder**: prior code + predictor tables → next input
9. **talker_text_step**: trailing text embeddings + loop index → one text embedding
10. **code_predictor_indices**: inner induction → embedding/head/frame indices

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
    """Multi-model split for Qwen3-TTS.

    The module must provide five required sub-modules and one optional:

    - ``talker``: Decoder producing logits + last_hidden_state
    - ``code_predictor``: Small decoder for remaining code groups
    - ``embedding``: Text + codec embedding model
    - ``talker_step_embedder``: Per-frame ``frame_codes [+ text_embed] ->
      inputs_embeds`` pre-embedder for the talker
    - ``talker_prefill_embedder``: ``text_ids -> prefill_embeds +
      trailing_text_embeds`` prefill/trailing-text builder
    - ``speaker_encoder``: ECAPA-TDNN speaker encoder *(optional — omitted
      when the model uses a pre-computed speaker embedding instead)*

    Each sub-module is wired into its own ONNX graph.
    """

    # Every key the package produces must appear here: ``model_roles`` is what
    # ``inspect_components`` reports and what ``build_from_module`` uses to pick
    # optimization passes. An undeclared component silently falls back to the
    # ``"decoder"`` role and would be handed GQA / QKV-packing fusion.
    model_roles: ClassVar[dict[str, str]] = {
        "talker": "decoder",
        "code_predictor": "decoder",
        "embedding": "embedding",
        "talker_step_embedder": "embedding",
        "talker_prefill_embedder": "embedding",
        "speaker_encoder": "encoder",
        # Parameter-free graphs that wire the generation loop. They read every
        # tensor they use from their own graph inputs, so they carry no weights
        # and no fusion pass applies to them.
        "code_predictor_prefill": "glue",
        "code_predictor_step_embedder": "glue",
        "code_predictor_indices": "glue",
        "talker_text_step": "glue",
    }
    components: ClassVar[ComponentSpec] = ComponentSpec(
        talker="talker",
        code_predictor="code_predictor",
        embedding="embedding",
        talker_step_embedder="talker_step_embedder",
        talker_prefill_embedder="talker_prefill_embedder",
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
        models["talker_step_embedder"] = self._build_talker_step_embedder(
            module.talker_step_embedder, config
        )
        models["talker_prefill_embedder"] = self._build_talker_prefill_embedder(
            module.talker_prefill_embedder, config
        )
        models["code_predictor_prefill"] = self._build_code_predictor_prefill(config)
        models["code_predictor_step_embedder"] = self._build_code_predictor_step_embedder(
            config
        )
        models["code_predictor_indices"] = self._build_code_predictor_indices()
        models["talker_text_step"] = self._build_talker_text_step(config)
        if module.speaker_encoder is not None:
            models["speaker_encoder"] = self._build_speaker_encoder(
                module.speaker_encoder, config
            )

        return ModelPackage(models, config=config)

    def _build_code_predictor_prefill(self, config: ArchitectureConfig) -> ir.Model:
        """Build the trained group-0 transition input for code-predictor prefill."""
        graph, builder = _make_graph(name="code_predictor_prefill")
        talker_hidden = builder.input(
            "talker_hidden",
            dtype=config.dtype,
            shape=["batch", 1, config.hidden_size],
        )
        group_0_embed = builder.input(
            "group_0_embed",
            dtype=config.dtype,
            shape=["batch", 1, config.hidden_size],
        )
        inputs_embeds = builder.op.Concat(talker_hidden, group_0_embed, axis=1)
        inputs_embeds.shape = ir.Shape(["batch", 2, config.hidden_size])
        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)

    def _build_code_predictor_step_embedder(self, config: ArchitectureConfig) -> ir.Model:
        """Gather the trained predictor embedding for the previously sampled code."""
        tts = config.tts
        cp = tts.code_predictor if tts else None
        num_groups = tts.num_code_groups if tts else 16
        cp_vocab = cp.vocab_size if cp else 2048
        graph, builder = _make_graph(name="code_predictor_step_embedder")
        codec_embeddings = builder.input(
            "codec_embeddings",
            dtype=config.dtype,
            shape=[num_groups - 1, cp_vocab, config.hidden_size],
        )
        token = builder.input("token", dtype=ir.DataType.INT64, shape=["batch"])
        embedding_index = builder.input("embedding_index", dtype=ir.DataType.INT64, shape=[])
        table = builder.op.Gather(codec_embeddings, embedding_index, axis=0)
        inputs_embeds = builder.op.Gather(table, token, axis=0)
        inputs_embeds = builder.op.Unsqueeze(inputs_embeds, [1])
        inputs_embeds.shape = ir.Shape(["batch", 1, config.hidden_size])
        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)

    def _build_talker_text_step(self, config: ArchitectureConfig) -> ir.Model:
        """Select one trailing-text embedding for the outer talker iteration."""
        graph, builder = _make_graph(name="talker_text_step")
        trailing = builder.input(
            "trailing_text_embeds",
            dtype=config.dtype,
            shape=["batch", "trailing_sequence", config.hidden_size],
        )
        iteration = builder.input("iteration", dtype=ir.DataType.INT64, shape=["batch"])
        sequence_length = builder.op.Shape(trailing, start=1, end=2)
        index = builder.op.Min(
            builder.op.Gather(iteration, 0, axis=0),
            builder.op.Sub(
                builder.op.Squeeze(sequence_length, [0]),
                builder.op.Constant(value_int=1),
            ),
        )
        text_embed = builder.op.Gather(trailing, index, axis=1)
        text_embed = builder.op.Unsqueeze(text_embed, [1])
        text_embed.shape = ir.Shape(["batch", 1, config.hidden_size])
        builder.add_output(text_embed, "text_embed")
        return _make_model(graph)

    def _build_code_predictor_indices(self) -> ir.Model:
        """Derive predictor/table/frame indices from the zero-based inner loop."""
        graph, builder = _make_graph(name="code_predictor_indices")
        iteration = builder.input("iteration", dtype=ir.DataType.INT64, shape=[])
        step_index = builder.op.Add(iteration, builder.op.Constant(value_int=1))
        frame_index = builder.op.Add(iteration, builder.op.Constant(value_int=2))
        builder.add_output(builder.op.Identity(iteration), "embedding_index")
        builder.add_output(step_index, "step_index")
        builder.add_output(frame_index, "frame_index")
        return _make_model(graph)

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
        for present in present_key_values:
            present[0].shape = ir.Shape(
                [batch, config.num_key_value_heads, "total_sequence_len", config.head_dim]
            )
            present[1].shape = ir.Shape(
                [batch, config.num_key_value_heads, "total_sequence_len", config.head_dim]
            )

        builder.add_output(logits, "logits")
        builder.add_output(last_hidden_state, "last_hidden_state")
        _register_kv_cache_outputs(builder, present_key_values)
        return _make_model(graph)

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
        for present in present_key_values:
            present[0].shape = ir.Shape(
                [batch, cp_num_key_value_heads, "total_sequence_len", cp_head_dim]
            )
            present[1].shape = ir.Shape(
                [batch, cp_num_key_value_heads, "total_sequence_len", cp_head_dim]
            )

        builder.add_output(logits, "logits")
        # Expose stacked codec embeddings for generation loop to extract.
        # The Identity node ensures renaming the output doesn't affect
        # the initializer name used for weight loading.
        builder.add_output(codec_embeddings, "codec_embeddings")
        _register_kv_cache_outputs(builder, present_key_values)
        return _make_model(graph)

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
        return _make_model(graph)

    def _build_talker_step_embedder(
        self,
        talker_step_embedder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build talker step embedder: frame_codes + text_embed → inputs_embeds.

        Materializes the per-step talker ``codec_sum`` construction in-graph:
        ``codec_sum = talker.codec_embed(code_0) + Σ cp_codec_weights[i][code]``
        and returns ``codec_sum + text_embed``. Lets a generic runtime loop
        drive the talker from the previous frame's raw integer codes instead
        of a pre-built ``inputs_embeds``.
        """
        tts = config.tts
        num_code_groups = tts.num_code_groups if tts else 16

        batch = ir.SymbolicDim("batch")

        graph, builder = _make_graph(name="talker_step_embedder")

        frame_codes = builder.input(
            "frame_codes",
            dtype=ir.DataType.INT64,
            shape=[batch, num_code_groups],
        )
        text_embed = builder.input(
            "text_embed",
            dtype=config.dtype,
            shape=[batch, 1, config.hidden_size],
        )

        inputs_embeds = talker_step_embedder(
            builder.op,
            frame_codes=frame_codes,
            text_embed=text_embed,
        )

        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)

    def _build_talker_prefill_embedder(
        self,
        talker_prefill_embedder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build talker prefill embedder: text_ids → prefill + trailing embeds.

        Materializes the up-front PREFILL and trailing-text embedding
        construction in-graph (Auto language, no speaker, no instruct):

        - ``prefill_embeds`` = role(3) + codec_text_pairs(N-1) + first_text_codec(1),
          length ``3 + 4 + 1 = 8`` (constant, independent of text length).
        - ``trailing_text_embeds`` = text_embeds[:, 4:-5] ++ tts_eos,
          length ``text_len - 8``.

        Reuses the embedding model's text + codec tables (shared weights). Lets
        a generic runtime loop drive the talker without Qwen3-TTS-specific
        slicing/interleaving logic.
        """
        batch = ir.SymbolicDim("batch")
        text_seq = ir.SymbolicDim("text_sequence_len")

        graph, builder = _make_graph(name="talker_prefill_embedder")

        text_ids = builder.input(
            "text_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, text_seq],
        )

        prefill_embeds, trailing_text_embeds = talker_prefill_embedder(
            builder.op,
            text_ids=text_ids,
        )
        prefill_embeds.shape = ir.Shape([batch, "prefill_sequence_len", config.hidden_size])
        trailing_text_embeds.shape = ir.Shape(
            [batch, "trailing_sequence_len", config.hidden_size]
        )

        builder.add_output(prefill_embeds, "prefill_embeds")
        builder.add_output(trailing_text_embeds, "trailing_text_embeds")
        return _make_model(graph)

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
        return _make_model(graph)
