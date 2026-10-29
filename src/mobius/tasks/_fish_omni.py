# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Drama 3 neural graphs and ONNX sampling policies."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._build_context import ep_capabilities
from mobius._configs.fish_omni import FishOmniConfig
from mobius._model_package import ModelPackage
from mobius.generation._fish_omni import (
    FishAudioSamplingInputs,
    build_fish_codebook_sampler,
    build_fish_main_sampler,
)
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import (
    _make_hybrid_cache_inputs,
    _make_kv_cache_inputs,
    _register_hybrid_cache_outputs,
    _register_kv_cache_outputs,
    _register_linear_attention_functions,
)


class FishOmniTask(ModelTask):
    """Drama 3 frame embedding, hybrid decoder, and temporal codebook tower."""

    model_roles: ClassVar[dict[str, str]] = {
        "embedding": "embedding",
        "decoder": "decoder",
        "audio_decoder": "decoder",
    }
    components: ClassVar[ComponentSpec] = ComponentSpec(
        embedding="embedding",
        decoder="decoder",
        audio_decoder="audio_decoder",
    )
    audio_model_name = "audio_decoder"

    def _sampling_inputs(self, builder) -> FishAudioSamplingInputs | None:
        return None

    def _validate_precision(self, config: FishOmniConfig) -> None:
        if config.dtype == ir.DataType.FLOAT16 and ep_capabilities().name != "cuda":
            raise ValueError(
                "Drama 3 FP16 currently requires execution_provider='cuda' "
                "for the native FP16-input/FP32-output projection"
            )
        if config.native_fp8_experts and ep_capabilities().name != "cuda":
            raise ValueError(
                "Drama 3 native FP8 requires execution_provider='cuda'; "
                "use --dequantize for portable FP32."
            )
        if config.native_fp8_experts and config.dtype == ir.DataType.FLOAT:
            raise ValueError(
                "Drama 3 native FP8 export is disabled pending real-weight numerical "
                "qualification; use --dequantize for the qualified dense FP32 route."
            )
        if (
            config.dtype == ir.DataType.BFLOAT16
            and not config.native_fp8_experts
            and ep_capabilities().name != "onnx-standard"
        ):
            raise ValueError(
                "Drama 3 BF16 currently requires execution_provider='onnx-standard' "
                "and a runtime with BF16 CUDA Attention support"
            )
        if config.dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16):
            raise ValueError(
                "Drama 3 reduced-precision export is disabled pending real-weight numerical "
                "qualification. Production-dimension FP16 and BF16 reference probes exceed "
                "the unchanged 1e-2 logit gate; use FP32."
            )

    def build(self, module, config: FishOmniConfig) -> ModelPackage:
        self._validate_precision(config)
        self._validate_components(module)
        assert config.tower is not None
        batch, seq = "batch", "sequence_len"
        past = ir.SymbolicDim("past_sequence_len")
        models = {}
        graph, builder = _make_graph("embedding")
        frames = builder.input(
            "frames", dtype=ir.DataType.INT64, shape=[batch, seq, config.num_codebooks + 1]
        )
        embeds, unseen = module.embedding(builder.op, frames)
        builder.add_output(embeds, "inputs_embeds")
        builder.add_output(unseen, "shared_input")
        models["embedding"] = _make_model(graph)

        graph, builder = _make_graph("decoder")
        embeds = builder.input(
            "inputs_embeds", dtype=config.dtype, shape=[batch, seq, config.hidden_size]
        )
        mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, "total_sequence_len"]
        )
        positions = builder.input("position_ids", dtype=ir.DataType.INT64, shape=[batch, seq])
        decoder_caches = _make_hybrid_cache_inputs(
            builder, config, config.dtype, ir.SymbolicDim(batch), past
        )
        logits, hidden, present = module.decoder(
            builder.op, embeds, mask, positions, decoder_caches
        )
        builder.add_output(logits, "logits")
        builder.add_output(hidden, "hidden_states")
        _register_hybrid_cache_outputs(builder, present, config.layer_types or [])
        models["decoder"] = _make_model(graph)
        _register_linear_attention_functions(models["decoder"], config)

        graph, builder = _make_graph(self.audio_model_name)
        hidden = builder.input(
            "hidden_states", dtype=config.dtype, shape=[batch, seq, config.hidden_size]
        )
        codes = builder.input(
            "codes", dtype=ir.DataType.INT64, shape=[batch, seq, config.num_codebooks]
        )
        semantic = builder.input("semantic_mask", dtype=ir.DataType.BOOL, shape=[batch, seq])
        unseen = builder.input(
            "shared_input", dtype=config.dtype, shape=[batch, seq, config.hidden_size]
        )
        mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, "total_sequence_len"]
        )
        positions = builder.input("position_ids", dtype=ir.DataType.INT64, shape=[batch, seq])
        tower = config.tower
        tower_caches = _make_kv_cache_inputs(
            builder,
            tower.num_hidden_layers,
            tower.num_key_value_heads,
            tower.head_dim,
            config.dtype,
            ir.SymbolicDim(batch),
            past,
        )
        sampling = self._sampling_inputs(builder)
        if sampling is None:
            logits, present = module.audio_decoder(
                builder.op, hidden, codes, semantic, unseen, mask, positions, tower_caches
            )
        else:
            # A second method must retain __call__'s module scope without
            # cloning any tower weights or placing parameters in subgraphs.
            decoder = module.audio_decoder
            builder.push_module(decoder.name or "audio_decoder", type(decoder).__qualname__)
            try:
                logits, present, frame, counter, valid = decoder.forward_sampled(
                    builder.op,
                    hidden,
                    codes,
                    semantic,
                    unseen,
                    mask,
                    positions,
                    tower_caches,
                    sampling,
                )
            finally:
                builder.pop_module()
            frame.type = ir.TensorType(ir.DataType.INT64)
            frame.shape = ir.Shape([batch, 1, config.num_codebooks])
            counter.type, counter.shape = ir.TensorType(ir.DataType.INT64), ir.Shape([batch])
            valid.type, valid.shape = ir.TensorType(ir.DataType.BOOL), ir.Shape([batch])
            builder.add_output(frame, "frame_codes")
            builder.add_output(counter, "next_counter")
            builder.add_output(valid, "valid")
        builder.add_output(logits, "codebook_logits")
        _register_kv_cache_outputs(builder, present)
        models[self.audio_model_name] = _make_model(graph)
        for model in models.values():
            model.metadata_props["mobius.fish_omni.scope"] = (
                "Neural token graphs with ONNX sampling policies; caller orchestrates "
                "temporal decoding and prompt formatting; waveform decoder exported separately"
            )
        if config.native_fp8_experts:
            models["decoder"].metadata_props["mobius.required_runtime_features"] = (
                "CUDA SM89+; fp8-qmoe=1; QMoE activation_quantization=dynamic_fp8"
            )
            assert config.num_experts_per_tok is not None
            models["decoder"].metadata_props["mobius.native_fp8.max_tokens_per_call"] = str(
                65535 // config.num_experts_per_tok
            )
        return ModelPackage(
            models,
            config=config,
            policy_components={
                "main_sampler": build_fish_main_sampler(config),
                "codebook_sampler": build_fish_codebook_sampler(config),
            },
        )


class FishOmniGenerationTask(FishOmniTask):
    """Single-pass temporal codebook generation with shared teacher-forced prefill."""

    audio_model_name = "audio_generator"
    model_roles: ClassVar[dict[str, str]] = {
        "embedding": "embedding",
        "decoder": "decoder",
        "audio_generator": "decoder",
    }
    components: ClassVar[ComponentSpec] = ComponentSpec(
        embedding="embedding",
        decoder="decoder",
        audio_generator="audio_decoder",
    )

    def _sampling_inputs(self, builder) -> FishAudioSamplingInputs:
        return FishAudioSamplingInputs(
            builder.input("generate", ir.DataType.BOOL, []),
            builder.input("do_sample", ir.DataType.BOOL, []),
            builder.input("temperature", ir.DataType.FLOAT, ["batch"]),
            builder.input("top_k", ir.DataType.INT64, ["batch"]),
            builder.input("top_p", ir.DataType.FLOAT, ["batch"]),
            builder.input("seed", ir.DataType.INT64, ["batch"]),
            builder.input("counter", ir.DataType.INT64, ["batch"]),
        )
