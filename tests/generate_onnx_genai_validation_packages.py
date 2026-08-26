from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import yaml
from onnxscript import GraphBuilder
from safetensors.numpy import save_file

from mobius import build_from_module
from mobius._model_package import ModelPackage
from mobius._passes import RemoveDeadGraphInputsPass
from mobius.adapter_io import load_peft_adapter
from mobius.adapters import (
    AdapterArtifact,
    AdapterServiceOptions,
    AdapterTarget,
    AdapterTargetDescriptor,
    AdapterTargetManifest,
    AdapterWeights,
    fingerprint_model_weights,
)
from mobius.integrations.diffusers._configs import (
    MiniMaxMusic3ConditionConfig,
    MiniMaxMusic3LanguageConfig,
    MiniMaxMusic3RVQConfig,
    MiniMaxMusic3TransformerConfig,
    MiniMaxMusic3VocoderConfig,
    MiniMaxMusic3WorkflowConfig,
)
from mobius.integrations.onnx_genai import write_onnx_genai_config
from mobius.integrations.onnx_genai.auto_export_test import (
    _Cfg,
    _VlmCfg,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    add_adapter_service_to_metadata,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    write_audio_codec_workflow_metadata,
    write_diffusion_workflow_metadata,
    write_encoder_embedding_workflow_metadata,
    write_hierarchical_audio_workflow_metadata,
    write_language_diffusion_workflow_metadata,
    write_speculative_workflow_metadata,
    write_video_diffusion_workflow_metadata,
)
from mobius.models.bert import BertModel
from mobius.models.bert_test import PROTBERT_TINY_CONFIG
from mobius.models.esm import EsmModel
from mobius.models.esm_test import TINY_CONFIG as ESM2_TINY_CONFIG
from mobius.models.minimax_music3 import (
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3LanguageModel,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)
from mobius.models.qwen3_tts import Qwen3TTSForConditionalGeneration
from mobius.models.qwen3_tts_test import _TINY_CONFIG
from mobius.tasks import FeatureExtractionTask, TTSTask


def _hierarchical_audio_package() -> ModelPackage:
    language_config = MiniMaxMusic3LanguageConfig.from_diffusers(
        {
            "model_type": "qwen3",
            "vocab_size": 168060,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
            "tie_word_embeddings": False,
            "hidden_act": "silu",
        }
    )
    language = build_from_module(
        MiniMaxMusic3LanguageModel(language_config),
        language_config,
        "minimax-music3-language",
    )
    rvq_config = MiniMaxMusic3RVQConfig(
        hidden_size=8,
        num_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        audio_vocab_size=1024,
        num_codebooks=8,
        max_position_embeddings=8,
    )
    rvq = build_from_module(
        MiniMaxMusic3RVQDepthDecoder(rvq_config),
        rvq_config,
        "minimax-music3-rvq",
    )
    condition_config = MiniMaxMusic3ConditionConfig(
        condition_hidden_dim=8,
        num_condition_layers=8,
        out_dim=2048,
    )
    condition = build_from_module(
        MiniMaxMusic3ConditionEncoder(condition_config),
        condition_config,
        "minimax-music3-condition",
    )
    transformer_config = MiniMaxMusic3TransformerConfig(
        in_channels=128,
        condition_dim=2048,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=4,
        ff_inner_dim=16,
        rotary_dim=4,
        fourier_embedding_dim=8,
    )
    transformer = build_from_module(
        MiniMaxMusic3Transformer1DModel(transformer_config),
        transformer_config,
        "minimax-music3-denoising",
    )
    vocoder_config = MiniMaxMusic3VocoderConfig(
        latent_channels=128,
        decoder_input_dim=8,
        decoder_hidden_dim=16,
        upsampling_ratios=(2,),
    )
    vocoder = build_from_module(
        MiniMaxMusic3Vocoder(vocoder_config),
        vocoder_config,
        "minimax-music3-vocoder",
    )
    package = ModelPackage(
        {
            "language_model": language["model"],
            "language_model_embedding": language["embedding"],
            "language_model_semantic_embedding": language["semantic_embedding"],
            "rvq_depth_decoder": rvq["model"],
            "rvq_depth_decoder_projection": rvq["projection"],
            "rvq_depth_decoder_embedding": rvq["embedding"],
            "rvq_depth_decoder_feedback_embedding": rvq["feedback_embedding"],
            "rvq_depth_decoder_heads": rvq["heads"],
            "condition_encoder": condition["model"],
            "transformer": transformer["model"],
            "vocoder": vocoder["model"],
        }
    )
    component_configs = {
        "language_model": {
            "max_position_embeddings": language_config.max_position_embeddings,
        },
        "condition_encoder": {
            "input_sampling_rate": condition_config.input_sampling_rate,
            "input_hop_length": condition_config.input_hop_length,
            "output_hop_length": condition_config.output_hop_length,
        },
        "vocoder": {"sampling_rate": vocoder_config.sampling_rate},
    }
    package.config = SimpleNamespace(
        component_configs=component_configs,
        # Build the workflow config the way production does: through the mobius
        # MiniMax config adapter, which owns the Music 3 defaults and derives the
        # context window from the language config. This proves the adapter emits
        # the checked-in fixture end to end.
        workflow_config=MiniMaxMusic3WorkflowConfig.from_diffusers(
            components={
                "global_decoder": "language_model",
                "global_embedding": "language_model_embedding",
                "semantic_embedding": "language_model_semantic_embedding",
                "local_decoder": "rvq_depth_decoder",
                "local_projection": "rvq_depth_decoder_projection",
                "local_embedding": "rvq_depth_decoder_embedding",
                "local_feedback_embedding": "rvq_depth_decoder_feedback_embedding",
                "local_heads": "rvq_depth_decoder_heads",
                "condition_encoder": "condition_encoder",
                "flow_transformer": "transformer",
                "vocoder": "vocoder",
            },
            component_configs=component_configs,
        ),
    )
    _materialize_deterministic_initializers(package)
    for model in package.values():
        for value in model.graph.initializers.values():
            if value.dtype.is_floating_point():
                shape = [int(dimension) for dimension in value.shape]
                value.const_value = ir.tensor(np.full(shape, 1e-4, dtype=value.dtype.numpy()))
    return package


def _write_hierarchical_audio_tokenizer(directory: Path) -> None:
    """Write a sparse-ID tokenizer that exercises the raw speech request path."""
    special_tokens = {
        "<|audio_cfg|>": 151654,
        "<|im_start|>": 151667,
        "<|im_end|>": 151668,
        "<|audio_start|>": 151669,
        "<|audio_end|>": 151670,
        "<|caption_start|>": 151671,
        "<|caption_end|>": 151672,
        "<|lyrics_start|>": 151673,
        "<|lyrics_end|>": 151674,
    }
    vocabulary = {
        "[UNK]": 0,
        "music": 1,
        "[start]": 2,
        "[verse]": 3,
        "hello": 4,
        **special_tokens,
    }
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": token_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token, token_id in special_tokens.items()
        ],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocabulary, "unk_token": "[UNK]"},
    }
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2) + "\n")


def _materialize_deterministic_initializers(package: ModelPackage) -> None:
    """Give real tiny producer graphs environment-independent synthetic weights."""
    for model in package.values():
        for value in model.graph.initializers.values():
            if value.const_value is not None and not value.dtype.is_floating_point():
                continue
            shape = [int(dimension) for dimension in value.shape]
            value.const_value = ir.tensor(np.zeros(shape, dtype=value.dtype.numpy()))


def _tts_package() -> ModelPackage:
    package = TTSTask().build(
        Qwen3TTSForConditionalGeneration(_TINY_CONFIG),
        _TINY_CONFIG,
    )
    _materialize_deterministic_initializers(package)
    graph, builder = _graph("codec")
    codes = builder.input("codes", ir.DataType.INT64, ["batch", 4, "frames"])
    waveform = builder.op.Cast(
        builder.op.Slice(
            codes,
            builder.op.Constant(value_ints=[0]),
            builder.op.Constant(value_ints=[1]),
            builder.op.Constant(value_ints=[1]),
        ),
        to=ir.DataType.FLOAT,
    )
    builder.add_output(
        _typed(waveform, ir.DataType.FLOAT, ["batch", 1, "frames"]),
        "waveform",
    )
    package["codec"] = ir.Model(graph, ir_version=11)
    return package


def _graph(name: str) -> tuple[ir.Graph, GraphBuilder]:
    graph = ir.Graph([], [], nodes=[], name=name, opset_imports={"": 24})
    return graph, GraphBuilder(graph)


def _typed(value: ir.Value, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    value.type = ir.TensorType(dtype)
    value.shape = ir.Shape(shape)
    return value


def _executable_decoder_package() -> ModelPackage:
    graph, builder = _graph("decoder")
    input_ids = builder.input("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    builder.input("attention_mask", ir.DataType.INT64, ["batch", "total_sequence"])
    builder.input("position_ids", ir.DataType.INT64, ["batch", "sequence"])
    past_key = builder.input(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        ["batch", 2, "past_sequence", 8],
    )
    past_value = builder.input(
        "past_key_values.0.value",
        ir.DataType.FLOAT,
        ["batch", 4, "past_sequence", 4],
    )
    shape = builder.op.Shape(input_ids)
    logits = builder.op.ConstantOfShape(
        builder.op.Concat(shape, builder.op.Constant(value_ints=[128]), axis=0),
        value=ir.tensor([0.0]),
    )
    batch = builder.op.Shape(input_ids, start=0, end=1)
    sequence = builder.op.Shape(input_ids, start=1, end=2)
    key_update_shape = builder.op.Concat(
        batch,
        builder.op.Constant(value_ints=[2]),
        sequence,
        builder.op.Constant(value_ints=[8]),
        axis=0,
    )
    value_update_shape = builder.op.Concat(
        batch,
        builder.op.Constant(value_ints=[4]),
        sequence,
        builder.op.Constant(value_ints=[4]),
        axis=0,
    )
    key_update = builder.op.ConstantOfShape(key_update_shape, value=ir.tensor([0.0]))
    value_update = builder.op.ConstantOfShape(value_update_shape, value=ir.tensor([0.0]))
    present_key = builder.op.Concat(past_key, key_update, axis=2)
    present_value = builder.op.Concat(past_value, value_update, axis=2)
    builder.add_output(
        _typed(logits, ir.DataType.FLOAT, ["batch", "sequence", 128]),
        "logits",
    )
    builder.add_output(
        _typed(present_key, ir.DataType.FLOAT, ["batch", 2, "present_sequence", 8]),
        "present.0.key",
    )
    builder.add_output(
        _typed(present_value, ir.DataType.FLOAT, ["batch", 4, "present_sequence", 4]),
        "present.0.value",
    )
    config = _Cfg()
    config.eos_token_id = 127
    return ModelPackage({"model": ir.Model(graph, ir_version=11)}, config=config)


def _executable_static_cache_package() -> ModelPackage:
    """A runnable decoder that scatters into fixed-capacity KV buffers.

    The appending decoder fixture grows its cache with ``Concat``; this one
    keeps a preallocated ``[batch, capacity, kv_hidden]`` buffer and writes each
    step at a per-row cursor with ``TensorScatter``. That is the shape the
    workflow's ``indexed_scatter`` state discipline describes, so the engine
    conformance run exercises the write cursor and the fixed-capacity carry
    rather than only the growing-tensor path.
    """
    capacity = 16
    graph, builder = _graph("decoder")
    input_ids = builder.input("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    builder.input("position_ids", ir.DataType.INT64, ["batch", "sequence"])
    key_cache = builder.input("key_cache.0", ir.DataType.FLOAT, ["batch", capacity, 8])
    value_cache = builder.input("value_cache.0", ir.DataType.FLOAT, ["batch", capacity, 8])
    write_indices = builder.input("write_indices", ir.DataType.INT64, ["batch"])
    builder.input("nonpad_kv_seqlen", ir.DataType.INT64, ["batch"])

    shape = builder.op.Shape(input_ids)
    logits = builder.op.ConstantOfShape(
        builder.op.Concat(shape, builder.op.Constant(value_ints=[128]), axis=0),
        value=ir.tensor([0.0]),
    )
    # This step's keys/values: (batch, sequence, 8), scattered into the buffer
    # at row `write_indices[b]` along the capacity axis.
    update_shape = builder.op.Concat(shape, builder.op.Constant(value_ints=[8]), axis=0)
    update = builder.op.ConstantOfShape(update_shape, value=ir.tensor([0.0]))
    updated_key = builder.op.TensorScatter(key_cache, update, write_indices, axis=1)
    updated_value = builder.op.TensorScatter(value_cache, update, write_indices, axis=1)

    builder.add_output(
        _typed(logits, ir.DataType.FLOAT, ["batch", "sequence", 128]),
        "logits",
    )
    builder.add_output(
        _typed(updated_key, ir.DataType.FLOAT, ["batch", capacity, 8]),
        "updated_key_cache.0",
    )
    builder.add_output(
        _typed(updated_value, ir.DataType.FLOAT, ["batch", capacity, 8]),
        "updated_value_cache.0",
    )
    config = _Cfg()
    config.eos_token_id = 127
    return ModelPackage({"model": ir.Model(graph, ir_version=11)}, config=config)


def _executable_vlm_package() -> ModelPackage:
    vision_graph, vision_builder = _graph("vision_encoder")
    pixel_values = vision_builder.input("pixel_values", ir.DataType.FLOAT, [4, 1176])
    vision_builder.input("grid_thw", ir.DataType.INT64, [1, 3])
    image_scalar = vision_builder.op.ReduceMean(pixel_values)
    image_features = vision_builder.op.Expand(
        image_scalar,
        vision_builder.op.Constant(value_ints=[1, 4, 32]),
    )
    vision_builder.add_output(
        _typed(image_features, ir.DataType.FLOAT, [1, 4, 32]),
        "image_features",
    )

    embedding_graph, embedding_builder = _graph("embedding")
    input_ids = embedding_builder.input("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    image_features = embedding_builder.input(
        "image_features", ir.DataType.FLOAT, ["batch", 4, 32]
    )
    token_values = embedding_builder.op.Cast(
        embedding_builder.op.Unsqueeze(input_ids, [2]),
        to=ir.DataType.FLOAT,
    )
    token_shape = embedding_builder.op.Concat(
        embedding_builder.op.Shape(input_ids),
        embedding_builder.op.Constant(value_ints=[32]),
        axis=0,
    )
    image_bias = embedding_builder.op.ReduceMean(image_features)
    inputs_embeds = embedding_builder.op.Add(
        embedding_builder.op.Expand(token_values, token_shape),
        image_bias,
    )
    embedding_builder.add_output(
        _typed(inputs_embeds, ir.DataType.FLOAT, ["batch", "sequence", 32]),
        "inputs_embeds",
    )

    decoder_graph, decoder_builder = _graph("decoder")
    inputs_embeds = decoder_builder.input(
        "inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence", 32]
    )
    decoder_builder.input(
        "attention_mask",
        ir.DataType.INT64,
        ["batch", "past_sequence + sequence"],
    )
    decoder_builder.input("position_ids", ir.DataType.INT64, ["batch", "sequence"])
    past_key = decoder_builder.input(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        ["batch", 2, "past_sequence", 8],
    )
    past_value = decoder_builder.input(
        "past_key_values.0.value",
        ir.DataType.FLOAT,
        ["batch", 2, "past_sequence", 8],
    )
    batch = decoder_builder.op.Shape(inputs_embeds, start=0, end=1)
    sequence = decoder_builder.op.Shape(inputs_embeds, start=1, end=2)
    logits_shape = decoder_builder.op.Concat(
        batch,
        sequence,
        decoder_builder.op.Constant(value_ints=[128]),
        axis=0,
    )
    logits = decoder_builder.op.Expand(
        decoder_builder.op.ReduceMean(inputs_embeds, axes=[2], keepdims=1),
        logits_shape,
    )
    cache_shape = decoder_builder.op.Concat(
        batch,
        decoder_builder.op.Constant(value_ints=[2]),
        sequence,
        decoder_builder.op.Constant(value_ints=[8]),
        axis=0,
    )
    cache_update = decoder_builder.op.Add(
        decoder_builder.op.ConstantOfShape(
            cache_shape,
            value=ir.tensor([0.0]),
        ),
        decoder_builder.op.ReduceMean(inputs_embeds),
    )
    present_key = decoder_builder.op.Concat(past_key, cache_update, axis=2)
    present_value = decoder_builder.op.Concat(past_value, cache_update, axis=2)
    decoder_builder.add_output(
        _typed(logits, ir.DataType.FLOAT, ["batch", "sequence", 128]),
        "logits",
    )
    decoder_builder.add_output(
        _typed(
            present_key,
            ir.DataType.FLOAT,
            ["batch", 2, "total_sequence", 8],
        ),
        "present.0.key",
    )
    decoder_builder.add_output(
        _typed(
            present_value,
            ir.DataType.FLOAT,
            ["batch", 2, "total_sequence", 8],
        ),
        "present.0.value",
    )
    return ModelPackage(
        {
            "vision_encoder": ir.Model(vision_graph, ir_version=11),
            "embedding": ir.Model(embedding_graph, ir_version=11),
            "decoder": ir.Model(decoder_graph, ir_version=11),
        },
        config=_VlmCfg(),
    )


def _executable_shared_state_pixel_flow_package() -> ModelPackage:
    """Tiny mixed-precision package for alternating shared-state execution."""
    vision_graph, vision = _graph("vision_encoder")
    pixels = vision.input("pixel_values", ir.DataType.FLOAT16, [1, 3, "height", "width"])
    feature_scalar = vision.op.ReduceMean(pixels, [0, 1, 2, 3], keepdims=0)
    features = vision.op.Expand(
        feature_scalar,
        vision.op.Constant(value_ints=[1, 256, 32]),
    )
    vision.add_output(
        _typed(features, ir.DataType.FLOAT16, [1, "image_tokens", 32]),
        "image_features",
    )

    embedding_graph, embedding = _graph("embedding")
    tokens = embedding.input("input_ids", ir.DataType.INT64, ["batch", "sequence_len"])
    image_features = embedding.input(
        "image_features", ir.DataType.FLOAT16, [1, "image_tokens", 32]
    )
    image_mask = embedding.input("image_mask", ir.DataType.BOOL, ["batch", "sequence_len"])
    token_values = embedding.op.Cast(
        embedding.op.Unsqueeze(tokens, [2]), to=ir.DataType.FLOAT16
    )
    token_shape = embedding.op.Concat(
        embedding.op.Shape(tokens),
        embedding.op.Constant(value_ints=[32]),
        axis=0,
    )
    token_values = embedding.op.Expand(token_values, token_shape)
    image_value = embedding.op.Expand(
        embedding.op.ReduceMean(image_features, [0, 1, 2], keepdims=0),
        token_shape,
    )
    embeds = embedding.op.Where(
        embedding.op.Unsqueeze(image_mask, [2]),
        image_value,
        token_values,
    )
    embedding.add_output(
        _typed(embeds, ir.DataType.FLOAT16, ["batch", "sequence_len", 32]),
        "inputs_embeds",
    )

    decoder_graph, decoder = _graph("decoder")
    embeds = decoder.input("inputs_embeds", ir.DataType.FLOAT16, ["batch", "sequence_len", 32])
    decoder.input("attention_mask", ir.DataType.INT64, ["batch", "total_sequence_len"])
    decoder.input("position_ids", ir.DataType.INT64, [3, "batch", "sequence_len"])
    past_key = decoder.input(
        "past_key_values.0.key",
        ir.DataType.FLOAT16,
        ["batch", 2, "past_sequence_len", 8],
    )
    past_value = decoder.input(
        "past_key_values.0.value",
        ir.DataType.FLOAT16,
        ["batch", 2, "past_sequence_len", 8],
    )
    sequence = decoder.op.Shape(embeds, start=1, end=2)
    logits = decoder.op.Expand(
        decoder.op.Cast(decoder.op.ReduceMean(embeds), to=ir.DataType.FLOAT),
        decoder.op.Concat(
            decoder.op.Shape(embeds, start=0, end=1),
            sequence,
            decoder.op.Constant(value_ints=[64]),
            axis=0,
        ),
    )
    update = decoder.op.Expand(
        decoder.op.ReduceMean(embeds),
        decoder.op.Concat(
            decoder.op.Concat(
                decoder.op.Shape(embeds, start=0, end=1),
                decoder.op.Constant(value_ints=[2]),
                axis=0,
            ),
            sequence,
            decoder.op.Constant(value_ints=[8]),
            axis=0,
        ),
    )
    decoder.add_output(
        _typed(logits, ir.DataType.FLOAT, ["batch", "sequence_len", 64]), "logits"
    )
    decoder.add_output(
        _typed(
            decoder.op.Concat(past_key, update, axis=2),
            ir.DataType.FLOAT16,
            ["batch", 2, "present_sequence_len", 8],
        ),
        "present.0.key",
    )
    decoder.add_output(
        _typed(
            decoder.op.Concat(past_value, update, axis=2),
            ir.DataType.FLOAT16,
            ["batch", 2, "present_sequence_len", 8],
        ),
        "present.0.value",
    )

    generation_embedding_graph, generation_embedding = _graph("image_gen_embedding")
    latent = generation_embedding.input(
        "latent", ir.DataType.FLOAT, ["batch", 3, "height", "width"]
    )
    timestep = generation_embedding.input("timestep", ir.DataType.FLOAT, [1])
    noise_scale = generation_embedding.input("noise_scale", ir.DataType.FLOAT, [1])
    token_height = generation_embedding.op.Div(
        generation_embedding.op.Shape(latent, start=2, end=3),
        generation_embedding.op.Constant(value_ints=[32]),
    )
    token_width = generation_embedding.op.Div(
        generation_embedding.op.Shape(latent, start=3, end=4),
        generation_embedding.op.Constant(value_ints=[32]),
    )
    image_tokens = generation_embedding.op.Mul(token_height, token_width)
    image_embeds = generation_embedding.op.Expand(
        generation_embedding.op.Add(
            generation_embedding.op.ReduceMean(latent, [0, 1, 2, 3], keepdims=0),
            generation_embedding.op.Add(timestep, noise_scale),
        ),
        generation_embedding.op.Concat(
            generation_embedding.op.Shape(latent, start=0, end=1),
            image_tokens,
            generation_embedding.op.Constant(value_ints=[32]),
            axis=0,
        ),
    )
    generation_embedding.add_output(
        _typed(image_embeds, ir.DataType.FLOAT, ["batch", "image_tokens", 32]),
        "image_embeds",
    )

    denoiser_graph, denoiser = _graph("image_gen_denoiser")
    image_embeds = denoiser.input(
        "image_embeds", ir.DataType.FLOAT, ["batch", "image_tokens", 32]
    )
    position_ids = denoiser.input(
        "position_ids", ir.DataType.INT64, [3, "batch", "image_tokens"]
    )
    token_grid = denoiser.input("token_grid", ir.DataType.INT64, [2])
    past_key = denoiser.input(
        "past_key_values.0.key", ir.DataType.FLOAT, ["batch", 2, "past_sequence_len", 8]
    )
    past_value = denoiser.input(
        "past_key_values.0.value", ir.DataType.FLOAT, ["batch", 2, "past_sequence_len", 8]
    )
    temporal_position = denoiser.op.ReduceMean(
        denoiser.op.Cast(
            denoiser.op.Gather(
                position_ids,
                denoiser.op.Constant(value_ints=[0]),
                axis=0,
            ),
            to=ir.DataType.FLOAT,
        ),
        [0, 1, 2],
        keepdims=0,
    )
    grid_extent = denoiser.op.ReduceSum(
        denoiser.op.Cast(token_grid, to=ir.DataType.FLOAT),
        [0],
        keepdims=0,
    )
    prediction_scalar = denoiser.op.Mul(
        denoiser.op.Add(
            denoiser.op.ReduceMean(image_embeds, [0, 1, 2], keepdims=0),
            denoiser.op.Add(
                denoiser.op.Add(
                    denoiser.op.ReduceMean(past_key, [0, 1, 2, 3], keepdims=0),
                    denoiser.op.ReduceMean(past_value, [0, 1, 2, 3], keepdims=0),
                ),
                denoiser.op.Add(temporal_position, grid_extent),
            ),
        ),
        denoiser.op.Constant(value_float=0.02),
    )
    prediction = denoiser.op.Expand(
        prediction_scalar,
        denoiser.op.Concat(
            denoiser.op.Shape(image_embeds, start=0, end=1),
            denoiser.op.Constant(value_ints=[3]),
            denoiser.op.Mul(
                token_grid,
                denoiser.op.Constant(value_ints=[32, 32]),
            ),
            axis=0,
        ),
    )
    denoiser.add_output(
        _typed(prediction, ir.DataType.FLOAT, ["batch", 3, "height", "width"]),
        "predicted_image",
    )
    denoiser.add_output(
        _typed(
            denoiser.op.Identity(past_key),
            ir.DataType.FLOAT,
            ["batch", 2, "past_sequence_len", 8],
        ),
        "present.0.key",
    )
    denoiser.add_output(
        _typed(
            denoiser.op.Identity(past_value),
            ir.DataType.FLOAT,
            ["batch", 2, "past_sequence_len", 8],
        ),
        "present.0.value",
    )

    class Config:
        pixels_per_token = 32
        t_eps = 0.02
        vocab_size = 64

    return ModelPackage(
        {
            "embedding": ir.Model(embedding_graph, ir_version=11),
            "vision_encoder": ir.Model(vision_graph, ir_version=11),
            "decoder": ir.Model(decoder_graph, ir_version=11),
            "image_gen_embedding": ir.Model(generation_embedding_graph, ir_version=11),
            "image_gen_denoiser": ir.Model(denoiser_graph, ir_version=11),
        },
        config=Config(),
    )


def _executable_diffusion_package() -> ModelPackage:
    text_graph, text_builder = _graph("text_encoder")
    input_ids = text_builder.input(
        "input_ids", ir.DataType.INT64, ["batch", "prompt_sequence"]
    )
    hidden_shape = text_builder.op.Concat(
        text_builder.op.Shape(input_ids),
        text_builder.op.Constant(value_ints=[32]),
        axis=0,
    )
    hidden = text_builder.op.Expand(
        text_builder.op.Cast(
            text_builder.op.Unsqueeze(input_ids, [2]),
            to=ir.DataType.FLOAT,
        ),
        hidden_shape,
    )
    text_builder.add_output(
        _typed(
            hidden,
            ir.DataType.FLOAT,
            ["batch", "prompt_sequence", 32],
        ),
        "encoder_hidden_states",
    )

    denoiser_graph, denoiser_builder = _graph("denoiser")
    sample = denoiser_builder.input(
        "sample", ir.DataType.FLOAT, ["batch", 4, "height", "width"]
    )
    timestep = denoiser_builder.input("timestep", ir.DataType.FLOAT, ["batch"])
    conditioning = denoiser_builder.input(
        "encoder_hidden_states",
        ir.DataType.FLOAT,
        ["batch", "prompt_sequence", 32],
    )
    batch = denoiser_builder.op.Shape(sample, start=0, end=1)
    scalar_shape = denoiser_builder.op.Concat(
        batch,
        denoiser_builder.op.Constant(value_ints=[1, 1, 1]),
        axis=0,
    )
    timestep_bias = denoiser_builder.op.Reshape(timestep, scalar_shape)
    conditioning_bias = denoiser_builder.op.Reshape(
        denoiser_builder.op.ReduceMean(conditioning, axes=[1, 2]),
        scalar_shape,
    )
    estimate = denoiser_builder.op.Add(
        sample,
        denoiser_builder.op.Add(timestep_bias, conditioning_bias),
    )
    denoiser_builder.add_output(
        _typed(
            estimate,
            ir.DataType.FLOAT,
            ["batch", 4, "height", "width"],
        ),
        "noise_pred",
    )

    vae_graph, vae_builder = _graph("vae_decoder")
    latent = vae_builder.input("latent", ir.DataType.FLOAT, ["batch", 4, "height", "width"])
    image = vae_builder.op.Slice(
        latent,
        vae_builder.op.Constant(value_ints=[0]),
        vae_builder.op.Constant(value_ints=[3]),
        vae_builder.op.Constant(value_ints=[1]),
    )
    # Standard VAE decoder semantic: bound the decoded pixels to [-1, 1] with
    # Tanh so the graph's numeric output actually honors the workflow
    # metadata's declared `value_range: negative_one_to_one` for this
    # synthetic conformance package (not just a documentation claim).
    image = vae_builder.op.Tanh(image)
    vae_builder.add_output(
        _typed(
            image,
            ir.DataType.FLOAT,
            ["batch", 3, "height", "width"],
        ),
        "image",
    )
    return ModelPackage(
        {
            "text_encoder": ir.Model(text_graph, ir_version=11),
            "denoiser": ir.Model(denoiser_graph, ir_version=11),
            "vae_decoder": ir.Model(vae_graph, ir_version=11),
        }
    )


def _causal_temporal_step(
    builder,
    value: ir.Value,
    cache: ir.Value,
    taps: int,
) -> tuple[ir.Value, ir.Value]:
    """One causal temporal convolution over frames, threaded through a cache.

    Reproduces the structure a real causal video decoder relies on: the frames
    that precede the current chunk come from the cache, the first chunk of a
    clip replicates its own first frame instead, and the tail of the padded
    input becomes the next chunk's cache. Returns the filtered frames and the
    cache to carry.
    """
    op = builder.op
    padded = op.Concat(cache, value, axis=2)
    # A zero-length cache means this is the clip's first chunk, so the missing
    # history is the first frame repeated -- the same branch-free trick the real
    # decoder uses to avoid a first-chunk special case.
    have = op.Min(
        op.Shape(cache, start=2, end=3),
        op.Constant(value_ints=[taps]),
    )
    front = op.Sub(op.Constant(value_ints=[taps]), have)
    padded = op.Pad(
        padded,
        op.Concat(front, op.Constant(value_ints=[0]), axis=0),
        None,
        op.Constant(value_ints=[2]),
        mode="edge",
    )
    length = op.Shape(padded, start=2, end=3)
    current = op.Slice(
        padded,
        op.Constant(value_ints=[taps]),
        length,
        op.Constant(value_ints=[2]),
    )
    history = op.Slice(
        padded,
        op.Constant(value_ints=[0]),
        op.Sub(length, op.Constant(value_ints=[taps])),
        op.Constant(value_ints=[2]),
    )
    filtered = op.Mul(
        op.Add(current, history),
        op.CastLike(op.Constant(value_float=0.5), current),
    )
    next_cache = op.Slice(
        padded,
        op.Sub(length, op.Constant(value_ints=[taps])),
        length,
        op.Constant(value_ints=[2]),
    )
    return filtered, next_cache


def _executable_video_package() -> ModelPackage:
    """A rank-5 video denoiser plus a causal, chunked video decoder.

    Small enough to run anywhere, but structurally a video pipeline: the latent
    carries a temporal axis through the denoise loop, and the decoder expands
    frames, works at two spatial resolutions, and keeps per-resolution
    convolution caches so a clip can be decoded a chunk at a time.
    """
    denoiser_graph, denoiser_builder = _graph("transformer")
    op = denoiser_builder.op
    sample = denoiser_builder.input(
        "sample", ir.DataType.FLOAT, ["batch", "num_frames", 4, "height", "width"]
    )
    timestep = denoiser_builder.input("timestep", ir.DataType.INT64, ["batch"])
    conditioning = denoiser_builder.input(
        "encoder_hidden_states", ir.DataType.FLOAT, ["batch", "prompt_sequence", 32]
    )
    scalar_shape = op.Concat(
        op.Shape(sample, start=0, end=1),
        op.Constant(value_ints=[1, 1, 1, 1]),
        axis=0,
    )
    timestep_bias = op.Reshape(
        op.Div(op.Cast(timestep, to=ir.DataType.FLOAT), op.Constant(value_float=1000.0)),
        scalar_shape,
    )
    conditioning_bias = op.Reshape(op.ReduceMean(conditioning, axes=[1, 2]), scalar_shape)
    # A frame-dependent term, so a stage that collapsed or reordered the temporal
    # axis would change the result rather than silently pass.
    frame_index = op.Cast(
        op.Range(
            op.Squeeze(op.Constant(value_ints=[0])),
            op.Squeeze(op.Shape(sample, start=1, end=2)),
            op.Squeeze(op.Constant(value_ints=[1])),
        ),
        to=ir.DataType.FLOAT,
    )
    frame_bias = op.Div(
        op.Reshape(frame_index, op.Constant(value_ints=[1, -1, 1, 1, 1])),
        op.Constant(value_float=100.0),
    )
    estimate = op.Add(
        op.Mul(sample, op.Constant(value_float=0.5)),
        op.Add(op.Add(timestep_bias, conditioning_bias), frame_bias),
    )
    denoiser_builder.add_output(
        _typed(estimate, ir.DataType.FLOAT, ["batch", "num_frames", 4, "height", "width"]),
        "noise_pred",
    )

    vae_graph, vae_builder = _graph("vae_decoder")
    op = vae_builder.op
    latent = vae_builder.input(
        "latent_sample",
        ir.DataType.FLOAT,
        ["batch", 4, "latent_frames", "latent_height", "latent_width"],
    )
    cache_in = vae_builder.input(
        "conv_cache.conv_in",
        ir.DataType.FLOAT,
        ["batch", 4, "cache_frames", "latent_height", "latent_width"],
    )
    cache_out_port = vae_builder.input(
        "conv_cache.conv_out",
        ir.DataType.FLOAT,
        ["batch", 3, "cache_frames", "2*latent_height", "2*latent_width"],
    )
    filtered, next_cache_in = _causal_temporal_step(vae_builder, latent, cache_in, 1)
    # Nearest-neighbour expansion in time and space: [B,C,T,H,W] -> [B,C,2T,2H,2W].
    shape = op.Shape(filtered)
    batch = op.Slice(shape, [0], [1], [0])
    channels = op.Slice(shape, [1], [2], [0])
    frames = op.Slice(shape, [2], [3], [0])
    height = op.Slice(shape, [3], [4], [0])
    width = op.Slice(shape, [4], [5], [0])
    one = op.Constant(value_ints=[1])
    two = op.Constant(value_ints=[2])
    expanded = op.Reshape(
        filtered,
        op.Concat(batch, channels, frames, one, height, one, width, one, axis=0),
    )
    expanded = op.Expand(
        expanded,
        op.Concat(batch, channels, frames, two, height, two, width, two, axis=0),
    )
    expanded = op.Reshape(
        expanded,
        op.Concat(
            batch,
            channels,
            op.Mul(frames, two),
            op.Mul(height, two),
            op.Mul(width, two),
            axis=0,
        ),
    )
    rgb = op.Slice(expanded, [0], [3], [1])
    decoded, next_cache_out = _causal_temporal_step(vae_builder, rgb, cache_out_port, 1)
    vae_builder.add_output(
        _typed(
            decoded,
            ir.DataType.FLOAT,
            ["batch", 3, "frames", "2*latent_height", "2*latent_width"],
        ),
        "sample",
    )
    vae_builder.add_output(
        _typed(
            next_cache_in,
            ir.DataType.FLOAT,
            ["batch", 4, "cache_frames", "latent_height", "latent_width"],
        ),
        "conv_cache_out.conv_in",
    )
    vae_builder.add_output(
        _typed(
            next_cache_out,
            ir.DataType.FLOAT,
            ["batch", 3, "cache_frames", "2*latent_height", "2*latent_width"],
        ),
        "conv_cache_out.conv_out",
    )
    vae_model = ir.Model(vae_graph, ir_version=11)
    vae_model.metadata_props["mobius.conv_cache.spatial_scale.conv_cache.conv_in"] = "1"
    vae_model.metadata_props["mobius.conv_cache.spatial_scale.conv_cache.conv_out"] = "2"
    return ModelPackage(
        {
            "transformer": ir.Model(denoiser_graph, ir_version=11),
            "vae_decoder": vae_model,
        }
    )


def _executable_speculative_package() -> ModelPackage:
    proposer_graph, proposer_builder = _graph("proposer")
    tokens = proposer_builder.input("tokens", ir.DataType.INT64, ["batch", 4])
    proposer_builder.input("proposal_budget", ir.DataType.INT64, ["batch"])
    proposal_scores = proposer_builder.op.ConstantOfShape(
        proposer_builder.op.Concat(
            proposer_builder.op.Shape(tokens),
            proposer_builder.op.Constant(value_ints=[32]),
            axis=0,
        ),
        value=ir.tensor([0.0]),
    )
    proposer_builder.add_output(
        _typed(
            proposer_builder.op.Identity(tokens),
            ir.DataType.INT64,
            ["batch", 4],
        ),
        "proposed_tokens",
    )
    proposer_builder.add_output(
        _typed(proposal_scores, ir.DataType.FLOAT, ["batch", 4, 32]),
        "proposal_scores",
    )

    verifier_graph, verifier_builder = _graph("verifier")
    proposed = verifier_builder.input("proposed_tokens", ir.DataType.INT64, ["batch", 4])
    past_key = verifier_builder.input(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        ["batch", 2, "past_sequence", 8],
    )
    past_value = verifier_builder.input(
        "past_key_values.0.value",
        ir.DataType.FLOAT,
        ["batch", 4, "past_sequence", 4],
    )
    batch = verifier_builder.op.Shape(proposed, start=0, end=1)
    row = verifier_builder.op.Range(
        verifier_builder.op.Constant(value_int=0),
        verifier_builder.op.Squeeze(batch, [0]),
        verifier_builder.op.Constant(value_int=1),
    )
    reject_at = verifier_builder.op.Min(
        verifier_builder.op.Add(row, verifier_builder.op.Constant(value_int=1)),
        verifier_builder.op.Constant(value_int=3),
    )
    indices = verifier_builder.op.Unsqueeze(reject_at, [1])
    corrections = verifier_builder.op.Expand(
        verifier_builder.op.Constant(value_int=31),
        batch,
    )
    target_tokens = verifier_builder.op.ScatterElements(
        proposed,
        indices,
        verifier_builder.op.Unsqueeze(corrections, [1]),
        axis=1,
    )
    target_scores = verifier_builder.op.OneHot(
        target_tokens,
        verifier_builder.op.Constant(value_int=32),
        verifier_builder.op.Constant(value_floats=[0.0, 1.0]),
        axis=-1,
    )
    key_update = verifier_builder.op.ConstantOfShape(
        verifier_builder.op.Concat(
            batch,
            verifier_builder.op.Constant(value_ints=[2, 4, 8]),
            axis=0,
        ),
        value=ir.tensor([0.0]),
    )
    value_update = verifier_builder.op.ConstantOfShape(
        verifier_builder.op.Concat(
            batch,
            verifier_builder.op.Constant(value_ints=[4, 4, 4]),
            axis=0,
        ),
        value=ir.tensor([0.0]),
    )
    present_key = verifier_builder.op.Concat(past_key, key_update, axis=2)
    present_value = verifier_builder.op.Concat(past_value, value_update, axis=2)
    verifier_builder.add_output(
        _typed(target_scores, ir.DataType.FLOAT, ["batch", 4, 32]),
        "target_scores",
    )
    verifier_builder.add_output(
        _typed(
            present_key,
            ir.DataType.FLOAT,
            ["batch", 2, "past_sequence + 4", 8],
        ),
        "present.0.key",
    )
    verifier_builder.add_output(
        _typed(
            present_value,
            ir.DataType.FLOAT,
            ["batch", 4, "past_sequence + 4", 4],
        ),
        "present.0.value",
    )
    return ModelPackage(
        {
            "proposer": ir.Model(proposer_graph, ir_version=11),
            "verifier": ir.Model(verifier_graph, ir_version=11),
        }
    )


def _executable_masked_package() -> ModelPackage:
    graph, builder = _graph("masked_denoiser")
    input_ids = builder.input("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    logits = builder.op.ConstantOfShape(
        builder.op.Concat(
            builder.op.Shape(input_ids),
            builder.op.Constant(value_ints=[128]),
            axis=0,
        ),
        value=ir.tensor([0.0]),
    )
    builder.add_output(
        _typed(logits, ir.DataType.FLOAT, ["batch", "sequence", 128]),
        "logits",
    )
    builder.add_output(
        _typed(
            builder.op.Identity(input_ids),
            ir.DataType.INT64,
            ["batch", "sequence"],
        ),
        "proposed_tokens",
    )
    return ModelPackage({"model": ir.Model(graph, ir_version=11)})


def _esm2_embedding_package() -> ModelPackage:
    """A tiny but real ESM-2 protein encoder.

    Built from the same producer path as ``facebook/esm2_t6_8M_UR50D`` so the
    fixture exercises rotary positions, pre-norm blocks and the token-dropout
    rescale rather than a stand-in graph. ESM-2 has no token-type embedding, so
    the saved artifact carries only ``input_ids`` and ``attention_mask`` -- the
    asymmetry against ProtBert below is the point of shipping both.
    """
    config = ESM2_TINY_CONFIG
    package = FeatureExtractionTask().build(EsmModel(config), config)
    # The feature-extraction task offers ``token_type_ids`` to every encoder,
    # but ESM-2 has no token-type embedding and never reads it. The real export
    # path drops the dead input during optimization, so drop it here too --
    # otherwise the committed metadata would declare a port the shipped
    # artifact does not expose.
    RemoveDeadGraphInputsPass()(package["model"])
    _materialize_deterministic_initializers(package)
    return package


def _protbert_embedding_package() -> ModelPackage:
    """A tiny but real ProtBert-shaped encoder (BERT with an amino-acid vocab)."""
    config = PROTBERT_TINY_CONFIG
    package = FeatureExtractionTask().build(BertModel(config), config)
    _materialize_deterministic_initializers(package)
    return package


def _executable_codec_package() -> ModelPackage:
    encoder_graph, encoder_builder = _graph("encoder")
    waveform = encoder_builder.input(
        "waveform",
        ir.DataType.FLOAT,
        ["batch", 1, "audio_samples"],
    )
    encoder_builder.add_output(
        _typed(
            encoder_builder.op.Identity(waveform),
            ir.DataType.FLOAT,
            ["batch", 1, "audio_samples"],
        ),
        "codes",
    )

    decoder_graph, decoder_builder = _graph("decoder")
    codes = decoder_builder.input(
        "codes",
        ir.DataType.FLOAT,
        ["batch", 1, "audio_samples"],
    )
    decoder_builder.add_output(
        _typed(
            decoder_builder.op.Identity(codes),
            ir.DataType.FLOAT,
            ["batch", 1, "audio_samples"],
        ),
        "waveform",
    )
    return ModelPackage(
        {
            "encoder": ir.Model(encoder_graph, ir_version=11),
            "decoder": ir.Model(decoder_graph, ir_version=11),
        }
    )


def _adapter_package(source_root: Path) -> ModelPackage:
    graph, builder = _graph("decoder")
    activations = builder.input("activations", ir.DataType.FLOAT, ["batch", 2])
    weight = ir.Value(
        name="projection",
        const_value=ir.tensor(np.eye(2, dtype=np.float32)),
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([2, 2]),
    )
    graph.initializers.add(weight)
    projection = builder.op.MatMul(activations, weight)
    projection.producer().name = "projection"
    projection.name = "projection.output"
    builder.add_output(
        _typed(projection, ir.DataType.FLOAT, ["batch", 2]),
        "projection.output",
    )
    model = ir.Model(graph, ir_version=11)
    target = AdapterTarget("decoder", "projection")
    descriptor = AdapterTargetDescriptor(
        target,
        semantic_name="projection",
        node_name="projection",
        output_name="projection.output",
        input_size=2,
        output_size=2,
        layer_index=0,
        rank=1,
        alpha=1.0,
        activation_dtype=ir.DataType.FLOAT,
        graph_input_a="lora.projection.a",
        graph_input_b="lora.projection.b",
        graph_input_scale="lora.projection.scale",
    )
    fingerprint = fingerprint_model_weights({"decoder": model}, (descriptor,))
    manifest = AdapterTargetManifest(fingerprint, (descriptor,))
    package = ModelPackage(
        {"decoder": model},
        adapter_target_manifest=manifest,
        adapter_service_options=AdapterServiceOptions(
            active="request.active",
            max_adapters=2,
            cache_max_entries=2,
            preserve_source_format=True,
        ),
    )
    for name, a, b in (
        (
            "red",
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[1.0], [2.0]], dtype=np.float32),
        ),
        (
            "blue",
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([[3.0], [4.0]], dtype=np.float32),
        ),
        (
            "green",
            np.array([[1.0, 1.0]], dtype=np.float32),
            np.array([[1.0], [1.0]], dtype=np.float32),
        ),
    ):
        package.add_adapter_artifact(
            AdapterArtifact(
                name,
                fingerprint,
                (AdapterWeights(target, ir.tensor(a), ir.tensor(b), 1.0),),
                identity=name,
                version="1",
            )
        )
    peft_source = source_root / "peft"
    peft_source.mkdir(parents=True, exist_ok=True)
    (peft_source / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "synthetic/adapter-base",
                "r": 1,
                "lora_alpha": 1.0,
                "target_modules": ["projection"],
                "revision": "synthetic-revision",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    save_file(
        {
            "base_model.projection.lora_A.weight": np.array([[1.0, -1.0]], dtype=np.float32),
            "base_model.projection.lora_B.weight": np.array([[0.5], [0.25]], dtype=np.float32),
        },
        peft_source / "adapter_model.safetensors",
    )
    package.add_adapter_artifact(
        load_peft_adapter(
            peft_source,
            name="peft",
            base_fingerprint=fingerprint,
            target_bindings={"projection": target},
        )
    )
    return package


def _write_adapter_metadata(package: ModelPackage, directory: Path) -> None:
    metadata = {
        "schema_version": "v1.1",
        "pipeline": {
            "workflow": {
                "manifest": {
                    "adapter_abis": {"onnx-genai.parameter-overlay": "1"},
                    "capabilities": [
                        "workflow_ssa",
                        "typed_emit",
                        "parameter_adapters",
                        "heterogeneous_adapter_batching",
                    ],
                },
                "inputs": {
                    "request.active": {
                        "contract": {
                            "dtype": "bool",
                            "rank": 1,
                            "shape": ["batch"],
                        },
                        "role": {
                            "kind": "runtime",
                            "version": "1.0",
                            "role": "adapter_active",
                        },
                        "source": {"kind": "request"},
                    },
                    "activations": {
                        "contract": {
                            "dtype": "float32",
                            "rank": 2,
                            "shape": ["batch", 2],
                        },
                        "role": {"kind": "opaque"},
                        "source": {"kind": "application", "name": "activations"},
                    },
                },
                "outputs": {
                    "result": {
                        "contract": {
                            "dtype": "float32",
                            "rank": 2,
                            "shape": ["batch", 2],
                        },
                        "role": "tensor",
                        "stage": "pre_adapter",
                    }
                },
                "components": {
                    "decoder": {
                        "implementation": {
                            "kind": "onnx",
                            "artifact": "model.onnx",
                        },
                        # An ONNX component declares no port contracts: the
                        # artifact shipped beside this metadata is authoritative
                        # for its own ports, and the runtime resolves them
                        # against the live session rather than against a copy
                        # that can drift. Only an adapter, which ships no graph,
                        # has to state its ports here.
                    },
                    "overlay": {
                        "implementation": {
                            "kind": "adapter",
                            "abi": "onnx-genai.parameter-overlay",
                            "version": "1",
                        },
                        "batch_capacity": {},
                        "ports": {
                            "inputs": {
                                "input": {
                                    "dtype": "float32",
                                    "rank": 2,
                                    "shape": ["batch", 2],
                                }
                            },
                            "outputs": {
                                "output": {
                                    "dtype": "float32",
                                    "rank": 2,
                                    "shape": ["batch", 2],
                                }
                            },
                        },
                        "contract": {
                            "id": "onnx-genai.parameter-overlay",
                            "version": "1",
                            "bindings": {"input": "input", "output": "output"},
                            "parameters": {
                                "action": "apply",
                                "component": "decoder",
                                "parameter": "projection",
                            },
                        },
                    },
                },
                "state": {},
                "steps": [
                    {
                        "kind": "invoke",
                        "component": "overlay",
                        "inputs": {"input": "activations"},
                        "outputs": {"output": "adapted"},
                    },
                    {
                        "kind": "emit",
                        "value": "adapted",
                        "output": "result",
                        "mode": "replace",
                    },
                ],
            }
        },
    }
    add_adapter_service_to_metadata(metadata, package, str(directory))
    with open(directory / "inference_metadata.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)


def generate_packages(output: Path) -> Path:
    """Write every conformance package under ``output``.

    The graphs are deterministic functions of this file, so they are generated
    on demand rather than committed. Callers that need the artifacts — the
    validator, the runtime conformance suite, the tests that resolve a port
    against the graph that really exposes it — materialize them here.
    """
    args = argparse.Namespace(output=output)
    decoder = _executable_decoder_package()
    static_cache = _executable_static_cache_package()
    packages = {
        "decoder": (decoder, {"config": decoder.config}),
        "static_cache": (static_cache, {"config": static_cache.config}),
        "vlm": (_executable_vlm_package(), {}),
        "shared_state_pixel_flow": (
            _executable_shared_state_pixel_flow_package(),
            {"num_inference_steps": 2, "guidance_scale": 2.0},
        ),
        "diffusion": (
            _executable_diffusion_package(),
            {"guidance_scale": 1.0},
        ),
        "tts": (_tts_package(), {}),
    }
    for name, (package, options) in packages.items():
        directory = args.output / name
        package.save(str(directory), progress_bar=False, check_weights=False)
        write_onnx_genai_config(package, str(directory), **options)

    hierarchical_audio = _hierarchical_audio_package()
    directory = args.output / "hierarchical_audio"
    write_hierarchical_audio_workflow_metadata(hierarchical_audio, str(directory))
    _write_hierarchical_audio_tokenizer(directory)
    hierarchical_audio.save(str(directory), progress_bar=False, check_weights=False)

    # A second image-diffusion fixture that exercises every optional part of the
    # workflow at once: classifier-free guidance from a negative prompt, a
    # multistep solver with a carried history cell, per-step trajectory emits,
    # a seeded latent drawn inside the workflow, and a scaled VAE input.
    guided = _executable_diffusion_package()
    directory = args.output / "diffusion_guided"
    guided.save(str(directory), progress_bar=False, check_weights=False)
    write_diffusion_workflow_metadata(
        guided,
        str(directory),
        num_inference_steps=3,
        schedule=[8.0, 4.0, 1.0, 0.0],
        timesteps=[900.0, 600.0, 300.0],
        solver="multistep",
        scale_model_input=False,
        decoder_input_scale=1.0 / 0.18215,
        guidance_scale=7.5,
        latent_source="seed",
        latent_row_shape=[4, 4, 4],
    )

    speculative = _executable_speculative_package()
    directory = args.output / "speculative"
    speculative.save(str(directory), progress_bar=False, check_weights=False)
    write_speculative_workflow_metadata(
        speculative,
        str(directory),
        grammar_guidance=True,
        adaptive_k_max=4,
    )

    masked = _executable_masked_package()
    directory = args.output / "masked"
    masked.save(str(directory), progress_bar=False, check_weights=False)
    write_language_diffusion_workflow_metadata(
        masked,
        str(directory),
        num_inference_steps=8,
    )

    video = _executable_video_package()
    directory = args.output / "video"
    video.save(str(directory), progress_bar=False, check_weights=False)
    write_video_diffusion_workflow_metadata(
        video,
        str(directory),
        num_inference_steps=3,
        schedule=[0.9, 0.6, 0.3, 1.0],
        timesteps=[600.0, 300.0, 0.0],
        solver="ddim",
        clip_sample_range=1.0,
        scaling_factor=1.15258426,
    )

    codec = _executable_codec_package()
    directory = args.output / "codec"
    codec.save(str(directory), progress_bar=False, check_weights=False)
    write_audio_codec_workflow_metadata(codec, str(directory))

    # Two encoder-embedding packages rather than one: ESM-2 has no token-type
    # embedding and ProtBert does, so together they pin down that the producer
    # declares the ports the artifact actually exposes instead of the ports the
    # feature-extraction task signature offers.
    esm2 = _esm2_embedding_package()
    directory = args.output / "esm2_protein_embeddings"
    esm2.save(str(directory), progress_bar=False, check_weights=False)
    write_encoder_embedding_workflow_metadata(esm2, str(directory), ESM2_TINY_CONFIG)

    protbert = _protbert_embedding_package()
    directory = args.output / "protbert_protein_embeddings"
    protbert.save(str(directory), progress_bar=False, check_weights=False)
    write_encoder_embedding_workflow_metadata(protbert, str(directory), PROTBERT_TINY_CONFIG)

    directory = args.output / "adapter"
    source_root = directory / ".sources"
    adapter = _adapter_package(source_root)
    adapter.save(str(directory), progress_bar=False)
    _write_adapter_metadata(adapter, directory)
    shutil.rmtree(source_root)

    shutil.copyfile(
        Path(__file__).parent / "fixtures/onnx_genai_workflows/README.md",
        args.output / "README.md",
    )
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    generate_packages(parser.parse_args().output)


if __name__ == "__main__":
    main()
