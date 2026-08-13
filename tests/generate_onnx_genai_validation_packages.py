from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx_ir as ir
from onnxscript import GraphBuilder

from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai import write_onnx_genai_config
from mobius.integrations.onnx_genai.auto_export_test import (
    _Cfg,
    _VlmCfg,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    write_audio_codec_workflow_metadata,
    write_language_diffusion_workflow_metadata,
    write_speculative_workflow_metadata,
)
from mobius.models.qwen3_tts import Qwen3TTSForConditionalGeneration
from mobius.models.qwen3_tts_test import _TINY_CONFIG
from mobius.tasks import TTSTask


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
    past = builder.input(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        ["batch", 2, "past_sequence", 8],
    )
    shape = builder.op.Shape(input_ids)
    logits = builder.op.ConstantOfShape(
        builder.op.Concat(shape, builder.op.Constant(value_ints=[128]), axis=0),
        value=ir.tensor([0.0]),
    )
    batch = builder.op.Shape(input_ids, start=0, end=1)
    sequence = builder.op.Shape(input_ids, start=1, end=2)
    cache_shape = builder.op.Concat(
        batch,
        builder.op.Constant(value_ints=[2]),
        sequence,
        builder.op.Constant(value_ints=[8]),
        axis=0,
    )
    update = builder.op.ConstantOfShape(cache_shape, value=ir.tensor([0.0]))
    present = builder.op.Concat(past, update, axis=2)
    builder.add_output(
        _typed(logits, ir.DataType.FLOAT, ["batch", "sequence", 128]),
        "logits",
    )
    builder.add_output(
        _typed(present, ir.DataType.FLOAT, ["batch", 2, "present_sequence", 8]),
        "present.0.key",
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
    past = verifier_builder.input(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        ["batch", 2, "past_sequence", 8],
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
    cache_update = verifier_builder.op.ConstantOfShape(
        verifier_builder.op.Concat(
            batch,
            verifier_builder.op.Constant(value_ints=[2, 4, 8]),
            axis=0,
        ),
        value=ir.tensor([0.0]),
    )
    present = verifier_builder.op.Concat(past, cache_update, axis=2)
    verifier_builder.add_output(
        _typed(target_scores, ir.DataType.FLOAT, ["batch", 4, 32]),
        "target_scores",
    )
    verifier_builder.add_output(
        _typed(
            present,
            ir.DataType.FLOAT,
            ["batch", 2, "past_sequence + 4", 8],
        ),
        "present.0.key",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    decoder = _executable_decoder_package()
    packages = {
        "decoder": (decoder, {"config": decoder.config}),
        "vlm": (_executable_vlm_package(), {}),
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

    codec = _executable_codec_package()
    directory = args.output / "codec"
    codec.save(str(directory), progress_bar=False, check_weights=False)
    write_audio_codec_workflow_metadata(codec, str(directory))

    (args.output / "README.md").write_text(
        """# ONNX GenAI workflow conformance fixtures

Generated by `tests/generate_onnx_genai_validation_packages.py` for semantic
validation and runtime conformance against `justinchuby/onnx-genai@c9bddd6e`.

The decoder, VLM, diffusion, masked diffusion, speculative, and codec packages
contain executable synthetic models. The TTS fixture uses the real tiny
Qwen3-TTS producer graphs with deterministic synthetic weights. No downloaded
model weights are included.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
