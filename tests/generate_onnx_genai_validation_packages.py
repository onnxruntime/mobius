from __future__ import annotations

import argparse
from pathlib import Path

import onnx_ir as ir
from onnxscript import GraphBuilder

from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai import write_onnx_genai_config
from mobius.integrations.onnx_genai.auto_export_test import (
    _Cfg,
    _diffusion_package,
    _model,
    _value,
    _vlm_package,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    write_audio_codec_workflow_metadata,
    write_language_diffusion_workflow_metadata,
    write_speculative_workflow_metadata,
)
from mobius.integrations.onnx_genai.workflow_metadata_test import _speculative_package
from mobius.models.qwen3_tts import Qwen3TTSForConditionalGeneration
from mobius.models.qwen3_tts_test import _TINY_CONFIG
from mobius.tasks import TTSTask


def _tts_package() -> ModelPackage:
    package = TTSTask().build(
        Qwen3TTSForConditionalGeneration(_TINY_CONFIG),
        _TINY_CONFIG,
    )
    package["codec"] = _model(
        "codec",
        [_value("codes", ir.DataType.INT64, ["batch", 4, "frames"])],
        [("waveform", ir.DataType.FLOAT, ["batch", 1, "samples"])],
    )
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
        "vlm": (_vlm_package(), {}),
        "diffusion": (_diffusion_package(text=True), {"guidance_scale": 1.0}),
        "tts": (_tts_package(), {}),
    }
    for name, (package, options) in packages.items():
        directory = args.output / name
        package.save(str(directory), progress_bar=False, check_weights=False)
        write_onnx_genai_config(package, str(directory), **options)

    speculative = _speculative_package()
    directory = args.output / "speculative"
    speculative.save(str(directory), progress_bar=False, check_weights=False)
    write_speculative_workflow_metadata(speculative, str(directory))

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

The decoder, VLM, diffusion, masked diffusion, real tiny Qwen3-TTS,
speculative, and codec packages contain graph-only synthetic models and policy
components. They contain no downloaded model weights.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
