from __future__ import annotations

import argparse
from pathlib import Path

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai import write_onnx_genai_config
from mobius.integrations.onnx_genai.auto_export_test import (
    _Cfg,
    _decoder_package,
    _diffusion_package,
    _model,
    _TTSCfg,
    _value,
    _vlm_package,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    write_speculative_workflow_metadata,
)
from mobius.integrations.onnx_genai.workflow_metadata_test import _speculative_package


def _tts_package() -> ModelPackage:
    batch = "batch"
    return ModelPackage(
        {
            "talker": _model(
                "talker",
                [_value("inputs_embeds", ir.DataType.FLOAT, [batch, "sequence", 16])],
                [("last_hidden_state", ir.DataType.FLOAT, [batch, 16])],
            ),
            "code_predictor": _model(
                "code_predictor",
                [
                    _value("last_hidden_state", ir.DataType.FLOAT, [batch, 16]),
                    _value("step_index", ir.DataType.INT64, [batch]),
                ],
                [("logits", ir.DataType.FLOAT, [batch, 64])],
            ),
            "talker_step_embedder": _model(
                "talker_step_embedder",
                [_value("frame_codes", ir.DataType.INT64, [batch, 16])],
                [("inputs_embeds", ir.DataType.FLOAT, [batch, 1, 16])],
            ),
            "talker_prefill_embedder": _model(
                "talker_prefill_embedder",
                [_value("text_ids", ir.DataType.INT64, [batch, "sequence"])],
                [("prefill_embeds", ir.DataType.FLOAT, [batch, "sequence", 16])],
            ),
            "codec": _model(
                "codec",
                [_value("codes", ir.DataType.INT64, [batch, 16, "frames"])],
                [("waveform", ir.DataType.FLOAT, [batch, 1, "samples"])],
            ),
        },
        config=_TTSCfg(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    packages = {
        "decoder": (_decoder_package(), {"config": _Cfg()}),
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


if __name__ == "__main__":
    main()
