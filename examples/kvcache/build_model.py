# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build a pretrained Mobius tensor-cache model and its kv_cache variant."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import onnx
from convert_to_kv_cache import convert_model
from huggingface_hub import hf_hub_download

from mobius.integrations.transformers import build

_MODEL_REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "google/gemma-3-270m": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(_MODEL_REVISIONS),
        default="Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--revision",
        help="Hugging Face revision (defaults to the pinned checkpoint commit).",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    revision = args.revision or _MODEL_REVISIONS[args.model]
    output_dir = args.output_dir.resolve()

    config_path = Path(
        hf_hub_download(
            repo_id=args.model,
            filename="config.json",
            revision=revision,
        )
    )

    print(
        f"Building {args.model}@{revision} as float32 with standard attention ...",
        flush=True,
    )
    package = build(
        args.model,
        revision=revision,
        dtype="f32",
        execution_provider="default",
        load_weights=True,
    )
    package.save(str(output_dir), external_data="onnx", max_workers=1, progress_bar=False)

    local_config = output_dir / "config.json"
    if config_path.resolve() != local_config.resolve():
        shutil.copyfile(config_path, local_config)
    model_path = output_dir / "model.onnx"
    converted_path = output_dir / "model.kv_cache.onnx"
    with local_config.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    onnx.checker.check_model(str(model_path))
    model = onnx.load_model(model_path, load_external_data=False)
    onnx.save_model(convert_model(model, config), converted_path)
    onnx.checker.check_model(str(converted_path))
    print(f"Tensor-cache model: {model_path}", flush=True)
    print(f"kv_cache model:     {converted_path}", flush=True)


if __name__ == "__main__":
    main()
