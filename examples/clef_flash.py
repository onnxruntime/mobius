# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export Clef-Flash and score one text-only schema record with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

import mobius
from mobius.integrations.clef import encode_clef_record
from mobius.models.clef import CLEF_FLASH_MODEL_ID, CLEF_FLASH_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--model", default=CLEF_FLASH_MODEL_ID)
    parser.add_argument("--revision", default=CLEF_FLASH_REVISION)
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    args = parser.parse_args()
    source = json.loads(args.record.read_text(encoding="utf-8"))
    if source.get("images") or source.get("videos"):
        raise ValueError("This example is text-only; media requires processor-derived MRoPE")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    record = encode_clef_record(tokenizer, source)
    package = mobius.build(args.model, revision=args.revision, dtype=args.dtype)
    package.save(str(args.output), max_workers=1)
    sessions = {
        name: ort.InferenceSession(
            str(args.output / name / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        for name in ("embedding", "decoder", "decision_head")
    }
    dtype = np.float32 if args.dtype == "f32" else np.float16
    absent_features = np.empty((0, package.config.hidden_size), dtype=dtype)
    embeds = sessions["embedding"].run(
        ["inputs_embeds"],
        {
            "input_ids": record.input_ids,
            "image_features": absent_features,
            "video_features": absent_features,
        },
    )[0]
    length = record.input_ids.shape[1]
    hidden = sessions["decoder"].run(
        ["hidden_states"],
        {
            "inputs_embeds": embeds,
            "attention_mask": np.ones_like(record.input_ids),
            "position_ids": np.tile(np.arange(length, dtype=np.int64), (3, 1, 1)),
        },
    )[0]
    probabilities = sessions["decision_head"].run(
        ["probabilities"], record.decision_feeds(hidden)
    )[0]
    print(json.dumps(record.probabilities_by_question(probabilities), indent=2))


if __name__ == "__main__":
    main()
