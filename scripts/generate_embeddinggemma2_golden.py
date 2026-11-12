# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate independent GPU reference embeddings with the pinned HF model."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoProcessor

from mobius.models.embedding_gemma2 import MODEL_ID, MODEL_REVISION

TRANSFORMERS_REVISION = "f210a32fbf2e3213a143dad8c587935216088b38"


def generate(output: Path) -> None:
    source = json.loads(distribution("transformers").read_text("direct_url.json") or "{}")
    reference_commit = source.get("vcs_info", {}).get("commit_id")
    if reference_commit != TRANSFORMERS_REVISION:
        raise RuntimeError(
            f"EmbeddingGemma 2 goldens require Transformers {TRANSFORMERS_REVISION}, "
            f"got {reference_commit!r}. Install requirements/ci/embedding_gemma2.txt."
        )
    assert torch.cuda.is_available(), "Golden generation requires a real CUDA GPU."
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    reference = (
        AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .eval()
        .cuda()
    )
    root = Path(__file__).resolve().parents[1]
    cat = Image.open(root / "testdata" / "pipeline-cat-chonk.jpeg").convert("RGB")
    sound, rate = sf.read(root / "testdata" / "652-129742-0006.flac", dtype="float32")
    assert rate == 16000
    # Nonzero, two-frame video goes through the native video processor.
    frames = np.stack(
        [
            np.asarray(cat.resize((96, 96))),
            np.asarray(cat.transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize((96, 96))),
        ]
    )
    queries = [
        "task: search result | query: What causes the northern lights?",
        "task: search result | query: How is bread made?",
        "task: search result | query: What is a domestic cat?",
    ]
    documents = [
        (
            "title: none | text: The northern lights occur when charged solar particles "
            "collide with gases in Earth's upper atmosphere."
        ),
        (
            "title: none | text: Bread is made by mixing flour, water and yeast, allowing "
            "the dough to rise, then baking it in an oven."
        ),
        (
            "title: none | text: A domestic cat is a small carnivorous mammal commonly "
            "kept as a household pet."
        ),
    ]
    scenarios = [
        ("text", {"text": ["Here is my poem:", "Poem."], "padding": True}),
        (
            "sliding_boundary",
            {"text": "A small orange cat rests quietly beside an open window. " * 64},
        ),
        (
            "image",
            {
                "text": "<|image|>",
                "images": cat,
                "images_kwargs": {"max_soft_tokens": 70},
            },
        ),
        ("audio", {"text": "<|audio|>", "audio": sound, "sampling_rate": rate}),
        (
            "video",
            {
                "text": "<|video|>",
                "videos": frames,
                "do_sample_frames": False,
                "video_metadata": {"fps": 1.0, "total_num_frames": 2},
                "videos_kwargs": {"max_soft_tokens": 70},
            },
        ),
        (
            "mixed",
            {
                "text": ["A cat: <|image|>", "A recording: <|audio|>"],
                "images": [[cat], []],
                "images_kwargs": {"max_soft_tokens": 70},
                "audio": [sound],
                "sampling_rate": rate,
                "padding": True,
            },
        ),
        ("retrieval", {"text": queries + documents, "padding": True}),
    ]
    arrays = {}
    samples = []
    rankings = {}
    for name, kwargs in scenarios:
        print(f"Generating {name} reference on CUDA.", flush=True)
        inputs = processor(**kwargs, return_tensors="pt")
        # Processor-only metadata is not a model operand.
        inputs = {
            key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)
        }
        if name == "text":
            assert (inputs["attention_mask"] == 0).any(), "Text fixture must exercise padding."
        if name == "sliding_boundary":
            assert inputs["input_ids"].shape[1] > reference.config.text_config.sliding_window
        with torch.no_grad():
            tokens = reference(**{key: value.cuda() for key, value in inputs.items()})
            hidden = tokens.last_hidden_state.float()
            mask = inputs["attention_mask"].cuda().unsqueeze(-1)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            sentence = torch.nn.functional.normalize(mean, p=2, dim=-1)
        for key, value in inputs.items():
            arrays[f"{name}/input/{key}"] = value.cpu().numpy()
        arrays[f"{name}/output/last_hidden_state"] = hidden.cpu().numpy()
        arrays[f"{name}/output/sentence_embedding"] = sentence.cpu().numpy()
        samples.append({"name": name, "input_names": list(inputs)})
        if name == "retrieval":
            for dim in (128, 256, 512, 768):
                normalized = torch.nn.functional.normalize(sentence[:, :dim], p=2, dim=-1)
                ranking = (normalized[:3] @ normalized[3:].T).argsort(descending=True)
                assert torch.equal(ranking[:, 0], torch.arange(3, device="cuda"))
                rankings[str(dim)] = ranking.cpu().tolist()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "embeddinggemma-2.npz", **arrays)
    metadata = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": "float32",
        "transformers_revision": reference_commit,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "image_max_soft_tokens": 70,
        "video_max_soft_tokens": 70,
        "samples": samples,
    }
    (output / "embeddinggemma-2.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "embeddinggemma-2_retrieval.json").write_text(
        json.dumps(
            {"rankings": rankings, "queries": queries, "documents": documents}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
