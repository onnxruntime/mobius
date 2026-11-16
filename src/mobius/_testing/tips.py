# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned independent TIPS reference, golden collection, and semantic replay."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import torch

from mobius._testing.golden import GoldenTestCase
from mobius.models.tips import TIPS_REVISION

OUTPUT_NAMES = ("cls_token", "register_tokens", "patch_tokens", "text_embeds")
SPATIAL_ROWS = (0, 31, 32, 511, 512, 1023)


def load_reference(device: str):
    from transformers import AutoModel

    return (
        AutoModel.from_pretrained(
            "google/tipsv1-b14",
            revision=TIPS_REVISION,
            code_revision=TIPS_REVISION,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        .eval()
        .to(device)
    )


def make_feeds(case: GoldenTestCase) -> dict[str, np.ndarray]:
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # Invoke upstream's tokenizer, pinning its model file separately: the
    # upstream model's lazy string loader does not forward checkpoint revision.
    cls = get_class_from_dynamic_module(
        "modeling_tips.TIPSv2Model",
        case.model_id,
        revision=case.revision,
        code_revision=case.revision,
    )
    tokenizer_cls = importlib.import_module(
        cls.__module__.rsplit(".", 1)[0] + ".text_encoder"
    ).Tokenizer
    tokenizer = tokenizer_cls(
        hf_hub_download(case.model_id, "tokenizer.model", revision=case.revision)
    )
    ids, paddings = tokenizer.tokenize(case.prompts, max_len=64)
    pixels = []
    root = case.yaml_path.parents[2]
    for filename in case.images:
        with Image.open(root / filename) as image:
            image = image.convert("RGB").resize((448, 448), Image.Resampling.BILINEAR)
            pixels.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return {
        "pixel_values": np.stack(pixels),
        "input_ids": ids.astype(np.int64),
        "padding_mask": paddings.astype(np.int64),
    }


def reference_outputs(reference, feeds: dict[str, np.ndarray], device: str):
    with torch.inference_mode():
        result = reference(
            **{name: torch.from_numpy(value).to(device) for name, value in feeds.items()}
        )
    values = (
        result.image_features.cls_token,
        result.image_features.register_tokens,
        result.image_features.patch_tokens,
        result.text_embeds,
    )
    return {name: value.float().cpu().numpy() for name, value in zip(OUTPUT_NAMES, values)}


def semantic_outputs(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

    text = normalize(outputs["text_embeds"])
    global_scores = normalize(outputs["cls_token"][:, 0]) @ text.T
    spatial_scores = normalize(outputs["patch_tokens"]) @ text.T
    # Dense retrieval checks all 1024 spatial positions, not just CLS pooling.
    return {"global_scores": global_scores, "spatial_scores": spatial_scores}


def generate_golden(case: GoldenTestCase, json_path: Path, device: str) -> None:
    import importlib.metadata

    import transformers

    if case.revision != TIPS_REVISION:
        raise ValueError(f"TIPS reference requires immutable revision {TIPS_REVISION}.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = load_reference(device)
    feeds = make_feeds(case)
    outputs = reference_outputs(reference, feeds, device)
    spatial_rows = outputs["patch_tokens"][:, SPATIAL_ROWS]
    record = {
        "model_id": case.model_id,
        "revision": case.revision,
        "reference_class": "modeling_tips.TIPSv2Model",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "transformers_source": json.loads(
            importlib.metadata.distribution("transformers").read_text("direct_url.json")
            or "{}"
        ).get("vcs_info", {}),
        "device": device,
        "tf32": False,
        "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
        "outputs": {
            name: value.tolist() for name, value in outputs.items() if name != "patch_tokens"
        },
        "spatial_rows": list(SPATIAL_ROWS),
        "patch_tokens": spatial_rows.tolist(),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    semantics = semantic_outputs(outputs)
    semantic_record = {
        "model_id": case.model_id,
        "revision": case.revision,
        "contract": "global-and-dense-image-text-retrieval",
        "prompts": case.prompts,
        "images": case.images,
        "global_scores": semantics["global_scores"].tolist(),
        "spatial_scores": semantics["spatial_scores"].tolist(),
        "global_ranking": np.argsort(-semantics["global_scores"], axis=-1).tolist(),
    }
    json_path.with_name(json_path.stem + "_generation.json").write_text(
        json.dumps(semantic_record, indent=2) + "\n"
    )


def assert_golden(case: GoldenTestCase, level: str, tmp_path: Path) -> None:
    import onnxruntime as ort

    from mobius import ModelPackage, build
    from mobius._testing.golden import golden_path_for_case

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "TIPS L4/L5 requires CUDAExecutionProvider; install ORT GPU and CUDA."
        )
    feeds = make_feeds(case)
    package = build(
        case.model_id,
        revision=case.revision,
        dtype="f32",
        load_weights=True,
        execution_provider="cuda",
    )
    package.save(str(tmp_path))
    restored = ModelPackage.load(str(tmp_path))
    assert set(restored) == {"model"}
    assert restored["model"].graph.metadata_props["tips.padding_mask"] == "1=padding,0=valid"
    restored_path = tmp_path / "restored"
    restored.save(str(restored_path))
    session = ort.InferenceSession(
        str(restored_path / "model.onnx"),
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    actual = dict(zip(OUTPUT_NAMES, session.run(None, feeds)))
    path = golden_path_for_case(case)
    if level == "L5":
        path = path.with_name(path.stem + "_generation.json")
    golden = json.loads(path.read_text())
    assert golden["revision"] == case.revision == TIPS_REVISION
    if level == "L4":
        for name, shape in golden["output_shapes"].items():
            assert list(actual[name].shape) == shape
            assert np.isfinite(actual[name]).all()
        for name, expected in golden["outputs"].items():
            np.testing.assert_allclose(actual[name], expected, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            actual["patch_tokens"][:, golden["spatial_rows"]],
            golden["patch_tokens"],
            atol=1e-3,
            rtol=1e-3,
        )
    else:
        semantics = semantic_outputs(actual)
        for name, value in semantics.items():
            np.testing.assert_allclose(value, golden[name], atol=1e-4, rtol=1e-3)
        np.testing.assert_array_equal(
            np.argsort(-semantics["global_scores"], axis=-1), golden["global_ranking"]
        )
        # The two real fixtures are a cat and dog; query indices 0 and 1.
        np.testing.assert_array_equal(semantics["global_scores"].argmax(-1), [0, 1])
