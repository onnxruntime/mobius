# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""TIPS full-output parity with the pinned independent upstream implementation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mobius import ModelPackage, build, build_from_module
from mobius._testing.golden import discover_test_cases
from mobius._testing.tips import (
    OUTPUT_NAMES,
    load_reference,
    make_feeds,
    reference_outputs,
)
from mobius.models.tips import TIPS_REVISION, TipsConfig, TipsModel


@pytest.mark.integration
def test_tipsv2_synthetic_parity(tmp_path):
    upstream = get_class_from_dynamic_module(
        "modeling_tips.TIPSv2Model",
        "google/tipsv1-b14",
        revision=TIPS_REVISION,
        code_revision=TIPS_REVISION,
    ).__module__.rsplit(".", 1)[0]
    vision_cls = importlib.import_module(upstream + ".image_encoder").VisionTransformer
    text_cls = importlib.import_module(upstream + ".text_encoder").TextEncoder
    config = TipsConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=128,
        image_size=28,
        patch_size=14,
        vision_hidden_size=32,
        vision_num_layers=2,
        vision_num_heads=4,
        max_position_embeddings=8,
    )
    torch.manual_seed(7)
    vision = vision_cls(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=2,
        num_heads=4,
        init_values=1.0,
        num_register_tokens=1,
        block_chunks=0,
    ).eval()
    text = text_cls(
        {"hidden_size": 32, "mlp_dim": 48, "num_heads": 4, "num_layers": 2},
        vocab_size=128,
    ).eval()
    module = TipsModel(config)
    package = build_from_module(module, config, task=module.default_task)
    weights = {
        **{f"vision_encoder.{k}": v for k, v in vision.state_dict().items()},
        **{f"text_encoder.{k}": v for k, v in text.state_dict().items()},
    }
    package.apply_weights(module.preprocess_weights(weights))
    package.save(str(tmp_path))
    inputs = {
        "pixel_values": np.random.default_rng(42).random((2, 3, 28, 28), dtype=np.float32),
        "input_ids": np.array(
            [[1, 2, 3, 0, 0, 0, 0, 0], [4, 5, 6, 7, 8, 0, 0, 0]], dtype=np.int64
        ),
    }
    inputs["padding_mask"] = (inputs["input_ids"] == 0).astype(np.int64)
    with torch.inference_mode():
        expected = (
            *vision(torch.from_numpy(inputs["pixel_values"])),
            text(
                torch.from_numpy(inputs["input_ids"]), torch.from_numpy(inputs["padding_mask"])
            ),
        )
    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    for actual, ref in zip(session.run(None, inputs), expected):
        np.testing.assert_allclose(actual, ref.numpy(), atol=1e-4, rtol=1e-4)


@pytest.mark.integration
def test_tipsv2_full_checkpoint_cuda(tmp_path):
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    case = next(c for c in discover_test_cases() if c.model_type == "tipsv2")
    feeds = make_feeds(case)
    assert np.any(feeds["padding_mask"] == 1)
    reference = load_reference("cuda")
    from huggingface_hub import hf_hub_download

    raw_config = json.loads(
        Path(hf_hub_download(case.model_id, "config.json", revision=TIPS_REVISION)).read_text()
    )
    assert TipsConfig.from_transformers(reference.config) == TipsConfig.from_transformers(
        raw_config
    )
    expected = reference_outputs(reference, feeds, "cuda")
    package = build(
        case.model_id,
        revision=TIPS_REVISION,
        dtype="f32",
        load_weights=True,
        execution_provider="cuda",
    )
    package.save(str(tmp_path))
    restored = ModelPackage.load(str(tmp_path))
    assert set(restored) == {"model"}
    assert restored["model"].graph.metadata_props["tips.pixel_range"] == "[0,1]"
    restored_path = tmp_path / "restored"
    restored.save(str(restored_path))
    options = ort.SessionOptions()
    options.enable_profiling = True
    session = ort.InferenceSession(
        str(restored_path / "model.onnx"),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    actual = dict(zip(OUTPUT_NAMES, session.run(None, feeds)))
    for name in OUTPUT_NAMES:
        assert actual[name].shape == expected[name].shape
        assert np.isfinite(actual[name]).all()
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-3, rtol=1e-3)
        print(f"{name}: max_abs={np.abs(actual[name] - expected[name]).max():.8g}")
    cpu_session = ort.InferenceSession(
        str(restored_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    for name, cpu_output in zip(OUTPUT_NAMES, cpu_session.run(None, feeds)):
        np.testing.assert_allclose(cpu_output, expected[name], atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(cpu_output, actual[name], atol=1e-3, rtol=1e-3)
    del cpu_session
    changed = {name: value.copy() for name, value in feeds.items()}
    changed["input_ids"][changed["padding_mask"] == 1] = 123
    changed["pixel_values"] = changed["pixel_values"][:1]
    changed_outputs = dict(zip(OUTPUT_NAMES, session.run(None, changed)))
    np.testing.assert_allclose(
        changed_outputs["text_embeds"], actual["text_embeds"], atol=1e-4, rtol=1e-4
    )
    assert changed_outputs["patch_tokens"].shape == (1, 1024, 768)
    changed["padding_mask"][:] = 1
    empty = dict(zip(OUTPUT_NAMES, session.run(None, changed)))["text_embeds"]
    np.testing.assert_array_equal(empty, np.zeros_like(empty))
    profile = Path(session.end_profiling())
    kernels = json.loads(profile.read_text())
    cuda_ops = {
        event["args"].get("op_name")
        for event in kernels
        if event.get("args", {}).get("provider") == "CUDAExecutionProvider"
    }
    assert "MatMul" in cuda_ops or "FusedMatMul" in cuda_ops
    print(f"CUDA kernels: {sorted(cuda_ops)}")
