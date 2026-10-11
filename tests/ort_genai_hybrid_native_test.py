# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Opt-in native Qwen hybrid state proof, separate from the pinned fast CPU lane.

Run with MOBIUS_QWEN_NATIVE_PROBE=1. No downloads, real weights, or manual ORT
cache loop: GenAI owns preprocessing, positions, caches, and token generation.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
from onnxscript import GraphBuilder

from mobius._model_package import ModelPackage
from mobius.integrations.ort_genai import write_ort_genai_config

pytestmark = pytest.mark.skipif(
    os.environ.get("MOBIUS_QWEN_NATIVE_PROBE") != "1",
    reason="opt-in native hybrid qualification, not the generic 0.15.2 CPU lane",
)

_HIDDEN_SIZE = 5120
_KV_HEADS = 4
_HEAD_DIM = 256
_CONV_CHANNELS = 10240
_STATE_HEADS = 48
_STATE_DIM = 128


def _disable_core_dumps() -> None:
    """Disable POSIX core dumps to avoid huge GPU crash artifacts."""
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _output(
    builder: GraphBuilder,
    value: ir.Value,
    name: str,
    dtype: ir.DataType,
    shape: list[int | str],
) -> None:
    value.type = ir.TensorType(dtype)
    value.shape = ir.Shape(shape)
    builder.add_output(value, name)


def _hybrid_package() -> ModelPackage:
    graph = ir.Graph([], [], nodes=[], name="decoder", opset_imports={"": 24})
    b = GraphBuilder(graph)
    embeds = b.input("inputs_embeds", ir.DataType.FLOAT16, ["batch", "sequence", _HIDDEN_SIZE])
    b.input("attention_mask", ir.DataType.INT64, ["batch", "total_sequence"])
    b.input("position_ids", ir.DataType.INT64, [3, "batch", "sequence"])
    score = b.op.Constant(value_float=0.0)
    for index in range(8):
        if index % 4 == 3:
            for role in ("key", "value"):
                past = b.input(
                    f"past_key_values.{index}.{role}",
                    ir.DataType.FLOAT16,
                    ["batch", _KV_HEADS, "past_sequence", _HEAD_DIM],
                )
                if index == 3 and role == "key":
                    length = b.op.Shape(past, start=2, end=3)
                    score = b.op.Add(score, b.op.Cast(length, to=ir.DataType.FLOAT))
                # Form new zero KV rows from zero token embeddings with ordinary
                # CUDA-capable tensor ops, not a CPU ConstantOfShape output.
                update = b.op.Slice(
                    embeds,
                    b.op.Constant(value_ints=[0]),
                    b.op.Constant(value_ints=[_KV_HEADS * _HEAD_DIM]),
                    b.op.Constant(value_ints=[2]),
                )
                update = b.op.Reshape(
                    update,
                    b.op.Concat(
                        b.op.Shape(embeds, start=0, end=2),
                        b.op.Constant(value_ints=[_KV_HEADS, _HEAD_DIM]),
                        axis=0,
                    ),
                )
                update = b.op.Transpose(update, perm=[0, 2, 1, 3])
                _output(
                    b,
                    b.op.Concat(past, update, axis=2),
                    f"present.{index}.{role}",
                    ir.DataType.FLOAT16,
                    ["batch", _KV_HEADS, "present_sequence", _HEAD_DIM],
                )
        else:
            for role, dtype, shape, increment in (
                ("conv_state", ir.DataType.FLOAT16, ["batch", _CONV_CHANNELS, 3], 1),
                (
                    "recurrent_state",
                    ir.DataType.FLOAT,
                    ["batch", _STATE_HEADS, _STATE_DIM, _STATE_DIM],
                    2,
                ),
            ):
                past = b.input(f"past_key_values.{index}.{role}", dtype, shape)
                # Both independently typed state families determine the token.
                # The other slots are verified via native outputs below.
                if index == 0:
                    score = b.op.Add(
                        score,
                        b.op.ReduceMean(b.op.Cast(past, to=ir.DataType.FLOAT), keepdims=0),
                    )
                constant = b.op.Constant(
                    value=ir.tensor(np.array(increment, dtype=dtype.numpy()))
                )
                _output(b, b.op.Add(past, constant), f"present.{index}.{role}", dtype, shape)
    token = b.op.Cast(score, to=ir.DataType.INT64)
    token = b.op.Mod(token, b.op.Constant(value_int=32))
    logits = b.op.OneHot(
        token, b.op.Constant(value_int=32), b.op.Constant(value_floats=[0.0, 10.0])
    )
    logits = b.op.Expand(
        logits,
        b.op.Concat(
            b.op.Shape(embeds, start=0, end=2), b.op.Constant(value_ints=[32]), axis=0
        ),
    )
    _output(b, logits, "logits", ir.DataType.FLOAT, ["batch", "sequence", 32])

    embedding = ir.Graph([], [], nodes=[], name="embedding", opset_imports={"": 24})
    e = GraphBuilder(embedding)
    ids = e.input("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    e.input("image_features", ir.DataType.FLOAT16, ["features", _HIDDEN_SIZE])
    # Use the real embedding's standard Gather contract. GDB located the prior
    # zero-fill probe's native prefill crash in CPU ConstantOfShape::Compute.
    table = e.op.Constant(value=ir.tensor(np.zeros((32, _HIDDEN_SIZE), np.float16)))
    _output(
        e,
        e.op.Gather(table, ids, axis=0),
        "inputs_embeds",
        ir.DataType.FLOAT16,
        ["batch", "sequence", _HIDDEN_SIZE],
    )

    vision = ir.Graph([], [], nodes=[], name="vision", opset_imports={"": 24})
    v = GraphBuilder(vision)
    pixels = v.input("pixel_values", ir.DataType.FLOAT, ["patches", 1536])
    v.input("image_grid_thw", ir.DataType.INT64, ["images", 3])
    features = v.op.ReduceMean(pixels, v.op.Constant(value_ints=[1]), keepdims=1)
    features = v.op.Slice(
        features,
        v.op.Constant(value_ints=[0]),
        v.op.Constant(value_ints=[np.iinfo(np.int64).max]),
        v.op.Constant(value_ints=[0]),
        v.op.Constant(value_ints=[4]),
    )
    features = v.op.Tile(
        v.op.Cast(features, to=ir.DataType.FLOAT16),
        v.op.Constant(value_ints=[1, _HIDDEN_SIZE]),
    )
    _output(
        v,
        features,
        "image_features",
        ir.DataType.FLOAT16,
        ["features", _HIDDEN_SIZE],
    )
    config = SimpleNamespace(
        model_type="qwen3_5",
        vocab_size=32,
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=8,
        num_attention_heads=24,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=31,
        pad_token_id=30,
        image_token_id=24,
        video_token_id=25,
        vision_start_token_id=26,
        vision_end_token_id=27,
        spatial_merge_size=2,
        temporal_patch_size=2,
        vision=SimpleNamespace(patch_size=16, spatial_merge_size=2),
    )
    return ModelPackage(
        {
            "decoder": ir.Model(graph, ir_version=10),
            "embedding": ir.Model(embedding, ir_version=10),
            "vision_encoder": ir.Model(vision, ir_version=10),
        },
        config=config,
    )


def _write_tokenizer(directory: Path) -> None:
    vocab = {f"token{i}": i for i in range(32)}
    vocab["hello"] = vocab.pop("token2")
    # Only the image markers needed by the native image prompt; IDs match the
    # toy config without changing its vocabulary size or text token.
    image_markers = {"<|image_pad|>": 24, "<|vision_start|>": 26, "<|vision_end|>": 27}
    for content, token_id in image_markers.items():
        vocab.pop(f"token{token_id}")
        vocab[content] = token_id
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": token_id,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for content, token_id in image_markers.items()
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "token0",
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [],
        },
    }
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "additional_special_tokens": list(image_markers),
            }
        ),
        encoding="utf-8",
    )


def test_native_sparse_mixed_state_progression_and_reset(tmp_path: Path) -> None:
    import onnxruntime_genai as og

    _disable_core_dumps()
    package = _hybrid_package()
    for graph in package.values():
        assert not graph.functions
        assert all(node.domain in {"", "ai.onnx"} for node in graph.graph.all_nodes())
    package.save(tmp_path, progress_bar=False)
    _write_tokenizer(tmp_path)
    write_ort_genai_config(
        package,
        str(tmp_path),
        ep="cuda",
        context_length=64,
    )
    payload = json.loads((tmp_path / "genai_config.json").read_text())
    assert payload["model"]["type"] == "qwen3_5"
    assert payload["model"]["decoder"]["num_hidden_layers"] == 8
    assert payload["search"]["past_present_share_buffer"] is False
    print("GenAI version:", og.__version__, flush=True)
    print("Synthetic config:", json.dumps(payload, indent=2), flush=True)
    config = og.Config(str(tmp_path))
    # The installed Config override does not route explicit CPU auxiliary
    # sessions to CUDA. All three session options must match before prefill.
    for name in ("decoder", "vision", "embedding"):
        providers = payload["model"][name]["session_options"]["provider_options"]
        assert providers[0]["cuda"]["enable_cuda_graph"] == "0"
    config.clear_providers()
    config.append_provider("CUDAExecutionProvider")
    model = og.Model(config)
    print("Native model constructed", flush=True)
    processor = model.create_multimodal_processor()
    inputs = processor("hello")
    print("Native processor text inputs created", flush=True)
    receipts = []
    for run in range(2):
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=4, do_sample=False)
        print("Constructing native generator", run, flush=True)
        generator = og.Generator(model, params)
        print("Native generator constructed; starting set_inputs/prefill", run, flush=True)
        generator.set_inputs(inputs)
        print("Native set_inputs/prefill completed", run, flush=True)
        tokens = []
        for step in range(3):
            generator.generate_next_token()
            tokens.append(int(generator.get_next_tokens()[0]))
            for index in range(8):
                if index % 4 == 3:
                    for role in ("key", "value"):
                        state = generator.get_output(f"present.{index}.{role}")
                        assert state.dtype == np.float16
                        assert state.shape == (1, _KV_HEADS, step + 1, _HEAD_DIM)
                else:
                    for role, dtype, increment in (
                        ("conv_state", np.float16, 1),
                        ("recurrent_state", np.float32, 2),
                    ):
                        state = generator.get_output(f"present.{index}.{role}")
                        assert state.dtype == dtype
                        assert state.shape == (
                            (1, _CONV_CHANNELS, 3)
                            if role == "conv_state"
                            else (1, _STATE_HEADS, _STATE_DIM, _STATE_DIM)
                        )
                        np.testing.assert_array_equal(state, (step + 1) * increment)
            print("Native run/step/token:", run, step, tokens[-1], flush=True)
        assert tokens == [0, 4, 8]
        receipts.append(tokens)
        del generator, params
        gc.collect()
    print("Native progression/reset receipts:", receipts, flush=True)
    (tmp_path / "native-proof.json").write_text(
        json.dumps(
            {
                "genai_version": og.__version__,
                "provider": "CUDAExecutionProvider",
                "sparse_kv_indices": [3, 7],
                "conv_dtype": "float16",
                "recurrent_dtype": "float32",
                "past_present_share_buffer": False,
                "progression_and_reset": receipts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del inputs, processor, model
    gc.collect()


def test_generic_native_control(tmp_path: Path) -> None:
    """Distinguish a hybrid-runtime blocker from a broken native installation."""
    import onnxruntime_genai as og
    from ort_genai_e2e_test import _synthetic_decoder_package, _write_tokenizer

    _disable_core_dumps()
    package = _synthetic_decoder_package()
    package.save(tmp_path, progress_bar=False)
    _write_tokenizer(tmp_path)
    write_ort_genai_config(package, str(tmp_path), context_length=32)
    config = og.Config(str(tmp_path))
    config.clear_providers()
    config.append_provider("CUDAExecutionProvider")
    model = og.Model(config)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens([2])
    tokens = []
    for _ in range(3):
        generator.generate_next_token()
        tokens.append(int(generator.get_next_tokens()[0]))
    assert tokens == [0, 1, 2]
    print("Generic native control receipts:", tokens, flush=True)
    del generator, params, model
    gc.collect()


def test_native_processor_text_contract(tmp_path: Path) -> None:
    """Validate the native expanded-input API used by the real runner."""
    import onnxruntime_genai as og

    _disable_core_dumps()
    package = _hybrid_package()
    package.save(tmp_path, progress_bar=False)
    _write_tokenizer(tmp_path)
    write_ort_genai_config(package, str(tmp_path), ep="onnx-standard", context_length=64)
    config = og.Config(str(tmp_path))
    config.clear_providers()
    config.append_provider("CUDAExecutionProvider")
    model = og.Model(config)
    processor = model.create_multimodal_processor()
    inputs = processor("hello")
    ids = inputs["input_ids"].as_numpy()
    print("Native expanded input_ids:", ids, "shape:", ids.shape, flush=True)
    assert ids.dtype == np.int32 or ids.dtype == np.int64
    assert ids.size == 1
    assert int(ids.reshape(-1)[0]) == 2
    del inputs, processor, model
    gc.collect()


def test_native_processor_preserves_uniform_rgb_channels(tmp_path: Path) -> None:
    """Check true native pixels before prefill, without checkpoint weights."""
    import onnxruntime_genai as og
    from PIL import Image

    _disable_core_dumps()
    package = _hybrid_package()
    package.save(tmp_path, progress_bar=False)
    _write_tokenizer(tmp_path)
    write_ort_genai_config(package, str(tmp_path), ep="cuda", context_length=64)
    payload = json.loads((tmp_path / "processor_config.json").read_text())
    operations = {
        item["operation"]["type"]: item["operation"].get("attrs", {})
        for item in payload["processor"]["transforms"]
    }
    rgb = [255, 128, 0]
    image_path = tmp_path / "distinct-rgb.png"
    Image.new("RGB", (64, 64), tuple(rgb)).save(image_path)
    config = og.Config(str(tmp_path))
    config.clear_providers()
    config.append_provider("CUDAExecutionProvider")
    model = og.Model(config)
    processor = model.create_multimodal_processor()
    images = og.Images.open(str(image_path))
    inputs = processor("<|vision_start|><|image_pad|><|vision_end|>hello", images=images)
    try:
        pixels = inputs["pixel_values"].as_numpy()
        grid = inputs["image_grid_thw"].as_numpy()
        ids = inputs["input_ids"].as_numpy()
        assert pixels.dtype == np.float32
        np.testing.assert_array_equal(grid, [[1, 4, 4]])
        assert pixels.shape == (16, 1536)
        assert np.count_nonzero(ids == 24) == 4
        norm = operations["Normalize"]
        expected = (
            np.asarray(rgb, dtype=np.float32) * operations["Rescale"]["rescale_factor"]
            - np.asarray(norm["mean"], dtype=np.float32)
        ) / np.asarray(norm["std"], dtype=np.float32)
        # Inspect the documented C,T,H,W patch layout only. No flips/reorders,
        # no tensor feeds, and no Generator/set_inputs/prefill are used here.
        patches = pixels.reshape(16, 3, 2, 16, 16)
        observed = patches.mean(axis=(0, 2, 3, 4))
        print(
            "Native before-prefill RGB:",
            rgb,
            "expected normalized channels:",
            expected.tolist(),
            "observed channels:",
            observed.tolist(),
            "grid:",
            grid.tolist(),
            flush=True,
        )
        np.testing.assert_allclose(
            patches,
            np.broadcast_to(expected[None, :, None, None, None], patches.shape),
            rtol=0,
            atol=1e-6,
        )
        (tmp_path / "native-rgb-proof.json").write_text(
            json.dumps(
                {
                    "genai_version": og.__version__,
                    "stage": "native processor before prefill",
                    "rgb": rgb,
                    "expected_normalized_channels": expected.tolist(),
                    "observed_normalized_channels": observed.tolist(),
                    "max_abs_error": float(
                        np.max(np.abs(patches - expected[None, :, None, None, None]))
                    ),
                    "pixel_shape": list(pixels.shape),
                    "grid": grid.tolist(),
                    "manual_channel_or_row_reorder": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        del inputs, images, processor, model
        gc.collect()
