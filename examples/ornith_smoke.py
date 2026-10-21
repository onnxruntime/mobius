#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate exported Ornith components, not Engine orchestration or accuracy."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


def _session(path: Path, *, diagnostics: bool = False) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    logger.info("Loading %s", path)
    if diagnostics:
        model = onnx.load(path, load_external_data=False)
        info = {value.name: value for value in model.graph.value_info}
        selected = {
            "LinearAttention",
            "GroupQueryAttention",
            "QMoE",
            "Add",
            "MatMulNBits",
            "RMSNormalization",
            "RotaryEmbedding",
        }
        for node in model.graph.node:
            if node.op_type not in selected:
                continue
            if any(
                node.output[0].startswith(f"v_decoder.model.layers.{layer}.")
                for layer in range(4)
            ):
                name = node.output[0]
                if name not in info:
                    raise RuntimeError(f"Diagnostic value lacks type metadata: {name}")
                model.graph.output.append(info[name])
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".onnx") as temporary:
            onnx.save_model(model, temporary.name)
            session = ort.InferenceSession(
                temporary.name, sess_options=options, providers=["CUDAExecutionProvider"]
            )
    else:
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CUDAExecutionProvider"]
        )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA provider is not active for {path}")
    return session


def _cuda(value: np.ndarray) -> ort.OrtValue:
    return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value), "cuda", 0)


def _empty_states(session: ort.InferenceSession) -> dict[str, ort.OrtValue]:
    states = {}
    for value in session.get_inputs():
        if not value.name.startswith("past_key_values."):
            continue
        if value.name.endswith((".key", ".value")):
            shape = [1, value.shape[1], 0, value.shape[3]]
            states[value.name] = ort.OrtValue.ortvalue_from_shape_and_type(
                shape, onnx.TensorProto.FLOAT8E4M3FN, "cuda", 0
            )
        else:
            dtype = np.float32 if value.type == "tensor(float)" else np.float16
            shape = [1, *value.shape[1:]]
            states[value.name] = _cuda(np.zeros(shape, dtype=dtype))
    return states


def _run(
    session: ort.InferenceSession, inputs: dict[str, ort.OrtValue]
) -> dict[str, ort.OrtValue]:
    binding = session.io_binding()
    for name, value in inputs.items():
        binding.bind_ortvalue_input(name, value)
    names = [value.name for value in session.get_outputs()]
    for name in names:
        binding.bind_output(name, "cuda", 0)
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    return dict(zip(names, binding.get_outputs(), strict=True))


def _finite(name: str, value: ort.OrtValue) -> list[int]:
    array = value.numpy()
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError(f"{name} returned empty or non-finite values")
    return list(array.shape)


def _check_states(values: dict[str, ort.OrtValue]) -> None:
    shape = ["batch", "heads", "sequence", "head_dim"]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Cast", ["cache"], ["value"], to=onnx.TensorProto.FLOAT)],
        "fp8-cache-probe",
        [onnx.helper.make_tensor_value_info("cache", onnx.TensorProto.FLOAT8E4M3FN, shape)],
        [onnx.helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, shape)],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 23)], ir_version=10
    )
    cast = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    for name, value in values.items():
        if not name.startswith("present."):
            continue
        if name.endswith((".key", ".value")):
            value = _run(cast, {"cache": value})["value"]
        array = value.numpy()
        if not np.isfinite(array).all():
            raise RuntimeError(f"{name} contains non-finite cache values")
        logger.info("%s absolute maximum: %.6g", name, np.abs(array).max())


def _check_diagnostics(values: dict[str, ort.OrtValue]) -> None:
    for name, value in values.items():
        if name.startswith(("present.", "logits", "mtp_seed")):
            continue
        array = value.numpy()
        if not np.isfinite(array).all():
            raise RuntimeError(f"First non-finite diagnostic value: {name}")
        logger.info("%s absolute maximum: %.6g", name, np.abs(array).max())


def vision_smoke(
    model: Path, processor, embedding: ort.InferenceSession, config: dict
) -> dict[str, object]:
    vision = _session(model / "vision_encoder/model.onnx")
    image = Image.new("RGB", (256, 256), color=(80, 120, 160))
    processed = processor.image_processor(images=[image], return_tensors="np")
    features = _run(
        vision,
        {
            "pixel_values": _cuda(processed["pixel_values"].astype(np.float32)),
            "image_grid_thw": _cuda(processed["image_grid_thw"].astype(np.int64)),
        },
    )["image_features"]
    feature_shape = _finite("vision features", features)
    merge = config["vision_config"]["spatial_merge_size"]
    expected_tokens = int(np.prod(processed["image_grid_thw"], axis=1).sum()) // merge**2
    if feature_shape != [expected_tokens, config["text_config"]["hidden_size"]]:
        raise RuntimeError(f"Unexpected image feature shape: {feature_shape}")
    image_ids = np.full((1, expected_tokens), config["image_token_id"], dtype=np.int64)
    fused = _run(embedding, {"input_ids": _cuda(image_ids), "image_features": features})[
        "inputs_embeds"
    ]
    np.testing.assert_array_equal(fused.numpy()[0], features.numpy())
    fusion_shape = _finite("image embedding fusion", fused)
    logger.info("Vision encoder and image embedding fusion passed")
    return {"vision_features": feature_shape, "image_embedding_fusion": fusion_shape}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--skip-mtp", action="store_true", help="Explicit decoder/vision-only probe"
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="Probe early decoder layers"
    )
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite smoke report: {args.report}")
    ort.preload_dlls()
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    config = json.loads((args.checkpoint / "config.json").read_text())
    hidden_size = config["text_config"]["hidden_size"]
    embedding = _session(args.model / "embedding/model.onnx")
    results = vision_smoke(args.model, processor, embedding, config)
    ids = np.array(
        [processor.tokenizer.encode("Hello!", add_special_tokens=False)], dtype=np.int64
    )
    dummy_features = np.zeros((1, hidden_size), dtype=np.float16)
    embeddings = _run(
        embedding, {"input_ids": _cuda(ids), "image_features": _cuda(dummy_features)}
    )["inputs_embeds"]
    decoder = _session(args.model / "decoder/model.onnx", diagnostics=args.diagnostics)
    states = _empty_states(decoder)
    length = ids.shape[1]
    prefill = _run(
        decoder,
        {
            "inputs_embeds": embeddings,
            "attention_mask": _cuda(np.ones((1, length), dtype=np.int64)),
            "position_ids": _cuda(np.broadcast_to(np.arange(length), (3, 1, length))),
            **states,
        },
    )
    results["decoder_prefill"] = _finite("prefill logits", prefill["logits"])
    _finite("MTP seed", prefill["mtp_seed"])
    _check_diagnostics(prefill)
    _check_states(prefill)
    next_id = prefill["logits"].numpy()[:, -1].argmax(-1).reshape(1, 1).astype(np.int64)
    next_embed = _run(
        embedding, {"input_ids": _cuda(next_id), "image_features": _cuda(dummy_features)}
    )["inputs_embeds"]
    states = {
        name: prefill[name.replace("past_key_values.", "present.", 1)] for name in states
    }
    decoded = _run(
        decoder,
        {
            "inputs_embeds": next_embed,
            "attention_mask": _cuda(np.ones((1, length + 1), dtype=np.int64)),
            "position_ids": _cuda(np.full((3, 1, 1), length, dtype=np.int64)),
            **states,
        },
    )
    _check_diagnostics(decoded)
    _check_states(decoded)
    results["decoder_decode"] = _finite("decode logits", decoded["logits"])
    cache_types = {
        name: value.data_type()
        for name, value in states.items()
        if name.endswith((".key", ".value"))
    }
    results["fp8_kv"] = cache_types
    if set(cache_types.values()) != {"tensor(float8e4m3fn)"}:
        raise RuntimeError("Decoder did not preserve FP8 KV cache storage")
    if not args.skip_mtp:
        mtp = _session(args.model / ".mobius-mtp/model.onnx")
        seed = prefill["mtp_seed"].numpy()[:, -1:, :]
        proposal = _run(
            mtp,
            {
                "input_ids": _cuda(next_id),
                "hidden_states": _cuda(seed),
                "attention_mask": _cuda(np.ones((1, 1), dtype=np.int64)),
                "position_ids": _cuda(np.full((1, 1), length, dtype=np.int64)),
                **_empty_states(mtp),
            },
        )
        results["mtp_forward"] = _finite("MTP logits", proposal["logits"])
    else:
        results["mtp_forward"] = "not run (--skip-mtp)"
    report = {
        "status": "component smoke passed",
        "engine_ready": False,
        "accuracy_test": False,
        "onnxruntime_build": ort.get_build_info(),
        "results": results,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not args.skip_mtp:
        manifest_path = args.model / "ornith_export.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["component_validation"] = report
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Component smoke passed; Engine/image generation and benchmarks remain unvalidated"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
