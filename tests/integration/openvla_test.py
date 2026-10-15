"""Replay independently captured original-class OpenVLA FP32 CPU evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from safetensors.torch import load_file
from transformers import LlamaConfig

from mobius import build_from_module
from mobius._configs import ArchitectureConfig, OpenVLAConfig, VisionConfig
from mobius.models import OpenVLAModel


@pytest.mark.integration
def test_original_class_tiny_reference_through_ort(tmp_path):
    root_value = os.environ.get("MOBIUS_OPENVLA_REFERENCE")
    if not root_value:
        pytest.skip("set MOBIUS_OPENVLA_REFERENCE to independently captured tiny-fixture")
    ort = pytest.importorskip("onnxruntime")
    root = Path(root_value)
    manifest = json.loads((root / "fixture.json").read_text())
    assert (
        manifest["status"]
        == "DETERMINISTIC_RANDOMIZED_ORIGINAL_CLASS_FIXTURE_NOT_CHECKPOINT_PARITY"
    )
    assert manifest["source_revision"] == "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
    assert manifest["reviewed_source_hashes"]["modeling_prismatic.py"] == (
        "9ce241c5ca09a4bed73654d0ca509baaff0fc5887d0bdd62e360ca8254f7794a"
    )
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == entry["sha256"]
    raw = json.loads((root / "source-config.json").read_text())
    text = ArchitectureConfig.from_transformers(LlamaConfig(**raw["text_config"]))
    config = OpenVLAConfig(
        **{field.name: getattr(text, field.name) for field in dataclasses.fields(text)}
    )
    assert raw["text_config"]["rope_scaling"] is None
    config.rope_scaling = None
    config.dtype = ir.DataType.FLOAT
    for name in ("dino", "siglip"):
        tower = manifest["vision"][name]
        setattr(
            config,
            name,
            VisionConfig(
                hidden_size=tower["width"],
                intermediate_size=tower["mlp_hidden_size"],
                num_hidden_layers=tower["depth"],
                num_attention_heads=tower["heads"],
                image_size=manifest["vision"]["common"]["image_size"],
                patch_size=manifest["vision"]["common"]["patch_size"],
                norm_eps=1e-6,
            ),
        )
    module = OpenVLAModel(config)
    package = build_from_module(module, config, task="openvla", execution_provider="default")
    package.apply_weights(
        module.preprocess_weights(load_file(str(root / "source-model.safetensors")))
    )
    package.save(str(tmp_path), progress_bar=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    sessions = {
        name: ort.InferenceSession(
            str(tmp_path / name / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for name in package
    }
    with np.load(root / "inputs.npz", allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    with np.load(root / "source-outputs.npz", allow_pickle=False) as archive:
        expected = {name: archive[name] for name in archive.files}

    def run(name, feed):
        session = sessions[name]
        return dict(
            zip(
                [value.name for value in session.get_outputs()],
                session.run(None, feed),
                strict=True,
            )
        )

    def compare(name, actual):
        np.testing.assert_allclose(actual, expected[name], atol=1e-6, rtol=1e-5, err_msg=name)

    features = run("vision_encoder", {"pixel_values": inputs["pixel_values"]})[
        "image_features"
    ]
    compare("projected_features", features)
    vision = ir.load(tmp_path / "vision_encoder" / "model.onnx")
    fusion = [
        node
        for node in vision.graph
        if node.op_type == "Concat"
        and node.metadata_props.get("namespace", "").endswith("vision_backbone: _FusedVision")
    ]
    assert len(fusion) == 1
    boundaries = [*fusion[0].inputs, fusion[0].outputs[0]]
    vision.graph.outputs.extend(boundaries)
    ir.save(vision, tmp_path / "vision_encoder" / "instrumented.onnx")
    instrumented = ort.InferenceSession(
        str(tmp_path / "vision_encoder" / "instrumented.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    outputs = instrumented.run(None, {"pixel_values": inputs["pixel_values"]})
    np.testing.assert_array_equal(outputs[0], features)
    for name, value in zip(
        ("dino_patches", "siglip_patches", "fused_patches"), outputs[1:], strict=True
    ):
        compare(name, value)

    embedded = run(
        "embedding", {"input_ids": inputs["input_ids"], "image_features": features}
    )["inputs_embeds"]
    compare("bos_inserted_embeddings", embedded)
    feed = {
        "inputs_embeds": embedded,
        "attention_mask": expected["multimodal_attention_mask"],
        "position_ids": np.arange(embedded.shape[1], dtype=np.int64)[None, :],
    }
    for layer in range(config.num_hidden_layers):
        for kind in ("key", "value"):
            feed[f"past_key_values.{layer}.{kind}"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
            )
    prefill = run("decoder", feed)
    compare("prefill_logits", prefill["logits"])
    for layer in range(config.num_hidden_layers):
        for kind in ("key", "value"):
            compare(f"prefill_{kind}_{layer}", prefill[f"present.{layer}.{kind}"])
    decode_embeds = run(
        "embedding",
        {
            "input_ids": inputs["decode_input_ids"],
            "image_features": np.zeros((1, 0, config.hidden_size), dtype=np.float32),
        },
    )["inputs_embeds"]
    feed = {
        "inputs_embeds": decode_embeds,
        "attention_mask": np.ones((1, embedded.shape[1] + 1), dtype=np.int64),
        "position_ids": np.array([[embedded.shape[1]]], dtype=np.int64),
    }
    for layer in range(config.num_hidden_layers):
        for kind in ("key", "value"):
            feed[f"past_key_values.{layer}.{kind}"] = prefill[f"present.{layer}.{kind}"]
    decoded = run("decoder", feed)
    compare("decode_logits", decoded["logits"])
    for layer in range(config.num_hidden_layers):
        for kind in ("key", "value"):
            compare(f"decode_{kind}_{layer}", decoded[f"present.{layer}.{kind}"])
