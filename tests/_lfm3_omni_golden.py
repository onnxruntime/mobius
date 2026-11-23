# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned real-checkpoint LFM3-Omni parity and replay of recorded MaskGIT orders."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from unittest import mock

import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mobius import ModelPackage
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.torch_reference import (
    lfm3_vl_reference_compatibility,
    restore_lfm3_reference_layer_types,
)
from mobius.integrations.lfm3_audio import build_lfm3_audio_package, load_generation_config
from mobius.integrations.lfm3_omni import Lfm3OmniPipeline
from tests._lfm3_audio_golden import _device, _verify_codec

MODEL_ID = "LiquidAI/LFM3-Omni-2B"
REVISION = "21d75123f86c1f7ffb4bef1ea202710c4e7ebe67"
ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "testdata" / "golden" / "multimodal"
UPDATE = "MOBIUS_LFM3_OMNI_UPDATE_GOLDEN"
# System prompts of the model card's usage examples.
IMAGE_SYSTEM = (
    "Answer questions about the provided image. Base every answer only on what can be seen."
)
TTS_SYSTEM = "Perform TTS in American English using a female voice."
VOICE_CHAT_SYSTEM = "Respond with interleaved text and audio."


def _golden(name, value=None):
    path = GOLDENS / f"{name}.json"
    if value is not None and os.environ.get(UPDATE) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pinned LFM3-Omni reference {path.name}; regenerate it on a GPU "
            f"with {UPDATE}=1, then commit it."
        )
    saved = json.loads(path.read_text(encoding="utf-8"))
    if value is not None:
        assert saved == value, f"Pinned upstream reference changed for {path.name}"
    return saved


def _reference():
    with lfm3_vl_reference_compatibility(MODEL_ID, REVISION):
        config = AutoConfig.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True
        )
        restore_lfm3_reference_layer_types(config.text_config)
        restore_lfm3_reference_layer_types(config.depth_config)
        encoder = config.audio_encoder_config.text_config
        encoder.layer_types = [
            "conv" if kind == "linear_attention" else kind for kind in encoder.layer_types
        ]
        model_class = get_class_from_dynamic_module(
            "modeling_lfm3_omni.Lfm3OmniForConditionalGeneration", MODEL_ID, revision=REVISION
        )
        model = model_class.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            config=config,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        for module in model.modules():
            if hasattr(module, "config"):
                restore_lfm3_reference_layer_types(module.config)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    )
    return model, processor


def _prepare(device, tmp_path):
    package = build_lfm3_audio_package(
        MODEL_ID,
        revision=REVISION,
        execution_provider="cuda" if device == "cuda" else "cpu",
    )
    assert set(package) == {
        "vision_encoder",
        "audio_encoder",
        "embedding",
        "decoder",
        "depth_decoder",
        "audio_decoder",
    }
    config = package.config
    package.save(tmp_path / "package")
    del package
    package = ModelPackage.load(tmp_path / "package")
    sessions = {}
    try:
        for name, graph in package.items():
            sessions[name] = OnnxModelSession(graph, device=device)
            if device == "cuda":
                assert "CUDAExecutionProvider" in sessions[name].providers
    except BaseException:
        for session in sessions.values():
            session.close()
        raise
    generation_config = load_generation_config(MODEL_ID, revision=REVISION)
    return Lfm3OmniPipeline(config, sessions, generation_config), sessions


def _clip():
    """The LibriSpeech fixture at the 16 kHz the feature extractor requires."""
    waveform, sample_rate = sf.read(ROOT / "testdata" / "652-129742-0006-24khz.wav")
    return librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000).astype(np.float32)


def _image(name):
    return Image.open(ROOT / "testdata" / name).convert("RGB")


def _inputs(processor, system, content):
    return processor.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"sampling_rate": 16000},
    )


def _feeds(inputs):
    """Processor tensors as the pipeline's float32 and int64 arrays."""
    feeds = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy(),
    }
    if "input_features" in inputs:
        feeds["input_features"] = inputs["input_features"].float().cpu().numpy()
        feeds["feature_lengths"] = (
            inputs["feature_attention_mask"].sum(-1).long().cpu().numpy()
        )
    if "pixel_values" in inputs:
        feeds["pixel_values"] = inputs["pixel_values"].float().cpu().numpy()
        feeds["image_grid_thw"] = inputs["image_grid_thw"].cpu().numpy()
    return feeds


def _generate(model, inputs, device, stored_orders=None, **kwargs):
    """Reference generate with recorded codebook orders, or replay of stored ones."""
    orders = []
    randperm = torch.randperm

    def record_order(*args, **order_kwargs):
        if stored_orders is None:
            permutation = randperm(*args, **order_kwargs)
        else:
            # CPU and CUDA RNG streams differ: replay the recorded upstream permutations.
            assert len(orders) < len(stored_orders), "Upstream requested extra speech frames."
            permutation = torch.tensor(
                stored_orders[len(orders)], device=order_kwargs.get("device")
            )
        orders.append(permutation.cpu().tolist())
        return permutation

    torch.manual_seed(42)
    with torch.no_grad(), mock.patch("torch.randperm", side_effect=record_order):
        output = model.generate(**inputs.to(device), return_dict_in_generate=True, **kwargs)
    # Speech and interleaved modes append their opening marker to the prompt.
    width = inputs["input_ids"].shape[1] + (
        kwargs.get("generation_mode") in ("audio", "interleaved")
    )
    tokens = output.sequences[0, width : int(output.sequence_lengths[0])].cpu().numpy()
    codes = output.audio_codes[:, :, : int(output.audio_lengths[0])].cpu().numpy()
    return tokens, codes, orders


def _close(sessions, device):
    for session in sessions.values():
        session.close()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def run_lfm3_omni_l4(tmp_path):
    """Image features and prefill logits of the model card's spoken question about an image."""
    device = _device()
    model, processor = _reference()
    inputs = _inputs(
        processor,
        IMAGE_SYSTEM + " Respond with text.",
        [
            {"type": "image", "image": _image("pipeline-cat-chonk.jpeg")},
            {"type": "audio", "audio": _clip()},
        ],
    )
    pipeline, sessions = _prepare(device, tmp_path)
    try:
        model.to(device)
        inputs = inputs.to(device)
        with torch.no_grad():
            expected_images = (
                model.model.get_image_features(
                    inputs["pixel_values"], inputs["image_grid_thw"]
                )
                .float()
                .cpu()
                .numpy()
            )
            expected = model(**inputs, use_cache=False).logits.float().cpu().numpy()
        feeds = _feeds(inputs)
        actual_images = sessions["vision_encoder"].run(
            {"pixel_values": feeds["pixel_values"], "image_grid_thw": feeds["image_grid_thw"]}
        )["image_features"]
        np.testing.assert_allclose(actual_images, expected_images, rtol=1e-3, atol=1e-3)
        actual = pipeline.prefill(**feeds)["logits"]
        assert actual.dtype == expected.dtype == np.float32
        # fp32 noise in the image features grows at the image placeholder positions, where the
        # reference's own CPU and CUDA logits differ too; generation reads none of them.
        image = feeds["input_ids"][0] == pipeline.config.image_token_id
        np.testing.assert_allclose(
            actual[:, ~image], expected[:, ~image], rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(actual[:, image], expected[:, image], rtol=1e-3, atol=0.5)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
        _golden(
            "lfm3-omni-2b",
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "dtype": "float32",
                "fixtures": ["pipeline-cat-chonk.jpeg", "652-129742-0006-24khz.wav"],
                "input_ids": inputs["input_ids"].cpu().tolist(),
                "image_grid_thw": inputs["image_grid_thw"].cpu().tolist(),
                "logits_shape": list(expected.shape),
                "prefill_argmax": expected.argmax(-1).tolist(),
            },
        )
    finally:
        del model
        _close(sessions, device)


def run_lfm3_omni_l5(tmp_path):
    """Greedy text, then speech and interleaved answers with greedy codes and recorded orders."""
    device = _device()
    model, processor = _reference()
    pipeline, sessions = _prepare(device, tmp_path)
    stored = None if os.environ.get(UPDATE) == "1" else _golden("lfm3-omni-2b_generation")
    try:
        model.to(device)
        question = _inputs(
            processor,
            IMAGE_SYSTEM,
            [
                {"type": "image", "image": _image("pipeline-cat-chonk.jpeg")},
                {"type": "text", "text": "What is in this image?"},
            ],
        )
        expected_text, _, _ = _generate(
            model, question, device, generation_mode="text", max_new_tokens=64
        )
        actual_text, _ = pipeline.generate(
            max_new_tokens=64, generation_mode="text", **_feeds(question)
        )
        assert actual_text.shape == expected_text.shape
        np.testing.assert_array_equal(actual_text, expected_text)
        answer = processor.decode(expected_text, skip_special_tokens=True)
        assert answer.strip(), "The image question must produce a nonempty answer."

        # Greedy codes under the shipped repetition penalty, so the orders fix every frame.
        pipeline.generation_config = {**pipeline.generation_config, "audio_temperature": 0.0}
        results = {}
        for name, inputs, mode in (
            (
                "tts",
                _inputs(processor, TTS_SYSTEM, "The quick brown fox jumps over the lazy dog."),
                "audio",
            ),
            (
                "voice_chat",
                _inputs(processor, VOICE_CHAT_SYSTEM, [{"type": "audio", "audio": _clip()}]),
                "interleaved",
            ),
        ):
            budget = 256 if mode == "audio" else 1024
            tokens, codes, orders = _generate(
                model,
                inputs,
                device,
                None if stored is None else stored[f"{name}_codebook_orders"],
                generation_mode=mode,
                max_new_tokens=budget,
                audio_temperature=0.0,
            )
            assert codes.shape[-1] > 20, f"{name} must produce real speech frames."
            assert len(tokens) < budget, f"{name} must end within {budget} slots."
            actual_tokens, actual_codes = pipeline.generate(
                max_new_tokens=budget,
                generation_mode=mode,
                codebook_orders=orders,
                **_feeds(inputs),
            )
            assert actual_tokens.shape == tokens.shape
            np.testing.assert_array_equal(actual_tokens, tokens)
            assert actual_codes.shape == codes.shape
            np.testing.assert_array_equal(actual_codes, codes)
            results[name] = (inputs, tokens, codes, orders)
        _golden(
            "lfm3-omni-2b_generation",
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "dtype": "float32",
                "question_input_ids": question["input_ids"].cpu().tolist(),
                "question_generated_ids": expected_text.tolist(),
                "answer": answer,
                **{
                    key: value
                    for name, (inputs, tokens, codes, orders) in results.items()
                    for key, value in (
                        (f"{name}_input_ids", inputs["input_ids"].cpu().tolist()),
                        (f"{name}_generated_ids", tokens.tolist()),
                        (f"{name}_codebook_orders", orders),
                        (f"{name}_audio_codes", codes.tolist()),
                    )
                },
            },
        )

        # Release the language model before decoding the speech with the codec.
        for name in list(sessions):
            if name != "audio_decoder":
                sessions.pop(name).close()
        del pipeline, model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        _verify_codec(
            results["tts"][2], device, tmp_path, sessions["audio_decoder"], MODEL_ID, REVISION
        )
    finally:
        _close(sessions, device)
