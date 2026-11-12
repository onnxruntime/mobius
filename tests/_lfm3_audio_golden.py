# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned real-checkpoint audio parity and deterministic MaskGIT replay."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mobius import ModelPackage
from mobius._configs.lfm3_audio import Lfm3AudioConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.torch_reference import (
    lfm3_reference_compatibility,
    restore_lfm3_reference_layer_types,
)
from mobius.integrations.lfm3_audio import (
    Lfm3AudioPipeline,
    build_lfm3_audio_package,
)

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
REVISION = "2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1"
ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "testdata" / "golden" / "audio"


def _golden(name, value=None):
    path = GOLDENS / f"{name}.json"
    if value is not None and os.environ.get("MOBIUS_LFM3_AUDIO_UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pinned LFM3 Audio reference {path.name}; explicitly run the "
            "L4 GPU workflow with update_lfm3_audio_goldens=true, then commit its artifacts."
        )
    saved = json.loads(path.read_text(encoding="utf-8"))
    if value is not None:
        assert saved == value, f"Pinned upstream reference changed for {path.name}"
    return saved


def _reference(device):
    with lfm3_reference_compatibility():
        config = AutoConfig.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True
        )
        restore_lfm3_reference_layer_types(config.text_config)
        restore_lfm3_reference_layer_types(config.depth_config)
        config.audio_encoder_config.text_config.layer_types = [
            "conv" if kind == "linear_attention" else kind
            for kind in config.audio_encoder_config.text_config.layer_types
        ]
        model_class = get_class_from_dynamic_module(
            "modeling_lfm3_audio.Lfm3AudioForConditionalGeneration",
            MODEL_ID,
            revision=REVISION,
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


def _prepare(model, device, tmp_path):
    config = Lfm3AudioConfig.from_transformers(model.config)
    package = build_lfm3_audio_package(
        MODEL_ID,
        revision=REVISION,
        execution_provider="cuda" if device == "cuda" else "cpu",
    )
    assert set(package) == {
        "audio_encoder",
        "embedding",
        "decoder",
        "depth_decoder",
        "audio_decoder",
    }
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
    return Lfm3AudioPipeline(config, sessions), sessions


def _device():
    device = os.environ.get("MOBIUS_TEST_DEVICE", "cpu")
    if device == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA requested but unavailable; do not fall back to CPU."
        )
        properties = torch.cuda.get_device_properties(0)
        memory_gib = properties.total_memory / 1024**3
        print(
            f"GPU evidence: {properties.name}; VRAM={memory_gib:.2f} GiB; "
            f"torch={torch.__version__}"
        )
        if memory_gib < 20:
            raise RuntimeError(
                f"Full float32 LFM3 Audio reference/ORT validation needs approximately "
                f"20 GiB VRAM, but {properties.name} exposes only {memory_gib:.2f} GiB. "
                "Use an existing larger GPU partition or an explicitly approved reduced "
                "real-weight fixture; this test never substitutes CPU or reduced precision."
            )
    return device


def _asr_inputs(processor):
    import librosa

    waveform, sample_rate = sf.read(ROOT / "testdata" / "652-129742-0006-24khz.wav")
    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000).astype(
        np.float32
    )
    return processor.apply_chat_template(
        [
            {"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio", "audio": waveform}]},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        sampling_rate=16000,
    )


def _feeds(inputs):
    return {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy(),
        "input_features": inputs["input_features"].float().cpu().numpy(),
        "feature_lengths": inputs["feature_attention_mask"].sum(-1).long().cpu().numpy(),
    }


def run_lfm3_audio_l4(tmp_path):
    device = _device()
    model, processor = _reference(device)
    inputs = _asr_inputs(processor)
    pipeline, sessions = _prepare(model, device, tmp_path)
    try:
        model.to(device)
        with torch.no_grad():
            expected = model(**inputs.to(device), use_cache=False).logits.float().cpu().numpy()
        actual = pipeline.prefill(**_feeds(inputs))["logits"]
        assert actual.dtype == expected.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
        value = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "dtype": "float32",
            "fixture": "652-129742-0006-24khz.wav",
            "input_ids": inputs["input_ids"].cpu().tolist(),
            "logits_shape": list(expected.shape),
            "prefill_argmax": expected.argmax(-1).tolist(),
        }
        _golden("lfm3-audio-2b", value)
    finally:
        for session in sessions.values():
            session.close()
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def run_lfm3_audio_l5(tmp_path):
    device = _device()
    model, processor = _reference(device)
    pipeline, sessions = _prepare(model, device, tmp_path)
    try:
        model.to(device)
        asr = _asr_inputs(processor)
        with torch.no_grad():
            expected_text = (
                model.generate(
                    **asr.to(device),
                    max_new_tokens=96,
                    do_sample=False,
                    repetition_penalty=1.0,
                    suppress_mixed=True,
                    suppress_audio_out=True,
                    return_dict_in_generate=True,
                )
                .sequences[:, asr["input_ids"].shape[1] :]
                .cpu()
                .numpy()[0]
            )
        pipeline.prefill(**_feeds(asr))
        actual_text = pipeline.generate_text(96)
        assert actual_text.shape == expected_text.shape
        np.testing.assert_array_equal(actual_text, expected_text)
        transcript = processor.decode(expected_text, skip_special_tokens=True)
        assert transcript.strip(), (
            "The real speech fixture must produce a nonempty transcript."
        )

        speech = processor.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Perform TTS in American English using a female voice.",
                },
                {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
                {"role": "assistant", "content": "<|audio_start|>"},
            ],
            continue_final_message=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        orders = []
        randperm = torch.randperm
        stored_orders = (
            None
            if os.environ.get("MOBIUS_LFM3_AUDIO_UPDATE_GOLDEN") == "1"
            else _golden("lfm3-audio-2b_generation")["codebook_orders"]
        )

        def record_order(*args, **kwargs):
            if stored_orders is None:
                permutation = randperm(*args, **kwargs)
            else:
                # CPU and CUDA RNG streams differ: replay the independently
                # recorded upstream permutations, not a device-dependent seed.
                assert len(orders) < len(stored_orders), (
                    "Upstream requested extra speech frames."
                )
                assert args[0] == len(stored_orders[len(orders)])
                permutation = torch.tensor(
                    stored_orders[len(orders)],
                    device=kwargs.get("device"),
                    dtype=kwargs.get("dtype", torch.int64),
                )
            orders.append(permutation.cpu().tolist())
            return permutation

        torch.manual_seed(42)
        with torch.no_grad(), mock.patch("torch.randperm", side_effect=record_order):
            generated = model.generate(
                **speech.to(device),
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.0,
                audio_temperature=0.0,
                audio_top_k=0,
                audio_repetition_penalty=1.0,
                return_dict_in_generate=True,
            )
        expected_codes = generated.audio_codes.cpu().numpy()
        expected_length = int(generated.audio_lengths[0])
        expected_codes = expected_codes[:, :, :expected_length]
        assert 20 <= expected_length < 128, (
            "Speech L5 must produce at least 20 real frames and terminate within 128 decisions."
        )
        pipeline.prefill(
            speech["input_ids"].cpu().numpy(),
            attention_mask=speech["attention_mask"].cpu().numpy(),
        )
        with torch.no_grad():
            hidden = model.model(**speech.to(device), use_cache=False).last_hidden_state[:, -1]
            depth_reference = (
                model.model.depth_decoder(
                    hidden,
                    torch.zeros(
                        (1, pipeline.config.num_codebooks), dtype=torch.int64, device=device
                    ),
                    torch.ones(
                        (1, pipeline.config.num_codebooks), dtype=torch.bool, device=device
                    ),
                )
                .cpu()
                .numpy()
            )
        depth_actual = sessions["depth_decoder"].run(
            {
                "hidden_states": pipeline.hidden_states[:, -1],
                "codes": np.zeros((1, pipeline.config.num_codebooks), dtype=np.int64),
                "masked": np.ones((1, pipeline.config.num_codebooks), dtype=np.bool_),
            }
        )["logits"]
        np.testing.assert_allclose(depth_actual, depth_reference, rtol=1e-3, atol=1e-3)
        # A terminal frame may stop before the requested budget, so replay only
        # the number of frame decisions independently made by upstream.
        actual_codes = pipeline.generate_speech(len(orders), codebook_orders=orders)
        assert actual_codes.shape == expected_codes.shape
        np.testing.assert_array_equal(actual_codes, expected_codes)
        _golden(
            "lfm3-audio-2b_generation",
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "dtype": "float32",
                "asr_generated_ids": expected_text.tolist(),
                "transcript": transcript,
                "tts_input_ids": speech["input_ids"].cpu().tolist(),
                "codebook_orders": orders,
                "audio_codes": expected_codes.tolist(),
                "audio_frames": expected_length,
                "depth_prefill_argmax": depth_reference.argmax(-1).tolist(),
            },
        )

        # Release the large language model before loading the separate 147MB codec.
        for name in list(sessions):
            if name != "audio_decoder":
                sessions.pop(name).close()
        del pipeline, model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        _verify_codec(actual_codes, device, tmp_path, sessions["audio_decoder"])
    finally:
        for session in sessions.values():
            session.close()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def _verify_codec(codes, device, tmp_path, session):
    config_class = get_class_from_dynamic_module(
        "configuration_liquid_codec.LiquidCodecConfig", MODEL_ID, revision=REVISION
    )
    codec_class = get_class_from_dynamic_module(
        "modeling_liquid_codec.LiquidCodecModel", MODEL_ID, revision=REVISION
    )
    config = config_class.from_pretrained(
        MODEL_ID, revision=REVISION, subfolder="audio_tokenizer"
    )
    reference = codec_class.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        subfolder="audio_tokenizer",
        config=config,
        dtype=torch.float32,
    ).eval()
    restore_lfm3_reference_layer_types(reference.decoder.lfm.config)
    reference.to(device)
    with torch.no_grad():
        expected = (
            reference.decode(torch.from_numpy(codes).to(device)).audio_values.cpu().numpy()
        )
    try:
        if device == "cuda":
            assert "CUDAExecutionProvider" in session.providers
        actual = session.run({"codes": codes})["waveform"]
    finally:
        session.close()
    assert actual.shape == expected.shape == (1, codes.shape[-1] * 1920)
    assert np.isfinite(actual).all() and np.sqrt(np.mean(actual**2)) > 1e-4
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    sf.write(tmp_path / "lfm3_audio_synthesis.wav", actual[0], 24000)
