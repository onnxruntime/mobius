# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Real-weight L4/L5 tests for the staged VibeVoice TTS package."""

from __future__ import annotations

import dataclasses
import gc
import json
import os
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
import soundfile as sf
import torch

from mobius._testing.vibevoice_generation import (
    VibeVoiceDiskGenerator,
    _DiskSession,
)
from mobius.integrations.transformers._builder import (
    _load_transformers_config,
    _select_primary_config,
)
from mobius.integrations.transformers._config_resolver import _config_from_hf
from mobius.models.vibevoice import (
    VIBEVOICE_MODEL_ID,
    VIBEVOICE_REVISION,
    VibeVoiceForConditionalGeneration,
)

_ROOT = Path(__file__).parents[1]
_GOLDEN_DIR = _ROOT / "testdata" / "golden" / "audio"
_AUDIO_PATH = _ROOT / "testdata" / "652-129742-0006-24khz.wav"
_SEED = 20260831
_L4_CASES = [
    "vibevoice-1.5b-text-only",
    "vibevoice-1.5b-reference",
    "vibevoice-1.5b-multispeaker",
]
_L5_CASES = ["vibevoice-1.5b-text-only", "vibevoice-1.5b-reference"]


def _selected_cases(cases: list[str]) -> list[str]:
    """Select exact VibeVoice golden cases requested by the GPU workflow."""
    requested = os.environ.get("MOBIUS_VIBEVOICE_GOLDEN_CASES")
    if requested is None:
        return cases
    try:
        selected = json.loads(requested)
    except json.JSONDecodeError as error:
        raise ValueError("MOBIUS_VIBEVOICE_GOLDEN_CASES must be a JSON array") from error
    if not isinstance(selected, list) or not all(isinstance(case, str) for case in selected):
        raise ValueError("MOBIUS_VIBEVOICE_GOLDEN_CASES must be a JSON array of case IDs")
    unknown = sorted(set(selected) - set(_L4_CASES) - set(_L5_CASES))
    if unknown:
        raise ValueError(f"Unknown VibeVoice golden case(s): {unknown}")
    return [case for case in cases if case in selected]


def _config():
    hf_config, _ = _load_transformers_config(
        VIBEVOICE_MODEL_ID,
        revision=VIBEVOICE_REVISION,
        trust_remote_code=False,
    )
    primary, parent, _ = _select_primary_config(hf_config)
    return dataclasses.replace(
        _config_from_hf(
            primary,
            parent_config=parent,
            module_class=VibeVoiceForConditionalGeneration,
        ),
        dtype=ir.DataType.FLOAT16,
    )


@pytest.fixture(scope="session")
def vibevoice_package_dir(tmp_path_factory) -> Path:
    existing = os.environ.get("MOBIUS_VIBEVOICE_PACKAGE")
    if existing:
        path = Path(existing)
        if not (path / "decoder" / "model.onnx").is_file():
            raise FileNotFoundError(f"Invalid MOBIUS_VIBEVOICE_PACKAGE: {path}")
        return path

    from mobius import build

    path = tmp_path_factory.mktemp("vibevoice-package")
    package = build(
        VIBEVOICE_MODEL_ID,
        revision=VIBEVOICE_REVISION,
        dtype="f16",
        execution_provider="cuda",
        load_weights=True,
    )
    package.save(str(path), max_shard_size_bytes=2_000_000_000)
    return path


@pytest.fixture(scope="session")
def vibevoice_processor():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoProcessor.from_pretrained(
        VIBEVOICE_MODEL_ID,
        revision=VIBEVOICE_REVISION,
    )


def _case_inputs(processor, case: str) -> dict[str, torch.Tensor]:
    audio, sampling_rate = sf.read(_AUDIO_PATH)
    if case == "vibevoice-1.5b-text-only":
        prompts = ["Mobius verifies text-only speech."]
        audios = None
    elif case == "vibevoice-1.5b-reference":
        prompts = ["Mobius verifies deterministic speech."]
        audios = [audio]
    elif case == "vibevoice-1.5b-multispeaker":
        prompts = [
            "The first speaker opens the test.",
            "The second speaker completes it.",
        ]
        audios = [audio, audio]
    else:
        raise ValueError(f"Unknown VibeVoice golden case: {case}")

    conversation = []
    for index, prompt in enumerate(prompts):
        content = []
        if audios is not None:
            content.append({"type": "audio", "audio": audios[index]})
        content.append({"type": "text", "text": prompt})
        conversation.append({"role": str(index), "content": content})
    rendered = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    kwargs = {"text": rendered, "return_tensors": "pt"}
    if audios is not None:
        kwargs.update(audio=audios, sampling_rate=sampling_rate)
    return processor(**kwargs)


def _golden(case: str, *, generation: bool) -> dict:
    suffix = "_generation" if generation else ""
    with open(_GOLDEN_DIR / f"{case}{suffix}.json", encoding="utf-8") as file:
        return json.load(file)


def _run(
    package_dir: Path,
    processor,
    case: str,
    *,
    device: str,
    max_new_tokens: int,
):
    return VibeVoiceDiskGenerator(
        package_dir,
        _config(),
        device=device,
        rng_device="cuda" if torch.cuda.is_available() else "cpu",
    ).generate(
        _case_inputs(processor, case),
        seed=_SEED,
        max_new_tokens=max_new_tokens,
        num_diffusion_steps=2,
        guidance_scale=1.3,
    )


@pytest.mark.golden
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    _selected_cases(_L4_CASES),
)
def test_vibevoice_l4_real_weight_prefill(
    vibevoice_package_dir,
    vibevoice_processor,
    case,
):
    if case != "vibevoice-1.5b-text-only" and not torch.cuda.is_available():
        pytest.skip("Reference-audio goldens preserve the native CUDA RNG stream")
    expected = _golden(case, generation=False)
    actual = _run(
        vibevoice_package_dir,
        vibevoice_processor,
        case,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens=1,
    )
    assert expected["revision"] == VIBEVOICE_REVISION
    np.testing.assert_allclose(
        actual.prefill_control_logits,
        expected["prefill_control_logits"],
        atol=0.1,
        rtol=0.02,
    )


@pytest.mark.golden
@pytest.mark.generation
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    _selected_cases(_L5_CASES),
)
def test_vibevoice_l5_generation(
    vibevoice_package_dir,
    vibevoice_processor,
    case,
):
    if case != "vibevoice-1.5b-text-only" and not torch.cuda.is_available():
        pytest.skip("Reference-audio goldens preserve the native CUDA RNG stream")
    expected = _golden(case, generation=True)
    actual = _run(
        vibevoice_package_dir,
        vibevoice_processor,
        case,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens=expected["generated_token_count"],
    )

    assert actual.generated_tokens == expected["generated_tokens"]
    assert len(actual.generated_tokens) == expected["generated_token_count"]
    assert actual.audio_chunk_count == expected["audio_chunk_count"]
    assert list(actual.scaled_audio_latents.shape) == expected["continuous_latent_shape"]
    assert list(actual.waveform.shape) == expected["waveform_shape"]
    assert actual.waveform.shape[-1] == (
        actual.audio_chunk_count * _config().acoustic_tokenizer.hop_length
    )

    np.testing.assert_allclose(
        actual.scaled_audio_latents[0, 0, :16],
        expected["continuous_latent_summary"]["first_frame"],
        atol=0.03,
        rtol=0.08,
    )
    np.testing.assert_allclose(
        actual.waveform.ravel()[:64],
        expected["waveform_first_64"],
        atol=5e-5,
        rtol=0.15,
    )

    waveform = actual.waveform.astype(np.float64).ravel()
    assert np.isfinite(waveform).all()
    assert np.count_nonzero(waveform) / waveform.size > 0.999
    assert waveform.std() > 1e-5
    assert np.abs(waveform).max() > 1e-4
    assert np.count_nonzero(np.diff(np.signbit(waveform))) > 100
    expected_summary = expected["waveform_summary"]
    assert waveform.std() == pytest.approx(expected_summary["std"], rel=0.05)
    assert np.abs(waveform).max() == pytest.approx(expected_summary["peak"], rel=0.05)


@pytest.mark.golden
@pytest.mark.integration
def test_vibevoice_cpu_cuda_semantic_parity(
    vibevoice_package_dir,
    vibevoice_processor,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CPU/CUDA parity gate")
    case = "vibevoice-1.5b-text-only"
    cpu = _run(
        vibevoice_package_dir,
        vibevoice_processor,
        case,
        device="cpu",
        max_new_tokens=1,
    )
    cuda = _run(
        vibevoice_package_dir,
        vibevoice_processor,
        case,
        device="cuda",
        max_new_tokens=1,
    )
    assert cpu.generated_tokens == cuda.generated_tokens
    assert cpu.waveform.shape == cuda.waveform.shape == (1, 1, 3200)
    np.testing.assert_allclose(cpu.waveform, cuda.waveform, atol=5e-5, rtol=0.1)


@pytest.mark.integration
def test_vibevoice_processor_contract(vibevoice_processor):
    text = _case_inputs(vibevoice_processor, "vibevoice-1.5b-text-only")
    reference = _case_inputs(vibevoice_processor, "vibevoice-1.5b-reference")
    multi = _case_inputs(vibevoice_processor, "vibevoice-1.5b-multispeaker")

    assert set(text) == {"input_ids", "attention_mask"}
    assert tuple(text["input_ids"].shape) == (1, 38)
    assert tuple(reference["input_values"].shape) == (1, 1, 220800)
    assert tuple(reference["padding_mask"].shape) == (1, 220800)
    assert int((reference["input_ids"] == 151654).sum()) == 69
    assert tuple(multi["input_values"].shape) == (2, 1, 220800)
    assert int((multi["input_ids"] == 151654).sum()) == 138
    assert vibevoice_processor.feature_extractor.sampling_rate == 24000
    assert vibevoice_processor.feature_extractor.pad_to_multiple_of == 3200


@pytest.mark.integration
def test_vibevoice_real_weight_stage_parity(
    vibevoice_package_dir,
    vibevoice_processor,
):
    """Compare every real-weight stage, wiring ONNX outputs into the next stage."""
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForTextToWaveform.from_pretrained(
        VIBEVOICE_MODEL_ID,
        revision=VIBEVOICE_REVISION,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    config = _config()
    latent_scale = model.model.latent_scaling_factor.detach().item()
    latent_bias = model.model.latent_bias_factor.detach().item()

    def run_stage(name: str, feeds: dict[str, np.ndarray]):
        session = _DiskSession(vibevoice_package_dir / name / "model.onnx", "cuda")
        try:
            return session.run(feeds), session
        except Exception:
            session.close()
            raise

    audio, _ = sf.read(_AUDIO_PATH, frames=config.acoustic_tokenizer.hop_length)
    waveform = audio.astype(np.float32)[None, None]
    input_values = torch.from_numpy(waveform).to("cuda", torch.float16)
    encoder = model.model.audio_tower.encoder.to("cuda")
    with torch.no_grad():
        raw_latents = encoder(input_values).latents
    torch.manual_seed(_SEED)
    sample_noise = torch.randn(1, device="cuda", dtype=torch.float16)
    latent_noise = torch.randn_like(raw_latents)
    expected_latents = (
        raw_latents
        + config.acoustic_tokenizer.vae_std * sample_noise[:, None, None] * latent_noise
    )
    encoder.cpu()
    torch.cuda.empty_cache()

    encoder_output, session = run_stage(
        "audio_encoder",
        {
            "input_values": waveform,
            "padding_mask": np.ones(
                (1, config.acoustic_tokenizer.hop_length),
                dtype=np.bool_,
            ),
            "sample_noise": sample_noise.cpu().numpy(),
            "latent_noise": latent_noise.cpu().numpy(),
        },
    )
    session.close()
    np.testing.assert_allclose(
        encoder_output["audio_latents"],
        expected_latents[0].float().cpu().numpy(),
        atol=0.08,
        rtol=0.08,
    )

    projector = model.model.multi_modal_projector.to("cuda")
    with torch.no_grad():
        scaled = (expected_latents + latent_bias) * latent_scale
        expected_audio_embeds = projector(scaled).float().cpu().numpy()[0]
    projector.cpu()
    projection_output, session = run_stage(
        "audio_projection",
        {
            "audio_latents": encoder_output["audio_latents"],
            "latents_are_scaled": np.array(False),
        },
    )
    session.close()
    np.testing.assert_allclose(
        projection_output["audio_embeds"],
        expected_audio_embeds,
        atol=0.08,
        rtol=0.08,
    )

    input_ids = np.array([[1, config.audio_token_id, 2]], dtype=np.int64)
    embedding = model.model.language_model.embed_tokens.to("cuda")
    with torch.no_grad():
        expected_inputs_embeds = embedding(torch.from_numpy(input_ids).cuda())
        expected_inputs_embeds[:, 1] = torch.from_numpy(
            projection_output["audio_embeds"]
        ).cuda()
    embedding.cpu()
    embedding_output, session = run_stage(
        "embedding",
        {
            "input_ids": input_ids,
            "audio_embeds": projection_output["audio_embeds"],
            "replace_audio_tokens": np.array(True),
        },
    )
    session.close()
    np.testing.assert_array_equal(
        embedding_output["inputs_embeds"],
        expected_inputs_embeds.cpu().numpy(),
    )

    attention_mask = np.ones_like(input_ids)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None]
    language_model = model.model.language_model.to("cuda")
    lm_head = model.lm_head.to("cuda")
    with torch.no_grad():
        reference_decoder = language_model(
            inputs_embeds=torch.from_numpy(embedding_output["inputs_embeds"]).cuda(),
            attention_mask=torch.from_numpy(attention_mask).cuda(),
            position_ids=torch.from_numpy(position_ids).cuda(),
            use_cache=True,
        )
        expected_logits = lm_head(reference_decoder.last_hidden_state).float().cpu().numpy()
    language_model.cpu()
    lm_head.cpu()
    torch.cuda.empty_cache()
    decoder_feeds = {
        "inputs_embeds": embedding_output["inputs_embeds"],
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for layer in range(config.num_hidden_layers):
        for kind in ("key", "value"):
            decoder_feeds[f"past_key_values.{layer}.{kind}"] = np.zeros(
                (
                    1,
                    config.num_key_value_heads,
                    0,
                    config.head_dim,
                ),
                dtype=np.float16,
            )
    decoder_output, session = run_stage("decoder", decoder_feeds)
    session.close()
    np.testing.assert_allclose(
        decoder_output["logits"].astype(np.float32),
        expected_logits,
        atol=0.08,
        rtol=0.05,
    )

    rng = np.random.default_rng(17)
    noisy = rng.standard_normal((2, 64)).astype(np.float16)
    timesteps = np.array([500, 500], dtype=np.float16)
    condition = rng.standard_normal((2, config.hidden_size)).astype(np.float16)
    diffusion = model.model.diffusion_head.to("cuda")
    with torch.no_grad():
        expected_velocity = (
            diffusion(
                torch.from_numpy(noisy).cuda(),
                torch.from_numpy(timesteps).cuda(),
                torch.from_numpy(condition).cuda(),
            )
            .float()
            .cpu()
            .numpy()
        )
    diffusion.cpu()
    diffusion_output, session = run_stage(
        "diffusion_head",
        {
            "noisy_audio_latents": noisy,
            "timesteps": timesteps,
            "condition": condition,
        },
    )
    session.close()
    np.testing.assert_allclose(
        diffusion_output["velocity"].astype(np.float32),
        expected_velocity,
        atol=0.01,
        rtol=0.01,
    )

    generated_latent = rng.standard_normal((1, 1, 64)).astype(np.float16)
    acoustic_decoder = model.model.audio_tower.decoder.to("cuda")
    with torch.no_grad():
        unscaled = torch.from_numpy(generated_latent).cuda() / latent_scale - latent_bias
        expected_waveform = (
            acoustic_decoder(
                unscaled.transpose(1, 2),
                use_cache=True,
            )
            .audio.float()
            .cpu()
            .numpy()
        )
    acoustic_decoder.cpu()
    audio_session = _DiskSession(
        vibevoice_package_dir / "audio_decoder" / "model.onnx",
        "cuda",
    )
    audio_feeds = {"scaled_audio_latents": generated_latent}
    for name, value in audio_session.input_info.items():
        if name.startswith("past_conv."):
            shape = list(value.shape)
            shape[0] = 1
            audio_feeds[name] = np.zeros(shape, dtype=np.float16)
    audio_output = audio_session.run(audio_feeds)
    audio_session.close()
    np.testing.assert_allclose(
        audio_output["waveform"].astype(np.float32),
        expected_waveform,
        atol=0.001,
        rtol=0.02,
    )

    semantic_encoder = model.model.semantic_tokenizer_encoder.to("cuda")
    with torch.no_grad():
        expected_semantic = (
            semantic_encoder(
                torch.from_numpy(audio_output["waveform"]).cuda(),
                use_cache=True,
            )
            .latents.float()
            .cpu()
            .numpy()
        )
    semantic_encoder.cpu()
    semantic_session = _DiskSession(
        vibevoice_package_dir / "semantic_encoder" / "model.onnx",
        "cuda",
    )
    semantic_feeds = {"waveform": audio_output["waveform"]}
    for name, value in semantic_session.input_info.items():
        if name.startswith("past_conv."):
            shape = list(value.shape)
            shape[0] = 1
            semantic_feeds[name] = np.zeros(shape, dtype=np.float16)
    semantic_output = semantic_session.run(semantic_feeds)
    semantic_session.close()
    np.testing.assert_allclose(
        semantic_output["semantic_latents"].astype(np.float32),
        expected_semantic,
        atol=0.02,
        rtol=0.05,
    )

    semantic_projector = model.model.semantic_connector.to("cuda")
    with torch.no_grad():
        expected_semantic_embeds = (
            semantic_projector(torch.from_numpy(semantic_output["semantic_latents"]).cuda())
            .float()
            .cpu()
            .numpy()
        )
    semantic_projector.cpu()
    semantic_projection, session = run_stage(
        "semantic_projection",
        {"semantic_latents": semantic_output["semantic_latents"]},
    )
    session.close()
    np.testing.assert_allclose(
        semantic_projection["semantic_embeds"].astype(np.float32),
        expected_semantic_embeds,
        atol=0.02,
        rtol=0.05,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
