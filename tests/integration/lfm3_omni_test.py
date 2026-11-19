# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned upstream numerical checks for LiquidAI LFM3-Omni on a tiny random-weight model."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from mobius import build_from_module
from mobius._configs.lfm3_omni import Lfm3OmniConfig
from mobius._testing.torch_reference import (
    lfm3_vl_reference_compatibility,
    restore_lfm3_reference_layer_types,
)
from mobius.integrations.lfm3_omni import Lfm3OmniPipeline
from mobius.models.lfm3_omni import Lfm3OmniForConditionalGeneration
from tests.integration._support import _make_session

MODEL_ID = "LiquidAI/LFM3-Omni-2B"
REVISION = "21d75123f86c1f7ffb4bef1ea202710c4e7ebe67"
AUDIO, AUDIO_OUT, AUDIO_START, TEXT_END, MIXED_START, IMAGE, VIDEO = range(240, 247)
EOS = 2


def _tiny_reference():
    with lfm3_vl_reference_compatibility(MODEL_ID, REVISION):
        config_class, model_class = (
            get_class_from_dynamic_module(name, MODEL_ID, revision=REVISION)
            for name in (
                "configuration_lfm3_omni.Lfm3OmniConfig",
                "modeling_lfm3_omni.Lfm3OmniForConditionalGeneration",
            )
        )
        config = config_class(
            text_config=dict(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                vocab_size=256,
                max_position_embeddings=128,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=EOS,
                layer_types=["conv", "full_attention", "sliding_attention"],
                sliding_window=8,
            ),
            audio_encoder_config=dict(
                hidden_size=32,
                conv_channels=4,
                sliding_window=3,
                text_config=dict(
                    model_type="lfm2",
                    hidden_size=32,
                    intermediate_size=64,
                    block_auto_adjust_ff_dim=False,
                    num_hidden_layers=2,
                    layer_types=["conv", "full_attention"],
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=128,
                    vocab_size=2,
                    conv_L_cache=3,
                    rope_parameters={"rope_type": "default", "rope_theta": 1e6},
                ),
            ),
            depth_config=dict(
                hidden_size=32,
                intermediate_size=64,
                head_dim=8,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=1,
                layer_types=["full_attention"],
                max_position_embeddings=3,
                rope_parameters={
                    "full_attention": {
                        "rope_type": "default",
                        "rope_theta": 1e6,
                        "partial_rotary_factor": 1.0,
                    }
                },
                sliding_window=None,
                use_sandwich_norm=False,
                vocab_size=1,
            ),
            vision_config=dict(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                layer_types=["conv", "full_attention", "conv"],
                patch_size=4,
                rope_parameters={"rope_type": "axial", "rope_theta": 1e6},
            ),
            num_codebooks=3,
            codebook_size=17,
            audio_eos_code=16,
            maskgit_schedule=[1, 1, 1],
            interleaved_n_text=2,
            interleaved_n_audio=3,
            audio_token_id=AUDIO,
            audio_out_token_id=AUDIO_OUT,
            audio_start_token_id=AUDIO_START,
            text_end_token_id=TEXT_END,
            mixed_start_token_id=MIXED_START,
            image_token_id=IMAGE,
            video_token_id=VIDEO,
            projector_hidden_size=96,
        )
        for sub_config in (
            config,
            config.text_config,
            config.depth_config,
            config.audio_encoder_config.text_config,
        ):
            sub_config._attn_implementation = "eager"
        restore_lfm3_reference_layer_types(config.text_config)
        restore_lfm3_reference_layer_types(config.depth_config)
        encoder = config.audio_encoder_config.text_config
        encoder.layer_types = [
            "conv" if kind == "linear_attention" else kind for kind in encoder.layer_types
        ]
        torch.manual_seed(42)
        model = model_class(config).eval()
    for module in model.modules():
        if hasattr(module, "config"):
            restore_lfm3_reference_layer_types(module.config)
    with torch.no_grad():
        # Norm weights and biases start at constants; vary them so misuse shows.
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.uniform_(0.5, 1.5)
        # At the 0.02 init the vision tower's gated conv branch stays under atol.
        for parameter in [
            *model.model.vision_tower.parameters(),
            *model.model.multi_modal_projector.parameters(),
        ]:
            if parameter.ndim > 1:
                parameter.normal_(0.0, parameter[0].numel() ** -0.5)
        # A zero end-of-speech logit keeps greedy speech from ending at the first frame.
        for codebook in model.model.depth_decoder.depth_embeddings:
            codebook.embedding.weight[config.audio_eos_code] = 0.0
    return model


def _pipeline(reference, generation_config=None):
    config = Lfm3OmniConfig.from_transformers(reference.config)
    module = Lfm3OmniForConditionalGeneration(config)
    package = build_from_module(module, config, task="lfm3-omni")
    package.apply_weights(module.preprocess_weights(reference.state_dict()))
    sessions = {name: _make_session(model) for name, model in package.items()}
    return Lfm3OmniPipeline(config, sessions, generation_config), sessions


def _positions(mask):
    return (np.cumsum(mask, axis=1) - 1).clip(0)


@pytest.mark.integration
def test_omni_prefill_and_cached_decode_match_upstream():
    reference = _tiny_reference()
    pipeline, sessions = _pipeline(reference)
    rng = np.random.default_rng(42)
    # Two images ([t, h, w] patch grids), two clips of unequal length, two prompt speech frames.
    grids = np.array([[1, 4, 4], [1, 2, 6]], dtype=np.int64)
    pixels = rng.standard_normal((int(grids.prod(-1).sum()), 96)).astype(np.float32)
    features = rng.normal(size=(2, 128, 17)).astype(np.float32)
    lengths = np.array([17, 9], dtype=np.int64)
    out_codes = rng.integers(0, 16, size=(2, 3), dtype=np.int64)
    # Row 1 is left-padded; each row holds one image, one clip and one frame.
    ids = np.array(
        [
            [1, 5, *[IMAGE] * 4, *[AUDIO] * 3, 7, AUDIO_OUT, 9],
            [0, 0, 1, 8, *[IMAGE] * 3, *[AUDIO] * 2, 6, AUDIO_OUT, 10],
        ],
        dtype=np.int64,
    )
    mask = (np.arange(ids.shape[1]) >= np.array([[0], [2]])).astype(np.int64)
    try:
        with torch.no_grad():
            expected_images = reference.model.get_image_features(
                torch.from_numpy(pixels), torch.from_numpy(grids)
            ).numpy()
            expected = reference.model(
                input_ids=torch.from_numpy(ids),
                attention_mask=torch.from_numpy(mask),
                position_ids=torch.from_numpy(_positions(mask)),
                pixel_values=torch.from_numpy(pixels),
                image_grid_thw=torch.from_numpy(grids),
                input_features=torch.from_numpy(features),
                feature_attention_mask=torch.arange(17)[None]
                < torch.from_numpy(lengths)[:, None],
                audio_out_codes=torch.from_numpy(out_codes),
                use_cache=True,
            )
            expected_logits = reference.lm_head(expected.last_hidden_state).numpy()
        actual_images = np.concatenate(
            [
                sessions["vision_encoder"].run(
                    {"pixel_values": patches, "image_grid_thw": grid[None]}
                )["image_features"]
                for patches, grid in zip(np.split(pixels, [16]), grids, strict=True)
            ]
        )
        np.testing.assert_allclose(actual_images, expected_images, rtol=1e-4, atol=1e-4)
        actual = pipeline.prefill(
            ids,
            attention_mask=mask,
            input_features=features,
            feature_lengths=lengths,
            audio_out_codes=out_codes,
            pixel_values=pixels,
            image_grid_thw=grids,
        )
        kept = mask.astype(bool)
        np.testing.assert_allclose(
            actual["logits"][kept], expected_logits[kept], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            actual["hidden_states"][kept],
            expected.last_hidden_state.numpy()[kept],
            rtol=1e-4,
            atol=1e-5,
        )
        for masked in (np.ones((2, 3), bool), np.array([[True, False, True], [False] * 3])):
            depth = sessions["depth_decoder"].run(
                {
                    "hidden_states": actual["hidden_states"][:, -1],
                    "codes": out_codes,
                    "masked": masked,
                }
            )["logits"]
            with torch.no_grad():
                expected_depth = reference.model.depth_decoder(
                    expected.last_hidden_state[:, -1],
                    torch.from_numpy(out_codes),
                    torch.from_numpy(masked),
                ).numpy()
            np.testing.assert_allclose(depth, expected_depth, rtol=1e-4, atol=1e-5)

        # Decode a text token, then one speech frame per row, from the reused caches.
        cache = expected.past_key_values
        frames = rng.integers(0, 17, size=(2, 3), dtype=np.int64)
        for step_ids, step_codes in (
            (np.argmax(actual["logits"][:, -1], axis=-1)[:, None], None),
            (np.full((2, 1), AUDIO_OUT, dtype=np.int64), frames),
        ):
            mask = np.concatenate([mask, np.ones_like(step_ids)], axis=1)
            with torch.no_grad():
                expected = reference(
                    input_ids=torch.from_numpy(step_ids),
                    attention_mask=torch.from_numpy(mask),
                    position_ids=torch.from_numpy(_positions(mask)[:, -1:]),
                    audio_out_codes=None
                    if step_codes is None
                    else torch.from_numpy(step_codes),
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = expected.past_key_values
            actual = pipeline.step(step_ids, step_codes)
            np.testing.assert_allclose(
                actual["logits"], expected.logits.numpy(), rtol=1e-4, atol=1e-5
            )

        with pytest.raises(ValueError, match="placeholders for image_features"):
            pipeline.prefill(np.array([[1, IMAGE]], dtype=np.int64))
        with pytest.raises(ValueError, match="video input is not supported"):
            pipeline.prefill(np.array([[1, VIDEO]], dtype=np.int64))
        for method in (pipeline.generate_text, pipeline.generate_speech):
            with pytest.raises(ValueError, match="generation_mode"):
                method(4)
    finally:
        for session in sessions.values():
            session.close()


def _generate_reference(reference, ids, mode, settings, max_new_tokens):
    """Run the reference generate, recording codebook orders and its speech draws."""
    orders, draws = [], []
    randperm, sample = torch.randperm, type(reference)._sample

    def record_order(*args, **kwargs):
        order = randperm(*args, **kwargs)
        orders.append(order.tolist())
        return order

    def record_draw(
        logits, temperature, top_k, repetition_penalty=None, history=None, *args, **kwargs
    ):
        code = sample(logits, temperature, top_k, repetition_penalty, history, *args, **kwargs)
        if logits.shape[-1] == reference.config.codebook_size:
            scores = logits.float()[None]
            if repetition_penalty not in (None, 1.0) and history is not None:
                processor = RepetitionPenaltyLogitsProcessor(repetition_penalty)
                scores = processor(history.reshape(1, -1), scores)
            draws.append((scores[0].cpu().numpy(), int(code)))
        return code

    prompt = torch.from_numpy(ids).to(reference.device)
    with (
        torch.no_grad(),
        mock.patch("torch.randperm", side_effect=record_order),
        mock.patch.object(type(reference), "_sample", staticmethod(record_draw)),
    ):
        output = reference.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=max_new_tokens,
            generation_mode=mode,
            do_sample=False,
            return_dict_in_generate=True,
            **settings,
        )
    config = reference.config
    markers = {
        "audio": (config.audio_start_token_id, config.audio_out_token_id),
        "interleaved": (config.mixed_start_token_id,),
    }
    width = ids.shape[1] + (mode in markers and int(ids[0, -1]) not in markers[mode])
    tokens = output.sequences.cpu().numpy()[0, width : int(output.sequence_lengths[0])]
    codes = output.audio_codes.cpu().numpy()[:, :, : int(output.audio_lengths[0])]
    return tokens, codes, orders, draws


@pytest.mark.integration
@pytest.mark.parametrize(
    "prompt,mode",
    [([5, 7], "text"), ([5, 7], "audio"), ([5, 7], "interleaved"), ([5, MIXED_START], None)],
)
def test_generation_replays_upstream(prompt, mode):
    """Greedy speech with recorded MaskGIT orders: text, speech and interleaved slots match."""
    reference = _tiny_reference()
    settings = {
        "eos_token_id": [EOS, TEXT_END],
        "audio_temperature": 0.0,
        "audio_repetition_penalty": 1.2,
        "audio_repetition_window": 64,
    }
    ids = np.array([prompt], dtype=np.int64)
    tokens, codes, orders, _ = _generate_reference(reference, ids, mode, settings, 24)
    pipeline, sessions = _pipeline(reference, settings)
    try:
        actual_tokens, actual_codes = pipeline.generate(
            ids, 24, generation_mode=mode, codebook_orders=orders
        )
    finally:
        for session in sessions.values():
            session.close()
    assert actual_tokens.shape == tokens.shape
    np.testing.assert_array_equal(actual_tokens, tokens)
    assert actual_codes.shape == codes.shape
    np.testing.assert_array_equal(actual_codes, codes)
    if mode != "text":
        assert codes.shape[-1] > 0 and AUDIO_OUT in tokens


@pytest.mark.integration
def test_sampled_speech_replays_upstream_draws():
    """Shipped-style sampling: replay upstream's orders and codes, compare the penalized scores."""
    reference = _tiny_reference()
    settings = {
        "eos_token_id": [EOS, TEXT_END],
        "audio_temperature": 0.4,
        "audio_repetition_penalty": 1.2,
        "audio_repetition_window": 64,
    }
    ids = np.array([[5, 7]], dtype=np.int64)
    torch.manual_seed(0)
    tokens, codes, orders, draws = _generate_reference(reference, ids, "audio", settings, 24)
    replay, scores = iter(draws), []

    def draw(actual):
        expected, code = next(replay)
        scores.append((actual, expected))
        return code

    pipeline, sessions = _pipeline(reference, settings)
    try:
        actual_tokens, actual_codes = pipeline.generate(
            ids, 24, generation_mode="audio", codebook_orders=orders, draw=draw
        )
    finally:
        for session in sessions.values():
            session.close()
    assert len(scores) == len(draws) == 3 * len(orders)
    # More sampled frames than returned ones: an end-of-speech frame was fed back.
    assert len(orders) > codes.shape[-1] > 0
    for actual, expected in scores:
        np.testing.assert_allclose(actual, expected, rtol=1.2e-4, atol=1.2e-5)
    np.testing.assert_array_equal(actual_tokens, tokens)
    np.testing.assert_array_equal(actual_codes, codes)


@pytest.fixture(scope="module")
def omni_checkpoint(tmp_path_factory):
    """The pinned reference and package, loaded once; CUDA must not fall back to CPU."""
    from tests._lfm3_omni_golden import _close, _device, _prepare, _reference

    device = _device()
    model, processor = _reference()
    pipeline, sessions = _prepare(device, tmp_path_factory.mktemp("lfm3-omni"))
    model.to(device)
    try:
        yield model, processor, pipeline
    finally:
        _close(sessions, device)


@pytest.mark.integration
@pytest.mark.integration_slow
def test_two_images_features_and_prompt_logits(omni_checkpoint):
    """Two grids, the second at the 2,048-token cap: features and the prompt logits."""
    from tests._lfm3_omni_golden import IMAGE_SYSTEM, _feeds, _image, _inputs

    model, processor, pipeline = omni_checkpoint
    inputs = _inputs(
        processor,
        IMAGE_SYSTEM,
        [
            {"type": "image", "image": _image("pipeline-cat-chonk.jpeg")},
            {"type": "image", "image": _image("mage-vl-dog.jpg")},
            {"type": "text", "text": "Which animals are in these two images?"},
        ],
    ).to(model.device)
    assert inputs["image_grid_thw"].tolist() == [[1, 42, 60], [1, 128, 64]]
    with torch.no_grad():
        expected_images = (
            model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            .cpu()
            .numpy()
        )
        expected = model(**inputs, use_cache=False).logits.cpu().numpy()
    feeds = _feeds(inputs)
    patches = np.split(feeds["pixel_values"], [42 * 60])
    actual_images = np.concatenate(
        [
            pipeline.sessions["vision_encoder"].run(
                {"pixel_values": rows, "image_grid_thw": grid[None]}
            )["image_features"]
            for rows, grid in zip(patches, feeds["image_grid_thw"], strict=True)
        ]
    )
    np.testing.assert_allclose(actual_images, expected_images, rtol=1e-3, atol=1e-3)
    actual = pipeline.prefill(**feeds)["logits"]
    # fp32 noise in the image features grows at the image placeholder positions, where the
    # reference's own CPU and CUDA logits differ too, in one argmax here.
    image = feeds["input_ids"][0] == pipeline.config.image_token_id
    np.testing.assert_allclose(actual[:, ~image], expected[:, ~image], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(actual[:, image], expected[:, image], rtol=1e-3, atol=0.5)
    np.testing.assert_array_equal(actual[:, ~image].argmax(-1), expected[:, ~image].argmax(-1))


@pytest.mark.integration
@pytest.mark.integration_slow
def test_prompt_speech_codes_condition_the_next_turn(omni_checkpoint):
    """An earlier spoken answer given as codes and an end-of-speech frame, then a user turn."""
    from tests._lfm3_omni_golden import TTS_SYSTEM, _golden

    model, processor, pipeline = omni_checkpoint
    spoken = np.array(_golden("lfm3-omni-2b_generation")["tts_audio_codes"][0]).T
    frames = np.concatenate([spoken, np.full((1, 12), pipeline.config.audio_eos_code)])
    text = processor.apply_chat_template(
        [
            {"role": "system", "content": TTS_SYSTEM},
            {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
            {"role": "assistant", "content": [{"type": "audio_out"}]},
            {"role": "user", "content": "Now say the same sentence a little slower."},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    # The processor makes these codes with the codec encoder, which is not exported.
    text = text.replace("<|audio_out|>", "<|audio_out|>" * len(frames))
    ids = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        expected = (
            model(
                input_ids=ids.to(model.device),
                audio_out_codes=torch.from_numpy(frames).to(model.device),
                use_cache=False,
            )
            .logits.cpu()
            .numpy()
        )
    actual = pipeline.prefill(ids.numpy(), audio_out_codes=frames.astype(np.int64))["logits"]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
@pytest.mark.integration_slow
def test_left_padded_rows_match_single_prompts(omni_checkpoint):
    """A spoken prompt and a left-padded text prompt in one prefill, each row as when alone."""
    from tests._lfm3_omni_golden import _clip, _feeds, _inputs

    _, processor, pipeline = omni_checkpoint
    spoken = _feeds(
        _inputs(processor, "Perform ASR in English.", [{"type": "audio", "audio": _clip()}])
    )
    short = _feeds(_inputs(processor, "Respond with text.", "What is the capital of France?"))
    alone = [pipeline.prefill(**feeds)["logits"][0] for feeds in (spoken, short)]
    width, length = spoken["input_ids"].shape[1], short["input_ids"].shape[1]
    ids = np.full((2, width), pipeline.config.pad_token_id, dtype=np.int64)
    mask = np.zeros_like(ids)
    ids[0], mask[0] = spoken["input_ids"][0], 1
    ids[1, -length:], mask[1, -length:] = short["input_ids"][0], 1
    batched = pipeline.prefill(
        ids,
        attention_mask=mask,
        input_features=spoken["input_features"],
        feature_lengths=spoken["feature_lengths"],
    )["logits"]
    np.testing.assert_allclose(batched[0], alone[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(batched[1, -length:], alone[1], rtol=1e-3, atol=1e-3)


@pytest.mark.integration
@pytest.mark.integration_slow
def test_shipped_speech_sampler_replays_upstream_draws(omni_checkpoint):
    """The pinned audio sampling settings: replay upstream's orders and drawn codes."""
    from tests._lfm3_omni_golden import TTS_SYSTEM, _inputs

    model, processor, pipeline = omni_checkpoint
    inputs = _inputs(processor, TTS_SYSTEM, "The quick brown fox jumps over the lazy dog.")
    ids = inputs["input_ids"].numpy()
    torch.manual_seed(0)
    tokens, codes, orders, draws = _generate_reference(model, ids, "audio", {}, 256)
    replay, gaps = iter(draws), []

    def draw(actual):
        expected, code = next(replay)
        gaps.append(float(np.abs(actual - expected).max()))
        return code

    actual_tokens, actual_codes = pipeline.generate(
        ids, 256, generation_mode="audio", codebook_orders=orders, draw=draw
    )
    assert len(gaps) == len(draws) == 12 * len(orders)
    assert len(orders) > codes.shape[-1] > 20, "Speech must end with an end-of-speech frame."
    # The 1e-3 logits tolerance times the 1.2 repetition penalty.
    assert max(gaps) <= 1.2e-3
    np.testing.assert_array_equal(actual_tokens, tokens)
    np.testing.assert_array_equal(actual_codes, codes)
