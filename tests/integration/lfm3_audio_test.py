# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned upstream numerical checks for LiquidAI LFM3 Audio components."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnxscript import GraphBuilder
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mobius import build_from_module
from mobius._configs import Lfm2Config
from mobius._configs.lfm3_audio import Lfm3AudioConfig
from mobius._constants import OPSET_VERSION
from mobius._optimizations import optimize_model
from mobius.models.lfm2 import rename_lfm2_weight_key
from mobius.models.lfm3_audio import (
    Lfm3AudioDownsampler,
    Lfm3AudioEncoder,
    Lfm3AudioForConditionalGeneration,
)
from tests.integration._support import _make_session

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
REVISION = "2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1"


@pytest.mark.integration
@pytest.mark.parametrize("frames,lengths", [(25, [25, 17]), (32, [32, 9]), (1, [1, 1])])
def test_downsampler_matches_pinned_upstream(frames, lengths):
    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    ).audio_encoder_config
    config.conv_channels = 4
    config.hidden_size = 32
    reference_class = get_class_from_dynamic_module(
        "modeling_lfm3_audio.Lfm3AudioDownsampler", MODEL_ID, revision=REVISION
    )
    torch.manual_seed(42)
    reference = reference_class(config).eval()
    module = Lfm3AudioDownsampler(
        config.num_mel_bins, config.conv_channels, config.hidden_size
    )

    graph = ir.Graph([], [], nodes=[], opset_imports={"": OPSET_VERSION}, name="downsampler")
    builder = GraphBuilder(graph)
    features = builder.input("input_features", dtype=ir.DataType.FLOAT, shape=[2, 128, frames])
    feature_lengths = builder.input("feature_lengths", dtype=ir.DataType.INT64, shape=[2])
    output = module(builder.op, features, feature_lengths)
    builder.add_output(output, "hidden_states")
    model = ir.Model(graph, ir_version=10)
    for name, value in reference.state_dict().items():
        model.graph.initializers[name].const_value = ir.Tensor(value.numpy())
    inputs = torch.randn(2, 128, frames)
    valid_lengths = torch.tensor(lengths, dtype=torch.int64)
    with torch.no_grad():
        expected = reference(inputs, valid_lengths).numpy()
    session = _make_session(model)
    try:
        actual = session.run(
            {"input_features": inputs.numpy(), "feature_lengths": valid_lengths.numpy()},
        )["hidden_states"]
    finally:
        session.close()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("frames,lengths", [(25, [25, 17]), (65, [65, 33])])
def test_encoder_matches_pinned_upstream(frames, lengths):
    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    ).audio_encoder_config
    config.conv_channels = 4
    config.hidden_size = 32
    config.sliding_window = 3
    text = config.text_config
    text.hidden_size = 32
    text.intermediate_size = 64
    text.block_auto_adjust_ff_dim = False
    text.num_hidden_layers = 2
    text.layer_types = ["conv", "full_attention"]
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text._attn_implementation = "eager"
    reference_class = get_class_from_dynamic_module(
        "modeling_lfm3_audio.Lfm3AudioEncoder", MODEL_ID, revision=REVISION
    )
    torch.manual_seed(42)
    reference = reference_class(config).eval()
    module = Lfm3AudioEncoder(
        Lfm2Config.from_transformers(text),
        num_mel_bins=config.num_mel_bins,
        conv_channels=config.conv_channels,
        sliding_window=config.sliding_window,
    )
    graph = ir.Graph([], [], nodes=[], opset_imports={"": OPSET_VERSION}, name="audio_encoder")
    builder = GraphBuilder(graph)
    features = builder.input("input_features", dtype=ir.DataType.FLOAT, shape=[2, 128, frames])
    feature_lengths = builder.input("feature_lengths", dtype=ir.DataType.INT64, shape=[2])
    output, output_lengths = module(builder.op, features, feature_lengths)
    builder.add_output(output, "audio_features")
    builder.add_output(output_lengths, "audio_feature_lengths")
    model = ir.Model(graph, ir_version=10)
    for name, value in reference.state_dict().items():
        name = rename_lfm2_weight_key(name)
        model.graph.initializers[name].const_value = ir.Tensor(value.numpy())
    optimize_model(model, ep="cpu", model_role="encoder")
    inputs = torch.randn(2, 128, frames)
    valid_lengths = torch.tensor(lengths, dtype=torch.int64)
    with torch.no_grad():
        expected, expected_lengths = reference(inputs, valid_lengths)
    session = _make_session(model)
    try:
        actual = session.run(
            {"input_features": inputs.numpy(), "feature_lengths": valid_lengths.numpy()}
        )
    finally:
        session.close()
    np.testing.assert_array_equal(actual["audio_feature_lengths"], expected_lengths.numpy())
    np.testing.assert_allclose(
        actual["audio_features"], expected.numpy(), rtol=1e-4, atol=1e-5
    )


def _tiny_audio_reference():
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    config.num_codebooks = 3
    config.codebook_size = 17
    config.audio_eos_code = 16
    config.maskgit_schedule = [1, 1, 1]
    for name, value in {
        "audio_token_id": 240,
        "audio_out_token_id": 241,
        "audio_start_token_id": 242,
        "text_end_token_id": 243,
        "mixed_start_token_id": 244,
    }.items():
        setattr(config, name, value)
    text = config.text_config
    text.hidden_size = 32
    text.intermediate_size = 64
    text.vocab_size = 256
    text.pad_token_id = 0
    text.bos_token_id = 1
    text.eos_token_id = 2
    text.head_dim = 8
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.num_hidden_layers = 3
    text.layer_types = ["conv", "full_attention", "sliding_attention"]
    text.sliding_window = 4
    text.max_position_embeddings = 128
    text._attn_implementation = "eager"
    encoder = config.audio_encoder_config
    encoder.hidden_size = 32
    encoder.conv_channels = 4
    encoder.sliding_window = 3
    encoder.text_config.hidden_size = 32
    encoder.text_config.intermediate_size = 64
    encoder.text_config.block_auto_adjust_ff_dim = False
    encoder.text_config.num_hidden_layers = 2
    encoder.text_config.layer_types = ["conv", "full_attention"]
    encoder.text_config.num_attention_heads = 4
    encoder.text_config.num_key_value_heads = 2
    encoder.text_config.max_position_embeddings = 128
    encoder.text_config._attn_implementation = "eager"
    depth = config.depth_config
    depth.hidden_size = 32
    depth.intermediate_size = 64
    depth.head_dim = 8
    depth.num_attention_heads = 4
    depth.num_key_value_heads = 2
    depth.num_hidden_layers = 1
    depth.layer_types = ["full_attention"]
    depth._attn_implementation = "eager"
    model_class = get_class_from_dynamic_module(
        "modeling_lfm3_audio.Lfm3AudioForConditionalGeneration", MODEL_ID, revision=REVISION
    )
    torch.manual_seed(42)
    return model_class(config).eval()


def _empty_audio_cache(config, batch):
    cache = {}
    for index, kind in enumerate(config.layer_types):
        prefix = f"past_key_values.{index}"
        if kind == "conv":
            cache[f"{prefix}.conv_state"] = np.zeros(
                (batch, config.hidden_size, config.short_conv_kernel - 1), dtype=np.float32
            )
        else:
            for name in ("key", "value"):
                cache[f"{prefix}.{name}"] = np.zeros(
                    (batch, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
                )
    return cache


@pytest.mark.integration
def test_audio_pipeline_and_cached_decode_match_upstream(tmp_path):
    from mobius import ModelPackage
    from mobius.integrations.lfm3_audio import Lfm3AudioPipeline

    reference = _tiny_audio_reference()
    config = Lfm3AudioConfig.from_transformers(reference.config)
    module = Lfm3AudioForConditionalGeneration(config)
    package = build_from_module(module, config, task="lfm3-audio")
    package.apply_weights(module.preprocess_weights(reference.state_dict()))
    package.save(tmp_path / "audio")
    package = ModelPackage.load(tmp_path / "audio")
    assert set(package) == {"audio_encoder", "embedding", "decoder", "depth_decoder"}
    sessions = {}
    try:
        sessions = {name: _make_session(model) for name, model in package.items()}
        rng = np.random.default_rng(42)
        features = rng.normal(size=(2, 128, 17)).astype(np.float32)
        lengths = np.array([17, 9], dtype=np.int64)
        ids = np.array(
            [[5, 240, 240, 240, 7, 241, 9], [8, 240, 240, 6, 9, 241, 10]], dtype=np.int64
        )
        out_codes = rng.integers(0, 17, size=(2, 3), dtype=np.int64)
        encoded = sessions["audio_encoder"].run(
            {"input_features": features, "feature_lengths": lengths}
        )
        packed = np.concatenate(
            [
                row[:length]
                for row, length in zip(
                    encoded["audio_features"], encoded["audio_feature_lengths"], strict=True
                )
            ]
        )
        embeds = sessions["embedding"].run(
            {"input_ids": ids, "audio_features": packed, "audio_out_codes": out_codes}
        )["inputs_embeds"]
        mask = np.ones_like(ids)
        positions = np.broadcast_to(np.arange(ids.shape[1], dtype=np.int64), ids.shape).copy()
        actual = sessions["decoder"].run(
            {
                "inputs_embeds": embeds,
                "attention_mask": mask,
                "position_ids": positions,
                **_empty_audio_cache(config, 2),
            }
        )
        with torch.no_grad():
            expected = reference.model(
                input_ids=torch.from_numpy(ids),
                attention_mask=torch.from_numpy(mask),
                input_features=torch.from_numpy(features),
                feature_attention_mask=torch.arange(17)[None]
                < torch.from_numpy(lengths)[:, None],
                audio_out_codes=torch.from_numpy(out_codes),
                use_cache=True,
            )
            expected_logits = reference.lm_head(expected.last_hidden_state).numpy()
        np.testing.assert_allclose(actual["logits"], expected_logits, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            actual["hidden_states"], expected.last_hidden_state.numpy(), rtol=1e-4, atol=1e-5
        )
        for masked in (
            np.ones((2, 3), bool),
            np.array([[True, False, True], [False, True, False]]),
        ):
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

        next_ids = np.argmax(actual["logits"][:, -1], axis=-1).astype(np.int64)[:, None]
        next_embeds = sessions["embedding"].run(
            {
                "input_ids": next_ids,
                "audio_features": np.empty((0, config.hidden_size), dtype=np.float32),
                "audio_out_codes": np.empty((0, config.num_codebooks), dtype=np.int64),
            }
        )["inputs_embeds"]
        next_mask = np.ones((2, ids.shape[1] + 1), dtype=np.int64)
        next_actual = sessions["decoder"].run(
            {
                "inputs_embeds": next_embeds,
                "attention_mask": next_mask,
                "position_ids": np.full((2, 1), ids.shape[1], dtype=np.int64),
                **{
                    name.replace("present.", "past_key_values.", 1): value
                    for name, value in actual.items()
                    if name.startswith("present.")
                },
            }
        )
        with torch.no_grad():
            next_expected = reference(
                input_ids=torch.from_numpy(next_ids),
                attention_mask=torch.from_numpy(next_mask),
                past_key_values=expected.past_key_values,
                use_cache=True,
            ).logits.numpy()
        np.testing.assert_allclose(next_actual["logits"], next_expected, rtol=1e-4, atol=1e-5)
        pipeline = Lfm3AudioPipeline(config, sessions)
        fused = pipeline.prefill(
            ids, input_features=features, feature_lengths=lengths, audio_out_codes=out_codes
        )
        np.testing.assert_allclose(fused["logits"], expected_logits, rtol=1e-4, atol=1e-5)
        cached = pipeline.step(next_ids)
        np.testing.assert_allclose(cached["logits"], next_expected, rtol=1e-4, atol=1e-5)
        with pytest.raises(ValueError, match="placeholders for audio_features"):
            pipeline.prefill(ids, audio_out_codes=out_codes)
    finally:
        for session in sessions.values():
            session.close()


@pytest.mark.integration
@pytest.mark.parametrize("speech", [False, True])
def test_offline_generation_replays_upstream(speech):
    from unittest import mock

    from mobius.integrations.lfm3_audio import Lfm3AudioPipeline

    reference = _tiny_audio_reference()
    config = Lfm3AudioConfig.from_transformers(reference.config)
    module = Lfm3AudioForConditionalGeneration(config)
    package = build_from_module(module, config, task="lfm3-audio")
    package.apply_weights(module.preprocess_weights(reference.state_dict()))
    sessions = {}
    try:
        sessions = {name: _make_session(model) for name, model in package.items()}
        pipeline = Lfm3AudioPipeline(config, sessions)
        ids = np.array([[5, config.audio_start_token_id if speech else 7]], dtype=np.int64)
        orders = []
        randperm = torch.randperm

        def record_order(*args, **kwargs):
            order = randperm(*args, **kwargs)
            orders.append(order.tolist())
            return order

        with torch.no_grad(), mock.patch("torch.randperm", side_effect=record_order):
            expected = reference.generate(
                input_ids=torch.from_numpy(ids),
                attention_mask=torch.ones_like(torch.from_numpy(ids)),
                max_new_tokens=4,
                do_sample=False,
                repetition_penalty=1.0,
                audio_temperature=0.0,
                audio_top_k=0,
                audio_repetition_penalty=1.0,
                suppress_mixed=not speech,
                suppress_audio_out=not speech,
                return_dict_in_generate=True,
            )
        pipeline.prefill(ids)
        if speech:
            actual = pipeline.generate_speech(len(orders), codebook_orders=orders)
            expected_codes = expected.audio_codes.numpy()[
                :, :, : int(expected.audio_lengths[0])
            ]
            np.testing.assert_array_equal(actual, expected_codes)
        else:
            actual = pipeline.generate_text(4)
            np.testing.assert_array_equal(
                actual, expected.sequences.numpy()[0, ids.shape[1] :]
            )
    finally:
        for session in sessions.values():
            session.close()
