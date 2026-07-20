# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Qwen3-TTS talker step embedder pre-embedding component."""

from __future__ import annotations

import numpy as np
import torch

from mobius._configs import ArchitectureConfig, CodePredictorConfig, TTSConfig
from mobius._model_package import ModelPackage
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSTalkerStepEmbedder,
)
from mobius.tasks import TTSTask

# Tiny synthetic config — small tables, no 1.7B download.
_HIDDEN = 8
_CODEC_VOCAB = 10
_CP_VOCAB = 6
_NUM_CODE_GROUPS = 4

_TINY_CONFIG = ArchitectureConfig(
    model_type="qwen3_tts",
    hidden_size=_HIDDEN,
    intermediate_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    vocab_size=_CODEC_VOCAB,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    tts=TTSConfig(
        num_code_groups=_NUM_CODE_GROUPS,
        code_predictor=CodePredictorConfig(
            hidden_size=_HIDDEN,
            vocab_size=_CP_VOCAB,
            num_code_groups=_NUM_CODE_GROUPS,
        ),
    ),
)


def _build_step_embedder_session(codec_table, stacked_codec):
    """Build the talker_step_embedder ONNX graph and load the given tables."""
    module = Qwen3TTSTalkerStepEmbedder(_TINY_CONFIG)
    # Match the initializer prefix used when the composite model is built.
    module._set_name("talker_step_embedder")
    model = TTSTask()._build_talker_step_embedder(module, _TINY_CONFIG)
    pkg = ModelPackage({"talker_step_embedder": model}, config=_TINY_CONFIG)
    pkg.apply_weights(
        {
            "talker_step_embedder.codec_embedding": torch.from_numpy(codec_table),
            "talker_step_embedder.stacked_codec_embedding": torch.from_numpy(stacked_codec),
        }
    )
    return OnnxModelSession(model)


def test_step_embedder_matches_numpy_gather_sum():
    """Gather+Sum in the graph equals the numpy codec_sum + text_embed reference."""
    rng = np.random.default_rng(0)
    codec_table = rng.standard_normal((_CODEC_VOCAB, _HIDDEN)).astype(np.float32)
    stacked_codec = rng.standard_normal(
        (_NUM_CODE_GROUPS - 1, _CP_VOCAB, _HIDDEN)
    ).astype(np.float32)

    session = _build_step_embedder_session(codec_table, stacked_codec)

    for _ in range(5):
        code_0 = int(rng.integers(0, _CODEC_VOCAB))
        rest = rng.integers(0, _CP_VOCAB, size=_NUM_CODE_GROUPS - 1).astype(np.int64)
        frame_codes = np.array([[code_0, *rest.tolist()]], dtype=np.int64)
        text_embed = rng.standard_normal((1, 1, _HIDDEN)).astype(np.float32)

        # numpy reference: codec_sum + text_embed
        codec_sum = codec_table[code_0].copy()
        for i in range(_NUM_CODE_GROUPS - 1):
            codec_sum = codec_sum + stacked_codec[i, rest[i], :]
        expected = codec_sum.reshape(1, 1, _HIDDEN) + text_embed

        out = session.run({"frame_codes": frame_codes, "text_embed": text_embed})
        got = out["inputs_embeds"]

        assert got.shape == (1, 1, _HIDDEN)
        np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_step_embedder_batched():
    """The component broadcasts correctly over a batch dimension."""
    rng = np.random.default_rng(1)
    codec_table = rng.standard_normal((_CODEC_VOCAB, _HIDDEN)).astype(np.float32)
    stacked_codec = rng.standard_normal(
        (_NUM_CODE_GROUPS - 1, _CP_VOCAB, _HIDDEN)
    ).astype(np.float32)
    session = _build_step_embedder_session(codec_table, stacked_codec)

    batch = 3
    code0 = rng.integers(0, _CODEC_VOCAB, size=batch)
    rest = rng.integers(0, _CP_VOCAB, size=(batch, _NUM_CODE_GROUPS - 1))
    frame_codes = np.concatenate([code0[:, None], rest], axis=1).astype(np.int64)
    text_embed = rng.standard_normal((batch, 1, _HIDDEN)).astype(np.float32)

    expected = np.empty((batch, 1, _HIDDEN), dtype=np.float32)
    for b in range(batch):
        codec_sum = codec_table[code0[b]].copy()
        for i in range(_NUM_CODE_GROUPS - 1):
            codec_sum = codec_sum + stacked_codec[i, rest[b, i], :]
        expected[b, 0] = codec_sum + text_embed[b, 0]

    got = session.run({"frame_codes": frame_codes, "text_embed": text_embed})[
        "inputs_embeds"
    ]
    assert got.shape == (batch, 1, _HIDDEN)
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_step_embedder_weights_shared_with_existing_tables():
    """preprocess_weights routes the same codec tables to the step embedder."""
    model = Qwen3TTSForConditionalGeneration(_TINY_CONFIG)

    codec_weight = torch.randn(_CODEC_VOCAB, _HIDDEN)
    state_dict: dict[str, torch.Tensor] = {
        "talker.model.codec_embedding.weight": codec_weight,
    }
    for i in range(_NUM_CODE_GROUPS - 1):
        state_dict[f"talker.code_predictor.model.codec_embedding.{i}.weight"] = torch.randn(
            _CP_VOCAB, _HIDDEN
        )

    cleaned = model.preprocess_weights(state_dict)

    # Talker codec table is shared with the embedding model.
    assert "talker_step_embedder.codec_embedding" in cleaned
    torch.testing.assert_close(
        cleaned["talker_step_embedder.codec_embedding"],
        cleaned["embedding.codec_embedding.weight"],
    )

    # Stacked CP codec table is shared with the code predictor's codec_embeddings.
    assert "talker_step_embedder.stacked_codec_embedding" in cleaned
    torch.testing.assert_close(
        cleaned["talker_step_embedder.stacked_codec_embedding"],
        cleaned["codec_embeddings"],
    )
    assert cleaned["talker_step_embedder.stacked_codec_embedding"].shape == (
        _NUM_CODE_GROUPS - 1,
        _CP_VOCAB,
        _HIDDEN,
    )
