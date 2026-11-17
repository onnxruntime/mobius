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
    Qwen3TTSTalkerPrefillEmbedder,
    Qwen3TTSTalkerStepEmbedder,
)
from mobius.tasks import TTSTask

# Tiny synthetic config — small tables, no 1.7B download.
_HIDDEN = 8
_CODEC_VOCAB = 2160  # > largest codec prefill id (2157)
_CP_VOCAB = 6
_NUM_CODE_GROUPS = 4
_TEXT_HIDDEN = 8
_TEXT_VOCAB = 151674  # > largest tts special id (151673)

_TINY_CONFIG = ArchitectureConfig(
    model_type="qwen3_tts",
    hidden_size=_HIDDEN,
    intermediate_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    vocab_size=_CODEC_VOCAB,
    max_position_embeddings=128,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    mrope_section=[1, 1, 0],
    mrope_interleaved=True,
    rope_type="default",
    tts=TTSConfig(
        num_code_groups=_NUM_CODE_GROUPS,
        text_hidden_size=_TEXT_HIDDEN,
        text_vocab_size=_TEXT_VOCAB,
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
    stacked_codec = rng.standard_normal((_NUM_CODE_GROUPS - 1, _CP_VOCAB, _HIDDEN)).astype(
        np.float32
    )

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
    stacked_codec = rng.standard_normal((_NUM_CODE_GROUPS - 1, _CP_VOCAB, _HIDDEN)).astype(
        np.float32
    )
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

    got = session.run({"frame_codes": frame_codes, "text_embed": text_embed})["inputs_embeds"]
    assert got.shape == (batch, 1, _HIDDEN)
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_code_predictor_transition_components_use_exported_embeddings():
    task = TTSTask()
    prefill = OnnxModelSession(task._build_code_predictor_prefill(_TINY_CONFIG))
    step = OnnxModelSession(task._build_code_predictor_step_embedder(_TINY_CONFIG))
    talker_hidden = np.arange(8, dtype=np.float32).reshape(1, 1, 8)
    group_0_embed = talker_hidden + 10
    got_prefill = prefill.run(
        {
            "talker_hidden": talker_hidden,
            "group_0_embed": group_0_embed,
        }
    )["inputs_embeds"]
    np.testing.assert_array_equal(
        got_prefill,
        np.concatenate([talker_hidden, group_0_embed], axis=1),
    )

    tables = np.arange(
        (_NUM_CODE_GROUPS - 1) * _CP_VOCAB * _HIDDEN,
        dtype=np.float32,
    ).reshape(_NUM_CODE_GROUPS - 1, _CP_VOCAB, _HIDDEN)
    got_step = step.run(
        {
            "codec_embeddings": tables,
            "token": np.array([4], np.int64),
            "embedding_index": np.array(1, np.int64),
        }
    )["inputs_embeds"]
    np.testing.assert_array_equal(got_step, tables[1, 4].reshape(1, 1, _HIDDEN))


def test_talker_text_step_clamps_to_last_trailing_embedding():
    session = OnnxModelSession(TTSTask()._build_talker_text_step(_TINY_CONFIG))
    trailing = np.arange(3 * _HIDDEN, dtype=np.float32).reshape(1, 3, _HIDDEN)
    got = session.run(
        {
            "trailing_text_embeds": trailing,
            "iteration": np.array([7], np.int64),
        }
    )["text_embed"]
    np.testing.assert_array_equal(got, trailing[:, 2:3])


def test_every_built_component_declares_a_role():
    """model_roles must cover every key the task puts in the package.

    ``build_from_module`` looks each component up in ``model_roles`` and falls
    back to ``"decoder"`` when it is absent, which would hand the parameter-free
    loop-wiring graphs the GQA / QKV-packing passes meant for attention stacks.
    ``inspect_components`` reports exactly ``model_roles``, so an undeclared
    component would also be invisible to callers planning per-component work.
    """
    module = Qwen3TTSForConditionalGeneration(_TINY_CONFIG)
    package = TTSTask().build(module, _TINY_CONFIG)

    # speaker_encoder is optional, so the package is a subset of the declared
    # roles; nothing may be built that is not declared.
    assert set(package) <= set(TTSTask.model_roles)
    assert set(TTSTask.model_roles) - set(package) == {"speaker_encoder"}

    # Glue components are pure wiring: every tensor they read is a graph input,
    # so preprocess_weights never routes a parameter to one.
    glue = {name for name, role in TTSTask.model_roles.items() if role == "glue"}
    weights = module.preprocess_weights({})
    assert not [name for name in weights if name.split(".")[0] in glue]


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


# ---------------------------------------------------------------------------
# Talker prefill embedder
# ---------------------------------------------------------------------------

# Special IDs for the Auto/no-speaker/no-instruct path (mirrors the module).
_TTS_BOS_ID = 151672
_TTS_EOS_ID = 151673
_TTS_PAD_ID = 151671
_AUTO_CODEC_PREFILL_IDS = [2155, 2156, 2157, 2148, 2149]  # N = 5


def _build_prefill_embedder_session(weights):
    """Build the talker_prefill_embedder ONNX graph and load the given tables."""
    module = Qwen3TTSTalkerPrefillEmbedder(_TINY_CONFIG)
    module._set_name("talker_prefill_embedder")
    model = TTSTask()._build_talker_prefill_embedder(module, _TINY_CONFIG)
    pkg = ModelPackage({"talker_prefill_embedder": model}, config=_TINY_CONFIG)
    pkg.apply_weights(
        {f"talker_prefill_embedder.{k}": torch.from_numpy(v) for k, v in weights.items()}
    )
    return OnnxModelSession(model)


def _numpy_prefill_reference(weights, text_ids):
    """Reproduce generate_codes's prefill_embeds + trailing_text (Auto path)."""
    text_emb = weights["text_embedding.weight"]
    fc1_w = weights["text_projection_fc1.weight"]
    fc1_b = weights["text_projection_fc1.bias"]
    fc2_w = weights["text_projection_fc2.weight"]
    fc2_b = weights["text_projection_fc2.bias"]
    codec_emb = weights["codec_embedding.weight"]

    def text_path(ids):
        e = text_emb[ids]  # (B, L, text_hidden)
        e = e @ fc1_w.T + fc1_b
        e = e * (1.0 / (1.0 + np.exp(-e)))  # SiLU
        return e @ fc2_w.T + fc2_b  # (B, L, hidden)

    all_text = text_path(text_ids)  # (B, L, H)
    special = text_path(np.array([[_TTS_BOS_ID, _TTS_EOS_ID, _TTS_PAD_ID]], dtype=np.int64))
    tts_bos = special[:, 0:1, :]
    tts_eos = special[:, 1:2, :]
    tts_pad = special[:, 2:3, :]

    codec_prefill = codec_emb[np.array([_AUTO_CODEC_PREFILL_IDS], dtype=np.int64)]  # (1,N,H)
    n = codec_prefill.shape[1]

    role = all_text[:, :3, :]
    text_side = np.concatenate([np.tile(tts_pad, (1, n - 2, 1)), tts_bos], axis=1)
    codec_text_pairs = text_side + codec_prefill[:, : n - 1, :]
    first_text_codec = all_text[:, 3:4, :] + codec_prefill[:, -1:, :]
    prefill = np.concatenate([role, codec_text_pairs, first_text_codec], axis=1)

    trailing = np.concatenate([all_text[:, 4:-5, :], tts_eos], axis=1)
    return prefill.astype(np.float32), trailing.astype(np.float32)


def _random_prefill_weights(seed):
    rng = np.random.default_rng(seed)
    return {
        "text_embedding.weight": rng.standard_normal((_TEXT_VOCAB, _TEXT_HIDDEN)).astype(
            np.float32
        ),
        "text_projection_fc1.weight": rng.standard_normal((_TEXT_HIDDEN, _TEXT_HIDDEN)).astype(
            np.float32
        ),
        "text_projection_fc1.bias": rng.standard_normal(_TEXT_HIDDEN).astype(np.float32),
        "text_projection_fc2.weight": rng.standard_normal((_HIDDEN, _TEXT_HIDDEN)).astype(
            np.float32
        ),
        "text_projection_fc2.bias": rng.standard_normal(_HIDDEN).astype(np.float32),
        "codec_embedding.weight": rng.standard_normal((_CODEC_VOCAB, _HIDDEN)).astype(
            np.float32
        ),
    }


def test_prefill_embedder_matches_numpy_reference():
    """Graph interleaving equals the numpy prefill/trailing reference."""
    weights = _random_prefill_weights(7)
    session = _build_prefill_embedder_session(weights)
    rng = np.random.default_rng(11)

    for text_len in (10, 14, 20):
        text_ids = rng.integers(0, 1000, size=(1, text_len)).astype(np.int64)
        exp_prefill, exp_trailing = _numpy_prefill_reference(weights, text_ids)

        out = session.run({"text_ids": text_ids})
        got_prefill = out["prefill_embeds"]
        got_trailing = out["trailing_text_embeds"]

        # prefill_len is constant: 3 (role) + 4 (N-1 pairs) + 1 (first_text_codec)
        assert got_prefill.shape == (1, 8, _HIDDEN)
        # trailing_len = (text_len - 9) + 1 = text_len - 8
        assert got_trailing.shape == (1, text_len - 8, _HIDDEN)

        np.testing.assert_allclose(got_prefill, exp_prefill, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(got_trailing, exp_trailing, atol=1e-4, rtol=1e-4)


def test_prefill_embedder_batched():
    """Batched text_ids (B>1) produce per-row output matching the B=1 reference.

    Guards the batch-broadcast of the batch=1 codec/special pieces: a naive
    Concat of batch=1 pairs with batch=B text would fail the batch dim for B>1.
    """
    weights = _random_prefill_weights(13)
    session = _build_prefill_embedder_session(weights)
    rng = np.random.default_rng(99)
    text_len = 16
    text_ids = rng.integers(0, 1000, size=(3, text_len)).astype(np.int64)

    out = session.run({"text_ids": text_ids})
    got_prefill = out["prefill_embeds"]
    got_trailing = out["trailing_text_embeds"]

    assert got_prefill.shape == (3, 8, _HIDDEN)
    assert got_trailing.shape == (3, text_len - 8, _HIDDEN)

    # Each batch row must equal the single-row reference for that row's ids.
    for b in range(3):
        exp_prefill, exp_trailing = _numpy_prefill_reference(weights, text_ids[b : b + 1])
        np.testing.assert_allclose(got_prefill[b : b + 1], exp_prefill, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(got_trailing[b : b + 1], exp_trailing, atol=1e-4, rtol=1e-4)


def test_prefill_embedder_weights_shared_with_embedding():
    """preprocess_weights routes the embedding tables to the prefill embedder."""
    model = Qwen3TTSForConditionalGeneration(_TINY_CONFIG)

    state_dict: dict[str, torch.Tensor] = {
        "talker.model.text_embedding.weight": torch.randn(_TEXT_VOCAB, _TEXT_HIDDEN),
        "talker.text_projection.linear_fc1.weight": torch.randn(_TEXT_HIDDEN, _TEXT_HIDDEN),
        "talker.text_projection.linear_fc1.bias": torch.randn(_TEXT_HIDDEN),
        "talker.text_projection.linear_fc2.weight": torch.randn(_HIDDEN, _TEXT_HIDDEN),
        "talker.text_projection.linear_fc2.bias": torch.randn(_HIDDEN),
        "talker.model.codec_embedding.weight": torch.randn(_CODEC_VOCAB, _HIDDEN),
    }

    cleaned = model.preprocess_weights(state_dict)

    shared = {
        "text_embedding.weight",
        "text_projection_fc1.weight",
        "text_projection_fc1.bias",
        "text_projection_fc2.weight",
        "text_projection_fc2.bias",
        "codec_embedding.weight",
    }
    for key in shared:
        assert f"talker_prefill_embedder.{key}" in cleaned
        torch.testing.assert_close(
            cleaned[f"talker_prefill_embedder.{key}"],
            cleaned[f"embedding.{key}"],
        )
