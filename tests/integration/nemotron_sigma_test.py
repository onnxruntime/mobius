# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned full-checkpoint CUDA parity and generation for both Sigma heads.

Run with CUDA_VISIBLE_DEVICES selecting one idle GPU and an isolated HF_HOME:
python -m pytest tests/integration/nemotron_sigma_test.py -m integration -sv
"""

from __future__ import annotations

import gc
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel

from mobius import ModelPackage, build_from_module
from mobius._testing.nemotron_sigma_reference import load_reference
from mobius.generation.nemotron_sigma import NemotronSigmaGenerator
from mobius.models.nemotron_sigma import (
    MODEL_ID,
    REVISION,
    NemotronSigmaConfig,
    NemotronSigmaModel,
)

pytestmark = [pytest.mark.integration, pytest.mark.integration_slow]
GOLDEN = (
    Path(__file__).parents[2] / "testdata" / "golden" / "causal-lm" / "nemotron-sigma.json"
)


def test_sigma_raw_and_trusted_configs_agree():
    trusted = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    raw = json.loads(
        Path(hf_hub_download(MODEL_ID, "config.json", revision=REVISION)).read_text()
    )
    expected = NemotronSigmaConfig.from_transformers(trusted)
    assert NemotronSigmaConfig.from_transformers(SimpleNamespace(**raw)) == expected


def test_sigma_synthetic_parity(tmp_path):
    """L3 uses independent upstream modules, including read-only draft caching."""
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.vocab_size = 256
    config.max_position_embeddings = 128
    torch.manual_seed(19)
    reference = AutoModel.from_config(config, trust_remote_code=True).float().eval()
    native = NemotronSigmaConfig.from_transformers(config)
    package = build_from_module(NemotronSigmaModel(native), native, task="nemotron-sigma")
    applied = package.apply_weights(reference.state_dict())
    assert set(reference.state_dict()) - applied == {"noise_schedule.increment_values"}
    package.save(str(tmp_path))
    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    runtime = NemotronSigmaGenerator(session)
    ids = torch.tensor([[1, 23, 45], [0, 7, 9]])
    valid = torch.tensor([[True, True, True], [False, True, True]])
    positions = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
    upstream = inspect.getmodule(reference)
    cache = upstream.KVCache(2)
    with torch.no_grad():
        hidden = reference._encode_clean(ids, 0, positions, valid, cache)
        result, onnx_cache = runtime._run(
            ids.numpy(), positions.numpy(), valid.numpy(), runtime._empty_cache(2)
        )
        _assert_close(result["ar_logits"], reference.ar_head(hidden))
        z = torch.randn(2, 4, 16, dtype=torch.float64)
        block_positions = valid.sum(dim=1)[:, None] + torch.arange(4)[None, :]
        gamma = torch.tensor([2.0], dtype=torch.float64)
        condition = torch.randn(2, 4, 16)
        logits, mean, _, _ = reference._denoise_block(
            z,
            gamma,
            cache,
            valid,
            block_positions,
            reference.replaid_embedding_matrix(),
            condition,
            1.0,
        )
        result, _ = runtime._run(
            np.zeros((2, 4), np.int64),
            block_positions.numpy(),
            valid.numpy(),
            onnx_cache,
            z=z.numpy(),
            gamma=2.0,
            condition=condition.numpy(),
        )
        _assert_close(result["diffusion_logits"], logits)
        _assert_close(result["denoised_mean"], mean)


def capture_noise(call):
    """Record actual upstream CUDA draws, without changing sampling behavior."""
    draws = []
    original = torch.randn

    def record(*args, **kwargs):
        tensor = original(*args, **kwargs)
        draws.append(tensor.cpu().numpy().copy())
        return tensor

    with patch("torch.randn", side_effect=record):
        output = call()
    return output, draws


def replay_noise(draws):
    iterator = iter(draws)

    def sample(shape):
        value = next(iterator)
        assert value.shape == shape
        return value.copy()

    return sample, iterator


@pytest.fixture(scope="module")
def full_sigma(tmp_path_factory):
    reference, tokenizer = load_reference()
    config = NemotronSigmaConfig.from_transformers(reference.config)
    module = NemotronSigmaModel(config)
    package = build_from_module(
        module, config, task="nemotron-sigma", execution_provider="cuda"
    )
    state = {name: tensor.cpu() for name, tensor in reference.state_dict().items()}
    applied = package.apply_weights(state)
    assert set(state) - applied == {"noise_schedule.increment_values"}
    del state
    directory = tmp_path_factory.mktemp("sigma-package")
    package.save(str(directory))
    del package, module
    gc.collect()
    restored = ModelPackage.load(str(directory))
    assert json.loads(restored["model"].metadata_props["mobius.sigma"])["native_nfe"] == 2
    options = ort.SessionOptions()
    options.add_session_config_entry("session.use_tf32", "0")
    options.enable_profiling = True
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    session = ort.InferenceSession(
        str(directory / "model.onnx"),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    generator = NemotronSigmaGenerator(session)
    yield reference, tokenizer, generator, directory
    profile = json.loads(Path(session.end_profiling()).read_text())
    assert any(
        event.get("args", {}).get("provider") == "CUDAExecutionProvider"
        and event.get("args", {}).get("op_name") in {"MatMul", "Attention", "FusedMatMul"}
        for event in profile
    ), "No actual CUDA compute appeared in the Sigma inference profile."


def _assert_close(actual, expected):
    expected = expected.detach().float().cpu().numpy()
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.golden
def test_full_checkpoint_sigma_prefill_draft_and_reused_cache(full_sigma):
    reference, tokenizer, runtime, _ = full_sigma
    upstream = inspect.getmodule(reference)
    ids = tokenizer("Here is my poem:", return_tensors="pt")["input_ids"].cuda()
    # A real padded batch has distinct valid lengths; padding must stay masked.
    ids = torch.cat((ids, ids.roll(1, dims=1)), dim=0)
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[1, :2] = False
    positions = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
    cache = upstream.KVCache(reference.config.num_hidden_layers)
    with torch.no_grad():
        hidden = reference._encode_clean(ids, 0, positions, valid, cache)
        result, onnx_cache = runtime._run(
            ids.cpu().numpy(),
            positions.cpu().numpy(),
            valid.cpu().numpy(),
            runtime._empty_cache(2),
        )
        _assert_close(result["ar_logits"], reference.ar_head(hidden))
        gold = json.loads(GOLDEN.read_text())
        assert result["ar_logits"][0].argmax(axis=-1).tolist() == gold["prefill_top1"]
        before = [(key.clone(), value.clone()) for key, value in zip(cache.keys, cache.values)]
        block = reference.config.block_size
        block_positions = (
            valid.sum(dim=1)[:, None] + torch.arange(block, device="cuda")[None, :]
        )
        torch.manual_seed(20261116)
        z = torch.randn(
            2, block, reference.config.embed_dim, dtype=torch.float64, device="cuda"
        )
        _, gamma = reference.gamma_bounds()
        logits, mean, _, _ = reference._denoise_block(
            z,
            gamma.reshape(1),
            cache,
            valid,
            block_positions,
            reference.replaid_embedding_matrix().float(),
            None,
            1.0,
        )
        draft_result, _ = runtime._run(
            np.zeros((2, block), np.int64),
            block_positions.cpu().numpy(),
            valid.cpu().numpy(),
            onnx_cache,
            z=z.cpu().numpy(),
            gamma=gamma.item(),
        )
        _assert_close(draft_result["diffusion_logits"], logits)
        _assert_close(draft_result["denoised_mean"], mean)
        for (old_key, old_value), key, value in zip(before, cache.keys, cache.values):
            assert torch.equal(old_key, key) and torch.equal(old_value, value)
        # Cached multi-token verification, rollback, and a different bonus token.
        draft = logits.argmax(dim=-1)[:, :5]
        extended = torch.cat(
            (valid, torch.ones((2, 5), dtype=torch.bool, device="cuda")), dim=1
        )
        verified = reference._encode_clean(
            draft, ids.shape[1], block_positions[:, :5], extended, cache
        )
        verify_result, tentative = runtime._run(
            draft.cpu().numpy(),
            block_positions[:, :5].cpu().numpy(),
            extended.cpu().numpy(),
            onnx_cache,
        )
        _assert_close(verify_result["ar_logits"], reference.ar_head(verified))
        keep = ids.shape[1] + 2
        cache.trim(keep)
        rolled_back = [(key[:, :, :keep], value[:, :, :keep]) for key, value in tentative]
        bonus = (draft[:, 2:3] + 1) % reference.config.vocab_size
        extended = extended[:, : keep + 1]
        hidden = reference._encode_clean(bonus, keep, block_positions[:, 2:3], extended, cache)
        bonus_result, _ = runtime._run(
            bonus.cpu().numpy(),
            block_positions[:, 2:3].cpu().numpy(),
            extended.cpu().numpy(),
            rolled_back,
        )
        _assert_close(bonus_result["ar_logits"], reference.ar_head(hidden))


@pytest.mark.generation
def test_full_checkpoint_sigma_speculation_and_native_multiblock(full_sigma):
    reference, tokenizer, runtime, _ = full_sigma
    ids = tokenizer("Here is my poem:", return_tensors="pt")["input_ids"].cuda()
    torch.manual_seed(20261116)
    (expected, stats), draws = capture_noise(
        lambda: reference.generate_speculative(ids, max_new_tokens=65, return_stats=True)
    )
    sample, remaining = replay_noise(draws)
    actual, actual_stats = runtime.generate_speculative(
        ids.cpu().numpy(), max_new_tokens=65, noise_sampler=sample, return_stats=True
    )
    assert expected.shape == actual.shape == (1, ids.shape[1] + 65)
    np.testing.assert_array_equal(actual, expected.cpu().numpy())
    assert actual_stats == stats
    assert stats["rounds"] > 1 and 0 < stats["accepted"] < 65
    assert next(remaining, None) is None
    gold = json.loads(GOLDEN.read_text())
    assert actual[0, ids.shape[1] :].tolist() == gold["speculative_tokens"]
    # Independent greedy AR decoding proves draft acceptance cannot change text.
    upstream = inspect.getmodule(reference)
    cache = upstream.KVCache(reference.config.num_hidden_layers)
    valid = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.arange(ids.shape[1], device="cuda")[None, :]
    greedy = []
    with torch.no_grad():
        hidden = reference._encode_clean(ids, 0, positions, valid, cache)
        for index in range(65):
            token = reference.ar_head(hidden[:, -1]).argmax(dim=-1)
            greedy.append(token.item())
            valid = torch.cat(
                (valid, torch.ones((1, 1), dtype=torch.bool, device="cuda")), dim=1
            )
            hidden = reference._encode_clean(
                token[:, None],
                ids.shape[1] + index,
                torch.tensor([[ids.shape[1] + index]], device="cuda"),
                valid,
                cache,
            )
    assert greedy == gold["speculative_tokens"]
    torch.manual_seed(20261117)
    (expected, nfe), draws = capture_noise(
        lambda: reference.generate(
            ids,
            max_new_tokens=65,
            time_schedule="log_snr",
            score_temp=1.0,
            guidance_scale=1.0,
        )
    )
    sample, remaining = replay_noise(draws)
    actual, actual_nfe = runtime.generate(
        ids.cpu().numpy(), max_new_tokens=65, noise_sampler=sample
    )
    assert actual.shape == expected.shape == (1, ids.shape[1] + 65)
    assert actual_nfe == nfe == 9
    np.testing.assert_array_equal(actual, expected.cpu().numpy())
    assert actual[0, ids.shape[1] :].tolist() == gold["diffusion_tokens"]
    assert next(remaining, None) is None
