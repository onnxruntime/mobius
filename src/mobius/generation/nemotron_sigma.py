# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Host sampling for the persisted Nemotron-Sigma dual-head ONNX ABI."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

NoiseSampler = Callable[[tuple[int, ...]], np.ndarray]


class NemotronSigmaGenerator:
    """Run native diffusion or exact-greedy self-speculation from an ORT session.

    All geometry is restored from ONNX metadata, not an in-memory HF config.
    ``noise_sampler`` may supply recorded upstream FP64 draws for exact replay;
    a NumPy seed alone does not reproduce CUDA's random-number stream.
    """

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        raw = session.get_modelmeta().custom_metadata_map.get("mobius.sigma")
        if raw is None:
            raise ValueError(
                "Missing mobius.sigma ONNX metadata. Load a package exported "
                "with the nemotron-sigma task."
            )
        self.contract = json.loads(raw)
        if self.contract.get("version") != 1:
            raise ValueError(
                f"Unsupported Sigma ABI version {self.contract.get('version')!r}; "
                "re-export the checkpoint with this Mobius version."
            )
        self.output_names = [output.name for output in session.get_outputs()]

    def _inputs(self, input_ids, attention_mask, max_new_tokens):
        ids = np.asarray(input_ids)
        if ids.dtype != np.int64 or ids.ndim != 2 or not ids.shape[1]:
            raise ValueError("Sigma input_ids must be a nonempty [batch, length] int64 array.")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("Sigma max_new_tokens must be a nonnegative integer.")
        valid = (
            np.ones(ids.shape, dtype=np.bool_)
            if attention_mask is None
            else np.asarray(attention_mask)
        )
        if valid.shape != ids.shape or not np.isin(valid, [0, 1]).all():
            raise ValueError(
                "Sigma attention_mask must contain 0/1 and match input_ids.shape."
            )
        valid = valid.astype(np.bool_)
        if not valid.any(axis=1).all() or np.any(np.diff(valid.astype(np.int8), axis=1) < 0):
            raise ValueError("Sigma prompts must contain real tokens with left padding only.")
        if ids.shape[1] + max_new_tokens > self.contract["max_context"]:
            raise ValueError(
                f"Sigma prompt + generation exceeds {self.contract['max_context']} tokens; "
                "shorten the prompt or reduce max_new_tokens."
            )
        return ids.copy(), valid

    def _empty_cache(self, batch):
        c = self.contract
        shape = (batch, c["num_kv_heads"], 0, c["head_dim"])
        return [
            (np.empty(shape, np.float32), np.empty(shape, np.float32))
            for _ in range(c["num_layers"])
        ]

    def _run(self, ids, positions, valid, cache, *, z=None, gamma=0.0, condition=None):
        batch, length = ids.shape
        past = cache[0][0].shape[2]
        denoise = z is not None
        shape = (batch, length, self.contract["embed_dim"])
        if denoise:
            mask = np.broadcast_to(valid[:, None, None, :], (batch, 1, length, past))
            mask = np.concatenate(
                (mask, np.ones((batch, 1, length, length), np.bool_)), axis=-1
            )
        else:
            mask = np.arange(past + length)[None, :] <= (past + np.arange(length))[:, None]
            mask = mask[None, None, :, :] & valid[:, None, None, :]
        feeds = {
            "input_ids": ids,
            "position_ids": positions,
            "attention_mask": mask,
            "z": np.zeros(shape, np.float64) if z is None else z,
            "gamma": np.asarray(gamma, np.float64),
            "self_condition": np.zeros(shape, np.float32) if condition is None else condition,
            "denoise": np.asarray(denoise, np.bool_),
        }
        for index, (key, value) in enumerate(cache):
            feeds[f"past_key_values.{index}.key"] = key
            feeds[f"past_key_values.{index}.value"] = value
        result = dict(zip(self.output_names, self.session.run(None, feeds)))
        present = [
            (result[f"present.{i}.key"], result[f"present.{i}.value"])
            for i in range(self.contract["num_layers"])
        ]
        return result, present

    def _noise(self, sampler, shape):
        draw = np.asarray(sampler(shape))
        if draw.shape != shape or draw.dtype != np.float64 or not np.isfinite(draw).all():
            raise ValueError(
                f"Sigma noise_sampler must return finite float64 noise shaped {shape}, "
                f"got {draw.dtype} {draw.shape}."
            )
        return draw

    def _check_block(self, committed_length, block):
        if committed_length + block > self.contract["max_context"]:
            raise ValueError(
                "Sigma's next complete draft block exceeds its context window; "
                "shorten the request (reserve one block of context headroom)."
            )

    def generate_speculative(
        self,
        input_ids,
        *,
        attention_mask=None,
        max_new_tokens=128,
        noise_sampler: NoiseSampler | None = None,
        return_stats=False,
    ):
        """Draft once, verify causally, commit the agreeing prefix and AR bonus.

        Like upstream, this route is batch-one, does not stop at EOS, and uses
        neither self-conditioning nor guidance. Returned statistics exclude the
        bonus token from accepted length.
        """
        ids, valid = self._inputs(input_ids, attention_mask, max_new_tokens)
        if ids.shape[0] != 1:
            raise ValueError("Sigma self-speculation supports batch size 1; split the batch.")
        if max_new_tokens == 0:
            return (ids, {}) if return_stats else ids
        sampler = noise_sampler or np.random.default_rng().standard_normal
        cache = self._empty_cache(1)
        positions = np.maximum(valid.astype(np.int64).cumsum(axis=1) - 1, 0)
        lengths = valid.sum(axis=1)
        result, cache = self._run(ids, positions, valid, cache)
        next_ar = result["ar_logits"][:, -1].argmax(axis=-1)
        gamma = float(result["gamma_bounds"][1])
        block = self.contract["block_size"]
        rounds = accepted = committed = 0
        while committed < max_new_tokens:
            start = ids.shape[1]
            self._check_block(start, block)
            positions = (
                lengths[:, None] + committed + np.arange(block, dtype=np.int64)[None, :]
            )
            z = self._noise(sampler, (1, block, self.contract["embed_dim"]))
            draft_result, _ = self._run(
                np.zeros((1, block), np.int64), positions, valid, cache, z=z, gamma=gamma
            )
            draft = draft_result["diffusion_logits"].argmax(axis=-1)
            draft_valid = np.concatenate((valid, np.ones((1, block), np.bool_)), axis=1)
            verify, tentative = self._run(draft, positions, draft_valid, cache)
            predictions = verify["ar_logits"].argmax(axis=-1)
            expected = np.concatenate((next_ar[:, None], predictions[:, :-1]), axis=1)
            count = int(np.cumprod((draft == expected)[0]).sum())
            bonus = next_ar if count == 0 else predictions[:, count - 1]
            take = min(count + 1, max_new_tokens - committed)
            new_ids = np.concatenate((draft[:, :count], bonus[:, None]), axis=1)[:, :take]
            ids = np.concatenate((ids, new_ids), axis=1)
            accepted += min(count, take)
            committed += take
            rounds += 1
            keep = start + min(count, take)
            cache = [(key[:, :, :keep, :], value[:, :, :keep, :]) for key, value in tentative]
            valid = np.concatenate((valid, np.ones((1, min(count, take)), np.bool_)), axis=1)
            if take == count + 1:
                valid = np.concatenate((valid, np.ones((1, 1), np.bool_)), axis=1)
                bonus_position = lengths[:, None] + committed - 1
                encoded, cache = self._run(bonus[:, None], bonus_position, valid, cache)
                next_ar = encoded["ar_logits"][:, -1].argmax(axis=-1)
            elif count > 0:
                next_ar = predictions[:, take - 1]
        stats = {
            "rounds": rounds,
            "accepted": accepted,
            "committed": committed,
            "mean_accepted_length": accepted / rounds,
        }
        return (ids, stats) if return_stats else ids

    def generate(
        self,
        input_ids,
        *,
        attention_mask=None,
        max_new_tokens=128,
        noise_sampler: NoiseSampler | None = None,
    ):
        """Native two-transition fresh-x0 diffusion with a final readout.

        Uses the checkpoint's fixed log-SNR grid, guidance/score temperature 1,
        32-token blocks and argmax readout. Self-conditioning carries only the
        previous conditional mean. Returns ``(ids, denoiser_evaluations)``.
        """
        ids, valid = self._inputs(input_ids, attention_mask, max_new_tokens)
        if max_new_tokens == 0:
            return ids, 0
        sampler = noise_sampler or np.random.default_rng().standard_normal
        batch, prompt = ids.shape
        cache = self._empty_cache(batch)
        positions = np.maximum(valid.astype(np.int64).cumsum(axis=1) - 1, 0)
        lengths = valid.sum(axis=1)
        result, cache = self._run(ids, positions, valid, cache)
        bounds = result["gamma_bounds"].astype(np.float64)
        grid = np.linspace(bounds[1], bounds[0], self.contract["native_nfe"] + 1)
        block = self.contract["block_size"]
        generated = evaluations = 0
        shape = (batch, block, self.contract["embed_dim"])
        while generated < max_new_tokens:
            self._check_block(ids.shape[1], block)
            positions = (
                lengths[:, None] + generated + np.arange(block, dtype=np.int64)[None, :]
            )
            z = self._noise(sampler, shape)
            condition = None
            for index, gamma in enumerate(grid):
                result, _ = self._run(
                    np.zeros((batch, block), np.int64),
                    positions,
                    valid,
                    cache,
                    z=z,
                    gamma=gamma,
                    condition=condition,
                )
                evaluations += 1
                if index + 1 < len(grid):
                    condition = result["denoised_mean"]
                    next_gamma = grid[index + 1]
                    alpha = np.sqrt(1 / (1 + np.exp(next_gamma)))
                    sigma = np.sqrt(1 / (1 + np.exp(-next_gamma)))
                    z = alpha * condition.astype(np.float64) + sigma * self._noise(
                        sampler, shape
                    )
            tokens = result["diffusion_logits"].argmax(axis=-1)
            ids = np.concatenate((ids, tokens), axis=1)
            generated += block
            if generated < max_new_tokens:
                valid = np.concatenate((valid, np.ones((batch, block), np.bool_)), axis=1)
                _, cache = self._run(tokens, positions, valid, cache)
        return ids[:, : prompt + max_new_tokens], evaluations
