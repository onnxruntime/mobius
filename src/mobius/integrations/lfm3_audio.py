# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Portable offline inference over LFM3 Audio's exported components."""

from __future__ import annotations

import functools
import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Protocol

import numpy as np
import onnx_ir as ir

from mobius._configs.lfm3_audio import Lfm3AudioConfig
from mobius._model_package import ModelPackage


def build_lfm3_audio_package(
    model_id: str = "LiquidAI/LFM3-Audio-2B",
    *,
    revision: str = "2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1",
    execution_provider: str = "cpu",
) -> ModelPackage:
    """Build the offline components, including the separately stored codec.

    Unlike the core ``mobius.build`` export, this also binds
    ``audio_tokenizer/`` weights to ``audio_decoder``. LFM3-Omni checkpoints
    ship the same codec. No upstream Python code is executed: both
    configurations are read as data.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from torch import Tensor

    from mobius import build
    from mobius._optimizations import optimize_model
    from mobius.models.liquid_codec import (
        LiquidCodecConfig,
        LiquidCodecDecoder,
        build_liquid_codec_decoder,
    )
    from mobius.tasks._causal_lm import _preserve_standard_attention

    package = build(
        model_id,
        revision=revision,
        dtype="f32",
        load_weights=True,
        execution_provider=execution_provider,
    )
    if not isinstance(package.config, Lfm3AudioConfig):
        raise TypeError(
            f"{model_id!r} is not an LFM3 Audio checkpoint; choose the composite "
            "LiquidAI/LFM3-Audio-2B checkpoint, not its text backbone."
        )
    config_path = hf_hub_download(
        model_id, filename="audio_tokenizer/config.json", revision=revision
    )
    raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    codec_config = LiquidCodecConfig.from_transformers(SimpleNamespace(**raw_config))
    decoder = build_liquid_codec_decoder(codec_config)
    weights_path = hf_hub_download(
        model_id, filename="audio_tokenizer/model.safetensors", revision=revision
    )
    with safe_open(weights_path, framework="pt", device="cpu") as source:
        tensor_names = source.keys()
        weights = LiquidCodecDecoder.preprocess_weights(
            {
                name: source.get_tensor(name)
                for name in tensor_names
                if name.startswith(("quantizer.project_out.", "from_quantizer.", "decoder."))
            }
        )
    for name, value in weights.items():
        if name in decoder.graph.initializers:
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"LiquidCodec weight {name!r} is not a tensor; load the matching "
                    "audio_tokenizer/model.safetensors checkpoint."
                )
            decoder.graph.initializers[name].const_value = ir.tensor(
                value.float().cpu().numpy()
            )
    missing = [
        name for name, value in decoder.graph.initializers.items() if value.const_value is None
    ]
    if missing:
        raise ValueError(
            f"LiquidCodec checkpoint {model_id}@{revision} is missing decoder weights: "
            f"{missing[:5]}; use the matching audio_tokenizer checkpoint."
        )
    # The codec's attention has no KV cache; keep it out of the GQA rewrite.
    _preserve_standard_attention(decoder)
    optimize_model(decoder, ep=execution_provider, model_role="decoder")
    package["audio_decoder"] = decoder
    return package


def load_generation_config(model_id: str, *, revision: str) -> dict[str, object]:
    """Read the checkpoint's ``generation_config.json``: stop ids and sampling settings."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, filename="generation_config.json", revision=revision)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _penalize(scores: np.ndarray, history: np.ndarray, penalty: float | None) -> np.ndarray:
    """Transformers' repetition penalty: divide positive, multiply negative seen scores."""
    if penalty is None or penalty == 1.0 or not history.size:  # noqa: RUF069
        return scores
    seen = np.unique(history)
    scores = scores.copy()
    scores[seen] = np.where(scores[seen] < 0, scores[seen] * penalty, scores[seen] / penalty)
    return scores


def _sample(
    scores: np.ndarray, rng: np.random.Generator, temperature: float, top_k: int | None
) -> int:
    """Temperature, then top-k, then one multinomial draw, as the reference warpers."""
    scores = scores.astype(np.float64) / temperature
    if top_k:
        scores[scores < np.sort(scores)[-min(top_k, scores.size)]] = -np.inf
    probs = np.exp(scores - scores.max())
    return int(rng.choice(scores.size, p=probs / probs.sum()))


class AudioComponentSession(Protocol):
    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


class Lfm3AudioPipeline:
    """Run offline ASR or speech-code generation with reusable hybrid caches.

    Sessions use the named-array ``run(feeds) -> outputs`` contract.
    :meth:`generate_speech` reveals codebooks greedily in an explicit permutation
    for every frame; supplying recorded upstream permutations makes reference
    replay reproducible.
    Prefill and cache steps support batches; generation requires one prompt.
    :meth:`generate` follows the checkpoint's own ``generate`` with the
    settings of ``generation_config``, its ``generation_config.json``.
    """

    # Extra embedding inputs, each with the config field naming its placeholder token.
    _feature_tokens: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        config: Lfm3AudioConfig,
        sessions: Mapping[str, AudioComponentSession],
        generation_config: Mapping[str, object] | None = None,
    ):
        self.config = config
        self.sessions = sessions
        self.generation_config = generation_config
        self._mask: np.ndarray | None = None
        self._cache: dict[str, np.ndarray] = {}
        self.logits: np.ndarray | None = None
        self.hidden_states: np.ndarray | None = None

    def _empty_cache(self, batch: int) -> dict[str, np.ndarray]:
        cache = {}
        for index, kind in enumerate(self.config.layer_types or []):
            prefix = f"past_key_values.{index}"
            if kind == "conv":
                cache[f"{prefix}.conv_state"] = np.zeros(
                    (batch, self.config.hidden_size, self.config.short_conv_kernel - 1),
                    dtype=np.float32,
                )
            else:
                for name in ("key", "value"):
                    cache[f"{prefix}.{name}"] = np.zeros(
                        (batch, self.config.num_key_value_heads, 0, self.config.head_dim),
                        dtype=np.float32,
                    )
        return cache

    def _forward(
        self,
        ids: np.ndarray,
        mask: np.ndarray,
        features: np.ndarray,
        codes: np.ndarray,
        cache: dict[str, np.ndarray],
        **media: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if (
            codes.ndim != 2
            or codes.dtype != np.int64
            or codes.shape[1] != self.config.num_codebooks
            or (
                codes.size
                and (np.min(codes) < 0 or np.max(codes) >= self.config.codebook_size)
            )
        ):
            raise ValueError(
                f"audio_out_codes must be int64 with shape (frames, {self.config.num_codebooks}) "
                f"and codes in [0, {self.config.codebook_size})."
            )
        unknown = set(media) - set(self._feature_tokens)
        if unknown:
            raise TypeError(f"Unexpected media inputs: {sorted(unknown)}.")
        media = {
            name: media.get(name, np.empty((0, self.config.hidden_size), dtype=np.float32))
            for name in self._feature_tokens
        }
        for token_id, count, name in (
            (self.config.audio_token_id, features.shape[0], "audio_features"),
            (self.config.audio_out_token_id, codes.shape[0], "audio_out_codes"),
            *(
                (getattr(self.config, field), media[name].shape[0], name)
                for name, field in self._feature_tokens.items()
            ),
        ):
            actual = int(np.count_nonzero(ids == token_id))
            if actual != count:
                raise ValueError(
                    f"LFM3 Audio has {actual} placeholders for {name} but {count} feature rows; "
                    "use the pinned processor and trim encoder padding before fusion."
                )
        embeds = self.sessions["embedding"].run(
            {"input_ids": ids, **media, "audio_features": features, "audio_out_codes": codes}
        )["inputs_embeds"]
        positions = np.maximum(np.cumsum(mask, axis=1) - 1, 0)[:, -ids.shape[1] :]
        outputs = self.sessions["decoder"].run(
            {
                "inputs_embeds": embeds,
                "attention_mask": mask,
                "position_ids": positions,
                **cache,
            }
        )
        self._mask = mask
        self._cache = {
            name.replace("present.", "past_key_values.", 1): value
            for name, value in outputs.items()
            if name.startswith("present.")
        }
        self.logits, self.hidden_states = outputs["logits"], outputs["hidden_states"]
        return outputs

    def prefill(
        self,
        input_ids: np.ndarray,
        *,
        input_features: np.ndarray | None = None,
        feature_lengths: np.ndarray | None = None,
        attention_mask: np.ndarray | None = None,
        audio_out_codes: np.ndarray | None = None,
        **media: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if input_ids.ndim != 2 or not input_ids.size or input_ids.dtype != np.int64:
            raise ValueError("input_ids must be a nonempty int64 (batch, sequence) array.")
        mask = np.ones_like(input_ids) if attention_mask is None else attention_mask
        if (
            mask.shape != input_ids.shape
            or not np.isin(mask, [0, 1]).all()
            or (not mask[:, -1].all() or (mask[:, :-1] > mask[:, 1:]).any())
        ):
            raise ValueError(
                "attention_mask must match input_ids and describe contiguous left-padded "
                "prompts ending in a valid token."
            )
        features = np.empty((0, self.config.hidden_size), dtype=np.float32)
        if input_features is not None:
            if feature_lengths is None:
                raise ValueError(
                    "input_features requires feature_lengths from the processor's "
                    "feature_attention_mask; padding is not real audio."
                )
            if (
                input_features.ndim != 3
                or input_features.dtype != np.float32
                or input_features.shape[1] != self.config.num_mel_bins
                or feature_lengths.shape != (input_features.shape[0],)
                or feature_lengths.dtype != np.int64
                or (feature_lengths < 1).any()
                or (feature_lengths > input_features.shape[-1]).any()
            ):
                raise ValueError(
                    "input_features must be float32 (clips, mel_bins, frames), with "
                    "one positive int64 feature length per clip bounded by frames."
                )
            encoded = self.sessions["audio_encoder"].run(
                {"input_features": input_features, "feature_lengths": feature_lengths}
            )
            features = np.concatenate(
                [
                    row[:length]
                    for row, length in zip(
                        encoded["audio_features"],
                        encoded["audio_feature_lengths"],
                        strict=True,
                    )
                ]
            )
        codes = (
            np.empty((0, self.config.num_codebooks), dtype=np.int64)
            if audio_out_codes is None
            else audio_out_codes
        )
        return self._forward(
            input_ids,
            mask.astype(np.int64),
            features,
            codes,
            self._empty_cache(input_ids.shape[0]),
            **media,
        )

    def step(
        self, input_ids: np.ndarray, audio_out_codes: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        if self._mask is None:
            raise ValueError("Call prefill before decoding an LFM3 Audio continuation.")
        if input_ids.dtype != np.int64 or input_ids.shape != (self._mask.shape[0], 1):
            raise ValueError("A decode step requires one int64 token per prefilled batch row.")
        mask = np.concatenate([self._mask, np.ones_like(input_ids)], axis=1)
        return self._forward(
            input_ids,
            mask,
            np.empty((0, self.config.hidden_size), dtype=np.float32),
            np.empty((0, self.config.num_codebooks), dtype=np.int64)
            if audio_out_codes is None
            else audio_out_codes,
            self._cache,
        )

    def generate_text(self, max_new_tokens: int = 24) -> np.ndarray:
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive; provide a text generation budget."
            )
        if self.logits is None or self.logits.shape[0] != 1:
            raise ValueError("Greedy text generation requires a single prefilled prompt.")
        eos = self.config.eos_token_id
        eos_ids = eos if isinstance(eos, (tuple, list)) else [eos]
        generated = []
        for _ in range(max_new_tokens):
            assert self.logits is not None
            scores = self.logits[0, -1].copy()
            scores[
                [
                    self.config.audio_token_id,
                    self.config.audio_out_token_id,
                    self.config.mixed_start_token_id,
                    self.config.audio_start_token_id,
                ]
            ] = -np.inf
            token = int(np.argmax(scores))
            generated.append(token)
            if token in eos_ids:
                break
            self.step(np.array([[token]], dtype=np.int64))
        return np.asarray(generated, dtype=np.int64)

    def sample_frame(
        self,
        order: Sequence[int],
        draw: Callable[[np.ndarray], int] | None = None,
        history: np.ndarray | None = None,
        penalty: float | None = None,
    ) -> np.ndarray:
        """Reveal the codebooks in ``order`` over the MaskGIT passes.

        Each code is the argmax, or ``draw``, of its logits after the
        repetition ``penalty`` over that codebook's codes in ``history``
        (frames, codebooks).
        """
        if self.hidden_states is None or self.hidden_states.shape[0] != 1:
            raise ValueError("Speech frame generation requires a single prefilled prompt.")
        if sorted(order) != list(range(self.config.num_codebooks)):
            raise ValueError(
                f"codebook order must be a permutation of [0, {self.config.num_codebooks})."
            )
        codes = np.zeros((1, self.config.num_codebooks), dtype=np.int64)
        masked = np.ones_like(codes, dtype=np.bool_)
        offset = 0
        for count in self.config.maskgit_schedule:
            logits = self.sessions["depth_decoder"].run(
                {"hidden_states": self.hidden_states[:, -1], "codes": codes, "masked": masked}
            )["logits"]
            indices = list(order[offset : offset + count])
            for index in indices:
                scores = logits[0, index]
                if history is not None:
                    scores = _penalize(scores, history[:, index], penalty)
                codes[0, index] = np.argmax(scores) if draw is None else draw(scores)
            masked[:, indices] = False
            offset += count
        return codes

    def generate_speech(
        self, max_frames: int = 24, *, codebook_orders: Sequence[Sequence[int]] | None = None
    ) -> np.ndarray:
        """Use sequential codebook order by default, or replay recorded orders.

        The sequential default is deterministic but differs from upstream's
        randomly permuted MaskGIT schedule, even with greedy code logits.
        """
        if max_frames <= 0:
            raise ValueError("max_frames must be positive; provide a speech frame budget.")
        if codebook_orders is not None and len(codebook_orders) < max_frames:
            raise ValueError(
                "Provide at least max_frames codebook permutations for speech replay."
            )
        frames = []
        for index in range(max_frames):
            order = (
                list(range(self.config.num_codebooks))
                if codebook_orders is None
                else codebook_orders[index]
            )
            codes = self.sample_frame(order)
            if np.any(codes == self.config.audio_eos_code):
                break
            frames.append(codes[0])
            self.step(np.array([[self.config.audio_out_token_id]], dtype=np.int64), codes)
        return (
            np.stack(frames, axis=-1)[None]
            if frames
            else np.empty((1, self.config.num_codebooks, 0), dtype=np.int64)
        )

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int,
        *,
        generation_mode: str | None = None,
        rng: np.random.Generator | None = None,
        codebook_orders: Sequence[Sequence[int]] | None = None,
        draw: Callable[[np.ndarray], int] | None = None,
        **inputs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate text, speech or interleaved text and speech for one prompt.

        Follows the checkpoint's ``generate``: ``generation_mode`` ``"audio"`` or
        ``"interleaved"`` appends the opening marker, ``"text"`` forbids both
        markers, and without a mode the prompt's last token picks it and the
        model may switch. An end-of-speech frame is fed back and text resumes.
        Text is greedy; speech uses the ``generation_config`` settings and
        reveals codebooks in ``rng`` permutations, or in ``codebook_orders``.
        ``draw`` replaces the speech sampler: it gets one codebook's scores
        after the repetition penalty and returns its code. ``inputs`` go to
        :meth:`prefill`; ``max_new_tokens`` counts text tokens and speech frames.

        Returns the generated ids, one ``audio_out_token_id`` per speech frame,
        and the speech codes ``(1, num_codebooks, frames)``.
        """
        settings = self.generation_config
        if settings is None:
            raise ValueError(
                "generate needs the checkpoint's generation_config.json; load it with "
                "load_generation_config and pass it as generation_config."
            )
        if settings.get("do_sample"):
            raise ValueError(
                "Text sampling is not supported; LFM3 checkpoints decode greedily."
            )
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive; provide a generation budget.")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("generate takes one (1, sequence) prompt.")
        config = self.config
        markers = {
            "audio": (config.audio_start_token_id, config.audio_out_token_id),
            "interleaved": (config.mixed_start_token_id,),
            "text": (),
        }
        last = int(input_ids[0, -1])
        mode = generation_mode
        if mode is None:
            mode = next((name for name, ids in markers.items() if last in ids), "text")
        elif mode not in markers:
            raise ValueError(
                f"generation_mode must be one of {sorted(markers)}, got {mode!r}."
            )
        elif markers[mode] and last not in markers[mode]:
            input_ids = np.concatenate([input_ids, [[markers[mode][0]]]], axis=1)
            if inputs.get("attention_mask") is not None:
                inputs["attention_mask"] = np.concatenate(
                    [inputs["attention_mask"], [[1]]], axis=1
                )
        self.prefill(input_ids, **inputs)
        prompt_codes = inputs.get("audio_out_codes")
        if (
            generation_mode is None
            and last == config.audio_out_token_id
            and prompt_codes is not None
            and (prompt_codes[-1] == config.audio_eos_code).any()
        ):
            mode = "text"

        def setting(name: str, default: object = None) -> object:
            value = settings.get(name)
            return default if value is None else value

        eos = setting("eos_token_id", config.eos_token_id)
        eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        never = [
            token
            for token in (
                config.audio_token_id,
                config.audio_out_token_id,
                config.image_token_id,
                config.video_token_id,
            )
            if token is not None
        ]
        if generation_mode == "text":
            never += [config.mixed_start_token_id, config.audio_start_token_id]
        temperature, top_k = setting("audio_temperature"), setting("audio_top_k")
        rng = np.random.default_rng() if rng is None else rng
        if draw is None and not (temperature == 0 or (temperature is None and not top_k)):
            draw = functools.partial(
                _sample, rng=rng, temperature=temperature or 1.0, top_k=top_k
            )

        # Window defaults are the reference generate's.
        text_history = deque(maxlen=setting("text_repetition_window", 32))
        audio_history = deque(maxlen=setting("audio_repetition_window", 64))
        slots, speaking, text_done = config.interleaved_n_text, False, False
        tokens, frames, sampled = [], [], 0
        for step in range(max_new_tokens):
            interleaving, finished = mode == "interleaved", False
            if mode == "audio" or (interleaving and speaking):
                if codebook_orders is None:
                    order = rng.permutation(config.num_codebooks).tolist()
                elif sampled < len(codebook_orders):
                    order = codebook_orders[sampled]
                else:
                    raise ValueError("Provide one codebook order per sampled speech frame.")
                sampled += 1
                frame = self.sample_frame(
                    order,
                    draw,
                    np.stack(audio_history) if audio_history else None,
                    setting("audio_repetition_penalty"),
                )[0]
                if (frame == config.audio_eos_code).any():
                    frame[:] = config.audio_eos_code
                    mode = "text"
                else:
                    audio_history.append(frame)
                    frames.append(frame)
                    tokens.append(config.audio_out_token_id)
                next_token, next_codes = config.audio_out_token_id, frame[None]
            else:
                scores = _penalize(
                    self.logits[0, -1],
                    np.array(text_history, dtype=np.int64),
                    setting("repetition_penalty"),
                ).copy()
                scores[never] = -np.inf
                next_token, next_codes = int(np.argmax(scores)), None
                text_history.append(next_token)
                tokens.append(next_token)
                if next_token == config.text_end_token_id:
                    # Inside an interleaved block it ends the text; the speech tail continues.
                    if interleaving:
                        text_done = speaking = True
                    else:
                        finished = True
                elif next_token in eos_ids:
                    finished = True
                elif not interleaving and next_token == config.audio_start_token_id:
                    mode = "audio"
                elif not interleaving and next_token == config.mixed_start_token_id:
                    mode, speaking, text_done = "interleaved", False, False
                    slots = config.interleaved_n_text
            if interleaving and not text_done:
                slots -= 1
                if not slots:
                    speaking = not speaking
                    slots = (
                        config.interleaved_n_audio if speaking else config.interleaved_n_text
                    )
            if finished or step == max_new_tokens - 1:
                break
            self.step(np.array([[next_token]], dtype=np.int64), next_codes)
        codes = (
            np.stack(frames, axis=-1)[None]
            if frames
            else np.empty((1, config.num_codebooks, 0), dtype=np.int64)
        )
        return np.asarray(tokens, dtype=np.int64), codes

    def synthesize(self, codes: np.ndarray) -> np.ndarray:
        """Decode nonterminal speech codes into a 24-kHz mono waveform."""
        if "audio_decoder" not in self.sessions:
            raise ValueError(
                "No LiquidCodec session is available; export with "
                "build_lfm3_audio_package and open its audio_decoder component."
            )
        if (
            codes.ndim != 3
            or codes.shape[1] != self.config.num_codebooks
            or codes.shape[-1] == 0
            or codes.dtype != np.int64
            or (codes < 0).any()
            or (codes >= self.config.audio_eos_code).any()
        ):
            raise ValueError(
                f"LiquidCodec requires nonempty int64 (batch, {self.config.num_codebooks}, frames) "
                f"codes in [0, {self.config.audio_eos_code}); remove terminal frames first."
            )
        return self.sessions["audio_decoder"].run({"codes": codes})["waveform"]
