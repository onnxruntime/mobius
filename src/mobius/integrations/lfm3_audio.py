# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Portable greedy offline inference over LFM3 Audio's exported components."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

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
    """Build all five offline components, including the separately stored codec.

    Unlike the four-component core ``mobius.build`` export, this also binds
    ``audio_tokenizer/`` weights to ``audio_decoder``. No upstream Python code
    is executed: both configurations are read as data.
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
    optimize_model(decoder, ep=execution_provider, model_role="decoder")
    package["audio_decoder"] = decoder
    return package


class AudioComponentSession(Protocol):
    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


class Lfm3AudioPipeline:
    """Run offline ASR or speech-code generation with reusable hybrid caches.

    Sessions use the named-array ``run(feeds) -> outputs`` contract. Speech
    uses greedy MaskGIT with an explicit permutation for every frame; supplying
    recorded upstream permutations makes reference replay reproducible.
    """

    def __init__(self, config: Lfm3AudioConfig, sessions: Mapping[str, AudioComponentSession]):
        self.config = config
        self.sessions = sessions
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
    ) -> dict[str, np.ndarray]:
        for token_id, count, name in (
            (self.config.audio_token_id, features.shape[0], "audio_features"),
            (self.config.audio_out_token_id, codes.shape[0], "audio_out_codes"),
        ):
            actual = int(np.count_nonzero(ids == token_id))
            if actual != count:
                raise ValueError(
                    f"LFM3 Audio has {actual} placeholders for {name} but {count} feature rows; "
                    "use the pinned processor and trim encoder padding before fusion."
                )
        if (
            codes.ndim != 2
            or codes.shape[1] != self.config.num_codebooks
            or (
                codes.size
                and (np.min(codes) < 0 or np.max(codes) >= self.config.codebook_size)
            )
        ):
            raise ValueError(
                f"audio_out_codes must have shape (frames, {self.config.num_codebooks}) "
                f"with codes in [0, {self.config.codebook_size})."
            )
        embeds = self.sessions["embedding"].run(
            {"input_ids": ids, "audio_features": features, "audio_out_codes": codes}
        )["inputs_embeds"]
        positions = np.maximum(np.cumsum(mask, axis=1) - 1, 0)[:, -ids.shape[1] :]
        outputs = self.sessions["decoder"].run(
            {
                "inputs_embeds": embeds,
                "attention_mask": mask,
                "position_ids": positions,
                **self._cache,
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
    ) -> dict[str, np.ndarray]:
        if input_ids.ndim != 2 or not input_ids.size or input_ids.dtype != np.int64:
            raise ValueError("input_ids must be a nonempty int64 (batch, sequence) array.")
        mask = np.ones_like(input_ids) if attention_mask is None else attention_mask
        if (
            mask.shape != input_ids.shape
            or not np.isin(mask, [0, 1]).all()
            or (not mask[:, -1].all() or (np.diff(mask, axis=1) < 0).any())
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
        self._cache = self._empty_cache(input_ids.shape[0])
        return self._forward(input_ids, mask.astype(np.int64), features, codes)

    def step(
        self, input_ids: np.ndarray, audio_out_codes: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        if self._mask is None:
            raise ValueError("Call prefill before decoding an LFM3 Audio continuation.")
        if input_ids.shape != (self._mask.shape[0], 1):
            raise ValueError("A decode step requires one int64 token per prefilled batch row.")
        mask = np.concatenate([self._mask, np.ones_like(input_ids)], axis=1)
        return self._forward(
            input_ids,
            mask,
            np.empty((0, self.config.hidden_size), dtype=np.float32),
            np.empty((0, self.config.num_codebooks), dtype=np.int64)
            if audio_out_codes is None
            else audio_out_codes,
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

    def sample_frame(self, order: Sequence[int]) -> np.ndarray:
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
            codes[:, indices] = np.argmax(logits[:, indices], axis=-1)
            masked[:, indices] = False
            offset += count
        return codes

    def generate_speech(
        self, max_frames: int = 24, *, codebook_orders: Sequence[Sequence[int]] | None = None
    ) -> np.ndarray:
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
