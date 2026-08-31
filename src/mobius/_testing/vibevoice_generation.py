# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deterministic staged runner used by VibeVoice real-weight and golden tests."""

from __future__ import annotations

import contextlib
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime_easy as ort_easy
import torch

from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mobius._configs import VibeVoiceConfig


_VALID_AUDIO_TOKENS = (151652, 151653, 151654, 151643)


def _torch_to_numpy(value: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    value = value.detach().contiguous().cpu()
    if dtype == np.dtype(ml_dtypes.bfloat16):
        return value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return value.numpy().astype(dtype, copy=False)


def _numpy_to_torch(
    value: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if dtype == torch.bfloat16:
        raw = np.ascontiguousarray(value).view(np.uint16)
        return torch.from_numpy(raw).view(torch.bfloat16).to(device)
    return torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=dtype)


class _DiskSession:
    def __init__(self, path: Path, device: str):
        self._session = ort_easy.load(str(path), device=device)
        self.input_info = {value.name: value for value in self._session.get_inputs()}
        self.output_names = [value.name for value in self._session.get_outputs()]

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        unknown = set(feeds) - set(self.input_info)
        if unknown:
            raise ValueError(f"Inputs are not present in this stage: {sorted(unknown)}")
        ort_feeds = {
            name: _numpy_to_ort_value(np.asarray(value))
            for name, value in feeds.items()
            if name in self.input_info
        }
        outputs = self._session(**ort_feeds)
        return dict(zip(self.output_names, (_ort_value_to_numpy(value) for value in outputs)))

    def close(self) -> None:
        del self._session


@dataclass
class VibeVoiceGenerationResult:
    """Observable outputs from a bounded deterministic VibeVoice generation."""

    generated_tokens: list[int]
    waveform: np.ndarray
    scaled_audio_latents: np.ndarray
    prefill_control_logits: list[float]
    audio_chunk_count: int


class VibeVoiceDiskGenerator:
    """Run a saved Mobius VibeVoice package one ONNX stage at a time."""

    def __init__(
        self,
        package_dir: str | Path,
        config: VibeVoiceConfig,
        *,
        device: str,
        rng_device: str = "cuda",
    ):
        self.package_dir = Path(package_dir)
        self.config = config
        self.device = device
        self.rng_device = torch.device(rng_device)
        dtype_map = {
            ir.DataType.FLOAT: (np.dtype(np.float32), torch.float32),
            ir.DataType.FLOAT16: (np.dtype(np.float16), torch.float16),
            ir.DataType.BFLOAT16: (np.dtype(ml_dtypes.bfloat16), torch.bfloat16),
        }
        try:
            self._numpy_dtype, self._torch_dtype = dtype_map[config.dtype]
        except KeyError as error:
            raise ValueError(
                f"Unsupported VibeVoice generation dtype: {config.dtype}"
            ) from error
        self._control_tokens = (
            config.audio_bos_token_id,
            config.audio_eos_token_id,
            config.audio_token_id,
            int(config.eos_token_id),
        )
        if self._control_tokens != _VALID_AUDIO_TOKENS:
            raise ValueError(
                "The VibeVoice 1.5B golden runner requires the pinned control-token IDs"
            )

    @contextlib.contextmanager
    def _stage(self, name: str) -> Iterator[_DiskSession]:
        session = _DiskSession(self.package_dir / name / "model.onnx", self.device)
        try:
            yield session
        finally:
            session.close()
            del session
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _model_array(self, value) -> np.ndarray:
        return np.asarray(value, dtype=self._numpy_dtype)

    def _empty_kv(self, batch: int) -> dict[str, np.ndarray]:
        shape = (
            batch,
            self.config.num_key_value_heads,
            0,
            self.config.head_dim,
        )
        empty = np.zeros(shape, dtype=self._numpy_dtype)
        return {
            name: empty.copy()
            for layer in range(self.config.num_hidden_layers)
            for name in (
                f"past_key_values.{layer}.key",
                f"past_key_values.{layer}.value",
            )
        }

    @staticmethod
    def _next_kv(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            name.replace("present.", "past_key_values."): value
            for name, value in outputs.items()
            if name.startswith("present.")
        }

    @staticmethod
    def _next_conv(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            name.replace("present_conv.", "past_conv."): value
            for name, value in outputs.items()
            if name.startswith("present_conv.")
        }

    def _empty_conv(
        self,
        session: _DiskSession,
        batch: int,
    ) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {}
        for name, value in session.input_info.items():
            if not name.startswith("past_conv."):
                continue
            shape = list(value.shape)
            shape[0] = batch
            feeds[name] = np.zeros(shape, dtype=self._numpy_dtype)
        return feeds

    @staticmethod
    def _position_ids(attention_mask: np.ndarray, sequence_length: int) -> np.ndarray:
        positions = np.cumsum(attention_mask, axis=-1, dtype=np.int64) - 1
        positions[attention_mask == 0] = 0
        return positions[:, -sequence_length:]

    def _embedding(
        self,
        input_ids: np.ndarray,
        audio_embeds: np.ndarray,
        *,
        replace_audio_tokens: bool,
    ) -> np.ndarray:
        with self._stage("embedding") as session:
            return session.run(
                {
                    "input_ids": input_ids,
                    "audio_embeds": audio_embeds,
                    "replace_audio_tokens": np.array(replace_audio_tokens),
                }
            )["inputs_embeds"]

    def _reference_audio_embeds(
        self,
        processed: dict[str, torch.Tensor],
    ) -> np.ndarray:
        input_values = processed["input_values"].cpu().numpy().astype(np.float32)
        padding_mask = processed["padding_mask"].cpu().numpy().astype(np.bool_)
        batch = input_values.shape[0]
        frames = input_values.shape[-1] // self.config.acoustic_tokenizer.hop_length
        sample_noise = torch.randn(
            batch,
            device=self.rng_device,
            dtype=self._torch_dtype,
        )
        latent_noise = torch.randn(
            (
                batch,
                frames,
                self.config.acoustic_tokenizer.hidden_size,
            ),
            device=self.rng_device,
            dtype=self._torch_dtype,
        )
        with self._stage("audio_encoder") as session:
            latents = session.run(
                {
                    "input_values": input_values,
                    "padding_mask": padding_mask,
                    "sample_noise": _torch_to_numpy(sample_noise, self._numpy_dtype),
                    "latent_noise": _torch_to_numpy(latent_noise, self._numpy_dtype),
                }
            )["audio_latents"]
        with self._stage("audio_projection") as session:
            return session.run(
                {
                    "audio_latents": latents,
                    "latents_are_scaled": np.array(False),
                }
            )["audio_embeds"]

    def _sample_audio_latent(
        self,
        positive_condition: np.ndarray,
        negative_condition: np.ndarray,
        *,
        num_diffusion_steps: int,
        guidance_scale: float,
    ) -> np.ndarray:
        from diffusers import DPMSolverMultistepScheduler

        condition = np.concatenate([positive_condition, negative_condition], axis=0)
        condition_torch = _numpy_to_torch(
            condition,
            self._torch_dtype,
            self.rng_device,
        )
        # HF intentionally creates this draw on CPU in float32 and only then
        # casts it to the diffusion condition's device/dtype.
        noisy = torch.randn(
            condition.shape[0],
            self.config.acoustic_tokenizer.hidden_size,
        ).to(device=self.rng_device, dtype=self._torch_dtype)
        scheduler = DPMSolverMultistepScheduler(
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
        )
        scheduler.set_timesteps(num_inference_steps=num_diffusion_steps)
        half = noisy.shape[0] // 2
        with self._stage("diffusion_head") as session:
            for timestep in scheduler.timesteps:
                combined = torch.cat([noisy[:half], noisy[:half]], dim=0)
                timestep_batch = timestep.repeat(combined.shape[0]).to(combined)
                velocity = session.run(
                    {
                        "noisy_audio_latents": _torch_to_numpy(combined, self._numpy_dtype),
                        "timesteps": _torch_to_numpy(timestep_batch, self._numpy_dtype),
                        "condition": _torch_to_numpy(condition_torch, self._numpy_dtype),
                    }
                )["velocity"]
                velocity_torch = _numpy_to_torch(
                    velocity,
                    self._torch_dtype,
                    self.rng_device,
                )
                conditional, unconditional = torch.split(velocity_torch, half, dim=0)
                guided = unconditional + guidance_scale * (conditional - unconditional)
                velocity_torch = torch.cat([guided, guided], dim=0)
                noisy = scheduler.step(velocity_torch, timestep, noisy).prev_sample
        return _torch_to_numpy(noisy[:half].unsqueeze(1), self._numpy_dtype)

    def generate(
        self,
        processed: dict[str, torch.Tensor],
        *,
        seed: int,
        max_new_tokens: int,
        num_diffusion_steps: int,
        guidance_scale: float,
    ) -> VibeVoiceGenerationResult:
        """Run bounded greedy control-token generation and diffusion decoding."""
        torch.manual_seed(seed)
        if self.rng_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        input_ids = processed["input_ids"].cpu().numpy().astype(np.int64)
        attention_mask = processed["attention_mask"].cpu().numpy().astype(np.int64)
        batch = input_ids.shape[0]
        if batch != 1:
            raise ValueError("The deterministic golden runner currently requires batch=1")

        if "input_values" in processed:
            audio_embeds = self._reference_audio_embeds(processed)
        else:
            audio_embeds = np.zeros(
                (0, self.config.hidden_size),
                dtype=self._numpy_dtype,
            )
        inputs_embeds = self._embedding(
            input_ids,
            audio_embeds,
            replace_audio_tokens=True,
        )

        positive_kv = self._empty_kv(batch)
        negative_kv = self._empty_kv(batch)
        negative_input_ids = np.full(
            (batch, 1),
            self.config.audio_bos_token_id,
            dtype=np.int64,
        )
        negative_attention = np.ones((batch, 1), dtype=np.int64)
        previous_inputs_embeds: np.ndarray | None = None
        acoustic_cache: dict[str, np.ndarray] | None = None
        semantic_cache: dict[str, np.ndarray] | None = None
        generated: list[int] = []
        audio_chunks: list[np.ndarray] = []
        scaled_latents: list[np.ndarray] = []
        prefill_control_logits: list[float] = []

        decoder = _DiskSession(
            self.package_dir / "decoder" / "model.onnx",
            self.device,
        )
        try:
            outputs = decoder.run(
                {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": self._position_ids(
                        attention_mask,
                        inputs_embeds.shape[1],
                    ),
                    **positive_kv,
                }
            )
            positive_kv = self._next_kv(outputs)
            for step in range(max_new_tokens):
                logits = outputs["logits"][0, -1].astype(np.float32)
                control_logits = [float(logits[token]) for token in self._control_tokens]
                if step == 0:
                    prefill_control_logits = control_logits
                token = self._control_tokens[int(np.argmax(control_logits))]
                generated.append(token)
                input_ids = np.concatenate(
                    [input_ids, np.array([[token]], dtype=np.int64)],
                    axis=1,
                )

                token_embed = self._embedding(
                    np.array([[token]], dtype=np.int64),
                    np.zeros((0, self.config.hidden_size), dtype=self._numpy_dtype),
                    replace_audio_tokens=False,
                )
                next_inputs_embeds = token_embed

                if token == self.config.audio_bos_token_id:
                    negative_attention = np.zeros_like(negative_attention)
                    negative_attention[:, -1] = 1
                    negative_input_ids[:, -1] = self.config.audio_bos_token_id
                    for layer in range(self.config.num_hidden_layers):
                        for kind in ("key", "value"):
                            name = f"past_key_values.{layer}.{kind}"
                            if negative_kv[name].shape[2]:
                                negative_kv[name][:, :, -1:, :] = negative_kv[name][
                                    :, :, 0:1, :
                                ]

                if token == self.config.audio_token_id:
                    if previous_inputs_embeds is None:
                        negative_embeds = self._embedding(
                            negative_input_ids[:, -1:],
                            np.zeros(
                                (0, self.config.hidden_size),
                                dtype=self._numpy_dtype,
                            ),
                            replace_audio_tokens=False,
                        )
                    else:
                        negative_embeds = previous_inputs_embeds
                    negative_outputs = decoder.run(
                        {
                            "inputs_embeds": negative_embeds,
                            "attention_mask": negative_attention,
                            "position_ids": self._position_ids(
                                negative_attention,
                                negative_embeds.shape[1],
                            ),
                            **negative_kv,
                        }
                    )
                    negative_condition = negative_outputs["last_hidden_state"][:, -1]
                    negative_kv = self._next_kv(negative_outputs)
                    negative_attention = np.concatenate(
                        [
                            negative_attention,
                            np.ones((batch, 1), dtype=np.int64),
                        ],
                        axis=1,
                    )
                    negative_input_ids = np.concatenate(
                        [
                            negative_input_ids,
                            np.array([[token]], dtype=np.int64),
                        ],
                        axis=1,
                    )
                    positive_condition = outputs["last_hidden_state"][:, -1]
                    scaled_latent = self._sample_audio_latent(
                        positive_condition,
                        negative_condition,
                        num_diffusion_steps=num_diffusion_steps,
                        guidance_scale=guidance_scale,
                    )
                    scaled_latents.append(scaled_latent)

                    with self._stage("audio_projection") as session:
                        acoustic_embed = session.run(
                            {
                                "audio_latents": scaled_latent[:, 0],
                                "latents_are_scaled": np.array(True),
                            }
                        )["audio_embeds"][:, None]

                    with self._stage("audio_decoder") as session:
                        decoder_feeds = (
                            self._empty_conv(session, batch)
                            if acoustic_cache is None
                            else acoustic_cache
                        )
                        audio_outputs = session.run(
                            {
                                "scaled_audio_latents": scaled_latent,
                                **decoder_feeds,
                            }
                        )
                    acoustic_cache = self._next_conv(audio_outputs)
                    waveform = audio_outputs["waveform"]
                    audio_chunks.append(waveform)

                    with self._stage("semantic_encoder") as session:
                        semantic_feeds = (
                            self._empty_conv(session, batch)
                            if semantic_cache is None
                            else semantic_cache
                        )
                        semantic_outputs = session.run(
                            {
                                "waveform": waveform,
                                **semantic_feeds,
                            }
                        )
                    semantic_cache = self._next_conv(semantic_outputs)
                    with self._stage("semantic_projection") as session:
                        semantic_embed = session.run(
                            {"semantic_latents": semantic_outputs["semantic_latents"]}
                        )["semantic_embeds"]
                    next_inputs_embeds = self._model_array(acoustic_embed + semantic_embed)

                if token == int(self.config.eos_token_id):
                    break
                if step + 1 == max_new_tokens:
                    continue

                attention_mask = np.concatenate(
                    [attention_mask, np.ones((batch, 1), dtype=np.int64)],
                    axis=1,
                )
                previous_inputs_embeds = next_inputs_embeds
                outputs = decoder.run(
                    {
                        "inputs_embeds": next_inputs_embeds,
                        "attention_mask": attention_mask,
                        "position_ids": self._position_ids(attention_mask, 1),
                        **positive_kv,
                    }
                )
                positive_kv = self._next_kv(outputs)
        finally:
            decoder.close()
            del decoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        waveform = (
            np.concatenate(audio_chunks, axis=-1).astype(np.float32)
            if audio_chunks
            else np.zeros((batch, 1, 0), dtype=np.float32)
        )
        latent_sequence = (
            np.concatenate(scaled_latents, axis=1).astype(np.float32)
            if scaled_latents
            else np.zeros(
                (batch, 0, self.config.acoustic_tokenizer.hidden_size),
                dtype=np.float32,
            )
        )
        return VibeVoiceGenerationResult(
            generated_tokens=generated,
            waveform=waveform,
            scaled_audio_latents=latent_sequence,
            prefill_control_logits=prefill_control_logits,
            audio_chunk_count=len(audio_chunks),
        )
