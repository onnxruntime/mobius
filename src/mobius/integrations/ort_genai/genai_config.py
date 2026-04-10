# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate genai_config.json for onnxruntime-genai.

This module takes an ``ArchitectureConfig`` (or ``BaseModelConfig``) and a
model type string and produces the config dict that onnxruntime-genai
expects. It does NOT import from core model/task/component layers — it
only reads config dataclass fields.
"""

from __future__ import annotations

import json
import os
from typing import Any


def _default_decoder_inputs(
    *,
    is_vlm: bool,
) -> dict[str, str]:
    """Return decoder input name mapping for genai_config.json."""
    inputs: dict[str, str] = {
        "attention_mask": "attention_mask",
        "position_ids": "position_ids",
        "past_key_names": "past_key_values.%d.key",
        "past_value_names": "past_key_values.%d.value",
    }
    # VLM decoders receive inputs_embeds; LLM decoders receive input_ids
    if is_vlm:
        inputs["inputs_embeds"] = "inputs_embeds"
    else:
        inputs["input_ids"] = "input_ids"
    return inputs


def _default_decoder_outputs() -> dict[str, str]:
    """Return decoder output name mapping for genai_config.json."""
    return {
        "logits": "logits",
        "present_key_names": "present.%d.key",
        "present_value_names": "present.%d.value",
    }


_WEBGPU_MAX_LENGTH_CAP = 4096


def _default_search_params(*, ep: str, context_length: int) -> dict[str, Any]:
    """Return sensible default search parameters.

    Args:
        ep: Execution provider (``"cpu"``, ``"cuda"``, ``"dml"``,
            ``"webgpu"``, ``"trt-rtx"``).  WebGPU sets
            ``past_present_share_buffer`` to ``True`` because the WebGPU
            runtime allocates a fixed KV-cache buffer up-front and requires
            the past and present KV tensors to share the same backing buffer.
        context_length: Model context window; used as the default
            ``max_length`` for generation so the limit matches the model.
    """
    webgpu = ep == "webgpu"
    if webgpu:
        # WebGPU pre-allocates KV-cache for the full max_length at load time.
        # Using the raw context_length (e.g. 128K tokens) would pre-allocate
        # ~8 GB of GPU memory by default.  Cap at _WEBGPU_MAX_LENGTH_CAP so
        # the model loads sensibly on consumer hardware; users can override
        # in genai_config.json for their target device.
        max_length = min(context_length, _WEBGPU_MAX_LENGTH_CAP)
    else:
        max_length = context_length
    return {
        "do_sample": True,
        "early_stopping": True,
        "max_length": max_length,
        "min_length": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        "past_present_share_buffer": webgpu,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
    }


def _make_session_options(ep: str) -> dict[str, Any]:
    """Return session options with EP-specific provider_options.

    Args:
        ep: Execution provider name (e.g. ``"cpu"``, ``"cuda"``,
            ``"dml"``, ``"trt-rtx"``).
    """
    from mobius.integrations.ort_genai.ep_config import make_provider_options

    return {
        "log_id": "onnxruntime-genai",
        "provider_options": make_provider_options(ep),
    }


class GenaiConfigGenerator:
    """Generates genai_config.json dicts for onnxruntime-genai.

    This class takes config fields as plain values (not model internals)
    and assembles the nested dict structure that ORT-GenAI expects.

    Args:
        model_type: The ORT-GenAI model type string (e.g. ``"qwen2"``,
            ``"llama"``, ``"qwen2_5_vl"``).
        vocab_size: Model vocabulary size.
        hidden_size: Decoder hidden dimension.
        num_hidden_layers: Number of decoder transformer layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of KV heads (for GQA).
        head_dim: Size per attention head.
        context_length: Minimum context length written to
            ``genai_config.json``. Overridden upward by
            ``max_position_embeddings`` from the model config when that
            value is larger. Defaults to 4096.
        ep: Execution provider for ``session_options`` (e.g. ``"cpu"``,
            ``"cuda"``, ``"dml"``, ``"trt-rtx"``). Defaults to ``"cpu"``.
        bos_token_id: Beginning-of-sequence token ID.
        eos_token_id: End-of-sequence token ID(s).
        pad_token_id: Padding token ID.
    """

    def __init__(
        self,
        model_type: str,
        *,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        context_length: int = 4096,
        ep: str = "cpu",
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.context_length = context_length
        self.ep = ep
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Optional VLM fields (set via with_vision())
        self._vision: dict[str, Any] | None = None
        self._embedding: dict[str, Any] | None = None
        self._vlm_token_ids: dict[str, int] = {}

        # Optional speech fields (set via with_speech())
        self._speech: dict[str, Any] | None = None

    @classmethod
    def from_config(
        cls,
        config: Any,
        model_type: str,
        *,
        context_length: int = 4096,
        ep: str = "cpu",
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
    ) -> GenaiConfigGenerator:
        """Create a generator from a BaseModelConfig-like dataclass.

        Reads ``vocab_size``, ``hidden_size``, ``num_hidden_layers``,
        ``num_attention_heads``, ``num_key_value_heads``, and ``head_dim``
        from the config object. Token IDs and context_length can be
        overridden since they are often not on the model config.
        """
        pad = pad_token_id
        if pad is None:
            raw_pad = getattr(config, "pad_token_id", None)
            if raw_pad is not None and raw_pad != -42:
                pad = raw_pad

        max_pos = getattr(config, "max_position_embeddings", None)
        if max_pos and max_pos > 0 and max_pos != -42:
            context_length = max(context_length, max_pos)

        return cls(
            model_type,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            context_length=context_length,
            ep=ep,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad,
        )

    def with_vision(
        self,
        *,
        image_token_id: int,
        filename: str = "vision/model.onnx",
        embedding_filename: str = "embedding/model.onnx",
        spatial_merge_size: int | None = 2,
        config_filename: str = "processor_config.json",
        input_names: dict[str, str] | None = None,
        output_names: dict[str, str] | None = None,
        vision_start_token_id: int | None = None,
        video_token_id: int | None = None,
    ) -> GenaiConfigGenerator:
        """Add VLM vision + embedding sections.

        Args:
            image_token_id: Token ID for image placeholders. Required —
                ORT-GenAI crashes without it.
            filename: Vision ONNX model filename.
            embedding_filename: Embedding ONNX model filename.
            spatial_merge_size: Spatial merge size for position ID
                computation. Set to ``None`` to omit (e.g. for Phi4MM
                which doesn't use spatial merge).
            config_filename: Vision processor config filename.
            input_names: Override vision model input name mapping.
                Defaults to pixel_values + image_grid_thw.
            output_names: Override vision model output name mapping.
                Defaults to image_features.
            vision_start_token_id: Token ID for ``<|vision_start|>``.
            video_token_id: Token ID for video placeholders.

        Returns self for chaining.
        """
        if input_names is None:
            input_names = {
                "pixel_values": "pixel_values",
                "image_grid_thw": "image_grid_thw",
            }
        if output_names is None:
            output_names = {
                "image_features": "image_features",
            }

        self._vision = {
            "filename": filename,
            "config_filename": config_filename,
            "inputs": input_names,
            "outputs": output_names,
            "session_options": _make_session_options(self.ep),
        }
        if spatial_merge_size is not None:
            self._vision["spatial_merge_size"] = spatial_merge_size

        self._embedding = {
            "filename": embedding_filename,
            "inputs": {
                "input_ids": "input_ids",
                "image_features": "image_features",
            },
            "outputs": {
                "inputs_embeds": "inputs_embeds",
            },
            "session_options": _make_session_options(self.ep),
        }
        self._vlm_token_ids["image_token_id"] = image_token_id
        if vision_start_token_id is not None:
            self._vlm_token_ids["vision_start_token_id"] = vision_start_token_id
        if video_token_id is not None:
            self._vlm_token_ids["video_token_id"] = video_token_id
        return self

    def with_speech(
        self,
        *,
        audio_token_id: int | None = None,
        filename: str = "speech/model.onnx",
        config_filename: str = "speech_processor.json",
        input_names: dict[str, str] | None = None,
        output_names: dict[str, str] | None = None,
    ) -> GenaiConfigGenerator:
        """Add speech/audio model section for multimodal models.

        Args:
            audio_token_id: Token ID for audio placeholders.
            filename: Speech ONNX model filename.
            config_filename: Speech processor config filename.
            input_names: Override speech model input name mapping.
                Defaults to audio_embeds + audio_sizes +
                audio_projection_mode.
            output_names: Override speech model output name mapping.
                Defaults to audio_features.

        Returns self for chaining.
        """
        if input_names is None:
            input_names = {
                "audio_embeds": "audio_embeds",
                "audio_sizes": "audio_sizes",
                "audio_projection_mode": "audio_projection_mode",
            }
        if output_names is None:
            output_names = {
                "audio_features": "audio_features",
            }

        self._speech = {
            "filename": filename,
            "config_filename": config_filename,
            "inputs": input_names,
            "outputs": output_names,
            "session_options": _make_session_options(self.ep),
        }

        if audio_token_id is not None:
            self._vlm_token_ids["audio_token_id"] = audio_token_id

        return self

    def generate(self) -> dict[str, Any]:
        """Generate the full genai_config.json dict."""
        is_multimodal = self._vision is not None or self._speech is not None

        # Decoder section
        decoder: dict[str, Any] = {
            "session_options": _make_session_options(self.ep),
            "filename": "model.onnx",
            "head_size": self.head_dim,
            "hidden_size": self.hidden_size,
            "inputs": _default_decoder_inputs(is_vlm=is_multimodal),
            "outputs": _default_decoder_outputs(),
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
        }

        # Model section
        model: dict[str, Any] = {
            "type": self.model_type,
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "decoder": decoder,
        }

        if self.bos_token_id is not None:
            model["bos_token_id"] = self.bos_token_id
        if self.eos_token_id is not None:
            model["eos_token_id"] = self.eos_token_id
        if self.pad_token_id is not None:
            model["pad_token_id"] = self.pad_token_id

        # VLM sections
        if self._vision is not None:
            model["vision"] = self._vision
        if self._embedding is not None:
            # Add audio_features to embedding inputs when speech is enabled
            if self._speech is not None:
                self._embedding["inputs"]["audio_features"] = "audio_features"
            model["embedding"] = self._embedding
        if self._speech is not None:
            model["speech"] = self._speech
        model.update(self._vlm_token_ids)

        return {
            "model": model,
            "search": _default_search_params(ep=self.ep, context_length=self.context_length),
        }

    def write(self, output_dir: str) -> str:
        """Write genai_config.json to the output directory.

        Returns the path to the written file.
        """
        config = self.generate()
        path = os.path.join(output_dir, "genai_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        return path
