# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate genai_config.json for onnxruntime-genai (Model Package format).

This module takes an ``ArchitectureConfig`` (or ``BaseModelConfig``) and a
model type string and produces the ``genai_config.json`` *base* document
that onnxruntime-genai expects in the Model Package layout.

The generated document is **EP-agnostic**: it carries no
``session_options``, ``provider_options``, ``filename``, or
EP-derived search defaults (``past_present_share_buffer``, KV-buffer
``max_length`` cap). All EP-specific concerns live in each variant's
``variant.json`` (written by a downstream packager); per-variant
GenAI overrides land via the variant's
``consumer_metadata.genai_config_overlay``.

What stays here is GenAI architecture identity (``model.type``,
hidden/head sizes, layer counts, vocab size, token IDs, context
length), per-component I/O name maps, and EP-agnostic ``search``
defaults. Each component block now carries a ``"component"`` field
naming the package component the role binds to (e.g.
``"component": "decoder"``, ``"component": "vision_encoder"``).

This module does NOT import from core model/task/component layers —
it only reads config dataclass fields.
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


def _default_search_params(*, context_length: int) -> dict[str, Any]:
    """Return EP-agnostic default search parameters.

    EP-specific knobs (``past_present_share_buffer``, KV-buffer
    ``max_length`` caps) are omitted — they belong in the variant's
    overlay or runtime config, not in the package-shipped base.
    """
    return {
        "do_sample": True,
        "early_stopping": True,
        "max_length": context_length,
        "min_length": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
    }


class GenaiConfigGenerator:
    """Generates the EP-agnostic ``genai_config.json`` base document.

    The output document is the *base* in the Model Package world:
    every variant in the package merges its overlay onto this document
    at runtime. The document declares only architecture identity and
    role↔component bindings, never EP-specific settings.

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
        bos_token_id: Beginning-of-sequence token ID.
        eos_token_id: End-of-sequence token ID(s).
        pad_token_id: Padding token ID.
        decoder_inputs: Explicit decoder input name mapping. When
            provided (e.g. from ONNX graph introspection), used
            directly instead of the default mapping from
            :func:`_default_decoder_inputs`. Must already include KV
            cache template entries (``past_key_names``,
            ``past_value_names``).
        decoder_component: Name of the package component the
            ``decoder`` role binds to. Defaults to ``"decoder"``.
            Emitted as ``model.decoder.component``.
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
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
        decoder_inputs: dict[str, str] | None = None,
        decoder_component: str = "decoder",
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.context_length = context_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Explicit decoder inputs (from graph introspection); None → use defaults
        self._decoder_inputs = decoder_inputs
        # Package component name for the decoder role
        self._decoder_component = decoder_component

        # Optional VLM fields (set via with_vision())
        self._vision: dict[str, Any] | None = None
        self._embedding: dict[str, Any] | None = None
        self._vlm_token_ids: dict[str, int] = {}

        # Optional audio fields (set via with_audio())
        self._audio: dict[str, Any] | None = None

        # Search config overrides applied in generate()
        self._search_overrides: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls,
        config: Any,
        model_type: str,
        *,
        context_length: int = 4096,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
        decoder_inputs: dict[str, str] | None = None,
        decoder_component: str = "decoder",
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
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad,
            decoder_inputs=decoder_inputs,
            decoder_component=decoder_component,
        )

    def with_vision(
        self,
        *,
        image_token_id: int,
        spatial_merge_size: int | None = 2,
        config_filename: str = "image_processor.json",
        input_names: dict[str, str] | None = None,
        output_names: dict[str, str] | None = None,
        embedding_input_names: dict[str, str] | None = None,
        vision_start_token_id: int | None = None,
        video_token_id: int | None = None,
        vision_component: str = "vision_encoder",
        embedding_component: str = "embedding",
    ) -> GenaiConfigGenerator:
        """Add VLM vision + embedding sections.

        Args:
            image_token_id: Token ID for image placeholders. Required —
                ORT-GenAI crashes without it.
            spatial_merge_size: Spatial merge size for position ID
                computation. Set to ``None`` to omit (e.g. for Phi4MM
                which doesn't use spatial merge).
            config_filename: Vision processor config filename (relative
                to the package's ``configs/`` directory). Defaults to
                ``"image_processor.json"``.
            input_names: Override vision model input name mapping.
                Defaults to pixel_values + image_grid_thw.
            output_names: Override vision model output name mapping.
                Defaults to image_features.
            embedding_input_names: Override embedding model input name
                mapping. When provided (e.g. from ONNX graph
                introspection), used directly. Defaults to
                input_ids + image_features.
            vision_start_token_id: Token ID for ``<|vision_start|>``.
            video_token_id: Token ID for video placeholders.
            vision_component: Package component name for the vision
                role. Defaults to ``"vision_encoder"``. Emitted as
                ``model.vision.component``.
            embedding_component: Package component name for the
                embedding role. Defaults to ``"embedding"``. Emitted
                as ``model.embedding.component``.

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
        if embedding_input_names is None:
            embedding_input_names = {
                "input_ids": "input_ids",
                "image_features": "image_features",
            }

        self._vision = {
            "component": vision_component,
            "config_filename": config_filename,
            "inputs": input_names,
            "outputs": output_names,
        }
        if spatial_merge_size is not None:
            self._vision["spatial_merge_size"] = spatial_merge_size

        self._embedding = {
            "component": embedding_component,
            "inputs": embedding_input_names,
            "outputs": {
                "inputs_embeds": "inputs_embeds",
            },
        }
        self._vlm_token_ids["image_token_id"] = image_token_id
        if vision_start_token_id is not None:
            self._vlm_token_ids["vision_start_token_id"] = vision_start_token_id
        if video_token_id is not None:
            self._vlm_token_ids["video_token_id"] = video_token_id
        return self

    def with_audio(
        self,
        *,
        audio_token_id: int | None = None,
        boa_token_id: int | None = None,
        config_filename: str = "audio_processor.json",
        input_names: dict[str, str] | None = None,
        output_names: dict[str, str] | None = None,
        audio_component: str = "audio_encoder",
    ) -> GenaiConfigGenerator:
        """Add audio model section for multimodal models.

        Args:
            audio_token_id: Token ID for audio placeholders.
            boa_token_id: Beginning-of-audio token ID.
            config_filename: Audio processor config filename (relative
                to the package's ``configs/`` directory). Defaults to
                ``"audio_processor.json"``.
            input_names: Override audio model input name mapping.
                Defaults to audio_embeds + audio_sizes +
                audio_projection_mode.
            output_names: Override audio model output name mapping.
                Defaults to audio_features.
            audio_component: Package component name for the audio
                role. Defaults to ``"audio_encoder"``. Emitted as
                ``model.speech.component``.

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

        self._audio = {
            "component": audio_component,
            "config_filename": config_filename,
            "inputs": input_names,
            "outputs": output_names,
        }

        if audio_token_id is not None:
            self._vlm_token_ids["audio_token_id"] = audio_token_id
        if boa_token_id is not None:
            self._vlm_token_ids["boa_token_id"] = boa_token_id

        return self

    def generate(self) -> dict[str, Any]:
        """Generate the full genai_config.json dict (package-world shape)."""
        is_multimodal = self._vision is not None or self._audio is not None

        # Decoder section — use explicit inputs when available (from
        # graph introspection), otherwise fall back to defaults.
        if self._decoder_inputs is not None:
            decoder_inputs = dict(self._decoder_inputs)
        else:
            decoder_inputs = _default_decoder_inputs(is_vlm=is_multimodal)
        decoder: dict[str, Any] = {
            "component": self._decoder_component,
            "head_size": self.head_dim,
            "hidden_size": self.hidden_size,
            "inputs": decoder_inputs,
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
            # Add audio_features to embedding inputs when speech is
            # enabled and not already present (graph-introspected
            # inputs already include it).
            if self._audio is not None and "audio_features" not in self._embedding["inputs"]:
                self._embedding["inputs"]["audio_features"] = "audio_features"
            model["embedding"] = self._embedding
        if self._audio is not None:
            model["speech"] = self._audio
        model.update(self._vlm_token_ids)

        search = _default_search_params(context_length=self.context_length)
        search.update(self._search_overrides)

        return {
            "model": model,
            "search": search,
        }

    def write(self, output_dir: str) -> str:
        """Write genai_config.json to the output directory.

        Returns the path to the written file. The output directory is
        the package's ``configs/`` directory in the new layout (callers
        in :mod:`mobius.integrations.ort_genai.auto_export` ensure
        that). Creates *output_dir* if it does not yet exist.
        """
        config = self.generate()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "genai_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        return path
