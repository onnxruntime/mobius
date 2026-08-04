# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests: multimodal (vision + language) models.

Verifies that multimodal ONNX models produce the same logits as the
HuggingFace PyTorch reference when given both text and image inputs.
Run with::

    pytest tests/multimodal_integration_test.py -m integration -sv

Note: The smallest multimodal Gemma3 model is 4B parameters. These tests
require significant memory and download time.
"""

from __future__ import annotations

import numpy as np
import pytest

from mobius._testing.comparison import assert_logits_close
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.torch_reference import (
    load_torch_multimodal_model,
    torch_multimodal_forward,
)

_MODELS = [
    pytest.param(
        "google/gemma-3-4b-pt",
        id="gemma3-4b-multimodal",
    ),
]


def _get_multimodal_config(model_id: str):
    """Load ArchitectureConfig for a multimodal model from HuggingFace.

    Extracts text config fields from text_config and vision config fields
    from the top-level vision_config, since ``from_transformers`` on
    text_config alone would lose vision fields.
    """
    import transformers

    from mobius._configs import ArchitectureConfig

    hf_config = transformers.AutoConfig.from_pretrained(model_id)
    text_config = hf_config.text_config
    config = ArchitectureConfig.from_transformers(text_config)

    # Add vision config fields from the top-level config
    hf_vision_config = hf_config.vision_config
    from mobius._configs import VisionConfig

    config.vision = VisionConfig(
        hidden_size=hf_vision_config.hidden_size,
        intermediate_size=hf_vision_config.intermediate_size,
        num_hidden_layers=hf_vision_config.num_hidden_layers,
        num_attention_heads=hf_vision_config.num_attention_heads,
        image_size=hf_vision_config.image_size,
        patch_size=hf_vision_config.patch_size,
        norm_eps=getattr(hf_vision_config, "layer_norm_eps", 1e-6),
        mm_tokens_per_image=getattr(hf_config, "mm_tokens_per_image", None),
    )
    config.mm_tokens_per_image = getattr(hf_config, "mm_tokens_per_image", None)
    config.image_token_id = getattr(hf_config, "image_token_id", None)

    return config


def _build_multimodal_onnx(model_id: str):
    """Build a multimodal ONNX model with weights.

    Uses ``build_from_module`` with ``VisionLanguageTask`` to produce
    a 3-model package (decoder, vision, embedding).
    """
    from mobius import apply_weights, build_from_module
    from mobius._weight_loading import _download_weights
    from mobius.models.gemma3 import Gemma3MultiModalModel
    from mobius.tasks import VisionLanguageTask

    config = _get_multimodal_config(model_id)
    module = Gemma3MultiModalModel(config)
    pkg = build_from_module(module, config, task=VisionLanguageTask())

    # Download and apply weights
    state_dict = _download_weights(model_id)
    state_dict = module.preprocess_weights(state_dict)
    # Apply weights to each component
    for model in pkg.values():
        apply_weights(model, state_dict)

    return pkg, config


def _create_dummy_pixel_values(config, batch_size: int = 1) -> np.ndarray:
    """Create random pixel values for testing."""
    image_size = (config.vision.image_size if config.vision else None) or 224
    rng = np.random.default_rng(42)
    return rng.standard_normal((batch_size, 3, image_size, image_size)).astype(np.float32)


def _get_test_device_kwargs() -> dict:
    """Return OnnxModelSession device kwargs from environment variables.

    Set ``MOBIUS_TEST_DEVICE`` to ``cuda`` to run on GPU; ``MOBIUS_TEST_EP``
    to override the execution provider.
    """
    import os

    kwargs: dict = {}
    device = os.environ.get("MOBIUS_TEST_DEVICE", "").lower()
    if device:
        kwargs["device"] = device
    ep = os.environ.get("MOBIUS_TEST_EP", "")
    if ep:
        kwargs["providers"] = [ep]
    return kwargs


def _run_onnx_multimodal_pipeline(
    pkg,
    config,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    pixel_values: np.ndarray | None,
    past_kv: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Run the 3-model vision-language pipeline (vision → embedding → decoder).

    Gemma3 multimodal is exported as a 3-model ``ModelPackage`` (``vision_encoder``,
    ``embedding``, ``decoder``) rather than a single fused graph. This helper
    chains them: ``pixel_values`` → vision → ``image_features``; then
    ``input_ids`` + ``image_features`` → embedding → ``inputs_embeds``; then
    ``inputs_embeds`` + KV cache → decoder → ``logits`` + present KV.

    Returns the decoder outputs dict (``logits`` and ``present.N.{key,value}``).
    """
    device_kwargs = _get_test_device_kwargs()

    # --- Step 1: Vision encoder → image_features (2D: [num_img_tokens, hidden]) ---
    image_features = None
    if pixel_values is not None:
        vis_session = OnnxModelSession(pkg["vision_encoder"], **device_kwargs)
        try:
            vis_out = vis_session.run({"pixel_values": pixel_values})
            image_features = vis_out["image_features"]
        finally:
            vis_session.close()

    # --- Step 2: Embedding → inputs_embeds ---
    emb_session = OnnxModelSession(pkg["embedding"], **device_kwargs)
    try:
        emb_feeds: dict[str, np.ndarray] = {"input_ids": input_ids}
        for name in emb_session.input_names:
            if name in emb_feeds:
                continue
            if name == "image_features" and image_features is not None:
                emb_feeds[name] = image_features
            else:
                # Empty / unused feature input for a text-only or decode step.
                shape = emb_session.get_input_shape(name) or []
                static = [d if isinstance(d, int) and d > 0 else 0 for d in shape]
                emb_feeds[name] = np.zeros(static, dtype=np.float32)
        emb_out = emb_session.run(emb_feeds)
    finally:
        emb_session.close()
    inputs_embeds = emb_out[next(iter(emb_out))]

    # --- Step 3: Decoder → logits + present KV ---
    dec_key = "decoder" if "decoder" in pkg else "model"
    dec_session = OnnxModelSession(pkg[dec_key], **device_kwargs)
    try:
        embeds_dtype = dec_session.get_input_dtype("inputs_embeds")
        if embeds_dtype is not None and inputs_embeds.dtype != embeds_dtype:
            inputs_embeds = inputs_embeds.astype(embeds_dtype)
        dec_feeds: dict[str, np.ndarray] = {"inputs_embeds": inputs_embeds}
        if past_kv is None:
            for i in range(config.num_hidden_layers):
                dec_feeds[f"past_key_values.{i}.key"] = np.zeros(
                    (1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
                )
                dec_feeds[f"past_key_values.{i}.value"] = np.zeros(
                    (1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
                )
        else:
            dec_feeds.update(past_kv)
        for name in dec_session.input_names:
            if name in dec_feeds:
                continue
            if name == "attention_mask":
                dec_feeds[name] = attention_mask
            elif name == "position_ids":
                dec_feeds[name] = position_ids
            elif name in emb_out:
                # Wire any extra embedding outputs (e.g. per_layer_inputs).
                dec_feeds[name] = emb_out[name]
        outputs = dec_session.run(dec_feeds)
    finally:
        dec_session.close()
    return outputs


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.parametrize("model_id", _MODELS)
class TestMultimodalForwardNumerical:
    """Compare single forward pass logits between multimodal ONNX and PyTorch."""

    def test_prefill_logits_match(self, model_id: str):
        """Prefill forward pass with text + dummy image input."""
        onnx_model, config = _build_multimodal_onnx(model_id)
        torch_model, tokenizer, _processor = load_torch_multimodal_model(model_id)

        # Create text input with an image placeholder token
        image_token_id = config.image_token_id
        # Build a prompt: some text tokens + image token + more text
        prompt = "Describe the image"
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)

        # Insert image tokens at position 1 (after BOS).
        # The vision encoder produces mm_tokens_per_image features, so the
        # input must contain exactly that many image_token_id tokens.
        mm_tokens = getattr(config, "mm_tokens_per_image", None) or 1
        if image_token_id is not None:
            img_tokens = np.full((1, mm_tokens), image_token_id, dtype=np.int64)
            input_ids = np.concatenate(
                [input_ids[:, :1], img_tokens, input_ids[:, 1:]], axis=1
            )

        seq_len = input_ids.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
        pixel_values = _create_dummy_pixel_values(config)

        # PyTorch forward
        torch_logits, _ = torch_multimodal_forward(
            torch_model, input_ids, attention_mask, position_ids, pixel_values
        )

        # ONNX forward through the 3-model (vision → embedding → decoder) pipeline
        onnx_outputs = _run_onnx_multimodal_pipeline(
            onnx_model, config, input_ids, attention_mask, position_ids, pixel_values
        )
        onnx_logits = onnx_outputs["logits"]

        assert_logits_close(onnx_logits, torch_logits, rtol=1e-2, atol=1e-2)

    def test_decode_step_after_multimodal_prefill(self, model_id: str):
        """Decode step after multimodal prefill (text-only, with KV cache)."""
        onnx_model, config = _build_multimodal_onnx(model_id)
        torch_model, tokenizer, _processor = load_torch_multimodal_model(model_id)

        # Prefill with text + image
        prompt = "What is this"
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)

        image_token_id = config.image_token_id
        mm_tokens = getattr(config, "mm_tokens_per_image", None) or 1
        if image_token_id is not None:
            img_tokens = np.full((1, mm_tokens), image_token_id, dtype=np.int64)
            input_ids = np.concatenate(
                [input_ids[:, :1], img_tokens, input_ids[:, 1:]], axis=1
            )

        seq_len = input_ids.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
        pixel_values = _create_dummy_pixel_values(config)

        # Prefill on both models
        torch_logits_1, torch_kv = torch_multimodal_forward(
            torch_model, input_ids, attention_mask, position_ids, pixel_values
        )

        onnx_out_1 = _run_onnx_multimodal_pipeline(
            onnx_model, config, input_ids, attention_mask, position_ids, pixel_values
        )

        # Decode step (text-only — no image features in this step)
        next_token = np.argmax(torch_logits_1[:, -1, :], axis=-1, keepdims=True)
        decode_input_ids = next_token.astype(np.int64)
        decode_attention_mask = np.ones((1, seq_len + 1), dtype=np.int64)
        decode_position_ids = np.array([[seq_len]], dtype=np.int64)

        from mobius._testing.torch_reference import torch_forward

        torch_logits_2, _ = torch_forward(
            torch_model,
            decode_input_ids,
            decode_attention_mask,
            decode_position_ids,
            past_key_values=torch_kv,
        )

        # Carry the present KV cache from prefill into the decode step.
        past_kv: dict[str, np.ndarray] = {}
        for i in range(config.num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = onnx_out_1[f"present.{i}.key"]
            past_kv[f"past_key_values.{i}.value"] = onnx_out_1[f"present.{i}.value"]

        onnx_out_2 = _run_onnx_multimodal_pipeline(
            onnx_model,
            config,
            decode_input_ids,
            decode_attention_mask,
            decode_position_ids,
            pixel_values=None,
            past_kv=past_kv,
        )

        assert_logits_close(onnx_out_2["logits"], torch_logits_2, rtol=1e-2, atol=1e-2)
