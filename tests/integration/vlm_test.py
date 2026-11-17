# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests for real-checkpoint vision-language models."""

from __future__ import annotations

import gc
import os

import numpy as np
import pytest
import torch
import transformers
from PIL import Image
from transformers.cache_utils import DynamicCache

from integration._support import (
    _get_config,
    _get_test_device,
    _make_decode_feeds,
    _make_prefill_feeds,
    _make_session,
    _model_accessible,
)
from mobius import build, models
from mobius._testing.comparison import (
    assert_generation_match,
    assert_logits_close,
)
from mobius._testing.generation import OnnxGenerator, torch_generate_greedy
from mobius._testing.torch_reference import (
    load_torch_multimodal_model,
)


@pytest.mark.integration
@pytest.mark.integration_slow
def test_nemotron_parse_real_weight_cuda_parity():
    """Compare real BF16 C-RADIO features and decoder logits on a document image."""
    if _get_test_device() != "cuda" or not torch.cuda.is_available():
        pytest.skip("Nemotron Parse real-weight parity requires CUDA")

    import ml_dtypes
    from transformers import AutoModel, AutoProcessor

    model_id = "nvidia/NVIDIA-Nemotron-Parse-2.0"
    revision = "635b84d9b09bb9526b9a684d0b2c953d3cc3df05"
    prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
    image_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "testdata",
        "nemotron-parse-document.png",
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    processed = processor(
        images=[Image.open(image_path).convert("RGB")],
        text=prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    hf_model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    hf_model.eval()
    pixel_values = processed["pixel_values"].to("cuda")
    decoder_input_ids = processed["input_ids"].to("cuda")
    with torch.no_grad():
        encoder_outputs = hf_model.encoder(pixel_values=pixel_values)
        hf_encoder = encoder_outputs[0].float().cpu().numpy()
        hf_logits = (
            hf_model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
            )
            .logits[:, -1]
            .float()
            .cpu()
            .numpy()
        )
    del hf_model, encoder_outputs
    gc.collect()
    torch.cuda.empty_cache()

    pkg = build(
        model_id,
        dtype="bf16",
        load_weights=True,
        trust_remote_code=True,
        execution_provider="cuda",
    )
    vision_session = _make_session(pkg["vision_encoder"])
    decoder_session = _make_session(pkg["decoder"])
    try:
        onnx_pixel_values = processed["pixel_values"].float().cpu().numpy()
        onnx_encoder = vision_session.run({"pixel_values": onnx_pixel_values})[
            "last_hidden_state"
        ]
        empty_cache = {
            name: np.zeros(
                (1, pkg.config.num_key_value_heads, 0, pkg.config.head_dim),
                dtype=ml_dtypes.bfloat16,
            )
            for name in decoder_session.input_names
            if name.startswith("past_key_values.")
        }
        onnx_logits = decoder_session.run(
            {
                "input_ids": processed["input_ids"].numpy().astype(np.int64),
                "attention_mask": np.ones_like(processed["input_ids"].numpy(), dtype=np.int64),
                "encoder_hidden_states": onnx_encoder,
                **empty_cache,
            }
        )["logits"][:, -1]
    finally:
        vision_session.close()
        decoder_session.close()

    onnx_encoder_f32 = onnx_encoder.astype(np.float32)
    encoder_cosine = np.dot(onnx_encoder_f32.ravel(), hf_encoder.ravel()) / (
        np.linalg.norm(onnx_encoder_f32) * np.linalg.norm(hf_encoder)
    )
    onnx_logits_f32 = onnx_logits.astype(np.float32)
    logits_cosine = np.dot(onnx_logits_f32.ravel(), hf_logits.ravel()) / (
        np.linalg.norm(onnx_logits_f32) * np.linalg.norm(hf_logits)
    )
    assert encoder_cosine > 0.99
    assert logits_cosine > 0.995
    np.testing.assert_array_equal(
        np.argmax(onnx_logits_f32, axis=-1),
        np.argmax(hf_logits, axis=-1),
    )


@pytest.mark.integration
def test_minicpmv4_6_real_weight_vision_parity():
    """Real nonzero pixels match through SigLIP2 and both visual mergers."""
    import gc

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = "openbmb/MiniCPM-V-4.6"
    processor = AutoProcessor.from_pretrained(model_id)
    image = Image.open("testdata/pipeline-cat-chonk.jpeg").convert("RGB")
    # A square source triggers the default overview + slice path with
    # non-uniform patch grids (e.g. 32x32 overview and 40x24 slices).
    image = image.resize((1024, 1024))
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "testdata/pipeline-cat-chonk.jpeg"},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    )

    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float32,
    ).eval()
    expected_parts = []
    start = 0
    with torch.no_grad():
        for size in inputs["target_sizes"]:
            num_patches = int(size.prod())
            end = start + num_patches * 14
            unit_pixels = inputs["pixel_values"][:, :, :, start:end]
            expected_parts.extend(
                hf_model.get_image_features(
                    unit_pixels,
                    size.unsqueeze(0),
                ).pooler_output
            )
            start = end
    expected = torch.cat(expected_parts, dim=0).numpy()
    del hf_model
    gc.collect()

    package = build(model_id, dtype="float32", load_weights=True)
    session = _make_session(package["vision_encoder"])
    actual = session.run(
        {
            "pixel_values": inputs["pixel_values"].numpy(),
            "target_sizes": inputs["target_sizes"].numpy(),
        }
    )["image_features"]
    session.close()

    assert float(np.linalg.norm(inputs["pixel_values"].numpy())) > 0.0
    assert np.unique(inputs["target_sizes"].numpy(), axis=0).shape[0] > 1
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


_VL_TEXT_MODELS = [
    # (model_id, module_class_name, trust_remote_code)
    pytest.param(
        "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen25VLTextModel", False, id="qwen2.5-vl-3b-text"
    ),
    pytest.param(
        "Qwen/Qwen3-VL-2B-Instruct", "Qwen3VLTextModel", False, id="qwen3-vl-2b-text"
    ),
]


def _vl_text_forward(model, input_ids, attention_mask, position_ids, past_key_values=None):
    """Text-only forward pass on a HuggingFace VL model (no pixel_values).

    Calls the model without visual inputs so only the text decoder runs.
    """
    device = next(model.parameters()).device

    ids_t = torch.from_numpy(input_ids).to(device)
    mask_t = torch.from_numpy(attention_mask).to(device)
    pos_t = torch.from_numpy(position_ids).to(device)

    kwargs: dict = {
        "input_ids": ids_t,
        "attention_mask": mask_t,
        "position_ids": pos_t,
        "use_cache": True,
    }

    if past_key_values is not None:
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_key_values):
            cache.update(
                torch.from_numpy(k).to(device),
                torch.from_numpy(v).to(device),
                layer_idx,
            )
        kwargs["past_key_values"] = cache

    with torch.no_grad():
        outputs = model(**kwargs)

    logits = outputs.logits.cpu().numpy()

    present_kv = []
    cache = outputs.past_key_values
    for layer_idx in range(len(cache.layers)):
        k = cache.layers[layer_idx].keys.cpu().numpy()
        v = cache.layers[layer_idx].values.cpu().numpy()
        present_kv.append((k, v))

    return logits, present_kv


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.parametrize("model_id,module_class_name,trust_remote_code", _VL_TEXT_MODELS)
class TestVLTextForward:
    """Text-only forward pass parity for VL models.

    Builds the ONNX text-only variant (stripping visual weights) and
    compares against the HuggingFace VL model called without pixel_values.
    """

    def test_prefill_logits_match(
        self,
        model_id: str,
        module_class_name: str,
        trust_remote_code: bool,
    ):
        """Prefill with a short prompt, no image inputs."""
        module_class = getattr(models, module_class_name)
        onnx_model = build(
            model_id,
            module_class=module_class,
            task="text-generation",
            dtype="f32",
            load_weights=True,
        )

        torch_model, _, _ = load_torch_multimodal_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        prompt = "The capital of France is"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        torch_logits, _ = _vl_text_forward(
            torch_model,
            input_ids,
            attention_mask,
            position_ids,
        )

        session = _make_session(onnx_model)
        feeds = _make_prefill_feeds(config, input_ids, attention_mask, position_ids)
        onnx_outputs = session.run(feeds)
        session.close()

        assert_logits_close(onnx_outputs["logits"], torch_logits, rtol=1e-3, atol=1e-3)

    def test_decode_step_logits_match(
        self,
        model_id: str,
        module_class_name: str,
        trust_remote_code: bool,
    ):
        """Single-token decode step with KV cache, no image inputs."""
        module_class = getattr(models, module_class_name)
        onnx_model = build(
            model_id,
            module_class=module_class,
            task="text-generation",
            dtype="f32",
            load_weights=True,
        )

        torch_model, _, _ = load_torch_multimodal_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        prompt = "Hello world"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        # Prefill
        torch_logits_1, torch_kv = _vl_text_forward(
            torch_model,
            input_ids,
            attention_mask,
            position_ids,
        )

        session = _make_session(onnx_model)
        feeds = _make_prefill_feeds(config, input_ids, attention_mask, position_ids)
        onnx_out_1 = session.run(feeds)

        # Decode step
        next_token = np.argmax(torch_logits_1[:, -1, :], axis=-1, keepdims=True)
        decode_input_ids = next_token.astype(np.int64)
        decode_attention_mask = np.ones((1, seq_len + 1), dtype=np.int64)
        decode_position_ids = np.array([[seq_len]], dtype=np.int64)

        torch_logits_2, _ = _vl_text_forward(
            torch_model,
            decode_input_ids,
            decode_attention_mask,
            decode_position_ids,
            past_key_values=torch_kv,
        )

        decode_feeds = _make_decode_feeds(
            config,
            decode_input_ids,
            decode_attention_mask,
            decode_position_ids,
            onnx_out_1,
        )
        onnx_out_2 = session.run(decode_feeds)
        session.close()

        assert_logits_close(onnx_out_2["logits"], torch_logits_2, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.parametrize("model_id,module_class_name,trust_remote_code", _VL_TEXT_MODELS)
class TestVLTextGeneration:
    """Compare greedy text generation between ONNX (text-only VL) and PyTorch."""

    def test_generate_tokens_match(
        self,
        model_id: str,
        module_class_name: str,
        trust_remote_code: bool,
    ):
        """Generated token IDs should be identical for greedy decoding."""
        module_class = getattr(models, module_class_name)
        onnx_model = build(
            model_id,
            module_class=module_class,
            task="text-generation",
            dtype="f32",
            load_weights=True,
        )

        torch_model, _, _ = load_torch_multimodal_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        prompt = "Once upon a time"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        max_new = 20

        session = _make_session(onnx_model)
        generator = OnnxGenerator(session, config)
        onnx_ids = generator.generate(
            input_ids,
            max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id,
        )
        session.close()

        torch_ids = torch_generate_greedy(
            torch_model,
            input_ids,
            max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id,
        )

        onnx_text = tokenizer.decode(onnx_ids[0], skip_special_tokens=True)
        torch_text = tokenizer.decode(torch_ids[0], skip_special_tokens=True)
        print(f"\n[{model_id} text-only] ONNX:  {onnx_text!r}")
        print(f"[{model_id} text-only] Torch: {torch_text!r}")

        assert_generation_match(onnx_ids[0].tolist(), torch_ids[0].tolist())


_VL_3MODEL_MODELS = [
    pytest.param(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        id="qwen2.5-vl-3b-3model",
    ),
]


def _build_qwen25vl_3model(model_id: str):
    """Build Qwen2.5-VL 3-model package with real weights."""
    pkg = build(model_id, dtype="f32", load_weights=True)
    return pkg


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.parametrize("model_id", _VL_3MODEL_MODELS)
class TestQwen25VL3Model:
    """Integration tests for Qwen2.5-VL 3-model split (decoder, vision, embedding).

    Verifies:
    - All 3 models build with correct weights
    - Decoder produces correct logits compared to HF text-only forward
    - Embedding model correctly fuses text + image features
    """

    def test_all_weights_assigned(self, model_id: str):
        """Verify every ONNX initializer has weights (no missing weights)."""
        pkg = _build_qwen25vl_3model(model_id)

        assert "decoder" in pkg, "Package should contain 'decoder' (decoder)"
        assert "vision_encoder" in pkg, "Package should contain 'vision_encoder'"
        assert "embedding" in pkg, "Package should contain 'embedding'"

        for name, model in pkg.items():
            for init_name, init in model.graph.initializers.items():
                if init_name.startswith("const_"):
                    continue
                assert init.const_value is not None, (
                    f"[{name}] Initializer '{init_name}' has no weights"
                )

    def test_decoder_prefill_logits_match(self, model_id: str):
        """Decoder produces logits matching HF text-only forward.

        Runs the embedding model to get inputs_embeds, then the decoder.
        Compares combined result vs HF text-only forward.
        """
        pkg = _build_qwen25vl_3model(model_id)
        config = _get_config(model_id)

        torch_model, _, _ = load_torch_multimodal_model(model_id)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        prompt = "The capital of France is"
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        seq_len = input_ids.shape[1]
        # For text-only, all 3 MRoPE dims are equal
        pos_1d = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
        position_ids_3d = np.stack([pos_1d, pos_1d, pos_1d], axis=0)  # (3, 1, seq)

        # HF reference (text-only forward, no image)
        torch_logits, _ = _vl_text_forward(
            torch_model,
            input_ids,
            attention_mask,
            pos_1d,
        )

        # ONNX: embedding → dummy image features (no images in text-only)
        # Pass at least 1 dummy row since Gather runs eagerly
        embedding_session = _make_session(pkg["embedding"])
        embed_feeds = {
            "input_ids": input_ids,
            "image_features": np.zeros((1, config.hidden_size), dtype=np.float32),
        }
        embed_out = embedding_session.run(embed_feeds)
        embedding_session.close()
        inputs_embeds = embed_out["inputs_embeds"]

        # ONNX: decoder
        decoder_session = _make_session(pkg["decoder"])
        decoder_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids_3d,
        }
        for i in range(config.num_hidden_layers):
            kv_shape = (1, config.num_key_value_heads, 0, config.head_dim)
            decoder_feeds[f"past_key_values.{i}.key"] = np.zeros(kv_shape, dtype=np.float32)
            decoder_feeds[f"past_key_values.{i}.value"] = np.zeros(kv_shape, dtype=np.float32)

        decoder_out = decoder_session.run(decoder_feeds)
        decoder_session.close()

        assert_logits_close(
            decoder_out["logits"],
            torch_logits,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_3model_vision_pipeline(self, model_id: str):
        """Run full 3-model pipeline (vision→embedding→decoder) with image.

        Processes a real image through all 3 ONNX models and compares
        the decoder logits against the HuggingFace single-model forward.
        This guards against regressions in vision encoding, embedding
        fusion, and the genai_config fields needed for correct MRoPE.
        """
        pkg = _build_qwen25vl_3model(model_id)
        config = _get_config(model_id)

        # HF reference: full VL forward with image
        torch_model, _, _ = load_torch_multimodal_model(model_id)
        processor = transformers.AutoProcessor.from_pretrained(model_id)

        image = Image.open("testdata/pipeline-cat-chonk.jpeg")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        hf_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            hf_out = torch_model(**hf_inputs, use_cache=False)
        hf_logits = hf_out.logits.cpu().numpy()

        # Step 1: ONNX vision model — process pixel_values + grid_thw
        pixel_values = hf_inputs["pixel_values"].numpy().astype(np.float32)
        grid_thw = hf_inputs["image_grid_thw"].numpy().astype(np.int64)

        vision_session = _make_session(pkg["vision_encoder"])
        vision_out = vision_session.run(
            {
                "pixel_values": pixel_values,
                "image_grid_thw": grid_thw,
            }
        )
        vision_session.close()
        image_features = vision_out["image_features"]

        # Verify vision model output has expected shape
        # Each image produces (t * h/merge * w/merge) patches
        merge_size = config.spatial_merge_size or 2
        t, h, w = grid_thw[0]
        expected_patches = int(t * (h // merge_size) * (w // merge_size))
        assert image_features.shape[0] == expected_patches, (
            f"Vision output patches {image_features.shape[0]} != "
            f"expected {expected_patches} for grid_thw={grid_thw[0]}"
        )

        # Step 2: ONNX embedding model — fuse text + image features
        input_ids = hf_inputs["input_ids"].numpy().astype(np.int64)

        embedding_session = _make_session(pkg["embedding"])
        embed_out = embedding_session.run(
            {
                "input_ids": input_ids,
                "image_features": image_features,
            }
        )
        embedding_session.close()
        inputs_embeds = embed_out["inputs_embeds"]

        assert inputs_embeds.shape == (
            1,
            input_ids.shape[1],
            config.hidden_size,
        )

        # Step 3: ONNX decoder — run with embedded inputs
        # Compute 3D MRoPE position_ids from HF (ground truth)
        with torch.no_grad():
            embed = torch_model.model.language_model.get_input_embeddings()
            hf_embeds = embed(hf_inputs["input_ids"])
            position_ids_3d = torch_model.model.compute_3d_position_ids(
                input_ids=hf_inputs["input_ids"],
                image_grid_thw=hf_inputs["image_grid_thw"],
                video_grid_thw=None,
                mm_token_type_ids=hf_inputs["mm_token_type_ids"],
                attention_mask=hf_inputs["attention_mask"],
                past_key_values=None,
                inputs_embeds=hf_embeds,
            )
        position_ids = position_ids_3d.numpy().astype(np.int64)
        attention_mask = hf_inputs["attention_mask"].numpy().astype(np.int64)

        decoder_session = _make_session(pkg["decoder"])
        decoder_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for i in range(config.num_hidden_layers):
            kv_shape = (1, config.num_key_value_heads, 0, config.head_dim)
            decoder_feeds[f"past_key_values.{i}.key"] = np.zeros(kv_shape, dtype=np.float32)
            decoder_feeds[f"past_key_values.{i}.value"] = np.zeros(kv_shape, dtype=np.float32)

        decoder_out = decoder_session.run(decoder_feeds)
        decoder_session.close()

        # 3-model pipeline should match HF with slightly looser tolerance
        # (vision + embedding + decoder accumulate small numerical differences)
        assert_logits_close(
            decoder_out["logits"],
            hf_logits,
            rtol=2e-2,
            atol=2e-1,
        )

    def test_vision_features_match_hf(self, model_id: str):
        """ONNX vision encoder features match HuggingFace with cos > 0.999.

        This is a targeted regression guard for:
        - Rotary embedding dimension (must be head_dim//2, not head_dim)
        - fullatt_block_indexes config extraction (windowed vs full attn)
        Both bugs produce cos < 0.3 when broken.
        """
        pkg = _build_qwen25vl_3model(model_id)

        # HF reference: run vision encoder on a real image
        torch_model, _, _ = load_torch_multimodal_model(model_id)
        processor = transformers.AutoProcessor.from_pretrained(model_id)

        image = Image.open("testdata/pipeline-cat-chonk.jpeg")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]
        hf_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # HF vision forward
        with torch.no_grad():
            hf_visual = torch_model.model.visual(
                hf_inputs["pixel_values"],
                grid_thw=hf_inputs["image_grid_thw"],
            )
        # transformers >=5.x returns BaseModelOutputWithPooling; the merged
        # patch features fed to the LLM are ``pooler_output`` (last_hidden_state
        # is the pre-merge sequence).
        if hasattr(hf_visual, "pooler_output"):
            hf_visual = hf_visual.pooler_output
        hf_features = hf_visual.cpu().numpy()

        # ONNX vision forward
        pixel_values = hf_inputs["pixel_values"].numpy().astype(np.float32)
        grid_thw = hf_inputs["image_grid_thw"].numpy().astype(np.int64)

        vision_session = _make_session(pkg["vision_encoder"])
        vision_out = vision_session.run(
            {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
        )
        vision_session.close()
        onnx_features = vision_out["image_features"]

        # Shape must match
        assert onnx_features.shape == hf_features.shape, (
            f"Shape mismatch: ONNX {onnx_features.shape} vs HF {hf_features.shape}"
        )

        # Cosine similarity — must be nearly identical
        dot = np.sum(onnx_features * hf_features)
        norm_a = np.sqrt(np.sum(onnx_features**2))
        norm_b = np.sqrt(np.sum(hf_features**2))
        cos_sim = dot / (norm_a * norm_b + 1e-12)
        max_diff = np.max(np.abs(onnx_features - hf_features))

        print(f"\n[vision features] cos={cos_sim:.6f} max_diff={max_diff:.6f}")

        # cos > 0.999 is tight; before fix it was 0.247
        assert cos_sim > 0.999, (
            f"Vision features diverged: cos={cos_sim:.6f} "
            f"(expected > 0.999). Check rotary dim and "
            f"fullatt_block_indexes config extraction."
        )
        assert max_diff < 0.01, f"Vision features max_diff={max_diff:.6f} (expected < 0.01)"

    def test_package_save_load(self, model_id: str, tmp_path):
        """Verify ModelPackage.save() creates correct directory structure."""
        pkg = _build_qwen25vl_3model(model_id)
        import os

        pkg.save(str(tmp_path))

        # 3-model package saves each component in its own subdirectory
        assert os.path.isfile(tmp_path / "model" / "model.onnx")
        assert os.path.isfile(tmp_path / "vision_encoder" / "model.onnx")
        assert os.path.isfile(tmp_path / "embedding" / "model.onnx")


_VL3_QWEN3_MODELS = [
    pytest.param(
        "Qwen/Qwen3-VL-2B-Instruct",
        id="qwen3-vl-2b-3model",
    ),
]


def _build_qwen3vl_3model(model_id: str):
    """Build Qwen3-VL 3-model package with real weights."""
    pkg = build(model_id, dtype="f32", load_weights=True)
    return pkg


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.parametrize("model_id", _VL3_QWEN3_MODELS)
class TestQwen3VL3Model:
    """Integration tests for Qwen3-VL 3-model split."""

    def test_all_weights_assigned(self, model_id: str):
        """Verify every ONNX initializer has weights (no missing weights)."""
        pkg = _build_qwen3vl_3model(model_id)

        assert "decoder" in pkg
        assert "vision_encoder" in pkg
        assert "embedding" in pkg

        for name, model in pkg.items():
            for init_name, init in model.graph.initializers.items():
                if init_name.startswith("const_"):
                    continue
                assert init.const_value is not None, (
                    f"[{name}] Initializer '{init_name}' has no weights"
                )

    def test_decoder_prefill_logits_match(self, model_id: str):
        """Decoder + embedding produce logits matching HF text-only forward."""
        import numpy as np

        from mobius._testing.torch_reference import (
            load_torch_multimodal_model,
        )

        pkg = _build_qwen3vl_3model(model_id)
        config = _get_config(model_id)

        torch_model, _, _ = load_torch_multimodal_model(model_id)
        torch_model.eval()

        # Run HF text-only forward
        input_ids = torch.randint(0, 100, (1, 8), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        # MRoPE: position_ids (3, batch, seq)
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len).unsqueeze(0)
        position_ids = pos.unsqueeze(0).expand(3, -1, -1)

        with torch.no_grad():
            hf_out = torch_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        hf_logits = hf_out.logits.numpy()

        # Run ONNX: first embedding, then decoder
        embed_sess = _make_session(pkg["embedding"])
        num_deepstack = len(config.deepstack_visual_indexes or [])
        image_features = np.zeros(
            (0, (num_deepstack + 1) * config.hidden_size), dtype=np.float32
        )
        embed_out = embed_sess.run(
            {
                "input_ids": input_ids.numpy(),
                "image_features": image_features,
            }
        )
        inputs_embeds = embed_out["inputs_embeds"]
        per_layer_inputs = embed_out["per_layer_inputs"]

        decoder_sess = _make_session(pkg["decoder"])
        past_kv = {}
        for i in range(config.num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
            )

        decoder_out = decoder_sess.run(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask.numpy(),
                "position_ids": position_ids.numpy(),
                "per_layer_inputs": per_layer_inputs,
                **past_kv,
            }
        )
        onnx_logits = decoder_out["logits"]

        np.testing.assert_allclose(
            onnx_logits,
            hf_logits,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_3model_vision_pipeline(self, model_id: str):
        """Run full 3-model pipeline (vision→embedding→decoder) with image.

        Processes a real image through all 3 ONNX models and compares
        the decoder logits against the HuggingFace single-model forward.
        """
        pkg = _build_qwen3vl_3model(model_id)
        config = _get_config(model_id)

        torch_model, _, _ = load_torch_multimodal_model(model_id)
        processor = transformers.AutoProcessor.from_pretrained(model_id)

        image = Image.open("testdata/pipeline-cat-chonk.jpeg")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        hf_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            hf_out = torch_model(**hf_inputs, use_cache=False)
        hf_logits = hf_out.logits.cpu().numpy()

        # Step 1: Vision model
        pixel_values = hf_inputs["pixel_values"].numpy().astype(np.float32)
        grid_thw = hf_inputs["image_grid_thw"].numpy().astype(np.int64)

        vision_session = _make_session(pkg["vision_encoder"])
        vision_out = vision_session.run(
            {
                "pixel_values": pixel_values,
                "image_grid_thw": grid_thw,
            }
        )
        vision_session.close()
        image_features = vision_out["image_features"]

        # Step 2: Embedding model
        input_ids = hf_inputs["input_ids"].numpy().astype(np.int64)
        embedding_session = _make_session(pkg["embedding"])
        embed_out = embedding_session.run(
            {
                "input_ids": input_ids,
                "image_features": image_features,
            }
        )
        embedding_session.close()
        inputs_embeds = embed_out["inputs_embeds"]
        per_layer_inputs = embed_out["per_layer_inputs"]

        # Step 3: Decoder with MRoPE position_ids from HF
        with torch.no_grad():
            embed = torch_model.model.language_model.get_input_embeddings()
            hf_embeds = embed(hf_inputs["input_ids"])
            position_ids_3d = torch_model.model.compute_3d_position_ids(
                input_ids=hf_inputs["input_ids"],
                image_grid_thw=hf_inputs["image_grid_thw"],
                video_grid_thw=None,
                mm_token_type_ids=hf_inputs["mm_token_type_ids"],
                attention_mask=hf_inputs["attention_mask"],
                past_key_values=None,
                inputs_embeds=hf_embeds,
            )
        position_ids = position_ids_3d.numpy().astype(np.int64)
        attention_mask = hf_inputs["attention_mask"].numpy().astype(np.int64)

        decoder_sess = _make_session(pkg["decoder"])
        decoder_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "per_layer_inputs": per_layer_inputs,
        }
        for i in range(config.num_hidden_layers):
            kv_shape = (1, config.num_key_value_heads, 0, config.head_dim)
            decoder_feeds[f"past_key_values.{i}.key"] = np.zeros(kv_shape, dtype=np.float32)
            decoder_feeds[f"past_key_values.{i}.value"] = np.zeros(kv_shape, dtype=np.float32)

        decoder_out = decoder_sess.run(decoder_feeds)
        decoder_sess.close()

        assert_logits_close(
            decoder_out["logits"],
            hf_logits,
            rtol=2e-2,
            atol=2e-1,
        )


_VL3_MISTRAL3_MODELS = [
    pytest.param(
        "mistralai/Ministral-3-3B-Instruct-2512",
        id="mistral3-3b-3model",
        marks=pytest.mark.skipif(
            not _model_accessible("mistralai/Ministral-3-3B-Instruct-2512"),
            reason="Model is gated — requires HuggingFace token with access",
        ),
    ),
]


def _build_mistral3_3model(model_id: str):
    """Build Mistral3 (Pixtral) 3-model package with real weights."""
    pkg = build(model_id, dtype="f32", load_weights=True)
    return pkg


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.parametrize("model_id", _VL3_MISTRAL3_MODELS)
@pytest.mark.skip(
    reason="Mistral3 requires finegrained-fp8 kernel not available in standard ORT"
)
class TestMistral3VL3Model:
    """Integration tests for Mistral3 (Pixtral) 3-model split.

    Mistral3 uses a LLaVA-style architecture:
    - vision: PixtralVisionTower + Mistral3MultiModalProjector
    - embedding: token lookup + image feature fusion
    - decoder: standard CausalLM with 1D RoPE (not MRoPE)
    """

    def test_all_weights_assigned(self, model_id: str):
        """Verify every ONNX initializer has weights."""
        pkg = _build_mistral3_3model(model_id)

        assert "decoder" in pkg
        assert "vision_encoder" in pkg
        assert "embedding" in pkg

        for name, model in pkg.items():
            for init_name, init in model.graph.initializers.items():
                if init_name.startswith("const_"):
                    continue
                assert init.const_value is not None, (
                    f"[{name}] Initializer '{init_name}' has no weights"
                )

    def test_decoder_prefill_logits_match(self, model_id: str):
        """Decoder + embedding produce logits matching HF text-only forward."""
        pkg = _build_mistral3_3model(model_id)
        config = _get_config(model_id)

        torch_model, _, _ = load_torch_multimodal_model(model_id)
        torch_model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        prompt = "The capital of France is"
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        seq_len = input_ids.shape[1]
        # Mistral3 uses standard 1D position_ids (not MRoPE)
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        # HF reference (text-only forward, no image)
        with torch.no_grad():
            hf_out = torch_model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
                position_ids=torch.from_numpy(position_ids),
            )
        hf_logits = hf_out.logits.numpy()

        # ONNX: embedding (no image features for text-only)
        embed_sess = _make_session(pkg["embedding"])
        image_features = np.zeros((1, config.hidden_size), dtype=np.float32)
        embed_out = embed_sess.run(
            {
                "input_ids": input_ids,
                "image_features": image_features,
            }
        )
        embed_sess.close()
        inputs_embeds = embed_out["inputs_embeds"]

        # ONNX: decoder
        decoder_sess = _make_session(pkg["decoder"])
        decoder_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for i in range(config.num_hidden_layers):
            kv_shape = (1, config.num_key_value_heads, 0, config.head_dim)
            decoder_feeds[f"past_key_values.{i}.key"] = np.zeros(kv_shape, dtype=np.float32)
            decoder_feeds[f"past_key_values.{i}.value"] = np.zeros(kv_shape, dtype=np.float32)

        decoder_out = decoder_sess.run(decoder_feeds)
        decoder_sess.close()

        assert_logits_close(
            decoder_out["logits"],
            hf_logits,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_3model_vision_pipeline(self, model_id: str):
        """Run full 3-model pipeline (vision→embedding→decoder) with image.

        Processes a real image through all 3 ONNX models and compares
        the decoder logits against the HuggingFace single-model forward.
        """
        pkg = _build_mistral3_3model(model_id)
        config = _get_config(model_id)

        torch_model, _, _ = load_torch_multimodal_model(model_id)
        processor = transformers.AutoProcessor.from_pretrained(model_id)

        image = Image.open("testdata/pipeline-cat-chonk.jpeg")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        hf_inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        )

        with torch.no_grad():
            hf_out = torch_model(**hf_inputs, use_cache=False)
        hf_logits = hf_out.logits.cpu().numpy()

        # Step 1: ONNX vision model — pixel_values → image_features
        pixel_values = hf_inputs["pixel_values"].numpy().astype(np.float32)

        vision_session = _make_session(pkg["vision_encoder"])
        vision_out = vision_session.run({"pixel_values": pixel_values})
        vision_session.close()
        image_features = vision_out["image_features"]

        # Step 2: ONNX embedding model — fuse text + image features
        input_ids = hf_inputs["input_ids"].numpy().astype(np.int64)

        embedding_session = _make_session(pkg["embedding"])
        embed_out = embedding_session.run(
            {
                "input_ids": input_ids,
                "image_features": image_features,
            }
        )
        embedding_session.close()
        inputs_embeds = embed_out["inputs_embeds"]

        assert inputs_embeds.shape == (
            1,
            input_ids.shape[1],
            config.hidden_size,
        )

        # Step 3: ONNX decoder with standard 1D position_ids
        attention_mask = hf_inputs["attention_mask"].numpy().astype(np.int64)
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        decoder_session = _make_session(pkg["decoder"])
        decoder_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for i in range(config.num_hidden_layers):
            kv_shape = (1, config.num_key_value_heads, 0, config.head_dim)
            decoder_feeds[f"past_key_values.{i}.key"] = np.zeros(kv_shape, dtype=np.float32)
            decoder_feeds[f"past_key_values.{i}.value"] = np.zeros(kv_shape, dtype=np.float32)

        decoder_out = decoder_session.run(decoder_feeds)
        decoder_session.close()

        # Looser tolerance for full VL pipeline (vision + embedding + decoder)
        assert_logits_close(
            decoder_out["logits"],
            hf_logits,
            rtol=2e-2,
            atol=2e-1,
        )

    def test_vision_features_parity(self, model_id: str):
        """Vision model output features match HF PyTorch reference.

        Catches regressions in:
        - Attention/RotaryEmbedding op attribute types (the swapped INT/FLOAT bug)
        - Weight dequantization correctness
        - 2D RoPE positional encoding
        - PatchMerger spatial reshaping
        """
        import math

        pkg = _build_mistral3_3model(model_id)
        hf_config = transformers.AutoConfig.from_pretrained(model_id)

        torch_model, _, _ = load_torch_multimodal_model(model_id)
        torch_model.eval()

        # Use the standard test image
        image = Image.open("testdata/pipeline-cat-chonk.jpeg").convert("RGB")
        w, h = image.size

        # HF Pixtral resize: scale longest side to max_image_size, ceil to patch_size
        patch_size = hf_config.vision_config.image_size // (
            hf_config.vision_config.image_size // hf_config.vision_config.patch_size
        )
        max_image_size = hf_config.vision_config.image_size
        merge_size = getattr(hf_config.vision_config, "spatial_merge_size", 2)
        effective_patch = patch_size * merge_size

        scale = max_image_size / max(h, w)
        new_h = math.ceil(h * scale / patch_size) * patch_size
        new_w = math.ceil(w * scale / patch_size) * patch_size
        if new_h % effective_patch != 0:
            new_h = math.ceil(new_h / effective_patch) * effective_patch
        if new_w % effective_patch != 0:
            new_w = math.ceil(new_w / effective_patch) * effective_patch

        resized = image.resize((new_w, new_h), Image.BICUBIC)
        arr = np.array(resized, dtype=np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        arr = (arr - mean) / std
        pixel_values = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

        # HF reference: vision_tower + multi_modal_projector
        pv_torch = torch.from_numpy(pixel_values).to(torch_model.dtype)
        with torch.no_grad():
            raw = torch_model.model.vision_tower(pv_torch).last_hidden_state.squeeze(0)
            hf_features = (
                torch_model.model.multi_modal_projector(raw, torch.tensor([[new_h, new_w]]))
                .float()
                .numpy()
            )

        # ONNX vision model
        vision_session = _make_session(pkg["vision_encoder"])
        onnx_out = vision_session.run({"pixel_values": pixel_values.astype(np.float32)})
        vision_session.close()
        onnx_features = onnx_out["image_features"].astype(np.float32)

        # Shape check
        assert hf_features.shape == onnx_features.shape, (
            f"Shape mismatch: HF={hf_features.shape}, ONNX={onnx_features.shape}"
        )

        # Cosine similarity (must be very high — catches attribute type bugs)
        cosine_sim = np.dot(hf_features.flatten(), onnx_features.flatten()) / (
            np.linalg.norm(hf_features) * np.linalg.norm(onnx_features)
        )
        assert cosine_sim > 0.99, (
            f"Vision cosine similarity {cosine_sim:.6f} < 0.99 — "
            f"ONNX vision model produces different features than HF. "
            f"HF norm={np.linalg.norm(hf_features):.2f}, "
            f"ONNX norm={np.linalg.norm(onnx_features):.2f}"
        )

        # Norm ratio (catches scale factor bugs like FP8 dequant issues)
        norm_ratio = np.linalg.norm(onnx_features) / np.linalg.norm(hf_features)
        assert 0.9 < norm_ratio < 1.1, (
            f"Vision norm ratio {norm_ratio:.4f} outside [0.9, 1.1] — "
            f"scale factor mismatch between ONNX and HF"
        )

        # Random independence check (catches attribute zero bugs)
        rng = np.random.RandomState(42)
        r1 = rng.randn(1, 3, new_h, new_w).astype(np.float32)
        r2 = rng.randn(1, 3, new_h, new_w).astype(np.float32)
        vision_session = _make_session(pkg["vision_encoder"])
        o1 = vision_session.run({"pixel_values": r1})["image_features"].flatten()
        o2 = vision_session.run({"pixel_values": r2})["image_features"].flatten()
        vision_session.close()
        random_cosine = np.dot(o1, o2) / (np.linalg.norm(o1) * np.linalg.norm(o2))
        assert random_cosine < 0.9, (
            f"Random input cosine similarity {random_cosine:.4f} > 0.9 — "
            f"vision model is not differentiating inputs (possible broken attention)"
        )
