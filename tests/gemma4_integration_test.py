# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests for heavyweight Gemma4 text and multimodal parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import transformers
from _integration_support import (
    _make_session,
    _model_accessible,
)
from PIL import Image

from mobius._testing.comparison import (
    assert_logits_close,
)
from mobius._testing.ort_inference import OnnxModelSession


def _make_gemma4_prefill_feeds(
    config,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build Gemma4CausalLMModel prefill feeds with dual head_dim and KV sharing.

    Gemma4 has per-layer head_dim (local vs global) and KV-shared layers that
    share K,V from source layers and have no independent KV cache entries.
    Only the first ``num_hidden_layers - num_kv_shared_layers`` layers get cache
    inputs; their head_dim depends on the layer type (sliding vs full attention).
    """
    local_head_dim = config.head_dim
    global_head_dim = getattr(config, "global_head_dim", None) or config.head_dim
    num_kv_shared = getattr(config, "num_kv_shared_layers", 0) or 0
    num_kv_layers = config.num_hidden_layers - num_kv_shared
    layer_types = getattr(config, "layer_types", None) or (
        ["sliding_attention"] * config.num_hidden_layers
    )
    feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for i in range(num_kv_layers):
        lt = layer_types[i] if i < len(layer_types) else "sliding_attention"
        hd = global_head_dim if lt == "full_attention" else local_head_dim
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, hd), dtype=np.float32
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, hd), dtype=np.float32
        )
    return feeds


@pytest.mark.integration
@pytest.mark.integration_slow
def test_gemma4_e2b_text_prefill():
    """Gemma 4 E2B text-only prefill: ONNX logits match HuggingFace.

    Builds Gemma4CausalLMModel (text backbone only) from the
    ``google/gemma-4-E2B-it`` checkpoint, runs a single prefill forward
    pass, and compares logits against the HuggingFace ``Gemma4ForCausalLM``
    text backbone running the same input.

    Tolerances: atol=1e-3, rtol=1e-3 (float32).
    """
    import dataclasses

    import onnx_ir as ir
    from transformers import Gemma4ForConditionalGeneration

    from mobius import build_from_module
    from mobius._configs import Gemma4Config
    from mobius.integrations._weight_loading import apply_weights
    from mobius.models.gemma4 import Gemma4CausalLMModel

    model_id = "google/gemma-4-E2B-it"

    if not _model_accessible(model_id):
        pytest.skip(f"{model_id} not accessible (requires HuggingFace authentication)")

    # Load HF multimodal config — text backbone is in hf_config.text_config
    hf_config = transformers.AutoConfig.from_pretrained(model_id)

    # Load full Gemma4ForConditionalGeneration in float32.
    # The model hierarchy is: hf_full.model.language_model (backbone) + hf_full.lm_head.
    # We run the full model with text-only inputs (no pixel_values/audio) for the
    # reference logits, and pass the full state_dict to preprocess_weights() which
    # strips the 'language_model.' substring from keys like 'model.language_model.*'.
    hf_full = Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # Build Gemma4Config from text_config sub-config, float32
    text_cfg = hf_config.text_config
    gemma4_config = Gemma4Config.from_transformers(text_cfg, parent_config=hf_config)
    gemma4_config = dataclasses.replace(gemma4_config, dtype=ir.DataType.FLOAT)

    # Build ONNX Gemma4CausalLMModel (text-only)
    onnx_module = Gemma4CausalLMModel(gemma4_config)
    pkg = build_from_module(onnx_module, gemma4_config, task="gemma4-text-generation")
    assert "model" in pkg

    # Transfer HF weights → ONNX.
    # preprocess_weights replaces 'model.language_model.' → 'model.' by
    # stripping the 'language_model.' substring wherever it appears.
    preprocessed = onnx_module.preprocess_weights(dict(hf_full.state_dict()))
    apply_weights(pkg["model"], preprocessed)

    # Tokenize a short prompt
    prompt = "Hello, world!"
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int64)
    attention_mask = tokens["attention_mask"].astype(np.int64)
    seq_len = input_ids.shape[1]
    position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

    # HF reference: text-only forward (no pixel_values / audio inputs).
    # Gemma4ForConditionalGeneration routes to language_model + lm_head when
    # no multimodal inputs are provided.
    with torch.no_grad():
        hf_out = hf_full(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
        )
    hf_logits = hf_out.logits.numpy()  # [1, seq_len, vocab_size]

    # ONNX inference
    session = _make_session(pkg["model"])
    feeds = _make_gemma4_prefill_feeds(gemma4_config, input_ids, attention_mask, position_ids)
    onnx_outputs = session.run(feeds)
    session.close()
    onnx_logits = onnx_outputs["logits"]  # [1, seq_len, vocab_size]

    max_diff = float(np.max(np.abs(onnx_logits - hf_logits)))
    mean_diff = float(np.mean(np.abs(onnx_logits - hf_logits)))
    print(
        f"\nGemma4 E2B prefill parity — "
        f"max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}"
    )

    assert_logits_close(onnx_logits, hf_logits, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
@pytest.mark.integration_slow
def test_gemma4_e2b_text_prefill_bf16():
    """Gemma 4 E2B text-only prefill in bfloat16: ONNX logits match HuggingFace.

    Same as ``test_gemma4_e2b_text_prefill`` but builds the ONNX model in
    bfloat16 and loads the HuggingFace reference in bfloat16.  bfloat16 has an
    ~8-bit mantissa, so element-wise logit agreement is not a meaningful gate:
    HuggingFace's own bf16-vs-f32 logits already differ by ~0.45 max-abs here,
    and different op/kernel ordering pushes mobius bf16 to a similar ~0.8
    max-abs noise floor.  The meaningful parity gate (matching the
    gemma-4-12B unified test) is last-token cosine similarity and argmax
    agreement, which hold exactly.
    """
    import dataclasses

    import ml_dtypes
    import onnx_ir as ir
    from transformers import Gemma4ForConditionalGeneration

    from mobius import build_from_module
    from mobius._configs import Gemma4Config
    from mobius.integrations._weight_loading import apply_weights
    from mobius.models.gemma4 import Gemma4CausalLMModel

    model_id = "google/gemma-4-E2B-it"

    if not _model_accessible(model_id):
        pytest.skip(f"{model_id} not accessible (requires HuggingFace authentication)")

    # Load HF multimodal config — text backbone is in hf_config.text_config
    hf_config = transformers.AutoConfig.from_pretrained(model_id)

    # Load full Gemma4ForConditionalGeneration in bfloat16.
    hf_full = Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # Build Gemma4Config from text_config sub-config, bfloat16
    text_cfg = hf_config.text_config
    gemma4_config = Gemma4Config.from_transformers(text_cfg, parent_config=hf_config)
    gemma4_config = dataclasses.replace(gemma4_config, dtype=ir.DataType.BFLOAT16)

    # Build ONNX Gemma4CausalLMModel (text-only, bfloat16)
    onnx_module = Gemma4CausalLMModel(gemma4_config)
    pkg = build_from_module(onnx_module, gemma4_config, task="gemma4-text-generation")
    assert "model" in pkg

    # Transfer HF weights → ONNX.
    preprocessed = onnx_module.preprocess_weights(dict(hf_full.state_dict()))
    apply_weights(pkg["model"], preprocessed)

    # Tokenize a short prompt
    prompt = "Hello, world!"
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int64)
    attention_mask = tokens["attention_mask"].astype(np.int64)
    seq_len = input_ids.shape[1]
    position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

    # HF reference: text-only forward in bfloat16; convert to float32 for numpy comparison
    with torch.no_grad():
        hf_out = hf_full(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
        )
    hf_logits = hf_out.logits.float().numpy()  # [1, seq_len, vocab_size]

    # Build bfloat16 KV cache feeds (empty — prefill has no prior context).
    # Replicates _make_gemma4_prefill_feeds but uses ml_dtypes.bfloat16 for
    # the KV tensors to match the model's declared input dtype.
    local_head_dim = gemma4_config.head_dim
    global_head_dim = getattr(gemma4_config, "global_head_dim", None) or gemma4_config.head_dim
    num_kv_shared = getattr(gemma4_config, "num_kv_shared_layers", 0) or 0
    num_kv_layers = gemma4_config.num_hidden_layers - num_kv_shared
    layer_types = getattr(gemma4_config, "layer_types", None) or (
        ["sliding_attention"] * gemma4_config.num_hidden_layers
    )
    feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for i in range(num_kv_layers):
        lt = layer_types[i] if i < len(layer_types) else "sliding_attention"
        hd = global_head_dim if lt == "full_attention" else local_head_dim
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, gemma4_config.num_key_value_heads, 0, hd), dtype=ml_dtypes.bfloat16
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, gemma4_config.num_key_value_heads, 0, hd), dtype=ml_dtypes.bfloat16
        )

    # ONNX inference
    session = _make_session(pkg["model"])
    onnx_outputs = session.run(feeds)
    session.close()
    # ONNX returns bfloat16; convert to float32 for numerical comparison
    onnx_logits = onnx_outputs["logits"].astype(np.float32)  # [1, seq_len, vocab_size]

    max_diff = float(np.max(np.abs(onnx_logits - hf_logits)))
    mean_diff = float(np.mean(np.abs(onnx_logits - hf_logits)))
    last_cos = float(
        np.dot(onnx_logits[0, -1], hf_logits[0, -1])
        / (np.linalg.norm(onnx_logits[0, -1]) * np.linalg.norm(hf_logits[0, -1]) + 1e-9)
    )
    argmax_match = bool((onnx_logits.argmax(-1) == hf_logits.argmax(-1)).all())
    print(
        f"\nGemma4 E2B bf16 prefill parity — "
        f"max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}, "
        f"last_token_cos_sim={last_cos:.6f}, argmax_match={argmax_match}"
    )

    # bf16 noise floor makes a tight element-wise atol meaningless (HF bf16 vs
    # f32 is already ~0.45 max-abs, and op/kernel ordering pushes mobius to
    # ~0.8); gate primarily on cosine + argmax.  Still keep a loose finite
    # max/mean-abs ceiling so a gross numerical regression (NaN-free but wildly
    # off) cannot slip through with a coincidentally high cosine.
    assert not np.isnan(onnx_logits).any()
    assert max_diff < 5.0, f"bf16 max-abs diff {max_diff:.4f} >= 5.0 (gross divergence)"
    assert mean_diff < 0.5, f"bf16 mean-abs diff {mean_diff:.4f} >= 0.5 (gross divergence)"
    assert last_cos > 0.999, f"last-token cosine {last_cos:.6f} <= 0.999"
    assert argmax_match, "per-position argmax mismatch vs HuggingFace bf16"


@pytest.mark.integration
@pytest.mark.integration_slow
def test_gemma4_unified_12b_text_prefill():
    """gemma-4-12B (``gemma4_unified``) text backbone: ONNX logits match HF.

    Builds Gemma4CausalLMModel from the ``google/gemma-4-12B`` unified text
    config and compares a single prefill forward pass against the HuggingFace
    ``Gemma4UnifiedForConditionalGeneration`` text path.  This exercises the
    real 48-layer 12B architecture: dual head_dim (local 256 / global 512),
    ``attention_k_eq_v`` with a single global KV head, dual RoPE, and
    final-logit softcapping.

    Loads the 24 GB checkpoint in float32 (~48 GB host RAM).  Runs on the
    device from ``MOBIUS_TEST_DEVICE`` (set ``cuda`` for GPU).  Tolerances
    atol=2e-2 / rtol=5e-2 account for the deep (48-layer) network and CUDA
    floating-point accumulation.
    """
    import dataclasses

    import onnx_ir as ir
    from transformers import AutoModelForImageTextToText

    from mobius import build_from_module
    from mobius._configs import Gemma4Config
    from mobius.integrations._weight_loading import apply_weights
    from mobius.models.gemma4 import Gemma4CausalLMModel

    model_id = "google/gemma-4-12B"

    if not _model_accessible(model_id):
        pytest.skip(f"{model_id} not accessible (requires HuggingFace authentication)")

    hf_config = transformers.AutoConfig.from_pretrained(model_id)
    hf_full = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # Build Gemma4Config from the unified text sub-config (parent supplies
    # boa/image/audio token ids and use_bidirectional_attention="vision").
    text_cfg = hf_config.text_config
    gemma4_config = Gemma4Config.from_transformers(text_cfg, parent_config=hf_config)
    gemma4_config = dataclasses.replace(gemma4_config, dtype=ir.DataType.FLOAT)

    onnx_module = Gemma4CausalLMModel(gemma4_config)
    pkg = build_from_module(onnx_module, gemma4_config, task="gemma4-text-generation")
    assert "model" in pkg

    preprocessed = onnx_module.preprocess_weights(dict(hf_full.state_dict()))
    apply_weights(pkg["model"], preprocessed)

    prompt = "Hello, world!"
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int64)
    attention_mask = tokens["attention_mask"].astype(np.int64)
    seq_len = input_ids.shape[1]
    position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

    with torch.no_grad():
        hf_out = hf_full(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
        )
    hf_logits = hf_out.logits.detach().cpu().numpy()

    session = _make_session(pkg["model"])
    feeds = _make_gemma4_prefill_feeds(gemma4_config, input_ids, attention_mask, position_ids)
    # gemma4_unified full-attention layers use a single global KV head; the
    # generic feed helper assumes num_key_value_heads, so rebuild KV feeds with
    # the per-layer-type KV head count.
    lt = gemma4_config.layer_types
    for i in range(
        gemma4_config.num_hidden_layers - (gemma4_config.num_kv_shared_layers or 0)
    ):
        is_full = lt[i] == "full_attention"
        hd = gemma4_config.global_head_dim if is_full else gemma4_config.head_dim
        kvh = (
            gemma4_config.num_global_key_value_heads
            if (is_full and gemma4_config.num_global_key_value_heads)
            else gemma4_config.num_key_value_heads
        )
        feeds[f"past_key_values.{i}.key"] = np.zeros((1, kvh, 0, hd), dtype=np.float32)
        feeds[f"past_key_values.{i}.value"] = np.zeros((1, kvh, 0, hd), dtype=np.float32)
    onnx_outputs = session.run(feeds)
    session.close()
    onnx_logits = onnx_outputs["logits"]

    max_diff = float(np.max(np.abs(onnx_logits - hf_logits)))
    mean_diff = float(np.mean(np.abs(onnx_logits - hf_logits)))
    print(
        f"\nGemma4 unified 12B text prefill parity — "
        f"max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}"
    )
    assert not np.isnan(onnx_logits).any()
    assert_logits_close(onnx_logits, hf_logits, rtol=5e-2, atol=2e-2)


@pytest.mark.integration
@pytest.mark.integration_slow
def test_gemma4_unified_12b_multimodal_prefill():
    """gemma-4-12B (``gemma4_unified``) full multimodal prefill parity vs HF.

    Exercises the complete encoder-free multimodal pipeline end to end and
    compares against ``Gemma4UnifiedForConditionalGeneration``:

    1. ``vision_encoder``: raw image patches → 3840-d image features
       (compared against HF ``model.get_image_features`` ``pooler_output``).
    2. ``embedding``: scatters the image features into the scaled word
       embeddings to produce ``inputs_embeds``.
    3. ``decoder``: 48-layer gemma4 text decoder with vision-block
       bidirectional attention (``use_bidirectional_attention="vision"``).
       It derives the vision-block ids internally from ``input_ids``.

    The bidirectional reference REQUIRES passing ``mm_token_type_ids`` to HF —
    without it HF falls back to a purely causal mask.  Loads the 24 GB
    checkpoint in float32 (~48 GB host RAM).  Runs on ``MOBIUS_TEST_DEVICE``
    (set ``cuda`` for GPU).
    """
    import dataclasses

    import onnx_ir as ir
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from mobius._configs import Gemma4Config
    from mobius.integrations._weight_loading import _download_weights, apply_weights
    from mobius.models.gemma4 import Gemma4UnifiedModel
    from mobius.tasks import TASK_REGISTRY

    model_id = "google/gemma-4-12B"

    if not _model_accessible(model_id):
        pytest.skip(f"{model_id} not accessible (requires HuggingFace authentication)")

    hf_config = transformers.AutoConfig.from_pretrained(model_id)
    hf_full = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # Build a single-image multimodal input. The unified processor expands the
    # `<|image|>` placeholder into the right number of soft tokens and emits
    # pixel_values [B, N, P^2*3], image_position_ids [B, N, 2] and
    # mm_token_type_ids [B, S] (1 = image span).
    image = Image.new("RGB", (112, 112), (100, 150, 200))
    proc_inputs = processor(
        text=[f"{processor.image_token} Describe the image."],
        images=[image],
        return_tensors="pt",
    )
    input_ids = proc_inputs["input_ids"].numpy().astype(np.int64)
    seq_len = input_ids.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
    pixel_values = proc_inputs["pixel_values"]
    image_position_ids = proc_inputs["image_position_ids"]
    mm_token_type_ids = proc_inputs["mm_token_type_ids"]

    # HF reference: full multimodal forward (mm_token_type_ids enables the
    # vision-block bidirectional mask) and the isolated image features.
    with torch.no_grad():
        hf_out = hf_full(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            mm_token_type_ids=mm_token_type_ids,
        )
        hf_image_features = (
            hf_full.model.get_image_features(
                pixel_values, image_position_ids, return_dict=True
            )
            .pooler_output.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    hf_logits = hf_out.logits.detach().cpu().numpy()

    # Build the 4-model gemma4_unified package and load real weights.
    gemma4_config = Gemma4Config.from_transformers(
        hf_config.text_config, parent_config=hf_config
    )
    gemma4_config = dataclasses.replace(gemma4_config, dtype=ir.DataType.FLOAT)
    module = Gemma4UnifiedModel(gemma4_config)
    pkg = TASK_REGISTRY["gemma4-unified"]().build(module, gemma4_config)
    # Load the raw safetensors checkpoint (production weight path). The unified
    # checkpoint stores `vision_embedder.*` / `embed_vision.embedding_projection.*`
    # names, which differ from the runtime module names in hf_full.state_dict();
    # preprocess_weights maps the checkpoint names.
    preprocessed = module.preprocess_weights(_download_weights(model_id))
    for name in ("decoder", "vision_encoder", "audio_encoder", "embedding"):
        if name in pkg:
            apply_weights(pkg[name], preprocessed)

    # Stage 1: vision embedder.
    vision_session = _make_session(pkg["vision_encoder"])
    image_features = vision_session.run(
        {
            "pixel_values": pixel_values.numpy().astype(np.float32),
            "pixel_position_ids": image_position_ids.numpy().astype(np.int64),
        }
    )["image_features"]
    vision_session.close()
    assert image_features.shape == hf_image_features.shape
    vis_cos = float(
        np.mean(
            np.sum(image_features * hf_image_features, axis=1)
            / (
                np.linalg.norm(image_features, axis=1)
                * np.linalg.norm(hf_image_features, axis=1)
                + 1e-9
            )
        )
    )
    print(f"\nGemma4 unified 12B vision cos_sim={vis_cos:.6f}")
    assert vis_cos > 0.999

    # Stage 2: embedding fusion → inputs_embeds.
    embedding_session = _make_session(pkg["embedding"])
    emb_out = embedding_session.run(
        {
            "input_ids": input_ids,
            "image_features": image_features,
            "audio_features": np.zeros((0, gemma4_config.hidden_size), dtype=np.float32),
        }
    )
    embedding_session.close()
    inputs_embeds = emb_out["inputs_embeds"]

    # Stage 3: decoder with vision-block bidirectional attention. The decoder
    # derives ``block_sequence_ids`` internally from ``input_ids`` (forwarded
    # alongside ``inputs_embeds``).
    decoder_session = _make_session(pkg["decoder"])
    feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "input_ids": input_ids,
    }
    layer_types = gemma4_config.layer_types
    for i in range(gemma4_config.num_hidden_layers):
        is_full = layer_types[i] == "full_attention"
        head_dim = gemma4_config.global_head_dim if is_full else gemma4_config.head_dim
        kv_heads = (
            gemma4_config.num_global_key_value_heads
            if (is_full and gemma4_config.num_global_key_value_heads)
            else gemma4_config.num_key_value_heads
        )
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, kv_heads, 0, head_dim), dtype=np.float32
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, kv_heads, 0, head_dim), dtype=np.float32
        )
    onnx_logits = decoder_session.run(feeds)["logits"]
    decoder_session.close()

    max_diff = float(np.max(np.abs(onnx_logits - hf_logits)))
    last_cos = float(
        np.dot(onnx_logits[0, -1], hf_logits[0, -1])
        / (np.linalg.norm(onnx_logits[0, -1]) * np.linalg.norm(hf_logits[0, -1]) + 1e-9)
    )
    print(
        f"Gemma4 unified 12B multimodal prefill parity — "
        f"max_abs_diff={max_diff:.4f}, last_token_cos_sim={last_cos:.6f}"
    )
    assert not np.isnan(onnx_logits).any()
    assert last_cos > 0.999
    assert onnx_logits[0, -1].argmax() == hf_logits[0, -1].argmax()


@pytest.mark.integration
def test_gemma4_bidirectional_mask_parity():
    """Mobius's vision-block attention bias matches HuggingFace exactly.

    Larger Gemma 4 models use ``use_bidirectional_attention="vision"``: a
    contiguous run of image/audio placeholder tokens attends bidirectionally
    within its block (on BOTH full and sliding layers), while text stays
    causal.  This test compares mobius's ``create_attention_bias`` output
    (with ``block_sequence_ids`` from ``_compute_block_sequence_ids``) against
    HuggingFace's real ``create_causal_mask`` / ``create_sliding_window_causal_mask``
    for the actual ``gemma-4-26b-a4b-it`` config.  It needs only the config
    (no weights), so it is cheap and deterministic.
    """
    import onnx_ir as ir
    import torch
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    from mobius._testing import create_test_builder, create_test_input
    from mobius.components._common import create_attention_bias
    from mobius.models.gemma4 import _compute_block_sequence_ids
    from mobius.tasks._base import _make_graph, _make_model

    model_id = "google/gemma-4-26b-a4b-it"
    if not _model_accessible(model_id):
        pytest.skip(f"{model_id} not accessible (requires HuggingFace authentication)")

    hf_config = transformers.AutoConfig.from_pretrained(model_id)
    text_cfg = hf_config.text_config
    text_cfg._attn_implementation = "eager"  # always returns a dense float mask
    assert text_cfg.use_bidirectional_attention == "vision"
    image_token_id = hf_config.image_token_id

    # Synthetic layout: text, image block, text, image block, text.
    seq_len = 12
    input_ids = np.full((1, seq_len), 5, dtype=np.int64)
    input_ids[0, 2:7] = image_token_id
    input_ids[0, 9:11] = image_token_id
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

    # Mobius block_sequence_ids from input_ids.
    graph, builder = _make_graph()
    op = builder.op
    iid = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[1, seq_len])
    bsid = _compute_block_sequence_ids(op, iid, image_token_id=image_token_id)
    builder.add_output(bsid, "bsid")
    block_ids = OnnxModelSession(_make_model(graph), device="cpu").run(
        {"input_ids": input_ids}
    )["bsid"]
    # Two contiguous image runs separated by text -> groups 0 and 1.
    np.testing.assert_array_equal(
        block_ids[0],
        np.array([-1, -1, 0, 0, 0, 0, 0, -1, -1, 1, 1, -1], dtype=np.int64),
    )

    def mobius_attended(sliding_window):
        b, bop, g = create_test_builder()
        ii = create_test_input(b, "input_ids", [1, seq_len], dtype=ir.DataType.INT64)
        am = create_test_input(b, "attention_mask", [1, seq_len], dtype=ir.DataType.INT64)
        bk = create_test_input(b, "block_sequence_ids", [1, seq_len], dtype=ir.DataType.INT64)
        bias = create_attention_bias(
            bop,
            ii,
            am,
            sliding_window=sliding_window,
            dtype=ir.DataType.FLOAT,
            block_sequence_ids=bk,
        )
        bias.name = "bias"
        g.outputs.append(bias)
        out = OnnxModelSession(ir.Model(g, ir_version=10), device="cpu").run(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "block_sequence_ids": block_ids,
            }
        )["bias"]
        return out[0, 0] > -1.0  # True where the position is attended

    inputs_embeds = torch.zeros(1, seq_len, 8)
    blk = torch.from_numpy(block_ids)
    hf_full = create_causal_mask(
        text_cfg,
        inputs_embeds,
        torch.from_numpy(attention_mask),
        None,
        torch.from_numpy(position_ids),
        block_sequence_ids=blk,
    )
    hf_sliding = create_sliding_window_causal_mask(
        text_cfg,
        inputs_embeds,
        torch.from_numpy(attention_mask),
        None,
        torch.from_numpy(position_ids),
        block_sequence_ids=blk,
    )

    def hf_attended(mask):
        m = mask[0, 0].numpy()
        return m if m.dtype == bool else (m > -1e30)

    # Full-attention layers: causal OR same-block, AND padding.
    np.testing.assert_array_equal(mobius_attended(None), hf_attended(hf_full))
    # Sliding layers: (causal AND window) OR same-block, AND padding.
    np.testing.assert_array_equal(
        mobius_attended(text_cfg.sliding_window), hf_attended(hf_sliding)
    )
