# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests for the Qwen3.5-VL three-model and hybrid-state pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import transformers
from _integration_support import (
    _make_session,
)
from PIL import Image
from transformers.cache_utils import DynamicCache

from mobius import models
from mobius._configs import ArchitectureConfig, VisionConfig
from mobius._constants import OPSET_VERSION
from mobius._testing.comparison import (
    assert_logits_close,
)


@pytest.mark.integration
@pytest.mark.integration_fast
def test_qwen35_vl_3model_builds_and_runs():
    """Qwen3.5-VL 3-model split: build + ORT execution with random weights.

    Verifies:
    - Package contains 3 models (decoder, vision, embedding)
    - Decoder has hybrid cache (conv_state/recurrent_state for DeltaNet,
      key/value for full attention layers)
    - All 3 models run through ORT with valid shapes
    - DeltaNet layers only support seq_len=1 (single-token decode)
    """
    import onnx_ir as ir

    from mobius._registry import registry
    from mobius.tasks import get_task

    # Tiny config matching Qwen3.5-VL structure: 4 layers with hybrid cache
    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=4,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attn_qk_norm=True,
        partial_rotary_factor=0.5,
        # InterleavedMRope: decoder receives 3D position_ids (3, batch, seq).
        # Without mrope_section, initialize_rope falls back to DefaultRope, and
        # Gather(cos_cache, (3,B,S)) produces 4D cos which ORT rejects.
        # rotary_dim = head_dim * partial_rotary_factor / 2 = 4; any mrope_section
        # values work because InterleavedMRope guards with `if i < rotary_dim`.
        mrope_section=[1, 1, 1],
        mrope_interleaved=True,
        # Hybrid: 3 DeltaNet + 1 full attention (matches real 27B pattern)
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        # DeltaNet dimensions (small for testing)
        linear_num_key_heads=4,
        linear_key_head_dim=8,
        linear_num_value_heads=4,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        deepstack_visual_indexes=[0],
        # Vision config (Qwen VL uses packed-attention ViT)
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=16,
            temporal_patch_size=2,
            in_channels=3,
            out_hidden_size=64,
            spatial_merge_size=2,
            num_position_embeddings=16,
            mrope_section=[8, 12, 12],
        ),
        image_token_id=248056,
        dtype=ir.DataType.FLOAT,
    )

    model_cls = registry.get("qwen3_5_vl")
    module = model_cls(config)
    task = get_task("hybrid-qwen-vl")
    pkg = task.build(module, config)

    # Verify 3-model split
    assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

    # Verify decoder has hybrid cache outputs
    decoder_outputs = {o.name for o in pkg["decoder"].graph.outputs}
    assert "logits" in decoder_outputs
    # DeltaNet layers (0-2): conv_state + recurrent_state
    for i in range(3):
        assert f"present.{i}.conv_state" in decoder_outputs
        assert f"present.{i}.recurrent_state" in decoder_outputs
    # Full attention layer (3): key + value
    assert "present.3.key" in decoder_outputs
    assert "present.3.value" in decoder_outputs

    decoder_inputs = {inp.name for inp in pkg["decoder"].graph.inputs}
    assert "inputs_embeds" in decoder_inputs
    assert "attention_mask" in decoder_inputs
    assert "position_ids" in decoder_inputs
    assert "per_layer_inputs" in decoder_inputs

    # Verify vision model I/O
    vision_inputs = {i.name for i in pkg["vision_encoder"].graph.inputs}
    assert "pixel_values" in vision_inputs

    # Verify embedding model I/O
    embed_inputs = {i.name for i in pkg["embedding"].graph.inputs}
    assert "input_ids" in embed_inputs
    assert "image_features" in embed_inputs
    assert {o.name for o in pkg["embedding"].graph.outputs} == {
        "inputs_embeds",
        "per_layer_inputs",
    }

    # Run through ORT: fill initializers with random weights
    rng = np.random.default_rng(42)
    for model in pkg.values():
        for init in model.graph.initializers.values():
            if init.const_value is None:
                shape = [d if isinstance(d, int) else 1 for d in init.shape]
                init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

    # Run embedding model with a single token (DeltaNet = decode-only)
    embed_sess = _make_session(pkg["embedding"])
    input_ids = np.array([[1]], dtype=np.int64)
    image_features = np.zeros(
        (
            0,
            (len(config.deepstack_visual_indexes) + 1) * config.hidden_size,
        ),
        dtype=np.float32,
    )
    embed_out = embed_sess.run({"input_ids": input_ids, "image_features": image_features})
    embed_sess.close()
    assert "inputs_embeds" in embed_out
    assert embed_out["inputs_embeds"].shape == (1, 1, config.hidden_size)

    # Run decoder model (seq_len=1 since DeltaNet is decode-only)
    decoder_sess = _make_session(pkg["decoder"])
    feeds: dict[str, np.ndarray] = {
        "inputs_embeds": embed_out["inputs_embeds"],
        "per_layer_inputs": embed_out["per_layer_inputs"],
        "attention_mask": np.ones((1, 1), dtype=np.int64),
        # MRoPE: 3D position IDs (3, batch, seq)
        "position_ids": np.zeros((3, 1, 1), dtype=np.int64),
    }
    # DeltaNet cache (layers 0-2): zero-initialized
    conv_dim = (
        config.linear_key_head_dim * config.linear_num_key_heads * 2
        + config.linear_value_head_dim * config.linear_num_value_heads
    )
    for i in range(3):
        feeds[f"past_key_values.{i}.conv_state"] = np.zeros(
            (1, conv_dim, config.linear_conv_kernel_dim - 1),
            dtype=np.float32,
        )
        feeds[f"past_key_values.{i}.recurrent_state"] = np.zeros(
            (
                1,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            ),
            dtype=np.float32,
        )
    # Full attention cache (layer 3): empty KV
    feeds["past_key_values.3.key"] = np.zeros(
        (1, config.num_key_value_heads, 0, config.head_dim),
        dtype=np.float32,
    )
    feeds["past_key_values.3.value"] = np.zeros(
        (1, config.num_key_value_heads, 0, config.head_dim),
        dtype=np.float32,
    )

    decoder_out = decoder_sess.run(feeds)
    decoder_sess.close()

    assert "logits" in decoder_out
    assert decoder_out["logits"].shape == (1, 1, config.vocab_size)

    # Verify DeltaNet state outputs have correct shapes
    for i in range(3):
        conv_out = decoder_out[f"present.{i}.conv_state"]
        rec_out = decoder_out[f"present.{i}.recurrent_state"]
        assert conv_out.shape == (
            1,
            conv_dim,
            config.linear_conv_kernel_dim - 1,
        )
        assert rec_out.shape == (
            1,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        )

    print(
        f"Qwen3.5-VL 3-model split OK: "
        f"decoder({len(list(pkg['decoder'].graph.inputs))} inputs), "
        f"vision({len(list(pkg['vision_encoder'].graph.inputs))} inputs), "
        f"embedding({len(list(pkg['embedding'].graph.inputs))} inputs)"
    )


@pytest.mark.integration
@pytest.mark.integration_fast
def test_qwen35_vl_deltanet_state_carry():
    """DeltaNet state carry: two decode steps produce different states.

    Uses a tiny Qwen3.5 text decoder config with random weights to verify
    that conv_state and recurrent_state are correctly updated across
    consecutive single-token decode steps. This is the highest-risk novel
    component in the Qwen3.5 architecture.
    """
    import onnx_ir as ir

    from mobius import build_from_module

    # Tiny config: 2 DeltaNet + 1 full attention layer
    layer_types = ["linear_attention", "linear_attention", "full_attention"]
    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=3,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        attn_qk_norm=True,
        partial_rotary_factor=0.5,
        layer_types=layer_types,
        linear_num_key_heads=4,
        linear_key_head_dim=8,
        linear_num_value_heads=4,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        dtype=ir.DataType.FLOAT,
    )

    onnx_module = models.Qwen35CausalLMModel(config)
    pkg = build_from_module(
        onnx_module,
        config,
        task="hybrid-text-generation",
    )
    onnx_model = pkg["model"]

    # Fill initializers with random weights
    rng = np.random.default_rng(42)
    for init in onnx_model.graph.initializers.values():
        if init.const_value is None:
            shape = [d if isinstance(d, int) else 1 for d in init.shape]
            init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

    session = _make_session(onnx_model)

    # DeltaNet cache dimensions
    num_k_heads = config.linear_num_key_heads
    head_k_dim = config.linear_key_head_dim
    num_v_heads = config.linear_num_value_heads
    head_v_dim = config.linear_value_head_dim
    conv_kernel = config.linear_conv_kernel_dim
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim

    def make_feeds(token_id, conv_states, rec_states, kv_cache, step):
        """Build ONNX input feeds for a single decode step."""
        feeds = {
            "input_ids": np.array([[token_id]], dtype=np.int64),
            "attention_mask": np.ones((1, step + 1), dtype=np.int64),
            "position_ids": np.array([[step]], dtype=np.int64),
        }
        for i in range(3):
            ltype = layer_types[i]
            if ltype == "linear_attention":
                feeds[f"past_key_values.{i}.conv_state"] = conv_states[i]
                feeds[f"past_key_values.{i}.recurrent_state"] = rec_states[i]
            else:
                feeds[f"past_key_values.{i}.key"] = kv_cache[i][0]
                feeds[f"past_key_values.{i}.value"] = kv_cache[i][1]
        return feeds

    # Initialize empty states
    conv_states: dict[int, np.ndarray] = {}
    rec_states: dict[int, np.ndarray] = {}
    kv_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(3):
        if layer_types[i] == "linear_attention":
            conv_states[i] = np.zeros(
                (1, conv_dim, conv_kernel - 1),
                dtype=np.float32,
            )
            rec_states[i] = np.zeros(
                (1, num_v_heads, head_k_dim, head_v_dim),
                dtype=np.float32,
            )
        else:
            kv_cache[i] = (
                np.zeros(
                    (1, config.num_key_value_heads, 0, config.head_dim),
                    dtype=np.float32,
                ),
                np.zeros(
                    (1, config.num_key_value_heads, 0, config.head_dim),
                    dtype=np.float32,
                ),
            )

    # Step 1: first token
    token1 = int(rng.integers(0, config.vocab_size))
    feeds1 = make_feeds(token1, conv_states, rec_states, kv_cache, 0)
    out1 = session.run(feeds1)

    # Extract states after step 1
    conv_states_1: dict[int, np.ndarray] = {}
    rec_states_1: dict[int, np.ndarray] = {}
    kv_cache_1: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(3):
        if layer_types[i] == "linear_attention":
            conv_states_1[i] = out1[f"present.{i}.conv_state"]
            rec_states_1[i] = out1[f"present.{i}.recurrent_state"]
        else:
            kv_cache_1[i] = (
                out1[f"present.{i}.key"],
                out1[f"present.{i}.value"],
            )

    # Verify DeltaNet states changed from zeros
    for i in range(2):  # layers 0, 1 are linear_attention
        assert not np.allclose(conv_states_1[i], 0.0), (
            f"Layer {i} conv_state should be non-zero after first token"
        )

    # Step 2: second token with carried states
    token2 = int(rng.integers(0, config.vocab_size))
    feeds2 = make_feeds(
        token2,
        conv_states_1,
        rec_states_1,
        kv_cache_1,
        1,
    )
    out2 = session.run(feeds2)

    # Verify states differ between steps (state is being updated)
    for i in range(2):
        conv_s2 = out2[f"present.{i}.conv_state"]
        rec_s2 = out2[f"present.{i}.recurrent_state"]
        assert not np.array_equal(conv_states_1[i], conv_s2), (
            f"Layer {i} conv_state should differ between steps"
        )
        assert not np.array_equal(rec_states_1[i], rec_s2), (
            f"Layer {i} recurrent_state should differ between steps"
        )

    # Verify full attention layer KV cache grew
    assert kv_cache_1[2][0].shape[2] == 1  # 1 token after step 1
    kv_key_2 = out2["present.2.key"]
    assert kv_key_2.shape[2] == 2  # 2 tokens after step 2

    session.close()
    print(
        "Qwen3.5 DeltaNet state carry OK: "
        "conv_state and recurrent_state updated across 2 steps"
    )


def _make_tiny_qwen35_vl_config():
    """Create a tiny Qwen3.5-VL config for fast HF parity testing.

    Downloads the real Qwen3.5-27B config structure, then overrides all
    dimensions to be tiny. Also overrides rope_theta to float to avoid
    a pre-existing float64 rotary cache bug (int ** np.float32 → float64).
    """
    c = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-27B")
    tc = c.text_config

    # Truncate layers: 3 DeltaNet + 1 full attention
    tc.num_hidden_layers = 4
    tc.layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]

    # Shrink dimensions for fast testing
    tc.hidden_size = 64
    tc.intermediate_size = 128
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.head_dim = 16
    tc.vocab_size = 256
    tc.linear_num_value_heads = 4
    tc.linear_num_key_heads = 4
    tc.linear_key_head_dim = 8
    tc.linear_value_head_dim = 8
    # MRoPE: must fit rotary_dim = head_dim * partial_rotary_factor / 2
    tc.partial_rotary_factor = 0.5
    tc.mrope_section = [8, 12, 12]
    # Avoid float64 rotary caches: use float rope_theta and small context
    tc.max_position_embeddings = 128
    tc.rope_theta = 10000.0
    # Update nested dicts to match
    if hasattr(tc, "rope_scaling") and tc.rope_scaling is not None:
        tc.rope_scaling["partial_rotary_factor"] = 0.5
        tc.rope_scaling["mrope_section"] = [8, 12, 12]
        tc.rope_scaling["rope_theta"] = 10000.0
    if hasattr(tc, "rope_parameters") and tc.rope_parameters is not None:
        tc.rope_parameters["partial_rotary_factor"] = 0.5
        tc.rope_parameters["mrope_section"] = [8, 12, 12]
        tc.rope_parameters["rope_theta"] = 10000.0

    # Shrink vision
    vc = c.vision_config
    vc.hidden_size = 32
    vc.intermediate_size = 64
    vc.depth = 1
    vc.num_heads = 2
    vc.patch_size = 16
    vc.out_hidden_size = 64

    return c


@pytest.mark.integration
@pytest.mark.integration_fast
def test_qwen35_vl_3model_text_only_parity():
    """Qwen3.5-VL 3-model text-only forward matches HuggingFace.

    Builds decoder, vision, and embedding ONNX models from a truncated
    Qwen3.5-27B config with random weights. Runs a text-only pass
    (embedding → decoder) and compares logits against HF.
    """
    import onnx_ir as ir
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    from mobius import build_from_module
    from mobius.integrations._weight_loading import apply_weights

    hf_config = _make_tiny_qwen35_vl_config()
    tc = hf_config.text_config

    # Build ONNX 3-model package
    arch_config = ArchitectureConfig.from_transformers(
        tc,
        parent_config=hf_config,
    )
    arch_config.dtype = ir.DataType.FLOAT
    onnx_module = models.Qwen35VL3ModelCausalLMModel(arch_config)
    pkg = build_from_module(
        onnx_module,
        arch_config,
        task="hybrid-qwen-vl",
    )
    assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

    # Build HF model with random weights
    hf_model = (
        Qwen3_5ForConditionalGeneration._from_config(
            hf_config,
            dtype=torch.float32,
        )
        .float()
        .eval()
    )

    # Transfer HF weights → ONNX
    preprocessed = onnx_module.preprocess_weights(
        dict(hf_model.state_dict()),
    )
    for onnx_model in pkg.values():
        apply_weights(onnx_model, preprocessed)

    # HF text-only forward (seq_len=1 for DeltaNet compatibility)
    rng = np.random.default_rng(42)
    input_ids = rng.integers(
        0,
        arch_config.vocab_size,
        size=(1, 1),
    ).astype(np.int64)
    attention_mask = np.ones((1, 1), dtype=np.int64)
    pos_1d = np.arange(1, dtype=np.int64)[np.newaxis, :]
    # MRoPE: 3D position IDs — all equal for text-only
    position_ids_3d = np.stack(
        [pos_1d, pos_1d, pos_1d],
        axis=0,
    )  # (3, 1, 1)

    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids_3d),
        ).logits.numpy()

    # ONNX: embedding model
    embed_sess = _make_session(pkg["embedding"])
    embed_out = embed_sess.run(
        {
            "input_ids": input_ids,
            "image_features": np.zeros(
                (0, arch_config.hidden_size),
                dtype=np.float32,
            ),
        }
    )
    embed_sess.close()

    # ONNX: decoder model
    decoder_sess = _make_session(pkg["decoder"])
    feeds: dict[str, np.ndarray] = {
        "inputs_embeds": embed_out["inputs_embeds"],
        "attention_mask": attention_mask,
        "position_ids": position_ids_3d,
    }
    # Build cache feeds: batch=1, symbolic past dims=0
    for inp in pkg["decoder"].graph.inputs:
        name = inp.name
        if name in feeds:
            continue
        shape = []
        for d in inp.shape:
            if isinstance(d, int):
                shape.append(d)
            elif "past" in str(d):
                shape.append(0)
            else:
                shape.append(1)  # batch
        feeds[name] = np.zeros(shape, dtype=np.float32)

    onnx_logits = decoder_sess.run(feeds)["logits"]
    decoder_sess.close()

    assert_logits_close(onnx_logits, hf_logits, rtol=2e-2, atol=2e-2)


@pytest.mark.integration
def test_qwen35_vl_vision_features_match():
    """Qwen3.5-VL vision encoder: ONNX features match HuggingFace.

    Processes a real image (testdata/pipeline-cat-chonk.jpeg) through
    both the HF and ONNX vision encoders built from a tiny random-weight
    config.  Verifies shape parity and cosine similarity > 0.999.

    This guards against regressions in:
    - Patch embedding (Conv3d → hidden_size)
    - Positional embedding interpolation
    - Rotary position embedding for vision
    - Spatial merge (pooling patches)
    """
    import onnx_ir as ir
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    from mobius import build_from_module
    from mobius.integrations._weight_loading import apply_weights

    hf_config = _make_tiny_qwen35_vl_config()

    # Build ONNX 3-model package
    arch_config = ArchitectureConfig.from_transformers(
        hf_config.text_config,
        parent_config=hf_config,
    )
    arch_config.dtype = ir.DataType.FLOAT
    onnx_module = models.Qwen35VL3ModelCausalLMModel(arch_config)
    pkg = build_from_module(
        onnx_module,
        arch_config,
        task="hybrid-qwen-vl",
    )
    assert "vision_encoder" in pkg

    # Build HF model with random weights and transfer to ONNX
    hf_model = (
        Qwen3_5ForConditionalGeneration._from_config(
            hf_config,
            dtype=torch.float32,
        )
        .float()
        .eval()
    )
    preprocessed = onnx_module.preprocess_weights(
        dict(hf_model.state_dict()),
    )
    for onnx_model in pkg.values():
        apply_weights(onnx_model, preprocessed)

    # Process real image (resized small for speed — 256 patches)
    processor = transformers.AutoProcessor.from_pretrained(
        "Qwen/Qwen3.5-27B",
    )
    image = Image.open("testdata/pipeline-cat-chonk.jpeg").resize(
        (64, 64),
    )
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

    pixel_values = hf_inputs["pixel_values"]
    grid_thw = hf_inputs["image_grid_thw"]

    # HF vision forward
    with torch.no_grad():
        hf_visual_out = hf_model.model.visual(
            pixel_values,
            grid_thw=grid_thw,
        )
    hf_features = hf_visual_out.pooler_output.numpy()

    # ONNX vision forward
    vision_session = _make_session(pkg["vision_encoder"])
    vision_out = vision_session.run(
        {
            "pixel_values": pixel_values.numpy().astype(np.float32),
            "image_grid_thw": grid_thw.numpy().astype(np.int64),
        }
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

    print(
        f"\n[Qwen3.5-VL vision] cos={cos_sim:.6f} "
        f"max_diff={max_diff:.6f} "
        f"patches={onnx_features.shape[0]}"
    )

    assert cos_sim > 0.999, (
        f"Vision features diverged: cos={cos_sim:.6f} "
        f"(expected > 0.999). Check patch_embed, rotary, "
        f"or spatial merge."
    )
    assert max_diff < 0.01, f"Vision features max_diff={max_diff:.6f} (expected < 0.01)"


@pytest.mark.integration
@pytest.mark.integration_fast
def test_qwen35_deltanet_single_layer_parity():
    """Single GatedDeltaNet layer: ONNX matches HuggingFace.

    Builds a standalone DeltaNet graph, loads random HF weights, runs a
    single-token decode step, and verifies:
    - hidden_states output matches HF
    - recurrent_state carry matches HF
    """
    import onnx_ir as ir
    from onnxscript import GraphBuilder

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5GatedDeltaNet,
        )
    except ImportError:
        pytest.skip("Qwen3_5GatedDeltaNet not available in this transformers version")

    from mobius.components._gated_deltanet import (
        GatedDeltaNet,
    )
    from mobius.integrations._weight_loading import apply_weights

    # Tiny config for isolated DeltaNet test
    hf_config = _make_tiny_qwen35_vl_config()
    tc = hf_config.text_config
    tc.num_hidden_layers = 1
    tc.layer_types = ["linear_attention"]

    arch_config = ArchitectureConfig.from_transformers(
        tc,
        parent_config=hf_config,
    )
    arch_config.dtype = ir.DataType.FLOAT

    # DeltaNet dimensions
    num_k_heads = arch_config.linear_num_key_heads
    num_v_heads = arch_config.linear_num_value_heads
    head_k_dim = arch_config.linear_key_head_dim
    head_v_dim = arch_config.linear_value_head_dim
    conv_kernel = arch_config.linear_conv_kernel_dim or 4
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim

    # Build standalone ONNX graph for GatedDeltaNet
    onnx_dn = GatedDeltaNet(arch_config)
    batch = ir.SymbolicDim("batch")
    hidden_in = ir.Value(
        name="hidden_states",
        shape=ir.Shape([batch, 1, arch_config.hidden_size]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    conv_in = ir.Value(
        name="conv_state",
        shape=ir.Shape([batch, conv_dim, conv_kernel - 1]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    rec_in = ir.Value(
        name="recurrent_state",
        shape=ir.Shape([batch, num_v_heads, head_k_dim, head_v_dim]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )

    graph = ir.Graph(
        inputs=[hidden_in, conv_in, rec_in],
        outputs=[],
        nodes=[],
        name="deltanet_test",
        opset_imports={"": OPSET_VERSION, "com.microsoft": 1},
    )
    graph_builder = GraphBuilder(graph)
    op = graph_builder.op

    output, new_conv, new_rec = onnx_dn(
        op,
        hidden_in,
        conv_in,
        rec_in,
    )
    output.name = "output"
    new_conv.name = "new_conv_state"
    new_rec.name = "new_recurrent_state"
    graph.outputs.extend([output, new_conv, new_rec])

    for name, param in onnx_dn.named_parameters():
        param.name = name
        # Initialize with zeros so register_initializer accepts them;
        # apply_weights will overwrite with real HF values.
        shape = [d if isinstance(d, int) else 1 for d in param.shape]
        param.const_value = ir.Tensor(
            np.zeros(shape, dtype=np.float32),
            name=name,
        )
        graph.register_initializer(param)

    onnx_model = ir.Model(graph, ir_version=10)

    # Register CausalConvWithState and LinearAttention function definitions.
    # Building a bare component graph omits these; ORT needs the ONNX local
    # function definitions embedded in the model to decompose the nodes.
    from mobius.functions import (
        causal_conv_nd_with_state,
    )
    from mobius.functions import (
        linear_attention as linear_attention_fn,
    )

    conv_func = causal_conv_nd_with_state(
        kernel_size=conv_kernel,
        channels=conv_dim,
        ndim=1,
        activation="silu",
    )
    attn_func = linear_attention_fn(
        q_num_heads=num_k_heads,
        kv_num_heads=num_v_heads,
        update_rule="gated_delta",
        scale=1.0 / (head_k_dim**0.5),
    )
    onnx_model.functions[conv_func.identifier()] = conv_func
    onnx_model.functions[attn_func.identifier()] = attn_func

    # Build HF DeltaNet layer with random weights
    hf_dn = Qwen3_5GatedDeltaNet(tc, layer_idx=0)
    hf_dn = hf_dn.to(torch.float32).eval()

    # Transfer HF weights → ONNX
    apply_weights(onnx_model, dict(hf_dn.state_dict()))

    # Prepare inputs
    rng = np.random.default_rng(42)
    hidden_np = rng.standard_normal(
        (1, 1, arch_config.hidden_size),
    ).astype(np.float32)
    conv_np = rng.standard_normal(
        (1, conv_dim, conv_kernel - 1),
    ).astype(np.float32)
    rec_np = rng.standard_normal(
        (1, num_v_heads, head_k_dim, head_v_dim),
    ).astype(np.float32)

    # HF forward (single-token decode with pre-filled cache)
    cache = DynamicCache(config=tc)
    # Use the cache API to pre-fill states so that dtype/device/initialized flags
    # are all set correctly.  Direct attribute assignment bypasses lazy_initialization
    # and leaves 'dtype' unset, causing AttributeError on the first update call.
    # HF conv_state shape is (batch, conv_dim, conv_kernel_size) —
    # pad with one extra left position vs ONNX (kernel_size - 1).
    # update_conv_state also sets has_previous_state = True (decode-mode trigger).
    padded_conv = torch.from_numpy(np.pad(conv_np, ((0, 0), (0, 0), (1, 0)))).float()
    cache.update_conv_state(padded_conv, layer_idx=0)
    cache.update_recurrent_state(torch.from_numpy(rec_np).float(), layer_idx=0)

    with torch.no_grad():
        hf_output = hf_dn(
            hidden_states=torch.from_numpy(hidden_np).float(),
            cache_params=cache,
        ).numpy()
    # transformers >=5.14 changed recurrent_states from a tensor to a dict
    # keyed by layer index; extract the tensor for either version.
    _rec_states = cache.layers[0].recurrent_states
    hf_rec = (_rec_states[0] if isinstance(_rec_states, dict) else _rec_states).numpy()

    # ONNX forward
    sess = _make_session(onnx_model)
    onnx_out = sess.run(
        {
            "hidden_states": hidden_np,
            "conv_state": conv_np,
            "recurrent_state": rec_np,
        }
    )
    sess.close()

    np.testing.assert_allclose(
        onnx_out["output"],
        hf_output,
        rtol=1e-3,
        atol=1e-3,
        err_msg="DeltaNet output mismatch",
    )
    np.testing.assert_allclose(
        onnx_out["new_recurrent_state"],
        hf_rec,
        rtol=1e-3,
        atol=1e-3,
        err_msg="DeltaNet recurrent_state mismatch",
    )
