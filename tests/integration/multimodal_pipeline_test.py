# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests for synthetic multimodal three-model pipelines."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from integration._support import (
    _fill_random_weights,
    _make_session,
)
from mobius import models
from mobius._configs import ArchitectureConfig, VisionConfig
from mobius._testing.comparison import (
    assert_logits_close,
)


@pytest.mark.integration
@pytest.mark.integration_fast
def test_gemma3_3model_builds_and_runs():
    """Gemma3 multimodal 3-model split: build + graph structure verification.

    Uses a tiny config with random weights. Verifies:
    - Package contains 3 models (decoder, vision, embedding)
    - Vision model has pixel_values input
    - Decoder model produces logits output
    - Embedding model has input_ids and image_features inputs
    """
    import onnx_ir as ir

    from mobius._registry import registry
    from mobius.tasks import get_task

    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10000.0,
        attn_qk_norm=True,
        rope_local_base_freq=10_000.0,
        layer_types=["full_attention", "sliding_attention"],
        # Vision config
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
            norm_eps=1e-6,
            mm_tokens_per_image=4,
        ),
        image_token_id=255999,
        dtype=ir.DataType.FLOAT,
    )

    model_cls = registry.get("gemma3")
    module = model_cls(config)
    task = get_task("vision-language")
    pkg = task.build(module, config)

    # Verify 3-model split structure
    assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

    # Verify vision model I/O
    vision_inputs = {i.name for i in pkg["vision_encoder"].graph.inputs}
    assert "pixel_values" in vision_inputs

    # Verify decoder model I/O
    decoder_outputs = {o.name for o in pkg["decoder"].graph.outputs}
    assert "logits" in decoder_outputs
    decoder_inputs = {i.name for i in pkg["decoder"].graph.inputs}
    assert "inputs_embeds" in decoder_inputs

    # Verify embedding model I/O
    embed_inputs = {i.name for i in pkg["embedding"].graph.inputs}
    assert "input_ids" in embed_inputs
    assert "image_features" in embed_inputs

    # Verify KV cache is present in decoder
    kv_inputs = [i.name for i in pkg["decoder"].graph.inputs if "past_key_values" in i.name]
    assert len(kv_inputs) > 0, "Decoder should have KV cache inputs"
    kv_outputs = [o.name for o in pkg["decoder"].graph.outputs if "present" in o.name]
    assert len(kv_outputs) > 0, "Decoder should have KV cache outputs"

    print(
        f"Gemma3 multimodal 3-model split OK: "
        f"decoder({len(list(pkg['decoder'].graph.inputs))} inputs), "
        f"vision({len(list(pkg['vision_encoder'].graph.inputs))} inputs), "
        f"embedding({len(list(pkg['embedding'].graph.inputs))} inputs)"
    )


@pytest.mark.integration
@pytest.mark.integration_fast
def test_gemma3_embedding_runs_with_empty_image_features():
    """Gemma3 embedding graph survives a decode step (empty image_features).

    Regression guard for the decode-step Gather crash: ORT-GenAI re-runs the
    embedding model per generated token, and a decode token is text-only, so
    ``image_features`` is ``[0, hidden]`` and the image mask is all-False. The
    Where would discard the gathered value, but ORT executes the Gather first
    and indexing an empty tensor at the Clip-clamped index 0 fails with
    "indices element out of data bounds, range [0,-1]".

    ``_Gemma3EmbeddingModel.forward`` pads ``image_features`` with a single zero
    row so index 0 always references a valid row that Where never selects. This
    test runs the embedding graph with empty features + text-only input_ids and
    asserts it returns pure text embeddings without error. It fails (Gather
    out-of-bounds) if the zero-row pad is removed.
    """
    import onnx_ir as ir

    from mobius._registry import registry
    from mobius.tasks import get_task

    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10000.0,
        attn_qk_norm=True,
        rope_local_base_freq=10_000.0,
        layer_types=["full_attention", "sliding_attention"],
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
            norm_eps=1e-6,
            mm_tokens_per_image=4,
        ),
        image_token_id=255999,
        dtype=ir.DataType.FLOAT,
    )

    model_cls = registry.get("gemma3")
    module = model_cls(config)
    task = get_task("vision-language")
    pkg = task.build(module, config)

    # Fill initializers with random weights so the graph can execute.
    rng = np.random.default_rng(42)
    for model in pkg.values():
        for init in model.graph.initializers.values():
            if init.const_value is None:
                shape = [d if isinstance(d, int) else 1 for d in init.shape]
                init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

    # Decode step: a single text-only token, no image_token_id present, and an
    # empty image_features tensor ([0, hidden]).  This is the exact condition
    # that crashed the unpadded Gather.
    embed_sess = _make_session(pkg["embedding"])
    input_ids = np.array([[1]], dtype=np.int64)
    image_features = np.zeros((0, config.hidden_size), dtype=np.float32)
    embed_out = embed_sess.run({"input_ids": input_ids, "image_features": image_features})
    embed_sess.close()

    assert "inputs_embeds" in embed_out
    assert embed_out["inputs_embeds"].shape == (1, 1, config.hidden_size)


@pytest.mark.integration
@pytest.mark.integration_fast
class TestBlip2VL:
    """BLIP-2 3-model split: vision, embedding, decoder with random weights."""

    @staticmethod
    def _build_blip2():
        """Build tiny BLIP-2 package and fill with random weights."""
        import onnx_ir as ir

        from mobius._configs import ArchitectureConfig
        from mobius._registry import registry
        from mobius.tasks import get_task

        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10_000.0,
            pad_token_id=0,
            dtype=ir.DataType.FLOAT,
            # Vision config
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=50265,
            # Q-Former config
            num_query_tokens=4,
            qformer_hidden_size=32,
            qformer_num_hidden_layers=1,
            qformer_num_attention_heads=2,
            qformer_intermediate_size=64,
        )

        model_cls = registry.get("blip-2")
        module = model_cls(config)
        task = get_task("vision-language")
        pkg = task.build(module, config)

        # Fill all 3 models with random weights
        rng = np.random.default_rng(42)
        for model in pkg.values():
            _fill_random_weights(model, rng)

        return pkg, config

    def test_blip2_3model_structure(self):
        """BLIP-2 produces decoder, vision, and embedding models."""
        pkg, _config = self._build_blip2()
        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

    def test_blip2_vision_model(self):
        """Vision model: pixel_values -> image_features via ViT + Q-Former."""
        pkg, config = self._build_blip2()
        session = _make_session(pkg["vision_encoder"])

        rng = np.random.default_rng(123)
        img_size = config.vision.image_size if config.vision else None
        pixel_values = rng.standard_normal((1, 3, img_size, img_size)).astype(np.float32)

        outputs = session.run({"pixel_values": pixel_values})
        session.close()

        assert "image_features" in outputs
        feats = outputs["image_features"]
        # Q-Former produces num_query_tokens features projected to hidden_size
        assert feats.shape[-1] == config.hidden_size
        assert np.all(np.isfinite(feats)), "Vision features contain NaN/Inf"

    def test_blip2_embedding_model(self):
        """Embedding model: input_ids + image_features -> inputs_embeds."""
        pkg, config = self._build_blip2()
        session = _make_session(pkg["embedding"])

        rng = np.random.default_rng(456)
        input_ids = rng.integers(0, config.vocab_size, size=(1, 5)).astype(np.int64)
        # Provide 1 dummy row — ORT evaluates Gather eagerly even when
        # the Where mask is all-false (no image tokens in input_ids).
        image_features = np.zeros((1, config.hidden_size), dtype=np.float32)

        outputs = session.run(
            {
                "input_ids": input_ids,
                "image_features": image_features,
            }
        )
        session.close()

        assert "inputs_embeds" in outputs
        embeds = outputs["inputs_embeds"]
        assert embeds.shape == (1, 5, config.hidden_size)
        assert np.all(np.isfinite(embeds)), "Embeds contain NaN/Inf"

    def test_blip2_decoder_model(self):
        """Decoder model: inputs_embeds -> logits + KV cache."""
        pkg, config = self._build_blip2()
        session = _make_session(pkg["decoder"])

        rng = np.random.default_rng(789)
        seq_len = 3
        inputs_embeds = rng.standard_normal((1, seq_len, config.hidden_size)).astype(
            np.float32
        )

        feeds = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64)[np.newaxis, :],
        }
        for i in range(config.num_hidden_layers):
            feeds[f"past_key_values.{i}.key"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
            feeds[f"past_key_values.{i}.value"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )

        outputs = session.run(feeds)
        session.close()

        assert "logits" in outputs
        logits = outputs["logits"]
        assert logits.shape == (1, seq_len, config.vocab_size)
        assert np.all(np.isfinite(logits)), "Decoder logits contain NaN/Inf"

        for i in range(config.num_hidden_layers):
            key = outputs[f"present.{i}.key"]
            assert key.shape[2] == seq_len, (
                f"Layer {i} key cache should have {seq_len} entries"
            )


class _TorchInternAttention(torch.nn.Module):
    """PyTorch reference for InternViT fused-QKV attention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        b, n, c = x.shape
        qkv = (
            self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(attn.transpose(1, 2).reshape(b, n, c))


class _TorchInternViTLayer(torch.nn.Module):
    """PyTorch reference for InternViT encoder layer with layer scale."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.attn = _TorchInternAttention(hidden_size, num_heads)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, intermediate_size),
            torch.nn.GELU(approximate="none"),
            torch.nn.Linear(intermediate_size, hidden_size),
        )
        self.norm1 = torch.nn.LayerNorm(hidden_size, eps=norm_eps)
        self.norm2 = torch.nn.LayerNorm(hidden_size, eps=norm_eps)
        self.ls1 = torch.nn.Parameter(torch.ones(hidden_size))
        self.ls2 = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.ls1
        x = x + self.mlp(self.norm2(x)) * self.ls2
        return x


class _TorchInternViTEmbeddings(torch.nn.Module):
    """Patch embedding + CLS token + position embedding."""

    def __init__(self, image_size: int, patch_size: int, hidden_size: int):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.class_embedding = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.patch_embedding = torch.nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.position_embedding = torch.nn.Parameter(
            torch.randn(1, num_patches + 1, hidden_size)
        )

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        p = self.patch_embedding(pixel_values)
        p = p.flatten(2).transpose(1, 2)  # (batch, num_patches, hidden)
        cls_tokens = self.class_embedding.expand(batch, -1, -1)
        h = torch.cat([cls_tokens, p], dim=1)
        return h + self.position_embedding


class _TorchInternViTEncoder(torch.nn.Module):
    """Stack of encoder layers."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                _TorchInternViTLayer(hidden_size, intermediate_size, num_heads, norm_eps)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _TorchInternViT(torch.nn.Module):
    """PyTorch reference for InternViT (matches HF InternVisionModel).

    Uses ``embeddings`` and ``encoder`` sub-modules to produce state_dict
    keys matching HF naming: ``embeddings.class_embedding``,
    ``encoder.layers.0.attn.qkv.weight``, etc.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.embeddings = _TorchInternViTEmbeddings(image_size, patch_size, hidden_size)
        self.encoder = _TorchInternViTEncoder(
            num_layers, hidden_size, intermediate_size, num_heads, norm_eps
        )

    def forward(self, pixel_values):
        h = self.embeddings(pixel_values)
        h = self.encoder(h)
        return h


def _pixel_shuffle_v2(x: torch.Tensor, scale: float = 0.5):
    """Pixel shuffle v2 matching HF InternVLChatModel.pixel_shuffle."""
    n, h, w, c = x.shape
    x = x.reshape(n, w, int(h * scale), int(c / scale))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.reshape(n, int(h * scale), int(w * scale), int(c / (scale * scale)))
    # v2: permute back
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.reshape(n, -1, int(c / (scale * scale)))
    return x


def _make_tiny_internvl2_config():
    """Create a tiny InternVL2 ArchitectureConfig for fast parity tests.

    Vision: 32-dim, 1 layer, 2 heads, 28x28 image, 14x14 patch.
    Text: 64-dim, 2 layers, 4 heads, Qwen2 decoder, vocab=256.
    """
    return ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        pad_token_id=0,
        attn_qkv_bias=True,
        image_token_id=200,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
            norm_eps=1e-6,
        ),
    )


@pytest.mark.integration
@pytest.mark.integration_fast
def test_internvl2_3model_parity():
    """InternVL2 3-model split matches PyTorch reference.

    Builds decoder, vision, and embedding ONNX models from a tiny
    InternVL2 config with random weights. Compares each sub-model
    against a PyTorch reference:
    1. Vision encoder (InternViT + pixel shuffle + MLP projector)
    2. Embedding model (token lookup + image feature scatter)
    3. Decoder model (Qwen2 text decoder)
    """
    import onnx_ir as ir
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from mobius import build_from_module
    from mobius.integrations._weight_loading import apply_weights

    config = _make_tiny_internvl2_config()
    config.dtype = ir.DataType.FLOAT
    vc = config.vision

    # ----- Build ONNX 3-model package -----
    onnx_module = models.InternVL2Model(config)
    pkg = build_from_module(onnx_module, config, task="vision-language")
    assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

    # ----- Build PyTorch reference models -----
    # Vision: InternViT + pixel shuffle + MLP projector
    ref_vit = _TorchInternViT(
        image_size=vc.image_size,
        patch_size=vc.patch_size,
        hidden_size=vc.hidden_size,
        intermediate_size=vc.intermediate_size,
        num_layers=vc.num_hidden_layers,
        num_heads=vc.num_attention_heads,
        norm_eps=vc.norm_eps,
    ).eval()

    # MLP projector: LayerNorm → Linear → GELU → Linear
    proj_input_dim = vc.hidden_size * 4  # after pixel shuffle(0.5)
    ref_mlp1 = torch.nn.Sequential(
        torch.nn.LayerNorm(proj_input_dim),
        torch.nn.Linear(proj_input_dim, config.hidden_size),
        torch.nn.GELU(approximate="none"),
        torch.nn.Linear(config.hidden_size, config.hidden_size),
    ).eval()

    # Decoder: Qwen2ForCausalLM
    qwen2_cfg = Qwen2Config(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        head_dim=config.head_dim,
        hidden_act=config.hidden_act,
    )
    ref_decoder = Qwen2ForCausalLM._from_config(qwen2_cfg).float().eval()

    # ----- Assemble HF-style state dict -----
    full_state: dict[str, torch.Tensor] = {}

    # Vision encoder weights: vision_model.embeddings.*, vision_model.encoder.*
    for k, v in ref_vit.state_dict().items():
        # Map PyTorch MLP keys (mlp.0/1/2) to HF keys (mlp.fc1/fc2)
        k = k.replace(".mlp.0.", ".mlp.fc1.")
        k = k.replace(".mlp.2.", ".mlp.fc2.")
        full_state[f"vision_model.{k}"] = v

    # MLP projector weights: mlp1.{0,1,3}.*
    for k, v in ref_mlp1.state_dict().items():
        full_state[f"mlp1.{k}"] = v

    # Decoder weights: language_model.model.* / language_model.lm_head.*
    for k, v in ref_decoder.state_dict().items():
        full_state[f"language_model.{k}"] = v

    # Preprocess and apply to ONNX models
    preprocessed = onnx_module.preprocess_weights(dict(full_state))
    for onnx_model in pkg.values():
        apply_weights(onnx_model, preprocessed)

    # ----- Test inputs -----
    rng = np.random.default_rng(42)
    num_patches = (vc.image_size // vc.patch_size) ** 2  # 4
    # After pixel shuffle(0.5): H/2 * W/2 = 1 token
    num_image_tokens = int(num_patches * 0.5 * 0.5)  # 1

    pixel_values = rng.standard_normal((1, 3, vc.image_size, vc.image_size)).astype(np.float32)

    # input_ids with one image token placeholder
    text_tokens = rng.integers(0, config.vocab_size, size=(1, 3)).astype(np.int64)
    image_tokens = np.full((1, num_image_tokens), config.image_token_id, dtype=np.int64)
    input_ids = np.concatenate([text_tokens[:, :1], image_tokens, text_tokens[:, 1:]], axis=1)
    seq_len = input_ids.shape[1]

    # ----- 1. Vision encoder parity -----
    # PyTorch reference: InternViT → strip CLS → pixel shuffle → MLP
    with torch.no_grad():
        vit_out = ref_vit(torch.from_numpy(pixel_values))
        # Strip CLS token
        vit_features = vit_out[:, 1:, :]  # (1, num_patches, hidden)
        # Pixel shuffle
        grid = int(vit_features.shape[1] ** 0.5)
        spatial = vit_features.reshape(1, grid, grid, -1)
        shuffled = _pixel_shuffle_v2(spatial, scale=0.5)
        # MLP projector
        ref_image_features = ref_mlp1(shuffled).numpy()

    # ONNX vision
    vision_sess = _make_session(pkg["vision_encoder"])
    onnx_vision_out = vision_sess.run({"pixel_values": pixel_values})
    vision_sess.close()
    onnx_image_features = onnx_vision_out["image_features"]

    assert onnx_image_features.shape == ref_image_features.shape, (
        f"Vision shape mismatch: ONNX {onnx_image_features.shape} "
        f"vs ref {ref_image_features.shape}"
    )
    assert_logits_close(
        onnx_image_features,
        ref_image_features,
        rtol=1e-3,
        atol=1e-3,
    )

    # ----- 2. Embedding model parity -----
    # PyTorch reference: embed_tokens + scatter
    # image_features is (1, num_tokens, hidden) from vision; squeeze to 2D
    img_feats_2d = ref_image_features.squeeze(0)  # (num_tokens, hidden)
    with torch.no_grad():
        ref_text_embeds = ref_decoder.model.embed_tokens(torch.from_numpy(input_ids))
        # Scatter image features at image token positions
        mask = torch.from_numpy(input_ids) == config.image_token_id
        mask_3d = mask.unsqueeze(-1)
        mask_int = mask.long()
        cumsum = torch.cumsum(mask_int, dim=1) - 1
        cumsum = cumsum.clamp(min=0)
        img_feats_torch = torch.from_numpy(img_feats_2d)
        gathered = img_feats_torch[cumsum.squeeze(0)]
        gathered = gathered.unsqueeze(0) if gathered.dim() == 2 else gathered
        ref_inputs_embeds = torch.where(mask_3d, gathered, ref_text_embeds).numpy()

    # ONNX embedding — image_features is 2D (num_tokens, hidden_size)
    embed_sess = _make_session(pkg["embedding"])
    onnx_embed_out = embed_sess.run(
        {
            "input_ids": input_ids,
            "image_features": ref_image_features.squeeze(0),
        }
    )
    embed_sess.close()
    onnx_inputs_embeds = onnx_embed_out["inputs_embeds"]

    assert onnx_inputs_embeds.shape == ref_inputs_embeds.shape, (
        f"Embedding shape mismatch: ONNX {onnx_inputs_embeds.shape} "
        f"vs ref {ref_inputs_embeds.shape}"
    )
    assert_logits_close(
        onnx_inputs_embeds,
        ref_inputs_embeds,
        rtol=1e-3,
        atol=1e-3,
    )

    # ----- 3. Decoder model parity -----
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

    with torch.no_grad():
        hf_logits = ref_decoder(
            inputs_embeds=torch.from_numpy(onnx_inputs_embeds),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
        ).logits.numpy()

    # ONNX decoder
    decoder_sess = _make_session(pkg["decoder"])
    feeds: dict[str, np.ndarray] = {
        "inputs_embeds": onnx_inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for i in range(config.num_hidden_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
    onnx_logits = decoder_sess.run(feeds)["logits"]
    decoder_sess.close()

    assert_logits_close(onnx_logits, hf_logits, rtol=1e-3, atol=1e-3)
