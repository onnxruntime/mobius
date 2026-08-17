# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._configs import Lfm2VlConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius._weight_loading import apply_weights
from mobius.models.lfm2_vl import Lfm2VlForConditionalGeneration
from mobius.tasks import Lfm2VlTask

_HIDDEN_SIZE = 64
_VISION_HIDDEN_SIZE = 32
_PATCH_SIZE = 4
_NUM_PATCHES = 16


def _hf_text_config(**overrides) -> SimpleNamespace:
    fields = dict(
        model_type="lfm2",
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=4,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        head_dim=16,
        pad_token_id=0,
        norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        layer_types=["conv", "conv", "full_attention", "conv"],
        block_auto_adjust_ff_dim=False,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=256,
        conv_L_cache=3,
        conv_bias=False,
        tie_word_embeddings=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _hf_vision_config(**overrides) -> SimpleNamespace:
    fields = dict(
        model_type="siglip2_vision_model",
        hidden_size=_VISION_HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_channels=3,
        patch_size=_PATCH_SIZE,
        num_patches=_NUM_PATCHES,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        attention_dropout=0.0,
        vision_use_head=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _hf_composite_config(**overrides) -> SimpleNamespace:
    fields = dict(
        model_type="lfm2_vl",
        text_config=_hf_text_config(),
        vision_config=_hf_vision_config(),
        image_token_id=250,
        downsample_factor=2,
        projector_hidden_act="gelu",
        projector_hidden_size=_HIDDEN_SIZE,
        projector_bias=True,
        projector_use_layernorm=False,
        tie_word_embeddings=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _tiny_config(**overrides) -> Lfm2VlConfig:
    composite = _hf_composite_config(**overrides)
    config = Lfm2VlConfig.from_transformers(
        composite.text_config,
        parent_config=composite,
    )
    config.dtype = ir.DataType.FLOAT
    return config


def test_config_lifts_projector_and_naflex_vision_fields():
    config = _tiny_config()
    assert config.model_type == "lfm2_vl"
    assert config.image_token_id == 250
    assert config.downsample_factor == 2
    assert config.projector_hidden_size == _HIDDEN_SIZE
    assert config.projector_bias is True
    assert config.projector_use_layernorm is False
    assert config.vision is not None
    # HF ``num_patches`` is the learned position table size, not a per-image
    # patch count, and NaFlex towers have no fixed ``image_size``.
    assert config.vision.num_position_embeddings == _NUM_PATCHES
    assert config.vision.hidden_act == "gelu_pytorch_tanh"
    assert config.vision.patch_size == _PATCH_SIZE


def test_package_exposes_three_models_with_hybrid_decoder_cache():
    config = _tiny_config()
    package = Lfm2VlTask().build(Lfm2VlForConditionalGeneration(config), config)

    assert set(package) == {"decoder", "vision_encoder", "embedding"}

    vision_inputs = [value.name for value in package["vision_encoder"].graph.inputs]
    assert vision_inputs == ["pixel_values", "pixel_attention_mask", "spatial_shapes"]

    decoder_inputs = [value.name for value in package["decoder"].graph.inputs]
    assert decoder_inputs[:3] == ["inputs_embeds", "attention_mask", "position_ids"]
    # conv layers carry a single conv_state; full_attention carries key/value.
    assert "past_key_values.0.conv_state" in decoder_inputs
    assert "past_key_values.2.key" in decoder_inputs
    assert "past_key_values.2.value" in decoder_inputs
    assert "past_key_values.2.conv_state" not in decoder_inputs


def _hf_vision_and_projector(composite):
    """Build the reference HF vision tower and projector with shared config."""
    from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig as HfConfig
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector
    from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

    vision_config = Siglip2VisionConfig(
        hidden_size=_VISION_HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        patch_size=_PATCH_SIZE,
        num_patches=_NUM_PATCHES,
        layer_norm_eps=1e-6,
        vision_use_head=False,
    )
    hf_config = HfConfig(
        vision_config=vision_config.to_dict(),
        text_config=vars(composite.text_config),
        downsample_factor=composite.downsample_factor,
        projector_hidden_act=composite.projector_hidden_act,
        projector_hidden_size=composite.projector_hidden_size,
        projector_bias=composite.projector_bias,
        projector_use_layernorm=composite.projector_use_layernorm,
        image_token_id=composite.image_token_id,
    )
    hf_vision = Siglip2VisionModel(hf_config.vision_config).eval()
    hf_projector = Lfm2VlMultiModalProjector(hf_config).eval()
    return hf_vision, hf_projector


def _hf_image_features(hf_vision, hf_projector, pixel_values, mask, spatial_shapes):
    """Reproduce ``Lfm2VlModel.get_image_features`` for the reference."""
    with torch.no_grad():
        last_hidden_state = hf_vision(
            pixel_values=pixel_values,
            pixel_attention_mask=mask,
            spatial_shapes=spatial_shapes,
        ).last_hidden_state
        lengths = mask.sum(dim=1)
        features = []
        for index in range(last_hidden_state.size(0)):
            feature = last_hidden_state[index][: lengths[index], :].unsqueeze(0)
            height, width = spatial_shapes[index]
            feature = feature.reshape(1, height, width, -1)
            projected = hf_projector(feature)
            features.append(projected.reshape(-1, projected.size(-1)))
        return torch.cat(features, dim=0).numpy()


def _naflex_inputs(shapes: list[tuple[int, int]], seed: int = 0):
    """Build padded NaFlex tensors for images with the given patch grids."""
    rng = np.random.default_rng(seed)
    max_patches = max(h * w for h, w in shapes)
    patch_dim = 3 * _PATCH_SIZE * _PATCH_SIZE
    pixel_values = np.zeros((len(shapes), max_patches, patch_dim), dtype=np.float32)
    mask = np.zeros((len(shapes), max_patches), dtype=np.int64)
    for index, (height, width) in enumerate(shapes):
        count = height * width
        pixel_values[index, :count] = rng.standard_normal((count, patch_dim))
        mask[index, :count] = 1
    return pixel_values, mask, np.array(shapes, dtype=np.int64)


@pytest.mark.parametrize(
    "shapes",
    [
        [(4, 4)],
        [(2, 6)],
        [(4, 4), (2, 8), (6, 2)],
    ],
)
def test_vision_encoder_matches_transformers(shapes):
    """L3: NaFlex tower + pixel-unshuffle projector match HF on random weights."""
    torch.manual_seed(42)
    composite = _hf_composite_config()
    hf_vision, hf_projector = _hf_vision_and_projector(composite)

    config = _tiny_config()
    module = Lfm2VlForConditionalGeneration(config)
    package = Lfm2VlTask().build(module, config)

    state_dict = {
        **{
            f"model.vision_tower.{name}": value
            for name, value in hf_vision.state_dict().items()
        },
        **{
            f"model.multi_modal_projector.{name}": value
            for name, value in hf_projector.state_dict().items()
        },
    }
    processed = module.preprocess_weights(state_dict)
    graph_parameters = set(package["vision_encoder"].graph.initializers)
    vision_weights = {
        name: value for name, value in processed.items() if name.startswith("vision_encoder.")
    }
    assert set(vision_weights) <= graph_parameters
    assert {name for name in graph_parameters if name.startswith("vision_encoder.")} == set(
        vision_weights
    )
    apply_weights(package["vision_encoder"], vision_weights)

    pixel_values, mask, spatial_shapes = _naflex_inputs(shapes)
    expected = _hf_image_features(
        hf_vision,
        hf_projector,
        torch.from_numpy(pixel_values),
        torch.from_numpy(mask),
        torch.from_numpy(spatial_shapes),
    )

    session = OnnxModelSession(package["vision_encoder"])
    actual = session.run(
        {
            "pixel_values": pixel_values,
            "pixel_attention_mask": mask,
            "spatial_shapes": spatial_shapes,
        }
    )["image_features"]
    session.close()

    assert actual.shape == expected.shape
    assert actual.shape[0] == sum((h // 2) * (w // 2) for h, w in shapes)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_embedding_scatters_image_features_and_supports_cached_decode():
    config = _tiny_config()
    package = Lfm2VlTask().build(Lfm2VlForConditionalGeneration(config), config)

    rng = np.random.default_rng(7)
    embedding_weight = None
    for name, initializer in package["embedding"].graph.initializers.items():
        if initializer.const_value is None:
            value = rng.standard_normal(initializer.shape).astype(np.float32)
            initializer.const_value = ir.tensor(value)
            if name.endswith("embed_tokens.weight"):
                embedding_weight = value
    assert embedding_weight is not None

    input_ids = np.array([[1, 250, 250, 2], [250, 3, 4, 250]], dtype=np.int64)
    features = np.arange(4 * config.hidden_size, dtype=np.float32).reshape(
        4, config.hidden_size
    )

    session = OnnxModelSession(package["embedding"])
    result = session.run({"input_ids": input_ids, "image_features": features})["inputs_embeds"]

    image_mask = input_ids == config.image_token_id
    expected = embedding_weight[input_ids].copy()
    expected[image_mask] = features
    np.testing.assert_array_equal(result, expected)

    # Cached decode steps carry no new image features.
    decode_ids = np.array([[5], [6]], dtype=np.int64)
    decode_result = session.run(
        {
            "input_ids": decode_ids,
            "image_features": np.empty((0, config.hidden_size), dtype=np.float32),
        }
    )["inputs_embeds"]
    session.close()
    np.testing.assert_array_equal(decode_result, embedding_weight[decode_ids])


def _hf_causal_lm():
    """Build the reference HF LFM2 decoder with the tiny test geometry."""
    from transformers.models.lfm2.configuration_lfm2 import Lfm2Config as HfLfm2Config
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

    text = _hf_text_config()
    hf_config = HfLfm2Config(
        vocab_size=text.vocab_size,
        hidden_size=text.hidden_size,
        intermediate_size=text.intermediate_size,
        num_hidden_layers=text.num_hidden_layers,
        num_attention_heads=text.num_attention_heads,
        num_key_value_heads=text.num_key_value_heads,
        max_position_embeddings=text.max_position_embeddings,
        norm_eps=text.norm_eps,
        rope_parameters=text.rope_parameters,
        layer_types=text.layer_types,
        block_auto_adjust_ff_dim=False,
        conv_L_cache=text.conv_L_cache,
        conv_bias=text.conv_bias,
        tie_word_embeddings=True,
        pad_token_id=text.pad_token_id,
    )
    return Lfm2ForCausalLM(hf_config).eval()


def _empty_decoder_cache(config: Lfm2VlConfig, batch: int) -> dict[str, np.ndarray]:
    """Zero-length hybrid cache inputs for a prefill step."""
    feeds: dict[str, np.ndarray] = {}
    for index, layer_type in enumerate(config.layer_types or []):
        if layer_type == "conv":
            feeds[f"past_key_values.{index}.conv_state"] = np.zeros(
                (batch, config.hidden_size, config.short_conv_kernel - 1),
                dtype=np.float32,
            )
        else:
            empty = np.zeros(
                (batch, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
            feeds[f"past_key_values.{index}.key"] = empty
            feeds[f"past_key_values.{index}.value"] = empty
    return feeds


def test_decoder_matches_transformers_on_prefill_and_cached_decode():
    """L3: the inputs_embeds decoder and hybrid cache match HF step for step."""
    torch.manual_seed(11)
    hf_model = _hf_causal_lm()

    config = _tiny_config()
    module = Lfm2VlForConditionalGeneration(config)
    package = Lfm2VlTask().build(module, config)

    state_dict = {
        f"model.language_model.{name[len('model.') :]}": value
        for name, value in hf_model.state_dict().items()
        if name.startswith("model.")
    }
    processed = module.preprocess_weights(state_dict)
    graph_parameters = set(package["decoder"].graph.initializers)
    # The decoder consumes inputs_embeds, so the embedding table only appears
    # through the tied lm_head initializer.
    assert "decoder.lm_head.weight" in graph_parameters
    apply_weights(
        package["decoder"],
        {name: value for name, value in processed.items() if name in graph_parameters},
    )

    batch, prefill_length = 1, 6
    prefill_embeds = torch.randn(batch, prefill_length, config.hidden_size)
    decode_embeds = torch.randn(batch, 1, config.hidden_size)
    with torch.no_grad():
        hf_prefill = hf_model(inputs_embeds=prefill_embeds, use_cache=True)
        hf_decode = hf_model(
            inputs_embeds=decode_embeds,
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
        )

    session = OnnxModelSession(package["decoder"])
    prefill_feeds = {
        "inputs_embeds": prefill_embeds.numpy(),
        "attention_mask": np.ones((batch, prefill_length), dtype=np.int64),
        "position_ids": np.arange(prefill_length, dtype=np.int64)[None],
        **_empty_decoder_cache(config, batch),
    }
    prefill_result = session.run(prefill_feeds)
    np.testing.assert_allclose(
        prefill_result["logits"],
        hf_prefill.logits.numpy(),
        rtol=1e-4,
        atol=1e-4,
    )

    decode_feeds = {
        "inputs_embeds": decode_embeds.numpy(),
        "attention_mask": np.ones((batch, prefill_length + 1), dtype=np.int64),
        "position_ids": np.array([[prefill_length]], dtype=np.int64),
    }
    for name, value in prefill_result.items():
        if name.startswith("present."):
            decode_feeds[f"past_key_values.{name[len('present.') :]}"] = value
    decode_result = session.run(decode_feeds)
    session.close()
    np.testing.assert_allclose(
        decode_result["logits"],
        hf_decode.logits.numpy(),
        rtol=1e-4,
        atol=1e-4,
    )


def test_preprocess_weights_routes_checkpoint_into_sub_models():
    config = _tiny_config()
    module = Lfm2VlForConditionalGeneration(config)
    embed = torch.zeros(config.vocab_size, config.hidden_size)
    state_dict = {
        # Checkpoints saved before the Transformers v5 flattening nest the
        # SigLIP2 tower under an extra ``vision_model.`` scope.
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight": torch.zeros(
            _VISION_HIDDEN_SIZE, 3 * _PATCH_SIZE * _PATCH_SIZE
        ),
        "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight": torch.zeros(
            64, _VISION_HIDDEN_SIZE
        ),
        "model.vision_tower.vision_model.post_layernorm.weight": torch.zeros(
            _VISION_HIDDEN_SIZE
        ),
        "model.multi_modal_projector.linear_1.bias": torch.zeros(config.hidden_size),
        "model.language_model.embed_tokens.weight": embed,
        "model.language_model.layers.2.self_attn.out_proj.weight": torch.zeros(
            config.hidden_size, config.hidden_size
        ),
        "model.language_model.layers.0.feed_forward.w1.weight": torch.zeros(
            config.intermediate_size, config.hidden_size
        ),
    }
    renamed = module.preprocess_weights(state_dict)

    assert "vision_encoder.vision_tower.embeddings.patch_embedding.weight" in renamed
    assert "vision_encoder.vision_tower.encoder.layers.0.mlp.up_proj.weight" in renamed
    assert "vision_encoder.vision_tower.post_layernorm.weight" in renamed
    assert "vision_encoder.multi_modal_projector.linear_1.bias" in renamed
    assert "decoder.model.embed_tokens.weight" in renamed
    assert "embedding.embed_tokens.weight" in renamed
    # LFM2 projection renames must apply to the routed decoder weights.
    assert "decoder.model.layers.2.self_attn.o_proj.weight" in renamed
    assert "decoder.model.layers.0.feed_forward.gate_proj.weight" in renamed
    # No lm_head in the checkpoint: it is tied to the embedding table.
    assert renamed["decoder.lm_head.weight"] is embed


def test_registered_model_and_task():
    from mobius._registry import registry

    assert registry.get("lfm2_vl") is Lfm2VlForConditionalGeneration
    assert registry.get_task("lfm2_vl") == "lfm2-vl"
    assert registry.get_config_class("lfm2_vl") is Lfm2VlConfig
