# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph tasks for GLM-5.3's heterogeneous NoPE KDA/DSA state."""

from __future__ import annotations

import json
from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import Glm5NextConfig
from mobius._model_package import ModelPackage
from mobius.functions import linear_attention
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model

_PINNED_MODEL_REVISION = "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
_PINNED_TRANSFORMERS_REVISION = "eb4d9e2a64a013bec12289288b85d0b1210ba0aa"


def _make_glm5_state_inputs(
    builder,
    config: Glm5NextConfig,
    batch: ir.SymbolicDim,
    past_length: ir.SymbolicDim,
) -> list[tuple[ir.Value, ...]]:
    assert config.layer_types is not None
    assert config.linear_num_heads is not None
    assert config.linear_head_dim is not None
    assert config.qk_nope_head_dim is not None
    assert config.v_head_dim is not None
    assert config.index_head_dim is not None
    projection = config.linear_num_heads * config.linear_head_dim
    states: list[tuple[ir.Value, ...]] = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        prefix = f"past_key_values.{layer_idx}"
        if layer_type == "linear_attention":
            conv_state = builder.input(
                f"{prefix}.conv_state",
                dtype=config.dtype,
                shape=[
                    batch,
                    3 * projection,
                    config.linear_conv_kernel_dim,
                ],
            )
            recurrent_state = builder.input(
                f"{prefix}.recurrent_state",
                dtype=ir.DataType.FLOAT,
                shape=[
                    batch,
                    config.linear_num_heads,
                    config.linear_head_dim,
                    config.linear_head_dim,
                ],
            )
            states.append((conv_state, recurrent_state))
        else:
            key = builder.input(
                f"{prefix}.key",
                dtype=config.dtype,
                shape=[
                    batch,
                    config.num_attention_heads,
                    past_length,
                    config.qk_nope_head_dim,
                ],
            )
            value = builder.input(
                f"{prefix}.value",
                dtype=config.dtype,
                shape=[
                    batch,
                    config.num_attention_heads,
                    past_length,
                    config.v_head_dim,
                ],
            )
            indexer_state = builder.input(
                f"{prefix}.indexer_state",
                dtype=config.dtype,
                shape=[
                    batch,
                    past_length,
                    2 * config.index_head_dim + 1,
                ],
            )
            states.append((key, value, indexer_state))
    return states


def _register_glm5_outputs(
    builder,
    config: Glm5NextConfig,
    logits: ir.Value,
    presents: list[tuple[ir.Value, ...]],
) -> None:
    assert config.layer_types is not None
    builder.add_output(logits, "logits")
    for layer_idx, (layer_type, states) in enumerate(zip(config.layer_types, presents)):
        names = (
            ("conv_state", "recurrent_state")
            if layer_type == "linear_attention"
            else ("key", "value", "indexer_state")
        )
        for state, name in zip(states, names):
            builder.add_output(state, f"present.{layer_idx}.{name}")


def _finalize_glm5_model(graph, config: Glm5NextConfig) -> ir.Model:
    assert config.linear_num_heads is not None
    assert config.linear_head_dim is not None
    model = _make_model(graph)
    recurrence = linear_attention(
        q_num_heads=config.linear_num_heads,
        kv_num_heads=config.linear_num_heads,
        update_rule="gated_delta",
        scale=config.linear_head_dim**-0.5,
        stash_type=ir.DataType.FLOAT,
    )
    model.functions[recurrence.identifier()] = recurrence
    model.metadata_props["mobius.cache_abi"] = (
        "glm5-next:linear=conv_state,recurrent_state;"
        "dsa=key,value,indexer_state(k|pool_gate|valid);NoPE"
    )
    model.metadata_props["mobius.state_manifest"] = json.dumps(
        {
            "schema_version": 1,
            "layers": [
                {
                    "index": index,
                    "type": layer_type,
                    "roles": (
                        ["conv_state", "recurrent_state"]
                        if layer_type == "linear_attention"
                        else ["key", "value", "indexer_state"]
                    ),
                    "update": {
                        role: (
                            "append"
                            if role in {"key", "value", "indexer_state"}
                            else "replace"
                        )
                        for role in (
                            ["conv_state", "recurrent_state"]
                            if layer_type == "linear_attention"
                            else ["key", "value", "indexer_state"]
                        )
                    },
                }
                for index, layer_type in enumerate(config.layer_types or [])
            ],
        },
        separators=(",", ":"),
    )
    model.metadata_props["mobius.semantic_reference_revision"] = (
        f"zai-org/GLM-5.3-Flash@{_PINNED_MODEL_REVISION};"
        f"transformers@{_PINNED_TRANSFORMERS_REVISION}"
    )
    return model


def _build_text_decoder(
    module: nn.Module,
    config: Glm5NextConfig,
    *,
    inputs_embeds: bool,
) -> ir.Model:
    batch = ir.SymbolicDim("batch")
    sequence_length = ir.SymbolicDim("sequence_length")
    past_length = ir.SymbolicDim("past_sequence_length")
    graph, builder = _make_graph("glm5_next")
    if inputs_embeds:
        primary = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=[batch, sequence_length, config.hidden_size],
        )
    else:
        primary = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, sequence_length],
        )
    attention_mask = builder.input(
        "attention_mask",
        dtype=ir.DataType.INT64,
        shape=[batch, "past_sequence_length + sequence_length"],
    )
    # Retain the standard decoder API even though the pinned architecture is
    # strictly NoPE and intentionally does not consume position values.
    position_ids = builder.input(
        "position_ids",
        dtype=ir.DataType.INT64,
        shape=[batch, sequence_length],
    )
    states = _make_glm5_state_inputs(builder, config, batch, past_length)
    if inputs_embeds:
        logits, presents = module(
            builder.op,
            inputs_embeds=primary,
            attention_mask=attention_mask,
            past_key_values=states,
            position_ids=position_ids,
        )
    else:
        logits, presents = module(
            builder.op,
            input_ids=primary,
            attention_mask=attention_mask,
            past_key_values=states,
            position_ids=position_ids,
        )
    _register_glm5_outputs(builder, config, logits, presents)
    return _finalize_glm5_model(graph, config)


class Glm5NextTextTask(ModelTask):
    """Build the standalone GLM-5.3 text decoder."""

    def __init__(self, *, static_cache: bool = False, **_: object) -> None:
        if static_cache:
            raise ValueError("GLM-5.3 heterogeneous recurrent state is dynamic-only")

    def build(self, module: nn.Module, config: Glm5NextConfig) -> ModelPackage:
        return ModelPackage(
            {"model": _build_text_decoder(module, config, inputs_embeds=False)},
            config=config,
        )


class Glm5NextVisionLanguageTask(ModelTask):
    """Build GLM-5.3 decoder, packed image/video encoder, and embedding mixer."""

    model_roles: ClassVar[dict[str, str]] = {
        "decoder": "decoder",
        "vision_encoder": "encoder",
        "embedding": "embedding",
    }
    components = ComponentSpec(
        decoder="decoder",
        vision_encoder="vision_encoder",
        embedding="embedding",
    )

    def __init__(self, *, static_cache: bool = False, **_: object) -> None:
        if static_cache:
            raise ValueError("GLM-5.3 heterogeneous recurrent state is dynamic-only")

    def build(
        self,
        module: nn.Module,
        config: Glm5NextConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models = {
            "decoder": _build_text_decoder(
                module.decoder,
                config,
                inputs_embeds=True,
            ),
            "vision_encoder": self._build_vision(module.vision_encoder, config),
            "embedding": self._build_embedding(module.embedding, config),
        }
        return ModelPackage(models, config=config)

    @staticmethod
    def _build_vision(
        vision_encoder: nn.Module,
        config: Glm5NextConfig,
    ) -> ir.Model:
        vision = config.vision
        if vision is None or vision.patch_size is None:
            raise ValueError("GLM-5.3 requires a complete vision configuration")
        total_patches = ir.SymbolicDim("total_patches")
        num_media = ir.SymbolicDim("num_media")
        pixel_dim = (
            vision.in_channels
            * vision.temporal_patch_size
            * vision.patch_size
            * vision.patch_size
        )
        graph, builder = _make_graph("glm5_next_vision_encoder")
        pixel_values = builder.input(
            "pixel_values",
            dtype=ir.DataType.FLOAT,
            shape=[total_patches, pixel_dim],
        )
        grid_thw = builder.input(
            "grid_thw",
            dtype=ir.DataType.INT64,
            shape=[num_media, 3],
        )
        image_features = vision_encoder(builder.op, pixel_values, grid_thw)
        builder.add_output(image_features, "image_features")
        model = _make_model(graph)
        model.metadata_props["mobius.processor_boundary"] = (
            "pixel_values=float32[N,3*2*14*14];"
            "grid_thw=int64[media,3];images-and-videos-share-encoder"
        )
        return model

    @staticmethod
    def _build_embedding(
        embedding: nn.Module,
        config: Glm5NextConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        sequence_length = ir.SymbolicDim("sequence_length")
        num_image_tokens = ir.SymbolicDim("num_image_tokens")
        num_video_tokens = ir.SymbolicDim("num_video_tokens")
        graph, builder = _make_graph("glm5_next_embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, sequence_length],
        )
        image_features = builder.input(
            "image_features",
            dtype=(
                ir.DataType.FLOAT if config.dtype == ir.DataType.BFLOAT16 else config.dtype
            ),
            shape=[num_image_tokens, config.hidden_size],
        )
        video_features = builder.input(
            "video_features",
            dtype=(
                ir.DataType.FLOAT if config.dtype == ir.DataType.BFLOAT16 else config.dtype
            ),
            shape=[num_video_tokens, config.hidden_size],
        )
        inputs_embeds = embedding(
            builder.op,
            input_ids,
            image_features,
            video_features,
        )
        builder.add_output(inputs_embeds, "inputs_embeds")
        model = _make_model(graph)
        model.metadata_props["mobius.multimedia_order"] = (
            "image=<|image|>-outside-video-span;"
            "video=<|image|>-inside-<|begin_of_video|>/<|end_of_video|>"
        )
        return model
