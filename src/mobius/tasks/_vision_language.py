# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision-language model task."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._cache_utils import (
    _make_kv_cache_inputs,
    _register_kv_cache_outputs,
)


class Qwen3VLVisionLanguageTask(ModelTask):
    """Qwen3-VL vision-language task with packed vision inputs and MRoPE.

    Inputs:
        - input_ids: [batch, sequence_len] INT64
        - attention_mask: [batch, total_seq_len] INT64
        - position_ids: [3, batch, sequence_len] INT64 (MRoPE: T, H, W)
        - pixel_values: [total_patches, pixel_dim] FLOAT
        - grid_thw: [num_images, 3] INT64 (T, H, W per image)
        - past_key_values.{i}.key/value

    Outputs:
        - logits: FLOAT
        - present.{i}.key/value: FLOAT
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")
        total_patches = ir.SymbolicDim("total_patches")

        graph, builder = _make_graph()
        op = builder.op

        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_seq_len + seq_len"],
        )
        # MRoPE: 3D position IDs (temporal, height, width)
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[3, batch, seq_len],
        )
        # Flattened image patches
        patch_size = config.vision.patch_size or 16 if config.vision else 16
        temporal_patch_size = config.temporal_patch_size
        in_channels = config.vision.in_channels if config.vision else 3
        pixel_dim = in_channels * temporal_patch_size * patch_size * patch_size
        pixel_values = builder.input(
            "pixel_values",
            dtype=ir.DataType.FLOAT,
            shape=[total_patches, pixel_dim],
        )
        # Vision input is always f32 (matching image processor output).
        # Cast at graph entry for f16/bf16 builds.
        if config.dtype and config.dtype != ir.DataType.FLOAT:
            pixel_values = op.Cast(pixel_values, to=config.dtype)
        # Image grid dimensions for position embedding interpolation
        num_images = ir.SymbolicDim("num_images")
        grid_thw = builder.input(
            "grid_thw",
            dtype=ir.DataType.INT64,
            shape=[num_images, 3],
        )

        past_key_values = _make_kv_cache_inputs(
            builder,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )

        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            past_key_values=past_key_values,
        )

        builder.add_output(logits, "logits")
        _register_kv_cache_outputs(builder, present_key_values)

        return ModelPackage({"model": _make_model(graph)}, config=config)
