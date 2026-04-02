# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""SAM segmentation task — builds vision_encoder and decoder ONNX models.

The task produces two ONNX graphs:

- **vision_encoder**: ``pixel_values`` (B, 3, H, W)
  → ``image_embeddings`` (B, C, H/p, W/p)
- **decoder**: ``image_embeddings`` + ``input_points`` + ``input_labels``
  → ``pred_masks`` + ``iou_predictions``
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, SamConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)


class SamSegmentationTask(ModelTask):
    """Build two ONNX models for the Segment Anything pipeline.

    The module must expose:
    - ``vision_encoder`` (:class:`SamVisionEncoder`)
    - ``prompt_encoder`` (:class:`_SamPromptEncoder`)
    - ``mask_decoder`` (:class:`SamMaskDecoder`)
    - ``shared_image_embedding`` (:class:`_SamPositionalEmbedding`)
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        if not isinstance(config, SamConfig):
            raise TypeError(
                f"SamSegmentationTask requires SamConfig, got {type(config).__name__}"
            )

        vision_model = self._build_vision_encoder(module.vision_encoder, config)
        decoder_model = self._build_decoder(module, config)
        return ModelPackage(
            {"vision_encoder": vision_model, "decoder": decoder_model},
            config=config,
        )

    # ── Vision encoder ────────────────────────────────────────────────

    def _build_vision_encoder(
        self,
        vision_encoder: nn.Module,
        config: SamConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, 3, config.image_size, config.image_size]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        graph, builder = _make_graph([pixel_values], name="vision_encoder")
        op = builder.op

        image_embeddings = vision_encoder(op, pixel_values)
        image_embeddings.name = "image_embeddings"
        graph.outputs.append(image_embeddings)

        return _make_model(graph)

    # ── Decoder (prompt encoder + mask decoder) ───────────────────────

    def _build_decoder(
        self,
        module: nn.Module,
        config: SamConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        num_points = ir.SymbolicDim("num_points")
        h_emb = config.image_embedding_size
        w_emb = config.image_embedding_size

        image_embeddings = ir.Value(
            name="image_embeddings",
            shape=ir.Shape([batch, config.output_channels, h_emb, w_emb]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        input_points = ir.Value(
            name="input_points",
            shape=ir.Shape([batch, num_points, 2]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        input_labels = ir.Value(
            name="input_labels",
            shape=ir.Shape([batch, num_points]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph(
            [image_embeddings, input_points, input_labels],
            name="decoder",
        )
        op = builder.op

        # ── Image positional encoding ──
        image_pe = module.shared_image_embedding.get_image_pe(op, h_emb, w_emb)

        # ── Prompt encoding ──
        sparse_embeddings, dense_embeddings = module.prompt_encoder(
            op,
            input_points,
            input_labels,
            module.shared_image_embedding,
        )

        # ── Mask decoding ──
        pred_masks, iou_predictions = module.mask_decoder(
            op,
            image_embeddings,
            image_pe,
            sparse_embeddings,
            dense_embeddings,
        )

        pred_masks.name = "pred_masks"
        iou_predictions.name = "iou_predictions"
        graph.outputs.extend([pred_masks, iou_predictions])

        return _make_model(graph)
