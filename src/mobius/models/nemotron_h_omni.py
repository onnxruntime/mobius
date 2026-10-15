# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""C-RADIOv4, spatial projector and hybrid decoder for Nemotron-H Omni.

Replicates ``NemotronHOmniModel`` using a split decoder, flat-feature embedding
and image/video encoder. Video calls process one clip and expose EVS retention
metadata; the caller must compact the corresponding prompt before decoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._component_quantization import _replace_child, _resolve_module
from mobius._configs import NemotronHOmniConfig
from mobius.components import (
    Embedding,
    FlatInputMixer,
    GGMLIQ2XXSLinear,
    GGMLQ8Linear,
    Linear,
    NVFP4A16Linear,
    RadioV4VisionModel,
    realize_parameters,
)
from mobius.models._nemotron_h_omni_vision import (
    NemotronHOmniVisionProjector,
    build_omni_image_features,
    build_omni_video_features,
)
from mobius.models.nemotron_h import (
    NemotronHPuzzleCausalLMModel,
    _rename_nemotron_h_weight,
)

if TYPE_CHECKING:
    import torch

BOLT_REVISION = "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a"


def _install_native_projections(decoder: nn.Module, config: NemotronHOmniConfig) -> None:
    for source_name, specification in config.quantized_projections.items():
        path = _rename_nemotron_h_weight(
            source_name.removeprefix("language_model."), config.layer_types or []
        )
        original = _resolve_module(decoder, path)
        if not isinstance(original, Linear) or original.weight.shape is None:
            raise TypeError(
                f"Quantized source does not identify a dense projection: {source_name}"
            )
        out_features, in_features = original.weight.shape
        if not isinstance(out_features, int) or not isinstance(in_features, int):
            raise TypeError(f"Quantized projection requires fixed dimensions: {source_name}")
        bias = original.bias is not None
        replacement: nn.Module
        if specification.quant_algo == "IQ2_XXS":
            replacement = GGMLIQ2XXSLinear(in_features, out_features, bias=bias)
        elif specification.quant_algo == "Q8_0":
            replacement = GGMLQ8Linear(in_features, out_features, bias=bias)
        elif specification.quant_algo == "W4A16_NVFP4":
            if config.dtype not in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
                raise ValueError("Native Omni NVFP4 requires FLOAT16 or BFLOAT16 activations")
            replacement = NVFP4A16Linear(in_features, out_features, bias=bias)
        else:
            raise ValueError(f"Unsupported native projection: {specification.quant_algo}")
        _replace_child(decoder, path, replacement)


class _OmniDecoder(NemotronHPuzzleCausalLMModel):
    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present = self.model(
            op,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return self.lm_head(op, hidden_states), present


class _OmniEmbedding(nn.Module):
    def __init__(self, config: NemotronHOmniConfig):
        super().__init__()
        if config.image_token_id is None:
            raise ValueError("Omni embedding requires an image placeholder ID")
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.input_mixer = FlatInputMixer(config.image_token_id)

    def forward(self, op: OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        return self.input_mixer(
            op, self.embed_tokens(op, input_ids), image_features, input_ids
        )


class _OmniVisionEncoder(nn.Module):
    def __init__(self, config: NemotronHOmniConfig):
        super().__init__()
        vision = config.vision
        if vision is None:
            raise ValueError("Omni requires a vision configuration")
        if (
            vision.patch_size is None
            or vision.hidden_size is None
            or vision.intermediate_size is None
            or vision.num_hidden_layers is None
            or vision.num_attention_heads is None
        ):
            raise ValueError(
                "Omni requires explicit RADIO patch size and architecture dimensions"
            )
        self.config = config
        self.vision_model = RadioV4VisionModel(
            patch_size=vision.patch_size,
            max_grid_size=config.radio_max_img_size // vision.patch_size,
            hidden_size=vision.hidden_size,
            intermediate_size=vision.intermediate_size,
            num_layers=vision.num_hidden_layers,
            num_heads=vision.num_attention_heads,
            norm_eps=vision.norm_eps,
            video_temporal_patch_size=config.video_temporal_patch_size,
            num_cls_tokens=config.radio_num_cls_tokens,
            num_registers=config.radio_num_registers,
            summary_idxs=config.radio_summary_idxs,
        )
        self.projector = NemotronHOmniVisionProjector(
            vision.hidden_size,
            config.projector_hidden_size,
            config.hidden_size,
            norm_eps=config.projector_norm_eps,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value, is_video: ir.Value):
        config = self.config
        if config.vision is None:
            raise ValueError("Omni requires a vision configuration")
        patch_size = config.vision.patch_size
        if patch_size is None:
            raise ValueError("Omni requires an explicit RADIO patch size")
        # Both branches capture the same root initializers. Never qualify shared
        # RADIO/projector weights under whichever branch happened to build first.
        realize_parameters(op.builder, self, "vision_encoder")
        pixels = op.Cast(pixel_values, to=config.dtype)

        def image_branch(branch_op: OpBuilder):
            dense = build_omni_image_features(
                branch_op,
                pixels,
                vision_model=self.vision_model,
                projector=self.projector,
                patch_size=patch_size,
            )
            features = branch_op.Reshape(dense, [-1, config.hidden_size])
            count = branch_op.Squeeze(branch_op.Shape(features, start=0, end=1), [0])
            indices = branch_op.Range(0, count, 1)
            return features, branch_op.Equal(indices, indices), indices

        def video_branch(branch_op: OpBuilder):
            return build_omni_video_features(
                branch_op,
                pixels,
                vision_model=self.vision_model,
                projector=self.projector,
                patch_size=patch_size,
                temporal_patch_size=config.video_temporal_patch_size,
                pruning_rate=config.video_pruning_rate,
            )

        branches = []
        for name, callback in (("video", video_branch), ("image", image_branch)):
            op.builder.push_module(name)
            try:
                branches.append(
                    op.builder.subgraph(
                        callback,
                        inputs=[],
                        outputs=[
                            ir.Value(name=f"{name}_features"),
                            ir.Value(name=f"{name}_retention"),
                            ir.Value(name=f"{name}_indices"),
                        ],
                        name=f"omni_{name}",
                    )
                )
            finally:
                op.builder.pop_module()
        features, retention, indices = op.If(
            is_video, then_branch=branches[0], else_branch=branches[1], _outputs=3
        )
        for value, dtype, shape in (
            (features, config.dtype, ir.Shape(["num_media_tokens", config.hidden_size])),
            (retention, ir.DataType.BOOL, ir.Shape(["dense_media_tokens"])),
            (indices, ir.DataType.INT64, ir.Shape(["num_media_tokens"])),
        ):
            value.dtype = dtype
            value.shape = shape
        return features, retention, indices


class NemotronHOmniModel(nn.Module):
    """Native mixed-format Nemotron-H Omni image/video and Puzzle decoder.

    Uses the source's heterogeneous experts and FP32 grouped gated norm.
    IQ2_XXS requires the dedicated ORT CUDA contrib kernel, not dense expansion.
    Video EVS requires host prompt compaction and one clip per encoder call.
    """

    config_class = NemotronHOmniConfig
    default_task = "nemotron-h-omni"
    category = "Multimodal"

    def __init__(self, config: NemotronHOmniConfig):
        super().__init__()
        if config.tie_word_embeddings:
            raise NotImplementedError("Omni split tied-embedding checkpoints are unsupported")
        self.config = config
        self.decoder = _OmniDecoder(config)
        _install_native_projections(self.decoder, config)
        self.embedding = _OmniEmbedding(config)
        self.vision_encoder = _OmniVisionEncoder(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        raise ValueError(
            "Nemotron-H Omni requires its source-accounted streaming weight loader; "
            "generic eager loading must not expand or reinterpret native packed weights"
        )
