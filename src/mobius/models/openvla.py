from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import OpenVLAConfig
from mobius.components import (
    DinoV2PatchFeatures,
    Embedding,
    Linear,
    SiglipPatchFeatures,
    TorchDefaultRope,
)
from mobius.models.base import TextModel


class _OpenVLADecoder(nn.Module):
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.model = TextModel(config)
        self.model.rotary_emb = TorchDefaultRope(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden, cache = self.model(
            op,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return self.lm_head(op, hidden), cache


class _OpenVLAEmbedding(nn.Module):
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)

    def forward(self, op: OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        text = self.embed_tokens(op, input_ids)
        # Insert patches after BOS; an empty feature sequence is ordinary decode.
        return op.Concat(
            op.Slice(text, [0], [1], [1]),
            image_features,
            op.Slice(text, [1], [9223372036854775807], [1]),
            axis=1,
        )


class _FusedVision(nn.Module):
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        assert config.dino.num_hidden_layers is not None
        assert config.siglip.num_hidden_layers is not None
        self.featurizer = DinoV2PatchFeatures(
            config.dino,
            num_layers=config.dino.num_hidden_layers - 1,
        )
        self.fused_featurizer = SiglipPatchFeatures(
            config.siglip,
            num_layers=config.siglip.num_hidden_layers - 1,
        )

    def forward(self, op: OpBuilder, pixels: ir.Value):
        dino = self.featurizer(op, op.Slice(pixels, [0], [3], [1]))
        siglip = self.fused_featurizer(op, op.Slice(pixels, [3], [6], [1]))
        return op.Concat(dino, siglip, axis=-1)


class _Projector(nn.Module):
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        assert config.dino.hidden_size is not None and config.siglip.hidden_size is not None
        width = config.dino.hidden_size + config.siglip.hidden_size
        self.fc1 = Linear(width, 4 * width)
        self.fc2 = Linear(4 * width, config.hidden_size)
        self.fc3 = Linear(config.hidden_size, config.hidden_size)

    def forward(self, op: OpBuilder, features: ir.Value):
        hidden = op.Gelu(self.fc1(op, features), approximate="none")
        hidden = op.Gelu(self.fc2(op, hidden), approximate="none")
        return self.fc3(op, hidden)


class _OpenVLAVision(nn.Module):
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.vision_backbone = _FusedVision(config)
        self.projector = _Projector(config)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        return self.projector(op, self.vision_backbone(op, pixel_values))


class OpenVLAModel(nn.Module):
    """Complete OpenVLA policy graph split with explicit external generation state."""

    default_task = "openvla"
    category = "Multimodal"
    config_class = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.config = config
        self.decoder = _OpenVLADecoder(config)
        self.vision_encoder = _OpenVLAVision(config)
        self.embedding = _OpenVLAEmbedding(config)
        self._omitted_shapes = {}
        for name, tower_config in (
            ("featurizer", config.dino),
            ("fused_featurizer", config.siglip),
        ):
            assert tower_config.num_hidden_layers is not None
            assert tower_config.hidden_size is not None
            prefix = f"vision_backbone.{name}."
            tower = getattr(self.vision_encoder.vision_backbone, name)
            for parameter_name, parameter in tower.blocks[0].named_parameters():
                assert parameter.shape is not None
                self._omitted_shapes[
                    f"{prefix}blocks.{tower_config.num_hidden_layers - 1}.{parameter_name}"
                ] = tuple(parameter.shape)
            for parameter_name in ("weight", "bias"):
                self._omitted_shapes[f"{prefix}norm.{parameter_name}"] = (
                    tower_config.hidden_size,
                )
        width = config.siglip.hidden_size
        intermediate = config.siglip.intermediate_size
        assert width is not None and intermediate is not None
        pool_shapes = {
            "latent": (1, 1, width),
            "q.weight": (width, width),
            "q.bias": (width,),
            "kv.weight": (2 * width, width),
            "kv.bias": (2 * width,),
            "proj.weight": (width, width),
            "proj.bias": (width,),
            "norm.weight": (width,),
            "norm.bias": (width,),
            "mlp.fc1.weight": (intermediate, width),
            "mlp.fc1.bias": (intermediate,),
            "mlp.fc2.weight": (width, intermediate),
            "mlp.fc2.bias": (width,),
        }
        self._omitted_shapes.update(
            {
                f"vision_backbone.fused_featurizer.attn_pool.{name}": shape
                for name, shape in pool_shapes.items()
            }
        )

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "OpenVLATask builds decoder, vision and embedding separately"
        )

    def omitted_weight_reason(self, name: str) -> str | None:
        """Account explicitly for checkpoint parameters unused by source features."""
        if name not in self._omitted_shapes:
            return None
        if ".blocks." in name:
            return "final block cannot affect the selected second-to-last intermediate"
        return "source get_intermediate_layers(norm=False) bypasses final normalization/head"

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        result = {}
        parameters = dict(self.named_parameters())
        for name, value in state_dict.items():
            if self.omitted_weight_reason(name) is not None:
                if tuple(value.shape) != self._omitted_shapes[name]:
                    raise ValueError(f"OpenVLA omitted checkpoint shape mismatch: {name}")
                continue
            if name == "language_model.model.embed_tokens.weight":
                result["embedding.embed_tokens.weight"] = value
            elif name.startswith("language_model."):
                result["decoder." + name.removeprefix("language_model.")] = value
            elif name.startswith(("vision_backbone.", "projector.")):
                result["vision_encoder." + name] = value
            else:
                raise ValueError(f"Unaccounted OpenVLA checkpoint tensor: {name}")
        for name, value in result.items():
            if name not in parameters:
                raise ValueError(f"Unknown OpenVLA parameter: {name}")
            expected = parameters[name].shape
            if expected is None or tuple(value.shape) != tuple(expected):
                raise ValueError(f"OpenVLA checkpoint shape mismatch: {name}")
        return result
