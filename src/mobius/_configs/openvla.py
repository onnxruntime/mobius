from __future__ import annotations

import dataclasses

from mobius._configs._base import ArchitectureConfig
from mobius._configs._sub_configs import VisionConfig


@dataclasses.dataclass
class OpenVLAConfig(ArchitectureConfig):
    """OpenVLA's Llama-2 decoder and fixed DINOv2/SigLIP patch-feature towers."""

    dino: VisionConfig = dataclasses.field(
        default_factory=lambda: VisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            norm_eps=1e-6,
        )
    )
    siglip: VisionConfig = dataclasses.field(
        default_factory=lambda: VisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            norm_eps=1e-6,
        )
    )

    @classmethod
    def from_transformers(cls, config, parent_config=None):
        from transformers import LlamaConfig

        source = parent_config or config
        if getattr(source, "use_fused_vision_backbone", None) is not True:
            raise ValueError("OpenVLA requires its fused DINOv2/SigLIP backbone")
        if list(getattr(source, "timm_model_ids", ())) != [
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ] or list(getattr(source, "image_sizes", ())) != [224, 224]:
            raise ValueError("Unsupported OpenVLA vision backbone or image geometry")
        text = getattr(source, "text_config", config)
        values = (
            dict(text)
            if isinstance(text, dict)
            else (text.to_dict() if hasattr(text, "to_dict") else vars(text).copy())
        )
        declared_type = (
            text.get("model_type")
            if isinstance(text, dict)
            else getattr(text, "model_type", None)
        )
        if declared_type != "llama":
            raise ValueError("OpenVLA currently requires a Llama decoder")
        values.pop("model_type", None)
        # The checkpoint omits Llama-2 defaults such as hidden_size=4096.
        base = ArchitectureConfig.from_transformers(LlamaConfig(**values))
        fields = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
        settings = base.rope_scaling or {}
        if (
            base.rope_type != "default"
            or set(settings) - {"rope_type", "rope_theta"}
            or settings.get("rope_type", "default") != "default"
            or settings.get("rope_theta", base.rope_theta) != base.rope_theta
        ):
            raise ValueError("OpenVLA requires its source unscaled default RoPE")
        fields["rope_scaling"] = None
        fields["model_type"] = "openvla"
        return cls(**fields)

    def validate(self) -> None:
        super().validate()
        if self.rope_type != "default" or self.rope_scaling:
            raise ValueError("OpenVLA requires its source unscaled default RoPE")
        if self.quantization is not None or self.component_quantization is not None:
            raise ValueError(
                "OpenVLA quantization requires a separately validated component policy"
            )
        for tower in (self.dino, self.siglip):
            dimensions = (
                tower.hidden_size,
                tower.intermediate_size,
                tower.num_attention_heads,
                tower.num_hidden_layers,
                tower.image_size,
                tower.patch_size,
            )
            if any(value is None or value <= 0 for value in dimensions):
                raise ValueError("OpenVLA vision dimensions must be positive")
            assert tower.hidden_size is not None and tower.num_attention_heads is not None
            assert tower.image_size is not None and tower.patch_size is not None
            assert tower.num_hidden_layers is not None
            if (
                tower.hidden_size % tower.num_attention_heads
                or tower.image_size % tower.patch_size
                or tower.num_hidden_layers < 2
            ):
                raise ValueError("Invalid OpenVLA tower head, patch or feature-layer geometry")
        if (
            self.dino.image_size != self.siglip.image_size
            or self.dino.patch_size != self.siglip.patch_size
        ):
            raise ValueError("OpenVLA towers must produce matching spatial patch grids")
