# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for Drama 3's hybrid backbone and temporal codebook tower."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from typing import Any

import onnx_ir as ir

from mobius._configs import ArchitectureConfig, _as_attribute_config

DRAMA3_REVISION = "b2b5ee998034b2975720906c815e7fb0e9f007bd"
DRAMA3_SOURCE_REVISION = "541566932abbaebcb0a40e562a1f846738383f38"


@dataclasses.dataclass
class FishOmniConfig(ArchitectureConfig):
    """Drama 3 token generation; the separately distributed codec is external."""

    tower: ArchitectureConfig | None = None
    num_codebooks: int = 10
    shared_enrichment_layers: int = 4
    vq_backbone_codebooks: int = 1
    semantic_start_token_id: int = 248099
    semantic_end_token_id: int = 252194
    fp8_experts: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> FishOmniConfig:
        # HF config classes and raw JSON expose architecture-specific dynamic fields.
        parent: Any = _as_attribute_config(parent_config or config)
        text: Any = _as_attribute_config(getattr(parent, "text_config", config))
        audio: Any = _as_attribute_config(parent.audio_decoder_config)
        if parent.model_type != "fish_omni":
            raise ValueError("FishOmniConfig requires model_type='fish_omni'")
        if text.model_type not in {"fish_omni_text", "qwen3_5_moe_text"}:
            raise ValueError("Unsupported Drama 3 text_config.model_type")
        if getattr(parent, "audio_encoder_config", None) is not None:
            raise ValueError("Drama 3 audio_encoder_config is not supported")
        if getattr(parent, "speech_template_version", None) != "reference-bracket-v1":
            raise ValueError("Drama 3 requires speech_template_version='reference-bracket-v1'")
        text_flags = {
            "rotary_interleaved": False,
            "use_qk_norm": True,
            "attn_output_gate": True,
            "layernorm_zero_centered_gamma": True,
            "shared_expert_gate": True,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "hidden_act": "silu",
        }
        audio_flags = {
            "decoder_type": "codebook_mtp_shared",
            "rotary_interleaved": True,
            "attention_output_gate": True,
            "layernorm_zero_centered_gamma": True,
            "use_moe": False,
            "mtp_use_swa": False,
            "attention_qkv_bias": False,
            "attention_o_bias": False,
            "dropout": 0.0,
        }
        for source, flags in ((text, text_flags), (audio, audio_flags)):
            for name, required in flags.items():
                if getattr(source, name, required) != required:
                    raise ValueError(f"Unsupported Drama 3 {name}; expected {required!r}")
        if any(kind != "moe" for kind in getattr(text, "mlp_layer_types", None) or []):
            raise ValueError("Drama 3 requires MoE in every backbone layer")
        text = copy.copy(text)
        rope = text.rope_parameters
        if not isinstance(rope, Mapping):
            rope = vars(rope)
        text.rope_parameters = dict(rope)
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Drama 3 requires default RoPE")
        base = ArchitectureConfig.from_transformers(text)
        if len(base.layer_types or []) != base.num_hidden_layers or not set(
            base.layer_types or []
        ) <= {"linear_attention", "full_attention"}:
            raise ValueError("Drama 3 requires one valid attention type per backbone layer")
        if (
            base.num_experts_per_tok is None
            or base.num_local_experts is None
            or not 0 < base.num_experts_per_tok <= base.num_local_experts
        ):
            raise ValueError("Invalid Drama 3 expert counts")
        dim = audio.dim
        n_shared = audio.shared_enrichment_layers
        n_codebooks = audio.num_codebooks
        n_backbone = audio.vq_backbone_codebooks
        vocab = audio.vocab_size
        if dim != base.hidden_size or getattr(audio, "text_dim", dim) != dim:
            raise ValueError("Drama 3 tower and backbone hidden dimensions must match")
        if audio.n_layer != n_shared or n_shared < 1:
            raise ValueError("Drama 3 n_layer must equal positive shared_enrichment_layers")
        if not 0 <= n_backbone < n_codebooks or n_codebooks < 2:
            raise ValueError("Invalid Drama 3 codebook counts")
        placement = getattr(audio, "vq_feature_placement", None)
        if placement not in {"backbone_partial", "mtp_shared_pre"} or (
            placement == "mtp_shared_pre" and n_backbone != 0
        ):
            raise ValueError("Unsupported Drama 3 vq_feature_placement")
        start = parent.semantic_start_token_id
        end = parent.semantic_end_token_id
        eos = getattr(parent, "eos_token_id", None)
        if (
            not 0 <= start <= end < base.vocab_size
            or vocab != end - start + 1
            or eos is None
            or not 0 <= eos < base.vocab_size
            or start <= eos <= end
        ):
            raise ValueError("Invalid Drama 3 semantic band or EOS token")
        quant = getattr(parent, "quantization_config", None)
        if quant is not None and not isinstance(quant, Mapping):
            quant = vars(quant)
        if quant is not None and (
            quant.get("quant_method") != "bsgemm-moe"
            or quant.get("weight_block_size") != [128, 128]
            or quant.get("activation_scheme") != "dynamic"
        ):
            raise ValueError("Drama 3 expects dynamic bsgemm-moe FP8 with 128x128 blocks")
        tower = ArchitectureConfig(
            hidden_size=dim,
            intermediate_size=audio.intermediate_size,
            num_attention_heads=audio.n_head,
            num_key_value_heads=audio.n_local_heads,
            head_dim=audio.head_dim,
            num_hidden_layers=n_shared + n_codebooks - 1,
            # The tower attends over time, not over the 11 codebook positions
            # named by the stale max_seq_len field in the released config.
            max_position_embeddings=base.max_position_embeddings,
            vocab_size=vocab,
            rms_norm_eps=audio.norm_eps,
            rope_type="default",
            rope_theta=audio.rope_base,
            partial_rotary_factor=audio.rotary_percent,
            rope_interleave=True,
            hidden_act="silu",
        )
        for heads, kv_heads, head_dim, fraction in (
            (
                base.num_attention_heads,
                base.num_key_value_heads,
                base.head_dim,
                base.partial_rotary_factor,
            ),
            (
                tower.num_attention_heads,
                tower.num_key_value_heads,
                tower.head_dim,
                tower.partial_rotary_factor,
            ),
        ):
            if fraction is None:
                raise ValueError("Drama 3 requires a partial rotary factor")
            width = int(head_dim * fraction)
            if kv_heads < 1 or heads % kv_heads or not 0 < width <= head_dim or width % 2:
                raise ValueError("Invalid Drama 3 attention or rotary geometry")
        fields = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
        fields.update(
            model_type="fish_omni",
            hidden_act="silu",
            attn_qk_norm=True,
            attn_qkv_bias=False,
            attn_o_bias=False,
            mamba_ssm_dtype=ir.DataType.FLOAT,
            norm_topk_prob=getattr(text, "norm_topk_prob", True),
            eos_token_id=eos,
            # bsgemm-moe is block-FP8, not the generic affine INT4 format.
            quantization=None,
            component_quantization={},
        )
        return cls(
            **fields,
            tower=tower,
            num_codebooks=n_codebooks,
            shared_enrichment_layers=n_shared,
            vq_backbone_codebooks=n_backbone,
            semantic_start_token_id=start,
            semantic_end_token_id=end,
            fp8_experts=quant is not None,
        )

    def validate(self) -> None:
        """Reject unsafe precision instead of overflowing the released checkpoint."""
        super().validate()
        if self.quantization is not None or self.component_quantization:
            raise ValueError(
                "Drama 3 affine component quantization is unsupported; "
                "use the FP32 graph with explicit FP8 checkpoint dequantization."
            )
        if self.dtype != ir.DataType.FLOAT:
            raise ValueError(
                "Drama 3 currently requires dtype='f32'. FP16 residuals/down projections "
                "overflow; BF16 has not passed the upstream numerical parity gate."
            )
        if self.tower is None:
            raise ValueError("Drama 3 requires an audio tower configuration")
        self.tower.validate()
