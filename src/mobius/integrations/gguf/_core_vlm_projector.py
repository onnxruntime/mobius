# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Config, tensor mapping, and standalone builders for core GGUF VLM projectors."""

from __future__ import annotations

__all__ = [
    "CORE_VLM_PROJECTOR_TYPES",
    "build_core_vlm_projector_mmproj",
    "core_vlm_projector_fingerprint",
    "map_core_vlm_projector_tensor",
    "read_core_vlm_projector_config",
    "validate_core_vlm_projector_shapes",
]

import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from mobius._configs import ArchitectureConfig, Gemma3nMultiModalConfig, Gemma4Config

CORE_VLM_PROJECTOR_TYPES = frozenset(
    {
        "gemma3na",
        "gemma3nv",
        "gemma4a",
        "gemma4ua",
        "gemma4uv",
        "idefics3",
        "internvl",
        "llama4",
        "pixtral",
    }
)

_VISION_BLOCK = re.compile(r"^v\.blk\.(\d+)\.(.+)$")
_MOBILENET_BLOCK = re.compile(r"^v\.blk\.(\d+)\.(\d+)\.(.+)$")
_AUDIO_BLOCK = re.compile(r"^a\.blk\.(\d+)\.(.+)$")
_CLIP_BOUND_SUFFIXES = (".input_min", ".input_max", ".output_min", ".output_max")
_PIXTRAL_BLOCK_SUFFIXES = frozenset(
    {
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_out.weight",
        "ln1.weight",
        "ln2.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    }
)


@dataclasses.dataclass
class _ProjectorConfig(ArchitectureConfig):
    def validate(self) -> None:
        if self.hidden_size <= 0 or self.vision is None:
            raise ValueError("Vision projector config requires positive output width and vision data.")


@dataclasses.dataclass
class _Gemma3nProjectorConfig(Gemma3nMultiModalConfig):
    def validate(self) -> None:
        if self.hidden_size <= 0 or (self.vision is None and self.audio is None):
            raise ValueError("Gemma3n projector config requires one explicit encoder role.")


@dataclasses.dataclass
class _Gemma4ProjectorConfig(Gemma4Config):
    def validate(self) -> None:
        if self.hidden_size <= 0 or (self.vision is None and self.audio is None):
            raise ValueError("Gemma4 projector config requires one explicit encoder role.")


def _positive_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}.")
    return int(value)


def _vision_config(mmproj_gguf: Any, *, projector_type: str):
    from mobius._configs import VisionConfig

    md = mmproj_gguf.metadata
    image_size = _positive_int(md, "clip.vision.image_size")
    patch_size = _positive_int(md, "clip.vision.patch_size")
    hidden = _positive_int(md, "clip.vision.embedding_length")
    intermediate = _positive_int(md, "clip.vision.feed_forward_length")
    layers = _positive_int(md, "clip.vision.block_count")
    heads = _positive_int(md, "clip.vision.attention.head_count")
    if hidden % heads:
        raise ValueError(f"{projector_type} vision hidden size must divide by its head count.")
    if image_size % patch_size:
        raise ValueError(f"{projector_type} image size must divide by its patch size.")
    epsilon = float(md["clip.vision.attention.layer_norm_epsilon"])
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("clip.vision.attention.layer_norm_epsilon must be positive.")

    merge = int(md.get("clip.vision.projector.scale_factor", 0))
    if projector_type == "pixtral":
        merge = int(md.get("clip.vision.spatial_merge_size", 1))
    if projector_type != "pixtral" and merge <= 1:
        raise ValueError(
            f"{projector_type} requires clip.vision.projector.scale_factor greater than one."
        )
    grid = image_size // patch_size
    if grid % merge:
        raise ValueError(
            f"{projector_type} {grid}x{grid} patch grid is not divisible by merge {merge}."
        )

    projector_intermediate = None
    if projector_type == "llama4":
        projector_intermediate = int(
            mmproj_gguf.get_tensor_shape("mm.model.mlp.1.weight")[0]
        )

    return VisionConfig(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        image_size=image_size,
        patch_size=patch_size,
        norm_eps=epsilon,
        mm_tokens_per_image=(grid // merge) ** 2,
        model_type="pixtral" if projector_type == "pixtral" else projector_type,
        head_dim=hidden // heads,
        rope_theta=10_000.0 if projector_type in {"llama4", "pixtral"} else None,
        out_hidden_size=_positive_int(md, "clip.vision.projection_dim"),
        spatial_merge_size=merge,
        temporal_patch_size=1,
        hidden_act=(
            "silu"
            if projector_type == "pixtral"
            else "gelu"
            if projector_type in {"internvl", "llama4"}
            else "gelu_pytorch_tanh"
        ),
        projector_intermediate_size=projector_intermediate,
    )


def _gemma3n_vision_config(mmproj_gguf: Any):
    from mobius._configs import VisionConfig

    md = mmproj_gguf.metadata
    if int(md["clip.vision.image_size"]) != 768 or int(md["clip.vision.patch_size"]) != 3:
        raise ValueError("gemma3nv requires the fixed 768px MobileNetV5 / 3px sentinel ABI.")
    if int(md.get("clip.vision.projector.scale_factor", 1)) != 1:
        raise ValueError("gemma3nv projector scale factor must be one.")
    return VisionConfig(
        hidden_size=_positive_int(md, "clip.vision.embedding_length"),
        intermediate_size=_positive_int(md, "clip.vision.feed_forward_length"),
        num_hidden_layers=_positive_int(md, "clip.vision.block_count"),
        num_attention_heads=_positive_int(md, "clip.vision.attention.head_count"),
        image_size=768,
        patch_size=3,
        norm_eps=1e-6,
        rms_norm_eps=1e-6,
        mm_tokens_per_image=256,
        spatial_merge_size=1,
        temporal_patch_size=1,
        architecture="mobilenetv5_300m_enc",
        do_pooling=False,
        vocab_size=int(mmproj_gguf.get_tensor_shape("mm.embedding.weight")[0]),
    )


def _gemma3n_audio_config(mmproj_gguf: Any):
    from mobius._configs import Gemma3nAudioConfig

    md = mmproj_gguf.metadata
    return Gemma3nAudioConfig(
        hidden_size=_positive_int(md, "clip.audio.embedding_length"),
        conf_num_hidden_layers=_positive_int(md, "clip.audio.block_count"),
        conf_num_attention_heads=_positive_int(md, "clip.audio.attention.head_count"),
        conf_attention_chunk_size=12,
        conf_attention_context_left=13,
        conf_attention_context_right=0,
        conf_attention_logit_cap=50.0,
        conf_conv_kernel_size=5,
        conf_reduction_factor=4,
        conf_residual_weight=0.5,
        input_feat_size=_positive_int(md, "clip.audio.num_mel_bins"),
        sscp_conv_channel_size=[
            int(mmproj_gguf.get_tensor_shape("a.conv1d.0.weight")[0]),
            int(mmproj_gguf.get_tensor_shape("a.conv1d.1.weight")[0]),
        ],
        sscp_conv_kernel_size=[[3, 3], [3, 3]],
        sscp_conv_stride_size=[[2, 2], [2, 2]],
        sscp_conv_group_norm_eps=1e-3,
        gradient_clipping=1e10,
        rms_norm_eps=1e-6,
        vocab_size=int(mmproj_gguf.get_tensor_shape("mm.a.embedding.weight")[0]),
    )


def _gemma4_audio_config(mmproj_gguf: Any, *, unified: bool):
    from mobius._configs import Gemma4AudioConfig

    md = mmproj_gguf.metadata
    if unified:
        if any(
            int(md[key]) != 0
            for key in (
                "clip.audio.block_count",
                "clip.audio.feed_forward_length",
            )
        ):
            raise ValueError("gemma4ua is encoder-free and requires zero audio block/FFN counts.")
        if int(md["clip.audio.num_mel_bins"]) != 128:
            raise ValueError("gemma4ua expects the converter's 128-bin metadata sentinel.")
        return Gemma4AudioConfig(
            hidden_size=_positive_int(md, "clip.audio.embedding_length"),
            num_layers=0,
            output_proj_dims=_positive_int(md, "clip.audio.embedding_length"),
            rms_norm_eps=1e-6,
        )

    return Gemma4AudioConfig(
        input_size=_positive_int(md, "clip.audio.num_mel_bins"),
        hidden_size=_positive_int(md, "clip.audio.embedding_length"),
        num_layers=_positive_int(md, "clip.audio.block_count"),
        attention_heads=_positive_int(md, "clip.audio.attention.head_count"),
        linear_units=_positive_int(md, "clip.audio.feed_forward_length"),
        subsampling_conv_channels=[
            int(mmproj_gguf.get_tensor_shape("a.conv1d.0.weight")[0]),
            int(mmproj_gguf.get_tensor_shape("a.conv1d.1.weight")[0]),
        ],
        output_proj_dims=int(mmproj_gguf.get_tensor_shape("a.pre_encode.out.weight")[0]),
        rms_norm_eps=1e-6,
    )


def _gemma4_unified_vision_config(mmproj_gguf: Any):
    from mobius._configs import VisionConfig

    md = mmproj_gguf.metadata
    if any(
        int(md[key]) != 0
        for key in (
            "clip.vision.block_count",
            "clip.vision.feed_forward_length",
        )
    ):
        raise ValueError("gemma4uv is encoder-free and requires zero vision block/FFN counts.")
    hidden = _positive_int(md, "clip.vision.embedding_length")
    patch_size = _positive_int(md, "clip.vision.patch_size")
    position_shape = mmproj_gguf.get_tensor_shape("v.position_embd.weight")
    if len(position_shape) != 3 or position_shape[0] != 2 or position_shape[2] != hidden:
        raise ValueError("gemma4uv position table must have shape [2, positions, hidden].")
    return VisionConfig(
        hidden_size=hidden,
        intermediate_size=0,
        num_hidden_layers=0,
        num_attention_heads=0,
        image_size=_positive_int(md, "clip.vision.image_size"),
        patch_size=patch_size,
        norm_eps=1e-6,
        pooling_kernel_size=3,
        position_embedding_size=int(position_shape[1]),
        out_hidden_size=hidden,
        spatial_merge_size=1,
        temporal_patch_size=1,
    )


def read_core_vlm_projector_config(mmproj_gguf: Any, projector_type: str):
    """Recover the revision-neutral graph configuration for one projector role."""
    md = mmproj_gguf.metadata
    if projector_type == "gemma3nv":
        return _Gemma3nProjectorConfig(
            model_type=projector_type,
            hidden_size=_positive_int(md, "clip.vision.projection_dim"),
            rms_norm_eps=1e-6,
            vision=_gemma3n_vision_config(mmproj_gguf),
            vision_soft_tokens_per_image=256,
        )
    if projector_type == "gemma3na":
        return _Gemma3nProjectorConfig(
            model_type=projector_type,
            hidden_size=_positive_int(md, "clip.audio.projection_dim"),
            rms_norm_eps=1e-6,
            audio=_gemma3n_audio_config(mmproj_gguf),
            audio_soft_tokens_per_image=188,
        )
    if projector_type == "gemma4a":
        return _Gemma4ProjectorConfig(
            model_type=projector_type,
            hidden_size=_positive_int(md, "clip.audio.projection_dim"),
            rms_norm_eps=1e-6,
            audio=_gemma4_audio_config(mmproj_gguf, unified=False),
        )
    if projector_type == "gemma4ua":
        return _Gemma4ProjectorConfig(
            model_type=projector_type,
            hidden_size=_positive_int(md, "clip.audio.projection_dim"),
            rms_norm_eps=1e-6,
            audio=_gemma4_audio_config(mmproj_gguf, unified=True),
        )
    if projector_type == "gemma4uv":
        return _Gemma4ProjectorConfig(
            model_type=projector_type,
            hidden_size=_positive_int(md, "clip.vision.projection_dim"),
            rms_norm_eps=1e-6,
            vision=_gemma4_unified_vision_config(mmproj_gguf),
        )
    if projector_type not in {"idefics3", "internvl", "llama4", "pixtral"}:
        raise ValueError(f"Unknown core GGUF projector type {projector_type!r}.")
    return _ProjectorConfig(
        model_type=projector_type,
        hidden_size=_positive_int(md, "clip.vision.projection_dim"),
        vision=_vision_config(mmproj_gguf, projector_type=projector_type),
    )


def core_vlm_projector_fingerprint(
    config: Any,
    projector_type: str,
    target_architecture: str | None = None,
) -> str:
    """Hash graph-affecting sidecar config with the projector discriminator."""
    if projector_type not in CORE_VLM_PROJECTOR_TYPES:
        raise ValueError(f"Unknown core GGUF projector type {projector_type!r}.")
    payload = json.dumps(
        {
            "config": dataclasses.asdict(config),
            "projector_type": projector_type,
            "target_architecture": target_architecture,
        },
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


_GEMMA3N_AUDIO_STEMS = {
    "ffn_norm.weight": "ffw_layer_start.pre_layer_norm.weight",
    "ffn_up.weight": "ffw_layer_start.ffw_layer_1.weight",
    "ffn_down.weight": "ffw_layer_start.ffw_layer_2.weight",
    "ffn_post_norm.weight": "ffw_layer_start.post_layer_norm.weight",
    "ln1.weight": "attention.pre_attn_norm.weight",
    "attn_q.weight": "attention.attn.q_proj.weight",
    "attn_k.weight": "attention.attn.k_proj.weight",
    "attn_v.weight": "attention.attn.v_proj.weight",
    "per_dim_scale": "attention.attn.per_dim_scale",
    "linear_pos.weight": "attention.attn.relative_position_embedding.pos_proj.weight",
    "attn_out.weight": "attention.post.weight",
    "ln2.weight": "attention.post_norm.weight",
    "conv_norm.weight": "lconv1d.pre_layer_norm.weight",
    "conv_pw1.weight": "lconv1d.linear_start.weight",
    "conv_dw.weight": "lconv1d.depthwise_conv1d.weight",
    "norm_conv.weight": "lconv1d.conv_norm.weight",
    "conv_pw2.weight": "lconv1d.linear_end.weight",
    "ffn_norm_1.weight": "ffw_layer_end.pre_layer_norm.weight",
    "ffn_up_1.weight": "ffw_layer_end.ffw_layer_1.weight",
    "ffn_down_1.weight": "ffw_layer_end.ffw_layer_2.weight",
    "ffn_post_norm_1.weight": "ffw_layer_end.post_layer_norm.weight",
    "layer_pre_norm.weight": "norm.weight",
}


def map_core_vlm_projector_tensor(name: str, projector_type: str) -> str | None:
    """Map a GGUF tensor name to its standalone role graph initializer name."""
    if projector_type == "gemma3nv":
        block = _MOBILENET_BLOCK.match(name)
        if block is not None:
            return (
                f"vision_encoder.encoder.blocks.{block.group(1)}."
                f"{block.group(2)}.{block.group(3)}"
            )
        top = {
            "v.conv_stem.conv.weight": "vision_encoder.encoder.conv_stem.conv.weight",
            "v.conv_stem.conv.bias": "vision_encoder.encoder.conv_stem.conv.bias",
            "v.conv_stem.bn.weight": "vision_encoder.encoder.conv_stem.bn.weight",
            "v.msfa.ffn.pw_exp.conv.weight": (
                "vision_encoder.encoder.msfa.ffn.pw_exp.conv.weight"
            ),
            "v.msfa.ffn.pw_exp.bn.weight": (
                "vision_encoder.encoder.msfa.ffn.pw_exp.bn.weight"
            ),
            "v.msfa.ffn.pw_proj.conv.weight": (
                "vision_encoder.encoder.msfa.ffn.pw_proj.conv.weight"
            ),
            "v.msfa.ffn.pw_proj.bn.weight": (
                "vision_encoder.encoder.msfa.ffn.pw_proj.bn.weight"
            ),
            "v.msfa.norm.weight": "vision_encoder.encoder.msfa.norm.weight",
            "mm.embedding.weight": "vision_encoder.embed_vision.embedding.weight",
            "mm.hard_emb_norm.weight": (
                "vision_encoder.embed_vision.hard_embedding_norm.weight"
            ),
            "mm.soft_emb_norm.weight": (
                "vision_encoder.embed_vision.soft_embedding_norm.weight"
            ),
            "mm.input_projection.weight": (
                "vision_encoder.embed_vision.embedding_projection.weight"
            ),
        }
        return top.get(name)

    if projector_type == "gemma3na":
        block = _AUDIO_BLOCK.match(name)
        if block is not None:
            suffix = _GEMMA3N_AUDIO_STEMS.get(block.group(2))
            return (
                None
                if suffix is None
                else f"audio_encoder.encoder.conformer.{block.group(1)}.{suffix}"
            )
        top = {
            "a.conv1d.0.weight": (
                "audio_encoder.encoder.subsample_conv_projection.conv_0.conv.weight"
            ),
            "a.conv1d.0.norm.weight": (
                "audio_encoder.encoder.subsample_conv_projection.conv_0.norm.weight"
            ),
            "a.conv1d.1.weight": (
                "audio_encoder.encoder.subsample_conv_projection.conv_1.conv.weight"
            ),
            "a.conv1d.1.norm.weight": (
                "audio_encoder.encoder.subsample_conv_projection.conv_1.norm.weight"
            ),
            "a.pre_encode.out.weight": (
                "audio_encoder.encoder.subsample_conv_projection.input_proj_linear.weight"
            ),
            "mm.a.embedding.weight": "audio_encoder.embed_audio.embedding.weight",
            "mm.a.hard_emb_norm.weight": (
                "audio_encoder.embed_audio.hard_embedding_norm.weight"
            ),
            "mm.a.soft_emb_norm.weight": (
                "audio_encoder.embed_audio.soft_embedding_norm.weight"
            ),
            "mm.a.input_projection.weight": (
                "audio_encoder.embed_audio.embedding_projection.weight"
            ),
        }
        return top.get(name)

    if projector_type == "gemma4a":
        from mobius.integrations.gguf._mmproj_mapping import map_mmproj_audio_to_hf

        hf_name = map_mmproj_audio_to_hf(name)
        if hf_name is None:
            return None
        if hf_name.startswith("audio_tower."):
            return "audio_encoder.encoder." + hf_name.removeprefix("audio_tower.")
        if hf_name.startswith("embed_audio.embedding_projection."):
            return "audio_encoder.projector." + hf_name.removeprefix(
                "embed_audio.embedding_projection."
            )
        return None

    if projector_type == "gemma4ua":
        return (
            "audio_encoder.projector.weight"
            if name == "mm.a.input_projection.weight"
            else None
        )

    if projector_type == "gemma4uv":
        return {
            "v.patch_norm.1.weight": "vision_encoder.patch_ln1.weight",
            "v.patch_norm.1.bias": "vision_encoder.patch_ln1.bias",
            "v.patch_embd.weight": "vision_encoder.patch_dense.weight",
            "v.patch_embd.bias": "vision_encoder.patch_dense.bias",
            "v.patch_norm.2.weight": "vision_encoder.patch_ln2.weight",
            "v.patch_norm.2.bias": "vision_encoder.patch_ln2.bias",
            "v.patch_norm.3.weight": "vision_encoder.pos_norm.weight",
            "v.patch_norm.3.bias": "vision_encoder.pos_norm.bias",
            "mm.input_projection.weight": "vision_encoder.projector.weight",
        }.get(name)

    block = _VISION_BLOCK.match(name)
    if projector_type == "idefics3":
        if block is not None:
            index, suffix = block.groups()
            suffix = suffix.replace("ln1.", "layer_norm1.")
            suffix = suffix.replace("ln2.", "layer_norm2.")
            suffix = suffix.replace("attn_out.", "self_attn.out_proj.")
            suffix = suffix.replace("attn_q.", "self_attn.q_proj.")
            suffix = suffix.replace("attn_k.", "self_attn.k_proj.")
            suffix = suffix.replace("attn_v.", "self_attn.v_proj.")
            suffix = suffix.replace("ffn_up.", "mlp.up_proj.")
            suffix = suffix.replace("ffn_down.", "mlp.down_proj.")
            return f"vision_encoder.vision_tower.encoder.{index}.{suffix}"
        return {
            "v.patch_embd.weight": (
                "vision_encoder.vision_tower.embeddings.patch_embedding.projection.weight"
            ),
            "v.patch_embd.bias": (
                "vision_encoder.vision_tower.embeddings.patch_embedding.projection.bias"
            ),
            "v.position_embd.weight": (
                "vision_encoder.vision_tower.embeddings.position_embedding.weight"
            ),
            "v.post_ln.weight": "vision_encoder.vision_tower.post_layernorm.weight",
            "v.post_ln.bias": "vision_encoder.vision_tower.post_layernorm.bias",
            "mm.model.fc.weight": "vision_encoder.projector.model_fc.weight",
        }.get(name)

    if projector_type == "internvl":
        if block is not None:
            index, suffix = block.groups()
            if suffix.startswith(("attn_q.", "attn_k.", "attn_v.")):
                return None
            suffix = suffix.replace("ln1.", "norm1.")
            suffix = suffix.replace("ln2.", "norm2.")
            suffix = suffix.replace("attn_out.", "attn.proj.")
            suffix = suffix.replace("ffn_up.", "mlp.fc1.")
            suffix = suffix.replace("ffn_down.", "mlp.fc2.")
            suffix = suffix.replace("ls1.weight", "ls1")
            suffix = suffix.replace("ls2.weight", "ls2")
            return f"vision_encoder.vision_tower.encoder.layers.{index}.{suffix}"
        return {
            "v.class_embd": (
                "vision_encoder.vision_tower.embeddings.class_embedding"
            ),
            "v.patch_embd.weight": (
                "vision_encoder.vision_tower.embeddings.patch_embedding.weight"
            ),
            "v.patch_embd.bias": (
                "vision_encoder.vision_tower.embeddings.patch_embedding.bias"
            ),
            "v.position_embd.weight": (
                "vision_encoder.vision_tower.embeddings.position_embedding"
            ),
            "mm.model.mlp.0.weight": "vision_encoder.projector.mlp.0.weight",
            "mm.model.mlp.0.bias": "vision_encoder.projector.mlp.0.bias",
            "mm.model.mlp.1.weight": "vision_encoder.projector.mlp.1.weight",
            "mm.model.mlp.1.bias": "vision_encoder.projector.mlp.1.bias",
            "mm.model.mlp.3.weight": "vision_encoder.projector.mlp.3.weight",
            "mm.model.mlp.3.bias": "vision_encoder.projector.mlp.3.bias",
        }.get(name)

    if projector_type == "llama4":
        if block is not None:
            index, suffix = block.groups()
            suffix = suffix.replace("attn_q.", "attn.q_proj.")
            suffix = suffix.replace("attn_k.", "attn.k_proj.")
            suffix = suffix.replace("attn_v.", "attn.v_proj.")
            suffix = suffix.replace("attn_out.", "attn.out_proj.")
            suffix = suffix.replace("ffn_up.", "mlp.up_proj.")
            suffix = suffix.replace("ffn_down.", "mlp.down_proj.")
            return f"vision_encoder.vision_tower.encoder.{index}.{suffix}"
        return {
            "v.patch_embd.weight": (
                "vision_encoder.vision_tower.embeddings.patch_embedding"
            ),
            "v.class_embd": "vision_encoder.vision_tower.embeddings.class_embedding",
            "v.position_embd.weight": (
                "vision_encoder.vision_tower.embeddings.position_embedding"
            ),
            "v.pre_ln.weight": "vision_encoder.vision_tower.pre_layernorm.weight",
            "v.pre_ln.bias": "vision_encoder.vision_tower.pre_layernorm.bias",
            "v.post_ln.weight": "vision_encoder.vision_tower.post_layernorm.weight",
            "v.post_ln.bias": "vision_encoder.vision_tower.post_layernorm.bias",
            "mm.model.mlp.1.weight": "vision_encoder.projector.model_mlp_1.weight",
            "mm.model.mlp.2.weight": "vision_encoder.projector.model_mlp_2.weight",
            "mm.model.fc.weight": "vision_encoder.projector.model_fc.weight",
        }.get(name)

    if projector_type == "pixtral":
        if block is not None:
            index, suffix = block.groups()
            if suffix not in _PIXTRAL_BLOCK_SUFFIXES:
                return None
            suffix = suffix.replace("ln1.", "attention_norm.")
            suffix = suffix.replace("ln2.", "ffn_norm.")
            suffix = suffix.replace("attn_q.", "attention.q_proj.")
            suffix = suffix.replace("attn_k.", "attention.k_proj.")
            suffix = suffix.replace("attn_v.", "attention.v_proj.")
            suffix = suffix.replace("attn_out.", "attention.o_proj.")
            suffix = suffix.replace("ffn_gate.", "feed_forward.gate_proj.")
            suffix = suffix.replace("ffn_up.", "feed_forward.up_proj.")
            suffix = suffix.replace("ffn_down.", "feed_forward.down_proj.")
            return f"vision_encoder.vision_tower.transformer.layers.{index}.{suffix}"
        return {
            "v.patch_embd.weight": "vision_encoder.vision_tower.patch_conv.weight",
            "v.pre_ln.weight": "vision_encoder.vision_tower.ln_pre.weight",
            "mm.1.weight": "vision_encoder.projector.linear_1.weight",
            "mm.1.bias": "vision_encoder.projector.linear_1.bias",
            "mm.2.weight": "vision_encoder.projector.linear_2.weight",
            "mm.2.bias": "vision_encoder.projector.linear_2.bias",
            "v.token_embd.img_break": "vision_encoder.projector.image_break",
        }.get(name)

    raise ValueError(f"Unknown core GGUF projector type {projector_type!r}.")


def _inverse_softplus(values: np.ndarray) -> np.ndarray:
    if np.any(values <= 0):
        raise ValueError("Baked softplus scale tensors must contain positive values.")
    return np.log(np.expm1(values.astype(np.float64))).astype(np.float32)


def _mapped_idefics3_target(
    mmproj_gguf: Any,
    name: str,
) -> str | None:
    target = map_core_vlm_projector_tensor(name, "idefics3")
    hidden = int(mmproj_gguf.metadata["clip.vision.embedding_length"])
    swapped = int(mmproj_gguf.get_tensor_shape("v.blk.0.ffn_down.weight")[1]) == hidden
    if target is not None and swapped:
        if ".ffn_down." in name:
            target = target.replace(".mlp.down_proj.", ".mlp.up_proj.")
        elif ".ffn_up." in name:
            target = target.replace(".mlp.up_proj.", ".mlp.down_proj.")
    return target


def _static_parameter_shape(parameter: Any) -> tuple[int, ...]:
    if parameter.shape is None:
        raise ValueError("Core VLM projector parameters must have static shapes.")
    shape = []
    for dim in parameter.shape:
        if not isinstance(dim, int):
            raise ValueError("Core VLM projector parameter dimensions must be integers.")
        shape.append(dim)
    return tuple(shape)


def validate_core_vlm_projector_shapes(mmproj_gguf: Any, projector_type: str) -> None:
    """Validate source shapes against every learned parameter of the role graph."""
    from mobius.models.gguf_core_projector import CoreVLMProjectorModel

    md = mmproj_gguf.metadata
    if projector_type in {"gemma3nv", "gemma3na"}:
        expected_pair = ("gemma3nv", "gemma3na")
    elif projector_type in {"gemma4uv", "gemma4ua"}:
        expected_pair = ("gemma4uv", "gemma4ua")
    elif projector_type == "gemma4a":
        expected_pair = ("gemma4v", "gemma4a")
    else:
        expected_pair = None
    if expected_pair is not None:
        actual_pair = (
            md.get("clip.vision.projector_type"),
            md.get("clip.audio.projector_type"),
        )
        if actual_pair != expected_pair:
            raise ValueError(
                f"{projector_type} requires co-resident projector pair "
                f"{expected_pair}, got {actual_pair}."
            )

    if projector_type in {"idefics3", "internvl", "llama4"}:
        if md.get("clip.use_gelu") is not True or bool(md.get("clip.use_silu", False)):
            raise ValueError(f"{projector_type} requires clip.use_gelu=true only.")
    if projector_type == "pixtral":
        if md.get("clip.use_silu") is not True or bool(md.get("clip.use_gelu", False)):
            raise ValueError("pixtral requires clip.use_silu=true only.")

    config = read_core_vlm_projector_config(mmproj_gguf, projector_type)
    module = CoreVLMProjectorModel(
        config,
        projector_type,
        with_image_break="v.token_embd.img_break" in mmproj_gguf.tensor_names,
    )
    expected = {
        name: _static_parameter_shape(parameter)
        for name, parameter in module.named_parameters()
        if parameter.const_value is None
    }
    mapped: dict[str, tuple[int, ...]] = {}
    intern_qkv: dict[tuple[int, str], list[tuple[int, ...] | None]] = {}

    for name in mmproj_gguf.tensor_names:
        if projector_type == "internvl":
            match = re.match(r"^v\.blk\.(\d+)\.attn_([qkv])\.(weight|bias)$", name)
            if match is not None:
                key = (int(match.group(1)), match.group(3))
                parts = intern_qkv.setdefault(key, [None, None, None])
                parts[{"q": 0, "k": 1, "v": 2}[match.group(2)]] = (
                    mmproj_gguf.get_tensor_shape(name)
                )
                continue

        target = (
            _mapped_idefics3_target(mmproj_gguf, name)
            if projector_type == "idefics3"
            else map_core_vlm_projector_tensor(name, projector_type)
        )
        if target is None:
            continue
        shape = tuple(int(dim) for dim in mmproj_gguf.get_tensor_shape(name))
        if projector_type == "gemma3nv" and (
            name == "v.conv_stem.conv.bias" or name.endswith(".layer_scale.gamma")
        ):
            shape = tuple(dim for dim in shape if dim != 1)
        elif projector_type == "gemma4a":
            if name.endswith(_CLIP_BOUND_SUFFIXES):
                shape = ()
            elif name.endswith(".conv_dw.weight") and len(shape) == 2:
                shape = (shape[0], 1, shape[1])
        if target in mapped:
            raise ValueError(f"{projector_type} maps multiple tensors to {target!r}.")
        mapped[target] = shape

    for (layer, kind), parts in intern_qkv.items():
        if any(part is None for part in parts):
            raise ValueError(f"InternVL layer {layer} has an incomplete fused QKV {kind}.")
        assert all(part is not None for part in parts)
        first = parts[0]
        assert first is not None
        if any(part[1:] != first[1:] for part in parts if part is not None):
            raise ValueError(f"InternVL layer {layer} QKV {kind} shapes disagree.")
        mapped[
            f"vision_encoder.vision_tower.encoder.layers.{layer}.attn.qkv.{kind}"
        ] = (sum(part[0] for part in parts if part is not None), *first[1:])

    if projector_type == "gemma4uv":
        position = tuple(
            int(dim)
            for dim in mmproj_gguf.get_tensor_shape("v.position_embd.weight")
        )
        mapped["vision_encoder.pos_emb_x.weight"] = (position[1], position[2])
        mapped["vision_encoder.pos_emb_y.weight"] = (position[1], position[2])

    missing = sorted(expected.keys() - mapped.keys())
    unexpected = sorted(mapped.keys() - expected.keys())
    mismatched = {
        name: (mapped[name], expected[name])
        for name in expected.keys() & mapped.keys()
        if mapped[name] != expected[name]
    }
    if missing or unexpected or mismatched:
        raise ValueError(
            f"{projector_type} tensor-to-graph closure mismatch: missing={missing}, "
            f"unexpected={unexpected}, shapes={mismatched}."
        )


def _load_core_vlm_projector_weights(mmproj_gguf: Any, projector_type: str) -> dict:
    import torch

    state: dict[str, torch.Tensor] = {}
    fused_internvl: dict[tuple[int, str], list[np.ndarray | None]] = {}
    for name in mmproj_gguf.tensor_names:
        if projector_type == "internvl":
            match = re.match(r"^v\.blk\.(\d+)\.attn_([qkv])\.(weight|bias)$", name)
            if match is not None:
                key = (int(match.group(1)), match.group(3))
                fused_internvl.setdefault(key, [None, None, None])[
                    {"q": 0, "k": 1, "v": 2}[match.group(2)]
                ] = np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
                continue

        mapped = (
            _mapped_idefics3_target(mmproj_gguf, name)
            if projector_type == "idefics3"
            else map_core_vlm_projector_tensor(name, projector_type)
        )
        if mapped is None:
            continue
        values = np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
        if projector_type == "gemma3nv" and (
            name == "v.conv_stem.conv.bias" or name.endswith(".layer_scale.gamma")
        ):
            values = values.reshape(-1)
        elif projector_type in {"gemma3na", "gemma4a"} and (
            name.endswith("per_dim_scale") or name.endswith("per_dim_scale.weight")
        ):
            # The converter stores softplus(raw); the ONNX components apply
            # softplus at runtime, so restore the trainable pre-softplus value.
            values = _inverse_softplus(values)
        elif projector_type == "gemma4a" and name.endswith(_CLIP_BOUND_SUFFIXES):
            values = values.reshape(())
        elif projector_type == "gemma4a" and name.endswith(".conv_dw.weight"):
            values = values[:, None, :]
        elif projector_type == "gemma4uv" and name in {
            "v.patch_embd.weight",
            "v.patch_norm.1.weight",
            "v.patch_norm.1.bias",
        }:
            patch = int(mmproj_gguf.metadata["clip.vision.patch_size"]) * 3
            indices = np.arange(patch * patch * 3)
            channels: np.ndarray = indices // (patch * patch)
            rows: np.ndarray = (indices % (patch * patch)) // patch
            columns: np.ndarray = indices % patch
            permutation = rows * patch * 3 + columns * 3 + channels
            inverse = np.argsort(permutation)
            values = values[:, inverse] if values.ndim == 2 else values[inverse]
        state[mapped] = torch.from_numpy(values.copy())

    if projector_type == "internvl":
        for (layer, kind), parts in fused_internvl.items():
            if any(part is None for part in parts):
                raise ValueError(f"InternVL layer {layer} has an incomplete fused QKV {kind}.")
            arrays = [part for part in parts if part is not None]
            state[
                f"vision_encoder.vision_tower.encoder.layers.{layer}.attn.qkv.{kind}"
            ] = torch.from_numpy(np.concatenate(arrays, axis=0).copy())

    if projector_type == "gemma4uv":
        position = np.array(mmproj_gguf.get_tensor("v.position_embd.weight")).astype(np.float32)
        state["vision_encoder.pos_emb_x.weight"] = torch.from_numpy(position[0].copy())
        state["vision_encoder.pos_emb_y.weight"] = torch.from_numpy(position[1].copy())
    return state


def build_core_vlm_projector_mmproj(
    mmproj_gguf_path: str | Path,
    *,
    projector_type: str,
    target_architecture: str,
    dtype: str | None = None,
    execution_provider: str = "default",
    _mmproj_gguf_model: Any | None = None,
):
    from mobius._builder import build_from_module, resolve_dtype
    from mobius.integrations.gguf._mmproj import (
        _canonical_text_architecture,
        _resolve_mmproj_companion_path,
    )
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.models.gguf_core_projector import CoreVLMProjectorModel
    from mobius.tasks._gguf_core_projector import CoreVLMProjectorTask
    from mobius.tasks._gguf_projector import (
        GGUFVisionProjectorModel,
        GGUFVisionProjectorTask,
    )

    resolved_path = _resolve_mmproj_companion_path(mmproj_gguf_path)
    target_architecture = _canonical_text_architecture(target_architecture)
    mmproj_gguf = (
        _mmproj_gguf_model
        if _mmproj_gguf_model is not None
        else GGUFModel(resolved_path)
    )
    config = read_core_vlm_projector_config(mmproj_gguf, projector_type)
    if dtype is not None:
        resolved_dtype = resolve_dtype(dtype)
        if resolved_dtype is not None:
            config = dataclasses.replace(config, dtype=resolved_dtype)
    module = CoreVLMProjectorModel(
        config,
        projector_type,
        with_image_break="v.token_embd.img_break" in mmproj_gguf.tensor_names,
    )
    if projector_type in {"gemma3na", "gemma4a", "gemma4ua"}:
        package = build_from_module(
            module,
            config,
            task=CoreVLMProjectorTask(projector_type),
            execution_provider=execution_provider,
        )
    else:
        package = build_from_module(
            GGUFVisionProjectorModel(module.vision_encoder),
            config,
            task=GGUFVisionProjectorTask(),
            execution_provider=execution_provider,
        )
    state = _load_core_vlm_projector_weights(mmproj_gguf, projector_type)
    expected = {
        name
        for model in package.values()
        for name, initializer in model.graph.initializers.items()
        if initializer.const_value is None
    }
    missing = sorted(expected - state.keys())
    if missing:
        raise ValueError(
            f"{projector_type} GGUF mapping did not fill graph initializer(s): {missing}"
        )
    package.apply_weights({name: value for name, value in state.items() if name in expected})
    package.gguf_source_path = str(Path(resolved_path).resolve())  # type: ignore[attr-defined]
    package.gguf_architecture = "clip"  # type: ignore[attr-defined]
    package.gguf_projector_type = projector_type  # type: ignore[attr-defined]
    package.gguf_import_route = json.dumps(  # type: ignore[attr-defined]
        {
            "architecture": "clip",
            "config_sha256": core_vlm_projector_fingerprint(
                config,
                projector_type,
                target_architecture,
            ),
            "execution_provider": execution_provider,
            "model_roles": sorted(package.keys()),
            "projector_type": projector_type,
            "route_schema": 1,
            "runtime": "deferred",
            "target_architecture": target_architecture,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return package
