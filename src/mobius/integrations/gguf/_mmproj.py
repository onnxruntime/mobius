# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build a full Gemma4 multimodal ONNX package from GGUF (text + mmproj).

Gemma4's text backbone ships in one ``*.gguf`` file while its vision (and
audio) encoders live in a companion ``mmproj-*.gguf`` whose
``general.architecture`` is ``clip``.  :func:`build_gemma4_vlm_from_gguf`
assembles both into a single onnx-genai-ready :class:`ModelPackage`:

- **decoder** — the Gemma4 text decoder (from the text GGUF), taking
  ``inputs_embeds``.
- **vision_encoder** — the Gemma4 SigLIP vision encoder + projector (from the
  mmproj), taking ``pixel_values`` + ``pixel_position_ids``.
- **embedding** — scaled token lookup that fuses text + image features (built
  from the text config, reusing :class:`Gemma4EmbeddingModel`).

The mmproj ``clip.vision.*`` metadata is read into a :class:`VisionConfig`
(:func:`read_mmproj_vision_config`) and merged onto the text
:class:`Gemma4Config`; the mmproj ``v.*``/``mm.*`` tensors are mapped to their
HF names (:mod:`_mmproj_mapping`) so they flow through the same tested
``Gemma4Model.preprocess_weights`` path as a real HF checkpoint.

Audio: :func:`read_mmproj_audio_config` extracts the Conformer
:class:`Gemma4AudioConfig` and the audio tensor mapping exists
(:func:`map_mmproj_audio_to_hf`), but the audio encoder is **not** yet wired
into the assembled package — its Conformer forward-pass weight layout still
needs validation against llama.cpp's ``clip.cpp`` gemma4a reference.  Pass
``include_audio=True`` to opt in to the (experimental) audio path.
"""

from __future__ import annotations

__all__ = [
    "build_gemma4_vlm_from_gguf",
    "read_mmproj_audio_config",
    "read_mmproj_vision_config",
]

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)

# Gemma4 vision RoPE base frequency. The ``clip`` mmproj metadata does not store
# it, so use the HF Gemma4VisionAttention default. Source: HF Gemma4 vision.
_GEMMA4_VISION_ROPE_THETA = 100.0
# Gemma4 vision pooler spatial average pooling kernel (k x k). Not present in
# mmproj metadata; the SigLIP encoder pools N patches to N/k^2 soft tokens.
_DEFAULT_POOLING_KERNEL_SIZE = 4


def read_mmproj_vision_config(gguf_model: Any):
    """Extract a Gemma4 :class:`VisionConfig` from ``clip.vision.*`` metadata.

    Args:
        gguf_model: A :class:`GGUFModel` for a ``clip`` mmproj file.

    Returns:
        A populated :class:`VisionConfig`, or ``None`` if the file has no
        vision encoder (``clip.has_vision_encoder`` is false/absent).
    """
    from mobius._configs._sub_configs import VisionConfig

    md = gguf_model.metadata
    if not md.get("clip.has_vision_encoder"):
        return None

    # Position embedding table size is authoritative from the tensor itself
    # ([2, pos_emb_size, hidden]); metadata carries no equivalent key.
    pos_emb_size = int(gguf_model.get_tensor("v.position_embd.weight").shape[1])

    return VisionConfig(
        hidden_size=int(md["clip.vision.embedding_length"]),
        intermediate_size=int(md["clip.vision.feed_forward_length"]),
        num_hidden_layers=int(md["clip.vision.block_count"]),
        num_attention_heads=int(md["clip.vision.attention.head_count"]),
        image_size=int(md["clip.vision.image_size"]),
        patch_size=int(md["clip.vision.patch_size"]),
        norm_eps=float(md.get("clip.vision.attention.layer_norm_epsilon", 1e-6)),
        # The mmproj carries activation-range stats, not clipped-linear weights,
        # so the encoder uses plain (non-clipped) Linear layers.
        use_clipped_linears=False,
        position_embedding_size=pos_emb_size,
        pooling_kernel_size=_DEFAULT_POOLING_KERNEL_SIZE,
        hidden_act="gelu_pytorch_tanh",
        rope_theta=_GEMMA4_VISION_ROPE_THETA,
    )


def read_mmproj_audio_config(gguf_model: Any):
    """Extract a Gemma4 :class:`Gemma4AudioConfig` from ``clip.audio.*``.

    Args:
        gguf_model: A :class:`GGUFModel` for a ``clip`` mmproj file.

    Returns:
        A populated :class:`Gemma4AudioConfig`, or ``None`` if the file has no
        audio encoder (``clip.has_audio_encoder`` is false/absent).
    """
    from mobius._configs._sub_configs import Gemma4AudioConfig

    md = gguf_model.metadata
    if not md.get("clip.has_audio_encoder"):
        return None

    # Subsampling conv channel sizes are read from the conv tensors:
    # a.conv1d.0.weight [c0, 1, 3, 3], a.conv1d.1.weight [c1, c0, 3, 3].
    conv0 = int(gguf_model.get_tensor("a.conv1d.0.weight").shape[0])
    conv1 = int(gguf_model.get_tensor("a.conv1d.1.weight").shape[0])
    # Projector output dim = mm.a.input_projection input width.
    output_proj_dims = int(gguf_model.get_tensor("mm.a.input_projection.weight").shape[1])

    return Gemma4AudioConfig(
        input_size=int(md["clip.audio.num_mel_bins"]),
        hidden_size=int(md["clip.audio.embedding_length"]),
        num_layers=int(md["clip.audio.block_count"]),
        attention_heads=int(md["clip.audio.attention.head_count"]),
        linear_units=int(md["clip.audio.feed_forward_length"]),
        subsampling_conv_channels=[conv0, conv1],
        output_proj_dims=output_proj_dims,
        rms_norm_eps=float(md.get("clip.audio.attention.layer_norm_epsilon", 1e-6)),
    )


def _resolve_local_path(path: str | Path) -> str:
    from mobius.integrations.gguf._builder import _resolve_gguf_path

    return _resolve_gguf_path(path)


def _text_gguf_to_hf_multimodal(text_gguf: Any) -> dict:
    """Load text-backbone GGUF tensors as HF multimodal (``language_model.*``).

    Reuses the text ``gemma4`` GGUF→HF name mapping, then rewrites names into
    the multimodal ``language_model.*`` namespace that
    ``Gemma4Model.preprocess_weights`` expects.
    """
    import torch

    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    # The three top-level per-layer-input tensors are not covered by the block
    # text mapping; route them to their HF language-model names directly.
    per_layer_top = {
        "per_layer_token_embd.weight": "language_model.embed_tokens_per_layer.weight",
        "per_layer_model_proj.weight": "language_model.per_layer_model_projection.weight",
        "per_layer_proj_norm.weight": "language_model.per_layer_projection_norm.weight",
    }

    state_dict: dict[str, torch.Tensor] = {}
    for gguf_name, array in text_gguf.tensor_items():
        if gguf_name in per_layer_top:
            hf_name = per_layer_top[gguf_name]
        else:
            text_hf = map_gguf_to_hf_names(gguf_name, "gemma4")
            if text_hf is None:
                continue
            # gemma4 text mapping yields ``model.*`` / ``lm_head.*``; nest under
            # the multimodal ``language_model.`` namespace (HF Gemma4 stores the
            # decoder layers directly under language_model, so strip ``model.``).
            if text_hf.startswith("model."):
                hf_name = "language_model." + text_hf[len("model.") :]
            else:
                hf_name = "language_model." + text_hf

        values = np.array(array).astype(np.float32)
        # layer_scalar is an nn.Parameter (no ``.weight`` module suffix, shape [1]).
        if hf_name.endswith(".layer_scalar.weight"):
            hf_name = hf_name[: -len(".weight")]
            values = values.reshape(-1)[:1]
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict


def _mmproj_vision_to_hf(mmproj_gguf: Any) -> dict:
    """Load mmproj vision tensors as HF names (``vision_tower.*``/``embed_vision.*``)."""
    import torch

    from mobius.integrations.gguf._mmproj_mapping import map_mmproj_vision_to_hf

    state_dict: dict[str, torch.Tensor] = {}
    for name in mmproj_gguf.tensor_names:
        if not (name.startswith("v.") or name == "mm.input_projection.weight"):
            continue
        hf_name = map_mmproj_vision_to_hf(name)
        if hf_name is None:
            continue
        values = np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
        if name == "v.patch_embd.weight":
            # Conv patch embed [out, in_ch, kh, kw] → Linear [out, in_ch*kh*kw].
            # The flattening order (in_ch, kh, kw) row-major matches the
            # pre-patchified pixel_values layout consumed by the encoder.
            values = values.reshape(values.shape[0], -1)
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict


def build_gemma4_vlm_from_gguf(
    text_gguf_path: str | Path,
    mmproj_gguf_path: str | Path,
    *,
    dtype: str | None = None,
    execution_provider: str = "default",
    image_token_id: int | None = None,
    include_audio: bool = False,
) -> ModelPackage:
    """Build a full Gemma4 multimodal ONNX package from text + mmproj GGUFs.

    Args:
        text_gguf_path: Path (or HF ref) to the Gemma4 text-backbone GGUF.
        mmproj_gguf_path: Path (or HF ref) to the companion ``clip`` mmproj GGUF.
        dtype: Optional dtype override (e.g. ``"f16"``); defaults to float32.
        execution_provider: Target EP for EP-aware optimisations.
        image_token_id: Vocabulary id of the image soft-token placeholder used
            to scatter image features into text embeddings. When ``None``, the
            value carried by the text config is used (if any).
        include_audio: When ``True``, also build the (experimental) audio
            encoder. Off by default — see the module docstring.

    Returns:
        A :class:`ModelPackage` with ``decoder`` + ``vision_encoder`` +
        ``embedding`` components (plus ``audio_encoder`` if ``include_audio``).
    """
    import dataclasses

    from mobius._builder import resolve_dtype
    from mobius.integrations.gguf._config_mapping import gguf_to_config
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.models.gemma4 import Gemma4Model
    from mobius.tasks._gemma4 import Gemma4Task

    text_gguf = GGUFModel(_resolve_local_path(text_gguf_path))
    mmproj_gguf = GGUFModel(_resolve_local_path(mmproj_gguf_path))
    if mmproj_gguf.architecture != "clip":
        raise ValueError(
            f"Expected a 'clip' mmproj GGUF, got architecture "
            f"{mmproj_gguf.architecture!r} for {mmproj_gguf_path!r}."
        )
    logger.info("Building Gemma4 VLM from text=%s mmproj=%s", text_gguf_path, mmproj_gguf_path)

    # 1. Text config + merged vision/audio sub-configs.
    config = gguf_to_config(text_gguf)
    vision_config = read_mmproj_vision_config(mmproj_gguf)
    if vision_config is None:
        raise ValueError(
            "mmproj GGUF has no vision encoder (clip.has_vision_encoder is unset)."
        )
    config = dataclasses.replace(config, vision=vision_config)

    if include_audio:
        config = dataclasses.replace(config, audio=read_mmproj_audio_config(mmproj_gguf))
    else:
        # Vision-only VLM: drop any audio sub-config so the package is the
        # 3-component (decoder + vision + embedding) multimodal shape.
        config = dataclasses.replace(config, audio=None)

    if image_token_id is not None:
        config = dataclasses.replace(config, image_token_id=image_token_id)
    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None:
            config = dataclasses.replace(config, dtype=resolved)

    # 2. Build the multimodal graph (decoder + vision + embedding [+ audio]).
    module = Gemma4Model(config)
    pkg = Gemma4Task().build(module, config)
    logger.info("Built Gemma4 VLM graph (%d components: %s)", len(pkg), list(pkg))

    # 3. Assemble the combined HF-multimodal state dict from both GGUFs.
    state_dict = _text_gguf_to_hf_multimodal(text_gguf)
    state_dict.update(_mmproj_vision_to_hf(mmproj_gguf))
    if include_audio:
        state_dict.update(_mmproj_audio_to_hf(mmproj_gguf))

    # 4. Run the tested HF→ONNX preprocessing and apply. Names are already
    #    component-qualified after preprocessing, so no prefix_map is needed.
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)
    logger.info("Applied %d mapped weights to the Gemma4 VLM package", len(state_dict))

    return pkg


def _mmproj_audio_to_hf(mmproj_gguf: Any) -> dict:
    """Load mmproj audio tensors as HF names (experimental; see module docstring)."""
    import torch

    from mobius.integrations.gguf._mmproj_mapping import map_mmproj_audio_to_hf

    state_dict: dict[str, torch.Tensor] = {}
    for name in mmproj_gguf.tensor_names:
        if not (name.startswith("a.") or name == "mm.a.input_projection.weight"):
            continue
        hf_name = map_mmproj_audio_to_hf(name)
        if hf_name is None:
            continue
        values = np.array(mmproj_gguf.get_tensor(name)).astype(np.float32)
        if hf_name.endswith(".per_dim_scale"):
            values = values.reshape(-1)
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict
