# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build a full Gemma4 multimodal ONNX package from GGUF (text + mmproj).

Gemma4's text backbone ships in one ``*.gguf`` file while its vision (and
audio) encoders live in a companion ``mmproj-*.gguf`` whose
``general.architecture`` is ``clip``.  :func:`build_gemma4_vlm_from_gguf`
assembles both into a runtime-ready :class:`ModelPackage`:

- **decoder** — the Gemma4 text decoder (from the text GGUF), taking
  ``inputs_embeds``.
- **vision_encoder** — the Gemma4 SigLIP vision encoder + projector (from the
  mmproj), taking ``pixel_values`` + ``pixel_position_ids``.
- **audio_encoder** — optional Gemma4 Conformer audio encoder + projector
  (from the mmproj), enabled with ``include_audio=True``.
- **embedding** — scaled token lookup that fuses text, image, and optional
  audio features (built from the text config, reusing
  :class:`Gemma4EmbeddingModel`).

The mmproj ``clip.vision.*`` metadata is read into a :class:`VisionConfig`
(:func:`read_mmproj_vision_config`) and merged onto the text
:class:`Gemma4Config`; the mmproj ``v.*``/``mm.*`` tensors are mapped to their
HF names (:mod:`_mmproj_mapping`) so they flow through the same tested
``Gemma4Model.preprocess_weights`` path as a real HF checkpoint.

Audio remains opt-in while its output quality is validated against the source
checkpoint. The graph, checkpoint mapping, and ORT-GenAI execution path are
covered; pass ``include_audio=True`` to include it.
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
# mmproj metadata; Gemma4VisionConfig defaults to 3.
_DEFAULT_POOLING_KERNEL_SIZE = 3


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
        # The official mmproj carries activation ranges for every vision
        # projection; preserve those Clip operations around each linear.
        use_clipped_linears=True,
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


def _special_token_id(gguf_model: Any, token: str) -> int:
    tokens = gguf_model.metadata.get("tokenizer.ggml.tokens")
    if not tokens:
        raise ValueError("Text GGUF has no tokenizer.ggml.tokens metadata.")
    try:
        return tokens.index(token)
    except ValueError as exc:
        raise ValueError(f"Text GGUF tokenizer does not contain {token!r}.") from exc


def _resolve_local_path(path: str | Path) -> str:
    from mobius.integrations.gguf._builder import _resolve_gguf_path

    return _resolve_gguf_path(path)


# Text-GGUF tensors whose names are not covered by the block ``gemma4`` text
# mapping; they route to their HF language-model names directly.
_PER_LAYER_TOP_HF_NAMES = {
    "per_layer_token_embd.weight": "language_model.embed_tokens_per_layer.weight",
    "per_layer_model_proj.weight": "language_model.per_layer_model_projection.weight",
    "per_layer_proj_norm.weight": "language_model.per_layer_projection_norm.weight",
}


def _text_gguf_name_to_hf_multimodal(gguf_name: str) -> str | None:
    """Map one text-GGUF tensor name to its HF multimodal (``language_model.*``) name.

    Returns ``None`` for tensors that have no multimodal counterpart.
    """
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    if gguf_name in _PER_LAYER_TOP_HF_NAMES:
        return _PER_LAYER_TOP_HF_NAMES[gguf_name]
    text_hf = map_gguf_to_hf_names(gguf_name, "gemma4")
    if text_hf is None:
        return None
    # gemma4 text mapping yields ``model.*`` / ``lm_head.*``; nest under the
    # multimodal ``language_model.`` namespace (HF Gemma4 stores the decoder
    # layers directly under language_model, so strip the ``model.`` prefix).
    if text_hf.startswith("model."):
        return "language_model." + text_hf[len("model.") :]
    return "language_model." + text_hf


def _text_gguf_to_hf_multimodal(text_gguf: Any) -> dict:
    """Load text-backbone GGUF tensors as HF multimodal (``language_model.*``).

    Reuses the text ``gemma4`` GGUF→HF name mapping, then rewrites names into
    the multimodal ``language_model.*`` namespace that
    ``Gemma4Model.preprocess_weights`` expects.
    """
    import torch

    state_dict: dict[str, torch.Tensor] = {}
    for gguf_name, array in text_gguf.tensor_items():
        hf_name = _text_gguf_name_to_hf_multimodal(gguf_name)
        if hf_name is None:
            continue

        values = np.array(array).astype(np.float32)
        # layer_scalar is an nn.Parameter (no ``.weight`` module suffix, shape [1]).
        if hf_name.endswith(".layer_scalar.weight"):
            hf_name = hf_name[: -len(".weight")]
            values = values.reshape(-1)[:1]
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict


# HF projection-weight name suffixes (under ``language_model.layers.N.``) that
# become MatMulNBits QuantizedLinear layers in the quantized text decoder.
_QUANTIZED_LINEAR_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".per_layer_input_gate.weight",
    ".per_layer_projection.weight",
)

# HF token-embedding weight names that become GatherBlockQuantized tables.
_QUANTIZED_EMBEDDING_NAMES = (
    "language_model.embed_tokens.weight",
    "language_model.embed_tokens_per_layer.weight",
)


def _text_gguf_to_hf_multimodal_quantized(
    text_gguf: Any,
    config: Any,
    *,
    bits: int,
    block_size: int,
    symmetric: bool,
) -> dict:
    """Load text-backbone GGUF tensors, quantizing the decoder + token embeddings.

    Mirrors :func:`_text_gguf_to_hf_multimodal` but keeps the text decoder
    projections in MatMulNBits form (``.weight`` uint8 + ``.scales`` [+
    ``.zero_points``]) and the token-embedding tables in GatherBlockQuantized
    form (``.qweight`` + ``.scales`` [+ ``.zero_points``]).  Norms and the
    float per-layer projections stay dequantized.  Vision/audio weights are
    loaded separately and always stay float.

    Names are the HF multimodal ``language_model.*`` names that
    :meth:`Gemma4Model.preprocess_weights` expects.
    """
    import torch

    from mobius.integrations.gguf._builder import repack_gguf_weight_to_target

    quantize_lm_head = bool(getattr(config.quantization, "quantize_lm_head", False))
    tie_word_embeddings = bool(config.tie_word_embeddings)

    def _emit_linear(state_dict: dict, stem: str, repacked) -> None:
        state_dict[f"{stem}.weight"] = torch.from_numpy(repacked.weight)
        state_dict[f"{stem}.scales"] = torch.from_numpy(repacked.scales)
        if repacked.zero_points is not None:
            state_dict[f"{stem}.zero_points"] = torch.from_numpy(repacked.zero_points)

    def _emit_embedding(state_dict: dict, stem: str, repacked) -> None:
        weight = repacked.weight
        state_dict[f"{stem}.qweight"] = torch.from_numpy(weight.reshape(weight.shape[0], -1))
        state_dict[f"{stem}.scales"] = torch.from_numpy(repacked.scales)
        if repacked.zero_points is not None:
            state_dict[f"{stem}.zero_points"] = torch.from_numpy(repacked.zero_points)

    state_dict: dict[str, torch.Tensor] = {}
    for gguf_name, raw, qtype, np_shape in text_gguf.tensor_items_raw():
        hf_name = _text_gguf_name_to_hf_multimodal(gguf_name)
        if hf_name is None:
            continue

        is_quant_linear = hf_name.endswith(_QUANTIZED_LINEAR_SUFFIXES) or (
            quantize_lm_head and hf_name == "language_model.lm_head.weight"
        )
        is_quant_embedding = bool(
            getattr(config.quantization, "quantize_embeddings", False)
            and hf_name in _QUANTIZED_EMBEDDING_NAMES
        )

        if (is_quant_linear or is_quant_embedding) and len(np_shape) == 2:
            repacked = repack_gguf_weight_to_target(
                text_gguf,
                raw,
                qtype,
                np_shape,
                target_bits=bits,
                target_block_size=block_size,
                target_symmetric=symmetric,
                tensor_name=hf_name,
            )
            stem = hf_name[: -len(".weight")]
            if is_quant_embedding:
                _emit_embedding(state_dict, stem, repacked)
                # A tied quantized LM head shares the token-embedding table but,
                # because the decoder is a separate ONNX graph, needs its own
                # MatMulNBits copy (same repacked bytes, linear layout).
                if (
                    hf_name == "language_model.embed_tokens.weight"
                    and quantize_lm_head
                    and tie_word_embeddings
                ):
                    _emit_linear(state_dict, "language_model.lm_head", repacked)
            else:
                _emit_linear(state_dict, stem, repacked)
            continue

        # Everything else (norms, float per-layer projections) stays float.
        values = np.array(text_gguf.dequantize_raw_tensor(raw, qtype, np_shape)).astype(
            np.float32
        )
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
        if name.endswith((".input_max", ".input_min", ".output_max", ".output_min")):
            values = values.reshape(())
        elif name == "v.patch_embd.weight":
            # GGUF stores Conv weights as [out, channel, height, width], while
            # Gemma4ImageTransform flattens each patch as [height, width, channel].
            values = values.transpose(0, 2, 3, 1).reshape(values.shape[0], -1)
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
    keep_quantized: bool = False,
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
        keep_quantized: When ``True``, preserve the text backbone's GGUF
            quantization: the decoder projections become MatMulNBits and the
            token-embedding tables become GatherBlockQuantized (int4), roughly
            an 8x size reduction over the dequantized package. The
            vision (and audio) encoder always stay float because their weights
            come from the mmproj as F16 — see the "Mixed precision" note below.

    Returns:
        A :class:`ModelPackage` with ``decoder`` + ``vision_encoder`` +
        ``embedding`` components (plus ``audio_encoder`` if ``include_audio``).

    Mixed precision:
        ``keep_quantized`` yields a mixed-precision package. Only the Gemma4
        *text* components read ``config.quantization`` (see
        :func:`mobius.models.gemma4._text_linear_class`); the vision/audio
        encoder modules always build float ``Linear`` layers, so a single
        module-global :class:`QuantizationConfig` quantizes the decoder +
        embedding while leaving the mmproj-sourced vision encoder float — no
        per-module quantization opt-out is required.
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
    resolved_image_token_id = (
        image_token_id
        if image_token_id is not None
        else _special_token_id(text_gguf, "<|image|>")
    )
    vision_config = dataclasses.replace(vision_config, image_token_id=resolved_image_token_id)
    config = dataclasses.replace(
        config,
        model_type="gemma4",
        vision=vision_config,
        image_token_id=resolved_image_token_id,
    )

    if include_audio:
        audio_config = read_mmproj_audio_config(mmproj_gguf)
        if audio_config is None:
            raise ValueError("Audio was requested but the mmproj GGUF has no audio encoder.")
        audio_token_id = _special_token_id(text_gguf, "<|audio|>")
        audio_config = dataclasses.replace(audio_config, audio_token_id=audio_token_id)
        config = dataclasses.replace(
            config,
            audio=audio_config,
            audio_token_id=audio_token_id,
            boa_token_id=_special_token_id(text_gguf, "<|audio>"),
        )
    else:
        # Vision-only VLM: drop any audio sub-config so the package is the
        # 3-component (decoder + vision + embedding) multimodal shape.
        config = dataclasses.replace(config, audio=None)

    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None:
            config = dataclasses.replace(config, dtype=resolved)

    # 1b. Quantized mode: set the module-global quantization config from the
    # text GGUF BEFORE building so the text graph emits MatMulNBits /
    # GatherBlockQuantized. The vision/audio encoders ignore it and stay float.
    quant_params: tuple[int, int, bool] | None = None
    if keep_quantized:
        from mobius._configs import QuantizationConfig
        from mobius.integrations.gguf._builder import (
            _can_quantize_embedding,
            _can_quantize_lm_head,
            _detect_quant_params,
        )

        bits, block_size, is_symmetric = _detect_quant_params(text_gguf, "gemma4")
        quant_params = (bits, block_size, is_symmetric)
        quantize_embeddings = _can_quantize_embedding(
            text_gguf, "gemma4", bits=bits, block_size=block_size
        )
        quantize_lm_head = (
            quantize_embeddings
            if config.tie_word_embeddings
            else _can_quantize_lm_head(text_gguf, "gemma4")
        )
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=block_size,
                quant_method="gguf",
                sym=is_symmetric,
                quantize_embeddings=quantize_embeddings,
                quantize_lm_head=quantize_lm_head,
                tie_word_embeddings=quantize_lm_head and config.tie_word_embeddings,
            ),
        )
        logger.info(
            "Quantized multimodal mode: bits=%d, block_size=%d, symmetric=%s, "
            "embedding=%s, lm_head=%s (vision/audio stay float)",
            bits,
            block_size,
            is_symmetric,
            quantize_embeddings,
            quantize_lm_head,
        )

    # 2. Build the multimodal graph (decoder + vision + embedding [+ audio]).
    #    Route through build_from_module so each component gets the EP-aware
    #    optimize_model passes (GQA fusion, etc.) — the same pipeline the
    #    text-only build_from_gguf path uses; calling Gemma4Task().build()
    #    directly would skip those optimizations.
    from mobius._builder import build_from_module

    module = Gemma4Model(config)
    pkg = build_from_module(
        module, config, task=Gemma4Task(), execution_provider=execution_provider
    )
    logger.info("Built Gemma4 VLM graph (%d components: %s)", len(pkg), list(pkg))

    # 3. Assemble the combined HF-multimodal state dict from both GGUFs. The
    #    text backbone is quantized when requested; vision/audio always float.
    if keep_quantized:
        bits, block_size, is_symmetric = quant_params
        state_dict = _text_gguf_to_hf_multimodal_quantized(
            text_gguf,
            config,
            bits=bits,
            block_size=block_size,
            symmetric=is_symmetric,
        )
    else:
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
        if name.endswith((".input_max", ".input_min", ".output_max", ".output_min")):
            values = values.reshape(())
        elif hf_name.endswith(".per_dim_scale"):
            values = values.reshape(-1)
        elif hf_name.endswith(".depthwise_conv1d.weight") and values.ndim == 2:
            values = values[:, None, :]
        state_dict[hf_name] = torch.from_numpy(values)
    return state_dict
