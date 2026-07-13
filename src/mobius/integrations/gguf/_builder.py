# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF → ONNX build pipeline.

Converts ``.gguf`` model files to ONNX using the standard build
pipeline.  Two modes:

- **Dequantized** (default): All quantized tensors are dequantized to
  float.  Simple, but loses the compression benefit of quantization.
- **Quantized** (``keep_quantized=True``): Linear-layer weights, including
  a quantized output head, are repacked into MatMulNBits format and token
  embeddings into GatherBlockQuantized format. Mixed presets such as
  Q4_K_M are normalized to one quantization layout. Other tensors are
  dequantized.
"""

from __future__ import annotations

__all__ = ["build_from_gguf"]

import logging
from collections import Counter
from pathlib import Path

import tqdm
from huggingface_hub import HfApi, hf_hub_download

from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)


def _looks_like_hf_repo_id(value: str) -> bool:
    """Heuristic: ``value`` matches ``owner/repo`` (no path separators, no .gguf suffix)."""
    if value.startswith((".", "/", "~")):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(p and not p.endswith(".gguf") for p in parts)


def _resolve_gguf_path(gguf_path: str | Path) -> str:
    """Resolve a GGUF reference to a local file path.

    Accepts:
    - An existing local filesystem path (returned unchanged).
    - A HuggingFace Hub reference ``"owner/repo"`` — the repo must contain
      exactly one ``*.gguf`` file, which is downloaded.
    - A HuggingFace Hub reference ``"owner/repo:filename.gguf"`` to pick a
      specific file from a multi-file repo.
    """
    raw = str(gguf_path)
    if Path(raw).exists():
        return raw

    # Split the optional ":filename" suffix before classifying so HF refs like
    # "owner/repo:weights.gguf" are not mistaken for a local path ending in .gguf.
    repo_id, _, filename = raw.partition(":")
    if not _looks_like_hf_repo_id(repo_id):
        # Looks like a local path that doesn't exist; let GGUFModel raise
        # FileNotFoundError with the original path.
        return raw

    if not filename:
        files = [f for f in HfApi().list_repo_files(repo_id) if f.endswith(".gguf")]
        if not files:
            raise FileNotFoundError(f"No *.gguf files found in HF repo {repo_id!r}")
        if len(files) > 1:
            raise ValueError(
                f"HF repo {repo_id!r} contains multiple .gguf files: {files}. "
                f"Specify one via '{repo_id}:<filename.gguf>'."
            )
        filename = files[0]

    logger.info("Downloading %s from %s", filename, repo_id)
    return hf_hub_download(repo_id=repo_id, filename=filename)


def build_from_gguf(
    gguf_path: str | Path,
    *,
    task: str | None = None,
    dtype: str | None = None,
    keep_quantized: bool = False,
    execution_provider: str = "default",
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a GGUF file.

    1. Parse GGUF metadata → :class:`ArchitectureConfig`
    2. Look up the model class and task from the registry
    3. Build the ONNX graph (standard ``build_from_module`` pipeline)
    4. Map GGUF tensor names → HuggingFace names
    5. Apply architecture-specific tensor processors
    6. Run ``preprocess_weights()`` (HF → ONNX name mapping)
    7. Apply weights to the ONNX model

    When *keep_quantized* is ``True``, supported quantized tensors are
    repacked into MatMulNBits format instead of dequantized.

    Args:
        gguf_path: Path to the ``.gguf`` file, *or* a HuggingFace Hub
            reference of the form ``"owner/repo"`` (the repo must
            contain exactly one ``*.gguf`` file) or
            ``"owner/repo:filename.gguf"`` to pick a specific file. HF
            references are downloaded via ``huggingface_hub`` into the
            standard local cache.
        task: Override the model task (e.g. ``"text-generation"``).
            When ``None``, the task is auto-detected from the
            model type.
        dtype: Override model dtype (e.g. ``"f16"``). When ``None``,
            defaults to float32.
        keep_quantized: When ``True``, preserve quantized linear-layer
            weights in MatMulNBits format. Mixed Q4_K_M source types are
            normalized to 4-bit, block-32 weights.
        execution_provider: Target execution provider for EP-aware
            optimisations (e.g. ``"cpu"`` to apply the
            GroupQueryAttention rewrite). Defaults to ``"default"``
            (portable, no vendor fusions).

    Returns:
        A :class:`ModelPackage` containing the built model(s).

    Raises:
        ImportError: If the ``gguf`` package is not installed.
        FileNotFoundError: If the GGUF file does not exist.
        KeyError: If the GGUF architecture is not in the registry.
    """
    import dataclasses

    from mobius._builder import (
        build_from_module,
        resolve_dtype,
    )
    from mobius._config_resolver import (
        _default_task_for_model,
    )
    from mobius._registry import registry
    from mobius.integrations.gguf._config_mapping import (
        GGUF_ARCH_TO_MODEL_TYPE,
        gguf_to_config,
    )
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._tensor_processors import (
        process_tensors,
    )

    # 1. Parse GGUF file (auto-download from HF Hub when given "owner/repo[:filename]")
    gguf_path = _resolve_gguf_path(gguf_path)
    gguf_model = GGUFModel(gguf_path)
    gguf_arch = gguf_model.architecture
    logger.info("Loaded GGUF file: %s (arch=%s)", gguf_path, gguf_arch)

    # 2. Extract config from GGUF metadata
    config = gguf_to_config(gguf_model)
    model_type = getattr(config, "_gguf_model_type", None)
    if model_type is None:
        model_type = GGUF_ARCH_TO_MODEL_TYPE.get(gguf_arch, gguf_arch)

    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None:
            config = dataclasses.replace(config, dtype=resolved)

    # 3. Quantized path: detect dominant type and set config
    if keep_quantized:
        from mobius._configs import QuantizationConfig
        from mobius._flags import flags
        from mobius.integrations.gguf._tencent_q1_0 import is_tencent_q1_0_layout

        bits, block_size, is_sym = _detect_quant_params(gguf_model, gguf_arch)
        # Float zero-point only when actually using Tencent's native 2-bit form.
        float_zp = is_tencent_q1_0_layout(gguf_model) and flags.tencent_q1_0_use_native_2bit
        quantize_embeddings = _can_quantize_embedding(
            gguf_model,
            gguf_arch,
            bits=bits,
            block_size=block_size,
        )
        quantize_lm_head = (
            quantize_embeddings
            if config.tie_word_embeddings
            else _can_quantize_lm_head(gguf_model, gguf_arch)
        )
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=block_size,
                quant_method="gguf",
                sym=is_sym,
                float_zero_point=float_zp,
                quantize_embeddings=quantize_embeddings,
                quantize_lm_head=quantize_lm_head,
                tie_word_embeddings=quantize_lm_head and config.tie_word_embeddings,
            ),
        )
        logger.info(
            "Quantized mode: bits=%d, block_size=%d, symmetric=%s, "
            "float_zp=%s, embedding=%s, lm_head=%s",
            bits,
            block_size,
            is_sym,
            float_zp,
            quantize_embeddings,
            quantize_lm_head,
        )

    # 4. Look up module class and resolve task
    module_class = registry.get(model_type)
    if task is None:
        task = _default_task_for_model(model_type)

    # 5. Build ONNX graph
    module = module_class(config)
    pkg = build_from_module(module, config, task, execution_provider=execution_provider)
    logger.info(
        "Built ONNX graph for %s (%d components)",
        model_type,
        len(pkg),
    )

    # 6. Load tensors from GGUF → state_dict
    if keep_quantized:
        state_dict = _load_quantized_state_dict(gguf_model, gguf_arch, module, config)
    else:
        state_dict = _load_dequantized_state_dict(gguf_model, gguf_arch)

    logger.info(
        "Mapped %d state_dict entries from GGUF tensors",
        len(state_dict),
    )

    # 7. Apply architecture-specific tensor processors.
    # For the quantized path, only float tensors go through
    # process_tensors; quantized Q/K tensors were permuted in
    # _load_quantized_state_dict already.
    if keep_quantized:
        float_keys = {
            k
            for k in state_dict
            if not (
                k.endswith((".scales", ".zero_points")) or _is_quantized_weight(k, state_dict)
            )
        }
        float_dict = {k: state_dict[k] for k in float_keys}
        quant_dict = {k: state_dict[k] for k in state_dict if k not in float_keys}
        float_dict = process_tensors(float_dict, config)
        state_dict = {**float_dict, **quant_dict}
    else:
        state_dict = process_tensors(state_dict, config)

    # 7b. Normalize GGUF-specific weight shapes to match HF conventions.
    # This converts GGUF tensor quirks (stacked experts, 1D gates, 2D
    # conv weights, suffix artifacts) into the shapes that HF models
    # produce, so preprocess_weights only needs to handle HF→ONNX.
    state_dict = _normalize_gguf_weights(state_dict)

    # 8. Run model-specific preprocess_weights (HF → ONNX names)
    if hasattr(module, "preprocess_weights"):
        state_dict = module.preprocess_weights(state_dict)

    # 9. Apply weights to ONNX model
    prefix_map = getattr(module, "weight_prefix_map", None)
    pkg.apply_weights(state_dict, prefix_map=prefix_map)

    return pkg


def _is_quantized_weight(key: str, state_dict: dict) -> bool:
    """Check if a .weight key has a matching .scales companion."""
    if not key.endswith(".weight"):
        return False
    stem = key[: -len(".weight")]
    return f"{stem}.scales" in state_dict


def _normalize_gguf_weights(
    state_dict: dict,
) -> dict:
    """Normalize GGUF-specific weight shapes to match HF conventions.

    GGUF tensor mapping + dequantization produces weights that differ
    from HuggingFace in several ways. This function converts them so
    that ``preprocess_weights`` only needs to handle HF→ONNX mapping.

    Transforms applied:

    - **Stacked expert weights**: GGUF provides separate 3D tensors
      ``experts.{gate,up,down}_proj.weight`` with shape
      ``[num_experts, out, in]``. These are unpacked into per-expert
      ``experts.{i}.{proj}.weight`` tensors, matching the HF
      ``experts.down_proj`` format that ``preprocess_weights`` expects.
    - **1D shared_expert_gate**: GGUF stores as ``[hidden]``; HF/ONNX
      ``Linear(hidden, 1)`` expects ``[1, hidden]``.
    - **2D conv1d**: GGUF stores as ``[channels, kernel]``; depthwise
      ``Conv1d`` expects ``[channels, 1, kernel]``.
    - **dt_bias suffix**: GGUF ``ssm_dt.bias`` maps to
      ``dt_bias.bias`` after suffix splitting, but the model parameter
      is just ``dt_bias`` (an ``nn.Parameter``, not a module bias).
    """
    import torch

    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        # Stacked expert weights [num_experts, out, in] → per-expert
        unpacked = False
        for proj in ("gate_proj", "up_proj", "down_proj"):
            suffix = f".mlp.experts.{proj}.weight"
            if key.endswith(suffix) and value.dim() == 3:
                prefix = key[: -len(suffix)]
                for i in range(value.shape[0]):
                    result[f"{prefix}.mlp.experts.{i}.{proj}.weight"] = value[i]
                unpacked = True
                break
        if unpacked:
            continue

        # 1D shared_expert_gate → [1, hidden]
        if key.endswith(".mlp.shared_expert_gate.weight") and value.dim() == 1:
            result[key] = value.unsqueeze(0)
            continue

        # 2D conv1d → [channels, 1, kernel]
        if key.endswith(".conv1d.weight") and value.dim() == 2:
            result[key] = value.unsqueeze(1)
            continue

        # dt_bias.bias → dt_bias (nn.Parameter, not module bias)
        if key.endswith(".dt_bias.bias"):
            result[key[: -len(".bias")]] = value
            continue

        result[key] = value

    return result


def _detect_quant_params(gguf_model, gguf_arch: str) -> tuple[int, int, bool]:
    """Detect the common MatMulNBits target for GGUF projection weights.

    Q4_K-containing mixed presets target 4-bit, block-32 asymmetric
    MatMulNBits. Other files use the most common directly repackable type.

    Returns:
        ``(bits, block_size, is_symmetric)`` tuple.

    Raises:
        ValueError: If no mapped or repackable weight tensors are found.
    """
    from gguf import GGMLQuantizationType

    from mobius.integrations.gguf._repacker import can_repack, repack_quant_params
    from mobius.integrations.gguf._tencent_q1_0 import is_tencent_q1_0_layout
    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )

    # Symmetry of each supported GGUF quantization type.
    #
    # Mainline Q1_0 (1-bit binary) is repacked into 2-bit MatMulNBits
    # with zp=1 — see _repack_q1_0. Tencent's custom Q1_0 (2-bit SEQ,
    # 512-elt blocks) is inflated to 4-bit MatMulNBits with zp=3 — see
    # parse_tencent_q1_0_tensor — because the ORT CPU unpacked-float-zp
    # path is currently only implemented for bits=4, and the half-integer
    # SEQ offset 1.5 cannot be expressed with integer zp at bits=2.
    type_symmetry: dict = {
        GGMLQuantizationType.Q4_0: True,
        GGMLQuantizationType.Q4_1: False,
        GGMLQuantizationType.Q4_K: False,
        GGMLQuantizationType.Q8_0: True,
        GGMLQuantizationType.Q1_0: False,
    }

    counts: Counter = Counter()
    for name, _raw, qtype, _shape in gguf_model.tensor_items_raw():
        hf_name = map_gguf_to_hf_names(name, gguf_arch)
        if hf_name is None or not hf_name.endswith(".weight"):
            continue
        counts[qtype] += 1

    if not counts:
        raise ValueError(
            "No mapped weight tensors found in GGUF file. "
            "Use keep_quantized=False for dequantized import."
        )

    # Q4_K_M is deliberately a mixed preset. Depending on tensor dimensions
    # and importance it may contain mostly Q5_0 plus Q4_K, Q6_K, and Q8_0.
    # The presence of Q4_K identifies the desired 4-bit MatMulNBits target;
    # choosing only among already-repackable types would incorrectly select
    # Q8_0 for the official Qwen2.5-0.5B Q4_K_M file.
    if GGMLQuantizationType.Q4_K in counts:
        dominant = GGMLQuantizationType.Q4_K
    else:
        repackable_counts = Counter(
            {
                qtype: count
                for qtype, count in counts.items()
                if can_repack(qtype.value if hasattr(qtype, "value") else qtype)
            }
        )
        if not repackable_counts:
            raise ValueError(
                "No repackable quantized tensors found in GGUF file. "
                "Use keep_quantized=False for dequantized import."
            )
        dominant = repackable_counts.most_common(1)[0][0]
    dominant_value = dominant.value if hasattr(dominant, "value") else dominant
    params = repack_quant_params(dominant_value)
    assert params is not None
    bits, block_size = params
    is_sym = type_symmetry[dominant]

    # Tencent Q1_0 files reuse the Q1_0 type id but ship a different
    # on-disk layout (2-bit SEQ, 512-element blocks, fp16 scale per block).
    # Override the mainline defaults so the resulting QuantizedLinear
    # matches what parse_tencent_q1_0_tensor produces (4-bit packed).
    if dominant == GGMLQuantizationType.Q1_0 and is_tencent_q1_0_layout(gguf_model):
        # See _tencent_q1_0.py — the bits/zp flavour depends on a flag:
        #   default (fast):  bits=4 packed-uint8 zp=3 (inflated codebook)
        #   opt-in (small):  bits=2 float zp=1.5 (native SEQ layout)
        from mobius._flags import flags

        if flags.tencent_q1_0_use_native_2bit:
            bits, block_size, is_sym = 2, 128, False
        else:
            bits, block_size, is_sym = 4, 128, False

    logger.info(
        "Dominant GGUF quant type: %s (%d tensors, bits=%d, block_size=%d)",
        dominant,
        counts[dominant],
        bits,
        block_size,
    )
    return bits, block_size, is_sym


def _can_quantize_embedding(
    gguf_model,
    gguf_arch: str,
    *,
    bits: int,
    block_size: int,
) -> bool:
    """Return whether the token embedding can use GatherBlockQuantized.

    The graph has one quantization configuration shared by its quantized
    modules. Preserve the GGUF embedding only when its repacked representation
    uses the same bit width and block size as the projection weights.
    """
    from mobius.integrations.gguf._repacker import repack_quant_params
    from mobius.integrations.gguf._tencent_q1_0 import is_tencent_q1_0_layout
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    # Tencent files reuse the Q1_0 type id for a custom layout that gguf-py
    # cannot size correctly. Embeddings from those files must stay dequantized.
    if is_tencent_q1_0_layout(gguf_model):
        return False

    for tensor in gguf_model._reader.tensors:
        if map_gguf_to_hf_names(tensor.name, gguf_arch) != "model.embed_tokens.weight":
            continue
        shape = tuple(reversed(tensor.shape))
        if len(shape) != 2:
            return False
        qtype = tensor.tensor_type
        qtype_val = qtype.value if hasattr(qtype, "value") else qtype
        return repack_quant_params(qtype_val) == (bits, block_size)
    return False


def _can_quantize_lm_head(gguf_model, gguf_arch: str) -> bool:
    """Return whether an untied GGUF output head can be kept quantized."""
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    supported_types = {
        "Q1_0",
        "Q2_K",
        "Q3_K",
        "Q4_0",
        "Q4_1",
        "Q4_K",
        "Q5_0",
        "Q5_1",
        "Q5_K",
        "Q6_K",
        "Q8_0",
    }
    for name, _raw, qtype, shape in gguf_model.tensor_items_raw():
        if map_gguf_to_hf_names(name, gguf_arch) != "lm_head.weight":
            continue
        return len(shape) == 2 and getattr(qtype, "name", None) in supported_types
    return False


def _require_supported_requantization(
    *,
    bits: int,
    block_size: int,
    tensor_name: str,
) -> None:
    if bits != 4 or block_size != 32:
        raise ValueError(
            "keep_quantized MatMulNBits requantization currently supports only "
            f"4-bit/block-32 targets; got bits={bits} block={block_size} "
            f"for tensor {tensor_name}. Use keep_quantized=False or a "
            "4-bit/block-32 target."
        )


def _load_dequantized_state_dict(
    gguf_model,
    gguf_arch: str,
) -> dict:
    """Load all tensors dequantized to float (Phase 1 path)."""
    import numpy as np
    import torch

    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )

    state_dict = {}
    for gguf_name, np_array in tqdm.tqdm(
        gguf_model.tensor_items(),
        desc="Dequantizing tensors",
        total=len(gguf_model._tensor_index),
    ):
        hf_name = map_gguf_to_hf_names(gguf_name, gguf_arch)
        if hf_name is not None:
            # F32/F16 tensors are mmap'd read-only views; make
            # writable so PyTorch can mutate if needed.
            if not np_array.flags.writeable:
                np_array = np.array(np_array)
            state_dict[hf_name] = torch.from_numpy(np_array)
        else:
            logger.warning("Unmapped GGUF tensor: %s (skipped)", gguf_name)
    return state_dict


def _load_quantized_state_dict(
    gguf_model,
    gguf_arch: str,
    module,
    config,
) -> dict:
    """Load tensors, normalizing quantized projections to MatMulNBits.

    Projection weights (Q/K/V/O, MLP, and a quantized output head) are
    converted to the graph's common MatMulNBits format, and token embeddings
    to GatherBlockQuantized format. Mixed or unsupported source types are
    dequantized and requantized when they do not match that target. Norms
    and other non-linear tensors remain dequantized.

    For llama-family models, quantized Q/K weights receive the
    row-level reverse-permutation that ``process_tensors`` would
    normally apply.
    """
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType, dequantize

    from mobius.components import QuantizedEmbedding, QuantizedLinear
    from mobius.integrations.gguf._repacker import (
        can_repack,
        repack_dequantized_tensor,
        repack_gguf_tensor,
    )
    from mobius.integrations.gguf._tencent_q1_0 import (
        is_tencent_q1_0_layout,
        parse_tencent_q1_0_tensor,
    )
    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )
    from mobius.integrations.gguf._tensor_processors import (
        _reverse_permute,
    )

    # Collect module paths that use QuantizedLinear so we know
    # which .weight parameters should receive repacked data.
    quantized_stems = set()
    quantized_embedding_stems = set()
    for mod_name, mod in module.named_modules():
        if isinstance(mod, QuantizedLinear):
            quantized_stems.add(mod_name)
        elif isinstance(mod, QuantizedEmbedding):
            quantized_embedding_stems.add(mod_name)

    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    model_type = getattr(config, "model_type", None)

    # Detect Tencent's non-mainline Q1_0 layout once per file. Reading
    # such tensors requires a custom parser keyed on the explicit
    # per-tensor file offset (mainline byte sizes are wrong).
    tencent_q1_0 = is_tencent_q1_0_layout(gguf_model)
    if tencent_q1_0:
        gguf_path = str(gguf_model._path)
        data_section_offset = gguf_model._reader.data_offset
        tensors_by_name = {t.name: t for t in gguf_model._reader.tensors}
        logger.info(
            "Detected Tencent Q1_0 layout (block_size=512, 2-bit SEQ); "
            "using custom per-tensor parser"
        )

    state_dict: dict[str, torch.Tensor] = {}
    n_repacked = 0
    n_requantized = 0
    target_bits = config.quantization.bits
    target_block_size = config.quantization.group_size
    target_symmetric = config.quantization.sym

    for gguf_name, raw, qtype, np_shape in tqdm.tqdm(
        gguf_model.tensor_items_raw(),
        desc="Repacking tensors",
        total=len(gguf_model._tensor_index),
    ):
        hf_name = map_gguf_to_hf_names(gguf_name, gguf_arch)
        if hf_name is None:
            logger.warning("Unmapped GGUF tensor: %s (skipped)", gguf_name)
            continue

        # Determine the int value of the quant type for can_repack
        qtype_val = qtype.value if hasattr(qtype, "value") else qtype

        # Repack every target QuantizedLinear weight. Mixed GGUF presets
        # otherwise leave unsupported source types as full float matrices,
        # which cannot fit the graph's packed MatMulNBits initializer shape.
        stem = hf_name[: -len(".weight")] if hf_name.endswith(".weight") else None
        is_tencent_q1_0_tensor = tencent_q1_0 and qtype == GGMLQuantizationType.Q1_0
        is_quantized_embedding = stem is not None and stem in quantized_embedding_stems
        should_repack = stem is not None and (
            stem in quantized_stems or is_quantized_embedding
        )

        if should_repack:
            if is_tencent_q1_0_tensor:
                repacked = parse_tencent_q1_0_tensor(
                    gguf_path,
                    data_section_offset,
                    tensors_by_name[gguf_name],
                )
            elif can_repack(qtype_val):
                shape_2d = (int(np_shape[0]), int(np_shape[1]))
                repacked = repack_gguf_tensor(
                    raw.ravel().view(np.uint8),
                    qtype_val,
                    shape_2d,
                )
                if repacked.bits != target_bits or repacked.block_size != target_block_size:
                    _require_supported_requantization(
                        bits=target_bits,
                        block_size=target_block_size,
                        tensor_name=hf_name,
                    )
                    values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                    repacked = repack_dequantized_tensor(
                        values,
                        bits=target_bits,
                        block_size=target_block_size,
                        symmetric=target_symmetric,
                    )
                    n_requantized += 1
            else:
                _require_supported_requantization(
                    bits=target_bits,
                    block_size=target_block_size,
                    tensor_name=hf_name,
                )
                values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                repacked = repack_dequantized_tensor(
                    values,
                    bits=target_bits,
                    block_size=target_block_size,
                    symmetric=target_symmetric,
                )
                n_requantized += 1
            w = torch.from_numpy(repacked.weight)
            s = torch.from_numpy(repacked.scales)

            # Apply Q/K row permutation to quantized tensors
            # (same transform as _process_llama, on all arrays). Only
            # llama-family archs use the interleaved-rope permute; Qwen
            # and others must NOT be permuted.
            if _needs_qk_permute(hf_name, num_heads, num_kv_heads, model_type):
                n_head = (
                    num_heads
                    if ".q_proj." in hf_name or ".qkv_proj." in hf_name
                    else num_kv_heads
                )
                w = _reverse_permute(w, n_head)
                s = _reverse_permute(s, n_head)

            if is_quantized_embedding:
                state_dict[f"{stem}.qweight"] = w.reshape(w.shape[0], -1)
            else:
                state_dict[hf_name] = w
            state_dict[f"{stem}.scales"] = s
            if repacked.zero_points is not None:
                zp = torch.from_numpy(repacked.zero_points)
                if _needs_qk_permute(hf_name, num_heads, num_kv_heads, model_type):
                    zp = _reverse_permute(zp, n_head)
                state_dict[f"{stem}.zero_points"] = zp
            n_repacked += 1
        else:
            # Dequantize to float
            if qtype in (
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
            ):
                arr = gguf_model.get_tensor(gguf_name)
                # F32/F16 tensors are mmap'd read-only views
                if not arr.flags.writeable:
                    arr = np.array(arr)
            else:
                arr = dequantize(raw, qtype).reshape(np_shape)
            state_dict[hf_name] = torch.from_numpy(arr)

    logger.info(
        "Loaded %d state_dict entries (%d GGUF tensors repacked for quantized ops, "
        "%d requantized from mixed source types)",
        len(state_dict),
        n_repacked,
        n_requantized,
    )
    return state_dict


def _needs_qk_permute(
    hf_name: str,
    num_heads: int | None,
    num_kv_heads: int | None,
    model_type: str | None = None,
) -> bool:
    """Check if this tensor needs Q/K reverse-permutation.

    Two conditions must hold: the tensor must be a Q/K projection weight,
    AND the model architecture must actually use llama.cpp's
    interleaved-rope permute. Name-based gating alone is insufficient —
    Qwen2/Qwen3 use ``.q_proj.``/``.k_proj.`` names too but store Q/K in
    plain HF order (NEOX rope) and must NOT be permuted, or their
    attention heads get scrambled and the model emits garbage.
    """
    from mobius.integrations.gguf._tensor_processors import (
        needs_llama_qk_permute,
    )

    if num_heads is None or num_kv_heads is None:
        return False
    if not needs_llama_qk_permute(model_type):
        return False
    return (
        ".q_proj." in hf_name or ".k_proj." in hf_name or ".qkv_proj." in hf_name
    ) and hf_name.endswith(".weight")
