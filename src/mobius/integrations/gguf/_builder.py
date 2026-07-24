# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF → ONNX build pipeline.

Converts ``.gguf`` model files to ONNX using the standard build
pipeline.  Two modes:

- **Dequantized** (default): All quantized tensors are dequantized to
  float.  Simple, but loses the compression benefit of quantization.
- **Quantized** (``keep_quantized=True``): Affine linear-layer weights are
  repacked into MatMulNBits format and token embeddings into
  GatherBlockQuantized format. Runtime-supported native IQ/MXFP4 projection
  blocks are preserved for BlockQuantizedMatMul. Mixed presets such as Q4_K_M are
  normalized to one affine layout. Other tensors are dequantized.
"""

from __future__ import annotations

__all__ = ["build_from_gguf"]

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import tqdm
from huggingface_hub import HfApi, hf_hub_download

from mobius._model_package import ModelPackage

if TYPE_CHECKING:
    from mobius.tasks import ModelTask

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
    mmproj: str | Path | None = None,
    static_cache: bool = False,
    max_seq_len: int | None = None,
    include_audio: bool = False,
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a GGUF file.

    1. Parse GGUF metadata → :class:`ArchitectureConfig`
    2. Look up the model class and task from the registry
    3. Build the ONNX graph (standard ``build_from_module`` pipeline)
    4. Map GGUF tensor names → HuggingFace names
    5. Replace native-block projection modules when present
    6. Apply architecture-specific tensor processors
    7. Run ``preprocess_weights()`` (HF → ONNX name mapping)
    8. Apply weights to the ONNX model

    When *keep_quantized* is ``True``, supported affine tensors are repacked
    into MatMulNBits format, while runtime-supported native IQ/MXFP4 projection
    blocks are retained byte-for-byte for BlockQuantizedMatMul.

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
        mmproj: Optional path (or HF ref) to a companion ``clip``
            multimodal-projector GGUF. When set, this becomes the single
            entry point for a multimodal build: the text GGUF and the
            mmproj vision/audio encoder are fused into one multimodal
            :class:`ModelPackage` (delegates to
            :func:`build_gemma4_vlm_from_gguf`).
        static_cache: When ``True``, build with a pre-allocated static KV
            cache (fixed-width buffers written in place via ``TensorScatter``)
            instead of the default dynamic concat-grow cache. Produces a
            fully static-shaped graph, which is required by fixed-shape
            runtimes such as the QNN HTP backend. Cannot be combined with an
            explicit *task* override.
        max_seq_len: Maximum sequence length for the static cache buffers.
            Only used when ``static_cache=True``. Defaults to the model's
            ``max_position_embeddings``.
        include_audio: When ``True`` and ``mmproj`` is set, also build the
            experimental audio encoder.

    Returns:
        A :class:`ModelPackage` containing the built model(s).

    Raises:
        ImportError: If the ``gguf`` package is not installed.
        FileNotFoundError: If the GGUF file does not exist.
        KeyError: If the GGUF architecture is not in the registry.
        ValueError: If *static_cache* is combined with an explicit *task*.
    """
    import dataclasses

    # A companion mmproj GGUF turns this into a multimodal build: the text +
    # vision/audio encoders are assembled by the dedicated VLM builder. Keep
    # build_from_gguf as the single public entry point (text-only or multimodal).
    if mmproj is not None:
        if static_cache:
            raise ValueError("static_cache=True is not supported with a companion mmproj.")
        from mobius.integrations.gguf._mmproj import build_gemma4_vlm_from_gguf

        return build_gemma4_vlm_from_gguf(
            gguf_path,
            mmproj,
            dtype=dtype,
            execution_provider=execution_provider,
            keep_quantized=keep_quantized,
            include_audio=include_audio,
        )

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

    if static_cache and task is not None:
        raise ValueError(
            "static_cache=True cannot be combined with an explicit task "
            "override; the static cache is wired through CausalLMTask."
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
    resolved_task: str | ModelTask
    if static_cache:
        from mobius.tasks import CausalLMTask

        resolved_task = CausalLMTask(static_cache=True, max_seq_len=max_seq_len)
    elif task is None:
        resolved_task = _default_task_for_model(model_type)
    else:
        resolved_task = task

    # 5. Build ONNX graph
    module = module_class(config)
    if keep_quantized:
        _replace_native_block_linears(module, gguf_model, gguf_arch)
    pkg = build_from_module(
        module, config, resolved_task, execution_provider=execution_provider
    )
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
                k.endswith((".scales", ".zero_points"))
                or _is_quantized_weight(k, state_dict)
                or _is_native_block_weight(k, state_dict)
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


def _is_native_block_weight(key: str, state_dict: dict) -> bool:
    """Check for a packed runtime-native GGUF weight."""
    from mobius.integrations.gguf._repacker import NATIVE_BLOCK_BYTE_SIZES

    if not key.endswith(".weight"):
        return False
    value = state_dict[key]
    return (
        value.dtype.is_floating_point is False
        and value.dim() == 3
        and value.shape[-1] in NATIVE_BLOCK_BYTE_SIZES
    )


def _native_block_spec(qtype):
    """Return the runtime-native layout for a GGUF quantization enum."""
    from mobius.integrations.gguf._repacker import native_block_spec

    qtype_val = qtype.value if hasattr(qtype, "value") else qtype
    return native_block_spec(qtype_val)


def _native_block_format(qtype) -> str | None:
    """Return the runtime format string for supported native GGUF blocks."""
    spec = _native_block_spec(qtype)
    return spec.format if spec is not None else None


def _native_block_target_stems(
    hf_name: str,
    np_shape: tuple[int, ...],
    available_stems: set[str],
) -> list[str]:
    """Map a GGUF weight to one or more native-block module stems."""
    if not hf_name.endswith(".weight"):
        return []
    stem = hf_name[: -len(".weight")]
    if stem in available_stems:
        return [stem]

    if len(np_shape) == 3 and ".experts." in stem:
        prefix, projection = stem.rsplit(".experts.", 1)
        for container in (f"{prefix}.experts", f"{prefix}.moe.experts"):
            candidates = [f"{container}.{i}.{projection}" for i in range(np_shape[0])]
            if all(candidate in available_stems for candidate in candidates):
                return candidates
    return []


def _replace_child_module(root, path: str, replacement) -> None:
    """Replace a named ONNXScript child module while retaining its graph name."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        try:
            parent = getattr(parent, part)
        except AttributeError as error:
            raise AttributeError(f"Module path {path!r} has no child {part!r}") from error
    child_name = parts[-1]
    try:
        old = getattr(parent, child_name)
    except AttributeError as error:
        raise AttributeError(f"Module path {path!r} has no child {child_name!r}") from error
    if hasattr(replacement, "_set_name") and hasattr(old, "name"):
        replacement._set_name(old.name)
    setattr(parent, child_name, replacement)


def _replace_native_block_linears(module, gguf_model, gguf_arch: str) -> None:
    """Swap MatMulNBits scaffolding for runtime-supported native linears."""
    from mobius.components import BlockQuantizedLinear, QuantizedLinear
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    module_map = dict(module.named_modules())
    quantized_stems = {
        name for name, child in module_map.items() if isinstance(child, QuantizedLinear)
    }
    replacements: dict[str, str] = {}
    for gguf_name, _raw, qtype, np_shape in gguf_model.tensor_items_raw():
        format_name = _native_block_format(qtype)
        if format_name is None:
            continue
        hf_name = map_gguf_to_hf_names(gguf_name, gguf_arch)
        if hf_name is None:
            continue
        for stem in _native_block_target_stems(hf_name, np_shape, quantized_stems):
            replacements[stem] = format_name

    for stem, format_name in replacements.items():
        old = module_map[stem]
        replacement = BlockQuantizedLinear(
            old._k,
            old._n,
            format=format_name,
            bias=old.bias is not None,
        )
        _replace_child_module(module, stem, replacement)

    if replacements:
        logger.info(
            "Preserving %d GGUF projection weights as runtime-native IQ/MXFP4 blocks",
            len(replacements),
        )


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

        # layer_scalar.weight → layer_scalar (Gemma4 per-layer output scale is an
        # nn.Parameter, not a module weight). GGUF stores it as
        # blk.{i}.layer_output_scale.weight, which the tensor mapping renames to
        # model.layers.{i}.layer_scalar.weight; strip the artefact .weight suffix.
        if key.endswith(".layer_scalar.weight"):
            result[key[: -len(".weight")]] = value
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

    native_counts = Counter(
        {qtype: count for qtype, count in counts.items() if _native_block_format(qtype)}
    )
    if native_counts:
        asymmetric_types = {"Q2_K", "Q4_1", "Q4_K", "Q5_1", "Q5_K"}
        is_sym = not any(
            getattr(qtype, "name", None) in asymmetric_types
            for qtype in counts
            if qtype not in native_counts
        )
        logger.info(
            "Native GGUF quant types present; using 4-bit/block-32 module "
            "scaffolding for non-native quantized tensors",
        )
        return 4, 32, is_sym

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
        "MXFP4",
        "IQ4_NL",
        "IQ4_XS",
        "IQ3_S",
        "IQ3_XXS",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ1_S",
        "IQ1_M",
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


def repack_gguf_weight_to_target(
    gguf_model,
    raw,
    qtype,
    np_shape,
    *,
    target_bits: int,
    target_block_size: int,
    target_symmetric: bool,
    tensor_name: str,
):
    """Repack one 2-D GGUF weight to the graph's common MatMulNBits target.

    Reuses the shared repacker machinery: a tensor whose native repacked layout
    already matches ``(target_bits, target_block_size)`` is repacked directly;
    otherwise it is dequantized and requantized to the target layout. This is
    the single-tensor building block reused by both the text-only
    (:func:`_load_quantized_state_dict`) and the multimodal quantized loaders.

    Args:
        gguf_model: The source :class:`GGUFModel`.
        raw: Raw tensor bytes (as returned by ``tensor_items_raw``).
        qtype: The GGUF quantization type of the tensor.
        np_shape: The tensor's logical ``(N, K)`` shape.
        target_bits: Target MatMulNBits bit width.
        target_block_size: Target MatMulNBits block size.
        target_symmetric: Whether the requantization path should omit
            zero-points (symmetric). Only used when requantizing.
        tensor_name: Name used for error messages.

    Returns:
        A ``RepackedTensor`` with the target ``(bits, block_size)`` layout.
    """
    import numpy as np

    from mobius.integrations.gguf._repacker import (
        can_repack,
        repack_dequantized_tensor,
        repack_gguf_tensor,
    )

    qtype_val = qtype.value if hasattr(qtype, "value") else qtype
    if can_repack(qtype_val):
        shape_2d = (int(np_shape[0]), int(np_shape[1]))
        repacked = repack_gguf_tensor(raw.ravel().view(np.uint8), qtype_val, shape_2d)
        if repacked.bits == target_bits and repacked.block_size == target_block_size:
            return repacked

    _require_supported_requantization(
        bits=target_bits,
        block_size=target_block_size,
        tensor_name=tensor_name,
    )
    values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
    return repack_dequantized_tensor(
        values,
        bits=target_bits,
        block_size=target_block_size,
        symmetric=target_symmetric,
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
    """Load tensors, preserving native blocks or normalizing to MatMulNBits.

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

    from mobius.components import BlockQuantizedLinear, QuantizedEmbedding, QuantizedLinear
    from mobius.integrations.gguf._repacker import (
        can_repack,
        preserve_native_blocks,
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
    native_block_stems: dict[str, str] = {}
    quantized_embedding_stems = set()
    for mod_name, mod in module.named_modules():
        if isinstance(mod, QuantizedLinear) or getattr(mod, "_gguf_quantized_linear", False):
            quantized_stems.add(mod_name)
        elif isinstance(mod, BlockQuantizedLinear):
            native_block_stems[mod_name] = mod._format
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

        native_targets = _native_block_target_stems(
            hf_name,
            np_shape,
            set(native_block_stems),
        )
        native_spec = _native_block_spec(qtype)
        if native_targets and native_spec is not None:
            n_out = int(np_shape[-2])
            k_in = int(np_shape[-1])
            packed = preserve_native_blocks(
                raw,
                qtype_val,
                (len(native_targets) * n_out, k_in),
            )
            packed = packed.reshape(
                len(native_targets),
                n_out,
                packed.shape[-2],
                native_spec.bytes,
            )
            for index, native_stem in enumerate(native_targets):
                w = torch.from_numpy(np.array(packed[index], copy=True))
                target_name = f"{native_stem}.weight"
                if _needs_qk_permute(
                    target_name,
                    num_heads,
                    num_kv_heads,
                    model_type,
                ):
                    n_head = (
                        num_heads
                        if ".q_proj." in target_name or ".qkv_proj." in target_name
                        else num_kv_heads
                    )
                    w = _reverse_permute(w, n_head)
                state_dict[target_name] = w
            n_repacked += len(native_targets)
        elif should_repack:
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
