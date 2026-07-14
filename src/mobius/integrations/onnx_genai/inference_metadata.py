# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate onnx-genai ``inference_metadata.yaml`` sidecars."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

import onnx_ir as ir

from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)

_DTYPE_NAMES = {
    ir.DataType.FLOAT: "float32",
    ir.DataType.FLOAT16: "float16",
    ir.DataType.BFLOAT16: "bfloat16",
}
_DEFAULT_MAX_SEQUENCE_LENGTH = 4096

# Tokenizer files to copy alongside the onnx-genai package.
# Mirrors the list in mobius.integrations.ort_genai.auto_export.
_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",  # SentencePiece
    "added_tokens.json",
    "merges.txt",  # BPE
    "vocab.json",  # BPE
]


def _positive_int(config: object, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"onnx-genai inference metadata requires a positive {name}, got {value!r}."
        )
    return value


def _kv_dtype(config: object) -> str:
    dtype = getattr(config, "dtype", None)
    try:
        return _DTYPE_NAMES[dtype]
    except KeyError:
        raise ValueError(
            "onnx-genai inference metadata supports float32, float16, or bfloat16 "
            f"KV caches, got {dtype!r}."
        ) from None


def _max_sequence_length(config: object, requested: int | None) -> int:
    model_max = _positive_int(config, "max_position_embeddings")
    if requested is None:
        return min(model_max, _DEFAULT_MAX_SEQUENCE_LENGTH)
    if not isinstance(requested, int) or requested <= 0:
        raise ValueError(
            f"onnx-genai max_sequence_length must be a positive integer, got {requested!r}."
        )
    if requested > model_max:
        raise ValueError(
            f"onnx-genai max_sequence_length {requested} exceeds the model limit {model_max}."
        )
    return requested


def _target_layers_by_type(layer_types: list[str]) -> dict[str, list[int]]:
    """Return ``{layer_type: [0-based indices]}`` derived from *layer_types*.

    Gemma4-Assistant requires ``num_kv_shared_layers == num_hidden_layers``,
    so every assistant layer index ``i`` corresponds 1-to-1 with a target-model
    KV slot of the same type.  The runtime uses these index lists to route each
    target layer's KV tensors into the correct ``shared_kv.{type}`` input of
    the assistant ONNX graph.

    Example — E2B-it-assistant (4 layers, sliding_window_pattern=4):
        layer_types = ["sliding_attention", "sliding_attention",
                       "sliding_attention", "full_attention"]
        → {"sliding_attention": [0, 1, 2], "full_attention": [3]}
    """
    result: dict[str, list[int]] = {}
    for i, lt in enumerate(layer_types):
        result.setdefault(lt, []).append(i)
    return result


def _speculative_block(
    config: object,
    *,
    model_path: str = "model.onnx",
    num_speculative_tokens: int = 3,
) -> dict[str, Any]:
    """Build the ``speculative`` metadata dict for a Gemma4-Assistant config.

    Args:
        config: A ``Gemma4AssistantConfig`` instance.
        model_path: Path to the assistant ``model.onnx`` relative to the
            package root (the directory containing ``inference_metadata.yaml``).
            Defaults to ``"model.onnx"`` for single-component packages.
        num_speculative_tokens: Number of tokens the assistant proposes per
            target step.  Defaults to 3.
    """
    layer_types: list[str] = getattr(config, "layer_types", None) or []
    layers_by_type = _target_layers_by_type(layer_types)

    backbone_hidden_size = _positive_int(config, "backbone_hidden_size")
    vocab_size = _positive_int(config, "vocab_size")

    shared_kv: list[dict[str, Any]] = []
    # Emit types in the order they first appear in layer_types so the
    # output is deterministic and matches the graph I/O declaration order.
    seen: set[str] = set()
    for lt in layer_types:
        if lt not in seen:
            seen.add(lt)
            shared_kv.append({"name": lt, "target_layers": layers_by_type[lt]})

    return {
        "proposal_type": "shared_kv",
        "num_speculative_tokens": num_speculative_tokens,
        "model": model_path,
        "backbone_hidden_size": backbone_hidden_size,
        "vocab_size": vocab_size,
        "projected_state_output": "projected_state",
        "logits_output": "logits",
        "shared_kv": shared_kv,
    }


def _folded_shared_kv_groups(target_config: object) -> list[dict[str, Any]]:
    """Derive ``shared_kv`` groups from a **target** decoder config.

    Unlike :func:`_target_layers_by_type` (which reads an *assistant* config
    where every layer maps 1-to-1 to a KV slot), this folds the target's KV
    sharing: the last ``num_kv_shared_layers`` layers borrow K,V from an earlier
    layer of the same type and are NOT exported, so only the first
    ``num_hidden_layers - num_kv_shared_layers`` layers have their own KV cache
    entry.  ``target_layers`` therefore index the target's **exported** KV
    entries (post-folding), which is exactly what the runtime slicer consumes.

    The runtime keys each group off ``target_layers.last()``; because a
    KV-shared layer reuses the *last* non-shared layer of its type, listing all
    exported entries of a type puts that source layer last automatically.  This
    is fully generic: it derives purely from ``layer_types`` +
    ``num_kv_shared_layers``, so it scales to any layer count (e.g. 12B).
    """
    layer_types: list[str] = getattr(target_config, "layer_types", None) or []
    if not layer_types:
        raise ValueError(
            "merged onnx-genai metadata requires the target config to declare "
            "layer_types to derive shared_kv groups."
        )
    num_hidden = getattr(target_config, "num_hidden_layers", len(layer_types))
    num_shared = getattr(target_config, "num_kv_shared_layers", 0) or 0
    num_kv_layers = num_hidden - num_shared
    exported = layer_types[:num_kv_layers]

    by_type: dict[str, list[int]] = {}
    for i, lt in enumerate(exported):
        by_type.setdefault(lt, []).append(i)

    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lt in exported:
        if lt not in seen:
            seen.add(lt)
            groups.append({"name": lt, "target_layers": by_type[lt]})
    return groups


def generate_merged_inference_metadata(
    target_config: object,
    assistant_config: object,
    *,
    max_sequence_length: int | None = None,
    num_speculative_tokens: int = 3,
    assistant_model_path: str = "assistant/model.onnx",
) -> dict[str, Any]:
    """Build merged metadata for a single-model target + shared-KV assistant.

    The ``model:`` block describes the **target** decoder (``target_config``);
    the ``speculative:`` block wires in the draft ``assistant_config`` with
    ``shared_kv`` groups derived from the *target's* exported KV layers
    (:func:`_folded_shared_kv_groups`), not the assistant's own layers.  This is
    the correct wiring for a merged package where the target and assistant have
    different layer counts (e.g. E2B: 15 exported target KV layers vs 4
    assistant layers).

    Stays model-agnostic: all shapes/wiring are read from the two configs.
    """
    metadata = generate_inference_metadata(
        target_config, max_sequence_length=max_sequence_length
    )
    # target_config is not a Gemma4AssistantConfig, so no speculative block was
    # added above; attach the merged one here.
    metadata["speculative"] = {
        "proposal_type": "shared_kv",
        "num_speculative_tokens": num_speculative_tokens,
        "model": assistant_model_path,
        "backbone_hidden_size": _positive_int(assistant_config, "backbone_hidden_size"),
        "vocab_size": _positive_int(assistant_config, "vocab_size"),
        "projected_state_output": "projected_state",
        "logits_output": "logits",
        "shared_kv": _folded_shared_kv_groups(target_config),
    }
    return metadata


def write_merged_inference_metadata(
    target_config: object,
    assistant_config: object,
    directory: str,
    *,
    max_sequence_length: int | None = None,
    num_speculative_tokens: int = 3,
    assistant_model_path: str = "assistant/model.onnx",
    hf_model_id: str | None = None,
    local_config_dir: str | None = None,
) -> dict[str, str]:
    """Write a merged ``inference_metadata.yaml`` (target model + assistant).

    See :func:`generate_merged_inference_metadata`.  Also copies tokenizer files
    from *local_config_dir* or *hf_model_id* when provided.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as file:
        file.write(
            _to_yaml(
                generate_merged_inference_metadata(
                    target_config,
                    assistant_config,
                    max_sequence_length=max_sequence_length,
                    num_speculative_tokens=num_speculative_tokens,
                    assistant_model_path=assistant_model_path,
                )
            )
        )
    artifacts: dict[str, str] = {"inference_metadata": path}
    if local_config_dir:
        for name in _copy_tokenizer_files_from_local(local_config_dir, directory):
            artifacts[name] = os.path.join(directory, name)
    elif hf_model_id:
        for name in _copy_tokenizer_files_from_hf(hf_model_id, directory):
            artifacts[name] = os.path.join(directory, name)
    return artifacts


def generate_inference_metadata(
    config: object,
    *,
    max_sequence_length: int | None = None,
    num_speculative_tokens: int = 3,
    assistant_model_path: str = "model.onnx",
) -> dict[str, Any]:
    """Map a decoder config to metadata with a conservative serving KV capacity.

    When *config* is a ``Gemma4AssistantConfig``, an additional ``speculative``
    block is included describing the draft-model contract (shared KV layout,
    projection dimensions, output names).
    """
    from mobius._configs import Gemma4AssistantConfig

    num_attention_heads = _positive_int(config, "num_attention_heads")
    num_kv_heads = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    max_sequence_length = _max_sequence_length(config, max_sequence_length)
    kv_dtype = _kv_dtype(config)

    is_gqa = num_kv_heads != num_attention_heads
    capabilities = ["grouped_query_attention" if is_gqa else "multi_head_attention"]

    attention: dict[str, Any] = {
        "type": "group_query_attention" if is_gqa else "multi_head_attention",
        "num_kv_heads": num_kv_heads,
        "num_attention_heads": num_attention_heads,
        "head_dim": head_dim,
    }
    sliding_window = getattr(config, "sliding_window", None)
    if isinstance(sliding_window, int) and sliding_window > 0:
        attention["sliding_window"] = sliding_window

    metadata: dict[str, Any] = {
        "required_capabilities": capabilities,
        "model": {
            "attention": attention,
            "max_sequence_length": max_sequence_length,
            "runtime_configurable": {"kv_cache": {"dtype": [kv_dtype]}},
        },
        "kv_cache": {"native_dtype": kv_dtype},
    }

    if isinstance(config, Gemma4AssistantConfig):
        metadata["speculative"] = _speculative_block(
            config,
            model_path=assistant_model_path,
            num_speculative_tokens=num_speculative_tokens,
        )

    return metadata


def _to_yaml(metadata: dict[str, Any]) -> str:
    capabilities = metadata["required_capabilities"]
    attention = metadata["model"]["attention"]
    kv_dtypes = metadata["model"]["runtime_configurable"]["kv_cache"]["dtype"]

    lines = ["required_capabilities:"]
    if capabilities:
        lines.extend(f"  - {capability}" for capability in capabilities)
    else:
        lines[-1] += " []"

    lines.extend(
        [
            "model:",
            "  attention:",
            f"    type: {attention['type']}",
            f"    num_kv_heads: {attention['num_kv_heads']}",
            f"    num_attention_heads: {attention['num_attention_heads']}",
            f"    head_dim: {attention['head_dim']}",
        ]
    )
    if "sliding_window" in attention:
        lines.append(f"    sliding_window: {attention['sliding_window']}")
    lines.extend(
        [
            f"  max_sequence_length: {metadata['model']['max_sequence_length']}",
            "  runtime_configurable:",
            "    kv_cache:",
            "      dtype:",
            *(f"        - {dtype}" for dtype in kv_dtypes),
            "kv_cache:",
            f"  native_dtype: {metadata['kv_cache']['native_dtype']}",
        ]
    )

    if "speculative" in metadata:
        spec = metadata["speculative"]
        lines.extend(
            [
                "speculative:",
                f"  proposal_type: {spec['proposal_type']}",
                f"  num_speculative_tokens: {spec['num_speculative_tokens']}",
                f"  model: {spec['model']}",
                f"  backbone_hidden_size: {spec['backbone_hidden_size']}",
                f"  vocab_size: {spec['vocab_size']}",
                f"  projected_state_output: {spec['projected_state_output']}",
                f"  logits_output: {spec['logits_output']}",
                "  shared_kv:",
            ]
        )
        for slot in spec["shared_kv"]:
            indices = ", ".join(str(i) for i in slot["target_layers"])
            lines.append(f"    - name: {slot['name']}")
            lines.append(f"      target_layers: [{indices}]")

    return "\n".join(lines) + "\n"


def _copy_tokenizer_files_from_hf(model_id: str, output_dir: str) -> list[str]:
    """Download tokenizer files from HuggingFace Hub and copy to *output_dir*.

    Returns the list of filenames successfully copied.  Silently skips files
    that are not present on the Hub (not all tokenizer variants ship all
    files).
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed; tokenizer files will not be copied."
        )
        return []

    copied: list[str] = []
    for filename in _TOKENIZER_FILES:
        try:
            src = hf_hub_download(model_id, filename)
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            copied.append(filename)
        except (EntryNotFoundError, OSError):
            continue
    return copied


def _copy_tokenizer_files_from_local(source_dir: str, output_dir: str) -> list[str]:
    """Copy tokenizer files from a local directory to *output_dir*.

    Silently skips absent files.  Returns the list of filenames copied.
    """
    if not os.path.isdir(source_dir):
        logger.warning(
            "Local tokenizer source directory does not exist: %s — "
            "no tokenizer files copied.",
            source_dir,
        )
        return []
    copied: list[str] = []
    for filename in _TOKENIZER_FILES:
        src = os.path.join(source_dir, filename)
        if os.path.isfile(src):
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            copied.append(filename)
    return copied


def write_inference_metadata(
    pkg: ModelPackage,
    directory: str,
    *,
    max_sequence_length: int | None = None,
    num_speculative_tokens: int = 3,
    assistant_model_path: str = "model.onnx",
    hf_model_id: str | None = None,
    local_config_dir: str | None = None,
) -> dict[str, str]:
    """Write ``inference_metadata.yaml`` for an already-built model package.

    Also copies tokenizer files when *hf_model_id* or *local_config_dir* is
    provided (mirrors the behaviour of
    :func:`mobius.integrations.ort_genai.write_ort_genai_config`).

    Args:
        pkg: Built :class:`~mobius._model_package.ModelPackage` with
            ``config`` set.
        directory: Output directory (created if needed).
        max_sequence_length: Override the model's maximum sequence length
            written into the metadata.  Must be a positive integer ≤ the
            model's ``max_position_embeddings``.
        num_speculative_tokens: Number of speculative tokens the assistant
            proposes per step (only used when the config is
            ``Gemma4AssistantConfig``).
        assistant_model_path: Relative path to the assistant ``model.onnx``
            from the package root, written into ``speculative.model``
            (only used when the config is ``Gemma4AssistantConfig``).
        hf_model_id: HuggingFace model ID.  When provided the tokenizer files
            are downloaded and placed alongside ``inference_metadata.yaml``.
        local_config_dir: Local directory containing the model config.
            Tokenizer files are copied from here when provided (takes
            precedence over *hf_model_id* for tokenizer discovery).

    Returns:
        Dict mapping artifact names to their absolute paths
        (``"inference_metadata"`` plus any copied tokenizer filenames).
    """
    config = getattr(pkg, "config", None)
    if config is None:
        raise ValueError(
            "write_inference_metadata requires ModelPackage.config to be set. "
            "This is set automatically when building with mobius.build()."
        )

    os.makedirs(directory, exist_ok=True)

    # --- inference_metadata.yaml ---
    path = os.path.join(directory, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as file:
        file.write(
            _to_yaml(
                generate_inference_metadata(
                    config,
                    max_sequence_length=max_sequence_length,
                    num_speculative_tokens=num_speculative_tokens,
                    assistant_model_path=assistant_model_path,
                )
            )
        )
    artifacts: dict[str, str] = {"inference_metadata": path}

    # --- tokenizer files ---
    if local_config_dir:
        copied = _copy_tokenizer_files_from_local(local_config_dir, directory)
        for name in copied:
            artifacts[name] = os.path.join(directory, name)
    elif hf_model_id:
        copied = _copy_tokenizer_files_from_hf(hf_model_id, directory)
        for name in copied:
            artifacts[name] = os.path.join(directory, name)

    return artifacts
