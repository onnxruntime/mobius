# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared weight preprocessing utilities for HuggingFace → ONNX conversion.

These helpers handle common weight transformations that appear across
multiple model architectures, such as splitting fused QKV projections
and fused gate/up projections.

Note: Some models (InternLM, Phi3-Small) use grouped/interleaved QKV
layouts that require reshape-based splitting. Those are too
model-specific to generalize here and remain inline.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import torch

logger = logging.getLogger(__name__)


def merge_lora_weights(
    base_state_dict: dict[str, torch.Tensor],
    lora_state_dict: dict[str, torch.Tensor],
    *,
    default_alpha: float | None = None,
) -> dict[str, torch.Tensor]:
    """Merge LoRA adapter weights into base model weights.

    Detects ``*.lora_A.weight`` / ``*.lora_B.weight`` pairs in
    *lora_state_dict* and merges them into *base_state_dict* using::

        merged = base + (alpha / rank) * (B @ A)

    where ``rank = A.shape[0]`` and ``alpha`` is read from
    ``*.lora_A.alpha`` (a scalar tensor) or falls back to
    *default_alpha* (which defaults to ``rank`` if not provided).

    Keys in *lora_state_dict* that are not LoRA deltas are ignored.

    Args:
        base_state_dict: Base model weights (modified in-place).
        lora_state_dict: PEFT adapter weights containing
            ``lora_A.weight``, ``lora_B.weight``, and optionally
            ``lora_A.alpha`` tensors.
        default_alpha: Fallback scaling alpha when no per-layer alpha
            tensor is found.  Defaults to ``rank`` (i.e. scale = 1.0).

    Returns:
        *base_state_dict* with LoRA deltas merged in.
    """
    # Collect LoRA A matrices keyed by their base weight name.
    # e.g. "model.layers.0.self_attn.q_proj.lora_A.weight"
    #   → base_key = "model.layers.0.self_attn.q_proj.weight"
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}

    for key, value in lora_state_dict.items():
        if key.endswith(".lora_A.weight"):
            base_key = key.replace(".lora_A.weight", ".weight")
            lora_a[base_key] = value
        elif key.endswith(".lora_B.weight"):
            base_key = key.replace(".lora_B.weight", ".weight")
            lora_b[base_key] = value
        elif key.endswith(".alpha"):
            # "...lora_A.alpha" or "...lora_B.alpha" — both map to same base
            base_key = key.rsplit(".lora_", 1)[0] + ".weight"
            alphas[base_key] = float(value)

    merged_count = 0
    for base_key in lora_a:
        if base_key not in lora_b:
            logger.warning(
                "LoRA lora_A found without matching lora_B for '%s'",
                base_key,
            )
            continue
        if base_key not in base_state_dict:
            logger.warning(
                "LoRA target '%s' not found in base model weights",
                base_key,
            )
            continue

        a_matrix = lora_a[base_key].float()  # [rank, in_features]
        b_matrix = lora_b[base_key].float()  # [out_features, rank]
        rank = a_matrix.shape[0]
        alpha = alphas.get(base_key, default_alpha if default_alpha is not None else rank)
        scale = alpha / rank

        # merged = base + scale * (B @ A)
        delta = (b_matrix @ a_matrix) * scale
        base_weight = base_state_dict[base_key]
        base_state_dict[base_key] = (base_weight.float() + delta).to(base_weight.dtype)
        merged_count += 1

    if merged_count > 0:
        logger.info("Merged %d LoRA adapter weights into base model", merged_count)
    elif lora_a:
        logger.warning(
            "Found %d LoRA A matrices but merged 0 — check weight name alignment",
            len(lora_a),
        )

    return base_state_dict


def split_fused_qkv(
    weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV weight tensor into separate Q, K, V tensors.

    Handles the common flat layout where Q, K, V are concatenated
    along dimension 0: ``[q_size + kv_size + kv_size, ...]``.

    Args:
        weight: Fused QKV weight tensor with shape
            ``[num_heads*head_dim + 2*num_kv_heads*head_dim, ...]``.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads.
        head_dim: Dimension per attention head.

    Returns:
        Tuple of (q_weight, k_weight, v_weight) split along dim 0.
    """
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    expected = q_size + 2 * kv_size
    if weight.shape[0] != expected:
        raise ValueError(
            f"QKV weight dim 0 is {weight.shape[0]}, expected "
            f"{expected} (num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim})"
        )
    q = weight[:q_size]
    k = weight[q_size : q_size + kv_size]
    v = weight[q_size + kv_size :]
    return q, k, v


def split_interleaved_qkv(
    weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV weight with per-head interleaved layout.

    Used by models like GPT-NeoX and Persimmon where the fused projection
    groups QKV **per head** rather than grouping all Q heads together:

        [h0_q, h0_k, h0_v,  h1_q, h1_k, h1_v, ...]

    This is the layout produced by ``nn.Linear(H, 3*H)`` when the output
    is then reshaped to ``[num_heads, 3, head_dim]`` and indexed.

    Args:
        weight: Fused QKV tensor of shape ``[num_heads * 3 * head_dim, ...]``
            or ``[num_heads * 3 * head_dim]`` for bias vectors.
        num_heads: Number of query attention heads (MHA only: equals num_kv_heads).
        num_kv_heads: Number of key/value heads (must equal num_heads for MHA).
        head_dim: Dimension per attention head.

    Returns:
        Tuple of (q, k, v) each of shape ``[num_heads * head_dim, ...]``.
    """
    expected = num_heads * 3 * head_dim
    if weight.shape[0] != expected:
        raise ValueError(
            f"Interleaved QKV dim 0 is {weight.shape[0]}, expected "
            f"{expected} (num_heads={num_heads}, head_dim={head_dim})"
        )
    if num_kv_heads != num_heads:
        raise ValueError(
            f"split_interleaved_qkv requires MHA (num_kv_heads == num_heads), "
            f"got GQA (num_kv_heads={num_kv_heads}, num_heads={num_heads})"
        )
    rest = weight.shape[1:]  # () for bias, (hidden_size,) for weight
    # Reshape to [num_heads, 3, head_dim, *rest] to un-interleave
    w = weight.reshape(num_heads, 3, head_dim, *rest)
    q = w[:, 0].reshape(num_heads * head_dim, *rest)
    k = w[:, 1].reshape(num_kv_heads * head_dim, *rest)
    v = w[:, 2].reshape(num_kv_heads * head_dim, *rest)
    return q, k, v


def split_interleaved_qkv_weights(
    state_dict: dict[str, torch.Tensor],
    fused_key: str,
    num_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Expand all fused interleaved QKV weights in a state dict.

    Scans *state_dict* for keys containing *fused_key* (e.g.
    ``"attention.query_key_value"``), splits each matched weight with
    :func:`split_interleaved_qkv`, and emits three new keys:
    ``{prefix}{attn_name}.q_proj{suffix}``,
    ``{prefix}{attn_name}.k_proj{suffix}``,
    ``{prefix}{attn_name}.v_proj{suffix}``.

    The ``attn_name`` is the segment of *fused_key* up to
    ``.query_key_value`` (e.g. ``"attention"`` or ``"self_attn"``).
    This consolidates the identical scaffolding code that appears in
    GPT-NeoX and Persimmon ``preprocess_weights`` implementations.

    Args:
        state_dict: Input weight dictionary.
        fused_key: Substring identifying fused QKV keys, e.g.
            ``"attention.query_key_value"`` or
            ``"self_attn.query_key_value"``.
        num_heads: Number of query attention heads.
        kv_heads: Number of key/value attention heads.
        head_dim: Dimension per attention head.

    Returns:
        New dictionary with fused QKV keys replaced by split q/k/v keys.
    """
    attn_name = fused_key.rsplit(".query_key_value", 1)[0]
    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if fused_key in key:
            q, k, v = split_interleaved_qkv(value, num_heads, kv_heads, head_dim)
            suffix = key.split(fused_key)[1]  # ".weight" or ".bias"
            prefix = key.split(fused_key)[0]  # e.g. "gpt_neox.layers.N."
            result[f"{prefix}{attn_name}.q_proj{suffix}"] = q
            result[f"{prefix}{attn_name}.k_proj{suffix}"] = k
            result[f"{prefix}{attn_name}.v_proj{suffix}"] = v
        else:
            result[key] = value
    return result


def rename_mlp_projections(
    name: str,
    old_up: str,
    old_down: str,
    new_up: str = "up_proj",
    new_down: str = "down_proj",
) -> str:
    """Rename MLP projection weight keys to the canonical ``up_proj``/``down_proj`` names.

    Many HuggingFace models use architecture-specific MLP projection names
    (``fc_in``/``fc_out``, ``c_fc``/``c_proj``, ``dense_h_to_4h``/
    ``dense_4h_to_h``, ``fc1``/``fc2``) while our ONNX ``FCMLP`` component
    always uses ``up_proj``/``down_proj``.  This helper centralises the
    two-replacement pattern that would otherwise be duplicated in every
    ``preprocess_weights`` implementation.

    Args:
        name: A single weight key string.
        old_up: HF name for the first (up) projection, e.g. ``"fc_in"``.
        old_down: HF name for the second (down) projection, e.g. ``"fc_out"``.
        new_up: Target name for the up projection (default ``"up_proj"``).
        new_down: Target name for the down projection (default ``"down_proj"``).

    Returns:
        The key with ``mlp.{old_up}`` → ``mlp.{new_up}`` and
        ``mlp.{old_down}`` → ``mlp.{new_down}`` applied.
    """
    return name.replace(f".mlp.{old_up}.", f".mlp.{new_up}.").replace(
        f".mlp.{old_down}.", f".mlp.{new_down}."
    )


def rename_weight_keys(
    state_dict: dict[str, torch.Tensor],
    replacements: Sequence[tuple[str, str]],
) -> dict[str, torch.Tensor]:
    """Apply ordered substring replacements to every key in a state dict.

    Centralises the ``for name, tensor in state_dict.items(): name =
    name.replace(...)`` rename loop that is duplicated across many model
    ``preprocess_weights`` implementations.  Each ``(old, new)`` pair is
    applied with :py:meth:`str.replace` to the *progressively updated* key,
    so the replacements are **ordered** and may cascade (the output of an
    earlier replacement can match the input of a later one) — this matches
    the semantics of the hand-written loops it replaces.  Use substrings
    specific enough (e.g. dot-delimited like ``".attention_layernorm."``)
    to avoid renaming unintended portions of a key.

    Tensor values are shared with *state_dict* (not cloned), matching the
    behaviour of the loops this replaces.

    Args:
        state_dict: Weight dictionary to transform (not modified).
        replacements: Ordered sequence of ``(old, new)`` substring pairs.

    Returns:
        A new dictionary with renamed keys.

    Raises:
        ValueError: If two distinct source keys map to the same renamed key
            (a collision that would otherwise silently drop a tensor).
    """
    result: dict[str, torch.Tensor] = {}
    producers: dict[str, str] = {}
    for name, tensor in state_dict.items():
        new_name = name
        for old, new in replacements:
            new_name = new_name.replace(old, new)
        if new_name in result:
            raise ValueError(
                f"Weight key collision after rename: {name!r} -> {new_name!r} "
                f"(already produced by {producers[new_name]!r})"
            )
        result[new_name] = tensor
        producers[new_name] = name
    return result


def split_codegen_qkv(
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    mp_num: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a CodeGen fused QKV weight with model-parallel interleaved layout.

    CodeGen uses a QVK (not QKV!) layout interleaved by model-parallel blocks:

        [q_mp0, v_mp0, k_mp0,  q_mp1, v_mp1, k_mp1, ...]

    where each block covers ``local_dim = num_heads * head_dim // mp_num``
    output neurons.  After splitting, the heads from each mp-block are
    concatenated to form the full Q, K, V projections.

    Args:
        weight: Fused QKV weight of shape ``[3 * num_heads * head_dim, hidden]``.
            CodeGen QKV has no bias, so this is always 2D.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        mp_num: Number of model-parallel blocks (default 4, matches CodeGen source).

    Returns:
        Tuple of (q, k, v) each of shape ``[num_heads * head_dim, hidden]``.
    """
    total = 3 * num_heads * head_dim
    if weight.shape[0] != total:
        raise ValueError(
            f"CodeGen QKV dim 0 is {weight.shape[0]}, expected {total} "
            f"(num_heads={num_heads}, head_dim={head_dim})"
        )
    if (num_heads * head_dim) % mp_num != 0:
        raise ValueError(
            f"num_heads * head_dim ({num_heads * head_dim}) must be divisible by "
            f"mp_num ({mp_num})"
        )
    local_dim = num_heads * head_dim // mp_num  # output neurons per mp-block per projection
    hidden = weight.shape[1]
    # [mp_num, 3 * local_dim, hidden] — each row is one mp-block
    w = weight.reshape(mp_num, 3 * local_dim, hidden)
    q = w[:, :local_dim, :].reshape(num_heads * head_dim, hidden)
    v = w[:, local_dim : 2 * local_dim, :].reshape(num_heads * head_dim, hidden)
    k = w[:, 2 * local_dim :, :].reshape(num_heads * head_dim, hidden)
    return q, k, v


def split_gate_up_proj(
    weight: torch.Tensor,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused gate_up_proj weight into gate_proj and up_proj.

    Handles the common layout where gate and up projections are
    concatenated along dimension 0: ``[2*intermediate_size, ...]``.

    Args:
        weight: Fused gate_up weight tensor with shape
            ``[2*intermediate_size, ...]``.
        intermediate_size: Size of the MLP intermediate layer.

    Returns:
        Tuple of (gate_weight, up_weight) split along dim 0.
    """
    expected = 2 * intermediate_size
    if weight.shape[0] != expected:
        raise ValueError(
            f"gate_up weight dim 0 is {weight.shape[0]}, expected "
            f"{expected} (intermediate_size={intermediate_size})"
        )
    return weight[:intermediate_size], weight[intermediate_size:]


def strip_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Remove a common prefix from all keys in a state dict.

    Keys that don't start with the prefix are dropped.

    Args:
        state_dict: Weight dictionary to transform.
        prefix: Prefix to strip. A trailing ``.`` is added if not present.

    Returns:
        New dictionary with stripped keys.
    """
    stripped = prefix if prefix.endswith(".") else prefix + "."
    return {k[len(stripped) :]: v for k, v in state_dict.items() if k.startswith(stripped)}


def tie_word_embeddings(
    state_dict: dict[str, torch.Tensor],
    embed_key: str = "model.embed_tokens.weight",
    head_key: str = "lm_head.weight",
) -> None:
    """Ensure both embedding and LM head weights are present and tied.

    When HuggingFace saves a model with ``tie_word_embeddings=True``, only
    one of the two weights (embedding or LM head) is stored in the
    checkpoint.  This function fills in the missing key by assigning it
    to **the same Python tensor object** as the existing key.

    This identity relationship is what enables true ONNX weight sharing
    downstream: :func:`~mobius._weight_loading.apply_weights` detects
    that two state-dict entries share the same underlying storage (via
    ``data_ptr()``), creates a **single** ONNX initializer for the first
    occurrence, and redirects all graph uses of the second initializer to
    point at the first one.  The result is one copy of the embedding
    table in the ONNX file, used by both the Gather (embedding lookup)
    and MatMul (LM head projection) nodes.

    Mutates *state_dict* in place.

    Args:
        state_dict: Weight dictionary (modified in place).
        embed_key: Key for the embedding weight.
        head_key: Key for the LM head weight.
    """
    if head_key not in state_dict and embed_key in state_dict:
        state_dict[head_key] = state_dict[embed_key]
    elif embed_key not in state_dict and head_key in state_dict:
        state_dict[embed_key] = state_dict[head_key]
    elif embed_key not in state_dict and head_key not in state_dict:
        raise ValueError(
            f"tie_word_embeddings: neither '{embed_key}' nor '{head_key}' "
            f"found in state_dict. Check that the key names match the "
            f"weight dictionary after any prefix stripping."
        )


def vlm_decoder_weights(
    state_dict: dict[str, torch.Tensor],
    prefix: str = "language_model.",
    tie: bool = False,
    embed_key: str = "model.embed_tokens.weight",
    head_key: str = "lm_head.weight",
) -> dict[str, torch.Tensor]:
    """Extract and rename decoder weights for a VLM model.

    Filters keys starting with *prefix*, strips the prefix, and
    optionally applies embedding/LM-head weight tying.

    This is the standard pattern for VLM decoder sub-models (LLaVA,
    Gemma3, BLIP-2, Mllama) where decoder weights are nested under
    ``language_model.`` in the HuggingFace checkpoint.

    Args:
        state_dict: Full model state dict.
        prefix: Prefix identifying decoder weights.
        tie: Whether to apply weight tying.
        embed_key: Embedding key (after prefix strip).
        head_key: LM head key (after prefix strip).

    Returns:
        New dictionary with decoder weights (prefix stripped).
    """
    stripped = prefix if prefix.endswith(".") else prefix + "."
    renamed = {k[len(stripped) :]: v for k, v in state_dict.items() if k.startswith(stripped)}
    if tie:
        tie_word_embeddings(renamed, embed_key, head_key)
    return renamed


def vlm_embedding_weights(
    state_dict: dict[str, torch.Tensor],
    keyword: str = "embed_tokens",
    prefixes: tuple[str, ...] = (
        "language_model.model.",
        "language_model.",
    ),
) -> dict[str, torch.Tensor]:
    """Extract embedding weights for a VLM embedding sub-model.

    Filters keys containing *keyword*, then strips the first matching
    prefix from each key.

    This is the standard pattern for VLM embedding sub-models (LLaVA,
    Gemma3, BLIP-2, Mllama) that need ``embed_tokens`` weights with
    ``language_model.model.`` or ``language_model.`` prefixes removed.

    Args:
        state_dict: Full model state dict.
        keyword: Substring that must appear in the key.
        prefixes: Prefixes to strip, tried in order (first match wins).

    Returns:
        New dictionary with embedding weights.
    """
    renamed: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if keyword not in key:
            continue
        new_key = key
        for pfx in prefixes:
            if new_key.startswith(pfx):
                new_key = new_key[len(pfx) :]
                break
        renamed[new_key] = value
    return renamed


def vlm_vision_weights(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Extract vision-tower weights for a VLM vision sub-model.

    Keeps only keys starting with one of *prefixes* and renames the vision
    MLP projections ``mlp.fc1`` → ``mlp.up_proj`` and ``mlp.fc2`` →
    ``mlp.down_proj`` to match our ``FCMLP`` component naming.

    This is the standard pattern for VLM vision sub-models (LLaVA, Gemma3,
    Mllama) whose HuggingFace vision encoders use ``fc1``/``fc2`` MLP names.

    Args:
        state_dict: Full model state dict.
        prefixes: Prefixes identifying vision-tower weights to keep, e.g.
            ``("vision_tower.", "multi_modal_projector.")`` or
            ``("vision_model.",)``.

    Returns:
        New dictionary with the kept vision weights and renamed MLP keys.
    """
    renamed: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith(prefixes):
            continue
        new_key = key.replace(".mlp.fc1.", ".mlp.up_proj.").replace(
            ".mlp.fc2.", ".mlp.down_proj."
        )
        renamed[new_key] = value
    return renamed


def _reshape_packed_qweight(value: torch.Tensor, blob_size: int) -> torch.Tensor:
    """Transpose and reshape a packed qweight tensor for MatMulNBits.

    Converts ``[..., K_packed, N]`` int32 to
    ``[..., N, n_blocks, blob_size]`` uint8.
    """
    transposed = value.transpose(-1, -2).contiguous()
    prefix = transposed.shape[:-2]
    n = transposed.shape[-2]
    packed = transposed.view(torch.uint8)
    n_blocks = packed.shape[-1] // blob_size
    return packed.reshape(*prefix, n, n_blocks, blob_size)


def _reshape_packed_qzeros(
    value: torch.Tensor, bits: int, n_blocks: int, offset: int = 0
) -> torch.Tensor:
    """Unpack GPTQ qzeros along N and repack along blocks for MatMulNBits.

    GPTQ stores zero points as ``[..., n_groups, N / pack_factor]`` int32,
    packing ``pack_factor = 32 // bits`` *output channels* into each int32.
    GPTQModel's packer writes
    ``qzeros[:, col] |= zeros[:, col * pack_factor + j] << (bits * j)``,
    so the packed axis is N while the group axis stays unpacked.

    MatMulNBits expects ``[..., N, ceil(n_blocks * bits / 8)]`` uint8 packed
    along the *block* axis instead. Because the two formats pack different
    axes, reinterpreting the buffer as bytes is not sufficient: the values
    must be unpacked, transposed, and repacked.

    Args:
        value: Packed qzeros tensor ``[..., n_groups, N / pack_factor]`` int32.
        bits: Quantization bit-width (4 or 8).
        n_blocks: Actual number of quantization blocks (``ceil(K / block_size)``).
        offset: Added to every zero point after unpacking, for checkpoint
            formats that store a biased value. Results are clamped to the
            representable range.
    """
    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    prefix = value.shape[:-2]
    n_groups = value.shape[-2]

    # Unpack along N: [..., n_groups, N/pack_factor, pack_factor] -> [..., n_groups, N]
    shifts = torch.arange(pack_factor, device=value.device, dtype=torch.int32) * bits
    unpacked = (value.unsqueeze(-1) >> shifts) & mask
    unpacked = unpacked.reshape(*prefix, n_groups, -1)

    if offset:
        unpacked = (unpacked + offset).clamp(0, mask)

    # [..., N, n_groups], trimmed to the real block count
    transposed = unpacked.transpose(-1, -2).contiguous()[..., :n_blocks]

    # Repack along blocks: 8 // bits zero points per byte
    per_byte = 8 // bits
    zp_cols = math.ceil(n_blocks * bits / 8)
    padded = zp_cols * per_byte
    if transposed.shape[-1] < padded:
        transposed = torch.nn.functional.pad(transposed, (0, padded - transposed.shape[-1]))
    grouped = transposed.reshape(*transposed.shape[:-1], zp_cols, per_byte)
    byte_shifts = torch.arange(per_byte, device=value.device, dtype=torch.int32) * bits
    packed = (grouped << byte_shifts).sum(dim=-1)
    return packed.to(torch.uint8)


def pack_qmoe_expert_weights(
    state_dict: dict[str, torch.Tensor],
    *,
    target_moe_path: str = ".mlp.moe",
) -> dict[str, torch.Tensor]:
    """Map fused expert-major quantized tensors to native QMoE parameters."""
    packed: dict[str, torch.Tensor] = {}
    projections = {
        ".mlp.experts.gate_up_proj.weight": (
            f"{target_moe_path}.fc1_experts_weights",
            True,
        ),
        ".mlp.experts.gate_up_proj.scales": (
            f"{target_moe_path}.fc1_scales",
            False,
        ),
        ".mlp.experts.gate_up_proj.zero_points": (
            f"{target_moe_path}.fc1_experts_zero_points",
            False,
        ),
        ".mlp.experts.down_proj.weight": (
            f"{target_moe_path}.fc2_experts_weights",
            True,
        ),
        ".mlp.experts.down_proj.scales": (
            f"{target_moe_path}.fc2_scales",
            False,
        ),
        ".mlp.experts.down_proj.zero_points": (
            f"{target_moe_path}.fc2_experts_zero_points",
            False,
        ),
    }
    for key, value in state_dict.items():
        for source, (target, flatten_blocks) in projections.items():
            if source in key:
                key = key.replace(source, target)
                if flatten_blocks:
                    value = value.flatten(-2)
                break
        packed[key] = value
    return packed


def infer_compressed_tensors_group_size(
    state_dict: dict[str, torch.Tensor],
    *,
    bits: int,
    group_size: int | None = None,
) -> int:
    """Infer and validate a uniform group size from packed checkpoint tensors.

    ``pack-quantized`` checkpoints do not always serialize ``group_size`` in
    ``config.json``. Each packed matrix does serialize its logical shape and
    per-group scales, which determine the group size without reading or
    dequantizing the weights.
    """
    group_size_candidates: set[int] | None = None
    packed_keys = [key for key in state_dict if key.endswith(".weight_packed")]
    if not packed_keys:
        raise ValueError(
            "Compressed-tensors checkpoint contains no '*.weight_packed' tensors."
        )

    for packed_key in packed_keys:
        stem = packed_key[: -len(".weight_packed")]
        shape_key = f"{stem}.weight_shape"
        scale_key = f"{stem}.weight_scale"
        if shape_key not in state_dict or scale_key not in state_dict:
            raise ValueError(
                f"Compressed tensor {packed_key!r} requires both {shape_key!r} "
                f"and {scale_key!r}."
            )

        logical_shape = tuple(int(dim) for dim in state_dict[shape_key].tolist())
        if len(logical_shape) != 2:
            raise ValueError(
                f"Compressed tensor {packed_key!r} must describe a 2D weight, "
                f"got shape {logical_shape}."
            )
        n, k = logical_shape
        scales = state_dict[scale_key]
        if scales.ndim != 2 or scales.shape[0] != n:
            raise ValueError(
                f"Scale tensor {scale_key!r} must have shape [N, n_blocks] "
                f"for N={n}, got {list(scales.shape)}."
            )
        n_blocks = scales.shape[1]
        if n_blocks <= 0:
            raise ValueError(f"Scale tensor {scale_key!r} must contain at least one block.")
        if group_size is not None:
            expected_blocks = math.ceil(k / group_size)
            if n_blocks != expected_blocks:
                raise ValueError(
                    f"Scale tensor {scale_key!r} must contain {expected_blocks} "
                    f"blocks for K={k} and group_size={group_size}, got {n_blocks}."
                )
        else:
            max_candidate = max(16, 1 << (k - 1).bit_length())
            tensor_candidates = {
                candidate
                for exponent in range(4, max_candidate.bit_length())
                if math.ceil(k / (candidate := 1 << exponent)) == n_blocks
            }
            if not tensor_candidates:
                raise ValueError(
                    f"Cannot infer a MatMulNBits group size for {packed_key!r}: "
                    f"K={k}, n_blocks={n_blocks}."
                )
            if group_size_candidates is None:
                group_size_candidates = tensor_candidates
            else:
                group_size_candidates &= tensor_candidates

        packed = state_dict[packed_key]
        expected_words = math.ceil(k * bits / 32)
        if packed.dtype != torch.int32 or list(packed.shape) != [n, expected_words]:
            raise ValueError(
                f"Packed tensor {packed_key!r} must be int32 with shape "
                f"[{n}, {expected_words}], got {packed.dtype} {list(packed.shape)}."
            )

    if group_size is None:
        if not group_size_candidates:
            raise ValueError(
                "No uniform MatMulNBits group size matches all compressed tensors."
            )
        # Multiple candidates are only possible when every affected matrix has
        # one scale block. The smallest candidate avoids unnecessary padding;
        # all candidates have identical dequantization in that case.
        group_size = min(group_size_candidates)
    if group_size < 16 or group_size & (group_size - 1):
        raise ValueError(
            f"MatMulNBits group size must be a power of two >= 16, got {group_size}."
        )
    return group_size


def preprocess_compressed_tensors_weights(
    state_dict: dict[str, torch.Tensor],
    *,
    bits: int,
    group_size: int,
) -> dict[str, torch.Tensor]:
    """Repack compressed-tensors INT weights for ORT ``MatMulNBits``.

    Compressed-tensors and ORT both store signed quantized values as unsigned
    bit patterns offset by ``2**(bits - 1)``, with the earliest value in the
    least-significant bits. The conversion therefore only reinterprets each
    contiguous INT32 row as bytes and reshapes those bytes into ORT blocks.
    Scales are preserved bit-for-bit; no dequantization or rounding occurs.
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"MatMulNBits supports 2, 4, or 8 bits, got {bits}.")
    if group_size < 16 or group_size & (group_size - 1):
        raise ValueError(f"group_size must be a power of two >= 16, got {group_size}.")

    result: dict[str, torch.Tensor] = {}
    packed_stems = {
        key[: -len(".weight_packed")] for key in state_dict if key.endswith(".weight_packed")
    }
    for key, value in state_dict.items():
        if key.endswith(".weight_g_idx"):
            raise ValueError(
                "Activation-ordered compressed-tensors weights are not supported: "
                f"found {key!r}."
            )
        if key.endswith(".weight_zero_point"):
            raise ValueError(
                "Asymmetric compressed-tensors checkpoints are not yet supported: "
                f"found {key!r}."
            )
        if key.endswith(".weight_shape"):
            continue
        if key.endswith(".weight_scale"):
            stem = key[: -len(".weight_scale")]
            if stem in packed_stems:
                result[f"{stem}.scales"] = value.contiguous()
            else:
                result[key] = value
            continue
        if not key.endswith(".weight_packed"):
            result[key] = value
            continue

        stem = key[: -len(".weight_packed")]
        shape_key = f"{stem}.weight_shape"
        scale_key = f"{stem}.weight_scale"
        if shape_key not in state_dict or scale_key not in state_dict:
            raise ValueError(
                f"Compressed tensor {key!r} requires both {shape_key!r} and {scale_key!r}."
            )
        logical_shape = tuple(int(dim) for dim in state_dict[shape_key].tolist())
        if len(logical_shape) != 2:
            raise ValueError(
                f"Compressed tensor {key!r} must describe a 2D weight, "
                f"got shape {logical_shape}."
            )
        n, k = logical_shape
        n_blocks = math.ceil(k / group_size)
        blob_size = group_size * bits // 8
        expected_words = math.ceil(k * bits / 32)
        if value.dtype != torch.int32 or list(value.shape) != [n, expected_words]:
            raise ValueError(
                f"Packed tensor {key!r} must be int32 with shape "
                f"[{n}, {expected_words}], got {value.dtype} {list(value.shape)}."
            )
        scales = state_dict[scale_key]
        if list(scales.shape) != [n, n_blocks]:
            raise ValueError(
                f"Scale tensor {scale_key!r} must have shape [{n}, {n_blocks}], "
                f"got {list(scales.shape)}."
            )

        packed_bytes = value.contiguous().view(torch.uint8)
        required_bytes = math.ceil(k * bits / 8)
        packed_bytes = packed_bytes[:, :required_bytes]
        padded_bytes = n_blocks * blob_size
        if required_bytes < padded_bytes:
            zero_code = 1 << (bits - 1)
            zero_byte = sum(zero_code << shift for shift in range(0, 8, bits))
            packed_bytes = torch.nn.functional.pad(
                packed_bytes, (0, padded_bytes - required_bytes), value=zero_byte
            )
        result[f"{stem}.weight"] = packed_bytes.reshape(n, n_blocks, blob_size).contiguous()

    return result


def unwrap_gptq_observer_modules(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip the observer-wrapper infix GPTQModel adds to unquantized layers.

    GPTQModel wraps every targeted ``nn.Linear`` in an observer module. Layers
    it chose to quantize are replaced outright, so their packed tensors keep
    the original path, but layers left in floating point end up one level
    deeper as ``<path>.linear.weight`` alongside scalar activation observers.

    The wrapped name does not exist in the built graph, so unless the infix is
    removed those weights never bind and the affected component silently runs
    on uninitialized values. This must run *before* a model's
    ``preprocess_weights`` so that its renames and transposes see canonical
    names. The activation observers are left untouched: the built graph
    declares matching initializers for them.
    """
    return {key.replace(".linear.", "."): value for key, value in state_dict.items()}


def preprocess_gptq_weights(
    state_dict: dict[str, torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Rename, transpose and reshape GPTQ weights for MatMulNBits.

    GPTQ stores quantized weights with these key suffixes:
      - ``*.qweight`` (int32): packed quantized values, shape [K_packed, N]
      - ``*.scales`` (float16): per-group scales, shape [n_groups, N]
      - ``*.qzeros`` (int32): packed zero points, shape [n_groups_packed, N]
      - ``*.g_idx`` (int32): group index (dropped with warning)

    MatMulNBits expects:
      - ``weight``:  [N, n_blocks, blob_size]  uint8
      - ``scales``:  [N, n_blocks]             float
      - ``zero_points``: [N, ceil(n_blocks * bits / 8)]  uint8 (packed)

    where ``n_blocks = K / group_size`` and
    ``blob_size = group_size * bits / 8``.

    Args:
        state_dict: Model state dict with GPTQ keys.
        bits: Quantization bit-width (typically 4).
        group_size: Number of elements per quantization group.

    Returns:
        State dict with renamed, transposed and reshaped weights.
    """
    import logging

    logger = logging.getLogger(__name__)
    blob_size = group_size * bits // 8
    result: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        # GPTQModel's observer wrapper is unwrapped earlier, before the
        # model's own preprocess_weights runs, so keys arriving here are
        # already canonical.
        if key.endswith(".g_idx"):
            # g_idx maps element i to its quantization group.  For
            # non-desc_act models this is simply i // group_size.
            trivial = torch.arange(value.numel(), dtype=value.dtype) // group_size
            if not torch.equal(value, trivial):
                logger.warning(
                    "Dropping %s — desc_act models with non-trivial "
                    "g_idx may produce incorrect results",
                    key,
                )
            continue

        if key.endswith(".qweight"):
            new_key = key.replace(".qweight", ".weight")
            result[new_key] = _reshape_packed_qweight(value, blob_size)

        elif key.endswith(".qzeros"):
            new_key = key.replace(".qzeros", ".zero_points")
            # Derive n_blocks from the corresponding qweight shape
            qw_key = key.replace(".qzeros", ".qweight")
            if qw_key not in state_dict:
                raise ValueError(
                    f"Missing {qw_key} — qweight must be present alongside qzeros for {key}"
                )
            k = state_dict[qw_key].shape[-2] * 32 // bits
            n_blocks = math.ceil(k / group_size)
            # GPTQ stores ``zero - 1``, so dequantization is
            # ``scale * (q - (z + 1))``. MatMulNBits applies
            # ``scale * (q - zero_point)`` with no bias of its own, so the
            # +1 has to be folded in here. Omitting it leaves the weights
            # highly correlated with the original but with roughly 3x the
            # reconstruction error, which is enough to reduce generation to
            # noise while still looking plausible in a shape check.
            result[new_key] = _reshape_packed_qzeros(value, bits, n_blocks, offset=1)

        elif key.endswith(".scales"):
            result[key] = value.transpose(-1, -2).contiguous()

        else:
            result[key] = value

    return result


def preprocess_olive_weights(
    state_dict: dict[str, torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
    quantize_embeddings: bool = False,
    quantize_lm_head: bool = False,
    tie_word_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """Rename and reshape Olive-packed quantized weights.

    Olive stores quantized weights with uint8 packing:
      - ``*.qweight``: [N, packed_K] uint8
      - ``*.scales``: [N, n_blocks]
      - ``*.qzeros``: [N, ceil(n_blocks * bits / 8)] uint8, asymmetric only

    Linear projections target ``MatMulNBits``, which expects ``weight`` as
    [N, n_blocks, blob_size]; scales and zero-points already match the
    expected orientation, so they are renamed but not transposed.

    The input embedding table (``*.embed_tokens.qweight``) instead targets
    ``GatherBlockQuantized``, which consumes the **2-D** uint8 ``qweight``
    directly — so it is kept as-is (only ``qzeros`` is renamed to
    ``zero_points``).

    Tied LM head: Olive RTN drops ``lm_head.*`` when the head is tied. When the
    head is **quantized**, no ``lm_head`` weights are produced here — the model
    shares the embedding's packed table via :class:`TiedQuantizedLMHead` (one
    initializer). When the head stays **float** (unquantized embedding), the
    standard float tie is applied so ``lm_head.weight`` aliases the embedding.

    Args:
        state_dict: Model state dict with Olive quantization keys.
        bits: Quantization bit-width (typically 4).
        group_size: Number of elements per quantization group.
        quantize_embeddings: Whether the embedding table is quantized.
        quantize_lm_head: Whether the LM head is quantized.
        tie_word_embeddings: Whether embedding and LM head share weights.

    Returns:
        State dict with renamed and reshaped weights (and, for a float tied
        head, the aliased ``lm_head.weight``).
    """
    blob_size = group_size * bits // 8
    result: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.endswith("embed_tokens.qweight"):
            # GatherBlockQuantized consumes the 2-D uint8 table directly.
            if value.dtype != torch.uint8:
                raise ValueError(
                    f"Olive embedding qweight must be uint8 for {key}, got {value.dtype}"
                )
            if value.shape[-1] % blob_size != 0:
                raise ValueError(
                    f"Olive embedding qweight packed dimension for {key} "
                    f"({value.shape[-1]}) must be divisible by blob_size ({blob_size})"
                )
            result[key] = value.contiguous()
        elif key.endswith("embed_tokens.qzeros"):
            if value.dtype != torch.uint8:
                raise ValueError(
                    f"Olive embedding qzeros must be uint8 for {key}, got {value.dtype}"
                )
            result[key.replace(".qzeros", ".zero_points")] = value.contiguous()
        elif key.endswith("embed_tokens.scales"):
            result[key] = value
        elif key.endswith(".qweight"):
            if value.dtype != torch.uint8:
                raise ValueError(f"Olive qweight must be uint8 for {key}, got {value.dtype}")
            if value.shape[-1] % blob_size != 0:
                raise ValueError(
                    f"Olive qweight packed dimension for {key} ({value.shape[-1]}) "
                    f"must be divisible by blob_size ({blob_size})"
                )
            new_key = key.replace(".qweight", ".weight")
            result[new_key] = value.reshape(value.shape[0], -1, blob_size).contiguous()
        elif key.endswith(".qzeros"):
            if value.dtype != torch.uint8:
                raise ValueError(f"Olive qzeros must be uint8 for {key}, got {value.dtype}")
            new_key = key.replace(".qzeros", ".zero_points")
            result[new_key] = value.contiguous()
        else:
            result[key] = value

    # A tied quantized head shares the embedding's Parameters in the module
    # (TiedQuantizedLMHead), so no lm_head initializers exist to fill here.
    # Only a tied *float* head needs an explicit weight alias.
    if (
        tie_word_embeddings
        and not quantize_lm_head
        and "lm_head.weight" not in result
        and "model.embed_tokens.weight" in result
    ):
        result["lm_head.weight"] = result["model.embed_tokens.weight"]

    return result


def preprocess_awq_weights(
    state_dict: dict[str, torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Rename, transpose and reshape AWQ weights for MatMulNBits.

    AWQ uses the same int32 packing as GPTQ for qweight/qzeros/scales
    but does **not** include ``g_idx``.  The key difference is that AWQ
    zero points are stored with an implicit ``+1`` offset that must be
    subtracted so MatMulNBits receives the correct raw values.

    Args:
        state_dict: Model state dict with AWQ keys.
        bits: Quantization bit-width (typically 4).
        group_size: Number of elements per quantization group.

    Returns:
        State dict with renamed, transposed and reshaped weights.
    """
    blob_size = group_size * bits // 8
    result: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.endswith(".qweight"):
            new_key = key.replace(".qweight", ".weight")
            result[new_key] = _reshape_packed_qweight(value, blob_size)

        elif key.endswith(".qzeros"):
            new_key = key.replace(".qzeros", ".zero_points")
            # Derive n_blocks from the corresponding qweight shape
            qw_key = key.replace(".qzeros", ".qweight")
            if qw_key not in state_dict:
                raise ValueError(
                    f"Missing {qw_key} — qweight must be present alongside qzeros for {key}"
                )
            k = state_dict[qw_key].shape[-2] * 32 // bits
            n_blocks = math.ceil(k / group_size)
            # AWQ zero points have an implicit +1 offset; subtract it
            # before unpacking so MatMulNBits sees the raw value.
            # For 4-bit, each byte packs TWO nibbles — subtract per-nibble
            # to avoid cross-nibble borrow (e.g. 0x80 - 1 = 0x7F is wrong).
            zp = _reshape_packed_qzeros(value, bits, n_blocks)
            if bits == 4:
                low = (zp & 0x0F).to(torch.int16) - 1
                high = ((zp >> 4) & 0x0F).to(torch.int16) - 1
                low = low.clamp(min=0).to(torch.uint8)
                high = high.clamp(min=0).to(torch.uint8)
                result[new_key] = (high << 4) | low
            else:
                zp_int16 = zp.to(torch.int16) - 1
                result[new_key] = zp_int16.clamp(min=0).to(torch.uint8)

        elif key.endswith(".scales"):
            result[key] = value.transpose(-1, -2).contiguous()

        else:
            result[key] = value

    return result


def _unpack_quark_int32(value: torch.Tensor, bits: int, value_count: int) -> torch.Tensor:
    """Unpack Quark values packed along the final dimension of an int32 tensor."""
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=value.device)
    unpacked = torch.bitwise_right_shift(value.unsqueeze(-1), shifts)
    unpacked = torch.bitwise_and(unpacked, (1 << bits) - 1)
    return unpacked.reshape(*value.shape[:-1], -1)[..., :value_count].to(torch.uint8)


def _pack_ort_uint8(value: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack values along the final dimension into ORT's uint8 representation."""
    values_per_byte = 8 // bits
    padding = (-value.shape[-1]) % values_per_byte
    if padding:
        value = torch.nn.functional.pad(value, (0, padding))
    packed = torch.zeros(
        *value.shape[:-1],
        value.shape[-1] // values_per_byte,
        dtype=torch.uint8,
        device=value.device,
    )
    for index in range(values_per_byte):
        packed |= value[..., index::values_per_byte] << (index * bits)
    return packed


def preprocess_quark_weights(
    state_dict: dict[str, torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Convert Quark-native packed tensors to MatMulNBits initializer layouts.

    Quark packs INT weights and zero points along the output-channel dimension:
    ``weight`` is ``[K, ceil(N * bits / 32)]`` and ``weight_zero_point`` is
    ``[groups, ceil(N * bits / 32)]``. MatMulNBits instead packs weights along
    K into ``[N, groups, group_size * bits / 8]`` and zero points along groups.
    Floating-point weights excluded from quantization retain the ordinary
    ``weight`` key and pass through unchanged.
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"Quark MatMulNBits import requires 2, 4, or 8 bits, got {bits}.")

    blob_size = group_size * bits // 8
    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".weight") and value.dtype == torch.int32:
            scale_key = key.replace(".weight", ".weight_scale")
            if scale_key not in state_dict:
                raise ValueError(f"Missing {scale_key} for quantized Quark weight {key}.")
            output_channels = state_dict[scale_key].shape[-1]
            codes = _unpack_quark_int32(value, bits, output_channels)
            packed = _pack_ort_uint8(codes.transpose(-1, -2).contiguous(), bits)
            if packed.shape[-1] % blob_size:
                raise ValueError(
                    f"Packed Quark weight {key} has {packed.shape[-1]} bytes per output channel; "
                    f"expected a multiple of blob_size {blob_size}."
                )
            result[key] = packed.reshape(*packed.shape[:-1], -1, blob_size).contiguous()
        elif key.endswith(".weight_scale"):
            result[key.replace(".weight_scale", ".scales")] = value.transpose(
                -1, -2
            ).contiguous()
        elif key.endswith(".weight_zero_point"):
            scale_key = key.replace(".weight_zero_point", ".weight_scale")
            if scale_key not in state_dict:
                raise ValueError(f"Missing {scale_key} for Quark zero points {key}.")
            output_channels = state_dict[scale_key].shape[-1]
            zeros = _unpack_quark_int32(value, bits, output_channels)
            zeros = zeros.transpose(-1, -2).contiguous()
            result[key.replace(".weight_zero_point", ".zero_points")] = _pack_ort_uint8(
                zeros, bits
            )
        else:
            result[key] = value
    return result
