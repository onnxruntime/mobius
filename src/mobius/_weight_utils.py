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
from collections.abc import Collection, Sequence

import torch

from mobius._configs import QuantizationConfig, QuantizedWeightFormat

logger = logging.getLogger(__name__)

# Key suffixes marking a *packed quantized* sidecar tensor rather than a float
# weight. A "sidecar" here is an auxiliary tensor that rides alongside a
# quantized parameter's canonical ``.weight`` key instead of replacing it —
# the packed integer payload (``qweight``) plus its per-group ``scales`` and
# ``qzeros`` are each stored under their own key, so one logical weight is
# split across several sidecar keys until they are unpacked back into a
# single dequantized tensor. Olive appends them with an underscore directly
# to the parameter name (``<pname>_qweight``, see Olive's
# ``olive/common/quant/state_dict.py``), while GPTQ/AWQ store dotted sibling
# buffers on the owning module (``<module>.qweight``). Both conventions occur
# in raw HF checkpoints.
OLIVE_PACKED_QUANT_SUFFIXES: frozenset[str] = frozenset({"_qweight", "_scales", "_qzeros"})
DOTTED_PACKED_QUANT_SUFFIXES: frozenset[str] = frozenset({".qweight", ".scales", ".qzeros"})
# str.endswith() requires a tuple (not a set), so the combined predicate below
# needs a tuple even though the two suffix sets above are otherwise unordered.
PACKED_QUANT_SUFFIXES: tuple[str, ...] = tuple(
    OLIVE_PACKED_QUANT_SUFFIXES | DOTTED_PACKED_QUANT_SUFFIXES
)


def is_packed_quant_key(name: str) -> bool:
    """Whether ``name`` is a packed-quantization sidecar key.

    Packed keys carry the quantized payload (``qweight``), the per-group
    ``scales`` or the ``qzeros`` of a quantized parameter, in either the Olive
    underscore convention (``…experts.gate_up_proj_qweight``) or the GPTQ/AWQ
    dotted convention (``…gate_proj.qweight``).

    HF→ONNX key rewriting must leave these tensors intact until
    :func:`preprocess_quantized_weights` unpacks them: they are packed bytes,
    not float weights, so reshaping or splitting them corrupts the payload —
    and several sidecars of one projection would collapse onto a single
    renamed ``.weight`` key.
    """
    return name.endswith(PACKED_QUANT_SUFFIXES)


def materialize_split_tied_olive_lm_head(
    state_dict: dict[str, torch.Tensor],
    *,
    embed_key: str,
    head_key: str,
    embedding_quantization: QuantizationConfig | None,
    head_quantization: QuantizationConfig | None,
) -> None:
    """Copy a tied Olive token table into a separately exported LM head.

    A single-model graph can share packed embedding parameters with
    ``TiedQuantizedLMHead``. Split decoder/embedding graphs cannot share ONNX
    initializers, so the raw Olive sidecars must be duplicated before each
    component applies its own shape normalization.
    """
    source_qweight = f"{embed_key}_qweight"
    target_qweight = f"{head_key}_qweight"
    if source_qweight not in state_dict or target_qweight in state_dict:
        return

    if (
        embedding_quantization is None
        or embedding_quantization.quant_method != "olive"
        or not embedding_quantization.quantize_embeddings
    ):
        raise ValueError(
            f"Packed tied embedding {source_qweight!r} does not match the "
            "embedding component quantization configuration."
        )
    if (
        head_quantization is None
        or head_quantization.quant_method != "olive"
        or not head_quantization.quantize_lm_head
    ):
        raise NotImplementedError(
            "A packed tied embedding in split component graphs requires "
            "quantize_lm_head=True for the decoder component."
        )

    embedding_layout = (
        embedding_quantization.bits,
        embedding_quantization.group_size,
        embedding_quantization.sym,
    )
    head_layout = (
        head_quantization.bits,
        head_quantization.group_size,
        head_quantization.sym,
    )
    if embedding_layout != head_layout:
        raise ValueError(
            "A packed tied embedding and its split LM head must use the same "
            f"quantization layout, got {embedding_layout!r} and {head_layout!r}."
        )

    required_suffixes = ["_qweight", "_scales"]
    if not embedding_quantization.sym:
        required_suffixes.append("_qzeros")
    missing = [
        suffix for suffix in required_suffixes if f"{embed_key}{suffix}" not in state_dict
    ]
    if missing:
        raise ValueError(
            f"Packed tied embedding {embed_key!r} is missing Olive sidecars {missing!r}."
        )

    for suffix in ("_qweight", "_scales", "_qzeros"):
        source = f"{embed_key}{suffix}"
        if source in state_dict:
            state_dict[f"{head_key}{suffix}"] = state_dict[source]


def supported_qmoe_quantization(
    quantization: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Return quantization settings when they match the native QMoE ABI.

    Accepts the integer-affine int4 block schemes whose ``(q - zero_point) *
    scale`` dequantization is byte-identical to ``MatMulNBits`` and to the
    ``com.microsoft::QMoE`` kernel: GPTQ, AWQ, and Olive RTN. The CUDA QMoE
    kernel requires a power-of-two ``block_size >= 16``; unsupported configs
    fall back to the portable dense representation instead of emitting an
    unrunnable node.
    """
    if (
        quantization is None
        or quantization.bits != 4
        or quantization.weight_format is not QuantizedWeightFormat.INTEGER_AFFINE
        or quantization.float_zero_point
        or quantization.quant_method not in {"gptq", "awq", "olive"}
    ):
        return None
    block_size = quantization.group_size
    if block_size < 16 or (block_size & (block_size - 1)) != 0:
        return None
    return quantization


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
    downstream: :func:`~mobius.integrations._weight_loading.apply_weights` detects
    that two state-dict entries share the same underlying storage (via
    ``data_ptr()``), creates a **single** ONNX initializer for the first
    occurrence, and redirects all graph uses of the second initializer to
    point at the first one.  The result is one copy of the embedding
    table in the ONNX file, used by both the Gather (embedding lookup)
    and MatMul (LM head projection) nodes.

    This also supports subclasses that replace ``self.model`` after
    :class:`CausalLMModel` construction (for example, Cohere and GPT-2-family
    models). Their embedding and head Parameters may initially be distinct,
    but the shared tensor storage lets weight loading unify them.

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


def _reshape_packed_qzeros(value: torch.Tensor, bits: int, n_blocks: int) -> torch.Tensor:
    """Transpose and unpack packed qzeros for MatMulNBits.

    Converts ``[..., n_groups_packed, N]`` int32 to
    ``[..., N, zp_cols]`` uint8
    where ``zp_cols = ceil(n_blocks * bits / 8)``.  For 4-bit this
    packs two zero-point values per byte, matching ORT's expectation.

    Args:
        value: Packed qzeros tensor ``[n_groups_packed, N]`` int32.
        bits: Quantization bit-width (4 or 8).
        n_blocks: Actual number of quantization blocks (``ceil(K / block_size)``).
    """
    transposed = value.transpose(-1, -2).contiguous()
    prefix = transposed.shape[:-2]
    n = transposed.shape[-2]
    flat_uint8 = transposed.reshape(-1).view(torch.uint8).reshape(*prefix, n, -1)
    zp_cols = math.ceil(n_blocks * bits / 8)
    return flat_uint8[..., :zp_cols]


def stack_per_expert_moe_weights(
    state_dict: dict[str, torch.Tensor],
    *,
    qmoe_target_path: str,
) -> dict[str, torch.Tensor]:
    """Stack per-expert quantized MoE projections into fused expert-major tensors.

    GPTQ/AWQ MoE checkpoints store every routed expert as a separate module
    (``…{qmoe_target_path}.experts.{i}.{gate,up,down}_proj.*``). The native QMoE
    packer (:func:`pack_qmoe_expert_weights`) instead expects the fused
    expert-major tensors that Olive emits
    (``…{qmoe_target_path}.experts.gate_up_proj.*`` and ``…experts.down_proj.*``
    with a leading expert dimension). This bridges the two so a per-expert
    GPTQ/AWQ checkpoint can drive the fused ``com.microsoft::QMoE`` path.

    Runs on the already-reshaped MatMulNBits layout produced by
    :func:`preprocess_gptq_weights` / :func:`preprocess_awq_weights` (suffixes
    ``.weight`` ``[N, n_blocks, blob]``, ``.scales`` ``[N, n_blocks]``,
    ``.zero_points`` ``[N, zp_cols]``). For each expert, gate and up are
    concatenated along the output-row dimension in HF-concatenated ``[gate; up]``
    order (the layout ``_interleave_gate_up_rows`` expects), then all experts are
    stacked into a new leading dimension. Non-expert keys pass through unchanged;
    per-expert ``bias`` and ``g_idx`` are dropped (the QMoE ABI carries neither).

    A no-op (returns the input unchanged) when the state dict already holds
    fused expert-major tensors (e.g. Olive checkpoints), so it is safe to call
    on any quantized MoE state dict.
    """
    experts_root = f"{qmoe_target_path}.experts."
    projections = ("gate_proj", "up_proj", "down_proj")
    stack_suffixes = ("weight", "scales", "zero_points")

    # groups[expert_prefix][suffix][proj][expert_index] = tensor
    groups: dict[str, dict[str, dict[str, dict[int, torch.Tensor]]]] = {}
    passthrough: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        marker = experts_root
        pos = key.find(marker)
        matched = False
        if pos != -1:
            rest = key[pos + len(marker) :]  # e.g. "0.gate_proj.weight"
            idx_str, _, tail = rest.partition(".")
            if idx_str.isdigit() and tail:
                proj, _, suffix = tail.partition(".")
                if proj in projections:
                    if suffix in stack_suffixes:
                        prefix = key[: pos + len(marker) - 1]  # ".../experts"
                        groups.setdefault(prefix, {}).setdefault(suffix, {}).setdefault(
                            proj, {}
                        )[int(idx_str)] = value
                        matched = True
                    elif suffix in ("bias", "g_idx"):
                        # Not carried into the QMoE ABI; drop.
                        matched = True
        if not matched:
            passthrough[key] = value

    if not groups:
        return state_dict

    def _stack_dense(by_idx: dict[int, torch.Tensor]) -> torch.Tensor:
        return torch.stack([by_idx[i] for i in range(len(by_idx))], dim=0)

    result = dict(passthrough)
    for prefix, by_suffix in groups.items():
        for suffix, by_proj in by_suffix.items():
            gate = by_proj.get("gate_proj")
            up = by_proj.get("up_proj")
            down = by_proj.get("down_proj")
            if gate is not None and up is not None:
                if gate.keys() != up.keys():
                    raise ValueError(f"Mismatched gate/up experts for {prefix} ({suffix})")
                fused_gate_up = torch.stack(
                    [torch.cat([gate[i], up[i]], dim=0) for i in range(len(gate))],
                    dim=0,
                )
                result[f"{prefix}.gate_up_proj.{suffix}"] = fused_gate_up
            if down is not None:
                result[f"{prefix}.down_proj.{suffix}"] = _stack_dense(down)
    return result


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
            result[new_key] = _reshape_packed_qzeros(value, bits, n_blocks)

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
    embed_key: str = "model.embed_tokens.weight",
    head_key: str = "lm_head.weight",
) -> dict[str, torch.Tensor]:
    """Rename and reshape Olive-packed quantized weights.

    Olive stores the quantized state of a parameter named ``<pname>`` (e.g.
    ``"weight"`` for ``nn.Linear``/``nn.Embedding``, or ``"gate_up_proj"`` for
    a fused-3D MoE expert parameter) as sibling buffers on the owning module,
    using an *underscore* suffix convention (see Olive's
    ``olive/common/quant/state_dict.py``) — **not** a dotted one:
      - ``<pname>_qweight``: [N, packed_K] uint8 (always present)
      - ``<pname>_scales``: [N, n_blocks] (always present)
      - ``<pname>_qzeros``: [N, ceil(n_blocks * bits / 8)] uint8 (asymmetric only)

    e.g. ``model.layers.0.mlp.gate_proj.weight_qweight`` (regular ``nn.Linear``)
    or ``model.layers.0.mlp.experts.gate_up_proj_qweight`` (fused MoE param,
    no ``.weight`` component since the parameter itself has no nested Linear).

    Linear projections target ``MatMulNBits``, which expects ``weight`` as
    [N, n_blocks, blob_size]; scales and zero-points already match the
    expected orientation, so they are renamed but not transposed.

    The input embedding table identified by ``embed_key`` instead targets
    ``GatherBlockQuantized``, which consumes the **2-D** uint8 ``qweight``
    directly. Its sidecars are renamed beneath the same owning module while
    preserving the table layout; similarly named linear projections continue
    through the ordinary MatMulNBits conversion.

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
        embed_key: Float embedding key used for the tied-head fallback.
        head_key: Float LM-head key used for the tied-head fallback.

    Returns:
        State dict with renamed and reshaped weights (and, for a float tied
        head, the aliased ``lm_head.weight``).
    """
    blob_size = group_size * bits // 8
    result: dict[str, torch.Tensor] = {}

    def _rename(key: str, raw_suffix: str, dotted_name: str) -> str:
        """Convert an Olive ``<owner>[.weight]{raw_suffix}`` key to ``<owner>.<dotted_name>``.

        Olive's ``pname`` is either the bare ``"weight"`` parameter of an
        ``nn.Linear``/``nn.Embedding`` (so the raw key carries a redundant
        ``.weight`` component before the suffix, e.g. ``...gate_proj.weight_qweight``)
        or a fused MoE parameter with no nested Linear (e.g.
        ``...experts.gate_up_proj_qweight``). Strip the suffix, drop a
        trailing ``.weight`` if present, then append ``.<dotted_name>`` so
        both shapes land on a single canonical ``<owner>.<dotted_name>`` key.
        """
        stem = key[: -len(raw_suffix)]
        if stem.endswith(".weight"):
            stem = stem[: -len(".weight")]
        return f"{stem}.{dotted_name}"

    embed_qweight_key = f"{embed_key}_qweight"
    embed_qzeros_key = f"{embed_key}_qzeros"
    embed_scales_key = f"{embed_key}_scales"
    for key, value in state_dict.items():
        is_embed_qweight = key == embed_qweight_key
        is_embed_qzeros = key == embed_qzeros_key
        is_embed_scales = key == embed_scales_key
        if (
            is_embed_qweight or is_embed_qzeros or is_embed_scales
        ) and not quantize_embeddings:
            # These Olive keys only exist when the embedding table itself was
            # quantized; ``quantize_embeddings=False`` means the caller
            # expects a float embedding, so a packed embedding key showing up
            # anyway indicates a caller/config mismatch rather than a case to
            # silently reroute through the generic Linear renaming below
            # (which would wrongly 3-D reshape ``qweight`` and break
            # ``GatherBlockQuantized``'s 2-D contract).
            raise ValueError(
                f"Found packed embedding quantization key {key!r} but "
                "quantize_embeddings=False; the caller's quantize_embeddings "
                "flag doesn't match the state dict."
            )
        if is_embed_qweight and quantize_embeddings:
            # GatherBlockQuantized consumes the 2-D uint8 table directly, so
            # this keeps the ``qweight`` name (unlike MatMulNBits linears,
            # which rename to ``weight`` below).
            if value.dtype != torch.uint8:
                raise ValueError(
                    f"Olive embedding qweight must be uint8 for {key}, got {value.dtype}"
                )
            if value.shape[-1] % blob_size != 0:
                raise ValueError(
                    f"Olive embedding qweight packed dimension for {key} "
                    f"({value.shape[-1]}) must be divisible by blob_size ({blob_size})"
                )
            result[key[: -len("weight_qweight")] + "qweight"] = value.contiguous()
        elif is_embed_qzeros and quantize_embeddings:
            if value.dtype != torch.uint8:
                raise ValueError(
                    f"Olive embedding qzeros must be uint8 for {key}, got {value.dtype}"
                )
            result[key[: -len("weight_qzeros")] + "zero_points"] = value.contiguous()
        elif is_embed_scales and quantize_embeddings:
            result[key[: -len("weight_scales")] + "scales"] = value
        elif key.endswith("_qweight"):
            if value.dtype != torch.uint8:
                raise ValueError(f"Olive qweight must be uint8 for {key}, got {value.dtype}")
            if value.shape[-1] % blob_size != 0:
                raise ValueError(
                    f"Olive qweight packed dimension for {key} ({value.shape[-1]}) "
                    f"must be divisible by blob_size ({blob_size})"
                )
            new_key = _rename(key, "_qweight", "weight")
            # Preserve all leading dims so fused expert-major tensors
            # ``[E, N, packed_K]`` reshape to ``[E, N, n_blocks, blob_size]``
            # for the QMoE repacker; 2-D linears ``[N, packed_K]`` are
            # unchanged (``[N, n_blocks, blob_size]``).
            result[new_key] = value.reshape(*value.shape[:-1], -1, blob_size).contiguous()
        elif key.endswith("_qzeros"):
            if value.dtype != torch.uint8:
                raise ValueError(f"Olive qzeros must be uint8 for {key}, got {value.dtype}")
            new_key = _rename(key, "_qzeros", "zero_points")
            result[new_key] = value.contiguous()
        elif key.endswith("_scales"):
            new_key = _rename(key, "_scales", "scales")
            result[new_key] = value
        else:
            result[key] = value

    # A tied quantized head shares the embedding's Parameters in the module
    # (TiedQuantizedLMHead), so no lm_head initializers exist to fill here.
    # Only a tied *float* head needs an explicit weight alias.
    if (
        tie_word_embeddings
        and not quantize_lm_head
        and head_key not in result
        and embed_key in result
    ):
        result[head_key] = result[embed_key]

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


def preprocess_quantized_weights(
    state_dict: dict[str, torch.Tensor],
    quantization: QuantizationConfig | None,
    *,
    tie_embeddings: bool = False,
    embed_key: str = "model.embed_tokens.weight",
    head_key: str = "lm_head.weight",
    qmoe_target_path: str | None = None,
    qmoe_quant_methods: Collection[str] = ("gptq", "awq", "olive"),
    reject_quantized_embeddings_lm_head: bool = False,
) -> dict[str, torch.Tensor]:
    """Apply shared quantization conversion, tying, and QMoE packing.

    Args:
        state_dict: Weight dictionary to preprocess.
        quantization: Quantization format and packing settings, or ``None`` for
            float weights.
        tie_embeddings: Whether to alias missing float embedding/head weights.
        embed_key: Canonical float embedding-table key.
        head_key: Canonical float LM-head key.
        qmoe_target_path: MoE path replaced by packed QMoE parameter names.
            ``None`` means this model does not support or require QMoE handling.
        qmoe_quant_methods: Quantization methods supported by this caller when
            the config matches the native QMoE ABI.
        reject_quantized_embeddings_lm_head: Reject packed embedding/head
            weights because the caller's graph requires float parameters.

    Returns:
        The preprocessed weight dictionary.

    GPTQ and AWQ conversion falls through to the shared float-tying and QMoE
    packing tail. Olive returns from its own branch because
    :func:`preprocess_olive_weights` must handle quantized embedding/head tying
    internally before optional QMoE packing.
    """
    use_qmoe = (
        supported_qmoe_quantization(quantization) is not None
        if qmoe_target_path is not None
        else False
    )
    if (
        use_qmoe
        and quantization is not None
        and quantization.quant_method not in qmoe_quant_methods
    ):
        methods = tuple(qmoe_quant_methods)
        if methods == ("olive",):
            raise NotImplementedError(
                "This model currently only supports QMoE export for "
                "Olive-quantized checkpoints (quant_method='olive'); got "
                f"quant_method={quantization.quant_method!r}. GPTQ/AWQ VL-MoE "
                "export is not yet implemented."
            )
        raise NotImplementedError(
            "This model supports QMoE export for quant_method values "
            f"{methods!r}; got quant_method={quantization.quant_method!r}."
        )

    if qmoe_target_path is not None and not use_qmoe:
        packed_expert_keys = [
            key
            for key in state_dict
            if qmoe_target_path in key and ".experts." in key and is_packed_quant_key(key)
        ]
        if packed_expert_keys:
            raise ValueError(
                "Quantized MoE expert weights were found for this model "
                f"(e.g. {packed_expert_keys[0]!r}) but this quantization "
                "config doesn't match the native QMoE ABI "
                "(supported_qmoe_quantization returned None). The dense "
                "loop-over-experts fallback only supports unquantized fused "
                "expert tensors, not packed quantized ones. Use a "
                "QMoE-ABI-compatible quantization config (see "
                "supported_qmoe_quantization) for MoE models instead."
            )

    def _is_packed_sidecar_of(key: str, float_key: str) -> bool:
        """Whether ``key`` is a packed sidecar of the specific ``float_key``.

        Unlike the module-level :func:`is_packed_quant_key` suffix predicate,
        this matches the *exact* sidecar keys of one named float parameter.
        """
        owner = float_key.removesuffix(".weight")
        return any(key == float_key + suffix for suffix in OLIVE_PACKED_QUANT_SUFFIXES) or any(
            key == owner + suffix for suffix in DOTTED_PACKED_QUANT_SUFFIXES
        )

    packed_embedding_or_head = (
        next(
            (
                key
                for key in state_dict
                if _is_packed_sidecar_of(key, embed_key)
                or _is_packed_sidecar_of(key, head_key)
            ),
            None,
        )
        if reject_quantized_embeddings_lm_head
        else None
    )
    configured_quantized_table = quantization is not None and (
        quantization.quantize_embeddings or quantization.quantize_lm_head
    )
    if reject_quantized_embeddings_lm_head and (
        configured_quantized_table or packed_embedding_or_head is not None
    ):
        packed_key_detail = (
            f" Packed checkpoint key {packed_embedding_or_head!r} was found."
            if packed_embedding_or_head is not None
            else ""
        )
        raise NotImplementedError(
            "Quantized embeddings and LM heads are not yet supported by "
            f"this model: {embed_key.removesuffix('.weight')}, "
            f"{head_key.removesuffix('.weight')}, and related embedding "
            f"initializers currently require float weights.{packed_key_detail}"
        )

    if quantization is not None and quantization.quant_method == "gptq":
        state_dict = preprocess_gptq_weights(
            state_dict, bits=quantization.bits, group_size=quantization.group_size
        )
    elif quantization is not None and quantization.quant_method == "awq":
        state_dict = preprocess_awq_weights(
            state_dict, bits=quantization.bits, group_size=quantization.group_size
        )
    elif quantization is not None and quantization.quant_method == "olive":
        olive_tie = tie_embeddings or quantization.tie_word_embeddings
        return_state_dict = preprocess_olive_weights(
            state_dict,
            bits=quantization.bits,
            group_size=quantization.group_size,
            quantize_embeddings=quantization.quantize_embeddings,
            quantize_lm_head=quantization.quantize_lm_head,
            tie_word_embeddings=olive_tie,
            embed_key=embed_key,
            head_key=head_key,
        )
        if use_qmoe and qmoe_target_path is not None:
            return_state_dict = stack_per_expert_moe_weights(
                return_state_dict, qmoe_target_path=qmoe_target_path
            )
            return_state_dict = pack_qmoe_expert_weights(
                return_state_dict, target_moe_path=qmoe_target_path
            )
        return return_state_dict

    tied_quantized_table = (
        quantization is not None
        and quantization.quantize_embeddings
        and quantization.quantize_lm_head
    )
    if tie_embeddings and not tied_quantized_table:
        tie_word_embeddings(state_dict, embed_key=embed_key, head_key=head_key)
    if use_qmoe and qmoe_target_path is not None:
        state_dict = stack_per_expert_moe_weights(
            state_dict, qmoe_target_path=qmoe_target_path
        )
        state_dict = pack_qmoe_expert_weights(state_dict, target_moe_path=qmoe_target_path)
    return state_dict
