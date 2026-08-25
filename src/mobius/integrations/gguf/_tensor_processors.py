# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Architecture-specific tensor processors for GGUF import.

GGUF files store tensors in different layouts or conventions than
HuggingFace checkpoints. This module applies the necessary transforms
after GGUF→HF name mapping but before ``preprocess_weights()``.

Known transforms (mirrored from HF's ``modeling_gguf_pytorch_utils.py``):

- **Llama/Mistral** — Q/K weight reverse-permutation. GGUF stores
  Q/K weights interleaved as ``(n_head, dim//2, 2, ...)``; HF uses
  standard ``(n_head * head_dim, hidden_size)`` layout.
- **Gemma2/Gemma3** — Norm weight offset. GGUF stores
  ``w_gguf = w_hf - 1``; we restore by adding 1.
- **Nemotron** — Same norm weight offset as Gemma.
- **GPT-2** — Weight transpose for attn and FFN projections.
- **Mamba** — ``conv1d.weight`` needs an extra dimension;
  ``A_log`` needs ``log(-x)`` transform.

Usage::

    from mobius.integrations.gguf._tensor_processors import (
        process_tensors,
    )

    state_dict = process_tensors(state_dict, config)
"""

from __future__ import annotations

__all__ = [
    "process_tensors",
    "_reverse_permute",
    "needs_llama_qk_permute",
    "LLAMA_QK_PERMUTE_MODEL_TYPES",
]

import logging
from typing import Any

import torch

from mobius.integrations.gguf._arch_registry import iter_arch_specs, try_get_arch_spec

logger = logging.getLogger(__name__)

#: mobius ``model_type`` values that no GGUF architecture maps to, but which
#: still store Q/K with the llama.cpp interleaved-rope permutation, so a caller
#: passing one directly must get the right answer.
#:
#: ``mistral`` is Llama-architecture; llama.cpp writes Mistral checkpoints with
#: ``general.architecture = "llama"``, so no spec produces this model_type.
_EXTRA_QK_PERMUTE_MODEL_TYPES = frozenset({"mistral"})

# Model types whose GGUF Q/K weights are stored with llama.cpp's
# interleaved-rope permutation and therefore require reverse-permutation
# on import. These are the architectures whose llama.cpp converter calls
# ``permute()`` on ``attn_q``/``attn_k`` (GGML "normal" rope).
#
# Qwen2/Qwen3, Gemma, GPT-2, Mamba, etc. use non-interleaved (NEOX-style)
# rope and store Q/K in plain HF row order — applying the permute to them
# scrambles the attention heads and produces garbage output. They must
# NOT be reverse-permuted.
#
# Derived from the architecture registry so that declaring a new architecture
# cannot leave this set behind.
LLAMA_QK_PERMUTE_MODEL_TYPES = (
    frozenset(
        spec.model_type
        for spec in iter_arch_specs()
        if spec.llama_qk_permute and spec.model_type is not None
    )
    | _EXTRA_QK_PERMUTE_MODEL_TYPES
)


def needs_llama_qk_permute(model_type: str | None) -> bool:
    """Return True if this model type needs llama.cpp Q/K reverse-permute."""
    return model_type in LLAMA_QK_PERMUTE_MODEL_TYPES


def _resolve_processor(config: Any) -> Any:
    """Return the weight processor for *config*, or ``None``.

    Dispatch prefers the GGUF architecture recorded on the config, because that
    is the identity the registry is keyed on. Falling back to ``model_type`` is
    what silently broke Gemma 3: its processor was registered under ``gemma3``
    while GGUF ``gemma3`` resolves to model_type ``gemma3_text``, so the Gemma
    norm un-offset never ran and every norm was left with llama.cpp's baked-in
    ``+1`` on top of the ``OffsetRMSNorm`` the graph applies at runtime.

    The ``model_type`` path is retained for callers that build a config without
    going through :func:`gguf_to_config`.
    """
    gguf_arch = getattr(config, "_gguf_arch", None)
    if gguf_arch is not None:
        spec = try_get_arch_spec(gguf_arch)
        if spec is not None:
            name = spec.tensor_processor
            return None if name is None else _PROCESSOR_IMPLS[name]

    model_type = getattr(config, "model_type", None)
    if model_type is None:
        return None
    name = _LEGACY_MODEL_TYPE_PROCESSORS.get(model_type)
    return None if name is None else _PROCESSOR_IMPLS[name]


def process_tensors(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Apply architecture-specific tensor transformations.

    Dispatches on the GGUF architecture recorded on *config*, falling back to
    ``config.model_type``. If no processor applies, the state dict is returned
    unchanged.

    Args:
        state_dict: HuggingFace-named state dict from GGUF import.
        config: The :class:`ArchitectureConfig` for this model.
            Must have ``model_type``, ``num_attention_heads``, and
            ``num_key_value_heads`` attributes.

    Returns:
        The transformed state dict (modified in-place).
    """
    processor = _resolve_processor(config)
    if processor is None:
        return state_dict

    return processor(state_dict, config)


# ---------------------------------------------------------------------------
# Processor implementations
# ---------------------------------------------------------------------------


def _process_llama(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Reverse-permute Q/K weights and biases for Llama/Mistral.

    Reference: ``LlamaTensorProcessor`` in HF's
    ``modeling_gguf_pytorch_utils.py``.
    Reference: https://github.com/ggerganov/llama.cpp/blob/
    a38b884c6c4b0c256583acfaaabdf556c62fabea/convert_hf_to_gguf.py#L1402
    """
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_heads is None or num_kv_heads is None:
        logger.warning(
            "Cannot reverse-permute Q/K weights: "
            "num_attention_heads or num_key_value_heads not in config"
        )
        return state_dict
    for name, tensor in state_dict.items():
        if ".q_proj." in name and name.endswith((".weight", ".bias")):
            state_dict[name] = _reverse_permute(tensor, num_heads)
        elif ".k_proj." in name and name.endswith((".weight", ".bias")):
            state_dict[name] = _reverse_permute(tensor, num_kv_heads)

    return state_dict


def _reverse_permute(
    weights: torch.Tensor,
    n_head: int,
) -> torch.Tensor:
    """Reverse the Q/K weight permutation applied by llama.cpp.

    llama.cpp's forward permute (``convert_hf_to_gguf.py``) is::

        weights.reshape(n_head, 2, dim, ...).swapaxes(1, 2).reshape(orig)

    where ``dim = out_features // n_head // 2``. The exact inverse — and
    the transform HF's ``modeling_gguf_pytorch_utils._reverse_permute_weights``
    applies when loading GGUF — reshapes with ``dim`` and ``2`` swapped::

        weights.reshape(n_head, dim, 2, ...).swapaxes(1, 2).reshape(orig)

    Using the forward reshape order here (``(n_head, 2, dim)``) only
    coincidentally inverts the permute when ``dim == 2`` (head_dim == 4);
    for real head dims (e.g. 64) it scrambles the Q/K rows, corrupting
    rope and producing garbage output.

    Args:
        weights: The Q or K projection weight tensor.
        n_head: Number of heads for this tensor — ``num_attention_heads``
            for Q weights, ``num_key_value_heads`` for K weights.
    """
    dim = weights.shape[0] // n_head // 2
    w = weights.reshape(n_head, dim, 2, *weights.shape[1:])
    return w.swapaxes(1, 2).reshape(weights.shape)


def _process_unoffset_norm(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Undo the ``+1`` llama.cpp bakes into centered norm weights.

    Several architectures scale by ``(1 + w)`` rather than by ``w``: Gemma's
    ``Gemma*RMSNorm`` and Nemotron's ``NemotronLayerNorm1P`` both store ``w_hf``
    in the HuggingFace checkpoint and add one at runtime. llama.cpp has no
    offset norm, so its converters fold the constant in and write
    ``w_gguf = w_hf + 1`` — see ``conversion/nemotron.py`` at the pinned commit
    ("Adding +1 to LayerNorm's weights here to implement layernorm1p w/o
    changing anything on the GGML engine side") and the Gemma equivalents.

    The mobius graphs normalize with :class:`OffsetRMSNorm` /
    :class:`OffsetLayerNorm`, which re-apply the ``1 +`` themselves, so the
    initializer they consume must be the raw ``w_hf``. Subtract one to undo the
    fold; otherwise the offset lands twice and every norm — hence the whole
    model — is corrupted.

    This matches HF's ``Gemma2TensorProcessor`` **and** ``NemotronTensorProcessor``
    in ``modeling_gguf_pytorch_utils.py``, both of which subtract one. mobius
    previously had a separate Nemotron processor that added one instead, leaving
    Nemotron GGUF imports scaling by ``w_hf + 3`` instead of ``w_hf + 1``.
    """
    for name in list(state_dict):
        if "norm" in name and name.endswith(".weight"):
            state_dict[name] = state_dict[name] - 1
    return state_dict


def _process_muse_glimmer(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Restore Muse Glimmer's centered per-block norm weights.

    Muse Glimmer's four per-block norms are *centered*: the HF checkpoint stores
    ``w`` and the model multiplies by ``w + 1``. llama.cpp folds the ``+ 1`` in
    at conversion time so its generic RMSNorm can use the tensor directly, so a
    GGUF file holds ``w_gguf = w_hf + 1`` and importing it needs the offset
    removed. This is the opposite direction from Gemma/Nemotron, which store
    ``w_hf - 1``.

    The final ``model.norm`` is a plain RMSNorm in this architecture and is
    stored uncentered, so it must be left alone -- verified against the
    published checkpoint, where ``model.norm.weight`` is centered on 0 while the
    per-block norms are centered on 1.

    Muse Glimmer's llama.cpp converter also stores Q/K with the interleaved-rope
    permutation, on every layer including the NoPE (full-attention) ones, so the
    llama reverse-permute has to run as well.
    """
    state_dict = _process_llama(state_dict, config)
    for name in list(state_dict):
        if not name.endswith(".weight"):
            continue
        if ".layers." not in name:
            continue
        if "layernorm" not in name:
            continue
        state_dict[name] = state_dict[name] - 1
    return state_dict


def _process_gpt2(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Transpose GPT-2 attention and FFN weights.

    GGUF stores these transposed relative to HF convention.
    Reference: ``GPT2TensorProcessor`` in HF's
    ``modeling_gguf_pytorch_utils.py``.
    """
    for name, tensor in list(state_dict.items()):
        needs_transpose = (
            ".c_attn." in name or ".c_proj." in name or ".c_fc." in name
        ) and name.endswith(".weight")
        if needs_transpose:
            state_dict[name] = tensor.T
    return state_dict


def _process_mamba(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Fix Mamba tensor shapes and transforms.

    - ``conv1d.weight``: unsqueeze dim 1 (GGUF is 2D, HF is 3D)
    - ``A_log``: GGUF stores ``-exp(A_log)``; restore with ``log(-x)``
    - Mamba2 ``A_log``/``D``: squeeze llama.cpp's trailing singleton axis

    Reference: ``MambaTensorProcessor`` in HF's
    ``modeling_gguf_pytorch_utils.py``.
    """
    layer_types = getattr(config, "layer_types", None) or ()
    is_mamba2 = config.model_type in {"mamba2", "falcon_h1"} or "mamba2" in layer_types
    for name, tensor in list(state_dict.items()):
        if "conv1d" in name and name.endswith(".weight"):
            if tensor.dim() == 2:
                state_dict[name] = tensor.unsqueeze(1)
        elif name.endswith(".A_log"):
            if not torch.all(tensor < 0):
                raise ValueError(
                    f"Malformed GGUF Mamba decay tensor {name!r}: "
                    "ssm_a must contain only negative -exp(A_log) values"
                )
            tensor = torch.log(-tensor)
            if is_mamba2 and tensor.dim() == 2 and tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
            state_dict[name] = tensor
        elif is_mamba2 and name.endswith(".D") and tensor.dim() == 2 and tensor.shape[-1] == 1:
            state_dict[name] = tensor.squeeze(-1)
        elif is_mamba2 and name.endswith(".norm.weight") and tensor.dim() == 2:
            state_dict[name] = tensor.flatten()
    return state_dict


def _process_plamo2(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Apply PLaMo2's exact GGUF-to-graph value and shape transforms."""
    state_dict = _process_mamba(state_dict, config)
    # llama.cpp has already folded the official additive norm constants.
    # Preserve those serialized float values exactly through model preprocessing.
    config._plamo2_norms_are_folded = True
    for name in tuple(state_dict):
        if name.endswith(".dt_proj.bias"):
            state_dict[name.removesuffix(".dt_proj.bias") + ".dt_bias"] = state_dict.pop(name)
    return state_dict


def _process_kimi_linear(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Invert the pinned Kimi Linear converter's recurrent tensor transforms."""
    del config
    for name in tuple(state_dict):
        tensor = state_dict[name]
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if tensor.dim() != 4 or tensor.shape[0] != 1 or tensor.shape[2] != 1:
                raise ValueError(
                    f"Kimi Linear convolution tensor {name!r} must have shape "
                    f"[1, channels, 1, kernel], got {tuple(tensor.shape)}"
                )
            state_dict[name] = tensor.reshape(tensor.shape[1], tensor.shape[3])
        elif name.endswith(".A_log"):
            if not torch.all(torch.isfinite(tensor)) or not torch.all(tensor < 0):
                raise ValueError(
                    f"Kimi Linear decay tensor {name!r} must contain finite negative values"
                )
            state_dict[name] = torch.log(-tensor)
        elif name.endswith(".k_b_proj.weight"):
            if tensor.dim() != 3:
                raise ValueError(
                    f"Kimi Linear K-B tensor {name!r} must be rank 3, got {tensor.dim()}"
                )
            state_dict[name] = tensor.transpose(1, 2).reshape(
                tensor.shape[0] * tensor.shape[2], tensor.shape[1]
            )
        elif name.endswith(".v_b_proj.weight"):
            if tensor.dim() != 3:
                raise ValueError(
                    f"Kimi Linear V-B tensor {name!r} must be rank 3, got {tensor.dim()}"
                )
            state_dict[name] = tensor.reshape(
                tensor.shape[0] * tensor.shape[1], tensor.shape[2]
            )
    return state_dict


def _process_granitehybrid(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Invert Granite transforms and reconstruct fused shared/expert gate-up weights."""
    state_dict = _process_llama(state_dict, config)
    state_dict = _process_mamba(state_dict, config)
    for name in list(state_dict):
        if name.endswith(".shared_mlp.gate_proj.weight"):
            prefix = name.removesuffix("gate_proj.weight")
            up_name = f"{prefix}up_proj.weight"
            if up_name not in state_dict:
                raise ValueError(
                    f"GraniteHybrid shared FFN is missing paired tensor {up_name!r}"
                )
            state_dict[f"{prefix}input_linear.weight"] = torch.cat(
                (state_dict.pop(name), state_dict.pop(up_name)), dim=0
            )
        elif name.endswith(".block_sparse_moe.gate_proj.weight"):
            prefix = name.removesuffix("gate_proj.weight")
            up_name = f"{prefix}up_proj.weight"
            if up_name not in state_dict:
                raise ValueError(
                    f"GraniteHybrid routed experts are missing paired tensor {up_name!r}"
                )
            # GGUF stores [E, F, H] gate and up tensors separately while the
            # Transformers/Mobius module stores [E, 2F, H] in gate-then-up order.
            state_dict[f"{prefix}input_linear.weight"] = torch.cat(
                (state_dict.pop(name), state_dict.pop(up_name)), dim=1
            )
    return state_dict


# Named weight processors. The architecture registry refers to these by name,
# which is why the table is keyed on the processor's own identity rather than on
# a model_type. Every name here must be referenced by at least one architecture
# spec, and every name a spec references must exist here; ``_arch_registry_test``
# checks both directions, so an orphaned processor or a typo in a spec fails the
# suite instead of silently doing nothing.
#
# NOTE: Qwen2/Qwen3 deliberately have no processor. Unlike Llama/Mistral,
# llama.cpp does not permute Qwen Q/K weights (Qwen uses NEOX-style rope), so
# reverse-permuting them corrupts the attention heads. See
# ``LLAMA_QK_PERMUTE_MODEL_TYPES``.
_PROCESSOR_IMPLS: dict[str, Any] = {
    "llama": _process_llama,
    "unoffset_norm": _process_unoffset_norm,
    "muse_glimmer": _process_muse_glimmer,
    "gpt2": _process_gpt2,
    "mamba": _process_mamba,
    "plamo2": _process_plamo2,
    "granitehybrid": _process_granitehybrid,
    "kimi_linear": _process_kimi_linear,
}

#: mobius ``model_type`` values that no GGUF architecture maps to, but that a
#: caller may pass directly in a hand-built config.
#:
#: ``mistral`` is Llama-architecture (llama.cpp writes it as ``llama``).
#: ``gemma3`` is the *multimodal* Gemma 3 model type; ``models/gemma3.py``
#: normalizes with ``OffsetRMSNorm``, so it needs the Gemma un-offset. GGUF
#: ``gemma3`` resolves to ``gemma3_text`` and is handled through its spec.
_EXTRA_MODEL_TYPE_PROCESSORS: dict[str, str] = {
    "mistral": "llama",
    "gemma3": "unoffset_norm",
}

#: Fallback ``model_type`` → processor-name map for configs that did not come
#: from :func:`~mobius.integrations.gguf._config_mapping.gguf_to_config` and so
#: carry no ``_gguf_arch``. Derived from the registry, then extended.
_LEGACY_MODEL_TYPE_PROCESSORS: dict[str, str] = {
    **{
        spec.model_type: spec.tensor_processor
        for spec in iter_arch_specs()
        if spec.model_type is not None and spec.tensor_processor is not None
    },
    **_EXTRA_MODEL_TYPE_PROCESSORS,
}
