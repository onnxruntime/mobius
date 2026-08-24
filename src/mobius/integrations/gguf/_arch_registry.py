# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Single source of truth for GGUF architecture support.

Every architecture mobius can say anything about gets exactly one
:class:`~mobius.integrations.gguf._spec.GGUFArchitectureSpec` here. Before this
module existed the same question was answered by nine hand-maintained
containers spread across five modules, keyed inconsistently on either the GGUF
``general.architecture`` string or the mobius ``model_type``. They disagreed:

* ``bloom`` and ``t5`` resolved to real model types but had no tensor mapping,
  so config extraction succeeded and the build then died contradicting itself.
* ``gemma``, ``internlm2``, ``qwen2moe`` and ``qwen3moe`` had tensor mappings
  but no config entry, so they fell through an unmapped ``.get(arch, arch)``.
* ``qwen2_moe``/``qwen3_moe``/``mistral``/``muse_glimmer``/``hunyuan_v1_dense``
  sat in architecture-keyed containers even though llama.cpp never emits them —
  they are mobius ``model_type`` strings that leaked into the architecture
  namespace, or defensive spellings.
* ``gemma3`` mapped to ``model_type`` ``gemma3_text`` while its weight processor
  was registered under ``gemma3``, so the Gemma norm un-offset silently never
  ran for Gemma 3 imports.

Two rules keep that from recurring:

1. **Capabilities are separate verdicts.** ``config``, ``tensor_map``, ``graph``
   and ``runtime`` are answered independently, and anything short of
   ``SUPPORTED`` must carry a reason. Being listed here is not a support claim.
2. **Behavior is referenced by name, never by callable.** Each spec names the
   config postprocessor, tensor-mapping recipe, and weight processor it wants;
   the module that owns each implementation resolves the name. That keeps this
   module an import leaf (it imports only ``_spec``, ``_errors`` and
   ``_upstream``) and makes both an unknown name and an unreferenced
   implementation a test failure.

Canonical ``gguf_arch`` values are validated against the pinned llama.cpp
census, so a mobius ``model_type`` can no longer masquerade as an architecture.
Defensive spellings belong in ``aliases``.
"""

from __future__ import annotations

__all__ = [
    "MMPROJ_ARCHITECTURE",
    "arch_names_with",
    "get_arch_spec",
    "iter_arch_specs",
    "model_type_for",
    "supported_architectures",
    "try_get_arch_spec",
]

import functools
from collections.abc import Callable
from types import MappingProxyType

from mobius.integrations.gguf._errors import (
    DisabledGGUFArchitectureError,
    UnsupportedGGUFArchitectureError,
)
from mobius.integrations.gguf._spec import GGUFArchitectureSpec, Support
from mobius.integrations.gguf._upstream import upstream_architecture, upstream_architectures

#: ``general.architecture`` of a multimodal projector sidecar. Upstream this is
#: a quant-only stub, not a loadable model, so it never goes down the text path.
MMPROJ_ARCHITECTURE = "clip"

# Reasons are written once and shared so the same rejection reads identically
# wherever it surfaces.
_NO_TENSOR_MAP = (
    "mobius has no GGUF→HuggingFace tensor-name mapping for this architecture, so "
    "its weights cannot be routed into a graph. Build from the Hugging Face "
    "checkpoint with `mobius build` instead."
)

_NEMOTRON_H_MOE_REASON = (
    "Direct GGUF conversion is intentionally disabled. GGUF block_count includes a "
    "combined attention+MoE MTP auxiliary block, so aliasing it to the 52-layer "
    "'nemotron_h' backbone would build the wrong graph. The Nemotron-H Mamba2 path "
    "also lacks passing full-logit/generation parity, and common GGUF presets "
    "contain Q5_0/Q5_1 expert tensors that cannot be preserved by MatMulNBits. Use "
    "llama.cpp/Unsloth to run the GGUF without changing its quantization, or start "
    "from the official pinned BF16 Hugging Face checkpoint and quantize the "
    "validated ONNX export with Olive only after L4/L5 semantic generation passes. "
    "See docs/api/build_from_gguf.md for the pinned recipe and waiver."
)

_MMPROJ_REASON = (
    "This is a multimodal projector sidecar, not a language model. Upstream it is a "
    "quant-only stub whose runtime lives outside libllama. Pass it to "
    "`build_from_gguf(text_gguf, mmproj=...)` alongside its text backbone rather "
    "than building it directly."
)

_RUNTIME_VALIDATION_PENDING = (
    "Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build "
    "are covered, but no representative real-weight GGUF has yet passed ORT parity or "
    "generation validation. Runtime packaging remains deferred until that evidence exists."
)

_NO_QUANTIZED_PROJECTION_REASON = (
    "The mobius graph uses floating Linear modules for this architecture, so no "
    "MatMulNBits or BlockQuantizedMatMul target can consume preserved GGUF projection "
    "weights. Use keep_quantized=False for explicit float import."
)

_RECURRENT_RUNTIME_VALIDATION_PENDING = (
    "Config extraction, exact pinned tensor-name closure, GGUF value transforms, and "
    "synthetic recurrent-state execution are covered, but no representative real-weight "
    "GGUF has yet passed independent full-logit parity and deterministic multi-token "
    "stateful ORT generation. Runtime packaging remains deferred until that evidence exists."
)

_NO_RWKV_GRAPH = (
    "llama.cpp implements this as a distinct RWKV recurrent architecture. Mobius has no "
    "matching RWKV model graph or state-cache contract, so mapping it to Mamba would build "
    "the wrong recurrence. Build from a supported source/runtime instead."
)


_SPECS: tuple[GGUFArchitectureSpec, ...] = (
    # ---------------------------------------------------------------- Llama
    GGUFArchitectureSpec(
        gguf_arch="llama",
        model_type="llama",
        # llama.cpp writes Mistral checkpoints with architecture "llama"; the
        # "mistral" string is not an upstream architecture, so it is accepted
        # defensively rather than treated as canonical.
        aliases=frozenset({"mistral"}),
        tensor_map_recipe=("llama",),
        tensor_processor="llama",
        llama_qk_permute=True,
    ),
    GGUFArchitectureSpec(
        gguf_arch="deci",
        model_type="llama",
        tensor_map_recipe=("llama",),
        tensor_processor="llama",
        llama_qk_permute=True,
    ),
    # ----------------------------------------------------------------- Qwen
    GGUFArchitectureSpec(
        gguf_arch="qwen2",
        model_type="qwen2",
        tensor_map_recipe=("llama",),
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen3",
        model_type="qwen3",
        tensor_map_recipe=("llama",),
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen2moe",
        model_type="qwen2_moe",
        # "qwen2_moe" is the mobius model_type, which previously sat in the
        # architecture map by mistake. Keep accepting it, but canonically this
        # architecture is spelled the way llama.cpp writes it.
        aliases=frozenset({"qwen2_moe"}),
        tensor_map_recipe=("llama", "moe_extras"),
        config_postprocessor="moe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        ),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen3moe",
        model_type="qwen3_moe",
        aliases=frozenset({"qwen3_moe"}),
        tensor_map_recipe=("llama", "moe_qk_norm_extras", "moe_extras"),
        config_postprocessor="moe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        ),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen35",
        model_type="qwen3_5_text",
        tensor_map_recipe=("llama", "qwen35_hybrid_extras"),
        offset_norm=True,
        v_head_reorder=True,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen35moe",
        model_type="qwen3_5_moe",
        tensor_map_recipe=("llama", "moe_extras", "qwen35_hybrid_extras"),
        offset_norm=True,
        v_head_reorder=True,
    ),
    # ---------------------------------------------------------------- Gemma
    GGUFArchitectureSpec(
        gguf_arch="gemma",
        model_type="gemma",
        tensor_map_recipe=("llama",),
        tensor_processor="unoffset_norm",
    ),
    GGUFArchitectureSpec(
        gguf_arch="gemma2",
        model_type="gemma2",
        tensor_map_recipe=("llama", "gemma2_extras"),
        tensor_processor="unoffset_norm",
        config_postprocessor="gemma2",
    ),
    GGUFArchitectureSpec(
        gguf_arch="gemma3",
        model_type="gemma3_text",
        tensor_map_recipe=("llama", "gemma3_extras"),
        # models/gemma3_text.py normalizes with OffsetRMSNorm, so the llama.cpp
        # `+1` baked into every *norm.weight must be removed on import.
        tensor_processor="unoffset_norm",
        config_postprocessor="gemma3",
    ),
    GGUFArchitectureSpec(
        gguf_arch="gemma4",
        model_type="gemma4_text",
        tensor_map_recipe=("llama", "gemma4_extras"),
        # Deliberately no tensor_processor: models/gemma4.py normalizes with
        # plain RMSNorm, not OffsetRMSNorm, so its GGUF weights are already in
        # the form the graph consumes. Applying the Gemma un-offset here would
        # corrupt every norm.
        config_postprocessor="gemma4",
        vlm_builder="gemma4",
    ),
    # -------------------------------------------------------------- Various
    GGUFArchitectureSpec(
        gguf_arch="phi3",
        model_type="phi3",
        tensor_map_recipe=("phi3",),
    ),
    GGUFArchitectureSpec(
        gguf_arch="falcon",
        model_type="falcon",
        tensor_map_recipe=("falcon",),
    ),
    GGUFArchitectureSpec(
        gguf_arch="gpt2",
        model_type="gpt2",
        tensor_map_recipe=("gpt2",),
        tensor_processor="gpt2",
    ),
    GGUFArchitectureSpec(
        gguf_arch="mamba",
        model_type="mamba",
        tensor_map_recipe=("mamba",),
        config_key_map="mamba",
        config_postprocessor="mamba",
        tensor_processor="mamba",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ),
        runtime=Support.DEFERRED,
        reason=_RECURRENT_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="mamba2",
        model_type="mamba2",
        tensor_map_recipe=("mamba2",),
        config_key_map="mamba",
        config_postprocessor="mamba2",
        tensor_processor="mamba",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ),
        runtime=Support.DEFERRED,
        reason=_RECURRENT_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="starcoder2",
        model_type="starcoder2",
        tensor_map_recipe=("llama",),
    ),
    GGUFArchitectureSpec(
        gguf_arch="stablelm",
        model_type="stablelm",
        tensor_map_recipe=("llama",),
    ),
    GGUFArchitectureSpec(
        gguf_arch="internlm2",
        model_type="internlm2",
        tensor_map_recipe=("llama",),
        tensor_processor="llama",
        llama_qk_permute=True,
        quantized_import=Support.REJECTED,
        reason=_NO_QUANTIZED_PROJECTION_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="olmo",
        model_type="olmo",
        tensor_map_recipe=("olmo",),
        config_postprocessor="olmo",
        required_metadata=("attention.layer_norm_epsilon",),
        tensor_processor="llama",
        llama_qk_permute=True,
    ),
    GGUFArchitectureSpec(
        gguf_arch="olmo2",
        model_type="olmo2",
        tensor_map_recipe=("llama", "olmo2_extras"),
        config_postprocessor="dense_sliding",
        required_metadata=("attention.layer_norm_rms_epsilon",),
    ),
    GGUFArchitectureSpec(
        gguf_arch="olmoe",
        model_type="olmoe",
        tensor_map_recipe=("llama", "moe_qk_norm_extras", "moe_extras"),
        config_postprocessor="moe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        ),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="phimoe",
        model_type="phimoe",
        module_type="phimoe_gguf",
        tensor_map_recipe=("llama", "phi3", "moe_extras"),
        config_postprocessor="phimoe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        ),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="granitemoe",
        model_type="granitemoe",
        tensor_map_recipe=("llama", "moe_extras"),
        config_postprocessor="granitemoe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "logit_scale",
        ),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="cohere2",
        model_type="cohere2",
        tensor_map_recipe=("llama",),
        config_postprocessor="dense_sliding",
        required_metadata=(
            "attention.layer_norm_epsilon",
            "attention.sliding_window",
            "logit_scale",
        ),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="arcee",
        model_type="arcee",
        tensor_map_recipe=("arcee",),
        required_metadata=("attention.layer_norm_rms_epsilon",),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="smollm3",
        model_type="smollm3",
        tensor_map_recipe=("llama",),
        config_postprocessor="dense_sliding",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        tensor_processor="llama",
        llama_qk_permute=True,
    ),
    GGUFArchitectureSpec(
        gguf_arch="exaone",
        model_type="exaone",
        tensor_map_recipe=("llama",),
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="nemotron",
        model_type="nemotron",
        tensor_map_recipe=("llama",),
        tensor_processor="unoffset_norm",
    ),
    GGUFArchitectureSpec(
        gguf_arch="hunyuan-dense",
        model_type="hunyuan_v1_dense",
        aliases=frozenset({"hunyuan_v1_dense"}),
        tensor_map_recipe=("llama", "hunyuan_extras"),
    ),
    GGUFArchitectureSpec(
        gguf_arch="muse-glimmer",
        model_type="muse_glimmer_text",
        aliases=frozenset({"muse_glimmer"}),
        tensor_map_recipe=("llama", "muse_glimmer_extras"),
        tensor_processor="muse_glimmer",
        config_key_map="muse_glimmer",
        config_postprocessor="muse_glimmer",
        vlm_builder="muse_glimmer",
        llama_qk_permute=True,
    ),
    GGUFArchitectureSpec(
        gguf_arch="deepseek4",
        model_type="deepseek_v4",
        tensor_map_recipe=("deepseek4",),
        config_key_map="deepseek4",
        rope_interleave=True,
    ),
    # GLM-5.2 GGUFs (e.g. unsloth/GLM-5.2-GGUF) tag the architecture 'glm-dsa'
    # (MLA + DeepSeek Sparse Attention + MoE) and mobius's registry key is
    # 'glm_moe_dsa'. The format bridge is keyed on the authoritative
    # general.architecture value, never on a filename, and
    # ``assert_glm_moe_dsa_resolvable`` verifies the head/layer/expert/DSA
    # properties before the builder selects GlmMoeDsaCausalLMModel.
    GGUFArchitectureSpec(
        gguf_arch="glm-dsa",
        model_type="glm_moe_dsa",
        aliases=frozenset({"glm_dsa"}),
        config_key_map="glm_dsa",
        tensor_map=Support.DEFERRED,
        reason=(
            "Config extraction and the glm_moe_dsa graph are both available, but "
            "no GGUF→HuggingFace tensor-name mapping has been written for GLM-5.2's "
            "MLA + DSA-indexer tensor families yet, so weights cannot be routed "
            "into the graph. " + _NO_TENSOR_MAP
        ),
    ),
    # ------------------------------------------------ known but not importable
    GGUFArchitectureSpec(
        gguf_arch="bloom",
        model_type="bloom",
        tensor_map=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=_NO_TENSOR_MAP,
    ),
    GGUFArchitectureSpec(
        gguf_arch="t5",
        model_type="t5",
        tensor_map=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            "T5 is an encoder-decoder with separate enc.*/dec.* tensor namespaces, "
            "relative attention bias, and cross-attention, none of which the "
            "decoder-only GGUF tensor mapping models. " + _NO_TENSOR_MAP
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="arwkv7",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_NO_RWKV_GRAPH,
    ),
    GGUFArchitectureSpec(
        gguf_arch="rwkv6",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_NO_RWKV_GRAPH,
    ),
    GGUFArchitectureSpec(
        gguf_arch="rwkv6qwen2",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_NO_RWKV_GRAPH,
    ),
    GGUFArchitectureSpec(
        gguf_arch="rwkv7",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_NO_RWKV_GRAPH,
    ),
    GGUFArchitectureSpec(
        gguf_arch="nemotron_h_moe",
        config=Support.REJECTED,
        tensor_map=Support.REJECTED,
        graph=Support.REJECTED,
        runtime=Support.REJECTED,
        quantized_import=Support.REJECTED,
        reason=_NEMOTRON_H_MOE_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch=MMPROJ_ARCHITECTURE,
        config=Support.REJECTED,
        tensor_map=Support.REJECTED,
        graph=Support.REJECTED,
        runtime=Support.REJECTED,
        quantized_import=Support.REJECTED,
        reason=_MMPROJ_REASON,
    ),
)


@functools.lru_cache(maxsize=1)
def _index() -> MappingProxyType[str, GGUFArchitectureSpec]:
    """Index every canonical name and alias, rejecting collisions."""
    index: dict[str, GGUFArchitectureSpec] = {}
    for spec in _SPECS:
        for name in spec.names:
            existing = index.get(name)
            if existing is not None:
                raise ValueError(
                    f"GGUF architecture {name!r} is claimed by both "
                    f"{existing.gguf_arch!r} and {spec.gguf_arch!r}"
                )
            index[name] = spec
    return MappingProxyType(index)


def iter_arch_specs() -> tuple[GGUFArchitectureSpec, ...]:
    """Return every registered architecture spec, in declaration order."""
    return _SPECS


def try_get_arch_spec(architecture: str) -> GGUFArchitectureSpec | None:
    """Return the spec for *architecture*, or ``None`` when it is unregistered.

    Accepts canonical names and declared aliases, case-insensitively.
    """
    return _index().get(architecture.lower())


def _unknown_architecture_message(architecture: str) -> str:
    """Build an actionable message for an architecture with no spec."""
    upstream = upstream_architecture(architecture)
    prefix = f"Unsupported GGUF architecture: {architecture!r}."
    if upstream is None:
        return (
            f"{prefix} It is not among the {len(upstream_architectures())} "
            "architectures llama.cpp defines at the pinned commit, so the file is "
            "either newer than this build of mobius or not a llama.cpp GGUF. "
            f"Supported: {', '.join(supported_architectures())}."
        )
    if not upstream.cpp_loader:
        return (
            f"{prefix} It is registered upstream but has no llama.cpp model loader, "
            "so no tool can load it. The file cannot be converted."
        )
    return (
        f"{prefix} It is a real llama.cpp architecture (upstream cohort "
        f"{upstream.cohort}) that mobius does not import yet. Build from the "
        "Hugging Face checkpoint with `mobius build` instead. Supported: "
        f"{', '.join(supported_architectures())}."
    )


def get_arch_spec(architecture: str) -> GGUFArchitectureSpec:
    """Return the importable spec for *architecture*.

    Args:
        architecture: A GGUF ``general.architecture`` value.

    Returns:
        The matching spec, which is guaranteed importable.

    Raises:
        DisabledGGUFArchitectureError: The architecture is known but deliberately
            turned off, because converting it would build a wrong graph.
        UnsupportedGGUFArchitectureError: The architecture is unregistered, or one of
            its capabilities is not ``SUPPORTED``.
    """
    spec = try_get_arch_spec(architecture)
    if spec is None:
        raise UnsupportedGGUFArchitectureError(_unknown_architecture_message(architecture))
    if spec.is_importable:
        return spec

    blocked = sorted(
        name for name, verdict in spec.verdicts.items() if verdict is not Support.SUPPORTED
    )
    detail = (
        f"Unsupported GGUF architecture: {architecture!r}. "
        f"{', '.join(blocked)} unavailable. {spec.reason}"
    )
    if any(verdict is Support.REJECTED for verdict in spec.verdicts.values()):
        raise DisabledGGUFArchitectureError(detail)
    raise UnsupportedGGUFArchitectureError(detail)


@functools.lru_cache(maxsize=1)
def supported_architectures() -> tuple[str, ...]:
    """Return the sorted canonical names of every importable architecture."""
    return tuple(sorted(spec.gguf_arch for spec in _SPECS if spec.is_importable))


def model_type_for(architecture: str) -> str | None:
    """Return the mobius ``model_type`` for *architecture*, if it has one."""
    spec = try_get_arch_spec(architecture)
    return None if spec is None else spec.model_type


def arch_names_with(predicate: Callable[[GGUFArchitectureSpec], bool]) -> frozenset[str]:
    """Return every canonical name and alias whose spec satisfies *predicate*.

    Used to derive the architecture-keyed frozensets that the builder and the
    tensor-mapping module used to declare by hand.
    """
    return frozenset(name for spec in _SPECS if predicate(spec) for name in spec.names)
