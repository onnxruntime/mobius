# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Auto-export pipeline for onnxruntime-genai.

Three entry points, in order of increasing convenience:

- :func:`write_ort_genai_config` — config-only API.  Generates the ORT-GenAI
  config artifacts (``genai_config.json``, tokenizer files,
  ``processor_config.json`` / ``image_processor.json``) for an already-built
  :class:`~mobius._model_package.ModelPackage`.  Does **not** write any ONNX
  files — call :meth:`ModelPackage.save` separately if the ONNX models are
  not already on disk.  The package only needs ``pkg.config`` and the model
  graph metadata; weights need not have been written yet.

- :func:`export_package` — save+config API. Takes an already-built
  ``ModelPackage`` and writes both the ONNX models AND the ORT-GenAI config
  artifacts in one call.  Use this when you built the package manually
  (e.g. with custom dtype / quantization).

- :func:`auto_export` — end-to-end API. Builds the model from a HuggingFace
  ID and calls :func:`export_package`. Use this for the common
  HF-model-id → ORT-GenAI-directory case.

All three produce a directory that ``onnxruntime-genai`` can load directly.

Example::

    # Config-only — assumes you've already saved the ONNX files yourself
    from mobius import build
    from mobius.integrations.ort_genai import write_ort_genai_config

    pkg = build("Qwen/Qwen3-0.6B", load_weights=True)
    pkg.save("/output/qwen3")  # write ONNX + weights first
    write_ort_genai_config(pkg, "/output/qwen3", hf_model_id="Qwen/Qwen3-0.6B")

    # Save + config — single call, when you have a built package in memory
    from mobius.integrations.ort_genai import export_package

    pkg = build("Qwen/Qwen3-0.6B", load_weights=True)
    export_package(pkg, "/output/qwen3", hf_model_id="Qwen/Qwen3-0.6B", ep="cuda")

    # End-to-end — when you only have an HF model id
    from mobius.integrations.ort_genai.auto_export import auto_export

    auto_export("Qwen/Qwen3-0.6B", "/output/qwen3", ep="cuda")
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mobius.upstream_patches import apply_asset_patches

if TYPE_CHECKING:
    import onnx_ir as ir

    from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)


def _revision_kwargs(revision: str | None) -> dict[str, str]:
    """Return an optional HuggingFace revision keyword without passing ``None``."""
    return {"revision": revision} if revision is not None else {}


# ORT-GenAI model type overrides for model types whose ORT-GenAI name
# differs from the HuggingFace model_type.
_ORT_GENAI_MODEL_TYPE: dict[str, str] = {
    "llama": "llama",
    "qwen2": "qwen2",
    "qwen3": "qwen2",
    # Qwen3-MoE shares the dense Qwen3 decoder contract (same inputs, position
    # IDs and KV cache layout); only the MLP differs, and that is fused into
    # the exported graph. ORT GenAI has no "qwen3_moe" entry in its LLM type
    # registry (see onnxruntime-genai/src/models/model_type.h), so passing the
    # HF type through fails to load with "Unsupported model_type in
    # config.json: qwen3_moe". It maps to "qwen3" rather than reusing the dense
    # "qwen3" -> "qwen2" alias: both types dispatch to DecoderOnly_Model, but
    # ORT GenAI's tokenizer tag fallback (tokenizer_tag_utils.cpp) only supplies
    # Qwen3 reasoning-token IDs (bor 151667 / eor 151668) for "qwen3"; under
    # "qwen2" those are absent and tokenizer.bor_token_id/eor_token_id throws.
    "qwen3_moe": "qwen3",
    "phi3": "phi3",
    "phi": "phi",
    "phi4mm": "phi4mm",
    "phi4_multimodal": "phi4mm",
    "gemma": "gemma",
    "gemma2": "gemma",
    "gemma4": "gemma4",
    "gemma4_text": "gemma4_text",
    # gemma-4-12B "unified" (encoder-free) variant reuses the gemma4 ORT GenAI
    # pipelines: the multimodal package (decoder taking inputs_embeds + vision
    # embedder + embedding fusion) maps to "gemma4"; the standalone text
    # backbone maps to "gemma4_text".
    "gemma4_unified": "gemma4",
    "gemma4_unified_text": "gemma4_text",
    "mistral": "mistral",
    "mistral3": "mistral3",
    "lfm2": "lfm2",
    "lfm2_vl": "lfm2_vl",
    # HunYuan-V1 dense / Hy-MT1.5 — generic decoder LLM type accepted by
    # ORT GenAI (see onnxruntime-genai/src/models/model_type.h LLM list).
    "hunyuan_v1_dense": "decoder",
    "deepseek_v4": "decoder",
    # PLaMo2 is a decoder-only hybrid. Released ORT GenAI does not have a
    # model-specific registry entry, so emit its generic decoder type.
    "plamo2": "decoder",
    # Qwen VL model families have separate ORT GenAI model types.
    "qwen2_vl": "qwen2_5_vl",
    "qwen2_vl_text": "qwen2_5_vl",
    "qwen2_5_vl_text": "qwen2_5_vl",
    "qwen3_vl": "qwen3_vl",
    "qwen3_vl_text": "qwen3_vl",
    # Preserve Qwen3.5 / Qwen3.6 source architecture identities here so package
    # topology selection can distinguish standalone text from dense and MoE
    # multimodal parents. Standalone text packages are normalized later to the
    # released generic "decoder" type; multimodal variants retain the matching
    # Qwen-VL type so the runtime constructs the vision+embedding pipeline.
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5_text",
    "qwen3_5_vl": "qwen3_5",
    "qwen3_5_vl_text": "qwen3_5",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen3_5_moe_text": "qwen3_5_moe_text",
    "qwen3_5_moe_vl": "qwen3_5_moe",
    # GLM-OCR uses the Qwen2.5-VL three-model runtime contract: packed image
    # patches, M-RoPE position IDs, an embedding mixer, and a cached decoder.
    "glm_ocr": "qwen2_5_vl",
    "glm_ocr_text": "qwen2_5_vl",
    # MiniCPM uses standard 1D decoder position IDs (unlike Qwen-VL MRoPE).
    # The phi3v multimodal runtime provides that contract; callers supply
    # HF-preprocessed packed pixels through Generator.set_inputs().
    "minicpmv4_6": "phi3v",
}

# These text types select runtime implementations with semantics that are not
# described by the ordinary decoder graph ABI. All other compatible, single-
# model decoder packages use ORT GenAI's released generic DecoderOnly_Model.
_ARCHITECTURE_SPECIFIC_TEXT_TYPES = {
    "lfm2": "lfm2",
    "lfm2_vl": "lfm2",
}
# Composite configs are unwrapped to their text sub-config during config-mode
# builds. Recover the parent runtime type when the exported package still has
# the full multimodal topology.
_UNWRAPPED_VLM_MODEL_TYPES = {
    "gemma3_text": "gemma3",
    # Gemma3n must retain its own pipeline: it binds per-layer inputs that the
    # Gemma3 runtime does not support.
    "gemma3n_text": "gemma3n",
    "qwen3_5_text": "qwen3_5",
    "qwen3_5_vl_text": "qwen3_5",
    "qwen3_5_moe_text": "qwen3_5_moe",
}
_LONGROPE_TEXT_TYPES = frozenset({"phi3", "phi3small", "phimoe"})
_GENERIC_DECODER_MIN_VERSION = (0, 14, 0)
_GENERIC_DECODER_TESTED_VERSIONS = ("0.15.2",)
_DECODER_SEMANTIC_INPUTS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_sequence_length",
        "current_sequence_length",
    }
)
_CACHE_NAME = re.compile(
    r"^(?P<prefix>.+\.)(?P<index>[0-9]+)\.(?P<kind>"
    r"key|value|conv_state|recurrent_state|ssm_state|"
    r"index_key|ple_conv_state|ple_context)$"
)
_QWEN4_EXP_MODEL_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text"})


@dataclass(frozen=True)
class _DecoderAbi:
    inputs: dict[str, str]
    outputs: dict[str, str]
    cache_slots: int
    has_recurrent_state: bool


_GEMMA4_MODEL_TYPES = frozenset(
    {"gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"}
)
# Encoder-free gemma-4-12B "unified" variants. Their image/audio inputs are raw
# merged pixel patches (48px, 6912-dim) / raw waveform frames (640-dim), NOT the
# SigLIP 16px / 128-dim log-mel contract that the ort-extensions
# ``Gemma4ImageTransform`` / ``Gemma4LogMel`` ops implement. There is no
# genai-native transform for the unified contract, so we deliberately do NOT
# emit image_processor.json / audio_processor.json for these models — callers
# must preprocess with the HuggingFace processor and feed tensors via
# ``Generator.set_inputs`` (see examples/gemma4_unified_ort_genai.py).
_GEMMA4_UNIFIED_MODEL_TYPES = frozenset({"gemma4_unified", "gemma4_unified_text"})
_GLMASR_MODEL_TYPES = frozenset({"glmasr"})
_MINICPM_MODEL_TYPES = frozenset({"minicpmv4_6"})
_LFM2_VL_MODEL_TYPES = frozenset({"lfm2_vl"})
# gemma-3 multimodal. build() unwraps the composite HF config to its text
# sub-config, so at export time ``config.model_type`` is "gemma3_text" (not
# "gemma3").
_GEMMA3_MODEL_TYPES = frozenset({"gemma3", "gemma3_text"})
# gemma-3n multimodal. Like gemma-3, build() unwraps the composite config, so
# ``config.model_type`` is "gemma3n_text" at export time. Its MobileNet-V5
# tower takes the same fixed NCHW ``pixel_values`` as gemma-3's SigLIP (at
# 768x768), so it shares that fixed-resize branch — but the checkpoint's
# ``SiglipImageProcessorFast`` sets ``do_normalize=False``, i.e. pixels stay in
# [0, 1] rather than being mapped to [-1, 1].
_GEMMA3N_MODEL_TYPES = frozenset({"gemma3n", "gemma3n_text"})
_PIXTRAL_MODEL_TYPES = frozenset({"mistral3"})
_QWEN_VL_MODEL_TYPES = frozenset(
    {
        "muse_glimmer",
        "muse_glimmer_text",
        "qwen2_vl",
        "qwen2_vl_text",
        "qwen2_5_vl",
        "qwen2_5_vl_text",
        "qwen3_vl",
        "mage_vl",
        "glm_ocr",
        "glm_ocr_text",
        "qwen3_vl_text",
        "qwen3_5",
        "qwen3_5_vl",
        "qwen3_5_vl_text",
        "qwen3_5_moe",
        "qwen3_5_moe_vl",
        "qwen3_5_moe_text",
        "videochat_flash_qwen",
        "qwen4_exp",
        "qwen4_exp_text",
    }
)
_QWEN35_VL_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_vl",
        "qwen3_5_vl_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_5_moe_vl",
    }
)
_QWEN35_TRT_RTX_VISION_PROVIDER_OPTIONS = {
    "nv_profile_min_shapes": "pixel_values:600x1536",
    "nv_profile_opt_shapes": "pixel_values:600x1536",
    "nv_profile_max_shapes": "pixel_values:600x1536",
}

_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer.jsonl",  # PLaMo2 scored vocabulary
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",  # SentencePiece
    "added_tokens.json",
    "merges.txt",  # BPE
    "vocab.json",  # BPE
    "chat_template.jinja",  # Chat template for ORT GenAI
    "tokenization_plamo.py",  # PLaMo2's exact custom tokenizer implementation
    # Preserve HuggingFace processor metadata for VLMs whose preprocessing
    # cannot be represented by an ort-extensions image_processor.json.
    "preprocessor_config.json",
    "processor_config.json",
]


def _resolve_ort_genai_model_type(model_type: str) -> str:
    """Map HuggingFace model_type to ORT-GenAI model type string."""
    return _ORT_GENAI_MODEL_TYPE.get(model_type, model_type)


def _select_ort_model_type(
    config_model_type: str | None,
    hf_model_type: str | None,
    *,
    is_decoder_only: bool,
    rope_type: str | None = None,
) -> str:
    """Choose the ORT-GenAI model type for an exported package.

    Released ORT GenAI dispatches ``decoder`` to its generic
    ``DecoderOnly_Model``. Decoder-only packages therefore use that type unless
    the runtime has genuinely different behavior: ``gpt2`` selects ``Gpt_Model``,
    ``lfm2`` selects ``LFM2_Model``/``LFM2Cache``, and Phi-3 family names are
    retained only for LongRoPE cache recomputation after the short-context
    threshold. Standalone Qwen3.5 text configs also normalize to ``decoder``:
    their specialized names dispatch to the same ``DecoderOnly_Model`` and are
    not available in the latest released ORT GenAI. Mobius GPT-2 graphs also
    normalize to ``decoder`` because they expose the generic separate-key/value
    cache ABI rather than ``Gpt_Model``'s rank-5 combined cache.

    Multimodal and encoder-decoder packages retain their architecture-specific
    type because those values select distinct runtime pipelines and position-ID
    semantics.
    """
    if is_decoder_only:
        for source_type in (config_model_type, hf_model_type):
            resolved = _resolve_ort_genai_model_type(source_type or "unknown")
            if resolved in _ARCHITECTURE_SPECIFIC_TEXT_TYPES:
                return _ARCHITECTURE_SPECIFIC_TEXT_TYPES[resolved]
            if resolved in _LONGROPE_TEXT_TYPES and rope_type == "longrope":
                return resolved
        return "decoder"
    return _resolve_ort_genai_model_type(hf_model_type or "unknown")


def _cache_names(names: list[str]) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {}
    for name in names:
        match = _CACHE_NAME.fullmatch(name)
        if match is None:
            continue
        result.setdefault(match["kind"], {})[int(match["index"])] = name
    return result


def _name_template(names: dict[int, str], *, label: str) -> str:
    templates: set[str] = set()
    for name in names.values():
        match = _CACHE_NAME.fullmatch(name)
        if match is None:
            raise ValueError(f"Invalid {label} cache name {name!r}")
        templates.add(f"{match['prefix']}%d.{match['kind']}")
    if len(templates) != 1:
        raise ValueError(f"ORT GenAI requires one consistent {label} name template")
    return templates.pop()


def _is_single_model_decoder_package(pkg: ModelPackage) -> bool:
    if set(pkg) != {"model"}:
        return False
    model = pkg.get("model")
    if model is None:
        return False
    input_names = {value.name for value in model.graph.inputs}
    output_names = {value.name for value in model.graph.outputs}
    return "input_ids" in input_names and "logits" in output_names


def _inspect_decoder_abi(model: ir.Model, *, model_type: str) -> _DecoderAbi:
    """Validate and describe the released ORT GenAI decoder graph contract."""
    input_names = [value.name for value in model.graph.inputs if value.name is not None]
    output_names = [value.name for value in model.graph.outputs if value.name is not None]
    if "input_ids" not in input_names:
        raise ValueError("Generic ORT GenAI decoder graphs must expose an input_ids input")
    if "logits" not in output_names:
        raise ValueError("Generic ORT GenAI decoder graphs must expose a logits output")
    if model_type == "gpt2":
        raise ValueError(
            "ORT GenAI's gpt2 runtime requires one rank-5 combined KV-cache tensor per "
            "layer, but Mobius GPT-2 graphs expose separate key/value tensors; refusing "
            "to emit an incompatible specialized-runtime config"
        )

    input_cache = _cache_names(input_names)
    output_cache = _cache_names(output_names)
    cache_input_names = {name for values in input_cache.values() for name in values.values()}
    unknown_inputs = set(input_names) - cache_input_names - _DECODER_SEMANTIC_INPUTS
    if unknown_inputs:
        raise ValueError(
            "ORT GenAI cannot automatically supply decoder graph inputs "
            f"{sorted(unknown_inputs)}; use a specialized runtime pipeline"
        )
    has_current_length = "current_sequence_length" in input_names
    has_past_length = "past_sequence_length" in input_names
    if has_current_length != has_past_length:
        raise ValueError(
            "ORT GenAI supplies current_sequence_length and past_sequence_length only "
            "as a pair"
        )
    unsupported_kinds = (set(input_cache) | set(output_cache)) - {
        "key",
        "value",
        "conv_state",
        "recurrent_state",
    }
    if unsupported_kinds:
        raise ValueError(
            "ORT GenAI released config cannot represent decoder state kinds "
            f"{sorted(unsupported_kinds)}; heterogeneous state manifests are deferred to #605"
        )

    key_indices = set(input_cache.get("key", {}))
    value_indices = set(input_cache.get("value", {}))
    present_key_indices = set(output_cache.get("key", {}))
    present_value_indices = set(output_cache.get("value", {}))
    if not key_indices or key_indices != value_indices:
        raise ValueError("ORT GenAI decoder graphs require paired key/value cache inputs")
    if key_indices != present_key_indices or key_indices != present_value_indices:
        raise ValueError(
            "ORT GenAI decoder cache outputs must match the graph's key/value inputs"
        )

    recurrent_indices = set(input_cache.get("conv_state", {}))
    has_recurrent_state = bool(input_cache.get("recurrent_state"))
    if model_type == "lfm2":
        if has_recurrent_state:
            raise ValueError(
                "LFM2's legacy runtime contract does not accept recurrent_state tensors"
            )
        if recurrent_indices != set(output_cache.get("conv_state", {})):
            raise ValueError("LFM2 conv_state outputs must match its conv_state inputs")
    elif recurrent_indices or has_recurrent_state:
        expected = set(input_cache.get("recurrent_state", {}))
        if not recurrent_indices or recurrent_indices != expected:
            raise ValueError(
                "Generic recurrent state requires paired conv_state/recurrent_state inputs"
            )
        if recurrent_indices != set(
            output_cache.get("conv_state", {})
        ) or recurrent_indices != set(output_cache.get("recurrent_state", {})):
            raise ValueError(
                "Generic recurrent state outputs must match conv_state/recurrent_state inputs"
            )

    decoder_inputs = {name: name for name in input_names if name in _DECODER_SEMANTIC_INPUTS}
    decoder_inputs["past_key_names"] = _name_template(input_cache["key"], label="past-key")
    decoder_inputs["past_value_names"] = _name_template(
        input_cache["value"], label="past-value"
    )
    decoder_outputs = {
        "logits": "logits",
        "present_key_names": _name_template(output_cache["key"], label="present-key"),
        "present_value_names": _name_template(output_cache["value"], label="present-value"),
    }
    if model_type == "lfm2" and recurrent_indices:
        decoder_inputs["past_conv_names"] = _name_template(
            input_cache["conv_state"], label="past-convolution"
        )
        decoder_outputs["present_conv_names"] = _name_template(
            output_cache["conv_state"], label="present-convolution"
        )
    elif recurrent_indices:
        expected_input_prefix = decoder_inputs["past_key_names"].rsplit(".", 1)[0]
        expected_output_prefix = decoder_outputs["present_key_names"].rsplit(".", 1)[0]
        recurrent_templates = {
            _name_template(input_cache["conv_state"], label="past-convolution"),
            _name_template(input_cache["recurrent_state"], label="past-recurrent"),
        }
        present_templates = {
            _name_template(output_cache["conv_state"], label="present-convolution"),
            _name_template(output_cache["recurrent_state"], label="present-recurrent"),
        }
        if {template.rsplit(".", 1)[0] for template in recurrent_templates} != {
            expected_input_prefix
        } or {template.rsplit(".", 1)[0] for template in present_templates} != {
            expected_output_prefix
        }:
            raise ValueError(
                "Released ORT GenAI derives recurrent state names from the key-cache "
                "templates; graph prefixes must match exactly"
            )
    all_indices = key_indices | recurrent_indices
    return _DecoderAbi(
        inputs=decoder_inputs,
        outputs=decoder_outputs,
        cache_slots=max(all_indices) + 1,
        has_recurrent_state=has_recurrent_state,
    )


def _runtime_version_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"^([0-9]+)\.([0-9]+)\.([0-9]+)", version)
    if match is None:
        raise ValueError(
            f"Invalid onnxruntime-genai version {version!r}; expected MAJOR.MINOR.PATCH"
        )
    return tuple(int(part) for part in match.groups())


def _write_runtime_compatibility(
    output_dir: str, *, model_type: str, runtime_version: str | None
) -> str:
    minimum_versions = {
        "decoder": _GENERIC_DECODER_MIN_VERSION,
        "lfm2": (0, 15, 2),
    }
    minimum_version = minimum_versions.get(model_type)
    if minimum_version is not None and runtime_version is not None:
        if _runtime_version_tuple(runtime_version) < minimum_version:
            raise ValueError(
                f"ORT GenAI {model_type} packages require onnxruntime-genai >= "
                f"{'.'.join(map(str, minimum_version))}; requested {runtime_version}"
            )
    tested_versions = {
        "decoder": list(_GENERIC_DECODER_TESTED_VERSIONS),
        "lfm2": ["0.15.2"],
    }
    metadata = {
        "runtime": "onnxruntime-genai",
        "model_type": model_type,
        "minimum_version": (
            ".".join(map(str, minimum_version)) if minimum_version is not None else None
        ),
        "tested_versions": tested_versions.get(model_type, []),
        "uses_main_only_state_groups": False,
        "heterogeneous_state_manifest": "deferred: https://github.com/onnxruntime/mobius/issues/605",
    }
    path = os.path.join(output_dir, "runtime_compatibility.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return path


def _load_generation_config(model_id: str):
    """Load optional Hugging Face generation settings without requiring the file."""
    import transformers

    try:
        return transformers.GenerationConfig.from_pretrained(model_id)
    except OSError:
        logger.debug(
            "No generation_config.json found for %s; using model config token IDs", model_id
        )
        return None


def _graph_input_names(model: ir.Model) -> list[str]:
    """Return non-KV-cache input names from an ONNX model graph.

    Filters out indexed cache inputs (``past_key_values.*``), since those
    are represented as template patterns in genai_config.json. Semantic state
    such as ``past_position_ids`` remains explicit and graph-derived.
    """
    return [
        inp.name
        for inp in model.graph.inputs
        if inp.name is not None and not inp.name.startswith("past_key_values.")
    ]


def _count_cache_layer_slots(model: ir.Model | None) -> int | None:
    """Return the number of globally indexed cache slots required by *model*.

    ORT-GenAI binds cache inputs by ``past_key_values.{i}.*`` names, so
    ``num_hidden_layers`` must cover the highest global layer index in the
    graph. Counting only ``.key`` inputs undercounts hybrid models whose other
    layers carry convolution or recurrent state; counting all inputs overcounts
    key/value pairs. The required slot count is therefore ``max(i) + 1``.

    This also preserves the smaller graph-derived count for KV-sharing models
    whose cache-owning layer indices form a shorter contiguous prefix. Returns
    ``None`` when *model* is absent or has no dynamic-cache inputs (e.g. a
    static-cache export using ``key_cache.{i}``).
    """
    if model is None:
        return None
    layer_indices: set[int] = set()
    for inp in model.graph.inputs:
        parts = inp.name.split(".") if inp.name is not None else []
        if len(parts) >= 3 and parts[0] == "past_key_values" and parts[1].isdigit():
            layer_indices.add(int(parts[1]))
    return max(layer_indices) + 1 if layer_indices else None


def _introspect_inputs(pkg: ModelPackage, key: str) -> dict[str, str] | None:
    """Return ``{name: name}`` identity mapping for a sub-model's inputs.

    Returns ``None`` when *key* is absent from *pkg*, letting callers
    fall back to hard-coded defaults.
    """
    model = pkg.get(key)
    if model is None:
        return None
    return {n: n for n in _graph_input_names(model)}


def _introspect_outputs(pkg: ModelPackage, key: str) -> dict[str, str] | None:
    """Return ``{name: name}`` identity mapping for a sub-model's outputs.

    Returns ``None`` when *key* is absent from *pkg*.
    """
    model = pkg.get(key)
    if model is None:
        return None
    return {out.name: out.name for out in model.graph.outputs if out.name is not None}


def _get_static_graph_input_dim(
    pkg: ModelPackage,
    component_name: str,
    input_name: str,
    axis: int,
) -> int:
    component = pkg.get(component_name)
    if component is None:
        raise ValueError(f"Component {component_name!r} is required.")
    value = next(
        (value for value in component.graph.inputs if value.name == input_name),
        None,
    )
    if value is None or value.shape is None:
        raise ValueError(
            f"Component {component_name!r} requires input {input_name!r} with a known rank."
        )
    normalized_axis = axis if axis >= 0 else len(value.shape) + axis
    if not 0 <= normalized_axis < len(value.shape):
        raise ValueError(
            f"Axis {axis} is out of range for {component_name!r} input "
            f"{input_name!r} with rank {len(value.shape)}."
        )
    dimension = value.shape[normalized_axis]
    if not isinstance(dimension, int):
        raise TypeError(
            f"Component {component_name!r} input {input_name!r} axis {axis} must be static."
        )
    return dimension


def _make_trt_rtx_embedding_provider_options(
    *,
    image_feature_width: int,
    input_id_lengths: tuple[int, int, int],
    image_feature_lengths: tuple[int, int, int],
) -> dict[str, str]:
    return {
        f"nv_profile_{profile}_shapes": (
            f"input_ids:1x{input_length},image_features:{feature_length}x{image_feature_width}"
        )
        for profile, input_length, feature_length in zip(
            ("min", "opt", "max"),
            input_id_lengths,
            image_feature_lengths,
            strict=True,
        )
    }


def _copy_tokenizer_files(
    model_id: str,
    output_dir: str,
    *,
    revision: str | None = None,
) -> list[str]:
    """Download and copy tokenizer files from HuggingFace Hub.

    Returns list of copied filenames.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    copied: list[str] = []
    for filename in _TOKENIZER_FILES:
        try:
            src = hf_hub_download(model_id, filename, **_revision_kwargs(revision))
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            copied.append(filename)
        except (EntryNotFoundError, OSError):
            continue
    return copied


def _copy_tokenizer_files_from_local(
    source_dir: str,
    output_dir: str,
) -> list[str]:
    """Copy tokenizer files from a local model directory.

    Silently skips files that are absent (not all tokenizer variants have
    all files — e.g. SentencePiece models have ``tokenizer.model`` but not
    ``merges.txt``).

    Returns list of copied filenames.
    """
    if not os.path.isdir(source_dir):
        logger.warning(
            "Local tokenizer source directory does not exist: %s — no tokenizer files copied.",
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


# Tokenizer class remapping: HF tokenizer classes that ORT GenAI
# (ort-extensions) does not support, mapped to compatible alternatives.
_TOKENIZER_CLASS_REMAP: dict[str, str] = {
    "TokenizersBackend": "LlamaTokenizer",
}


def _fix_tokenizer_config(output_dir: str) -> bool:
    """Remap unsupported tokenizer classes for ORT GenAI compatibility.

    Some HuggingFace models use tokenizer classes (e.g.
    ``TokenizersBackend``) that ORT GenAI's ort-extensions
    tokenizer doesn't support. This fixes tokenizer_config.json
    to use a compatible class.

    Returns True if a fix was applied, False otherwise.
    """
    tc_path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.exists(tc_path):
        return False

    with open(tc_path, encoding="utf-8") as f:
        tc = json.load(f)

    original_class = tc.get("tokenizer_class", "")
    replacement = _TOKENIZER_CLASS_REMAP.get(original_class)
    if replacement is None:
        return False

    tc["tokenizer_class"] = replacement
    with open(tc_path, "w", encoding="utf-8") as f:
        json.dump(tc, f, indent=2, ensure_ascii=False)
    logger.info(
        "Fixed tokenizer_class: %s -> %s",
        original_class,
        replacement,
    )
    return True


_SPECIAL_TOKEN_FIELDS = {
    "<tool_call>": "bot_token_id",
    "</tool_call>": "eot_token_id",
    "<|tool_call|>": "bot_token_id",
    "<|/tool_call|>": "eot_token_id",
    "<think>": "bor_token_id",
    "</think>": "eor_token_id",
}


def _special_token_ids_from_tokenizer_config(
    output_dir: str, vocab_size: int
) -> dict[str, int]:
    """Read delimiter IDs from copied tokenizer_config.json or tokenizer.json."""
    special_token_ids: dict[str, int] = {}
    ambiguous_fields: set[str] = set()
    token_sources = (
        ("tokenizer_config.json", "added_tokens_decoder"),
        ("tokenizer.json", "added_tokens"),
    )
    for filename, added_tokens_key in token_sources:
        path = os.path.join(output_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                added_tokens = json.load(f).get(added_tokens_key, {})
        except (OSError, json.JSONDecodeError, AttributeError):
            logger.warning("Could not read special tokens from %s", path, exc_info=True)
            continue
        if isinstance(added_tokens, dict):
            entries = added_tokens.items()
        elif isinstance(added_tokens, list):
            entries = (
                (token.get("id"), token) for token in added_tokens if isinstance(token, dict)
            )
        else:
            continue
        for raw_token_id, token in entries:
            if not isinstance(token, dict):
                continue
            field = _SPECIAL_TOKEN_FIELDS.get(token.get("content"))
            try:
                token_id = int(raw_token_id)
            except (TypeError, ValueError):
                continue
            if field is None or not 0 <= token_id < vocab_size or field in ambiguous_fields:
                continue
            if field in special_token_ids and special_token_ids[field] != token_id:
                special_token_ids.pop(field)
                ambiguous_fields.add(field)
            else:
                special_token_ids[field] = token_id
    return special_token_ids


def _fix_chat_template(
    output_dir: str,
    hf_model_id: str | None,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> bool:
    """Ensure chat_template is present in tokenizer_config.json.

    Some HuggingFace models don't store ``chat_template`` in the
    raw ``tokenizer_config.json`` file — transformers injects it
    dynamically from the model class at runtime. ORT GenAI reads
    the file directly and needs it to be present.

    This function loads the tokenizer via transformers (which
    applies the dynamic template) and writes it back.

    Returns True if the template was added, False otherwise.
    """
    tc_path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.exists(tc_path):
        return False

    with open(tc_path, encoding="utf-8") as f:
        tc = json.load(f)

    if tc.get("chat_template"):
        return False

    if hf_model_id is None:
        return False

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id,
            trust_remote_code=trust_remote_code,
            **_revision_kwargs(revision),
        )
        template = getattr(tokenizer, "chat_template", None)
        if template:
            tc["chat_template"] = template
            with open(tc_path, "w", encoding="utf-8") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)
            logger.info(
                "Added chat_template to tokenizer_config.json from %s",
                hf_model_id,
            )
            return True
    except Exception:
        logger.warning(
            "Could not load tokenizer for %s to extract chat_template",
            hf_model_id,
            exc_info=True,
        )
    return False


# PIL resample constant -> ort-extensions Resize "interpolation" name.  HF image
# processors store ``resample`` as the PIL integer; ort-extensions names the same
# filters differently and defaults to CUBIC, so an unmapped value silently
# resamples with the wrong kernel.  PIL BOX/HAMMING (4/5) have no counterpart.
_PIL_RESAMPLE_TO_INTERPOLATION = {0: "NEAREST", 1: "LANCZOS", 2: "LINEAR", 3: "CUBIC"}


def _size_mapping(size: Any) -> dict[str, Any]:
    """Normalise an HF ``image_processor.size`` to a plain dict.

    transformers >= 5 hands back a ``SizeDict``, which is *not* a ``dict``
    subclass, so an ``isinstance(size, dict)`` guard silently discards it and
    falls through to whatever default the caller hardcoded.
    """
    if isinstance(size, dict):
        return size
    if hasattr(size, "get"):  # SizeDict and friends
        return {
            k: getattr(size, k, None)
            for k in ("height", "width", "longest_edge", "shortest_edge")
        }
    return {}


def _resize_interpolation(resample: Any) -> str | None:
    """Map an HF ``image_processor.resample`` to an ort-extensions filter name."""
    if resample is None:
        return None
    try:
        name = _PIL_RESAMPLE_TO_INTERPOLATION.get(int(resample))
    except (TypeError, ValueError):
        return None
    if name is None:
        logger.warning(
            "Unsupported image resample %s; ort-extensions has no matching "
            "filter and will fall back to its CUBIC default",
            resample,
        )
    return name


def _build_vision_transform_pipeline(
    *,
    image_size: int,
    patch_size: int,
    merge_size: int,
    rescale_factor: float,
    image_mean: list[float],
    image_std: list[float],
    min_pixels: int = 784,
    max_pixels: int = 2371600,
    resample: Any = None,
) -> list[dict[str, Any]]:
    """Build the common 4-step vision transform pipeline.

    Returns the base transforms: DecodeImage → Resize → Rescale → Normalize.
    Callers may append model-specific steps (e.g. Permute3D, PixtralImageSizes)
    after this.

    No ``ConvertRGB``: ort-extensions' ``convert_to_rgb`` *unconditionally*
    swaps R and B (it exists to fix up a BGR decode), while ``DecodeImage`` with
    ``color_space="RGB"`` already emits RGB.  Chaining the two hands the encoder
    BGR — see :func:`_write_vision_processor_config`.
    """
    resize_attrs: dict[str, Any] = {
        "height": image_size,
        "width": image_size,
        "smart_resize": 1,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "patch_size": patch_size,
        "merge_size": merge_size,
    }
    interpolation = _resize_interpolation(resample)
    if interpolation is not None:
        resize_attrs["interpolation"] = interpolation

    return [
        {
            "operation": {
                "name": "decode_image",
                "type": "DecodeImage",
                "attrs": {"color_space": "RGB"},
            }
        },
        {
            "operation": {
                "name": "resize",
                "type": "Resize",
                "attrs": resize_attrs,
            }
        },
        {
            "operation": {
                "name": "rescale",
                "type": "Rescale",
                "attrs": {
                    "rescale_factor": rescale_factor,
                },
            }
        },
        {
            "operation": {
                "name": "normalize",
                "type": "Normalize",
                "attrs": {
                    "mean": image_mean,
                    "std": image_std,
                },
            }
        },
    ]


def _write_vision_processor_config(
    config: Any,
    output_dir: str,
    *,
    hf_model_id: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> str | None:
    """Write the vision processor config file for VLM models.

    Generates the ORT-extensions image transform pipeline derived from the
    HuggingFace image processor config. When ``hf_model_id`` is provided,
    loads the HF processor to extract normalization values and resize
    parameters. Otherwise falls back to CLIP-standard defaults.

    The output format depends on the model type:

    - **Gemma4** (``gemma4``, ``gemma4_text``): Writes ``image_processor.json``
      with a ``DecodeImage → Gemma4ImageTransform`` pipeline.
    - **Gemma4 unified** (``gemma4_unified*``): Returns ``None`` — the
      encoder-free model has no matching ort-extensions transform; callers feed
      HF-preprocessed pixel_values via ``Generator.set_inputs``.
    - **Gemma3** (``gemma3`` or ``gemma3_text``): Writes
      ``processor_config.json`` with a 5-step pipeline (DecodeImage →
      Resize[fixed] → Rescale → Normalize → Permute3D). Uses a
      fixed-size resize (no ``smart_resize``) so the SigLIP encoder's fixed
      NCHW ``pixel_values`` input contract is met.
    - **Gemma3n** (``gemma3n`` or ``gemma3n_text``): Same fixed-resize pipeline
      at 768x768 for the MobileNet-V5 tower, but with the ``Normalize`` step
      *omitted* — the checkpoint's processor sets ``do_normalize=False``, so
      pixels stay in [0, 1] (4 steps).
    - **Pixtral / Mistral3**: Writes ``processor_config.json`` with a 6-step
      pipeline (DecodeImage → Resize → Rescale → Normalize →
      Permute3D → PixtralImageSizes).
    - **Mage-VL**: Writes ``image_processor.json`` with Qwen-style smart resize,
      CLIP normalization, and packed patch extraction.
    - **Other VLMs**: Writes ``processor_config.json`` with a 4-step pipeline
      (DecodeImage → Resize → Rescale → Normalize).

    No pipeline emits ``ConvertRGB``.  ort-extensions' ``convert_to_rgb``
    unconditionally swaps R and B — it is the fix-up for a BGR decode — so
    pairing it with ``DecodeImage(color_space="RGB")`` fed every VLM BGR
    pixels.  The ``Resize`` step carries an explicit ``interpolation`` derived
    from the HF processor's ``resample``, since ort-extensions otherwise
    defaults to CUBIC where HF models overwhelmingly use BILINEAR.

    Returns the written file path, or None if the config has no vision section.
    """
    vision = getattr(config, "vision", None)
    if vision is None:
        return None

    model_type = getattr(config, "model_type", "")
    if model_type in _GEMMA4_UNIFIED_MODEL_TYPES:
        # Encoder-free unified model: no ort-extensions transform matches its
        # raw merged-patch contract. Emit no image_processor.json; callers feed
        # HF-preprocessed pixel_values via Generator.set_inputs.
        logger.info(
            "Skipping image_processor.json for encoder-free %s "
            "(no native ort-extensions transform; use HF processor + set_inputs)",
            model_type,
        )
        return None
    if model_type in _MINICPM_MODEL_TYPES:
        # MiniCPM needs adaptive slicing and NaViT horizontal patch packing.
        # ort-extensions has no equivalent transform, so preserving the HF
        # processor output and injecting it through set_inputs is the only
        # numerically faithful runtime path.
        logger.info(
            "Skipping image_processor.json for %s "
            "(use MiniCPMV4_6Processor + Generator.set_inputs)",
            model_type,
        )
        return None
    if model_type in _LFM2_VL_MODEL_TYPES:
        # LFM2-VL uses adaptive tiling, thumbnail insertion, NaFlex patchification,
        # and prompt-token expansion. No ort-extensions transform implements that
        # contract; preserve the pinned HF processor_config.json copied above and
        # require callers to feed its three tensors through set_inputs().
        logger.info(
            "Skipping generated image processor for %s "
            "(use Lfm2VlProcessor + Generator.set_inputs)",
            model_type,
        )
        return None

    vision_model_type = getattr(vision, "model_type", None)
    is_pixtral = vision_model_type == "pixtral" or model_type in _PIXTRAL_MODEL_TYPES

    if model_type in _GEMMA4_MODEL_TYPES:
        # Gemma4 needs an onnxruntime-extensions format processor config
        # with a transforms pipeline (DecodeImage -> Gemma4ImageTransform).
        max_soft_tokens = (
            getattr(vision, "mm_tokens_per_image", None)
            or getattr(config, "mm_tokens_per_image", None)
            or 280
        )
        patch_size = getattr(vision, "patch_size", None) or 16
        pooling_kernel_size = getattr(vision, "pooling_kernel_size", None) or 3
        processor_config: dict[str, Any] = {
            "processor": {
                "name": "gemma_4_image_processing",
                "transforms": [
                    {
                        "operation": {
                            "name": "decode_image",
                            "type": "DecodeImage",
                            "attrs": {"color_space": "RGB"},
                        }
                    },
                    {
                        "operation": {
                            "name": "gemma4_image_transform",
                            "type": "Gemma4ImageTransform",
                            "attrs": {
                                "patch_size": patch_size,
                                "max_soft_tokens": max_soft_tokens,
                                "pooling_kernel_size": pooling_kernel_size,
                            },
                        }
                    },
                ],
            }
        }
        path = os.path.join(output_dir, "image_processor.json")
    elif model_type in _GEMMA3_MODEL_TYPES or model_type in _GEMMA3N_MODEL_TYPES:
        # Gemma3's SigLIP encoder and Gemma3n's MobileNet-V5 tower both take a
        # plain NCHW image tensor ([batch, 3, image_size, image_size]). The
        # generic-VLM branch below emits smart_resize (variable HxW) and no
        # Permute3D, leaving a variable-size HWC tensor that fails either
        # encoder's fixed input. Emit a fixed-size resize (no smart_resize) +
        # trailing Permute3D.
        is_gemma3n = model_type in _GEMMA3N_MODEL_TYPES
        family = "gemma3n" if is_gemma3n else "gemma3"
        image_size = getattr(vision, "image_size", None) or (768 if is_gemma3n else 896)
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
        rescale_factor = 1.0 / 255.0
        # Gemma3n's SiglipImageProcessorFast sets do_normalize=False: the tower
        # is trained on [0, 1] pixels, so a mean/std-0.5 Normalize would shift
        # them to [-1, 1] and silently degrade every caption.
        do_normalize = not is_gemma3n
        # Both families' processors resample with PIL BILINEAR; ort-extensions
        # would otherwise apply its CUBIC default.
        resample: Any = 2
        if hf_model_id is not None:
            try:
                from transformers import AutoProcessor

                hf_proc = AutoProcessor.from_pretrained(
                    hf_model_id,
                    trust_remote_code=trust_remote_code,
                    **_revision_kwargs(revision),
                )
                ip = getattr(hf_proc, "image_processor", None)
                if ip is not None:
                    image_mean = list(getattr(ip, "image_mean", image_mean))
                    image_std = list(getattr(ip, "image_std", image_std))
                    rescale_factor = getattr(ip, "rescale_factor", rescale_factor)
                    hf_do_normalize = getattr(ip, "do_normalize", None)
                    if hf_do_normalize is not None:
                        do_normalize = bool(hf_do_normalize)
                    resample = getattr(ip, "resample", resample)
                    size = _size_mapping(getattr(ip, "size", None))
                    if size:
                        image_size = (
                            size.get("height") or size.get("longest_edge") or image_size
                        )
            except Exception:
                logger.warning(
                    "Could not load HF processor for %s; using %s defaults "
                    "(image_size=%s, mean/std=0.5, do_normalize=%s)",
                    hf_model_id,
                    family,
                    image_size,
                    do_normalize,
                    exc_info=True,
                )
        resize_attrs: dict[str, Any] = {
            "height": image_size,
            "width": image_size,
            "smart_resize": 0,
        }
        interpolation = _resize_interpolation(resample)
        if interpolation is not None:
            resize_attrs["interpolation"] = interpolation
        transforms = [
            {
                "operation": {
                    "name": "decode_image",
                    "type": "DecodeImage",
                    "attrs": {"color_space": "RGB"},
                }
            },
            {
                "operation": {
                    "name": "resize",
                    "type": "Resize",
                    "attrs": resize_attrs,
                }
            },
            {
                "operation": {
                    "name": "rescale",
                    "type": "Rescale",
                    "attrs": {"rescale_factor": rescale_factor},
                }
            },
        ]
        if do_normalize:
            transforms.append(
                {
                    "operation": {
                        "name": "normalize",
                        "type": "Normalize",
                        "attrs": {"mean": image_mean, "std": image_std},
                    }
                }
            )
        transforms.append(
            {
                "operation": {
                    "name": "permute",
                    "type": "Permute3D",
                    "attrs": {"dims": [2, 0, 1]},
                }
            }
        )
        processor_config = {"processor": {"name": "image_processor", "transforms": transforms}}
        path = os.path.join(output_dir, "processor_config.json")
    else:
        # Pixtral and generic VLMs share the same base pipeline;
        # Pixtral adds Permute3D + PixtralImageSizes at the end.
        patch_size = getattr(vision, "patch_size", 14) or 14
        merge_size = (
            getattr(vision, "spatial_merge_size", None)
            or getattr(config, "spatial_merge_size", 2)
            or 2
        )

        is_qwen4_exp = model_type in _QWEN4_EXP_MODEL_TYPES
        # Qwen4-Exp is pinned to the checkpoint's Qwen processor constants.
        # Other generic VLMs retain the CLIP-standard fallback.
        image_mean = [0.5, 0.5, 0.5] if is_qwen4_exp else [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.5, 0.5, 0.5] if is_qwen4_exp else [0.26862954, 0.26130258, 0.27577711]
        rescale_factor = 1.0 / 255.0
        min_pixels = 65_536 if is_qwen4_exp else 784
        max_pixels = 16_777_216 if is_qwen4_exp else 2_371_600
        image_size = getattr(vision, "image_size", None)
        resample = 3 if is_qwen4_exp else None

        if hf_model_id is not None:
            try:
                from transformers import AutoProcessor

                hf_proc = AutoProcessor.from_pretrained(
                    hf_model_id,
                    trust_remote_code=trust_remote_code,
                    **_revision_kwargs(revision),
                )
                ip = getattr(hf_proc, "image_processor", None)
                if ip is not None:
                    image_mean = list(getattr(ip, "image_mean", image_mean))
                    image_std = list(getattr(ip, "image_std", image_std))
                    rescale_factor = getattr(ip, "rescale_factor", rescale_factor)
                    resample = getattr(ip, "resample", resample)
                    if hasattr(ip, "size"):
                        size = ip.size
                        if isinstance(size, int):
                            image_size = size
                        else:
                            size = _size_mapping(size)
                            # Qwen-style processors encode shortest_edge and
                            # longest_edge as pixel-count bounds, not side lengths.
                            # Keep the vision config's nominal image size for the
                            # Resize metadata and preserve those values below as
                            # smart-resize bounds.
                            if size.get("height") is not None:
                                image_size = size["height"]
                            elif size.get("width") is not None:
                                image_size = size["width"]
                            min_pixels = size.get("shortest_edge") or min_pixels
                            max_pixels = size.get("longest_edge") or max_pixels
            except Exception:
                if is_qwen4_exp:
                    image_mean = [0.5, 0.5, 0.5]
                    image_std = [0.5, 0.5, 0.5]
                    rescale_factor = 1.0 / 255.0
                    resample = 3
                    min_pixels = 65_536
                    max_pixels = 16_777_216
                logger.warning(
                    "Could not load HF processor for %s; using %s",
                    hf_model_id,
                    (
                        "pinned Qwen4-Exp processor constants"
                        if is_qwen4_exp
                        else "CLIP-standard normalization defaults"
                    ),
                    exc_info=True,
                )

        if image_size is None:
            image_size = 1540 if is_pixtral else 448

        transforms = _build_vision_transform_pipeline(
            image_size=image_size,
            patch_size=patch_size,
            merge_size=merge_size,
            rescale_factor=rescale_factor,
            image_mean=image_mean,
            image_std=image_std,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            resample=resample,
        )

        if is_pixtral:
            # Pixtral requires Permute3D (HWC→CHW) and PixtralImageSizes
            # for the per-image slicing loop in PixtralVisionState.
            transforms.append(
                {
                    "operation": {
                        "name": "permute",
                        "type": "Permute3D",
                        "attrs": {"dims": [2, 0, 1]},
                    }
                }
            )
            transforms.append(
                {
                    "operation": {
                        "name": "pixtral_image_sizes",
                        "type": "PixtralImageSizes",
                    }
                }
            )
        elif model_type in _QWEN_VL_MODEL_TYPES:
            # Qwen-VL models need the PatchImage transform to extract
            # temporal+spatial patches, and qwen2_5_vl/qwen3_vl flag
            # on Normalize for correct interleaving.
            temporal_patch_size = config.temporal_patch_size
            # Add qwen3_vl flag to the Normalize step
            for t in transforms:
                op = t.get("operation", {})
                if op.get("type") == "Normalize":
                    op.setdefault("attrs", {})["qwen2_5_vl"] = 1
            transforms.append(
                {
                    "operation": {
                        "name": "patch_image",
                        "type": "PatchImage",
                        "attrs": {
                            "patch_size": patch_size,
                            "temporal_patch_size": temporal_patch_size,
                            "merge_size": merge_size,
                        },
                    }
                }
            )

        processor_name = (
            "pixtral_image_processor"
            if is_pixtral
            else "qwen2_5_image_processor"
            if model_type in _QWEN_VL_MODEL_TYPES
            else "image_processor"
        )
        processor_config = {
            "processor": {
                "name": processor_name,
                "transforms": transforms,
            }
        }
        path = os.path.join(
            output_dir,
            "image_processor.json" if model_type == "mage_vl" else "processor_config.json",
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(processor_config, f, indent=4)
    return path


def _write_audio_processor_config(
    config: Any,
    output_dir: str,
) -> str | None:
    """Write audio_processor.json for models with audio encoders.

    Returns the path if written, None otherwise.
    """
    audio = getattr(config, "audio", None)
    if audio is None:
        return None

    model_type = getattr(config, "model_type", "")

    if model_type in _GEMMA4_UNIFIED_MODEL_TYPES:
        # Encoder-free unified model: raw 640-dim waveform frames, not the
        # 128-dim log-mel Gemma4LogMel contract. Emit no audio_processor.json;
        # callers feed HF-preprocessed input_features via Generator.set_inputs.
        logger.info(
            "Skipping audio_processor.json for encoder-free %s "
            "(no native ort-extensions transform; use HF processor + set_inputs)",
            model_type,
        )
        return None

    if model_type in _GEMMA4_MODEL_TYPES:
        # Gemma4 USM-style 128-dim log-mel spectrogram.
        # OrtxCreateSpeechFeatureExtractor requires the feature_extraction.sequence format.
        processor = {
            "feature_extraction": {
                "sequence": [
                    {
                        "operation": {
                            "name": "audio_decoder",
                            "type": "AudioDecoder",
                        }
                    },
                    {
                        "operation": {
                            "name": "gemma4_log_mel",
                            "type": "Gemma4LogMel",
                            "attrs": {
                                "feature_size": 128,
                                "sampling_rate": 16000,
                                "frame_length_ms": 20.0,
                                "hop_length_ms": 10.0,
                                "min_frequency": 0.0,
                                "max_frequency": 8000.0,
                                "preemphasis": 0.0,
                                "preemphasis_htk_flavor": 1,
                                "fft_overdrive": 0,
                                "mel_floor": 0.001,
                            },
                        }
                    },
                ]
            }
        }
        proc_filename = "audio_feature_extraction.json"
    elif model_type in _GEMMA3N_MODEL_TYPES:
        # Gemma3n reuses gemma4's Gemma4LogMel op, but NOT its attribute values:
        # Gemma3nAudioFeatureExtractor is a different filterbank. Values below are
        # from the E4B preprocessor_config.json; the ones that differ from the
        # gemma4 branch above are marked.
        #
        # frame_length/hop_length are given in samples upstream (512/160 @ 16 kHz)
        # and converted to the milliseconds this op takes: 512/16000 = 32 ms,
        # 160/16000 = 10 ms. fft_length 1024 is implied by fft_overdrive (the next
        # power of two above frame_length, doubled) and has no separate attribute.
        processor = {
            "feature_extraction": {
                "sequence": [
                    {
                        "operation": {
                            "name": "audio_decoder",
                            "type": "AudioDecoder",
                        }
                    },
                    {
                        "operation": {
                            "name": "gemma3n_log_mel",
                            "type": "Gemma4LogMel",
                            "attrs": {
                                "feature_size": 128,
                                "sampling_rate": 16000,
                                "frame_length_ms": 32.0,  # differs (gemma4: 20.0)
                                "hop_length_ms": 10.0,
                                "min_frequency": 125.0,  # differs (gemma4: 0.0)
                                "max_frequency": 7600.0,  # differs (gemma4: 8000.0)
                                "preemphasis": 0.97,  # differs (gemma4: 0.0)
                                "preemphasis_htk_flavor": 1,
                                "fft_overdrive": 1,  # differs (gemma4: 0)
                                "mel_floor": 1e-05,  # differs (gemma4: 0.001)
                            },
                        }
                    },
                ]
            }
        }
        proc_filename = "audio_feature_extraction.json"
    elif model_type in _GLMASR_MODEL_TYPES:
        # GLM-ASR uses the standard Whisper log-mel contract with 128 mel
        # bins and a fixed 30-second window. These operation names and attrs
        # are consumed by OrtxCreateSpeechFeatureExtractor.
        processor = {
            "feature_extraction": {
                "sequence": [
                    {
                        "operation": {
                            "name": "audio_decoder",
                            "type": "AudioDecoder",
                        }
                    },
                    {
                        "operation": {
                            "name": "stft",
                            "type": "STFTNorm",
                            "attrs": {
                                "n_fft": 400,
                                "frame_length": 400,
                                "hop_length": 160,
                            },
                        }
                    },
                    {
                        "operation": {
                            "name": "log_mel",
                            "type": "LogMelSpectrum",
                            "attrs": {
                                "chunk_size": 30,
                                "hop_length": 160,
                                "n_fft": 400,
                                "n_mel": 128,
                            },
                        }
                    },
                ]
            }
        }
        proc_filename = "audio_processor.json"
    else:
        # Generic audio processor — add model-specific branches as needed.
        return None

    path = os.path.join(output_dir, proc_filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(processor, f, indent=4)
    return path


def _uses_compact_sliding_kv_cache(decoder_model: ir.Model | None, ep: str) -> bool:
    """Whether the graph and EP can keep sliding layers in a compact KV cache."""
    if ep == "trt-rtx":
        return True
    if decoder_model is None or ep not in {"cpu", "cuda"}:
        return False
    return any(
        node.op_type == "GroupQueryAttention"
        and node.domain == "com.microsoft"
        and (attribute := node.attributes.get("sliding_window_cache")) is not None
        and attribute.as_int() == 1
        for node in decoder_model.graph
    )


def _write_genai_config(
    config: Any,
    output_dir: str,
    *,
    pkg: ModelPackage,
    ort_model_type: str,
    ep: str,
    context_length: int,
    bos_token_id: int | None,
    eos_token_id: int | list[int] | None,
    pad_token_id: int | None,
    is_vlm: bool,
    has_speech: bool,
) -> str:
    """Generate and write genai_config.json.

    Input names for each sub-model (decoder, vision, embedding) are
    introspected from the ONNX graphs in *pkg* rather than hard-coded
    per model type.

    Returns the path to the written file.
    """
    if getattr(config, "model_type", None) in _QWEN4_EXP_MODEL_TYPES:
        raise ValueError(
            "onnxruntime-genai cannot represent Qwen4-Exp's heterogeneous "
            "state contract; use ModelPackage.save() and mobius.state_manifest"
        )
    from mobius.integrations.ort_genai.genai_config import GenaiConfigGenerator

    # --- Discover decoder inputs from the ONNX graph ---
    decoder_key = "decoder" if "decoder" in pkg else "model"
    decoder_model = pkg.get(decoder_key)
    decoder_abi: _DecoderAbi | None = None
    if _is_single_model_decoder_package(pkg):
        if decoder_model is None:
            raise ValueError("ORT GenAI text packages require a decoder ONNX graph")
        decoder_abi = _inspect_decoder_abi(decoder_model, model_type=ort_model_type)
        decoder_inputs = decoder_abi.inputs
        decoder_outputs = decoder_abi.outputs
    else:
        decoder_inputs = _introspect_inputs(pkg, decoder_key)
        decoder_outputs = None
        if decoder_inputs is not None:
            # Multimodal runtime types retain their architecture-specific cache contract.
            decoder_inputs["past_key_names"] = "past_key_values.%d.key"
            decoder_inputs["past_value_names"] = "past_key_values.%d.value"

    # Derive decoder filename from the actual package key
    decoder_filename = (
        f"{decoder_key}/model.onnx" if len(pkg) > 1 or decoder_key != "model" else "model.onnx"
    )

    # ORT GenAI's ``past_present_share_buffer`` mode requires the decoder
    # graph to write the KV cache in place. Only ``com.microsoft.
    # GroupQueryAttention`` does that; the standard ONNX ``Attention`` op
    # concatenates ``past_key`` with the new ``K`` and returns a dynamic-
    # shape ``present_key``, which is incompatible with the pre-allocated
    # shared buffer. Introspect the graph: if there is at least one GQA
    # node, the model supports shared-buffer mode; otherwise force it off
    # regardless of the EP capability flag.
    #
    # ``com.microsoft.LinearAttention`` (linear/recurrent-attention layers,
    # e.g. Qwen3.5's GatedDeltaNet) is a separate, *mandatory* case: its
    # recurrent state requires ``past_present_share_buffer=True`` regardless
    # of whether any other layer uses GQA (ORT GenAI raises "RecurrentState
    # requires past_present_share_buffer=true" otherwise).
    #
    # Hybrid models mix LinearAttention layers with full-attention layers,
    # which may lower to GQA *or* to the standard (non-GQA) ``Attention`` op
    # depending on EP/dtype (e.g. the CPU EP only lowers to GQA for fp32;
    # fp16 falls back to standard Attention -- see ``_execution_providers.py``
    # ``gqa_dtypes``). If a hybrid graph has LinearAttention but its
    # full-attention layers are still standard (non-GQA) Attention, forcing
    # ``past_present_share_buffer=True`` produces an unrunnable config: the
    # recurrent state requires it, but standard Attention's dynamic-shape KV
    # concat cannot honor a pre-allocated shared buffer, which fails at
    # generation time with an ``attn_mask``/``total_sequence_length``
    # mismatch rather than at load time. Rather than silently emit a broken
    # config, raise a clear error so the caller picks an EP/dtype combination
    # (e.g. fp32 on CPU) that lowers full attention to GQA.
    supports_in_place_kv_cache: bool | None = None
    if decoder_model is not None:
        has_gqa = any(
            node.op_type == "GroupQueryAttention" and node.domain == "com.microsoft"
            for node in decoder_model.graph
        )
        has_recurrent_state = any(
            node.op_type == "LinearAttention" and node.domain == "com.microsoft"
            for node in decoder_model.graph
        )
        has_standard_attention = any(
            node.op_type == "Attention" and node.domain in ("", "ai.onnx")
            for node in decoder_model.graph
        )
        if has_recurrent_state and has_standard_attention:
            # A GQA node elsewhere in the graph does NOT make a co-existing
            # standard Attention node compatible with a shared buffer --
            # each op instance is independently (in)compatible, so this
            # must reject on the mere presence of standard Attention, not
            # only when GQA is completely absent (partial GQA fusion still
            # leaves the unfused standard Attention layers broken).
            raise ValueError(
                "This decoder graph mixes com.microsoft.LinearAttention "
                "(recurrent state, requires past_present_share_buffer=True) "
                "with standard (non-GQA) Attention (incompatible with "
                "past_present_share_buffer=True). This EP/dtype combination "
                "cannot produce a runnable genai_config -- pick an EP/dtype "
                "that lowers *all* full-attention layers to "
                "GroupQueryAttention instead (e.g. fp32 on the CPU EP)."
            )
        else:
            supports_in_place_kv_cache = has_gqa or has_recurrent_state

    sliding_window = None
    window_size = getattr(config, "sliding_window", None)
    # ORT GenAI uses this block to allocate a compact present cache. A
    # local_window_size mask alone does not compact GQA outputs.
    if (
        isinstance(window_size, int)
        and window_size > 0
        and _uses_compact_sliding_kv_cache(decoder_model, ep)
    ):
        layer_types = getattr(config, "layer_types", None)
        local_types = {"local", "sliding_attention", "window_attention"}
        layers = (
            [
                index
                for index, layer_type in enumerate(layer_types)
                if layer_type in local_types
            ]
            if layer_types
            else list(range(config.num_hidden_layers))
        )
        sliding_window = {
            "window_size": window_size,
            "slide_key_value_cache": False,
            "slide_inputs": False,
            "layers": layers,
        }

    generator = GenaiConfigGenerator.from_config(
        config,
        ort_model_type,
        context_length=context_length,
        ep=ep,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        decoder_inputs=decoder_inputs,
        decoder_outputs=decoder_outputs,
        decoder_filename=decoder_filename,
        supports_in_place_kv_cache=supports_in_place_kv_cache,
        num_cache_layer_slots=(
            decoder_abi.cache_slots
            if decoder_abi is not None
            else _count_cache_layer_slots(decoder_model)
        ),
        sliding_window=sliding_window,
        has_specialized_topology=not _is_single_model_decoder_package(pkg),
    )
    generator.with_special_tokens(
        **_special_token_ids_from_tokenizer_config(output_dir, config.vocab_size)
    )

    if is_vlm:
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is not None:
            model_type = getattr(config, "model_type", "")
            vision_input_mapping = _introspect_inputs(pkg, "vision_encoder")
            embedding_input_mapping = _introspect_inputs(pkg, "embedding")
            if model_type in _MINICPM_MODEL_TYPES and vision_input_mapping is not None:
                # ORT GenAI's VisionInputs schema only accepts its predefined
                # semantic keys. ``target_sizes`` remains an ONNX graph input
                # and is supplied as a named tensor through set_inputs().
                vision_input_mapping.pop("target_sizes", None)

            # spatial_merge_size and config_filename are config-level
            # properties that cannot be inferred from the graph.
            vision_kwargs: dict[str, Any] = {}
            if model_type in _GEMMA4_MODEL_TYPES:
                vision_cfg = getattr(config, "vision", None)
                vision_kwargs["spatial_merge_size"] = getattr(
                    vision_cfg, "spatial_merge_size", 2
                )
            elif model_type in _MINICPM_MODEL_TYPES:
                # MiniCPM performs both 2x2 merges inside the ONNX vision
                # graph and consumes HF-prepacked pixels, not Qwen grid_thw.
                vision_kwargs["spatial_merge_size"] = None
            elif model_type in _LFM2_VL_MODEL_TYPES:
                # Pixel unshuffle is already part of the ONNX vision encoder;
                # ORT GenAI must not perform another spatial merge.
                vision_kwargs["spatial_merge_size"] = None
                vision_kwargs["config_filename"] = "processor_config.json"
            elif has_speech:
                vision_kwargs["spatial_merge_size"] = None
                # Gemma3n shares gemma3's fixed-resize branch in
                # _write_vision_processor_config, which writes
                # processor_config.json. Without this, with_vision's
                # "image_processor.json" default would be referenced instead —
                # a filename only the gemma4 branch above ever writes.
                if model_type in _GEMMA3N_MODEL_TYPES:
                    vision_kwargs["config_filename"] = "processor_config.json"
            elif (
                model_type in _PIXTRAL_MODEL_TYPES
                or getattr(getattr(config, "vision", None), "model_type", None) == "pixtral"
            ):
                vision_cfg = getattr(config, "vision", None)
                sms = getattr(vision_cfg, "spatial_merge_size", None) or getattr(
                    config, "spatial_merge_size", 2
                )
                vision_kwargs["spatial_merge_size"] = sms
                vision_kwargs["config_filename"] = "processor_config.json"
            else:
                # All other VLMs (Qwen-VL, LLaVA, InternVL, etc.) use
                # processor_config.json written by _write_vision_processor_config.
                vision_cfg = getattr(config, "vision", None)
                sms = getattr(vision_cfg, "spatial_merge_size", None) or getattr(
                    config, "spatial_merge_size", None
                )
                if sms is not None:
                    vision_kwargs["spatial_merge_size"] = sms
                vision_kwargs["config_filename"] = (
                    "image_processor.json"
                    if model_type == "mage_vl"
                    else "processor_config.json"
                )
                if (
                    model_type in {"mage_vl", "qwen3_vl", "qwen3_vl_text"}
                    or model_type in _QWEN35_VL_MODEL_TYPES
                    or model_type in _QWEN4_EXP_MODEL_TYPES
                ):
                    patch_size = getattr(vision_cfg, "patch_size", None)
                    window_size = getattr(vision_cfg, "window_size", None)
                    if patch_size is not None:
                        vision_kwargs["patch_size"] = patch_size
                    if window_size is not None:
                        vision_kwargs["window_size"] = window_size
                    vision_kwargs["tokens_per_second"] = float(
                        getattr(config, "tokens_per_second", 2.0)
                    )
                if ep == "trt-rtx" and model_type in _QWEN35_VL_MODEL_TYPES:
                    image_feature_width = _get_static_graph_input_dim(
                        pkg,
                        "embedding",
                        "image_features",
                        -1,
                    )
                    vision_kwargs["embedding_provider_options"] = (
                        _make_trt_rtx_embedding_provider_options(
                            image_feature_width=image_feature_width,
                            input_id_lengths=(1, 226, 1024),
                            image_feature_lengths=(0, 192, 2520),
                        )
                    )
                    vision_kwargs["vision_provider_options"] = (
                        _QWEN35_TRT_RTX_VISION_PROVIDER_OPTIONS
                    )

            if vision_input_mapping is not None:
                vision_kwargs["input_names"] = vision_input_mapping
            # Introspect runtime-semantic vision outputs from the ONNX graph.
            # Model-specific auxiliary features must be packed into a supported
            # output by the task rather than emitted as arbitrary config keys.
            vision_output_mapping = _introspect_outputs(pkg, "vision_encoder")
            if vision_output_mapping is not None:
                vision_kwargs["output_names"] = vision_output_mapping
            if embedding_input_mapping is not None:
                vision_kwargs["embedding_input_names"] = embedding_input_mapping

            embedding_output_mapping = _introspect_outputs(pkg, "embedding")
            if embedding_output_mapping is not None:
                vision_kwargs["embedding_output_names"] = embedding_output_mapping
            vision_start_token_id = getattr(config, "vision_start_token_id", None)
            video_token_id = getattr(config, "video_token_id", None)
            if vision_start_token_id is not None:
                vision_kwargs["vision_start_token_id"] = vision_start_token_id
            if video_token_id is not None:
                vision_kwargs["video_token_id"] = video_token_id

            generator.with_vision(image_token_id=image_token_id, **vision_kwargs)

    if has_speech:
        audio_input_mapping = _introspect_inputs(pkg, "audio_encoder")

        audio_config = getattr(config, "audio", None)
        audio_token_id = (
            getattr(audio_config, "token_id", None)
            or getattr(audio_config, "audio_token_id", None)
            or getattr(config, "audio_token_id", None)
        )
        boa_token_id = getattr(config, "boa_token_id", None)

        audio_kwargs: dict[str, Any] = {}
        model_type = getattr(config, "model_type", "")
        if model_type in _GEMMA4_MODEL_TYPES:
            # Gemma4 audio encoder uses different filename and config
            audio_kwargs["filename"] = "audio_encoder/model.onnx"
            audio_kwargs["config_filename"] = "audio_feature_extraction.json"
            # Gemma4 speech model input is 'input_features' + 'input_features_mask'
            audio_kwargs["input_names"] = {
                "audio_embeds": "input_features",
                "attention_mask": "input_features_mask",
            }
        elif model_type in _GEMMA3N_MODEL_TYPES:
            # Same USM log-mel contract as gemma4 (and the same writer), so it
            # names the same file. The reference must be present: ORT-GenAI
            # rejects a speech section that sets ``filename`` without
            # ``config_filename`` ("Both are required for audio support"),
            # so omitting it would turn a missing-file error into a
            # load-time throw.
            audio_kwargs["config_filename"] = "audio_feature_extraction.json"
            # Same two graph inputs as gemma4, so the same schema mapping. This
            # must not come from _introspect_inputs: unlike the decoder and
            # vision sections, ``model.speech.inputs`` keys are a *closed set*
            # the runtime defines (audio_embeds / attention_mask / audio_sizes /
            # audio_projection_mode), not graph input names. An identity map is
            # rejected outright with 'model:speech:inputs: Unknown value
            # "input_features"'.
            audio_kwargs["input_names"] = {
                "audio_embeds": "input_features",
                "attention_mask": "input_features_mask",
            }
        elif model_type in _GLMASR_MODEL_TYPES:
            audio_kwargs["config_filename"] = "audio_processor.json"
            audio_kwargs["input_names"] = {
                "audio_embeds": "input_features",
                "attention_mask": "input_features_mask",
            }
        else:
            if audio_input_mapping is not None:
                audio_kwargs["input_names"] = audio_input_mapping
        generator.with_audio(
            audio_token_id=audio_token_id,
            boa_token_id=boa_token_id,
            **audio_kwargs,
        )
        embedding_inputs = _introspect_inputs(pkg, "embedding")
        embedding_outputs = _introspect_outputs(pkg, "embedding")
        if embedding_inputs is not None:
            generator.with_embedding(
                input_names=embedding_inputs,
                output_names=embedding_outputs,
            )

    return generator.write(output_dir)


def _validate_ort_genai_compatibility(pkg: ModelPackage) -> None:
    """Reject packages whose required inputs cannot be supplied by ORT GenAI."""
    config = getattr(pkg, "config", None)
    if getattr(config, "model_type", None) in _QWEN4_EXP_MODEL_TYPES:
        raise ValueError(
            "onnxruntime-genai 0.15.2 cannot represent Qwen4-Exp's explicit "
            "four-axis position state or heterogeneous per-layer PLE/QSA state "
            "membership. Use ModelPackage.save() and the decoder ONNX model's "
            "'mobius.state_manifest' metadata; refusing to emit an unsupported "
            "genai_config.json."
        )
    if getattr(config, "model_type", None) == "parakeet_ctc":
        raise ValueError(
            "ORT GenAI does not define a feature-input CTC ASR pipeline; "
            "export Parakeet CTC as ONNX and run it directly with ONNX Runtime."
        )
    if getattr(config, "model_type", None) in _GLMASR_MODEL_TYPES:
        raise ValueError(
            "onnxruntime-genai does not register a GLM-ASR multimodal model type. "
            "Export without --runtime ort-genai and run the audio_encoder, embedding, "
            "and decoder models directly with ONNX Runtime."
        )
    if {"vision_encoder", "decoder"}.issubset(pkg) and "embedding" not in pkg:
        model_type = getattr(config, "model_type", "unknown")
        raise NotImplementedError(
            "onnxruntime-genai does not support generic vision encoder-decoder "
            f"packages such as {model_type!r}. Run the vision_encoder and decoder "
            "ONNX sessions directly; emitting genai_config.json would create an "
            "artifact that the runtime cannot load."
        )
    if getattr(config, "model_type", None) == "mage_vl":
        raise ValueError(
            "ORT GenAI does not support Mage-VL's required patch_positions vision "
            "input or its 1D decoder position_ids contract. Export without "
            "--runtime ort-genai to save the runnable direct three-model ONNX package."
        )


def write_ort_genai_config(
    pkg: ModelPackage,
    directory: str,
    *,
    hf_model_id: str | None = None,
    revision: str | None = None,
    ep: str = "cpu",
    context_length: int = 4096,
    local_config_dir: str | None = None,
    trust_remote_code: bool = False,
    runtime_version: str | None = None,
) -> dict[str, str]:
    """Generate ORT-GenAI config artifacts for an already-built ModelPackage.

    Writes ``genai_config.json``, optionally copies tokenizer files from
    HuggingFace Hub or a local directory, and writes ``image_processor.json``
    for VLM models.  Does NOT build or save ONNX models — call
    :meth:`~mobius._model_package.ModelPackage.save` separately before or
    after this function.

    Args:
        pkg: Already-built :class:`~mobius._model_package.ModelPackage` with
            weights applied and ``config`` set.
        directory: Output directory (created if needed).
        hf_model_id: HuggingFace model ID or local model directory. When provided,
            used to fetch token IDs (``bos``/``eos``/``pad``) and copy tokenizer files.
            Token IDs from ``generation_config.json`` take precedence over model config
            values because generation configs may define additional stop tokens.
            When ``None``, token IDs are read from ``pkg.config`` fields
            (``bos_token_id``, ``eos_token_id``, ``pad_token_id``) populated
            by :meth:`~mobius._configs.ArchitectureConfig.from_transformers`,
            and tokenizer files are not copied unless ``local_config_dir`` is set.
        revision: Immutable HuggingFace revision used for the config, tokenizer,
            processor, and copied assets.
        ep: Execution provider for ``session_options`` in
            ``genai_config.json`` (e.g. ``"cpu"``, ``"cuda"``, ``"dml"``,
            ``"trt-rtx"``). Defaults to ``"cpu"``.
        context_length: Minimum context length written to
            ``genai_config.json``. Overridden upward by
            ``max_position_embeddings`` from ``pkg.config``.
        local_config_dir: Path to a local model directory. When provided
            and ``hf_model_id`` is ``None``, tokenizer files are copied from
            this directory instead of downloaded from HuggingFace Hub.
            Typically set when the CLI ``--config`` flag points to a local
            directory rather than a HuggingFace model ID.
        trust_remote_code: Allow custom HuggingFace configuration code when
            resolving token IDs and model type.
        runtime_version: Optional onnxruntime-genai version that will consume the
            package. Generic decoder packages reject versions older than 0.14.0.

    Returns:
        Dict mapping artifact name to file path, e.g.::

            {
                "genai_config": "/output/genai_config.json",
                "tokenizer.json": "/output/tokenizer.json",
                "image_processor": "/output/image_processor.json",
            }

    Raises:
        ValueError: If ``pkg.config`` is ``None`` (required for config
            generation).
    """
    config = getattr(pkg, "config", None)
    if config is None:
        raise ValueError(
            "write_ort_genai_config requires ModelPackage.config to be set. "
            "This is set automatically when building with mobius.build(). "
            "Diffusion models (which have no config) are not supported."
        )
    _validate_ort_genai_compatibility(pkg)

    if getattr(config, "model_type", None) == "moonshine":
        raise NotImplementedError(
            "onnxruntime-genai does not support Moonshine's variable-length raw-waveform "
            "encoder. Run the exported encoder and cached decoder directly with "
            "ONNX Runtime."
        )

    os.makedirs(directory, exist_ok=True)

    # Normalize EP: 'default' and 'onnx-standard' are portable-ONNX modes
    # that carry no EP-specific session options → treat as CPU.
    if ep in ("default", "onnx-standard"):
        ep = "cpu"

    # Resolve token IDs and ORT model type from HF config (if provided)
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    ort_model_type: str

    # Generic decoder dispatch is intentionally limited to one-graph text packages.
    # Auxiliary encoder, pipeline, and sidecar graphs require their own runtime contract.
    is_vlm = "vision_encoder" in pkg and "embedding" in pkg
    has_speech = "audio_encoder" in pkg
    is_decoder_only = _is_single_model_decoder_package(pkg)

    if hf_model_id is not None:
        import transformers

        hf_config = transformers.AutoConfig.from_pretrained(
            hf_model_id,
            trust_remote_code=trust_remote_code,
            **_revision_kwargs(revision),
        )
        model_type = hf_config.model_type
        cfg_model_type = getattr(config, "model_type", None)
        # See _select_ort_model_type: decoder-only packages prefer the package's
        # own config.model_type; multimodal packages keep the HF parent type.
        ort_model_type = _select_ort_model_type(
            cfg_model_type,
            model_type,
            is_decoder_only=is_decoder_only,
            rope_type=getattr(config, "rope_type", None),
        )
        # Token IDs may live on the parent config or the text sub-config
        # (e.g. Gemma4Config has text_config with bos_token_id=2).
        _tok_cfg = getattr(hf_config, "text_config", hf_config)
        bos_token_id = getattr(
            hf_config,
            "bos_token_id",
            getattr(_tok_cfg, "bos_token_id", None),
        )
        eos_token_id = getattr(
            hf_config,
            "eos_token_id",
            getattr(_tok_cfg, "eos_token_id", None),
        )
        pad_token_id = getattr(
            hf_config,
            "pad_token_id",
            getattr(_tok_cfg, "pad_token_id", None),
        )
    else:
        # Fall back to fields stored in ArchitectureConfig (set by from_transformers()).
        # This path is taken when hf_model_id is not provided (e.g. --config mode).
        raw_type = getattr(config, "model_type", None) or "unknown"
        if is_vlm and raw_type in _UNWRAPPED_VLM_MODEL_TYPES:
            ort_model_type = _UNWRAPPED_VLM_MODEL_TYPES[raw_type]
        else:
            ort_model_type = _select_ort_model_type(
                raw_type,
                raw_type,
                is_decoder_only=is_decoder_only,
                rope_type=getattr(config, "rope_type", None),
            )
        if ort_model_type == "unknown":
            logger.warning(
                "Could not determine ORT-GenAI model type: pkg.config.model_type "
                "is missing, None, or not mapped to an ORT-GenAI type (got %r). "
                "Pass hf_model_id to resolve it from the HuggingFace config, or "
                "the generated genai_config.json may not load correctly.",
                raw_type,
            )
        # Read token IDs from ArchitectureConfig (populated by from_transformers()
        # when --config is used with a local directory).
        bos_token_id = getattr(config, "bos_token_id", None)
        eos_token_id = getattr(config, "eos_token_id", None)
        # pad_token_id lives in BaseModelConfig with DEFAULT_INT (-42) as the
        # "not set" sentinel (negative IDs are never valid token positions).
        _pad = getattr(config, "pad_token_id", None)
        pad_token_id = None if (_pad is None or _pad < 0) else _pad

    generation_config_source = hf_model_id or local_config_dir
    if generation_config_source and (
        generation_config := _load_generation_config(generation_config_source)
    ):
        generation_bos_token_id = getattr(generation_config, "bos_token_id", None)
        generation_eos_token_id = getattr(generation_config, "eos_token_id", None)
        generation_pad_token_id = getattr(generation_config, "pad_token_id", None)
        if generation_bos_token_id is not None:
            bos_token_id = generation_bos_token_id
        if generation_eos_token_id is not None:
            eos_token_id = generation_eos_token_id
        if generation_pad_token_id is not None:
            pad_token_id = generation_pad_token_id

    # Phi4MM quirk: HF reports model_type='phi' but the model package
    # includes an 'audio_encoder' component that distinguishes it from plain Phi.
    # Override to 'phi4mm' so ORT-GenAI loads the correct pipeline.
    if ort_model_type == "phi" and has_speech:
        ort_model_type = "phi4mm"
    if (
        ort_model_type == "decoder"
        and runtime_version is not None
        and _runtime_version_tuple(runtime_version) < _GENERIC_DECODER_MIN_VERSION
    ):
        raise ValueError(
            "Generic ORT GenAI decoder packages require onnxruntime-genai >= 0.14.0; "
            f"requested {runtime_version}"
        )
    result: dict[str, str] = {}

    if "mtp" in pkg:
        mtp_model = pkg["mtp"]
        mtp_path = os.path.join(directory, "mtp_config.json")
        with open(mtp_path, "w") as f:
            json.dump(
                {
                    "model": {"filename": "mtp/model.onnx"},
                    "inputs": [
                        value.name
                        for value in mtp_model.graph.inputs
                        if value.name is not None
                    ],
                    "outputs": [
                        value.name
                        for value in mtp_model.graph.outputs
                        if value.name is not None
                    ],
                    "num_nextn_predict_layers": getattr(config, "num_nextn_predict_layers", 0),
                    "shared_embedding": "model.embed_tokens",
                    "shared_lm_head": "lm_head",
                    "runtime_orchestration": "external",
                },
                f,
                indent=2,
            )
            f.write("\n")
        result["mtp_config"] = mtp_path

    # Copy tokenizer files. A local hf_model_id is a local model directory, not a
    # Hub repo id; copy directly instead of calling hf_hub_download.
    if hf_model_id is not None:
        if os.path.isdir(hf_model_id):
            logger.info("Copying tokenizer files from local model directory %s", hf_model_id)
            tokenizer_files = _copy_tokenizer_files_from_local(hf_model_id, directory)
        else:
            logger.info("Copying tokenizer files from %s", hf_model_id)
            tokenizer_files = _copy_tokenizer_files(
                hf_model_id,
                directory,
                **_revision_kwargs(revision),
            )
        for tf in tokenizer_files:
            result[tf] = os.path.join(directory, tf)
    elif local_config_dir is not None:
        logger.info("Copying tokenizer files from local directory %s", local_config_dir)
        tokenizer_files = _copy_tokenizer_files_from_local(local_config_dir, directory)
        if not tokenizer_files:
            logger.warning(
                "No tokenizer files were copied from local directory %s. "
                "The export may be missing tokenizer artifacts required by ORT-GenAI.",
                local_config_dir,
            )

        for tf in tokenizer_files:
            result[tf] = os.path.join(directory, tf)

    logger.info("Generating genai_config.json for %s (ep=%s)", ort_model_type, ep)
    genai_path = _write_genai_config(
        config,
        directory,
        pkg=pkg,
        ort_model_type=ort_model_type,
        ep=ep,
        context_length=context_length,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        is_vlm=is_vlm,
        has_speech=has_speech,
    )
    result["genai_config"] = genai_path
    compatibility_path = _write_runtime_compatibility(
        directory,
        model_type=ort_model_type,
        runtime_version=runtime_version,
    )
    result["runtime_compatibility"] = compatibility_path

    # Write processor config for VLMs
    processor_path = _write_vision_processor_config(
        config,
        directory,
        hf_model_id=hf_model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    if processor_path:
        result["processor_config"] = processor_path

    # Write audio_processor.json for models with audio encoders
    audio_proc_path = _write_audio_processor_config(config, directory)
    if audio_proc_path:
        result["audio_processor"] = audio_proc_path

    # Fix unsupported tokenizer classes
    _fix_tokenizer_config(directory)

    # Ensure chat_template is in tokenizer_config.json
    _fix_chat_template(
        directory,
        hf_model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )

    # Correct assets that ship broken from upstream
    apply_asset_patches(directory)

    logger.info("ORT-GenAI artifacts written: %d files", len(result))
    return result


def export_package(
    pkg: ModelPackage,
    output_dir: str,
    *,
    hf_model_id: str | None = None,
    revision: str | None = None,
    ep: str = "cpu",
    context_length: int = 4096,
    local_config_dir: str | None = None,
    trust_remote_code: bool = False,
    runtime_version: str | None = None,
    external_data: str = "onnx",
    progress_bar: bool = True,
) -> dict[str, str]:
    """Save an already-built ModelPackage as a complete ORT-GenAI directory.

    This is the convenience function for users who built a ``ModelPackage``
    themselves (e.g. with custom dtype / quantization / weight overrides) and
    want a single call that produces an ``onnxruntime-genai``-loadable
    directory.  It calls :meth:`ModelPackage.save` followed by
    :func:`write_ort_genai_config`.

    For the end-to-end case where you start from a HuggingFace model id, use
    :func:`auto_export` instead — it builds the package for you.

    Args:
        pkg: Already-built :class:`~mobius._model_package.ModelPackage` with
            weights applied and ``config`` set.  All components in *pkg* are
            saved; selective export via :meth:`ModelPackage.save`'s
            ``components`` filter is intentionally not exposed here, because
            the ORT-GenAI runtime expects the on-disk file layout to match
            what ``model.type`` in :file:`genai_config.json` declares (e.g. a
            multimodal ``model.type`` implies a vision encoder file is
            present).  If you need a partial export, build a separate
            filtered ``ModelPackage`` first.
        output_dir: Output directory (created if needed).
        hf_model_id: HuggingFace model ID for tokenizer download / token-id
            resolution.  When ``None``, token IDs are read from ``pkg.config``
            and tokenizer files are not copied (unless ``local_config_dir``
            is provided).
        revision: Immutable HuggingFace revision used for all downloaded
            configuration, tokenizer, processor, and asset files.
        ep: Execution provider written to ``session_options`` in
            ``genai_config.json`` (e.g. ``"cpu"``, ``"cuda"``, ``"dml"``,
            ``"webgpu"``, ``"trt-rtx"``).
        context_length: Minimum context length written to ``genai_config.json``.
            Overridden upward by ``pkg.config.max_position_embeddings`` when
            larger.
        local_config_dir: Local model directory to copy tokenizer files from
            when ``hf_model_id`` is ``None``.
        trust_remote_code: Allow custom HuggingFace configuration code when
            resolving token IDs and model type.
        revision: Optional immutable HuggingFace revision used for remote
            configuration, tokenizer, and processor requests.
        runtime_version: Optional onnxruntime-genai version that will consume
            the package.
        external_data: External-data format passed to :meth:`ModelPackage.save`
            (``"onnx"`` or ``"safetensors"``).
        progress_bar: Whether to show the save progress bar.

    Returns:
        Manifest dict mapping artifact names to paths::

            {
                "model": "/output/model.onnx",          # or per-component paths
                "genai_config": "/output/genai_config.json",
                "tokenizer.json": "/output/tokenizer.json",
                ...
            }

    Raises:
        ValueError: If ``pkg.config`` is ``None`` (required for genai_config
            generation; e.g. diffusion models have no config and are not
            supported).

    Example::

        from mobius import build
        from mobius.integrations.ort_genai import export_package

        pkg = build("Qwen/Qwen3-0.6B", load_weights=True)
        export_package(pkg, "/output/qwen3", hf_model_id="Qwen/Qwen3-0.6B", ep="cuda")
    """
    # Preflight: fail fast before writing ONNX so the user doesn't end up
    # with a half-exported directory containing only the model file.
    if getattr(pkg, "config", None) is None:
        raise ValueError(
            "export_package requires ModelPackage.config to be set. "
            "This is set automatically when building with mobius.build(). "
            "Diffusion models (which have no config) are not supported — "
            "use ModelPackage.save() directly for those."
        )
    _validate_ort_genai_compatibility(pkg)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Save ONNX models + weights
    logger.info("Saving ONNX models to %s", output_dir)
    pkg.save(
        output_dir,
        external_data=external_data,
        progress_bar=progress_bar,
    )

    # 2. Write ORT-GenAI config artifacts
    result = write_ort_genai_config(
        pkg,
        output_dir,
        hf_model_id=hf_model_id,
        revision=revision,
        ep=ep,
        context_length=context_length,
        local_config_dir=local_config_dir,
        trust_remote_code=trust_remote_code,
        runtime_version=runtime_version,
    )

    # 3. Add ONNX paths to the manifest
    if len(pkg) == 1:
        result["model"] = os.path.join(output_dir, "model.onnx")
    else:
        for name in pkg:
            result[name] = os.path.join(output_dir, name, "model.onnx")

    logger.info("Export complete: %d artifacts", len(result))
    return result


def auto_export(
    model_id: str,
    output_dir: str,
    *,
    revision: str | None = None,
    dtype: str | None = None,
    task: str | None = None,
    external_data: str = "onnx",
    trust_remote_code: bool = False,
    context_length: int = 4096,
    ep: str = "cpu",
    progress_bar: bool = True,
    text_only: bool = False,
) -> dict[str, str]:
    """Build and export a model for onnxruntime-genai.

    This is the end-to-end convenience function for producing ORT-GenAI-ready
    model directories. It:

    1. Builds the ONNX graph(s) via :func:`mobius.integrations.transformers.build`
    2. Downloads and applies HuggingFace weights
    3. Saves ONNX model(s) with external data
    4. Calls :func:`write_ort_genai_config` to write ``genai_config.json``,
       tokenizer files, and ``image_processor.json``

    Args:
        model_id: HuggingFace model repository ID.
        output_dir: Directory to write all output files.
        revision: Immutable HuggingFace revision used for all downloads.
        dtype: Override model dtype (``"f32"``, ``"f16"``, ``"bf16"``).
        task: Override model task (auto-detected if ``None``).
        external_data: External data format (``"onnx"`` or
            ``"safetensors"``).
        trust_remote_code: Trust remote code for HuggingFace config.
        revision: Optional immutable HuggingFace revision used for all Hub
            configuration, weight, tokenizer, and processor requests.
        context_length: Minimum context length for genai_config.json.
        ep: Execution provider for ``session_options`` in
            ``genai_config.json``. Defaults to ``"cpu"``. For non-CPU providers
            this value also drives build-time ``execution_provider`` so the
            exported ONNX graph is fused for the same provider the runtime will
            use (e.g. ``"cuda"`` enables ``GroupQueryAttention`` fusion).
            ``"cpu"`` builds the portable ``"default"`` graph (unchanged
            behavior).
        progress_bar: Show progress bar during save.
        text_only: When ``True``, export the text backbone of a multimodal
            checkpoint as a standalone decoder-only LLM (see
            :func:`mobius.integrations.transformers.build`). Produces a single
            ``model.onnx``
            with a decoder-only ``genai_config.json`` (no vision/audio
            sections). Currently supported for ``gemma4_unified``
            (``google/gemma-4-12B``).

    Returns:
        Dict mapping output artifact names to file paths, e.g.::

            {
                "genai_config": "/output/genai_config.json",
                "model": "/output/model.onnx",
                "tokenizer.json": "/output/tokenizer.json",
            }
    """
    from mobius.integrations.transformers import build

    os.makedirs(output_dir, exist_ok=True)

    # Build ONNX graph(s) with weights. The runtime EP (``ep``) also drives
    # EP-aware graph construction so fused ops (e.g. GroupQueryAttention on
    # CUDA) match the provider declared in genai_config.json. ``cpu`` maps to
    # the portable ``default`` build to preserve historical CPU/f32 output
    # (the CPU EP would otherwise fuse f32 GroupQueryAttention).
    build_ep = "default" if ep == "cpu" else ep
    logger.info("Building ONNX model for %s (build ep=%s)", model_id, build_ep)
    pkg = build(
        model_id,
        task=task,
        revision=revision,
        dtype=dtype,
        load_weights=True,
        trust_remote_code=trust_remote_code,
        execution_provider=build_ep,
        text_only=text_only,
    )

    if getattr(pkg, "config", None) is None:
        raise ValueError(
            f"Model package for '{model_id}' has no config attribute. "
            "auto_export requires a config to generate genai_config.json. "
            "Diffusion models are not yet supported."
        )

    # Delegate save + config generation to the integration helper.
    # `export_package` already logs "Export complete" so we don't repeat it here.
    result = export_package(
        pkg,
        output_dir,
        hf_model_id=model_id,
        revision=revision,
        ep=ep,
        context_length=context_length,
        trust_remote_code=trust_remote_code,
        external_data=external_data,
        progress_bar=progress_bar,
    )

    return result
