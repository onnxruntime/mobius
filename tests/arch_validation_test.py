# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""L2 Architecture Validation Tests — real HF config, no weights.

Downloads config.json from HuggingFace for each registered model type
that has a ``test_model_id``, builds the full-size ONNX graph (no weights),
and validates graph structure and shape consistency.

These tests verify that our ONNX graph construction is compatible with
real-world model configurations — catching shape mismatches, missing
config fields, and initialization errors that tiny synthetic configs
would not trigger.

Must stay under 2 GB RAM. Run sequentially (no pytest-xdist)::

    pytest tests/arch_validation_test.py -m arch_validation -v --tb=short

To run a single architecture::

    pytest tests/arch_validation_test.py -k "llama" -v

These tests are designed for nightly CI or tag-triggered runs.
They require network access to download config.json from HuggingFace.
"""

from __future__ import annotations

import logging

import onnx_ir as ir
import pytest

from mobius._registry import registry
from mobius.integrations.transformers._config_resolver import (
    _config_from_hf,
    _default_task_for_model,
    _try_load_config_json,
)
from mobius.tasks import get_task

logger = logging.getLogger(__name__)

# Build parametrized test cases from registry entries that have a test_model_id.
#
# We split known failures by which subset of tests they apply to:
#
# * ``_PARSE_AND_GRAPH_XFAILS`` — config loads successfully *and* would build
#   a graph successfully if not for a known limitation. xfail all three
#   sub-tests (currently empty; phi3small previously lived here but starts
#   passing once gegelu is implemented and was XPASS-ing across all three).
# * ``_GRAPH_ONLY_XFAILS`` — config parses fine; only the full-graph build
#   and shape-consistency tests fail. This is the common case for VL models
#   whose vision sub-config requires ``trust_remote_code=True``.
_PARSE_AND_GRAPH_XFAILS: dict[str, str] = {}

_GRAPH_ONLY_XFAILS: dict[str, str] = {
    # VL models with missing/incomplete vision_config when loaded without
    # trust_remote_code — the HF config JSON doesn't expose vision fields.
    "deepseek_vl": "VisionConfig.hidden_size missing without trust_remote_code",
    "deepseek_vl_hybrid": "VisionConfig.hidden_size missing without trust_remote_code",
    "fuyu": "FuyuConfig has no vision_config (image processing is in-model)",
    "florence2": "Florence2 DaViT vision encoder is multi-stage (not standard ViT)",
    "got_ocr2": "VisionConfig missing without trust_remote_code",
    "janus": "VisionConfig.hidden_size missing without trust_remote_code",
    "kimi_k3": "Selective MXFP4 compressed-tensors experts are unsupported by the "
    "generic quantized-linear loader",
    "molmo": "VisionConfig missing without trust_remote_code",
    "ovis2": "VisionConfig missing without trust_remote_code",
}


def _build_arch_params(extra_xfails: dict[str, str]):
    """Build the pytest.param list, applying xfail marks from both dicts."""
    params = []
    for model_type in sorted(registry.architectures()):
        registration = registry.get_registration(model_type)
        if registration.test_model_id is None:
            continue
        reason = _PARSE_AND_GRAPH_XFAILS.get(model_type) or extra_xfails.get(model_type)
        marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason is not None else []
        params.append(
            pytest.param(
                model_type,
                registration.test_model_id,
                id=model_type,
                marks=marks,
            )
        )
    return params


# Parse test sees only the always-failing models; VL graph-only failures are
# expected to PARSE cleanly.
_PARSE_PARAMS = _build_arch_params({})
# Graph tests see both always-failing and graph-only failures.
_GRAPH_PARAMS = _build_arch_params(_GRAPH_ONLY_XFAILS)


def _load_hf_config(model_id: str, revision: str | None = None):
    """Load HF config using AutoConfig first, fallback to raw config.json.

    AutoConfig handles model-specific field mappings (e.g. GPT-2's
    ``n_embd`` → ``hidden_size``).  Raw config.json is used when the
    model type isn't registered in transformers.
    """
    import transformers
    from huggingface_hub import errors as hub_errors

    strict_validation_error = getattr(
        hub_errors,
        "StrictDataclassClassValidationError",
        ValueError,
    )
    try:
        return transformers.AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
        )
    except (strict_validation_error, ValueError, OSError):
        return _try_load_config_json(model_id, revision=revision)


def _resolve_hf_config(hf_config, registration=None):
    """Resolve nested config wrappers (thinker, talker, text, llm, decoder).

    Mirrors the resolution logic in ``build()`` — some models
    wrap the actual config inside a parent config.

    The generic ``decoder`` unwrap is skipped for a model whose registry entry
    names its own ``config_class``: such a class consumes the *composite* config
    and reads the sub-configs itself (Nemotron Parse reads both
    ``config.decoder`` and ``config.encoder``). Unwrapping first hands it a
    sub-config, which loses the parent's ``model_type``, so resolution falls back
    to plain ``ArchitectureConfig``.

    Only that branch. Production ``_select_primary_config`` has no generic
    ``decoder`` unwrap at all — this harness added one — whereas the
    ``talker``/``thinker``/``text``/``llm`` branches do mirror production and are
    still needed by models that also declare a ``config_class`` (lfm2_vl,
    muse_glimmer).
    """
    parent_config = hf_config
    owns_composite = (
        registration is not None and getattr(registration, "config_class", None) is not None
    )
    architectures = getattr(hf_config, "architectures", None) or []
    if (
        getattr(hf_config, "model_type", None) == "vibevoice"
        and architectures == ["VibeVoiceForASRStreamingTraining"]
        and hasattr(hf_config, "decoder_config")
    ):
        hf_config = hf_config.decoder_config
    elif hasattr(hf_config, "talker_config"):
        talker = hf_config.talker_config
        # Qwen3-Omni talker nests the real model config under text_config
        if hasattr(talker, "text_config"):
            hf_config = talker.text_config
        else:
            hf_config = talker
    elif hasattr(hf_config, "thinker_config"):
        thinker = hf_config.thinker_config
        if hasattr(thinker, "text_config"):
            hf_config = thinker.text_config
        else:
            hf_config = thinker
    elif hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config
    elif hasattr(hf_config, "llm_config"):
        # InternVL2 wraps the LLM config under llm_config
        hf_config = hf_config.llm_config
    elif hasattr(hf_config, "decoder") and not owns_composite:
        # VisionEncoderDecoder (TrOCR) wraps decoder config
        hf_config = hf_config.decoder
    return hf_config, parent_config


def _build_graph(model_type: str, model_id: str, *, revision: str | None = None):
    """Download config, build ONNX graph, return ``(ModelPackage, task)``.

    Uses get_task().build() directly (same pattern as the L1 graph tests)
    to bypass ArchitectureConfig.validate() which rejects non-LM configs
    (e.g. vision models with vocab_size=0).
    """
    registration = registry.get_registration(model_type)
    hf_config = _load_hf_config(
        model_id,
        revision=registration.test_revision if revision is None else revision,
    )
    if hf_config is None:
        pytest.skip(
            f"Cannot download config for {model_id} (gated/private model or network error)"
        )

    hf_config, parent_config = _resolve_hf_config(hf_config, registration)
    config = _config_from_hf(
        hf_config,
        parent_config=parent_config,
        module_class=registration.module_class,
    )

    module = registration.module_class(config)
    task_name = registration.task or _default_task_for_model(model_type)
    task = get_task(task_name)
    return task.build(module, config), task


@pytest.mark.arch_validation
class TestArchValidation:
    """L2 architecture validation: download real config, build full graph.

    Each test downloads only the config.json (no model weights) from
    HuggingFace, constructs the full-size ONNX graph, and validates
    that the graph has a reasonable structure.
    """

    @pytest.mark.parametrize("model_type,model_id", _PARSE_PARAMS)
    def test_config_downloads_and_parses(self, model_type: str, model_id: str):
        """Verify config.json can be downloaded and parsed."""
        revision = registry.get_registration(model_type).test_revision
        hf_config = _load_hf_config(model_id, revision=revision)
        if hf_config is None:
            pytest.skip(
                f"Cannot download config for {model_id} (gated/private model or network error)"
            )
        # The config must have a model_type
        assert hasattr(hf_config, "model_type"), f"Config for {model_id} missing model_type"

    @pytest.mark.parametrize("model_type,model_id", _GRAPH_PARAMS)
    def test_full_graph_builds(self, model_type: str, model_id: str):
        """Build full-size ONNX graph from real HF config (no weights).

        This is the core L2 validation: if the real config triggers
        shape mismatches, missing fields, or initialization errors, this
        test will catch them.
        """
        pkg, task = _build_graph(model_type, model_id)

        # Validate: every component has a non-empty graph
        assert len(pkg) > 0, "ModelPackage is empty"
        for component_name, model in pkg.items():
            assert model.graph is not None, f"{component_name} graph is None"
            nodes = list(model.graph)
            assert len(nodes) > 0, f"{component_name} has no nodes"
            assert len(model.graph.inputs) > 0, f"{component_name} has no inputs"
            assert len(model.graph.outputs) > 0, f"{component_name} has no outputs"

        # Every component the task builds must declare a role. An undeclared
        # component falls back to the "decoder" role in build_from_module and
        # would be handed fusion passes meant for attention stacks, and
        # inspect_components would not report it at all.
        undeclared = sorted(set(pkg) - set(task.model_roles or {}))
        assert not undeclared, (
            f"{model_type}: components {undeclared} are built but missing from "
            f"{type(task).__name__}.model_roles"
        )

        del pkg

    @pytest.mark.parametrize("model_type,model_id", _GRAPH_PARAMS)
    def test_graph_shapes_consistent(self, model_type: str, model_id: str):
        """Validate structural consistency in the full-size graph.

        Checks that the graph has initializers (parameters) and that
        inputs/outputs are defined. Note: output type info may not be
        available for all outputs when building without shape inference.
        """
        pkg, task = _build_graph(model_type, model_id)
        roles = task.model_roles or {}

        for component_name, model in pkg.items():
            # Model should have initializers (parameters). "glue" components are
            # parameter-free loop wiring — they read every tensor they use from
            # a graph input, so they hold only hoisted constants (if anything)
            # and the parameter check does not apply.
            if roles.get(component_name) == "glue":
                continue
            initializers = list(model.graph.initializers)
            assert len(initializers) > 0, (
                f"{component_name} has no initializers — graph may be missing parameters"
            )

            # All inputs should have names
            for inp in model.graph.inputs:
                assert inp.name, f"{component_name} has an unnamed input"

            # All outputs should have names
            for output in model.graph.outputs:
                assert output.name, f"{component_name} has an unnamed output"

        del pkg

    @pytest.mark.parametrize(
        ("model_id", "revision", "hidden_size", "q_heads", "kv_heads", "tied_embeddings"),
        (
            (
                "microsoft/VibeVoice-ASR-Streaming-1.5B",
                "4262d23d8a539a6530cf64fbd0b1751ef9a30853",
                1536,
                12,
                2,
                True,
            ),
            (
                "microsoft/VibeVoice-ASR-Streaming-7B",
                "60d858b518b4e19d404af3737f848fc185b30177",
                3584,
                28,
                4,
                False,
            ),
        ),
    )
    def test_vibevoice_asr_pinned_variants_build(
        self,
        model_id: str,
        revision: str,
        hidden_size: int,
        q_heads: int,
        kv_heads: int,
        tied_embeddings: bool,
    ):
        """Build both official streaming-ASR variants from their pinned configs."""
        pkg, task = _build_graph(
            "VibeVoiceForASRStreamingTraining",
            model_id,
            revision=revision,
        )

        config = pkg.config
        assert config.hidden_size == hidden_size
        assert config.num_attention_heads == q_heads
        assert config.num_key_value_heads == kv_heads
        assert config.tie_word_embeddings is tied_embeddings
        assert config.dtype == ir.DataType.FLOAT
        assert config.acoustic_tokenizer.hidden_size == 64
        assert config.semantic_tokenizer.hidden_size == 128
        assert config.compression_ratio == 3200
        assert config.sampling_rate == 24_000
        assert set(pkg) == {"audio_encoder", "embedding", "decoder"}
        assert task.model_roles == {
            "audio_encoder": "encoder",
            "embedding": "embedding",
            "decoder": "decoder",
        }


class TestRegistryConsistency:
    """Verify registry and model class metadata are consistent."""

    def test_config_class_declared_on_model(self):
        """Registry config_class must match the model class declaration.

        When a registry entry specifies a non-default config_class,
        the model class must also declare it (not silently inherit a
        different one from a parent).

        Catches bugs like LongcatFlash where a missing config_class
        declaration caused the model to load with the wrong config,
        producing incorrect ONNX graphs without any error.
        """
        from mobius._configs import ArchitectureConfig, CausalLMConfig

        issues = []
        for model_type, reg in registry._map.items():
            reg_config = reg.config_class
            if reg_config is None or reg_config is ArchitectureConfig:
                continue

            cls = reg.module_class
            model_config = getattr(cls, "config_class", None)

            # If registry specifies a specialized config (not the base
            # CausalLMConfig), the model class should agree — unless the
            # model doesn't define config_class at all (multimodal models
            # that aren't CausalLMModel subclasses rely on the registry).
            if (
                reg_config is not CausalLMConfig
                and model_config is not None
                and model_config is not reg_config
            ):
                issues.append(
                    f"{model_type}: registry says "
                    f"{reg_config.__name__} but {cls.__name__} "
                    f"has {getattr(model_config, '__name__', None)}"
                )
        assert not issues, "Registry/model config_class mismatch:\n" + "\n".join(issues)
