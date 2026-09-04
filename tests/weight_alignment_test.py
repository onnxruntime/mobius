# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tier 1: Weight alignment tests for preprocess_weights().

Verifies that preprocess_weights() doesn't drop or mangle ONNX
initializer names when the state dict already uses ONNX-aligned names
(identity mapping).  This catches bugs like:

- Falcon's ``h.`` prefix replacement corrupting weight names
- MoE fused weight names being dropped
- Unintended name collisions in weight renaming

Test design:
    1. Build the ONNX graph with a tiny config
    2. Collect parameter initializer names (excluding computed constants)
    3. Create an identity state dict: ``{name: ones(shape)}``
    4. Run ``preprocess_weights()`` on that dict
    5. Assert every original parameter name is still present

Models whose ``preprocess_weights()`` intentionally filters through
HF-specific patterns (GPT2, OPT, etc.) are marked ``xfail`` since
they cannot roundtrip ONNX-aligned names by design.
"""

from __future__ import annotations

import pytest
import torch
from _test_configs import (
    ALL_CAUSAL_LM_CONFIGS,
    DETECTION_CONFIGS,
    ENCODER_CONFIGS,
    SEQ2SEQ_CONFIGS,
    SPEECH_CONFIGS,
    VISION_CONFIGS,
    VL_CONFIGS,
    _base_config,
    vl_overrides,
)

from mobius._registry import registry
from mobius.integrations.transformers._config_resolver import _default_task_for_model
from mobius.tasks import get_task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_parameter_names(pkg: dict) -> set[str]:
    """Return ONNX initializer names that represent model parameters.

    Excludes computed constants (RoPE caches, scalar constants, etc.)
    which have ``const_value is not None``.
    """
    names: set[str] = set()
    for sub_model in pkg.values():
        for name in sub_model.graph.initializers:
            init = sub_model.graph.initializers[name]
            if init.const_value is None:
                names.add(name)
    return names


def _build_identity_state_dict(pkg: dict, param_names: set[str]) -> dict[str, torch.Tensor]:
    """Create a state dict with ones for each parameter initializer."""
    state_dict: dict[str, torch.Tensor] = {}
    for sub_model in pkg.values():
        for name in sub_model.graph.initializers:
            if name not in param_names or name in state_dict:
                continue
            init = sub_model.graph.initializers[name]
            if init.shape is not None and len(init.shape) > 0:
                shape = list(init.shape)
                state_dict[name] = torch.ones(shape)
            else:
                # Scalar parameter
                state_dict[name] = torch.ones(())
    return state_dict


# Models whose preprocess_weights() intentionally filters names through
# HF-specific patterns. These cannot roundtrip ONNX-aligned names.
_FILTERING_PREPROCESS_MODELS: set[str] = {
    # OPT: expects model.decoder.* HF format
    "opt",
    # ModernBert decoder: expects model.layers.* HF format with renames
    "modernbert-decoder",
}


def _mark_xfail_if_filtering(
    configs: list[tuple[str, dict, bool]],
) -> list:
    """Build pytest params, marking filtering models as xfail."""
    return _mark_xfail_if_filtering_set(configs, _FILTERING_PREPROCESS_MODELS)


def _mark_xfail_if_filtering_set(
    configs: list[tuple[str, dict, bool]],
    filtering_models: set[str],
) -> list:
    """Build pytest params, marking specified models as xfail."""
    params = []
    for model_type, overrides, _ in configs:
        if model_type in filtering_models:
            params.append(
                pytest.param(
                    model_type,
                    overrides,
                    id=model_type,
                    marks=pytest.mark.xfail(
                        reason="preprocess_weights() filters HF-only patterns",
                        strict=True,
                    ),
                )
            )
        else:
            params.append(pytest.param(model_type, overrides, id=model_type))
    return params


def _assert_identity_roundtrip(model_type: str, config_overrides: dict) -> None:
    """Build model, run identity state dict through preprocess_weights()."""
    config = _base_config(**config_overrides)
    model_cls = registry.get(model_type)
    module = model_cls(config)
    task_name = _default_task_for_model(model_type)
    task = get_task(task_name)
    pkg = task.build(module, config)

    param_names = _collect_parameter_names(pkg)
    if not param_names:
        pytest.skip("No parameter initializers in model")

    if not hasattr(module, "preprocess_weights"):
        pytest.skip("Model has no preprocess_weights()")

    state_dict = _build_identity_state_dict(pkg, param_names)
    result = module.preprocess_weights(state_dict)

    missing = param_names - set(result.keys())
    assert not missing, (
        f"preprocess_weights() dropped {len(missing)} parameter(s): {sorted(missing)[:10]}"
    )


def test_vibevoice_native_hf_weights_cover_every_stage_parameter():
    """The converted checkpoint namespace routes without missing trained weights."""
    modeling = pytest.importorskip("transformers.models.vibevoice.modeling_vibevoice")
    from mobius._configs import VibeVoiceConfig
    from mobius.models.vibevoice import VibeVoiceForConditionalGeneration
    from mobius.models.vibevoice_test import _make_tiny_hf_config
    from mobius.tasks import VibeVoiceTask

    torch.manual_seed(11)
    hf_config = _make_tiny_hf_config()
    config = VibeVoiceConfig.from_transformers(
        hf_config.text_config,
        parent_config=hf_config,
    )
    module = VibeVoiceForConditionalGeneration(config)
    package = VibeVoiceTask().build(module, config)
    parameter_names = _collect_parameter_names(package)
    routed = module.preprocess_weights(
        modeling.VibeVoiceForConditionalGeneration(hf_config).state_dict()
    )

    assert parameter_names == set(routed)


@pytest.mark.arch_validation
def test_vibevoice_asr_checkpoint_index_classifies_every_tensor_once(tmp_path):
    """The pinned original ASR checkpoint routes only source-executed tensors."""
    import json

    from huggingface_hub import hf_hub_download

    from mobius.models.vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from mobius.models.vibevoice_test import _make_tiny_hf_config

    index_path = hf_hub_download(
        "microsoft/VibeVoice-ASR",
        filename="model.safetensors.index.json",
        revision="d0c9efdb8d614685062c04425d91e01b6f37d944",
        local_dir=tmp_path,
    )
    with open(index_path, encoding="utf-8") as handle:
        checkpoint_names = set(json.load(handle)["weight_map"])

    categories = {
        "acoustic_encoder": {
            name
            for name in checkpoint_names
            if name.startswith("model.acoustic_tokenizer.encoder.")
        },
        "acoustic_decoder_unused": {
            name
            for name in checkpoint_names
            if name.startswith("model.acoustic_tokenizer.decoder.")
        },
        "semantic_encoder": {
            name
            for name in checkpoint_names
            if name.startswith("model.semantic_tokenizer.encoder.")
        },
        "connectors": {
            name
            for name in checkpoint_names
            if name.startswith(("model.acoustic_connector.", "model.semantic_connector."))
        },
        "embedding": {
            name
            for name in checkpoint_names
            if name.startswith("model.language_model.embed_tokens.")
        },
        "decoder": {
            name
            for name in checkpoint_names
            if name.startswith(("model.language_model.layers.", "model.language_model.norm."))
            or name == "lm_head.weight"
        },
    }
    assert set().union(*categories.values()) == checkpoint_names
    assert sum(map(len, categories.values())) == len(checkpoint_names) == 1_177
    assert {name: len(values) for name, values in categories.items()} == {
        "acoustic_encoder": 276,
        "acoustic_decoder_unused": 276,
        "semantic_encoder": 276,
        "connectors": 10,
        "embedding": 1,
        "decoder": 338,
    }

    # A tiny canonical source proves the same router accepts current
    # Transformers names, while the index census proves every original name
    # is either routed or intentionally omitted.
    from mobius._configs import VibeVoiceASRConfig

    hf_config = _make_tiny_hf_config()
    config = VibeVoiceASRConfig.from_transformers(
        hf_config.text_config, parent_config=hf_config
    )
    module = VibeVoiceASRForConditionalGeneration(config)
    routed = module.preprocess_weights({name: torch.empty(0) for name in checkpoint_names})
    assert len(routed) == len(checkpoint_names) - len(categories["acoustic_decoder_unused"])
    assert all(
        name.startswith(
            ("acoustic_encoder.", "semantic_encoder.", "connectors.", "embedding.", "decoder.")
        )
        for name in routed
    )


# ---------------------------------------------------------------------------
# Causal LM weight alignment
# ---------------------------------------------------------------------------
_CAUSAL_LM_PARAMS = _mark_xfail_if_filtering(ALL_CAUSAL_LM_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _CAUSAL_LM_PARAMS)
class TestCausalLMWeightAlignment:
    """Verify preprocess_weights() preserves all parameter names for causal LM models."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)


# ---------------------------------------------------------------------------
# Encoder-only weight alignment
# ---------------------------------------------------------------------------
# Encoder models that filter HF-only patterns in preprocess_weights().
_FILTERING_ENCODER_MODELS: set[str] = {
    # CLIP text: expects text_model.encoder.* HF format
    "clip_text_model",
    # ModernBert: expects model.layers.* HF format with renames
    "modernbert",
}

_ENCODER_PARAMS = _mark_xfail_if_filtering_set(ENCODER_CONFIGS, _FILTERING_ENCODER_MODELS)


@pytest.mark.parametrize("model_type,config_overrides", _ENCODER_PARAMS)
class TestEncoderWeightAlignment:
    """Verify preprocess_weights() preserves all parameter names for encoder models."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)


# ---------------------------------------------------------------------------
# Seq2Seq weight alignment
# ---------------------------------------------------------------------------
# T5-family models: preprocess_weights() renames from HF block.N.layer.K.*
# format which doesn't roundtrip ONNX-aligned decoder.block.N.* names.
_FILTERING_SEQ2SEQ_MODELS: set[str] = {
    "longt5",
    "mt5",
    "t5",
    "switch_transformers",
    "umt5",
}

_SEQ2SEQ_PARAMS = _mark_xfail_if_filtering_set(SEQ2SEQ_CONFIGS, _FILTERING_SEQ2SEQ_MODELS)


@pytest.mark.parametrize("model_type,config_overrides", _SEQ2SEQ_PARAMS)
class TestSeq2SeqWeightAlignment:
    """Verify preprocess_weights() preserves all parameter names for seq2seq models."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)


# ---------------------------------------------------------------------------
# Speech encoder-decoder weight alignment
# ---------------------------------------------------------------------------
_SPEECH_PARAMS = _mark_xfail_if_filtering(SPEECH_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _SPEECH_PARAMS)
class TestSpeechWeightAlignment:
    """Verify speech encoder-decoder preprocessors preserve aligned parameters."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)


# ---------------------------------------------------------------------------
# Vision model weight alignment
# ---------------------------------------------------------------------------
# Vision models: preprocess_weights() renames from HF encoder.layers.*
# format which doesn't roundtrip ONNX-aligned encoder.N.* names.
_FILTERING_VISION_MODELS: set[str] = {mt for mt, _, _ in VISION_CONFIGS}

_VISION_PARAMS = _mark_xfail_if_filtering_set(VISION_CONFIGS, _FILTERING_VISION_MODELS)


@pytest.mark.parametrize("model_type,config_overrides", _VISION_PARAMS)
class TestVisionWeightAlignment:
    """Verify preprocess_weights() preserves all parameter names for vision models."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)


@pytest.mark.parametrize(
    "model_type,config_overrides",
    [
        pytest.param(model_type, overrides, id=model_type)
        for model_type, overrides, _ in VL_CONFIGS
        if model_type == "nemotron_parse"
    ],
)
class TestNemotronParseWeightAlignment:
    """Verify the two-model package preserves every aligned parameter."""

    def test_identity_state_dict_roundtrip(
        self, model_type: str, config_overrides: dict
    ) -> None:
        _assert_identity_roundtrip(model_type, config_overrides)


def test_lfm2_vl_huggingface_weight_alignment() -> None:
    """Every three-model initializer is populated from the upstream key layout."""
    config = _base_config(**vl_overrides("lfm2_vl"))
    module = registry.get("lfm2_vl")(config)
    pkg = get_task(_default_task_for_model("lfm2_vl")).build(module, config)
    parameter_names = _collect_parameter_names(pkg)

    hf_state: dict[str, torch.Tensor] = {}
    for name in parameter_names:
        if name.startswith("vision_encoder.vision_tower."):
            suffix = name[len("vision_encoder.vision_tower.") :]
            suffix = suffix.replace(".mlp.up_proj.", ".mlp.fc1.")
            suffix = suffix.replace(".mlp.down_proj.", ".mlp.fc2.")
            hf_name = f"model.vision_tower.vision_model.{suffix}"
        elif name.startswith("vision_encoder.multi_modal_projector."):
            hf_name = f"model.multi_modal_projector.{name[len('vision_encoder.multi_modal_projector.') :]}"
        elif name in {
            "decoder.lm_head.weight",
            "decoder.model.embed_tokens.weight",
            "embedding.embed_tokens.weight",
        }:
            hf_name = "model.language_model.embed_tokens.weight"
        elif name.startswith("decoder.model."):
            suffix = name[len("decoder.model.") :]
            suffix = suffix.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
            suffix = suffix.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
            suffix = suffix.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")
            suffix = suffix.replace(".feed_forward.gate_proj.", ".feed_forward.w1.")
            suffix = suffix.replace(".feed_forward.up_proj.", ".feed_forward.w3.")
            suffix = suffix.replace(".feed_forward.down_proj.", ".feed_forward.w2.")
            hf_name = f"model.language_model.{suffix}"
        else:
            raise AssertionError(f"Unrecognized LFM2-VL parameter name: {name}")
        hf_state[hf_name] = torch.ones(1)

    aligned = module.preprocess_weights(hf_state)
    missing = parameter_names - set(aligned)
    assert not missing, f"LFM2-VL preprocessing missed: {sorted(missing)}"


def _lfm2moe_module():
    overrides = next(
        overrides for mt, overrides, _ in ALL_CAUSAL_LM_CONFIGS if mt == "lfm2_moe"
    )
    return registry.get("lfm2_moe")(
        _base_config(**{**overrides, "tie_word_embeddings": False})
    )


def test_lfm2moe_individual_expert_weight_alignment() -> None:
    """The published checkpoint's w1/w2/w3 experts map to the dedicated MoE graph."""
    module = _lfm2moe_module()
    state_dict = {
        "model.layers.1.feed_forward.experts.0.w1.weight": torch.ones(32, 64),
        "model.layers.1.feed_forward.experts.0.w2.weight": torch.ones(64, 32),
        "model.layers.1.feed_forward.experts.0.w3.weight": torch.ones(32, 64),
    }

    aligned = module.preprocess_weights(state_dict)

    assert set(aligned) == {
        "model.layers.1.feed_forward.experts.0.gate_proj.weight",
        "model.layers.1.feed_forward.experts.0.down_proj.weight",
        "model.layers.1.feed_forward.experts.0.up_proj.weight",
    }


def test_lfm2moe_fused_expert_weight_alignment() -> None:
    """Current Transformers fused expert tensors split into per-expert projections."""
    module = _lfm2moe_module()
    gate_up = torch.arange(4 * 64 * 64).reshape(4, 64, 64)
    down = torch.arange(4 * 64 * 32).reshape(4, 64, 32)
    state_dict = {
        "model.layers.1.feed_forward.experts.gate_up_proj.weight": gate_up,
        "model.layers.1.feed_forward.experts.down_proj.weight": down,
    }

    aligned = module.preprocess_weights(state_dict)

    assert len(aligned) == 12
    assert torch.equal(
        aligned["model.layers.1.feed_forward.experts.3.gate_proj.weight"],
        gate_up[3, :32],
    )
    assert torch.equal(
        aligned["model.layers.1.feed_forward.experts.3.up_proj.weight"],
        gate_up[3, 32:],
    )
    assert torch.equal(
        aligned["model.layers.1.feed_forward.experts.3.down_proj.weight"],
        down[3],
    )


# ---------------------------------------------------------------------------
# Detection model weight alignment
# ---------------------------------------------------------------------------
# Detection models: preprocess_weights() renames from HF encoder.layers.*
# format which doesn't roundtrip ONNX-aligned encoder.N.* names.
_FILTERING_DETECTION_MODELS: set[str] = {mt for mt, _, _ in DETECTION_CONFIGS}

_DETECTION_PARAMS = _mark_xfail_if_filtering_set(
    DETECTION_CONFIGS, _FILTERING_DETECTION_MODELS
)


@pytest.mark.parametrize("model_type,config_overrides", _DETECTION_PARAMS)
class TestDetectionWeightAlignment:
    """Verify preprocess_weights() preserves all parameter names for detection models."""

    def test_identity_state_dict_roundtrip(self, model_type: str, config_overrides: dict):
        _assert_identity_roundtrip(model_type, config_overrides)


def test_mage_vl_huggingface_checkpoint_alignment():
    """Every Mage-VL package parameter is populated from its exact HF key."""
    config_overrides = next(overrides for mt, overrides, _ in VL_CONFIGS if mt == "mage_vl")
    config = _base_config(**config_overrides)
    module = registry.get("mage_vl")(config)
    pkg = get_task("mage-vl").build(module, config)
    parameter_names = _collect_parameter_names(pkg)

    hf_state: dict[str, torch.Tensor] = {}
    for name in parameter_names:
        if name == "embedding.embed_tokens.weight":
            continue
        if name.startswith("decoder.model.language_model."):
            hf_name = name[len("decoder.") :]
        elif name.startswith("decoder.lm_head."):
            hf_name = name[len("decoder.") :]
        elif name.startswith("vision_encoder.visual."):
            hf_name = f"model.{name[len('vision_encoder.') :]}"
        else:
            raise AssertionError(f"Unrecognized Mage-VL parameter name: {name}")
        hf_state[hf_name] = torch.ones(1)

    # The shared source embedding must populate both decoder and embedding models.
    hf_state["model.language_model.embed_tokens.weight"] = torch.ones(1)
    aligned = module.preprocess_weights(hf_state)
    assert set(aligned) == parameter_names


def test_glm_ocr_huggingface_checkpoint_alignment():
    """Every exported GLM-OCR parameter maps to its checkpoint tensor name."""
    config_overrides = next(overrides for mt, overrides, _ in VL_CONFIGS if mt == "glm_ocr")
    config = _base_config(**config_overrides)
    module = registry.get("glm_ocr")(config)
    pkg = get_task("glm-ocr").build(module, config)
    parameter_names = _collect_parameter_names(pkg)

    hf_state: dict[str, torch.Tensor] = {}
    for name in parameter_names:
        if name == "embedding.embed_tokens.weight":
            hf_name = "model.language_model.embed_tokens.weight"
        elif name.startswith("decoder.model."):
            hf_name = f"model.language_model.{name.removeprefix('decoder.model.')}"
        elif name == "decoder.lm_head.weight":
            hf_name = "lm_head.weight"
        elif name.startswith("vision_encoder.visual."):
            hf_name = f"model.{name.removeprefix('vision_encoder.')}"
        else:
            raise AssertionError(f"Unrecognized GLM-OCR parameter name: {name}")
        hf_state[hf_name] = torch.ones(1)

    # The real checkpoint has one auxiliary predictor layer after the decoder.
    hf_state["model.language_model.layers.2.input_layernorm.weight"] = torch.ones(1)
    aligned = module.preprocess_weights(hf_state)
    assert set(aligned) == parameter_names
