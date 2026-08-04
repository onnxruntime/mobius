# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the extractor-hook registry.

Two concerns are covered:

1. The decorator/dispatcher machinery itself: bare ``@register_*_hook``,
   parameterised ``@register_*_hook("type_a", "type_b")``, short-circuit
   semantics, and the no-vision-fields fallback.

2. Cross-contamination guards: every per-model hook registered in
   :mod:`mobius._configs.per_model` is verified to *not* fire for an
   unrelated model. Filtered hooks must drop out via the dispatcher;
   bare hooks must short-circuit themselves on the unrelated config.
"""

from __future__ import annotations

import pytest

from mobius._configs import _extractors


class _FakeHFConfig:
    """Minimal HuggingFace-config stand-in.

    Provides attribute-style access to any kwargs passed to the
    constructor, plus a sentinel ``model_type`` so hooks that check it
    behave the way they would against a real ``PretrainedConfig``.
    """

    def __init__(self, model_type: str = "_unrelated_", **kwargs):
        self.model_type = model_type
        self.__dict__.update(kwargs)


# ---------------------------------------------------------------------------
# Decorator + dispatcher mechanism
# ---------------------------------------------------------------------------


def _isolated_registry(monkeypatch):
    """Swap the module-level registries for empty lists for one test."""
    audio: list = []
    vision: list = []
    monkeypatch.setattr(_extractors, "_AUDIO_HOOKS", audio)
    monkeypatch.setattr(_extractors, "_VISION_HOOKS", vision)
    # Re-bind decorators to the empty registries so newly registered hooks
    # actually land in our copies instead of the module-level ones.
    monkeypatch.setattr(_extractors, "register_audio_hook", _extractors._make_register(audio))
    monkeypatch.setattr(
        _extractors, "register_vision_hook", _extractors._make_register(vision)
    )
    return audio, vision


def test_bare_decorator_runs_for_every_model_type(monkeypatch):
    """Bare ``@register_audio_hook`` registers a hook that runs for every model_type.

    Reserved for hooks that need to inspect ``parent_config`` to decide
    whether to fire (e.g. ``_gemma4_audio`` fires when the parent is
    gemma4 even when model_type points at a sub-config).
    """
    _audio, _ = _isolated_registry(monkeypatch)
    seen: list[str] = []

    @_extractors.register_audio_hook
    def _hook(config, parent, model_type, fields):
        seen.append(model_type)
        return None

    assert _extractors._AUDIO_HOOKS == [(None, _hook)]  # noqa: SIM300
    _extractors.extract_audio_config(_FakeHFConfig(), None, "anything")
    _extractors.extract_audio_config(_FakeHFConfig(), None, "another")
    assert seen == ["anything", "another"]


def test_filtered_decorator_skips_unrelated_model_types(monkeypatch):
    _audio, _ = _isolated_registry(monkeypatch)
    fired_for: list[str] = []

    @_extractors.register_audio_hook("phi4mm", "qwen3_asr")
    def _hook(config, parent, model_type, fields):
        fired_for.append(model_type)
        fields["d_model"] = 1
        return None

    # Unrelated → filter skips
    _extractors.extract_audio_config(_FakeHFConfig(), None, "llama")
    assert fired_for == []

    # Match → fires
    _extractors.extract_audio_config(_FakeHFConfig(), None, "phi4mm")
    assert fired_for == ["phi4mm"]

    fired_for.clear()
    _extractors.extract_audio_config(_FakeHFConfig(), None, "qwen3_asr")
    assert fired_for == ["qwen3_asr"]


def test_short_circuit_dict_skips_remaining_hooks(monkeypatch):
    _audio, _ = _isolated_registry(monkeypatch)
    after_short_circuit_ran = False

    @_extractors.register_audio_hook
    def _early(config, parent, model_type, fields):
        return {"audio": "sentinel"}  # short-circuit

    @_extractors.register_audio_hook
    def _later(config, parent, model_type, fields):
        nonlocal after_short_circuit_ran
        after_short_circuit_ran = True
        return None

    result = _extractors.extract_audio_config(_FakeHFConfig(), None, "x")
    assert result == {"audio": "sentinel"}
    assert after_short_circuit_ran is False


def test_no_contributions_returns_empty_dict(monkeypatch):
    _isolated_registry(monkeypatch)

    @_extractors.register_audio_hook
    def _hook(config, parent, model_type, fields):
        return None

    assert _extractors.extract_audio_config(_FakeHFConfig(), None, "x") == {}


def test_defaults_run_before_per_model_hooks(monkeypatch):
    """The explicit ``apply_*_defaults`` first pass runs before any hook.

    Regression guard for the order-of-execution invariant. Per-model
    hooks can observe and override fields set by the defaults.
    """
    _isolated_registry(monkeypatch)
    default_observed: list[bool] = []

    @_extractors.register_vision_hook("phi4mm")
    def _override(config, parent, model_type, fields):
        # apply_vision_defaults wrote hidden_size first (from the
        # vision_config sub-dict); the hook can see it then override.
        default_observed.append("hidden_size" in fields)
        fields["image_token_id"] = 999
        return None

    cfg = _FakeHFConfig(vision_config={"hidden_size": 768})
    out = _extractors.extract_vision_config(cfg, None, "phi4mm")
    # Default ran first and populated hidden_size from vision_config.
    assert default_observed == [True]
    # Per-model override took precedence on the shared field.
    assert out.get("image_token_id") == 999


# ---------------------------------------------------------------------------
# Per-model cross-contamination guard
# ---------------------------------------------------------------------------


def _import_per_model():
    """Force-import the per-model package so its hooks are registered."""
    from mobius._configs import per_model  # noqa: F401  side-effect import


@pytest.fixture
def loaded_hooks():
    _import_per_model()
    return list(_extractors._AUDIO_HOOKS)


def test_filtered_hooks_have_concrete_model_type_filters(loaded_hooks):
    """At least one ``@register_audio_hook("x", ...)`` registration must exist.

    Sanity check that the parameterised decorator form is in use — every
    per-model audio hook that can be filtered by model_type alone should
    prefer it over bare ``@register_audio_hook`` + an internal
    ``if model_type != "x":`` guard. (Bare form is reserved for hooks
    whose firing condition depends on ``parent_config`` etc.)
    """
    has_filter = any(filter_set is not None for filter_set, _ in loaded_hooks)
    assert has_filter, (
        'No filtered audio hooks registered — register_audio_hook("type") '
        "should be preferred over bare @register_audio_hook for model-specific hooks."
    )


def test_audio_hooks_do_not_contribute_for_unrelated_models(loaded_hooks):
    """Plain text model_types must not trigger any audio hook.

    Runs ``extract_audio_config`` against a deliberately empty config for a
    plain text model_type. No hook should produce an ``audio`` sub-config
    (since none of phi4mm/sensevoice/qwen3-asr/gemma4 indicators are set).
    """
    result = _extractors.extract_audio_config(_FakeHFConfig(), None, "llama")
    assert result == {}, f"unrelated model_type 'llama' produced audio output: {result}"

    # Same for other plain text architectures.
    for plain_model_type in ("qwen2", "gpt2", "bloom", "mamba"):
        result = _extractors.extract_audio_config(_FakeHFConfig(), None, plain_model_type)
        assert result == {}, (
            f"unrelated model_type {plain_model_type!r} produced audio output: {result}"
        )


def test_phi4mm_filter_only_fires_for_phi4mm(loaded_hooks):
    # Build a config that *would* trigger phi4mm's body if the filter wasn't
    # applied: it has an audio_config attribute with audio_token_id set.
    cfg = _FakeHFConfig(audio_config={"audio_token_id": 42})

    # phi4mm → fires
    out = _extractors.extract_audio_config(cfg, None, "phi4mm")
    assert "audio" in out
    assert out["audio"].token_id == 42

    # llama with the same shape → must NOT pick up phi4mm fields
    out = _extractors.extract_audio_config(cfg, None, "llama")
    assert out == {}, f"phi4mm hook fired for model_type='llama' (cross-contamination): {out}"


def test_sensevoice_filter_only_fires_for_sensevoice(loaded_hooks):
    cfg = _FakeHFConfig(
        encoder_conf={
            "output_size": 512,
            "attention_heads": 4,
            "num_blocks": 50,
            "tp_blocks": 20,
            "linear_units": 2048,
            "kernel_size": 11,
        },
        frontend_conf={"n_mels": 80},
        input_size=560,
    )

    out = _extractors.extract_audio_config(cfg, None, "sensevoice")
    assert out["audio"].attention_dim == 512

    # llama with sensevoice-shaped fields → no audio
    out = _extractors.extract_audio_config(cfg, None, "llama")
    assert out == {}, (
        f"sensevoice hook fired for model_type='llama' (cross-contamination): {out}"
    )


# ---------------------------------------------------------------------------
# Vision hooks
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_vision_hooks():
    _import_per_model()
    return list(_extractors._VISION_HOOKS)


def test_vision_hooks_do_not_contribute_for_unrelated_models(loaded_vision_hooks):
    """Plain text model_types must not trigger any vision hook."""
    for plain_model_type in ("llama", "qwen2", "gpt2", "bloom", "mamba"):
        result = _extractors.extract_vision_config(_FakeHFConfig(), None, plain_model_type)
        assert result == {}, (
            f"unrelated model_type {plain_model_type!r} produced vision output: {result}"
        )


def test_phi4mm_vision_filter_only_fires_for_phi4mm(loaded_vision_hooks):
    # Bare config without vision_config: phi4mm hook should still fire (its
    # body hard-codes the SigLIP dims) for "phi4mm", but NOT for "llama".
    cfg = _FakeHFConfig(special_image_token_id=200010)
    out = _extractors.extract_vision_config(cfg, None, "phi4mm")
    assert "vision" in out and out["vision"].hidden_size == 1152

    out = _extractors.extract_vision_config(cfg, None, "llama")
    assert out == {}, (
        f"phi4mm vision hook fired for model_type='llama' (cross-contamination): {out}"
    )


def test_hunyuan_vl_mot_filter_only_fires_for_that_model(loaded_vision_hooks):
    cfg = _FakeHFConfig(mask_init_id=12)
    out = _extractors.extract_vision_config(cfg, None, "hunyuan_vl_mot")
    assert out["vision"].hidden_size == 1152

    # llama: no vision_config + no hardcoded path → empty
    out = _extractors.extract_vision_config(cfg, None, "llama")
    assert out == {}


def test_phi3_v_vision_parses_img_processor(loaded_vision_hooks):
    """Phi-3.5-Vision reads image_dim_out + layer_idx from the img_processor dict."""
    cfg = _FakeHFConfig(
        img_processor={
            "name": "clip_vision_model",
            "model_name": "openai/clip-vit-large-patch14-336",
            "image_dim_out": 1024,
            "num_img_tokens": 144,
        }
    )
    out = _extractors.extract_vision_config(cfg, None, "phi3_v")
    vision = out["vision"]
    # CLIP ViT-L/14-336 geometry.
    assert vision.hidden_size == 1024
    assert vision.intermediate_size == 4096
    assert vision.num_hidden_layers == 24
    assert vision.num_attention_heads == 16
    assert vision.image_size == 336
    assert vision.patch_size == 14
    assert vision.hidden_act == "quick_gelu"
    # layer_idx defaults to -2 (feature extraction skips the last layer).
    assert vision.feature_layer == -2


def test_phi3_v_vision_respects_explicit_layer_idx(loaded_vision_hooks):
    cfg = _FakeHFConfig(
        img_processor={"name": "clip_vision_model", "image_dim_out": 1024, "layer_idx": -1}
    )
    out = _extractors.extract_vision_config(cfg, None, "phi3_v")
    assert out["vision"].feature_layer == -1


def test_phi3_v_vision_does_not_fire_for_unrelated_model(loaded_vision_hooks):
    cfg = _FakeHFConfig(img_processor={"image_dim_out": 1024})
    out = _extractors.extract_vision_config(cfg, None, "llama")
    assert out == {}


def test_phi4mm_image_token_id_survives_default_hook(loaded_vision_hooks):
    """Per-model image_token_id must not be clobbered by _vision_default.

    Regression guard for hook ordering: _vision_default runs the same field
    assignment as per-model hooks, and earlier versions overwrote any value
    set by an earlier per-model hook (or by a later one whose registration
    order put _vision_default after it).
    """
    cfg = _FakeHFConfig(special_image_token_id=200010)
    out = _extractors.extract_vision_config(cfg, None, "phi4mm")
    assert out["vision"].image_token_id == 200010
    assert out.get("image_token_id") == 200010


def test_hunyuan_vl_mot_image_token_id_survives_default_hook(loaded_vision_hooks):
    """Same regression guard as phi4mm but for hunyuan_vl_mot."""
    cfg = _FakeHFConfig(mask_init_id=12)
    out = _extractors.extract_vision_config(cfg, None, "hunyuan_vl_mot")
    assert out["vision"].image_token_id == 12
    assert out.get("image_token_id") == 12
