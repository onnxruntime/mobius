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
    _audio, _ = _isolated_registry(monkeypatch)
    seen: list[str] = []

    @_extractors.register_audio_hook
    def _hook(config, parent, model_type, fields):
        seen.append(model_type)
        return None

    assert [(None, _hook)] == _extractors._AUDIO_HOOKS
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
    r"""At least one ``@register_audio_hook("x", ...)`` registration must exist.

    Sanity check that the parameterised decorator form is actually in use —
    every model-specific hook should prefer it over bare ``@register_audio_hook``
    + an internal ``if model_type != "x":`` guard.
    """
    seen_filter_sets = [filter_set for filter_set, _ in loaded_hooks if filter_set is not None]
    # At least one parameterised registration (sanity check the mechanism is in use).
    assert any(seen_filter_sets), (
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
