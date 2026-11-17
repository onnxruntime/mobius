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


def test_minicpm_vision_defaults_explicit_none_kernels(loaded_vision_hooks):
    """Explicit None merger kernels fall back to MiniCPM's 2x2 defaults."""
    vision_config = _FakeHFConfig(
        model_type="minicpmv4_6_vision",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=56,
        patch_size=14,
        layer_norm_eps=1e-6,
        num_channels=3,
        window_kernel_size=None,
    )
    parent_config = _FakeHFConfig(
        model_type="minicpmv4_6",
        vision_config=vision_config,
        image_token_id=250,
        merge_kernel_size=None,
    )
    text_config = _FakeHFConfig(model_type="qwen3_5_text")

    out = _extractors.extract_vision_config(
        text_config,
        parent_config,
        "qwen3_5_text",
    )

    assert out["vision"].window_kernel_size == (2, 2)
    assert out["vision"].merge_kernel_size == (2, 2)


# ---------------------------------------------------------------------------
# Gemma 3n hooks (MobileNet-V5 vision tower + USM Conformer audio tower)
# ---------------------------------------------------------------------------


def _gemma3n_hf_config(**overrides):
    """Stand-in for the outer HF ``Gemma3nConfig``, with E4B's real values."""
    vision = _FakeHFConfig(
        model_type="gemma3n_vision",
        architecture="mobilenetv5_300m_enc",
        hidden_size=2048,
        do_pooling=False,
        rms_norm_eps=1e-6,
        vocab_offset=262144,
        vocab_size=128,
    )
    audio = _FakeHFConfig(
        model_type="gemma3n_audio",
        hidden_size=1536,
        conf_num_hidden_layers=12,
        conf_num_attention_heads=8,
        conf_attention_chunk_size=12,
        conf_attention_context_left=13,
        conf_attention_context_right=0,
        conf_attention_logit_cap=50.0,
        conf_conv_kernel_size=5,
        conf_reduction_factor=4,
        conf_residual_weight=0.5,
        input_feat_size=128,
        sscp_conv_channel_size=[128, 32],
        sscp_conv_kernel_size=[[3, 3], [3, 3]],
        sscp_conv_stride_size=[[2, 2], [2, 2]],
        sscp_conv_group_norm_eps=1e-3,
        gradient_clipping=1e10,
        rms_norm_eps=1e-6,
        vocab_offset=262272,
        vocab_size=128,
    )
    fields = dict(
        vision_config=vision,
        audio_config=audio,
        image_token_id=262145,
        audio_token_id=262273,
        vision_soft_tokens_per_image=256,
        audio_soft_tokens_per_image=188,
    )
    fields.update(overrides)
    return _FakeHFConfig(model_type="gemma3n", **fields)


def test_gemma3n_vision_maps_mobilenet_fields(loaded_vision_hooks):
    """The tower is MobileNet-V5, so the timm spec name and 768x768 are pinned."""
    out = _extractors.extract_vision_config(_gemma3n_hf_config(), None, "gemma3n")
    vision = out["vision"]

    assert vision.model_type == "gemma3n_vision"
    assert vision.architecture == "mobilenetv5_300m_enc"
    assert vision.hidden_size == 2048
    # HF ships no image_size; MobileNet-V5 has no dynamic-resolution path and
    # the processor resizes to a fixed 768x768 (16x16 grid = 256 soft tokens).
    assert vision.image_size == 768
    # Gemma3n needs the spatial feature map, not a pooled vector.
    assert vision.do_pooling is False
    assert (vision.vocab_offset, vision.vocab_size) == (262144, 128)
    assert vision.image_token_id == 262145
    assert vision.mm_tokens_per_image == 256


def test_gemma3n_vision_fires_from_text_sub_config(loaded_vision_hooks):
    """build() may resolve to the text sub-config; the parent still drives the hook."""
    parent = _gemma3n_hf_config()
    text = _FakeHFConfig(model_type="gemma3n_text")

    out = _extractors.extract_vision_config(text, parent, "gemma3n_text")

    assert out["vision"].architecture == "mobilenetv5_300m_enc"


def test_gemma3n_vision_does_not_fire_for_unrelated_model(loaded_vision_hooks):
    """A gemma3n-shaped vision_config must not leak into another model type.

    Regression guard: reading the parent model_type off the
    ``parent_config or config`` composite (rather than ``parent_config``)
    made the hook fire for *any* dispatched model_type as long as the config
    itself was a gemma3n one.
    """
    out = _extractors.extract_vision_config(_gemma3n_hf_config(), None, "llama")

    assert out.get("vision") is None or out["vision"].architecture is None, (
        f"gemma3n vision hook fired for model_type='llama': {out}"
    )


def test_gemma3n_audio_maps_usm_conformer_fields(loaded_hooks):
    """The USM field names pass through verbatim onto Gemma3nAudioConfig."""
    from mobius._configs._sub_configs import Gemma3nAudioConfig

    out = _extractors.extract_audio_config(_gemma3n_hf_config(), None, "gemma3n")
    audio = out["audio"]

    assert isinstance(audio, Gemma3nAudioConfig)
    assert audio.hidden_size == 1536
    assert audio.conf_num_hidden_layers == 12
    assert audio.conf_num_attention_heads == 8
    assert audio.conf_attention_chunk_size == 12
    assert audio.conf_attention_context_left == 13
    assert audio.conf_attention_context_right == 0
    assert audio.conf_attention_logit_cap == pytest.approx(50.0)
    assert audio.conf_conv_kernel_size == 5
    assert audio.conf_reduction_factor == 4
    assert audio.conf_residual_weight == pytest.approx(0.5)
    assert audio.input_feat_size == 128
    assert audio.sscp_conv_channel_size == [128, 32]
    assert audio.sscp_conv_kernel_size == [[3, 3], [3, 3]]
    assert audio.sscp_conv_stride_size == [[2, 2], [2, 2]]
    assert audio.sscp_conv_group_norm_eps == pytest.approx(1e-3)
    assert (audio.vocab_offset, audio.vocab_size) == (262272, 128)
    assert audio.audio_token_id == 262273


def test_gemma3n_audio_fires_from_text_sub_config(loaded_hooks):
    """Same parent-resolution path as the vision hook."""
    parent = _gemma3n_hf_config()
    text = _FakeHFConfig(model_type="gemma3n_text")

    out = _extractors.extract_audio_config(text, parent, "gemma3n_text")

    assert out["audio"].conf_num_hidden_layers == 12


def test_gemma3n_audio_absent_without_audio_config(loaded_hooks):
    """Audio is optional: no audio_config means no audio sub-config."""
    cfg = _gemma3n_hf_config(audio_config=None)

    assert _extractors.extract_audio_config(cfg, None, "gemma3n") == {}


def test_gemma3n_audio_does_not_fire_for_unrelated_model(loaded_hooks):
    """A gemma3n-shaped audio_config must not leak into another model type.

    Same parent-resolution regression guard as the vision hook above.
    """
    from mobius._configs._sub_configs import Gemma3nAudioConfig

    out = _extractors.extract_audio_config(_gemma3n_hf_config(), None, "llama")

    assert not isinstance(out.get("audio"), Gemma3nAudioConfig), (
        f"gemma3n audio hook fired for model_type='llama': {out}"
    )
