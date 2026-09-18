# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import onnx_ir as ir
import pytest
import torch
from _test_configs import CAUSAL_LM_CONFIGS, _base_config
from synthetic_parity_test import (
    _build_onnx_model,
    _create_hf_config,
    _create_hf_model,
    _fill_random_weights,
)

from mobius._configs import FalconH1Config
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations._weight_loading import apply_weights


def _tiny_overrides() -> dict:
    return next(
        dict(overrides)
        for model_type, overrides, _representative in CAUSAL_LM_CONFIGS
        if model_type == "falcon_h1"
    )


def _linear_attention_state(
    states: torch.Tensor | Mapping[int, torch.Tensor | None], state_name: str
) -> torch.Tensor:
    if isinstance(states, torch.Tensor):
        return states
    elif isinstance(states, Mapping):
        assert set(states) == {0}, (
            f"Falcon-H1 expected one {state_name} tensor at state index 0, "
            f"got indices {sorted(states)}"
        )
        state = states[0]
        assert state is not None, (
            f"Falcon-H1 expected {state_name} at state index 0 to be a tensor, got None"
        )
        return state
    else:
        raise AssertionError(  # noqa: TRY004 - unexpected test data is an assertion failure
            f"Falcon-H1 received unexpected {state_name} container type: "
            f"{type(states).__name__}"
        )


def test_falcon_h1_rejects_invalid_attention_and_ssm_geometry() -> None:
    base = _tiny_overrides()
    with pytest.raises(ValueError, match="hidden_size must equal"):
        _base_config(**(base | {"head_dim": 15}))
    with pytest.raises(ValueError, match="mamba_d_head"):
        _base_config(**(base | {"mamba_d_head": 15}))
    with pytest.raises(ValueError, match="mamba_n_groups must divide both"):
        _base_config(
            **(
                base
                | {
                    "mamba_d_ssm": 24,
                    "mamba_n_heads": 3,
                    "mamba_d_head": 8,
                    "mamba_n_groups": 2,
                }
            )
        )
    with pytest.raises(ValueError, match="exactly five"):
        _base_config(**(base | {"ssm_multipliers": [1.0] * 4}))
    with pytest.raises(ValueError, match="ordered and non-negative"):
        _base_config(**(base | {"time_step_limit": [1.0, 0.5]}))


def test_falcon_h1_task_rejects_static_cache() -> None:
    from mobius.tasks import FalconH1CausalLMTask

    with pytest.raises(ValueError, match="four-state ABI"):
        FalconH1CausalLMTask(static_cache=True)


def test_falcon_h1_transformers_config_extracts_exact_controls() -> None:
    from transformers import FalconH1Config as HFFalconH1Config

    hf_config = HFFalconH1Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        mamba_d_ssm=64,
        mamba_n_heads=4,
        mamba_d_head=16,
        mamba_n_groups=2,
        mamba_d_state=8,
        mamba_d_conv=4,
        attention_bias=True,
        mlp_bias=True,
        mamba_proj_bias=True,
        projectors_bias=True,
        mamba_rms_norm=True,
        mamba_norm_before_gate=False,
        mlp_multipliers=[0.75, 1.25],
        ssm_multipliers=[0.5, 0.75, 1.0, 1.25, 1.5],
    )
    config = FalconH1Config.from_transformers(hf_config)
    assert config.attn_qkv_bias and config.attn_o_bias
    assert config.mlp_bias and config.mamba_proj_bias and config.projectors_bias
    assert config.mamba_rms_norm and not config.mamba_norm_before_gate
    assert config.mlp_multipliers == (0.75, 1.25)
    assert config.ssm_multipliers == (0.5, 0.75, 1.0, 1.25, 1.5)


def test_falcon_h1_explicit_projection_biases_override_global_fallback() -> None:
    config = _base_config(
        **(
            _tiny_overrides()
            | {"attention_bias": False, "attn_qkv_bias": True, "attn_o_bias": True}
        )
    )
    assert isinstance(config, FalconH1Config)
    assert config.attn_qkv_bias is True
    assert config.attn_o_bias is True


def test_falcon_h1_global_attention_bias_remains_a_compatibility_fallback() -> None:
    config = _base_config(
        **(
            _tiny_overrides()
            | {"attention_bias": True, "attn_qkv_bias": False, "attn_o_bias": False}
        )
    )
    assert isinstance(config, FalconH1Config)
    assert config.attn_qkv_bias is True
    assert config.attn_o_bias is True


def test_falcon_h1_prefill_decode_logits_and_four_states_match_transformers() -> None:
    overrides = _tiny_overrides()
    config = _base_config(**overrides)
    assert isinstance(config, FalconH1Config)
    config.dtype = ir.DataType.FLOAT
    module, package = _build_onnx_model("falcon_h1", config)
    hf_model = _create_hf_model(
        "falcon_h1",
        _create_hf_config("falcon_h1", overrides),
        seed=42,
    )
    weights = module.preprocess_weights(dict(hf_model.state_dict()))
    apply_weights(package["model"], weights)
    _fill_random_weights(package["model"], np.random.default_rng(42))

    input_ids = np.asarray([[1, 2, 3]], np.int64)
    attention_mask = np.ones((1, 3), np.int64)
    position_ids = np.asarray([[0, 1, 2]], np.int64)
    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for layer in range(config.num_hidden_layers):
        feeds.update(
            {
                f"past_key_values.{layer}.key": np.zeros(
                    (1, config.num_key_value_heads, 0, config.head_dim),
                    np.float32,
                ),
                f"past_key_values.{layer}.value": np.zeros(
                    (1, config.num_key_value_heads, 0, config.head_dim),
                    np.float32,
                ),
                f"past_key_values.{layer}.conv_state": np.zeros(
                    (
                        1,
                        config.mamba_d_ssm + 2 * config.mamba_n_groups * config.mamba_d_state,
                        config.mamba_d_conv - 1,
                    ),
                    np.float32,
                ),
                f"past_key_values.{layer}.ssm_state": np.zeros(
                    (
                        1,
                        config.mamba_n_heads,
                        config.mamba_d_state,
                        config.mamba_d_head,
                    ),
                    np.float32,
                ),
            }
        )

    with torch.no_grad():
        hf_prefill = hf_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            use_cache=True,
        )

    session = OnnxModelSession(package["model"])
    try:
        ort_prefill = session.run(feeds)
        np.testing.assert_allclose(
            ort_prefill["logits"],
            hf_prefill.logits.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
        for layer, hf_state in enumerate(hf_prefill.past_key_values.layers):
            hf_conv_state = _linear_attention_state(hf_state.conv_states, "convolution state")
            hf_recurrent_state = _linear_attention_state(
                hf_state.recurrent_states, "recurrent state"
            )
            np.testing.assert_allclose(
                ort_prefill[f"present.{layer}.key"],
                hf_state.keys.numpy(),
                rtol=1e-4,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                ort_prefill[f"present.{layer}.value"],
                hf_state.values.numpy(),
                rtol=1e-4,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                ort_prefill[f"present.{layer}.conv_state"],
                hf_conv_state[:, :, -(config.mamba_d_conv - 1) :].numpy(),
                rtol=1e-4,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                ort_prefill[f"present.{layer}.ssm_state"],
                hf_recurrent_state.transpose(2, 3).numpy(),
                rtol=1e-3,
                atol=1e-3,
            )

        decode_ids = np.asarray([[4]], np.int64)
        decode_mask = np.ones((1, 4), np.int64)
        decode_position = np.asarray([[3]], np.int64)
        decode_feeds = {
            "input_ids": decode_ids,
            "attention_mask": decode_mask,
            "position_ids": decode_position,
            **{
                name.replace("present.", "past_key_values."): value
                for name, value in ort_prefill.items()
                if name.startswith("present.")
            },
        }
        with torch.no_grad():
            hf_decode = hf_model(
                input_ids=torch.from_numpy(decode_ids),
                attention_mask=torch.from_numpy(decode_mask),
                position_ids=torch.from_numpy(decode_position),
                past_key_values=hf_prefill.past_key_values,
                use_cache=True,
            )
        ort_decode = session.run(decode_feeds)
        np.testing.assert_allclose(
            ort_decode["logits"],
            hf_decode.logits.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
    finally:
        session.close()


@pytest.mark.integration
@pytest.mark.integration_slow
def test_falcon_h1_pinned_tiny_checkpoint_full_logits() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mobius import build
    from mobius._configs import FalconH1Config
    from mobius.integrations.transformers._config_resolver import _config_from_hf

    model_id = "tiiuae/Falcon-H1-Tiny-90M-Base"
    revision = "7994372e93b62822ae25f8bfb19f653649cea3a3"
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
    ).float()
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    hf_config = hf_model.config
    config = _config_from_hf(hf_config)
    assert isinstance(config, FalconH1Config)
    package = build(
        model_id,
        revision=revision,
        dtype="f32",
        load_weights=True,
        trust_remote_code=False,
    )

    encoded = tokenizer("Here is my poem:", return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
    feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    conv_dim = config.mamba_d_ssm + 2 * config.mamba_n_groups * config.mamba_d_state
    for layer in range(config.num_hidden_layers):
        feeds.update(
            {
                f"past_key_values.{layer}.key": np.zeros(
                    (1, config.num_key_value_heads, 0, config.head_dim),
                    np.float32,
                ),
                f"past_key_values.{layer}.value": np.zeros(
                    (1, config.num_key_value_heads, 0, config.head_dim),
                    np.float32,
                ),
                f"past_key_values.{layer}.conv_state": np.zeros(
                    (1, conv_dim, config.mamba_d_conv - 1),
                    np.float32,
                ),
                f"past_key_values.{layer}.ssm_state": np.zeros(
                    (
                        1,
                        config.mamba_n_heads,
                        config.mamba_d_state,
                        config.mamba_d_head,
                    ),
                    np.float32,
                ),
            }
        )

    with torch.no_grad():
        expected = hf_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            use_cache=False,
        ).logits.numpy()
    session = OnnxModelSession(package["model"])
    try:
        actual = session.run(feeds)["logits"]
    finally:
        session.close()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
