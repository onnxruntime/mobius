# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import torch

from mobius import build_from_module
from mobius._configs import Plamo2Config, QuantizationConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.plamo2 import Plamo2ForCausalLM
from mobius.tasks import Plamo2CausalLMTask


def _config() -> Plamo2Config:
    return Plamo2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=32,
        rope_type="default",
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_head_counts=(0, 4),
        attention_kv_head_counts=(0, 2),
        mamba_num_heads=4,
        mamba_d_state=4,
        mamba_d_conv=4,
        mamba_dt_rank=8,
        attention_window_size=32,
        tie_word_embeddings=True,
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
    )


def _fill_weights(model: ir.Model) -> None:
    rng = np.random.default_rng(7)
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = [dim if isinstance(dim, int) else 1 for dim in initializer.shape]
        values = (rng.standard_normal(shape) * 0.03).astype(initializer.dtype.numpy())
        if "norm" in initializer.name:
            values += 1.0
        initializer.const_value = ir.Tensor(values)


def _empty_states(config: Plamo2Config, batch: int) -> dict[str, np.ndarray]:
    return {
        "past_key_values.0.conv_state": np.zeros(
            (batch, config.mamba_inner_size, config.mamba_d_conv - 1), np.float32
        ),
        "past_key_values.0.ssm_state": np.zeros(
            (
                batch,
                config.mamba_num_heads,
                config.mamba_head_dim,
                config.mamba_d_state,
            ),
            np.float32,
        ),
        "past_key_values.1.key": np.zeros(
            (batch, config.num_key_value_heads, 0, config.head_dim), np.float32
        ),
        "past_key_values.1.value": np.zeros(
            (batch, config.num_key_value_heads, 0, config.head_dim), np.float32
        ),
    }


def _next_states(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name.replace("present.", "past_key_values."): value
        for name, value in outputs.items()
        if name.startswith("present.")
    }


def _rms_norm(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + 1e-6) * weight


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


def _mlp_reference(
    hidden: np.ndarray,
    gate_up_weight: np.ndarray,
    down_weight: np.ndarray,
) -> np.ndarray:
    gate, up = np.split(hidden @ gate_up_weight.T, 2, axis=-1)
    return (_silu(gate) * up) @ down_weight.T


def test_plamo2_mamba_matches_independent_reduced_reference() -> None:
    config = _config()
    model = build_from_module(
        Plamo2ForCausalLM(config),
        config,
        task=Plamo2CausalLMTask(),
    )["model"]
    rng = np.random.default_rng(11)
    values: dict[str, np.ndarray] = {}
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = tuple(dim if isinstance(dim, int) else 1 for dim in initializer.shape)
        value = np.zeros(shape, dtype=np.float32)
        if "norm" in initializer.name:
            value.fill(1.0)
        initializer.const_value = ir.Tensor(value)
        values[initializer.name] = value

    def assign(name: str, scale: float = 0.05) -> np.ndarray:
        value = (rng.standard_normal(values[name].shape) * scale).astype(np.float32)
        model.graph.initializers[name].const_value = ir.Tensor(value)
        values[name] = value
        return value

    assign("model.embed_tokens.weight")
    assign("model.layers.0.mixer.in_proj.weight")
    assign("model.layers.0.mixer.conv1d.weight")
    assign("model.layers.0.mixer.bcdt_proj.weight")
    assign("model.layers.0.mixer.dt_proj.weight")
    assign("model.layers.0.mixer.dt_bias")
    assign("model.layers.0.mixer.A_log")
    assign("model.layers.0.mixer.D")
    assign("model.layers.0.mixer.out_proj.weight")

    input_ids = np.array([[1, 2, 3]], dtype=np.int64)
    outputs = OnnxModelSession(model, device="cpu").run(
        {
            "input_ids": input_ids,
            "position_ids": np.array([[0, 1, 2]], dtype=np.int64),
            "attention_mask": np.ones((1, 3), dtype=np.int64),
            **_empty_states(config, 1),
        }
    )

    hidden = values["model.embed_tokens.weight"][input_ids]
    normalized = _rms_norm(hidden, values["model.layers.0.pre_mixer_norm.weight"])
    projected = normalized @ values["model.layers.0.mixer.in_proj.weight"].T
    projected = projected.reshape(
        1, input_ids.shape[1], config.mamba_num_heads, 2 * config.mamba_head_dim
    )
    gate, conv_input = np.split(projected, 2, axis=-1)
    gate = gate.reshape(1, input_ids.shape[1], config.mamba_inner_size)
    conv_input = conv_input.reshape(1, input_ids.shape[1], config.mamba_inner_size)
    history = np.concatenate(
        [
            np.zeros((1, config.mamba_inner_size, config.mamba_d_conv - 1), np.float32),
            np.transpose(conv_input, (0, 2, 1)),
        ],
        axis=-1,
    )
    conv_weight = values["model.layers.0.mixer.conv1d.weight"][:, 0, :]
    conv_output = np.stack(
        [
            np.sum(
                history[:, :, index : index + config.mamba_d_conv] * conv_weight[None, :, :],
                axis=-1,
            )
            for index in range(input_ids.shape[1])
        ],
        axis=1,
    )
    conv_output = _silu(conv_output)
    bcdt = conv_output @ values["model.layers.0.mixer.bcdt_proj.weight"].T
    b_mat, c_mat, dt_raw = np.split(
        bcdt,
        [config.mamba_d_state, 2 * config.mamba_d_state],
        axis=-1,
    )
    b_mat = _rms_norm(b_mat, values["model.layers.0.mixer.B_norm_weight"])
    c_mat = _rms_norm(c_mat, values["model.layers.0.mixer.C_norm_weight"])
    dt_raw = _rms_norm(dt_raw, values["model.layers.0.mixer.dt_norm_weight"])
    dt = np.logaddexp(
        0.0,
        dt_raw @ values["model.layers.0.mixer.dt_proj.weight"].T
        + values["model.layers.0.mixer.dt_bias"],
    )
    x_heads = conv_output.reshape(
        1, input_ids.shape[1], config.mamba_num_heads, config.mamba_head_dim
    )
    state = np.zeros(
        (1, config.mamba_num_heads, config.mamba_head_dim, config.mamba_d_state),
        np.float32,
    )
    scan_outputs = []
    for index in range(input_ids.shape[1]):
        decay = np.exp(
            dt[:, index, :, None, None]
            * -np.exp(values["model.layers.0.mixer.A_log"])[None, :, None, None]
        )
        update = (
            dt[:, index, :, None, None]
            * x_heads[:, index, :, :, None]
            * b_mat[:, index, None, None, :]
        )
        state = decay * state + update
        scan = np.sum(state * c_mat[:, index, None, None, :], axis=-1)
        scan += values["model.layers.0.mixer.D"][None, :, None] * x_heads[:, index]
        scan_outputs.append(scan.reshape(1, config.mamba_inner_size))
    scan_output = np.stack(scan_outputs, axis=1)
    mixed = _silu(gate) * scan_output
    mixed = mixed @ values["model.layers.0.mixer.out_proj.weight"].T
    hidden = hidden + _rms_norm(mixed, values["model.layers.0.post_mixer_norm.weight"])
    hidden = _rms_norm(hidden, values["model.norm.weight"])
    expected_logits = hidden @ values["model.embed_tokens.weight"].T

    np.testing.assert_allclose(outputs["logits"], expected_logits, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        outputs["present.0.conv_state"],
        history[:, :, -config.mamba_d_conv + 1 :],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(outputs["present.0.ssm_state"], state, rtol=2e-5, atol=2e-6)


def test_plamo2_attention_and_mlp_match_independent_reduced_reference() -> None:
    config = _config()
    model = build_from_module(
        Plamo2ForCausalLM(config),
        config,
        task=Plamo2CausalLMTask(),
    )["model"]
    rng = np.random.default_rng(19)
    values: dict[str, np.ndarray] = {}
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = tuple(dim if isinstance(dim, int) else 1 for dim in initializer.shape)
        value = np.zeros(shape, dtype=np.float32)
        if "norm" in initializer.name or initializer.name.endswith((".q_weight", ".k_weight")):
            value.fill(1.0)
        initializer.const_value = ir.Tensor(value)
        values[initializer.name] = value

    def assign(name: str, scale: float = 0.04) -> np.ndarray:
        value = (rng.standard_normal(values[name].shape) * scale).astype(np.float32)
        model.graph.initializers[name].const_value = ir.Tensor(value)
        values[name] = value
        return value

    assign("model.embed_tokens.weight")
    for layer in range(2):
        assign(f"model.layers.{layer}.mlp.gate_up_proj.weight")
        assign(f"model.layers.{layer}.mlp.down_proj.weight")
    assign("model.layers.1.mixer.qkv_proj.weight")
    assign("model.layers.1.mixer.o_proj.weight")

    input_ids = np.array([[1, 2, 3]], dtype=np.int64)
    outputs = OnnxModelSession(model, device="cpu").run(
        {
            "input_ids": input_ids,
            "position_ids": np.array([[0, 1, 2]], dtype=np.int64),
            "attention_mask": np.ones((1, 3), dtype=np.int64),
            **_empty_states(config, 1),
        }
    )

    hidden = values["model.embed_tokens.weight"][input_ids]
    mlp_input = _rms_norm(hidden, values["model.layers.0.pre_mlp_norm.weight"])
    mlp_output = _mlp_reference(
        mlp_input,
        values["model.layers.0.mlp.gate_up_proj.weight"],
        values["model.layers.0.mlp.down_proj.weight"],
    )
    hidden = hidden + _rms_norm(mlp_output, values["model.layers.0.post_mlp_norm.weight"])

    normalized = _rms_norm(hidden, values["model.layers.1.pre_mixer_norm.weight"])
    qkv = normalized @ values["model.layers.1.mixer.qkv_proj.weight"].T
    q_end = config.num_attention_heads * config.head_dim
    k_end = q_end + config.num_key_value_heads * config.head_dim
    q, k, v = np.split(qkv, [q_end, k_end], axis=-1)
    q = q.reshape(1, 3, config.num_attention_heads, config.head_dim)
    k = k.reshape(1, 3, config.num_key_value_heads, config.head_dim)
    v = v.reshape(1, 3, config.num_key_value_heads, config.head_dim)
    q = _rms_norm(q, values["model.layers.1.mixer.q_weight"])
    k = _rms_norm(k, values["model.layers.1.mixer.k_weight"])
    frequencies = 1.0 / (
        config.rope_theta
        ** (np.arange(0, config.head_dim, 2, dtype=np.float32) / config.head_dim)
    )
    angles = np.arange(3, dtype=np.float32)[:, None] * frequencies[None, :]
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[None, :, None, :]
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[None, :, None, :]

    def rotate_half(value: np.ndarray) -> np.ndarray:
        first, second = np.split(value, 2, axis=-1)
        return np.concatenate([-second, first], axis=-1)

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    repeats = config.num_attention_heads // config.num_key_value_heads
    k = np.repeat(k, repeats, axis=2)
    v = np.repeat(v, repeats, axis=2)
    scores = np.einsum("bthd,bshd->bhts", q, k) / np.sqrt(config.head_dim)
    scores = np.where(np.tril(np.ones((3, 3), dtype=bool))[None, None], scores, -np.inf)
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    attended = np.einsum("bhts,bshd->bthd", probabilities, v).reshape(1, 3, -1)
    mixed = attended @ values["model.layers.1.mixer.o_proj.weight"].T
    hidden = hidden + _rms_norm(mixed, values["model.layers.1.post_mixer_norm.weight"])
    mlp_input = _rms_norm(hidden, values["model.layers.1.pre_mlp_norm.weight"])
    mlp_output = _mlp_reference(
        mlp_input,
        values["model.layers.1.mlp.gate_up_proj.weight"],
        values["model.layers.1.mlp.down_proj.weight"],
    )
    hidden = hidden + _rms_norm(mlp_output, values["model.layers.1.post_mlp_norm.weight"])
    hidden = _rms_norm(hidden, values["model.norm.weight"])
    expected_logits = hidden @ values["model.embed_tokens.weight"].T
    np.testing.assert_allclose(outputs["logits"], expected_logits, rtol=3e-5, atol=3e-6)


def test_plamo2_prefill_decode_replay_reorder_and_rollback() -> None:
    config = _config()
    model = build_from_module(
        Plamo2ForCausalLM(config),
        config,
        task=Plamo2CausalLMTask(),
    )["model"]
    _fill_weights(model)
    session = OnnxModelSession(model, device="cpu")

    prompt = np.array([[1, 2], [3, 4]], dtype=np.int64)
    prefill = session.run(
        {
            "input_ids": prompt,
            "position_ids": np.array([[0, 1], [0, 1]], dtype=np.int64),
            "attention_mask": np.ones((2, 2), dtype=np.int64),
            **_empty_states(config, 2),
        }
    )
    saved = {name: value.copy() for name, value in _next_states(prefill).items()}
    decode_feed = {
        "input_ids": np.array([[5], [6]], dtype=np.int64),
        "position_ids": np.array([[2], [2]], dtype=np.int64),
        "attention_mask": np.ones((2, 3), dtype=np.int64),
        **saved,
    }
    first = session.run(decode_feed)
    replay = session.run(decode_feed)
    np.testing.assert_array_equal(first["logits"], replay["logits"])

    reordered_feed = {
        "input_ids": decode_feed["input_ids"][::-1].copy(),
        "position_ids": decode_feed["position_ids"][::-1].copy(),
        "attention_mask": decode_feed["attention_mask"][::-1].copy(),
        **{name: value[::-1].copy() for name, value in saved.items()},
    }
    reordered = session.run(reordered_feed)
    np.testing.assert_allclose(
        reordered["logits"], first["logits"][::-1], rtol=1e-5, atol=1e-6
    )

    # Rollback is copy-based: reusing the saved pre-decode state reproduces the step exactly.
    rolled_back = session.run(decode_feed)
    np.testing.assert_array_equal(first["logits"], rolled_back["logits"])


def test_plamo2_left_padding_does_not_change_recurrent_state_or_valid_logits() -> None:
    config = _config()
    model = build_from_module(
        Plamo2ForCausalLM(config),
        config,
        task=Plamo2CausalLMTask(),
    )["model"]
    _fill_weights(model)
    session = OnnxModelSession(model, device="cpu")

    padded = session.run(
        {
            "input_ids": np.array([[0, 7]], dtype=np.int64),
            "position_ids": np.array([[0, 0]], dtype=np.int64),
            "attention_mask": np.array([[0, 1]], dtype=np.int64),
            **_empty_states(config, 1),
        }
    )
    unpadded = session.run(
        {
            "input_ids": np.array([[7]], dtype=np.int64),
            "position_ids": np.array([[0]], dtype=np.int64),
            "attention_mask": np.array([[1]], dtype=np.int64),
            **_empty_states(config, 1),
        }
    )

    np.testing.assert_allclose(padded["logits"][:, -1], unpadded["logits"][:, 0], atol=1e-6)
    np.testing.assert_allclose(
        padded["present.0.conv_state"], unpadded["present.0.conv_state"], atol=1e-7
    )
    np.testing.assert_allclose(
        padded["present.0.ssm_state"], unpadded["present.0.ssm_state"], atol=1e-7
    )


def test_plamo2_attention_cache_is_bounded_by_local_window() -> None:
    config = _config()
    config.attention_window_size = 2
    model = build_from_module(
        Plamo2ForCausalLM(config),
        config,
        task=Plamo2CausalLMTask(),
    )["model"]
    _fill_weights(model)
    session = OnnxModelSession(model, device="cpu")
    prefill = session.run(
        {
            "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int64),
            "position_ids": np.array([[0, 1, 2, 3]], dtype=np.int64),
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            **_empty_states(config, 1),
        }
    )
    assert prefill["present.1.key"].shape[2] == 2
    decode = session.run(
        {
            "input_ids": np.array([[5]], dtype=np.int64),
            "position_ids": np.array([[4]], dtype=np.int64),
            "attention_mask": np.ones((1, 3), dtype=np.int64),
            **_next_states(prefill),
        }
    )
    assert decode["present.1.key"].shape[2] == 2
    assert decode["present.1.value"].shape[2] == 2


def test_plamo2_preprocesses_offsets_and_decay_values() -> None:
    model = Plamo2ForCausalLM(_config())
    weights = {
        "model.norm.weight": torch.tensor([2.0]),
        "model.layers.layers.0.pre_mixer_norm.weight": torch.tensor([3.0]),
        "model.layers.layers.0.post_mixer_norm.weight": torch.tensor([4.0]),
        "model.layers.layers.0.pre_mlp_norm.weight": torch.tensor([5.0]),
        "model.layers.layers.0.post_mlp_norm.weight": torch.tensor([6.0]),
        "model.layers.layers.0.mixer.A_log": torch.tensor([0.0, 1.0]),
        "model.embed_tokens.weight": torch.tensor([8.0]),
        "lm_head.weight": torch.tensor([9.0]),
    }
    actual = model.preprocess_weights(weights)
    torch.testing.assert_close(actual["model.norm.weight"], torch.tensor([3.0]))
    torch.testing.assert_close(
        actual["model.layers.0.pre_mixer_norm.weight"], torch.tensor([4.0])
    )
    torch.testing.assert_close(
        actual["model.layers.0.post_mixer_norm.weight"], torch.tensor([4.2])
    )
    torch.testing.assert_close(
        actual["model.layers.0.pre_mlp_norm.weight"], torch.tensor([6.0])
    )
    torch.testing.assert_close(
        actual["model.layers.0.post_mlp_norm.weight"],
        torch.tensor([6.0 + 1.0 / (5.0**1.5)]),
    )
    torch.testing.assert_close(actual["model.layers.0.mixer.A_log"], torch.tensor([0.0, 1.0]))
    gguf_actual = model.preprocess_weights(
        {"model.layers.0.mixer.A": -torch.exp(torch.tensor([0.0, 1.0]))}
    )
    torch.testing.assert_close(
        gguf_actual["model.layers.0.mixer.A_log"], torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(actual["lm_head.weight"], torch.tensor([8.0]))


def test_plamo2_preprocesses_effectively_tied_quantized_embeddings() -> None:
    config = _config()
    config.tie_word_embeddings = False
    config.quantization = QuantizationConfig(tie_word_embeddings=True)
    model = Plamo2ForCausalLM(config)

    actual = model.preprocess_weights(
        {
            "model.embed_tokens.weight": torch.tensor([8.0]),
            "lm_head.weight": torch.tensor([9.0]),
        }
    )

    torch.testing.assert_close(actual["model.embed_tokens.weight"], torch.tensor([8.0]))
    torch.testing.assert_close(actual["lm_head.weight"], torch.tensor([8.0]))


def test_plamo2_transformers_config_uses_explicit_schedule_and_pinned_defaults() -> None:
    source = SimpleNamespace(
        model_type="plamo2",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=32,
        hidden_size_per_head=8,
        hidden_act=None,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        rope_local_theta=8_000.0,
        attention_head_counts=[0, 4, 0, 4],
        attention_kv_head_counts=[0, 2, 0, 2],
        mamba_num_heads=6,
        mamba_d_state=6,
        mamba_d_conv=3,
        mamba_dt_rank=10,
        tie_word_embeddings=True,
    )
    config = Plamo2Config.from_transformers(source)
    assert config.attention_head_counts == (0, 4, 0, 4)
    assert config.attention_kv_head_counts == (0, 2, 0, 2)
    assert config.mamba_num_heads == 6
    assert config.mamba_d_state == 6
    assert config.mamba_d_conv == 3
    assert config.mamba_dt_rank == 10
    assert config.mamba_inner_size == 48
    assert np.isclose(config.rope_theta, 8_000.0)
    assert config.hidden_act == "silu"
    assert config.mamba_group_count == 0
    assert config.use_predefined_initial_state is False
