# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end tests for the pinned Kimi-K3 GGUF contract."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest


def _write_kimi_k3_gguf(
    path: Path,
    *,
    quantized: bool = False,
    omit: str | None = None,
    extra: str | None = None,
    malformed_shape: str | None = None,
    kv_heads: list[int] | None = None,
    gating: int = 2,
    shared: int = 2,
    fused_kv_b: bool = False,
    omit_metadata: str | None = None,
    conv: int = 4,
) -> dict[str, np.ndarray]:
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden = 64
    dense_intermediate = 64
    expert_intermediate = 32
    expert_latent = 32
    vocab = 64
    heads = 2
    kda_dim = 32
    qk_dim = 48
    extra_dim = 16
    value_dim = 32
    q_rank = 32
    kv_rank = 32
    experts = 2
    rng = np.random.default_rng(623)
    tensors: dict[str, np.ndarray] = {}

    writer = GGUFWriter(str(path), "kimi-k3")
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(dense_intermediate)
    writer.add_block_count(2)
    writer.add_head_count(heads)
    writer.add_array(
        "kimi-k3.attention.head_count_kv",
        kv_heads if kv_heads is not None else [0, 1],
    )
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_uint32("kimi-k3.attention.q_lora_rank", q_rank)
    writer.add_uint32("kimi-k3.attention.key_length_mla", qk_dim)
    writer.add_uint32("kimi-k3.attention.value_length_mla", value_dim)
    writer.add_uint32("kimi-k3.attention.kv_lora_rank", kv_rank)
    writer.add_rope_dimension_count(extra_dim)
    writer.add_uint32("kimi-k3.attention.key_length", kv_rank + extra_dim)
    writer.add_uint32("kimi-k3.attention.value_length", kv_rank)
    writer.add_uint32("kimi-k3.ssm.conv_kernel", conv)
    writer.add_uint32("kimi-k3.kda.head_dim", kda_dim)
    if omit_metadata != "kda.gate_lower_bound":
        writer.add_float32("kimi-k3.kda.gate_lower_bound", -5.0)
    writer.add_expert_count(experts)
    writer.add_expert_used_count(1)
    writer.add_uint32("kimi-k3.expert_feed_forward_length", expert_intermediate)
    writer.add_uint32("kimi-k3.expert_shared_count", shared)
    if omit_metadata != "leading_dense_block_count":
        writer.add_uint32("kimi-k3.leading_dense_block_count", 1)
    if omit_metadata != "expert_weights_scale":
        writer.add_float32("kimi-k3.expert_weights_scale", 1.0)
    if omit_metadata != "expert_weights_norm":
        writer.add_bool("kimi-k3.expert_weights_norm", True)
    writer.add_uint32("kimi-k3.expert_gating_func", gating)
    writer.add_uint32("kimi-k3.expert_latent_length", expert_latent)
    writer.add_uint32("kimi-k3.attn_res.block_size", 1)
    writer.add_float32("kimi-k3.activation.situ_beta", 4.0)
    writer.add_float32("kimi-k3.activation.situ_linear_beta", 25.0)
    writer.add_vocab_size(vocab)

    def shape_for(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
        if name == malformed_shape:
            return (*shape[:-1], shape[-1] + 1)
        return shape

    def add_float(name: str, shape: tuple[int, ...]) -> None:
        if name == omit:
            return
        shape = shape_for(name, shape)
        values = rng.normal(0.0, 0.02, shape).astype(np.float32)
        if name.endswith("ssm_a"):
            values = -np.exp(values)
        elif name.endswith(
            (
                "output_norm.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn_q_a_norm.weight",
                "attn_kv_a_norm.weight",
                "ssm_norm.weight",
                "ffn_routed_norm.weight",
            )
        ):
            values.fill(1.0)
        tensors[name] = values
        writer.add_tensor(name, values)

    def add_q4(name: str, shape: tuple[int, ...]) -> None:
        if name == omit:
            return
        shape = shape_for(name, shape)
        assert shape[-1] % 32 == 0
        raw = np.zeros((*shape[:-1], shape[-1] // 32 * 18), dtype=np.uint8)
        for index in np.ndindex(shape[:-1]):
            for block in range(shape[-1] // 32):
                offset = block * 18
                raw[(*index, slice(offset, offset + 2))] = np.array(
                    [rng.uniform(0.01, 0.05)], dtype=np.float16
                ).view(np.uint8)
                raw[(*index, slice(offset + 2, offset + 18))] = rng.integers(
                    0, 256, 16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    def add_experts(name: str, shape: tuple[int, ...], base: float) -> None:
        if quantized:
            add_q4(name, shape)
            return
        values = np.empty(shape, dtype=np.float32)
        for expert in range(shape[0]):
            values[expert].fill(base + expert)
        tensors[name] = values
        writer.add_tensor(name, values)

    add_float("token_embd.weight", (vocab, hidden))
    add_float("output_norm.weight", (hidden,))
    add_float("output_res_score.weight", (hidden,))
    projection = add_q4 if quantized else add_float
    projection("output.weight", (vocab, hidden))
    projection_width = heads * kda_dim

    for layer in range(2):
        prefix = f"blk.{layer}."
        add_float(prefix + "attn_norm.weight", (hidden,))
        add_float(prefix + "ffn_norm.weight", (hidden,))
        add_float(prefix + "attn_res_score.weight", (hidden,))
        add_float(prefix + "ffn_res_score.weight", (hidden,))
        if layer == 0:
            projection(prefix + "attn_q.weight", (projection_width, hidden))
            projection(prefix + "attn_k.weight", (projection_width, hidden))
            projection(prefix + "attn_v.weight", (projection_width, hidden))
            add_float(prefix + "ssm_conv1d_q.weight", (1, projection_width, 1, conv))
            add_float(prefix + "ssm_conv1d_k.weight", (1, projection_width, 1, conv))
            add_float(prefix + "ssm_conv1d_v.weight", (1, projection_width, 1, conv))
            projection(prefix + "ssm_f_a.weight", (kda_dim, hidden))
            projection(prefix + "ssm_f_b.weight", (projection_width, kda_dim))
            projection(prefix + "ssm_beta.weight", (heads, hidden))
            add_float(prefix + "ssm_a", (heads,))
            add_float(prefix + "ssm_dt.bias", (projection_width,))
            projection(prefix + "ssm_g.weight", (projection_width, hidden))
            add_float(prefix + "ssm_norm.weight", (kda_dim,))
            projection(prefix + "attn_output.weight", (hidden, projection_width))
        else:
            projection(prefix + "attn_q_a.weight", (q_rank, hidden))
            add_float(prefix + "attn_q_a_norm.weight", (q_rank,))
            projection(prefix + "attn_q_b.weight", (heads * qk_dim, q_rank))
            projection(prefix + "attn_kv_a_mqa.weight", (kv_rank + extra_dim, hidden))
            add_float(prefix + "attn_kv_a_norm.weight", (kv_rank,))
            if fused_kv_b:
                projection(
                    prefix + "attn_kv_b.weight",
                    (heads * (qk_dim - extra_dim + value_dim), kv_rank),
                )
            else:
                projection(prefix + "attn_k_b.weight", (heads, kv_rank, qk_dim - extra_dim))
                projection(prefix + "attn_v_b.weight", (heads, value_dim, kv_rank))
            projection(prefix + "attn_gate.weight", (heads * value_dim, hidden))
            projection(prefix + "attn_output.weight", (hidden, heads * value_dim))

        if layer == 0:
            projection(prefix + "ffn_gate.weight", (dense_intermediate, hidden))
            projection(prefix + "ffn_up.weight", (dense_intermediate, hidden))
            projection(prefix + "ffn_down.weight", (hidden, dense_intermediate))
        else:
            projection(prefix + "ffn_gate_inp.weight", (experts, hidden))
            add_float(prefix + "exp_probs_b.bias", (experts,))
            add_experts(
                prefix + "ffn_gate_exps.weight",
                (experts, expert_intermediate, expert_latent),
                11.0,
            )
            add_experts(
                prefix + "ffn_up_exps.weight",
                (experts, expert_intermediate, expert_latent),
                21.0,
            )
            add_experts(
                prefix + "ffn_down_exps.weight",
                (experts, expert_latent, expert_intermediate),
                31.0,
            )
            projection(prefix + "ffn_routed_down.weight", (expert_latent, hidden))
            projection(prefix + "ffn_routed_up.weight", (hidden, expert_latent))
            add_float(prefix + "ffn_routed_norm.weight", (expert_latent,))
            shared_width = shared * expert_intermediate
            projection(prefix + "ffn_gate_shexp.weight", (shared_width, hidden))
            projection(prefix + "ffn_up_shexp.weight", (shared_width, hidden))
            projection(prefix + "ffn_down_shexp.weight", (hidden, shared_width))

    if extra is not None:
        add_float(extra, (1,))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


def _kimi_k3_reference(
    tensors: dict[str, np.ndarray], input_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reduced direct implementation of the pinned K3 equations for two layers."""

    def linear(x: np.ndarray, name: str) -> np.ndarray:
        if name.endswith("]"):
            base, _, index = name[:-1].rpartition("[")
            weight = tensors[base][int(index)]
        else:
            weight = tensors[name]
        return x @ weight.T

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def rms(x: np.ndarray, name: str, eps: float = 1e-5) -> np.ndarray:
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * tensors[name]

    def situ(x: np.ndarray, gate_name: str, up_name: str, down_name: str) -> np.ndarray:
        gate = linear(x, gate_name)
        up = linear(x, up_name)
        activated = 4.0 * np.tanh(gate / 4.0) * sigmoid(gate)
        activated *= 25.0 * np.tanh(up / 25.0)
        return linear(activated, down_name)

    def mix(prefix: np.ndarray, bank: list[np.ndarray], name: str) -> np.ndarray:
        values = np.stack([*bank, prefix], axis=2)
        normalized = values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + 1e-5)
        scores = normalized @ tensors[name][:, None]
        scores = scores[..., 0]
        probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        return np.sum(values * probabilities[..., None], axis=2)

    def causal_conv(x: np.ndarray, name: str) -> np.ndarray:
        # GGUF stores K3 depthwise kernels as [1, channels, 1, kernel].
        weight = tensors[name][0, :, 0, :]
        padded = np.pad(x.transpose(0, 2, 1), ((0, 0), (0, 0), (3, 0)))
        output = np.empty_like(x)
        for token in range(x.shape[1]):
            convolved = np.sum(padded[:, :, token : token + 4] * weight, axis=-1)
            output[:, token, :] = convolved * sigmoid(convolved)
        return output

    hidden = tensors["token_embd.weight"][input_ids]
    initial_embedding = hidden.copy()

    # Layer 0: KDA with full-rank decay gate and FP32 gated-delta recurrence.
    normed = rms(hidden, "blk.0.attn_norm.weight")
    q = causal_conv(linear(normed, "blk.0.attn_q.weight"), "blk.0.ssm_conv1d_q.weight")
    k = causal_conv(linear(normed, "blk.0.attn_k.weight"), "blk.0.ssm_conv1d_k.weight")
    v = causal_conv(linear(normed, "blk.0.attn_v.weight"), "blk.0.ssm_conv1d_v.weight")
    q = q.reshape(1, 2, 2, 32)
    k = k.reshape(1, 2, 2, 32)
    q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    z = (
        linear(linear(normed, "blk.0.ssm_f_a.weight"), "blk.0.ssm_f_b.weight")
        + tensors["blk.0.ssm_dt.bias"]
    )
    decay = -5.0 * sigmoid(
        (-tensors["blk.0.ssm_a"])[None, None, :, None] * z.reshape(1, 2, 2, 32)
    )
    beta = sigmoid(linear(normed, "blk.0.ssm_beta.weight"))
    state = np.zeros((1, 2, 32, 32), np.float32)
    recurrent_output = np.empty((1, 2, 2, 32), np.float32)
    for token in range(2):
        state *= np.exp(decay[:, token, :, :, None])
        retrieval = np.einsum("bhd,bhdv->bhv", k[:, token], state)
        delta = (v.reshape(1, 2, 2, 32)[:, token] - retrieval) * beta[:, token, :, None]
        state += np.einsum("bhd,bhv->bhdv", k[:, token], delta)
        recurrent_output[:, token] = np.einsum(
            "bhd,bhdv->bhv", q[:, token] * (32**-0.5), state
        )
    gate = linear(normed, "blk.0.ssm_g.weight").reshape(1, 2, 2, 32)
    recurrent_output = (
        recurrent_output
        / np.sqrt(np.mean(recurrent_output**2, axis=-1, keepdims=True) + 1e-5)
        * tensors["blk.0.ssm_norm.weight"]
        * sigmoid(gate)
    )
    attention = linear(recurrent_output.reshape(1, 2, 64), "blk.0.attn_output.weight")
    mixed = mix(attention, [initial_embedding], "blk.0.ffn_res_score.weight")
    dense_input = rms(mixed, "blk.0.ffn_norm.weight")
    hidden = attention + situ(
        dense_input,
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    )
    first_layer_output = hidden.copy()

    # Layer 1: gated NoPE MLA, then latent routed MoE plus shared experts.
    hidden = mix(hidden, [initial_embedding], "blk.1.attn_res_score.weight")
    normed = rms(hidden, "blk.1.attn_norm.weight")
    query = linear(
        rms(
            linear(normed, "blk.1.attn_q_a.weight"),
            "blk.1.attn_q_a_norm.weight",
            1e-6,
        ),
        "blk.1.attn_q_b.weight",
    ).reshape(1, 2, 2, 48)
    kv = linear(normed, "blk.1.attn_kv_a_mqa.weight")
    compressed = rms(kv[..., :32], "blk.1.attn_kv_a_norm.weight", 1e-6)
    extra = np.broadcast_to(kv[..., 32:][:, :, None, :], (1, 2, 2, 16))
    key_nope = np.einsum("btk,hkn->bthn", compressed, tensors["blk.1.attn_k_b.weight"])
    key = np.concatenate((key_nope, extra), axis=-1)
    value = np.einsum("btk,hvk->bthv", compressed, tensors["blk.1.attn_v_b.weight"])
    scores = np.einsum("bthd,bshd->bhts", query, key) * (48**-0.5)
    scores += np.triu(np.full((2, 2), -np.inf, np.float32), 1)[None, None]
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    attention = np.einsum("bhts,bshv->bthv", probabilities, value).reshape(1, 2, 64)
    attention *= sigmoid(linear(normed, "blk.1.attn_gate.weight"))
    attention = linear(attention, "blk.1.attn_output.weight")

    mixed = mix(
        attention,
        [initial_embedding, first_layer_output],
        "blk.1.ffn_res_score.weight",
    )
    moe_input = rms(mixed, "blk.1.ffn_norm.weight")
    latent = linear(moe_input, "blk.1.ffn_routed_down.weight")
    router_scores = sigmoid(linear(moe_input, "blk.1.ffn_gate_inp.weight"))
    selected = np.argmax(router_scores + tensors["blk.1.exp_probs_b.bias"], axis=-1)
    routed = np.empty_like(latent)
    for expert in range(2):
        expert_output = situ(
            latent,
            f"blk.1.ffn_gate_exps.weight[{expert}]",
            f"blk.1.ffn_up_exps.weight[{expert}]",
            f"blk.1.ffn_down_exps.weight[{expert}]",
        )
        routed = np.where((selected == expert)[..., None], expert_output, routed)
    routed = linear(
        rms(routed, "blk.1.ffn_routed_norm.weight"),
        "blk.1.ffn_routed_up.weight",
    )
    shared = situ(
        moe_input,
        "blk.1.ffn_gate_shexp.weight",
        "blk.1.ffn_up_shexp.weight",
        "blk.1.ffn_down_shexp.weight",
    )
    hidden = attention + routed + shared

    hidden = mix(
        hidden,
        [initial_embedding, first_layer_output],
        "output_res_score.weight",
    )
    logits = linear(rms(hidden, "output_norm.weight"), "output.weight")
    return logits, state


class TestKimiK3GGUFBuild:
    @staticmethod
    def _inputs(
        tokens: np.ndarray,
        states: dict[str, np.ndarray] | None = None,
        attention_mask: np.ndarray | None = None,
    ):
        batch, sequence = tokens.shape
        if states is None:
            states = {
                "past_key_values.0.q_conv_state": np.zeros((batch, 64, 3), np.float32),
                "past_key_values.0.k_conv_state": np.zeros((batch, 64, 3), np.float32),
                "past_key_values.0.v_conv_state": np.zeros((batch, 64, 3), np.float32),
                "past_key_values.0.recurrent_state": np.zeros((batch, 2, 32, 32), np.float32),
                "past_key_values.1.key": np.zeros((batch, 2, 0, 48), np.float32),
                "past_key_values.1.value": np.zeros((batch, 2, 0, 32), np.float32),
            }
        past = states["past_key_values.1.key"].shape[2]
        return {
            "input_ids": tokens,
            "attention_mask": (
                np.ones((batch, past + sequence), np.int64)
                if attention_mask is None
                else attention_mask
            ),
            "position_ids": np.broadcast_to(
                np.arange(past, past + sequence, dtype=np.int64), (batch, sequence)
            ).copy(),
            **states,
        }

    def test_reduced_authoritative_equations_match_ort(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-k3-reference.gguf"
        tensors = _write_kimi_k3_gguf(path)
        model = build_from_gguf(path)["model"]
        tokens = np.asarray([[1, 2]], np.int64)
        expected_logits, expected_state = _kimi_k3_reference(tensors, tokens)

        session = OnnxModelSession(model)
        try:
            actual = session.run(self._inputs(tokens))
        finally:
            session.close()

        np.testing.assert_allclose(actual["logits"], expected_logits, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(
            actual["present.0.recurrent_state"],
            expected_state,
            rtol=2e-4,
            atol=2e-4,
        )

    def test_float_runtime_replay_reorder_and_roundtrip(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-k3-f32.gguf"
        _write_kimi_k3_gguf(path)
        package = build_from_gguf(path)
        assert package.config.model_type == "kimi_k3"
        assert package.config.layer_types == ["kimi_k3_attention", "full_attention"]
        model = package["model"]
        assert "KDA:q_conv_state" in model.metadata_props["mobius.cache_abi"]
        for expert in range(2):
            prefix = f"model.layers.1.block_sparse_moe.moe.experts.{expert}"
            np.testing.assert_array_equal(
                model.graph.initializers[f"{prefix}.gate_proj.weight_t"].const_value.numpy(),
                11.0 + expert,
            )
            np.testing.assert_array_equal(
                model.graph.initializers[f"{prefix}.up_proj.weight_t"].const_value.numpy(),
                21.0 + expert,
            )
            np.testing.assert_array_equal(
                model.graph.initializers[f"{prefix}.down_proj.weight_t"].const_value.numpy(),
                31.0 + expert,
            )

        session = OnnxModelSession(model)
        try:
            prefill = session.run(self._inputs(np.asarray([[1, 2], [3, 4]], np.int64)))
            snapshot = {
                name.replace("present.", "past_key_values."): value.copy()
                for name, value in prefill.items()
                if name.startswith("present.")
            }
            first = session.run(self._inputs(np.asarray([[5], [6]], np.int64), snapshot))
            replay = session.run(self._inputs(np.asarray([[5], [6]], np.int64), snapshot))
            np.testing.assert_array_equal(first["logits"], replay["logits"])
            swapped_states = {name: value[::-1].copy() for name, value in snapshot.items()}
            swapped = session.run(
                self._inputs(np.asarray([[6], [5]], np.int64), swapped_states)
            )
            np.testing.assert_allclose(
                swapped["logits"], first["logits"][::-1], rtol=1e-5, atol=1e-5
            )
        finally:
            session.close()

        output = tmp_path / "roundtrip"
        package.save(output, progress_bar=False)
        reloaded = ModelPackage.load(output)
        assert (
            reloaded["model"].metadata_props["mobius.cache_abi"]
            == model.metadata_props["mobius.cache_abi"]
        )

    def test_right_padding_preserves_kda_state_for_cached_decode(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-k3-right-padding.gguf"
        _write_kimi_k3_gguf(path)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        try:
            unpadded = session.run(self._inputs(np.asarray([[7, 8]], np.int64)))
            padded = session.run(
                self._inputs(
                    np.asarray([[7, 8, 0, 0]], np.int64),
                    attention_mask=np.asarray([[1, 1, 0, 0]], np.int64),
                )
            )
            for suffix in (
                "q_conv_state",
                "k_conv_state",
                "v_conv_state",
                "recurrent_state",
            ):
                np.testing.assert_allclose(
                    padded[f"present.0.{suffix}"],
                    unpadded[f"present.0.{suffix}"],
                    rtol=1e-5,
                    atol=1e-5,
                )

            unpadded_states = {
                name.replace("present.", "past_key_values."): value
                for name, value in unpadded.items()
                if name.startswith("present.")
            }
            padded_states = {
                name.replace("present.", "past_key_values."): value
                for name, value in padded.items()
                if name.startswith("present.")
            }
            token = np.asarray([[9]], np.int64)
            expected = session.run(self._inputs(token, unpadded_states))
            actual = session.run(
                self._inputs(
                    token,
                    padded_states,
                    attention_mask=np.asarray([[1, 1, 0, 0, 1]], np.int64),
                )
            )
            np.testing.assert_allclose(
                actual["logits"], expected["logits"], rtol=1e-5, atol=1e-5
            )
        finally:
            session.close()

    def test_separate_quantized_mla_requires_explicit_dequantization(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-k3-q4.gguf"
        _write_kimi_k3_gguf(path, quantized=True)
        with pytest.raises(ValueError, match=r"attn_k_b\.weight \(Q4_0\)"):
            build_from_gguf(path, keep_quantized=True)

        model = build_from_gguf(path, keep_quantized=False)["model"]
        assert all(node.op_type != "MatMulNBits" for node in model.graph)

    def test_fused_kv_b_float_values_and_quantized_import(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.integrations.gguf._repacker import repack_gguf_tensor

        float_path = tmp_path / "kimi-k3-fused-kv-f32.gguf"
        tensors = _write_kimi_k3_gguf(float_path, fused_kv_b=True)
        model = build_from_gguf(float_path)["model"]
        fused = tensors["blk.1.attn_kv_b.weight"].reshape(2, 64, 32)
        np.testing.assert_array_equal(
            model.graph.initializers[
                "model.layers.1.self_attn.k_b_proj.weight_t"
            ].const_value.numpy(),
            fused[:, :32].reshape(64, 32).T,
        )
        np.testing.assert_array_equal(
            model.graph.initializers[
                "model.layers.1.self_attn.v_b_proj.weight_t"
            ].const_value.numpy(),
            fused[:, 32:].reshape(64, 32).T,
        )

        quantized_path = tmp_path / "kimi-k3-fused-kv-q4.gguf"
        _write_kimi_k3_gguf(quantized_path, quantized=True, fused_kv_b=True)
        quantized = build_from_gguf(quantized_path, keep_quantized=True)["model"]
        packed_inputs = {
            node.inputs[1].name
            for node in quantized.graph
            if node.op_type == "MatMulNBits" and len(node.inputs) > 1
        }
        assert any("k_b_proj" in name for name in packed_inputs)
        assert any("v_b_proj" in name for name in packed_inputs)
        source = GGUFModel(quantized_path)
        raw, qtype, shape = next(
            (raw, qtype, shape)
            for name, raw, qtype, shape in source.tensor_items_raw()
            if name == "blk.1.attn_kv_b.weight"
        )
        repacked = repack_gguf_tensor(raw, qtype.value, shape)
        for target, start, end in (("k_b_proj", 0, 32), ("v_b_proj", 32, 64)):
            stem = f"model.layers.1.self_attn.{target}"
            expected_weight = repacked.weight.reshape(2, 64, *repacked.weight.shape[1:])[
                :, start:end
            ].reshape(-1, *repacked.weight.shape[1:])
            expected_scales = repacked.scales.reshape(2, 64, *repacked.scales.shape[1:])[
                :, start:end
            ].reshape(-1, *repacked.scales.shape[1:])
            np.testing.assert_array_equal(
                quantized.graph.initializers[f"{stem}.weight"].const_value.numpy(),
                expected_weight,
            )
            np.testing.assert_array_equal(
                quantized.graph.initializers[f"{stem}.scales"].const_value.numpy(),
                expected_scales,
            )
            if repacked.zero_points is not None:
                expected_zero_points = repacked.zero_points.reshape(
                    2, 64, *repacked.zero_points.shape[1:]
                )[:, start:end].reshape(-1, *repacked.zero_points.shape[1:])
                np.testing.assert_array_equal(
                    quantized.graph.initializers[f"{stem}.zero_points"].const_value.numpy(),
                    expected_zero_points,
                )

    def test_static_cache_is_rejected(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-k3-static.gguf"
        _write_kimi_k3_gguf(path)
        with pytest.raises(ValueError, match="does not support static cache"):
            build_from_gguf(path, static_cache=True)

    def test_generic_task_override_is_rejected(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-k3-task.gguf"
        _write_kimi_k3_gguf(path)
        with pytest.raises(ValueError, match="heterogeneous-state task"):
            build_from_gguf(path, task="text-generation")

    def test_cli_build(self, tmp_path: Path) -> None:
        from mobius.__main__ import main
        from mobius._model_package import ModelPackage

        path = tmp_path / "kimi-k3-cli.gguf"
        output = tmp_path / "kimi-k3-cli-output"
        _write_kimi_k3_gguf(path)
        main(["build-gguf", str(path), "--output", str(output), "--dequantize"])
        package = ModelPackage.load(output)
        assert "KDA:q_conv_state" in package["model"].metadata_props["mobius.cache_abi"]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omit": "output_res_score.weight"}, "tensor closure"),
            ({"extra": "blk.2.attn_q.weight"}, "out_of_range"),
            ({"malformed_shape": "blk.1.attn_q_b.weight"}, "invalid tensor shape"),
            ({"kv_heads": [0, 0]}, "requires both KDA and MLA"),
            ({"gating": 0}, "inconsistent pinned architecture metadata"),
            ({"shared": 1}, "inconsistent pinned architecture metadata"),
            ({"conv": 1}, "inconsistent pinned architecture metadata"),
            (
                {"omit_metadata": "leading_dense_block_count"},
                "inconsistent pinned architecture metadata",
            ),
            (
                {"omit_metadata": "expert_weights_scale"},
                "inconsistent pinned architecture metadata",
            ),
            (
                {"omit_metadata": "expert_weights_norm"},
                "inconsistent pinned architecture metadata",
            ),
            (
                {"omit_metadata": "kda.gate_lower_bound"},
                "inconsistent pinned architecture metadata",
            ),
        ],
    )
    def test_invalid_contract_fails_before_graph(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-k3-invalid.gguf"
        _write_kimi_k3_gguf(path, **kwargs)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)
        with pytest.raises((TypeError, ValueError), match=match):
            build_from_gguf(path)
        graph_build.assert_not_called()
