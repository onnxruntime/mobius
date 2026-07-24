# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Numerical parity of the from-scratch LLaDA model against a torch reference.

Builds the Mobius LLaDA ONNX graph and a self-contained PyTorch reference
(Llama block: RMSNorm, HF-Llama RoPE, *bidirectional* multi-head attention,
SwiGLU MLP, no biases) with the *same* random weights — no checkpoint download.
The weights are generated in the HuggingFace ``LLaDAModelLM`` naming scheme so
:meth:`LLaDAModel.preprocess_weights` is exercised end to end.

Two properties are asserted:

* **Parity** — ``max|Δ|`` between the ONNX logits and the torch reference is
  below ``1e-4``.
* **Bidirectionality** — perturbing a *later* token's input id changes an
  *earlier* position's logits. A causal model could not do this.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from mobius._configs import ArchitectureConfig
from mobius._weight_loading import apply_weights
from mobius.models.llada import LLaDAModel
from mobius.tasks._masked_diffusion import MaskedDiffusionTask


def _make_config() -> ArchitectureConfig:
    """Build a tiny LLaDA config for a disk-light parity test."""
    hidden_size = 64
    num_heads = 4
    return ArchitectureConfig(
        vocab_size=50,
        hidden_size=hidden_size,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=hidden_size // num_heads,
        hidden_act="silu",
        pad_token_id=0,
        tie_word_embeddings=False,
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_type="default",
        rope_theta=500000.0,
    )


def _random_llada_weights(config: ArchitectureConfig) -> dict[str, torch.Tensor]:
    """Generate random weights in HuggingFace ``LLaDAModelLM`` naming.

    Projection matrices use a small init scale (like a real transformer) and
    norm weights sit near ``1.0`` so activations stay ``O(1)`` and the parity
    comparison is not dominated by float32 accumulation on large magnitudes.
    """
    torch.manual_seed(0)
    hidden = config.hidden_size
    inter = config.intermediate_size
    scale = 0.02
    prefix = "model.transformer."
    state: dict[str, torch.Tensor] = {
        f"{prefix}wte.weight": torch.randn(config.vocab_size, hidden) * scale,
        f"{prefix}ln_f.weight": 1.0 + torch.randn(hidden) * scale,
        f"{prefix}ff_out.weight": torch.randn(config.vocab_size, hidden) * scale,
    }
    for i in range(config.num_hidden_layers):
        block = f"{prefix}blocks.{i}."
        state[block + "q_proj.weight"] = torch.randn(hidden, hidden) * scale
        state[block + "k_proj.weight"] = torch.randn(hidden, hidden) * scale
        state[block + "v_proj.weight"] = torch.randn(hidden, hidden) * scale
        state[block + "attn_out.weight"] = torch.randn(hidden, hidden) * scale
        state[block + "ff_proj.weight"] = torch.randn(inter, hidden) * scale
        state[block + "up_proj.weight"] = torch.randn(inter, hidden) * scale
        state[block + "ff_out.weight"] = torch.randn(hidden, inter) * scale
        state[block + "attn_norm.weight"] = 1.0 + torch.randn(hidden) * scale
        state[block + "ff_norm.weight"] = 1.0 + torch.randn(hidden) * scale
    return state


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_cos_sin(
    seq_len: int, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _reference_logits(
    config: ArchitectureConfig,
    state: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Self-contained bidirectional Llama-block reference forward."""
    prefix = "model.transformer."
    num_heads = config.num_attention_heads
    head_dim = config.head_dim
    scale = head_dim**-0.5
    eps = config.rms_norm_eps
    _, seq_len = input_ids.shape

    hidden_states = torch.nn.functional.embedding(input_ids, state[f"{prefix}wte.weight"])
    cos, sin = _rope_cos_sin(seq_len, head_dim, config.rope_theta)

    for i in range(config.num_hidden_layers):
        block = f"{prefix}blocks.{i}."
        residual = hidden_states
        normed = _rms_norm(hidden_states, state[block + "attn_norm.weight"], eps)

        query = normed @ state[block + "q_proj.weight"].T
        key = normed @ state[block + "k_proj.weight"].T
        value = normed @ state[block + "v_proj.weight"].T

        batch = normed.shape[0]
        query = query.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin

        scores = (query @ key.transpose(-1, -2)) * scale
        # Bidirectional: no causal mask.
        weights = torch.softmax(scores, dim=-1)
        attn = weights @ value
        attn = attn.transpose(1, 2).reshape(batch, seq_len, num_heads * head_dim)
        attn = attn @ state[block + "attn_out.weight"].T
        hidden_states = residual + attn

        residual = hidden_states
        normed = _rms_norm(hidden_states, state[block + "ff_norm.weight"], eps)
        gate = normed @ state[block + "ff_proj.weight"].T
        up = normed @ state[block + "up_proj.weight"].T
        mlp = (torch.nn.functional.silu(gate) * up) @ state[block + "ff_out.weight"].T
        hidden_states = residual + mlp

    hidden_states = _rms_norm(hidden_states, state[f"{prefix}ln_f.weight"], eps)
    return hidden_states @ state[f"{prefix}ff_out.weight"].T


def _build_onnx_session(config: ArchitectureConfig, state: dict[str, torch.Tensor]):
    import onnx_ir
    import onnxruntime as ort

    module = LLaDAModel(config)
    model = MaskedDiffusionTask().build(module, config)["model"]
    apply_weights(model, module.preprocess_weights(state))

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.onnx"
        onnx_ir.save(model, model_path)
        # onnxruntime reads the model fully at construction, so the temp file
        # can be removed as soon as the session exists.
        return ort.InferenceSession(str(model_path))


def test_llada_matches_torch_reference():
    """ONNX logits match the self-contained torch reference to < 1e-4."""
    pytest.importorskip("onnxruntime")

    config = _make_config()
    state = _random_llada_weights(config)

    input_ids = torch.tensor([[3, 7, 1, 9, 4, 2]], dtype=torch.int64)
    with torch.no_grad():
        expected = _reference_logits(config, state, input_ids).numpy()

    session = _build_onnx_session(config, state)
    actual = session.run(None, {"input_ids": input_ids.numpy()})[0]

    max_delta = np.abs(actual - expected).max()
    assert max_delta < 1e-4, f"max|Δ|={max_delta}"


def test_llada_attention_is_bidirectional():
    """Perturbing a later token changes an earlier position's logits."""
    pytest.importorskip("onnxruntime")

    config = _make_config()
    state = _random_llada_weights(config)
    session = _build_onnx_session(config, state)

    base_ids = np.array([[3, 7, 1, 9, 4, 2]], dtype=np.int64)
    perturbed_ids = base_ids.copy()
    perturbed_ids[0, -1] = 5  # change the LAST token only

    base = session.run(None, {"input_ids": base_ids})[0]
    perturbed = session.run(None, {"input_ids": perturbed_ids})[0]

    # Position 0's logits must react to a change at the final position — this
    # is only possible under bidirectional attention. A causal model would
    # leave every position before the perturbation untouched.
    earlier_delta = np.abs(base[0, 0] - perturbed[0, 0]).max()
    assert earlier_delta > 1e-4, f"earlier position unchanged (Δ={earlier_delta})"


def test_llada_export_signature_matches_masked_diffusion_metadata():
    """The exported ONNX I/O matches the onnx-genai masked-diffusion contract.

    Builds a tiny LLaDA package and asserts the graph exposes exactly
    ``input_ids [B, S]`` int64 in and ``logits [B, S, V]`` f32 out with no
    past/present KV, then checks that
    :func:`build_language_diffusion_pipeline_metadata` emits a pipeline whose
    denoiser self-edge references those same ports.
    """
    import onnx_ir as ir

    from mobius.integrations.onnx_genai.inference_metadata import (
        build_language_diffusion_pipeline_metadata,
    )

    config = _make_config()
    module = LLaDAModel(config)
    model = MaskedDiffusionTask().build(module, config)["model"]
    graph = model.graph

    # Exactly one input: input_ids [B, S] int64.
    assert [value.name for value in graph.inputs] == ["input_ids"]
    input_ids = graph.inputs[0]
    assert input_ids.dtype == ir.DataType.INT64
    assert len(input_ids.shape) == 2

    # Exactly one output: logits [B, S, V] float.
    assert [value.name for value in graph.outputs] == ["logits"]
    logits = graph.outputs[0]
    assert logits.dtype == ir.DataType.FLOAT
    assert len(logits.shape) == 3
    assert logits.shape[2] == config.vocab_size

    # No KV-cache ports on either side.
    io_names = [value.name for value in (*graph.inputs, *graph.outputs)]
    assert not any(
        token in name for name in io_names for token in ("past", "present", "cache")
    )

    # The emitted metadata must wire those exact ports into a masked-diffusion
    # iterative loop with a logits -> input_ids self-edge.
    meta = build_language_diffusion_pipeline_metadata(
        mask_token_id=126336,
        num_inference_steps=8,
        input_ids_port="input_ids",
        logits_port="logits",
    )
    pipeline = meta["pipeline"]
    assert pipeline["models"]["denoiser"]["type"] == "denoiser"
    assert pipeline["dataflow"] == [{"from": "denoiser.logits", "to": "denoiser.input_ids"}]
    strategy = pipeline["strategy"]
    assert strategy["kind"] == "iterative"
    assert strategy["denoiser"] == "denoiser"
    assert strategy["num_steps"] == 8
    assert strategy["scheduler_config"] == {
        "kind": "masked_diffusion",
        "mask_token_id": 126336,
    }
