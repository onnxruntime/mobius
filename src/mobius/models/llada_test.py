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

import dataclasses
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mobius._configs import ArchitectureConfig
from mobius.integrations._weight_loading import apply_weights
from mobius.models.llada import (
    DreamConfig,
    DreamModel,
    LLaDAModel,
    LLaDAMoEConfig,
    LLaDAMoEModel,
    RND1Config,
    RND1Model,
)
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


def _official_hf_config(model_type: str, **overrides):
    values = {
        "model_type": model_type,
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_official_transformers_aliases_and_config_fields():
    from mobius.integrations.transformers._builder import _resolve_module_class

    dream = DreamConfig.from_transformers(_official_hf_config("Dream", mask_token_id=255))
    assert dream.attn_qkv_bias is True
    assert dream.mask_token_id == 255
    assert dream.diffusion_shift_logits is True

    llada_moe = LLaDAMoEConfig.from_transformers(
        _official_hf_config(
            "llada",
            architectures=["LLaDAMoEModel"],
            num_experts=8,
            num_experts_per_tok=2,
            expert_intermediate_size=32,
            qk_layernorm=True,
            norm_topk_prob=None,
        )
    )
    assert llada_moe.moe_intermediate_size == 32
    assert llada_moe.attn_qk_norm is True
    assert llada_moe.norm_topk_prob is False

    rnd1 = RND1Config.from_transformers(
        _official_hf_config(
            "rnd1",
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
        )
    )
    assert rnd1.attn_qk_norm is True
    assert rnd1.norm_topk_prob is True

    module_class, _, resolved_type = _resolve_module_class(
        "llada",
        SimpleNamespace(architectures=["LLaDAMoEModel"]),
        None,
        None,
    )
    assert module_class is LLaDAMoEModel
    assert resolved_type == "LLaDAMoEModel"

    module_class, _, resolved_type = _resolve_module_class(
        "Dream",
        SimpleNamespace(architectures=["DreamModel"]),
        None,
        None,
    )
    assert module_class is DreamModel
    assert resolved_type == "Dream"


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

    # Windows keeps the ORT model file mapped; ignore cleanup errors so the dir can be removed.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        model_path = Path(temp_dir) / "model.onnx"
        onnx_ir.save(model, model_path)
        return ort.InferenceSession(model_path)


def test_llada_matches_torch_reference():
    """ONNX logits match the self-contained torch reference to < 1e-4."""
    pytest.importorskip("onnxruntime")

    config = _make_config()
    state = _random_llada_weights(config)

    input_ids = torch.tensor([[3, 7, 1, 9, 4, 2]], dtype=torch.int64)
    with torch.no_grad():
        expected = _reference_logits(config, state, input_ids).numpy()

    session = _build_onnx_session(config, state)
    actual, proposed = session.run(None, {"input_ids": input_ids.numpy()})

    max_delta = np.abs(actual - expected).max()
    assert max_delta < 1e-4, f"max|Δ|={max_delta}"
    np.testing.assert_array_equal(proposed, np.argmax(actual, axis=-1))


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


def test_shifted_diffusion_logits_follow_llama_cpp_alignment():
    config = _make_config()
    state = _random_llada_weights(config)
    unshifted = _build_onnx_session(config, state)
    shifted = _build_onnx_session(
        dataclasses.replace(config, diffusion_shift_logits=True),
        state,
    )
    input_ids = np.array([[3, 7, 1, 9, 4, 2]], dtype=np.int64)

    raw_logits = unshifted.run(None, {"input_ids": input_ids})[0]
    shifted_logits = shifted.run(None, {"input_ids": input_ids})[0]

    np.testing.assert_array_equal(shifted_logits[:, 0], raw_logits[:, 0])
    np.testing.assert_array_equal(shifted_logits[:, 1:], raw_logits[:, :-1])


@pytest.mark.parametrize(
    ("model_class", "norm_topk_prob"),
    [(LLaDAMoEModel, False), (RND1Model, True)],
)
def test_diffusion_moe_graph_executes_without_cache(model_class, norm_topk_prob):
    import onnx_ir as ir
    import onnxruntime as ort

    from mobius.rewrite_rules._testing_utils import fill_random_weights

    config = dataclasses.replace(
        _make_config(),
        num_hidden_layers=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        attn_qk_norm=True,
        norm_topk_prob=norm_topk_prob,
    )
    module = model_class(config)
    model = MaskedDiffusionTask().build(module, config)["model"]
    fill_random_weights(model)

    assert module.model.layers[0].mlp.gate.norm_topk_prob is norm_topk_prob
    assert [value.name for value in model.graph.inputs] == ["input_ids"]
    assert not any(
        token in value.name
        for value in (*model.graph.inputs, *model.graph.outputs)
        for token in ("past", "present", "cache")
    )

    session = ort.InferenceSession(
        ir.serde.serialize_model(model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    logits, proposed = session.run(
        None, {"input_ids": np.array([[3, 7, 1, 9]], dtype=np.int64)}
    )
    assert logits.shape == (1, 4, config.vocab_size)
    assert proposed.shape == (1, 4)
    assert np.isfinite(logits).all()


def test_llada_export_signature_matches_masked_diffusion_metadata():
    """The exported ONNX I/O matches the onnx-genai masked workflow contract.

    Builds a tiny LLaDA package and checks that the graph exposes the logits and
    executable proposal ports consumed by the generic SSA workflow.
    """
    import onnx_ir as ir

    from mobius.integrations.onnx_genai.workflow_metadata import (
        build_language_diffusion_pipeline_metadata,
    )

    config = _make_config()
    module = LLaDAModel(config)
    package = MaskedDiffusionTask().build(module, config)
    model = package["model"]
    graph = model.graph

    # Exactly one input: input_ids [B, S] int64.
    assert [value.name for value in graph.inputs] == ["input_ids"]
    input_ids = graph.inputs[0]
    assert input_ids.dtype == ir.DataType.INT64
    assert len(input_ids.shape) == 2

    assert [value.name for value in graph.outputs] == ["logits", "proposed_tokens"]
    logits = graph.outputs[0]
    assert logits.dtype == ir.DataType.FLOAT
    assert len(logits.shape) == 3
    assert logits.shape[2] == config.vocab_size
    proposed = graph.outputs[1]
    assert proposed.dtype == ir.DataType.INT64
    assert len(proposed.shape) == 2

    # No KV-cache ports on either side.
    io_names = [value.name for value in (*graph.inputs, *graph.outputs)]
    assert not any(
        token in name for name in io_names for token in ("past", "present", "cache")
    )

    meta = build_language_diffusion_pipeline_metadata(
        package,
        num_inference_steps=8,
    )
    workflow = meta["pipeline"]["workflow"]
    assert "strategy" not in meta["pipeline"]
    assert workflow["steps"][0]["kind"] == "loop"
    assert workflow["components"]["masked_update"]["contract"]["id"] == (
        "onnx-genai.masked-update"
    )
