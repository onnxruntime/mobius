# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned-source CUDA parity for the Phi-4 Flash SambaY cache contract.

This test intentionally runs only in the scoped CUDA reference environment:
the pinned implementation force-casts differential-attention Q/K/V to BF16,
which CPU and DirectML ONNX Runtime builds cannot execute. It compares a
batch-two, local-window-overflow prefill and two decode steps against the exact
``trust_remote_code`` implementation, including every externally visible cache
slot and each post-layer residual stream. The tiny model has eight layers so it
executes the producer and both YOCO shared-memory/shared-KV consumers.
"""

from __future__ import annotations

import importlib
import os
from importlib.metadata import version
from typing import Any

import ml_dtypes
import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius._configs import Phi4FlashConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.phi4flash import Phi4FlashCausalLMModel
from mobius.tasks import Phi4FlashCausalLMTask

_MODEL_ID = "microsoft/Phi-4-mini-flash-reasoning"
_REVISION = "1dff8163d28ec880ca2411c474ddc0a927792810"
_REQUIRED_DISTRIBUTIONS = {
    "torch": "2.6.0",
    "transformers": "4.46.1",
    "accelerate": "1.4.0",
    "flash-attn": "2.7.4.post1",
    "mamba-ssm": "2.2.4",
    "causal-conv1d": "1.5.0.post8",
}


def _require_pinned_cuda_reference() -> None:
    """Fail on version drift instead of silently comparing a different source ABI."""
    if os.environ.get("MOBIUS_PHI4FLASH_REFERENCE") != "1":
        pytest.skip("set MOBIUS_PHI4FLASH_REFERENCE=1 in the pinned CUDA reference job")
    installed = {distribution: version(distribution) for distribution in _REQUIRED_DISTRIBUTIONS}
    assert installed == _REQUIRED_DISTRIBUTIONS
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers() or not torch.cuda.is_available():
        pytest.skip("Phi-4 Flash parity requires CUDA BF16 Attention and Scan kernels")


def _small_remote_config() -> Any:
    """Load the pinned custom config class and build producer/consumer SambaY layers."""
    from transformers import AutoConfig

    remote = AutoConfig.from_pretrained(_MODEL_ID, revision=_REVISION, trust_remote_code=True)
    config = type(remote)(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_d_state=4,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=2,
        sliding_window=4,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "flash_attention_2"
    config.torch_dtype = torch.bfloat16
    return config


def _new_remote_cache(reference_model: Any, config: Any, batch: int, max_length: int) -> Any:
    """Instantiate the source's fixed-capacity cache so decode can grow past prefill."""
    module = importlib.import_module(type(reference_model.model).__module__)
    cache_type = getattr(module, "SambaYCache")
    return cache_type(
        config,
        max_batch_size=batch,
        max_cache_len=max_length,
        device="cuda",
        dtype=torch.bfloat16,
    )


def _empty_onnx_cache(config: Phi4FlashConfig, batch: int) -> dict[str, np.ndarray]:
    """Create explicit zero states for all physical source cache slots."""
    states: dict[str, np.ndarray] = {}
    d_inner = config.hidden_size * config.mamba_expand
    for slot, layer_type in enumerate(config.layer_types[: config.cache_slot_count]):
        if layer_type in {"mamba", "shared_memory_mamba"}:
            states[f"past_key_values.{slot}.conv_state"] = np.zeros(
                (batch, d_inner, config.mamba_d_conv), dtype=ml_dtypes.bfloat16
            )
            states[f"past_key_values.{slot}.ssm_state"] = np.zeros(
                (batch, d_inner, config.mamba_d_state), dtype=ml_dtypes.bfloat16
            )
        else:
            states[f"past_key_values.{slot}.key"] = np.zeros(
                (batch, config.num_key_value_heads, 0, config.head_dim),
                dtype=ml_dtypes.bfloat16,
            )
            states[f"past_key_values.{slot}.value"] = np.zeros(
                (batch, config.num_key_value_heads, 0, config.head_dim),
                dtype=ml_dtypes.bfloat16,
            )
    return states


def _present_to_past(config: Phi4FlashConfig, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Chain all dynamic state classes without manufacturing static cache padding."""
    result: dict[str, np.ndarray] = {}
    for slot, layer_type in enumerate(config.layer_types[: config.cache_slot_count]):
        names = ("conv_state", "ssm_state") if "mamba" in layer_type else ("key", "value")
        for name in names:
            result[f"past_key_values.{slot}.{name}"] = outputs[f"present.{slot}.{name}"]
    return result


def _to_float(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _assert_cache_matches(
    config: Phi4FlashConfig,
    cache: Any,
    outputs: dict[str, np.ndarray],
    sequence_length: int,
) -> None:
    """Compare source state content, trimming only its unused static allocation tail."""
    for slot, layer_type in enumerate(config.layer_types[: config.cache_slot_count]):
        if "mamba" in layer_type:
            np.testing.assert_allclose(
                _to_float(outputs[f"present.{slot}.conv_state"]),
                _to_float(cache.key_cache[slot]),
                rtol=0,
                atol=2e-2,
            )
            np.testing.assert_allclose(
                _to_float(outputs[f"present.{slot}.ssm_state"]),
                _to_float(cache.value_cache[slot]),
                rtol=0,
                atol=2e-2,
            )
            continue

        source_key = cache.key_cache[slot]
        source_value = cache.value_cache[slot]
        # The source's global cache is statically allocated to the full test
        # length, while Mobius exposes only the chronological prefix.
        if layer_type == "global_differential_attention":
            source_key = source_key[:, :, :sequence_length, :]
            source_value = source_value[:, :, :sequence_length, :]
        np.testing.assert_allclose(
            _to_float(outputs[f"present.{slot}.key"]), _to_float(source_key), rtol=0, atol=2e-2
        )
        np.testing.assert_allclose(
            _to_float(outputs[f"present.{slot}.value"]),
            _to_float(source_value),
            rtol=0,
            atol=2e-2,
        )


@pytest.mark.integration
class TestPhi4FlashPinnedSourceParity:
    """L3: tiny, source-exact numerical parity over a full stateful SambaY cycle."""

    def test_prefill_and_decode_match_all_states_and_post_layer_residuals(self) -> None:
        _require_pinned_cuda_reference()
        torch.manual_seed(7)
        reference_config = _small_remote_config()
        reference_module = importlib.import_module(
            type(reference_config).__module__.replace(
                "configuration_phi4flash", "modeling_phi4flash"
            )
        )
        reference_class = getattr(reference_module, "Phi4FlashForCausalLM")
        reference = reference_class(reference_config).cuda().bfloat16().eval()

        config = Phi4FlashConfig.from_transformers(reference_config)
        config.dtype = ir.DataType.BFLOAT16
        config.output_layer_indices = list(range(reference_config.num_hidden_layers))
        exported = Phi4FlashCausalLMModel(config)
        package = build_from_module(exported, config, task=Phi4FlashCausalLMTask())
        package.apply_weights(
            exported.preprocess_weights(
                {name: tensor.detach().cpu() for name, tensor in reference.state_dict().items()}
            )
        )
        session = OnnxModelSession(package["model"], device="cuda")
        assert session.providers[0] == "CUDAExecutionProvider"

        batch, prompt_length, decode_steps = 2, 5, 2
        cache = _new_remote_cache(reference, reference_config, batch, prompt_length + decode_steps)
        states = _empty_onnx_cache(config, batch)
        input_ids = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=np.int64)
        try:
            for position in range(decode_steps + 1):
                tokens = input_ids if position == 0 else np.full((batch, 1), 6 + position, np.int64)
                full_length = prompt_length + position
                with torch.inference_mode():
                    reference_output = reference(
                        input_ids=torch.from_numpy(tokens).cuda(),
                        attention_mask=torch.ones((batch, full_length), device="cuda", dtype=torch.long),
                        past_key_values=cache,
                        cache_position=torch.arange(
                            full_length - tokens.shape[1], full_length, device="cuda"
                        ),
                        use_cache=True,
                        output_hidden_states=True,
                    )
                feeds = {
                    "input_ids": tokens,
                    "attention_mask": np.ones((batch, full_length), dtype=np.int64),
                    **states,
                }
                outputs = session.run(feeds)
                np.testing.assert_allclose(
                    _to_float(outputs["logits"]),
                    _to_float(reference_output.logits),
                    rtol=0,
                    atol=3e-2,
                )
                for layer_idx in config.output_layer_indices:
                    np.testing.assert_allclose(
                        _to_float(outputs[f"hidden_states.{layer_idx}"]),
                        _to_float(reference_output.hidden_states[layer_idx + 1]),
                        rtol=0,
                        atol=3e-2,
                    )
                _assert_cache_matches(config, cache, outputs, full_length)
                states = _present_to_past(config, outputs)
        finally:
            session.close()
