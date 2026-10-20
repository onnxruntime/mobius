# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import ml_dtypes
import numpy as np
import pytest
import torch
import transformers

from mobius._testing import torch_reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["scalar", "empty", "transposed", "strided"])
def test_tensor_numpy_bridge_is_lossless(dtype, layout):
    value = torch.tensor([0.0, -0.0, 1.25, -2.5, float("inf"), float("nan")], dtype=dtype)
    if layout == "scalar":
        value = value[2]
    elif layout == "empty":
        value = value[:0].reshape(0, 2)
    elif layout == "transposed":
        value = value.reshape(2, 3).T
    else:
        value = value[::2]
    array = torch_reference._tensor_to_numpy(value)
    expected_dtype = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: ml_dtypes.bfloat16,
    }[dtype]
    assert array.dtype == np.dtype(expected_dtype)
    assert array.shape == tuple(value.shape)
    restored = torch_reference._numpy_to_tensor(array)
    assert restored.dtype == dtype
    assert restored.shape == value.shape
    # Compare raw bytes, including signed zero and NaN payloads.
    assert restored.contiguous().reshape(-1).view(torch.uint8).tolist() == (
        value.contiguous().reshape(-1).view(torch.uint8).tolist()
    )
    if array.ndim == 2:
        roundtrip = torch_reference._numpy_to_tensor(array.T)
        torch.testing.assert_close(roundtrip, value.T, equal_nan=True)


class _ReferenceModel(torch.nn.Module):
    """A real-tensor reference double that records exact forward inputs."""

    def __init__(self, dtype, output, cache=None, layer_types=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))
        self.output = output
        self.cache = cache
        self.config = SimpleNamespace(layer_types=layer_types)
        self.calls = []

    def forward(self, position_ids=None, **kwargs):
        self.calls.append({"position_ids": position_ids, **kwargs})
        return SimpleNamespace(
            logits=self.output, last_hidden_state=self.output, past_key_values=self.cache
        )

    def get_encoder(self):
        return self


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("legacy_output", [False, True])
def test_causal_logits_and_mixed_dtype_kv_roundtrip(dtype, legacy_output):
    from transformers.cache_utils import DynamicCache

    key = torch.tensor([[[[1.25, -2.5]]]], dtype=dtype)
    value = torch.tensor([[[[1.0001, -2.0002]]]], dtype=torch.float32)
    cache = DynamicCache()
    cache.update(key, value, 0)
    output = torch.tensor([[[1.25, -2.5, 0.0]]], dtype=dtype)
    model = _ReferenceModel(dtype, output, ((key, value),) if legacy_output else cache)
    ids = np.array([[1]], dtype=np.int64)
    mask = np.array([[True]], dtype=np.bool_)
    logits, kv = torch_reference.torch_forward(model, ids, mask, ids)
    torch.testing.assert_close(torch_reference._numpy_to_tensor(logits), output)
    assert kv[0][1].dtype == np.float32
    torch_reference.torch_forward(model, ids, mask, ids, kv)
    reused = model.calls[-1]["past_key_values"].layers[0]
    torch.testing.assert_close(reused.keys, key)
    torch.testing.assert_close(reused.values, value)
    assert reused.values.dtype == torch.float32
    assert model.calls[-1]["input_ids"].dtype == torch.int64
    assert model.calls[-1]["attention_mask"].dtype == torch.bool


@pytest.mark.parametrize("layer_types", [["conv"], ["mamba2"], None])
def test_opaque_recurrent_cache_identity(layer_types):
    from transformers.cache_utils import Cache

    class RecurrentCache(Cache):
        def __init__(self):
            self.layers = [SimpleNamespace(ssm_state=torch.ones(1, dtype=torch.float32))]

    cache = RecurrentCache()
    model = _ReferenceModel(
        torch.bfloat16, torch.ones((1, 1, 3), dtype=torch.bfloat16), cache, layer_types
    )
    ids = np.array([[1]], dtype=np.int64)
    _, returned = torch_reference.torch_forward(model, ids, ids, ids, cache)
    assert returned is cache
    assert model.calls[-1]["past_key_values"] is cache
    assert cache.layers[0].ssm_state.dtype == torch.float32


@pytest.mark.parametrize("iterable_cache", [False, True])
def test_non_transformers_opaque_cache_survives_prefill_and_decode(iterable_cache):
    class IterableCache:
        def __init__(self):
            self.ssm_state = torch.ones(1, dtype=torch.float32)

        def __iter__(self):
            raise AssertionError("Opaque cache must not be enumerated")

    cache = (
        IterableCache()
        if iterable_cache
        else SimpleNamespace(ssm_state=torch.ones(1, dtype=torch.float32))
    )
    state = cache.ssm_state
    output = torch.ones((1, 1, 3), dtype=torch.bfloat16)
    model = _ReferenceModel(torch.bfloat16, output, cache)
    ids = np.array([[1]], dtype=np.int64)

    _, prefill_cache = torch_reference.torch_forward(model, ids, ids, ids)
    _, decode_cache = torch_reference.torch_forward(model, ids, ids, ids, prefill_cache)

    assert prefill_cache is cache
    assert decode_cache is cache
    assert model.calls[-1]["past_key_values"] is cache
    assert cache.ssm_state is state
    assert state.dtype == torch.float32
    assert len(model.calls) == 2


@pytest.mark.parametrize("invalid_attribute", ["keys", "values"])
@pytest.mark.parametrize("invalid_value", [None, "opaque-state"])
def test_cache_with_non_tensor_layer_state_remains_opaque(invalid_attribute, invalid_value):
    layer = SimpleNamespace(
        keys=torch.ones((1, 1, 1, 2), dtype=torch.bfloat16),
        values=torch.ones((1, 1, 1, 2), dtype=torch.float32),
    )
    setattr(layer, invalid_attribute, invalid_value)
    cache = SimpleNamespace(layers=[layer])
    model = _ReferenceModel(torch.bfloat16, torch.ones((1, 1, 3)), cache)
    ids = np.array([[1]], dtype=np.int64)

    _, prefill_cache = torch_reference.torch_forward(model, ids, ids, ids)
    _, decode_cache = torch_reference.torch_forward(model, ids, ids, ids, prefill_cache)

    assert prefill_cache is cache
    assert decode_cache is cache
    assert model.calls[-1]["past_key_values"] is cache


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "wrapper",
    [
        "torch_encoder_forward",
        "torch_seq2seq_encoder_forward",
        "torch_seq2seq_decoder_forward",
        "torch_vision_forward",
        "torch_audio_forward",
        "torch_whisper_encoder_forward",
    ],
)
def test_ordinary_forward_outputs_and_media_dtypes(dtype, wrapper):
    output = torch.tensor([[[1.25, -2.5]]], dtype=dtype)
    model = _ReferenceModel(dtype, output)
    ids = np.array([[1]], dtype=np.int64)
    mask = np.array([[True]], dtype=np.bool_)
    media = torch_reference._tensor_to_numpy(output)
    forward = getattr(torch_reference, wrapper)
    if wrapper == "torch_encoder_forward":
        result = forward(model, ids, mask, ids)
    elif wrapper == "torch_seq2seq_encoder_forward":
        result = forward(model, ids, mask)
    elif wrapper == "torch_seq2seq_decoder_forward":
        result, _ = forward(model, ids, media, mask)
        assert model.calls[-1]["encoder_outputs"][0].dtype == dtype
    elif wrapper == "torch_whisper_encoder_forward":
        model.model = SimpleNamespace(encoder=lambda feats: model(input_features=feats))
        result = forward(model, media)
    else:
        result = forward(model, media)
    torch.testing.assert_close(torch_reference._numpy_to_tensor(result), output)
    for name, value in model.calls[-1].items():
        if name in ("input_ids", "decoder_input_ids", "token_type_ids"):
            assert value.dtype == torch.int64
        elif name == "attention_mask":
            assert value.dtype == torch.bool
        elif name in ("pixel_values", "input_values", "input_features"):
            assert value.dtype == dtype


@pytest.mark.parametrize("wrapper", ["multimodal", "whisper"])
def test_bfloat16_media_decoder_logits_and_kv_reuse(wrapper):
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    output = torch.tensor([[[1.25, -2.5]]], dtype=torch.bfloat16)
    key = output.reshape(1, 1, 1, 2)
    value = key.float()
    cache = DynamicCache()
    cache.update(key, value, 0)
    model = _ReferenceModel(torch.bfloat16, output, cache)
    ids = np.array([[1]], dtype=np.int64)
    media = torch_reference._tensor_to_numpy(output)
    if wrapper == "multimodal":

        def forward(past=None):
            return torch_reference.torch_multimodal_forward(model, ids, ids, ids, media, past)
    else:
        model.cache = EncoderDecoderCache(cache, DynamicCache())
        model.model = SimpleNamespace(decoder=model)
        model.proj_out = lambda hidden: hidden

        def forward(past=None):
            return torch_reference.torch_whisper_decoder_forward(model, ids, media, ids, past)

    logits, kv = forward()
    torch.testing.assert_close(torch_reference._numpy_to_tensor(logits), output)
    forward(kv)
    reused = model.calls[-1]["past_key_values"]
    if wrapper == "whisper":
        reused = reused.self_attention_cache
    torch.testing.assert_close(reused.layers[0].keys, key)
    torch.testing.assert_close(reused.layers[0].values, value)


def test_load_torch_model_pins_every_huggingface_resource() -> None:
    revision = "a" * 40
    tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="</s>")
    config = SimpleNamespace(model_type="llama")
    model = mock.Mock()

    with (
        mock.patch.object(torch_reference, "_install_dynamic_cache_legacy_shims"),
        mock.patch.object(
            transformers.AutoTokenizer,
            "from_pretrained",
            return_value=tokenizer,
        ) as load_tokenizer,
        mock.patch.object(
            transformers.AutoConfig,
            "from_pretrained",
            return_value=config,
        ) as load_config,
        mock.patch.object(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            return_value=model,
        ) as load_model,
        mock.patch.object(torch_reference, "_fix_nemotron_h_dt_bias") as fix_weights,
    ):
        loaded_model, loaded_tokenizer = torch_reference.load_torch_model(
            "owner/model",
            revision=revision,
        )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    load_tokenizer.assert_called_once_with(
        "owner/model",
        revision=revision,
        trust_remote_code=True,
    )
    load_config.assert_called_once_with(
        "owner/model",
        revision=revision,
        trust_remote_code=True,
    )
    load_model.assert_called_once_with(
        "owner/model",
        config=config,
        dtype=torch.float32,
        device_map="cpu",
        revision=revision,
        trust_remote_code=True,
    )
    fix_weights.assert_called_once_with(model, "owner/model", revision)
    model.eval.assert_called_once_with()
