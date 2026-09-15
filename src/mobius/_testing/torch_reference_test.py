# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch
import transformers

from mobius._testing import torch_reference


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
