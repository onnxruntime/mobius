# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import pytest
import torch

from mobius._registry import registry
from mobius._testing import make_config
from mobius.models.gpt2 import GPT2CausalLMModel, ScaledEmbeddingGPT2CausalLMModel
from mobius.tasks import CausalLMTask


@pytest.mark.parametrize("tied", [False, True])
def test_gpt2_embedding_parameter_ownership_follows_config(tied: bool) -> None:
    config = make_config(tie_word_embeddings=tied)
    module = GPT2CausalLMModel(config)

    assert (module.lm_head.weight is module.transformer.wte.weight) is tied

    graph = CausalLMTask().build(module, config)["model"]
    assert "transformer.wte.weight" in graph.graph.initializers
    assert ("lm_head.weight" not in graph.graph.initializers) is tied


@pytest.mark.parametrize("source_key", ["transformer.wte.weight", "lm_head.weight"])
def test_gpt2_tied_preprocessing_keeps_single_embedding_owner(source_key: str) -> None:
    config = make_config(tie_word_embeddings=True)
    module = GPT2CausalLMModel(config)
    weight = torch.arange(config.vocab_size * config.hidden_size, dtype=torch.float32).reshape(
        config.vocab_size, config.hidden_size
    )

    processed = module.preprocess_weights({source_key: weight})

    assert set(processed) == {"transformer.wte.weight"}
    assert processed["transformer.wte.weight"] is weight


def test_gpt2_untied_preprocessing_preserves_distinct_tables() -> None:
    config = make_config(tie_word_embeddings=False)
    module = GPT2CausalLMModel(config)
    embedding = torch.zeros(config.vocab_size, config.hidden_size)
    output = torch.ones_like(embedding)

    processed = module.preprocess_weights(
        {"transformer.wte.weight": embedding, "lm_head.weight": output}
    )

    assert processed["transformer.wte.weight"] is embedding
    assert processed["lm_head.weight"] is output


def test_starcoder_tied_graph_has_single_embedding_owner() -> None:
    config = make_config(tie_word_embeddings=True)
    module = registry.get("gpt_bigcode")(config)

    assert isinstance(module, GPT2CausalLMModel)
    assert module.lm_head.weight is module.transformer.wte.weight

    graph = CausalLMTask().build(module, config)["model"]
    assert "transformer.wte.weight" in graph.graph.initializers
    assert "lm_head.weight" not in graph.graph.initializers


@pytest.mark.parametrize(
    ("model_type", "source_key"),
    [("biogpt", "biogpt.embed_tokens.weight"), ("xglm", "model.embed_tokens.weight")],
)
def test_scaled_embedding_families_keep_unscaled_output_weight(
    model_type: str, source_key: str
) -> None:
    config = make_config(tie_word_embeddings=True)
    module = registry.get(model_type)(config)
    weight = torch.arange(config.vocab_size * config.hidden_size, dtype=torch.float32).reshape(
        config.vocab_size, config.hidden_size
    )

    assert isinstance(module, ScaledEmbeddingGPT2CausalLMModel)
    assert module.lm_head.weight is not module.transformer.wte.weight

    graph = CausalLMTask().build(module, config)["model"]
    assert "transformer.wte.weight" in graph.graph.initializers
    assert "lm_head.weight" in graph.graph.initializers

    processed = module.preprocess_weights({source_key: weight})
    torch.testing.assert_close(
        processed["transformer.wte.weight"], weight * config.hidden_size**0.5
    )
    assert processed["lm_head.weight"] is weight
