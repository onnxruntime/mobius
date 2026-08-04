# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for model building — base class and infrastructure unit tests.

Architecture-specific graph construction tests live in tests/build_graph_test.py,
which covers every registered model type via ALL_CAUSAL_LM_CONFIGS.  This file
focuses on the base class contracts and build infrastructure that are not
exercised by the parametrized tests there.
"""

from __future__ import annotations

import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._registry import (
    MODEL_MAP,
    ModelRegistry,
    registry,
)
from mobius._testing import make_config
from mobius.models.base import CausalLMModel, TextModel
from mobius.tasks import CausalLMTask


class TestTextModel:
    def test_text_model_params(self):
        config = make_config()
        model = TextModel(config)
        param_names = [n for n, _ in model.named_parameters()]
        assert any("embed_tokens" in n for n in param_names)
        assert any("norm" in n for n in param_names)
        assert any("layers" in n for n in param_names)

    def test_text_model_num_layers(self):
        config = make_config(num_hidden_layers=4)
        model = TextModel(config)
        assert len(model.layers) == 4


class TestCausalLMModel:
    def test_causal_lm_model_has_lm_head(self):
        config = make_config()
        model = CausalLMModel(config)
        param_names = [n for n, _ in model.named_parameters()]
        assert any("lm_head" in n for n in param_names)

    def test_preprocess_weights_tied_embeddings(self):
        config = make_config(tie_word_embeddings=True)
        model = CausalLMModel(config)

        weight = torch.zeros(100, 64)
        sd = {"lm_head.weight": weight}
        sd = model.preprocess_weights(sd)
        assert "model.embed_tokens.weight" in sd
        assert sd["model.embed_tokens.weight"] is weight

    def test_preprocess_weights_no_tied(self):
        config = make_config(tie_word_embeddings=False)
        model = CausalLMModel(config)
        sd = {"lm_head.weight": torch.zeros(100, 64)}
        sd = model.preprocess_weights(sd)
        assert "model.embed_tokens.weight" not in sd


class TestBuildFromModule:
    def test_build_base_model(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        assert isinstance(model, ir.Model)
        assert model.graph.num_nodes() > 0
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

    def test_build_model_inputs(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        input_names = [v.name for v in model.graph.inputs]
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.0.value" in input_names

    def test_build_model_outputs(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        output_names = [v.name for v in model.graph.outputs]
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names

    def test_build_model_num_kv_caches(self):
        config = make_config(num_hidden_layers=3)
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        output_names = [v.name for v in model.graph.outputs]
        for i in range(3):
            assert f"present.{i}.key" in output_names
            assert f"present.{i}.value" in output_names

    def test_build_model_has_initializers(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        init_names = list(model.graph.initializers.keys())
        assert len(init_names) > 0
        assert any("embed_tokens" in n for n in init_names)
        assert any("lm_head" in n for n in init_names)

    def test_build_model_save_load_roundtrip(self, tmp_path):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        path = str(tmp_path / "test_roundtrip.onnx")
        ir.save(model, path)
        loaded = ir.load(path)
        assert loaded.graph.num_nodes() == model.graph.num_nodes()

    def test_build_with_task_instance(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config, task=CausalLMTask())["model"]
        assert isinstance(model, ir.Model)
        assert model.graph.num_nodes() > 0

    def test_build_with_task_string(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config, task="text-generation")["model"]
        assert isinstance(model, ir.Model)
        assert model.graph.num_nodes() > 0

    def test_build_with_output_layer_indices(self):
        config = make_config(num_hidden_layers=4, output_layer_indices=[1, 2])
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        output_names = [v.name for v in model.graph.outputs]
        assert "logits" in output_names
        assert "hidden_states.1" in output_names
        assert "hidden_states.2" in output_names
        assert "hidden_states.0" not in output_names
        assert "hidden_states.3" not in output_names

    def test_build_without_output_layer_indices_unchanged(self):
        # Default (None) must preserve the legacy 2-tuple output set:
        # logits + present.{i}.key/value only, no hidden_states.* outputs.
        config = make_config(num_hidden_layers=2)
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        output_names = [v.name for v in model.graph.outputs]
        assert not any(n.startswith("hidden_states.") for n in output_names)

    def test_build_output_layer_indices_preserves_order(self):
        # Caller-supplied order is preserved in the graph outputs, so a
        # downstream draft model can zip(indices, outputs) without sorting.
        config = make_config(num_hidden_layers=5, output_layer_indices=[3, 0, 2])
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        hs_outputs = [
            v.name for v in model.graph.outputs if v.name.startswith("hidden_states.")
        ]
        assert hs_outputs == ["hidden_states.3", "hidden_states.0", "hidden_states.2"]


class TestTextModelOutputHiddenStates:
    def test_textmodel_output_layer_indices_default_none(self):
        config = make_config()
        model = TextModel(config)
        assert model.output_layer_indices is None

    def test_textmodel_output_layer_indices_set(self):
        config = make_config(num_hidden_layers=4, output_layer_indices=[0, 3])
        model = TextModel(config)
        assert model.output_layer_indices == [0, 3]


class TestDeepStackCaptureOrdering:
    """``output_layer_indices`` must capture the post-DeepStack-injection state.

    Regression for the ordering bug where intermediate hidden states were
    captured *before* the DeepStack ``Add``, so requested layers within the
    DeepStack range missed the injected vision contribution (and diverged from
    HuggingFace ``output_hidden_states`` semantics, where ``hidden_states[k+1]``
    is layer ``k``'s output with DeepStack already added).
    """

    def test_intermediate_capture_reflects_deepstack_injection(self):
        from mobius.tasks._base import _make_graph

        # 3 layers, all captured; DeepStack embeds cover only layers 0 and 1.
        config = make_config(num_hidden_layers=3, output_layer_indices=[0, 1, 2])
        model = TextModel(config)

        _graph, builder = _make_graph()
        op = builder.op
        batch = ir.SymbolicDim("batch")
        seq = ir.SymbolicDim("sequence_len")
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq])
        attention_mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, seq]
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq]
        )
        deepstack = [
            builder.input(
                f"deepstack_embeds.{i}",
                dtype=config.dtype,
                shape=[batch, seq, config.hidden_size],
            )
            for i in range(2)
        ]

        _hidden, _present, intermediates = model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deepstack_embeds=deepstack,
        )
        captured = dict(zip([0, 1, 2], intermediates))

        # Layers 0 and 1: the captured value is the output of the DeepStack Add,
        # whose inputs include the corresponding deepstack_embeds tensor.
        for layer_idx in (0, 1):
            producer = captured[layer_idx].producer()
            assert producer is not None and producer.op_type == "Add"
            assert deepstack[layer_idx] in list(producer.inputs)

        # Layer 2 has no DeepStack embed: its captured value is the raw layer
        # output and must not consume any deepstack_embeds tensor.
        prod2 = captured[2].producer()
        assert prod2 is None or all(d not in list(prod2.inputs) for d in deepstack)


class TestModelRegistry:
    def test_registry_not_empty(self):
        assert len(registry) > 0

    def test_registry_has_architectures(self):
        assert len(registry) >= 50

    def test_all_registry_values_are_callable(self):
        for name in registry.architectures():
            cls = registry.get(name)
            assert callable(cls), f"registry['{name}'] is not callable"

    def test_register_custom_architecture(self):
        reg = ModelRegistry()
        reg.register("test_arch", CausalLMModel)
        assert "test_arch" in reg
        assert reg.get("test_arch") is CausalLMModel

    def test_get_unknown_raises(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError, match="Unknown model_type"):
            reg.get("nonexistent")

    def test_model_map_backward_compat(self):
        """MODEL_MAP dict still works for backward compatibility."""
        assert len(MODEL_MAP) >= 50
        assert "llama" in MODEL_MAP
