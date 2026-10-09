# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit target and draft graph contracts for Lightning NextN/MTP."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import NemotronHConfig
from mobius._model_package import ModelPackage
from mobius.models.nemotron_h_mtp import NemotronHSpeculativeModel
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import _make_kv_cache_inputs, _register_kv_cache_outputs
from mobius.tasks._causal_lm import HybridCausalLMTask


class NemotronHMtpTask(ModelTask):
    """Build target ``decoder`` with ``mtp_seed`` and independent ``mtp`` graph.

    The caller pairs target h_i with t_(i+1), initializes/advances draft KV
    independently, verifies draft tokens against the target, and restores
    rejected target SSM/conv/KV and draft KV state. No runtime ABI is implied.
    """

    components = ComponentSpec(decoder="decoder", mtp="mtp")

    def build(self, module: nn.Module, config: NemotronHConfig) -> ModelPackage:
        self._validate_components(module)
        if not isinstance(module, NemotronHSpeculativeModel):
            raise TypeError("NemotronHMtpTask requires NemotronHSpeculativeModel")
        target = HybridCausalLMTask().build(module.decoder, config)["model"]
        graph, builder = _make_graph("lightning_mtp")
        batch = ir.SymbolicDim("batch")
        input_ids = builder.input(
            "input_ids", dtype=ir.DataType.INT64, shape=["batch", "sequence_len"]
        )
        hidden = builder.input(
            "hidden_states",
            dtype=config.dtype,
            shape=["batch", "sequence_len", config.hidden_size],
        )
        mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=["batch", "past_sequence_len + sequence_len"],
        )
        past = _make_kv_cache_inputs(
            builder,
            1,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            ir.SymbolicDim("past_sequence_len"),
        )
        logits, mtp_hidden, present = module.mtp(builder.op, input_ids, hidden, mask, past[0])
        builder.add_output(logits, "logits")
        builder.add_output(mtp_hidden, "mtp_hidden")
        _register_kv_cache_outputs(
            builder,
            [present],
            batch=batch,
            num_kv_heads=config.num_key_value_heads,
            key_head_dim=config.head_dim,
            value_head_dim=config.head_dim,
            total_seq_len="past_sequence_len + sequence_len",
            dtype=config.dtype,
        )
        mtp = _make_model(graph)
        for model in (target, mtp):
            model.metadata_props["mobius.export_variant"] = "target-with-mtp"
            model.metadata_props["mobius.mtp_contract"] = "nemotron-h-lightning-nextn@1"
            model.metadata_props["mobius.mtp_inventory_status"] = "not_inspected"
            model.metadata_props["mobius.runtime_support"] = (
                "Standalone graphs only: requires caller-managed token/hidden alignment, "
                "draft KV, target verification and hybrid-state rollback; not an "
                "ORT GenAI speculative-generation ABI."
            )
        return ModelPackage({"decoder": target, "mtp": mtp}, config=config)
