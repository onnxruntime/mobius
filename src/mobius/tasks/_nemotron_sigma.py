# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit dual-mode ONNX ABI for continuous Sigma self-speculation."""

from __future__ import annotations

import json

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.models.nemotron_sigma import NemotronSigmaConfig
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import _make_kv_cache_inputs, _register_kv_cache_outputs


class NemotronSigmaTask(ModelTask):
    """Build a shared-weight clean-encoding / continuous-denoising graph.

    ``denoise`` selects noisy embeddings instead of token embeddings. The bool
    mask is ``[B,1,L,past+L]`` and includes all padding and causal constraints.
    Prefix caches are explicit and immutable inputs; draft presents are discarded.
    Gamma and noise are DOUBLE, matching upstream sampling arithmetic.
    """

    def build(self, module, config: NemotronSigmaConfig) -> ModelPackage:
        config.validate_sigma()
        graph, builder = _make_graph()
        batch, length, past = (
            ir.SymbolicDim("batch"),
            ir.SymbolicDim("sequence"),
            ir.SymbolicDim("past"),
        )
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, length])
        z = builder.input(
            "z", dtype=ir.DataType.DOUBLE, shape=[batch, length, config.embed_dim]
        )
        gamma = builder.input("gamma", dtype=ir.DataType.DOUBLE, shape=[])
        condition = builder.input(
            "self_condition", dtype=ir.DataType.FLOAT, shape=[batch, length, config.embed_dim]
        )
        denoise = builder.input("denoise", dtype=ir.DataType.BOOL, shape=[])
        positions = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, length]
        )
        mask = builder.input(
            "attention_mask", dtype=ir.DataType.BOOL, shape=[batch, 1, length, "total"]
        )
        cache = _make_kv_cache_inputs(
            builder,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past,
        )
        logits, mean, ar_logits, bounds, present = module(
            builder.op, input_ids, z, gamma, condition, denoise, positions, mask, cache
        )
        builder.add_output(logits, "diffusion_logits")
        builder.add_output(mean, "denoised_mean")
        builder.add_output(ar_logits, "ar_logits")
        builder.add_output(bounds, "gamma_bounds")
        _register_kv_cache_outputs(
            builder,
            present,
            batch=batch,
            num_kv_heads=config.num_key_value_heads,
            key_head_dim=config.head_dim,
            value_head_dim=config.head_dim,
            total_seq_len="total",
            dtype=config.dtype,
        )
        model = _make_model(graph)
        model.metadata_props["mobius.sigma"] = json.dumps(
            {
                "version": 1,
                "embed_dim": config.embed_dim,
                "block_size": config.block_size,
                "num_layers": config.num_hidden_layers,
                "num_kv_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "max_context": config.max_position_embeddings,
                "native_nfe": 2,
            }
        )
        return ModelPackage({"model": model}, config=config)
