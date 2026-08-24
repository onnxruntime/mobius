# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph task for GLM-5.2 (``glm_moe_dsa``): DSA's packed IndexShare KV cache."""

from __future__ import annotations

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._causal_lm import CausalLMTask


class GlmMoeDsaTask(ModelTask):
    """Causal LM build for GLM-5.2, aware of DSA's packed per-layer KV cache.

    ``GlmMoeDsaAttention`` (the DSA/IndexShare attention path, default when
    ``config.use_dsa=True``) packs the indexer's own key cache into the
    *same* present-KV tensor as the main attention -- one packed "head" of
    width ``main_key_dim`` (plus ``index_head_dim`` extra columns on layers
    whose indexer is "full" rather than "shared") -- instead of the generic
    per-head ``[batch, num_heads, seq, head_dim]`` convention every other
    registered causal-LM task assumes. That per-layer-varying width can't be
    expressed by the generic :class:`~mobius.tasks._causal_lm.CausalLMTask`
    (uniform shape across layers), so DSA mode needs its own cache
    declaration, read directly off the built module via
    ``GlmMoeDsaCausalLMModel.dsa_kv_cache_specs()`` to guarantee it never
    drifts from what ``_pack_present``/``_unpack_past`` actually produce.

    ``config.use_dsa=False`` (the ``--glm-full-attention`` fallback) builds
    a plain dense ``DeepSeekV3TextModel`` with the standard MLA present-KV
    convention, which ``CausalLMTask`` already declares correctly -- so
    that mode delegates to it unchanged rather than duplicating its
    input/output wiring.
    """

    def build(self, module, config: ArchitectureConfig) -> ModelPackage:
        if not config.use_dsa:
            return CausalLMTask().build(module, config)
        return self._build_dsa(module, config)

    def _build_dsa(self, module, config: ArchitectureConfig) -> ModelPackage:
        specs_fn = getattr(module, "dsa_kv_cache_specs", None)
        if not callable(specs_fn):
            raise TypeError(
                f"{type(module).__name__} must implement dsa_kv_cache_specs() "
                "to build with GlmMoeDsaTask in DSA mode."
            )
        cache_specs = specs_fn()

        graph, builder = _make_graph("glm_moe_dsa")
        op = builder.op

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_sequence_len + sequence_len"],
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )

        past_key_values = []
        for i, (key_head_dim, value_head_dim) in enumerate(cache_specs):
            past_key = builder.input(
                f"past_key_values.{i}.key",
                dtype=config.dtype,
                shape=[batch, 1, past_seq_len, key_head_dim],
            )
            past_value = builder.input(
                f"past_key_values.{i}.value",
                dtype=config.dtype,
                shape=[batch, 1, past_seq_len, value_head_dim],
            )
            past_key_values.append((past_key, past_value))

        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        builder.add_output(logits, "logits")

        # Stamp explicit present shapes/dtypes: ``pkg.nxrt::IndexShare`` has
        # no registered symbolic-shape-inference function (unlike the plain
        # ONNX ``Attention``/``com.microsoft::GroupQueryAttention`` ops), so
        # the present.* outputs would otherwise be left untyped.
        total_seq_len = "past_sequence_len + sequence_len"
        for i, (present_key, present_value) in enumerate(present_key_values):
            key_head_dim, value_head_dim = cache_specs[i]
            present_key.shape = ir.Shape([batch, 1, total_seq_len, key_head_dim])
            present_key.type = ir.TensorType(config.dtype)
            present_value.shape = ir.Shape([batch, 1, total_seq_len, value_head_dim])
            present_value.type = ir.TensorType(config.dtype)
            builder.add_output(present_key, f"present.{i}.key")
            builder.add_output(present_value, f"present.{i}.value")

        return ModelPackage({"model": _make_model(graph)}, config=config)
