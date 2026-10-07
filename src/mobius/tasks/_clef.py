# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Full-record Clef decision export; no autoregressive cache is exposed."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.models.clef import ClefConfig
from mobius.tasks._base import (
    ComponentSpec,
    _make_graph,
    _make_model,
)
from mobius.tasks._cache_utils import (
    _register_linear_attention_functions,
    linear_attention_dims,
)
from mobius.tasks._vision_language_3model import Qwen2VLMultimediaTask


class ClefDecisionTask(Qwen2VLMultimediaTask):
    """Vision, embedding, full-record backbone, and flat ragged decision head."""

    model_roles: ClassVar[dict[str, str]] = {
        "decoder": "encoder",
        "vision_encoder": "encoder",
        "embedding": "embedding",
        "decision_head": "encoder",
    }
    components = ComponentSpec(
        decoder="decoder",
        vision_encoder="vision_encoder",
        embedding="embedding",
        decision_head="decision_head",
    )

    def build(self, module, config: ArchitectureConfig):
        if not isinstance(config, ClefConfig):
            raise TypeError("ClefDecisionTask requires ClefConfig")
        self._validate_components(module)
        graph, builder = _make_graph("clef_backbone")
        op = builder.op
        embeds = builder.input(
            "inputs_embeds", dtype=config.dtype, shape=[1, "sequence", config.hidden_size]
        )
        mask = builder.input("attention_mask", dtype=ir.DataType.INT64, shape=[1, "sequence"])
        positions = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[3, 1, "sequence"]
        )
        states: list[tuple[ir.Value, ir.Value] | None] = []
        for kind in config.layer_types or ["full_attention"] * config.num_hidden_layers:
            if kind == "linear_attention":
                dims = linear_attention_dims(config)
                conv = op.ConstantOfShape(
                    op.Constant(value_ints=[1, dims.conv_dim, dims.conv_kernel - 1]),
                )
                recurrent = op.ConstantOfShape(
                    op.Constant(
                        value_ints=[
                            1,
                            dims.num_v_heads,
                            dims.head_k_dim,
                            dims.head_v_dim,
                        ]
                    ),
                )
                states.append((op.CastLike(conv, embeds), op.CastLike(recurrent, embeds)))
            elif kind == "full_attention":
                states.append(None)
            else:
                raise ValueError(f"Unsupported Clef backbone layer type: {kind}")
        # Call the wrapper to retain decoder.model.* checkpoint naming scopes.
        hidden, _ = module.decoder(
            op,
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=states,
        )
        builder.add_output(hidden, "hidden_states")
        decoder = _make_model(graph)
        _register_linear_attention_functions(decoder, config)

        graph, builder = _make_graph("clef_decision_head")
        hidden = builder.input(
            "hidden_states", dtype=config.dtype, shape=[1, "sequence", config.hidden_size]
        )
        ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[1, "sequence"])
        specs: list[tuple[str, list[str | int]]] = [
            ("question_spans", ["questions", 2]),
            ("option_spans", ["options", 2]),
            ("option_question_ids", ["options"]),
            ("question_types", ["questions"]),
        ]
        inputs = [
            builder.input(name, dtype=ir.DataType.INT64, shape=shape) for name, shape in specs
        ]
        logits, probabilities = module.decision_head(builder.op, hidden, ids, *inputs)
        builder.add_output(logits, "logits")
        builder.add_output(probabilities, "probabilities")
        return ModelPackage(
            {
                "decoder": decoder,
                "vision_encoder": self._build_vision(module.vision_encoder, config),
                "embedding": self._build_multimedia_embedding(module.embedding, config),
                "decision_head": _make_model(graph),
            },
            config=config,
        )
