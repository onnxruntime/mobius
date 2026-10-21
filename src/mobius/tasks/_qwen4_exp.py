# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph I/O contract for the exact Qwen4-Exp heterogeneous text cache."""

from __future__ import annotations

import json
from typing import ClassVar

import onnx_ir as ir
from onnxscript import GraphBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._component_manifest import (
    ComponentDescriptor,
    ComponentManifest,
    SharedWeightEndpoint,
    SharedWeightInfo,
    _config_ties_word_embeddings,
    _infer_shared_weights,
)
from mobius._configs import BaseModelConfig, Qwen4ExpConfig
from mobius._model_package import ModelPackage
from mobius._split_identity import (
    COMPONENT_KEY_PROPERTY,
    SPLIT_ABI,
    SPLIT_ABI_PROPERTY,
    TEXT_COMPONENT_KEYS,
    VLM_COMPONENT_KEYS,
)
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
    build_embedding_from_features,
)
from mobius.tasks._cache_utils import _register_linear_attention_functions
from mobius.tasks._vision_language_3model import QwenVLTask


def _require_layer_types(config: Qwen4ExpConfig) -> list[str]:
    if config.layer_types is None:
        raise ValueError("Qwen4-Exp requires explicit layer_types")
    return config.layer_types


def _make_qwen4_state_inputs(
    builder,
    config: Qwen4ExpConfig,
    batch: ir.SymbolicDim,
    past_seq_len: ir.SymbolicDim,
    past_position_ids: ir.Value,
    *,
    split_ngram: bool = False,
) -> list[tuple[ir.Value, ...]]:
    """Create every Qwen4-Exp recurrent, PLE, QSA, and KV state input."""
    layer_types = _require_layer_types(config)
    linear_key_dim = (
        config.linear_num_key_heads * config.linear_key_head_dim
        if config.linear_num_key_heads is not None and config.linear_key_head_dim is not None
        else 0
    )
    linear_value_dim = (
        config.linear_num_value_heads * config.linear_value_head_dim
        if config.linear_num_value_heads is not None
        and config.linear_value_head_dim is not None
        else 0
    )
    linear_conv_dim = 2 * linear_key_dim + linear_value_dim
    states = []
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type == "linear_attention":
            if not linear_key_dim or not linear_value_dim:
                raise ValueError("Qwen4-Exp linear-attention dimensions must be configured")
            conv_state = builder.input(
                f"past_key_values.{layer_idx}.conv_state",
                dtype=config.dtype,
                shape=[
                    batch,
                    linear_conv_dim,
                    config.linear_conv_kernel_dim,
                ],
            )
            recurrent_state = builder.input(
                f"past_key_values.{layer_idx}.recurrent_state",
                dtype=config.mamba_ssm_dtype,
                shape=[
                    batch,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                ],
            )
            layer_states: tuple[ir.Value, ...] = (
                conv_state,
                recurrent_state,
            )
            if layer_idx + 1 in (config.ple_layer_ids or []):
                ple_state_len = (config.ple_conv_kernel_size - 1) * config.ngram_size
                ple_conv_state = builder.input(
                    f"past_key_values.{layer_idx}.ple_conv_state",
                    dtype=config.dtype,
                    shape=[
                        batch,
                        config.hc_count * config.hidden_size,
                        ple_state_len,
                    ],
                )
                layer_states += (ple_conv_state,)
                if split_ngram:
                    states.append(layer_states)
                    continue
                ple_context = builder.input(
                    f"past_key_values.{layer_idx}.ple_context",
                    dtype=ir.DataType.INT64,
                    shape=[batch, config.ngram_size - 1],
                )
                eos_token_id = (
                    config.eos_token_id[0]
                    if isinstance(config.eos_token_id, list)
                    else config.eos_token_id
                )
                assert eos_token_id is not None
                empty_prefix = builder.op.Equal(
                    builder.op.Shape(past_position_ids, start=-1),
                    builder.op.Constant(value_ints=[0]),
                )
                ple_context = builder.op.Where(
                    empty_prefix,
                    builder.op.Expand(
                        builder.op.Constant(value_int=eos_token_id),
                        builder.op.Shape(ple_context),
                    ),
                    ple_context,
                )
                layer_states += (ple_context,)
        else:
            key = builder.input(
                f"past_key_values.{layer_idx}.key",
                dtype=config.dtype,
                shape=[
                    batch,
                    config.num_key_value_heads,
                    past_seq_len,
                    config.head_dim,
                ],
            )
            value = builder.input(
                f"past_key_values.{layer_idx}.value",
                dtype=config.dtype,
                shape=[
                    batch,
                    config.num_key_value_heads,
                    past_seq_len,
                    config.head_dim,
                ],
            )
            index_key = builder.input(
                f"past_key_values.{layer_idx}.index_key",
                dtype=config.dtype,
                shape=[batch, past_seq_len, config.indexer_head_dim],
            )
            layer_states = (key, value, index_key)
        states.append(layer_states)
    return states


def _register_qwen4_outputs(
    builder,
    config: Qwen4ExpConfig,
    logits: ir.Value,
    presents: list[tuple[ir.Value, ...]],
    present_position_ids: ir.Value,
) -> None:
    """Register the complete heterogeneous state ABI without dropping roles."""
    layer_types = _require_layer_types(config)
    builder.add_output(logits, "logits")
    builder.add_output(present_position_ids, "present_position_ids")
    for layer_idx, (layer_type, present) in enumerate(zip(layer_types, presents)):
        if layer_type == "linear_attention":
            builder.add_output(present[0], f"present.{layer_idx}.conv_state")
            builder.add_output(present[1], f"present.{layer_idx}.recurrent_state")
            if len(present) >= 3:
                builder.add_output(present[2], f"present.{layer_idx}.ple_conv_state")
            if len(present) == 4:
                builder.add_output(present[3], f"present.{layer_idx}.ple_context")
        else:
            builder.add_output(present[0], f"present.{layer_idx}.key")
            builder.add_output(present[1], f"present.{layer_idx}.value")
            builder.add_output(present[2], f"present.{layer_idx}.index_key")


def _finalize_qwen4_decoder(
    graph,
    config: Qwen4ExpConfig,
    *,
    position_axes: list[str],
    split_ngram: bool = False,
) -> ir.Model:
    model = _make_model(graph)
    _register_linear_attention_functions(model, config)
    model.metadata_props["mobius.cache_abi"] = (
        "qwen4-exp:position_ids;"
        "linear=conv_state,recurrent_state[,ple_conv_state,ple_context];"
        "qsa=key,value,index_key"
    )
    if split_ngram:
        model.metadata_props["mobius.cache_abi"] = model.metadata_props[
            "mobius.cache_abi"
        ].replace("ple_conv_state,ple_context", "ple_conv_state")
    model.metadata_props["mobius.semantic_reference_revision"] = (
        "Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c;"
        "transformers@598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1"
    )
    layer_types = _require_layer_types(config)
    layers = []
    for layer_index, layer_type in enumerate(layer_types):
        if layer_type == "linear_attention":
            roles = ["conv_state", "recurrent_state"]
            if layer_index + 1 in (config.ple_layer_ids or []):
                roles.append("ple_conv_state")
                if not split_ngram:
                    roles.append("ple_context")
        else:
            roles = ["key", "value", "index_key"]
        layers.append(
            {
                "index": layer_index,
                "type": layer_type,
                "roles": roles,
                "update": {
                    role: "append" if role in {"key", "value"} else "replace" for role in roles
                },
            }
        )
    model.metadata_props["mobius.state_manifest"] = json.dumps(
        {
            "schema_version": 1,
            "position_state": {
                "input": "past_position_ids",
                "output": "present_position_ids",
                "axes": position_axes,
                "update": "replace",
            },
            "layers": layers,
        },
        separators=(",", ":"),
    )
    return model


def _guard_nonempty(builder: GraphBuilder, input_ids: ir.Value) -> ir.Value:
    """Use the existing fail-closed Reshape input-guard convention on every EP."""
    op = builder.op
    empty = op.Equal(op.Shape(input_ids, start=1, end=2), [0])
    shape = op.Add(op.Cast(empty, to=ir.DataType.INT64), [1])
    guard = op.Reshape(op.Constant(value_ints=[0]), shape)
    return op.Add(input_ids, op.ReduceSum(guard, keepdims=False))


def _ngram_input(
    builder: GraphBuilder,
    config: Qwen4ExpConfig,
    batch: ir.SymbolicDim,
    seq_len: ir.SymbolicDim,
) -> ir.Value:
    return builder.input(
        "ngram_embeddings",
        dtype=config.dtype,
        shape=[len(config.ple_layer_ids or []), batch, seq_len, config.ple_embed_dim],
    )


class _Qwen4ExpSplitOptions:
    """Instance-local selection, manifest ownership, and split graph metadata."""

    _is_vlm: ClassVar[bool] = False

    def __init__(
        self,
        *,
        split_ngram: bool = False,
        requested_components: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.split_ngram = split_ngram
        self.requested_components = (
            frozenset(requested_components) if requested_components is not None else None
        )
        keys = VLM_COMPONENT_KEYS if self._is_vlm else TEXT_COMPONENT_KEYS
        if not split_ngram and requested_components is not None:
            raise ValueError("requested_components requires split_ngram=True")
        if self.requested_components is not None:
            if not self.requested_components or not self.requested_components <= keys:
                raise ValueError(f"Expected a nonempty selection from {sorted(keys)}")
        self._selected = (
            self.requested_components
            if self.requested_components is not None
            else frozenset(keys)
        )

    def validate_build(self, config: BaseModelConfig, *, execution_provider: str) -> None:
        if not self.split_ngram:
            return
        if execution_provider != "onnx-standard":
            raise ValueError("split_ngram requires execution_provider='onnx-standard'")
        if not isinstance(config, Qwen4ExpConfig) or not config.ple_layer_ids:
            raise ValueError("split_ngram requires at least one configured PLE layer")
        if config.dtype not in {ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
            raise ValueError("split_ngram supports only F32/F16/BF16")
        if (
            config.block_quant_scheme is not None
            or config.quantization is not None
            or config.component_quantization is not None
        ):
            raise ValueError(
                "split_ngram requires unquantized weights; FP8/block/affine are unsupported"
            )

    def _split_manifest(self, config: object | None) -> ComponentManifest:
        text_config = getattr(config, "text_config", config)
        if isinstance(text_config, dict):
            layer_ids = text_config.get("ple_layer_ids", [])
        else:
            layer_ids = getattr(text_config, "ple_layer_ids", [])
        lookups = tuple(
            f"model.language_model.layers.{layer_id - 1}.ple.ple_embedding"
            for layer_id in layer_ids
        )
        main_roots: tuple[str, ...] = (
            "model.language_model.layers",
            "model.language_model.rotary_emb",
            "model.language_model.hyper_connection_mixer",
            "lm_head",
        )
        if not self._is_vlm:
            main_roots += ("model.language_model.embed_tokens",)
        specs = ComponentSpec(
            ngram="",
            main="decoder" if self._is_vlm else "",
            **(
                {"vision_encoder": "vision_encoder", "embedding": "embedding"}
                if self._is_vlm
                else {}
            ),
        )
        descriptors = []
        for name, path in specs.items():
            if name == "ngram":
                descriptor = ComponentDescriptor(name, path, "embedding", lookups)
            elif name == "main":
                descriptor = ComponentDescriptor(
                    name,
                    path,
                    "decoder",
                    main_roots,
                    (
                        ("lm_head", "lm_head"),
                        ("model.embed_tokens", "model.language_model.embed_tokens"),
                    )
                    if not self._is_vlm
                    else (("lm_head", "lm_head"),),
                    source_path_exclusions=lookups,
                )
            elif name == "embedding":
                descriptor = ComponentDescriptor(
                    name,
                    path,
                    "embedding",
                    ("model.language_model.embed_tokens",),
                    (("embed_tokens", "model.language_model.embed_tokens"),),
                )
            else:
                descriptor = ComponentDescriptor(name, path, "encoder", ("model.visual",))
            descriptors.append(descriptor)
        manifest = ComponentManifest(tuple(descriptors))
        shared_weights = _infer_shared_weights(config, manifest) if config is not None else ()
        if not self._is_vlm and config is not None and _config_ties_word_embeddings(config):
            # Text main owns both ordinary token consumers. A tied checkpoint
            # may omit the head even though neither consumer belongs to ngram.
            shared_weights = (
                SharedWeightInfo(
                    name="word_embeddings",
                    canonical=SharedWeightEndpoint(
                        "main", "model.language_model.embed_tokens.weight"
                    ),
                    aliases=(SharedWeightEndpoint("main", "lm_head.weight"),),
                    kind="tied_word_embeddings",
                ),
            )
        return ComponentManifest(
            manifest.components,
            shared_weights,
        )

    def _mark_split(self, package: ModelPackage, config: Qwen4ExpConfig) -> ModelPackage:
        roles = self._split_manifest(config)
        for name, model in package.items():
            model.metadata_props.update(
                {
                    SPLIT_ABI_PROPERTY: SPLIT_ABI,
                    COMPONENT_KEY_PROPERTY: name,
                    "mobius.qwen4_exp.role": roles[name].role,
                    "mobius.qwen4_exp.ple_layer_ids": json.dumps(config.ple_layer_ids),
                    "mobius.qwen4_exp.ngram_size": str(config.ngram_size),
                    "mobius.qwen4_exp.ple_embed_dim": str(config.ple_embed_dim),
                    "mobius.qwen4_exp.dtype": str(config.dtype),
                }
            )
        return package

    def _build_ngram(self, module: nn.Module, config: Qwen4ExpConfig) -> ir.Model:
        batch, seq_len = ir.SymbolicDim("batch"), ir.SymbolicDim("sequence_len")
        graph, builder = _make_graph("qwen4_exp_ngram")
        input_ids = _guard_nonempty(
            builder,
            builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]),
        )
        context = builder.input(
            "past_ngram_context",
            dtype=ir.DataType.INT64,
            shape=[len(config.ple_layer_ids or []), batch, config.ngram_size - 1],
        )
        embeddings, present = module(
            builder.op, input_ids=input_ids, past_ngram_context=context
        )
        builder.add_output(embeddings, "ngram_embeddings")
        builder.add_output(present, "present_ngram_context")
        model = _make_model(graph)
        model.metadata_props["mobius.qwen4_exp.context"] = (
            "host-first-eos-cold;configured-layer-order;past_ngram_context->present_ngram_context"
        )
        return model


class Qwen4ExpCausalLMTask(_Qwen4ExpSplitOptions, ModelTask):
    """Build Qwen4-Exp with DeltaNet, PLE, QSA, and position cache states."""

    def component_manifest(
        self,
        *,
        module_class: type | None = None,
        model_type: str | None = None,
        hf_config: object | None = None,
    ) -> ComponentManifest:
        if self.split_ngram:
            return self._split_manifest(hf_config)
        return super().component_manifest(
            module_class=module_class, model_type=model_type, hf_config=hf_config
        )

    def build(self, module: nn.Module, config: Qwen4ExpConfig) -> ModelPackage:
        if self.split_ngram:
            self.validate_build(config, execution_provider=ep_capabilities().name)
            models = {}
            if "ngram" in self._selected:
                models["ngram"] = self._build_ngram(module, config)
            if "main" in self._selected:
                models["main"] = self._build_main(module, config)
            return self._mark_split(ModelPackage(models, config=config), config)
        return ModelPackage({"model": self._build_main(module, config)}, config=config)

    def _build_main(self, module: nn.Module, config: Qwen4ExpConfig) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")
        graph, builder = _make_graph("qwen4_exp")

        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])
        if self.split_ngram:
            input_ids = _guard_nonempty(builder, input_ids)
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_sequence_len + sequence_len"],
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        past_position_ids = builder.input(
            "past_position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, past_seq_len],
        )

        states = _make_qwen4_state_inputs(
            builder,
            config,
            batch,
            past_seq_len,
            past_position_ids,
            split_ngram=self.split_ngram,
        )

        logits, presents, present_position_ids = module(
            builder.op,
            input_ids,
            attention_mask,
            position_ids,
            past_position_ids,
            states,
            **(
                {"ngram_embeddings": _ngram_input(builder, config, batch, seq_len)}
                if self.split_ngram
                else {}
            ),
        )
        _register_qwen4_outputs(builder, config, logits, presents, present_position_ids)
        return _finalize_qwen4_decoder(
            graph, config, position_axes=["text"], split_ngram=self.split_ngram
        )


class Qwen4ExpVisionLanguageTask(_Qwen4ExpSplitOptions, QwenVLTask):
    """Build the Qwen4-Exp decoder, shared Qwen vision tower, and embedding mixer."""

    _is_vlm = True

    def component_manifest(
        self,
        *,
        module_class: type | None = None,
        model_type: str | None = None,
        hf_config: object | None = None,
    ) -> ComponentManifest:
        if self.split_ngram:
            return self._split_manifest(hf_config)
        return super().component_manifest(
            module_class=module_class, model_type=model_type, hf_config=hf_config
        )

    def build(self, module: nn.Module, config: Qwen4ExpConfig) -> ModelPackage:
        if self.split_ngram:
            self.validate_build(config, execution_provider=ep_capabilities().name)
        self._validate_components(module)
        if config.deepstack_visual_indexes or (
            config.vision is not None and config.vision.deepstack_visual_indexes
        ):
            raise ValueError("Qwen4-Exp multimodal export does not support DeepStack")
        models = {}
        if self.split_ngram and "ngram" in self._selected:
            models["ngram"] = self._build_ngram(module, config)
        if not self.split_ngram or "main" in self._selected:
            models["main" if self.split_ngram else "decoder"] = self._build_decoder(
                module.decoder, config, split_ngram=self.split_ngram
            )
        if not self.split_ngram or "vision_encoder" in self._selected:
            models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        if not self.split_ngram or "embedding" in self._selected:
            models["embedding"] = self._build_embedding(module, config)
        package = ModelPackage(models, config=config)
        return self._mark_split(package, config) if self.split_ngram else package

    @staticmethod
    def _build_embedding(module: nn.Module, config: Qwen4ExpConfig) -> ir.Model:
        embedding = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
        )
        embedding.metadata_props["mobius.unsupported_token_ids"] = json.dumps(
            {"video": config.unsupported_video_token_id},
            separators=(",", ":"),
        )
        return embedding

    @staticmethod
    def _build_decoder(
        decoder: nn.Module, config: Qwen4ExpConfig, *, split_ngram: bool = False
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")
        graph, builder = _make_graph("qwen4_exp_decoder")

        inputs_embeds = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=[batch, seq_len, config.hidden_size],
        )
        # PLE hashes the exact token stream independently of image feature
        # replacement, so it must not be inferred from inputs_embeds.
        ple_input_ids = builder.input(
            "ple_input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        if split_ngram:
            ple_input_ids = _guard_nonempty(builder, ple_input_ids)
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_sequence_len + sequence_len"],
        )
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[4, batch, seq_len],
        )
        past_position_ids = builder.input(
            "past_position_ids",
            dtype=ir.DataType.INT64,
            shape=[4, batch, past_seq_len],
        )
        states = _make_qwen4_state_inputs(
            builder,
            config,
            batch,
            past_seq_len,
            past_position_ids,
            split_ngram=split_ngram,
        )

        logits, presents, present_position_ids = decoder(
            builder.op,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            ple_input_ids=ple_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_position_ids=past_position_ids,
            past_key_values=states,
            **(
                {"ngram_embeddings": _ngram_input(builder, config, batch, seq_len)}
                if split_ngram
                else {}
            ),
        )
        _register_qwen4_outputs(builder, config, logits, presents, present_position_ids)
        return _finalize_qwen4_decoder(
            graph,
            config,
            position_axes=["text", "temporal", "height", "width"],
            split_ngram=split_ngram,
        )
