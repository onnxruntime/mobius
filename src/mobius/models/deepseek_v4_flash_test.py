# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnxscript import GraphBuilder

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius._constants import OPSET_VERSION
from mobius._testing import count_op_type, make_config
from mobius.components import create_attention_bias
from mobius.components._rotary_embedding import initialize_rope
from mobius.integrations.ort_genai import export_package
from mobius.integrations.transformers._config_resolver import _default_task_for_model
from mobius.models._deepseek_v4_csa import (
    CSA_DOMAIN,
    HCA_COMPRESSION_RATIO,
    NativeCsaExportError,
    plan_native_csa,
)
from mobius.models.deepseek_v4 import (
    DeepSeekV4Attention,
    DeepSeekV4CausalLMModel,
    _gqa_kv_lengths,
)


def _tiny_config(**overrides):
    values = dict(
        model_type="deepseek_v4",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        o_groups=2,
        o_lora_rank=8,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        num_hash_layers=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        swiglu_limit=10.0,
        rope_interleave=True,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
    )
    values.update(overrides)
    return make_config(**values)


def test_real_config_fields_extract():
    hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        vocab_size=129280,
        hidden_size=4096,
        intermediate_size=None,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        max_position_embeddings=1048576,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        n_routed_experts=256,
        num_experts_per_tok=6,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        num_hash_layers=None,
        mlp_layer_types=["hash_moe", "hash_moe", "hash_moe", "moe"],
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        compress_ratios=None,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        },
        compress_rope_theta=160000,
        sliding_window=128,
        swiglu_limit=10.0,
        tie_word_embeddings=False,
        num_nextn_predict_layers=1,
        attention_bias=False,
        pad_token_id=0,
    )
    config = ArchitectureConfig.from_transformers(hf_config)
    assert config.model_type == "deepseek_v4"
    assert config.head_dim == 512
    assert config.qk_rope_head_dim == 64
    assert config.num_local_experts == 256
    assert config.num_hash_layers == 3
    assert config.hc_mult == 4
    assert config.compress_ratios == [0, 0, 4, 128]
    assert config.num_nextn_predict_layers == 1
    assert config.rope_type == "yarn"
    assert _default_task_for_model("deepseek_v4") == "deepseek-v4"


def test_tiny_graph_builds_v4_backbone():
    config = _tiny_config()
    graph = build_from_module(DeepSeekV4CausalLMModel(config), config)["model"].graph
    assert graph.num_nodes() > 0
    assert count_op_type(graph, "Attention") == 0
    # Default EP declares an empty `gqa_dtypes` (see
    # `mobius._execution_providers`) specifically to guarantee portable,
    # `com.microsoft`-free graphs -- DeepSeek-V4 must fall back to the
    # decomposed attention path here, exactly like every other model in
    # this codebase. See `test_tiny_graph_builds_v4_backbone_fused_gqa_on_cpu_ep`
    # below for the fused-GQA structural proof under a GQA-capable EP.
    assert count_op_type(graph, "GroupQueryAttention") == 0
    assert count_op_type(graph, "Softmax") >= config.num_hidden_layers
    assert count_op_type(graph, "ScatterElements") == 0
    assert count_op_type(graph, "Softplus") >= config.num_hidden_layers
    assert count_op_type(graph, "TopK") >= 1
    assert count_op_type(graph, "Gather") >= 1
    assert count_op_type(graph, "RMSNormalization") >= 1
    assert sum("attn_sink" in name for name in graph.initializers) == config.num_hidden_layers
    assert "hidden_states" not in {value.name for value in graph.outputs}


def test_tiny_graph_builds_v4_backbone_fused_gqa_on_cpu_ep():
    """Same backbone, built under a GQA-capable EP, must fuse attention.

    ``"cpu"`` declares ``gqa_dtypes={FLOAT}`` (``mobius._execution_providers``),
    which the tiny config's default ``dtype=ir.DataType.FLOAT`` matches, so
    this exercises ``_use_fused_gqa()``'s true branch -- the same EP-gating
    idiom ``lfm2_test.py``/``world_model_test.py`` use for their own
    GQA-capable-EP structural tests.
    """
    config = _tiny_config()
    graph = build_from_module(
        DeepSeekV4CausalLMModel(config), config, execution_provider="cpu"
    )["model"].graph
    assert count_op_type(graph, "Attention") == 0
    gqa_nodes = [node for node in graph if node.op_type == "GroupQueryAttention"]
    assert len(gqa_nodes) == config.num_hidden_layers
    assert all(node.domain == "com.microsoft" for node in gqa_nodes)
    # head_sink (12th positional input) carries the learned attention sink;
    # attention_bias (11th) is intentionally omitted -- GQA's implicit
    # causal mask + seqlens_k/total_seq_len already covers plain causal
    # decoding, matching every other direct-GQA model in this codebase.
    assert all(len(node.inputs) == 12 and node.inputs[10] is None for node in gqa_nodes)
    assert all(node.inputs[11] is not None for node in gqa_nodes)
    # `_tiny_config()` sets no `sliding_window`, so `local_window_size` must
    # be entirely absent (the `Attention._forward_gqa` convention -- see
    # `mobius/components/_attention_test.py::test_gqa_context_no_local_window_size_when_default`)
    # rather than present with the -1 "disabled" sentinel value.
    assert all("local_window_size" not in node.attributes for node in gqa_nodes)
    # Only the Hyper-Connection combination softmax (2 per layer) remains;
    # the manual Concat(scores, sink) -> Softmax -> slice attention softmax
    # the decomposed path uses is gone now that GQA computes it internally.
    assert count_op_type(graph, "Softmax") == 2 * config.num_hidden_layers
    assert count_op_type(graph, "ScatterElements") == 0
    assert count_op_type(graph, "Softplus") >= config.num_hidden_layers
    assert count_op_type(graph, "TopK") >= 1
    assert count_op_type(graph, "Gather") >= 1
    assert count_op_type(graph, "RMSNormalization") >= 1
    assert sum("attn_sink" in name for name in graph.initializers) == config.num_hidden_layers
    assert "hidden_states" not in {value.name for value in graph.outputs}


def test_sliding_window_sets_local_window_size_on_fused_gqa_regardless_of_compress_ratio():
    """The reference's mandatory per-layer window must reach GQA as an attribute.

    Official DeepSeek-V4 (``inference/model.py::Attention.forward``,
    ``get_window_topk_idxs``) unconditionally restricts *every* layer --
    regardless of ``compress_ratio`` -- to a circular-buffer window of the
    most recent ``sliding_window`` positions. Regression test for the bug
    where ``DeepSeekV4Attention`` never read ``config.sliding_window`` at
    all (see ``DeepSeekV4Attention.local_window_size``), silently computing
    unbounded causal attention on both the fused-GQA and decomposed paths.
    Ratio>0 layers still only get the *local* component of the reference's
    attention correctly here -- the additional compressed/indexer-selected
    positions those layers union in remain a separate, tracked gap (see
    ``docs/models/DEEPSEEK_CSA_MTP_RUNTIME.md``, not addressed by this test.
    The MTP sidecar (``DeepSeekV4Mtp``) is a regular ratio-0 decoder layer
    bound by the same mandatory window, so it's checked here too (a plain
    ``compress_ratios=[0]`` layer, since MTP doesn't schedule its own
    ratios).
    """
    config = _tiny_config(
        num_hidden_layers=3,
        compress_ratios=[0, 4, 128],
        sliding_window=8,
        num_nextn_predict_layers=1,
    )
    package = build_from_module(
        DeepSeekV4CausalLMModel(config), config, task="deepseek-v4", execution_provider="cpu"
    )

    graph = package["model"].graph
    gqa_nodes = [node for node in graph if node.op_type == "GroupQueryAttention"]
    assert len(gqa_nodes) == config.num_hidden_layers
    assert all(node.attributes["local_window_size"].as_int() == 8 for node in gqa_nodes)

    mtp_graph = package["mtp"].graph
    (mtp_gqa_node,) = [node for node in mtp_graph if node.op_type == "GroupQueryAttention"]
    assert mtp_gqa_node.attributes["local_window_size"].as_int() == 8


def test_csa_schedule_exports_compressor_and_indexer_tensors_with_dense_attention():
    config = _tiny_config(
        num_hidden_layers=4,
        compress_ratios=[0, 0, 4, 128],
    )
    graph = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")[
        "model"
    ].graph
    names = set(graph.initializers)

    assert "model.layers.2.self_attn.compressor.wkv.weight" in names
    assert "model.layers.2.self_attn.compressor.wgate.weight" in names
    assert "model.layers.2.self_attn.compressor.ape" in names
    assert "model.layers.2.self_attn.compressor.norm.weight" in names
    assert "model.layers.2.self_attn.indexer.wq_b.weight" in names
    assert "model.layers.2.self_attn.indexer.weights_proj.weight" in names
    assert "model.layers.2.self_attn.indexer.compressor.wkv.weight" in names
    assert "model.layers.3.self_attn.compressor.wkv.weight" in names
    assert not any("model.layers.3.self_attn.indexer" in name for name in names)
    assert count_op_type(graph, "Attention") == 0
    assert count_op_type(graph, "GroupQueryAttention") == 0
    assert count_op_type(graph, "ScatterElements") == 0
    assert count_op_type(graph, "Softmax") >= config.num_hidden_layers


def test_csa_schedule_exports_fused_gqa_regardless_of_compress_ratio_on_cpu_ep():
    config = _tiny_config(
        num_hidden_layers=4,
        compress_ratios=[0, 0, 4, 128],
    )
    graph = build_from_module(
        DeepSeekV4CausalLMModel(config),
        config,
        task="deepseek-v4",
        execution_provider="cpu",
    )["model"].graph
    names = set(graph.initializers)

    assert "model.layers.2.self_attn.compressor.wkv.weight" in names
    assert "model.layers.3.self_attn.compressor.wkv.weight" in names
    assert not any("model.layers.3.self_attn.indexer" in name for name in names)
    assert count_op_type(graph, "Attention") == 0
    # Dense-CSA schedule attention is still a single fused GQA call per
    # layer regardless of compress_ratio: the compressor/indexer tensors
    # above are retained as a zero-valued shape anchor (see
    # `_shape_anchor`/`DeepSeekV4CompressorTensors.forward`) for future
    # sparse-runtime handoff, they do not participate in the dense
    # attention computation this graph actually executes.
    assert count_op_type(graph, "GroupQueryAttention") == config.num_hidden_layers
    assert count_op_type(graph, "ScatterElements") == 0
    assert count_op_type(graph, "Softmax") == 2 * config.num_hidden_layers


def test_mtp_sidecar_exports_official_block_and_hyper_connection_state():
    config = _tiny_config(
        num_nextn_predict_layers=1,
        compress_ratios=[0, 0, 0],
    )
    package = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")

    assert set(package) == {"model", "mtp"}
    assert "hidden_states" in {value.name for value in package["model"].graph.outputs}
    mtp_graph = package["mtp"].graph
    names = set(mtp_graph.initializers)
    assert "mtp.0.e_proj.weight" in names
    assert "mtp.0.h_proj.weight" in names
    assert "mtp.0.hc_head_fn.weight" in names
    assert "mtp.0.self_attn.attn_sink" in names
    assert count_op_type(mtp_graph, "GroupQueryAttention") == 0
    assert count_op_type(mtp_graph, "Softmax") >= 1
    assert {value.name for value in mtp_graph.outputs} == {
        "mtp_hidden",
        "present.0.key",
        "present.0.value",
    }


def test_mtp_sidecar_exports_fused_gqa_on_cpu_ep():
    config = _tiny_config(
        num_nextn_predict_layers=1,
        compress_ratios=[0, 0, 0],
    )
    package = build_from_module(
        DeepSeekV4CausalLMModel(config),
        config,
        task="deepseek-v4",
        execution_provider="cpu",
    )

    assert set(package) == {"model", "mtp"}
    mtp_graph = package["mtp"].graph
    names = set(mtp_graph.initializers)
    assert "mtp.0.self_attn.attn_sink" in names
    assert count_op_type(mtp_graph, "GroupQueryAttention") == 1
    assert {value.name for value in mtp_graph.outputs} == {
        "mtp_hidden",
        "present.0.key",
        "present.0.value",
    }


def test_ort_genai_sidecar_metadata(tmp_path):
    config = _tiny_config(
        num_nextn_predict_layers=1,
        compress_ratios=[0, 0, 0],
    )
    package = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")

    for model in package.values():
        for initializer in model.graph.initializers.values():
            if initializer.const_value is None:
                initializer.const_value = ir.tensor(
                    np.zeros(initializer.shape, dtype=initializer.dtype.numpy())
                )
    result = export_package(package, str(tmp_path), progress_bar=False)
    assert (tmp_path / "model" / "model.onnx").is_file()
    assert (tmp_path / "mtp" / "model.onnx").is_file()
    with open(result["genai_config"]) as f:
        genai_config = json.load(f)
    assert genai_config["model"]["decoder"]["filename"] == "model/model.onnx"
    with open(result["mtp_config"]) as f:
        sidecar = json.load(f)

    assert sidecar["model"]["filename"] == "mtp/model.onnx"
    assert sidecar["num_nextn_predict_layers"] == 1
    assert "hidden_states" in sidecar["inputs"]
    assert "mtp_hidden" in sidecar["outputs"]


def test_parameter_shapes_match_v4_projections():
    config = _tiny_config(num_hidden_layers=1)
    model = DeepSeekV4CausalLMModel(config)
    attn = model.model.layers[0].self_attn
    assert list(attn.q_b_proj.weight.shape) == [32, 8]
    assert list(attn.kv_proj.weight.shape) == [16, 32]
    assert list(attn.o_a_proj.weight.shape) == [16, 16]
    assert list(attn.o_b_proj.weight.shape) == [32, 16]
    # DeepSeekV4MoE composes the shared MoELayer, so the gate now lives at
    # mlp.moe.gate (see DeepSeekV4MoE.__init__).
    assert list(model.model.layers[0].mlp.moe.gate.bias.shape) == [config.num_local_experts]


def test_four_and_eight_bit_graphs_use_matmul_nbits():
    for bits in (4, 8):
        config = _tiny_config(
            num_hidden_layers=1,
            compress_ratios=[4],
            quantization=QuantizationConfig(
                bits=bits,
                group_size=16,
                quant_method="gguf",
                sym=False,
            ),
        )
        graph = build_from_module(DeepSeekV4CausalLMModel(config), config)["model"].graph
        assert count_op_type(graph, "MatMulNBits") > 0
        assert "model.layers.0.self_attn.compressor.wkv.scales" in graph.initializers
        model = DeepSeekV4CausalLMModel(config)
        assert model.model.layers[0].self_attn.compressor.wkv._gguf_quantized_linear


def test_qmoe_eligible_quantization_fuses_routed_experts_into_one_qmoe_per_layer():
    """gptq/awq/olive int4 quantization must fuse each layer's routed experts.

    QMoE's native ABI must collapse every layer's routed experts into a single
    QMoE node -- not one MatMulNBits set per expert -- while the shared expert (untouched by this change) still emits
    its own MatMulNBits/dense projections, and hash-routed layers keep using
    DeepSeekV4Gate's Gather-based lookup rather than TopK.

    The structural proof is that QMoE count == num_hidden_layers and, unlike
    the old per-expert dense loop, MatMulNBits count for the *routed* experts
    does not scale with num_local_experts: doubling num_local_experts must
    not change the graph's total MatMulNBits count, since QMoE folds all
    experts' weights into its own initializers instead of one MatMulNBits
    triple per expert.
    """
    num_hidden_layers = 2

    def _build(num_local_experts):
        config = _tiny_config(
            num_hidden_layers=num_hidden_layers,
            compress_ratios=[4, 4],
            num_hash_layers=1,  # layer 0 hash-routed, layer 1 learned top-k.
            num_local_experts=num_local_experts,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="gptq",
                sym=True,
            ),
        )
        model = DeepSeekV4CausalLMModel(config)
        for layer in model.model.layers:
            assert layer.mlp.moe.experts is None, (
                "quantized routed experts must take the QMoE path"
            )
        return build_from_module(model, config)["model"].graph

    graph_2_experts = _build(2)
    graph_8_experts = _build(8)

    for graph in (graph_2_experts, graph_8_experts):
        assert count_op_type(graph, "QMoE") == num_hidden_layers
        # Hash-routed layer 0 still uses the Gather-based tid2eid lookup,
        # learned top-k layer 1 still uses TopK -- selection algorithm is
        # unchanged by fusing the expert compute into QMoE.
        assert count_op_type(graph, "TopK") >= 1
        assert count_op_type(graph, "Gather") >= 1

    matmul_nbits_2 = count_op_type(graph_2_experts, "MatMulNBits")
    matmul_nbits_8 = count_op_type(graph_8_experts, "MatMulNBits")
    assert matmul_nbits_2 == matmul_nbits_8, (
        "MatMulNBits count must not scale with num_local_experts once routed "
        "experts are fused into QMoE"
    )
    # Only the shared expert's own gate/up/down projections remain as
    # individually quantized MatMulNBits, one triple per layer.
    assert matmul_nbits_2 >= num_hidden_layers * 3


def test_official_weight_names_are_remapped():
    config = _tiny_config(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        compress_ratios=[0, 0, 4, 128, 0],
    )
    model = DeepSeekV4CausalLMModel(config)
    weights = {
        "embed.weight": torch.zeros(config.vocab_size, config.hidden_size),
        "layers.0.attn.wq_a.weight": torch.zeros(config.q_lora_rank, config.hidden_size),
        "layers.0.ffn.experts.0.w1.weight": torch.zeros(
            config.moe_intermediate_size, config.hidden_size
        ),
        "layers.0.hc_attn_base": torch.zeros((2 + config.hc_mult) * config.hc_mult),
        "layers.2.attn.attn_sink": torch.zeros(config.num_attention_heads),
        "layers.2.attn.compressor.ape": torch.zeros(4, 2 * config.head_dim),
        "layers.2.attn.indexer.weights_proj.weight": torch.zeros(
            config.index_n_heads, config.hidden_size
        ),
        "mtp.0.e_proj.weight": torch.zeros(config.hidden_size, config.hidden_size),
        "mtp.0.hc_head_fn": torch.zeros(config.hc_mult, config.hc_mult * config.hidden_size),
    }
    result = model.preprocess_weights(weights)
    assert "model.embed_tokens.weight" in result
    assert "model.layers.0.self_attn.q_a_proj.weight" in result
    # Dense (unquantized) fallback: experts live under the shared MoELayer's
    # ModuleList (mlp.moe.experts.*), unlike the QMoE path below.
    assert "model.layers.0.mlp.moe.experts.0.gate_proj.weight" in result
    assert "model.layers.0.hc_attn_base" in result
    assert "model.layers.2.self_attn.attn_sink" in result
    assert "model.layers.2.self_attn.compressor.ape" in result
    assert "model.layers.2.self_attn.indexer.weights_proj.weight" in result
    assert "mtp.0.e_proj.weight" in result
    assert "mtp.0.hc_head_fn.weight" in result


def test_preprocess_weights_rejects_hash_table_with_duplicate_expert_per_token():
    """Fail loudly on a corrupted/malformed hash table.

    QMoE export requires ``_scatter_selected_to_full``'s
    distinct-experts-per-token invariant; a corrupted/malformed tid2eid
    table must fail at weight load time instead of silently dropping a
    contribution at inference time.
    """
    config = _tiny_config(
        num_hidden_layers=1,
        num_hash_layers=1,
        num_experts_per_tok=2,
        quantization=QuantizationConfig(bits=4, group_size=16, quant_method="gptq", sym=True),
    )
    model = DeepSeekV4CausalLMModel(config)
    tid2eid = torch.stack(
        [torch.arange(config.num_experts_per_tok, dtype=torch.int32)] * config.vocab_size
    )  # every token starts distinct: row i = [0, 1]
    tid2eid[5] = torch.tensor([1, 1], dtype=torch.int32)  # duplicate expert for token 5
    weights = {"layers.0.ffn.gate.tid2eid": tid2eid}
    with pytest.raises(ValueError, match="duplicate expert"):
        model.preprocess_weights(weights)


def test_preprocess_weights_accepts_hash_table_with_distinct_experts_per_token():
    config = _tiny_config(
        num_hidden_layers=1,
        num_hash_layers=1,
        num_experts_per_tok=2,
        quantization=QuantizationConfig(bits=4, group_size=16, quant_method="gptq", sym=True),
    )
    model = DeepSeekV4CausalLMModel(config)
    tid2eid = torch.stack(
        [torch.arange(config.num_experts_per_tok, dtype=torch.int32)] * config.vocab_size
    )  # every token distinct: row i = [0, 1]
    tid2eid[5] = torch.tensor([1, 0], dtype=torch.int32)
    weights = {"layers.0.ffn.gate.tid2eid": tid2eid}
    result = model.preprocess_weights(weights)
    assert torch.equal(result["model.layers.0.mlp.moe.gate.tid2eid"], tid2eid)


def _attention_only_config(**overrides):
    """A ``DeepSeekV4Attention``-sized config with a RoPE-capable ``rope_type``.

    Deliberately smaller/simpler than ``_tiny_config`` (no MoE/HC fields
    needed): only what ``DeepSeekV4Attention.__init__`` and its own
    ``rotary_emb`` construction (mirroring ``DeepSeekV4TextModel.__init__``)
    require.
    """
    values = dict(
        model_type="deepseek_v4",
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        o_groups=2,
        o_lora_rank=8,
        rope_interleave=True,
        rope_type="default",
    )
    values.update(overrides)
    return make_config(**values)


def _run_attention_graph(
    config,
    *,
    hidden_values,
    attention_mask_values,
    state,
    fused_gqa,
    decomposed_sliding_window,
):
    """Build and execute a standalone ``DeepSeekV4Attention`` prefill graph.

    A fresh ``DeepSeekV4Attention``/rotary-embedding pair is constructed
    *inside* this call (rather than reused across invocations) because
    ``onnxscript.nn.Parameter._realize`` is a one-shot, per-parameter-object
    flag: reusing the same module/rotary instance across more than one
    ``GraphBuilder``/graph would silently skip re-registering its
    initializers (e.g. ``cos_cache``) on the second and later graphs. ``state``
    (a ``name -> ir.tensor`` dict of fixed random weight values, built once
    by the caller from one throwaway probe instance) is reapplied via
    ``load_state_dict`` so every call sees byte-identical weights.

    ``decomposed_sliding_window`` is the caller's explicit choice for the
    decomposed path's bias (ignored when ``fused_gqa=True``, which instead
    reads ``config.sliding_window`` via ``attn.local_window_size``) -- this
    lets one attention module built with a fixed ``config.sliding_window``
    (so the fused-GQA case is representative of that config) also produce
    an "unbounded" decomposed baseline for comparison.
    """
    attn = DeepSeekV4Attention(config, layer_id=0)
    attn.load_state_dict(state)
    rope_config = dataclasses.replace(config, head_dim=config.qk_rope_head_dim)
    rotary_emb = initialize_rope(
        dataclasses.replace(
            rope_config,
            rope_type="default",
            rope_scaling=None,
            original_max_position_embeddings=None,
        )
    )

    batch, seq_len, hidden_size = hidden_values.shape
    graph = ir.Graph(
        inputs=[],
        outputs=[],
        nodes=[],
        name="deepseek_v4_attention_window_fixture",
        opset_imports={"": OPSET_VERSION, "com.microsoft": 1},
    )
    builder = GraphBuilder(graph)
    op = builder.op
    hidden_states = ir.Value(
        name="hidden_states",
        shape=ir.Shape([batch, seq_len, hidden_size]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    attention_mask = ir.Value(
        name="attention_mask",
        shape=ir.Shape([batch, seq_len]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    position_ids = ir.Value(
        name="position_ids",
        shape=ir.Shape([batch, seq_len]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    graph.inputs.extend([hidden_states, attention_mask, position_ids])

    position_embeddings = rotary_emb(op, position_ids)
    if fused_gqa:
        attention_bias = None
        seqlens_k, total_seq_len = _gqa_kv_lengths(op, attention_mask)
    else:
        attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states,
            attention_mask=attention_mask,
            sliding_window=decomposed_sliding_window,
            dtype=ir.DataType.FLOAT,
        )
        seqlens_k, total_seq_len = None, None

    output, _, _ = attn(
        op, hidden_states, position_embeddings, None, attention_bias, seqlens_k, total_seq_len
    )
    output.name = "output"
    graph.outputs.append(output)

    model = ir.Model(graph, ir_version=11)
    proto = ir.to_proto(model)
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(
        None,
        {
            "hidden_states": hidden_values,
            "attention_mask": attention_mask_values,
            "position_ids": np.arange(seq_len, dtype=np.int64)[None, :].repeat(batch, axis=0),
        },
    )
    return actual


def test_missing_sliding_window_regression_fused_gqa_matches_windowed_decomposed():
    """Regression test for the missing mandatory sliding-window bug.

    Official DeepSeek-V4 (``inference/model.py::Attention.forward``,
    ``get_window_topk_idxs``) restricts every layer, regardless of
    ``compress_ratio``, to a circular-buffer window of the ``sliding_window``
    (128 in the real checkpoint) most-recent positions. Before this fix,
    *neither* the decomposed (``create_attention_bias``) nor the fused-GQA
    (``GroupQueryAttention.local_window_size``) path applied this
    restriction: both silently computed full/unbounded causal attention.

    This builds byte-identical weights/inputs (seq_len=6) through a module
    configured with ``sliding_window=2`` (smaller than seq_len, so the
    restriction actually bites), then executes three graphs:

    * "unbounded": decomposed path, no window applied to the bias
      (``create_attention_bias(sliding_window=None)``) -- the pre-fix
      behavior for every layer, window-configured or not.
    * "windowed": same decomposed path, but with the module's own
      ``sliding_window=2`` baked into the bias.
    * "fused": the fused-``GroupQueryAttention`` path, built from the same
      module (``config.sliding_window=2``, so ``local_window_size=2`` per
      ``DeepSeekV4Attention.__init__``).

    Assertions:
    1. unbounded vs. windowed must DIFFER -- proves ``sliding_window``
       actually changes the decomposed path's output (sanity check that the
       fixture is exercising the window at all).
    2. windowed vs. fused must MATCH -- proves the fused-GQA
       ``local_window_size`` attribute reproduces the *exact* same
       restriction as the decomposed path's explicit float bias. This is
       the decisive assertion: on the pre-fix code, ``local_window_size``
       was never wired into the ``GroupQueryAttention`` call at all, so
       "fused" would equal "unbounded", not "windowed", and this assertion
       would fail.
    """
    window = 2
    seq_len = 6  # > window, so the restriction actually bites.
    config = _attention_only_config(sliding_window=window)

    rng = np.random.default_rng(0)
    probe = DeepSeekV4Attention(config, layer_id=0)
    state = {}
    for name, param in probe.named_parameters():
        shape = [int(d) for d in param.shape]
        state[name] = ir.tensor((rng.standard_normal(shape) * 0.1).astype(np.float32))

    batch, hidden_size = 1, config.hidden_size
    hidden_values = (
        np.random.default_rng(1).standard_normal((batch, seq_len, hidden_size)) * 0.1
    ).astype(np.float32)
    attention_mask_values = np.ones((batch, seq_len), dtype=np.int64)

    def _run(*, fused_gqa, decomposed_sliding_window):
        return _run_attention_graph(
            config,
            hidden_values=hidden_values,
            attention_mask_values=attention_mask_values,
            state=state,
            fused_gqa=fused_gqa,
            decomposed_sliding_window=decomposed_sliding_window,
        )

    unbounded = _run(fused_gqa=False, decomposed_sliding_window=None)
    windowed = _run(fused_gqa=False, decomposed_sliding_window=window)
    fused = _run(fused_gqa=True, decomposed_sliding_window=None)

    assert not np.allclose(unbounded, windowed, atol=1e-4), (
        "sliding_window=2 must change the decomposed path's output for "
        "seq_len=6 > window -- if this passes, the fixture isn't "
        "exercising the window at all"
    )
    np.testing.assert_allclose(
        windowed,
        fused,
        atol=1e-4,
        err_msg=(
            "fused-GQA local_window_size must reproduce the exact same "
            "restriction as the decomposed path's explicit windowed bias"
        ),
    )


# ---------------------------------------------------------------------------
# Slice C1: default-off native ``pkg.nxrt::CompressedSparseAttention`` (HCA
# ratio-128) export. These are shape-faithful/structural tests only: the
# frozen op has no Python shape inference and no ORT runtime in this env, so
# we assert exact op attrs, input/output names/shapes/dtypes, threaded
# compressed state IO, real dataflow (no dead shape anchor), typed rejects,
# and a byte-identical disabled baseline.
# ---------------------------------------------------------------------------

_EXPECTED_HCA_ATTRS = {
    "num_heads": 2,
    "head_dim": 16,
    "qk_rope_head_dim": 4,
    "compression_ratio": 128,
    "index_num_heads": 0,
    "index_head_dim": 0,
    "index_topk": 0,
    "causal": 1,
    "cache_layout_version": 1,
    "index_layout_version": 1,
    "sink_mode": "logit_only",
    "cache_format": "f32",
    "scale": 0.0,
}


def _csa_nodes(graph):
    return [n for n in graph if n.op_type == "CompressedSparseAttention"]


def _named(values):
    return {v.name: v for v in values}


def test_native_csa_emits_hca_op_for_ratio128_layer_only():
    config = _tiny_config(
        num_hidden_layers=2,
        compress_ratios=[0, 128],
        native_csa=True,
    )
    graph = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")[
        "model"
    ].graph

    nodes = _csa_nodes(graph)
    assert len(nodes) == 1, "exactly the ratio-128 layer emits the native op"
    node = nodes[0]
    assert node.domain == CSA_DOMAIN == "pkg.nxrt"
    assert len(node.inputs) == 11
    assert len(node.outputs) == 3
    assert {k: node.attributes[k].value for k in node.attributes} == _EXPECTED_HCA_ATTRS
    # The native op fully replaces dense/fused attention for its layer and must
    # not silently coexist with a fused GQA/Attention path.
    assert count_op_type(graph, "GroupQueryAttention") == 0
    assert count_op_type(graph, "Attention") == 0


def test_native_csa_threads_compressed_state_io():
    config = _tiny_config(
        num_hidden_layers=2,
        compress_ratios=[0, 128],
        native_csa=True,
    )
    graph = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")[
        "model"
    ].graph

    inputs = _named(graph.inputs)
    outputs = _named(graph.outputs)

    # Compressed state is threaded as ADDITIONAL parallel IO for the enabled
    # layer only, with deterministic names and dynamic record axes.
    assert "past_compressed_kv.1" in inputs
    assert "past_compression_carry.1" in inputs
    assert "present_compressed_kv.1" in outputs
    assert "present_compression_carry.1" in outputs
    assert "past_compressed_kv.0" not in inputs
    assert "past_compression_carry.0" not in inputs

    past_kv = inputs["past_compressed_kv.1"]
    pres_kv = outputs["present_compressed_kv.1"]
    assert past_kv.dtype == ir.DataType.FLOAT
    assert pres_kv.dtype == ir.DataType.FLOAT
    assert [str(d) for d in past_kv.shape] == ["batch", "past_compressed_records", "16"]
    assert [str(d) for d in pres_kv.shape] == ["batch", "present_compressed_records", "16"]

    past_carry = inputs["past_compression_carry.1"]
    pres_carry = outputs["present_compression_carry.1"]
    assert past_carry.dtype == ir.DataType.FLOAT
    assert [int(d) for d in past_carry.shape[1:]] == [128, 2, 16]
    # Fixed carry planes are stable across the step (chainable decode state).
    assert [str(d) for d in past_carry.shape] == [str(d) for d in pres_carry.shape]

    # Dense sliding-window KV IO stays present and DISTINCT alongside the
    # compressed state (MQA data identical, but must be separate values).
    assert outputs["present.1.key"] is not outputs["present.1.value"]


def test_native_csa_present_state_is_chainable_for_decode():
    # Shape-faithful prefill + >=16 decode: present_* record axis is dynamic
    # and carry planes fixed, so present_* names/shapes chain back into past_*
    # inputs for any decode length (no static per-step sizing).
    config = _tiny_config(num_hidden_layers=2, compress_ratios=[0, 128], native_csa=True)
    graph = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")[
        "model"
    ].graph
    inputs = _named(graph.inputs)
    outputs = _named(graph.outputs)

    past_kv, pres_kv = inputs["past_compressed_kv.1"], outputs["present_compressed_kv.1"]
    # Same rank + same trailing stored width => output can feed the input.
    assert len(past_kv.shape) == len(pres_kv.shape) == 3
    assert int(past_kv.shape[2]) == int(pres_kv.shape[2]) == 16
    assert isinstance(past_kv.shape[1], ir.SymbolicDim)
    assert isinstance(pres_kv.shape[1], ir.SymbolicDim)
    past_carry, pres_carry = (
        inputs["past_compression_carry.1"],
        outputs["present_compression_carry.1"],
    )
    assert [int(d) for d in past_carry.shape[1:]] == [int(d) for d in pres_carry.shape[1:]]


def test_native_csa_f32_inputs_under_fp16():
    # Mirrors GLM DSA precedent: even when the model dtype is FLOAT16, the
    # frozen op's float inputs must be explicitly cast to FLOAT (f32 cache
    # format); integer length inputs keep their integer dtype.
    config = _tiny_config(
        num_hidden_layers=2,
        compress_ratios=[0, 128],
        native_csa=True,
        dtype=ir.DataType.FLOAT16,
    )
    graph = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")[
        "model"
    ].graph
    node = _csa_nodes(graph)[0]

    float_input_positions = [0, 1, 2, 3, 4, 5, 6, 7, 10]
    for pos in float_input_positions:
        assert node.inputs[pos].dtype == ir.DataType.FLOAT, (
            f"CSA input[{pos}] must be FLOAT under a FLOAT16 model dtype"
        )
    assert node.inputs[8].dtype == ir.DataType.INT32  # seqlens_k
    assert node.inputs[9].dtype == ir.DataType.INT64  # total_sequence_length


def test_native_csa_no_dead_anchor_for_hca_layer():
    # The enabled layer's compressor weights must feed REAL dataflow into the
    # native op (not be retained only as a zero-valued shape anchor).
    config = _tiny_config(
        num_hidden_layers=2,
        compress_ratios=[0, 128],
        native_csa=True,
    )
    graph = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")[
        "model"
    ].graph
    node = _csa_nodes(graph)[0]
    initializers = set(graph.initializers)

    # compressor_kv / compressor_gate are produced by real MatMul projections.
    assert node.inputs[2].producer() is not None
    assert node.inputs[2].producer().op_type == "MatMul"
    assert node.inputs[3].producer() is not None
    assert node.inputs[3].producer().op_type == "MatMul"
    # compressor_ape / compressor_norm are wired straight from the preserved
    # initializers (not zero anchors).
    assert node.inputs[4].name == "model.layers.1.self_attn.compressor.ape"
    assert node.inputs[5].name == "model.layers.1.self_attn.compressor.norm.weight"
    # Weights survive by actual dataflow, not a discarded shape anchor.
    assert "model.layers.1.self_attn.compressor.wkv.weight" in initializers
    assert "model.layers.1.self_attn.compressor.wgate.weight" in initializers
    assert "model.layers.1.self_attn.compressor.ape" in initializers
    assert "model.layers.1.self_attn.compressor.norm.weight" in initializers


def _fill_and_serialize(model):
    for name in list(model.graph.initializers):
        value = model.graph.initializers[name]
        if value.const_value is None:
            shape = tuple(int(d) for d in value.shape)
            dtype = value.dtype.numpy() if value.dtype is not None else np.float32
            value.const_value = ir.tensor(np.zeros(shape, dtype=dtype), name=name)
    return ir.to_proto(model).SerializeToString()


def test_native_csa_disabled_is_byte_identical():
    # Feature off => byte-identical to the existing dense correctness export,
    # and no CSA node / no compressed IO leaks in.
    ratios = [0, 0, 4, 128]
    cfg_unset = _tiny_config(num_hidden_layers=4, compress_ratios=ratios)
    cfg_false = _tiny_config(num_hidden_layers=4, compress_ratios=ratios, native_csa=False)
    model_unset = build_from_module(
        DeepSeekV4CausalLMModel(cfg_unset), cfg_unset, task="deepseek-v4"
    )["model"]
    model_false = build_from_module(
        DeepSeekV4CausalLMModel(cfg_false), cfg_false, task="deepseek-v4"
    )["model"]

    assert cfg_unset.native_csa is False
    assert count_op_type(model_unset.graph, "CompressedSparseAttention") == 0
    assert not any("compress" in v.name for v in model_unset.graph.inputs)
    assert not any("compress" in v.name for v in model_unset.graph.outputs)
    assert _fill_and_serialize(model_unset) == _fill_and_serialize(model_false)


def test_native_csa_defaults_off_and_opt_in():
    assert _tiny_config().native_csa is False
    assert _tiny_config(native_csa=True).native_csa is True


def test_native_csa_rejects_ratio4():
    # ratio-4 CSA (learned FP4 indexer + top-k) is out of C1 scope: requesting
    # native export for it must fail closed at construction, never emit dense.
    config = _tiny_config(num_hidden_layers=2, compress_ratios=[0, 4], native_csa=True)
    with pytest.raises(NativeCsaExportError):
        DeepSeekV4CausalLMModel(config)


def test_native_csa_rejects_quantized():
    config = _tiny_config(
        num_hidden_layers=2,
        compress_ratios=[0, 128],
        native_csa=True,
        quantization=QuantizationConfig(bits=4, group_size=16, quant_method="gptq", sym=True),
    )
    with pytest.raises(NativeCsaExportError):
        DeepSeekV4CausalLMModel(config)


def test_plan_native_csa_rejects_mtp():
    # MTP compressed-state recurrence is not modeled in C1.
    config = _tiny_config(num_hidden_layers=1, compress_ratios=[128], native_csa=True)
    with pytest.raises(NativeCsaExportError):
        plan_native_csa(config, 0, is_mtp=True)


def test_plan_native_csa_rejects_unknown_ratio():
    # Unknown compression ratios cannot match the frozen v1 contract. Call the
    # planner directly to exercise its branch (the model __init__ ValueError
    # guard would otherwise reject an unknown ratio before planning).
    config = _tiny_config(num_hidden_layers=1, compress_ratios=[0], native_csa=True)
    object.__setattr__(config, "compress_ratios", [7])
    with pytest.raises(NativeCsaExportError):
        plan_native_csa(config, 0)


def test_plan_native_csa_off_and_dense_layers_return_none():
    # Fail-closed only when requested: feature off or a ratio-0 dense layer
    # legitimately opts out (returns None), not a typed error.
    off = _tiny_config(num_hidden_layers=1, compress_ratios=[128], native_csa=False)
    assert plan_native_csa(off, 0) is None
    dense = _tiny_config(num_hidden_layers=1, compress_ratios=[0], native_csa=True)
    assert plan_native_csa(dense, 0) is None


def test_plan_native_csa_ratio128_layer_matches_contract():
    config = _tiny_config(num_hidden_layers=2, compress_ratios=[0, 128], native_csa=True)
    plan = plan_native_csa(config, 1)
    assert plan is not None
    assert plan.layer_id == 1
    assert plan.num_heads == 2
    assert plan.head_dim == 16
    assert plan.qk_rope_head_dim == 4
    assert plan.compression_ratio == HCA_COMPRESSION_RATIO == 128
    assert plan.cache_format == "f32"
    assert plan.past_compressed_kv_name == "past_compressed_kv.1"
    assert plan.present_compression_carry_name == "present_compression_carry.1"
