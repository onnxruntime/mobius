# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.gguf._arch_registry import Support, get_arch_spec
from mobius.integrations.gguf._builder import (
    _normalize_gguf_weights,
    _raise_for_invalid_moe_cohort_tensor_contract,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.tasks import CausalLMTask, GGUFEncoderFeatureExtractionTask

_IMPORT_EVIDENCE = {
    "arctic": {
        "repository": "ggml-org/llama.cpp",
        "revision": "8d9af256337d1a501250f9bbf4c0859a654bddd6",
        "filename": "synthetic architecture fixture",
        "size": 0,
        "sha256": None,
        "downloaded": 0,
    },
    "dbrx": {
        "repository": "mradermacher/dbrx-instruct-i1-GGUF",
        "revision": "8b95d1b1d9c4aa65e5edd8bcc74b15a47f6cdeae",
        "filename": "dbrx-instruct.i1-IQ1_S.gguf",
        "size": 26_896_613_856,
        "sha256": "5f9e5571027abb3eb85594a0cd2444048cb377a9a7384cc6ab30719e79d0df11",
        "downloaded": 64 * 1024**2,
    },
    "ernie4_5-moe": {
        "repository": "mradermacher/ERNIE-4.5-21B-A3B-Thinking-i1-GGUF",
        "revision": "c9901475a93c03aeff985750153c93ac9a325e1a",
        "filename": "ERNIE-4.5-21B-A3B-Thinking.i1-IQ1_S.gguf",
        "size": 4_476_480_032,
        "sha256": "eccafe7b09e0cf42fc6eda50ec7ebaf28bb6c47e63b1a4b09a5a27e60cd425ae",
        "downloaded": 4_476_480_032,
    },
    "nomic-bert-moe": {
        "repository": "nomic-ai/nomic-embed-text-v2-moe-GGUF",
        "revision": "ffbcf4c99e5d617dda10ec8c0e9f75754b0cbb80",
        "filename": "nomic-embed-text-v2-moe.Q2_K.gguf",
        "size": 273_286_112,
        "sha256": "11843331c8f0d14dca2be4809e1146a3a0411892e33b78ae6971b3f619f8e78b",
        "downloaded": 273_286_112,
    },
}


class _FakeGGUF:
    def __init__(
        self,
        architecture: str,
        metadata: dict[str, object],
        tensors: dict[str, tuple[int, ...]],
    ):
        self.architecture = architecture
        self.metadata = metadata
        self.tensor_names = list(tensors)
        self._tensors = tensors

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        for name, shape in self._tensors.items():
            yield name, None, SimpleNamespace(value=0, name="F32"), shape


def _attention_tensors(
    layer: int,
    *,
    hidden: int,
    query_width: int,
    kv_width: int,
    fused: bool = False,
    bias: bool = False,
) -> dict[str, tuple[int, ...]]:
    prefix = f"blk.{layer}."
    tensors = {
        prefix + "attn_output.weight": (hidden, query_width),
    }
    if fused:
        width = query_width + 2 * kv_width
        tensors[prefix + "attn_qkv.weight"] = (width, hidden)
        if bias:
            tensors[prefix + "attn_qkv.bias"] = (width,)
    else:
        for projection, width in (("q", query_width), ("k", kv_width), ("v", kv_width)):
            tensors[prefix + f"attn_{projection}.weight"] = (width, hidden)
            if bias:
                tensors[prefix + f"attn_{projection}.bias"] = (width,)
    return tensors


def _fixture(architecture: str) -> _FakeGGUF:
    hidden, intermediate, expert_intermediate = 8, 12, 6
    experts, layers, vocab = 4, 4, 24
    heads, kv_heads, head_dim = 2, 1, 4
    if architecture == "nomic-bert-moe":
        kv_heads = heads
    query_width, kv_width = heads * head_dim, kv_heads * head_dim
    metadata: dict[str, object] = {
        f"{architecture}.context_length": 32,
        f"{architecture}.embedding_length": hidden,
        f"{architecture}.feed_forward_length": intermediate,
        f"{architecture}.block_count": layers,
        f"{architecture}.attention.head_count": heads,
        f"{architecture}.attention.head_count_kv": kv_heads,
        f"{architecture}.rope.dimension_count": head_dim,
        f"{architecture}.vocab_size": vocab,
        f"{architecture}.expert_count": experts,
        f"{architecture}.expert_used_count": 2,
    }
    tensors: dict[str, tuple[int, ...]] = {"token_embd.weight": (vocab, hidden)}

    if architecture == "arctic":
        metadata[f"{architecture}.attention.layer_norm_rms_epsilon"] = 1e-5
        tensors["output_norm.weight"] = (hidden,)
        expert_intermediate = intermediate
        for layer in range(layers):
            prefix = f"blk.{layer}."
            tensors.update(
                {
                    **_attention_tensors(
                        layer,
                        hidden=hidden,
                        query_width=query_width,
                        kv_width=kv_width,
                    ),
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "ffn_norm.weight": (hidden,),
                    prefix + "ffn_gate.weight": (hidden, hidden),
                    prefix + "ffn_up.weight": (hidden, hidden),
                    prefix + "ffn_down.weight": (hidden, hidden),
                    prefix + "ffn_norm_exps.weight": (hidden,),
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_gate_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        hidden,
                        expert_intermediate,
                    ),
                }
            )
        return _FakeGGUF(architecture, metadata, tensors)

    if architecture == "dbrx":
        expert_intermediate = intermediate
        metadata.update(
            {
                f"{architecture}.attention.layer_norm_epsilon": 1e-5,
                f"{architecture}.attention.clamp_kqv": 8.0,
            }
        )
        tensors.update(
            {
                "output_norm.weight": (hidden,),
                "output.weight": (vocab, hidden),
            }
        )
        for layer in range(layers):
            prefix = f"blk.{layer}."
            tensors.update(
                {
                    **_attention_tensors(
                        layer,
                        hidden=hidden,
                        query_width=query_width,
                        kv_width=kv_width,
                        fused=True,
                    ),
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_output_norm.weight": (hidden,),
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_gate_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        hidden,
                        expert_intermediate,
                    ),
                }
            )
        return _FakeGGUF(architecture, metadata, tensors)

    if architecture == "ernie4_5-moe":
        metadata.update(
            {
                f"{architecture}.attention.layer_norm_rms_epsilon": 1e-5,
                f"{architecture}.expert_feed_forward_length": expert_intermediate,
                f"{architecture}.expert_shared_count": 1,
                f"{architecture}.expert_shared_feed_forward_length": expert_intermediate,
                f"{architecture}.leading_dense_block_count": 1,
                f"{architecture}.interleave_moe_layer_step": 2,
            }
        )
        tensors.update(
            {
                "output_norm.weight": (hidden,),
                "output.weight": (vocab, hidden),
            }
        )
        for layer in range(layers):
            prefix = f"blk.{layer}."
            tensors.update(
                {
                    **_attention_tensors(
                        layer,
                        hidden=hidden,
                        query_width=query_width,
                        kv_width=kv_width,
                    ),
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "ffn_norm.weight": (hidden,),
                }
            )
            if layer >= 1 and (layer + 1) % 2 == 0:
                tensors.update(
                    {
                        prefix + "ffn_gate_inp.weight": (experts, hidden),
                        prefix + "exp_probs_b.bias": (experts,),
                        prefix + "ffn_gate_exps.weight": (
                            experts,
                            expert_intermediate,
                            hidden,
                        ),
                        prefix + "ffn_up_exps.weight": (
                            experts,
                            expert_intermediate,
                            hidden,
                        ),
                        prefix + "ffn_down_exps.weight": (
                            experts,
                            hidden,
                            expert_intermediate,
                        ),
                        prefix + "ffn_gate_shexp.weight": (expert_intermediate, hidden),
                        prefix + "ffn_up_shexp.weight": (expert_intermediate, hidden),
                        prefix + "ffn_down_shexp.weight": (hidden, expert_intermediate),
                    }
                )
            else:
                tensors.update(
                    {
                        prefix + "ffn_gate.weight": (intermediate, hidden),
                        prefix + "ffn_up.weight": (intermediate, hidden),
                        prefix + "ffn_down.weight": (hidden, intermediate),
                    }
                )
        return _FakeGGUF(architecture, metadata, tensors)

    assert architecture == "nomic-bert-moe"
    expert_intermediate = intermediate
    metadata.pop(f"{architecture}.rope.dimension_count")
    metadata.update(
        {
            f"{architecture}.attention.causal": False,
            f"{architecture}.attention.layer_norm_epsilon": 1e-5,
            f"{architecture}.moe_every_n_layers": 2,
            f"{architecture}.pooling_type": 1,
            f"{architecture}.rope.freq_base": 1000.0,
            "tokenizer.ggml.token_type_count": 1,
        }
    )
    tensors.update(
        {
            "token_types.weight": (hidden,),
            "token_embd_norm.weight": (hidden,),
            "token_embd_norm.bias": (hidden,),
        }
    )
    for layer in range(layers):
        prefix = f"blk.{layer}."
        tensors.update(
            {
                **_attention_tensors(
                    layer,
                    hidden=hidden,
                    query_width=query_width,
                    kv_width=kv_width,
                    fused=True,
                    bias=True,
                ),
                prefix + "attn_output.bias": (hidden,),
                prefix + "attn_output_norm.weight": (hidden,),
                prefix + "attn_output_norm.bias": (hidden,),
                prefix + "layer_output_norm.weight": (hidden,),
                prefix + "layer_output_norm.bias": (hidden,),
            }
        )
        if layer % 2 == 1:
            tensors.update(
                {
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_up_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        hidden,
                        expert_intermediate,
                    ),
                }
            )
        else:
            tensors.update(
                {
                    prefix + "ffn_up.weight": (intermediate, hidden),
                    prefix + "ffn_up.bias": (intermediate,),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                    prefix + "ffn_down.bias": (hidden,),
                }
            )
    return _FakeGGUF(architecture, metadata, tensors)


@pytest.mark.parametrize(
    ("architecture", "module_type"),
    [
        ("arctic", "arctic_gguf"),
        ("dbrx", "dbrx_gguf"),
        ("ernie4_5-moe", "ernie4_5_moe_gguf"),
        ("nomic-bert-moe", "nomic_bert_moe_gguf"),
    ],
)
def test_cohort_routes_are_float_only_and_runtime_deferred(
    architecture: str, module_type: str
) -> None:
    spec = get_arch_spec(architecture)
    assert spec.is_importable
    assert spec.module_type == module_type
    assert spec.quantized_import is Support.REJECTED
    assert spec.runtime is Support.DEFERRED


def test_cohort_import_evidence_is_immutable_and_within_download_budget() -> None:
    assert set(_IMPORT_EVIDENCE) == {
        "arctic",
        "dbrx",
        "ernie4_5-moe",
        "nomic-bert-moe",
    }
    assert (
        sum(int(record["downloaded"]) for record in _IMPORT_EVIDENCE.values()) < 16 * 1024**3
    )
    for architecture, record in _IMPORT_EVIDENCE.items():
        assert len(str(record["revision"])) == 40
        if architecture == "arctic":
            assert record["sha256"] is None
        else:
            assert len(str(record["sha256"])) == 64


def test_cohort_records_architecture_specific_rope_layouts() -> None:
    arctic = get_arch_spec("arctic")
    assert arctic.tensor_processor == "llama"
    assert arctic.llama_qk_permute
    assert not arctic.rope_interleave

    dbrx = get_arch_spec("dbrx")
    assert not dbrx.llama_qk_permute
    assert not dbrx.rope_interleave

    ernie = get_arch_spec("ernie4_5-moe")
    assert ernie.rope_interleave


@pytest.mark.parametrize(
    "architecture",
    ["arctic", "dbrx", "ernie4_5-moe", "nomic-bert-moe"],
)
def test_cohort_exact_tensor_closure_and_tiny_graph(architecture: str) -> None:
    source = _fixture(architecture)
    _raise_for_invalid_moe_cohort_tensor_contract(source)
    config = gguf_to_config(source)
    spec = get_arch_spec(architecture)
    module = registry.get(spec.module_type)(config)
    task = (
        GGUFEncoderFeatureExtractionTask()
        if architecture == "nomic-bert-moe"
        else CausalLMTask()
    )
    graph = task.build(module, config)["model"].graph

    mapped_non_experts = {
        map_gguf_to_hf_names(name, architecture)
        for name in source.tensor_names
        if "_exps." not in name
    }
    parameters = {name for name, _parameter in module.named_parameters()}
    assert mapped_non_experts <= parameters
    assert not any("com.microsoft" in node.domain for node in graph)
    if architecture == "nomic-bert-moe":
        assert [value.name for value in graph.outputs] == ["sentence_embedding"]
        assert not any("past_key_values" in value.name for value in graph.inputs)
    else:
        assert graph.outputs[0].name == "logits"


@pytest.mark.parametrize(
    "architecture",
    ["arctic", "dbrx", "ernie4_5-moe", "nomic-bert-moe"],
)
def test_cohort_stacked_experts_unpack_to_graph_parameters(architecture: str) -> None:
    source = _fixture(architecture)
    config = gguf_to_config(source)
    module = registry.get(get_arch_spec(architecture).module_type)(config)
    stacked_experts = {
        map_gguf_to_hf_names(name, architecture): torch.zeros(shape)
        for name, shape in source._tensors.items()
        if any(
            projection in name
            for projection in ("ffn_gate_exps.", "ffn_up_exps.", "ffn_down_exps.")
        )
    }

    normalized = _normalize_gguf_weights(stacked_experts, architecture, config)
    expected = {name for name, _parameter in module.named_parameters() if ".experts." in name}

    assert set(normalized) == expected


def test_arctic_moe_normalization_rejects_malformed_router_and_experts() -> None:
    source = _fixture("arctic")
    config = gguf_to_config(source)
    router = map_gguf_to_hf_names("blk.0.ffn_gate_inp.weight", "arctic")
    experts = map_gguf_to_hf_names("blk.0.ffn_gate_exps.weight", "arctic")

    with pytest.raises(ValueError, match="Invalid router shape"):
        _normalize_gguf_weights({router: torch.zeros(3, 8)}, "arctic", config)
    with pytest.raises(ValueError, match="Invalid stacked expert shape"):
        _normalize_gguf_weights({experts: torch.zeros(4, 11, 8)}, "arctic", config)


@pytest.mark.parametrize(
    "architecture",
    ["arctic", "dbrx", "ernie4_5-moe", "nomic-bert-moe"],
)
@pytest.mark.parametrize("mutation", ["missing", "unexpected", "malformed"])
def test_cohort_tensor_contract_rejects_mutations(architecture: str, mutation: str) -> None:
    source = _fixture(architecture)
    if mutation == "missing":
        name = next(name for name in source._tensors if name.startswith("blk.0."))
        del source._tensors[name]
    elif mutation == "unexpected":
        source._tensors["blk.0.unowned.weight"] = (8, 8)
    else:
        name = next(
            name
            for name in source._tensors
            if name.startswith("blk.0.") and name.endswith(".weight")
        )
        source._tensors[name] = (1,)
    source.tensor_names = list(source._tensors)

    with pytest.raises(ValueError, match=r"missing|unexpected|malformed"):
        _raise_for_invalid_moe_cohort_tensor_contract(source)


def test_cohort_config_preserves_routing_and_schedule_semantics() -> None:
    arctic = gguf_to_config(_fixture("arctic"))
    assert arctic.norm_topk_prob
    assert arctic.moe_intermediate_size is None

    dbrx = gguf_to_config(_fixture("dbrx"))
    assert dbrx.norm_topk_prob
    assert dbrx.attention_clamp == pytest.approx(8.0)
    assert dbrx.tie_word_embeddings is False

    ernie = gguf_to_config(_fixture("ernie4_5-moe"))
    assert ernie.moe_layer_frequency == 2
    assert ernie.first_k_dense_replace == 1
    assert ernie.use_expert_bias
    assert ernie.norm_topk_prob
    assert ernie.routing_weight_normalization_floor == pytest.approx(6.103515625e-5)

    nomic = gguf_to_config(_fixture("nomic-bert-moe"))
    assert nomic.moe_layer_frequency == 2
    assert nomic.norm_topk_prob is False
    assert nomic.hidden_act == "gelu_pytorch_tanh"


def test_ernie_rejects_ignored_attention_output_bias() -> None:
    source = _fixture("ernie4_5-moe")
    for layer in range(int(source.metadata["ernie4_5-moe.block_count"])):
        source._tensors[f"blk.{layer}.attn_output.bias"] = (8,)
    source.tensor_names = list(source._tensors)

    with pytest.raises(ValueError, match="unexpected"):
        _raise_for_invalid_moe_cohort_tensor_contract(source)


def test_nomic_requires_positive_token_type_count() -> None:
    source = _fixture("nomic-bert-moe")
    del source.metadata["tokenizer.ggml.token_type_count"]

    with pytest.raises(ValueError, match=r"tokenizer\.ggml\.token_type_count"):
        _raise_for_invalid_moe_cohort_tensor_contract(source)


@pytest.mark.parametrize(
    "architecture",
    ["arctic", "dbrx", "ernie4_5-moe", "nomic-bert-moe"],
)
def test_cohort_tiny_float_graph_executes_on_cpu(architecture: str) -> None:
    source = _fixture(architecture)
    config = gguf_to_config(source)
    module = registry.get(get_arch_spec(architecture).module_type)(config)
    task = (
        GGUFEncoderFeatureExtractionTask()
        if architecture == "nomic-bert-moe"
        else CausalLMTask()
    )
    model = task.build(module, config)["model"]
    rng = np.random.default_rng(0)
    for _name, parameter in module.named_parameters():
        values = rng.standard_normal(tuple(parameter.shape)).astype(np.float32) * 0.02
        parameter.const_value = ir.tensor(values)

    feeds = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.ones((1, 1), dtype=np.int64),
    }
    if architecture == "nomic-bert-moe":
        feeds["token_type_ids"] = np.zeros((1, 1), dtype=np.int64)
        expected_output = "sentence_embedding"
    else:
        feeds["position_ids"] = np.zeros((1, 1), dtype=np.int64)
        for layer in range(config.num_hidden_layers):
            shape = (1, config.num_key_value_heads, 0, config.head_dim)
            feeds[f"past_key_values.{layer}.key"] = np.empty(shape, dtype=np.float32)
            feeds[f"past_key_values.{layer}.value"] = np.empty(shape, dtype=np.float32)
        expected_output = "logits"

    output = OnnxModelSession(model).run(feeds)[expected_output]
    expected_shape = (
        (1, config.hidden_size) if architecture == "nomic-bert-moe" else (1, 1, 24)
    )
    assert output.shape == expected_shape
    assert np.isfinite(output).all()
