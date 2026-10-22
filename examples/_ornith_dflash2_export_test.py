# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Ornith DFlash2 target taps, FP8 projections and experiment isolation."""

import copy
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch
from ornith_dflash2_export import OrnithDFlash2Builder, add_target_taps
from ornith_engine_benchmark import merge_overlay


def target_model():
    graph = ir.Graph(inputs=[], outputs=[], nodes=[], opset_imports={"": 24})
    for i in range(48):
        hidden = ir.Value(
            name=f"hidden.{i}",
            type=ir.TensorType(ir.DataType.FLOAT16),
            shape=ir.Shape(["num_tokens", 3072]),
        )
        graph.inputs.append(hidden)
        value = ir.Value(name=f"norm.{i}")
        graph.append(
            ir.Node(
                "",
                "Cast",
                [hidden],
                {"to": ir.AttrInt64("to", 1)},
                outputs=[value],
                name=f"decoder/model/layers.{i}/input_layernorm/Cast_node",
            )
        )
    return ir.Model(graph, ir_version=10)


def test_target_taps_use_post_residual_states_in_checkpoint_order():
    model = target_model()
    taps = add_target_taps(model, [7, 1, 45], 3072)
    assert taps == [model.graph.inputs[i] for i in (8, 2, 46)]
    concat = list(model.graph)[-1]
    assert concat.attributes["axis"].as_int() == -1
    assert list(concat.inputs) == taps
    assert model.graph.outputs[0].shape == ir.Shape(["num_tokens", 9216])
    with pytest.raises(ValueError, match="already"):
        add_target_taps(model, [1], 3072)


@pytest.mark.parametrize("ids", [[], [1, 1], [-1], [47], [48]])
def test_invalid_tap_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="tap IDs"):
        add_target_taps(target_model(), ids, 3072)


def test_missing_or_incompatible_target_tap_is_rejected():
    model = target_model()
    with pytest.raises(ValueError, match="incompatible"):
        add_target_taps(model, [1], 64)
    model.graph.inputs[2].dtype = ir.DataType.FLOAT
    with pytest.raises(ValueError, match="incompatible"):
        add_target_taps(model, [1], 3072)
    list(model.graph)[2].name = "unexpected"
    with pytest.raises(ValueError, match="exactly one"):
        add_target_taps(model, [1], 3072)


def test_fp8_body_reuses_weight_and_keeps_sensitive_projections_dense():
    builder = object.__new__(OrnithDFlash2Builder)
    builder.fp8 = True
    builder.io_dtype = ir.DataType.BFLOAT16
    builder.make_graph("test", "dflash2")
    weight = torch.randn(16, 128)
    for name in ("/ctx_k", "/query_k"):
        builder.matmul(name, "x", weight, 128, 16, "rows", weight_name="shared.weight")
    builder.matmul("/selector", "x", weight, 128, 16, "rows", quantize=False)
    nodes = list(builder.graph)
    assert [node.op_type for node in nodes] == [
        "MatMulBlockQuantizedFp8Weight",
        "MatMulBlockQuantizedFp8Weight",
        "MatMul",
    ]
    assert nodes[0].inputs[1] is nodes[1].inputs[1]
    assert nodes[0].inputs[1].dtype == ir.DataType.FLOAT8E4M3FN
    assert nodes[0].inputs[2].dtype == ir.DataType.FLOAT
    assert nodes[0].inputs[2].shape == ir.Shape([16, 1])
    assert nodes[2].inputs[1].dtype == ir.DataType.BFLOAT16


@pytest.mark.parametrize("width", [1, 7, 15])
@pytest.mark.parametrize("fp32_conv", [False, True])
def test_draft_geometry_has_distinct_files_and_preserves_runtime_cap(
    tmp_path, width, fp32_conv
):
    import json

    config = {
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "intermediate_size": 64,
        "vocab_size": 64,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000},
        "dflash_config": {
            "conv_kernel_size": 2,
            "conv_group_size": 4,
            "selector_rank": 4,
            "selector_top_k": 2,
            "mask_token_id": 63,
            "target_layer_ids": [1],
            "block_size": 16,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    builder = OrnithDFlash2Builder(
        tmp_path,
        target_model(),
        num_draft_tokens=width,
        fp32_conv=fp32_conv,
    )
    assert builder.block_size == width + 1
    assert builder.num_draft_tokens == width
    assert builder.conv_fp32_accumulation is fp32_conv
    suffix = f"-block{width + 1}" if width != 7 else ""
    suffix += "-fp32conv" if fp32_conv else ""
    assert builder.filename == f"dflash2-fp8{suffix}.onnx"


def test_experiment_overlay_preserves_drafter_and_does_not_mutate_default():
    base = {
        "model": {
            "mtp": {"enabled": False},
            "dflash2": {"filename": "draft.onnx"},
            "decoder": {"filename": "target.onnx", "outputs": {"aux_hidden_states": "aux"}},
        },
        "speculative": {"max_draft_tokens": 7},
    }
    original = copy.deepcopy(base)
    result = merge_overlay(base, {"model": {"decoder": {"session_options": {"enable": "1"}}}})
    assert base == original
    assert result["model"]["dflash2"] == base["model"]["dflash2"]
    assert result["model"]["decoder"]["outputs"] == {"aux_hidden_states": "aux"}
    assert result["model"]["decoder"]["filename"] == "target.onnx"
    assert result["speculative"]["max_draft_tokens"] == 7


@pytest.mark.parametrize("failure", ["dflash2_failures", "dflash2_disables"])
def test_drafter_failures_cannot_produce_a_successful_benchmark(monkeypatch, failure):
    import ornith_engine_benchmark as benchmark

    class Options:
        def __init__(self, request):
            pass

        def set_do_sample(self, value):
            pass

        def set_min_generated_tokens(self, value):
            pass

        def set_max_generated_tokens(self, value):
            pass

    class Engine:
        done = False

        def get_speculative_stats(self):
            counters = dict.fromkeys(
                (
                    "draft_tokens_proposed",
                    "draft_tokens_evaluated",
                    "draft_tokens_accepted",
                    "mtp_failures",
                    "dflash2_failures",
                    "dflash2_disables",
                    "draft_forward_passes",
                    "target_forward_passes",
                    "zero_accept_rounds",
                    "partial_accept_rounds",
                    "full_accept_rounds",
                ),
                0,
            )
            if self.done:
                counters.update(draft_tokens_proposed=2, draft_tokens_evaluated=2)
                counters[failure] = 1
            return counters

        def create_request(self):
            return SimpleNamespace(begin_turn=lambda *args: None, close=lambda: None)

        def create_event_buffer(self, capacity):
            return None

        def has_pending_requests(self):
            return not self.done

        def run(self, buffer):
            self.done = True
            return [
                SimpleNamespace(flags=flag, token=1)
                for flag in (
                    benchmark.og.EngineEventFlags.TOKEN,
                    benchmark.og.EngineEventFlags.TOKEN,
                    benchmark.og.EngineEventFlags.TURN_FINISHED,
                )
            ]

    monkeypatch.setattr(benchmark.og, "TurnOptions", Options)
    with pytest.raises(RuntimeError, match="drafting reported failures"):
        benchmark.run_turn(Engine(), [1, 2], 2)
