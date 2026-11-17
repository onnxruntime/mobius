# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native packed-state parity and pinned Engine probes (CUDA, no Hub downloads).

Run with pytest -m integration tests/integration/paged_hybrid_test.py.
The Engine probe additionally requires MOBIUS_GENAI_REVISION to identify the
source build being qualified. These tests are not a full 27B checkpoint run.
"""

from __future__ import annotations

import os

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from mobius import build_from_module
from mobius._testing import make_config
from mobius.components._paged_attention import GENAI_REVISION, ORT_REVISION
from mobius.integrations.ort_genai.auto_export import _write_genai_config
from mobius.models.qwen35 import Qwen35CausalLMModel
from mobius.tasks import HybridCausalLMTask, PagedHybridCausalLMTask

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def cuda_runtime():
    if (
        not torch.cuda.is_available()
        or "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        pytest.skip("Native packed hybrid operators require a CUDA ORT build")
    if os.environ.get("MOBIUS_ORT_REVISION") != ORT_REVISION:
        pytest.skip(f"Set MOBIUS_ORT_REVISION={ORT_REVISION} for the pinned CUDA build")


def _tiny_config(dtype=ir.DataType.FLOAT16):
    return make_config(
        model_type="qwen3_5_text",
        dtype=dtype,
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=192,
        vocab_size=128,
        max_position_embeddings=2048,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        partial_rotary_factor=0.5,
        mrope_section=[8, 4, 4],
        mrope_interleaved=True,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )


def _package(dtype, *, packed=True, prune=False):
    config = _tiny_config(dtype)
    module = Qwen35CausalLMModel(config)
    rng = np.random.default_rng(731)
    for name, parameter in module.named_parameters():
        if parameter.const_value is not None:
            continue
        data = rng.normal(0, 0.04, tuple(parameter.shape)).astype(np.float32)
        if name.endswith("linear_attn.norm.weight"):
            data += 1
        parameter.const_value = ir.tensor(data)
    task = (
        PagedHybridCausalLMTask(prune_prefill_prefix=prune) if packed else HybridCausalLMTask()
    )
    return build_from_module(
        module, config, task, execution_provider="cuda" if packed else "cpu"
    )


def _session(package, path, provider):
    ir.save(package["model"], path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=[provider])


def _torch_type(dtype):
    return torch.float16 if dtype == ir.DataType.FLOAT16 else torch.bfloat16


def _run_packed(session, feed, *, pruned, vocab_size):
    """Bind page outputs onto inputs, as the native kernel requires."""
    binding = session.io_binding()
    element_types = {
        torch.float16: 10,
        torch.bfloat16: 16,
        torch.float32: 1,
        torch.int64: 7,
        torch.int32: 6,
    }
    tensors = {}
    for name, value in feed.items():
        tensor = value.contiguous()
        tensors[name] = tensor
        binding.bind_input(
            name,
            tensor.device.type,
            0,
            element_types[tensor.dtype],
            tuple(tensor.shape),
            tensor.data_ptr(),
        )
    outputs = {}
    for output in session.get_outputs():
        name = output.name
        if name == "logits":
            rows = len(feed["past_sequence_lengths"]) if pruned else len(feed["input_ids"])
            tensor = torch.empty((rows, vocab_size), dtype=torch.float32, device="cuda")
        else:
            past = feed[name.replace("present.", "past_key_values.", 1)]
            tensor = past if name.endswith((".key", ".value")) else torch.empty_like(past)
        binding.bind_output(
            name,
            "cuda",
            0,
            element_types[tensor.dtype],
            tuple(tensor.shape),
            tensor.data_ptr(),
        )
        outputs[name] = tensor
    session.run_with_iobinding(binding)
    torch.cuda.synchronize()
    return outputs


def _initial_fixed(config, rng=None):
    result = {}
    conv_dim = (
        2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    for layer in range(3):
        for suffix, shape in (
            ("conv_state", (1, conv_dim, config.linear_conv_kernel_dim - 1)),
            (
                "recurrent_state",
                (
                    1,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                ),
            ),
        ):
            result[f"past_key_values.{layer}.{suffix}"] = (
                np.zeros(shape, np.float32)
                if rng is None
                else rng.normal(0, 0.02, shape).astype(np.float32)
            )
    return result


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_packed_prefill_continuation_reorder_and_page_reuse(dtype, tmp_path):
    if dtype == ir.DataType.BFLOAT16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 requires a supported CUDA device")
    dense_pkg = _package(ir.DataType.FLOAT, packed=False)
    packed_pkg = _package(dtype)
    pruned_pkg = _package(dtype, prune=True)
    dense = _session(dense_pkg, tmp_path / "dense.onnx", "CPUExecutionProvider")
    packed = _session(packed_pkg, tmp_path / "packed.onnx", "CUDAExecutionProvider")
    pruned = _session(pruned_pkg, tmp_path / "pruned.onnx", "CUDAExecutionProvider")
    config = packed_pkg.config
    compute = _torch_type(dtype)
    rng = np.random.default_rng(42)
    # A uses nonconsecutive physical pages. B's page is later reused for C
    # without clearing it, so masking must prevent stale-token contamination.
    pages = {"A": [2, 0], "B": [1], "C": [1]}
    page_shape = (4, 256, config.num_key_value_heads, config.head_dim)
    caches = [
        {
            f"past_key_values.3.{kind}": torch.zeros(page_shape, device="cuda", dtype=compute)
            for kind in ("key", "value")
        }
        for _ in range(2)
    ]
    fixed = [{}, {}]
    dense_state = {}
    lengths = {}
    tolerance = 0.025 if dtype == ir.DataType.FLOAT16 else 0.09
    schedule = [
        [("A", 255), ("B", 3)],
        [("B", 2), ("A", 3)],
        [("A", 1), ("C", 2)],
        [("C", 3), ("A", 2)],
    ]
    for step, requests in enumerate(schedule):
        ids, expected = [], []
        past_lengths = []
        for request, count in requests:
            if request not in lengths:
                lengths[request] = 0
                initial = _initial_fixed(config, rng if request != "C" else None)
                dense_state[request] = {
                    **initial,
                    **{
                        f"past_key_values.3.{kind}": np.zeros(
                            (1, config.num_key_value_heads, 0, config.head_dim), np.float32
                        )
                        for kind in ("key", "value")
                    },
                }
                for state in fixed:
                    state[request] = {
                        name: torch.as_tensor(
                            value.transpose(0, 1, 3, 2).copy()
                            if name.endswith("recurrent_state")
                            else value,
                            device="cuda",
                            dtype=torch.float32
                            if name.endswith("recurrent_state")
                            else compute,
                        )
                        for name, value in initial.items()
                    }
            past = lengths[request]
            past_lengths.append(past)
            tokens = rng.integers(1, config.vocab_size - 1, count, dtype=np.int64)
            ids.append(tokens)
            feed = {
                **dense_state[request],
                "input_ids": tokens[None],
                "position_ids": np.arange(past, past + count, dtype=np.int64)[None],
                "attention_mask": np.ones((1, past + count), np.int64),
            }
            result = dict(zip([v.name for v in dense.get_outputs()], dense.run(None, feed)))
            expected.append(result)
            dense_state[request] = {
                name.replace("present.", "past_key_values.", 1): value
                for name, value in result.items()
                if name != "logits"
            }
        counts = [count for _, count in requests]
        cu = np.asarray([0, *np.cumsum(counts)], np.int32)
        positions = np.concatenate(
            [np.arange(past, past + count) for past, count in zip(past_lengths, counts)]
        )
        block_table = np.zeros((len(requests), 2), np.int32)
        for row, (request, _) in enumerate(requests):
            block_table[row, : len(pages[request])] = pages[request]
        shared = {
            "input_ids": torch.tensor(np.concatenate(ids), device="cuda"),
            "position_ids": torch.tensor(np.tile(positions, (3, 1)), device="cuda"),
            "block_table": torch.tensor(block_table, device="cuda"),
            "cumulative_sequence_lengths": torch.tensor(cu, device="cuda"),
            "past_sequence_lengths": torch.tensor(
                past_lengths, device="cuda", dtype=torch.int32
            ),
            "attention_metadata": torch.tensor(
                [max(counts), max(p + c for p, c in zip(past_lengths, counts)), 0],
                dtype=torch.int32,
            ),
        }
        results = []
        for index, session in enumerate((packed, pruned)):
            feed = {
                **shared,
                **caches[index],
                **{
                    name: torch.cat([fixed[index][request][name] for request, _ in requests])
                    for name in fixed[index][requests[0][0]]
                },
            }
            result = _run_packed(
                session, feed, pruned=bool(index), vocab_size=config.vocab_size
            )
            results.append(result)
            for row, (request, count) in enumerate(requests):
                for name, value in result.items():
                    if name.endswith(("conv_state", "recurrent_state")):
                        fixed[index][request][
                            name.replace("present.", "past_key_values.", 1)
                        ] = value[row : row + 1].clone()
                        reference = expected[row][name]
                        if name.endswith("recurrent_state"):
                            reference = reference.transpose(0, 1, 3, 2)
                        np.testing.assert_allclose(
                            value[row : row + 1].float().cpu().numpy(),
                            reference,
                            atol=tolerance,
                            rtol=tolerance,
                            err_msg=f"{step=} {request=} {name=}",
                        )
                length = past_lengths[row] + count
                for kind in ("key", "value"):
                    logical = result[f"present.3.{kind}"][pages[request]].reshape(
                        -1, config.num_key_value_heads, config.head_dim
                    )[:length]
                    reference = expected[row][f"present.3.{kind}"][0].transpose(1, 0, 2)
                    np.testing.assert_allclose(
                        logical.float().cpu().numpy(),
                        reference,
                        atol=tolerance,
                        rtol=tolerance,
                    )
        full_logits = results[0]["logits"].cpu().numpy()
        np.testing.assert_allclose(
            full_logits,
            np.concatenate([result["logits"][0] for result in expected]),
            atol=tolerance,
            rtol=tolerance,
        )
        np.testing.assert_allclose(
            results[1]["logits"].cpu().numpy(),
            full_logits[cu[1:] - 1],
            atol=2e-3 if dtype == ir.DataType.FLOAT16 else 1e-2,
            rtol=2e-3 if dtype == ir.DataType.FLOAT16 else 1e-2,
        )
        for request, count in requests:
            lengths[request] += count
        if step == 1:
            for state in fixed:
                del state["B"]
            del dense_state["B"]


def test_pinned_engine_overlap_and_late_admission(tmp_path):
    if os.environ.get("MOBIUS_GENAI_REVISION") != GENAI_REVISION:
        pytest.skip(f"Set MOBIUS_GENAI_REVISION={GENAI_REVISION} for the pinned Engine build")
    import onnxruntime_genai as og

    package = _package(ir.DataType.FLOAT16)
    ir.save(package["model"], tmp_path / "model.onnx")
    _write_genai_config(
        package.config,
        str(tmp_path),
        pkg=package,
        ort_model_type="decoder",
        ep="cuda",
        context_length=2048,
        bos_token_id=1,
        eos_token_id=127,
        pad_token_id=0,
        is_vlm=False,
        has_speech=False,
    )
    model = og.Model(str(tmp_path))
    prompts = [
        np.arange(2, 19, dtype=np.int32),
        np.arange(21, 25, dtype=np.int32),
        np.arange(30, 38, dtype=np.int32),
    ]

    def run(indices, *, overlap=False):
        engine = og.Engine(model)
        requests, generated = {}, {}

        def admit(index):
            options = og.RequestOptions()
            options.set_max_session_tokens(64)
            request = engine.create_request(options=options)
            turn = og.TurnOptions(request)
            turn.set_do_sample(False)
            turn.set_max_generated_tokens(6)
            request.begin_turn(prompts[index], turn)
            requests[request] = index
            generated[index] = []

        for index in indices[:2] if overlap else indices:
            admit(index)
        late = overlap
        finished = set()
        event_buffer = engine.create_event_buffer(1)
        for _ in range(1000):
            events = engine.run(event_buffer)
            for event in events:
                assert not event.flags & og.EngineEventFlags.FAILED
                if event.request is None:
                    continue
                index = requests[event.request]
                if event.flags & og.EngineEventFlags.TOKEN:
                    generated[index].append(event.token)
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    finished.add(index)
                    event.request.close()
            if late and any(generated.values()):
                admit(indices[2])
                late = False
            if not late and len(finished) == len(indices):
                break
        assert not late and len(finished) == len(indices)
        assert all(generated.values())
        return generated

    isolated = {index: run([index])[index] for index in range(3)}
    assert run([0, 1, 2], overlap=True) == isolated
