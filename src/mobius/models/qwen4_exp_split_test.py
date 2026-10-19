# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""N1 teacher-forced split ABI tests, not closed-loop generation qualification."""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import safetensors.torch
import torch

from mobius._builder import build_from_module
from mobius._model_package import ModelPackage
from mobius._optimizations import validate_standard_onnx
from mobius.integrations._weight_loading import apply_weights
from mobius.integrations.transformers._qwen4_exp_weights import (
    stream_qwen4_exp_safetensors_to_package,
)
from mobius.integrations.transformers._qwen4_exp_weights_test import _official_package_state
from mobius.models.qwen4_exp import (
    Qwen4ExpCausalLMModel,
    Qwen4ExpForConditionalGeneration,
)
from mobius.models.qwen4_exp_test import (
    _assert_reduced_huggingface_states,
    _config,
    _reduced_huggingface_numpy,
    _reduced_huggingface_text_config,
    _vl_config,
)
from mobius.tasks._qwen4_exp import Qwen4ExpCausalLMTask, Qwen4ExpVisionLanguageTask


def _build(config, *, vlm=False, selected=None, split=True):
    module = Qwen4ExpForConditionalGeneration(config) if vlm else Qwen4ExpCausalLMModel(config)
    task = (Qwen4ExpVisionLanguageTask if vlm else Qwen4ExpCausalLMTask)(
        split_ngram=split, requested_components=selected
    )
    package = build_from_module(module, config, task, execution_provider="onnx-standard")
    return package, task


def _compare(actual, expected, tolerance):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        reference = expected[name]
        assert value.shape == reference.shape, name
        assert value.dtype == reference.dtype, name
        if np.issubdtype(value.dtype, np.integer):
            np.testing.assert_array_equal(value, reference, strict=True, err_msg=name)
        else:
            assert np.isfinite(value).all(), name
            assert np.isfinite(reference).all(), name
            np.testing.assert_allclose(
                value, reference, rtol=tolerance, atol=tolerance, strict=True, err_msg=name
            )


def _initial(model, *, batch=2):
    """Initialize the declared ABI instead of assuming a particular layer schedule."""
    result = {}
    for value in model.graph.inputs:
        name = value.name
        if not name.startswith(("past_key_values.", "past_position_ids")):
            continue
        shape = [
            dim
            if isinstance(dim, int)
            else batch
            if str(dim).rsplit(".", 1)[-1] == "batch"
            else 0
            for dim in value.shape
        ]
        dtype = {
            ir.DataType.INT64: np.int64,
            ir.DataType.FLOAT: np.float32,
            ir.DataType.FLOAT16: np.float16,
        }[value.dtype]
        result[name] = np.zeros(shape, dtype=dtype)
    return result


def _next(outputs):
    return {"past_position_ids": outputs["present_position_ids"]} | {
        name.replace("present.", "past_key_values.", 1): value
        for name, value in outputs.items()
        if name.startswith("present.")
    }


class _Session:
    def __init__(self, model, directory, device):
        directory.mkdir()
        self.device = device
        validate_standard_onnx(model)
        assert model.opset_imports[""] == 24
        ir.save(model, directory / "model.onnx", external_data="weights.data")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_profiling = device == "cuda"
        options.profile_file_prefix = str(directory / "placement")
        self.session = ort.InferenceSession(
            str(directory / "model.onnx"),
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"],
        )

    def run(self, feeds):
        return dict(
            zip(
                [output.name for output in self.session.get_outputs()],
                self.session.run(None, feeds),
            )
        )

    def close(self):
        if self.device == "cuda":
            events = json.loads(Path(self.session.end_profiling()).read_text())
            # Provider registration alone is not evidence of CUDA execution.
            assert any(
                event.get("args", {}).get("provider") == "CUDAExecutionProvider"
                and event["args"].get("op_name")
                in {"MatMul", "FusedMatMul", "Gemm", "Conv", "Gather", "GatherND"}
                for event in events
            ), "No substantive CUDA kernel was profiled"


def _device_guard(device):
    if device == "cuda" and (
        not torch.cuda.is_available()
        or "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        pytest.skip("Both Torch CUDA and ORT CUDA are required")


@pytest.mark.parametrize("vlm", [False, True], ids=["text", "vlm"])
@pytest.mark.parametrize(
    "device,dtype", [("cpu", ir.DataType.FLOAT), ("cuda", ir.DataType.FLOAT16)]
)
@pytest.mark.parametrize("chunks", [(8, 3, 2), (3, 2, 3, 3, 2)], ids=["prefill", "chunked"])
def test_split_all_logits_and_states(tmp_path, vlm, device, dtype, chunks):
    """Common-source parity, including unsorted multiple PLE layers and integer masks."""
    _device_guard(device)
    torch.manual_seed(776)
    config = (_vl_config if vlm else _config)(
        dtype=dtype,
        num_hidden_layers=3,
        layer_types=["linear_attention", "linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[2, 1],
        eos_token_id=[1, 7],
    )
    # Construction normalizes PLE IDs; exercise the graph seam's effective order.
    assert config.ple_layer_ids == [1, 2]
    config.ple_layer_ids = [2, 1]
    legacy, _ = _build(config, vlm=vlm, split=False)
    split, task = _build(config, vlm=vlm)
    for key in ("ngram", "main"):
        assert json.loads(split[key].metadata_props["mobius.qwen4_exp.ple_layer_ids"]) == [
            2,
            1,
        ]
    source = _official_package_state(legacy, config)
    # Small random weights avoid artificial overflow, but preserve distinct values.
    for name, value in source.items():
        if value.is_floating_point():
            source[name] = (value * 0.05).to(
                torch.float16 if device == "cuda" else torch.float32
            )
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    safetensors.torch.save_file(source, checkpoint / "model.safetensors")
    stream_qwen4_exp_safetensors_to_package(legacy, str(checkpoint), config)
    stream_qwen4_exp_safetensors_to_package(
        split, str(checkpoint), config, manifest=task.component_manifest(hf_config=config)
    )
    # Persistence is part of the executable acceptance, not just an identity test.
    split.save(str(tmp_path / "saved"), progress_bar=False)
    split = ModelPackage.load(str(tmp_path / "saved"))
    decoder_key = "decoder" if vlm else "model"
    with ExitStack() as stack:
        sessions = {}
        for name, model in {f"old_{key}": value for key, value in legacy.items()}.items():
            session = _Session(model, tmp_path / name, device)
            stack.callback(session.close)
            sessions[name] = session
        for name, model in split.items():
            session = _Session(model, tmp_path / name, device)
            stack.callback(session.close)
            sessions[name] = session
        old_states = _initial(legacy[decoder_key])
        main_states = _initial(split["main"])
        context = np.full((2, 2, config.ngram_size - 1), 1, dtype=np.int64)
        ids = np.array(
            [[0, 2, 2, 1, 3, 3, 7, 4, 4, 1, 5, 5, 6], [0, 0, 8, 8, 1, 9, 9, 2, 1, 3, 3, 4, 4]],
            dtype=np.int64,
        )
        mask = np.array(
            [
                [0, 1, 2, 1, 0, 1, -1, 1, 1, 0, 1, 1, 1],
                [0, 0, 1, 1, 1, 0, 2, 1, 0, 1, 1, 1, 1],
            ],
            dtype=np.int64,
        )
        start = 0
        tolerance = 1e-3 if device == "cpu" else 1e-2
        for size in chunks:
            end = start + size
            current = ids[:, start:end].copy()
            positions = np.broadcast_to(
                np.arange(start, end, dtype=np.int64), (2, size)
            ).copy()
            if vlm:
                positions = np.stack([positions, positions + 2, positions // 2, positions % 3])
            normal = {"attention_mask": mask[:, :end], "position_ids": positions}
            if vlm:
                # First chunk includes a new image; later calls have no new image.
                features = np.zeros(
                    (0, config.hidden_size),
                    dtype=np.float16 if device == "cuda" else np.float32,
                )
                if start == 0:
                    current[:, -1] = config.image_token_id
                    pixels = (
                        np.random.default_rng(19).normal(0, 0.2, (8, 1536)).astype(np.float32)
                    )
                    grid = np.array([[1, 2, 4]], dtype=np.int64)
                    features = sessions["vision_encoder"].run(
                        {"pixel_values": pixels, "image_grid_thw": grid}
                    )["image_features"]
                    old_features = sessions["old_vision_encoder"].run(
                        {"pixel_values": pixels, "image_grid_thw": grid}
                    )["image_features"]
                    _compare({"features": features}, {"features": old_features}, tolerance)
                embedding_feed = {"input_ids": current, "image_features": features}
                embeds = sessions["embedding"].run(embedding_feed)["inputs_embeds"]
                old_embeds = sessions["old_embedding"].run(embedding_feed)["inputs_embeds"]
                _compare({"embedding": embeds}, {"embedding": old_embeds}, tolerance)
                normal |= {"inputs_embeds": embeds, "ple_input_ids": current}
            else:
                normal["input_ids"] = current
            ngram = sessions["ngram"].run(
                {
                    "input_ids": np.where(mask[:, start:end] == 0, 1, current),
                    "past_ngram_context": context,
                }
            )
            actual = sessions["main"].run(
                main_states | normal | {"ngram_embeddings": ngram["ngram_embeddings"]}
            )
            expected = sessions[f"old_{decoder_key}"].run(old_states | normal)
            reconstructed = dict(actual)
            for index, layer_id in enumerate(config.ple_layer_ids):
                reconstructed[f"present.{layer_id - 1}.ple_context"] = ngram[
                    "present_ngram_context"
                ][index]
            _compare(reconstructed, expected, tolerance)
            context = ngram["present_ngram_context"]
            main_states, old_states = _next(actual), _next(expected)
            start = end


_SELECTIONS = [
    (vlm, frozenset(selection))
    for vlm in (False, True)
    for size in range(1, 5 if vlm else 3)
    for selection in itertools.combinations(
        ("ngram", "main", "vision_encoder", "embedding") if vlm else ("ngram", "main"), size
    )
]


@pytest.mark.parametrize("vlm,selected", _SELECTIONS)
def test_selected_construction_and_optimization_only(vlm, selected):
    config = (_vl_config if vlm else _config)()
    task_type = Qwen4ExpVisionLanguageTask if vlm else Qwen4ExpCausalLMTask
    task = task_type(split_ngram=True, requested_components=set(selected))
    builders = {
        "ngram": "_build_ngram",
        "main": "_build_decoder" if vlm else "_build_main",
    } | ({"embedding": "_build_embedding", "vision_encoder": "_build_vision"} if vlm else {})
    with ExitStack() as stack:
        spies = {
            key: stack.enter_context(
                mock.patch.object(task, method, wraps=getattr(task, method))
            )
            for key, method in builders.items()
        }
        import mobius._builder as builder

        optimize = stack.enter_context(
            mock.patch.object(builder, "optimize_model", wraps=builder.optimize_model)
        )
        module = (
            Qwen4ExpForConditionalGeneration(config) if vlm else Qwen4ExpCausalLMModel(config)
        )
        package = build_from_module(module, config, task, execution_provider="onnx-standard")
    assert set(package) == selected
    assert all(spy.call_count == int(key in selected) for key, spy in spies.items())
    assert optimize.call_count == len(selected)
    manifest = task.component_manifest(hf_config=config)
    assert set(manifest) == set(builders)
    assert [call.kwargs["model_role"] for call in optimize.call_args_list] == [
        manifest[key].role for key in package
    ]
    assert manifest["main"].module_attribute_path == ("decoder" if vlm else "")
    lookup = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    assert manifest["ngram"].owns_source(lookup)
    assert not manifest["main"].owns_source(lookup)
    assert manifest["main"].owns_source("model.language_model.layers.0.ple.key_proj.weight")


@pytest.mark.parametrize("task_type", [Qwen4ExpCausalLMTask, Qwen4ExpVisionLanguageTask])
def test_options_are_instance_local_and_reject_invalid_requests(task_type):
    selection = {"ngram"}
    task = task_type(split_ngram=True, requested_components=selection)
    selection.add("bad")
    assert task.requested_components == frozenset({"ngram"})
    assert not task_type().split_ngram
    for requested in (set(), {"bad"}):
        with pytest.raises(ValueError, match="nonempty"):
            task_type(split_ngram=True, requested_components=requested)
    with pytest.raises(ValueError, match="requires split_ngram"):
        task_type(requested_components={"main"})


@pytest.mark.parametrize("ep", ["default", "cpu", "cuda"])
def test_wrong_ep_rejected_before_build(ep):
    config = _config()
    task = Qwen4ExpCausalLMTask(split_ngram=True)
    with (
        mock.patch.object(task, "build") as build,
        pytest.raises(ValueError, match="onnx-standard"),
    ):
        build_from_module(Qwen4ExpCausalLMModel(config), config, task, execution_provider=ep)
    build.assert_not_called()


def test_no_ple_split_rejected_but_unsplit_supported():
    config = _config(ple_layer_ids=[])
    with pytest.raises(ValueError, match="at least one"):
        _build(config)
    legacy, _ = _build(config, split=False)
    assert set(legacy) == {"model"}


@pytest.mark.parametrize("vlm", [False, True])
@pytest.mark.parametrize("ep", ["default", "onnx-standard"])
def test_explicit_unsplit_preserves_default_graph_contract(vlm, ep):
    config = (_vl_config if vlm else _config)()
    module_type = Qwen4ExpForConditionalGeneration if vlm else Qwen4ExpCausalLMModel
    default = build_from_module(
        module_type(config),
        config,
        "qwen4-exp-vision-language" if vlm else "qwen4-exp-text-generation",
        execution_provider=ep,
    )
    explicit = build_from_module(
        module_type(config),
        config,
        (Qwen4ExpVisionLanguageTask if vlm else Qwen4ExpCausalLMTask)(split_ngram=False),
        execution_provider=ep,
    )
    assert default.keys() == explicit.keys()
    for key, model in default.items():
        other = explicit[key]
        assert model.metadata_props == other.metadata_props
        assert list(model.graph.initializers) == list(other.graph.initializers)
        assert [(value.name, value.dtype, value.shape) for value in model.graph.inputs] == [
            (value.name, value.dtype, value.shape) for value in other.graph.inputs
        ]
        assert [value.name for value in model.graph.outputs] == [
            value.name for value in other.graph.outputs
        ]
        assert [(node.domain, node.op_type) for node in model.graph] == [
            (node.domain, node.op_type) for node in other.graph
        ]


@pytest.mark.parametrize("kind", ["fp8", "affine", "component"])
def test_split_rejects_quantized_config_before_construction(kind):
    from mobius._configs import QuantizationConfig
    from mobius.integrations._block_quant import BlockQuantScheme

    config = _config()
    module = Qwen4ExpCausalLMModel(config)
    if kind == "fp8":
        config.block_quant_scheme = BlockQuantScheme(
            quant_method="fp8",
            weight_block_size=(128, 128),
        )
    elif kind == "affine":
        config.quantization = QuantizationConfig(quant_method="gptq", bits=4, group_size=32)
    else:
        config.component_quantization = {}
    task = Qwen4ExpCausalLMTask(split_ngram=True)
    with (
        mock.patch.object(task, "build") as build,
        pytest.raises(ValueError, match="unquantized"),
    ):
        build_from_module(module, config, task, execution_provider="onnx-standard")
    build.assert_not_called()


def test_split_preserves_bfloat16_vision_guard_and_allows_other_selected_graphs():
    config = _vl_config(dtype=ir.DataType.BFLOAT16)
    with pytest.raises(ValueError, match="BF16 vision export"):
        _build(config, vlm=True, selected={"vision_encoder"})
    package, _ = _build(config, vlm=True, selected={"main", "ngram", "embedding"})
    assert set(package) == {"main", "ngram", "embedding"}


@pytest.mark.parametrize("vlm", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zero_length_split_calls_reject(tmp_path, vlm, device):
    _device_guard(device)
    config = (_vl_config if vlm else _config)(
        dtype=ir.DataType.FLOAT16 if device == "cuda" else ir.DataType.FLOAT
    )
    package, _ = _build(config, vlm=vlm, selected={"ngram", "main"})
    # No weight-dependent operation should turn an invalid chunk into success.
    for model in package.values():
        for value in model.graph.initializers.values():
            if value.const_value is None:
                value.const_value = ir.tensor(
                    np.zeros(
                        tuple(value.shape),
                        dtype=np.float16 if device == "cuda" else np.float32,
                    )
                )
    for key in ("ngram", "main"):
        session = _Session(package[key], tmp_path / key, device)
        if key == "ngram":
            feeds = {
                "input_ids": np.zeros((2, 0), dtype=np.int64),
                "past_ngram_context": np.ones((1, 2, 2), dtype=np.int64),
            }
        else:
            feeds = _initial(package[key]) | {
                "attention_mask": np.zeros((2, 0), dtype=np.int64),
                "position_ids": np.zeros((4, 2, 0) if vlm else (2, 0), dtype=np.int64),
                "ngram_embeddings": np.zeros(
                    (1, 2, 0, 8), dtype=np.float16 if device == "cuda" else np.float32
                ),
            }
            if vlm:
                feeds |= {
                    "ple_input_ids": np.zeros((2, 0), dtype=np.int64),
                    "inputs_embeds": np.zeros(
                        (2, 0, 16), dtype=np.float16 if device == "cuda" else np.float32
                    ),
                }
            else:
                feeds["input_ids"] = np.zeros((2, 0), dtype=np.int64)
        with pytest.raises(
            (
                ort.capi.onnxruntime_pybind11_state.RuntimeException,
                ort.capi.onnxruntime_pybind11_state.Fail,
            ),
            match="Reshape",
        ):
            session.run(feeds)
        # Invalid-call profiles need not contain substantive kernels.
        if device == "cuda":
            session.session.end_profiling()


@pytest.mark.integration
@pytest.mark.parametrize(
    "device,dtype", [("cpu", ir.DataType.FLOAT), ("cuda", ir.DataType.FLOAT16)]
)
@pytest.mark.parametrize("vlm", [False, True], ids=["text", "vlm"])
def test_pinned_huggingface_teacher_forced_reference(tmp_path, device, dtype, vlm):
    """Independent pinned HF oracle; no generated token feedback (N2)."""
    _device_guard(device)
    from transformers import DynamicCache, Qwen4ExpForCausalLM

    source_path = Path(inspect.getfile(Qwen4ExpForCausalLM))
    assert (
        hashlib.sha256(source_path.read_bytes()).hexdigest()
        == "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459"
    )
    torch.manual_seed(42)
    torch_dtype = torch.float16 if dtype == ir.DataType.FLOAT16 else torch.float32
    hf_config = _reduced_huggingface_text_config(
        num_experts_per_tok=2,
        dtype=torch_dtype,
        **(
            {
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                    "partial_rotary_factor": 1.0,
                    "mrope_section": [1, 1, 2],
                    "mrope_interleaved": True,
                }
            }
            if vlm
            else {}
        ),
    )
    from mobius._configs import Qwen4ExpConfig

    if vlm:
        from transformers import Qwen4ExpConfig as HfConfig
        from transformers import Qwen4ExpForConditionalGeneration as HfModel
        from transformers import Qwen4ExpVisionConfig

        full_config = HfConfig(
            text_config=hf_config,
            vision_config=Qwen4ExpVisionConfig(
                depth=1,
                hidden_size=32,
                intermediate_size=64,
                num_heads=2,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                out_hidden_size=16,
                num_position_embeddings=16,
                hidden_act="gelu_pytorch_tanh",
                deepstack_visual_indexes=[],
            ),
            image_token_id=30,
            video_token_id=31,
            vision_start_token_id=29,
            vision_end_token_id=28,
            tie_word_embeddings=False,
        )
        hf = HfModel(full_config).to(torch_dtype).eval()
        config = _vl_config(
            num_experts_per_tok=2,
            partial_rotary_factor=1.0,
            mrope_section=[1, 1, 2],
            dtype=dtype,
        )
        module = Qwen4ExpForConditionalGeneration(config)
        task = Qwen4ExpVisionLanguageTask(split_ngram=True)
    else:
        hf = Qwen4ExpForCausalLM(hf_config).to(torch_dtype).eval()
        config = Qwen4ExpConfig.from_transformers(hf_config)
        module = Qwen4ExpCausalLMModel(config)
        task = Qwen4ExpCausalLMTask(split_ngram=True)
    package = build_from_module(module, config, task, execution_provider="onnx-standard")
    weights = module.preprocess_weights(dict(hf.state_dict()))
    for model in package.values():
        apply_weights(
            model,
            {key: value for key, value in weights.items() if key in model.graph.initializers},
        )
        validate_standard_onnx(model)
    main, ngram = (_Session(package[key], tmp_path / key, device) for key in ("main", "ngram"))
    cache = DynamicCache(config=hf_config)
    states = _initial(package["main"], batch=1)
    context = np.ones((1, 1, 2), dtype=np.int64)
    past = 0
    tolerance = 1e-3 if dtype == ir.DataType.FLOAT else 1e-2
    with ExitStack() as stack:
        stack.callback(main.close)
        stack.callback(ngram.close)
        if vlm:
            vision = _Session(package["vision_encoder"], tmp_path / "vision", device)
            embedding = _Session(package["embedding"], tmp_path / "embedding", device)
            stack.callback(vision.close)
            stack.callback(embedding.close)
        chunks = (
            ([2, 29, 30, 30, 28, 3, 1], [4, 4], [5, 6, 7])
            if vlm
            else ([2, 2, 1], [3, 4], [4, 5, 6])
        )
        for tokens in chunks:
            ids = np.array([tokens], dtype=np.int64)
            positions = np.arange(past, past + len(tokens), dtype=np.int64)[None]
            if vlm:
                positions = np.stack([positions] * 4)
            hf_inputs = {"input_ids": torch.from_numpy(ids)}
            if vlm:
                hf_inputs["position_ids"] = torch.from_numpy(positions)
            features = np.zeros(
                (0, config.hidden_size), dtype=np.float16 if device == "cuda" else np.float32
            )
            if vlm and past == 0:
                grid = np.array([[1, 2, 4]], dtype=np.int64)
                pixels = np.random.default_rng(5).normal(0, 0.2, (8, 1536)).astype(np.float32)
                hf_inputs |= {
                    "pixel_values": torch.from_numpy(pixels),
                    "image_grid_thw": torch.from_numpy(grid),
                }
                features = vision.run({"pixel_values": pixels, "image_grid_thw": grid})[
                    "image_features"
                ]
                with torch.no_grad():
                    expected_features = _reduced_huggingface_numpy(
                        hf.model.visual(
                            torch.from_numpy(pixels).to(torch_dtype),
                            grid_thw=torch.from_numpy(grid),
                        ).pooler_output
                    )
                _compare({"features": features}, {"features": expected_features}, tolerance)
            with torch.no_grad():
                reference = _reduced_huggingface_numpy(
                    hf(**hf_inputs, past_key_values=cache, use_cache=True).logits
                )
            lookup = ngram.run({"input_ids": ids, "past_ngram_context": context})
            lexical = {"input_ids": ids}
            if vlm:
                embeds = embedding.run({"input_ids": ids, "image_features": features})[
                    "inputs_embeds"
                ]
                lexical = {"ple_input_ids": ids, "inputs_embeds": embeds}
            result = main.run(
                states
                | {
                    **lexical,
                    "attention_mask": np.ones((1, past + len(tokens)), dtype=np.int64),
                    "position_ids": positions,
                    "ngram_embeddings": lookup["ngram_embeddings"],
                }
            )
            _compare({"logits": result["logits"]}, {"logits": reference}, tolerance)
            complete = result | {"present.0.ple_context": lookup["present_ngram_context"][0]}
            _assert_reduced_huggingface_states(
                complete, cache, multimodal=vlm, tolerance=tolerance
            )
            states, context = _next(result), lookup["present_ngram_context"]
            past += len(tokens)
