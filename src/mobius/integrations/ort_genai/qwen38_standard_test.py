# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline standard Qwen3.8 VLM packaging and local-asset regressions."""

from __future__ import annotations

import importlib.util
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx_ir as ir
import pytest
import transformers

from mobius.__main__ import build_parser, main
from mobius.integrations.ort_genai import write_ort_genai_config
from mobius.integrations.transformers._builder import build_transformers_model

_FIXTURE = Path(__file__).resolve().parents[4] / "testdata/configs/qwen3_8-27b.json"


def _tiny_config(directory: Path) -> None:
    """Keep two 3-DeltaNet/1-KV cycles, MRoPE, and the FP32 state policy."""
    config = transformers.Qwen3_5Config.from_dict(json.loads(_FIXTURE.read_text()))
    text = config.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_hidden_layers = 8
    text.layer_types = text.layer_types[:8]
    text.num_attention_heads = 6
    text.num_key_value_heads = 1
    text.head_dim = 32
    text.vocab_size = 256
    text.linear_num_key_heads = 2
    text.linear_num_value_heads = 6
    text.linear_key_head_dim = 8
    text.linear_value_head_dim = 8
    text.max_position_embeddings = 128
    text.rope_parameters["mrope_section"] = [2, 1, 1]
    text.bos_token_id = 1
    text.eos_token_id = 2
    config.image_token_id = 240
    config.video_token_id = 241
    config.vision_start_token_id = 242
    config.vision_end_token_id = 243
    vision = config.vision_config
    vision.depth = 1
    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_heads = 2
    vision.out_hidden_size = 64
    vision.num_position_embeddings = 16
    config.save_pretrained(directory)


def _assert_abi(model: ir.Model) -> None:
    assert not model.functions
    assert all(node.domain in {"", "ai.onnx"} for node in model.graph.all_nodes())
    ports = {value.name: value for value in model.graph.inputs}
    outputs = {value.name: value for value in model.graph.outputs}
    assert ports["inputs_embeds"].dtype == ir.DataType.FLOAT16
    assert ports["position_ids"].shape[0] == 3
    assert {name for name in ports if name.endswith(".key")} == {
        "past_key_values.3.key",
        "past_key_values.7.key",
    }
    for index in range(8):
        roles = (
            (("key", ir.DataType.FLOAT16), ("value", ir.DataType.FLOAT16))
            if index % 4 == 3
            else (("conv_state", ir.DataType.FLOAT16), ("recurrent_state", ir.DataType.FLOAT))
        )
        for role, dtype in roles:
            assert ports[f"past_key_values.{index}.{role}"].dtype == dtype
            assert outputs[f"present.{index}.{role}"].dtype == dtype


@pytest.mark.parametrize("runtime", [None, "ort-genai"])
def test_tiny_standard_vlm_cli_routes(tmp_path: Path, runtime: str | None) -> None:
    source = tmp_path / "hf"
    _tiny_config(source)
    destination = tmp_path / "package"
    argv = [
        "build",
        "--config",
        str(source),
        str(destination),
        "--ep",
        "onnx-standard",
        "--dtype",
        "f16",
        "--no-weights",
    ]
    if runtime:
        argv.extend(["--runtime", runtime])
    main(argv)
    assert {path.name for path in destination.iterdir() if path.is_dir()} == {
        "decoder",
        "embedding",
        "vision_encoder",
    }
    models = {
        name: ir.load(destination / name / "model.onnx")
        for name in ("decoder", "embedding", "vision_encoder")
    }
    _assert_abi(models["decoder"])
    for model in models.values():
        assert not model.functions
        assert all(node.domain in {"", "ai.onnx"} for node in model.graph.all_nodes())
    assert models["vision_encoder"].graph.inputs[0].dtype == ir.DataType.FLOAT
    metadata = destination / "genai_config.json"
    assert metadata.exists() == bool(runtime)
    if runtime:
        payload = json.loads(metadata.read_text())
        assert payload["model"]["type"] == "qwen3_5"
        decoder = payload["model"]["decoder"]
        assert decoder["num_hidden_layers"] == 8  # Never compact the two KV slots.
        assert decoder["inputs"]["inputs_embeds"] == "inputs_embeds"
        assert "input_ids" not in decoder["inputs"]
        assert decoder["inputs"]["past_key_names"] == "past_key_values.%d.key"
        assert decoder["outputs"]["present_key_names"] == "present.%d.key"
        assert payload["search"]["past_present_share_buffer"] is False
        for section in ("decoder", "vision", "embedding"):
            assert payload["model"][section]["session_options"]["provider_options"] == []
        processor = json.loads((destination / "processor_config.json").read_text())
        assert processor["processor"]["name"] == "qwen2_5_image_processor"
        assert processor["processor"]["transforms"][-1]["operation"]["type"] == "PatchImage"


def test_no_runtime_ep_cli_flag() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["build", "--model", "Qwen/Qwen3.8-27B", "/unused", "--runtime-ep", "cuda"]
        )


def test_local_hf_processor_source_is_used(tmp_path: Path) -> None:
    """Local assets must not silently leave CLIP normalization/resize defaults."""
    source = tmp_path / "hf"
    _tiny_config(source)
    package = build_transformers_model(
        str(source), load_weights=False, dtype="f16", execution_provider="onnx-standard"
    )
    processor = mock.Mock()
    processor.image_processor = mock.Mock(
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        rescale_factor=1 / 255,
        resample=3,
        size={"shortest_edge": 65536, "longest_edge": 16777216},
    )
    with mock.patch(
        "transformers.AutoProcessor.from_pretrained", return_value=processor
    ) as load_processor:
        result = write_ort_genai_config(
            package, str(tmp_path / "overlay"), local_config_dir=str(source)
        )
    load_processor.assert_called_once_with(str(source), trust_remote_code=False)
    pipeline = json.loads(Path(result["processor_config"]).read_text())
    operations = {
        item["operation"]["type"]: item["operation"].get("attrs", {})
        for item in pipeline["processor"]["transforms"]
    }
    assert operations["Normalize"]["mean"] == [0.5, 0.5, 0.5]
    assert operations["Normalize"]["std"] == [0.5, 0.5, 0.5]
    assert set(operations["Normalize"]) == {"mean", "std"}
    assert operations["DecodeImage"] == {"color_space": "RGB"}
    assert operations["Resize"]["min_pixels"] == 65536
    assert operations["Resize"]["max_pixels"] == 16777216
    assert operations["PatchImage"] == {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
    }


@pytest.mark.parametrize("local_source", [False, True])
def test_processor_asset_source_precedence(tmp_path: Path, local_source: bool) -> None:
    """An explicit HF source still wins over a fallback local config directory."""
    source = tmp_path / "hf"
    _tiny_config(source)
    package = build_transformers_model(
        str(source), load_weights=False, dtype="f16", execution_provider="onnx-standard"
    )
    with (
        mock.patch(
            "mobius.integrations.ort_genai.auto_export._write_vision_processor_config",
            return_value=None,
        ) as writer,
        mock.patch("mobius.integrations.ort_genai.auto_export._copy_tokenizer_files"),
        mock.patch(
            "mobius.integrations.ort_genai.auto_export._load_generation_config",
            return_value=None,
        ),
        mock.patch("mobius.integrations.ort_genai.auto_export._fix_chat_template"),
    ):
        # Both are local to guarantee offline resolution without mocking AutoConfig.
        write_ort_genai_config(
            package,
            str(tmp_path / "overlay"),
            hf_model_id=str(source) if local_source else None,
            local_config_dir=str(tmp_path / "fallback") if local_source else None,
        )
    assert writer.call_args.kwargs["hf_model_id"] == (str(source) if local_source else None)


def test_standard_hybrid_cuda_metadata_keeps_capture_and_sharing_off(tmp_path: Path) -> None:
    source = tmp_path / "hf"
    _tiny_config(source)
    package = build_transformers_model(
        str(source), load_weights=False, dtype="f16", execution_provider="onnx-standard"
    )
    result = write_ort_genai_config(package, str(tmp_path / "overlay"), ep="cuda")
    payload = json.loads(Path(result["genai_config"]).read_text())
    assert payload["search"]["past_present_share_buffer"] is False
    for section in ("decoder", "vision", "embedding"):
        options = payload["model"][section]["session_options"]["provider_options"]
        assert options[0]["cuda"]["enable_cuda_graph"] == "0"
    _assert_abi(package["decoder"])


def _qualification_example() -> dict:
    return runpy.run_path(
        str(Path(__file__).resolve().parents[4] / "examples/qwen38_standard_genai.py")
    )


@pytest.mark.parametrize(
    "relative_path",
    ["examples/qwen38_standard_genai.py", "tests/ort_genai_hybrid_native_test.py"],
)
@pytest.mark.parametrize("loader", ["module", "runpy"])
def test_qualification_import_without_resource(
    relative_path: str, loader: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate missing Windows resource support, not actual Windows execution."""
    path = Path(__file__).resolve().parents[4] / relative_path
    monkeypatch.setenv("MOBIUS_QWEN_NATIVE_PROBE", "0")
    with mock.patch.dict("sys.modules", {"resource": None}):
        if loader == "runpy":
            namespace = runpy.run_path(str(path))
        else:
            spec = importlib.util.spec_from_file_location("_qwen_import_boundary", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            namespace = vars(module)
    if relative_path.startswith("tests/"):
        assert namespace["pytestmark"].name == "skipif"
        assert namespace["pytestmark"].args == (True,)
        assert callable(namespace["test_native_sparse_mixed_state_progression_and_reset"])
    else:
        assert callable(namespace["prepare_overlay"])


@pytest.mark.parametrize(
    "relative_path",
    ["examples/qwen38_standard_genai.py", "tests/ort_genai_hybrid_native_test.py"],
)
@pytest.mark.parametrize("os_name", ["posix", "nt"])
def test_core_dump_setup_platform_guard(relative_path: str, os_name: str) -> None:
    path = Path(__file__).resolve().parents[4] / relative_path
    disable = runpy.run_path(str(path))["_disable_core_dumps"]
    resource = mock.Mock(RLIMIT_CORE=4) if os_name == "posix" else None
    with (
        mock.patch.dict(disable.__globals__, {"os": SimpleNamespace(name=os_name)}),
        mock.patch.dict("sys.modules", {"resource": resource}),
    ):
        disable()
    if resource is not None:
        resource.setrlimit.assert_called_once_with(4, (0, 0))


@pytest.mark.parametrize(
    "relative_path",
    ["examples/qwen38_standard_genai.py", "tests/ort_genai_hybrid_native_test.py"],
)
def test_core_dump_setup_does_not_hide_posix_import_failure(relative_path: str) -> None:
    path = Path(__file__).resolve().parents[4] / relative_path
    disable = runpy.run_path(str(path))["_disable_core_dumps"]
    with (
        mock.patch.dict(disable.__globals__, {"os": SimpleNamespace(name="posix")}),
        mock.patch.dict("sys.modules", {"resource": None}),
        pytest.raises(ModuleNotFoundError, match="resource"),
    ):
        disable()


def test_overlay_refuses_relative_paths_and_existing_destination(tmp_path: Path) -> None:
    prepare = _qualification_example()["prepare_overlay"]
    with pytest.raises(ValueError, match="absolute paths"):
        prepare(Path("relative"), tmp_path, tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "already-exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare(source, source, output)
    assert list(output.iterdir()) == []


def test_overlay_refuses_dangling_destination_symlink(tmp_path: Path) -> None:
    from mobius.integrations.gguf._builder_test_utils import _symlink_or_skip

    prepare = _qualification_example()["prepare_overlay"]
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "missing-target"
    output = tmp_path / "dangling-overlay"
    _symlink_or_skip(output, target)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare(source, source, output)
    assert output.is_symlink()
    assert not target.exists()
    assert list(source.iterdir()) == []


def test_real_native_load_requires_passing_synthetic_proof(tmp_path: Path) -> None:
    qualify = _qualification_example()["qualify"]
    proof = tmp_path / "failed-proof.json"
    proof.write_text(json.dumps({"progression_and_reset": []}))
    runtime = mock.Mock(__version__="test-runtime")
    with (
        mock.patch.dict("sys.modules", {"onnxruntime_genai": runtime}),
        pytest.raises(ValueError, match="passing tiny native proof"),
    ):
        qualify(tmp_path, proof, None, 4)
    runtime.Config.assert_not_called()
    runtime.Model.assert_not_called()


@pytest.mark.parametrize("early_done", [False, True])
def test_example_native_generation_contract(early_done: bool) -> None:
    """Unit-check orchestration, not a substitute for native qualification."""
    generate = _qualification_example()["_generate_native"]
    runtime = mock.Mock()
    model = object()
    # Native image preprocessing expands placeholders before this length is used.
    tensor = mock.Mock()
    tensor.as_numpy.return_value = np.zeros((1, 37), dtype=np.int64)
    inputs = {"input_ids": tensor}
    processor = mock.Mock(return_value=inputs)
    tokenizer = mock.Mock()
    generator = runtime.Generator.return_value
    if early_done:
        generator.is_done.return_value = True
    else:
        generator.is_done.side_effect = [False, False]
        generator.get_next_tokens.side_effect = [np.array([4]), np.array([8])]
    with mock.patch.dict("sys.modules", {"onnxruntime_genai": runtime}):
        if early_done:
            with pytest.raises(RuntimeError, match="produced no tokens"):
                generate(model, processor, tokenizer, "prompt", None, 2)
        else:
            assert generate(model, processor, tokenizer, "prompt", None, 2) == [4, 8]
    runtime.GeneratorParams.return_value.set_search_options.assert_called_once_with(
        max_length=39, do_sample=False, past_present_share_buffer=False
    )
    generator.set_inputs.assert_called_once_with(inputs)
    assert generator.generate_next_token.call_count == (0 if early_done else 2)


@pytest.mark.parametrize("max_new_tokens", [0, 17])
def test_example_rejects_unbounded_native_generation(max_new_tokens: int) -> None:
    qualify = _qualification_example()["qualify"]
    with pytest.raises(ValueError, match="between 1 and 16"):
        qualify(Path("/unused"), Path("/unused-proof"), None, max_new_tokens)


def test_example_second_image_requires_first_image() -> None:
    qualify = _qualification_example()["qualify"]
    with pytest.raises(ValueError, match="requires the first image"):
        qualify(Path("/unused"), Path("/unused-proof"), None, 4, Path("/second"))
