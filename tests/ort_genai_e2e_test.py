# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fast, network-free end-to-end tests for released ONNX Runtime GenAI wheels."""

from __future__ import annotations

import dataclasses
import gc
import importlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import onnx_ir as ir
import pytest
from onnxscript import GraphBuilder

from mobius._model_package import ModelPackage
from mobius.integrations.ort_genai import write_ort_genai_config

_EXPECTED_ORT_GENAI_VERSION = "0.15.2"


@dataclasses.dataclass
class _SyntheticConfig:
    model_type: str = "synthetic_ci_decoder"
    vocab_size: int = 8
    hidden_size: int = 8
    num_hidden_layers: int = 1
    num_attention_heads: int = 2
    num_key_value_heads: int = 2
    head_dim: int = 4
    max_position_embeddings: int = 32
    bos_token_id: int = 1
    eos_token_id: int = 7
    pad_token_id: int = 6


def _typed(value: ir.Value, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    value.type = ir.TensorType(dtype)
    value.shape = ir.Shape(shape)
    return value


def _synthetic_decoder_package() -> ModelPackage:
    """Build a decoder whose next token proves the prior KV cache was threaded."""
    graph = ir.Graph([], [], nodes=[], name="decoder", opset_imports={"": 23})
    builder = GraphBuilder(graph)
    input_ids = builder.input("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    builder.input("attention_mask", ir.DataType.INT64, ["batch", "total_sequence"])
    builder.input("position_ids", ir.DataType.INT64, ["batch", "sequence"])
    past_key = builder.input(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        ["batch", 2, "past_sequence", 4],
    )
    past_value = builder.input(
        "past_key_values.0.value",
        ir.DataType.FLOAT,
        ["batch", 2, "past_sequence", 4],
    )

    token_shape = builder.op.Shape(input_ids)
    past_sequence = builder.op.Shape(past_key, start=2, end=3)
    token_id = builder.op.Cast(past_sequence, to=ir.DataType.INT64)
    one_hot = builder.op.OneHot(
        token_id,
        builder.op.Constant(value_int=8),
        builder.op.Constant(value_floats=[0.0, 1.0]),
        axis=-1,
    )
    logits = builder.op.Expand(
        builder.op.Unsqueeze(one_hot, builder.op.Constant(value_ints=[0])),
        builder.op.Concat(token_shape, builder.op.Constant(value_ints=[8]), axis=0),
    )
    batch = builder.op.Shape(input_ids, start=0, end=1)
    sequence = builder.op.Shape(input_ids, start=1, end=2)
    cache_update_shape = builder.op.Concat(
        batch,
        builder.op.Constant(value_ints=[2]),
        sequence,
        builder.op.Constant(value_ints=[4]),
        axis=0,
    )
    cache_update = builder.op.ConstantOfShape(
        cache_update_shape,
        value=ir.tensor([0.0]),
    )
    present_key = builder.op.Concat(past_key, cache_update, axis=2)
    present_value = builder.op.Concat(past_value, cache_update, axis=2)

    builder.add_output(
        _typed(logits, ir.DataType.FLOAT, ["batch", "sequence", 8]),
        "logits",
    )
    builder.add_output(
        _typed(present_key, ir.DataType.FLOAT, ["batch", 2, "present_sequence", 4]),
        "present.0.key",
    )
    builder.add_output(
        _typed(present_value, ir.DataType.FLOAT, ["batch", 2, "present_sequence", 4]),
        "present.0.value",
    )
    return ModelPackage(
        {"model": ir.Model(graph, ir_version=10)},
        config=_SyntheticConfig(),
    )


def _write_tokenizer(output_dir: Path) -> None:
    vocabulary = {
        "[UNK]": 0,
        "[BOS]": 1,
        "hello": 2,
        "world": 3,
        "!": 4,
        "unused": 5,
        "[PAD]": 6,
        "[EOS]": 7,
    }
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocabulary,
            "merges": [],
        },
    }
    (output_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizer",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def ort_genai_module() -> Any:
    """Import the selected lane's required runtime only when these tests execute."""
    return importlib.import_module("onnxruntime_genai")


def _generate(
    ort_genai: Any,
    package_dir: Path,
    prompt: str,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    model = ort_genai.Model(str(package_dir))
    tokenizer = ort_genai.Tokenizer(model)
    prompt_ids = tokenizer.encode(prompt)
    params = ort_genai.GeneratorParams(model)
    params.set_search_options(
        max_length=len(prompt_ids) + max_new_tokens,
        do_sample=False,
    )
    generator = ort_genai.Generator(model, params)
    generator.append_tokens(prompt_ids)

    generated: list[int] = []
    for _ in range(max_new_tokens):
        assert not generator.is_done()
        generator.generate_next_token()
        generated.append(generator.get_next_tokens()[0])
    decoded = tokenizer.decode(prompt_ids)
    del generator, tokenizer, model
    gc.collect()
    return generated, decoded


@pytest.mark.integration
@pytest.mark.ort_genai_fast
def test_generic_decoder_end_to_end_and_reload(
    tmp_path: Path,
    ort_genai_module: Any,
) -> None:
    """Exercise tokenizer, prefill, threaded KV decode, and package reload."""
    expected_version = os.environ.get("MOBIUS_EXPECTED_ORT_GENAI_VERSION")
    installed_version = version("onnxruntime-genai")
    if expected_version:
        assert installed_version == expected_version
    assert installed_version == _EXPECTED_ORT_GENAI_VERSION

    package_dir = tmp_path / "synthetic-decoder"
    package = _synthetic_decoder_package()
    package.save(package_dir)
    _write_tokenizer(package_dir)
    write_ort_genai_config(
        package,
        str(package_dir),
        runtime_version=installed_version,
    )

    config = json.loads((package_dir / "genai_config.json").read_text(encoding="utf-8"))
    assert config["model"]["type"] == "decoder"
    assert "state_groups" not in config["model"]["decoder"]

    # Prefill sees an empty cache; each decode step must receive the prior
    # present cache for the selected token to advance.
    expected_tokens = [0, 1, 2, 3]
    first_tokens, decoded_prompt = _generate(
        ort_genai_module,
        package_dir,
        "hello",
        len(expected_tokens),
    )
    assert decoded_prompt == "hello"
    assert len(first_tokens) == len(expected_tokens)
    assert first_tokens == expected_tokens

    reloaded_tokens, _ = _generate(
        ort_genai_module,
        package_dir,
        "hello",
        len(expected_tokens),
    )
    assert len(reloaded_tokens) == len(expected_tokens)
    assert reloaded_tokens == expected_tokens


@pytest.mark.integration
@pytest.mark.ort_genai_fast
def test_malformed_genai_config_is_rejected(
    tmp_path: Path,
    ort_genai_module: Any,
) -> None:
    package_dir = tmp_path / "malformed"
    package = _synthetic_decoder_package()
    package.save(package_dir)
    _write_tokenizer(package_dir)
    write_ort_genai_config(
        package,
        str(package_dir),
        runtime_version=version("onnxruntime-genai"),
    )
    config_path = package_dir / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["model"]["decoder"]["filename"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"filename|model path is a directory"):
        ort_genai_module.Model(str(package_dir))
