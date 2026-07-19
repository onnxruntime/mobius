# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for checkpoint_export tokenizer emission (no torch / no network)."""

from __future__ import annotations

import os

from mobius.integrations.onnx_genai.checkpoint_export import _save_tokenizer_json


class _FakeBackend:
    """Stand-in for a `tokenizers.Tokenizer` backend that records save calls."""

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"version": "1.0"}')


class _FakeFastTokenizer:
    """A fast tokenizer exposes `.backend_tokenizer` (a `tokenizers.Tokenizer`)."""

    backend_tokenizer = _FakeBackend()


def test_save_tokenizer_json_writes_package_root(tmp_path):
    output_dir = str(tmp_path)
    path = _save_tokenizer_json("some/checkpoint", _FakeFastTokenizer(), output_dir)
    assert path == os.path.join(output_dir, "tokenizer.json")
    assert os.path.isfile(path)


def test_save_tokenizer_json_honors_filename(tmp_path):
    output_dir = str(tmp_path)
    path = _save_tokenizer_json(
        "some/checkpoint",
        _FakeFastTokenizer(),
        output_dir,
        filename="tokenizer_2.json",
        subfolder="tokenizer_2",
    )
    assert path == os.path.join(output_dir, "tokenizer_2.json")
    assert os.path.isfile(path)
