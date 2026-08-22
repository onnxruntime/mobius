# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit a runnable onnx-genai package from a GGUF build.

Saving the ONNX graph is not enough to run a model: the runtime also needs a
tokenizer and the inference metadata contract. Those come from two different
places — the tokenizer from the GGUF's embedded ggml metadata, the metadata
from the built package — so a caller that saves the graph and stops produces a
directory that loads nowhere.

This module is the single place that knows the full artifact set, so the CLI
and the Python API cannot drift apart on what a complete package contains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mobius.integrations.gguf._tokenizer import write_gguf_tokenizer_json

__all__ = ["write_gguf_runtime_package"]


def write_gguf_runtime_package(
    pkg: Any,
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    save_model: bool = True,
    **save_kwargs: Any,
) -> dict[str, str]:
    """Write a complete, loadable onnx-genai package for a GGUF-built model.

    Emits the ONNX graph (unless it is already saved), a ``tokenizer.json``
    reconstructed from the GGUF's embedded ggml metadata, and the
    ``inference_metadata.yaml`` runtime contract.

    Args:
        pkg: The :class:`~mobius.ModelPackage` returned by
            :func:`~mobius.integrations.gguf.build_from_gguf`.
        gguf_path: The source ``.gguf`` file. The tokenizer is reconstructed
            from it because a GGUF checkpoint has no Hugging Face source
            directory to copy one from.
        output_dir: Destination directory.
        save_model: Save the ONNX graph too. Pass ``False`` when the caller has
            already saved it and only wants the runtime artifacts.
        **save_kwargs: Forwarded to :meth:`ModelPackage.save`.

    Returns:
        Mapping of artifact name to written path. The ``tokenizer`` key is
        absent when the GGUF carries no tokenizer metadata to rebuild from.
    """
    from mobius.integrations.onnx_genai import write_onnx_genai_config

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    if save_model:
        pkg.save(str(output_dir), **save_kwargs)

    tokenizer_path = write_gguf_tokenizer_json(gguf_path, output_dir)
    if tokenizer_path is not None:
        artifacts["tokenizer"] = tokenizer_path

    artifacts.update(
        write_onnx_genai_config(
            pkg, str(output_dir), config=getattr(pkg, "config", None), source=None
        )
    )
    return artifacts
