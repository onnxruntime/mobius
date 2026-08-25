# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit a runnable package from a GGUF build.

Saving the ONNX graph is not enough to run a model: the runtime also needs a
tokenizer and its own configuration contract. Those come from two different
places — the tokenizer from the GGUF's embedded ggml metadata, the contract
from the built package — so a caller that saves the graph and stops produces a
directory that loads nowhere.

This module is the single place that knows the full artifact set, so the CLI
and the Python API cannot drift apart on what a complete package contains, and
so both supported runtimes are reachable from one entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mobius.integrations.gguf._tokenizer import write_gguf_tokenizer_json

__all__ = ["write_gguf_runtime_package"]

Runtime = Literal["onnx-genai", "ort-genai"]


def write_gguf_runtime_package(
    pkg: Any,
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    runtime: Runtime = "onnx-genai",
    save_model: bool = True,
    **save_kwargs: Any,
) -> dict[str, str]:
    """Write a complete, loadable package for a GGUF-built model.

    Emits the ONNX graph (unless it is already saved), a ``tokenizer.json``
    reconstructed from the GGUF's embedded ggml metadata, and the selected
    runtime's configuration contract.

    Args:
        pkg: The :class:`~mobius.ModelPackage` returned by
            :func:`~mobius.integrations.gguf.build_from_gguf`.
        gguf_path: The source ``.gguf`` file. The tokenizer is reconstructed
            from it because a GGUF checkpoint has no Hugging Face source
            directory to copy one from.
        output_dir: Destination directory.
        runtime: Which runtime contract to emit. ``"onnx-genai"`` writes
            ``inference_metadata.yaml``; ``"ort-genai"`` writes
            ``genai_config.json``.
        save_model: Save the ONNX graph too. Pass ``False`` when the caller has
            already saved it and only wants the runtime artifacts.
        **save_kwargs: Forwarded to :meth:`ModelPackage.save`.

    Returns:
        Mapping of artifact name to written path. The ``tokenizer`` key is
        absent when the GGUF carries no tokenizer metadata to rebuild from.

    Raises:
        ValueError: If ``runtime`` is not a supported runtime name.
    """
    if runtime not in ("onnx-genai", "ort-genai"):
        raise ValueError(f"Unknown runtime {runtime!r}; expected 'onnx-genai' or 'ort-genai'.")
    if runtime == "ort-genai" and getattr(pkg, "gguf_reuse_plan", None) is not None:
        raise ValueError(
            "ORT GenAI packaging is not supported with reused GGUF weights because "
            "genai_config.json has no supported setting that disables ORT constant "
            "folding. Use direct ONNX Runtime with ORT_DISABLE_ALL, or build without "
            "reuse_gguf_weights."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    if save_model:
        pkg.save(str(output_dir), **save_kwargs)

    tokenizer_path = write_gguf_tokenizer_json(gguf_path, output_dir)
    if tokenizer_path is not None:
        artifacts["tokenizer"] = tokenizer_path

    if runtime == "ort-genai":
        from mobius.integrations.ort_genai import write_ort_genai_config

        # A GGUF checkpoint has no Hugging Face source, so there is no
        # `hf_model_id` or local config directory to copy tokenizer files from;
        # the tokenizer written above is the one this package ships.
        artifacts.update(write_ort_genai_config(pkg, str(output_dir)))
    else:
        from mobius.integrations.onnx_genai import write_onnx_genai_config

        artifacts.update(
            write_onnx_genai_config(
                pkg, str(output_dir), config=getattr(pkg, "config", None), source=None
            )
        )
    return artifacts
