# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime inference session wrapper for ir.Model objects.

Uses ``onnxruntime-easy`` which handles bfloat16 and other non-standard
dtypes transparently.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime_easy as ort_easy

from mobius._model_package import ModelPackage


def _fix_scan_body_value_info(graph) -> None:
    """Patch Scan body value_info names after ONNX function inlining.

    The ONNX inliner appends a suffix (e.g. ``__2``) to node output
    names inside inlined function bodies but does *not* update the
    corresponding ``value_info`` entries.  This leaves the shape
    annotations orphaned — ORT cannot match them to actual values,
    so CUDA EP falls back to ``dim_value=0`` (size 1) for all dims.

    This function detects the suffix by comparing original value_info
    names against actual node output names, then renames value_info
    entries to match.
    """
    for node in graph.node:
        if node.op_type != "Scan":
            continue
        for attr in node.attribute:
            if attr.name != "body":
                continue
            body = attr.g

            # Collect all node output names in the body
            node_outputs = set()
            for n in body.node:
                for o in n.output:
                    node_outputs.add(o)

            # Detect the inliner suffix by finding a value_info name
            # that is a prefix of a node output name
            suffix = ""
            for vi in body.value_info:
                for name in node_outputs:
                    if name.startswith(vi.name) and len(name) > len(vi.name):
                        suffix = name[len(vi.name):]
                        break
                if suffix:
                    break

            if not suffix:
                continue

            # Update value_info names to include the suffix
            for vi in body.value_info:
                new_name = vi.name + suffix
                if new_name in node_outputs:
                    vi.name = new_name


class OnnxModelSession:
    """Wraps an ``onnxruntime_easy.EasySession`` for an ``ir.Model``.

    Serializes the model to a temporary file and creates an ORT session.
    Provides a simple ``run()`` interface that accepts and returns numpy arrays.

    Example::

        from mobius._testing.ort_inference import OnnxModelSession

        session = OnnxModelSession(model)
        outputs = session.run({"input_ids": np.array([[1, 2, 3]])})
        logits = outputs["logits"]
    """

    def __init__(
        self,
        model: ir.Model | ModelPackage,
        **load_kwargs,
    ):
        if isinstance(model, ModelPackage):
            if len(model) != 1:
                raise ValueError(
                    f"ModelPackage has {len(model)} models; pass a "
                    f"single ir.Model or index into the package."
                )
            model = next(iter(model.values()))
        self._tmpdir = tempfile.TemporaryDirectory()
        self._model_path = str(Path(self._tmpdir.name) / "model.onnx")
        ir.save(model, self._model_path, external_data="model.onnx.data")

        # Inline local functions before loading with ORT.
        # ORT's own function inliner drops dim_value annotations from
        # Scan body subgraph inputs, causing CUDA EP to misallocate
        # carry buffers (dim_value=0 → size 1).  Pre-inlining with
        # the ONNX library preserves concrete shapes.
        #
        # Additionally, disable memory pattern pre-allocation for
        # models containing Scan nodes.  ORT's memory planner
        # cannot resolve symbolic dims in Scan body outputs, so
        # it pre-allocates undersized buffers.  Disabling the
        # pattern forces runtime allocation with actual shapes.
        if model.functions:
            self._inline_functions()
            load_kwargs.setdefault("enable_mem_pattern", False)
            load_kwargs.setdefault("enable_mem_reuse", False)

        self._session = ort_easy.load(self._model_path, **load_kwargs)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    def _inline_functions(self) -> None:
        """Inline local functions in the saved model.

        ORT's own function inliner renames Scan body node outputs
        but does not update the body's ``value_info`` entries, leaving
        them orphaned.  Without matching value_info, CUDA EP cannot
        resolve intermediate shapes and falls back to ``dim_value=0``
        (size 1) for symbolic dims.

        This method:
        1. Inlines with the ONNX library (preserves body input shapes).
        2. Patches each Scan body's value_info names to match the
           inliner's suffix convention (e.g. ``__2``).
        """
        import onnx
        from onnx.inliner import inline_local_functions

        proto = onnx.load(self._model_path)
        inlined = inline_local_functions(proto)
        _fix_scan_body_value_info(inlined.graph)
        onnx.save(
            inlined,
            self._model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
        )

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    def get_input_shape(self, name: str) -> list[int | str] | None:
        """Return the declared shape of an input, or ``None`` if not found.

        Shape elements may be ``int`` (static) or ``str`` (symbolic).
        """
        for inp in self._session.get_inputs():
            if inp.name == name:
                return list(inp.shape)
        return None

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference and return outputs as a name→array dict.

        Args:
            feeds: Input name → numpy array mapping. Only inputs present
                in the model are used; extra keys are ignored.

        Returns:
            Dict mapping output names to numpy arrays.
        """
        # Filter to only inputs the model expects and convert to OrtValues
        ort_feeds = {}
        for k, v in feeds.items():
            if k not in self._input_names:
                continue
            # Ensure contiguous layout for ORT. Skip 0-d scalars
            # because np.ascontiguousarray promotes them to 1-d.
            if v.ndim > 0:
                v = np.ascontiguousarray(v)
            ort_feeds[k] = ort_easy.ort_value(v)
        raw_outputs = self._session(**ort_feeds)
        return dict(zip(self._output_names, (o.numpy() for o in raw_outputs)))

    def close(self) -> None:
        self._tmpdir.cleanup()

    def __del__(self) -> None:
        self.close()
