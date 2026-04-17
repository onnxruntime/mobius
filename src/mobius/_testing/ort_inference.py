# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ONNX Runtime inference session wrapper for ir.Model objects.

Uses ``onnxruntime-easy`` which handles bfloat16 and other non-standard
dtypes transparently.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime_easy as ort_easy

from mobius._flags import flags
from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)

# Maximum default-domain ONNX opset that ORT ≤1.24.x CUDA/TRT EPs
# register kernels for.  Models built with a higher opset can be
# loaded after lowering the declared import — the op semantics have
# not changed, only the version label.
_MAX_EP_OPSET = 23


def _should_lower_opset(model: ir.Model, device: str) -> bool:
    """Return True when lowering the opset import is safe and needed.

    Lowering is only attempted when
    :attr:`~mobius._flags._Flags.ort_lower_opset_for_ep` is enabled and the
    target device is non-CPU with a default-domain opset exceeding
    ``_MAX_EP_OPSET``.

    Lowering is *not* safe when the graph contains ops that were first
    introduced in a post-23 opset (e.g. ``TensorScatter``).  In that
    case the model requires genuine opset 24+ support and lowering
    would produce an invalid model.
    """
    if not flags.ort_lower_opset_for_ep:
        return False
    if device == "cpu":
        return False
    current_opset = model.opset_imports.get("", 0)
    if current_opset <= _MAX_EP_OPSET:
        return False

    # Ops that were *introduced* in opset 24 and have no opset 23
    # equivalent.  If any appear in the graph, lowering is unsafe.
    opset_24_only_ops = {"TensorScatter"}
    for node in model.graph:
        if node.domain == "" and node.op_type in opset_24_only_ops:
            return False
    return True


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

        # Workaround: ORT ≤1.24.x CUDA/TRT EPs don't register kernels
        # for ONNX opset 24 standard ops (Squeeze, Reshape, etc.).  The
        # op semantics are identical to opset 23, so lowering the import
        # declaration lets the EP find its existing kernels.  The model
        # object is restored to its original opset after saving.
        device = load_kwargs.get("device", "cpu")
        lower_opset = _should_lower_opset(model, device)
        original_opset = model.opset_imports.get("", 0) if lower_opset else 0
        if lower_opset:
            logger.info(
                "Lowering default-domain opset from %d to %d for %s",
                original_opset,
                _MAX_EP_OPSET,
                device,
            )
            model.opset_imports[""] = _MAX_EP_OPSET

        self._tmpdir = tempfile.TemporaryDirectory()
        self._model_path = str(Path(self._tmpdir.name) / "model.onnx")
        try:
            ir.save(model, self._model_path, external_data="model.onnx.data")
        finally:
            if lower_opset:
                model.opset_imports[""] = original_opset

        self._session = ort_easy.load(self._model_path, **load_kwargs)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

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
