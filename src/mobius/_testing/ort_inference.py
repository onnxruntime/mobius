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
import onnxruntime as ort
import onnxruntime_easy as ort_easy

from mobius._flags import flags
from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)


# Map ORT element type strings to ml_dtypes numpy dtypes.  These are types
# that ``onnxruntime.OrtValue.numpy()`` cannot return natively because NumPy
# has no built-in dtype, but ``ml_dtypes`` extends NumPy with them.
_ML_DTYPE_OUTPUT_MAP: dict[str, str] = {
    "tensor(bfloat16)": "bfloat16",
    "tensor(float8e4m3fn)": "float8_e4m3fn",
    "tensor(float8e4m3fnuz)": "float8_e4m3fnuz",
    "tensor(float8e5m2)": "float8_e5m2",
    "tensor(float8e5m2fnuz)": "float8_e5m2fnuz",
    "tensor(uint4)": "uint4",
    "tensor(int4)": "int4",
    "tensor(float4e2m1)": "float4_e2m1fn",
}


def _ort_value_to_numpy(value: ort.OrtValue) -> np.ndarray:
    """Convert an OrtValue to numpy, with bf16 / ml_dtypes fallback.

    ``OrtValue.numpy()`` raises for dtypes that are not in core NumPy
    (e.g. bfloat16).  For those types we copy the raw bytes via DLPack
    + torch and reinterpret as the matching ``ml_dtypes`` scalar.
    """
    elem_type = value.data_type()
    ml_name = _ML_DTYPE_OUTPUT_MAP.get(elem_type)
    if ml_name is None:
        return value.numpy()
    import ml_dtypes
    import torch

    target_dtype = np.dtype(getattr(ml_dtypes, ml_name))
    bytes_per_elem = target_dtype.itemsize
    tensor = torch.utils.dlpack.from_dlpack(value._ortvalue.to_dlpack())
    raw = tensor.contiguous().view(torch.uint8).cpu().numpy()
    arr = raw.view(target_dtype)
    shape = tuple(value.shape())
    if bytes_per_elem * int(np.prod(shape) or 1) != raw.size:
        # 4-bit types pack 2 elements per byte — keep packed view.
        return arr
    return arr.reshape(shape)


def _numpy_to_ort_value(value: np.ndarray) -> ort.OrtValue:
    """Convert a NumPy array to an OrtValue, supporting ml_dtypes (bf16, etc).

    ``onnxruntime_easy.ort_value`` prefers the DLPack path, which is broken
    for ml_dtypes scalars (NumPy's __dlpack__ rejects non-standard dtypes).
    Route ml_dtypes arrays through ``ortvalue_from_numpy_with_onnx_type``
    instead.
    """
    if isinstance(value, np.ndarray):
        try:
            import ml_dtypes
        except ImportError:  # pragma: no cover
            ml_dtypes = None
        if ml_dtypes is not None:
            onnx_type = ort_easy._ml_dtypes_to_onnx_type(value.dtype)
            if onnx_type is not None:
                return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                    np.ascontiguousarray(value), onnx_element_type=onnx_type
                )
    return ort_easy.ort_value(value)


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

    def get_input_dtype(self, name: str) -> np.dtype | None:
        """Return the numpy dtype for an input, or ``None`` if not found.

        For non-NumPy native dtypes (e.g. bfloat16), returns the matching
        ``ml_dtypes`` dtype.
        """
        for inp in self._session.get_inputs():
            if inp.name == name:
                t = inp.type
                ml_name = _ML_DTYPE_OUTPUT_MAP.get(t)
                if ml_name is not None:
                    import ml_dtypes

                    return np.dtype(getattr(ml_dtypes, ml_name))
                # Map common ORT type strings to numpy
                mapping = {
                    "tensor(float)": np.float32,
                    "tensor(float16)": np.float16,
                    "tensor(double)": np.float64,
                    "tensor(int32)": np.int32,
                    "tensor(int64)": np.int64,
                    "tensor(uint8)": np.uint8,
                    "tensor(int8)": np.int8,
                    "tensor(bool)": np.bool_,
                }
                np_t = mapping.get(t)
                return np.dtype(np_t) if np_t is not None else None
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
            ort_feeds[k] = _numpy_to_ort_value(v)
        raw_outputs = self._session(**ort_feeds)
        return dict(zip(self._output_names, (_ort_value_to_numpy(o) for o in raw_outputs)))

    def close(self) -> None:
        self._tmpdir.cleanup()

    def __del__(self) -> None:
        self.close()
