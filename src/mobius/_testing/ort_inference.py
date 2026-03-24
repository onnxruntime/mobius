# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime inference session wrapper for ir.Model objects.

Handles bfloat16 and other non-standard dtypes transparently via ml_dtypes.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import onnxruntime.capi._pybind_state as _ort_c

from mobius._model_package import ModelPackage

_HAS_ML_DTYPES = importlib.util.find_spec("ml_dtypes") is not None
if _HAS_ML_DTYPES:
    # ml_dtypes provides bfloat16 and float8 variants not in standard numpy.
    # When absent, OrtValues are created via the plain numpy path and these
    # dtypes are unsupported.
    import ml_dtypes


def _ml_dtype_to_onnx_type(dtype: np.dtype) -> int | None:
    """Return the ONNX element type integer for an ml_dtypes dtype, or None.

    Only call this function when ``_HAS_ML_DTYPES`` is True.
    """
    if not _HAS_ML_DTYPES:
        return None
    tp = onnx.TensorProto
    if dtype == ml_dtypes.bfloat16:
        return tp.BFLOAT16
    if dtype == ml_dtypes.float8_e4m3fn:
        return tp.FLOAT8E4M3FN
    if dtype == ml_dtypes.float8_e4m3fnuz:
        return tp.FLOAT8E4M3FNUZ
    if dtype == ml_dtypes.float8_e5m2:
        return tp.FLOAT8E5M2
    if dtype == ml_dtypes.float8_e5m2fnuz:
        return tp.FLOAT8E5M2FNUZ
    if dtype == ml_dtypes.uint4:
        return tp.UINT4
    if dtype == ml_dtypes.int4:
        return tp.INT4
    if dtype == ml_dtypes.float4_e2m1fn:
        return tp.FLOAT4E2M1
    return None


def _to_ort_value(value: np.ndarray, device: str = "cpu") -> ort.OrtValue:
    """Convert a numpy array to an OrtValue, handling special ml_dtypes dtypes."""
    # Use DLPack when available (e.g. torch tensors or non-zero-size arrays)
    if hasattr(value, "__dlpack__"):
        is_zero_size = hasattr(value, "size") and value.size == 0
        if not is_zero_size:
            return ort.OrtValue(
                _ort_c.OrtValue.from_dlpack(value.__dlpack__(), False), value
            )
    if _HAS_ML_DTYPES and isinstance(value, np.ndarray):
        onnx_type = _ml_dtype_to_onnx_type(value.dtype)
        if onnx_type is not None:
            return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                value, onnx_element_type=onnx_type
            )
    return ort.OrtValue.ortvalue_from_numpy(np.asarray(value), device)


def _create_session(model_path: str, device: str = "cpu") -> ort.InferenceSession:
    """Create an ORT InferenceSession with sensible defaults.

    Args:
        model_path: Path to the ONNX model file.
        device: Execution device, ``"cpu"`` or ``"cuda"``.
    """
    if device == "cpu":
        providers = ("CPUExecutionProvider",)
    elif device == "cuda":
        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
    else:
        raise ValueError(f"Unsupported device: {device!r}. Expected 'cpu' or 'cuda'.")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 2  # warning
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


class OnnxModelSession:
    """Wraps an ``ort.InferenceSession`` for an ``ir.Model``.

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
        device: str = "cpu",
    ):
        if isinstance(model, ModelPackage):
            if len(model) != 1:
                raise ValueError(
                    f"ModelPackage has {len(model)} models; pass a "
                    f"single ir.Model or index into the package."
                )
            model = next(iter(model.values()))
        self._device = device
        self._tmpdir = tempfile.TemporaryDirectory()
        self._model_path = str(Path(self._tmpdir.name) / "model.onnx")
        ir.save(model, self._model_path, external_data="model.onnx.data")

        self._session = _create_session(self._model_path, device=device)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    @property
    def input_names(self) -> list[str]:
        return self._input_names

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
            ort_feeds[k] = _to_ort_value(v, device=self._device)
        run_opts = ort.RunOptions()
        run_opts.log_severity_level = 2  # warning
        raw_outputs = self._session.run_with_ort_values(
            None, ort_feeds, run_options=run_opts
        )
        return dict(zip(self._output_names, (o.numpy() for o in raw_outputs)))

    def close(self) -> None:
        self._tmpdir.cleanup()

    def __del__(self) -> None:
        self.close()
