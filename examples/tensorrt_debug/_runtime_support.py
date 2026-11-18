"""TensorRT setup, buffer allocation, validation, and cleanup.

These mechanics are separate from the per-step debugging flow in runtime.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import ml_dtypes
import numpy as np


class RuntimeResources:
    """Own the engine and GPU resources used by TensorRTRunner."""

    def __init__(self, engine_path: Path, sdk: Path):
        self.dll_handles = []
        self.allocations = []
        self.stream = None
        self.context = self.engine = self.runtime = None
        directories = [sdk / name for name in ("bin", "lib") if (sdk / name).exists()]
        os.environ["PATH"] = (
            os.pathsep.join(map(str, directories)) + os.pathsep + os.environ["PATH"]
        )
        try:
            if os.name == "nt":
                for directory in directories:
                    self.dll_handles.append(os.add_dll_directory(str(directory)))
            self._load_engine(engine_path)
            self.buffers = {}
            self.stream = self.checked(self.cuda.cudaStreamCreate())
            self._allocate_buffers()
        except BaseException:
            self.close()
            raise

    def _load_engine(self, engine_path):
        # Load TensorRT only after the SDK DLL directories are available.
        import tensorrt as trt
        from cuda.bindings import runtime as cuda

        self.trt, self.cuda = trt, cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("Engine deserialization failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Execution context creation failed")
        self.names = [
            self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)
        ]
        self.inputs = [
            name
            for name in self.names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.caches = [
            name for name in self.inputs if name.startswith(("key_cache.", "value_cache."))
        ]
        if not self.caches:
            raise ValueError("Engine has no static KV cache inputs")
        self.capacity = int(self.engine.get_tensor_shape(self.caches[0])[2])
        self.max_chunk = int(self.engine.get_tensor_profile_shape("input_ids", 0)[2][1])

    def checked(self, result):
        status, *values = result
        if status != self.cuda.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA call failed: {status}")
        return values[0] if len(values) == 1 else values

    def numpy_dtype(self, name):
        dtype = self.engine.get_tensor_dtype(name)
        if dtype == self.trt.bfloat16:
            return np.dtype(ml_dtypes.bfloat16)
        return np.dtype(self.trt.nptype(dtype))

    def validate_tensor(self, name, show_layout=False):
        """Reject layouts that cannot be read as a contiguous NumPy array."""
        shape = tuple(self.context.get_tensor_shape(name))
        dtype = self.numpy_dtype(name)
        strides = tuple(self.context.get_tensor_strides(name))
        location = self.engine.get_tensor_location(name)
        tensor_format = self.engine.get_tensor_format(name)
        expected_strides = tuple(int(np.prod(shape[axis + 1 :])) for axis in range(len(shape)))
        if show_layout:
            print(
                f"Cache layout {name}: shape={shape}; dtype={self.engine.get_tensor_dtype(name)}; "
                f"location={location}; format={tensor_format}; strides={strides}",
                flush=True,
            )
        if (
            any(dim <= 0 for dim in shape)
            or location != self.trt.TensorLocation.DEVICE
            or tensor_format != self.trt.TensorFormat.LINEAR
            or strides != expected_strides
        ):
            raise ValueError(f"{name}: runner requires resolved contiguous device I/O")
        return shape, dtype

    def _allocate_buffers(self):
        """Allocate once for the profile's maximum chunk length."""
        total_bytes = 0
        for name in self.names:
            if name.startswith(("updated_key_cache.", "updated_value_cache.")):
                continue
            shape = tuple(self.engine.get_tensor_shape(name))
            if name in self.inputs:
                shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[2])
            else:
                shape = tuple(
                    1 if dim < 0 and axis == 0 else self.max_chunk if dim < 0 else dim
                    for axis, dim in enumerate(shape)
                )
            if any(dim < 1 for dim in shape):
                raise ValueError(f"Unresolved allocation shape for {name}: {shape}")
            size = int(np.prod(shape)) * self.numpy_dtype(name).itemsize
            pointer = self.checked(self.cuda.cudaMalloc(size))
            self.allocations.append(pointer)
            self.buffers[name] = (pointer, size)
            self.checked(self.cuda.cudaMemset(pointer, 0, size))
            total_bytes += size
        # The tested TensorRT engines require each updated cache to alias its input.
        for name in self.caches:
            self.buffers["updated_" + name] = self.buffers[name]
        print(f"Allocated I/O buffers: {total_bytes / 1024**2:.1f} MiB", flush=True)

    def set_input_shapes(self, feeds):
        for name in self.inputs:
            shape = (
                feeds[name].shape
                if name in feeds
                else tuple(1 if dim < 0 else dim for dim in self.engine.get_tensor_shape(name))
            )
            minimum, _, maximum = self.engine.get_tensor_profile_shape(name, 0)
            if len(shape) != len(minimum) or any(
                actual < low or actual > high
                for actual, low, high in zip(shape, minimum, maximum)
            ):
                raise ValueError(
                    f"Input {name} shape {shape} outside profile {minimum}..{maximum}"
                )
            if not self.context.set_input_shape(name, shape):
                raise RuntimeError(f"Cannot set input shape for {name}")

    def bind_buffers(self):
        for name in self.names:
            shape, dtype = self.validate_tensor(name)
            if int(np.prod(shape)) * dtype.itemsize > self.buffers[name][1]:
                raise ValueError(f"Buffer too small for {name}")
            if not self.context.set_tensor_address(name, int(self.buffers[name][0])):
                raise RuntimeError(f"Cannot bind {name}")

    def synchronize(self):
        self.checked(self.cuda.cudaStreamSynchronize(self.stream))

    def close(self):
        if self.stream is not None:
            self.cuda.cudaStreamSynchronize(self.stream)
        self.context = None
        self.engine = None
        self.runtime = None
        for pointer in self.allocations:
            self.cuda.cudaFree(pointer)
        self.allocations.clear()
        if self.stream is not None:
            self.cuda.cudaStreamDestroy(self.stream)
            self.stream = None
        for handle in self.dll_handles:
            handle.close()
        self.dll_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
