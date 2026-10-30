"""Per-step GPU flow: prepare inputs -> execute -> read outputs.

Setup, allocation, safety checks, and cleanup live in _runtime_support.py.
Cache input/output buffers share GPU addresses: execution updates them in place.
"""

from __future__ import annotations

import numpy as np

from tensorrt_debug._runtime_support import RuntimeResources


class TensorRTRunner(RuntimeResources):
    """The inference steps; resource management is inherited from RuntimeResources."""

    def prepare_inputs(self, feeds):
        """Set shapes, bind GPU addresses, then upload feeds. Leave caches untouched."""
        self.set_input_shapes(feeds)
        self.bind_buffers()
        for name, array in feeds.items():
            array = np.ascontiguousarray(array, dtype=self.numpy_dtype(name))
            self.checked(
                self.cuda.cudaMemcpy(
                    self.buffers[name][0],
                    array.ctypes.data,
                    array.nbytes,
                    self.cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )

    def execute(self):
        """Compute next-token scores. The caller selects a token with argmax()."""
        # Run the compiled model on the GPU: update KV caches and compute logits.
        inference_started = self.context.execute_async_v3(int(self.stream))
        if not inference_started:
            raise RuntimeError("TensorRT execution failed")

        # The launch is asynchronous; wait before reading the GPU results.
        self.synchronize()
        logits_on_cpu = self.read_tensor("logits")  # [batch, input_length, vocabulary]

        # Batch 0, last input position: one score per possible next-token ID.
        next_token_scores = logits_on_cpu[0, -1, :].astype(np.float32)
        if not np.isfinite(next_token_scores).all():
            raise RuntimeError("Non-finite logits")
        return next_token_scores

    def read_tensor(self, name):
        """Return an independent CPU copy, preserving the original dtype/bytes."""
        self.synchronize()
        shape, dtype = self.validate_tensor(name)
        snapshot = np.empty(shape, dtype=dtype)
        if snapshot.nbytes > self.buffers[name][1]:
            raise ValueError(f"{name}: snapshot exceeds allocated buffer")
        self.checked(
            self.cuda.cudaMemcpy(
                snapshot.ctypes.data,
                self.buffers[name][0],
                snapshot.nbytes,
                self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
        )
        return snapshot

    def reset_caches(self):
        """Full-prefix control: erase cached history before recomputing all tokens."""
        for name in self.caches:
            pointer, size = self.buffers[name]
            self.checked(self.cuda.cudaMemsetAsync(pointer, 0, size, self.stream))
        self.synchronize()
