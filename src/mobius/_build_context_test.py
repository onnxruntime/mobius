# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import threading

import onnx_ir as ir
import pytest

from mobius._build_context import (
    build_context,
    ep_capabilities,
    get_build_dtype,
    is_prefill_prefix_pruning_enabled,
    prefill_prefix_pruning,
)
from mobius._execution_providers import EpCapabilities, ep_registry

_CUDA_CAPABILITIES = EpCapabilities(
    name="cuda",
    gqa_dtypes=frozenset({ir.DataType.FLOAT16, ir.DataType.BFLOAT16}),
)
_CPU_CAPABILITIES = EpCapabilities(
    name="cpu",
    gqa_dtypes=frozenset({ir.DataType.FLOAT}),
)


class TestBuildContextDefaults:
    def test_default_capabilities_is_default_ep(self):
        """No context active → returns the 'default' EP descriptor."""
        capabilities = ep_capabilities()
        assert capabilities.name == "default"

    def test_default_dtype_is_float32(self):
        """No context active → returns FLOAT."""
        assert get_build_dtype() == ir.DataType.FLOAT

    def test_prefill_prefix_pruning_is_disabled(self):
        assert not is_prefill_prefix_pruning_enabled()

    def test_default_capabilities_has_no_fusions(self):
        """Default EP has no GQA dtypes (portable ONNX)."""
        capabilities = ep_capabilities()
        assert len(capabilities.gqa_dtypes) == 0
        assert len(capabilities.qkv_pack_dtypes) == 0

    def test_webgpu_capabilities_enable_fp16_gqa(self):
        capabilities = ep_registry.require("webgpu")
        assert ir.DataType.FLOAT16 in capabilities.gqa_dtypes
        assert capabilities.supports_past_present_share_buffer


class TestBuildContextScoping:
    def test_capabilities_visible_inside_context(self):
        with build_context(_CUDA_CAPABILITIES, ir.DataType.FLOAT16):
            assert ep_capabilities().name == "cuda"
            assert get_build_dtype() == ir.DataType.FLOAT16

    def test_capabilities_restored_after_context(self):
        with build_context(_CUDA_CAPABILITIES, ir.DataType.FLOAT16):
            pass
        assert ep_capabilities().name == "default"
        assert get_build_dtype() == ir.DataType.FLOAT

    def test_capabilities_restored_on_exception(self):
        """Context restores state even if an exception is raised inside."""
        with (
            pytest.raises(RuntimeError, match="deliberate"),
            build_context(_CUDA_CAPABILITIES, ir.DataType.FLOAT16),
        ):
            raise RuntimeError("deliberate")
        assert ep_capabilities().name == "default"
        assert get_build_dtype() == ir.DataType.FLOAT

    def test_prefill_prefix_pruning_restored_after_context(self):
        with prefill_prefix_pruning(True):
            assert is_prefill_prefix_pruning_enabled()
        assert not is_prefill_prefix_pruning_enabled()


class TestBuildContextNesting:
    def test_inner_context_shadows_outer(self):
        with build_context(_CUDA_CAPABILITIES, ir.DataType.FLOAT16):
            assert ep_capabilities().name == "cuda"
            with build_context(_CPU_CAPABILITIES, ir.DataType.FLOAT):
                # Inner context takes over
                assert ep_capabilities().name == "cpu"
                assert get_build_dtype() == ir.DataType.FLOAT
            # Outer context restored
            assert ep_capabilities().name == "cuda"
            assert get_build_dtype() == ir.DataType.FLOAT16

    def test_outer_context_restored_after_inner(self):
        with build_context(_CUDA_CAPABILITIES, ir.DataType.BFLOAT16):
            with build_context(_CPU_CAPABILITIES, ir.DataType.FLOAT):
                pass
            assert ep_capabilities().name == "cuda"
            assert get_build_dtype() == ir.DataType.BFLOAT16
        # Fully unwound
        assert ep_capabilities().name == "default"


class TestBuildContextThreadIsolation:
    def test_threads_have_independent_contexts(self):
        """Two threads with different EPs must not interfere."""
        results: dict[str, str] = {}
        errors: list[Exception] = []

        def thread_cuda() -> None:
            try:
                with build_context(_CUDA_CAPABILITIES, ir.DataType.FLOAT16):
                    import time

                    time.sleep(0.05)  # yield to let other thread set its context
                    results["cuda"] = ep_capabilities().name
            except Exception as exc:
                errors.append(exc)

        def thread_cpu() -> None:
            try:
                with build_context(_CPU_CAPABILITIES, ir.DataType.FLOAT):
                    import time

                    time.sleep(0.05)
                    results["cpu"] = ep_capabilities().name
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=thread_cuda)
        t2 = threading.Thread(target=thread_cpu)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"
        assert results["cuda"] == "cuda", "cuda thread saw wrong EP"
        assert results["cpu"] == "cpu", "cpu thread saw wrong EP"

    def test_main_thread_unaffected_by_child_threads(self):
        """Main thread context is not polluted by child thread contexts."""
        child_finished = threading.Event()

        def child_thread() -> None:
            with build_context(_CUDA_CAPABILITIES, ir.DataType.FLOAT16):
                child_finished.set()
                import time

                time.sleep(0.1)

        thread = threading.Thread(target=child_thread)
        thread.start()
        child_finished.wait()

        # Main thread should still see defaults
        assert ep_capabilities().name == "default"
        assert get_build_dtype() == ir.DataType.FLOAT

        thread.join()
