"""Export the tiny synthetic DeepSeek-V2 MLA + MoE model for onnx-genai E2E."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

import numpy as np
import onnx_ir as ir

from _test_configs import ALL_CAUSAL_LM_CONFIGS, _base_config
from mobius._config_resolver import _default_task_for_model
from mobius._registry import registry
from mobius.integrations.onnx_genai import write_inference_metadata
from mobius.tasks import get_task

OUT = "/home/justinchu/ds-e2e-artifacts/deepseek-v2-tiny"


def _fill_random_weights(model: ir.Model, rng: np.random.Generator) -> None:
    for init in model.graph.initializers.values():
        if init.const_value is not None:
            continue
        shape = tuple(int(d) for d in init.shape)
        if not shape:
            continue
        if init.dtype == ir.DataType.FLOAT:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        elif init.dtype == ir.DataType.FLOAT16:
            data = (rng.standard_normal(shape) * 0.02).astype(np.float16)
        elif init.dtype in (ir.DataType.INT64, ir.DataType.INT32):
            npd = np.int64 if init.dtype == ir.DataType.INT64 else np.int32
            data = rng.integers(0, 10, size=shape).astype(npd)
        else:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        init.const_value = ir.Tensor(data)


def main() -> None:
    overrides = dict(
        next(ov for mt, ov, _ in ALL_CAUSAL_LM_CONFIGS if mt == "deepseek_v2")
    )
    config = _base_config(**overrides)
    config.dtype = ir.DataType.FLOAT

    module = registry.get("deepseek_v2")(config)
    task = get_task(_default_task_for_model("deepseek_v2"))
    pkg = task.build(module, config)

    rng = np.random.default_rng(0)
    for model in pkg.values():
        _fill_random_weights(model, rng)

    os.makedirs(OUT, exist_ok=True)
    pkg.save(OUT, external_data="onnx", check_weights=False)
    print("inference_metadata:", write_inference_metadata(pkg, OUT))
    print("Saved to", OUT)
    print("files:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
