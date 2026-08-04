"""Export the tiny synthetic glm_moe_dsa model for onnx-genai E2E bring-up.

Builds the tiny `glm_moe_dsa` config from tests/_test_configs.py, fills random
weights, and writes an onnx-genai-loadable artifact directory:
  <out>/model.onnx (+ external data), inference_metadata.yaml, tokenizer.json
"""

from __future__ import annotations

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

import numpy as np
import onnx_ir as ir

from _test_configs import _base_config, ALL_CAUSAL_LM_CONFIGS
from mobius._config_resolver import _default_task_for_model
from mobius._registry import registry
from mobius.tasks import get_task
from mobius.integrations.onnx_genai import write_inference_metadata

OUT = "/home/justinchu/glm-e2e-artifacts/glm-5.2-tiny"


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
    overrides = dict(next(ov for mt, ov, _ in ALL_CAUSAL_LM_CONFIGS if mt == "glm_moe_dsa"))
    config = _base_config(**overrides)
    config.dtype = ir.DataType.FLOAT

    model_type = "glm_moe_dsa"
    module = registry.get(model_type)(config)
    task = get_task(_default_task_for_model(model_type))
    pkg = task.build(module, config)

    rng = np.random.default_rng(0)
    for model in pkg.values():
        _fill_random_weights(model, rng)

    os.makedirs(OUT, exist_ok=True)
    pkg.save(OUT, external_data="onnx", check_weights=False)
    path = write_inference_metadata(pkg, OUT)
    print("inference_metadata:", path)

    # Report config essentials
    for attr in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                 "head_dim", "hidden_size", "vocab_size", "max_position_embeddings",
                 "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "kv_lora_rank",
                 "q_lora_rank"):
        print(f"  {attr} =", getattr(config, attr, None))

    print("Saved to", OUT)
    print("files:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
