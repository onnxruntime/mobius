"""Export a tiny synthetic glm_moe_dsa model with a FUSED ``com.microsoft::QMoE``.

Same tiny `glm_moe_dsa` config as ``export_glm_tiny_quant.py`, but sets
``config.fused_quantized_moe = True`` so the routed MoE experts are emitted as a
single fused ``com.microsoft::QMoE`` node (int4, block-32, expert-major layout)
instead of a per-expert unroll of ``com.microsoft::MatMulNBits``.

Routing (GLM sigmoid + noaux_tc, n_group=1) is preserved exactly: the QMoE
kernel re-derives top-k selection from ``router_probs`` (the gate's
``scores_for_choice``), and the normalized+scaled combine weights are scattered
into the optional ``router_weights`` input with ``normalize_routing_weights=0``.

Writes an onnx-genai-loadable artifact directory:
  <out>/model.onnx (+ external data), inference_metadata.yaml, tokenizer.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

import numpy as np
import onnx_ir as ir
from _test_configs import ALL_CAUSAL_LM_CONFIGS, _base_config

from mobius._config_resolver import _default_task_for_model
from mobius._configs import QuantizationConfig
from mobius._registry import registry
from mobius.integrations.onnx_genai import write_onnx_genai_config
from mobius.tasks import get_task

ARTIFACTS_DIR = os.environ.get(
    "MOBIUS_ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "artifacts")
)


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
        elif init.dtype == ir.DataType.UINT8:
            # Packed int4/int8 quantized weights or bit-packed zero points.
            data = rng.integers(0, 256, size=shape).astype(np.uint8)
        elif init.dtype in (ir.DataType.INT64, ir.DataType.INT32):
            npd = np.int64 if init.dtype == ir.DataType.INT64 else np.int32
            data = rng.integers(0, 10, size=shape).astype(npd)
        else:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        init.const_value = ir.Tensor(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ARTIFACTS_DIR, "glm-5.2-tiny-qmoe"),
    )
    parser.add_argument(
        "--fp32-output-dir",
        default=os.path.join(ARTIFACTS_DIR, "glm-5.2-tiny"),
    )
    args = parser.parse_args()

    overrides = dict(next(ov for mt, ov, _ in ALL_CAUSAL_LM_CONFIGS if mt == "glm_moe_dsa"))
    config = _base_config(**overrides)
    config.dtype = ir.DataType.FLOAT
    # int4, block-32, symmetric -> fused QMoE (no zero points; the kernel
    # defaults per-block zero-point to 1 << (bits - 1)).
    config.quantization = QuantizationConfig(
        bits=4,
        group_size=32,
        quant_method="gguf",
        sym=True,
    )
    # Emit routed experts as a single fused com.microsoft::QMoE node.
    config.fused_quantized_moe = True

    model_type = "glm_moe_dsa"
    module = registry.get(model_type)(config)
    task = get_task(_default_task_for_model(model_type))
    pkg = task.build(module, config)

    rng = np.random.default_rng(0)
    for model in pkg.values():
        _fill_random_weights(model, rng)

    os.makedirs(args.output_dir, exist_ok=True)
    pkg.save(args.output_dir, external_data="onnx", check_weights=False)
    artifacts = write_onnx_genai_config(pkg, args.output_dir, config=config)
    for name, path in artifacts.items():
        print(f"{name}:", path)

    # Reuse tokenizer from the fp32 artifacts if present.
    tok_src = os.path.join(args.fp32_output_dir, "tokenizer.json")
    if os.path.exists(tok_src):
        shutil.copy(tok_src, os.path.join(args.output_dir, "tokenizer.json"))

    for attr in (
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_size",
        "vocab_size",
        "max_position_embeddings",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "kv_lora_rank",
        "q_lora_rank",
        "num_local_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    ):
        print(f"  {attr} =", getattr(config, attr, None))

    print("Saved to", args.output_dir)
    print("files:", sorted(os.listdir(args.output_dir)))


if __name__ == "__main__":
    main()
