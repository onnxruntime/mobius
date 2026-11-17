# `build_from_gguf()`

Build an ONNX `ModelPackage` directly from GGUF metadata and tensors without
tracing PyTorch.

GGUF support is capability-specific: metadata parsing, tensor mapping, graph
construction, quantized storage, tokenizer materialization, and runtime validation
have independent verdicts. A successful graph import does not by itself mean that
runtime packaging is validated. See the
[GGUF capability and evidence catalog](../gguf-capability-catalog.md) for exhaustive,
generated verdicts and immutable evidence.

## Python usage

```python
from mobius import build_from_gguf

package = build_from_gguf("model.gguf")
package.save("output")
```

Quantized target storage is requested by default. Use
`keep_quantized=False` when the output must use float storage.

## Original GGUF external data (opset 29)

Enable `MOBIUS_EXPERIMENTAL_OPSET29=1`, retain `keep_quantized=True`, and select
`execution_provider="onnx-standard"`. Save **beside the original GGUF file(s)**:
Construction does not require installed opset-29 schemas. Checking and reference
execution require ONNX with `DequantizePacked-29` and `GroupedMatMul-29` schemas.

```bash
MOBIUS_EXPERIMENTAL_OPSET29=1 mobius build-gguf weights/model.gguf \
    --ep onnx-standard --output weights
```

The experimental path keeps original typed tensors and packed payloads as ONNX
external-data ranges in those files. It emits `DequantizePacked` and standard
layout operations, with `GroupedMatMul` for ordinary routed experts. It does not
repack, requantize, copy payloads into a new blob, or materialize a dense vocabulary
at export time. Row-aligned packed embeddings decode only requested rows.

Packages may contain multiple components and an independently cached MTP head,
sharing the same original, little-endian GGUF shards. `gguf-packed.json` v2
records component names and the MTP chain; single `model` exports retain v1.
Multi-component saves publish unique graph-generation filenames, then atomically
replace the manifest. Loading selects only that committed generation; unreferenced
graphs from prior or interrupted saves are ignored, not substituted for missing
components. Policy and adapter artifacts anywhere in the MTP chain must be
explicitly omitted or saved separately.
All sources must have unique filenames and reside in the output directory.
Treat source files as immutable; load/save verifies source digests, ranges, and
original external bindings. Transcoded safetensors output and new weight sharding
remain unsupported.

Lossless standalone mmproj conversion currently supports `janus_pro`:

```python
from mobius.integrations.gguf import build_mmproj_from_gguf

vision = build_mmproj_from_gguf(
    "weights/mmproj.gguf",
    projector_type="janus_pro",
    target_architecture="llama",
    execution_provider="onnx-standard",
)
vision.save("weights")
```

This exports a functional vision/projector component, not a paired text-insertion
runtime. Other lossless projector routes and the paired `mmproj=` path still
raise explicitly. Existing Qwen3.5/3.8 and Hunyuan-V3 MTP contract restrictions
remain: one physical prediction block, independent dynamic cache, explicit
hidden/embedding/head sharing; representation does not implement acceptance policy.

Ordinary MLP MoE experts now retain biases, activations, and exact gated or ungated
shared-expert composition under grouped dispatch. Specialized custom expert
compositions are not silently replaced. Flash-Next's existing fused expert banks
also use grouped dispatch, retaining FP32 routing/SiLU, normalization, scaling,
and shared gating. Grouped graphs avoid per-expert execution,
but full-bank dequantization still requires expert-sized workspace unless the EP
fuses decode with selection; no such fusion/performance claim is made.
LFM2MoE also has an exact grouped adapter retaining its selection-only correction
bias, sigmoid weights, optional normalization and routed scaling.

Merged `LinearAttention-29` supports independent `v_num_heads`; the portable
function can opt into a post-update state trajectory. This costs one recurrent
state per token and is not enabled as a whole-model rollback policy. Convolution,
PLE and indexed-attention history still need independent restoration. Actual
schema checking requires an ONNX build containing onnx/onnx#8693, not just a
version string reporting opset 29.
The flag does **not** add runtime kernels. See the
[opset-29 survey](../gguf-capability-catalog.md#lossless-opset-29-survey).

Former lossy affine/requantization routes now raise actionable errors when the
flag is off. Exact legacy routes remain available. Explicit `--dequantize` is a
separate float-storage choice, not source-byte preservation.

## Signature

```python
def build_from_gguf(
    gguf_path: str | Path,
    *,
    task: str | ModelTask | None = None,
    dtype: str | None = None,
    keep_quantized: bool = True,
    execution_provider: str = "default",
    mmproj: str | Path | None = None,
    image_token_id: int | None = None,
    static_cache: bool = False,
    max_seq_len: int | None = None,
    allow_dense_moe: bool | None = None,
    reuse_gguf_weights: bool = False,
    target_config: str | Path | Mapping[str, object] | None = None,
    output_layer_indices: Sequence[int] | None = None,
) -> ModelPackage:
```

| Parameter | Description |
|---|---|
| `gguf_path` | GGUF model path. |
| `task` | Optional task override; otherwise selected from GGUF architecture metadata. |
| `dtype` | Target float dtype, such as `"f32"`, `"f16"`, or `"bf16"`. |
| `keep_quantized` | Request supported exact legacy packed storage; with the opset-29 flag, retain original GGUF external payloads. Unsupported routes fail rather than requantize. |
| `execution_provider` | Target EP for EP-aware graph optimization; `"default"` emits portable ONNX. |
| `mmproj` | Companion projector GGUF for a registry-evidenced multimodal route. |
| `image_token_id` | Processor-owned image placeholder ID for an `mmproj` package. |
| `static_cache`, `max_seq_len` | Request a fixed-width KV cache and optionally set its length. |
| `allow_dense_moe` | Opt in to a dense fallback for supported MoE imports. |
| `reuse_gguf_weights` | Reuse compatible source tensor byte ranges in the saved package. |
| `target_config` | Exact target configuration for a supported speculative draft model. |
| `output_layer_indices` | Optional hidden-layer outputs to expose. |

## CLI usage

```text
mobius build-gguf GGUF_PATH --output OUTPUT_DIR [options]
```

```bash
# Quantized target storage where supported
mobius build-gguf model.gguf --output output/

# Explicit float storage
mobius build-gguf model.gguf --output output-float/ --dequantize
```

Key options:

| Option | Purpose |
|---|---|
| `--dequantize` | Store all mapped weights as float instead of requesting quantized target storage. |
| `--dtype {bf16,f16,f32}` | Set the target float dtype. |
| `--ep EP`, `--execution-provider EP` | Select EP-aware optimization; use `mobius list eps` for available values. |
| `--mmproj PATH`, `--image-token-id ID` | Build an evidenced multimodal package with a companion projector. |
| `--runtime {ort-genai,onnx-genai}` | Request runtime-specific metadata and tokenizer packaging. |
| `--runtime-version VERSION` | Record the selected runtime version for exact evidence matching. |
| `--static-cache`, `--max-seq-len N` | Build a fixed-width cache; `--max-seq-len` requires `--static-cache`. |
| `--target-gguf PATH`, `--target-config PATH` | Build a supported DFlash/EAGLE3 target-draft pair. |
| `--reuse-gguf-weights` | Reuse compatible source tensor byte ranges directly from the GGUF; converted weights are written to `model.onnx.data`. |
| `--external-data {onnx,safetensors}`, `--max-shard-size SIZE` | Select and size external weight storage. |
| `--release` | Strip build-only debug and provenance metadata. |

See the [complete CLI reference](../cli_reference.md) for all options and
compatibility constraints.

## Output and status behavior

The Python API returns a `ModelPackage`; the CLI saves it to `--output`. Saved GGUF
packages record conversion fidelity in `quantization_report.json`. Runtime packaging
also records component dispositions in `export_report.json` and compatibility status
in `runtime_compatibility.json`.

`export_status` describes whether requested package components were emitted.
`runtime_validation_status` independently reports whether the exact artifact,
tokenizer, runtime version, and final package bytes match structured evidence.
`unvalidated` is not a claim that the graph is invalid.

## Fail-closed outcomes

- Unsupported architecture semantics, tensor mappings, malformed or corrupt input,
  and source-identity mismatches fail before durable output.
- A quantized tensor without a supported import route fails instead of silently
  becoming float; use `--dequantize` or `keep_quantized=False` when appropriate.
- An authoritative tokenizer blocker preserves a valid model export, emits a
  structured warning, omits unverified tokenizer assets, and records a partial
  component disposition in `export_report.json`.
- Missing exact runtime evidence preserves an accurate model package with
  `runtime_validation_status: unvalidated`; it does not upgrade the package to a
  validated runtime claim.

In the generated catalog, `SUPPORTED` means the named capability is implemented and
mechanically tested, `DEFERRED` means it is intentionally unavailable pending the
stated work, and `REJECTED` means the input or route is invalid by policy.
