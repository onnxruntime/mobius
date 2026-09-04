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
| `keep_quantized` | Request quantized target storage where the selected graph and qtype route support it. This does not promise source-byte or numerical fidelity. |
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
