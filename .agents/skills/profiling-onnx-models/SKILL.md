---
name: profiling-onnx-models
description: >
  Use this skill when profiling ONNX model performance on CUDA with
  OnnxRuntime. Covers ORT session profiling, reading JSON results,
  identifying compute and memory bottlenecks, debugging memcpy nodes,
  and measuring GenAI pipeline throughput (tok/s, TTFT).
---

# Skill: Profiling ONNX Models on CUDA

## When to use

Use this skill when:
- Measuring ONNX model latency on GPU
- Identifying compute vs memory bottlenecks
- Debugging why inference is slower than expected
- Investigating memcpy nodes (CPU↔GPU transfers)
- Measuring GenAI pipeline throughput (tokens/second, TTFT)
- Comparing fused vs unfused attention kernels

## ORT session profiling

### Enable profiling

```python
import onnxruntime as ort
import json

opts = ort.SessionOptions()
opts.enable_profiling = True

sess = ort.InferenceSession(
    "model.onnx",
    opts,
    providers=["CUDAExecutionProvider"],
)

# Run inference (warmup + measured runs)
for _ in range(3):  # warmup
    sess.run(None, inputs)

for _ in range(10):  # measured
    sess.run(None, inputs)

# End profiling and get output file
prof_file = sess.end_profiling()
print(f"Profile saved to: {prof_file}")
```

### Profile output format

The profile is a JSON file with an array of trace events:

```json
[
  {
    "cat": "Node",
    "name": "MatMul_42",
    "dur": 156,
    "args": {
      "op_name": "MatMul",
      "provider": "CUDAExecutionProvider",
      "input_type_shape": [{"float": [1, 128, 4096]}],
      "output_type_shape": [{"float": [1, 128, 14336]}]
    }
  }
]
```

Key fields:

| Field | Description |
|-------|-------------|
| `name` | Node name in the ONNX graph |
| `dur` | Duration in microseconds |
| `cat` | Category — `"Node"` for ops, `"Kernel"` for CUDA kernels |
| `args.op_name` | ONNX op type (MatMul, Attention, etc.) |
| `args.provider` | Execution provider (CUDA vs CPU) |

## Reading profile results

### Parse and analyze by op type

```python
import json
from collections import defaultdict

with open(prof_file) as f:
    events = json.load(f)

# Filter to node events only
nodes = [e for e in events if e.get("cat") == "Node"]

# Group by op type
op_times = defaultdict(list)
for n in nodes:
    op = n["args"].get("op_name", n["name"])
    op_times[op].append(n["dur"])

# Summary: total time per op type, sorted
print(f"{'Op Type':<30} {'Count':>6} {'Total (ms)':>12} {'Avg (µs)':>10}")
print("-" * 62)
for op, times in sorted(op_times.items(), key=lambda x: -sum(x[1])):
    total_ms = sum(times) / 1000
    avg_us = sum(times) / len(times)
    print(f"{op:<30} {len(times):>6} {total_ms:>12.2f} {avg_us:>10.1f}")
```

### Check node placement (CPU vs CUDA)

```python
cpu_nodes = [n for n in nodes
             if n["args"].get("provider") == "CPUExecutionProvider"]
cuda_nodes = [n for n in nodes
              if n["args"].get("provider") == "CUDAExecutionProvider"]

print(f"CUDA nodes: {len(cuda_nodes)}")
print(f"CPU nodes:  {len(cpu_nodes)}")

if cpu_nodes:
    print("\nCPU-placed ops (may cause memcpy):")
    for n in cpu_nodes:
        print(f"  {n['args'].get('op_name', '?')}: {n['name']}")
```

### Identify memcpy overhead

```python
memcpy_events = [n for n in nodes
                 if "Memcpy" in n.get("name", "")]
total_memcpy_us = sum(n["dur"] for n in memcpy_events)
total_compute_us = sum(n["dur"] for n in nodes)

print(f"Memcpy: {total_memcpy_us/1000:.2f} ms "
      f"({100*total_memcpy_us/total_compute_us:.1f}% of total)")
```

## Identifying bottlenecks

### Compute-bound ops

| Op | Typical role | What to check |
|----|-------------|---------------|
| `MatMul` / `Gemm` | QKV projections, FFN up/down/gate | Should dominate profile for large models. Verify cuBLAS is used. |
| `Attention` | Self-attention | Check which kernel: Flash, GQA, MEA, or unfused. Fused is 3-7x faster. |
| `com.microsoft.GroupQueryAttention` | Fused GQA | Best for GQA models. Check `head_dim` ≤ 256 for Flash path. |
| `Conv` | Vision encoder, conv layers | Should be on CUDA. CPU fallback is very slow. |

### Memory-bound ops

| Op | Typical role | What to check |
|----|-------------|---------------|
| `MemcpyFromHost` / `MemcpyToHost` | CPU↔GPU transfer | Each one is a sync point. 280+ means serious issue. |
| `Transpose` | Layout conversion | Should be fused into adjacent ops where possible. |
| `Reshape` / `Squeeze` / `Unsqueeze` | Shape manipulation | Zero-copy on CUDA (metadata only). Non-zero time = problem. |

### Attention kernel hierarchy

From fastest to slowest for GQA models:

1. **Flash Attention** (via GQA op) — requires `head_dim` ≤ 256
2. **Memory-Efficient Attention (MEA)** — fallback when Flash unavailable
3. **Unfused Attention** — standard `Attention` op decomposed to MatMul + Softmax + MatMul

**Real example (Gemma4, head_dim=256):**
- GQA with Flash: ~66 µs per decode step
- Unfused attention: ~450 µs (6.8x slower)

## Common CUDA performance issues

### 1. Excessive memcpy nodes (280+)

**Symptom:** Hundreds of `MemcpyFromHost`/`MemcpyToHost` nodes in the
profile, each adding latency and forcing GPU sync.

**Cause:** Ops that don't have a CUDA kernel fall to CPU, requiring
data transfer. Common culprits:
- Opset version mismatch (e.g. opset 24 ops not yet in ORT CUDA EP)
- Dynamic `Shape` + `Gather` patterns
- `ConstantOfShape` with unusual dtypes

**Fix:** See the `debugging-memcpy` skill for root-cause analysis and
fix patterns. The correct fix is to update ORT kernel registration for
the missing opset versions (e.g. Equal opset 19 CUDA gap). As a
secondary option, replace unsupported ops with CUDA-friendly alternatives.

### 2. cuBLAS warmup spike

**Symptom:** First decode step is 40-100ms, subsequent steps are
~66 µs.

**Cause:** cuBLAS initializes and JIT-compiles kernels on first use.
This is a one-time cost per session.

**Fix:** Not a bug — this is expected CUDA behavior. Exclude the
first inference call from benchmarks. For latency-sensitive
applications, run a warmup inference before serving.

### 3. head_dim > 256 breaks Flash Attention

**Symptom:** Attention is 3-7x slower than expected. Profile shows
unfused attention ops instead of `GroupQueryAttention` with Flash.

**Cause:** Flash Attention v2 requires `head_dim` ≤ 256. Models with
larger head dimensions (e.g. Gemma4 global attention with
`head_dim=512`) fall back to unfused or MEA attention.

**Fix:** This is a hardware/library limitation. Options:
- Use the `--ep default` (CPU) build which doesn't need Flash
- Wait for Flash Attention v3 / updated ORT with larger head_dim support
- For Gemma4 specifically, the GQA bypass was removed so models use
  standard attention ops with runtime kernel selection

**Gemma4 Flash Attention note:** Flash Attention gives only 1-3%
benefit for Gemma4. With float attention masks (KV-shared layers),
only 7/35 layers are eligible for Flash. With all-GQA (`onnxruntime/mobius#279`),
25/35 layers become eligible but the `head_dim=512` global layers
still cannot use Flash. The performance impact is minimal because
global attention layers dominate compute.

### 4. Batch size=1 on large GPUs

**Symptom:** GPU utilization is low (10-30%). Throughput doesn't
improve with larger GPU.

**Cause:** Single-token decode with batch=1 is memory-bandwidth bound,
not compute bound. Large GPUs (H100, H200) have excess compute for
small batch sizes.

**Fix:** Increase batch size if possible, or use continuous batching
(vLLM-style). For single-user latency, smaller GPUs may be more
cost-effective.

## Profiling GenAI pipeline

### Measure tokens per second

```python
import time
import onnxruntime_genai as og

model = og.Model("model_dir/")
tokenizer = og.Tokenizer(model)

prompt = "Explain quantum computing in simple terms."
input_ids = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=200, do_sample=False)

gen = og.Generator(model, params)
gen.append_tokens(input_ids)

# Measure TTFT (time to first token)
t0 = time.perf_counter()
gen.generate_next_token()
ttft = time.perf_counter() - t0

# Measure decode throughput
t1 = time.perf_counter()
num_tokens = 0
while not gen.is_done():
    gen.generate_next_token()
    num_tokens += 1
decode_time = time.perf_counter() - t1

print(f"TTFT: {ttft*1000:.1f} ms")
print(f"Decode: {num_tokens} tokens in {decode_time:.2f}s")
print(f"Throughput: {num_tokens/decode_time:.1f} tok/s")
```

### Key metrics

| Metric | Description | Typical values |
|--------|-------------|----------------|
| **TTFT** | Time to first token (prefill) | 50-500 ms (depends on prompt length) |
| **Decode tok/s** | Tokens per second during generation | 10-100+ tok/s (depends on model size, GPU) |
| **Prefill tok/s** | Prompt processing throughput | 500-5000 tok/s |

### Real example: Gemma4 on H200

| Metric | Value |
|--------|-------|
| TTFT (short prompt) | ~85 ms |
| Decode throughput | ~12-15 tok/s (CPU), ~60+ tok/s (CUDA) |
| cuBLAS warmup | ~40 ms (first step only) |
| Steady-state decode | ~66 µs per MatMul |

## Benchmark methodology

### Benchmark dimensions

A thorough benchmark varies these dimensions independently:

#### 1. Pipeline (inference framework)

| Pipeline | Description |
|----------|-------------|
| ORT standalone CUDA | Raw ORT `InferenceSession` with CUDA EP |
| GenAI CUDA | Full `onnxruntime-genai` pipeline with CUDA |
| GenAI CPU | Full `onnxruntime-genai` pipeline, CPU only |
| HF PyTorch CUDA | HuggingFace `transformers` baseline (GPU) |
| HF PyTorch CPU | HuggingFace `transformers` baseline (CPU) |

#### 2. Model dtype

| Dtype | Notes |
|-------|-------|
| F16 | Default for GPU inference |
| BF16 | Better numerical range, Ampere+ GPUs |
| F32 | CPU inference, reference baseline |
| Q4_K_M | INT4 k-quant (Olive) |
| NF4 | INT4 NormalFloat (Olive) |

#### 3. Execution provider variant

| EP | Attention kernel | Notes |
|----|-----------------|-------|
| `default` | ONNX Attention → MEA | Portable, no vendor fusions |
| `cuda` | GQA rewrite → hybrid GQA/Attention | Best CUDA perf for simple models |
| `onnx-standard` | Inlined functions → standard ops | Strict ONNX-only |

#### 4. Input modality

| Modality | Example |
|----------|---------|
| Text only | Standard LLM prompt |
| Image + text | VLM with image input |
| Audio + text | ALM with speech input |
| Image + audio + text | Multimodal (e.g. Gemma4) |

#### 5. Metrics

| Metric | Description |
|--------|-------------|
| **Decode tok/s** | Tokens per second AFTER first token |
| **TTFT** | Time to first token — includes encoder + prefill |
| **VRAM** | GPU memory usage (`nvidia-smi`) |
| **Model size** | On-disk size of ONNX + weights |

#### 6. Additional dimensions (for thorough studies)

| Dimension | Values |
|-----------|--------|
| Cache type | Static cache vs dynamic cache |
| Prompt length | Short (20), medium (500), long (2000+ tokens) |
| Batch size | 1, 4, 8 |
| Padding strategy | `nonpad_kv_seqlens` vs `attention_mask` |
| KV buffer | `past_present_share_buffer`: true vs false |

### Best practices

- **One session, one build:** Run ALL benchmarks in a single session
  with the SAME ORT/GenAI build to ensure comparability.
- **Warm up:** Run 2-3 inference calls before timing to amortize
  cuBLAS JIT compilation and CUDA context setup.
- **Record the build:** Report which ORT/GenAI commit or version was
  used — results are not reproducible across builds.
- **Separate TTFT from decode:** TTFT includes encoder + prefill
  overhead. Decode throughput measures steady-state generation.
- **Sufficient output length:** Generate 50+ tokens to amortize TTFT
  and get stable decode throughput numbers.
- **Control for GPU state:** Avoid running other GPU workloads during
  benchmarks. Check `nvidia-smi` for baseline memory usage.

## Debugging workflow

1. **Profile the model** with ORT session profiling
2. **Check memcpy count** — if >10, investigate CPU-placed ops
3. **Check attention kernel** — verify GQA/Flash is being used
4. **Group by op type** — find the top time consumers
5. **Compare prefill vs decode** — decode should be much faster
6. **Measure GenAI throughput** — tok/s is the user-facing metric
7. **Check GPU utilization** — `nvidia-smi` during inference

## Quantization benchmark reference (Gemma4 E2B-IT)

Measured with Gemma4 E2B-IT on GenAI (text-only, 50+ token decode):

| Dtype | CUDA tok/s | CPU tok/s | Notes |
|-------|-----------|----------|-------|
| F16 | 104 | 6.8 | Baseline |
| Q4_K_M | 111 (+7%) | 16.2 (+138%) | **Recommended** — best speed, no observed quality regression in spot checks |
| NF4 | 93 (-11%) | 4.0 (-41%) | Slower than F16 on both EPs |

**Recommendation:** Q4_K_M wins on both CPU and CUDA with no observed
quality regression in spot checks. NF4 is slower than even F16 in this
measurement and should be avoided for inference speed. Use Q4_K_M as
the default quantization format.

## Cross-references

- **Debugging memcpy:** `.agents/skills/debugging-memcpy/SKILL.md`
- **Building ORT with CUDA:** `.agents/skills/building-ort-genai/SKILL.md`
- **ONNX export:** `.agents/skills/onnx-export-quantization/SKILL.md`
- **Reusable components:** `.agents/skills/reusable-components/SKILL.md`
