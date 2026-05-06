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

## ORT CUDA Attention Kernel Dispatch Reference

ORT selects attention kernels via a cascade — the first matching kernel
wins. Understanding the dispatch rules helps diagnose why a model uses
a slower kernel than expected.

### Contrib MultiHeadAttention (`com.microsoft`)

Cascade: LeanAttention → Flash → cuDNN SDPA → TRT FusedCross →
TRT FusedRunner → MEA → Unfused

| Kernel | Required conditions |
|--------|---------------------|
| LeanAttention | `USE_LEAN_ATTENTION` build flag, `seq_len==1`, `past_seq>0`, no bias, no padding mask, `head_size==v_head_size` |
| Flash | No bias, no padding mask, no `past_seq`, no `cache_indirection`, `head_size==v_head_size`, fp16/bf16, SM≥8.0 |
| cuDNN SDPA | `enable_cudnn_flash_attention_`, mask NONE or 1D_KEY_SEQ_LEN |
| TRT FusedCross | NOT unidirectional, no padding/bias/past, `hidden==v_hidden` |
| TRT FusedRunner | NOT unidirectional, no bias, mask none or 1D, `seq_len==kv_seq_len` |
| MEA (CUTLASS) | Long sequence, bias alignment OK (null or `seq % 4*sizeof(T) == 0`), no past/cache |
| Unfused | Always available (fallback) |

### Contrib GroupQueryAttention (`com.microsoft`)

Cascade: XQA → Flash → MEA → Unfused. **Rejects `attention_bias`
entirely** — if bias is provided, GQA falls back immediately.

| Kernel | Required conditions |
|--------|---------------------|
| XQA | SM≥8.0, `seq==1`, `past_present_share_buffer`, `softcap==0`, `local_window==-1`, `head_size ∈ {64, 128, 256}` |
| Flash | fp16/bf16, SM≥8.0. FastDecode: `seq==1`, `past_present_share_buffer`, no KV quant |
| MEA | No bias (rejected upstream), head_size check |
| Unfused | Fallback |

### ONNX Attention — MHA (`q_num_heads == kv_num_heads`)

Cascade: Flash → MEA → Unified Unfused

| Kernel | Required conditions |
|--------|---------------------|
| Flash | fp16/bf16, `head_size ≤ 256`, `head_size == v_head_size`, `attn_mask == nullptr`, SM≥8.0 |
| MEA | `head_size ≤ 1024` & `% 8 == 0`, if mask then `total_seq % 4 == 0`, if `past_key` then `head_size == v_head_size` |
| Unfused | Always available |

### ONNX Attention — GQA flavor (`q_num_heads != kv_num_heads`)

Same cascade as MHA, with extra MEA constraints:

| Kernel | Required conditions |
|--------|---------------------|
| Flash | Same as MHA |
| MEA | MHA conditions + `head_size == v_head_size` + not float32 |
| Unfused | Always available, handles GQA via in-kernel reshape |

### Key takeaways

- **Flash requires `attn_mask == nullptr`** — any explicit mask
  (bool or float) disables Flash and falls to MEA or unfused.
- **`nonpad_kv_seqlens`** enables Flash with variable-length sequences
  without providing an explicit mask.
- **GQA contrib op rejects `attention_bias`** — use the standard ONNX
  `Attention` op if you need bias with GQA.
- **SM≥8.0** (Ampere+) is required for Flash on all paths.

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
fix patterns. The most impactful fix is usually lowering opset version
or replacing unsupported ops with CUDA-friendly alternatives.

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

### Real example: Gemma4-e2b on H200

| Metric | Value |
|--------|-------|
| GenAI CUDA decode | **158 tok/s** |
| ORT CUDA decode | 151.7 tok/s |
| GenAI CPU decode | 11.8 tok/s |
| VRAM usage | 12.2 GB |
| cuBLAS warmup | ~40 ms (first step only) |
| Steady-state MatMul | ~66 µs |

**Bandwidth analysis:**
| Metric | Value |
|--------|-------|
| Theoretical max (H200) | 507 tok/s |
| Achieved | 158 tok/s |
| Bandwidth utilization | **31%** |
| Primary bottleneck | Kernel launch overhead |

The 31% utilization gap is dominated by kernel launch overhead — many
small CUDA kernels between MatMuls add latency that isn't spent on
memory or compute. Fusing more ops (e.g. skip-norm, activation)
would close this gap.

## Debugging workflow

1. **Profile the model** with ORT session profiling
2. **Check memcpy count** — if >10, investigate CPU-placed ops
3. **Check attention kernel** — verify GQA/Flash is being used
4. **Group by op type** — find the top time consumers
5. **Compare prefill vs decode** — decode should be much faster
6. **Measure GenAI throughput** — tok/s is the user-facing metric
7. **Check GPU utilization** — `nvidia-smi` during inference

## Cross-references

- **Debugging memcpy:** `.agents/skills/debugging-memcpy/SKILL.md`
- **Building ORT with CUDA:** `.agents/skills/building-ort-genai/SKILL.md`
- **ONNX export:** `.agents/skills/onnx-export-quantization/SKILL.md`
- **Reusable components:** `.agents/skills/reusable-components/SKILL.md`
