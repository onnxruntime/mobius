---
name: mobius-onnx-export-gotchas
description: Use when building/exporting ONNX models with the `mobius build` CLI (especially Phi-3 / Phi-3.5 or any model with `--execution-provider cuda` GQA fusion and/or `--static-cache`). Covers the current CLI syntax, the dtype flag values, the GQA-vs-static-cache interaction, how to verify fp16 GQA exports load in onnxruntime (the historical packed-QKV FLOAT32 load bug is fixed as of df203cc), and why fp16 GQA exports need VALUE-based weight checks (corr≈1.0 / norm), not just initializer count/dtype, to catch silently-zeroed packed-QKV weights.
---

# mobius ONNX export gotchas

## 1. CLI syntax (editable repo differs from older docs)
`mobius build` requires `--model <hf_id>` and takes the **output dir as a POSITIONAL** arg.
There is **no `-o` flag for `build`** (`-o` exists only on `build-gguf`).

```bash
mobius build --model microsoft/Phi-3.5-mini-instruct \
  --dtype f16 --execution-provider cuda \
  --external-data onnx --trust-remote-code \
  /path/to/output_dir
```

- `--dtype` choices: `f16`/`float16`, `bf16`/`bfloat16`, `f32`/`float32`. **`fp16` is INVALID.**
- `--execution-provider` is an alias of `--ep`. `cuda` + fp16/bf16 triggers GQA fusion;
  `default` keeps plain ONNX `Attention`.

## 2. `--static-cache` is incompatible with GQA fusion
`--static-cache` wraps each attention with `TensorScatter` (in-place KV cache for the **ONNX Attention**
op). That breaks the pattern the GQA rewrite matches, so combining
`--execution-provider cuda --static-cache` yields **0 GroupQueryAttention + N Attention + 2N TensorScatter**
(mobius prints: "GQA fusion expected … but found 0 GroupQueryAttention and N Attention nodes").

- **GQA model:** `--execution-provider cuda` **alone**. GQA's shared KV buffer
  (`past_present_share_buffer`) is enabled at **runtime** via IO-binding past & present to the same
  OrtValue — NOT via `--static-cache`.
- **ONNX-Attention + in-place cache:** `--execution-provider default --static-cache --max-seq-len N`.

## 3. FIXED: fp16 GQA export previously left packed-QKV weights as FLOAT32 → model wouldn't load
**Status: fixed as of commit `df203cc`.** Native fp16 Phi-3.5 GQA export now loads directly in the ORT
CUDA EP with **no manual post-cast** (32 GroupQueryAttention nodes, all-fp16 initializers). If you are on
that commit or later, you should not hit this — skip to the verification snippet below. The history is
kept here because old artifacts exported before the fix still carry fp32 packed weights.

### Symptom (pre-fix)
For an fp16 GQA export, a folded per-layer packed QKV weight
(`..q_proj.weight__k_proj.weight__v_proj.weight__axis_0__concat`) was emitted as **FLOAT32**, while its
MatMul's other input was fp16. onnxruntime then rejected the model at load on both CPU and CUDA EPs:

```
Type Error: Type parameter (T) of Optype (MatMul) bound to different types
(tensor(float16) and tensor(float)) in node (node_MatMul_*)
```

You'd also see at save time: `The value type for shape [H, 3H] is not known. Skipping serialization`.

### Root cause
`_cast_module_dtype` casts module params to fp16, but the resulting initializer `Value`s lose their
declared `.dtype` (it becomes `None`) while their `const_value` stays fp16. The fold passes
`FoldConcatInitializersPass` (`src/mobius/_passes/_fold_concat.py`) and `FoldTransposedInitializerPass`
(`src/mobius/_passes/_fold_transpose.py`) then defaulted the folded initializer's dtype to `FLOAT`,
serializing the packed QKV / transposed weights as fp32.

### The fix
A shared helper `initializer_dtype()` (`src/mobius/_passes/_dtype_utils.py`) resolves the effective dtype
from the declared type, **falling back to `const_value` when the type annotation was dropped** (preferring
the data dtype and warning on stale-metadata disagreement). Both fold passes use it to stamp the correct
dtype on the new initializer's `TensorType` and `LazyTensor`, and `FoldConcatInitializersPass` now also
skips folding before weights are loaded (mirroring `FoldTransposedInitializerPass`). A regression test
loads the fp16 GQA export in the ORT CPU EP to lock this in.

### Verify (still worth running on any fp16 build)
```python
import onnx
m = onnx.load("model.onnx", load_external_data=False)
fp32 = [i.name for i in m.graph.initializer if i.data_type == onnx.TensorProto.FLOAT]
print(len(fp32), "FLOAT32 initializers (should be 0 for fp16)")
```

### Salvaging a stale pre-fix artifact (only if re-exporting is not an option)
Prefer re-exporting on the fixed code. If you must repair an old model, cast its FLOAT32 initializers to
fp16 and re-save. **Gotcha when re-saving with external data:** if you save with `location="X.data"` and
then rename the file, the references inside `model.onnx` still point to `X.data`. Either save directly
with `location="model.onnx.data"`, or rewrite each initializer's `external_data` `location` entry.

```python
import onnx, numpy as np
from onnx import numpy_helper, TensorProto
m = onnx.load("model.onnx", load_external_data=True)
for init in m.graph.initializer:
    if init.data_type == TensorProto.FLOAT:
        arr = numpy_helper.to_array(init).astype(np.float16)
        init.CopyFrom(numpy_helper.from_array(arr, init.name))
onnx.save(m, "model.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
          location="model.onnx.data", size_threshold=1024, convert_attribute=False)
```

## 4. Always validate the export in ORT before profiling
Load the model on `CUDAExecutionProvider` and run one prefill + one decode `session.run`. Confirm:
(a) the expected attention op (`com.microsoft::GroupQueryAttention` vs `ai.onnx::Attention`),
(b) finite fp16 logits, (c) no FLOAT32 initializers for an fp16 build.

These checks are **necessary but NOT sufficient** for a fp16 GQA export — see §5. A model can pass all
three and still have silently-zeroed packed-QKV weights.

## 5. Verifying a fp16 GQA export: use VALUE-based weight checks, NOT initializer count/dtype
**A fp16 GQA export can be all-fp16, right-count, and still all-zeros — only a corr≈1.0 / norm≈126 VALUE
check on the packed QKV proves the weights are real.**

### Symptom
The GQA model loads cleanly (32 `GroupQueryAttention` nodes, all-fp16, finite logits) but generates
garbage (e.g. `holdou_(...artersarters`). Prefill logits come out ~3× the reference scale, with
`max|Δlogit|` ~50+ versus the reference.

### Root cause
The packed-QKV initializer is `Transpose(Concat(q, k, v, axis=0))`. If the fold passes
(`FoldConcatInitializersPass` / `FoldTransposedInitializerPass`) leave the packed-Concat output dtype
UNKNOWN / defaulted-to-fp32 while the data is fp16, the serializer **skips** it and it loads as
**near-zero** — the weights are silently dead. (This is the §3 failure mode; the upstream fix in
`df203cc` stamps the fp16 dtype at the fold-pass source. A post-hoc cast is NOT a fix — it re-corrupts.)

### Why count/dtype checks fail (the trap)
The BROKEN export and the FIXED export can have the **same initializer count and the same fp16/fp32 dtype
ratio** (e.g. both 197 fp16 after dead-weight stripping). Counting initializers or checking
"0 fp32 / all fp16" does **not** distinguish a healthy model from a zeroed-weight one. §4(c) alone will
pass a dead model.

### Canonical verification (load-bearing, not optional)
VALUE-based per-slice check on each packed-QKV initializer against its source q/k/v weights:
- per-slice correlation **≈ 1.000** (broken ≈ 0.000), AND
- packed-QKV L2 norm **≈ 126.6** at layer 0 / mean(|abs|) **≈ 0.013** (broken ≈ 0.80 / ≈ 5e-6).

Plus an end-to-end next-token greedy-argmax parity check vs the `attn_dynamic` reference (expect
**~19–20 / 20**). Isolated single-token divergences are fp16 dead-ties (reference top1−top2 gap = 0.0000),
not bugs. Optional hardening: assert **0 unused initializers** and that all N packed-QKV initializers are
present, to catch dead-weight OVER-stripping.

QA's `gqa_weight_integrity_gate.py` (`--self-check --strip-audit --scan-all`, per-layer corr/norm)
implements exactly this gate.

## 6. FIXED: GQA `present.*` KV-cache outputs declared the wrong `head_dim`
**Status: fixed as of commit `cf6c5c4`.** A native fp16 GQA export now declares
`present.{i}.{key,value}` with the correct `head_dim`, symmetric to its `past_key_values.{i}.*` inputs.

### Symptom (pre-fix)
The graph **output** `present.{i}.key/value` declared the wrong `head_dim` (e.g. `32` instead of the real
`96` on Phi-3.5) while the matching `past_key_values.{i}.*` **input** was correct (`96`). At load ORT logged
(once per key+value per layer — 64 on Phi-3.5):

```
[W ...MergeShapeInfo] Error merging shape info for output. 'present.0.key'
source:{-1,32,-1,96} target:{-1,32,-1,32}. Falling back to lenient merge.
```

Runtime still produced correct (96-wide) arrays via lenient merge, but any consumer that **trusts declared
shapes** (e.g. `onnxruntime-genai`) would see inconsistent past-vs-present KV cache types.

### Root cause
`GroupQueryAttention`'s contrib-op shape inference mis-derives the present `head_dim` (it does **not**
reproduce on the plain `Attention` op, which infers correctly). `_register_kv_cache_outputs`
(`src/mobius/tasks/_cache_utils.py`) added the present outputs with **no explicit shape**, so the buggy
inference won.

### The fix
`_register_kv_cache_outputs` now opt-in **stamps** `present.{i}.{key,value}` shape+dtype symmetric to the
past inputs when the caller passes `batch`/`num_kv_heads`/`key_head_dim`/`value_head_dim`/`total_seq_len`/
`dtype` (wired from `_causal_lm.py`). Omitting them preserves inference-only behavior, so the other ~10
callers are unaffected. The stamp survives `SymbolicShapeInferencePass` (policy `refine` only tightens
unknown dims; it won't replace a concrete `96` with a conflicting `32`).

### Verify
```python
import onnx
m = onnx.load("model.onnx", load_external_data=False)
d = lambda vi: [(x.dim_param or x.dim_value) for x in vi.type.tensor_type.shape.dim]
o = {v.name: v for v in m.graph.output}
print("present.0.key:", d(o["present.0.key"]))   # head_dim must equal the past input's (e.g. 96, NOT 32)
```

### Known remaining (separate, pre-existing, harmless)
ORT still logs ~32 `Error merging shape info ... source:{-1,-1,3072} target:{-1,-1,1024}` warnings on the
GQA op's **internal hidden-state output** value_info (`v_*.GroupQueryAttention_*_0`, `1024`=32×32 vs the
correct `3072`=32×96). That value is **not** a declared graph I/O — runtime is correct and `onnxruntime-genai`
does not trust it — so it does not bite shape-trusting consumers the way the present-output bug did. Tracked
as a follow-up in the GQA rewrite emission path (not the KV-cache output path).
