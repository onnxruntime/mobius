---
name: mobius-onnx-export-gotchas
description: Use when building/exporting ONNX models with the `mobius build` CLI (especially Phi-3 / Phi-3.5 or any model with `--execution-provider cuda` GQA fusion and/or `--static-cache`). Covers the current CLI syntax, the dtype flag values, the GQA-vs-static-cache interaction, and how to verify fp16 GQA exports load in onnxruntime (the historical packed-QKV FLOAT32 load bug is fixed as of df203cc).
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
