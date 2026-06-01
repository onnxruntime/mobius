---
name: mobius-onnx-export-gotchas
description: Use when building/exporting ONNX models with the `mobius build` CLI (especially Phi-3 / Phi-3.5 or any model with `--execution-provider cuda` GQA fusion and/or `--static-cache`). Covers the current CLI syntax, the dtype flag values, the GQA-vs-static-cache interaction, and a known fp16 packed-QKV FLOAT32 bug that makes GQA fp16 exports fail to load in onnxruntime.
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

## 3. BUG: GQA fp16 export leaves packed-QKV weights as FLOAT32 → model won't load
For an fp16 GQA export, the per-layer packed QKV weight
(`..q_proj.weight__k_proj.weight__v_proj.weight__axis_0__concat`) is emitted as **FLOAT32**, while its
MatMul's other input is fp16. onnxruntime then rejects the model at load:

```
Type Error: Type parameter (T) of Optype (MatMul) bound to different types
(tensor(float16) and tensor(float)) in node (node_MatMul_*)
```

You'll also see at save time: `The value type for shape [H, 3H] is not known. Skipping serialization`.

**Root cause:** `_cast_module_dtype` (`src/mobius/_builder.py:84`) casts module params to fp16 *before*
graph build. The GQA `PackQKVWithBias` rewrite (`src/mobius/rewrite_rules/_group_query_attention.py`)
then emits the packed weight as a graph-level `op.Concat(q_w,k_w,v_w)` that a constant-fold collapses
into a NEW initializer whose dtype is FLOAT32/untyped — the fp16 cast never reaches it.

### Detect
```python
import onnx
m = onnx.load("model.onnx", load_external_data=False)
fp32 = [i.name for i in m.graph.initializer if i.data_type == onnx.TensorProto.FLOAT]
print(len(fp32), "FLOAT32 initializers (should be 0 for fp16)")
```

### Fix (post-export, numerically == intended fp16)
Cast the FLOAT32 initializers to fp16, optionally strip dead pre-pack q/k/v initializers, re-save.
**Gotcha when re-saving with external data:** if you save with `location="X.data"` and then rename the
file, the references inside `model.onnx` still point to `X.data`. Either save directly with
`location="model.onnx.data"`, or rewrite each initializer's `external_data` `location` entry.

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

**Proper upstream fix:** set the packed-Concat output type to the model dtype in the GQA pack rewrite,
or cast ALL float initializers at save time regardless of registered value_info; add an e2e test that
loads the fp16 GQA export in onnxruntime.

## 4. Always validate the export in ORT before profiling
Load the model on `CUDAExecutionProvider` and run one prefill + one decode `session.run`. Confirm:
(a) the expected attention op (`com.microsoft::GroupQueryAttention` vs `ai.onnx::Attention`),
(b) finite fp16 logits, (c) no FLOAT32 initializers for an fp16 build.
