# Debugging ORT CUDA EP Scan Shape Mismatches

## When to use this skill

Use this when you see a **shape mismatch error in a Scan node** that
only occurs on CUDA EP (CPU works fine). The typical error message:

```
Shape mismatch attempting to re-use buffer. {1,16,1,1} != {1,16,128,1}
```

in a node like `_inlfunc_<FunctionName>_Scan_node_<N>`.

## Symptom pattern

- Error mentions **"re-use buffer"** — this is the CUDA EP memory planner
  trying to reuse the carry state buffer between Scan iterations.
- The initial carry shape has **unexpected 1s** where concrete values
  should appear (e.g. `{1,16,1,1}` instead of `{1,16,128,128}`).
- The Scan body output shape is **partially correct** — some dims are
  resolved, others are still 1.
- The node name contains `_inlfunc_` — the Scan is inside an **inlined
  function** (ir.Function).
- **CPU EP works fine** with the same model and inputs.

## Root cause

**ORT CUDA EP fails to resolve symbolic `dim_param` annotations in Scan
body subgraphs when the Scan is inside an inlined `ir.Function`.**

The failure chain:

1. An `ir.Function` (e.g. `LinearAttention`) contains a Scan op.
2. The Scan body declares carry state inputs with symbolic dim_params:
   `shape=[B, H, d_k, d_v]` where `B`, `H`, `d_k`, `d_v` are all
   `dim_param` strings.
3. ORT inlines the function at load time (prefix: `_inlfunc_`).
4. CUDA EP's memory planner needs to allocate carry buffers **before**
   running the Scan.
5. It reads the body's input shape annotations to determine buffer size.
6. After function inlining, the symbolic dims are **not properly
   resolved** from the actual input tensors.
7. Unresolved `dim_param` values fall back to `dim_value=0`, which the
   buffer allocator treats as **1**.
8. The allocated buffer shape (e.g. `{1,16,1,1}`) does not match what
   the body actually produces (e.g. `{1,16,128,1}`).

CPU EP doesn't hit this because it allocates lazily from actual tensor
shapes rather than from annotations.

## Diagnosis process

### Step 1: Identify the Scan node and its context

The error message tells you the node name. Look for `_inlfunc_` prefix —
this confirms the Scan is inside an inlined function.

```
_inlfunc_LinearAttention_Scan_node_23
         ^^^^^^^^^^^^^^^^ function name
                          ^^^^^^^^^^^^^ node within the function
```

### Step 2: Check the Scan body input shapes

Find the function definition and inspect its Scan body:

```python
import onnx_ir as ir

# Load or build the model
model = ir.load("model.onnx")

# Find the function containing the Scan
for fid, func in model.functions.items():
    for node in func.graph:
        if node.op_type == "Scan":
            body = node.attributes["body"].value
            print("Body inputs:")
            for v in body.inputs:
                print(f"  {v.name}: shape={v.shape}")
```

If you see **all symbolic dims** (dim_params like `B`, `H`, `d_k`,
`d_v`), that's the problem.

### Step 3: Determine which dims are actually static

Check the function's construction parameters. For LinearAttention:
- `B` (batch) — dynamic ✓ (should stay symbolic)
- `H` (num_heads) — **static** (known at build time from config)
- `d_k` (key head dim) — **static** (from config)
- `d_v` (value head dim) — **static** (from config)

### Step 4: Verify with CPU EP first

Run the same model on CPU to confirm the shapes are correct when
symbolic dim resolution works:

```python
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
outputs = session.run(None, inputs)  # Should succeed
```

## Fix pattern

**Replace symbolic dim_params with concrete `int` values for any
dimension known at graph build time.** Only truly dynamic dimensions
(batch size, sequence length) should remain symbolic.

### Before (broken on CUDA EP)

```python
state_in = ir.Value(
    name="state",
    shape=ir.Shape([
        ir.SymbolicDim("B"),   # symbolic — OK (truly dynamic)
        ir.SymbolicDim("H"),   # symbolic — BAD (known at build time)
        ir.SymbolicDim("d_k"), # symbolic — BAD (known at build time)
        ir.SymbolicDim("d_v"), # symbolic — BAD (known at build time)
    ]),
    type=ir.TensorType(stash_type),
)
```

### After (works on both CPU and CUDA EP)

```python
state_in = ir.Value(
    name="state",
    shape=ir.Shape([
        ir.SymbolicDim("B"),  # symbolic — truly dynamic
        kv_num_heads,         # concrete int (e.g. 16)
        head_k_dim,           # concrete int (e.g. 128)
        head_v_dim,           # concrete int (e.g. 128)
    ]),
    type=ir.TensorType(stash_type),
)
```

### What to change

1. **Scan body input shapes**: Replace symbolic dims with concrete
   values for all architecture constants. You only need to fix the
   body *inputs* — shape inference propagates through the body ops.

2. **Function builder signature**: Add the concrete dimension values
   as parameters so the body builder can use them:

   ```python
   def _build_recurrence_body(
       uses_decay: bool,
       uses_beta: bool,
       *,
       kv_num_heads: int,  # NEW
       head_k_dim: int,    # NEW
       head_v_dim: int,    # NEW
       stash_type: ir.DataType = ir.DataType.FLOAT,
   ) -> ir.Graph:
   ```

3. **Call site**: Pass the concrete values from config:

   ```python
   scan_body = _build_recurrence_body(
       uses_decay, uses_beta,
       kv_num_heads=kv_num_heads,
       head_k_dim=head_k_dim,
       head_v_dim=head_v_dim,
       stash_type=stash_type,
   )
   ```

### Body output shapes

You do **not** need to update body output shape annotations. If body
inputs have concrete dims, ORT shape inference propagates them through
Unsqueeze, MatMul, Add, etc. to produce correct output shapes.

## General rules for Scan/Loop body subgraphs

1. **Prefer concrete dimensions.** Any value known at graph build time
   (num_heads, head_dim, hidden_size, etc.) should be a concrete `int`
   in the body shape annotations, not a `dim_param`.

2. **Only batch and sequence length should be symbolic.** These are the
   only truly dynamic dimensions at runtime.

3. **Sequence length is the scan axis** — it does not appear in the
   body input shapes (Scan slices along this axis). So in practice,
   only batch (`B`) should be symbolic in body inputs.

4. **Test on CUDA EP.** CPU EP is more forgiving with symbolic dims.
   Always verify Scan-containing models on CUDA EP if that's a target.

5. **Watch for function inlining.** Scan inside `ir.Function` is the
   highest-risk pattern. Scan in the main graph is less likely to have
   this issue because ORT resolves shapes from the graph inputs
   directly.

## Verification

After applying the fix:

```bash
# CPU (should still work)
python examples/qwen35_text_generation.py --model Qwen/Qwen3.5-0.8B --dtype f16 --device cpu

# CUDA (should now work)
python examples/qwen35_text_generation.py --model Qwen/Qwen3.5-0.8B --dtype f16 --device cuda
```

## Related files

- `src/mobius/functions/linear_attention.py` — LinearAttention function
  with Scan body
- `src/mobius/components/_scan_utils.py` — `create_body_graph()` helper
- `src/mobius/tasks/_base.py` — `_register_linear_attention_functions()`
  call site
- `src/mobius/functions/linear_attention_test.py` — Unit tests for the
  function builder
