# Extracting Intermediate ONNX Values

Three methods for extracting intermediate values from ONNX models to
compare block-by-block against HuggingFace. Used when a pipeline stage
(e.g., vision encoder) diverges and you need to narrow down the root cause.

---

## Method 1: Add intermediate outputs to the ONNX graph

The most reliable approach — expose any internal node's output as a graph
output so ORT returns it alongside normal outputs.

```python
import onnx

model = onnx.load("vision.onnx")
graph = model.graph

# Find the node whose output you want to inspect
for node in graph.node:
    if node.op_type == "RMSNormalization" and "block_0" in node.output[0]:
        # Name the output (if unnamed, give it a name)
        target_output = node.output[0]
        break

# Add as a graph output
graph.output.append(
    onnx.helper.make_tensor_value_info(target_output, onnx.TensorProto.FLOAT, None)
)
onnx.save(model, "vision_debug.onnx")

# Now ORT will return this value alongside image_features
session = ort.InferenceSession("vision_debug.onnx")
results = session.run(None, feeds)
# results[-1] is the intermediate value
```

## Method 2: Use `ir.Model` graph manipulation (preferred for mobius)

When working with `ir.Model` objects from the build pipeline, manipulate
the graph directly without saving/loading:

```python
from mobius._testing.ort_inference import OnnxModelSession

pkg = build(model_id, dtype="f32", load_weights=True)
vision_model = pkg["vision"]
graph = vision_model.graph

# Find target nodes by op type or name pattern
target_nodes = [n for n in graph if n.op_type == "RMSNormalization"]

# RMSNorm input nodes are natural block boundaries:
# rms_nodes[0] = block 0 norm1 (input = patch_embed output)
# rms_nodes[2] = block 1 norm1 (input = block 0 output)
# rms_nodes[2*i] = block i norm1 (input = block i-1 output)
block_0_output = target_nodes[2].inputs[0]  # block 0's output
block_0_output.name = "block_0_output"
graph.outputs.append(block_0_output)

session = OnnxModelSession(vision_model)
out = session.run({"pixel_values": pv, "grid_thw": grid_thw})
block_0_out = out["block_0_output"]
session.close()
```

## Method 3: Hook HuggingFace model for reference values

Use PyTorch hooks to extract intermediate values from HuggingFace at
the same points:

```python
intermediates = {}

def hook_fn(name):
    def fn(module, input, output):
        if isinstance(output, tuple):
            intermediates[name] = output[0].detach().cpu().numpy()
        else:
            intermediates[name] = output.detach().cpu().numpy()
    return fn

# Register hooks on specific blocks
for i, block in enumerate(hf_model.model.visual.blocks):
    block.register_forward_hook(hook_fn(f"block_{i}"))

# Run forward pass — hooks capture all intermediate values
with torch.no_grad():
    hf_out = hf_model.model.visual(pixel_values, grid_thw=grid_thw)

# Now compare block by block
for i in range(num_blocks):
    hf_block_out = intermediates[f"block_{i}"]
    cos = cosine_similarity(onnx_block_out, hf_block_out)
    print(f"Block {i}: cos={cos:.6f}")
```

## Block-by-block comparison strategy

When overall output diverges, narrow down by comparing each transformer
block's output sequentially:

```python
for i in range(num_blocks):
    onnx_out_i = extract_onnx_block_output(vision_model, i, feeds)
    hf_out_i = intermediates[f"block_{i}"]

    cos = cosine_similarity(onnx_out_i.flatten(), hf_out_i.flatten())
    max_diff = np.max(np.abs(onnx_out_i - hf_out_i))
    print(f"Block {i:2d}: cos={cos:.6f}  max_diff={max_diff:.4f}")
```

Typical pattern for a bug in block N:
```
Block 0:  cos=1.000000  max_diff=0.0001  ← perfect
Block 1:  cos=1.000000  max_diff=0.0001  ← perfect
...
Block N:  cos=0.961000  max_diff=4.8500  ← divergence starts!
Block N+1: cos=0.892000  max_diff=25.00  ← error compounds
```

Once you identify the divergent block, drill deeper into that block's
sub-operations (attention, MLP, normalization) to find the root cause.

## Comparing specific weight values

To verify weights loaded correctly, compare ONNX initializers against
HuggingFace state dict:

```python
from safetensors import safe_open

# Load HF weights
with safe_open(safetensors_path, framework="numpy") as f:
    hf_weight = f.get_tensor("visual.blocks.0.attn.qkv.weight")

# Get ONNX weight from ir.Model
onnx_weight = vision_model.graph.initializers["blocks.0.attn.qkv.weight"]
onnx_np = onnx_weight.const_value.numpy()

max_diff = np.max(np.abs(hf_weight.astype(np.float32) - onnx_np))
print(f"Weight diff: {max_diff}")  # Should be 0.0
```
