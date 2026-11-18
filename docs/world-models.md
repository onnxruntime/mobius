# Export world models

Mobius exports a world model as a directory containing multiple ONNX models
and a `pipeline.json` file. Use a compatible runtime, such as
`onnx-world-model`, to run the package.

## Quick start

```bash
# Cosmos3 Edge
mobius build --model nvidia/Cosmos3-Edge output/cosmos3-edge \
    --features world-model

# Cosmos3 Omni
mobius build --model nvidia/Cosmos3-Nano output/cosmos3-nano \
    --features world-model
```

Python API:

```python
from mobius import build_world_model

package = build_world_model("nvidia/Cosmos3-Edge")
package.save("output/cosmos3-edge")
```

## Output

The components depend on the checkpoint. A Cosmos3 Edge package contains:

```text
cosmos3-edge/
├── pipeline.json
├── reasoner_decoder/model.onnx
├── reasoner_embedding/model.onnx
├── reasoner_vision_encoder/model.onnx
├── generator/model.onnx
├── video_encoder/model.onnx
├── video_decoder/model.onnx
├── tokenizer.json
└── scheduler/scheduler_config.json
```

`pipeline.json` tells the runtime how to execute the models, including
generated inputs, recurrent state, sampling, scheduling, and component
dtype/EP requirements.

## Run the package

```python
from onnx_world_model import Pipeline

pipeline = Pipeline("output/cosmos3-edge")
session = pipeline.create_session()
```

See the `onnx-world-model` documentation for text, image/video, and action
inference examples.

## Supported checkpoints

| Family | Examples |
|---|---|
| Cosmos3 Edge | `nvidia/Cosmos3-Edge`, `nvidia/Cosmos3-Edge-Policy-DROID` |
| Cosmos3 Omni | Cosmos3-Nano, Cosmos3-Super, Text2Image, Image2Video, and Policy-DROID variants |

Some variants omit optional components such as the vision or audio encoder.
Mobius includes only the components present in the checkpoint.

## Options

- Use `load_weights=False` to build graph structure without downloading model
  weights.
- Use `--dtype f32` for CPU inference. The native Cosmos3 transformer dtype is
  BF16 and normally targets CUDA.

## Cosmos3 Edge image and video input

`reasoner_vision_encoder` accepts packed image or video patches:

| Input | Shape |
|---|---|
| `pixel_values` | `[total_patches, patch_dim]` |
| `grid_thw` | `[3]` (`frames`, `grid_height`, `grid_width`) |

Use the checkpoint's Cosmos3 Edge image/video processor to resize, normalize,
and patchify media. Route the encoder output to `image_features` for images or
`video_features` for videos.

The Edge Reasoner vision, fusion, and decoder outputs are numerically verified
against the published Transformers implementation with the real checkpoint.
