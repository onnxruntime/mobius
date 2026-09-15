# `ModelPackage`

A dict-like collection of named `ir.Model` objects forming a complete model.

```python
from mobius import ModelPackage
```

## Class Signature

```python
class ModelPackage(UserDict[str, ir.Model]):
    config: object | None

    def __init__(
        self,
        models: dict[str, ir.Model] | None = None,
        config: object | None = None,
    ) -> None: ...
```

## Methods

### `save()`

Save all component models to a directory.

```python
def save(
    self,
    directory: str,
    *,
    external_data: str = "onnx",
    max_shard_size_bytes: int | None = None,
    max_workers: int = 8,
    components: Callable[[str], bool] | None = None,
    progress_bar: bool = True,
    check_weights: bool = True,
    include_policy_components: bool = True,
    include_adapter_artifacts: bool = True,
) -> None:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `directory` | `str` | (required) | Output directory path. |
| `external_data` | `str` | `"onnx"` | `"onnx"` or `"safetensors"` format. |
| `max_shard_size_bytes` | `int \| None` | `None` | Maximum shard size for ONNX or safetensors external data. |
| `max_workers` | `int` | `8` | ONNX external-data writer threads when supported; use `1` for serial writes. |
| `components` | `Callable \| None` | `None` | Predicate to select components to save. |
| `progress_bar` | `bool` | `True` | Show serialization progress. |
| `check_weights` | `bool` | `True` | Verify all initializers have weight data. |
| `include_policy_components` | `bool` | `True` | Save attached generation-policy models under `policies/`. |
| `include_adapter_artifacts` | `bool` | `True` | Save attached parameter-adapter bundles under `adapters/`. |

For packages whose weight-loading report sets `streaming_external_data=True`,
and for qualifying packages with an export report, `save()` publishes
transactionally to a destination that must not already exist. Models and
external data are first written to a staging directory and moved into place
only after every selected component succeeds. A failed lazy transform or
serializer call does not leave a success-shaped output directory.

Loaders may register source checkpoint artifacts on the package. For packages
that do so, `save()` rejects exact source-directory collisions for safetensors
outputs and output files that resolve to the same file as a registered source,
including hard-link and symlink aliases. Native GPT-OSS MXFP4 opts into both
this registration and transactional publication.

### `load()`

Load models from a directory.

```python
@classmethod
def load(cls, directory: str) -> ModelPackage:
```

### `apply_weights()`

Apply weights from a state dict to all component models.

```python
def apply_weights(
    self,
    state_dict: dict[str, torch.Tensor],
    prefix_map: dict[str, str] | None = None,
) -> None:
```

## Examples

```python
from mobius import build

# Build and save
pkg = build("meta-llama/Llama-3.2-1B")
pkg.save("output/llama/")

# Access individual models
model = pkg["model"]
print(model.graph.name)

# Check components
print(list(pkg.keys()))  # ["model"] for single-model
# ["model", "vision", "embedding"] for VLM

# Load from disk
pkg = ModelPackage.load("output/llama/")

# Save as safetensors
pkg.save("output/llama/", external_data="safetensors")

# Bound serializer concurrency
pkg.save("output/llama-serial/", max_workers=1)
```

## Output Layout

- **Single model**: `directory/model.onnx` + `directory/model.onnx.data`
- **Multi model**: `directory/{name}/model.onnx` for each component

Checkpoint shards used by lazy sources are inputs, not package outputs. Saving
does not remove them from the HuggingFace cache.
