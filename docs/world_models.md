# World models

Mobius can build world models directly as ONNX graphs without tracing a
PyTorch `forward()` method. The initial framework defines a deterministic,
single-step state-transition contract:

```text
(observation, action, state)
    -> (next_state, observation_prediction, reward, continuation)
```

All tensors have a dynamic leading batch dimension. `reward` and
`continuation` have shape `[batch, 1]`; `continuation` is a probability.

## Build the reference model

`MLPWorldModel` is a minimal directly declared implementation. Its input shapes
exclude the batch dimension:

```python
from safetensors.torch import load_file

from mobius import MLPWorldModel, WorldModelConfig, build_from_module

config = WorldModelConfig(
    observation_shape=(64,),
    action_shape=(6,),
    state_shape=(128,),
    hidden_size=512,
    num_hidden_layers=3,
)
module = MLPWorldModel(config)
package = build_from_module(module, config, task="world-model")

weights = load_file("world_model.safetensors")
package.apply_weights(module.preprocess_weights(weights))
package.save("world-model-onnx")
```

The ONNX graph is constructed through `onnxscript.nn.Module` and
`onnx_ir.GraphBuilder`; PyTorch is used only as a possible source of weight
tensors.

## Run with ONNX Runtime

Install the optional runtime dependency:

```bash
pip install -e ".[runtime]"
```

`WorldModelRunner` validates the graph contract and retains `next_state`
between calls:

```python
import numpy as np

from mobius.integrations.onnxruntime import WorldModelRunner

runner = WorldModelRunner.from_path(
    "world-model-onnx/model.onnx",
    providers=["CPUExecutionProvider"],
)

observation = np.zeros((1, 64), dtype=np.float32)
action = np.zeros((1, 6), dtype=np.float32)

# The first call creates a zero state. Later calls reuse next_state.
result = runner.step(observation, action)
result = runner.step(observation, action)

trajectory = runner.rollout(
    [observation, observation],
    [action, action],
)
```

Pass `state=` to `step()` or `initial_state=` to `rollout()` when the model
uses a learned or externally sampled initial state. Each `rollout()` starts
from zero state unless `initial_state` is provided; repeated `step()` calls
continue from the runner's retained state.

## Implement a custom world model

A custom module must use the task's forward and return contracts:

```python
from onnxscript import nn


class MyWorldModel(nn.Module):
    def forward(self, op, observation, action, state):
        # Directly declare ONNX operations here.
        ...
        return next_state, observation_prediction, reward, continuation
```

Use a custom `ModelTask` when an architecture needs multiple recurrent-state
tensors, stochastic latent outputs, separate observe/imagine graphs, or a
different prediction contract. Keep sampling outside the graph unless it must
be part of the deployed model.
