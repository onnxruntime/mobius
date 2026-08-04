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

Runtime execution is intentionally outside Mobius. A runtime only needs to
implement this tensor contract and feed each `next_state` back as the following
step's `state`; it may use any ONNX-compatible execution backend.

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
