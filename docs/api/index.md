# API Reference

Public API documentation for `mobius`.

## Core Functions

- [`build()`](build.md) — Build from a HuggingFace model ID
- [`build_from_module()`](build_from_module.md) — Build from a module instance
- [`build_from_gguf()`](build_from_gguf.md) — Build from a GGUF file
- [`apply_weights()`](apply_weights.md) — Apply weights to a built model

## Core Classes

- [`ModelPackage`](model_package.md) — Collection of named ONNX models
- [`BaseModelConfig`](base_model_config.md) — Base configuration class
- [`ModelRegistry`](model_registry.md) — Architecture registry

```{toctree}
:hidden:
:maxdepth: 1

build
build_from_module
build_from_gguf
apply_weights
model_package
base_model_config
model_registry
```
