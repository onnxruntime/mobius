---
name: building-ort-genai
description: >
  Use this skill when building OnnxRuntime and onnxruntime-genai from
  source with CUDA support. Covers CUDA toolkit and cuDNN setup,
  ORT build flags, GenAI build linked to custom ORT, verification,
  and common build issues.
---

# Skill: Building ORT and GenAI from Source

## When to use

Use this skill when:
- You need a custom ORT build (e.g. unreleased features, CUDA support,
  custom ops)
- You need a custom GenAI build linked to your ORT build
- Deploying to Foundry Local requires overriding bundled ORT/GenAI
- Debugging inference issues that require source-level ORT changes
- The pip-released ORT/GenAI version doesn't support your model

## Prerequisites

- Linux machine with NVIDIA GPU
- conda or any Python 3.10+ environment
- ~20 GB disk space for builds
- CMake 3.26+, gcc/g++ 11+

## Step 1: CUDA toolkit

Install CUDA toolkit 12.8+ or 13.0 to a local directory if not
system-installed:

```bash
# Check existing installation
nvcc --version
nvidia-smi

# If not installed, download from:
# https://developer.nvidia.com/cuda-downloads

# Set environment (adjust path to your installation)
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Step 2: cuDNN 9.x

ORT requires cuDNN for conv and attention kernels. Install via pip
and create a directory layout for the ORT build:

```bash
# Install cuDNN via pip
pip install nvidia-cudnn-cu12

# Create cuDNN home directory for the ORT build system
mkdir -p ~/cudnn9/{lib,include}
CUDNN_PKG=$(python -c "import nvidia.cudnn; import pathlib; print(pathlib.Path(nvidia.cudnn.__file__).parent)")
ln -sf $CUDNN_PKG/lib/* ~/cudnn9/lib/
ln -sf $CUDNN_PKG/include/* ~/cudnn9/include/
export CUDNN_HOME=~/cudnn9
```

## Step 3: Build ORT from source

```bash
git clone https://github.com/microsoft/onnxruntime.git ~/dev/onnxruntime
cd ~/dev/onnxruntime

./build.sh \
  --config Release \
  --use_cuda \
  --cuda_home $CUDA_HOME \
  --cudnn_home $CUDNN_HOME \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=native \
    onnxruntime_USE_FLASH_ATTENTION=ON \
  --build_wheel \
  --enable_pybind \
  --parallel \
  --skip_tests
```

### Build flag reference

| Flag | Description |
|------|-------------|
| `--use_cuda` | Enable CUDA execution provider |
| `--cuda_home <path>` | Path to CUDA toolkit installation |
| `--cudnn_home <path>` | Path to cuDNN directory (lib/ + include/) |
| `CMAKE_CUDA_ARCHITECTURES=native` | Compile for the GPU architecture on this machine |
| `onnxruntime_USE_FLASH_ATTENTION=ON` | Enable Flash Attention kernels |
| `--build_wheel` | Build a pip-installable wheel |
| `--enable_pybind` | Build the Python bindings |
| `--parallel` | Parallel compilation |
| `--skip_tests` | Skip building test targets |

### Install the wheel

```bash
pip install build/Linux/Release/dist/onnxruntime-*.whl \
  --force-reinstall --no-deps
```

### Verify

```python
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider'
print(ort.get_device())
# Should print 'GPU'
```

## Step 4: Create ORT install layout for GenAI

GenAI needs ORT headers and libraries in a specific layout:

```bash
mkdir -p ~/ort-install/{include,lib}

# Headers
cp ~/dev/onnxruntime/include/onnxruntime/core/session/*.h \
  ~/ort-install/include/

# Libraries
cp ~/dev/onnxruntime/build/Linux/Release/libonnxruntime.so \
  ~/ort-install/lib/
cp ~/dev/onnxruntime/build/Linux/Release/libonnxruntime_providers_cuda.so \
  ~/ort-install/lib/
cp ~/dev/onnxruntime/build/Linux/Release/libonnxruntime_providers_shared.so \
  ~/ort-install/lib/
```

## Step 5: Build GenAI from source

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git \
  ~/dev/onnxruntime-genai
cd ~/dev/onnxruntime-genai

python build.py \
  --config Release \
  --use_cuda \
  --cuda_home $CUDA_HOME \
  --ort_home ~/ort-install \
  --parallel \
  --skip_tests \
  --skip_examples \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=native \
  --update --build
```

### Install the wheel

```bash
pip install build/Linux/Release/wheel/onnxruntime_genai_cuda-*.whl \
  --no-deps
```

### Verify

```python
import onnxruntime_genai as og
print(og.__version__)
print(og.is_cuda_available())  # Should print True
```

If `is_cuda_available()` returns `False`, check that the ORT CUDA
provider `.so` files are in the library path (see troubleshooting).

## Common issues

### 1. cuDNN version mismatch

**Symptom:** ORT build fails with cuDNN-related errors, or CUDA EP
doesn't load at runtime.

**Fix:** Ensure `nvidia-cudnn-cu12` pip package matches your CUDA
version. cuDNN 9.x works with CUDA 12.x and 13.x:

```bash
pip install nvidia-cudnn-cu12
# Verify version
python -c "import nvidia.cudnn; print(nvidia.cudnn.__version__)"
```

### 2. LD_LIBRARY_PATH not set

**Symptom:** `import onnxruntime` fails with `.so` not found errors,
or CUDA EP is missing from available providers.

**Fix:** Add CUDA and cuDNN libraries to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_HOME/lib:$LD_LIBRARY_PATH
```

Add this to your shell profile (`~/.bashrc` or conda
`activate.d/env_vars.sh`) for persistence.

### 3. pip packages overriding custom builds

**Symptom:** After `pip install foundry-local-sdk` or
`pip install onnxruntime-genai`, your custom build is replaced with
the pip version. `ort.get_device()` returns `'CPU'` instead of `'GPU'`.

**Fix:** Always reinstall custom wheels after installing packages that
depend on ORT or GenAI:

```bash
pip install foundry-local-sdk
# THEN reinstall your builds:
pip install --force-reinstall ~/dev/onnxruntime/build/Linux/Release/dist/onnxruntime-*.whl
pip install --force-reinstall ~/dev/onnxruntime-genai/build/Linux/Release/wheel/onnxruntime_genai_cuda-*.whl
```

See the `foundry-local` skill for more details on the Foundry
dependency override mechanism.

### 4. ABI mismatch between ORT and GenAI

**Symptom:** GenAI build fails with undefined symbols or linker errors
referencing ORT internal types.

**Fix:** Both must be built with the same compiler and C++ ABI. Ensure:
- Same gcc/g++ version for both builds
- Same Python version
- ORT install layout (Step 4) uses the exact libraries from your ORT
  build, not from a different build or pip install

### 5. Build fails on test targets

**Symptom:** ORT build fails on `onnxruntime_perf_test` or similar
test targets due to abseil linking errors.

**Fix:** The core library and Python wheel still build successfully.
Build the wheel manually:

```bash
cd ~/dev/onnxruntime/build/Linux/Release
python ~/dev/onnxruntime/setup.py bdist_wheel
pip install dist/onnxruntime-*.whl --force-reinstall --no-deps
```

### 6. Flash Attention not available

**Symptom:** Attention ops are slow or fall back to non-fused path.

**Fix:** Ensure `onnxruntime_USE_FLASH_ATTENTION=ON` was set during
the ORT build. Flash Attention requires a GPU with compute capability
≥ 8.0 (Ampere or newer).

## Reference

- [Full Gemma4 + Foundry Local tutorial](https://github.com/onnxruntime/mobius/issues/245)
- [ORT build documentation](https://onnxruntime.ai/docs/build/)
- [GenAI build documentation](https://github.com/microsoft/onnxruntime-genai/blob/main/BUILD.md)

## Cross-references

- **Foundry Local deployment:** `.agents/skills/foundry-local/SKILL.md`
- **ONNX export:** `.agents/skills/onnx-export-quantization/SKILL.md`
- **ORT GenAI config:** `.agents/skills/ort-genai-config/SKILL.md`
