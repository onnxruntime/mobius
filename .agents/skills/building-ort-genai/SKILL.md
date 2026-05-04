---
name: building-ort-genai
description: >
  Use this skill when building OnnxRuntime and onnxruntime-genai from
  source with CUDA support. Covers CUDA toolkit and cuDNN installation,
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
- The pip-released ORT/GenAI version doesn't support your model

## Prerequisites

- Linux (tested on Ubuntu)
- NVIDIA GPU
- conda or any Python 3.10+ environment
- ~20 GB disk space for builds
- CMake 3.26+, gcc/g++ 11+

## Step 1: Install CUDA toolkit

```bash
# CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run \
  --toolkit --toolkitpath=$HOME/cuda12.8 \
  --silent --override --no-man-page

# Or CUDA 13.0 if available — adjust the URL and path accordingly
```

Verify:

```bash
$HOME/cuda12.8/bin/nvcc --version
```

## Step 2: Install cuDNN 9.x

```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.8.0.87_cuda12-archive.tar.xz
mkdir -p $HOME/cudnn9.8
tar -xf cudnn-linux-x86_64-9.8.0.87_cuda12-archive.tar.xz \
  -C $HOME/cudnn9.8 --strip-components=1
```

Alternative — install cuDNN via pip and create symlinks:

```bash
pip install nvidia-cudnn-cu12
mkdir -p ~/cudnn9/{lib,include}
CUDNN_PKG=$(python -c "import nvidia.cudnn; import pathlib; print(pathlib.Path(nvidia.cudnn.__file__).parent)")
ln -sf $CUDNN_PKG/lib/* ~/cudnn9/lib/
ln -sf $CUDNN_PKG/include/* ~/cudnn9/include/
# Then use ~/cudnn9 as CUDNN_HOME below
```

## Step 3: Set environment

```bash
export PATH=$HOME/cuda12.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda12.8/lib64:$HOME/cudnn9.8/lib:$LD_LIBRARY_PATH
```

Add these to your shell profile (`~/.bashrc`) or conda
`activate.d/env_vars.sh` for persistence.

## Step 4: Build ORT from source

```bash
git clone https://github.com/microsoft/onnxruntime.git ~/dev/onnxruntime
cd ~/dev/onnxruntime

./build.sh \
  --config Release \
  --use_cuda \
  --cuda_home $HOME/cuda12.8 \
  --cudnn_home $HOME/cudnn9.8 \
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
| `CMAKE_CUDA_ARCHITECTURES=native` | Compile for the GPU on this machine |
| `onnxruntime_USE_FLASH_ATTENTION=ON` | Enable Flash Attention kernels (requires compute ≥ 8.0) |
| `--build_wheel` | Build a pip-installable wheel |
| `--parallel` | Parallel compilation |
| `--skip_tests` | Skip test targets (may fail on abseil linking) |

### Install

```bash
pip install build/Linux/Release/dist/onnxruntime-*.whl \
  --force-reinstall --no-deps
```

> **Note:** If the build fails on test targets (`onnxruntime_perf_test`)
> but the wheel was produced, you can still install it. Alternatively,
> build the wheel manually:
> ```bash
> cd build/Linux/Release
> python ~/dev/onnxruntime/setup.py bdist_wheel
> pip install dist/onnxruntime-*.whl --force-reinstall --no-deps
> ```

## Step 5: Create ORT install layout for GenAI

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

## Step 6: Build GenAI from source

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git \
  ~/dev/onnxruntime-genai
cd ~/dev/onnxruntime-genai

python build.py \
  --config Release \
  --use_cuda \
  --cuda_home $HOME/cuda12.8 \
  --ort_home ~/ort-install \
  --parallel \
  --skip_tests \
  --skip_examples \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=native \
  --update --build
```

### Install

```bash
pip install build/Linux/Release/wheel/onnxruntime_genai_cuda-*.whl \
  --no-deps
```

## Step 7: Verify

```bash
python -c 'import onnxruntime; print(onnxruntime.__version__, onnxruntime.get_device())'
# Expected: 1.x.x GPU

python -c 'import onnxruntime_genai as og; print(og.__version__, og.is_cuda_available())'
# Expected: 0.x.x True
```

If `get_device()` returns `CPU` or `is_cuda_available()` returns
`False`, check `LD_LIBRARY_PATH` and that you installed the correct
wheels (not pip overrides — see below).

## Common issues

### 1. cuDNN version mismatch

**Symptom:** ORT build fails with cuDNN errors, or CUDA EP doesn't
load at runtime.

**Fix:** cuDNN 9.x works with CUDA 12.x and 13.x. Ensure the cuDNN
version matches your CUDA major version:

```bash
# Check installed cuDNN version
python -c "import nvidia.cudnn; print(nvidia.cudnn.__version__)"
```

### 2. LD_LIBRARY_PATH not set

**Symptom:** `import onnxruntime` fails with `.so` not found, or
CUDA EP missing from providers.

**Fix:**

```bash
export LD_LIBRARY_PATH=$HOME/cuda12.8/lib64:$HOME/cudnn9.8/lib:$LD_LIBRARY_PATH
```

### 3. pip packages overriding custom builds

**Symptom:** After `pip install foundry-local-sdk` or other packages,
custom build is replaced. `ort.get_device()` returns `'CPU'`.

**Fix:** Always reinstall custom wheels after installing packages that
depend on ORT:

```bash
pip install foundry-local-sdk
# THEN reinstall your builds:
pip install --force-reinstall <your_ort_wheel>.whl
pip install --force-reinstall <your_genai_wheel>.whl
```

See the **foundry-local** skill for the full dependency override
mechanism.

### 4. ABI mismatch between ORT and GenAI

**Symptom:** GenAI build fails with undefined symbols or linker errors.

**Fix:** Both must be built with the same compiler, Python version,
and C++ ABI. The ORT install layout (Step 5) must use the exact
libraries from your ORT build.

### 5. Flash Attention not available

**Symptom:** Attention ops are slow or fall back to non-fused path.

**Fix:** Ensure `onnxruntime_USE_FLASH_ATTENTION=ON` was set during
the ORT build. Requires GPU compute capability ≥ 8.0 (Ampere+).

## Reference

- [Full Gemma4 + Foundry Local tutorial](https://github.com/onnxruntime/mobius/issues/245)
- [ORT build documentation](https://onnxruntime.ai/docs/build/)
- [GenAI build documentation](https://github.com/microsoft/onnxruntime-genai/blob/main/BUILD.md)

## Cross-references

- **Foundry Local deployment:** `.agents/skills/foundry-local/SKILL.md`
- **ONNX export:** `.agents/skills/onnx-export-quantization/SKILL.md`
- **ORT GenAI config:** `.agents/skills/ort-genai-config/SKILL.md`
