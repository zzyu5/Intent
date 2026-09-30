# Source installation

The supported setup path in this guide is Linux with an NVIDIA GPU, a CUDA-compatible driver, Python 3.10–3.12, and the LLVM/MLIR 20 SDK. The pinned Triton dependency set is in [triton.txt](triton.txt); its NumPy requirement limits this route to Python 3.12 or earlier. CPU and MLU toolchains have their own [experiment instructions](../experiments/README.md).

## LLVM/MLIR SDK and Python bindings

Install CMake, Ninja, a C++17 compiler, Python development headers, and Python's venv support. An LLVM/MLIR SDK needs the headers, libraries, TableGen tools, and CMake packages. On Debian/Ubuntu installations with the [LLVM package repository](https://apt.llvm.org/) configured, the relevant SDK packages are `llvm-20-dev`, `libmlir-20-dev`, and `mlir-20-tools`.

The SDK's CMake package alone does **not** supply the Python bindings used by Intent's frontend. Follow the [upstream MLIR Python build instructions](https://mlir.llvm.org/docs/Bindings/Python/) with the matching LLVM 20 source. An explicit source setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
git clone --depth 1 --branch llvmorg-20.1.8 \
  https://github.com/llvm/llvm-project.git /path/to/llvm-project
python -m pip install -r /path/to/llvm-project/mlir/python/requirements.txt

cmake -S /path/to/llvm-project/llvm -B /path/to/llvm-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=Native \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
cmake --build /path/to/llvm-build --target MLIRPythonModules --parallel 4
```

Choose actual source and build paths outside the Intent checkout. This is a substantial first-time LLVM build; an existing matching build can be reused. The Python interpreter used to build the bindings must match the destination environment's Python ABI.

## Install the Triton route

From the Intent checkout:

```bash
python3 environment/install_triton.py \
  --venv .venv \
  --mlir-build /path/to/llvm-build
source .venv/bin/activate
python examples/softmax.py
```

The installer:

1. Creates or reuses the chosen virtual environment, without adding the caller's `PYTHONPATH` or user site packages.
2. Uses upstream `cmake --install --component MLIRPythonModules` to install the bindings into that environment and registers their package directory with a `.pth` file.
3. Installs PyTorch 2.10.0 from the CUDA 13.0 wheel index, the existing Triton requirements, and Intent with its manual MCP.
4. Builds `intent-compile` and installs it, the tuning profiles, and public manual resources with the Python package.

If MLIR bindings are already installed in the selected environment, omit `--mlir-build`. For a different SDK location, supply `--mlir-dir` and `--llvm-dir`, or set `INTENT_MLIR_DIR` and `INTENT_LLVM_DIR`. `--torch-index-url` selects another official CUDA wheel index for the same PyTorch release; check the [PyTorch installation combinations](https://pytorch.org/get-started/previous-versions/).

Intent's C++ build defaults to the user cache, outside the checkout. `--build-dir` selects a different build directory and `--jobs` limits build parallelism. Re-running the installer reuses the environment and build directory. Failures stop installation and preserve the underlying command's diagnostics.

## Build the package directly

Once the SDK, Python bindings, and selected backend dependencies are installed:

```bash
python -m pip install '.[manual]' \
  --config-settings=cmake.define.MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  --config-settings=cmake.define.LLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  --config-settings=build-dir=/path/to/intent-build
```

The installed package discovers its own compiler. Developers can override it with `compiler="/path/to/intent-compile"` or `INTENT_COMPILER`. The compiler's `profiles/` directory must remain next to the executable.

The wheel uses the LLVM/MLIR runtime libraries from the selected SDK; it is not a self-contained binary distribution for machines without those libraries. MLIR Python bindings are installed separately through the upstream component and must remain available. The package metadata is derived from Git; no release tag is created by installation.

## Other GPU providers

[cutile.txt](cutile.txt) and [tilelang.txt](tilelang.txt) retain the dependency sets used by the existing experiments. Install them in separate environments with their documented PyTorch and toolchain requirements, then build the same Intent package. The automated installer above only covers the Triton route; it does not configure Mojo, Weft, or NeuWare. See the [backend instructions](../experiments/README.md) for those environments and the corresponding measured coverage.
