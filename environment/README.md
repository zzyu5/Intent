# Source installation

The public source setup uses Linux, Python 3.10–3.12, and the LLVM/MLIR 20 SDK. One installer selects the backend dependencies and builds the same Intent package. GPU execution requires an appropriate NVIDIA driver; CPU and MLU toolchains are selected explicitly below. Installation, successful compilation, and numerical correctness are separate checks.

## LLVM/MLIR SDK and Python bindings

Install CMake, Ninja, a C++17 compiler, Python development headers, and Python's venv support. An LLVM/MLIR SDK needs the headers, libraries, TableGen tools, and CMake packages. On Debian/Ubuntu installations with the [LLVM package repository](https://apt.llvm.org/) configured, the relevant SDK packages are `llvm-20-dev`, `libmlir-20-dev`, and `mlir-20-tools`.

The SDK's CMake package alone does **not** supply the Python bindings used by Intent's frontend. Follow the [upstream MLIR Python build instructions](https://mlir.llvm.org/docs/Bindings/Python/) with the matching LLVM 20 source. An explicit source setup is:

```bash
python3 -m venv .venv-triton
source .venv-triton/bin/activate
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

## Choose a backend

Use a separate virtual environment for each public GPU route. The dependency declarations are in [backends.py](../python/intent/tools/backends.py); the installer and installed environment checker consume the same backend descriptions.

| Backend | Python runtime route | External requirements and current scope |
|---|---|---|
| `triton` | CUDA PyTorch 2.10.0, [triton.txt](triton.txt) | NVIDIA driver; callable GPU artifacts |
| `cutile` | CUDA PyTorch 2.10.0, [cutile-runtime.txt](cutile-runtime.txt) | NVIDIA driver compatible with the selected CUDA toolkit; callable GPU artifacts |
| `mojo` | CPU PyTorch 2.10.0 | Existing Mojo compiler, Linux x86-64 with AVX2/AVX512; callable CPU artifacts |
| `weft` | No additional tensor framework for source generation | Matching Weft source/build; public `WeftTarget` generates canonical Weft source, native execution requires its deployment/toolchain |
| `bangc` | Native buffer interface uses the Python standard library | Existing NeuWare SDK and MLU370 implementation; execution requires an actual compatible MLU device |

For Triton, from the Intent checkout:

```bash
python3 environment/install.py --backend triton \
  --venv .venv-triton \
  --mlir-build /path/to/llvm-build
source .venv-triton/bin/activate
intent doctor --target triton
python examples/softmax.py
```

For cuTile, using a matching Python ABI and the same SDK build:

```bash
python3 environment/install.py --backend cutile \
  --venv .venv-cutile --mlir-build /path/to/llvm-build
source .venv-cutile/bin/activate
intent doctor --target cutile
python examples/softmax.py --target cutile
```

The public cuTile route uses cuda-tile 1.6 and CUDA toolkit 13.3.1. The older [cutile.txt](cutile.txt), including TileGym, remains the historical experiment environment; it is not installed into the public route. This does not change its recorded results. Backend/device support still depends on the selected SDK and the kernel's operations.

The common installer:

1. Creates or reuses the chosen virtual environment, without adding the caller's `PYTHONPATH` or user site packages.
2. Uses upstream `cmake --install --component MLIRPythonModules` to install the bindings into that environment and registers their package directory with a `.pth` file.
3. Installs the selected Python dependencies and the runtime NumPy/dtype dependencies of MLIR's Python bindings.
4. Builds `intent-compile` and installs its profiles, the public CLI, and both MCP entry points with the package.

If MLIR bindings are already installed in the selected environment, omit `--mlir-build`. For another SDK location, supply `--mlir-dir` and `--llvm-dir`, or set `INTENT_MLIR_DIR` and `INTENT_LLVM_DIR`. `--torch-index-url` overrides the selected route's wheel index for the same PyTorch release; check the [PyTorch installation combinations](https://pytorch.org/get-started/previous-versions/).

Intent's C++ build defaults to the user cache, outside the checkout. `--build-dir` selects a different build directory and `--jobs` limits build parallelism. Re-running the installer reuses the environment and build directory. Failures stop installation and preserve the underlying command's diagnostics.

## Select an external CPU or MLU toolchain

These commands install Intent around existing toolchains; they do not install a compiler, driver, remote machine, or NeuWare license:

```bash
python3 environment/install.py --backend mojo --venv .venv-mojo \
  --mlir-build /path/to/llvm-build --provider-compiler /path/to/mojo

python3 environment/install.py --backend weft --venv .venv-weft \
  --mlir-build /path/to/llvm-build \
  --weft-source-dir /path/to/Weft --weft-binary-dir /path/to/weft-build

python3 environment/install.py --backend bangc --venv .venv-bangc \
  --mlir-build /path/to/llvm-build --neuware /path/to/neuware
```

Use the selected paths in later target constructors or CLI options. For example:

```bash
.venv-mojo/bin/intent doctor --target mojo --target-option 'compiler="/path/to/mojo"'
.venv-weft/bin/intent doctor --target weft --target-option vector_bits=256 --target-option workers=8
.venv-bangc/bin/intent doctor --target bangc --target-option 'neuware="/path/to/neuware"'
```

Weft's vector width and worker count above are an example of explicit construction budgets; supply the values for your deployment. Its CMake integration requires matching LLVM/MLIR and generated Weft headers/libraries. [CPU](../experiments/cpu/README.md) and [MLU](../experiments/mlu/README.md) retain the production deployment instructions and measured coverage. The public installer does not infer those settings from a private profile.

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

## Compile and diagnose

`intent doctor --target BACKEND --json` reports dependencies and resolved target facts, without executing an operator. `intent compile path/to/program.py:kernel --target BACKEND --json` uses `intent.generate` and returns artifact paths or the compiler's actual failure stage. An importable `package.module:kernel` is also accepted by the CLI. `--constexpr NAME=JSON`, `--target-option NAME=JSON`, `--compiler`, and `--tuning-config` forward existing public settings. `--materialize` materializes that same generated program; it never launches the kernel.

The `intent-manual` MCP stays read-only. Explicitly enable `intent-compiler-mcp` when an agent should compile an existing user-supplied `.py` file; its compilation tool shares the CLI implementation. Both servers use the `manual` dependency extra. Module loading executes ordinary top-level Python host code, while kernel execution and numerical validation remain separate.

[tilelang.txt](tilelang.txt) retains its prior environment and source corpus. It is not one of the installer routes above.
