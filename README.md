# Requirements

- Linux with an NVIDIA GPU and a CUDA-compatible driver.
- CMake 3.20 or newer, Ninja, and a C++17 compiler.
- LLVM and MLIR CMake packages. The default paths used by the project are `/usr/lib/llvm-20/lib/cmake/llvm` and `/usr/lib/llvm-20/lib/cmake/mlir`.
- Python 3 and the CUDA build of PyTorch required by the selected backend: PyTorch 2.10.0 for Triton or TileLang, and PyTorch 2.13.0 for cuTile.

# Environment

Create one virtual environment for the backend you want to use. For Triton:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the required CUDA build of PyTorch 2.10.0 first.
python -m pip install -r environment/triton.txt
```

For cuTile or TileLang, install the required CUDA build of PyTorch first and replace the requirements file with `environment/cutile.txt` or `environment/tilelang.txt` respectively.

Configure the LLVM/MLIR locations (override the values if your installation differs):

```bash
export INTENT_MLIR_DIR="${INTENT_MLIR_DIR:-/usr/lib/llvm-20/lib/cmake/mlir}"
export INTENT_LLVM_DIR="${INTENT_LLVM_DIR:-/usr/lib/llvm-20/lib/cmake/llvm}"
```

# Build

```bash
cmake -S . -B /tmp/intentdsl-build -G Ninja \
  -DMLIR_DIR="${INTENT_MLIR_DIR}" \
  -DLLVM_DIR="${INTENT_LLVM_DIR}"
cmake --build /tmp/intentdsl-build --target intent-compile
```

# Test

Emit Triton backend code and run one numerical reproduction:

```bash
INTENT_PYTHON="$PWD/.venv/bin/python" ./examples/run/repro.sh triton softmax
```
