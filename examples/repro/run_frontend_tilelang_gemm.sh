#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build_root=/tmp/intentdsl-build
python_bin=/home/kingdom/.venvs/intentdsl-mlir20/bin/python

cmake \
  -S "${project_root}" \
  -B "${build_root}" \
  -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build "${build_root}" \
  --target intent-tilelang-realize intent-tilelang-translate

(
  cd /tmp
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${project_root}/python" \
  "${python_bin}" "${project_root}/examples/repro/frontend_tilelang_gemm.py" \
    --intent-realize "${build_root}/tools/intent-tilelang-realize/intent-tilelang-realize" \
    --intent-translate "${build_root}/tools/intent-tilelang-translate/intent-tilelang-translate" \
    --baseline-source "${project_root}/source/tilelang/tilelang/gemm/dense/example_gemm.py"
)
