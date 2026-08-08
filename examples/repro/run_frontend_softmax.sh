#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build_root=/tmp/intentdsl-build

cmake \
  -S "${project_root}" \
  -B "${build_root}" \
  -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build "${build_root}" --target intent-opt intent-realize intent-translate

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${project_root}/python" \
python "${project_root}/examples/repro/frontend_softmax.py" \
  --intent-opt "${build_root}/tools/intent-opt/intent-opt" \
  --intent-realize "${build_root}/tools/intent-realize/intent-realize" \
  --intent-translate "${build_root}/tools/intent-translate/intent-translate" \
  --baseline-source "${project_root}/source/triton/triton/normalization/softmax/02-fused-softmax.py"
