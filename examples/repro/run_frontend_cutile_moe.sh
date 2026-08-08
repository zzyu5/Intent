#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build_root=/tmp/intentdsl-build
python_bin=/home/kingdom/.venvs/intentdsl-cutile/bin/python

cmake \
  -S "${project_root}" \
  -B "${build_root}" \
  -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build "${build_root}" \
  --target intent-cutile-realize intent-cutile-translate

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${project_root}/python:${project_root}/examples/repro" \
"${python_bin}" "${project_root}/examples/repro/frontend_cutile_moe.py" \
  --intent-realize "${build_root}/tools/intent-cutile-realize/intent-cutile-realize" \
  --intent-translate "${build_root}/tools/intent-cutile-translate/intent-cutile-translate" \
  --baseline-source "${project_root}/source/cutile/cutile-python/moe/fused/MoE.py"
