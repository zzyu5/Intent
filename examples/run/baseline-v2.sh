#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <triton|cutile|tilelang> <output.csv> [kernel ...]" >&2
  exit 2
fi

provider=$1
output=$2
shift 2

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ "${output}" != /* ]]; then
  output="${PWD}/${output}"
fi
build_root=${INTENT_BUILD_ROOT:-/tmp/intentdsl-build}
cmake_generator=${INTENT_CMAKE_GENERATOR:-Ninja}
mlir_dir=${INTENT_MLIR_DIR:-/usr/lib/llvm-20/lib/cmake/mlir}
llvm_dir=${INTENT_LLVM_DIR:-/usr/lib/llvm-20/lib/cmake/llvm}
tuning_config=${INTENT_TUNING_CONFIG:-}

case "${provider}" in
  triton)
    default_python=/home/kingdom/.venvs/intentdsl-mlir20/bin/python
    ;;
  cutile)
    default_python=/home/kingdom/.venvs/intentdsl-cutile/bin/python
    tuning_config=${INTENT_TUNING_CONFIG:-${project_root}/examples/repro/v2/providers/cutile/tuning.json}
    ;;
  tilelang)
    default_python=/home/kingdom/.venvs/intentdsl-tilelang/bin/python
    ;;
  *)
    echo "unsupported provider: ${provider}" >&2
    exit 2
    ;;
esac

cmake \
  -S "${project_root}" \
  -B "${build_root}" \
  -G "${cmake_generator}" \
  -DMLIR_DIR="${mlir_dir}" \
  -DLLVM_DIR="${llvm_dir}"
cmake --build "${build_root}" --target intent-compile --parallel "${INTENT_BUILD_JOBS:-24}"

arguments=(
  "${provider}"
  --compiler "${build_root}/tools/intent-compile/intent-compile"
  --output "${output}"
  --jobs "${INTENT_BENCHMARK_JOBS:-4}"
  --worker-timeout "${INTENT_WORKER_TIMEOUT:-300}"
  --cutile-compiler-timeout "${INTENT_CUTILE_COMPILER_TIMEOUT:-15}"
)
if [[ -n "${tuning_config}" ]]; then
  if [[ "${tuning_config}" != /* ]]; then
    tuning_config="${PWD}/${tuning_config}"
  fi
  arguments+=(--tuning-config "${tuning_config}")
fi
for kernel in "$@"; do
  arguments+=(--kernel "${kernel}")
done

(
  cd /tmp
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${project_root}/python:${project_root}/examples" \
  "${INTENT_PYTHON:-${default_python}}" \
    -m repro.v2.runner "${arguments[@]}"
)
