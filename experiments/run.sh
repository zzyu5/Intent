#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <triton|cutile|tilelang|mojo|weft|bangc> <output.csv> [kernel ...]" >&2
  exit 2
fi

provider=$1
output=$2
shift 2

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "${output}" != /* ]]; then
  output="${PWD}/${output}"
fi
build_root=${INTENT_BUILD_ROOT:-/tmp/intentdsl-build}
cmake_generator=${INTENT_CMAKE_GENERATOR:-Ninja}
mlir_dir=${INTENT_MLIR_DIR:-/usr/lib/llvm-20/lib/cmake/mlir}
llvm_dir=${INTENT_LLVM_DIR:-/usr/lib/llvm-20/lib/cmake/llvm}
tuning_config=${INTENT_TUNING_CONFIG:-}

case "${provider}" in
  mojo)
    experiment=cpu
    default_python="${HOME}/.venvs/intentdsl-mlir20/bin/python"
    export INTENT_MOJO="${INTENT_MOJO:-${HOME}/.venvs/intentdsl-mojo/bin/mojo}"
    ;;
  triton)
    experiment=gpu
    default_python=/home/kingdom/.venvs/intentdsl-mlir20/bin/python
    ;;
  cutile)
    experiment=gpu
    default_python=/home/kingdom/.venvs/intentdsl-cutile/bin/python
    tuning_config=${INTENT_TUNING_CONFIG:-${project_root}/experiments/gpu/providers/cutile/tuning.json}
    ;;
  tilelang)
    experiment=gpu
    default_python=/home/kingdom/.venvs/intentdsl-tilelang/bin/python
    ;;
  weft)
    experiment=cpu
    default_python="${HOME}/.venvs/intentdsl-mlir20/bin/python"
    ;;
  bangc)
    experiment=mlu
    default_python="${HOME}/.venvs/intentdsl-cutile/bin/python"
    export INTENT_MOJO="${INTENT_MOJO:-${HOME}/.venvs/intentdsl-mojo/bin/mojo}"
    ;;
  *)
    echo "unsupported provider: ${provider}" >&2
    exit 2
    ;;
esac

cmake_arguments=()
if [[ -n "${INTENT_WEFT_SOURCE_DIR:-}" ]]; then
  cmake_arguments+=("-DINTENT_WEFT_SOURCE_DIR=${INTENT_WEFT_SOURCE_DIR}")
fi
if [[ -n "${INTENT_WEFT_BINARY_DIR:-}" ]]; then
  cmake_arguments+=("-DINTENT_WEFT_BINARY_DIR=${INTENT_WEFT_BINARY_DIR}")
fi
cmake \
  -S "${project_root}" \
  -B "${build_root}" \
  -G "${cmake_generator}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="${mlir_dir}" \
  -DLLVM_DIR="${llvm_dir}" \
  "${cmake_arguments[@]}"
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

python_path="${project_root}:${project_root}/python:${project_root}/examples"
if [[ "${provider}" == weft && -n "${INTENT_WEFT_SOURCE_DIR:-}" ]]; then
  python_path+=":${INTENT_WEFT_SOURCE_DIR}/python"
fi

(
  cd /tmp
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${python_path}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${INTENT_PYTHON:-${default_python}}" \
    -m "experiments.${experiment}" "${arguments[@]}"
)
