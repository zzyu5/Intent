#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <triton|cutile|tilelang> <softmax|gemm|attention|moe>" >&2
  exit 2
fi

backend=$1
kernel=$2
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build_root=/tmp/intentdsl-build

case "${backend}" in
  triton)
    default_python=/home/kingdom/.venvs/intentdsl-mlir20/bin/python
    ;;
  cutile)
    default_python=/home/kingdom/.venvs/intentdsl-cutile/bin/python
    ;;
  tilelang)
    default_python=/home/kingdom/.venvs/intentdsl-tilelang/bin/python
    ;;
  *)
    echo "unsupported backend: ${backend}" >&2
    exit 2
    ;;
esac

case "${backend}:${kernel}" in
  triton:softmax)
    baseline=source/triton/triton/normalization/softmax/02-fused-softmax.py
    ;;
  triton:gemm)
    baseline=source/triton/triton/gemm/dense/03-matrix-multiplication.py
    ;;
  triton:attention)
    baseline=source/triton/triton/attention/fused/06-fused-attention.py
    ;;
  triton:moe)
    baseline=source/triton/triton/gemm/grouped/08-grouped-gemm.py
    ;;
  cutile:softmax)
    baseline=source/cutile/tilegym/normalization/softmax/softmax.py
    ;;
  cutile:gemm)
    baseline=source/cutile/tilegym/gemm/dense/matmul.py
    ;;
  cutile:attention)
    baseline=source/cutile/cutile-python/attention/fmha/AttentionFMHA.py
    ;;
  cutile:moe)
    baseline=source/cutile/cutile-python/moe/fused/MoE.py
    ;;
  tilelang:softmax)
    baseline=source/tilelang/tilelang/normalization/online_softmax/online_softmax.py
    ;;
  tilelang:gemm)
    baseline=source/tilelang/tilelang/gemm/dense/example_gemm.py
    ;;
  tilelang:attention)
    baseline=source/tilelang/tilelang/attention/flash_forward_bshd/example_mha_fwd_bshd.py
    ;;
  tilelang:moe)
    baseline=source/tilelang/tilelang/gemm/grouped/example_grouped_gemm_fwd.py
    ;;
  *)
    echo "unsupported repro: ${backend}:${kernel}" >&2
    exit 2
    ;;
esac

python_bin=${INTENT_PYTHON:-${default_python}}
cmake \
  -S "${project_root}" \
  -B "${build_root}" \
  -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build "${build_root}" --target intent-compile

(
  cd /tmp
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${project_root}/python:${project_root}/examples" \
  "${python_bin}" "${project_root}/examples/repro/${backend}/main.py" "${kernel}" \
    --compiler "${build_root}/tools/intent-compile/intent-compile" \
    --baseline-source "${project_root}/${baseline}"
)
