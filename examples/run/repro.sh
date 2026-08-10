#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <triton|cutile|tilelang> <softmax|layer_norm|layer_norm_backward|rms_norm|fused_add_rms_norm|logsumexp|gemm|bf16_gemm|batched_gemm|quantized_gemm|dual_gemm|attention|attention_bias|varlen_attention|online_softmax|moe|grouped_gemm|swiglu_forward|swiglu_backward>" >&2
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

baseline=
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
  triton:attention_bias)
    baseline=source/triton/flash-attention/attention/fused/flash_attn_triton_runtime.py
    ;;
  triton:moe)
    baseline=source/triton/triton/gemm/grouped/08-grouped-gemm.py
    ;;
  triton:layer_norm)
    baseline=source/triton/flash-attention/normalization/layer_norm/layer_norm_runtime.py
    ;;
  triton:layer_norm_backward)
    baseline=source/triton/triton/normalization/layer_norm/05-layer-norm.py
    ;;
  triton:rms_norm)
    baseline=source/triton/liger-kernel/normalization/rms_norm/rms_norm_runtime.py
    ;;
  triton:fused_add_rms_norm)
    baseline=source/triton/liger-kernel/normalization/fused_add_rms_norm/fused_add_rms_norm_runtime.py
    ;;
  triton:dual_gemm)
    baseline=source/triton/triton/gemm/dense/03-matrix-multiplication.py
    ;;
  triton:grouped_gemm)
    baseline=source/triton/triton/gemm/grouped/08-grouped-gemm.py
    ;;
  triton:online_softmax)
    baseline=source/triton/triton/normalization/softmax/02-fused-softmax.py
    ;;
  triton:swiglu_backward)
    baseline=source/triton/liger-kernel/activation/swiglu/swiglu_runtime.py
    ;;
  triton:swiglu_forward)
    baseline=source/triton/liger-kernel/activation/swiglu/swiglu_runtime.py
    ;;
  cutile:softmax)
    baseline=source/cutile/tilegym/normalization/softmax/softmax.py
    ;;
  cutile:gemm)
    baseline=source/cutile/tilegym/gemm/dense/matmul.py
    ;;
  cutile:bf16_gemm)
    baseline=source/cutile/tilegym/gemm/dense/matmul.py
    ;;
  cutile:batched_gemm)
    baseline=source/cutile/tilegym/gemm/batched/bmm.py
    ;;
  cutile:attention)
    baseline=source/cutile/tilegym/attention/dense/attention_runtime.py
    ;;
  cutile:moe)
    baseline=source/cutile/cutile-python/moe/fused/MoE.py
    ;;
  cutile:layer_norm)
    baseline=source/cutile/cutile-python/normalization/layer_norm/LayerNorm.py
    ;;
  cutile:dual_gemm)
    baseline=source/cutile/tilegym/gemm/dense/matmul.py
    ;;
  cutile:grouped_gemm)
    baseline=source/cutile/tilegym/gemm/grouped/group_gemm.py
    ;;
  cutile:online_softmax)
    baseline=source/cutile/tilegym/normalization/softmax/softmax.py
    ;;
  cutile:swiglu_forward)
    baseline=source/cutile/tilegym/activation/silu_and_mul/silu_and_mul.py
    ;;
  tilelang:softmax)
    baseline=source/tilelang/tilelang/normalization/online_softmax/online_softmax.py
    ;;
  tilelang:gemm)
    baseline=source/tilelang/tilelang/gemm/dense/example_gemm.py
    ;;
  tilelang:bf16_gemm)
    baseline=source/tilelang/tilelang/gemm/dense/example_gemm.py
    ;;
  tilelang:attention)
    baseline=source/tilelang/tilelang/attention/flash_forward_bshd/example_mha_fwd_bshd.py
    ;;
  tilelang:moe)
    baseline=source/tilelang/tilelang/gemm/grouped/example_grouped_gemm_fwd.py
    ;;
  tilelang:rms_norm)
    baseline=source/tilelang/tilelang/normalization/rms_norm/rms_norm.py
    ;;
  tilelang:dual_gemm)
    baseline=source/tilelang/tilelang/gemm/dense/example_gemm.py
    ;;
  tilelang:grouped_gemm)
    baseline=source/tilelang/tilelang/gemm/grouped/example_grouped_gemm_fwd.py
    ;;
  tilelang:online_softmax)
    baseline=source/tilelang/tilelang/normalization/online_softmax/online_softmax.py
    ;;
  tilelang:varlen_attention)
    baseline=source/tilelang/tilelang/attention/flash_forward_varlen/example_gqa_fwd_varlen.py
    ;;
  triton:bf16_gemm | triton:batched_gemm | triton:quantized_gemm | triton:logsumexp | \
  cutile:attention_bias | cutile:quantized_gemm | cutile:rms_norm | cutile:fused_add_rms_norm | cutile:logsumexp | \
  cutile:layer_norm_backward | cutile:swiglu_backward | \
  tilelang:attention_bias | tilelang:batched_gemm | tilelang:layer_norm | tilelang:layer_norm_backward | tilelang:fused_add_rms_norm | tilelang:logsumexp | \
  tilelang:quantized_gemm | tilelang:swiglu_forward | tilelang:swiglu_backward | \
  triton:varlen_attention | cutile:varlen_attention)
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

runner_arguments=(
  "${kernel}"
  --compiler "${build_root}/tools/intent-compile/intent-compile"
)
if [[ -n "${baseline}" ]]; then
  runner_arguments+=(--baseline-source "${project_root}/${baseline}")
fi

(
  cd /tmp
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${project_root}/python:${project_root}/examples" \
  "${python_bin}" "${project_root}/examples/repro/${backend}/main.py" \
    "${runner_arguments[@]}"
)
