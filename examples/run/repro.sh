#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <triton|cutile|tilelang> <kernel>" >&2
  exit 2
fi

backend=$1
kernel=$2
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build_root=${INTENT_BUILD_ROOT:-/tmp/intentdsl-build}
cmake_generator=${INTENT_CMAKE_GENERATOR:-Ninja}
mlir_dir=${INTENT_MLIR_DIR:-/usr/lib/llvm-20/lib/cmake/mlir}
llvm_dir=${INTENT_LLVM_DIR:-/usr/lib/llvm-20/lib/cmake/llvm}

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
unfamiliar=false
case "${kernel}" in
  histogram | csr_spmv | radix2_fft | bitonic_sort | kmeans_assign | \
  viterbi_decode | smith_waterman | greedy_nms | roi_align | barrier_option | \
  nonzero_compact | unique_consecutive | moe_align_block | nested_ragged_pool | \
  adamw_update | adafactor_update | reshape_and_cache | \
  group_norm_silu_backward | batched_cholesky | batched_householder_qr | \
  causal_conv1d_update | \
  batch_norm_training | csr_spmm | max_pool2d | softmax_backward | \
  triangular_solve | \
  variant_gemm_loop_interchange | variant_softmax_online | \
  variant_online_softmax_inline | variant_attention_inline | \
  variant_attention_select | variant_rope_index | variant_swiglu_helper | \
  variant_layer_norm_second_moment | variant_conv2d_reduce_order | \
  variant_transpose_scalar_domains | variant_moe_product_domain | \
  variant_adamw_split_pipeline | variant_reshape_cache_split | \
  variant_nested_ragged_identity | variant_nested_ragged_split | \
  variant_cholesky_right_looking | variant_adafactor_scalar_product | \
  variant_transpose_product_domain | variant_ordered_prefix_nested | \
  variant_attention_full_causal_stream)
    unfamiliar=true
    ;;
esac

if [[ ${unfamiliar} == false ]]; then
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
  triton:paged_attention)
    baseline=source/triton/xformers/attention/splitk/splitk_kernels_runtime.py
    ;;
  triton:paged_mla_decode | cutile:paged_mla_decode | tilelang:paged_mla_decode)
    ;;
  triton:paged_splitk_attention | cutile:paged_splitk_attention | tilelang:paged_splitk_attention)
    ;;
  triton:moe)
    baseline=source/triton/triton/gemm/grouped/08-grouped-gemm.py
    ;;
  triton:layer_norm)
    baseline=source/triton/flash-attention/normalization/layer_norm/layer_norm_runtime.py
    ;;
  triton:cross_entropy)
    baseline=source/triton/liger-kernel/loss/cross_entropy/cross_entropy_runtime.py
    ;;
  triton:layer_norm_backward)
    baseline=source/triton/triton/normalization/layer_norm/05-layer-norm.py
    ;;
  triton:attention_backward | cutile:attention_backward | tilelang:attention_backward)
    ;;
  triton:varlen_gqa_decode_logits | cutile:varlen_gqa_decode_logits | tilelang:varlen_gqa_decode_logits)
    ;;
  triton:sparse_2to4_gemm | cutile:sparse_2to4_gemm)
    ;;
  tilelang:sparse_2to4_gemm)
    baseline=source/tilelang/tilelang/gemm/sparse_2to4/example_gemm_sp.py
    ;;
  triton:causal_conv1d_backward | cutile:causal_conv1d_backward | tilelang:causal_conv1d_backward)
    ;;
  triton:block_sparse_attention | cutile:block_sparse_attention | tilelang:block_sparse_attention)
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
  triton:embedding_backward_atomic)
    baseline=source/triton/liger-kernel/embedding/lookup/embedding.py
    ;;
  triton:embedding_forward_lookup)
    baseline=source/triton/liger-kernel/embedding/lookup/embedding.py
    ;;
  triton:index_select_rows)
    baseline=source/triton/xformers/indexing/index_select_cat/k_index_select_cat.py
    ;;
  triton:scaled_index_add)
    baseline=source/triton/xformers/indexing/scaled_index_add/k_scaled_index_add.py
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
  cutile:rope_qk_full | cutile:rope_qk_partial | cutile:rope_qk_inverse)
    baseline=source/cutile/tilegym/position/rope/rope.py
    ;;
  triton:rope_qk_full | triton:rope_qk_partial | triton:rope_qk_inverse | \
  tilelang:rope_qk_full | tilelang:rope_qk_partial | tilelang:rope_qk_inverse)
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
  tilelang:varlen_gqa_prefill)
    baseline=source/tilelang/tilelang/attention/flash_forward_varlen/example_gqa_fwd_varlen.py
    ;;
  cutile:mla_prefill)
    baseline=source/cutile/tilegym/attention/mla/mla.py
    ;;
  triton:bf16_gemm | triton:batched_gemm | triton:quantized_gemm | triton:logsumexp | \
  cutile:attention_bias | cutile:paged_attention | cutile:quantized_gemm | cutile:rms_norm | cutile:fused_add_rms_norm | cutile:dropout_residual_rms_norm | cutile:logsumexp | \
  cutile:layer_norm_backward | cutile:swiglu_backward | cutile:cross_entropy | \
  tilelang:attention_bias | tilelang:paged_attention | tilelang:batched_gemm | tilelang:layer_norm | tilelang:layer_norm_backward | tilelang:fused_add_rms_norm | tilelang:dropout_residual_rms_norm | tilelang:logsumexp | \
  tilelang:quantized_gemm | tilelang:swiglu_forward | tilelang:swiglu_backward | tilelang:cross_entropy | \
  triton:varlen_attention | cutile:varlen_attention | triton:varlen_gqa_prefill | cutile:varlen_gqa_prefill | \
  triton:varlen_gqa_rope_prefill | cutile:varlen_gqa_rope_prefill | tilelang:varlen_gqa_rope_prefill | \
  triton:dropout_residual_rms_norm | triton:sorted_nucleus_cutoff | \
  cutile:sorted_nucleus_cutoff | tilelang:sorted_nucleus_cutoff | \
  triton:insertion_top_k | cutile:insertion_top_k | tilelang:insertion_top_k | \
  cutile:embedding_backward_atomic | tilelang:embedding_backward_atomic)
    ;;
  triton:conv1d | cutile:conv1d | tilelang:conv1d | \
  triton:causal_conv1d | cutile:causal_conv1d | tilelang:causal_conv1d | \
  triton:conv2d | cutile:conv2d | tilelang:conv2d)
    ;;
  triton:selective_scan | cutile:selective_scan | tilelang:selective_scan)
    ;;
  triton:mamba_chunk_scan | cutile:mamba_chunk_scan | tilelang:mamba_chunk_scan)
    ;;
  triton:splitk_attention_reduce | tilelang:splitk_attention_reduce)
    ;;
  cutile:splitk_attention_reduce)
    baseline=source/cutile/tilegym/attention/flash_decode/splitk_reduce.py
    ;;
  triton:mla_prefill | cutile:mla_prefill | tilelang:mla_prefill | \
  triton:absorbed_mla_prefill | cutile:absorbed_mla_prefill | tilelang:absorbed_mla_prefill | \
  triton:mla_head_projection | cutile:mla_head_projection | tilelang:mla_head_projection | \
  triton:token_sparse_mla_prefill | cutile:token_sparse_mla_prefill | tilelang:token_sparse_mla_prefill | \
  triton:fp8_mqa_logits | cutile:fp8_mqa_logits | tilelang:fp8_mqa_logits)
    ;;
  cutile:embedding_forward_lookup | tilelang:embedding_forward_lookup)
    ;;
  triton:continuous_gqa_decode | cutile:continuous_gqa_decode | tilelang:continuous_gqa_decode)
    ;;
  triton:weight_only_int4 | cutile:weight_only_int4 | tilelang:weight_only_int4)
    ;;
  triton:block_scaled_matmul | tilelang:block_scaled_matmul)
    ;;
  cutile:block_scaled_matmul)
    baseline=source/cutile/cutile-python/gemm/block_scaled/BlockScaledMatMul.py
    ;;
  triton:w4a8_packed | cutile:w4a8_packed | \
  triton:fp8_gemm | cutile:fp8_gemm)
    ;;
  tilelang:w4a8_packed)
    baseline=source/tilelang/tilelang/gemm/dequantize_w4a8/example_dequant_gemm_w4a8.py
    ;;
  tilelang:fp8_gemm)
    baseline=source/tilelang/tilelang/gemm/fp8/example_tilelang_gemm_fp8.py
    ;;
  triton:shifted_row_copy | cutile:shifted_row_copy | tilelang:shifted_row_copy | \
  triton:grouped_query_head_add | cutile:grouped_query_head_add | tilelang:grouped_query_head_add | \
  triton:scalar_table_lookup | cutile:scalar_table_lookup | tilelang:scalar_table_lookup | \
  cutile:index_select_rows | tilelang:index_select_rows | \
  cutile:scaled_index_add | tilelang:scaled_index_add)
    ;;
  triton:matrix_transpose | cutile:matrix_transpose | tilelang:matrix_transpose)
    ;;
  triton:boolean_reduction | cutile:boolean_reduction | tilelang:boolean_reduction)
    ;;
  triton:value_select | cutile:value_select | tilelang:value_select)
    ;;
  triton:batched_row_affine | cutile:batched_row_affine | tilelang:batched_row_affine)
    ;;
  triton:atomic_compare_exchange | cutile:atomic_compare_exchange)
    ;;
  triton:record_fields | cutile:record_fields | tilelang:record_fields)
    ;;
  triton:scalar_while | cutile:scalar_while | tilelang:scalar_while)
    ;;
  triton:ordered_prefix | cutile:ordered_prefix | tilelang:ordered_prefix)
    ;;
    *)
      echo "unsupported repro: ${backend}:${kernel}" >&2
      exit 2
      ;;
  esac
fi

python_bin=${INTENT_PYTHON:-${default_python}}
cmake \
  -S "${project_root}" \
  -B "${build_root}" \
  -G "${cmake_generator}" \
  -DMLIR_DIR="${mlir_dir}" \
  -DLLVM_DIR="${llvm_dir}"
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
