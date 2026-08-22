# Intent examples

`examples/` 是 Intent DSL 算法源码与可执行复现入口，不保存上游实现或环境。目录职责固定为：

- `kernels/`：作者写下的目标无关算法；按算法职责分层，不按 target language 分叉。
- `repro/common/`：V1 与 V2 共用的数值、计时和 artifact 执行支持。
- `repro/{triton,cutile,tilelang}/`：冻结的 V1 provider 接线。
- `repro/v2/`：三家 source inventory（Triton 38、cuTile 37、TileLang 37）、结构化测量与 provider ABI adapter；没有同语义 DSL 的 entry 在比较前明确拒绝，不用相似算法冒充。
- `run/`：人工可执行入口。

`V1` 指 `report/baseline/` 冻结矩阵实际使用过的算法；`V1+V2` 指 V2 继续复用同一算法构造；`V2` 指为了和新 source 的算法边界对齐而新增的 DSL。相似但算法不同的 entry 不会因为名称相近而合并。

## V1 算法清单

| algorithm group | V1 kernel / case |
|---|---|
| normalization/softmax-rowwise | `softmax`, `softmax_backward` |
| normalization/softmax-online | `online_softmax`, `variant_softmax_online`, `variant_online_softmax_inline` |
| normalization/layer-norm | `layer_norm`, `layer_norm_backward`, `variant_layer_norm_second_moment` |
| normalization/rms-norm | `rms_norm`, `fused_add_rms_norm`, `dropout_residual_rms_norm` |
| normalization/batch-norm | `batch_norm_training` |
| normalization/group-norm | `group_norm_silu_backward`, `group_norm_backward` |
| loss/logsumexp | `logsumexp` |
| loss/cross-entropy | `cross_entropy` |
| activation/swiglu | `swiglu_forward`, `swiglu_backward`, `variant_swiglu_helper` |
| contraction/dense | `gemm` base/tail, `bf16_gemm`, `batched_gemm` NN/TN/NT/TT, `dual_gemm`, `variant_gemm_loop_interchange` |
| contraction/low-precision | `quantized_gemm`, `weight_only_int4`, `w4a8_packed`, `block_scaled_matmul`, `fp8_gemm` e4m3/e5m2, `sparse_2to4_gemm` |
| convolution/direct | `conv1d`, `conv2d`, `variant_conv2d_reduce_order` |
| convolution/causal | `causal_conv1d`, `causal_conv1d_update`, `causal_conv1d_backward` |
| attention/dense-forward | `attention`, `attention_bias`, `variant_attention_inline`, `variant_attention_select`, `variant_attention_full_causal_stream` |
| attention/variable-length | `varlen_attention` causal/noncausal, `varlen_gqa_prefill`, `varlen_gqa_rope_prefill` |
| attention/decode | `paged_attention`, `continuous_gqa_decode`, `splitk_attention_reduce`, `paged_splitk_attention`, `varlen_gqa_decode_logits` |
| attention/mla | `mla_prefill`, `mla_head_projection`, `absorbed_mla_prefill`, `token_sparse_mla_prefill`, `paged_mla_decode`, `fp8_mqa_logits` |
| attention/block-sparse | `block_sparse_attention` |
| attention/backward | `attention_backward` |
| moe | `moe`, `grouped_gemm` base/tail/empty-groups, `moe_align_block`, `variant_moe_product_domain` |
| scan/prefix | `ordered_prefix`, `variant_ordered_prefix_nested` |
| scan/state-space | `selective_scan`, `mamba_chunk_scan` |
| position/rope | `rope_qk_full`, `rope_qk_partial`, `rope_qk_inverse`, `variant_rope_index` |
| indexing/embedding | `embedding_forward_lookup`, `embedding_backward_atomic` |
| indexing/gather-scatter | `shifted_row_copy`, `grouped_query_head_add`, `scalar_table_lookup`, `index_select_rows`, `scaled_index_add` |
| cache/kv | `reshape_and_cache`, `variant_reshape_cache_split` |
| layout/transpose | `matrix_transpose`, `variant_transpose_scalar_domains`, `variant_transpose_product_domain` |
| pointwise/broadcast | `batched_row_affine`, `value_select` |
| reduction/statistics | `boolean_reduction`, `histogram`, `atomic_compare_exchange` |
| selection/sort-sample | `sorted_nucleus_cutoff`, `insertion_top_k`, `bitonic_sort` |
| compaction | `nonzero_compact`, `unique_consecutive` |
| sparse/csr | `csr_spmv`, `csr_spmm` |
| ragged/pooling | `jagged_mean`, `nested_ragged_pool`, `variant_nested_ragged_identity`, `variant_nested_ragged_split` |
| optimization/update | `adamw_update`, `variant_adamw_split_pipeline`, `adafactor_update`, `variant_adafactor_scalar_product` |
| factorization/solve | `batched_cholesky`, `variant_cholesky_right_looking`, `batched_householder_qr`, `triangular_solve` |
| vision/pooling | `max_pool2d`, `max_pool2d_with_indices` |
| vision/roi-align | `roi_align` |
| vision/nms | `greedy_nms` |
| dynamic-programming | `smith_waterman`, `viterbi_decode`, `record_fields`, `scalar_while` |
| spectral/fft | `radix2_fft` |
| clustering/assignment | `kmeans_assign` |
| simulation/monte-carlo | `barrier_option` |

V1 中的 `variant_*` 只证明同一算法的等价 DSL 分解可以编译，不进入 V2 的独立性能 entry 计数。

## V2 Triton entries

| 集合 | entry | model case | DSL |
|---|---|---|---|
| V1+V2 | fused softmax | `8192x8192-fp16` | `normalization/softmax.py:stable_softmax_f16` |
| V1+V2 | dense GEMM | `M4096-N14336-K4096-fp16` | `contraction/gemm.py:gemm` |
| V1+V2 | grouped GEMM | `E8-M1024-K4096-N14336-fp16` | `ragged/grouped_gemm.py:ragged_grouped_gemm` |
| V1+V2 | FlashAttention forward | `B4-H32-S4096-D128-fp16-causal` | `streaming/attention.py:flash_attention_fwd` |
| V1+V2 | LayerNorm | `8192x4096-fp16` | `normalization/layer_norm.py:layer_norm_f16` |
| V2 | FlashAttention cross entropy | `8192x32768-bf16` | `loss/cross_entropy.py:flash_cross_entropy_bf16` |
| V2 | FlashAttention LayerNorm | `8192x4096-bf16` | `normalization/layer_norm.py:layer_norm_bf16` |
| V2 | rotary embedding | `B4-S4096-H32-D128-bf16` | `position/rope.py:rotary_embedding_bf16` |
| V1+V2 | SwiGLU | `8192x14336-bf16` | `activation/swiglu.py:swiglu_forward` |
| V1+V2 | embedding lookup | `tokens16384-vocab32768-hidden4096-bf16` | `backward/embedding.py:embedding_forward_lookup_bf16` |
| V1+V2 | cross entropy | `8192x32768-bf16` | `loss/cross_entropy.py:fused_cross_entropy_bf16` |
| V2 | padded RoPE/cache update | `B32-QH32-KVH8-D128-cache8192-fp16` | `position/rope_cache.py:padded_rope_cache_update` |
| V1+V2 | fused add RMSNorm | `8192x4096-bf16` | `normalization/fused_add_rms_norm.py:fused_add_rms_norm` |
| V1+V2 | RMSNorm | `8192x4096-bf16` | `normalization/rms_norm.py:rms_norm_bf16` |
| V2 | xFormers RMSNorm | `8192x4096-bf16` | `normalization/rms_norm.py:rms_norm_bf16` |
| V2 | QKV projection pipeline | `M4096-K4096-QKV4096-fp16` | `contraction/gemm.py:gemm`，作者在外层调用三次 |
| V1+V2 | index select | `source65536-selected32768-hidden4096-fp16` | `indexing/relations.py:index_select_rows` |
| V1+V2 | scaled index add | `destination65536-routed32768-hidden4096-fp16` | `indexing/relations.py:scaled_index_add_unique` |
| V2 | scaled FP8 split-K GEMM | `M4096-N14336-K4096-e4m3-split4` | `contraction/block_scaled.py:scaled_fp8_splitk_matmul` |
| V1+V2 | jagged mean | `B512-T33024-D128-fp32` | `ragged/jagged_mean.py:jagged_mean` |
| V2 | FP8 groupwise quantize | `8192x4096-group128-bf16-e4m3` | `quantization/fp8.py:bf16_groupwise_fp8_quantize` |
| V1+V2 | RoPE Q/K | `B4-S4096-QH32-KVH8-D128-fp16` | `position/rope.py:rotary_qk_inplace` |
| V1+V2 | causal Conv1D | `B4-C4096-S4096-W4-bf16-silu` | `convolution/direct.py:causal_depthwise_conv1d_bf16` |
| V2 | varlen causal Conv1D | `lengths2048-1536-1024-512-D4096-W4-bf16` | `convolution/varlen.py:varlen_aligned_causal_depthwise_conv1d+varlen_causal_conv1d_final_state` |
| V1+V2 | causal Conv1D update | `B32-D4096-W4-bf16` | `convolution/direct.py:causal_depthwise_conv1d_update_bf16` |
| V2 | modern FlashAttention forward | `B2-H16-S2048-D128-fp16-causal` | `streaming/attention.py:flash_attention_fwd` |
| V2 | Mamba3 SISO step | `B32-HQK4-H16-DQK32-DV64-bf16` | `streaming/mamba.py:mamba3_siso_step` |
| V1+V2 | FlashAttention backward | `B2-H16-S2048-D128-fp16-causal` | `backward/attention.py:attention_backward_delta+attention_backward_dkdv+attention_backward_dq` |
| V2 | MoE expert projection | `T2048-E8-top2-4096x14336-bf16` | `ragged/grouped_gemm.py:routed_expert_projection_bf16` |
| V2 | MoE split-K expert projection | same contract | `ragged/grouped_gemm.py:routed_expert_projection_bf16` |
| V2 | Mamba chunk state | `B1-S2048-H32-G8-P64-N128-C256-bf16` | `streaming/mamba.py:mamba_chunk_state_bf16_fwd` |
| V2 | Mamba state passing | `B1-C8-H32-state8192-fp32` | `streaming/mamba.py:mamba_state_passing_fwd` |
| V1+V2 | Mamba chunk scan | `B1-S2048-H32-G8-P64-N128-C256-bf16` | `streaming/selective_scan.py:mamba_chunk_scan_bf16_fwd` |
| V2 | Mamba3 SISO sequence forward | `B1-S2048-HQK4-H16-DQK32-DV64-bf16` | — |
| V1+V2 | paged GQA decode | `B16-QH32-KVH8-S8192-D128-page16-fp16` | `streaming/paged_attention.py:paged_gqa_decode_attention` |
| V2 | split-K paged attention | `B16-QH32-KVH8-S8192-D128-page16-split8-fp16` | `streaming/paged_attention.py:paged_gqa_decode_partials+streaming/splitk_reduce.py:splitk_attention_weighted_sum_reduce` |
| V1+V2 | paged MLA decode | `B8-QH128-KVH1-S8192-C512-R64-page16-fp16` | `streaming/mla.py:paged_mla_decode` |
| V1+V2 | block-sparse GQA decode | `B8-QH32-KVH8-S8192-D128-blocks32x128` | `streaming/block_sparse_attention.py:block_sparse_gqa_decode_partials+block_sparse_gqa_decode_combine` |

## V2 cuTile entries

| 集合 | entry | model case | DSL |
|---|---|---|---|
| V2 | official FMHA | `B4-QH32-KVH8-S4096-D128-fp16-causal` | `streaming/attention.py:flash_gqa_attention_fwd` |
| V1+V2 | block-scaled GEMM | `M4096-N14336-K4096-fp8-block32` | `contraction/block_scaled.py:block_scaled_matmul` |
| V1+V2 | dense GEMM | `M4096-N14336-K4096-fp16` | `contraction/gemm.py:gemm` |
| V2 | MoE expert projection | `T4096-E8-top2-4096x14336-bf16` | `ragged/grouped_gemm.py:routed_expert_projection_bf16` |
| V1+V2 | LayerNorm | `8192x4096-bf16` | `normalization/layer_norm.py:layer_norm_bf16` |
| V2 | SiLU-and-mul | `4096x28672-to14336-bf16` | `activation/swiglu.py:silu_and_mul_packed` |
| V1+V2 | attention backward | `B2-QH8-KVH2-S1024-D64-fp16-causal` | `backward/attention.py:attention_backward_delta+attention_backward_dkdv+attention_backward_dq` |
| V2 | dense attention forward | `B2-QH32-KVH8-S4096-D128-bf16-causal` | `streaming/attention.py:flash_gqa_attention_fwd` |
| V2 | grouped flash decode | `B8-QH32-KVH8-S8192-D128-bf16` | — |
| V1+V2 | SwiGLU | `8192x14336-bf16` | `activation/swiglu.py:swiglu_forward` |
| V1+V2 | split-K attention reduce | `B8-H32-S8192-splits16-D128` | `streaming/splitk_reduce.py:splitk_attention_reduce` |
| V1+V2 | MLA prefill | `B1-QH128-KVH1-S2048-D128-R64-fp16` | `streaming/attention.py:mla_prefill` |
| V1+V2 | batched GEMM | `B32-M512-N512-K1024-bf16` | `contraction/batched_gemm.py:batched_gemm_nn` |
| V2 | TileGym dense GEMM | `M8192-N11008-K4096-bf16` | `contraction/gemm.py:gemm` |
| V1+V2 | grouped GEMM | `rows256-512-1024-2048-K4096-N4096-bf16` | `ragged/grouped_gemm.py:ragged_grouped_gemm_bf16` |
| V1+V2 | MoE alignment | `T4096-top2-E64` | `routing/moe_align.py` four-stage pipeline |
| V2 | mHC GEMM/RMS scale | `T2048-H4096-streams4-bf16` | `routing/mhc.py:mhc_gemm_rms_scale` |
| V2 | chunked softmax | `8192x32768-bf16` | `normalization/softmax.py:chunked_softmax_bf16` |
| V1+V2 | RoPE Q/K | `B2-S4096-QH32-KVH8-D128-bf16` | `position/rope.py:rotary_qk_bf16_inplace` |
| V2 | GELU | `8192x4096-fp16-tanh` | `activation/pointwise.py:gelu_tanh` |
| V2 | GEGLU | `4096x28672-fp16-tanh` | `activation/pointwise.py:geglu_tanh` |
| V2 | ReLU | `8192x4096-fp16` | `activation/pointwise.py:relu_forward` |
| V2 | dropout | `8192x4096-fp16-p0.1` | `regularization/dropout.py:xor_shift_dropout` |
| V2 | attention-sink prefill | `B1-S4096-QH32-KVH8-D128-bf16` | `streaming/attention_specialized.py:attention_sink_prefill` |
| V2 | attention-sink decode | `B32-S8192-QH32-KVH8-D128-bf16` | — |
| V2 | mHC residual | `T2048-H4096-streams4-bf16` | `routing/mhc.py:mhc_apply_residual` |
| V2 | Gemma prefill | `B2-S4096-QH32-KVH8-D128-window1024-cap50-bf16` | `streaming/attention_specialized.py:gemma_gqa_prefill` |
| V2 | Gemma split-K decode | `B32-S8192-QH32-KVH8-D128-window1024-cap50-bf16` | — |
| V2 | mHC Sinkhorn | `T8192-streams4-fp32` | `routing/mhc.py:mhc_sinkhorn` |
| V2 | absorbed MLA decode | `B8-H64-S8192-C512-R64-fp16` | `streaming/mla.py:absorbed_mla_decode` |
| V2 | split-K MLA decode | `B8-H64-S8192-C512-R64-split512-fp16` | `streaming/mla.py:splitk_mla_decode_partials+streaming/splitk_reduce.py:splitk_attention_reduce_f16` |
| V1+V2 | sparse MLA prefill | `S2048-SKV4096-H64-topk512-D128-R64-bf16` | `streaming/mla.py:token_sparse_mla_value_prefill` |
| V2 | sliding-window attention | `B2-S4096-QH32-KVH8-D128-window1024-fp16` | `streaming/attention_specialized.py:sliding_window_gqa_prefill` |
| V1+V2 | RMSNorm | `8192x4096-bf16` | `normalization/rms_norm.py:rms_norm_bf16` |
| V2 | recurrent gated delta | `B2-S2048-H8-K128-V128-bf16` | `streaming/gated_delta.py:recurrent_gated_delta_fwd` |
| V2 | chunk gated delta | `B2-S2048-H8-K128-V128-C64-bf16` | — |
| V2 | NVFP4 quantization | `8192x4096-block32-bf16` | — |

## V2 TileLang entries

| 集合 | entry | model case | DSL |
|---|---|---|---|
| V1+V2 | block-sparse GQA decode | `B8-QH32-KVH8-S8192-D128-blocks128` | `streaming/block_sparse_attention.py:block_sparse_gqa_decode_partials+block_sparse_gqa_decode_combine` |
| V1+V2 | dense FlashAttention | `B4-S4096-H32-D128-fp16-causal` | `streaming/attention.py:flash_attention_fwd` |
| V1+V2 | varlen GQA prefill | `packed8-QH32-KVH8-D128-fp16-causal` | `streaming/attention.py:flash_varlen_gqa_prefill` |
| V1+V2 | GQA decode | `B32-QH32-KVH8-S8192-D128-fp16` | `streaming/attention.py:continuous_gqa_decode` |
| V1+V2 | varlen GQA decode logits | `B16-QH32-KVH8-S4096-D64-fp16` | `streaming/attention.py:varlen_gqa_decode_with_sink_logits` |
| V1+V2 | paged MLA decode | `B32-QH128-KVH1-S8192-V128-R64-page64-fp16` | `streaming/mla.py:paged_mla_decode` |
| V2 | persistent MLA decode | `B8-H64-S16384-C128-R64-split4-fp16` | — |
| V1+V2 | Conv2D | `NHWC32x128x128x256-K3x3x256x512-fp16` | `convolution/direct.py:conv2d_nhwc` |
| V1+V2 | dense GEMM | `M4096-N14336-K4096-fp16` | `contraction/gemm.py:gemm` |
| V1+V2 | W4A8 GEMM | `M4096-N14336-K4096-int8-int4` | `contraction/weight_only_int4.py:w4a8_packed_matmul` |
| V1+V2 | FP8 GEMM | `M4096-N14336-K4096-e4m3` | `contraction/weight_only_int4.py:fp8_e4m3_matmul` |
| V1+V2 | grouped GEMM | `rows256-512-1024-2048-K4096-N4096-fp16` | `ragged/grouped_gemm.py:ragged_grouped_gemm` |
| V1+V2 | sparse 2:4 GEMM | `M8192-N14336-K8192-fp16` | `contraction/sparse_2to4.py:sparse_2to4_gemm` |
| V2 | DeepGEMM FP8 2×acc | `M4096-N4096-K4096-e4m3-bf16` | `contraction/block_scaled.py:deepgemm_fp8_2xacc` |
| V1+V2 | online softmax | `8192x8192-fp16` | `streaming/online_softmax.py:streamed_online_softmax_f16` |
| V2 | DeepSeek V3.2 top-k selector | `B32-S32768-topk2048-fp32` | — |
| V1+V2 | RMSNorm | `8192x4096-fp32` | `normalization/rms_norm.py:rms_norm_f32` |
| V1+V2 | Mamba chunk scan | `B1-S2048-H32-G8-P64-N128-C256-fp16` | `streaming/selective_scan.py:mamba_chunk_scan_fwd` |
| V2 | Mamba chunk state | `B1-S2048-H32-G8-P64-N128-C256-fp16` | `streaming/mamba.py:mamba_chunk_state_fwd` |
| V2 | linear attention forward | `B1-S2048-H16-D128-fp16` | `streaming/linear_attention.py:fused_chunk_linear_attention_fwd` |
| V2 | linear attention backward | `B1-S2048-H16-D128-fp16` | — |
| V2 | retention forward | `B1-S2048-H16-D128-fp16` | `streaming/linear_attention.py:chunk_retention_fwd` |
| V2 | mHC pre | `T2048-H4096-streams4-bf16` | `routing/mhc.py:mhc_pre_gemm_sqrsum+mhc_pre_fuse` |
| V2 | block-causal attention | `B2-S4096-H16-D128-block64-fp16` | `streaming/attention_specialized.py:block_causal_attention_fwd` |
| V2 | varlen block-causal attention | `lengths4096-3840-3584-3328-H16-D128-fp16` | `streaming/attention_specialized.py:varlen_block_causal_attention_fwd` |
| V2 | native sparse attention forward | `B2-S4096-QH32-KVH4-D128-blocks64` | `streaming/attention_specialized.py:native_sparse_attention_fwd` |
| V2 | native sparse attention decode | `B8-S8192-QH32-KVH2-D128-blocks32` | `streaming/attention_specialized.py:native_sparse_attention_fwd` |
| V1+V2 | GQA attention backward | `B1-S4096-QH32-KVH8-D64-fp16-causal` | `backward/attention.py` three-stage pipeline |
| V2 | sparse MLA backward | `B1-S4096-SKV8192-H64-D576-topk2048-bf16` | — |
| V1+V2 | FP8 lighting indexer | `S4096-SKV8192-H32-D64-fp8` | `routing/mqa_logits.py:fp8_mqa_logits` |
| V2 | per-token FP8 | `8192x8192-group128-f32-e4m3` | `quantization/fp8.py:f32_groupwise_fp8_quantize` |
| V2 | block-sparse GEMM | `M4096-N4096-K4096-block128x128x32-50pct` | `contraction/block_sparse.py:block_sparse_matmul` |
| V2 | grouped GEMM backward | `rows256-512-1024-2048-K4096-N4096-fp16` | `ragged/grouped_gemm.py:ragged_grouped_gemm_backward_weight` |
| V2 | BitNet int8×packed-int2 decode | `M1-N4096-K4096-int8-int2` | — |
| V2 | BF16×FP4 dequant GEMM | `M4096-N4096-K4096-bf16-fp4` | — |
| V2 | block FP4 activation quantization | `8192x4096-block32-bf16` | — |
| V2 | mHC post | `T4096-H2560-streams4-bf16` | `routing/mhc.py:mhc_apply_residual` |

## 运行

单条 V1 repro：

```bash
./examples/run/repro.sh triton softmax
```

单条或整组 V2；第二个参数是该设备对应的 CSV：

```bash
./examples/run/baseline-v2.sh triton report/baseline-new/triton-5090.csv fused_softmax
./examples/run/baseline-v2.sh triton report/baseline-new/triton-5090.csv
```

V2 runner 只从 `repro/v2/registry.py` 取 entry 顺序；provider adapter 只处理 source ABI、输入 view、multi-kernel launch 与数值/计时接线，不改变 DSL 算法。输入、输出和 workspace 在计时前构造，source 首次 JIT/编译也在计时外；CSV 的两列时间都只覆盖已经准备好的单 kernel launch 或完整 multi-kernel pipeline launch。registry 中的 source 路径是可审计 inventory，真正的 callable 接线位于相邻 provider adapter。
