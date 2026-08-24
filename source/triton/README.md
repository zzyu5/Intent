# Triton source inventory

本目录只保存两类内容：冻结的 Baseline V1 实际使用过的上游源码，以及 Baseline V2 已登记、可由相邻 runtime 独立调用的高性能入口。表中的计数单位是一次可独立调用与计时的算法入口，不是文件数，也不是 `@triton.jit` 内部 helper 数。

运行前进入项目的 Triton Python 环境；命令均从仓库根目录执行。当前 registry 共 54 个入口：38 个原 baseline-v2 入口，加上 16 个继续保留历史来源边界的 baseline-v1 入口。

Upstream roots：`triton-lang/triton`、`Dao-AILab/flash-attention`、`linkedin/Liger-Kernel`、`facebookresearch/xformers`、`meta-pytorch/tritonbench`、`meta-pytorch/applied-ai`、`state-spaces/mamba`、`fla-org/flash-linear-attention`、`vllm-project/vllm`；V1-only 的 FlagGems 来源为 `FlagOpen/FlagGems`。

## Baseline V2 主清单（38）

| 集合 | entry | source / public boundary | 模型级输入 | runtime |
|---|---|---|---|---|
| V1+V2 | fused softmax | `triton/normalization/softmax/02-fused-softmax.py` / `softmax` | `8192×8192`, fp16 | `python source/triton/triton/normalization/softmax/02-fused-softmax_runtime.py` |
| V1+V2 | dense GEMM | `triton/gemm/dense/03-matrix-multiplication.py` / `matmul` | `4096×4096×14336`, fp16 | `python source/triton/triton/gemm/dense/03-matrix-multiplication_runtime.py` |
| V2 | grouped GEMM | `triton/gemm/grouped/08-grouped-gemm.py` / `group_gemm_fn` | 8 experts, `M=1024,K=4096,N=14336`, fp16 | `python source/triton/triton/gemm/grouped/08-grouped-gemm_runtime.py` |
| V1+V2 | FlashAttention forward | `triton/attention/fused/06-fused-attention.py` / `attention` | `B=4,H=32,S=4096,D=128`, fp16 causal | `python source/triton/triton/attention/fused/06-fused-attention_runtime.py` |
| V1+V2 | LayerNorm | `triton/normalization/layer_norm/05-layer-norm.py` / `layer_norm` | `8192×4096`, fp16 | `python source/triton/triton/normalization/layer_norm/05-layer-norm_runtime.py` |
| V2 | fused cross entropy | `flash-attention/loss/cross_entropy/cross_entropy.py` / forward+backward kernels | `8192×32768`, bf16 | `python source/triton/flash-attention/loss/cross_entropy/cross_entropy_runtime.py` |
| V1+V2 | fused LayerNorm family | `flash-attention/normalization/layer_norm/layer_norm.py` / `layer_norm_fn` | `8192×4096`, bf16 | `python source/triton/flash-attention/normalization/layer_norm/layer_norm_runtime.py` |
| V2 | rotary embedding | `flash-attention/position/rotary/rotary.py` / `apply_rotary` | `B=4,S=4096,H=32,D=128`, bf16 | `python source/triton/flash-attention/position/rotary/rotary_runtime.py` |
| V1+V2 | SwiGLU | `liger-kernel/activation/swiglu/swiglu.py` / `swiglu_forward` | `8192×14336`, bf16 | `python source/triton/liger-kernel/activation/swiglu/swiglu_runtime.py` |
| V1+V2 | embedding lookup | `liger-kernel/embedding/lookup/embedding.py` / `LigerEmbeddingFunction` | vocab `32768×4096`, tokens `8×2048`, bf16 | `python source/triton/liger-kernel/embedding/lookup/embedding_runtime.py` |
| V1+V2 | cross entropy | `liger-kernel/loss/cross_entropy/cross_entropy.py` / `LigerCrossEntropyFunction` | `8192×32768`, bf16 | `python source/triton/liger-kernel/loss/cross_entropy/cross_entropy_runtime.py` |
| V1+V2 | fused add RMSNorm | `liger-kernel/normalization/fused_add_rms_norm/fused_add_rms_norm.py` / forward function | `8192×4096`, bf16 | `python source/triton/liger-kernel/normalization/fused_add_rms_norm/fused_add_rms_norm_runtime.py` |
| V1+V2 | RMSNorm | `liger-kernel/normalization/rms_norm/rms_norm.py` / `rms_norm_forward` | `8192×4096`, bf16 | `python source/triton/liger-kernel/normalization/rms_norm/rms_norm_runtime.py` |
| V2 | RoPE | `liger-kernel/position/rope/rope.py` / `_triton_rope` | model Q/K sequence tensors, bf16 | `python source/triton/liger-kernel/position/rope/rope_runtime.py` |
| V1+V2 | split-K paged attention | `xformers/attention/splitk/splitk_kernels.py` / split-K callable | paged KV, `S=8192,D=128` | `python source/triton/xformers/attention/splitk/splitk_kernels_runtime.py` |
| V2 | tiled matmul pipeline | `xformers/gemm/tiled/tiled_matmul_kernels.py` / tiled matmul callable | `4096×4096`, three weights, fp16 | `python source/triton/xformers/gemm/tiled/tiled_matmul_kernels_runtime.py` |
| V1+V2 | index-select concat | `xformers/indexing/index_select_cat/k_index_select_cat.py` / forward entry | source `65536×4096`, selected `32768`, fp16 | `python source/triton/xformers/indexing/index_select_cat/k_index_select_cat_runtime.py` |
| V1+V2 | scaled index add | `xformers/indexing/scaled_index_add/k_scaled_index_add.py` / forward entry | destination `65536×4096`, routed `32768×4096`, fp16 | `python source/triton/xformers/indexing/scaled_index_add/k_scaled_index_add_runtime.py` |
| V2 | xFormers RMSNorm | `xformers/normalization/rms_norm/rmsnorm_kernels.py` / RMSNorm entry | `8192×4096`, bf16 | `python source/triton/xformers/normalization/rms_norm/rmsnorm_kernels_runtime.py` |
| V2 | padded RoPE/cache update | `xformers/position/rope_padded/rope_padded_kernels.py` / padded RoPE entry | `B=32,QH=32,KVH=8,D=128`, cache 8193, fp16 | `python source/triton/xformers/position/rope_padded/rope_padded_kernels_runtime.py` |
| V1+V2 | jagged mean | `tritonbench/ragged/jagged_mean/kernels.py` / jagged mean callable | model-scale ragged rows, fp32 | `python source/triton/tritonbench/ragged/jagged_mean/jagged_mean_runtime.py` |
| V2 | FP8 groupwise quantization | `meta-applied-ai/quantization/fp8_groupwise/float8_groupwise_quant.py` / `float8_groupwise_quantize` | `8192×4096`, group 128, bf16→e4m3 | `python source/triton/meta-applied-ai/quantization/fp8_groupwise/float8_groupwise_quant_runtime.py` |
| V2 | scaled FP8 split-K GEMM | `meta-applied-ai/gemm/fp8_scaled/scaled_fp8_gemm.py` / `scaled_mm_splitk` | `4096×4096×14336`, e4m3 | `python source/triton/meta-applied-ai/gemm/fp8_scaled/scaled_fp8_gemm_runtime.py` |
| V2 | causal Conv1D | `meta-applied-ai/convolution/causal_conv1d/causal_1d_conv.py` / `causal_conv1d_fwd` | `B=4,C=4096,S=4096,W=4`, bf16 | `python source/triton/meta-applied-ai/convolution/causal_conv1d/causal_1d_conv_runtime.py` |
| V2 | varlen causal Conv1D forward | `fla/conv/causal1d/ops.py` + `kernels.py` / `causal_conv1d_fwd` | packed lengths 2048/1536/1024/512, hidden 4096, width 4, bf16 | `python source/triton/fla/conv/causal1d/causal_conv_varlen_runtime.py` |
| V2 | causal Conv1D decode cache update | same source / `causal_conv1d_update` | decode batch 32, hidden 4096, width 4, bf16 | `python source/triton/fla/conv/causal1d/causal_conv_update_runtime.py` |
| V2 | modern FlashAttention forward | `meta-applied-ai/attention/flash_backward/flash_backward.py` / `flash` | `B=2,H=16,S=2048,D=128`, fp16 causal | `python source/triton/meta-applied-ai/attention/flash_backward/flash_forward_runtime.py` |
| V2 | modern FlashAttention backward | same source / `flash_bwd` | same shape, dQ/dK/dV | `python source/triton/meta-applied-ai/attention/flash_backward/flash_backward_runtime.py` |
| V2 | MoE grouped expert projection | `meta-applied-ai/moe/grouped/v0_moe_fused.py` / `invoke_fused_moe_kernel` | 2048 tokens, 8 experts, top-2, `4096→14336`, bf16 | `python source/triton/meta-applied-ai/moe/grouped/v0_moe_fused_runtime.py` |
| V2 | MoE split-K expert projection | `meta-applied-ai/moe/splitk/v1_moe_fused.py` / split-K invoke | same contract | `python source/triton/meta-applied-ai/moe/splitk/v1_moe_fused_runtime.py` |
| V2 | Mamba2 SSD chunk state | `state-spaces-mamba/mamba_ssm/ops/triton/ssd_chunk_state.py` / `_chunk_state_fwd` | `B=1,S=2048,H=32,P=64,G=8,N=128`, chunk 256, bf16 | `python source/triton/state-spaces-mamba/mamba_ssm/ops/triton/ssd_chunk_state_runtime.py` |
| V2 | Mamba2 SSD state passing | `state-spaces-mamba/mamba_ssm/ops/triton/ssd_state_passing.py` / `_state_passing_fwd` | 8 chunks, 32 heads, flattened state `64×128`, fp32 | `python source/triton/state-spaces-mamba/mamba_ssm/ops/triton/ssd_state_passing_runtime.py` |
| V2 | Mamba2 SSD chunk scan | `state-spaces-mamba/mamba_ssm/ops/triton/ssd_chunk_scan.py` / `_chunk_scan_fwd` | `B=1,S=2048,H=32,P=64,G=8,N=128`, chunk 256, bf16 | `python source/triton/state-spaces-mamba/mamba_ssm/ops/triton/ssd_chunk_scan_runtime.py` |
| V2 | Mamba3 SISO sequence forward | `state-spaces-mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py` / `mamba3_siso_fwd` | `B=1,S=2048,HQK=4,H=16,DQK=32,DV=64`, bf16 | `python source/triton/state-spaces-mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd_runtime.py` |
| V2 | Mamba3 SISO decode step | `state-spaces-mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_step.py` / `mamba3_siso_step` | decode batch 32, `HQK=4,H=16,DQK=32,DV=64`, bf16 | `python source/triton/state-spaces-mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_step_runtime.py` |
| V2 | paged GQA decode | `vllm/attention/paged_decode/triton_decode_attention.py` / `decode_attention_fwd` | `B=16,QH=32,KVH=8,S=8192,D=128`, page 16, fp16 | `python source/triton/vllm/attention/paged_decode/paged_gqa_decode_runtime.py` |
| V2 | paged MLA decode | same source / `decode_attention_fwd(..., is_mla=True)` | `B=8,QH=128,KVH=1,S=8192,latent=512,rope=64`, page 16, fp16 | `python source/triton/vllm/attention/paged_decode/paged_mla_decode_runtime.py` |
| V2 | MiniMax-M3 block-sparse GQA decode | `vllm/attention/minimax_m3/sparse_attn.py` / `minimax_m3_sparse_attn_decode` | `B=8,QH=32,KVH=8,S=8192,D=128`, 32 selected 128-token blocks, fp16 | `python source/triton/vllm/attention/minimax_m3/sparse_decode_runtime.py` |

V2 的 Meta entries 来自公开的 `meta-pytorch/applied-ai`；MoE runtime 只计 expert projection，routing/alignment 在计时外预构造，绝不把 Python adapter 时间混成 kernel 时间。

## baseline-v1 复用来源（16）

以下源码既保留旧 CSV 的真实来源，也已经作为独立 entry 进入 Baseline V2 registry。单列是为了保留来源边界，不表示它们被排除在 54 个入口之外。

| V1 entry | source | runtime |
|---|---|---|
| Conv1D | `flag-gems/convolution/conv1d/conv1d.py` + `conv2d.py` | `python source/triton/flag-gems/convolution/conv1d/conv1d_runtime.py` |
| triangular solve | `flag-gems/factorization/triangular_solve/linalg_solve_triangular.py` | `python source/triton/flag-gems/factorization/triangular_solve/linalg_solve_triangular_runtime.py` |
| embedding | `flag-gems/indexing/embedding/embedding.py` | `python source/triton/flag-gems/indexing/embedding/embedding_runtime.py` |
| shifted row copy | `flag-gems/indexing/roll/roll.py` | `python source/triton/flag-gems/indexing/roll/roll_runtime.py` |
| matrix copy/transpose | `flag-gems/layout/copy/copy.py` | `python source/triton/flag-gems/layout/copy/copy_runtime.py` |
| BatchNorm training | `flag-gems/normalization/batch_norm/batch_norm.py` | `python source/triton/flag-gems/normalization/batch_norm/batch_norm_runtime.py` |
| GroupNorm backward | `flag-gems/normalization/group_norm/groupnorm.py` | `python source/triton/flag-gems/normalization/group_norm/groupnorm_runtime.py` |
| logsumexp | `flag-gems/normalization/logsumexp/logsumexp.py` | `python source/triton/flag-gems/normalization/logsumexp/logsumexp_runtime.py` |
| softmax backward | `flag-gems/normalization/softmax/softmax.py` | `python source/triton/flag-gems/normalization/softmax/softmax_runtime.py` |
| AdamW | `flag-gems/optimization/adamw/_fused_adam.py` | `python source/triton/flag-gems/optimization/adamw/fused_adam_runtime.py` |
| addcmul | `flag-gems/pointwise/addcmul/addcmul.py` | `python source/triton/flag-gems/pointwise/addcmul/addcmul_runtime.py` |
| FP8 MQA logits | `flag-gems/routing/fp8_mqa_logits/fp8_mqa_logits.py` | `python source/triton/flag-gems/routing/fp8_mqa_logits/fp8_mqa_logits_runtime.py` |
| ordered prefix | `flag-gems/scan/cumsum/cumsum.py` | `python source/triton/flag-gems/scan/cumsum/cumsum_runtime.py` |
| histogram | `flag-gems/statistics/histogram/histc.py` | `python source/triton/flag-gems/statistics/histogram/histc_runtime.py` |
| max-pool with indices | `flag-gems/vision/max_pool2d/max_pool2d_with_indices.py` | `python source/triton/flag-gems/vision/max_pool2d/max_pool2d_with_indices_runtime.py` |
| legacy attention with bias | `flash-attention/attention/fused/flash_attn_triton.py` | `python source/triton/flash-attention/attention/fused/flash_attn_triton_runtime.py` |

## Necessary support

- `flash-attention/normalization/layer_norm/support/{library.py,torch.py}`：上游 LayerNorm 的导入依赖。
- `liger-kernel/support/liger_kernel/{__init__.py,utils.py,ops/utils.py}`：Liger runtime 的最小包边界。
- `xformers/gemm/tiled/matmul_perf_model.py` 与 `xformers/support/triton/{importing.py,vararg_kernel.py}`：xFormers kernels 的直接依赖。
- `meta-applied-ai/support/runtime.py` 与 `meta-applied-ai/moe/support/projection_runtime.py`：只负责加载 vendored 源码、构造未计时 metadata 与打印一次运行结果。
- `state-spaces-mamba/mamba_ssm/ops/triton/{ssd_bmm.py,softplus.py,mamba3/utils.py}` 与 `mamba_ssm/utils/determinism.py`：Mamba2/Mamba3 上游 kernel 的直接依赖；`state-spaces-mamba/support/runtime.py` 只接入本地 namespace package。
- `fla/conv/causal1d/kernels.py` 是两个已纳入 FLA causal-conv public entry 的原始 Triton kernel；`fla/support/runtime.py` 只接入上游包边界和 chunk metadata，不改写 kernel。
- `vllm/support/runtime.py`：只提供原文件导入所需的当前设备 capability 与 Triton module 接线；paged decode 和 MiniMax-M3 算法源码保持上游原样。

除本清单列出的 entry、runtime 和 support 外，`source/triton/` 不保留其它 Python 文件。
