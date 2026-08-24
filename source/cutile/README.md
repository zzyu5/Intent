# cuTile source inventory

本目录只保留 NVIDIA `cutile-python` 官方样例和 NVIDIA TileGym 的可运行 kernel。每个 V2 entry 都有相邻 runtime；同一 source 的 forward/backward 只有在调用边界和计时范围可以独立时才分成两条。运行前进入项目的 cuTile Python 环境，命令从仓库根目录执行。

Upstream roots：`NVIDIA/cutile-python` 与 `NVIDIA/TileGym`。

## Baseline V2 entries（37）

| 集合 | entry | source / public boundary | 模型级输入 | runtime |
|---|---|---|---|---|
| V2 | official FMHA | `cutile-python/attention/fmha/AttentionFMHA.py` / `cutile_fmha` | `B=4,QH=32,KVH=8,S=4096,D=128`, fp16 causal | `python source/cutile/cutile-python/attention/fmha/AttentionFMHA_runtime.py` |
| V1+V2 | block-scaled GEMM | `cutile-python/gemm/block_scaled/BlockScaledMatMul.py` / block-scaled matmul | `4096×4096×14336`, FP8, scale block 32 | `python source/cutile/cutile-python/gemm/block_scaled/BlockScaledMatMul_runtime.py` |
| V2 | official dense/persistent GEMM | `cutile-python/gemm/dense/MatMul.py` / `cutile_matmul` | `4096×4096×14336`, fp16 | `python source/cutile/cutile-python/gemm/dense/MatMul_runtime.py` |
| V1+V2 | official LayerNorm fwd/bwd | `cutile-python/normalization/layer_norm/LayerNorm.py` / `cutile_layer_norm` | `8192×4096`, bf16 | `python source/cutile/cutile-python/normalization/layer_norm/LayerNorm_runtime.py` |
| V1+V2 | SiLU-and-mul forward | `tilegym/activation/silu_and_mul/silu_and_mul.py` / `silu_and_mul` | `4096×28672`, bf16 | `python source/cutile/tilegym/activation/silu_and_mul/silu_and_mul_runtime.py` |
| V1+V2 | dense attention forward | `tilegym/attention/dense/attention.py` / `tile_fmha` | `B=2,QH=32,KVH=8,S=4096,D=128`, bf16 causal | `python source/cutile/tilegym/attention/dense/attention_runtime.py` |
| V1+V2 | dense attention backward | same source / backward callable | same contract, dQ/dK/dV | `python source/cutile/tilegym/attention/dense/attention_backward_runtime.py` |
| V2 | grouped flash decode | `tilegym/attention/flash_decode/flash_decode.py` / `fmha_decode` | `B=8,QH=32,KVH=8,S=8192,D=128`, bf16 | `python source/cutile/tilegym/attention/flash_decode/flash_decode_runtime.py` |
| V1+V2 | split-K attention reduce | `tilegym/attention/flash_decode/splitk_reduce.py` / `splitk_reduce` | `B=8,H=32,splits=16,D=128,S=8192` | `python source/cutile/tilegym/attention/flash_decode/splitk_reduce_runtime.py` |
| V1+V2 | MLA prefill | `tilegym/attention/mla/mla.py` / `tile_mla` | `B=1,QH=128,KVH=1,S=2048,D=128,PE=64`, fp16 | `python source/cutile/tilegym/attention/mla/mla_runtime.py` |
| V1+V2 | batched GEMM | `tilegym/gemm/batched/bmm.py` / `bmm` | `B=32,M=N=512,K=1024`, bf16 | `python source/cutile/tilegym/gemm/batched/bmm_runtime.py` |
| V1+V2 | dense GEMM | `tilegym/gemm/dense/matmul.py` / `matmul` | `M=8192,N=11008,K=4096`, bf16 | `python source/cutile/tilegym/gemm/dense/matmul_runtime.py` |
| V2 | grouped GEMM | `tilegym/gemm/grouped/group_gemm.py` / `group_gemm` | rows 256/512/1024/2048, `K=N=4096`, bf16 | `python source/cutile/tilegym/gemm/grouped/group_gemm_runtime.py` |
| V2 | MoE alignment | `tilegym/moe/alignment/moe_align_block.py` / `moe_align_block_size` | 8192 tokens, top-8, 64 experts | `python source/cutile/tilegym/moe/alignment/moe_align_block_runtime.py` |
| V2 | MoE expert projection | `tilegym/moe/fused/moe.py` / `invoke_fused_moe_kernel` | 4096 tokens, hidden 4096, intermediate 14336, 8 experts, top-2 | `python source/cutile/tilegym/moe/fused/moe_runtime.py` |
| V1+V2 | chunked softmax | `tilegym/normalization/softmax/softmax.py` / `softmax` | `8192×32768`, bf16 | `python source/cutile/tilegym/normalization/softmax/softmax_runtime.py` |
| V1+V2 | RoPE Q/K | `tilegym/position/rope/rope.py` / `apply_rope_base` | `B=2,S=4096,QH=32,KVH=8,D=128`, bf16 | `python source/cutile/tilegym/position/rope/rope_runtime.py` |
| V2 | GELU | `tilegym/activation/fused/gelu.py` / `gelu` | `8192×4096`, fp16 | `python source/cutile/tilegym/activation/fused/gelu_runtime.py` |
| V2 | GEGLU | `tilegym/activation/fused/geglu.py` / `geglu` | `4096×28672`, fp16 | `python source/cutile/tilegym/activation/fused/geglu_runtime.py` |
| V2 | ReLU | `tilegym/activation/relu/relu.py` / `relu` | `8192×4096`, fp16 | `python source/cutile/tilegym/activation/relu/relu_runtime.py` |
| V2 | SwiGLU | `tilegym/activation/swiglu/swiglu.py` / `swiglu` | `8192×14336`, bf16 | `python source/cutile/tilegym/activation/swiglu/swiglu_runtime.py` |
| V2 | dropout | `tilegym/regularization/dropout/dropout.py` / `dropout` | `8192×4096`, fp16, p=0.1 | `python source/cutile/tilegym/regularization/dropout/dropout_runtime.py` |
| V2 | attention sink prefill | `tilegym/attention/sink_prefill/attention_sink.py` / `attention_sink` | `B=1,S=4096,QH=32,KVH=8,D=128`, bf16 | `python source/cutile/tilegym/attention/sink_prefill/attention_sink_runtime.py` |
| V2 | attention sink decode | `tilegym/attention/sink_decode/attention_sink_decode.py` / `attention_sink_decode` | `B=32,S=8192,QH=32,KVH=8,D=128`, bf16 | `python source/cutile/tilegym/attention/sink_decode/attention_sink_decode_runtime.py` |
| V2 | Gemma prefill attention | `tilegym/attention/gemma_prefill/gemma_attention.py` / `gemma_attention_cutile` | `B=2,S=4096,QH=32,KVH=8,D=128`, window 1024, soft-cap 50, bf16 | `python source/cutile/tilegym/attention/gemma_prefill/gemma_attention_runtime.py` |
| V2 | Gemma split-K decode attention | `tilegym/attention/gemma_decode/gemma_attention_decode.py` / `gemma_fmha_decode` | `B=32,S=8192,QH=32,KVH=8,D=128`, window 1024, soft-cap 50, bf16 | `python source/cutile/tilegym/attention/gemma_decode/gemma_attention_decode_runtime.py` |
| V2 | absorbed MLA decode | `tilegym/attention/mla_decode/mla_decoding.py` / `mla_decoding` | `B=8,H=64,S=8192,D=512,PE=64`, fp16 | `python source/cutile/tilegym/attention/mla_decode/mla_decoding_runtime.py` |
| V2 | split-K MLA decode | `tilegym/attention/mla_decode_split/mla_decoding_split_kv.py` / `mla_decoding_split_kv` | same shape, split 512 | `python source/cutile/tilegym/attention/mla_decode_split/mla_decoding_split_kv_runtime.py` |
| V2 | sparse MLA prefill | `tilegym/attention/sparse_mla/sparse_mla.py` / `tile_sparse_mla` | `S=2048,SKV=4096,H=64,topk=512,D=128+64`, bf16 | `python source/cutile/tilegym/attention/sparse_mla/sparse_mla_runtime.py` |
| V2 | sliding-window attention | `tilegym/attention/sliding_window/swa_attention.py` / `tile_swa_attention` | `B=2,QH=32,KVH=8,S=4096,D=128,window=1024`, fp16 | `python source/cutile/tilegym/attention/sliding_window/swa_attention_runtime.py` |
| V2 | chunk gated delta rule | `tilegym/scan/gated_delta_chunk/chunk_gated_delta_rule.py` / `chunk_gated_delta_rule` | `B=2,T=2048,H=8,K=V=128`, bf16 | `python source/cutile/tilegym/scan/gated_delta_chunk/chunk_gated_delta_rule_runtime.py` |
| V2 | recurrent gated delta rule | `tilegym/scan/gated_delta_recurrent/recurrent_gated_delta_rule.py` / recurrent entry | same model shape | `python source/cutile/tilegym/scan/gated_delta_recurrent/recurrent_gated_delta_rule_runtime.py` |
| V2 | RMSNorm | `tilegym/normalization/rms_norm/rms_norm.py` / `rms_norm` | `8192×4096`, bf16 | `python source/cutile/tilegym/normalization/rms_norm/rms_norm_runtime.py` |
| V2 | NVFP4 quantization | `tilegym/quantization/nvfp4/nvfp4_quantize.py` / `tile_nvfp4_quantize` | `8192×4096`, bf16→packed FP4 | `python source/cutile/tilegym/quantization/nvfp4/nvfp4_quantize_runtime.py` |
| V2 | mHC GEMM + RMS scaling | `tilegym/mhc/fused/mhc.py` / `mhc_gemm_rms_scale` | 2048 tokens, hidden 4096, 4 residual streams, bf16 | `python source/cutile/tilegym/mhc/fused/mhc_gemm_rms_runtime.py` |
| V2 | mHC residual mixing | same source / `mhc_apply_residual` | 2048 tokens, hidden 4096, 4 residual streams, bf16 | `python source/cutile/tilegym/mhc/fused/mhc_apply_residual_runtime.py` |
| V2 | mHC Sinkhorn normalization | same source / `mhc_sinkhorn` | 8192 tokens, 4 residual streams, fp32 | `python source/cutile/tilegym/mhc/fused/mhc_sinkhorn_runtime.py` |

V2 新增部分均直接取自当前公开 NVIDIA TileGym；没有用 PyTorch composition 或手写参考实现冒充 cuTile baseline。

## 冻结在 baseline-v1 的来源

以下完整 MoE wrapper 继续保留，因为旧 repro 明确调用它；它包含 PyTorch routing、多个 cuTile kernel 与最终归并，不进入上面的 baseline-v2 inventory。

| V1 entry | source | runtime |
|---|---|---|
| MoE | `cutile-python/moe/fused/MoE.py` | `python source/cutile/cutile-python/moe/fused/MoE_runtime.py` |

## Necessary support

- `tilegym/support/utils.py`：TileGym kernels 的 `next_power_of_2` 等直接依赖。
- `tilegym/support/runtime.py` 与 `tilegym/support/runtime_cases.py`：加载本地 vendored source、构造模型级输入并输出一次运行结果。
- GEGLU 对相邻 `activation/fused/gelu.py` 的依赖、attention-sink/MLA split 对现有 `attention/flash_decode/splitk_reduce.py` 的依赖，均由 runtime 显式装载；没有复制第二份实现。
- Gemma decode 复用同一份 split-K reduce；mHC 的三个 entry 来自 TileGym 当前 `tilegym/mhc/fused/mhc.py`，在本地保持一份原始 source、按三个真实 callable 分别计时，并保留上游的 experimental 状态，不冒充 NVIDIA `cutile-python` 正式样例。

除本清单列出的 entry、runtime 和 support 外，`source/cutile/` 不保留其它 Python 文件。
