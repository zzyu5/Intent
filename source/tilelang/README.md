# TileLang source inventory

本目录保存 TileLang 官方 examples 中可独立调用的高性能算法。forward/backward、prefill/decode、dense/sparse 只有算法合同确实不同才分行，不按参数变体凑数。运行前进入项目的 TileLang Python 环境，命令从仓库根目录执行。

Upstream root：`tile-ai/tilelang`。本轮没有用第三方复写或 PyTorch reference 充当 TileLang source。

## baseline-v2 entries（39）

| 集合 | entry | source / public boundary | 模型级输入 | runtime |
|---|---|---|---|---|
| V1+V2 | block-sparse GQA decode | `tilelang/attention/blocksparse_gqa_decode_varlen/example_tilelang_sparse_gqa_decode_varlen_indice.py` / `flashattn` | `B=8,QH=32,KVH=8,S=8192,D=128`, 128 selected blocks | `python source/tilelang/tilelang/attention/blocksparse_gqa_decode_varlen/example_tilelang_sparse_gqa_decode_varlen_indice_runtime.py` |
| V1+V2 | dense FlashAttention forward | `tilelang/attention/flash_forward_bshd/example_mha_fwd_bshd.py` / `flashattn` | `B=4,S=4096,H=32,D=128`, fp16 causal | `python source/tilelang/tilelang/attention/flash_forward_bshd/example_mha_fwd_bshd_runtime.py` |
| V1+V2 | varlen GQA prefill | `tilelang/attention/flash_forward_varlen/example_gqa_fwd_varlen.py` / `flashattn` | packed 8 sequences, `QH=32,KVH=8,D=128`, fp16 causal | `python source/tilelang/tilelang/attention/flash_forward_varlen/example_gqa_fwd_varlen_runtime.py` |
| V2 | GQA decode | `tilelang/attention/gqa_decode/example_gqa_decode.py` / `flashattn` | `B=32,QH=32,KVH=8,S=8192,D=128`, fp16 | `python source/tilelang/tilelang/attention/gqa_decode/example_gqa_decode_runtime.py` |
| V2 | varlen GQA decode logits | `tilelang/attention/gqa_decode_varlen_logits/example_gqa_decode_varlen_logits.py` / `flashattn` | `B=16,QH=32,KVH=8,S≤8192,D=64`, fp16 | `python source/tilelang/tilelang/attention/gqa_decode_varlen_logits/example_gqa_decode_varlen_logits_runtime.py` |
| V2 | paged MLA decode | `tilelang/attention/mla_decode_paged/example_mla_decode_paged.py` / `mla_decode_tilelang` | `B=32,QH=128,KVH=1,S≈8192,D=128,PE=64`, fp16 | `python source/tilelang/tilelang/attention/mla_decode_paged/example_mla_decode_paged_runtime.py` |
| V2 | persistent MLA decode | `tilelang/attention/mla_decode_persistent/example_mla_decode_persistent.py` / `flashattn` | `B=8,H=64,S=16384,D=128,PE=64`, fp16 | `python source/tilelang/tilelang/attention/mla_decode_persistent/example_mla_decode_persistent_runtime.py` |
| V2 | Conv2D | `tilelang/convolution/basic/example_convolution.py` / `convolution` | NHWC `32×128×128×256`, kernel `3×3×256×512`, fp16 | `python source/tilelang/tilelang/convolution/basic/example_convolution_runtime.py` |
| V1+V2 | dense GEMM | `tilelang/gemm/dense/example_gemm.py` / `matmul` | `4096×4096×14336`, fp16/bf16 | `python source/tilelang/tilelang/gemm/dense/example_gemm_runtime.py` |
| V1+V2 | W4A8 dequant GEMM | `tilelang/gemm/dequantize_w4a8/example_dequant_gemm_w4a8.py` / `matmul_int8xint4` | `4096×4096×14336`, int8×packed int4 | `python source/tilelang/tilelang/gemm/dequantize_w4a8/example_dequant_gemm_w4a8_runtime.py` |
| V1+V2 | FP8 GEMM | `tilelang/gemm/fp8/example_tilelang_gemm_fp8.py` / `matmul` | `4096×4096×14336`, FP8 | `python source/tilelang/tilelang/gemm/fp8/example_tilelang_gemm_fp8_runtime.py` |
| V1+V2 | grouped GEMM | `tilelang/gemm/grouped/example_grouped_gemm_fwd.py` / `grouped_gemm` | group rows 256/512/1024/2048, `K=N=4096`, fp16 | `python source/tilelang/tilelang/gemm/grouped/example_grouped_gemm_fwd_runtime.py` |
| V1+V2 | 2:4 sparse GEMM | `tilelang/gemm/sparse_2to4/example_gemm_sp.py` / `matmul_sp_fp16` | `M=8192,N=14336,K=8192`, fp16 | `python source/tilelang/tilelang/gemm/sparse_2to4/example_gemm_sp_runtime.py` |
| V2 | fused routed/shared MoE | `tilelang/moe/fused/example_fusedmoe_tilelang.py` / `custom_kernel` | DeepSeek widths, 8192 tokens, 8 routed experts, top-4 | `python source/tilelang/tilelang/moe/fused/example_fusedmoe_tilelang_runtime.py` |
| V1+V2 | online softmax | `tilelang/normalization/online_softmax/online_softmax.py` / `softmax_kernel` | `8192×8192`, fp16 | `python source/tilelang/tilelang/normalization/online_softmax/online_softmax_runtime.py` |
| V2 | RMSNorm | `tilelang/normalization/rms_norm/rms_norm.py` / `rms_norm` | `8192×4096`, fp32 | `python source/tilelang/tilelang/normalization/rms_norm/rms_norm_runtime.py` |
| V2 | DeepSeek V3.2 top-k selector | `tilelang/routing/deepseek_v32_topk/topk_selector.py` / `tl_topk` | `32×32768`, top-k 2048, fp32 | `python source/tilelang/tilelang/routing/deepseek_v32_topk/topk_selector_runtime.py` |
| V1+V2 | Mamba chunk scan | `tilelang/scan/mamba_chunk_scan/example_mamba_chunk_scan.py` / `chunk_scan_fwd` | `B=1,S=2048,H=32,G=8,P=64,N=128,C=256`, fp16 | `python source/tilelang/tilelang/scan/mamba_chunk_scan/example_mamba_chunk_scan_runtime.py` |
| V2 | Mamba chunk state construction | `tilelang/scan/mamba_chunk_state/example_mamba_chunk_state.py` / `chunk_state_fwd` | `B=1,S=2048,H=32,G=8,P=64,N=128`, chunk 256, fp16 | `python source/tilelang/tilelang/scan/mamba_chunk_state/example_mamba_chunk_state_runtime.py` |
| V2 | fused chunk linear attention forward | `tilelang/linear_attention/fused_chunk_forward/example_linear_attn_fwd.py` / `tl_fused_chunk_fwd_kernel` | `B=1,S=2048,H=16,D=128`, fp16 | `python source/tilelang/tilelang/linear_attention/fused_chunk_forward/example_linear_attn_fwd_runtime.py` |
| V2 | fused chunk linear attention backward | `tilelang/linear_attention/fused_chunk_backward/example_linear_attn_bwd.py` / `tl_fused_chunk_bwd_kernel` | same model shape, dQ/dK/dV | `python source/tilelang/tilelang/linear_attention/fused_chunk_backward/example_linear_attn_bwd_runtime.py` |
| V2 | chunk retention forward | `tilelang/linear_attention/retention/example_retention_fwd.py` / `chunk_retention_fwd_kernel` | `B=1,S=2048,H=16,D=128`, fp16 | `python source/tilelang/tilelang/linear_attention/retention/example_retention_fwd_runtime.py` |
| V2 | attention-sink backward | `tilelang/attention/attention_sink_backward/example_mha_sink_bwd_bhsd.py` / `flashattn_bwd` | `B=2,H=16,S=2048,D=128`, fp16 | `python source/tilelang/tilelang/attention/attention_sink_backward/example_mha_sink_bwd_bhsd_runtime.py` |
| V2 | block-causal attention | `tilelang/attention/block_causal/block_causal_attention.py` / `block_causal_attention` | `B=2,S=4096,H=16,D=128`, fp16 | `python source/tilelang/tilelang/attention/block_causal/block_causal_attention_runtime.py` |
| V2 | varlen block-causal attention | `tilelang/attention/block_causal_varlen/block_causal_attention_varlen.py` / varlen wrapper | packed lengths 4096/3840/3584/3328, `H=16,D=128` | `python source/tilelang/tilelang/attention/block_causal_varlen/block_causal_attention_varlen_runtime.py` |
| V2 | native sparse attention forward | `tilelang/attention/native_sparse_forward/example_tilelang_nsa_fwd.py` / `native_sparse_attention` | `B=2,S=4096,QH=32,KVH=4,D=128`, 64 blocks | `python source/tilelang/tilelang/attention/native_sparse_forward/example_tilelang_nsa_fwd_runtime.py` |
| V2 | native sparse attention decode | `tilelang/attention/native_sparse_decode/example_tilelang_nsa_decode.py` / `native_sparse_attention` | `B=8,S=8192,QH=32,KVH=2,D=128`, 32 selected 128-token blocks | `python source/tilelang/tilelang/attention/native_sparse_decode/example_tilelang_nsa_decode_runtime.py` |
| V2 | GQA attention backward | `tilelang/attention/gqa_backward/example_gqa_bwd.py` / split dK/dV backward pipeline | `B=1,S=4096,QH=32,KVH=8,D=64`, fp16 causal | `python source/tilelang/tilelang/attention/gqa_backward/example_gqa_bwd_runtime.py` |
| V2 | sparse MLA backward | `tilelang/attention/sparse_mla_backward/sparse_mla_bwd.py` / `sparse_mla_bwd` | `S=4096,SKV=8192,H=64,D=576,topk=2048`, bf16 | `python source/tilelang/tilelang/attention/sparse_mla_backward/sparse_mla_bwd_runtime.py` |
| V2 | FP8 lighting indexer | `tilelang/attention/fp8_lighting_indexer/fp8_lighting_indexer.py` / MQA logits interface | `S=4096,SKV=8192,H=32,D=64`, FP8 | `python source/tilelang/tilelang/attention/fp8_lighting_indexer/fp8_lighting_indexer_runtime.py` |
| V2 | DeepGEMM FP8 2× accumulation | `tilelang/gemm/fp8_2xacc/example_deepgemm_fp8_2xAcc.py` / `tl_gemm` | `4096³`, FP8→bf16 | `python source/tilelang/tilelang/gemm/fp8_2xacc/example_deepgemm_fp8_2xAcc_runtime.py` |
| V2 | per-token FP8 cast | `tilelang/quantization/per_token_fp8/example_per_token_cast_to_fp8.py` / `per_token_cast_to_fp8` | `8192×8192`, group 128 | `python source/tilelang/tilelang/quantization/per_token_fp8/example_per_token_cast_to_fp8_runtime.py` |
| V2 | block-sparse GEMM | `tilelang/gemm/block_sparse/example_blocksparse_gemm.py` / `blocksparse_matmul` | `4096³`, block `128×128×32`, 50% blocks | `python source/tilelang/tilelang/gemm/block_sparse/example_blocksparse_gemm_runtime.py` |
| V2 | grouped GEMM backward | `tilelang/gemm/grouped_backward/example_grouped_gemm_bwd.py` / `grouped_gemm_bwd` | rows 256/512/1024/2048, `K=N=4096`, fp16 | `python source/tilelang/tilelang/gemm/grouped_backward/example_grouped_gemm_bwd_runtime.py` |
| V2 | BitNet 1.58 int8×packed-int2 decode | `tilelang/gemm/bitnet_int2_decode/tilelang_bitnet_158_int8xint2_decode.py` / `bitnet_158_int8xint2_decode` | `M=1,N=K=4096`, int8×int2→int32 | `python source/tilelang/tilelang/gemm/bitnet_int2_decode/tilelang_bitnet_158_int8xint2_decode_runtime.py` |
| V2 | BF16×FP4 dequant GEMM | `tilelang/gemm/dequant_bf16_fp4/example_dequant_gemm_bf16_fp4_hopper.py` / `matmul` | `4096³`, packed FP4 | `python source/tilelang/tilelang/gemm/dequant_bf16_fp4/example_dequant_gemm_bf16_fp4_hopper_runtime.py` |
| V2 | block FP4 activation quantization | `tilelang/quantization/block_fp4/act_quant.py` / `fp4_act_quant` | `8192×4096`, bf16→packed E2M1, block 32 | `python source/tilelang/tilelang/quantization/block_fp4/act_quant_runtime.py` |
| V2 | mHC post | `tilelang/mhc/post/example_mhc_post.py` / `mhc_post_tilelang` | 4096 tokens, hidden 2560, HC=4 | `python source/tilelang/tilelang/mhc/post/example_mhc_post_runtime.py` |
| V2 | mHC pre | `tilelang/mhc/pre/example_mhc_pre.py` / fused pre pipeline | 2048 tokens, hidden 4096, HC=4 | `python source/tilelang/tilelang/mhc/pre/example_mhc_pre_runtime.py` |

## Necessary support

- `tilelang/attention/blocksparse_gqa_decode_varlen/heuristic.py`：block-sparse decode 配置。
- `tilelang/attention/flash_forward_varlen/{bert_padding.py,varlen_utils.py}`：packed varlen 输入准备。
- `tilelang/attention/native_sparse_forward/reference.py`：上游文件的直接导入；不进入 V2 计时。
- `tilelang/attention/native_sparse_decode/reference.py`：decode example 的原始 reference import；不进入 V2 计时。
- `tilelang/attention/sparse_mla_backward/sparse_mla_fwd.py` 与 `tilelang/attention/support/deepseek_v32_utils.py`：backward 的 forward state 与上游工具依赖。
- `tilelang/gemm/sparse_2to4/sparse_utils.py`、`tilelang/gemm/dequant_bf16_fp4/dequantize_utils.py`：压缩 metadata / 解包参考依赖。
- `tilelang/moe/fused/example_fusedmoe_torch.py`：上游 source 的 reference import；不作为 baseline。
- `tilelang/support/{runtime.py,runtime_cases.py}`：加载 vendored 源码、构造模型级输入并输出一次运行结果。
- 两个 fused linear-attention 文件只在导入时引用 FLA reference；runtime 为这些未计时 reference symbols 提供明确拒绝的包边界，实际运行的 TileLang kernel 源码保持原样。

除本清单列出的 entry、runtime 和 support 外，`source/tilelang/` 不保留其它 Python 文件。
