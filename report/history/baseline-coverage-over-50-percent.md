# Baseline 覆盖率超过 50%：严格同语言上游对照接线报告

## 1. 结论

本轮把固定语料中的严格 upstream baseline 覆盖从 **30 个独立算法提高到 47 个**：

- 固定语料有 113 个 kernel 名；
- 其中 20 个 `variant_*` 是已有算法的等价 DSL 写法，不计作独立算法；
- 分母因此是 93；
- 5090 表中至少有一家严格 source 的独立算法为 47，覆盖率为 **47/93 = 50.54%**；
- H100 表同样为 **47/93 = 50.54%**；
- 相对本轮开始时的 30/93，本轮净增 **17 个独立算法**。

两张表做并集时是 48/93，因为旧有的 `embedding_backward_atomic` 只有 H100 表留有 source 数字。这个设备特有旧格子不用于本轮 headline；本轮采用更严格的“每张固定表各自达到 47/93”。

本轮没有修改 Kernel IR、Physical Plan、realizer 或 emitter。改动只落在：

1. 让 DSL 算法与确定要比较的上游算法一致；
2. 保存公开上游源码，并在源码旁放置 runtime；
3. 让既有 repro runner 能加载 source callable、做数值对照并测同一 scope；
4. 更新两张固定性能表。

## 2. 严格纳入标准

一个 source 单元只有同时满足以下条件才进入固定表：

- generated 与 source 使用同一种目标语言；
- source 来自公开、面向性能的实现，而不是临时参考代码；
- 两边算法结构、可观察输入输出语义、GPU kernel 数量与计时 scope 一致；
- 输出、workspace 和可复用状态在计时前分配；
- ABI view、静态形状和 dtype 可以由计时外 adapter 对齐，但 adapter 不改算法，也不把必要 GPU 工作藏到计时外；
- source 在当前设备上真实编译、运行，并分别对 reference 做数值验证；
- `variant_*` 不重复贡献覆盖率。

手写 CUDA、PyTorch composition、临时 reference、算法不同但名字相同的实现均未计入。旧的 Triton/cuTile `grouped_gemm` source 还额外做了输出列表的 GPU 拼接，与 generated 的扁平输出合同不同，本轮一并撤掉；该算法仍由严格的 TileLang source 覆盖，所以覆盖率不变。

## 3. 新接入的 17 个独立算法

表中的比值是“同语言 generated p50 / source p50”。小于 1 表示 generated 更快；这里只记录事实，本轮不修性能差距。

| 独立算法 | Provider | 公开实现与本地位置 | 为算法对齐所做的处理 | 5090 比值 | H100 比值 |
|---|---|---|---|---:|---:|
| `logsumexp` | Triton | FlagGems；`source/triton/flag-gems/normalization/logsumexp/` | 无算法改写；直接对齐逐行 log-sum-exp | 0.997× | 0.972× |
| `conv1d` | Triton | FlagGems；`source/triton/flag-gems/convolution/conv1d/` | 保持 same-padding、无 bias 的一维卷积；计时外建立 singleton-channel view、上游要求的通道 padding、零 bias 与输出，计时只调用其公开内层 Triton kernel | 0.107× | 0.086× |
| `softmax_backward` | Triton | FlagGems；`source/triton/flag-gems/normalization/softmax/` | 无算法改写；两边都是单 kernel Jacobian-vector product | 0.843× | 1.021× |
| `shifted_row_copy` | Triton | FlagGems roll；`source/triton/flag-gems/indexing/roll/` | 选择与 DSL 相同的固定轴循环移位 | 0.978× | 1.048× |
| `scalar_table_lookup` | Triton | FlagGems embedding；`source/triton/flag-gems/indexing/embedding/` | 保留 i32 label；对齐单次 embedding lookup | 1.023× | 1.000× |
| `fp8_mqa_logits` | Triton | FlagGems；`source/triton/flag-gems/routing/fp8_mqa_logits/` | 对齐 FP8 输入、加权 logits 与 f32 输出 | 0.420× | 0.967× |
| `matrix_transpose` | Triton | FlagGems copy；`source/triton/flag-gems/layout/copy/` | 以转置 view 调用公开 copy kernel；view 构造不产生 GPU 工作 | 0.299× | 0.228× |
| `batched_row_affine` | Triton | FlagGems addcmul；`source/triton/flag-gems/pointwise/addcmul/` | 对齐 `x + value * tensor1 * tensor2` 与多秩广播 | 0.995× | 0.979× |
| `ordered_prefix` | Triton | FlagGems cumsum；`source/triton/flag-gems/scan/cumsum/` | 对齐包含式逐行前缀和 | 1.961× | 1.863× |
| `histogram` | Triton | FlagGems histc；`source/triton/flag-gems/statistics/histogram/` | DSL 改为 f32 样本/f32 计数和固定 `[0,256]` 区间，与 source 的单 kernel histc 路径一致 | 3.055× | 2.330× |
| `batch_norm_training` | Triton | FlagGems；`source/triton/flag-gems/normalization/batch_norm/` | DSL 增加 running mean/variance 与 momentum，使用无偏 running variance 更新；两边都输出 batch mean/rstd | 2.056× | 7.964× |
| `triangular_solve` | Triton | FlagGems；`source/triton/flag-gems/factorization/triangular_solve/` | 对齐 batched lower-triangular、非 unit diagonal、原地 solution buffer；每次计时前恢复 RHS | 1.281× | 1.093× |
| `adamw_update` | Triton | FlagGems fused Adam；`source/triton/flag-gems/optimization/adamw/` | DSL 改为标准 AdamW 的 `sqrt(v_hat) + epsilon`；每次计时前恢复 parameter/m/v | 1.018× | 0.985× |
| `attention_backward` | cuTile | TileGym；`source/cutile/tilegym/attention/dense/` | 对齐 causal GQA backward 的 delta、dK/dV、dQ 三阶段 pipeline | 1.191× | 1.858× |
| `mamba_chunk_scan` | TileLang | TileLang；`source/tilelang/tilelang/scan/mamba_chunk_scan/` | DSL state dimension 从 64 对齐到上游的 128；其余分块多状态算法不变 | 1.361× | 1.881× |
| `continuous_gqa_decode` | TileLang | TileLang；`source/tilelang/tilelang/attention/gqa_decode/` | DSL 从逐 query-head 程序改为上游实际使用的“每个 KV head 持有一组 query heads”，数学结果不变但程序算法结构对齐 | 1.083× | 2.075× |
| `block_sparse_attention` | TileLang | TileLang；`source/tilelang/tilelang/attention/blocksparse_gqa_decode_varlen/` | 输入改为上游要求的逆序块索引；两边均为 partial + combine 两段 GPU pipeline | 2.654× | 13.988× |

Provider 分布为：Triton 13 个、cuTile 1 个、TileLang 3 个。cuTile 的公开高性能生态确实较小，本轮没有通过降低标准来填数量。

## 4. 接线如何保持公平

### 4.1 通用 source callable

三个 provider 入口都能加载与上游源码相邻的 `*_runtime.py`。runtime 的职责只有：

- 把固定 repro ABI 投影到上游 ABI；
- 在首次调用时建立和缓存 executable；
- 预分配 output/workspace；
- 返回真正要计时的 source callable。

evaluation runner 统一完成 source/reference 数值检查和 source p50/p95 测量。它没有引入新的测试目录、fixture 或校验系统，仍然只有既有的手动 repro 入口。

### 4.2 状态型和多阶段算法

- BatchNorm 在每次 generated/source launch 前恢复 running statistics；
- AdamW 在每次 launch 前恢复 parameter、first moment 和 second moment；
- triangular solve 在每次 launch 前恢复 RHS/solution；
- histogram 两边都把 output 清零计入 end-to-end GPU scope；
- attention backward 两边都计 delta、dK/dV、dQ 三个 launch；
- block-sparse attention 的 TileLang `PrimFunc` 内有两个 `T.Kernel`，一个 executable 调用仍兑现 partial 与 combine 两段 GPU 工作，和 generated 的两个 artifact 对齐。

### 4.3 Conv1d 的 kernel-only scope

FlagGems 的 public Conv1d 入口会为单通道输入创建 padded tensor、零 bias 和 output。直接围住 public callable 会把分配算进 source，而 generated 采用预分配输出，口径不公平。因此 runtime 仅在计时外做静态 ABI 适配，并直接调用同一份上游源码暴露的 `conv2d_forward_kernel`。修正后：

- 5090 source 从 0.0794/0.0802 ms 变为 0.0569/0.0579 ms；
- H100 source 从 0.1129/0.1141 ms 变为 0.0931/0.0937 ms；
- 两台机器数值均继续通过。

## 5. 找过但没有纳入的候选

以下不是“没找到文件”，而是经过算法和 scope 审计后没有资格进入固定表：

| 候选 | 没有纳入的原因 |
|---|---|
| `moe`（三家） | 现有公开文件提供 grouped-GEMM building block；当前 adapter 需要用 PyTorch gather/ReLU/merge 或额外 kernel 拼出完整两层 expert FFN。它不是同范围的单份高性能 source，不能作为严格 baseline。 |
| `absorbed_mla_prefill` | 公开 TileLang persistent/paged MLA 实现的调用形态和算法阶段与当前 prefill DSL 不一致；不能仅因同属 MLA 就挂接。 |
| `paged_mla_decode` | 有 TileLang 公开实现，但当前 generated TileLang 对这一单行 contraction 明确 unsupported，尚未形成双方可运行的格子。 |
| `token_sparse_mla_prefill` | 公开实现的稀疏调度/输出合同与当前 token-selected prefill 不同；已有 generated 目标还存在首次编译成本或能力边界，未伪造对照。 |
| `mla_head_projection` | 找到的是完整 MLA pipeline 或 decode kernel，不是这两个独立 projection 调用。 |
| `varlen_gqa_decode_logits` | source 存在，但当前 TileLang generated 对单行 contraction 明确 unsupported。 |
| `selective_scan` | 当前 DSL 是定步单状态 recurrence；公开现代 Mamba selective scan 是分块多状态算法。两者同名但不是同一算法。 |
| `conv2d` | 当前 DSL 是单通道二维 stencil；FlagGems 是 dense-channel im2col/matmul。尝试照上游重写后暴露 symbolic flatten provenance 与多轴 contraction 投影缺口，本轮恢复原算法并留空。 |
| `max_pool2d` | 找到的 FlagGems 高性能接口同时产生 value 和 index，而 DSL 只产生 value，observable output contract 不同。 |
| `batched_cholesky` | 上游是输入原地覆盖，当前 DSL 是独立输出；别名/调用合同不同，未把 adapter 成本藏起来。 |
| `reshape_and_cache` | 尝试对齐公开高秩 cache 写入时暴露多秩广播 store 投影缺口；未保留一条只为过 baseline 的改写。 |
| signed W4A16 / quantized GEMM | 已有公开 W4A8、FP8 或带不同 zero-point/group schema 的实现，packing、scale 与算法合同不同，不能互相冒充。 |
| top-k / sampling 变体 | vendor 目录中的实现要么是不同排序/采样算法，要么依赖 PyTorch composition；本轮未找到满足同语言、同算法、同调用 scope 的公开实现。 |

拒绝尝试对应的 runtime、source 接线和 DSL 临时改写都已删除；它们不贡献分子。

## 6. 本轮暴露、留给下一轮的性能问题

以下仅列同语言 generated/source 超过 1.05× 的新增格子：

| 算法 / Provider | 5090 | H100 | 当前事实 |
|---|---:|---:|---|
| `ordered_prefix` / Triton | 1.961× | 1.863× | 同算法；生成扫描投影需要追 |
| `histogram` / Triton | 3.055× | 2.330× | 同算法与相同清零 scope；生成原子 histogram 明显慢 |
| `continuous_gqa_decode` / TileLang | 1.083× | 2.075× | 算法已按 grouped-head 对齐，H100 差距更明显 |
| `mamba_chunk_scan` / TileLang | 1.361× | 1.881× | 同一分块多状态算法；需要比较生成 TileLang 与原生 source |
| `attention_backward` / cuTile | 1.191× | 1.858× | 三阶段调用一致；H100 差距明显 |
| `block_sparse_attention` / TileLang | 2.654× | 13.988× | 两阶段调用一致；H100 是最严重的新缺口 |
| `batch_norm_training` / Triton | 2.056× | 7.964× | 状态更新和输出均已对齐；H100 是严重缺口 |
| `triangular_solve` / Triton | 1.281× | 1.093× | 同算法；5090 差距更大 |

这些差距没有在本轮通过改 DSL、缩小 scope 或放宽 source 来“修”。它们是接完严格 baseline 后第一次可见的编译器/投影性能问题。

## 7. 实际验证范围

本轮只运行新接入和 DSL 真正受影响的 repro，没有做全量：

```text
triton: logsumexp, conv1d, softmax_backward, shifted_row_copy,
        scalar_table_lookup, fp8_mqa_logits, matrix_transpose,
        batched_row_affine, ordered_prefix, histogram,
        batch_norm_training, triangular_solve, adamw_update
cutile: attention_backward
tilelang: mamba_chunk_scan, continuous_gqa_decode, block_sparse_attention
```

每一格均通过既有命令形态运行：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <kernel>
```

5090 与 H100 均实际运行。H100 使用独立的临时项目树和项目外既有 provider 环境；TileLang 需要的 CUDA 12.2 由机器已有工具链提供，没有把环境、cache 或临时构建产物放入仓库。

## 8. 最终状态

- 严格覆盖目标已完成：**47/93 = 50.54%**；
- 新接 17 个独立算法，不含任何 `variant_*` 重复计数；
- 两张固定表都已更新且格式不变；
- 旧 grouped-GEMM 的两格不公平 source 已撤掉；
- 接线重复分支、拒绝尝试和本地运行缓存已清理；
- 本轮没有修改编译器架构，也没有为某个 kernel/target 在共享层增加特例。
