# 全量回归与 baseline 性能收口报告

## 1. 结论

前两轮改动已经在 RTX 5090 D 与 H100 上完成一次并行全量回归。每台机器实际执行：

- `113` 个 runner × Triton/cuTile/TileLang 三个 provider，共 `339` 条真实 repro；
- 展开多 case 后，两份固定 CSV 各为 `122` 行；
- 每条仍通过 `./examples/run/repro.sh <provider> <kernel>` 完成 DSL lowering、目标源码生成、下层编译、GPU 运行与数值对照；
- 5090 累计执行时间 `2785 s`，H100 累计执行时间 `2796 s`；两边在数秒内同时启动，并行墙钟约 `46 min 36 s`，没有串行等待另一台机器。

全量没有留下新的静默数值错误，也没有留下由前两轮编译器改动造成的稳定性能回归。初次全量暴露的三类真实问题已经关闭：

1. 八个已有 Triton source 只有固定数字、没有可执行 `upstream()` 接线，全量时在 generated 运行前失败；现在都直接调用上游内层 kernel，并把输出和 workspace 预分配在计时区外。
2. `continuous_gqa_decode / TileLang` 在撤掉无效 source 对照时，被一起从 generated-only 路由删掉；现已恢复为正常 generated-only repro。
3. `variant_adamw_split_pipeline` 的两阶段写法实际用了 `sqrt(v + eps)`，而原写法和上游算法是 `sqrt(v) + eps`；这不是 compiler bug，而是“等价变体”本身不等价。DSL 与 reference 已对齐，六个设备/provider 组合全部通过。

最终状态不是“所有 baseline 都在 5% 内”：

| 设备 | 严格可比格子 | `generated/source <= 1.05` | `> 1.05` | 门槛内比例 |
|---|---:|---:|---:|---:|
| RTX 5090 D | 58 | 49 | 9 | 84.48% |
| H100 | 58 | 46 | 12 | 79.31% |
| 合计 | 116 | 95 | 21 | 81.90% |

相对上一份固定表的 `93/116`，本次全量最终为 `95/116`，净增加两个门槛内格子。仍超标的 21 格里，10 格已有证据表明是目标原生原语或下层编译质量边界，10 格仍是 Intent 的物理决定/投影缺口，另有 1 格是本轮稳定测出的新性能事实、尚未完成层次归因。

## 2. 全量通过状态

按 runner 计数，而不是按展开后的 CSV 行计数：

| 设备 | Provider | pass | unsupported | compile failed | compile timeout |
|---|---|---:|---:|---:|---:|
| RTX 5090 D | Triton | 111 | 1 | 1 | 0 |
| RTX 5090 D | cuTile | 110 | 3 | 0 | 0 |
| RTX 5090 D | TileLang | 102 | 10 | 1 | 0 |
| H100 | Triton | 111 | 1 | 1 | 0 |
| H100 | cuTile | 108 | 4 | 0 | 1 |
| H100 | TileLang | 103 | 10 | 0 | 0 |

这些非 pass 状态与全量前固定表的显式能力边界一致；没有把编译失败改写成 pass，也没有用慢路径冒充支持。两份 CSV 的 schema、case 顺序、scope、status 和所有 source 数字均保持不变，只更新本轮实测 generated p50/p95。

## 3. 全量暴露并关闭的问题

### 3.1 已有 source 被 generated-only runner 挡住

初次全量中，下面八个 Triton repro 在加载 source runtime 后找不到 `upstream()`，因此旧固定表虽然有 source 数字，却不能由当前统一入口复现：

| Kernel | 上游实现 | 接线方式 |
|---|---|---|
| `layer_norm` | FlashAttention | 直接调用单 pass LayerNorm kernel；output/mean/rstd 复用 |
| `rms_norm` | Liger Kernel | 直接调用 RMSNorm forward kernel；output/rstd 复用 |
| `fused_add_rms_norm` | Liger Kernel | 直接调用 fused kernel；output/residual/rstd 复用 |
| `cross_entropy` | Liger Kernel | 直接调用 fused CE kernel；loss/predicted 复用，保留其原地 logits 语义 |
| `attention_bias` | FlashAttention | 直接调用 forward kernel；output/LSE/tmp 复用，使用同算法的 vector bias |
| `paged_attention` | xFormers | 复用现有 paged split-K adapter；页表和 ABI 变换只准备一次 |
| `swiglu_forward` | Liger Kernel | 直接调用 forward kernel；output 复用 |
| `swiglu_backward` | Liger Kernel | 直接调用上游原地 backward kernel |

这些改动只恢复 source/runtime 接线，没有改上游 kernel，也没有把 adapter 的分配、页表构造或 ABI 转换塞进 kernel-only 计时。八条在两台机器上都重新通过数值对照。

### 3.2 generated-only 路由误删

上一轮确认 `continuous_gqa_decode / TileLang` 的旧 source 是 split-K pipeline，而 generated 是单 kernel streaming 算法，因此撤掉 source 数字是正确的；但 runner 同时失去了该格子的 generated-only 路由，导致全量报 `unsupported repro`。

修复只恢复 generated-only dispatch，不恢复无效 baseline。最终数字：

- 5090 TileLang：`0.7502 ms`；
- H100 TileLang：`0.8130 ms`；
- 两边均通过数值对照，source 列继续为空。

### 3.3 AdamW 变体不是同一个算法

原 fused AdamW 的分母是：

```text
sqrt(corrected_second) + epsilon
```

两阶段变体与它自己的 reference 原先却是：

```text
sqrt(corrected_second + epsilon)
```

全量把这个差异暴露为三家同时数值失败。修复没有改变 Kernel IR 或 realizer，而是把作者 DSL 与 reference 改回原算法。修复后：

| 设备 | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| 5090 | 0.1147 ms | 0.1126 ms | 0.1024 ms |
| H100 | 0.1078 ms | 0.1080 ms | 0.1104 ms |

六格的 fused/reference、split/reference、split/fused 三组误差都通过既有容差。

H100 TileLang 在新源码首次编译时还暴露了一个独立环境问题：默认 `/usr/bin/nvcc` 是 CUDA 11.5，不能接受 `-arch=sm_90a`。直接编译单个候选取得的真实错误是 `Value 'sm_90a' is not defined`。把该 H100 TileLang 进程的 CUDA compiler 设为机器已有的 `/usr/local/cuda-12.2/bin/nvcc` 后，13 个候选正常编译并通过。这不是 AdamW、Plan 或 TileLang leaf 的能力问题，也没有为它增加算子分支。

## 4. 性能回归复核

### 4.1 为什么没有直接采用初次全量的异常值

初次全量里有几项相对旧表突然慢 20% 到 140%。这些 kernel 的算法、Plan 和 emitter 在本轮没有改动，所以先定向复测，而不是把一次值写进固定表或为 kernel 开特例。

| 设备 / 格子 | 初次全量 | 定向复测 | 旧固定表 | 结论 |
|---|---:|---:|---:|---|
| 5090 TileLang `moe_align_block` | 0.0963 | 0.0405 | 0.0397 | 非稳定回归 |
| 5090 Triton `layer_norm_backward` | 0.1277 | 0.0706 | 0.0710 | 非稳定回归 |
| 5090 TileLang `csr_spmm` | 0.0306/0.0314 | 0.0212 | 0.0213 | 同一候选的运行态漂移 |
| 5090 TileLang `batch_norm_training` | 0.0342/0.0324 | 0.0268 | 0.0280 | 第二轮改进仍成立 |
| H100 Triton `cross_entropy` | 0.2346 | 0.1930 | 0.1918 | 非稳定回归 |
| H100 Triton `fp8_mqa_logits` | 0.0572 | 0.0495 | 0.0497 | 非稳定回归 |
| H100 Triton `varlen_gqa_prefill` | 5.2507 | 4.8214 | 4.8615 | 非稳定回归 |

其它超过 5% 的短核变化只有 1–2 微秒，没有改变算法、Plan 或 threshold 结论。这里没有用“抖动”一词替代复测；上表每项都重新执行了完整 repro，明显异常均得到第二个实际数值。

### 4.2 前一轮性能修复是否保住

前一轮的主要修复在全量下均保留：

- `histogram / Triton`：5090 `1.2755/1.2740 = 1.001x`，H100 `2.1636/2.1647 = 0.999x`；
- `ordered_prefix / Triton`：5090 `0.0191/0.0203 = 0.941x`，H100 `0.0191/0.0205 = 0.932x`；
- `online_softmax / TileLang`：5090 `0.3603/0.3697 = 0.975x`，H100 `0.1844/0.1826 = 1.010x`；
- `w4a8_packed / TileLang`：5090 `0.0770/0.0895 = 0.860x`，H100 `0.0986/0.1579 = 0.624x`；
- RoPE 的 cuTile full/partial 在两台机器均不慢于 source；inverse 在 H100 为 `1.003x`，5090 仍是既有 `1.063x` 缺口；
- `batch_norm_training` 的三 provider generated 路径全部通过；其 Triton 与 source 的剩余差距仍按下节列为物理映射问题。

### 4.3 H100 softmax backward 的新稳定事实

H100 Triton `softmax_backward` 初次全量为 `0.1024 ms`，随后两次独立复测为 `0.1007 ms` 和 `0.1018 ms`；同次上游为 `0.0927 ms`。旧固定表是 `0.0925 ms`。

这一变化不是前两轮代码回归：当前与旧表对应的 DSL、Kernel IR、Plan、generated source 和 `BLOCK_SIZE=8192`/row-autotune 路径没有变化，source 同机仍在原区间。它因此不能被写成“修复失败”，但也不能继续靠旧表把它算进门槛内。本轮固定表采用最后一次稳定实测 `0.1018 ms`，把它列为尚未完成物理/投影归因的新性能事实。

## 5. Baseline 覆盖率

当前有 `113` 个 runner，其中 `20` 个 `variant_*` 是同一算法的另一种 DSL 分解，不进入独立算法覆盖率。独立算法数为 `93`。

- 至少一家有严格 source baseline 的独立算法：`46`；
- 当前可复现覆盖率：`46/93 = 49.46%`。

第一轮曾按 `47/93 = 50.54%` 报告达标；随后算法对齐审计证明 `continuous_gqa_decode / TileLang` 的 source 是两段 split-K pipeline，而 generated 是单 streaming kernel，调用数和结构不一致，因此那一项必须撤掉。撤掉后覆盖率回到 `49.46%`。本轮没有用 variant 重复计数，也没有为了维持“超过 50%”保留一个无效 baseline。

这意味着 baseline 数量目标在严格审计后仍差一个独立算法；与此同时，现有 58 个 provider 格子已经重新做到统一入口可执行，而不再只是 CSV 中的历史数字。

## 6. 最终 5% 门槛清单

下面只列当前仍超过 5% 的严格可比格子。比值均来自本轮最终固定表。

| 算法 / Provider | 5090 | H100 | 当前定性 |
|---|---:|---:|---|
| `block_sparse_attention / TileLang` | 2.656x | 13.949x | 已用原生 `T.gemm`；TileLang layout/GEMM 下层边界 |
| `varlen_gqa_prefill / TileLang` | 1.530x | 1.930x | ordered-ragged、GQA、visible-prefix 的共享联合物理决定未闭合 |
| `mamba_chunk_scan / TileLang` | 1.371x | 1.881x | 已用目标原生 scan/GEMM；目标下层质量 |
| `varlen_attention / TileLang` | 1.320x | 2.065x | ordered-ragged query/K tile 与 mask/contract 联合决定未闭合 |
| `attention_backward / cuTile` | 1.183x | 1.874x | 三阶段 scope 与原生 contraction/reduction 已对齐；目标下层质量 |
| `attention / cuTile` | 1.122x | 1.664x | 原生 `ct.mma` 已对齐；目标下层质量 |
| `attention / Triton` | 1.101x | 1.355x | descriptor/TMA/warp-specialized 投影能力尚未闭合 |
| `layer_norm_backward / Triton` | 1.130x | — | many-to-one partial accumulation realization 尚未闭合 |
| `rope_qk_inverse / cuTile` | 1.063x | — | 二维与扁平 program-space 结构选择尚未闭合 |
| `batch_norm_training / Triton` | — | 1.305x | chunk Welford 与 lane-local 二维 Welford 的物理映射差距 |
| `mla_prefill / cuTile` | — | 1.267x | 已用原生 `ct.mma`；目标 tuner/下层质量 |
| `softmax_backward / Triton` | — | 1.124x | 本轮稳定新事实；尚未完成物理决定/投影归因 |
| `swiglu_forward / cuTile` | — | 1.106x | 原生 pointwise 路径；目标 tuner/下层质量 |
| `triangular_solve / Triton` | — | 1.093x | diagonal/RHS lane 化物理映射尚未闭合 |

所以“5% 门槛”目前是测量事实，不是已经兑现的全局保证。当前 81.90% 严格格子达标；其余不能用缩小 scope、改变算法、按 kernel 名特化或拿旧数字覆盖当前实测来消失。

## 7. 性能层面的最终状态

这轮以后，编译器在性能层面可以准确描述为：

- 三个 provider、两台结构不同的 GPU 上，113 个 runner 的统一生成与执行路径已经被一次完整矩阵验证；
- 严格 baseline 格子的多数（`95/116`）在同算法、同调用数、同 scope 下不慢于 source 5%；
- 前两轮新增的算法对齐、shared Plan 事实和 target-local 原语投影在全量下没有互相冲突；
- 固定表中的 source 不再有八个“有数字但统一入口不能运行”的空壳接线；
- 复杂 attention、流式统计、Triton descriptor/TMA 和少数目标下层质量仍形成清楚的性能边界；
- 独立算法 baseline 覆盖率经严格审计为 `49.46%`，尚未达到真实的 50% 以上；
- 因此编译器已经具备广泛、可复现的性能基础，但还不能诚实宣称“所有严格 baseline 均在 5% 内”或“性能收尾完成”。

本轮没有修改 canonical Kernel IR、Physical Plan、realizer 或 emitter，也没有新增任何 kernel 名分支。修复只落在 source runtime 接线、generated-only dispatch 和一份原本不等价的 DSL 变体上。
