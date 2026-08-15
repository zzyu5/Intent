# Plan 收缩后的跨设备全量核验

> 后续状态：本报告第八节的挂账、E8M0 完整 dtype 合同与非 pass 状态分类，已在 [编译器边界复审与遗留关闭](compiler-boundary-reaudit.md) 中完成复核与关闭。

## 结论

这一轮完成了上一轮 Physical Plan 大范围收缩后的第一次双机全量核验。RTX 5090 与 H100 在同一秒启动，分别独立完成全部 runner；没有发现 Plan 字段删减导致的数值回归，也没有出现原有 `pass` 退化为失败的格子。

最初登记的 `grouped_query_head_add` 跨设备不一致不是同一代码基线上的设备差异：5090 表中的失败来自修复前记录，H100 表中的通过来自修复后记录。本轮用同一份 Kernel IR、Physical Plan 和当前代码重跑，两台设备的 TileLang 都通过。全量另外发现了真正的跨设备 leaf 投影缺口：TileLang 的 E8M0 FP8 到 f32 转换在 sm120 能编译、在 sm90a 不能编译；它已经用不区分架构的位级语义投影修复，并在两台设备上分别跑通。

`private_workspace` 的两个预算系数和 `partition(count=P)` 都经过了真实语料审计。本轮没有发现足以支持修改它们的证据，因此二者均维持现状：前者继续挂账，后者继续由 frontend 明确拒绝。

## 1. 核验基线与并行执行

两台机器都从提交 `7fe9b5e` 的干净代码快照开始：

- 5090 使用当前项目目录；
- H100 使用单独的干净临时快照，没有借用 H100 上可能陈旧或带本地修改的活动目录；
- 两端都通过既有的 `examples/run/repro.sh <provider> <kernel>` 路径完成 DSL → Kernel MLIR → Physical Plan → target source → 下层编译 → GPU 数值与计时；
- 每台机器内部按 provider 顺序执行，避免同一张 GPU 上多个 benchmark 相互干扰；两台机器之间完全并行。

两端的机器时间起点相同，均为 epoch `1786788880`。实际耗时为：

| 设备 | runner/provider 组合 | 耗时 |
|---|---:|---:|
| RTX 5090 | 109 × 3 = 327 | 5538 秒，约 92 分 18 秒 |
| H100 | 109 × 3 = 327 | 5278 秒，约 87 分 58 秒 |

一个 runner 可以产出多个 case，因此最终 CSV 均为 118 条 kernel/case 记录，而不是 109 条。

## 2. `grouped_query_head_add` 的真实根因

### 2.1 为什么旧表看起来像跨设备失败

旧的 5090 TileLang 日志稳定失败，错误来自 sm120 下生成的 FP16 bulk fill 路径：生成代码把 `cutlass::half_t` 传给了只接受 CUDA `half` 的 `tl::pack_float16x4`。但提交历史的同机 A/B 表明，这个失败在 `fcb9016` 之前稳定存在，在 `fcb9016` 之后已经稳定消失。旧的 5090 CSV 没有在该修复后刷新，而 H100 那一格是在修复后测出的。

所以旧表混合了两个代码时间点，不能据此推出“同一 target 在两台设备上语义不一致”。本轮同基线全量结果为：

| 设备 | TileLang 数值 | p50 / p95 |
|---|---:|---:|
| RTX 5090 | 最大误差 0 | 0.0036 / 0.0056 ms |
| H100 | 最大误差 0 | 0.0055 / 0.0059 ms |

### 2.2 `fcb9016` 修掉的共享判据

问题并不是 GQA 名字或头映射本身，而是 validity provenance 被压得过宽。修复后的 `BoundaryNeutralizationProof` 只让“可以唯一绑定到 load result tensor axis 的 domain”参与消费者中和证明。对 `key[key_head, tokens]`，token lane 的边界事实现在能沿真实 value provenance 绑定，TileLang 走有条件的整块 copy/clear，而不再落入会实例化错误 FP16 fill 的路径。

这是一处共享合法性证明修正，不是 GQA 特判。

### 2.3 暴露面

对现有全部索引类算子做了条件审计。能够同时满足“F16 load、由标量 SSA 派生的索引、存在 runtime padding/check-bounds、不是 tensor-indirect、未被 `assume_in_bounds` 排除”的已确认实例只有 `grouped_query_head_add`。相似算子分别因为使用 f32、数据依赖张量索引、store/scatter 路径或显式 in-bounds 前置条件而不进入该路径。

因此没有证据表明还有一批 5090 case 被相同问题静默影响；`rotary_embedding_equivalent_index` 只有相似索引形态，但没有证据表明它会形成同一 boundary-fill transfer，不能把它写成已暴露问题。

## 3. 全量新发现：TileLang E8M0 转换并非跨架构闭合

全量中的 `block_scaled_matmul` 在 5090 TileLang 通过，在 H100 TileLang 的所有候选上编译失败。把每个候选从 autotuner 中拆开后，sm90a 的 NVCC 错误被定位为：

```text
fp8_e8_t -> float 没有可用的转换函数
```

原 leaf 对所有 cast 都机械打印 `T.cast`。TileLang 0.1.13 在 sm120 的下层路径恰好接受 E8M0 直接转换，但 sm90a 的 CUTLASS 类型没有提供同一 C++ 转换。这是 target leaf 的拼写不完整，不是算法、Plan、资源容量或 tuner 决策问题。

修复只作用于 canonical `float8_e8m0fnu -> f32` cast：

1. 用 `T.reinterpret(..., T.uint8)` 取得指数编码；
2. 用 `T.exp2(float(bits) - 127)` 机械兑现正常值；
3. 对 `0xff` 机械兑现该 dtype 的 NaN 编码；
4. 标量和 tensor cast 共享同一个表达式生成函数；
5. 不检查 GPU 型号，也不在共享层增加机制。

修复后的定向 repro：

| 设备 | 最大误差 | p50 / p95 |
|---|---:|---:|
| RTX 5090 | `7.629e-06` | 0.0369 / 0.0390 ms |
| H100 | `7.629e-06` | 0.0478 / 0.0487 ms |

这说明该缺口是跨架构 target surface 合同，而不是某一设备专属分支。

## 4. `private_workspace` 驻留预算审计

当前 GPU realizer 的两个经验预算仍为：

- scalar array budget：`registers_per_unit / 1024`；
- local vector budget：`registers_per_unit / 128`。

5090 与 H100 当前都报告每 SM 65536 个寄存器，因此两台机器分别得到 64 和 512。两台设备并不能扰动这两个系数，但现有真实算子确实覆盖了预算边界，而不仅是看代码猜测：

| 真实算子 | 触发的选择 | 边界证据 | 5090 Triton 结果 |
|---|---|---|---:|
| `viterbi_decode` | scalar array + predecessor workspace | 两个 64 元素 scalar array，正好落在 scalar budget | pass，8.4866 ms |
| `smith_waterman` | private vector | 两个 129 元素数组各取整为 256，总量正好 512 | pass，1.1801 ms |
| `insertion_top_k` | scalar array | 小型有序状态 | pass，0.5224 ms |
| `bitonic_sort` | workspace | XOR 型非结构动态访问 | pass，1.4047 ms |
| `greedy_nms` | workspace | 动态集合访问 | pass，27.7473 ms |
| `radix2_fft` | workspace | 两个可变索引状态缓冲 | pass，1.3514 ms |

这些选择与算法的访问结构一致，也没有出现“本该在 register/local 却被错误赶到 workspace”或反向情况。历史上两种更激进的规则已经被真实结果否决：只要容量放得下就落 local 会使 cuTile 编译资源爆炸；仅凭 in-bounds 就落 local vector 会使 bitonic 变慢。

本轮结论是：**扫过了真实边界，没有找到这两个系数选错的证据，所以不改规则。**它们仍是仅在相同寄存器容量设备上验证过的经验系数，这一点继续作为未决项，而不是被包装成已证明的通用规律。

## 5. `partition(count=P)` 需求审计

审计范围包含当前两份 CSV 的 118 条记录、109 个 runner，以及全部 DSL kernel：

- 25 个文件中共有 67 处 `I.partition(...)`；
- 使用 `count=` 的真实算子为 0；
- frontend 仍在源码位置明确拒绝 count，而不是生成 IR 后在深层失败。

重点检查了看起来最可能需要固定 part 数的结构：paged split-K、block-sparse、LayerNorm backward 分组、causal-conv backward 和 split-K reducer。它们当前都需要作者可见的 offset、partial-buffer ABI 或固定 stage identity；`extent` 与作者外层编排已经完整表达了算法。把它们改写成 count 反而会改变作者写下的程序结构，并不只是补一个物理决定。

本轮结论是：**扫过了全部当前语料，仍没有真实算法只能用 count 而不能用 extent 表达。**因此没有实现 GPU count realizer，frontend 的明确拒绝保持不变。

## 6. Plan 收缩的全量结果

上一轮从 `StageOp`、`StageAxisOp`、`Range`、scan/pointwise 等 Plan 节点中删除了可以从 Kernel IR 或其他已选决定唯一重算的字段。直接受影响的 moe、unique 等 repro 当时通过，但这不足以证明不常见组合没有依赖旧字段。

本次 327 × 2 的全量覆盖了：

- 多阶段与私有 intermediate；
- ragged/member identity；
- ordered stream 与 scan；
- staged contraction；
- attention 的 streaming、分页和变长组合；
- convolution、workspace、动态控制流和多种 decomposition variant。

最终状态计数：

| 设备 | Triton | cuTile | TileLang |
|---|---|---|---|
| RTX 5090 | 116 pass / 1 unsupported / 1 compile-timeout | 116 pass / 2 unsupported | 106 pass / 12 unsupported |
| H100 | 116 pass / 1 unsupported / 1 compile-timeout | 114 pass / 3 unsupported / 1 compile-timeout | 107 pass / 11 unsupported |

相对上一次固定表，没有 pass → failed/unsupported/timeout 的退化。新增通过为：

- 5090 TileLang：`grouped_query_head_add`，旧表是修复前残留；
- 5090/H100 TileLang：`mamba_chunk_scan`；
- 5090 cuTile：`token_sparse_mla_prefill`，首次编译约 66 秒后完成；
- H100 TileLang：`block_scaled_matmul` 在本轮 leaf 修复后重新通过。

H100 的 `token_sparse_mla_prefill` 仍不是 capability unsupported：Triton 超过整项 900 秒，cuTile 能进入候选编译，但每个候选超过当前下层的单候选编译时限，最终没有有效配置。它保持 `compile_timeout`，没有被错误改写成能力边界。

由此可以确认：被删除的 Plan 字段没有在未覆盖的算子形状中承担独立真理；leaf 可以从 Kernel IR 与剩余 Physical Plan 唯一取得它们需要的信息。

## 7. 性能变化的归因

原始全量里有几处看起来很大的慢化。为避免把一次抖动固化进 CSV，本轮只对绝对值或比例明显的格子定向复测，并在同一台机器上运行 Plan 收缩前的提交 `6ed16e7` 做 A/B。

### 7.1 能排除为本轮代码回归的项

5090 上，原始全量中的以下尖峰在当前代码定向复测后消失：

- Triton `ordered_prefix`：0.0619 → 0.0397 ms；
- Triton `moe_align_block`：0.2005 → 0.1133 ms；
- Triton `group_norm_silu_backward`：0.0833 → 0.0499 ms；
- cuTile `group_norm_silu_backward`：0.0710 → 0.0459 ms；
- TileLang grouped-GEMM empty-groups：0.0857 → 0.0478 ms；
- TileLang `selective_scan`：0.1199 → 0.0998 ms。

这些是单次全量测量尖峰，不对应生成结构变化。最终表采用定向复测值。

H100 上几处稳定差异在旧提交同机复现：

| 项 | 当前提交 | Plan 收缩前提交 | 上一次固定表 |
|---|---:|---:|---:|
| cuTile embedding lookup | 0.1437 | 0.1437 | 0.0813 ms |
| cuTile paged split-K attention | 0.6175 | 0.6175 | 0.3917 ms |
| cuTile paged attention | 0.6296 | 0.6338 | 0.4134 ms |
| cuTile varlen GQA logits | 0.1489 | 0.1489 | 0.1189 ms |
| Triton grouped GEMM base | 1.5424 | 1.5321 | 1.3889 ms |

旧提交与当前提交在同一机器、当前环境中一致，足以排除 Plan 收缩和本轮 leaf 改动。变化来自固定表测量之后的下层编译/候选选择或机器运行状态，而不是当前源码回归。

5090 的 paged cuTile 三项也得到同样结论：Plan 收缩前后同机分别为 0.2732/0.2718、0.3434/0.3436、0.1049/0.1050 ms。`moe_align_block` 的 cuTile 差异则能进一步看到 tuner 在接近的 scan 候选间翻转：旧运行选择 tile 64，当前运行选择 tile 128；生成的算法结构没有变化。按照既定边界，不在共享层为这种下层候选抖动建立 cost model。

### 7.2 当前赢家分布

按每条 kernel/case 的 generated p50 取通过 provider 的最小值：

| 设备 | Triton | cuTile | TileLang | 有至少一个通过 provider 的记录 |
|---|---:|---:|---:|---:|
| RTX 5090 | 48 | 26 | 44 | 118 |
| H100 | 53.5 | 33.5 | 30 | 117 |

小数来自并列最优的均分。两台机器的赢家分布明显不同，说明三种 surface 并不是固定由同一家全面占优；同一算法与 Physical Plan 投影到不同下层后，设备与下层实现仍会改变最终赢家。

## 8. 固定产物与未决项

已更新但未改变结构的固定表：

- `report/baseline/kernel-performance.csv`：RTX 5090；
- `report/baseline/kernel-performance-h100.csv`：H100。

两表保持原有 20 列、118 条记录、case 顺序和 source baseline 数值；没有合并、没有版本号，也没有加入检查逻辑。

本轮没有遗留 correctness 失败。仍明确挂账的只有：

1. `private_workspace` 两个预算系数尚未被寄存器容量不同的真实设备扰动，但现有真实边界没有显示其错误；
2. `partition(count=P)` 语义存在，当前语料没有真实需求，frontend 继续明确拒绝；
3. 少数下层首次编译超时仍按 compile-time cost 记录，不冒充 capability unsupported。
