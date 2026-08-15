# 双设备全量回归与成熟编译器收尾

## 结论

这一轮完成了前三轮改动叠加后的第一次双设备全量回归，并把全量中暴露的两个真实回归修到根因：

1. `fp8_mqa_logits` 的 head lane 在 contraction 角色尚未分配时被过早提升成独立 program 轴。结果仍然正确，但 Triton 在 RTX 5090 和 H100 上分别退化约 44% 和 20%。修复是调整共享轴角色分配的先后关系，使 pointwise lane 的 program ownership 只在 contraction/reduction/scan 等结构角色全部明确之后决定。
2. cuTile 的间接读取曾统一改成 masked gather。RTX 5090 从中受益，但 H100 的 paged attention 从约 0.63 ms 退化到约 0.81 ms。两种写法语义相同、两台设备的赢家相反，因此这不是 Plan 决策，也不能按架构分支。现在它是 cuTile 目标内部的离散拼写候选，由 cuTile 自带 tuner 在 `masked gather` 与 `gather + where` 之间实测选择。

全量成功的格子没有数值回归。修复后，受两项改动影响的所有现有消费者都做了两机定向复验，数值继续通过。两份 baseline CSV 已按同一格式更新；source/upstream 数字、计时 scope、行顺序和字段结构均未改变。

在当前声明的 GPU 能力子集内，这套系统已经可以称为一套成熟的单算子编译器：算法只在 canonical Kernel MLIR 中表达，Physical Plan 只保存已选物理决定，三个 target leaf 只做能力检查、目标原语投影和运行接线；113 个 kernel runner、122 条 kernel/case 记录在两种明显不同的 GPU 上完成了全链路核验。这里的“成熟”不等于“所有目标都支持所有构造”，也不包含尚未接入的 CPU、RISC-V 和 RVV。

## 1. 全量范围与执行方式

两台机器使用同一代码基线、彼此独立的工作目录和各自既有的 Triton、cuTile、TileLang 环境。5090 与 H100 同时启动，不在一台结束后才启动另一台。

每台机器执行：

- 113 个 kernel runner；
- 每个 runner 分别走 Triton、cuTile、TileLang；
- 合计 339 次 `DSL -> Kernel MLIR -> Physical Plan -> target source -> 下层编译 -> GPU 数值与计时`；
- 多 case runner 展开后形成 122 条 baseline 记录。

执行耗时按各 runner 记录的 elapsed time 求和：

| 设备 | 调用数 | 累计耗时 | 两机关系 |
|---|---:|---:|---|
| RTX 5090 | 339 | 2569 秒，约 42 分 49 秒 | 与 H100 并行 |
| H100 | 339 | 2626 秒，约 43 分 46 秒 | 与 5090 并行 |

两机并行后的主等待时间约等于较慢一侧，而不是两者相加。H100 使用独立临时快照；没有覆盖远端活动仓库中的本地状态。

## 2. 全量正确性与状态

所有返回成功的日志均包含明确的 reference/equivalent-decomposition 数值通过标记；没有发现“进程返回 0，但日志中出现 NaN、Traceback 或数值失败”的隐藏格子。

更新后的状态计数如下。计数以 122 条 kernel/case 记录为单位：

| 设备 | Triton | cuTile | TileLang |
|---|---|---|---|
| RTX 5090 | 120 pass / 1 unsupported / 1 compile-failed | 120 pass / 2 unsupported | 107 pass / 11 unsupported / 1 compile-failed / 3 failed |
| H100 | 120 pass / 1 unsupported / 1 compile-failed | 118 pass / 3 unsupported / 1 compile-timeout | 111 pass / 11 unsupported |

相对上一份固定表，唯一状态文字变化是两台机器上的 Triton `token_sparse_mla_prefill`：从 `compile_timeout` 改为 `compile_failed`。本次日志已经得到确定错误：生成的三维 tensor 有 `64 * 64 * 512 = 2,097,152` 个元素，超过当前 Triton 单 tensor `1,048,576` 的编译限制。它不是本轮从 pass 退化，也不再用 timeout 模糊真正原因。

没有旧 `pass` 变成 `failed`、`unsupported` 或错误结果的记录。

## 3. 回归一：FP8 MQA 的轴角色分配时序

### 3.1 现象与同机 A/B

`fp8_mqa_logits` 在全量中数值正确，但 Triton 明显慢于上一代码基线。同一台机器、同一环境的前后 A/B 为：

| 设备 | 修复前当前代码 | 较早代码 | 修复后 |
|---|---:|---:|---:|
| RTX 5090 | 约 0.0528–0.0530 ms | 约 0.0364–0.0368 ms | 约 0.0367–0.0369 ms |
| H100 | 约 0.063 ms | 约 0.052 ms | 约 0.0493–0.0520 ms |

同机旧代码能够恢复原性能，排除了测量环境和下层版本变化。

### 3.2 根因

上一轮为纯 pointwise lane 增加 program ownership 时，提升判断发生在 contraction 的 M/N 角色分配之前。此时 head domain 的直接消费者看起来只有 load/store，`canDistributePointwiseLane` 因而认为它可以成为独立 parallel program 轴；稍后 contraction 分析又给同一轴添加 `contraction_m`。

错误 Plan 最终包含：

```text
["lane", "parallel", "contraction_m"]
```

这使生成代码增加了 head 方向的 program grid、`BLOCK_SIZE_M` 和 group swizzle；但该轴随后参与 contraction，已不再满足“主体只有纯逐元素语义”的打包前提。问题不是 FP8、MQA 或某个 shape 的特例，而是共享角色分配读取了尚未完成的事实。

### 3.3 修复

共享 GPU realizer 现在先完成 contraction matrix-axis 角色分配，再尝试把仍然只有 `lane` 单一角色的 pointwise domain 提升为 parallel program axis。

这没有改写 Kernel IR，也没有新增 kernel 名字分支。它只是保证一个物理选择在其依赖的结构事实完整之后作出。

全量 Plan 审计显示，只有 `fp8_mqa_logits` 出现过矛盾的 `lane + parallel + contraction_m` 组合。真正需要 pointwise 二维 ownership 的 `embedding_forward_lookup` 仍保持 `lane + parallel`，并在三个 target 上复验通过、性能未退回旧的整行映射。

cuTile 和 TileLang 对该 MQA 形态仍在发射前给出明确、带源码位置的 unsupported 诊断：它们当前不能把 runtime-sized lane 机械投影为 matrix-M。这是 target capability 子集，不用串行慢路径冒充支持。

## 4. 回归二：cuTile gather 拼写不能在两台机器上固定为同一答案

### 4.1 并排源码与同机 A/B

H100 的 cuTile paged attention 在前一轮把：

```python
value = ct.gather(..., check_bounds=True)
value = ct.where(valid, value, fill)
```

收成：

```python
value = ct.gather(..., mask=valid, check_bounds=True)
```

之后，从旧代码的约 0.625–0.636 ms 退到约 0.811–0.816 ms。并排生成源码后，唯一会改变该访存路径的差异就是这两种拼写。

RTX 5090 的结果方向相反：

- masked gather：约 0.326–0.328 ms；
- gather 后 where：约 0.342–0.344 ms。

由此可以排除三种错误修法：

- 不能把 masked 或 post-where 永久写死；
- 不能按 `sm90`/`sm120` 或 GPU 名字分支；
- 不能把 cuTile 私有的语法选择抬进共享 Kernel IR 或 Physical Plan。

### 4.2 正确边界

两种写法使用同一份 indices、validity 和 fill，算法语义、所有权、分块与边界决定都相同。变化只在于当前 cuTile 用哪种原生表面语法兑现这份决定。因此它属于 target leaf 内部可以委托给下层 tuner 的拼写候选，而不是新的机器决定。

cuTile emitter 现在只在真实存在 guarded indirect/staged gather 时增加 `GATHER_SPELLING: ConstInt`。两个值生成两个编译期专门化版本：

- `1`：masked `ct.gather`；
- `0`：普通 `ct.gather` 后接 `ct.where`。

cuTile 的 `exhaustive_search` 与原有 tile/occupancy 候选一起实测这两个版本。候选不会进入共享 `intent_plan.search_space`，Triton 和 TileLang 也看不到 cuTile 的目标语法概念。

### 4.3 结果与影响范围

Paged attention 修复后：

| 设备 | cuTile 选择 | p50 | 数值 |
|---|---|---:|---|
| RTX 5090 | masked gather | 约 0.3260–0.3272 ms | pass |
| H100 | gather + where | 约 0.6316–0.6337 ms | pass |

这是一条真实的跨设备反例：同一条固定规则无法同时让两台机器正确选择性能赢家，因此该维度应搜索，而不是继续扩大静态规则的设备输入。

影响面按“谁读取 guarded-gather 拼写”确定，而不是按 paged-attention 名字确定。两台机器均定向复验了：

- paged attention；
- paged MLA decode；
- paged split-K attention；
- MoE；
- grouped GEMM；
- 普通 cuTile softmax 和 GEMM，用于确认没有把 synthetic target 参数泄漏给无关 kernel。

全部数值通过。grouped GEMM 也因 tuner 能选择更合适的拼写而显著改善：

| 设备 | case | 旧固定表 | 当前 p50 |
|---|---|---:|---:|
| RTX 5090 | base | 1.4695 ms | 1.3400 ms |
| RTX 5090 | tail | 1.4704 ms | 1.3482 ms |
| H100 | base | 1.9021 ms | 1.1918 ms |
| H100 | tail | 2.1028 ms | 1.3519 ms |

H100 cuTile MoE 也从 12.2624 ms 收到 10.2493 ms。这里没有改变作者算法或共享 Plan，只是让下层为当前设备选择自己的等价拼写。

## 5. 其余性能变化如何处理

全量结果没有直接把每一次变化都固化为“代码提升”或“代码回归”。对比例或绝对值明显的项重复测量；有怀疑时，还在同一机器、当前环境中运行较早代码做 A/B。

### 5.1 5090

以下全量尖峰在定向重复后恢复到旧表附近或更好：triangular solve、GroupNorm backward、MoE align、ordered prefix、right-looking Cholesky variant、cuTile FFT。因此最终表使用定向重复的中位数。

TileLang `csr_spmm` 与 `selective_scan` 分别出现约 0.021/0.030 ms、0.100/0.119 ms 的双峰。相同生成代码和较早提交在当前机器上也出现同样双峰，没有源码或 Plan 差异可以归因。本表如实保存本轮重复样本的中位数 0.0305 ms 和 0.1196 ms，没有把较快的偶发样本挑出来维持旧数字。

### 5.2 H100

对 attention p95、varlen GQA prefill、ROI Align、TileLang bf16 GEMM 和 Triton MoE 做了定向重复，并用较早提交在同机当前环境复现。较早提交和当前代码落在同一分布，生成源码也没有相关结构差异，因此不能归因给前三轮代码。

本轮固定值采用重复样本中位数。典型当前 p50 为：

- TileLang dense attention：约 3.9185 ms；
- Triton varlen GQA prefill：约 5.1006 ms；
- Triton ROI Align：约 1.1528 ms；
- TileLang bf16 GEMM：约 1.7179 ms；
- Triton MoE：约 7.7515 ms。

这些变化没有被“抖动”一句带过；同机旧代码 A/B 是排除本轮代码回归的依据。

## 6. 两份固定表与赢家分布

更新后的文件仍是两张互相独立的表：

- `report/baseline/kernel-performance.csv`：RTX 5090；
- `report/baseline/kernel-performance-h100.csv`：H100。

两表保持 20 列、122 条 kernel/case 记录、相同顺序。没有合并、版本号、阈值或校验逻辑；source/upstream 列完全保持原值。

按每条记录中通过 provider 的最小 generated p50 统计：

| 设备 | 旧表 Triton / cuTile / TileLang | 新表 Triton / cuTile / TileLang | 有通过 provider 的记录 |
|---|---:|---:|---:|
| RTX 5090 | 47 / 31 / 44 | 48 / 33 / 41 | 122 |
| H100 | 51 / 35 / 35 | 55 / 30 / 36 | 121 |

三家在两台机器上都继续拥有独立赢家，且分布随设备变化。H100 少一条可选记录是因为 token-sparse MLA 当前没有任何 provider 在可接受编译范围内完成，不是把某家失败格子排除后人为制造赢家。

## 7. 收尾后的系统状态

### 7.1 真正闭合的核心

以下部分已经由代码合同和双机真实语料共同支撑，而不是只写在设计文档里：

1. **编程模型**：作者表达 domain、region、tensor algebra、ordered state、ragged relation、stage 与显式外层多调用；不写 program id、tile size、warp、pipeline、地址整数宽度或存储层级。
2. **唯一算法表示**：Python frontend 直接构造 canonical Intent Kernel MLIR；没有并行维护一套 typed Python Kernel IR。
3. **共享 realization**：逐逻辑轴组合 parallel、lane、ordered、reduction、contraction、member 等角色；Physical Plan 只记录多个合法物理答案中选中的一个，不按 kernel 类别入口匹配。
4. **目标投影边界**：Triton、cuTile、TileLang 共用 Kernel IR 遍历和 Plan。leaf 只进行能力检查、逐 op/概念到语法映射、目标接口接线，以及明确委托给下层的 target-local 候选。
5. **复杂能力**：generic reduce/scan combine 以 typed Kernel IR closure 表达；多阶段执行具有显式依赖、buffer 生命周期和可见性合同；这些不是依赖 CUDA stream 偶然顺序的隐藏语义。
6. **真实覆盖**：113 个 runner 覆盖逐元素、不同归约、dense/ragged/streaming attention、GEMM 与复杂 contraction、MoE、卷积、扫描、排序、动态规划、量化、反向、多阶段、原地更新和多个等价分解；两机三 target 的所有 pass 都经过真实运行与数值对照。
7. **跨设备选择**：本轮 gather spelling 给出了明确证据——同一物理决定在不同设备上可能需要不同的 target 原生拼写，答案由下层 tuner 实测，而不是由共享层复制架构知识。

### 7.2 有证据支撑的明确能力边界

这些状态不是“没测过”，也不与核心完成混写：

- Triton 与 cuTile 当前没有稀疏 2:4 contraction 的等价 target 原语，发射前明确 unsupported；
- cuTile 的 FP8 MQA 不能把 runtime-sized lane 投影成其 matrix-M，明确 unsupported；
- H100 cuTile 的 `float8_e8m0fnu` block-scaled matmul 不受当前 sm90 target 支持，明确 unsupported；
- TileLang 当前不支持本语料需要的 atomic compare-exchange、二维联合 read coverage、若干多轴 checked indirect transfer、split-K reducer，以及 FP8 MQA 的该矩阵形态；这些在发射前拒绝，不走慢一两个数量级的伪支持；
- token-sparse MLA 对 Triton 是确定的编译表示上限，对 H100 cuTile 是下层候选编译超时；这是编译成本/下层限制，不冒充算法或 DSL 表达缺口；
- 5090 TileLang 的三个 Q/K RoPE 在生成 CUDA 后由 `nvcc sm_120a` 崩溃，absorbed MLA 也停在下层编译；同一 RoPE 路径在 H100 通过，因此记录为下层失败而不是 target capability unsupported。

### 7.3 仍不在本次完成范围内的目标族

CPU、RISC-V 和 RVV 尚未接入。Kernel IR 中不携带 GPU tile、CTA 或 warp 所有权，因此不需要从 GPU 化的源码逆向恢复算法；但这只说明算法表示允许接入新机器族，不证明对应 realizer 和 leaf 已经存在或已经高性能。

因此准确状态是：**GPU 主线已经形成能力完整、边界显式、经过双设备大规模真实 repro 的单算子编译器；跨 CPU/RISC-V/RVV 的机器 realization 与目标投影仍是下一目标族的独立工程。**

## 8. 本轮验证闭环

本轮先完成双机全量，再对全量暴露的问题做根因修复。修复后没有机械地重跑第二次全量，而是按共享判据的真实消费者确定复验范围：

- 轴角色时序：FP8 MQA、embedding，以及全量 Plan 中所有出现 `lane + parallel`/contraction 组合的记录；
- cuTile gather spelling：paged attention、paged MLA、paged split-K、MoE、grouped GEMM，以及不应收到该参数的普通 softmax/GEMM；
- 两项均在 RTX 5090 与 H100 上运行，并检查数值与性能。

这既覆盖了改动影响面，也避免为了更新表而重复运行不读取这些判据的无关格子。最终 CSV 由全量结果与上述定向重复的中位数组成，每一行都来自本轮当前代码或经过同机 A/B 证明与当前代码一致的实测；没有复制旧 generated 数字填空。
