# Baseline 5% 性能门槛收口报告

## 1. 结论

这一轮没有达到“所有严格可比格子都在 5% 内”。真实结果是：

| 设备 | 严格可比 provider 格子 | `generated/source <= 1.05` | `> 1.05` | 门槛内比例 |
|---|---:|---:|---:|---:|
| 5090 | 58 | 47 | 11 | 81.03% |
| H100 | 58 | 46 | 12 | 79.31% |
| 合计 | 116 | 93 | 23 | 80.17% |

相对本轮开始时的固定表：

- 原表共有 118 个带 source 数字的格子，其中 82 个在 5% 内、36 个超出；
- `continuous_gqa_decode / TileLang` 的两台设备 source 实际是两段 split-K pipeline，而 generated 是一个 streaming kernel，调用数和算法结构不一致，因此撤掉这两个无效对照；
- 在余下 116 个严格格子中，本轮让 11 个格子从超标进入 5% 内；
- 最终仍有 23 个原始数值超标。其中 11 个仍是编译器自身没有关闭的物理决定或投影能力问题，12 个在并排检查生成源码后归为目标原生原语或下层编译质量边界。

以上是按既定规则冻结 source 数字后的固定表统计。提交前的 5090 定向 A/B 还暴露了一条额外事实：BatchNorm source 在不改变测试逻辑的情况下当前稳定为 `0.0394–0.0406 ms`，而 generated 为 `0.0575–0.0607 ms`，实际同机比值为 `1.46–1.50×`。依照“source 测量逻辑未改时不刷新固定 source 数字”的既定规则，本轮没有覆盖 CSV 中的 `0.0673 ms`；但这条差距仍是未关闭问题，不能因为固定表显示 0.860× 就宣称真实 A/B 已达标。

所以，这一轮得到的不是一个虚假的“全部达标”，而是一条更清楚的边界：已修掉能够由共享事实、机械投影、参数委托或算法对齐解释的差距；没有为了数字改计时口径、缩小 scope、按 kernel 名特化，或者保留一条只在单一 provider 上成立的算法改写。

## 2. 门槛口径

只有同时满足以下条件的格子进入 5% 统计：

- generated 与 source 是同一种目标语言；
- 算法结构与可观察语义一致；
- GPU kernel 调用数一致；
- 计时 scope 一致；
- generated/source 都在当前设备真实编译、运行并通过数值对照；
- source 是公开高性能实现，不是 PyTorch composition 或临时 reference。

`generated/source <= 1.05` 是原始性能事实。对于超标格子，本报告再区分：

1. **编译器责任未关闭**：Plan 少了结构选择，或 leaf 没有投影目标已经具备的能力；
2. **目标/下层边界**：生成源码已经使用该目标的原生原语，没有找到缺失的共享事实，剩余差距落在目标编译器的 layout、MMA、scan 或代码生成质量；
3. **无效对照**：算法、调用数或 scope 不一致，source 数字必须撤掉，不能拿来计算门槛。

第二类不是“数值达标”，只是按本轮约定不把下层固有质量冒充为 Intent 自己能修的缺口。第一类仍然是明确未完成项。

## 3. 本轮关闭的差距

### 3.1 结果总表

| 算法 / Provider | 5090 旧比值 → 新比值 | H100 旧比值 → 新比值 | 改动所在层 | 结论 |
|---|---:|---:|---|---|
| `histogram` / Triton | 3.055× → 1.000× | 2.330× → 1.000× | DSL 算法表达 | 两台设备进入门槛 |
| `ordered_prefix` / Triton | 1.961× → 0.941× | 1.863× → 0.912× | DSL + shared Plan | 两台设备进入门槛 |
| `online_softmax` / TileLang | 1.061× → 0.975× | 1.382× → 1.009× | target-local tuner | 两台设备进入门槛 |
| `w4a8_packed` / TileLang | 1.092× → 0.858× | 0.820× → 0.630× | TileLang leaf 投影 | 5090 进入门槛，H100 继续优于 source |
| `rope_qk_full` / cuTile | 1.000× → 1.000× | 1.068× → 1.000× | shared fact + cuTile leaf + DSL dtype 对齐 | H100 进入门槛 |
| `rope_qk_partial` / cuTile | 1.000× → 1.000× | 1.180× → 0.988× | shared fact + cuTile leaf + DSL dtype 对齐 | H100 进入门槛 |
| `rope_qk_inverse` / cuTile | 1.063× → 1.063× | 1.072× → 1.002× | shared fact + cuTile leaf + DSL dtype 对齐 | H100 进入门槛；5090 尚差 4.1 μs |
| `batch_norm_training` / Triton | 2.056× → 固定表 0.860×；当前同机 1.46–1.50× | 7.964× → 1.334× | DSL 算法表达 | 大幅收窄，但两台设备的当前真实 A/B 都仍未关闭 |

这里一共关闭 11 个设备/provider 格子。表中 W4A8 的 H100 原本已在门槛内，因此它的继续提升不重复计数。

### 3.2 Histogram：作者算法对齐，不是编译器特判

原 DSL 是“每个样本一个 program，再做一次标量 atomic add”。FlagGems source 是块内处理一段样本、以向量形式发起原子更新。二者结果相同，但程序算法结构并不相同。

DSL 现在显式写出：

- 对 sample domain 做 region partition；
- region 内批量读取样本；
- 用 region-shaped value 做 atomic add。

编译器没有识别 histogram，也没有把标量程序偷偷升级成块程序；作者主体从一开始就明确看见一个 region。结果：

- 5090 Triton：`3.8923 → 1.2746 ms`，source `1.2740 ms`；
- H100 Triton：`5.0446 → 2.1652 ms`，source `2.1647 ms`；
- cuTile 和 TileLang 也走同一份新 DSL 并通过数值对照。

### 3.3 Ordered prefix：算法对齐与 scan 的程序映射分开处理

原 DSL 是嵌套 ordered loop；上游是对一条 row-major 线性序列做原生 inclusive scan。DSL 改为显式 flatten `(row, column)`、调用 canonical `I.scan`，再按同一 row-major 索引 scatter 回输出。

这仍然是作者算法选择。共享 realizer 只修了一个真实的物理映射问题：存在物化 scan 时，不再复用外层 worker，否则同一 scan 的程序 ownership 会被错误压缩。这个判据只读 `facts.scans`，没有 kernel 名分支。

结果：

- 5090 Triton：`0.0398 → 0.0191 ms`，source `0.0203 ms`；
- H100 Triton：`0.0382 → 0.0187 ms`，source `0.0205 ms`。

因为这是共享的 scan 判定，额外定向复验了所有读取该判定的已有语料：`moe_align_block`、`unique_consecutive`、`nonzero_compact`、`sorted_nucleus_cutoff`，三种 provider、两台设备均保持数值通过。

### 3.4 Online softmax：参数性选择交回 TileLang tuner

长 stream 的 TileLang 候选原来只到 1024。stream tile 不改变算法结构，只是固定发射形态中的参数值，属于参数性选择；它应交给目标 tuner 实测，而不是在 shared Plan 里按设备写阈值。

TileLang 的 target-local 候选增加 2048、4096、8192，并让 profile 实际覆盖这些值。结果：

- 5090：`0.3924 → 0.3604 ms`，source `0.3697 ms`；
- H100：`0.2524 → 0.1843 ms`，source `0.1826 ms`。

没有新增按设备或按 kernel 的选择分支。

### 3.5 W4A8：保留 compact coverage，leaf 兑现为连续 private fragment

`k // 2` 的 quasi-affine compact coverage 已经存在于 shared facts/Plan。问题出在 TileLang leaf：当结果驻留在 private fragment 时，它没有消费这条 compact span，而是按 logical-K 逐元素读取 packed 权重。

修复严格限定为：

- 只有一个 compact access range；
- source axis 是外部 view 的最内层连续轴；
- shared 结果继续用 `T.alloc_shared`；
- private 结果用 `T.alloc_fragment`；
- 只有 shared transfer 需要 `T.sync_threads()`。

它没有在 shared Plan 中加入 TileLang 字段，也没有检查 W4A8 名字。结果：

- 5090 TileLang：`0.0977 → 0.0768 ms`，source `0.0895 ms`；
- H100 TileLang：`0.1295 → 0.0994 ms`，source `0.1579 ms`。

第一次把所有 private compact access 都走这条路径时，非连续的 `weight_only_int4` 被破坏；该宽泛实现已删除。最终判据是“最内层连续物理 span”，并复验 `weight_only_int4` 数值通过。

### 3.6 RoPE：保留作者的 translated index，而不是退化成 gather

作者写下的 `phase + HALF_DIMENSION` 在 Kernel IR 中一直存在，但 shared facts 原来只保存 quotient compact access，未保存“单一来源轴、单位系数、正常量偏移”的 translated access。cuTile leaf 因此把第二半读写退化成 gather/scatter，而上游使用两个原生 tile load/store。

修复分成三层：

1. shared facts 对精确形态 `axis + positive_constant` 记录 offset-only access range；load 和 store 共用同一来源；
2. Plan verifier 允许 offset-only range 绑定到 lane axis，Plan 继续只保存这份已经确定的访问事实；
3. cuTile leaf 仅在单一、固定 tile 对齐的正偏移上投影成 native tile load/store，并把元素偏移换算成 cuTile 的 tile coordinate。

此外 DSL 去掉了与 cuTile source 不一致的显式 f32 中间计算，恢复 source 使用的 f16 数值路径。这是算法对齐，不是为 cuTile 修改公共 ABI。

H100 三个严格 cuTile 格子全部进入门槛：

- full：`0.0641 → 0.0600 ms`，source `0.0600 ms`；
- partial：`0.0393 → 0.0329 ms`，source `0.0333 ms`；
- inverse：`0.0642 → 0.0600 ms`，source `0.0599 ms`。

5090 full/partial 与 source 持平；inverse 仍为 `0.0696 / 0.0655 = 1.063×`。两边地址运算与 load/store 已一致，剩余差异是 source 把 `B*S` 折成一维 program grid，而 generated 仍采用 `(S,B)` 二维 grid。当前没有足够证据把“折叠任意两个 parallel axis”提升为共享结构规则，因此没有用这一格制造宽泛特判。

### 3.7 BatchNorm：从全量物化改成两遍流式统计

原 DSL 一次物化完整 `B×S` region，再做两级 reduce；这不是 FlagGems source 的分块统计算法。DSL 改为：

- 第一遍沿 spatial stream；
- 每块内部计算 count/mean/M2；
- chunk 之间使用 Chan/Welford 标量合并公式；
- 第二遍流式归一化和写回；
- running statistics、saved mean/rstd 与 source 保持相同可观察语义。

结果：

- 5090 Triton：固定表为 `0.0579/0.0673 = 0.860×`；提交前两次同机复测分别为 `0.0575/0.0394 = 1.462×` 与 `0.0607/0.0406 = 1.495×`；
- H100 Triton：`0.3337 → 0.0559 ms`，source `0.0419 ms`；
- 两台设备都已比旧 generated 大幅改善，但按当前同机 A/B 都没有被记成完成。

上游在块内使用 lane-local 二维 Welford；generated 使用通用 chunk reduce 加 Chan 合并。剩余 H100 差距属于尚未关闭的流式统计物理映射，不是简单的字符串拼写差异。

## 4. 仍超过 5%：5090

| 算法 / Provider | generated/source | 超出 | 定性 | 是否仍属编译器责任 |
|---|---:|---:|---|---|
| `block_sparse_attention` / TileLang | `0.1019/0.0384 = 2.654×` | 165.36% | 两段 pipeline 与 source 一致，leaf 已使用原生 `T.gemm`；差距在 TileLang layout/GEMM 质量 | 否，目标/下层边界 |
| `varlen_gqa_prefill` / TileLang | `9.8022/6.3852 = 1.535×` | 53.51% | ordered-ragged、GQA head mapping、visible prefix 与 contraction layout 的联合物理选择尚未闭合 | 是，共享物理决定未关闭 |
| `mamba_chunk_scan` / TileLang | `0.0279/0.0205 = 1.361×` | 36.10% | 生成代码已使用目标原生 scan/GEMM 结构，未发现丢失的 shared fact | 否，目标/下层质量 |
| `varlen_attention` / TileLang | `0.3329/0.2535 = 1.313×` | 31.32% | ordered-ragged 的 query/K tile、visible-prefix、mask/contract 边界仍不是一个完整联合决定 | 是，共享物理决定未关闭 |
| `triangular_solve` / Triton | `0.0511/0.0399 = 1.281×` | 28.07% | source 对 diagonal reciprocal 与 RHS 做 lane 化；当前 generated 的递推物理映射仍偏标量 | 是，物理映射未关闭 |
| `attention_backward` / cuTile | `0.2036/0.1709 = 1.191×` | 19.13% | 三阶段 scope 对齐，生成代码使用 cuTile 原生 contraction/reduction，未发现 Plan 丢事实 | 否，目标/下层质量 |
| `layer_norm_backward` / Triton | `0.0710/0.0625 = 1.136×` | 13.60% | many-to-one partial accumulation 的 lock/atomic、partial dtype 与清零策略仍未形成更优 realization | 是，物理决定未关闭 |
| `attention` / cuTile | `5.3955/4.7954 = 1.125×` | 12.51% | 算法与候选对齐，使用原生 `ct.mma`，未找到 leaf 漏用原语 | 否，目标/下层质量 |
| `attention` / Triton | `5.4543/4.9526 = 1.101×` | 10.13% | upstream 使用 tensor descriptor/TMA 与 warp-specialized 路径；当前 leaf 仍是 raw `tl.load`/`tl.dot` | 是，Triton 投影能力缺口 |
| `rope_qk_inverse` / cuTile | `0.0696/0.0655 = 1.063×` | 6.26% | native tile access 已对齐；剩余是二维与扁平 program grid 的结构选择 | 是，program-space 选择未关闭 |
| `mla_prefill` / cuTile | `0.0431/0.0410 = 1.051×` | 5.12% | guarded bulk-read 已修，当前使用原生 `ct.mma`，未找到新的 shared 缺失事实 | 否，目标 tuner/下层质量 |

`batch_norm_training / Triton` 没出现在上表，是因为固定 CSV 的 source 数字使它成为 `0.860×`；当前两次同机 A/B 的 `1.46–1.50×` 说明它实质上仍是 5090 的流式统计物理映射缺口。这里保留固定表统计与当前实测两份事实，不用其中一份覆盖另一份。

## 5. 仍超过 5%：H100

| 算法 / Provider | generated/source | 超出 | 定性 | 是否仍属编译器责任 |
|---|---:|---:|---|---|
| `block_sparse_attention` / TileLang | `0.8197/0.0586 = 13.988×` | 1298.81% | 两段 pipeline、候选和 `T.gemm` 投影与 5090 相同；异常集中在 H100 下层 layout/GEMM | 否，目标/下层边界 |
| `varlen_attention` / TileLang | `0.3486/0.1695 = 2.057×` | 105.66% | ordered-ragged 联合物理决定未闭合 | 是，共享物理决定未关闭 |
| `varlen_gqa_prefill` / TileLang | `9.2442/4.7898 = 1.930×` | 93.00% | ordered-ragged + GQA + visible-prefix 的联合选择未闭合 | 是，共享物理决定未关闭 |
| `mamba_chunk_scan` / TileLang | `0.0299/0.0159 = 1.881×` | 88.05% | 已使用目标原生结构，未发现丢失的 Plan 事实 | 否，目标/下层质量 |
| `attention_backward` / cuTile | `0.1429/0.0769 = 1.858×` | 85.83% | 三阶段调用一致，使用 cuTile 原生 contraction/reduction | 否，目标/下层质量 |
| `attention` / cuTile | `7.5004/4.5378 = 1.653×` | 65.29% | 生成代码与候选稳定，使用原生 `ct.mma`；H100 上是既有 cuTile attention 质量缺口 | 否，目标/下层质量 |
| `attention` / Triton | `4.2023/3.1012 = 1.355×` | 35.51% | upstream TMA/descriptor/warp specialization 尚未由 leaf 机械投影 | 是，Triton 投影能力缺口 |
| `batch_norm_training` / Triton | `0.0559/0.0419 = 1.334×` | 33.41% | 通用 chunk Welford 与 source lane-local 二维 Welford 的物理映射仍有差距 | 是，物理映射未关闭 |
| `mla_prefill` / cuTile | `0.0252/0.0195 = 1.292×` | 29.23% | native `ct.mma` 已使用，未找到缺失共享事实 | 否，目标/下层质量 |
| `swiglu_forward` / cuTile | `0.1474/0.1334 = 1.105×` | 10.49% | 相同 fused pointwise 算法，原生逐元素路径；未见 leaf 退化拼写 | 否，目标 tuner/下层质量 |
| `triangular_solve` / Triton | `0.0413/0.0378 = 1.093×` | 9.26% | 与 5090 相同的 lane 化物理映射缺口 | 是，物理映射未关闭 |
| `gemm` / cuTile | `1.4010/1.3232 = 1.059×` | 5.88% | 单个 dense contraction，已使用 `ct.mma`；差距没有对应的 shared Plan 缺项 | 否，目标 tuner/下层质量 |

## 6. 为什么部分大差距没有用“修法”硬压

### 6.1 Ordered-ragged attention

现有 Plan 已经分别保存 query/stream tile、逻辑读取终点、consumer neutralization、direct/deferred transfer 和有效性信息。真正缺的是这些量与 target layout 的联合选择，而不是再复制一遍同样字段。

本轮没有为 TileLang 加专属 shared schema，也没有在 leaf 里按 attention 名字决定 tile。下一步若继续处理，必须先把联合选择的合法空间定义成共享物理决定；否则只是把第二套 attention 编译器塞进 TileLang leaf。

### 6.2 Triton dense attention

generated 与 upstream 的差异不是一个 `num_warps` 常量：upstream 使用 host tensor descriptor/TMA 和 warp specialization，generated 仍是 pointer ABI 上的 `tl.load`/`tl.dot`。Triton 本身支持这些能力，因此它不是“下层做不到”，而是 Intent 的 Triton target 尚未提供该机械投影。

这项属于真实未完成能力；本轮没有用 kernel matcher 把 attention 替换成手写模板。

### 6.3 LayerNorm backward

同机定向复测为 generated `0.0708 ms`、source `0.0667 ms`，差距约 6.1%，仍超过门槛但小于固定表的 13.6%。该差距不是简单测量抖动：源码结构上，upstream 使用 row-group lock 与部分和存储，generated 使用 bf16 many-to-one scatter-reduce/atomic。既有 lock 尝试没有改善并已删除。

这说明 collision 与 partial-storage realization 仍有选择空间；没有足够证据时不能靠固定阈值选 lock 或 atomic。

## 7. 被否决并已删除的尝试

- **W4A8 一律搬到 shared**：5090 从约 `0.0977` 恶化到 `0.1038 ms`，证明“装得下就放 shared”不是正确规则；已完整撤销。
- **所有 private compact access 都走 bulk fragment**：破坏了非最内层连续的 W4A16 读取；已收紧为可证明的最内层连续 span，宽泛路径已删除。
- **Triangular solve 在 DSL 中预计算 diagonal reciprocal tile**：Triton 有改善，但 TileLang 无法兑现同一写法；这会让公共 DSL 为单一 provider 迁就，已撤销。
- **BatchNorm 嵌套 state-stream / 多轴 reduce**：不是当前 canonical op 可机械闭合的结构；未保留半支持路径，最终采用两遍 stream + 标量 carry。

这些尝试没有留下替代分支、fallback 或未调用 helper。

## 8. Baseline 完整性修正

`continuous_gqa_decode / TileLang` 原 source 是两 kernel split-K partial + combine，generated 是一次 streaming kernel。即使数学输出接近，两者也不满足“算法结构和调用数一致”，因此两台设备的 source 数字均撤掉，runner 不再接这条 baseline。

这会修正上一份覆盖报告的 headline。按当前固定表实际内容计算：

- 表中共有 123 个入口；
- 21 个 `variant_*` 不计入独立算法；
- 102 个独立算法中，50 个至少有一个严格 source；
- 当前严格覆盖率是 **50/102 = 49.02%**。

也就是说，随着语料后来增加且这一条无效对照被撤掉，当前仓库已不再满足“覆盖率超过 50%”。本轮没有用不公平 source 补这个数字；覆盖率缺口应在后续 baseline 接线轮用新的严格对照补回。

## 9. 实际验证范围

本轮没有做全量，只运行直接受修改判定影响的 repro。统一入口仍是：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <kernel>
```

两台设备实际运行的定向范围包括：

- `histogram`：Triton、cuTile、TileLang；
- `ordered_prefix`：Triton、cuTile、TileLang；
- scan 判定消费者：`moe_align_block`、`unique_consecutive`、`nonzero_compact`、`sorted_nucleus_cutoff`，三 provider；
- `batch_norm_training`：三 provider；
- `online_softmax`：TileLang；
- `w4a8_packed`：TileLang；
- `rope_qk_full`、`rope_qk_partial`、`rope_qk_inverse`：三 provider；
- `weight_only_int4`：TileLang，用于确认非连续 compact fallback 没有被误接到新 bulk path。

所有最终保留路径均通过各自 reference 数值对照。临时失败的宽泛实验在撤销后不进入固定表。

## 10. 架构边界自查

- Kernel IR 没有变化；
- 没有按 kernel 名或算法类别增加 realizer/emitter 分支；
- `axis + positive_constant` 是共享、受限且可验证的访问事实，不是 RoPE 名字特化；
- cuTile translated tile 和 TileLang compact fragment 都是目标能力检查与机械语法投影；
- stream tile 候选留在 TileLang tuner，因为它是参数性选择；
- scan worker reuse 判定留在 shared Plan，因为它改变程序 ownership 结构；
- histogram、ordered prefix、BatchNorm、RoPE dtype 的变化属于作者算法与 baseline 对齐，没有让编译器改写 Kernel IR；
- 被替代或失败的路径已删除，没有两套实现并存。

## 11. 最终判断

本轮实质性修复了六类问题：作者算法没有对齐、scan ownership、target-local 参数空间过窄、compact span 没有被 leaf 消费、translated index 在中途丢失、流式统计表达过度物化。它们分别落在 DSL、shared Plan、target tuner 和 leaf 投影的正确边界，没有混成一个按 kernel 分叉的优化器。

但“所有严格 baseline 在 5% 内”尚未完成。余下 23 个固定表原始超标格子中：

- 11 个仍是 Intent 自身必须继续承担的共享物理决定或 target 投影能力缺口；
- 12 个已有源码证据表明生成端使用了目标原生原语，本轮没有发现缺失 Plan 事实，暂归目标 tuner/下层质量边界；
- 另有 5090 BatchNorm 不在固定表超标计数中，但当前同机 A/B 明确仍超标；
- baseline 覆盖率经严格纠正后为 49.02%，需要另一次严格 source 接线才能重新超过 50%。

这三个数字是当前真实状态，不能合并成“性能已经完成”。
