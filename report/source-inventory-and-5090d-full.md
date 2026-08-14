# Source 清点与 RTX 5090 D 全量复验

## 结论

这一轮把 `source/` 中尚未落地的上游候选逐项过完，并在同一代码基线上完成 RTX 5090 D 的全量复验。结果不是“全部通过”：新增的分页、稀疏与反向记录大多走通，但全量也暴露出 3 组确定回归和若干已经明确的后端能力边界。H100 本轮没有执行；等待进程已停止，远端没有留下 snapshot，H100 固定表保持不变，后续只在人工通知机器可用后再跑。

## 1. Source 清点结果

`source/` 当前共有 159 个文件，但这个数字包含 runtime、测试、helper、cost model 和同一算法的多后端拼写，不能当作算法数。按“是否是一条独立的单次调用算法”清点后，尚未审视的候选为 **0**。

### 本轮落成独立 DSL 记录

| 算法 | DSL 记录 | 上游结构来源 | 处理结果 |
|---|---|---|---|
| 结构化 2:4 稀疏 GEMM | `sparse_2to4_gemm` | TileLang `gemm_sp` | 新增 canonical sparse contract；TileLang 机械投影到 `T.gemm_sp`，Triton/cuTile 在发射前明确拒绝 |
| 变长 GQA decode logits | `varlen_gqa_decode_logits` | cuTile TileGym 变长 decode | 复用 ordered stream、ragged relation 和多对一 head mapping，没有新增按算子入口 |
| 分页 MLA decode | `paged_mla_decode` | TileLang paged MLA | 分页间接映射与 ordered state stream 组合走通；TileLang 被既有单行 contraction 能力边界拒绝 |
| 真正的分页 split-K attention | `paged_splitk_attention` | xFormers split-K | 作者侧两次调用：producer 生成 partial output/max/lse，既有 reducer 完成第二阶段；没有把调用次数放进编译器 |

此外，全量表补入了已经存在但此前没有进入固定 CSV 的 `attention_backward`、`causal_conv1d_backward` 和 `block_sparse_attention` 三条记录。

### 仍是“还没轮到”，不能写成“无需处理”

| 上游算法 | 当前最早边界 | 结论 |
|---|---|---|
| DeepSeek V3.2 radix top-k | raw bit reinterpret；可移植的 workgroup shared mutable buffer、barrier 与 thread/workgroup 执行语义 | 是独立算法，当前没有一对一 DSL 记录，保留为未落地 |
| vLLM fused top-k/top-p | 作者管理的全局 workspace、数据依赖 compaction/control、多轮 pivot search | 不是现有 insertion top-k 与 nucleus cutoff 的简单别名，保留为未落地 |

### 确认不需要另建记录

- TileLang persistent MLA 与现有 paged MLA 数学相同；persistence、grid barrier 和 `T.Kernel(sm_num, ...)` 是物理执行方式，不是另一份算法 IR。
- Liger fused linear cross entropy 是 Python 外层的 chunk/multi-call 编排；单次 inner cross entropy 已有记录。遵守“一份 DSL 源码对应一次调用”，不把调用方编排伪装成单 kernel。
- `*_runtime.py`、sparse input/compression helper、autotune cost model、import/support plumbing，以及同一算法的 provider 特定拼写都不是新算法。

因此，`source/` 的真实剩余是：**0 个未审视候选，2 个已确认独立但尚未落地的算法，其他候选均已有等价算法记录或确认不属于独立单 kernel。**

## 2. 编译器如何承接这些算法

### Canonical Kernel IR

新增 `intent.sparse_contract`，由作者显式陈述 2:4 稀疏收缩的数学角色。它不是把 dense contract 在后端按名字改写成 sparse，也没有把 TileLang 的字段表搬进算法层。

Python 前端新增 `I.sparse_contract_2to4`，直接构造这一个 canonical op。分页与变长 attention 没有新增专用 op，而是复用既有的 ragged relation、ordered state、逻辑读取终点、间接索引和 contraction。

### Physical Plan

新增 `intent_plan.sparse_contract`，只保存机器层必须兑现的选择；共享 facts 和逐轴角色分析识别 contraction 的自由轴、收缩轴与稀疏 operand。另有两处共享闭环：

- ordered role 会沿同一个 ragged relation 的所有 member domain 传播，使分页成员轴与流推进组合时不丢 ownership；
- unit tensor extract 保留为 `extract_unit_scalar` 投影，使 private block maxima 等标量读取直接消费 IR 表达式，不从 shape 反猜来源。

### 三个目标叶子

- TileLang 的能力模型声明支持结构化 2:4，并将 Plan 中的 sparse contract 机械拼写成 `T.gemm_sp`。
- Triton 与 cuTile 当前没有相同层级的稀疏矩阵原语，能力检查在发射前给出 `unsupported`；没有展开成 dense 慢路径冒充支持。
- 分页、GQA、split-K 和 MLA 都继续走共享 Kernel IR 遍历和同一套 physical decision；`lib/` 中没有按 `sparse_2to4`、`paged_mla`、`paged_splitk` 或 `varlen_gqa` 名字分支。

对应代码已提交为 `8502f5b Add source-driven sparse and paged kernels`。

## 3. 上游对照接线

只接了算法相同、调用次数相同且计时范围可对齐的上游内核。

- `sparse_2to4_gemm` 的 TileLang generated 为 `0.2300 / 0.2318 ms`，上游为 `0.2332 / 0.2335 ms`（p50 / p95），数值通过。
- xFormers runtime 固定为非 split-K 或单 split，不能拿来对照真实的双 kernel `paged_splitk_attention`，因此 source 列留空。
- paged MLA 的上游 ABI 与本记录不同，source 列留空。
- varlen GQA 的 TileLang 上游数学接近，但 generated TileLang 在进入 runtime 前已触发单行 contraction 能力边界，不能形成公平数字。

原固定表中已有的所有 source p50/p95 均保持原值；本轮没有重测、覆盖或挪用旧上游数字。

## 4. RTX 5090 D 全量范围

统一入口仍是：

```bash
examples/run/repro.sh <triton|cutile|tilelang> <runner>
```

实际遍历 104 个 runner 名、3 个 provider，共 **312 次独立调用**。多 case runner 被展开后，固定表现在包含 **113 个 kernel/case、339 个 provider 单元格**。

执行级统计：

| 状态 | 数量 |
|---|---:|
| PASS | 287 |
| FAIL | 13 |
| TIMEOUT | 1 |
| UNSUPPORTED | 11 |

展开到 kernel/case 单元格后的统计：

| Provider | pass | failed | compile_timeout | unsupported |
|---|---:|---:|---:|---:|
| Triton | 109 | 2 | 1 | 1 |
| cuTile | 107 | 3 | 1 | 2 |
| TileLang | 98 | 4 | 0 | 11 |
| 合计 | 314 | 9 | 2 | 14 |

runner 统计与 case 单元格统计不同，是因为 `gemm`、batched GEMM、varlen attention、grouped GEMM、FP8 GEMM 和 MLA projection 等一个 runner 会生成多行。

完整日志保存在 `/tmp/intentdsl-full-local.DL6gTj/`；该目录是临时运行证据，没有进入仓库。

## 5. 新增固定表记录

`E` 为相同 scope 的端到端计时，`K` 为 kernel-only，`R` 为读取运行时元数据的 scope。

| Kernel | Triton | cuTile | TileLang | 可比 source |
|---|---:|---:|---:|---:|
| `attention_backward` | E pass 0.1584/0.1625 | E pass 0.2041/0.2046 | E pass 0.1781/0.1798 | 无 |
| `block_sparse_attention` | E pass 0.1031/0.1063 | E pass 0.0580/0.0647 | failed | 无 |
| `causal_conv1d_backward` | E pass 0.2607/0.2613 | E pass 0.4736/0.4748 | E pass 0.2658/0.2671 | 无 |
| `paged_mla_decode` | K pass 0.1163/0.1183 | K pass 0.2219/0.2232 | unsupported | 无 |
| `paged_splitk_attention` | E pass 0.1709/0.1718 | E pass 0.2738/0.2754 | unsupported | 无 |
| `sparse_2to4_gemm` | unsupported | unsupported | K pass 0.2300/0.2318 | TileLang 0.2332/0.2335 |
| `varlen_gqa_decode_logits` | R pass 0.0682/0.0726 | R pass 0.1050/0.1061 | unsupported | 无 |

表内数字均为 p50/p95，单位 ms。

## 6. 相对上一张固定表的状态变化

全量发现 10 个状态变化，没有在同一轮顺手修复，以免把“观测全量回归”和“修改后再测”混成不同代码基线。

| Kernel | Provider | 旧状态 → 新状态 | 根因/边界 |
|---|---|---|---|
| `cross_entropy` | Triton/cuTile/TileLang | pass → failed | 确定回归：structured intrinsic 只导入了 `i16`，`arg_reduce.max` 仍引用 `i32`，前端在生成 IR 前触发 `NameError` |
| `sorted_nucleus_cutoff` | Triton/cuTile/TileLang | pass → failed | 与上项完全相同的 `arg_reduce.max` 前端回归 |
| `moe` | TileLang | pass → failed | generated/reference 最大误差 8.7247；上游/reference 正常，是 generated 数值回归，尚未定位到更窄的共享事实 |
| `mamba_chunk_scan` | cuTile | pass → failed | 下层有 3 个候选可运行，但 generated/reference 最大误差 0.1483154297，是数值回归 |
| `continuous_gqa_decode` | TileLang | pass → unsupported | 单行 contraction 不能投影为 TileLang 矩阵原语；删掉了串行慢路径，属于明确能力边界 |
| `splitk_attention_reduce` | TileLang | pass → unsupported | 同一个单行 contraction 能力边界 |

`token_sparse_mla_prefill` 维持 Triton/cuTile `compile_timeout`、TileLang `unsupported`；它不是本轮新回归。

## 7. 性能变化与赢家分布

本轮没有把每次 p50 抖动都解释成编译器因果。尤其是低于 0.01 ms 的短核，较大的百分比可能只对应数微秒。下面列出绝对或相对上值得继续保留证据的变化；所有当前数字均已写入固定 CSV，完整旧值可从提交 diff 直接看到。

明显变慢：

- `layer_norm_backward` Triton：0.0969 → 0.1388 ms（+43.2%）
- batched GEMM cuTile：NT 0.0917 → 0.1039 ms（+13.3%），TT 0.0930 → 0.1163 ms（+25.1%）
- `selective_scan` TileLang：0.1032 → 0.1185 ms（+14.8%）
- `w4a8_packed` TileLang：0.1163 → 0.1612 ms（+38.6%）
- `variant_reshape_cache_split` Triton：0.0287 → 0.0399 ms（+39.0%）
- `mla_head_projection/query_absorb` cuTile：0.0098 → 0.0114 ms（+16.3%）
- `ordered_prefix` Triton：0.0388 → 0.0622 ms（+60.3%）

明显变快：

- `block_scaled_matmul` cuTile：0.1775 → 0.0703 ms（-60.4%）
- `roi_align` TileLang：0.6121 → 0.3103 ms（-49.3%）
- `moe_align_block` TileLang：0.0822 → 0.0462 ms（-43.8%）
- `variant_moe_product_domain` Triton：0.0373 → 0.0217 ms（-41.8%）
- `grouped_gemm/empty_groups` TileLang：0.0755 → 0.0477 ms（-36.8%）
- `mamba_chunk_scan` Triton：0.0307 → 0.0220 ms（-28.3%）
- `variant_rope_index`：Triton -22.8%，cuTile -26.9%

逐 kernel 取最低 generated p50 后的赢家分布：

| 赢家 | 上一张固定表 | 当前 5090D 表 |
|---|---:|---:|
| Triton | 30 | 37 |
| cuTile | 28 | 27 |
| TileLang | 38 | 36 |
| Triton + cuTile 并列 | 2 | 2 |
| Triton + TileLang 并列 | 2 | 4 |
| cuTile + TileLang 并列 | 1 | 1 |
| 三者并列 | 4 | 3 |
| 无可运行 provider | 1 | 3 |

三家仍分别赢下大量记录，provider 赢家不是单一后端固定垄断；但由于本轮含明确回归，这组分布只能描述当前代码基线，不能单独当作架构优越性的结论。

## 8. H100 本轮状态

H100 被另一项 vLLM/评测任务持续占满，因此本轮没有创建隔离代码快照、没有构建、没有执行任何 H100 repro，也没有触碰对方进程。后台等待脚本已经停止。

`report/baseline/kernel-performance-h100.csv` 与本轮前快照一致，故没有用 5090D 数字填空、没有从旧日志拼接新行，也没有产生一张伪装成“全量完成”的 H100 表。后续 H100 执行以人工通知机器空闲为唯一触发条件。

## 9. 当前诚实边界

- source 清点已经闭合，但 radix top-k 和 fused top-k/top-p 仍是明确未落地算法，不能算完成。
- `cross_entropy`/`sorted_nucleus_cutoff` 的 6 个失败已定位到一个确定的前端 import 回归；本报告记录的是全量时的真实状态，没有在测后悄悄修改基线。
- TileLang 的单行 contraction、联合二维覆盖范围、若干 layout conflict，以及 Triton/cuTile 对结构化 2:4 的缺失都以 `unsupported` 表示；没有保留数量级更慢的伪支持。
- `token_sparse_mla_prefill` 能生成目标源码并以单候选做过数值核验，但完整 autotune 的首次编译成本不可接受，所以仍是 `compile_timeout`，不是 pass。
- 本轮没有 H100 全量，因此不能回答两台机器的新增记录赢家分布与跨机器回归；该项明确延期，不作推断。
