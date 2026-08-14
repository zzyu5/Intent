# Intent Kernel 编译器四轮推进报告

## 报告边界

这份报告严格对应四个连续的用户任务 prompt，而不是按日期或提交数量自行分轮。起点是 `e45b84e` 对应的“处理前几轮暴露的问题”，终点是 `7025343` 固定 RTX 5090 D 全量结果。

四轮与提交的对应关系为：

| 轮次 | 用户任务 | 实现/数据提交 | 是否按原要求完整结束 |
|---|---|---|---|
| 第一轮 | 处理前几轮暴露的问题；H100 A/B；下层编译成本；删除 emitter 事实重建；只补表格空洞 | `e45b84e`、`b55f21a` | 完成；按要求没有跑全量 |
| 第二轮 | 修转置、分页注意力、Viterbi、连续去重、NMS 的跨后端高离散 | `ebb1c3f` | 完成；按要求没有跑全量 |
| 第三轮 | 补 attention backward、causal-conv backward、block-sparse attention | `2626597` | 当轮定向验证完成；按要求没有跑全量 |
| 第四轮 | 清空剩余 source 候选，然后两台机器做一次全量 | `8502f5b`、`7025343` | source 与 5090D 完成；H100 因外部占用延期 |

不属于这四轮的更早提交只构成起始背景，不在本报告重复展开。

从 `e45b84e^` 到最后实现提交 `8502f5b`，非 `report/` 代码共修改 41 个文件，约 `+3139/-421`。固定表在四轮开始时有 106 个 `(kernel, case)`；第三、四轮代码新增 7 条记录，最终 RTX 5090 D 表为 113 条，H100 表仍停在此前真实运行过的 106 条。

## 总结论

这四轮不是简单增加七个 kernel。它们依次验证并修改了四个不同层面：

1. **决策归属是否正确。** Shared-memory 容量由下层候选合法性筛选；batched contraction 是作者算法语义；owner extent、scan extent 和 staged member 身份必须从 Plan 读取。
2. **三个目标是否只是投影。** TileLang transpose 修成原生原语；M=1 contraction 没有等价能力时删除慢速伪支持；无收益的 target A/B 全部撤回。
3. **作者能否写真正的反向和多调用。** Attention backward、causal-conv backward 和 block-sparse pipeline 由作者明确拆成多个 kernel，编译器没有决定调用次数。
4. **陌生 source 与全量回归是否诚实。** 2:4 sparse、paged MLA、paged split-K、varlen GQA decode 进入同一主链；5090D 全量发现 10 个状态变化，没有在同一轮顺手修表；H100 没跑就明确延期。

## 第一轮：处理前几轮暴露的问题

### 用户要求

这一轮的原始任务有四部分：

1. 测试节奏优先：禁止跑全量，只跑共享判定的直接消费者。
2. 在 H100 上对 `absorbed_mla_prefill` 做一次能分清责任的 A/B。
3. 查清 `token_sparse_mla_prefill` 是表达不了、目标投影膨胀，还是下层首次编译成本过高。
4. 系统删除 target leaf 从 tensor shape、逻辑维度或 role name 重建已有 IR/Plan 事实的路径，并只补两张表中的原有空洞。

对应实现提交是 `e45b84e fix(compiler): project sparse MLA batch contractions`，报告与表格提交是 `b55f21a docs(report): record targeted cross-device closure`。

### 1. H100 shared-memory A/B

起始现象：同一份 absorbed MLA Triton 目标源码在 RTX 5090 D 上需要 `102400 B` shared memory，但该 kernel 可用 `101376 B`，无法启动。H100 每个 SM 的 shared memory 大得多，因此它能直接区分：

- 如果 H100 自动通过，说明下层已根据设备排除装不下的候选；
- 如果仍失败，说明 shared footprint 是我们提前写死的物理结构决定。

实际结果：H100 Triton 自动通过。

- output 最大误差：`0.0001220703125`；
- p50/p95：`0.0556/0.0568 ms`；
- 同一算法 cuTile：`0.2861/0.2878 ms`；
- TileLang 的候选在下层 fragment layout 推断中冲突，明确标为 `unsupported`。

结论是：shared-memory 容量影响候选是否可实现，但当前下层已经按设备完成合法性筛选。本轮没有把容量上移成新的算法结构决策，也没有增加 H100/5090D 分支。

这修正了之前过于绝对的说法。准确表述应是：设备容量进入能力合同，并参与候选可实现性；当前没有证据要求它进入更上层的算法结构选择。

### 2. Token-sparse MLA 的真实问题

最初三个目标都能生成源码，但 Triton/TileLang 首次 JIT 超过五分钟，cuTile 候选超时或全部无效。这不等于语言表达不了，也不等于目标语义上不支持。

并排阅读公开 FlashMLA 与 generated source 后发现，真正差异是作者需要的 batched contraction 没有在 DSL/Kernel IR 中表达：

```text
公开结构：Q @ focused_KV^T，S @ focused_KV
旧生成：broadcast multiply + elementwise reduce
```

旧写法把一个矩阵原语展开成规模很大的逐元素表达式，导致三个下层编译压力异常。

#### 前端与 Kernel IR

`I.contract` 新增 `batch=`：

```python
I.contract(
    lhs,
    rhs,
    reduce=((lhs_reduction_axis, rhs_reduction_axis),),
    batch=((lhs_batch_axis, rhs_batch_axis),),
    acc_dtype=I.f32,
)
```

前端在 `python/intent/frontend/lowering/intrinsics/structured.py::_contract` 中检查：

- batch 轴索引合法；
- 每侧 batch 轴唯一；
- batch 轴不与 reduction 轴重叠；
- 对应 batch 维度兼容。

验证后的 relation 直接写入 canonical `intent.contract`。Batch 是作者写下的算法语义，不在 Physical Plan 复制第二份，也不由 emitter 从周围 shape 推导。

#### Shared facts 与轴角色

`KernelFacts` 保存 contract 的 batch axis provenance。GPU axis assignment 在选 contraction M/N 自由轴时排除 batch 轴，避免把一个批次维误当成矩阵自由维。

#### 三个目标投影

- Triton direct contract 投影成三维 `tl.dot`；
- cuTile direct contract 投影成三维 `ct.mma`；
- TileLang 0.1.13 没有当前模型能机械映射的 batched GEMM leaf，因此在 emission 阶段按源码位置报 `unsupported`；没有继续展开成慢速逐元素实现。

改动后，单一 Triton 候选 `Q_TILE=1/2, K_TILE=32` 完成数值验证：

- output 最大误差：`0.0001220703125`；
- maximum 最大误差：`6.33e-8`；
- LSE 最大误差：`9.54e-7`。

完整 autotune 在 RTX 5090 D 和 H100 上仍超过五分钟，cuTile 也会触发 `tileiras` 的单候选编译超时。因此最终分类是：

- 算法表达缺口：已修；
- direct matrix primitive 投影：Triton/cuTile 已修；
- 下层首次编译成本：仍存在，CSV 记 `compile_timeout`；
- TileLang batched contraction：明确 target capability 边界。

共享 query tile 搜索轴补入 `1/2`，因为公开算法是 single-query 物理组织。这是既有轴的合法候选补值，不是 token-sparse kernel 特判。

### 3. 删除 emitter 对已有事实的重复重建

这一轮系统审计 target leaf 后删除三类高置信重复：

#### Private/scan workspace owner extent

旧路径：

```text
owner node -> program_order -> roleDimensions -> 重新拼 extent
```

新路径：所有目标直接从 Plan owner node 读取已经解析的 axis dimension。

#### Scan workspace extent

旧路径：三个目标分别回 Kernel IR 找 scan axis，再调用各自的 `dimensionName`。

新路径：直接读取共享 physical binding 的 `axisDimensions[axis_node]`。

#### Staged ragged member 身份

旧路径：三个目标遍历 ragged runtime operation，反推哪个节点是 member。

新路径：直接读取 `StageAxisOp(role="member", axis_node=...)`；叶子只保留 `expert/member_offsets/member_start` 的目标语法。

审计也明确保留了两类合法的厚代码：

- tensor result shape 属于 Kernel IR 语义；
- `BLOCK_SIZE_M/N/K`、`TILE_SIZE_M/N/K` 是已选 StageAxis tile 的目标拼写。

它们没有重新选择物理结构，因此没有为了“叶子看起来薄”而删除。

### 4. 只补原有表格空洞

RTX 5090 D 新补：

- continuous GQA decode：cuTile `1.2711/1.2755 ms`，TileLang `5.5533/5.5612 ms`；
- MLA prefill：cuTile `0.0428/0.0446 ms`，source `0.0410/0.0416 ms`；
- embedding forward：Triton `0.1531/0.1742 ms`，source `0.0430/0.0451 ms`；
- absorbed MLA：Triton `0.2124/0.2133 ms`；
- token-sparse MLA：Triton/cuTile `compile_timeout`，TileLang `unsupported`。

H100 新补：

- MLA query absorb：Triton `0.0090/0.0094 ms`，cuTile `0.0087/0.0092 ms`；
- MLA value reconstruct：Triton `0.0104/0.0112 ms`，cuTile `0.0097/0.0101 ms`；
- absorbed MLA：Triton `0.0556/0.0568 ms`，cuTile `0.2861/0.2878 ms`；
- FP8 MQA logits：Triton `0.0496/0.0501 ms`。

已有非空格没有重跑。

### 5. 实际验证范围

本轮没有运行全量。执行范围为：

- H100：`absorbed_mla_prefill`、`mla_head_projection`、`token_sparse_mla_prefill`、`fp8_mqa_logits` 的相关 provider；
- RTX 5090 D：continuous GQA、MLA prefill、embedding lookup、absorbed MLA、token-sparse MLA 的原空洞；
- 共享 workspace/staged 判定的直接消费者：三后端 `nonzero_compact`、`viterbi_decode`、`ordered_prefix`。

三个直接消费者均数值通过。

### 第一轮交付结论

- `e45b84e`：batched contraction 与三类 emitter 重建删除；
- `b55f21a`：两机表空洞和定向报告；
- 没有全量；
- token-sparse 的语义/投影缺口已修，完整编译成本仍未解决；
- shared-memory 容量没有变成新结构决策。

## 第二轮：修跨后端高离散格

### 用户要求

逐个并排阅读三个目标生成源码，处理五个高离散格：

| 记录 | 当时约最大离散度 |
|---|---:|
| matrix transpose | 6.5x |
| paged attention | 5.9x |
| Viterbi | 4.9x |
| unique consecutive | 3x |
| greedy NMS | 2.1x |

处理规则是：原语选错就改目标拼写；Plan 欠定就补 Plan；目标确实做不到就提前拒绝。只跑被修改格，不做全量。

对应提交：`ebb1c3f fix(tilelang): use native transpose and reject scalar contraction fallback`。

### 1. Matrix transpose：目标叶子原语选错

三家读取同一 Plan：

- Triton：`tl.permute`；
- cuTile：`ct.permute`；
- TileLang 旧实现：`T.Parallel` 下逐元素赋值。

TileLang 0.1.13 已提供 `T.transpose(src, dst)`。因此只修改 TileLang operation handler，不改变 Kernel IR 或 Plan。

结果：

- 数值最大误差：0；
- TileLang p50/p95：`0.6053/0.6083 -> 0.0916/0.0943 ms`；
- Triton `0.0926 ms`、cuTile `0.0943 ms`、TileLang `0.0916 ms` 回到同一水平。

分类：目标叶子原语选择错误，已修。

### 2. Paged attention：删除慢速伪支持

Triton QK/PV 使用 `tl.dot`，cuTile 使用 `ct.mma`。TileLang 旧 leaf 把 M=1 contraction 展开成：

```text
products fragment -> elementwise multiply -> T.reduce_sum
```

这正是约 5.9x 离散的来源。

核验 TileLang 0.1.13 后确认：

- 没有可调用的 GEMV/matvec/dot；
- `T.gemm` 对 M=1 报 `M must be divisible by 16`；
- 将 M 补成 16 后仍与同一程序的 fragment layout 冲突。

因此删除 products+reduce fallback，在 target emission 阶段、进入 JIT 前，以原 `intent.contract` 源码位置明确报不支持。

- 状态由 pass 收紧为 `unsupported`；
- 原 `1.3210 ms` 不再冒充有效支持；
- TileLang 的 M=16/layout 约束没有进入共享 Plan。

分类：目标语言当前没有等价原生单行 contraction 能力。

### 3. Viterbi：A/B 排除错误归因

首先将 cuTile 外部 1x1 scalar load 改成零维 `ct.load(..., shape=())`：

- 数值通过；
- p50 `28.9815 -> 28.9926 ms`；
- 没有改善，改动撤回。

随后尝试让动态索引的一维 private buffer 从 scalar array 改用可寻址 private vector。cuTile 首次编译数分钟仍未完成，这条共享 residency 改动也撤回。

最终结论：三家都忠实发射作者写下的顺序、loop-carried scalar dynamic programming；cuTile 当前没有不改写算法即可替换的 argmax/state 原语。

分类：cuTile 对该作者程序形态的下层质量边界，不补共享机制。

### 4. Continuous unique：raw store 无收益

cuTile 原来使用 `ct.scatter(check_bounds=True)` 写回扫描结果。A/B 改为 `get_raw_memory().store_offset(mask=...)`：

- 数值通过；
- p50 `1.0153 -> 1.0063 ms`；
- 与未改动当轮复测 `1.0053 ms` 等价。

它没有解释三倍离散，因此完整撤回。

分类：cuTile 对“向量 scan 后由顺序 scalar consumer 逐项读取并条件/原子散写”的整体质量边界。

### 5. Greedy NMS：拒绝换算法制造好数字

三家都忠实保留作者写下的顺序 greedy NMS。曾实现共享 32-bit bit-pack workspace 作为 A/B：

- Triton p50 `27.7411 -> 30.8760 ms`，性能退化；
- 上游高性能 bitmask NMS 实际是并行生成块 IoU mask 后再做第二阶段筛选，是另一种作者算法。

因此 bit-pack 完整撤回。没有把 Kernel IR 改写成上游算法，也没有保留退化的 storage 方案。

### 6. 本轮实际验证

只运行：

```bash
examples/run/repro.sh tilelang matrix_transpose
examples/run/repro.sh tilelang paged_attention
examples/run/repro.sh cutile viterbi_decode
examples/run/repro.sh cutile unique_consecutive
examples/run/repro.sh triton greedy_nms
```

Viterbi、unique 和 NMS 命令分别用于多个 A/B。所有无收益候选均在提交前撤回。本轮只更新本机 transpose 数字和 paged attention 状态，没有重跑全量，也没有改 H100 数字。

### 第二轮交付结论

- 提交 `ebb1c3f`；
- 修正一处真实目标原语错误；
- 删除一条数量级慢的 TileLang 伪支持；
- 三类无收益/改算法方向全部撤回；
- 没有增加 Plan 字段或 target-specific realizer。

## 第三轮：补反向与作者多调用

### 用户要求

补当前最大的结构空白：attention backward、causal convolution backward 和 block-sparse attention。必须先读 source，保留上游算法的所有权、重算、多输出与多调用结构；只跑新增项和共享修改的直接消费者，不做全量。

对应提交：`2626597 Add source-driven backward kernel pipelines`。

### 1. Attention backward

作者侧拆成三个 kernel：

| Kernel | 所有权与职责 |
|---|---|
| `attention_backward_delta` | 从 output 与 dO 得到每行 delta |
| `attention_backward_dkdv` | K/V block 拥有外层工作，跨 Q block 重算并累积 dK/dV |
| `attention_backward_dq` | Query block 拥有工作，重算 score/probability 并得到 dQ |

这保留了反向与前向的结构差异：

- 所有权方向反转；
- softmax 中间量重算而非保存；
- dK/dV 与 dQ 由不同拥有者处理；
- 调用次数和中间 tensor 由作者 Python 外层编排。

编译器仍然一次编译一个 kernel function，没有决定整个 backward pipeline 的 launch 数量。

### 2. Causal Conv1D backward

作者侧保留：

1. partial gradient kernel，同时产生输入、权重和 bias 相关中间结果；
2. 第二阶段 reduction kernel，跨程序归并 partial。

它压到多输出、causal offset、跨 program reduction 和作者两次调用。编译器没有将它融合成单 kernel，也没有自行拆分调用。

### 3. Block-sparse attention

此前“它是多调用 pipeline”曾被用作不建立记录的理由，这一轮纠正了该判断。实现按 source 的 block selection 与 partial/combine forward decode 结构写入 DSL，没有为了满足“补反向”而伪造不存在的 block-sparse backward。

### 4. 共享 compiler 修改

#### 多 domain source provenance

旧 facts 假设一个 source 只来自单 domain。反向和 block-sparse 中，一个值会同时携带多个逻辑轴来源。当前 `KernelFacts` 遍历 `source->domains`，pointwise operand 与 shape label 能传播多轴 provenance。

#### Tensor loop-carried state

Sequential loop 现在允许 tensor carrier。Facts 检查 init/carried/yield 的 axis provenance 稳定，防止状态在循环中静默换域。

TileLang leaf 支持 tensor carrier 的 `T.copy` yield，而不是只处理 scalar carry。

#### Unit tensor scalar

Rank-1 unit tensor 可以通过受限 projection 读取为 scalar，用于 delta、maximum 等反向状态；不是从任意 shape=1 值猜 scalar。

#### 重复 contraction operand 的 TileLang 隔离

同一 operand 参与多个 contraction 时，TileLang 下层可能给同一个 fragment 推导出不兼容 layout。当前 leaf 将 Plan 已决定的 tile copy 到 shared buffer，并在 `T.gemm` 前同步。这里没有重新决定 tile，只隔离目标下层的 fragment 约束。

### 5. 当轮验证与后续全量的区别

当轮定向验证回复记录：

- attention backward 三后端真实运行与数值对照通过；
- causal-conv backward 三后端通过；
- block-sparse pipeline 三后端通过；
- 共享判定消费者 grouped GEMM、online softmax、dual GEMM 等通过；
- 没有公平的同算法、同调用次数 upstream adapter，source 数字留空；
- 没有跑全量。

最终第四轮 5090D 全量得到：

| 记录 | Triton p50/p95 | cuTile p50/p95 | TileLang p50/p95 |
|---|---:|---:|---:|
| attention backward | 0.1584/0.1625 | 0.2041/0.2046 | 0.1781/0.1798 |
| causal Conv1D backward | 0.2607/0.2613 | 0.4736/0.4748 | 0.2658/0.2671 |
| block-sparse attention | 0.1031/0.1063 | 0.0580/0.0647 | failed |

因此需要区分两件事：第三轮定向验证时三项三后端通过；第四轮当前代码全量中 TileLang block-sparse 变成 failed。它是后续全量暴露的回归/不稳定项，不能继续按第三轮回复写成当前全通过。

### 第三轮交付结论

- 提交 `2626597`；
- 作者多调用、反向所有权、多输出和重算进入真实 DSL；
- 新增共享能力依赖 axis/SSA/op 语义，没有 kernel-name 分支；
- 当轮没有 blocker、没有全量；
- 后续全量重新暴露 TileLang block-sparse failure。

## 第四轮：清空剩余 source，并做全量

### 用户要求

第一部分：把 source 中“尚未处理但值得独立审视”的候选逐项过完，特别是结构化稀疏；明确区分“确认无需处理”和“还没轮到”。

第二部分：前两轮没有做全量，这一轮在 RTX 5090 D 和 H100 都做一次完整矩阵；记录所有回退，但不要在同一轮顺手修；更新两份独立 CSV 和赢家分布。

实现提交：`8502f5b Add source-driven sparse and paged kernels`。RTX 表与报告提交：`7025343 Record source inventory and 5090D full matrix`。

### 1. Source 清点结果

`source/` 共有 159 个文件，但包含 runtime、test、helper、cost model、support plumbing 和同一算法的多 target 拼写，不能按文件数计算算法数。

逐项语义清点后：

- 尚未审视候选：0；
- 已确认独立但尚未落地：2；
- 其余候选已有等价算法记录，或确认不属于独立单 kernel。

#### 仍是“还没轮到”

| 上游算法 | 最早真实边界 |
|---|---|
| DeepSeek V3.2 radix top-k | raw bit reinterpret；workgroup shared mutable buffer；barrier/thread-workgroup 语义 |
| vLLM fused top-k/top-p | 作者管理全局 workspace；数据依赖 compaction/control；多轮 pivot search |

它们不同于已有 insertion top-k 和独立 nucleus cutoff，不能写成“确认无需处理”。

#### 确认不另建记录

- TileLang persistent MLA：与 paged MLA 数学相同，persistence、grid barrier 和 SM-sized launch 是物理执行方式；
- Liger fused linear cross entropy：Python 外层 chunk/multi-call 编排，inner CE 已有记录；
- `*_runtime.py`、输入压缩 helper、cost model、support/import plumbing；
- 同一算法的不同 provider 拼写。

### 2. 新落成的四条 source-driven 记录

| 记录 | 算法结构 | 编译器处理 |
|---|---|---|
| `varlen_gqa_decode_logits` | 变长 GQA decode、sink/block logits | 复用 ragged relation、ordered stream、多对一 head mapping、unit scalar |
| `paged_mla_decode` | 页表间接 KV、MLA decode | 复用分页 index relation 与 state stream |
| `paged_splitk_attention` | partial output/max/LSE + reducer | 作者明确两次调用，不把 pipeline 放进 compiler |
| `sparse_2to4_gemm` | compressed values + metadata + RHS | 新增 canonical sparse contract 与 target capability |

### 3. 2:4 sparse contract

这是本轮唯一新增的 canonical 算法 op。

#### 前端

`I.sparse_contract_2to4` 检查：

- compressed、metadata、RHS 均为 rank-2；
- compressed/RHS dtype 相同；
- metadata dtype 为 i16；
- sparse format 为 `two_of_four`。

#### Kernel IR 与 facts

生成 `intent.sparse_contract`。共享 facts 验证三个 operand 和一个 result 的 rank/provenance，不从 kernel 名识别 sparse GEMM。

#### Physical Plan

新增 `intent_plan.sparse_contract`，保存已确定的 operand residency、matrix roles 和 accumulator binding。Verifier 要求 format/residency schema 完整。

#### Target projection

- TileLang 分配 compressed `(M_tile,K_tile/2)`、metadata `(M_tile,K_tile/16)`、RHS shared tile，并调用原生 `T.gemm_sp`；
- Triton/cuTile 当前接入的 target model 没有同层级 2:4 sparse primitive，在 emission 前明确 `unsupported`；
- 没有 dense fallback，也没有 sparse kernel-name 分支。

### 4. 分页与 ordered/ragged 组合修复

同一 ragged relation 可能有多个 member domain。此前 ordered role 只传播给直接标记的 member，分页 relation 与 state stream 组合时可能丢 ownership。

当前只要同一 relation 中任一 member 参与 ordered traversal，相关 member domains 都保留该 ordered relation。判断依据是 relation 和 role，不是 paged attention 名字。

### 5. Unit tensor scalar projection

变长 GQA 的 block maximum 等状态以每维 extent=1 的 tensor 存在。Index relation 新增受限 `extract_unit_scalar`：

- index 必须静态全 0；
- source tensor 每个维度 extent 必须为 1。

三个 target 从这一 binding 机械投影；TileLang 使用 `T.reduce_sum(source)` 取得 scalar。它不是看到 shape=1 就任意猜 scalar 来源。

### 6. 公平 source 接线

只在算法、调用次数和 scope 都一致时接 source：

- `sparse_2to4_gemm` TileLang generated `0.2300/0.2318 ms`，source `0.2332/0.2335 ms`；
- xFormers runtime 固定为非 split-K 或单 split，不能对照真实双 kernel `paged_splitk_attention`；
- paged MLA 上游 ABI 与本记录不同；
- varlen GQA 的 TileLang generated 在 runtime 前被单行 contraction capability 拒绝。

后三者 source 列保持空白。此前固定 source 数字全部保留，没有重测或挪用。

### 7. RTX 5090 D 全量

统一入口：

```bash
examples/run/repro.sh <triton|cutile|tilelang> <runner>
```

实际遍历 104 个 runner × 3 个 provider，共 312 次独立执行。多 case runner 展开后形成 113 个 `(kernel, case)`、339 个 provider 单元格。

#### 执行级统计

| 状态 | 数量 |
|---|---:|
| PASS | 287 |
| FAIL | 13 |
| TIMEOUT | 1 |
| UNSUPPORTED | 11 |

#### Case 单元格统计

| Provider | pass | failed | compile_timeout | unsupported |
|---|---:|---:|---:|---:|
| Triton | 109 | 2 | 1 | 1 |
| cuTile | 107 | 3 | 1 | 2 |
| TileLang | 98 | 4 | 0 | 11 |
| 合计 | 314 | 9 | 2 | 14 |

完整日志保存在 `/tmp/intentdsl-full-local.DL6gTj/`，没有进入仓库。

### 8. 新增七条固定表记录

第四轮最终表一次补入第三轮的三条和本轮四条：

| Kernel | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| attention backward | E pass 0.1584/0.1625 | E pass 0.2041/0.2046 | E pass 0.1781/0.1798 |
| block-sparse attention | E pass 0.1031/0.1063 | E pass 0.0580/0.0647 | failed |
| causal Conv1D backward | E pass 0.2607/0.2613 | E pass 0.4736/0.4748 | E pass 0.2658/0.2671 |
| paged MLA decode | K pass 0.1163/0.1183 | K pass 0.2219/0.2232 | unsupported |
| paged split-K attention | E pass 0.1709/0.1718 | E pass 0.2738/0.2754 | unsupported |
| sparse 2:4 GEMM | unsupported | unsupported | K pass 0.2300/0.2318 |
| varlen GQA decode logits | R pass 0.0682/0.0726 | R pass 0.1050/0.1061 | unsupported |

表中数字是 p50/p95 ms。`K` 是 kernel-only，`E` 是取得最终结果的相同端到端 GPU pipeline，`R` 包含算法必须读取的 runtime metadata。

### 9. 全量暴露的 10 个状态变化

用户明确要求本轮只记录回退，不顺手修。因此 CSV 对应同一个实际运行代码基线。

| Kernel | Provider | 旧状态 -> 当前状态 | 结论 |
|---|---|---|---|
| cross entropy | Triton/cuTile/TileLang | pass -> failed | 同一个前端回归：structured intrinsic 只导入 i16，arg-reduce 仍引用 i32，触发 `NameError` |
| sorted nucleus cutoff | Triton/cuTile/TileLang | pass -> failed | 与上项相同 |
| MoE | TileLang | pass -> failed | generated/reference 最大误差 8.7247，上游/reference 正常 |
| Mamba chunk scan | cuTile | pass -> failed | 3 个候选可运行，但最大误差 0.1483154297 |
| continuous GQA decode | TileLang | pass -> unsupported | 第二轮删除单行 contraction 伪支持后的真实能力状态 |
| split-K attention reducer | TileLang | pass -> unsupported | 同一个单行 contraction 能力边界 |

这里合计 10 个 provider 单元格。

`token_sparse_mla_prefill` 维持 Triton/cuTile `compile_timeout`、TileLang `unsupported`，不是新回归。

### 10. 性能与赢家变化

代表性变慢项：

- layer norm backward Triton：`0.0969 -> 0.1388 ms`，+43.2%；
- batched GEMM cuTile：NT +13.3%，TT +25.1%；
- selective scan TileLang：+14.8%；
- W4A8 TileLang：+38.6%；
- reshape-cache split variant Triton：+39.0%；
- ordered prefix Triton：+60.3%。

代表性变快项：

- block-scaled matmul cuTile：`0.1775 -> 0.0703 ms`，-60.4%；
- ROI Align TileLang：-49.3%；
- MoE align TileLang：-43.8%；
- MoE product-domain variant Triton：-41.8%；
- grouped GEMM empty-groups TileLang：-36.8%；
- Mamba scan Triton：-28.3%。

短 kernel 的百分比可能只对应数微秒。本轮没有给所有 p50 波动强行归因，只固定实测数字。

逐 case 最低 generated p50 的赢家分布：

| 赢家 | 上一固定表 | 当前 5090D 表 |
|---|---:|---:|
| Triton | 30 | 37 |
| cuTile | 28 | 27 |
| TileLang | 38 | 36 |
| Triton + cuTile | 2 | 2 |
| Triton + TileLang | 2 | 4 |
| cuTile + TileLang | 1 | 1 |
| 三者并列 | 4 | 3 |
| 无可运行 provider | 1 | 3 |

三家仍各自赢下大量 case，但当前分布含已知回归，只描述该代码基线。

### 11. H100 没有完成最新全量

第四轮原始要求是两台机器都跑全量。实际执行期间 H100 被另一项 vLLM/评测任务持续占满，原任务 turn 被用户新消息中断。随后用户明确要求停止等待，等人工通知机器空闲。

实际处理：

- 停止本地 H100 polling script；
- 没有终止或干扰远端任务；
- 没有创建远端 snapshot；
- 没有部署最新七条记录；
- `kernel-performance-h100.csv` 与本轮前快照逐字节一致。

因此第四轮准确状态是：

| 项目 | 状态 |
|---|---|
| source 候选清点 | 完成 |
| 四条新 source-driven DSL/编译器路径 | 完成 |
| RTX 5090 D 113-case 全量 | 完成 |
| RTX 表更新 | 完成 |
| H100 最新 113-case 全量 | 未执行，延期 |
| 两机最新赢家分布比较 | 未完成 |

H100 表仍是此前 106-case 的真实数据，不能与当前 113-case 5090D 表冒充同一代码基线。

### 第四轮交付结论

- `8502f5b`：source-driven sparse/paged/varlen 实现；
- `7025343`：5090D 全量与固定 CSV；
- source 清点已闭合，但仍有两个明确未落地算法；
- 5090D 全量暴露回归并如实保留；
- H100 部分没有完成，等待人工通知。

## 四轮合起来，编译器具体发生了什么

### 1. Frontend / Kernel IR

新增两项作者算法表达：

- `contract(..., batch=...)`；
- `sparse_contract_2to4(...)` / canonical `intent.sparse_contract`。

Attention backward、causal-conv backward、block sparse、paged MLA、paged split-K 和 varlen GQA 都由既有 op 组合，没有专用 kernel op。

### 2. Shared facts

这四轮补入或收紧：

- contract batch-axis provenance；
- multi-domain value provenance；
- tensor loop-carried state；
- ordered role 跨 ragged member relation 传播；
- sparse contraction facts；
- unit tensor scalar index relation。

### 3. Physical Plan

Plan 新增 sparse contract 物理 binding，并成为 workspace owner、scan extent、staged member、batch/contraction axis 和 target capability 的唯一消费来源。

没有新增 kernel category、算法名 matcher 或第二套 target-specific realizer。

### 4. Target leaf

目标侧的实质变化：

- Triton/cuTile direct batched matrix primitive；
- TileLang native transpose；
- TileLang tensor loop carrier 与 repeated contraction operand isolation；
- TileLang native `T.gemm_sp`；
- 对 batched GEMM、M=1 contraction、2:4 sparse 等真实能力子集提前拒绝；
- 删除 products+reduce、dense sparse fallback 等慢速伪支持。

### 5. 作者多调用边界

编译器仍坚持一份 DSL kernel 对应一次调用。下面这些调用图由作者 Python 外层明确编排：

- attention backward：delta -> dK/dV -> dQ；
- causal-conv backward：partials -> reduce；
- block-sparse attention：partial -> combine；
- paged split-K attention：partial producer -> reducer。

没有把调用次数、跨 kernel 融合或中间 tensor layout 偷偷移进 single-kernel compiler。

## 四轮提交表

| 提交 | 对应轮次 | 内容 |
|---|---|---|
| `e45b84e` | 第一轮 | Batched contraction、token-sparse direct primitive、删除 emitter 重建 |
| `b55f21a` | 第一轮 | 定向 A/B 报告与两机原有空洞 |
| `ebb1c3f` | 第二轮 | TileLang native transpose、删除 scalar contraction fallback |
| `2626597` | 第三轮 | Attention/causal-conv backward 与 block-sparse pipeline |
| `8502f5b` | 第四轮 | 2:4 sparse、paged MLA、paged split-K、varlen GQA decode |
| `7025343` | 第四轮 | 5090D 当前全量、source 清点和固定 CSV |

## 当前未闭合项

### 全量发现的回归

- cross entropy / sorted nucleus cutoff 的 i32 import 前端回归；
- TileLang MoE 数值回归；
- cuTile Mamba chunk scan 数值回归；
- TileLang block-sparse attention 在最终全量中失败。

### Target/downstream 边界

- TileLang batched contraction、M=1 contraction、部分 fragment layout 组合；
- Triton/cuTile 当前 2:4 sparse target model；
- token-sparse MLA 完整首次编译成本。

### Source 中仍未落地的独立算法

- DeepSeek V3.2 radix top-k；
- vLLM fused top-k/top-p。

### 外部阻塞

- H100 最新七条记录与当前 113-case 全量尚未执行；
- 最新两机赢家分布尚不能给结论。

## 当前数据与证据

- RTX 5090 D 当前表：`report/baseline/kernel-performance.csv`；
- H100 既有 106-case 表：`report/baseline/kernel-performance-h100.csv`；
- 第一轮详细定向记录：`report/targeted-issue-closure.md`；
- 第二轮详细离散度记录：`report/cross-backend-performance-closure.md`；
- 5090D 全量临时日志：`/tmp/intentdsl-full-local.DL6gTj/`。

这份报告的边界就是上述四个用户 prompt。更早的跨设备初始化、基础 source-driven 12 case 和最初 MLA/FP8 引入只作为第一轮的前置状态，不计入本报告完成项。
