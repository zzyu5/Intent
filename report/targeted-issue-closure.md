# 本轮问题闭环

本轮没有跑全量矩阵。验证严格跟随被修改事实的消费者：先做跨机器 A/B；再只跑私有工作区、扫描工作区、staged ownership 和新 batched contraction 的直接用户；最后只补两张 CSV 中原有的 `not_measured` 空洞。

## 1. H100 上的共享内存 A/B

同一份 `absorbed_mla_prefill` Triton 源码在 RTX 5090 D 上曾因 `102400 B > 101376 B` 失败；在 H100 上直接数值通过：

- 数值最大误差：`0.0001220703125`
- p50 / p95：`0.0556 / 0.0568 ms`

因此这次没有把共享内存容量上移成新的算法结构决策。这个量确实会改变具体设备上的可实现候选，但 A/B 表明下层已经按设备完成合法性筛选；“设备容量已进入能力契约但不参与结构选择”仍成立，但应补充限定：容量参与候选可实现性，而非当前的上游结构选择。

H100 上同一算子的 cuTile 也数值通过，p50 / p95 为 `0.2861 / 0.2878 ms`。TileLang 的所有候选在下层布局推断中出现同一块 `latent_block` 的两种不一致 fragment layout，因此标为 `unsupported`；没有为了 TileLang 把布局决策搬进共享层。

## 2. 学习式稀疏 MLA 首次编译失败

并排比较后，原 DSL 与公开 FlashMLA 的差异不是算法数学不同，而是作者写下的 batched contraction 没有可表达位置：公开实现的核心是 `Q @ focused_kv^T` 和 `S @ focused_kv`，原 DSL 只能写 broadcast multiply + reduce，三个目标因此得到很大的逐元素表达式。

本轮增加了 `I.contract(..., batch=((lhs_axis, rhs_axis),))`：

- `intent.batch` 是 Kernel IR 中的逻辑语义，不在 Physical Plan 再复制一份。
- Kernel facts 保留 batch 轴的来源关系；轴角色分配不会把 batch 轴误认作矩阵自由轴。
- Triton 和 cuTile 的 direct contract 机械投影为三维 `tl.dot` / `ct.mma`；staged 与 deferred 组合尚无机械投影时明确拒绝。
- TileLang 0.1.13 没有当前模型可直接映射的 batched GEMM 叶子，明确报源码位置的不支持，不展开成慢路径。

改后 Triton 源码从逐元素乘加变成三次矩阵原语：两次 batched QK 和一次 batched PV；cuTile 同样变成 `ct.mma`。单一 Triton 候选 `Q_TILE=1/2, K_TILE=32` 已实际运行并对参考：output/max/lse 最大误差分别为 `0.0001220703125 / 6.33e-8 / 9.54e-7`。

但是完整 autotune 在 RTX 5090 D 和 H100 上仍都超过五分钟，cuTile 也触发 `tileiras` 的 10 秒单候选编译超时。公开实现按单 query、head tile、selected-token tile 组织；因此给共享 query tile 候选补入 `1` 和 `2`，这是原来搜索空间缺的合法物理取值，不是按算子特判。即便如此，完整下层首次编译仍不可接受。最终结论是：表达与投影缺口已修，剩余失败类别确实是下层编译成本；CSV 用 `compile_timeout`，没有冒充通过，也没有为它增加共享机制。

## 3. 从 IR / Plan 重建事实的系统审计

本轮删除了三类高置信重复重建：

1. 私有工作区与 scan 工作区的 owner extent，原先都经过 `owner node -> program_order -> roleDimensions` 重新拼接；现在所有目标直接按 Plan 中的 owner node 读取已经解析好的 axis dimension。
2. scan workspace extent 原先三个叶子各自回到 Kernel IR 找 axis 再调用 `dimensionName`；现在直接消费共享 physical binding 已建立的 `axisDimensions[axis_node]`。
3. staged parallel 的 member 身份原先三个叶子都遍历 ragged runtime 的 operation 列表反推；现在 member 直接读取 `StageAxisOp(role="member", axis_node=...)`，叶子只保留 `expert/member_offsets/member_start` 的目标拼写。outer 身份仍由 ragged relation 的 outer node 给出，因为 StageAxis schema 没有重复保存它。

审计中保留了两类看起来“厚”但合法的代码：张量 result shape 是 Kernel IR 语义；`BLOCK_SIZE_M/N/K`、`TILE_SIZE_M/N/K` 是 StageAxis tile role 的目标拼写，不是目标重新选 tile。

## 4. 只补原有表格空洞

RTX 表新增的可运行数字：

- continuous GQA decode：cuTile `1.2711/1.2755 ms`；TileLang `5.5533/5.5612 ms`
- MLA prefill：cuTile `0.0428/0.0446 ms`，上游 `0.0410/0.0416 ms`
- embedding forward：Triton `0.1531/0.1742 ms`，上游 `0.0430/0.0451 ms`
- absorbed MLA：Triton `0.2124/0.2133 ms`；TileLang 明确 `unsupported`
- token-sparse MLA：Triton/cuTile `compile_timeout`；TileLang `unsupported`

H100 表新增的可运行数字：

- MLA query absorb：Triton `0.0090/0.0094 ms`；cuTile `0.0087/0.0092 ms`
- MLA value reconstruct：Triton `0.0104/0.0112 ms`；cuTile `0.0097/0.0101 ms`
- absorbed MLA：Triton `0.0556/0.0568 ms`；cuTile `0.2861/0.2878 ms`
- FP8 MQA logits：Triton `0.0496/0.0501 ms`

H100 的 TileLang head projection、absorbed MLA、FP8 MQA，以及 cuTile FP8 MQA 都被下层能力或布局边界拒绝，已写成 `unsupported`。没有重跑 CSV 中已有数字。

## 5. 实际验证范围

本轮实际执行的手动 repro 只有：

```bash
# H100 A/B 与空洞
examples/run/repro.sh triton absorbed_mla_prefill
examples/run/repro.sh cutile absorbed_mla_prefill
examples/run/repro.sh tilelang absorbed_mla_prefill
examples/run/repro.sh triton mla_head_projection
examples/run/repro.sh cutile mla_head_projection
examples/run/repro.sh tilelang mla_head_projection
examples/run/repro.sh triton token_sparse_mla_prefill
examples/run/repro.sh cutile token_sparse_mla_prefill
examples/run/repro.sh tilelang token_sparse_mla_prefill
examples/run/repro.sh triton fp8_mqa_logits
examples/run/repro.sh cutile fp8_mqa_logits
examples/run/repro.sh tilelang fp8_mqa_logits

# RTX 空洞
examples/run/repro.sh cutile continuous_gqa_decode
examples/run/repro.sh tilelang continuous_gqa_decode
examples/run/repro.sh cutile mla_prefill
examples/run/repro.sh triton embedding_forward_lookup
examples/run/repro.sh triton absorbed_mla_prefill
examples/run/repro.sh tilelang absorbed_mla_prefill
examples/run/repro.sh cutile token_sparse_mla_prefill
examples/run/repro.sh tilelang token_sparse_mla_prefill

# 被共享 workspace / staged 判定直接影响的消费者
examples/run/repro.sh {triton,cutile,tilelang} nonzero_compact
examples/run/repro.sh {triton,cutile,tilelang} viterbi_decode
examples/run/repro.sh {triton,cutile,tilelang} ordered_prefix
```

三后端的 `nonzero_compact`、`viterbi_decode`、`ordered_prefix` 均数值通过。没有跑全量，也没有补跑任何非空 CSV 格子。

## 6. 仍有不确定性的取舍

- 稀疏 MLA 的 direct batched contraction 已能正确生成并由单候选数值验证；完整 autotune 仍无法在可接受时间结束，所以没有性能数字。这里选择明确记录下层 compile-time boundary，而不是继续增加共享结构或缩小真实问题规模。
- query tile 的 `1/2` 候选来自公开实现的单-query 物理组织，也由本机共享内存可实现性验证；它属于原有 `query` 搜索轴的补值。没有用设备型号分支。
- TileLang 的 batched GEMM 和多处 layout conflict 均停在下层模型边界；本轮没有把 TileLang 所需的 layout 决策上移，也没有保留慢的逐元素替代路径。
