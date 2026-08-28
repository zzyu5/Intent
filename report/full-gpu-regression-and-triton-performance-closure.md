# 第五轮完整推进报告：全量 GPU 回归与 Triton 性能闭合

## 0. 结论先行

这一轮做了大量真实工作，但没有完成第五轮原定目标。

从第四轮完成提交 `669410b` 到当前提交 `bab7a40`，这段推进包含 26 个提交，改动 88 个文件，净变化约为 `+15589/-1474`。六张 `baselinev2` CSV 都产生过一次完整结果；Triton、cuTile 和 TileLang 的新执行链也确实分别暴露并修掉了一批 frontend、shared GPU Program、provider legalization、serialization、JIT 与性能问题。

但是，工作量大不等于第五轮闭合：

- 六张表不是最终代码状态下并行、同步重跑得到的一组最终表；
- 没有证据表明 5090 与 H100 的全量任务按要求并行启动并完成；
- 当前 Triton 103 个有数值与性能结果的通过行中，只有 82 个低于 1.05 倍，仍有 21 个超标；
- cuTile 当前仅 27/74 行通过，28 行 timeout，另有 shared/provider verification 与 JIT 缺口；
- TileLang 当前 CSV 只有 9/74 行通过，其中 H100 表还是后续修复前的旧快照，不能代表当前实现的真实能力；
- Prompt 5 要求的 candidate set、winner、双机公平比较证据没有逐 entry 落全；
- 本轮最终宏观架构自查直到这份报告才真正进行，此前的修复仍有“局部可运行，但抽象是否成熟尚未证明”的问题。

按 Prompt 5 的实质目标加权评估，本轮完成度是 **44%**，合理区间约为 **39%—50%**。这不是按条目数打钩，而是按各目标对最终结论的承重程度估算。现在补出这份报告不会增加代码和验证证据，因此不把完成率人为抬高。

最重要的架构结论也必须先写明：

> “没有按 kernel 名称分支”只是最低要求，不是成熟编译器的充分条件。把名字换成 role、source ID、整数编号或某个局部结构 matcher，仍可能只是把特例换了一种编码。成熟性要由同类成熟编译器怎样保存语义、表达访问区域、建立依赖、选择合法 provider form，以及我们的实现与它们具体差在哪里来判断。

用户指出的 `operator_kind == 12/11` 正是一个真实例子。该段代码不是 kernel-name 特判；它所做的“识别一个简单 native reducer”本身也有合理性。但是它依赖没有 typed authority 的裸整数，并且不同 pass/provider 各自解释这些编号。它目前是“合理的局部 matcher 建在不成熟的语义编码上”，不能因为没有格式名或 kernel 名就判定为正确架构。

---

## 1. 报告范围与证据边界

### 1.1 本报告覆盖什么

本报告覆盖从 Prompt 5 开始执行，到用户要求暂停并完整复盘为止的全部推进。实现边界取：

```text
第四轮完成：669410b  Reconstruct cuTile and TileLang provider lowering
第五轮当前：bab7a40  Close TileLang sparse and grouped execution paths
```

Prompt 5 本身来自：

```text
report/prompt/05-intentdsl-full-validation-triton-closure-prompt.md
```

原定任务不是继续搭后端，而是：

1. 在重构后的唯一新链上做两机、三 provider 全量回归；
2. 保证六张表的算法、输入、计时 scope 与 tuner candidate set 公平；
3. 修复所有重构回归；
4. 把所有严格可比的 Triton 行压到 source 的 1.05 倍以内；
5. 说明 cuTile/TileLang 的失败究竟是 shared、provider、工具链还是资源边界；
6. 最终在同一代码状态上并行重跑六表并形成闭合报告。

### 1.2 本报告没有把什么伪装成证据

- CSV 中已有的数字是各提交时点的运行结果，不自动等于当前 `HEAD` 的结果；
- 某个 timeout 被一个候选跑通，只证明该路径存在可运行候选，不证明整个 provider 已闭合，也不证明性能达标；
- 一次定向通过不等于全量无回归；
- 当前 H100 TileLang 表没有在后续修复后重跑，因此不能用“0 pass”断言当前代码在 H100 上真的一项都不能运行；
- 代码中出现 pass、verifier、role 或 typed-looking attribute，不自动证明它承担了正确的编译器职责；
- 旧 baseline 的通过数与当前表不能直接当作同一执行链的普通性能回归，因为中间发生了后端整体替换。

---

## 2. 这轮为什么持续很久：实际任务与原 Prompt 发生了偏移

第五轮开始后，第一次全量没有只暴露少量回归，而是证明第三、四轮声称已经横向闭合的 GPU Program 与 provider lowering 实际仍有大面积未完成部分。于是第五轮从“验证与 Triton 性能收尾”变成了三件事同时进行：

1. 补完 shared GPU Program 的 executable lowering；
2. 补完 cuTile/TileLang provider form 与 bufferization；
3. 同时接着做全量、性能和 measurement 公平性。

这就是尾大不掉的根因。`669410b..bab7a40` 的变化达到 88 个文件、约 1.56 万行新增；其中 shared GPU transforms、KIR→GPU construction 与 TileLang Bufferize 占据了主要体量。一个真正处于“最终全量”阶段的轮次，不应该还新增两千行级别的 pointwise/reduction/region-fold realization，也不应该在 provider bufferization 中继续补千行级别的执行结构。

因此，这轮不是“测试跑得慢所以耗时长”这么简单。更准确的描述是：

> Prompt 5 的全量第一次把 Prompt 3/4 尚未横向完成的事实暴露出来；随后大量时间被用于补前两轮的后端骨架，而不是完成 Prompt 5 自己的最终验收。

用户在推进中连续提出的几次纠偏是正确的：

- 看到长时间没有 CSV 更新和提交，要求先形成可见 checkpoint；
- 要求不要继续无限追 Triton 小差距，先收住 row traversal 的回归；
- 要求处理 cuTile/TileLang 新出现的 timeout，而不是把它们长期挂着；
- 明确指出 grouped MoE 因 row traversal 修改从接近 source 退到约 1.59 倍，不能以“正在收尾”为由跳过；
- 最后要求暂停继续扩张，先做宏观调查和完整报告。

这些纠偏最终促成了多个较小、可追踪的提交，也确实修回 grouped GEMM 与 MoE；但它们发生得较晚，不能改变第五轮整体已经偏成“边测边继续造后端”的事实。

---

## 3. 完整提交时间线

下面按实际提交顺序记录这一轮发生了什么。提交信息只能说明改动意图；是否真正闭合，要以后文的 CSV 和架构审计为准。

| 提交 | 作用 | 本轮中的真实意义 |
|---|---|---|
| `f8622d0` | Checkpoint reconstructed GPU lowering before full validation | 在全量前先保存新链状态，说明此时后端仍在重构而非最终稳定态 |
| `4463693` | Record reconstructed Triton full validation | 产生第一份新链 Triton 全量结果 |
| `765911d` | Record reconstructed cuTile full validation | 产生第一份新链 cuTile 全量结果，暴露 14+14 timeout 与 shared/provider 缺口 |
| `5f05bf6` | Record reconstructed TileLang H100 validation | 产生 H100 TileLang 快照，0/37 pass |
| `b14ded3` | Record reconstructed TileLang 5090 validation | 产生 5090 TileLang 快照，初始仅 3/37 pass |
| `e743d20` | Preserve physical scalar and branch schemas | 修复 scalar/branch schema 在 physical lowering 中丢失 |
| `2dc0a07` | Rank lift Cartesian pointwise worksets | 把 pointwise workset 从局部点扩成 Cartesian physical workset |
| `eab336d` | Record restored AdamW validation | AdamW 恢复到 5090 `1.000254×`、H100 `1.003470×` |
| `8eca8dd` | Realize program-local full-coverage fragments | 补 program-local full-coverage fragment realization |
| `2d21078` | Record program-local fragment validation | 记录该修复对 mhc 等 entry 的定向结果，但 cuTile H100 `mhc_apply_residual` 仍为 `1.591466×` |
| `a844502` | Refresh RoPE candidate contract | 对齐 RoPE candidate contract；5090 `0.541513×`、H100 `0.303465×` |
| `76ac0f7` | Refresh varlen convolution candidate contract | varlen convolution 5090 `0.976560×`、H100 `0.944121×` |
| `a3d10e5` | Distribute physical histogram traversal | 改 histogram physical traversal；5090 `0.095600×`、H100 `0.088983×` |
| `3a21c99` | Align QKV Triton candidate sets | 对齐 generated/source 的 QKV candidate set；5090 `0.943132×`、H100 `0.923747×` |
| `fec8bed` | Realize persistent grouped traversal | 加入 persistent grouped traversal；grouped GEMM 当时 5090 `1.169679×`、H100 `1.029627×` |
| `c2616ed` | Classify Triton source resource gaps | 把 source 自身的资源/兼容问题与 Intent 编译器失败分开 |
| `51f8a2f` | Preserve independent physical axis provenance | 修复独立 physical axis provenance 丢失 |
| `db86b6a` | Refresh H100 scan validation | 更新 H100 scan 定向验证 |
| `aad09e6` | Refresh causal convolution update validation | 更新 5090 causal conv 结果 |
| `7b1c950` | Refresh H100 causal convolution update validation | 更新 H100 causal conv 结果 |
| `d3f0916` | Preserve mutable load order in blocked pointwise programs | 修复 mutable load 的次序；但 causal conv 留下约 `1.076×/1.073×` 性能差距 |
| `9fd39d3` | Measure grouped GEMM kernel scope fairly | 修正 grouped GEMM source/generated 计时 scope，使比较对象一致 |
| `5d147a1` | Preserve runtime row traversal candidates | 把 runtime row traversal 纳入候选；grouped GEMM 5090 降到 `1.027027×`，但 MoE 回退到 `1.585500×` |
| `64a3b66` | Separate indirect and persistent row traversal | 区分 indirect 与 persistent row traversal；grouped GEMM `1.029804×`，MoE 恢复到 `1.007162×` |
| `d580ba8` | Close native GEMM candidate timeouts | 关闭 dense/native GEMM timeout：cuTile dense `0.897887×`，TileLang dense `0.998681×`，TileLang FP8 `1.004245×` |
| `bab7a40` | Close TileLang sparse and grouped execution paths | 5090 TileLang block sparse、grouped forward/backward 从失败/timeout 变为 pass，但比 source 分别为 `1.218700×`、`1.174771×`、`1.579348×` |

这条时间线说明两件事：

1. 这轮不是什么也没做；相反，它做了远超“全量收尾”应有范围的实现工作。
2. 正因为仍在持续增加 shared/provider executable structure，所以不能把六张表或单项定向结果包装成最终成熟状态。

---

## 4. 这一轮实际完成的工作

## 4.1 第一次把新链完整跑进六张表

本轮确实首次让重构后的：

```text
DSL → canonical KIR → shared GPU Program → provider legalization
    → terminal source → provider compile/JIT → GPU numerical run
```

分别进入 Triton、cuTile、TileLang 在 5090 与 H100 上的全量 runner，并将阶段性结果写入六张 CSV。这个动作非常重要，因为它把“IR 看起来完整”变成了真实阶段失败分布。

但六张表的含义是六次阶段快照，不是同一最终提交上的六表全量：

- Triton 表在后续多个性能提交中被局部更新；
- cuTile 表主要保留第一次新链全量结果，只有少数定向修复项更新；
- TileLang 5090 表有后续修复，H100 表没有在 `d580ba8`、`bab7a40` 后重跑；
- 没有运行记录证明两台机器并行完成最终全量。

## 4.2 修正了 measurement 与比较 scope 的真实问题

`examples/repro/v2/measurement.py` 现在对 generated/source 都做 warmup，并以正反执行顺序测量后取中值，减少固定先后顺序对结果的系统性偏置。数值失败时不会再记录性能数字，避免把错误 kernel 的时间混进性能结论。

grouped GEMM 还修正了 source/generated 的计时 scope。之前一边量 wrapper 或 orchestration，另一边量单 kernel，数字即使精确也没有比较意义；`9fd39d3` 关闭的是这个接线问题，不是编译器性能优化。

QKV、RoPE、varlen convolution 等 entry 也补了 candidate contract 对齐。这些工作让若干数字从“不可公平比较”变成“可以比较”。

仍未完成的是：

- 没有为每个 strict entry 保存 candidate set 与双方 winner；
- 没有证据确认所有 source 都在同一候选集合上实际 autotune；
- 某些 provider 的候选空间本身导致首次 JIT 超时，当前 CSV 只给出 `worker_timeout`，没有持久化 candidate 数、单 candidate 成本或最后进度。

因此，measurement 基础比第五轮开始时更可信，但没有达到 Prompt 5 所要求的逐格公平证据完整度。

## 4.3 shared GPU Program 补了多项此前没有横向成立的路径

第五轮实际补进 shared 层的主要内容包括：

- scalar 与 branch schema 的保持；
- Cartesian pointwise workset 与 rank lift；
- program-local full-coverage fragment；
- independent physical axis provenance；
- histogram 的 distributed physical traversal；
- reduction、region fold 与 contraction blocking 的大面积实现；
- grouped traversal 与 row traversal 的 physical mapping；
- subregion、range、validity 与 structured access 的更多传播。

这些改动确实让 AdamW、histogram、varlen convolution、grouped GEMM、MoE 等真实 entry 从失败或退化状态恢复。

但是从变更体量看，这不是少量回归修复，而是补齐 Prompt 3 本应横向铺开的 executable GPU Program。尤其：

- `RealizePointwiseBlocking.cpp` 新增约 2461 行；
- `RealizeReductionBlocking.cpp` 增长约 2040 行；
- `RealizeRegionFold.cpp` 新增约 1824 行；
- `RealizeContractionBlocking.cpp` 增长约 1561 行；
- `KIRToGPU.cpp` 增长约 2196 行。

这说明 Prompt 5 开始时，“shared GPU Program 已经完整，剩下只是全量和性能”的前提并不成立。

## 4.4 Triton 修掉了一批具体回归和不公平项

有明确数值闭合的代表项：

- AdamW：5090 `1.000254×`，H100 `1.003470×`；
- RoPE：5090 `0.541513×`，H100 `0.303465×`；
- varlen convolution：5090 `0.976560×`，H100 `0.944121×`；
- histogram：5090 `0.095600×`，H100 `0.088983×`；
- QKV projection：5090 `0.943132×`，H100 `0.923747×`；
- grouped GEMM：当前 5090 `1.027504×`，H100 `1.029627×`；
- MoE expert projection：当前 5090 `1.008372×`，H100 `0.978705×`。

其中 grouped GEMM/MoE 的过程最能说明本轮既做了真实编译器工作，也一度跑偏：

1. persistent grouped traversal 先改善了结构，但 5090 grouped GEMM 仍为约 `1.17×`；
2. runtime row candidate 加入后，grouped GEMM 到 `1.027×`，MoE 却退到 `1.5855×`；
3. 把 indirect row 与 persistent row 区分后，两者同时回到约 1.03× 和 1.01×。

这次修复不是 kernel-name 分支，而是根据 row access 是否 indirect、是否存在 persistent traversal 的 physical facts 区分候选。它解决了当时的两项回归。

但后面的成熟性审计会说明：这仍不能自动证明 row traversal policy 已成为成熟、完整的 pass。当前规则仍有单一 full-program segment、unit-step、固定候选集合等覆盖门槛；它只是比原来更正确，不是已经达到 ref 中依赖分析和 access-region 驱动的成熟程度。

## 4.5 cuTile/TileLang 关闭了少数高成本阻塞项

`d580ba8` 证明 dense/native GEMM 的 timeout 并非不可避免的 provider 硬边界。收缩到合法且有意义的 candidate 后：

- cuTile dense GEMM 5090 为 `0.897887×`；
- TileLang dense GEMM 5090 为 `0.998681×`；
- TileLang FP8 GEMM 5090 为 `1.004245×`；
- Triton dense GEMM 5090 为 `0.987929×`。

`bab7a40` 又让 5090 TileLang 的 block-sparse 与 grouped contraction 路径能够运行：

- `block_sparse_gqa_decode`：`1.218700×`；
- `grouped_gemm`：`1.174771×`；
- `grouped_gemm_backward`：`1.579348×`。

这些结果证明相关 provider path 不再完全缺失，但不能称为性能闭合。尤其 grouped backward 仍慢 1.58 倍。

cuTile 的 14+14 个 timeout、TileLang 的大量 H100 JIT/verification 失败并没有在本轮解决。用户要求“先适量放大时间、并行测试、判断互相扰动”，当前实现只对少数 native GEMM / TileLang sparse-grouped 路径做了定向收缩与复测，没有形成对全部 timeout 的逐项根因证据。

---

## 5. 当前六张表的真实状态

下表是当前仓库 CSV 的直接统计。它描述文件中记录的状态，不保证每一行都对应当前 `HEAD` 的重新运行。

| 表 | 总行数 | pass | timeout | shared physical 失败 | provider 失败/JIT | 其他 gap | pass 中 ratio ≥1.05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Triton 5090 | 54 | 51 | 0 | 0 | 0 | 3 个 source gap | 8 |
| Triton H100 | 54 | 52 | 1 | 0 | 0 | 1 个 source gap | 13 |
| cuTile 5090 | 37 | 14 | 14 | 6 | 2 | 1 implementation gap | 2 |
| cuTile H100 | 37 | 13 | 14 | 6 | 3 | 1 implementation gap | 3 |
| TileLang 5090 | 37 | 9 | 0 | 11 | 12 | 5 其他失败/gap | 4 |
| TileLang H100 | 37 | 0 | 3 | 12 | 17 | 5 其他失败/gap | 0 |

六表合计：

```text
222 行
148 行 pass
总 pass 率 66.7%
```

这个总 pass 率不能用来描述当前代码的最终能力，因为 TileLang H100 未在后续修复后刷新，也因为六表没有在同一最终状态重跑。但它足以说明 Prompt 5 尚未闭合。

### 5.1 当前 Triton 超过 1.05 倍的全部通过行

5090：

| entry | ratio |
|---|---:|
| `flash_attention_forward` | 1.202316 |
| `causal_conv1d_update` | 1.075916 |
| `mamba3_siso_step` | 1.497076 |
| `moe_splitk_expert_projection` | 2.679961 |
| `mamba_chunk_state` | 1.143991 |
| `paged_mla_decode` | 2.173265 |
| `block_sparse_gqa_decode` | 1.061323 |
| `flaggems_addcmul` | 1.167258 |

H100：

| entry | ratio |
|---|---:|
| `flash_attention_forward` | 2.270342 |
| `rotary_embedding` | 1.149789 |
| `fused_add_rms_norm` | 1.073317 |
| `causal_conv1d_update` | 1.073421 |
| `modern_flash_attention_forward` | 1.388260 |
| `mamba3_siso_step` | 1.159447 |
| `moe_splitk_expert_projection` | 1.692905 |
| `mamba_chunk_state` | 1.189305 |
| `mamba3_siso_forward` | 1.299728 |
| `paged_gqa_decode` | 1.138319 |
| `paged_mla_decode` | 1.829327 |
| `block_sparse_gqa_decode` | 1.378707 |
| `flaggems_batch_norm_training` | 1.265398 |

因此不能说“只剩两机 winner 反转或几微秒差异”。至少 FlashAttention、Mamba3、split-K MoE、paged MLA 与 BatchNorm 仍是明显差距。它们是否都具备严格公平资格还要逐项核对算法分解、candidate set 与 timing scope；在核对完成前，既不能一律算编译器失败，也不能一律排除。

### 5.2 当前 cuTile 的主要状态

两机各有 14 个 `worker_timeout`。当前 5090 的 14 个通过行中有：

- `batched_gemm`：`1.111247×`；
- `mhc_sinkhorn`：`2.889757×`。

H100 的 13 个通过行中有：

- `chunked_softmax`：`1.170858×`；
- `mhc_apply_residual`：`1.591466×`；
- `mhc_sinkhorn`：`3.340369×`。

所以 cuTile 不是“只是 timeout”。即使已经通过的部分，也仍有明显 provider/shared 质量问题。

### 5.3 当前 TileLang 的主要状态

5090 的 9 个通过行里还有四项明显超标：

- `block_sparse_gqa_decode`：`1.218700×`；
- `grouped_gemm`：`1.174771×`；
- `per_token_fp8`：`1.287420×`；
- `grouped_gemm_backward`：`1.579348×`。

H100 当前表没有 pass，但这张表早于后续 5090 provider fixes，不能把 0/37 当成当前实现的最终 H100 能力。正确结论是：**当前缺一轮 H100 定向刷新和最终全量，因此真实状态未知；不是已证明全部失败，也不是可以假设已随 5090 修复。**

---

## 6. Prompt 5 完成度

完成度不按 prompt 的段落数量平均，而按目标对“第五轮可以结束”这个结论的重要性加权：

| 目标 | 权重 | 当前得分 | 依据 |
|---|---:|---:|---|
| 公平资格、measurement 与 scope | 20 | 12 | warmup/order/scope 有实质修复；但 candidate/winner 未逐项保存，算法分解资格未全部重审 |
| 两机并行、六张最终同步表 | 20 | 10 | 六张表都产生过；无并行证据，也不是最终 HEAD 同步重跑 |
| 数值正确性与重构回归关闭 | 20 | 10 | 修掉 AdamW、schema、pointwise、fragment、row traversal 等；cuTile/TileLang 仍有大量 shared/provider verification 失败 |
| Triton 全部严格项 <1.05× | 30 | 8 | 103 个 pass 性能行中 82 个达标、21 个超标；缺逐项公平资格和最终闭合 |
| 成熟编译器对照自查 | 5 | 3 | 已完成本报告中的宏观对照；此前实现过程缺少这一层约束，仍暴露出 magic number、依赖模型和 provider Bufferize 越界 |
| 最终报告、同状态证据与干净交付 | 5 | 1 | 现在有报告，但没有最终六表同状态证据；报告本身不替代验证 |
| **合计** | **100** | **44** | **实质完成度 44%** |

为什么不是按“26 个提交、上万行代码”给更高比例：Prompt 5 的目标是验证、性能闭合和最终状态，不是代码体量。第五轮补了很多前轮遗漏，这些工作有价值，但不能自动折算成 Prompt 5 验收完成。

为什么给区间 39%—50%：有些 Triton 超标项可能在严格重审后因 source 算法分解、资源或 scope 不同而不具备 1.05× 资格；反过来，部分目前看似通过的表行又可能因 candidate set 不一致而失去公平资格。在逐 entry 证据完整前，44% 是最诚实的中值，不是精确测量。

---

## 7. 宏观自查：对照成熟编译器，而不是检查自己有没有违反几个名字规则

本节不按用户给的条目打钩。判断来自当前 Intent 实现与 `ref/triton`、`ref/tilelang` 中同类问题的直接对照。

## 7.1 `operator_kind`：用户的担心成立

当前 cuTile reduction legalization 中：

```cpp
auto result = dyn_cast<gpu::FragmentType>(binary.getResult().getType());
if (result && result.getElementType().isInteger(1)) {
  if (binary.getOperatorKind() == 12)
    return 1;
  if (binary.getOperatorKind() == 11)
    return 2;
}
```

完整函数 `nativeCombineKind` 先要求：

- combine region 只有一个 block；
- 两个 block argument；
- body 只有一个 `gpu::BinaryOp` 和一个直接 yield；
- binary 的两个操作数正好是两个 accumulator，允许交换顺序。

这部分结构匹配是合理的。成熟编译器也会识别“这个 generic reduction 正好等于 provider native sum/max/min”，然后使用原生 collective，而不是永远发射通用闭包。

问题在语义编码：

- `gpu::BinaryOp` 的 `operator_kind` 在 `GPUOps.td` 中只是 `I64Attr`；
- `IntentOps.cpp` 用 `0..17` 的范围与若干整数分支解释类型合法性；
- cuTile、TileLang、Triton 以及 shared transforms 分别重复解释同一编号；
- `7/9` 和 `8/10` 在 cuTile native mapping 中被分别合并为 max/min，而 `9/10` 在 canonical verifier 中明确是 NaN-selecting 变体；这要求 provider 原语的 NaN 语义完全相同，否则是静默语义收窄；
- `RealizeContractionBlocking.cpp` 的 tail predicate 分析同时接受 kind `11` 和 `13` 作为 conjunction，但代码没有在这里显式证明 `13` 的值一定是 `i1`。即便 `i1` 上 bitwise-and 与 logical-and 等价，这个等价条件也没有成为 typed legality 的一部分。

对照成熟实现：

- Triton 的 `CombineBroadcastMulReducePattern` 匹配 `arith::AddFOp`、`arith::MulFOp`、typed `ReduceOp` 与明确 axis/rank，不靠一个全局裸整数表；
- TileLang 的 reduction 用 `ReduceTypeEnum` 和 `IsSum/IsMax/IsBitAnd/...` 这样的 typed category，并在 op 层保存 reduction 类型。

因此明确答案是：

> 这段代码不是“按 kernel 名称特化”，但确实有“去掉名字后改用编号”的架构问题。matcher 的职责可以保留，裸整数不能继续作为跨 canonical/shared/provider 的语义权威。

正确的成熟形态不是简单把 `11/12` 换成字符串，而是：

- canonical/shared arithmetic 使用 typed op class 或至少 typed enum attr；
- logical、bitwise、ordinary max/min、NaN-selecting max/min 保持不同语义；
- provider native combine matcher 匹配这些 typed semantics；
- provider enum 数值只在 terminal API translation 时出现；
- 若 provider 原语不能保持相同 NaN/类型语义，就不把该 combine 映射成 native kind。

裸整数可以作为序列化实现细节，但不能继续被多个语义 pass 当作唯一真理。

## 7.2 row traversal：修复了真实问题，但当前仍是窄 coverage rule

`RealizeContractionBlocking.cpp` 当前通过 source ID、coordinate role、indirect row、runtime row traversal 等 typed physical facts 建立 grouped/persistent mapping。它不按 `grouped_gemm` 或 `moe` 名称分支，这一点比旧式特例好。

但它仍要求：

- 只有一个 full-program execution segment；
- segment offset 必须为零；
- 当前 program space 必须与 segment length 完全相同；
- BLOCK_M/N/K 来自固定候选集合；
- ROW_WORKERS 固定为 `{1,2,4,8}`；
- 多种 access/shape 只在当前 recognizer 能恢复的形式下进入决策空间。

这不是“错”，但它说明当前 pass 的泛化范围是由 recognizer 形状门控制的。换一种 execution segmentation、同一 contraction 中出现多段 mapping，或 range 不是当前 full-program 形式，就直接 `unhandled`。所以“grouped 和 MoE 都过了”只能证明两个样本落在该覆盖域内，不能证明 row traversal 已成为完整的编译器能力。

对照 TileLang 的 pipeline planning：它先用 `BufferRegionCollector` 收集 statement 的 reads/writes，再判断 replayability、pipeline write buffer 与 stage scheduling。对照 Triton 的布局/调度 passes：它们基于 op types、effects、dominance、use 与 encoding 进行变换。两者都不是靠“一个 entry 没有名字分支”来证明泛化，而是有明确的 dependence/access representation 支撑更宽输入。

Intent 当前与成熟实现的差距是：row traversal 的 typed role 已经存在，但依赖、访问区域与 execution segmentation 还没有形成足够通用的分析输入，导致 pass 仍然靠一组形状前置条件保护自身。

## 7.3 mutable load 次序：当前修法正确，但权威层仍偏低

`d3f0916` 修复了 blocked pointwise 中 mutable load 被重新排序的问题。保持 physical SSA 中作者可观察的读写次序是正确的，不能为了 tile 化把一次更新前的 load 移到更新后。

问题是当前 TileLang Bufferize 仍承担了大量次序恢复和执行选择。成熟 Triton 的 `ReorderInstructions` 会显式检查：

- side effects；
- dominance；
- first use；
- 是否跨越 write side effect；
- register pressure。

TileLang 的 pipeline planner则以 statement read/write `BufferRegion` 和 replayability 为输入。

Intent 如果只依赖 `kernel.walk` 或原始 block 顺序来保证正确，就没有形成真正的 dependence model。当前修复阻止了一个回归，但执行依赖仍应由 shared executable program 的 effects/def-use/region facts明确表达，provider Bufferize 只消费结果，而不是重新猜哪些 op 可搬、哪些必须保持顺序。

## 7.4 region/subregion authority：方向正确，但传播仍依赖白名单

这轮对 source ID、source dimension、range start/extent、subregion authority 的保留是正确方向。它比 provider 根据 tensor shape 重新猜逻辑区域更符合“作者写下的关系不能丢”。

当前差距是这些事实主要作为 attributes 附着在多个 op 上，并通过一组手工维护的 structured-op user cases 传播。每新增一种 op 或 reshape/broadcast/helper passage，都有可能漏传；漏传后 provider 又只能从邻接和 shape 恢复。

成熟 TileLang 在多个 op 和 pipeline analysis 中把 `BufferRegion` 当作正式对象；Triton 的 tensor/memdesc/encoding 与 pointer analysis也让 access footprint 进入 IR 与 analysis，而不是靠若干 consumer 白名单同步。

因此当前不是“shared facts 完全没有”，而是 authority 尚不稳固：同一 access relation 仍可能同时存在于 source ID attrs、range ops、fragment shape 和 provider reconstruction 中。它需要真正唯一的 representation/analysis source，而不是再增加更多属性别名。

## 7.5 TileLang Bufferize 仍然太厚

`lib/Target/TileLang/Transforms/Bufferize.cpp` 当前超过两千行，并且仍在做：

- constant/range expression simplification；
- fragment scalarization 与 shape判断；
- direct copy 与 per-lane gather 的选择；
- allocation shape 与 storage form；
- parallel loop 生成；
- sync/pipeline 相关结构；
- 某些 validity 与 access pattern 的重新识别。

其中不是所有内容都越界：TileLang 的 shared/local fragment、`T.copy`、pipeline primitive 确实需要 provider-specific legalization。但“读 shared GPU Program 后确定性翻译 TileLang form”与“重新决定 value/access/execution structure”混在同一个 Bufferize 中，说明第四轮所谓 leaf 横向闭合并没有把 provider materializer 真正掏空。

对照 `ref/tilelang/src/transform/pipeline_planning.cc`，TileLang 自己也是先收集 read/write `BufferRegion`、分析 replayability 与 pipeline write buffer，再形成 schedule；不是到 terminal string generation 时根据 shape 临时猜。

当前 Intent 最大的 provider 边界泄漏就在这里：shared GPU Program 没带够或没规范化好的 execution/access/storage facts，由 TileLang Bufferize 重新构造。它解释了为什么 5090 某几个路径可以靠加代码跑通，而横向 37 项仍大面积失败。

## 7.6 candidate legality 与 tuner：当前既不是成熟 cost model，也不能全推给下层

Prompt 5 的一部分修复通过收缩/恢复候选解决了 timeout 和 row performance，这本身合理。Intent 应声明合法参数范围，provider tuner 在范围内实测 winner。

当前问题有两类：

1. 某些 candidate set 是固定数组，例如 BLOCK_M/N/K 与 ROW_WORKERS `{1,2,4,8}`，合法性和有用性没有全部由 typed device/resource/access facts推出来；
2. 某些 provider legality 只做“存在一个 shape 能过”的局部检查，例如 TileLang `isLegalMmaWarpPartition`，不足以覆盖多 GEMM、layout、copy、pipeline 与 workspace 联合约束。

这会导致两个相反问题：

- 候选太宽，cuTile/TileLang 首次 JIT 在 300 秒 worker timeout 内跑不完；
- 候选太窄或错误排除，真实 winner 根本没进入搜索空间。

成熟性不是 Intent 替 tuner 选 winner，也不是把所有组合无条件扔给 tuner。成熟的边界应是：

- pass 根据 typed legality 与设备资源计算合法范围；
- source 与 generated 使用相同合法 candidate set；
- provider autotuner各自实测 winner；
- candidate 数、失败原因和 winner成为可解释证据；
- 结构性 form 不藏在 tuner 数值中。

当前 runner 和 CSV 没有保存这些逐 entry 证据，因此本轮不能证明 timeout 是纯下层成本，也不能证明当前 winner 是编译器空间内的真实最优。

---

## 8. 为什么 cuTile/TileLang 以前通过很多，现在大量失败

## 8.1 先给宏观答案

不是一个单一 shared bug，也不是简单把 timeout 调大就能恢复。

过去的高通过率来自旧 compiler/materializer 执行链；第三、四轮删除了旧链，换成新 canonical KIR → shared GPU Program → provider Program 的唯一执行路径。第五轮的表是新链第一次完整横向运行。旧链能跑，证明算法、source/runtime 和目标生态曾经可工作；新链失败，主要证明新链尚未把旧链覆盖过的 physical realization 横向搬完。

因此这既是回归，也是架构重构的未完成：不能用“新旧链不同所以不算退化”回避，因为用户要求已有能力不得退化；也不能把全部问题归成 shared，因为当前失败明确跨越 shared verification、provider verification、provider JIT 和 candidate timeout 多层。

## 8.2 旧表与当前表的数量变化

重构前较完整的一组旧表（提交 `8d27372`）：

| provider/device | 行数 | pass |
|---|---:|---:|
| cuTile 5090 | 37 | 35 |
| cuTile H100 | 37 | 34 |
| TileLang 5090 | 37 | 19 |
| TileLang H100 | 37 | 18 |

更早的 30-entry 表（提交 `6bc7a61`）：

| provider/device | 行数 | pass |
|---|---:|---:|
| cuTile 5090 | 30 | 30 |
| cuTile H100 | 30 | 28 |
| TileLang 5090 | 30 | 15 |
| TileLang H100 | 30 | 15 |

当前：

| provider/device | 行数 | pass |
|---|---:|---:|
| cuTile 5090 | 37 | 14 |
| cuTile H100 | 37 | 13 |
| TileLang 5090 | 37 | 9 |
| TileLang H100 | 37 | 0（旧快照，未随最新修复刷新） |

即使 registry 行数同为 37，也不应假定是完全相同的 37 个 runtime-visible entry；中间 inventory 和接线发生过调整。真正可靠的是逐 entry 对比和失败阶段，而不是只看分母。

## 8.3 cuTile：三种问题同时存在

### A. shared physical-program coverage gap

当前每台有 6 个 `physical_program_verification_failed`。这类失败发生在进入 cuTile provider 之前，根因属于 shared GPU Program 的 construction/refinement/verifier 覆盖不足。集中形态包括：

- region fold/scan；
- runtime reduction/free axes；
- split-K 或 staged-looking、但必须在单 kernel physical program 内表达的结构；
- attention/ragged/sparse 的联合 axis roles。

这部分明确是 shared 问题，不能在 cuTile leaf 里补形状 matcher。

### B. cuTile provider legalization/JIT gap

两机各有 provider verification 或 JIT 失败。这说明 shared program已经产生，但 cuTile Program 无法合法表达或下层首次 launch 失败。它们可能涉及 gather/scatter、collective、layout/tile、资源约束或当前 cuTile API 能力。

这部分不能自动上推成 shared；需要看 shared fact 是否已经足够。如果足够，就是 cuTile-local form/legalization；如果 provider 仍回读 shape/relation 重建结构，才说明 shared 表示或 provider pass 边界有问题。

### C. candidate/JIT timeout

两机各 14 个 timeout，是当前最大块。runner 外层 worker timeout 为 300 秒；cuTile 的 generated/source 都可能包含较宽候选或 `exhaustive_search`。旧链能在这些 entry 上运行，而新链出现集中 timeout，至少说明：

- 新链的候选笛卡尔积、生成结构或首次编译成本发生了变化；
- runner 现在没有足够细的阶段进度判断是单 candidate 慢、候选过多、还是某个候选污染 context；
- 不能仅凭 `worker_timeout` 归因为“cuTile 下层成本”。

`d580ba8` 对 dense GEMM 的修复证明其中至少一部分是我们声明 candidate space 的问题，而非工具链硬边界。但只修 dense GEMM 不能外推到剩余 14 项。

所以 cuTile 的宏观定性是：**shared coverage、provider form、candidate/JIT 成本三者叠加；当前证据不足以给 14 个 timeout 逐项定性。**

## 8.4 TileLang：shared gap 更重，provider reconstruction 也更重

TileLang 新链回退更明显。当前失败包含：

- physical program verification / construction；
- provider program verification；
- provider JIT/initial launch；
- KIR parse/compiler process/adapter 与 implementation gap；
- H100 未刷新造成的证据缺失。

shared 层尚未横向覆盖的典型结构包括：

- multi-contraction 与同一逻辑轴的多重角色；
- region fold/scan；
- runtime reduction/free axes；
- sparse physical relation；
- ragged/attention 的联合 access 与 validity。

TileLang provider 侧尚未闭合的典型结构包括：

- atomic/scatter；
- sparse four-buffer / `gemm_sp` 形式；
- loop/workspace/pipeline 的合法 bufferization；
- fragment/shared storage、copy 与 synchronization 的联合选择。

`bab7a40` 关闭三个 5090 entry，说明一部分确实是我们 provider path 没做完，而不是 TileLang 0.1.13 无能力。但这三个 entry 仍慢 1.17—1.58 倍，说明“能发射”与“provider-native form 正确”是两个阶段。

TileLang 的宏观结论是：**shared 与 provider 都未闭合，而且 provider Bufferize 仍承担过多从 shape/邻接重建执行结构的工作。H100 缺最终复测，不能给当前代码一个可靠通过数。**

## 8.5 为什么不能简单放大 timeout 解决

适当放大超时可以区分“最终能完成但首次编译很贵”和“实际卡死”，并且多个独立 entry 可以并行跑以缩短墙钟时间。用户关于推进优先级的判断是对的：测试阶段不必为了微小显存扰动把所有工作完全串行化。

但只放大 timeout 不足以完成根因调查：

- 如果候选空间是我们的 compiler 无约束地产生的，等待更久是在掩盖 search-space 问题；
- 如果某个非法候选使 CUDA context 失效，后续候选都会伪失败；
- 如果 shared program本身构造了不合理的循环/fragment，provider JIT 变慢是上层结构错误的症状；
- 如果单个合法 provider primitive 的编译就很慢，那才是下层成本。

当前本轮只在 dense GEMM 与少数 TileLang sparse/grouped entry 上做到了这层区分，剩余 timeout 仍未分类。

---

## 9. 本轮是否“跑偏”

答案不是简单的“是”或“否”。

### 9.1 没有跑偏的部分

- 第一次把新链推进到真实六表，暴露了此前被结构自评遮住的缺口；
- measurement order、scope 与 candidate contract 的公平性修复是必要的；
- scalar/branch schema、axis provenance、Cartesian pointwise、fragment、histogram、row indirect/persistent 的区分都来自跨 entry 的 shared facts，不是 kernel-name 分支；
- grouped GEMM 与 MoE 的联动回归最终被修回，说明没有为了一个数字牺牲另一个已通过项；
- cuTile/TileLang 少数 timeout 和 provider path 被真实关闭，而不是只改 CSV 状态。

### 9.2 明显跑偏或尚不成熟的部分

- Prompt 5 应是最终验证，却继续新增大规模 shared/provider executable structure；这说明前轮完成度判断过于乐观；
- 修复方法长期从 failing entry 和 generated source 倒推，直到本报告才系统地与成熟 ref 的 IR/dependence/access-region 机制对照；
- 多处 semantic decision 仍依赖裸 `operator_kind` 整数、固定 candidate 数组、单一 segment 与 recognizer shape gate；
- TileLang Bufferize 仍像第二个 compiler，重新建立 access/storage/copy/sync，而不是薄 provider legalization；
- 当前“pass”有些只是把一个候选或一组固定规则放进独立文件，换输入就 `unhandled`；形式上是 pass，决策空间仍很窄；
- 六表没有在同一最终状态并行重跑，报告前也没有完整说明 Prompt 5 到底完成多少；
- cuTile/TileLang timeout 只解决少数，长期测试时间被局部 JIT 消耗，而没有尽早形成分阶段、并行、可定位的执行策略。

所以更准确的总评是：

> 这轮把新架构从“少数纵向路径成立”推进到了“更多真实 entry 能运行”，也修掉了一批真问题；但它仍没有证明 shared GPU Program 与 provider boundary 已横向成熟。当前最危险的不是显式 kernel-name 特判，而是更隐蔽的编号语义、形状门、固定候选和 provider reconstruction。

---

## 10. 当前真正未关闭的事项及性质

这里不是下一轮计划清单，而是第五轮结束条件中仍缺失的事实。

### 10.1 验证事实缺失

- 当前 `HEAD` 上的六表两机并行全量不存在；
- TileLang H100 没有吸收后续修复；
- cuTile/TileLang timeout 未逐项记录 candidate 数、阶段进度和单 candidate 成本；
- source/generated candidate set 与 winner 没有逐 entry 证据；
- 六表不是同一运行时段和同一代码状态，不能直接做最终横向统计。

### 10.2 Triton 性能未闭合

- 21 个当前 pass 行超过 1.05×；
- 其中多项是 1.2—2.68× 的结构性差距，不是微秒噪声；
- strict eligibility 尚未逐项重审，既不能全部算失败，也不能全部排除；
- grouped GEMM/MoE 已闭合到约 1.03/1.01，但 split-K MoE 仍明显落后。

### 10.3 shared GPU Program 未闭合

- region fold/scan、runtime axes、multi-contraction、sparse/ragged 联合结构仍能在 physical verification 前失败；
- dependence/access region 尚未成为足够统一的 analysis authority；
- row traversal 与多种 blocking rule 仍受窄 shape/segment 前置条件控制；
- `operator_kind` 裸整数跨 pass/provider 承担语义权威。

### 10.4 provider 未闭合

- cuTile：14+14 timeout，加上 verification/JIT gaps；
- TileLang：Bufferize 过厚，atomic/scatter/sparse/workspace/pipeline 等多种 provider-native form 未横向闭合；
- H100 TileLang 当前状态未知；
- 已跑通的 TileLang grouped/sparse 项仍有 1.17—1.58× 差距。

### 10.5 旧能力回归的责任

旧链 cuTile 曾达到 35/37、34/37，TileLang 达到 19/37、18/37。新链当前远低于这个状态。即使旧链已按架构要求删除，这些旧通过项仍是能力回归证据，不能因为实现路径换了就注销。

当前宏观归因是：

```text
旧 materializer 覆盖的 executable structure
        ↓ 重构删除旧链
new shared GPU Program 尚未横向表达完整
        + provider Program/Bufferize 尚未横向消费完整
        + candidate/JIT 成本未受约束
        ↓
cuTile/TileLang 大面积 verification、JIT 与 timeout
```

不是一个 shared 修复就能全部恢复，也不是全部下放给 leaf。每个失败必须沿当前唯一执行链定位到最早缺失事实的层。

---

## 11. 最终状态判断

第五轮没有完成，当前编译器不能被描述为“全量回归完成、Triton 已追平、cuTile/TileLang 边界清楚”。

它目前处于这个更准确的状态：

- canonical DSL/KIR 与新 GPU Program/provider 链已经成为唯一执行路径；
- Triton 大多数 entry 可以运行，103 个通过性能行中 82 个在 1.05× 内；
- shared GPU Program 已从少数纵向 skeleton 扩展到 pointwise、reduction、region fold、contraction、grouped/row traversal 的更多真实结构；
- measurement 公平性比本轮开始时更好；
- grouped GEMM/MoE、AdamW、histogram、varlen convolution、QKV、RoPE 等一批真实问题已闭合；
- cuTile/TileLang 新链远未恢复旧链覆盖，且失败跨 shared、provider 和 JIT/search 三层；
- mature-compiler 对照揭示出 typed arithmetic、dependence/access-region authority、provider Bufferize 边界和 candidate legality 仍需结构性收敛；
- 没有最终同状态、双机并行六表，因而没有最终性能和覆盖结论。

Prompt 5 的实质完成度据此记为 **44%**。最关键的未完成不是再多修几个绿色格子，而是让当前 shared/provider architecture 对旧有通过项和新增真实结构形成横向、可解释、能在最终六表中重复验证的能力。
