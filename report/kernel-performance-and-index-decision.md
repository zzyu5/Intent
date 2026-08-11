# Intent Kernel 编译器全量 Kernel、三后端与测量审计报告

## 结论

本报告对应实现提交 `00c0c41 compose paged attention across gpu targets`，数据来自该实现完成后的同一轮真实 GPU repro，不混用旧报告数字。原始运行日志位于本机 `/tmp/intentdsl-matrix-*.log`，不是提交进仓库的永久 artifact；本报告是这轮外部运行证据的结构化快照，复现仍以文末唯一 repro 命令为准。

当前状态可以压缩成五个结论：

1. 当前有 **23 个公开 repro 入口**。三个后端各执行一次，共运行 **69 条 repro 命令**。
2. GEMM、batched GEMM、grouped GEMM 和 varlen attention 会在一次入口内展开多个 case，因此实际得到 **29 个 case × 3 个后端 = 87 组数值与性能记录**；**87/87 数值 PASS**。
3. 全量测量已经按当前口径重新执行，不是从几轮旧报告中拼表。日志中只存在 `kernel-only`、`end-to-end`、`runtime-metadata` 三种 scope，旧的 `runtime-launch` 和 `timing=unavailable` 已不存在。
4. 分页 KV-cache decode attention 已把间接 ragged ownership、有状态流、GQA 多对一头映射、逻辑读取终点和尾块同时压到同一条 canonical Kernel MLIR → GPU Physical Plan → shared traversal → target projection 链路中。没有新增 paged-attention analyzer 或 emitter。
5. **TileLang 目前应保留，但不应作为默认性能后端。**它在 29 个 case 中有真实胜出项，并且是检查显式存储、流水线和 buffer 语义的最强 surface；但 paged attention 仍比 Triton 慢约 `17.6×`，说明它目前更适合作为表达力与分层验证后端，而不是性能 headline。

当前主链仍是：

```text
Python DSL
  -> canonical Intent Kernel MLIR
  -> shared GPU realization / Physical Plan
  -> shared operation traversal
  -> Triton / cuTile / TileLang capability projection and source emission
```

## 1. 本轮测量是否真的全部重跑

### 1.1 运行规模

统一入口为：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <kernel>
```

入口集合共 23 个：

```text
softmax
layer_norm
layer_norm_backward
rms_norm
fused_add_rms_norm
logsumexp
gemm
bf16_gemm
batched_gemm
quantized_gemm
dual_gemm
attention
attention_bias
varlen_attention
paged_attention
online_softmax
moe
grouped_gemm
swiglu_forward
swiglu_backward
shifted_row_copy
grouped_query_head_add
scalar_table_lookup
```

展开关系为：

- `gemm`：规则形状、M/N/K 三轴尾块；
- `batched_gemm`：NN、TN、NT、TT；
- `grouped_gemm`：规则 grouped case、member/K/N 尾块；
- `varlen_attention`：noncausal、causal；
- 其余入口各一个 case。

因此计数是：

```text
23 entries × 3 backends = 69 manual repro commands
29 cases   × 3 backends = 87 numerical/performance rows
```

69 条命令在本轮外部运行中全部退出成功；87 个 case-provider 组合全部完成真实 GPU 执行并通过 reference 数值对照。仓库本身不保存测试结果数据库，因此这些数字应理解为 `00c0c41` 在上述机器上的一次完整测量快照，而不是仅靠 checkout 即可静态证明的属性。

### 1.2 当前机器与环境

- GPU：NVIDIA GeForce RTX 5090 D，32607 MiB；
- Driver：580.95.05；
- Triton：3.6.0；
- cuTile：`cuda-tile 1.5.0`，TileGym 1.4.0；
- TileLang：0.1.13。

### 1.3 三种测量 scope

| Scope | 计时 callable | 计时器 | 用途 |
|---|---|---|---|
| `kernel-only` | 输出和 workspace 已在区间外准备，只重放实际 launch | CUDA Graph | 双方算法和 kernel 边界可对应的短核或稳定内核对比 |
| `end-to-end` | 为得到最终用户结果必须发生的完整 GPU pipeline | CUDA Event | 融合改变 kernel 数量、作者主导多 kernel 编排或上游只能整体调用 |
| `runtime-metadata` | 保留运行时 shape/sequence metadata 路径，不做 graph capture | CUDA Event | varlen 等必须在运行时兑现 metadata 的结构 |

所有 CUDA Graph 测量在每次 start event 前访问超过两倍 L2 容量的 flush buffer。冲刷本身不进入计时区间，避免同一地址反复 replay 造成不对称的 L2 驻留优势。

报告中的 event 时间只统计排入 GPU stream 的工作，不把 Python CPU 调度时间伪装成 GPU kernel 时间。

## 2. 全量 generated 性能表

表中数字为 `p50 / p95`，单位 ms；所有单元格均已数值 PASS。`K/E/R` 分别对应 `kernel-only/end-to-end/runtime-metadata`。

| Case | Scope（T/C/L） | Triton p50/p95 | cuTile p50/p95 | TileLang p50/p95 | 当前最低 p50 |
|---|---|---:|---:|---:|---|
| softmax | K/K/K | 0.3653 / 0.3684 | 0.3686 / 0.3707 | 0.3537 / 0.3559 | TileLang |
| weighted LayerNorm | K/K/K | 0.1848 / 0.1878 | 0.1907 / 0.1927 | 0.1755 / 0.1777 | TileLang |
| LayerNorm backward pipeline | E/E/E | 0.1163 / 0.1185 | 0.0971 / 0.0980 | 0.1120 / 0.1125 | cuTile |
| weighted RMSNorm | K/K/E | 0.1855 / 0.1876 | 0.1898 / 0.1911 | 0.1737 / 0.1759 | scope 不同，不横比 |
| fused add RMSNorm | K/K/K | 0.1879 / 0.1900 | 0.1884 / 0.1907 | 0.1793 / 0.1814 | TileLang |
| row logsumexp | K/K/K | 0.1836 / 0.1879 | 0.1900 / 0.1921 | 0.1754 / 0.1794 | TileLang |
| GEMM | K/K/K | 2.1166 / 2.1207 | 2.0756 / 2.0900 | 2.1242 / 2.1367 | cuTile |
| GEMM M/N/K tail | K/K/K | 2.1264 / 2.1406 | 2.0653 / 2.0791 | 2.1283 / 2.1386 | cuTile |
| BF16 GEMM | K/K/K | 2.0565 / 2.0696 | 2.0260 / 2.0511 | 2.0711 / 2.0920 | cuTile |
| BF16 batched GEMM NN | K/K/K | 0.1071 / 0.1089 | 0.0938 / 0.0956 | 0.1060 / 0.1085 | cuTile |
| BF16 batched GEMM TN | K/K/K | 0.1229 / 0.1249 | 0.0924 / 0.0939 | 0.1085 / 0.1096 | cuTile |
| BF16 batched GEMM NT | K/K/K | 0.1019 / 0.1046 | 0.0917 / 0.0928 | 0.1081 / 0.1101 | cuTile |
| BF16 batched GEMM TT | K/K/K | 0.1311 / 0.1331 | 0.0930 / 0.0945 | 0.1065 / 0.1071 | cuTile |
| fused quantized GEMM | K/K/K | 2.1428 / 2.1489 | 2.2175 / 2.2350 | 2.1640 / 2.1937 | Triton |
| gated dual GEMM | E/E/E | 0.6984 / 0.7006 | 0.6831 / 0.6851 | 0.7391 / 0.7423 | cuTile |
| dense attention | K/K/K | 5.4016 / 5.4084 | 6.0477 / 6.0580 | 6.1281 / 6.1384 | Triton |
| vector-bias attention | K/K/K | 5.6066 / 5.6167 | 6.4107 / 6.4333 | 6.2097 / 6.2290 | Triton |
| varlen attention noncausal | R/R/R | 0.4306 / 0.4348 | 0.4994 / 0.5025 | 0.4462 / 0.4492 | Triton |
| varlen attention causal | R/R/R | 0.3235 / 0.3271 | 0.3658 / 0.3704 | 0.2619 / 0.2660 | TileLang |
| paged GQA decode attention | K/K/K | 0.2621 / 0.2638 | 0.3651 / 0.3687 | 4.6236 / 4.6358 | Triton |
| streamed online softmax | K/K/K | 0.3926 / 0.4030 | 0.3557 / 0.3578 | 0.3915 / 0.4029 | cuTile |
| MoE | E/E/E | 8.8516 / 8.8749 | 10.1480 / 10.2805 | 11.4572 / 11.4985 | Triton |
| ragged grouped GEMM | E/E/E | 1.3104 / 1.3227 | 1.4620 / 1.4811 | 1.2776 / 1.2879 | TileLang |
| grouped GEMM member/K/N tail | E/E/E | 1.3329 / 1.3370 | 1.4588 / 1.4957 | 1.2838 / 1.2942 | TileLang |
| SwiGLU forward | K/K/K | 0.2329 / 0.2371 | 0.2412 / 0.2453 | 0.2882 / 0.2914 | Triton |
| SwiGLU backward | K/K/K | 0.1121 / 0.1141 | 0.1163 / 0.1184 | 0.1121 / 0.1141 | Triton / TileLang |
| shifted row copy | K/K/K | 0.0220 / 0.0241 | 0.0220 / 0.0241 | 0.0235 / 0.0239 | Triton / cuTile |
| grouped-query head add | K/K/K | 0.0036 / 0.0057 | 0.0036 / 0.0050 | 0.0033 / 0.0053 | TileLang |
| scalar table lookup | K/K/K | 0.0405 / 0.0445 | 0.0565 / 0.0584 | 0.0426 / 0.0446 | Triton |

除 RMSNorm 的 TileLang cell 使用不同 scope 外，剩余 28 个可横向观察的 case 中：

- Triton 单独最低 8 项；
- cuTile 单独最低 10 项；
- TileLang 单独最低 8 项；
- 另有 2 项并列。

这说明三个 provider 并非三条“恰好能运行”的重复路径；当前设备和形状下，它们确实在不同结构上形成不同赢家。

## 3. 上游 baseline 覆盖与可比性

87 个 case-provider cell 中，39 个取得了上游性能数字，48 个明确记录 `upstream baseline: unavailable`。没有 baseline 不等于 generated 失败，也不能用 PyTorch reference 时间代替高性能上游。

表中数字为 `generated / upstream p50`：小于 1 表示 generated 更快。

| Entry / case | Triton | cuTile | TileLang | 解释 |
|---|---:|---:|---:|---|
| softmax | 0.9972× K | 0.9998× K | 0.9542× K-D | TileLang 上游采用 online 写法；同任务单核，但内部算法不同 |
| LayerNorm | 1.0742× K | 0.5509× K | — | TileLang 无上游 |
| LayerNorm backward | 0.5557× E-D | — | — | 上游只能通过 autograd wrapper 调完整 backward；不是 kernel-only 结论 |
| RMSNorm | 1.0653× K | — | 0.4968× E | TileLang 只能按完整 callable 对比，不能与另两列横比 |
| fused add RMSNorm | 1.0633× K | — | — | 仅 Triton 有上游 |
| logsumexp | — | — | — | 三者均无上游 |
| GEMM | 1.0307× K | 0.9240× K | 0.9142× K | 同形状单核对比；tail case 三者均无上游 |
| BF16 GEMM | — | 0.9198× K | 0.9090× K | Triton 无上游 |
| batched GEMM NN/TN/NT/TT | — | 1.0175 / 1.0024 / 0.9951 / 1.0136× K | — | 只有 cuTile 有四布局上游；四者均在约 ±2% |
| quantized GEMM | — | — | — | 三者均无同融合边界上游 |
| dual GEMM | 0.9002× E-F | 0.8446× E-F | 0.8004× E-F | generated 融合双 contraction；上游由普通 GEMM 组合，优势属于 E2E fusion |
| dense attention | 1.0912× K | 1.2594× K | 0.9158× K | 同任务、同 kernel 数；目标调度不同 |
| vector-bias attention | 0.7927× K | — | — | 仅 Triton 有上游；上游数值误差更大但仍在既定 tolerance 内 |
| varlen attention causal | — | — | 1.0331× R | 仅 TileLang causal case 有上游；noncausal 无上游 |
| paged attention | 0.8951× K-D | — | — | xFormers 内层 paged kernel；算法/布局不同但计时区均只含 kernel |
| streamed online softmax | 1.0711× K | 0.9643× K | 1.0545× K | 三者均有上游 |
| MoE | 0.8694× E-D | 1.0618× E-D | 1.2163× E-D | routing、分组与 merge 编排不同，只作端到端观察 |
| grouped GEMM | 0.6987× E | 0.7820× E | 0.8953× E | 基础 case 有上游；尾块 case均无上游 |
| SwiGLU forward | 1.0246× K | 0.9897× K | — | cuTile adapter 的 packed ABI 在计时前准备 |
| SwiGLU backward | 1.0324× K | — | — | 仅 Triton 有上游 |
| shifted row copy | — | — | — | 索引机制 repro，无高性能上游 |
| grouped-query head add | — | — | — | 索引机制 repro，无高性能上游 |
| scalar table lookup | — | — | — | tensor-derived index 机制 repro，无高性能上游 |

标记含义：

- `K`：kernel-level scope，可用于同任务 kernel 性能判断；
- `E`：完整 GPU pipeline，可用于用户结果的端到端判断；
- `R`：包含必须兑现的运行时 metadata 路径；
- `D`：算法编排、布局或 adapter 形态不同，只能观察，不能归因成某个 emitter 更优；
- `F`：generated 改变了融合边界，优势是合法的端到端 fusion 优势，不是单 GEMM 更快。

分页上游来自未修改的 xFormers `splitk_kernels.py`。项目只在相邻 runtime adapter 中完成当前 ABI、页表、输出和调用参数接线；adapter 的准备发生在 graph capture 前，steady-state 数字只包含 xFormers 内层 kernel。这是项目定义的 steady-state adapter 边界，不是原始 xFormers Python callable 的端到端时间。cuTile 和 TileLang source 中没有找到任务与接口都足够接近的通用 paged-GQA baseline，因此明确留空，没有拼接伪 baseline。

## 4. 分页 KV cache single-query decode 如何从现有机制组合出来

### 4.1 DSL 写下的算法事实

分页 decode kernel 在 DSL 中直接表达：

1. `page_offsets + page_indices` 组成带索引映射的 ragged page ownership；
2. query head 通过 `query_head // HEAD_GROUP` 映射到 KV head；
3. 外层 state stream 每次推进一个逻辑 page；
4. 内层 state stream 在 page 内沿 token 轴推进；
5. logical token 由 `page_ordinal * PAGE_SIZE + token_in_page` 计算；
6. `logical_token < sequence_length` 决定最后一页的有效 lane；
7. online maximum、denominator、accumulator 跨 page 保持状态；
8. 输出为每个 `(batch, query_head)` 的单个 decode vector。

当前 query 代表序列末端的单个 decode token，因此它可以读取 `sequence_length` 以内的全部历史 KV。DSL/reference 使用 `is_causal=False` 表达“单个末端 query 对全部历史可见”，xFormers 内层 kernel 使用 causal flag；在这个 single-query decode 形态下两者语义等价。该结论不能外推到多 query prefill。

这些都是作者知道的算法结构，没有让 realizer 从 kernel 名称反推“这是 paged attention”。

### 4.2 共享 realization 承担的事实

共享层只补算法无法自行决定、但三个目标都需要的事实：

- indirect ragged member 的 ownership；
- page member 轴取 one-page tile，使页内访问恢复连续；
- ordered + reduction 轴统一取得 `stream_contract` 角色；
- GQA 的 floor-divide 索引沿 SSA 保留精确值和来源轴；
- `assume_in_bounds` 直接从 Kernel IR 读取，不再复制成独立 precondition cache；
- tensor-valued indirect index 与“还要不要边界谓词”分开建模；
- contraction 消费前已被逻辑 mask 中和的 K/V 尾 lane，可标记 `consumer_neutralized`；
- store 的 ragged 逻辑边界不能被 consumer neutralization 抵消，必须真实兑现。

这最后一点在全量回归中暴露并修复了一个真实错误：TileLang 原先会用 bulk store 把某条变长序列的 query 尾块写进下一条序列。当前投影在未中和的 ragged store 上机械选择逐元素谓词，不再依赖 attention 特判。

### 4.3 三个 target 只做投影

- Triton：把同一 tensor-indirect relation 投影为精确 pointer expression、broadcast index 和 mask；
- cuTile：把 tensor-valued index 投影为 `ct.gather` index tuple；间接性决定 gather，边界事实只决定 check-bounds；
- TileLang：把 singleton page index 投影为 `T.assume` 和 page-local bulk copy；不能由 `T.gemm` 表达的 `M=1` contraction 走通用 unit-row lowering。

没有一个 target emitter 重新决定 page tile、GQA ownership、逻辑 stop 或 token validity。三个 leaf 仍然复用既有的 ragged/staged 结构化 handlers；“没有 paged 专属 emitter”不等于“没有通用 ragged lowering”。

### 4.4 当前性能结论

| Provider | Generated p50 | Upstream p50 | 结论 |
|---|---:|---:|---|
| Triton | 0.2621 ms | 0.2929 ms | 与 xFormers 内层 paged kernel 同量级，generated 为 0.8951× |
| cuTile | 0.3651 ms | — | 正确运行；无可比上游 |
| TileLang | 4.6236 ms | — | 正确运行，但当前不是可接受的性能主力 |

TileLang 的差距不是页表或流机制又做了一遍，而是 0.1.13 的通用矩阵原语不能直接覆盖该 `M=1` contraction。当前 unit-row fallback 保持正确的 f32 accumulation，但其串行 reduction 使它比 Triton 慢约 `17.6×`、比 cuTile 慢约 `12.7×`。

## 5. 后端是否已经分裂成三套编译器

结论：**当前没有。**三个 leaf 的代码量不小，但本轮新增的算法理解、索引事实和机器决策仍在共享层；leaf 增长主要来自不同 API、ABI、buffer 和语法的投影。

| 问题 | 唯一 owner | Target leaf 允许做什么 | 当前审计 |
|---|---|---|---|
| 算法结构、state、ragged、contract 数值语义 | Kernel MLIR | 逐 op 读取并发射 | 共享 |
| logical axis roles、ownership、tile role、stream stop | GPU Physical Plan | 把已选 role 映射为参数和循环语法 | 共享 |
| index use-def、来源轴、in-bounds 前置条件 | Common analysis | 打印精确目标地址/索引 | 共享 |
| consumer neutralization、fill、store validity | Common proofs + Plan | 选择目标支持的 bulk/guarded spelling | 共享 |
| op 遍历与 unsupported 诊断 | Common emission lifecycle | 注册 leaf handler | 共享框架 |
| `tl.load` / `ct.gather` / `T.copy` | Target leaf | 能力检查与语法映射 | 合法差异 |
| target runtime、ABI wrapper、autotuner API | Target leaf | 接入真实下层工具 | 合法差异 |

验收层面：

- realization/emission 中没有 `if kernel_name == paged_attention`；
- paged attention 没有新增专属 analyzer、realizer 或 emitter 文件；
- staged gather 的 valid/fill 延迟发射、InOut merge ABI、ragged store boundary 等修复都按 op/SSA/ABI 语义生效；
- cuTile 的 `scatter_unique` 保留自身 op handler，不再被 generic tensor-indirect 分类误写成普通 scatter；
- TileLang 的 singleton indirect index 和 unit-row contract 是能力子集投影，没有回写或修改共享 Physical Plan。

仍需诚实保留一个架构观察：TileLang 的 singleton indirect-index 限制目前是 emitter 内显式 capability check，而不是一张独立声明式 capability 表。它仍然只“拒绝或拼写”已有决定，因此还没有成为第二个编译器；但以后不能在这里继续堆逻辑分析或机器选择。

## 6. TileLang 还需不需要保留

### 6.1 为什么现在不应删除

TileLang 29/29 case 数值通过，而且不是全面落后：

- softmax：三者最低，0.3537 ms；
- LayerNorm：三者最低，0.1755 ms；
- logsumexp：三者最低，0.1754 ms；
- grouped GEMM：三者最低，1.2776 ms；
- varlen causal attention：三者最低，0.2619 ms；
- dense attention 对自身上游为 0.9158×；
- GEMM 对自身上游为 0.9142×。

更重要的是，TileLang 要求显式表达 shared/local/fragment buffer、`T.copy`、`T.Pipelined`、同步和 `T.gemm`。从本项目的分层方法看，我们把它视作抽象线最低的 surface，因此用它检查 Physical Plan 是否漏掉存储、边界或流水结构；这是保留它的设计理由，不是由代码自动推出的定理。此次变长 store 覆盖错误就是在这条 surface 的本轮运行中首先暴露。

### 6.2 为什么它不应成为默认性能后端

- paged attention：4.6236 ms，对 Triton 为约 17.6×；
- MoE：11.4572 ms，对 Triton 为约 1.29×，对自身上游为 1.2163×；
- SwiGLU forward：0.2882 ms，慢于 Triton 0.2329 ms 和 cuTile 0.2412 ms。

当前最优策略不是“为了三后端齐全而强行让 TileLang 活着”，也不是立即删除，而是：

> 保留 TileLang 作为独立 surface、表达力验证后端和候选 provider；默认性能选择由逐 kernel 实测决定，paged attention 暂不把 TileLang 当性能候选赢家。

### 6.3 保留边界

TileLang 只有在以下边界内才值得继续存在：

1. 新结构继续从共享 Kernel IR + Physical Plan 自然投影；
2. TileLang leaf 只增加 capability check、target spelling、ABI/runtime 接线或 op handler；
3. 表达不了时允许明确 unsupported，不为它新增专属机器决策；
4. 若以后必须在 TileLang 路径里重新推 ownership、tile、stream stop、layout policy 或算法结构，应优先放弃该 backend，而不是接受第二套编译器。

按当前代码，尚未触发删除条件。

## 7. 当前完整能力与仍存在的缺口

### 7.1 已由真实 kernel 压到的结构

- 行归约、稳定 softmax、在线 softmax、logsumexp；
- affine normalization、weighted normalization、residual fusion；
- 2D/3D contraction、四种转置布局、双 contraction、三轴尾块；
- BF16 accumulation、混合精度、int8 输出量化；
- dense、bias、varlen、paged-GQA 四种 attention 结构；
- ragged ownership、indirect membership、stateful streaming、logical stop；
- MoE 两阶段 contraction、scatter-reduce、grouped GEMM；
- 多输出、作者主导多 kernel backward pipeline；
- 偏移索引、floor-divide 多对一映射、tensor-derived scalar index。

### 7.2 当前证据缺口

1. **上游覆盖不足。**48/87 case-provider cell 没有可比高性能上游，尤其 cuTile/TileLang paged attention、量化融合和三个索引机制 repro。
2. **部分比值只能作端到端观察。**MoE、dual GEMM、LayerNorm backward 等两侧算法或 kernel 数不同，不能当作 emitter 单核优劣。
3. **TileLang paged contraction 性能不成立。**功能与分层成立，但当前 unit-row fallback 慢一个数量级。
4. **能力声明仍部分隐式。**TileLang singleton indirect-index 等限制目前由 leaf handler 直接检查；只要它不承担决定仍是合法的，但继续增长会使能力边界难审计。

这些缺口没有被默认值、异常吞噬、伪 baseline 或 kernel 名特判掩盖。

## 8. 对应实现与复现

本报告对应：

- `00c0c41 compose paged attention across gpu targets`

关键源码位置：

- 分页 DSL：`examples/kernels/streaming/paged_attention.py`；
- 全量 repro 与测量 scope：`examples/repro/common/extended.py`；
- 唯一执行入口：`examples/run/repro.sh`；
- 共享 index/precondition：`lib/Target/Common/Analysis/IndexRelation.cpp`；
- 共享 Kernel facts/proofs：`lib/Target/Common/Realization/`；
- 共享 GPU 物理决策：`lib/Target/GPU/Realization/`；
- 三个 surface：`lib/Target/{Triton,CuTile,TileLang}/Emission/`；
- xFormers paged baseline 与相邻 runtime：`source/triton/xformers/attention/splitk/`。

任一结果都可通过同一条命令手工复现：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <kernel>
```

该入口完成 DSL lowering、`intent-compile` 构建、目标源码发射、真实 GPU 执行、reference 数值检查，以及存在时的上游性能对照；没有新增 test 目录、fixture 或另一套验证脚手架。
