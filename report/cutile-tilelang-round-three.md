# 第三轮：cuTile 与 TileLang provider 接纳和性能状态

## 1. 范围与结论

本轮把前两轮在 Triton 路径上确认过的事实带到 cuTile 和 TileLang，但没有把 Triton 的
target spelling 直接复制过去。检查顺序仍然是：

1. 先读 Physical Program，确认 shared Plan 是否已经表达访问、范围、stream、contract 和
   ownership；
2. 再判断 provider pass 是否缺少 target-native form；
3. 最后才看 terminal source 的机械拼写和 provider tuner 的参数范围。

最终留下的代码改动有三组：

- 把 Triton 原先私有的 causal prefix 判定收敛成一份共享 derived-fact 查询，并给 cuTile
  增加显式的 `prefix_boundary` stream form；
- 扩充 cuTile 的 `program_m × stream` 联合合法候选，解决已有 search surface 没覆盖 source
  使用的二维 tile 组合；
- 闭合 TileLang 的 pointwise-lane 拼写和 structured rank-one bulk bounds，并在 provider pass
  明确拒绝当前不能正确投影的 structured-indirect reduction replay。

本轮最重要的定性修正是：之前把五个 TileLang single-row/subwarp contraction 写成
“TileLang 0.1.13 没有能力”并不准确。ref 中存在真实、可运行的 SIMT GEMV 实现；缺的是
Intent 的 TileLang provider program 尚未表达 thread topology、vectorized K load、thread
all-reduce 和唯一 writer。它是我们的 provider-form 缺口，不是 target 语言缺口。

这是一轮定向验证，不是全量轮。`report/baseline-new/` 六张固定 CSV 没有覆盖；下文同时列出
固定表事实与本轮定向事实，避免把局部结果伪装成新全量。

## 2. 哪些 Triton 发现可以复用

| Triton 侧发现 | cuTile | TileLang | 本轮处理 |
|---|---|---|---|
| `region_end` 界定 causal prefix，prefix 内 mask 可由消费者中和 | cuTile 的 stream/`ct.mma` 结构可表达同样两段循环 | TileLang 当前 attention form 的 pipeline 和 state materialization 不同，不能只换一句语法 | classifier 收敛到 common derived fact；cuTile 新增 provider-local form；没有向 shared Plan 塞 target 字段 |
| query/program tile 与 stream tile 需要联合搜索 | `program_m`、`stream`/`stream_contract` 都是 provider tuner 可实测参数 | TileLang 已有 matrix profile，但纯 pointwise lane 没有 spelling/candidate | cuTile 补联合候选；TileLang 补 `pointwise_lane → TILE_SIZE_M` 与候选 |
| 二维带步长访问是 provider transfer form | cuTile view/load 本身保留运行时 shape/stride，当前样本没有证明需要第二套 descriptor op | TileLang `T.copy` 能消费带 stride view，真正失败点在 index span 与 transfer span 的绑定 | 没有为了“形式对称”造三份 descriptor；只修真实撞到的 TileLang span 绑定 |
| stage/occupancy 等参数应交给 provider tuner，同时非法候选应在最早有信息处排除 | cuTile JIT 能独立淘汰资源非法候选 | TileLang autotuner 同样逐候选编译、验证 | 没有新增 shared resource 猜测或设备型号分支 |

共同 classifier 只沿 Kernel IR 的 use-def 与 Physical Program 的 region binding 识别：

- stream 的 stop 必须是作者写下的 `region_end`；
- boundary 必须绑定 ownership range；
- compare 两侧必须分别唯一来自 boundary ownership 与 stream traversal；
- 只收集该 stream 自己 body 中的 neutralizable mask，显式跳过嵌套 stream。

它没有按 kernel 名、attention 名或 op 数量匹配，也没有选择 tile。Triton 和 cuTile 各自在
provider pass 中决定是否采用这个 form，并把 boundary axis 与 mask node 写成 target-local attrs。

## 3. cuTile

### 3.1 causal prefix provider form

旧 generated cuTile 对 causal attention 的每个 K tile 都保留逐元素谓词。新路径由 cuTile
provider pass 选择 `prefix_boundary`，terminal 只机械兑现已经写入 provider program 的三项事实：

- prefix 的物理终点；
- boundary ownership axis；
- prefix 中可以中和的 mask node。

生成形态是两段：完整可见 prefix 内移除这些 mask，tail 继续使用作者原来的 predicate。第二段
不是 terminal 临时重新分类出来的；分类和合法性检查都已经发生在 provider pass，translator
只是把选定 form 展开成两段 cuTile 循环。

ragged stream 和 `partition(count)` stream 没有强行套用该 form。它们有复合边界，本轮继续走
原来单循环路径，避免把 dense causal 事实错误推广到不规则关系。

### 3.2 tuner surface

固定表中的 dense attention 与 MLA source 都显式搜索 query/program tile 和 K/stream tile 的
组合；旧 Intent cuTile tuner 主要按单 role 或 `query × stream` 建 profile，`program_m × stream`
组合不完整。本轮增加：

- `program_m = 256` 的合法候选；
- `program_m × stream` 与 `program_m × stream_contract` 的联合 profile；
- occupancy 与 `num_ctas` 仍由 cuTile tuner 实测，不在 shared pass 写设备经验表。

这不是算法或 Plan 结构变换；同一份 provider program 只替换参数值。候选中仍有少数 TileIR
单候选编译超时，它们被 provider tuner 独立淘汰，没有污染其它候选。

### 3.3 性能结果

以下“之前”来自当前固定 CSV，“本轮”来自最终代码的定向复测：

| entry | 设备 | 之前 generated/source/ratio | 本轮 generated/source/ratio | generated 变化 |
|---|---|---|---|---:|
| `dense_attention_forward` | 5090 | `2.941328 / 1.272032 / 2.312307` | `2.488144 / 1.277952 / 1.946978` | -15.4% |
| `dense_attention_forward` | H100 | `2.391152 / 1.336688 / 1.788863` | `1.684432 / 1.333808 / 1.262874` | -29.6% |
| `mla_prefill` | 5090 | `1.964576 / 0.934912 / 2.101349` | `1.561904 / 0.934912 / 1.670643` | -20.5% |
| `mla_prefill` | H100 | `2.569104 / 0.554224 / 4.635498` | `1.059680 / 0.533792 / 1.985193` | -58.8% |

causal prefix 本身在 5090 dense attention 上只带来小幅追加收益；主要收益来自补齐联合候选。
H100 对新增组合更敏感，因此降幅更大。两机都保持数值通过。

### 3.4 还没有关闭的 cuTile 差距

`grouped_gemm` 定向复测仍为：

| 设备 | generated ms | source ms | ratio |
|---|---:|---:|---:|
| 5090 | 1.590608 | 0.634528 | 2.506758 |
| H100 | 2.627552 | 0.463120 | 5.673588 |

这个差距不是再补 `program_m` 常数能解决的。source 使用 single-worker persistent traversal 跨
group 分配工作；当前 shared `PersistentTraversal` 对 ragged contraction 明确不选择 persistent。
cuTile terminal 只是读 `launch.persistent`，没有自己补第二个判断。因此它是尚未关闭的 shared
execution-policy 问题，不能在 cuTile leaf 加 grouped-GEMM 特例。

固定表中更大的 `absorbed_mla_decode`、`splitk_mla_decode` 差距也没有被本轮两项改动消除。
Plan 已经有 head ownership、K stream、两次 contraction 与 split reducer；source 另外使用
16-head packing、TMA/MMA operand form、可见范围收紧和专门的 partial layout。责任仍在 cuTile
provider form/tuning，而不是算法语义或 terminal 字符串。

## 4. TileLang

### 4.1 pointwise lane 和 structured rank-one bounds

shared Physical Program 已经会给纯逐元素的一维程序选择 `pointwise_lane`，Triton 和 cuTile 都有
对应 provider spelling；TileLang 原来没有。这导致 Plan 已经选了物理 lane，TileLang 却无法把
它绑定到 block parameter。本轮补成 `pointwise_lane → TILE_SIZE_M`，并给 tuner 提供
`64/128/256/512` 候选。

`per_token_fp8` 的另一个失败来自 structured rank-one index：旧 TileLang bulk bounds 只接受
singleton index tile，即使 index tensor 的 Physical Program 已经给出一段连续物理 span。修复后：

- base 继续使用现有 `structuredIndexExpression`；
- span 直接读取 index tensor 自己的 selected physical extent；
- transfer tile 必须与该 span 精确一致，否则 provider program 立即拒绝；
- 多轴 data-dependent index、没有 canonical base 的 index 没有被顺手放开。

这避免了从目标 tensor shape 猜跨度，也避免了“删掉 singleton 检查后默认连续”的 OOB 风险。

结果：

| entry | 设备 | 固定表 | 本轮定向结果 |
|---|---|---|---|
| `per_token_fp8` | 5090 | `compiler_invocation_failed` | pass，`0.299920 / 0.216544 = 1.385030x` |
| `per_token_fp8` | H100 | `provider_program_failed` | pass，`0.320528 / 0.127808 = 2.507887x` |

它现在“能正确投影”，但 H100 性能仍明显落后 source；本轮没有把“从失败变成能跑”写成性能闭合。

### 4.2 structured indirect reduction replay

同一处 bounds 放开后，`mamba_chunk_state` 能进入 TileLang JIT，但所有候选都失败。中间曾试过把
所有 reduction-shaped producer 在 terminal 重放时强制改成 K tile；该实验会同时误拒绝原本通过
的 `grouped_gemm`，说明“结果 shape 绑定 K”不是足够的合法性判据，相关代码已全部删除。

最终保留的检查更窄，也直接读取 Physical Program：

- contract 必须选择 producer replay；
- replay 中存在 `tensor_indexing = structured` 的 deferred transfer；
- transfer domain 明确包含该 contract reduction axis。

当前 TileLang provider program 不能把这种 structured indirect transfer 在 K replay 中重新定形，
所以在 provider pass 立即报错，不再让全部候选进入下层后失败：

```text
the current TileLang producer-replay form cannot retile a structured
indirect transfer along the contraction reduction axis
```

`grouped_gemm` 的 deferred weight load 是直接 region index，不是 structured indirect transfer；两机
均继续数值通过：5090 `1.334608 / 0.603760 = 2.210494x`，H100
`1.558688 / 0.589872 = 2.642417x`。

### 4.3 ref 核查纠正了哪些旧结论

#### Single-row/subwarp contraction

`ref/tilelang/testing/python/kernel/test_tilelang_kernel_gemv_simt.py` 证明 TileLang 有高性能 SIMT
GEMV：二维 thread binding、vectorized K load、`tvm_thread_allreduce` 和唯一 writer 都已存在。
因此下面五个 entry 的失败统一定性为 Intent TileLang provider form 缺失：

- `gqa_decode`；
- `varlen_gqa_decode_logits`；
- `paged_mla_decode`；
- `native_sparse_attention_forward`；
- `native_sparse_attention_decode`。

本轮把诊断从“TileLang 不能 pad native MMA”改为“当前 TileLang provider program 没有 subwarp
SIMT GEMV form”。没有用串行点积、原子-only 或把 row 假装补成 16 的慢路径冒充支持。

要真正接入该 form，provider program 还需显式承载 reduce-thread 数、vector width、all-reduce
scope 和 writer lane；当前 `ContractOp` 的 MMA attrs 不足以机械打印这段结构。本轮没有把这些
决定藏进 terminal。

#### FP8、strided copy 与 scan

- TileLang ref 有 FP8 dtype/layout/MMA 基础，不能把所有 FP8 runtime-lane 失败归为语言没有；当前
  `fp8_lighting_indexer` 是我们的 provider form 与下层 JIT 组合尚未闭合。
- TileLang `T.copy` 可以处理带 stride view；本轮没有发现一条需要新增 shared“strided descriptor”
  才能表达的 TileLang case。
- ref 有固定 combine 的 reduce 与 cumsum/cummax，但没有与 Intent generic scan closure 等价的通用
  scan surface。generic scan 继续是明确 target capability subset；没有手写串行 loop fallback。

### 4.4 “18 个 provider_program_failed”到底是什么

固定 CSV 两机各 37 行、各 15 pass、各 22 non-pass。“18 个 provider_program_failed”是两机
行数相加：5090 8 行、H100 10 行，不是 18 个独立算法；去重后是 10 个 entry。

本轮 `per_token_fp8` 两机定向通过后，在其它固定状态不变的前提下：

- 定向状态为两机各 16/37 pass；
- provider-program failure 为 5090 8 行、H100 9 行，共 17 行；
- 去重后 9 个 entry。

完整的原 22 个 non-pass 去重 entry 现状如下：

| entry | 当前性质 |
|---|---|
| `gqa_decode`、`varlen_gqa_decode_logits`、`paged_mla_decode`、`native_sparse_attention_forward/decode` | Intent TileLang provider 缺 subwarp SIMT GEMV form；TileLang ref 已证明 target 能力存在 |
| `mamba_chunk_state` | Intent TileLang provider 缺 structured-indirect reduction replay form；现已在 provider pass 早拒绝 |
| `linear_attention_forward/backward` | Intent TileLang provider 只能投影同一 worker 上的 two-axis group；当前三轴 physical group 尚无合法 provider form |
| `sparse_mla_backward` | 5090 为 source adapter preparation；H100 已进入 provider program，仍有 target form 缺口 |
| `persistent_mla_decode` | Intent 侧尚无与 source cooperative persistent algorithm 一致的作者程序 |
| `deepseek_topk_selector` | Intent 仍缺该算法需要的 typed bit reinterpret/shared histogram/barrier 合同 |
| `block_fp4_quant` | Intent 仍缺 E2M1 conversion 与 packed-nibble output ABI |
| `w4a8_gemm`、`mamba_chunk_scan`、`varlen_block_causal_attention`、`dequant_bf16_fp4` | worker timeout/下层首次编译成本；不是 target capability 声明 |
| `block_sparse_gqa_decode`、`retention_forward`、`mhc_pre`、`mhc_post` | provider source 已生成，停在 TileLang JIT/initial launch；尚不能仅凭状态称为语言不支持 |
| `fp8_lighting_indexer` | 5090 compiler invocation、H100 provider JIT；ref 已证明 FP8 基础存在，当前是 provider form/JIT 未闭合 |
| `per_token_fp8` | 本轮两机定向数值通过，不再是当前缺口；固定 CSV 仍保留旧全量状态 |

这里没有把 timeout 写成 unsupported，也没有把 provider JIT 失败升级成语言能力边界。

## 5. 回归与数值验证

本轮实际跑的是受影响 entry，不是全量：

- cuTile：`dense_attention_forward`、`mla_prefill`、`grouped_gemm`，5090/H100；
- TileLang：`per_token_fp8`、`dense_gemm`、`grouped_gemm`、`mamba_chunk_state`、
  `linear_attention_forward/backward`、五个 subwarp entry，以及 `gqa_attention_backward`；关键路径
  在两机验证，较长的 GQA backward 数值回归在 5090 验证；
- Triton 回归：`flash_attention_forward` 仍为
  `2.689840 / 2.755584 = 0.976142x`，`mamba_chunk_state` 仍数值通过。

代表性的本地手动复现命令是：

```bash
PYTHONPATH=python:examples \
/home/kingdom/.venvs/intentdsl-cutile/bin/python -m repro.v2.runner cutile \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --output /tmp/intent-round3-cutile.csv \
  --kernel dense_attention_forward --kernel mla_prefill --kernel grouped_gemm
```

H100 使用同步到 `/tmp/intentdsl-round3` 的工作副本与独立 build；没有修改远端项目工作树。TileLang
在 H100 上显式使用 CUDA 12.2 的 `nvcc`，避免系统 CUDA 11.5 不认识 `sm_90a` 被误记成 compiler
能力失败。

## 6. 最终状态

本轮真实关闭了两类问题：

1. cuTile causal attention/MLA 的 provider form 与候选空间明显改善，两机均有真实数值和性能收益；
2. TileLang `per_token_fp8` 从两机失败变为数值通过，并把 Mamba 的深层候选失败提前为精确的
   provider capability 诊断。

没有关闭、但责任已经改准的主要问题是：

- TileLang subwarp SIMT GEMV 是我们的 provider form 缺口，不是 TileLang 没能力；
- TileLang linear-attention 三轴 group、Mamba structured replay 仍需新的 provider program form；
- cuTile grouped persistent traversal 是 shared execution policy 缺口；
- cuTile MLA head packing/TMA/partial layout 和 TileLang 若干 attention/FP8 pipeline 仍是 provider
  realization 质量缺口。

没有新增 kernel matcher、设备型号分支、shared target 字段、串行伪支持或第二条 executable path；
中间试验中没有形成通用判据的 replay 扩展已经删除。
