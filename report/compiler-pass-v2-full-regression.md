# IntentDSL GPU Compiler 最终全量收尾报告

## 1. 最终结论

本轮完成了两台机器并行全量、全量暴露回归的通用修复、受影响 provider 的完整复跑，以及六张
`baseline-new` 固定表更新。最终快照覆盖：

```text
DSL
  -> canonical Kernel IR
  -> complete Physical Program
  -> shared GPU realization passes
  -> provider-local form passes
  -> terminal source
  -> provider compile/JIT
  -> GPU numerical comparison and steady-state measurement
```

最终共有 `112` 个 provider-specific registry entries；在两台设备上形成 `224` 行结果，其中
`161` 行完整通过，`63` 行停在明确阶段，没有任何 `numerical_failed`，也没有把失败压成宽泛的
`compile_failed`。

这份结果证明了两件事：

1. 编译器主链已经能横向覆盖 contraction、normalization、attention、ragged/MoE、scan/state
   stream、convolution、quantization、sparse、backward 和多阶段程序，而不只是一条规整 rowwise
   路径；
2. 编译器还没有达到“所有结构相同的 baseline 都在 5% 内”的性能闭环。`161` 个通过行中，
   `103` 个在 `1.05x` 内，`58` 个仍超过；数量级差距集中在 paged/MLA/state-stream/ragged/
   block-sparse 等尚未形成高质量 provider-native form 的结构。

因此，当前状态可以作为下一阶段稳定起点：编程模型和主链不需要再因这些结果随意变化；剩余工作
已经被压缩为明确的 shared Physical Program 缺口、provider form 缺口、算法/存储类型缺口、设备
资源边界和下层编译成本。它还不能被描述成“性能已经全面成熟”。

## 2. 执行与证据口径

### 2.1 两台机器真实并行

- 5090 使用当前工作树与本地独立 build；
- H100 使用同一代码快照、独立 build 和三个独立 provider 环境；
- 两台机器的 Triton、cuTile、TileLang 全量同时开始、彼此不等待，完整一轮各约 18 分钟；
- 最终一个 Triton tuner 合法性修复只影响 Triton，因此随后两台机器再次并行完整复跑全部 Triton
  entries；cuTile/TileLang 没有被无关重复测量；
- 每个 entry 在独立 worker 进程中运行，单 entry 首次编译超过 300 秒时由 watchdog 终止并记成
  `worker_timeout`，不会污染或永久阻塞后续 entry。

六张固定表为：

- `report/baseline-new/triton-5090.csv`
- `report/baseline-new/triton-h100.csv`
- `report/baseline-new/cutile-5090.csv`
- `report/baseline-new/cutile-h100.csv`
- `report/baseline-new/tilelang-5090.csv`
- `report/baseline-new/tilelang-h100.csv`

`report/baseline-new/` 只包含这六张 CSV。六表继续使用同一列合同：

```text
kernel,case,generated_p50_ms,source_p50_ms,ratio,status
```

CSV 中的 `1.05x` 只在本报告里作为观察分界，不是自动验收逻辑，也没有被写成 runner 阈值。

### 2.2 测量含义

- `generated_p50_ms` 与 `source_p50_ms` 是同一 entry 的稳态 GPU 测量；
- 短 kernel 使用 CUDA Graph，并按设备报告的 L2 容量在 replay 之间冲刷缓存；
- 多 kernel pipeline 只有在两边调用数和 scope 可对齐时才把 ratio 当成 kernel/pipeline 对比；
- adapter、source 首次编译、generated provider JIT、数值比较和 benchmark 是不同阶段，状态不会混写；
- `pass` 表示 generated 与 source 都完成数值对照和计时，不表示该 source 一定是该设备上的最优配置。

## 3. 最终全量矩阵

| provider / device | entries | pass | non-pass | 通过率 |
|---|---:|---:|---:|---:|
| Triton / 5090 | 38 | 34 | 4 | 89.47% |
| Triton / H100 | 38 | 36 | 2 | 94.74% |
| cuTile / 5090 | 37 | 30 | 7 | 81.08% |
| cuTile / H100 | 37 | 29 | 8 | 78.38% |
| TileLang / 5090 | 37 | 16 | 21 | 43.24% |
| TileLang / H100 | 37 | 16 | 21 | 43.24% |
| **合计** | **224** | **161** | **63** | **71.88%** |

按实际失败阶段汇总：

| 状态类别 | device/provider 行数 | 含义 |
|---|---:|---|
| `intent_implementation_gap` | 26 | 13 个 provider entry 尚无等价 Intent 算法/存储合同 |
| `physical_program_failed` | 10 | 5 个结构在 shared Physical Program 合法化阶段未闭合 |
| `provider_program_failed` | 12 | 6 个结构缺明确的 TileLang provider-native form |
| `provider_jit_or_initial_launch_failed` | 10 | provider source 已生成，但下层编译、布局、资源或首次 launch 失败 |
| `worker_timeout` | 2 | TileLang `mhc_pre` 首次编译超过 300 秒 |
| `adapter_preparation_failed` | 2 | 5090 上游 attention 准备核超 shared-memory 容量 |
| `source_provider_jit_or_initial_launch_failed` | 1 | H100 flash-attention backward 的 source 侧失败 |
| `numerical_failed` | 0 | 没有静默数值错误留在最终表中 |

两台机器通过数相近不是同一组边界：H100 能运行 5090 因 shared-memory 容量失败的 Triton
block-sparse GQA，但 H100 的 cuTile block-scaled GEMM 又停在 SM90 不具备对应 E8M0 scaled-MMA
能力。跨设备结果不能用 GPU 型号分支抹平。

## 4. 本轮关闭的真实回归

### 4.1 Physical axis 决定曾经不确定

#### 根因

axis role 的候选来自多个 `DenseMap`/`DenseSet`，后续 range 命名、program order 和 worker assignment
沿容器插入顺序消费。同一个 kernel 重复编译会产生不同的 `_PARAMETER_MAP`，使同一算法和设备可能
走不同物理程序。

#### 修复

shared GPU decision pass 现在按稳定的 canonical node order 排列所有 axis choices，再产生
program/lane order、range 名称和 worker assignment。Triton leaf 只消费确定结果，没有在 emitter
里补排序。重复编译 `rotary_qk_inplace` 已稳定产生同一 parameter map；dense GEMM 与 attention
smoke 没有因该修复改变算法路径。

该改动位于 `lib/Target/GPU/Realization/Plan/Decisions.cpp`，不包含 kernel 名称或 provider 分支。

### 4.2 state-stream 覆盖了外层 axis identity

#### 根因

Triton 进入 `state_stream` 时，把当前 stream offset 写回逻辑 axis 的全局 index binding。一个值同时
使用外层 row domain 与 stream region argument 时，外层 row 被错误替换成 stream tile。Mamba chunk
scan 因而把应为 `[S, C]` 的两个轴打印成同一个 `[C, C]`；当 tuner 选到 `C != S` 时才暴露。

另一个同类问题出现在 tensor-axis index：KIR 已保留精确的 region block argument，leaf 却从 axis
shape 重新合成 index，cross-entropy 的 label/gather 路径因此会丢失作者写下的来源。

#### 修复

- state-stream 只在没有外层 active index 时绑定 logical axis；stream block argument 继续使用独立
  region projection；
- tensor-axis index 优先读取 `RegionBindingOp -> selected range -> exact block argument`，不再从
  shape 或 role 猜来源；
- 找不到精确 provenance 就诊断，不恢复 extent/name fallback。

最终 Mamba chunk scan 与 cross-entropy 在两台机器均数值通过。改动位于
`lib/Target/Triton/Lowering/Handlers/Operations.cpp`。

### 4.3 Triton tuner 声明了静态或运行时不可能的 tile

#### 根因

Triton search space 只有 parameter role，没有已选 range 的 logical extent。`pointwise_lane` 可声明到
16384；当实际 Q=2048、另一维 D=128 时，下层会先编译一个超过 Triton tensor numel 上限的候选，
mask 只能防止访存越界，不能缩小编译期 tensor shape。

#### 修复

- terminal source 从 Physical Program 的 selected ranges 生成 provider parameter 对应的静态和运行时
  extent metadata；
- 静态 extent 在 config 构造时把候选限制到 logical extent 的 power-of-two ceiling；
- 运行时 extent 通过 Triton 原生 `early_config_prune` 在 JIT 前裁掉不可能候选；
- 若 joint constraints 没有配置可用则明确失败，不用默认值或低质量 fallback 假装支持。

这仍是 delegated tuner 的合法候选约束，不是 Intent 自建 cost model。最终 H100
flash-attention backward 的 generated delta/dK-dV/dQ 三个 kernel 已完成首次运行；该 entry 现在只剩
upstream source backward JIT 失败。相关实现位于
`lib/Target/Triton/Lowering/Materialization/Program.cpp` 与
`python/intent/runtime/tuning/triton.py`。

### 4.4 全量 runner 不再被单个下层编译永久阻塞

runner 现在按 entry 启动独立进程并在 300 秒后终止该进程组。`mhc_pre` 两机都真实进入 TileLang/
NVCC 编译，然后被记为 `worker_timeout`；它没有被误归为 DSL、Physical Program 或目标能力不支持。
这个改动只修证据边界，不改变任何 kernel、Plan 或 provider decision。

## 5. 覆盖情况

### 5.1 inventory 覆盖

| provider | registry entries / device | 至少完成数值运行 | family 数 | implementation gaps |
|---|---:|---:|---:|---:|
| Triton | 38 | 34（5090）/ 36（H100） | 11 | 1 |
| cuTile | 37 | 30 / 29 | 9 | 5 |
| TileLang | 37 | 16 / 16 | 8 | 7 |

所有 112 个 registry entry 都绑定了仓库内存在的同语言公开 source runtime；三家各自超过 30 条，且
README inventory 与 registry 一致。覆盖门类包括：

- dense/grouped/batched/sparse/scaled/quantized contraction；
- softmax、LayerNorm、RMSNorm、activation 与 fused normalization；
- dense、varlen、paged、split-K、block-sparse、sink/window/Gemma/MLA attention；
- MoE routing/expert projection 与 ragged grouped GEMM；
- causal/varlen convolution；
- Mamba/linear-attention/retention state stream 与 scan；
- FP8、W4A8、FP4/INT2 source inventory；
- attention backward、grouped GEMM backward 与多阶段 reduction。

这里的“source 已登记”不等于“Intent 已实现”。13 个 implementation gap 被明确保留：

- Triton：完整 `mamba3_siso_forward`；
- cuTile：`grouped_flash_decode`、`attention_sink_decode`、`gemma_decode`、`chunk_gated_delta`、
  `nvfp4_quantize`；
- TileLang：`persistent_mla_decode`、`deepseek_topk_selector`、`linear_attention_backward`、
  `sparse_mla_backward`、`bitnet_int2_decode`、`dequant_bf16_fp4`、`block_fp4_quant`。

这些条目没有 source wrapper 冒充 DSL 接纳：完整 Mamba3/chunk-gated-delta/attention backward 是尚未写下
的算法合同；FP4/INT2/top-k 是 packed storage、bit reinterpretation、barrier/intrinsic 等语言/KIR
能力；三个 decode entry 则要求 compiler-private split-K staging，不能把 partial/workspace 反塞给作者。

### 5.2 覆盖广不等于三家能力对称

Triton 的 34/36 个通过行说明主流 GPU tensor/stream 路径较完整；cuTile 的 29/30 个通过行说明同一
Physical Program 能投影到更高层的自动 tile 语言，但复杂 state-stream/MLA 性能仍不足；TileLang 只有
16 个通过行，清楚显示抽象线更低时，shared program 与 provider form 尚未为全部结构闭合。没有为了
追求对称数字恢复串行 dot、逐元素 ragged copy 或其它数量级慢的伪支持。

## 6. 性能状态

### 6.1 全表分布

| provider / device | pass | ratio <= 1.05 | ratio > 1.05 | > 2x | > 5x | generated <= source |
|---|---:|---:|---:|---:|---:|---:|
| Triton / 5090 | 34 | 25 | 9 | 2 | 1 | 20 |
| Triton / H100 | 36 | 31 | 5 | 3 | 1 | 25 |
| cuTile / 5090 | 30 | 21 | 9 | 6 | 3 | 17 |
| cuTile / H100 | 29 | 11 | 18 | 9 | 5 | 8 |
| TileLang / 5090 | 16 | 8 | 8 | 2 | 0 | 7 |
| TileLang / H100 | 16 | 7 | 9 | 5 | 1 | 6 |
| **合计** | **161** | **103** | **58** | **27** | **11** | **83** |

该表直接否定了“性能已经全部站住”的说法，但也说明 generic 主路径没有整体塌陷：Triton dense GEMM
两机约 `1.00x`，normalization/elementwise 大量行在 `1.05x` 内；TileLang dense/W4A8/FP8 GEMM 与
cuTile 多个 dense/grouped/batched contraction 也能接近或优于对应 source。

### 6.2 Triton 尚未闭合的性能结构

主要差距：

- flash-attention forward：5090 `2.31x`，H100 `2.39x`；
- padded RoPE cache update：`1.79x / 1.72x`；
- paged GQA decode：`1.39x / 2.58x`；
- paged MLA decode：`13.93x / 15.72x`；
- scaled FP8 split-K GEMM：5090 `1.70x`，H100 `0.94x`。

generated paged attention 仍使用通用 page/ragged stream，source 显式组织 head grouping、page traversal、
split/workspace 和 decode schedule；缺的是 Triton provider-local paged-attention form。flash source 的
FA-v2 block/warp/pipeline specialization 也没有由当前 provider pass 形成。两者都是编译器性能缺口，
不是 terminal 字符串拼写问题，更不能用 kernel matcher 替换成手写模板。

FP8 split-K 跨设备反转，表明它不是一条跨设备写死常量能解决的 shared rule；合法 form/参数应继续
交给 provider tuner。`qkv_projection_pipeline` 的 generated/source decomposition 并非完全相同，
其 ratio 只保留为 pipeline 观察，不作为单 kernel 性能结论。

### 6.3 cuTile 尚未闭合的性能结构

数量级差距集中在：

- absorbed MLA decode：5090 `12.40x`，H100 `101.08x`；
- split-K MLA decode：`9.86x / 63.19x`；
- sliding-window attention：`6.09x / 5.92x`；
- H100 Gemma prefill：`9.64x`；
- dense/MLA/sink attention 与 MHC 若干行：约 `1.6x` 到 `5.1x`。

generated 仍是通用 state-stream/ragged traversal，source 使用 cuTile/TileGym 的 native decode、sink、
window、Gemma 和 persistent organization。这里缺 provider-local access/packing/persistent/MMA form；
不能把这些结构塞进 shared IR，也不能给每个 attention 名字加分支。

两处 baseline 需要谨慎解释：

- `official_fmha` source 在 5090 为约 33 ms、H100 为约 2.67 ms，同一 source 固定 form 跨设备反转；
  5090 上 generated 的 `0.18x` 不能当成普遍 compiler 优势；
- H100 `dense_gemm` source 约 13.3 ms，而同表另一个 TileGym dense GEMM source 处于正常毫秒量级；
  该 apparent win 证明 entry 可运行，但不能证明 source 已代表该设备最优实现。

这两类异常在旧固定表和本次全量中重复出现，已经排除单次测量抖动；它们被保留而没有用别的数字
覆盖。

### 6.4 TileLang 尚未闭合的性能结构

主要差距：

- dense flash attention：`1.45x / 1.28x`；
- varlen GQA prefill：`1.66x / 2.50x`；
- block-causal attention：`3.34x / 5.28x`；
- per-token FP8：`1.38x / 2.51x`；
- MHC post：`1.62x / 2.94x`；
- GQA attention backward：`3.16x / 2.00x`。

TileLang source 显式选择 shared/fragment storage、bulk copy、block M/N、threads、stages 和 pipeline；当前
provider passes 尚未形成完全等价的 native form。GQA backward 的比较还有 decomposition 限制：
generated 计时 delta、dK/dV、dQ 三个作者调用，source 是另一套 preprocess/split/postprocess pipeline，
该 ratio 不能全部归因于一个 target op。

## 7. 剩余能力与资源边界

### 7.1 shared Physical Program：5 个结构

两机都稳定停在 `physical_program_failed`：

- `varlen_gqa_decode_logits`；
- `conv2d`；
- `linear_attention_forward`；
- `retention_forward`；
- `varlen_block_causal_attention`。

它们分别暴露 runtime-bounded ordinary sequential loop、tensor loop-carried/multi-result axis、同一 logical
axis 的 incompatible contraction roles、以及 ragged member + state-stream 的组合合法性。这些是 shared
Physical Program 还没闭合，不是 TileLang API 做不到，也不是应该扩 public DSL 的证据。

### 7.2 TileLang provider program：6 个结构

- `paged_mla_decode`、`native_sparse_attention_forward/decode`：没有 native single-row contraction form；
- `deepgemm_fp8_2xacc`、`fp8_lighting_indexer`：FP8 runtime-lane MMA form 缺失；
- `grouped_gemm_backward`：ragged reduction 没有机械的 masked bulk-copy projection。

这些在 terminal source 前明确拒绝。没有保留串行 scalar product/reduce 或逐元素 ragged copy 的慢路径。
前五项是当前 TileLang 0.1.13 capability/form 子集；最后一项是 provider materialization 尚未闭合。

### 7.3 provider 编译/JIT 成本与能力

- cuTile `sparse_mla_prefill`、`recurrent_gated_delta`：两机每个候选都超过 TileIR 单候选 5 秒编译上限，
  属于生成形态/下层编译成本，不是算法表达失败；
- TileLang `block_sparse_gqa_decode`、`gqa_decode`：provider program 可生成，但下层 MMA/layout/JIT 没有
  合法候选；
- TileLang `mhc_pre`：两机首次编译超过 300 秒，记为 `worker_timeout`；
- cuTile `block_scaled_gemm`：5090 通过，H100 的 SM90 不具备 source/leaf 所需的 SM100 E8M0 scaled MMA；
- Triton `block_sparse_gqa_decode`：H100 通过，5090 候选需 133120 B shared memory，超过 101376 B。

### 7.4 source/adapter 边界

- 5090 `modern_flash_attention_forward` 与 `flash_attention_backward` 的 Meta source 准备核要求
  163840 B shared memory，超过设备 101376 B；generated 侧不能因此得到公平 ratio；
- H100 `flash_attention_backward` 的 generated 三阶段已完成首次运行，随后 upstream source backward
  JIT 失败，最终状态专门记为 `source_provider_jit_or_initial_launch_failed`。

这些状态不应被写成 Intent 语言或 Physical Program 不支持。

## 8. 架构与冗余路径核验

本轮修复没有新增以下禁止形态：

- 没有按 kernel 名称、op 数量或 whole-kernel shape matcher 分支；
- 没有用手写 attention 模板替换生成结果；
- 没有把运行时 extent、stream provenance 或 candidate legality 在三个 leaf 各判断一遍；
- 没有恢复 extent/name 猜轴、串行 contraction、逐元素 ragged copy 或 legacy fallback；
- 没有为 5090/H100 写架构型号分支；静态/运行时容量与 extent 通过 typed metadata 进入已有 provider
  tuner/capability 路径；
- 没有把 provider 参数搜索改成 Intent cost model。

合法的厚 leaf 代码继续保留：它们读取已选 Plan/provider form，负责目标 ABI、allocation/copy 语法和
原生 primitive 调用。cuTile gather 的多个 spelling 是两台设备会选出不同 winner 的 provider-local
候选，不是历史冗余路径。

## 9. 可作为下一阶段起点的最终状态

已经真正做完并由本轮全量支撑的核心：

1. 一条统一的 DSL -> KIR -> Physical Program -> shared passes -> provider passes -> terminal source
   编译链；
2. 三个 provider、两台不同 GPU、112 个 provider entries 的真实 source/runtime inventory；
3. 161 个跨设备数值与稳态测量通过行，且无静默数值失败；
4. 失败阶段可以区分算法/语言、shared Physical Program、provider program、provider JIT、设备资源、
   source/adapter；
5. generic contraction、normalization、elementwise、部分 ragged/scan/attention 已证明能自然从同一机制
   长出，而不是一 kernel 一路径；
6. axis decision、region provenance、stream scope 与 tuner candidate legality 已各有唯一权威来源。

仍未关闭、不能写成“成熟支持”的边界：

1. 13 个尚未形成等价 Intent 算法或 packed storage/intrinsic 合同的 inventory entries；
2. 5 个 shared Physical Program 组合与 6 个 TileLang provider forms；
3. cuTile 两项稳定编译成本失败和 TileLang `mhc_pre` 五分钟以上首次编译；
4. paged/MLA/复杂 attention/MHC 等结构的显著 provider-native form 性能差距；
5. 58 个通过行仍超过 source 的 5%，其中部分 baseline 还存在 decomposition 或设备适配解释限制。

这份报告和六张 CSV 共同定义当前最终快照：后续问题应落到上述明确层次处理，而不再借性能或新
baseline 之名重开编程模型、复制物理事实，或在 terminal emitter 中建立第二套编译器。
