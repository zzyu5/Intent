# Compiler Pass V2 至 baseline-new：五轮推进总报告

## 0. 报告定位

这份报告统一收敛从“按 Pass V2 重构编译器”开始的五轮工作：

1. 按 `compiler-pass-v2.md` 重构编译主链；
2. 在新结构下关闭 baseline-new 接线越权与编程模型边界问题；
3. 用既有和新 kernel 检验新结构的接纳能力；
4. 在 RTX 5090D 与 H100 上完成三 provider 的 baseline-new 全量；
5. 根据全量暴露出的性能差距，修复共享决策和 target projection 中已经定位清楚的问题。

证据范围是提交 `5bfd69a` 至 `9e284a5`。其中：

- `5bfd69a` 是 V2 目标架构的设计基线；
- `22b39d2`、`458493d`、`a19f911` 是第一轮主体重构；
- `6bc7a61` 发布第四轮双机六张 CSV；
- `9e284a5` 是第五轮性能修复；
- 第五轮只对受影响项做了定向数值与性能 A/B，没有重新跑六张全量表。因此 `report/baseline-new/*.csv` 是第四轮固定快照，第五轮最新结果只在本报告中单独列出，不能把二者混成同一次全量。

本报告不修改正式规格 `doc/`，也不把实验状态写进设计文档。`report/baseline-new/` 只保留六张 CSV；原先放在该目录中的 construction、边界审计和全量过程说明已经吸收到本文。

---

## 1. 五轮结论总览

| 轮次 | 核心问题 | 主要结果 | 验证范围 |
|---|---|---|---|
| 第一轮 | V1 的 Kernel IR 与旁表 Plan 共同承担执行权威，leaf 现场重建程序 | 建立 `ConstructPhysicalProgramPass → VerifyPhysicalProgramPass → MaterializeTargetProgramPass → terminal translator` 唯一路径 | 4 条代表 repro；attention backward 用旧提交同机 A/B 排除重构回归 |
| 第二轮 | baseline 接线曾混入语言、Plan、leaf 和 DSL 算法修改 | replay 成为显式 selected decision；`partition` 收敛到 fixed/auto extent；adapter failure 与 compiler failure 分离 | split-K attention reduce、Mamba3 SISO step 定向运行 |
| 第三轮 | 检验重构后是否还能接纳不同 axis、range、ragged、state、contraction 结构 | 补齐 result-axis index group、planned region range、静态 reshape provenance、TileLang bounds/constant binding 等共享或 leaf 缺口；加入对齐的 causal Conv1D acceptance entry | 只跑新增和受影响 kernel；两机复验 reshape/native-sparse 暴露面 |
| 第四轮 | 前三轮从未经过两机三 provider 全量 | 产出 6 张表、182 行、每行都有同语言公开 source；147 行完整 pass，35 行 compile_failed | 5090D/H100 并行执行全部 registry entry |
| 第五轮 | 全量显示若干结构性投影差距和跨 provider 离散 | 补 loop-carried accumulator flow、compact quasi-affine coverage、one-sided deferred contraction、persistent range、scalar lane packing 和 TileLang 正常 profile 搜索 | 两机定向数值/性能 A/B；无全量回填 |

这五轮没有把 baseline-new 当成编译器版本。baseline-new 只是新的性能对照集合；编程模型、Kernel IR 和编译器主链只在有独立算法或架构依据时修改。

---

## 2. 第一轮：按 Pass V2 重构编译器

### 2.1 起点问题

重构前的真实路径是：

```text
canonical Kernel IR
  → 分析 Kernel IR，构造 KIR-referenced Plan records
  → provider emitter 再遍历 Kernel IR
  → emitter 结合 Plan 和自己的局部判断构造 target source
```

问题不是 Plan 字段数量少，而是当前程序没有唯一位置：

- Kernel IR 保存完整算法、control、SSA 和 effect；
- Plan 保存 axis/range/residency/transfer/stage 等已选决定；
- leaf 同时读取两者，并现场补齐 executable control、copy、loop、workspace 和 structured primitive；
- terminal emission 与 provider program construction 混在同一入口。

这种结构会持续诱发两类错误：一是 leaf 从 shape、角色名或邻近 op 重建上一层已经决定的事实；二是同一件事在三个 leaf 中分别决定，形成三套隐式编译器。

### 2.2 重构后的唯一主链

第一轮把主链改为：

```text
canonical Kernel IR
  → ConstructPhysicalProgramPass
  → intent_plan.program 中唯一 intent.kind="physical" function
  → MLIR verifier
  → VerifyPhysicalProgramPass
  → MaterializeTargetProgramPass
  → intent_plan.target_program { provider, source }
  → terminal translator
  → Triton / cuTile / TileLang source
```

关键不变量是：进入 physical-program 阶段后，module 顶层不再保留另一份可执行 canonical function。当前 physical function 被移入 `intent_plan.program`，后续代码只能通过 program 取得它。Kernel node/value ID 继续用于 provenance 和 decision binding，但不再指向另一份并存的 executable kernel。

### 2.3 Physical Program 的承载内容

`intent_plan.program` 当前拥有：

- 唯一 physical function；
- device 与 launch mapping；
- axis、range、region、stream 绑定；
- buffer residency、padding、transfer；
- reduce、scan、contract、sparse contract、stage decisions；
- 可选的唯一 provider `target_program`；
- 与 executable authority 分离的 `search_space` metadata。

本轮采用 partial conversion，而不是复制一套与 Intent op 一一对应的新 dialect。已经具有合法 physical semantics 的 Intent SSA、control 和 structured op 保留在 current physical function 中；selected physical decisions 作为同一 program 内的 typed operations 存在。

这解决了“双份 executable authority”，但不等于成熟 V2 的全部目标已经完成。当前 provider materializer 仍遍历 physical Intent operations 并直接构造 source 字符串；尚未形成可由多个 provider passes 继续变换的完整 provider-legal SSA IR。这个边界在第七节单独说明。

### 2.4 Pass 与目录边界

第一轮完成了三项结构拆分：

1. `ConstructPhysicalProgramPass`
   - 验证 canonical Kernel IR；
   - 建立 pass-local `KernelModel`/`KernelFacts`；
   - 执行 axis/range/stage、residency、padding、transfer 和 structured-op policies；
   - 产出完整 physical program。
2. `VerifyPhysicalProgramPass`
   - 验证 program、device、launch、range、workspace、stream、stage 和 structured decisions；
   - 不接受缺失 binding 的 partial state。
3. `MaterializeTargetProgramPass`
   - 建立 pass-local `PhysicalProgramIndex`；
   - 调用 provider 自己的 `ProgramMaterializer`；
   - 写入唯一 `target_program`；
   - 再次验证组合合法性。

目标目录从模糊的 `Emission/` 收敛为：

```text
lib/Target/<Provider>/Lowering/
  Driver/
  Handlers/
  Materialization/
  Support/
  Syntax/
```

共享层相应使用 `Target/Common/Lowering/`。目录现在表达 provider materialization 与 terminal translation 的边界，而不是把所有工作都称为 emitter。

### 2.5 删除的旧路径

第一轮没有保留兼容开关或 fallback。以下旧结构被删除或替代：

- `intent_plan.realization` / `RealizationOp`；
- 旧 launch record `intent_plan.program`，由新的 program container 与 `intent_plan.launch` 分担职责；
- `realizeKernel` 与 `Realization/Driver/Realize.cpp`；
- `emitTargetSource`、`emitTritonSource`、`emitCuTileSource`、`emitTileLangSource`；
- `TargetSourceEmitter`、provider `SourceEmitter`；
- `SurfacePlan`；
- 三个 provider 和 common 的 `Emission/` 目录；
- legacy/new pipeline 开关与过渡入口。

`SearchSpaceOp` 被保留，因为它声明的是下层 tuner 的合法数值参数，不是另一份 executable program。

### 2.6 验证与提交

本轮按真实影响面运行：

| repro | 覆盖点 | 结果 |
|---|---|---|
| `triton value_select` | physical SSA/value 与 terminal translation | PASS，p50 约 0.0774 ms |
| `cutile value_select` | cuTile provider materialization | PASS，p50 约 0.0895 ms |
| `tilelang matrix_transpose` | access coverage 与 TileLang buffer/copy | PASS，p50 约 0.0916 ms |
| `triton layer_norm_backward` | multi-stage/workspace/structured path | PASS，p50 约 0.0761 ms |

`attention_backward` 当时能编译运行，但数值 reference 报错。旧提交 `5bfd69a` 与新结构在同机独立 build root 下得到完全相同的 `(dQ,dK,dV)` 误差，因此没有把既有问题伪装成重构回归，也没有修改 reference 或容差制造 PASS。

对应提交：

- `22b39d2 compiler: make physical program the executable authority`
- `458493d compiler: run physical construction as verified passes`
- `a19f911 compiler: separate provider materialization from translation`
- `7c29c3f report: document compiler pass refactor`

---

## 3. 第二轮：继续 baseline-new 重构并关闭越权

### 3.1 本轮首先重新划线

baseline-new adapter 接不上，不自动意味着语言或编译器能力缺失。正确顺序被固定为：

```text
公开 source 的真实算法
  → DSL 是否已经能表达同一算法
  → canonical KIR 是否保留作者语义
  → derived fact 是否能从 KIR 重算
  → Plan 是否只保存已选物理决定
  → leaf 是否只做 capability / target form / materialization
  → runtime adapter 与公平计时
```

只有真实算法无法用已有 Core 表达，才允许单独讨论语言能力；只有存在多个合法物理答案、且 leaf 不应重选时，才允许增加 Plan decision。

### 3.2 contraction replay 从反推改为显式选择

producer-chain contraction replay 会改变生成程序的结构，不能由 KIR 唯一推出。旧路径只留下 shared residency 和 deferred transfer，common materialization 再从这些结果反推是否“选择过 replay”。

本轮在 `intent_plan.contract` 中加入显式 `producer_replay` selected decision：

- GPU construction 唯一选择并写入；
- verifier 检查 replay 与 operand residency 的合法组合；
- producer/transfers/exclusive-owner 列表仍从 KIR def-use 重算，不复制到 Plan；
- 三个 leaf 消费同一 common derived index；
- selected replay 无法解析出 canonical chain 时直接诊断。

这里区分了两类对象：是否 replay 是物理选择；具体 producer node 列表是算法 def-use 的派生索引。

### 3.3 `partition` 收敛到当前真正可 lowering 的 Core

审计时，examples 中 117 个 `I.partition(...)` 调用只有两种实际形态：

- 109 个 `extent=I.auto(...)`；
- 8 个正的 compile-time fixed extent；
- 0 个 `count=`；
- 0 个 runtime scalar extent。

public lowering 合同因此收敛为：

```text
I.partition(axis, extent=<positive constexpr integer | I.auto(name)>)
```

具体处理：

- 删除不可达的 `PartitionMode.COUNT` 与 part-ordinal frontend 分支；
- frontend 直接、带源码位置拒绝 runtime scalar extent；
- fixed extent 要求正整数；
- `auto` 只授权 Realizer 选择具体 physical extent，不授权插入新的 partition；
- 不把 `count=P` 偷换成 extent，因为 count 会暴露 part identity，并可能改变 partial ABI 和多调用编排。

这是删掉名义存在但从未闭环的假能力，不是为了某个 baseline 改写算法。

### 3.4 语言能力复核

`sin`、`cos`、`floor` 被保留，但不再与 replay 或 adapter 混成一个理由：

- RoPE 与 Mamba3 rotation 直接需要 `sin/cos`；
- Mamba3 upstream 的角度 wrap 直接使用 `floor`；
- 三个 target 都有原生 spelling；
- 它们沿现有 canonical unary op lowering，不增加 Plan fact。

相反，不能仅因 source 文件使用了某种写法，就顺带扩大语言、修改 numerical path 或增加 target-specific shared decision。

### 3.5 runner 错误边界

旧 runner 会把 provider factory 内任意异常统一写成 `compile_failed`，而 factory 同时包含 generated compile、source import/JIT、ABI adapter 和 workspace 准备。

本轮引入明确边界：

- 只有 generated compile/run/launcher preparation 的 `GeneratedCompilationError` 写 `compile_failed`；
- 数值树比较失败写 `numerical_failed`；
- source import、source JIT、adapter 和 runtime 异常保留原始异常，不伪装成 compiler failure；
- 不增加含义模糊的 `adapter_failed` 状态。

### 3.6 定向验证

| entry | 覆盖点 | generated/source |
|---|---|---:|
| cuTile `splitk_attention_reduce` | producer replay + auto partition | 0.005344 / 0.006144 ms，数值 PASS |
| Triton `mamba3_siso_step` | fixed partition + floor/sin/cos | 0.106432 / 0.095824 ms，数值 PASS |

对应提交：

- `5a30095 compiler: make contraction replay an explicit plan choice`
- `0f4381e frontend: narrow partition to realizable extents`
- `78c3d92 baseline: preserve adapter failure boundaries`
- `7d80159 report: close baseline-new refactor boundary audit`

---

## 4. 第三轮：检验新结构的 kernel 接纳能力

### 4.1 接纳原则

这一轮既使用旧 baseline 中已经跑通过的结构，也使用 source 中尚未完整进入新矩阵的结构。选择顺序始终是先固定真实算法，再看编译器断在哪；没有为了让新架构显得通用而改写 source 算法。

失败被区分为：

1. canonical 语言或 lowering 的真实能力缺口；
2. leaf 又从 shape/邻近结构重建本该从 Plan 或 common analysis 读取的事实；
3. target surface 或下层工具链确实没有可接受实现。

### 4.2 result-axis index group

复杂 reshape、broadcast 和 indexed tensor 暴露出一个共同问题：三个 provider 需要的是“某个结果轴对应哪些 source index components”，旧代码却在 leaf 中分别从输入关系重建。

`add9ea0` 把 result-axis index-group projection 收敛到 `Target/Common/Analysis/IndexRelation`，三个 materializer 读取同一派生关系。它没有把可从 KIR 唯一推出的 index group 写进 Plan，也没有为某个 kernel 建 handler。

### 4.3 causal Conv1D acceptance entry

`bd96703` 增加与公开 source 算法边界对齐的 causal depthwise Conv1D entry，并接入 Triton baseline-new registry。它用于区分：已有 convolution Core 是否能表达真实 causal update；如果断裂，问题发生在算法表达、range/access realization 还是 provider leaf。

这个修改没有把同 padding Conv1D 改名冒充 causal，也没有用不相同的 source scope 填数字。

### 4.4 TileLang planned region range

TileLang 某些消费者仍从逻辑 extent 或 shape 推导 target range，而 Plan 已经保存最终 selected region range。`ad6bf2c` 将这些路径改为消费 planned range，并删除相应旧缓存字段。

这类修复的判断标准是：如果 leaf 从 Plan 已有的 selected range 以外再选一次范围，就是假发射；如果它只是把已选范围拼成 `T.Parallel`、slice 或 buffer shape，则是合法 materialization。

### 4.5 全量前暴露并关闭的共享缺口

`1b48674` 集中关闭了 baseline-new acceptance 阶段真正撞到的实现缺口：

- 静态 reshape 先允许局部正 extent 占位，最终必须沿 source use-def 和精确元素乘积恢复 provenance；没有恢复“相同 extent 猜一根轴”的 fallback；
- TileLang `assume_in_bounds` 将“是否支持该假设”与“是否需要 rank-1 scalarization”分开；
- entry-block constant 即使不打印语句，也必须形成 SSA binding；
- row-vector 的所有消费者统一读取最终 selected physical range；
- tensor loop-carried contraction 的结构识别收敛为 common derived query，不把 KIR 可重算结构复制进 Plan；
- fully-static kernel 可以声明 tunable parameters 而没有 specialization dimension；
- ragged contraction 没有 masked bulk-copy 投影时提前拒绝，不生成忽略 group offset 的静默错误；
- MHC、Mamba、gated-delta adapter 的算法范围、数值输入与 timed scope 被重新对齐。

### 4.6 定向验证与暴露面

共享 reshape/provenance 修复后，两台机器都重新执行：

- Triton `rope_qk`；
- cuTile `rope_qk`；
- TileLang `w4a8_gemm`；
- native-sparse forward/decode。

前三项数值通过；native-sparse 两机稳定到达相同的 single-row contraction 显式诊断。后者被保留为 target capability boundary，没有用串行归约冒充支持。

对应提交：

- `add9ea0 compiler: project tensor index groups at result axes`
- `bd96703 baseline: add aligned causal conv1d acceptance kernel`
- `ad6bf2c compiler: honor planned region ranges in tilelang`
- `1b48674 compiler: close baseline-new lowering gaps`

---

## 5. 第四轮：双机全量并产出 baseline-new

### 5.1 表格合同

baseline-new 的一行是 runtime-visible algorithm entry：有明确输入输出、可独立调用、可独立计时。以下不单独计数：

- 内部 helper；
- autotune config 或同算法不同 tile；
- reference/packing/import support；
- `variant_*` 等价 DSL 拼写；
- PyTorch composition、临时 reference、手写 CUDA；
- 没有可执行 runtime 的源码快照。

合格 source 必须：

1. 公开且以性能为目标维护；
2. 与表所属 provider 相同；
3. 与 DSL 在算法、输入输出、调用次数和 timed scope 上一致；
4. 原始 source 不改算法，ABI 适配位于相邻 runtime；
5. 能在目标设备当前环境真实编译运行。

### 5.2 模型级输入与计时

固定 case 使用模型级规模，而不是小 tile：

- GEMM/FFN 使用典型 hidden/intermediate/token 规模；
- attention prefill 使用 2K/4K 级 sequence、D128 与真实 head 数；
- decode 的 query 可为 1，但 KV cache 是 4K/8K/16K 级；
- MoE 使用真实 expert、top-k 和 token 数；
- normalization/pointwise 覆盖完整 token×hidden；
- quantization 对完整矩阵计时。

统一计时规则：

- JIT、autotune、输入生成和 output/workspace 分配在计时外；
- timed region 只含用户得到结果必须执行的 GPU work；
- generated 与 source 的算法、launch 数和 scope 一致；
- sub-ms kernel 双方都用 CUDA Graph，并按设备 L2 容量冲刷；
- 非算法 ABI packing 在计时外；算法本身所需 preprocessing 在双方同侧；
- 多 kernel pipeline 只与相同 pipeline 比；
- 数值对照先通过，再记录性能。

CSV 统一为：

```csv
kernel,case,generated_p50_ms,source_p50_ms,ratio,status
```

provider 和 device 由文件名编码，不再保留重复的 `triton_scope` 或 `algorithm_group`。

### 5.3 两机执行方式

RTX 5090D 与 H100 的全量流程并行启动，互不等待。每个 entry 都执行：

```text
DSL
  → canonical Kernel IR
  → Physical Program
  → target source
  → provider compiler/autotuner
  → GPU launch
  → numerical comparison
  → CUDA timing
```

某个 source 异常或首次编译过长导致主 runner 中断时，后续 entry 使用独立进程继续，但仍走同一完整链路。300 秒内不能得到 executable candidate 的行记录 `compile_failed`，不填写伪造的性能数字。

### 5.4 六张固定表

第四轮发布：

- `report/baseline-new/triton-5090.csv`
- `report/baseline-new/triton-h100.csv`
- `report/baseline-new/cutile-5090.csv`
- `report/baseline-new/cutile-h100.csv`
- `report/baseline-new/tilelang-5090.csv`
- `report/baseline-new/tilelang-h100.csv`

固定快照统计：

| provider / device | entry | pass | compile_failed | pass 中 ≤1.05× | pass 中 >1.05× |
|---|---:|---:|---:|---:|---:|
| Triton / 5090 | 31 | 29 | 2 | 18 | 11 |
| Triton / H100 | 31 | 30 | 1 | 22 | 8 |
| cuTile / 5090 | 30 | 30 | 0 | 20 | 10 |
| cuTile / H100 | 30 | 28 | 2 | 11 | 17 |
| TileLang / 5090 | 30 | 15 | 15 | 8 | 7 |
| TileLang / H100 | 30 | 15 | 15 | 6 | 9 |
| 合计 | 182 | 147 | 35 | 85 | 62 |

六张表的 182 行都能通过 `examples/repro/v2/registry.py` 映射到 README inventory 中同 provider 的非空 source runtime，所以表内 source 覆盖率是 100%。`source_runtime` 不是 CSV 列，而是 registry 与 source README 的独立 provenance。这不等于完整 source inventory 已全部接入：README inventory 为 Triton 41、cuTile 39、TileLang 39，仍有 10/9/9 个公开 runtime 尚未进入 registry。

### 5.5 失败边界

Triton：

- 5090 的 `flash_attention_backward` 是 source shared-memory 超限；
- 5090 的 `block_sparse_gqa_decode` 是 generated candidate shared-memory 超限；
- H100 的 `flash_attention_backward` 是当前 source/toolchain 的 dot dtype mismatch。

cuTile：

- 5090 为 30/30；
- H100 `block_scaled_gemm` 依赖 SM100 E8M0 scaled MMA，SM90 不具备该能力；
- H100 `sparse_mla_prefill` 的 tileiras candidate 全部在当前编译上限内失败。

TileLang：

- 两台设备是同一组 15 pass / 15 compile_failed；
- 失败集中在 single-row contraction、tensor loop-carried accumulator、runtime-lane FP8 MMA、ragged reduction bulk-copy、block-sparse GQA packing、runtime-bounded scalar traversal 和部分复杂 layout；
- 没有使用慢一两个数量级的串行 fallback 冒充支持。

### 5.6 全量暴露的性能问题

第四轮的重要意义不是把表填满，而是第一次暴露“三家都慢”以及同一 Plan 在不同 leaf 上投影质量差异：

- Triton rotary mapping、Mamba head-block、FlashAttention block/warp specialization；
- Triton/cuTile MLA decode 与 source split-KV 多阶段算法分解不一致，ratio 不能当同结构差距；
- cuTile absorbed/split-K MLA、gated delta、MHC 缺 stage、packing 和 persistent topology；
- TileLang block-sparse、W4A8、varlen GQA、Mamba 在 target-local materialization 上明显落后。

这批问题成为第五轮的输入。对应发布提交：

- `6bc7a61 baseline: publish cross-device baseline-new matrix`

---

## 6. 第五轮：修复已经定位的性能问题

### 6.1 修复原则

第五轮没有按 kernel 名选择手写模板，也没有为 H100/5090 增加架构型号分支。每项修改必须先回答：

- KIR 作者是否已经写下这个算法事实；
- 这是唯一 correctness fact、多个合法结构选择，还是下层可调参数；
- shared Plan 是否已经保存，leaf 是否又重建了一次；
- target 是否有原生 primitive/spelling；
- 修改能否由 axis、def-use、index relation、validity、state 或 structured op 的 typed facts 触发。

不能用这套条件说明的实验，即使某一格变快，也不进入最终代码。

### 6.2 persistent mapping 的完整范围

shared decision 在计算 persistent program reuse 时，曾只统计局部 partition，而没有保留完整 logical range 的 program-count 约束。修复把 full-range partition accounting 放回 GPU `Decisions.cpp`，没有用“遇到 state_stream 就禁用 persistent”这种全局门槛。

### 6.3 block-sparse contraction 与 accumulator flow

公开 TileLang source 会先读取 block mask，只在 block enabled 时执行 copy/GEMM。DSL 被对齐为相同控制流，不再先执行无效 contraction 再在结果上 mask。这是算法对齐，不是 compiler 自动改写 KIR。

真正的编译器缺口是：`contract → add(previous accumulator) → yield` 的 loop-carried flow 过去只在 leaf 附近结构中隐含存在。

修复后：

- shared Build 从 KIR def-use 严格识别 accumulator flow；
- `intent_plan.contract` 保存 `accumulator_flow=loop_carried` 以及 owner/update/value/conditional node bindings；
- verifier 检查 flow 字段成组存在；
- TileLang 在 target 支持时复用 accumulator fragment；
- Triton/cuTile 继续可以用 fresh contraction result + SSA add 保持同一语义；
- leaf 不能证明 planned conditional accumulator 时直接失败，不静默降级。

Plan 保存的是 flow binding 与 materialization obligation，不是复制一份 combine 算法。

### 6.4 W4A8 compact quasi-affine coverage

packed weight 的 `k // 2` 曾被压成普通 data-dependent indexing，导致 TileLang 逐个 logical-K 元素搬运，丢失“一个来源轴、正常数除数、连续 compact physical span”的事实。

修复包括：

- Transfer Plan 增加 `coverage_space`；
- compact quasi-affine coverage 限定为可证明的单-source floor-divide 物理跨度；
- 先搬运 compact physical coverage，再机械展开 logical view；
- producer replay 只接受 compact tensor indexing 的受限链；
- 三个 leaf 支持 one-sided deferred stream contraction；
- W4A8 DSL 与 source 对齐为 signed weight × activation 的同 orientation，删除额外 output transpose。

这不是 W4A8 matcher；触发条件是 index relation、coverage、stream/reduction role 和 contraction def-use。

### 6.5 scalar lane packing

纯 pointwise/scalar-domain kernel 在没有 inner lane role 时，旧 mapping 可能让每个 logical scalar instance 启动一个 program。

修复将 packable independent scalar axis 绑定为 physical lane，并在 Triton target 中统一处理：

- emitted rank；
- pointer tensor；
- load/store mask；
- validity；
- scalar insertion axis。

主体包含 contraction/reduction/scan 等 region-sensitive structured operation 时不自动做这种 promotion，避免把作者写的 scalar algorithm 改成块算法。

### 6.6 TileLang 联合参数进入正常 profile

stream/reduction/query/program-M 的组合参数不再由一个 post-hoc 角色集合 matcher 强制覆盖。候选进入普通 profile family，和其它 profile 一样经过 role intersection、equal-role/divisor legality 和去重；GEMM warp policy 继续作为 target-local 参数交给 TileLang tuner。

### 6.7 定向性能结果

以下是 `9e284a5` 后的定向 A/B，不是第四轮 CSV 的全量回填。这些数字来自本轮交互式 repro 运行记录，没有另存为 CSV 或独立原始日志；它们用于确认受影响路径和判断改动取舍，证据强度不同于第 5.4 节的仓库固定快照。

| entry / provider | RTX 5090D generated/source | H100 generated/source | 当前结论 |
|---|---:|---:|---|
| TileLang block-sparse GEMM | 0.333 / 0.383 ms | 0.320 / 0.336 ms | 两机均追平或快于 source |
| TileLang W4A8 | 1.154 / 1.391 ms | 1.377 / 1.715 ms | compact coverage 与 deferred contraction 闭合 |
| TileLang dense attention | 2.879 / 3.709 ms | 2.607 / 3.537 ms | shared flow/stream materialization 不退化 |
| TileLang varlen GQA prefill | 约 6.33 / 6.36 ms | 6.320 / 5.343 ms | 5090 持平；H100 仍慢约 18% |
| TileLang Mamba chunk scan | 0.0336 / 0.0323 ms | 0.0545 / 0.0271 ms | 5090 接近；H100 仍约 2.0× |
| Triton rotary embedding | 0.1818 / 0.1802 ms | 未在本轮重填 | 5090 已在 1% 内 |
| Triton Mamba chunk scan | 0.0517 / 0.0471 ms | 未在本轮重填 | 仍约 9.8% 差距 |

cuTile official FMHA 的 generated/source ratio 很小，但 source 固定配置在 5090 上明显失配，不能把该数字解释成 Intent 的普遍性能优势。

### 6.8 经实测否决并回退的方向

| 尝试 | 结果 | 处理 |
|---|---|---|
| 对所有 state-stream 禁用 persistent mapping | 牺牲其它合法结构，不能解释具体差距 | 回退，修 full-range accounting |
| Mamba 的 dA hoist、固定 reduction K=128、group-M 扩大 | 没有形成跨设备稳定收益 | 回退 |
| fragment×fragment in-place 等 target-local强行复用 | 不能由当前 Plan obligation普遍证明 | 回退 |
| generic stream-contract role 全笛卡尔积 | 产生 63 个候选，首次编译成本约 11 秒/候选 | 回退，改为正常 joint profile family |
| planned-validity 的动态 bulk fast path | H100 无明显收益 | 回退 |
| K tile 的 guarded bulk branch | 5090 从约 6.37 ms 退到约 7.49 ms | 回退 |

这些 A/B 说明剩余差距不是再放宽一个 boundary condition 或堆一个 tile 常数就能解决。

对应提交：

- `9e284a5 gpu: close structural projection performance gaps`

---

## 7. 当前实际架构与成熟 V2 的对应

### 7.1 每层持有什么

| 层 | 当前唯一职责 | 不能做的事 |
|---|---|---|
| canonical Kernel IR | 作者算法、logical workset、SSA/dataflow、state、effects、ABI、数值角色 | 保存 provider layout、warp、buffer scope 或目标语法 |
| pass-local analyses | provenance、axis relation、def-use、reuse、lifetime、validity proof、stage index | 序列化成第二份算法真理 |
| `intent_plan.program` | current physical function、已选 ownership/range/residency/transfer/structured/stage decisions | 按 kernel 名保存模板；复制可由 KIR 唯一重算的 producer list |
| provider materializer | capability、target form、buffer/copy/loop/primitive composition、wrapper construction | 重选 shared ownership、logical range、算法或 stage topology |
| `target_program` / translator | 持有并输出已经 materialized 的 provider source | 再遍历 physical function 或新增 tuner/workspace/mask |
| provider tuner | 在声明的合法参数面上实测 tile/warp/thread/stage/provider policy | 改变算法、ownership、ordered/state 或 stage grouping |
| 下层 compiler | layout、register、instruction、copy lowering、低层 pipeline 与 machine code | 恢复 Intent 没有传下来的算法结构 |

### 7.2 已经完成的 V2 核心

- physical program 是 compiler 中唯一 executable authority；
- construction 与 verification 进入显式 MLIR pass pipeline；
- provider materialization 与 terminal translation 分离；
- old realization/emission/SurfacePlan 路径已删除；
- Plan verifier 覆盖 program、launch、range、transfer、contract flow 等组合；
- target-local tuner 不构成第二份 Intent executable authority；
- 当前新增机制均由 op、axis、def-use、index relation 或 Plan 属性驱动，没有 kernel-name registry。

### 7.3 尚未达到成熟 V2 的地方

当前 `intent_plan.target_program` 主要保存最终 source 字符串。provider form selection、bufferization、operation handling 和 source construction虽然已位于独立 materialization pass，但尚未全部物化为可由后续 provider passes 继续观察、验证和替换的 provider-legal SSA IR。

所以可以准确地说：

- V1 的双执行权威已经消失；
- Physical Program 已成为 current executable program；
- terminal translator 已纯化；
- provider materialization 仍比成熟 V2 目标更厚，且一部分 target-form decision 仍存在于 C++ materializer 的结构中；
- 不能声称三个 leaf 已经只是逐 op printer，也不能声称完整 provider IR pass pipeline 已经完成。

这也是 H100 TileLang 剩余差距不能通过继续加 leaf 条件分支解决的原因：需要的是可审计的 target-form/layout/resource realization，而不是另一个隐藏 policy。

---

## 8. baseline-new 当前事实

### 8.1 目录状态

`report/baseline-new/` 现在只保留六张 CSV：

```text
cutile-5090.csv
cutile-h100.csv
tilelang-5090.csv
tilelang-h100.csv
triton-5090.csv
triton-h100.csv
```

旧 baseline 继续冻结在 `report/baseline/`，二者不是编译器版本，也不互相覆盖。

### 8.2 表内覆盖与完整 source inventory 的差别

六表当前共有 182 行；每一行都能由 registry 映射到可执行、同 provider source，这是 registry-defined 表内 source 100% 覆盖。CSV 本身不重复保存 source 路径。

但 source README 的 runtime-visible inventory 是：

| provider | source inventory | 当前 registry | 尚未进入 registry |
|---|---:|---:|---:|
| Triton | 41 | 31 | 10 |
| cuTile | 39 | 30 | 9 |
| TileLang | 39 | 30 | 9 |
| 合计 | 119 | 91 | 28 |

因此 baseline-new 已满足“每门语言至少 30 个、每行有真实 source”的第一阶段目标，但还没有清空完整 inventory。未进入 registry 的 source 只能标记为待算法对齐/adapter、重复候选或真实能力边界；不能未经审计就写成“不需要处理”。

### 8.3 CSV 的解释边界

- `ratio ≤ 1.05` 只表示该行计时，不自动证明算法和物理分解完全相同；
- source 使用 split-KV、多 kernel partial/finalize，而 generated 是单 kernel 时，只能做端到端结构参照；
- source 固定配置在某设备失配时，generated 更快不能被宣传为编译器普遍优势；
- `compile_failed` 必须继续保留，不能为了表格整齐删除；
- 第五轮定向修复没有全量回填，所以评估当前 HEAD 时要同时看六表和第 6.7 节。

---

## 9. 仍未关闭的问题

### 9.1 共享 Physical Program 结构

- split-KV / split-K 的 compiler-private 多阶段 program 与 intermediate 生命周期；
- query-head/KV-head 联合 packing、MLA latent/rope 分段；
- ordered/ragged stream 的上下界收紧和 reverse traversal；
- runtime-bounded domain 的合法 scalar sequential mapping；
- 复杂 state/FP8 contraction 中 reduction region identity；
- 能供 provider pass 消费的 target-form/layout/resource decision 位置。

### 9.2 TileLang target projection

- single-row contraction 的高质量机械投影；
- tensor loop-carried contraction accumulator 的更完整覆盖；
- runtime-lane FP8 MMA；
- ragged reduction masked bulk copy；
- block-sparse GQA 跨组 M packing；
- Mamba 所需的 shared swizzle、register policy 和多级 staging 如何由通用 target facts 表达；
- `contract → scale → mask` 如何在不重写 KIR 的前提下投影成 target accumulator initialization。

H100 上 varlen GQA 和 Mamba 的 exact tile/stage/thread 候选已经存在；剩余差距不能归因于 tuner 少了源实现参数。source 的差异集中在 mask placement、swizzled shared layout、`no_set_max_nreg` 和 staging/buffer topology。当前没有足够通用 typed facts 支持机械投影，因此没有加入 kernel 或架构特判。

### 9.3 下层/source 工具链

- Meta FlashAttention backward source 在两台设备当前工具链上分别有资源与 dtype failure；
- H100 cuTile sparse MLA 的候选编译失败；
- TileLang `mhc_pre` 首次编译成本过高；
- 部分 candidate 会崩溃或全部无效，需要隔离 worker 保证后续候选不被污染；
- provider source 固定配置本身可能跨设备失配。

### 9.4 非 GPU target

CPU、RISC-V 和 RVV 尚未接入。这不改变 Kernel IR 的 target-independent 定位，但说明当前只能声称 GPU physical-program pipeline 已经落地，不能声称跨机器族的 realizer 已经验证。

---

## 10. 提交序列

| commit | 内容 |
|---|---|
| `5bfd69a` | 定义 Compiler Pass V2 目标架构 |
| `22b39d2` | Physical Program 成为 executable authority |
| `458493d` | construction/verification 进入显式 pass pipeline |
| `a19f911` | provider materialization 与 terminal translation 分离 |
| `7c29c3f` | 记录第一轮实施结果 |
| `5a30095` | contraction replay 成为显式 selected Plan decision |
| `0f4381e` | `partition` 收敛到 fixed/auto extent |
| `78c3d92` | adapter/source failure 不再伪装 compiler failure |
| `7d80159` | 记录第二轮边界关闭 |
| `add9ea0` | result-axis index group 收敛到 common analysis |
| `bd96703` | 增加对齐的 causal Conv1D acceptance entry |
| `ad6bf2c` | TileLang 消费 planned region ranges |
| `1b48674` | 关闭 baseline-new acceptance lowering gaps |
| `6bc7a61` | 发布双机六张 baseline-new CSV |
| `9e284a5` | 修复结构性 projection 性能缺口 |

---

## 11. 最终结论

五轮之后，项目取得的不是“又有一套 baseline”，而是两项可以分开审计的结果。

第一，编译器主链完成了实质重构。Physical Program 已经取代“Kernel IR + 旁表 Plan + emitter 局部判断”的双执行权威；construction、verification、provider materialization 和 terminal translation 具有明确入口，旧路径已经删除。新增的 replay、range、coverage 和 accumulator flow 都进入 typed Plan 或 common derived analysis，而不是通过 kernel-name matcher 落在 leaf。

第二，baseline-new 给出了严格得多的性能证据。三门语言、两台设备、182 个 source 对照行证明了哪些路径能完整 lowering、编译、数值对照和计时，也暴露了 shared physical structure、target materialization 与下层工具链三种性质不同的问题。第五轮已经关闭 block-sparse、W4A8、dense attention、scalar packing 等有通用依据的差距；没有把 H100 TileLang varlen/Mamba 的残余用启发式或架构特判做绿。

当前最准确的状态是：GPU 编译主骨架已经切换到 Pass V2 的唯一权威路径，baseline-new 第一阶段矩阵已经完成；provider-legal IR 的进一步显式化、若干真实多阶段/packing 结构和 TileLang target-form realization 仍是明确未闭合项。它们已经被定位到具体层，而不是继续混在“某个 kernel 慢”或“某个 adapter 接不上”的模糊状态里。
