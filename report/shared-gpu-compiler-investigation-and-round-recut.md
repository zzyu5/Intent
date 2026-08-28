# Shared GPU Compiler 调查与第五至第八轮重新切分

## 0. 调查结论

原第五轮失败，不只是因为第三、四轮“少做了几个 structured op”。当前 shared GPU compiler 的问题更基础：

1. **第一份 GPU Program 并不完整。**`lowerCanonicalKIRToGPU` 的接口注释声称它产生 complete conservative program，但 `region_fold/region_scan` 要等后续 family pass 才变成 segment loop；最终 `verifyGPUProgram` 明确拒绝尚存的这两个 op。`runSharedGPUPasses` 中间只调用通用 `mlir::verify`，没有任何一步证明完整 executable invariants。
2. **shared facts 有 IR 载体，但没有统一 analysis authority。**`FragmentType`、`RangeType`、`AxisMapAttr`、access coordinates、validity 与 owner 是真实 current-IR facts；问题不是“什么都没有”，而是 dependence、access footprint、coordinate projection、lifetime、tail predicate 等被每个 family pass就地沿 producer graph 重算，没有共享 lattice、失效或保留边界。
3. **按 op family 切 pass 本身不是错误；同一 physical decision 在多个 family 内独立实现才是错误。**当前 zero/fill 有 4 份，tail/validity 有 6 个实现簇，ownership/program mapping 有约 5 个形成点，fragment replay 至少 5 处，access composition 分散在 4 层；ordinary/scaled contraction 还有高度同构的 mapping/blocking 代码。
4. **pipeline 顺序承重是正常的，假设“Triton pass 可随便重排”不成立。**Triton 和 TileLang 都有严格有序 pipeline。Intent 的真实缺口是没有明确的 analysis preservation/invalidation，以及前序 pass 会 erase/重建承载 provenance 的 IR，后序只能依赖形状门；不是“有顺序”本身。
5. **`operator_kind` 假设成立，而且已经是具体 correctness 问题。**规格明确区分 propagating `maximum/minimum` 与忽略单侧 NaN 的 `maximum_num/minimum_num`；当前 cuTile matcher 将 `7/9` 和 `8/10` 合并，而当前 cuTile 实现的 float min/max bytecode明确设置 `propagate_nan=False`。因此 canonical propagating maximum 会被静默改成 non-propagating maximum。
6. **Triton 51/54 不能证明 shared 层完整。**cuTile 和 TileLang 各只有 7 个 kernel 名与 Triton registry 重合、各有 30 个不重合；当前 cuTile 的 6 个 shared failures 和 TileLang 的 14 个 shared failures（2 个 `physical_program_failed` 加 12 个 `physical_program_verification_failed`）全部落在这 30 个 Triton 未覆盖 kernel 上。
7. **14+14 cuTile timeout 目前无法归因。**runner 只在 300 秒后写 `worker_timeout`，不记录 candidate 数、当前阶段或单 candidate 成本。`d580ba8` 只在 repro adapter 中过滤 generated candidates；它没有让 compiler 计算合法范围，而且 cuTile dense 的 generated 候选与固定 source config并不相同，所以不能把它当作 timeout 根因或公平性能闭合证据。
8. **旧链的高通过率是必须恢复的能力证据，但旧架构不能复用。**旧链靠 KernelModel、Plan side attrs 与厚 materializer覆盖 access/transfer、row/persistent、contract orientation、packed decode、stream/ragged 等结构；新链删除它是对的，但没有把这些 executable facts完整搬进 current IR/analysis/provider extensions。

因此第五至第八轮应重新切为：

```text
第五轮  重建 shared GPU compiler 的 analysis、typed semantics 与 pass authority；Triton 受影响路径不退化
第六轮  cuTile provider closure 与 timeout forensic
第七轮  TileLang provider/storage/copy/pipeline closure
第八轮  同一 HEAD 上两机三家全量、公平 candidate 对照与 Triton 1.05× 性能闭合
```

原第五轮中关于六表全量、算法/计时资格、candidate set 对齐、两机并行和 1.05× 的内容没有取消，整体迁到第八轮。

---

## 1. 调查范围

本调查读取并对照：

- `report/prompt/WRITE-05.md`；
- `report/full-gpu-regression-and-triton-performance-closure.md`；
- 现有 `report/prompt/05-intentdsl-full-validation-triton-closure-prompt.md`；
- `doc/compiler/` 全部规格；
- 当前 KIR→GPU construction、shared GPU IR、transforms、verifier、三家 provider 与 runner；
- `/home/kingdom/phdworks/ref/triton` 中的 pipeline、AxisInfo、BufferRegion、Alias/Allocation 与 instruction reordering；
- `/home/kingdom/phdworks/ref/tilelang` 中的 BufferRegion、pipeline planning、semantic checks 与 CUDA pipeline；
- 当前六张 `report/baselinev2/*.csv`；
- 旧链提交 `8d27372` 的 provider forms，以及该提交中 `report/baseline-new/*.csv` 的历史表；
- 当前实际 cuTile Python package 的 arithmetic lowering，用于确认 NaN semantics。

当前工作树通过实时枚举 `examples/kernels/**/*.py` 并计数 `@intent.kernel` 得到 93 个文件、217 个 kernel。`WRITE-05.md` 中的 216 是写作时快照；本次调查没有修改 `examples/kernels/`，因此不把这 1 个差值归因于本任务。新闸门必须在执行时枚举完整 corpus，报告当时的文件数、kernel 数和数量变化原因，不能把 217 变成下一个写死的分母。

---

## 2. 五个待验证假设的判定

| 假设 | 判定 | 调查后的准确表述 |
|---|---|---|
| 三家 registry 几乎不重叠，Triton 不能证明 shared 完整 | 成立 | cuTile/TileLang 各只有 7 个 kernel 名与 Triton 重合、各有 30 个不重合；当前 6 个 cuTile 与 14 个 TileLang shared failures 全在不重合集合 |
| shared 层没有 analysis authority | 部分成立 | typed current-IR facts真实存在；缺的是 dependence/access/provenance/lifetime等可复用分析及失效边界，不是简单缺一个 `Analysis/` 目录 |
| pass 按 op family 切，所以结构错误 | 部分成立 | family-specific structured lowering合理；错误是 zero/tail/replay/ownership/access等共同 decision在多个 family内独立重推导 |
| pipeline 顺序承重，因此应可任意重排 | 原表述不成立 | Triton/TileLang 同样有严格顺序；Intent的问题是 intermediate program不完整、analysis无 preservation/invalidation、顺序靠将被 erase 的形状事实承重 |
| `operator_kind` 是去掉名字改用编号 | 成立且更严重 | cuTile 已把 propagating max/min静默映射到 `propagate_nan=False` 的原语，是 concrete correctness bug |

调查还发现两个原假设未覆盖的关键问题：construction无分析依据地创建所有 dynamic-dimension fragment candidates；`d580ba8` 在 adapter按 entry过滤 generated candidates，不能证明 compiler legality或公平 timeout closure。

---

## 3. 当前 shared compiler 的真实结构

当前执行链是：

```text
canonical KIR
  → lowerCanonicalKIRToGPU / constructGPUProgram
  → runSharedGPUPasses（手写函数序列）
  → provider legalize
  → provider serialize
```

`constructGPUProgram` 已经不是“KIR 改名加旁表”：它建立独立 physical function、physical ABI、program ID、workset coordinates、program-space segments、GPU scalar/fragment/access/structured ops，并最终删除原 KIR function（`KIRToGPU.cpp:4306-4603`）。这是第五轮一万多行中真实的架构进展，不能全部归类为窄特判。

shared GPU IR 也有真实的 current-program facts：

- `FragmentType` 保存 physical shape、axis maps、validity 与 owner（`GPUTypes.td:24-37`）；
- `RangeType` 保存 source ID/source axis（`GPUTypes.td:39-45`）；
- `BufferType` 保存 scope、instance、owner、initialization、lifetime、visibility 与 workspace（`GPUTypes.td:47-64`）；
- `Load/Gather/Store` 显式保存 coordinates、valid/fill、source axes（`GPUOps.td:162-203`）；
- reduce/scan/fold/scan/contract family都有 operands/results/regions（`GPUOps.td:205-293`）；
- atomic family显式保存 kind/order/sharing 与 result（`GPUOps.td:302-365`）。

所以“shared 层没有任何事实”是错误结论。问题在于这些事实是否完整、由谁推导、被改写后如何重算，以及是否在每个 pass 边界仍构成 executable program。

---

## 4. 更根本的问题：initial program 和中间 program 不完整

### 4.1 接口声明与 verifier 互相矛盾

`KIRToGPU.h:16-19` 声称 conversion：

> replaces it with one complete conservative provider-neutral executable GPU program

但 `VerifyGPUProgram.cpp:186-189` 对任何仍存在的 `RegionFoldOp` 或 `RegionScanOp` 直接报错：

> is not a complete physical program until segment traversal, source slicing, carry and output flow are materialized

construction 会建立这些 structured ops，后续 `realizeRegionFolds/Scans` 才把它们物化。也就是说：

- construction 结果不是 complete executable program；
- region realization 之前的中间 IR 不是 complete executable program；
- interface 注释、`doc/compiler/kir-to-gpu.md:52-65` 与实际实现不一致。

### 4.2 “每个 pass 后 verify”实际只检查了 MLIR well-formedness

`runSharedGPUPasses` 在绝大多数变换后调用的是：

```cpp
mlir::verify(module.getOperation())
```

只有整个序列结束后才调用 `verifyGPUProgram`（`Passes.cpp:9-57`）。通用 verifier能检查 op/type 自己的局部约束，却不会检查：

- effect coverage是否完整；
- program mapping与logical ownership是否完整；
- structured op 是否已形成完整 loop/carry/access flow；
- coordinate provenance 是否仍覆盖所有 effect；
- provider-neutral program 是否已经独立可执行。

因此原第三、四轮报告把“中间穿插 verifier”当作成熟骨架证据并不成立；当前 verifier 只在 pipeline 尾部形成一道完整性门。

### 4.3 这比“PassManager 还是手写函数”更重要

Triton 与 TileLang 的 pipeline 同样严格有序：

- Triton NVIDIA backend 在 `ref/triton/third_party/nvidia/backend/compiler.py:294-371` 依次加入 coalesce、thread locality、matmul acceleration、pipeline、layout removal、fence、lower MMA 等 passes；
- TileLang 在 `ref/tilelang/tilelang/cuda/pipeline.py:68-163` 明确要求 pipeline planning 在 layout inference 前、buffer-init verification 在 tile-op access regions 仍存在时执行。

所以成熟性不要求 pass 任意重排。真正要求是：

- 每个 pass 的输入事实和输出不变量明确；
- 当前 IR 在阶段边界合法；
- analysis 被 mutation 后要么重算、要么显式声明保留；
- 后序 pass 不靠已经消失的 op 形状恢复事实。

当前 Intent 在这四点上都不完整。

---

## 5. shared analysis authority：假设部分成立

### 5.1 当前谁“知道事实”

Intent 当前主要有三种事实来源：

1. **IR-resident typed facts**：`FragmentType`、`RangeType`、`AxisMapAttr`、access operands、validity、owner、buffer fields；
2. **construction 内部临时模型**：`ParallelWorkset`、ABI maps、dimension maps、value maps；construction 返回后这些 C++ objects消失；
3. **family-pass 局部递归分析**：每个 pass各自沿 producer/consumer graph 寻找 source ranges、tail predicates、replayable values、dependencies。

例如：

- `RealizeAccessComposition.cpp:26-98` 自己组合 axis map 与 gather/load；
- `RealizeContractionBlocking.cpp:132-271` 自己收集 producer ranges 与 source coordinates；
- `RealizeRegionFold.cpp:115-207` 自己建立 `SourcePlan`；
- `RealizePointwiseBlocking.cpp:819-885` 有一个只服务当前 pass 的 reduction-dependency 递归；
- `Utilities.cpp:24-227` 集中了一些跨 pass helper，但没有形成 lattice/result、analysis key或 invalidation contract。

`lib/Dialect/GPU/` 下确实没有 `Analysis/` 模块或 MLIR analysis objects。但“没有 Analysis 目录”不是判据；要紧的是相同事实是否只有一个权威推导并被多个 transformation 消费。当前答案是否定的。

### 5.2 Triton 的对应结构

Triton 使用可复用分析结果：

- `AxisInfo` lattice保存每轴 contiguity/divisibility/constancy 与 constant value（`ref/triton/include/triton/Analysis/AxisInfo.h:22-84`）；coalescing、async copy、pipeliner 等共同消费；
- `BufferRegionAnalysis` 保存 exact/unknown physical regions、allocation-frame provenance 与 intersects/contains（`ref/triton/include/triton/Analysis/BufferRegion.h:32-154`）；fence、sanitizer 等共同消费；
- `SharedMemoryAliasAnalysis`、`AllocationAnalysis` 与 liveness处理 alias、offset、size 与 live buffers；
- `ReorderInstructions` 直接使用 effects、dominance、first use 与 register-pressure判据，而不是假设 block walk 顺序可改（`ReorderInstructions.cpp:43-142`）。

Triton 的 analysis也不是万能：Alias 的部分接口仍 conservatively 返回 MayAlias，BufferIndexAnalysis证不出时退回保守路径。成熟点在于 unknown 是显式结果，而不是换一套 shape matcher猜答案。

### 5.3 TileLang 的对应结构

TileLang 没有复制 Triton 那套通用 lattice。它主要依赖 TIR/TVM 与正式 `BufferRegion`：

- pipeline planner 用 `BufferRegionCollector` 收集每条 statement 的 reads/writes，再判断 replayability 与 pipeline write buffers（`pipeline_planning.cc:271-338`）；
- semantic checks拒绝非法 nested parallel 与 fragment-loop indexing；
- StorageRewrite、layout inference、buffer init verifier 等在明确 pipeline 阶段消费 TIR facts。

这推翻了“成熟编译器一定要有六个 Analysis 目录”的做法。Intent 应建立自己真正需要的 authority，不复制 Triton 类名；但 dependence/access/coordinate/lifetime 不能继续由四个 family pass 各算一遍。

---

## 6. pass 边界：不是禁止 family pass，而是消除重复 decision authority

### 6.1 假设“按 op family 切就是错”不成立

Reduce、scan、region fold、region scan 与 contract 的 accumulator/carry 语义不同，分别需要专门 lowering。Triton 也有 `AccelerateMatmul`、pipeline、descriptor 等面向特定结构的 passes；TileLang 也有 reducer、GEMM 和 tile-op lowering。

所以不能把目标写成“所有 pass 都必须按 decision 命名”或“一个 universal blocking pass 处理一切”。

### 6.2 当前可数的真重复

问题是下列 cross-family decision 被独立推导多次：

| physical decision/fact | 当前独立实现簇 | 证据 |
|---|---:|---|
| ownership / program mapping | 约 5 | KIR construction、pointwise ownership、RefineProgramMapping、ordinary contract、scaled contract |
| zero/fill materialization | 4 | `Utilities.cpp:170`、Pointwise `:385`、Reduction `:868`、RegionFold `:398` |
| tail predicate / validity | 6 | Utilities、Pointwise、Reduction padded/runtime、RegionFold、Contraction |
| fragment/value replay | 至少 5 | KIR conversion、Pointwise、Reduction、RegionFold、Contraction |
| access composition/source-axis recovery | 4 层 | KIR conversion、AccessComposition、RegionFold replay、Reduction replay |
| ordinary/scaled contraction mapping | 2 高度同构块 | `RealizeContractionBlocking.cpp:1641-1789` 与 `:2274-2420` |

`zeroFill` 的机械重复可以直接从当前搜索确认：四个文件各定义一份，而 `Utilities.cpp` 已有 shared 版本。更严重的是 tail predicate：同一件“physical range→compare stop→project/broadcast→与原 valid 合并”在多个 family 各自解释。

这些重复的后果不是代码难看，而是：

- `operator_kind 11/13` 之类的语义差异会在某一份 recognizer 中被遗漏；
- 新 op/helper/reshape 出现时，要同时修改多个 replay whitelist；
- ordinary/scaled contract 很容易只修一条路径；
- multi-contract joint realization无法先形成统一 execution/access decision，再由各 contract consumer消费。

### 6.3 新第五轮的可数性质

闸门不能要求“只剩一个函数”，因为 family-specific rewrite patterns可以合理存在。应当数的是：

> 对每个 shared physical fact/decision，有几处代码在没有读取统一 analysis/result 的情况下独立推导它。

完成态要求每类事实只有一个 semantic/analysis authority；多个 consumer/pattern 可以存在，但只能消费该 authority，不得另做一份 source-axis、tail、ownership 或 replay 推理。

---

## 7. construction 自己已经嵌入窄 physical policy

这是原 `WRITE-05.md` 没有点出的更重要问题。

### 7.1 workset 不是由 dependence/effect analysis形成

`constructGPUProgram` 只从顶层/嵌套 `intent.parallel` 收集 worksets；没有 parallel 时直接建立 singleton（`KIRToGPU.cpp:4306-4323`）。`collectParallelWorksets` 还通过“region 内是否出现 Reduce/Scan/Contract/Fold/For/If 等 op family”设置 `pointwiseCoordinates`（`:4262-4287`）。

这不是 docs 所定义的：workset 应从 parallel、unique writes、structured free axes、def-use、alias、effect 与 dependence共同形成（`doc/compiler/kir-to-gpu.md:15-35`）。当前 construction 仍依赖作者源码恰好已经把结构包进 `parallel`，并用 whole-region op-family presence决定后续 physical path。

### 7.2 construction 无条件为动态 dimensions 创建固定 fragment parameters

`KIRToGPU.cpp:4359-4392` 扫描所有动态 tensor dimensions，为每个 dimension创建：

```text
FRAGMENT_D<id>
role = OwnershipN
candidates = {64, 128, 256}
```

这个决定发生在 access、reuse、structured operation 和 target capability analysis之前；所有 dimension 还统一标成 `OwnershipN`。它不是保守 scalar baseline，而是 construction 中写死的一张 physical policy。

这说明当前“后续 pass 只是 refinement”也不准确：initial construction 已经做了一个没有统一 legality/analysis支持的 blocking选择，后续 family pass 再覆盖或解释它。

### 7.3 capability schema 不足以支撑规格中的 legality

当前 `GPUCapabilities` 只有 compute units、shared memory、registers、matrix units 与 dynamic vector width（`KIRToGPU.h:8-14`）。规格中的 grid limits、dtype、structured primitive、atomic 与 provider-form capability没有进入同一 typed capability object。

并不是所有 provider能力都应进入 shared IR；但当前 shared pass 一方面试图筛选 structured/blocking candidates，另一方面缺少判断这些合法性的完整 typed inputs，结果只能把问题拖到 provider verifier 或 JIT。

---

## 8. pipeline 顺序：假设需要重写

当前顺序是：

```text
access composition
→ multi-axis reduction decomposition
→ pointwise ownership
→ pointwise blocking
→ region fold
→ region scan
→ contraction blocking
→ reduction blocking
→ program mapping refinement
→ access composition（第二次）
→ final shared verifier
```

结论分三部分：

1. **顺序有依赖是正常的。**Triton 和 TileLang 同样有必须前后相邻的 passes。
2. **第二次 access composition 当前有实际原因。**前面的 structured realization会 replay source slices并产生新 gather，第二次 composition处理这些新 access；不能只按“重复调用”删除。
3. **当前顺序过度承重。**因为 analysis只存在于 op shape/attrs与局部递归中，某个 pass erase/replace source graph后，后序不能通过统一 analysis重算，只能要求“趁 logical source 还完整先做 X”。这使插入新 transformation 很容易打断 provenance。

另外，`decomposeMultiAxisReductions` 在 top-level pipeline显式运行一次，`realizeReductionBlocking` 内又无条件调用一次。第二次通常因第一次已改写而幂等，但这是 transformation 的隐式依赖，不是 analysis recomputation。

新第五轮不应要求“pass 能任意重排”，而应要求：顺序依赖有明确输入/输出不变量；分析结果有失效/重算规则；每个阶段的 current program合法；重复 normalization由显式 fixed-point/canonicalization职责解释，而不是函数里偷偷再调用。

---

## 9. 跨层语义载体：`operator_kind` 是 correctness bug，不只是架构味道

### 9.1 规格已经定义语义，不需要重写语言

`doc/dsl/types-numerics-and-effects.md:87` 已经明确：

- `I.maximum/I.minimum` 传播 NaN；
- `maximum_num/minimum_num` 忽略单侧 NaN；
- 两者是不同 pointwise operations。

所以这不是 doc 缺失导致的开放问题。新第五轮要修的是 canonical KIR/shared GPU IR/provider boundary 没有可靠保存已冻结语义。

### 9.2 当前载体

- Python frontend有 `BinaryOperator` IntEnum，`MAXIMUM=7`、`MINIMUM=8`、`MAXIMUM_NUM=9`、`MINIMUM_NUM=10`；
- canonical `Intent_BinaryOp` 与 shared `IntentGPU_BinaryOp` 都只保存 `I64Attr operator_kind`；
- KIR→GPU 原样复制整数；
- shared passes和三个 provider分别用数字比较或数组下标解释。

当前 `include/`、`lib/`、`python/` 中 `operator_kind/OperatorKind` 共出现 103 次、分布在 20 个文件；仅 `getOperatorKind() ==/!= <number>` 形式就有 59 处、分布在 9 个文件。`WRITE-05.md` 中“约 75 处、41 个直接比较”已经过时，实际扩散面更大。

有 Python enum 不等于跨层 typed authority。真正参与 verifier、rewrite 和 serializer 的仍是裸整数协议。

### 9.3 已确认的静默语义收窄

cuTile `nativeCombineKind`：

```text
7 or 9  → ct.max
8 or 10 → ct.min
```

当前实际使用的 `intentdsl-cutile` 环境中，`cuda-tile 1.5.0` 的 float `min/max` lowering在 `/home/kingdom/.venvs/intentdsl-cutile/lib/python3.10/site-packages/cuda/tile/_ir/arithmetic_ops.py:419-432` 明确传：

```text
propagate_nan=False
```

因此：

- canonical `MAXIMUM_NUM/MINIMUM_NUM` 映射到该 form有语义依据；
- canonical propagating `MAXIMUM/MINIMUM` 映射到同一 form会改变含 NaN 输入的结果。

这不是“可能风险”，是当前代码已经存在的错误映射。TileLang `Bufferize.cpp:378-399` 也把 `7/9`、`8/10` 合并，但需要在第七轮结合 TileLang/TVM 的精确定义决定 native form或明确拒绝。

### 9.4 第二个实例

`RealizeContractionBlocking.cpp:423-428` 同时把 kind `11`（logical-and）与 `13`（bitwise-and）识别为 tail conjunction，却没有在该 recognizer中证明 kind `13` 的元素类型一定是 `i1`。如果输入确实为 `i1`，两者可以等价；但这个 legality没有进入 typed matcher。

### 9.5 不应机械把所有整数改成 op class

axis index、field index、source ID、component count等结构性整数不是同一问题。Provider API enum在 terminal serialization中使用整数也正常。需要 typed 化的是会跨层决定 observable semantics或 physical legality 的类别：arithmetic/compare semantics、NaN/tie、atomic kind/order/sharing、format schema、parameter/capability role等。

最终判据是：消费者能否通过类型/enum attr/API得到唯一语义，而不是在多个文件共享一张未声明的数字表。

---

## 10. registry 与 shared coverage：假设成立

当前 registry：

```text
Triton   54
cuTile   37
TileLang 37
```

按 `Entry.kernel` 名称：

- cuTile 与 Triton 交集 7，cuTile-only 30；
- TileLang 与 Triton 交集 7，TileLang-only 30。

cuTile 交集是：

```text
dense_gemm, grouped_gemm, layer_norm, moe_expert_projection,
rms_norm, rope_qk, swiglu
```

TileLang 交集是：

```text
block_sparse_gqa_decode, dense_gemm, grouped_gemm,
mamba_chunk_scan, mamba_chunk_state, paged_mla_decode, rms_norm
```

更重要的是失败层核对：

- cuTile 当前 6 个 `physical_program_verification_failed` 全部在 cuTile-only 30 项中；
- TileLang 当前 2 个 `physical_program_failed` 与 12 个至少一机 `physical_program_verification_failed` 全部在 TileLang-only 30 项中。

`tilelang-h100.csv` 仍是修复前快照；上述 union 用来证明失败结构与 Triton registry 的覆盖关系，不冒充同一 HEAD 的最终通过率。同一 HEAD 的六表状态只能由第八轮并行全量重建。

所以原假设“shared failures 落在 Triton 不触碰的 kernel 上”成立。Triton 51/54只能证明 Triton registry覆盖的 shared shapes；它不能证明 shared compiler横向完整。

但另一个极端也不成立：Triton 交集项在 cuTile/TileLang 仍有 provider JIT、verification 或 timeout，说明 shared pass不是唯一问题。第五轮只解决 shared；第六、七轮仍必须单独闭合 provider。

---

## 11. cuTile 14+14 timeout：当前不能下结论

### 11.1 当前 runner 丢失了诊断维度

`examples/repro/v2/runner.py:35,192-221` 对每个 worker只有 300 秒总 timeout。超时后杀进程组并写 `worker_timeout`；CSV不知道：

- frontend/shared/provider compile到了哪一步；
- generated/source哪一侧在运行；
- 实际 candidate 数；
- 当前 candidate；
- 单 candidate compile/JIT/benchmark耗时；
- 是否某个非法候选污染了后续 context。

所以当前数据无法回答“候选笛卡尔积过大”还是“单候选自身编译不完”。

### 11.2 generated 确实会形成笛卡尔积

`CuTile/Serialization/Serializer.cpp:137-176` 展开每个 physical parameter domain，`:431-451` 对 `_CONFIGS` 运行 exhaustive search。`gather_spelling` 等 provider form还会增加维度。

source 侧也有各自的 exhaustive/autotune逻辑；当前没有证据证明两侧候选集合相同。

### 11.3 `d580ba8` 不能作为 compiler closure 证据

该提交在 `examples/repro/v2/providers/*/contraction.py` 中给 `compile_single` 增加 `generated_candidate_filter`：

- cuTile dense 只保留 M/N `{64,128}`、K `{32,64}`、group `8`；
- TileLang dense 固定 `128/128/32`、stages 3、threads 128、group 1；
- TileLang FP8 类似固定一组。

这是 benchmark adapter按 entry过滤 generated candidates，不是 shared/provider pass根据 typed legality计算范围。

TileLang dense source本身固定 `128/128/32`、threads 128、stages 3，因此该 singleton对齐基本成立。cuTile source却在 `MatMul.py:210-218` 对 fp16固定 `tm=128, tn=256, tk=64`；generated filter没有 `N=256`，反而允许不同的 8 个组合。`0.897887×` 不能被解释为同候选预算下的 compiler program-quality结论。

因此 `d580ba8` 只证明“adapter过滤能让该 worker在 300 秒内结束”，没有证明其余 timeout的性质，也没有证明 compiler candidate legality已闭合。

这些问题迁到第六轮 cuTile forensic 与第八轮公平全量；第五轮不继续追 timeout。

---

## 12. 旧链覆盖了什么，为什么不能直接搬回来

使用 `git show 8d27372:report/baseline-new/<provider>-<device>.csv` 直接统计该历史提交内保存的四张表，得到：

```text
cuTile   5090 35/37, H100 34/37
TileLang 5090 19/37, H100 18/37
```

旧 provider forms显式记录过：

- access/bounds/transfer/load-shape/exact-store；
- contraction orientation、batched/MMA/scaled layout；
- row occupancy、row launch、equal program tiles、GEMM warp policy；
- reduction/scan/pointwise/gather/stream/ragged forms；
- TileLang native contraction operand alias/isolation；
- packed INT2 decode load、column axis与producer nodes；
- loop-carrier spaces、selected-contiguous/guarded transfer。

这解释了旧链为什么能覆盖更多 entry。

但旧实现通过 `KernelModel` 回读 KIR，再把大量 executable facts存成 Plan/provider attributes；它正是已经否决的双 authority 与厚 leaf。旧代码只能作为：

1. 已有能力不得退化的 entry-level证据；
2. 新 shared/provider IR 还需表达哪些 structure 的 inventory。

不能恢复旧 path、复制旧 attrs或加 compatibility flag。

---

## 13. 第五至第八轮重新切分

## 13.1 第五轮：shared compiler

第五轮只处理 provider-independent GPU compiler architecture：

- construction 产出真正完整、合法、保守 executable program；
- current KIR/GPU IR 上的 reusable analyses与明确 invalidation；
- observable semantics使用 typed authority贯穿 canonical/shared/provider boundary；
- cross-family physical decisions只有一个推导 authority；
- family-specific structured lowering消费共同结果，不各自重建；
- 每个 shared pass边界有对应完整性 verifier；
- 当前 93 文件、217 kernels 全部独立到达 shared verifier；
- Triton 只做受影响路径的 end-to-end non-regression，不跑 54-entry 性能全量，不追 source 1.05×。

第五轮不处理 cuTile/TileLang provider forms、JIT timeout、性能或六表。

## 13.2 第六轮：cuTile

第六轮使用 37 个 cuTile registry entries，尤其 30 个 Triton registry coverage 之外的结构，验证第五轮 shared facts是否足够；只在 shared program确实缺事实时回报 shared regression，不在 cuTile leaf重建。

同时完成：

- 14+14 timeout forensic；
- candidate 数、阶段、单 candidate成本与 context污染隔离；
- access/gather/scatter/reduction/MMA/scaled/stream/ragged provider forms；
- 旧链 cuTile 35/37、34/37 的逐 entry能力恢复。

## 13.3 第七轮：TileLang

第七轮用 TileLang 30 个非 Triton重合 kernel验证 shared 通用性，并完成真正 provider-local 的：

- storage/allocation；
- BufferRegion/copy；
- fragment/shared；
- structured primitive operands；
- synchronization与pipeline；
- sparse/atomic/scatter/workspace forms。

重点是把当前两千行 Bufferize 中重新推导 shared access/execution 的部分移回第五轮建立的 authority，保留 TileLang真正需要的 storage/copy/sync legalization。

## 13.4 第八轮：全量与性能

原第五轮以下内容整体迁到第八轮：

- 两机并行、同一 HEAD、六张表；
- 完整 registry分母；
- source 必须是同语言公开高性能实现，并保留 upstream provenance；输入 shape、dtype、ABI、输出与 runtime-visible callable 对齐；
- 算法、kernel 数、pipeline、dtype、NaN/tie、workspace、reset 与 timed closure资格；
- `region_fold/region_scan` 与 multi-kernel wrapper 逐 entry 核对算法分解，不把 ordinary loop、显式 chunk ABI 或上游分阶段算法改成 compiler-selected segmentation；
- source/generated使用相同的实际过滤后 candidate set，各自由 provider tuner选 winner；既包括多候选 autotune，也包括明确的 fixed-singleton 对照；
- candidate 资格需保存过滤前/后集合与实际 winner，不用 adapter 中的 entry-local 筛选代替 compiler/provider legality；
- measurement order、warmup、JIT/autotune排除、CUDA Graph 边界、mutable reset 和 multi-kernel timed closure；
- 两机使用同一 HEAD 并行执行，六张 CSV 保持当前 schema，不合并、不加版本字段或表内检查逻辑；
- 所有 strict Triton rows `<1.05×`；
- 数值/编译回归优先修复；
- ratio 只对公平资格成立的 entry 有意义；不成立时保留具体阶段和原因，不用空值或宽泛 `compile_failed` 混过去；
- candidate/winner、runner 失败阶段与不适用 ratio 的逐 entry证据。

`d580ba8` 暴露的 cuTile candidate mismatch必须在第八轮公平资格中重新核对，不能沿用当前 ratio结论。

---

## 14. 新第五轮的双闸门

单靠 corpus coverage 会再次诱导在四个 family 文件里加窄规则；单靠架构整理又会重演“一条 kernel 跑通就算完成”。所以必须同时通过两道闸门。

### 可执行 coverage 闸门

执行时完整 `examples/kernels/` corpus中的每个 `@intent.kernel`，必须用一条生产 compiler 边界的 compile-only 枚举命令通过：

```text
frontend → canonical KIR → KIR-to-GPU construction
         → 全部 shared passes → full shared GPU verifier
```

全部成功。这是新第五轮 prompt 对仓库默认验证纪律的一项明确限定例外：额外允许一条 compile-only compiler-stage inventory audit，仅用于完整 corpus 的 shared-stage 闸门。它不新建测试基础设施、不运行 GPU、不做 benchmark。当前基线是 93 文件/217 kernels；数量变化必须由真实 corpus变化解释。不得 allowlist、不得只跑 registry、不得通过某个 provider serializer间接宣称 shared成功。除这条例外外，GPU 数值验证仍只保留一条受影响 Triton kernel 的手动 end-to-end repro；54-entry runtime/performance 全量属于第八轮。

若现有 CLI 无法在 shared verifier后独立停止，闭合这个正式 compiler boundary属于本轮架构工作；不能为闸门建立 test-only第二条 lowering path。

### decision-authority 闸门

报告必须给出 before/after 表：

```text
physical fact/decision
→ 唯一 authoritative analysis/IR carrier
→ 唯一 mutating transformation entry
→ family/provider consumers
→ invalidation/recompute rule
```

同一事实被独立推导的次数必须降为 1。多个 rewrite patterns可以存在，但不得各自从 source ID、shape、origin、op邻接重新得到同一 ownership/tail/replay/access结论。

两道闸门缺一不可：217/217 通过但 decision仍重复，不算完成；结构整理完成但任一 kernel过不了 full shared verifier，也不算完成。

---

## 15. `doc/` 与 `AGENTS.md` 权限结论

执行第五轮时可以修改 `doc/`，但只在 ref 与代码证明现有规格缺少稳定语义/架构事实时修改。当前调查已经确认：

- propagating maximum 与 maximum_num 的区别已在 `doc/dsl/types-numerics-and-effects.md:87` 定义，不应借实现 bug重写语言；
- initial program completeness、analysis invalidation与每 pass完整性已在 `doc/compiler/` 定义，当前应修实现，不应把 family限制写进 doc；
- 如果进一步发现 atomic/format/capability 等规格确实没有唯一语义，可以补规格；不能把“当前只支持 unit-step/full-program segment”写成合法边界。

`AGENTS.md` 只需增加一条长期自查纪律：对承载语义或 physical decision 的改动，必须找到 `ref/triton` 或 `ref/tilelang` 的同类实现，给出双方 file:line、差异及实际后果；找不出具体差异等于没有完成自查。不要加入长检查清单。

---

## 16. 最终判断

原 `05-intentdsl-full-validation-triton-closure-prompt.md` 的公平比较内容本身大体正确，失败在轮次前提：它假设 shared/provider reconstruction已经横向完成，实际全量里又补了一万多行后端。

新的第五轮必须先让 shared compiler在 217 个真实 kernels 上独立成立，并把 analysis、semantic carrier、decision authority与 pass verifier做成成熟编译器结构。只有这样，第六、七轮接 provider时才不会一边补 leaf、一边继续造 shared；第八轮的六表与 1.05× 才有稳定对象可测。
