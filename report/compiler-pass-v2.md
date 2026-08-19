# Compiler Pass V2：GPU 算子编译器的成熟态骨架

## 0. 文档定位

这份报告是在 V1 现状审计以及后续多轮反驳、Triton/TileLang 实现核查之后形成的 V2 设计定义。它回答的是：

> **固定 Intent Kernel IR 所表达的算法以后，GPU 算子编译器究竟还要编译什么；这些决定应如何存在于一份可变换的物理程序 IR 中；哪些步骤是真正的结构 decision pass，哪些只是 target materialization；Triton、cuTile、TileLang leaf 与它们下层编译器分别负责什么。**

本文只讨论 GPU realization。CPU、RVV、图级编译、多 source-kernel orchestration 不在本文范围内。

本文是成熟态设计报告，不描述当前代码已经完成了哪些迁移，也不设计实验、探针或性能对照。凡是使用“V2 应当”“成熟态”之处，都是目标架构；凡是使用“当前”之处，才是仓库现状。

V2 继承 V1 的以下结论：

- IntentDSL 不是 whole-op library selector，而是按 typed Kernel IR 的 axis、value、access、def-use、state、structured operation 和 effect 组合目标程序；
- 固定 Kernel IR 后仍存在有限、结构化、以确定性 policy 为主的 physical realization space；
- autotune config、TMA、`program_id` 或单个 mask 都不能代表整个编译空间；
- target leaf 不是无意义的字符串 printer，provider-native program construction 本身就是编译工作；
- 下层 Triton、TileLang、cuTile compiler 仍拥有 layout、寄存器、指令和低层流水等重要优化。

V2 修正 V1 的两个根本判断：

1. 当前 `Kernel IR + Physical Plan` 不能继续作为成熟态的双重 executable authority；
2. 三种 GPU surface 不是“同一份 Plan 的同等粒度语法打印”。它们暴露给上层的物理抽象高度不同，因而需要不同的 provider-local passes。

---

## 1. 最终定位

Compiler Pass V2 是一条：

> **typed-fact-driven、target-aware、deterministic 的 GPU physical-program realization pipeline。**

它在不改变 Kernel IR 算法、logical workset、value flow、state、effects、ABI 和数值角色的前提下，把一份 target-independent 的结构化区域算法实现成一份完整、合法、可持续改写的 GPU physical program，再由 provider leaf passes 将它合法化为 Triton、cuTile 或 TileLang 的目标程序。

本文统一使用三个术语：

- **shared physical IR**：shared GPU passes 正在变换的 executable GPU program；
- **provider-legal IR**：已经完成 Triton、cuTile 或 TileLang form selection 与 materialization、可交给 terminal translator 的 IR；
- **translator**：只序列化 provider-legal IR 为 source、entry 和 wrapper 的终端组件。

责任边界可以先压缩成三句：shared passes 构造跨 provider 仍成立的完整 GPU program skeleton；provider passes 选择并兑现该 surface 确实要求的 concrete forms；provider compiler 继续完成 layout、register、instruction 和低层 pipeline。

对于固定的：

- Kernel IR；
- device；
- provider；
- compile policy；
- 已实例化的 tuner configuration；

V2 产生一条 canonical physical-program path，以及该 provider/configuration 对应的一份 canonical target program。Tuner configuration 只能实例化已声明的数值参数，不能重新选择 execution、value、access/validity 或 structured-operation structure。V2 不要求枚举多个 structural schedules，不要求全局 cost model，也不通过运行时测量选择程序结构。

“deterministic”不意味着没有编译空间。它表示多个合法 realization 客观存在，但 compiler policy 根据 typed semantics、def-use、region structure、reuse、lifetime、validity、structured-op contract、device facts 和 provider capability 确定性地选出一个合法而高质量的结果。

---

## 2. 算法边界：V2 不能编译什么

Kernel IR 是 source-visible kernel algorithm 的权威表示。它已经固定：

- runtime-visible ABI、shape、dtype、stride/alignment/effect contract；
- logical domain、region、ragged membership 和原始 logical identity；
- tensor/value flow、broadcast/reshape/transpose/index relation；
- reduce、scan、contract、sparse contract 及 typed combiner；
- runtime control、`parallel`、`ordered`、`state_stream` 和 carry schema；
- logical buffers、mutable access、scatter/atomic conflict semantics；
- 数值路径、累加语义和 observation-visible effects。

V2 可以改变物理 representation，但不能改变这些算法事实。例如：

- GEMM 不能变成 Strassen；
- stable softmax 不能变成另一种 online recurrence；
- `ordered` 或 `state_stream` 不能降格为 unordered partial merge；
- effectful operation 不能被非法复制、删除或跨依赖重排；
- 一个 source scalar instance 可以被装入 physical lane，但 compiler 不能自动把标量算法改写成 source 未表达的块 contraction；
- graph-level fusion/fission、跨 source callable 融合和 wrapper orchestration 不属于这条 pipeline。

一项 physical transform 是否越界，判断依据不是它改了多少 IR，而是它是否保持同一 logical workset、value/effect semantics、state transition、ABI 和数值角色。

---

## 3. 当前 V1 骨架的准确性质

当前 `intent_plan` 不是 Python side object，也不是无效的临时表。它是 module 内正式、可验证的 MLIR dialect，并真实约束三个 emitter。当前 Realizer 也已经拥有 axis/range、worker/fold/reuse、persistent、residency、transfer、scan/contract 和 stage 等确定性 policy。

但当前 Plan 仍不是一份完整、可执行、可独立变换的 physical program。

### 3.1 Plan 是 KIR-referenced physical skeleton

`plan.realization`、`plan.axis`、`plan.range`、`plan.buffer`、`plan.transfer`、`plan.contract`、`plan.stage` 等 operation 主要通过 KIR node/value ID 和 attribute 保存决定。绝大多数 Plan operation 没有构成 physical computation 的 SSA operands/results，也没有形成完整的 physical control/dataflow graph。

因此当前 Plan 最准确的名称是：

> **KIR-referenced selected physical skeleton / reference Plan IR。**

它比普通注释强，因为它有 dialect、verifier 和明确 schema；但它仍然依赖 KIR 才能解释实际程序。

### 3.2 当前程序构造仍发生在 leaf

当前 Realizer 的基本动作是：分析 KIR，创建 `plan.realization`，遍历 KIR，再追加 Plan records。

当前 emission 的基本动作则是：

```text
verify Kernel IR + Plan
        ↓
重新分析 Kernel IR
        ↓
把 KernelModel + RealizationOp + SearchSpaceOp 交给 target leaf
        ↓
target leaf 再次遍历 Kernel IR，并查询 Plan/index
        ↓
构造 target source
```

所以当前真实关系不是：

```text
KIR → executable Plan → source
```

而是：

```text
KIR → selected Plan skeleton
(KIR, Plan skeleton) → target source
```

这解释了为什么 leaf 中会出现完整 K-loop、pointer/mask、scan workspace loop、TileLang allocation/copy/sync/pipeline，以及若干未记录的局部 policy：可执行 physical program 在 leaf 之前尚不存在。

### 3.3 根本问题不是“Plan 字段不够多”

继续给当前 Plan 增加 attribute，可以记录更多 decisions，却不能自动把它变成 physical program。成熟态所缺的是：

- physical values 和 def-use；
- physical operation operands/results；
- explicit physical regions、loops 和 control；
- transfer、collective、materialization、workspace 和 synchronization 的程序位置；
- provider-specific type/attribute/op；
- 后续 pass 能直接观察、替换和重组的 executable structure。

因此 V2 不是“把更多 leaf `if` 搬进 Build.cpp”，而是改变 compiler passes 所变换的对象。

---

## 4. V2 的核心骨架：每个阶段只有一个 executable authority

成熟态应采用：

```text
Canonical Kernel IR
        ↓  dialect conversion / physical-program construction
GPU physical program IR
        ↓  shared GPU decision/refinement passes
refined GPU physical program IR
        ↓  provider leaf passes
provider-legal physical program IR
        ↓  terminal translator
Triton / cuTile / TileLang source
        ↓
provider compiler / tuner / machine code
```

这里的“只有一个 executable authority”不是要求磁盘或 module 中永远只有一个对象，也不要求必须原地销毁 KIR。原始 KIR 可以作为缓存、调试产物、provenance 或语义验证输入保留。

真正的不变量是：

> **进入 physical-program 阶段后，execution、control、physical def-use、storage 和 operation structure 只能由当前 physical IR 提供；后续 passes 和 leaf 不再依赖“旧 KIR 程序 + 一张旁表”重建缺失的 physical structure。**

这不取代 KIR 的算法权威。KIR 仍然是 logical workset、算法、数值角色、state 和 effect semantics 的 provenance/semantic authority；physical IR 必须显式承载这些语义的合法投影。leaf 必须能遍历 physical IR 完成 provider lowering，而不是遍历 KIR 再回查 Plan。KIR reference 可以作为 provenance attribute 存在，但不能承担缺失的 physical def-use、control、storage 或 operation semantics。

### 4.1 V2 不要求字面上的“新 module”

Triton 的 TTIR→TTGIR 是同一 Module 上的 partial dialect/type conversion：给未编码 tensor 类型挂上合法初始 encoding，保留仍然 legal 的 Triton/SCF/arith operation，同时插入或改写 TritonGPU operation。后续 passes 继续修改同一 executable IR。

Intent 可以选择原地 conversion，也可以保留 KIR 后构造新的 physical function/module。V2 关注的是 compiler authority 和 IR semantics，不是 C++ object 是否相同。

### 4.2 Physical Plan 能否继续使用这个名字

可以。名字不是关键。

如果现有 Plan 逐步获得完整 physical function、SSA values、operations、regions、types 和 provider extensions，并最终成为 leaf 唯一遍历的程序，那么它可以继续叫 Physical Plan。

但一旦做到这些，它在语义上已经不再是当前 reference Plan，而是一份 GPU physical program IR。仅增加更多 attribute、同时保持 leaf 遍历 KIR，不算完成这次转变。

---

## 5. 共同 GPU IR 不能以 tile 为本体

V2 是 GPU 路径，但 GPU physical program 不能被等同于 rectangular tile program。

Intent Kernel IR 的真实覆盖包括：

- pure scalar SSA 和 runtime `while`；
- scalar/lane packing；
- source-visible region 和 partition；
- ragged membership、dynamic offset 和 data-dependent gather/scatter；
- ordered traversal 和 state carry；
- dense、block-sparse、CSR 和 2:4 sparse structure；
- mutable logical buffers、atomic、effect ordering；
- reduce、scan、contract 和 multi-stage value flow。

因此共同 GPU physical IR 必须能够表示：

- scalar value；
- shaped/distributed value；
- fragment、buffer、workspace 或 provider-specific value form；
- logical membership 与 physical ownership；
- serial、ordered、persistent、ragged 和 sparse traversal；
- state carry 与 logical stop；
- effectful transfer、atomic 和 visibility；
- collective 和 structured operation；
- compiler-private execution stages 与 dependencies。

tile 仍然重要，但它只是 physical granularity 的一种。其他 granularity 还包括：

- scalar pack width；
- program chunk；
- lane/worker group；
- reduction chunk；
- state-stream segment；
- access footprint；
- sparse block；
- matrix tile；
- stage-local range。

V2 不能把所有 kernel 压成一个 `tile` 类型，也不能用 row/tiled/ragged 等 mutually-exclusive kernel class 代替逐轴、逐值、逐 operation 的组合事实。

---

## 6. 真正的结构 decision space

固定同一 KIR 后，V2 的 compiler-owned 结构空间由四组可组合 decisions 构成。它们不是四种 kernel 类别，也不要求每组恰好对应一个 C++ pass。

四组的归类规则是：execution 回答“谁在何时、以什么 traversal 执行”；value 回答“值以什么 representation 和 lifetime 存在”；access/validity 回答“如何合法读写”；structured-operation 回答“高层 structured node 采用什么 loop/collective/primitive skeleton”。四组不是互斥字段集合，但同一个 physical choice 必须有一个主语义归属，其他组只保存约束或派生结果。

Effect、order、dependency 和 visibility 不是第五组可自由选择的 decisions，而是横切四组的 legality constraints。相关 obligation 必须 materialize 为 physical IR 中的 dependency、effect 或 synchronization entity；任何 pass 都不能重写 KIR 已固定的 effect order。

### 6.1 Execution realization

这一组决定 logical instances 和 regions 如何成为 GPU 程序的执行结构：

- 哪些 logical axes/ranges 形成 grid/program ownership；
- 哪些形成 worker/lane/serial traversal；
- 不同层级怎样嵌套；
- grouped、folded、reuse 和 persistent traversal；
- ragged membership 如何绑定 owner 与 variable extent；
- `ordered`、state stream 和普通 serial loop 的物理遍历方式；
- collective participants 属于何种执行 scope；
- compiler-private execution stage 如何分组与提交。

单独打印一个 `program_id` 不是重要优化。只有 ownership 与 traversal、reuse、granularity、value lifetime 共同改变目标程序时，它才成为有性能含量的 execution policy。

### 6.2 Value realization

这一组决定逻辑 value 在物理程序中如何存在：

- inline expression；
- 保留为 SSA value；
- replay/recompute；
- materialize；
- private/shared/fragment/TMEM 等 provider-local concrete form；
- 跨 stage workspace；
- owner、sharing scope、lifetime 和 visibility；
- logical mutable buffer 的物理 realization；
- pure producer 是复制、重算还是成为 intermediate。

共享层可以先记录 provider-independent obligation，例如：

- value 是否必须跨 participant 共享；
- 是否必须跨 stage 持久化；
- 是否允许 replay/recompute；
- lifetime 与 visibility 要求；
- workspace 是否不可避免。

具体 `shared`、`fragment`、TMEM、virtual tensor 或 cuTile tile/register form，可以由 provider-local pass 选择。不能把“block 内复用”直接等同于唯一的 shared allocation。

### 6.3 Access and validity realization

这一组决定 logical access 如何成为合法且高质量的 physical transfer：

- access footprint 和 physical coverage；
- bulk transfer、elementwise transfer、gather/scatter 或直接 scalar access；
- pointer/index/address construction；
- boundary 在 producer、transfer 或 consumer 兑现；
- mask、range guard、checked operation、neutral fill；
- consumer neutralization 能否消除 materialized mask；
- ragged/data-dependent index 的有效范围；
- conflict、atomic scope/order 和 effect constraints 如何限制 physical scheduling。

KIR 的 conflict、ordering 和 effect semantics 不能由 V2 重选；V2 选择的是保持这些语义的 physical access form。

### 6.4 Structured-operation realization

这一组决定高层 structured node 如何成为 physical loop/collective/primitive skeleton：

- contract 使用哪个合法 primitive family/variant；
- reduce 的 participant topology、private partial 和 collective level；
- scan 使用 fragment collective、ordered loop 还是 workspace-backed realization；
- sparse contract 使用 native primitive、format-specific lowering 或明确拒绝；
- CSR/dynamic sparse access 如何与 execution/access realization 组合；
- state stream 如何形成 loop、carry、segment 和 final projection；
- staged structured operation 对 replay/materialize/private-stage 的 requirement。

structured-op realization 不得改变算法语义。例如 reduce combiner、scan inclusive contract、contract reduction axes、accumulator dtype 和 sparse metadata semantics 仍来自 KIR。

Stage grouping 的主归属是 execution realization；structured-operation pass 只能声明 structured node 对 stage、workspace 或 materialization 的 requirement，不能自行重新划分 execution stage。Compiler-private multi-kernel stage 与 kernel 内 software-pipeline stage 必须是两个不同的 IR 概念和 verifier contract。Workspace requirement 可以由 structured operation 提出，但具体 workspace representation 回到 value realization；structured-operation pass 不直接选择 target storage。

### 6.5 Physical granularity 是附着参数，不是第五种 kernel

program chunk、lane pack、reduction chunk、stream segment、matrix tile、sparse block、pipeline depth 等参数附着在前四组 decisions 上。例如 program chunk/lane pack 主要约束 execution，reduction chunk 主要约束 structured operation，stream segment 同时约束 execution 与 state realization，sparse block 同时约束 execution 与 access footprint，pipeline depth 则是 provider pipeline configuration。一个参数横跨多组时仍不形成新的 kernel 类别。

它们的来源可以不同：

- 由 source/KIR extent 或 primitive legality 唯一固定；
- 由 deterministic device heuristic 选择；
- 作为 provider tuner 的合法参数；
- 由 provider compiler 继续推导。

V2 必须逐项说明来源，不能把所有数值都叫 tuner，也不能把所有 granularity 都叫 tile。

### 6.6 非 tile workload 的四组覆盖

| KIR 结构 | Execution | Value | Access/validity | Structured operation |
|---|---|---|---|---|
| scalar/runtime loop | scalar instance 的 program/lane/serial binding | scalar SSA、replay 或 local state | scalar pointer/guard | 普通 control 保持原语义，不升级成块算法 |
| ragged | outer owner、member traversal、variable extent | member-local value 或 workspace | offset/index relation、invalid member handling | ragged 内部 reduce/contract 与 membership 组合 |
| ordered/state stream | 严格顺序、segment traversal、logical stop | carry、state buffer、跨 segment lifetime | partial segment validity | state transition、yield 和 final projection |
| sparse | row/block/member traversal | compressed value、metadata 和 accumulator form | dynamic gather/scatter、conflict/validity | CSR traversal 或 format-specific sparse primitive |
| effectful access | participant 与 happens-before 约束 | mutable/atomic state 的 lifetime 和 visibility | atomic/scatter physical access form | structured pass 只能消费 effect obligation，不能重写顺序 |

这张表不是为五类 workload 建模板，而是说明同一四组 decisions 可以组合覆盖非 rectangular、非 dense、非 tile-centric 的 GPU 程序。

---

## 7. 不属于结构 decision space 的对象

### 7.1 唯一 correctness lowering

下列内容通常不构成性能选择：

- KIR node/value 到 physical provenance 的绑定；
- 由已选 ownership/range 唯一推出的 index expression；
- 必要地址宽度；
- effect/state/ABI 保持检查；
- 已固定 boundary strategy 后唯一得到的 mask/guard；
- 已固定 stage slice 后派生的 dependency/input/output/terminal/lifetime。

它们可以由 analysis 或 lowering pass 计算，但不能被包装成“优化空间”。

### 7.2 Derived materialization

一旦 producer representation、consumer representation、primitive variant、dependency 和 visibility 已经固定，以下操作常常是确定性派生：

- 创建 target buffer；
- 插入高层 copy/conversion edge；
- 创建 accumulator；
- 兑现 synchronization obligation；
- 把 selected loop 标记为某种 provider pipeline region；
- 生成 target operation 的 operands/results。

这类步骤仍可以且常常应该成为 pass，因为它们改变 IR、形成 verifier 边界并供后续 pass 消费。但它们是 materialization/legalization pass，不应被误算成新的结构选择。

### 7.3 Provider tuner

Provider tuner 在 shared/provider decision passes 已经证明并声明的 legal parameter surface 上选择数值，例如 block/chunk、warp、thread、numeric pipeline depth、group 或 provider policy parameter。

它不允许改变：

- logical ownership relation；
- ordered/state semantics；
- stage grouping；
- boundary/effect semantics；
- 算法和数值角色。

手写 Triton/TileLang/cuTile source 同样可以拥有 tuner config，因此 tuner 本身不是 Intent compiler 的本质差异。

### 7.4 Lower compiler decisions

只要 target source/IR 已经提供足够事实，以下工作尽量交给 provider compiler：

- thread/warp/CTA layout；
- coalescing 和 vectorization；
- shared swizzle/padding/byte offset；
- register allocation 和 spill；
- 具体 MMA/MFMA/WMMA instruction；
- copy instruction selection；
- software-pipeline stage/order、buffer multiversion、barrier protocol；
- instruction scheduling 和 machine code。

---

## 8. 成熟 pass pipeline 的纪律

V2 的价值不来自“有很多 pass”这个形式，而来自 pass 变换的是一份完整 physical program，并且每一步有可审计的输入、输出和顺序。

### 8.1 每个 pass 边界都必须合法

V2 的目标是在 physical-program dialect 及其 verifier 定义完成后，使每个已发布的 pass 边界都是一份完整、合法、可验证的程序。不能用缺字段或“尚未决定”表示 pipeline 状态。

初始 conversion 可以产生保守但合法的默认 realization；其中 abstract execution/access/materialization form 必须是具有明确 SSA、region、effect semantics 和 verifier 的合法 operation，而不是 executable hole。只有 verifier 能证明语义和资源合法时才允许使用默认 realization；否则 conversion 必须明确拒绝，不能伪造一个“能跑”的 fallback。后续 pass 可以把合法默认值替换为更好的 realization。单个 pass 内部可以短暂处于未完成状态，但 pass 返回后必须通过当前阶段的完整 verifier，否则 pipeline 终止。

在这套 mature contract 中，不需要额外发明 `partial Plan` 或字段级 freeze 协议；当前 reference Plan 在 physical-program dialect/verifier 尚未建立前则不能声称已经满足这项保证。

### 8.2 没有不可变的字段唯一写入者

后续 pass 可以在自己的前置 legality、provider capability 和 semantic-preservation contract 内，重写前面已经合法选择的 execution、value、access 或 structured-operation representation。它不能任意改变自己无法证明保持的 shared obligations；provider pass 尤其不能反向改变 KIR semantics、shared logical ownership/range、compiler-private stage grouping 或跨 provider 仍成立的 materialization obligation。

纪律来自：

- IR verifier；
- analysis invalidation；
- pass ordering；
- legality/capability constraints；
- semantic preservation。

而不是“某字段永远只能由某一个 pass 写”的人为 ownership contract。Shared/provider 边界是 semantic contract，不是字段 owner 表；一项重写是否合法由它保持了什么语义和 capability 决定。

### 8.3 Analysis 与 IR 分离

Provenance、def-use、reuse、lifetime、axis relation、validity proof、effect summary 和 device-cost facts 应当实现为 analysis：

- 按需计算和缓存；
- 不修改被分析 IR；
- 默认假定会被任意 transform pass 失效；
- 只有 pass 明确证明 preserve 时才保留；
- 不作为 serializer 中的第二份真理。

真正影响后续 executable structure 的已选决定则必须 materialize 到当前 IR 的 type、attribute、operation 或 region 中。

### 8.4 Pass 顺序本身是设计对象

先决定 execution，还是先选择 provider primitive；先 bufferize，还是先规划 target pipeline，都会改变后续 pass 可观察的事实和合法空间。

V2 不要求用 fixed-point 证明收敛。像 Triton 一样，可以在 pipeline 的不同位置固定次数地重复 canonicalization、layout cleanup 或某个 refinement pass。

### 8.5 Target-specific facts 与 shared facts 存在于同一 current IR

Provider-local physical decision 不应隐藏在 source emitter 的调用栈中，也不需要另造一张秘密旁表。它可以通过 target-specific type、attribute、operation、interface 或 dialect extension 存在于同一 physical program 中。Provider extension 只能追加或细化 provider-local form，不能成为 shared semantic fact 的第二份 authority，也不能反向污染 shared legality。

NVIDIA/AMD 在 TritonGPU IR 中共享 core，同时拥有 NVIDIA MMA、AMD MFMA/WMMA encoding 和 target-specific passes，已经证明“共享 core + target extension”是成熟且可行的结构。

### 8.6 Translator 是终端阶段

最终 translator 只做：

- provider API spelling；
- symbol/name/source formatting；
- wrapper 与 compile/run 接线；
- 对已 materialized provider IR 的逐 op translation。

Translator 不再：

- 遍历 KIR 选择 ownership/range；
- 按 kernel name、shape、op count 或 whole-region matcher 套 schedule；
- 临时决定 replay/materialize/workspace；
- 创建未记录的 tuner axis；
- 根据字符串输出进度改变 physical structure。

---

## 9. V2 的概念 pipeline

下面是职责顺序，不是当前代码文件清单，也不要求每一行恰好对应一个 pass class。

### 9.1 KIR → baseline GPU physical program

输入是 canonical Kernel IR。conversion 构造第一份完整、保守、合法的 GPU physical program，保留：

- logical provenance；
- scalar/shaped value flow；
- structured control/state/effects；
- abstract physical execution scopes；
- abstract access、collective 和 materialization operations；
- source-visible region 与 scalar-instance 边界。

这些 abstract forms 必须已经有可执行语义和 verifier contract，而不是“以后某个 leaf 再猜”的占位符。它不必已经最优，但不能留下由 leaf 才能补上的 executable holes；对于无法保持 ragged、ordered/state、sparse、effect 或 provider capability semantics 的输入，conversion 必须明确拒绝。

### 9.2 Shared GPU analyses

按需提供：

- axis/region/provenance；
- def-use、reuse、lifetime；
- validity、neutralization 和 effect constraints；
- structured-op semantics；
- stage dependency；
- device capability/resource facts。

这些 analysis 是缓存，不是独立 Plan authority。

### 9.3 Shared GPU decision/refinement passes

根据第 6 节四组 decisions，确定性地改写 physical program：

- execution ownership/traversal；
- value replay/materialization/workspace；
- access footprint/validity realization；
- structured-op skeleton；
- compiler-private stage grouping；
- provider-independent legality requirements；
- granularity parameter roles 和约束。

这里的 shared 表示 decision 在多个 GPU provider 上仍然成立，不表示每个 provider 必须产生完全相同的 concrete storage 或 primitive。

### 9.4 Provider leaf decision/refinement passes

Provider leaf 读取同一 physical program，并加入只有该 provider surface 才有意义的具体 forms：

- capability rejection；
- target value/storage form；
- target primitive family/variant；
- target-specific access form；
- target pipeline eligibility/hints；
- provider-local alias、thread 或 runtime constraints。

每个 provider-local decision 都必须能指出它细化的是 execution、value、access/validity 或 structured-operation 中的哪一项 obligation；否则必须明确标为 legality/materialization，而不能在 leaf 中新增一项无来源的结构 policy。如果一项决定改变跨 provider 仍成立的 ownership、logical range、materialization obligation、stage grouping 或 algorithm semantics，它就不再是合法 leaf decision。

### 9.5 Provider materialization/legalization passes

在 provider form selection 之后，把已选 forms 兑现为完整 provider IR，例如：

- Triton pointer/mask/load/store/loop/dot/reduce/scan；
- TileLang buffers、copy edges、sync obligations、pipeline markers、`T.gemm` operands；
- cuTile tile/array/gather/collective forms。

这些 passes 可以很大，但代码量不等于独立 decision-space 大小。Bufferization 是 materialization 的 provider-specific 子类：它必须兑现前一阶段已经写入 IR 的 shape/dtype/scope，不能在 materialization 过程中重新选择 scope。

### 9.6 Provider translator

只序列化已经 legal 的 provider IR 为目标 source/entry/wrapper，然后交给 provider toolchain。完整顺序是：shared decision → provider form selection → provider IR materialization/legalization → terminal translation。

---

## 10. Triton 路径：薄的是 provider decision，不是整个编译过程

### 10.1 Intent 必须在 Triton 之前完成什么

Triton source/TTIR 要求源程序显式表达它实际依赖的高层程序结构：

- launch grid 由 Triton runtime/调用方提供；若 kernel 使用 `program_id`，program-instance mapping 和索引逻辑由源程序表达；
- KIR 所需的 range/loop/traversal structure；
- pointer arithmetic；
- load/store，以及需要边界或逻辑保护时的 mask/other validity semantics；
- 算法确实需要时的 accumulator/carry；
- `dot`、reduce、scan 等 operation；
- ordered/state/replay/workspace 等不能由 Triton 从 Intent KIR 猜回的结构。

`program_id`、mask 和 dot accumulator 在 Triton API 中都不是每个 kernel 的必选项；这里的要求是：凡是该 KIR realization 实际需要的 program decomposition、persistent traversal、K-loop、logical validity 或 state machine，都必须在 V2 physical program 中表达，不能期待 Triton 从缺失的 Intent semantics 中猜回。

### 10.2 Triton leaf 为什么可以很薄

一旦 shared physical IR 已经是完整 GPU program，Triton provider path 仍需执行 materialization pass，将 shared operations 变成合法 Triton operations：

- physical ownership → kernel 内 `program_id` mapping 和对应 launch-grid contract；
- shaped/lane range → `tl.arange`/loop；
- access/validity → pointer tensor、mask、`tl.load/store`；
- structured op → `tl.dot`、`tl.reduce`、`tl.associative_scan` 等合法 surface；
- tuner parameter → `num_warps`、`num_stages` 等 compile config/hint。

这些步骤读取并改写 Triton provider IR，不属于 terminal translator。Triton leaf 不需要普通 TileLang 式显式 bufferization，也不应由 Intent 生成 TritonGPU layout encoding、shared-memory byte allocation、MMA instruction 或 barrier schedule。

### 10.3 Triton 下层继续决定什么

TTIR→TTGIR conversion 会为没有 encoding 的 tensor type 建立合法初始 blocked encoding；coalesce、matmul acceleration、dot operand optimization、layout conversion cleanup、latency/schedule、pipeline、TMA、fence、MMA lowering 等 passes 继续反复改写同一 TTGIR。

所以最准确的结论是：

> **Intent→Triton 的 provider-local structural space 较薄，但 KIR→完整 GPU program 的 shared compiler work 并不薄。高质量结果依赖正确的 Intent program skeleton，加上 Triton 自己强大的 physical compiler。**

---

## 11. TileLang 路径：先选择 target form，再确定性 bufferize

TileLang source 比 Triton source 显式承载更多 memory/copy/pipeline structure。它要求 source 构造：

- `T.Kernel`/thread binding；
- shared/local/fragment/TMEM 等 buffer；
- `T.copy`/region transfer；
- `T.Pipelined` 或其他 loop annotation；
- `T.gemm`/reduction 等 buffer-oriented primitive。

因此 TileLang leaf 比 Triton leaf 多出的核心不是一个“通用内存 allocator”，而是两阶段工作：

1. provider-local form selection；
2. 对已选 forms 的 deterministic bufferization/materialization。

### 11.1 `T.gemm` 不唯一推出 storage form

TileLang GEMM 按 operand scope 至少区分：

- SS：shared/shared；
- SR：shared/fragment；
- RS：fragment/shared；
- RR：fragment/fragment；
- TS：TMEM/shared。

不同 MMA/WGMMA/TCGEN5 family 支持不同组合；warp policy、target、dtype、transpose 和 capability 又会约束合法 variant。具体 GEMM implementation 的 infer-layout hook 产生 variant-specific layout requirement，TileLang `LayoutInference` 再协调、传播 fragment/shared/TMEM layout；不能把所有 storage/layout 选择归给一个通用 LayoutInference pass。

因此 V2 不能写成：

```text
contract → T.gemm → 唯一 alloc_shared/alloc_fragment
```

正确关系是：

```text
contract semantics
+ execution/reuse/lifetime requirements
+ device/provider capability
+ selected GEMM family/variant
→ concrete operand/accumulator forms
→ deterministic buffers/alloc operations
```

`alloc` 语句本身通常是投影；选择哪一种合法 form 才是 provider-local physical decision。

### 11.2 TileLang bufferization pass 的准确职责

在 concrete value forms 已选后，V2 拟议的 bufferization/materialization pass 负责：

- SSA/shaped value 到 buffer identity；
- materialize 已选的 buffer shape、dtype、scope；
- producer/consumer region；
- 必要 copy/conversion edge；
- accumulator 与 intermediate buffer；
- alias/isolation 兑现；
- dependency 和 visibility 对 target IR 的显式化。

它不负责：

- allocation lexical placement；
- shared byte offset；
- layout/swizzle；
- pipeline buffer version 数；
- 具体 TMA/CPAsync/LDSM/STSM/TMEM copy instruction；
- register allocation。

其中 allocation lexical placement 由 TileLang `PlanAndUpdateBufferAllocationLocation` 等 transform 完成；layout、copy lowering、storage rewrite 和最终 register allocation 则分别由 TileLang 后续 passes 及硬件后端完成。这里描述的是 V2 provider bufferization 的拟议抽象，不暗示 TileLang 当前存在一个统一的同名 bufferization pass。

### 11.3 Copy：edge 可派生，instruction 交下层

当 producer/consumer forms 不同且不能 alias/replay 时，高层 copy/conversion edge 常由 def-use 和 representation mismatch 唯一推出。

V2 应优先生成普通、高层 `T.copy`，并提供 logical region、validity、capability requirement 和必要 hint。Copy edge 与 copy instruction selection 必须分层：`T.copy` annotation 可以约束 lowering，但 TileLang 的 target-aware copy analysis 仍根据 target、layout、alignment、stride、pipeline context 和 pass configuration，在 TMA、CPAsync、LDSM、STSM、TMEM 或普通 SIMT copy 中选择。

Provider pipeline legalization 也可以从普通 `T.copy`、selected stage 和 target facts 自动 materialize `tma_copy` 与 pipeline barrier；不要求 V2 预先显式选择 TMA。只有当 provider-local IR 明确选择 warp-specialized 或手写 split-phase protocol 时，显式 producer/consumer split、barrier ownership 等才成为更高层的 provider-local decision。TMA 仍只是 target mechanism，不是 V2 宏观编译空间的中心。

### 11.4 Barrier：记录义务，不抬高具体 primitive

KIR effects、selected execution participants、buffer scope 和 dependency 可以派生：

- producer/consumer happens-before；
- visibility scope；
- 哪些 participants 必须同步；
- 哪段程序不能重排。

V2 physical IR 应保存足够的 dependency/effect/synchronization obligation。普通 `T.sync_threads`、async wait、barrier slot/parity、多个 copy 的 barrier 合并和 pipeline barrier protocol，可以由 TileLang legalization/pipeline passes 按适用条件兑现；并非所有 barrier 都会被统一合并或 pipeline-version，例如由某些 primitive 内部管理的 barrier 仍遵守自己的 contract。

### 11.5 Pipeline：区分 region eligibility、深度和最终 schedule

V2 可以决定或标记：

- 哪个已经存在的 physical loop 可以作为 TileLang pipeline region；
- copy/compute dependency 是否允许 overlap；
- provider capability 是否允许该 realization。

`num_stages` 可以来自 tuner configuration。TileLang `PipelinePlanning` 再决定 software-pipeline stage/order/async grouping，`InjectSoftwarePipeline` 再完成 prologue/steady-state/epilogue、buffer multiversion 和适用 barrier 的 expansion；provider capability 也可以拒绝该 pipeline realization 并保留合法顺序 loop。

不能把 compiler-private multi-kernel execution stage 与 intra-kernel software-pipeline stage 混为一谈：前者是 V2 structural decision；后者主要由 provider compiler 完成。Tuner 只能选择已声明的 numeric pipeline depth/configuration，不能改变 compiler-private stage grouping、loop decomposition、dependency topology 或 barrier ownership。

---

## 12. cuTile 路径

cuTile 同样消费 shared execution、value、access/validity 和 structured-op facts，但使用自己的 program/grid、tile/array、load/gather/scatter、collective 和 primitive surface。

V2 对 cuTile 采用与另外两家相同的纪律：

- shared decisions 不能在 cuTile leaf 重新选择；
- cuTile-only gather/load/collective form 进入 target-specific IR/pass；
- target API spelling 和 wrapper 进入 terminal translator；
- layout、指令和低层机器实现交 cuTile toolchain；
- tuner axis 必须由正式 search surface 声明，不能在 emitter walk 后临时加入。

cuTile 可能在某些 operation 上比 Triton 更显式、在另一些 operation 上比 TileLang 更抽象。V2 不预设它与任一 provider 具有相同 pass 数量，只要求每项 concrete decision 有明确边界和 IR 位置。

---

## 13. 三家“决定数量是否一样”的最终回答

只能接受弱命题：

> 三家面对同一份 KIR 时，共享相同的 algorithm semantics 和 physical obligations，例如 ownership、validity、reuse、materialization、structured-op role 和 dependency。

不能接受强命题：

> 三家需要作出完全相同数量、相同粒度的 concrete physical decisions，只是打印详略不同。

原因是 provider source contract 不同：

- Triton 可以把 layout、shared staging、copy instruction 和大量 pipeline 细节交给 TTGIR/backend；
- TileLang source 在进入其 compiler 前必须已经有 concrete buffers/scopes、copy edges 和 buffer-oriented primitive；
- cuTile 又拥有自己的 native tile/collective/gather contracts。

因此，shared decision families 可以相同，provider-local decision space 和 materialization 工作量并不必然相同。

TileLang leaf 更重，既可能来自投影更详细，也可能来自 source boundary 确实要求 Intent 先选择 concrete form。不能把所有 unsupported 都归因于“只是打印代码多”。它还可能来自 provider capability subset、缺失 target-local decision pass 或 implementation gap。

---

## 14. 性能从哪里产生

V2 的性能不由“存在一条 pass pipeline”自动保证。真实性能来源分为五层。

### 14.1 Kernel IR 算法

作者决定 recurrence、numerical path、logical workset 和 structured algorithm。这决定性能上限和大结构。V2 不通过偷换算法补救错误 source algorithm。

### 14.2 Shared GPU decision passes

它们决定 execution decomposition、reuse、persistent/ordered traversal、materialization/workspace、access/validity 和 structured-op skeleton。写得不好会形成：

- 差的 program decomposition；
- 无效复用；
- 过量 materialization；
- 错误 workspace/stage boundary；
- 不适合 provider 的 structured program；
- 无法被下层优化的 access form。

它们的目标是自动复现优秀手写 GPU kernel 中关键的 execution、value、access 和 primitive structure；是否达到该目标仍取决于具体 policy，不能由 pipeline 形式本身保证。

### 14.3 Provider leaf decision passes

它们决定同一 shared obligation 在 provider surface 上的合法高质量 form，例如：

- Triton 的 pointer/mask/loop/primitive canonical form；
- TileLang 的 GEMM variant、buffer form、copy/pipeline region；
- cuTile 的 gather/load/collective form。

这些是 target-specific 性能知识的合法位置，但必须 materialize 到 provider IR，而不是隐藏在 source-emission call stack。

### 14.4 Materialization 与 translator 质量

即使结构 decisions 正确，错误的 bufferization、冗余 copy、非 canonical pointer expression 或不恰当 target API construction 仍可能造成数量级性能差距。

这类质量重要，但要与“重新选择一个物理结构”区分。V2 通过 provider IR、verifier 和独立 materialization pass 使两者可分别审计。

### 14.5 Provider tuner 与下层 compiler

Tuner 选择已声明参数；下层完成 layout、register、instruction、pipeline 和 machine code。Triton 路径尤其依赖强下层 compiler，TileLang 路径则同时强依赖 Intent 产生的 buffer/copy/primitive topology 与 TileLang 后续 passes。

V2 不保证所选 program 是所有合法实现中的全局最快，也不预先保证每条 canonical policy 都高性能。它像成熟 compiler 的 optimization pipeline 一样，用有明确依据、可审计和可替换的 typed rules 与 provider capabilities 产生一个 canonical realization，并以形成高质量 target program 为目标。

---

## 15. 为什么这仍然不是模板选择器

V2 可以使用 operation-local canonical patterns，也可以让某些 structured op 映射到成熟 primitive，但实现单位仍然是：

- typed axis/range；
- individual value and def-use；
- access/index relation；
- state/effect dependency；
- structured operation；
- provider capability。

它不是按 kernel name、whole-op signature、shape/dtype key 选择一份完整预写 source。

对简单 GEMM，V2 产生的 program 可能和一个 canonical TileLang/Triton template 非常相似；这不构成退化。判断标准是：

- 是否逐 value/access/op 组合；
- 是否能与 pointwise、ragged、state、scan、effect 等 KIR 结构组合；
- 是否由 typed facts 而不是 whole-kernel matcher 驱动；
- unsupported 是否在最小 capability/operation boundary 明确拒绝。

如果未来 leaf 只读取 whole-kernel signature 并跳到完整手写 kernel registry，才真正退化为 library selection。

---

## 16. V1 与 V2 的本质差别

下表的“当前 V1”来自现有 Driver/Lifecycle/Emitter 调用链和 Plan schema，是仓库事实；“成熟 V2”是拟议架构，不表示当前已经存在 physical-program dialect、provider IR 或相应 pass implementation。

| 方面 | 当前 V1 | 成熟 V2 |
|---|---|---|
| compiler state | Kernel IR + KIR-referenced Plan skeleton | 每阶段一份完整 executable physical IR authority |
| Plan shape | ID/attribute records 为主 | SSA/type/op/region 构成真实 physical program |
| construction | 一次性 facts + builder + KIR traversal | 显式 pass pipeline 反复合法 refinement |
| legality | 验证 KIR、Plan 引用和组合 | 每个 pass 边界验证完整 physical program 和 capability |
| analysis | 手工 C++ facts/index 生命周期 | 独立 analysis，默认失效、显式 preserve |
| decision update | 构造路径中一次性填写 | 后续 pass 可以合法替换前面决定 |
| leaf | 遍历 KIR、查 Plan、现场构造 source | provider passes 改写 physical IR；translator 只读 provider IR |
| target-specific policy | 部分隐含在 emitter 分支 | target-specific type/attr/op/pass 显式记录 |
| Triton | leaf 构造完整 block program | shared IR 已有 program；leaf 接近 target conversion |
| TileLang | leaf 现场 alloc/copy/sync/pipeline | provider form selection 后确定性 bufferize/legalize |
| tuning | Plan search space + 部分隐式 leaf config | 所有 tuner axis 正式声明且不得改变结构语义 |

V2 不是简单增加 pass 数量，而是让真正影响 target program structure 的决定都有一个可变换、可验证、可被后续 pass 观察的位置。

---

## 17. 当前资产如何理解

V2 不意味着当前工作全部作废。

可以保留和发展：

- canonical Kernel IR 和稳定 node/value identity；
- KernelModel、axis/ragged/state/contract 等 typed analyses；
- 当前 axis/range/ownership/reuse/persistent policies；
- transfer/padding/neutralization proofs；
- scan producer/materialization 与 stage grouping 知识；
- target capability 和 spelling assets；
- 三个 provider leaves 中已经验证过的 operation-local construction 经验。

需要重新定性的部分是：

- 当前 Plan fields 是未来 physical IR vocabulary 的种子，不是完整 executable IR；
- 当前 Build/Decisions 中的规则是未来 passes 的 policy 资产，不是成熟 pipeline 本身；
- 当前 leaf 大段代码中，一部分会成为 provider decision pass，一部分会成为 materialization pass，最后剩下的才是 translator；
- 当前 V1 `SurfacePlan`/cache 中保存的 derived facts 应转为正式 analyses，而不是继续与 Plan/KIR 共同形成第三份隐式 authority。

这个差距是结构性的，但不等于必须重写 frontend/KIR 或丢弃现有 policy。

---

## 18. V2 成熟态必须能回答的问题

对每一个会改变 target source structure 的当前分支，成熟 V2 必须能回答：

1. 它来自哪一条 KIR semantic fact？
2. 它是唯一 correctness/derived fact，还是多个合法 realization 中的选择？
3. 如果是选择，属于 execution、value、access/validity 还是 structured-op realization？
4. 它读取哪些 def-use、reuse、lifetime、effect、device 和 capability facts？
5. 它由 shared GPU pass 还是 provider leaf pass 决定？
6. 决定写入 physical IR 的哪个 type/attribute/op/region？
7. 哪个 verifier 保证它仍保持 logical workset、state、effects、ABI 和数值语义？
8. 后续哪个 pass 可以观察并替换它？
9. translator 是否只在打印，还是仍在偷偷决定结构？
10. 哪些参数明确交给 tuner，哪些 lower facts 明确交给 provider compiler？
11. 这项决定如何改变最终 source，为什么有机会形成优秀手写实现拥有的结构？
12. 它是否完全由 typed facts 驱动，是否存在 kernel name、op count、whole-region matcher 或未声明 fallback？

如果某项工作只能回答“leaf 里有一个 `if`，然后拼出一段高性能代码”，它还没有进入成熟 V2 compiler contract。

---

## 19. V2 的稳定主张与明确不主张

### 19.1 可以稳定主张

1. V2 编译的是同一 KIR 算法的 GPU physical program，而不是替换算法或选择完整 kernel。
2. 共同 GPU IR 是可执行 physical program，不是 KIR 旁边的 reference-only decision table。
3. 共同 IR 不以 tile 为唯一中心；tile 是可组合 physical granularity 之一。
4. 核心结构空间是 execution、value、access/validity、structured-operation 四组 decisions，加附着的 granularity parameters。
5. 对固定输入，V2 采用 deterministic canonical path；不要求 global cost model 或结构搜索。
6. 在 physical-program dialect/verifier 完成后，每个 pass 边界的 IR 始终完整合法；后续 pass 可以在 legality contract 内重写前面选择；analysis 默认失效。
7. target-specific decisions 进入同一 physical program 的 provider extensions/passes，不隐藏在 translator 中。
8. Triton provider-local path 可以较薄，因为 Triton 下层承担大量 physical inference；但 shared KIR→GPU program work 仍然重要。
9. TileLang 需要 provider-local form selection，再由 deterministic bufferization/materialization 产生 buffers、copy 和 sync obligations。
10. `alloc`、高层 copy edge，以及已选 dependency/effect obligation 所要求的普通同步，常是 forms/dependencies 的派生 materialization；具体 GEMM/storage/copy/barrier protocol/pipeline variant 不一定唯一。
11. 三家共享 semantic obligations 和 decision families，但 concrete provider-local decision 数量与抽象高度不必相同。
12. 性能来自算法、shared policy、provider policy、materialization 质量、tuner 和下层 compiler 的组合。

### 19.2 不能主张

1. 不能主张当前仓库已经实现成熟 V2 pass pipeline。
2. 不能主张当前 Plan 已经是一份独立 executable physical IR。
3. 不能主张 leaf 当前只做语法打印。
4. 不能主张四组 decisions 是四种 kernel 模板或穷尽了所有字段。
5. 不能主张 `T.gemm` 唯一决定 shared/fragment allocation。
6. 不能主张 copy instruction、barrier protocol 或 pipeline schedule 由 producer/consumer form 唯一决定。
7. 不能主张三家 concrete decision 数量完全相同。
8. 不能主张建立 pipeline 本身会自动提高性能；性能来自每个 pass 内部 policy 和 provider construction 质量。
9. 不能把 autotune、TMA、mask、program id 或单一 kernel 当成宏观架构的证明。
10. 不能把 target compiler 已经完成的 layout、register、instruction 和低层 pipeline 重新纳入 Intent shared compiler 以制造虚假的空间。

---

## 20. 最短答案

如果只保留一段：

> **Compiler Pass V2 把 canonical Kernel IR 转换为一份完整、始终合法、可持续改写的 GPU physical program IR。它不改变算法，也不以 tile 为唯一中心；deterministic passes 选择 execution、value、access/validity 和 structured-operation realization，granularity 只是附着参数。Shared GPU passes 形成跨 provider 仍成立的 physical obligations，provider passes 再选择并 materialize Triton、cuTile 或 TileLang 所需的 concrete forms，terminal translator 只序列化 provider-legal IR。Triton provider decision 可以较薄，TileLang 则需要 target-form selection 与 bufferization；最终质量取决于这些 policy、materialization、tuner 和 provider compiler 的共同作用，而不是 pass 数量。**

---

## 21. 主要事实依据

当前 IntentDSL：

- `doc/compiler/kernel-ir.md`
- `doc/compiler/architecture.md`
- `doc/compiler/physical-plan.md`
- `doc/compiler/backend-lowering.md`
- `include/Intent/Dialect/Plan/IR/PlanOps.td`
- `lib/Target/GPU/Realization/Plan/Decisions.cpp`
- `lib/Target/GPU/Realization/Plan/Build.cpp`
- `lib/Target/Common/Emission/Driver.cpp`
- `lib/Target/Common/Emission/Lifecycle.cpp`
- `lib/Target/{Triton,CuTile,TileLang}/Emission`
- `report/compiler-space-v1.md`

Triton 对照：

- `ref/triton/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp`
- `ref/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp`
- `ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`
- `ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td`
- `ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`
- `ref/triton/third_party/nvidia/backend/compiler.py`
- `ref/triton/third_party/amd/backend/compiler.py`

TileLang 对照：

- `ref/tilelang/tilelang/language/allocate.py`
- `ref/tilelang/tilelang/language/copy_op.py`
- `ref/tilelang/tilelang/language/loop.py`
- `ref/tilelang/tilelang/tileop/gemm/gemm_base.py`
- `ref/tilelang/tilelang/cuda/op/gemm`
- `ref/tilelang/src/transform/layout_inference/layout_inference.cc`
- `ref/tilelang/src/transform/pipeline_planning.cc`
- `ref/tilelang/src/transform/inject_pipeline.cc`
- `ref/tilelang/src/transform/plan_update_buffer_allocation_location.cc`
- `ref/tilelang/src/transform/storage_rewrite.cc`
- `ref/tilelang/src/cuda/op/copy_analysis.cc`
