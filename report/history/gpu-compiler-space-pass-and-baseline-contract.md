# 从 Intent Kernel IR 到 Triton：真实 GPU 编译空间、Pass 边界与 Baseline 合同

## 0. 这份报告回答什么

这份报告重新回答三个问题：

1. 作者算法固定以后，Intent 到底还需要编译哪些东西；
2. 这些工作分别应存在于 Kernel IR、GPU physical IR、Triton provider IR、Triton 下层编译器和 runtime autotuner 的哪一层；
3. generated Triton 与手写 Triton source 怎样形成公平、能归因的性能对照。

结论只依据两类事实：当前 IntentDSL 实现，以及 `/home/kingdom/phdworks/ref/triton` 中真实 Triton 编译器实现。`compiler-pass-v2.md` 只作为被核查的历史设计，不作为真理；`doc/` 不作为事实依据。

这次审计撤回五种过度简化的说法：

- 不能把固定算法后的全部工作统称为选择一份“physical structure”；其中包含程序实例映射、块内迭代、物理 SSA、访问图、structured-op lowering、存储和同步义务、provider surface、下层分布布局与机器 lowering，它们不是同一种对象。
- 不能把作者写下的算法结构和 lowering 以后产生的 physical op/loop topology 混称为“程序结构”。canonical KIR 中硬件无关的控制、状态、索引与数值合同仍是作者权威；独立 physical IR 才可以在证明等价后改写 op graph、循环、布局、流水和执行拓扑。
- 不能说只有“不改变代码结构的数值参数”才属于 autotune。Triton 的 `num_warps`、`num_stages`、`num_ctas` 会直接驱动布局、流水和 CTA 结构变换。
- 不能拿外部 Triton 编译器内部的 NVIDIA/TTGIR passes，为 Intent 中厚重的 `Target/Triton` 路径背书。前者是我们已经委托的下层编译器；后者原则上只应兑现 shared physical program 到 Triton DSL surface 的真实差异。目录只说明代码位置，不能证明职责合法。
- Intent 的当前目标产物是 Triton Python DSL source，不是 TTIR 或“TTIR-like IR”。TTIR/TTGIR 是外部 Triton frontend 接收这份 source 后才建立的下层程序。

最重要的总判断是：

> Intent 当前已经有正式 pass pipeline 和大量有价值的事实、规则与 capability 资产，但尚未拥有一份自足的 executable physical IR。当前所谓 Physical Program 是一个真实的 MLIR 容器，却主要由 KIR 同构副本和 side records 组成；真正的 Triton block program 仍在字符串 materializer 中现场构造。因此目前最缺的不是更多局部 heuristic 或更多 provider attribute，而是把程序实例、物理 value、访问和控制真正变成可验证、可重写的 IR。

### 0.1 Compiler V2 到底有没有进入主链

已经进入，而且不是空壳命名：

- `ProgramOp` 成为唯一 physical entry 的容器，旧的双 executable authority 已被移除；
- construct、shared refinement、provider realization、verification、materialization 和 terminal translation 已经按 pass manager 串成唯一主链；
- Axis/Range/Launch/Transfer/Contract/Buffer 等决定已经是正式 Plan dialect operations；
- provider materialization 与最终字符串输出已经分开，最后的 translator 确实只输出 `TargetProgramOp.source`。

但是，V2 最核心的三个目标尚未落地：

- `exec_*` 没有形成 physical types、program-id、pointer/access、mask、physical loop 和新的 def-use，只是把 KIR op 改名并复制；
- shared passes 大多只改 Axis/Range/Launch/Transfer 等 records，没有改写 executable function；
- provider-legal program IR 不存在，三家 materializer 仍结合 KIR、KernelFacts 和 side records 现场合成完整目标程序。

所以，“Compiler V2 根本没有放进去”是错的；“Compiler V2 已经完成”也同样是错的。准确定位是：**V2 已完成单一主链、pass/verifier 和 decision-record 骨架，但尚未完成 executable physicalization 与 provider-legal program。**当前实现之所以能生成并运行，并不是因为 physical IR 已经闭合，而是厚 materializer 仍承担着隐藏编译器的职责。

---

## 1. 先划清真正不能动的边界

### 1.1 canonical KIR 不动；独立 physical program 可以等价变换

Kernel IR 应当继续作为下面这些内容的唯一语义权威：

- runtime ABI、输入输出、alias 与 observable effects；
- logical workset、domain、ragged/sparse membership 与 logical identity；
- runtime control condition、state transition、carry、stop 与顺序依赖；
- logical indexing 和数值路径；
- reduce/scan/contract 的 combiner、axes、identity、accumulator dtype 与精度合同；
- exact/approximate math、tie-break、atomic/conflict 等会改变结果的约定。

这些事实不仅要求“结果数值差不多”，还要求 KIR 中硬件无关的算法分支、状态推进、effect 顺序和 kernel 边界继续由作者决定。设备型号、provider capability、TMA/MMA/layout 选择等硬件分支不得进入 canonical KIR。

另一方面，canonical KIR 保持不变，不等于 lowering 后的独立 physical program 必须逐 op 同构。真实 Triton 会在它自己的 TTIR/TTGIR 上进行下面这些等价变换：

- `CombineBroadcastMulReducePattern` 把 broadcast-multiply-reduce 子图替换成 `tt.dot`：[Combine.cpp](../../ref/triton/lib/Dialect/Triton/Transforms/Combine.cpp#L110)；
- dot-add 被合并成带 accumulator 的 dot：[Combine.cpp](../../ref/triton/lib/Dialect/Triton/Transforms/Combine.cpp#L249)；
- software pipeliner 生成 modulo schedule，并把原 loop 改成 prologue、新 steady-state loop 与 epilogue：[SoftwarePipeliner.cpp](../../ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp#L20)；
- CLC pass 把整个 kernel body 搬入 `scf.while`，再用运行时取得的 logical program id 替换原 `tt.get_program_id`：[ToCLC.cpp](../../ref/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ToCLC.cpp#L53)。

所以正确纪律是：

1. canonical KIR 本身不被偷偷改写，始终保留作者算法；
2. physical passes 可以重写独立的 physical IR；
3. 每次重写必须能说明保持了哪些 KIR 语义；
4. 不能证明语义等价的变换就不能做。

例如，把显式 pointwise-multiply + associative reduce 变成 dot 可能合法，但前提是 dtype、累加、reassociation 与数值合同一致；不能只看形状像 GEMM 就替换。把 ordinary ordered recurrence 变成无序归约则不合法，因为 state/order semantics 已改变。

### 1.2 一个 kernel 的边界仍然是一份 launchable program

一个 `@intent.kernel` specialization 对应一个 target kernel artifact 和一次正式 launch。一个 kernel 内可以有属于硬件无关算法的 runtime/constexpr `if`、`for`、`while`、state、reduce、scan 和 contract；这些都不违反 single-kernel 边界。persistent traversal、软件流水和 provider capability branch 则是 physical/lower-compiler structure，不是作者为硬件写进 KIR 的算法分支。

Triton 的 CLC、warp specialization 和 software pipeline 都是 kernel 内重写，没有把一个 kernel 自动裂成多次 host launch。Intent 因此也不应恢复 compiler-private multi-launch stage、隐藏 workspace 或隐藏跨 kernel 同步。

如果 upstream 是两个 `@triton.jit` kernel 加 Python wrapper，Intent 应写两个 `@intent.kernel` 并由 Python wrapper 编排。若 upstream 的一个 kernel 内存在硬件无关的算法 compile-time/runtime `if`，Intent 应在一个 Kernel IR 中表达同样的条件结构，而不是删掉分支，也不是把它误解为多个 kernel。若该 `if` 只是在选择 TMA、MMA、layout 或设备能力，则它属于 physical/provider/lower-compiler 路径，不能照抄进算法 KIR。

### 1.3 Source 算法不同不是 compiler performance 问题

有 upstream baseline 时，Intent example 必须先对齐：

- 调用数和计时范围；
- control flow、state、logical indexing；
- dtype、近似函数、累加与输出合同；
- visible intermediate/workspace ABI。

如果不同，应直接改 example，使作者层写成同一算法。只有现有 Core 确实表达不了时，才是 DSL 能力缺口。编译器不能用 physical pass 把错误 example 偷换成 upstream 算法。

---

## 2. 真实 Triton 编译器究竟编译什么

### 2.1 Triton 的输入已经是一份静态 block program

Triton Python frontend lower 出来的 TTIR 不是 Intent 这种逻辑 region 算法。它已经显式包含：

- program instance：`tt.get_program_id`，[TritonOps.td](../../ref/triton/include/triton/Dialect/Triton/IR/TritonOps.td#L624)；
- 静态 ranked tensor/block shape；
- pointer tensor、pointer arithmetic、`tt.load`/`tt.store` 和 mask；
- `tt.make_range` 形成的块内 lane/index tensor；
- `tt.dot`/`tt.dot_scaled`；
- 带 typed combine region 的 `tt.reduce` 和 `tt.scan`，[TritonOps.td](../../ref/triton/include/triton/Dialect/Triton/IR/TritonOps.td#L758)；
- SCF/control flow、descriptor、gather、atomic 等 provider program structure。

也就是说，手写 Triton 作者在 source 中已经回答了以下问题：

- 一个 program instance 负责哪一块逻辑工作；
- block tensor 是什么形状；
- 块内索引怎样构造；
- 每次 load/store 的 pointer 与 mask 长什么样；
- 哪些循环和 structured primitive 出现在 kernel body 中。

Triton 编译器不会从一份无 program id、无 block shape、无 pointer/access body 的逻辑算法重新发明这些内容。它从 TTIR 这个已经成形的 block program 开始继续优化。

### 2.2 TTIR passes 会改写运算图

NVIDIA backend 的 TTIR pipeline 包含 inlining、descriptor fallback、canonicalization、combine、broadcast reorder、CSE/DCE 与 loop unroll：[compiler.py](../../ref/triton/third_party/nvidia/backend/compiler.py#L273)。

这里的 `combine` 不是字符串清理，而是语义受控的 operation-graph rewrite。它说明成熟编译器的算法边界不是“保留每一个原 op”，而是“只做 TTIR semantics 允许的等价 canonicalization/strength reduction”。

### 2.3 TTIR → TTGIR 首先把分布布局变成类型事实

TTIR→TTGIR conversion 给未编码 ranked tensor 增加默认 `BlockedEncodingAttr`，并通过 `ttg.convert_layout` 做类型 materialization：[TritonGPUConversion.cpp](../../ref/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp#L25)。转换还专门处理 gather/scatter index layout，而不是把 layout 留到打印源码时猜。

TTGIR 的 distributed encoding 是实际 IR 类型/属性：

- `LinearEncodingAttr` 明确 register/lane/warp/block 到 logical dimensions 的映射：[TritonGPUAttrDefs.td](../../ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L665)；
- `BlockedEncodingAttr` 明确 thread/warp/CTA 对 tensor elements 的所有权：[TritonGPUAttrDefs.td](../../ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L738)；
- MMA、dot operand、slice 和 shared encodings分别表达矩阵 primitive 与共享存储布局。

这些不是 KIR 旁边的记录表。它们直接附在当前 executable values/types 上，任何后续 pass 都消费当前 IR，而不是回到 Python source 重新猜。

### 2.4 TTGIR passes 反复重写 layout、memory、primitive 和 loop

真实 pipeline 的核心步骤包括 coalescing、thread locality、matmul acceleration、dot operand optimization、loop fusion、warp specialization、latency/schedule、software pipeline、async-copy coalescing、TMA lowering、layout cleanup、fence 与 MMA lowering：[compiler.py](../../ref/triton/third_party/nvidia/backend/compiler.py#L289)。

几个具体例子说明这些 pass 的成熟形态：

- `CoalescePass` 先运行 `ModuleAxisInfoAnalysis`，为每个 memory op 选择 coalesced encoding，然后插入 layout conversion 并替换 memory op：[Coalesce.cpp](../../ref/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp#L77)。
- `AccelerateMatmul` 把 blocked dot 的 result/operands 改成 NVIDIA MMA、dot-operand、shared/TMEM representations，并插入 `local_alloc`、`convert_layout` 和 target MMA op：[AccelerateMatmul.cpp](../../ref/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L430)。
- TTGIR 本身拥有 `convert_layout`、async global-to-local copy、local alloc/load/store、warp-specialize regions 与 barrier ops：[TritonGPUOps.td](../../ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td#L32)。
- software pipeline 直接重写 `scf.for`，而不是在一个旁表里写 `pipeline=true`：[SoftwarePipeliner.cpp](../../ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp#L93)。

成熟的关键不在于 pass 数量，而在于：决定一旦作出，就成为当前 IR 的 type/op/region/def-use，后面的 pass 能看见、验证、替换和清理它。

### 2.5 Triton 内部 passes 不能为 Intent 的厚 leaf 提供类比

真实 Triton 有大量 NVIDIA-specific dialect 和 passes，例如 CTA planning、TMA/TMEM、CLC、fence、MMA lowering。它们位于外部 Triton compiler 内部，消费 TTIR/TTGIR，并负责 NVIDIA lower compiler 已经承诺承担的 representation、layout、pipeline 和 machine lowering。

Intent 当前输出的是 Triton Python DSL source，然后把它交给这套外部 pipeline。因此，Triton 内部拥有很多 passes，不能推出 Intent 也应在 `Target/Triton` 再建立一套同等厚度的 passes；那会把已经委托出去的编译层复制一遍。

Intent 的 Triton 路径只有在下面两种情况下才需要 pass：

- 把 shared executable physical program 中已经确定的结构，合法化成 Triton DSL 能直接表达的 program-id、static block value、pointer/mask、loop 和 primitive；
- 处理只在 Triton DSL surface 上存在、且外部 Triton frontend 无法从普通 DSL 程序自行选择的 capability/form 差异。

任何从 KIR、shape、relation、role 或周围结构重新推导 axis、range、boundary、ownership、value flow 的 Triton pass，都是 shared executable program 尚未闭合后的补洞。当前 `Target/Triton` 的问题不是文件名叫 pass，而是这类补洞占了主体；完成重构后它应显著收缩，而不是以 Triton 内部 TTGIR pass 的规模为目标。

### 2.6 Autotune 参数会索引下层结构变换

Triton `Config` 不只是普通标量字典。它正式包含：

- `num_warps`：决定一个 kernel instance 的协作线程数；
- `num_stages`：控制 compiler software-pipeline stages；
- `num_ctas`：控制 cluster 中 CTA 数；
- `maxnreg` 与可选 IR override。

定义见 [autotuner.py](../../ref/triton/python/triton/runtime/autotuner.py#L328)。`num_warps` 被 TTIR→TTGIR conversion 用于初始 distributed layout，`num_stages` 被 latency/schedule/pipeline passes 消费；CUDA options 中的 `clc` 还会决定是否运行 `ToCLC`：[compiler.py](../../ref/triton/third_party/nvidia/backend/compiler.py#L289)。

所以“结构选择”和“autotune 参数”不是互斥类别。一个 autotune candidate 可以索引不同的 lower-compiler structure。正确责任划分是：

- 哪些 program facts 必须在进入 Triton 前显式；
- 哪些 provider forms 必须由 Intent→Triton conversion 选择；
- 哪些 lower transformations Triton 已经能由 config/capability完成；
- winner 由 Triton autotuner 实测选择。

---

## 3. Intent 比 TTIR 高在哪里

Intent KIR 的 `domain`、`parallel`、`state_stream`、`if/for/while`、logical load/store、reduce/scan/contract 等 op 描述逻辑算法和区域程序：[IntentOps.td](../include/Intent/Dialect/Intent/IR/IntentOps.td#L47)。它刻意不要求作者写：

- `program_id` 与 launch grid；
- static block tensor shape；
- physical lane/warp ownership；
- pointer tensor 与 broadcasted address expression；
- load/store mask 的具体张量形状；
- shared/TMEM allocation、layout conversion、MMA variant；
- lower software-pipeline schedule。

因此 Intent 的真实编译责任不是模糊的“找一份 physical structure”，而是先完成 Triton 作者已经手写、Triton 编译器又不会替作者补的那一段：

```text
Intent logical/region kernel
    → executable GPU physical program
       （program-instance mapping、physical loop/value/access/structured op）
    → provider-legal Triton DSL program
       （tl.program_id/arange/load/store/dot/reduce/scan/control）
    → 确定性序列化为 Triton Python DSL source
    → 外部 Triton frontend/TTIR/TTGIR/NVIDIA compiler pipeline
```

Intent 的编译责任到 Triton DSL source 为止。外部 Triton frontend 负责由 source 建立 TTIR；此后从 TTIR 到 TTGIR、layout、MMA、pipeline 和机器代码的步骤也已经由 Triton 成熟实现拥有，不应复制。当前代码尚未实现上面中间两份自足程序：它从 KIR clone 与 side records 直接跳到字符串 materializer。

---

## 4. 固定算法以后，真实存在的十个层面

下面这些层面共同决定最终性能，但它们不能再被压成一个“Physical Program structure”。这里的“层面”不要求一层恰好对应一个 dialect 或一个 pass；它要求每类事实有清楚的 IR 主体、分析来源和责任方。

### 4.1 语义与 provenance

内容：logical domains、value/effect/state semantics、structured-op contract、ABI、数值合同。

权威来源：canonical KIR。

允许的编译行为：分析、证明、引用；不能建立第二份可分叉的算法真理。

### 4.2 Logical work → program-instance mapping

内容：哪些 logical instances 由哪个 grid dimension/program instance 拥有，多个 logical axes 如何线性化、折叠、group/swizzle，是否形成 grid-stride/persistent worker traversal。

这是 Triton source 中 `program_id`、grid lambda 和 program index arithmetic 已经回答的问题。Intent 作者没写，所以必须由 Intent physicalization 显式形成。

### 4.3 Program 内 iteration 与 fragment extent

内容：ownership tile、lane range、ordered/serial chunk、reduction chunk、stream segment、tail padding，以及这些层次的嵌套关系。

这些选择决定 body 中出现 `tl.arange` 还是 scalar loop、tensor 的 static shape 是多少、一次 program 看见多少 logical instances。它们必须进入 physical types/loops/regions，不能只作为字符串名字留在 side records。

### 4.4 Physical SSA 与 value representation graph

内容：shaped/scalar SSA、loop carry、accumulator、replay/rematerialization、materialized intermediate、logical/private buffer、sharing scope 与 lifetime。

这一层可以与 KIR def-use 不同。例如 pure producer 可以合法重算，dot accumulator 可以融合，某个 logical tensor 可以成为 fragment 或显式 buffer；但是 physical def-use 必须真实存在，不能由 emitter 临时拼出来。

### 4.5 Access graph

内容：base pointer/descriptor、index expression、physical footprint、stride、gather/scatter、mask/other、validity、neutralization、conflict/atomic obligation。

这一层不只是“transfer form”字符串。成熟 IR 应有明确的 access operands/results 和 relation，使 coalescing、descriptor legality、bulk transfer 与边界证明能够直接消费当前程序。

### 4.6 Structured-operation realization

内容：contract/reduce/scan/state/sparse op 采用什么 physical primitive/loop skeleton、participant topology、accumulator flow 与 operand representation。

这里既可能是唯一合法 lowering，也可能有多个等价 realization。它允许把 physical op graph 改成 dot/MMA-friendly form，但不能改变 KIR 的 combiner、state transition 或精度合同。

### 4.7 Intra-kernel schedule、storage 与 synchronization obligation

内容：persistent loop、pipeline eligibility、async dependency、buffer lifetime、visibility、必要 synchronization，以及 provider 是否能 materialize。

这层必须继续拆分：跨 provider 成立的是 dependency/lifetime/visibility obligation；TMA、TMEM、warp-specialize region、barrier protocol 等是 provider/lower-compiler representation。不能因为都与“流水”有关就放进一张 shared flag 表。

### 4.8 Provider-legal program

对 Triton，它应当是一份能一一序列化为 Triton Python DSL 的 provider program：明确的 program mapping、静态 block tensor values、pointer/descriptor accesses、mask、loops、reduce/scan/dot 和 control flow。它不是 TTIR；真正 TTIR 由外部 Triton frontend 从生成的 Python DSL 建立。到这一层后，Triton conversion/materializer 不应再遍历 KIR 重建 access geometry 或 program ownership。

### 4.9 Provider lower compiler IR

对 Triton，这是 TTGIR 与 TritonNvidiaGPU IR：distributed/shared/TMEM encodings、layout conversions、async copies、warp-specialize regions、MMA/TMA/barrier operations。

它们由外部 Triton compiler 构造和反复重写。Intent 不应复制这层，只需生成让 Triton frontend 能正确建立 TTIR、并充分暴露下层优化机会的高质量 Triton DSL program。

### 4.10 Runtime configuration 与 winner selection

内容：meta tile/chunk values、`num_warps`、`num_stages`、`num_ctas` 以及双方都能合法暴露的 provider options。

Intent 可以声明语义角色和 legality constraints；Triton autotuner 对 candidates 编译、运行并选 winner。这个过程可能触发不同 lower IR，但 winner 仍不属于 Intent shared pass。

---

## 5. 当前 Intent 实现的准确状态

### 5.1 `exec_*` 仍是 KIR 同构副本

当前 `intent_plan.exec_*` 定义明确写着“mirror the algorithmic arity and region contracts”，并大量使用 `AnyType`：[PlanOps.td](../include/Intent/Dialect/Plan/IR/PlanOps.td#L53)。

实际 conversion 只是：

1. 把 `intent.foo` 改名为 `intent_plan.exec_foo`；
2. 原样复制 operands、result types、attributes 和 regions；
3. 添加 `intent_plan.source_op` provenance。

代码见 [Operations.cpp](../lib/Target/GPU/Realization/Plan/Operations.cpp#L15)。这一步没有建立 program id、physical tensor encoding、pointer/mask access op、physical loop 或 provider value representation。

因此，当前 physical function 虽然是正式 MLIR function，却仍然保持 KIR 的控制/SSA 拓扑和类型。它不是 Triton TTIR 意义上的 block program，更不是 TTGIR 意义上的 distributed program。

### 5.2 Physical decisions 主要是 function 旁边的 side records

`ProgramOp` 的 body 同时包含 cloned physical function，以及 `AxisOp`、`RangeOp`、`LaunchOp`、`BufferOp`、`TransferOp`、`ReductionOp`、`ScanOp`、`ContractOp`、`StreamBindingOp`、`AutotuneOp` 等无 SSA 结果的 decision records：[PlanOps.td](../include/Intent/Dialect/Plan/IR/PlanOps.td#L224)。

`buildPhysicalProgram` 先生成这些 records，再 clone KIR function 并机械替换 op 名：[Build.cpp](../lib/Target/GPU/Realization/Plan/Build.cpp#L611)。所以当前真实 authority 是：

```text
exec_* function 保留算法 graph
    + side records 保存部分物理答案
    + KIR provenance/metadata 供后续重新推导
```

这比早期 Python Plan 强得多，但仍不是一份自足的 executable physical IR。

### 5.3 Shared passes 大多修改 records，而不是物理程序

当前 pipeline 已正式拆成 construct、automatic blocking、access range、transfer、contraction、scan、value、private residency、persistent、boundary 与 search-space passes，并在多处插入 verifier：[Passes.cpp](../lib/Target/GPU/Transforms/Passes.cpp#L65)。这证明 pass 调度骨架已经接入主链，**不证明 executable Physical Program 已经形成**。

按 mutation 对象逐个核对，11 个非 verifier shared transforms 中，只有 `ConstructPhysicalProgram` 触及 executable op graph，而它做的只是 clone function、把 `intent.*` 机械改名成 `intent_plan.exec_*`。其余十个 transform 修改或重建的是 Axis/Range/Launch/Transfer/Contract/Buffer/SearchSpace 等 records/attributes；没有一个创建 program-id、physical loop、pointer/mask access 或新的 physical SSA graph。

但具体行为显示，很多 pass 还没有改变 executable program：

- automatic blocking 重新运行 `assignAxes/emitPhysicalDecisions`，删除旧 Axis/Range/Launch records 再生成一组新的 records：[Decisions.cpp](../lib/Target/GPU/Realization/Plan/Decisions.cpp#L1261)；
- transfer pass 把 result/coverage space 固定成 scalar/fragment/direct 字符串：[TransferRealization.cpp](../lib/Target/GPU/Transforms/Access/TransferRealization.cpp#L14)；
- value pass 把 reduction、scan carry、pointwise 与 sparse spaces写成固定字符串：[OperationRealization.cpp](../lib/Target/GPU/Transforms/Value/OperationRealization.cpp#L13)；
- persistent pass 只把 `LaunchOp.persistent` 与 Axis 的 worker/fold attrs 改掉，没有在 physical function 中创建 persistent loop：[PersistentTraversal.cpp](../lib/Target/GPU/Transforms/Execution/PersistentTraversal.cpp#L67)。

其中 contraction realization 的 def-use/replay/accumulator 分析比其它 pass 更实质，但输出仍主要落到 `ContractOp`/`TransferOp` records，而不是重写 physical operands、results 和 loops。

所以不能用“已经有多个 pass 和 verifier”推导“shared physical compiler 已闭合”。当前 passes 中混有四种性质：

- 从 KIR 重算语义事实；
- 唯一 correctness/materialization 分类；
- 真正的多个合法选择；
- 固定默认值或经验 heuristic。

它们都可以是 pass，但只有第三类构成可选择的性能空间；最重要的是，四类结果都还没有充分进入 executable physical graph。

逐个看当前 pipeline，其真实性质是。表中除明确说明外，“输出”都是 record-level decision/attribute，不是 executable op graph rewrite：

| 当前 pass | 当前真正修改的载体 | 主要性质 | 尚未闭合之处 |
|---|---|---|---|
| ConstructPhysicalProgram | 建 side records，clone KIR 并改名为 `exec_*` | conversion 骨架 | 没有形成 program mapping、physical types/access/control |
| AutomaticBlocking | 从 KernelFacts 分配 axis roles、tile parameter names、ownership/traversal ranges，写入 Axis/Range/Launch records | shared physical policy | 未改写 function、program mapping 或 physical loops |
| AccessRanges | 把已选 execution range 投影到 transfer footprint | derived projection | footprint 仍是 record，access op graph 不存在 |
| TransferRealization | 按 scalar/tensor 和 compact indexing 填 `direct`、result/coverage space | fixed legal default | 没有生成 pointer/descriptor/mask/value |
| ContractionRealization | 分析 replay、deferred producers、accumulator flow 与 physical reduction axis | 混合 semantic analysis、真实 value-form choice和derived binding | 结果仍是 IDs/strings；accumulator/def-use 没有成为 physical SSA |
| ScanRealization | 依据 consumer 选择 fragment 或 workspace，并记录 producer/materialized value IDs | value materialization choice | workspace access与scan result graph 留给 materializer |
| ValueRealization | 为 reduction/scan carry/pointwise/sparse 填固定 spaces | conservative default | 大部分路径不是经过比较的 representation policy |
| PrivateBufferResidency | 用 shape、dynamic access 与 device register budget选择 scalar-array/vector/workspace，写入 Buffer records | 真实但粗粒度的 device-aware policy | 具体 value/lifetime/storage graph尚未形成 |
| PersistentTraversal | 用 contraction、parallel axes、ragged 与 multi-tile 条件选择 persistent，修改 Launch/Axis records | structure policy | 未创建 persistent loop；Triton 已有 CLC lowering，委托边界还需收敛 |
| BoundaryNeutralization | 证明 consumer 已中和越界值，删除冗余 padding records | derived proof + legality simplification | mask/other/access op仍到 provider materializer才存在 |
| SearchSpace | 从 tile/group role收集 provider parameters | candidate-interface construction | provider forms和Python profile又在后面追加候选，合同不唯一 |

这张表不否定已有 pass 的价值。它说明下一阶段不能继续以“再拆一个 pass 文件”为完成标志；同一个 pass 的 policy 可以保留，但输出载体必须从旁表逐步变成真实程序变换。

### 5.4 Analysis 仍把 physical function 当作 KIR 重新建模

`PhysicalProgramAnalysis::compute` 对 physical entry 重新运行 `analyzeKernel` 和 `analyzeOperations`，建立新的 `KernelModel/KernelFacts`，然后再索引 side records：[PhysicalProgram.cpp](../lib/Target/GPU/Transforms/Analysis/PhysicalProgram.cpp#L91)。

这本身不一定错误——KIR provenance 可以支持语义分析——但当前 physical exec ops 与 KIR 同构，意味着后续 passes 经常同时依赖：

- cloned algorithm graph；
- 重建的 KernelFacts；
- side records。

三者共同决定 emission，仍然存在多重解释源。成熟形态应当是：KIR/KIR-derived analysis提供 semantic constraints，当前 physical IR 提供唯一 executable topology；analysis 不能替 physical IR 保存尚未 materialize 的程序结构。

### 5.5 Triton provider pass 仍在重新理解 KIR

`Target/Triton/Lowering/Passes.h` 中的属性可以分成三类：[Passes.h](../include/Intent/Target/Triton/Lowering/Passes.h#L12)。

本应是 shared/KIR-derived facts：

- contraction orientation、batched semantics；
- reduction axis/all-axes；
- ragged compact/indexed route；
- stream logical boundary 与 neutral masks。

在目标架构中确实需要由 Triton conversion 兑现的 surface forms：

- pointer 还是 pointer/descriptor candidate；
- descriptor block axes/layout；
- `tl.dot`、`tl.reduce`、`tl.scan`、gather 的 Triton surface form；
- row launch 或 Triton-specific pipeline candidate form。

当前这些 forms 还没有成为一份可独立验证的 Triton provider IR；它们主要以 records/attributes/candidate names 存在，随后仍由 materializer 结合 KIR 现场构造 Triton DSL source。因此这份列表是职责分类，不是“当前 Triton leaf 已经成熟”的清单。

需要拆开判断的 mixed facts：

- fused accumulator：shared 层应有 accumulator flow，Triton 层可以决定能否用 `tl.dot` accumulator operand兑现；
- stream pipeline：shared 层应有 order/dependency/validity，Triton 层只决定该 surface 和下层 pipeline 是否可用。

当前 `realizeProgram` 同时扫描 physical/KIR graph、重新统计 load/aggregation/scan/effect，重新推导 ragged route、contract orientation、reduction axes、gather category、descriptor affine expression 和 stream prefix boundary：[ProgramForms.cpp](../lib/Target/Triton/Lowering/Transforms/ProgramForms.cpp#L450)。

这不是“文件名叫 pass”的问题，而是这一个 pass 同时承担 semantic re-derivation、shared fact recovery、provider form selection 与 capability checking。前两项应由 KIR-derived shared analysis 或 executable physical IR 唯一提供；后两项只有在 Triton DSL surface 确实存在差异时才留在精简后的 target conversion。

### 5.6 真正的 Triton block program仍在字符串 materializer 中生成

当前 materializer 做的远不只是 syntax spelling：

- 把 side records 和 KIR 再索引成一个 C++ `PhysicalProgramIndex`：[Program.cpp](../lib/Target/Triton/Lowering/Materialization/Program.cpp#L140)；
- 解析 ABI shape/stride、partition、region range、workspace、scan 与 deferred replay；
- 从 logical index relation 现场构造 broadcasted pointer expression：[Program.cpp](../lib/Target/Triton/Lowering/Materialization/Program.cpp#L2118)；
- 现场构造 tail、indirect-index、partition 与 ragged mask：[Program.cpp](../lib/Target/Triton/Lowering/Materialization/Program.cpp#L2249)；
- 从 `intent.result_shapes` 和 side range 计算最终 emitted tensor shape：[Program.cpp](../lib/Target/Triton/Lowering/Materialization/Program.cpp#L2588)；
- 生成 autotune decorators、kernel signature、grid、private/scan workspace wrapper 和 launch glue。

最终 `materializeTargetProgram` 把整段 source 塞入 `TargetProgramOp.source`；所谓 terminal translation 只把这个字符串输出：[Driver.cpp](../lib/Target/Common/Lowering/Driver.cpp#L12)。

所以“terminal translator 已经很薄”在字面上成立，却掩盖了真正问题：隐藏编译器只是前移到了 `materializeProgramSource`，还没有成为 provider IR passes。

### 5.7 当前错误的性质：不是 facts 全错，而是程序本体没有承载它们

当前已有的 axis role、range、validity、replay、residency、capability 等事实并非全部错误；不少规则已经通过真实 kernel 的数值与性能验证。错误在承载和依赖方向：

- executable function 没有物化这些决定，side records 与 cloned KIR 必须共同解释才知道程序怎样运行；
- verifier 主要验证 record 完整性、ID/provenance 和属性合同，无法验证真正尚未存在的 pointer、mask、loop、physical shape 与 def-use；
- provider materializer 被迫同时读取 KIR、重建 KernelFacts、查询 records，再补出缺失程序，因此它事实上仍是隐藏的 GPU compiler；
- shared pass 即使改变了一个重要决定，后续也不能像成熟 IR pass 那样通过局部 op/type/use-def rewrite 组合、canonicalize 或删除旧形态。

所以不能把现状概括成“所有编译决策都错了”。更准确的是：**已有分析与规则资产可以保留，但 V2 选择的 executable representation 仍停在过渡形态，导致正确的决定没有成为当前程序本身。**这也是为什么继续向 records 增加字段、继续在 materializer 增加分支，无法完成这次重构。

---

## 6. Intent 与成熟 Triton 的差别到底在哪里

| 编译阶段 | 成熟 Triton 中的程序对象 | 当前 Intent 中的对应物 | 真实差距 |
|---|---|---|---|
| 作者算法 | Triton Python AST/TTIR block program | Intent DSL/KIR logical-region program | Intent 更高层，尚未写 program decomposition、block tensor 与 pointer/mask |
| 初始 GPU program | TTIR 中已有 pid、static tensor、access、loop、dot/reduce/scan | `exec_*` KIR clone | 当前 clone 没有真正形成 block program |
| 程序实例与 blocking | Triton 作者 source 已显式写下 | Axis/Range/Launch side records | 决定存在，但未改写 control/types/SSA |
| 物理 value/access | TTIR operands/results、pointer tensors、mask、descriptor | Buffer/Transfer/Padding records + KIR metadata | 实际 pointer/mask/value graph 到 materializer 才形成 |
| 分布与 layout | TTGIR encoding、`convert_layout` | 没有对应 shared/provider IR | 正确地应主要交给 Triton，而不是在 Intent 复制 |
| primitive/storage | TTGIR/NVIDIA ops、memdesc、shared/TMEM、MMA | Contract/Transfer string forms | 当前 provider form 没有成为 executable op graph |
| loop/pipeline/WS | SCF + schedule attrs + async/warp-specialize ops | Launch/Range flags与 emitter branches | 当前选择和实际 loop rewrite分离 |
| terminal lowering | TTGIR/NVIDIA→LLVM/PTX | Python source string→外部 Triton compiler | Intent 应止于高质量 Triton DSL source，不重复 TTIR 之后的下层 |

因此当前最根本的差距不是“我们的 heuristic 还不如手写 Triton”，也不是“缺几个 TMA form”。更早的问题是：

> 手写 Triton source 本身已经是一份明确的 block program；Intent 当前 passes 只选择了这份 program 的若干 side facts，真正把这些 facts 组合成 block program 的工作仍藏在 source materializer 中。只要这一点不改变，shared/provider passes 就难以像 Triton passes 一样对当前程序进行局部、可验证、可组合的变换。

---

## 7. 决策分类不能再只用 U/S/P 三格

“唯一解 / 结构选择 / 参数选择”可以帮助初步讨论，但不足以决定代码放哪。真实 Triton 已证明：参数可以驱动结构变换，结构变换也可能完全由下层 compiler 负责。

每一项工作至少要沿四个维度判断：

### 7.1 语义地位

- KIR semantic fact：作者已经写下；
- derived proof/fact：可从 KIR 或当前 physical IR 唯一重算；
- unique legal lowering：只有一种正确兑现；
- semantics-preserving alternative：存在多个合法实现。

### 7.2 必须在哪个 surface 之前显式

- Triton source/TTIR 必须已经表达；
- TTGIR 能从 TTIR 推导；
- NVIDIA backend 才需要；
- runtime/config 才绑定。

### 7.3 谁拥有足够信息

- Intent shared analysis/pass；
- Triton provider conversion/pass；
- Triton lower compiler；
- Triton autotuner。

### 7.4 结果怎样存在

- analysis cache；
- current physical IR 的 type/op/region/attribute；
- provider IR；
- config candidate；
- 纯 terminal spelling。

例如：

| 决定 | 语义地位 | 正确 owner / representation |
|---|---|---|
| 地址宽度 | unique legal lowering | Intent/shared access lowering，唯一来源 |
| reduce axis/combiner | KIR semantic fact | KIR + derived analysis；不在三家分别复制 |
| logical work 到 program id | semantics-preserving physical mapping | Intent GPU physical IR，因为 Triton 无法从 KIR 看见 |
| block tensor static extent | physical granularity | Intent physical IR，可绑定 meta parameter |
| pointer vs descriptor surface | provider form | Triton provider IR/candidate，消费明确 access graph |
| coalesced lane/warp layout | lower compiler transform | Triton TTGIR `CoalescePass` |
| MMA encoding/shared operand | lower compiler transform | Triton `AccelerateMatmul` |
| `num_stages` winner | config-indexed lower transform | Triton autotuner + software pipeline |
| persistent CLC loop | NVIDIA lower transform | 优先委托 Triton `ToCLC`；Intent 只需暴露合法 pid/program contract和候选入口 |

这张表也说明：一个 pass 只有一个合法结果并不等于“假 pass”；它可能是必要 correctness lowering。反过来，一个 pass 里有多个 `if` 也不等于拥有真实编译空间；如果它只按 KIR 名字填字符串、实际 executable graph 不变，它仍然只是迟到的分类器。

---

## 8. Shared GPU 与 Triton provider 的准确边界

### 8.1 Shared physicalization 必须完成

进入 Triton provider conversion 前，当前 physical IR 至少应明确：

- logical instances 到 program/grid ownership 的映射；
- program 内 loop/range nesting 与 static block/lane shapes；
- physical SSA、carry、accumulator、replay/materialize 与 lifetime；
- access operands、logical-to-physical index、footprint、validity 与 conflict obligations；
- reduce/scan/contract/state/ragged 的 physical skeleton与参与关系；
- effect/dependency/visibility 约束；
- 哪些 granularity 是常量，哪些是正式 provider parameters。

这里的“shared”表示这些事实不依赖 `tl.*` 语法，而不是要求 cuTile/TileLang 与 Triton 最终拥有同样的 op 数量或 storage representation。

### 8.2 尚未完成的 Triton DSL conversion

- physical ownership → `tl.program_id` 与 grid contract；
- physical ranges → `tl.arange`、Triton/Python loops 与 static block values；
- access graph → pointer/descriptor、mask、`tl.load/store/gather/scatter`；
- structured skeleton → `tl.dot/reduce/scan` 等 Triton DSL operations；
- Triton-only form/capability，例如 descriptor legality；
- formal config parameter mapping。

当前实现尚未在 IR 中闭合这些步骤；`ProgramForms` 与 materializer 仍回看 KIR、shape、relation 和 use-def。完成后的 Triton conversion 可以包含少量 legalization/form passes，也可以使用 Triton-specific provider ops，但它们必须只消费当前 executable physical program，不能回到 KIR 重新决定 shared structure。目标不是建立一套仿照 TTGIR 的厚 Triton 子编译器，而是得到一份可机械序列化的 Triton DSL program。

### 8.3 Triton lower compiler继续完成

- distributed layout、coalescing、thread locality；
- dot→MMA、shared/TMEM representation；
- layout conversion elimination；
- async copy、TMA lowering；
- software pipeline、warp specialization、CLC；
- fence/barrier、register、LLVM/PTX 与 instruction lowering。

Intent 只有在真实 Triton source surface 必须显式选择、而下层无法从 provider program 与 config 得到时，才应把某项能力上提。不能仅因手写 source 使用了 TMA 或某种 layout，就在 Intent shared IR 中复制它。

### 8.4 当前 `Target/Triton/Lowering/Passes.h` 为什么是架构症状

不能用“真实 Triton 也有很多 NVIDIA passes”保留当前规模。真实 Triton passes 位于我们委托的下层；Intent 的 Triton 路径只有在 shared physical program 与 Triton DSL surface 之间确有 target-specific legalization/form choice 时才需要 pass。

当前代码应按下面的方向收敛：

- orientation、batched、reduction axes、ragged membership route、logical stream boundary 等不应在 Triton、cuTile、TileLang 三处各推一遍；
- pointer/descriptor、Triton primitive compatibility、provider config mapping 等应留在 Triton；
- 真正的 Triton DSL program form 应进入 provider IR，而不是只挂属性后由 2700 行 materializer重新构造程序；
- terminal source generation只遍历 provider-legal IR并做确定性 serialization；
- 若完成 shared physicalization 后某个 Triton pass 只剩机械一对一 conversion，就应合并或删除，不能为了“pass pipeline 对称”而保留。

因此不是机械地把所有 Triton 文件搬到 GPU shared；而是把当前重复推导的 shared facts 和结构决定消除，让 Triton 路径收缩到真实 surface/legalization 差异，然后立即委托外部 Triton compiler。当前实现离这个状态仍有实质距离。

---

## 9. Pass 怎样才真正扩展编译空间

### 9.1 尚未满足的目标：初始 conversion 必须产生保守但完整的程序

KIR→GPU physical IR 的第一份结果可以慢，但必须已经有：

- 明确 program instances；
- 明确 physical loops/ranges；
- 明确 value/access/control graph；
- 明确、可验证的 legality。

不能留下“AxisOp 已选择 ownership，但 pointer tensor以后由 emitter从 KIR relation 猜”“LaunchOp 写了 persistent，但 loop 以后由字符串分支创建”这种 executable hole。

### 9.2 Shared passes 必须改写当前 physical program

例如：

- automatic blocking 应创建/替换 program mapping、physical loops 和 shaped types，而不是只换一组 Range records；
- replay/materialization 应改变 physical def-use；
- contraction realization 应创建合适的 accumulator/operand graph；
- access realization 应创建明确 access ops 与 validity operands；
- persistent realization若由 Intent负责，应创建真实 kernel 内 loop；若 Triton 已负责，则 shared pass只保留必要 eligibility/semantic contract并把选择交出去。

### 9.3 尚未满足的目标：provider conversion 必须形成 provider-legal IR

当前 provider pass/materializer 的中间结果不能独立完成下面的验证。完成后的 Triton DSL provider program 应能独立验证：每一个 shaped value 有合法静态 shape，每一个 access 有 pointer/descriptor 与 mask，每一个 structured op 有合法 operands/results，每一个 loop/control region 完整。

到 translator 阶段，不应再需要：

- `KernelModel.nodes.lookup`；
- `intent.result_shapes` 反推 emitted tensor shape；
- `parseIndexRelation` 现场拼 pointer；
- 从 op 名、role 或 parent structure重建 range/route；
- 临时创造 workspace、loop 或 tuner axes。

### 9.4 Pass 的外部成立证据

一项性能能力不能靠“代码位于 pass 文件”“有 verifier”“没有 kernel-name branch”自证。至少要给出：

1. pass 前后真实 IR type/op/region/def-use 差异；
2. legality 与 semantic-preservation 条件；
3. 换一个 kernel/shape 后会否作出不同决定；
4. 触发 kernel之外的结构复用；
5. 同一 provider candidate contract下的数值与性能 A/B。

如果只看到某个 attr 从 `single` 变成 `pipeline_candidates`，却看不到 program 如何改变，不能声称这个 pass 已经拥有完整结构能力。

---

## 10. Intent 怎样超过一份手写 Triton source

手写 source 已经固定了一份 Triton DSL-level block program；外部 Triton frontend/lower compiler 又会对它做强力优化。Intent 若要超过它，真实可增加的空间是：

- 从逻辑 workset 系统地选择更好的 program ownership与grid mapping；
- 组合多层 blocking、ordered/stream segmentation 与 program traversal；
- 根据 def-use、reuse 和 lifetime 选择 replay、materialize、carry 与 accumulator graph；
- 根据完整 access relation 形成更好的 pointer/descriptor/gather surface；
- 让 boundary/neutralization 保持合法同时减少 mask与搬运；
- 把 reduce/scan/contract/ragged/sparse 组合成更利于 Triton 下层识别的 TTIR program；
- 根据 device/capability形成不同的合法 provider surface，而不是固定照抄一份 source。

这不是“复制 source”与“随便选另一份结构”二选一。正确过程是：

1. 用 source 确认作者算法与高性能 program中关键结构；
2. 把其中能由 typed facts 泛化的关系变成 analysis 与 transformations；
3. 让 passes 在 physical IR 上重建这些关系，而不是做 whole-kernel template matching；
4. 同一套 pass 能在其它算法组合上产生合理结构；
5. 最终把更好的 Triton DSL program交给同一个 Triton frontend/lower compiler。

Intent 的编译空间不是一个需要完整枚举的集合，也不必立刻有全局 cost model。它可以由 deterministic canonicalization、legality-driven conversion、typed profitability rules和少量正式 provider alternatives共同组成。但若一个所谓 decision 永远只有固定字符串、从未改变 executable IR，它不能被拿来证明编译空间已经建立。

---

## 11. Tuner 与 baseline 的正式合同

### 11.1 Winner 完全交给 Triton autotuner

Intent 不应写一张设备无关的 `num_warps` 阶梯表替 Triton选 winner，也不应把固定 `num_stages` 当 shared compiler知识。Intent 可以做的是：

- 声明哪些 meta parameters绑定哪些 physical ranges/forms；
- 从语义、shape、alignment、resource 与 capability证明 candidate legality；
- 删除确定非法的 candidate；
- 把剩余完整集合交给 Triton autotuner。

Triton autotuner会逐 candidate benchmark并选最小 runtime：[autotuner.py](../../ref/triton/python/triton/runtime/autotuner.py#L228)。

当前 Intent shared search-space pass主要收集 tile/group parameter roles：[Decisions.cpp](../lib/Target/GPU/Realization/Plan/Decisions.cpp#L1192)；Triton materializer又额外加入 `USE_TMA`/`USE_NATIVE_SCALED` candidates：[Program.cpp](../lib/Target/Triton/Lowering/Materialization/Program.cpp#L752)。这说明现有 candidate contract 仍分散在 shared records、provider materializer和 Python runtime profile 三处，尚未成为一份正式 provider search surface。

### 11.2 同一候选集合的准确含义

正式 source/generated 对照必须做到：

1. 双方算法、调用数、scope 与输入完全一致；
2. 双方暴露同一组具有相同语义角色的 provider candidates；
3. 每一个 candidate 的 block/chunk roles、`num_warps`、`num_stages`、`num_ctas` 与双方共有 provider options取值一致；
4. 参数名称可以不同，但不能把 source 的 `BLOCK_K=64` 映成 generated 的另一种逻辑粒度；
5. 双方各自运行真实 Triton autotune并选择各自 winner；
6. autotune/JIT成本不进入 steady-state p50。

“同一候选集合”不是强行让双方使用同一个 winner，也不是拿单一 config做结论。它是让双方在相同调优预算和可比 parameter semantics 下各自取最优。

由于 `num_warps`/`num_stages` 会改变下层结构，它们仍必须进入共同 candidate contract；不能因为它们“有结构效果”就排除，也不能把它们的效果冒充 Intent passes 相对 source 的优势。

### 11.3 Strict ratio 与功能参考要分开

只有满足上述算法与 candidate 合同的 entry 才能形成严格 compiler/source ratio。下面情况只能是结构/功能参考：

- source 固定单 config、generated autotune；
- 双方候选集合不同；
- source/generated算法或 launch pipeline 不同；
- source adapter计时 scope 不一致；
- source 无法在当前 provider/runtime真实运行。

CSV 可以继续只保存数字，不需要加入校验逻辑；但 report/runtime adapter必须明确一格是否具备严格比较资格。

---

## 12. 对当前重构方向的具体结论

### 12.1 应保留的资产

- canonical KIR、stable node/value identity与 semantic facts；
- axis/domain/ragged/state/contract/index/effect analyses；
- 已验证的 blocking、validity、neutralization、replay、residency 与 capability知识；
- MLIR pass manager、verifier骨架和 provider syntax资产；
- source/runtime/baseline中已经建立的真实算法与性能证据。

### 12.2 需要重新实现其承载方式的部分

- `exec_*` 不能长期只是 KIR clone；要获得 physical types、ops、regions与def-use；这不是新增一个旁表，而是替换当前 executable carrier 的表示方式；
- Axis/Range/Transfer/Contract等 records中真正影响执行的选择，要 materialize进当前程序；
- shared passes不能只删改 records，要改 executable physical IR；
- Triton ProgramForms中的 shared fact recovery要前移或成为共享 analysis；
- Triton-local form要成为 provider IR，而不是字符串 emission前的 attrs；
- materializer中的 pointer、mask、shape、loop、workspace与program binding构造要逐步迁移到 provider conversion；
- terminal translator最终只序列化完整 provider program。

### 12.3 不应上提到 Intent 的部分

- TTGIR distributed layout inference与coalescing；
- MMA/shared/TMEM具体 encoding与instruction selection；
- Triton software-pipeline expansion、warp specialization、TMA lowering、fence与LLVM/PTX lowering；
- autotune winner选择。

如果某个 Intent optimization看起来必须复制这些能力才能成立，应先检查是不是 TTIR surface没有把足够信息暴露给下层，或者是不是该把它做成 provider candidate，而不是在 shared compiler再造一个 Triton。

### 12.4 这轮仅对 Triton 路径下结论

Triton的真实源码足以确定“Intent 不应复制 TTIR/TTGIR/NVIDIA lower compiler”这条边界。对当前 shared program 已经拥有的 execution/value/access/validity/structured facts，Triton 与 cuTile 默认应复用同一来源，而不是各自重建。cuTile 与 TileLang 还需要哪些真实 target-only legalization，应分别读取实现后确认；不能预先把它们扩张成三套完整 leaf compiler，也不能仅凭 Python surface 相似就断言完全没有差异。

---

## 13. 最终判断

1. Intent 的 KIR 高于 Triton TTIR：它保留逻辑算法，却省略了 Triton作者通常手写的 program decomposition、static block tensors与pointer/mask程序。
2. Intent真正必须补的是一份完整 executable GPU program，再将其确定性转换成 Triton DSL program/source；不是直接生成TTIR，也不是重新实现TTIR→TTGIR→NVIDIA backend。
3. KIR语义不可被偷换，但physical SSA、op graph、loop、layout与schedule可以在证明语义保持后被重写。真实Triton就是这样工作的。
4. Compiler V2 已经真实接入唯一主链、pass manager、verifier和decision records；但当前`exec_*`仍是KIR同构副本，Axis/Range/Transfer等主要是side records，shared passes多数还没有重写executable program。因此V2是已运行的骨架，不是已完成的physical compiler。
5. 当前Triton ProgramForms仍从KIR重新推导shared facts，materializer仍现场构造pointer、mask、shape、loop、workspace和wrapper；provider-legal IR尚未真正存在。
6. 外部Triton拥有大量NVIDIA/TTGIR passes，不能证明Intent的`Target/Triton`也应厚重。当前其中大量shared事实重建正是架构症状；完成后只应留下Triton DSL surface/capability真正需要的conversion/legalization，并且terminal serialization不再作决定。
7. 编译空间必须按程序实例、块内迭代、physical value、access graph、structured realization、schedule/storage obligations、provider forms和lower compiler层次展开，不能再用一个“physical structure”概括。
8. 参数与结构不是二分：Triton Config会索引layout、pipeline和CTA结构。Intent负责合法candidate contract，Triton autotuner负责winner。
9. 正式baseline必须让source/generated使用同一可比candidate集合，各自真实autotune选winner；单config或不同候选预算不能证明compiler性能。
10. Intent若要通过passes超过手写Triton，必须在KIR→executable GPU physical program→Triton DSL program这段建立可组合、typed、可验证的变换，并把生成的高质量DSL source交给Triton成熟下层，而不是靠更多leaf字符串分支或更大的config搜索。

这才是后续判断某项改动“属于编译器能力还是玩具补丁”的依据：看它改变了哪一层当前IR、保持了什么语义、是否委托了Triton已有能力，以及换一个算法组合后是否仍然成立。
