# Intent DSL 到 GPU Tile Source：当前到底在编译什么（V1）

## 0. 这份报告回答什么

这份报告只讨论一个严格限定的问题：

> **固定同一份 Intent DSL / Kernel IR 所表达的算法后，从 Kernel IR 到 Triton、cuTile 或 TileLang 源代码之间，当前编译器究竟还要决定和构造什么；这些工作为何可能影响性能；它们今天实际位于 GPU Realizer、Physical Plan、target leaf、provider tuner 和下层编译器中的哪里。**

这里不把以下内容算作答案：

- 不把一个算法替换成另一个算法。例如 GEMM 变成 Strassen、稳定 softmax 变成 online softmax，都不是同一 Kernel IR 的不同 realization，而是源算法已经变化。`doc/compiler/kernel-ir.md:70-74` 和 `doc/dsl/core-and-algorithms.md:60-71` 对这条边界已有明确约束。
- 不把图级融合、算子替换或整图调度带回来。当前对象是一个 kernel 内部的 physicalization。
- 不把地址宽度、必要 mask、合法性检查等唯一正确的 lowering 夸大成性能优化空间。
- 不把 `BLOCK`、`num_warps`、`num_stages` 等候选 config 或测出来的 winner 当作本系统最核心的“性能知识”。这些主要属于 provider tuner。
- 不把 Triton/TileLang/cuTile 之后的 layout lowering、指令选择、寄存器分配和机器级流水归到本层。

本报告依据当前代码做静态审计。它既不是对理想架构的复述，也不是性能测量报告。

## 1. 先给出不会再随质疑改变的结论

当前 IntentDSL 最准确的定位是：

> **它是一个面向受支持 canonical Kernel IR 子集的、由 typed facts 驱动的 canonical target-program constructor。Kernel IR 固定算法和逻辑数据流；GPU Realizer 选择并记录一部分跨 provider 的物理骨架；target leaf 再把这个骨架补全成 provider-specific 的 Triton/cuTile/TileLang 程序；下层编译器继续完成机器级实现。**

这句话包含四个同等重要的限定。

第一，它确实是编译器，而不只是整算子库选择器。输入不是“op 名称 + shape”，输出也不是从若干完整手写 kernel 中挑一个；当前代码沿 KIR 的 axis、value、access、def-use、structured op 和 effect 逐项构造目标程序。

第二，它目前也不是一个拥有巨大显式 schedule 搜索空间的优化器。对于固定的 KIR、device、provider 和 tuner winner，当前实现大体沿一条确定性的 canonical path 产生 Plan 和目标源码。代码中存在多个本可采用不同合法实现的决策点，但目前多数由固定规则或 leaf policy 决定，并没有枚举成一个 Cartesian search space，也没有 cost model 在这些结构之间搜索。

第三，当前高性能并不能单独归功于 GPU Realizer。真实归因是：

1. DSL/KIR 中已经选对的算法和逻辑数据流；
2. GPU Realizer 中有限但有意义的 canonical physical policies；
3. leaf 中大量手写的 target-specific program construction；
4. provider tuner 给出的 launch/config 参数；
5. Triton/cuTile/TileLang 及其下层编译器完成的物理推导和机器优化。

所以，如果问题是“现在的 kernel 高性能主要是 leaf 写得好，还是 shared compiler 写得好”，诚实答案是：**两者都属于编译器，但目前 target-specific 性能实现的重量明显更多地落在 leaf 和下层；shared GPU Realizer 已经决定了一些物理结构，却还不是一个广而强的硬件 schedule optimizer。**

第四，现状不是“GPU Realizer 已经做完全部 physical decisions，leaf 只是打印”。当前 leaf 会读取 KIR、查询 Plan、重建循环和 transfer 结构、回放 scan producer，并作出若干 target-local 选择。有些是 leaf 正当职责，有些只是从已选 Plan 推导出的 emission cache，还有少数决策没有被现有 Plan contract 清楚记录。第 6 节会正面区分它们。

## 2. 同一算法固定以后，仍然缺少什么

### 2.1 Kernel IR 已经固定的东西

`doc/compiler/architecture.md:3-23` 给出的主路径是：Python wrapper → frontend → canonical Intent Kernel MLIR → Realizer → Physical Plan MLIR → target translator → target source → target toolchain/runtime。

进入 Realizer 之前，Kernel IR 已经固定：

- runtime-visible ABI；
- logical workset；
- logical tensor/value flow；
- structured state/control；
- contract、reduce、scan、state stream 等算法节点；
- indexing、effects 和数值语义。

具体边界见 `doc/compiler/kernel-ir.md:9-56`。因此，DSL 中的纯 tensor SSA 中间值本来就是算法表达的一部分，不能再把“创建逻辑中间值”说成编译器的自由空间。`doc/dsl/tensor-flow.md:5-13` 也明确区分了逻辑 value flow 和独立物理 buffer；`I.buffer` 固定的是逻辑可变对象，而不是最终 address space、stage 或机器存放位置（`doc/dsl/tensor-flow.md:146-152`）。

### 2.2 Kernel IR 没有固定的 target program

同一 KIR 仍不足以直接成为 Triton/cuTile/TileLang 源码。缺失的是 target-visible physical program，至少包括：

- 逻辑 axis/range 如何形成目标程序实际迭代的 tile、lane、reduction/traversal 和 worker grouping；
- 哪些逻辑 value 只作为表达式继续传播，哪些需要物理 materialization、workspace、shared/local allocation 或跨 stage 搬运；
- 一个 KIR transfer 在目标 DSL 中实现成普通 load/store、gather/scatter、bulk copy、并行逐元素搬运还是某个 target 专有 primitive；
- contract、reduce、scan、ragged、ordered/state-stream 等 structured operation 如何被组合成目标 DSL 的循环、临时存储、同步和 primitive 调用；
- provider 的 launch surface、合法参数和 runtime wrapper 如何生成。

这些不是“换算法”。它们是在保持同一 workset、同一 value/effect 语义、同一数值约束时，构造一个可被目标编译器继续 lowering 的程序。

这里也必须澄清 `program_id` 的地位：**单纯把某个 axis 映射到 `program_id`，往往只是必要的寻址骨架，并不足以构成重要优化。** 只有当它与 grouping、folding、persistent traversal、reuse、tile residence 等结合，改变目标程序的数据复用和执行粒度时，才形成值得讨论的性能结构。因而本报告不拿“谁拥有 program id”作为主论点。

## 3. 一个真实锚点：同一 GEMM KIR 如何成为不同 target source

这里使用 GEMM 不是为了用一个模板代替宏观结论，而是为了精确指出每一层到底写了什么。

### 3.1 DSL/KIR 写的是算法

`examples/kernels/contraction/gemm.py:15-37` 声明 M/N/K 逻辑范围、A/B 的 view 和一个 `contract`。它已经决定这是经典 contraction 的算法与数值关系；编译器不能把它替换为别的矩阵乘算法。

### 3.2 GPU Realizer 写入的是共享物理骨架

当前 Realizer 会形成 axis role/range、lane packing、worker/fold/group、reuse/persistent 以及 contract operand/result residency 等 facts。主要实现集中在：

- `lib/Target/GPU/Realization/Plan/Decisions.cpp:390-499`：axis roles；
- `lib/Target/GPU/Realization/Plan/Decisions.cpp:508-531`：lane distribution/packing；
- `lib/Target/GPU/Realization/Plan/Decisions.cpp:552-739`：role/range、worker/fold/group/reuse；
- `lib/Target/GPU/Realization/Plan/Decisions.cpp:741-781`：persistent policy；
- `lib/Target/GPU/Realization/Plan/Build.cpp:1051-1134`：contract realization facts。

这些 facts 不是一份完整 Triton 或 TileLang schedule，但它们约束了 leaf 不能任意改变的共享物理意图。

### 3.3 Triton leaf 构造的是一个真正的 block program

Triton leaf 并非只打印一个 `contract` 名字。`lib/Target/Triton/Emission/Handlers/Operations.cpp:2572-2623` 实际构造 accumulator、K-loop、`tl.arange`、A/B pointer 与 mask、`tl.load`、必要转置和 `tl.dot`。

所以“Triton leaf 比较薄”如果要成立，只能表达下面这个有限含义：**生成 Triton source 之后，layout 分布、指令映射、寄存器分配和更低层流水由 Triton compiler 继续完成。** 它绝不能被解释为“Intent 到 Triton source 之间几乎没编译工作”。从 KIR contract 到上述 block program，本身就是本项目正在做的 target-program construction。

### 3.4 TileLang leaf 构造的是另一种目标程序

对于相同的 contract 语义，TileLang leaf 会构造 shared/local allocation、copy、同步、`T.Pipelined` 和 `T.gemm` 等 TileLang-native 结构，相关代码见 `lib/Target/TileLang/Emission/Handlers/Operations.cpp:3394-3701`，其中主要 contract 路径位于 `:3551-3645`。

这不是另一个算法，而是同一 KIR 加同一类 Plan facts 在另一种 target DSL 上的 realization。与 Triton 相比，TileLang source 显式承载了更多 copy、buffer 和 pipeline 结构；Triton source 则把更多物理推导留给其 compiler。这正是 leaf 必须 target-specific 的原因，而不是需要再发明一个位于 leaf 之上的新层级。

### 3.5 下层仍然决定什么

目标 DSL 编译器随后还会决定或完成：

- target layout 到线程/warp/CTA 的更细分布；
- primitive 到机器指令的选择；
- register allocation 和 spill；
- shared-memory lowering、barrier 和机器级 pipeline；
- instruction scheduling 与最终 binary construction。

因此，Intent 编译器不需要重复实现这些已有能力。它需要做的是产生一个结构足够好的 target program，使下层看到合适的 tile、访问、循环、primitive 和可优化边界。

## 4. GEMM 之外，反复出现的真正编译对象

下面四类 kernel 展示的是同一个边界，而不是四个孤立技巧。

| Kernel | DSL/KIR 已固定 | Shared Plan 要记录 | Leaf 要完成 | 下层继续完成 |
|---|---|---|---|---|
| GEMM | M/N/K workset、A/B access、contract 数值语义 | tile/range roles、worker/group/reuse、operand/result residence | block loop、load/copy、allocation、sync、target MMA primitive | layout、指令、寄存器、机器流水 |
| stable softmax | max、sub、exp、sum、div 的逻辑 value flow；它不能被改成 online 算法 | reduction/lane/range 和必要 residence/materialization | target reduction primitive、临时 value 的目标表达、store | collective lowering 和机器调度 |
| attention/state stream | online recurrence、state update、causal/ragged 语义 | ordered/traversal/stream-contract binding、stage facts | provider-specific 循环、state carry、contract/reduce/transfer 组合 | primitive 与内存指令 lowering |
| ragged/scan | membership/indexing、producer、scan/reduction 语义 | ragged ranges、ordered stages、scan producers/materialized values | checked/gather transfer、scan loop、producer replay、workspace/source construction | target 指令和低层同步 |

对应的真实 DSL 入口包括：

- `examples/kernels/normalization/softmax.py:19-27`；
- `examples/kernels/streaming/attention.py:107-161`；
- `examples/kernels/ragged/jagged_mean.py:20-45`。

这里反复出现的编译对象不是新的逻辑中间值，而是：

> **把已固定的 logical value/access/state graph，绑定为目标 DSL 可以执行的 physical iteration、materialization、transfer、storage 和 structured primitive composition。**

这可以叫 physical dataflow realization，但不能偷换成图编译器里的任意 dataflow rewrite。它只在同一个 kernel、同一个 KIR 算法之内工作。

## 5. 当前“编译空间”到底有多大

### 5.1 先区分五类东西

现有 `report/history/formal-decision-space-and-algorithm-alignment-audit.md:14-132` 的唯一答案、结构 realization、provider 参数三分法是有用的，但还需要把 leaf 的 target-local realization 单独讲清楚。当前事实可以分成五类：

| 类别 | 含义 | 是否是本报告所说的性能编译空间 | 当前归属 |
|---|---|---|---|
| 唯一 correctness/derived facts | mask、合法地址、由 KIR+Plan 唯一推出的依赖/index cache | 通常不是 | shared verifier/index/lowering |
| Shared structural policy | 同一 KIR 下存在多个合法物理骨架，选择会影响复用、粒度、materialization 或 traversal | 是 | GPU Realizer → Physical Plan |
| Target-local realization policy | 同一 Plan 在 Triton/cuTile/TileLang 中采用不同 API、transfer、allocation、loop 或 alias-safe structure | 是，但必须限定为 provider-specific | target leaf |
| Provider tuner 参数 | block/group/stage/warp 等可测量参数候选与 winner | 不是本系统区别于手写 target source 的核心 | provider tuner/runtime |
| Lower-compiler decisions | layout、instruction、register、低层 pipeline | 不是本层空间 | target compiler/toolchain |

这张表最重要的作用，是避免两个相反错误：既不把所有 deterministic lowering 都包装成“优化”，也不因为当前没有 cost model 就断言中间没有编译器-owned decisions。

### 5.2 当前 shared structural policy 的真实内容

当前 GPU Realizer 确实有一组确定性 policy，而不是完全填空：

- lane distribution/packing；
- worker grouping、folding、reuse 和 persistent traversal；
- value/operand/result residence；
- direct transfer 与 deferred/staged materialization 的部分选择；
- ragged contraction 的部分 stage construction；
- reduce、scan、contract 对 Physical Plan 节点的形成。

主要实现位置为 `lib/Target/GPU/Realization/Plan/Decisions.cpp:508-781`、`:992-1223` 和 `lib/Target/GPU/Realization/Plan/Build.cpp:74-131`、`:718-1004`、`:1051-1134`。

但不能据此声称空间很大：

- 一部分 role/range/index 只是由 KIR 唯一或近乎唯一推导；
- 一部分不同分支只服务于不同 KIR 结构，并不是固定同一 KIR 时的备选 schedule；
- 当前代码通常由 `if`/规则直接选择一个结果，没有把多个结构候选显式保留下来；
- device model 的实际影响还很窄。

`include/Intent/Target/GPU/Config/Device.h:8-15` 虽然定义了 compute units、matrix units、registers、shared memory 和 dynamic vector width 等字段，但当前编译时使用主要集中在 register-based residency（`lib/Target/GPU/Realization/Plan/Build.cpp:74-131`）和 matrix-unit capability gate（`:1183-1189`）；其余若干字段主要用于校验，`dynamicVectorWidth` 尚未形成广泛 policy。

因此，对当前空间大小最准确的回答是：

> **它不是一个巨大 schedule search space，而是一个有限、结构化、目前大多确定性的 target-realization gap。shared Realizer 覆盖其中一部分跨 provider 物理决策，leaf 覆盖相当一部分 provider-specific 决策。**

### 5.3 “确定性 canonical path”并不等于“没有编译”

Triton 自己也会对某些高层 layout/operation 采用确定性 physical derivation。一个 compiler pass 不需要保留多个候选或调用 cost model 才算编译。

本项目当前的价值可以是：从 typed KIR facts 出发，稳定地构造一个正确且适合目标 DSL 的 canonical target program。真正应受质疑的不是“它为什么不搜索”，而是：

- canonical policy 是否确实利用了 KIR 和 hardware facts；
- 它选择的物理结构是否影响目标程序质量；
- 这些选择是否被显式记录并由 leaf 一致消费；
- 是否仍有本应共享的 policy 隐藏在 leaf 的局部代码里。

### 5.4 provider tuner 到底是什么

provider tuner 是目标 provider 运行时对一组候选 config 做实测选择的机制。当前代码中按 GEMM、stream、ragged 等 role 组织的经验配置，本质上就是类似 Triton `autotune configs` 的候选表和参数规则。

它们可以很重要，但这里只把它们视作“给已经构造出的程序选参数”，而不视作本项目最核心的性能知识。手写 Triton source 同样可以提供和调优这些 config；仅靠 config 无法说明 Intent compiler 与 source Triton 的本质差异。

## 6. GPU Realizer 与 leaf：当前真实边界，而不是理想口号

### 6.1 理想 contract

`doc/compiler/physical-plan.md:3-21`、`:33-41` 和 `:65-77` 规定：Physical Plan 应保存编译器已经选择的 machine facts；leaf 不应重新决定 axis ownership、range、materialization 或 stage；由 KIR+Plan 唯一推出的派生事实可以放在 common index 中重算。

这个方向是正确的，但“leaf 只是机械打印”并不能准确描述当前代码，也不应该成为目标。leaf 的合理输入本来就是 **KIR 语义 + selected Plan + target capability/policy**，而不是脱离 KIR 仅序列化 Plan。

### 6.2 三种不能混在一起的 leaf 工作

#### A. 共享派生查询：不是新决策

`include/Intent/Target/Common/Emission/SurfacePlan.h` 会从 KIR 与 selected Plan 建立若干 cache/index：

- ragged/stream index：`:1235-1346`；
- stage dependencies、inputs、outputs、terminal facts：`:1659-1809`；
- scan producer/materialized-value index：`:727-828`。

这类事实若能由既有输入唯一推出，就不需要膨胀 Plan。它们只是 common emission query，不是优化空间。

同理，contraction orientation 或 axis correspondence 若由 KIR type/access 和已选 transfer domain 唯一推出，应该集中为共享查询；“leaf 读取了 KIR”本身不证明它重新做了 physical decision。

#### B. 正当的 target-local realization：应该就在 leaf

当前 leaf 中下列工作具有明确的 target-specific 性质：

- TileLang compact/guarded transfer 根据同一 access binding 构造 allocation、`T.copy`、并行 elementwise 路径和同步（`lib/Target/TileLang/Emission/Handlers/Operations.cpp:1112-1488`）；
- TileLang contract 构造 shared allocation、copy、pipeline、`T.gemm` 和别名隔离（`:3394-3701`）；
- Triton load 根据同一逻辑 view 构造 pointer、mask、padding/validity 形式（`lib/Target/Triton/Emission/Handlers/Operations.cpp:1198-1265`）；
- cuTile transfer 选择其 target API 可表达的 load/store/gather/scatter 结构（`lib/Target/CuTile/Emission/Source/Emitter.cpp:182-232`）；
- 三个 target 的 scan handler 根据 Plan 已记录的 producer/materialized values，构造各自的循环、workspace 和 producer replay（Triton `Handlers/Operations.cpp:1517-1729`；cuTile `:1608-1826`；TileLang `:2042-2180`）。

scan replay 的代码很多，但代码量本身不等于重新选择算法：如果 Plan 已经决定哪些 producer 被 replay、哪些 value materialize，leaf 重发这些 operation 是目标程序构造，而不是 policy 泄漏。

将来同一 transfer 在支持它的 Triton target 上选择 descriptor/TMA，而在另一个 target 上选择 pointer load，也属于这一类。TMA 只是 target API realization 的一个小例子，不是建立整个编译空间的主论据。

#### C. 边界尚未写清或确有 policy 泄漏的部分

当前代码也有不能被“机械 leaf”概括掉的事实：

1. 三个 leaf 会通过 `feedsStagedContraction(...)` 决定 pointwise 是否 defer：Triton `lib/Target/Triton/Emission/Source/Emitter.cpp:206-221`、TileLang `lib/Target/TileLang/Emission/Source/Emitter.cpp:217-237`、cuTile `lib/Target/CuTile/Emission/Source/Emitter.cpp:164-180`；公共 helper 位于 `include/Intent/Target/Common/Emission/SurfacePlan.h:1118-1151`。如果这只是已选 stage 的唯一后果，它应被明确命名为 derived fact；如果它实质改变跨 op materialization/fusion，则应成为 Plan fact。当前 contract 没有把这一区别说透。

2. TileLang transfer emitter 会根据 ragged/validity/packed/atomic 等上下文，在 bulk copy、parallel elements、gather/scatter 和 defer 之间组织目标结构（`lib/Target/TileLang/Emission/Source/Emitter.cpp:239-282`）。其中 API/spelling 选择属于 leaf；如果选择改变了 provider-neutral materialization boundary，则相应事实不应只藏在 leaf。

3. Triton、TileLang 和 cuTile emitter 都有面向整份 KIR 的 row-config eligibility 扫描，例如 Triton `lib/Target/Triton/Emission/Source/Emitter.cpp:298-342`、TileLang `:328-354`、cuTile `:286-342`。它更接近 target tuner/launch policy，而不是 shared Plan；但这说明当前 leaf 确实包含 kernel-analysis policy，不能再声称没有这条路径。

4. cuTile emitter 会在 KIR walk 后额外加入 `GATHER_SPELLING` tuner 参数（`lib/Target/CuTile/Emission/Source/Emitter.cpp:298-320`），而不是消费 Plan 中声明的 search space。这是现有“Plan 声明搜索维度、runtime 只测这些维度”contract 的一个清楚不一致。

5. TileLang 会依据局部 buffer 情况设定线程约束（`lib/Target/TileLang/Emission/Source/Emitter.cpp:1134-1140`）。这种约束很可能是 target legality/performance policy，留在 leaf 合理；但它应被承认为 leaf 决策，而不是说所有 machine facts 都来自 shared Plan。

因此，对“是不是有一些本应在 GPU Realizer 的东西放进了 leaf”可以给出比此前更严格的回答：**有边界候选和至少一处 search-space contract 不一致，但没有证据把 leaf 中大量 target construction 一概判成放错位置。** 跨三个 provider 都相同的 defer/materialization 语义，应收敛成 shared derived query 或显式 Plan fact；`GATHER_SPELLING` 应进入声明过的 target search surface。相反，TileLang 的 copy/pipeline/buffer 组织、Triton 的 pointer/tile 表达以及 target-specific thread/alias 约束，本来就应留在各自 leaf。

### 6.3 因而真正边界是什么

不需要新建一个“realization decision layer”。正确的两段职责是：

- **GPU Realizer / Physical Plan**：决定所有跨 provider、会改变 ownership、range、stage、residency、workspace、materialization 或结构 traversal 的物理事实，并显式记录；
- **target leaf**：在不改变上述 facts 的前提下，用 Triton/cuTile/TileLang 的原生 API、storage、loop、copy、sync、primitive 和合法性约束完成目标程序。

公共 `SurfacePlan` 可以保存唯一派生的 emission facts。provider tuner 只测 Plan/leaf 明确暴露的参数，不应通过隐式 KIR matcher 再发明另一套结构决策。

所以，问题不是“为什么 leaf 做了很多工作”。**leaf 本来就必须构造 provider-specific target program。真正的问题是每个分支究竟是在投影一个已选物理事实，还是在无记录地重新选择物理事实。** 当前代码已经有两类情况，现有文档把它们都叫“机械 projection”是不够精确的。

## 7. 与“给一个 op 选择高性能 TileLang/library kernel”有什么区别

必须先承认相似处：当前 target handlers 中有大量组合式局部 pattern；对简单 GEMM 而言，生成结果会很像实例化一个 canonical schedule template。仅仅因为代码由编译器打印出来，并不会自动产生本质差异。

真正的区别应看“选择和组合的单位”：

| 模型 | 输入单位 | 实现单位 | 跨节点组合依据 |
|---|---|---|---|
| 整算子库/registry 选择 | op 名称、shape、dtype、device | 一份完整预写 kernel | 通常是 registry key 和 wrapper contract |
| 手写 TileLang kernel | 作者直接写完整算法与 schedule | `T.Kernel`、allocation、copy、pipeline、gemm/elementwise | 作者在源代码中显式组织 |
| 当前 Intent compiler | 完整 KIR 的 axis、value、access、def-use、structured op、effects，加 selected Plan | 每个 KIR op/value/access 的 target realization，再组合为一个 kernel | typed KIR provenance + Plan binding + leaf target policy |

仓库中这三种形态都有可核查例子：

- `examples/repro/v2/registry.py:6-46`、`:83-114` 是按 entry 映射到完整 upstream runtime/kernel 的 registry，属于第一种；
- `source/tilelang/tilelang/gemm/dense/example_gemm.py:5-24` 由作者显式写 `T.Kernel`、allocation、`T.Pipelined`、`T.copy` 和 `T.gemm`，属于第二种；
- `examples/kernels/contraction/gemm.py:15-37` 和 `examples/kernels/streaming/selective_scan.py:36-169` 提供完整 KIR 算法，target handlers 再逐节点消费 Plan，属于第三种。

这使当前编译器能够在同一 kernel 中组合 contract、pointwise、reduce、ragged、scan、state/effect 等受支持节点，而不需要先命中一个“完整 kernel 名称”。这就是它没有退化成整算子库选择的核心证据。

但也不能夸大：当前组合能力受 finite handler schema 和 validator 约束，若某种节点组合没有实现就会拒绝；它不是任意 KIR 的通用 schedule synthesis。若未来 leaf 改成只看 whole-kernel signature，然后直接跳到一份预写 source，绕过 per-node KIR/Plan composition，它才会实质退化为 library selection。

## 8. 为什么当前结果可以高性能，以及性能知识实际在哪里

### 8.1 source algorithm 的贡献

对于 attention、selective scan、online softmax 等 kernel，DSL 作者已经选择了 recurrence、state update 和数值算法。若 source algorithm 本身没有表达高性能算法，Realizer 不能在“保持同一算法”的前提下偷偷换掉它。因此，这部分性能首先属于 DSL/kernel author。

### 8.2 shared compiler 的贡献

GPU Realizer 能通过 typed facts 统一选择 worker/group/reuse/persistent、residency、stage 和 materialization 等 canonical physical skeleton。写得不好会造成不必要 materialization、差的复用、错误粒度或不适合 provider 的结构；写得好则给所有 leaf 一个更有利且一致的骨架。

这是真正的 compiler-owned 优化位置，但当前 hardware model 和 policy 覆盖仍有限，不能把它描述成已经从大空间中自动发现 schedule。

### 8.3 leaf 的贡献

leaf 决定同一骨架如何成为目标 DSL 的高质量程序：

- Triton 是否形成编译器能够识别和优化的 pointer/tile/loop/dot 结构；
- TileLang 是否形成恰当的 copy、allocation、pipeline、sync 和 primitive；
- cuTile 是否使用其原生 load/gather/collective 表达；
- target-specific alias、buffer、thread 和 launch constraints 是否正确。

这些 pattern 写得不好，即使 Plan 正确，目标源码仍可能低效。因此，**leaf 中确实存有性能知识，而且其中一部分目前可能本应被抽成 shared Plan policy；另一部分因为只对某个 target API 有意义，正当地留在 leaf。**

### 8.4 provider tuner 与下层的贡献

provider tuner 负责给既定结构选参数；下层 compiler 负责把良好的 target source 变成良好的机器代码。二者都影响最终速度，但都不能替代 Intent compiler 对 target program structure 的构造。

因此，当前性能归因不是单选题。更接近事实的排序是：

> **算法质量决定上限和大结构；leaf 与下层目前承载最多 target-specific 实现质量；shared Realizer 提供有限但正在形成的跨 provider physical policy；tuner 选参数。**

## 9. 对“我们的编译空间”的最终回答

### 9.1 当前已经存在的空间

固定同一 KIR 后，仍存在三层可由本项目负责的实现差异：

1. shared structural realization：worker/group/reuse/persistent、residency、stage、materialization、structured traversal；
2. target-local realization：目标 DSL 的 transfer/storage/loop/copy/sync/primitive/alias-safe construction；
3. target parameter surface：tile/group/stage/warp 等 tuner 参数。

第 1 和第 2 才是“Intent 到 target source”最关键的编译空间；第 3 是普通 provider tuning。

### 9.2 当前空间的实际规模

今天它是**有限而非巨大、确定性多于搜索、局部组合多于全局 schedule synthesis**。对于一个固定输入，大多数路径只产生一个 canonical Plan 和一个 canonical target source。所谓“空间”主要表示这里客观存在多个合法实现维度，以及 policy 写好或写坏会改变目标源码质量；并不表示当前系统已经把所有维度枚举出来。

这与“原本很大的实现空间被压成一条 canonical path”并不矛盾。编译器可以依据 typed KIR 和 hardware facts 确定性地做出好选择。未来即使增强，也未必需要 cost model；可以继续采用明确的 hardware rule/heuristic。只有当同一组 facts 下确实存在难以静态判断的候选时，才有必要把某一维度交给 tuner。

### 9.3 当前最实质的缺口

当前缺口不是“缺少更多 config”，也不是“尚未发射一个 TMA primitive”。宏观缺口是：

- shared hardware facts 对 canonical physical policy 的影响仍窄；
- Physical Plan、derived emission facts 和 leaf-local policy 的边界还没有完全逐项闭合；
- 现有文档把 leaf 描述得过于机械，掩盖了 leaf 实际承担的 target program construction；
- 个别 leaf 会重新扫描 KIR 并增加 Plan 未声明的 policy/tuner axis；
- 尚不能证明对固定 KIR 存在一组被明确建模、可系统比较的结构候选。

因此，真正值得推进的“建立编译空间”不是先造一个 cost model，而是先让每个 target-source branch 都能回答：

1. 它是否由 KIR 语义唯一决定；
2. 它是否是由 selected Plan 唯一推出的 derived fact；
3. 它是否是跨 provider 的 shared structural policy；
4. 它是否只因 target API/capability 而存在、因此属于 leaf；
5. 它是否只是 provider tuner 参数。

只有第 3 和第 4 类共同构成这个项目从算法到 target source 的核心优化面。

## 10. V1 的稳定判断与明确不主张

### 可以稳定主张

1. IntentDSL 当前不是 whole-op library selector；它沿 KIR/Plan 的 typed structure 组合目标程序。
2. 同一 KIR 并不唯一决定 Triton/cuTile/TileLang source；中间存在有限的 physical structure 和 target realization decisions。
3. 当前大多数决策是 deterministic canonical policies，不是 cost-model search；这不妨碍它们是编译工作。
4. leaf 不是纯 printer，而是 target-specific realizer/emitter；其大量程序构造是合理职责。
5. 当前真正需要厘清的是 shared Plan decision、derived fact 和 leaf policy 的边界，而不是把 tuner config 当作性能知识主体。

### 当前不能主张

1. 不能主张 GPU Realizer 已经覆盖了一个很大的 schedule space。
2. 不能主张当前高性能主要由 shared compiler 自动发现。
3. 不能主张所有 physical facts 都已经进入 Plan、leaf 完全不重分析 KIR。
4. 不能主张每个 target leaf 只做语法翻译。
5. 不能用 TMA、`program_id`、mask 或 autotune config 中的任何单点，代替对整个 KIR→Plan→target-source construction 的解释。

## 11. 最短答案

如果只保留一句：

> **我们编译的是“同一逻辑算法如何成为某个 GPU tile DSL 的物理程序”：shared Realizer 决定可跨 provider 表达的 canonical physical skeleton，leaf 依据该 skeleton 和 KIR 完成 provider-specific 的循环、transfer、storage 与 primitive composition。今天这条路径大体确定性、空间有限，而且性能实现相当大一部分仍在 leaf；它与整算子模板选择的根本区别，是按 KIR value/access/op 及其 typed provenance 组合程序，而不是按 op signature 选择一份完整 kernel。**
