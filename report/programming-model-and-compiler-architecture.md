# Intent 编程模型与编译器架构收敛审计

## 结论先行

Intent 的正确定位不是“另一门 Triton”，也不是通用 Python 编译器。它是一门 **Python-hosted、单 kernel、跨后端、tile-parametric 的结构化逻辑区域程序语言**：作者写完整的 kernel 内算法与逻辑工作集，编译器补上作者有意省略的机器 realization，再把同一份 realization 投影成 Triton、cuTile 或 TileLang 源码。

它的核心编程单位不是 Triton 的 program instance，也不是 CUDA thread/block，而是：

> **带逻辑坐标、结构化区域、tensor-flow、状态与 effect 的单-kernel 算法。**

编译器的核心工作也不是重新猜算法，而是：

> **把 source 已经确定的逻辑工作，兑现成轴角色、物理范围、ownership、遍历、有效区间、访问覆盖、必要存储与合法调优轴。**

所以，“我们本质上是不是在处理范围问题”的答案是：**范围与轴是主骨架，但不止范围。** 完整 realization 还包含 ownership、program folding、状态推进、padding/fill、访问覆盖、private storage、primitive 数值角色以及 effect 的物理兑现。layout、寄存器分配、指令选择、候选值和低层流水线仍交给下层 tile compiler。

当前总体分层方向是正确的，也已有很广的真实 kernel 覆盖；但实现还不能原样冻结。冻结前必须先解决三个性质不同的问题：

1. 规范坚持一个 source invocation 对应一个 target dispatch，而当前分阶段 ragged contraction 会生成并依次启动多个 target kernels。这是编程模型冲突，不是普通代码瑕疵。
2. 公开 DSL、canonical Kernel MLIR 与 realizer 的实际能力不完全闭合，存在“前端看起来支持、后端边界才拒绝”的假能力。
3. emitter 仍在若干位置从 Kernel IR 的结构或张量 shape 重建本该由 Physical Plan 直接给出的物理事实，尚未完全退化成机械投影。

本报告只做架构审计与收敛判断，没有修改编译器，也没有运行性能或数值 repro。

---

## 一、我们的编程模型到底是什么

### 1.1 最准确的名字：结构化逻辑区域程序

Intent source 描述的是一个 machine-unbound kernel：

- ABI、输入输出、alias、effects；
- logical domain、region、partition 关系；
- 坐标与 index relation；
- 独立工作 `parallel`；
- 有序工作 `ordered`；
- 带 carry 的物理分块流 `state_stream`；
- `reduce`、`scan`、`contract`；
- gather、scatter、原子、logical buffer；
- dtype、cast、数值路径、控制流；
- ragged membership 与作者声明的边界/前置条件。

Source 不描述：

- `program_id`、grid、block/thread/warp identity；
- `num_warps`、`num_stages`；
- 内部 tile 的具体值；
- register/shared/global address space；
- MMA fragment layout；
- instruction selection、register allocation、pipeline schedule。

“tile-parametric”不等于作者完全不知道区域。作者可以决定 **算法上是否存在 partition，以及主体看见一个元素还是一块区域**；但 `I.auto("...")` 的具体 extent 是物理选择。这个边界很重要：

- 是否分区、是否分成多个算法阶段，是作者的算法结构；
- 分区取 32、64 还是 128，是 realization/tuner 的机器选择；
- 编译器不能把作者写的逐元素主体偷偷改成块矩阵算法；
- 编译器可以把互相独立的标量实例打包到一个物理 program 的 lanes 中，只要 source body 的可观察语义不变。

### 1.2 它与 Triton 的根本差别

| 问题 | Triton source | Intent source |
|---|---|---|
| 作者面对的执行单位 | program instance 与块张量 | logical domain、region 与结构化 tensor-flow |
| grid/program mapping | 作者直接写 | realizer 选择 |
| tile 大小 | 通常由作者/launcher/tuner 暴露 | source 只声明 fixed 或 `auto`；合法轴由 realizer给出，值交给下层 tuner |
| mask 与 tail | 作者常显式按 block 写 | source 固定 logical validity；realizer 决定 tail/mask/fill 的兑现 |
| layout/thread mapping | Triton compiler 从显式 block program 推 | 下层 Triton/cuTile/TileLang compiler 继续负责；Intent 不复制 |
| 算法结构 | 作者写 | 作者写，且 Intent 不得替换 |
| 跨表面语言 | 不是核心合同 | 同一 Kernel IR + 同一 Physical Plan 投影到多个 GPU surface |

因此，Triton 可以概括为“作者给出 tile program，下层推 layout 与线程映射”；Intent 应概括为：

> **作者给出完整的逻辑区域算法，Intent 选择该算法在 GPU 上的物理区域实现，下层 tile compiler 再完成低层布局和指令实现。**

### 1.3 单-kernel 边界必须保留

正式语言合同已经写得很清楚：

- 一个 `@intent.kernel` invocation 对应一个 target entry invocation；
- 多 kernel 算法由普通 Python wrapper 按作者决定的顺序编排；
- 编译器不替作者决定调用次数，不把一个 source kernel 拆成 runtime-visible 的多个 dispatch；
- `@intent.fn` 只是 kernel 内 inline helper，不是另一个 entry。

这个边界不只是 API 习惯，它决定了算法所有权：拆成几遍、跨 kernel workspace、多阶段归约与同步，都是作者可观察的算法决定。

**本次收敛判断：保留这一不变量，不放宽。** 如果某算法需要多次启动，就应在 wrapper 中写成多个 `@intent.kernel`。编译器可以在单次 dispatch 内做多级 physical implementation，也可以使用 compiler-private scratch，但不能偷偷增加 launch。

---

## 二、编译器到底在做什么

### 2.1 编译器不是算法推断器

Intent compiler 不负责把“数学结果”自动变成某个高性能算法。以下变化都不允许：

- stable softmax 自动换成 online softmax；
- 普通 GEMM 自动换成 Strassen；
- 原子 bucket 自动换成排序分组；
- 一个 source kernel 自动拆成多个 runtime kernels；
- 为追求块原语，把逐元素 source body 改写成块矩阵 body。

算法差异必须由不同 DSL source 或 wrapper orchestration 表达。编译器只允许做保持 source tensor-flow、logical workset、state、effect、ABI 与调用边界不变的物理变换。

### 2.2 编译器是 realization compiler

它在“多个合法物理实现”中选择：

1. 每个逻辑轴承担哪些角色：parallel、ordered、reduction、ragged member、contraction、lane；
2. 同一轴有哪些用途不同的物理 range：ownership、traversal、reduction、lane、access；
3. 哪些轴进入 program space，怎样折叠、swizzle、复用 worker、是否 persistent；
4. logical validity 如何变成 stop、guard、mask、fill 或 checked transfer；
5. 一个输出 tile 对输入的 access footprint，包含 tail 与重叠 halo；
6. logical buffer 的生命周期、所有者与必要 residency；
7. reduction/scan/contract 的数值角色、identity、accumulator dtype 与 target primitive 合同；
8. 哪些参数是合法搜索轴，以及它们受哪些结构/资源约束。

这就是我们比 Triton source 更上一层的部分：Triton 作者必须手工写的 program geometry、tile、mask、部分 storage/launch 事实，由 Intent realizer 填一次；三门表面语言只渲染同一份决定。

### 2.3 编译器明确不做的事情

下列工作应持续交给下层：

- layout inference；
- register allocation；
- instruction selection；
- 对给定参数的 pipeline、prefetch、unroll；
- 不同 GPU 架构的具体指令路径；
- tuner 的候选具体取值、排序与最终 winner；
- 下层 JIT、module load 与 runtime launch 实现。

判断标准仍然是：**答案是否依赖只有 Intent 的算法结构才知道的信息。** 依赖，realizer 决定；不依赖，交给下层。下层变强后，Intent emitter 应该变薄。

---

## 三、系统实际组成与具体数据流

```text
普通 Python wrapper
  │  负责 shape/device 检查、输出/跨 kernel workspace、多 kernel 编排
  ▼
Python DSL Frontend
  │  AST、constexpr、symbol/shape/region、源码位置、就地诊断
  ▼
Canonical Intent Kernel MLIR                 ← 唯一算法真理
  │
  ├─ Kernel IR verifier
  └─ KernelFacts / def-use / provenance       ← 可重算的派生分析，不是表示层
  ▼
Physical Plan MLIR                           ← 唯一已选物理决定
  │  Axis / Range / Program / Boundary / Transfer / Storage /
  │  Reduction / Scan / Contract / SearchSpace ...
  ▼
Common surface projection + target leaf
  │  能力检查、逐 op 投影、目标语法、编译/运行接线
  ├─ Triton Python
  ├─ cuTile Python
  └─ TileLang Python
  ▼
下层 target compiler
  │  layout、线程/warp 细化、寄存器、指令、pipeline、JIT
  ▼
单次 runtime dispatch
```

### 3.1 Python Frontend

当前 frontend 的正确职责是：

- 解析受限 Python AST；
- 处理 `Constexpr`、symbolic shape、region nesting、source location；
- 维护 lowering 期间的 `ValueType` 等临时状态；
- 内联 `@intent.fn`；
- 将普通 Python `break`/`continue` 正规化成 loop-carried control state；
- 在构造 IR 时直接报 source-located error；
- 直接创建 canonical Intent MLIR。

这些临时 Python 类型不是第二套持久化 Kernel IR。当前 pipeline 也没有 Python Plan、Python emitter 或 Python physical decision 旁路，这一点符合目标架构。

### 3.2 Canonical Kernel MLIR

Kernel IR 应是 source-visible 算法的唯一真理，保存：

- ABI 与 view access mode；
- stable node/value IDs；
- logical domain/partition/ragged relation；
- structured regions 与 carried state；
- tensor-flow、index relation、dtype；
- reduce/scan/contract；
- memory effects 与 source preconditions。

后续阶段不得回到 Python object 猜算法，也不得让 derived facts 变成第二份语义真理。

### 3.3 KernelFacts 不是第四层 IR

`KernelFacts` 包含 ABI、region tree、axis provenance、def-use、access range、contraction/scan/ragged/stream 等索引。它是从 Kernel IR 可重算的分析缓存，合理存在，但必须遵守：

- 没有独立序列化 schema；
- 不承载“多个合法方案中选了哪个”；
- 不拥有一套与 Kernel IR 冲突的 verifier；
- 丢失时可以从 Kernel IR 重建；
- emitter 不应绕过 Kernel IR/Plan，把它当第三份输入。

### 3.4 Physical Plan MLIR

Plan 是唯一承重的物理表示，只保存不能从算法 IR 唯一推出、且确实由我们决定的事实：

- `Program`：program root、是否 persistent；
- `Axis`：角色、program order、worker/fold、reuse/group；
- `Range`：某轴在 ownership/traversal/reduction/lane/access 用途下的 tile 与层级；
- `BlockExtent`：需要物理取整的 extent 与 fill；
- boundary/transfer/padding；
- buffer residency 与 workspace；
- reduction/scan/contract 的物理绑定；
- 合法 autotune 参数轴。

Plan 通过 stable node/value IDs 绑定 Kernel IR。它不是目标方言，也不应该分别存在 Triton Plan、cuTile Plan 与 TileLang Plan。

### 3.5 Surface projection 与 target leaf

发射层的合法职责只有：

1. 检查目标是否能表达 Plan 要求的概念；
2. 把概念映射到目标语法或目标原语；
3. 按 Kernel IR 的 def-use 逐 op 发射；
4. 生成 ABI、JIT、workspace 物化与 launch 接线；
5. 对不能表达的组合给 source-located unsupported。

三门语言的合法差异是：

- 能表达的 Plan 子集不同；
- 某概念是否需要显式拼写；
- 目标 API、编译器与 runtime 接线不同；
- 对应的 target primitive 不同。

它们不应各自重做 ownership、tile、stream stop、padding、storage 或算法阶段选择。

---

## 四、DSL 表面：什么应该留，什么不能随便加

### 4.1 当前 Core 的稳定主干

下列构造确实表达作者独有的算法信息，应保留为 Core：

- `domain`、domain product、`partition`、`indices`；
- `parallel`、`ordered`、`state_stream`；
- tensor expression、broadcast、reshape、transpose、record、cast；
- `reduce`、`scan`、`contract`；
- gather、unique scatter、reduction scatter；
- ragged descriptor 与 member relation；
- logical buffer、mutable load/store；
- atomic add/CAS 与明确 effect；
- counter-based random；
- scalar `if/for/while` 与 frontend-normalized break/continue；
- kernel-local inline `@intent.fn`。

这些不是“为了某个 kernel 加的名字”，而是可组合的算法结构。

### 4.2 后来加入、但需要正式定性的构造

| 构造 | 正确归属 | 当前判断 |
|---|---|---|
| `I.sigmoid` | 数学/数值语义 | 可以保留；它与 `exp`、`log` 同类，不是调度 hint |
| `I.end(domain_or_region)` | 作者声明的 logical read/stream endpoint | 应保留，但必须在 DSL 文档正式定义，不应只靠 attention 示例暗示 |
| `I.assume_in_bounds` | 作者给出的调用前置条件/合法性合同 | 可以保留，但应明确是 unsafe assertion，不是 compiler 推断结果，也不是性能 hint |
| `I.arg_reduce.max` | 同时返回值与位置的算法原语 | 应保留；必须正式写明 single-axis、lowest-index tie、index dtype 与 identity 语义 |
| `I.sparse_contract_2to4` | 稀疏数据格式 + 稀疏 contraction 语义 | 不能继续处于“无规范的专用 builtin”状态。要么把它正式定义为 generic sparse contraction 的首个 format，要么移到算法库层；不能把 TileLang/CUTLASS 的 target primitive 直接当 Core 理由 |
| `I.fence` | 无共同 portable semantics | 不应继续作为公开可调用能力。当前它被导出但 frontend 永远报错，这是假的 public surface；应从公开入口移除，直到有明确跨目标合同 |

### 4.3 新 DSL 构造的准入门槛

以后默认 **不增加 DSL 构造**。只有同时满足下面条件才允许增加：

1. 某个真实算法必须表达这个语义；
2. 正确答案依赖作者知道的算法信息，而不是机器或下层 compiler 信息；
3. 现有 Core 不能自然组合表达，且不是换一种方便写法就能绕过；
4. 能给出 target-independent 的语义、类型、effect 与错误条件；
5. 有 canonical Kernel IR 节点或明确的 frontend normalization；
6. Kernel IR verifier 能守住合同；
7. 至少一个后端能机械投影，其他后端能在 emission 前明确 capability rejection；
8. 文档与真实 repro 同步闭合。

“某个后端有一个好用 API”“某个 kernel 用它会快”“加一个 builtin 最省事”都不是准入理由。

---

## 五、按层定位当前缺陷

### 5.1 编程模型/DSL 问题：数量少，但必须先定合同

#### 复合 combiner 的文档与实现不一致

`doc/dsl/tensor-flow.md` 展示了把自定义 `@intent.fn welford_combine` 传给 `I.reduce`；当前 frontend 的 `_callable_symbol` 只接受内建 `Intrinsic`，自定义 helper 会报错。这里不能模糊处理：

- 若 Core 要支持自定义 monoid/record combiner，需要结构化 combiner region 的 Kernel IR 与 lowering；
- 若暂时只支持固定 intrinsic，就应删除该文档承诺并明确 operator set。

目前属于 **语言合同未闭合**，不是 emitter 缺 op spelling。

#### `partition(count=...)` 是公开能力但 realizer 只接受 extent

Frontend 与文档允许 `partition(count=P)`；`KernelFacts` 只接受 `intent.mode == "extent"` 且 extent 为 named auto 或固定值。它在前端成功、realization 才失败。

`count` 会影响 wrapper-visible/算法可见分区数量，属于 source 决定，不应被静默改写成 extent。冻结前必须二选一：真正实现 count realization，或从公开合同删除。

#### strided/dynamic domain 表面宽于 GPU realization

Frontend 的 domain 接受 start/stop/step；GPU realization 实际主要闭合 unit-step、静态或特定 runtime sequential domain。任意 stride 目前不是普遍支持。应把 frontend 接受范围和 realizer 合同对齐，不能让后端边界承担基础语言诊断。

#### `state_stream` 的 runtime extent 实际要求正编译期常量

Frontend 能生成 runtime extent operand；`KernelFacts` 随后要求它来自正的 `intent.constant`。这不是 runtime extent。应实现真正动态 extent，或在 frontend/规范将它命名并限制为 fixed extent。

#### logical buffer 的表面语义宽，实际 realization 窄

规范把 `I.buffer` 描述为一般 kernel-local logical mutable object；当前 realizer 要求它是某一个 `parallel` owner 的直接 child，且访问需静态可界定或带前置条件。这个限制可以合理存在，但必须是语言能力边界，而不能伪装成一般 buffer 后在深层报错。

### 5.2 Frontend → Kernel IR：不要丢作者已经写下的东西

这一层应只做语法正规化与 canonical construction。典型故障定位规则：

- source 已写 index expression，IR 丢掉 offset/div/mod → frontend bug；
- source 写了 region stop，IR 只留下 bool → frontend/IR schema bug；
- source 写了 alias/effect，IR ABI metadata 没保存 → frontend bug；
- break/continue 被正规化成 carried state → 合法 frontend normalization，不需要 IR `break` op；
- helper 被内联 → 合法 frontend normalization，但不能改变 helper 数值语义。

这一层不应创建 physical tile、storage、program mapping 或 target capability。

### 5.3 Canonical Kernel MLIR：当前最薄弱的承重边界

当前 ODS 大量使用 `AnyType`，logical types 又把核心结构编码成字符串 `spec`。许多 rank、dtype、operand/result、attribute 兼容性靠后续 `KernelFacts` 手工检查。结果是 parser 接受的“canonical IR”范围比真正能 realization 的范围大。

这不表示要再造一套 typed Python IR；正确修法是让 MLIR 自己更权威：

- 将能静态表达的 operand/result/type constraint 下沉到 ODS/type；
- op verifier 检查 attribute schema 与 SSA type/rank/dtype 一致性；
- Kernel IR verifier 守 stable IDs、region schema、ABI、effects；
- `KernelFacts` 只分析 provenance/legality，不补一遍 IR 类型系统。

还有一个具体 ID 闭合问题：frontend builder 为 region block arguments 写入 `intent.region_argument_nodes`，Kernel IR verifier 也检查这些 ID；但 `analyzeKernel()` 的 `KernelModel.values/valueIDs` 只索引 ABI arguments 和 operation results，没有索引 block arguments。于是“所有 canonical values 都有稳定 ID”的合同在公共 KernelModel 中并未闭合，叶子只能直接重读 metadata。

### 5.4 Derived analysis / Realizer：决定应组合，不应按 kernel 分类

好的部分已经成立：当前 realizer 没有按 kernel symbol/name 分支，轴角色可组合，Range 也按 purpose 区分 ownership、traversal、reduction、lane 与 access。

仍需收敛的部分：

#### GPU program root 被限制为恰好一个顶层 parallel

`programRoot()` 要求一个 GPU kernel 函数恰好含一个 outer `intent.parallel`；纯 sequential root、两个独立顶层 program region 都不在当前实现内。这个限制未必错误，但它必须成为正式 GPU capability，而不是被 DSL 的“完整单 kernel”表述掩盖。

#### `axisFromLabel`/shape fallback 会制造第二份 provenance

当精确 domain provenance 缺失时，通过相同 extent label 选择第一个 domain，或构造 implicit axis，会把“名字/shape 恰好相同”当作逻辑身份。正确方向是沿 SSA、region argument 与显式 index relation 保存唯一来源；缺失就诊断，不应猜。

#### persistent 决定仍是固定结构启发式

当前条件近似为：某 contraction 位于至少三个 parallel axes 下，其中至少两个 tiled，且没有 ragged，即把 program axes 折叠为 persistent traversal。它不是 kernel-name 特判，但仍是一个 whole-program 静态规则。两台机器尚未证明它必须成为搜索维；现阶段可保留为受控 policy，但必须单独登记，不能继续散落成“默认正确”的魔法条件。

#### private buffer residency 仍有经验常数与 workspace 未决项

buffer placement 读取寄存器容量，但容量预算使用固定除数；unstructured dynamic access 一律退到全局 private workspace。它保证可实现，不等于选对物理位置。这里属于 realizer policy，而不是 DSL 或 emitter。

#### scan 当前只真正闭合 inclusive add

Frontend 表面接受 `combine=`；Physical Plan/leaf 当前核心路径按 `scan_inclusive_add` 兑现。要么将 Core 明确限制为 inclusive add，要么让 Plan 保存并验证一般 associative combine。不能靠 frontend generic 参数制造假能力。

#### SearchSpace 只保存参数名，没有合法候选关系

`intent_plan.autotune` 当前只有 `key` 和 `parameters`。这能把具体值交给下层，但不能表达 realizer 推出的合法集合、参数耦合和资源上界。我们不应自建 cost model；但“哪些候选合法”若依赖算法结构，仍应有可传递的合同，而不是只给下层一个名字。

### 5.5 Physical Plan → emitter：仍有几处假发射

共享 `SurfacePlan` 已经统一了 Axis/Range、program folding、workspace offset、scan/stage 等大量投影；三个 leaf 也没有按 kernel 名字或整 kernel matcher 发射。这一方向正确。

但以下事实仍被重新拼装：

- 三个 leaf 都从 `parallel/state_stream` region 名字重新选择 `ownership/traversal` range，再用 `intent.region_argument_nodes` 配对 region argument；
- row-vector tile 会从 logical domain dimension 与 `BlockExtent` 再推一次 physical extent；
- `SurfacePlan::indexCanonicalStructure` 从 Kernel IR use-def 重建 ragged outer/member 与 stream stop/axis binding；
- target leaf 的 `dimensionName` 会从 domain → dim → ABI view shape 恢复维度名字。

其中 ABI shape 拼写本身是合法 leaf 工作；但 **已经属于物理决定的 range/axis/binding 不应靠 region 名字和 shape 再选一次**。Plan 应直接绑定“这个 region argument 消费哪个 range”“这个 emitted axis 的最终 physical extent 是什么”。修完后叶子只查 binding，不再重建。

另外，TileLang `fp8_mqa_logits` 在能力检查阶段被接受，最终在下层 CUTLASS FP8 MMA assertion 失败。这说明 capability predicate 仍偏宽；应在 emitter 前明确拒绝该不能兑现的组合，而不是把下层 crash 记作支持。

### 5.6 下层 target compiler：到这里就应停止向上加机制

以下故障原则上属于下层：

- 首次 JIT 编译超时；
- layout inference 找不到合法布局；
- 某候选 shared memory/register 超限但 tuner 能筛掉；
- target runtime/module load 失败；
- 某架构上的 CUTLASS/编译器 bug；
- 同一合法搜索空间中不同 provider/device 选出不同赢家。

处理方式是：改机械拼写、缩窄 capability，或交给下层升级。除非能证明 Plan 缺了一个依赖算法结构的决定，否则不能往共享层增加机制。

---

## 六、冻结前必须解决的架构冲突

### 6.1 分阶段 ragged contraction 违反 single-kernel invariant

这是本次审计最重要的发现。

Realizer 的 `contractionStages()` 会：

- 找出具有 ragged axis 且最终到达 scatter terminal 的 contractions；
- 沿 def-use 收集每一 stage 的 operation slice；
- 在 Plan 中生成 `StageOp`、stage inputs/outputs、terminal 与 stage axes。

三个 emitter 随后都会：

- 生成 `kernel_stage_0`、`kernel_stage_1` 等多个目标 kernel；
- 分配 stage workspace；
- 在生成的 `launch()` 中逐 stage 发起独立 target launch。

这与 `doc/dsl/model.md` 和 `doc/compiler/compiled-artifact.md` 的不变量直接冲突：compiler-private 多级实现不得暴露为额外 runtime dispatch。

这条路径也不是单纯“目标语言内部实现”：每个 stage 都有独立 grid、autotune、workspace 与 launch，调用次数已经改变。

**收敛决定：不放宽语言模型，分阶段 runtime orchestration 回到作者 wrapper。**

因此冻结前应做到：

- 一个 `@intent.kernel` 的 Plan 不再含会产生额外 dispatch 的 `StageOp`；
- 需要分阶段的 MoE/grouped contraction 等算法写成多个 source kernels；
- wrapper 明确分配跨 kernel workspace 并决定调用顺序；
- Plan 可保留单次 dispatch 内的 stage/pipeline 概念，但不能复用当前“operation slice → 多 kernel launch”的语义。

否则必须反过来正式改写整个语言定义为“一个 source kernel 对应一个 callable artifact，artifact 可含多次 dispatch”。这会改变作者/编译器的算法所有权，并与此前所有收敛原则冲突，本报告不建议这样做。

### 6.2 公开 surface 必须与可实现合同对齐

冻结不要求所有想象中的能力都实现，但要求不存在假能力。下列项必须逐一选择“实现”或“前端提前拒绝/从文档删除”：

- custom reduce/scan/contract combiner；
- `partition(count=...)`；
- 非 unit-step domain；
- 真正 runtime `state_stream` extent；
- 一般 logical buffer placement/ownership；
- 一般 scan combine；
- public `I.fence`；
- 2:4 sparse contraction 的 Core 定位。

### 6.3 Kernel MLIR 必须成为真正的唯一语义边界

冻结前至少需要闭合：

- region block arguments 的 stable value ID 索引；
- ODS/type/verifier 能表达的静态约束不再推迟到 target facts；
- attribute schema 与 SSA type/rank/dtype 的一致性；
- 不允许 extent label/shape fallback 猜 logical axis identity。

否则“canonical”只是一种文本格式，而不是后端可以独立信任的算法合同。

### 6.4 Plan 必须给 emitter 足够精确的 binding

冻结前应消除这几种重建：

- region argument → range purpose；
- row-vector logical extent → final physical extent；
- ragged/stream 的物理 binding；
- 已选 program/range 事实在三个 leaf 中各拼一遍。

不是把所有 Kernel IR 信息复制进 Plan；只把“多个合法物理方案中已经选了哪个”保存下来。纯算法结构仍从 Kernel IR 读，纯 spelling 仍留在 leaf。

---

## 七、以后每类问题固定落在哪里

| 观察到的现象 | 应先检查的层 | 正确处理 |
|---|---|---|
| 作者无法说出算法必需的语义 | DSL / programming model | 先证明现有 Core 不能组合表达，再提新构造；同步定义 IR/type/effect |
| 作者已经写出 X，但 Kernel MLIR 没保存 | Frontend → Kernel IR | 修 lowering/schema，禁止后续重新推 X |
| MLIR parser 接受，到了 facts 才发现基本 rank/type/schema 错 | Kernel IR verifier | 把能静态验证的合同下沉到 ODS/type/verifier |
| IR 语义明确，但没有轴角色、range、validity、storage 决定 | Realizer / Physical Plan | 增加共享 per-op fact、legality proof 或 physical decision；不按 kernel 名分支 |
| 三个后端各自从 shape/role 拼同一个物理事实 | Plan → emitter boundary | 在 Plan 建唯一 binding，删除三份重建 |
| Plan 已经明确，但某 surface 没有对应表达 | Target capability / leaf | 明确 unsupported，给源码位置；不能加慢几个数量级的伪支持 |
| 目标源码结构异常庞大、原语选错 | Target leaf spelling | 换机械等价的目标原语，不改 Kernel IR/Plan |
| 目标源码合理但 JIT timeout/layout/资源候选失败 | 下层 compiler/tuner | 交给下层、缩窄 capability 或候选；没有算法证据就不上移机制 |
| 同一 source 需要多次 kernel launch | Python wrapper / 多个 source kernels | 作者明确编排；不能由 single-kernel realizer 自动拆分 |
| 某 provider 快很多 | 先并排读生成源码与 capability | 判断 spelling、Plan 欠定或真实后端边界；不能直接归因下层质量 |

这张表就是后续收尾的定位规则。以后不再用“这个 kernel 特殊”作为入口，也不再因为 emitter 某段很长就机械上移；只看该事实属于算法、已选物理决定、目标拼写还是下层实现。

---

## 八、当前证据能证明什么，不能证明什么

当前两份 baseline 各有 113 个 kernel/case、339 个 provider cells；其中 100 个 case 在 Triton、cuTile、TileLang 三个 provider 上都通过。语料已经覆盖：

- pointwise、broadcast、reshape、transpose；
- 多种 reduction、scan、compaction；
- GEMM、batched/grouped/dual/quantized/sparse contraction；
- dense/varlen/paged/MLA/块稀疏 attention；
- ragged、indirect、atomic、scatter；
- logical buffer、动态规划、排序、控制流；
- convolution、状态空间与反向；
- 大量等价表达与分解变体。

这足以证明：当前架构不是 softmax/rowwise 的一次特化，也没有按 kernel name 建十几条后端。

但它不能证明：

- 所有公开 DSL 组合都闭合；
- 任意 shape、stride、空域与 runtime extent 都正确；
- Kernel IR verifier 已经完整；
- 三个 leaf 完全不重建物理事实；
- 每个 pass 都严格遵守 single-dispatch；
- `unsupported`、`compile_timeout`、`failed` 都处于正确边界。

现有 H100 表中仍有 14 个 unsupported、2 个 compile timeout 和 1 个 failed cell；这些大多是 target/downstream capability 边界，但 `fp8_mqa_logits` 的 failed 也说明 capability 检查还没有完全挡住下层不能兑现的组合。

因此，覆盖广度是架构成立的证据，不是冻结语义合同的替代品。

---

## 九、最终应冻结成什么样

### 编程模型

一门面向 **结构化逻辑区域单-kernel 算法** 的 Python eDSL。作者拥有算法、ABI、状态、effects、逻辑索引、partition 是否存在以及多 kernel 编排；机器相关的 tile 值、program mapping、validity 兑现和 storage placement 被抽掉。

### 编译器

一个 realization compiler：

```text
作者算法
  → canonical Kernel MLIR
  → 可重算的 facts
  → 一次、共享的 Physical Plan
  → 三个只做 capability + projection + spelling 的 GPU surface
  → 下层 compiler
```

### 只有三份承重表示

1. Source/Kernel IR：算法真理；
2. Physical Plan：已选机器 realization；
3. Target source：机械投影产物，不再是需要独立验证的 IR。

KernelFacts、SurfacePlan index、target bindings 都只是派生索引，不拥有独立语义。

### 冻结后的变化规则

- 新 kernel 优先只增加 DSL source；
- 真有新算法语义才增加 Core op；
- 新 target 只增加 capability、op mapping、API/runtime glue；
- shared realizer 只增加可组合 fact/legality/decision，不增加 kernel 分类入口；
- emitter 不再从 shape、角色名或周围结构重建 Plan 已选事实；
- 下层已经做好的事不上移；
- 一个 source kernel 永远不被拆成额外 runtime dispatch。

## 最终判断

架构思想已经收敛：**Intent 不是 tile 语言本身，而是位于算法 source 与 tile 语言之间、负责逻辑区域到机器 realization 的单-kernel 编译器。** 当前代码的大部分也已经符合这一形状。

真正还挡着“之后不要再乱动”的，不是缺更多 kernel，而是四个边界要闭合：single-dispatch、公开 DSL 合同、canonical Kernel MLIR、Plan-to-leaf 唯一 binding。把这四处处理完，编程模型就可以冻结；之后的问题都能按本报告第七节落到明确层次，而不再通过扩 DSL、加 kernel 分支或让某个后端长成第二个编译器来解决。
