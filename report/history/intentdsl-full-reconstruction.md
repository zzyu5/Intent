# IntentDSL 完整重构：讨论脉络与持续判断

## 背景与总目标

IntentDSL 已经经历了多轮 DSL、compiler、三后端、examples 和 baseline 推进，但最近的编译器自查暴露了基础问题：当前 Physical Program 仍主要是 KIR 同构副本加 side records，shared passes 多数没有改写 executable program，真正的 target program 仍由厚 materializer 结合 KIR 现场构造。详细证据保存在 [`gpu-compiler-space-pass-and-baseline-contract.md`](gpu-compiler-space-pass-and-baseline-contract.md)。

这意味着问题不能继续靠补 pass、leaf form 或 benchmark 格子解决。Compiler V2 中关于语义权威、physical program 和 lowering 边界的内容可以参考，但它不是已经证明正确的最终设计，更不能用已有代码反向定义规格。

因此你决定进入一次完整重构：先清理混入旧假设的 `doc/`、baseline 与 history，再从作者编程模型和 DSL 开始建立真正规格；你审查这一层后，继续定义 Kernel IR、compiler decision space、executable physical program、pass pipeline、target/backend 边界，以及最终 examples/source/runtime/baseline。当前讨论只走到了第一阶段，不代表整份重构报告只讨论 DSL。

完整重构的主问题始终是：

1. 作者写下哪些跨后端算法语义；
2. canonical Kernel IR 必须保存什么；
3. Intent compiler 要生成和优化什么程序；
4. Triton、cuTile、TileLang 以及未来 CPU/RISC-V/RVV 后端继续完成什么；
5. 怎样用真实 source、examples 和 runtime 证明整条链成立且高性能。

## 当前阶段：先固定 programming model 与 DSL

Intent 要占据 Triton kernel 的生态位置：用户仍在普通 Python 项目里写 kernel，多个 kernels 仍由 Python wrapper 编排；但 Intent 作者写跨后端 logical kernel algorithm，而不是 GPU block/SPMD skeleton。只有先固定这一点，后续 compiler 才知道哪些结构必须保持、哪些 realization 可以自由选择。

下面按真实发生的讨论轮次记录你的输入、我的输出和当前校正。

## 第一轮：清理旧规格，从 DSL 开始完整重构

### 你的输入

> “所以现在全错……doc 内容全部删除，report 错误的 baseline/baselinev2、history 之类的全部删除……接下来第一步，还是先建立出我们真正的 spec……别的先不做，我审查确定后，再进行后面编译器等的文档 spec 的最终编写。”

你同时强调 Compiler V2 不是既定真理，只能参考其中仍成立的编程模型与边界判断。随后又纠正我：新规格仍然放在 `doc/`，不是另建顶层 `spec/`。

### 我的输出

我在 `doc/programming-model/` 与 `doc/dsl/` 建立了新草案和理想化 examples，并一并清理了旧文档、baseline/history。

但草案混入了未经确认的设计：用 `Core/sugar` 分类语言、删除 partition、改变 state-stream extent，还写入了部分当前 frontend 不存在的理想语法。

### 当前校正

清理旧权威、先写 programming model/DSL、审定后再进入 compiler，是完整重构的阶段顺序。新 `doc/` 只是第一阶段草案，不因写入稳定目录就自动正确；后续 compiler 重构仍是这份持续报告的主体之一。

## 第二轮：追问事实依据与 `Core`

### 你的输入

> “这些文档，都是根据实际情况来的是么？没有再自己修改、YY 啥的？……Core 这个是啥意思，你自己定义的？”

### 我的输出

我承认新文档并非全部来自已确认事实：partition、state-stream extent、generic reduce surface、typed format 等包含自己的设计判断。`Core` 是我借用的语言设计术语，不是项目已经确认的正式概念。

### 当前校正

不能再用 `Core/sugar` 二分替项目做决定。当前更可靠的三层是：

1. 作者写下的跨后端算法语义；
2. 语言明确交给 Intent compiler 绑定的 realization 自由；
3. target 与下层 compiler 的具体实现。

便利写法能否在 frontend 内部归一化，不决定它是否应从 public DSL 删除。

## 第三轮：`reduce.sum`、partition、RVV 与现有 examples

### 你的输入

> “所以呢？现在情况是什么，你不打算用 reduce.sum？为什么不用 partition？……我们还得接 RISC-V……不要再大改目前 example 中已经写好的 DSL。”

### 我的输出

我确认：

- `reduce.sum/max` 应保留，generic reduce 用于自定义 combine，两者并存；
- partition 中可观察的 part identity、boundary 和 partial ABI 是跨后端算法；
- domain、parallel、state、reduce、scan、contract、ragged 和 effects 不因跨后端就应被删成纯数学表达；
- examples 应默认保留，只在算法与 upstream 不一致或明确属于旧 compiler workaround 时定向修改。

但我随后又把 `state_stream(..., extent=I.auto("K_TILE"))` 整体解释成应保留的跨后端参数化结构，没有先核对此前 DSL 重构为什么删除作者可见 blocking。

### 当前校正

- `partition(count=P)` 保留，因为 part identity 和中间 ABI 可观察；
- 固定 `partition(extent=B)` 在 B 是算法中的 chunk/page/window 时保留；
- `partition(auto)` 保持删除，普通 physical blocking 不由作者预建；
- examples 是语言设计证据，不为配合新文档而整体迁移。

## 第四轮：重新审视 `K_TILE`、Triton 编程性与跨后端语言

### 你的输入

> “怎么还有 K_TILE？不是之前有一次重构说不保留这些东西了吗？……到底我们的 DSL 要设计成什么样子，能够让作者写下来正确算法结构，而不是硬件相关？……语法尽量参考 Triton，我们毕竟是替代 Triton 的生态位置。”

### 我的输出

历史和当前代码核查确认：此前重构删除了 `partition(auto)` 和 public `ordered`，但有意把 `I.auto(...)` 保留给 `state_stream`，称为 segment-parametric recurrence。当前 60 个 `*_TILE` 都集中在这条路径，因此不是偶然回归，而是当时留下的一项编程模型选择。

进一步形成的判断是：

- `state_stream` 本身可以保留，因为作者选择的是 region-level、有序、携带状态的真实算法；
- ordinary scalar loop 不能被 compiler 擅自升级成 state stream；
- 固定 extent 在模型 chunk、page 或 window 本身属于算法时可以保留；
- compiler-selected segmentation 可以在 GPU、CPU、RVV 上使用不同 realization；
- 但作者可见的 `I.auto("K_TILE")` 泄漏了 tuning/physical parameter identity，不应作为最终作者表面；
- Intent 应借用 Triton 已被接受的 Python 函数、普通控制流、constexpr、索引、reduce/scan/contract、helper 和可检查产物，而不照搬 `program_id`、block size、pointer、physical mask、warp/stage/TMA/layout。

### 当前校正

Intent 替代的是 Triton kernel 的生态位置，不复制 Triton 的 GPU block programming model。作者选择算法结构；Intent compiler 从这些语义生成跨后端 physical program；各 target/lower compiler 完成 layout、primitive、pipeline 与机器 lowering。

## 第五轮：`state_stream` 到底要不要保留

### 你的输入

> “`/home/kingdom/intentdsl-state-stream-thinking.md` 是我和 Claude 讨论我们到底要不要 state_stream。我现在都糊涂了，你根据我们一直以来的抽象给我想清楚。”

那份讨论最后走到 P7：删除 `state_stream`，让所有普通有序循环直接迭代 compiler-chosen region。它的出发点是对的——物理 extent 不应由作者写，普通循环也已经能承载 SSA carry——但最后把适用范围推得过宽。

### 我的输出

对当前 frontend、74 个真实用例以及 Triton/TileLang 的循环模型重新核查后，得到四项事实：

1. 当前普通 `for x in domain` 的 `x` 是逻辑标量；只有遍历 partition 时 body 才看见 region。普通循环还保留逐次 carry、`break/continue` 和每次迭代的 effects。把所有普通循环 region 化会改变程序，不是 lowering。
2. 当前 `state_stream` 真正额外表达的是：沿一个逻辑范围有序地遍历连续 region，region 的物理 extent 可以由 compiler 选择，carry 按 region 顺序传递。它不是普通 `for` 的别名。
3. 但它把不必绑在一起的东西绑死了：compiler-chosen segmentation、普通 loop carry、显式 `yield_`、单独的 `stop` 和至少一个 state。74 个用例中有 10 个根本不更新 state，只是借它获得 compiler-chosen region。
4. Triton 和 TileLang 都不会把任意标量顺序循环自动改成 region recurrence。它们只优化作者已经表达的 loop、reduce、scan 或 tile structure。因此 Intent 若要选择分段，作者程序必须有一条明确的 segmentation contract，不能靠 compiler 猜。

### 当前校正

**应保留的是 state-stream 背后的语义能力，不是当前 `I.state_stream` 这个独立构造。**

更基础、也更可组合的模型是：

```python
maximum = I.cast(-I.inf, I.f32)
denominator = I.cast(0.0, I.f32)

for column_region in I.partition(columns):
    values = x[row, column_region]
    local_maximum = I.reduce.max(values, axis=0, identity=-I.inf)
    next_maximum = I.maximum(maximum, local_maximum)
    denominator = (
        I.exp(maximum - next_maximum) * denominator
        + I.reduce.sum(I.exp(values - next_maximum), axis=0, identity=0.0)
    )
    maximum = next_maximum
```

这里 `partition(columns)` 不带 extent/count，表示作者声明 body 对任意合法的连续有序分段成立；compiler 选择实际分段。状态由普通 `for` 的 loop-carried SSA 承载。它与另外两类 partition 必须严格区分：

- `partition(axis)`：无稳定 part identity，边界由 compiler 选择；作者承担 segment-parametric 合同；
- `partition(axis, extent=E)`：`E` 是算法中的 page、group、window 或 chunk boundary；
- `partition(axis, count=P)`：part identity、数量或 partial ABI 对作者可见。

因此本轮的明确结论是：

- 不保留当前 public `state_stream(..., extent=..., init=..., stop=...)` 形态；
- 不把所有普通 `for` 自动 region 化；
- 用 compiler-chosen logical partition、普通 `for` 和普通 SSA carry 组合出同一能力；
- KIR 仍必须明确保存“有序、连续、无重叠、完整覆盖、compiler-chosen segmentation”的合同，不能只把 `StateStreamOp` 改名成 `ForOp`；
- logical stop 应成为遍历范围的一部分，而不是 physical extent，也不能在删除 `state_stream` 时丢失；
- 这是一项待你审定的编程模型结论，当前 DSL、KIR、examples 和 lowering 尚未据此修改。

## 第六轮：first-class reduction 纠正了“普通代码由 compiler 猜结构”的方向

### 你的输入

你和 Claude 一度用 TileLang 的普通 fragment、串行循环和 `T.max` 更新说明 reduction 可以完全从普通代码推导；随后你亲自核查 Triton/TileLang，确认这个判断是错的：

> Triton 依赖 first-class `tt.reduce`，作者通过 `tl.max(...)` 或 `tl.reduce(..., combine_fn)` 明确建立它；TileLang 依赖 `tileop.reduce` 或 reducer epoch。两者都不会看见任意普通 `for` 后猜出 reduce-max。

你据此重新追问：`state_stream` 和 `partition(count=P)` 是否仍应保留。

### 我的输出

真实实现支持这个纠正：

- Triton 的普通 `tl.range` 只 lower 成带 carry 的 `scf.for`；`tl.maximum` 是逐元素运算。只有 `tl.max/tl.reduce` 才产生带 combine region 的 `tt.reduce`，并进入后端 reduction layout/tree lowering。
- TileLang 的普通 fragment `BufferStore` 按作者写下的 serial/parallel 结构 lowering。只有显式 `T.reduce_*`、`alloc_reducer` 或 reducer epoch 才进入 reducer planning；它不会从任意 buffer update 证明结合律并创造 reduction。
- 因而“用基础循环能够表达”只证明语言有 fallback expression path，不等于 compiler 有权或有能力把它提升为 first-class structured operation。

这使上一轮提出的 `partition(axis)` 替代方案失去依据：它仍然需要作者声明“这段普通 loop 允许 compiler 任意分段”，只是把声明从 `state_stream` 换到了 partition；同时还把作者可见 partition 与 ordered state transition 混成了一个概念。

### 当前校正

`state_stream` 与 `partition(count=P)` 都应保留，但理由完全不同：

1. **`state_stream` 是 first-class ordered region recurrence。**作者明确写下“沿逻辑范围按连续 region 推进，并按顺序携带 state”；compiler 选择 region extent、target loop form 和物理实现，但不能从普通 scalar `for` 猜出这个结构，也不能把它重排成 reduce。它和 `reduce/scan/contract` 一样，是带 typed body/region/result 的算法 IR 节点，不是附着在任意代码上的优化许可。
2. **`partition(count=P)` 是作者可观察的逻辑分解。**`P`、part identity、每个 part 的边界以及可能的 partial-buffer ABI 属于算法或调用合同；GPU、CPU、RVV 都必须保存同一个 P-part 语义。它不是 compiler blocking，也不应由 tuner 改写。
3. **普通 `for` 继续保持普通顺序语义。**它可以正确表达 scalar recurrence，也可以作为低层写法存在；但 compiler 不负责从任意 loop/buffer update 反推 first-class reduce、scan、contract 或 state stream。
4. **当前 `state_stream` surface 仍需收敛。**`extent=I.auto("K_TILE")` 泄漏 physical/tuning identity，应由 compiler 决定；固定 extent 只有在 page/group/window/chunk 本身是算法边界时才由作者表达。当前强制 `with stream`、显式 `yield_` 和至少一个 state 是否保留，是表面可编程性问题，不改变 first-class state-stream 语义必须存在的结论。

因此，第五轮关于“删除 `StateStreamOp`、用 compiler-chosen partition + 普通 For 取代”的结论正式撤回。保留第五轮原文是为了记录错误是怎样产生的，不把历史讨论伪装成从未发生。

## 第七轮：为什么 Triton/TileLang 不需要 `state_stream` 与 `partition`

### 你的输入

> “为什么 Triton/TileLang 不用 state_stream/partition？Triton 明明能用普通算法结构表达 online softmax，我们屏蔽硬件以后，为什么不是沿用同样的算法抽象层级，而要把它变成一个 op？”

你提出用这一问题反过来检验 Intent 新增抽象的必要性：不能因为当前 compiler/materializer 更容易消费一个专用节点，就把实现便利包装成语言语义。

### 我的输出

对真实 upstream 和当前 74 个 state-stream、4 个 count-partition 用例重新拆解后，得到以下事实：

1. Triton FlashAttention/online softmax 使用普通 loop、普通 state carry、显式 `BLOCK_N` region 和 first-class `tl.max/tl.sum/tl.dot`；TileLang 同样使用 `T.serial/T.Pipelined`、显式 buffers 和 `T.reduce/T.gemm`。两者没有 state-stream primitive。
2. split-K/decode 的 `part_id` 在两者中都是普通 program/kernel coordinate；`begin/end` 用 `ceildiv/min` 等普通算术计算，partial tensor 显式带 part 维，再由另一个 kernel/loop 合并。没有 count-partition primitive。
3. 去掉 program id、thread、storage 和 tile extent 后，剩余算法结构是：普通 sequential/parallel loop、SSA/buffer state、logical subregion、first-class reduce/scan/contract，以及显式 partial buffers。
4. 当前 Intent 恰好缺 first-class logical subregion：动态 domain 可以写 begin/end，但类型与 IR 不保存它来自哪根 source axis；Plan/materializer 因而依赖 `PartitionOp` 和 `StateStreamOp` 的 canonical identity 重建 provenance、range 和 offsets。
5. 74 个 state-stream 中，约 58 个是 reduction/contract 形态，7 个是 scan/ordered recurrence，9 个只是无状态 region traversal；没有现有算法证明必须有一个独立 StateStreamOp。4 个 count-partition 全部可以写成 parallel part loop + 普通边界算术 + source-derived logical subregion + partial ABI。

### 当前校正

第六轮“二者都作为 first-class op 保留”的结论也不再成立。更准确的模型是：

- `reduce/scan/contract` 保留 first-class，因为它们携带 axis、combine、prefix/contraction relation 等普通 loop 无法可靠恢复、并直接决定合法 lowering 的语义；
- online softmax/attention 的可重结合 summary 用 generic reduce 表达，prefix/state recurrence 用 scan 表达，严格不可重排的递推用普通 `for`/`while` 和 carry 表达；
- Physical Program 负责把 first-class reduce/scan/contract lower 成 streamed/blocked provider program，`K_TILE` 等物理范围在这里产生，而不是由作者预写 state-stream skeleton；
- split-K 的 `P`、part identity 和 partial ABI仍由作者/Python wrapper表达，但通过普通 parallel part loop、逻辑 subregion 和显式 partial tensor表达，不要求 canonical `PartitionOp(count)`；
- `state_stream` 与 `partition(count=P)` 可以至多是 frontend mechanical sugar，不应作为 canonical Kernel IR 的独立算法 authority。按当前“只保留唯一真实路径”的纪律，若没有额外可编程性证据，最终 public surface 也应删除；
- 删除之前必须先补齐通用 logical subregion（source axis、begin/end、tail/empty/provenance）、普通 loop carry、parallel part identity，以及 reduce/scan 的完整 lowering。不能先删节点再让 emitter 从 shape 猜回来。

因此，Triton/TileLang 的比较给出的不是“Intent 也要暴露 BLOCK_N/program_id”，而是：去掉这些硬件量后，算法仍应由普通控制、逻辑区域和真正的 structured tensor ops 组成；不能用专用 op 替作者预搭一种 provider-friendly physical skeleton。

## 第八轮：抽象物理实现不是简单删除，判断 op 要看抽象后剩余语义

### 你的输入

你认可 `reduce.max` 之上可以有更高层 semantic op，但指出“保留算法、删除 program_id/BLOCK/storage/layout”说得过于简单：Intent 的职责是抽象并重新求出物理实现，不是把物理事实抹掉。你进一步追问 generic reduce combine 是否仍是 combine closure，以及 `partition(count)` 是否应删除或分解成 `parallel`。

### 我的输出

物理量应当逐项完成责任迁移，而不是文字删除：

- `program_id` → logical parallel coordinate/ownership domain → target execution mapping；
- `BLOCK_N` → Physical Program 中尚未绑定的 blocking extent → provider/tuner 的合法数值；
- pointer arithmetic → logical index/subregion relation → target address/mask；
- fragment/shared buffer → logical value/buffer与 lifetime/reuse → storage placement；
- tiled state loop → reduce/scan/ordinary recurrence 的 physical streaming realization。

因此判断一个高层 op 是否合理，不能看它是否“比 Triton 更高”，而要看抽走 target realization 后是否仍有独立的可观察语义或合法变换合同：

- `reduce.max` 剩下 axis、identity、max combine 和 reduction-tree freedom，因此合理；
- generic reduce 剩下 typed accumulator、identity 和作者写下的 combine body，因此合理。它仍然使用 combine closure，和 Triton `tl.reduce(..., combine_fn)` 同类；closure 是计算本身，不是额外优化 annotation；
- `softmax/logsumexp` 若定义稳定数值、axis、mask 和输出合同，也可以作为更高层 semantic op，再由 compiler 选择普通、online 或 fused realization；
- `state_stream` 抽掉 segment extent、storage 和 provider loop 后，剩余通常只是普通 ordered fold/carry。若允许任意分段，真正提供合法性的应是 reduce/scan algebra；若不允许重排，则普通 `for/while` 已经完整表达。因此它更像 physical streaming skeleton，应主要出现在 Physical Program，而不是 canonical Kernel IR；
- `partition(count=P)` 抽掉 program mapping 后，剩下 parallel part identity、P、part→subregion 公式和 partial ABI。`parallel` 只表达独立性，不能单独替代 subregion relation；完整分解应是 parallel part loop + typed logical subregion + explicit partial tensor。

### 当前校正

最终编程模型不应表述成“删除物理实现”，而应表述成：

```text
DSL/KIR 保存算法语义和逻辑关系
        ↓
Physical Program 为每项未绑定的物理责任选择答案
        ↓
provider lowering 将答案投影成 Triton/cuTile/TileLang/CPU/RVV 结构
```

`state_stream` 的在线块循环并非消失，而是作为 reduce/scan/ordinary recurrence 的一种 physical realization 在 compiler 中重新产生。`partition(count=P)` 的 P-part 算法也并非消失，而是由普通 parallel coordinate、logical subregion 和 partial ABI显式组成。专用 public/canonical op 是否保留，只能由可编程性决定，不能再由当前 Plan/materializer 是否依赖它决定；语义上两者都已经可以分解。

## 第九轮：跨后端不是给 GPU 概念换名，算子 DSL 也不是整算子目录

### 你的输入

你指出上一轮仍有两个根本问题：一是把 `BLOCK_N/program_id/pointer/fragment` 换成 blocking extent、logical coordinate、subregion、lifetime，看起来可能只是给 GPU 概念换中性名字，未证明 CPU/RVV 可以自然消费；二是提出 `I.softmax/I.logsumexp` 已越过算子内部 DSL 的边界，变成整算子库。你同时追问这是否意味着准备大改现有抽象，并提醒 Intent 最终必须跨 GPU、CPU 与 RISC-V/RVV。

### 我的输出

`I.softmax/I.logsumexp` 的提议明确撤回。Intent 的作者必须能够用 pointwise、reduce、scan、contract、控制、state 和 effects 写下 softmax/logsumexp 的算法；整算子名称不能替作者决定稳定公式、pass 数、状态更新或融合边界。generic reduce 的 typed combine仍属于 reduction primitive，和整算子 builtin 不是一层。

跨后端分层也必须修正为：

```text
target-independent Kernel IR
    ├─ logical domains/regions/index relations
    ├─ independence/order/state/effects
    └─ reduce/scan/contract algebra
          ↓ shared analyses/proofs
    ├─ GPU physical program
    ├─ CPU physical program
    └─ RVV physical program
          ↓ each target/provider lowering
```

共享的是算法事实和可复用分析，不是要求三类目标共享一份 GPU-style physical topology：

- logical parallel independence 跨目标成立；GPU 可映射为 grid/program，CPU 可映射为串行/线程任务，RVV 可映射为 strip-mined vector iteration；
- physical blocking 只是共同责任类别，不是一种共同表示。GPU fixed block、CPU cache/SIMD tiling、RVV runtime VL 不能被压成同一个 `BLOCK_N` 政名字段；
- logical subregion/source provenance 跨目标成立；pointer tensor、CPU scalar/SIMD address、RVV vector/mask address分别由目标 physicalization产生；
- logical buffer、mutation、sharing与 lifetime obligation 跨目标成立；fragment/shared、CPU stack/cache、RVV scalar/vector register是不同目标的存储答案。

当前仓库没有 CPU/RVV backend，因此这些是对跨目标合同的设计检验，不是已经完成的实现事实。

### 当前校正

本轮不据此修改 DSL/KIR/compiler。`state_stream` 和 `partition(count=P)` 恢复为待裁决状态：

- `partition(count=P)` 的 P、part identity、part-sensitive branch 和 partial ABI 已由4个真实用例证明是 target-independent 可观察语义；是否保留专用 public/canonical op，仍需在 typed logical subregion 完整定义后比较可编程性，不能先删；
- `state_stream` 删除 physical extent 后仍可能剩下 target-independent 的 ordered region transition、logical stop 与 state schema；它是否只是 reduce/scan/ordinary loop 的组合，必须逐类检查 strict recurrence、effectful region loop、fixed logical chunk 和 floating-point order，当前证据不足以直接删除；
- 任何最终共享抽象都必须能分别说明 GPU、CPU、RVV 如何投影；只能自然解释 GPU 的概念不得进入跨后端 programming model。

因此，第七、八轮关于 canonical `state_stream/partition(count)` 应删除的判断降级为待验证假设，不再作为当前统一结论。

## 第十轮：按“抽走物理实现后还剩什么”裁决全部语言构造

### 你的输入

> “我认为需要大改抽象……state stream、partition 相关需要改。先把文档确定好，后面再确定编译器文档，一层一层地改。彻底想清楚类似 op 到底有多少、哪些删除不删除。判断是：抽走硬件后，如果剩余结构没有独立意义或强依赖硬件表达，就不要加 op；只有剩余算法无法被合理表达时才增加相关 op。”

你进一步纠正我：这不是把硬件概念换成中性名字放入 DSL，也不是害怕当前 compiler/backend 不支持而保守维持现状。programming model 可以大改，当前实现随后按层迁移。

### 我的输出

我先枚举了当前 52 个 public callable spellings（含 namespace members）、frontend/KIR 内部控制节点、结构化运算、数据/内存/effect 构造，并用同一条判据审计，而不是只讨论 `state_stream` 和 `partition` 两个名字：

1. 从真实算法中抽走 program/lane id、physical tile、storage/layout、copy/pipeline 和 provider primitive；
2. 检查剩余内容是否仍改变 logical values、control/state、effects、ABI、numeric contract 或合法变换空间；
3. 尝试用现有跨后端基础构造自然表达；“可以用标量循环模拟”不算自然表达，若会丢失 axis relation、associativity、prefix、conflict 或 encoding 合同，仍需 first-class construct；
4. 若剩余只是 compiler-chosen decomposition 或一种 target-friendly skeleton，则不得成为作者/canonical op；
5. backend 尚未实现不是删除语义的理由，backend 可以在能力边界处明确拒绝。

这条判据得到两个明确结论。

`state_stream` 应删除。它允许 compiler 选择 segment extent：若任意合法 segmentation 不改变结果，合法性必然来自作者写下的 associative summary/prefix contract，应使用 generic `reduce` 或 `scan`；若 segmentation 会改变结果，就必须使用普通有序 `for/while`，或由作者显式写出固定 page/window/chunk subregion。不存在第四种既依赖 compiler 任意分段、又没有代数合同、结果仍定义完整的语义。

`partition` 也应删除。`count=P`、固定 extent 和 part-sensitive branch 的剩余算法是 part identity、边界公式、source-derived subregion 与显式 partial ABI；这些由普通 `parallel` loop、整数算术、source-axis slicing 和 external partial tensor 完整表达。专用 op 反而强迫语言替不同算法选同一套边界公式。不可观察的 blocking 则完全属于 target physical program。

同时完成了其它同类构造的归属：

- canonical：domain/source-derived subregion、普通控制与 carry、parallel independence、tensor/value operations、generic reduce、scan、contract、packed/scale/sparse 的显式 arithmetic 与 indexed relations、logical buffers 与 effects；
- surface shorthand/accessor：`reduce.sum/max`、`any/all`、`arg_reduce.max`、`zeros`、value mask、`ragged`、members/indices/endpoints、普通 advanced indexing；
- 不属于语言：`auto`、`ordered`、`state_stream`、`partition`、program/lane id、physical tile/storage/layout/pipeline，以及 softmax/attention 等 whole-operator builtin；
- 收敛掉的 whole-op：`scaled_contract` 和 `sparse_contract_2to4` 不作为永久独立 op；作者显式写 scale/sparse decoding arithmetic 与 index relations，计算使用普通 `contract`。

### 当前校正

第九轮把 `state_stream` 与 `partition(count=P)` 恢复为未决，是为了避免在缺少完整 inventory 时过早删除；本轮完整审计后，这两项已正式裁决为删除。决定依据不是“Triton 没有同名 API”，而是抽走 physical segmentation 后没有不可组合的独立算法语义。同样，`ragged(...)` 收敛为 offsets-derived subregion 与 indexed relation 的 surface helper；需要保留的是 relation provenance，不是现有专用 op。

规格同步采用 source-derived subregion：`axis[begin:end]` 保留 source axis、半开边界、tail/empty 和 provenance。它不是把 physical tile 改名为 region；只有作者边界、输入 relation 或算法 metadata 能在 Kernel IR 中产生它，compiler blocking 只存在于 target-specific physical program。

本轮只修改 programming-model/DSL 规格与理想示例；当前 frontend、KIR、compiler、examples corpus 尚未迁移，后续必须在 compiler 规格审定后按层收敛成唯一路径。

## 第十一轮：重新审查 subregion 与 `typed data semantics`

### 你的输入

> “按照我们的讨论，你再看看 doc 还有什么要修改的东西，先不做而是先讨论。然后 `[begin:end]` 是合理的吗？`typed data semantics` 是什么，也是 DSL 结构吗？”

### 我的输出

我重新检查了持续报告与全部 DSL/programming-model 文档，确认 source-derived subregion 的语义需求成立：作者算法、offsets 或 metadata 产生的逻辑子区间必须保留 parent source 与 provenance，不能退化为一个独立的同 extent domain，更不能和 compiler tile 混同。

但我同时指出 `axis[begin:end]` 的具体 surface 还需要把 logical coordinate、ordinal position、step、empty 和越界语义讲清楚；并承认 `typed data semantics` 当时只是“存储 bits 如何解释成 logical values”的模糊责任类别，尚未证明需要一个新 DSL type/descriptor。

### 当前校正

subregion 的 canonical 语义是 source domain 中连续的一段成员，保留 parent/bounds/provenance；非连续成员属于 indexed relation。它不是实现分阶段能力，不使用“第一版只支持”的措辞。

## 第十二轮：撤销“第一版”和未经证明的 typed-format 复杂化

### 你的输入

> “为什么又第一版？我们定义的是最终 spec。现在错误做法是什么、Triton 是什么、我们要把它提升成什么？我怀疑 typed data semantics 是你自己增加复杂度。”

### 我的输出

我撤回“第一版”措辞，并把 subregion 定义为最终语义上的连续 source interval：`axis[begin:end]` 表达 parent domain 中的 `[begin,end)`，empty 合法，越界不隐式 clamp；非连续访问用 indexed relation。

对于 packed data，我确认 Triton 作者本来就用普通 load、整数索引、shift/mask、sign extension、scale arithmetic 和 `tl.dot` 写出算法。Intent 作者同样可以用 target-independent 的普通算术与 `contract` 表达；无需为了这件事预先发明 `EncodedType`、format descriptor 或 generic decode 系统。

但该轮输出又越过证据，声称 compiler 可以把显式 decode + contract 可靠识别成 `tl.dot_scaled` 一类 native primitive，并把它称为成熟编译器的普通局部 rewrite。这个说法把未来可能采用的实现技术写成了已经确定的能力，仍然不严谨。

### 当前校正

programming-model 规格只要求作者能够完整写下 packed/scale/sparse 的数学与索引关系；它不承诺某个特定 native fusion，也不靠这种 fusion 证明语言设计。后续 compiler 规格必须独立回答怎样从这些事实产生高质量 target program，不能反过来扩 DSL。

## 第十三轮：回到最终抽象，不用当前实现替设计作答

### 你的输入

> “现在在重构，所以当前肯定不行；我们要考虑新抽象。为什么不能像 Triton 一样直接表达 packed INT4？”

### 我的输出

当前统一结论是：Intent 在 packed/quantized/sparse 数据上不增加一套预设 typed-format 抽象。作者直接写 carrier tensor、bit/index arithmetic、signedness 恢复、scale/zero-point relation、metadata interpretation 与普通 `contract`；这些都是跨 GPU、CPU、RVV 成立的算法语义。

Triton 与 Intent 的差别不在是否允许这段代码，而在作者是否还必须同时写 target physical structure。Intent 抽走 program id、physical tile、storage、target primitive 与 pipeline，但不抽走 unpack/dequantize 的数学表达。

`scaled_contract` 与 `sparse_contract_2to4` 不因目标存在相应 primitive 就成为 canonical whole-op；作者程序由普通 arithmetic/index relations + `contract` 构成。以后 compiler 是否、如何形成 `tl.dot_scaled`、sparse MMA 或 CPU/RVV fused form，属于 compiler/target 规格和 pass 设计，本阶段不预判实现办法。

本轮只更新持续讨论报告，没有修改 `doc/`。其中关于 typed encoded/sparse values/descriptor 的现有措辞已标记为下一次规格同步需要清除的旧结论。

## 第十四轮：用固定判据复核 scaled/sparse contract 与第十轮

### 你的输入

> “`tl.dot_scaled` 或 sparse MMA 在 Triton 里是显式写的吗？`scaled_contract` 是一个算法吗？第十轮到底有没有问题，region 到底改不改？不要因为我随便反驳就立即换结论，要给出经得起反驳的方法和这一轮结果。”

### 我的输出

本轮不再用“能否理论展开”为唯一判据，而采用四项固定检查：

1. 写出该构造不含 target spelling 的 denotational semantics；
2. 判断它是完整算法、局部 structured operation、普通 operations 组合，还是 physical schedule；
3. 删除它是否迫使 compiler 从任意代码重新识别 axis、format、reassociation 或 metadata relation；
4. GPU、CPU、RVV 是否能对同一语义给出实现，即使某个 target 没有 native primitive。

真实 Triton 证据表明，`tl.dot_scaled` 是作者显式调用的 public operation，并直接建立独立 TTIR `DotScaledOp`；Triton compiler 不会从普通 scale arithmetic + `tl.dot` 自动形成它。它定义的是 OCP microscaling matrix product：FP4/FP8 等 element format、per-K-group scale、packing relation、accumulator 与数值规则。target compiler 再选择 native scaled MMA 或显式 unpack/scale +普通 dot。

这与普通 packed INT4 必须区分。普通 INT4 source 可以、也应该直接写 carrier indexing、`k//2`、shift/mask、sign extension、scale/zero-point arithmetic 和普通 `contract`；不需要 `EncodedType` 或 format descriptor。`tl.dot_scaled` 不能作为 arbitrary packed decode 会被 compiler 自动融合的证据。

2:4 sparse contraction 也有独立语义：compressed nonzero values、metadata 对 logical K positions 的映射、logical dense-K contraction。TileLang 作者显式调用 `T.gemm_sp`；当前 Triton public DSL 没有对应 sparse-dot op。这是 provider capability 差异，不消除该 operation 在 CPU/RVV 上通过解压/索引与普通 contraction 实现的跨目标含义。

### 当前校正

`scaled_contract` 与 `sparse_contract_2to4` 不应和 softmax/attention 一样被归为 whole-operator builtin。它们是与 `contract` 同层的局部 structured tensor operations：作者选择一项有明确 formats/scales 或 compressed/metadata relation 的 contraction，compiler/target 选择 native primitive 或等价 realization。

当前实现的问题不是存在这两个 op，而是把 canonical 语义与 target 限制混在一起。例如 scaled frontend 允许一般 group size，三家 target 实际只接受 32；sparse Plan 把 shared placement、metadata physical tile 与 `T.gemm_sp` 形态写死。这些应在后续 KIR/compiler 规格中拆开：Kernel IR 只保存 logical formats、scale/metadata relation、axes、accumulator 与 numeric contract，target capability/physical layout 留给 target program。

第十轮其它结论经反例复核后维持：

- `state_stream` 删除：若 compiler-chosen segmentation 可任意变化却无 reduce/scan algebra，effect 次数、边界观察或浮点结果都会变化，语义不完整；固定 region effects/边界必须由作者显式 subregion loop 表达。
- `partition` 删除：part identity、边界公式、覆盖/空 part 与 partial ABI 由普通 loop、source-derived subregion、preconditions 和 external tensor 明确写出；不可观察 blocking 属于 target realization。
- source-derived subregion 保留，不撤销第十轮结构。`axis[begin:end]` 表达 parent domain 中连续的 logical interval；非连续成员使用 indexed relation。需要修改的是文档对 coordinate/empty/bounds/provenance 的精确定义，不是删除 region 或恢复 partition。
- `ragged` 可继续作为 surface helper，canonical semantics 是 offsets-derived subregion、可选 indexed mapping、relation provenance 与明确 preconditions。
- generic reduce、scan、contract first-class，mask/zeros 等机械 shorthand 的结论维持。

本轮只更新持续讨论报告，不修改 `doc/`；当前 doc 中删除 scaled/sparse contract、引入 typed encoded/sparse value 的文字已被本轮推翻，待 programming-model 讨论完成后一次同步。

## 第十五轮：下层显式 op 是证据，不是 Intent surface 的复制规则

### 你的输入

> “`scaled_contract` 就是映射成 Triton 的 scaled dot 吗？`sparse_contract_2to4` 也是类似吗？是不是 Triton 有的 op，我们至少要有类似但更高层的 op？TileLang 喜欢把高层东西直接做成 op，TileLang 单独有的要保持怀疑。”

### 我的输出

下层显式 op 的存在只能证明“这段语义在进入下层时必须保持可见”，不能推出 Intent 必须复制同名 op。`program_id`、block pointer、shared allocation、pipeline 也都是下层显式 op，但它们是 target physical surface；`reduce`、`dot`、`dot_scaled` 则携带跨 target 的局部 tensor semantics。

`scaled_contract` 的证据已经足够：Triton 作者显式写 `tl.dot_scaled`，frontend 建立独立 `DotScaledOp`，target compiler 再选择 native scaled MMA 或显式 unpack/scale/dot。Intent 应保留同一语义层、但更 target-neutral 的 first-class scaled contraction：formats、scale relation、contraction axes、accumulator 与 numeric contract属于 Kernel IR；K packing、layout、MMA family 和 native/software realization 不属于作者 surface。

普通 packed INT4 仍然不是 scaled contraction。作者直接写 bit/index/dequantize arithmetic + `contract`；不能因为 Triton 有 `dot_scaled` 就把所有 packed formats 塞进 scaled op。

`sparse_contract_2to4` 的情况尚不能用 TileLang `T.gemm_sp` 直接裁决。当前能确定必须保存的是：compressed values、metadata schema、每组四个 logical K 中两个 nonzero positions 的 relation，以及 logical contraction。存在三种候选表示：

1. 继续使用 combined `sparse_contract_2to4`；
2. 先构造 concrete 2:4 sparse operand/relation，再交给普通 `contract`；
3. 作者显式解 metadata、gather RHS 后普通 `contract`。

第三种无需新构造，但要得到 native sparse MMA 就可能要求 compiler 重新识别完整 2:4 relation；第一种直接映射 TileLang，却把 data representation 与唯一 consumer 焊在一起；第二种最可组合，但会新增一种具体 sparse value/relation。当前 Triton 没有 public sparse-dot op，TileLang 单独有 target-specific `T.gemm_sp`，现有 corpus 也只有 contraction consumer，因此证据不足以在第一、第二种之间冻结。

### 当前校正

- `scaled_contract` 作为 target-independent local structured operation 保留；它在 Triton target 上可投影为 `tl.dot_scaled`，但语义不由 Triton API 定义。
- 2:4 compressed/metadata relation 必须 first-class 保存；`sparse_contract_2to4` 这个 combined surface/canonical op 是否保留，降为明确未决。
- “Triton 有什么 Intent 就至少有什么”不是规则；规则仍是抽走 target physical 后是否剩下不可丢失、不能自然重建的跨 target 算法合同。
- 本轮只更新持续报告，不修改 `doc/`。

## 第十六轮：不是比较“新增几个”，而是把语义放在正确粒度

### 你的输入

> “A 不也是新增一个东西吗？继续讨论，并继续找还有没有其它 op 有类似情况。”

你同时要求不再重复已经确认的基础边界；后续讨论只处理新的语义分叉和证据。

### 我的输出

A 与 B 都会保留一个 first-class semantic construct，上一轮把 B 的缺点写成“它会新增构造”是不成立的。真正区别是构造粒度：

- combined `sparse_contract_2to4` 把 2:4 relation 定义为该 structured operation 的 operand contract；
- sparse operand/relation +普通 `contract` 把 2:4 data relation 提升为可以被多个 consumers 复用的独立 value kind。

当前真实语料只有 sparse contraction consumer；TileLang 的高性能 surface也是 combined `T.gemm_sp`，Triton没有独立 sparse value abstraction。没有第二个真实 consumer 迫使 2:4 relation 独立存在，因此为它预建通用 sparse value 是无证据的拆分。当前选择 combined `sparse_contract_2to4`：保留 compressed values、metadata→logical positions、logical K contraction 与 accumulator contract；移除 shared placement、metadata tile、fragment/layout 和 provider primitive 等 physical 限制。

同一判据审计其余 public/KIR constructs 后得到：

- `ragged` 不应降成纯 frontend helper。现有语料有 22 个 `I.ragged` 使用、跨 10 个文件，relation 被 members、gather、contract、scatter 等多个 consumers 复用；outer/member domains、offsets、optional mapping、preconditions 与 provenance 本身是稳定跨 target relation。保留 first-class ragged relation，`members`/outer/member accessors 可以是 surface shorthand。
- `arg_reduce.max` 继续是 generic record/multi-result reduce 的 shorthand；tie-break 必须进入 combine contract，不需要独立 op。
- `gather(valid,fill)` 的 validity 必须在可能越界的访问发生前生效，不能拆成先非法读取再 value mask；access validity 与 pure value mask 是两种语义。
- `scatter_unique` 与 `scatter_reduce` 按 collision contract 分开合理；scatter reducer 应采用 typed combine，而不是不透明字符串，但无需把 indexed write 与 collision reduction拆成两个操作。
- atomic RMW 与 memory order/scope 是同一 effect operation 的合同，不属于错误焊接；当前 atomic family 缺 max/min 等属于语言能力覆盖，不是抽象层级错误。
- `buffer(init=...)`、`random(seed,index)`、`assume_in_bounds`、shape/value operations 没有发现同类粒度错误；`zeros`/value mask/`I.add` 仍应机械归一为 full/select/普通加法。

审计还发现一个新的真实缺口：Triton 有 first-class `tt.histogram`，而当前 Intent histogram 只能写 external buffer initialization + indexed `atomic_add`。二者的 value/effect/初始化结构不同，compiler 不能在不分析 alias、初始化和冲突 effects 的情况下把后者恢复成前者。histogram 有明确跨 target denotation——对输入 logical values 按 bin 计数并返回 count tensor——因此应成为 first-class local structured operation；bins、mask/out-of-range、count dtype、overflow 与 empty semantics 属于它的数值合同。

### 当前校正

- `scaled_contract` 与 combined `sparse_contract_2to4` 均作为 concrete local structured operations 保留；不为二者预建 generic encoded/sparse value type。
- `ragged` 恢复为 first-class relation，不再只作为 frontend helper；accessor 可归一化。
- histogram 是本轮发现的唯一需要新增的 first-class structured operation；其余审计到的缺口属于 shorthand normalization、typed combiner 或 atomic family coverage。
- 本轮只更新持续报告，不修改 `doc/`。

## 第十七轮：回到 Triton 作者表达，撤销以当前实现为依据的保留结论

### 你的输入

> “怎么 ragged 又不删了，Triton 有这样的 op 吗？不是 Triton 的 op 映射过来的，必须说明抽走物理抽象以后不用它就表达不出来，才有保留理由。前几轮思考的依据到底是什么？”

你指出第十六轮实际上换掉了既定判据：使用次数、consumer 数量、当前 Plan/leaf 是否依赖某个节点，只能说明迁移影响，不能证明它属于作者编程模型。

### 我的输出

我按 Triton 作者级代码重新核查控制、域和 relation 构造。Triton grouped GEMM、varlen 与 grouped 结构使用普通 metadata tensor、offset load、普通循环、整数边界算术和 indexed access；没有 `ragged`、`members`、`partition` 或 `state_stream` op。抽走 program id、physical tile、pointer spelling 与 storage 后，ragged 剩下的是：outer domain、offsets、source-derived `[begin:end)` subregion，以及可选 indexed mapping。

这套基础关系已经完整保存算法。把它包装成 `RaggedOp` 不增加任何不可约语义；多个 consumers 需要共享的是同一 source/provenance/index relation，而不是一个名为 ragged 的第二份 authority。当前 frontend 直接生成 `OperationKind.RAGGED`，KIR 与 Plan 也有独立 ragged 节点，只能说明旧实现尚未迁移，不能反过来定义最终规格。

同一严格审计还确认：

- `state_stream` 可由 first-class reduce/scan 或普通 ordered loop/carry 表达；
- `partition(count/extent)` 的作者可见部分可由 part domain、边界算术、source subregion 与 partial ABI 表达；
- `indices`、endpoint、`assume_in_bounds` 可以是 accessor/precondition，不需要独立 executable op；
- value mask 归一为 select，gather/scatter 名称必须继续按 access validity 与 collision effect 拆解；
- reduce、scan、contract、scaled contract 有真实 structured-operation 证据，不能按普通循环语法糖处理。

### 当前校正

- 删除 first-class `ragged` 结论恢复有效：最终 canonical semantics 是 offsets-derived subregion、indexed relation 与 provenance；`ragged/members` 不保留独立 public/canonical op。
- 第十六轮用“22 个使用、多个 consumers”恢复 `RaggedOp` 的依据正式作废。
- 当前代码中的 RaggedOp、ragged route 与 leaf 消费属于后续迁移对象，不是规格证据。
- 这一轮一度把“能由基础构造算出同一结果”当成删除所有专用 op 的充分条件；下一轮由 TileLang 高层 structured ops 的反例继续收紧这一判据。
- 本轮只更新持续报告，不修改 `doc/`。

## 第十八轮：TileLang 高层 op 迫使区分 desugaring 与 lowering

### 你的输入

> “TileLang 的一些 op 层次比较高，我们怎么办？会不会形成一种映射关系？如果不能从基础代码合法恢复，我们也应该有一个高层次 op 才合理。”

你追问的核心是：跨后端 canonical 层不能只取 Triton、cuTile、TileLang public API 的同名交集；否则某个 target 独有但确有稳定算法合同的高层 primitive，会迫使 leaf 从任意低层代码重新猜回来。

### 我的输出

真实 TileLang 实现表明，它的高层 surface 混合了三类内容：

1. **稳定 structured semantics**：dense GEMM、sparse GEMM、reduce、scan、scaled GEMM 的数学/数值与 operand relation；
2. **semantic op 加 provider form**：`T.gemm` 同时接受 transpose/accumulate 语义和 warp policy、`k_pack`、mbarrier 等物理参数；
3. **纯 target realization**：`T.wgmma_gemm`、`T.tcgen05_gemm`、`T.Pipelined`、TMA/async copy、shared/fragment/TMEM allocation、layout 与 warp-specialization annotation。

因此 canonical-to-target 映射不要求同名或一对一：一个 Intent structured op 可以在 TileLang 上直接调用高层 op，在 Triton 上展开成若干 DSL statements，或者在目标确实不能表达时明确 unsupported。反过来，TileLang 高层 op 中的 warp/storage/pipeline/instruction 参数不能整体抬进 Intent。

这也修正了上一轮过强的“能用基础构造表达就删除”规则。所有 structured operation 最终都能用循环表达；真正区别是展开属于哪一类：

- **desugaring** 只把同一语义机械展开，不选择求值顺序、中间表示、精度、storage 或 target primitive；这类不需要独立 canonical op。`ragged` 展开成 offsets + source subregion + index relation 属于此类。
- **lowering** 已经选择具体实现，例如显式解码、物化 dense intermediate、确定 reduction order、用 atomic 实现 histogram，或放弃某个 structured primitive；这类不能替作者预先展开，canonical 层应保留高层合同。

按这一判据，`sparse_contract` 应保留。TileLang `T.gemm_sp` 明确接收 compressed values、metadata、dense operand 与 accumulator；这些是 structured sparse contraction 的作者级合同。TileLang provider 可直接投影为 `T.gemm_sp`；Triton 当前无 `tl.dot_sp`，可从明确的 sparse-contract op 合法展开 metadata decode + ordinary dot，或在不能保证语义/质量时明确拒绝。不能删除该 op 后再让 TileLang leaf 从任意 bitwise/gather/contract producer chain 猜回 sparse GEMM。

当前代码已经同时展示了正确与错误方向：`intent.contract → T.gemm`、`intent.sparse_contract → T.gemm_sp` 是明确 structured mapping；TileLang provider 中从 `cast → bitwise_and → right_shift → view_load → contract` 整链匹配 packed INT2 form，则说明 representation fact 或 provider-local idiom boundary 仍需在后续 compiler 规格中单独审计。

### 当前校正

- Intent canonical semantics 不是三家 API 的最小公分母；合法映射可以是一对一、一对多或明确 unsupported。
- 保留高层 op 的判据是：抽掉 target physical 字段后是否剩下稳定的 structured algorithm/numeric/effect contract，以及删除它是否迫使 compiler 选择一种 lowering 或从任意代码重新证明该结构。
- `ragged` 仍删除独立 op，因为 offsets/subregion/index relation 是它的语义本身；`sparse_contract` 恢复为 first-class structured operation，因为显式 decode + dense contract 已是一种 lowering。
- `scaled_contract`、`contract`、reduce、scan 与 histogram 按同一规则保留；具体 target capability 与 physical form 留在 provider program。
- TileLang 的 WGMMA/TCGEN05/TMA/pipeline/storage/layout surface 只作为 provider lowering 目标，不进入跨 target public DSL。
- 本轮只更新持续报告，不修改 `doc/`。

## 第十九轮：编程模型骨架基本确定，但尚有少数冻结项

### 你的输入

> “是不是现在基本上确定了？还有什么问题？”

### 我的输出

当前 programming model 的主骨架已经稳定：Intent 是跨后端 kernel DSL，不是 whole-operator library；作者写普通控制流、逻辑域与 source subregion、数值路径、state/effects 和 first-class structured operations；compiler 产生 GPU/CPU/RVV 各自的 physical program；provider 可以把一个 canonical op 一对一投影、一对多展开或明确拒绝，但不能从任意周围代码重新猜回作者没有明确保存的 structured operation。

已经可以定稿的包括：删除 `state_stream`、全部 `partition` 和独立 `ragged/members`；保留 source-derived subregion 与 indexed relation；保留 reduce、scan、contract、scaled contract、structured sparse contract 和 histogram；packed/普通量化解码继续由作者写显式 bit/index/scale arithmetic；WGMMA、TCGEN05、TMA、storage/layout/pipeline 等只属于 provider realization。

尚未闭合的内容分两类。

第一类仍会改变 programming model，必须在冻结前决定：

1. `parallel` 是必要的作者级 independence contract，还是又一种让作者授权 compiler 重排的标志；普通 loop 与 dependence/effect analysis 能否自然承接其语义仍需按 Triton/TileLang 作者模型作最后反例审计。
2. `contract` 的承诺范围是一般 axis-pair tensor contraction，还是只保留能稳定映射到 `tl.dot/T.gemm` 的 dot-shaped structured operation；更一般形态是否应写成 pointwise + reduce 不能由当前宽 API 反向决定。
3. `sparse_contract` 的 public surface 与 format schema：保留 structured op 已确定，但应是固定 `sparse_contract_2to4`，还是显式 format operand/attribute 的可扩展 family，仍需由真实格式而不是预建 generic descriptor 决定。
4. plain region copy 是否需要作者可写/canonical op，还是精确的 read→write relation 足以让 provider 合法形成 `T.copy`；这决定 copy 是算法 effect 还是 provider form。
5. logical mutable buffer 与 deterministic RNG 的最终合同：buffer 的初始化、alias、lifetime 与可观察 mutation，以及 RNG algorithm/counter relation 是否要求跨 target bitwise 一致。

第二类不再改变总体抽象，但必须把数值和错误边界写完整：generic reduce/scan 的 identity、empty、NaN、tie-break 与 finite-precision reassociation；scaled contract 的 format、scale grouping、packing、rounding 与 accumulator；sparse metadata interpretation；histogram 的 bins、mask/out-of-range、count dtype、overflow 与 empty；external view 的 stride/alignment/alias ABI；indexed access 的 validity、fill 与 precondition surface。

### 当前校正

- programming model 已从“方向未定”进入“少数合同待闭合”，但尚不能称为正式冻结。
- `state_stream/partition/ragged` 不再是未决项；后续若重新保留，必须提出新的不可约语义反例，不能再引用当前代码或语料数量。
- 下一步优先审定 `parallel`、`contract` 范围、sparse surface 与 copy；其余主要是数值/ABI 合同补全。
- Kernel IR、Physical Program、compiler decision space 与 passes 仍属于下一阶段；programming model 基本确定不等于整个编译器规格已经确定。
- 本轮只更新持续报告，不修改 `doc/`。

## 第二十轮：删除“许可”框架，`parallel` 不再作为作者语言构造

### 你的输入

> “parallel 是什么意思，为什么还有合同？一个编译器根本不应该有合同或许可的思想。Triton 是怎么做的？所谓数值/ABI 合同又是什么，不要逃避。”

你进一步追问：一个正常编程语言为什么需要作者额外许可 compiler 优化；如果 `parallel` 只是告诉 compiler“这里可以并行”，它本身就是错误抽象。

### 我的输出

“许可/合同”这一套措辞正式撤回。我此前把三件不同的事混在一起：语言 operation 的可观察语义、作者输入的前置条件、以及作者授予 compiler 重排的许可。前两者是任何语言都必须定义的普通语义；第三者不应成为 Intent programming model 的组织原则。

真实 Triton 没有 target-independent `parallel`/`independent` loop。作者在 host 侧用 launch grid 指定 program instances，在 kernel 中用 `tl.program_id` 取得 GPU program coordinate；`tl.arange` 只是 block index tensor。普通 Python `range` 与 `tl.range` 都 lower 成顺序 `scf.for`，后续 passes 只做 pipeline、unroll、layout propagation 与 warp specialization，不把任意普通 loop 自动拆成新的 launch-grid programs。非原子重叠 global stores 也没有一套作者 independence declaration；需要并发 read-modify-write 时作者显式写 atomic。

当前 `I.parallel` 实际正是一个“independence 声明 + ownership handoff”：frontend 禁止 loop-carried SSA，文档要求不同 iterations 无冲突且顺序不可观察，GPU Decisions 再把对应 domain 标成 parallel role 并建立 ownership range，三家 leaf 最终映射为 program index、persistent worker loop 或 `T.Parallel`。它虽然比 GPU `program_id` 中性，但仍要求作者提前指出 compiler 应建立哪一层 physical ownership；这与本次重构要让 compiler 从完整逻辑算法求 physical program 的目标冲突。

因此 public/canonical `parallel` 应删除。作者写普通 logical domain iteration；compiler 用 def-use、index relation、unique/atomic effects、alias 与 dependence analysis判断哪些迭代可以成为独立 ownership。证明成立就形成 GPU grid、CPU task/loop 或 RVV strip-mining；证明不了就保持作者顺序或明确暴露尚未支持的 analysis boundary，不能要求作者再写 `parallel` 给 compiler“许可”。reduce、scan、contract 的合法重排来自这些 operation 自己定义的结果语义，不是另一类许可标志。

“数值/ABI 合同”也改回准确名称：

- **operation semantics**：例如 reduce/scan 的 axis、combine/result schema、dtype promotion、NaN/tie/empty 行为、是否定义固定浮点顺序；cast 的 rounding；scaled/sparse operation 的 format 与 accumulator。Triton 自己也明确 sum/cumsum 的 promotion、max/min 的 NaN policy 与 argmax tie-break，同时 generic reduce 不定义 identity/empty，也不保证固定浮点树。Intent 必须明确自己采用、拒绝或留作未定义的行为，否则同一 source 在三后端可能静默得到不同结果。
- **kernel interface/view semantics**：参数的 dtype/rank、view shape/stride、合法 bounds 与 alias 是否允许。这不是新 DSL “合同”构造，只是函数参数和内存操作原本就必须有的含义。alignment specialization、divisibility/contiguity hints、num warps/stages、descriptor ABI packing 等继续属于 compiler/provider/runtime，不进入作者算法表面。

### 当前校正

- public/canonical `parallel` 删除，不再把 independence 当成作者给 compiler 的许可；当前 `ParallelOp`、parallel role 和 ownership bridge 属于后续 compiler 迁移对象。
- 作者只表达普通 logical iteration 与真实 operations/effects；iteration independence 和 target ownership 由 compiler analyses/passes 求出。
- reduce/scan/contract 等 first-class op 保留，因为它们定义计算结果，不是优化 annotation。
- 不再使用“数值/ABI 合同”笼统表达；后续分别写 operation semantics 与 kernel interface/view semantics，而且只定义可观察行为。
- 本轮只更新持续报告，不修改 `doc/`。

## 第二十一轮：`parallel` 是循环语义，不是轴属性；partition 是 region 构造

### 你的输入

> “不一定吧，我们确实可以显式定义，只要是语义就行，TileLang 不是也有 Parallel？但对于 CPU 呢？如果这根轴声明可以并行，下一根轴也声明并行，不就很奇怪？我记得 partition 当时论证过，它存在是因为这是特性，而不是声明。”

你指出第二十轮又从一个极端走到另一个极端：拒绝“作者许可 compiler”不等于所有显式并行结构都没有语言意义；关键在于它究竟是附着在轴上的优化属性，还是一种有自身执行语义的控制构造。

### 我的输出

真实 TileLang 将 `T.Parallel` 建成 `ForKind::kParallel`，内部记录 `kDataPar` iteration variables；`coalesced_width`、loop layout、async preference 等是可分离的物理 annotations，layout 可以由后续 inference 产生。Triton 没有同名 target-independent op，但 GEMM 教程明确把外层 `(m,n)` 写成“do in parallel”的算法迭代，每个 logical tile iteration 再由一个 program instance 实现。两者共同留下的 target-independent residue 不是 GPU thread/grid，而是一个无序的数据并行 iteration space。

因此 `parallel` 若定义成“某根轴可以被 compiler 并行化”的属性，确实应删除；若定义成与普通 `for` 不同的循环语义，则可以保留：

- iteration space 中每个 logical point 执行一次；
- points 之间没有可观察顺序；
- 没有跨 iteration 的 loop-carried value；
- 非 atomic/reduction 的冲突 effects 使程序非法，而不是交给 target 猜；
- 不指定 program count、thread、lane、vector width、tile 或真实并发程度。

CPU 可以串行、线程化或 SIMD 实现同一语义，RVV 可以 flatten 后 strip-mine，GPU 可以映射为 grid、worker loop 或 lanes。顺序不可观察，所以 serial execution 也是合法 realization；`parallel(a,b)` 或嵌套 parallel 只定义 Cartesian-product iteration space，不要求生成嵌套 CPU 线程或两层 GPU grid。compiler 仍可从纯整域 expression 推出同样的 data-parallel structure，作者不必为每个 pointwise expression 都写 `parallel`。

当前实现真正的问题不是存在 `ParallelOp`，而是 GPU Decisions 几乎把 parallel domain 直接当作 ownership skeleton。后续 compiler 规格应让 `parallel` 只留在 KIR 中表达 unordered iteration；Physical Program 再结合 access、reuse、effects 和 device facts选择 flatten、blocking、ownership 与 execution mapping，不能机械地“一根 parallel 轴等于一层 program mapping”。

`partition` 与此不同。它不是“这根轴允许分块”，而是构造 author-visible regions：fixed extent 时 body 看见连续 region，count 时还看见 part identity，并可能据此 branch、reduce/contract、写 partial ABI。这个能力在 GPU、CPU、RVV 上都成立。当前 examples 也显示：大量 fixed extent 用法只是旧 GPU blocking skeleton，应迁走；two-pass reduction 与 split-K/count 用法则真正观察 part/region。

因此更准确的 partition 结论是：

- `partition(auto)` 或作者不可观察的 blocking 不进入语言；
- `partition(extent=E)` 只在 E 是 page/window/group/chunk 等作者算法边界时保留；
- `partition(count=P)` 在 P、part identity、边界或 partial ABI 可观察时保留；
- public `partition` 可以是有稳定公式的 region-construction feature；canonical KIR 是保留专用 `PartitionOp`，还是机械展开成 part domain + begin/end + source subregion，仍需从 provenance、诊断和变换需要判断，不能再把“能展开”误写成“语言没有这项能力”。

### 当前校正

- 拒绝“许可体系”的结论维持，但第二十轮删除所有 `parallel` 的结论撤回。
- `parallel` 保留为 unordered logical iteration construct，不作为 axis annotation，也不直接绑定 physical ownership。
- compiler 可以推导额外 data parallelism；显式 `parallel` 与普通 ordered `for` 是两种程序结构，而不是优化开关。
- partition 的 author-visible fixed/count region 能力恢复；删除的是 `auto` 与纯 physical blocking。专用 canonical `PartitionOp` 是否必要仍未冻结。
- 本轮只更新持续报告，不修改 `doc/`。

## 第二十二轮：保留 partition 语义能力，不保留 `partition` 构造

### 你的输入

> “为什么 `partition(count=P)` 又不删除了？明明讨论了半天，标准为什么一直改变？”

你指出第二十一轮再次把两个不同结论混写在一起：算法必须能表达 P-part 分解，不等于语言必须保留一个名为 `partition` 的 public/canonical operation。

### 我的输出

第二十一轮“`partition(count=P)` 保留”的措辞不准确，应撤销。固定判据没有改变：如果一个 named construct 可以用已经接受的普通构造机械展开，展开既不丢可观察语义，也不替作者选择具体 lowering，就不需要独立 public/canonical op。

`partition(count=P)` 可以完整写成：

```python
parts = I.domain(0, P)
source = I.domain(0, N)
width = (N + P - 1) // P

for part in I.parallel(parts):
    begin = I.minimum(part * width, N)
    end = I.minimum((part + 1) * width, N)
    region = source[begin:end]
    partial[part] = I.reduce.sum(x[region], axis=0, identity=0.0)
```

这里 P、part identity、边界公式、empty/tail region 与 partial ABI 全部仍由作者写下；删除 `partition` 没有删除算法功能。`partition(extent=E)` 同样可由普通 part/domain loop、`begin=i*E`、`end=min(begin+E,N)` 与 source slicing 表达。`partition(auto)` 则本来就是 compiler-owned blocking，不进入作者语言。

这与 `parallel` 不同：把 unordered parallel iteration 展开成普通 ordered `for` 会改变迭代顺序与 effect legality，因而不是机械 desugaring；`parallel` 仍有独立控制语义。它也与 `sparse_contract` 不同：把 structured sparse contraction 展开成 metadata decode、dense intermediate 与 ordinary contract 已经选择了一种 lowering，因此该 structured op 仍应保留。

### 当前校正

- 删除全部 public/canonical `partition` 构造，包括 count、fixed extent 与 auto 形式。
- 保留的是普通语言组合出的 logical partitioning 能力：part domain、identity、boundary arithmetic、source-derived subregion 和 explicit partial ABI。
- 第二十一轮关于 partition 是 author-visible feature 的实质含义只到“能力必须可表达”为止，不能推出 `PartitionOp` 必须存在。
- 当前 frontend/KIR/Plan/examples 中的 `PartitionOp` 路径属于后续迁移对象。
- 本轮只更新持续报告，不修改 `doc/`。

## 第二十三轮：`parallel(parts)` 与直接迭代 `parts` 的唯一差别

### 你的输入

> “到底还能不能收敛？`for part in I.parallel(parts)` 中 `I.parallel` 代表什么；如果直接迭代 `parts` 会怎样？”

### 我的输出

两种写法定义两种普通控制流语义，不是同一循环加一个优化开关：

```python
for part in parts:
    ...
```

是 ordered loop。按照 `parts` 的 logical order 执行，可以读取上一迭代产生的 state，可以观察 effect 顺序。compiler 只有在正常 dependence/effect analysis 证明变换等价时，才能把它并行化或重排。

```python
for part in I.parallel(parts):
    ...
```

是 unordered forall loop。iteration space 中每个 point 执行一次，但 points 之间没有顺序；不能有 loop-carried state；非 atomic/reduction 的冲突 effects 使程序非法。compiler 可以 serial、flatten、thread、SIMD、GPU grid 或 RVV strip-mine 实现它，`parallel` 本身不保证墙钟时间上的同时执行，也不绑定任何 physical level。

为避免它退化成作者“保证这里独立”的许可，legality 必须尽可能由 KIR verifier/analysis 从 SSA、index relation、alias 与 effects 证明；不能证明的 dynamic collision 不能因为作者写了 `parallel` 就静默放行。显式 unique/atomic/reduction operation 提供相应的可观察写入语义。纯 pointwise/唯一写入的普通整域表达即使没有显式 `parallel`，compiler 也可以直接推导 data-parallel realization。

因此 `parallel` 的存在价值不是帮助一个能力不足的 compiler，而是让作者在确实选择 unordered data-parallel control 时写出这一程序结构，并让依赖顺序的后续修改成为非法程序。普通 `for` 则保留 ordered algorithm。

### 当前校正

- `parallel` 的最终语义固定为 unordered forall，不再用 independence contract、permission、axis hint 或 ownership marker 描述。
- 直接迭代 domain/subregion 是 ordered loop；compiler 可以证明后再优化，但不能预设无序。
- 显式 parallel 不直接产生 GPU ownership；Physical Program 仍需选择 flatten、blocking 与 target execution mapping。
- 编程模型可以继续收敛；剩余问题集中在 contract 范围、sparse surface、plain copy、buffer/RNG 和数值/接口语义，不再包括 parallel/partition/ragged/state-stream 的方向摇摆。
- 本轮只更新持续报告，不修改 `doc/`。

## 第二十四轮：完整 DSL inventory 后剩余的冻结问题

### 你的输入

> “别的呢？我们的 programming model/DSL 还有其他需要讨论的问题吗？”

你要求不能只看此前点名的五项，而要把 public DSL、frontend semantics 与 canonical KIR 的完整表面过一遍，确认还有没有会在冻结后重新冒出来的基础问题。

### 我的输出

完整 inventory 审计后，剩余问题收敛为六组。它们都来自当前真实 surface/IR 的空白或矛盾，不是为了完整感而发明的新抽象。

1. **Domain、shape 与 tensor value。**需要定义 subregion iteration variable 是 source coordinate 还是局部 ordinal；symbolic/dynamic dimension 怎样绑定 source identity，不能继续把任意两个 dynamic dimensions 当成兼容；broadcast 的尾轴/标量/零维规则、reshape 的 numel 与 inferred dimension、`join` 究竟是 stack/pair/interleave 也必须唯一。当前 `DynamicDim` 宽松兼容和 `reshape` 不校验元素数会产生静默错误。

2. **函数与普通控制。**`@intent.fn` 是否允许 effects、是否允许 lexical capture、capture 怎样显式成为 operands、是否禁止递归，以及 helper 多结果/record 返回需要确定。kernel entry 仍应 zero-result，device outputs 通过 Out/InOut；普通 `break/continue` 采用 Python 最内层循环语义并结构化归一，不另造 op；structured `if` 只接受 scalar bool，tensor predicate 使用 select；独立 stop API 删除，由 loop condition/domain endpoint/subregion 表达。constexpr 只做硬件无关算法 specialization，选中分支进入 KIR，不查询 target。

3. **基础类型与数值。**要冻结真正的 canonical dtype 集合，特别是 i4/u4 是可计算 scalar type 还是仅 packed carrier；明确是否完全禁止隐式 promotion；负整数 div/rem、signed shift、overflow/除零/越界 shift；cast 的 rounding/saturation/NaN/Inf，bitcast 的位序与低位类型解释。这些是现有普通表达式的含义，不是 provider capability。

4. **Structured operations 的完整形状。**generic reduce/scan combine 的 typed pure closure、capture 与 effect 限制；identity/empty/NaN/tie/order；`contract` 范围；scaled/sparse format；histogram 数值语义。`arg_reduce.max`、`reduce.sum/max` 等只是 generic reduce shorthand；不能继续有第二条 canonical path。

5. **Memory 与 effects。**普通 indexing、gather 与 access validity/fill 如何统一；unique store、collision reduction 与 atomic family怎样归一；非 atomic load/store 不应携带未定义的 ordering/scope；logical buffer 的 initialization/alias/lifetime/mutation；RNG 的 algorithm、seed/counter 与跨 target reproducibility；in-place/partial overlap 是否只由 view alias semantics 表达。独立 fence/barrier 仍无跨 target 算法语义，不进入 DSL。

6. **Kernel interface 与 views。**shape/stride 的单位与动态传递、zero/negative stride、bounds、may-alias/noalias/同组 alias 的确切含义需要确定；alignment 若只是 vectorization/TMA 条件留给 provider，只有调用正确性真正依赖时才进入参数前置条件。这里不新增“ABI contract”语言，只定义函数参数和 view 的普通含义。

另有一批不需要继续讨论、只需迁移到唯一 canonical path 的清理：`mask → select`、`zeros → full(0)`、`I.add → +`、arg-reduce 与 builtin reduce sugar、ragged/members/accessors、以及已经判定删除但代码仍保留的 state-stream/partition/ragged paths。当前 doc/report/code 不一致也属于后续实施，不是新的设计分叉。

### 当前校正

- programming model 的方向已经有限收敛，但上述六组基础语义没有闭合前仍不能正式冻结。
- 真正需要讨论的优先顺序是：domain/shape → type/numeric → helper/control → memory/effects → structured-op细节 → view interface；后续 compiler 规格依赖这些事实。
- 普通 Python control、kernel/wrapper 边界、target physical exclusion 与 public/canonical sugar 分层不再作为开放方向反复讨论，只补精确定义和实现一致性。
- 本轮只更新持续报告，不修改 `doc/`。

## 第二十五轮：只保留真正未决的设计分叉

### 你的输入

你指出上一轮又把已经确定的 kernel/host、helper existence、parallel、partition 等内容重新讲了一遍，并把“尚未写出精确定义”和“编程模型仍有设计分叉”混成了一张大清单。你要求这一轮不修改 `doc/`、不实施迁移，只回答：经过前二十四轮之后，到底还有哪些抽象或 operation 真的没有想清楚。

你同时再次校正调查方法：当前 DSL/KIR/编译器实现必须调查，因为它能暴露真实算法需求、历史耦合和迁移影响；Triton/TileLang 也必须调查，因为它们提供成熟 kernel language 与 lowering 的反例。但二者都只是证据，不能反向规定最终 Intent surface，更不能因为当前实现已经有某个 op 就保留它。

### 我的输出

第二十四轮的六组 inventory 需要进一步分成三类，不能继续都叫“未决”。

#### A. 仍有真实抽象分叉，必须继续讨论

1. **`contract` 的语义边界。**它究竟是固定 multiply-plus 的一般 axis-pair tensor contraction，还是只覆盖能够稳定保持为 dot/GEMM-shaped structured operation 的形态。若定义过宽，target 会出现名义支持、实际只能 pointwise+reduce 的假能力；若定义过窄，作者又必须手写本来应保存的 contraction relation。这不是数值细节，而是 canonical op 的边界。
2. **structured sparse contraction 的 public/canonical 形状。**“必须 first-class 保存 compressed values、metadata→logical positions、logical contraction”已经确定；尚未确定的是使用 concrete `sparse_contract_2to4`，还是 `sparse_contract +` 一组语言定义的 concrete format schemas。这里不能退回 opaque generic descriptor，也不能让 provider 从显式 decode producer chain 猜 sparse MMA。
3. **`join` 是否存在以及表示什么。**当前名字没有唯一 denotation：pair/record、stack、concatenate、interleave 是不同的值构造。必须选择删除 `join` 并分别使用明确构造，或者给它一个唯一、不可与其它构造混淆的语义；不能保留含糊 op。
4. **memory access/effect 的 canonical 粒度。**普通 indexing 与 gather 是否归一为同一个“index relation + validity/fill”的 read；unique store、collision reduction 与 atomic 是三种独立 effect，还是一个带 collision mode 的 write family；这决定 canonical effects 怎样表达，而不是 surface 名字怎样拼。
5. **plain region copy 是否是算法 operation。**若精确 read→write relation 已能机械地产生整块 copy，`copy` 只是 provider form；若 copy 还定义 snapshot、overlap 或可观察 transfer 语义，它才需要作者级 operation。必须用真实重叠与非重叠例子裁决，不能因为 TileLang 有 `T.copy` 就抬进公共 DSL。
6. **跨 target atomic/RNG 的最小公共语义。**atomic 是否只承诺单地址原子 RMW，还是公共语言还承诺 scope/order；RNG 是固定 counter algorithm、保证跨 target bitwise 一致，还是只定义 distribution。这两项会改变程序可观察结果，不能留给 provider 偶然行为。

#### B. 方向已经确定，只需要把规格写精确

下面这些不是新的抽象分叉，不应再被包装成“是否保留某个 op”的讨论：

- source-derived subregion 已确定保留 parent/source coordinate；剩余工作只是写清 dynamic bounds、empty 和 provenance；
- dynamic shape 必须保留 identity/equality relation，不能让两个未知 extent 自动兼容；broadcast 与 reshape 需要唯一 shape rule；
- generic reduce/scan、scaled contract 与 histogram 已确定保留，剩余的是 identity、empty、NaN、tie、finite-precision reassociation、format 和 overflow 等 operation semantics；
- ordinary packed data 已确定使用 carrier tensor + 显式 bit/index/scale arithmetic；scaled/sparse structured ops 保存各自不可丢失的 format relation，不再重开 generic encoded type 讨论；
- logical buffer 已确定是作者级 kernel-local mutable state，剩余的是 definite initialization、escape、alias 和 lifetime；
- external view 已确定保存 dtype/shape/read-write/必要 alias 与 bounds 语义，stride 单位、zero/negative stride 和 noalias 只是需要精确定义；alignment 若不影响调用正确性仍属于 provider；
- `@intent.fn` 的存在、kernel/wrapper 边界、constexpr 只能选择硬件无关算法等方向已经确定；只需补 effects、capture、recursion 和 multi-result 的普通函数语义；
- integer div/rem、shift、overflow、cast/bitcast 与 implicit promotion 必须跨 target 唯一；这是基础数值定义，不是另建一种语言抽象。

#### C. 已经裁决，不再重新讨论

- `parallel` 是 unordered forall control，不是许可、axis hint 或 ownership marker；
- 普通 domain iteration 是 ordered control；
- `state_stream`、全部 `partition`、public `ordered` 和 `auto` 删除；
-独立 `ragged/members` op 删除，算法用 offsets、source subregion 与 indexed relation 表达；
- reduce、scan、contract family 是 structured tensor operations，不从任意普通循环猜回；
- scaled contract 与 structured sparse contract 不是 whole-operator builtin；
- softmax、attention、MoE 等仍由作者组合基础构造；
- 一个 kernel 对应一次 target launch，多 kernel 由 Python wrapper 编排；
- target storage/layout/tile/pipeline/copy spelling 不进入公共 DSL。

### 当前校正

- 下一轮真正需要继续裁决的只有 A 类六项；B 类在这些分叉确定后直接写精确定义，不再逐项重新辩论“要不要这个概念”；C 类只有出现新的不可约算法反例才允许重开。
- 当前实现调查的用途限定为发现真实 consumer、反例和迁移依赖；Triton/TileLang 调查的用途限定为检验作者级语义与 target lowering 边界。两者都不拥有 Intent 规格的最终决定权。
- 本轮只更新持续讨论报告，不修改 `doc/`、实现或 examples。

## 第二十六轮：用统一判据和真实下层设计裁决六项分叉

### 你的输入

你要求先给出判据和理由，再由你审查；不能只靠此前对话记忆，也不能避开当前实现和 Triton/TileLang 的真实源码。当前实现用于发现真实 consumer、历史耦合和迁移影响，成熟下层用于检验 operation 的作者级语义和 lowering 边界，但都不能反向规定 Intent 最终必须维持现状。

### 我的输出

#### 统一裁决判据

六项都按同一顺序判断：

1. 先写出不含 target API、storage、tile、participant 或 instruction 的 denotation；写不出来的不是跨 target 作者语义。
2. 尝试用已经接受的普通 value/control/effect constructs 展开。若展开只是在语法上机械改写，删除专用 op；若展开已经选择 reduction order、物化中间值、collision behavior、snapshot、format decode 或其它 lowering，保留 first-class semantics。
3. 检查删除后 compiler 是否必须从任意 producer/use pattern 重新证明该结构。若必须猜回 axis relation、sparse metadata、atomicity 等事实，说明 canonical IR 丢了语义。
4. 检查 GPU、CPU、RVV 是否能实现同一 denotation；不要求都有同名或 native primitive。单个 target 不支持只能形成 capability subset，不能决定公共语义不存在。
5. 对照 Triton/TileLang 时拆开它们 public op 中的算法字段与 target-only 字段。只复用前者的证据，不整体复制 surface。
6. 现有 Intent op/use 只证明需求和迁移影响。实现里已有字段、Plan 路径或 emitter 特判不能作为保留理由。

#### 1. `contract`：保留 dot-normalizable contraction，而不是任意 einsum

Triton `tl.dot` 的作者语义是 2D/3D matrix product；实现允许多个同形 batch axes，但会把它们 flatten 成一个 batch，再进入 2D/3D dot。其固定关系是 A 的最后一维 K、B 的倒数第二维 K、可选同形 batch 与 M/N free dimensions（`ref/triton/python/triton/language/core.py:2215-2287`，`ref/triton/include/triton/Dialect/Triton/IR/TritonOps.td:666-701`）。TileLang `T.gemm` 同样只消费最后两维 M/K/N，leading axes 必须为 1；batched GEMM 由作者 loop 外包（`ref/tilelang/tilelang/language/gemm_op.py:62-92`）。

这证明 first-class structured boundary 不是任意 einsum，也不能窄化成只有两个固定 rank-2 tensor。倾向裁决是：

- canonical `contract` 固定 multiply-plus 与显式 accumulator dtype；不接受任意 semiring closure；
- operand axes 显式分成同形 batch axes、lhs free group、shared reduction group 和 rhs free group；每根 operand axis 必须且只能归入一组；
- 多个 free/reduction axes只有在 relation 可证明为双射 flatten 时，才属于同一个 dot-normalizable contract；flatten/reshape 是 target program 的合法 reindexing，不由 leaf 从 shape 猜；
- diagonal、重复轴、无法分组成 M/K/N 的任意 einsum，以及其它 semiring 使用 pointwise + generic reduce。

这样保留 grouped/ragged/batched contraction 的逻辑轴 provenance，又不会把“目标只能展开成普通 reduce”的任意 einsum冒充 matrix structured support。

#### 2. sparse contraction：统一 operation，加 closed typed format schemas

Triton 当前没有 public sparse-dot；`tl.dot_scaled` 只有 microscaling formats/scales，不带 sparse metadata（`ref/triton/include/triton/Dialect/Triton/IR/TritonOps.td:708-755`）。TileLang `T.gemm_sp` 明确接收 compressed A、metadata E、dense B 与 accumulator C，但同时混入 warp policy、k-pack、wait、stride/offset 等 target 字段（`ref/tilelang/tilelang/language/experimental/gemm_sp_op.py:95-159`）。更关键的是 TileLang 的真实 layout 表显示 float32 使用 1:2，f16/bf16/int8/fp8 使用 2:4，并非只有一种 2:4 schema（`ref/tilelang/tilelang/cuda/intrinsics/sparse_layout.py:13-20`）。

倾向裁决是：

- canonical 使用一个 `sparse_contract` family；
- `format` 不是任意字符串或 opaque descriptor，而是语言逐个定义的 closed typed schema；真实存在 1:2、2:4 就分别定义各自的 group、nonzero count、compressed ordering、metadata→logical-K mapping、合法 metadata 与 dtype rules；
- op 继续保存 logical contraction axes、accumulator 与 numeric semantics；
- metadata physical dtype/layout、shared/private placement、K tile、warp policy、native instruction 等留给 target program；
- `sparse_contract_2to4` 可以保留为 surface shorthand，但归一到唯一 `sparse_contract(format=two_of_four(...))` canonical path。

这既不要求 Triton 伪装拥有 sparse MMA，也不迫使 TileLang provider 从 decode/gather/contract producer chain 猜回 structured sparsity。

#### 3. `join`：不是待删的含糊 op，而是文档漏掉了唯一语义

真实 Triton `tt.join` 明确定义为：两个同 dtype、同 shape tensor 沿新的最末轴堆叠，结果 shape 是 `input_shape + [2]`；`tt.split` 是其逆操作（`ref/triton/include/triton/Dialect/Triton/IR/TritonOps.td:534-571`）。当前 Intent frontend、verifier 和 emitter 实际也是这一语义。Mamba 的真实 consumer 先 `join(first, second)`，再按 row-major logical order reshape，得到 `[first0, second0, first1, second1, ...]` 的 pair interleave（`examples/kernels/streaming/mamba.py:275-333`）。

因此第二十五轮把 pair/record/stack/concat/interleave 都列为 `join` 的候选含义，是文档缺定义导致的误判。倾向裁决是保留 `join`，精确定义为 two-input same-shape minor-axis stack：

```text
result[..., 0] = lhs[...]
result[..., 1] = rhs[...]
shape(result) = shape(lhs) + [2]
```

它不是 record、concatenate 或一般 interleave。当前没有真实需求迫使语言预建任意 arity/axis 的 generic stack；也无需为了名字更直白而无证据重命名。

#### 4. memory access/effect：共享 index relation，不合并不同 effect semantics

Triton 本身区分 predicated `tt.load/store`、pure tensor `tt.gather` 与 `tt.atomic_rmw/cas`；atomic 还返回旧值并携带 memory semantics（`ref/triton/include/triton/Dialect/Triton/IR/TritonOps.td:214-314,364-440,949-979`）。当前 Intent 的普通 indexing 与 `gather` 已经证明一个有用方向：二者都可产生 index relation，而 validity/fill 决定 invalid lane 是否访问以及返回什么；但 external view、logical buffer 和 immutable tensor 的 effect/resource 不同，不能合并成一个无类型“read”。

倾向裁决是：

- 所有 read/write 共用一等 `IndexRelation` 与 active validity；read 另外保存 invalid fill；
- immutable tensor indexed selection 是 pure gather；external view load 与 logical-buffer load 分别保留对应 read effect；三者共享 relation 表示，不共享 effect identity；
- 普通 assignment 与 `scatter_unique` 归一为同一个 arbitrary-index unique store：目的 relation 必须可证明 injective，或由明确 precondition 保证；direct/indirect 只是 relation 形态，不是两种 collision semantics；
- collision reduction 继续是独立 `scatter_reduce`，保存 typed combine/reassociation；
- atomic 继续是独立 `atomic_rmw/cas`，保存 per-update indivisibility 与旧值结果；
- 非 atomic load/store/gather/scatter 不携带 ordering/scope。validity 必须阻止非法访问，不能先越界访问再 value-select。

因此这里收敛的是共享 relation machinery，不是把 pure gather、memory read、reduction collision 与 atomicity 压成一个 generic write mode。

#### 5. plain copy：不进入 canonical Kernel IR

Triton 没有通用 `copy(src,dst)`；作者写 load/store，descriptor load/store、cp.async、TMA 是 target transfer forms。TileLang 有一等 `T.copy`，但它的 public op 同时携带 coalesced width、TMA/cp.async preference、eviction policy 和 loop layout；scalar copy甚至直接机械化为 BufferStore（`ref/tilelang/tilelang/language/copy_op.py:17-50,54-140`）。这正说明 TileLang `copy` 混合了 region assignment 与 target memory-space realization，不能整体抬到 Intent。

倾向裁决是：

- canonical 算法使用一次精确 indexed read 产生 immutable SSA tensor value，再用精确 indexed write 消费；RHS value semantics 自然给出 snapshot；
- source/destination overlap、broadcast、reorder、cast 与 validity 都由这两条 relation 和普通 SSA/effect order明确表达；
- shared/provider physical pass 可以在 producer-consumer relation满足单 use、双射覆盖、dtype/validity兼容、alias合法等条件时，形成 synchronous copy、vector transfer、cp.async 或 TMA；
- async completion、barrier、memory space 和 instruction selection只存在于 target program；
- 若作者需要顺序相关的 in-place move，就显式写 ordinary loop 或 logical temporary，而不是用一个含糊 copy op隐藏 overlap 语义。

也就是说，识别 exact read→write transfer 是 compiler 的局部 def-use/access analysis，不是 whole-kernel matcher；当前 compiler 识别不好应修 compiler，不能要求作者预写 provider-friendly copy skeleton。

#### 6. atomic 与 RNG：不能照搬 GPU scope；RNG 必须固定 bits

Triton atomic 把 `acquire/release/acq_rel/relaxed` 和 `cta/gpu/sys` 暴露给作者（`ref/triton/python/triton/language/core.py:2533-2557`）；这在 Triton 的 GPU SPMD model 中有明确 participant。TileLang 也暴露 memory order，但 tile-region atomic 路径会忽略它，说明“API 有字段”不等于所有 lowering 都兑现同一语义（`ref/tilelang/tilelang/language/atomic.py:39-122`）。Intent 抽走了 workgroup/subgroup/program ownership，因此不能把 `cta/workgroup/subgroup/device` 字符串直接复制进跨 CPU/RVV 的作者语义。

倾向裁决是当前公共 atomic 只承诺：

- 对同一 logical address 的 RMW/CAS 在本次 kernel 的全部 logical executions 间线性化；
- RMW/CAS 返回旧值；
- 对其它地址和普通 memory effects 不建立同步顺序，即固定为 relaxed；
- 不暴露 physical scope。若真实算法需要 acquire/release 同步其它 memory，必须先建立跨 GPU/CPU/RVV 的 logical participant 与 memory model；在此之前明确不支持，不能用 target scope 字符串冒充。

这足以覆盖 histogram、gradient accumulation、routing counter 等真实 collision uses；纯碰撞合并优先使用 `scatter_reduce`，不应为了实现方便一律变成 atomic。

RNG 的情况相反：seed/counter 到结果 bits 是直接可观察算法，provider 不能自由替换。Triton 的 `tl.rand` 使用固定十轮 Philox，seed 与 logical offset组成 counter（`ref/triton/python/triton/language/random.py:5-110,126-156`）。TileLang 的通用 language surface 没有同形的 stateless counter API，但 CUDA 专属 namespace 封装了 cuRAND state，默认 generator 是 `curandStatePhilox4_32_10_t`；若 sequence 未给出，还会从 block/thread binding 取得 physical identity（`ref/tilelang/tilelang/cuda/language/random.py:7-39`）。它证明 TileLang CUDA 可以调用 Philox family，不证明该 stateful/physical mapping 与 Intent 的 logical counter semantics 相同。当前 Intent 的 xorshift32 只能证明已有需求和实现，不足以成为最终算法选择。

倾向裁决是使用固定 counter-based Philox4x32-10 作为 canonical random-bits primitive：seed、logical counter、uint32 wrap 和输出 words 全部定义为跨 target bitwise 一致；uniform f32 由固定 bit mapping 得到。normal 等复杂 distribution 由普通作者 library 在该 bits primitive 上构造，不让 target 自选 RNG。CPU/RVV/TileLang 没有 native API时机械发出同一整数算法。

### 当前校正

- 六项中，`join` 已由真实 Triton IR 和现有 consumer 证明有唯一语义，不再属于抽象分叉，只需补规格。
- `contract` 倾向收敛为 dot-normalizable relation；sparse 倾向收敛为统一 op 加 closed typed format schemas；memory access倾向共享 relation但保留 effect families；plain copy 倾向排除出 canonical KIR；atomic倾向固定 relaxed、无 physical scope；RNG倾向固定 Philox bits。
- 上述仍是供你审查的倾向裁决，尚未同步 `doc/`，也没有修改 DSL、KIR、compiler 或 examples。

## 第二十七轮：澄清 atomic、当前 contract 与 TileLang RNG

### 你的输入

你追问三点：为什么 histogram/gradient/routing counter 足以使用当前建议的 atomic，而 acquire/release 还需要另一个 logical memory model；dot-normalizable contract 与当前 `I.contract` 到底有什么差别；以及若 RNG 固定 Philox，TileLang 实际怎样处理。

### 我的输出

#### Atomicity 与 memory ordering 是两件事

`atomic_add(counter[index], value)` 解决的是同一地址上的更新不能丢失。所有参与者对该地址的更新形成一个原子次序，就足以定义 histogram bin、gradient sum 和 routing cursor。它不要求某个参与者看见另一个参与者此前写入的其它地址。

Acquire/release 解决的是另一种程序：一个参与者先写普通 data，再 release 写 flag；另一个参与者 acquire 读到 flag 后，必须同时看见此前的 data。要跨 target 定义它，语言还必须说明：哪些 logical executions 是同步参与者、同步覆盖哪些 memory、kernel 内是否保证等待方与发布方共同取得执行进展。Triton 的 `cta/gpu/sys` 依赖它已经暴露的 GPU execution model；Intent 当前没有 public CTA/subgroup/program identity，直接复制这些名字没有跨 CPU/RVV 的 denotation。

因此“先只保留 relaxed”不是说 acquire/release 永远不属于算子语言，而是：当前真实 uses 只需要单地址 atomicity；在没有 logical participant、visibility 和 progress semantics 前，不应先暴露一组无法跨 target 兑现的 scope/order 枚举。若后续真实算法需要用 atomic 发布其它 memory，再单独定义 target-independent memory model，而不是让 provider 猜。

#### 当前 `I.contract` 已经基本是 dot-normalizable relation

重新对照 frontend 后，当前 surface 已经要求 reduction axis pairs 唯一且 extent compatible，batch pairs 唯一且不能与 reduction 重叠，其余 lhs/rhs axes自然成为 free axes；同时只接受 multiply/add（`python/intent/frontend/lowering/intrinsics/structured.py:314-405`）。这正是“batch + lhs-free + reduction + rhs-free”分组，多个 reduction/free axes只要做轴置换与双射 flatten 就能归一成 M/K/N。

所以第二十六轮的建议不是大改当前 DSL，也不是删除多轴 contraction。真正要收紧的是规格与假 surface：

- 把这种 axis grouping 明写成 `contract` 的正式边界，不再用“任意 axis-pair contraction”让人误解为 arbitrary einsum；
- public surface 不再暴露实际上永远只允许 `multiply`/`add` 的伪 generic parameters；其它 semiring直接写 pointwise + reduce；
- diagonal/repeated-axis、非双射 relation 等当前本来就不支持的形态继续排除；
- 当前 Plan 只保留单 reduction-axis node 是 compiler implementation gap，不能反向缩窄 DSL 的多轴语义。

除去是否允许 empty reduction/outer-product 继续由 `contract` 表达这一小项，当前 public semantics 与建议边界基本一致。

#### TileLang 有 CUDA RNG，但不是同一抽象

TileLang CUDA 专属 `rng_init` 封装 cuRAND，允许 Philox、MRG32k3a 和 XORWOW，默认 Philox4x32-10；它返回 mutable RNG state，后续 `rng_rand/rng_rand_float` 按调用顺序推进。若作者没有显式传 sequence，它使用 block/thread binding 生成 sequence（`ref/tilelang/tilelang/cuda/language/random.py:7-70`）。

Intent 要保存的是 `random_bits(seed, logical_counter)`：结果不能随 GPU block/thread mapping、CPU scheduling、RVV strip-mining 或调用顺序变化。因此 TileLang provider 有两条合法实现路线：

1. 机械发出同一套 Philox integer rounds，最直接保证 bitwise 一致；
2. 只有在证明 cuRAND 的 seed/sequence/offset 到输出 word 与 canonical logical-counter mapping 完全一致后，才投影到 TileLang CUDA RNG。

不能省略 sequence 让 TileLang 自动读取 physical id，也不能因为双方都叫 Philox 就假设 bit stream 相同。TileLang 缺一个同形 stateless API 是 provider projection 问题，不是公共 RNG 语义需要改成 stateful 的理由。

### 当前校正

- atomic 的真实未决点缩小为：是否现在就需要一套跨 target acquire/release memory model；现有 corpus 没有给出 forcing case，relaxed per-address atomicity 已覆盖当前算法。
- `contract` 的建议与当前 frontend 语义基本一致，主要是正式命名边界、删除伪 generic 参数并保留多轴 grouping；不应把它描述成一次新的 surface 收窄。
- 第二十六轮“TileLang 没有 kernel-level RNG”的说法已纠正为：TileLang 有 CUDA/cuRAND stateful RNG，但没有与 logical stateless counter 完全同形的通用 surface。
- 本轮仍只更新持续报告，不修改 `doc/` 或实现。

## 第二十八轮：成熟规格不能把 atomic memory model 留到以后

### 你的输入

你指出“当前 histogram/gradient/routing 只需要 relaxed”不能成为成熟语言只定义 relaxed 的理由。Intent 正在定义最终 programming model，不是按当前 corpus 做最小功能集；atomic 既然进入语言，就必须说明 acquire/release 怎样跨 target 成立、怎样投影到 Triton，而不是把问题挂到未来。你同时追问 TileLang 是否意味着固定 Philox 实际能够实现。

### 我的输出

前一轮把“当前 forcing cases 的最小需求”和“最终语言边界”混为一谈，结论应修正。Intent 保留完整的 portable atomic family与 memory ordering，但不复制 GPU physical participant names。

#### Portable atomic semantics

每个 atomic address 有单一 modification order；atomic load/store/RMW/CAS 对该地址不可分割。公共 memory order 为：

- `relaxed`：只保证该 atomic address 的原子性和 modification order；
- `acquire`：读取到 matching release/release sequence 后，观察到其之前的 ordinary/atomic effects；
- `release`：把本次操作之前的 ordinary/atomic effects发布给 matching acquire；
- `acq_rel`：同时具有两者语义。

公共 visibility scope 只保留有跨 target denotation 的：

- `device`：当前执行设备上的 logical agents；GPU 映射为 Triton `gpu`，CPU 映射为本进程/设备执行域，RVV 映射为该执行设备的 harts/agents；
- `system`：共享该 external allocation 的 system agents；GPU 映射为 Triton `sys`，CPU/RVV 映射为 system-scope atomics。

不暴露 `cta/workgroup/subgroup/work_item`。这些 identity 由 GPU/RVV/CPU physicalization 产生，不是作者逻辑程序的一部分。若 compiler 证明一次 atomic 的全部参与者落在同一个物理 group，可以在 target program 中把 `device` 合法收窄成 `cta`，但作者语义仍是 device scope。

普通非 atomic conflicting accesses 仍是非法 data race。语言不保证两个 parallel iterations 的同时驻留或 spin-wait forward progress；这是一项明确的 execution guarantee，而不是未定义的 scope。Acquire/release memory visibility可以完整定义，即使某个依赖同时驻留的锁算法可能因目标调度不保证进展而不终止。

canonical atomic family至少包括 typed `load`、`store`、`exchange`、RMW（add/max/min/and/or/xor 等）与 compare-exchange；RMW/CAS 明确返回旧值，CAS 还返回 success 或可由旧值比较得到。ordering/scope 是 atomic op 的字段，不出现在普通 read/write/gather/scatter 上。

#### Triton/TileLang/CPU/RVV projection

- Triton：`relaxed/acquire/release/acq_rel` 直接映射 `sem`；`device/system` 映射 `scope="gpu"/"sys"`；RMW/CAS 映射 `tl.atomic_*`。如果某种 dtype/op 没有直接 primitive，provider可用保持同一 ordering/scope 的 CAS loop，或明确 unsupported，不能弱化语义。
- TileLang：scalar atomic path可携带 memory order；当前 tile-region path 对 order/return-old 的兑现不完整，因此 acquire/release 或 old-value-sensitive op必须选择能兑现的 scalar/provider form，做不到的组合明确 unsupported。不能因为它的 tile form忽略字段就降低公共语义。
- CPU：映射为对应 LLVM/C/C++ atomic ordering 与 device/system visibility。
- RVV：映射为 RISC-V AMO/LR-SC 以及必要的 acquire/release bits/fence；向量化只改变发射方式，不改变每个 logical address 的 atomic semantics。

这是一套完整的语言语义，不依赖当前 corpus 是否已经出现锁或跨地址发布算法。

#### RNG 已可完整闭合

固定 Philox4x32-10 不依赖 target 提供同名高层 API。canonical primitive 定义 `seed + logical counter -> four uint32 words` 的 rounds、wrap 和 bit mapping；Triton可以调用/展开自己的 Philox，TileLang可以用普通 uint32 arithmetic 发出同一 rounds，CPU/RVV同样机械实现。TileLang CUDA/cuRAND 只是可选加速路径：只有验证 seed/sequence/offset 与 canonical counter逐 bit 相同才使用，否则不使用。

因此答案是“能做”，而且不存在 TileLang capability gap；最坏情况是 provider 发出若干普通整数运算，不需要新增公共 DSL 抽象。

### 当前校正

- 第二十六、二十七轮“atomic 倾向只定义 relaxed、其它留待未来”的建议撤回。
- 最终方向改为：atomic family、四种 memory order、device/system 两种 logical scope现在一次定义完整；target-specific narrow scopes不进入 DSL。
- RNG 固定 Philox4x32-10 的方向可以闭合，TileLang 通过机械整数 lowering 或验证后的 cuRAND projection 实现。
- 本轮仍只更新持续报告，不修改 `doc/` 或实现。

## 第二十九轮：atomic family 必须落实成具体 operations

### 你的输入

你追问为什么说 atomic “family”而不是直接定义 operation。这个问题指出了措辞风险：如果 family 最终只是一个 `atomic(kind=任意东西)` 的泛化入口，它会隐藏不同操作的 operand/result、合法 ordering 和 target capability，不能成为正式语言定义。

### 我的输出

“family”只表示文档分类，不是 public 或 canonical operation。正式模型应当有四类具体 canonical ops：

1. `atomic_load(address, order, scope) -> value`；
2. `atomic_store(address, value, order, scope)`；
3. `atomic_rmw(address, value, kind, order, scope) -> old_value`，其中 `kind` 是封闭 enum：`exchange/add/max/min/and/or/xor` 等具有同一 operand/result schema 的 RMW；
4. `atomic_compare_exchange(address, expected, desired, order, scope) -> {old_value, success}`。

不能把四类压成一个 op，因为 load 没有写入 value、store没有结果、RMW有一个 update operand并返回旧值、CAS有 expected/desired 两个 operand和成功结果；它们允许的 memory order也不同：

- load 只允许 `relaxed/acquire`；
- store只允许 `relaxed/release`；
- RMW允许 `relaxed/acquire/release/acq_rel`；
- CAS使用一个 success order，并按标准单-order规则派生 failure order：`release -> relaxed`、`acq_rel -> acquire`，failure不能具有 release 语义。

同一 schema 的 add/max/min 等无需各造一份 canonical op；封闭 `kind` enum不是 opaque callback，也不允许用户自定义任意 atomic combine。public surface可以提供 `I.atomic.add/max/...` 等清楚方法，frontend机械归一到 `atomic_rmw(kind=...)`。CAS独立，因为其 schema和语义确实不同。

这与 Triton 的真实分层一致：public surface有 `tl.atomic_add/max/.../cas` 等具体函数，TTIR把同 schema 的 RMW 收敛到带 `atomic_rmw_op` enum 的 `tt.atomic_rmw`，CAS则是单独的 `tt.atomic_cas`（`ref/triton/include/triton/Dialect/Triton/IR/TritonOps.td:364-440`）。Intent 复用的是这种语义粒度，不复制 Triton 的 GPU scope。

### 当前校正

- “atomic family”只保留为章节分类；正式 DSL/KIR 使用具体 load、store、RMW 和 compare-exchange operations。
- public convenience methods与 canonical op 分开：多个 RMW名字归一到一条 typed `atomic_rmw`，不是多条 lowering路径。
- 本轮仍只更新持续报告，不修改 `doc/` 或实现。

## 第三十轮：target 可以出现在编译说明中，不能成为 DSL 语义

### 你的输入

你追问我是不是把“任何暗示 GPU 的文字”都判断成错误，并要求说明理由。你同时强调，裁撤 operation 必须沿项目已经形成的抽象逻辑判断，不能因为 Triton 没有同名 op、某个词看起来像 GPU，或者当前实现方便与否就拍脑袋删除。本轮结论必须写入持续对话报告。

### 我的输出

上一轮把“语言不携带 target identity”推成“规格不能提到 target、device artifact 或 lowering”，过度了。正确边界不是词汇过滤，而是语义依赖。

#### 合法出现 target/GPU 的地方

- programming-model 文档说明同一 Kernel IR 会由外部编译调用选择 Triton、TileLang、CPU 或 RVV；
- 说明一个 `@intent.kernel` 最终生成某个 target 的 kernel artifact，并由 host/runtime launch；
- 解释某项 canonical semantics 在不同 target 上可能一对一投影、一对多展开或明确 unsupported；
- 用 Triton/TileLang 源码作为语义粒度与 lowering 边界的证据；
- compiler/physical-program 文档描述 GPU grid、CTA、warp、TMA，或 CPU/RVV 的物理 realization。

这些文字描述编译链和结果，不要求作者程序依赖 target，因此不是 DSL 泄漏。`kernel-and-host.md` 中 “target kernel artifact/device launch” 需要澄清 target 由编译调用外部选择，但不是因为出现 `target/device` 单词就语义错误。

#### 真正错误的 target 泄漏

只有当作者 source 或 canonical KIR 必须携带下面内容时才越界：

- `target == "triton"`、GPU 型号、ISA 或 provider capability 分支；
- program/CTA/warp/lane/hart 等 physical participant identity；
- physical tile、memory space、layout、pipeline、TMA/MMA form；
- 只有选定 target 后才有定义的 scope、hint 或 tuning value；
- 用某个 target 是否有 native primitive来决定一项跨 target 算法语义是否存在。

同一个 source 可以在不同 target 上得到不同 physical program，但 logical values、control、effects 与接口语义不能因此改变。

#### operation 保留或裁撤的固定逻辑

不再使用“某个 target 有/没有同名 op”作为裁决。每项 operation 依次经过：

1. 抽掉 target spelling、physical identity、storage、tile 与 instruction 后，写出其 logical values/control/effects denotation。
2. 用已经接受的基础构造展开，检查是否完全机械，是否保存结果、顺序、effects、alias、shape、numeric 和 interface semantics。
3. 如果展开需要替作者选择 reduction order、中间物化、snapshot、collision behavior、format decode或其它 realization，它不是 desugaring，该 semantics 必须 first-class 保留。
4. 如果机械展开后没有信息损失，专用 canonical op删除；确有可编程性价值时可保留 surface helper，但只走唯一 canonical path。
5. 最后用 GPU、CPU、RVV 反例检查同一 denotation 是否成立；target capability只影响 lowering 是否成功，不反向改定义。

#### 按该逻辑重新核对 operation

- `parallel` 保留：展开成 ordinary ordered `for` 会改变顺序、loop carry 与冲突 effect legality。
- reduce/scan/contract 保留：普通循环无法机械恢复 reassociation、prefix 与 paired-axis structured semantics。
- scaled contract保留：microscaling format、scale group 与 logical operand interpretation是数值语义；不是因为 Triton 有 `dot_scaled`。
- sparse contract保留：compressed values、metadata→logical positions 与 logical contraction不可在不选择 decode/materialization lowering 的情况下删除；不是因为 TileLang 有 `gemm_sp`。
- histogram保留：它是返回 count tensor 的纯 structured value semantics；外部 buffer初始化加 atomic update已经选择了 effectful lowering。
- atomic load/store/RMW/CAS保留：不可分割性、modification order、旧值与 happens-before是可观察 effects。
- `join` 保留：two-input same-shape、new trailing logical axis 的 element mapping是明确 value construction；不是因为 Triton 有同名 op。“minor axis”应改写成 trailing logical axis，避免误读成物理 layout。
- `state_stream` 删除：其合法含义分别由 reduce、scan 或 ordinary ordered recurrence 完整表达，剩余 segmentation skeleton属于 realization。
- `partition` 删除：part identity、边界公式、source subregion 与 partial ABI可由普通 control/value/relation机械表达；不可观察 blocking属于 realization。
-独立 `ragged/members` op删除：offsets-derived subregion和indexed relation就是其全部算法语义；helper可机械展开。
- plain canonical `copy` 删除：immutable SSA read加indexed write已经完整定义 snapshot、mapping、validity、cast和effect order；bulk/async/TMA copy是physical realization。可保留普通 assignment或机械 helper，不建立第二条 canonical op。

#### atomic scope 为什么删除

删除 `scope=` 不是因为 `device` 这个词暗示 GPU，而是因为在单个 kernel 的算法语义中，atomic object 已由 logical allocation identity、alias/index relation 与本次 invocation 的 executions确定。作者再写 `device/cta/system` 没有增加一项稳定、跨 target 的算法结果，反而提前约束尚未生成的 physical participant domain。

`order=` 不同：acquire/release会改变其它普通 memory effects 的可见顺序，因此是作者可观察语义，必须保留。provider选定 target 后，从 logical sharing与access关系选择 Triton `cta/gpu/sys`、CPU atomic domain或RVV AMO/fence，这是 physical answer。

若 host 需要多个 kernel 或 host/device 并发共享 atomic object，那是 Python wrapper/runtime concurrency 与 external allocation semantics，不应靠单个 kernel body中的 physical scope字符串隐式扩大。

### 当前校正

- “规格里出现 GPU/target 一词就是错误”的判断撤回；判断标准改为作者/KIR语义是否依赖 target identity。
- contract 的 public语义应继续写一般二元 paired-axis multiply-add contraction，并要求非空 reduction group；没有 reduction 的 outer product机械写成 broadcast multiply。“dot-normalizable/M-K-N”只是 compiler analysis与lowering theorem，不进入 DSL 定义。
- operation 去留按 denotation与机械 desugaring裁决；上述保留/删除结果在该判据下保持一致，不依赖 Triton/TileLang 同名 API。
- 本轮只更新持续讨论报告，不修改 `doc/` 或实现。

## 第一阶段当前统一判断

### 作者写下

- kernel interface、logical workset、数值路径和 effects；
- 硬件无关 `if/for/while`、unordered `parallel`、state、终止条件与顺序；
- domain、source-derived subregion、parallel iteration、offset/index data relations；
- reduce、scan、contract、scaled/sparse contract、histogram 与 atomic/RNG 语义，以及显式 part identity、boundary formula 与 partial tensor interface；
- 一个算法使用一个还是多个 kernels。

### Intent compiler 补回

- GPU program/grid、CPU loop/SIMD、RVV strip-mining；
- 从 def-use、index relation、effects、alias/dependence 推导 iteration independence 与 ownership；
- structured-op streaming、physical blocking、machine vector packing、value/access realization；
- 地址、尾块、storage、copy、sync 和 provider program；
- 合法 tuning surface，但不替下层选择 winner。

### Target 与下层继续完成

- Triton/cuTile/TileLang/RVV 的具体类型、layout、primitive 与机器 lowering；
- register、instruction、低层 pipeline 和最终 autotune winner。

### 保留的 canonical semantics

- generic reduce、scan、contract、scaled contract、structured sparse contract 与 histogram；
- domain、source-derived subregion、unordered parallel iteration、普通 ordered control、typed shapes、join、indexed relations 与 effects；
- packed/quantized data 的显式 indexing、bitwise 与 scale arithmetic；
- 普通循环的 loop-carried SSA、source-derived subregion 和 part identity；
- atomic load/store/RMW/CAS 的 memory order，以及 stateless Philox random bits；
- 作者可观察的 page/window/chunk boundary、显式 partial tensor 与 wrapper orchestration；
- Python wrapper 的 multi-kernel orchestration。

### 保留的 surface shorthand

- `reduce.sum/max`、`any/all` 与 `arg_reduce.max`；
- `indices`、endpoint、source slicing 与普通 advanced indexing；
- `ragged/members` 等机械 relation helper；
- `zeros`、value mask、常用 pointwise helper 与 format-specific sparse spelling。

### 排除的作者表面

- public `ordered`；
- `state_stream`、全部 `partition` 形式以及 `auto`；
- 独立 `ragged/members` relation constructor；
- 无作者可观察语义的 compiler blocking skeleton；
- 命名或数值化的 physical partition/tile extent；
- program id、warp/thread、physical storage/layout/pipeline；
- 作者命名的 compiler-owned `K_TILE/N_TILE/...` tuning identity。
- TileLang/Triton provider 专属的 WGMMA、TCGEN05、TMA、mbarrier、warp policy 与显式 pipeline form。
- canonical copy operation 与作者可见 physical atomic scope；
- target/provider/device/ISA identity、capability branch 与 autotune configuration。

## 第一阶段已闭合

programming model 与 DSL 的抽象分叉已经闭合，最终定义已同步到 `doc/programming-model/` 与 `doc/dsl/`：

- `state_stream`、全部 `partition`、public `ordered` 与独立 `ragged/members` canonical ops 删除；
- `parallel` 固定为 unordered forall，普通 domain iteration保持 ordered；
- reduce/scan/一般二元paired-axis contract、scaled/sparse contract与histogram first-class保留；
- source subregion、dynamic shape identity、broadcast/reshape/join、types/numerics、views/buffers、indexed access、atomic operations与Philox RNG均有唯一语义；
- target由编译调用外部选择，physical copy、scope、storage、layout、pipeline和execution mapping不进入DSL/KIR。

剩余工作是按该规格迁移frontend、canonical KIR、compiler、providers与examples，删除旧节点且不保留第二条executable path；这不再是programming-model未决项。

## 后续完整重构范围

programming model 与 DSL 审定后，本报告继续记录：

- canonical Kernel IR 的唯一语义合同；
- executable physical program 的真实 IR 形态；
- compiler decision space 与 pass pipeline；
- shared、Triton、cuTile、TileLang 以及未来 CPU/RVV 的责任边界；
- examples/source/runtime/baseline 的重新接入和最终验证。

这些阶段在前三十轮中尚未展开，不能用第一阶段的 DSL 结论提前替它们作答；第三十一轮开始进入 compiler 阶段。

## 第三十一轮：从已冻结的 KIR 进入 compiler，重读 V2 自查与 Triton 两层 IR

### 你的输入

> “编程模型和 DSL 基本确定了。下一步是 compiler 本身。先看 `gpu-compiler-space-pass-and-baseline-contract.md` 暴露的问题；继续参考 Triton 的两层，一个 TTIR、一个 TTGIR。先搞清楚我们现在的 compiler 到底应该怎样设计，对话继续记录在这份持续报告中。”

你同时强调：前一阶段已经裁决的是作者语言，不应在进入 compiler 后又把旧 `state_stream`、`partition`、独立 ragged op 或旧 Plan 结构当成新设计前提。Compiler V2 和自查报告都只是证据；当前代码、Triton 实现与新的 KIR 语义要重新对齐。

### 我的输出

我完整复核了自查报告、当前 `intent_plan` 主链和真实 Triton TTIR→TTGIR pipeline。自查的核心诊断仍成立，但其中列举的旧 KIR operation inventory 已被前三十轮 DSL 裁决淘汰，不能机械照搬。

当前 `intent_plan.exec_*` 并不是一份已物理化的 executable program。conversion 只是把 `intent.foo` 改名为 `intent_plan.exec_foo`，原样复制 operands、types、attributes 与 regions；`ProgramOp` 旁边的 Axis/Range/Launch/Transfer/Contract/Buffer 等 records 保存部分决定，但 program-id、physical loops、fragment values、access operands、mask、accumulator def-use 和 storage lifetime 并未进入 executable function。shared passes 多数更新 records，Triton `ProgramForms` 又回到 KIR/KernelModel 重推 orientation、reduction axes、ragged route、gather 与 stream facts，最终 materializer 才现场生成 pointer、mask、shape、loop、workspace、grid 与 wrapper。当前真正缺的是程序载体，不是再加一层 records。

真实 Triton 两层提供的关键方法不是名称，而是 IR 不变量：

- TTIR 已是一份完整 GPU block program：program id、静态 block tensor、pointer/mask access、SCF loop/control、dot/reduce/scan 都在当前 SSA program中；TTIR passes直接替换、移动或删除 op。
- TTIR→TTGIR conversion把 distributed encoding materialize到 tensor types，并在需要时插入 `convert_layout`；coalesce、matmul acceleration、pipeline、warp specialization、TMA/TMEM 等继续通过 op/type/region rewrite形成当前程序，而不是写旁表后交给 printer 猜。
- Intent 输出 Triton DSL source后，外部 Triton仍会建立 TTIR、TTGIR并完成这些 lower transformations。因此 Intent不能复制一套 TTGIR/NVIDIA compiler；它缺的是 Triton作者本来手写、而 Triton compiler不会从逻辑算法替作者补出的 block program。

由此得到当前 compiler 的候选分层：

```text
canonical Intent KIR
    │  immutable algorithm authority
    ▼
executable GPU block program IR
    │  explicit program mapping, loops, fragment SSA,
    │  accesses/validity, buffers/lifetimes and physical structured ops
    ▼
provider-legal program IR
    │  Triton/cuTile/TileLang surface legalization and capability forms
    ▼
deterministic provider source serialization
    ▼
external provider compiler
    └─ Triton: source → TTIR → TTGIR/NVIDIA IR → LLVM/PTX
```

这里不存在把 Intent KIR 简单类比成 TTIR、再把当前 Plan 类比成 TTGIR的关系。Intent KIR 比 TTIR更高；新建的 shared GPU program应拥有一份 Triton-source/TTIR级别所需的完整 block-program内容，但不拥有 TTGIR 的 lane/warp/layout/MMA/TMA/pipeline machine representation。Triton provider IR也不是一份自造 TTIR，它只是能独立验证并一一序列化为 Triton DSL的程序。

CPU/RVV不消费这份 GPU IR。它们与 GPU 从同一 canonical KIR分叉，各自建立适合自己的 executable physical program；可复用的是 KIR semantics、shape/index/effect/dependence/reuse analyses与 transformation reasoning，不是把 GPU program-id、fragment或block IR改名后继续使用。同一作者算法在 CPU 上未必是全局最优算法，但可以被正确复用并采用不同物理实现；若需要另一算法，仍由作者提供另一 kernel/module。

下一阶段的硬不变量应当是：

1. KIR 是唯一算法权威；physical IR 可以等价改写，但每个 physical op保留可追溯 provenance与明确的语义保持条件。
2. 第一份 GPU program可以保守、可以慢，但必须完整可执行；不能有“Range record已选、pointer/mask以后由 emitter重建”之类的洞。
3. 会改变执行的决定必须成为当前 IR 的 op/type/region/operand/result/def-use。side records只可保存analysis cache、provenance或search schema，不能与 executable function共同解释程序。
4. shared passes直接改写当前 GPU program：ownership/blocking改变program mapping、loop和fragment types；replay/materialization改变def-use；access lowering产生显式access与validity；structured realization产生真实accumulator与physical reduction/contract skeleton。
5. provider conversion只消费当前 GPU program。它可以选择pointer或descriptor等真实provider surface form，但不能回到KIR、shape或role重新建立shared ownership、range、axis与validity。
6. provider-legal IR必须能独立验证，terminal translator只序列化；字符串生成阶段不得创造loop、workspace、mask、grid或tuner axis。
7. Triton已有的distributed layout、coalescing、MMA/shared/TMEM encoding、software pipeline、warp specialization、TMA lowering与winner selection继续委托Triton。

这也说明当前 Plan records的去向不能一刀切：Launch/Axis/Range中影响执行的部分要进入program mapping、loop与types；Transfer/Padding进入显式access/validity graph；Contract/Reduction/Scan进入physical SSA和structured ops；Buffer进入allocation/lifetime/use-def；只有纯derived analysis、provenance与candidate schema适合继续作为非executable记录。

当前仍需继续裁决、不能在本轮拍定的是三件事：

1. KIR→GPU block program的“完整保守初始程序”采用什么普遍合法的program mapping；不能为了容易实现就假定每个kernel都是一个输出点一个program，也不能用全kernel单program掩盖dynamic buffer与effect合法性。
2. shared GPU block program需要哪些最小types/ops，才能同时机械投影到Triton与cuTile，并允许TileLang在provider层增加显式storage/copy/sync legalization；不能先按当前三个materializer函数反推op列表。
3. provider-legal program是独立dialect，还是同一GPU dialect的严格legal subset；判据不是文件数量，而是它能否在不访问KIR和KernelModel的情况下独立verify并确定性serialization。

### 当前校正

- `gpu-compiler-space-pass-and-baseline-contract.md` 对“exec clone + side records + hidden materializer compiler”的诊断继续成立；其中以旧 `state_stream/partition/ragged` KIR为前提的清单不再是目标设计。
- 参考 Triton 的正确方式是采用“决定进入当前IR并由passes重写”的纪律，以及严格区分block-program语义与distributed/machine lowering；不是在Intent内部复制TTIR/TTGIR两套dialect。
- 当前最先需要定义的是executable GPU block program的完整不变量与保守initial conversion。未定义这一步之前，不应继续搬pass、拆materializer或增加provider attrs。
- 本轮只形成compiler阶段的起始判断并更新持续报告，不修改compiler实现或正式compiler spec。

## 第三十二轮：撤回“每个 provider 必须有独立 IR”的过度设计

### 你的输入

> “为什么 Triton 一定要抽象出自己的 IR？此前已经确定：shared program 已经拥有的 execution/value/access/validity/structured facts，Triton 与 cuTile 默认复用同一来源；不能预先把三家扩张成三套完整 leaf compiler。你凭什么又改变？”

### 我的输出

你的质疑成立。第三十一轮把两个不同要求错误地合并了：

1. source emitter 之前必须已有一份自足、可执行、可验证的程序，不能由字符串 materializer临时创造grid、loop、pointer、mask、workspace与value flow；
2. 每个provider都必须拥有一份独立provider IR。

第一项由当前失败结构直接逼出；第二项没有被证据逼出。我因为当前Triton materializer很厚，就把“消除隐藏编译器”过度推导成“建立Triton provider dialect”。但隐藏决定也可以通过完善shared GPU executable IR而消失；若Triton/cuTile只是对同一program model使用不同API spelling，随后完全可以是薄的deterministic lowering/serialization，不需要再复制一份program authority。

正确的最小架构恢复为：

```text
canonical Intent KIR
    ▼
shared executable GPU program IR
    ├─ thin Triton legalization/serialization → Triton DSL source
    ├─ thin cuTile legalization/serialization → cuTile source
    └─ 仅在真实差异逼出时增加 TileLang-local legalization
```

这里的“thin legalization”不预设独立dialect。它可以是对shared GPU IR运行的capability verifier、少量form selection与直接serializer。只有某个provider的source surface确实要求shared GPU program没有、也不应公共化的结构，而且该结构需要多步变换、独立legality和后续消费时，才由证据建立provider-local IR/ops。不能为了形式完整或对称预建三套。

因此，第三十一轮所写“provider-legal program IR”不再是架构必选层。真正必须独立成立的是shared GPU executable IR；provider边界只要求在serialization前所有决定已经闭合，并不要求决定存在于一份provider专属dialect中。

Triton TTIR/TTGIR的参考价值也重新限定：

- 借鉴的是决定进入当前IR的op/type/region/def-use、pass直接改写当前程序、legality驱动conversion这些编译器方法；
- 不复制TTIR/TTGIR层数，不在Intent中重建Triton provider IR，更不复制NVIDIA lower compiler；
- shared GPU IR应表达Triton与cuTile共同需要的block program事实；TileLang额外需要的显式storage/copy/sync必须先读真实surface与lowering，确认无法由共同表示直接序列化后再决定是否增加local legalization。

### 当前校正

- 撤回“每个provider都需要一份独立provider-legal IR”的推导。
- 保留“shared GPU executable IR必须自足、terminal source emission不得重新作决定”的硬要求。
- Triton/cuTile默认从同一shared GPU program薄映射；provider-local IR或额外pass只能由已确认的目标语义/legality差异逼出，不能预建。
- compiler阶段当前真正需要继续定义的是shared GPU executable IR，而不是Triton dialect的形状。

## 第三十三轮：共同 GPU IR、vendor 扩展与架构版本不是同一层

### 你的输入

> “还是说其实有这样的 IR 更好？Triton 面对 NVIDIA/AMD 是不是一个 TTGIR；下面是什么？SM90 之类不同版本又如何表达？”

你要求把“独立provider IR”从二元争论改成真实编译器问题：共同IR、vendor扩展与具体架构版本分别如何存在，Intent应当类比哪一层。

### 我的输出

真实 Triton 不是“每个vendor一份完整TTGIR”，也不是“所有差异只能由printer处理”。它采用共同核心加局部扩展：

```text
Triton source
    ▼
TTIR (`tt`)
    ▼
共同 TritonGPU IR (`ttg`)
    ├─ NVIDIA extension ops (`ttng`, 以及更低层 `nvg`/`nvws`)
    └─ AMD extension ops (`amdg`)
    ▼
target-specific LLVM conversion
    ├─ NVVM / PTX
    └─ ROCDL / AMDGPU
```

NVIDIA与AMD都调用同一个TTIR→TritonGPU conversion，先得到共同`ttg` program。后续pipeline在同一个current module中运行共同TTGIR passes，并按需引入/消费`ttng`或`amdg` operations；源码没有一个必须把完整program复制成“独立NVIDIA IR”或“独立AMD IR”的统一阶段。Vendor dialect只承载共同TTGIR无法正确表示、且后续passes/lowering确实需要操作的target-local constructs。

SM80、SM90、SM100、SM120也没有各自独立dialect。它们共享`ttg + ttng`：module保存`ttg.target = "cuda:<capability>"`，`TargetFeatures`从该属性得到compute capability；同一pipeline按capability启用Hopper warp specialization、Blackwell TMEM/CLC、TMA lowering或其它passes，具体ops/patterns再用feature predicates验证与lower。SM120即使与SM100同属Blackwell分支，也可以由feature predicates排除不支持的cluster、`ld.red`或reuse能力。AMD同样以`gfx arch → ISA family → feature predicates/pipeline branches`表达CDNA/RDNA差异，而不是每个gfx建立一份IR。

这把前两轮的二元说法进一步收紧：

- **必须有**一份共同、完整、唯一的shared executable GPU IR；
- **可以有**target/provider-local extension ops或小dialect，但它们扩展当前共同program，不复制完整program authority；
- **不默认有**Triton、cuTile、TileLang三套完整leaf IR；
- **不为每个SM/gfx版本建IR**，版本差异进入physical module的target capability、legality、pass pipeline与pattern selection。

Intent中还必须区分两条正交维度：

1. Triton/cuTile/TileLang是生成source所面向的provider surface；
2. NVIDIA/AMD以及SM90/SM100/gfx942等是provider下层的hardware target。

Triton一个provider本身可以继续面向NVIDIA或AMD，因此不能把“一个provider”机械类比成“Triton的一个vendor dialect”。Intent的shared GPU IR应先保存三家共同消费的program mapping、fragment SSA、access/validity、structured physical ops与dependency；provider surface若只差API spelling，直接legalize/serialize。只有例如TileLang显式storage/copy/sync，或Triton descriptor form这类结构满足下面条件时，才值得成为local extension op：

- 共同IR无法无损表达且不应公共化；
- 不只是一个终端函数名或字符串差异；
- 有多个后续passes/consumers需要检查或改写；
- 需要独立legality/verifier；
- 其选择在serialization之前必须显式存在于current program。

若只需把shared access机械写成`tl.load`或`ct.load`，没有local IR的必要；若选择descriptor后会产生descriptor value、改变access operands、约束block shape并被后续pass消费，则可以在同一program中引入一个Triton-local extension op，而不是复制整份Triton program。

不同GPU版本同理。Target capability只进入KIR之后的physical compiler；shared/provider passes可以据此产生不同program，但作者KIR不出现SM/gfx分支。对Triton路径，TTGIR以下的layout、MMA、TMA、TMEM、pipeline与ISA feature gating仍由外部Triton拥有；Intent只有在Triton DSL source必须显式表达某种surface、下层不能从普通source与config得到时，才需要provider-local表示。

### 当前校正

- 第三十二轮“默认不建独立provider IR”继续成立，但不能被解释成“target-local op/dialect永远不合法”。
- 正确目标是一个shared executable GPU program，加由真实差异逼出的局部extensions；不是三份完整leaf programs，也不是所有差异都拖到字符串printer。
- SM/gfx版本通过physical target attributes、feature queries、legality与pass/pattern selection表达，不建立architecture-specific完整IR。
- 下一步设计shared GPU IR时，应预留可扩展operation legality，但不预建Triton/cuTile/TileLang对称dialect。

## 第三十四轮：compiler 总体分层已定，但 shared GPU IR 本体尚未冻结

### 你的输入

> “可以，按照这个和自查，直接给出 `doc/compiler/` 的设计；还是你认为还有需要讨论、不清晰的地方？”

### 我的输出

当前已经足以冻结compiler的总体边界：canonical KIR是算法权威；KIR之后按target family建立独立physical program；GPU拥有一份共同、完整的executable IR；Triton/cuTile默认薄映射，TileLang等只有在真实差异逼出时增加local extensions；SM/gfx版本由physical target features与passes表达；terminal serialization不作决定；外部provider compiler继续拥有layout、machine pipeline与instruction lowering。

但这还不足以直接写正式`doc/compiler/`。如果现在只把这些原则换成目录和类名，仍然没有回答当前V2失败的根因：到底什么operation/type/region使program本身可执行。仍有五项IR本体问题必须先裁决。

#### 1. Shared GPU IR的执行模型

必须精确定义program instance、grid/virtual coordinate、program内ordered loops、unordered logical work的physical ownership、fragment/static shape与scalar value。尤其要说明多个logical outputs、不同worksets、ordered recurrence和effects怎样共存于一个kernel，而不能再次用Axis/Range records共同解释。

#### 2. 第一份完整、普遍合法的KIR→GPU program

conversion不能留下hole，但目前没有证明“每个logical point一个program”或“全kernel一个program”能覆盖全部kernel。必须确定一个对multi-output、reduce/scan/contract、dynamic subregion、logical buffer和effects都定义完整的保守baseline；后续blocking/ownership passes才有明确的可改写起点。

#### 3. KIR语义与physical executable authority的连接

需要决定physical op怎样引用而不复制第二份算法真理：哪些内容通过稳定origin/symbol reference读取，哪些combine/control region必须lower成physical region，哪些derived facts只在analysis中重算。目标是KIR仍是语义权威，同时physical program无需结合KIR side metadata才能执行和verify。

#### 4. Physical parameters与search surface怎样进入IR

block/fragment extent可以是target compile-time参数，但参数必须直接约束当前types/loops/access，而不是继续只在SearchSpaceOp或字符串里出现。需要定义parameterized physical program、candidate instantiation、legality pruning与provider autotuner之间的边界；winner仍由下层实测，但每个candidate必须对应一份完整合法program。

#### 5. 共同op与local extension的真实inventory

共同GPU IR至少要覆盖program mapping、loops、fragment SSA、access/validity、buffers/lifetimes和physical structured operations。但pointer/descriptor、explicit allocation/copy/sync等究竟是共同form还是Triton/cuTile/TileLang local extension，必须逐项读取三家source surface与lowering后裁决，不能按当前materializer函数或追求三家对称预建。

这五项不是实现细节：它们分别决定dialect边界、executable authority、pass可变换空间与provider复用程度。任何一项含糊，都可能再次产生“KIR clone + side records + hidden materializer compiler”。

### 当前校正

- 暂不创建正式`doc/compiler/`，避免把尚未定义的shared GPU IR写成稳定规格。
- 下一轮应先闭合shared GPU IR的执行模型与第一份完整initial conversion；这是其它pass、provider extension和目录结构的地基。
- 总体分层不再重开：后续讨论只解决上述IR本体，不再重新争论是否需要共同GPU IR或是否复制TTIR/TTGIR。

## 第三十五轮：闭合 shared GPU IR，本轮正式写入 compiler 规格

### 你的输入

> “这些问题不是已经有一些答案了吗？可以直接给我吗？现在设计与目标差别到底大不大？我不熟悉编译器细节，你参考 Triton 等真实实现，把旧的错误结构彻底清除后给出这个抽象。”

你要求停止把已经有证据的问题继续挂成开放讨论，也不要只给分层口号；要把 initial program、executable authority、parameter/tuner 和三家 target 边界写成可实现、可验证的正式 compiler 规格。

### 我的输出

我以已经冻结的 KIR 语义为输入，复核真实 Triton TTIR/TTGIR、cuTile 与 TileLang source surface，并用现有 ragged、scan、dynamic-programming、atomic、multi-output 和 mutable-buffer kernels 反证过于简单的 initial mapping。由此在 `doc/compiler/` 建立了正式设计：

1. KIR 保持 immutable algorithm authority。Conversion 产生一份不依赖 KIR clone 或 side records 才能执行的 shared GPU program；origin IDs只用于验证、诊断和追踪。
2. Initial conversion先由parallel/unique effects/free axes、def-use、alias和dependence形成logical worksets，再把必须共同拥有的computations/effects组成execution groups。Launch前可知的group domains组成带runtime segment length与prefix offset的disjoint program space；grid使用只能读取host-visible ABI metadata的typed `LaunchExpr`。Device data决定大小的ragged members不参与grid cardinality，而留在outer program内遍历，避免为了数成员偷偷引入第二个kernel。
3. Execution group只是同一kernel内的dispatch region，不是hidden kernel。一个`@intent.kernel` specialization仍对应一个artifact和一次launch；多kernel算法由作者定义多个kernels并由host wrapper编排。
4. Shared GPU IR显式保存program mapping、structured control与loop carries、scalar/fragment SSA、每次access的coordinates/validity/effect、structured reduce/scan/contract的完整schema、buffer allocation scope/initialization或first-write obligation/lifetime/ownership，以及atomic order与logical sharing domain。Fragment不携带lane/warp/register/shared/TMEM layout。
5. Logical buffer根据语义落为program-private、iteration-private、invocation workspace或external view。跨program非atomic dependence在保守程序里必须收回同一program；不能假设kernel-global barrier。Invocation workspace可以作为hidden ABI argument分配，但不能引入hidden launch。
6. Physical parameters是直接约束types、loops、accesses与launch的typed symbols。Intent给出合法domain并删除可证明非法的candidate；Triton/cuTile/TileLang各自的tuner编译、实测并选择winner，winner不是IR事实。
7. Triton与cuTile默认薄映射同一shared program；TileLang只为其真实要求的storage/copy/sync形式增加local legalization。Provider-local extension必须由无法无损公共化、需要独立legality和后续consumer的真实结构逼出；API spelling不建立第二份IR。Triton descriptor若成为local form，其shape/stride/block/alignment、runtime allocator requirement与lifetime也必须在serialization前进入current program。SM/gfx版本由target capability与feature predicates处理，不建立架构专属完整dialect。
8. External Triton继续拥有TTIR之后的distributed layout、coalescing、MMA/shared/TMEM、TMA lowering、software pipeline、warp specialization和LLVM/PTX。Intent不复制这部分compiler。

正式规格分为：

- `doc/compiler/kir-to-gpu.md`：workset、execution group、initial mapping、buffer/effect scope与construction verification；
- `doc/compiler/gpu-program-ir.md`：共同executable GPU IR的types、ops与不变量；
- `doc/compiler/passes-and-analyses.md`：canonical/construction/physical analyses、passes、semantic-preserving rewrites与verifiers；
- `doc/compiler/physical-parameters.md`：typed parameters、candidate legality、provider tuner与baseline合同；
- `doc/compiler/target-lowering.md`：Triton/cuTile/TileLang、hardware features、local extensions与terminal serialization边界。

### 当前校正

- 与目标规格相比，当前V2实现不是“小修几处字段”即可完成：`exec_*`同构clone、side decision records、provider回读KIR和厚materializer都不再属于最终架构。需要按新IR纵向迁移并删除旧authority，而不是给旧Plan继续补record。
- 目标并不是复制TTIR/TTGIR，也不是三套provider leaf compilers；真正新增的是Intent作者没有写、而provider compiler也不会从logical KIR替作者发明的完整GPU block program。
- Compiler总体分层和shared GPU IR表示层已经闭合。具体dialect类名、pass切分与provider extension数量留给实现从真实rewrite依赖决定，但它们不能改变这里规定的authority、可执行字段与target边界。
- 本轮只建立稳定compiler设计，没有修改现有compiler实现，也没有用实现现状反向缩窄规格。

## 维护方式

后续每一轮按“你的输入 → 我的输出 → 当前校正”追加，并同步更新对应阶段的统一判断。这里只记录重构思想和阶段结论；性能数字、测试日志和实现流水放在独立报告中。
