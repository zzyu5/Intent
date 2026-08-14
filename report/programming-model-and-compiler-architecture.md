# Intent 编程模型与编译器架构收敛审计

## 这份报告怎样得出结论

这次不把 `doc/` 当作真理。文档只记录某个时刻想要的设计；后续实现、真实 kernel 和讨论可能已经推翻它，而文档没有同步。

本报告按下面的证据顺序判断：

1. 当前 Python frontend、canonical MLIR、realizer、Plan、三个 emitter 的真实代码；
2. 现有 DSL kernel 实际采用的算法结构；
3. Triton、cuTile、TileLang 当前安装版本的真实 API；
4. 一个实际编译并运行的 Welford/state-stream repro；
5. 将来接入 RISC-V/RVV、CPU/Scalar、专用矩阵扩展时仍然成立的边界；
6. 最后才用文档解释最初意图，并标出需要修改的旧判断。

当前环境中核验的版本是：

- Triton 3.6.0；
- cuTile Python 1.5.0；
- TileLang 0.1.13。

最终目标不是让文档和代码彼此迁就，而是确定一个之后可以冻结的编程模型，再把每个偏差登记到唯一的责任层。

---

## 一、最终编程模型：不是 tile program，而是结构化逻辑区域 kernel

### 1.1 最短定义

Intent 是一门 **面向算子内部算法、target-independent、region-parametric 的结构化 kernel DSL**。

作者写：

- 一个逻辑 callable 的 ABI、输入输出、alias 与 effects；
- logical domain、region、index relation；
- 哪些工作独立、哪些有序、哪些携带状态；
- reduce、scan、contract、gather、scatter、原子与 logical buffer；
- 数值路径、dtype、identity、逻辑边界与调用前置条件；
- 一个算法包含哪些逻辑阶段以及阶段间的数据依赖。

作者不写：

- GPU program/block/thread/warp identity；
- RVV hart、`vl`、LMUL 或具体 vector register grouping；
- CPU thread id、SIMD width；
- 具体 tile/chunk/vector width；
- register/shared/stack/scratch 的具体 placement；
- 低层 layout、指令、pipeline、prefetch、unroll；
- 某门下层语言特有的 launcher 参数。

因此，Intent 的核心单位既不是 Triton program instance，也不是 CUDA tile。它是：

> **在逻辑坐标与逻辑区域上定义的、带 tensor-flow、状态和 effects 的算子内程序。**

### 1.2 为什么应叫 region-parametric，而不应把 tile 当本体

`partition(domain, extent=I.auto(...))` 的本质不是“请生成一个 GPU tile”，而是：

> 作者允许 compiler 选择一个 region extent，但 region body 的逻辑意义不随这个选择改变。

在不同 target 上，它可以兑现为：

- GPU：program/CTA 拥有的 tile；
- RVV：一次 strip-mine 的 VLA chunk；
- CPU SIMD：一个 vector chunk；
- Scalar：普通循环的一段；
- 专用矩阵扩展：喂给 microkernel 的局部区域。

去掉 source 中的 tile/worker identity，价值正是在这里：同一个逻辑索引不能因为换成 GPU program id、RVV lane 或 CPU thread 而改变；RNG counter、ragged membership、state carry 与 effect 顺序也不会绑定某种物理执行模型。

但“去掉 tile”不等于“去掉物理 mapping”。ownership、chunking、vectorization、遍历与存储仍然必须由 target realizer 决定，只是不进入算法身份。

### 1.3 作者决定 partition 是否存在，compiler 只填 extent

这是一个必须固定的边界：

- 作者写逐元素 body，body 就只看见一个逻辑元素；
- 作者显式写 partition，body 才看见一块逻辑区域；
- compiler 可以把互相独立的标量实例打包进一个物理 program 的 lanes；
- compiler 不能把逐元素 body 改写成作者没写的块矩阵算法；
- `auto` 只授权 extent/chunk realization，不授权改变 body 的可观察工作集。

所以 Intent 不是自动张量化器，也不通过识别 operand 形状把 scalar program 偷换成 block program。

### 1.4 一个 source kernel 对应一个 logical callable，不必强行等于一次机器 launch

上一版报告把“多个 target launch”直接判成违反 single-kernel，这是过度解释。

真正应冻结的不变量是：

- 一个 `@intent.kernel` 产生一个对调用方可见的 callable/ABI/effect boundary；
- 作者写下的逻辑阶段、数据依赖、数值路径与外部 effects 不被改变；
- compiler-private workspace 不进入用户 ABI；
- 物理 stage 不能改变结果、别名、外部可见顺序或调用方需要支付的额外协议；
- 多个 source kernels 之间的算法级 orchestration 仍由普通 wrapper 明确表达。

一个 logical callable 的机器实现可以是：

- 一个 GPU kernel launch；
- split reduction 的若干 GPU launches；
- 同一个 CPU/RVV function 内的多个 loop/microkernel stages；
- 一部分 target 可以 fuse，而另一部分 target 需要 materialize intermediate。

这仍是算子编译器内部的 physical realization，不是 framework graph partition。关键不是 launch 数量，而是 **是否只物化作者已经写下的数据依赖，还是 compiler 发明了新的算法阶段**。

当前 ragged contraction 的 `StageOp` 方向因此并非天然错误；真正的问题是它尚未形成完整 execution contract，后文单独分析。

---

## 二、编译器的职责：从逻辑区域到 target-family realization

### 2.1 我们不是在“推算法”

Intent compiler 不应自动做：

- stable softmax 与 online softmax 的互换；
- radix grouping 与 atomic bucket 的互换；
- ordinary GEMM 与不同数学算法的互换；
- 为了某个 target primitive 重写 source body；
- 决定 framework operator fusion/fission；
- 改变 source-visible passes、外部 workspace 或调用顺序。

这些都需要作者独有的算法意图，必须留在 source 或普通 Python wrapper。

### 2.2 我们真正求解的东西

编译器求的是一个 target 上的 physical realization：

1. logical axis 在这个 target 上承担 parallel、ordered、reduction、ragged member、lane 等哪些角色；
2. ownership、traversal、reduction、access 等用途分别用什么物理范围；
3. 哪些区域被分块、strip-mine、vectorize 或 persistent traversal；
4. logical validity 如何兑现成 loop bound、mask、fill、checked transfer 或完整块搬运；
5. 输出区域对应怎样的输入 access footprint，包括 tail 与重叠 halo；
6. logical buffer 的 lifetime、owner、residency 与必要 workspace；
7. structured primitive 是交给下层，还是由本 target realizer 明确展开一个物理 hierarchy；
8. 哪些 realization 参数可以交给下层 tuner，哪些结构选择必须先确定；
9. 一个 logical callable 是否需要多个私有 machine stages，以及它们的依赖和 intermediate lifetime。

所以“范围”确实是主骨架，但完整职责还包括 ownership、effects、validity、storage、primitive delegation 与 execution stages。

### 2.3 target 在哪里分叉

正确分叉点在 canonical Kernel IR 之后，而不是 source 语言里：

```text
Python DSL
    ↓
Canonical Kernel IR
    ↓
共享 semantic facts / provenance / legality
    ├── GPU realizer ── GPU Physical Plan
    │                    ├── Triton surface
    │                    ├── cuTile surface
    │                    └── TileLang surface
    │
    ├── RVV realizer ── RVV Physical Plan ── intrinsic C / MLIR / object
    ├── Scalar/CPU realizer ── CPU Physical Plan ── C/LLVM
    └── 其他机器 realizer
```

GPU、RVV、Scalar 是不同的 target family；Triton、cuTile、TileLang 是同一个 GPU realization 的不同 surface/provider。

### 2.4 不应强迫 GPU 与 RVV 共用一份具体 Plan

可以共享的是 Plan 的概念骨架：

- iteration/ownership；
- chunk/range；
- traversal/order；
- validity/access footprint；
- storage requirement；
- primitive realization/delegation；
- execution stage/dependency；
- tunable parameter 与合法性约束。

不能强行共享的是具体选择：

- GPU 的 3D program grid、worker axis、persistent CTA；
- RVV 的 VLA、LMUL、unroll、microtile；
- CPU 的 thread partition 与 SIMD width；
- 某专用扩展的 fragment/register rule。

当前 `intent_plan` 中的 `ProgramOp`、`program_order`、`worker_axis`、三维 grid 投影明显是 GPU Plan。未来接 RVV 时，应保留共同 Plan vocabulary，并允许 target-family extension；不能把 RVV 塞进 GPU program/worker 字段，也不能为了“统一”把 GPU 决定搬回 Kernel IR。

---

## 三、实际系统的表示与边界

### 3.1 Python Frontend

Frontend 只应维护 lowering 期间的临时状态：AST、constexpr、symbol/shape、region、source location 与临时 `ValueType`。它负责：

- 将受限 Python 语法正规化；
- 内联 kernel-local `@intent.fn`；
- 把 break/continue 变成 loop-carried control state；
- 构造 canonical Intent MLIR；
- 在作者源码位置报基础语言错误。

这些临时对象不是第二套持久化 typed Kernel IR。

### 3.2 Canonical Kernel IR

Kernel IR 是唯一算法真理，应保存：

- ABI、view access、alias、effects；
- logical domains/regions/partitions/ragged relations；
- structured control 与 carried state；
- tensor-flow、index relation、dtype；
- reduce/scan/contract 及其算法 closure；
- source precondition 与 numerical contract；
- 稳定 operation/value identity。

GPU、RVV、Scalar realizer 都只能从这里读取算法事实，不能回到 Python object 或根据 kernel 名字重新识别。

### 3.3 KernelFacts 是派生分析，不是第四份真理

当前 `KernelFacts` 保存 axis provenance、region tree、def-use、contraction、scan、ragged、stream、boundary 和 access range。它合理存在，因为这些是昂贵但可重算的分析。

它不应：

- 持久化成另一套 IR；
- 保存多个合法方案中“选了哪个”；
- 用 extent label/shape 猜 logical identity；
- 替代 Kernel IR 的基础 type/schema verifier。

### 3.4 Physical Plan 是 target-family 的已选 realization

Plan 只保存无法从 Kernel IR 唯一推出的选择，以及 lower target 必须直接消费的 binding。

它不应复制完整算法；也不能只保存模糊角色，让三个 leaf 各自再选一次。

### 3.5 Surface/leaf

一个合格 leaf 只做：

- capability check；
- Plan binding 到目标原语/语法的机械投影；
- Kernel IR op 的逐 op spelling；
- ABI、JIT、workspace materialization、runtime 接线；
- source-located unsupported。

某个 surface 的下层 primitive 更强时，可以明确 delegate；更弱时可以拒绝，或消费 Plan 中更显式的 hierarchy。不能因为 TileLang 需要写得更细，就把 TileLang 字段表变成所有 target 的共同算法模型。

---

## 四、generic combine：重新验证后的准确结论

### 4.1 Welford 能否用当前原语表达

能。

Welford/Chan 的单遍方差可以写成：

- chunk 内用现有 `I.reduce.sum` 求 count、sum、局部 M2；
- chunk 间用 `I.state_stream` 携带 `(count, mean, m2)`；
- 合并公式用普通标量算术表达。

这和 online softmax 在 chunk 内求 max/sum、chunk 间显式更新 `(maximum, denominator)` 是同一结构。因此：

> **“表达 Welford”本身不构成新增 generic combine 的理由。**

### 4.2 真实 repro 暴露了另一件事

我用 `M=64, N=257` 写了上述 state-stream Welford，并实际经过 Triton 路径编译和运行。结果：

- 编译、JIT、运行均成功；
- 输出均为有限值；
- mean 最大误差约 `0.079155`；
- variance 最大误差约 `0.586592`。

该 probe 故意使用非整除尾块。结构能编不等于 lowering 正确；当前最可疑的是 `I.full((column_region,), 1)` 形成的逻辑 count 在 physical tail 上没有按 reduction identity 正确 neutralize，但根因尚未完成定位。

这项发现应登记为 **validity/padding correctness bug**，不能拿来否定 state-stream 的表达力，也不能用 generic combine 掩盖。

### 4.3 下层真实 API 支持到哪里

Triton 3.6.0 当前 API：

```python
tl.reduce(input, axis, combine_fn, keep_dims=False)
tl.associative_scan(input, axis, combine_fn, reverse=False)
```

`input` 可以是 tuple，`combine_fn` 是 `@triton.jit` 函数。官方接口见 [Triton reduce](https://triton-lang.org/main/python-api/generated/triton.language.reduce.html)。

cuTile Python 1.5.0 当前 API：

```python
ct.reduce(x, axis, func, identity, keepdims=False)
ct.scan(x, axis, func, identity, reverse=False)
```

`x` 可以是 tuple，`func` 接收两组 0D tiles 并返回同结构结果。官方接口见 [cuTile reduce](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.reduce.html)。

TileLang 0.1.13 的高层 `T.reduce` 仍是固定 `ReduceKind`：sum/max/min/abs/bitwise 等，没有等价的任意 Python combine 参数。TileLang/TVM 的更低层有 `comm_reducer`，但它是否能在当前 eager/prim-func emission 路径中机械闭合尚未验证；不能先写成“TileLang 已支持”。其当前高层接口见 [TileLang reduce](https://www.tilelang.com/autoapi/tilelang/language/reduce_op/index.html)。

### 4.4 generic combine 是真实能力缺口，但不是算法表达力堵点

准确分类是：

- Welford 等算法可以用现有 state-stream 表达，所以不是“没有 combine 就写不出算法”；
- 但作者若选择 **一个结构化 reduction/scan，并授权下层自行选择归约树**，当前 canonical IR 只能传固定字符串，无法携带作者写的 closure；
- Triton/cuTile 已原生支持这条委托，Intent 当前把它焊死在固定 add/max/or/and；
- 因此它是 structured primitive 完整性与薄投影能力的缺口。

### 4.5 正确设计不是“把函数名字符串放开”

`combine_fn` 应成为 Kernel IR 中的 typed region，而不是 opaque Python callable 或字符串：

- region 参数是两组 accumulator scalar/record values；
- region 结果与 accumulator schema 完全相同；
- identity 按每个 component 显式给出；
- region 必须 pure，不允许 load/store/atomic/RNG state 等 effects；
- runtime capture 必须成为显式 operand；只允许 constexpr 直接捕获；
- `reduce` 的语义本身表示作者接受合法的 reassociation/tree；compiler 不证明数学结合律，但作者选择该 op 就承担这项合同；
- 需要严格顺序时使用 `ordered/state_stream`；
- source location 和 helper-inline 后的 region body必须保留。

这仍是算法 IR，不是图调度。Emitter 对 Triton/cuTile 只需把 region 机械转成 jitted function/lambda，并把 tuple state 对齐；reduction tree 继续交给下层。

工程量也不能说成“只改一个属性”：它涉及 region schema、tuple/record arity、identity、purity/effect verification、helper/capture lowering、三个 surface 的 nested function emission 与 unsupported capability。但它是边界清楚的纵向闭环，不需要自建 reduction scheduler。

### 4.6 reduce、scan、contract 不能混成一个问题

- `reduce`：Triton/cuTile 原生 generic combine，应该支持 typed closure。
- `scan`：Triton/cuTile 同样有 generic associative scan，应该与 reduce 共用 closure contract，但保留 prefix/reverse 语义。
- `contract`：`tl.dot`、`ct.mma`、TileLang GEMM 并不接受任意 semiring closure。任意 multiply/combine 会失去现成矩阵原语，不能因 reduce 支持 lambda 就一起放开。现阶段 contract 应明确支持的 semiring/dtype capability，或回到显式 pointwise + reduce source。

### 4.7 对 arg-reduce 的影响

`I.arg_reduce.max` 当前固定 lowest-index tie 与 i32 index。它解决了真实需求，但从编程模型看，它可以由 tuple-valued generic reduce 表达：输入 `(value, index)`，closure 写 maximum 与 tie-break。

因此它更适合作为：

- 方便作者使用的 library/frontend sugar；
- lower 到通用 typed reduce region；

而不是永久增加一个独立 canonical reduction 家族。generic combine 闭合前可以保留现有 node，但应把它视为过渡特化。

---

## 五、逐项重审上一版列出的“DSL 问题”

### 5.1 `partition(count=P)`：可移植的 source 语义，但当前没有真实使用

`count=P` 与 `extent=T` 不是同义词。固定 part 数会影响 part identity、partial buffer 与算法可见分片，属于作者决定；GPU、RVV、CPU 都能实现。

但当前 113 个 case 没有一个使用 `count=`，现有 split-K 也用 extent partition。它不是当前算法阻塞项。

结论：

- 不应静默改写成 extent；
- 可以保留为待闭合的 Core 语义；
- 在实现 realizer 前，frontend 应明确报“当前 target 不支持 count partition”，不能生成 IR 后在深层失败；
- 优先级低于 correctness 与公共 IR 合同。

### 5.2 非 unit-step domain：大多是可删除的 convenience

真实算法可以用 dense logical domain 加显式 affine/quasi-affine index 表达 stride：

```text
i in [0, count)
address = start + i * step
```

这样 logical iteration identity 更清楚，range/provenance 也不必同时承载方向与 stride。当前语料没有非 unit-step domain。

结论：

- 保留 runtime unit-step dense domain；
- 非 unit-step domain 先在 frontend 明确拒绝；
- 不把“完整 Python range”当语言完整性的指标；
- 真有算法证明 explicit index 不够时再重开，而不是现在扩 realizer。

### 5.3 `state_stream` runtime extent：当前不需要

现有 attention/ragged kernel 需要的是 runtime logical stop，已经由 `I.end(...)` 表达；stream chunk extent 使用 fixed 或 `I.auto`。没有 case 需要一个运行时值直接决定 physical chunk width。

RVV 的动态 `vl` 也可以是 target 对 `auto` 的 realization，不需要 source runtime extent。

结论：

- 将 source contract 限制为 fixed extent 或 `auto`；
- runtime endpoint 与 extent 分开；
- 删除“runtime extent”假能力，除非未来真实算法证明 chunk size 是 source-visible 值。

### 5.4 Logical buffer：Core 正确，当前只是 GPU capability 子集

`I.buffer` 表达 kernel-local mutable logical object，不指定 register/shared/global/stack。这个抽象对 GPU、RVV、CPU 都成立，应保留。

当前 GPU realizer 只支持一个 parallel owner 的直接 child，并把难以结构化的动态访问 spill 到 global private workspace。这是当前 target realization 的能力边界，不是 logical buffer 的完整语义。

结论：

- 不把“direct child of parallel”写成整个语言永久规则；
- shared analysis 负责 lexical lifetime、owner/effect 合法性；
- GPU/RVV/CPU realizer分别声明能兑现的 residency/共享范围；
- 跨 parallel owner 的共享若没有 barrier/atomic/order 合同，应明确拒绝。

### 5.5 `I.end`：真正的 Core 语义

`I.end(domain_or_region)` 表示逻辑区域的 exclusive endpoint，被 causal、varlen、paged、ragged stream 实际使用。它不是 physical tile end，普通 shape arithmetic 也不能替代 ragged member endpoint。

结论：保留，并正式定义 domain/region、empty region、exclusive endpoint 与 stop 的语义。

### 5.6 `I.assume_in_bounds`：真正的调用前置条件

间接索引、routing、embedding、paged cache 中，compiler 无法一般证明数据读出的 index 范围。作者必须能声明前置条件。

结论：保留，但明确：

- 违反时是调用方错误/undefined behavior；
- 它不是 compiler 自动推导结果，也不是性能 hint；
- verifier 只验证声明的类型、view、axis 与支配关系；
- target 可以用它消除 mask，也可以只用作 legality proof。

### 5.7 `I.arg_reduce.max`：真实需求，canonical 形态可在 combine 闭合后收敛

Cross entropy 与 nucleus sampling 确实需要 value+index 与确定 tie。需求是真实的；专门 canonical op 未必永久必要。处理见 4.7。

### 5.8 `I.sparse_contract_2to4`：需要 semantic anchor，但当前 API 形状过专

2:4 compressed values、metadata interpretation 与 dense RHS 是真实数据格式语义，不能假装成普通 dense contract；有 native primitive 的 target 也需要一个 semantic anchor 才能利用。

但 `sparse_contract_2to4` 直接作为 Core 名字把一个 format 焊进 API，不利于以后 RVV/CPU fallback 与其他 sparse formats。

结论：

- 保留 sparse contraction 的 canonical 概念；
- format、compressed axis、metadata schema 显式化；
- 2:4 是第一个 format descriptor，而不是一个永久独立算子家族；
- GPU target 可用 native sparse MMA，RVV/CPU 可解压+dot 或明确 capability；
- surface 无原语时提前 unsupported，不能慢路径冒充高性能支持。

### 5.9 `I.fence`：当前 public API 应删除

当前 `I.fence` 被导出，但 frontend 无条件报错；它没有 scope、ordering、参与者或 barrier 语义。这不是“暂时 backend 不支持”，而是 source operation 本身没有定义。

未来 RVV/CPU 确实可能需要 memory fence，但应从真实算法重新设计：

- memory fence 与 execution barrier 分开；
- ordering、scope、participation 明确；
- 与 atomic/effect model 一致。

在那之前，删除当前 public `I.fence` 比保留一个假入口更干净。

---

## 六、Kernel IR：上一版“全面强类型化”的判断过重

### 6.1 `AnyType` 本身不是 correctness bug

Intent op 同时处理 scalar、tensor、record、logical domain 等多态值，ODS 使用 `AnyType` 可以是合理选择。当前 Kernel verifier、`KernelFacts` 与 GPU analysis 已经检查大量 rank、dtype、shape、region schema 和 provenance。

不需要为了形式好看把所有 ODS 重写成庞大的类型层级，也不能把 padding proof、index range、axis provenance 塞进 ODS。

### 6.2 真问题一：SSA type 与 shadow metadata 可以分叉

当前 `intent.result_types`、ABI metadata 的 `type/shape` 与真实 MLIR SSA type 没有集中一致性验证。后续 analysis 有时读 metadata，emitter 又读实际 `ViewType`。错误或外部构造的 MLIR 可以让两份事实不一致。

最小修法：

- verifier 比对 function argument metadata 与真实 parameter type；
- 比对 `intent.result_types/result_shapes` 与真实 SSA result；
- 检查 access mode、rank、element dtype 与必要 attrs；
- 冗余且无人消费的 metadata 直接删除，不再维护影子真理。

### 6.3 真问题二：block argument stable ID 没进入公共 KernelModel

Frontend 为 region block arguments 生成 `intent.region_argument_nodes`，verifier 也检查其唯一性；但 `analyzeKernel()` 只把 ABI arguments 和 operation results 放入 `KernelModel.values/valueIDs`。

三个 emitter 因而绕过公共模型，自己解析 `region_argument_nodes`。

最小修法：

- `analyzeKernel()` 统一索引 block arguments；
- 验证 ID 数量、顺序、实际 type；
- 所有 consumer 使用 `getValueID()`；
- 删除三个 leaf 对 metadata 的重复解析。

这是明确、有限的公共合同修复，也直接帮助未来 RVV consumer。

### 6.4 字符串 logical spec 不必全删

symbolic extent、relation name 等本来就是符号内容，字符串可以合理存在。需要的是规范格式与 producer/type metadata 一致性，不是再造一套 Python typed IR 或复杂字符串类型系统。

---

## 七、Plan 与 emitter：逐项区分“合法读取”和“假发射”

### 7.1 Region argument → selected range：Plan 确实少一个 binding

三个 leaf 都从 Kernel IR region 名字重新决定：

- `parallel` argument 使用 ownership range；
- `state_stream` argument 使用 traversal range。

Region argument 身份属于 Kernel IR；但“这个 argument 消费哪个已经选定的 range”是 physical binding。

结论：Plan 应显式绑定 stable region-argument ID → axis/range。Leaf 只读取，不再按 op 名选择。

### 7.2 Row-vector physical extent：决定已经有一半，最终 binding 仍缺

Realizer 已创建 `BlockExtentOp(rounding=power_of_two, fill=...)`，但 leaf 又从 domain → ABI dimension → BlockExtent 查找最终 physical extent。

结论：

- ABI symbol 怎样拼成 Python/C 表达式，仍是 leaf 工作；
- axis/range 绑定哪个 logical/block extent，应在 Plan 明确；
- Triton `next_power_of_2`、RVV `vl` 等 target spelling 留在各自 leaf。

### 7.3 Ragged/stream use-def：主要是共享 semantic analysis，不应全复制进 Plan

Ragged outer/member、stream stop 是算法结构，应从 Kernel IR 得到；把它们全复制进 Plan 会形成第二份算法真理。

正确拆分：

- shared analysis 一次建立 canonical ragged/stream binding；
- Plan 只保存 relation/stream → selected physical axis/range 的选择；
- leaf 消费 shared semantic binding + Plan physical binding。

### 7.4 `dimensionName`：一半合法，一半危险

- 把 ABI dynamic shape 变成目标语言中的参数表达式，是合法 leaf spelling；
- 用相同 shape label/extent 反猜 logical axis identity，是 provenance bug。

当前 `axisFromLabel` 找不到精确 provenance 时会选择第一个同 extent domain，甚至构造 implicit axis。这个 fallback 应删除：沿 SSA、region argument、index relation无法唯一解析时直接诊断。

### 7.5 persistent heuristic：是 GPU policy，不是 emitter 假发射

当前 persistent bool 由 contraction/parallel/tiled/ragged 结构规则选择，并已写进 `ProgramOp`。Leaf 没有重新决定。

问题只在于：这条规则是否对不同 GPU 都好。现有跨设备证据还没有证明它必须进入 search space。

结论：

- 保留为 GPU realizer policy；
- 不推广到 RVV/CPU；
- 出现一条规则无法兼顾设备的实测证据时，再把这一维变成候选；
- 不提前建搜索框架。

### 7.6 SearchSpace 只有参数名：目前不是已证实 bug

当前 Plan 只告诉下层“哪些参数可调”，具体候选值、排序和 winner 由 provider tuner 决定。这符合我们不复制下层知识的原则。

只有当某个合法性约束依赖 Intent 独有算法结构，而下层无法自行筛掉时，Plan 才必须增加参数域/耦合/资源上界。

结论：不因为 schema 看起来薄就补 candidate list。先保留 names-only delegation；用真实失败决定是否增加约束。

---

## 八、当前 StageOp：允许多 machine stages，但 execution contract 未闭合

### 8.1 当前真实行为

`contractionStages()` 会找到带 ragged member、最终到达 scatter 的 contractions，沿 def-use 收集 operation slice，并在 Plan 里记录 inputs、outputs、operations、terminals。

三个 emitter 都会生成多个私有 target kernels，在同一个生成 wrapper 中按顺序 launch；intermediate workspace 由 wrapper 私下分配，调用方仍只看见一个 callable。

### 8.2 它为什么可以属于 physical realization

该拆分：

- 没改变 source ABI；
- 没增加用户可见 output；
- 没改变 contraction/scatter 的 logical dataflow；
- 只 materialize compiler-private intermediate；
- 对 CPU/RVV 可以变成一个 function 内的多个 loop/microkernel stages。

因此，多 launch 本身不应被禁止。

### 8.3 真正缺少的合同

当前 StageOp 没完整表达：

- stage dependency/拓扑顺序；
- intermediate buffer lifetime 与读写 owner；
- memory visibility/synchronization；
- 是否允许 fuse；
- target 是否可以选择不同 stage grouping。

GPU 路径目前依赖同一 CUDA stream 的隐含顺序；cuTile 显式取得当前 stream，Triton/TileLang 也依赖当前 stream launch order。这使当前路径能工作，却不是可移植 Plan 语义。

结论：

- 将 single-kernel invariant 改写成 single logical callable；
- Plan 的 execution stage 必须显式可验证；
- stage grouping 是 target realization，不能按 kernel 名；
- wrapper-visible 多 kernel orchestration仍由作者负责；
- compiler-private stage 只允许在同一 callable 内、对 ABI/effects 不可见。

---

## 九、未来 RISC-V/RVV 后端对当前架构的约束

### 9.1 Kernel IR 必须保持 target-independent

RVV backend 应直接消费同一份：

- logical axes/regions；
- typed values、index relation、alias/effects；
- parallel/ordered/state_stream；
- reduce/scan/contract；
- ragged、logical buffer、validity；
- generic combine region；
- source ABI 与 logical callable contract。

不能让 GPU `program_id`、three-dimensional grid、shared fragment、TileLang buffer scope 进入 Kernel IR。

### 9.2 RVV realizer 做什么

它应选择：

- 哪些 logical axes 由外层 runtime/worker 拥有；
- 哪些 axis strip-mine 为 VLA loop；
- vector width/VL policy、LMUL、unroll；
- reduction/scan 的 vector hierarchy；
- contraction 是 RVV FMA、dot/microkernel 还是专用矩阵扩展；
- logical buffer 落 vector register、stack/local scratch；
- tail policy、mask 与 scalar fallback；
- target-local search/candidate。

这些是 RVV Physical Plan，不是 GPU Plan 的另一种 spelling。

### 9.3 RVV emitter 同样不能成为第二个编译器

Emitter 只消费 typed decisions：

- 不能看 kernel 名选择实现；
- 不能按 op 数量、邻近模式、whole-region shape 猜 VLA/IME；
- 不能重新决定 LMUL、microtile、residency；
- 没有 target decision 就明确 unsupported。

### 9.4 generic combine 在 RVV 上仍然成立

Combine region 是算法 closure；RVV realizer可以选择 scalar fold、vector tree 或 scratch hierarchy，并把 closure body内联到选定结构。它不是 GPU lambda 特例。

但 RVV 是否能高效 vectorize某个 closure是 target capability；不能因为 Triton/cuTile 能委托就宣称 RVV 自动高效支持。

---

## 十、冻结前的问题登记表

下面只登记已经有具体证据的问题，不把“可能更漂亮”列成任务。

### A. Correctness 与唯一语义合同

| 问题 | 责任层 | 已有证据 | 闭合标准 |
|---|---|---|---|
| Welford/state-stream 在非整除尾块数值错误 | GPU validity/padding realization | `N=257` 真实 repro，mean/variance 显著误差 | 定位具体 value validity 丢失点；同一 source 尾块数值正确，不加 Welford 特判 |
| SSA type 与 ABI/result metadata 可分叉 | Kernel IR verifier | verifier 只检查 metadata 形状/非空，analysis/emitter 读取不同来源 | canonical boundary 统一验证或删除冗余 metadata |
| Region block argument ID 不进入 KernelModel | Common Kernel analysis | builder/verifier有 ID，三个 leaf 重读 metadata | 所有 block argument 可由 `getValueID()` 查询，leaf 删除重复解析 |
| extent-label/implicit axis fallback | Shared semantic analysis | `axisFromLabel` 可按同 extent 猜第一个 domain | provenance 只能来自 SSA/region/index relation；不唯一即诊断 |

### B. Physical Plan 与 emission 边界

| 问题 | 责任层 | 闭合标准 |
|---|---|---|
| Region argument 未直接绑定 selected range | Physical Plan | Plan 保存 argument→axis/range；三个 leaf 不再按 region 名重选 |
| Row-vector final extent binding不完整 | Physical Plan | Plan 明确 axis/range→logical/block extent；leaf 只拼 target syntax |
| Ragged/stream binding在 SurfacePlan 重建 | Common semantic index + Plan | 算法 use-def共享分析一次；Plan只保存 selected physical relation |
| Stage execution 只靠数组顺序和 CUDA stream | Target-family execution Plan | dependency、workspace lifetime、visibility可验证；GPU/RVV均有明确投影 |

### C. 语言表面收敛

| 项目 | 决定 |
|---|---|
| generic reduce/scan combine | 应进入 typed Kernel IR region；Triton/cuTile机械委托，TileLang按真实能力支持或拒绝，RVV由target realizer兑现 |
| arbitrary contract multiply/combine | 不随 reduce 一起放开；按目标矩阵原语 capability定义 |
| `arg_reduce.max` | 真实需求；generic combine闭合后收敛为 sugar/通用 tuple reduction |
| `partition(count)` | 语义合理但当前未使用；实现前 target-aware 提前拒绝，不列 correctness blocker |
| 非 unit-step domain | 当前拒绝；使用 dense domain + 显式 index relation |
| runtime `state_stream` extent | 当前移除假能力；保留 fixed/auto extent + runtime stop |
| logical buffer | 保留 portable Core；owner-private 是当前 GPU capability，不是永久语言定义 |
| `end` | 保留并补正式语义 |
| `assume_in_bounds` | 保留为 unsafe source precondition |
| sparse 2:4 | 收敛为 sparse contract + format descriptor，而非继续扩专用算子名字 |
| `fence` | 删除当前无语义 public API；未来从真实 memory model需求重新设计 |

### D. 不是当前任务的问题

以下事项没有证据要求现在修改：

- 全面删除 ODS `AnyType`；
- 强制一个 logical callable 只有一次机器 launch；
- 为 search space 自建候选值/排序/cost model；
- 为所有 target 统一 GPU `ProgramOp/worker_axis`；
- 为了让 TileLang 跟上，把其显式 buffer/layout 字段抬进 Kernel IR；
- 没有真实需求时实现一般 strided domain 或 runtime stream tile。

---

## 十一、以后遇到问题的固定提问顺序

### 第一步：作者是否已经写下了

- 写下了但丢失：修 frontend/Kernel IR/provenance；
- 没写且属于算法：考虑 DSL/Core；
- 没写且属于机器：进入 target realizer。

### 第二步：下层是否已经原生支持

- Triton/cuTile generic reduce/scan：传 typed closure，不自建树；
- TileLang 固定 reduction：能力检查或显式 target realization；
- RVV intrinsic/microkernel：使用其现成 primitive，不复制 instruction selection。

### 第三步：这是算法语义、派生事实、已选决定还是 spelling

| 性质 | 唯一归属 |
|---|---|
| 算法语义、closure、logical index/effect | Kernel IR |
| 可从 Kernel IR 重算的 provenance/use-def | Shared facts |
| 多个合法物理方案中选了哪个 | Target-family Physical Plan |
| 目标 API 与语法 | Leaf/emitter |
| layout/register/instruction/tuner winner | 下层 compiler |

### 第四步：跨 target 后是否仍成立

若一个“Core”字段只能用 GPU program/warp/shared memory解释，它大概率放错层；若一个 physical decision 在 GPU 与 RVV 应不同，它应在 target realizer 分叉，而不是 source 分叉。

---

## 最终判断

Intent 的稳定核心不是“跨三种 GPU tile 语言”，而是：

> **用一份 target-independent 的结构化逻辑区域算法，驱动多个算子级 target realizer；每个 realizer产生自己的 Physical Plan，surface/emitter只投影，下层 compiler继续完成其擅长的布局、指令与调优。**

当前 GPU 主线已经证明这不是 rowwise/softmax 特化，但还不能立即冻结。需要先处理登记表中的四类具体问题：tail correctness、Kernel IR 唯一合同、Plan binding、stage execution contract；同时把 generic reduce/scan closure 和几个假 public capability定下来。

完成这些之后，后续新增 GPU provider 或 RISC-V/RVV backend 都不应再改编程模型：只新增 target-family realizer、Physical Plan extension、capability 与机械 emission。新的 DSL 构造只有在真实算法无法用现有 Core表达时才允许进入。
