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

最终目标不是让文档和代码彼此迁就，而是确定一套可以冻结的编程模型，再把每个偏差登记到唯一的责任层。本轮已经按这份报告完成最后两项承重能力，并以代码与跨设备运行结果反向修正文档。

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

### 1.2 真正的分界主要在分配层，而不是张量代数层

可以把一个算子内程序概念性地拆成两部分：

- **分配/ownership 层**：当前执行实例拿到哪一部分逻辑工作与数据；
- **算法层**：拿到这些值后执行什么 exp、broadcast、reduce、scan、contract、状态更新与 effects。

这不是要求实现中增加两层 IR，只是用来定位双方差别。在算法层，Intent 与 Triton、cuTile、TileLang 都提供张量计算原语；不能笼统说 Intent 比它们“更张量”。它们在控制、effects、closure 与结构化原语的具体表达上仍有差异，但最关键的抽象分界确实在前一层。

Triton、cuTile、TileLang 的 source 通常已经拿到了一个与具体执行结构绑定的数据块，再在块上写算法。三者的 tile 合同并不完全相同，不能压成一句统一定义；以当前 cuTile 为最清楚的例子：官方模型中的 tile 是单个 tile block 内的多维 value，shape/dtype 编译期已知，每个维度要求 2 的幂，而 block 内的线程映射由 compiler 管理。也就是说，cuTile 区分了“数据单位 tile”和“执行单位 block”，却仍然把二者通过 block-local ownership 与静态 shape 绑定在一起。参见 [CUDA tile programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html)。

Intent 在同一位置给出的不是 tile，而是 **region**：domain 上的一个逻辑子集。Region 本身不携带 program/block/hart/thread owner，也不规定 register/shared/vector register。当前 frontend 实际可以生成类似：

```text
intent.result_shapes = [["?region_13_0"]]
```

这样的动态 symbolic region shape；它在 Kernel IR 中不是编译期常量，也没有 2 的幂要求。

这里也要避免反向绝对化：

- 不是每个 Intent region 都运行时动态，fixed partition 与静态 domain 当然可以产生静态 region；
- “没有 physical owner”只描述算法 IR，realizer 最终必须给它选择 owner；
- 当前 GPU target 为了生成 Triton/cuTile/TileLang，仍会把 region 兑现成编译期 tile、lane range 或其他 surface 能表达的形态。

所以“Intent 抽掉 tile”最准确的含义是：

> **tile 不再是 source/Kernel IR 中的算法身份；它被推迟为 target realization 给 region 的一种物理答案。**

它不是“更硬件无关的 tile”。从 tile 中去掉静态 shape 与执行单元 ownership 后，剩下的是另一种对象——logical region。到了 GPU Plan 和生成源码，tile 会被重新引入；到了 RVV，它可能变成 VLA chunk；到了 Scalar target，它可能只是 loop interval。

这项设计只保证未来 target 不必从已经拍扁成 GPU program/tile 的代码中逆向恢复 logical workset，并不保证 RVV/CPU 自动高效。能否高效，仍取决于对应 realizer 是否正确解决 ownership、chunk、storage 与 primitive realization；Kernel IR 干净只是必要条件，不是性能充分条件。

`partition(domain, extent=I.auto(...))` 因而表示：

> 作者允许 compiler 选择一个 region extent，但 region body 的逻辑意义不随这个选择改变。

在不同 target 上，它可以兑现为 GPU tile、RVV VLA chunk、CPU SIMD chunk、Scalar loop segment 或专用矩阵 microkernel region。去掉的是 target-specific identity，不是 physical mapping 这项编译器责任。

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

Ragged contraction 的 `StageOp` 方向因此并非天然错误；真正需要的是一份完整 execution contract。本轮已将这份合同闭合到 Physical Plan、verifier 与三个 surface 的公共 preflight，第八节给出最终语义。

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
- 内联普通 kernel-local `@intent.fn`；当 helper 被用作 structured combiner 时，将其保存为 typed、effect-free 的 Kernel IR helper body；
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

该 probe 故意使用非整除尾块。结构能编不等于 lowering 正确；修复前的嫌疑是 `I.full((column_region,), 1)` 形成的逻辑 count 在 physical tail 上没有按 reduction identity 正确 neutralize。

这项发现应登记为 **validity/padding correctness bug**，不能拿来否定 state-stream 的表达力，也不能用 generic combine 掩盖。

本轮定位确认：Plan 已经为这个 `full` 结果给出 reduction identity padding，丢失发生在三个 `emitFull` 没有消费该绑定。修复后同一 `M=64, N=257` source 的 mean 最大误差为 `1.49e-8`、variance 最大误差为 `2.38e-7`；修法是所有 region-shaped `full` 统一消费 Plan padding，不含 Welford 特判。

generic combine 闭合后，又用 tuple-valued Welford reduction 对同一非整除形态做了独立验证。第一次运行暴露出另一处同类丢失：reduction 要求的是 record field 的 identity padding，Plan 却把 binding 留在 `extract` 别名上，leaf 最终发射的是底层 field；Triton 同时缺少 domain-result value ID 到已选 physical tile 的机械索引。修复后 padding 在共享 Plan 构造中规范化到真正 materialized field，Triton只补读取 Plan 的 shape spelling。5090 与 H100 上 Triton/cuTile 的 mean 最大误差约 `3e-8`、variance 最大误差 `2.38e-7`。这再次说明问题不是 Welford 或 generic closure 特殊，而是物理 binding 必须落在真正被发射的 SSA value 上。

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

### 4.4 generic combine 已闭合，但它仍不是算法表达力的唯一入口

准确分类是：

- Welford 等算法仍然可以用 state-stream 表达，所以 generic combine 不是唯一写法；
- 作者若选择 **一个结构化 reduction/scan，并授权合法 reassociation**，现在可以把 typed combiner 随 Kernel IR 一起交给 compiler；
- Triton/cuTile 原生接受该委托，Intent 不再把 structured primitive 焊死在固定 add/max/or/and；
- TileLang 0.1.13 的当前 PrimFunc surface 没有可机械承接任意 closure 的入口，因此在 source emission 前明确 unsupported，而不是生成串行慢路径。

### 4.5 正确设计不是“把函数名字符串放开”

`combine_fn` 现在以 Kernel IR 中带 `intent.role = "combiner"` 的 typed helper body 表示。它在语义上就是 reduction/scan 的 closure region，不是 opaque Python callable 或函数名字符串：

- region 参数是两组 accumulator scalar/record values；
- region 结果与 accumulator schema 完全相同；
- identity 按每个 component 显式给出；
- region 必须 pure，不允许 load/store/atomic/RNG state 等 effects；
- runtime capture 必须成为显式 operand；只允许 constexpr 直接捕获；
- `reduce` 的语义本身表示作者接受合法的 reassociation/tree；compiler 不证明数学结合律，但作者选择该 op 就承担这项合同；
- 需要严格顺序时使用 `ordered/state_stream`；
- source location 与 lowering 后的 helper body必须保留。

Canonical verifier 集中检查 helper ABI、component/result schema、identity dtype、capture operand、return schema 与 purity；load/store/atomic/RNG 等 effectful operation 不能进入 combiner。Runtime capture 必须作为 reduction/scan 的显式 scalar operand，constexpr 才能直接进入 helper body。

Emitter 对 Triton/cuTile 将 helper 机械转成 `@triton.jit` function 或 cuTile function。由于 cuTile 的 identity 必须是常量，runtime capture 在目标投影中携带为 `(capture, valid)` 附加 component：真实 lane 写入 capture，identity lane 的 `valid=false`，combiner只选择有效 capture。这个 carrier 不改变作者的 accumulator schema，也不参与数学求解；它只是把同一个显式 operand适配到下层 tuple ABI。Reduction tree、scan hierarchy 与低层 collective 继续由下层决定。

这项能力涉及 helper schema、tuple/record arity、component-wise identity、purity/effect verification、显式 capture、三个 surface 的 function emission 与 capability rejection；它已经形成一条边界清楚的纵向闭环，没有自建 reduction scheduler。

### 4.6 reduce、scan、contract 不能混成一个问题

- `reduce`：Triton/cuTile 已机械承接 typed closure；固定内建 combiner仍走目标原语。
- `scan`：Triton/cuTile 与 reduce 共用 closure contract，并保留 inclusive prefix 语义；长轴实现为 block-local native scan 加 block 间 typed scalar carry。TileLang 只承接当前固定 combiner能力，generic closure 明确拒绝。
- `contract`：仍只表示当前矩阵原语支持的 multiply/add contraction 与 dtype/accumulator 组合。`tl.dot`、`ct.mma`、TileLang GEMM 不接受任意 semiring closure；作者需要其他 semiring 时必须显式写 pointwise + reduce，不能因 reduce 支持 closure 就在 contract 上开一个假通道。

### 4.7 对 arg-reduce 的影响

`I.arg_reduce.max` 固定 lowest-index tie 与 i32 index。它现在由 frontend lower 成 tuple-valued generic `intent.reduce`：输入 `(value, index)`，typed closure 表达 maximum 与 tie-break。

因此它的正式定位是：

- 方便作者使用的 library/frontend sugar；
- lower 到通用 typed reduce closure；

而不是独立 canonical reduction 家族。Kernel IR 中的 helper 是语义权威；`combine_builtin = argmax_lowest` 只允许 target 选择数值等价的原生 `max_with_index` spelling，不能替代 closure 语义。替换旧路径前已对 cross entropy 的 value/index/tie 结果做数值对照，三个 target 的现行路径均通过。

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

真实算法可以用 unit-step logical domain 加显式 affine/quasi-affine index 表达 stride：

```text
i in [0, count)
address = start + i * step
```

这样 logical iteration identity 更清楚，range/provenance 也不必同时承载方向与 stride。当前语料没有非 unit-step domain。

结论：

- 保留 runtime unit-step logical domain；
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

### 5.7 `I.arg_reduce.max`：真实需求，canonical 形态已收敛

Cross entropy 与 nucleus sampling 确实需要 value+index 与确定 tie。Public sugar 保留，独立 canonical op 已删除；当前统一 lower 到 typed tuple reduction，处理见 4.7。

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

修复前 `I.fence` 被导出，但 frontend 无条件报错；它没有 scope、ordering、参与者或 barrier 语义。这不是“暂时 backend 不支持”，而是 source operation 本身没有定义。

未来 RVV/CPU 确实可能需要 memory fence，但应从真实算法重新设计：

- memory fence 与 execution barrier 分开；
- ordering、scope、participation 明确；
- 与 atomic/effect model 一致。

本轮已经删除该 public API；未来若有真实 memory-model 需求，再从完整合同重新设计。

---

## 六、Kernel IR：上一版“全面强类型化”的判断过重

### 6.1 `AnyType` 本身不是 correctness bug

Intent op 同时处理 scalar、tensor、record、logical domain 等多态值，ODS 使用 `AnyType` 可以是合理选择。当前 Kernel verifier、`KernelFacts` 与 GPU analysis 已经检查大量 rank、dtype、shape、region schema 和 provenance。

不需要为了形式好看把所有 ODS 重写成庞大的类型层级，也不能把 padding proof、index range、axis provenance 塞进 ODS。

### 6.2 真问题一：SSA type 与 shadow metadata 可以分叉

修复前 `intent.result_types`、ABI metadata 的 `type/shape` 与真实 MLIR SSA type 没有集中一致性验证。后续 analysis 有时读 metadata，emitter 又读实际 `ViewType`。错误或外部构造的 MLIR 可以让两份事实不一致。

最小修法：

- verifier 比对 function argument metadata 与真实 parameter type；
- 比对 `intent.result_types/result_shapes` 与真实 SSA result；
- 检查 access mode、rank、element dtype 与必要 attrs；
- 冗余且无人消费的 metadata 直接删除，不再维护影子真理。

本轮按该最小修法闭合：canonical verifier 统一核对 function ABI、operation result metadata 与真实 SSA type/rank/shape/access mode。

### 6.3 真问题二：block argument stable ID 没进入公共 KernelModel

修复前 frontend 为 region block arguments 生成 `intent.region_argument_nodes`，verifier 也检查其唯一性；但 `analyzeKernel()` 只把 ABI arguments 和 operation results 放入 `KernelModel.values/valueIDs`。

三个 emitter 因而绕过公共模型，自己解析 `region_argument_nodes`。

最小修法：

- `analyzeKernel()` 统一索引 block arguments；
- 验证 ID 数量、顺序、实际 type；
- 所有 consumer 使用 `getValueID()`；
- 删除三个 leaf 对 metadata 的重复解析。

这是明确、有限的公共合同修复，也直接帮助未来 RVV consumer。

本轮已经按此闭合：nested-region block argument 与 ABI/result value 共用 KernelModel value-ID 索引，三个 leaf 的重复 metadata 解析已删除。

### 6.4 字符串 logical spec 不必全删

symbolic extent、relation name 等本来就是符号内容，字符串可以合理存在。需要的是规范格式与 producer/type metadata 一致性，不是再造一套 Python typed IR 或复杂字符串类型系统。

---

## 七、Plan 与 emitter：逐项区分“合法读取”和“假发射”

### 7.1 Region argument → selected range：修复前 Plan 少一个 binding

三个 leaf 都从 Kernel IR region 名字重新决定：

- `parallel` argument 使用 ownership range；
- `state_stream` argument 使用 traversal range。

Region argument 身份属于 Kernel IR；但“这个 argument 消费哪个已经选定的 range”是 physical binding。

结论：Plan 应显式绑定 stable region-argument ID → axis/range。Leaf 只读取，不再按 op 名选择。

本轮已增加可验证的 region argument → axis/range purpose/level binding，三个 leaf 只读该绑定。

### 7.2 Row-vector physical extent：修复前最终 binding 仍缺

Realizer 已创建 `BlockExtentOp(rounding=power_of_two, fill=...)`，但 leaf 又从 domain → ABI dimension → BlockExtent 查找最终 physical extent。

结论：

- ABI symbol 怎样拼成 Python/C 表达式，仍是 leaf 工作；
- axis/range 绑定哪个 logical/block extent，应在 Plan 明确；
- Triton `next_power_of_2`、RVV `vl` 等 target spelling 留在各自 leaf。

本轮 range 已同时携带 canonical logical extent 与 selected tile；row-vector 与 stream consumer 从该 range 读取上界，leaf 只保留 target spelling。

### 7.3 Ragged/stream use-def：主要是共享 semantic analysis，不应全复制进 Plan

Ragged outer/member、stream stop 是算法结构，应从 Kernel IR 得到；把它们全复制进 Plan 会形成第二份算法真理。

正确拆分：

- shared analysis 一次建立 canonical ragged/stream binding；
- Plan 只保存 relation/stream → selected physical axis/range 的选择；
- leaf 消费 shared semantic binding + Plan physical binding。

本轮 KernelModel 已一次建立 ragged/state-stream semantic index，Plan 只记录 selected stream axis/range/relation；SurfacePlan 不再遍历周围 operation 重建 use-def。

### 7.4 `dimensionName`：一半合法，一半危险

- 把 ABI dynamic shape 变成目标语言中的参数表达式，是合法 leaf spelling；
- 用相同 shape label/extent 反猜 logical axis identity，是 provenance bug。

修复前 `axisFromLabel` 找不到精确 provenance 时会选择第一个同 extent domain，甚至构造 implicit axis。本轮已删除该 fallback：provenance 只能沿 SSA、ABI shape symbol、region argument 与 index relation 得到，无法唯一解析时直接诊断。

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

## 八、Stage execution：多 machine stages 已有显式合同

### 8.1 当前表示

GPU realizer 沿 canonical def-use 形成 physical stage operation slice，并为每个 stage 记录 dependencies、inputs、outputs、operations、terminals、synchronization、fusion 与 grouping policy。

每个 intermediate 另有唯一 `StageBufferOp`，显式保存 value、唯一 producer、consumer stages、owner roles、single-writer/read-only-consumer access、`producer_to_last_consumer` lifetime 与 visibility。三个 emitter 仍可生成多个私有 target kernels，但调用方只看见一个 logical callable。

### 8.2 它为什么可以属于 physical realization

该拆分：

- 没改变 source ABI；
- 没增加用户可见 output；
- 没改变 contraction/scatter 的 logical dataflow；
- 只 materialize compiler-private intermediate；
- 对 CPU/RVV 可以变成一个 function 内的多个 loop/microkernel stages。

因此，多 launch 本身不应被禁止。

### 8.3 可验证的 execution contract

Plan 与公共 emission preflight 现在验证：

- dependency 必须引用拓扑上更早的 stage，并且精确等于所有 input buffer 的 producer；
- 每个 intermediate 只有一个 writer，consumer 必须是后继 stage，owner role 必须存在于 producer stage；
- intermediate 至少活到最后一个 consumer，当前 visibility/synchronization 合同为 `same_stream`；
- intermediate stage 必须产出 buffer，final stage 必须拥有唯一 terminal，final stage不能再成为后继依赖；
- 当前 GPU realization 的 fusion policy 为 `forbidden`、grouping policy 为 `fixed_operation_slice`，三个 surface 在 emission 前拒绝自己不能兑现的其他 policy。

同一 operation 可以出现在多个 stage slice 中，但这表示 Plan 明确选择的 pure recomputation；effectful terminal 与 intermediate writer不能重复。实际 grouping 由每个 target-family realizer产生的 stage operation slice表示，surface leaf 无权重新分组。未来 RVV/CPU 可以产生不同 grouping 的 Plan；不需要在 GPU leaf 中再放一份判断。

当前三个 GPU surface 都消费 `same_stream + forbidden + fixed_operation_slice` 合同。cuTile 显式把同一 current stream 传给每次 launch；Triton/TileLang 的 runtime launch同样提交到 current stream。CUDA 的同 stream happens-before 与可见性不再是未登记假设，而是 Plan 要求、preflight 检查和 runtime 投影共同兑现的语义。

冻结结论：

- 将 single-kernel invariant 改写成 single logical callable；
- Plan 的 execution stage 已显式可验证；
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

| 问题 | 责任层 | 已有证据 | 当前状态 |
|---|---|---|---|
| Welford/state-stream 在非整除尾块数值错误 | GPU validity/padding realization | `N=257` 真实 repro，mean/variance 显著误差 | 已闭合：根因是 region-shaped `full` 没有消费 Plan padding；三叶子统一按 reduction identity 中和物理尾块，无 Welford 特判 |
| SSA type 与 ABI/result metadata 可分叉 | Kernel IR verifier | verifier 只检查 metadata 形状/非空，analysis/emitter 读取不同来源 | 已闭合：canonical verifier 集中核对 function ABI、operation result metadata 与真实 SSA type/rank/shape/access mode |
| Region block argument ID 不进入 KernelModel | Common Kernel analysis | builder/verifier有 ID，三个 leaf 重读 metadata | 已闭合：nested-region block argument 进入公共 value-ID index，leaf 不再解析底层 metadata |
| extent-label/implicit axis fallback | Shared semantic analysis | `axisFromLabel` 可按同 extent 猜第一个 domain | 已闭合：删除同 extent 猜轴；provenance 只来自 SSA、ABI shape symbol、region 与 index relation，缺失即诊断 |

### B. Physical Plan 与 emission 边界

| 问题 | 责任层 | 当前状态 |
|---|---|---|
| Region argument 未直接绑定 selected range | Physical Plan | 已闭合：Plan 显式保存 argument→axis/range purpose/level，三个 leaf 只消费绑定 |
| Row-vector final extent binding不完整 | Physical Plan | 已闭合：range 同时保存 logical extent 与已选 tile，leaf 只做目标符号拼写 |
| Ragged/stream binding在 SurfacePlan 重建 | Common semantic index + Plan | 已闭合：算法 relation/use-def 在 KernelModel 建一次；Plan 只保存已选 stream axis/range/relation |
| Stage execution 只靠数组顺序和 CUDA stream | Target-family execution Plan | 已闭合：dependency、topology、intermediate producer/consumer/owner/lifetime/visibility、sync、fusion 与 grouping policy 都进入 Plan；三个 GPU surface 只消费 `same_stream + forbidden + fixed_operation_slice` 合同 |

### C. 语言表面收敛

| 项目 | 决定 |
|---|---|
| generic reduce/scan combine | 已闭合：typed helper/closure、component identity、purity、显式 capture 进入 canonical Kernel IR；Triton/cuTile机械委托，TileLang 0.1.13 明确 unsupported |
| arbitrary contract multiply/combine | 正式不随 reduce 一起放开；当前 contract 是目标矩阵原语支持的 multiply/add 与 dtype capability，其他 semiring显式写 pointwise + reduce |
| `arg_reduce.max` | 已收敛为 frontend sugar + tuple-valued generic reduce；target 原生 argmax 只是经验证的等价 spelling hint，不是第二份语义 |
| `partition(count)` | 语义合理但当前 realizer 不支持；frontend 在构造 IR 前明确拒绝，不列 correctness blocker |
| 非 unit-step domain | frontend 明确拒绝；当前使用 unit-step logical domain + 显式 index relation |
| runtime `state_stream` extent | frontend 明确拒绝；保留 compile-time fixed/auto extent + runtime logical stop |
| logical buffer | 保留 portable Core；owner-private 是当前 GPU capability，不是永久语言定义 |
| `end` | 保留并补正式语义 |
| `assume_in_bounds` | 保留为 unsafe source precondition |
| sparse 2:4 | 真实需求但当前形态是过渡；以后收敛为 sparse contract + format descriptor，本轮不改 API |
| `fence` | 无 scope/ordering/participant 合同的 public API 已删除；未来只从真实 memory model 需求重新设计 |

### D. 不是当前任务的问题

以下事项没有证据要求现在修改：

- 全面删除 ODS `AnyType`；
- 强制一个 logical callable 只有一次机器 launch；
- 为 search space 自建候选值/排序/cost model；
- 为所有 target 统一 GPU `ProgramOp/worker_axis`；
- 为了让 TileLang 跟上，把其显式 buffer/layout 字段抬进 Kernel IR；
- 没有真实需求时实现一般 strided domain 或 runtime stream tile。

### E. 冻结前定向验证

本轮没有跑全量矩阵，只运行直接消费新合同的入口；一次性 probe 位于 `/tmp`，验证后删除，不进入语料或测试设施。

| 能力 | 5090 | H100 |
|---|---|---|
| generic record reduce/scan + runtime scalar capture，Triton/cuTile | 通过；长轴 `N=4093`，sum 最大误差 `2.29e-5`、max 误差 `0` | 同样通过，误差一致 |
| generic Welford record reduction，Triton/cuTile | `M=64,N=257` 通过；mean 误差约 `3e-8`、variance `2.38e-7` | 同样通过，误差一致 |
| fixed add 的多组件长轴 scan，三个 surface | 通过；两个 component 最大误差分别 `7.63e-6`、`1.53e-5` | 通过；误差一致 |
| TileLang generic combine capability boundary | emission 前明确 unsupported | emission 前明确 unsupported |
| `I.arg_reduce.max` sugar 的 cross entropy forward/backward | 三 target 数值通过；旧/新路径分别对 reference 的 loss/prediction/gradient 一致 | 三 target 数值通过 |
| 两 stage ragged MoE execution contract | 三 target 数值通过 | 三 target 数值通过 |

H100 的 TileLang 初次运行曾调用系统 CUDA 11.5 `nvcc`，该工具不识别 `sm_90a`；切换到机器已有 CUDA 12.2 后，同一生成源码直接通过。这个失败属于运行环境工具链选择，没有转化成 compiler 或 Plan 特判。

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

当前 GPU 主线已经证明这不是 rowwise/softmax 特化。Tail correctness、Kernel IR 唯一合同、region/row/stream Plan binding、generic reduce/scan closure 与 stage execution contract 都已闭合；不同 surface 不能机械表达的能力会在 emission 前明确拒绝，而不是形成第二套编译器或慢路径。

编程模型由此冻结。除尚未接入 CPU/RISC-V/RVV target family 外，Intent 已形成完整的算子级语言—Kernel IR—target-family Plan—surface emission 闭环。后续新增 GPU provider 或 RISC-V/RVV backend 不应修改现有编程模型：只新增 target-family realizer、必要的 Physical Plan extension、capability 与机械 emission。新的 DSL 构造只有在真实算法无法用现有 Core表达、且不能由下层已有能力承接时，才按修改正式规格的标准进入。
