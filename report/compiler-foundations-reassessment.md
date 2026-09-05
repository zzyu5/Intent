# IntentDSL 编译基础复审

## 1. 结论与适用范围

当前 Intent 已有能承载复杂算法的可编程 DSL 骨架，而不只是一些独立数学 op 或按算子名称套模板的发射器。现有 `examples/kernels` 中的 FA、Mamba、Welford、MQA 和三角求解，分别保留了分段汇总、矩阵化计算、普通状态循环、自定义归约与顺序读写。编译器在这些作者程序之上形成 program mapping、fragments、blocking、访问和 accumulator，再交给 provider。当前工作的出发点应是改善和推进这套已有能力，不是先假定语言只能表达简单计算。

本次讨论确认的语言设计方向是：**公开 DSL 以作者熟悉的具名计算 op 为主要入口，以普通函数复用已有算法组合；能完整、等价映射的部分复用现有 contract、reduce/scan、region、普通控制与访问语义。** 常用 op 不应要求作者手写通用轴关系；算法作者仍可使用通用构造，不因增加具名 op 被限制在封闭算子目录内。原先“先维持 public contract 中心，再酌情补几个 shorthand”的建议被这一结论替代。最终分工是：作者决定算法，语言简化表达，编译器决定语义允许的物理组织。具体映射与新算法边界见 §2.6–2.7，下一轮行动依据集中在 §7.2；本文不是已经生效的新 API 规格。

本轮没有发现必须推翻 domain / structured operation / KIR / shared GPU Program 的证据，同时也发现具体实现未完全兑现语义：Triton f32 contraction 输入精度与 signed division/remainder、cuTile logical index 宽度与 autotune 可变输入污染。另一组失败严格限于 `a11b80a`、RTX 5090D、cuTile target 下的 BF16 最小 contract 形态，包括直接使用 rank-1 operand 的内积/GEMV；其中多轴与直接 paired-batch 例子在 shared construction 阶段失败。它们不是“Triton 的 GEMV 失败”或“DSL 不能表达这些算法”的证据，更不能抹掉已有复杂 kernel 的成功记录。具体实现问题按其路径处理，不作为重做语言的笼统理由。

维护上的主要问题也很具体：真实 transformations 大量包在名为 construction 的阶段中，类型/关系修补依靠手工编排；config 已从 serializer 分离，但数值表、适用条件和搜索策略仍混在 C++ 中；部分策略又被 verifier 固化成了唯一允许的候选域。这里应改善职责和可维护性，不能以新增限制取得“干净”结论。

调查基线为已提交的 `a11b80a`，工作区为 `comet/compiler-foundations-reassessment`。旧 cuTile change 继续暂停；原工作区未提交的 `StatefulPointwise` 和 metadata `ConstInt → int` 试验未纳入本轮编译器。没有修改生产实现、`doc/` 或 baseline CSV。

本文位置约定：`C:` 表示本仓库；`R_T:` 表示 `/home/kingdom/phdworks/ref/triton`；`R_L:` 表示 `/home/kingdom/phdworks/ref/tilelang`；`R_C:` 表示本机 `/home/kingdom/.venvs/intentdsl-cutile/lib/python3.10/site-packages/cuda/tile`。`source/` 是 kernel corpus，与这些真实 compiler 实现分开使用。此前因仅在项目内寻找 `ref/` 而判断参考源码不存在，是定位错误。

动态复现使用独立构建的本基线编译器、RTX 5090D、Torch `2.13.0+cu130`、Triton `3.7.1`、cuda-tile `1.5.0`。这是环境记录，不是引入版本管理。真实 ref checkout 用于设计和实现对照，不假定它与安装包逐行相同；下文明确区分实测结论和静态风险。

### 1.1 从已有作者程序判断抽象，而不是从少数失败推断整门语言

`C:report/baselinev2/triton-h100.csv` 的 54 个 entries 中，47 个为 `pass`；其余为 3 个 source ABI gap、2 个 source semantics gap、1 个 worker timeout 和 1 个 source compatibility gap。这里的 pass 经过 generated/source 调用及数值比较，generated 路径由 `compile_single → intent.compile → artifact.run` 形成，见 `C:examples/repro/v2/measurement.py:32`、`:203`，不是只运行了 source。CSV 最近一次提交更新是 2026-09-03 的 `61a6d9e`，不冒充 `a11b80a` 的新全量结果，但它是不能忽略的既有运行事实。`doc/` 用于界定语义，不替代这些实际代码和调用证据。

| 实际算法与已有 Triton/H100 entry | 当前作者表达与位置 | 对抽象的直接启示 |
|---|---|---|
| `modern_flash_attention_forward`，CSV `:27` pass | `flash_attention_fwd` 用 `region_fold`，helper 内写 QK/PV、softmax summary 与 merge；`C:examples/kernels/streaming/attention.py:399`，adapter `C:examples/repro/v2/providers/triton/attention.py:78` | 已有分段算法组织，不是只写最终 attention 公式等待 compiler 发明 FA |
| `mamba_state_passing`，CSV `:33` pass | 普通 ordered loop 携带 state，写每个 chunk 的 incoming state；`C:examples/kernels/streaming/mamba.py:159` | 状态算法不必统一成 scan/region，ABI-visible chunk 与循环依赖已经有普通表达 |
| `mamba_chunk_scan`，CSV `:34` pass | `mamba_chunk_scan_bf16_fwd` 中两次 contract、衰减系数与三角 mask；`C:examples/kernels/streaming/selective_scan.py:153` | 名字叫 scan，不表示应使用 scan intrinsic；作者已选择矩阵化算法 |
| `flaggems_batch_norm_training`，CSV `:45` pass | record `{count,mean,m2}`、自定义 `welford_combine` 与 generic reduce；`C:examples/kernels/normalization/batch_norm.py:10`、`:45` | 新统计组合可复用既有语义，不必增加 compiler 专用 Welford/batch-norm op |
| `flaggems_fp8_mqa_logits`，CSV `:51` pass | contract → ReLU/head weighting → reduce → scale/range mask；`C:examples/kernels/routing/mqa_logits.py:27` | 新计算可以由基础词汇组成，不需要 whole-kernel name matcher |
| `flaggems_triangular_solve`，CSV `:41` pass | 有序嵌套循环、residual carry、`InOut solution` 的逐行读写；`C:examples/kernels/factorization/triangular_solve.py:10` | 顺序依赖和外部可变状态并未被 DSL 控制边界删除 |
| `flaggems_cumsum`，CSV `:52` pass | `row_cumsum_f32` 显式 generic scan + add + inclusive；`C:examples/kernels/streaming/ordered_prefix.py:34` | 常见 prefix op 可以包装已有 scan，不需新算法机制 |

对照真实实现也支持这个分工：Triton FA 作者在 `R_T/python/tutorials/06-fused-attention.py:69` 用 BLOCK_N loop、两次 dot 和 `m_i/l_i/acc` 更新表达在线算法；Intent 在已有 `summarize/merge` 中保留算法知识，将分片大小交给 physical lowering。Triton `sum/cumsum` 则由具名函数转入 generic reduce/scan（`R_T/python/triton/language/standard.py:283`、`:328`）；TileLang `reduce_sum` 也包装统一 reduction（`R_L/tilelang/language/reduce_op.py:227`）。差异来自作者程序所处的抽象层，不意味着具名接口与通用机制只能二选一。

还必须追到实际 adapter，而不能取同文件中一个相似函数来代表 CSV：`paged_gqa_decode` 编译的是 `paged_gqa_decode_partials` 加 `splitk_attention_f32_to_f16_reduce` 两个 kernels，见 `C:examples/repro/v2/providers/triton/attention.py:204`、`:224`。本 case 的 HEAD_GROUP=4，partials gather 多个 query heads 形成矩阵，再在 region fold 中做收缩（`C:examples/kernels/streaming/paged_attention.py:122`），并非旁边单 query reshape 为 `(1,D)` 的 `paged_gqa_decode_attention`。因此其 CSV `:36` pass 与 §2.3 的 cuTile rank-1 最小例子并不冲突，二者也不能互相替代。

以上成功、当前源码和局部失败共同给出的结论是：**骨架已经承载多种复杂算法，常用表达仍可以明显改善，具体 lowering 形态也可能需要补齐。** 不是所有 examples 都能仅凭文件存在视作所有 provider 已运行，也不能从某个 provider 的局部失败判断整类算法不可表达。本轮收尾依据已有代码和 ref 的职责分工，不以再次全量运行或增加验证记录为前提。

## 2. DSL：保留什么，改善什么

### 2.1 先回答作者能写什么，而不是先给 `contract` 换名

`contract` 是**二元、显式配轴的乘加收缩**，不是 GEMM 的别名，也不是任意张量算子的总入口。作者声明哪些 lhs/rhs 轴配对求和，哪些轴只配对不求和；编译器负责把这一逻辑关系变成物理分块、轴重排和目标指令。`reduce=` 在这里是轴对列表，不是要调用另一个 `I.reduce`。规格：`C:doc/dsl/core.md:243`；真实 binding 和结果轴计算：`C:python/intent/frontend/lowering/intrinsics/structured.py:553`、`:1399`。

以下表格描述**语言和 frontend 的表达能力**，不表示这些形态已在所有 provider 上跑通。A/B 是已经读出的 tensor values，轴编号按它们当前的 rank 计算；每个调用还必须写 `acc_dtype=...`。

| 作者要计算的东西 | A / B 的逻辑 shape | `reduce=`；非空时另列 `batch=` | 结果 shape |
|---|---|---|---|
| 向量内积 | `[K]` / `[K]` | `((0,0),)` | `[]`，rank-0 tensor |
| 矩阵乘向量 GEMV | `[M,K]` / `[K]` | `((1,0),)` | `[M]` |
| 向量乘矩阵 | `[K]` / `[K,N]` | `((0,0),)` | `[N]` |
| 矩阵乘矩阵 GEMM | `[M,K]` / `[K,N]` | `((1,0),)` | `[M,N]` |
| 转置方向的矩阵收缩 | `[K,M]` / `[K,N]` | `((0,0),)` | `[M,N]` |
| 同 batch 的矩阵乘 | `[B,M,K]` / `[B,K,N]` | `((2,1),)`；`batch=((0,0),)` | `[B,M,N]` |
| 两个 batch 维、两个归约维 | `[B,H,M,K1,K2]` / `[B,H,K1,K2,N]` | `((3,2),(4,3))`；`batch=((0,0),(1,1))` | `[B,H,M,N]` |
| 双方各有多个 free axes | `[P,M,K]` / `[K,N,Q]` | `((2,0),)` | `[P,M,N,Q]` |

结果轴顺序有一条统一规则：先取 lhs 未归约的轴（包括 batch 的 lhs 代表），再接 rhs 未归约且不是 batch 配对成员的轴。extent 相同不会自动形成 batch；例如 `[B,M,K] × [B,K,N]` 只写 `reduce=((2,1),)`、不写 batch，得到的是 `[B,M,B,N]`，不是逐 batch 的 `[B,M,N]`。这正是显式 batch relation 必须保留的理由。

真实源码已经使用这些语义，不只是在设计文档中举例：普通 GEMM 在 `C:examples/kernels/contraction/gemm.py:24`；按 batch 标量迭代后写二维转置收缩在 `C:examples/kernels/contraction/batched_gemm.py:33`；显式 batch pairing 在 `C:examples/kernels/streaming/mla.py:250`；两对 reduction axes 在 `C:examples/kernels/contraction/block_scaled.py:79`。标量 indexing 会消掉一个 tensor axis，因此 `a[batch,k,m]` 的收缩轴编号是所得二维 tensor 的编号，不是原三维 view 的编号。这是作者目前需要手算的真实负担。

没有归约轴的 outer product **不属于 contract**，因为 `reduce` 必须非空。其作者写法是显式增加 size-one 轴后广播乘法，例如 `a[m][:, None] * b[n][None, :]`，结果为 `[M,N]`；不是 `contract(reduce=())`。同样，max-plus 等其它 semiring 用 pointwise 与 generic reduce 表达，三个及以上 operands 要由作者组合二元 operations；不能把 `contract` 宣传成完整的任意 `einsum`。空的 reduction **extent** 则合法，返回加法零，不能与空 reduction **列表** 混淆。依据：`C:doc/dsl/core.md:255`、`:264`、`:266`；frontend 的非空检查在 `C:python/intent/frontend/lowering/intrinsics/structured.py:1408`。

### 2.2 这套调用怎样读，哪些地方确实不顺手

一个完整 GEMM 作者例子只需 logical domains、tensor indexing、轴关系和显式输出转换，不写 tile/grid/warp：

```python
m = I.domain(0, M)
k = I.domain(0, K)
n = I.domain(0, N)
acc = I.contract(a[m, k], b[k, n], reduce=((1, 0),), acc_dtype=I.f32)
c[m, n] = I.cast(acc, I.bf16)
```

读法是“把左边第 1 轴与右边第 0 轴相乘求和，保留其它轴，以 f32 累加”。`acc_dtype` 声明数值语义，不是物理 accumulator storage；也不等于允许 TF32 输入。它不自动包含 BLAS 的 alpha/beta 或外部 C 初值，额外缩放/相加由作者写，输出 dtype 用 cast 表达。实际同类定义：`C:examples/kernels/contraction/gemm.py:39`；输入精度问题见第 4.1 节。

这套写法的优点是能力集中、关系明确、不随 provider API 改变；缺点是读者需要在两个 operand 的位置编号之间来回对应，尤其经过 indexing/transpose 后容易错。`contract` 这个术语对熟悉张量收缩的人准确，对只想写 dot/GEMV/GEMM 的作者则把本可由操作定义确定的轴关系交给了人。**“核心语义成立”并不等于“适合作为普通作者的主要接口”。** 这是 DSL 设计问题，不只是缺一份签名说明，也不必等到出现 correctness 反例才能成立。

参考 `tl.dot(input, other, acc=..., input_precision=...)` 实际主要表达最后两轴矩阵乘，并非数学教材里 rank-1 的 dot；当前 ref 的实现还会把多个前导 batch 维 flatten。`T.gemm(A,B,C,transpose_A=...,transpose_B=...,...)` 则写入显式 C buffer，并暴露 policy、clear-accum 等低层选择。位置：`R_T/python/triton/language/core.py:2215`、`:2260`；`R_L/tilelang/language/gemm_op.py:148`。它们并非 Intent 高阶配轴能力的全部替代，但都让作者直接选择熟悉的计算，不能因为输入更低层就否认这一接口设计的启示。

更重要的是，它们并不遵循“具名计算必须先抹成 generic contraction IR”的原则：Triton 有独立 `TT_DotOp/TT_DotScaledOp`，见 `R_T/include/triton/Dialect/Triton/IR/TritonOps.td:666`、`:706`；TileLang `T.gemm` 发出 `tl.tileop.gemm`，再由专门 `Gemm` 表示 lowering，见 `R_L/tilelang/language/gemm_op.py:187`、`R_L/tilelang/tileop/gemm/__init__.py:32`、`:127`。另一方面，Triton `sum/max` 共享 generic reduce，TileLang `reduce_sum/reduce_max` 也转调统一 reduction，见 `R_T/python/triton/language/standard.py:168`、`:261`，`R_L/tilelang/language/reduce_op.py:164`、`:227`。真实对照支持的是**具名公共接口与内部复用可以并存**，不是越多独立 IR op 越好，也不是越少越好。

因此，应正式建设面向作者的计算 op，再为其选择等价映射。对于语义已被当前 canonical constructs 完整覆盖的 dot/matvec/matmul 等，复用现有 contract 是本项目的低成本实现方向，不要求每个名字另建一套 canonical 或 backend path。对高级用户保留通用构造；不再要求普通用户先学习它们才能写普通计算，也不为获得短写法牺牲高阶表达能力。

### 2.3 特定 cuTile 调用形态的缺口，不是整类算法的能力结论

本轮补充一个 production repro，使用本基线、cuTile、BF16 全一输入和 f32 输出，通过 `lower_to_mlir → intent.compile → artifact.run`。全部定义先形成 canonical KIR；能编译的继续 emit 后端源码、JIT 并对数值，编译失败的保留原始诊断，不换算法或改 generated source。命令和核心定义见第 8 节。

| 形态与输入 | 本轮实际结果 | 能说明什么 |
|---|---|---|
| rank-1 dot：`[64] × [64]` | cuTile legalization 拒绝，native MMA 要求二维或标准三维 physical axes | canonical 允许，当前这条 provider 路径没有完成等价 realization |
| 直接 rank-1 operand GEMV：`[32,64] × [64]` | 同上 | 仅说明此 cuTile 表达形态未闭合，不等于 GEMV 算法或 Triton provider 不支持 |
| 转置 GEMM：`[64,32] × [64,32]`，reduce `((0,0),)` | 成功 emit/JIT/launch；输出 `[32,32]`，与全 64 的最大误差为 0 | 已有 shared 轴正规化确实起作用，不能说作者必须自己先 transpose |
| 多轴收缩：`[32,2,32] × [2,32,32]`，reduce `((1,0),(2,1))` | shared construction 报 broadcast 的 non-singleton physical extents 冲突 | 本例在 provider 前已失败，不能直接归因于 cuTile MMA 不支持 |
| 直接 paired batch：`[2,32,64] × [2,64,32]` | shared construction 报 `could not form pointwise store validity` | 这个写法尚有 shared 覆盖缺口；不等于所有 batched GEMM 均不支持 |

对应 current 代码：`C:lib/Dialect/GPU/Transforms/RealizeContractionBlocking.cpp:873` 的 `normalizeMatrixContractForms` 会压缩特定 unit free axes，并在 `:1053`、`:1070` 重排最后两轴；它不是任意高阶 contraction 的通用 flatten。cuTile 的 MMA form 检查在 `C:lib/Target/CuTile/Transforms/Legalize.cpp:1192`；Triton/TileLang terminal 的标准 dot/GEMM 条件分别在 `C:lib/Target/Triton/Transforms/Legalize.cpp:1713`、`C:lib/Target/TileLang/Transforms/Bufferize.cpp:1599`。这些检查消费的是**已经经过 shared transformations 的 physical operands**，不能拿它们直接限制 public DSL 的 rank 或轴方向。

双方差异与后果：Triton `tl.dot` 的标准矩阵输入边界在 `R_T/python/triton/language/core.py:2260`，TileLang `T.gemm` 的矩阵 buffer/transpose 契约在 `R_L/tilelang/language/gemm_op.py:148`。Intent 接收更一般的 logical contract，因此还欠缺某些 logical→physical form 的正规化或等价展开；下层接受窄 form 本身不是下层的错误。源码中已有外层 `I.parallel` batch 写法，也已有 shared 轴转换，不能把上表的单例失败泛化成所有 batch、所有 permutation 或所有 provider 失败。本轮未动态验证向量乘矩阵、任意多 free/batch axes、TileLang/Triton 对应形态；这些保持未知。

### 2.4 `reduce`、`scan`、region 与普通循环：作者按结果和依赖选用

这些构造不是一组任选的“流式计算”别名。首先看作者需要一个最终结果还是每个位置的结果，再看 summary 是逐 element 已经存在，还是必须在任意连续 slice 内做 tensor operations；严格 recurrence 另走普通循环。

| 作者的问题 | 合适构造 | 作者提供什么 | 结果与真实例子 |
|---|---|---|---|
| 一行求和/最大值，或多字段统计量的最终汇总 | `reduce.sum/max` 或 generic `reduce` | source、axes、identity；custom 情况再提供 combine | 删除归约轴；layer norm：`C:examples/kernels/normalization/layer_norm.py:21`；Welford：`C:examples/kernels/normalization/batch_norm.py:45` |
| 每个位置的前缀和，或可组合 transition 的每个前缀 | `scan` | source、一个 axis、identity、combine、inclusive；可指定 reverse | 保留 source shape；`C:examples/kernels/streaming/ordered_prefix.py:41` |
| 对任意连续 K slice 显式做 QK/PV 等运算，再只取合并后的 summary | `region_fold` | 同步切片的 source、summarize、combine、identity、未切片 captures | summary schema 不必像 source；attention：`C:examples/kernels/variants/streaming.py:36` |
| slice 内做 tensor operations，同时需要 incoming state 与每个位置的输出 | `region_scan` | 上述 summary algebra，再加 initial_state、apply、emit | 完整输出与 final state；linear attention：`C:examples/kernels/streaming/linear_attention.py:223` |
| 更新不能重结合、存在 ordered effects/动态停止，或必须保留显式 chunk 状态 | 普通 `for/while` 与 carry；可组合时也可在显式 logical chunk 轴上用 ordinary scan | 作者自己的 transition、边界、状态和读写 | chunk 数和 per-chunk state 保持 ABI；规格例 `C:doc/dsl/examples/mamba_state_passing.py:16` |

例如一行 inclusive 前缀和的实际调用是：

```python
output[row, columns] = I.scan(
    x[row, columns], axis=0, identity=0.0, combine=I.add, inclusive=True,
)
```

不需要为加法再写四个 region helpers。反过来，generic reduce 不只返回单一 sum：Welford 例子把每个元素表示为 `{count,mean,m2}`，identity 是同字段 record，`welford_combine(left,right)` 返回同 schema；这样保留统计量的组合规则，而非让 compiler 从三个互不相干的 reductions 猜 Welford 算法。combine 见 `C:examples/kernels/normalization/batch_norm.py:10`。

scan 也不等于“任意循环自动并行”：仿射更新 `s' = a*s+b` 可以先把 `(a,b)` 当 transition summary，按 `combine(left,right)=(a_right*a_left, a_right*b_left+b_right)` 组合，再对 prefix 应用初始状态；combine 可以不交换。current retention helper 正是这个模式的 matrix-state 形式，见 `C:examples/kernels/streaming/linear_attention.py:190`。没有这样的可重结合表示时，作者写 ordinary loop，不应为性能把错误的 combine 塞进 scan。语言允许的是保持 logical order 的 reassociation，不保证 left fold；编译器检查类型/purity/relations，不证明作者的结合律。规格依据：`C:doc/programming-model/logical-program.md:97`、`:120`。

### 2.5 Region 的使用负担是真实的，但要分清哪些不能删

attention 的 region-fold 作者提供的不是一个名为 attention 的 compiler hint，而是明确的分片算法：summarize 中有 QK contract、mask、max/sum 和 PV contract，返回 `{valid,maximum,denominator,accumulator}`；combine 合并该 summary；query、query coordinates、scale 等放在 `operands` 中不随 K slice 切分。源码：`C:examples/kernels/streaming/attention.py:137`；调用：`C:examples/kernels/variants/streaming.py:36`。读者必须跨这两个 helper 才能看全算法，负担确实高于一个 sum，但它换来的是作者明确指定 slice 内的 tensor algorithm，而非 compiler 识别 whole-operator 模板。

region-scan 更重。当前 linear-attention 作者分别定义：`summarize` 做 slice 的 KᵀV；`combine` 加矩阵 summary；`apply(prefix,initial)` 求 incoming state；`emit` 用 QK/PV 与 Q×incoming-state 产生本 slice 输出。四种角色见 `C:examples/kernels/streaming/linear_attention.py:11`、`:21`、`:36`、`:223`。这里 source 包含 query/key/value/absolute token coordinates，slice 边界不可观察；例子展示了调用结构，不单独证明所有分片下的有限精度等价或全部 provider 可运行。

`identity` 和 `initial_state` 不能一般性合并：仿射 transition 的 identity 是 `(1,0)`，实际初始状态可以是任意 state tensor，二者甚至不同 schema。`apply` 将 transition 作用于 state，`emit` 才生成 source-aligned output；这不是 TileLang reducer 的“init/update/finalize”换个名字。当前 frontend 分别检查 transition、state、emit output schema，见 `C:python/intent/frontend/lowering/intrinsics/structured.py:483`、`:508`、`:519`。同一 linear-attention 例恰好两者都用零矩阵，不构成删参理由。

captures 也不只是拼写不同：reduce/scan 的 `combine_operands` 传给 combine；region 的 `operands` 传给 summarize，及 region-scan 的 emit，**不传给 region combine/apply**。位置：`C:python/intent/frontend/lowering/intrinsics/structured.py:356`、`:402`、`:426`、`:499`、`:519`。可以评估更清楚的命名/签名说明，却不能把两个参数机械合并并改变可见依赖。普通 `@intent.fn` 本身可在允许的调用点有 effects；这里的 structured helpers 才要求 pure，检查见同文件 `:831`、`:890`。因此“所有 helper 都不能有 effects”同样不成立。

参考差异：Triton `tl.sum`/`tl.cumsum` 已为加法选好 combine，generic `associative_scan` 接 tensor/tuple 与 `@triton.jit` combine，不提供 Intent 的 arbitrary-slice summarize/apply/emit。位置：`R_T/python/triton/language/standard.py:283`、`:328`；`R_T/python/triton/language/core.py:3021`。TileLang `T.reduce(buffer,out,...)`、`T.cumsum(src,dst,...)` 是 buffer 级操作，reducer init/update 是状态生命周期接口，见 `R_L/tilelang/language/reduce_op.py:24`、`:351`，`R_L/tilelang/language/scan_op.py:88`。可以借鉴其显式签名和 built-in 简写，不能把这些较低层操作当作 Intent region-scan 的等价替代。

当前已有 `realizeRegionFolds` 和 `realizeRegionScans`，见 `C:lib/Dialect/GPU/Transforms/Passes.cpp:68`、`:76`；不能拿旧报告的“尚无物化”描述现在。另一个需要区分的事实是：frontend `_region_fold/_region_scan` 直接 emit 对应 op，`canonicalize_mlir` 只 parse/print，见 `C:python/intent/frontend/lowering/intrinsics/structured.py:435`、`:531`，`C:python/intent/frontend/mlir/builder.py:377`；这条 frontend 路径尚未落实规格要求的“退化 element-fold/scan 归一回 ordinary reduce/scan”。这是静态识别到的 canonicalization 缺口，不等于上述 region realization 不存在，也不是本轮已复现的数值错误。

### 2.6 具名 op 与统一语义怎样共同成立

用户提出的“如果现有语义已能满足，那么具名 op 可以映射到统一轴”成立。§1.1 的已有复杂 kernels 表明，这不是仅有理论可能性的设想。应先从已有表达中识别可由操作定义自动确定的机械部分；映射保持全部可观察语义即可复用。只有进入某个确有缺口的表达/provider 路径时，才结合 §2.3 等具体事实补齐 lowering，不能把局部缺口变成设计整个公共接口前的泛化阻塞。

建议形成两个并存的作者入口，而不是两套编译器：普通作者使用具名计算；算法作者在 `@intent.fn` 或 kernel 中组合具名计算与通用 contract/reduce/scan/region、indexing 和 control。二者都进入现有 typed canonical 路径，再由 shared/provider 实现。**“公开 op 是正式接口”不要求“每个公开名字都是一个独立 canonical op”；“内部共享表示”也不要求“公开只剩一个通用名字”。**

下面是下一轮确定 public API 时可直接采用的**语义映射依据**，其中新名字是候选，不是已实现接口；现有 `reduce.sum/max` 等不因此重复建一套入口。

| 用户侧计算 | 可复用的统一语义 | 不能省掉的契约 |
|---|---|---|
| 实数/整数向量内积 `dot` | rank-1 contract，`reduce=((0,0),)` | 输入 rank、归约长度、accumulator/result dtype 与 rank-0 结果；不暗含当前类型系统没有的复数共轭 |
| `matvec` 与向量乘矩阵 | 相应的 `((1,0),)` / `((0,0),)` contract | 两种方向的 operand/result 规则；名字或 rank 分派必须明确 |
| `matmul` 与 batch 矩阵乘 | 按操作定义生成 reduction/batch pairs；必要的逻辑 transpose/broadcast 显式进入 KIR | batch 对齐、是否允许 batch broadcast、transpose、输出顺序、dtype；不能凭 extent 相同猜 batch |
| `outer` | 增加 size-one 轴后的 broadcast multiply | 输出轴序与乘法 dtype；不是空 reduction 的 contract |
| `sum/max` 等 builtin reduction | generic reduce 加规定的 combine/identity/promotion | empty、NaN、tie 与累加 dtype；内置 identity 可以由 op 定义产生，不需每次由作者重复证明 |
| `cumsum/cummax` 等 prefix op | generic scan 加规定的 combine/identity/direction | inclusive/exclusive、forward/reverse、dtype；严格 left fold 不能假装是可重结合 scan |
| scaled/sparse 矩阵计算的具名入口 | 当前 scaled/sparse contract，及其明确的 logical transforms | format、scale-group、metadata interpretation、rounding；不能一律降格成普通乘加 |

例如拟议的 `I.matmul(a, b, acc_dtype=I.f32)`，由 API 契约确定轴关系，而不是继续要求 `reduce=((1,0),)`。这是把结构选择和错误检查交还语言，不是 kernel-name matcher：匹配的是作者显式调用的语言操作；不按 `@intent.kernel` 的函数名或整段源码识别 GEMM/attention 模板。映射不能额外选 provider、tile、hidden launch 或另一种数值模式。若 canonical 结果与旧表达完全相同，下游不应仅因公开名字变化获得另一套 config/serializer 策略。

“等价映射”也不只是结果 shape 一样。要同时保持 dtype/promotion、累加与近似模式、NaN/empty/tie、source order 和允许的 reassociation、view/alias/effects、输出与 host-visible invocation。比如 `outer` 的低精度逐元素乘法，不能随手塞一个 unit reduction 后继承不同累加语义；`matmul` 的 f32 accumulator 不能隐式授权 TF32 输入；一个带 read-modify-write 的操作不能因为叫 GEMM 就按纯 Out 包装。现行对应规格：`C:doc/dsl/types-numerics-and-effects.md:35`、`:73`、`:94`、`:140`；已复现风险见 §4。

公开操作需要完整签名、类型/shape 检查、用户侧诊断、用例和生产 lowering，而不只是增加 `Intrinsic("matmul")`。当前 `Intrinsic.__call__` 只有 `(*args, **kwargs)`，真实 required/default 契约在 AST `bind_call`，见 `C:python/intent/language/builtins.py:9`、`C:python/intent/frontend/lowering/intrinsics/common.py:37`；scan 的 inclusive 必填、reverse 可省及 contract 的 binding 在 `structured.py:284`、`:553`。这部分应与新增 op 一起收敛为同一份 surface 契约，不另维护一套容易漂移的签名。

专用 op 可吸收真正固定的参数：scaled contract 的双侧 group size 相同、reduction 固定 `((1,0),(2,1))`、batch 固定为空，见 `C:python/intent/frontend/lowering/intrinsics/structured.py:574`、`:619`；可在对应具名入口中简化这些参数，canonical scale/format relation 不变。相反，自定义 combine 的 identity、region 的 transition 与 initial state 不一定固定，不能为统一短签名而自动猜测。对已明确的算法，author-library helper 可以构造这些值并封装协议；改变数学语义的选择仍必须显式可见。

改善作者体验也不等于不断增加 intrinsic。已有 FA 的 summarize/merge、Welford 的 combine 等可以通过正常的 typed `@intent.fn` 接口复用：普通使用者调用已定义的算法函数，算法作者进入其 DSL body 修改组合；两者都沿原编译路径，不调用 provider source 模板。应同时完善“常用计算词汇”和“函数化算法复用”，而不是要求每个使用者重写完整 generic/region 协议，或给每个算法名称增加一个 compiler op。

现行 `C:doc/dsl/README.md:59` 的机械展开判据约束的是 canonical 语义重复，不能用于否决具名 public op。下一轮默认复用现有 canonical constructs；若某个操作有尚无法完整保存的语义，再提出具体 typed 扩展，不为每个名字复制 IR，也不为了“统一”提前丢掉操作的独有语义。现在不删除 `contract/region`，也不决定它们必须私有化或移动 namespace；本轮确认的是常用入口的设计重心，不是移除高级用户能力。

### 2.7 新算法怎样进入：通用能力有价值，但不是万能承诺

**新算法名称不等于新语言语义。** 一个新的 attention/统计量/递推算法，如果能用既有 pointwise、indexing、具名乘法、通用 reduce/scan/region 和普通 control 准确写出，就应先是作者程序或 `@intent.fn`，不需要 compiler 认识它的论文名。只有频繁使用、具有稳定用户契约时，才另评估是否提供库级具名入口；库调用同样可以是正式可用接口，而不一定新增 compiler intrinsic。

| 新算法带来的变化 | 现有表达方式与成立条件 | 不能据此承诺的事 |
|---|---|---|
| 新的收缩轴组合、输入转置、free/batch 关系 | generic contract 配合 logical transforms；满足二元数值乘加和显式配轴语义 | 不等于任意高阶形态当前已经实现、能映射到 MMA，或性能不需新 shared rewrite |
| 新统计 summary 或 prefix transition | tuple/record 加自定义 pure combine，使用 reduce/scan，明确 identity 与允许的顺序 | 不自动推导结合律，也不自动发现最优 summary；Welford/仿射 transition 的源码结构见 §2.4 |
| 新的分片内张量算法，例如修改 attention 的 slice 计算 | region_fold 的 summarize/combine；需要逐位置输出时用 region_scan 的 apply/emit；必须仍满足分段等价关系 | 不是任何 mask、归一化、舍入或状态修改都能保留原来的 summary algebra |
| 严格非线性 recurrence、early stop、ordered effects | 普通 for/while、helper、carry、logical buffer 和明确读写 | 没有可用 summary 时不能强行装进 region；语言能写 loop 也不代表 compiler 能并行化它 |
| 新的逻辑 group/window/chunk 或多 kernel 算法 | 显式 domain/subregion/index relation；必要时由 host 显式编排多个 kernels | 不能把 ABI-visible chunks 当 compiler-selected segments，也不能由一个 wrapper 暗增 host-visible kernel |
| 真正新的 format、数值模式或 effect 契约 | 先判断现有 operations 是否可准确组合；若必须保存而现有 schema 无法承载，再讨论 typed canonical 扩展 | “contract 很通用”不能代替新的 metadata/rounding/atomic 语义；仅新增硬件指令通常是 provider lowering，而非新 DSL op |

contract 与 region 是**互补**的，不是替代关系：contract 表达“局部张量怎样乘加”，region 表达“对 source 怎样分段汇总/传播状态而仍是同一程序”。region summarizer/emitter 可以调用 contract，也可以只含 pointwise/reduce；新 public `matmul` 归一到 contract 后，仍可在同样的 pure helper 中使用。§2.5 的 attention QK/PV 与 linear-attention KᵀV 就是已有源码层面的组合证据，不是对任意新算法的完整覆盖证明。

FA 因而不构成“公共接口必须叫 contract”的理由：其中局部 QK/PV 可以用具名矩阵 op 表达。需要额外保留的是作者选择的在线 summary/merge，而不是那个名字。单纯写 attention 公式并不强制物化完整 attention matrix，但也不等于已经向编译器表达了在线算法。当前 `C:lib/Dialect/GPU/Transforms/OnlineSummary.cpp:190` 能识别具体的 validity/max/mass/moment 代数形态；这是一类已实现的优化，不是从任意数学图发明算法的承诺。Region 的作用是允许作者明确算法，同时把合法的物理分段留给编译器；需要最终汇总用 fold，需要逐位置结果和 incoming state 才用 scan。

判定能否使用 region 必须具体到作者给出的关系，而不是算法听起来像 streaming。除固定 summary schema、pure helpers、source-axis/capture/output relation 外，还要求：

```text
S(A ++ B) = combine(S(A), S(B))
combine(identity, s) = combine(s, identity) = s
apply(identity, state) = state
apply(combine(a, b), state) = apply(b, apply(a, state))
emit(A ++ B, state)
  = concat(emit(A, state), emit(B, apply(S(A), state)))
```

后面三式针对 region_scan；region_fold 只需前面的 summary 合并关系。A/B 是相邻且保持顺序的 source slices；chosen segment count、extent 与内部 prefix states 不可观察。浮点等价按现行允许的 segmentation/reassociation 理解，不承诺逐 bit 相同，但不能改变 NaN、accumulator 或 approximation 契约。依据：`C:doc/dsl/core.md:173`、`:185`、`:219`、`:226`；`C:doc/dsl/types-numerics-and-effects.md:106`。编译器检查可验证的结构和 effects，不替作者证明这些代数律。

一个正例是仿射 recurrence `state' = a*state+b`：transition `(A,B)` 的顺序组合为 `(A_right*A_left, A_right*B_left+B_right)`，identity 为 `(1,0)`，apply 为 `A*state+B`。每个 element 已经是这种 summary 时用 ordinary scan 即可；只有 slice 内确实有需要保留的 tensor algorithm 时才用 region。一个不应直接宣称可 region 化的例子是 `state' = tanh(W@state+x)`：在没有给出固定、可组合 summary 的情况下，写普通 loop；不能把任意 Python 函数塞作 opaque transition。可观察 chunk 状态的 Mamba 例也是 ordinary control，见 `C:doc/dsl/examples/mamba_state_passing.py:20`。

因此遇到新算法，必须区分三种实际工作：**已有语义的作者组合；已有语义缺少实现的 shared/provider 补齐；已有语义无法完整表达时的语言扩展。** 若语义与 lowering 均已覆盖，增加常用具名入口主要是 frontend/API 工作；若只是能写出 KIR，却像 §2.3 那样编译失败，仍需实现工作；若只能通过改变数值、effects、format 或可观察分段来“表达”，则不能算已覆盖。现有通用能力使新算法不必总是新增 op，但不构成“以后任何算法都无需扩展语言/编译器”的保证。

最终控制边界是：作者决定数据关系、数值计算、顺序与状态、summary 或矩阵化算法；语言消除具名操作已确定的机械表达并支持函数组合；编译器形成语义允许的 blocking、mapping、storage 和 provider 程序。普通循环不会因为存在 region 就被排除，名字叫 scan 的算法也不必被强迫改成 scan intrinsic。只有一个具体算法的必要语义确实无法由这些构造保留下来，才讨论抽象缺口；不从 kernel 名称、某个 target 的 form 限制或一次失败推导整门语言的能力边界。

## 3. 实际编译结构及 pass 的贡献

### 3.1 真实 production 顺序

```text
受限 Python DSL
  → frontend specialization / desugaring / canonical KIR
  → KIRToGPU：初始 mapping、value/access/control/resource
      → completeGPUProgramConstruction：access composition、ownership/blocking、
        online summary、region fold/scan、reduce/contract realization 与关系维护
  → runSharedGPUPasses：mapping refinement、shared config tuples
  → 选定 provider 的 legalization / local IR
  → source serialization 与已声明 launch wrapper
  → 外部 provider JIT / autotune / 机器 lowering
  → artifact 调用
```

入口是 `C:python/intent/compiler/pipeline.py:15` 和 `C:tools/intent-compile/intent-compile.cpp:101`。`C:lib/Conversion/KIRToGPU/KIRToGPU.cpp:5362` 调用 construction completion；完整 shared transformation 顺序见 `C:lib/Dialect/GPU/Transforms/Passes.cpp:9`，随后 `:124` 才是名字上的 `runSharedGPUPasses`。

因此只看后一个函数会误以为 shared 只做 mapping 和 config。大量实际工作被包在前面的 construction completion 中，这是当前编译结构难以解释和定位耗时的重要原因。

### 3.2 已经形成了真实物理程序

本轮从现有 `bf16_gemm` 实际输出 canonical KIR、shared-final IR、cuTile source，并 JIT/launch。KIR 是一个动态 tensor contract；shared IR 已有 BM/BN/BK、GROUP_SIZE_M、program id、分块 K loop、fragment accumulator 与带有效性的访问。以下仅缩短 SSA 和参数名字，结构来自实际输出：

```text
KIR:    contract tensor<M,K> × tensor<K,N> → tensor<M,N>

Shared: BM, BN, BK = typed parameters
        (m_owner, n_owner) = grouped_mapping(program_id)
        acc = full<fragment<f32,[BM,BN]>>(0)
        for k0 = 0 to K step BK iter_args(acc):
            a_tile = load A[m_owner*BM + range(BM), k0 + range(BK)]
            b_tile = load B[k0 + range(BK), n_owner*BN + range(BN)]
            acc = contract a_tile, b_tile, acc
        store C[owned_m, owned_n] = cast(acc, bf16)

cuTile: ct.bid / ct.load 或合法 gather form / ct.mma / ct.store
```

该例完整默认候选有 45 项。输入为两个 `128×128` BF16 全一矩阵，输出 shape 为 `(128,128)`，与精确值 128 的最大误差为 0。这里证明真实 lowering 可运行，不把该例或候选数作为通用性能结论。

其他可回查的实质改写：

| 改写 | 实际改变 | 当前实现 | 同类参考与边界 |
|---|---|---|---|
| ownership/blocking | rank-lift value graph、重写轴关系、插入 chunk loop 和 tail、重放访问并替换 store | `C:lib/Dialect/GPU/Transforms/RealizePointwiseBlocking.cpp:1504`、`:2496` | Triton 输入已是 block program，不能要求其重复 Intent 的 logical→block 工作；其布局/类型转换见 `R_T/lib/Conversion/TritonToTritonGPU/RelayoutTritonGPU.cpp:118` |
| reduction/summary realization | source 分块、局部 reduce、carry combine；满足条件时共同实现 online summary | `C:lib/Dialect/GPU/Transforms/RealizeReductionBlocking.cpp:2593`、`C:lib/Dialect/GPU/Transforms/OnlineSummary.cpp:190` | 局部代数 pattern 合法；Triton 同样做 combine，包括 dot+add，见 `R_T/lib/Dialect/Triton/Transforms/Combine.cpp:249` |
| persistent mapping | 创建 resident worker 参数和真实 grid-stride `scf.for`，移动 task body，改写 grid | `C:lib/Dialect/GPU/Transforms/RefineProgramMapping.cpp:181`、`:215` | source persistent GEMM 明确具有同类 traversal，见 `C:source/cutile/tilegym/gemm/dense/matmul.py:218`；shared 负责此结构还是下层负责，应按输入层级判断 |
| TileLang bufferization | 把 fragment SSA/carry 实现为明确 buffer、copy、GEMM 和更新 | `C:lib/Target/TileLang/Transforms/Bufferize.cpp:1085`、`:1443` | `R_L/tilelang/cuda/pipeline.py:125` 继续做 pipeline/layout/lower-tile，属于后续更低层职责 |

这些改写不是调几个 config 名字。另一方面，局部 summary pattern 也不意味着编译器能自动从任意数学代码发现任意最佳算法；当前仍依赖作者显式 structured semantics 和实现覆盖的 rewrite forms。

### 3.3 阶段维护需要改善，但不是“验证器偷偷修程序”

`alignPointwiseValueRelations`、`alignAccessValueRelations`、`alignAggregateValueRelations` 会插入/rebuild op、修改类型、替换 operands 和 loop carries，它们是 transformation repair。位置分别为 `C:lib/Dialect/GPU/Transforms/Utilities.cpp:1650`、`:1954`、`:2151`。`verifyGPUProgram` 本身是只读验证，见 `C:lib/Dialect/GPU/Transforms/VerifyGPUProgram.cpp:151`。

当前每组主要 realization 加 repair 后都调用 verifier，并非整个管线只验证一次。组内连续 repair 允许出现中间不完整类型关系，完整性边界实际是这一组函数；不能把每个 helper 都当独立可运行的 pass。

真正的维护风险是这组后置条件和依赖顺序不够集中：例如 pointwise blocking 后手工重复多次 pointwise/aggregate/access alignment，后续加一类 op 时容易漏掉一处关系维护。`retargetDimensionExtent` 只以 dimension ID 选择传播，而更精确的 `retargetSourceExtent` 使用 source identity；调用者的 authority/projection 条件很重要。位置：`C:lib/Dialect/GPU/Transforms/Utilities.cpp:2609`、`:2621`。

不能据此断言已存在误绑定：当前 `refinePhysicalSchema` 会保留 source axis，并在竞争 authority 时查询 exact lockstep；见 `C:lib/Dialect/GPU/Transforms/Utilities.cpp:1430`、`:1477`、`:1517`。本轮没有复现 shared 跨 source 误绑定。

参考 Triton 使用 TypeConverter、conversion target 和 rewrite patterns 管理类型转换；其 axis analysis 也明确在 mutation 前收集、随后改 IR，见 `R_T/lib/Conversion/TritonToTritonGPU/RelayoutTritonGPU.cpp:118`、`R_T/lib/Dialect/TritonGPU/Transforms/CoalesceAsyncCopy.cpp:175`。Intent 当前实例级 analysis 生命周期与后一原则一致，没有证据支持笼统的 stale-cache 指控。

可维护方向是把现有语义完整的 transformation groups 明确命名，收拢各自 postconditions，再按需要接入标准 pass instrumentation。当前 `registerIntentPasses` 只注册 canonical verifier，shared transformations 是普通 C++ 调用；因此标准逐 pass dump/timing 并不能直接覆盖它们。位置：`C:lib/Transforms/VerifyKernelIR.cpp:354`、`:390`。这不要求重做 GPU IR，也不应把额外 profiling 框架当本轮必需品。

## 4. 已复现的语义缺口

以下都是现行规格下的实现问题。对应脚本和运行方式集中在第 8 节；本轮只调查，尚未修复。

### 4.1 f32 contraction 输入精度被下层默认值改变

当前 `C:lib/Target/Triton/Serialization/Serializer.cpp:988` 只发出 `tl.dot(lhs,rhs,acc)`。参考 `R_T/python/triton/language/core.py:2236`、`:2249` 显示 NVIDIA 默认允许 TF32 输入，accumulator f32 不表示完整 f32 输入精度。

反例使用 `64×64` 矩阵，每个结果只有一个非零乘积 `1 + 3×2^-11` 乘 1，其余全零。production Triton 输出 `1.0009765625`，规范值 `1.00146484375`；最大绝对差 `0.00048828125`。同一 DSL 在 production cuTile 输出规范值、最大差 0。单非零项排除了 reassociation、求和树或 FMA 抵消造成该差异的解释。

cuTile 区分 f32 与 tfloat32 输入，见 `R_C/_stub.py:2066`、`:2083`，不能因为其 full-f32 路径比显式 TF32 source 慢就擅自转精度。TileLang ref 的 MMA generator 对 f32/f32 accumulator 有明确 TF32 operand 选择，见 `R_L/tilelang/cuda/intrinsics/macro/mma_macro_generator.py:145`、`:162`；当前 Intent `T.gemm` 未表达输入精度，见 `C:lib/Target/TileLang/Serialization/Serializer.cpp:622`。TileLang 的具体选中机器路径本轮未运行，列为必须核清的同类风险，不冒充已实测失败。

最小方向：先兑现现行 f32 数值契约，provider lowering 明确选择等价 form。若项目希望允许 TF32 等近似，应作为明确的算法数值模式讨论；不能通过 config 或默认 provider 参数静默放宽。`contract` 名称或 paired-axis 抽象不需要因此推翻。

### 4.2 signed floor division/remainder 继承了 Triton 的截断规则

`C:doc/dsl/types-numerics-and-effects.md:55` 要求 Python floor quotient、remainder 与 divisor 同号。当前 serializer 直接发 `//`、`%`，见 `C:lib/Target/Triton/Serialization/Serializer.cpp:1271`；参考分别 lower 成 signed divide/remainder，见 `R_T/python/triton/language/semantic.py:312`、`:338`。

从 runtime tensor 读取 `-3` 和 `2` 的 production repro 得到 quotient `-1`、remainder `-1`，规范要求 `-2`、`1`。该问题不依赖超大 shape，也不是 compile-time Python constant folding。

最小方向：把 canonical floor/remainder 机械兑现成等价 provider operations；有已证明非负输入时仍可保留简化形式。不要为了匹配 Triton 默认值改写 DSL 整数语义。

### 4.3 cuTile 有序循环无证明地将 logical index 缩为 i32

当前 `C:lib/Target/CuTile/Serialization/Serializer.cpp:27` 将 index spell 为 `ct.int32`，`:918` 对每个 `scf.for` lower/upper/step 强制 cast i32。现行 logical index 是 signed64，physical 缩宽必须有相应证明。

本轮只分配一个 i64 元素，读取值 `2^31`，运行 `[begin, begin+1)` 一次循环，再把 induction value 写回 i64 输出。实际得到 `-2147483648`，规范值为 `2147483648`。生成源码明确包含边界 i32 cast，输出回转 i64 无法恢复已经丢失的高位。

参考 Triton 自身的 loop lowering 在 `R_T/python/triton/compiler/code_generator.py:1309` 对 lower/upper/step 先统一类型后创建 induction variable；并不无条件固定成 i32。cuTile 也公开 i64 scalar/array indexing 注解，见 `R_C/_stub.py:1084`、`:1096`。问题是当前 terminal lowering 的选择，不能把它包装为 cuTile 只支持 32 位。

相关静态风险：cuTile `ScalarABI.type` 被收集后没有用于普通 scalar 参数注解（`C:lib/Target/CuTile/Serialization/Serializer.cpp:242`、`:375`），而 provider 默认 Python int 参数为 i32；large i64 scalar ABI 需要同样核清。该 scalar-argument 路径未单独动态复现。

### 4.4 autotune 首次调用污染可变输入

对 `I.InOut x` 执行 `x[i] = x[i] + 1`，128 个初始零元素，production cuTile 首次 `artifact.run(x)` 后首元素为 **97**，第二次调用只增加 **1**。97 是本次调优试跑次数的观察，不是固定承诺；问题是首次调用的可见增量不为 1。

当前 wrapper 直接将原 views 交给 exhaustive search，再用同一 views 执行 winner，见 `C:lib/Target/CuTile/Serialization/Serializer.cpp:479`、`:500`。runtime 没有状态隔离，见 `C:python/intent/runtime/artifact.py:31`。Triton serializer 同样未声明 restore/reset hook，见 `C:lib/Target/Triton/Serialization/Serializer.cpp:470`，属于同类明确路径风险；本轮的副作用动态复现只执行了 cuTile。

参考 Triton 的 `restore_value`/pre/post hook 为每次 trial 保留和恢复值，见 `R_T/python/triton/runtime/autotuner.py:59`、`:77`、`:150`。这说明试跑是正常 provider 能力，而隔离其可观察 effects 是调用方必须兑现的责任。

不能把结果直接推广到 TileLang：Intent 使用 `set_autotune_inputs`，其 ref 会 clone captured tensors，见 `C:lib/Target/TileLang/Serialization/Serializer.cpp:388`、`R_L/tilelang/autotuner/tuner.py:370`。副本是否在每次 trial 重置、是否保持 aliased view 的关系，仍需单独看目标调用契约；本轮不宣称 TileLang 已有相同 caller-buffer 污染。

最小方向：根据当前 effect/resource/alias facts 声明 trial state 与最终 invocation 的绑定，让 provider/runtime 隔离调优 effects。只给 benchmark 外层补一次 reset 不够；纯 Out 全覆盖与 InOut、atomic/read-dependent writes 必须分开。无需禁止 autotune，也无需让 DSL 作者写 provider hook。

## 5. Config：分离方向正确，但还可以更好调整

### 5.1 当前已有的权威分工

| 内容 | 当前 authority | 含义 |
|---|---|---|
| typed parameter、角色、候选域 | `C:include/Intent/Dialect/GPU/IR/GPUAttrs.td:67`；对应 dialect verifier | 参数参与真实 fragment、loop、grid；合法域不是 winner |
| shared 相关 tuples | `C:lib/Dialect/GPU/Transforms/MaterializeConfigTuples.cpp:367`、`:516` | 按当前 IR 分类、选择一组相关粒度 |
| provider options 与闭包 | Triton `Legalize.cpp:912`；cuTile `Legalize.cpp:2075`；TileLang `Legalize.cpp:335` | provider-specific warps/stages/threads/access/occupancy 与局部约束 |
| terminal config emission | 三个 Serializer 读取对应 closed config attribute | 打印已经形成的候选；不应重新选 shared block |
| winner | provider tuner/runtime cache | 运行时观察，不回写 canonical KIR |

这说明 05c 把 config 从 serializer 拆出是正确方向，而且基本已经落地；它没有自动解决 candidate policy 的可维护性，也没有解决第 4.4 节的 tuning effects。

当前 `profilesFor` 同文件混有：数值行、IR graph classification、correlation detection、候选投影和 profile index 选择。改变 shared 数值表要重编译 compiler，且会影响所有消费 shared tuples 的 providers。位置：`C:lib/Dialect/GPU/Transforms/MaterializeConfigTuples.cpp:236`、`:275`、`:367`、`:503`、`:618`。

### 5.2 建议的数据与逻辑分离方式

建议首先把**有限候选数据**与**判断逻辑**分开。数据只表达 profile 家族及相关 tuple；source-axis/ownership 分析、角色绑定、capability 匹配和 legality 保留在代码，并继续把结果写入同一 typed IR。不要引入可执行的配置脚本或新的策略语言。

| 方式 | 调整体验 | 重编译 | 判断 |
|---|---|---|---|
| 独立 `.def/.inc` 或普通 typed C++ 数据表 | 数值集中，主 pass 不再堆字面量 | 仍需重编译包含该表的目标 | 最小机械整理，适合主要诉求是代码清晰 |
| 编译调用读取独立 JSON 等有限数据表 | 调候选不用改 C++；显式输入路径，解析后进入既有 typed tuples | 不需重编译 C++，但需重新编译/调优受影响 kernel artifact | 更符合持续调表的需要；推荐作为后续维护方案评估 |
| 新建 TableGen schema/通用策略引擎 | 可扩展生成多套接口 | build-time 数据仍需重编译 | 当前收益不足，不建议因“成熟”而额外建设 |

数据例子只需类似 `contraction_narrow: [[128,128,32], [64,128,64]]` 的相关行；各列含义由已存在的 profile 类型定义。适用条件仍由 typed facts 选择家族。数据加载必须发生在 materialize config 的编译阶段，不能由 serializer 或 launch-time 表重新解释程序；无效输入明确报错，不静默回退。

这不需要引入版本号或迁移体系。实际选入的配置仍完整存在于 provider artifact，重新编译即可形成新的 candidate identity。若未来跨进程缓存 compiler 结果，才需要让该显式编译输入参与既有缓存身份；不在本轮扩建缓存系统。

参考并不要求所有表外置：Triton 的 Config 数据常直接写在 Python，autotuner 将 performance model、top-k、early prune 与运行机制分开，见 `R_T/python/triton/runtime/autotuner.py:21`。TileLang 的 `CompileArgs`、config dict 与 Roller Hint 分离参数和分析结果，见 `R_L/tilelang/autotuner/param.py:47`、`R_L/tilelang/carver/roller/hint.py:75`。可维护性的重点是职责和可调输入，不是文件扩展名。

### 5.3 一个已确认过强的合法性解释：occupancy

cuTile 当前对任意 `ResidentWorkers` program 固定 occupancy domain `{1}`，注释解释为大于 1 会描述不存在的 CTA，见 `C:lib/Target/CuTile/Transforms/Legalize.cpp:1393`。但 shared batched contraction 会把 resident worker 数设为 `2×computeUnits`，见 `C:lib/Dialect/GPU/Transforms/RefineProgramMapping.cpp:181`。

provider API 的 occupancy 是 expected active CTA/SM，默认 auto，并不创建或删除 logical program IDs，见 `R_C/_execution.py:74`。source persistent GEMM 同时具有 grid-stride loop 和 occupancy 候选，其 grid 与 hint 可以联合调整，见 `C:source/cutile/tilegym/gemm/dense/matmul.py:218`、`:356`。

因此固定 1 可以是当前 mapping 下的策略，但不是由“persistent”证明的唯一合法值。当前 verifier 还要求候选域逐项等于 `occupancyDomain`，见 `C:lib/Target/CuTile/Transforms/Legalize.cpp:2225`、`:2266`，把策略一致性与 provider legality 混在了一起。建议区分可接受值/结构约束和被选中的搜索预算；这不是建议无条件扩大候选，更不是声称 occupancy 2 必然更快。

Triton 的 `localOptionsFor` 同样是性能启发式，真实资源/form 检查在后面，见 `C:lib/Target/Triton/Transforms/Legalize.cpp:62`、`:997`。它也不应被描述成数学上唯一正确的 warps/stages 组合。

## 6. 性能到底多少来自 pass

目前不能从已有 CSV 给出 shared / provider / external compiler 的贡献百分比。它们存在交互：blocking 改变可用 primitive、访问 form 和候选资源；同一表值在不同物理程序上不是同一工作量。必要 lowering 若没有合法的“关闭版本”，关闭后的 failure 更不能算性能消融。

能够明确说出的责任是：Intent 形成逻辑程序到 block program 的 mapping、粒度、访问、状态与结构；provider-local 层形成各家需要的 source form 和参数；外部 Triton/TileLang/cuTile compiler 继续决定线程布局、MMA 实现、pipeline、同步及机器代码。参考 `R_T/third_party/nvidia/backend/compiler.py:273`、`:289` 和 `R_L/tilelang/cuda/pipeline.py:100`、`:125` 显示这些后续 passes 的实际顺序。Intent 没有复制它们，符合当前分层。

### 6.1 当前数字能说明什么

最近一次完整 cuTile 观察是 `a11b80a`、2026-09-05 14:28 的两张 `/tmp/r6-cutile-{5090,h100}-a11b80a.csv`；仓库正式 CSV 仍为 8 月 28 日。最新每张 37 项，5090 有 35 项数值 pass，其中 31 项 ratio≤1.05；H100 有 33 项数值 pass，其中 15 项 ratio≤1.05。这里仅计数表内观察，**不把它们升级为可比性已全面审定的成绩**。

旧表的 timeout/verification failure 不能用来描述本基线全部条目的当前状态；新表中的 `pass` 也不等于性能达标、精度契约相同或 shared correctness 已证明。前一轮将最终更新 CSV 一直拖到收尾，客观上使工作状态不可见；本调查保留该事实，但不替旧 change 修改成绩。

### 6.2 具体可比性问题

| 条目/问题 | 已核实差异 | 可作出的结论 |
|---|---|---|
| chunk gated-delta | 两边都是两个 cuTile kernels。source 另做 output transpose/contiguous/cast；generated 直接写 BF16 BTHV。source `_ct_mm` 对 f32 输入显式转 TF32，generated 保留 f32 contract | 不能把先前约 10.5× 单候选诊断归因于 shared mapping；它来自旧未提交试验，且精度/调用范围没有对齐，不是本轮性能数据 |
| Gemma prefill | adapter 对 source 明确 `use_autotune=False`，generated 走候选搜索 | 可记录固定 source 与 generated closure 的运行观察，不能称相同 candidate 契约下的编译器质量比较 |
| dense attention | source 的 search 调用没传 occupancy hints，之后却固定 `replace_hints(occupancy=2)` | config 字段存在不证明该维度真正被搜索；要读 actual launch/tuner 调用 |
| 最终 CSV | 只有 p50、ratio、status；没有受控 pass 前后程序、具体 winner、统一搜索条件 | 能说明该次端到端观察；无法据此拆分 pass 性能贡献 |

位置：chunk 的 `C:examples/repro/v2/providers/cutile/scan.py:90`、`C:source/cutile/tilegym/scan/gated_delta_chunk/chunk_gated_delta_rule.py:16`、`:396`、`:438`、`:464`；Gemma 的 `C:examples/repro/v2/providers/cutile/attention.py:292`；dense 的 `C:source/cutile/tilegym/attention/dense/attention.py:797`、`:821`；measurement 的 `C:examples/repro/v2/measurement.py:203`。

`.contiguous()` 也不是每次必然产生 copy：本来连续的输入可以原样返回。计时可能使用 CUDA graph 或普通 events，必须按具体 closure 解释，不能机械地把所有 Python 调用/分配成本相加。本轮没有重测全量，也没有报告新的性能提升百分比。

### 6.3 怎样获得足够的归因而不建新体系

一个具体性能问题只需要对齐其数值和调用契约，说明当前 shared-final IR 的工作分配及 provider form，再记录所选 candidate 和现有 production repro 结果；涉及两个合法程序的比较时，控制其它条件。不是所有 pass 都需要 ablation，更不需要为本轮造一套长期 profiling 框架。

同样，`ConstInt` specialization 与 `range` 自动展开不能混淆。cuTile 普通 range 会 lower 为运行时 ForOp，显式 `static_iter` 才在 frontend 展开，见 `R_C/_passes/ast2hir.py:707`、`R_C/_stub.py:4335`。此前把 chunk 超时直接解释成“ConstInt 导致 32 倍文本展开”没有成立。动态 metadata 是否更快需要受控比较；统一取消 constexpr 不能作为已证明的解决方案。

## 7. 对后续 cuTile、TileLang 和其他硬件的判断

### 7.1 已知事实与继续条件

| 类别 | 当前判断 | 对后续工作的实际影响 |
|---|---|---|
| 已成立基础 | §1.1 的已有复杂作者程序、canonical structured semantics、真实 shared GPU Program、provider 扩展同一程序 | 从已有能力改进常用计算入口与函数复用，不先把语言判断为只能表达简单计算 |
| 已复现 correctness 缺口 | 输入精度、signed floor/remainder、cuTile index 宽度、autotune effects | 受影响程序的正确性与相关性能比较不能宣称闭合；优先做局部 provider/runtime 修复 |
| 特定表达/provider 的 lowering 缺口 | §2.3 在 cuTile 编译调用下的 rank-1 dot/GEMV、直接 paired batch、多 reduction axes 最小例子 | 前两者停在 cuTile form，后两者停在 shared construction；保留具体限定，不能泛化为 Triton/GEMV 算法失败或覆盖掉已有成功 |
| 需要纠正的策略边界 | occupancy 的策略被解释为唯一 legality | 分离合法约束与预算选择，避免用过度收缩掩盖后端能力 |
| 已确认的 DSL 设计方向 | 以具名常用计算 op 为主要作者入口；复用完整等价的统一语义，保留通用算法构造 | 不是仅给 public contract 补说明；按 §2.6 的映射和 §7.2 的完整链路落地，不复制 backend 模板 |
| 规范收敛缺口 | frontend 退化 region 尚未归一回 reduce/scan | 在对应退化形式进入实现范围时落实；不以此删除真正有 slice-level 算法的 region |
| 维护机会 | config 数据组织、transformation group 的后置条件 | 值得有界改善，但不阻止继续调查已经合法运行的 kernel，也不要求重做抽象 |
| 尚无定论 | 任意新语料的 shared 组合覆盖、TileLang 完整数值/alias/tuning 行为、精确性能贡献 | 保留未知，不能作“以后只改 leaf”的保证 |
| 新硬件边界 | current device resolver 直接使用 CUDA driver，能力中有 NVIDIA compute capability；没有 CPU/RVV production target | AMD GPU 需要设备发现/capability/provider 契约接入；CPU/RVV 应有独立 execution-family lowering，不把 GPU topology 改名复用 |

新硬件的具体证据是 `C:python/intent/targets/gpu/device.py:34`、`:58`、`:83`。参考 Triton 区分 NVIDIA/AMD backend options 与 lowering，见 `R_T/third_party/nvidia/backend/compiler.py:127`、`R_T/third_party/amd/backend/compiler.py:84`。这是当前实现覆盖范围，不是证明 language 必须包含设备名，也不是要求现在建设所有目标。

本轮不主张先推翻现有 canonical/physical IR，也不再把“核心表示成立”解释成“不必改变 public DSL 的设计重心”。用户已确认具名 op 与通用语义复用的方向；具体 API 契约仍须在下一轮实现前写入对应 `doc/`。现行规范要求的数值/effect 修复不必等待重新设计；§2.3 的 shared 缺口则已说明后续不能只按 provider-local 工作估算。

### 7.2 下一轮据此行动的范围与完成条件

下一轮从 §1.1 的实际作者程序与 ref 接口出发，围绕**一批作者能直接写、并能实际 lowering 的计算 op 与可复用函数**推进。重点是减少已有写法中的机械负担、保留算法编程能力；不先展开无关 IR 整理或额外验证工程。下列是本报告的行动依据，不是本轮已经授权并完成的实现任务。

1. **先明确第一批 public 契约，再改入口。** 以 §2.6 的收缩类与 prefix 类为候选，选定具体 op 后，把签名、输入/输出 rank、batch/transpose、dtype/promotion、empty/NaN 和数值模式写入 `doc/dsl/` 对应章节。`matmul` 是否统一接收 rank-1/批量输入、是否做 batch broadcast，还是提供独立 dot/matvec/batch 入口，以及 scan 简写的命名/default，尚未由本报告定稿；这些必须在各 op 实现前明确，不能由某条当前能跑的 provider form 倒推。现有 `reduce.sum/max` 已是具名计算，应评估一致性而不是无条件再造同义入口。
2. **把公开操作正规化到现有语义。** 主要落点是 `python/intent/language/` 的公开声明和 `python/intent/frontend/lowering/intrinsics/` 的 binding/检查/归一。为普通用户自动产生由契约唯一确定的 paired axes、broadcast relation、builtin combine/identity，保留显式算法参数及 source diagnostics。等价映射成立就不新建 canonical op；同一 canonical 程序不因换一个公开名字而走另一 serializer/config policy。不要通过往 registry、source corpus 或 kernel-name matcher 中加条目“实现”语言 op。
3. **复用已成立路径，补齐当前切片实际需要的 lowering。** 先看相关现有 kernel 怎样表达、adapter 实际编译哪个函数、shared/provider 已形成什么程序；遇到 §2.3 那样的具体缺口再按其阶段修复。修复的是对应 typed 程序的通用关系和 realization，不是给临时 kernel 写专用模板。不能把某个 rank-1 cuTile form 的失败叫作“GEMV 不支持”，也不能因一个表达已产出 KIR 就宣称其目标路径完成。
4. **把已有语义错误纳入受影响路径。** §4 的 f32 输入精度、signed div/rem、index 宽度、autotune effects 会影响相应新旧程序；宣称该路径完成前必须兑现原数值/effect 契约。不得用新 op 的默认值、精度放宽、输入重置包装或改变容差掩盖错误，也不要求与选定 op 无关的所有后端问题先全部清零。
5. **保留算法编程与函数复用。** 选定 op 不仅能单独调用，也应在本来允许它的 `@intent.fn`/structured pure helper 中沿同一归一路径使用；已有 FA/Welford 等组合可通过库函数复用，不让普通使用者重复编写完整协议，也不增设 whole-operator 模板。generic contract、custom reduce/scan、region、普通 control 和明确读写保留各自职责；新组合按 §2.7 判断，不按算法名称选择或扩张语言机制。

调研和设计判断使用已有代码、调用链与 ref，不以重新跑 baseline、增加台账或验证记录为前提。下一轮真正改动实现时，才按仓库规则用必要的单条 production repro 完成验证，不新增 pytest、fixture 或长期验证框架。若 public 与 generic 写法等价，应能解释它们如何进入同一 canonical semantics；新名字本身不是性能优化，也不通过 hash 检验来证明语义相同。

config 外置、transformation-group 后置条件、occupancy policy/legality 分离仍按 §3、§5 的证据处理：只有影响当前实现切片时才一并动，不把所有维护事项绑成一次大重构。旧 cuTile change、TileLang 扩展和新硬件工作尚未自动恢复；本报告提供它们的继续条件，不冒充已获得下一轮实现或合并授权。

## 8. 动态复现与检查边界

独立编译器：`/tmp/intentdsl-foundations-build.GePTCn/tools/intent-compile/intent-compile`。临时 repro：`/tmp/intentdsl-foundations-repro.h6uHVZ/numerics.py`；它只调用 `intent.compile` 和 `artifact.run`，没有替换 generated source、裁剪候选、修改容差或绕过 provider。临时文件不进入仓库。

在调查 worktree 中的实际命令形式如下；表中的 provider/case 替换相应两参数即可：

```bash
PYTHONPATH=$PWD/python:$PWD/examples:$PWD \
  /home/kingdom/.venvs/intentdsl-cutile/bin/python \
  /tmp/intentdsl-foundations-repro.h6uHVZ/numerics.py triton contract \
  --compiler /tmp/intentdsl-foundations-build.GePTCn/tools/intent-compile/intent-compile
```

| provider / case | 规范预期 | 本轮实测 |
|---|---|---|
| triton / contract | 1.00146484375 | 1.0009765625，max_abs=0.00048828125 |
| cutile / contract | 1.00146484375 | 1.00146484375，max_abs=0 |
| triton / divrem | quotient=-2，remainder=1 | quotient=-1，remainder=-1 |
| cutile / wide-loop | 2147483648 | -2147483648 |
| cutile / inout | 首次 +1，第二次 +1 | 首次 +97，第二次 +1 |

repro 的核心 DSL 定义如下，输入和调用与上文一致，可在临时目录重新构造：

```python
import intent
import intent.language as I

@intent.kernel
def f32_contract(a: I.In[I.f32, ("M", "K")],
                 b: I.In[I.f32, ("K", "N")],
                 c: I.Out[I.f32, ("M", "N")]):
    M, K = a.shape
    _, N = b.shape
    m = I.domain(0, M)
    k = I.domain(0, K)
    n = I.domain(0, N)
    c[m, n] = I.contract(a[m, k], b[k, n],
                         reduce=((1, 0),), acc_dtype=I.f32)

@intent.kernel
def signed_divrem(a: I.In[I.i32, ("N",)],
                  b: I.In[I.i32, ("N",)],
                  quotient: I.Out[I.i32, ("N",)],
                  remainder: I.Out[I.i32, ("N",)]):
    n = I.domain(0, a.shape[0])
    quotient[n] = a[n] // b[n]
    remainder[n] = a[n] % b[n]

@intent.kernel
def wide_loop(origin: I.In[I.i64, (1,)], output: I.Out[I.i64, (1,)]):
    begin = origin[0]
    output[0] = I.cast(0, I.i64)
    for position in I.domain(begin, begin + 1):
        output[0] = I.cast(position, I.i64)

@intent.kernel
def inout_increment(x: I.InOut[I.i32, ("N",)]):
    for i in I.parallel(I.domain(0, x.shape[0])):
        x[i] = x[i] + 1
```

contract 输入为 `A=zeros(64,64,f32)`、`B=zeros_like(A)`，随后 `A[:,0]=1+3*2**-11`、`B[0,:]=1`；divrem 输入为 i32 `[-3]`、`[2]`；wide-loop 输入为 i64 `[2**31]`；inout 输入为 128 个 i32 零。输出通过 `artifact.run` 取得或从原 InOut buffer 读取，完成 CUDA synchronize 后比较。首次调用保留默认 tuning，因此能观察第 4.4 节的错误。

此外，第 3.2 节的正向运行使用现有 `C:examples/kernels/contraction/gemm.py:39`，经 `lower_to_mlir`、`compile_shared_gpu`、`compile` 输出三个层级，再运行 `128×128` BF16 全一输入，max_abs=0。

本调查没有做双机全量、性能消融、TileLang JIT 或新硬件运行，没有 pytest/fixture，也没有修改 compiler 来增加调试入口。此节保留的是前两次调查候选的数值与 lowering 事实；本次最终落稿只依据已有 kernel、CSV、调用链和 ref 完成设计结论，不新增运行或验证记录，也未实现表中的新 public 名字。调查验收通过不代表这些缺陷已修复或候选 API 已可用。

### 作者表面补充 repro

§2.3 的实际脚本为 `/tmp/intentdsl-foundations-repro.h6uHVZ/contract_forms.py`，同一 worktree、compiler 与 cuTile 环境，未改候选、数值模式或后端源码。其五个 kernel 定义如下（shape 和 domain 常量均为本次输入，不是语言新增限制）：

```python
@intent.kernel
def vector_dot(a: I.In[I.bf16, (64,)], b: I.In[I.bf16, (64,)],
               c: I.Out[I.f32, (1,)]):
    k = I.domain(0, 64)
    c[0] = I.contract(a[k], b[k], reduce=((0, 0),), acc_dtype=I.f32)

@intent.kernel
def matrix_vector(a: I.In[I.bf16, (32, 64)], b: I.In[I.bf16, (64,)],
                  c: I.Out[I.f32, (32,)]):
    m = I.domain(0, 32)
    k = I.domain(0, 64)
    c[m] = I.contract(a[m, k], b[k], reduce=((1, 0),), acc_dtype=I.f32)

@intent.kernel
def matrix_transposed(a: I.In[I.bf16, (64, 32)], b: I.In[I.bf16, (64, 32)],
                      c: I.Out[I.f32, (32, 32)]):
    m = I.domain(0, 32)
    n = I.domain(0, 32)
    k = I.domain(0, 64)
    c[m, n] = I.contract(a[k, m], b[k, n], reduce=((0, 0),), acc_dtype=I.f32)

@intent.kernel
def multi_reduce(a: I.In[I.bf16, (32, 2, 32)], b: I.In[I.bf16, (2, 32, 32)],
                 c: I.Out[I.f32, (32, 32)]):
    m = I.domain(0, 32)
    n = I.domain(0, 32)
    p = I.domain(0, 2)
    q = I.domain(0, 32)
    c[m, n] = I.contract(a[m, p, q], b[p, q, n],
                         reduce=((1, 0), (2, 1)), acc_dtype=I.f32)

@intent.kernel
def paired_batch(a: I.In[I.bf16, (2, 32, 64)], b: I.In[I.bf16, (2, 64, 32)],
                 c: I.Out[I.f32, (2, 32, 32)]):
    batch = I.domain(0, 2)
    m = I.domain(0, 32)
    n = I.domain(0, 32)
    k = I.domain(0, 64)
    c[batch, m, n] = I.contract(a[batch, m, k], b[batch, k, n],
                               reduce=((2, 1),), batch=((0, 0),), acc_dtype=I.f32)
```

单条手动 repro 命令会逐项执行，失败不被替换成通过；各进程原样退出，诊断与 generated source 留在工作区外日志：

```bash
for form in dot gemv transpose multi batch; do
  PYTHONPATH=$PWD/python:$PWD/examples:$PWD \
    /home/kingdom/.venvs/intentdsl-cutile/bin/python \
    /tmp/intentdsl-foundations-repro.h6uHVZ/contract_forms.py "$form" \
    --compiler /tmp/intentdsl-foundations-build.GePTCn/tools/intent-compile/intent-compile \
    > /tmp/intentdsl-foundations-repro.h6uHVZ/contract-$form.log 2>&1
  printf '%s exit=%s\n' "$form" "$?"
done
```

五项进程退出码依次为 `1,1,0,1,1`。每项先打印 `intent.lower_to_mlir(kernel)`；compile 成功时再打印 `artifact.source`，创建表列 shape 的 BF16 全一输入，调用 `artifact.run`、CUDA synchronize，并比较 output shape 与全 64 的 f32 结果。唯一成功的转置 GEMM 输出 `(32,32)`、`max_abs=0.0`。四个失败项停在编译期，没有生成可运行 artifact，因而没有数值结果；这里只把它们认定为已复现的 lowering 缺口，不把错误诊断当成功的后端数值验证，也不声称全一输入证明一般矩阵正确性。
