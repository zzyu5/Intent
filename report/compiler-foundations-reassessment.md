# IntentDSL 编译基础复审

## 1. 结论与适用范围

当前 Intent 是一个已经做了实质工作的结构化算子 kernel 编译器。它从作者未写 program id、物理 block 和访问 mask 的逻辑程序，形成 program mapping、参数化 fragments、分块循环、访问和 accumulator，再交给 provider 编译器。因此不能把它归为按算子名称套模板的发射器；也不能据此宣称它的跨 provider 语义和性能归因已经闭合。

本轮没有发现必须推翻 domain / structured operation / KIR / shared GPU Program 这些核心抽象的证据。发现的优先问题是**实现没有完全兑现已经写下的语言语义**：Triton f32 contraction 输入精度、Triton signed division/remainder、cuTile logical index 宽度，以及 cuTile autotune 对调用者可变输入的污染，均已通过 production 路径复现。它们需要修复，但不要求新建一套语言或 GPU IR。

维护上的主要问题也很具体：真实 transformations 大量包在名为 construction 的阶段中，类型/关系修补依靠手工编排；config 已从 serializer 分离，但数值表、适用条件和搜索策略仍混在 C++ 中；部分策略又被 verifier 固化成了唯一允许的候选域。这里应改善职责和可维护性，不能以新增限制取得“干净”结论。

调查基线为已提交的 `a11b80a`，工作区为 `comet/compiler-foundations-reassessment`。旧 cuTile change 继续暂停；原工作区未提交的 `StatefulPointwise` 和 metadata `ConstInt → int` 试验未纳入本轮编译器。没有修改生产实现、`doc/` 或 baseline CSV。

本文位置约定：`C:` 表示本仓库；`R_T:` 表示 `/home/kingdom/phdworks/ref/triton`；`R_L:` 表示 `/home/kingdom/phdworks/ref/tilelang`；`R_C:` 表示本机 `/home/kingdom/.venvs/intentdsl-cutile/lib/python3.10/site-packages/cuda/tile`。`source/` 是 kernel corpus，与这些真实 compiler 实现分开使用。此前因仅在项目内寻找 `ref/` 而判断参考源码不存在，是定位错误。

动态复现使用独立构建的本基线编译器、RTX 5090D、Torch `2.13.0+cu130`、Triton `3.7.1`、cuda-tile `1.5.0`。这是环境记录，不是引入版本管理。真实 ref checkout 用于设计和实现对照，不假定它与安装包逐行相同；下文明确区分实测结论和静态风险。

## 2. DSL：保留什么，改善什么

### 2.1 `contract` 名称成立，常见调用可以更友好

`I.contract` 保存任意显式 paired reduction/batch axes，比普通矩阵乘更一般。当前调用要写 `reduce=((1, 0),), acc_dtype=I.f32`，对 GEMM 确实显得接近 IR，但这不说明 contraction 抽象错误。真实契约在 `C:python/intent/frontend/lowering/intrinsics/structured.py:553`；常见作者调用见 `C:examples/kernels/contraction/gemm.py:27`。

参考的 `tl.dot` 具有独立 accumulator/input-precision 参数，核心仍是最后两轴矩阵乘；当前 ref 还可将前导 batch 维 flatten。`T.gemm` 则写入显式 C buffer，并有 transpose、policy、clear-accum 等参数。二者的输入抽象都比 Intent 低，不能直接拿它们的短名字替换任意 paired-axis contract。对照：`R_T/python/triton/language/core.py:2215`、`:2260`；`R_L/tilelang/language/gemm_op.py:148`。

建议保留 generic `contract`。若常见矩阵写法的使用负担值得改善，可增加明确限定最后两轴矩阵语义的 surface `matmul`，机械归一到同一个 contract；不要从附近索引、变量名或碰巧相等的 shape 猜 reduction axis。这属于小范围 public surface 选择，不是本轮自动实施项。

### 2.2 更确定的问题是签名不可发现和固定参数冗余

当前 `Intrinsic.__call__` 只有 `(*args, **kwargs)`；真实参数名和 required/default 契约散在 AST lowering 的 `bind_call` 中。读 Python API、IDE 或 `inspect.signature` 不能直接知道 `I.contract` 怎么调用。对照 Triton/TileLang 的显式 Python 签名与文档，这是比“名字不像 DSL”更客观的可用性问题。位置：`C:python/intent/language/builtins.py:9`、`C:python/intent/frontend/lowering/intrinsics/common.py:37`；参考同上。

`scaled_contract` 采用明确的 closed positional schema，这个选择可以保留；但实现强制两侧 group size 相等、reduction 固定为 `((1,0),(2,1))`、batch 为空，同时又要求调用者反复写出其中固定的参数。可评估 surface 使用一个 `group_size`、固定轴约定内置到该专用入口，canonical schema 保持不变。位置：`C:python/intent/frontend/lowering/intrinsics/structured.py:574`、`:619`；对照 `R_T/python/triton/language/core.py:2291` 的 `dot_scaled` 参数结构。这里是可简化表达，不是已证实 correctness bug。

推荐先把 public signature、参数说明和 frontend binding 组织成同一份 surface 契约，再处理常见写法。若只加一份与 lowering 独立维护的 IDE stub，仍有双份签名漂移的问题。

### 2.3 不应为了短写法删除的语义

`region_fold/region_scan` 的 summarize、combine、identity、apply、emit、initial state 并非全部是仪式参数：transition summary 与 state 一般不是同一 schema，不能默认把 identity 当 initial state。captures 必须显式成为依赖；`combine_operands` 与 region `operands` 的命名可统一评估，但该依赖关系不能删除。位置：`C:python/intent/frontend/lowering/intrinsics/structured.py:284`、`:393`、`:451`；现行语义：`C:doc/dsl/core.md:158`。

参考 TileLang 的 reducer API 也区分 allocation、init、update、finalize，因为 epoch、状态与结果不同；不能据其调用更短就删掉 Intent 的 transition/state 区分。对照：`R_L/tilelang/language/allocate.py:287`、`R_L/tilelang/transform/__init__.py:395`。

generic reduce 与 `reduce.sum/max`、generic sparse contract 与 2:4 shorthand 的并存，本身不等于两个 executable paths。关键是是否归一到相同 canonical operation；当前 frontend 各入口最终发出同类 canonical op，不能仅数 surface 入口数量判定架构分裂。可用性调整无需恢复已废弃的兼容层或创建新的抽象家族。

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

| 类别 | 当前判断 | 对后续工作的实际影响 |
|---|---|---|
| 已成立基础 | canonical structured semantics、真实 shared GPU Program、provider 扩展同一程序、typed configs | 保留主架构，继续在现有路径修正和补齐 |
| 已复现 correctness 缺口 | 输入精度、signed floor/remainder、cuTile index 宽度、autotune effects | 受影响程序的正确性与相关性能比较不能宣称闭合；优先做局部 provider/runtime 修复 |
| 需要纠正的策略边界 | occupancy 的策略被解释为唯一 legality | 分离合法约束与预算选择，避免用过度收缩掩盖后端能力 |
| 维护机会 | public signatures、固定参数冗余、config 数据组织、transformation group 的后置条件 | 值得有界改善，但不阻止继续调查已经合法运行的 kernel，也不要求重做抽象 |
| 尚无定论 | 任意新语料的 shared 组合覆盖、TileLang 完整数值/alias/tuning 行为、精确性能贡献 | 保留未知，不能作“以后只改 leaf”的保证 |
| 新硬件边界 | current device resolver 直接使用 CUDA driver，能力中有 NVIDIA compute capability；没有 CPU/RVV production target | AMD GPU 需要设备发现/capability/provider 契约接入；CPU/RVV 应有独立 execution-family lowering，不把 GPU topology 改名复用 |

新硬件的具体证据是 `C:python/intent/targets/gpu/device.py:34`、`:58`、`:83`。参考 Triton 区分 NVIDIA/AMD backend options 与 lowering，见 `R_T/third_party/nvidia/backend/compiler.py:127`、`R_T/third_party/amd/backend/compiler.py:84`。这是当前实现覆盖范围，不是证明 language 必须包含设备名，也不是要求现在建设所有目标。

本轮建议不做核心抽象重构。需要用户讨论的语言选择只有在后续确实希望新增明确的近似数值模式或修改 public surface 时才出现；先修复现行规范要求的行为不需要重新设计语义。cuTile/TileLang 后续主要工作可以继续落在 provider-local 层，但本轮证据不支持“shared 不会再有实质修改”的承诺。

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

本轮没有做双机全量、性能消融、TileLang JIT 或新硬件运行，没有 pytest/fixture，也没有修改 compiler 来增加调试入口。通过独立构建、静态 current/ref 对照与这些有明确问题的 production repro，报告完成调查；发现实现缺陷与调查验收通过并不矛盾，后者不代表缺陷已经修复。
