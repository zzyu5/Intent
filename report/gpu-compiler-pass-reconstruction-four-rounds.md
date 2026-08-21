# IntentDSL GPU 编译器四轮重构推进报告

## 1. 报告范围

本报告对应以下四轮工作：

1. 重新闭合 DSL 语义与 Physical Program 骨架；
2. 让 shared GPU realization 成为 pass pipeline；
3. 把 provider leaf 收敛成 provider passes 与 terminal translation；
4. 用真实 kernel 接纳能力收口整条编译链。

共同的设计依据是：

```text
report/compiler-pass-v2-reconstruction-basis.md
```

本报告不以新增文件数、pass 数量或 registry 数量判断完成度，而是逐层回答：

- 作者的算法语义现在由谁持有；
- Physical Program 是否已经成为可变换、可验证的物理程序；
- shared 与 provider 决定是否在 source 生成前显式存在；
- terminal translation 是否只剩机械翻译；
- 真实 kernel 是否沿唯一主链完成 JIT、launch 和数值对照。

## 2. 总结论

四轮并非全部完整达到原要求。

当前状态是一个可以构建、可以运行真实纵向 repro、并已经具有 shared/provider pass 骨架的过渡编译器，而不是已经彻底闭合的最终 Physical Program 架构。

| 轮次 | 结论 | 实际完成度 |
|---|---|---|
| 第一轮 | 部分完成 | 建立了局部 physical operation 与更严格 verifier，但没有完成 DSL 语义重审，也没有把整个 executable structure 从 KIR clone 中独立出来 |
| 第二轮 | 实质推进但未闭合 | 建成 construct/refinement/verify pipeline，三类真实规则成为独立 pass；大部分初始结构仍由一次性 Build 构造 |
| 第三轮 | 实质推进但未闭合 | 三个 provider 都有本地 form pass 和 verifier，部分 leaf 决定已经前移；大型 materializer 仍承担相当多 provider program 构造 |
| 第四轮 | 一个纵向集合完整完成 | xFormers split-K paged attention 真实通过两段 DSL、两段 provider JIT、顺序 launch 和数值对照；没有伪造其余 inventory 的覆盖率 |

因此，四轮的主要成果是把原来集中在 constructor 和 emitter 中的编译责任第一次切出可运行的 pass 边界，并用一个结构复杂的真实 multi-kernel entry 找到并修正这套边界中的错误。尚未完成的是：让 Physical Program 本身完整承载 executable structure，以及让 terminal translator 真正退化成只消费 provider IR 的机械打印器。

## 3. 当前真实编译链

当前实现实际运行的链条是：

```text
Python DSL
  → canonical Kernel IR
  → ConstructPhysicalProgramPass
      → intent_plan.program
      → Plan decisions
      → 一份标记为 physical 的 KIR function clone
      → 少量已物化 physical op（当前包括 intent_plan.unary）
  → VerifyPhysicalProgramPass
  → RefinePrivateBufferResidencyPass
  → VerifyPhysicalProgramPass
  → RefinePersistentTraversalPass
  → VerifyPhysicalProgramPass
  → RefineBoundaryNeutralizationPass
  → VerifyPhysicalProgramPass
  → provider-local ProgramForms pass
  → provider verifier
  → MaterializeTargetProgramPass
      → provider source string
      → intent_plan.target_program
  → terminal translation 输出 source
  → Triton / cuTile / TileLang compiler/JIT
  → GPU launch 与数值比较
```

这条链已经只有一个公共入口，没有旧 compiler fallback，也没有 corpus 专用第二条路径。但是，`intent_plan.program` 内的 physical function 仍主要是 canonical KIR 的 clone，shared/provider analysis 仍会分析这份 clone。它不是旧版那种两套完全独立的 executable pipeline，但也还不是一份脱离 KIR executable structure、自己完整承载 loop/value/access/structured-op 的成熟 Physical Program。

## 4. 第一轮：DSL 语义与 Physical Program 骨架

### 4.1 实际完成的内容

第一轮没有修改 public DSL 或 canonical Intent dialect。没有删除或重新定义 `partition`、`parallel`、`state_stream`、`contract` 等语言构造，也没有为保住旧例子增加兼容 surface。

真正落地的是一个局部的 Physical Program 纵向切片：

- 新增 `intent_plan.unary`，把 unary 的输入、结果、canonical node 和 semantic 作为 typed physical SSA operation 保存；
- Physical Program 构造时，把 physical function clone 中对应的 `intent.unary` 替换为 `intent_plan.unary`；
- verifier 拒绝 physical function 中残留的 `intent.unary`，并要求 physical unary node 有对应的 pointwise decision；
- 初始私有 buffer residency 变成保守、完整、合法的 baseline，后续 pass 可以再 refinement；
- region argument 到 selected range 的绑定继续由 `intent_plan.region_binding` 显式持有，leaf 找不到精确绑定就诊断，不走 extent/name 猜测。

主要实现位置：

- `include/Intent/Dialect/Plan/IR/PlanOps.td`
- `lib/Dialect/Plan/IR/PlanOps.cpp`
- `lib/Target/GPU/Realization/Plan/Build.cpp`
- `include/Intent/Target/Common/Lowering/ProgramAnalysis.h`

### 4.2 没有完成的内容

第一轮原目标要求重新判断 public DSL 哪些属于算法语义、哪些混入 GPU blocking，并让 Physical Program 从一开始就是完整、合法、可改写的 executable program。

当前没有完成这两点：

- public DSL/KIR 没有经过新的语义收敛；这一轮只能说明没有随意改语言，不能说明语言边界已经重新证明；
- `buildPhysicalProgram` 仍先生成 decisions，再 clone canonical function；除 unary 外，大多数 operation 仍保留 canonical 形态；
- verifier 只对已经迁移的 physical unary 建立了不变量，没有要求其它 canonical operation 全部物理化；
- 后续 pass 仍通过 physical function clone 重建 `KernelModel` 和 `KernelFacts`。

所以第一轮完成的是“Physical Program 可变换骨架的第一条真实纵向切片”，不是“完整 Physical Program 已经闭合”。

## 5. 第二轮：shared GPU realization pass pipeline

### 5.1 建立的 pipeline

shared GPU 主链现在显式执行：

1. 构造 Physical Program；
2. 验证；
3. refinement private buffer residency；
4. 验证；
5. refinement persistent traversal；
6. 验证；
7. refinement boundary neutralization；
8. 验证。

实现入口位于：

```text
lib/Target/GPU/Transforms/Passes.cpp
```

三个 refinement pass 都通过 `PhysicalProgramAnalysis::compute` 重新读取当前 program、physical function、axes、ranges、stages 和派生 Kernel facts，而不是持有 constructor 时代的指针或缓存。这保证前一个 pass 改写 IR 后，后一个 pass 看到的是当前事实。

### 5.2 三类真实 physical refinement

#### Private buffer residency

位置：

```text
lib/Target/GPU/Transforms/Value/PrivateBufferResidency.cpp
```

它根据 logical buffer 的 owner、lifetime/element count 与设备寄存器预算，把初始 conservative workspace refinement 成 `private_scalar_array`、`private_vector` 或继续保持 workspace。决定写回 `intent_plan.buffer`，不是由三个 leaf 各自重算。

#### Persistent traversal

位置：

```text
lib/Target/GPU/Transforms/Execution/PersistentTraversal.cpp
```

它读取 contraction、parallel ownership、ragged relation 和每个 ownership range 是否跨多个 tile，决定 launch 是否 persistent，并同步改写 worker/fold/reuse。原来嵌在 axis assignment 中的完整判断已经删除，初始构造只给合法非 persistent baseline。

#### Boundary neutralization

位置：

```text
lib/Target/GPU/Transforms/Access/BoundaryNeutralization.cpp
```

它沿 producer-consumer use-def 证明越界通道是否在产生可观察结果前被中和，并把结果写回 transfer 的 `consumer_neutralized`。三个 provider 都消费这项事实来选择是否物化 validity/bounds，而不是把证明复制三遍。

### 5.3 第二轮仍未闭合的部分

- `buildPhysicalProgram` 仍负责 axes/ranges、transfer、reduction、scan、contract、stage、stream、padding 等绝大多数初始结构；
- 当前 `PhysicalProgramAnalysis` 是每个 pass 手动重算的 typed helper，不是接入 MLIR AnalysisManager、带 preserved/invalidation contract 的正式 analysis；
- 三个 refinement 的算法来自原有 constructor 中已经验证过的真实规则，架构上完成了职责分离，但并不等于新性能算法；
- structured-operation、value flow、access topology 的大部分决定仍没有拆成可独立重写的 passes。

因此第二轮建立了真实 pass pipeline，而没有完成“所有 shared realization 主要发生在 passes 中”的最终目标。

## 6. 第三轮：provider-local passes 与 terminal translation

### 6.1 三个 provider 的共同骨架

Triton、cuTile、TileLang 都新增了 provider-local `ProgramForms` pass：

```text
lib/Target/Triton/Lowering/Transforms/ProgramForms.cpp
lib/Target/CuTile/Lowering/Transforms/ProgramForms.cpp
lib/Target/TileLang/Lowering/Transforms/ProgramForms.cpp
```

公共 driver 的顺序现在是：

```text
provider form passes
  → provider verifier
  → MaterializeTargetProgramPass
  → intent_plan.target_program
  → terminal source extraction
```

provider decisions 在 source string 生成前写到 Physical Program operation 的 provider attrs，并由 provider verifier 检查存在性和取值范围。

### 6.2 已前移的 provider decisions

#### Triton

- row launch form：`configured` 或 `generic`；
- materializer 不再重新 walk 整个 kernel 决定是否配置 row-vector launch，而只读取 provider decision。

Triton 的 runtime ABI dimension 判断使用 `StringRef::getAsInteger` 的失败返回值识别符号维度；这里经过审计，现有条件不是整数判断反转。

#### cuTile

- transfer access：load/store/gather/scatter；
- explicit bounds；
- row occupancy 是否委托 tuner；
- gather spelling 是否成为 target-local 候选。

这些决定由 cuTile pass 读取 shared Plan、typed index facts 和 capability 后写回；materializer 只消费 attrs。

#### TileLang

- pointwise form；
- transfer 的 access、bulk-copy/parallel-elements、bounds、defer；
- contraction operands 是否需要 isolation；
- program tile symmetry；
- row launch；
- GEMM warp policy 是否进入 provider tuning。

这些是 TileLang 程序模型需要显式表达、但不应在 terminal source 拼接时才决定的 provider-local facts。

### 6.3 第三轮仍未闭合的部分

第三轮没有把三个大型 materializer 彻底变成薄 translator：

- Triton 目前主要前移 row launch；packed-scalar projection、ragged/stream index、mask、stage replay 等仍在 materializer/handler 中组合；
- cuTile 的 `indexTuple`、`tileShape` 仍根据 relation、rank 和 selected access ranges形成具体 provider expression；
- TileLang 的 wrapper/grid、persistent/reused-axis 与 buffer/copy 生成仍集中在 materializer；
- provider passes 仍分析 physical function 中的 KIR clone，因此它们的一部分 form selection 依旧依赖 KIR operation/use-def；
- 当前 terminal translation 本身只是读取 `intent_plan.target_program.source`，真正的 source 构造仍发生在 `MaterializeTargetProgramPass` 调用的大型 provider materializer 中。

其中一部分厚代码是合法的目标语法投影，不能只因行数多就上移；但仍会改变 provider program structure 的判断，还没有全部获得显式 provider IR 位置。第三轮完成的是“provider decision pass 的真实起点”，不是“leaf 已经只剩拼写”。

## 7. 第四轮：真实 split-K paged attention 纵向闭合

### 7.1 为什么选择它

source inventory 中的 xFormers split-K paged attention 同时包含：

- paged KV 间接读取；
- GQA 多对一 head mapping；
- nested ordered state streams；
- split-K partial workspace；
- 作者主导的两个 kernel 调用；
- weighted LSE merge。

它比再接一个同形状 GEMM 更能检验 KIR、Physical Program、stream/ragged range、provider lowering 和 multi-kernel runtime boundary。

### 7.2 按 upstream 算法建立 DSL

接入由两份 DSL kernel 组成：

1. `paged_gqa_decode_partials` 生成每个 split 的 partial output 和 LSE；
2. `splitk_attention_weighted_sum_reduce` 按 xFormers 的结构执行 `exp2` 权重、逐元素乘法与 `reduce.sum`。

没有把第二段写成小矩阵 `contract`。现有通用 reducer 使用 contraction，但 upstream `_splitK_reduce` 实际使用 pointwise multiply + `tl.sum`；直接复用会在 Triton 上触发小 K `tl.dot` 的能力限制，也会把不同算法写法当成同一个 baseline。

partial 输出的第三维改为显式 `SPLITS` constexpr。`split_offsets.shape[0]` 是 `B*SPLITS+1`，不能作为每个 batch 输出 split 数的权威来源。

### 7.3 真实 source 接线

source runtime 邻接 wrapper 调用未改写的 xFormers：

- `_get_splitk_kernel(1)`；
- `_splitK_reduce`。

wrapper 负责真实 case 的 page/block ABI、partial workspace、两次 launch 和最后输出视图。当前固定 case 是：

```text
B16-QH32-KVH8-S8192-D128-page16-split8-fp16
```

generated 与 source 都执行 partial + reduction 两次 GPU launch。source 的 `NUM_PROGRAMS_DIM2_CONST` 已绑定为真实 split 数，避免关闭 upstream 的 split early-exit 判断。

### 7.4 真实 kernel 暴露并修复的 shared/provider 问题

#### 错误一：padding consumer 没有 index relation

Triton validity emitter 原来无条件要求 consumer 带 `intent.index`，普通 pointwise `intent.mask` 因此在 provider program 阶段失败。

修复后：

- 普通 tensor validity 直接使用 Plan 的 domain binding；
- 只有 packed-scalar validity 真正需要额外 physical-axis projection；
- 缺 projection 时明确诊断，不猜测。

#### 错误二：block extent 覆盖了别的 range purpose

同一个逻辑 `PS` 同时参与 contraction 和 ordered stream。错误实现只要看到 `PS` 有 `block_extent`，就把 stream 已选的 `BLOCK_SIZE_C` 改回 `next_power_of_2(PS)`，造成 operand 有 32/64 个物理元素、reshape 却要求 16 个元素。

修复后：

- 只有 row-vector range 使用 logical block-extent rounding；
- stream、reduction、ownership、access 等 range 使用 Physical Program 中各自 selected tile；
- Triton、cuTile、TileLang 的消费判据保持一致；
- 没有给 paged attention 或 split-K 写名字特例。

#### 错误三：失败阶段被压成 compile_failed

compiler、runner 和 measurement 现在区分：

- frontend/KIR；
- Kernel IR parse；
- Physical Program construction；
- Physical Program verification；
- provider program；
- provider verification；
- terminal translation/output；
- generated source materialization；
- provider JIT/initial launch；
- launcher preparation；
- source provider JIT/initial launch；
- generated/source launch；
- generated/source benchmark；
- numerical comparison；
- adapter preparation。

compiler executable 缺失或无法执行也被归到 `compiler_invocation`，不会再被 runner 错标成 adapter 问题。

### 7.5 唯一端到端验证

执行命令：

```bash
./examples/run/baseline-v2.sh triton /tmp/intentdsl-round4-splitk.csv splitk_paged_attention
```

结果：

| kernel | generated p50 | source p50 | ratio | status |
|---|---:|---:|---:|---|
| splitk_paged_attention | 0.480688 ms | 0.423152 ms | 1.135970 | pass |

这条结果证明当前固定 case 完成了 generated/source JIT、两段 launch、输出结构/类型检查和数值容差比较。它不证明全量矩阵通过，也不证明 cuTile/TileLang 已经复验。

generated 仍比 xFormers source 慢约 13.6%。本轮目标是检验接纳链和架构边界，没有用扩大 tuner、缩小 scope 或更换算法隐藏这项差距。

## 8. JIT、tuner 与 runtime 的边界

split-K 接入过程中先后出现 provider-program failure、provider JIT reshape failure 和 provider JIT primitive-limit failure。新的错误分层证明这三者不是同一种失败。

这次对 `BLOCK_SIZE_C` 的判断也说明：

- 32/64 大于逻辑 `PS=16` 并不自动构成非法 tuner candidate；padded physical tile 可以合法覆盖较小逻辑域；
- 真正错误是 translator 用逻辑 shape 重建 reshape extent，没有消费 range-specific physical tile；
- 该问题应修 Plan range 的消费边界，不能靠过滤候选掩盖；
- upstream reducer 使用 `tl.sum`，而不是小 K `tl.dot`，因此算法对齐优先于用 tuner 绕过 primitive 限制。

当前仍然没有新增 structural autotune 或 cost model。固定输入生成一份 provider source structure，数值 tile/warp/stage 候选继续交给 provider tuner。

## 9. 对四轮约束的核对

### 没有发生的违规

- 没有按 kernel 名称在 shared/provider pass 中选择编译路径；
- 没有 whole-kernel matcher 或手写完整模板替换；
- 没有 legacy fallback 或兼容开关；
- 没有为了接 baseline 修改 canonical 算法为更容易编译的版本；
- 没有为第四轮建立 test 目录、pytest、fixture 或额外测试脚手架；
- 没有用 registry 状态伪造未运行 entry 的通过；
- 没有全量重跑或回写固定 baseline CSV。

canonical operation 名称仍用于逐 op handler dispatch，这属于 dialect lowering 的正常分派，不等于按 kernel 名称分类。

### 仍然存在的架构欠账

1. Physical Program 的 physical function 仍主要是 KIR clone；
2. 除 unary 外，大部分 executable operation 还没有 physical dialect 形态；
3. Build 仍是大部分 shared structure 的首要构造者；
4. provider passes 只前移了部分 form decisions；
5. 三个 materializer 仍会读取 KIR/use-def 并形成部分 provider structure；
6. terminal translation 尚未成为只消费完整 provider IR 的独立机械 translator；
7. 旧固定表中的 35 个宽泛 `compile_failed` 没有重新全量运行，因此只是历史状态，不会自动变成新的精确阶段标签；
8. source inventory 从 119 个 entry 增加到 registry 92 个 entry，仍有 27 个没有真实 DSL/adapter 纵向接入；
9. 第四轮只验证 Triton split-K；cuTile 和 TileLang 没有因这一轮自动获得通过证据；
10. split-K generated 相对 source 的 13.6% 性能差距仍然存在。

## 10. 没有强行接入的真实算法

审计过 causal Conv1D backward，但没有直接复用现有 DSL：

- upstream case 是 bf16 输入、fp32 weight/bias、SiLU backward；
- upstream 会先重算 pre-activation，再在 backward 中应用 SiLU derivative；
- 现有 DSL 是无 activation、无 bias、f16 的线性 causal convolution backward；
- upstream partial workspace 按 batch×time-chunk，现有 DSL 按 batch 保存全序列 partial。

直接登记会比较不同算法和不同调用结构，因此没有用 adapter 把它冒充成已接入。

Mamba3 SISO sequence forward 也没有被现有 step kernel 冒充：已有 Core 基本能够表达 contract、ordered recurrence、state 和 elementwise math，但完整 sequence 算法还需要新的作者 DSL 编排，当前没有运行证据。

## 11. 构建与证据边界

已完成：

- LLVM/MLIR 20 下完整 CMake configure、C++ build 和链接；
- `intent-compile` 与 `intent-opt` 生成成功；
- `git diff --check` 通过；
- 一个真实 Triton multi-kernel entry 完成端到端数值运行。

没有声称：

- 四轮后的全量 baseline 通过；
- 三个 provider 全量数值正确；
- 当前所有 source inventory 已接入；
- terminal translator 已经完全机械化；
- Physical Program 已经达到最终冻结形态。

## 12. 当前阶段定位

这四轮把编译器从“constructor + 三个大型 emitter”推进到了“可验证 Physical Program baseline + shared refinement passes + provider form passes + terminal source materialization”的状态。

它已经能够用真实 kernel 暴露并修正跨层责任错误；这证明新骨架不是纯目录重构。但第一至第三轮尚未完成设计依据要求的最终 authority 收敛，因此当前最准确的定位是：

> Pass-based GPU compiler skeleton 已经真实运行；完整 executable Physical Program 和薄 terminal translation 仍在构建中。
