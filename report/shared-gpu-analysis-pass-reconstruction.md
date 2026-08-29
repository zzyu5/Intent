# Shared GPU analysis 与 pass authority 重构报告

## 结论

本轮闭合了 canonical KIR 之后的 shared GPU compiler 横向闸门：construction 直接产生完整、可执行、provider-neutral 的 GPU Program；每个 shared transformation 前后都运行 full shared verifier；canonical 与 physical facts 分别由只读 KIR analysis 和 current-GPU-IR analysis 提供；family rewrite 只负责把统一事实物化成各自不同的 physical structure。

执行时实时枚举结果为 93 个文件、217 个 `@intent.kernel`，全部通过：

```text
files=93 kernels=217 passed=217 failed=0
```

唯一 Triton 数值 repro 也通过：

```text
triton:grouped_gemm: pass generated=4.692376 ms source=4.559776 ms ratio=1.029080
```

本轮没有运行或更新六张 baseline 表，没有处理 cuTile/TileLang provider timeout，也没有把 Triton 的一次通过当作 shared coverage 证据。

## 1. construction 与 intermediate completeness

原矛盾是 construction 声称产生 complete program，而 `RegionFoldOp`/`RegionScanOp` 要到后续 family pass 才第一次变成可执行结构。现在两类 op 在 construction 时已经携带：

- 完整 physical sources、identity、capture、state 与 result schema；
- summarize/combine 以及 scan apply/emit regions；
- typed segment parameter及其 canonical operation/dimension provenance；
- carry、yield、result assembly 与 observable origin。

`lib/Dialect/GPU/IR/GPUOps.cpp:789-920` 对 reduce、scan、region fold 和 region scan 的 axes、arity、helper arguments/yields、state/output flow 与 result schema做 op-local verification；`lib/Dialect/GPU/Transforms/VerifyGPUProgram.cpp:151-377` 再检查 module-level program space、execution groups、physical parameters、access footprint、effect coverage 与 buffer dataflow。

`lib/Dialect/GPU/Transforms/Passes.cpp:9-64` 在 construction 结束后先运行 full verifier，并在 access composition、multi-axis reduction normalization、pointwise ownership/blocking、region fold/scan、contraction、reduction以及第二次 access composition之后再次验证。第二次 access composition有明确 produced/consumed relation：structured realization会创建新的 typed gather/access；它不是无解释重复。`Passes.cpp:67-77` 在 mapping refinement和删除未使用参数后再次验证。

因此后续 pass 是 complete program 到 complete program 的变换，不再负责把 shell 补成第一次可执行的程序。

## 2. analysis authority

### 2.1 canonical analyses

`CanonicalKernelAnalysis` 只读 immutable KIR，当前提供：

| fact | 输入 | exact/unknown 边界 | consumer |
|---|---|---|---|
| coordinate provenance | `I.indices`、typed domain/subregion、helper arguments与 shape-preserving ops | 无唯一 source/axis 即 unknown，不从 shape 或名字猜 | index relation construction、physical axis mapping |
| index relation | canonical `IndexRelationAttr` 与显式 operands | malformed term、越界 operand即失败 | access coordinates/source-axis construction |
| logical workset | typed `ParallelOp` domain product、nested worksets与 observable writes | 无显式 independent workset时产生合法 singleton baseline | KIR-to-GPU initial execution groups |
| logical buffer | `BufferOp`、lexical ordered loop与 explicit initializer | 无 stable allocation identity即 unknown | physical buffer scope/init/lifetime construction |
| region segment | region fold/scan source axis、dimension identity与 canonical node identity | sources不同维或无 stable node即 unknown | typed segment declaration |

删除了无人消费的 `shapeFacts`、`ShapeAxisFact` 与对应 verify 空调用；它们此前只是 side analysis，既不验证也不参与 lowering。

### 2.2 physical analyses

`PhysicalProgramAnalysis` 只读 current executable GPU IR，集中提供：

- typed source/dimension/fragment-axis projection；
- source ranges、fragment-axis ranges与 exact logical range；
- replayability 与 blockers；
- structured reduction dependency；
- access footprint；
- tail-predicate recognition；
- buffer first-write/full-initialization dataflow。

unknown 是显式的 `Unknown` 或 `Ambiguous`，并携带 blocker；它不会降成默认 extent、附近 op、source 名或 role 名。分析对象的生命周期是：某个 transformation 在前一 mutation 后构造，完成所需查询，随后丢弃，才开始改写；下一阶段从新 current IR重算。唯一对象内缓存只服务同一 snapshot 的 unrestricted range query，不能跨 mutation 保存。

本轮把仍绕开统一 authority 的路径收回：

- reduction、contraction、region-fold replay 在 clone/retype 前必须通过 `PhysicalProgramAnalysis::replayability`；各 family 的递归代码只负责不同 IR 的物化；
- full-coverage rewrite 不再自己沿 coordinate producer graph重建 access ranges，而是在 mutation 前取得 `PhysicalAccessFootprint` snapshot；
- external view source identity 不再用 `(abi + 1) * 65536 + axis + 1` 合成。`KIRToGPU.cpp:451-547` 从 canonical `intent.parameter_nodes` 取得 stable source ID，写入 `gpu::ViewType`，axis继续作为独立 typed 字段；
- source/range comparisons统一使用完整的 `{sourceId, sourceAxis, derived}` identity，重复 source occurrence由 fragment axis 与 dimension共同区分。

## 3. decision authority before/after

这里统计 semantic inference authority，不统计 helper 调用数或合法 rewrite pattern数。Reduce、scan、region fold/scan、ordinary/scaled/sparse contract需要不同 mutation；它们不是同一实现的重复副本。

| fact/decision | authoritative carrier | authoritative query | 唯一 mutating entry | consumers | recompute rule | before → after独立重推导 |
|---|---|---|---|---|---|---:|
| logical workset | `LogicalWorksetFact`，GPU `program_space`/execution group/`DelinearizeOp` | `CanonicalKernelAnalysis::logicalWorksets` | KIR-to-GPU construction | all shared passes/providers | construction 后只读 current mapping；refinement 后重验 | 多处结构猜测 → 1 |
| pointwise ownership | typed access effects、source axes、current ranges、coordinate roles | physical range/dependence queries | `realizePointwiseOwnership` + 同一 family 的 blocking phase | pointwise access/value rewrites | mutation 后重建 physical analysis | 约 5 簇 → 1 |
| contract free/reduction relation | physical Contract operands、typed axes与 reduction/batch attrs | fragment/source/range/replay queries | contract-family realization；ordinary/scaled各物化自己的正式 op | provider contract legalization | 每个 contract rewrite后重算 | ordinary/scaled两套恢复 → 1 relation authority |
| group swizzle | `DelinearizeOp` typed coordinate roles/extents | current mapping本身 | `refineProgramMapping` grouped transform | providers | swizzle写回 current SSA并立即验证 | leaf/role推测 → 1 |
| persistent traversal | traversal-worker role、program segment与 typed capabilities | current mapping/resource attrs | `refineProgramMapping` persistent transform | providers | loop/program-space写回后立即验证 | 多处候选判断 → 1 |
| region segment | `RegionSegmentFact` + `ParameterAttr` + origin/dimension attrs | canonical segment query / physical symbol query | construction declaration | region fold/scan realization | current IR symbol query | family局部命名 → 1 |
| source/range axis | `AxisMapAttr`、`RangeType`、`MakeRangeOp` | `PhysicalProgramAnalysis::{sourceRanges,axisRanges,programRanges}` | 产生新 slice/range 的对应 structured transform | all family passes/verifier | mutation 后丢弃旧 analysis | 4 层 → 1 |
| replay eligibility | current SSA graph | `PhysicalProgramAnalysis::replayability` | 无第二个“决定”mutator；family clone只消费结果 | access/pointwise/reduction/contract/region | 每次 mutation 前 snapshot | 至少 5 → 1 |
| tail proof | typed range、end、predicate SSA | `PhysicalProgramAnalysis::isTailPredicate` | 需要扩大/分段的 family负责 materialize自己的 validity | accesses/structured ops | range改变后重算 | 约 6 → 1 proof authority |
| neutral fill | typed fragment element/schema | `materializeZeroFragment`/scalar constant utilities | 创建具体 access 的 family | load/gather/reduction paths | 无 side cache | 4 → 1 |
| access footprint | access resource、coordinates、source axes、validity/fill | `PhysicalProgramAnalysis::footprint` | 产生该 access 的 transform | shared verifier、full-coverage rewrite、providers | access改变后重算 | 4 层 → 1 |
| buffer init/dataflow | `BufferType` + `BufferOp.initial_value` + current resource uses | canonical logical-buffer fact / `bufferDataflow` | construction创建 allocation；provider只选storage form | verifier/providers | access/buffer mutation后重算 | local checks → 1 |
| effect coverage | kernel `effect_origins` + physical effect op `origin` | canonical origin collection / full verifier set equality | construction及保持 origin 的 effect rewrite | verifier/providers | 每阶段重新扫描 current IR | 多处隐式保持 → 1 |
| physical parameter binding | `ParameterAttr`、dimension/source attrs、typed `PhysicalExprAttr` | `queryParameterBinding`/symbol/blocking query | owning structured transform | shared passes/tuner/provider | parameter mutation后重新查询 | 名字解析 → 1 |
| operator/atomic/format semantics | MLIR enum attrs与 typed op fields | generated typed accessors与 op verifier | KIR-to-GPU semantic-preserving conversion | provider legality/serialization | op mutation保留 attrs并重验 | 裸整数解释 → 1 carrier authority |

Initial workset mapping、pointwise ownership、contract blocking、group swizzle和persistent traversal是五个不同 physical decisions；把它们写成五行是为了避免把“唯一权威”误解成一个万能 mapping pass。后者会把 op-specific accumulator/carry/contract semantics重新混在一起。

## 4. 保留的 family-specific lowering

保留以下专门 rewrite：

- reduce：identity/combine、多轴 normalization、padding neutralization与 reduction tree；
- scan：carry、inclusive/reverse与 output flow；
- region fold/scan：source segmentation、summary/combine、apply/emit；
- contract/scaled/sparse contract：free/reduction/batch axes与 accumulator schema；
- pointwise：effect-connected ownership、internal traversal与 access tail materialization。

它们之所以不是重复 authority，是因为“是否 replay、source/range是哪一个、tail predicate是否成立、access footprint是什么”来自统一 analysis；family代码只回答“该正式 physical op如何在 current IR中重写并保持自己的语义”。这与把四个 family都塞进一个 blocking pass不同。

## 5. observable semantics typed 化

跨层 observable semantics不再由共享的裸数字表解释：

- `BinaryOperator` 与 `ComparePredicate` 在 canonical KIR、GPU IR和 provider boundary中使用 typed enum attr；logical/bitwise `and` 只有 i1 条件满足时才可作为 tail conjunction；
- propagating `maximum/minimum` 与 `maximum_num/minimum_num` 保持不同枚举。cuTile只有 num 语义的 native spelling时，`lib/Target/CuTile/Serialization/Serializer.cpp:871-882` 对 propagating 版本明确拒绝，不再静默改成 `propagate_nan=False`；
- atomic RMW kind、ordering和sharing domain均是 typed attrs；provider只可保持映射或明确拒绝；
- scaled contract format、group relation及 sparse format/schema作为 typed op fields跨 shared/provider边界传递，provider legality不能按 kernel名恢复格式。

结构性整数如 record field index、axis position与 component count仍保留整数，因为它们由所在 typed op schema唯一解释，不是跨层可分歧的 observable semantic enum。

## 6. shared executable coverage

使用正式 compiler inventory boundary：

```bash
PYTHONPATH=python:examples:. /home/kingdom/.venvs/intentdsl-mlir20/bin/python \
  -m intent.compiler.inventory \
  --kernel-root examples/kernels \
  --bindings-module shared_compile_bindings \
  --compiler build/tools/intent-compile/intent-compile \
  --jobs 8
```

结果：

```text
files=93 kernels=217 passed=217 failed=0
```

该 gate只经过 frontend → canonical KIR → KIR-to-GPU construction → 全部 shared passes → full shared verifier，不依赖任一 provider serializer，也没有 allowlist。

重构中曾出现一次 216/217：`mhc_pre_fuse` 的同一 source在 Cartesian/indexed value中出现两次，两个 load都成为旧 source-ID-only查询的 blocker。没有在该 kernel或 family中加规则；修复落在 `PhysicalProgramAnalysis::collectAxisRanges`，先按 concrete fragment occurrence与完整 source/dimension identity匹配，再使用仅在无歧义时成立的 typed projection。随后完整 gate恢复 217/217。这也是原 cuTile/TileLang 非 Triton重合结构需要 shared层闭合、而不能靠 Triton registry背书的直接例子。

## 7. Triton non-regression repro

选择 grouped GEMM，是因为它真实穿过本轮修改的 workset/program mapping、typed access footprint、contraction source replay与 grouped swizzle路径。

```bash
./examples/run/baseline-v2.sh triton /tmp/intentdsl-gpu-ir.csv grouped_gemm
```

结果：

```text
triton:grouped_gemm: pass generated=4.692376 ms source=4.559776 ms ratio=1.029080
```

该命令完成 Triton source generation、provider compile/JIT、GPU launch与数值对照。ratio只随 runner输出记录；本轮没有据此做性能结论，也没有更新 CSV。命令生成的 `/tmp` CSV与临时 build目录已清理。

## 8. ref 对照、差异与实际后果

### 8.1 current-IR analysis 生命周期

Intent：`include/Intent/Dialect/GPU/Analysis/PhysicalProgram.h:161-220` 明确 analysis只读 current GPU IR；每个 transformation构造 snapshot并在 mutation前丢弃。Triton：`ref/triton/lib/Dialect/TritonGPU/Transforms/CoalesceAsyncCopy.cpp:175-200` 先收集完整 AxisInfo结果再 rewrite，原注释明确 changing IR invalidates analysis。

具体差异是 Intent使用短生命周期 query object，Triton在 pass内显式分成 analysis snapshot和 rewrite两段。实际后果是本轮没有引入跨 mutation cache；structured pass产生新 gather后必须重新运行 access composition和 physical verifier，旧 footprint不能复用。

### 8.2 exact/unknown access facts

Intent：`PhysicalProgram.cpp:1297-1353` 的 footprint保存 resource、coordinates、typed source axes、range roots、validity与fill，blocker产生 explicit unknown；`VerifyGPUProgram.cpp:325-340` 拒绝 non-scalar unknown footprint。Triton：`ref/triton/include/triton/Analysis/BufferRegion.h:114-152,178-223` 用 allocation frame与 `Exact/Unknown` region表示物理 storage view；unknown是 may-alias，不是空集合。

两者层次不同：Intent此处证明 provider-neutral logical access relation，Triton证明已选 layout后的physical address region。实际后果是 shared verifier现在会在 source/range provenance丢失处直接失败，而不会让 provider从 shape重猜；physical storage overlap仍属于 provider compiler，不被错误复制进 shared层。

### 8.3 buffer initialization

Intent：`PhysicalProgram.cpp:1355-1394` 从 current resource uses证明 FirstWrite read均由dominant write或完整 loop/branch initialization覆盖；FullValue必须由 `BufferOp` 的 explicit initial operand承载。TileLang：`ref/tilelang/tilelang/language/allocate.py:151-168` 把 initializer显式生成成 buffer store，`ref/tilelang/src/transform/verify_buffer_init.cc:346-394` 再按 typed access regions检查read-before-write。

具体差异是 Intent把 full initializer保留在 physical allocation op，TileLang把它展开为 TIR store。实际后果相同：type上的“已初始化”标签不能代替真实 value/store；缺失写入会在 shared verifier而非 provider serializer阶段失败。

### 8.4 structured reduce/scan

Intent：`GPUOps.cpp:789-920` 验证 generic typed combine、axes、arity与region flow；TileLang：`ref/tilelang/tilelang/language/reduce_op.py:19-65` 与 `scan_op.py:67-84,138-152` 在其 surface限定 fixed reduce/scan kinds与 shape。Triton：`ref/triton/lib/Dialect/Triton/IR/Ops.cpp:551-622` 同样以 first-class op和typed combine region验证结构，而不是识别普通循环。

具体差异是 Intent shared IR保留跨 provider generic semantics，TileLang native form只是目标能力子集。实际后果是 TileLang不能表达的 combine在 provider legality明确拒绝；shared层不会为了让 TileLang通过而把 generic combine改写成固定 kind。

### 8.5 repeated source occurrence

Intent的 `PhysicalSourceAxis` 同时保留 source、axis与 derived bit，axis-range query另用 fragment occurrence/dimension消歧；Triton `AxisInfo` 是 value-attached conservative lattice，`ref/triton/lib/Analysis/AxisInfo.cpp:1418-1441` join时只保留双方一致事实。实际后果由 `mhc_pre_fuse` 直接验证：把两个 occurrence仅按 source ID合并会错误得到 blocker；按 current typed occurrence分析后同一 kernel通过，而没有任何 kernel-name分支。

## 9. 删除的冗余与旧路径

本轮删除或收敛了：

- family-local source-ID-only range/replay fallback；
- ABI/axis magic-number source identity；
- unused `shapeFacts` analysis及 verify空调用；
- unused `sourceRankAttr`/`unitRoleAttr`常量；
- function name充当 physical origin的 fallback；缺 stable `intent.source`现在直接失败；
- full-coverage utility中第二套 coordinate producer traversal；
- reduction/contraction/region-fold各自独立决定 replay eligibility的入口；
- source identity三元组的重复手写比较。

没有删除合法的 provider terminal spelling，也没有把 target-specific storage/layout/MMA policy移入 shared层。

## 10. 规格与边界

实现调查没有发现需要修改 public DSL或 `doc/` 稳定语义的歧义；现有 compiler规格已经定义了 complete initial program、current-IR analysis、typed physical parameters、provider boundary与 verifier职责，因此本轮只修实现，没有把进度或失败写进 `doc/`。

当前 shared gate的结论只覆盖 provider-neutral executable GPU Program。cuTile timeout、TileLang storage/copy/pipeline、provider-specific legality与六表性能属于后续轮次，不能反向解释为 shared capability失败，也没有在本轮用慢 fallback冒充支持。
