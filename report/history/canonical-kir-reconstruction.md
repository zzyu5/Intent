# Canonical Intent Kernel IR 重构

## 1. 本轮边界与结果

本轮把最终 public DSL 收敛到一条唯一的 canonical Kernel IR 路径：

```text
@intent.kernel / @intent.fn
  -> Python frontend 与机械 desugaring
  -> typed canonical Intent KIR
  -> KIR verifier
  -> 可重算的 canonical analyses
  -> KIR-to-GPU typed boundary
```

本轮没有构造 shared executable GPU Program，没有恢复旧 Plan，也没有进入 Triton、cuTile 或 TileLang materialization。现有 93 个 `examples/kernels/` 文件中的 216 个 `@intent.kernel` 均完成 frontend lowering、MLIR 解析、canonical verifier 和 canonical analysis；随后全部在同一条 KIR-to-GPU 边界以明确的 `PhysicalProgram` 未实现状态停止。

最终验证结果：

```text
files=93 kernels=216 canonical_analysis_and_exact_boundary=216 failed=0
```

完整 CMake 构建 `ninja -C build -j2` 通过。按本轮边界，没有生成 provider source、没有执行 provider JIT，也没有做 GPU 数值对照。

## 2. 最终 canonical KIR 的组成

### 2.1 类型、身份与 ABI

KIR 现在具有正式的 `DomainType`、`RegionType`、`LogicalIndexType`、`BufferType`、`TupleType`、`RecordType`、`ConstexprType`、`EnumType` 和 `ViewType`。其中：

- domain、source-derived region、logical index 和 logical buffer 都携带稳定 provenance identity；
- tensor 的动态轴不再只打印成无法区分的 `?`，而由 `TensorShapeAttr` 保存逐轴 identity；
- view 的访问方向、strides、alias 与 noalias 由 typed `ViewConstraintsAttr` 进入 `ViewType`；
- kernel/helper 身份和参数角色分别由 `FunctionKindAttr` 与 `ParameterAttr` 表达；
- source location、node ID 和 block-argument ID 只用于诊断与 provenance，不参与 physical decision。

Shape 与 indexing 不再依赖字符串字典：`ShapeExprAttr` / `ShapeRelationAttr` 表达结果轴如何来自静态值、SSA extent 或 inferred axis；`IndexTermAttr` / `IndexRelationAttr` 表达 source rank、result rank、result dimension identities、每个 source axis 的索引形式及其 SSA operands。

### 2.2 普通控制流

`if`、`for`、`while` 和 `parallel` 都是普通 typed KIR operation。Verifier 精确检查 block arguments、condition、loop carry、yield 与 result schema，carry 可以是 scalar、tensor、tuple 或 record。

Canonical KIR 不再包含 `state_stream`、physical `partition`、public `ordered` 或旧 ragged/members executable node。仍存在的 surface helper 在 frontend 中机械展开为 domain、subregion、typed index relation 与普通控制流。

### 2.3 Structured operation

以下 family 均有唯一 first-class KIR representation：

- generic reduce / scan，以及统一降到该表示的内建 shorthand；
- region fold / region scan；
- ordinary / scaled / sparse contract；
- histogram；
- tuple/record construction 与 projection；
- cast、bitcast、broadcast、reshape、transpose、join/interleave、compare、select、mask 和普通 unary/binary value op。

Generic combine 与 region fold/scan 的 summarize、combine、apply、emit 均是真实 KIR region：captures 是显式 operands，region arguments 与 yield schema 是 typed 的，verifier 禁止 load/store/atomic/RNG 等 effect。Verifier 不替作者证明结合律、同态律或 action law。

Region fold/scan 还检查所有 source components 的 source-axis logical extent 相同、按同一 boundaries lockstep slicing，并完整保存 summary/state/result schema。`I.indices` 的 absolute-coordinate来源由 canonical op 语义建立，analysis 只沿已知的 slice、helper argument、broadcast、reshape、transpose、tuple/record 和普通 value flow 机械传播；传播失败返回 unknown，不从名称或相同 extent 反猜。

Sparse contraction 使用 typed `SparseFormatAttr`；format kind 唯一决定 group size、nonzero count 和 metadata schema，verifier 同时检查 compression axis、reduction relation、logical extent、静态 group divisibility 和 metadata 类型。Scaled contraction 保留 operand、scale tensor、format、group size、reduce/batch relation 与 accumulator result dtype；尚不能无歧义表达的 scale-axis relation列在第 8 节。

### 2.4 访问、effects 与 RNG

External view、logical buffer、gather、scatter、scatter-reduce、atomic family 和 RNG 均为正式 KIR operation。Indexed access 的 source/result rank、result dimension identity、advanced-index broadcasting、valid/fill、写入 value schema 都由 typed relation 与 verifier 共同约束。

Atomic 不携带 GPU scope 字符串；operation kind、address relation、value、ordering 和返回值 schema由各 canonical op 明确表达。CAS 返回 typed record。RNG 只保存 Philox4x32-10 的 counter、key 与逻辑 result schema，不保存 lane 或 vector packing。

Logical buffer 的 definite-write analysis 已纳入 canonical verifier：读取必须被完整初始化、同 element-set 写入或保守可证明的完整循环写入支配；无法证明时直接拒绝，不用默认初始化掩盖。

## 3. 唯一权威来源与 analysis 边界

Frontend 只生成新的 `intent.*` canonical operations。Verifier 同时拒绝以下旧节点或重复语义字段：

```text
partition, state_stream, ragged, members,
random, atomic_add, atomic_cas,
result_types, result_shapes, effects, spec, mode, scope
```

动态 shape identity、index relation、view constraints、function/parameter role 与 sparse format 都只有一份 typed authority，不再同时存在于 type、opaque metadata 和 emitter-side reconstruction 中。

`CanonicalKernelAnalysis` 只从 canonical KIR 重算 shape、coordinate provenance、index relation、def-use 和 logical-buffer initialization facts。它不选择 physical structure，也不形成第二份 executable representation。`verifyKernelModule` 会执行这组 canonical legality checks；KIR-to-GPU boundary 只消费验证后的 KIR。

旧 Plan dialect、GPU realization 和三个 provider Target 已从当前 CMake 与两个 compiler tool 的活跃构建/注册链断开。相关第三轮源码目录没有在本轮重写；它们不再是当前可执行路径，也没有被包装成兼容 adapter。

## 4. 第一轮迁移的 frontend 判卷结果

Frontend 首次横向检查 216 个 kernel 后，发现并修正了 8 个文件中的迁移遗漏。它们都是最终 surface 的 shape/type/provenance 机械修正，没有重新选择算法：

| 文件 | 修正 |
| --- | --- |
| `contraction/block_sparse.py` | 动态 zero shape 改为 source-derived row/column regions |
| `convolution/varlen.py` | buffer shape 改为 tokens region，并删除失去用途的 scalar extent |
| `loss/fused_linear_cross_entropy.py` | mean denominator 显式转换为 `f32` |
| `streaming/attention.py` | cu-seqlens 结果显式转换为 logical index；fold identity 使用 source region shape |
| `streaming/attention_specialized.py` | runtime sequence length 转为 logical index；identity 使用 member region shape |
| `streaming/linear_attention.py` | scalar transition scale 直接按 canonical broadcast 使用 |
| `streaming/mamba.py` | reshape/full 的动态 shape 改用 chunk source region |
| `streaming/paged_attention.py` | runtime page count 显式转换为 logical index |

第一轮报告列出的 34 个手写 summary algebra、packed decode、tail/subregion、numerical path、effect 或 ABI 重写没有在本轮被再次改写；它们仍应由第三轮首次 GPU 数值运行重点核验。本轮没有用 frontend 特判去接受错误源码。

Summary presence 统计也复核为：38 个 kernel 使用 bool-valid online summary，另有 1 个 BatchNorm kernel 使用 Welford `count=0` 表达 presence。该形态集中在 causal/ragged/paged/sparse/empty-domain summary，没有证据表明需要再新增一种隐式 presence 语义。

## 5. 横向语料覆盖

216 个 kernel 在同一条 frontend/KIR 路径上覆盖了：

- FlashAttention 的 region fold、explicit-valid summary、QK/PV contracts 与 absolute coordinate predicate；
- causal linear attention 的 region scan、同步 source components、transition/apply/emit/final state；
- online softmax 与 Welford 的多 component generic reduce；
- two-pass reduction、split-K 与显式多-kernel partial ABI；
- typed record carry/projection；
- ragged MoE 的多个 index relations、contract 与 scatter-reduce；
- nested ragged pooling 的嵌套 relation、tensor carry 与多结果；
- scaled contract 与 sparse 2:4 contract；
- compare-exchange 与其它 atomic family；
- Viterbi/Smith-Waterman 的 logical buffer、ordinary ordered loop 与 dynamic stop；
- nonzero/compaction 的 scan、indexed write 与多结果；
- fused cross entropy 的作者显式多-kernel编排；
- attention backward 的多结果、reduce 与 effects。

这些类别只用于横向验收；frontend、verifier 和 analysis 中没有 kernel-name、固定 shape、op-count 或 provider 分支。

## 6. 第一次强制自查

主要结构形成后，第一轮逐 family 审计发现并修掉：

- indexed access、dynamic shape、view ABI 与 sparse format仍以 `DictionaryAttr` 或裸整数承载语义；现已全部换成 typed attributes；
- result shape 的动态 identity没有在 relation 中逐轴校验；现同时核对 result rank 与 dimension identities；
- access verifier没有完整核对 advanced-index broadcasting、valid/fill 和 write/value partition；现由统一 relation verifier处理；
- logical-buffer definite-write只在旧下游路径执行；现进入 canonical verifier；
- canonical analysis曾保留无消费者的 provenance ID lookup side table；现已删除，只保留可重算事实；
- active build/tool 仍注册旧 Plan/Target；现只构建 Intent dialect、Transforms、Analysis 与 KIRToGPU boundary。

修正后重新跑完 93/216 验收，无失败。

## 7. 第二次强制自查与减法收尾

第二轮由 KIR family、旧路径、examples 与规格四个方向独立复核。收尾结果是：

- active frontend/KIR/analysis/conversion 中不再有 semantic `DictionaryAttr`；
- 不存在遗留 `ScaleRelation` 猜测、compatibility flag、legacy KIR producer 或 provider fallback；
- sparse 2:4 没有保留旧专用 executable path，public spelling机械汇入 generic sparse contract；
- tuple/record reduce、scan 和 loop carry 共享同一 component schema，不存在第二条实现；
- histogram scalar validity在 frontend机械 broadcast为显式 tensor validity，不形成第二种 KIR schema；
- 8 个 example 修正逐项复核均未改变算法、kernel 数量或 host-visible ABI；
- `git diff --check` 与完整构建通过，最终 216 个 kernel再次全部通过 canonical verifier、analysis 与精确 boundary检查。

## 8. 规格中尚未闭合的三处表面

现有 216 个 kernel 的 canonical KIR 路径已闭合，但完整规格还存在三处不能由实现自行猜测的表面缺口：

1. **Tuple/record logical-buffer element。**规格文字允许 tuple/record 作为 logical-buffer element schema，但 `I.buffer` 目前只接受一个 `dtype`，没有定义 structural element schema 的 public spelling、initializer 或 indexed read/write结果形状。把它私自 flatten 成多个 buffer 会改变 alias、lifetime 与 mutation identity，因此本轮没有发明该规则。
2. **Scaled-contract scale relation。**规格要求 scale tensor 到 logical groups 的轴关系属于算法语义，但当前调用面只有 scale tensors、format 与 group sizes。对多个 reduction axes，group axis不能由 shape或位置唯一推出。本轮曾用真实 multi-axis examples检验显式 relation表示，确认任何“取某个 reduction axis”的默认规则都会猜作者语义，因此没有保留该尝试。
3. **一般调用前置条件。**规格列出 sorted/unique、dynamic-extent equality、non-alias 与 encoded-format validity 等前置条件，但 public/KIR surface目前只定义 `assume_in_bounds` 和 view alias/noalias constraints。没有权威 syntax与 typed schema时，verifier不能自行添加通用 precondition op。

这三处均未被当前 216 个 kernel 触发，因此不影响本轮全量验收；但它们阻止把对应的语言子能力宣称为已经完整实现。报告只保留事实，没有用 metadata、默认值或 target-specific规则填补。

## 9. 下游状态

已经切到新 KIR 的 consumers 是：

- `verifyKernelModule` / `verify-intent-kernel`；
- `CanonicalKernelAnalysis`；
- `lowerCanonicalKIRToGPU` typed boundary；
- `intent-opt` 与 `intent-compile` 的活跃 Intent dialect注册和构建依赖。

`lowerCanonicalKIRToGPU` 会先再次验证 canonical KIR，然后明确报告 shared executable GPU Program construction 尚未实现。它不调用旧 Plan、provider materializer 或 terminal translator。本轮因此完成的是 canonical KIR，而不是用旧后端路径伪造一次 GPU 运行。
