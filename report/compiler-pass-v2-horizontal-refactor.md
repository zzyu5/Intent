# IntentDSL GPU Compiler V2 横向重构报告

## 1. 本轮目标与结论

这一轮不是继续增加一条只覆盖单个 op、单个 provider 或单个 kernel 的纵向样例，而是把已有 V2 骨架横向铺开，使主编译链形成唯一执行路径：

```text
Python DSL
  → canonical Intent Kernel IR
  → conservative Physical Program construction
  → shared GPU realization passes
  → provider-local form passes
  → verified provider-legal Physical Program
  → terminal source translation
```

本轮已经完成：

- public `I.ordered` 删除，普通 `for` 直接形成 ordered/serial facts；
- `partition(auto)` 从 public frontend 与全部 kernel 语料中删除；
- compiler 从完整逻辑域自动建立 ownership、blocking 与 contraction ranges；
- 49 个 canonical Intent operations 均有对应的 `intent_plan.exec_*` physical executable op；
- Physical Program verifier 禁止残留 canonical `intent.*` operation；
- `Build.cpp` 收缩为完整、合法但保守的初始程序构造，晚期物理决定进入独立 passes；
- Triton、cuTile、TileLang 的主要 program-form 选择进入 provider passes，terminal translator 不再各自反推 contraction axes、gather form、ragged route 等结构；
- 旧的 contraction physical-axis 推导路径已经删除，只保留 Plan 中的一份已选绑定。

本轮没有完成 `partition(count=P)`。原因不是实现困难，而是当前规格没有定义可观察分段的精确边界公式，详见第 9 节。

## 2. 编程模型迁移

### 2.1 删除 public `I.ordered`

作者不再使用额外的 `I.ordered` 标志授权编译器按顺序执行。普通 Python `for` 本身表达 source-visible 的顺序关系：

- frontend 直接生成 canonical `intent.for`；
- Kernel facts 将其 domain 登记为 ordered domain；
- shared realization 为对应 logical axis 建立 ordered/traversal role；
- provider lowering 从已选 traversal range 发射循环。

因此不再存在“普通顺序控制流”和“额外 ordered 权限”两套语义。

### 2.2 删除 `partition(auto)`

此前大量 kernel 由作者先写出 M/N/Q 等 region skeleton，再把 extent 留给 compiler：

```python
for mr in I.parallel(I.partition(m_axis, extent=I.auto("M_TILE"))):
    ...
```

这实际上要求作者预先决定 target program 中必须存在 blocking。现在改为直接表达完整逻辑算法：

```python
accumulator = I.contract(
    a[m_axis, k_axis],
    b[k_axis, n_axis],
    reduce=((1, 0),),
    acc_dtype=I.f32,
)
```

编译器随后从 result axes、contraction axes、parallel independence、value provenance 和 access facts 中建立 physical ownership 与 blocking。

本轮共迁移了 contraction、attention、MLA、Mamba、ragged、convolution、backward、quantization 等目录下原有的 `partition(auto)` 使用。迁移保留完整逻辑 workset、数值路径和 effect，只移除作者预先写下的 physical region skeleton。

### 2.3 保留 fixed `partition(extent=B)`

固定 extent 的 partition 仍然存在，因为其 region boundary 可以被 source body、状态、索引、RNG 或 ABI 观察。它不是 compiler-owned tile，不能与 `partition(auto)` 一起删除。

### 2.4 arbitrary stop expression

`state_stream(stop=...)` 不再被压缩成 target leaf 自己猜测的 axis endpoint。Frontend/KIR 保存原始 stop SSA value，Physical Program 的 `StreamBindingOp` 记录对应 value ID，三个 provider 按同一绑定发射并限制在合法逻辑范围内。

## 3. Physical Program 铺满

### 3.1 executable op 覆盖

`IntentOps.td` 中的 49 个 canonical operations 现在与 `PlanOps.td` 中的 49 个 `intent_plan.exec_*` operations 一一对应，包括：

- domain、control、state 与 condition；
- view load/store、gather/scatter 与 atomics；
- pointwise、reshape、transpose、broadcast；
- reduce、scan、contract、scaled/sparse contract；
- ragged、buffer、random 与 return。

Physical Program construction 会统一把 canonical operation 转成对应 physical executable operation，复制其 SSA operands/results、regions 和算法属性，并保留稳定 `intent.node` 与 `intent_plan.source_op` provenance。

### 3.2 唯一 executable authority

Physical verifier 现在检查：

- physical entry 中不能残留 canonical `intent.*` operation；
- 每个 `intent_plan.exec_*` 的 `source_op` 必须与其 physical mnemonic 对应；
- physical node ID 必须非负且唯一；
- 每个 operation decision 必须绑定一个实际 physical executable node。

KIR 仍然是算法语义来源，但进入 physical-program 阶段后，执行结构由 Physical Program 提供，不再是“KIR clone 加旁表”共同解释。

## 4. `Build.cpp` 的职责收缩

### 4.1 初始程序只选择保守合法值

`buildPhysicalProgram` 现在负责：

- 分析 canonical Kernel IR；
- 建立稳定 node/value/region provenance；
- 创建一份完整且能通过 verifier 的 Physical Program；
- 为尚未 refinement 的字段提供保守初值。

compiler-owned ranges 初始使用单 logical element，program axes 初始映射到同一 worker，不启用 reuse/group/persistent。Source-visible fixed partition 和 fixed state-stream segmentation 不会被抹掉，因为它们属于算法语义。

### 4.2 不再由 Build 持有最终值

为了确认 Build 中的值确实只是 baseline，而不是隐藏的最终权威，本轮让若干初值与最终值明确不同：

- tensor transfer/reduction/pointwise 初始 residency 使用保守 workspace；
- transfer realization pass 再选择 fragment/scalar、direct materialization 与 compact coverage；
- value realization pass 再选择 pointwise、reduction、scan carry 和 sparse-contraction residency；
- contraction、scan 和 buffer 的完整形态分别由其独立 passes 覆盖。

这样可以从 IR 变化直接观察 pass 是否真正承担了决定，而不是只把已有代码换了一个函数名。

## 5. Shared GPU pass pipeline

当前 pipeline 为：

```text
Construct Physical Program
  → Verify
  → Automatic Blocking
  → Verify
  → Access Range Realization
  → Verify
  → Transfer Realization
  → Verify
  → Stage Formation
  → Verify
  → Contraction Realization
  → Verify
  → Scan Realization
  → Verify
  → Value Realization
  → Verify
  → Private Buffer Residency
  → Verify
  → Persistent Traversal
  → Verify
  → Boundary Neutralization
  → Verify
  → Search-space Materialization
```

主要职责如下。

| Pass 责任 | 写入的 physical facts |
|---|---|
| Automatic blocking | axis roles、program order、worker/fold、reuse/group、ownership/traversal/reduction/lane ranges |
| Access range | transfer 对应的 selected physical footprint |
| Transfer realization | result residency、direct/deferred materialization、compact coverage |
| Stage formation | compiler-private stage slice 与 member/feature/reduction axes |
| Contraction realization | direct/staged/replay/deferred/scaled form、operand residency、accumulator flow、physical M/N/K axes、staged operand forms |
| Scan realization | fragment/workspace result、producer replay slice、materialized values |
| Value realization | pointwise/reduction/scan-carry/sparse value residency |
| Buffer residency | scalar array、private vector、global private workspace 与 owner binding |
| Persistent traversal | persistent program mapping |
| Boundary neutralization | consumer 是否在可观察结果前中和越界输入 |
| Search-space materialization | 只把合法数值参数交给 provider tuner |

每个结构 pass 后都运行 Physical Program verifier。Derived facts 由 analysis 重算；verifier 检查 legality/completeness，不承担性能选择。

## 6. Contraction 结构的单一来源

此前三个 materializer 会分别从 transfer relation、rank 和邻近 use-def 重新寻找 contraction reduction/M/N axes。只要三个 tile 偶然相同，这类错误就会被隐藏。

本轮将以下决定写入 shared `ContractOp`：

- `form`：`direct`、`staged`、`replay`、`deferred_one`、`deferred_two`、`scaled_direct`；
- selected reduction axis；
- two-sided deferred form 的 lhs/rhs result axes；
- staged lhs form：`member_row_gather` 或 `workspace`；
- staged rhs form：`expert_matrix`；
- lhs/rhs/accumulator residency；
- loop-carried accumulator owner、update、value 与 conditional provenance。

三个 provider 只读取这些绑定。原有的 `exactContractionReductionAxis`、`contractionReductionAxis` 和 `contractionAxes` 重推导路径已经删除。

## 7. Provider-local form passes

三家 provider 没有被强行做成完全相同的 dialect，但现在都先运行 provider-form pass，再进入 source translation。

### 7.1 共同前移的 form

- contraction primitive spelling；
- contraction orientation 与 batched capability；
- scaled-contract rank-2 grouped / rank-3 flattened layout；
- reduction/scan primitive spelling与 reduction axis；
- pointwise primitive spelling与 deferred form；
- gather form；
- stream tile spelling；
- transfer access/bounds/copy form；
- ragged compact/indexed route。

Provider verifier 要求这些 form 完整且相互一致。Unsupported capability 在 provider pass 阶段拒绝，不再等到 source 已经生成后才由下层偶然失败。

### 7.2 Triton

Triton provider pass 选择 row-launch form、collective/pointwise/contract spelling、transfer access、ragged route 和 scaled layout。Terminal translator 负责把已选 program structure 拼成 `tl.*` 源码，不生成 TTGIR layout、MMA instruction、register allocation 或 software pipeline。

### 7.3 cuTile

cuTile provider pass 选择 load/store 与 gather/scatter form、bounds policy、row occupancy delegation、gather spelling tuning surface 和 structured primitive spelling。保留两种 gather spelling 是合法 provider-local tuner candidate，不是重复旧路径。

### 7.4 TileLang

TileLang provider pass 显式选择 bulk/guarded/parallel transfer、buffer-space spelling、pointwise contract-operand form、GEMM operand isolation、launch form 与 primitive capability。Terminal translator 根据这些 form 产生 `T.alloc_*`、`T.copy`、`T.gemm` 等源码。

### 7.5 terminal translator 仍然读取什么

Translator 仍需读取 physical executable op 的 SSA operands、result type、index relation 和 selected axis/range，用于机械生成：

- 地址表达式；
- tensor shape；
- mask/guard；
- target API 参数。

这些读取不再决定 compact/indexed、direct/deferred、rank-2/rank-3、M/N/K axis 或 primitive form。Stage dependency/input/output/terminal 仍由共享的 derived analysis 从已选 stage slice 重算，而不是作为第二份优化决定保存。

## 8. 清理的旧路径

本轮删除或替代了：

- public `I.ordered` intrinsic、operation kind、lowering 和 canonical op；
- `partition(auto)` frontend 接受路径及全部语料调用；
- Physical Program 中 residual canonical op 的合法路径；
- 三家各自推导 contraction reduction/M/N axes 的模板函数；
- materializer 根据 rank/relation 决定 scaled layout、gather form和 ragged route的分支；
- Build 中直接产生最终 search/stage/access/contraction/scan 决定的单次前向路径。

没有增加 compatibility flag、legacy fallback 或 kernel-name matcher。

## 9. 尚未闭合：`partition(count=P)`

当前文档只规定：

- axis 被分为 source-visible 的 `P` 个连续 regions；
- part identity 可被 partial buffer、wrapper 或多个 kernel 观察；
- `P` 不是 compiler-owned worker count。

但尚未规定长度为 `N` 的 axis 第 `i` 个 part 的精确边界。至少存在两种都满足“P 个连续 regions”的定义：

```text
begin_i = floor(i * N / P)
end_i   = floor((i + 1) * N / P)
```

以及基于 `ceil(N/P)` 的固定最大 extent 切法。两者在 `N % P != 0` 时产生不同 part contents；part result 又是 source-visible 的，因此这是算法语义，不能由 realizer 临时选择。

在精确公式冻结之前，frontend 继续明确拒绝 `partition(count=P)`，而不是生成一份语义不确定的 IR。需要同时确定 `P > N` 时是否保留空 parts。

## 10. 验证范围与证据

遵守仓库验证纪律，本轮没有建立 test 目录、pytest、fixture 或全量回归设施。

完成的验证为：

1. C++/TableGen 完整构建：

   ```bash
   cmake --build /tmp/intentdsl-build -j2
   ```

   `intent-opt` 与 `intent-compile` 均成功链接。

2. 唯一端到端 repro：

   ```bash
   examples/run/repro.sh triton bf16_gemm
   ```

   该命令完成 DSL → KIR → Physical Program → shared/provider passes → Triton source → JIT → GPU 数值对照；结果为 PASS，最大误差 `0.03125`，generated `p50 = 2.0239 ms`、`p95 = 2.0281 ms`。

3. 静态一致性：

   - canonical op 与 physical executable op 为 49 对 49；
   - public/corpus 中没有 `I.ordered`；
   - public/corpus 中没有 `partition(..., I.auto(...))`；
   - `git diff --check` 通过。

这些证据证明新主链能够实际构建并运行一个完整 contraction kernel，但不等价于两台机器的全量矩阵回归；本轮按要求没有做全量测试。

## 11. 当前准确状态

当前已经不再是“一个 unary physical op 加少量纵向切片”的 V2 演示：physical executable op、shared pass pipeline、provider-form passes 和 terminal translation 已经横向覆盖现有 op 家族。

仍需严格区分两件事：

- 架构主链已经闭合，并且已有一个真实 GPU repro 通过；
- 全部历史 kernel 是否都没有数值或性能回退，本轮没有通过全量矩阵证明。

`partition(count=P)` 是当前唯一明确的 public programming-model 语义阻塞。它需要先冻结 source-visible 分段公式，然后才能继续 frontend、KIR、facts、Physical Program 与三家 lowering 的端到端闭合。
