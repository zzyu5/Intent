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

随后第三轮已经冻结并闭合 `partition(count=P)`，并用 split-style 两阶段归约实际压过 frontend、Kernel IR、Physical Program、三家 provider 与 wrapper ABI。真实算子接纳还暴露并修复了同一 logical axis 同时承担 ownership、ordered traversal 与 reduction 时的用途覆盖，以及 result-axis region provenance 在 lowering 中被压扁的问题，详见第 9 至 11 节。

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

compiler-owned ranges 初始使用单 logical element，program axes 初始映射到同一 worker，不启用 reuse/group/persistent。Ranked transfer、reduction 与 pointwise result 先用合法的 `private_fragment`，scalar result 使用 `private_scalar`；这些都是可被后续 pass 覆盖的完整初值。Source-visible fixed partition、count partition 和 fixed state-stream segmentation 不会被抹掉，因为它们属于算法语义。

### 4.2 不再由 Build 持有最终值

为了确认 Build 中的值确实只是 baseline，而不是隐藏的最终权威，本轮让若干初值与最终值明确不同：

- tensor transfer/reduction/pointwise 初始 residency 使用合法 fragment/scalar；
- transfer realization pass 再选择 fragment/scalar/workspace、direct materialization 与 compact coverage；
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

## 9. `partition(count=P)` 已闭合

### 9.1 唯一 source 语义

对长度为 `N` 的 axis 和 source-visible count `P >= 1`：

```text
block   = ceil(N / P)
begin_i = min(i * block, N)
end_i   = min((i + 1) * block, N)
```

Part identity 是 `i ∈ [0, P)`。尾部 part 可以较短或为空；空 part 保留 wrapper-visible slot，但不执行 body、不写出、不产生 effect。后续 kernel 读取全部 `P` 个 partial slots 时，wrapper 必须先按 reduction identity 初始化 buffer。Target 可以不启动空 part 的 worker，但不能压缩或重编号非空 identity。

### 9.2 编译链中的唯一表示

- frontend 接受 `extent=`/`count=` 二选一；count iteration body 固定接收 `(part, region)`，`part` 是 logical index；
- Kernel IR verifier 检查 canonical 两 operand schema、mode/result type 对齐和静态 count 正数约束；
- `KernelFacts` 保存 partition、source domain、count SSA、part argument、region argument，不从 shape 猜 part identity；
- Physical Program 使用 `intent_plan.partition_binding` 显式绑定上述身份，并让 ownership range 引用稳定的 `partition_extent_<node>`；
- 这个 extent 是 source 公式的唯一派生值，不进入 tuner search space；
- 三家 provider 都从同一 binding 生成 `ceil(N/P)`、program ordinal、region index 与 launch grid；grid 只覆盖非空 prefix，wrapper ABI 仍保留全部 `P` slots；
- 只由输出 view 才能确定 `P` 的 callable 必须走预分配 `launch`。自动分配的 `run` 不再引用一个不存在的 output 参数，而是明确拒绝。

Count 必须是 launch-visible 的 literal、ABI dimension、runtime scalar 或 constexpr。它不能是在 kernel body 执行后才得到的值，因为 launch grid 在 body 运行前就必须确定。

## 10. 真实算子接纳暴露并修复的问题

### 10.1 同一 logical axis 的多种 physical purpose

Mamba chunk scan 的同一 `rows` axis 同时承担：

- output ownership；
- state-stream ordered traversal；
- contraction reduction；
- row-vector lane。

Plan 原本已经分别保存 `ownership`、`traversal`、`reduction` 与 `lane` ranges，但三个 materializer 都把 stream traversal index 回写到按 axis node 单键索引的全局表，覆盖了 output ownership index。结果是一个 tile 的 global offset 被另一个 purpose 的局部 offset 替代。

修复没有增加新的算法字段：stream body 只使用 `StreamBindingOp` 选择的 traversal range；普通 axis value 继续使用 ownership range；region block argument 按 `RegionBindingOp` 的 purpose 取精确投影。同一 logical axis 不再只有一个“当前物理索引”。

### 10.2 result-axis region provenance 不再由 leaf 解析字符串

Tensor result 的动态 axis 标签可能来自完整 domain，也可能来自 nested region block argument。后者必须继续绑定到该 argument 的 selected physical range，否则 validity、padding、`I.indices(region)` 与 ragged member projection 会退回逻辑 shape 或另一个 purpose 的 range。

现在 canonical shape metadata 只在公共 `KernelModel` analysis 中解析一次，形成结构化的 result-axis→region block argument provenance。Shared lowering 再把它与 `RegionBindingOp` 结合，三个 provider 只消费查询结果；terminal leaf 不再解析 `?region_*` 字符串，也不再按相同 extent 猜 axis。

### 10.3 staged ragged 与 TileLang 投影

MoE 接纳暴露了三处共享问题：ragged member 的 compiler-selected ownership 没被 staged contraction 接受；descriptor load 与 valid/fill producer 被一个过宽的“staged metadata absorbed”判定一起删除；cuTile gather candidate 使用了错误的 tuning role。修复均基于 ownership/use-def/operand role，不含 kernel-name 分支。

TileLang 对 rank>1、但只有一个非 singleton 轴变化的 structured index，原来退化成 materialized broadcast fragment，导致下层 layout inference 冲突。现在 leaf 直接投影精确的结构化索引；这只是目标语法兑现，不改变 shared physical decision。

TileLang 0.1.13 对 staged dynamic-row scatter reduction 的向量 atomic 会产生错误地址，而串行 fallback 会成为数量级更慢的伪支持。Provider pass 现在在 emission 前明确拒绝该 form；实验性的 terminal 慢路径已经删除。

### 10.4 source inventory 与历史失败的边界

上一轮确认的口径是 119 个 runtime-visible source entries、baseline-new registry 92 个，尚缺 27 个。这个数字本轮没有被 adapter 名称或相近算法虚假缩小：除上一轮已经进入 registry 的 Triton split-K paged attention 外，本轮没有新增 baseline-new registry row，因此剩余仍是 Triton 9、cuTile 9、TileLang 9。

本轮给旧 repro 接上 official cuTile MoE、Triton grouped GEMM 与 TileLang grouped GEMM source，目的是在同一 DSL 上观察 provider projection，并不等于把 cuTile official fused MoE、TileLang fused routed/shared MoE 或 Triton 两种现代 MoE projection 接入 baseline-new。它们的算法、shape、routing/partial ABI 或调用结构并不相同；只有新增相应 DSL/adapter、进入 registry、真实运行并与该 source 对照后，才能从 27 中扣除。

历史 `compile_failed` 也没有与 inventory 相加。本轮定向重跑只更新被重构路径直接触及的事实：Mamba 三家通过；MoE 两家通过、TileLang 提前明确拒绝；Triton split-K 与 in-place 通过；attention 的普通和非二次幂 head dimension 通过、`D=256` 是资源失败。其余历史失败仍是旧全量状态，不能由这些定向结果推断已经消失。

## 11. 验证范围与证据

遵守仓库验证纪律，本轮没有建立 test 目录、pytest、fixture 或全量回归设施。

完成的验证为：

1. C++/TableGen 完整构建：

   ```bash
   cmake --build /tmp/intentdsl-build --target intent-compile -j8
   ```

   `intent-compile` 成功链接。

2. `partition(count)` 两阶段归约：

   ```bash
   examples/run/repro.sh triton partitioned_two_pass_max
   examples/run/repro.sh cutile partitioned_two_pass_max
   examples/run/repro.sh tilelang partitioned_two_pass_max
   ```

   三家都完成两份 DSL → KIR → Physical Program → provider source → JIT → 两次 GPU launch，`P=300 > N=257`，partial 预填 `-inf`，最大误差均为 `0.0`。最近一次 Triton 复验为 `p50=0.0627 ms`、`p95=0.0675 ms`；本轮 cuTile 与 TileLang 运行分别为 `p50=0.0424 ms`、`0.0465 ms`。

3. stateful streaming：

   ```bash
   examples/run/repro.sh triton mamba_chunk_scan
   examples/run/repro.sh cutile mamba_chunk_scan
   examples/run/repro.sh tilelang mamba_chunk_scan
   ```

   三家数值均通过；Triton/cuTile 最大误差 `0.0006580352783203125`，TileLang 最大误差 `0.0137786865234375`。最近一次 Triton 复验为 `p50=0.0220 ms`；本轮 cuTile 为 `0.0220 ms`，TileLang 为 `0.0258 ms`。这条链真实制造了同轴 ownership/traversal/reduction/lane 组合。

4. ragged、split-K 与 in-place：

   - MoE：Triton 与 cuTile 数值通过，最大误差 `9.05944e-06`；Triton generated/upstream `0.8549x`，cuTile `1.0262x`；TileLang 在 provider pass 明确拒绝 staged dynamic-row scatter reduction；
   - `triton paged_splitk_attention`：数值通过，最大误差 `0.0001220703125`，`p50=0.1713 ms`；
   - `triton reshape_and_cache`：两个输出最大误差均为 `0.0`，`p50=0.0302 ms`。

5. attention 与非典型 head dimension：

   - base attention 数值通过，最大误差 `3.0518e-05`；
   - `D=64/80/96` 且 `Q=127, K=131` 的 tail 均数值通过；
   - `D=256` 当前候选需要 `116736 B` shared memory，超过本机 `101376 B`，下层明确报资源不足；没有用慢路径伪装支持。

6. 静态一致性：

   - canonical op 与 physical executable op 为 49 对 49；
   - public/corpus 中没有 `I.ordered`；
   - public/corpus 中没有 `partition(..., I.auto(...))`；
   - `git diff --check` 通过。

这些证据覆盖 count partition、stateful streaming、ragged staged contraction、split-K、in-place 与 attention tail；不等价于两台机器的全量矩阵回归，本轮按要求没有做全量测试。

## 12. 当前准确状态

当前已经不再是“一个 unary physical op 加少量纵向切片”的 V2 演示：physical executable op、shared pass pipeline、provider-form passes 和 terminal translation 已经横向覆盖现有 op 家族；`partition(count=P)` 也已从 source 语义闭合到三家真实 GPU execution。

仍需严格区分两件事：

- 架构主链已经闭合，并且已有多种结构的定向 GPU repro 通过；
- 全部历史 kernel 是否都没有数值或性能回退，本轮没有通过全量矩阵证明。

目前留下的是有证据的 provider/resource 边界，不是未定义语言语义：TileLang 0.1.13 无法安全、高吞吐地兑现 staged dynamic-row scatter reduction；本机 attention `D=256` 候选超过 shared-memory 容量。二者均显式失败，没有 fallback 或静默错误。历史全量能力与性能是否完全无退化，仍需要后续明确要求的全量轮证明。
