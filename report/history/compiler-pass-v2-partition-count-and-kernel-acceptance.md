# IntentDSL GPU Compiler V2：`partition(count)` 与真实算子接纳报告

## 1. 本轮范围

这一轮完成两件事：

1. 冻结并实现 source-visible `partition(axis, count=P)`；
2. 用 stateful streaming、ragged MoE、split-K、in-place cache update 和 attention tail 检验上一轮重构后的 automatic blocking 与 provider projection。

本轮没有做全量矩阵，也没有把尚未进入 baseline-new registry 的 source entry 伪装成已经接入。验证只使用既有可手动执行的 repro 入口。

最终主链为：

```text
Python DSL
  → canonical Kernel IR
  → KernelModel / KernelFacts
  → complete Physical Program
  → shared GPU passes
  → provider-local form passes
  → terminal source
  → provider JIT
  → GPU numerical comparison
```

## 2. `partition(count=P)` 的正式语义

对长度为 `N` 的逻辑轴和 `P >= 1`：

```text
block   = ceil(N / P)
begin_i = min(i * block, N)
end_i   = min((i + 1) * block, N)
```

第 `i` 个 part 的 identity 是 `i ∈ [0, P)`，对应 region 为 `[begin_i, end_i)`。

由此得到以下合同：

- region 连续，使用和 `partition(extent=ceil(N/P))` 相同的普通 tail；
- 尾部 part 可以较短或为空；
- 空 part 保留 identity 和 wrapper-visible ABI slot；
- 空 part 不执行 body，不写出，也不产生 effect；
- target 可以省略空 part 的 physical worker，但不能压缩或重编号非空 part；
- 后续 kernel 如果读取全部 `P` 个 partial slots，wrapper 必须先用归约单位元初始化整个 partial buffer。

`P` 是 source/ABI 值，不是 worker count 或 tuner 参数。它必须在 launch 前可见，可以来自 literal、shape dimension、runtime scalar 或 constexpr；不能依赖 kernel body 运行后才产生的值。

## 3. Frontend 与 Kernel IR

### 3.1 Frontend surface

`I.partition` 现在要求 `extent=` 与 `count=` 恰好提供一个：

```python
for part, region in I.parallel(I.partition(axis, count=P)):
    ...
```

Count mode 的 iteration body 固定获得：

- `part`：`LogicalIndexType("partition_part")`；
- `region`：保持原 logical relation 的 `RegionType`。

Frontend 会拒绝：

- 同时给出 `extent` 和 `count`；
- 两者都不提供；
- `count=I.auto(...)`；
- 可静态证明的非正 count。

### 3.2 Canonical Kernel IR

`intent.partition` 对 extent/count 使用同一 canonical 两 operand schema：

```text
(source axis, source-visible parameter) -> partition<mode, region-type>
```

Kernel IR verifier 集中检查：

- mode 只能是 `extent` 或 `count`；
- operand/result schema 完整；
- result type 与 mode 一致；
- extent 是正的固定值；
- 静态 count 为正。

旧的一 operand partition 路径已经从 verifier、facts 与 index-relation analysis 中收掉，不再保留兼容 schema。

## 4. Physical Program 中的唯一绑定

本轮新增 `intent_plan.partition_binding`，显式保存：

- canonical partition node；
- source axis node；
- count SSA value；
- part block argument；
- region block argument；
- canonical segment extent 引用。

Segment extent 使用稳定名字 `partition_extent_<partition-node>`，其唯一含义是 `ceil(N/P)`。它是算法语义的派生值，不是多个合法方案中的物理选择，因此：

- 不进入 search space；
- 不由三个 leaf 各自重算边界语义；
- 不允许从 worker count 或相同 shape 猜测。

Physical Program verifier 进一步保证：

- count partition axis 具有 parallel ownership；
- ownership range 的 tile 与 segment extent 一致；
- region argument 精确绑定该 ownership range；
- part identity 与 region argument 是两个不同 value；
- 同一 argument 不存在重复 binding。

## 5. 三个 provider 的机械投影

Triton、cuTile、TileLang 都从同一 `partition_binding` 生成：

- `_intent_partition_extent(N, P)`；
- program ordinal 对应的 part identity；
- `part * block + local_lane` 对应的 region indices；
- 只覆盖非空 prefix 的 launch grid；
- tail validity 与 reduction identity padding。

只由输出 view 才能确定 `P` 的 callable 不能由 `run()` 自动分配结果，因为 `run()` 没有 output view 可读取该 shape。现在三家统一要求调用者预分配 output 并走 `launch()`，不再生成引用未定义 output 参数的 wrapper。

用于实际验证的两阶段算法是：

```text
pass 1: partition N columns into P source-visible parts, write partial max
pass 2: reduce all P partial slots
```

验证 shape 使用 `N=257, P=300`，真实制造空 parts；wrapper 每次 launch 前把 partial 全部填成 `-inf`。

## 6. 真实算子接纳暴露的共享问题

### 6.1 同一 logical axis 被一个物理索引覆盖

Mamba chunk scan 的 `rows` axis 同时承担：

- output ownership；
- ordered state-stream traversal；
- contraction reduction；
- row-vector lane。

Physical Program 已经分别保存四种 purpose 的 range，但三个 materializer 曾把 stream traversal index 写回按 logical axis node 单键索引的全局表，覆盖 ownership index。结果是 output/global row 与 stream-local row 混用。

修复后：

- 普通 axis value 使用 ownership projection；
- stream body 使用 `StreamBindingOp` 指定的 traversal projection；
- region block argument 使用 `RegionBindingOp` 指定的 purpose/range；
- 同一 logical axis 不再只有一个隐含的“当前物理索引”。

这是一处 fact consumption 修复，没有新增 kernel 分类或 Mamba 特判。

### 6.2 result-axis region provenance 被 shape 标签压扁

Tensor result 的动态轴既可能来自完整 domain，也可能来自 nested region block argument。后者必须继续指向具体 block argument，才能选择正确的 ownership/traversal range。

此前 leaf 会从 `intent.result_shapes` 的字符串标签恢复这项关系；同一事实容易被三家重复解析或按 extent 猜错。

现在：

1. canonical metadata 只在公共 `KernelModel` analysis 中解析一次；
2. `KernelModel` 保存结构化的 result-axis → region block argument provenance；
3. shared lowering 将 provenance 与 `RegionBindingOp` 组合成精确 selected range；
4. validity、padding、`I.indices(region)` 和 member projection 只消费结构化查询结果。

完整 domain 产生的 `?region_*` 标签不会被误当成 block argument；只有真实 nested block argument 才进入该映射。

### 6.3 staged ragged contraction 的信息丢失

MoE 接纳暴露了三处问题：

- ragged member 已有 compiler-selected ownership range，但 staged contraction 仍只认可旧 source partition 形态；
- 旧的 staged metadata absorption 会连同 gather 的 valid/fill SSA producer 一起删除；
- cuTile gather candidate 使用了不正确的 tuning role。

对应修复是：

- stage member 从 selected ownership range 判断，不从旧 source skeleton 判断；
- 只吸收被 `intent.ragged` 独占消费的 descriptor whole-view load；
- valid/fill producer 按正常 use-def replay；
- cuTile 使用正式 `gather_spelling` candidate role。

这些规则均按 operation/use/range 事实工作，不含 `moe` 名字或 whole-kernel matcher。

### 6.4 TileLang structured index projection

TileLang 对 rank 大于一、但只有一个非 singleton 轴变化的 structured index，原来会物化 broadcast fragment，再交给 `T.Parallel`，导致下层 layout inference 冲突。

现在 leaf 直接投影该唯一变化轴的 structured expression；这是目标语法质量修复，不改变 shared ownership、tile 或算法结构。

### 6.5 TileLang staged dynamic-row scatter

实验确认 TileLang 当前路径的 vectorized dynamic-row atomic 会形成错误地址；改成串行 atomic 虽可运行，但会成为数量级更慢的伪支持。

本轮没有保留这两条错误路径。TileLang provider pass 在 source emission 前明确拒绝 staged dynamic-row scatter reduction；terminal translator 只保留“未 legalize form 不得抵达终端”的 invariant diagnostic。

## 7. 定向验证结果

### 7.1 构建

```bash
cmake --build /tmp/intentdsl-build --target intent-compile -j8
```

`intent-compile` 成功链接，`git diff --check` 通过。

### 7.2 `partition(count)`

```bash
examples/run/repro.sh triton partitioned_two_pass_max
examples/run/repro.sh cutile partitioned_two_pass_max
examples/run/repro.sh tilelang partitioned_two_pass_max
```

| provider | 数值 | generated p50 |
|---|---:|---:|
| Triton | max error `0.0` | `0.0627 ms`（最终复验） |
| cuTile | max error `0.0` | `0.0424 ms` |
| TileLang | max error `0.0` | `0.0465 ms` |

### 7.3 Mamba chunk scan

```bash
examples/run/repro.sh triton mamba_chunk_scan
examples/run/repro.sh cutile mamba_chunk_scan
examples/run/repro.sh tilelang mamba_chunk_scan
```

| provider | max error | generated p50 |
|---|---:|---:|
| Triton | `0.0006580352783203125` | `0.0220 ms` |
| cuTile | `0.0006580352783203125` | `0.0220 ms` |
| TileLang | `0.0137786865234375` | `0.0258 ms` |

### 7.4 MoE

| provider | 状态 | 结果 |
|---|---|---|
| Triton | PASS | max error `9.05944e-06`；generated/upstream `0.8549x` |
| cuTile | PASS | max error `9.05944e-06`；generated/upstream `1.0262x` |
| TileLang | explicit unsupported | provider pass 拒绝 staged dynamic-row scatter reduction |

### 7.5 其它受影响结构

- Triton paged split-K attention：max error `0.0001220703125`，`p50=0.1713 ms`；
- Triton reshape-and-cache：两个输出 max error 都是 `0.0`，`p50=0.0302 ms`；
- base attention：max error `3.0518e-05`；
- `Q=127, K=131, D=64/80/96`：全部数值通过；
- `D=256`：候选需要 `116736 B` shared memory，超过本机 `101376 B`，由下层明确报资源不足，没有 fallback。

## 8. Inventory 与历史失败没有混写

上一轮的准确口径是：119 个 runtime-visible source entries，baseline-new registry 92 个，尚有 27 个未完成真实纵向接入，即 Triton 9、cuTile 9、TileLang 9。

本轮没有新增 baseline-new registry row，所以这个数字仍是 27。给旧 repro 接上 source，只能证明当前 DSL/provider projection 可以与该 source 做一次定向比较；如果算法、shape、routing、partial ABI 或调用结构不同，就不能把它当成对应 inventory entry 已闭合。

同样，27 个 inventory 缺口和旧固定矩阵中的 `compile_failed` 是两类事实：

- inventory missing：source entry 尚无相同算法的 DSL、adapter、registry 与真实对照；
- compile/JIT/runtime/numerical failure：已有 entry 走到某个具体阶段后失败。

本轮只更新真实定向运行触及的状态，没有把其余历史失败推断为已经消失，也没有回写全量 CSV。

## 9. 本轮准确结论

本轮完成了 `partition(count=P)` 从语言语义到三家 GPU execution 的闭合，并用真实算子修掉了 automatic blocking 重构后最关键的两类信息丢失：同轴多 purpose 覆盖、result-axis region provenance 丢失。

当前仍存在两条有证据的 target/resource 边界：

- TileLang 无法安全、高吞吐地投影 staged dynamic-row scatter reduction；
- 本机 attention `D=256` 的当前合法候选超过 shared-memory 容量。

它们都以明确失败结束，没有静默错误、串行伪支持或 legacy fallback。27 个 source inventory entry 仍未接入；本轮没有用相近算法或状态文字缩小这个数字。
