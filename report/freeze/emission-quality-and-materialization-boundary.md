# 生成质量与物化边界

## 本轮结论

本轮没有改动 DSL、Kernel IR、Physical Plan 或 realization，也没有引入算法替换、图级融合和按 kernel 名字分支。保留下来的改动只有两类：

1. cuTile 的四条间接读取路径改为直接使用原生 masked gather，去掉 gather 后额外生成的逐元素 `where`；
2. 三个 target 内部把“消费既定物理事实”和“拼写目标 API”分开，目标 API 名称、参数名和机械表达式集中到各自的 `Emission/Syntax/Spelling` 模块。

这不是新增一层表示。`Syntax` 没有 schema、验证器或独立事实，只接受 Kernel IR/Plan 消费者已经确定的角色、类型和物化位置，并返回该 target 当前 API 的拼写。

## 生成质量问题如何选出

固定的 5090 表中，分页注意力和 MoE 的 cuTile 生成侧分别是 0.3436 ms 和 10.2724 ms；同一算法结构在其他 target 上存在明显更快的投影。这类跨 target 离散度不能直接证明 Plan 正确或错误，但适合优先检查 leaf 是否漏用了目标原生能力。

并排阅读生成源码后，cuTile 的间接读取存在一个具体且局部的问题：

```python
value = ct.gather(..., check_bounds=True, padding_value=fill)
value = ct.where(logical_mask, value, fill)
```

cuTile 当前的 `ct.gather` 已经同时接受 `mask`、`check_bounds` 和 `padding_value`。第二条语句不是新的物理决定，也不是算法语义，只是在目标语法层重复兑现同一份有效性事实。因此改为：

```python
value = ct.gather(
    ..., mask=logical_mask, check_bounds=True, padding_value=fill
)
```

改动覆盖四个共享形态，而不是某个 kernel：

- staged indirect gather；
- ordered indexed members；
- staged contraction 的间接 lhs；
- ragged stage route materialization。

Plan 仍然决定成员有效性、索引表达式和填充值；cuTile leaf 只选择能一次表达这些已定事实的原生调用。

## 受影响结果

本轮只重跑直接经过上述路径的 repro，没有更新固定全量表。

| repro | 固定表 generated p50 | 本轮 generated p50 | 数值结果 |
|---|---:|---:|---|
| cuTile `paged_attention` | 0.3436 ms | 0.3270 ms | pass；D=80 的 fully-masked row 同样通过 |
| cuTile `moe` | 10.2724 ms | 10.0743 ms | pass |

这两项分别约改善 4.8% 和 1.9%。收益不被解释成新的调度优势：物理计划没有变化，差异来自把两条目标语句收成一个原生 masked gather。

## 物化层如何解耦

三个 emitter 原先把以下内容混在 `Source/Emitter.cpp` 和逐 op handler 中：

- Plan role 到 tile/tuner 参数名的映射；
- canonical pointwise/reduction/scan/contraction 到目标 API 名的映射；
- TileLang buffer space 与维度名字的拼写；
- cast 的目标表达式，包括 Triton 和 TileLang 的 E8M0 decode 序列。

其中已经出现过两类真实维护成本：tile 参数名和 tuner 参数名共享同一规则却各自维护；E8M0 的跨架构修复需要进入逐 op handler 才能找到目标 API 序列。它们说明耦合不是单纯的整洁问题。

现在每个 target 的职责边界是：

- `Source` 与 `Handlers`：读取 Kernel IR 和 Physical Plan；确定当前 op 的角色、类型、空间、有效性与 capability；决定应发射哪个已选概念；
- `Syntax/Spelling`：把已选概念机械拼成 Triton、cuTile 或 TileLang 的 API 名和表达式。

例如 cast handler 仍负责判断源/目标类型、是否为 E8M0 decode、结果是否为 f32；`Syntax` 只打印对应 target 的 cast/reinterpret/shift/where 序列。若目标库改变 API 名或调用形式，修改范围现在局限在该 target 的 `Syntax` 模块，不需要重新穿过 Plan 消费和 op 语义处理。

这次没有把 target 拼写塞入共享 Plan，也没有新增 target 方言字段；三个 target 的 spelling 仍各自独立，因为目标接口差异本来就属于 leaf。

## 没有保留的改动

两项看起来合理的简化在审查后被撤回：

- TileLang 的 physical-padding 临时拷贝消除：现有真实 repro 没有制造出该分支，无法证明它是当前生成质量问题；
- Triton 在掩码恒真时省略 `mask=True`：现有 repro 没有证明当前实际生成源码因它产生多余工作。

这两项都没有越过语义边界，但也没有满足“由真实成本逼出”的门槛，因此没有作为投机性清理留在代码中。

## 验证

执行并通过：

```bash
cmake --build /tmp/intentdsl-build --target intent-compile
./examples/run/repro.sh cutile moe
./examples/run/repro.sh cutile paged_attention
./examples/run/repro.sh triton block_scaled_matmul
./examples/run/repro.sh tilelang block_scaled_matmul
git diff --check
```

`block_scaled_matmul` 用来确认 E8M0 拼写搬家没有改变语义：Triton 与 TileLang 的最大误差均为 `7.62939453125e-06`，generated p50 分别为 0.0180 ms 和 0.0364 ms。

本轮没有因模块化打开新的 target capability，也没有新增 unsupported；它只改善已有 cuTile 投影，并使目标 API 变化的修改范围更局部。
