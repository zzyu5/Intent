# 数值与确定性

Intent 是高性能 kernel DSL，不是逐项审批 compiler 数值自由的合同语言。

## 默认模型

1. Source 中显式 dtype、`cast`、数学操作与算法结构必须保持。
2. Realizer 与 selected backend 可以使用目标正常的高性能浮点实现。
3. Reduction tree、contraction mechanism、tile、物理执行顺序与 target 不同，可以产生正常浮点差异。
4. 默认不承诺跨 target、跨 Plan、跨 compiler 或跨运行逐 bit 相同。
5. 真正的 dtype narrowing 使用显式 `I.cast`。
6. Base-2 exponential 使用显式 `I.exp2`。
7. 严格数学或确定性路径属于 host/backend compile policy，不是每个 primitive 的 source 参数。

```python
compiled = intent.compile(
    kernel,
    target=device,
    options={
        "matmul_precision": "strict",
        "deterministic": True,
    },
)
```

具体 option 由 backend 定义。无法兑现时，backend 可以拒绝该 compile policy。

Portable source 不增加：

```text
ContractPrecision / ContractNumerics
MathAccuracy / max_ulp
AccumulationSemantics
per-op determinism contracts
@associative / mergeable
```

Backend 在 f32 contraction 内选择 IEEE、TF32、TF32x3 或其他 target-native mechanism，与 source 显式把一个 f32 value cast 成 f16/bf16 是两件事。

Logical validity 与数值 mask 分开。Realizer 可以依据 logical predicate 消除完全无效的 physical region；无法整体证明时生成 target predicate、tail loop 或 `vsetvl`。
