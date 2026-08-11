# Stable Softmax DSL 模板

## Canonical kernel

```python
ROW_MAJOR = I.constraints(
    strides=(None, 1),
    layout="row_major",
    noalias=True,
)


@intent.kernel
def stable_softmax(
    x: I.In[I.f32, ("M", "N"), ROW_MAJOR],
    y: I.Out[I.f32, ("M", "N"), ROW_MAJOR],
):
    M, N = x.shape
    cols = I.domain(0, N)

    for row in I.parallel(I.domain(0, M)):
        values = x[row, cols]

        m = I.reduce.max(values, axis=0, identity=-I.inf)
        z = I.exp(values - m)
        s = I.reduce.sum(z, axis=0, identity=0.0)

        y[row, cols] = z / s
```

## Source 固定

```text
full-row max → subtract → exp → full-row sum → divide
```

Source 明确选择 stable softmax，而不是 online recurrence。它同时固定 logical row/column domain、两个 reductions、identities、`exp` 与 output relation。

## Physical Plan 决定

- row 到 worker 的 ownership；
- column strip-mining；
- SIMD、warp、block 或多级 reduction tree；
- boundary predicate、tail loop 或 `vsetvl`；
- 算法结构要求的 storage class 与 launch ownership。

Pure `exp` 是否重算或 spill、寄存器分配和低层指令选择交给目标 compiler；它们不因某门 surface 要求显式语法就变成共享 Plan 决策。

## 边界

Realizer 可以改变 reduction tree 和物理顺序，并产生正常浮点差异；不能把 stable 算法替换为携带 `(m, l)` state 的 online softmax。
