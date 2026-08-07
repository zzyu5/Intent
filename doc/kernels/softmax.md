# Stable Softmax DSL 模板

## Canonical kernel

```python
@intent.kernel
def stable_softmax(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
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
- pure `exp` 的保存、重算或 spill；
- boundary predicate、tail loop 或 `vsetvl`；
- storage 与 launch。

## 边界

Realizer 可以改变 reduction tree 和物理顺序，并产生正常浮点差异；不能把 stable 算法替换为携带 `(m, l)` state 的 online softmax。
