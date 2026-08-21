# Host-visible Two-pass Reduction DSL 模板

当 partial buffer、part count 与 kernel 数量对 wrapper 可见时，它们属于 source program，而不是 compiler-private realization。

下面的 canonical 结构使用 `partition(count=P)` 表达 host-visible 两遍算法；不能把它改写成 `extent` 模式，因为两者的 source-visible part identity 不同。

## Canonical kernels

```python
@intent.kernel
def pass1(
    x: I.In[I.f32, ("M", "N")],
    partial: I.Out[I.f32, ("M", "P")],
):
    M, N = x.shape
    P = partial.shape[1]
    cols = I.domain(0, N)

    for row in I.parallel(I.domain(0, M)):
        for p, region in I.parallel(I.partition(cols, count=P)):
            partial[row, p] = I.reduce.max(
                x[row, region],
                axis=0,
                identity=-I.inf,
            )


@intent.kernel
def pass2(
    partial: I.In[I.f32, ("M", "P")],
    out: I.Out[I.f32, ("M",)],
):
    M, P = partial.shape

    for row in I.parallel(I.domain(0, M)):
        out[row] = I.reduce.max(
            partial[row, :],
            axis=0,
            identity=-I.inf,
        )
```

## Canonical wrapper

```python
def two_pass_max(x):
    parts = choose_parts(x.shape[-1], x.device)
    partial = torch.empty((x.shape[0], parts), device=x.device)
    out = torch.empty((x.shape[0],), device=x.device)

    pass1(x, partial)
    pass2(partial, out)
    return out
```

## Source 与 wrapper 固定

- 两个 runtime-visible kernels；
- invocation 顺序；
- partial buffer ABI 与 shape；
- source-visible `P` 和 `partition(count=P)`；
- 两个 kernel 内各自的 logical reduction。

`parts` 被 wrapper 和两个 kernels 共同观察，因此绝不是 `I.auto`。

## Physical Plan 决定

每个 kernel 内部仍可独立决定 region ownership、internal extent、reduction tree、storage、target collective 与 launch。

## 边界

Compiler 不把两个 source kernels 融合，也不改变 wrapper-visible 的两次调用与 partial ABI。单个 source callable 的 realizer 可以为其已有数据依赖选择 compiler-private stages，但这些 stages 必须由 Physical Plan 显式约束且对 wrapper 不可见。完全内部的 tiling 由 Physical Plan 直接引入，不形成 source partition；Core 不提供 `count=I.auto(...)`。
