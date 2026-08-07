# Host-visible Two-pass Reduction DSL 模板

当 partial buffer、part count 与 kernel 数量对 wrapper 可见时，它们属于 source program，而不是 compiler-private realization。

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

Compiler 不把两个 source kernels 融合，也不把任意一个 source kernel 拆成多个 runtime-visible dispatches。完全内部的 tiling 使用 `extent=I.auto(...)`；Core 不提供 `count=I.auto(...)`。
