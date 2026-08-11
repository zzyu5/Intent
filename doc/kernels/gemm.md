# GEMM DSL 模板

## Canonical kernel

```python
import intent
import intent.language as I


class Activation(I.Enum):
    NONE = 0
    RELU = 1


@intent.kernel
def gemm(
    a: I.In[I.f16, ("M", "K")],
    b: I.In[I.f16, ("K", "N")],
    c: I.Out[I.f16, ("M", "N")],
    ACTIVATION: I.Constexpr[Activation],
):
    M, K = a.shape
    _, N = b.shape

    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)

    for mr in I.parallel(
        I.partition(m_axis, extent=I.auto("M_TILE"))
    ):
        for nr in I.parallel(
            I.partition(n_axis, extent=I.auto("N_TILE"))
        ):
            acc = I.contract(
                a[mr, k_axis],
                b[k_axis, nr],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )

            if ACTIVATION == Activation.RELU:
                acc = I.maximum(acc, 0.0)

            c[mr, nr] = I.cast(acc, I.f16)
```

## Source 固定

- M/N region-level output algorithm；
- K contraction 与 positional reduction axes；
- source operand dtype 与 f32 accumulator；
- activation specialization 与 epilogue；
- output 的显式 f16 narrowing。

## Physical Plan 决定

- M/N/K physical tile；
- program mapping、grid 与 grouped swizzle；
- worker hierarchy；
- 算法结构要求的 packing、storage/reuse boundary 与 contraction primitive 数值角色；
- launch ownership 与合法 tuner 参数轴。

各目标把 contraction role 拼成 MMA、`tl.dot`、`T.gemm`、cuTile matmul 或 CPU/RVV microkernel；fragment layout、寄存器分配、指令选择和给定候选后的低层 pipeline/prefetch 由目标 compiler 决定。

## 边界

一个 logical `contract` 可以由多条目标指令、补偿步骤与私有临时量实现。Ordinary GEMM 不能被 realizer 替换成 Strassen。

`BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps` 与 `num_stages` 不进入 source。
