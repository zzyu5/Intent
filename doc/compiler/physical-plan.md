# Physical Plan

Physical Plan 是 compiler-owned target realization，不是用户需要填写的 schedule DSL 或 approval contract。

## Plan 内容

```text
Extent
    auto region/subtile binding

Ownership
    region → program/CTA/thread/task
    static / grid-stride / persistent / swizzled

Storage
    register / shared / local / cache / hidden scratch

Layout
    tensor layout / fragment layout / packed representation

Primitive
    MMA / tl.dot / T.gemm / ct.mma / vector FMA / target collective

Pipeline
    prefetch / async copy / stage count / overlap

Boundary
    predicate / tail loop / padding / vsetvl

Launch
    grid / worker count / target attributes
```

Extent、ownership、storage、layout、primitive、pipeline 与 launch 相互耦合，realizer 可以联合生成和搜索候选。Cost model、搜索和剪枝是 compiler implementation，不属于 source semantics。

## Ownership 与 physical identity

Source 定义 logical region space：

\[
R=\{\text{logical region instances}\}
\]

Plan 构造 physical worker space：

\[
W=\{\text{program / CTA / thread / task}\}
\]

并定义：

\[
\operatorname{own}:W\rightarrow\operatorname{Seq}(R)
\]

因此同一 Kernel IR 可以采用 ordinary grid、grid-stride traversal、persistent worker、grouped swizzle、CPU thread ownership 或 RVV task + strip-mine。

`program_id` 是 emitter 对 ownership 的实现，不是 portable source identity。

## GEMM 中的 Plan 信息

GEMM source 保留 M/N region algorithm、K contraction、source dtype、accumulator 与 epilogue。Plan 保存：

- M/N/K tile；
- program mapping 与 grouped ordering；
- worker hierarchy；
- packing 与 storage；
- MMA/microkernel 与 fragment layout；
- pipeline 与 prefetch。

Triton 中的 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps` 与 `num_stages` 都属于这一层。

## 合法性边界

Plan 可以改变物理树、工作分配、storage 与 target-native floating-point mechanism，并产生正常浮点差异；不能改变 Kernel IR 保存的 tensor-flow、logical workset、state、effect、ABI 或 wrapper-visible partition。
