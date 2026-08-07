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

## Plan IR

Physical Plan 使用 Kernel IR 的稳定 operation/value ID 引用逻辑节点，不复制 source algorithm，也不依赖函数名或变量名识别 kernel：

```text
PhysicalPlan
    entry_name
    target: backend / architecture / device / warp_size
    extents: loop_node / source_region / tile_size
    ownership: loop_node / worker_kind / worker_axis / traversal
    storage: value_node / storage_space / access_mode
    layouts: value_node / layout_kind / axis_order
    primitives: operation_node / primitive_kind / operator
    pipeline: stages / prefetch / async_copy
    boundaries: loop_node / logical_extent / tail_kind
    launch: loop_node / grid / block_size / num_warps
```

这是一份 compiler IR，而不是开放字典。每类 binding 都必须能回指一个存在且种类相符的 Kernel IR 节点；backend emitter 只消费经过 Kernel IR verifier 与 Plan verifier 共同验证的组合。

Plan verifier 至少保证：

- entry 与 ABI view 集合不变，storage/layout binding 完整覆盖这些 view；
- extent 的 source region 正是被绑定 loop 的迭代 region；
- boundary extent 等于 source domain extent，launch grid 完整覆盖该 extent；
- ownership 与 launch 指向同一个 loop，不能把另一个 region 的 worker identity 注入当前 loop；
- primitive binding 精确覆盖被实现 loop 内的 target primitive 候选，不能引用 helper 或其他 loop 中同名操作；
- storage access mode 不弱化 source view 的读写权限，layout 不改变 logical axes；
- pipeline 不重排违反 effect、ordered 或 state-stream 依赖的操作。

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

## 首个 Triton pointwise Plan

首个可执行 realization 接受一个静态一维 f32 logical loop，并产生：

- `tile_size = 256`，一维 program ownership 与 static traversal；
- ABI views 位于 global storage，layout 为一维 contiguous；
- loop 内每个受支持的二元算子绑定为 pointwise primitive；
- `stages = 1`，不启用 prefetch 或 async copy；
- 整除时使用 exact boundary，否则使用 `index < logical_extent` 的 masked tail；
- `grid = (ceil_div(logical_extent, tile_size),)`。

这不是按 `vector_add` 名字匹配的模板。任何满足同一结构和类型约束、且只包含已支持 add/subtract/multiply/true-divide 的 kernel 都可进入该 realization；helper、多维/动态 view、非 contiguous layout、reduce/contract、复杂 pipeline 等尚未实现的组合直接产生 `NotImplementedError`。

## 合法性边界

Plan 可以改变物理树、工作分配、storage 与 target-native floating-point mechanism，并产生正常浮点差异；不能改变 Kernel IR 保存的 tensor-flow、logical workset、state、effect、ABI 或 wrapper-visible partition。
