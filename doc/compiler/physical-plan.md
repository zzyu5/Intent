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
intent_plan.plan
    entry_name
    target: backend / architecture / device / warp_size
    extents: node / logical axis / logical extent / tile expression
    ownership: loop_node / worker_kind / worker_axis / traversal / mapping
    storage: value_node / storage_space / access_mode
    layouts: value_node / layout_kind / axis_order
    primitives: operation_node / primitive_kind / operator / axis / identity
    pipeline: stage policy / prefetch / async_copy
    boundaries: extent_node / logical_extent / tail / predicate / load fill
    launch: loop_node / grid policy / num_warps
```

这是一份 compiler IR，而不是开放字典。每类 binding 都必须能回指一个存在且种类相符的 Kernel IR 节点；backend emitter 只消费经过 Kernel IR verifier 与 Plan verifier 共同验证的组合。

当前 Plan 只由 C++ `intent-realize` 构造。Python frontend 到 Kernel MLIR 为止；host compiler 通过 stdin/stdout 调用 realizer，随后把同一个组合 MLIR 交给 C++ translator。项目中不保留 Python `PhysicalPlan`、Python Plan verifier 或 Python Plan serializer。

完整 Plan verifier 的长期合法性边界是：

- entry 与 ABI view 集合不变，storage/layout binding 完整覆盖这些 view；
- extent 的 source region 正是被绑定 loop 的迭代 region；
- boundary extent 等于 source domain extent，launch realization 完整覆盖该 extent；
- ownership 与 launch 指向同一个 loop，不能把另一个 region 的 worker identity 注入当前 loop；
- primitive binding 精确覆盖被实现 loop 内的 target primitive 候选，不能引用 helper 或其他 loop 中同名操作；
- storage access mode 不弱化 source view 的读写权限，layout 不改变 logical axes；
- pipeline 不重排违反 effect、ordered 或 state-stream 依赖的操作。

当前首个 stable-softmax verifier 已具体实现其中与该 kernel 有关的部分：ABI/storage/layout 完整覆盖、`M/N` source domain、row-major index relation、primitive def-use/axis/identity、masked boundary、persistent grid-stride ownership，以及 pipeline/launch policy 的一致性。该结构没有 `ordered`、state stream 或可重排 effectful intermediate；包含这些结构的 Plan 当前不会被接受，而不是假装已经完成通用依赖证明。动态 program count 由 translator 按 Plan 的 persistent-occupancy policy 和 JIT resource usage 计算，再以 grid-stride loop 覆盖全部 row extent。

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

## 首个 Triton stable-softmax Plan

首个可执行 realization 对应 canonical stable softmax，并产生：

- row extent 每次处理一行，column tile 为 `next_power_of_two(N)`；
- row loop 由 program axis 0 persistent ownership，并以 `num_programs(0)` grid-stride 遍历；
- input/output 位于 global storage，保持 row-major logical axes；
- 两个 reduction、两个 broadcast、subtract、exp 与 divide 分别绑定回 Kernel IR node；
- column boundary 使用 `column < N` mask，load fill 为 negative infinity，store 使用同一 predicate；
- stage policy 为 shared-memory threshold 上的 2/4 stages，`num_warps = 8`；
- grid policy 使用 target resource/occupancy 计算，并限制 program count 不超过 row extent。

Realizer 的共享 C++ analysis 按 operation 数量、region、ABI、index relation 与 def-use 识别 `load → max → broadcast → subtract → exp → sum → broadcast → divide → store`，不检查函数名。它随后采用确定 policy 构造上述 Plan；当前没有候选枚举、搜索或 cost model。该 Plan 只是第一份可执行实例，不代表其他 traversal 或通用 reduction realization 已完成。

## 合法性边界

Plan 可以改变物理树、工作分配、storage 与 target-native floating-point mechanism，并产生正常浮点差异；不能改变 Kernel IR 保存的 tensor-flow、logical workset、state、effect、ABI 或 wrapper-visible partition。
