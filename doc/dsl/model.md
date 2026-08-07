# 语言定位与边界

## 定义

Intent 是一门 Python-hosted、单-kernel、跨后端、tile-parametric 的 Structured Tensor-Flow DSL。

用户在一个 `@intent.kernel` 中写出完整的 kernel 内算法：logical domain、region、tensor-flow、顺序或独立工作、carry state、structured computation、控制流与 effects。用户不写物理 worker identity、内部 tile、grid、地址运算、storage placement、fragment layout 或 pipeline。

最短定义是：

> Intent 保留完整的单-kernel 算法，抽掉该 kernel 对具体机器 realization 的绑定。

形式上：

\[
\operatorname{compile}_{t}(K[s]) \rightarrow (E_t, L_t)
\]

- \(K\)：Intent source kernel；
- \(s\)：用户 specialization；
- \(E_t\)：目标上的一个 callable kernel entry；
- \(L_t\)：该 entry 的 launch configuration。

一次 source-kernel invocation 对应一次 target-kernel invocation。

## 在完整程序中的位置

Intent 是嵌入普通 Python 模块的 eDSL，不是独立的 `.intent` 文件语言。

```text
Python / framework wrapper
        ↓
一个可 dispatch 的 Intent kernel entry
        ↓
设备执行
```

它替换 Triton、TileLang 或 cuTile kernel 所在的位置，不接管其上方的计算图。

普通 Python wrapper 负责：

- 检查 shape、dtype 和 device；
- 分配输出与跨 kernel workspace；
- 绑定用户 specialization；
- 按作者写定的顺序调用一个或多个 kernels；
- 根据 target、shape、dtype 或库策略选择 source variant；
- 注册 framework custom op 与 autograd 接口。

多-kernel 算法仍由 wrapper 明确表达。Intent 不自动融合多个 source kernels，也不把一个 source kernel 拆成多个 runtime-visible dispatches。

## 权限分界

Source 固定：

- kernel ABI、输入输出、alias 与 effects；
- 算法阶段、数据遍数、状态 schema 与更新；
- logical domain、region 与 indexing relation；
- `parallel`、`ordered`、`state_stream`；
- `reduce`、`scan`、`contract` 的组合；
- stable、online、multi-pass 等算法选择；
- 显式 dtype、`cast` 与数学表达；
- gather/scatter 的索引和冲突语义；
- runtime control flow 与用户 specialization；
- wrapper、输出或其他 kernel 可见的 partition count/extent。

Realizer 决定：

- 内部 `auto` extent 与 sub-tiling；
- region 到 program、CTA、thread、task 的 ownership；
- grid、grid-stride、persistent worker 与 swizzle；
- address calculation、coalescing、tail 与 physical mask；
- register/shared/local/cache/scratch placement；
- physical layout、packing 与 fragment layout；
- target contraction、reduction 和 scan primitive；
- pipeline、prefetch、async copy、unroll 与 launch。

根规则是算法可观察性：

> Realizer 可以自由改变物理实现，但不能改变 source 的 tensor-flow、logical workset、state、effect、ABI 或 wrapper-visible 约定。

因此 pure expression 可以 CSE、融合、重算或 spill；reduction 可以选择不同物理树；f32 contraction 可以使用目标正常支持的机制。只有真正选择了不同算法时，才需要不同 source。

## 明确不属于 Intent Core

Intent Core 不做：

- graph partition 或 framework operator fusion/fission；
- 自动决定一个 framework op 使用几个 kernels；
- 改变 kernel 调用顺序；
- stable softmax 与 online softmax之间的算法替换；
- ordinary GEMM 与 Strassen 之间的算法替换；
- atomic bucket 与 radix grouping 之间的算法替换。

Portable source 不暴露 `program_id`、block/thread/warp id、grid、`num_warps`、`num_stages`、物理 address space、MMA fragment layout、physical barrier 或 pipeline schedule。
