# Intent DSL

## 1. 语言表面

Intent DSL 使用受限 Python AST 表达一个 kernel 的硬件无关算法。Python 负责可读语法；语言语义由 Intent Core 定义，不等同于任意 Python 执行。

这组文档定义理想作者表面，不以当前 frontend 是否已经接受某个拼写为依据，也不包含 compiler、backend 或性能状态。

## 2. 三层构造

### Core

不能在不丢失算法语义的情况下删除：

- kernel ABI、runtime/constexpr 参数；
- domain、region、logical index 与 index relation；
- structured control、parallel independence 与 state carry；
- typed tensor/value operations；
- generic reduce、scan、contract；
- ragged/sparse/packed data semantics；
- external effects、logical buffers、scatter、atomic 和 RNG。

### Sugar

可以在 frontend 中无损归一化为 Core：

- `reduce.sum`、`reduce.max`、`any`、`all`；
- `arg_reduce.max`；
- Python `range`、条件表达式、`break`、`continue`；
- `zeros`、常用 activation 和其它纯 helper；
- `partition(count=P)`，当它只是 part domain 与连续 subdomain 公式的简写时。

### 不属于 DSL

- `auto("TILE")` 和 physical extent；
- program id、grid、warp、thread、lane；
- register/shared/TMEM/local storage；
- pointer arithmetic、physical mask、padding；
- target primitive、layout、copy、pipeline、barrier；
- provider capability 分支和 autotune config。

## 3. 设计纪律

一个新构造只有同时满足下面条件才应进入 Core：

1. 真实算法无法由现有 Core 自然表达；
2. 去掉它会改变 logical values、control/state、effects、ABI 或数值合同；
3. 它在不同硬件族上仍然有同一算法含义；
4. 它不是下层语言已经能够从普通目标程序完成的 physical lowering。

如果一个需求只是让作者手写 tile、storage、copy 或 provider hint，答案应是改进编译器，而不是扩 DSL。

## 4. 文档

- [`core.md`](core.md)：Core 构造和语义；
- [`types-numerics-and-effects.md`](types-numerics-and-effects.md)：类型、数值、views、资源与 effects；
- [`examples/`](examples/)：理想化的完整 DSL 示例。
