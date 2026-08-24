# Kernel 与 Host 边界

## 1. Kernel definition

一个 `@intent.kernel` 定义一份 device kernel 算法。对给定 constexpr specialization 和 ABI，编译结果对应一个 target kernel artifact 和一次正式 device launch。

kernel：

- 通过 `In`、`Out`、`InOut` views 与 runtime scalars 接收 ABI；
- 不通过 Python return 返回 device tensor；
- 可以包含硬件无关的控制流、状态、structured operations 和 effects；
- 不分配或发射隐藏的第二个 kernel；
- 不决定 target、设备资源或 launch 配置。

编译器不得把一个 kernel 自动裂成多次 host launch，也不得创建对 host 不可见的跨 kernel workspace 与同步。

## 2. Multi-kernel algorithm

真实算法若需要多个 kernel，作者显式定义多个 `@intent.kernel`，并由普通 Python host code：

1. 选择 specialization；
2. 分配输出与中间 tensors；
3. 按算法顺序 launch；
4. 管理跨 kernel 可见的状态与生命周期。

这不是图编译。Intent 编译器逐个编译 kernel；Python wrapper 是多 kernel orchestration 的权威。

## 3. Helper function

`@intent.fn` 是 kernel 内的硬件无关 helper：

- 可以抽取纯计算或结构化的算法片段；
- 参数和结果是 DSL values；
- 不形成独立 kernel artifact；
- 不能承担 host dispatch、设备查询或 provider selection。

helper 是否 inline 是实现问题，不改变语言语义。

## 4. Runtime 与 constexpr

runtime 参数在每次 launch 时取值，可以参与数据计算和 runtime control flow。

`Constexpr[T]` 在 specialization 时绑定，可以控制硬件无关的算法变体，例如：

- causal 与 non-causal；
- 是否应用某个 activation；
- 算法定义中的 group count 或格式参数。

constexpr 不得用于查询或选择：

- GPU 型号或 compute capability；
- TMA/MMA/layout；
- warp、stage、tile、memory space；
- Triton/cuTile/TileLang provider form。

这些选择属于编译器、target 或下层 toolchain。

## 5. Kernel 内 `if`

kernel 内允许两类硬件无关条件：

- runtime `if`：条件来自 runtime values；
- algorithmic constexpr `if`：条件来自作者声明的 specialization 参数。

二者都属于同一个 kernel。它们不能被误解成多 kernel dispatch。

纯硬件条件不进入 Kernel IR。例如“设备支持 TMA 时使用 descriptor”不是作者算法分支；它由 target lowering 或下层 compiler 处理。

## 6. ABI 与资源

external view 的作者可见合同是 logical shape、element dtype、读写方向以及必要的 alias/precondition。具体 strides 可以由调用 ABI 携带，但作者不在 kernel 中手算地址步长。

kernel-local mutable buffer 是算法资源：作者可以读写它，但不指定它最终位于寄存器、shared memory、local memory 或 workspace。若一个中间 tensor 必须跨 kernel 存活，它由 host 显式分配，不再是 kernel-local buffer。
