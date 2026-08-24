# Types、Numerics 与 Effects

## 1. Scalar 与 tensor types

Core 至少区分：

- `bool`；
- logical `index`；
- 有符号/无符号整数；
- 浮点与低精度浮点；
- records/tuples 形式的 typed accumulator；
- external view；
- kernel-local logical buffer。

tensor value 的 rank 与 logical extents 来自 domains/regions 和形状变换。physical lane shape、padding 和 target encoding 不进入 Core type。

## 2. Index 语义

logical index 与地址计算分开：

- logical index 的整除、取模、比较和转换具有跨 target 唯一语义；
- 负整数行为必须由语言定义，不能继承目标语言偶然差异；
- physical address width 由 compiler 根据 shape/stride 上界保证，不是作者选项；
- tensor-derived index 可以依赖作者前置条件，compiler 不自动 clamp。

## 3. Numeric contract

作者能够明确表达：

- 输入、输出与 accumulator dtype；
- cast、bitcast、舍入和饱和；
- exact/approximate math；
- reduction/scan reassociation；
- NaN、无穷、空归约与 tie-break 行为；
- counter-based RNG 的 seed/counter 映射。

编译器不能为了选择更快 primitive 偷换 accumulator dtype、近似函数或算法公式。

## 4. Packed、quantized 与 sparse data

如果存储编码改变 logical value 的解释，编码合同属于作者可见语义。例如：

- nibble/bit-field 的 signedness 与 bit order；
- scale/zero-point/group relation；
- sparse metadata 格式与压缩轴；
- E8M0/FP8 等 scale encoding。

这些合同应由 typed format/descriptor 或显式 decode arithmetic 表达，而不是为每一种格式建立永久独立的 whole-op。一次解包多少元素、解码值放在哪一级存储、是否使用 native MMA 则由 compiler/target 决定。

## 5. External views

view annotation 定义：

- element dtype；
- logical shape；
- read/write direction；
- 必要的 alias 与调用前置条件。

stride 可以由运行时 view descriptor 携带，但作者不在算法中手传或手算每个 target 的地址步长。alignment/layout 只有在它们是调用 ABI 的真实前置条件时才允许声明；不能作为性能 hint 混入算法。

## 6. Logical buffer

logical buffer 是 kernel 内可变资源。作者定义 shape、dtype、初始化与读写顺序，不指定 physical residency。

compiler 可以把它实现成 SSA、寄存器、local/shared memory 或显式 workspace，只要 lifetime、sharing、effects 和数值语义保持。跨 kernel 存活的 tensor 不属于 logical buffer，由 host 显式拥有。

## 7. Effects

effect 至少区分：

- external read/write；
- logical-buffer read/write；
- unique scatter；
- reduction scatter；
- atomic read-modify-write。

effect target、冲突语义、scope/order 是算法合同。target language 没有等价能力时应明确拒绝，不能静默串行化成慢路径或弱化内存语义。

## 8. Preconditions

作者可以声明不能从类型和 shape 单独证明的调用前置条件，例如：

- index in bounds；
- indices sorted/unique；
- views non-alias；
- extent 满足算法要求。

precondition 不生成运行时 clamp 或修复代码。违反前置条件是调用方错误；compiler 可以用它证明合法 lowering。
