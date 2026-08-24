# Intent 编程模型

## 1. 定位

Intent 是结构化的算子 kernel DSL。作者描述一份硬件无关、可由单次 device launch 执行的 kernel 算法；编译器为目标机器补出程序实例映射、物理分块、值表示、访问、存储与目标语言程序。

Intent 既不是图编译器，也不是完整算子库选择器。它的编译单位是一个 kernel definition 的 specialization，而不是一个模型图、Python 函数调用树或预写 kernel 模板。

## 2. 作者写下什么

作者拥有所有会改变程序可观察结果的算法事实：

- 输入、输出、可变参数和别名合同；
- logical domains、regions、ragged/sparse relations 与 logical identity；
- 硬件无关的 `if`、`for`、`while`、状态转移、停止条件和顺序依赖；
- 张量索引、广播、形状变换和数值表达式；
- reduce、scan、contract 的 axes、combiner、identity、累加类型和精度合同；
- gather、scatter、atomic、mutable buffer、RNG 和其它 observable effects；
- 一个算法由一个 kernel 还是多个 kernel 与 host wrapper 组成。

这些内容构成 canonical Kernel IR 的语义权威。编译器可以分析和引用它们，但不能静默替换成另一种算法。

## 3. 作者不写什么

下面这些信息只有在选定具体机器或 target surface 后才有意义，不属于作者算法：

- program id、launch grid、thread block、warp、lane 或 CPU vector lane；
- physical tile、chunk、pack width 与目标相关的静态 block shape；
- 寄存器、shared memory、TMEM、local memory 等存储层级；
- pointer tensor 展开、物理地址宽度、尾块 mask 与 neutral fill；
- copy primitive、MMA variant、layout、swizzle、pipeline、barrier 和同步协议；
- Triton/cuTile/TileLang API form、autotune candidate 与 winner；
- provider capability 或设备型号分支。

作者可以声明算法前置条件，例如索引范围、别名关系或必需的数值精度；这些是合同，不是物理实现提示。

## 4. Region 不是 tile

domain 是一组有序的 logical indices；region 是 domain 的逻辑子集或有序子区间。它们不归属于某个 program、block、warp 或线程，长度可以在运行时确定，也不要求二次幂。

region 可以改变 body 能看见的逻辑数据，因此属于算法模型。physical tile 只改变同一逻辑程序怎样被机器执行，因此不进入 DSL。

例如，作者可以表达“对一条变长序列的有效前缀做归约”，但不表达“每个 CTA 处理 128 个 token”。前者决定算法读取集合，后者由编译器选择。

## 5. 程序式与张量式语义并存

Intent 不是纯表达式图。kernel body 可以同时包含：

- 对完整 logical regions 的张量运算；
- 具有 SSA merge 的条件控制；
- 有序循环和循环携带状态；
- 明确可重结合的 reduce/scan；
- contraction、ragged relation 与间接访问；
- external view 与 kernel-local mutable resource 上的 effects。

张量式构造不会消除程序顺序；程序式控制也不会强迫作者写机器执行单位。编译器必须同时保持 logical tensor-flow 和 control/effect semantics。

## 6. 可观察语义与实现自由

可观察语义包括：

- 输出值、输出形状与 dtype；
- external effects、原子冲突合同和别名行为；
- 作者声明的顺序、state transition 与 stop condition；
- 数值精度、累加、舍入、近似函数和 reassociation 合同；
- host 可见的 kernel 数量、调用顺序和中间 tensor ABI。

只要保持这些语义，编译器可以在独立 physical program 中改变程序实例映射、物理循环、SSA 表示、重算/物化、访问形式和 target primitive。它不能修改 canonical Kernel IR 来掩盖这种变换。

## 7. 规格组成

- [`kernel-and-host.md`](kernel-and-host.md) 定义 kernel、helper、specialization 与 host orchestration；
- [`logical-program.md`](logical-program.md) 定义 domain、region、control、state、relations、structured operations 和 effects；
- [`../dsl/`](../dsl/) 定义 Python DSL 表面。
