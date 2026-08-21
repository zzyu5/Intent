# Core 与算法库

## 构造分类

| 定性 | 构造 | 边界 |
|---|---|---|
| Core definition/ABI | `@intent.kernel`、`@intent.fn`、`In/Out/InOut`、runtime scalar、`Constexpr` | 一个 source kernel 对应一个 logical callable；helper 不产生 dispatch |
| Core logical work/control | domain、作者可观察的 region/partition、`parallel`、普通顺序 `for`、`state_stream`、ragged relation | domain/region/view 是语义对象；普通 physical blocking 不属于 public API |
| Core tensor-flow | positional broadcasting、reshape、transpose、pointwise、mask、cast、record、typed tuple/state | 保留作者的数据流和 shape，不保存 target layout |
| Core structured computation | `reduce`、`scan`、typed pure combiner、multiply/add `contract` | op 语义不指定一棵物理树；普通顺序 loop/state-stream 不会被归一化成 reduction |
| Core indexing/effects | gather、scatter、logical buffer、atomic、RNG、`I.end`、`I.assume_in_bounds` | index relation、effect、前置条件和 logical identity 属算法语义 |
| Core control | runtime `if/for/while` 与 source specialization branch | Python `break/continue` 在 frontend 正规化，不形成 Kernel IR op |
| 语法糖 | 固定 `reduce.max/sum`、`any/all`、`arg_reduce.max` | lowering 到同一 canonical reduce/typed combiner 语义；不是第二套 primitive |
| 格式语法糖 | `sparse_contract_2to4` | canonical 语义显式保存 `two_of_four` format、compressed/metadata/RHS axes、`i16` metadata 与 accumulator schema；2:4 intrinsic 是 convenience spelling |
| Core source-visible partition | `partition(extent=...)`、`partition(count=...)` | extent/count 必须被算法、ABI、effect 或 wrapper 观察；不能使用 `I.auto` 代替 physical tiling |
| 不属于 public Core | fence、stage、physical barrier、worker/grid identity | fence 没有完整同步合同；stage 是 compiler-private realization |

Target 可以只承接 Core 的能力子集，但不能反过来改变 Core 定性。缺少等价机械投影的构造必须在 emission 前按 target capability 诊断。

Core 不包含 opaque 的：

```text
generic group_by
generic sort
generic topk
generic histogram
softmax / FlashAttention / MoE 算子名
```

这些名字可能隐藏不同的 source algorithms，不能成为 realizer 偷换算法的入口。

## 算法库边界

算法库属于普通 host-side library：调用哪个 implementation，就是作者或 wrapper 选择哪个算法，而不是 realizer 根据算子名字替换 Kernel IR。

这样的库可以包含：

- portable Intent implementation；
- target-specific Intent/source variant；
- Triton、TileLang、cuTile、CUDA、CPU 或 RVV 手写实现；
- vendor library adapter；
- 缺少专用实现时的 portable fallback。

普通 Python wrapper 负责 target dispatch。更高层的 `intent.ops.*` 可以选择库实现，但这属于用户显式调用的 library policy，不是 realizer 在一个 `@intent.kernel` 内替换算法。

## Core 不是表达能力白名单

没有专用 structured primitive 时，作者仍可使用：

- 普通 `for`、`while`、`if`；
- logical buffer；
- dynamic indexing、gather/scatter；
- atomics 与 effects；
- `@intent.fn` 组合算法；
- wrapper 中的多个 kernels；
- target-specific variant。

Structured primitives 是后端优化锚点，不是语言表达能力的边界。

## Physical refinement 的边界

一个 Core node 可以使用多条目标指令、内部临时量、补偿步骤、不同物理树、microkernel 与 pipeline，只要这些共同实现同一 logical node。

```text
contract → TF32x3 composite MMA       合法
reduce   → 多级 private partial       合法
scan     → 分层并行实现               合法
GEMM     → Strassen                    不是同一 source node
stable softmax → online softmax        不是同一 source algorithm
atomic bucket → radix grouping         不是同一 source algorithm
```
