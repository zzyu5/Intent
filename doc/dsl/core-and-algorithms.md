# Core 与算法库

## 小而通用的 Core

Core 包含：

```text
domain / region / view
partition / parallel / ordered / state_stream
pure tensor expressions
reduce / scan + typed pure combiner / multiply-add contract
gather / scatter
ragged descriptor
logical buffer
runtime control flow
atomic / effect / RNG
```

Core 不包含 opaque 的：

```text
generic group_by
generic sort
generic topk
generic histogram
softmax / FlashAttention / MoE 算子名
```

这些名字可能隐藏不同的 source algorithms，不能成为 realizer 偷换算法的入口。

## 算法库

Intent 可以提供明确命名的算法实现：

```python
intent.algorithms.bitonic_sort(...)
intent.algorithms.radix_sort(...)
intent.algorithms.heap_topk(...)
intent.algorithms.radix_select(...)
intent.algorithms.atomic_bucket(...)
intent.algorithms.count_scan_scatter(...)
```

调用哪个 implementation，就是选择哪个算法。

算法库可以包含：

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
