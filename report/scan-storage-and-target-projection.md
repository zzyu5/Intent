# 扫描分块、私有存储与目标投影修复报告

## 结论

这一轮修正了三个过去由编译器“固定填错”的位置：

1. scalar-access scan 不再把整个逻辑轴一次性物化为一个目标 fragment，而是由共享 Physical Plan 给出可调 scan tile，并生成 chunk loop、块间 scalar carry、producer replay 与 owner-private workspace；
2. 私有逻辑缓冲不再无条件落全局 workspace，也不再按单个缓冲粗略判断，而是依据访问结构、物理容量和同一 program owner 下的保守同时存活集合，在 `private_scalar_array`、`private_vector`、`private_workspace` 之间选择一次；
3. 三个后端中几处数量级差距被逐项归因：布尔归约和动态本地索引是目标叶子投影质量，已经修正；TileLang 二维联合访问覆盖范围目前没有机械等价投影，因此删除串行伪支持并明确报不支持；转置和分页注意力仍是目标投影质量问题，没有为它们增加共享层机制。

最终固定表包含 89 个 case、267 个 provider-case：

| 状态 | Triton | cuTile | TileLang | 合计 |
|---|---:|---:|---:|---:|
| PASS | 89 | 89 | 86 | 264 |
| unsupported | 0 | 0 | 3 | 3 |
| downstream failure | 0 | 0 | 0 | 0 |

三个明确不支持的格子均在 TileLang：`atomic_compare_exchange`、`conv2d`、`variant_conv2d_reduce_order`。固定数字位于 `report/baseline/kernel-performance.csv`；本轮没有改动任何 source baseline 数字。

## 一、扫描轴不再永远填满

### 1. 原来的问题

原来的 `intent_plan.scan` 只保存 scan 节点、语义和结果驻留。对于 scan 结果随后被标量索引访问的程序，三个发射器仍把 scan 输入看作完整 fragment，并一次性对整个逻辑长度执行 scan。

这在短轴上可以运行，但长度 4096 的 `nonzero_compact` 与 `unique_consecutive` 直接暴露了问题：

- Triton 可以编译，但完整 fragment 造成明显开销；
- cuTile 两个 kernel 均无法在可接受时间内完成下层编译；
- TileLang 的 `unique_consecutive` 也无法完成可接受的首次 JIT。

问题不在算法 IR，也不在某个 kernel 缺一条特化路径。作者只写了逻辑 scan，物理块是编译器选择的，因此 scan tile、块间 carry 和结果物化方式都必须由 Physical Plan 填写。

### 2. KernelFacts 只提取已有算法事实

`KernelFacts` 中的 `ScanFact` 现在保存：

- scan 的逻辑轴；
- 可在每个 chunk 内重新发射的纯 producer slice；
- producer 中逃逸、必须同步物化的 SSA value；
- scan 结果是否只被 scalar-result gather 消费。

producer slice 不是新的算法表示。它来自 Kernel IR 的 use-def 链，只允许没有副作用、没有嵌套控制流、可以安全 replay 的普通值生产者。遇到 buffer、atomic、scatter、state stream、另一个 scan 或未知 operation 时直接诊断，不复制副作用。

这一层回答的是“作者已经写下的哪些运算必须在物理 chunk 内重放”，不推导另一个等价算法。

### 3. Physical Plan 明确保存物化决定

`intent_plan.scan` 现在保存：

- `semantics = scan_inclusive_add`；
- `axis_node` 与结果 `tensor_axis`；
- `result_space`；
- `carry_space`；
- `materialization`；
- `owner_nodes`；
- `producers`；
- `materialized_values`。

当前形成两条由消费方式决定的通用路径：

| 路径 | 使用条件 | 物理结果 |
|---|---|---|
| `fragment_access` | scan 结果继续作为 fragment 参与张量运算 | 目标原生 fragment scan |
| `scalar_access` | scan 结果由标量 gather 随机读取 | chunked scan + scalar carry + owner-private workspace |

verifier 同时约束：

- `scalar_access` 必须对应 `private_workspace`；
- `fragment_access` 必须对应 `private_fragment`；
- carry 当前只能是 `private_scalar`；
- workspace scan 必须有非空 producer slice和明确 owner；
- fragment scan 不得携带 replay slice 或 materialized values；
- persistent program、compiler-generated staged program 和非 scalar/reused owner 当前明确不支持 workspace scan。

最后一组限制不是后端私有规则，已经从三个 emitter 中收回共享 Plan 校验。

### 4. scan tile 进入下层搜索空间

scalar-access scan 的逻辑轴获得 `ordered` role，并拥有独立的 `traversal` range。range 的 tile role 为 `scan`、`scan_1` 等，不再等于整个逻辑 extent。

三个 runtime tuner 都提供 `32/64/128/256/512/1024` 的合法 scan tile 候选；realizer 只声明可搜索轴和合法结构，不建立 cost model，也不在共享层替下层选赢家。

### 5. 三个目标如何机械投影

三条发射路径都读取同一份 scan binding 和 traversal tile：

- Triton：外层 `tl.range` chunk loop，chunk 内 `tl.cumsum`，workspace 使用 `tl.load/tl.store`；
- cuTile：外层 chunk loop，chunk 内 `ct.cumsum`，workspace 使用 checked gather/scatter；
- TileLang：外层 chunk loop与内层显式有序循环，carry 使用 `T.alloc_local`，workspace 显式读写。

producer replay 由共享 operation registry 重新 dispatch 原有 per-op handler。发射器没有按 `nonzero`、`unique` 或 MoE 名字选择路径。

Triton 与 cuTile 的 carry 最终取当前 chunk 的最后有效 lane，而不是固定的 `tile - 1` lane，因此非整除尾块不会把物理 carry 错写为 padding lane 的值。TileLang 是逐个有效位置更新 scalar carry，天然满足同一语义。

### 6. 四个真实 scan 使用者

以下均为最终固定 p50/p95，单位 ms：

| Kernel | scope | Triton | cuTile | TileLang |
|---|---|---:|---:|---:|
| `sorted_nucleus_cutoff` | K | 0.0261 / 0.0282 | 0.0220 / 0.0228 | 0.0258 / 0.0268 |
| `nonzero_compact` | K | 0.1407 / 0.1411 | 0.2633 / 0.2659 | 0.1978 / 0.1985 |
| `unique_consecutive` | E | 0.3368 / 0.3371 | 1.0153 / 1.0165 | 0.3712 / 0.3725 |
| `moe_align_block` | E | 0.0972 / 0.1024 | 0.0888 / 0.0982 | 0.0822 / 0.0912 |

相对旧固定表：

- `nonzero_compact`：cuTile 从 downstream failure 变为 PASS；Triton p50 从 0.8270 降到 0.1407，TileLang从 1.6404 降到 0.1978；
- `unique_consecutive`：cuTile 与 TileLang 从 downstream failure 变为 PASS；Triton p50 从 1.5498 降到 0.3368；
- `moe_align_block`：三后端保持 PASS，Triton从 0.1594 降到 0.0972；cuTile 从 0.0757 变为 0.0888，TileLang从 0.0799 变为 0.0822，属于本轮仍存在的小幅回退；
- `sorted_nucleus_cutoff` 原本就是 fragment scan，生成结构没有被强行改成 workspace 路径，数字保持同一量级。

## 二、私有工作区不再一律填全局显存

### 1. 原来的问题

作者通过 `I.buffer` 表达 kernel 内可变状态，但不指定它位于寄存器、本地数组还是外层 workspace。旧实现为了保证总能运行，大部分动态缓冲直接选择全局设备 workspace。

这保证了可实现性，却没有完成编译器应承担的物理驻留选择。Smith–Waterman 的两行滚动状态、排序的局部数组、动态规划的 predecessor table 被混在同一个选择中，导致三个后端可能对同一算法付出完全不同的访存代价。

### 2. 访问事实的分类

`LogicalBufferFact` 保存：

- owner parallel region；
- 静态 shape 与 element type；
- 是否存在动态索引；
- 是否存在无法由 bounded iterator、仿射范围或作者前置条件证明的非结构化动态访问。

最后一项不再命名为“需要可寻址存储”。本地向量本身也可以动态寻址；真正决定退回 workspace 的是访问结构无法被当前本地表示安全、稳定地兑现。

### 3. owner 级容量规则

realizer 对每个逻辑缓冲计算 32-bit register unit：

```text
logical units = element count × ceil(element bits / 32)
vector units  = next_power_of_two(element count) × ceil(element bits / 32)
```

所有乘法和二次幂取整均检查整数溢出。

当前规则为：

1. 非结构化动态访问直接选择 `private_workspace`；
2. 很小的一维缓冲可以选择 `private_scalar_array`；
3. 其余可证明有界的动态缓冲尝试选择 `private_vector`；
4. 同一 program owner 下的所有候选按保守的“同时存活集合”累计物理容量；超出 owner 预算的缓冲退回 `private_workspace`。

当前 scalarization budget 是 `registersPerUnit / 1024`，owner-local budget 是 `registersPerUnit / 128`。这是一条简单、静态且保守的机器规则，不是 cost model，也不是性能证明。当前没有做精确 liveness；同一 owner 下的候选按同时存活处理。

### 4. 三种计划内驻留

| Plan space | 含义 | 目标投影 |
|---|---|---|
| `private_scalar_array` | 很小、可展开的 owner-private state | 标量变量或一元素 local allocation |
| `private_vector` | 容量允许、动态索引可证明有界的 owner-private state | Triton/cuTile register fragment，TileLang `T.alloc_local` |
| `private_workspace` | 非结构化动态访问或超出本地预算 | 外层预分配的 global device workspace |

workspace 的容量和地址只有一份共享投影：

```text
workspace size = product(owner program extents) × product(logical buffer extents)
offset         = row-major(owner indices, logical indices)
```

三个叶子不再各自重新决定 owner 线性化、逻辑秩或驻留位置，只负责参数拼写、分配 API 和 load/store 语法。

workspace 当前只允许 scalar、unreused program owner；persistent 或 compiler-generated staged program 明确拒绝。原因是当前外层 wrapper 与 stage ABI 没有证明 owner-private 状态能够跨这些映射保持唯一所有权，不能用一条未经证明的全局内存路径假装支持。

### 5. 动态本地索引的目标原语

相同 Plan 对三个 surface 的投影不同，但不引入新的物理决定：

- Triton 非布尔 local vector 读取使用 `tl.gather`；
- cuTile 非布尔 local vector 读取使用 `ct.extract(..., shape=(1,)).item()`；
- TileLang 直接使用 `T.alloc_local` 的动态索引；
- 布尔向量在 Triton/cuTile 保留 lane predicate + integer reduction，因为对应 local gather/extract 的布尔支持与收益没有同等保证。

多维 local vector 在三个目标中统一按 row-major 线性化。目标叶子只打印这一共享形状的地址表达式，不重新选择布局。

### 6. 被实际运行否决的两条错误规则

本轮没有把第一次实现当成结论：

1. “只要单个 buffer 容量装得下就放本地”的规则让 NMS 的 1024 项长期状态进入 local vector，cuTile 首次 JIT 两小时仍未完成，随后运行还出现资源失败。规则被改为 owner 级累计容量；NMS 自动退回 workspace 后三后端重新可运行。
2. “有作者的 in-bounds 声明就一定适合 local vector”会把 Bitonic 的 XOR partner 随机交换也搬进寄存器向量。Triton 明显变慢，cuTile 资源失败。最终把这种无法由结构范围分析直接兑现的访问归为非结构化动态访问，继续使用 workspace。

这两次失败都没有转化成 NMS 或 Bitonic 名字特判，修正的是共享驻留判据。

### 7. 六个私有缓冲 kernel

以下均为最终固定 p50/p95，单位 ms：

| Kernel | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| `insertion_top_k` | 0.5223 / 0.5257 | 0.4807 / 0.4843 | 0.7081 / 0.7101 |
| `bitonic_sort` | 1.4030 / 1.4081 | 0.7039 / 0.7052 | 0.5710 / 0.5731 |
| `greedy_nms` | 27.7411 / 27.7489 | 58.0208 / 58.0267 | 45.9212 / 45.9241 |
| `radix2_fft` | 1.3369 / 1.3478 | 0.7494 / 0.7516 | 0.5106 / 0.5131 |
| `smith_waterman` | 1.1796 / 1.1823 | 0.2986 / 0.3007 | 0.2427 / 0.2436 |
| `viterbi_decode` | 8.4866 / 8.4891 | 28.9815 / 28.9844 | 5.9107 / 5.9125 |

主要变化：

- Smith–Waterman：Triton从 47.6296 降至 1.1796，cuTile 从 0.4183 降至 0.2986，TileLang从 9.4866 降至 0.2427；
- Greedy NMS：Triton从 77.4252 降至 27.7411，cuTile 从 106.2183 降至 58.0208，TileLang从 51.2142 降至 45.9212；
- Bitonic TileLang：从 1.7169 降至 0.5710；
- FFT：Triton和cuTile分别从 2.0441、0.9953 降至 1.3369、0.7494；TileLang从 0.4698 变为 0.5106，约回退 8.7%；
- Viterbi 的 Triton/cuTile基本不变，TileLang从 5.6541 变为 5.9107，约回退 4.5%；
- Top-k 保持同一量级。

这些回退没有被隐藏成总体 PASS，也没有在本轮被归因到某个确定机制。CSV 只记录当前固定数字，不把推测写成结论。

## 三、跨后端巨大差距的源码归因

### 1. TileLang 布尔归约：已修叶子投影

旧 TileLang 路径把布尔归约发射成低质量逐元素结构，p50 为 9.2256 ms；这与 Triton/cuTile 的约 0.004 ms 相差三个数量级，不可能由算法或机器计划解释。

当前 TileLang 投影先把 bool fragment 转为 i32，再用 `T.reduce_max` 表达 any、用 `T.reduce_min` 表达 all，最后恢复 bool：

| Triton | cuTile | TileLang |
|---:|---:|---:|
| 0.0036 / 0.0049 | 0.0036 / 0.0044 | 0.0057 / 0.0065 |

这是 per-op target projection 的修复，没有增加布尔归约 kernel 分支。

### 2. Smith–Waterman：已修动态 local projection

地址投影统一后，Smith–Waterman 仍存在巨大差距，说明“重复 workspace 地址字符串”不是根因。并排阅读生成源码后发现，Triton/cuTile 对 local vector 的动态标量读取仍使用全 lanes 比较加归约，而 TileLang 有直接动态 local indexing。

非布尔读取改用各自已有的 local extraction primitive 后，三后端均数值通过，差距从约 47.6/0.42/9.49 ms 收敛到 1.18/0.30/0.24 ms。Triton仍慢于另外两个 surface，但已经不再是错误的全向量选择路径。

### 3. TileLang Conv2D：删除伪支持，明确能力边界

Conv2D 的共享 Plan 已经表达两个输出 ownership 轴及其各自更大的输入 access footprint。问题发生在 TileLang 投影：当前 surface 路径不能把两个带边界检查的轴联合投影为一次 cooperative transfer。

旧路径把它展开成串行逐元素搬运，虽然数值 PASS，但 p50 约 4.97 ms；这不是可接受的后端支持。当前 emitter 在原始 `intent.view_load` 源码位置明确报告：

```text
TileLang cannot project a multi-axis checked access footprint as one cooperative transfer
```

因此当前状态为：

| Kernel | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| `conv2d` | 0.0696 / 0.0717 | 0.0614 / 0.0621 | unsupported |
| `variant_conv2d_reduce_order` | 0.0950 / 0.0979 | 0.0425 / 0.0430 | unsupported |

这里没有补一个 TileLang 私有调度决策，也没有把串行慢路径继续标为 PASS。

### 4. 仍然存在的目标投影质量问题

并排阅读后没有发现 Physical Plan 少一个决定，但两个 TileLang 路径仍明显落后：

| Kernel | Triton | cuTile | TileLang | 当前判断 |
|---|---:|---:|---:|---|
| `matrix_transpose` | 0.0926 / 0.0944 | 0.0943 / 0.0987 | 0.6053 / 0.6083 | TileLang 用显式 `T.Parallel` 标量索引兑现 fragment permutation，功能闭合但原语质量较低 |
| `paged_attention` | 0.2248 / 0.2272 | 0.3620 / 0.3652 | 1.3210 / 1.3250 | ragged、双层 stream、GQA 映射和 stop 均已在 Plan；TileLang 目标投影仍约慢于 Triton 5.9 倍 |

这两处没有通过增加共享字段或 target 私有 realizer 路径来“解决”。当前事实是功能 PASS、目标投影质量未立住。

## 四、边界是否仍然干净

### 1. Realizer 与 emitter 的职责

本轮形成的最终数据流仍是：

```text
Kernel IR
  └─ use-def、轴、访问和消费事实
       ↓
GPU Physical Plan
  └─ scan tile role、materialization、carry、owner、buffer residency
       ↓
Triton / cuTile / TileLang emitter
  └─ capability check、目标原语映射、ABI 与运行接线
```

扫描的 producer slice、workspace owner 和 materialized values 均由 Plan 显式携带；三个 leaf 不从 tensor shape 或 kernel 名重建。私有 workspace 的 owner 线性化与容量公式只有一份共享实现。

### 2. 删除的重复逻辑

三个 emitter 原先分别验证：

- private workspace 是否能和 persistent mapping 共存；
- 是否能跨 compiler-generated stages；
- owner 是否 scalar、unreused；
- scan workspace owner 是否满足同样条件。

这些都是同一机器计划的生命周期合法性，不是 Triton/cuTile/TileLang 语法差异。现在统一由 GPU Plan verifier 检查；叶子只建立参数名、分配 workspace 并发射 load/store。

### 3. 没有新增的东西

本轮没有：

- 新增 kernel 名分支或整 kernel matcher；
- 新增 target-specific realizer；
- 将作者的 scan 改写成另一种有序算法；
- 建 cost model；
- 新建 test/fixture/pytest 基础设施；
- 改动 source baseline 计时或 source 数字。

## 五、验证范围与最终状态

验证只使用已有的手动 repro 入口：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <kernel>
```

实际覆盖：

- 四个 `I.scan` 使用者的三后端数值与性能；
- 六个 logical-buffer 使用者的三后端数值与性能；
- 三后端布尔归约；
- 三后端 softmax 活体约束；
- TileLang Conv2D 源码位置诊断。

softmax 最终保持数值 PASS，固定 p50/p95 为：

| Triton | cuTile | TileLang |
|---:|---:|---:|
| 0.3646 / 0.3670 | 0.3686 / 0.3707 | 0.3538 / 0.3561 |

全表 scope 分布为 69 个 K、17 个 E、3 个 R。当前没有 downstream failure；不支持项均给出明确诊断，没有静默 fallback。

## 六、仍然明确存在的限制

1. 当前 structured scan 只闭合 `inclusive add`；不是任意 combine 的通用 scan。
2. chunked workspace scan 当前服务于 scalar gather 消费；producer replay 只接受纯、可重放的 operation slice。
3. scan/private workspace 不支持 persistent program 或 compiler-generated staged program；当前 ABI 没有证明跨这些映射的 owner-private 生命周期。
4. private residency 使用静态 owner 容量规则，没有精确 liveness，也不等于 occupancy 保证。
5. `private_workspace` 当前仍由外层分配 global device memory；没有在共享层决定 shared memory 或其他 target-private 层级。
6. TileLang 二维联合 checked footprint 当前明确不支持。
7. TileLang transpose、TileLang paged attention，以及 cuTile MoE/TileLang FFT/TileLang Viterbi 的性能差距或小幅回退仍然存在；本轮没有给它们写未经证实的归因。

## 七、提交

| Commit | 内容 |
|---|---|
| `4fab97d` | scan tile/carry/materialization、producer replay、owner 级私有驻留、共享 workspace 投影和三目标机械发射 |
| `8fbfc6d` | 更新固定性能 CSV，删除已经被新实现取代的旧实现报告 |
