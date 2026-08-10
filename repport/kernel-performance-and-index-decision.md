# Intent Kernel 编译器关键机制与当前实测报告

## 结论

当前编译主链已经稳定为：

```text
Python DSL
  -> canonical Intent Kernel MLIR
  -> GPU Physical Plan
  -> shared op traversal
  -> Triton / cuTile / TileLang source
```

本轮闭环的不是若干 kernel 专用补丁，而是四组共享能力：

1. CUDA Graph 短 kernel 的 L2 隔离测量；
2. 动态索引表达式的精确地址保留、来源轴追踪和边界条件；
3. 由同一份 Physical Plan 驱动的 persistent 程序映射；
4. 可证明的消费者中和事实，使 TileLang 能安全采用整块搬运。

同时，LayerNorm backward 的 DSL 算法改为与高性能上游一致的分组部分和结构。上述路径都通过同一个 canonical Kernel MLIR、同一个 GPU realization 和同一个 emitter 框架，没有新增按 kernel 名字分派的分析器或发射器。

本报告只列当前代码树上重新执行过的受影响路径，不把更早报告中的全量 corpus 数字混入同一张表。

## 1. 表示和职责边界

### 1.1 Python frontend

Python 只负责语法、constexpr、符号和 region lowering，以及带源码位置的诊断。标量索引表达式直接 lowering 成普通 SSA 运算；Python 中没有与 MLIR 平行的 typed Kernel IR。

### 1.2 Canonical Kernel MLIR

Kernel MLIR 是算法语义的唯一真理，保存：

- parallel、ordered、reduction、contraction、state stream 等算法结构；
- view、index relation、effect 和数值语义；
- 作者明确写下的前置条件，例如 tensor-derived scalar index 的 in-bounds 合同。

本轮新增的 `intent.assume_in_bounds` 是算法接口前置条件，不是目标实现提示。它不被标为 pure，因此不能被死代码消除提前删掉。

### 1.3 GPU Physical Plan

Physical Plan 只保留必须从多个合法物理方案中选择的事实。本轮涉及两项：

- `intent_plan.program.persistent`：程序空间是否映射为固定 worker 集合，再由 worker 线性遍历 tile；
- `intent_plan.transfer.consumer_neutralized`：某次带逻辑边界的读取，其无效 lane 是否已被后续消费者链可证明地中和。

来源轴、边界域和 use-def 等分析结果只是从 Kernel MLIR 重算的派生索引，不构成独立 schema，也不是 emitter 的第三个事实来源。

### 1.4 Target emission

三个 emitter 都从 Kernel MLIR 与同一份 Physical Plan 取事实。共享遍历决定“发射哪个 op”，target leaf 只负责：

- 能力检查；
- 概念到目标 API/语法的映射；
- 把明确委托给下层的事项留给 Triton、cuTile 或 TileLang。

Persistent 的 tile 数、线性 worker 映射和程序索引不是三个 emitter 各算一遍；consumer neutralization 也不是 TileLang 根据 attention 名字猜出来的。

## 2. CUDA Graph 测量口径

### 2.1 问题

CUDA event 会统计区间内排入 stream 的全部 GPU 工作。对于 CUDA Graph，反复 replay 同一批地址还会形成稳定的 L2 驻留状态。

SwiGLU backward 的上游实现原地写回、主要触碰三个缓冲；生成实现保留独立输出、触碰五个缓冲。在 96 MiB L2 的 RTX 5090 D 上，不冲刷缓存会让上游工作集获得不对称的重放驻留优势，旧的约 `2.9x` 差距甚至违反按显存带宽估算的物理下限。

### 2.2 当前做法

所有 `cuda_graph=True` 的测量统一分配大于两倍 L2 容量的 flush buffer。每次测量在 start event 之前读取该缓冲，从而冲掉上一次 replay 的驻留数据；冲刷本身不进入计时区间。

输入、输出和 workspace 仍在计时区外预分配。计时区间只保留为了得到结果必须发生的 GPU launch。

### 2.3 重测结果

| Kernel | Generated Triton p50 | Upstream p50 | Generated / upstream | 数值 |
|---|---:|---:|---:|---|
| SwiGLU backward | 0.1121 ms | 0.1085 ms | 1.0327x | PASS |

结论：旧的约 `2.9x` 不是编译器性能缺口，而是图重放缓存驻留造成的测量偏差。修正后两者相差约 3.3%，generated 已处于同一带宽水平。

## 3. 动态索引：保留作者已经写下的表达式

### 3.1 原错误路径

偏移、整除等索引表达式在 frontend 和 Kernel MLIR 中原本已经是完整 SSA。错误发生在后段：分析没有沿标量 SSA 回溯来源轴，emitter 又把动态索引重新解释为一个物理轴并打印轴基址，导致 `row + 1` 中的 `+ 1` 被静默丢失。

### 3.2 当前实现

当前路径分成两个互补事实：

1. **地址值**直接使用已经发射的精确 SSA 表达式，不再由逻辑轴重建；
2. **所有权/来源轴**沿 cast、unary、binary 等标量 SSA 操作数回溯，只有能归到唯一逻辑循环轴时才建立来源关系。

对 derived exact address，三个 emitter 都机械生成：

```text
address >= 0 && address < logical_extent
```

恒真的比较交给下层常量折叠，不再额外建立 quasi-affine 求值器。

索引来自张量标量读取时，use-def 中不存在可推导的结构范围。作者必须用 `I.assume_in_bounds(index, view, axis=...)` 声明调用前置条件；编译器验证声明位于访问前、同一 block 且绑定正确 view/axis，目标代码不再额外钳制。

### 3.3 同时补全的表面映射

三个 GPU target 都已覆盖：

- 比较：`eq`、`ne`、`lt`、`le`、`gt`、`ge`；
- 整数索引运算：floor divide、remainder；
- 精确 derived scalar address 与相应边界谓词。

这只是补全已有 IR 到目标语法的映射，没有新增索引表达式 IR。

### 3.4 三种独立结构的实测

| 结构 | DSL 表达 | Triton p50 | cuTile p50 | TileLang p50 | 数值 |
|---|---|---:|---:|---:|---|
| 带偏移仿射索引 | `row + 1` | 0.0220 ms | 0.0221 ms | 0.0220 ms | 3/3 PASS |
| 多对一头映射 | `query_head // 4` | 0.0036 ms | 0.0037 ms | 0.0034 ms | 3/3 PASS |
| tensor-derived scalar gather | 标量读取 + `assume_in_bounds` | 0.0405 ms | 0.0566 ms | 0.0430 ms | 3/3 PASS |

这 9 次运行验证了三件不同的事，没有用一个“索引 kernel 模式”把它们合并处理。

## 4. LayerNorm backward：修改 DSL 算法，而不是伪造编译器优化

### 4.1 原算法差异

旧 DSL 第一阶段为每一行各写一份 `dw/db` 部分和，部分缓冲规模约为 `4096 × 4096`。高性能上游使用固定数量的分组，让多行先原子合并到小得多的部分和缓冲，再做第二阶段归约。

这是作者选择的算法编排，不是 realizer 能从 tile 大小自动推出的特化，因此修改发生在 DSL 源码。

### 4.2 当前算法

```text
group = row % 128

stage 1:
  计算 dx
  scatter-reduce atomic_add 到 partial_dw[group, feature]
  scatter-reduce atomic_add 到 partial_db[group, feature]

stage 2:
  沿 128 个 group 做 state-stream reduction
  写出最终 dw、db
```

运行接线在计时前分配并清零两个 `(128, 4096)` f32 workspace。多输出、作者主导的多 kernel 编排、跨并行边界的原子合并和第二阶段有序归约都通过现有 IR/Plan/emitter 机制表达。

### 4.3 当前结果

| Provider | Generated p50 | 数值 |
|---|---:|---|
| Triton | 0.1157 ms | PASS |
| cuTile | 0.0972 ms | PASS |
| TileLang | 0.1121 ms | PASS |

最大观测误差约为：`dx=0.001953125`、`dw≈2e-6`、`db≈1e-6`。

上游 Triton autograd wrapper 可以完成数值对照，但当前 adapter 无法把其内部 kernel 与 wrapper 拆开并纳入 graph capture，因此没有填写虚假的 kernel-only 延迟或比值。

## 5. 共享 persistent 程序映射与 batched GEMM

### 5.1 物理决策

Persistent mapping 是 GPU Physical Plan 的共享机器决策：

- 原逻辑程序轴折叠到固定 worker 轴；
- worker 以线性编号遍历总 tile 空间；
- 共享投影负责计算 program volume、线性 tile 到多轴索引的反解，以及 group index；
- target 只把相同映射打印成各自的 launch/grid 和循环语法。

当前结构条件是：contraction 位于至少三个 parallel 轴之下，并且至少两个轴需要分块。判断只读结构和轴角色，不读取 kernel 名称。

Ragged member 轴暂不进入 persistent 映射，因为其动态每序列最大 extent 尚未成为 Plan 合同的一部分。当前行为是保守地不选 persistent，而不是用静态 tensor shape 猜测总工作量。

### 5.2 三个 target 的机械投影

- Triton：固定 grid 上限为 SM 数，worker 在 kernel 内遍历后续 wave；
- cuTile：wrapper 根据 SM 数、`num_ctas` 和 occupancy 形成固定 grid，候选仍交给下层 tuner；
- TileLang：固定 worker grid，加串行 wave 遍历。

Emitter 没有为 batched GEMM 新增入口或整 kernel matcher。

### 5.3 cuTile 转置访问修正

Rank > 2 的转置 load 现在按真实物理顺序处理：

```text
ct.load
  -> ct.permute(last_two_axes)
  -> reshape(using permuted extents)
```

原路径先按未转置 extent reshape，方形 tile 会掩盖错误；非方形候选暴露出 18/24 个配置失败。修正后，NN/TN/NT/TT 每种布局都是 24/24 候选成功。

### 5.4 cuTile 与上游 BMM

| Layout | Generated / upstream p50 | 候选成功 | 数值 |
|---|---:|---:|---|
| NN | 1.0167x | 24/24 | PASS |
| TN | 1.0024x | 24/24 | PASS |
| NT | 0.9946x | 24/24 | PASS |
| TT | 1.0201x | 24/24 | PASS |

四种布局都进入上游约 ±2% 的范围，剩余差异不再支持“批量矩阵乘存在约一成通用编译器缺口”的旧判断。

同一物理机制也在 Triton 和 TileLang 上实际运行。Triton 四种布局全部数值通过，已记录的 TN/NT/TT p50 分别为 `0.1229/0.1020/0.1311 ms`；TileLang 四种布局全部数值通过，已记录的 NN/NT/TT p50 分别为 `0.1058/0.1079/0.1055 ms`。这两组当前没有同 scope 的上游 BMM 数字，因此不计算比值。

## 6. Consumer neutralization 与 TileLang 变长 attention

### 6.1 下层真实语义

TileLang 0.1.13 的 `T.copy` 会按完整 `Buffer.shape` 防止物理越界，源端越界 lane 安全填零。但 packed varlen attention 中，“越过当前 sequence_end、仍在 packed buffer 总 shape 内”的 lane 会实际读到下一序列；这不是物理越界，只能由后续逻辑有效性消除。

因此不能简单把所有有界 load 改成 bulk copy，也不该在 emitter 里写 attention 特判。

### 6.2 共享证明

GPU realization 为 transfer 计算 `consumer_neutralized`。证明以某个读取的每个边界 domain 为起点，沿全部 use-def 路径检查：

- materialized padding 必须同时匹配 value、logical axis 和 domain；
- pointwise 运算必须保持已知 padding 语义，不能把任意 unary/binary/compare/select 当成安全；
- contraction 中，仍存活的污染轴继续传播；被收缩的污染轴只有在配对 operand 对同一 domain 可证明为 zero 时才能停止；
- store/scatter 必须确认当前值就是 `value_operand_index` 指向的真实写入值，不能把作为地址索引的值误判为已消费；
- 无 user、轴关系不唯一、未知 op、重复 contraction pair 或无法证明的路径一律返回 false；
- 两个原始 load 不能仅凭各自期望的 fill 互相证明安全。

证明成立后，Plan 记录事实；TileLang emitter 只消费该事实：成立时选择 bulk `T.copy`，不成立时保持 full-tile fast path 与逐元素 guarded tail。

### 6.3 不重复下层工作

Bulk copy 前不再额外 `T.clear/T.fill`。物理 Buffer 边界由 TileLang 自身负责；我们只负责 packed sequence 的逻辑有效性。曾尝试保留 redundant clear，但它与流水化 copy 交互后使 noncausal 数值误差达到约 `0.039`，因此该做法没有进入当前实现。

### 6.4 当前结果

| TileLang varlen attention | Generated p50 | Upstream p50 | Generated / upstream | 数值 |
|---|---:|---:|---:|---|
| causal | 0.2323 ms | 0.2555 ms | 0.9092x | PASS |
| noncausal | 0.3925 ms | — | — | PASS |

同一形状上，旧逐元素谓词与同步路径的 causal 比值约为 `1.1064x`；共享合法性事实落地后，generated 比上游约快 9.1%。性能提升来自消除不必要的逐元素搬运与同步，不来自更换算法或把边界检查静默删掉。

## 7. 活体约束与相关回归

### 7.1 Stable softmax

形状为 `8192 × 8192`、dtype 为 f32。三个 provider 都数值通过：

| Provider | Generated p50 | Upstream p50 | Generated / upstream | 数值 |
|---|---:|---:|---:|---|
| Triton | 0.3659 ms | 0.3661 ms | 0.9994x | PASS |
| cuTile | 0.3686 ms | 0.3689 ms | 0.9993x | PASS |
| TileLang | 0.3535 ms | 0.3707 ms | 0.9536x | PASS |

这证明索引、persistent mapping 和 transfer 合法性重构没有破坏最早的 stable-softmax 活体约束。

### 7.2 TileLang ragged 与 grouped contraction

| Kernel | Generated p50 | Upstream p50 | 比值 | 数值与口径 |
|---|---:|---:|---:|---|
| MoE | 11.4183 ms | 9.3368 ms | 1.2229x | PASS；E2E 算法/ABI 不同，不作 compiler-kernel 结论 |
| grouped GEMM | 1.2747 ms | — | — | PASS |
| grouped GEMM tail | 1.2808 ms | — | — | PASS |

MoE 重跑用于确认 TileLang InOut wrapper 仍采用正确的 staged 语义。它当前的比值包含外层工作且两边算法/ABI 不同，只能作为功能回归和端到端观测，不能冒充 kernel-only 性能差距。

## 8. 当前仍明确存在的边界

### 8.1 Ragged 与 persistent 的组合

代码结构已经允许 ragged、有序流和 contraction 的轴角色共存，但 persistent program volume 尚不能从 Plan 读取动态的 per-sequence 最大 extent。为避免静默少算或多算，realizer 对该组合保守不选 persistent。现有 ragged kernel 正确运行，不受阻塞。

### 8.2 动态负整数的 floor/mod 语义

当前索引样本使用非负逻辑索引。动态负整数在 Python、C/C++ 以及三个目标语言中的 floor-division/remainder 精确语义尚未形成跨 target 合同，因此不能把当前非负结果外推为对任意负动态索引的证明。

### 8.3 不可拆分的上游 wrapper

LayerNorm backward 的上游入口无法在现有 adapter 中取得独立 kernel-only graph 时间。报告保留空值，没有用 wrapper 时间与 generated kernel 时间混比。

这些是被显式拒绝或明确留空的证据边界；本轮已运行路径没有通过默认值、异常吞噬或 kernel 名字特判绕过它们。

## 9. 可复现入口

所有验证共用唯一入口：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <kernel>
```

本轮关键实例可直接按同一入口执行：

```bash
./examples/run/repro.sh triton swiglu_backward
./examples/run/repro.sh cutile batched_gemm
./examples/run/repro.sh tilelang varlen_attention
./examples/run/repro.sh triton shifted_row_copy
./examples/run/repro.sh cutile grouped_query_head_add
./examples/run/repro.sh tilelang scalar_table_lookup
./examples/run/repro.sh triton layer_norm_backward
./examples/run/repro.sh cutile layer_norm_backward
./examples/run/repro.sh tilelang layer_norm_backward
```

入口统一完成 DSL lowering、`intent-compile` 构建、目标源码发射、真实 GPU 执行、数值对照和可取得时的上游性能对照；没有新增 test 目录、fixture 或另一套验证脚手架。

## 10. 对应实现提交

- `f0d2f22 fix cuda graph cache isolation`
- `ec41ee0 generalize indexed boundaries and gpu realization`

第一项修正短 kernel 的测量物理条件；第二项包含精确索引边界、显式 index 前置条件、比较映射、LayerNorm backward 分组算法、共享 persistent mapping、cuTile 转置 load，以及 consumer-neutralized transfer 的共享证明和三个 target 投影。
