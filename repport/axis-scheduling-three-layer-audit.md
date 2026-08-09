# Intent Kernel 逐轴调度与三层表示重构验收报告

> 验收日期：2026-08-10
>
> 验收范围：本轮调度决策、物理 Plan、Triton/cuTile/TileLang emission 重构
>
> 验收集合：现有 11 个 kernel，不新增 kernel

## 结论

本轮要求的代码重构已经完成：调度入口不再判断 kernel 属于哪一类，而是针对每个逻辑轴分别确定角色和物理映射；从算法 IR 到目标源码之间只保留一份持久化 GPU 物理 Plan；三个后端共用 canonical Kernel IR 遍历和物理组件索引，不再经过各自的 target dialect 或另一份持久化 projection。

最终重新执行了 11 个 kernel × 3 个后端，共 33 条 repro。全部完成以下链路并通过数值比较：

1. Python DSL lowering 为 canonical Intent Kernel MLIR；
2. C++ realizer 构造 `intent_plan.realization` 和必要时的 `intent_plan.search_space`；
3. 目标 emitter 生成真实 Triton、cuTile 或 TileLang Python 源码；
4. 下层编译器编译生成代码；
5. GPU 实际执行；
6. 与 PyTorch reference 比较数值。

当前没有发现仍可达的旧 schedule、旧 target dialect、旧 projection、旧 Python emitter 或按 kernel 名称分派的旁路。Git 中也没有跟踪构建目录、虚拟环境、日志、缓存或二进制产物。

需要准确限定“完成”的含义：本轮架构重构和现有验收集合已经完成；这不等于 Intent Kernel 整门语言的所有未来能力已经完成。当前 corpus 也没有动态覆盖所有理论组合，具体边界见本文最后一节。

## 一、最终持久化链路

当前编译链路是：

```text
Python DSL
  -> canonical Intent Kernel MLIR
  -> 临时派生分析索引
  -> GPU physical realization + optional search space
  -> Triton / cuTile / TileLang source
```

只有三类会跨阶段保存的表示：

1. **算法 IR**：canonical Intent Kernel MLIR，是算法语义的唯一真理；
2. **物理决策**：`intent_plan.realization`，保存从多个合法物理方案中选出的方案；
3. **目标源码**：Triton、cuTile 或 TileLang 源码。

`KernelFacts` 和 `SurfacePlan` 是每次从 Kernel IR 与 Plan 重算的临时索引。它们没有 dialect、序列化格式、独立 verifier 或持久 schema，因此不构成第四层表示。

`intent_plan.search_space` 与已确定 realization 分开表达。没有未决 tile 参数时不生成空 search space；存在未决参数时，realizer 只声明合法参数与 specialization key，候选值和赢家仍由下层 tuner 决定。

## 二、逐轴调度如何构造

### 2.1 角色按逻辑轴累积

realizer 以 logical domain 为键建立 `AxisChoice`，然后分别添加：

- `parallel`：该轴进入 GPU program space；
- `ordered`：该轴必须保持顺序遍历；
- `reduction`：该轴被 contraction 收缩；
- `ragged_member`：该轴是不规则关系的成员轴；
- `lane`：该轴在一个 program 内作为向量或 tile lane。

这些角色不是互斥枚举。同一个 `intent_plan.axis` 可以同时拥有多个角色。代码中不存在 `softmax`、`attention`、`gemm`、`moe` 等 kernel 名称到调度模式的分派。

同一逻辑 domain 如果被多个 `intent.parallel` 实例使用，它们现在共享同一个轴决策，同时全部参与 ordered-stream ownership 和 lane-reuse 分析；后出现的 parallel 实例不会再被静默忽略。

### 2.2 物理映射是轴字段，不是 kernel 模式

每个 parallel 轴独立保存：

- program order；
- worker axis；
- fold order；
- 是否复用 worker；
- contraction 相关 program axes 的 grouping；
- tile role。

program axes 通过最多三维 GPU program space 投影。同一 worker axis 上的多个逻辑轴按 fold order 折叠；contraction 只把实际相关的 result axes 连接为 group，不把整个 kernel 归类为某种 mapping。

多个同类逻辑轴使用独立的 tile role，例如：

```text
stream, stream_1, ...
reduction, reduction_1, ...
ragged_member, ragged_member_1, ...
query, query_1, ...
row_vector, row_vector_1, ...
```

三个 Python tuner 不再要求角色集合精确等于某几个模板。它们先选择覆盖当前角色最多的下层 profile，再为缺失的单个角色补充该后端自己的候选值；任意新增角色仍会逐角色显式拒绝，不会回退到伪默认配置。

### 2.3 contraction 不再固定使用第一个 reduction 轴

旧的目标 handler 会直接查找 `reduction_0`。这意味着 Plan 即使表达了多个独立 reduction 轴，后端仍可能把 contraction 绑定到错误的第一个轴。

最终实现改为：

1. 找到当前 contraction 的两个 deferred operand transfer；
2. 读取两者在物理 Plan 中的有效 domain；
3. 求两个 operand 共同拥有、且角色为 `reduction` 的逻辑轴；
4. 要求结果恰好为一个；
5. 使用该轴自己的 dimension、tile spelling 和 index。

因此，不同 contraction 可以绑定不同 reduction 轴。这里使用的是 Kernel IR operand 关系与 Plan axis 决策，没有引入新的持久字段，也没有按 contraction 所属 kernel 分类。

## 三、stage、stream 与 ragged 的组合边界

### 3.1 stream 与 ragged 是从 canonical IR 重建的临时索引

Plan 不再保存 `stream` 或 `ragged` 的镜像 op。emission 在共享的 `indexCanonicalStructure` 中遍历全部：

- `intent.state_stream`；
- `intent.ragged`；
- `intent.ragged_outer`；
- `intent.ragged_member`。

随后按 relation、logical axis 和 stage 建立多值映射，而不是保存一个全局 `front()` 状态。多个 stream 可以独立保存 carrier、stop、axis 和 loop；多个 ragged relation 可以独立保存 offsets、indices、outer/member ownership。

一个 canonical ragged member node 必须明确属于一个 relation。若同一个 node 同时被多个 relation 声称，emitter 会报告逻辑关系歧义，而不是任意选择第一个。

### 3.2 stage 是物理切片，不是 kernel 类别

stage 从 contraction 之间的 SSA 依赖和终端写出关系构造。每个 stage 保存其输入、输出、operation slice 和 terminal，并把结构祖先 `parallel/state_stream` 纳入同一切片。

三个 emitter 的所有 region handler 都会先选择当前 operation 所属 stage。最终又增加了完整性检查：只要一个 transfer、reduction、pointwise 或 contraction 位于 staged program 内并进入物理 Plan，它就必须至少属于一个 stage；否则 emission 立即报错，不允许通过“没有选中 stage”静默跳过。

stage 与 ordered/ragged 的索引是独立建立的，realizer 中不存在 `staged else ordered` 这样的互斥入口。

## 四、emission 当前只做什么

三个目标 emitter 的事实来源只有：

1. canonical Kernel IR；
2. `intent_plan.realization` 与可选 search space；
3. 目标后端的静态 spelling/capability 表。

目标叶子负责：

- 将 semantic tile role 拼写成 `BLOCK_SIZE_*` 或 `TILE_SIZE_*`；
- 将 pointwise/reduction/contract 概念映射为 `tl.*`、`ct.*` 或 `T.*`；
- 将同一 program-space 决策打印为各语言的 grid/kernel 语法；
- 将同一 residency 决策打印为该语言存在的 storage 概念；
- 明确委托目标自带的 autotuner、layout、寄存器分配、指令选择和流水线实现；
- 对目标无法表达的 op、dtype、rank 或 residency 就地报错。

ABI shape 仍会用于把 symbolic extent 代入 runtime 表达式。这是对算法 IR 中 shape 的机械实例化，不是 emitter 重新选择 tile、ownership 或 traversal。

三个后端中存在结构相似的 target leaf 代码，但它们分别承载不同 API、buffer 模型、launch/tuner 接线和 capability 检查；它们不是旧实现旁路，也不是另一套持久化物理决策。

## 五、冗余与仓库卫生审计

### 5.1 已确认不存在的旧路径

当前可编译源码与 CMake 中均不存在以下实现或引用：

- `SchedulePolicy`；
- `ScheduleStructure`；
- Common/GPU Projection 层；
- Triton MLIR target dialect；
- cuTile MLIR target dialect；
- TileLang MLIR target dialect；
- 三套 target Projection；
- Python typed Kernel IR；
- Python target emitter；
- 按 kernel 名称选择 realizer/emitter 的分支。

没有发现无人引用的已跟踪 C++ source/header、空的旧 IR/Projection 目录或指向已删除文件的 CMake 条目。

### 5.2 Git 与生成物

- Git 只跟踪源码、规范、例子、source baseline 和本报告；
- 没有跟踪 `.venv`、build、`__pycache__`、`.pyc`、日志、临时文件、目标文件、静态库或共享库；
- Python 执行生成的 `__pycache__` 由 `.gitignore` 排除；
- CMake 构建目录固定在 `/tmp/intentdsl-build`，没有写入项目目录；
- 最终提交后 `git status --short` 应为空。

`repport/` 只保留本报告；旧报告描述的 schedule policy、machine projection 和 target dialect 已与当前实现不符，因此没有继续保留为历史版本。

## 六、最终 repro 结果

### 6.1 数值与运行状态

| kernel | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| softmax | PASS | PASS | PASS |
| layer_norm | PASS | PASS | PASS |
| rms_norm | PASS | PASS | PASS |
| logsumexp | PASS | PASS | PASS |
| gemm | PASS | PASS | PASS |
| dual_gemm | PASS | PASS | PASS |
| attention | PASS | PASS | PASS |
| varlen_attention，causal=False | PASS | PASS | PASS |
| varlen_attention，causal=True | PASS | PASS | PASS |
| online_softmax | PASS | PASS | PASS |
| moe | PASS | PASS | PASS |
| grouped_gemm | PASS | PASS | PASS |

表中的 PASS 均表示 generated source 已实际执行并通过 reference 数值阈值，不表示每个 provider 组合都存在上游 baseline。

上游 baseline 的缺口仍如实保留：

- `logsumexp`：三个 provider 都没有直接上游 baseline；
- `layer_norm`：TileLang 缺直接 baseline；
- `rms_norm`：cuTile 缺直接 baseline；
- `varlen_attention`：Triton、cuTile 缺当前直接 baseline，TileLang 只有对应 adapter。

没有上游 baseline 的组合仍然执行 generated/reference 比较，不会跳过 generated kernel，也不会打印假 PASS。

### 6.2 最终 generated p50

单位：ms。varlen attention 的 TileLang 数据采用异常 p95 出现后立即重复执行的稳定结果。

| kernel | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| softmax | 0.3559 | 0.3580 | 0.3491 |
| layer_norm | 0.1778 | 0.1808 | 0.1737 |
| rms_norm | 0.1778 | 0.1799 | 0.1737 |
| logsumexp | 0.1635 | 0.1696 | 0.1614 |
| gemm | 2.1202 | 2.3067 | 2.1634 |
| dual_gemm | 0.6960 | 0.7195 | 0.7374 |
| attention | 4.9947 | 5.0451 | 4.8921 |
| varlen_attention，causal=False | 0.3930 | 0.3825 | 0.4803 |
| varlen_attention，causal=True | 0.2884 | 0.2923 | 0.3452 |
| online_softmax | 0.3826 | 0.3499 | 0.3785 |
| moe | 8.8398 | 10.2535 | 11.3307 |
| grouped_gemm | 1.3124 | 1.4447 | 1.2834 |

与同一设备、同一命令、审计修补前紧邻执行的结果相比，受影响与未受影响路径的 p50 变化都处于约 1% 的普通运行波动范围；没有出现调度结构改变造成的性能退化。一次 TileLang varlen causal=False 的 p95 出现孤立长尾，立即重复后恢复为 p50/p95 = 0.4803/0.4884 ms。

## 七、完成边界与仍未被证明的部分

### 7.1 本轮可以确认完成

1. 调度决策已经从 whole-kernel 模式树变为逐 logical axis 的角色与物理映射；
2. 同一个轴可组合多个角色；同一 domain 的多个 parallel 实例不会丢失；
3. 多个 ordered/reduction/ragged/lane 轴拥有可区分的索引与 tile role；
4. contraction 不再全局固定取第一个 reduction 轴；
5. staged operation 不会被 emitter 静默遗漏；
6. 持久表示已经压到 Kernel IR、一个 GPU Plan、目标源码三层；
7. realization 和 unresolved search space 是两种对象；
8. 三个 target 不存在各自的持久物理决策或 kernel matcher；
9. 旧路径和仓库生成物已经清理；
10. 33 条现有 repro 全部通过，当前性能没有结构性退化。

### 7.2 这是明确边界，不是遗留的 kernel 模式

- 一个 Kernel IR kernel function 必须有一个 outer program root，因为它对应一次 GPU launch；多个独立 launch 应由多个 kernel function 表达，而不是把两个 launch 强塞进一个 program mapping。
- 一个 rank-two contraction 当前要求解析出一个共同 reduction axis；不同 contraction 可以选择不同轴。多轴 contraction 属于新的内层原语能力，不应伪装成现有 rank-two primitive。
- search space 在没有未决参数时可以不存在；这不是缺一层表示。
- 一个 canonical ragged member node 必须属于一个 relation；多个 relation 使用各自的 member node。歧义会显式报错。

### 7.3 当前 11-kernel corpus 尚未动态证明

虽然代码中已经没有对应的互斥 kernel 分支，但现有 11 个 kernel 没有同时制造以下全部组合：

- 两个彼此独立的 ragged relation 同时参与一个 kernel；
- staged contraction 与 ordered state stream 位于同一个数据流切片；
- 两个 contraction 分别使用不同的 reduction domain。

因此，本报告可以确认这些机制不再被 whole-kernel classifier 排斥，也确认多实例索引不再固定取第一个；但不能把“代码结构允许组合”写成“每一种组合都已经有动态 repro”。用户要求本轮不新增 kernel，所以这属于验收 corpus 的证明边界，而不是保留一条旧实现。

更大的编译器能力，例如 backward、作者主导的多-kernel orchestration、autograd、更多 dtype 和完整部署 epilogue，不属于本轮调度/表示重构的完成声明。

## 八、手动复核入口

唯一使用的验收入口为：

```bash
./examples/run/repro.sh <backend> <kernel>
```

其中：

```text
backend = triton | cutile | tilelang
kernel  = softmax | layer_norm | rms_norm | logsumexp |
          gemm | dual_gemm | attention | varlen_attention |
          online_softmax | moe | grouped_gemm
```

该入口会重新构建 `intent-compile`、生成目标源码、导入并编译目标 kernel、在 GPU 上执行，并进行 reference 数值比较。
