# 卷积、选择性扫描与权重量化矩阵乘：结构扩展报告

## 结论

本报告对应实现基线 `b3d60f6`。这一轮增加了三类结构明显不同的 DSL kernel，并全部经过同一条主链：

```text
Python DSL
  -> canonical Intent Kernel MLIR
  -> shared GPU realization / Physical Plan
  -> shared operation traversal
  -> Triton / cuTile / TileLang projection
```

结果不是“三个名字被接进 runner”，而是三组新的结构事实进入了现有表示和逐 op lowering：

- direct conv1d/conv2d 压到了多轴仿射索引、相邻 tensor index 的 broadcast、滤波器归约和边界填充；
- selective state scan 压到了 fixed-step ordered axis 和真实的逐位置状态递推；
- W4A16 groupwise matmul 压到了 packed `i32` 权重、位运算、收缩轴分组缩放和 contraction 前的逐元素反量化链。

这轮同时暴露了两个尚未闭合的物理能力，不能写成已经完成：

1. 卷积当前能正确生成 direct patch load，但 Physical Plan 还没有物化“读覆盖范围大于写覆盖范围”的唯一 halo tile；同一输入位置会作为多个输出窗口的地址重复出现。
2. selective scan 当前是正确的 fixed-step recurrence，不是 chunked selective scan；同一逻辑轴上的“外层 chunk + 内层 step”还不能由一个 axis 的单一 tile 决定同时表达。

权重量化矩阵乘的 unsigned W4 路径已经闭合；signed W4 没有被悄悄混进来。

## 1. 当前运行矩阵

所有数字来自 `b3d60f6` 代码基线上的实际 GPU repro。没有上游 adapter 的项只报告 generated 数值和时延，不制造 generated/upstream 比值。

| kernel | provider | 状态 | 最大绝对误差 | generated p50 | 计时口径 | upstream |
|---|---|---|---:|---:|---|---|
| conv1d same, `B=64,L=16384,K=5` | Triton | PASS | `0.0` | `0.0082 ms` | CUDA Graph | 无 adapter |
| conv1d same | cuTile | PASS | `0.0` | `0.0164 ms` | CUDA Graph | 无 adapter |
| conv1d same | TileLang | PASS | `6.10e-5` | `0.0082 ms` | CUDA Graph | 无 adapter |
| conv2d same, `B=16,H=W=256,R=S=3` | Triton | PASS | `6.10e-5` | `0.0696 ms` | CUDA Graph | 无 adapter |
| conv2d same | cuTile | PASS | `6.10e-5` | `0.0614 ms` | CUDA Graph | 无 adapter |
| conv2d same | TileLang | 明确不支持 | — | — | — | 无 adapter |
| selective state scan, `B=128,L=4096` | Triton | PASS | `1.12e-8` | `0.1471 ms` | CUDA Event | 无 adapter |
| selective state scan | cuTile | PASS | `1.12e-8` | `0.1573 ms` | CUDA Event | 无 adapter |
| selective state scan | TileLang | PASS | `1.12e-8` | `0.1020 ms` | CUDA Event | 无 adapter |
| unsigned W4A16, `M=512,K=2048,N=4096` | Triton | PASS | `9.77e-4` | `0.0582 ms` | CUDA Event | 无 adapter |
| unsigned W4A16 | cuTile | PASS | `9.77e-4` | `0.0664 ms` | CUDA Event | 无 adapter |
| unsigned W4A16 | TileLang | PASS | `9.77e-4` | `0.1668 ms` | CUDA Event | 无 adapter |

TileLang conv2d 在编译阶段给出确定诊断：

```text
'intent.view_load' op requires a multi-axis broadcasted indirect read
footprint that TileLang cannot project as one parallel fragment
```

这不是静默 fallback，也没有为 TileLang 写第二套卷积分析。当前能力边界是：一维窗口可以投影；rank 大于 2 的多 tensor-index broadcast footprint 不能投影为一个 TileLang parallel fragment。

cuTile conv2d 的 tuner 中 `3/12` 个候选成功编译，其余候选由当前 cuTile compiler 拒绝或超时；最终有效候选数值通过。报告不把“至少有一个候选可运行”解释成所有 tile 组合都受支持。

## 2. 卷积

### 2.1 DSL 算法

两份算法位于 `examples/kernels/convolution/direct.py`：

- conv1d：输出位置 `l` 读取 `l + r - padding`，对 5 个 filter tap 做 f32 累加；
- conv2d：输出位置 `(h,w)` 读取 `(h+r-padding_h, w+s-padding_w)`，对 `R,S` 两轴分别归约。

作者写下的是卷积本身：输出域、filter 域、索引关系、归约和 same padding。作者没有写 tile 大小、程序映射或 target 语法。

### 2.2 共享层实际增加的能力

前端原先把多个 tensor index 当作拼接关系，不能表达：

```python
x[output_rows[:, None] + filter_rows[None, :] - pad]
```

现在相邻 ranked tensor indices 先按结果 shape broadcast，再形成一个 canonical access relation；多轴版本使用同一规则。`KernelFacts` 按 broadcast 后的结果 shape 重新绑定 logical axes，边界分析看到的是每个输出 lane 与 filter lane 的真实来源，而不是 kernel 名字。

filter 域是静态小域。GPU realizer 将静态 lane extent 向二次幂物化为 fixed tile；归约 padding 继续由最终消费者的恒等元决定。对卷积乘加，越界输入用 `0`，不是作者额外提供的魔法参数。

三个 emitter 增加的是同一个 ranked indirect-index 概念的语法投影。Triton 和 cuTile 能直接打印 broadcast 后的地址张量；TileLang 对它能表达的 rank 做能力检查。

### 2.3 对 overlapping footprint 的实际回答

当前生成代码每个输出 program 形成一个 patch 张量：

```text
conv1d: [output_tile, filter_extent]
conv2d: [output_h_tile, output_w_tile, filter_h, filter_w]
```

它是一条向量化 load，而不是 Python 层的逐元素 launch；但 patch 中确实包含重复地址。相邻输出窗口共享的输入点没有先被收集成唯一的：

```text
conv1d halo: [output_tile + K - 1]
conv2d halo: [output_h_tile + R - 1, output_w_tile + S - 1]
```

因此，目前 Physical Plan 能表达输出 tile 和精确 access relation，却还不能表达独立的 read footprint tile、它相对 write tile 的偏移，以及该 footprint 的片上唯一物化。这一项属于共享 GPU 物理决定，不属于 TileLang 专属字段；为了避免把最显式后端的 buffer schema搬进共享层，本轮没有伪造一个只被 TileLang 消费的 halo op。

所以卷积的结论分两层：

- 算法表达、边界正确性和 direct lowering 已经闭合；
- 跨窗口唯一 halo 复用尚未闭合，当前性能不能代表成熟卷积实现。

## 3. 扫描与状态空间递推

### 3.1 已有能力和本轮新增能力的边界

项目原先已有 `I.scan` 的 inclusive-add lowering，sampling repro 已经真实使用它。本轮不能把这件已有能力重复记成新成果。

新 kernel 位于 `examples/kernels/streaming/selective_scan.py`，递推为：

```text
state[t] = decay[t] * state[t-1] + drive[t] * x[t]
output[t] = state[t]
```

每个 batch row 独立，位置轴严格有序。它不是可用一个固定加法 primitive 代替的 cumulative sum，而是作者明确写出的有状态 recurrence。

### 3.2 shared realization 如何落地

DSL 使用 `state_stream(..., extent=1)` 表达一次推进一个逻辑位置。此前显式正整数 extent 在分析后会被重新当作待调 tile，可能把标量 recurrence 错误扩成向量 recurrence。

现在 shared facts 保留 ordered stream 的 constant extent，Physical Plan 为该轴写出 `fixed_1`。fixed tile 不进入 target tuner；三个 emitter 只把同一个 fixed extent 打印成各自循环语法。

这同时修正了 delegated-tuning 合同：`row_vector` 和 `fixed_*` 都由运行时 shape configuration 或明确 physical extent 决定，不属于 search space。早期回归中 softmax 曾被错误要求携带 tuner；`8c90fab` 统一了 search-space 生成与 emitter 校验对 fixed row-vector 的理解。

### 3.3 尚未完成的 chunked scan

尝试表达“同一位置轴先按 chunk 分块，再在 chunk 内逐 step 推进”时，现有 Plan 暴露出一个真实边界：一个 logical axis 目前只对应一个 tile role/extent。外层 chunk 与内层 step 若都绑到同一个轴，会互相覆盖，而不是形成层级关系。

因此当前 selective scan 是正确但串行的每-row recurrence；它证明了任意 state update 能从 DSL 进入三种目标源码，却没有证明 Mamba 风格的 chunk-local parallel scan，也没有实现块间两阶段的作者编排实例。source 中的 Mamba chunk scan 仅作为结构参考，不能拿来给当前 kernel 冒充 baseline。

## 4. unsigned W4A16 groupwise matmul

### 4.1 ABI 和算法

kernel 位于 `examples/kernels/contraction/weight_only_int4.py`。逻辑参数为：

```text
activation:    f16[M, K]
packed_weight: i32[K / 8, N]
scales:        f16[K / 64, N]
output:        f16[M, N]
```

每个 `i32` 存 8 个 unsigned 4-bit 值。收缩循环内部执行：

```text
packed_row = k // 8
shift      = (k % 8) * 4
q          = (packed_weight[packed_row, n] >> shift) & 15
scale      = scales[k // 64, n]
w          = f32(q) * f32(scale)
acc       += activation[m, k] * f16(w)
```

这压到了“沿收缩轴分段广播”的真实形式：scale 既依赖 `n`，也依赖 `k // group_size`，不是已有的单纯 per-channel broadcast。

### 4.2 shared 能力

Python AST 与 canonical IR 的普通 binary op 现在覆盖：

```text
bitwise_and, bitwise_or, bitwise_xor,
left_shift, right_shift
```

前端要求两侧均为整数类型；GPU operation validator 接受这些 op；padding proof 只对能保持零的位运算链传播 zero fill。位运算后、cast 后、乘 scale 后的 tensor 继续携带收缩轴 validity，最终 contract 的 K 尾块仍以乘加恒等元 `0` 处理。

这一能力没有生成 `weight_only_int4` 专用 handler。W4 只是现有 index、binary、cast、load、state_stream 和 contract 的组合。

### 4.3 contraction residency 的结构性修正

W4 暴露出 direct-load contraction 与 transformed-operand contraction 不能一刀切：

- 普通 GEMM 两侧都是直接外存 tile，适合 shared residency 和 delayed materialization；
- W4 的 packed weight 先经过解包与反量化，结果已经是 fragment，不能假装成可 deferred 的外存 tile；
- attention 又有 `shared × shared` 的首次 contraction，以及“前一 contraction 的派生 fragment × direct V tile”的后续 contraction。

shared GPU plan 现在按 producer chain 决定 residency：

- 两侧均为 direct view load 时，direct tiles 可进入 shared；
- 一侧来自前一个 contraction 的 tensor 数据流时，另一侧 direct tile可进入 shared，支持 chained contraction 的复用；
- 纯逐元素解包/反量化链不会被误判成外存 shared tile。

是否 defer 不由三个 target 各猜一次。`SurfacePlan.h` 中一个共享机械判定同时读取 canonical transfer 与 Plan 中两侧 residency；只有 Plan 已确定为 `shared × shared`，并且存在 staged/group physical structure时才 defer。三个 emitter 调用同一个 helper。

最终实际 Plan 为：

```text
W4 contract:          private_fragment × private_fragment
attention QK:         shared × shared
attention probability·V: private_fragment × shared
```

这次修正把 TileLang dense attention 从错误路径的约 `9.69 ms` 恢复到 `4.85 ms` 左右，同时 TileLang W4 保持约 `0.167 ms`。Triton、cuTile 的 attention 与 W4 也重新完成数值运行。

### 4.4 unsigned 与 signed 的取舍

最初尝试用 `(nibble << 28) >> 28` 做 signed 4-bit 符号扩展。Triton 能得到预期结果，但 cuTile 的右移语义与该假设不一致，出现跨目标数值错误。

本轮没有在 target emitter 里靠特判补符号，而是把当前 kernel 合同明确为 unsigned W4。signed W4 需要先确定跨目标一致的算术右移、显式 sign extension 或 zero-point 语义，再进入 canonical 算法；当前报告不声称支持 signed W4。

## 5. target-specific 内容

本轮放在各 target 中的内容只有四类：

| 内容 | 为什么属于 target |
|---|---|
| bitwise/shift 的函数名或 operator 拼写 | 同一 canonical binary op 在三种 surface API 中名称不同 |
| typed integer zero fill | cuTile/TileLang 对 padding 常量的构造语法不同 |
| tile parameter 名称 | `TILE_SIZE_Q`、`BLOCK_SIZE_Q` 等是 surface interface spelling |
| TileLang 高 rank indirect fragment 拒绝 | 当前 surface capability 不能表达该投影 |

以下事情没有进入 target emitter：卷积是什么、scan 是什么、W4 是什么、哪条轴归约、哪块属于哪个 program、哪个值最终用什么恒等元、direct tile 是否应驻留 shared。没有新增 per-kernel emitter 或 per-kernel realizer。

## 6. 上游结构参考

`74d949f` 按现有 source 层级加入三份未改写的结构参考：

- `source/cuda/causal-conv1d/convolution/causal/causal_conv1d.cpp`，来自 [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)；
- `source/tilelang/tilelang/scan/mamba_chunk_scan/example_mamba_chunk_scan.py`，来自 [TileLang Mamba chunk scan](https://github.com/tile-ai/tilelang/blob/main/examples/linear_attention/example_mamba_chunk_scan.py)；
- `source/triton/gemlite/gemm/weight_only_int4/gemm_kernels.py`，来自 [GemLite](https://github.com/dropbox/gemlite)。

它们是高性能结构参考，不是当前三份 DSL kernel 的算法匹配 baseline：

- causal-conv1d 是 causal、小 kernel-width 实现；当前 DSL 是 5-tap same-padding direct conv；
- Mamba example 是 chunked multi-state scan；当前 DSL 是 fixed-step 单 state recurrence；
- GemLite 覆盖更广的 zero-point、group mode 和 target specialization；当前 DSL 是 unsigned、无 zero-point 的 groupwise W4A16。

三份文件目前没有本地 runtime/adapter 接线，部分还依赖未一并引入的上游包内文件。因此报告没有给它们填 upstream 时延，也没有用 PyTorch 拼接实现冒充上游 kernel。

## 7. 既有路径回归

本轮最终代码重新运行了受影响的代表入口：

```text
softmax                  × Triton/cuTile/TileLang
gemm                     × Triton/cuTile/TileLang
attention                × Triton/cuTile/TileLang
selective_scan           × Triton/cuTile/TileLang
sorted_nucleus_cutoff    × Triton/cuTile/TileLang
conv1d                   × Triton/cuTile/TileLang
conv2d                   × Triton/cuTile/TileLang
weight_only_int4         × Triton/cuTile/TileLang
```

除明确不支持的 TileLang conv2d 外，以上入口数值均通过。代表性能哨兵如下：

| kernel | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| stable softmax generated p50 | `0.3666 ms` | `0.3686 ms` | `0.3538 ms` |
| GEMM M/N/K tail generated p50 | `2.1057 ms` | `2.0800 ms` | `2.1550 ms` |
| dense attention generated p50 | `5.0603 ms` | `5.0140 ms` | `4.8581 ms` |
| dense attention upstream p50 | `4.9300 ms` | `4.8129 ms` | `6.6527 ms` |

attention 还重新覆盖 `D=64/80/96/256`、`Q=127,K=131` 和 `Q=K=1,D=80`；三个 provider 全部数值通过。Triton 的大于 32 位元素偏移实际访问仍通过，cuTile/TileLang 仍在错误发生前明确拒绝其当前不能表达的宽地址。

## 8. 完成边界与当时的判断

| 项目 | 状态 | 判断依据 |
|---|---|---|
| direct conv1d/conv2d 算法与边界 | 已完成 | canonical access relation、zero fill、两级 reduction 和实际数值均通过 |
| 唯一 halo tile / overlap reuse | 未完成 | 当前 patch 地址存在重复；Plan 没有独立 read-footprint 决定 |
| fixed-step selective recurrence | 已完成 | 三 target 实际执行并与逐步 reference 一致 |
| chunk-local selective scan | 未完成 | 同一 logical axis 还不能同时承载 chunk 与 step 两级 extent |
| unsigned W4 + groupwise scale | 已完成 | 三 target 实际解包、反量化、contract 并数值通过 |
| signed W4 | 未完成且未假装支持 | 三 surface 的负数右移合同不一致 |
| 三个 upstream 性能对比 | 当前拿不到 | source 参考与 DSL 算法不完全同构，且没有完整 runtime adapter |

对 halo 的暂缓是一个设计判断：它确实应该进入共享物理层，但当前还没有证明“唯一 footprint + 片上 buffer + 相对索引”应如何在三种不同程序模型中保持同一语义；先为 TileLang 建一套字段会把 target abstraction 上移。对 chunked scan 的暂缓也是表示边界，而不是把它偷偷退化成现有 `I.scan`。

## 9. 提交边界

本轮实现按职责拆成以下提交：

```text
7472d46  lower direct overlapping convolutions
b63323f  realize fixed-step selective scans
218f56a  lower packed weight-only int4 matmul
74d949f  add upstream structural kernel references
8c90fab  align fixed row-vector tuning validation
b3d60f6  preserve structural contraction residency
```

最后两个提交来自既有 repro 的活体约束：前者修复 softmax 的错误 tuner 要求；后者修复 W4 residency 调整对 dense attention 造成的实际性能回退。二者均是共享结构规则，没有按 softmax、attention 或 W4 名称分支。
