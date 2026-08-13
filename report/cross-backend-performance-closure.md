# 跨后端高离散格收敛

本轮没有跑全量，只对五个指定格逐一比较生成源码，并只运行直接受候选改动影响的 repro。两张基线表已经没有未测的 generated runtime 空洞，因此没有为补表重跑旧数字。

## 转置：修正 TileLang 原语选择

三家物理计划相同；Triton 使用 `tl.permute`，cuTile 使用 `ct.permute`，TileLang 却把二维交换展开成 `T.Parallel` 下的逐元素赋值。TileLang 0.1.13 已提供 `T.transpose(src, dst)`，因此将 TileLang 点运算拼写和处理器改为原生 tile primitive，结果驻留在 shared buffer 后直接写回。

- 类别：目标叶子的原语选错。
- 数值：PASS，最大误差 `0.0`。
- TileLang p50 / p95：`0.6053 / 0.6083 ms` → `0.0916 / 0.0943 ms`。
- 新值与 Triton `0.0926 ms`、cuTile `0.0943 ms` 已处于同一水平。

## 分页注意力：删除 TileLang 的慢速伪支持

Triton 的 QK/PV 使用 `tl.dot`，cuTile 使用 `ct.mma`；TileLang 叶子把两个 M=1 的收缩展开成三维 products fragment、逐元素乘法和 `T.reduce_sum`，这是原来约 `5.9×` 离散度的直接来源。

实际核验 TileLang 0.1.13 后确认：语言层没有可调用的 GEMV/matvec/dot 原语；直接把同一收缩交给 `T.gemm` 会在布局推断时报 `M must be divisible by 16, but got 1`。尝试把 M 补成 16 仍在同一程序的 fragment 布局推断中失败。因此删除 products+reduce 慢路径，在 target emission 时、进入 TileLang 下层编译前，以原 `intent.contract` 源码位置明确诊断不支持。

- 类别：目标语言当前没有等价的原生单行收缩能力。
- TileLang 状态：`pass` 改为 `unsupported`；原 `1.3210 ms` 不再冒充有效支持。
- 没有把 TileLang 的 M=16/layout 约束搬进共享物理计划。

## Viterbi：排除错误归因，保留真实下层边界

cuTile 比另外两家慢的直接源码差异包括外部标量读取的 1×1 tile 形态，以及顺序动态规划中大量标量控制流。先把外部标量读取改成 cuTile 的零维 `ct.load(..., shape=())` 做 A/B：数值 PASS，但 p50 `28.9815 ms` → `28.9926 ms`，没有改善，故该改动已撤回。

继续并排核对后确认，三家都忠实发射了作者写下的逐状态顺序动态规划；cuTile 在这种 loop-carried scalar DP 上没有更合适、且不改写算法的 argmax/状态原语。把循环替换成 `ct.max/ct.argmax` 会重写作者的中间表示，因此没有做。

随后又对共享驻留判据做了一次 A/B：让带动态索引的一维缓冲从 scalar-array 改用可寻址 private vector。cuTile 首次编译超过数分钟仍未完成，说明该目标的下层对 64-lane 动态私有向量同样不能形成可接受的实现；该判据也已撤回，没有把一次目标局限固化成所有后端的共享决定。

- 类别：cuTile 对该标量动态规划形态的下层能力/质量边界。
- 处理：保留正确实现和现有数字，不加入无效改动，也不静默改算法。

## 连续去重：原生 raw store A/B 未改善

cuTile 的扫描写回原来使用 `ct.scatter(..., check_bounds=True)`。将连续 offset 写回替换为官方 `get_raw_memory().store_offset(..., mask=...)` 后，数值仍 PASS，但 p50 `1.0153 ms` → `1.0063 ms`，与未改动的当轮复测 `1.0053 ms` 等价，未解释约 3 倍差距，因此该替换已撤回。

当前差距来自 cuTile 对“向量扫描后由顺序标量消费者逐项读取并做原子/条件散写”这一作者程序形态的整体质量；没有一个可单独替换、同时保持 IR 不变的更好原语。

- 类别：下层对该混合扫描/标量消费形态的质量边界。
- 处理：保留正确实现和既有数字，不把无收益拼写提交进编译器。

## 非极大值抑制：拒绝用另一算法制造好数字

三家生成源码都忠实保留作者写下的顺序 greedy NMS。其 bool 私有缓冲由当前物理计划放在全局 workspace；三家性能差异主要来自各下层对动态标量读写和深层顺序控制流的处理。

曾实现一个共享的 32-bit bit-pack 物理方案作为 A/B。它把 Triton p50 从 `27.7411 ms` 退化到 `30.8760 ms`，因此完整撤回。更重要的是，上游 CUDA 的高性能 bitmask NMS 会并行计算成块 IoU mask，再做另一阶段筛选；那不是当前作者源码的机械存储投影。仅压缩当前顺序 suppression workspace 既不解决结构瓶颈，也不能把作者程序改成上游算法。

- 类别：当前三家差异是下层对顺序控制流的质量差异；真正高性能方案属于不同算法。
- 处理：保留当前正确实现与既有数字，不提交退化的 packing，也不改写 Kernel IR。

## 能力检查与表格

TileLang 单行收缩现在在 emitter 进入下层 JIT 前直接拒绝，避免两个 autotune 候选逐个编译失败或运行慢速替代。此前已知的 batched contraction 等组合也已经在 target emission 阶段按源码位置拒绝，不会进入下层 autotune；本轮没有再造一层 target-specific realization。

`report/baseline/kernel-performance.csv` 只更新了本轮真实改变的两个格：TileLang 转置的新实测数字，以及 TileLang 分页注意力的 `unsupported` 状态。H100 表没有空洞，也没有在本机改动尚未部署到 H100 时伪造或挪用数字。

## 实际运行的 repro

```bash
examples/run/repro.sh tilelang matrix_transpose
examples/run/repro.sh tilelang paged_attention
examples/run/repro.sh cutile viterbi_decode
examples/run/repro.sh cutile unique_consecutive
examples/run/repro.sh triton greedy_nms
```

其中 Viterbi 命令运行了两轮候选 A/B，第二轮在 cuTile 首次编译超过数分钟后人工终止；连续去重和 NMS 的命令也用于候选 A/B。所有无收益候选均已撤回。没有运行全量矩阵。
