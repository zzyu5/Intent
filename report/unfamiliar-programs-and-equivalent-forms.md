# 陌生程序与等价写法检验报告

## 结论

这一轮没有从现有能力反推语料，而是先选 10 个公开实现中的算法，再按原算法结构写 DSL；同时给 10 个已有算法各写一份自然的等价表达。检验结果是：

| 检验组 | Triton | cuTile | TileLang | 合计 |
|---|---:|---:|---:|---:|
| 10 个陌生算法 | 10/10 | 10/10 | 9/10 | 29/30 |
| 10 个等价写法 | 10/10 | 10/10 | 9/10 | 29/30 |
| 合计 | 20/20 | 20/20 | 18/20 | 58/60 |

两个没有通过的格子性质不同：

- `radix2_fft × TileLang`：Kernel IR、Physical Plan 和 TileLang Python 源码都能生成，卡在 TileLang 0.1.13 第一次 JIT，限时内不返回；当前记为 `downstream_fail`，没有伪装成 Intent 编译成功或明确不支持。
- `variant_conv2d_reduce_order × TileLang`：原始 Conv2D 已因两个 access footprint 无法投影成一个 TileLang parallel fragment 而明确不支持；变体在编译原始对照时即遇到同一诊断。它不是“换一种写法后 emitter 才坏”，而是该 target 对这一整个算法结构的既有能力边界。

全量固定表目前是 62 个 kernel、69 个 case、3 个 provider，共 207 格：

| 状态 | 数量 |
|---|---:|
| 数值通过 | 201 |
| 明确不支持 | 4 |
| 下层失败 | 2 |

完整数字在 `report/baseline/kernel-performance.csv`。原有 source baseline 的测量代码没有变化，因此本轮没有重测或改写那些固定数字；新增 20 行没有可拆出的同算法上游 runtime adapter，source 数字保持空白。

这轮最重要的结论不是 58/60，而是暴露出了两件相反的事实：

1. 编译器确实能够从同一套 Kernel IR、Physical Plan 和三套机械投影中长出此前没有为它设计过的动态规划、排序、稀疏、图像和模拟程序；
2. 等价 transpose 的数值虽然通过，但显式标量并行域被实现成逐元素 program，比显式 partition 慢约 25–165 倍。这说明逻辑并行域的规范化与自动物理分块还没有闭合，是当前最明确的 realizer 质量缺口。

## 一、检验是怎样构造的

### 1. 先定算法，再接编译器

新增 source 不是为 Intent DSL 改写的简化 baseline，而是按既有 `source/<language>/<upstream>/<operator>` 组织保留的公开实现：

| 算法 | 上游结构参考 |
|---|---|
| 256-bin histogram | CUDA Samples |
| CSR SpMV | CUDA Library Samples |
| radix-2 FFT | VkFFT |
| bitonic sort | CUDA Samples |
| k-means assignment | FAISS |
| Viterbi decode | Flashlight |
| Smith–Waterman | CUDASW4 |
| greedy NMS | TorchVision |
| ROI Align | TorchVision |
| barrier option Monte Carlo | CUDA Samples |

这些 source 用于确定真实算法包含哪些状态、索引和控制流，不被改写成 Intent 版本，也没有拿 PyTorch reference 的时间冒充上游 kernel 时间。

### 2. DSL 源码不按 provider 分叉

每个算法只有一份 DSL 源码。它先经 Python frontend 生成 canonical Kernel MLIR，再由共享 GPU realizer 生成 Physical Plan，最后投影到 Triton、cuTile 和 TileLang。新增算法没有在 realizer 或 emitter 中按函数名注册路径。

### 3. 等价写法不是只看“能编译”

每个 variant 的 runner 同时编译 original 与 variant，并检查：

```text
original vs reference
variant  vs reference
variant  vs original
```

整数和布尔输出使用精确比较；浮点输出同时检查有限性和误差。计时记录 variant 自己的 launch，而不是把 original 的数字抄给 variant。

## 二、十个陌生算法的真实结果

| Kernel | 主要陌生结构 | Triton p50 | cuTile p50 | TileLang p50 | 结果 |
|---|---|---:|---:|---:|---|
| histogram | `u8` 输入、数据相关原子更新 | 3.9010 ms | 3.8990 ms | 3.6606 ms | 3/3 PASS |
| CSR SpMV | tensor 读出的 runtime `start/stop`、间接 gather | 0.0570 ms | 0.0605 ms | 0.0562 ms | 3/3 PASS |
| radix-2 FFT | 位运算、嵌套循环、动态可寻址 mutable buffer | 2.0435 ms | 0.9955 ms | — | TileLang JIT 超时 |
| bitonic sort | 两层 runtime while、动态 partner、原地交换 | 1.4048 ms | 0.7029 ms | 1.7162 ms | 3/3 PASS |
| k-means assign | cluster 顺序遍历、向量归约、标量 argmin 状态 | 0.8734 ms | 0.9212 ms | 0.6294 ms | 3/3 PASS |
| Viterbi decode | rank-2 predecessor、前向 DP、反向 traceback | 8.4841 ms | 28.9904 ms | 5.6541 ms | 3/3 PASS |
| Smith–Waterman | 两行滚动状态、二维递推、最大值状态 | 48.2728 ms | 0.4166 ms | 9.4839 ms | 3/3 PASS |
| greedy NMS | mutable suppression mask、嵌套 `continue` | 77.4099 ms | 106.2377 ms | 51.2188 ms | 3/3 PASS |
| ROI Align | float→index、clamp、四点动态 gather、双线性插值 | 0.5876 ms | 0.3088 ms | 0.6178 ms | 3/3 PASS |
| barrier option | counter RNG、数据相关 `break` | 0.9090 ms | 0.2389 ms | 1.1659 ms | 3/3 PASS |

这里的延迟只说明生成代码能够真实运行；新增 source 尚未接 kernel-only adapter，因此不能据此声称超过公开实现。

## 三、陌生程序迫使编译器补了什么

### 1. 普通循环中的 `break` / `continue` 重新成为真实语言能力

上一轮因为既有语料没有需要终止类构造，曾把它们判断为“没有真实算子需要”。陌生程序直接推翻了这个前提：Monte Carlo 需要收敛/触障后退出，NMS 需要跳过已抑制候选。

实现没有重新引入 `intent.break` 或 `intent.continue`，也没有要求三个 target 各发射一种非结构化控制流。frontend 在构造 Kernel IR 前做结构化归一化：

- `break` 引入跨迭代的 `live` 状态，并把本次迭代的 `active` 置假；
- `continue` 只把本次迭代的 `active` 置假；
- exit 后的剩余语句放进 `if active`；
- while 条件与 `live` 合取；
- canonical IR 最终仍只有 `for` / `while` / `if` / `yield`。

因此作者写下的退出语义没有丢，后端也不需要从算法形状反推“这里像 break”。

### 2. runtime-bounded sequential domain

CSR SpMV 的 `range(row_offsets[row], row_offsets[row + 1])` 不能变成静态 extent，也不能被改写成全长循环加 mask。frontend 保留 start/stop SSA；Facts 把它识别为普通 ordered traversal；三个 emitter 从 IR 中的真实边界生成 target loop。

这不是新建稀疏算子路径。任何由标量 SSA 给出上下界的普通顺序循环都走同一机制。

### 3. rank-N、动态可寻址 logical buffer

FFT、bitonic、Viterbi、Smith–Waterman 和 NMS 都需要编译期无法展开成固定 SSA 名称的 mutable buffer。Viterbi 还第一次要求 rank-2 `(time, state)` predecessor。

共享层现在做的是：

1. `LogicalBufferInfo` 保留完整静态 shape；buffer access 保留 rank-N index relation；
2. Facts 区分“可 SSA 展开的固定元素”与“需要动态地址”的 buffer；
3. Physical Plan 用 `intent_plan.buffer space="private_workspace"` 保存 residency，并记录 owner axis nodes；
4. wrapper 在外部创建隐藏 workspace，大小为 `owner program volume × product(logical shape)`；
5. kernel 用 owner 的 program 坐标形成 owner slot，再按 row-major 线性化 logical indices；
6. workspace 只进入生成 artifact 的内部 launch ABI，不成为作者的 DSL 参数。

这使三种 surface 都能实现同一份 owner-private 状态，而不假设它们都有等价的 addressable register array。早期 `local_array` residency 已删除，没有一条旧路径继续绕过 workspace。

当前代价也必须说清楚：这是一份 wrapper 管理的 device workspace，不是 CUDA shared memory；Viterbi 在 cuTile 上的 28.99 ms 说明“语义可实现”不等于已经选到了好的物理 residency。

### 4. branch-local SSA 与动态 private mask

K-means、NMS、bitonic 和动态规划都会在条件分支中更新标量状态。frontend 现在只合并真正越过分支边界的值，分支内部临时量保持局部；动态条件则成为结构化 carried state 或显式 select，不再要求所有分支变量在外部都有假定义。

这修的是普通 Python 控制流 lowering，不是为某个算法添加 handler。

### 5. 静态 domain bounds 与计划中的 padding

循环/索引分析现在同时保存静态 begin、end 和 extent。边界填充值由最终消费者的语义决定并绑定到具体 Plan value；发射器只读取 planned padding 与 planned physical extent。

这让动态规划、bitonic 和非二次幂向量访问能共享同一套越界处理，也避免 emitter 从 tensor shape 再造一份范围事实。

### 6. 基本类型与比较合同

- histogram 贯通了 external `u8` ABI、load、cast 和 target dtype spelling；
- bitonic/loop state 在 TileLang 上需要真实 i64 状态，不能回落成 i32；
- unfamiliar runner 对整数和 bool 改为精确比较，且不再让 NaN/Inf 被普通误差比较悄悄放过。

## 四、十个等价写法的结果

| Variant | 改变的表达方式 | Triton p50 | cuTile p50 | TileLang p50 | 结果 |
|---|---|---:|---:|---:|---|
| GEMM loop interchange | M/N 两层 parallel 顺序互换 | 2.0856 ms | 2.0681 ms | 2.1326 ms | 3/3 PASS |
| softmax online | reduce 形式改为 state stream | 0.3871 ms | 0.3584 ms | 0.3871 ms | 3/3 PASS |
| online softmax inline | 命名中间量改为内联 SSA | 0.3912 ms | 0.3572 ms | 0.3993 ms | 3/3 PASS |
| attention inline | helper 调用改为内联递推体 | 0.0553 ms | 0.0594 ms | 0.0512 ms | 3/3 PASS |
| attention select | `I.mask` 改为值选择 | 0.0553 ms | 0.0594 ms | 0.0512 ms | 3/3 PASS |
| RoPE index | 两半分支改为加法与取模 | 0.0451 ms | 0.0553 ms | 0.0349 ms | 3/3 PASS |
| SwiGLU helper | 内联 sigmoid 改为 `@intent.fn` | 0.2324 ms | 0.2396 ms | 0.2867 ms | 3/3 PASS |
| LayerNorm second moment | centered-square 改为 `E[x²]-E[x]²` | 0.1905 ms | 0.1904 ms | 0.1761 ms | 3/3 PASS |
| Conv2D reduction order | 交换两个 filter reduction 的顺序 | 0.0943 ms | 0.0425 ms | — | TileLang 原结构 N/S |
| transpose scalar domains | partition tile 改为两个显式 scalar parallel domain | 14.3288 ms | 15.5433 ms | 15.4123 ms | 3/3 数值 PASS |

前八项表明 op handler 基本不依赖“表达式周围必须长成某个固定形状”：helper/inline、命名 SSA/内联 SSA、mask/select、索引等价式都能进入同一套 Plan 与 emitter。

### transpose 暴露的不是 emitter 崩溃，而是 realizer 质量缺口

原写法显式使用 `I.partition(..., extent=I.auto(...))`，生成 tile program；变体直接对两个完整 domain 使用 `I.parallel`。当前 realizer 把后者解释成每个输出元素一个 scalar program，没有先把“逻辑上完全独立的二维并行域”规范化为可分块的 program space。

结果是：

- Triton：约 155×；
- cuTile：约 165×；
- TileLang：约 25×。

三个 emitter 都是在忠实打印各自收到的 Plan；问题不在某个 target leaf，而在共享 realizer 仍把作者是否显式写 `partition` 当成物理决定。对“作者没有预先替编译器设计程序”的目标而言，这是这一轮最清楚的未完成项。

## 五、一次明确的假发射如何被修掉

`batched_row_affine` 在 cuTile 上暴露了一个与算法无关的错误：

- Plan 已决定列轴是 `row_vector`，并为非二次幂 `N` 建立 `PHYSICAL_N`；
- bulk load 正确使用 `PHYSICAL_N`；
- scatter、indices 和 validity 却分别从 `row_vector` 角色重新拼出没有参数绑定的 `TILE_SIZE`。

这正是假发射：同一物理事实在 leaf 中被重建了两次。修复没有改 DSL，也没有给 affine kernel 开特例，而是在 cuTile emitter 内建立唯一 `physicalAxisTile` 投影，并让 tensor shape、exact indices、scatter 和 validity 都从它读取计划中的物理范围。修复后该项数值通过，p50 为 0.1019 ms。

## 六、明确不支持与下层失败

### 1. TileLang transformed f16 load

负整数 floor 语义会把 GQA head 映射展开成若干 i64 quotient/remainder SSA。由此产生的动态边界需要对 f16 fragment 做条件填充。TileLang 0.1.13 的 CUDA codegen 会生成 `cutlass::half_t`，而其 SM100 pack helper 接受 CUDA `half`，最终在 NVCC 阶段失败。

这不是换一个 typed zero 能解决的问题，也不应该通过把 parallel load 偷偷改成 serial 来绕开。当前 TileLang leaf 在对应 `intent.view_load` 的源码位置明确诊断这一能力边界；CSV 中 `grouped_query_head_add × TileLang` 因而是 `unsupported`，不再是模糊的 downstream crash。

### 2. TileLang Conv2D joint footprint

共享 Plan 已记录 H、W 两个 access ranges。TileLang 目前只能把一个 affine access range 投影成一个 parallel fragment，不能把两轴联合 footprint 物化成一个 fragment，因此原始和 reduction-order variant 都明确不支持。Triton 与 cuTile 继续发射精确地址，把具体 load 合并交给下层。

### 3. TileLang radix-2 FFT

FFT 已完成 frontend、realization、source emission 和 Python artifact materialization；超时发生在第一次 TileLang JIT。它不是 Intent 语言缺口，也没有证据支持按 FFT 名字拒绝。当前只能诚实记为 `downstream_fail`。

### 4. 既有边界

- TileLang 没有与当前合同等价的 compare-and-swap primitive，明确不支持；
- TileLang LayerNorm backward 的候选在下层 autotune/模块加载阶段失败，仍记 downstream failure。

## 七、架构自查

### 已确认干净的部分

- realizer 与三个 emitter 没有按 kernel 名选择路径；entry 名只用于生成 symbol；
- `local_array` 旧 residency 已完全删除；
- Python frontend 仍直接构造 canonical Kernel MLIR，没有恢复 typed Python Kernel IR；
- 新增 source 按语言、上游和算子职责组织，DSL 与 runtime 没有塞进 source；
- 新 kernel 没有增加 target-specific DSL 源码。

### 仍然存在的重复与边界

1. 三个 leaf 都在机械拼写 owner-private workspace 的 owner 线性化与 rank-N row-major offset。决定本身来自 Plan，但字符串计算重复了三份；它还没有收敛成共享 projection helper。
2. cuTile 的 row-vector physical extent 已集中到一个 helper；TileLang 内仍有两处同类 `dimension → physicalExtent` 拼写，尚未统一。
3. staged、ragged、persistent row 和 workspace 的 target leaf 仍然较厚。审计没有发现它们按 kernel 名重新决定算法，但其中混有能力检查、runtime 接线与结构投影，不能说已经退化成纯静态查表。
4. owner-private workspace 当前拒绝与 persistent/staged program 组合。这是显式能力边界，不是静默 fallback；组合本身仍未兑现。

## 八、代码落点

本轮承重修改主要落在：

- frontend structured control normalization：`python/intent/frontend/lowering/ast/statements.py`；
- logical buffer/index facts：`include/Intent/Target/Common/Analysis/LogicalBuffer.h`、`lib/Target/Common/Realization/KernelFacts.cpp`；
- Physical Plan buffer/owner schema：`include/Intent/Dialect/Plan/IR/PlanOps.td`、`lib/Target/GPU/Realization/Plan/Build.cpp`；
- shared workspace size/projection：`include/Intent/Target/Common/Emission/SurfacePlan.h`；
- 三 target 的 workspace internal ABI 与 load/store projection：各自 `Emission/Source/Emitter.cpp` 与 `Emission/Handlers/Operations.cpp`；
- 统一 cuTile row extent 投影：`lib/Target/CuTile/Emission/Source/Emitter.cpp`；
- TileLang 明确能力诊断：`lib/Target/TileLang/Emission/Handlers/Operations.cpp`；
- 10 个陌生算法与 10 个 variant：`examples/kernels/` 与 `examples/repro/common/unfamiliar.py`。

相关实现从 `9488d35` 开始，到 `09706c8`；固定性能表提交为 `016e5ef`，随后随 TileLang 能力诊断更新一格状态。

## 九、当前可据此做出的判断

当前编译器已经不只是“能处理作者为它写的 42 个 kernel”：动态 bounds、结构化 loop exit、rank-N mutable state、位运算、动态 gather、原子与多种自然等价表达都由陌生程序真实压过。

但它还不能被称为对作者写法充分稳健的算子编译器，证据也很具体：

- 显式 scalar parallel domain 与 partitioned domain 没有先归一到同一逻辑并行结构，导致 transpose 物理计划相差两个数量级；
- wrapper workspace 解决了可实现性，但没有解决动态规划/排序状态应落在哪一级存储；
- TileLang 仍有一个 JIT 超时、一个下层 autotune failure，以及若干明确能力子集；
- 三个 leaf 的 workspace 地址投影仍有可收敛的机械重复。

因此这一轮证明了骨架具备明显的外推能力，也给出了下一次决策所需的、没有被通过率掩盖的具体缺口。
