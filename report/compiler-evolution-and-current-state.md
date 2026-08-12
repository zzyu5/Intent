# Intent Kernel 编译器多轮演进与当前状态

## 结论

当前项目已经不是 stable-softmax 专用链路，也不是把十几个已知 kernel 套进若干模式。它形成了一条单一编译主链：

```text
Python DSL
  → canonical Intent Kernel MLIR
  → shared GPU realization
  → Physical Plan MLIR
  → Triton / cuTile / TileLang target projection
  → 下层 JIT
  → 真实 GPU 执行与数值对照
```

当前固定矩阵包含 82 个 kernel、89 个 case、3 个 provider，共 267 个 provider-case：

| 状态 | Triton | cuTile | TileLang | 合计 |
|---|---:|---:|---:|---:|
| 数值通过 | 89 | 87 | 87 | 263 |
| 明确不支持 | 0 | 0 | 1 | 1 |
| 下层失败 | 0 | 2 | 1 | 3 |

唯一明确不支持是 `atomic_compare_exchange × TileLang`：目标没有可核验的 CAS 原语，因此没有非原子 fallback。三个下层失败是 `nonzero_compact × cuTile`、`unique_consecutive × cuTile` 和 `unique_consecutive × TileLang`；它们共享同一个未闭合的 full-row scan 物理实现问题。

这份结论必须同时带两个限定：第一，263 个 PASS 证明的是 DSL→Kernel MLIR→Plan→目标源码→JIT→GPU 数值链路真实闭合，不等于每一格性能都成立；第二，只有 41 个 provider-case 取得了可比较的 source 数字，不能把其余 PyTorch reference 或相邻算法当作高性能上游 baseline。

完整数字只维护在 [kernel-performance.csv](baseline/kernel-performance.csv)。旧 [compiler-closure.md](baseline/compiler-closure.md) 是 42-case 收官快照，不再代表当前全量状态。

## 一、最终留下的架构

### 1. Python 只做前端

Python frontend 负责受限 Python AST、constexpr、symbol/shape/region、临时语义类型和带源码位置的诊断，并直接构造 canonical Intent Kernel MLIR。项目没有与 MLIR 平行、持久化的 typed Python Kernel IR，也没有 Python realization 或 Python target emitter。

Python compiler pipeline 只做三件事：lower 到 MLIR、调用 `intent-compile`、物化 target artifact。进入 backend boundary 后，后端不得回到 Python object 重新解释算法。

### 2. Kernel IR 是算法唯一真理

Kernel IR 保存 ABI、logical workset、tensor-flow、structured primitive、state/control、index relation、dtype 与 effect。每个 operation/value 都有稳定 node ID；Plan 通过 ID 引用逻辑节点，不按函数名、Python 变量名或源码文本猜测。

它明确不保存 worker/grid、physical tile、storage address space、pipeline、layout 或 launch 参数。`reduce`、`scan`、`contract` 保持 structured node，直到 realization 选择物理兑现方式。

### 3. Physical Plan 只保存不能从算法唯一推出的决定

`intent_plan.realization` 保存已经选择的机器决定；`intent_plan.search_space` 保存委托给下层 tuner 的合法参数轴。二者不混用：源码结构、ownership、遍历、边界和数值语义不能在 tuner 中重新猜。

Realizer 不是按 kernel 分类的 if-else 树。它逐轴组合：

- `parallel`、`ordered`、`reduction`、`ragged_member`、`lane` 等角色；
- `ownership`、`traversal`、`reduction`、`lane`、`access` 等物理 range；
- 同一轴上的多个 purpose 与多个 traversal level；
- program-space folding、persistent traversal、logical validity、padding、storage、staging 与合法 search roles。

因此“attention”“MoE”“卷积”“扫描”不是 realization 入口类别，而是这些角色、range、relation 和 structured op 的组合结果。

### 4. 三个 target 是投影层，不是三套编译器

Triton、cuTile、TileLang 共享 Kernel IR traversal、Facts、Physical Plan、operation registry 和机器决定。各自叶子只保留：能力检查、概念到目标 API 的拼写、逐 op emission、编译/运行与目标 tuner 接线。

目前没有按 kernel 名称选择 realizer/emitter，没有 whole-kernel matcher，也没有第三份 target dialect 或 schedule IR。目标 leaf 确实不是“纯字段打印”——workspace、ragged、目标原语和 runtime ABI 需要目标控制代码——但这些路径只消费 Plan，没有另选 ownership、tile 或 stream stop。

## 二、编译器是怎样从单 kernel 长到当前规模的

### 阶段一：前端与 canonical MLIR

最初建立 Python eDSL、Kernel IR dialect、node/value ID、verifier 和 Python→Intent MLIR lowering。随后删除 Python 侧 typed Kernel IR，让 frontend 在构造 MLIR 时完成语义检查。

第一个重要分界不是“能生成一个例子”，而是 stable softmax 的完整路径：算法 MLIR、C++ realization、Physical Plan MLIR、Triton source、真实 launch 与 baseline 对照。

### 阶段二：realization 与 emission 分离

早期物理决定曾部分存在于 Python 或目标发射路径。重构后，C++/MLIR realizer 统一构造 `intent_plan`，emitter 只消费 `Kernel IR + Plan`。Python 发射与旧旁路被删除，编译/repro 入口统一。

### 阶段三：从 softmax 扩到四类结构与三个 surface

先后走通 tiled contraction、ordered state stream、ragged ownership/scatter reduction 和 staged ragged contraction，再接入 cuTile 与 TileLang。这个阶段把“一个 kernel 一条路径”收敛成共享 KernelFacts、per-op handler 与 target projection。

### 阶段四：从互斥模式改成逐轴角色

旧 realization 曾把 rowwise、tiled、ragged、stream 当成互斥 mapping。逐轴重构后，每个逻辑轴独立获得角色、program-space 位置、遍历和 tile；不规则关系、有序流、收缩和 parallel ownership 可以共存。

随后又分离 Axis 与 Range，解决“一个逻辑轴只能有一个物理范围”：

- 卷积输出轴同时有写 ownership 与更大的读取 access footprint；
- selective scan 的位置轴同时有外层 chunk traversal 与内层 scalar step；
- emitter 必须按 `(purpose, level)` 精确消费，不再从另一个 range fallback。

### 阶段五：语言基本盘与真实算法压力

这些轮次补齐了对真实模型有直接意义的能力：

- helper 在 frontend SSA 内联，四个 attention 共用在线归一化递推；
- 多输出、InOut/alias、混合 dtype、不同秩广播、atomic 返回值；
- quasi-affine 地址表达、动态 tensor index 与调用前置条件；
- Python floor division/remainder 的跨目标语义，以及可证明非负时的普通快速路径；
- signed W4 的可移植解码；
- 64 位地址不变量与不能表达时的明确 target 拒绝；
- contraction/reduction tail padding、consumer-neutralized bounds、ragged 与 stream 组合；
- owner-private workspace 的唯一共享线性地址投影；
- 纯逐元素 scalar parallel domain 的 lane packing，同时明确禁止把带 contraction 的标量主体自动升级成块算法。

### 阶段六：陌生算法与等价程序

为了避免“作者只写编译器会做的程序”，项目增加了四组压力测试：

| 检验组 | Triton | cuTile | TileLang | 合计 |
|---|---:|---:|---:|---:|
| 第一批 10 个陌生算法 | 10/10 | 10/10 | 10/10 | 30/30 |
| 10 个已有算法的等价写法 | 10/10 | 10/10 | 10/10 | 30/30 |
| 第二批 10 个陌生算法 | 10/10 | 8/10 | 9/10 | 27/30 |
| 10 个算法的等价分解 | 10/10 | 10/10 | 10/10 | 30/30 |
| 合计 | 40/40 | 38/40 | 39/40 | 117/120 |

第一批陌生算法覆盖 histogram、CSR SpMV、FFT、排序、聚类、Viterbi、Smith–Waterman、NMS、ROI Align 和带 barrier 选择的 simulation；第二批覆盖数据相关 compaction、多调用 MoE alignment、嵌套 ragged、优化器 InOut、KV cache 原地间接写、归一化反向和矩阵分解。

等价写法检验 loop interchange、reduction/stream、helper/inline、mask/select、索引等价式与 reduction order。等价分解进一步改变作者程序结构：乘积域与嵌套域、一个 kernel 与两个作者 kernel、恒等 index map 与连续 ragged、左右看 Cholesky、完整 causal mask 与逻辑读取终点。

这些变体没有被编译器悄悄重写成原算法。运行时恒等 index map 比连续 ragged 慢数倍被保留，因为它的算法合同仍要求读取运行时映射；单 kernel 与两 kernel 的 E-scope 也保留作者调用边界。

## 三、最近这轮实际修掉了什么

### 1. TileLang 双 access-range 不再直接拒绝

Conv2D 的 input row/column 各有一个 access range。旧 TileLang leaf 看到同一 transfer 有两个 range 就直接 N/S。本轮删除这个硬拒绝：

- 多 range 禁用不合法的 whole-tile/bulk fast path；
- 按 Plan 的联合 footprint 生成嵌套 `T.Serial` 逐元素搬运；
- 只有同时存在 `program_m/program_n` 时加入二维小 tile profile，单轴 Conv1D 仍使用原候选。

结果：`conv2d` 与 `variant_conv2d_reduce_order` 的 TileLang 格都数值 PASS；Conv1D 保持约 0.0082 ms，没有被全局缩小候选拖慢。

这里的结论是“功能投影闭合”，不是“高性能 halo materialization 完成”。两个二维格约 4.97 ms，比 Triton/cuTile 慢约两个数量级；TileLang 的串行 joint footprint 是当前最明显的目标投影质量缺口。

### 2. 非负 loop-carried 标量事实不再丢失

FFT 的 `span` 从 2 开始，每轮乘 2；`half`、`group`、`lane` 的除法与取模在数学上全部非负。旧 realizer 只能证明 affine domain expression，不能穿过 canonical `intent.for` carried scalar，因此三个内层运算都发射了完整 signed floor-correction 语句。

本轮加入保守的 scalar lower-bound 证明：从循环初值和 yield 递推证明不变量，穿过不会破坏下界的常量、cast、add/multiply、bitwise、shift 与正除数的除/模；未知 op、while 或无法证明的输入仍返回 unknown。由 logical index cast 得出的范围必须同时证明不会溢出目标整数位宽。

这让 FFT 的内层普通除/模恢复为单条目标表达式，同时 `shifted_row_copy` 的负除数、GQA 的负被除数和 cross entropy 的潜在负 label 仍走完整 Python floor 合同并保持数值通过。

### 3. TileLang owner-private workspace 的线程所有权被兑现

`private_workspace` 的 Plan 语义是一份 owner-private addressable state。旧 TileLang kernel 仍以 128 threads launch，导致每个 program 内 128 个线程并发读写同一份 FFT rolling state；JIT 能完成但数值严重错误。

本轮让包含 `private_vector` 或 `private_workspace` 的 TileLang program 使用单线程投影。它不改变 Plan 的 workspace 位置，只兑现既有所有权语义。`radix2_fft × TileLang` 因此从 downstream failure 变为真实 PASS，p50 约 0.4698 ms；同时 bitonic、Viterbi、Smith–Waterman、NMS、top-k 等 logical buffer 使用者保持数值通过。

### 4. scan 的语义与轴归 Physical Plan 所有

旧 `intent_plan.scan` 只有 `node/result_space`，三个 target 在建立 ScanBinding 时分别回到 Kernel op 重读 combine、inclusive 和 tensor axis。这是三份派生真理。

现在 Plan 明确保存：

- canonical semantics：`scan_inclusive_add`；
- scan 的 logical `axis_node`；
- result tensor 的 `tensor_axis`；
- result residency。

Plan verifier 检查这些字段，三个 emitter 只读 Plan 并选择自己的 `tl.cumsum`、`ct.cumsum` 或 `T.cumsum` 拼写。这个改动消掉了假发射，但没有把仍未设计完整的 chunk/carry 偷偷写进 target 叶子。

## 四、当前全量覆盖意味着什么

89 个 case 的 scope 分布为：Triton/cuTile 各 K=69、E=17、R=3；TileLang 记录为 K=66、E=17、R=3，另有 `layer_norm_backward` 与 `grouped_query_head_add` 两个 PASS 格的 scope 尚未回填，`unique_consecutive` 失败格也为空。`K` 是只计已准备输入/输出/workspace 后的 launch；`E` 覆盖作者流水线取得结果必须执行的全部 GPU 工作；`R` 保留运行时 metadata 路径。不同 scope 不横向判断最低。

在三家都 PASS 且 scope 相同的 83 行中，独胜次数为：

| Provider | 独胜行数 | 并列参与 |
|---|---:|---:|
| Triton | 19 | 6 |
| cuTile | 27 | 4 |
| TileLang | 31 | 6 |

赢家随 kernel 改变，说明三条 surface 不是同一份代码换名字。但 provider 差异也很大：

- TileLang `boolean_reduction` 比 cuTile 慢约 2563×；
- Triton `smith_waterman` 比 cuTile 慢约 114×；
- TileLang Conv2D 与等价变体比最快 provider 慢约 81×/117×；
- TileLang `matrix_transpose` 慢约 6.5×，`paged_attention` 慢约 5.9×；
- cuTile `viterbi_decode` 比 TileLang 慢约 5.1×。

因此“能运行”与“目标投影质量成立”必须分开。当前架构通用性已经比性能均衡性更成熟。

## 五、与上游性能对照的诚实边界

267 格中只有 41 格具有 source p50，涉及 20 个 kernel-case。数字分三类解释：

1. 同算法、同 kernel 数、同 K-scope，才判断 generated kernel 是否接近上游；
2. 融合或多 kernel pipeline 只在 E-scope 判断用户总代价，不能冒充单 kernel 优势；
3. 算法、layout、adapter 或 runtime metadata 不同的格只观察，不下结论。

按 generated/source 比值 1.2 为慢阈值，当前明显慢的 source 格是：

| 格 | generated/source | 性质 |
|---|---:|---|
| Triton cross entropy | 1.53× | E-scope；算法编排/部分和与上游仍需逐项对齐 |
| TileLang causal varlen attention | 1.35× | R-scope；流式调度与目标投影差距 |
| TileLang varlen GQA prefill | 1.50× | R-scope；GQA+ragged+stream 组合质量 |
| TileLang MoE | 1.22× | E-scope；staged ragged contraction 质量 |

明显快的格包括 cuTile LayerNorm、TileLang RMSNorm、TileLang dense attention、三 surface dual GEMM、Triton attention+bias/paged attention、Triton/cuTile grouped GEMM；Triton dense attention 本身与 source 基本持平。这里的 “attention+bias” 与 paged attention 还带算法或 adapter 差异，只能观察，不能当单 kernel codegen 结论。其余优势有些来自按当前形状/设备重新调优，有些来自融合边界；只有严格同算法 K-scope 才能解释为单 kernel codegen 优势。

当前 source 目录的陌生算法参考多数只是高质量结构来源，没有同算法、同 kernel 数、同计时范围的 adapter；相关 CSV source 格保持空白，没有用 PyTorch 时间填充。

## 六、没有闭环的能力

### 1. full-row scan 的 chunk/carry/consumer realization

当前四个 `I.scan` 使用者中，MoE prefix 与 sorted nucleus cutoff 在三个 provider 可运行；长度 4096 的 nonzero/unique 暴露出当前 Plan 仍把 scan 结果当完整 private fragment，后续 ordered consumer 再逐位置提取。

cuTile 对 nonzero/unique 的候选均在 tile 编译器限时内失败；把 stream tile 降到 32、把超时加到 10 秒仍失败，说明不是多堆几个 tile 常数能解决。TileLang unique 也有不可接受的首次 JIT。

正确的共享结构至少需要：

- scan axis 的 finite chunk traversal；
- chunk-local inclusive scan；
- 跨 chunk scalar carry；
- streaming consumer 与必须 materialize full result 的 consumer 区分；
- 需要随机访问时明确的 owner-local result storage。

nonzero、unique 和 MoE 的 prefix 只在当前位置被消费，理论上可以流式融合；nucleus 还要写完整 cumulative output，必须物化。这个判断属于 use-def/physical realization，不属于 cuTile/TileLang 各自的算子分支。本轮没有贸然把 consumer fusion 塞进 leaf，因此三格继续记 `downstream_fail`。

### 2. 动态输出仍是静态上界加 runtime count

nonzero、unique、MoE alignment 会产生真实 runtime count，但 output 仍由调用者预分配上界；count 可以进入后续显式 kernel 的地址和有效区间，不能自动改变下一次 launch 的 program 数，也不能触发恰好大小的动态分配。

这是当前单-kernel编译器与 host orchestration 的边界，不应由一个 target emitter 悄悄跨越。

### 3. workspace placement 尚未选优

owner-private workspace 的线性地址只有一份共享投影，但目前统一由 wrapper 分配 global device memory。它保证可实现，不代表 shared/local/global placement 已选择最优。FFT、DP、排序等算子会直接暴露这个差距。

### 4. TileLang CAS 与若干性能能力边界

TileLang 无等价 CAS primitive，明确 unsupported。另有若干功能 PASS 但性能不成立的原语/投影：二维 joint footprint、boolean reduction、transpose、paged attention。它们没有让 TileLang 演化成第二套 realizer；是否继续保留这个 surface，应以这些叶子能否收敛为机械高质量投影判断，而不是只看 PASS 数。

## 七、冗余与错误抽象自查

当前没有发现：

- typed Python Kernel IR 或 Python emitter；
- 按 kernel 名字选择 realization/emission；
- row/tiled/ragged 等 kernel 类别作为调度入口；
- target 方言或独立 schedule IR 作为第三份真理；
- private workspace owner 地址在三个 leaf 各拼一份；
- target 从 tensor shape 重建 ownership、access range 或 stream stop。

本轮实际删除/收敛的是两处明确重复：TileLang 多 access-range 的无条件拒绝，以及三个 target 对 scan 语义/轴的重复重推导。

仍需诚实保留的厚代码有 target ABI/runtime、workspace 分配拼写、ragged API、GEMM/reduction primitive 和 tuner 接线。它们厚不等于错误；判据是是否只读 Plan 并打印目标语义。当前 shared persistent traversal 也是一个 whole-program 结构策略，因为它改变 program mapping，不能硬拆成互不使用的逐轴缓存。

## 八、当前成熟度判断

从“骨架是否成立”看，答案是基本成立：陌生算法、等价写法和等价分解没有迫使 realization/emission 按 kernel 裂变；三个 surface 共用一份 Kernel IR 和一份 Physical Plan；263 个 provider-case 能真实运行。

从“是不是比较完整的高性能算子编译器”看，答案仍是否定的。主要原因不是 kernel 数量不够，而是两个承重层仍未完全闭合：full-row scan 的 chunk/carry/consumer 结构，以及 workspace/storage placement 的物理选择。除此之外，部分 TileLang leaf 的性能差距说明“机械投影”已成立，但“高质量机械投影”还没有普遍成立。

所以当前最准确的定位是：语言、canonical IR、共享 realization 骨架和多 surface emission 已经形成一门真实可运行的单-kernel编译器；它具备广泛算法表达与结构组合能力，但仍有一个明确 structured primitive realization 缺口，以及若干 target performance coverage 缺口。

## 九、复现入口与本轮提交

所有数值格仍由唯一人工入口生成目标代码、真实 launch 并对 reference：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <kernel>
```

本轮代码提交：

| Commit | 内容 |
|---|---|
| `d6f33f4` | TileLang 按 Plan 投影多个 access ranges，闭合 Conv2D 正确性 |
| `c70eb0a` | 证明 canonical loop-carried scalar 下界，并兑现 TileLang owner-private workspace 所有权 |
| `31e992c` | scan semantics 与 logical/tensor axis 进入 Physical Plan，删除三 target 重推导 |

CSV 中既有 source baseline 数字和计时逻辑未改；只更新了本轮从非 PASS 变为真实 PASS 的三格。
