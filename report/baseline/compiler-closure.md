# Intent Kernel 编译器收官报告

> 本文是 42 个公开 repro 的固定收官快照；当前编译器已经扩展到 89 个 case。请以 `report/compiler-evolution-and-current-state.md` 的统一分析和同目录 `kernel-performance.csv` 的全量数字为准，本文旧数字不滚动覆盖。

## 结论

当前编译器已经形成一条统一主链：

```text
Python DSL
  -> canonical Intent Kernel MLIR
  -> shared GPU Physical Plan
  -> Triton / cuTile / TileLang 逐 op 投影
  -> 下层 JIT 与真实 GPU 执行
```

本轮完成了最后一个共享表示限制：逻辑轴与物理范围不再是一对一关系。一个逻辑轴可以同时拥有 ownership、分层 traversal、reduction、lane 和访问 footprint；角色与范围均可组合，不需要把 kernel 归入某个模式。

当前 42 个公开 repro 在三个 provider 上构成 126 个格：

| 状态 | Triton | cuTile | TileLang | 合计 |
|---|---:|---:|---:|---:|
| 数值 PASS | 42 | 42 | 39 | 123 |
| 明确 N/S | 0 | 0 | 2 | 2 |
| 下层运行时 FAIL | 0 | 0 | 1 | 1 |

两个 N/S 是 TileLang 的 compare-and-swap 与二维联合 affine footprint；唯一 FAIL 是 TileLang LayerNorm backward 的 reduce stage 在当前下层 autotuner 中全部配置触发 CUDA launch failure。其余格都重新经过 DSL lowering、Plan、目标源码、JIT、真实 GPU 执行和 reference 数值对照。

## 一、最后一处骨架限制如何闭合

### 1. 轴与范围分离

`intent_plan.axis` 只记录逻辑轴角色与 program-space 分配，不再保存一个含糊的“主 tile”。物理范围改为独立的 `intent_plan.range`：

| purpose | 含义 | 层级 |
|---|---|---|
| `ownership` | 一个 program 对该轴拥有的写出区域 | level 0 |
| `traversal` | 有序轴的物理推进方式 | level 0 为 chunk，level 1 为 chunk 内 step |
| `access` | 某次 transfer 相对 ownership 的读取覆盖 | level 0，绑定 transfer 与 source axis |
| `reduction` | 收缩轴的物理范围 | level 0 |
| `lane` | 稠密片内 lane 范围 | level 0 |

同一轴可以同时带多个角色，也可以为不同 purpose 带多个范围。发射器必须按自己正在投影的角色读取精确 purpose，不存在 ownership、traversal、reduction、lane 之间的 fallback。

Plan verifier 保证：

- range 的 purpose 必须与轴角色匹配；
- access range 必须锚定在已有 parallel ownership 上，并绑定唯一 transfer/source-axis；
- 只有 traversal 可以分层；当前闭合的层级必须连续为 level 0/1，level 1 是标量 step；
- 发射器不能从 tensor shape 重新构造这些决定。

### 2. 卷积：写范围与读覆盖成为两份事实

卷积 DSL 仍只陈述输出域、滤波器域和 affine 索引。共享事实分析沿索引 SSA 恢复访问 footprint：

```text
conv1d: output ownership + access[-2, +2]
conv2d: height ownership + access[-1, +1]
        width  ownership + access[-1, +1]
```

因此 Physical Plan 已能区分“这一块写多少”和“为了它要读到哪里”。这不是 TileLang buffer 字段，也没有按卷积名字识别；任何由一个 partitioned ownership 轴与静态小域形成的 affine 访问都会得到同类事实。

三种 surface 对同一决定的投影深度不同：Triton 与 cuTile 发射精确地址/边界并把具体合并、缓存和搬运实现委托给下层；TileLang 能投影单轴 access range，因此 Conv1D 可运行，但当前不能把两个 access range 的联合 footprint 表成一个并行 fragment，Conv2D 明确 N/S。

曾把 Conv1D 的唯一 halo hull 在 Triton surface 中显式物化；数值正确，但 p50 从约 0.0082 ms 退到约 0.0102 ms。该实现把下层本可自行合并的搬运强行固定在 surface 上，因此没有保留。共享 Plan 保留完整覆盖事实，是否显式物化由能获益且能表达的下层决定。

### 3. 扫描：同一位置轴上的两级 traversal

选择性状态扫描现在为同一位置轴生成：

```text
traversal level 0: chunk
traversal level 1: scalar step
```

三个 emitter 都从这两级 range 机械生成外层 chunk 循环与内层有序递推；外层可映射 program，内层保持状态依赖。没有新增 scan 调度模式，也没有让第二个 range 覆盖第一个 range。

## 二、语言闭环

### 1. 辅助函数

`@intent.fn` 已作为单-kernel DSL 的前端抽象闭合：调用点在 frontend lowering 中直接内联函数体及其 SSA 值，最终只产生唯一 canonical kernel MLIR，不再生成一套未闭合的 helper function/call ABI。

四份注意力源码中的在线归一化状态更新现在共用同一个 helper；helper 的 constexpr、符号、类型与源码位置仍在 frontend 处理，realizer 和 emitter 只看到普通 canonical ops。递归、跨 kernel 调用和独立设备函数 ABI 不属于当前单-kernel 语言合同。

### 2. 负整数整除与取模

跨目标语义定为 Python floor division：

```text
q = floor(a / b)
r = a - q * b
```

三个 target 都先在 i64 中得到 truncating quotient/remainder，再在余数非零且符号不一致时把 quotient 减一；remainder 由修正后的 quotient 重建。因此负 dividend、负 divisor 和混合符号在三个目标上一致，不依赖各 surface 原生 `%` 或右移行为。

### 3. 带符号四比特权重

signed W4 不再依赖目标是否提供算术右移。每个 nibble 用下面的二补数恒等式解码：

```text
nibble = packed & 15
signed = nibble - ((nibble & 8) << 1)
```

它在三个目标上把 `[0, 15]` 一致映射到 `[0, 7] ∪ [-8, -1]`，并已进入真实 W4A16 groupwise matmul 的 contraction 内反量化路径。

### 4. 终止与内存序构造

当前真实 kernel 没有需要 kernel 内非结构化退出的算法。Python `break` / `continue` 在源码位置直接给出不支持诊断；未闭合的 canonical op、Facts、Plan 和 emitter 路径已经删除，不保留“看似存在但不能 lowering”的语法。

三个 surface 对 fence 的 scope、ordering 和可见性没有共同合同。`I.fence` 保留为面向作者的明确诊断入口，但不会产生 canonical op，也不会降成 no-op。这个边界比虚假的跨目标内存序更准确。

## 三、冗余与编译器边界自查

### 已清理

- canonical `intent.call`、`intent.break`、`intent.continue`、`intent.fence` 及其只服务于旧形态的 enum/effect/signature 已删除；
- helper lowering 的旧 callable signature 已删除，reduce、scan、scatter-reduce 统一走当前 SSA helper 接口；
- axis 的单一 tile 字段及 emitter 中的“主范围”fallback 已删除；
- access footprint 只有一份共享事实和一份 Plan 表示，target 不再各自推一遍；
- 没有 typed Python Kernel IR，也没有 Python emitter；Python 只负责 source frontend 与启动 C++/MLIR toolchain。

### 当前结构审计

- realization 与 emission 入口按 module、canonical op、SSA/use-def 和 Plan 遍历，没有 kernel 名字分支；
- operation registry 以 canonical op 名选择逐 op handler，缺失 handler 直接报 unsupported；
- 三个 target leaf 只包含能力检查、surface spelling、ABI/runtime 接线和逐 op 投影；没有重新决定 ownership、tile、stream stop、fill 或 layout policy；
- Triton/cuTile/TileLang 的函数名只使用 source entry symbol 作为输出符号，不参与算法分类；
- target emitter 读取算法 IR 与 Physical Plan，没有第三份 schedule/target dialect 真理。

仍保留一个 whole-program 物理策略：当 contraction 同时拥有至少三个 program axes、其中至少两个 tiled 且不含 ragged ownership 时，shared realizer 选择 persistent traversal，并把这些 program axes 折叠到同一 worker。它没有按 kernel 名字或 provider 分支，也不替逐轴 range 做决定；它选择的是会改变源码循环结构的 program traversal，不能交给只调 tile 参数的下层 tuner。当前 batched contraction 的性能依赖这项共享选择，因此没有把它删成 target 各自判断。

## 四、全量 generated 性能

数字为 p50 / p95，单位 ms。`K` 是 kernel-only，`E` 是取得最终结果的 end-to-end GPU pipeline，`R` 是保留 runtime metadata 的 GPU 路径。只有同一行、同一 case 且 scope 相同时才比较“最低”。

| Repro / case | Scope T/C/L | Triton | cuTile | TileLang | 当前最低 p50 |
|---|---|---:|---:|---:|---|
| `softmax` | K/K/K | 0.3663 / 0.3686 | 0.3686 / 0.3707 | 0.3536 / 0.3564 | TileLang |
| `layer_norm` | K/K/K | 0.1920 / 0.1941 | 0.1900 / 0.1925 | 0.1756 / 0.1778 | TileLang |
| `layer_norm_backward` | E/E/E | 0.1623 / 0.1808 | 0.0830 / 0.0836 | FAIL | cuTile |
| `embedding_backward_atomic` | K/K/K | 0.1509 / 0.1530 | 0.1470 / 0.1498 | 0.1469 / 0.1480 | TileLang |
| `atomic_compare_exchange` | K/K/— | 0.0335 / 0.0348 | 0.0343 / 0.0344 | N/S | Triton |
| `rms_norm` | K/K/E | 0.1858 / 0.1879 | 0.1899 / 0.1913 | 0.1736 / 0.1758 | scope 不同 |
| `fused_add_rms_norm` | K/K/K | 0.1879 / 0.1900 | 0.1879 / 0.1900 | 0.1794 / 0.1818 | TileLang |
| `dropout_residual_rms_norm` | E/E/E | 0.2246 / 0.2269 | 0.2391 / 0.2432 | 0.2189 / 0.2215 | TileLang |
| `logsumexp` | K/K/K | 0.1835 / 0.1874 | 0.1900 / 0.1921 | 0.1756 / 0.1796 | TileLang |
| `cross_entropy` | E/E/E | 0.5233 / 0.5270 | 0.8271 / 0.8330 | 0.5096 / 0.5136 | TileLang |
| `gemm` base<br>tail | K/K/K | 2.0879 / 2.0926<br>2.1059 / 2.1121 | 2.0760 / 2.0797<br>2.0699 / 2.0978 | 2.1223 / 2.1346<br>2.1489 / 2.1591 | cuTile / cuTile |
| `bf16_gemm` | K/K/K | 2.0308 / 2.0429 | 2.0280 / 2.0505 | 2.0751 / 2.0875 | cuTile |
| `batched_gemm` NN<br>TN<br>NT<br>TT | K/K/K | 0.1081 / 0.1101<br>0.1208 / 0.1229<br>0.1024 / 0.1040<br>0.1393 / 0.1413 | 0.0938 / 0.0958<br>0.0918 / 0.0939<br>0.0917 / 0.0925<br>0.0934 / 0.0944 | 0.1059 / 0.1085<br>0.1085 / 0.1106<br>0.1079 / 0.1106<br>0.1065 / 0.1085 | cuTile ×4 |
| `batched_row_affine` | K/K/K | 0.0996 / 0.1004 | 0.1037 / 0.1058 | 0.0998 / 0.1009 | Triton |
| `quantized_gemm` | K/K/K | 2.0853 / 2.1138 | 2.2185 / 2.2388 | 2.1784 / 2.1960 | Triton |
| `dual_gemm` | E/E/E | 0.6916 / 0.6939 | 0.6838 / 0.6868 | 0.7396 / 0.7414 | cuTile |
| `weight_only_int4` signed W4 | K/K/K | 0.0889 / 0.0920 | 0.0687 / 0.0707 | 0.2015 / 0.2025 | cuTile |
| `conv1d` | K/K/K | 0.0082 / 0.0082 | 0.0164 / 0.0185 | 0.0082 / 0.0088 | Triton / TileLang |
| `conv2d` | K/K/— | 0.0696 / 0.0717 | 0.0614 / 0.0620 | N/S | cuTile |
| `selective_scan` | K/K/K | 0.1614 / 0.1627 | 0.1157 / 0.1159 | 0.1034 / 0.1042 | TileLang |
| `attention` | K/K/K | 4.9756 / 4.9787 | 5.0053 / 5.0096 | 4.8574 / 4.8650 | TileLang |
| `attention_bias` | K/K/K | 5.0151 / 5.0402 | 5.1430 / 5.1605 | 5.0014 / 5.0230 | TileLang |
| `varlen_attention` noncausal<br>causal | R/R/R | 0.3930 / 0.3962<br>0.2967 / 0.2997 | 0.3826 / 0.3868<br>0.2883 / 0.3013 | 0.3893 / 0.3926<br>0.2682 / 0.2787 | cuTile / TileLang |
| `varlen_gqa_prefill` | R/R/R | 5.2153 / 5.2334 | 5.6791 / 5.7022 | 5.5792 / 5.6130 | Triton |
| `varlen_gqa_rope_prefill` | E/E/E | 5.7557 / 5.7706 | 6.4220 / 6.4426 | 6.0952 / 6.1161 | Triton |
| `paged_attention` | K/K/K | 0.2248 / 0.2281 | 0.3620 / 0.3661 | 1.3571 / 1.3617 | Triton |
| `online_softmax` | K/K/K | 0.3927 / 0.4069 | 0.3555 / 0.3578 | 0.3938 / 0.4060 | cuTile |
| `ordered_prefix` | K/K/K | 0.0386 / 0.0417 | 0.0392 / 0.0406 | 0.0199 / 0.0214 | TileLang |
| `moe` | E/E/E | 8.6463 / 8.6680 | 10.2314 / 10.3649 | 11.4316 / 11.4978 | Triton |
| `grouped_gemm` base<br>tail<br>empty groups | E/E/E | 1.3993 / 1.4208<br>1.4216 / 1.4363<br>0.0884 / 0.0980 | 1.4657 / 1.5025<br>1.4625 / 1.5017<br>0.0606 / 0.0706 | 1.2884 / 1.3015<br>1.2987 / 1.3039<br>0.0926 / 0.1038 | TileLang / TileLang / cuTile |
| `swiglu_forward` | K/K/K | 0.2330 / 0.2364 | 0.2417 / 0.2451 | 0.2883 / 0.2921 | Triton |
| `swiglu_backward` | K/K/K | 0.1116 / 0.1126 | 0.1159 / 0.1180 | 0.1120 / 0.1142 | Triton |
| `shifted_row_copy` | K/K/K | 0.0220 / 0.0241 | 0.0221 / 0.0242 | 0.0220 / 0.0241 | Triton / TileLang |
| `grouped_query_head_add` | K/K/K | 0.0036 / 0.0057 | 0.0036 / 0.0056 | 0.0033 / 0.0053 | TileLang |
| `scalar_table_lookup` | K/K/K | 0.0404 / 0.0441 | 0.0568 / 0.0581 | 0.0425 / 0.0446 | Triton |
| `matrix_transpose` | K/K/K | 0.0917 / 0.0942 | 0.0942 / 0.0963 | 0.6053 / 0.6079 | Triton |
| `boolean_reduction` | K/K/K | 0.0051 / 0.0054 | 0.0034 / 0.0044 | 9.2236 / 9.2522 | cuTile |
| `value_select` | K/K/K | 0.0881 / 0.0903 | 0.1142 / 0.1164 | 0.0887 / 0.0911 | Triton |
| `record_fields` | K/K/K | 0.4870 / 0.4892 | 0.4890 / 0.5043 | 0.4888 / 0.4900 | Triton |
| `scalar_while` | K/K/K | 0.1224 / 0.1244 | 0.1244 / 0.1255 | 0.1224 / 0.1231 | Triton / TileLang |
| `sorted_nucleus_cutoff` | K/K/K | 0.0258 / 0.0272 | 0.0220 / 0.0231 | 0.0261 / 0.0267 | cuTile |
| `insertion_top_k` | K/K/K | 0.5228 / 0.5250 | 0.4828 / 0.4883 | 0.7081 / 0.7110 | cuTile |

cuTile Conv2D 的 autotuner 有 3 个候选成功、9 个候选编译失败或超时；入口最终选择有效候选、真实执行并数值 PASS，所以它不是一个失败格。TileLang LayerNorm backward 则是 4 个候选全部在 benchmark/module load 阶段失败，入口没有执行完成，因此不能记为 PASS 或 N/S。

## 五、上游可比项

下表只列本轮日志中真实取得上游数字的 41 个 provider-case 单元格。数字为 generated p50 / upstream p50；括号内为比值。`D` 表示算法、布局或 adapter 边界不同，只能观察；`F` 表示 generated 改变了融合边界，优势属于端到端融合。

| Entry / case | Scope | Triton | cuTile | TileLang |
|---|---|---:|---:|---:|
| `softmax` | K | 0.3663 / 0.3661 (1.0003×) | 0.3686 / 0.3690 (0.9990×) | 0.3536 / 0.3707 (0.9540×) |
| `layer_norm` | K | 0.1920 / 0.1720 (1.1158×) | 0.1900 / 0.3461 (0.5490×) | — |
| `layer_norm_backward` | E-D | 0.1623 / 0.3103 (0.5230×) | — | — |
| `rms_norm` | T:K, L:E | 0.1858 / 0.1741 (1.0673×) | — | 0.1736 / 0.3494 (0.4967×) |
| `fused_add_rms_norm` | K | 0.1879 / 0.1770 (1.0616×) | — | — |
| `cross_entropy` | E-D | 0.5233 / 0.3410 (1.5347×) | — | — |
| `gemm` | K | 2.0879 / 2.0506 (1.0182×) | 2.0760 / 2.2498 (0.9227×) | 2.1223 / 2.3229 (0.9136×) |
| `bf16_gemm` | K | — | 2.0280 / 2.2036 (0.9203×) | 2.0751 / 2.2766 (0.9115×) |
| `batched_gemm` NN<br>TN<br>NT<br>TT | K | — | 0.0938 / 0.0922 (1.0174×)<br>0.0918 / 0.0922 (0.9955×)<br>0.0917 / 0.0922 (0.9951×)<br>0.0934 / 0.0918 (1.0174×) | — |
| `dual_gemm` | E-F | 0.6916 / 0.7752 (0.8921×) | 0.6838 / 0.8095 (0.8447×) | 0.7396 / 0.9236 (0.8008×) |
| `attention` | K | 4.9756 / 4.9526 (1.0046×) | 5.0053 / 4.7954 (1.0438×) | 4.8574 / 6.6883 (0.7263×) |
| `attention_bias` | K-D | 5.0151 / 7.0369 (0.7127×) | — | — |
| `varlen_attention` causal | R | — | — | 0.2682 / 0.2535 (1.0580×) |
| `varlen_gqa_prefill` | R | — | — | 5.5792 / 6.3852 (0.8738×) |
| `paged_attention` | K-D | 0.2248 / 0.2949 (0.7623×) | — | — |
| `online_softmax` | K | 0.3927 / 0.3656 (1.0743×) | 0.3555 / 0.3697 (0.9617×) | 0.3938 / 0.3697 (1.0652×) |
| `moe` | E-D | 8.6463 / 10.2094 (0.8469×) | 10.2314 / 9.6022 (1.0655×) | 11.4316 / 9.3524 (1.2223×) |
| `grouped_gemm` base | E | 1.3993 / 1.8777 (0.7453×) | 1.4657 / 1.8613 (0.7875×) | 1.2884 / 1.4895 (0.8650×) |
| `swiglu_forward` | K | 0.2330 / 0.2273 (1.0248×) | 0.2417 / 0.2447 (0.9875×) | — |
| `swiglu_backward` | K | 0.1116 / 0.1094 (1.0208×) | — | — |

同算法、同 kernel 数的 K 项可以判断 generated kernel 与上游 kernel；E/R 只判断对应用户 scope。没有高质量上游的格保持为空，没有用 PyTorch reference 或拼接 adapter 冒充高性能 baseline。

source 中的 causal Conv1D、Mamba chunk scan 与该快照中的 DSL kernel 算法并不相同；前者是 causal 小宽度卷积，后者是多状态、块间编排的扫描。它们继续作为结构参考，但不接成虚假的一对一性能对照。该快照中的 TileLang Conv2D generated 目标明确 N/S；这一格后来已闭合，当前结果见全量 CSV。

## 六、该快照当时的明确边界

1. **TileLang compare-and-swap：N/S。** 当前 surface 没有可核验的等价 CAS primitive；没有串行或非原子 fallback。
2. **TileLang Conv2D：当时 N/S。** shared Plan 已有两个 access ranges，但当时 TileLang 不能把它们的联合 footprint 投影；后续以通用串行 joint-footprint 路径闭合正确性。
3. **TileLang LayerNorm backward：当时下层 FAIL。** source 与四组候选均成功生成，失败发生在 TileLang autotuner benchmark/module load 的 CUDA launch；当前矩阵中该格已经通过。
4. **cuTile/TileLang 超 32 位外部地址：明确拒绝。** Triton 使用 i64 地址算术并实际访问过元素偏移 `2^31`；另外两个 surface 在当前 tensor descriptor/bulk-copy 合同下于 launch 前拒绝，不会静默回绕。
5. **TileLang 个别通用原语性能不成立。** `boolean_reduction` 为 9.2236 ms、`matrix_transpose` 为 0.6053 ms，功能正确但明显落后；这是当前下层投影/原语质量，不被写成编译器算法成功的性能结论。

在该快照中，TileLang 有 39 个入口格真实通过，并在 softmax、LayerNorm、scan、dense attention、causal varlen attention 和 grouped GEMM 等结构上形成最低值；同时它继续作为抽象线最低的 surface 暴露显式 storage/layout 能力边界。项目没有为它建立第二套 realizer。

## 七、复现口径

全量核验没有新增 test 目录、pytest、fixture 或结果数据库。唯一入口仍是：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <repro>
```

短核的 CUDA Graph 测量会在 replay 之间冲刷 L2；输入、输出与 workspace 预分配在计时区外。end-to-end 与 runtime-metadata 项按表中 scope 保留用户为取得结果必须支付的 GPU 工作。
