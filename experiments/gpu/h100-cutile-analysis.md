**H100：相同 Intent 程序的 cuTile 正确性与性能分析（2026-09-23）**

这轮已把 TritonBench 的 cuTile 正确性追到与 Intent→Triton 相同的 **98/100**，没有剩余编译失败或准备超时。性能尚未追平：96 个有效 GPU 配对中，cuTile/Intent→Triton 的耗时几何均值为 **1.2876**，还有 **16 题慢于 2 倍**。已确认并修复的原因包括 shared 归约循环结构、cuTile 数学库缺项和重复 native 编译；没有改写 agent 的算法或增加 kernel。

完整结果入口：

- [TritonBench 100 题四列对照](../agent_tritonbench/results/h100-selected.csv)：Intent→Triton、Intent→cuTile、reference、agent 直接写的 Triton，单位 μs；三组状态分别标明。
- [程序来源、配置与测量记录](../agent_tritonbench/results/h100-selected.json)：原 172 条记录保留，`cutile.records` 增加相同 Intent 程序的 100 条记录。
- [生产库 H100 整体对照](results/h100-comparison.csv)：89 个工作量，单位 ms；`dense_gemm`、`swiglu` 有两个可比较的 source，其余保留各自 source。RMSNorm 的两个 corpus 使用不同 epsilon，因此分别保留。

本轮是已选程序的开发复测，不是重新让 Luna 生成一轮。复用原输入、dtype、容差、每题 CUDA Graph/event 边界及已有 reference 时间；reference 输出仍参与数值检查。准备并发，GPU 计时共用锁。5090D 未运行。

**正确性已经解决了什么**

| 同一批 100 份 Intent 程序 | cuTile 本轮起点 | 修复后 |
|---|---:|---:|
| pass | 80 | 98 |
| 编译失败 | 8 | 0 |
| 准备/启动超时 | 10 | 0 |
| 数值失败 | 2 | 2 |

8 个编译失败是 cuTile leaf 缺 `erf`、`log1p`、`lgamma` 的 lowering，不是 DSL 写错。现在提供 f16/bf16/f32 路径；尚未实现的 f64 明确拒绝，不把窄精度实现冒充 f64。

10 个超时不等于 kernel 执行很慢。部分程序有 200 多个 native 配置，失败配置每个可花 15 秒；原先预编译与启动调优还会重复处理。现在让启动复用同一签名的 native 编译结果及失败结果，并仅对未完成项目延长准备预算。例如 instance-norm+SELU+conv 有 238 个卷积配置和 54 个归一化配置，准备约 529 秒，但最终完整算子只有 **0.062032 ms**。这些准备时间不计入性能列。

两项数值失败是 `matrix_power_eig`、`solve_symmetric_ldl`，原 Intent→Triton 也失败。本轮不通过修改程序或放宽容差把它们改成 pass。性能统计还排除 `sum_std` 的 CPU 常量结果、`fused_svd_reconstruct` 的恒等式捷径，得到 96 个有效 GPU 配对。

**整体性能**

下表均是耗时比，小于 1 表示 cuTile 更快；reference 和 agent Triton 使用已保存的 H100 时间。

| 比较对象 | 配对数 | cuTile 耗时几何均值比 | cuTile 不慢于对方 |
|---|---:|---:|---:|
| 相同 Intent→Triton | 96 | 1.2876 | 13/96 |
| reference | 96 | 0.8514 | 64/96 |
| agent 直接写的 Triton，同精度 | 69 | 1.1910 | 24/69 |

所以目前可以说：cuTile 生成的完整算子总体快于 reference，但总体仍慢于 Intent→Triton 和手写 agent Triton。相对 Intent→Triton，35 题慢超过 20%，其中 16 题超过 2 倍；不能把几何均值快于 reference 说成逐题达标。`tensordot_rsqrt` 的手写 Triton 使用 TF32，仍排除于同精度 agent 配对。

生产库保留原来更快且有效的测量，只补入本轮改善的 cuTile BN。两个 corpus 各自使用同一份 source 时间作分母：

| 生产工作量来源 | 两个 target 均 pass | Intent→Triton / source | Intent→cuTile / source | cuTile / Triton |
|---|---:|---:|---:|---:|
| Triton 高性能库 | 46 | 0.7876 | 1.9528 | 2.4794 |
| cuTile 高性能库 | 35 | 0.7782 | 0.6769 | 0.8698 |

这是保留已有有效成绩的整体对照，不是全部用当前编译器重新测量的回归表。差距明显依赖程序的物理结构：在 cuTile corpus 上，Intent→cuTile 总体比 Intent→Triton 快约 13%；不能只按 backend 名称解释 Triton corpus 上的落后。

**已落实的 compiler 修复及证据**

1. **shared：多结果归约的 collective 放错了循环层级。**

   Welford 的 `(count, mean, variance)` 已经有作者声明的普通归约 combine。旧优化只认识单个二元归约更新，不能把 tuple combine 的 native collective 移出遍历循环，导致每轮重复通信。现在在既有 `RealizeReductionBlocking` 中匹配相同的 typed combine、identity 和 use-def 关系，循环内携带 fragment，循环后做 native reduction；有可观察中间状态或不满足条件的程序不变。没有合成新算法、临时全局 buffer 或额外 kernel。

   [实现入口](../../lib/Dialect/GPU/Transforms/RealizeReductionBlocking.cpp)为 `matchReductionCombine`、`isolatedReductionUpdate`、`hoistNestedReduction`。成熟 [FlagGems BN source](baselines/triton/flag-gems/normalization/batch_norm/batch_norm.py) 第 84–126 行同样在循环内更新局部 Welford 状态，循环后做最终归约；参考的是这条执行边界，不复制它的线程归约树。

   | 生产 BN，原容差通过 | 本轮修复前 ms | 修复后 ms | 耗时减少 |
   |---|---:|---:|---:|
   | Intent→Triton | 0.350824 | 0.120992 | 65.5% |
   | Intent→cuTile | 0.387536 | 0.285664 | 26.3% |

   这项修复对两个 leaf 都有效。历史表中还有更早的 Triton 0.022784 ms，有效旧成绩按用户要求保留；它与本轮编译器的差距仍未定位，不能拿历史值证明本次 pass 回归，也不能当作当前同轮结果。

2. **cuTile leaf：普通数学函数缺实现，以及实现精度档位不合适。**

   Triton leaf 直接调用 libdevice（[Serializer.cpp](../../lib/Target/Triton/Serialization/Serializer.cpp) 第 1585–1596 行）；外部参考 `../ref/triton/third_party/nvidia/language/cuda/libdevice.py` 第 1254、1443、1506 行分别按 fp32/fp64 映射 native 函数。cuTile 当前没有相同的完整数学库入口，因此在 [cutile_math.py](../../python/intent/runtime/cutile_math.py) 补目标数学函数，serializer 只调用已确定的 helper，没有新建 shared IR/form/plan。

   `erf` 使用 [OpenLibm 的单精度实现](https://github.com/JuliaMath/openlibm/blob/master/src/s_erff.c)，`lgamma` 使用 [Boost Lanczos 系数](https://github.com/boostorg/math/blob/boost-1.85.0/include/boost/math/special_functions/lanczos.hpp)。初始 erf helper 的双精度中间计算过重，换到成熟 f32 路径后，仍通过原容差：

   | 完整算子 | 双精度 helper ms | 当前 ms |
   |---|---:|---:|
   | add_gelu | 0.059408 | 0.032080 |
   | gelu | 0.057840 | 0.031928 |
   | sub_gelu | 0.057536 | 0.034232 |
   | erfc_sqrt | 0.059456 | 0.033272 |
   | gelu_conv2d | 0.441792 | 0.356408 |

   这些结果仍有性能差距，不能把“补齐了普通函数”说成已达到 libdevice 性能；没有用 tanh-GELU 近似替换原 erf 算法。

3. **runner：预编译结果没有直接供启动调优复用。**

   [ProgramContext](../agent_tritonbench/program.py) 在单个 benchmark 进程内复用 cuTile `_compile` 返回值。键比较包含函数、编译选项、参数约束、调用约定、架构和 context，忽略由两条调用路径不同时间生成的 symbol。失败配置仍明确失败，只是不重复编译；cuTile 自己继续负责调优和真实 launch。该修改解决评测准备效率，不冒充 kernel 性能优化，也没有另建持久缓存系统。

**仍未解决的性能与正确性问题**

| 一组问题 | 目前证据 | 下一步修复边界 |
|---|---|---|
| 分组归一化的规则访问落成 gather/scatter | 当前 groupnorm 产物有 32 个 gather、2 个 scatter、没有 tile load/store；channel 坐标是 `group*8+channel_offset`，同一 view 轴对应两个计算轴 | 扩展现有 cuTile rectangular-access 分析，证明轴组合连续、extent/stride/边界相符后复用原生 load/store；不改作者分组、不拆 kernel |
| 多处重复读取 | shared common-value cleanup 当前排除有 Read effect 的 load；groupnorm 的 input2 在同一计算段重复加载 | 只在相同访问且没有中间可别名写入时消除重复；跨循环缓存需另外证明生命周期 |
| 卷积、linear、zeta/lgamma 等仍落后 | 96 项中有 16 项超过 2 倍；卷积/归一化/数学函数是主要成组差距 | 按同一程序的物理访存、运算展开、native 配置分析，不靠题目名称设规则，不要求 agent 换 DSL 写法掩盖 lowering 缺陷 |
| QKV | 当前同轮 Triton **2.120656 ms**、cuTile **7.616696 ms**，约 **3.59 倍**；两边 grid 相同，均有 K loop 和 MMA，cuTile native load 允许 TMA | native layout、实际指令和资源使用尚未定位。不能从 cuTile 源没有 `num_stages` 就断言没有 pipeline，更不能替 native compiler 重建 pipeline |
| 生产 softmax backward | cuTile 仍超原容差；静态核对未发现 accumulator 降精度，普通 reduction 的树/FMA 可以不同 | 仍保留 numerical_failed。允许重排不等于通过 comparator，也没有证据支持改容差或强加保序语义 |

分组访问的具体限制位于 [CuTile Legalize.cpp](../../lib/Target/CuTile/Transforms/Legalize.cpp)：`NativeTileAxisPlan` 第 498–514 行只记录一个 computation axis，`analyzeNativeTileAccess` 第 1224–1249 行要求单个 range 投影。因此合法、连续的复合轴访问也可能落成 gather。这是 Intent leaf 的表达/分析缺口；尚未实现修复，不能提前宣称获得收益。重复加载的限制位于 [EliminateCommonValues.cpp](../../lib/Dialect/GPU/Transforms/EliminateCommonValues.cpp) 第 87–94 行。

**为什么一些旧 pass 看起来消失了**

旧 CSV 的 `intent_status` 是 Intent→Triton，`triton_status` 是 agent 手写 Triton，名称容易混淆。手写 Triton 原始 100 题只有 72 个 pass，这 72 个全部进入 H100 selected，没有漏掉旧 pass；另外 28 个原来就失败，本次未入选，所以留了空白。H100 对入选 72 个重新测量，得到 71 pass 和 `sub_gelu` 一项数值失败。

现在列名明确为 `intent_triton_status`、`intent_cutile_status`、`agent_triton_status`，那 28 项写 `not_selected`。没有把原生成失败伪装成 H100 编译失败，也没有覆盖原始单次生成成绩。

本轮提交：`3da4ead5`（reference 时间复用），`d6e8d4ff`（shared tuple reduction），`b7e7211c`（数学函数），`b63b772e`（native 编译结果复用）。新的 todo 保留后续性能节点；正确性追平是已完成节点，性能追平仍未完成。
