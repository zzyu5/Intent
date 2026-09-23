**H100：相同 Intent 程序的 cuTile 正确性与性能分析（2026-09-23）**

本轮在高性能库中修到了真实瓶颈：QKV 的 cuTile 完整算子从 **7.6167 ms 降至 1.6253 ms**，减少 **78.7%**，通过原容差且快于 source **1.8186 ms**。根因是 shared pass 没有跨单例轴 reshape 合并加载后的重复补零，造成额外的寄存器/shared-memory 搬运；不是作者必须改 DSL，也不是 cuTile 天生不能高效计算 QKV。另修复了调优候选遗漏访存参数组合的问题，Mamba chunk scan 从 **0.2491 ms 降至 0.1254 ms**。

整体目标仍未完成。TritonBench 已建立的 cuTile 全量基线为 **98/100 pass**，与 Intent→Triton 相同；本轮只做受影响单点，没有重新生成或再跑全 100。96 个有效 GPU 配对中，cuTile/Intent→Triton 耗时几何均值仍为 **1.2872**，有 **16 题慢于 2 倍**。高性能 Triton 库的 46 个严格配对也仍有 **25 项慢于 Intent→Triton 的 2 倍**。下面分别列已解决的原因、实际收益和未解决项。

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

生产库总表的 89 个工作量中，cuTile 为 86 pass、1 项 `softmax_backward` 数值失败、2 项受硬件或 source 支持限制；这些状态继续保留。严格性能配对只纳入两 target 均 pass 且有对应 source 的项目。

**整体性能**

下表均是耗时比，小于 1 表示 cuTile 更快；reference 和 agent Triton 使用已保存的 H100 时间。

| 比较对象 | 配对数 | cuTile 耗时几何均值比 | cuTile 不慢于对方 |
|---|---:|---:|---:|
| 相同 Intent→Triton | 96 | 1.2872 | 13/96 |
| reference | 96 | 0.8511 | 64/96 |
| agent 直接写的 Triton，同精度 | 69 | 1.1906 | 24/69 |

所以目前可以说：cuTile 生成的完整算子总体快于 reference，但总体仍慢于 Intent→Triton 和手写 agent Triton。相对 Intent→Triton，35 题慢超过 20%，其中 16 题超过 2 倍；不能把几何均值快于 reference 说成逐题达标。`tensordot_rsqrt` 的手写 Triton 使用 TF32，仍排除于同精度 agent 配对。

生产库保留已有更快且有效的测量，补入本轮改善的 QKV、Mamba、三角求解和 MQA。以下均直接按最终整体表回算；双 source 行共享同一组 Intent 成绩，各 corpus 使用表中自己的 source 时间作分母：

| 生产工作量来源 | 两个 target 均 pass | Intent→Triton / source | Intent→cuTile / source | cuTile / Triton |
|---|---:|---:|---:|---:|
| Triton 高性能库 | 46 | 0.7876 | 1.8114 | 2.3000 |
| cuTile 高性能库 | 35 | 0.7614 | 0.6769 | 0.8889 |

这是保留已有有效成绩的整体对照，不是全部用当前编译器重新测量的回归表。按同一整体表口径，Triton corpus 的 cuTile/Triton 从本轮起点 **2.4786** 降到 **2.3000**，几何均值耗时减少约 **7.2%**；cuTile corpus 为 **0.8889**。差距明显依赖程序的物理结构，不能只按 backend 名称解释。

Triton corpus 中 cuTile 仅 2/46 项不慢于历史 Intent→Triton，39/46 项仍慢于 source；cuTile corpus 则有 23/35 项不慢于 Intent→Triton，7/35 项慢于 source。高性能库的巨大差距还没有整体解决。

本轮生产库结果如下，时间单位 ms，均使用原输入、dtype、容差与完整算子计时：

| 算子 | 原表 cuTile | 当前保留 cuTile | 耗时减少 | 已确认的归因 |
|---|---:|---:|---:|---|
| QKV projection | 7.616696 | 1.625320 | 78.7% | shared 重复补零合并遗漏 |
| triangular solve | 0.123688 | 0.067136 | 45.7% | 原分块候选使并行 program 数过少 |
| Mamba chunk scan | 0.249072 | 0.125384 | 49.7% | 默认候选遗漏 access-form/load-latency 组合 |
| Mamba chunk state | 0.089832 | 0.070296 | 21.7% | 同一访存参数组合问题 |
| FP8 MQA logits | 0.299888 | 0.210448 | 29.8% | 保留更快的正确观测；早期前后生成代码相同，不能归因给新增 pass |

QKV 本轮定向对照为 Intent→Triton **1.601104 ms**、Intent→cuTile **1.625320 ms**，相差约 **1.5%**。总表仍保留更早的 Triton **1.138512 ms**，因此相对历史最好值 cuTile 仍慢 **42.8%**，不能说已超过全部历史成绩。旧当前编译器的 Triton 点测为 **2.120656 ms**；本轮两 target 均通过原容差。

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

4. **shared：单例轴 reshape 阻断了 masked-load 合并。**

   QKV 作者只有 matmul，没有额外的 `where`。原产物的权重路径是 `load → reshape → where(K 有效, 值, 0) → MMA`；load 已经负责相同的越界补零。[RealizeAccessComposition.cpp](../../lib/Dialect/GPU/Transforms/RealizeAccessComposition.cpp) 的 `composeSelectLoad` 原来只跨 broadcast/transpose。现在它也跨可逆的单例轴插入/删除，将条件和 fill 逆 reshape 回 load，并保留原坐标、validity、读写依赖与 execution prefix。一般 flatten/split 不套用这个规则。

   外部 Triton 的 `../ref/triton/lib/Dialect/Triton/Transforms/Combine.cpp` 第 69–106 行也在 IR combine 阶段将 select 与 masked load 合并。Intent 的修复仍放在 shared pass，两个 leaf 只接收已经合并的结构。

   使用相同的 QKV 获胜配置 `BM=BN=128, BK=32, CTA=1, warps=4, occupancy=2, load_policy=3`，H100 原容差通过，cuTile **7.616704 → 1.625320 ms**。实际 winner cubin 的 `LDS.128` 从 8 条变为 0，`STS.U16` 从 64 条变为 0，`PRMT` 从 32 条变为 0；HGMMA 仍为 8 条，TMA 输入维数没有改变。这支持“重复 select 阻断直接的矩阵操作数布局”这一归因，不把全部时间收益归给某一条指令。

   此前仅把 3D TMA 输入折成 2D 的实验仍为 **7.614392 ms**，这些搬运指令没有减少；该实验代码已撤回，没有保留额外 alias 路径。

5. **物理候选：遗漏了小分块及互相影响的访存参数组合。**

   [shared TuningProfiles.json](../../lib/Dialect/GPU/Transforms/TuningProfiles.json) 的 pointwise 家族加入 32/64 分块。三角求解每个 batch 内的顺序依赖保持不变；原最小 batch tile 256 只产生 16 个 programs，而 H100 有 132 个 SM。更小分块改善并行度，cuTile **0.123688 → 0.067136 ms**，没有把顺序求解改成另一个算法。

   [CuTile Legalize.cpp](../../lib/Target/CuTile/Transforms/Legalize.cpp) 的 `materializeClosedConfigs` 原来顺序扩展 load latency 和 access form，却把已选 latency 当作固定 core 属性，遗漏了 `CTA=1, warps=4, gather, latency=3, occupancy=2` 组合。现在在原来的两个 worker/CTA 端点保留 memory-option 组合，不对全部参数做无界笛卡尔积。

   Mamba state/scan 的同配置前后对照分别约 **0.0703/0.0705 ms**、**0.1257/0.1254 ms**，说明收益不是 QKV 的 select 修复带来的。新默认候选确实包含获胜配置；从默认候选中定向选择后再次通过原 registry 数值检查，分别为 **0.070424/0.125632 ms**，总表保留更快有效值。两项默认候选由 204 增至 300；本轮只验证已确认的候选，没有再全部调优。这解决候选遗漏，准备成本仍需另行收敛。

6. **cuTile：规则的复合轴访问现在可以进入 native load/store。**

   [Legalize.cpp](../../lib/Target/CuTile/Transforms/Legalize.cpp) 的既有 `NativeTileAxisPlan` 支持 `group*width+channel` 这类正仿射轴组合；只有 stride 等于内部 tile extent 乘积、对齐和边界条件成立时才使用原生访问。内部 channel tile 不完整且外层 group 跨多组时存在空洞，仍保留 gather/scatter。shared [PhysicalProgram.cpp](../../lib/Dialect/GPU/Analysis/PhysicalProgram.cpp) 提供对应的 component-range tail 与乘积正值事实，没有新增另一套 IR/plan。

   TritonBench groupnorm 的 native 路径已在原用例上单独通过容差，但耗时 **0.924544 ms**，比 gather 慢；默认调优仍选择 gather，当前最好 **0.470544 ms**，旧值 **0.482664 ms**。因此这里只确认能力缺口补齐，不宣称 groupnorm 性能问题已解决。

**仍未解决的性能与正确性问题**

| 一组问题 | 目前证据 | 下一步修复边界 |
|---|---|---|
| groupnorm | native 访问已经可生成且数值通过，但默认 winner 仍是 gather，耗时约为历史 Intent→Triton 的 2.99 倍 | 继续查实际 tile/layout、重复读取和 collective 成本，不能把“可生成 native”当作提速 |
| 多处重复读取 | shared common-value cleanup 当前排除有 Read effect 的 load；groupnorm 的 input2 在同一计算段重复加载 | 只在相同访问且没有中间可别名写入时消除重复；跨循环缓存需另外证明生命周期 |
| 卷积、linear、zeta/lgamma 等仍落后 | 96 项中有 16 项超过 2 倍；卷积/归一化/数学函数是主要成组差距 | 按同一程序的物理访存、运算展开、native 配置分析，不靠题目名称设规则，不要求 agent 换 DSL 写法掩盖 lowering 缺陷 |
| QKV | shared 修复后当前定向对照只差约 1.5%；距历史 Triton 最好值还慢 42.8% | 大幅软件搬运已消除，剩余配置/生成代码差距仍需定位；不能从没有显式 `num_stages` 推断缺 pipeline |
| 高性能库 BN、CE、pooling、causal conv、FP8 GEMM | BN 相对历史 Tr 为 12.54 倍，cross entropy 5.66 倍，max pooling 5.48 倍，causal-conv update 5.19 倍，scaled FP8 GEMM 5.06 倍；这些巨大差距仍在 | 优先成组核对相同 Intent 的物理循环、归约、访存和精度，尚不能统一归因给某个 backend |
| 联合归约尚未全面接入 cuTile | shared `fuseIndependentReductions` 已存在；cuTile native callback 也存在，但当前 leaf 只接受显式常量 identity，CE 使用经 cast 的 shape `N`。直接打开 capability 会编译失败，已撤回该开关 | 先补齐合法 identity 的 lowering 边界，再复用现有 shared fuser；不新建 online-summary 算法或把 callback 限制推给作者 |
| 生产 softmax backward | cuTile 仍超原容差；静态核对未发现 accumulator 降精度，普通 reduction 的树/FMA 可以不同 | 仍保留 numerical_failed。允许重排不等于通过 comparator，也没有证据支持改容差或强加保序语义 |

联合归约的 capability 入口为 [intent-compile.cpp](../../tools/intent-compile/intent-compile.cpp) 第 256–267 行；shared 合并位于 [FuseIndependentTraversals.cpp](../../lib/Dialect/GPU/Transforms/FuseIndependentTraversals.cpp) 第 149–247 行，cuTile identity 检查在 `formNativeTiles`。同一 flash-CE 程序的两 target 都保留四字段 summary，未使用字段是否被 native DCE 消除尚未验证；不能用 cuTile KIR 与手写 Triton source 的差别推断同 DSL 性能原因。重复加载的限制仍位于 [EliminateCommonValues.cpp](../../lib/Dialect/GPU/Transforms/EliminateCommonValues.cpp) 第 87–94 行。

**为什么一些旧 pass 看起来消失了**

旧 CSV 的 `intent_status` 是 Intent→Triton，`triton_status` 是 agent 手写 Triton，名称容易混淆。手写 Triton 原始 100 题只有 72 个 pass，这 72 个全部进入 H100 selected，没有漏掉旧 pass；另外 28 个原来就失败，本次未入选，所以留了空白。H100 对入选 72 个重新测量，得到 71 pass 和 `sub_gelu` 一项数值失败。

现在列名明确为 `intent_triton_status`、`intent_cutile_status`、`agent_triton_status`，那 28 项写 `not_selected`。没有把原生成失败伪装成 H100 编译失败，也没有覆盖原始单次生成成绩。

已提交的实现：`3da4ead5`（reference 时间复用）、`d6e8d4ff`（shared tuple reduction）、`b7e7211c`（数学函数）、`b63b772e`（native 编译结果复用）、`adc53c25`（复合轴 native 访问）、`2585221b`（小 pointwise 分块）、`ab420223`（跨单例 reshape 合并 masked load）、`1e50a0fe`（访存参数组合）。本轮构建成功，QKV 两 target、三角求解两 target、MQA、Mamba state/scan 和定向 agent 用例均通过原生产检查；没有重新计时已有 source、改算法、加 kernel 或放宽容差。todo 保留后续节点；性能整体追平仍未完成。
